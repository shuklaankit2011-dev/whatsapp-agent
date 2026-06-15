const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const express = require('express');
const axios = require('axios');

const AGENT_WEBHOOK_URL   = process.env.AGENT_WEBHOOK_URL   || 'http://localhost:8000/webhook/bridge';
const AGENT_WEBHOOK_TOKEN = process.env.AGENT_WEBHOOK_TOKEN || 'change_me';
const BRIDGE_PORT         = parseInt(process.env.BRIDGE_PORT || '3000', 10);
const BRIDGE_AUTH_TOKEN   = process.env.BRIDGE_AUTH_TOKEN   || 'change_me';

const client = new Client({
  authStrategy: new LocalAuth(),
  puppeteer: {
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--disable-gpu',
    ],
  },
});

// Cache: real phone number (digits only) → chat object
const chatCache = new Map();

client.on('qr', (qr) => {
  console.log('Scan this QR with WhatsApp > Linked devices:');
  qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
  console.log('WhatsApp bridge ready');
  chatCache.clear();
});

client.on('disconnected', (reason) => {
  console.log('Client disconnected:', reason);
  chatCache.clear();
  setTimeout(() => client.initialize(), 5000);
});

client.on('message', async (msg) => {
  if (msg.fromMe) return;
  if (msg.from === 'status@broadcast') return;

  let phone, name;
  try {
    const contact = await msg.getContact();
    // Always use the real numeric phone number, not the @lid id
    phone = contact.number;
    name  = contact.pushname || contact.name || null;
    // Cache chat by real phone number
    const chat = await msg.getChat();
    chatCache.set(phone, chat);
  } catch (err) {
    // Fallback: strip suffix from msg.from
    phone = msg.from.split('@')[0];
    name  = null;
  }

  const isGroup = msg.from.endsWith('@g.us');
  let groupName = null;
  if (isGroup) {
    try { groupName = chatCache.get(phone)?.name || null; } catch {}
  }

  try {
    await axios.post(AGENT_WEBHOOK_URL, {
      phone,
      name,
      text:       msg.body,
      message_id: msg.id._serialized,
      is_group:   isGroup,
      group_name: groupName,
    }, {
      headers: { Authorization: `Bearer ${AGENT_WEBHOOK_TOKEN}` },
      timeout: 30000,
    });
  } catch (err) {
    console.error('Failed to forward to agent:', err.message);
  }
});

const app = express();
app.use(express.json());

app.post('/send', async (req, res) => {
  if (req.headers.authorization !== `Bearer ${BRIDGE_AUTH_TOKEN}`) {
    return res.status(401).json({ ok: false, error: 'unauthorized' });
  }
  const { phone, text } = req.body;
  if (!phone || !text) {
    return res.status(400).json({ ok: false, error: 'phone and text required' });
  }

  // Normalize: strip @lid/@c.us suffixes, keep digits only
  const digits = phone.split('@')[0].replace(/[^\d]/g, '');
  const chatId = `${digits}@c.us`;

  // Try cached chat first, fall back to fresh lookup
  const tryWithChat = async (chat) => {
    await chat.sendMessage(text);
  };

  try {
    const cached = chatCache.get(digits);
    if (cached) {
      try {
        await tryWithChat(cached);
        return res.json({ ok: true });
      } catch (err) {
        console.warn('Cached chat send failed, retrying fresh:', err.message);
        chatCache.delete(digits);
      }
    }
    // Fresh lookup
    const chat = await client.getChatById(chatId);
    chatCache.set(digits, chat);
    await tryWithChat(chat);
    res.json({ ok: true });
  } catch (err) {
    console.error('send failed:', err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.listen(BRIDGE_PORT, () => console.log(`Bridge HTTP listening on :${BRIDGE_PORT}`));

client.initialize();
