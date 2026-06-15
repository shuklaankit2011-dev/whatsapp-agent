# WhatsApp Reply Agent

An AI agent that reads your WhatsApp messages, understands them, and replies in your voice — with a layered memory system that gets smarter about each contact over time.

Built for **Ankit**, but generalizable. Stack: **Python (FastAPI) + Gemini + ChromaDB + SQLite**.

---

## What it does

```
Incoming WhatsApp message
        ↓
[Contact resolver] → loads profile
        ↓
[Memory retrieval] ← pulls relevant past episodes + facts + recent buffer
        ↓
[Understanding LLM pass] → intent, emotion, urgency, action_required
        ↓
[Decision gate] → reply_now / draft / silent / escalate_to_you
        ↓
[Reply generator LLM pass] ← persona + memories + context
        ↓
[Memory writer] → updates all 5 memory layers
        ↓
Send via WhatsApp (or queue as draft)
```

---

## The five memory layers

| Layer | Storage | What it holds | Lifespan |
|---|---|---|---|
| **1. Working** | RAM | Current turn state | 1 request |
| **2. Short-term** | In-memory deque | Last 15 messages per contact | Until consolidation |
| **3. Profile** | SQLite | Name, relationship, comm style, full turn log, pending actions | Forever |
| **4. Episodic** | SQLite | Summaries of past conversation episodes | Forever |
| **5. Semantic** | ChromaDB | Atomic facts ("X has a daughter named Y") | Forever |

When the short-term buffer fills, the older half is **automatically summarized into an episode** and embedded into semantic memory — so the agent can recall "we discussed X last month" without keeping every word in context.

---

## Project layout

```
whatsapp_agent/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                # env settings
├── schemas.py               # Pydantic models
├── persona.md               # your voice/style guide
├── main.py                  # FastAPI app
├── bridge.js                # optional Node.js WhatsApp bridge
├── memory/
│   ├── manager.py           # coordinates all layers
│   ├── short_term.py        # rolling buffer
│   ├── profile.py           # contact profiles + turn log + draft queue
│   ├── episodic.py          # episode summaries
│   └── semantic.py          # ChromaDB facts
├── agent/
│   ├── understanding.py     # message analysis LLM pass
│   ├── reply_generator.py   # reply LLM pass + episode summarizer
│   └── orchestrator.py      # the main loop
└── whatsapp/
    └── client.py            # Twilio + bridge senders
```

---

## Setup

### 1. Python side

```bash
cd whatsapp_agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GEMINI_API_KEY, USER_NAME, USER_PHONE, WEBHOOK_SECRET
```

Run:
```bash
python main.py
# server: http://localhost:8000
```

### 2. Connect to WhatsApp — pick one

#### Option A: whatsapp-web.js bridge (free, your personal WhatsApp)
```bash
mkdir -p ../bridge && cd ../bridge
npm init -y
npm install whatsapp-web.js qrcode-terminal express axios
cp ../whatsapp_agent/bridge.js .
AGENT_WEBHOOK_TOKEN=<same as WEBHOOK_SECRET> \
BRIDGE_AUTH_TOKEN=<same as in .env> \
node bridge.js
# scan the QR code with WhatsApp > Linked devices
```

In your Python `.env`:
```
WHATSAPP_PROVIDER=bridge
BRIDGE_SEND_URL=http://localhost:3000/send
BRIDGE_AUTH_TOKEN=<same value>
WEBHOOK_SECRET=<same value>
```

> ⚠️ This uses an unofficial library. WhatsApp's ToS does not formally allow automation of personal accounts. Use for personal/experimental work; switch to Twilio for production.

#### Option B: Twilio WhatsApp Business API (official, paid)

Set up a Twilio number, configure their WhatsApp sandbox or production sender. In `.env`:
```
WHATSAPP_PROVIDER=twilio
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```
Configure Twilio's webhook to POST to `https://your-host/webhook/twilio`.

---

## How you stay in control

The agent **never auto-sends by default**. Replies go into a draft queue you approve.

```bash
# list drafts
curl localhost:8000/drafts

# approve and send draft #5
curl -X POST localhost:8000/drafts/5/approve

# edit then send
curl -X POST localhost:8000/drafts/5/edit -H 'content-type: application/json' \
     -d '{"text":"my edited version"}'

# reject (don't send)
curl -X POST localhost:8000/drafts/5/reject
```

When you trust a contact:
```bash
curl -X PATCH localhost:8000/contacts/+919999999999 \
     -H 'content-type: application/json' \
     -d '{"auto_reply_enabled": true, "relationship": "friend", "communication_style": "hinglish casual"}'
```

Even then, the agent will **still draft (not send)** if:
- confidence < threshold (default 0.85)
- urgency ≥ 4
- the analysis layer flagged `escalate_to_user` (money, contracts, deadlines)
- the contact is in `ALWAYS_DRAFT_CONTACTS`

Set `AUTO_SEND=true` globally to flip the master switch.

---

## Inspecting memory

```bash
# everything the agent knows about a contact
curl localhost:8000/contacts/+919999999999
```

Returns: profile, recent turns, pending actions, recent episode summaries, semantic fact count.

---

## Customizing the agent's voice

Edit `persona.md`. This file is loaded at startup and injected into the reply-generator's system prompt. Sections that matter most:
- **Voice** (tone, length, language mix)
- **Style rules** (what you do)
- **What I never say** (anti-patterns)
- **Defaults by relationship**
- **Hard rules** (safety constraints)

---

## Production checklist

- [ ] Run behind HTTPS (Caddy/Nginx)
- [ ] Validate Twilio webhook signatures (add a middleware in `main.py`)
- [ ] Move SQLite → Postgres if multi-process
- [ ] Add Redis for short-term memory if scaling horizontally
- [ ] Add rate limiting (don't auto-reply to the same person 50 times in 5 minutes)
- [ ] Add a "do not auto-reply between 11pm-7am" quiet hours check
- [ ] Push notifications to your phone when `escalate_to_user` fires
- [ ] Build a tiny web UI for the draft queue (vs curl)

---

## Tuning ideas

- **Per-relationship personas.** Split `persona.md` into multiple files keyed by relationship; load the right one based on profile.
- **Two-stage retrieval.** Currently semantic memory pulls top-5 by similarity. Add a reranker (BM25 + cross-encoder) for sharper recall on long conversations — same pattern as your RAG work.
- **Fact deduplication.** Right now `add_facts` doesn't dedupe. Add a pre-check: query semantic memory for near-duplicates before insert.
- **Tool calls in the reply layer.** If the message asks "what's my calendar tomorrow?" let the reply generator call Google Calendar via a tool rather than escalating.
- **Confidence calibration.** Sample 100 of your approved drafts vs rejected ones; train a simple classifier on the analysis features to set per-contact auto-send thresholds.
