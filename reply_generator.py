"""ReplyGenerator: writes a reply in the user's voice given full context.

Uses the persona file + all five memory layers + the incoming message + the
understanding analysis. Outputs reply text + confidence score + rationale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas import (
    IncomingMessage, ContactProfile, MessageAnalysis, GeneratedReply,
    ConversationTurn,
)


REPLY_SYSTEM_PROMPT = """You are an AI replying to WhatsApp messages AS {user_name}.
You impersonate {user_name}'s voice, tone, and judgment — not your own.

Below is {user_name}'s persona document. Follow it precisely.

==== PERSONA ====
{persona}
=================

Output strict JSON:
{{
  "text": "the reply text, ready to send",
  "confidence": 0.0-1.0,
  "rationale": "brief why this reply"
}}

Rules:
- Reply in {user_name}'s voice. Match length, casing, language mix.
- NEVER include placeholders like [name], [date], [link]. If you don't know a value, ask or draft.
- NEVER commit to meetings, money, deadlines without {user_name} confirming — escalate instead.
- If the safest reply is a short acknowledgment ("got it, will revert"), prefer that over fabricating.
- confidence: how sure you are this reply is appropriate AND accurate.
  - 0.9+ : simple ack, no fabricated facts, low risk
  - 0.7-0.9 : reasonable reply, minor uncertainty
  - <0.7   : significant uncertainty — should be reviewed before sending
"""


class ReplyGenerator:
    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=settings.gemini_api_key)

        persona_path = Path(settings.persona_file)
        persona = persona_path.read_text() if persona_path.exists() else "(no persona file)"

        self.model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=REPLY_SYSTEM_PROMPT.format(
                user_name=settings.user_name,
                persona=persona,
            ),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, msg: IncomingMessage, profile: ContactProfile,
                 analysis: MessageAnalysis, recent_conv: str,
                 past_episodes: str, relevant_facts: str,
                 pending_actions: list[str]) -> GeneratedReply:
        prompt = self._build_prompt(
            msg, profile, analysis, recent_conv, past_episodes, relevant_facts, pending_actions
        )
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.6},
        )
        data = self._safe_parse(response.text)
        return GeneratedReply(**data)

    def summarize_episode(self, turns: list[ConversationTurn]) -> tuple[str, list[str]]:
        """Summarize a chunk of turns into an episode summary + key points."""
        convo = "\n".join(
            f"{'Them' if t.role == 'contact' else 'Me'}: {t.text}" for t in turns
        )
        prompt = f"""Summarize this WhatsApp conversation chunk in 2-3 sentences,
then list 1-5 key points worth remembering.

Conversation:
{convo}

Return strict JSON: {{"summary": "...", "key_points": ["..."]}}"""
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.3},
        )
        data = self._safe_parse(response.text)
        return data.get("summary", ""), data.get("key_points", [])

    @staticmethod
    def _build_prompt(msg, profile, analysis, recent_conv, past_episodes,
                      relevant_facts, pending_actions) -> str:
        pending = "\n".join(f"- {a}" for a in pending_actions) or "(none)"
        return f"""Write a reply to this WhatsApp message.

CONTACT
  Name: {profile.name or "unknown"}  ({profile.phone})
  Relationship: {profile.relationship or "unknown"}
  Communication style: {profile.communication_style or "unknown"}
  Preferred language: {profile.preferred_language or "unknown"}
  Notes: {profile.notes or "(none)"}

RECENT CONVERSATION
{recent_conv}

PAST EPISODES (older summaries)
{past_episodes}

RELEVANT FACTS ABOUT THIS CONTACT
{relevant_facts}

PENDING ACTIONS
{pending}

INCOMING MESSAGE
"{msg.text}"

UNDERSTANDING (from analysis layer)
  Intent: {analysis.intent}
  Emotion: {analysis.emotion}
  Urgency: {analysis.urgency}/5
  Topic: {analysis.topic}
  Requires my personal knowledge: {analysis.requires_my_knowledge}
  Action items detected: {analysis.action_items}
  Reasoning: {analysis.reasoning}

Write the reply now. Output JSON only."""

    @staticmethod
    def _safe_parse(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[-2] if text.count("```") >= 2 else text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise
