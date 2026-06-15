"""Understanding: analyze an incoming message and decide what to do.

Outputs a structured MessageAnalysis with:
- intent, emotion, urgency, topic
- whether reply needs Ankit's personal knowledge
- extracted action items + facts to remember
- suggested action (reply_now / draft_for_review / stay_silent / escalate_to_user)
"""
from __future__ import annotations

import json
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas import IncomingMessage, MessageAnalysis, ContactProfile


SYSTEM_PROMPT = """You are the analytical layer of {user_name}'s WhatsApp agent.
Your job: read an incoming WhatsApp message and output a structured analysis.

You decide WHAT the message means and WHAT should happen — but you do NOT write the reply.

Output strict JSON with this schema:
{{
  "intent": "question|request|social_chat|scheduling|information_share|complaint|urgent|spam|other",
  "emotion": "neutral|happy|sad|angry|anxious|excited|frustrated",
  "urgency": 1-5,
  "topic": "short phrase",
  "requires_my_knowledge": true|false,
  "action_items": ["..."],
  "extracted_facts": ["..."],
  "suggested_action": "reply_now|draft_for_review|stay_silent|escalate_to_user",
  "reasoning": "brief why"
}}

Decision rules:
- escalate_to_user: any of — money/payment, contract/legal, deadline commitments, emergencies,
  sensitive personal topics, anything the user must personally decide.
- stay_silent: spam, broadcasts, accidental messages, messages that don't expect a reply.
- reply_now: simple acknowledgments, social pleasantries, questions you can confidently answer
  from context, scheduling confirmations of already-agreed times.
- draft_for_review: default for anything in between. When in doubt, draft.

extracted_facts = atomic durable facts worth remembering about the contact (preferences, dates,
relationships, ongoing projects). NOT chit-chat. Examples:
  GOOD: "Has a daughter named Anika, age 6"
  GOOD: "Prefers WhatsApp calls over voice notes"
  BAD: "Said hi today"
"""


class Understanding:
    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=SYSTEM_PROMPT.format(user_name=settings.user_name),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def analyze(self, msg: IncomingMessage, profile: ContactProfile,
                recent_conv: str, past_episodes: str, relevant_facts: str,
                pending_actions: list[str]) -> MessageAnalysis:
        user_prompt = self._build_prompt(
            msg, profile, recent_conv, past_episodes, relevant_facts, pending_actions
        )
        response = self.model.generate_content(
            user_prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        data = self._safe_parse(response.text)
        return MessageAnalysis(**data)

    @staticmethod
    def _build_prompt(msg, profile, recent_conv, past_episodes, relevant_facts, pending_actions) -> str:
        pending = "\n".join(f"- {a}" for a in pending_actions) or "(none)"
        return f"""Analyze this incoming WhatsApp message.

CONTACT PROFILE
  Phone: {profile.phone}
  Name: {profile.name or "unknown"}
  Relationship: {profile.relationship or "unknown"}
  Communication style: {profile.communication_style or "unknown"}
  Notes: {profile.notes or "(none)"}

RECENT CONVERSATION (last few turns)
{recent_conv}

PAST EPISODES (summaries from before)
{past_episodes}

RELEVANT FACTS (from semantic memory)
{relevant_facts}

PENDING ACTIONS WITH THIS CONTACT
{pending}

INCOMING MESSAGE (just received)
"{msg.text}"

Return the JSON analysis. Be conservative: when in doubt, suggest draft_for_review or escalate_to_user."""

    @staticmethod
    def _safe_parse(text: str) -> dict:
        # strip possible code fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[-2] if text.count("```") >= 2 else text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # last-resort: extract first {...}
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise
