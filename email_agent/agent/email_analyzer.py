"""
Email Analyzer - Uses OpenAI to deeply understand emails.
Classifies, extracts intent, key info, and suggests actions.
"""

import json
from openai import OpenAI


ANALYSIS_PROMPT = """
You are an expert email analyst. Analyze the following email and return a JSON object with this exact structure:

{
  "category": "work|personal|promo|spam|newsletter|urgent|finance|social",
  "intent": "request|complaint|meeting_request|info|follow_up|confirmation|introduction|other",
  "urgency": "high|medium|low",
  "sentiment": "positive|neutral|negative|urgent",
  "key_info": {
    "dates": ["list of any dates/times mentioned"],
    "names": ["list of people mentioned"],
    "tasks": ["list of action items or requests"],
    "deadlines": ["list of deadlines"],
    "topics": ["main topics"]
  },
  "requires_reply": true,
  "suggested_actions": ["list of recommended actions"],
  "summary": "2-3 sentence summary of the email",
  "reply_points": ["key points to address in a reply, if needed"]
}

Return ONLY valid JSON, no markdown, no explanation.
"""

REPLY_PROMPT = """
You are a professional email assistant. Write a reply to the following email.

Instructions:
- Tone: {tone}
- Additional instructions: {instructions}
- Be concise and natural
- Address all key points from the original email
- Do NOT include a subject line — only the body text
- Sign off with just "{user_name}"

Original email:
From: {sender}
Subject: {subject}
---
{body}
---

Write only the reply body text.
"""

SUMMARIZE_PROMPT = """
You are an email inbox assistant. Summarize the following emails clearly and concisely.

Format your response as:
📊 INBOX SUMMARY ({count} emails)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 URGENT / ACTION REQUIRED:
[List urgent items needing immediate attention]

📋 ACTION ITEMS:
[List tasks, requests, or items needing a response]

📅 MEETINGS & SCHEDULING:
[List any meeting requests or scheduling needs]

📨 FYI / INFO:
[Brief mentions of informational emails]

💡 RECOMMENDATION:
[What to tackle first]

Focus on what matters. Be brief but complete.
"""


class EmailAnalyzer:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o"):
        self.client = openai_client
        self.model = model

    def analyze(self, email: dict) -> dict:
        """Deeply analyze an email and return structured insights."""
        email_text = f"""
From: {email['from']}
To: {email['to']}
Subject: {email['subject']}
Date: {email['date']}
---
{email['body'][:3000]}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": email_text},
                ],
                temperature=0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            return {
                "category": "unknown",
                "intent": "unknown",
                "urgency": "low",
                "sentiment": "neutral",
                "key_info": {"dates": [], "names": [], "tasks": [], "deadlines": [], "topics": []},
                "requires_reply": False,
                "suggested_actions": [],
                "summary": email.get("snippet", "Unable to analyze."),
                "reply_points": [],
                "error": str(e),
            }

    def draft_reply(
        self,
        email: dict,
        tone: str = "professional",
        instructions: str = "",
        user_name: str = "User",
    ) -> str:
        """Draft a context-aware reply to an email."""
        prompt = REPLY_PROMPT.format(
            tone=tone,
            instructions=instructions or "Write a helpful, appropriate reply",
            user_name=user_name,
            sender=email["from"],
            subject=email["subject"],
            body=email["body"][:3000],
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def summarize_emails(self, emails: list[dict]) -> str:
        """Summarize a list of emails into an actionable digest."""
        if not emails:
            return "No emails to summarize."

        email_list = ""
        for i, e in enumerate(emails[:20], 1):
            email_list += f"\n{i}. From: {e['from']} | Subject: {e['subject']}\n   {e['snippet'][:200]}\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT.format(count=len(emails))},
                {"role": "user", "content": email_list},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
