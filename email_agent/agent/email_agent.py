"""
Email Agent - Core orchestrator.
Uses OpenAI function calling to decide what tools to use.
Runs as an interactive CLI loop.
"""

import json
from typing import Optional
from openai import OpenAI

from config.settings import Settings
from tools.gmail_tool import GmailTool
from tools.task_manager import TaskManager
from agent.email_analyzer import EmailAnalyzer
from agent.tool_definitions import TOOLS


SYSTEM_PROMPT = """You are an intelligent email assistant for {user_name} ({user_email}).

Your capabilities:
- Read and understand emails deeply (classify, detect intent, extract key info)
- Draft context-aware replies matching the right tone
- Search and summarize the inbox
- Turn emails into tasks/to-dos
- Send emails ONLY after explicit user approval

Your personality:
- Proactive: suggest actions the user hasn't asked for when they're clearly needed
- Concise: don't over-explain, be direct
- Safe: ALWAYS show draft replies before sending — never send without approval
- Smart: if an email has a task/deadline, mention creating a task for it

When analyzing emails, always call analyze_email first for deep understanding.
When drafting replies, always call draft_reply and show the user before asking for send approval.
Today's date for context: {date}.
"""


class EmailAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.gmail = GmailTool(settings.gmail_credentials_file, settings.gmail_token_file)
        self.tasks = TaskManager(settings.tasks_file)
        self.analyzer = EmailAnalyzer(self.client, settings.openai_model)
        self.conversation: list[dict] = []
        self._pending_draft: Optional[dict] = None  # Draft awaiting send approval

        # Get user email from Gmail profile if not set
        if not settings.user_email:
            profile = self.gmail.get_profile()
            settings.user_email = profile.get("email", "")

        print(f"✅ Connected to Gmail: {settings.user_email}")
        print(f"✅ OpenAI model: {settings.openai_model}")
        print(f"✅ Tasks file: {settings.tasks_file}")
        print("\nType your request below. Examples:")
        print('  • "Summarize my inbox"')
        print('  • "Show unread emails"')
        print('  • "Find emails from John about the project"')
        print('  • "Analyze the last email and draft a reply"')
        print('  • "Show my tasks"')
        print('  • Type "exit" to quit\n')

    def _system_prompt(self) -> str:
        from datetime import datetime
        return SYSTEM_PROMPT.format(
            user_name=self.settings.user_name,
            user_email=self.settings.user_email,
            date=datetime.now().strftime("%A, %B %d, %Y"),
        )

    def run(self):
        """Main interactive loop."""
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye"):
                print("\n👋 Goodbye!")
                break

            response = self._chat(user_input)
            print(f"\n🤖 Agent: {response}\n")

    def _chat(self, user_message: str) -> str:
        """Process a user message through the OpenAI function-calling loop."""
        self.conversation.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self._system_prompt()}] + self.conversation

        # Agentic loop: keep calling OpenAI until no more tool calls
        max_iterations = 8
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1500,
            )

            msg = response.choices[0].message

            # No tool calls — we have the final answer
            if not msg.tool_calls:
                final_text = msg.content or ""
                self.conversation.append({"role": "assistant", "content": final_text})
                return final_text

            # Execute tool calls
            messages.append(msg)
            tool_results = []
            for tc in msg.tool_calls:
                result = self._execute_tool(tc.function.name, tc.function.arguments)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })

            messages.extend(tool_results)

        return "I ran into an issue processing your request. Please try again."

    def _execute_tool(self, name: str, args_json: str) -> object:
        """Route tool calls to the right implementation."""
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            args = {}

        print(f"  🔧 Using tool: {name}", end="", flush=True)

        try:
            result = self._dispatch(name, args)
            print(" ✓")
            return result
        except Exception as e:
            print(f" ✗ ({e})")
            return {"error": str(e)}

    def _dispatch(self, name: str, args: dict) -> object:
        """Dispatch to the correct tool handler."""

        if name == "list_emails":
            emails = self.gmail.list_emails(
                max_results=args.get("max_results", 10),
                query=args.get("query", ""),
            )
            return self._format_email_list(emails)

        elif name == "get_email":
            email = self.gmail.get_email(args["email_id"])
            return email if email else {"error": "Email not found"}

        elif name == "search_emails":
            emails = self.gmail.search_emails(
                query=args["query"],
                max_results=args.get("max_results", 10),
            )
            return self._format_email_list(emails)

        elif name == "analyze_email":
            email = self.gmail.get_email(args["email_id"])
            if not email:
                return {"error": "Email not found"}
            analysis = self.analyzer.analyze(email)
            return {
                "email": {
                    "id": email["id"],
                    "from": email["from"],
                    "subject": email["subject"],
                    "date": email["date"],
                },
                "analysis": analysis,
            }

        elif name == "draft_reply":
            email = self.gmail.get_email(args["email_id"])
            if not email:
                return {"error": "Email not found"}
            draft = self.analyzer.draft_reply(
                email,
                tone=args.get("tone", "professional"),
                instructions=args.get("instructions", ""),
                user_name=self.settings.user_name,
            )
            # Store draft for potential approval
            self._pending_draft = {
                "to": self._extract_email_address(email["from"]),
                "subject": f"Re: {email['subject']}",
                "body": draft,
                "reply_to_id": email["id"],
                "thread_id": email["thread_id"],
            }
            return {
                "draft": draft,
                "to": self._pending_draft["to"],
                "subject": self._pending_draft["subject"],
                "note": "Draft ready. Ask user to approve before sending. Show the full draft text.",
            }

        elif name == "send_email":
            result = self.gmail.send_email(
                to=args["to"],
                subject=args["subject"],
                body=args["body"],
                reply_to_id=args.get("reply_to_id"),
                thread_id=args.get("thread_id"),
            )
            self._pending_draft = None
            return result
        elif name == "send_draft":
            try:
                draft_id = args.get("draft_id")
                if not draft_id:
                    return {"error": "No draft_id provided. Call list_drafts first to get a draft_id."}
                result = self.gmail.service.users().drafts().send(
                    userId="me",
                    body={"id": draft_id}
                ).execute()
                return {"success": True, "message_id": result.get("id"), "thread_id": result.get("threadId")}
            except Exception as e:
                return {"error": str(e)}

        elif name == "summarize_inbox":
            focus = args.get("focus", "")
            query = focus if focus else ""
            emails = self.gmail.list_emails(
                max_results=args.get("max_results", 15),
                query=query,
            )
            summary = self.analyzer.summarize_emails(emails)
            return {"summary": summary, "email_count": len(emails)}

        elif name == "create_task":
            task = self.tasks.add_task(
                title=args["title"],
                description=args.get("description", ""),
                deadline=args.get("deadline"),
                email_id=args.get("email_id"),
                email_subject=args.get("email_subject"),
                priority=args.get("priority", "normal"),
            )
            return {"task_created": task}

        elif name == "list_tasks":
            status = args.get("status", "pending")
            task_list = self.tasks.list_tasks(status)
            return {
                "tasks": task_list,
                "display": self.tasks.format_tasks_display(task_list),
                "count": len(task_list),
            }

        elif name == "complete_task":
            success = self.tasks.complete_task(args["task_id"])
            return {"success": success, "task_id": args["task_id"]}
        
        elif name == "forward_email":
            email_id = args.get("email_id")
            if not email_id:
                return {"error": "No email_id provided. Call list_emails first to find the email ID."}

            original = self.gmail.get_email(email_id)
            if not original:
                return {"error": f"Email {email_id} not found. Call list_emails to get valid IDs."}

            fwd_body = (
                f"{args.get('message', '')}\n\n"
                f"---------- Forwarded message ----------\n"
                f"From: {original['from']}\n"
                f"Date: {original['date']}\n"
                f"Subject: {original['subject']}\n"
                f"To: {original['to']}\n\n"
                f"{original['body']}"
            )

            result = self.gmail.send_email(
                to=args["to"],
                subject=f"Fwd: {original['subject']}",
                body=fwd_body,
            )

            return {"success": True, "sent_to": args["to"], "subject": f"Fwd: {original['subject']}", **result}

        elif name == "delete_email":
            try:
                self.gmail.service.users().messages().trash(
                    userId="me", id=args["email_id"]
                ).execute()
                return {"success": True, "email_id": args["email_id"], "action": "moved to trash"}
            except Exception as e:
                return {"error": str(e)}

        elif name == "draft_email":
            try:
                import base64
                from email.mime.text import MIMEText
                message = MIMEText(args["body"])
                message["to"] = args.get("to", "")
                message["subject"] = args.get("subject", "")
                raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
                draft = self.gmail.service.users().drafts().create(
                    userId="me",
                    body={"message": {"raw": raw}}
                ).execute()
                return {
                    "success": True,
                    "draft_id": draft["id"],
                    "to": args.get("to", ""),
                    "subject": args.get("subject", ""),
                }
            except Exception as e:
                return {"error": str(e)}

        elif name == "star_email":
            try:
                label_action = (
                    {"addLabelIds": ["STARRED"]}
                    if args.get("star", True)
                    else {"removeLabelIds": ["STARRED"]}
                )
                self.gmail.service.users().messages().modify(
                    userId="me", id=args["email_id"], body=label_action
                ).execute()
                action = "starred" if args.get("star", True) else "unstarred"
                return {"success": True, "email_id": args["email_id"], "action": action}
            except Exception as e:
                return {"error": str(e)}

        else:
            return {"error": f"Unknown tool: {name}"}

    def _format_email_list(self, emails: list[dict]) -> dict:
        """Format email list for the LLM."""
        formatted = []
        for e in emails:
            formatted.append({
                "id": e["id"],
                "from": e["from"],
                "subject": e["subject"],
                "date": e["date"],
                "snippet": e["snippet"][:200],
                "unread": e.get("unread", False),
            })
        return {"emails": formatted, "count": len(formatted)}

    def _extract_email_address(self, from_field: str) -> str:
        """Extract email address from 'Name <email>' format."""
        if "<" in from_field and ">" in from_field:
            return from_field.split("<")[1].split(">")[0].strip()
        return from_field.strip()
