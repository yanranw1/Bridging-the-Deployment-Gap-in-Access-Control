# debug.py
import json
from config.settings import Settings
from tools.gmail_tool import GmailTool
from tools.task_manager import TaskManager
from agent.email_analyzer import EmailAnalyzer
from agent.email_agent import EmailAgent
from openai import OpenAI

AVAILABLE_TOOLS = [
    "list_emails", "get_email", "search_emails", "analyze_email",
    "draft_reply", "send_email", "summarize_inbox", "create_task",
    "list_tasks", "complete_task", "forward_email", "delete_email",
    "draft_email", "send_draft", "star_email",
]

def prompt_args(tool_name: str) -> dict:
    """Ask user for args based on which tool they picked."""
    print()
    args = {}

    if tool_name in ("get_email", "analyze_email", "delete_email", "star_email"):
        args["email_id"] = input("  email_id: ").strip()
        if tool_name == "star_email":
            args["star"] = input("  star? (y/n) [y]: ").strip().lower() != "n"

    elif tool_name == "list_emails":
        args["max_results"] = int(input("  max_results [10]: ").strip() or "10")
        args["query"] = input("  query (optional): ").strip()

    elif tool_name == "search_emails":
        args["query"] = input("  query: ").strip()
        args["max_results"] = int(input("  max_results [10]: ").strip() or "10")

    elif tool_name == "draft_reply":
        args["email_id"] = input("  email_id: ").strip()
        args["tone"] = input("  tone (formal/professional/casual/friendly) [professional]: ").strip() or "professional"
        args["instructions"] = input("  instructions (optional): ").strip()

    elif tool_name == "send_email":
        args["to"] = input("  to: ").strip()
        args["subject"] = input("  subject: ").strip()
        args["body"] = input("  body (one line): ").strip()
        args["reply_to_id"] = input("  reply_to_id (optional): ").strip() or None
        args["thread_id"] = input("  thread_id (optional): ").strip() or None

    elif tool_name == "forward_email":
        args["email_id"] = input("  email_id: ").strip()
        args["to"] = input("  to: ").strip()
        args["message"] = input("  prepend message (optional): ").strip()

    elif tool_name == "draft_email":
        args["to"] = input("  to: ").strip()
        args["subject"] = input("  subject: ").strip()
        args["body"] = input("  body (one line): ").strip()

    elif tool_name == "summarize_inbox":
        args["max_results"] = int(input("  max_results [15]: ").strip() or "15")
        args["focus"] = input("  focus (optional): ").strip()

    elif tool_name == "create_task":
        args["title"] = input("  title: ").strip()
        args["description"] = input("  description (optional): ").strip()
        args["deadline"] = input("  deadline (optional): ").strip()
        args["email_id"] = input("  email_id (optional): ").strip()
        args["priority"] = input("  priority (high/normal/low) [normal]: ").strip() or "normal"

    elif tool_name == "list_tasks":
        args["status"] = input("  status (pending/completed/all) [pending]: ").strip() or "pending"

    elif tool_name == "complete_task":
        args["task_id"] = input("  task_id: ").strip()
    
    elif tool_name == "send_draft":
        args["draft_id"] = input("  draft_id: ").strip()

    return args


def main():
    print("\n" + "="*60)
    print("  🔧  Email Agent — Direct Dispatch Debugger")
    print("="*60)

    settings = Settings()
    agent = EmailAgent(settings)

    while True:
        print("\nAvailable tools:")
        for i, t in enumerate(AVAILABLE_TOOLS, 1):
            print(f"  {i:>2}. {t}")
        print("   q. quit")

        choice = input("\nPick a tool (number or name): ").strip()
        if choice.lower() == "q":
            break

        # Accept number or name
        if choice.isdigit():
            idx = int(choice) - 1
            if not 0 <= idx < len(AVAILABLE_TOOLS):
                print("❌ Invalid number")
                continue
            tool_name = AVAILABLE_TOOLS[idx]
        elif choice in AVAILABLE_TOOLS:
            tool_name = choice
        else:
            print("❌ Unknown tool")
            continue

        print(f"\n→ {tool_name}")
        args = prompt_args(tool_name)

        print(f"\n⚙️  Calling _dispatch({tool_name}, {args})")
        result = agent._dispatch(tool_name, args)

        print("\n📤 Result:")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()