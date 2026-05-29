"""
OpenAI tool/function definitions for the email agent.
These are passed to the OpenAI API for function calling.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_emails",
            "description": "List recent emails from the inbox. Use to show unread emails, recent messages, or emails matching a filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "How many emails to fetch (default 10)",
                        "default": 10,
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional Gmail search query (e.g. 'is:unread', 'from:john@example.com', 'subject:invoice')",
                        "default": "",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_email",
            "description": "Get the full content of a specific email by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The Gmail message ID",
                    }
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search emails using Gmail search syntax. Use for queries like 'find emails from John about the project' or 'emails with attachments this week'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query string",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_email",
            "description": "Deeply analyze an email: classify it, detect intent, extract key info (dates, names, tasks, deadlines), and suggest a reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The Gmail message ID to analyze",
                    }
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_reply",
            "description": "Draft a reply to an email. Always shows the draft to the user for approval before sending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The Gmail message ID to reply to",
                    },
                    "tone": {
                        "type": "string",
                        "enum": ["formal", "professional", "casual", "friendly"],
                        "description": "Tone of the reply",
                        "default": "professional",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Additional instructions for the reply (e.g. 'decline the meeting', 'ask for more details', 'confirm attendance')",
                        "default": "",
                    },
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send a new email or a pre-approved reply. Only call this after the user has explicitly approved the draft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text",
                    },
                    "reply_to_id": {
                        "type": "string",
                        "description": "Optional: message ID this is a reply to",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Optional: thread ID to reply in",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_inbox",
            "description": "Summarize recent inbox emails, highlighting action items, urgent messages, and key information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Number of recent emails to summarize",
                        "default": 15,
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional focus area (e.g. 'unread only', 'today', 'from my team')",
                        "default": "",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a task or to-do from an email. Use when an email contains a request, deadline, or action item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short task title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed task description",
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Deadline in natural language (e.g. 'Monday', '2025-01-20', 'end of week')",
                    },
                    "email_id": {
                        "type": "string",
                        "description": "ID of the source email",
                    },
                    "email_subject": {
                        "type": "string",
                        "description": "Subject of the source email",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "normal", "low"],
                        "description": "Task priority",
                        "default": "normal",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "Show the current task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "completed", "all"],
                        "description": "Filter by task status",
                        "default": "pending",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a task as completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to mark complete",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forward_email",
            "description": "Forward an email to another recipient. IMPORTANT: You must have a valid email_id before calling this. If the user says 'this email' or 'that email' without specifying an ID, call list_emails first to get the ID of the most recent relevant email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The Gmail message ID to forward"},
                    "to": {"type": "string", "description": "Recipient email address"},
                    "message": {"type": "string", "description": "Optional message to prepend", "default": ""},
                },
                "required": ["email_id", "to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_email",
            "description": "Move an email to trash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The Gmail message ID to trash"},
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Use this whenever the user asks to draft, compose, or write a NEW email to someone. Saves it to Gmail drafts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "star_email",
            "description": "Star or unstar an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The Gmail message ID"},
                    "star": {"type": "boolean", "description": "True to star, False to unstar", "default": True},
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Send an existing Gmail draft by its draft ID. Use after draft_email to actually send it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "The Gmail draft ID returned by draft_email",
                    },
                },
                "required": ["draft_id"],
            },
        },
    },
]
