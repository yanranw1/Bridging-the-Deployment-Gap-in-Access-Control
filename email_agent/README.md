# 📧 AI Email Agent

An intelligent email agent powered by **OpenAI GPT-4o** and **Gmail API**. It understands your emails deeply, drafts smart replies, manages tasks, and summarizes your inbox — all through a natural language CLI.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Understanding** | Classify emails, detect intent, extract dates/names/deadlines |
| ✉️ **Smart Replies** | Context-aware drafts with tone matching — requires your approval before sending |
| 🔎 **Search & Summarize** | Natural language inbox search, actionable summaries |
| 📌 **Task Extraction** | Auto-convert emails into to-dos with deadlines and priorities |
| 🔧 **Function Calling** | Uses OpenAI function calling for reliable, structured tool use |

---

## 🚀 Setup

### 1. Install dependencies

```bash
cd email_agent
pip install -r requirements.txt
```

### 2. Configure OpenAI

Copy the env template and add your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Set up Gmail API

#### Step 1: Create a Google Cloud project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or use existing)
3. Enable the **Gmail API**: APIs & Services → Enable APIs → search "Gmail API"

#### Step 2: Create OAuth 2.0 credentials
1. APIs & Services → Credentials → Create Credentials → **OAuth client ID**
2. Application type: **Desktop app**
3. Download the JSON file
4. Save it as `credentials.json` in the `email_agent/` folder

#### Step 3: Configure OAuth consent screen
1. APIs & Services → OAuth consent screen
2. User type: **External** (or Internal if using Google Workspace)
3. Add your Gmail address as a test user
4. Scopes needed: Gmail read, send, modify

### 4. Run the agent

```bash
python main.py
```

On first run, a browser window will open for Gmail authorization. After approving, a `token.json` is saved for future runs.

---

## 💬 Example Commands

```
You: Summarize my inbox
You: Show unread emails
You: Find emails from Sarah about the contract
You: Analyze email [id] and draft a formal reply
You: What tasks do I have?
You: Create a task: Review proposal — due Friday
You: Mark task abc123 as done
```

---

## 🏗️ Architecture

```
email_agent/
├── main.py                    # Entry point
├── .env                       # Your config (git-ignored)
├── requirements.txt
│
├── config/
│   └── settings.py            # Loads .env settings
│
├── tools/
│   ├── gmail_tool.py          # Gmail API wrapper (read/send/search)
│   └── task_manager.py        # JSON-based task persistence
│
└── agent/
    ├── email_agent.py         # Orchestrator — OpenAI function calling loop
    ├── email_analyzer.py      # Deep analysis & reply drafting with GPT
    └── tool_definitions.py    # OpenAI tool/function schemas
```

### How it works

1. You type a natural language request
2. The agent sends it to GPT-4o with a set of **tool definitions**
3. GPT-4o decides which tools to call (e.g. `list_emails`, `analyze_email`)
4. Tools execute against Gmail API / task store
5. Results feed back to GPT-4o which synthesizes a final response
6. For replies: a draft is shown for your approval before any send

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-4o` | Model to use |
| `GMAIL_CREDENTIALS_FILE` | `credentials.json` | OAuth credentials path |
| `GMAIL_TOKEN_FILE` | `token.json` | Auth token (auto-created) |
| `USER_NAME` | `User` | Your name for email sign-offs |
| `USER_EMAIL` | auto-detected | Your Gmail address |
| `MAX_EMAILS_FETCH` | `20` | Max emails per fetch |
| `TASKS_FILE` | `tasks.json` | Task storage path |
| `AUTO_SEND` | `false` | Skip send confirmation (not recommended) |

---

## 🔒 Safety

- **Never auto-sends**: All replies are drafted and shown for approval first
- **No data leaves**: Emails are sent to OpenAI API only for analysis — standard API privacy applies
- **Local tasks**: Task data stored in a local JSON file only
- **Scoped permissions**: Only requests Gmail read, send, and modify scopes

---

## 🛠️ Extending the Agent

To add a new capability:

1. Add a new tool definition in `agent/tool_definitions.py`
2. Add the handler in `agent/email_agent.py` → `_dispatch()`
3. Optionally add a helper class in `tools/`

The OpenAI function-calling loop handles everything else automatically.
