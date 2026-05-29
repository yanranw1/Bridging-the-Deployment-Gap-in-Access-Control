"""
Configuration and settings management.
Loads from .env file or environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    def __init__(self):
        # Load .env from project root
        load_dotenv(Path(__file__).parent.parent / ".env")

        # OpenAI
        self.openai_api_key: str = self._require("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Gmail OAuth
        self.gmail_credentials_file: str = os.getenv(
            "GMAIL_CREDENTIALS_FILE", "credentials.json"
        )
        self.gmail_token_file: str = os.getenv("GMAIL_TOKEN_FILE", "token.json")

        # Agent behavior
        self.auto_send: bool = os.getenv("AUTO_SEND", "false").lower() == "true"
        self.max_emails_fetch: int = int(os.getenv("MAX_EMAILS_FETCH", "20"))
        self.tasks_file: str = os.getenv("TASKS_FILE", "tasks.json")
        self.user_name: str = os.getenv("USER_NAME", "User")
        self.user_email: str = os.getenv("USER_EMAIL", "")

    def _require(self, key: str) -> str:
        val = os.getenv(key)
        if not val:
            raise EnvironmentError(
                f"Missing required environment variable: {key}\n"
                f"Please add it to your .env file."
            )
        return val
