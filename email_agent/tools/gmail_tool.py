"""
Gmail Tool - Handles all Gmail API interactions.
Supports: read, send, search, list, reply, label, trash.
"""

import os
import base64
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


class GmailTool:
    def __init__(self, credentials_file: str, token_file: str):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self._authenticate()

    def _authenticate(self):
        """Authenticate with Gmail OAuth2."""
        creds = None

        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Gmail credentials file not found: {self.credentials_file}\n"
                        "Download it from Google Cloud Console → APIs & Services → Credentials."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_file, "w") as token:
                token.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    def list_emails(self, max_results: int = 20, query: str = "") -> list[dict]:
        """List emails from inbox, optionally filtered by query."""
        try:
            q = f"in:inbox {query}".strip()
            result = self.service.users().messages().list(
                userId="me", maxResults=max_results, q=q
            ).execute()

            messages = result.get("messages", [])
            emails = []
            for msg in messages:
                email = self.get_email(msg["id"])
                if email:
                    emails.append(email)
            return emails
        except HttpError as e:
            raise RuntimeError(f"Gmail API error listing emails: {e}")

    def get_email(self, message_id: str) -> Optional[dict]:
        """Fetch a single email by ID and parse it."""
        try:
            msg = self.service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()

            headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
            body = self._extract_body(msg["payload"])
            snippet = msg.get("snippet", "")

            return {
                "id": message_id,
                "thread_id": msg.get("threadId", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "date": headers.get("Date", ""),
                "snippet": snippet,
                "body": body or snippet,
                "labels": msg.get("labelIds", []),
                "unread": "UNREAD" in msg.get("labelIds", []),
            }
        except HttpError:
            return None

    def _extract_body(self, payload: dict) -> str:
        """Recursively extract plain text body from email payload."""
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        for part in payload.get("parts", []):
            result = self._extract_body(part)
            if result:
                return result
        return ""

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        reply_to_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """Send an email, optionally as a reply in a thread."""
        message = MIMEMultipart()
        message["to"] = to
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))

        if reply_to_id:
            message["In-Reply-To"] = reply_to_id
            message["References"] = reply_to_id

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body_payload = {"raw": raw}
        if thread_id:
            body_payload["threadId"] = thread_id

        try:
            sent = self.service.users().messages().send(
                userId="me", body=body_payload
            ).execute()
            return {"success": True, "id": sent["id"], "thread_id": sent.get("threadId")}
        except HttpError as e:
            raise RuntimeError(f"Failed to send email: {e}")

    def search_emails(self, query: str, max_results: int = 10) -> list[dict]:
        """Search Gmail with a query string."""
        return self.list_emails(max_results=max_results, query=query)

    def mark_as_read(self, message_id: str) -> bool:
        """Mark an email as read."""
        try:
            self.service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()
            return True
        except HttpError:
            return False

    def get_thread(self, thread_id: str) -> list[dict]:
        """Get all messages in a thread."""
        try:
            thread = self.service.users().threads().get(
                userId="me", id=thread_id, format="full"
            ).execute()
            emails = []
            for msg in thread.get("messages", []):
                headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
                body = self._extract_body(msg["payload"])
                emails.append({
                    "id": msg["id"],
                    "from": headers.get("From", ""),
                    "date": headers.get("Date", ""),
                    "body": body or msg.get("snippet", ""),
                })
            return emails
        except HttpError:
            return []

    def get_profile(self) -> dict:
        """Get the authenticated user's Gmail profile."""
        try:
            profile = self.service.users().getProfile(userId="me").execute()
            return {
                "email": profile.get("emailAddress", ""),
                "total_messages": profile.get("messagesTotal", 0),
                "threads_total": profile.get("threadsTotal", 0),
            }
        except HttpError:
            return {}
