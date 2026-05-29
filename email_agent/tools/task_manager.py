"""
Task Manager Tool - Persists tasks/todos derived from emails.
Stores tasks in a local JSON file.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class TaskManager:
    def __init__(self, tasks_file: str = "tasks.json"):
        self.tasks_file = Path(tasks_file)
        self.tasks: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.tasks_file.exists():
            with open(self.tasks_file, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.tasks_file, "w") as f:
            json.dump(self.tasks, f, indent=2)

    def add_task(
        self,
        title: str,
        description: str = "",
        deadline: Optional[str] = None,
        email_id: Optional[str] = None,
        email_subject: Optional[str] = None,
        priority: str = "normal",
    ) -> dict:
        task = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "description": description,
            "deadline": deadline,
            "email_id": email_id,
            "email_subject": email_subject,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        self.tasks.append(task)
        self._save()
        return task

    def list_tasks(self, status: str = "pending") -> list[dict]:
        if status == "all":
            return self.tasks
        return [t for t in self.tasks if t["status"] == status]

    def complete_task(self, task_id: str) -> bool:
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                self._save()
                return True
        return False

    def delete_task(self, task_id: str) -> bool:
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        if len(self.tasks) < before:
            self._save()
            return True
        return False

    def format_tasks_display(self, tasks: Optional[list] = None) -> str:
        tasks = tasks or self.list_tasks()
        if not tasks:
            return "No pending tasks."

        lines = []
        priority_icons = {"high": "🔴", "normal": "🟡", "low": "🟢"}
        for t in tasks:
            icon = priority_icons.get(t.get("priority", "normal"), "🟡")
            deadline = f" | Due: {t['deadline']}" if t.get("deadline") else ""
            source = f" | From: {t['email_subject']}" if t.get("email_subject") else ""
            lines.append(f"  [{t['id']}] {icon} {t['title']}{deadline}{source}")
        return "\n".join(lines)
