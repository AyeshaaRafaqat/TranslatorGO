from __future__ import annotations

import json
from pathlib import Path
from typing import List, TypedDict

from config import get_settings


class ChatMessage(TypedDict, total=False):
    role: str  # "user" or "assistant"
    content: str
    insight: str


class MemoryService:
    def __init__(self, store_path: Path | None = None) -> None:
        settings = get_settings()
        self.history_limit = settings.history_limit
        self.store_path = store_path or Path("data/sessions.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_store(self) -> dict[str, list[ChatMessage]]:
        if not self.store_path.exists():
            return {}
        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Corrupt store, start fresh
            return {}

    def _save_store(self, store: dict[str, list[ChatMessage]]) -> None:
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        store = self._load_store()
        return store.get(session_id, [])

    def append_message(self, session_id: str, role: str, content: str, **kwargs) -> List[ChatMessage]:
        store = self._load_store()
        history = store.get(session_id, [])
        message = {"role": role, "content": content}
        message.update(kwargs)
        history.append(message)
        history = history[-self.history_limit :]
        store[session_id] = history
        self._save_store(store)
        return history

    def clear_history(self, session_id: str) -> None:
        store = self._load_store()
        if session_id in store:
            del store[session_id]
            self._save_store(store)
