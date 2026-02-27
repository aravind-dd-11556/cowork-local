"""
Memory Tool — Lets the agent explicitly store and retrieve knowledge.

Actions:
  remember — Store a fact, preference, or decision
  recall   — Retrieve a specific entry by key
  search   — Search knowledge by query string
  forget   — Remove an entry
"""

from __future__ import annotations

from .base import BaseTool
from ..core.models import ToolResult
from ..core.knowledge_store import KnowledgeStore


class MemoryTool(BaseTool):
    """Tool for the agent to persist and retrieve cross-session knowledge."""

    name = "memory_store"
    description = (
        "Store and retrieve knowledge that persists across sessions. "
        "Use this to remember important facts, user preferences, and key decisions. "
        "Actions: 'remember' (store), 'recall' (get by key), 'search' (find by query), 'forget' (delete)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall", "search", "forget"],
                "description": "The action to perform.",
            },
            "category": {
                "type": "string",
                "enum": ["facts", "preferences", "decisions"],
                "description": "Category of knowledge.",
            },
            "key": {
                "type": "string",
                "description": "The key or name for this piece of knowledge.",
            },
            "value": {
                "type": "string",
                "description": "The value to store (required for 'remember' action).",
            },
        },
        "required": ["action", "category", "key"],
    }

    def __init__(self, knowledge_store: KnowledgeStore):
        self._store = knowledge_store

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "")
        category = kwargs.get("category", "facts")
        key = kwargs.get("key", "")
        value = kwargs.get("value", "")
        tool_id = kwargs.get("tool_id", "")

        if not action:
            return self._error("Missing 'action' parameter.", tool_id)
        if not key and action != "search":
            return self._error("Missing 'key' parameter.", tool_id)

        try:
            if action == "remember":
                if not value:
                    return self._error("Missing 'value' parameter for remember action.", tool_id)
                self._store.remember(category, key, value)
                return self._success(
                    f"Remembered [{category}] {key}: {value}",
                    tool_id,
                )

            elif action == "recall":
                result = self._store.recall(category, key)
                if result:
                    return self._success(f"{key}: {result}", tool_id)
                else:
                    return self._success(f"No entry found for key '{key}' in category '{category}'.", tool_id)

            elif action == "search":
                query = key or value  # Allow searching with either field
                if not query:
                    return self._error("Provide a search query in 'key' or 'value'.", tool_id)
                entries = self._store.search(query, limit=10)
                if entries:
                    lines = [f"Found {len(entries)} result(s):"]
                    for e in entries:
                        lines.append(f"  [{e.category}] {e.key}: {e.value}")
                    return self._success("\n".join(lines), tool_id)
                else:
                    return self._success(f"No results found for '{query}'.", tool_id)

            elif action == "forget":
                removed = self._store.forget(key)
                if removed:
                    return self._success(f"Forgot '{key}'.", tool_id)
                else:
                    return self._success(f"No entry found for key '{key}'.", tool_id)

            else:
                return self._error(f"Unknown action '{action}'. Use: remember, recall, search, forget.", tool_id)

        except Exception as e:
            return self._error(f"Memory tool error: {str(e)}", tool_id)
