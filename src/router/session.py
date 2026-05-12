from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionState:
    """
    Keeps conversation-level state so routing is stable.

    - active_tool: tool lock (if set, every next user turn goes to this tool)
    - pending_action: system-tracked offer waiting for user confirmation
      Example: {"action": "confirm_tool", "tool": "hotel_book"}
    - tool_allowed: seatbelt flag checked by agent_core before executing any tool
    - history: short-term memory (ONLY natural user/assistant turns)
    """
    active_tool: Optional[str] = None

    # set when assistant offers an action and waits for "yes/ok"
    pending_action: Optional[Dict[str, str]] = None

    # only True when user explicitly wants the action, or while active_tool is locked
    tool_allowed: bool = False

    history: List[Dict[str, str]] = field(default_factory=list)

    # keep last N turns to avoid huge context (you can tune this)
    max_history: int = 8
