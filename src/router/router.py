from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .session import SessionState
from .intent import (
    looks_like_explicit_booking_request,
    looks_like_no,
    llm_classify_booking_intent,
)

ROOT = Path(__file__).resolve().parents[2]


def infer_role_from_prompt(role_prompt: str) -> Tuple[str, float]:
    """
    Robust: use Ollama to choose the closest role from role_tools.json keys.
    Falls back to string match if Ollama fails.
    """
    text = (role_prompt or "").strip()
    mapping = load_role_tools()
    roles = list(mapping.keys())

    # fallback if config missing
    if not roles:
        return "general", 0.5

    # Fast fallback: string match
    low = text.lower()
    for role in roles:
        if role.lower() in low:
            return role, 0.95

    # LLM decide
    try:
        from ..ollama_client import OllamaClient

        # Keep this consistent with the default runtime model.
        model = "qwen2.5:7b-instruct"

        prompt = (
            "Pick the best role name for this assistant.\n"
            "Return ONLY one role name from the list.\n\n"
            f"Roles: {roles}\n\n"
            f"Role prompt:\n{text}\n"
        )

        client = OllamaClient()
        out = client.chat(model=model, messages=[{"role": "user", "content": prompt}]).strip()

        # normalize: must be exactly one of the roles
        for role in roles:
            if out.strip().lower() == role.lower():
                return role, 0.9

        return "general", 0.5
    except Exception:
        return "general", 0.5


def load_role_tools() -> dict:
    """
    Load configs/role_tools.json.
    Only roles that have tools need to exist in the file.
    Missing role => no tools (free chat).
    """
    path = ROOT / "configs" / "role_tools.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def allowed_tools_for_role(role_name: str) -> List[str]:
    mapping = load_role_tools()
    tools = mapping.get(role_name, [])
    return [t for t in tools if isinstance(t, str)]

"""router.py

Routing policy:
- The router decides the ROLE and which tools are ALLOWED.
- It does NOT auto-start clinic booking tools (Jarvis mode).
- Tool lock (session.active_tool) still wins.
"""


@dataclass
class RouteDecision:
    role_name: str
    role_confidence: float
    allowed_tools: List[str]
    forced_tool: Optional[str] = None


def route_turn(role_prompt: str, user_text: str, session: SessionState) -> RouteDecision:
    """
    Decide:
    - which tools are allowed (based on inferred role)
    - whether to force-call a tool (tool lock + booking intent)

    Priority order:
    1) tool lock always wins
    2) if hotel role + booking intent => force hotel_book AND start lock
    3) otherwise free chat (LLM can respond; tools may still be available)
    """
    role_name, conf = infer_role_from_prompt(role_prompt)
    allowed = allowed_tools_for_role(role_name)

    # ---- Jarvis intent gating ----
    lookup_keywords = (
        "find booking",
        "find appointment",
        "my booking",
        "my appointment",
        "cancel",
        "reschedule",
        "change appointment",
        "modify appointment",
    )

    text_l = (user_text or "").lower()

    is_lookup_intent = any(k in text_l for k in lookup_keywords)

    # Only allow clinic_lookup for lookup / cancel / reschedule intents
    if "clinic_lookup" in allowed and not is_lookup_intent:
        allowed = []


    # Default: tools not allowed unless we decide otherwise
    session.tool_allowed = False
    forced_tool: Optional[str] = None

    # 1) tool lock always wins (mid booking flow)
    if session.active_tool:
        session.tool_allowed = True
        return RouteDecision(
            role_name=role_name,
            role_confidence=conf,
            allowed_tools=allowed,
            forced_tool=session.active_tool,
        )

    # Decide which booking tool applies in this role (if any).
    # Note: clinic booking is intentionally NOT a tool in Jarvis mode.
    booking_tool: Optional[str] = None
    if "hotel_book" in allowed:
        booking_tool = "hotel_book"

    # Helper: fetch last assistant message from history (if any)
    last_assistant = ""
    for m in reversed(session.history):
        if m.get("role") == "assistant":
            last_assistant = m.get("content", "")
            break

    # 2) If there is a pending_action, interpret yes/no ONLY for that action
    if session.pending_action and booking_tool:
        if looks_like_no(user_text):
            session.pending_action = None
            session.tool_allowed = False
            forced_tool = None
            return RouteDecision(role_name, conf, allowed, forced_tool)

        decision = llm_classify_booking_intent(
            user_text=user_text,
            last_assistant_text=last_assistant,
            pending_action=session.pending_action,
        )

        if decision == "BOOK_NOW":
            session.active_tool = session.pending_action.get("tool", booking_tool)
            session.pending_action = None
            session.tool_allowed = True
            forced_tool = session.active_tool
            return RouteDecision(role_name, conf, allowed, forced_tool)

        # ambiguous reply -> stay free-flow
        session.tool_allowed = False
        forced_tool = None
        return RouteDecision(role_name, conf, allowed, forced_tool)

    # 3) Jarvis mode: do NOT auto-start tools from user text.
    # Any booking/action should be confirmed in chat and later finalized offline.
    # We keep hotel tool lock/pending_action behavior only.
    if booking_tool:
        # We still allow the LLM to interpret an explicit YES to a pending action (handled above).
        # But we don't start new tool locks here.
        pass

    # 4) normal free chat path
    session.tool_allowed = False
    return RouteDecision(
        role_name=role_name,
        role_confidence=conf,
        allowed_tools=allowed,
        forced_tool=None,
    )
