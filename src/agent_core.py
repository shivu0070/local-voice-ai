from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .ollama_client import OllamaClient
from .tools import TOOL_REGISTRY
from .router.session import SessionState
from .router.router import route_turn

def _extract_phone_from_text(text: str) -> Optional[str]:
    digits = re.sub(r"\D+", "", text or "")
    # accept 8-15 digit numbers
    if 8 <= len(digits) <= 15:
        return digits
    return None


def _extract_name_from_user_history(history: List[Dict[str, str]]) -> Optional[str]:
    # Very simple: look for "name is X" / "my name is X" / "name X"
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        t = (m.get("content") or "").strip()
        low = t.lower()

        m1 = re.search(r"\bmy name is\s+([a-zA-Z][a-zA-Z\s'-]{1,40})", t, flags=re.I)
        if m1:
            return m1.group(1).strip()

        m2 = re.search(r"\bname is\s+([a-zA-Z][a-zA-Z\s'-]{1,40})", t, flags=re.I)
        if m2:
            return m2.group(1).strip()

        m3 = re.search(r"\bname\s+([a-zA-Z][a-zA-Z\s'-]{1,40})", t, flags=re.I)
        if m3 and "phone" not in low:
            return m3.group(1).strip()

    return None


def _extract_date_like_from_text(text: str) -> Optional[str]:
    # Accept ISO yyyy-mm-dd already
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text or "")
    if m:
        return m.group(0)
    return None


def _has_any_date_in_history(history: List[Dict[str, str]]) -> bool:
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        if _extract_date_like_from_text(m.get("content") or ""):
            return True
    return False


def _is_placeholder_or_fake_booking(text: str) -> bool:
    """
    Blocks early/invalid confirmations like:
    - placeholders: [patient name], [phone]
    - fake phone: 1234567890
    - relative dates in confirmation: today/tomorrow
    - missing phone digits
    """
    t = (text or "").lower()

    bad_markers = [
        "[patient name]", "[phone]", "phone not provided",
        "1234567890", "today", "tomorrow",
    ]
    if any(m in t for m in bad_markers):
        return True

    # If it's a Confirm but doesn't contain a real-looking phone number, block it
    if "confirm:" in t:
        digits = re.sub(r"\D+", "", text or "")
        # Require at least 8 digits in the message to count as "phone present"
        if len(digits) < 8:
            return True

    return False


def _rewrite_early_confirm(assistant_text: str) -> str:
    """
    Converts a bad Confirm into a natural next question.
    Prefer asking for phone first if it's missing.
    """
    t = (assistant_text or "").lower()

    if "[patient name]" in t or "patient name" in t:
        return "Thanks. What is the patient name for the booking?"
    if "[phone]" in t or "phone" in t or "1234567890" in t:
        return "Thanks. What is your phone number?"
    if "today" in t or "tomorrow" in t:
        return "Sure. Which date would you prefer? (YYYY-MM-DD)"
    return "Sure. To continue the booking, may I have your phone number?"


def _strip_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove any system messages; agent_core injects its own system prompt."""
    return [m for m in messages if m.get("role") in ("user", "assistant")]

def _append_if_new(history: List[Dict[str, str]], role: str, content: str) -> None:
    content = (content or "").strip()
    if not content:
        return
    if history and history[-1].get("role") == role and (history[-1].get("content") or "").strip() == content:
        return
    history.append({"role": role, "content": content})

def _trim_history(session: SessionState) -> None:
    if session.max_history > 0 and len(session.history) > session.max_history:
        session.history = session.history[-session.max_history:]


def _pick_booking_tool(allowed_tools: List[str]) -> Optional[str]:
    return None


def _requires_consent(tool_name: str) -> bool:
    """Only workflow / write tools should require explicit user confirmation."""
    return False


def _looks_like_booking_offer(text: str) -> bool:
    """
    Detect when the assistant is OFFERING to book (not when it's already booking).
    Keep this conservative so we don't set pending_action for random messages.
    """
    t = (text or "").lower()
    offer_signals = [
        "i can book", "i can help you book", "i can schedule",
        "would you like me to book", "do you want me to book",
        "should i book", "want me to book", "shall i book",
        "i can arrange an appointment", "i can book an appointment",
        "i can schedule an appointment",
    ]
    return any(s in t for s in offer_signals)


def coerce_to_agent_json(raw: str):
    """
    JSON salvage with an 'invalid' type so we can force retry.
    Returns: (obj, salvaged)
      - salvaged=False means raw was valid JSON dict
      - salvaged=True means we extracted/failed and should be careful
    """
    raw = (raw or "").strip()

    # 1) direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, False
    except Exception:
        pass

    # 2) try to extract the first JSON object in text
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj, True
        except Exception:
            pass

    # 3) invalid (force retry)
    return {"type": "invalid", "content": raw}, True



def tool_message(name: str, result: Any) -> Dict[str, str]:
    """
    Put tool result back into conversation (as user message)
    so the model can continue after tool call.
    """
    payload = {"tool": name, "result": result}
    return {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}


def build_tools_block(tool_registry: dict, allowed_tools: List[str]) -> str:
    """
    Inject a strict list of allowed tools into system prompt
    so the model doesn't invent tool names (like hotel_search).
    """
    if not allowed_tools:
        return "TOOLS: none"

    lines = ["TOOLS (you may call ONLY these exact tool names):"]
    for name in allowed_tools:
        meta = tool_registry.get(name, {})
        desc = meta.get("description", "")
        lines.append(f"- {name}: {desc}")
    lines.append("Do NOT invent tool names. If no tool fits, respond with type=final.")
    return "\n".join(lines)


def _call_tool(name, args):
    fn = TOOL_REGISTRY[name]["fn"]
    print("TOOL FN:", fn)
    print("TOOL FILE:", fn.__code__.co_filename)
    print("TOOL VARS:", fn.__code__.co_varnames)
    return fn(**args)


def run_agent_once(
    client: OllamaClient,
    base_prompt: str,
    role_prompt: str,
    messages: List[dict],
    session: SessionState,
    model: str,
    max_tool_loops: int = 6,
    use_router: bool = True,
    use_base: bool = True,
) -> str:

    """
    One user turn -> assistant response.
    Uses router to:
      - force hotel_book on booking intent
      - enforce tool lock
      - restrict allowed tools based on role prompt
    """
    # --- Normalize incoming messages (CLI may include a system message; we don't want that) ---
    clean_messages = _strip_system(list(messages))

    # The last message must be the current user turn
    if not clean_messages or clean_messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user message")

    user_text = clean_messages[-1].get("content", "") or ""

    # --- Seed session history once ---
    # If session.history is empty, initialize it from provided messages (user/assistant only),
    # excluding the current user turn.
    if not session.history:
        session.history = clean_messages[:-1]

    # Add latest user turn to session memory
    _append_if_new(session.history, "user", user_text)
    _trim_history(session)

    # ROUTE DECISION
    if use_router:
        decision = route_turn(role_prompt=role_prompt, user_text=user_text, session=session)
        allowed_tools = decision.allowed_tools
        print("[ROUTER]", decision.role_name, decision.allowed_tools, "forced=", decision.forced_tool)
    else:
        # Router disabled -> no tools, no forced tool, single-pass chat
        decision = type("Decision", (), {"role_name": "general", "allowed_tools": [], "forced_tool": None})()
        allowed_tools = []
        max_tool_loops = 2
        print("[ROUTER] disabled general [] forced=None")

    print("[ROLE_PROMPT_HEAD]", role_prompt[:200].replace("\n", "\\n"))

    if session.active_tool:
        session.tool_allowed = True


    # If router forces a tool, bypass LLM completely
    if use_router and decision.forced_tool:
        tool_name = decision.forced_tool

        # SEATBELT: router must explicitly allow tool usage
        if not getattr(session, "tool_allowed", False):
            booking_tool = _pick_booking_tool(allowed_tools)
            assistant_text = (
                "I can help with that. If you want, I can proceed with the booking now. "
                "Do you want me to book it now?"
            )
            if booking_tool:
                session.pending_action = {"action": "book_appointment", "tool": booking_tool}
            _append_if_new(session.history, "assistant", assistant_text)
            _trim_history(session)
            return assistant_text

        if tool_name not in TOOL_REGISTRY:
            # Safety: clear lock if a stale tool name is present
            session.active_tool = None
            return "I can't run that action. Please rephrase your request."

        if allowed_tools and tool_name not in allowed_tools:
            session.active_tool = None
            return f"Tool '{tool_name}' is not allowed for this role."

        result = _call_tool(tool_name, {"text": user_text})
        done = bool(isinstance(result, dict) and result.get("done") is True)

        session.active_tool = None if done else tool_name
        session.pending_action = None

        if isinstance(result, dict) and "assistant" in result:
            assistant_text = str(result["assistant"])
        else:
            assistant_text = json.dumps(result, ensure_ascii=False)

        _append_if_new(session.history, "assistant", assistant_text)
        _trim_history(session)
        return assistant_text


    # Normal LLM path (with tool loop)
    # Build system prompt (lighter when router/base disabled)
    system_parts: List[str] = []

    if use_base and base_prompt.strip():
        system_parts.append(base_prompt.strip())

    if role_prompt.strip():
        system_parts.append(role_prompt.strip())

    if use_router:
        tools_block = build_tools_block(TOOL_REGISTRY, allowed_tools)
        system_parts.append(tools_block)

        schema_block = """
    OUTPUT FORMAT (MANDATORY):
    Return EXACTLY one JSON object and nothing else.

    Valid outputs only:
    {"type":"final","content":"..."}
    {"type":"redirect","content":"..."}

    If you output anything else, it is invalid.
    """.strip()

        if allowed_tools:
            schema_block = """
    OUTPUT FORMAT (MANDATORY):
    Return EXACTLY one JSON object and nothing else.

    Valid outputs only:
    {"type":"final","content":"..."}
    {"type":"tool_call","tool_name":"...","arguments":{}}
    {"type":"redirect","content":"..."}

    If you output anything else, it is invalid.
    """.strip()

        system_parts.append(schema_block)
    else:
        # No-router mode: keep schema minimal (faster, less token overhead)
        system_parts.append(
            'OUTPUT FORMAT (MANDATORY): Return EXACTLY one JSON object and nothing else.\n'
            'Valid output only:\n'
            '{"type":"final","content":"..."}\n'
            'If you output anything else, it is invalid.'
        )

    system_prompt = "\n\n".join([p for p in system_parts if p])




    convo: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}] + list(session.history)

    invalid_count = 0

    for _ in range(max_tool_loops):
        raw = client.chat(
            model=model,
            messages=convo,
            options={
                "num_ctx": 4096,       # keep same context you already see in `ollama ps`
                "num_predict": 80,    # cap output (BIG speed win)
                "num_thread": 14,      # i5-12500H (16 threads) -> leave 2 free
                "temperature": 0.4,
                "top_p": 0.9,
            },
        )

        agent, salvaged = coerce_to_agent_json(raw)

        # If the model did not output valid JSON, force it to retry in JSON
        if agent.get("type") == "invalid":
            invalid_count += 1

            # After 2 failures, stop looping and just return the raw model text.
            # This prevents "tool loop" for normal chat questions.
            if invalid_count >= 2:
                assistant_text = (raw or "").strip()

                # If the model accidentally returned a JSON-looking object, try to extract content
                # so we don't speak raw braces to the user.
                if assistant_text.startswith("{"):
                    try:
                        obj = json.loads(assistant_text)
                        if isinstance(obj, dict) and isinstance(obj.get("content"), str):
                            assistant_text = obj["content"].strip()
                    except Exception:
                        pass

                if not assistant_text:
                    assistant_text = "Sorry, I didn’t catch that. What would you like to do?"
                _append_if_new(session.history, "assistant", assistant_text)
                _trim_history(session)
                return assistant_text


            convo.append({"role": "assistant", "content": raw})
            convo.append({
                "role": "user",
                "content": (
                    "Reply using EXACTLY one JSON object, nothing else.\n"
                    "Format:\n"
                    "{\"type\":\"final\",\"content\":\"...\"}"
                ),
            })
            continue

        msg_type = agent.get("type", "final")

        if msg_type in ("redirect", "final"):
            assistant_text = str(agent.get("content", ""))

            # SAFETY NET: prevent early/invalid confirmations for clinic booking
            if "Confirm:" in assistant_text:
                # Decide if we have enough real user-provided info to allow confirmation.
                user_name = _extract_name_from_user_history(session.history)
                has_date = _has_any_date_in_history(session.history)

                if not user_name:
                    assistant_text = "Thanks. What is the patient name for the booking?"
                elif not has_date:
                    assistant_text = "Thanks. Which date would you prefer? (YYYY-MM-DD)"
                elif _is_placeholder_or_fake_booking(assistant_text):
                    assistant_text = _rewrite_early_confirm(assistant_text)

            # If assistant is OFFERING booking, set pending_action (system memory)
            booking_tool = _pick_booking_tool(allowed_tools)
            if booking_tool and _looks_like_booking_offer(assistant_text):
                session.pending_action = {"action": "book_appointment", "tool": booking_tool}

            _append_if_new(session.history, "assistant", assistant_text)
            _trim_history(session)
            return assistant_text

        if msg_type != "tool_call":
            assistant_text = str(agent.get("content", raw))
            _append_if_new(session.history, "assistant", assistant_text)
            _trim_history(session)
            return assistant_text

        tool_name = agent.get("tool_name") or agent.get("tool") or ""
        args = agent.get("arguments") or agent.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        # HARD BLOCK: unknown / invented tools -> force model to retry with allowed tools
        if tool_name not in TOOL_REGISTRY:
            convo.append({"role": "assistant", "content": raw})
            convo.append({
                "role": "user",
                "content": (
                    f"Invalid tool name '{tool_name}'. You may ONLY call these tools: "
                    f"{', '.join(allowed_tools) if allowed_tools else 'none'}. "
                    "Respond with one JSON object. If no tool fits, use type=final."
                ),
            })
            continue

        # HARD BLOCK: if NO tools are allowed, reject any tool_call
        if not allowed_tools:
            convo.append({"role": "assistant", "content": raw})
            convo.append({
                "role": "user",
                "content": (
                    "No tools are allowed for this request. "
                    "Respond with ONE JSON object of type=final."
                ),
            })
            continue

        # HARD BLOCK: tool not in allowlist
        if tool_name not in allowed_tools:
            convo.append({"role": "assistant", "content": raw})
            convo.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' is not allowed. Allowed tools: {', '.join(allowed_tools)}. "
                    "Respond with one JSON object. If no tool fits, use type=final."
                ),
            })
            continue

        # SEATBELT: only require confirmation for write tools (e.g., hotel_book).
        if _requires_consent(tool_name) and not getattr(session, "tool_allowed", False):
            assistant_text = "I can proceed with that booking now. Do you want me to book it now?"
            session.pending_action = {"action": "confirm_tool", "tool": tool_name}
            _append_if_new(session.history, "assistant", assistant_text)
            _trim_history(session)
            return assistant_text

        # Call tool
        result = _call_tool(tool_name, args)

        done = bool(isinstance(result, dict) and result.get("done") is True)
        session.active_tool = None if done else tool_name
        session.pending_action = None

        # Append tool result to conversation and continue loop
        convo.append({"role": "assistant", "content": raw})
        convo.append(tool_message(tool_name, result))

        # If tool wants to directly answer without LLM, we can return tool response here.
        if isinstance(result, dict) and "assistant" in result:
            assistant_text = str(result["assistant"])
            _append_if_new(session.history, "assistant", assistant_text)
            _trim_history(session)
            return assistant_text

    # If too many tool loops, return safe fallback
    return "Sorry—something went wrong generating a response. Please try again."
