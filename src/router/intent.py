from __future__ import annotations

import re
from typing import Optional


_YES_RE = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|please|do it|go ahead|confirm|book it)\b", re.I)
_NO_RE = re.compile(r"\b(no|nope|not now|later|don't|do not|cancel|stop)\b", re.I)

_EXPLICIT_BOOK_RE = re.compile(
    r"\b(book|booking|schedule|appointment|reserve|reservation|make an appointment|fix an appointment)\b",
    re.I,
)


def looks_like_yes(text: str) -> bool:
    return bool(_YES_RE.search(text or ""))


def looks_like_no(text: str) -> bool:
    return bool(_NO_RE.search(text or ""))


def looks_like_explicit_booking_request(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(_EXPLICIT_BOOK_RE.search(t))


def llm_classify_booking_intent(
    user_text: str,
    last_assistant_text: str = "",
    pending_action: Optional[dict] = None,
) -> str:
    """
    Returns: BOOK_NOW or NOT_BOOKING

    Safe rules first, then optional Ollama fallback for ambiguous confirmations.
    """
    if looks_like_explicit_booking_request(user_text):
        return "BOOK_NOW"

    if pending_action and looks_like_yes(user_text):
        return "BOOK_NOW"

    if looks_like_no(user_text):
        return "NOT_BOOKING"

    # LLM fallback for ambiguous cases (optional but useful)
    try:
        from ..ollama_client import OllamaClient

        model = "qwen2.5:7b-instruct"

        prompt = (
            "You are a strict intent classifier.\n"
            "Decide if the user is committing to booking an appointment.\n\n"
            "Return ONLY one token:\n"
            "BOOK_NOW or NOT_BOOKING\n\n"
            f"Pending action: {pending_action}\n"
            f"Last assistant message: {last_assistant_text}\n"
            f"User message: {user_text}\n"
        )
        client = OllamaClient()
        out = client.chat(model=model, messages=[{"role": "user", "content": prompt}]).strip().upper()
        return "BOOK_NOW" if "BOOK" in out else "NOT_BOOKING"
    except Exception:
        return "NOT_BOOKING"
