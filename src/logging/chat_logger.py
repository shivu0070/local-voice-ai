from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ChatLogConfig:
    """Configuration for chat logging."""

    csv_path: Path


def new_session_id() -> str:
    """Create a short, unique session id suitable for filenames/CSV."""
    return uuid.uuid4().hex


def _utc_iso() -> str:
    # ISO-8601 with timezone, stable for sorting.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_header_if_needed(path: Path, fieldnames: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    _ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def log_turn(
    cfg: ChatLogConfig,
    *,
    session_id: str,
    role: str,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a single turn to an append-only CSV chat log.

    Columns are intentionally simple and stable:
      - ts_utc: ISO timestamp in UTC
      - session_id: unique per run/session
      - role: user | assistant | tool | system
      - text: the message content
      - meta_json: optional json blob (model, tool_name, etc.)
    """

    fieldnames = ["ts_utc", "session_id", "role", "text", "meta_json"]
    _write_header_if_needed(cfg.csv_path, fieldnames)

    row = {
        "ts_utc": _utc_iso(),
        "session_id": session_id,
        "role": role,
        "text": text,
        "meta_json": json.dumps(meta or {}, ensure_ascii=False),
    }

    with cfg.csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)
