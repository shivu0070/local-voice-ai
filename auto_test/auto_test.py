"""
Auto-test runner: text-in → text-out, saves results to auto_test/results.csv

Mirrors the Brain logic from voice_server.py exactly (same env vars, same
Brain/OllamaClient/SessionState/run_agent_once wiring) but skips STT/TTS.

Usage:
    python auto_test/auto_test.py

Input:  auto_test/questions.txt   (one question per line; blank lines skipped)
Output: auto_test/results.csv     (question, answer, latency_ms, timestamp)
"""

from __future__ import annotations

import csv
import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup – make sure `src/` is importable regardless of cwd.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Load .env from project root (same as voice_server.py would see).
# ---------------------------------------------------------------------------
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Replicate DEFAULT_HF_CACHE setup from voice_server.py
# ---------------------------------------------------------------------------
DEFAULT_HF_CACHE = ROOT / "models" / "hub"
DEFAULT_HF_CACHE.mkdir(parents=True, exist_ok=True)
if not os.getenv("HUGGINGFACE_HUB_CACHE"):
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(DEFAULT_HF_CACHE)

# ---------------------------------------------------------------------------
# Imports from src/ (mirrors voice_server.py try/except import pattern)
# ---------------------------------------------------------------------------
from src.agent_core import run_agent_once
from src.logging.chat_logger import ChatLogConfig, log_turn, new_session_id
from src.ollama_client import OllamaClient
from src.router.session import SessionState

# ---------------------------------------------------------------------------
# Brain – identical to voice_server.py Brain class
# ---------------------------------------------------------------------------
class Brain:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        keep_alive: int = -1,
        history_max: int = 10,
        warmup: bool = True,
        mode: str = "chat",
    ) -> None:
        self.client = OllamaClient()
        self.model = model
        self.keep_alive = keep_alive
        self.session = SessionState(max_history=history_max)
        self.base_prompt = (ROOT / "configs" / "agents" / "base.txt").read_text(encoding="utf-8")
        self.role_prompt = (ROOT / "configs" / "agents" / "role.txt").read_text(encoding="utf-8")
        self.mode = (mode or "chat").strip().lower()
        self.messages: List[dict] = []
        self._msg_lock = threading.Lock()

        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        self.session_id = new_session_id()
        (data_dir / "last_session.txt").write_text(self.session_id, encoding="utf-8")
        self.log_cfg = ChatLogConfig(csv_path=data_dir / "chat_log.csv")

        if warmup:
            try:
                self.client.chat(
                    self.model,
                    [{"role": "system", "content": "warmup"}],
                    keep_alive=self.keep_alive,
                )
            except Exception:
                pass

    def _trim_messages(self) -> None:
        try:
            max_hist = int(getattr(self.session, "max_history", 0) or 0)
        except Exception:
            max_hist = 0
        if max_hist > 0 and len(self.messages) > max_hist:
            self.messages = self.messages[-max_hist:]

    def _chat_system_prompt(self) -> str:
        return (self.role_prompt or "").strip()

    def _chat_options(self) -> dict:
        def _get_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return int(default)

        def _get_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return float(default)

        return {
            "temperature": _get_float("CHAT_TEMPERATURE", 0.2),
            "num_predict": _get_int("CHAT_NUM_PREDICT", 80),
            "num_ctx": _get_int("CHAT_NUM_CTX", 1024),
        }

    def chat(self, user_text: str) -> str:
        """
        Chat mode: stream tokens, collect full answer, return it.
        Mirrors the token-streaming logic in voice_server._run_turn (chat branch).
        """
        clean = (user_text or "").strip()
        if not clean:
            return ""

        # KB / RAG (same as voice_server)
        kb_answer = ""
        try:
            try:
                from rag_knowledge import maybe_compact_company_answer
            except Exception:
                from src.rag_knowledge import maybe_compact_company_answer
            kb_answer = maybe_compact_company_answer(clean) or ""
        except Exception:
            kb_answer = ""

        rag_prompt = ""
        if not kb_answer:
            try:
                try:
                    from rag_knowledge import build_company_context_block
                except Exception:
                    from src.rag_knowledge import build_company_context_block
                rag_prompt = build_company_context_block(clean) or ""
            except Exception:
                rag_prompt = ""

        with self._msg_lock:
            self.messages.append({"role": "user", "content": clean})
            self._trim_messages()
            log_turn(
                self.log_cfg,
                session_id=self.session_id,
                role="user",
                text=clean,
                meta={"client": "auto_test"},
            )

        if kb_answer:
            with self._msg_lock:
                self.messages.append({"role": "assistant", "content": kb_answer})
                self._trim_messages()
                log_turn(
                    self.log_cfg,
                    session_id=self.session_id,
                    role="assistant",
                    text=kb_answer,
                    meta={"client": "auto_test", "source": "kb"},
                )
            return kb_answer

        sys_prompt = self._chat_system_prompt()
        with self._msg_lock:
            system_msgs = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
            if rag_prompt:
                system_msgs.append({"role": "system", "content": rag_prompt})
            send_messages = system_msgs + list(self.messages)

        # Normalize noinfo (same as voice_server)
        try:
            try:
                from rag_knowledge import COMPANY_NOINFO_TEXT, normalize_company_noinfo
            except Exception:
                from src.rag_knowledge import COMPANY_NOINFO_TEXT, normalize_company_noinfo
        except Exception:
            COMPANY_NOINFO_TEXT = "I can't find any information regarding this."

            def normalize_company_noinfo(s: str) -> str:
                return (s or "").strip()

        token_stream = self.client.chat_stream(
            self.model,
            send_messages,
            options=self._chat_options(),
            keep_alive=self.keep_alive,
        )

        answer_parts: List[str] = []
        forced_noinfo = False
        for tok in token_stream:
            if forced_noinfo:
                continue
            answer_parts.append(tok)
            combined = "".join(answer_parts)
            norm = normalize_company_noinfo(combined)
            if norm == COMPANY_NOINFO_TEXT:
                forced_noinfo = True
                try:
                    self.client.cancel_active_stream()
                except Exception:
                    pass
                answer_parts = [norm]
                break

        answer = "".join(answer_parts).strip()
        try:
            answer = normalize_company_noinfo(answer) or answer
        except Exception:
            pass

        with self._msg_lock:
            self.messages.append({"role": "assistant", "content": answer})
            self._trim_messages()
            log_turn(
                self.log_cfg,
                session_id=self.session_id,
                role="assistant",
                text=answer,
                meta={"client": "auto_test", "model": self.model, "mode": "chat"},
            )
        return answer

    def router(self, user_text: str) -> str:
        """Router/tools mode – identical to voice_server brain.__call__."""
        clean = (user_text or "").strip()
        if not clean:
            return ""

        self.messages.append({"role": "user", "content": clean})
        log_turn(
            self.log_cfg,
            session_id=self.session_id,
            role="user",
            text=clean,
            meta={"client": "auto_test"},
        )

        answer = run_agent_once(
            client=self.client,
            base_prompt=self.base_prompt,
            role_prompt=self.role_prompt,
            messages=self.messages,
            session=self.session,
            model=self.model,
            use_router=True,
            use_base=True,
        )

        answer = (answer or "").strip()
        if not answer:
            return ""

        self.messages.append({"role": "assistant", "content": answer})
        log_turn(
            self.log_cfg,
            session_id=self.session_id,
            role="assistant",
            text=answer,
            meta={"client": "auto_test", "model": self.model},
        )
        return answer

    def ask(self, user_text: str) -> str:
        if self.mode == "chat":
            return self.chat(user_text)
        return self.router(user_text)


# ---------------------------------------------------------------------------
# Build brain (mirrors voice_server._build_brain)
# ---------------------------------------------------------------------------
def _build_brain() -> Brain:
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    keep_alive = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
    history_max = int(os.getenv("HISTORY_MAX", "10"))
    warmup = os.getenv("OLLAMA_WARMUP", "1").lower() not in {"0", "false", "no"}
    mode = os.getenv("BRAIN_MODE", "chat").strip().lower()
    if mode not in {"chat", "router"}:
        mode = "chat"
    return Brain(model=model, keep_alive=keep_alive, history_max=history_max, warmup=warmup, mode=mode)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
AUTO_TEST_DIR = ROOT / "auto_test"
QUESTIONS_FILE = AUTO_TEST_DIR / "questions.txt"
RESULTS_FILE = AUTO_TEST_DIR / "results.csv"

CSV_FIELDS = ["index", "question", "answer", "latency_ms", "mode", "model", "timestamp"]


def load_questions(path: Path) -> List[str]:
    if not path.exists():
        print(f"[auto_test] questions file not found: {path}")
        print("[auto_test] Creating a sample questions.txt for you ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "What services do you offer?\n"
            "How can I contact you?\n"
            "Tell me about your pricing.\n",
            encoding="utf-8",
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def run_tests() -> None:
    questions = load_questions(QUESTIONS_FILE)
    if not questions:
        print("[auto_test] No questions found. Add questions to auto_test/questions.txt")
        return

    print(f"[auto_test] Loaded {len(questions)} question(s) from {QUESTIONS_FILE}")
    print("[auto_test] Building brain (warmup) ...")
    brain = _build_brain()
    print(f"[auto_test] Brain ready. model={brain.model} mode={brain.mode}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for idx, question in enumerate(questions, start=1):
            print(f"\n[{idx}/{len(questions)}] Q: {question}")
            t0 = time.monotonic()
            try:
                answer = brain.ask(question)
            except Exception as exc:
                answer = f"ERROR: {exc}"
            latency_ms = round((time.monotonic() - t0) * 1000)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")

            print(f"         A: {answer}")
            print(f"         latency={latency_ms}ms")

            writer.writerow(
                {
                    "index": idx,
                    "question": question,
                    "answer": answer,
                    "latency_ms": latency_ms,
                    "mode": brain.mode,
                    "model": brain.model,
                    "timestamp": ts,
                }
            )
            csvfile.flush()

    print(f"\n[auto_test] Done. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_tests()
