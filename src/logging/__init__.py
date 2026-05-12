"""Lightweight CSV chat logging utilities.

We keep this module dependency-free and very small so it can be used from
both text and voice CLIs without changing the agent core.
"""

# Re-export stdlib logging.NullHandler so external libs (requests, etc.)
# don't break when our package name shadows the stdlib `logging` module.
import logging as _stdlib_logging  # noqa: E402

NullHandler = _stdlib_logging.NullHandler

__all__ = ["NullHandler"]
