"""Deterministic routing layer.

The router decides:
- Which tools are allowed for the current role prompt
- Whether to force-call a tool (intent trigger / tool lock)
"""

from .session import SessionState
from .router import route_turn, infer_role_from_prompt
