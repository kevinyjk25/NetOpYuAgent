"""Compatibility alias to the shared implementation; no second compiler."""
import sys
from skill_authoring import task_context as _shared
sys.modules[__name__] = _shared
