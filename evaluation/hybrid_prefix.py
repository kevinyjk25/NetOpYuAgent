"""Compatibility alias to the shared implementation; no second compiler."""
import sys
from skill_authoring import prefix as _shared
sys.modules[__name__] = _shared
