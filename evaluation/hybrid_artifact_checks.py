"""Compatibility import for the shared, non-executing artifact checker."""
import sys
from skill_authoring import artifact_checks as _shared

sys.modules[__name__] = _shared
