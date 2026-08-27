"""DSH bridge public API without eager imports into the L1 runtime."""

from typing import Any

__all__ = ["build_manifest", "invoke_tool"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import bridge

        return getattr(bridge, name)
    raise AttributeError(name)
