"""
schema/types.py
---------------
Core types for the unified tool-argument schema.

Design choices:
  - ArgField is the atomic unit (one named argument)
  - ArgSchema is the full tool args schema (a struct of fields)
  - FieldType is a string enum mirroring JSON Schema 'type' (object, array,
    string, integer, number, boolean, null) — direct compatibility with MCP/OpenAPI
  - All fields default to permissive/optional so half-specified schemas still work
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Any, Optional


# ---------------------------------------------------------------------------
# JSON-Schema-aligned types
# ---------------------------------------------------------------------------
# We use plain strings rather than Enum so MCP/OpenAPI imports map directly
# without translation tables.

FieldType = str   # one of: object | array | string | integer | number | boolean | null | any

VALID_TYPES = {
    "object", "array", "string", "integer", "number", "boolean", "null", "any",
}


@dataclass
class ArgField:
    """One named argument of a tool.

    Mirrors a single JSON Schema property. Most fields are optional so that
    importers from sparse formats (e.g. dict-style metadata with only descriptions)
    still produce useful schemas.

    Coercion rules:
      - If the LLM passes the wrong type but it's losslessly convertible
        (e.g. string "3" when integer expected), validate_and_coerce() will
        produce the right type and emit a warning, not an error.
      - If `items` is set on an array field, list elements are recursively
        validated/coerced.
      - If `properties` is set on an object field, dict keys are recursively
        validated/coerced.

    Permissiveness:
      - additional_properties=True (default) means extra keys are kept untouched.
        Set False to strip unknown keys (useful for strict pragmatic tools).
    """
    name:          str
    type:          FieldType                        = "any"
    description:   str                              = ""
    required:      bool                             = False
    default:       Any                              = None
    enum:          Optional[list[Any]]              = None      # value must be one of these
    items:         Optional["ArgField"]             = None      # element schema for arrays
    properties:    Optional[dict[str, "ArgField"]]  = None      # field schema for objects
    additional_properties: bool                     = True      # for objects: keep unknown keys?
    examples:      list[Any]                        = field(default_factory=list)

    def __post_init__(self):
        if self.type not in VALID_TYPES:
            raise ValueError(f"ArgField {self.name!r}: type {self.type!r} not in {VALID_TYPES}")


@dataclass
class ArgSchema:
    """Full argument schema for one tool.

    Conceptually a JSON Schema 'object' at the top level; fields are top-level
    properties. The required list mirrors JSON Schema's required[] array.
    """
    tool_name:     str
    fields:        dict[str, ArgField]   = field(default_factory=dict)
    description:   str                   = ""
    examples:      list[dict[str, Any]]  = field(default_factory=list)

    @property
    def required_names(self) -> list[str]:
        return [n for n, f in self.fields.items() if f.required]

    def get(self, name: str) -> Optional[ArgField]:
        return self.fields.get(name)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CoerceResult:
    """Outcome of validate_and_coerce.

    Three terminal states:
      - ok=True, no errors                  : args are perfectly shaped
      - ok=True, warnings non-empty         : args were silently coerced (e.g. "3" → 3)
      - ok=False, errors non-empty          : args could not be coerced; tool MUST NOT run
    """
    ok:        bool
    args:      dict[str, Any]
    errors:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)


@dataclass
class ValidationError(Exception):
    """Raised when an argument cannot be validated/coerced and the caller
    asked for raise-on-error behaviour. Most users prefer the soft
    CoerceResult path so they can compose error messages back to the LLM."""
    field:     str
    expected:  str
    got:       str
    detail:    str = ""

    def __str__(self) -> str:
        msg = f"field {self.field!r}: expected {self.expected}, got {self.got}"
        if self.detail:
            msg += f" — {self.detail}"
        return msg
