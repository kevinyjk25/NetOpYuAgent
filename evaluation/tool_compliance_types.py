"""
evaluation/tool_compliance_types.py
-----------------------------------
Dataclasses for tool-call compliance evaluation — a separate dimension
from the retrieval bench. Retrieval asks "did the system surface
the right SKILL/TOOL to consider?"; tool-compliance asks "given a query
where we already know the right tool, did the LLM emit it correctly?"

Compliance has three sub-metrics, scored per case:
  1. parse_ok   — could the directive parser extract a [TOOL:...] call?
                  (i.e. the LLM produced valid syntax at all)
  2. name_ok    — did the parsed tool name match the expected one?
  3. args_ok    — did all REQUIRED arg keys appear, with correct types,
                  and did the LLM put values into the right slots?

These compose: a case fails name_ok if the tool name is wrong even if
parse_ok succeeded. args_ok requires name_ok (you can't validate args
against a schema without knowing the schema).

The point of separating them: when a metric regresses, the breakdown
tells you what kind of degradation it is — protocol parse failure
(model can't follow [TOOL:...] format), tool selection failure
(model picked the wrong tool), or arg-filling failure (model picked
right tool but botched the args). The latter is what native-tools
mode is supposed to make structurally impossible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Any, Optional


@dataclass
class ToolCallCase:
    """One labeled tool-compliance example.

    Fields:
      query              — what the user typed (any language)
      expected_tool      — the canonical tool name the LLM SHOULD pick
      expected_args      — dict of {arg_name: value | None}
                           value=None means "key must be present, any value is fine"
                           (useful when the right value is query-dependent and
                           hard to pin down — e.g. a free-form reason field)
      required_arg_names — subset of expected_args.keys() that MUST be present
                           (others are "if-emitted-must-be-correct" but optional)
      forbidden_args     — keys that must NOT appear (model hallucinating extra fields)
      acceptable_tools   — additional tool names also acceptable
                           (e.g. {"get_device_status", "query_interface_metrics"}
                           for a query like "what's wrong with ap-01?")
      notes              — free-form, not consumed
      language           — "en" | "zh" | "mixed"
      tags               — ["destructive", "multi-target", "ambiguous", ...]
    """
    query:              str
    expected_tool:      str
    expected_args:      dict[str, Any]    = field(default_factory=dict)
    required_arg_names: list[str]         = field(default_factory=list)
    forbidden_args:     list[str]         = field(default_factory=list)
    acceptable_tools:   list[str]         = field(default_factory=list)
    notes:              str               = ""
    language:           str               = "en"
    tags:               list[str]         = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("ToolCallCase: query must be non-empty")
        if not self.expected_tool.strip():
            raise ValueError("ToolCallCase: expected_tool must be non-empty")
        # If required_arg_names is empty, default it to every key in
        # expected_args (caller probably meant "all of these are required").
        # Opting out is explicit: pass required_arg_names=[].
        if not self.required_arg_names and self.expected_args:
            self.required_arg_names = list(self.expected_args.keys())


@dataclass
class ToolCallResult:
    """Outcome of running one ToolCallCase.

    parse_ok / name_ok / args_ok compose in that order. If parse_ok is
    False, the others are False by definition (we never inspected args).
    """
    case:           ToolCallCase
    raw_response:   str
    parsed_name:    Optional[str]        = None
    parsed_args:    dict[str, Any]       = field(default_factory=dict)
    parse_ok:       bool                 = False
    name_ok:        bool                 = False
    args_ok:        bool                 = False
    missing_args:   list[str]            = field(default_factory=list)
    wrong_value_args: list[str]          = field(default_factory=list)
    forbidden_present: list[str]         = field(default_factory=list)
    elapsed_ms:     float                = 0.0
    error:          Optional[str]        = None    # exception during the call

    @property
    def fully_compliant(self) -> bool:
        return self.parse_ok and self.name_ok and self.args_ok


@dataclass
class ToolComplianceReport:
    """Aggregate report from ToolComplianceBench.run().

    All counts are scored against `total`. The breakdown lets you ask
    "did parse_ok hold across both modes but args_ok flipped from 60%
    to 95% when I switched to native tools?"
    """
    backend_name:    str           # which LLM engine + mode (e.g. "qwen3.5:27b/native")
    total:           int
    parse_ok_count:  int
    name_ok_count:   int
    args_ok_count:   int
    fully_compliant_count: int
    avg_elapsed_ms:  float
    errored_count:   int           # cases where LLM call raised
    cases:           list[ToolCallResult] = field(default_factory=list)

    @property
    def parse_rate(self)   -> float: return self.parse_ok_count / self.total if self.total else 0.0
    @property
    def name_rate(self)    -> float: return self.name_ok_count / self.total if self.total else 0.0
    @property
    def args_rate(self)    -> float: return self.args_ok_count / self.total if self.total else 0.0
    @property
    def compliance(self)   -> float:
        return self.fully_compliant_count / self.total if self.total else 0.0
    @property
    def error_rate(self)   -> float:
        return self.errored_count / self.total if self.total else 0.0
