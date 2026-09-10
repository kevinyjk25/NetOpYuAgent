"""Parse a small Python-shaped planning language. NEVER compile/eval/exec it.

Only read assignments, scalar/array-length equality and explicit ends lower to
the original plan blocks. This module has no transport, provider or file loader.
"""
from __future__ import annotations

import ast
import copy
import math
import re

from network_runtime.l0.structured_schema import pointer_parts, schema_location, schema_types

LANGUAGE = "inactive-read-plan/python-shaped-v3"
NAME = r"[a-zA-Z][a-zA-Z0-9_]{0,39}"


def _string(node, minimum=1):
    if not isinstance(node, ast.Constant) or type(node.value) is not str or not minimum <= len(node.value) <= 600:
        raise ValueError("planning syntax requires bounded literal strings")
    return node.value


def _call(node, name, counts, keywords=()):
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != name or len(node.args) not in counts
            or any(k.arg not in keywords for k in node.keywords)
            or len({k.arg for k in node.keywords}) != len(node.keywords)
            or any(isinstance(arg, ast.Starred) for arg in node.args)):
        raise ValueError("unsupported planning call; no arbitrary Python execution")
    return node.args


def _scalar(node):
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
        if not isinstance(node, ast.Constant) or type(node.value) not in {int, float}:
            raise ValueError("unary sign requires a numeric literal")
    if (not isinstance(node, ast.Constant) or type(node.value) not in {str, int, float, bool, type(None)}
            or (type(node.value) is float and not math.isfinite(node.value))):
        raise ValueError("predicate requires a finite scalar equality constant")
    return sign * node.value if type(node.value) in {int, float} else node.value


def parse(program, *, input_schema=None, catalog=None):
    if not isinstance(program, str) or not 16 <= len(program) <= 18000:
        raise ValueError("planning program size exceeded")
    try:
        module = ast.parse(program, mode="exec")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("invalid bounded planning syntax") from error
    if sum(1 for _ in ast.walk(module)) > 2000:
        raise ValueError("planning AST budget exceeded")
    count = 0
    declared = {"input", "read", "field", "length", "end"}
    tools = {t["name"]: t for t in catalog["tools"]} if catalog is not None else None
    schemas = {"input": input_schema} if input_schema is not None else {}
    definitions = []

    def reference(call, environment):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in {"field", "length"}:
            raise ValueError("only named field/length references are supported")
        name, pointer, source = _call(call, call.func.id, {3})
        if not isinstance(name, ast.Name) or name.id not in environment:
            raise ValueError("reference does not dominate this lexical use")
        expression, _ = environment[name.id]
        if expression["kind"] != "reference":
            raise ValueError("computed scalar cannot be dereferenced")
        relative = _string(pointer, 0)
        pointer_parts(relative)
        expression = {**expression, "pointer": expression["pointer"] + relative}
        if schemas:
            actual, _ = schema_location(schemas[expression["source"]], expression["pointer"])
            if call.func.id == "length" and schema_types(actual) != {"array"}:
                raise ValueError("length requires a declared array")
        expression["kind"] = "array_length" if call.func.id == "length" else "reference"
        return expression, {"block_id": _string(source)}

    def block(statements, inherited, depth=0):
        nonlocal count
        if depth > 8:
            raise ValueError("planning nesting exceeds eight levels")
        steps, open_path = [], True
        environment = copy.deepcopy(inherited)
        for statement in statements:
            count += 1
            if count > 32 or not open_path:
                raise ValueError("planning statement budget or unreachable continuation")
            if isinstance(statement, ast.Assign):
                if (len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name)
                        or not re.fullmatch(NAME, statement.targets[0].id)):
                    raise ValueError("read needs one non-input observation name")
                name = statement.targets[0].id
                if name in declared:
                    raise ValueError("names must be unique; no input/helper/observation shadowing")
                declared.add(name)
                if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id in {"field", "length"}:
                    expression, source = reference(statement.value, environment)
                    environment[name] = (expression, source)
                    definitions.append({"name": name, "expression": expression, "source": source,
                                        "programLine": statement.lineno, "kind": "lazy_reference_alias_not_observation"})
                    continue
                tool, source, reason = _call(statement.value, "read", {3}, ("operation_mode", "source_operation"))
                tool_name = _string(tool)
                if tools is not None:
                    if tool_name not in tools:
                        raise ValueError("tool is absent from the original catalog")
                    schemas[name] = tools[tool_name]["outputSchema"]
                environment[name] = ({"kind": "reference", "source": name, "pointer": ""}, {"block_id": _string(source)})
                optional = {({"operation_mode": "operationMode", "source_operation": "sourceOperation"})[k.arg]: _string(k.value)
                            for k in statement.value.keywords}
                steps.append({"kind": "read", "observation": name,
                    "tool": tool_name, "source": {"block_id": _string(source)}, "whyNeeded": _string(reason, 8), **optional})
            elif isinstance(statement, ast.If):
                test = statement.test
                if (not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq)
                        or len(test.comparators) != 1):
                    raise ValueError("only field/length equality is a planning predicate")
                if isinstance(test.left, ast.Name):
                    if test.left.id not in environment or environment[test.left.id][1] is None:
                        raise ValueError("named predicate requires a preceding field definition")
                    expression, source = copy.deepcopy(environment[test.left.id])
                else:
                    expression, source = reference(test.left, environment)
                comparator = test.comparators[0]
                if isinstance(comparator, ast.Call):
                    right, right_source = reference(comparator, environment)
                    if right_source != source:
                        raise ValueError("both predicate operands require the same predicate witness")
                elif isinstance(comparator, ast.Name) and comparator.id in environment:
                    right, right_source = copy.deepcopy(environment[comparator.id])
                    if right_source is None:
                        raise ValueError("predicate RHS requires an explicit scalar field")
                else:
                    right = _scalar(comparator)
                yes, yes_open = block(statement.body, environment, depth + 1)
                no, no_open = block(statement.orelse, environment, depth + 1)
                steps.append({"kind": "if_equal", "source": source, "left": expression,
                    "equals": right, "when_equal": yes, "otherwise": no})
                open_path = yes_open or no_open
            elif isinstance(statement, ast.Expr):
                args = _call(statement.value, "end", {3, 4})
                outcome, source, explanation = args[:3]
                outcome = _string(outcome)
                if outcome not in {"read_path_completed", "needs_l1", "unsupported"}:
                    raise ValueError("unknown planning terminal")
                duties = []
                if len(args) == 4:
                    if not isinstance(args[3], ast.List) or len(args[3].elts) > 16:
                        raise ValueError("bounded explicit remaining duties required")
                    for duty in args[3].elts:
                        if not isinstance(duty, ast.Tuple) or len(duty.elts) != 3:
                            raise ValueError("duty must be (source_block, condition, requirement)")
                        duties.append({"source": {"block_id": _string(duty.elts[0])},
                            "when": _string(duty.elts[1]), "requirement": _string(duty.elts[2], 8)})
                if bool(duties) != (outcome != "read_path_completed"):
                    raise ValueError("non-completed terminals need duties; completed terminals cannot owe work")
                if statement is not statements[-1]:
                    raise ValueError("unreachable statements after end")
                return {"steps": steps, "exit": {"kind": "end", "source": {"block_id": _string(source)},
                    "outcome": outcome, "explanation": _string(explanation, 8), "duties": duties}}, False
            else:
                raise ValueError("only read assignments, if/elif/else and end are supported; scripts are never executed")
        if depth == 0 and open_path:
            raise ValueError("all root paths require explicit end; missing success is never inferred")
        return {"steps": steps, "exit": {"kind": "continue" if open_path else "already_closed"}}, open_path

    result = block(module.body, {"input": ({"kind": "reference", "source": "input", "pointer": ""}, None)})[0]
    return {**result, "fieldBindings": definitions}
