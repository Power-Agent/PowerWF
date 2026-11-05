"""Parse explicit free-text planning statements without synthesizing a plan.

This is an execution adapter, not a planner: no case, oracle, capability defaults,
or compiler template is read here. Ambiguous/unsupported text fails closed.
"""
from __future__ import annotations

import ast
import json
import re

from b2g_agent.contracts import ActionType, PlanAction


_HEADER = re.compile(
    r"^\s*(?:[-*]\s+|\#{1,6}\s+)?(?:step\s+)?(?:\d+[.):]\s*)?"
    r"(?:action\s*[:=]?\s+)?`?([a-z][a-z0-9_]*)`?(.*)$", re.I)
_LABEL = re.compile(r"^(depends_on|depends\s+on|dependencies|inputs?|parameters|outputs?|produces)\b\s*[:=]?\s*(.*)$", re.I | re.S)
_TOOL = re.compile(r"^(?:using|with|tool\s*[:=])\s*`?([a-z][a-z0-9_]*)`?\s*[.]?$", re.I)


def _parts(text: str, separators: str) -> list[str]:
    """Split prose separators, preserving nested/quoted literal values."""
    parts, stack, quote, escaped, start = [], [], None, False, 0
    for i, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "[{(":
            stack.append(char)
        elif char in "]})":
            if not stack or stack.pop() != {"]": "[", "}": "{", ")": "("}[char]:
                raise ValueError("unbalanced_literal")
        elif char in separators and not stack:
            if text[start:i].strip():
                parts.append(text[start:i].strip())
            start = i + 1
    if quote or stack:
        raise ValueError("unfinished_literal")
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_input_key:{key}")
        result[key] = value
    return result


def _literal(text):
    text = text.strip().strip("`")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError:
        node = ast.parse(text, mode="eval")
        for item in ast.walk(node):
            if isinstance(item, ast.Dict):
                keys = [ast.literal_eval(key) for key in item.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError("duplicate_input_key")
        value = ast.literal_eval(node)
    # Keep unsupported Python objects and non-finite values out of downstream
    # plan hashing and journal serialization; report them as parse failures.
    json.dumps(value, allow_nan=False)
    return value


def _names(text):
    text = text.strip().rstrip(".")
    if text.lower() in {"none", "no dependencies", "no outputs", "[]"}:
        return []
    if text.startswith("["):
        names = _literal(text)
        if not isinstance(names, list) or any(not isinstance(v, str) for v in names):
            raise ValueError("expected_name_list")
        return names
    tokens = re.split(r"\s*,\s*|\s+and\s+", text, flags=re.I)
    names = [t.strip().strip("`") for t in tokens]
    if not names or any(not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", v) for v in names):
        raise ValueError("ambiguous_name_list")
    return names


def _inputs(text):
    if text.strip().lower() in {"none", "no inputs"}:
        return {}
    if text.lstrip().startswith("{"):
        value = _literal(text)
        if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
            raise ValueError("expected_named_inputs")
        return value
    pairs = []
    for item in _parts(text, ","):
        match = re.fullmatch(r"([a-zA-Z][a-zA-Z0-9_]*)\s*[:=]\s*(.+)", item, re.S)
        if not match:
            raise ValueError("expected_explicit_input_value")
        pairs.append((match[1], _literal(match[2])))
    if not pairs:
        raise ValueError("missing_explicit_inputs")
    return _unique_object(pairs)


def parse_free_plan(text: str) -> tuple[list[PlanAction], list[str]]:
    """Accept prose action blocks with explicit tool, dependencies, inputs, outputs.

    Action headers, optional numbering/bullets, inline/paragraph clauses, singular
    labels, JSON/Python value literals and comma/and lists are supported. Repeated
    action names inside dependency clauses are references, not new actions.
    """
    known = {a.value for a in ActionType}
    blocks, errors, current = [], [], None
    for raw in text.splitlines():
        line = raw.strip().replace("**", "")
        if not line or line.startswith("```"):
            continue
        header = _HEADER.match(line)
        name = header[1].lower() if header else None
        if header and (name in known or re.match(r"\s+(using|with)\b", header[2], re.I)):
            current = [name, header[2].strip()]
            blocks.append(current)
        elif current is not None:
            current[1] += "\n" + line
        else:
            # Unrecognized narrative is not allowed to conceal a missing action.
            errors.append("free_plan_unparsed_text_before_action")
    if not blocks:
        return [], [*errors, "free_plan_no_recognized_actions"]
    counts = {name: sum(b[0] == name for b in blocks) for name, _ in blocks}
    identifiers = {name: f"fp{i:02d}_{name}" for i, (name, _) in enumerate(blocks, 1) if counts[name] == 1}
    actions = []
    for i, (name, body) in enumerate(blocks, 1):
        prefix = f"free_plan:{name}:"
        if name not in known:
            errors.append(prefix + "unknown_action")
            continue
        if counts[name] != 1:
            errors.append(prefix + "duplicate_action")
        fields, tool = {}, None
        try:
            for part in _parts(body, ";\n"):
                label = _LABEL.match(part)
                tool_match = _TOOL.fullmatch(part)
                if tool_match:
                    if tool is not None:
                        raise ValueError("duplicate_tool")
                    tool = tool_match[1].lower()
                elif label:
                    key = label[1].lower()
                    key = "dependencies" if key.startswith("depend") else "inputs" if key.startswith(("input", "parameter")) else "outputs"
                    if key in fields:
                        raise ValueError("duplicate_" + key)
                    fields[key] = _inputs(label[2]) if key == "inputs" else _names(label[2])
                else:
                    raise ValueError("unparsed_clause:" + part[:120])
            for key in ("dependencies", "inputs", "outputs"):
                if key not in fields:
                    errors.append(prefix + "missing_" + key)
            if tool is None:
                errors.append(prefix + "missing_tool")
                continue
            dependencies = [identifiers.get(d, "unresolved_" + d) for d in fields.get("dependencies", [])]
            actions.append(PlanAction(action_id=f"fp{i:02d}_{name}", action_type=ActionType(name),
                tool=tool, depends_on=dependencies, inputs=fields.get("inputs", {}), outputs=fields.get("outputs", [])))
        except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
            errors.append(prefix + str(exc)[:240])
    return actions, errors
