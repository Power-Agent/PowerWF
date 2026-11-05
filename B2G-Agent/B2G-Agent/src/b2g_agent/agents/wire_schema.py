"""Lossless transport for dynamic contracts in strict Structured Outputs.

OpenAI disallows untyped values and open objects. On the wire, mappings use
key/value entries and arbitrary JSON values use a recursive typed union. Internal
contracts and simulator inputs keep their original types.
"""

from __future__ import annotations

from typing import Any


def strict_wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    uses_json_value = False
    def visit(node: dict[str, Any]) -> dict[str, Any]:
        nonlocal uses_json_value
        if not any(key in node for key in ("type", "$ref", "anyOf", "allOf")):
            uses_json_value = True
            return {"$ref": "#/$defs/B2GJsonValue"}
        result = {key: value for key, value in node.items() if key != "default"}
        if "$defs" in node:
            result["$defs"] = {key: visit(value) for key, value in node["$defs"].items()}
        if node.get("type") == "object":
            extra = node.get("additionalProperties", False)
            if isinstance(extra, dict) or extra is True:
                return {
                    "type": "array",
                    "description": "Dictionary encoded as unique key/value entries; use [] for an empty dictionary.",
                    "items": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}, "value": visit(extra if isinstance(extra, dict) else {})},
                        "required": ["key", "value"],
                        "additionalProperties": False,
                    },
                }
            result["properties"] = {key: visit(value) for key, value in node.get("properties", {}).items()}
            result["required"] = list(result["properties"])
            result["additionalProperties"] = False
        if "items" in node:
            result["items"] = visit(node["items"])
        for union in ("anyOf", "allOf"):
            if union in node:
                result[union] = [visit(item) for item in node[union]]
        return result

    result = visit(schema)
    if uses_json_value:
        reference = {"$ref": "#/$defs/B2GJsonValue"}
        result.setdefault("$defs", {})["B2GJsonValue"] = {
            "anyOf": [
                {"type": "null"}, {"type": "boolean"},
                {"type": "number"}, {"type": "string"},
                {"type": "array", "items": reference},
                {
                    "type": "object", "additionalProperties": False,
                    "description": "An arbitrary JSON object represented by unique key/value entries.",
                    "properties": {"entries": {
                        "type": "array", "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {"key": {"type": "string"}, "value": reference},
                            "required": ["key", "value"],
                        },
                    }},
                    "required": ["entries"],
                },
            ],
        }
    return result


def decode_wire_value(value: Any, schema: dict[str, Any]) -> Any:
    def arbitrary(item: Any) -> Any:
        if isinstance(item, list):
            return [arbitrary(val) for val in item]
        if isinstance(item, dict):
            if set(item) != {"entries"}:
                raise ValueError("An arbitrary object must have an entries array")
            result = {}
            for entry in item["entries"]:
                if entry["key"] in result:
                    raise ValueError(f"Duplicate mapping key: {entry['key']}")
                result[entry["key"]] = arbitrary(entry["value"])
            return result
        return item

    def visit(item: Any, node: dict[str, Any]) -> Any:
        if "$ref" in node:
            target = schema
            for component in node["$ref"].removeprefix("#/").split("/"):
                target = target[component]
            return visit(item, target)
        if "anyOf" in node:
            if item is None and any(part.get("type") == "null" for part in node["anyOf"]):
                return None
            for part in node["anyOf"]:
                try:
                    return visit(item, part)
                except (ValueError, TypeError):
                    pass
            raise ValueError("Response value matches no transport union branch")
        kind = node.get("type")
        if kind is None:
            return arbitrary(item)
        if kind == "object":
            extra = node.get("additionalProperties", False)
            if isinstance(extra, dict) or extra is True:
                if not isinstance(item, list):
                    raise ValueError("A dynamic mapping must be encoded as key/value entries")
                result = {}
                for entry in item:
                    key = entry["key"]
                    if key in result:
                        raise ValueError(f"Duplicate mapping key: {key}")
                    result[key] = visit(entry["value"], extra if isinstance(extra, dict) else {})
                return result
            if not isinstance(item, dict):
                raise ValueError("Expected an object")
            properties = node.get("properties", {})
            return {key: visit(val, properties[key]) if key in properties else val for key, val in item.items()}
        if kind == "array":
            if not isinstance(item, list):
                raise ValueError("Expected an array")
            return [visit(val, node["items"]) for val in item]
        valid = {
            "string": isinstance(item, str),
            "number": isinstance(item, (int, float)) and not isinstance(item, bool),
            "integer": isinstance(item, int) and not isinstance(item, bool),
            "boolean": isinstance(item, bool),
            "null": item is None,
        }
        if not valid.get(kind, False):
            raise ValueError(f"Expected {kind}")
        return item

    return visit(value, schema)
