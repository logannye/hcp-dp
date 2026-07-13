#!/usr/bin/env python3
"""Validate JSON instances against the repository's self-contained v2 schema.

This deliberately small validator implements the JSON Schema keywords used by
`hcp-align.v2.schema.json`, keeping local and CI checks dependency-free.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    pass


def resolve(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValidationError(f"unsupported external reference: {reference}")
    value: Any = root
    for component in reference[2:].split("/"):
        value = value[component.replace("~1", "/").replace("~0", "~")]
    return value


def type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise ValidationError(f"unsupported schema type: {expected}")


def validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    if "$ref" in schema:
        validate(value, resolve(root, schema["$ref"]), root, path)
        return

    if "oneOf" in schema:
        failures: list[str] = []
        matches = 0
        for branch in schema["oneOf"]:
            try:
                validate(value, branch, root, path)
                matches += 1
            except ValidationError as error:
                failures.append(str(error))
        if matches != 1:
            detail = "; ".join(failures[:3])
            raise ValidationError(f"{path}: expected exactly one schema match, got {matches}: {detail}")
        return

    if "const" in schema and value != schema["const"]:
        raise ValidationError(f"{path}: expected {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: {value!r} is not in {schema['enum']!r}")

    expected_type = schema.get("type")
    if expected_type is not None and not type_matches(value, expected_type):
        raise ValidationError(f"{path}: expected {expected_type}, got {type(value).__name__}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ValidationError(f"{path}: missing required properties {missing!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise ValidationError(f"{path}: unexpected properties {unexpected!r}")
        for name, child in value.items():
            if name in properties:
                validate(child, properties[name], root, f"{path}.{name}")

    if isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            validate(child, schema["items"], root, f"{path}[{index}]")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValidationError(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValidationError(f"{path}: array is longer than maxItems")
        if schema.get("uniqueItems"):
            normalized = [json.dumps(item, sort_keys=True) for item in value]
            if len(normalized) != len(set(normalized)):
                raise ValidationError(f"{path}: array items are not unique")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValidationError(f"{path}: string is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(f"{path}: string is longer than maxLength")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValidationError(f"{path}: string does not match {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: value is below minimum {schema['minimum']}")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(f"usage: {argv[0]} SCHEMA INSTANCE [INSTANCE ...]", file=sys.stderr)
        return 2
    schema_path = Path(argv[1])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for name in argv[2:]:
        instance_path = Path(name)
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
        try:
            if isinstance(instance, list) and schema.get("type") != "array":
                for index, item in enumerate(instance):
                    validate(item, schema, schema, f"$[{index}]")
            else:
                validate(instance, schema, schema)
        except ValidationError as error:
            print(f"{instance_path}: {error}", file=sys.stderr)
            return 1
        print(f"[schema] valid: {instance_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
