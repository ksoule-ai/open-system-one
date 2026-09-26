#!/usr/bin/env bash
# Regenerate src/open_system_one/schema/models.py from a TypeSafe OpenAPI snapshot.
# Usage: scripts/generate_schema.sh [snapshot.json]   (default: newest schemas/typesafe-openapi-*.json)
# Never hand-edit the output; refresh the snapshot and rerun instead (see .claude/rules/api-compat.md).
set -euo pipefail
snapshot="${1:-$(ls schemas/typesafe-openapi-*.json | sort | tail -1)}"
out=src/open_system_one/schema/models.py
uv run datamodel-codegen \
    --input "$snapshot" --input-file-type openapi \
    --output "$out" --output-model-type pydantic_v2.BaseModel \
    --use-union-operator --use-standard-collections --target-python-version 3.11 \
    --field-constraints --disable-timestamp --formatters ruff-format
# Record the snapshot by file name, not by the machine-specific path it was read from.
sed -i.bak "s|^#   filename:.*|#   filename:  $(basename "$snapshot")|" "$out" && rm "$out.bak"
echo "generated $out from $snapshot"
