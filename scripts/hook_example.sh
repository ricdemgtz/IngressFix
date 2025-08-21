#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Hook script for preprocessing UBMS CSV uploads with ingressfix.
# Intended use: run before importing a CSV to validate and repair the file.
# ---------------------------------------------------------------------------
# Inputs:
#   $1 - batch type (e.g., cash_journal)
#   $2 - source CSV path
#   $3 - destination path for fixed CSV
# TODO markers highlight values that must be adjusted for the production environment.
# ingressfix.py relies on MySqlAdapter and MyLog wrappers included with the server template.

set -euo pipefail

# Verify input: ensure exactly three arguments are provided
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <type> <src_csv> <dst_csv>" >&2
  exit 1
fi

# TYPE: batch type used to select the appropriate transformation rules
TYPE="$1"
# SRC: incoming CSV file to be fixed
SRC="$2"
# DST: path where the corrected CSV will be written
DST="$3"

# TODO: adjust RULES path for target environment.
# RULES: JSON file with field-level fixing rules
RULES="/home/settle/ubms_pro/server/template/ubms_csv_fix/rules.json"

# TODO: configure log path or set UBMS_LOG_PATH for target environment.
# LOG: file where processing logs will be written
LOG="${UBMS_LOG_PATH:-/opt/tasks/log/ubms_batch.log}"

# TODO: adjust ingressfix.py path for target environment.
# Step: run ingressfix to validate and repair the CSV
python3 /home/settle/ubms_pro/server/template/ubms_csv_fix/ingressfix.py \
  --in "$SRC" --out "$DST" --batch-type "$TYPE" --rules "$RULES" \
  --max-errors 0 --strict --log "$LOG"

# TODO: integrate with production loader; pass "$DST" to existing loader
# After successful fixing, invoke the loader with the cleaned file.
# Example placeholder:
# /path/to/ubms_loader "$DST"

