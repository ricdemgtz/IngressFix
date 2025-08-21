#!/usr/bin/env bash
# Example: UBMS upload handler invoking the fixer.
# Inputs: $1=type (e.g., cash_journal), $2=src_csv, $3=dst_csv (fixed)
# TODO markers highlight values to update for the target environment.
set -euo pipefail
TYPE="$1"; SRC="$2"; DST="$3"

# TODO: adjust RULES path for target environment
RULES="/home/settle/ubms_pro/server/template/ubms_csv_fix/rules.json"

# TODO: configure log path or set UBMS_LOG_PATH for target environment
LOG="${UBMS_LOG_PATH:-/opt/tasks/log/ubms_batch.log}"

# TODO: adjust ingressfix.py path for target environment
python3 /home/settle/ubms_pro/server/template/ubms_csv_fix/ingressfix.py \
  --in "$SRC" --out "$DST" --batch-type "$TYPE" --rules "$RULES" \
  --max-errors 0 --strict --log "$LOG"
# then pass "$DST" to the existing loader...
