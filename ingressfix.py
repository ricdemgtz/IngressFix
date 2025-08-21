#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTP-137 | UBMS CSV Ingress Fix (header-safe, RFC-4180, Zabbix-friendly).
- Preserves header row verbatim.
- Quotes ALL fields on output; doubles inner quotes.
- Numeric normalization for $, parentheses negatives, thousands separators, +/-.
- Heuristic rebuild when numeric commas cause column-count drift.
- ET logging; lines starting with 'ERROR ' for critical failures (Zabbix).
- Optional --load step (temp table + LOAD DATA + INSERT … SELECT) for local tests.
"""

from __future__ import annotations
import argparse, csv, io, json, os, re, sys, time
from dataclasses import dataclass
from typing import Iterable, List, Set, Dict, Tuple, Optional
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    ET_TZ = None

try:
    import pymysql
except Exception:
    pymysql = None

DEFAULT_LOG = "/opt/tasks/log/ubms_batch.log"
FALLBACK_NUMERIC_NAMES = {
    "amount","amt","qty","quantity","price","debit","credit","balance",
    "monto","importe","net_amount","shares","nav","rate"
}

def now_et_str() -> str:
    """Return the current timestamp formatted in Eastern Time.

    The function falls back to the system timezone if the `zoneinfo` package is
    unavailable.  A plain string is returned to keep logging lightweight and
    avoid cross-module imports at call sites.
    """
    dt = datetime.now(ET_TZ) if ET_TZ else datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dir(path: str) -> None:
    """Ensure the parent directory for *path* exists."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def log_line(msg: str, log_path: str) -> None:
    """Append *msg* to *log_path* with an ET timestamp."""
    _ensure_dir(log_path)
    safe = msg.encode("ascii", "replace").decode("ascii")
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        f.write(f"{now_et_str()} {safe}\n")

def log_info(msg, log_path):  """Write an INFO level message."""; log_line(f"INFO  {msg}", log_path)
def log_warn(msg, log_path):  """Write a WARNING level message."""; log_line(f"WARNING  {msg}", log_path)
def log_error(msg, log_path): """Write an ERROR level message (Zabbix expects 'ERROR ')."""; log_line(f"ERROR {msg}", log_path)  # exact "ERROR "

# ---------- Numeric normalization ----------
_num_re = re.compile(r"^\d+(?:\.\d+)?$")

def normalize_numeric_cell(val: str) -> Tuple[str, bool, bool]:
    """Return (normalized_value, changed, bad). bad=True if irrecoverably non-numeric."""
    raw = val
    s = (val or "").strip()
    if s == "":
        return ("", False, False)

    t = s
    neg = False

    # handle $(...) or ($...)
    if (t.startswith("$(") and t.endswith(")")) or (t.startswith("($") and t.endswith(")")):
        neg, t = True, t[2:-1].strip()
    else:
        if t.startswith("$"): t = t[1:].strip()
        if t.startswith("(") and t.endswith(")"):
            neg, t = True, t[1:-1].strip()

    # thousands & signs
    t = t.replace(",", "").strip()
    if t.startswith("+"): t = t[1:].strip()
    if t.startswith("-"):
        neg, t = True, t[1:].strip()

    if not _num_re.fullmatch(t):
        return (raw, False, True)

    out = ("-" if (neg and t != "") else "") + t
    return (out, out != raw.strip(), False)

# ---------- Date normalization ----------
_date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y", "%Y%m%d"]

def normalize_date_cell(val: str) -> Tuple[str, bool, bool]:
    """Return (normalized_value, changed, bad). bad=True if not a valid date."""
    raw = val
    s = (val or "").strip()
    if s == "":
        return ("", False, False)
    for fmt in _date_formats:
        try:
            dt = datetime.strptime(s, fmt)
            norm = dt.strftime("%Y-%m-%d")
            return (norm, norm != s, False)
        except ValueError:
            pass
    return (raw, False, True)

def _looks_like_numeric_token(token: str) -> bool:
    """Heuristically determine if *token* resembles a numeric value."""
    t = (token or "").strip()
    # support $, parentheses, thousands
    if (t.startswith("$(") and t.endswith(")")) or (t.startswith("($") and t.endswith(")")):
        t = t[2:-1].strip()
    else:
        if t.startswith("$"):
            t = t[1:].strip()
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1].strip()
    t = t.replace(",", "")
    pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
    return bool(re.fullmatch(pattern, (token or "").strip())) or bool(_num_re.fullmatch(t))

def heuristic_rebuild(raw_line: str, expected_cols: int, numeric_idx: Set[int],
                      text_idx: Set[int]) -> Optional[List[str]]:
    """Greedy attempt to rebuild a mis-parsed CSV row.

    Parameters
    ----------
    raw_line:
        Original CSV line without surrounding newline.
    expected_cols:
        Desired number of fields once rebuilt.
    numeric_idx:
        Column indexes expected to contain numeric values.  Tokens at these
        positions are merged first using a numeric heuristic.
    text_idx:
        Column indexes for non-numeric fields that may legitimately contain
        commas.  Remaining extra tokens are distributed across these positions
        from left to right.

    Returns
    -------
    list[str] | None
        A rebuilt list of fields if the final column count matches
        ``expected_cols``; otherwise ``None`` to signal failure.
    """
    parts = raw_line.split(",")
    fields, idx = [], 0
    extra_total = max(0, len(parts) - expected_cols)
    while idx < len(parts) and len(fields) < expected_cols:
        pos = len(fields)
        if pos in numeric_idx:
            best, best_used = None, 0
            max_used = min(8, len(parts) - idx)
            for used in range(1, max_used + 1):
                cand = ",".join(parts[idx:idx+used]).strip()
                if _looks_like_numeric_token(cand):
                    best, best_used = cand, used
            if best is None:
                remaining_parts = len(parts) - idx
                remaining_fields = expected_cols - len(fields)
                used = max(1, remaining_parts - (remaining_fields - 1))
                best, best_used = ",".join(parts[idx:idx+used]), used
            fields.append(best)
            idx += best_used
            extra_total = max(0, extra_total - (best_used - 1))
        elif pos in text_idx:
            remaining_text = sum(1 for i in text_idx if i >= pos)
            if extra_total >= remaining_text:
                used = 1 + (extra_total - (remaining_text - 1))
            else:
                used = 1
            fields.append(",".join(parts[idx:idx+used]))
            idx += used
            extra_total = max(0, extra_total - (used - 1))
        else:
            fields.append(parts[idx])
            idx += 1
    if len(fields) < expected_cols and idx < len(parts):
        fields.append(",".join(parts[idx:]))
    return fields if len(fields) == expected_cols else None

def open_db(args):
    """Open a MySQL connection using arguments from the CLI."""
    if pymysql is None:
        raise RuntimeError("PyMySQL not installed; cannot use DB features")
    return pymysql.connect(
        host=args.db_host, port=args.db_port,
        user=args.db_user, password=args.db_pass,
        database=args.db_name, autocommit=True, local_infile=True, charset="utf8mb4"
    )

def get_numeric_columns(conn, table_fullname: str) -> Set[str]:
    """Inspect ``information_schema`` to discover numeric columns."""
    if "." in table_fullname:
        schema, table = table_fullname.split(".", 1)
    else:
        with conn.cursor() as c:
            c.execute("SELECT DATABASE()")
            schema = c.fetchone()[0]
        table = table_fullname
    sql = """
    SELECT COLUMN_NAME, DATA_TYPE
    FROM information_schema.columns
    WHERE table_schema=%s AND table_name=%s
    """
    numeric_types = {"decimal","numeric","float","double","double precision","real",
                     "int","integer","bigint","smallint","mediumint","tinyint"}
    out = set()
    with conn.cursor() as c:
        c.execute(sql, (schema, table))
        for name, dtype in c.fetchall():
            if str(dtype).lower() in numeric_types:
                out.add(name.lower())
    return out

def load_rules_json(path: Optional[str]) -> Dict[str, dict]:
    """Load the optional JSON rules manifest.

    The manifest maps batch types to configuration, including the import table
    name and the lists of numeric and date columns.  Missing or malformed files
    result in an empty dictionary.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def repair_and_write_csv(in_path: str, out_path: str, sidecar_path: str,
                         numeric_cols: Set[str], date_cols: Set[str], log_path: str,
                         strict: bool, max_errors: int,
                         column_count: int = 0) -> Tuple[int,int,int]:
    """Repair *in_path* and write sanitized output to *out_path*.

    Numeric columns are normalized with :func:`normalize_numeric_cell` and date
    columns with :func:`normalize_date_cell`.

    Returns a tuple ``(total_rows, repaired_rows, unrecoverable_rows)``.  Rows
    deemed unrecoverable are written to *sidecar_path*, which is created lazily
    only when needed.
    """
    total = repaired = unrecoverable = 0
    bad_writer = None
    sidecar_fh = None
    sidecar_tmp = sidecar_path + ".tmp"
    _ensure_dir(out_path)
    out_tmp = out_path + ".tmp"

    try:
        with open(in_path, "r", encoding="utf-8-sig", newline="") as fin, \
             open(out_tmp, "w", encoding="utf-8", newline="") as fout:

            # header: read once, write verbatim
            first_line = fin.readline()
            if not first_line:
                log_warn("Empty file; nothing to do", log_path)
                return (0,0,0)

            # handle optional leading comment like "# batch_type=..."
            if first_line.lstrip().startswith("#"):
                fout.write(first_line)  # preserve comment line
                header_line = fin.readline()
                if not header_line:
                    log_warn("Missing header after comment line; nothing to do", log_path)
                    return (0,0,0)
                fout.write(header_line)
                header = next(csv.reader([header_line]))
            else:
                # write the header exactly as read, preserving original newline
                fout.write(first_line)
                header = next(csv.reader([first_line]))

            # writer is only used for subsequent rows
            writer = csv.writer(fout, quoting=csv.QUOTE_ALL)

            header_map = {h.strip().lower(): i for i, h in enumerate(header)}
            header_keys = set(header_map)

            if column_count and len(header) != column_count:
                log_warn(
                    f"Header has {len(header)} column(s) but column_count={column_count}",
                    log_path,
                )

            if numeric_cols:
                missing_numeric = numeric_cols - header_keys
                if missing_numeric:
                    log_warn(
                        "numeric_cols not in header: " + ", ".join(sorted(missing_numeric)),
                        log_path,
                    )
                numeric_idx = {header_map[h] for h in numeric_cols if h in header_map}
            else:
                numeric_idx = {header_map[h] for h in header_map if h in FALLBACK_NUMERIC_NAMES}

            if date_cols:
                missing_dates = date_cols - header_keys
                if missing_dates:
                    log_warn(
                        "date_cols not in header: " + ", ".join(sorted(missing_dates)),
                        log_path,
                    )
                date_idx = {header_map[h] for h in date_cols if h in header_map}
            else:
                date_idx = set()

            expected = column_count or len(header)
            if not column_count:
                log_warn(
                    f"column_count not provided; assuming {expected}",
                    log_path,
                )

            text_idx = set(range(expected)) - numeric_idx if numeric_idx else set()
            reader = csv.reader(fin)
            line_no = 1

            for row in reader:
                line_no += 1
                if len(row) < expected:
                    pad = expected - len(row)
                    row = row + [""] * pad
                    log_warn(
                        f"Row {line_no} missing {pad} column(s); padded", log_path
                    )
                elif len(row) > expected:
                    raw_record = reader.dialect.delimiter.join(row)
                    extra = len(row) - expected
                    rebuilt = heuristic_rebuild(raw_record, expected, numeric_idx, text_idx)
                    if rebuilt is None:
                        if any(idx >= expected for idx in numeric_idx):
                            unrecoverable += 1
                            if bad_writer is None:
                                _ensure_dir(sidecar_path)
                                sidecar_fh = open(sidecar_tmp, "w", encoding="utf-8", newline="")
                                bad_writer = csv.writer(sidecar_fh, quoting=csv.QUOTE_ALL)
                                bad_writer.writerow(header)
                            bad_writer.writerow(row)
                            log_error(
                                f"Row {line_no} unrecoverable: extra columns; raw saved -> {os.path.basename(sidecar_path)}",
                                log_path,
                            )
                            if strict or (max_errors and unrecoverable >= max_errors):
                                return (total, repaired, unrecoverable)
                            continue
                        else:
                            row = row[:expected]
                            log_warn(
                                f"Row {line_no} had {extra} extra column(s); truncated",
                                log_path,
                            )
                    else:
                        row = rebuilt
                        log_warn(
                            f"Row {line_no} had {extra} extra column(s); rebuilt",
                            log_path,
                        )

                total += 1
                changed_any = False
                bad_cell = False
                out_row: List[str] = []
                for idx, val in enumerate(row):
                    v = (val or "").strip()
                    if idx in numeric_idx:
                        norm, changed, bad = normalize_numeric_cell(v)
                        if bad: bad_cell = True
                        if changed: changed_any = True
                        out_row.append(norm)
                    elif idx in date_idx:
                        norm, changed, bad = normalize_date_cell(v)
                        if bad: bad_cell = True
                        if changed: changed_any = True
                        out_row.append(norm)
                    else:
                        out_row.append(v)
                if bad_cell:
                    unrecoverable += 1
                    if bad_writer is None:
                        _ensure_dir(sidecar_path)
                        sidecar_fh = open(sidecar_tmp, "w", encoding="utf-8", newline="")
                        bad_writer = csv.writer(sidecar_fh, quoting=csv.QUOTE_ALL)
                        bad_writer.writerow(header)
                    bad_writer.writerow(row)
                    log_error(
                        f"Row {line_no} unrecoverable: invalid numeric or date value; raw saved",
                        log_path,
                    )
                    if strict or (max_errors and unrecoverable >= max_errors):
                        return (total, repaired, unrecoverable)
                    continue

                if changed_any:
                    repaired += 1
                writer.writerow(out_row)

    finally:
        if sidecar_fh is not None:
            sidecar_fh.close()
            os.replace(sidecar_tmp, sidecar_path)
        if os.path.exists(out_tmp):
            os.replace(out_tmp, out_path)

    return (total, repaired, unrecoverable)

SQL_NORMALIZE_EXPR = r"""
  REPLACE(REPLACE(
    CASE
      WHEN {col} REGEXP '^\(.*\)$' THEN CONCAT('-', TRIM(BOTH ')' FROM TRIM(BOTH '(' FROM {col})))
      ELSE {col}
    END
  , '$', ''), ',', '')
"""

def db_load_insert(args, out_path: str, import_table: str, log_path: str):
    """Load the fixed CSV into MySQL for local testing purposes."""
    if pymysql is None:
        raise RuntimeError("PyMySQL not installed; cannot use --load")
    conn = open_db(args)
    tbl = import_table if "." in import_table else f"{args.db_name}.{import_table}"
    tmp = f"{import_table}_stg_{int(time.time())}"
    with conn.cursor() as c:
        c.execute(f"CREATE TEMPORARY TABLE {tmp} LIKE {tbl}")
    with conn.cursor() as c:
        c.execute(f"""
            LOAD DATA LOCAL INFILE %s
            INTO TABLE {tmp}
            FIELDS TERMINATED BY ',' ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES
        """, (out_path,))
        loaded = c.rowcount
        log_info(f"Loaded {loaded} rows into {tmp}", log_path)
    # fetch column types
    with conn.cursor() as c:
        c.execute("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ORDINAL_POSITION
        """, (args.db_name, import_table))
        cols = c.fetchall()
    numeric_types = {"decimal","numeric","float","double","double precision","real",
                     "int","integer","bigint","smallint","mediumint","tinyint"}
    select_exprs, col_names = [], []
    for (col, dtype) in cols:
        col_names.append(f"`{col}`")
        if str(dtype).lower() in numeric_types:
            select_exprs.append(f"NULLIF(TRIM({SQL_NORMALIZE_EXPR.format(col=f'`{col}`')}), '')")
        else:
            select_exprs.append(f"TRIM(`{col}`)")
    insert_sql = f"INSERT INTO {tbl} ({', '.join(col_names)}) SELECT {', '.join(select_exprs)} FROM {tmp}"
    with conn.cursor() as c:
        c.execute(insert_sql)
        inserted = c.rowcount
    log_info(f"Inserted {inserted} rows into {tbl}", log_path)
    conn.close()

def main():
    p = argparse.ArgumentParser(description="VTP-137 UBMS CSV Ingress Fix")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--batch-type", dest="batch_type", required=True)
    p.add_argument("--rules", dest="rules_path", default="rules.json")
    p.add_argument("--log", dest="log_path", default=os.getenv("UBMS_LOG_PATH", DEFAULT_LOG))
    p.add_argument("--strict", action="store_true")
    p.add_argument("--max-errors", type=int, default=0)
    p.add_argument("--run", dest="run_id", default="", help="Optional run identifier")
    # DB flags (for local dev only)
    p.add_argument("--load", action="store_true")
    p.add_argument("--db-user", default="ricardo")
    p.add_argument("--db-pass", default="test123")
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-name", default="veloxdb")
    args = p.parse_args()

    run_note = f" run={args.run_id}" if args.run_id else ""
    log_info(f"#VTP-137 start{run_note}; in={args.in_path} out={args.out_path} batch={args.batch_type}", args.log_path)

    # Resolve rules for numeric and date columns
    rules = load_rules_json(args.rules_path)
    r = rules.get(args.batch_type, {})
    import_table = r.get("import_table", "")
    numeric_cols_from_manifest = {c.strip().lower() for c in r.get("numeric_cols", [])}
    date_cols_from_manifest = {c.strip().lower() for c in r.get("date_cols", [])}
    column_count = int(r.get("column_count", 0))

    numeric_cols: Set[str] = set()
    date_cols: Set[str] = date_cols_from_manifest
    # Prefer information_schema for numeric cols if table is known
    if import_table and pymysql is not None:
        try:
            conn = open_db(args)
            numeric_cols = get_numeric_columns(conn, import_table)
            conn.close()
        except Exception as e:
            log_warn(
                f"info_schema not available for {import_table}, fallback to manifest/names. detail={e}",
                args.log_path,
            )
            numeric_cols = numeric_cols_from_manifest
    else:
        numeric_cols = numeric_cols_from_manifest

    sidecar = os.path.splitext(args.out_path)[0] + ".unrecoverable.csv"
    try:
        total, repaired, bad = repair_and_write_csv(
            args.in_path, args.out_path, sidecar, numeric_cols, date_cols,
            args.log_path, args.strict, args.max_errors, column_count
        )
    except Exception as e:
        log_error(f"Repair failed: {e}", args.log_path)
        sys.exit(2)

    log_info(f"Repair stats total={total} repaired={repaired} unrecoverable={bad}", args.log_path)
    if bad > 0 and args.strict:
        log_error(f"Strict mode: {bad} unrecoverable rows. See {sidecar}", args.log_path)
        sys.exit(1)

    if args.load:
        if not import_table:
            log_warn("No import_table in rules for this batch-type; skipping --load.", args.log_path)
        else:
            try:
                db_load_insert(args, args.out_path, import_table, args.log_path)
            except Exception as e:
                log_error(f"Load failed: {e}", args.log_path)
                sys.exit(2)

    log_info("#VTP-137 done", args.log_path)
    sys.exit(0)

if __name__ == "__main__":
    main()
