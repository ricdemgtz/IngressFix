# UBMS CSV Ingress Fix

Utility for normalizing broker batch CSV files before loading into UBMS. The
script is header-safe, quotes every field, and cleans numeric columns so that
values are database friendly.  All timestamps in logs are in the
`America/New_York` timezone.

## Requirements
* Python 3.10+
* Optional: [`PyMySQL`](https://pymysql.readthedocs.io/) when using `--load`

## Basic usage
```bash
export UBMS_LOG_PATH="$PWD/ubms_batch.log"   # optional override
python3 ingressfix.py \
  --in tests/tests_csvs/sample_cash_journal.csv \
  --out sample_cash_journal_fixed.csv \
  --batch-type cash_journal --rules rules.json \
  --strict --max-errors 0
```

This command processes the included `sample_cash_journal.csv` and writes
`sample_cash_journal_fixed.csv` in the current directory.

The original header row is written verbatim.  All fields are quoted on output
and inner quotes are doubled.  Numeric columns are stripped of currency symbols,
thousands separators and parentheses negatives.

### Batch mode helper
Process every `*.csv` in the current folder while skipping files that were
already fixed or quarantined:

```bash
for f in *.csv; do
  [[ $f == *_fixed.csv || $f == *.unrecoverable.csv ]] && continue
  python3 ingressfix.py --in "$f" --out "${f%.csv}_fixed.csv" \
    --batch-type cash_journal --rules rules.json --strict --max-errors 0
  # feed "${f%.csv}_fixed.csv" to the existing loader
done
```

## Optional local DB load
When `--load` is supplied the tool loads the fixed file into MySQL using a
temporary table.  Example with local development credentials:

```bash
python3 ingressfix.py --in sample.csv --out sample_fixed.csv \
  --batch-type cash_journal --rules rules.json --strict --max-errors 0 \
  --load --db-user ricardo --db-pass test123 --db-host localhost \
  --db-port 3306 --db-name veloxdb
```

## Testing

Example CSV files for development and testing reside in `tests/tests_csvs/`.
Run the test suite with:

```bash
pytest
```

Set the `UBMS_LOG_PATH` environment variable if you need to change the log file
location.

## Future integration
`hook_example.sh` demonstrates how an upload handler in UBMS DEV will invoke the
fixer and then pass the `_fixed.csv` file to existing loaders.  TODO markers in
code note areas that may require adjustments during real deployment.

## Production configuration
Update placeholder values before deploying:

* Paths in `scripts/hook_example.sh` such as `RULES`, the log file, and the
  location of `ingressfix.py`.
* Set the `UBMS_LOG_PATH` environment variable if a different log directory is
  required.
* Provide database credentials (`--db-user`, `--db-pass`, `--db-host`, etc.)
  when using the optional `--load` feature.
