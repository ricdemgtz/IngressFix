Local run

export UBMS_LOG_PATH=$PWD/ubms_batch.log
python3 ingressfix.py --in sample.csv --out sample_fixed.csv \
    --batch-type cash_journal --rules rules.json --strict --max-errors 0


Local load (optional)

python3 ingressfix.py --in sample.csv --out sample_fixed.csv \
    --batch-type cash_journal --rules rules.json --strict --max-errors 0 \
    --load --db-user ricardo --db-pass test123 --db-host localhost --db-port 3306 --db-name veloxdb

