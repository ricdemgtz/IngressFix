import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingressfix import normalize_numeric_cell, normalize_date_cell, repair_and_write_csv


def test_normalize_numeric_cell():
    cases = [
        ("$1,234.50", "1234.50"),
        ("(1,000.25)", "-1000.25"),
        ("$+5", "5"),
        ("-42", "-42"),
    ]
    for raw, expected in cases:
        out, changed, bad = normalize_numeric_cell(raw)
        assert out == expected
        assert not bad
    out, changed, bad = normalize_numeric_cell("oops")
    assert bad


def test_normalize_date_cell():
    cases = [
        ("2023-01-02", "2023-01-02"),      # already ISO
        ("01/02/2023", "2023-01-02"),      # slashes
        ("2023/01/02", "2023-01-02"),      # ISO with slashes
        ("02-03-2023", "2023-02-03"),      # dashes with month first
        ("20230104", "2023-01-04"),        # compact digits
    ]
    for raw, expected in cases:
        out, changed, bad = normalize_date_cell(raw)
        assert out == expected
        assert not bad
    for bad_input in ["2023-13-01", "02/30/2023"]:
        out, changed, bad = normalize_date_cell(bad_input)
        assert bad


def test_repair_and_sidecar(tmp_path: Path):
    inp = tmp_path / "sample.csv"
    inp.write_text(
        "account,amount,description\n"
        '"A1","1,732.50","text, with comma"\n'
        "A2,1,000.25,desc2\n"  # unquoted numeric
        "A3,oops,bad\n"        # non-numeric amount
        "A4,1\n"               # missing description
    )
    out = tmp_path / "sample_fixed.csv"
    side = tmp_path / "sample_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), {"amount"}, set(), str(log), False, 0
    )
    # two good rows, two unrecoverable
    assert total == 3
    assert bad == 2

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["account", "amount", "description"]
    assert rows[1] == ["A1", "1732.50", "text, with comma"]
    assert rows[2] == ["A2", "1000.25", "desc2"]

    with side.open() as f:
        bad_rows = list(csv.reader(f))
    assert bad_rows[0] == ["account", "amount", "description"]
    assert bad_rows[1] == ["A3", "oops", "bad"]
    assert bad_rows[2] == ["A4", "1"]


def test_date_normalization(tmp_path: Path):
    inp = tmp_path / "dates.csv"
    inp.write_text(
        "account,date,amount\n"
        "A1,2023-01-02,10\n"
        "A2,1/3/2023,20\n"
        "A3,not-a-date,30\n"
    )
    out = tmp_path / "dates_fixed.csv"
    side = tmp_path / "dates_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), {"amount"}, {"date"}, str(log), False, 0
    )
    assert total == 3
    assert bad == 1

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["A1", "2023-01-02", "10"]
    assert rows[2] == ["A2", "2023-01-03", "20"]

    with side.open() as f:
        side_rows = list(csv.reader(f))
    assert side_rows[1] == ["A3", "not-a-date", "30"]


def test_no_sidecar_when_clean(tmp_path: Path):
    inp = tmp_path / "clean.csv"
    inp.write_text(
        "account,amount,description\n"
        '"A1","1,000.00","ok"\n'
    )
    out = tmp_path / "clean_fixed.csv"
    side = tmp_path / "clean_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), {"amount"}, set(), str(log), False, 0
    )
    assert total == 1 and bad == 0
    assert not side.exists()


def test_preserve_newline_in_field(tmp_path: Path):
    inp = tmp_path / "multiline.csv"
    inp.write_text(
        "account,description\n"
        "A1,\"line1\nline2\"\n"
        "A2,\"ok\"\n"
    )
    out = tmp_path / "multiline_fixed.csv"
    side = tmp_path / "multiline_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), set(), set(), str(log), False, 0
    )
    assert total == 2 and bad == 0

    with out.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3
    assert rows[1] == ["A1", "line1\nline2"]
