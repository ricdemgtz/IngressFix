import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingressfix import (
    heuristic_rebuild,
    normalize_date_cell,
    normalize_numeric_cell,
    repair_and_write_csv,
)


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
        ("2023-01-02", "2023-01-02", False),  # already ISO
        ("01/02/2023", "2023-01-02", True),  # slashes
        ("2023/01/02", "2023-01-02", True),  # ISO with slashes
        ("02-03-2023", "2023-02-03", True),  # dashes with month first
        ("20230104", "2023-01-04", True),    # compact digits
        ("", "", False),                    # empty remains empty
    ]
    for raw, expected, changed_flag in cases:
        out, changed, bad = normalize_date_cell(raw)
        assert out == expected
        assert changed == changed_flag
        assert not bad
    for bad_input in ["2023-13-01", "02/30/2023", "2023/13/01"]:
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
        str(inp), str(out), str(side), {"amount"}, set(), str(log), False, 0, 3
    )
    # three good rows, one unrecoverable
    assert total == 4
    assert bad == 1

    # header line should be preserved byte-for-byte
    with inp.open("rb") as fin, out.open("rb") as fout:
        assert fin.readline() == fout.readline()

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["account", "amount", "description"]
    assert rows[1] == ["A1", "1732.50", "text, with comma"]
    assert rows[2] == ["A2", "1000.25", "desc2"]
    assert rows[3] == ["A4", "1", ""]

    with side.open() as f:
        bad_rows = list(csv.reader(f))
    assert bad_rows[0] == ["account", "amount", "description"]
    assert bad_rows[1] == ["A3", "oops", "bad"]




def test_header_preserved(tmp_path: Path):
    inp = tmp_path / 'header.csv'
    # include CRLF to ensure newline preserved
    inp.write_bytes(b'col1,col2\r\n1,2\r\n')
    out = tmp_path / 'header_fixed.csv'
    side = tmp_path / 'header_fixed.unrecoverable.csv'
    log = tmp_path / 'test.log'

    total, repaired, bad = repair_and_write_csv(str(inp), str(out), str(side), set(), set(), str(log), False, 0)
    assert total == 1 and bad == 0

    with inp.open('rb') as fin, out.open('rb') as fout:
        assert fin.readline() == fout.readline()


def test_comment_line_before_header(tmp_path: Path):
    inp = tmp_path / "with_comment.csv"
    inp.write_text("# batch_type=foo\ncol1,col2\nv1,1\n")
    out = tmp_path / "with_comment_fixed.csv"
    side = tmp_path / "with_comment_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), set(), set(), str(log), False, 0
    )
    assert total == 1 and bad == 0

    with out.open() as f:
        first = f.readline().strip()
        assert first == "# batch_type=foo"
        rows = list(csv.reader(f))
    assert rows[0] == ["col1", "col2"]
    assert rows[1] == ["v1", "1"]
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


def test_pad_missing_columns(tmp_path: Path):
    inp = tmp_path / "missing.csv"
    inp.write_text("a,b,c\n1,2\n")
    out = tmp_path / "missing_fixed.csv"
    side = tmp_path / "missing_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), set(), set(), str(log), False, 0, 3
    )
    assert total == 1 and bad == 0

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["1", "2", ""]
    with log.open() as f:
        content = f.read()
    assert "missing 1 column" in content
    assert not side.exists()


def test_extra_columns_truncated(tmp_path: Path):
    inp = tmp_path / "extra.csv"
    inp.write_text("a,b,c\n1,2,3,4\n")
    out = tmp_path / "extra_fixed.csv"
    side = tmp_path / "extra_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), set(), set(), str(log), False, 0, 3
    )
    assert total == 1 and bad == 0

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["1", "2", "3"]
    with log.open() as f:
        content = f.read()
    assert "extra column" in content
    assert not side.exists()


def test_repair_text_column(tmp_path: Path):
    inp = tmp_path / "text.csv"
    inp.write_text(
        "account,description,amount\n"
        "A1,desc,with comma,100\n"
    )
    out = tmp_path / "text_fixed.csv"
    side = tmp_path / "text_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), {"amount"}, set(), str(log), False, 0
    )
    assert total == 1 and bad == 0

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["A1", "desc,with comma", "100"]


def test_heuristic_rebuild_numeric_and_text():
    raw = "A1,1,234,desc,with,comma"
    rebuilt = heuristic_rebuild(raw, 4, {1}, {0,2,3})
    assert rebuilt == ["A1", "1,234", "desc", "with,comma"]


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
    assert total == 2 and repaired == 0 and bad == 0

    with out.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3
    assert rows[1] == ["A1", "line1\nline2"]


def test_multiline_field_count(tmp_path: Path):
    inp = tmp_path / "multiline_count.csv"
    inp.write_text(
        "account,description\n"
        "A1,\"line1\nline2\"\n"
        "A2,\"second\"\n"
    )
    out = tmp_path / "multiline_count_fixed.csv"
    side = tmp_path / "multiline_count_fixed.unrecoverable.csv"
    log = tmp_path / "test.log"

    total, repaired, bad = repair_and_write_csv(
        str(inp), str(out), str(side), set(), set(), str(log), False, 0
    )
    assert total == 2 and repaired == 0 and bad == 0

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["A1", "line1\nline2"]
    assert rows[2] == ["A2", "second"]


def test_sample_csvs(tmp_path: Path):
    samples_dir = Path(__file__).resolve().parent / "tests_csvs"
    rules_path = Path(__file__).resolve().parents[1] / "rules.json"
    rules = json.loads(rules_path.read_text())

    for sample in samples_dir.glob("sample_*.csv"):
        lines = sample.read_text().splitlines(True)
        assert lines, "sample file must not be empty"
        first = lines[0].strip()
        assert first.startswith("# batch_type="), "missing batch_type comment"
        batch_type = first.split("=", 1)[1]
        cfg = rules.get(batch_type, {})
        numeric_cols = {c.lower() for c in cfg.get("numeric_cols", [])}
        date_cols = {c.lower() for c in cfg.get("date_cols", [])}

        tmp_in = tmp_path / sample.name
        tmp_in.write_text("".join(lines[1:]))
        out = tmp_path / f"{sample.stem}_fixed.csv"
        side = tmp_path / f"{sample.stem}_fixed.unrecoverable.csv"
        log = tmp_path / f"{sample.stem}.log"

        total, repaired, bad = repair_and_write_csv(
            str(tmp_in), str(out), str(side), numeric_cols, date_cols, str(log), False, 0
        )

        assert total == 1
        assert bad == 0
        assert not side.exists()

        with tmp_in.open("rb") as fin, out.open("rb") as fout:
            assert fin.readline() == fout.readline()

        with tmp_in.open() as fin:
            expected_rows = list(csv.reader(fin))
        with out.open() as fout:
            out_rows = list(csv.reader(fout))

        # header should always be preserved byte-for-byte
        assert out_rows[0] == expected_rows[0]
