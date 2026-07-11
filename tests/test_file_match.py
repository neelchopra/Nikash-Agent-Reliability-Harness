from arh.grader.file_match import grade_file_match
from arh.models import GraderSpec

SPEC = GraderSpec(
    kind="file_match",
    expect_files=["notes.md", "report.md"],
    forbid_files=["*.txt"],
    expect_content={"notes.md": "Buy milk"},
)


def _seed(tmp_path, files: dict[str, str]):
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


def test_pass_when_all_checks_hold(tmp_path):
    d = _seed(tmp_path, {"notes.md": "Buy milk\n", "report.md": "Q2\n"})
    ok, detail = grade_file_match(d, SPEC)
    assert ok, detail


def test_fail_on_missing_expected_file(tmp_path):
    d = _seed(tmp_path, {"notes.md": "Buy milk\n"})
    ok, detail = grade_file_match(d, SPEC)
    assert not ok
    assert "report.md" in detail


def test_fail_on_forbidden_file_present(tmp_path):
    d = _seed(tmp_path, {"notes.md": "Buy milk\n", "report.md": "Q2\n", "left.txt": "x"})
    ok, detail = grade_file_match(d, SPEC)
    assert not ok
    assert "left.txt" in detail


def test_fail_on_content_mismatch(tmp_path):
    d = _seed(tmp_path, {"notes.md": "wrong\n", "report.md": "Q2\n"})
    ok, detail = grade_file_match(d, SPEC)
    assert not ok
    assert "notes.md" in detail


MULTI_SPEC = GraderSpec(
    kind="file_match",
    expect_content={"log.txt": ["System started.", "Task completed successfully."]},
)


def test_pass_when_all_required_substrings_present(tmp_path):
    d = _seed(tmp_path, {"log.txt": "System started.\nTask completed successfully.\n"})
    ok, detail = grade_file_match(d, MULTI_SPEC)
    assert ok, detail


def test_fail_when_one_of_multiple_substrings_missing(tmp_path):
    d = _seed(tmp_path, {"log.txt": "System started.\n"})
    ok, detail = grade_file_match(d, MULTI_SPEC)
    assert not ok
    assert "Task completed successfully." in detail


def test_single_string_expect_content_coerces_to_list():
    spec = GraderSpec(kind="file_match", expect_content={"notes.md": "Buy milk"})
    assert spec.expect_content == {"notes.md": ["Buy milk"]}
