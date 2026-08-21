from app.evaluation.test_snapshot import take_test_snapshot


async def test_snapshot_reports_passed_and_failed_tests(tmp_path):
    (tmp_path / "test_sample.py").write_text(
        "def test_pass():\n    assert 1 == 1\n\n\ndef test_fail():\n    assert 1 == 2\n"
    )

    snapshot = await take_test_snapshot(tmp_path)

    assert snapshot.exit_code == 1
    assert snapshot.passed_tests == ["test_sample.py::test_pass"]
    assert snapshot.failed_tests == ["test_sample.py::test_fail"]


async def test_snapshot_all_passed_has_zero_exit_code(tmp_path):
    (tmp_path / "test_sample.py").write_text("def test_pass():\n    assert True\n")

    snapshot = await take_test_snapshot(tmp_path)

    assert snapshot.exit_code == 0
    assert snapshot.passed_tests == ["test_sample.py::test_pass"]
    assert snapshot.failed_tests == []
