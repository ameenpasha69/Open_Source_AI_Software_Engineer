import pytest
from app.evaluation.models import EvalReport, TaskResult, TestSnapshot


def _result(**overrides) -> TaskResult:
    defaults = {
        "task_id": "t",
        "agent_status": "done",
        "verification_status": "verified",
        "success": True,
        "first_attempt_success": True,
        "regressed_tests": [],
        "iterations": 2,
        "tool_call_count": 3,
        "duration_seconds": 10.0,
        "modified_files": ["a.py"],
        "retrieved_expected_file": True,
        "baseline": TestSnapshot(exit_code=1, passed_tests=["t1"], failed_tests=["t2"]),
        "final": TestSnapshot(exit_code=0, passed_tests=["t1", "t2"], failed_tests=[]),
    }
    defaults.update(overrides)
    return TaskResult(**defaults)


def test_empty_report_has_zeroed_rates():
    report = EvalReport(config={}, results=[])
    assert report.task_count == 0
    assert report.success_rate == 0.0
    assert report.average_iterations == 0.0
    assert report.retrieval_hit_rate is None


def test_success_and_first_attempt_rates():
    report = EvalReport(
        config={},
        results=[
            _result(task_id="a", success=True, first_attempt_success=True),
            _result(task_id="b", success=True, first_attempt_success=False),
            _result(task_id="c", success=False, first_attempt_success=False),
        ],
    )
    assert report.success_count == 2
    assert report.success_rate == pytest.approx(2 / 3)
    assert report.first_attempt_count == 1
    assert report.first_attempt_rate == pytest.approx(1 / 3)


def test_averages():
    report = EvalReport(
        config={},
        results=[
            _result(iterations=2, tool_call_count=4, duration_seconds=10.0),
            _result(iterations=4, tool_call_count=6, duration_seconds=20.0),
        ],
    )
    assert report.average_iterations == 3.0
    assert report.average_tool_calls == 5.0
    assert report.average_duration_seconds == 15.0


def test_regression_rate_counts_tasks_with_any_regression():
    report = EvalReport(
        config={},
        results=[
            _result(task_id="a", regressed_tests=[]),
            _result(task_id="b", regressed_tests=["some_test"]),
        ],
    )
    assert report.regression_count == 1
    assert report.regression_rate == pytest.approx(0.5)


def test_retrieval_hit_rate_ignores_unscored_tasks():
    report = EvalReport(
        config={},
        results=[
            _result(task_id="a", retrieved_expected_file=True),
            _result(task_id="b", retrieved_expected_file=False),
            _result(task_id="c", retrieved_expected_file=None),
        ],
    )
    assert report.retrieval_hit_rate == pytest.approx(0.5)
