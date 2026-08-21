from app.evaluation.models import EvalReport, TaskResult, TestSnapshot
from app.evaluation.report import format_report


def test_format_report_includes_headline_metrics_and_per_task_detail():
    report = EvalReport(
        config={"llm_model": "qwen2.5-coder:7b"},
        results=[
            TaskResult(
                task_id="task_001_calc_accumulate",
                agent_status="done",
                verification_status="verified",
                success=True,
                first_attempt_success=True,
                regressed_tests=[],
                iterations=3,
                tool_call_count=4,
                duration_seconds=12.3,
                modified_files=["calc.py"],
                retrieved_expected_file=True,
                baseline=TestSnapshot(exit_code=1, passed_tests=["a"], failed_tests=["b"]),
                final=TestSnapshot(exit_code=0, passed_tests=["a", "b"], failed_tests=[]),
            ),
            TaskResult(
                task_id="task_002_broke_something",
                agent_status="max_iterations_reached",
                verification_status="unverified",
                success=False,
                first_attempt_success=False,
                regressed_tests=["test_x"],
                iterations=8,
                tool_call_count=10,
                duration_seconds=40.1,
                modified_files=["x.py"],
                retrieved_expected_file=False,
                baseline=TestSnapshot(exit_code=1, passed_tests=["test_x"], failed_tests=["test_y"]),
                final=TestSnapshot(exit_code=1, passed_tests=[], failed_tests=["test_x", "test_y"]),
            ),
        ],
    )

    text = format_report(report)

    assert "Tasks:                 2" in text
    assert "Successful:            1" in text
    assert "Success rate:          50%" in text
    assert "llm_model=qwen2.5-coder:7b" in text
    assert "task_001_calc_accumulate" in text
    assert "task_002_broke_something" in text
    assert "regressed: test_x" in text
