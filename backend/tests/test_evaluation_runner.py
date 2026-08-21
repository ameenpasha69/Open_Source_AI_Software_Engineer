from pathlib import Path

import pytest
from app.evaluation.runner import EvalTaskRunner
from app.evaluation.task_loader import load_tasks

from tests.conftest import FakeLLMProvider

_REAL_TASKS_DIR = Path(__file__).resolve().parents[2] / "evals" / "tasks"


@pytest.fixture
def calc_task():
    return next(t for t in load_tasks(_REAL_TASKS_DIR) if t.id == "task_001_calc_accumulate")


_PLAN_RESPONSE = '{"steps": ["Read calc.py", "Fix the bug", "Run tests"]}'
_CORRECT_PATCH_ACTION = (
    '{"thought": "fix it", "action": {"tool": "apply_patch", '
    '"input": {"path": "calc.py", "old_content": "total = item.price", "new_content": "total += item.price"}}, '
    '"finish": null}'
)
_WRONG_PATCH_ACTION = (
    '{"thought": "fix it", "action": {"tool": "apply_patch", '
    '"input": {"path": "calc.py", "old_content": "total = item.price", "new_content": "total = total"}}, '
    '"finish": null}'
)
_RUN_TESTS_ACTION = '{"thought": "verify", "action": {"tool": "run_tests", "input": {}}, "finish": null}'
_FINISH_RESPONSE = '{"thought": "done", "action": null, "finish": {"answer": "fixed", "root_cause": "accumulation bug"}}'


def _make_runner(db_session, tmp_path, llm, embedding_provider) -> EvalTaskRunner:
    return EvalTaskRunner(
        session=db_session,
        llm=llm,
        embedding_provider=embedding_provider,
        vector_index_dir=tmp_path / "vector_indexes",
        work_dir=tmp_path / "runs",
        max_iterations=6,
        chunk_max_lines=200,
        chunk_overlap_lines=20,
        max_indexable_file_size_bytes=1_000_000,
    )


async def test_correct_fix_scores_as_success(db_session, tmp_path, fake_embedding_provider, calc_task):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _CORRECT_PATCH_ACTION, _RUN_TESTS_ACTION, _FINISH_RESPONSE])
    runner = _make_runner(db_session, tmp_path, llm, fake_embedding_provider)

    result = await runner.run(calc_task)

    assert result.success is True
    assert result.first_attempt_success is True
    assert result.regressed_tests == []
    assert result.agent_status == "done"
    assert result.baseline.failed_tests == ["test_calc.py::test_calculate_total_multiple_items"]
    assert result.final.failed_tests == []
    assert result.modified_files == ["calc.py"]
    assert result.retrieved_expected_file is True  # apply_patch's own output includes "path": "calc.py"


async def test_wrong_fix_that_also_regresses_is_scored_as_failure_with_regression(
    db_session, tmp_path, fake_embedding_provider, calc_task
):
    # "total = total" is a no-op assignment, so total never leaves 0 — this
    # doesn't fix the multi-item case *and* breaks the single-item case that
    # was passing at baseline (it now also returns 0 instead of the price).
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _WRONG_PATCH_ACTION, _RUN_TESTS_ACTION, _FINISH_RESPONSE])
    runner = _make_runner(db_session, tmp_path, llm, fake_embedding_provider)

    result = await runner.run(calc_task)

    assert result.success is False
    assert result.first_attempt_success is False
    assert result.regressed_tests == ["test_calc.py::test_calculate_total_single_item"]


async def test_fix_that_breaks_a_previously_passing_test_is_flagged_as_regression(
    db_session, tmp_path, fake_embedding_provider, calc_task
):
    # Deletes the accumulation entirely, which breaks the previously-passing
    # single-item and empty-order tests too, not just failing to fix the bug.
    breaking_patch_action = (
        '{"thought": "fix it", "action": {"tool": "apply_patch", '
        '"input": {"path": "calc.py", "old_content": "def calculate_total(items):\\n    '
        '\\"\\"\\"Sum up the price of every item in the order.\\"\\"\\"\\n    total = 0\\n    '
        'for item in items:\\n        total = item.price\\n    return total", '
        '"new_content": "def calculate_total(items):\\n    return None"}}, "finish": null}'
    )
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, breaking_patch_action, _RUN_TESTS_ACTION, _FINISH_RESPONSE])
    runner = _make_runner(db_session, tmp_path, llm, fake_embedding_provider)

    result = await runner.run(calc_task)

    assert result.success is False
    assert set(result.regressed_tests) == {
        "test_calc.py::test_calculate_total_single_item",
        "test_calc.py::test_calculate_total_empty_order",
    }


async def test_agent_never_patching_scores_as_failure(db_session, tmp_path, fake_embedding_provider, calc_task):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _FINISH_RESPONSE])
    runner = _make_runner(db_session, tmp_path, llm, fake_embedding_provider)

    result = await runner.run(calc_task)

    assert result.success is False
    assert result.modified_files == []
    assert result.verification_status == "not_applicable"
    assert result.regressed_tests == []


async def test_crashing_agent_is_scored_not_raised(db_session, tmp_path, fake_embedding_provider, calc_task, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("app.agents.runner.AgentRunner.run", _boom)
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE])
    runner = _make_runner(db_session, tmp_path, llm, fake_embedding_provider)

    result = await runner.run(calc_task)

    assert result.agent_status == "crashed"
    assert result.success is False
    assert "simulated crash" in result.error


async def test_run_creates_inspectable_repo_copy(db_session, tmp_path, fake_embedding_provider, calc_task):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _CORRECT_PATCH_ACTION, _RUN_TESTS_ACTION, _FINISH_RESPONSE])
    runner = _make_runner(db_session, tmp_path, llm, fake_embedding_provider)

    await runner.run(calc_task)

    patched = tmp_path / "runs" / calc_task.id / "repo" / "calc.py"
    assert patched.is_file()
    assert "total += item.price" in patched.read_text()
    # the original fixture template must be untouched — the runner works on a copy
    original = calc_task.repo_dir / "calc.py"
    assert "total = item.price" in original.read_text()
    assert "total += item.price" not in original.read_text()
