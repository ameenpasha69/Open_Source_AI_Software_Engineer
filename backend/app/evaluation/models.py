from pathlib import Path

from pydantic import BaseModel


class TaskDefinition(BaseModel):
    id: str
    description: str
    expected_file: str | None = None
    repo_dir: Path  # the fixture repo template, copied fresh for every run


class TestSnapshot(BaseModel):
    """A ground-truth pytest run performed by the evaluator itself, not
    trusted from the agent's own (possibly skipped, possibly never-called)
    run_tests calls — the evaluator needs an answer regardless of whether
    the agent chose to verify its own work."""

    __test__ = False  # not a pytest test class — this name just mirrors the domain

    exit_code: int
    passed_tests: list[str]
    failed_tests: list[str]


class TaskResult(BaseModel):
    task_id: str
    agent_status: str
    verification_status: str
    success: bool
    first_attempt_success: bool
    regressed_tests: list[str]
    iterations: int
    tool_call_count: int
    duration_seconds: float
    modified_files: list[str]
    retrieved_expected_file: bool | None  # None when the task has no expected_file annotated
    baseline: TestSnapshot
    final: TestSnapshot
    error: str | None = None


class EvalReport(BaseModel):
    config: dict[str, str]
    results: list[TaskResult]

    @property
    def task_count(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.task_count if self.task_count else 0.0

    @property
    def first_attempt_count(self) -> int:
        return sum(1 for r in self.results if r.first_attempt_success)

    @property
    def first_attempt_rate(self) -> float:
        return self.first_attempt_count / self.task_count if self.task_count else 0.0

    @property
    def average_iterations(self) -> float:
        return sum(r.iterations for r in self.results) / self.task_count if self.task_count else 0.0

    @property
    def average_duration_seconds(self) -> float:
        return sum(r.duration_seconds for r in self.results) / self.task_count if self.task_count else 0.0

    @property
    def average_tool_calls(self) -> float:
        return sum(r.tool_call_count for r in self.results) / self.task_count if self.task_count else 0.0

    @property
    def regression_count(self) -> int:
        return sum(1 for r in self.results if r.regressed_tests)

    @property
    def regression_rate(self) -> float:
        return self.regression_count / self.task_count if self.task_count else 0.0

    @property
    def retrieval_hit_rate(self) -> float | None:
        scored = [r for r in self.results if r.retrieved_expected_file is not None]
        if not scored:
            return None
        return sum(1 for r in scored if r.retrieved_expected_file) / len(scored)
