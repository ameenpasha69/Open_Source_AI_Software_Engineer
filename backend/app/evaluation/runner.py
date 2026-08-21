import json
import shutil
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.agents.runner import AgentRunner
from app.config.settings import Settings
from app.database.models import Repository
from app.embeddings.base import EmbeddingProvider
from app.evaluation.models import TaskDefinition, TaskResult, TestSnapshot
from app.evaluation.test_snapshot import take_test_snapshot
from app.llm.base import LLMProvider
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.retrieval.indexer import RepositoryIndexer
from app.tools.registry_factory import build_tool_registry

_EVAL_TEST_COMMAND = ["python3", "-m", "pytest"]


class EvalTaskRunner:
    """Runs one TaskDefinition end to end: copy the fixture repo to an
    inspectable working directory, snapshot its tests (ground truth,
    independent of anything the agent does), index it, run the real agent
    loop, snapshot tests again, and score the result. All fixture tasks are
    Python, so the test command is fixed rather than auto-detected —
    detection itself is already covered by its own tests elsewhere.
    """

    def __init__(
        self,
        session: Session,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        vector_index_dir: Path,
        work_dir: Path,
        max_iterations: int,
        chunk_max_lines: int,
        chunk_overlap_lines: int,
        max_indexable_file_size_bytes: int,
    ):
        self._session = session
        self._llm = llm
        self._embedding_provider = embedding_provider
        self._vector_index_dir = vector_index_dir
        self._work_dir = work_dir
        self._max_iterations = max_iterations
        self._chunk_max_lines = chunk_max_lines
        self._chunk_overlap_lines = chunk_overlap_lines
        self._max_indexable_file_size_bytes = max_indexable_file_size_bytes
        self._registry_settings = Settings(
            _env_file=None,
            max_indexable_file_size_bytes=max_indexable_file_size_bytes,
            data_dir=vector_index_dir.parent,
        )

    async def run(self, task: TaskDefinition) -> TaskResult:
        repo_copy = self._work_dir / task.id / "repo"
        if repo_copy.exists():
            shutil.rmtree(repo_copy)
        shutil.copytree(task.repo_dir, repo_copy)

        baseline = await take_test_snapshot(repo_copy)
        repository_id = self._index(task, repo_copy)
        await EmbeddingPipeline(self._session, self._embedding_provider, self._vector_index_dir).sync(repository_id)
        tool_registry = build_tool_registry(self._session, self._registry_settings, self._embedding_provider)
        runner = AgentRunner(self._session, self._llm, tool_registry, self._max_iterations)

        started = time.monotonic()
        try:
            state = await runner.run(repository_id, task.description)
        except Exception as exc:  # noqa: BLE001 - a crashed task is a scored failure, not an aborted eval run
            final = await take_test_snapshot(repo_copy)
            return TaskResult(
                task_id=task.id,
                agent_status="crashed",
                verification_status="not_applicable",
                success=final.exit_code == 0,
                first_attempt_success=False,
                regressed_tests=_regressions(baseline, final),
                iterations=0,
                tool_call_count=0,
                duration_seconds=round(time.monotonic() - started, 3),
                modified_files=[],
                retrieved_expected_file=None,
                baseline=baseline,
                final=final,
                error=str(exc),
            )
        duration = round(time.monotonic() - started, 3)

        final = await take_test_snapshot(repo_copy)
        success = final.exit_code == 0
        apply_patch_successes = sum(
            1 for tc in state.tool_calls if tc.tool_name == "apply_patch" and tc.result.success
        )

        return TaskResult(
            task_id=task.id,
            agent_status=str(state.status.value),
            verification_status=state.verification_status,
            success=success,
            first_attempt_success=success and apply_patch_successes == 1,
            regressed_tests=_regressions(baseline, final),
            iterations=state.iteration,
            tool_call_count=len(state.tool_calls),
            duration_seconds=duration,
            modified_files=state.modified_files,
            retrieved_expected_file=_check_retrieved(state, task.expected_file),
            baseline=baseline,
            final=final,
            error=state.error,
        )

    def _index(self, task: TaskDefinition, repo_copy: Path) -> str:
        indexer = RepositoryIndexer(
            session=self._session,
            chunk_max_lines=self._chunk_max_lines,
            chunk_overlap_lines=self._chunk_overlap_lines,
            max_file_size_bytes=self._max_indexable_file_size_bytes,
        )
        result = indexer.index(repo_copy, name=task.id)

        repository = self._session.get(Repository, result.repository_id)
        repository.test_command_json = json.dumps(_EVAL_TEST_COMMAND)
        self._session.commit()

        return result.repository_id


def _regressions(baseline: TestSnapshot, final: TestSnapshot) -> list[str]:
    return [t for t in baseline.passed_tests if t not in final.passed_tests]


def _check_retrieved(state, expected_file: str | None) -> bool | None:
    if expected_file is None:
        return None
    return any(
        tool_call.result.output and expected_file in json.dumps(tool_call.result.output)
        for tool_call in state.tool_calls
    )
