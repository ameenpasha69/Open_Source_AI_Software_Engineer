"""Runs the evaluation suite against a real (or overridden) local model.

Usage (from the project root):

    source .venv/bin/activate
    PYTHONPATH=backend python3 -m app.evaluation.run
    PYTHONPATH=backend python3 -m app.evaluation.run --llm-model qwen2.5-coder:14b --max-iterations 12
    PYTHONPATH=backend python3 -m app.evaluation.run --output evals/runs/latest.json

This is the honest, real end of "evaluation framework": every task actually
runs the full agent loop against a real local LLM (or whatever LLM_PROVIDER
is configured), indexing a fresh copy of the fixture repo and taking its own
independent pytest snapshots before and after — it does not trust the
agent's self-reported test results, since the whole point is to catch cases
where the agent claims success without having verified anything.
"""

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.database.session import create_sqlite_engine, get_session_factory
from app.embeddings.factory import build_embedding_provider
from app.evaluation.models import EvalReport
from app.evaluation.report import format_report
from app.evaluation.runner import EvalTaskRunner
from app.evaluation.task_loader import load_tasks
from app.llm.factory import build_llm_provider
from app.observability.logging import configure_logging

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Local AI Software Engineer evaluation suite.")
    parser.add_argument("--tasks-dir", type=Path, default=_PROJECT_ROOT / "evals" / "tasks")
    parser.add_argument("--llm-model", type=str, default=None, help="Override LLM_MODEL for this run.")
    parser.add_argument(
        "--embedding-model", type=str, default=None, help="Override EMBEDDING_MODEL for this run."
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Override MAX_AGENT_ITERATIONS.")
    parser.add_argument("--output", type=Path, default=None, help="Write the full JSON report here too.")
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    overrides = {
        k: v
        for k, v in {
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "max_agent_iterations": args.max_iterations,
        }.items()
        if v is not None
    }
    settings = settings.model_copy(update=overrides)
    configure_logging(settings.log_level, settings.log_format)

    llm = build_llm_provider(settings)
    embedding_provider = build_embedding_provider(settings)

    if not await llm.health_check():
        print(f"LLM backend unreachable or model '{settings.llm_model}' not pulled — aborting.", file=sys.stderr)
        return 1
    if not await embedding_provider.health_check():
        print(
            f"Embedding backend unreachable or model '{settings.embedding_model}' not pulled — aborting.",
            file=sys.stderr,
        )
        return 1

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    runs_dir = _PROJECT_ROOT / "evals" / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    vector_index_dir = runs_dir / "vector_indexes"

    engine = create_sqlite_engine(f"sqlite:///{runs_dir / 'eval.db'}")
    session = get_session_factory(engine)()

    tasks = load_tasks(args.tasks_dir)
    if not tasks:
        print(f"No tasks found under {args.tasks_dir}", file=sys.stderr)
        return 1

    task_runner = EvalTaskRunner(
        session=session,
        llm=llm,
        embedding_provider=embedding_provider,
        vector_index_dir=vector_index_dir,
        work_dir=runs_dir,
        max_iterations=settings.max_agent_iterations,
        chunk_max_lines=settings.chunk_max_lines,
        chunk_overlap_lines=settings.chunk_overlap_lines,
        max_indexable_file_size_bytes=settings.max_indexable_file_size_bytes,
    )

    results = []
    for task in tasks:
        print(f"Running {task.id}...", file=sys.stderr)
        results.append(await task_runner.run(task))

    session.close()

    report = EvalReport(
        config={
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "max_agent_iterations": str(settings.max_agent_iterations),
        },
        results=results,
    )
    print()
    print(format_report(report))
    print()
    print(f"Full run artifacts (patched repos, SQLite DB): {runs_dir}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
        print(f"JSON report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
