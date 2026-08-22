"""Regression tests for the upgrade path.

`create_all()` creates missing tables and never missing columns, so adding a
field to an existing model used to break every installation that already had
a data/app.db — observed on a real database as
`no such column: agent_runs.session_id` right after the sessions feature
landed, while every test still passed because tests build their schema from
scratch.
"""

import datetime

from app.database.migrations import add_missing_columns
from app.database.models import AgentRun, Base, Repository
from app.database.session import create_sqlite_engine, get_session_factory
from sqlalchemy import inspect, text

# agent_runs exactly as it was before sessions and token accounting existed.
# Written out rather than produced by dropping columns, because SQLite refuses
# to drop a column another constraint refers to — and because this is what an
# actually-old database contains.
_LEGACY_AGENT_RUNS = """
CREATE TABLE agent_runs (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    repository_id VARCHAR(32) NOT NULL,
    task TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    plan_json TEXT NOT NULL,
    final_answer TEXT,
    root_cause TEXT,
    iteration_count INTEGER NOT NULL,
    verification_status VARCHAR(32) NOT NULL,
    started_at DATETIME NOT NULL,
    finished_at DATETIME,
    error TEXT
)
"""

_NEW_COLUMNS = {
    "session_id",
    "prompt_tokens",
    "completion_tokens",
    "llm_call_count",
    "peak_prompt_tokens",
    "model",
}


def columns_of(engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def legacy_database(tmp_path, with_row: bool = True):
    """An engine whose agent_runs table predates the current models."""
    path = tmp_path / "legacy.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE agent_runs"))
        connection.execute(text(_LEGACY_AGENT_RUNS))
        if with_row:
            connection.execute(
                text(
                    "INSERT INTO agent_runs (id, repository_id, task, status, plan_json, "
                    "iteration_count, verification_status, started_at) "
                    "VALUES ('run1', 'repo1', 'do a thing', 'done', '[]', 3, 'verified', :now)"
                ),
                {"now": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)},
            )
    return engine, path


def test_a_fresh_database_needs_no_columns_added(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    assert add_missing_columns(engine) == []


def test_every_new_column_is_added_to_a_legacy_table(tmp_path):
    engine, _ = legacy_database(tmp_path)
    assert not _NEW_COLUMNS & columns_of(engine, "agent_runs")

    added = add_missing_columns(engine)

    assert {name.removeprefix("agent_runs.") for name in added} == _NEW_COLUMNS
    assert _NEW_COLUMNS <= columns_of(engine, "agent_runs")


def test_existing_rows_survive_and_stay_readable(tmp_path):
    engine, _ = legacy_database(tmp_path)
    add_missing_columns(engine)

    session = get_session_factory(engine)()
    run = session.get(AgentRun, "run1")
    assert run.task == "do a thing"
    assert run.iteration_count == 3
    # A NOT NULL int column has to arrive with a SQL default, or the existing
    # row comes back NULL and blows up loading as an int.
    assert run.prompt_tokens == 0
    assert run.llm_call_count == 0
    # Nullable columns get no default, and that's correct — nobody knows which
    # model answered a run recorded before the column existed.
    assert run.model is None
    assert run.session_id is None
    session.close()


def test_new_rows_still_write_normally_afterwards(tmp_path):
    engine, _ = legacy_database(tmp_path)
    add_missing_columns(engine)

    session = get_session_factory(engine)()
    repository = Repository(name="r", path=str(tmp_path / "r"))
    session.add(repository)
    session.flush()
    session.add(
        AgentRun(id="run2", repository_id=repository.id, task="t", status="done", prompt_tokens=42, model="m")
    )
    session.commit()

    assert session.get(AgentRun, "run2").prompt_tokens == 42
    session.close()


def test_it_is_idempotent(tmp_path):
    engine, _ = legacy_database(tmp_path)
    assert add_missing_columns(engine)
    assert add_missing_columns(engine) == []


def test_a_missing_table_is_left_to_create_all(tmp_path):
    # add_missing_columns only patches tables that exist; building a new table
    # in full stays create_all's job.
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE chat_messages"))

    assert add_missing_columns(engine) == []

    Base.metadata.create_all(engine)
    assert "chat_messages" in inspect(engine).get_table_names()


def test_opening_an_engine_upgrades_the_schema(tmp_path):
    # The whole point: no separate migrate step anyone has to remember.
    engine, path = legacy_database(tmp_path)
    engine.dispose()

    reopened = create_sqlite_engine(f"sqlite:///{path}")

    assert _NEW_COLUMNS <= columns_of(reopened, "agent_runs")
    assert get_session_factory(reopened)().get(AgentRun, "run1").task == "do a thing"
