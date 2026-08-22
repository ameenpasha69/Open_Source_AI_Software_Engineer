"""Bringing an existing SQLite file up to the current model definitions.

`Base.metadata.create_all()` creates missing *tables*, and stops there — it
will never add a *column* to a table that already exists. That is invisible
while every test builds its database from scratch, and breaks the moment a
real installation with an existing `data/app.db` picks up a release that
added a field: every query naming the new column fails with
`no such column`, which is how this module came to exist.

Deliberately not Alembic. This project's schema is one SQLite file that only
ever grows columns; a migration framework would add a dependency, a
versions directory, and a step to forget, to solve a problem that is one
`ALTER TABLE ... ADD COLUMN` per new field. What it does not handle —
dropping columns, changing types, backfilling — is also not something this
schema has ever needed. If that changes, this is the point to swap in a real
migration tool rather than to extend this.
"""

import logging

from sqlalchemy import Engine, inspect, text
from sqlalchemy.schema import Column

from app.database.models import Base

logger = logging.getLogger(__name__)


def add_missing_columns(engine: Engine) -> list[str]:
    """Add any column that the models declare and the database lacks.

    Idempotent: run it on every startup. Returns the qualified names of the
    columns it added, for logging and for tests to assert on.
    """
    added: list[str] = []

    with engine.begin() as connection:
        # Inspect the same connection the DDL runs on. Reflecting through
        # `inspect(engine)` instead takes a second pooled connection, whose
        # view of the schema can lag the one being altered — which shows up
        # as the nonsense pair "get_columns says the column is missing" and
        # "ALTER says duplicate column".
        inspector = inspect(connection)
        existing_tables = set(inspector.get_table_names())
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all() just built it, with every column
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                connection.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {_column_ddl(column, engine)}"))
                added.append(f"{table.name}.{column.name}")

    if added:
        logger.info("database schema updated", extra={"columns_added": added})
    return added


def _column_ddl(column: Column, engine: Engine) -> str:
    """DDL for one added column.

    Foreign-key clauses are deliberately omitted: SQLite only accepts a
    REFERENCES clause on an added column when the default is NULL, and it
    doesn't enforce foreign keys at all unless `PRAGMA foreign_keys=ON`.
    Carrying the constraint here would buy nothing and fail on some shapes.
    """
    parts = [column.name, column.type.compile(engine.dialect)]
    default = _default_literal(column)

    if default is not None:
        # Existing rows need a value. A Python-side default only applies to
        # rows this process inserts, so a NOT NULL column added without a SQL
        # default would leave every existing row unreadable.
        parts.append(f"DEFAULT {default}")
        if not column.nullable:
            parts.append("NOT NULL")
    elif not column.nullable:
        # SQLite rejects NOT NULL with no default on a populated table, and
        # there's no correct value to invent. Added nullable instead, which
        # keeps the database readable; the model's Python default still fills
        # new rows.
        logger.warning(
            "adding %s.%s as nullable: it is declared NOT NULL but has no constant default",
            column.table.name,
            column.name,
        )

    return " ".join(parts)


def _default_literal(column: Column) -> str | None:
    """The column's default as a SQL literal, when it's a constant.

    Callable defaults (`uuid4().hex`, `utcnow()`) produce a different value
    per row and can't be expressed as a table default — those columns are
    added without one.
    """
    if column.default is None or column.default.is_callable:
        return None
    value = column.default.arg
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return None
