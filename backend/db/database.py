from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vienko")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)

Base = declarative_base()

logger = logging.getLogger(__name__)


def _sync_missing_columns() -> None:
    """Best-effort, dev-grade schema sync: ADD COLUMN for anything present
    in the SQLAlchemy models but missing from the actual database table.

    Why this exists: Base.metadata.create_all() (below) only creates
    tables that don't already exist -- it never alters an existing table,
    so a column added to a model AFTER that table was first created is
    silently never added to a database that's already running. Every
    query touching that column then fails with "column does not exist"
    (a real 500 this project hit in practice: fl_rounds/dp_enabled/etc.
    were added to DatasetAccessRequest, and secagg_enabled/etc. to
    FederatedExperiment, well after those tables already existed in a
    live Postgres volume from earlier testing).

    This is NOT a substitute for a real migration tool (Alembic) -- it
    can't handle column removal, renames, or type changes, and it can't
    backfill meaningful data into a new column, only a static default.
    For a project with real users and real migrations to reason about,
    replace this with Alembic. For local/dev use, this means "add a field
    to a model, restart the backend" just works, without needing to
    manually wipe and reseed the database every time -- which is what
    would otherwise have to happen after every single model change.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all() already handles brand-new tables entirely
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                # Each column gets its OWN savepoint (nested transaction).
                # In Postgres, once any statement in a transaction errors,
                # the whole transaction is marked aborted and every
                # subsequent statement fails too -- a plain Python
                # try/except around each ALTER would NOT actually isolate
                # failures the way it looks like it does; one bad column
                # would silently doom every column after it in the same
                # loop. A savepoint per column means a failure rolls back
                # to just that savepoint, not the whole outer transaction.
                try:
                    with conn.begin_nested():
                        
                        col_type = column.type.compile(dialect=conn.dialect)
                        default_sql = ""
                        nullable_sql = ""

                        if not column.nullable:
                            default_val = _static_default(column)

                            if default_val is not None:
                                default_sql = f" DEFAULT {default_val}"
                                nullable_sql = " NOT NULL"
                            else:
                     # Existing rows cannot safely receive a NOT NULL value
                        # without a default or an explicit data migration.
                                nullable_sql = ""

                        stmt = (
                                f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" '
                                f"{col_type}{default_sql}{nullable_sql}"
                            )

                        conn.execute(text(stmt))
                    logger.warning(
                        "db schema sync: added missing column %s.%s (this should only ever "
                        "happen once per new column, right after a deploy)",
                        table.name, column.name,
                    )
                except Exception:
                    logger.exception(
                        "db schema sync: failed to add column %s.%s -- leaving it out; "
                        "queries touching this column will keep failing until this is "
                        "resolved manually",
                        table.name, column.name,
                    )


def _static_default(column) -> str | None:
    """Renders a column's Python-side `default=` as a SQL literal, for
    backfilling existing rows when adding a NOT NULL column. Returns None
    if there's no simple scalar default to use.
    """
    if column.default is None or not getattr(column.default, "is_scalar", False):
        return None
    val = column.default.arg
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return "'" + val.replace("'", "''") + "'"
    return None


def init_db() -> None:
    """Create tables if they do not exist, then add any columns that are
    missing from tables that DO already exist (see _sync_missing_columns).
    """
    from db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _sync_missing_columns()


def get_session() -> Generator[Session, None, None]:
    """FastAPI-friendly DB session dependency."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for scripts: `with session_scope() as s:`."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

