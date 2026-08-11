"""Alembic environment for contextgraph's own schema.

Three decisions here exist because this schema is a guest in someone else's
database:

**Its own version table.** ``cg_alembic_version`` rather than the default
``alembic_version``. A host application that already runs Alembic owns that
name; sharing it means whichever tool stamps last convinces the other that
migrations it has never seen are applied. Two independent histories need two
independent stamps.

**Autogenerate is fenced to ``cg_`` objects.** Without ``include_object``, an
autogenerate run against a host database reads every table it does not know
about as "dropped" and emits ``drop_table`` for the host's data. The filter is
the difference between a diff and an outage.

**The URL is coerced to asyncpg.** ``DATABASE_URL`` in the wild is whatever the
host's ORM, its cloud provider, or ``psql`` wanted; libpq-shaped URLs
(``postgres://``, ``?sslmode=require``) reach asyncpg as a connect-time
TypeError that reads like a driver bug rather than a URL problem.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from contextgraph.models import Base

config = context.config

# Guarded, unlike the stock template: a minimal alembic.ini carrying only
# [alembic] is a reasonable thing for a host to write, and fileConfig raises
# KeyError('formatters') on one — a logging-config crash before any migration
# runs, which reads as a broken migration.
if config.config_file_name is not None and config.get_section("formatters"):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

VERSION_TABLE = "cg_alembic_version"

_ASYNC_DRIVER = "postgresql+asyncpg"
_SYNC_SCHEME = "postgresql"

# libpq spellings that mean "the host DB is Postgres", including the heroku-era
# ``postgres://`` that SQLAlchemy stopped accepting in 1.4.
_PG_SCHEMES = frozenset(
    {
        "postgres",
        "postgresql",
        "postgresql+asyncpg",
        "postgresql+psycopg",
        "postgresql+psycopg2",
        "postgresql+pg8000",
    }
)

# asyncpg has no ``sslmode``; SQLAlchemy's asyncpg dialect takes ``ssl``.
# ``allow``/``prefer`` have no asyncpg equivalent and are dropped rather than
# upgraded to ``require`` — silently tightening TLS is still lying about the
# connection.
_SSLMODE_TO_SSL = {
    "require": "require",
    "verify-ca": "verify-ca",
    "verify-full": "verify-full",
}
_DROPPED_LIBPQ_PARAMS = frozenset({"sslmode", "channel_binding", "gssencmode"})


def _coerce_url(url: str, *, driver: str) -> str:
    """Rewrite a Postgres URL onto ``driver``, translating libpq-only params."""
    parts = urlsplit(url)
    if parts.scheme not in _PG_SCHEMES:
        raise ValueError(
            f"contextgraph requires a PostgreSQL URL; got scheme {parts.scheme!r}"
        )

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    sslmode = query.get("sslmode")
    for key in _DROPPED_LIBPQ_PARAMS:
        query.pop(key, None)
    if driver == _ASYNC_DRIVER and sslmode in _SSLMODE_TO_SSL and "ssl" not in query:
        query["ssl"] = _SSLMODE_TO_SSL[sslmode]

    return urlunsplit(
        (driver, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _database_url(*, driver: str) -> str:
    raw = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url", None)
    if not raw:
        raise RuntimeError(
            "No database URL. Set DATABASE_URL, or sqlalchemy.url in alembic.ini."
        )
    return _coerce_url(raw, driver=driver)


# Objects 0001 creates that models.py deliberately does not declare.
# Autogenerate diffs the database against the ORM and reads anything it cannot
# find there as dropped. Measured against a freshly migrated database: without
# these two sets it proposed drop_column on both embedding columns and
# drop_index on all seven migration-only indexes — a "no-op" revision that
# removes vector search and the hot read path.
_MIGRATION_OWNED_COLUMNS = frozenset({"embedding"})
_MIGRATION_OWNED_INDEXES = frozenset(
    {
        "ix_cg_edges_graph_live",
        "ix_cg_runs_tenant",
        "ix_cg_sources_tenant",
        "ix_cg_segments_tenant",
        "ix_cg_nodes_tenant",
        "ix_cg_edges_tenant",
        "ix_cg_changesets_tenant",
        "ix_cg_nodes_embedding_hnsw",
        "ix_cg_segments_embedding_hnsw",
    }
)


def _include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    if type_ == "table":
        # The cg_ prefix would otherwise sweep in our own version table.
        return bool(name and name.startswith("cg_") and name != VERSION_TABLE)
    if type_ == "column" and name in _MIGRATION_OWNED_COLUMNS:
        return False
    if type_ == "index" and name in _MIGRATION_OWNED_INDEXES:
        return False
    # Remaining indexes, constraints and columns ride on their table's
    # decision; anything hanging off an excluded table is already unreachable.
    table = getattr(obj, "table", None)
    if table is not None and table.name is not None:
        return table.name.startswith("cg_")
    return True


def _configure(**kwargs: Any) -> None:
    context.configure(
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        include_object=_include_object,
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    # Emitted as SQL text, so the sync scheme is used: the DDL is identical and
    # asyncpg's paramstyle has nothing to contribute to a script nobody runs
    # through a driver.
    _configure(
        url=_database_url(driver=_SYNC_SCHEME),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url(driver=_ASYNC_DRIVER)

    engine = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        # One connection, discarded after. A pool outliving the process is how
        # `alembic upgrade` ends up hanging on exit instead of returning.
        poolclass=pool.NullPool,
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    # Callers embedding migrations (test fixtures, an app's bootstrap) pass an
    # open connection through config.attributes so the schema is created inside
    # a transaction they control.
    provided = config.attributes.get("connection")
    if provided is not None:
        _run_migrations(provided)
        return
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
