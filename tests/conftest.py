from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = os.environ.get("DATABASE_URL_TEST")


@pytest.fixture(scope="session")
def db_url() -> Iterator[str]:
    if not TEST_DB:
        pytest.skip("DATABASE_URL_TEST not set (integration tests need PostgreSQL + pgvector)")
    env = {**os.environ, "DATABASE_URL": TEST_DB}
    run = lambda *a: subprocess.run(  # noqa: E731
        [sys.executable, "-m", "alembic", *a], cwd=ROOT, env=env, check=True,
        capture_output=True, text=True,
    )
    run("downgrade", "base")
    run("upgrade", "head")
    yield TEST_DB


@pytest.fixture()
def db(db_url: str):
    """A session on a clean database (all tables truncated before the test)."""
    from sqlalchemy import text

    from researchmind.db.session import get_engine, get_sessionmaker

    engine = get_engine(db_url)
    with engine.begin() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public' "
                 "AND tablename NOT IN ('alembic_version') AND tablename NOT LIKE 'checkpoint%'")
        ).scalars().all()
        if tables:
            conn.execute(text("TRUNCATE " + ", ".join(tables) + " CASCADE"))
    session = get_sessionmaker(db_url)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch):
    from researchmind import config

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("RERANKER_MODEL", "none")
    if TEST_DB:
        monkeypatch.setenv("DATABASE_URL", TEST_DB)
    config.get_settings.cache_clear()
    s = config.get_settings()
    s.ensure_dirs()
    yield s
    config.get_settings.cache_clear()
