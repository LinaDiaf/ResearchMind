"""Smoke tests of every Streamlit page against a real database (no LLM call on page load)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
APP = Path(__file__).resolve().parents[2] / "app" / "Home.py"

VIEWS = ["views/dashboard.py", "views/projects.py", "views/library.py", "views/assistant.py",
         "views/comparison.py", "views/evidence.py", "views/knowledge_graph.py", "views/settings.py"]


@pytest.fixture()
def app_env(db, tmp_settings, tmp_path, monkeypatch):
    from researchmind import config
    from researchmind.db.session import get_sessionmaker
    from researchmind.embeddings import HashingEmbedder
    from researchmind.ingestion.service import Library
    from tests.fixtures.synthetic_papers import build_corpus

    monkeypatch.setenv("LOG_JSON", "false")
    config.get_settings.cache_clear()
    lib = Library(get_sessionmaker(tmp_settings.database_url), HashingEmbedder(tmp_settings.embedding_dim),
                  config.get_settings())
    pid = lib.create_project("UI project")
    for p in build_corpus(tmp_path / "pdfs").values():
        lib.ingest_pdf(pid, p, reindex=False)
    lib.reindex(pid)
    return pid


@pytest.mark.parametrize("view", VIEWS)
def test_page_renders_without_error(app_env, view):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page(view)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_projects_page_creates_project(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page("views/projects.py")
    at.run()
    at.text_input[0].input("Second project")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("Second project" in s.value for s in at.success)


def test_report_and_evidence_pages_show_completed_investigation(app_env, tmp_settings):
    from streamlit.testing.v1 import AppTest

    from researchmind.config import AgentBudgetSettings
    from researchmind.db.session import get_sessionmaker
    from researchmind.embeddings import HashingEmbedder
    from researchmind.retrieval.engine import RetrievalEngine
    from researchmind.workspace.investigations import (
        InvestigationRunner,
        make_postgres_checkpointer,
    )
    from tests.integration.test_investigations import scripted

    sf = get_sessionmaker(tmp_settings.database_url)
    engine = RetrievalEngine(sf, HashingEmbedder(tmp_settings.embedding_dim), None, tmp_settings.index_dir)
    runner = InvestigationRunner(sf, engine, scripted(), AgentBudgetSettings(),
                                 make_postgres_checkpointer(tmp_settings.psycopg_url))
    inv = runner.create(app_env, "How do SPARTA-X and DenseQA compare?")
    report = runner.run(inv)

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page("views/assistant.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert report["title"] in [s.value for s in at.subheader]
    at.switch_page("views/evidence.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("retrieved passages" in c.value for c in at.caption)
