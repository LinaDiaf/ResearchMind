from __future__ import annotations

import pytest

from researchmind.cli import main
from tests.fixtures.synthetic_papers import build_corpus

pytestmark = pytest.mark.integration


def test_cli_ingest_folder(db, tmp_settings, tmp_path, capsys):
    build_corpus(tmp_path / "pdfs")
    assert main(["ingest", "CLI project", str(tmp_path / "pdfs")]) == 0
    out = capsys.readouterr().out
    assert out.count("ingested") == 3 and "indexed" in out
    assert main(["ingest", "CLI project", str(tmp_path / "pdfs" / "dense.pdf")]) == 0
    assert "duplicate" in capsys.readouterr().out
