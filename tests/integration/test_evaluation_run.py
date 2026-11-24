"""The benchmark runner end-to-end on a tiny QASPER-format fixture with a scripted LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.fixtures.agent_script import ScriptedLLM, claim_ids, passage_ids

pytestmark = pytest.mark.integration
FIX = Path(__file__).resolve().parents[1] / "fixtures" / "qasper_sample.json"


def llm():
    def baseline(p):
        ids = passage_ids(p)
        ev = [i for i in ids if "AIDA" in p.split(f'id="{i}"')[1][:300]]
        return {"answer": "AIDA-CoNLL and WNED-WIKI", "unanswerable": False, "evidence_ids": ev[:1]}

    def grade(p):
        return {"grades": [{"evidence_id": e, "relevance": "high", "aspects": ["direct answer"]}
                           for e in passage_ids(p)]}

    def synth(p):
        return {"title": "t", "sections": [{"heading": "Answer", "claims": [
            {"text": "They evaluate on AIDA-CoNLL and WNED-WIKI.", "claim_type": "reported_finding",
             "evidence_ids": passage_ids(p)[:1]}]}]}

    handlers = {
        "analyze": lambda p: {"intent": "single_paper", "complexity": "simple", "key_terms": []},
        "grade": grade, "critique": lambda p: {"issues": []}, "synthesize": synth,
        "verify": lambda p: {"verdicts": [{"claim_id": c, "support": "supported"} for c in claim_ids(p)]},
        "unknown": baseline,
    }
    return ScriptedLLM(handlers)


def test_evaluation_series(db, tmp_settings, tmp_path, monkeypatch):
    from evaluation import report, run
    from researchmind import services

    fake = llm()
    monkeypatch.setattr(services.Services, "provider", property(lambda self: fake))
    cfg = {"name": "fixture", "split": "test", "n_papers": 1, "max_questions_per_paper": 3, "seed": 1,
           "k": 4, "systems": ["vector", "hybrid", "hybrid_rerank", "agentic", "agentic_single_round"]}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    assert run.main([str(cfg_path), "--out", str(out), "--data", str(FIX)]) == 0
    rows = [json.loads(x) for x in (out / "fixture" / "runs" / "hybrid.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["status"] == "ok" for r in rows)
    q1 = next(r for r in rows if r["question_id"] == "q1")
    assert q1["recall@3"] == 1.0 and q1["answer_f1"] > 0.5
    # re-running skips completed pairs
    assert run.main([str(cfg_path), "--out", str(out), "--data", str(FIX)]) == 0
    assert len((out / "fixture" / "runs" / "hybrid.jsonl").read_text().splitlines()) == 3
    assert report.main([str(out / "fixture")]) == 0
    summary = (out / "fixture" / "summary.md").read_text()
    assert "agentic" in summary and re.search(r"\| hybrid \| 2 \|", summary)
