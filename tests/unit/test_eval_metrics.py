import math
from pathlib import Path

from evaluation import metrics as M
from evaluation.qasper import load, sample

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "qasper_sample.json"


def test_retrieval_metrics():
    gold = ["alpha beta gamma delta", "epsilon zeta eta theta"]
    ranked = ["unrelated text", "x alpha beta gamma delta y", "epsilon zeta eta theta and more"]
    flags = M.relevant_flags(ranked, gold)
    assert flags == [set(), {0}, {1}]
    assert M.recall_at_k(flags, 2, 1) == 0 and M.recall_at_k(flags, 2, 3) == 1
    assert M.mrr(flags) == 0.5
    assert 0 < M.ndcg_at_k(flags, 2, 3) < 1
    assert math.isnan(M.recall_at_k(flags, 0, 3))
    assert M.citation_precision(ranked[:2], gold) == 0.5


def test_answer_f1_matches_squad_definition():
    assert M.answer_f1("The AIDA-CoNLL dataset", ["AIDA-CoNLL dataset"]) == 1.0
    assert M.token_f1("yes", "No") == 0.0
    assert 0 < M.answer_f1("AIDA-CoNLL", ["AIDA-CoNLL, WNED-WIKI"]) < 1


def test_binary_and_bootstrap():
    s = M.binary_scores([True, False, True], [True, True, False])
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    lo, hi = M.bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0])
    assert 0 <= lo <= hi <= 1


def test_qasper_loader():
    papers = load("test", Path("/nonexistent"), FIX)
    p = papers[0]
    assert [s[0] for s in p.sections] == ["Introduction", "Datasets", "Results"]
    q1, q2, q3 = p.questions
    assert q1.evidence == ["We evaluate on the AIDA-CoNLL dataset and on the WNED-WIKI dataset."]  # FLOAT removed
    assert q1.answers[0] == "AIDA-CoNLL, WNED-WIKI" and q2.answer_type == "abstractive"
    assert q3.unanswerable and not q1.unanswerable
    assert sample(papers, 5, 2, seed=1)[0].questions.__len__() == 2
