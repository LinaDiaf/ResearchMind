"""Run a benchmark series.

    python -m evaluation.run evaluation/configs/smoke.json [--out results/eval]

Resumable: (question, system) pairs already present in the output are skipped, so the same
command can be re-run after an interruption. Raw outputs are stored for every question.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import traceback
import uuid
from pathlib import Path

from evaluation import metrics as M
from evaluation.qasper import QasperPaper, load, sample
from evaluation.systems import AgenticSystem, RetrievalBaseline, SystemOutput

KS = (1, 3, 5, 8)


def build_systems(names: list[str], svc, k: int):
    from researchmind.workspace.investigations import InvestigationRunner

    def runner_factory(budget):
        return InvestigationRunner(svc.sessionmaker, svc.engine, svc.provider, budget, svc.checkpointer)

    base = svc.budget
    table = {
        "vector": lambda: RetrievalBaseline("vector", svc.engine, svc.provider, strategy="dense", rerank=False, k=k),
        "hybrid": lambda: RetrievalBaseline("hybrid", svc.engine, svc.provider, strategy="hybrid", rerank=False, k=k),
        "hybrid_rerank": lambda: RetrievalBaseline("hybrid_rerank", svc.engine, svc.provider, strategy="hybrid", rerank=True, k=k),
        "agentic": lambda: AgenticSystem("agentic", runner_factory, base.model_copy(update={"top_k": k})),
        "agentic_single_round": lambda: AgenticSystem(
            "agentic_single_round", runner_factory,
            base.model_copy(update={"top_k": k, "max_rounds_per_subquestion": 1})),
        "agentic_no_critique": lambda: AgenticSystem(
            "agentic_no_critique", runner_factory, base.model_copy(update={"top_k": k, "enable_critique": False})),
    }
    unknown = set(names) - set(table)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    return [table[n]() for n in names]


def ingest(svc, project_id: uuid.UUID, papers: list[QasperPaper]) -> dict[str, uuid.UUID]:
    from researchmind.discovery.models import PaperRecord

    ids = {}
    for i, p in enumerate(papers, 1):
        rec = PaperRecord(title=p.title, abstract=p.abstract, source="qasper", arxiv_id=p.paper_id)
        res = svc.library.ingest_structured(project_id, p.title, p.sections, abstract=p.abstract,
                                            source="qasper", record=rec, reindex=False)
        ids[p.paper_id] = res.paper_id
        print(f"[ingest {i}/{len(papers)}] {res.status} chunks={res.n_chunks} {p.title[:80]}", flush=True)
    svc.library.reindex(project_id)
    return ids


def score(q, out: SystemOutput) -> dict:
    flags = M.relevant_flags(out.ranked_passages, q.evidence)
    n_gold = len(q.evidence)
    row = {f"recall@{k}": M.recall_at_k(flags, n_gold, k) for k in KS}
    row.update({
        "mrr": M.mrr(flags) if n_gold else float("nan"),
        "ndcg@8": M.ndcg_at_k(flags, n_gold, 8),
        "citation_precision": M.citation_precision(out.cited_passages, q.evidence) if n_gold else float("nan"),
        "citation_hit": (float(any(M.relevant_flags(out.cited_passages, q.evidence)))
                         if n_gold and not q.unanswerable else float("nan")),
        "answer_f1": M.answer_f1(out.answer, q.answers),
    })
    return row


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--data", default=None, help="local QASPER json (skips download)")
    args = ap.parse_args(argv)

    from researchmind.agent.prompts import PROMPT_VERSION
    from researchmind.services import build_services

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out) / cfg["name"]
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    svc = build_services()

    papers = sample(load(cfg["split"], svc.settings.cache_dir / "qasper",
                         Path(args.data) if args.data else None),
                    cfg["n_papers"], cfg["max_questions_per_paper"], cfg["seed"])
    questions = [q for p in papers for q in p.questions]
    meta = {**cfg, "n_questions": len(questions),
            "n_answerable_with_evidence": sum(1 for q in questions if q.evidence and not q.unanswerable),
            "llm": f"{svc.settings.llm_provider}:{svc.settings.llm_model}",
            "embedding": f"{svc.settings.embedding_provider}:{svc.settings.embedding_model}",
            "reranker": svc.settings.reranker_model, "budget": svc.budget.model_dump(),
            "prompt_version": PROMPT_VERSION, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    with open(out_dir / "questions.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q.__dict__) + "\n")

    project_id = svc.library.create_project(f"eval-{cfg['name']}", "QASPER evaluation corpus")
    paper_ids = ingest(svc, project_id, papers)
    systems = build_systems(cfg["systems"], svc, cfg.get("k", 8))

    total = len(questions) * len(systems)
    done_n = ok = 0
    for system in systems:
        path = out_dir / "runs" / f"{system.name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["question_id"] for line in path.read_text().splitlines() if line.strip()}
        with open(path, "a") as f:
            for q in questions:
                done_n += 1
                if q.question_id in done:
                    continue
                row = {"question_id": q.question_id, "paper_id": q.paper_id, "system": system.name,
                       "question": q.question, "gold_unanswerable": q.unanswerable,
                       "gold_answers": q.answers, "n_gold_evidence": len(q.evidence)}
                try:
                    out = system.run(project_id, paper_ids[q.paper_id], q.question)
                    row.update(status="ok", answer=out.answer, pred_unanswerable=out.unanswerable,
                               llm_calls=out.llm_calls, prompt_tokens=out.prompt_tokens,
                               completion_tokens=out.completion_tokens, latency_s=round(out.latency_s, 2),
                               retrieval_rounds=out.retrieval_rounds, extra=out.extra,
                               n_ranked=len(out.ranked_passages), n_cited=len(out.cited_passages),
                               **{k: _clean(v) for k, v in score(q, out).items()})
                    ok += 1
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               trace=traceback.format_exc()[-2000:])
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                print(f"[{done_n}/{total}] {system.name:<22} {row['status']:<6} "
                      f"{q.question[:70]}", flush=True)
    print(f"done: {ok} ok this session, results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
