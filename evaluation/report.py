"""Aggregate a benchmark series into summary.md / summary.json.

    python -m evaluation.report results/eval/qasper_smoke
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation import metrics as M

RETRIEVAL = ["recall@1", "recall@3", "recall@5", "recall@8", "mrr", "ndcg@8"]
ANSWER = ["answer_f1", "citation_precision", "citation_hit"]
COST = ["llm_calls", "prompt_tokens", "completion_tokens", "latency_s", "retrieval_rounds"]


def _fmt(x: float, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    answerable = [r for r in ok if not r["gold_unanswerable"] and r["n_gold_evidence"] > 0]
    out = {"n": len(rows), "ok": len(ok), "failed": len(rows) - len(ok),
           "n_answerable_with_evidence": len(answerable)}
    for m in RETRIEVAL + ANSWER:
        vals = [r.get(m) for r in (answerable if m != "answer_f1" else ok)]
        vals = [v for v in vals if v is not None]
        out[m] = {"mean": M.mean(vals), "ci95": M.bootstrap_ci(vals), "n": len(vals)}
    for m in COST:
        vals = [r.get(m) or 0 for r in ok]
        out[m] = {"mean": M.mean(vals), "n": len(vals)}
    out["unanswerable_detection"] = M.binary_scores(
        [bool(r["pred_unanswerable"]) for r in ok], [bool(r["gold_unanswerable"]) for r in ok])
    out["task_success"] = M.mean([1.0 if (r["answer_f1"] >= 0.5) else 0.0 for r in ok])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    args = ap.parse_args(argv)
    d = Path(args.series)
    cfg = json.loads((d / "config.json").read_text())
    systems = {}
    for path in sorted((d / "runs").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        systems[path.stem] = summarize(rows)
    order = [s for s in cfg["systems"] if s in systems]

    lines = [f"# Evaluation: `{cfg['name']}`", "",
             f"- QASPER {cfg['split']} split, {cfg['n_papers']} papers sampled with seed {cfg['seed']}, "
             f"{cfg['n_questions']} questions ({cfg['n_answerable_with_evidence']} answerable with gold evidence)",
             f"- LLM `{cfg['llm']}` · embeddings `{cfg['embedding']}` · reranker `{cfg['reranker']}` · k={cfg.get('k', 8)}",
             f"- prompt version {cfg['prompt_version']} · started {cfg['started']}",
             "- Retrieval and citation metrics use answerable questions with human gold evidence; "
             "a passage counts as relevant when it contains ≥50% of a gold paragraph's tokens.",
             "- Answer F1: token F1, max over annotators (official QASPER definition). "
             "95% bootstrap confidence intervals in brackets.", "",
             "## Retrieval", "",
             "| system | n | " + " | ".join(RETRIEVAL) + " |",
             "|---|---|" + "---|" * len(RETRIEVAL)]
    for s in order:
        r = systems[s]
        lines.append(f"| {s} | {r['n_answerable_with_evidence']} | "
                     + " | ".join(_fmt(r[m]["mean"]) for m in RETRIEVAL) + " |")
    lines += ["", "## Answers and citations", "",
              "| system | answer F1 | task success (F1≥0.5) | citation precision | citation hit | unanswerable P / R / F1 |",
              "|---|---|---|---|---|---|"]
    for s in order:
        r = systems[s]
        u = r["unanswerable_detection"]
        ci = r["answer_f1"]["ci95"]
        lines.append(f"| {s} | {_fmt(r['answer_f1']['mean'])} [{_fmt(ci[0], 2)}, {_fmt(ci[1], 2)}] | "
                     f"{_fmt(r['task_success'])} | {_fmt(r['citation_precision']['mean'])} | "
                     f"{_fmt(r['citation_hit']['mean'])} | {_fmt(u['precision'], 2)} / {_fmt(u['recall'], 2)} / "
                     f"{_fmt(u['f1'], 2)} (n={u['support']}) |")
    lines += ["", "## Cost", "", "| system | ok / failed | LLM calls | prompt tokens | completion tokens | latency (s) | retrieval rounds |",
              "|---|---|---|---|---|---|---|"]
    for s in order:
        r = systems[s]
        lines.append(f"| {s} | {r['ok']} / {r['failed']} | {_fmt(r['llm_calls']['mean'], 1)} | "
                     f"{_fmt(r['prompt_tokens']['mean'], 0)} | {_fmt(r['completion_tokens']['mean'], 0)} | "
                     f"{_fmt(r['latency_s']['mean'], 1)} | {_fmt(r['retrieval_rounds']['mean'], 1)} |")
    lines += ["", "_Generated from the raw per-question outputs in `runs/`. Small samples give wide "
              "intervals: differences inside overlapping intervals are not conclusive._"]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    (d / "summary.json").write_text(json.dumps(systems, indent=2, default=str))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
