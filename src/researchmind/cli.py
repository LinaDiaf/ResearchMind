"""Command-line interface (also used to run investigations on the evaluation machine)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path


def _services():
    from researchmind.services import build_services

    return build_services()


def _project_id(svc, name_or_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(name_or_id)
    except ValueError:
        return svc.library.create_project(name_or_id)


def cmd_init_db(_args) -> int:
    root = Path.cwd() if (Path.cwd() / "alembic.ini").exists() else Path(__file__).resolve().parents[2]
    return subprocess.call([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=root)


def cmd_ingest(args) -> int:
    svc = _services()
    pid = _project_id(svc, args.project)
    paths = [p for a in args.paths for p in (sorted(Path(a).glob("*.pdf")) if Path(a).is_dir() else [Path(a)])]
    for i, path in enumerate(paths, 1):
        res = svc.library.ingest_pdf(pid, path, reindex=False)
        print(f"[{i}/{len(paths)}] {res.status:<9} chunks={res.n_chunks:<4} {res.title[:90]}"
              + (f"  ({res.error})" if res.error else ""))
    print(f"indexed {svc.library.reindex(pid)} chunks for BM25")
    return 0


def cmd_search(args) -> int:
    svc = _services()
    records, errors = svc.discovery.search(args.query, args.sources, args.limit, args.year_min)
    for i, r in enumerate(records, 1):
        ids = r.doi or (f"arXiv:{r.arxiv_id}" if r.arxiv_id else "")
        print(f"{i:>2}. {r.title[:100]} ({r.year}) {ids} [{r.source}]"
              + (" PDF" if r.pdf_url else ""))
    for name, err in errors.items():
        print(f"   ! {name}: {err}", file=sys.stderr)
    if args.import_to:
        pid = _project_id(svc, args.import_to)
        chosen = records if args.import_all else [records[int(x) - 1] for x in args.pick]
        for r in chosen:
            res = svc.library.import_record(pid, r, download_pdf=True)
            print(f"   {res.status:<13} {res.title[:90]}")
    return 0


def _print_event(ev: dict) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {ev.get('node', ''):<11} {ev.get('message', '')}", flush=True)


def cmd_ask(args) -> int:
    from researchmind.synthesis.report import render_markdown

    svc = _services()
    pid = _project_id(svc, args.project)
    inv = svc.runner.create(pid, args.question)
    print(f"investigation {inv}")
    report = svc.runner.run(inv, on_event=_print_event)
    _emit(report, args, render_markdown)
    return 0


def cmd_resume(args) -> int:
    from researchmind.synthesis.report import render_markdown

    svc = _services()
    report = svc.runner.run(uuid.UUID(args.investigation), on_event=_print_event)
    _emit(report, args, render_markdown)
    return 0


def _emit(report: dict, args, render) -> None:
    md = render(report)
    print("\n" + md)
    if getattr(args, "out", None):
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md)
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"saved to {out}")


def cmd_check(_args) -> int:
    """Check database, LLM (structured output) and embedding model connectivity."""
    from pydantic import BaseModel
    from sqlalchemy import text

    from researchmind.llm import LLMClient, Message

    svc = _services()
    ok = True
    try:
        with svc.sessionmaker() as s:
            ver = s.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
        print(f"database     ok (pgvector {ver})")
    except Exception as exc:
        ok = False
        print(f"database     FAILED: {exc}")

    class Probe(BaseModel):
        answer: str
        confidence: float

    try:
        t = time.perf_counter()
        out = LLMClient(svc.provider).structured("check", [Message("user", "What is 2+3? Reply as JSON with answer and confidence.")], Probe, max_tokens=200)
        print(f"llm          ok ({svc.provider.provider}:{svc.provider.model}, {time.perf_counter() - t:.1f}s, answer={out.answer!r})")
    except Exception as exc:
        ok = False
        print(f"llm          FAILED ({svc.provider.provider}:{svc.provider.model}): {exc}")
    try:
        v = svc.embedder.embed_query("dense retrieval")
        print(f"embeddings   ok ({svc.embedder.model}, dim={len(v)})")
    except Exception as exc:
        ok = False
        print(f"embeddings   FAILED: {exc}")
    try:
        print(f"reranker     ok ({svc.reranker.name}, score={svc.reranker.score('q', ['q passage'])[0]:.3f})")
    except Exception as exc:
        ok = False
        print(f"reranker     FAILED: {exc}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="researchmind")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db", help="apply database migrations").set_defaults(fn=cmd_init_db)
    sub.add_parser("check", help="check database, LLM, embeddings, reranker").set_defaults(fn=cmd_check)

    s = sub.add_parser("ingest", help="ingest PDF files or folders into a project")
    s.add_argument("project", help="project name (created if missing) or id")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="search OpenAlex, Semantic Scholar, Crossref, arXiv")
    s.add_argument("query")
    s.add_argument("--sources", nargs="*", default=None)
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--year-min", type=int, default=None)
    s.add_argument("--import-to", default=None, help="project to import results into")
    s.add_argument("--pick", nargs="*", default=[], help="result numbers to import")
    s.add_argument("--import-all", action="store_true")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="run an investigation")
    s.add_argument("project")
    s.add_argument("question")
    s.add_argument("--out", default=None, help="folder for report.md and report.json")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("resume", help="resume an interrupted investigation")
    s.add_argument("investigation")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_resume)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
