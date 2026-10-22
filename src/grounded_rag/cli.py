"""Command line entry point: grounded-rag ingest | ask | eval | serve."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from grounded_rag import factory
from grounded_rag.config import ConfigError, Settings
from grounded_rag.evaluation.dataset import load_cases
from grounded_rag.evaluation.judge import LLMJudge
from grounded_rag.evaluation.report import summary_dict, write_report
from grounded_rag.evaluation.runner import run_eval
from grounded_rag.ingest.pipeline import ingest_folder
from grounded_rag.rag.engine import UpstreamError

EVAL_EXCERPT_CHARS = 4000  # judge sees whole passages, not UI-sized excerpts
EXIT_CONFIG_ERROR = 2
EXIT_UPSTREAM_ERROR = 3


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    folder = Path(args.pdf_dir or settings.pdf_dir)
    store = factory.build_store(settings)
    report = ingest_folder(folder, store, settings.chunk_size, settings.chunk_overlap)
    for result in report.results:
        detail = f"{result.pages} pages, {result.chunks} chunks"
        if result.error:
            detail = f"{detail} — {result.error}"
        print(f"  {result.status.value:<16} {result.doc_name} ({detail})")
    print(
        f"Loaded {report.loaded}/{len(report.results)} files, {report.total_chunks} chunks. "
        f"Library now holds {store.count()} chunks."
    )
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    answer = factory.build_answerer(settings).answer(args.question)
    print(answer.answer)
    for number, source in enumerate(answer.sources, start=1):
        print(f"  [{number}] {source.doc_name}, page {source.page} (score {source.score})")
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    cases = load_cases(Path(args.questions))
    answerer = factory.build_answerer(settings, excerpt_chars=EVAL_EXCERPT_CHARS)
    judge = None if args.no_judge else LLMJudge(factory.build_generator(settings))
    summary = run_eval(cases, answerer, judge)
    json_path, md_path = write_report(summary, Path(args.out))
    for key, value in summary_dict(summary).items():
        print(f"  {key:<28} {value}")
    print(f"Report: {md_path} and {json_path}")
    return 0


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "grounded_rag.api.app:create_default_app", factory=True, host=args.host, port=args.port
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load all PDFs from a folder into the library")
    ingest.add_argument("--pdf-dir", help="Folder with PDFs (default: PDF_DIR)")
    ingest.set_defaults(handler=_cmd_ingest)

    ask = sub.add_parser("ask", help="Ask one question from the terminal")
    ask.add_argument("question")
    ask.set_defaults(handler=_cmd_ask)

    evaluate = sub.add_parser("eval", help="Run the evaluation question set")
    evaluate.add_argument("--questions", default="eval/questions.json")
    evaluate.add_argument("--out", default="eval/results")
    evaluate.add_argument("--no-judge", action="store_true", help="Skip LLM-judge scoring")
    evaluate.set_defaults(handler=_cmd_eval)

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args, Settings())
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except (UpstreamError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_UPSTREAM_ERROR


if __name__ == "__main__":
    sys.exit(main())
