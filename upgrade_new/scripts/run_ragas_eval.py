"""Run offline RAG/RAGAS evaluation from the project root.

Examples:
    python upgrade_new/scripts/run_ragas_eval.py \
        --testset upgrade_new/eval/testset.example.jsonl

    python upgrade_new/scripts/run_ragas_eval.py \
        --testset upgrade_new/eval/testset.example.jsonl \
        --compare-pipelines --formats json,csv,md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upgrade_new.src.evaluation import (
    DEFAULT_REPORT_FORMATS,
    PIPELINE_PRESETS,
    run_evaluation,
    run_pipeline_comparison,
    write_report,
    write_report_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation and write report files.")
    parser.add_argument("--testset", required=True, help="Path to JSONL testset.")
    parser.add_argument("--output", default=None, help="Legacy single report path ending with .json, .csv or .md.")
    parser.add_argument("--output-dir", default="upgrade_new/eval/reports", help="Directory for report bundle output.")
    parser.add_argument("--report-name", default="rag_eval_report", help="Base filename for report bundle.")
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_REPORT_FORMATS),
        help="Comma-separated report formats: json,csv,md.",
    )
    parser.add_argument("--compare-pipelines", action="store_true", help="Run all selected pipeline presets.")
    parser.add_argument(
        "--pipelines",
        default=",".join(PIPELINE_PRESETS.keys()),
        help="Comma-separated pipeline presets for --compare-pipelines.",
    )
    parser.add_argument("--retrieval-mode", default="hybrid_rrf", help="Retrieval mode for single-pipeline eval.")
    parser.add_argument("--candidate-k", type=int, default=None, help="Candidate count before rerank/MMR.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--request-delay", type=float, default=0.0, help="Seconds to wait between questions.")
    parser.add_argument("--retry-attempts", type=int, default=0, help="Additional retries per question after failure.")
    parser.add_argument("--retry-delay", type=float, default=0.0, help="Seconds to wait before each retry.")
    parser.add_argument("--ragas-batch-size", type=int, default=1, help="RAGAS scoring batch size.")
    parser.add_argument("--ragas-delay", type=float, default=0.0, help="Seconds to wait between RAGAS batches.")
    parser.add_argument("--ragas-retry-attempts", type=int, default=0, help="Additional retries per RAGAS batch.")
    parser.add_argument("--ragas-retry-delay", type=float, default=0.0, help="Seconds to wait before retrying RAGAS.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Gemini generation and benchmark retrieval sources only.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable rerank for single-pipeline eval.")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR for single-pipeline eval.")
    parser.add_argument("--use-ragas", action="store_true", help="Run RAGAS metrics if eval deps are installed.")
    args = parser.parse_args()

    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    if args.compare_pipelines:
        report = run_pipeline_comparison(
            testset_path=args.testset,
            pipeline_names=[item.strip() for item in args.pipelines.split(",") if item.strip()],
            candidate_k=args.candidate_k,
            limit=args.limit,
            use_ragas=args.use_ragas,
            request_delay_seconds=args.request_delay,
            retry_attempts=args.retry_attempts,
            retry_delay_seconds=args.retry_delay,
            generate_answers=not args.retrieval_only,
            ragas_batch_size=args.ragas_batch_size,
            ragas_delay_seconds=args.ragas_delay,
            ragas_retry_attempts=args.ragas_retry_attempts,
            ragas_retry_delay_seconds=args.ragas_retry_delay,
        )
    else:
        report = run_evaluation(
            testset_path=args.testset,
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
            rerank_enabled=False if args.no_rerank else None,
            mmr_enabled=False if args.no_mmr else None,
            limit=args.limit,
            use_ragas=args.use_ragas,
            request_delay_seconds=args.request_delay,
            retry_attempts=args.retry_attempts,
            retry_delay_seconds=args.retry_delay,
            generate_answers=not args.retrieval_only,
            ragas_batch_size=args.ragas_batch_size,
            ragas_delay_seconds=args.ragas_delay,
            ragas_retry_attempts=args.ragas_retry_attempts,
            ragas_retry_delay_seconds=args.ragas_retry_delay,
            pipeline_name=args.retrieval_mode,
            pipeline_label=args.retrieval_mode,
        )

    if args.output:
        created = {"single": write_report(report, args.output)}
    else:
        created = write_report_bundle(report, args.output_dir, basename=args.report_name, formats=formats)

    print("Evaluation finished.")
    for report_format, path in created.items():
        print(f"- {report_format}: {path}")


if __name__ == "__main__":
    main()
