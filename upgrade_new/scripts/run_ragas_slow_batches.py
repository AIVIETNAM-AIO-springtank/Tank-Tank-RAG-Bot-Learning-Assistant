"""Run RAGAS evaluation slowly with per-record checkpoint reports.

This runner is intentionally conservative for live API quotas:
- process one question/pipeline at a time,
- write JSON/CSV/Markdown after each answer and after each RAGAS score,
- resume from the JSON checkpoint unless --reset is provided.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upgrade_new.src import config
from upgrade_new.src.evaluation import (
    DEFAULT_METRICS,
    DEFAULT_REPORT_FORMATS,
    PIPELINE_PRESETS,
    load_testset,
    refresh_report_summary,
    retrieve_context_only,
    score_records_with_ragas,
    write_report_bundle,
)
from upgrade_new.src.rag_chain import answer_question


DEFAULT_PIPELINES = "vector,hybrid_rrf_rerank_mmr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run slow checkpointed RAGAS evaluation.")
    parser.add_argument("--testset", required=True, help="Path to JSONL testset.")
    parser.add_argument("--output-dir", default="upgrade_new/eval/reports", help="Directory for report output.")
    parser.add_argument("--report-name", default="ragas_slow_report", help="Base report filename.")
    parser.add_argument("--formats", default=",".join(DEFAULT_REPORT_FORMATS), help="Comma-separated json,csv,md.")
    parser.add_argument("--pipelines", default=DEFAULT_PIPELINES, help="Comma-separated pipeline preset names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional testset sample limit.")
    parser.add_argument("--candidate-k", type=int, default=None, help="Candidate count before rerank/MMR.")
    parser.add_argument("--top-k", type=int, default=None, help="Final context count per question.")
    parser.add_argument("--request-delay", type=float, default=20.0, help="Seconds to wait between answer calls.")
    parser.add_argument("--retry-attempts", type=int, default=1, help="Additional answer retries after failure.")
    parser.add_argument("--retry-delay", type=float, default=45.0, help="Seconds to wait before answer retry.")
    parser.add_argument("--ragas-delay", type=float, default=20.0, help="Seconds to wait between RAGAS calls.")
    parser.add_argument("--ragas-retry-attempts", type=int, default=1, help="Additional RAGAS retries after failure.")
    parser.add_argument("--ragas-retry-delay", type=float, default=60.0, help="Seconds to wait before RAGAS retry.")
    parser.add_argument("--ragas-batch-size", type=int, default=1, help="RAGAS batch size.")
    parser.add_argument("--retry-error-records", action="store_true", help="Retry records that already have errors.")
    parser.add_argument("--reset", action="store_true", help="Start a fresh report instead of resuming checkpoint.")
    args = parser.parse_args()

    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    pipeline_names = [item.strip() for item in args.pipelines.split(",") if item.strip()]
    report_path = Path(args.output_dir) / f"{args.report_name}.json"

    samples = load_testset(args.testset)
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]

    report = _load_or_create_report(
        report_path=report_path,
        reset=args.reset,
        testset_path=args.testset,
        samples=samples,
        pipeline_names=pipeline_names,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        request_delay=args.request_delay,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        ragas_delay=args.ragas_delay,
        ragas_retry_attempts=args.ragas_retry_attempts,
        ragas_retry_delay=args.ragas_retry_delay,
        ragas_batch_size=args.ragas_batch_size,
    )

    created = _write_checkpoint(report, args.output_dir, args.report_name, formats)
    print("Checkpoint ready.")
    _print_created(created)

    for run in report["runs"]:
        pipeline_name = run["config"]["pipeline_name"]
        preset = PIPELINE_PRESETS[pipeline_name]
        for index, sample in enumerate(samples, start=1):
            record = _find_record(run, str(sample.get("sample_id") or f"q{index:03d}"))
            if args.retry_error_records and record is not None and record.get("error"):
                record["answer"] = ""
                record["error"] = ""
                record["ragas"] = {}
            if record is not None and record.get("error") and not record.get("source_ids"):
                _backfill_error_context(
                    record=record,
                    sample=sample,
                    preset=preset,
                    candidate_k=args.candidate_k,
                    top_k=args.top_k,
                )
                report["generated_at"] = _now_iso()
                refresh_report_summary(report)
                created = _write_checkpoint(report, args.output_dir, args.report_name, formats)
                print(f"Backfilled retrieval context: {pipeline_name}/{record['sample_id']}")
                _print_created(created)
            if _record_is_complete(record):
                print(f"Skip complete: {pipeline_name}/{record['sample_id']}")
                continue

            if record is None or not record.get("answer") and not record.get("error"):
                record = _answer_sample(
                    sample=sample,
                    index=index,
                    preset=preset,
                    candidate_k=args.candidate_k,
                    top_k=args.top_k,
                    retry_attempts=args.retry_attempts,
                    retry_delay=args.retry_delay,
                )
                _upsert_record(run, record)
                report["generated_at"] = _now_iso()
                refresh_report_summary(report)
                created = _write_checkpoint(report, args.output_dir, args.report_name, formats)
                print(f"Answered: {pipeline_name}/{record['sample_id']} error={bool(record.get('error'))}")
                _print_created(created)
                if args.request_delay > 0:
                    time.sleep(args.request_delay)

            if not _ragas_complete(record):
                ragas_scores = score_records_with_ragas(
                    [record],
                    batch_size=args.ragas_batch_size,
                    delay_seconds=0,
                    retry_attempts=args.ragas_retry_attempts,
                    retry_delay_seconds=args.ragas_retry_delay,
                )
                record["ragas"] = ragas_scores[0] if ragas_scores else {}
                report["generated_at"] = _now_iso()
                refresh_report_summary(report)
                created = _write_checkpoint(report, args.output_dir, args.report_name, formats)
                print(f"RAGAS scored: {pipeline_name}/{record['sample_id']} ragas={record['ragas']}")
                _print_created(created)
                if args.ragas_delay > 0:
                    time.sleep(args.ragas_delay)

    refresh_report_summary(report)
    created = _write_checkpoint(report, args.output_dir, args.report_name, formats)
    print("Slow RAGAS evaluation finished.")
    _print_created(created)


def _load_or_create_report(
    *,
    report_path: Path,
    reset: bool,
    testset_path: str,
    samples: list[dict[str, Any]],
    pipeline_names: list[str],
    candidate_k: int | None,
    top_k: int | None,
    request_delay: float,
    retry_attempts: int,
    retry_delay: float,
    ragas_delay: float,
    ragas_retry_attempts: int,
    ragas_retry_delay: float,
    ragas_batch_size: int,
) -> dict[str, Any]:
    if report_path.exists() and not reset:
        return json.loads(report_path.read_text(encoding="utf-8"))

    unknown = [name for name in pipeline_names if name not in PIPELINE_PRESETS]
    if unknown:
        raise ValueError(f"Unknown pipeline preset(s): {', '.join(unknown)}")

    runs = [
        _new_run(
            pipeline_name=name,
            candidate_k=candidate_k,
            top_k=top_k,
            request_delay=request_delay,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            ragas_delay=ragas_delay,
            ragas_retry_attempts=ragas_retry_attempts,
            ragas_retry_delay=ragas_retry_delay,
            ragas_batch_size=ragas_batch_size,
        )
        for name in pipeline_names
    ]
    report = {
        "report_type": "pipeline_comparison",
        "generated_at": _now_iso(),
        "testset_path": testset_path,
        "sample_count": len(samples),
        "metrics": DEFAULT_METRICS,
        "runs": runs,
        "summary": [],
    }
    refresh_report_summary(report)
    return report


def _new_run(
    *,
    pipeline_name: str,
    candidate_k: int | None,
    top_k: int | None,
    request_delay: float,
    retry_attempts: int,
    retry_delay: float,
    ragas_delay: float,
    ragas_retry_attempts: int,
    ragas_retry_delay: float,
    ragas_batch_size: int,
) -> dict[str, Any]:
    preset = PIPELINE_PRESETS[pipeline_name]
    return {
        "report_type": "single_pipeline",
        "generated_at": _now_iso(),
        "sample_count": 0,
        "metrics": DEFAULT_METRICS,
        "config": {
            "pipeline_name": pipeline_name,
            "pipeline_label": preset["label"],
            "retrieval_mode": preset["retrieval_mode"],
            "candidate_k": candidate_k or config.RERANK_CANDIDATE_K,
            "top_k": top_k or config.DEFAULT_TOP_K,
            "rerank_enabled": preset["rerank_enabled"],
            "rerank_provider": config.RERANK_PROVIDER,
            "rerank_top_n": config.RERANK_TOP_N,
            "mmr_enabled": preset["mmr_enabled"],
            "mmr_lambda": config.MMR_LAMBDA,
            "final_context_k": top_k or config.FINAL_CONTEXT_K,
            "use_ragas": True,
            "request_delay_seconds": request_delay,
            "retry_attempts": retry_attempts,
            "retry_delay_seconds": retry_delay,
            "generate_answers": True,
            "ragas_batch_size": ragas_batch_size,
            "ragas_delay_seconds": ragas_delay,
            "ragas_retry_attempts": ragas_retry_attempts,
            "ragas_retry_delay_seconds": ragas_retry_delay,
            "ragas_judge_model": config.get_setting("RAGAS_JUDGE_MODEL", "gemini-2.5-flash-lite"),
        },
        "records": [],
    }


def _answer_sample(
    *,
    sample: dict[str, Any],
    index: int,
    preset: dict[str, Any],
    candidate_k: int | None,
    top_k: int | None,
    retry_attempts: int,
    retry_delay: float,
) -> dict[str, Any]:
    question = str(sample.get("question") or sample.get("query") or "").strip()
    if not question:
        raise ValueError(f"Empty question at sample index {index}")

    max_attempts = max(1, int(retry_attempts) + 1)
    last_error = ""
    result: dict[str, Any] = {"answer": "", "sources": [], "retrieval_debug": {}}
    for attempt in range(1, max_attempts + 1):
        try:
            result = answer_question(
                question,
                top_k=int(sample.get("top_k") or top_k or config.DEFAULT_TOP_K),
                retrieval_mode=str(preset["retrieval_mode"]),
                candidate_k=candidate_k,
                rerank_enabled=bool(preset["rerank_enabled"]),
                mmr_enabled=bool(preset["mmr_enabled"]),
            )
            last_error = ""
            break
        except Exception as exc:
            last_error = _preview(str(exc), 800)
            if attempt < max_attempts and retry_delay > 0:
                time.sleep(retry_delay)

    sources = result.get("sources", [])
    if last_error and not sources:
        result = _retrieval_only_result(
            question=question,
            top_k=int(sample.get("top_k") or top_k or config.DEFAULT_TOP_K),
            preset=preset,
            candidate_k=candidate_k,
        )
        sources = result.get("sources", [])
    return {
        "sample_id": sample.get("sample_id") or f"q{index:03d}",
        "question": question,
        "answer": result.get("answer", ""),
        "error": last_error,
        "attempt_count": attempt,
        "generation_skipped": False,
        "ground_truth": sample.get("ground_truth") or sample.get("reference") or "",
        "expected_source_notes": sample.get("expected_source_notes", ""),
        "tags": sample.get("tags", []),
        "contexts": [str(source.get("text") or "") for source in sources],
        "source_ids": [source.get("id", "") for source in sources],
        "source_titles": [_source_title(source) for source in sources],
        "retrieval_debug": result.get("retrieval_debug", {}),
        "ragas": {},
    }


def _backfill_error_context(
    *,
    record: dict[str, Any],
    sample: dict[str, Any],
    preset: dict[str, Any],
    candidate_k: int | None,
    top_k: int | None,
) -> None:
    question = str(record.get("question") or sample.get("question") or sample.get("query") or "").strip()
    if not question:
        return
    result = _retrieval_only_result(
        question=question,
        top_k=int(sample.get("top_k") or top_k or config.DEFAULT_TOP_K),
        preset=preset,
        candidate_k=candidate_k,
    )
    sources = result.get("sources", [])
    record["contexts"] = [str(source.get("text") or "") for source in sources]
    record["source_ids"] = [source.get("id", "") for source in sources]
    record["source_titles"] = [_source_title(source) for source in sources]
    record["retrieval_debug"] = result.get("retrieval_debug", {})


def _retrieval_only_result(
    *,
    question: str,
    top_k: int,
    preset: dict[str, Any],
    candidate_k: int | None,
) -> dict[str, Any]:
    return retrieve_context_only(
        question,
        top_k=top_k,
        retrieval_mode=str(preset["retrieval_mode"]),
        candidate_k=candidate_k,
        rerank_enabled=bool(preset["rerank_enabled"]),
        mmr_enabled=bool(preset["mmr_enabled"]),
    )


def _write_checkpoint(report: dict[str, Any], output_dir: str, report_name: str, formats: list[str]) -> dict[str, str]:
    for run in report.get("runs", []):
        run["generated_at"] = report.get("generated_at")
        run["sample_count"] = len(run.get("records", []))
    return write_report_bundle(report, output_dir, basename=report_name, formats=formats)


def _find_record(run: dict[str, Any], sample_id: str) -> dict[str, Any] | None:
    for record in run.get("records", []):
        if str(record.get("sample_id")) == sample_id:
            return record
    return None


def _upsert_record(run: dict[str, Any], record: dict[str, Any]) -> None:
    records = run.setdefault("records", [])
    for index, existing in enumerate(records):
        if existing.get("sample_id") == record.get("sample_id"):
            records[index] = record
            return
    records.append(record)


def _record_is_complete(record: dict[str, Any] | None) -> bool:
    if record is None:
        return False
    if record.get("answer"):
        return _ragas_complete(record)
    ragas = record.get("ragas")
    return bool(record.get("error")) and bool(ragas) and bool(ragas.get("ragas_error"))


def _ragas_complete(record: dict[str, Any]) -> bool:
    ragas = record.get("ragas")
    if not ragas:
        return False
    if ragas.get("ragas_error"):
        return True
    return all(
        isinstance(ragas.get(metric), (int, float)) and math.isfinite(float(ragas.get(metric)))
        for metric in DEFAULT_METRICS
    )


def _source_title(source: dict[str, Any]) -> str:
    metadata = source.get("metadata") or {}
    source_type = metadata.get("source_type", "")
    if source_type == "pdf":
        return f"PDF:{metadata.get('source_file', '')}:page={metadata.get('page_number', '')}:id={source.get('id', '')}"
    if source_type == "notion":
        return f"Notion:{metadata.get('title', '')}:heading={metadata.get('heading_path', '')}:id={source.get('id', '')}"
    return str(source.get("id", ""))


def _print_created(created: dict[str, str]) -> None:
    for report_format, path in created.items():
        print(f"- {report_format}: {path}")


def _preview(text: str, limit: int) -> str:
    clean = str(text).replace("\n", " ").strip()
    return clean[:limit]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
