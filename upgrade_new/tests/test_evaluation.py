"""Tests for offline evaluation reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from upgrade_new.src.evaluation import (
    refresh_report_summary,
    render_markdown_report,
    run_evaluation,
    run_pipeline_comparison,
    write_report,
    write_report_bundle,
)


def fake_answer_question(
    question: str,
    top_k: int,
    retrieval_mode: str,
    candidate_k: int | None = None,
    rerank_enabled: bool | None = None,
    mmr_enabled: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "answer": f"Answer for {question}",
        "sources": [
            {
                "id": f"{retrieval_mode}_source_1",
                "text": "Context about RRF and reranking.",
                "metadata": {"source_type": "notion", "title": "RAG lesson", "heading_path": "Retrieval"},
            },
            {
                "id": f"{retrieval_mode}_source_2",
                "text": "Context about MMR diversity.",
                "metadata": {"source_type": "pdf", "source_file": "rag.pdf", "page_number": 12},
            },
        ],
        "retrieval_debug": {
            "retrieved_count": candidate_k or top_k,
            "reranked_count": 2 if rerank_enabled else 0,
            "final_count": 2,
            "mmr_enabled": bool(mmr_enabled),
        },
    }


def write_testset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "q001",
                        "question": "RRF dung de lam gi?",
                        "ground_truth": "RRF fuses vector and BM25 ranks.",
                        "expected_source_notes": "Needs retrieval source.",
                        "tags": ["retrieval", "rrf"],
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "q002",
                        "question": "MMR giup ich gi?",
                        "ground_truth": "MMR increases diversity.",
                        "expected_source_notes": "Needs MMR source.",
                        "tags": ["rerank", "mmr"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_run_evaluation_writes_markdown_csv_and_json(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)

    report = run_evaluation(
        str(testset_path),
        retrieval_mode="hybrid_rrf",
        candidate_k=12,
        rerank_enabled=True,
        mmr_enabled=True,
        answer_fn=fake_answer_question,
    )
    paths = write_report_bundle(report, str(tmp_path / "reports"), basename="eval", formats=["json", "csv", "md"])

    json_report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(Path(paths["csv"]).open("r", encoding="utf-8")))
    markdown = Path(paths["md"]).read_text(encoding="utf-8")

    assert json_report["sample_count"] == 2
    assert json_report["config"]["candidate_k"] == 12
    assert len(csv_rows) == 2
    assert csv_rows[0]["pipeline_name"] == "custom"
    assert csv_rows[0]["source_count"] == "2"
    assert "error" in csv_rows[0]
    assert csv_rows[0]["attempt_count"] == "1"
    assert "# RAG Evaluation Report" in markdown
    assert "Candidate K" in markdown
    assert "Per-Question Results" in markdown
    assert "retrieved=12" in markdown


def test_pipeline_comparison_report_contains_each_pipeline(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)

    report = run_pipeline_comparison(
        str(testset_path),
        pipeline_names=["vector", "hybrid_rrf", "hybrid_rrf_rerank_mmr"],
        candidate_k=8,
        answer_fn=fake_answer_question,
    )
    markdown = render_markdown_report(report)

    assert report["report_type"] == "pipeline_comparison"
    assert len(report["runs"]) == 3
    assert len(report["summary"]) == 3
    assert "Pipeline Summary" in markdown
    assert "Vector only" in markdown
    assert "Hybrid RRF + Cohere Rerank + MMR" in markdown


def test_write_report_uses_suffix_format(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)
    report = run_evaluation(str(testset_path), answer_fn=fake_answer_question)

    json_path = write_report(report, str(tmp_path / "single.json"))
    csv_path = write_report(report, str(tmp_path / "single.csv"))
    md_path = write_report(report, str(tmp_path / "single.md"))

    assert Path(json_path).exists()
    assert Path(csv_path).exists()
    assert Path(md_path).exists()
    assert "RAG Evaluation Report" in Path(md_path).read_text(encoding="utf-8")


def test_evaluation_records_errors_without_crashing(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)

    def failing_answer_question(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Gemini API error 429: quota exceeded")

    report = run_evaluation(str(testset_path), answer_fn=failing_answer_question)
    markdown = render_markdown_report(report)

    assert report["records"][0]["error"] == "Gemini API error 429: quota exceeded"
    assert report["records"][0]["answer"] == ""
    assert report["records"][0]["attempt_count"] == 1
    assert "quota exceeded" in markdown


def test_evaluation_retries_failed_answer_calls(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)
    calls = {"count": 0}

    def flaky_answer_question(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary quota")
        return fake_answer_question(*args, **kwargs)

    report = run_evaluation(
        str(testset_path),
        limit=1,
        retry_attempts=1,
        retry_delay_seconds=0,
        answer_fn=flaky_answer_question,
    )

    assert calls["count"] == 2
    assert report["records"][0]["error"] == ""
    assert report["records"][0]["attempt_count"] == 2


def test_retrieval_only_mode_does_not_call_answer_fn(tmp_path: Path, monkeypatch) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)

    def forbidden_answer_fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("answer_fn should not be called in retrieval-only mode")

    def fake_retrieve_context_only(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "",
            "sources": [{"id": "source", "text": "retrieved context", "metadata": {"source_type": "notion"}}],
            "retrieval_debug": {"retrieved_count": 1, "reranked_count": 1, "final_count": 1},
        }

    monkeypatch.setattr("upgrade_new.src.evaluation.retrieve_context_only", fake_retrieve_context_only)

    report = run_evaluation(
        str(testset_path),
        limit=1,
        generate_answers=False,
        answer_fn=forbidden_answer_fn,
    )

    assert report["config"]["generate_answers"] is False
    assert report["records"][0]["generation_skipped"] is True
    assert report["records"][0]["source_ids"] == ["source"]


def test_refresh_report_summary_includes_ragas_metrics(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)
    report = run_evaluation(str(testset_path), limit=1, answer_fn=fake_answer_question)

    report["records"][0]["ragas"] = {
        "faithfulness": 1.0,
        "answer_relevancy": 0.5,
        "context_precision": 0.75,
        "context_recall": 0.25,
    }
    refresh_report_summary(report)
    paths = write_report_bundle(report, str(tmp_path / "reports"), basename="eval", formats=["csv", "md"])

    assert report["summary"]["ragas_average"]["faithfulness"] == 1.0
    assert report["summary"]["ragas_average"]["answer_relevancy"] == 0.5

    csv_rows = list(csv.DictReader(Path(paths["csv"]).open("r", encoding="utf-8")))
    markdown = Path(paths["md"]).read_text(encoding="utf-8")

    assert csv_rows[0]["faithfulness"] == "1.0"
    assert csv_rows[0]["answer_relevancy"] == "0.5"
    assert "faithfulness=1.000" in markdown


def test_refresh_report_summary_ignores_nan_ragas_metrics(tmp_path: Path) -> None:
    testset_path = tmp_path / "testset.jsonl"
    write_testset(testset_path)
    report = run_evaluation(str(testset_path), limit=1, answer_fn=fake_answer_question)

    report["records"][0]["ragas"] = {
        "faithfulness": 1.0,
        "answer_relevancy": math.nan,
    }
    refresh_report_summary(report)
    paths = write_report_bundle(report, str(tmp_path / "reports"), basename="eval", formats=["json", "csv", "md"])

    assert report["summary"]["ragas_average"]["faithfulness"] == 1.0
    assert "answer_relevancy" not in report["summary"]["ragas_average"]

    json_text = Path(paths["json"]).read_text(encoding="utf-8")
    csv_rows = list(csv.DictReader(Path(paths["csv"]).open("r", encoding="utf-8")))
    markdown = Path(paths["md"]).read_text(encoding="utf-8")

    assert "NaN" not in json_text
    assert csv_rows[0]["answer_relevancy"] == ""
    assert "answer_relevancy=nan" not in markdown
