"""Offline RAG evaluation and report generation helpers.

The Streamlit app does not import or run RAGAS. This module is intended for
manual offline experiments against a small golden testset and can write
human-readable Markdown plus machine-friendly JSON/CSV reports.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from upgrade_new.src import config
from upgrade_new.src.embeddings import embed_documents, embed_query
from upgrade_new.src.rag_chain import answer_question
from upgrade_new.src.reranker import rerank_candidates, select_context_mmr
from upgrade_new.src.retriever import retrieve


DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
DEFAULT_REPORT_FORMATS = ["json", "csv", "md"]

PIPELINE_PRESETS: dict[str, dict[str, Any]] = {
    "vector": {
        "label": "Vector only",
        "retrieval_mode": "vector",
        "rerank_enabled": False,
        "mmr_enabled": False,
    },
    "keyword": {
        "label": "BM25 keyword only",
        "retrieval_mode": "keyword",
        "rerank_enabled": False,
        "mmr_enabled": False,
    },
    "hybrid_rrf": {
        "label": "Hybrid RRF",
        "retrieval_mode": "hybrid_rrf",
        "rerank_enabled": False,
        "mmr_enabled": False,
    },
    "hybrid_rrf_rerank": {
        "label": "Hybrid RRF + Cohere Rerank",
        "retrieval_mode": "hybrid_rrf",
        "rerank_enabled": True,
        "mmr_enabled": False,
    },
    "hybrid_rrf_rerank_mmr": {
        "label": "Hybrid RRF + Cohere Rerank + MMR",
        "retrieval_mode": "hybrid_rrf",
        "rerank_enabled": True,
        "mmr_enabled": True,
    },
}


def run_evaluation(
    testset_path: str,
    output_path: str | None = None,
    *,
    retrieval_mode: str = "hybrid_rrf",
    candidate_k: int | None = None,
    rerank_enabled: bool | None = None,
    mmr_enabled: bool | None = None,
    limit: int | None = None,
    use_ragas: bool = False,
    pipeline_name: str = "custom",
    pipeline_label: str | None = None,
    request_delay_seconds: float = 0.0,
    retry_attempts: int = 0,
    retry_delay_seconds: float = 0.0,
    generate_answers: bool = True,
    ragas_batch_size: int = 1,
    ragas_delay_seconds: float = 0.0,
    ragas_retry_attempts: int = 0,
    ragas_retry_delay_seconds: float = 0.0,
    answer_fn: Callable[..., dict[str, Any]] = answer_question,
) -> dict[str, Any]:
    """Run one RAG pipeline over a JSONL testset and optionally score with RAGAS."""
    samples = load_testset(testset_path)
    if limit is not None:
        samples = samples[: max(0, limit)]

    report = _evaluate_samples(
        samples=samples,
        testset_path=testset_path,
        pipeline_name=pipeline_name,
        pipeline_label=pipeline_label or pipeline_name,
        retrieval_mode=retrieval_mode,
        candidate_k=candidate_k,
        rerank_enabled=rerank_enabled,
        mmr_enabled=mmr_enabled,
        use_ragas=use_ragas,
        request_delay_seconds=request_delay_seconds,
        retry_attempts=retry_attempts,
        retry_delay_seconds=retry_delay_seconds,
        generate_answers=generate_answers,
        ragas_batch_size=ragas_batch_size,
        ragas_delay_seconds=ragas_delay_seconds,
        ragas_retry_attempts=ragas_retry_attempts,
        ragas_retry_delay_seconds=ragas_retry_delay_seconds,
        answer_fn=answer_fn,
    )

    if output_path:
        write_report(report, output_path)
    return report


def run_pipeline_comparison(
    testset_path: str,
    *,
    pipeline_names: list[str] | None = None,
    candidate_k: int | None = None,
    limit: int | None = None,
    use_ragas: bool = False,
    request_delay_seconds: float = 0.0,
    retry_attempts: int = 0,
    retry_delay_seconds: float = 0.0,
    generate_answers: bool = True,
    ragas_batch_size: int = 1,
    ragas_delay_seconds: float = 0.0,
    ragas_retry_attempts: int = 0,
    ragas_retry_delay_seconds: float = 0.0,
    answer_fn: Callable[..., dict[str, Any]] = answer_question,
) -> dict[str, Any]:
    """Run multiple retrieval/rerank/MMR variants against the same testset."""
    samples = load_testset(testset_path)
    if limit is not None:
        samples = samples[: max(0, limit)]

    selected_names = pipeline_names or list(PIPELINE_PRESETS)
    runs: list[dict[str, Any]] = []
    for name in selected_names:
        preset = PIPELINE_PRESETS.get(name)
        if preset is None:
            raise ValueError(f"Unknown evaluation pipeline: {name}")
        runs.append(
            _evaluate_samples(
                samples=samples,
                testset_path=testset_path,
                pipeline_name=name,
                pipeline_label=str(preset["label"]),
                retrieval_mode=str(preset["retrieval_mode"]),
                candidate_k=candidate_k,
                rerank_enabled=bool(preset["rerank_enabled"]),
                mmr_enabled=bool(preset["mmr_enabled"]),
                use_ragas=use_ragas,
                request_delay_seconds=request_delay_seconds,
                retry_attempts=retry_attempts,
                retry_delay_seconds=retry_delay_seconds,
                generate_answers=generate_answers,
                ragas_batch_size=ragas_batch_size,
                ragas_delay_seconds=ragas_delay_seconds,
                ragas_retry_attempts=ragas_retry_attempts,
                ragas_retry_delay_seconds=ragas_retry_delay_seconds,
                answer_fn=answer_fn,
            )
        )

    return {
        "report_type": "pipeline_comparison",
        "generated_at": _now_iso(),
        "testset_path": testset_path,
        "sample_count": len(samples),
        "metrics": DEFAULT_METRICS,
        "runs": runs,
        "summary": [_summarize_run(run) for run in runs],
    }


def write_report(report: dict[str, Any], output_path: str) -> str:
    """Write a report in the format implied by output_path suffix."""
    path = Path(output_path)
    suffix = path.suffix.lower().lstrip(".") or "json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == "json":
        path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    elif suffix == "csv":
        _write_csv(report, path)
    elif suffix in {"md", "markdown"}:
        path.write_text(render_markdown_report(report), encoding="utf-8")
    else:
        raise ValueError("Report output must end with .json, .csv or .md")
    return str(path)


def write_report_bundle(
    report: dict[str, Any],
    output_dir: str,
    *,
    basename: str = "rag_eval_report",
    formats: list[str] | None = None,
) -> dict[str, str]:
    """Write one report to several formats and return created paths."""
    selected_formats = _normalize_formats(formats or DEFAULT_REPORT_FORMATS)
    created: dict[str, str] = {}
    for report_format in selected_formats:
        suffix = "md" if report_format == "markdown" else report_format
        created[report_format] = write_report(report, str(Path(output_dir) / f"{basename}.{suffix}"))
    return created


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a polished Markdown report with test metadata and per-question details."""
    if report.get("report_type") == "pipeline_comparison":
        return _render_comparison_markdown(report)
    return _render_single_run_markdown(report)


def load_testset(path: str) -> list[dict[str, Any]]:
    """Load a JSONL testset with one question sample per line."""
    samples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean = line.strip()
            if not clean:
                continue
            sample = json.loads(clean)
            sample.setdefault("sample_id", f"q{line_number:03d}")
            samples.append(sample)
    return samples


def retrieve_context_only(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    retrieval_mode: str = "hybrid_rrf",
    candidate_k: int | None = None,
    rerank_enabled: bool | None = None,
    mmr_enabled: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run retrieval/rerank/MMR without calling Gemini generation."""
    final_k = max(1, int(top_k or config.FINAL_CONTEXT_K))
    should_rerank = config.ENABLE_RERANKING if rerank_enabled is None else rerank_enabled
    should_mmr = config.ENABLE_MMR if mmr_enabled is None else mmr_enabled
    retrieve_k = max(
        final_k,
        int(candidate_k or (config.RERANK_CANDIDATE_K if should_rerank or should_mmr else top_k)),
    )
    candidates = retrieve(
        question,
        top_k=retrieve_k,
        retrieval_mode=retrieval_mode,
        candidate_k=retrieve_k,
    )
    reranked = candidates
    if should_rerank:
        reranked = rerank_candidates(question, candidates, top_k=max(final_k, config.RERANK_TOP_N))
    sources = (
        select_context_mmr(question, reranked, final_k=final_k, lambda_mult=config.MMR_LAMBDA)
        if should_mmr
        else reranked[:final_k]
    )
    return {
        "answer": "",
        "sources": sources,
        "retrieval_debug": {
            "retrieved_count": len(candidates),
            "reranked_count": len(reranked),
            "final_count": len(sources),
            "rerank_enabled": should_rerank,
            "mmr_enabled": should_mmr,
            "generation_skipped": True,
        },
    }


def score_with_ragas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible RAGAS scorer entrypoint."""
    return score_records_with_ragas(records)


def score_records_with_ragas(
    records: list[dict[str, Any]],
    *,
    batch_size: int = 1,
    delay_seconds: float = 0.0,
    retry_attempts: int = 0,
    retry_delay_seconds: float = 0.0,
    judge_model: str | None = None,
) -> list[dict[str, Any]]:
    """Score records with RAGAS in small batches with retry/delay controls."""
    if not records:
        return []

    _install_ragas_langchain_compat()
    try:
        from datasets import Dataset
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.embeddings import Embeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install upgrade_new/requirements-eval.txt to run RAGAS evaluation.") from exc

    class CohereLangchainEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_documents(list(texts))

        def embed_query(self, text: str) -> list[float]:
            return embed_query(str(text))

    class CohereChatModel(BaseChatModel):
        model: str
        api_key: str
        temperature: float = 0
        max_tokens: int = 768

        @property
        def _llm_type(self) -> str:
            return "cohere_chat_v2"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            payload = {
                "model": self.model,
                "messages": [_cohere_message_from_langchain(message) for message in messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if stop:
                payload["stop_sequences"] = stop
            response = requests.post(
                "https://api.cohere.com/v2/chat",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Cohere Chat API error {response.status_code}: {_safe_metric_error_detail(response.text)}")
            data = response.json()
            content = data.get("message", {}).get("content", [])
            if isinstance(content, list):
                text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            else:
                text = str(content or "")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text.strip()))])

    embeddings = LangchainEmbeddingsWrapper(CohereLangchainEmbeddings())
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    judge_model_name = judge_model or config.get_setting("RAGAS_JUDGE_MODEL", "gemini-2.5-flash-lite")
    judge_provider = config.get_setting("RAGAS_JUDGE_PROVIDER", "gemini").strip().lower()
    if judge_provider == "cohere":
        api_keys = [config.COHERE_API_KEY] if config.COHERE_API_KEY else []
        llm_factory = lambda api_key: LangchainLLMWrapper(
            CohereChatModel(
                model=judge_model or config.get_setting("RAGAS_JUDGE_MODEL", config.COHERE_GENERATION_MODEL),
                api_key=api_key,
                temperature=0,
            )
        )
    else:
        api_keys = list(config.GEMINI_API_KEYS)
        llm_factory = lambda api_key: LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=judge_model_name,
                google_api_key=api_key,
                temperature=0,
            )
        )

    scores: list[dict[str, Any]] = []
    clean_batch_size = max(1, int(batch_size))
    for start in range(0, len(records), clean_batch_size):
        batch_records = records[start : start + clean_batch_size]
        scores.extend(
            _score_ragas_batch_with_retries(
                batch_records=batch_records,
                dataset_factory=Dataset.from_list,
                evaluate_fn=evaluate,
                metrics=metrics,
                llm_factory=llm_factory,
                api_keys=api_keys,
                embeddings=embeddings,
                retry_attempts=retry_attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
        )
        if delay_seconds > 0 and start + clean_batch_size < len(records):
            time.sleep(delay_seconds)
    return scores


def _install_ragas_langchain_compat() -> None:
    """Patch old RAGAS LangChain import paths against current Google integrations."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        from langchain_google_vertexai import ChatVertexAI
    except Exception:
        return
    module = types.ModuleType(module_name)
    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def _cohere_message_from_langchain(message: Any) -> dict[str, str]:
    role_by_type = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "chat": "user",
    }
    content = message.content
    if isinstance(content, list):
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content or "")
    return {"role": role_by_type.get(getattr(message, "type", ""), "user"), "content": text}


def _score_ragas_batch_with_retries(
    *,
    batch_records: list[dict[str, Any]],
    dataset_factory: Callable[[list[dict[str, Any]]], Any],
    evaluate_fn: Callable[..., Any],
    metrics: list[Any],
    llm_factory: Callable[[str], Any],
    api_keys: list[str],
    embeddings: Any,
    retry_attempts: int,
    retry_delay_seconds: float,
) -> list[dict[str, Any]]:
    skip_reasons = [_skip_ragas_reason(record) for record in batch_records]
    if all(skip_reasons):
        return [_empty_ragas_score(reason) for reason in skip_reasons]
    if not api_keys:
        return [_empty_ragas_score("Missing GEMINI_API_KEY or GEMINI_API_KEYS for RAGAS judge.") for _ in batch_records]

    max_attempts = max(1, int(retry_attempts) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        for key_index, api_key in enumerate(api_keys, start=1):
            try:
                dataset = dataset_factory([_record_to_ragas_row(record) for record in batch_records])
                result = evaluate_fn(
                    dataset,
                    metrics=metrics,
                    llm=llm_factory(api_key),
                    embeddings=embeddings,
                    raise_exceptions=False,
                    show_progress=False,
                    batch_size=1,
                )
                rows = result.to_pandas().to_dict(orient="records")
                return [
                    _extract_ragas_score(rows[index] if index < len(rows) else {})
                    for index in range(len(batch_records))
                ]
            except Exception as exc:
                last_error = _preview(_safe_metric_error_detail(str(exc)), 800)
                last_error = f"key {key_index}/{len(api_keys)}: {last_error}"
        if attempt < max_attempts and retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)
    return [_empty_ragas_score(last_error or "RAGAS scoring failed") for _ in batch_records]


def _record_to_ragas_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_input": record.get("question", ""),
        "response": record.get("answer", ""),
        "retrieved_contexts": list(record.get("contexts") or []),
        "reference": record.get("ground_truth", ""),
    }


def _skip_ragas_reason(record: dict[str, Any]) -> str:
    if record.get("error"):
        return f"skipped because answer generation failed: {record.get('error')}"
    if not str(record.get("answer") or "").strip():
        return "skipped because answer is empty"
    if not record.get("contexts"):
        return "skipped because retrieved contexts are empty"
    if not str(record.get("ground_truth") or "").strip():
        return "skipped because ground_truth/reference is empty"
    return ""


def _empty_ragas_score(reason: str) -> dict[str, Any]:
    return {"ragas_error": reason}


def _safe_metric_error_detail(detail: str) -> str:
    clean = detail.replace("\n", " ").strip()
    for secret in list(config.GEMINI_API_KEYS) + [config.GEMINI_API_KEY]:
        if secret:
            clean = clean.replace(secret, "<redacted>")
    if config.COHERE_API_KEY:
        clean = clean.replace(config.COHERE_API_KEY, "<redacted>")
    return clean


def _extract_ragas_score(row: dict[str, Any]) -> dict[str, Any]:
    score: dict[str, Any] = {}
    for metric in DEFAULT_METRICS:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            score[metric] = float(value)
    if not score and row:
        score["ragas_error"] = _preview(str(row), 500)
    return score


def _evaluate_samples(
    *,
    samples: list[dict[str, Any]],
    testset_path: str,
    pipeline_name: str,
    pipeline_label: str,
    retrieval_mode: str,
    candidate_k: int | None,
    rerank_enabled: bool | None,
    mmr_enabled: bool | None,
    use_ragas: bool,
    request_delay_seconds: float,
    retry_attempts: int,
    retry_delay_seconds: float,
    generate_answers: bool,
    ragas_batch_size: int,
    ragas_delay_seconds: float,
    ragas_retry_attempts: int,
    ragas_retry_delay_seconds: float,
    answer_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        question = str(sample.get("question") or sample.get("query") or "").strip()
        if not question:
            continue
        result, error, attempt_count = _answer_with_retries(
            answer_fn=answer_fn if generate_answers else retrieve_context_only,
            question=question,
            top_k=int(sample.get("top_k") or config.DEFAULT_TOP_K),
            retrieval_mode=retrieval_mode,
            candidate_k=candidate_k,
            rerank_enabled=rerank_enabled,
            mmr_enabled=mmr_enabled,
            retry_attempts=retry_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        sources = result.get("sources", [])
        records.append(
            {
                "sample_id": sample.get("sample_id") or f"q{index:03d}",
                "question": question,
                "answer": result.get("answer", ""),
                "error": error,
                "attempt_count": attempt_count,
                "generation_skipped": not generate_answers,
                "ground_truth": sample.get("ground_truth") or sample.get("reference") or "",
                "expected_source_notes": sample.get("expected_source_notes", ""),
                "tags": sample.get("tags", []),
                "contexts": [str(source.get("text") or "") for source in sources],
                "source_ids": [source.get("id", "") for source in sources],
                "source_titles": [_source_title(source) for source in sources],
                "retrieval_debug": result.get("retrieval_debug", {}),
            }
        )
        if request_delay_seconds > 0 and index < len(samples):
            time.sleep(request_delay_seconds)

    ragas_scores = (
        score_records_with_ragas(
            records,
            batch_size=ragas_batch_size,
            delay_seconds=ragas_delay_seconds,
            retry_attempts=ragas_retry_attempts,
            retry_delay_seconds=ragas_retry_delay_seconds,
        )
        if use_ragas
        else []
    )
    _attach_ragas_scores(records, ragas_scores)

    report = {
        "report_type": "single_pipeline",
        "generated_at": _now_iso(),
        "testset_path": testset_path,
        "sample_count": len(records),
        "metrics": DEFAULT_METRICS,
        "config": {
            "pipeline_name": pipeline_name,
            "pipeline_label": pipeline_label,
            "retrieval_mode": retrieval_mode,
            "candidate_k": candidate_k or config.RERANK_CANDIDATE_K,
            "rerank_enabled": config.ENABLE_RERANKING if rerank_enabled is None else rerank_enabled,
            "rerank_provider": config.RERANK_PROVIDER,
            "rerank_top_n": config.RERANK_TOP_N,
            "mmr_enabled": config.ENABLE_MMR if mmr_enabled is None else mmr_enabled,
            "mmr_lambda": config.MMR_LAMBDA,
            "final_context_k": config.FINAL_CONTEXT_K,
            "use_ragas": use_ragas,
            "request_delay_seconds": request_delay_seconds,
            "retry_attempts": retry_attempts,
            "retry_delay_seconds": retry_delay_seconds,
            "generate_answers": generate_answers,
            "ragas_batch_size": ragas_batch_size,
            "ragas_delay_seconds": ragas_delay_seconds,
            "ragas_retry_attempts": ragas_retry_attempts,
            "ragas_retry_delay_seconds": ragas_retry_delay_seconds,
            "ragas_judge_model": config.get_setting("RAGAS_JUDGE_MODEL", "gemini-2.5-flash-lite"),
        },
        "records": records,
    }
    report["summary"] = _summarize_run(report)
    return report


def _attach_ragas_scores(records: list[dict[str, Any]], ragas_scores: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        record["ragas"] = ragas_scores[index] if index < len(ragas_scores) else {}


def refresh_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute report summary after records are updated incrementally."""
    if report.get("report_type") == "pipeline_comparison":
        report["summary"] = [_summarize_run(run) for run in report.get("runs", [])]
    else:
        report["summary"] = _summarize_run(report)
    return report


def _answer_with_retries(
    *,
    answer_fn: Callable[..., dict[str, Any]],
    question: str,
    top_k: int,
    retrieval_mode: str,
    candidate_k: int | None,
    rerank_enabled: bool | None,
    mmr_enabled: bool | None,
    retry_attempts: int,
    retry_delay_seconds: float,
) -> tuple[dict[str, Any], str, int]:
    max_attempts = max(1, int(retry_attempts) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            result = answer_fn(
                question,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                candidate_k=candidate_k,
                rerank_enabled=rerank_enabled,
                mmr_enabled=mmr_enabled,
            )
            return result, "", attempt
        except Exception as exc:
            last_error = _preview(str(exc), 800)
            if attempt < max_attempts and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
    return {"answer": "", "sources": [], "retrieval_debug": {}}, last_error, max_attempts


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = _flatten_report_rows(report)
    fieldnames = [
        "pipeline_name",
        "pipeline_label",
        "sample_id",
        "question",
        "answer",
        "error",
        "attempt_count",
        "generation_skipped",
        "ground_truth",
        "expected_source_notes",
        "tags",
        "source_count",
        "source_ids",
        "source_titles",
        "retrieved_count",
        "reranked_count",
        "final_count",
        "retrieval_mode",
        "candidate_k",
        "rerank_enabled",
        "mmr_enabled",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "ragas_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = report.get("runs") if report.get("report_type") == "pipeline_comparison" else [report]
    rows: list[dict[str, Any]] = []
    for run in runs:
        config_data = run.get("config", {})
        for record in run.get("records", []):
            debug = record.get("retrieval_debug", {})
            ragas = record.get("ragas", {})
            rows.append(
                {
                    "pipeline_name": config_data.get("pipeline_name", ""),
                    "pipeline_label": config_data.get("pipeline_label", ""),
                    "sample_id": record.get("sample_id", ""),
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "error": record.get("error", ""),
                    "attempt_count": record.get("attempt_count", ""),
                    "generation_skipped": record.get("generation_skipped", ""),
                    "ground_truth": record.get("ground_truth", ""),
                    "expected_source_notes": record.get("expected_source_notes", ""),
                    "tags": _join(record.get("tags", [])),
                    "source_count": len(record.get("source_ids", [])),
                    "source_ids": _join(record.get("source_ids", [])),
                    "source_titles": _join(record.get("source_titles", [])),
                    "retrieved_count": debug.get("retrieved_count", ""),
                    "reranked_count": debug.get("reranked_count", ""),
                    "final_count": debug.get("final_count", ""),
                    "retrieval_mode": config_data.get("retrieval_mode", ""),
                    "candidate_k": config_data.get("candidate_k", ""),
                    "rerank_enabled": config_data.get("rerank_enabled", ""),
                    "mmr_enabled": config_data.get("mmr_enabled", ""),
                    "faithfulness": _csv_metric(ragas.get("faithfulness", "")),
                    "answer_relevancy": _csv_metric(ragas.get("answer_relevancy", "")),
                    "context_precision": _csv_metric(ragas.get("context_precision", "")),
                    "context_recall": _csv_metric(ragas.get("context_recall", "")),
                    "ragas_error": ragas.get("ragas_error", ""),
                }
            )
    return rows


def _render_comparison_markdown(report: dict[str, Any]) -> str:
    generate_values = {
        str(run.get("config", {}).get("generate_answers", True))
        for run in report.get("runs", [])
    }
    generate_label = ", ".join(sorted(generate_values)) if generate_values else "unknown"
    lines = [
        "# RAG Evaluation Benchmark Report",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Testset: `{report.get('testset_path', '')}`",
        f"- Questions evaluated per pipeline: `{report.get('sample_count', 0)}`",
        f"- Generate answers: `{generate_label}`",
        f"- Metrics: `{', '.join(report.get('metrics', DEFAULT_METRICS))}`",
        "",
        "## Pipeline Summary",
        "",
        "| Pipeline | Retrieval | Rerank | MMR | Questions | Avg Contexts | Avg Final Sources | RAGAS Metrics |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in report.get("summary", []):
        lines.append(
            "| {label} | `{retrieval}` | `{rerank}` | `{mmr}` | {questions} | {avg_contexts:.2f} | "
            "{avg_sources:.2f} | {metrics} |".format(
                label=_escape_md(summary.get("pipeline_label", "")),
                retrieval=summary.get("retrieval_mode", ""),
                rerank=summary.get("rerank_enabled", ""),
                mmr=summary.get("mmr_enabled", ""),
                questions=summary.get("question_count", 0),
                avg_contexts=float(summary.get("average_context_count", 0.0)),
                avg_sources=float(summary.get("average_source_count", 0.0)),
                metrics=_format_metric_summary(summary.get("ragas_average", {})),
            )
        )

    for run in report.get("runs", []):
        lines.extend(["", f"## {run.get('config', {}).get('pipeline_label', 'Pipeline')}", ""])
        lines.extend(_render_run_details(run))
    return "\n".join(lines) + "\n"


def _render_single_run_markdown(report: dict[str, Any]) -> str:
    config_data = report.get("config", {})
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Testset: `{report.get('testset_path', '')}`",
        f"- Pipeline: `{config_data.get('pipeline_label', config_data.get('pipeline_name', ''))}`",
        f"- Retrieval mode: `{config_data.get('retrieval_mode', '')}`",
        f"- Candidate K: `{config_data.get('candidate_k', '')}`",
        f"- Rerank enabled: `{config_data.get('rerank_enabled', '')}`",
        f"- MMR enabled: `{config_data.get('mmr_enabled', '')}`",
        f"- Generate answers: `{config_data.get('generate_answers', True)}`",
        f"- Request delay seconds: `{config_data.get('request_delay_seconds', 0)}`",
        f"- Retry attempts: `{config_data.get('retry_attempts', 0)}`",
        f"- Retry delay seconds: `{config_data.get('retry_delay_seconds', 0)}`",
        f"- Metrics: `{', '.join(report.get('metrics', DEFAULT_METRICS))}`",
        "",
        "## Summary",
        "",
        "| Questions | Avg Contexts | Avg Final Sources | RAGAS Metrics |",
        "| ---: | ---: | ---: | --- |",
    ]
    summary = report.get("summary", {})
    lines.append(
        "| {questions} | {avg_contexts:.2f} | {avg_sources:.2f} | {metrics} |".format(
            questions=summary.get("question_count", 0),
            avg_contexts=float(summary.get("average_context_count", 0.0)),
            avg_sources=float(summary.get("average_source_count", 0.0)),
            metrics=_format_metric_summary(summary.get("ragas_average", {})),
        )
    )
    lines.extend(["", "## Per-Question Results", ""])
    lines.extend(_render_run_details(report))
    return "\n".join(lines) + "\n"


def _render_run_details(run: dict[str, Any]) -> list[str]:
    lines = [
        "| ID | Question | Answer Preview | Sources | Debug | Attempts | RAGAS | Error |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in run.get("records", []):
        debug = record.get("retrieval_debug", {})
        debug_text = (
            f"retrieved={debug.get('retrieved_count', '')}; "
            f"reranked={debug.get('reranked_count', '')}; "
            f"final={debug.get('final_count', '')}"
        )
        source_text = "<br>".join(_escape_md(str(item)) for item in record.get("source_titles", [])[:5])
        lines.append(
            "| {sample_id} | {question} | {answer} | {sources} | `{debug}` | {attempts} | {ragas} | {error} |".format(
                sample_id=_escape_md(str(record.get("sample_id", ""))),
                question=_escape_md(str(record.get("question", ""))),
                answer=_escape_md(_preview(str(record.get("answer", "")), 180)),
                sources=source_text or "-",
                debug=debug_text,
                attempts=record.get("attempt_count", ""),
                ragas=_escape_md(_format_metric_summary(record.get("ragas", {}))),
                error=_escape_md(_preview(str(record.get("error", "")), 220)) or "-",
            )
        )
    return lines


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    records = run.get("records", [])
    config_data = run.get("config", {})
    question_count = len(records)
    context_counts = [len(record.get("contexts", [])) for record in records]
    source_counts = [len(record.get("source_ids", [])) for record in records]
    return {
        "pipeline_name": config_data.get("pipeline_name", ""),
        "pipeline_label": config_data.get("pipeline_label", ""),
        "retrieval_mode": config_data.get("retrieval_mode", ""),
        "candidate_k": config_data.get("candidate_k", ""),
        "rerank_enabled": config_data.get("rerank_enabled", ""),
        "mmr_enabled": config_data.get("mmr_enabled", ""),
        "question_count": question_count,
        "average_context_count": _average(context_counts),
        "average_source_count": _average(source_counts),
        "ragas_average": _average_ragas(records),
    }


def _average_ragas(records: list[dict[str, Any]]) -> dict[str, float]:
    averages: dict[str, float] = {}
    for metric in DEFAULT_METRICS:
        values: list[float] = []
        for record in records:
            value = record.get("ragas", {}).get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            averages[metric] = _average(values)
    return averages


def _source_title(source: dict[str, Any]) -> str:
    metadata = source.get("metadata") or {}
    source_type = metadata.get("source_type", "")
    if source_type == "pdf":
        return f"PDF:{metadata.get('source_file', '')}:page={metadata.get('page_number', '')}:id={source.get('id', '')}"
    if source_type == "notion":
        return f"Notion:{metadata.get('title', '')}:heading={metadata.get('heading_path', '')}:id={source.get('id', '')}"
    return str(source.get("id", ""))


def _normalize_formats(formats: list[str]) -> list[str]:
    clean: list[str] = []
    for item in formats:
        normalized = item.strip().lower().lstrip(".")
        if not normalized:
            continue
        if normalized == "markdown":
            normalized = "md"
        if normalized not in {"json", "csv", "md"}:
            raise ValueError("Report formats must be a subset of json,csv,md")
        if normalized not in clean:
            clean.append(normalized)
    return clean or list(DEFAULT_REPORT_FORMATS)


def _format_metric_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "not run"
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                parts.append(f"{key}={float(value):.3f}")
        elif value not in {"", None}:
            parts.append(f"{key}={_preview(str(value), 80)}")
    return ", ".join(parts) if parts else "not run"


def _average(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _join(values: Any) -> str:
    if isinstance(values, list):
        return " | ".join(str(value) for value in values)
    return str(values or "")


def _csv_metric(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else ""
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _preview(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
