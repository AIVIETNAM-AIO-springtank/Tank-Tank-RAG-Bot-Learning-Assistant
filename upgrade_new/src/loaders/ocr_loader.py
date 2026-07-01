"""Gemini Vision OCR helpers for PDF and Notion image content."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Callable

import requests

from upgrade_new.src import config


DEFAULT_VISION_PROMPT = (
    "Extract all useful learning content from this image for a RAG system. "
    "Preserve Vietnamese/English text, code, formulas, table-like structure, "
    "and diagram labels. Return concise markdown/plain text only."
)


def extract_text_from_image(
    image_path_or_url: str,
    prompt: str | None = None,
    *,
    request_fn: Callable[..., Any] = requests.post,
    get_fn: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    """Extract text from a local image path or URL using Gemini Vision.

    The function returns a structured result instead of raising for API/content
    failures so ingestion can degrade gracefully.
    """
    if not config.GEMINI_API_KEYS:
        return _error_result("missing Gemini API key")

    try:
        image_bytes, mime_type = _read_image_bytes(image_path_or_url, get_fn=get_fn)
    except Exception as exc:
        return _error_result(f"could not read image: {exc}")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt or DEFAULT_VISION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1024},
    }

    errors: list[str] = []
    for index, api_key in enumerate(config.GEMINI_API_KEYS, start=1):
        try:
            response = request_fn(_gemini_vision_url(api_key), json=payload, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            errors.append(f"key {index}/{len(config.GEMINI_API_KEYS)} request error: {_safe_error_detail(str(exc))}")
            continue

        if response.status_code >= 400:
            errors.append(
                f"key {index}/{len(config.GEMINI_API_KEYS)} Gemini Vision error "
                f"{response.status_code}: {_safe_error_detail(response.text)}"
            )
            continue

        data = response.json()
        text = _extract_candidate_text(data)
        return {
            "text": text,
            "error": "",
            "provider": "gemini_vision",
            "model": config.GEMINI_VISION_MODEL,
            "mime_type": mime_type,
        }

    return _error_result(" | ".join(errors) or "Gemini Vision returned no usable response")


def extract_text_from_image_bytes(
    image_bytes: bytes,
    mime_type: str = "image/png",
    prompt: str | None = None,
    *,
    request_fn: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    """Extract text from image bytes using Gemini Vision."""
    if not config.GEMINI_API_KEYS:
        return _error_result("missing Gemini API key")
    if not image_bytes:
        return _error_result("empty image bytes")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt or DEFAULT_VISION_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1024},
    }

    errors: list[str] = []
    for index, api_key in enumerate(config.GEMINI_API_KEYS, start=1):
        try:
            response = request_fn(_gemini_vision_url(api_key), json=payload, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            errors.append(f"key {index}/{len(config.GEMINI_API_KEYS)} request error: {_safe_error_detail(str(exc))}")
            continue
        if response.status_code >= 400:
            errors.append(
                f"key {index}/{len(config.GEMINI_API_KEYS)} Gemini Vision error "
                f"{response.status_code}: {_safe_error_detail(response.text)}"
            )
            continue
        return {
            "text": _extract_candidate_text(response.json()),
            "error": "",
            "provider": "gemini_vision",
            "model": config.GEMINI_VISION_MODEL,
            "mime_type": mime_type,
        }

    return _error_result(" | ".join(errors) or "Gemini Vision returned no usable response")


def _read_image_bytes(image_path_or_url: str, *, get_fn: Callable[..., Any]) -> tuple[bytes, str]:
    if image_path_or_url.startswith(("http://", "https://")):
        response = get_fn(image_path_or_url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "").split(";")[0] or "image/png"
        return response.content, mime_type

    path = Path(image_path_or_url)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return path.read_bytes(), mime_type


def _extract_candidate_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()


def _gemini_vision_url(api_key: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_VISION_MODEL}:generateContent?key={api_key}"
    )


def _safe_error_detail(detail: str) -> str:
    clean = detail.replace("\n", " ").strip()
    for secret in list(config.GEMINI_API_KEYS) + [config.GEMINI_API_KEY]:
        if secret:
            clean = clean.replace(secret, "<redacted>")
    return clean[:500]


def _error_result(error: str) -> dict[str, Any]:
    return {
        "text": "",
        "error": error,
        "provider": "gemini_vision",
        "model": config.GEMINI_VISION_MODEL,
        "mime_type": "",
    }
