"""Tests for Gemini Vision OCR helpers."""

from __future__ import annotations

from pathlib import Path

from upgrade_new.src.loaders.ocr_loader import extract_text_from_image


class FakeResponse:
    def __init__(self, status_code: int, data: dict | None = None, text: str = "", content: bytes = b"") -> None:
        self.status_code = status_code
        self._data = data or {}
        self.text = text
        self.content = content
        self.headers = {"Content-Type": "image/png"}

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.text or f"status {self.status_code}")


def test_extract_text_from_image_uses_gemini_inline_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("upgrade_new.src.config.GEMINI_API_KEYS", ["key-1"])
    monkeypatch.setattr("upgrade_new.src.config.GEMINI_VISION_MODEL", "vision-test")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")
    seen_payloads: list[dict] = []

    def fake_post(url: str, json: dict, timeout: int):
        seen_payloads.append(json)
        return FakeResponse(
            200,
            {
                "candidates": [
                    {"content": {"parts": [{"text": "Detected OCR text"}]}},
                ]
            },
        )

    result = extract_text_from_image(str(image_path), request_fn=fake_post)

    assert result["text"] == "Detected OCR text"
    assert result["error"] == ""
    assert seen_payloads[0]["contents"][0]["parts"][1]["inline_data"]["data"]


def test_extract_text_from_image_returns_error_on_api_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("upgrade_new.src.config.GEMINI_API_KEYS", ["key-1"])
    image_path = Path(tmp_path / "image.png")
    image_path.write_bytes(b"fake-image")

    result = extract_text_from_image(
        str(image_path),
        request_fn=lambda url, json, timeout: FakeResponse(403, text="permission denied"),
    )

    assert result["text"] == ""
    assert "Gemini Vision error 403" in result["error"]
