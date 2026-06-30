from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from purikura_demo.app import app


@pytest.fixture(autouse=True)
def disable_rembg_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", lambda image, model_name: None)


def test_index_renders() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Purikura Demo" in response.text
    assert "hx-post=\"/process\"" in response.text
    assert "name=\"pipeline\"" in response.text
    assert "name=\"effect_mode\"" in response.text
    assert "name=\"white_background\"" in response.text


def test_process_image_returns_result_partial() -> None:
    client = TestClient(app)
    image_bytes = _sample_image()
    response = client.post(
        "/process",
        files={"image": ("sample.png", image_bytes, "image/png")},
        data={"preset": "natural", "pipeline": "quality", "effect_mode": "strong", "decorations": "true"},
    )

    assert response.status_code == 200
    assert "data:image/jpeg;base64" in response.text
    assert "Original" in response.text
    assert "Processed" in response.text
    assert "Segmentation Debug" in response.text
    assert "Background Debug" in response.text
    assert "BG Segmenter" in response.text
    assert "Segmenter" in response.text
    assert "Download" in response.text
    assert "strong" in response.text
    assert "quality" in response.text


def _sample_image() -> bytes:
    image = Image.new("RGB", (320, 240), (220, 190, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
