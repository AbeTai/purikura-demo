from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from purikura_demo.app import app
from purikura_demo.processing import FACE_OVAL, LEFT_EYE, LIPS, NOSE, RIGHT_EYE, FaceRegion


@pytest.fixture(autouse=True)
def disable_rembg_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", lambda image, model_name: None)
    monkeypatch.setattr("purikura_demo.processing._detect_faces_with_mediapipe_tasks", lambda image, person_mask=None: [])


def test_index_renders() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Purikura Demo" in response.text
    assert "hx-post=\"/process\"" in response.text
    assert "name=\"pipeline\"" in response.text
    assert "name=\"effect_mode\"" in response.text
    assert "name=\"white_background\"" in response.text
    assert "data-input-mode=\"camera\"" in response.text
    assert "name=\"camera_image\"" in response.text
    assert "name=\"camera_landmarks\"" in response.text
    assert "id=\"camera-overlay\"" in response.text
    assert "camera.js" in response.text


def test_camera_script_requires_browser_face_landmarker() -> None:
    script = _camera_script()

    assert "@mediapipe/tasks-vision" in script
    assert "FACE_MODEL_URL" in script
    assert "FaceLandmarker.createFromOptions" in script
    assert "detectForVideo" in script
    assert "numFaces: 8" in script
    assert "顔を中央に入れてください" in script


def test_camera_script_blocks_submit_until_face_detected() -> None:
    script = _camera_script()

    assert "lastFaceBox" in script
    assert "captureButton.disabled = !(stream && detectorReady && lastFaceBox)" in script
    assert "顔を検出してから撮影してください" in script
    assert "OUTPUT_WIDTH = 960" in script
    assert "OUTPUT_HEIGHT = 1200" in script
    assert "cropFromFaceBox" in script
    assert "landmarksForOutputImage" in script
    assert "#camera_landmarks" in script


def test_process_image_returns_result_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_quality_pipeline(monkeypatch)
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


def test_process_camera_image_returns_result_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_quality_pipeline(monkeypatch)
    client = TestClient(app)
    image_data = base64.b64encode(_sample_image()).decode("ascii")
    response = client.post(
        "/process",
        data={
            "camera_image": f"data:image/png;base64,{image_data}",
            "preset": "natural",
            "pipeline": "quality",
            "effect_mode": "strong",
        },
    )

    assert response.status_code == 200
    assert "data:image/jpeg;base64" in response.text
    assert "Processed" in response.text
    assert "Background Debug" in response.text


def test_process_camera_image_uses_browser_landmarks_when_server_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_rembg(image: np.ndarray, model_name: str) -> np.ndarray:
        assert model_name == "birefnet-portrait"
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        height, width = image.shape[:2]
        mask[round(height * 0.08) : round(height * 0.96), round(width * 0.12) : round(width * 0.88)] = 1.0
        return mask

    monkeypatch.setattr("purikura_demo.processing._detect_faces_and_eyes", lambda image, person_mask=None: [])
    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", fake_rembg)
    client = TestClient(app)
    image_data = base64.b64encode(_sample_image()).decode("ascii")
    response = client.post(
        "/process",
        data={
            "camera_image": f"data:image/png;base64,{image_data}",
            "camera_landmarks": json.dumps([_sample_normalized_landmarks()]),
            "preset": "natural",
            "pipeline": "quality",
            "effect_mode": "strong",
        },
    )

    assert response.status_code == 200
    assert "data:image/jpeg;base64" in response.text
    assert "Processed" in response.text
    assert "mediapipe-face-mesh" in response.text


def test_process_image_returns_quality_error_without_required_models() -> None:
    client = TestClient(app)
    response = client.post(
        "/process",
        files={"image": ("sample.png", _sample_image(), "image/png")},
        data={"preset": "natural", "pipeline": "quality", "effect_mode": "strong"},
    )

    assert response.status_code == 200
    assert "処理できませんでした" in response.text
    assert "birefnet-portrait" in response.text


def test_process_camera_image_returns_specific_face_detection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_rembg(image: np.ndarray, model_name: str) -> np.ndarray:
        assert model_name == "birefnet-portrait"
        return np.ones(image.shape[:2], dtype=np.float32)

    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", fake_rembg)
    monkeypatch.setattr("purikura_demo.processing._detect_faces_and_eyes", lambda image, person_mask=None: [])
    client = TestClient(app)
    image_data = base64.b64encode(_sample_image()).decode("ascii")
    response = client.post(
        "/process",
        data={
            "camera_image": f"data:image/png;base64,{image_data}",
            "preset": "natural",
            "pipeline": "quality",
            "effect_mode": "strong",
        },
    )

    assert response.status_code == 200
    assert "処理できませんでした" in response.text
    assert "撮影前チェックは通ったが" in response.text
    assert "送信画像サイズ: 320x240" in response.text


def _camera_script() -> str:
    return (Path(__file__).resolve().parents[1] / "src/purikura_demo/static/camera.js").read_text()


def _sample_image() -> bytes:
    image = Image.new("RGB", (320, 240), (220, 190, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_normalized_landmarks() -> list[dict[str, float]]:
    points = [{"x": 0.50, "y": 0.45} for _ in range(478)]
    _assign_normalized_polygon(
        points,
        FACE_OVAL,
        [
            (0.50, 0.20),
            (0.72, 0.34),
            (0.66, 0.62),
            (0.50, 0.72),
            (0.34, 0.62),
            (0.28, 0.34),
        ],
    )
    _assign_normalized_polygon(points, LEFT_EYE, [(0.56, 0.40), (0.65, 0.44)])
    _assign_normalized_polygon(points, RIGHT_EYE, [(0.35, 0.40), (0.44, 0.44)])
    _assign_normalized_polygon(points, NOSE, [(0.46, 0.47), (0.54, 0.58)])
    _assign_normalized_polygon(points, LIPS, [(0.42, 0.62), (0.58, 0.67)])
    _assign_normalized_polygon(points, (469, 470, 471, 472), [(0.395, 0.42)])
    _assign_normalized_polygon(points, (474, 475, 476, 477), [(0.605, 0.42)])
    return points


def _assign_normalized_polygon(points: list[dict[str, float]], indices: tuple[int, ...], polygon: list[tuple[float, float]]) -> None:
    for offset, index in enumerate(indices):
        x, y = polygon[offset % len(polygon)]
        points[index] = {"x": x, "y": y}


def _install_quality_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_faces(image: np.ndarray) -> list[FaceRegion]:
        return [_sample_mesh_face(image.shape[1], image.shape[0])]

    def fake_rembg(image: np.ndarray, model_name: str) -> np.ndarray:
        assert model_name == "birefnet-portrait"
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        height, width = image.shape[:2]
        mask[round(height * 0.08) : round(height * 0.96), round(width * 0.12) : round(width * 0.88)] = 1.0
        return mask

    monkeypatch.setattr("purikura_demo.processing._detect_faces_and_eyes", fake_faces)
    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", fake_rembg)


def _sample_mesh_face(width: int, height: int) -> FaceRegion:
    face_w = round(width * 0.48)
    face_h = round(height * 0.50)
    face_x = round((width - face_w) * 0.5)
    face_y = round(height * 0.14)
    landmarks = np.zeros((478, 2), dtype=np.float32)
    _assign_polygon(
        landmarks,
        FACE_OVAL,
        [
            (face_x + face_w * 0.50, face_y),
            (face_x + face_w * 0.92, face_y + face_h * 0.28),
            (face_x + face_w * 0.82, face_y + face_h * 0.74),
            (face_x + face_w * 0.50, face_y + face_h),
            (face_x + face_w * 0.18, face_y + face_h * 0.74),
            (face_x + face_w * 0.08, face_y + face_h * 0.28),
        ],
    )
    _assign_polygon(landmarks, LEFT_EYE, [(face_x + face_w * 0.58, face_y + face_h * 0.38), (face_x + face_w * 0.72, face_y + face_h * 0.46)])
    _assign_polygon(landmarks, RIGHT_EYE, [(face_x + face_w * 0.28, face_y + face_h * 0.38), (face_x + face_w * 0.42, face_y + face_h * 0.46)])
    _assign_polygon(landmarks, NOSE, [(face_x + face_w * 0.46, face_y + face_h * 0.48), (face_x + face_w * 0.58, face_y + face_h * 0.70)])
    _assign_polygon(landmarks, LIPS, [(face_x + face_w * 0.36, face_y + face_h * 0.78), (face_x + face_w * 0.64, face_y + face_h * 0.86)])
    left_eye = (face_x + face_w * 0.65, face_y + face_h * 0.42, face_w * 0.08)
    right_eye = (face_x + face_w * 0.35, face_y + face_h * 0.42, face_w * 0.08)
    return FaceRegion(face_x, face_y, face_w, face_h, (left_eye, right_eye), landmarks, "mediapipe-face-mesh")


def _assign_polygon(landmarks: np.ndarray, indices: tuple[int, ...], points: list[tuple[float, float]]) -> None:
    for offset, index in enumerate(indices):
        landmarks[index] = points[offset % len(points)]
