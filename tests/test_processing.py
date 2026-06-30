from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from purikura_demo.processing import (
    FACE_OVAL,
    LEFT_EYE,
    LIPS,
    NOSE,
    FaceRegion,
    PurikuraSettings,
    build_feathered_region,
    _build_part_masks,
    _build_person_mask,
    _build_skin_mask,
    _extract_rembg_alpha,
    _refine_hair_mask,
    suppress_mask_on_edges,
    apply_purikura_effect,
)


@pytest.fixture(autouse=True)
def disable_rembg_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", lambda image, model_name: None)


def test_apply_purikura_effect_returns_jpeg() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(decorations=True))

    image = Image.open(io.BytesIO(result.image_bytes))
    original = Image.open(io.BytesIO(result.original_bytes))
    segmentation = Image.open(io.BytesIO(result.segmentation_bytes))
    background = Image.open(io.BytesIO(result.background_bytes))
    assert image.format == "JPEG"
    assert original.format == "JPEG"
    assert segmentation.format == "JPEG"
    assert background.format == "JPEG"
    assert image.size == (420, 560)
    assert original.size == (420, 560)
    assert segmentation.size == (420, 560)
    assert background.size == (420, 560)
    assert result.metrics["width"] == 420
    assert result.metrics["height"] == 560
    assert result.metrics["background"] == "white"


def test_unknown_preset_falls_back_to_strawberry() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(preset="missing"))

    assert result.metrics["preset"] == "strawberry"


def test_strong_effect_mode_is_reported() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(effect_mode="ultra", eye_enlarge=1.0, face_slim=1.0))

    assert result.metrics["mode"] == "ultra"
    assert result.metrics["pipeline"] == "quality"
    assert result.metrics["preset"] == "sample_match"
    assert result.metrics["accelerator"] in {"opencv-cpu", "torch-mps"}
    assert result.metrics["segmenter"] in {"mediapipe-face-mesh", "opencv-haar-fallback", "fallback-soft-mask"}
    assert result.metrics["background_segmenter"] in {
        "rembg-birefnet-portrait",
        "rembg-isnet-general-use",
        "mediapipe-selfie-segmentation",
        "face-fallback-person-mask",
    }


def test_skin_mask_combines_multiple_faces() -> None:
    image = Image.new("RGB", (640, 360), (235, 220, 210))
    rgb = np.array(image)
    faces = [
        FaceRegion(90, 60, 150, 190, ((140.0, 130.0, 24.0), (190.0, 130.0, 24.0))),
        FaceRegion(380, 70, 150, 190, ((430.0, 140.0, 24.0), (480.0, 140.0, 24.0))),
    ]

    mask = _build_skin_mask(rgb, faces)

    assert mask[145, 165] > 0
    assert mask[155, 455] > 0


def test_person_mask_combines_multiple_faces() -> None:
    rgb = np.full((360, 640, 3), 235, dtype=np.uint8)
    faces = [
        FaceRegion(90, 60, 150, 190, ((140.0, 130.0, 24.0), (190.0, 130.0, 24.0))),
        FaceRegion(380, 70, 150, 190, ((430.0, 140.0, 24.0), (480.0, 140.0, 24.0))),
    ]

    mask, segmenter = _build_person_mask(rgb, faces)

    assert mask[150, 165] > 0
    assert mask[160, 455] > 0
    assert segmenter in {
        "rembg-birefnet-portrait",
        "rembg-isnet-general-use",
        "mediapipe-selfie-segmentation",
        "face-fallback-person-mask",
    }


def test_person_mask_uses_rembg_alpha(monkeypatch) -> None:
    rgb = np.full((80, 100, 3), 235, dtype=np.uint8)
    rembg_alpha = np.zeros((80, 100), dtype=np.float32)
    rembg_alpha[20:60, 25:75] = 1.0

    def fake_rembg(image: np.ndarray, model_name: str) -> np.ndarray | None:
        assert image.shape == rgb.shape
        return rembg_alpha if model_name == "birefnet-portrait" else None

    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", fake_rembg)
    monkeypatch.setattr("purikura_demo.processing._detect_person_with_mediapipe", lambda image: None)

    mask, segmenter = _build_person_mask(rgb, [])

    assert segmenter == "rembg-birefnet-portrait"
    assert mask[40, 50] > 200
    assert mask[5, 5] < 16


def test_rembg_mask_does_not_promote_face_fallback_outline(monkeypatch) -> None:
    rgb = np.full((180, 180, 3), 235, dtype=np.uint8)
    rembg_alpha = np.zeros((180, 180), dtype=np.float32)
    rembg_alpha[70:155, 65:115] = 1.0
    face = FaceRegion(35, 20, 100, 120, ((70.0, 75.0, 14.0), (100.0, 75.0, 14.0)))

    def fake_rembg(image: np.ndarray, model_name: str) -> np.ndarray | None:
        assert image.shape == rgb.shape
        return rembg_alpha if model_name == "birefnet-portrait" else None

    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", fake_rembg)
    monkeypatch.setattr("purikura_demo.processing._detect_person_with_mediapipe", lambda image: None)

    mask, segmenter = _build_person_mask(rgb, [face])

    assert segmenter == "rembg-birefnet-portrait"
    assert mask[100, 90] > 200
    assert mask[35, 85] < 48


def test_person_mask_falls_back_when_rembg_unavailable(monkeypatch) -> None:
    rgb = np.full((160, 160, 3), 235, dtype=np.uint8)
    face = FaceRegion(45, 30, 70, 90, ((70.0, 65.0, 12.0), (92.0, 65.0, 12.0)))
    monkeypatch.setattr("purikura_demo.processing._run_rembg_model", lambda image, model_name: None)
    monkeypatch.setattr("purikura_demo.processing._detect_person_with_mediapipe", lambda image: None)

    mask, segmenter = _build_person_mask(rgb, [face])

    assert segmenter == "face-fallback-person-mask"
    assert mask[75, 80] > 0


def test_extract_rembg_alpha_accepts_rgba() -> None:
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[4:16, 7:23, 3] = 240

    alpha = _extract_rembg_alpha(rgba, (20, 30))

    assert alpha is not None
    assert alpha[10, 15] > 0.9
    assert alpha[0, 0] == 0.0


def test_part_masks_use_landmark_polygons() -> None:
    landmarks = np.zeros((478, 2), dtype=np.float32)
    _assign_polygon(landmarks, FACE_OVAL, [(120, 80), (260, 80), (280, 210), (190, 300), (100, 210)])
    _assign_polygon(landmarks, LEFT_EYE, [(205, 145), (235, 145), (240, 162), (220, 172), (200, 162)])
    _assign_polygon(landmarks, NOSE, [(185, 165), (205, 170), (215, 225), (175, 225)])
    _assign_polygon(landmarks, LIPS, [(160, 245), (220, 245), (235, 265), (190, 280), (145, 265)])
    face = FaceRegion(100, 80, 180, 220, ((220.0, 158.0, 20.0), (160.0, 158.0, 20.0)), landmarks, "mediapipe-face-mesh")

    masks = _build_part_masks((360, 420), [face])

    assert masks.face_skin[190, 190] > 0
    assert masks.eyes[158, 220] > 0
    assert masks.nose.max() > 0
    assert masks.lips.max() > 0


def test_fallback_part_masks_include_side_hair() -> None:
    face = FaceRegion(100, 80, 180, 220, ((220.0, 158.0, 20.0), (160.0, 158.0, 20.0)))

    masks = _build_part_masks((420, 420), [face])

    assert masks.hair[260, 115] > 0
    assert masks.hair[260, 265] > 0


def test_feathered_region_has_core_and_transition() -> None:
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[30:90, 30:90] = 255

    region = build_feathered_region(mask, inner_px=12, outer_px=30)

    assert region.core[60, 60] > 0.9
    assert 0.0 < region.transition[31, 60] < region.core[60, 60]
    assert region.alpha[29, 60] > 0.0
    assert region.alpha[5, 5] == 0.0


def test_edge_suppression_reduces_mask_on_strong_edges() -> None:
    rgb = np.full((80, 80, 3), 230, dtype=np.uint8)
    rgb[:, 40:] = 25
    mask = np.full((80, 80), 255, dtype=np.uint8)

    suppressed = suppress_mask_on_edges(mask, rgb)

    assert suppressed[40, 40] < suppressed[40, 20]


def test_hair_refinement_keeps_dark_hair_over_skin_overlap() -> None:
    rgb = np.full((160, 160, 3), (226, 190, 172), dtype=np.uint8)
    rgb[20:130, 25:70] = (40, 30, 28)
    rough_hair = np.zeros((160, 160), dtype=np.uint8)
    rough_hair[15:135, 20:76] = 180
    skin_mask = np.zeros((160, 160), dtype=np.uint8)
    skin_mask[30:135, 45:120] = 220

    hair = _refine_hair_mask(rgb, rough_hair, skin_mask, PurikuraSettings())

    assert hair[70, 50] > 0


def _sample_face_image() -> bytes:
    image = Image.new("RGB", (420, 560), (238, 226, 222))
    draw = ImageDraw.Draw(image)
    draw.ellipse((105, 95, 315, 345), fill=(230, 186, 164), outline=(120, 80, 70), width=3)
    draw.ellipse((155, 180, 195, 210), fill=(28, 25, 28))
    draw.ellipse((225, 180, 265, 210), fill=(28, 25, 28))
    draw.arc((170, 230, 250, 300), start=20, end=160, fill=(145, 65, 80), width=4)
    draw.rectangle((80, 350, 340, 560), fill=(196, 210, 224))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _assign_polygon(landmarks: np.ndarray, indices: tuple[int, ...], points: list[tuple[int, int]]) -> None:
    for offset, index in enumerate(indices):
        landmarks[index] = points[offset % len(points)]
