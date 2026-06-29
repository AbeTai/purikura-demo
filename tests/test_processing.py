from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from purikura_demo.processing import FaceRegion, PurikuraSettings, _build_skin_mask, apply_purikura_effect


def test_apply_purikura_effect_returns_jpeg() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(decorations=True))

    image = Image.open(io.BytesIO(result.image_bytes))
    original = Image.open(io.BytesIO(result.original_bytes))
    segmentation = Image.open(io.BytesIO(result.segmentation_bytes))
    assert image.format == "JPEG"
    assert original.format == "JPEG"
    assert segmentation.format == "JPEG"
    assert image.size == (420, 560)
    assert original.size == (420, 560)
    assert segmentation.size == (420, 560)
    assert result.metrics["width"] == 420
    assert result.metrics["height"] == 560


def test_unknown_preset_falls_back_to_strawberry() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(preset="missing"))

    assert result.metrics["preset"] == "strawberry"


def test_strong_effect_mode_is_reported() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(effect_mode="max", eye_enlarge=1.0, face_slim=1.0))

    assert result.metrics["mode"] == "max"


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
