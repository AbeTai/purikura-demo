from __future__ import annotations

import io

from PIL import Image, ImageDraw

from purikura_demo.processing import PurikuraSettings, apply_purikura_effect


def test_apply_purikura_effect_returns_jpeg() -> None:
    source = _sample_face_image()
    result = apply_purikura_effect(source, PurikuraSettings(decorations=True))

    image = Image.open(io.BytesIO(result.image_bytes))
    assert image.format == "JPEG"
    assert image.size == (420, 560)
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
