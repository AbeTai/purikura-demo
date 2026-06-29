from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class PurikuraSettings:
    preset: str = "strawberry"
    purikura_intensity: float = 0.78
    skin_smoothing: float = 0.72
    eye_enlarge: float = 0.55
    face_slim: float = 0.42
    glow: float = 0.55
    decorations: bool = True

    @staticmethod
    def available_presets() -> tuple[tuple[str, str], ...]:
        return (
            ("strawberry", "いちごミルク"),
            ("natural", "ナチュラル盛れ"),
            ("cool", "透明感クール"),
            ("film", "フィルムポップ"),
            ("neon", "夜景ネオン"),
        )


@dataclass(frozen=True)
class ProcessedImage:
    image_bytes: bytes
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    eyes: tuple[tuple[float, float, float], ...]

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w * 0.5, self.y + self.h * 0.52)


def apply_purikura_effect(source_bytes: bytes, settings: PurikuraSettings) -> ProcessedImage:
    image = _decode_image(source_bytes)
    image = _resize_to_limit(image, max_side=1800)
    rgb = np.array(image, dtype=np.uint8)

    settings = _clamp_settings(settings)
    faces = _detect_faces_and_eyes(rgb)

    warped = rgb
    if faces:
        warped = _apply_geometry_warp(warped, faces, settings)

    retouched = _apply_skin_retouch(warped, faces, settings)
    toned = _apply_color_preset(retouched, settings)
    polished = _apply_glow_and_grain(toned, settings)
    decorated = _draw_decorations(polished, settings) if settings.decorations else polished

    output = Image.fromarray(decorated).convert("RGB")
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=94, optimize=True)
    return ProcessedImage(
        image_bytes=buffer.getvalue(),
        metrics={
            "width": output.width,
            "height": output.height,
            "faces": len(faces),
            "preset": settings.preset,
        },
    )


def _decode_image(source_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(source_bytes))
        image.load()
    except Exception as exc:  # noqa: BLE001 - convert decoder details into a user-facing error.
        raise ValueError("画像を読み込めませんでした。JPEG / PNG / WebP などを指定してください。") from exc
    return image.convert("RGB")


def _resize_to_limit(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def _clamp_settings(settings: PurikuraSettings) -> PurikuraSettings:
    def unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    preset_names = {key for key, _ in PurikuraSettings.available_presets()}
    preset = settings.preset if settings.preset in preset_names else "strawberry"
    return PurikuraSettings(
        preset=preset,
        purikura_intensity=unit(settings.purikura_intensity),
        skin_smoothing=unit(settings.skin_smoothing),
        eye_enlarge=unit(settings.eye_enlarge),
        face_slim=unit(settings.face_slim),
        glow=unit(settings.glow),
        decorations=bool(settings.decorations),
    )


def _detect_faces_and_eyes(rgb: np.ndarray) -> list[FaceRegion]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    detected = face_detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(80, 80))
    faces: list[FaceRegion] = []
    for x, y, w, h in detected[:6]:
        face_gray = gray[y : y + h, x : x + w]
        upper = face_gray[: max(1, int(h * 0.62)), :]
        raw_eyes = eye_detector.detectMultiScale(
            upper,
            scaleFactor=1.08,
            minNeighbors=7,
            minSize=(max(16, w // 12), max(10, h // 16)),
        )
        eyes = _normalize_eyes(raw_eyes, x, y, w, h)
        faces.append(FaceRegion(int(x), int(y), int(w), int(h), eyes))
    return sorted(faces, key=lambda face: face.w * face.h, reverse=True)


def _normalize_eyes(
    raw_eyes: np.ndarray,
    face_x: int,
    face_y: int,
    face_w: int,
    face_h: int,
) -> tuple[tuple[float, float, float], ...]:
    candidates: list[tuple[float, float, float]] = []
    for ex, ey, ew, eh in raw_eyes:
        cx = face_x + ex + ew * 0.5
        cy = face_y + ey + eh * 0.5
        if face_y + face_h * 0.16 <= cy <= face_y + face_h * 0.58:
            candidates.append((cx, cy, max(ew, eh) * 0.88))

    if len(candidates) >= 2:
        candidates = sorted(candidates, key=lambda item: item[2], reverse=True)[:4]
        left = min(candidates, key=lambda item: item[0])
        right = max(candidates, key=lambda item: item[0])
        if abs(left[0] - right[0]) > face_w * 0.18:
            return (left, right)

    radius = face_w * 0.18
    return (
        (face_x + face_w * 0.34, face_y + face_h * 0.40, radius),
        (face_x + face_w * 0.66, face_y + face_h * 0.40, radius),
    )


def _apply_geometry_warp(rgb: np.ndarray, faces: list[FaceRegion], settings: PurikuraSettings) -> np.ndarray:
    height, width = rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    for face in faces:
        if settings.face_slim > 0:
            _add_face_slim_map(map_x, face, settings.face_slim)
        if settings.eye_enlarge > 0:
            for cx, cy, radius in face.eyes:
                _add_eye_enlarge_map(map_x, map_y, (cx, cy), radius * 1.28, settings.eye_enlarge)

    return cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _add_eye_enlarge_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
    center: tuple[float, float],
    radius: float,
    strength: float,
) -> None:
    cx, cy = center
    alpha = 0.04 + 0.20 * strength
    beta = 2.6
    x0 = max(int(cx - radius), 0)
    x1 = min(int(cx + radius) + 1, map_x.shape[1])
    y0 = max(int(cy - radius), 0)
    y1 = min(int(cy + radius) + 1, map_x.shape[0])

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - cx
    dy = yy.astype(np.float32) - cy
    distance = np.sqrt(dx * dx + dy * dy) / max(radius, 1.0)
    mask = distance < 1.0
    scale = 1.0 + alpha * np.power(1.0 - distance, beta)

    region_x = map_x[y0:y1, x0:x1]
    region_y = map_y[y0:y1, x0:x1]
    region_x[mask] = cx + dx[mask] / scale[mask]
    region_y[mask] = cy + dy[mask] / scale[mask]


def _add_face_slim_map(map_x: np.ndarray, face: FaceRegion, strength: float) -> None:
    cx, cy = face.center
    rx = face.w * 0.54
    ry = face.h * 0.58
    x0 = max(int(cx - rx), 0)
    x1 = min(int(cx + rx) + 1, map_x.shape[1])
    y0 = max(int(cy - ry * 0.35), 0)
    y1 = min(int(cy + ry) + 1, map_x.shape[0])

    yy, xx = np.mgrid[y0:y1, x0:x1]
    nx = (xx.astype(np.float32) - cx) / max(rx, 1.0)
    ny = (yy.astype(np.float32) - cy) / max(ry, 1.0)
    ellipse = nx * nx + ny * ny
    lower_weight = np.clip((ny + 0.18) / 1.18, 0.0, 1.0)
    side_weight = np.clip(1.0 - np.abs(nx) ** 1.8, 0.0, 1.0)
    mask = ellipse < 1.0
    shrink = 0.04 + 0.17 * strength
    scale = 1.0 - shrink * lower_weight * side_weight
    scale = np.clip(scale, 0.76, 1.0)

    region_x = map_x[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - cx
    region_x[mask] = cx + dx[mask] / scale[mask]


def _apply_skin_retouch(rgb: np.ndarray, faces: list[FaceRegion], settings: PurikuraSettings) -> np.ndarray:
    if settings.skin_smoothing <= 0:
        return rgb

    mask = _build_skin_mask(rgb, faces)
    if mask.max() == 0:
        mask = _soft_full_image_mask(rgb.shape[:2])

    smoothed = cv2.bilateralFilter(rgb, d=9, sigmaColor=38 + settings.skin_smoothing * 34, sigmaSpace=7)
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=0.7 + settings.skin_smoothing * 1.5)
    detail = cv2.addWeighted(rgb, 1.18, cv2.GaussianBlur(rgb, (0, 0), 2.0), -0.18, 0)
    retouch = cv2.addWeighted(smoothed, 0.84, detail, 0.16, 0)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * (0.28 + 0.56 * settings.skin_smoothing)
    return np.clip(rgb.astype(np.float32) * (1.0 - alpha) + retouch.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _build_skin_mask(rgb: np.ndarray, faces: list[FaceRegion]) -> np.ndarray:
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.52))
        axes = (round(face.w * 0.43), round(face.h * 0.50))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        for cx, cy, radius in face.eyes:
            cv2.circle(mask, (round(cx), round(cy)), round(radius * 0.92), 0, -1)
        mouth_center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.73))
        cv2.ellipse(mask, mouth_center, (round(face.w * 0.18), round(face.h * 0.08)), 0, 0, 360, 0, -1)

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin_color = cv2.inRange(ycrcb, np.array([0, 132, 70], dtype=np.uint8), np.array([255, 180, 145], dtype=np.uint8))
    mask = cv2.bitwise_and(mask, cv2.dilate(skin_color, np.ones((5, 5), np.uint8), iterations=1))
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=5.0)


def _soft_full_image_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (round(width * 0.36), round(height * 0.42)), 0, 0, 360, 120, -1)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) * 0.025)


def _apply_color_preset(rgb: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    intensity = settings.purikura_intensity
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    if settings.preset == "natural":
        lab[:, :, 0] += 8.0 * intensity
        lab[:, :, 1] += 1.4 * intensity
        lab[:, :, 2] -= 1.2 * intensity
        hsv[:, :, 1] *= 1.0 + 0.10 * intensity
    elif settings.preset == "cool":
        lab[:, :, 0] += 13.0 * intensity
        lab[:, :, 1] -= 0.5 * intensity
        lab[:, :, 2] -= 5.8 * intensity
        hsv[:, :, 1] *= 1.0 + 0.05 * intensity
    elif settings.preset == "film":
        lab[:, :, 0] += 5.0 * intensity
        lab[:, :, 1] += 0.8 * intensity
        lab[:, :, 2] += 1.0 * intensity
        hsv[:, :, 1] *= 1.0 - 0.10 * intensity
    elif settings.preset == "neon":
        lab[:, :, 0] += 6.0 * intensity
        lab[:, :, 1] += 2.2 * intensity
        lab[:, :, 2] -= 3.0 * intensity
        hsv[:, :, 1] *= 1.0 + 0.22 * intensity
    else:
        lab[:, :, 0] += 12.0 * intensity
        lab[:, :, 1] += 3.4 * intensity
        lab[:, :, 2] -= 2.4 * intensity
        hsv[:, :, 1] *= 1.0 + 0.16 * intensity

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mixed = cv2.addWeighted(lab_rgb, 0.72, hsv_rgb, 0.28, 0)

    gamma = 1.0 - 0.10 * intensity
    if settings.preset == "film":
        gamma = 1.04
    return _apply_gamma(mixed, gamma)


def _apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 0.05)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(rgb, table)


def _apply_glow_and_grain(rgb: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    glow_strength = settings.glow
    if glow_strength > 0:
        luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bright = cv2.threshold(luminance, 178, 255, cv2.THRESH_TOZERO)[1]
        bright = cv2.GaussianBlur(bright, (0, 0), sigmaX=8 + 16 * glow_strength)
        alpha = (bright.astype(np.float32) / 255.0)[:, :, None] * (0.22 + 0.38 * glow_strength)
        screen = 255 - (255 - rgb.astype(np.float32)) * (255 - np.full_like(rgb, 245, dtype=np.float32)) / 255
        rgb = np.clip(rgb.astype(np.float32) * (1 - alpha) + screen * alpha, 0, 255).astype(np.uint8)

    if settings.preset == "film":
        rng = np.random.default_rng(12)
        noise = rng.normal(0, 4.2, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        rgb = _add_vignette(rgb, amount=0.18)
    return rgb


def _add_vignette(rgb: np.ndarray, amount: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / max(width / 2, 1)
    ny = (yy - height / 2) / max(height / 2, 1)
    distance = np.sqrt(nx * nx + ny * ny)
    factor = 1.0 - np.clip(distance - 0.32, 0, 1) * amount
    return np.clip(rgb.astype(np.float32) * factor[:, :, None], 0, 255).astype(np.uint8)


def _draw_decorations(rgb: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    image = Image.fromarray(rgb).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    border = max(10, round(min(width, height) * 0.025))
    color = (255, 245, 250, 224)
    accent = (255, 114, 171, 188)
    draw.rounded_rectangle(
        (border // 2, border // 2, width - border // 2, height - border // 2),
        radius=border,
        outline=color,
        width=border,
    )
    draw.rounded_rectangle(
        (border + 4, border + 4, width - border - 4, height - border - 4),
        radius=max(4, border // 2),
        outline=accent,
        width=max(2, border // 5),
    )

    size = max(18, round(min(width, height) * 0.045))
    _draw_heart(draw, width - border * 3.5, border * 2.2, size, (255, 91, 151, 210))
    _draw_heart(draw, border * 2.5, height - border * 3.2, round(size * 0.78), (255, 146, 191, 190))
    for i, (x, y) in enumerate(
        (
            (border * 2.3, border * 2.0),
            (width - border * 2.5, height - border * 2.3),
            (width * 0.16, height * 0.22),
            (width * 0.84, height * 0.68),
        )
    ):
        _draw_star(draw, x, y, size * (0.36 + 0.08 * (i % 2)), (255, 235, 128, 180))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.15))
    return np.array(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _draw_heart(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, fill: tuple[int, int, int, int]) -> None:
    points: list[tuple[float, float]] = []
    for step in range(64):
        t = math.tau * step / 64
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        points.append((cx + x * size / 32, cy + y * size / 32))
    draw.polygon(points, fill=fill)


def _draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float, fill: tuple[int, int, int, int]) -> None:
    points = []
    for i in range(8):
        angle = math.tau * i / 8 - math.pi / 2
        r = radius if i % 2 == 0 else radius * 0.38
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    draw.polygon(points, fill=fill)
