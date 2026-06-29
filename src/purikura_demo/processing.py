from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
LEFT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_BROW = (276, 283, 282, 295, 285, 336, 296, 334, 293, 300)
RIGHT_BROW = (46, 53, 52, 65, 55, 107, 66, 105, 63, 70)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
NOSE = (1, 2, 4, 5, 6, 45, 48, 64, 98, 97, 94, 326, 327, 294, 278, 275, 168, 197)
NOSE_BRIDGE = (6, 168, 197, 195, 5, 4, 1, 2)
LEFT_CHEEK = (50, 101, 118, 117, 123, 147, 187, 205)
RIGHT_CHEEK = (280, 330, 347, 346, 352, 376, 411, 425)
FOREHEAD = (10, 67, 109, 338, 297, 151, 9)
LEFT_IRIS = (474, 475, 476, 477)
RIGHT_IRIS = (469, 470, 471, 472)


@dataclass(frozen=True)
class PurikuraSettings:
    preset: str = "sample_match"
    pipeline: str = "quality"
    effect_mode: str = "ultra"
    purikura_intensity: float = 0.92
    skin_smoothing: float = 0.86
    eye_enlarge: float = 1.0
    face_slim: float = 0.58
    glow: float = 0.42
    decorations: bool = False

    @staticmethod
    def available_presets() -> tuple[tuple[str, str], ...]:
        return (
            ("sample_match", "サンプル寄せ"),
            ("strawberry", "いちごミルク"),
            ("natural", "ナチュラル盛れ"),
            ("cool", "透明感クール"),
            ("film", "フィルムポップ"),
            ("neon", "夜景ネオン"),
        )

    @staticmethod
    def available_effect_modes() -> tuple[tuple[str, str], ...]:
        return (
            ("normal", "Natural"),
            ("strong", "Strong"),
            ("max", "Max"),
            ("ultra", "Ultra"),
        )

    @staticmethod
    def available_pipelines() -> tuple[tuple[str, str], ...]:
        return (
            ("quality", "Quality"),
            ("classic", "Classic"),
        )


@dataclass(frozen=True)
class ProcessedImage:
    image_bytes: bytes
    original_bytes: bytes
    segmentation_bytes: bytes
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FeatheredRegion:
    alpha: np.ndarray
    core: np.ndarray
    transition: np.ndarray


@dataclass(frozen=True)
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    eyes: tuple[tuple[float, float, float], ...]
    landmarks: np.ndarray | None = None
    detector: str = "opencv-haar"

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

    skin_mask = _build_skin_mask(warped, faces)
    if skin_mask.max() == 0:
        skin_mask = _soft_full_image_mask(warped.shape[:2])

    retouched = _apply_skin_retouch(warped, skin_mask, settings)
    if settings.pipeline == "quality":
        retouched = _apply_local_beauty_layers(retouched, faces, skin_mask, settings)
    toned = _apply_color_preset(retouched, skin_mask, settings)
    polished = _apply_glow_and_grain(toned, settings)
    if settings.pipeline == "quality" and faces:
        polished = _polish_retouch_boundaries(polished, warped, skin_mask, faces, settings)
    decorated = _draw_decorations(polished, settings) if settings.decorations else polished

    output = Image.fromarray(decorated).convert("RGB")
    segmentation = Image.fromarray(_build_segmentation_debug(rgb, faces, skin_mask, settings)).convert("RGB")
    return ProcessedImage(
        image_bytes=_encode_jpeg(output, quality=94),
        original_bytes=_encode_jpeg(Image.fromarray(rgb).convert("RGB"), quality=92),
        segmentation_bytes=_encode_jpeg(segmentation, quality=92),
        metrics={
            "width": output.width,
            "height": output.height,
            "faces": len(faces),
            "preset": settings.preset,
            "mode": settings.effect_mode,
            "pipeline": settings.pipeline,
            "accelerator": _accelerator_name(),
            "segmenter": _segmenter_name(faces),
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


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _clamp_settings(settings: PurikuraSettings) -> PurikuraSettings:
    def unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    preset_names = {key for key, _ in PurikuraSettings.available_presets()}
    pipeline_names = {key for key, _ in PurikuraSettings.available_pipelines()}
    mode_names = {key for key, _ in PurikuraSettings.available_effect_modes()}
    preset = settings.preset if settings.preset in preset_names else "strawberry"
    pipeline = settings.pipeline if settings.pipeline in pipeline_names else "quality"
    effect_mode = settings.effect_mode if settings.effect_mode in mode_names else "normal"
    return PurikuraSettings(
        preset=preset,
        pipeline=pipeline,
        effect_mode=effect_mode,
        purikura_intensity=unit(settings.purikura_intensity),
        skin_smoothing=unit(settings.skin_smoothing),
        eye_enlarge=unit(settings.eye_enlarge),
        face_slim=unit(settings.face_slim),
        glow=unit(settings.glow),
        decorations=bool(settings.decorations),
    )


def _mode_multiplier(settings: PurikuraSettings) -> float:
    return {"normal": 1.0, "strong": 1.32, "max": 1.68, "ultra": 2.06}[settings.effect_mode]


def _effective_strength(value: float, settings: PurikuraSettings, cap: float = 1.0) -> float:
    return min(cap, value * _mode_multiplier(settings))


def _detect_faces_and_eyes(rgb: np.ndarray) -> list[FaceRegion]:
    mediapipe_faces = _detect_faces_with_mediapipe(rgb)
    if mediapipe_faces:
        return mediapipe_faces

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    min_side = max(48, min(rgb.shape[:2]) // 12)
    detected = list(face_detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_side, min_side)))
    detected.extend(profile_detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_side, min_side)))

    flipped = cv2.flip(gray, 1)
    image_width = gray.shape[1]
    for px, py, pw, ph in profile_detector.detectMultiScale(
        flipped,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_side, min_side),
    ):
        detected.append((image_width - px - pw, py, pw, ph))

    detected = _dedupe_face_boxes(detected)
    faces: list[FaceRegion] = []
    for x, y, w, h in detected[:8]:
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


def _detect_faces_with_mediapipe(rgb: np.ndarray) -> list[FaceRegion]:
    try:
        import mediapipe as mp  # type: ignore[import-not-found]
    except Exception:
        return []

    height, width = rgb.shape[:2]
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=0.45,
        ) as face_mesh:
            result = face_mesh.process(rgb)
    except Exception:
        return []

    faces: list[FaceRegion] = []
    for face_landmarks in result.multi_face_landmarks or []:
        landmarks = np.array(
            [(point.x * width, point.y * height) for point in face_landmarks.landmark],
            dtype=np.float32,
        )
        if landmarks.shape[0] < 468:
            continue
        x0, y0 = np.floor(np.min(landmarks[:, :2], axis=0)).astype(int)
        x1, y1 = np.ceil(np.max(landmarks[:, :2], axis=0)).astype(int)
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))

        left_eye = _eye_from_landmarks(landmarks, LEFT_EYE, LEFT_IRIS)
        right_eye = _eye_from_landmarks(landmarks, RIGHT_EYE, RIGHT_IRIS)
        faces.append(
            FaceRegion(
                x=x0,
                y=y0,
                w=x1 - x0,
                h=y1 - y0,
                eyes=(left_eye, right_eye),
                landmarks=landmarks,
                detector="mediapipe-face-mesh",
            )
        )
    return sorted(faces, key=lambda face: face.w * face.h, reverse=True)


def _eye_from_landmarks(
    landmarks: np.ndarray,
    eye_indices: tuple[int, ...],
    iris_indices: tuple[int, ...],
) -> tuple[float, float, float]:
    center_indices = iris_indices if max(iris_indices) < landmarks.shape[0] else eye_indices
    center_points = landmarks[list(center_indices)]
    eye_points = landmarks[list(eye_indices)]
    cx, cy = np.mean(center_points, axis=0)
    width = float(np.max(eye_points[:, 0]) - np.min(eye_points[:, 0]))
    height = float(np.max(eye_points[:, 1]) - np.min(eye_points[:, 1]))
    return (float(cx), float(cy), max(width, height, 8.0) * 0.62)


def _dedupe_face_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(_box_iou(box, existing) < 0.28 for existing in kept):
            kept.append(tuple(int(value) for value in box))
    return kept


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


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
    support = np.zeros((height, width), dtype=np.uint8)

    for face in faces:
        if settings.face_slim > 0:
            _add_face_slim_map(map_x, face, _effective_strength(settings.face_slim, settings, cap=1.25))
            cv2.ellipse(
                support,
                (round(face.center[0]), round(face.center[1] + face.h * 0.20)),
                (round(face.w * 0.64), round(face.h * 0.70)),
                0,
                0,
                360,
                255,
                -1,
            )
        if settings.eye_enlarge > 0:
            for cx, cy, radius in face.eyes:
                eye_cap = 2.18 if settings.preset == "sample_match" else 1.35
                eye_radius = radius * (1.76 if settings.preset == "sample_match" else 1.28)
                support_radius = eye_radius * (1.0 + 0.10 * (_mode_multiplier(settings) - 1.0))
                cv2.circle(support, (round(cx), round(cy)), round(support_radius), 255, -1)
                _add_eye_enlarge_map(
                    map_x,
                    map_y,
                    (cx, cy),
                    support_radius,
                    _effective_strength(settings.eye_enlarge, settings, cap=eye_cap),
                )

    warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    if support.max() == 0:
        return warped
    return _repair_warp_boundary(rgb, warped, support)


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
    falloff = np.power(np.clip(1.0 - distance, 0.0, 1.0), beta)
    scale = 1.0 + alpha * falloff

    region_x = map_x[y0:y1, x0:x1]
    region_y = map_y[y0:y1, x0:x1]
    target_x = cx + dx / scale
    target_y = cy + dy / scale
    region_x[mask] += (target_x - xx.astype(np.float32))[mask]
    region_y[mask] += (target_y - yy.astype(np.float32))[mask]


def _add_face_slim_map(map_x: np.ndarray, face: FaceRegion, strength: float) -> None:
    cx, cy = face.center
    rx = face.w * 0.60
    ry = face.h * 0.68
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
    boundary_weight = _smoothstep(0.0, 0.72, np.clip(1.0 - ellipse, 0.0, 1.0))
    mask = ellipse < 1.0
    shrink = 0.035 + 0.16 * strength
    scale = 1.0 - shrink * lower_weight * side_weight * boundary_weight
    scale = np.clip(scale, 0.76, 1.0)

    region_x = map_x[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - cx
    target_x = cx + dx / scale
    region_x[mask] += (target_x - xx.astype(np.float32))[mask]


def _repair_warp_boundary(original: np.ndarray, warped: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    region = build_feathered_region(support_mask, inner_px=14, outer_px=46)
    if region.transition.max() <= 0:
        return warped
    matched = match_boundary_tone(original, warped, region.transition)
    repaired = _blend_with_float_mask(warped.astype(np.float32), matched, region.transition, 0.42)
    return np.clip(repaired, 0, 255).astype(np.uint8)


def _apply_skin_retouch(rgb: np.ndarray, mask: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    if settings.skin_smoothing <= 0:
        return rgb

    strength = _effective_strength(settings.skin_smoothing, settings, cap=1.35)
    safe_mask = suppress_mask_on_edges(mask, rgb)
    if settings.pipeline == "quality":
        retouch = _frequency_separated_skin(rgb, strength)
    else:
        smoothed = cv2.bilateralFilter(rgb, d=9, sigmaColor=38 + strength * 38, sigmaSpace=7)
        smoothed = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=0.7 + strength * 1.8)
        broader = cv2.GaussianBlur(rgb, (0, 0), sigmaX=2.5 + strength * 2.0)
        detail = cv2.addWeighted(rgb, 1.15, broader, -0.15, 0)
        retouch = cv2.addWeighted(smoothed, 0.88, detail, 0.12, 0)

    region = build_feathered_region(safe_mask, inner_px=18, outer_px=64)
    base_opacity = min(0.92, 0.26 + 0.52 * strength)
    boundary_matched = match_boundary_tone(rgb, retouch, region.transition)
    retouch = _blend_with_float_mask(retouch.astype(np.float32), boundary_matched, region.transition, 1.0)
    alpha = np.clip(region.core * base_opacity + region.transition * base_opacity * 0.38, 0.0, 1.0)
    return np.clip(rgb.astype(np.float32) * (1.0 - alpha[:, :, None]) + retouch * alpha[:, :, None], 0, 255).astype(np.uint8)


def _frequency_separated_skin(rgb: np.ndarray, strength: float) -> np.ndarray:
    sigma = 2.4 + 2.2 * strength
    low = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    try:
        edge_smooth = cv2.edgePreservingFilter(rgb, flags=1, sigma_s=38 + 18 * strength, sigma_r=0.20 + 0.08 * strength)
    except cv2.error:
        edge_smooth = cv2.bilateralFilter(rgb, d=11, sigmaColor=48 + strength * 44, sigmaSpace=9)
    high = rgb.astype(np.float32) - low.astype(np.float32)
    detail_keep = 0.62 - 0.18 * min(strength, 1.0)
    retouch = edge_smooth.astype(np.float32) + high * detail_keep
    return np.clip(retouch, 0, 255).astype(np.uint8)


def _build_skin_mask(rgb: np.ndarray, faces: list[FaceRegion]) -> np.ndarray:
    height, width = rgb.shape[:2]
    region_mask = np.zeros((height, width), dtype=np.uint8)
    protect_mask = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        if face.landmarks is not None:
            _fill_landmark_polygon(region_mask, face, FACE_OVAL, 255)
        else:
            center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.52))
            axes = (round(face.w * 0.54), round(face.h * 0.62))
            cv2.ellipse(region_mask, center, axes, 0, 0, 360, 255, -1)

        neck_center = (round(face.x + face.w * 0.5), round(face.y + face.h * 1.04))
        neck_axes = (round(face.w * 0.36), round(face.h * 0.30))
        cv2.ellipse(region_mask, neck_center, neck_axes, 0, 0, 360, 150, -1)

        if face.landmarks is not None:
            _fill_landmark_polygon(protect_mask, face, LEFT_EYE, 255)
            _fill_landmark_polygon(protect_mask, face, RIGHT_EYE, 255)
            _fill_landmark_polygon(protect_mask, face, LEFT_BROW, 220)
            _fill_landmark_polygon(protect_mask, face, RIGHT_BROW, 220)
            _fill_landmark_polygon(protect_mask, face, LIPS, 255)
        else:
            for cx, cy, radius in face.eyes:
                cv2.circle(protect_mask, (round(cx), round(cy)), round(radius * 1.04), 255, -1)
            mouth_center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.73))
            cv2.ellipse(protect_mask, mouth_center, (round(face.w * 0.20), round(face.h * 0.09)), 0, 0, 360, 255, -1)

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin_color = cv2.inRange(ycrcb, np.array([0, 132, 70], dtype=np.uint8), np.array([255, 180, 145], dtype=np.uint8))
    skin_color = cv2.dilate(skin_color, np.ones((9, 9), np.uint8), iterations=1)
    skin_color = cv2.GaussianBlur(skin_color, (0, 0), sigmaX=8.0)
    region_mask = cv2.GaussianBlur(region_mask, (0, 0), sigmaX=max(10.0, min(width, height) * 0.022))

    region = region_mask.astype(np.float32) / 255.0
    skin = skin_color.astype(np.float32) / 255.0
    mask = region * (0.42 + 0.58 * skin)

    protect = cv2.GaussianBlur(protect_mask, (0, 0), sigmaX=4.0).astype(np.float32) / 255.0
    mask *= 1.0 - protect * 0.82

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dark_detail = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 190, 126], dtype=np.uint8))
    dark_detail = cv2.GaussianBlur(dark_detail, (0, 0), sigmaX=5.0).astype(np.float32) / 255.0
    mask *= 1.0 - dark_detail * 0.30
    return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


def _build_segmentation_debug(
    rgb: np.ndarray,
    faces: list[FaceRegion],
    skin_mask: np.ndarray,
    settings: PurikuraSettings,
) -> np.ndarray:
    base = rgb.astype(np.float32)
    mask = skin_mask.astype(np.float32) / 255.0
    overlay_color = np.zeros_like(base)
    overlay_color[:, :, 0] = 255
    overlay_color[:, :, 1] = 91
    overlay_color[:, :, 2] = 151
    alpha = (0.50 * mask)[:, :, None]
    debug = np.clip(base * (1.0 - alpha) + overlay_color * alpha, 0, 255).astype(np.uint8)
    if settings.pipeline == "quality":
        parts = _build_part_masks(rgb.shape[:2], faces)
        skin_region = build_feathered_region(suppress_mask_on_edges(skin_mask, rgb), inner_px=18, outer_px=64)
        eyes_region = build_feathered_region(parts.eyes, inner_px=5, outer_px=22)
        brows_region = build_feathered_region(parts.brows, inner_px=4, outer_px=18)
        lips_region = build_feathered_region(parts.lips, inner_px=5, outer_px=24)
        protected = np.maximum.reduce((eyes_region.alpha, brows_region.alpha * 0.85, lips_region.alpha))
        debug = _debug_overlay_float_mask(debug, skin_region.transition, (255, 184, 84), 0.42)
        debug = _debug_overlay_float_mask(debug, skin_region.core, (255, 91, 151), 0.36)
        debug = _debug_overlay_mask(debug, parts.cheeks, (255, 150, 188), 0.34)
        debug = _debug_overlay_mask(debug, parts.lips, (230, 72, 118), 0.45)
        debug = _debug_overlay_mask(debug, parts.eyes, (244, 198, 79), 0.45)
        debug = _debug_overlay_float_mask(debug, protected, (87, 95, 112), 0.28)
        debug = _debug_overlay_mask(debug, parts.nose, (150, 210, 255), 0.36)
        debug = _debug_overlay_mask(debug, parts.hair, (80, 110, 130), 0.35)
        cv2.putText(
            debug,
            "skin core / transition / protected parts",
            (24, min(debug.shape[0] - 24, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    for index, face in enumerate(faces, start=1):
        cv2.rectangle(debug, (face.x, face.y), (face.x + face.w, face.y + face.h), (21, 168, 143), 3)
        cv2.putText(
            debug,
            f"face {index}",
            (face.x, max(20, face.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (21, 168, 143),
            2,
            cv2.LINE_AA,
        )
        for cx, cy, radius in face.eyes:
            cv2.circle(debug, (round(cx), round(cy)), round(radius), (244, 198, 79), 2)

    if not faces:
        cv2.putText(
            debug,
            "no face detected: fallback soft mask",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (241, 95, 154),
            2,
            cv2.LINE_AA,
        )
    return debug


@dataclass(frozen=True)
class PartMasks:
    face_skin: np.ndarray
    eyes: np.ndarray
    brows: np.ndarray
    nose: np.ndarray
    cheeks: np.ndarray
    lips: np.ndarray
    hair: np.ndarray
    highlights: np.ndarray


def _build_part_masks(shape: tuple[int, int], faces: list[FaceRegion]) -> PartMasks:
    height, width = shape
    face_skin = np.zeros((height, width), dtype=np.uint8)
    eyes = np.zeros((height, width), dtype=np.uint8)
    brows = np.zeros((height, width), dtype=np.uint8)
    nose = np.zeros((height, width), dtype=np.uint8)
    cheeks = np.zeros((height, width), dtype=np.uint8)
    lips = np.zeros((height, width), dtype=np.uint8)
    hair = np.zeros((height, width), dtype=np.uint8)
    highlights = np.zeros((height, width), dtype=np.uint8)

    for face in faces:
        if face.landmarks is not None:
            _fill_landmark_polygon(face_skin, face, FACE_OVAL, 230)
            _fill_landmark_polygon(eyes, face, LEFT_EYE, 255)
            _fill_landmark_polygon(eyes, face, RIGHT_EYE, 255)
            _fill_landmark_polygon(brows, face, LEFT_BROW, 220)
            _fill_landmark_polygon(brows, face, RIGHT_BROW, 220)
            _fill_landmark_polygon(nose, face, NOSE, 200)
            _fill_landmark_polygon(lips, face, LIPS, 230)
            _fill_landmark_polygon(cheeks, face, LEFT_CHEEK, 170)
            _fill_landmark_polygon(cheeks, face, RIGHT_CHEEK, 170)
            _fill_landmark_polyline(highlights, face, NOSE_BRIDGE, 160, width=max(2, round(face.w * 0.025)))
            hair_top = max(0, round(face.y - face.h * 0.16))
            hair_center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.13))
            hair_axes = (round(face.w * 0.55), round(face.h * 0.30))
            cv2.ellipse(hair, hair_center, hair_axes, 0, 180, 360, 160, -1)
            cv2.ellipse(
                hair,
                (round(face.x + face.w * 0.5), round(face.y + face.h * 0.44)),
                (round(face.w * 0.68), round(face.h * 0.72)),
                0,
                0,
                360,
                90,
                -1,
            )
            cv2.rectangle(hair, (face.x, hair_top), (face.x + face.w, round(face.y + face.h * 0.30)), 90, -1)
            side_y = round(face.y + face.h * 0.78)
            side_axes = (round(face.w * 0.24), round(face.h * 0.76))
            cv2.ellipse(hair, (round(face.x + face.w * 0.10), side_y), side_axes, -6, 0, 360, 78, -1)
            cv2.ellipse(hair, (round(face.x + face.w * 0.90), side_y), side_axes, 6, 0, 360, 78, -1)
        else:
            for cx, cy, radius in face.eyes:
                cv2.circle(eyes, (round(cx), round(cy)), round(radius * 1.18), 255, -1)

            center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.52))
            cv2.ellipse(face_skin, center, (round(face.w * 0.48), round(face.h * 0.55)), 0, 0, 360, 210, -1)
            cheek_y = round(face.y + face.h * 0.60)
            cheek_axes = (round(face.w * 0.15), round(face.h * 0.075))
            cv2.ellipse(cheeks, (round(face.x + face.w * 0.33), cheek_y), cheek_axes, -8, 0, 360, 190, -1)
            cv2.ellipse(cheeks, (round(face.x + face.w * 0.67), cheek_y), cheek_axes, 8, 0, 360, 190, -1)

            mouth_center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.73))
            cv2.ellipse(lips, mouth_center, (round(face.w * 0.18), round(face.h * 0.065)), 0, 0, 360, 210, -1)

            hair_center = (round(face.x + face.w * 0.5), round(face.y + face.h * 0.16))
            hair_axes = (round(face.w * 0.50), round(face.h * 0.28))
            cv2.ellipse(hair, hair_center, hair_axes, 0, 180, 360, 160, -1)
            cv2.ellipse(
                hair,
                (round(face.x + face.w * 0.5), round(face.y + face.h * 0.45)),
                (round(face.w * 0.66), round(face.h * 0.72)),
                0,
                0,
                360,
                92,
                -1,
            )
            side_y = round(face.y + face.h * 0.82)
            side_axes = (round(face.w * 0.25), round(face.h * 0.82))
            cv2.ellipse(hair, (round(face.x + face.w * 0.08), side_y), side_axes, -8, 0, 360, 82, -1)
            cv2.ellipse(hair, (round(face.x + face.w * 0.92), side_y), side_axes, 8, 0, 360, 82, -1)

    return PartMasks(
        face_skin=cv2.GaussianBlur(face_skin, (0, 0), sigmaX=5.0),
        eyes=cv2.GaussianBlur(eyes, (0, 0), sigmaX=3.0),
        brows=cv2.GaussianBlur(brows, (0, 0), sigmaX=2.0),
        nose=cv2.GaussianBlur(nose, (0, 0), sigmaX=3.0),
        cheeks=cv2.GaussianBlur(cheeks, (0, 0), sigmaX=10.0),
        lips=cv2.GaussianBlur(lips, (0, 0), sigmaX=2.8),
        hair=cv2.GaussianBlur(hair, (0, 0), sigmaX=8.0),
        highlights=cv2.GaussianBlur(highlights, (0, 0), sigmaX=2.0),
    )


def _fill_landmark_polygon(mask: np.ndarray, face: FaceRegion, indices: tuple[int, ...], value: int) -> None:
    if face.landmarks is None or max(indices) >= face.landmarks.shape[0]:
        return
    points = np.round(face.landmarks[list(indices)]).astype(np.int32)
    cv2.fillPoly(mask, [points], value)


def _fill_landmark_polyline(mask: np.ndarray, face: FaceRegion, indices: tuple[int, ...], value: int, width: int) -> None:
    if face.landmarks is None or max(indices) >= face.landmarks.shape[0]:
        return
    points = np.round(face.landmarks[list(indices)]).astype(np.int32)
    cv2.polylines(mask, [points], isClosed=False, color=value, thickness=width, lineType=cv2.LINE_AA)


def _apply_local_beauty_layers(
    rgb: np.ndarray,
    faces: list[FaceRegion],
    skin_mask: np.ndarray,
    settings: PurikuraSettings,
) -> np.ndarray:
    if not faces:
        return rgb

    strength = _mode_multiplier(settings)
    parts = _build_part_masks(rgb.shape[:2], faces)
    out = rgb.astype(np.float32)

    eyes_region = build_feathered_region(parts.eyes, inner_px=5, outer_px=22)
    brows_region = build_feathered_region(parts.brows, inner_px=4, outer_px=18)
    lips_region = build_feathered_region(parts.lips, inner_px=5, outer_px=24)
    cheeks_region = build_feathered_region(parts.cheeks, inner_px=12, outer_px=42)
    protect = np.maximum.reduce((eyes_region.alpha, brows_region.alpha * 0.85, lips_region.alpha))

    eye_detail = _unsharp(rgb, sigma=1.0, amount=0.75 + 0.18 * strength)
    if settings.preset == "sample_match":
        eye_detail = _sample_match_eye_detail(eye_detail)
    eye_detail = match_boundary_tone(rgb, eye_detail, eyes_region.transition)
    out = _blend_with_float_mask(out, eye_detail, eyes_region.alpha, 0.70 if settings.preset == "sample_match" else 0.62)

    cheek_color = np.full_like(rgb, (255, 120, 170), dtype=np.uint8)
    lip_color = np.full_like(rgb, (218, 72, 118), dtype=np.uint8)
    cheek_alpha = cheeks_region.alpha * (1.0 - protect * 0.80)
    out = _soft_light_with_float_mask(out, cheek_color, cheek_alpha, 0.20 + 0.08 * strength)
    lip_layer = match_boundary_tone(np.clip(out, 0, 255).astype(np.uint8), lip_color, lips_region.transition)
    out = _soft_light_with_float_mask(out, lip_layer, lips_region.alpha, 0.34 + 0.10 * strength)

    hair_mask = _refine_hair_mask(rgb, parts.hair, skin_mask, settings)
    hair_region = build_feathered_region(hair_mask, inner_px=10, outer_px=44)
    if hair_mask.max() > 0:
        hair_smooth = cv2.bilateralFilter(rgb, d=7, sigmaColor=32, sigmaSpace=7)
        hair_gloss = cv2.addWeighted(hair_smooth, 1.08, cv2.GaussianBlur(hair_smooth, (0, 0), 4.0), -0.08, 0)
        hair_gloss = match_boundary_tone(rgb, hair_gloss, hair_region.transition)
        out = _blend_with_float_mask(out, hair_gloss, hair_region.alpha, 0.32)
        if settings.preset == "sample_match":
            brown_hair = _sample_match_hair_tone(np.clip(out, 0, 255).astype(np.uint8))
            brown_hair = match_boundary_tone(rgb, brown_hair, hair_region.transition)
            out = _blend_with_float_mask(out, brown_hair, hair_region.alpha, 0.68)

    skin_tone = _skin_tone_lift(np.clip(out, 0, 255).astype(np.uint8), settings)
    skin_region = build_feathered_region(suppress_mask_on_edges(skin_mask, rgb), inner_px=18, outer_px=64)
    skin_alpha = skin_region.alpha * (1.0 - protect * 0.90) * (1.0 - hair_region.alpha * 0.78)
    skin_tone = match_boundary_tone(np.clip(out, 0, 255).astype(np.uint8), skin_tone, skin_region.transition)
    out = _blend_with_float_mask(out, skin_tone, skin_alpha, 0.28 if settings.preset == "sample_match" else 0.42)
    return np.clip(out, 0, 255).astype(np.uint8)


def _refine_hair_mask(
    rgb: np.ndarray,
    rough_hair: np.ndarray,
    skin_mask: np.ndarray,
    settings: PurikuraSettings | None = None,
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sample_match = settings is not None and settings.preset == "sample_match"
    max_saturation = 172 if sample_match else 120
    max_value = 150 if sample_match else 105
    dark = cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([180, max_saturation, max_value], dtype=np.uint8),
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    mask = (rough_hair.astype(np.float32) / 255.0) * (dark.astype(np.float32) / 255.0)
    skin_suppression = 0.28 if sample_match else 0.55
    mask *= 1.0 - (skin_mask.astype(np.float32) / 255.0) * skin_suppression
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=7.0 if sample_match else 5.0)
    return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


def _skin_tone_lift(rgb: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    strength = _effective_strength(settings.skin_smoothing, settings, cap=1.25)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if settings.preset == "sample_match":
        lab[:, :, 0] += 1.2 + 1.9 * strength
        lab[:, :, 1] += 1.0 + 0.6 * strength
        lab[:, :, 2] -= 2.2 + 2.4 * strength
    else:
        lab[:, :, 0] += 4.0 + 5.0 * strength
        lab[:, :, 1] += 0.7 + 1.0 * strength
        lab[:, :, 2] -= 0.6 + 1.5 * strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _sample_match_hair_tone(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] += 28.0
    lab[:, :, 1] += 5.6
    lab[:, :, 2] += 8.0
    toned = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    brown = np.full_like(rgb, (170, 127, 104), dtype=np.uint8)
    return _soft_light_with_mask(toned.astype(np.float32), brown, np.full(rgb.shape[:2], 220, dtype=np.uint8), 0.62).astype(np.uint8)


def _sample_match_eye_detail(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] -= 7.0
    lab[:, :, 1] += 1.6
    lab[:, :, 2] -= 0.8
    contrasted = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    crisp = cv2.addWeighted(contrasted, 1.16, cv2.GaussianBlur(contrasted, (0, 0), 0.9), -0.16, 0)
    return np.clip(crisp, 0, 255).astype(np.uint8)


def _unsharp(rgb: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(rgb, 1.0 + amount, blur, -amount, 0)


def build_feathered_region(mask: np.ndarray, inner_px: int, outer_px: int) -> FeatheredRegion:
    mask_float = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
    binary = (mask_float > 0.025).astype(np.uint8)
    if binary.max() == 0:
        empty = np.zeros_like(mask_float, dtype=np.float32)
        return FeatheredRegion(alpha=empty, core=empty, transition=empty)

    inner_distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    signed_distance = inner_distance - outside_distance
    outer = max(float(outer_px), 1.0)
    inner = max(float(inner_px), 1.0)
    alpha = _smoothstep(-outer, inner, signed_distance)
    alpha *= cv2.GaussianBlur(mask_float, (0, 0), sigmaX=max(1.0, outer * 0.18))
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    core = np.clip(_smoothstep(inner * 0.55, inner * 1.25, inner_distance) * mask_float, 0.0, 1.0).astype(np.float32)
    transition = np.clip(alpha - core, 0.0, 1.0).astype(np.float32)
    return FeatheredRegion(alpha=alpha, core=core, transition=transition)


def suppress_mask_on_edges(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    if float(edge.max()) > 0.0:
        edge /= float(edge.max())
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=2.8)
    mask_float = mask.astype(np.float32) / 255.0
    suppressed = mask_float * (1.0 - np.clip(edge, 0.0, 1.0) * 0.34)
    return np.clip(suppressed * 255.0, 0, 255).astype(np.uint8)


def _edge_limited_boundary_mask(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    if float(edge.max()) > 0.0:
        edge /= float(edge.max())
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=3.5)
    expanded = cv2.GaussianBlur(np.clip(mask.astype(np.float32), 0.0, 1.0), (0, 0), sigmaX=7.0)
    return np.clip(expanded * (0.68 + edge * 0.54), 0.0, 1.0).astype(np.float32)


def match_boundary_tone(base: np.ndarray, layer: np.ndarray, transition_mask: np.ndarray) -> np.ndarray:
    transition = np.clip(transition_mask.astype(np.float32), 0.0, 1.0)
    if transition.max() <= 0.0:
        return layer.astype(np.float32)

    base_lab = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    layer_lab = cv2.cvtColor(np.clip(layer, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    weight = transition[:, :, None]
    total = float(np.sum(weight))
    if total <= 1e-6:
        return layer.astype(np.float32)

    base_mean = np.sum(base_lab * weight, axis=(0, 1)) / total
    layer_mean = np.sum(layer_lab * weight, axis=(0, 1)) / total
    shift = (base_mean - layer_mean) * np.array([0.45, 0.62, 0.62], dtype=np.float32)
    adjusted_lab = layer_lab + shift[None, None, :] * weight
    return cv2.cvtColor(np.clip(adjusted_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)


def _blend_with_float_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[:, :, None] * opacity
    return base.astype(np.float32) * (1.0 - alpha) + layer.astype(np.float32) * alpha


def _blend_with_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * opacity
    return base * (1.0 - alpha) + layer.astype(np.float32) * alpha


def _screen_with_float_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    screen = 255.0 - (255.0 - base.astype(np.float32)) * (255.0 - layer.astype(np.float32)) / 255.0
    return _blend_with_float_mask(base, screen, mask, opacity)


def _screen_with_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    screen = 255.0 - (255.0 - base) * (255.0 - layer.astype(np.float32)) / 255.0
    return _blend_with_mask(base, screen.astype(np.uint8), mask, opacity)


def _soft_light_with_float_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    cb = np.clip(base.astype(np.float32) / 255.0, 0.0, 1.0)
    cs = layer.astype(np.float32) / 255.0
    soft = np.where(cs <= 0.5, cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb), cb + (2.0 * cs - 1.0) * (_soft_light_d(cb) - cb))
    return _blend_with_float_mask(base, np.clip(soft * 255.0, 0, 255), mask, opacity)


def _soft_light_with_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    cb = np.clip(base / 255.0, 0.0, 1.0)
    cs = layer.astype(np.float32) / 255.0
    soft = np.where(cs <= 0.5, cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb), cb + (2.0 * cs - 1.0) * (_soft_light_d(cb) - cb))
    return _blend_with_mask(base, np.clip(soft * 255.0, 0, 255).astype(np.uint8), mask, opacity)


def _soft_light_d(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.25, ((16 * value - 12) * value + 4) * value, np.sqrt(value))


def _debug_overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], opacity: float) -> np.ndarray:
    overlay = np.full_like(base, color, dtype=np.uint8)
    return _blend_with_mask(base.astype(np.float32), overlay, mask, opacity).astype(np.uint8)


def _debug_overlay_float_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], opacity: float) -> np.ndarray:
    overlay = np.full_like(base, color, dtype=np.uint8)
    return _blend_with_float_mask(base.astype(np.float32), overlay, mask, opacity).astype(np.uint8)


def _soft_full_image_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (round(width * 0.36), round(height * 0.42)), 0, 0, 360, 120, -1)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) * 0.025)


def _apply_color_preset(rgb: np.ndarray, skin_mask: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    intensity = _effective_strength(settings.purikura_intensity, settings, cap=1.35)
    if settings.preset == "sample_match":
        return _apply_sample_match_color(rgb, skin_mask, settings, intensity)

    mps_result = _apply_mps_color_preset(rgb, skin_mask, settings, intensity)
    if mps_result is not None:
        return mps_result

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


def _apply_sample_match_color(
    rgb: np.ndarray,
    skin_mask: np.ndarray,
    settings: PurikuraSettings,
    intensity: float,
) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = 128.0 + (lab[:, :, 0] - 128.0) * 0.88 + 11.2 * intensity
    lab[:, :, 1] = 128.0 + (lab[:, :, 1] - 128.0) * 0.88 + 2.2 * intensity
    lab[:, :, 2] = 128.0 + (lab[:, :, 2] - 128.0) * 0.72 - 0.8 * intensity
    matched = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    hsv = cv2.cvtColor(matched, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.82
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.025 + 2.0, 0, 255)
    matched = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    skin_region = build_feathered_region(suppress_mask_on_edges(skin_mask, rgb), inner_px=20, outer_px=74)
    skin = np.clip(skin_region.core + skin_region.transition * 0.34, 0.0, 1.0)
    skin_lab = cv2.cvtColor(matched, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_lab[:, :, 0] += skin * (2.8 + 1.2 * intensity)
    skin_lab[:, :, 1] += skin * 1.6
    skin_lab[:, :, 2] -= skin * 0.6
    matched = cv2.cvtColor(np.clip(skin_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    return _apply_gamma(matched, 0.985)


def _polish_retouch_boundaries(
    rgb: np.ndarray,
    guide: np.ndarray,
    skin_mask: np.ndarray,
    faces: list[FaceRegion],
    settings: PurikuraSettings,
) -> np.ndarray:
    parts = _build_part_masks(rgb.shape[:2], faces)
    skin_region = build_feathered_region(suppress_mask_on_edges(skin_mask, guide), inner_px=12, outer_px=92)
    face_region = build_feathered_region(parts.face_skin, inner_px=10, outer_px=74)
    eyes_region = build_feathered_region(parts.eyes, inner_px=6, outer_px=34)
    lips_region = build_feathered_region(parts.lips, inner_px=5, outer_px=28)
    hair_region = build_feathered_region(_refine_hair_mask(guide, parts.hair, skin_mask, settings), inner_px=8, outer_px=58)
    boundary = np.maximum.reduce(
        (
            skin_region.transition,
            face_region.transition * 0.85,
            eyes_region.transition * 0.70,
            lips_region.transition * 0.52,
            hair_region.transition * 0.50,
        )
    )
    if boundary.max() <= 0.0:
        return rgb

    edge_limited = _edge_limited_boundary_mask(boundary, rgb)
    tone_matched = match_boundary_tone(guide, rgb, edge_limited)
    smoothed = cv2.bilateralFilter(np.clip(rgb, 0, 255).astype(np.uint8), d=9, sigmaColor=28, sigmaSpace=7)
    smoothed = cv2.addWeighted(smoothed, 0.70, cv2.GaussianBlur(smoothed, (0, 0), 1.2), 0.30, 0)
    repaired = _blend_with_float_mask(rgb.astype(np.float32), tone_matched, edge_limited, 0.58)
    repaired = _blend_with_float_mask(repaired, smoothed, edge_limited, 0.34)
    return np.clip(repaired, 0, 255).astype(np.uint8)


def _accelerator_name() -> str:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return "opencv-cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "torch-mps"
    return "opencv-cpu"


def _segmenter_name(faces: list[FaceRegion]) -> str:
    if any(face.detector == "mediapipe-face-mesh" for face in faces):
        return "mediapipe-face-mesh"
    if faces:
        return "opencv-haar-fallback"
    return "fallback-soft-mask"


def _apply_mps_color_preset(
    rgb: np.ndarray,
    skin_mask: np.ndarray,
    settings: PurikuraSettings,
    intensity: float,
) -> np.ndarray | None:
    if settings.pipeline != "quality":
        return None
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
        return None

    device = torch.device("mps")
    image = torch.as_tensor(rgb, device=device, dtype=torch.float32) / 255.0
    skin = torch.as_tensor(skin_mask, device=device, dtype=torch.float32).unsqueeze(-1) / 255.0
    image = image.pow(max(0.78, 1.0 - 0.08 * intensity))
    saturation = 1.0 + (0.08 if settings.preset != "film" else -0.04) * intensity
    luma = (image[..., 0:1] * 0.2126 + image[..., 1:2] * 0.7152 + image[..., 2:3] * 0.0722)
    image = luma + (image - luma) * saturation
    skin_lift = torch.tensor([1.02, 0.985, 1.01], device=device, dtype=torch.float32)
    image = image * (1.0 - skin * 0.20 * intensity) + torch.clamp(image * skin_lift + 0.025 * intensity, 0.0, 1.0) * skin * 0.20 * intensity
    return torch.clamp(image * 255.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()


def _apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 0.05)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(rgb, table)


def _apply_glow_and_grain(rgb: np.ndarray, settings: PurikuraSettings) -> np.ndarray:
    glow_strength = _effective_strength(settings.glow, settings, cap=1.3)
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


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
