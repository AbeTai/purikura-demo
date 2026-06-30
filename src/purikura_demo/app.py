from __future__ import annotations

import base64
import binascii
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from purikura_demo.processing import PurikuraSettings, apply_purikura_effect

app = FastAPI(title="Purikura Demo")

PACKAGE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "settings": PurikuraSettings(),
            "presets": PurikuraSettings.available_presets(),
            "pipelines": PurikuraSettings.available_pipelines(),
            "effect_modes": PurikuraSettings.available_effect_modes(),
        },
    )


@app.post("/process", response_class=HTMLResponse)
async def process_image(
    request: Request,
    image: UploadFile | None = File(None),
    camera_image: str = Form(""),
    preset: str = Form(PurikuraSettings.preset),
    pipeline: str = Form("quality"),
    effect_mode: str = Form(PurikuraSettings.effect_mode),
    purikura_intensity: float = Form(PurikuraSettings.purikura_intensity),
    skin_smoothing: float = Form(PurikuraSettings.skin_smoothing),
    eye_enlarge: float = Form(PurikuraSettings.eye_enlarge),
    face_slim: float = Form(PurikuraSettings.face_slim),
    glow: float = Form(PurikuraSettings.glow),
    decorations: bool = Form(False),
    white_background: bool = Form(PurikuraSettings.white_background),
) -> HTMLResponse:
    source = await _read_source_image(image, camera_image)

    settings = PurikuraSettings(
        preset=preset,
        pipeline=pipeline,
        effect_mode=effect_mode,
        purikura_intensity=purikura_intensity,
        skin_smoothing=skin_smoothing,
        eye_enlarge=eye_enlarge,
        face_slim=face_slim,
        glow=glow,
        decorations=decorations,
        white_background=white_background,
    )

    try:
        result = apply_purikura_effect(source, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    original_encoded = base64.b64encode(result.original_bytes).decode("ascii")
    processed_encoded = base64.b64encode(result.image_bytes).decode("ascii")
    segmentation_encoded = base64.b64encode(result.segmentation_bytes).decode("ascii")
    background_encoded = base64.b64encode(result.background_bytes).decode("ascii")
    return templates.TemplateResponse(
        request,
        "_result.html",
        {
            "original_image_data": f"data:image/jpeg;base64,{original_encoded}",
            "image_data": f"data:image/jpeg;base64,{processed_encoded}",
            "segmentation_image_data": f"data:image/jpeg;base64,{segmentation_encoded}",
            "background_image_data": f"data:image/jpeg;base64,{background_encoded}",
            "metrics": result.metrics,
            "settings": asdict(settings),
        },
    )


async def _read_source_image(image: UploadFile | None, camera_image: str) -> bytes:
    if image is not None and image.filename:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="画像ファイルを選択してください。")

        source = await image.read()
        if not source:
            raise HTTPException(status_code=400, detail="画像ファイルが空です。")
        return source

    camera_image = camera_image.strip()
    if camera_image:
        return _decode_camera_data_url(camera_image)

    raise HTTPException(status_code=400, detail="画像ファイルを選択するか、カメラで撮影してください。")


def _decode_camera_data_url(data_url: str) -> bytes:
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="撮影画像の形式が不正です。") from exc

    if not header.startswith("data:image/") or ";base64" not in header:
        raise HTTPException(status_code=400, detail="撮影画像の形式が不正です。")

    try:
        source = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="撮影画像の読み込みに失敗しました。") from exc

    if not source:
        raise HTTPException(status_code=400, detail="撮影画像が空です。")
    return source
