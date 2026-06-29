from __future__ import annotations

import base64
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
    image: UploadFile = File(...),
    preset: str = Form("strawberry"),
    pipeline: str = Form("quality"),
    effect_mode: str = Form("normal"),
    purikura_intensity: float = Form(0.78),
    skin_smoothing: float = Form(0.72),
    eye_enlarge: float = Form(0.55),
    face_slim: float = Form(0.42),
    glow: float = Form(0.55),
    decorations: bool = Form(False),
) -> HTMLResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルを選択してください。")

    source = await image.read()
    if not source:
        raise HTTPException(status_code=400, detail="画像ファイルが空です。")

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
    )

    try:
        result = apply_purikura_effect(source, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    original_encoded = base64.b64encode(result.original_bytes).decode("ascii")
    processed_encoded = base64.b64encode(result.image_bytes).decode("ascii")
    segmentation_encoded = base64.b64encode(result.segmentation_bytes).decode("ascii")
    return templates.TemplateResponse(
        request,
        "_result.html",
        {
            "original_image_data": f"data:image/jpeg;base64,{original_encoded}",
            "image_data": f"data:image/jpeg;base64,{processed_encoded}",
            "segmentation_image_data": f"data:image/jpeg;base64,{segmentation_encoded}",
            "metrics": result.metrics,
            "settings": asdict(settings),
        },
    )
