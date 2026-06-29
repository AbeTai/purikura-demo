# Purikura Demo

静止画をアップロードして、プリクラ風の明るい肌、目元強調、輪郭補正、グロー、フレーム装飾を合成する FastAPI + htmx アプリです。

`docs/purikura-research-result.md` の推奨パイプラインを、モデルダウンロード不要の古典画像処理で動く最小実装に落とし込んでいます。

## Features

- FastAPI による画像アップロード API
- htmx による結果部分の非同期差し替え
- OpenCV Haar cascade による複数人の顔・目の簡易検出
- 境界フェザーつき逆写像ワープによる目拡大
- 境界で自然に戻る楕円 ROI の簡易小顔ワープ
- 顔・首周辺のソフトマスクつき bilateral / Gaussian blend
- Lab / HSV ベースのプリセット色調補正
- Screen blend のソフトグロー、粒状ノイズ、フレーム、スタンプ風装飾
- `Natural / Strong / Max` の加工強度モード
- 元画像と加工後画像の比較表示
- 顔 bbox、目領域、肌マスクを重ねたセグメンテーションデバッグ表示

## Setup

```bash
uv sync --all-extras
```

## Run

```bash
uv run uvicorn purikura_demo.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Test

```bash
uv run pytest
```
