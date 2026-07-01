# Purikura Demo

静止画アップロードまたはカメラ撮影から、プリクラ風の明るい肌、目元強調、輪郭補正、グロー、背景白塗り、フレーム装飾を合成する FastAPI + htmx アプリです。

`docs/purikura-research-result.md` の推奨パイプラインを元に、MediaPipe Face Mesh / FaceLandmarker と rembg birefnet-portrait を必須経路にした品質優先の実装です。

## Features

- FastAPI による画像アップロード / カメラ撮影 API
- htmx による結果部分の非同期差し替え
- カメラ撮影ではブラウザ側 FaceLandmarker で検出済みフレームだけを顔中心 `960 x 1200` に切り出して送信
- MediaPipe Face Mesh / FaceLandmarker による複数人の顔・目・眉・鼻・唇・頬の細かいパーツ推定
- rembg `birefnet-portrait` による人物背景分離
- Face Mesh / birefnet が使えない場合は低品質処理へ落とさず、明示的にエラーを返す
- 境界フェザーつき逆写像ワープによる目拡大
- 境界で自然に戻る楕円 ROI の簡易小顔ワープ
- 顔・首周辺のソフトマスクつき bilateral / Gaussian blend
- Lab / HSV ベースのプリセット色調補正
- Screen blend のソフトグロー、粒状ノイズ、フレーム、スタンプ風装飾
- 周波数分離、局所メイク、髪のなめらか化を含む `Quality` パイプライン
- `Natural / Strong / Max / Ultra` の加工強度モード
- 元画像と加工後画像の比較表示
- 顔 bbox、目、眉、鼻、唇、頬、肌、髪近似マスクを重ねたセグメンテーションデバッグ表示
- PyTorch MPS が利用可能な環境では一部の色調処理を `torch-mps` で実行

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

## Pipeline

- `Quality`: 静止画向けの高品質処理です。周波数分離で肌の低周波だけを整え、目元シャープ、キャッチライト、チーク、リップ、髪の軽いなめらか化、肌トーン補正を局所マスクで重ねます。

PyTorch がインストールされ、Apple Silicon の MPS が使える場合は、`Quality` の色調処理の一部が自動で MPS に切り替わります。MPS 非対応の場合は OpenCV/NumPy CPU で動きます。顔パーツ推定は MediaPipe Face Mesh / FaceLandmarker、背景分離は rembg `birefnet-portrait` を必須にしています。どちらかが使えない場合は加工を継続せず、撮り直しやモデル取得状態の確認を促します。サーバ側 FaceLandmarker の task model は初回実行時に `~/.cache/purikura-demo/` へ取得します。カメラ画面ではブラウザ側でも MediaPipe Tasks の FaceLandmarker モデルを読み込み、顔検出できたフレームだけを送信します。
