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
- 周波数分離、局所メイク、髪のなめらか化を含む `Quality` パイプライン
- `Natural / Strong / Max` の加工強度モード
- 元画像と加工後画像の比較表示
- 顔 bbox、目領域、肌マスクを重ねたセグメンテーションデバッグ表示
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
- `Classic`: 以前の軽量寄り処理です。比較や高速確認用に残しています。

PyTorch がインストールされ、Apple Silicon の MPS が使える場合は、`Quality` の色調処理の一部が自動で MPS に切り替わります。未インストールまたは MPS 非対応の場合は OpenCV/NumPy CPU で動きます。
