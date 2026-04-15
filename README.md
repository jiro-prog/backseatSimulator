# BackseatSimulator

**[日本語](#概要) | [English](#overview)**

<!-- デモGIFをここに配置 -->
<!-- ![BackseatSimulator Demo](docs/demo.gif) -->

> **🚧 Under Construction**
> このリポジトリは開発中のため、動作が不安定な場合があります。
> 画面認識のみで使いたい方は、セットアップが簡単な **[Lite版](https://github.com/jiro-prog/backseatSimulator-lite)** を推奨します。

---

## 概要

デスクトップ画面をリアルタイムでキャプチャし、ローカルLLM（Gemma 4）が弾幕風スクロールコメントを生成してオーバーレイ表示するアプリケーションです。デスクトップ音声のキャプチャにも対応し、映像と音声の両方に反応したコメントを生成できます。

### 主な機能

- アクティブウィンドウのリアルタイムキャプチャと差分検知
- Gemma 4 E2B による画面内容へのコメント生成（4bit量子化対応）
- 弾幕風スクロールオーバーレイ表示
- WASAPI loopback によるデスクトップ音声キャプチャ（Windows）
- ペルソナ切り替え（ヤジ / 指示厨 / ワイワイ / ミックス）
- システムトレイからの操作（一時停止・ペルソナ変更・再起動）

### 必要環境

- **OS:** Windows 10/11
- **GPU:** NVIDIA GPU（VRAM 8GB以上推奨、RTX 4060 Ti 8GB で動作確認済み）
- **Python:** 3.10+
- **CUDA:** 12.x

### セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/jiro-prog/backseatSimulator.git
cd backseatSimulator

# 2. 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate

# 3. PyTorch をインストール（CUDA バージョンに合わせる）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. 依存パッケージをインストール
pip install -r requirements.txt

# 5. 設定ファイルを作成
copy config.yaml.example config.yaml
# config.yaml を編集（model_id, quantization 等を環境に合わせて調整）

# 6. 起動
python main.py
```

初回起動時に HuggingFace からモデル（約5GB）がダウンロードされます。

2回目以降は `start.bat` をダブルクリックするだけで起動できます。

### 設定

`config.yaml.example` を `config.yaml` にコピーして編集してください。主な設定項目:

| 項目 | 説明 | デフォルト |
|------|------|-----------|
| `capture_interval` | キャプチャ間隔（秒） | `8` |
| `model_id` | HuggingFace モデルID | `google/gemma-4-E2B-it` |
| `quantization` | 量子化方式 (`4bit` / `8bit` / `none`) | `4bit` |
| `persona` | プロンプトプリセット (`heckle` / `backseat` / `hype` / `mix`) | `heckle` |
| `enable_audio` | デスクトップ音声キャプチャ | `false` |
| `ple_offload` | PLE CPU オフロード（VRAM削減） | `true` |
| `vision_fp16` | Vision tower を bf16 で保持 | `true` |

詳細は `config.yaml.example` のコメントを参照してください。

### VRAM 使用量の目安

| 構成 | ロード時 | 運用時 |
|------|----------|--------|
| 4bit + PLE offload + 選択的vision bf16 | ~2.7 GB | ~5.2 GB |
| 4bit + PLE offload なし | ~7.2 GB | ~6.0 GB超 |

RTX 4060 Ti (8GB) では `ple_offload: true` + `vision_fp16: true`（選択的ブロック指定）を推奨します。

### アーキテクチャ

詳細な設計ドキュメントは `BackseatSimulator_architecture.md` を参照してください。

---

## Overview

A desktop overlay application that captures your screen in real-time and generates scrolling danmaku-style comments using a local LLM (Gemma 4). It can also capture desktop audio via WASAPI loopback, generating comments that react to both what's on screen and what's being heard.

### Features

- Real-time active window capture with change detection
- Comment generation via Gemma 4 E2B (4-bit quantization supported)
- Danmaku-style scrolling overlay on your desktop
- Desktop audio capture via WASAPI loopback (Windows)
- Persona switching (Heckle / Backseat / Hype / Mix)
- System tray controls (pause, persona change, restart)

### Requirements

- **OS:** Windows 10/11
- **GPU:** NVIDIA GPU (8GB+ VRAM recommended, tested on RTX 4060 Ti 8GB)
- **Python:** 3.10+
- **CUDA:** 12.x

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/jiro-prog/backseatSimulator.git
cd backseatSimulator

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create config file
copy config.yaml.example config.yaml
# Edit config.yaml to match your environment

# 6. Launch
python main.py
```

The model (~5GB) will be downloaded from HuggingFace on first launch.

After the first run, just double-click `start.bat` to launch.

### Configuration

Copy `config.yaml.example` to `config.yaml` and edit. Key settings:

| Key | Description | Default |
|-----|-------------|---------|
| `capture_interval` | Capture interval (seconds) | `8` |
| `model_id` | HuggingFace model ID | `google/gemma-4-E2B-it` |
| `quantization` | Quantization mode (`4bit` / `8bit` / `none`) | `4bit` |
| `persona` | Prompt preset (`heckle` / `backseat` / `hype` / `mix`) | `heckle` |
| `enable_audio` | Desktop audio capture | `false` |
| `ple_offload` | PLE CPU offload (saves ~4.5GB VRAM) | `true` |
| `vision_fp16` | Keep vision tower in bf16 | `true` |

See `config.yaml.example` for full documentation.

### VRAM Usage

| Configuration | At Load | Runtime |
|---------------|---------|---------|
| 4bit + PLE offload + selective vision bf16 | ~2.7 GB | ~5.2 GB |
| 4bit without PLE offload | ~7.2 GB | ~6.0 GB+ |

For RTX 4060 Ti (8GB), `ple_offload: true` + `vision_fp16: true` with selective block config is recommended.

### Architecture

See `BackseatSimulator_architecture.md` for detailed design documentation.

> **🚧 Under Construction**
> This repository is under active development and may be unstable.
> If you only need screen recognition, the **[Lite version](https://github.com/jiro-prog/backseatSimulator-lite)** is recommended for its simpler setup.

---

## Disclaimer

This project is not affiliated with DWANGO Co., Ltd. or Niconico.

## License

MIT License
