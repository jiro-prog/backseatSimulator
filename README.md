# BackseatSimulator

デスクトップ画面をリアルタイムでキャプチャし、ローカルLLM（Gemma 4）が弾幕風スクロールコメントを生成してオーバーレイ表示するアプリケーションです。デスクトップ音声のキャプチャにも対応し、映像と音声の両方に反応したコメントを生成できます。

## 主な機能

- アクティブウィンドウのリアルタイムキャプチャと差分検知
- Gemma 4 E2B による画面内容へのコメント生成（4bit量子化対応）
- 弾幕風スクロールオーバーレイ表示
- WASAPI loopback によるデスクトップ音声キャプチャ（Windows）
- ペルソナ切り替え（実況ツッコミ / 応援）
- システムトレイからの操作（一時停止・ペルソナ変更・再起動）

## 必要環境

- **OS:** Windows 10/11
- **GPU:** NVIDIA GPU（VRAM 8GB以上推奨、RTX 4060 Ti 8GB で動作確認済み）
- **Python:** 3.10+
- **CUDA:** 12.x

## セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/<your-username>/backseatSimulator.git
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

## 設定

`config.yaml.example` を `config.yaml` にコピーして編集してください。主な設定項目:

| 項目 | 説明 | デフォルト |
|------|------|-----------|
| `capture_interval` | キャプチャ間隔（秒） | `8` |
| `model_id` | HuggingFace モデルID | `google/gemma-4-E2B-it` |
| `quantization` | 量子化方式 (`4bit` / `8bit` / `none`) | `4bit` |
| `persona` | プロンプトプリセット (`heckle` / `backseat` / `hype`) | `heckle` |
| `enable_audio` | デスクトップ音声キャプチャ | `false` |
| `ple_offload` | PLE CPU オフロード（VRAM削減） | `true` |
| `vision_fp16` | Vision tower を bf16 で保持 | `true` |

詳細は `config.yaml.example` のコメントを参照してください。

## VRAM 使用量の目安

| 構成 | ロード時 | 運用時 |
|------|----------|--------|
| 4bit + PLE offload + 選択的vision bf16 | ~2.7 GB | ~6.0 GB |
| 4bit + PLE offload なし | ~7.2 GB | ~6.0 GB超 |

RTX 4060 Ti (8GB) では `ple_offload: true` + `vision_fp16: true`（選択的ブロック指定）を推奨します。

## アーキテクチャ

詳細な設計ドキュメントは `BackseatSimulator_architecture.md` を参照してください。

## 免責事項

This project is not affiliated with DWANGO Co., Ltd. or Niconico.

## ライセンス

MIT License
