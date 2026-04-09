# CLAUDE.md

## 環境

- Python仮想環境: `.venv` を使うこと（`pip install` や `python` 実行時は `.venv/Scripts/python` を経由）

## 起動ルール

- アプリ（`main.py`）を起動する前に、既存プロセスが動いていないか必ず確認し、動いていたら先にkillしてから起動すること。多重起動は厳禁

## 設計文書

- アーキテクチャ設計書: `BackseatSimulator_architecture.md`（プロジェクトルート直下）
