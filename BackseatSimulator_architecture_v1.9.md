# BackseatSimulator
## デスクトップオーバーレイアプリケーション

**アーキテクチャ設計書 v2.1**
プラットフォーム: Windows / Python / Gemma 4
2026年4月7日

> v1.9 → v2.1: vision_tower bf16差し替えで画像認識・OCR精度を劇的改善。scene機能を完全廃止（JSON応答・プロンプト・内部状態すべて）。音声ピークノーマライズ追加。オーバーレイ最前面維持のWin32タイマー追加。デバッグダンプ機能追加。推論速度 20〜25秒→7〜13秒/サイクル。

---

## 1. プロジェクト概要

### 1.1 コンセプト

ユーザのデスクトップ画面を定期的にキャプチャし、ローカルで動作するGemma 4（Vision対応）が画面内容を解析。「ニコニコ動画の視聴者たち」としてリアルタイム風コメントを生成し、デスクトップ上を右から左へ流れるオーバーレイとして表示するアプリケーション。

### 1.2 ユースケース

- ユーザがアプリを起動すると、まずモデルがロードされる（初回30秒〜数分）
- ロード完了後、透明オーバーレイがデスクトップ全体に表示される
- 起動時に「わこつ」「はじまった」等の挨拶コメントが流れる
- 一定間隔でアクティブウィンドウをキャプチャし、AIが分析
- 変化の大きい領域を自動検出し、全体画像+フォーカスクロップの2枚をLLMに渡す
- AIが5〜8個のニコニコ風短文コメントを生成
- コメントがpending残量ベースの動的間隔で右→左へ流れる
- システムトレイアイコンからペルソナ切替・一時停止・再起動・終了が可能

### 1.3 システム要件

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11 |
| ランタイム | Python 3.10+ |
| LLMバックエンド | HuggingFace Transformers（インプロセス推論） |
| Visionモデル | **Gemma 4 E2B**（推奨）/ E4B / 26B MoE（VRAMに応じて選択） |
| GPU | RTX 3060 Ti 8GB（最小構成） |
| GUI | PyQt5（透明オーバーレイウィンドウ） |
| キャプチャ | mss + pywin32（アクティブウィンドウ取得） |

---

## 2. アーキテクチャ全体像

### 2.1 コンポーネント構成

アプリケーションは4つの主要コンポーネントで構成される。各コンポーネントはそれぞれ独立したスレッドで動作し、2つのQueueを介して通信する。

```
[Screen Capture]  ── image ──>  [AI Analyzer]  ── comment ──>  [Comment Queue]  ──>  [Overlay Renderer]
   (mss+win32)      image_queue   (Transformers)  comment_queue  (queue.Queue)     drip    (PyQt5 Window)
       |                |                                                               |
   グリッド差分       ウィンドウタイトル注入                                       pending残量
   フォーカスクロップ  vision_tower bf16差し替え                                   動的間隔制御
   ウィンドウタイトル                                                              _raise_topmost(5秒)

[Audio Capture]  ── get_audio() ──>  (AI Analyzerが画像取得時に音声スナップショットも取得)
  (sounddevice        循環バッファ
   WASAPI loopback)   16kHz mono変換
                      ピークノーマライズ
```

### 2.2 データフロー

| # | ステップ | データ | 説明 |
|---|----------|--------|------|
| 1 | Screen Capture | PNG (base64) x 1-2枚 | アクティブウィンドウをキャプチャ。グリッド差分で変化領域を検出し、全体画像+フォーカスクロップを生成。ウィンドウタイトルも取得。 |
| 1.5 | Audio Capture | numpy float32 | WASAPI loopbackでデスクトップ音声を常時キャプチャ。AI分析時に直近N秒のスナップショットを16kHz monoで取得。無音時はスキップ。 |
| 2 | AI Analysis | JSON | Transformersでインプロセス推論。画像+音声をchat templateで入力を構築。ウィンドウタイトルをsystemプロンプトに注入。 |
| 3 | Filter | list[dict] | 重複フィルター（deque直近100件）+ 先頭一致重複判定 + NGワードフィルター + ゴミフィルター。色をanalyzer側で割り当て。 |
| 4 | Drip | Comment | pending残量に応じた動的間隔でコメントを1個ずつOverlayに流す。 |
| 5 | Overlay Render | 描画イベント | PyQt5のタイマーで毎フレーム更新。Y座標スロット管理で重なり回避。 |

---

## 3. コンポーネント詳細

### 3.1 Screen Captureモジュール

**ライブラリ:** mss + pywin32

- キャプチャ間隔: デフォルト **8秒**
- キャプチャモード: `active_window`（デフォルト）/ `full_desktop`
- 画像リサイズ: 長辺 1280px に縮小
- 出力形式: base64エンコードされたPNG

**変更検知:**
- 前回キャプチャとの差分を**256pxに縮小した画像**でnumpy比較（高速化）
- 差分が閾値（0.05）未満の場合はスキップ
- **ただし `max_skip_count`（デフォルト4）回連続スキップしたら強制キャプチャ**

**動的フォーカス (v1.4):**
- 画面を3x3グリッドに分割し、セル単位でdiff値を計算
- 変化が最も大きいセルを切り出し、640pxにリサイズしてフォーカスクロップとして生成
- 均等変化判定: UNIFORM_DIFF_RATIO=0.85。画面全体が同時に変わった場合はフォーカスなし
- full_image + focus_image の2枚をLLMに渡す（focus時はvisual_token_budgetを自動2倍）

**ウィンドウタイトル取得 (v1.7):**
- win32gui.GetForegroundWindow() + GetWindowText() でアクティブウィンドウのタイトルを取得
- キャプチャ結果に window_title として同梱

**フォールバック:** active_windowモードでウィンドウハンドル取得に失敗した場合、自動的にfull_desktopにフォールバック。

### 3.2 Audio Captureモジュール (v1.9+)

**ライブラリ:** sounddevice + scipy

デスクトップ音声（ユーザに聞こえている音）をキャプチャし、Gemma 4 E2Bのネイティブ音声エンコーダに直接渡す。別途STTモデルは不要。

**キャプチャ方式:**
- sounddevice + WASAPI loopback（デフォルト出力デバイスをloopback入力として使用）
- `sd.WasapiSettings(exclusive=False)` で非排他モード
- コールバック内は生データコピーのみ（高優先度オーディオスレッドを阻害しない）

**循環バッファ:**
- ネイティブレート（通常48kHz stereo）で保持。pre-allocated numpy array + write_pos + Lock
- `get_audio()` 呼び出し時（8秒に1回）にstereo→mono + scipy.signal.resample_poly(48k→16k)
- リサンプルが3:1整数比のため高効率
- **ピークノーマライズ (v2.1):** リサンプル後に `peak / max(abs)` で正規化。WASAPI loopbackの音量がフルスケールの3%未満だった問題を解決。`[-1, 1]` クリップで安全性確保

**無音判定:**
- `get_audio()` 内でRMS計算。閾値（default 0.001）以下ならNone返却
- 無音時は音声トークン分の推論コストを節約

**Gemma 4 音声仕様:**
- 入力: 1D numpy float32、[-1,1]正規化、16kHz
- トークンレート: 40ms/token（10秒→250トークン）
- 上限: 30秒（750トークン）
- `{"type": "audio", "audio": numpy_array}` でchat templateに渡す

**音声認識の実測結果 (v2.1):**

| 機能 | 状態 | 備考 |
|------|------|------|
| プロソディ検知（エネルギー・ピッチ・テンポ） | 有効 | 音楽のテンポ変化やエネルギーの変動に反応可能 |
| ASR（音声認識・会話内容理解） | 不能（4bit量子化下） | conformer 12層の4bit量子化で音素弁別精度が失われている |

**graceful fallback:**
- `enable_audio: false`（デフォルト）の場合は一切ロードしない
- 初期化失敗時もアプリは正常起動する（音声なしで従来動作）

### 3.3 AI Analyzerモジュール

**バックエンド:** HuggingFace Transformers（インプロセス推論）
**モデルクラス:** AutoModelForMultimodalLM（音声対応を見据えた選択）
**デフォルトモデル:** google/gemma-4-E2B-it

#### モデルロード

- `load_model(config)` をアプリ起動時に1回だけ呼ぶ
- model と processor を返し、AIAnalyzer に渡す
- 量子化オプション: auto / 4bit / 8bit / none（BitsAndBytesConfig使用）
- ロード時間: 初回30秒〜数分（2回目以降はOSキャッシュで高速化）
- ロード失敗時はエラーメッセージを表示して終了

**vision_tower bf16差し替え (v2.1):**

BitsAndBytes 4bit量子化はvision_tower内の113 Linear層も量子化してしまい、画像認識・OCRが壊滅する問題があった。`llm_int8_skip_modules` は一部のLinear層で効かなかった（audio_towerと同じ問題）ため、ロード後に手動差し替えする方式を採用。

- `_dequantize_vision_tower()` 関数: ロード後にvision_tower内の `bnb.nn.Linear4bit` を検出し、safetensorsからオリジナルweightをbf16で差し替え
- VRAM増加: +302MiB（ロード時 2071→2373 MiB）。コスト極小
- 効果: 画面内テキスト（日本語・英語）のOCR、画面内容の正確な認識が可能に
- config.yaml: `vision_fp16: true` で有効化

#### 推論パイプライン

1. base64画像 → `PIL.Image` に変換（`_b64_to_pil()`）
2. messages を Transformers chat template 形式で構築
3. `processor.apply_chat_template()` で入力をトークナイズ（`enable_thinking=False`）
4. `model.generate()` で推論（`do_sample=True, temperature=1.0, top_p=0.95, top_k=64`）
5. `processor.batch_decode()` でデコード
6. `_parse_response()` でJSON解析 → コメントリスト

#### ビジュアルトークン予算

| トークン予算 | 速度 | 用途 |
|-------------|------|------|
| 140 | 最速 | テキスト読み取りも含めバランスが良い。 |
| 280 | 高速 | 細かい画面要素の認識。 |
| **1120** | 普通 | **実運用推奨 (v2.1)** vision bf16化で高解像度が活きる。focus時は自動2240。 |

> **注:** vision_tower bf16化により高解像度トークンの恩恵が劇的に改善。v1.9の140では情報を捨てすぎていた。

#### プロンプト設計

**設計方針:**
- ロール: 「ニコニコ動画の視聴者たち（複数人）」として振る舞わせる
- コメント構成: A節（画面言及、半分以上）+ B節（短いリアクション/応援、残り）
- 文字数: 15文字以内。短いほどよい
- 生成数: 5〜8個
- 色: **プロンプトでは指定しない**。analyzer側でランダム割り当て
- レスポンス形式: `{"comments": [{"text": "コメント"}]}`

**ペルソナ (v1.8):**

| キー | 名称 | 特徴 | 差別化要素 |
|------|------|------|-----------|
| **shijicyu** | 視聴者 | ニコニコ動画の視聴者。ツッコミ+リアクション | B節: リアクション辞書（草、それな、は？等18種） |
| **home** | 応援 | ポジティブ限定の視聴者。画面内容に触れつつ肯定的 | B節: 応援辞書（えらい、すごい、天才か？等18種）+ ネガティブ禁止 |

**ペルソナ設計の原則:**
- A節はshijicyu/homeで同じ粒度（「画面に見えるものに具体的に触れるコメント」）
- 差別化はB節（リアクション辞書 vs 応援辞書）とネガティブ禁止の有無で作る
- 「褒める」等の目的指示はLLMをパターン収束させるため避ける。「何をしないか」で制約した方がバリエーションが出る
- userプロンプトで最終的なトーンを制御（「ニコニコ風コメントして」vs「ポジティブなコメントして」）
- LLMが無視する指示（守られない禁止事項等）はトークンの無駄。削って他の指示の有効性を上げる

**コンテクスト注入 (v1.7, v2.1で簡素化):**

systemプロンプトに以下を動的に追加:

1. **ウィンドウタイトル**: `現在のアクティブウィンドウ: {title}` — 画像誤認を防ぐファクトのアンカー

> **注 (v2.1):** v1.6〜v1.9で使用していた場面コンテクスト（SCENE_TRANSITION_TEMPLATE / SCENE_CONTINUE_TEMPLATE、前回sceneの注入、場面転換/継続分岐）は完全廃止。vision_tower bf16化で画面認識が劇的に改善し、sceneによるコンテクスト補助が不要になった。

**プロンプト設計の知見:**
- 「画面のテキストやコードをそのまま読み上げるな。内容を踏まえた感想を言え」で丸写し防止
- 具体的なコメント例を書くとGemma 4がそのままコピペする。カテゴリ指示のみで具体コメント文は書かない
- リアクション例（B節）は18個程度のバリエーション辞書として機能する
- 禁止事項は強い口調（「死んでも使うな」）で書くとルール遵守率が上がる
- 生成数は固定（7個）より幅（5〜8個）の方が質が安定する
- ウィンドウタイトルはsystemに入れてuserは画像のみにする線引きが重要（テキスト偏重リスク防止）

**userプロンプト連結方式 (v1.9+):**

ペルソナごとにfocus有無×audio有無の4パターンを持つとスケールしない。ベース文に条件付きsuffixを連結する方式を採用:

```python
FOCUS_SUFFIX = "2枚目は注目部分のズーム。そこに見えるものに具体的に触れろ。"
AUDIO_SUFFIX = "聞こえる音にも反応しろ。会話の内容が聞き取れたら触れろ。"

user_prompt = persona["user"]                          # ベース: "この画面にニコニコ風コメントして。短く、説明なし。"
if pil_focus:  user_prompt += FOCUS_SUFFIX             # + focus
if audio_data: user_prompt += AUDIO_SUFFIX             # + audio
```

Transformers推論の入力構造:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "（ペルソナプロンプト + ウィンドウタイトル）"}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_full_image},
            # focus_image があれば:
            # {"type": "image", "image": pil_focus_image},
            # audio_data があれば:
            # {"type": "audio", "audio": numpy_array_16khz},
            {"type": "text", "text": "この画面にニコニコ風コメントして。短く、説明なし。"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt", enable_thinking=False,
).to(model.device)

outputs = model.generate(
    **inputs, max_new_tokens=120,
    do_sample=True, temperature=1.0, top_p=0.95, top_k=64,
)
```

**AI応答フォーマット:**

```json
{
  "comments": [
    {"text": "ギターやば"},
    {"text": "おいおい"},
    {"text": "センスある"},
    {"text": "それな"},
    {"text": "草"}
  ]
}
```

> **注 (v2.1):** v1.9まで存在した `"scene"` フィールド（画面説明50文字）を廃止。vision bf16化で画面認識精度が劇的改善し、sceneによるコンテクスト補助が不要になった。max_new_tokensの33〜50%がsceneに食われていた問題も解消。

> **注 (v1.9):** Ollama の `format: "json"` による強制JSONモードがなくなったため、プロンプト側でJSON出力を指示し、`_parse_response()` でthinkingタグ除去・Markdownコードブロック除去・JSON抽出フォールバックを追加。正規表現フォールバックの重要度が増加。

#### 色割り当てロジック（analyzer側）

```python
COLORS_ACCENT = ["#FF4444", "#44FF44", "#FFFF00", "#FF69B4", "#87CEEB"]

def _assign_color(self) -> str:
    if random.random() < 0.2:  # 20%の確率で色付き
        return random.choice(COLORS_ACCENT)
    return "#FFFFFF"  # 80%は白
```

#### フィルター

**重複フィルター:**
- `collections.deque(maxlen=100)` で直近100件のコメントテキストを保持
- 完全一致チェック + **先頭一致重複判定**（PREFIX_LEN=2、4文字以上のみ適用）で表記ゆれをカバー

**NGワードフィルター:**
- 「すごい」「なんか」等の頻出汎用ワード、メタ的自己言及をフィルタ
- 「何これ」「なにこれ」等の逃げコメント（モデルが画像を認識できない時に出る）をフィルタ

**ゴミフィルター:**
- コード片・ログ文字列（パス区切り・ブレース等を含む）を除去

#### レスポンスパース強化 (v1.9)

Ollama の `format: "json"` がなくなったことへの対応:

1. **thinkingタグ除去**: Gemma 4 が `<|channel>thought\n...<channel|>` を出力する場合がある → 正規表現で除去
2. **Markdownコードブロック除去**: ` ```json ... ``` ` で囲まれる場合 → 中身を抽出
3. **JSON抽出フォールバック**: テキスト+JSONが混在する場合 → 最初の `{` から最後の `}` を抽出してパース
4. **正規表現フォールバック**: 上記すべて失敗時 → `{"text": "..."}` パターンで個別抽出

### 3.3 Comment Queue

2段構成のキューで分離:

| キュー | 方向 | 用途 |
|--------|------|------|
| `image_queue` | Capture → AI | キャプチャ画像の受け渡し。古い画像は捨てて**最新のみ保持**。 |
| `comment_queue` | AI → Overlay | コメントリストの受け渡し。 |

### 3.4 Overlay Rendererモジュール

**ライブラリ:** PyQt5

デスクトップ全体を覆う透明ウィンドウを作成し、コメントを描画する。

**ウィンドウ属性:**

| 属性 | 設定値 |
|------|--------|
| FramelessWindowHint | タイトルバーなし |
| WindowStaysOnTopHint | 常に最前面 |
| WA_TranslucentBackground | 背景透明 |
| WA_TransparentForMouseEvents | マウスクリック透過（背面操作可能） |
| Tool flag (Windows) | タスクバーに非表示 |

**最前面維持 (v2.1):**
- `_raise_topmost` タイマー: 5秒ごとにWin32 `SetWindowPos(HWND_TOPMOST)` で最前面を再設定
- `SWP_NOACTIVATE` フラグでフォーカスを奪わない
- 他のアプリが最前面を奪った場合でもオーバーレイが自動復帰

**アニメーションロジック:**

- QTimerで 16ms間隔（約60fps）で描画更新
- 各コメントのX座標を毎フレームspeed分だけ減算
- 画面左端を超えたコメントはリストから除去
- **Y座標スロット管理 (v1.3):** QFontMetricsで正確なテキスト幅を計算し、コメント同士の重なりを防止
- QPainterでテキスト描画。**8方向**黒アウトラインで可読性確保

**ドリップ均等化 (v1.5):**

pending残量ベースの動的間隔制御:

| pending数 | ドリップ間隔 |
|-----------|-------------|
| 5個以上 | 2.0秒 |
| 3〜4個 | 3.0秒 |
| 1〜2個 | 4.0秒 |
| 0個 | 停止（待機） |

- MAX_PENDING=8の上限管理。オーバーフロー時のみ古いコメントを破棄
- time.monotonic()ベースの時間管理

**起動時挨拶:**
起動直後に「わこつ」「わこつです」「わこつー」「きた」「はじまった」等の挨拶コメントをドリップ方式で流す。

### 3.5 システムトレイ

**右クリックメニュー:**

| 項目 | 機能 |
|------|------|
| 一時停止 / 再開 | キャプチャの一時停止・再開。再開時にwindow_titleをリセット |
| ペルソナ | 視聴者 / 応援 のラジオボタン切替 |
| 再起動 (v1.8) | `os.execv`でプロセスを丸ごと置換。config.yaml・プロンプト変更を反映。再起動後にモデル再ロードが走る |
| 終了 | アプリケーション終了 |

---

## 4. スレッディングモデル

| スレッド | タイプ | 役割 |
|----------|--------|------|
| Main Thread | GUI (PyQt5) | PyQt5イベントループ。オーバーレイ描画とシステムトレイ。QTimerでcomment_queueをポーリング+ドリップ制御。 |
| Capture Thread | daemon | capture_interval秒ごとにキャプチャ。グリッド差分検知+フォーカスクロップ+ウィンドウタイトル取得。image_queueにput()。 |
| AI Thread | daemon | image_queueからget()。音声スナップショット取得。ウィンドウタイトル注入+コメント生成+フィルター+色付与。comment_queueにput()。 |
| Audio Thread (v1.9+) | daemon | enable_audio: true時のみ起動。sounddeviceコールバックでWASAPI loopback音声を循環バッファに常時記録。AI Threadがget_audio()で読み出し。 |

> PyQt5のGUI操作は必ずMain Threadで行うこと。Queueを介した間接通信のみを使用。
> `TOKENIZERS_PARALLELISM=false` を設定してトークナイザのスレッド競合を防止。

---

## 5. 設定項目

config.yaml でカスタマイズ可能:

| キー | デフォルト | 型 | 説明 |
|------|-----------|-----|------|
| `capture_interval` | 8 | int (sec) | キャプチャ間隔 |
| `capture_mode` | active_window | str | active_window / full_desktop |
| `image_max_size` | 1280 | int (px) | リサイズ長辺 |
| `change_threshold` | 0.05 | float | 差分検知の閾値 |
| `max_skip_count` | 4 | int | 強制キャプチャまでのスキップ回数 |
| `model_id` | google/gemma-4-E2B-it | str | HuggingFace モデルID |
| `quantization` | auto | str | auto / 4bit / 8bit / none |
| `device_map` | auto | str | auto / cpu / cuda:0 |
| `max_new_tokens` | 120 | int | 生成トークン上限（v2.1: scene廃止で256→120に削減） |
| `visual_token_budget` | 1120 | int | ビジュアルトークン予算（v2.1: vision bf16で高解像度が活きるため140→1120） |
| `persona` | shijicyu | str | ペルソナ名（shijicyu / home） |
| `enable_focus` | false | bool | 動的フォーカス機能の有効/無効（v2.1: budget増でフォーカスクロップの効果が薄れたため暫定false） |
| `focus_grid` | [3, 3] | list | グリッド分割 [rows, cols] |
| `focus_diff_threshold` | 0.05 | float | フォーカス対象とするdiff閾値 |
| `focus_crop_size` | 640 | int (px) | クロップ画像の長辺 |
| `enable_audio` | false | bool | デスクトップ音声キャプチャの有効/無効 |
| `audio_buffer_seconds` | 10 | int (sec) | 音声ローリングバッファ長（v2.1: 5→10秒に拡張） |
| `audio_device` | null | str/int | 音声デバイス（null=デフォルト出力のloopback自動検出） |
| `audio_silence_threshold` | 0.001 | float | 無音判定RMS閾値 |
| `vision_fp16` | true | bool | vision_tower bf16差し替えの有効/無効（v2.1新設） |
| `debug_dump` | false | bool | デバッグダンプの有効/無効（v2.1新設。2サイクル目にdebug_dump/へ保存） |
| `font_size` | 36 | int | コメントのフォントサイズ |
| `scroll_speed` | 3.0 | float | コメントの流れる速度 (px/frame) |
| `max_comments` | 20 | int | 同時表示コメント数の上限 |

---

## 6. セットアップ手順

```bash
# 1. 仮想環境作成 + 依存インストール
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. モデルダウンロード（初回のみ、約15GB）
python -c "from transformers import AutoProcessor, AutoModelForMultimodalLM; AutoProcessor.from_pretrained('google/gemma-4-E2B-it'); AutoModelForMultimodalLM.from_pretrained('google/gemma-4-E2B-it')"

# 3. 起動
python main.py
```

---

## 7. ディレクトリ構成

```
BackseatSimulator/
├── main.py              # エントリポイント（環境変数設定+モデルロード+3スレッド統合+コメントログ+再起動+デバッグダンプ）
├── config.yaml          # 設定ファイル
├── requirements.txt     # 依存ライブラリ
├── comments.log         # コメント履歴（自動生成、window/focus情報付き）
├── debug_dump/          # デバッグダンプ出力先（debug_dump: true時、2サイクル目に生成）
│   ├── full.png         # 全体キャプチャ画像
│   ├── focus.png        # フォーカスクロップ画像
│   ├── audio.wav        # 音声スナップショット
│   └── prompt.txt       # LLMに渡したプロンプト全文
├── capture/
│   ├── __init__.py
│   ├── screen.py       # キャプチャ + 差分検知 + グリッドフォーカス + ウィンドウタイトル取得
│   └── audio.py        # デスクトップ音声キャプチャ (WASAPI loopback + 循環バッファ)
├── ai/
│   ├── __init__.py
│   ├── analyzer.py     # Transformersインプロセス推論 + vision_tower bf16差し替え + コンテクスト注入 + フィルター + 色割り当て
│   └── prompts.py      # ペルソナ定義
├── overlay/
│   ├── __init__.py
│   ├── window.py       # PyQt5透明ウィンドウ + ドリップ均等化 + スロット管理 + 起動挨拶 + _raise_topmost
│   └── comment.py      # Commentデータクラス + アニメーション
└── tray/
    ├── __init__.py
    └── system_tray.py  # システムトレイ（ペルソナ切替 + 再起動）
```

---

## 8. VRAM管理

| モデル | dtype=auto | 4bit量子化 | 4bit + vision bf16 | 備考 |
|--------|-----------|-----------|-------------------|------|
| E2B | 約6GB | 約3.5GB | 約3.8GB (+302MiB) | RTX 3060 Ti 8GBで動作 |
| E4B | 約10GB | 約5GB | — | 12GB VRAM推奨 |

**v2.1 実測値 (E2B 4bit + vision bf16, budget=1120):**

| 状態 | VRAM |
|------|------|
| モデルロード後 | ~2373 MiB |
| 推論運用時 | ~5800 MiB |
| VRAM余裕 (8.2GB中) | ~2.3 GB |

- モデルは起動時に1回ロード、プロセス終了まで保持
- 推論ごとの `torch.cuda.empty_cache()` は不要（逆に遅くなる）
- `torch.inference_mode()` で推論時のメモリ効率を最適化
- vision_tower bf16差し替えのVRAMコストは+302MiBと極小

---

## 9. 既知の課題

| # | 課題 | ステータス | 備考 |
|---|------|-----------|------|
| 1 | 推論速度依存のバランス | 改善 | v2.1でscene廃止により7〜13秒/サイクルに高速化。ドリップ方式は引き続きフィット |
| 2 | ウィンドウタイトル偏重 | 許容 | タイトルをsystemに入れることで緩和済み。デスクトップ作業時にやや収束傾向 |
| 3 | 応援ペルソナのバリエーション | 構造的制約 | 「褒める」は画面内容から切り口を見つけにくい。ネガティブ禁止+B節応援辞書で実用的な品質 |
| 4 | JSON出力の安定性 | 要監視 | プロンプト指示+パースフォールバックに依存。thinkingタグ混入の可能性あり |
| 5 | 音声認識(ASR)の4bit量子化限界 | 構造的制約 | conformer 12層の4bit量子化で音素弁別精度が失われている。プロソディ検知（エネルギー・ピッチ・テンポ）は有効だが、会話内容の理解（ASR）は4bit量子化下では不能 |
| 6 | enable_focusの暫定無効化 | 要再検討 | budget=1120でフォーカスクロップの効果が薄れたため暫定false。特定ユースケースで再有効化の余地あり |

---

## 10. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1.0 | 2026/04/06 | 初版作成 |
| v1.1 | 2026/04/06 | Gemma 4対応。ビジュアルトークン予算機能追加 |
| v1.2 | 2026/04/06 | プロトタイプ実装知見反映。ペルソナ変更、カテゴリ指示方式、色割り当てanalyzer側移動、ドリップ方式、起動挨拶、2段キュー、強制キャプチャ、重複フィルターdeque化 |
| v1.3 | 2026/04/06 | Y座標スロット管理で重なり回避。ペルソナ4種追加(shijicyu/home/jikkyo/mix)。NGワードフィルター+先頭一致重複判定 |
| v1.4 | 2026/04/06 | 動的フォーカス（3x3グリッド差分検知+フォーカスクロップ2枚渡し） |
| v1.5 | 2026/04/06 | ドリップ均等化（pending残量ベースの動的間隔制御） |
| v1.6 | 2026/04/06 | 場面記述コンテクスト（AI生成sceneを次サイクルに注入） |
| v1.7 | 2026/04/07 | ウィンドウタイトル取得・注入。場面転換/継続テンプレート分岐 |
| v1.8 | 2026/04/07 | ペルソナ整理（4種→2種: shijicyu+home）。homeのA節から「褒める」削除、ネガティブ禁止+B節応援辞書で差別化。トレイメニューに再起動追加（os.execv）。LLMが無視する指示を削除 |
| v1.9 | 2026/04/07 | LLMバックエンドをOllama REST APIからHuggingFace Transformers（インプロセス推論）に移行。AutoModelForMultimodalLM採用（音声対応の布石）。モデルロードをmain.pyで起動時に実行。_parse_responseにthinkingタグ除去・Markdownブロック除去・JSON抽出フォールバック追加。環境変数(HF_HUB_OFFLINE等)設定追加。librosa追加（音声対応準備）。config.yamlからollama_url/model_name削除、model_id/quantization/device_map/max_new_tokens追加 |
| v1.9+audio | 2026/04/07 | デスクトップ音声入力。Gemma 4 E2Bネイティブaudio encoder活用（別STT不要）。capture/audio.py新規（WASAPI loopback+sounddevice+循環バッファ+scipy resample）。userプロンプトを連結方式に変更（user_with_focus廃止→FOCUS_SUFFIX/AUDIO_SUFFIXを条件連結）。config.yamlにaudio設定セクション追加（enable_audio: false デフォルト）。comments.logにaudioフィールド追加 |
| **v2.1** | **2026/04/07** | **vision_tower bf16差し替え（`_dequantize_vision_tower()`、+302MiB、OCR・画面認識の劇的改善）。scene機能の完全廃止（sceneフィールド・SCENE_TRANSITION/CONTINUE_TEMPLATE・_prev_scene・_update_scene・reset_scene・scene TTL関連すべて削除、JSON応答形式を`{"comments":[...]}`に簡素化）。音声ピークノーマライズ追加（`peak/max(abs)`正規化）。オーバーレイ`_raise_topmost`タイマー追加（5秒ごとSetWindowPos HWND_TOPMOST）。デバッグダンプ機能（debug_dump: true→2サイクル目にfull.png/focus.png/audio.wav/prompt.txt保存）。設定値変更: visual_token_budget 140→1120、max_new_tokens 150→120、audio_buffer_seconds 5→10、enable_focus true→false（暫定）、新設 vision_fp16/debug_dump。推論速度 20〜25秒→7〜13秒/サイクル。VRAM運用時 ~4000→~5800 MiB。AUDIO_SUFFIXに「会話の内容が聞き取れたら触れろ。」追加。プロンプト「そのままコピーするな」→「そのまま読み上げるな。内容を踏まえた感想を言え」に変更。音声認識実測: プロソディ検知有効、ASR（音声認識）は4bit量子化下では不能（conformer 12層の精度劣化）** |
