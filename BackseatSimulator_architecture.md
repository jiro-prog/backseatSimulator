# BackseatSimulator
## デスクトップオーバーレイアプリケーション

**アーキテクチャ設計書 v3.3**
プラットフォーム: Windows / Python / Gemma 4
2026年4月9日

> v2.4 → v2.5: 5秒音声バッファでのASRプロンプト問題を解決（C枠の文字数制限緩和+感想コメント誘導）。audio_towerグリーディ探索完了（conformer全12層bf16必須、output_proj/subsample_convは4bit可能だが-5MiBで見合わず現状維持）。

> v2.3 → v2.4: per-layer感度分析に基づくvision_towerの選択的bf16差し替え（全層289MiB→55MiB、4ブロックで97%品質維持）。`_dequantize_tower`にブロックフィルタ追加。`vision_fp16_blocks`/`audio_fp16_blocks`設定新設。

> v2.2 → v2.3: 推論高速化（bnb_4bit_compute_dtype=bf16、quantized KV cache廃止）。音声バッファ10→5秒（音声トークン250→125、コメント数1.8→3.2個に改善）。音声反応スロット（C枠1個まで）追加。滞留フレームスキップ＋動的待機で短サイクル1コメント連鎖を解消（最終平均4.1個）。オーバーレイをWDA_EXCLUDEFROMCAPTUREでキャプチャ除外。

---

## 1. プロジェクト概要

### 1.1 コンセプト

ユーザのデスクトップ画面を定期的にキャプチャし、ローカルで動作するGemma 4（Vision対応）が画面内容を解析。弾幕風スクロールコメントを生成し、デスクトップ上を右から左へ流れるオーバーレイとして表示するアプリケーション。

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
| GPU | RTX 4060 Ti 8GB（最小構成） |
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
  (pyaudiowpatch      循環バッファ
   WASAPI loopback)   16kHz mono変換
                      RMS正規化
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

**ライブラリ:** pyaudiowpatch + scipy（v2.2でsounddevice→pyaudiowpatchに移行）

デスクトップ音声（ユーザに聞こえている音）をキャプチャし、Gemma 4 E2Bのネイティブ音声エンコーダに直接渡す。別途STTモデルは不要。

**キャプチャ方式:**
- pyaudiowpatch + WASAPI loopback（デフォルト出力デバイスをloopback入力として自動検出）
- `get_loopback_device_info_generator()` でloopbackデバイスを列挙し、デフォルト出力に対応するものを選択
- 専用読み取りスレッドでstreamからchunk単位で読み取り、循環バッファに書き込み

**循環バッファ:**
- ネイティブレート（通常48kHz stereo）で保持。pre-allocated numpy array + write_pos + Lock
- `get_audio()` 呼び出し時（8秒に1回）にstereo→mono + scipy.signal.resample_poly(48k→16k)
- リサンプルが3:1整数比のため高効率

**音声前処理パイプライン (v2.2):**

`get_audio()` 内でリサンプル後に以下の順で適用:

1. **DCオフセット除去** — `raw - np.mean(raw)`
2. **プリエンファシス** — α=0.5（`audio_preemphasis`）。高周波のSNR改善。α=0.97は攻めすぎ（「うるさい」連発）で0.5に調整
3. **HPフィルタ 150Hz** — Butterworth 4th order（`audio_highpass`）。ランブル・低周波ノイズ除去
4. **LPフィルタ** — `audio_lowpass=11000`だが16kHzサンプリングのナイキスト(8kHz)超過で自動無効化。resample_polyのLPFで十分
5. **RMS正規化** — target_rms=0.05（`audio_target_rms`）。loopbackの低音量補正

フィルタ係数はscipy.signal.butter/sosfiltで実装。係数は`__init__`で事前計算。VRAMインパクト: ゼロ（CPU上のnumpy/scipy処理）。

**RMS正規化の詳細 (v2.2):**
- WASAPI loopbackのキャプチャ音量はユーザの聴感より大幅に低い（RMS=0.003〜0.015程度）
- 正規化後に`[-1, 1]`クリップで安全性確保
- v2.1のピークノーマライズ（`peak/max(abs)`→1.0）はピーク張り付き（Peak=1.0 dBFS）によるクリッピング歪みが発生し、モデルが「ノイズ」と誤認する原因だったため廃止
- target_rms=0.05でPeak≈0.27（-11.4 dBFS）、クリッピングサンプル数ゼロを確認

**無音判定:**
- `get_audio()` 内でRMS計算。閾値（default 0.001）以下ならNone返却
- 無音時は音声トークン分の推論コストを節約

**Gemma 4 音声仕様:**
- 入力: 1D numpy float32、16kHz
- feature_extractor: 128-bin MEL spectrogram（20msフレーム、10msホップ）
- トークンレート: 40ms/token（5秒→125トークン、v2.3で10→5秒に短縮）
- 上限: 30秒（750トークン）
- `{"type": "audio", "audio": numpy_array}` でchat templateに渡す

**音声認識の実測結果 (v2.2):**

| 機能 | 状態 | 備考 |
|------|------|------|
| 楽器識別 | **有効** | ギター・ベース・ドラムを聞き分けてコメント生成 |
| 音楽ムード把握 | **有効** | テンポ変化・エネルギー変動に反応（「ライブ熱い」「テンション上がる」等） |
| 歌詞聞き取り（英語） | **部分的に有効** | 英語歌詞の断片を拾える（"NO REASON WHY I'M ONLY DOING..."等） |
| 歌詞聞き取り（日本語） | **部分的に有効** | 空耳レベルだが文脈は拾える（「後ろついて」「切り抜けて」等） |
| コンテンツ認識 | **有効** | ラジオ番組名・ON AIR等の文脈を音声+画面から統合認識 |

> **注 (v2.5):** audio_towerグリーディ探索の結果、conformer全12層がbf16必須であることを確認。output_proj(-3MiB)とsubsample_conv(-2MiB)は4bit化可能だが合計-5MiBで設定複雑化に見合わないため全層bf16を維持。v2.5でC枠プロンプトを改善し、5秒バッファでの音声反応コメントが安定動作。

> **注 (v2.2):** v2.1時点では4bit量子化下でASR不能だったが、audio_tower bf16差し替えで音声認識品質が劇的改善。完全なASRではないが、楽器・ムード・歌詞断片・コンテンツ文脈の認識が実用レベルに到達。

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

**tower bf16差し替え (v2.1 vision, v2.2 audio, v2.4 選択的差し替え):**

BitsAndBytes 4bit量子化はvision_tower/audio_tower内のLinear層も量子化してしまい、画像認識・OCR・音声認識が壊滅する問題があった。`llm_int8_skip_modules` は一部のLinear層で効かないため、ロード後に手動差し替えする方式を採用。

- `_dequantize_tower(model, model_id, tower_name, blocks=None)` 関数: 指定tower内の `bnb.nn.Linear4bit` を検出し、safetensorsからオリジナルweightをbf16で差し替え。v2.2で汎用化、v2.4で `blocks` パラメータ追加（encoder layer単位の選択的差し替え）
- `_get_block_key(module_name, tower_name)` 関数: モジュール名からブロックキー（patch_embedder, encoder.layers.N, layers.N, output_proj等）を抽出
- vision_tower: 113層（16 encoder layers × 7 modules + patch_embedder）。選択的bf16で4ブロック22層のみ差し替え（+55MiB）、全層比cos_sim 0.997以上を維持。全層差し替え時は+302MiB
- audio_tower: 134層（12 conformer layers × 11 modules + subsample_conv + output_proj）。全層bf16が必要（+582MiB）。conformer層の量子化感度が高く選択的削減は要グリーディ探索
- config.yaml: `vision_fp16: true` / `audio_fp16: true` で有効化。`vision_fp16_blocks` / `audio_fp16_blocks` でブロック指定（null=全層）

**v2.4 per-layer感度分析の結果:**

| vision_tower | cos_sim | 判定 |
|---|---|---|
| patch_embedder | 0.485 | bf16必須（入力空間の量子化誤差が全層に伝播） |
| encoder.layers.15 | 0.987 | bf16推奨（最終層） |
| encoder.layers.0, 5 | 0.993-0.994 | bf16推奨 |
| encoder.layers.1-4, 6-14 | 0.995-0.998 | 4bitで十分 |

| audio_tower | cos_sim | 判定 |
|---|---|---|
| layers.0 | 0.491 | bf16必須 |
| layers.1-5 | 0.67-0.89 | bf16推奨 |
| layers.6-11 | 0.81-0.96 | 個別では許容だが累積で劣化 |
| subsample_conv, output_proj | 0.986-0.994 | 4bit可（畳み込みは量子化ロバスト） |

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
- ロール: 弾幕コメントの視聴者たち（複数人）として振る舞わせる
- コメント構成: A節（画面言及、半分以上）+ B節（短いリアクション/応援、残り）+ C節（音声反応、1個まで、音声有りの時のみ）
- 文字数: A/B節は15文字以内。C節は30文字以内（ASR能力を活かすため緩和）
- 生成数: 5〜8個
- 色: **プロンプトでは指定しない**。analyzer側でランダム割り当て
- レスポンス形式: `{"comments": [{"text": "コメント"}]}`

**ペルソナ (v3.3):**

| キー | 名称 | max_new_tokens | summary | 特徴 |
|------|------|---------------|---------|------|
| **heckle** | ヤジ | 120 | OFF | 辛口ツッコミ。ケチをつけたりツッコんだり |
| **backseat** | 指示厨 | 150 | ON | 操作に口出し。おせっかいな常連のノリ。方向語・書き言葉禁止 |
| **hype** | ワイワイ | 120 | OFF | 面白い部分を見つけて盛り上がる。5カテゴリ（共感・驚き・発見・疑問・期待） |
| **mix** ⚠β | ミックス | 170 | ON | 3タイプ混在（β版）。煽のみ赤、他は白。88%がdict形式出力→パーサーで展開。avg 3.2コメント/cycle。煽でヤバ/マジ禁止で語彙分離 |


**ペルソナ設計の原則:**
- A節は全ペルソナで同じ粒度（「画面に映っている具体的なものを名指しして反応しろ」）
- 差別化は冒頭のタスク指示（ケチをつける / 口出しする / 盛り上がる）とB節で作る
- 「褒める」等の目的指示はLLMをパターン収束させるため避ける。タスク（見つけろ/口出ししろ）で誘導する
- userプロンプトで最終的なトーンを制御
- LLMが無視する指示（守られない禁止事項等）はトークンの無駄。削って他の指示の有効性を上げる
- 禁止語は「絶対守れ」セクションに集約すると遵守率が高い（A節に分散させると無視される）
- max_new_tokensはペルソナごとに最適値が異なる。summary生成がある場合は+30tok必要
- 口調アンカー（語尾の例示）は書き言葉への収束を防ぐのに有効
- mixモードではstring配列（`"盛:コメント"`）を指示するが、4-bit Gemmaは88%dict形式（`{"盛":"text"}`）で出力。パーサー側でdict展開して吸収。形式変更はハイリスク（実証済み）
- 語彙分離は「禁止」が有効（煽でヤバ/マジ禁止→煽ヤバ0%）。冒頭パターン/エグゼンプラは4-bitモデルに効かない
- 画面言及指示はsystemプロンプト冒頭に配置すると遵守率が上がる（絶対守れセクションだけでは埋もれる）
- パース失敗時のフォールバック（前回コメント延命）で無言の間を解消。延命は1回限りで連続延命を防止

**コンテクスト注入 (v3.2):**

systemプロンプトに以下を動的に追加:

1. **ウィンドウタイトル**: `現在のアクティブウィンドウ: {title}` — 画像誤認を防ぐファクトのアンカー
2. **状況要約** (backseatのみ): `前回までの状況: 要約1 → 要約2 → 要約3` — 直近3サイクルのキャプション。モデルがJSON出力の`summary`フィールドで生成（50文字上限）。画面遷移の文脈を提供し、的確な口出しを可能にする。heckle/hypeではトークン圧迫のため無効

> **注 (v2.1→v3.2):** v1.6〜v1.9の場面コンテクスト（AI生成scene注入）は完全廃止後、v3.2でsummary方式として軽量復活。旧sceneとの違い: (1)出力は1行のみ(10-20tok) (2)直近3件のみ保持 (3)ペルソナごとにON/OFF可能

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
AUDIO_SUFFIX = "音声も聞こえている。会話が聞こえたら内容に反応しろ。"
AUDIO_SYSTEM_SUFFIX = "\nC. 聞こえた会話への反応（1個まで、30文字以内）:\n   聞こえた話の内容を踏まえた感想を書け。「〜って言ってるw」「〜は草」のような反応\n   逐語書き起こし禁止。聞こえた内容への短いツッコミや感想にしろ\n   BGMだけ・無音ならこの枠は不要"
# audio有りの時のみsystemプロンプトに動的注入（analyzer.py）

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
    **inputs, max_new_tokens=120,  # v2.3: quantized KV cache廃止、bnb_4bit_compute_dtype=bf16
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

**キャプチャ除外 (v2.3):**
- `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` でスクリーンキャプチャAPIから除外
- 画面上では通常通り表示されるが、mss等のキャプチャにはオーバーレイが映らない
- モデルが自分の生成したコメントに反応して品質低下する問題を防止

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
| AI Thread | daemon | image_queueからget()。音声スナップショット取得。ウィンドウタイトル注入+コメント生成+フィルター+色付与。comment_queueにput()。推論後に滞留フレームをスキップ。コメント≤1個の場合はcapture_interval分追加待機（v2.3）。 |
| Audio Thread (v1.9+) | daemon | enable_audio: true時のみ起動。pyaudiowpatchでWASAPI loopback音声を循環バッファに常時記録。AI Threadがget_audio()で読み出し。 |

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
| `max_new_tokens` | 120 | int | 生成トークン上限フォールバック（ペルソナ定義側で上書き可。v3.2: heckle/hype=120, backseat=150） |
| `visual_token_budget` | 1120 | int | ビジュアルトークン予算（v2.1: vision bf16で高解像度が活きるため140→1120） |
| `persona` | heckle | str | ペルソナ名（heckle / backseat / hype / mix） |
| `mix_weights` | {hype:5,heckle:3,backseat:2} | dict | mixモード時のペルソナ比率 |
| `ple_offload` | true | bool | PLEテーブルをCPUにオフロード（VRAM ~4.5GB削減）（v1.9新設） |
| `enable_focus` | false | bool | 動的フォーカス機能の有効/無効（v2.1: budget増でフォーカスクロップの効果が薄れたため暫定false） |
| `focus_grid` | [3, 3] | list | グリッド分割 [rows, cols] |
| `focus_diff_threshold` | 0.10 | float | フォーカス対象とするdiff閾値 |
| `focus_crop_size` | 640 | int (px) | クロップ画像の長辺 |
| `enable_audio` | true | bool | デスクトップ音声キャプチャの有効/無効（v2.3でデフォルトtrue化） |
| `audio_buffer_seconds` | 5 | int (sec) | 音声ローリングバッファ長（v2.3: 10→5秒に短縮。音声トークン削減でコメント数改善） |
| `audio_device` | null | str/int | 音声デバイス（null=デフォルト出力のloopback自動検出） |
| `audio_silence_threshold` | 0.001 | float | 無音判定RMS閾値 |
| `audio_target_rms` | 0.05 | float | RMS正規化の目標レベル。0で無効（v2.2新設） |
| `audio_preemphasis` | 0.5 | float | プリエンファシス係数。0で無効（v2.2新設） |
| `audio_highpass` | 150 | int (Hz) | HPフィルタ周波数。0で無効（v2.2新設） |
| `audio_lowpass` | 11000 | int (Hz) | LPフィルタ周波数。ナイキスト超過時は自動無効（v2.2新設） |
| `vision_fp16` | true | bool | vision_tower bf16差し替えの有効/無効（v2.1新設） |
| `vision_fp16_blocks` | (list) | list/null | bf16差し替え対象ブロック。null=全層（v2.4新設） |
| `audio_fp16` | true | bool | audio_tower bf16差し替えの有効/無効（v2.2新設。+~582MiB VRAM） |
| `audio_fp16_blocks` | null | list/null | bf16差し替え対象ブロック。null=全層（v2.4新設） |
| `debug_dump` | true | bool | デバッグダンプの有効/無効（v2.1新設。2サイクル目にdebug_dump/へ保存） |
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
│   ├── analyzer.py     # Transformersインプロセス推論 + tower bf16差し替え(汎用) + コンテクスト注入 + フィルター + 色割り当て
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

| モデル | dtype=auto | 4bit量子化 | 4bit + vision bf16 | 4bit + vision&audio bf16 | 備考 |
|--------|-----------|-----------|-------------------|------------------------|------|
| E2B | 約6GB | 約3.5GB | 約3.8GB (+302MiB) | **約4.4GB (+884MiB)** | RTX 4060 Ti 8GBで動作 |
| E4B | 約10GB | 約5GB | — | — | 12GB VRAM推奨 |

**v2.4 実測値 (E2B 4bit + 選択的vision bf16 + audio全層bf16, budget=1120):**

| 状態 | VRAM | 備考 |
|------|------|------|
| モデルロード後（量子化のみ） | ~2071 MiB | |
| + vision_tower 選択的bf16 (4ブロック) | ~2127 MiB (+55) | v2.4: 全層+302から81%削減 |
| + audio_tower 全層bf16 | ~2709 MiB (+582) | |
| 推論運用時 | ~6000 MiB（v2.4実測） | v2.3比 -500MiB |
| VRAM余裕 (8.2GB中) | ~2.2 GB | |

**参考: v2.3 全層bf16時:**

| 状態 | VRAM |
|------|------|
| + vision_tower 全層bf16 | ~2373 MiB (+302) |
| + audio_tower 全層bf16 | ~2955 MiB (+582) |
| 推論運用時 | ~6500 MiB |

- モデルは起動時に1回ロード、プロセス終了まで保持
- 推論ごとの `torch.cuda.empty_cache()` は不要（逆に遅くなる）
- `torch.inference_mode()` で推論時のメモリ効率を最適化
- vision_towerの選択的bf16: patch_embedder + encoder.layers.0/5/15 の4ブロック（22/113層）で全層比cos_sim 0.997以上。感度分析スクリプト `scripts/sensitivity_analysis.py` で測定
- audio_fp32昇格は実験の結果bf16と有意差なしのため不採用（v2.3で検証済み）

---

## 9. 既知の課題

| # | 課題 | ステータス | 備考 |
|---|------|-----------|------|
| 1 | 推論速度依存のバランス | 改善 | v2.1でscene廃止により7〜13秒/サイクルに高速化。ドリップ方式は引き続きフィット |
| 2 | ウィンドウタイトル偏重 | 許容 | タイトルをsystemに入れることで緩和済み。デスクトップ作業時にやや収束傾向 |
| 3 | 応援ペルソナのバリエーション | 構造的制約 | 「褒める」は画面内容から切り口を見つけにくい。ネガティブ禁止+B節応援辞書で実用的な品質 |
| 4 | JSON出力の安定性 | 要監視 | プロンプト指示+パースフォールバックに依存。thinkingタグ混入の可能性あり |
| 5 | 音声認識の精度限界 | 改善（v2.2） | audio_tower bf16化で楽器識別・ムード把握・歌詞断片の認識が可能に。ただし完全なASR（正確な書き起こし）には至らず、日本語歌詞は空耳レベル。英語歌詞は断片的に拾える |
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
| v2.1 | 2026/04/07 | vision_tower bf16差し替え（`_dequantize_vision_tower()`、+302MiB、OCR・画面認識の劇的改善）。scene機能の完全廃止（sceneフィールド・SCENE_TRANSITION/CONTINUE_TEMPLATE・_prev_scene・_update_scene・reset_scene・scene TTL関連すべて削除、JSON応答形式を`{"comments":[...]}`に簡素化）。音声ピークノーマライズ追加（`peak/max(abs)`正規化）。オーバーレイ`_raise_topmost`タイマー追加（5秒ごとSetWindowPos HWND_TOPMOST）。デバッグダンプ機能（debug_dump: true→2サイクル目にfull.png/focus.png/audio.wav/prompt.txt保存）。設定値変更: visual_token_budget 140→1120、max_new_tokens 150→120、audio_buffer_seconds 5→10、enable_focus true→false（暫定）、新設 vision_fp16/debug_dump。推論速度 20〜25秒→7〜13秒/サイクル。VRAM運用時 ~4000→~5800 MiB。AUDIO_SUFFIXに「会話の内容が聞き取れたら触れろ。」追加。プロンプト「そのままコピーするな」→「そのまま読み上げるな。内容を踏まえた感想を言え」に変更。音声認識実測: プロソディ検知有効、ASR（音声認識）は4bit量子化下では不能（conformer 12層の精度劣化） |
| v2.2 | 2026/04/08 | audio_tower bf16差し替え（`_dequantize_tower()`汎用化、134層、+582MiB、音声認識品質の劇的改善）。ピークノーマライズ→RMS正規化に変更（target_rms=0.05、クリッピング歪み解消）。sounddevice→pyaudiowpatchに移行。音声認識実測: 楽器識別・音楽ムード把握・英語歌詞断片・コンテンツ文脈認識が実用レベルに到達。日本語歌詞は空耳レベルだが文脈は拾える。新設 audio_fp16/audio_target_rms。VRAM: ロード時2955 MiB（vision+audio bf16込み） |
| **v2.3** | **2026/04/08** | **推論高速化: bnb_4bit_compute_dtype=bf16（~20%高速化）、quantized KV cache廃止（quanto 4bitが短コンテキストで逆効果）。音声バッファ10→5秒（音声トークン250→125、コメント数1.8→4.1個に改善）。音声反応スロット: AUDIO_SYSTEM_SUFFIXでC枠（1個まで）をsystemプロンプトに動的注入、画面/音声バランス確保。パイプライン改善: 推論後の滞留フレームスキップ＋低コメント時の動的待機（≤1コメント→+capture_interval秒待機）で短サイクル1コメント連鎖を解消。オーバーレイ: WDA_EXCLUDEFROMCAPTUREで自己コメントのキャプチャ除外。プロンプト: 「5〜8個返せ。5個未満は禁止」追加。transcribeペルソナ（診断用）追加。LogitsProcessorによるEOS抑制は品質劣化で不採用。VRAM: 運用時~6500 MiB** |
| **v2.5** | **2026/04/09** | **ASRプロンプト問題の解決: 5秒バッファでaudio_tower自体はASR可能だがC枠の15文字制限+JSON形式がASR能力を圧殺していたことを特定。C枠を30文字以内+「聞こえた内容への感想コメント」誘導に変更、逐語書き起こしではなく反応コメント（「〜って言ってるw」等）に誘導しニコニコ風に収まる形で解決。audio_towerグリーディ探索: サブプロセス分離方式で全14ブロックを全bf16→段階的4bit化テスト。output_proj(-3MiB, score=1.00)とsubsample_conv(-2MiB, score=1.00)は4bit化可能だが合計-5MiBで設定複雑化に見合わず現状維持。conformer層は1層でも4bit化すると書き起こし変質、3層で幻覚・反復ループ。結論: audio_fp16=true+blocks指定なし（全層bf16）が最適、量子化削減余地なし** |
| **v3.0** | **2026/04/09** | **公開準備+機能追加。README/config.yaml.example/requirements.txtピン/start.bat新規作成。「ニコニコ動画風」→「弾幕風」表記変更（特許・商標リスク対策）。ペルソナ改修: shijicyuを辛口指示厨化（「名指しして反応+ケチをつける」）、homeをポジティブ方向に再設計。プロンプト否定形→肯定形化（研究に基づく）。repetition_penalty=1.08追加。AudioCapture.stop()未呼び出しバグ修正。デフォルト値を設計書と一致（visual_token_budget 70→1120、max_new_tokens 256→120）。再起動をサイクル完了待ち+DETACHED_PROCESS方式に変更。トレイにキャプチャモード切り替え・音声トグル追加。.state.jsonによる設定永続化。画面変化なし+音声ありで音声のみ推論モード追加（画像トークン節約）。音声あり時のmax_skip_count無効化。鉤括弧strip追加。パース診断ログ（raw→validate→dedup）追加。_parse_response()ユニットテスト18ケース新規。最前面維持タイマー5→2秒** |
| **v3.1** | **2026/04/09** | **テスト拡充+品質改善。テスト18→56ケース: test_screen_capture.py新規（差分検出・グリッド分割・隣接マージ・スキップ制御 22件）、test_audio_capture.py新規（前処理チェーン全段・循環バッファ 10件）、test_ai_loop.py新規（再起動フラグ・サイクル制御・フレーム間引き 6件）。homeペルソナ改善:「ポジティブに反応」→「面白い部分を見つけて盛り上がれ」にタスク化、A節に5カテゴリ（共感・驚き・発見・疑問・期待）、B節の具体例12個を方向性指示に置換。NG_WORDSに「すごい/すげー/すげえ」追加。エラーハンドリング: AudioCapture音声デバイス消失時の指数バックオフ自動再接続、config.yaml YAMLError catch、.state.jsonキーホワイトリスト化、mssモニター未検出チェック、snapshot_downloadキャッシュ未存在時のフォールバック** |
| **v3.3** | **2026/04/09** | **mixモード β版（マルチペルソナ混在コメント欄）。hype/heckle/backseatを1サイクル内で混在。カラー: 煽のみ赤、他は白。88%がdict形式出力→パーサーでdict展開して吸収。avg 3.2コメント/cycle、usable 97%。語彙分離: 煽でヤバ/マジ禁止（煽ヤバ0%）。フォールバック延命に`_recent_texts`重複フィルタ追加。パーサーがstring配列/dict形式/commentフィールド/タグ埋め込みの全形式に対応。鉤括弧除去をstrip→replace化。チューニング知見: 4-bit Gemmaはdict形式に強いバイアス、形式変更はハイリスク、禁止ルールが語彙分離に有効、冒頭パターン/エグゼンプラは無効** |
| **v3.2** | **2026/04/09** | **ペルソナ体系再構築+状況要約。3ペルソナ体制: heckle(ヤジ)・backseat(指示厨)・hype(ワイワイ)。キー名をshijicyu→heckle、home→hypeに英語化。backseatは新規ペルソナ（操作への口出し、画面依存タスク設計）。ペルソナごとのmax_new_tokens分離（heckle/hype=120, backseat=150）。backseatのみ状況要約（summary）有効: JSON出力に`summary`フィールド追加、直近3件をsystemプロンプトに注入（画面遷移の文脈提供）。heckle/hypeはsummary無効（120tokのトークン圧迫防止）。backseat「絶対守れ」に方向語禁止（右/左/あっち/こっち）・書き言葉禁止（すべき/である/間違ってる）追加。口調アンカー（しろよ/でよくね/にしとけ）。B節の汎用否定を間引き（は？/なんでだよは最大1個）。起動時わこつを15個プールからランダム5個抽出。NG_WORDS整理: 口出し/指示厨削除、ケチ追加。restart.logによる再起動デバッグ出力追加** |
