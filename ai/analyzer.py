import base64
import collections
import io
import json
import logging
import os
import random
import re

import numpy as np
import scipy.io.wavfile
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from ai.prompts import (AUDIO_SUFFIX, AUDIO_SYSTEM_SUFFIX, FOCUS_SUFFIX,
                        MIX_TYPE_INFO, PERSONAS, PREV_SUMMARY_TEMPLATE,
                        WINDOW_TITLE_TEMPLATE)

logger = logging.getLogger(__name__)

COLORS_ACCENT = ["#FF4444", "#44FF44", "#FFFF00", "#FF69B4", "#87CEEB"]
# mixモード: タイプ別カラー
MIX_TYPE_COLORS = {
    "盛": "#FFFFFF",   # white (hype)
    "煽": "#FF4444",   # red (heckle)
    "指": "#FFFFFF",   # white (backseat)
}
# タグ文字 → configキー
_TAG_TO_KEY = {tag: key for key, (tag, _) in MIX_TYPE_INFO.items()}
NG_WORDS = ["実況", "ツッコミ", "カテゴリ", "コメント", "リアクション", "応援",
            "何これ", "なにこれ", "何それ", "なにそれ", "なんだこれ", "何だこれ",
            "ケチ", "すごい", "すげー", "すげえ",
            "音声", "画面", "デスクトップ", "スクリーン"]
PREFIX_LEN = 2  # 先頭N文字一致で表記ゆれ重複を判定（4文字以上のコメントのみ適用）


_PLE_OFFLOAD_DEVICE_MAP = {
    "model.vision_tower": 0,
    "model.audio_tower": 0,
    "model.embed_vision": 0,
    "model.embed_audio": 0,
    "model.multimodal_projector": 0,
    "model.language_model.embed_tokens": 0,
    "model.language_model.embed_tokens_per_layer": "cpu",
    "model.language_model.per_layer_model_projection": 0,
    "model.language_model.per_layer_projection_norm": 0,
    "model.language_model.per_layer_input_norm": 0,
    "model.language_model.layers": 0,
    "model.language_model.norm": 0,
    "model.language_model.rotary_emb": 0,
    "lm_head": 0,
}


def load_model(config: dict) -> tuple:
    """
    Gemma 4 モデルとプロセッサをロードして返す。
    起動時に1回だけ呼ぶ。ロードには30秒〜数分かかる。

    Returns:
        (model, processor) のタプル
    """
    model_id = config.get("model_id", "google/gemma-4-E2B-it")
    quantization = config.get("quantization", "auto")
    ple_offload = config.get("ple_offload", False)

    logger.info("モデルロード開始: %s (quantization=%s, ple_offload=%s)",
                model_id, quantization, ple_offload)

    processor = AutoProcessor.from_pretrained(model_id)

    kwargs = {"torch_dtype": "auto"}

    if ple_offload:
        kwargs["device_map"] = _PLE_OFFLOAD_DEVICE_MAP
    else:
        kwargs["device_map"] = config.get("device_map", "auto")

    if quantization == "4bit":
        bnb_kwargs = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16}
        if ple_offload:
            bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
    elif quantization == "8bit":
        bnb_kwargs = {"load_in_8bit": True}
        if ple_offload:
            bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)

    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)

    if ple_offload:
        _setup_ple_cpu_lookup(model)

    if config.get("vision_fp16", False):
        _dequantize_tower(model, model_id, "vision_tower",
                          blocks=config.get("vision_fp16_blocks"))
    if config.get("audio_fp16", False):
        _dequantize_tower(model, model_id, "audio_tower",
                          blocks=config.get("audio_fp16_blocks"))

    logger.info("モデルロード完了 (VRAM allocated=%.0f MiB, reserved=%.0f MiB)",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2)
    return model, processor


def _get_block_key(module_name: str, tower_name: str) -> str:
    """モジュール名からブロックキー（encoder layer単位）を抽出する。"""
    if "vision_tower" in tower_name:
        if "patch_embedder" in module_name:
            return "patch_embedder"
        m = re.search(r"encoder\.layers\.(\d+)\.", module_name)
        return f"encoder.layers.{m.group(1)}" if m else "other"
    else:  # audio_tower
        if "subsample_conv_projection" in module_name:
            return "subsample_conv_projection"
        if module_name.endswith("output_proj"):
            return "output_proj"
        m = re.search(r"layers\.(\d+)\.", module_name)
        return f"layers.{m.group(1)}" if m else "other"


def _dequantize_tower(model, model_id: str, tower_name: str,
                      blocks: list[str] | None = None):
    """指定tower内の量子化Linear層をbf16で再ロードして差し替える。

    blocks: Noneなら全層（従来動作）、リストなら指定ブロックのみ差し替え。
    """
    import bitsandbytes as bnb
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    try:
        model_path = snapshot_download(model_id, local_files_only=True)
    except Exception:
        logger.warning("%s: ローカルキャッシュが見つかりません、bf16差し替えをスキップ", tower_name)
        return

    import glob as glob_mod
    st_files = glob_mod.glob(os.path.join(model_path, "*.safetensors"))

    block_set = set(blocks) if blocks else None

    # 量子化されたLinear層を特定
    quantized_modules = {}
    skipped = 0
    for name, module in model.named_modules():
        if tower_name in name and isinstance(module, bnb.nn.Linear4bit):
            if block_set is not None:
                if _get_block_key(name, tower_name) not in block_set:
                    skipped += 1
                    continue
            quantized_modules[name] = module
    if block_set is not None:
        logger.info("%s: 対象 %d個 (スキップ %d個, blocks=%s)",
                    tower_name, len(quantized_modules), skipped, sorted(block_set))
    else:
        logger.info("%s内の量子化Linear層: %d個", tower_name, len(quantized_modules))

    if not quantized_modules:
        logger.info("%s: 量子化Linear層なし、スキップ", tower_name)
        return

    # safetensorsから元のweightを読み込み
    original_weights = {}
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if tower_name in key:
                    original_weights[key] = f.get_tensor(key)

    # 量子化Linear層をtorch.nn.Linearに差し替え
    replaced = 0
    for module_name, bnb_module in quantized_modules.items():
        weight_key = module_name + ".weight"
        bias_key = module_name + ".bias"

        if weight_key not in original_weights:
            logger.warning("weightが見つからない: %s", weight_key)
            continue

        w = original_weights[weight_key].to(torch.bfloat16)
        b = original_weights.get(bias_key)
        if b is not None:
            b = b.to(torch.bfloat16)

        new_linear = torch.nn.Linear(
            w.shape[1], w.shape[0],
            bias=b is not None,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        new_linear.weight.data.copy_(w)
        if b is not None:
            new_linear.bias.data.copy_(b)

        # モジュールツリー内で差し替え
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)
        replaced += 1

    logger.info("%s: %d/%d Linear層をbf16に差し替え完了", tower_name, replaced, len(quantized_modules))

    # 検証
    dtypes = set()
    for name, param in model.named_parameters():
        if tower_name in name:
            dtypes.add(str(param.dtype))
    logger.info("%s dtypes (差し替え後): %s", tower_name, dtypes)
    logger.info("VRAM after %s dequant: %.0f MiB", tower_name, torch.cuda.memory_allocated() / 1024**2)



def _setup_ple_cpu_lookup(model):
    """accelerateのフックを外し、CPU上でlookup→結果だけGPU転送するカスタムforwardに差し替える。
    accelerateデフォルトは毎forward時にweight全体(~4.4GB)をGPUに転送するため非常に遅い。"""
    from accelerate.hooks import remove_hook_from_module

    ple = model.model.language_model.embed_tokens_per_layer
    remove_hook_from_module(ple)

    ple_weight = ple.weight.data.to("cpu").pin_memory()
    ple_padding_idx = ple.padding_idx
    ple_scale = ple.embed_scale.to(ple_weight.dtype)

    def ple_forward_cpu(input_ids):
        result = torch.nn.functional.embedding(input_ids.cpu(), ple_weight, padding_idx=ple_padding_idx)
        return (result * ple_scale).to("cuda:0")

    ple.forward = ple_forward_cpu
    logger.info("PLE CPU lookup設定完了 (weight=%.0f MiB, pinned memory)",
                ple_weight.nelement() * ple_weight.element_size() / 1024**2)


class AIAnalyzer:
    def __init__(self, config: dict, model, processor):
        self.model = model
        self.processor = processor
        self.config = config
        self.persona = config.get("persona", "heckle")
        self.visual_token_budget = config.get("visual_token_budget", 1120)
        self.max_new_tokens = config.get("max_new_tokens", 120)
        self._recent_texts: collections.deque = collections.deque(maxlen=100)
        self._prev_window_title: str = ""
        self._prev_summaries: collections.deque = collections.deque(maxlen=3)
        self._last_good_comments: list[dict] = []
        self._mix_weights: dict = config.get("mix_weights",
                                             {"hype": 5, "heckle": 3, "backseat": 2})
        self._debug_dump_countdown: int = 2 if config.get("debug_dump", False) else 0

    def analyze(self, full_image: str, focus_image: str | None = None,
                window_title: str = "",
                audio_data: "np.ndarray | None" = None) -> list[dict]:
        """Transformersでインプロセス推論し、コメントリストを返す。"""
        persona = PERSONAS.get(self.persona, PERSONAS["heckle"])

        # base64 → PIL.Image 変換
        pil_full = self._b64_to_pil(full_image) if full_image else None
        pil_focus = self._b64_to_pil(focus_image) if focus_image else None

        # userプロンプト連結方式（ベース + focus + audio）
        if pil_full is None and audio_data is not None:
            # 画面変化なし＋音声あり → 音声専用プロンプト
            user_prompt = AUDIO_SUFFIX
        else:
            user_prompt = persona["user"]
            if pil_focus:
                user_prompt = user_prompt.rstrip("。") + "。" + FOCUS_SUFFIX
            if audio_data is not None:
                user_prompt = user_prompt.rstrip("。") + "。" + AUDIO_SUFFIX

        # コンテクスト注入
        system_prompt = persona["system"]
        if "mix_ratio" in persona.get("system", ""):
            mix_keys = persona.get("mix_keys", ["hype", "heckle", "backseat"])
            system_prompt = system_prompt.replace("{mix_ratio}", self._build_mix_ratio(mix_keys))
        if audio_data is not None:
            system_prompt += AUDIO_SYSTEM_SUFFIX
        if window_title:
            system_prompt += WINDOW_TITLE_TEMPLATE.format(window_title=window_title)
        if persona.get("enable_summary") and self._prev_summaries:
            system_prompt += PREV_SUMMARY_TEMPLATE.format(
                summaries=" → ".join(self._prev_summaries))

        # messages 構築 (Transformers chat template形式)
        user_content = []
        if pil_full is not None:
            user_content.append({"type": "image", "image": pil_full})
        if pil_focus:
            user_content.append({"type": "image", "image": pil_focus})
        if audio_data is not None:
            user_content.append({"type": "audio", "audio": audio_data})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        # デバッグダンプ: 推論に渡すデータを1回だけ保存（2サイクル目で実行）
        if self._debug_dump_countdown > 0:
            self._debug_dump_countdown -= 1
            if self._debug_dump_countdown == 0:
                self._save_debug_dump(pil_full, pil_focus, audio_data,
                                      system_prompt, user_prompt)

        # visual_token_budget の適用（画像なしなら設定不要）
        if pil_full is not None:
            token_budget = self.visual_token_budget * 2 if pil_focus else self.visual_token_budget
            self.processor.image_processor.image_seq_length = token_budget
            self.processor.image_processor.max_soft_tokens = token_budget

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=persona.get("max_new_tokens", self.max_new_tokens),
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    repetition_penalty=1.08,
                )

            response_text = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )[0]

            logger.info("AI生レスポンス: %s", response_text[:500])
        except torch.cuda.OutOfMemoryError:
            logger.error("VRAM不足で推論失敗")
            return []
        except Exception:
            logger.exception("推論中にエラー")
            return []

        comments = self._parse_response(response_text)
        # mixモード: 比率フィルタ + タイプ別カラー
        if persona.get("mix_keys"):
            comments = self._apply_mix_ratio(comments, persona["mix_keys"])
        # summary抽出（enable_summary有効時のみ）
        if persona.get("enable_summary"):
            summary = self._extract_summary(response_text)
            if summary:
                self._prev_summaries.append(summary)
                logger.info("状況要約: %s", summary)
        if window_title:
            self._prev_window_title = window_title
        # フォールバック: パース失敗時に前回コメントを延命（重複除去付き）
        if not comments and self._last_good_comments:
            fallback = [c for c in self._last_good_comments
                        if c["text"] not in self._recent_texts]
            self._last_good_comments = []  # 延命は1回限り
            if fallback:
                for c in fallback:
                    self._recent_texts.append(c["text"])
                logger.info("パース失敗、前回コメント延命 (%d個)", len(fallback))
                return fallback
            logger.info("パース失敗、延命候補も重複のため破棄")
        if comments:
            self._last_good_comments = comments
        return comments

    @staticmethod
    def _b64_to_pil(b64_str: str) -> Image.Image:
        """base64文字列をPIL.Imageに変換する。"""
        return Image.open(io.BytesIO(base64.b64decode(b64_str)))

    def _parse_response(self, raw: str) -> list[dict]:
        """AI応答をパースしてコメントリストを返す。"""
        comments = []

        # 前処理: thinking タグ除去
        raw = re.sub(r'<\|channel\>thought\n.*?<channel\|>', '', raw, flags=re.DOTALL)

        # 前処理: Markdownコードブロック除去
        md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if md_match:
            raw = md_match.group(1)

        # まずjson.loadsを試行
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                comments_raw = parsed.get("comments")
                if isinstance(comments_raw, list):
                    comments = comments_raw
                else:
                    for v in parsed.values():
                        if isinstance(v, list):
                            comments = v
                            break
            elif isinstance(parsed, list):
                comments = parsed
        except json.JSONDecodeError:
            # JSONオブジェクト抽出を試みる（テキスト混在対応）
            brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if brace_match:
                try:
                    parsed = json.loads(brace_match.group())
                    if isinstance(parsed, dict):
                        comments_raw = parsed.get("comments")
                        if isinstance(comments_raw, list):
                            comments = comments_raw
                except json.JSONDecodeError:
                    pass

            # 正規表現フォールバック
            if not comments:
                # オブジェクト形式: {"text": "..."} or {"t": "...", "text": "..."}
                pattern = r'\{\s*"text"\s*:\s*"([^"]+)"\s*\}'
                matches = re.findall(pattern, raw)
                for text in matches:
                    comments.append({"text": text})
            if not comments:
                # 文字列配列形式: "盛:コメント" (mixモード)
                matches = re.findall(r'"([^"]{1,30}:[^"]{1,30})"', raw)
                for m in matches:
                    comments.append(m)  # 文字列としてcommentsに追加

        # dict形式のコメント展開: [{"盛": "text", "煽": "text"}] → ["盛:text", "煽:text"]
        expanded = []
        did_expand = False
        for c in comments:
            if isinstance(c, dict) and not c.get("text") and not c.get("comment"):
                tag_items = [(k, v) for k, v in c.items()
                             if k in _TAG_TO_KEY and isinstance(v, str)]
                if tag_items:
                    for k, v in tag_items:
                        expanded.append(f"{k}:{v}")
                    did_expand = True
                    continue
            expanded.append(c)
        if did_expand:
            comments = expanded

        # バリデーションとサニタイズ
        result = []
        seen_texts = set()
        seen_prefixes = set()
        for c in comments:
            tag = ""
            if isinstance(c, str):
                # "盛:コメント" 形式の文字列要素（mixモード）
                if ":" in c:
                    tag, text = c.split(":", 1)
                    tag = tag.strip()
                    text = text.strip()
                else:
                    text = c.strip()
            elif isinstance(c, dict):
                text = str(c.get("text", c.get("comment", ""))).strip()
                tag = str(c.get("t", "")).strip()
                # "盛: テキスト" がtext内に埋まっているケース
                if not tag and text and ":" in text[:3]:
                    maybe_tag, rest = text.split(":", 1)
                    if maybe_tag.strip() in _TAG_TO_KEY:
                        tag = maybe_tag.strip()
                        text = rest.strip()
            else:
                continue
            text = text.replace("「", "").replace("」", "")
            if not text:
                continue
            if len(text) > 30:
                text = text[:30]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            if len(text) >= 4:
                prefix = text[:PREFIX_LEN]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
            if re.search(r'[{}\[\]\\/:;=]', text):
                continue
            if any(ng in text for ng in NG_WORDS):
                continue
            entry = {"text": text, "color": self._assign_color()}
            if tag:
                entry["type"] = tag
            result.append(entry)

        pre_dedup_count = len(result)

        # 前回と同じコメントを弾く（完全一致のみ）
        result = [c for c in result if c["text"] not in self._recent_texts]
        for c in result:
            self._recent_texts.append(c["text"])

        logger.info("パース結果: raw=%d → validate=%d → dedup=%d",
                     len(comments), pre_dedup_count, len(result))
        return result

    @staticmethod
    def _extract_summary(raw: str) -> str | None:
        """AI応答からsummaryフィールドを抽出する。なければNone。"""
        # Markdownコードブロック除去
        md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        text = md_match.group(1) if md_match else raw
        # JSON抽出
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not brace_match:
            return None
        try:
            parsed = json.loads(brace_match.group())
            if isinstance(parsed, dict):
                summary = parsed.get("summary", "")
                if isinstance(summary, str) and summary.strip():
                    return summary.strip()[:50]  # 50文字上限
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _save_debug_dump(self, pil_full, pil_focus, audio_data,
                          system_prompt, user_prompt):
        """推論に渡すデータをdebug_dump/ディレクトリに保存する。"""
        dump_dir = "debug_dump"
        os.makedirs(dump_dir, exist_ok=True)
        try:
            pil_full.save(os.path.join(dump_dir, "full.png"))
            if pil_focus is not None:
                pil_focus.save(os.path.join(dump_dir, "focus.png"))
            if audio_data is not None:
                wav_path = os.path.join(dump_dir, "audio.wav")
                audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                scipy.io.wavfile.write(wav_path, 16000, audio_int16)
            with open(os.path.join(dump_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(system_prompt)
                f.write("\n\n=== USER PROMPT ===\n")
                f.write(user_prompt)
            logger.info("デバッグダンプ保存: %s", dump_dir)
        except Exception:
            logger.exception("デバッグダンプ保存失敗")

    def _build_mix_ratio(self, keys=None) -> str:
        """configのmix_weightsから比率指示テキストを生成する。"""
        if keys is None:
            keys = ["hype", "heckle", "backseat"]
        total_w = sum(self._mix_weights.get(k, 0) for k in keys)
        parts = []
        total_n = 0
        for key in keys:
            w = self._mix_weights.get(key, 0)
            if w <= 0:
                continue
            n = max(1, round(w / total_w * 8))
            total_n += n
            tag, _ = MIX_TYPE_INFO[key]
            parts.append(f"{tag}{n}個")
        return f"{total_n}個を混ぜろ: {'、'.join(parts)}"

    def _apply_mix_ratio(self, comments: list[dict],
                         keys: list[str] | None = None) -> list[dict]:
        """mixモードのコメントを比率でフィルタし、タイプ別カラーを割り当てる。"""
        if keys is None:
            keys = ["hype", "heckle", "backseat"]
        # タグ別にグループ化
        groups: dict[str, list[dict]] = {}
        untagged: list[dict] = []
        for c in comments:
            tag = c.get("type", "")
            if tag in _TAG_TO_KEY and _TAG_TO_KEY[tag] in keys:
                groups.setdefault(tag, []).append(c)
            else:
                untagged.append(c)

        # 比率に基づいて各タグから取得
        total_w = sum(self._mix_weights.get(k, 0) for k in keys)
        target_total = 8
        result = []
        for key in keys:
            w = self._mix_weights.get(key, 0)
            if w <= 0:
                continue
            tag, _ = MIX_TYPE_INFO[key]
            target_n = max(1, round(w / total_w * target_total))
            available = groups.get(tag, [])
            result.extend(available[:target_n])

        # 足りなければ未タグで補充
        remaining = target_total - len(result)
        if remaining > 0:
            result.extend(untagged[:remaining])

        # タイプ別カラー割り当て
        for c in result:
            tag = c.get("type", "")
            if tag in MIX_TYPE_COLORS:
                c["color"] = MIX_TYPE_COLORS[tag]

        random.shuffle(result)
        logger.info("mixフィルタ: 入力%d → 出力%d (盛=%d 煽=%d 指=%d 他=%d)",
                     len(comments), len(result),
                     sum(1 for c in result if c.get("type") == "盛"),
                     sum(1 for c in result if c.get("type") == "煽"),
                     sum(1 for c in result if c.get("type") == "指"),
                     sum(1 for c in result if c.get("type", "") not in _TAG_TO_KEY))
        return result

    def _assign_color(self) -> str:
        if random.random() < 0.2:
            return random.choice(COLORS_ACCENT)
        return "#FFFFFF"
