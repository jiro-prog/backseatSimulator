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
                        PERSONAS, WINDOW_TITLE_TEMPLATE)

logger = logging.getLogger(__name__)

COLORS_ACCENT = ["#FF4444", "#44FF44", "#FFFF00", "#FF69B4", "#87CEEB"]
NG_WORDS = ["実況", "ツッコミ", "カテゴリ", "コメント", "リアクション", "応援",
            "何これ", "なにこれ", "何それ", "なにそれ", "なんだこれ", "何だこれ"]
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
        _dequantize_tower(model, model_id, "vision_tower")
    if config.get("audio_fp16", False):
        _dequantize_tower(model, model_id, "audio_tower")

    logger.info("モデルロード完了 (VRAM allocated=%.0f MiB, reserved=%.0f MiB)",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2)
    return model, processor


def _dequantize_tower(model, model_id: str, tower_name: str):
    """指定tower内の量子化Linear層をbf16で再ロードして差し替える。"""
    import bitsandbytes as bnb
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(model_id, local_files_only=True)

    import glob as glob_mod
    st_files = glob_mod.glob(os.path.join(model_path, "*.safetensors"))

    # 量子化されたLinear層を特定
    quantized_modules = {}
    for name, module in model.named_modules():
        if tower_name in name and isinstance(module, bnb.nn.Linear4bit):
            quantized_modules[name] = module
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
        self.persona = config.get("persona", "shijicyu")
        self.visual_token_budget = config.get("visual_token_budget", 70)
        self.max_new_tokens = config.get("max_new_tokens", 256)
        self._recent_texts: collections.deque = collections.deque(maxlen=100)
        self._prev_window_title: str = ""
        self._debug_dump_countdown: int = 2 if config.get("debug_dump", False) else 0

    def analyze(self, full_image: str, focus_image: str | None = None,
                window_title: str = "",
                audio_data: "np.ndarray | None" = None) -> list[dict]:
        """Transformersでインプロセス推論し、コメントリストを返す。"""
        persona = PERSONAS.get(self.persona, PERSONAS["shijicyu"])

        # base64 → PIL.Image 変換
        pil_full = self._b64_to_pil(full_image)
        pil_focus = self._b64_to_pil(focus_image) if focus_image else None

        # userプロンプト連結方式（ベース + focus + audio）
        user_prompt = persona["user"]
        if pil_focus:
            user_prompt = user_prompt.rstrip("。") + "。" + FOCUS_SUFFIX
        if audio_data is not None:
            user_prompt = user_prompt.rstrip("。") + "。" + AUDIO_SUFFIX

        # コンテクスト注入
        system_prompt = persona["system"]
        if audio_data is not None:
            system_prompt += AUDIO_SYSTEM_SUFFIX
        if window_title:
            system_prompt += WINDOW_TITLE_TEMPLATE.format(window_title=window_title)

        # messages 構築 (Transformers chat template形式)
        user_content = [{"type": "image", "image": pil_full}]
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

        # visual_token_budget の適用（focus時は2倍）
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
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
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
        if window_title:
            self._prev_window_title = window_title
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
                pattern = r'\{\s*"text"\s*:\s*"([^"]+)"\s*\}'
                matches = re.findall(pattern, raw)
                for text in matches:
                    comments.append({"text": text})

        # バリデーションとサニタイズ
        result = []
        seen_texts = set()
        seen_prefixes = set()
        for c in comments:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text", ""))
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
            result.append({"text": text, "color": self._assign_color()})

        # 前回と同じコメントを弾く（完全一致のみ）
        result = [c for c in result if c["text"] not in self._recent_texts]
        for c in result:
            self._recent_texts.append(c["text"])

        return result

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

    def _assign_color(self) -> str:
        if random.random() < 0.2:
            return random.choice(COLORS_ACCENT)
        return "#FFFFFF"
