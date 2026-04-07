import base64
import collections
import io
import json
import logging
import random
import re

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from ai.prompts import (PERSONAS, SCENE_CONTINUE_TEMPLATE,
                        SCENE_TRANSITION_TEMPLATE, WINDOW_TITLE_TEMPLATE)

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
        bnb_kwargs = {"load_in_4bit": True}
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

    logger.info("モデルロード完了 (VRAM allocated=%.0f MiB, reserved=%.0f MiB)",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2)
    return model, processor


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
        self._prev_scene: str | None = None
        self._prev_window_title: str = ""

    def analyze(self, full_image: str, focus_image: str | None = None,
                window_title: str = "") -> list[dict]:
        """Transformersでインプロセス推論し、コメントリストを返す。"""
        persona = PERSONAS.get(self.persona, PERSONAS["shijicyu"])

        # base64 → PIL.Image 変換
        pil_full = self._b64_to_pil(full_image)

        if focus_image:
            pil_focus = self._b64_to_pil(focus_image)
            user_prompt = persona.get("user_with_focus", persona["user"])
        else:
            pil_focus = None
            user_prompt = persona["user"]

        # コンテクスト注入
        system_prompt = persona["system"]
        if window_title:
            system_prompt += WINDOW_TITLE_TEMPLATE.format(window_title=window_title)
        if self._prev_scene:
            if window_title and self._prev_window_title and window_title != self._prev_window_title:
                system_prompt += SCENE_TRANSITION_TEMPLATE.format(
                    prev_window=self._prev_window_title,
                    prev_scene=self._prev_scene,
                    window=window_title,
                )
            else:
                system_prompt += SCENE_CONTINUE_TEMPLATE.format(prev_scene=self._prev_scene)

        # messages 構築 (Transformers chat template形式)
        user_content = [{"type": "image", "image": pil_full}]
        if pil_focus:
            user_content.append({"type": "image", "image": pil_focus})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

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
                    cache_implementation="quantized",
                    cache_config={"backend": "quanto", "nbits": 4},
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
        """AI応答をパースしてコメントリストを返す。sceneは内部で保持。"""
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
                scene = parsed.get("scene")
                if isinstance(scene, str) and scene:
                    if len(scene) > 50:
                        scene = scene[:50]
                    self._prev_scene = scene
                    logger.info("scene: %s", scene)
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
                        scene = parsed.get("scene")
                        if isinstance(scene, str) and scene:
                            if len(scene) > 50:
                                scene = scene[:50]
                            self._prev_scene = scene
                            logger.info("scene: %s", scene)
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

    def reset_scene(self):
        """一時停止→再開時などにsceneをクリアする。"""
        self._prev_scene = None
        self._prev_window_title = ""

    def _assign_color(self) -> str:
        if random.random() < 0.2:
            return random.choice(COLORS_ACCENT)
        return "#FFFFFF"
