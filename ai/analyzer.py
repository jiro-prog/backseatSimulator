import collections
import json
import logging
import random
import re

import requests

from ai.prompts import (PERSONAS, SCENE_CONTINUE_TEMPLATE,
                        SCENE_TRANSITION_TEMPLATE, WINDOW_TITLE_TEMPLATE)

logger = logging.getLogger(__name__)

COLORS_ACCENT = ["#FF4444", "#44FF44", "#FFFF00", "#FF69B4", "#87CEEB"]
NG_WORDS = ["実況", "ツッコミ", "カテゴリ", "コメント", "リアクション", "応援",
            "何これ", "なにこれ", "何それ", "なにそれ", "なんだこれ", "何だこれ"]
PREFIX_LEN = 2  # 先頭N文字一致で表記ゆれ重複を判定（4文字以上のコメントのみ適用）


class AIAnalyzer:
    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "gemma4:e2b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.persona = config.get("persona", "shijicyu")
        self.visual_token_budget = config.get("visual_token_budget", 70)
        self._recent_texts: collections.deque = collections.deque(maxlen=100)
        self._prev_scene: str | None = None
        self._prev_window_title: str = ""

    def analyze(self, full_image: str, focus_image: str | None = None,
                window_title: str = "") -> list[dict]:
        """Ollama APIにリクエストを送り、コメントリストを返す。"""
        persona = PERSONAS.get(self.persona, PERSONAS["shijicyu"])

        if focus_image:
            images = [full_image, focus_image]
            user_prompt = persona.get("user_with_focus", persona["user"])
        else:
            images = [full_image]
            user_prompt = persona["user"]

        # コンテクスト注入
        system_prompt = persona["system"]
        if window_title:
            system_prompt += WINDOW_TITLE_TEMPLATE.format(window_title=window_title)
        if self._prev_scene:
            # ウィンドウが変わった = 場面転換、同じ = 同一場面継続
            if window_title and self._prev_window_title and window_title != self._prev_window_title:
                system_prompt += SCENE_TRANSITION_TEMPLATE.format(
                    prev_window=self._prev_window_title,
                    prev_scene=self._prev_scene,
                    window=window_title,
                )
            else:
                system_prompt += SCENE_CONTINUE_TEMPLATE.format(prev_scene=self._prev_scene)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": images,
                },
            ],
            "format": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string"},
                    "comments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    },
                },
                "required": ["scene", "comments"],
            },
            "stream": False,
            "options": {
                "visual_token_budget": self.visual_token_budget * 2 if focus_image else self.visual_token_budget,
                "temperature": 1.1,
                "frequency_penalty": 0.3,
                "repeat_last_n": 128,
            },
        }

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException:
            logger.exception("Ollama接続失敗")
            return []

        try:
            data = resp.json()
            raw = data.get("message", {}).get("content", "")
            logger.info("AI生レスポンス: %s", raw[:500])
        except (ValueError, KeyError):
            logger.exception("レスポンスの解析失敗")
            return []

        comments = self._parse_response(raw)
        if window_title:
            self._prev_window_title = window_title
        return comments

    def _parse_response(self, raw: str) -> list[dict]:
        """AI応答をパースしてコメントリストを返す。sceneは内部で保持。"""
        comments = []

        # まずjson.loadsを試行
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # sceneを抽出・保存
                scene = parsed.get("scene")
                if isinstance(scene, str) and scene:
                    if len(scene) > 50:
                        scene = scene[:50]
                    self._prev_scene = scene
                    logger.info("scene: %s", scene)
                # commentsを抽出
                comments_raw = parsed.get("comments")
                if isinstance(comments_raw, list):
                    comments = comments_raw
                else:
                    # commentsキーがない場合、リスト値を探すフォールバック
                    for v in parsed.values():
                        if isinstance(v, list):
                            comments = v
                            break
            elif isinstance(parsed, list):
                comments = parsed
        except json.JSONDecodeError:
            # 正規表現フォールバック
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
            # 30文字超なら切り詰め
            if len(text) > 30:
                text = text[:30]
            # 重複排除（完全一致）
            if text in seen_texts:
                continue
            seen_texts.add(text)
            # 先頭N文字一致の表記ゆれ重複排除（4文字以上のみ）
            if len(text) >= 4:
                prefix = text[:PREFIX_LEN]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
            # コード片・ログ文字列っぽいものだけ弾く
            if re.search(r'[{}\[\]\\/:;=]', text):
                continue
            # メタ的ワ���ドフィルター
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
