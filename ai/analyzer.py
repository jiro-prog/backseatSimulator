import collections
import json
import logging
import random
import re

import requests

from ai.prompts import PERSONAS

logger = logging.getLogger(__name__)

COLORS_ACCENT = ["#FF4444", "#44FF44", "#FFFF00", "#FF69B4", "#87CEEB"]


class AIAnalyzer:
    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "gemma4:e2b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.persona = config.get("persona", "shijicyu")
        self.visual_token_budget = config.get("visual_token_budget", 70)
        self._recent_texts: collections.deque = collections.deque(maxlen=100)

    def analyze(self, image_base64: str) -> list[dict]:
        """Ollama APIにリクエストを送り、コメントリストを返す。"""
        persona = PERSONAS.get(self.persona, PERSONAS["shijicyu"])

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": persona["system"]},
                {
                    "role": "user",
                    "content": persona["user"],
                    "images": [image_base64],
                },
            ],
            "format": {
                "type": "object",
                "properties": {
                    "comments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    }
                },
                "required": ["comments"],
            },
            "stream": False,
            "options": {
                "visual_token_budget": self.visual_token_budget,
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

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> list[dict]:
        """AI応答をパースしてコメントリストを返す。"""
        comments = []

        # まずjson.loadsを試行
        try:
            parsed = json.loads(raw)
            # トップレベルが辞書でリストを含む場合に対応
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    parsed = [parsed]
            if isinstance(parsed, list):
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
        for c in comments:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text", ""))
            if not text:
                continue
            # 30文字超なら切り詰め
            if len(text) > 30:
                text = text[:30]
            # 重複排除
            if text in seen_texts:
                continue
            seen_texts.add(text)
            # コード片・ログ文字列っぽいものだけ弾く
            if re.search(r'[{}\[\]\\/:;=]', text):
                continue
            result.append({"text": text, "color": self._assign_color()})

        # 前回と同じコメントを弾く
        result = [c for c in result if c["text"] not in self._recent_texts]
        for c in result:
            self._recent_texts.append(c["text"])

        return result

    def _assign_color(self) -> str:
        if random.random() < 0.2:
            return random.choice(COLORS_ACCENT)
        return "#FFFFFF"
