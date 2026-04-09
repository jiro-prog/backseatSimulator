"""_parse_response() のユニットテスト。

テーブル駆動で以下をカバー:
- 正常JSON / dict包み / リスト直接
- Markdownコードブロック付き
- thinkingタグ付き
- 壊れたJSON + テキスト混在
- 正規表現フォールバック
- 30文字切り詰め
- 完全重複除去
- プレフィックス重複除去
- 特殊文字フィルタ
- NGワードフィルタ
- _recent_texts による跨呼び出し重複除去
"""

import collections
import json
import random
import sys
import os

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeAnalyzer:
    """_parse_response と _assign_color だけを持つ軽量スタブ。"""

    def __init__(self):
        self._recent_texts: collections.deque = collections.deque(maxlen=100)

    def _assign_color(self) -> str:
        return "#FFFFFF"

    # 実装をそのまま借用
    from ai.analyzer import AIAnalyzer
    _parse_response = AIAnalyzer._parse_response


# ---------- ヘルパー ----------

def make_analyzer() -> FakeAnalyzer:
    return FakeAnalyzer()


def texts(result: list[dict]) -> list[str]:
    """コメントリストからテキストだけ抽出。"""
    return [c["text"] for c in result]


# ---------- 正常系: JSON パース ----------

class TestJsonParsing:

    def test_standard_json(self):
        """標準的な {"comments": [...]} 形式。"""
        raw = json.dumps({"comments": [{"text": "草"}, {"text": "やば"}]})
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["草", "やば"]

    def test_list_json(self):
        """リスト直接 [{...}, ...] 形式。"""
        raw = json.dumps([{"text": "きた"}, {"text": "おお"}])
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["きた", "おお"]

    def test_dict_with_arbitrary_key(self):
        """キーが "comments" 以外でもリスト値を抽出。"""
        raw = json.dumps({"results": [{"text": "すごい"}]})
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["すごい"]

    def test_markdown_code_block(self):
        """```json ... ``` で囲まれた応答。"""
        inner = json.dumps({"comments": [{"text": "なるほど"}]})
        raw = f"Here is the result:\n```json\n{inner}\n```"
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["なるほど"]

    def test_thinking_tags_removed(self):
        """thinking タグが除去されて本体がパースされる。"""
        inner = json.dumps({"comments": [{"text": "ほう"}]})
        raw = f"<|channel>thought\nこれは考え中...\n<channel|>{inner}"
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["ほう"]


# ---------- フォールバック ----------

class TestFallback:

    def test_text_mixed_json(self):
        """前後にテキストが混在する壊れた応答からJSON抽出。"""
        raw = 'Here are comments: {"comments": [{"text": "ええ"}]} end'
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["ええ"]

    def test_regex_fallback(self):
        """JSONパース完全失敗時に正規表現フォールバック。"""
        raw = 'broken {"text": "おつ"} and {"text": "わこつ"}'
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["おつ", "わこつ"]

    def test_empty_response(self):
        """空文字列 → 空リスト。"""
        result = make_analyzer()._parse_response("")
        assert result == []

    def test_no_text_key(self):
        """text キーがない dict はスキップ。"""
        raw = json.dumps([{"comment": "abc"}, {"text": "ok"}])
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["ok"]


# ---------- バリデーション・フィルタ ----------

class TestValidation:

    def test_truncate_30_chars(self):
        """30文字を超えるコメントは切り詰め。"""
        long_text = "あ" * 50
        raw = json.dumps([{"text": long_text}])
        result = make_analyzer()._parse_response(raw)
        assert len(texts(result)[0]) == 30

    def test_exact_duplicate_removed(self):
        """同一バッチ内の完全重複は除去。"""
        raw = json.dumps([{"text": "草"}, {"text": "草"}, {"text": "やば"}])
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["草", "やば"]

    def test_prefix_duplicate_removed(self):
        """4文字以上のコメントで先頭2文字一致は後のものを除去。"""
        raw = json.dumps([{"text": "それはすごい"}, {"text": "それなんだ"}])
        result = make_analyzer()._parse_response(raw)
        assert len(result) == 1
        assert texts(result)[0] == "それはすごい"

    def test_prefix_duplicate_short_exempt(self):
        """3文字以下はプレフィックス重複判定の対象外。"""
        raw = json.dumps([{"text": "草w"}, {"text": "草"}])
        result = make_analyzer()._parse_response(raw)
        # "草w" は3文字なのでプレフィックス判定なし、"草" は1文字なので同様
        assert len(result) == 2

    def test_special_chars_filtered(self):
        r"""特殊文字 {}[]\/:;= を含むコメントは除去。"""
        raw = json.dumps([
            {"text": "正常"},
            {"text": "{やば}"},
            {"text": "a=b"},
            {"text": "path\\dir"},
        ])
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["正常"]

    def test_ng_words_filtered(self):
        """NGワードを含むコメントは除去。"""
        raw = json.dumps([
            {"text": "実況うまい"},
            {"text": "ツッコミ最高"},
            {"text": "きたああ"},
        ])
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["きたああ"]

    def test_empty_text_skipped(self):
        """空テキストはスキップ。"""
        raw = json.dumps([{"text": ""}, {"text": "ok"}])
        result = make_analyzer()._parse_response(raw)
        assert texts(result) == ["ok"]


# ---------- 跨呼び出し重複 ----------

class TestCrossCallDedup:

    def test_recent_texts_dedup(self):
        """前回の呼び出しと同じコメントは弾く。"""
        analyzer = make_analyzer()
        raw1 = json.dumps([{"text": "草"}, {"text": "やば"}])
        result1 = analyzer._parse_response(raw1)
        assert texts(result1) == ["草", "やば"]

        raw2 = json.dumps([{"text": "草"}, {"text": "新しい"}])
        result2 = analyzer._parse_response(raw2)
        assert texts(result2) == ["新しい"]

    def test_color_assigned(self):
        """全コメントに color フィールドがある。"""
        raw = json.dumps([{"text": "テスト"}])
        result = make_analyzer()._parse_response(raw)
        assert result[0]["color"] == "#FFFFFF"
