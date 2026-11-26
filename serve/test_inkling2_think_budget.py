"""Inkling 2's reasoning ends at <|content_text|>, not at </think>: the server must find that token
or it sets think_budget = 0 and a short request's reasoning consumes every output token, leaving
content: null (upstream issue #19, request 4).

    python -m unittest serve.test_inkling2_think_budget -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class InklingTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<|content_text|>"]


class InklingThinkEnd(unittest.TestCase):
    def test_inkling_template_uses_the_typed_text_token(self):
        tok = InklingTokenizer()
        tok.special_tokens = {s: 256 + k for k, s in enumerate(tok.SPECIALS)}
        tpl = ChatTemplate(ROOT / "tools/ik2/chat_template.jinja")
        self.assertTrue(tpl.inkling)
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, tpl)
        self.assertEqual(svc.think_end_id, tok.special_tokens["<|content_text|>"])

    def test_a_plain_template_still_uses_think_end(self):
        tok = ByteTokenizer()
        tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        self.assertFalse(tpl.inkling)
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, tpl)
        self.assertIsNone(svc.think_end_id)   # ByteTokenizer has no </think> special


if __name__ == "__main__":
    unittest.main()
