"""serve/test_anthropic_api.py - /v1/messages as Anthropic's SDKs read it, tool call ids through the templates, and a
streamed answer whose client stopped reading (mock engine: no GPU, no pack).

    python -m unittest serve.test_anthropic_api -v
"""
from __future__ import annotations

import json
import os
import random
import socket
import string
import sys
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, anthropic_error, serve, stream_write_timeout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096

CALL_TURNS = [
    {"role": "user", "content": "weather in Lima?"},
    {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                                      {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                       "input": {"city": "Lima"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18 C, cloudy"}]},
]


class Server:
    def __init__(self, engine=None, ctx=CTX, api_key=None):
        tok = ByteTokenizer()
        self.engine = engine or MockEngine(tok, "</think>\n\nok", max_context=ctx)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        if api_key:
            self.svc.api_key = api_key
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None, raw=None, timeout=60):
        req = urllib.request.Request(self.base + path, data=raw if raw is not None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


class AnthropicErrors(unittest.TestCase):
    """Every error under /v1/messages in Anthropic's shape ({"type": "error", "error": {"type", "message"}}), which its
    SDKs and Claude Code read; the OpenAI routes keep OpenAI's."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server()

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def assertAnthropic(self, body, kind):
        self.assertEqual(body.get("type"), "error", body)
        self.assertEqual(body["error"]["type"], kind, body)
        self.assertTrue(body["error"]["message"], body)

    def test_bad_request(self):
        code, b = self.s.post("/v1/messages", None, raw=b"[1, 2]")             # not an object
        self.assertEqual(code, 400)
        self.assertAnthropic(b, "invalid_request_error")
        code, b = self.s.post("/v1/messages?beta=true", {"model": "m", "max_tokens": 5,
                                                         "messages": [{"role": "user", "content": [7]}]})
        self.assertEqual(code, 400)                                           # a malformed block
        self.assertAnthropic(b, "invalid_request_error")

    def test_unknown_block_is_named(self):
        code, b = self.s.post("/v1/messages", {"model": "m", "max_tokens": 5, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "search_result", "source": "x"}]}]})
        self.assertEqual(code, 400)
        self.assertAnthropic(b, "invalid_request_error")
        self.assertIn("search_result", b["error"]["message"])

    def test_not_found(self):
        code, b = self.s.post("/v1/messages/nothing", {"model": "m"})
        self.assertEqual(code, 404)
        self.assertAnthropic(b, "not_found_error")

    def test_openai_routes_unchanged(self):
        code, b = self.s.post("/v1/chat/completions", None, raw=b"[1, 2]")
        self.assertEqual(code, 400)
        self.assertNotIn("type", b)
        self.assertEqual(b["error"]["type"], "invalid_request_error")

    def test_authentication(self):
        s = Server(api_key="k")
        try:
            code, b = s.post("/v1/messages", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(code, 401)
            self.assertAnthropic(b, "authentication_error")
            code, b = s.post("/v1/messages", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]},
                             headers={"x-api-key": "k"})
            self.assertEqual(code, 200, b)
        finally:
            s.close()

    def test_mapping(self):
        self.assertEqual(anthropic_error(503, {"error": {"type": "server_error", "message": "m"}})["error"]["type"],
                         "api_error")
        self.assertEqual(anthropic_error(413, {"error": {"message": "big"}})["error"]["type"], "request_too_large")
        self.assertEqual(anthropic_error(500, {"error": "plain"})["error"], {"message": "plain", "type": "api_error"})
        kept = anthropic_error(400, {"error": {"type": "invalid_request_error", "message": "m", "code": "c"}})
        self.assertEqual(kept["error"]["code"], "c")                            # (ContextOverflow's code stays)
        already = {"type": "error", "error": {"type": "engine_error", "message": "m", "partial_text": "p"}}
        self.assertIs(anthropic_error(422, already), already)


class ToolIds(unittest.TestCase):
    """tool_use ids and tool_result tool_use_ids (OpenAI: tool_calls ids and tool_call_id) reach the templates that
    read them: GLM-5.3's (glm_dsa, as before) and now Inkling's (tool_ids, ChatTemplate.inkling), which names each
    tool result by its call's id - without them its results came with no tool's name.  Other templates' prompts stay
    as they were (no ids)."""

    def test_anthropic_ids(self):
        msgs, _, _ = anthropic_to_messages({"messages": CALL_TURNS}, tool_ids=True)
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(msgs[2], {"role": "tool", "content": "18 C, cloudy", "tool_call_id": "toolu_1"})
        plain, _, _ = anthropic_to_messages({"messages": CALL_TURNS})
        self.assertNotIn("id", plain[1]["tool_calls"][0])
        self.assertNotIn("tool_call_id", plain[2])

    def test_openai_ids(self):
        req = {"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Lima\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18 C"}]}
        msgs, _, _ = openai_to_messages(req, tool_ids=True)
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msgs[2]["tool_call_id"], "call_1")
        plain, _, _ = openai_to_messages(req)
        self.assertNotIn("tool_call_id", plain[2])

    def test_inkling_names_the_result(self):
        tpl = ChatTemplate(ROOT / "tools/ik2/chat_template.jinja")
        self.assertTrue(tpl.inkling)                                           # (what the server passes as tool_ids)
        tools = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {}}}]
        msgs, t, kw = anthropic_to_messages({"messages": CALL_TURNS, "tools": tools}, tool_ids=tpl.inkling)
        self.assertIn("<|message_tool|>get_weather", tpl.render(msgs, t, **kw))
        msgs, t, kw = anthropic_to_messages({"messages": CALL_TURNS, "tools": tools})
        self.assertNotIn("<|message_tool|>get_weather", tpl.render(msgs, t, **kw))   # (before: no tool's name)


class Blocks(unittest.TestCase):
    def test_redacted_thinking_ignored(self):
        msgs, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "xyz"}, {"type": "text", "text": "yo"}]}]})
        self.assertEqual(msgs[1], {"role": "assistant", "content": "yo"})

    def test_unknown_block_raises(self):
        with self.assertRaisesRegex(ValueError, "server_tool_use"):
            anthropic_to_messages({"messages": [{"role": "assistant", "content": [{"type": "server_tool_use", "id": "s"}]}]})

    def test_image_beside_tool_result_kept(self):
        img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
        msgs, _, _ = anthropic_to_messages({"messages": CALL_TURNS[:2] + [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "done"},
            {"type": "text", "text": "and this picture:"}, img]}]})
        self.assertEqual(msgs[2]["role"], "tool")
        self.assertEqual(msgs[3]["role"], "user")
        self.assertEqual([p["type"] for p in msgs[3]["content"]], ["text", "image"])


class StuckStreamClient(unittest.TestCase):
    """A streamed answer whose client stops reading with its connection open (a suspended laptop) holds the queue's
    turn until a write gives up: stream_write_timeout() (MAYA_STREAM_WRITE_TIMEOUT_S), not the 300 s request
    timeout.  The next request then runs."""

    def test_timeout_setting(self):
        with unittest.mock.patch.dict(os.environ, {"MAYA_STREAM_WRITE_TIMEOUT_S": "", "INKLING-CUSTOM-ENGINE_REQUEST_TIMEOUT_S": ""}):
            self.assertEqual(stream_write_timeout(), 30.0)
        with unittest.mock.patch.dict(os.environ, {"MAYA_STREAM_WRITE_TIMEOUT_S": "2.5"}):
            self.assertEqual(stream_write_timeout(), 2.5)
        with unittest.mock.patch.dict(os.environ, {"MAYA_STREAM_WRITE_TIMEOUT_S": "900", "INKLING-CUSTOM-ENGINE_REQUEST_TIMEOUT_S": "60"}):
            self.assertEqual(stream_write_timeout(), 60.0)                    # never above the request timeout

    def test_stuck_client_frees_the_turn(self):
        n = 120_000                                     # ~30 MB of SSE frames: past any socket buffer
        answer = "".join(random.Random(7).choices(string.ascii_lowercase, k=n))
        tok = ByteTokenizer()
        with unittest.mock.patch.dict(os.environ, {"MAYA_STREAM_WRITE_TIMEOUT_S": "1"}):
            s = Server(MockEngine(tok, ["</think>\n\n" + answer, "</think>\n\nok"], max_context=n + 1024), ctx=n + 1024)
            stuck = socket.socket()
            try:
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", s.port))
                body = json.dumps({"model": "m", "stream": True, "max_tokens": n,
                                   "messages": [{"role": "user", "content": "go"}]}).encode()
                stuck.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                time.sleep(1.0)                         # its answer starts, then its writes block: nobody reads
                t0 = time.time()
                code, b = s.post("/v1/messages", {"model": "m", "max_tokens": 5,
                                                  "messages": [{"role": "user", "content": "next"}]}, timeout=400)
                waited = time.time() - t0
                self.assertEqual(code, 200, b)
                # (~15 s here: the stuck answer first fills ~4 MB of socket buffers, then gives up after 1 s; before,
                #  each blocked write waited the request timeout, 300 s)
                self.assertLess(waited, 120, "the next request waited for the stuck stream")
            finally:
                stuck.close()
                s.close()


if __name__ == "__main__":
    unittest.main()
