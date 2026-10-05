"""Unit tests for the Antigravity vision and image materialization bridge."""

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import native_tools
from client import AntigravityClient
from prompt import (
    _materialize_image_item,
    _render_message_content,
    _format_messages_as_prompt,
)


class VisionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.media_dir = Path(self.tmp.name) / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def test_materialize_base64_data_url(self):
        # 1x1 transparent PNG
        raw_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00"
            b"\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64_str = base64.b64encode(raw_png).decode("ascii")
        data_url = f"data:image/png;base64,{b64_str}"

        item = {"type": "image_url", "image_url": {"url": data_url}}
        out_path = _materialize_image_item(item, self.media_dir)

        self.assertIsNotNone(out_path)
        self.assertTrue(Path(out_path).exists())
        self.assertEqual(Path(out_path).read_bytes(), raw_png)
        self.assertTrue(out_path.endswith(".png"))

    def test_materialize_local_file_path(self):
        dummy_file = self.media_dir / "sample.jpg"
        dummy_file.write_bytes(b"dummy jpeg bytes")

        item = {"type": "image_url", "image_url": {"url": str(dummy_file)}}
        out_path = _materialize_image_item(item, self.media_dir)
        self.assertEqual(out_path, str(dummy_file.resolve()))

    def test_render_message_content_with_image_list(self):
        raw_png = b"\x89PNG\r\ndummy"
        b64_str = base64.b64encode(raw_png).decode("ascii")
        content = [
            {"type": "text", "text": "What is in this screenshot?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
        ]
        rendered = _render_message_content(content, self.media_dir)

        self.assertIn("What is in this screenshot?", rendered)
        self.assertIn("[Attached image file:", rendered)
        self.assertIn("view_file", rendered)

    def test_render_message_content_with_multimodal_dict(self):
        raw_png = b"\x89PNG\r\ndummy"
        b64_str = base64.b64encode(raw_png).decode("ascii")
        content = {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "Image loaded into your context"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
            ],
        }
        rendered = _render_message_content(content, self.media_dir)

        self.assertIn("Image loaded into your context", rendered)
        self.assertIn("[Attached image file:", rendered)
        self.assertNotIn("data:image/png", rendered)

    def test_render_message_content_with_json_multimodal_string(self):
        raw_png = b"\x89PNG\r\ndummy"
        b64_str = base64.b64encode(raw_png).decode("ascii")
        envelope = {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "Image loaded into your context"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
            ],
        }
        rendered = _render_message_content(json.dumps(envelope), self.media_dir)

        self.assertIn("Image loaded into your context", rendered)
        self.assertIn("[Attached image file:", rendered)
        self.assertNotIn("data:image/png", rendered)

    def test_render_message_content_with_embedded_b64_in_text(self):
        raw_png = b"\x89PNG\r\ndummy"
        b64_str = base64.b64encode(raw_png).decode("ascii")
        text = f"Check this out: data:image/png;base64,{b64_str} please!"
        rendered = _render_message_content(text, self.media_dir)

        self.assertIn("Check this out:", rendered)
        self.assertIn("[Attached image file:", rendered)
        self.assertNotIn("data:image/png", rendered)

    def test_native_image_view_check(self):
        step_img = {
            "step_type": "tool",
            "tool_name": "view_file",
            "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/tmp/pic.png"}},
        }
        self.assertTrue(native_tools.is_native_image_view(step_img))
        # Must not translate to Hermes read_file
        self.assertIsNone(native_tools.translate(step_img, {"read_file"}, 0))

        step_code = {
            "step_type": "tool",
            "tool_name": "view_file",
            "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/tmp/main.py"}},
        }
        self.assertFalse(native_tools.is_native_image_view(step_code))
        call = native_tools.translate(step_code, {"read_file"}, 0)
        self.assertIsNotNone(call)
        self.assertEqual(call.function.name, "read_file")

    def test_stream_allows_native_image_view_without_terminating(self):
        events = [
            {"event": "init", "conversation_id": "conv-1"},
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "tool_name": "view_file",
                    "tool_info": {"parameters": {"AbsolutePath": "/data/chart.png"}},
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "text_delta": "The chart shows revenue up 25%.",
                },
            },
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "The chart shows revenue up 25%.",
                },
            },
        ]

        client = AntigravityClient(cwd=self.tmp.name)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.side_effect = [json.dumps(e) + "\n" for e in events] + [""] * 20
        proc.stdin = MagicMock()

        with patch("client.is_authenticated", return_value=True), \
             patch("subprocess.Popen", return_value=proc), \
             patch.object(client, "_terminate_process") as term:
            chunks = list(
                client.chat.completions.create(
                    model="gemini-3.8-flash",
                    stream=True,
                    messages=[{"role": "user", "content": "Analyze /data/chart.png"}],
                )
            )

        # Process should NOT have been terminated early during the image view
        self.assertFalse(term.called)
        # Content chunk must have been yielded
        texts = [
            ch.choices[0].delta.content
            for ch in chunks
            if ch.choices and getattr(ch.choices[0].delta, "content", None)
        ]
        self.assertIn("The chart shows revenue up 25%.", texts)


if __name__ == "__main__":
    unittest.main()
