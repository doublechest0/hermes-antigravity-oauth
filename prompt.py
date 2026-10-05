"""Prompt construction, message translation, and tool-call parsing for Antigravity."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Sequence
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}

_PROMPT_PREAMBLE = (
    "You are being used strictly as an LLM inference backend for Hermes Agent.",
    "All tool executions and filesystem interactions are performed exclusively by Hermes Agent.",
    "You MUST NOT attempt to invoke native agent tools (such as run_command, write_to_file, etc.).",
    "EXCEPTION FOR IMAGES: If an attached image file is provided in the prompt, you MUST use your native view_file tool to inspect the image and see its visual content.",
    "IMPORTANT INSTRUCTIONS FOR TOOLS:",
    "- If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output.",
    "- Each tool call must be a JSON object containing 'id', 'type': 'function', and 'function': {'name': '...', 'arguments': '...'}.",
    "- 'arguments' must be a JSON-encoded string containing the function arguments.",
    "- Do NOT execute local shell commands or file operations directly; only output <tool_call> tags so Hermes can execute them safely.",
    "- NEVER simulate or hallucinate Tool Results (e.g. 'Tool Result (...):'). Hermes Agent executes tools externally and will supply real results in subsequent turns.",
    "- After emitting tool calls, STOP generating immediately. Do NOT generate results, execution output, or commentary after tool calls.",
    "- If no tool is needed, respond naturally with standard text.",
)

_TOOL_CALL_PREFIXES = tuple(
    "<tool_call>"[:i] for i in range(len("<tool_call>"), 0, -1)
)

IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".tiff", ".ico"
})

MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "image/tiff": ".tiff",
}


def _materialize_image_item(item: dict[str, Any], media_dir: str | Path | None = None) -> str | None:
    """Extract or decode image from an image_url dict and return a local absolute file path."""
    url_val = None
    if "image_url" in item:
        iu = item["image_url"]
        if isinstance(iu, dict):
            url_val = iu.get("url")
        elif isinstance(iu, str):
            url_val = iu
    elif item.get("type") in ("image_url", "image") and "url" in item:
        url_val = item["url"]
    elif item.get("type") == "image" and "source" in item:
        source = item["source"]
        if isinstance(source, dict) and source.get("type") == "base64":
            data = source.get("data", "")
            media_type = source.get("media_type", "image/png")
            url_val = f"data:{media_type};base64,{data}"

    if not url_val or not isinstance(url_val, str):
        return None

    target_dir = Path(media_dir) if media_dir else Path(tempfile.gettempdir()) / "hermes_agy_media"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        target_dir = Path(tempfile.gettempdir())

    if url_val.startswith("data:image/"):
        try:
            header, b64 = url_val.split(",", 1)
            mime = header.split(";")[0].split(":", 1)[1].strip().lower()
            ext = MIME_TO_EXT.get(mime, ".png")
            raw_bytes = base64.b64decode(b64)
            file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            out_path = target_dir / f"img_{file_hash}{ext}"
            if not out_path.exists():
                out_path.write_bytes(raw_bytes)
            return str(out_path.resolve())
        except Exception as exc:
            logger.warning("Failed to decode base64 image: %s", exc)
            return None

    if url_val.startswith("file://"):
        parsed = urllib.parse.urlparse(url_val)
        local_path = urllib.request.url2pathname(parsed.path)
        if os.name == "nt" and local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
            local_path = local_path[1:]
        if os.path.exists(local_path):
            return str(Path(local_path).resolve())

    if os.path.exists(url_val):
        return str(Path(url_val).resolve())

    if url_val.startswith(("http://", "https://")):
        try:
            file_hash = hashlib.sha256(url_val.encode()).hexdigest()[:16]
            ext = ".png"
            for e in IMAGE_EXTENSIONS:
                if url_val.lower().endswith(e):
                    ext = e
                    break
            out_path = target_dir / f"web_{file_hash}{ext}"
            if not out_path.exists():
                req = urllib.request.Request(url_val, headers={"User-Agent": "Hermes-Agent"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    out_path.write_bytes(resp.read())
            return str(out_path.resolve())
        except Exception as exc:
            logger.warning("Failed to download image URL %s: %s", url_val, exc)
            return None

    return None


def _render_message_content(content: Any, media_dir: str | Path | None = None) -> str:
    """Normalize multimodal or structured message content into a string, preserving image links."""
    if content is None:
        return ""
    if isinstance(content, dict):
        if content.get("type") in ("image_url", "image") or "image_url" in content:
            img_path = _materialize_image_item(content, media_dir)
            if img_path:
                return f'[Attached image file: "{img_path}". You MUST use your native view_file tool with AbsolutePath="{img_path}" to inspect this image.]'
        if "text" in content:
            return str(content.get("text") or "").strip()
        return str(content.get("content") or "").strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                if item.get("type") in ("image_url", "image") or "image_url" in item:
                    img_path = _materialize_image_item(item, media_dir)
                    if img_path:
                        parts.append(f'[Attached image file: "{img_path}". You MUST use your native view_file tool with AbsolutePath="{img_path}" to inspect this image.]')
                elif item.get("type") == "text":
                    parts.append(str(item.get("text") or "").strip())
                elif "content" in item:
                    parts.append(str(item.get("content") or "").strip())
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    rules_in_workspace: bool = False,
    media_dir: str | Path | None = None,
) -> str:
    """Assemble Hermes message history, system prompt, and tool schemas into a canonical prompt.

    ``rules_in_workspace``: the preamble and system prompt already live in the workspace GEMINI.md
    (see ``soul.py``), so the stdin turn carries only tools and conversation.
    """
    try:
        from agent.acp_openai_bridge import render_tool_bridge_sections
        tool_sections = render_tool_bridge_sections(tools, tool_choice)
    except Exception:
        tool_sections = []
        if tools:
            tool_sections.append(
                "Available tools (OpenAI function schema):\n"
                + json.dumps(tools, ensure_ascii=False)
            )

    sections: list[str] = [*([] if rules_in_workspace else _PROMPT_PREAMBLE), *tool_sections]

    valid_messages = [m for m in messages if isinstance(m, dict)]
    last_msg = valid_messages[-1] if valid_messages else None
    last_role = str(last_msg.get("role") or "").strip().lower() if last_msg else ""

    system_parts: list[str] = []
    history_messages: list[dict[str, Any]] = []

    for msg in valid_messages:
        role = str(msg.get("role") or "unknown").strip().lower()
        if role == "system":
            rendered = _render_message_content(msg.get("content"), media_dir=media_dir)
            if rendered:
                system_parts.append(rendered)
        else:
            history_messages.append(msg)

    if system_parts and not rules_in_workspace:
        sections.append("### SYSTEM INSTRUCTIONS (HERMES AGENT):\n" + "\n\n".join(f"System:\n{p}" for p in system_parts))

    _HISTORICAL_TOOL_RETENTION_COUNT = 8
    _HISTORICAL_TOOL_MAX_CHARS = 300
    _TOOL_MAX_CHARS_CAP = 20000

    transcript: list[str] = []
    prior_messages = history_messages[:-1] if last_role == "user" and len(history_messages) > 1 else history_messages

    tool_indices = [
        i for i, msg in enumerate(prior_messages)
        if str(msg.get("role") or "").strip().lower() == "tool"
    ]
    recent_tool_cutoff = (
        set(tool_indices[-_HISTORICAL_TOOL_RETENTION_COUNT:])
        if len(tool_indices) > _HISTORICAL_TOOL_RETENTION_COUNT
        else set(tool_indices)
    )

    for i, message in enumerate(prior_messages):
        role = str(message.get("role") or "unknown").strip().lower()
        rendered_content = _render_message_content(message.get("content"), media_dir=media_dir)

        if role == "tool":
            tool_id = str(message.get("tool_call_id") or message.get("name") or "tool").strip()
            if i not in recent_tool_cutoff and len(rendered_content) > _HISTORICAL_TOOL_MAX_CHARS:
                rendered_content = (
                    rendered_content[:200]
                    + f"\n[... tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-100:]
                )
            elif len(rendered_content) > _TOOL_MAX_CHARS_CAP:
                rendered_content = (
                    rendered_content[:12000]
                    + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-3000:]
                )
            transcript.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue

        if role == "assistant":
            parts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
            if rendered_content:
                parts.append(rendered_content)
            if tool_calls := message.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        parts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if parts:
                transcript.append("Assistant:\n" + "\n".join(parts))
            continue

        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            transcript.append(f"{label}:\n{rendered_content}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    if last_role == "tool":
        sections.append(
            "### LATEST TOOL RESULTS RECEIVED.\n"
            "INSTRUCTION: Evaluate the latest tool results in the transcript above and continue the task. "
            "If more tools are needed, emit <tool_call> tags. If you have enough information, "
            "respond clearly to the user without repeating prior summaries."
        )
    elif last_role == "user" and last_msg is not None:
        user_text = _render_message_content(last_msg.get("content"), media_dir=media_dir)
        sections.append(
            f"### LATEST USER REQUEST TO ANSWER:\nUser:\n{user_text}\n\n"
            "INSTRUCTION: Respond directly and specifically to the LATEST USER REQUEST above. "
            "Do NOT repeat previous architectural summaries, code reviews, or overview boilerplate unless explicitly asked."
        )
    else:
        sections.append("Continue the conversation from the latest message.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _messages_match_prefix(history: Sequence[dict[str, Any]], incoming: Sequence[dict[str, Any]]) -> bool:
    """Return True if incoming messages strictly extend history as a continuation."""
    if not history or len(incoming) <= len(history):
        return False
    for i, h_msg in enumerate(history):
        inc_msg = incoming[i]
        if not isinstance(inc_msg, dict) or not isinstance(h_msg, dict):
            return False
        if inc_msg.get("role") != h_msg.get("role"):
            return False
        if inc_msg.get("content") != h_msg.get("content"):
            return False
        if inc_msg.get("tool_calls") != h_msg.get("tool_calls"):
            return False
    return True


def _format_delta_prompt(new_messages: Sequence[dict[str, Any]], media_dir: str | Path | None = None) -> str:
    """Format only the incremental messages in an ongoing multi-turn interaction."""
    parts: list[str] = []
    last_role = ""
    for msg in new_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        last_role = role
        rendered_content = _render_message_content(msg.get("content"), media_dir=media_dir)
        if role == "tool":
            tool_id = str(msg.get("tool_call_id") or msg.get("name") or "tool").strip()
            if len(rendered_content) > 20000:
                rendered_content = (
                    rendered_content[:12000]
                    + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-3000:]
                )
            parts.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue
        if role == "assistant":
            subparts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
            if rendered_content:
                subparts.append(rendered_content)
            if tool_calls := msg.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        subparts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if subparts:
                parts.append("Assistant:\n" + "\n".join(subparts))
            continue
        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            parts.append(f"{label}:\n{rendered_content}")

    if last_role == "tool":
        parts.append("Continue the conversation from the latest tool result.")
    elif last_role == "user":
        parts.append(
            "Respond directly and specifically to the latest user request above. "
            "Do NOT repeat previous architectural summaries or boilerplate."
        )
    else:
        parts.append("Continue the conversation from the latest message above.")
    return "\n\n".join(s.strip() for s in parts if s and s.strip())


def _longest_tool_call_prefix_match(text: str) -> int:
    """Return the length of the longest suffix of `text` that matches a prefix of `<tool_call>`."""
    for prefix in _TOOL_CALL_PREFIXES:
        if text.endswith(prefix):
            return len(prefix)
    return 0


def _parse_tool_block(block: str) -> tuple[list[Any], str]:
    """Parse tool call blocks, returning (tool_call_deltas, cleaned_text)."""
    calls = []
    cleaned_text = ""
    try:
        from agent.acp_openai_bridge import extract_tool_calls_from_text
        calls, cleaned_text = extract_tool_calls_from_text(block)
    except Exception:
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", block, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                fn = obj.get("function", {})
                call_id = obj.get("id") or "call_1"
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", "{}")
                if not isinstance(fn_args, str):
                    fn_args = json.dumps(fn_args, ensure_ascii=False)
                calls = [
                    SimpleNamespace(
                        id=call_id,
                        type="function",
                        function=SimpleNamespace(name=fn_name, arguments=fn_args),
                    )
                ]
                cleaned_text = block[:m.start()] + block[m.end():]
            except Exception:
                cleaned_text = block
        else:
            cleaned_text = block

    deltas = []
    for i, call in enumerate(calls):
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        args = getattr(fn, "arguments", "{}") if fn else "{}"
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        deltas.append(
            SimpleNamespace(
                index=i,
                id=getattr(call, "id", f"call_{i+1}"),
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
            )
        )
    return deltas, cleaned_text.strip()
