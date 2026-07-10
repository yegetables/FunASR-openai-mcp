"""
FunASR MCP Server (HTTP Proxy)

Uses the official MCP Python SDK (FastMCP) to expose a transcription tool
that proxies to the running FunASR HTTP server.

Env:
    FUNASR_URL  — base URL (default http://localhost:8221)
"""

import json
import os
import urllib.request
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("funasr")

FUNASR_URL = os.environ.get("FUNASR_URL", "http://localhost:8221")

# ── Model knowledge base ──────────────────────────────────────────────
#
# ASR Models (all available via this MCP tool):
#   sensevoice    : 5 lang (zh/en/ja/ko/yue), 170x realtime, emotion tags
#   paraformer    : zh+en, 120x realtime, needs punc_model, mature Chinese ASR
#   paraformer-en : en only, 120x realtime
#   fun-asr-nano  : 31 lang, 17x realtime, LLM-based, best for hard cases
#   qwen3-asr     : 52 lang, large model (1.7B), highest accuracy, needs GPU
#
# Auto-select rules:
#   1. If user specifies a model → use it
#   2. If language is ja/ko/yue → sensevoice (only option)
#   3. If language is en only → paraformer-en
#   4. If highest accuracy requested / don't care about speed → qwen3-asr
#   5. If 31+ languages / dialects / lyrics → fun-asr-nano
#   6. Default → sensevoice (fastest, most features)
#
# OpenAI API server only exposes: sensevoice, paraformer, paraformer-en, fun-asr-nano
# MCP tool exposes ALL models (including qwen3-asr)


def auto_select_model(user_model: str, language: str) -> str:
    """Automatically select the best model based on context."""
    if user_model and user_model != "auto":
        return user_model

    lang = (language or "auto").lower()

    # Japanese, Korean, Cantonese → only sensevoice supports these
    if lang in ("ja", "japanese", "ko", "korean", "yue", "cantonese"):
        return "sensevoice"

    # English only → paraformer-en is optimized for it
    if lang in ("en", "english"):
        return "paraformer-en"

    # Default → sensevoice (fastest, most features, built-in emotion)
    return "sensevoice"


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    model: str = "auto",
    language: str = "auto",
    spk: bool = False,
) -> str:
    """Transcribe speech audio to text via the running FunASR server.

    Supports 50+ languages with automatic model selection.
    Input: local file path to audio (wav, mp3, flac, m4a, ogg).

    Model auto-selection (when model="auto"):
      - Japanese/Korean/Cantonese → sensevoice
      - English only → paraformer-en
      - Default → sensevoice (fastest, emotion detection)

    You can also explicitly choose:
      - sensevoice: fast, 5 languages, emotion tags
      - paraformer: Chinese production-grade
      - paraformer-en: English optimized
      - fun-asr-nano: 31 languages, dialects, lyrics, timestamps
      - qwen3-asr: 52 languages, highest accuracy, large model (slow)

    Args:
        audio_path: Path to audio file
        model: Model name or "auto" for automatic selection
        language: Language hint (e.g. "zh", "en", "ja", "auto")
        spk: Enable speaker diarization (slower, uses more GPU memory)
    """
    audio_path = os.path.expanduser(audio_path)

    if not os.path.exists(audio_path):
        return f"Error: file not found: {audio_path}"

    resolved_model = auto_select_model(model, language)

    boundary = "----FunASRMCPBoundary"
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        file_data = f.read()

    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    parts.append(file_data)
    parts.append(b"\r\n")

    # model field
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    parts.append(resolved_model.encode())
    parts.append(b"\r\n")

    # language field
    if language and language != "auto":
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(b'Content-Disposition: form-data; name="language"\r\n\r\n')
        parts.append(language.encode())
        parts.append(b"\r\n")

    # spk field
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="spk"\r\n\r\n')
    parts.append(b"true" if spk else b"false")
    parts.append(b"\r\n")

    # response_format
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
    parts.append(b"verbose_json")
    parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)

    req = urllib.request.Request(
        f"{FUNASR_URL}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode())
            if isinstance(result, dict):
                text = result.get("text", "")
                segments = result.get("segments", [])
                model_used = result.get("model", resolved_model)

                lines = [f"[model: {model_used}] {text}"]
                if segments:
                    lines.append("\n--- Segments ---")
                    for seg in segments:
                        spk_id = seg.get("speaker")
                        spk_label = f" [Speaker {spk_id}]" if spk_id is not None else ""
                        lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]{spk_label} {seg['text']}")
                return "\n".join(lines)
            return str(result)
    except Exception as e:
        return f"Transcription error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
