"""Direct LLM interaction utilities without extra features like temporal context.

Supports two API formats:
- "ollama" (default): Ollama's /api/chat endpoint
- "openai": OpenAI-compatible /v1/chat/completions endpoint (MLX, LM Studio, vLLM, etc.)

OpenAI responses are normalised to Ollama's internal format so callers
don't need to handle format differences.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Callable
import requests
import json


class ToolsNotSupportedError(Exception):
    """Raised when the model returns HTTP 400 because native tool calling is not supported."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chat_url(base_url: str, api_format: str) -> str:
    """Build the chat endpoint URL for the given format."""
    base = base_url.rstrip("/")
    if api_format == "openai":
        return f"{base}/v1/chat/completions"
    return f"{base}/api/chat"


def _build_payload(
    chat_model: str,
    messages: List[Dict[str, str]],
    api_format: str,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the request payload for the given API format."""
    payload: Dict[str, Any] = {
        "messages": messages,
        "stream": stream,
    }

    # Model field: omit when empty (OpenAI servers infer from loaded model)
    if chat_model:
        payload["model"] = chat_model

    if api_format == "ollama":
        # Ollama-specific options
        payload["options"] = {"num_ctx": 4096}
        if extra_options and isinstance(extra_options, dict):
            payload["options"].update(extra_options)
        # Disable "thinking mode" for qwen3 models (causes very slow responses)
        if chat_model.startswith("qwen3"):
            payload["think"] = False
    # OpenAI format: no options/num_ctx, no think toggle

    if tools and isinstance(tools, list) and len(tools) > 0:
        payload["tools"] = tools

    return payload


def _normalise_openai_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise an OpenAI-compatible response to Ollama's internal format.

    Ollama format:  {"message": {"content": "...", "tool_calls": [...]}}
    OpenAI format:  {"choices": [{"message": {"content": "...", "tool_calls": [...]}}]}

    Tool call arguments in OpenAI format are JSON strings; we parse them to dicts
    to match Ollama's native format.
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return data  # Not recognisable — return as-is

    msg = choices[0].get("message", {})
    normalised_msg: Dict[str, Any] = {"content": msg.get("content", "")}

    # Normalise tool_calls: OpenAI sends arguments as JSON string, Ollama as dict
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        normalised_tcs = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            # Parse JSON string arguments to dict
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            normalised_tcs.append({
                "id": tc.get("id", ""),
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            })
        normalised_msg["tool_calls"] = normalised_tcs

    return {"message": normalised_msg}


def _parse_openai_sse_line(line: bytes) -> Optional[str]:
    """Extract content token from an OpenAI SSE data line.

    SSE format: b'data: {"choices":[{"delta":{"content":"token"}}]}'
    Returns the content string or None if not a content line.
    """
    decoded = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
    if not decoded.startswith("data: "):
        return None
    payload_str = decoded[len("data: "):]
    if payload_str.strip() == "[DONE]":
        return None
    try:
        data = json.loads(payload_str)
        choices = data.get("choices", [])
        if choices and isinstance(choices, list):
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                return content
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm_direct(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    timeout_sec: float = 10.0,
    api_format: str = "ollama",
) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload = _build_payload(chat_model, messages, api_format)
    url = _chat_url(base_url, api_format)

    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            if api_format == "openai":
                data = _normalise_openai_response(data)
            content = extract_text_from_response(data)
            if isinstance(content, str) and content.strip():
                return content
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None

    return None


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    api_format: str = "ollama",
) -> Optional[str]:
    """
    Streaming LLM call that invokes on_token callback for each token received.

    Supports both Ollama NDJSON streaming and OpenAI SSE streaming.

    Returns:
        Complete response text, or None on error
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload = _build_payload(chat_model, messages, api_format, stream=True)
    url = _chat_url(base_url, api_format)

    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec, stream=True)
        resp.raise_for_status()

        full_response = []
        for line in resp.iter_lines():
            if not line:
                continue

            if api_format == "openai":
                token = _parse_openai_sse_line(line)
                if token:
                    full_response.append(token)
                    if on_token:
                        on_token(token)
            else:
                # Ollama NDJSON format
                try:
                    data = json.loads(line)
                    if "message" in data and isinstance(data["message"], dict):
                        content = data["message"].get("content", "")
                        if content:
                            full_response.append(content)
                            if on_token:
                                on_token(content)
                except json.JSONDecodeError:
                    continue

        result = "".join(full_response)
        return result if result.strip() else None

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from LLM response - supports multiple response formats."""
    # Preferred: Ollama chat non-stream format
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content

    # Fallback: OpenAI-style format
    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content

    # Another fallback: direct "content" field
    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content

    return None


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    api_format: str = "ollama",
) -> Optional[Dict[str, Any]]:
    """
    Send an arbitrary messages array to the LLM and return the raw response JSON.

    Responses are normalised to Ollama's internal format regardless of backend,
    so callers always see: {"message": {"content": "...", "tool_calls": [...]}}

    Args:
        base_url: Server base URL
        chat_model: Model name (empty string = server decides)
        messages: Conversation messages
        timeout_sec: Request timeout
        extra_options: Additional model options (Ollama only)
        tools: Optional list of tools in OpenAI-compatible JSON schema format
        api_format: "ollama" or "openai"

    Returns the parsed JSON response dict on success, or None on error/timeout.
    """
    payload = _build_payload(
        chat_model, messages, api_format,
        tools=tools, extra_options=extra_options,
    )
    url = _chat_url(base_url, api_format)

    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if api_format == "openai":
                return _normalise_openai_response(data)
            return data
    except requests.exceptions.Timeout:
        print("  ⏱️ LLM request timed out", flush=True)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ LLM connection error: {e}", flush=True)
        return None
    except requests.exceptions.HTTPError as e:
        # Raise a specific error when the model rejects the tools parameter (HTTP 400).
        # This lets the caller fall back to text-based tool calling automatically.
        if e.response is not None and e.response.status_code == 400 and tools:
            raise ToolsNotSupportedError(
                f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
            )
        print(f"  ❌ LLM HTTP error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  ❌ LLM error: {e}", flush=True)
        return None

    return None
