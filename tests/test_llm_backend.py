"""
Tests for LLM backend abstraction (Ollama vs OpenAI-compatible API support).

Tests configuration resolution, request building, and response normalisation
for both Ollama and OpenAI-compatible backends.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from jarvis.config import (
    resolve_llm_backend,
    get_llm_chat_config,
    get_default_config,
    load_settings,
)


# ---------------------------------------------------------------------------
# resolve_llm_backend
# ---------------------------------------------------------------------------

class TestResolveLlmBackend:
    """Tests for resolve_llm_backend() platform resolution."""

    def test_ollama_returns_ollama(self):
        assert resolve_llm_backend("ollama") == "ollama"

    def test_openai_returns_openai(self):
        assert resolve_llm_backend("openai") == "openai"

    def test_invalid_value_falls_back_to_ollama(self):
        assert resolve_llm_backend("invalid") == "ollama"

    @patch("jarvis.config._is_apple_silicon", return_value=True)
    def test_auto_on_apple_silicon_returns_openai(self, _mock):
        assert resolve_llm_backend("auto") == "openai"

    @patch("jarvis.config._is_apple_silicon", return_value=False)
    def test_auto_on_non_apple_silicon_returns_ollama(self, _mock):
        assert resolve_llm_backend("auto") == "ollama"


# ---------------------------------------------------------------------------
# get_llm_chat_config
# ---------------------------------------------------------------------------

class TestGetLlmChatConfig:
    """Tests for get_llm_chat_config() returning resolved (url, model, format)."""

    def test_ollama_backend_returns_ollama_config(self, mock_config):
        mock_config.llm_backend = "ollama"
        base_url, chat_model, api_format = get_llm_chat_config(mock_config)
        assert base_url == mock_config.ollama_base_url
        assert chat_model == mock_config.ollama_chat_model
        assert api_format == "ollama"

    def test_openai_backend_returns_openai_config(self, mock_config):
        mock_config.llm_backend = "openai"
        mock_config.openai_base_url = "http://localhost:9999"
        mock_config.openai_chat_model = "my-mlx-model"
        base_url, chat_model, api_format = get_llm_chat_config(mock_config)
        assert base_url == "http://localhost:9999"
        assert chat_model == "my-mlx-model"
        assert api_format == "openai"

    @patch("jarvis.config._is_apple_silicon", return_value=True)
    def test_auto_on_apple_silicon_returns_openai_config(self, _mock, mock_config):
        mock_config.llm_backend = "auto"
        _, _, api_format = get_llm_chat_config(mock_config)
        assert api_format == "openai"

    @patch("jarvis.config._is_apple_silicon", return_value=False)
    def test_auto_on_non_apple_silicon_returns_ollama_config(self, _mock, mock_config):
        mock_config.llm_backend = "auto"
        _, _, api_format = get_llm_chat_config(mock_config)
        assert api_format == "ollama"


# ---------------------------------------------------------------------------
# Default config includes new fields
# ---------------------------------------------------------------------------

class TestDefaultConfigBackendFields:
    """Tests that default config includes the new backend fields."""

    def test_default_config_has_llm_backend(self):
        config = get_default_config()
        assert "llm_backend" in config
        assert config["llm_backend"] == "auto"

    def test_default_config_has_openai_base_url(self):
        config = get_default_config()
        assert "openai_base_url" in config
        assert config["openai_base_url"] == "http://127.0.0.1:8080"

    def test_default_config_has_openai_chat_model(self):
        config = get_default_config()
        assert "openai_chat_model" in config
        assert config["openai_chat_model"] == ""


# ---------------------------------------------------------------------------
# LLM gateway: Ollama format (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestLlmOllamaFormat:
    """Tests that Ollama API format works as before."""

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_ollama_uses_api_chat(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "hello"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = call_llm_direct(
            base_url="http://localhost:11434",
            chat_model="test-model",
            system_prompt="sys",
            user_content="hi",
            api_format="ollama",
        )

        assert result == "hello"
        call_url = mock_post.call_args[0][0]
        assert "/api/chat" in call_url

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_ollama_includes_num_ctx(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        call_llm_direct(
            base_url="http://localhost:11434",
            chat_model="test",
            system_prompt="s",
            user_content="u",
            api_format="ollama",
        )

        payload = mock_post.call_args[1]["json"]
        assert "options" in payload
        assert payload["options"]["num_ctx"] == 4096

    @patch("jarvis.llm.requests.post")
    def test_chat_with_messages_ollama_format(self, mock_post):
        from jarvis.llm import chat_with_messages

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "response", "tool_calls": []}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = chat_with_messages(
            base_url="http://localhost:11434",
            chat_model="test",
            messages=[{"role": "user", "content": "hi"}],
            api_format="ollama",
        )

        assert result is not None
        assert result["message"]["content"] == "response"


# ---------------------------------------------------------------------------
# LLM gateway: OpenAI-compatible format
# ---------------------------------------------------------------------------

class TestLlmOpenaiFormat:
    """Tests for OpenAI-compatible API format."""

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_openai_uses_v1_endpoint(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello from mlx"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = call_llm_direct(
            base_url="http://localhost:8080",
            chat_model="mlx-model",
            system_prompt="sys",
            user_content="hi",
            api_format="openai",
        )

        assert result == "hello from mlx"
        call_url = mock_post.call_args[0][0]
        assert "/v1/chat/completions" in call_url

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_openai_excludes_num_ctx(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        call_llm_direct(
            base_url="http://localhost:8080",
            chat_model="mlx",
            system_prompt="s",
            user_content="u",
            api_format="openai",
        )

        payload = mock_post.call_args[1]["json"]
        assert "options" not in payload

    @patch("jarvis.llm.requests.post")
    def test_chat_with_messages_openai_normalises_to_ollama_shape(self, mock_post):
        """OpenAI responses should be normalised to Ollama's internal format."""
        from jarvis.llm import chat_with_messages

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "normalised response",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "webSearch", "arguments": '{"query": "test"}'},
                        }
                    ],
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = chat_with_messages(
            base_url="http://localhost:8080",
            chat_model="mlx",
            messages=[{"role": "user", "content": "search for test"}],
            api_format="openai",
        )

        # Should be normalised to Ollama shape: {"message": {...}}
        assert result is not None
        assert "message" in result
        assert result["message"]["content"] == "normalised response"
        assert len(result["message"]["tool_calls"]) == 1
        tc = result["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "webSearch"
        # Arguments should be parsed from JSON string to dict
        assert isinstance(tc["function"]["arguments"], dict)

    @patch("jarvis.llm.requests.post")
    def test_chat_with_messages_openai_empty_model_omits_field(self, mock_post):
        """When openai_chat_model is empty, the model field should be omitted."""
        from jarvis.llm import chat_with_messages

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        chat_with_messages(
            base_url="http://localhost:8080",
            chat_model="",
            messages=[{"role": "user", "content": "hi"}],
            api_format="openai",
        )

        payload = mock_post.call_args[1]["json"]
        assert "model" not in payload

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_default_api_format_is_ollama(self, mock_post):
        """Default api_format should be 'ollama' for backwards compatibility."""
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        call_llm_direct(
            base_url="http://localhost:11434",
            chat_model="test",
            system_prompt="s",
            user_content="u",
        )

        call_url = mock_post.call_args[0][0]
        assert "/api/chat" in call_url


# ---------------------------------------------------------------------------
# OpenAI streaming
# ---------------------------------------------------------------------------

class TestLlmOpenaiStreaming:
    """Tests for OpenAI-compatible streaming format (SSE)."""

    @patch("jarvis.llm.requests.post")
    def test_call_llm_streaming_openai_parses_sse(self, mock_post):
        from jarvis.llm import call_llm_streaming

        # Simulate SSE response lines
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b'data: [DONE]',
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_resp

        tokens = []
        result = call_llm_streaming(
            base_url="http://localhost:8080",
            chat_model="mlx",
            system_prompt="sys",
            user_content="hi",
            on_token=lambda t: tokens.append(t),
            api_format="openai",
        )

        assert result == "Hello world"
        assert tokens == ["Hello", " world"]
        call_url = mock_post.call_args[0][0]
        assert "/v1/chat/completions" in call_url
