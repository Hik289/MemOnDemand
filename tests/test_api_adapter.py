from __future__ import annotations

import json

import pytest

from memondemand.core import api_adapter


def test_general_alias_is_exclusive() -> None:
    with pytest.raises(ValueError, match="exposes only 'general'"):
        api_adapter.get_alias_config("default")


def test_general_gateway_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMONDEMAND_API_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("MEMONDEMAND_API_KEY", "test-secret-value")
    monkeypatch.setenv("MEMONDEMAND_API_MODEL", "chat-model")
    captured: dict = {}

    def fake_post(url, headers, body, timeout):
        captured.update(
            url=url,
            headers=headers,
            body=body,
            timeout=timeout,
        )
        return {
            "choices": [{"message": {"content": "grounded answer"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }

    monkeypatch.setattr(api_adapter, "_http_post", fake_post)

    result = api_adapter.call(
        "general",
        [{"role": "user", "content": "question"}],
        max_tokens=64,
        temperature=0.1,
        max_retries=1,
    )

    assert captured["url"] == "https://gateway.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-secret-value"
    assert captured["body"]["model"] == "chat-model"
    assert captured["body"]["max_tokens"] == 64
    assert result["text"] == "grounded answer"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert result["provider"] == "general"


def test_general_embedding_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMONDEMAND_API_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("MEMONDEMAND_API_KEY", "chat-secret")
    monkeypatch.setenv(
        "MEMONDEMAND_EMBED_API_BASE_URL", "https://embedding.example/v1"
    )
    monkeypatch.setenv("MEMONDEMAND_EMBED_API_KEY", "embedding-secret")
    monkeypatch.setenv("MEMONDEMAND_EMBED_API_MODEL", "embedding-model")
    captured: dict = {}

    def fake_post(url, headers, body, timeout):
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 3},
        }

    monkeypatch.setattr(api_adapter, "_http_post", fake_post)
    result = api_adapter.embed(["first", "second"], max_retries=1)

    assert captured["url"] == "https://embedding.example/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer embedding-secret"
    assert captured["body"] == {
        "model": "embedding-model",
        "input": ["first", "second"],
    }
    assert result["vectors"] == [[1.0, 0.0], [0.0, 1.0]]
    assert result["provider"] == "general"


def test_general_gateway_supports_alternate_token_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMONDEMAND_API_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("MEMONDEMAND_API_KEY", "test-secret-value")
    monkeypatch.setenv("MEMONDEMAND_API_MODEL", "chat-model")
    monkeypatch.setenv(
        "MEMONDEMAND_API_TOKEN_PARAMETER", "max_completion_tokens"
    )
    monkeypatch.setenv("MEMONDEMAND_API_SEND_TEMPERATURE", "false")
    captured: dict = {}

    def fake_post(url, headers, body, timeout):
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(api_adapter, "_http_post", fake_post)
    api_adapter.call(
        "general",
        [{"role": "user", "content": "question"}],
        max_tokens=32,
        temperature=0.7,
        max_retries=1,
    )

    assert captured["body"]["max_completion_tokens"] == 32
    assert "max_tokens" not in captured["body"]
    assert "temperature" not in captured["body"]


def test_gateway_errors_redact_configured_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-token-value"
    monkeypatch.setenv("MEMONDEMAND_API_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("MEMONDEMAND_API_KEY", secret)
    monkeypatch.setenv("MEMONDEMAND_API_MODEL", "chat-model")

    def fail_post(*_args, **_kwargs):
        raise RuntimeError(f"upstream rejected {secret}")

    monkeypatch.setattr(api_adapter, "_http_post", fail_post)

    with pytest.raises(api_adapter.APIError) as exc:
        api_adapter.call(
            "general",
            [{"role": "user", "content": "question"}],
            max_retries=1,
        )

    rendered = str(exc.value)
    assert secret not in rendered
    assert "REDACTED" in rendered


def test_gateway_report_is_json_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMONDEMAND_API_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("MEMONDEMAND_API_KEY", "test-secret-value")
    monkeypatch.setenv("MEMONDEMAND_API_MODEL", "chat-model")
    monkeypatch.setattr(
        api_adapter,
        "_http_post",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        },
    )

    result = api_adapter.call(
        "general",
        [{"role": "user", "content": "question"}],
        max_retries=1,
    )

    assert json.loads(json.dumps(result))["alias"] == "general"
