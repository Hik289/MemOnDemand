"""General model gateway for MemOnDemand.

The public runtime exposes one model alias, ``general``. Chat and embedding
requests use configurable general REST endpoints; no cloud vendor,
deployment name, or model family is encoded in the package.

Credentials are loaded from the process environment or an optional ``.env``
file. Secret values are never included in structured results or log messages.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(os.environ.get("MEMONDEMAND_REPO_ROOT", Path.cwd())).resolve()
DEFAULT_ENV_PATH = Path(os.environ.get("MEMONDEMAND_ENV_FILE", PROJECT_ROOT / ".env"))


def load_env(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Load simple ``KEY=VALUE`` entries without replacing existing values."""
    if not env_path.is_file():
        return
    try:
        with env_path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = (part.strip() for part in line.split("=", 1))
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if key:
                    os.environ.setdefault(key, value)
    except OSError:
        return


logger = logging.getLogger("memondemand.api_adapter")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class AliasConfig:
    """Resolved, non-secret configuration for the general model endpoint."""

    alias: str
    provider: str
    model: str
    token_parameter: str = "max_tokens"
    send_temperature: bool = True


class APIError(RuntimeError):
    """Raised when a general model-gateway request cannot be completed."""


_SECRET_ENV_KEYS = (
    "MEMONDEMAND_API_KEY",
    "MEMONDEMAND_EMBED_API_KEY",
)


def _sanitize(text: str) -> str:
    """Remove configured and token-shaped secrets from diagnostic text."""
    rendered = text
    for name in _SECRET_ENV_KEYS:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            rendered = rendered.replace(value, f"<REDACTED:{name}>")
    rendered = re.sub(
        r"(Bearer\s+)[A-Za-z0-9._+/=-]{12,}",
        r"\1<REDACTED>",
        rendered,
        flags=re.IGNORECASE,
    )
    rendered = re.sub(
        r"(api[-_]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._+/=-]{12,}",
        r"\1<REDACTED>",
        rendered,
        flags=re.IGNORECASE,
    )
    return rendered


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def get_alias_config(alias: str = "general") -> AliasConfig:
    """Resolve the sole public model alias."""
    if alias != "general":
        raise ValueError(
            f"Unknown model alias {alias!r}; MemOnDemand exposes only 'general'."
        )
    token_parameter = os.environ.get(
        "MEMONDEMAND_API_TOKEN_PARAMETER", "max_tokens"
    ).strip()
    if token_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise RuntimeError(
            "MEMONDEMAND_API_TOKEN_PARAMETER must be 'max_tokens' or "
            "'max_completion_tokens'."
        )
    return AliasConfig(
        alias="general",
        provider="general",
        model=os.environ.get("MEMONDEMAND_API_MODEL", "").strip(),
        token_parameter=token_parameter,
        send_temperature=_bool_env("MEMONDEMAND_API_SEND_TEMPERATURE", True),
    )


def _endpoint(base_url: str, resource: str) -> str:
    base = base_url.rstrip("/")
    suffix = f"/{resource}"
    return base if base.endswith(suffix) else f"{base}{suffix}"


def _http_post(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("General API response must be a JSON object")
    return parsed


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _usage(raw: Dict[str, Any]) -> Dict[str, int]:
    usage = raw.get("usage") or {}
    return {
        "input_tokens": int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        ),
        "output_tokens": int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        ),
    }


def _configured_prices() -> Dict[str, float]:
    try:
        return {
            "input": float(
                os.environ.get("MEMONDEMAND_API_INPUT_COST_PER_MILLION", "0")
            ),
            "output": float(
                os.environ.get("MEMONDEMAND_API_OUTPUT_COST_PER_MILLION", "0")
            ),
        }
    except ValueError as exc:
        raise RuntimeError("General API cost settings must be numeric") from exc


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost from deployment-configured per-million-token prices."""
    del model
    prices = _configured_prices()
    return (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000.0


def _retry_delay(attempt: int, backoff_base: float, rate_limited: bool) -> float:
    if rate_limited:
        return min(30.0, max(1.0, backoff_base * 4))
    return min(30.0, backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.25))


def _http_error(error: urllib.error.HTTPError) -> APIError:
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return APIError(f"HTTP {error.code}: {_sanitize(body)[:200]}")


def call(
    alias: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 5,
    backoff_base: float = 1.0,
) -> Dict[str, Any]:
    """Send a chat-completions request through the general model gateway."""
    cfg = get_alias_config(alias)
    base_url = _require_env("MEMONDEMAND_API_BASE_URL")
    api_key = _require_env("MEMONDEMAND_API_KEY")
    model = cfg.model or _require_env("MEMONDEMAND_API_MODEL")
    if not messages or any(not isinstance(item, dict) for item in messages):
        raise ValueError("messages must be a non-empty list of message objects")
    if max_tokens <= 0 or max_retries <= 0:
        raise ValueError("max_tokens and max_retries must be positive")

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        cfg.token_parameter: max_tokens,
    }
    if cfg.send_temperature:
        body["temperature"] = temperature

    last_error: Optional[APIError] = None
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            raw = _http_post(
                _endpoint(base_url, "chat/completions"),
                _headers(api_key),
                body,
                timeout,
            )
            choices = raw.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise RuntimeError("General chat response is missing 'choices'")
            message = choices[0].get("message") or {}
            text = _content_text(message.get("content"))
            usage = _usage(raw)
            latency_ms = (time.perf_counter() - started) * 1000.0
            cost = estimate_cost(model, usage["input_tokens"], usage["output_tokens"])
            logger.info(
                "call ok alias=general model=%s input_tokens=%d "
                "output_tokens=%d latency_ms=%.1f attempt=%d",
                model,
                usage["input_tokens"],
                usage["output_tokens"],
                latency_ms,
                attempt,
            )
            return {
                "text": text,
                "usage": usage,
                "latency_ms": latency_ms,
                "model": model,
                "provider": "general",
                "alias": "general",
                "cost_usd": cost,
            }
        except urllib.error.HTTPError as exc:
            last_error = _http_error(exc)
            if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                raise last_error
            rate_limited = exc.code == 429
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            RuntimeError,
        ) as exc:
            last_error = APIError(f"{type(exc).__name__}: {_sanitize(str(exc))[:200]}")
            rate_limited = "429" in str(exc)

        logger.warning(
            "call failed alias=general model=%s attempt=%d error=%s",
            model,
            attempt,
            _sanitize(str(last_error)),
        )
        if attempt < max_retries:
            time.sleep(_retry_delay(attempt, backoff_base, rate_limited))

    raise last_error or APIError("General model request failed")


def embed(
    texts: Sequence[str],
    *,
    timeout: float = 60.0,
    max_retries: int = 5,
    backoff_base: float = 1.0,
) -> Dict[str, Any]:
    """Embed text through the configured general embeddings endpoint."""
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    values = [str(text) for text in texts]
    if not values:
        return {
            "vectors": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": os.environ.get("MEMONDEMAND_EMBED_API_MODEL", ""),
            "provider": "general",
        }
    base_url = os.environ.get("MEMONDEMAND_EMBED_API_BASE_URL", "").strip()
    if not base_url:
        base_url = _require_env("MEMONDEMAND_API_BASE_URL")
    api_key = os.environ.get("MEMONDEMAND_EMBED_API_KEY", "").strip()
    if not api_key:
        api_key = _require_env("MEMONDEMAND_API_KEY")
    model = _require_env("MEMONDEMAND_EMBED_API_MODEL")
    body = {"model": model, "input": values}

    last_error: Optional[APIError] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = _http_post(
                _endpoint(base_url, "embeddings"),
                _headers(api_key),
                body,
                timeout,
            )
            data = raw.get("data") or []
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            vectors = [item.get("embedding") for item in ordered]
            if len(vectors) != len(values) or any(not isinstance(v, list) for v in vectors):
                raise RuntimeError("General embedding response has invalid vector data")
            return {
                "vectors": vectors,
                "usage": _usage(raw),
                "model": model,
                "provider": "general",
            }
        except urllib.error.HTTPError as exc:
            last_error = _http_error(exc)
            if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                raise last_error
            rate_limited = exc.code == 429
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            RuntimeError,
        ) as exc:
            last_error = APIError(f"{type(exc).__name__}: {_sanitize(str(exc))[:200]}")
            rate_limited = "429" in str(exc)
        if attempt < max_retries:
            time.sleep(_retry_delay(attempt, backoff_base, rate_limited))

    raise last_error or APIError("General embedding request failed")


load_env()


if __name__ == "__main__":
    if not all(
        os.environ.get(name)
        for name in (
            "MEMONDEMAND_API_BASE_URL",
            "MEMONDEMAND_API_KEY",
            "MEMONDEMAND_API_MODEL",
        )
    ):
        print("general smoke skipped: configure MEMONDEMAND_API_* first")
    else:
        result = call(
            "general",
            [{"role": "user", "content": "Reply with exactly: TEST_GENERAL"}],
            max_tokens=32,
            max_retries=1,
        )
        print(
            "general endpoint ok: "
            f"model={result['model']} latency_ms={result['latency_ms']:.0f}"
        )
