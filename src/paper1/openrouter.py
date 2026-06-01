"""Async OpenRouter client with retries and a Protocol for mock injection in tests.

Single API gateway for all three agents. Pinned model IDs in config/models.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class CompletionResult:
    """One LLM response, plus token-usage metadata for cost tracking."""

    text: str
    tokens_in: int
    tokens_out: int
    model_id: str
    raw: dict[str, Any]


class OpenRouterClientProtocol(Protocol):
    """Interface every client (real or mock) must satisfy."""

    async def complete(
        self,
        *,
        model_id: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> CompletionResult:
        ...

    async def aclose(self) -> None:
        ...


class OpenRouterClient:
    """Production async client for OpenRouter.

    Uses httpx.AsyncClient under the hood. Retries with exponential backoff on
    transient errors (5xx, network errors, rate-limit 429).
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 90.0,
        max_retries: int = 4,
        retry_backoff_s: float = 2.0,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_s,
        )

    async def complete(
        self,
        *,
        model_id: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> CompletionResult:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._retry_backoff_s, min=1, max=30),
            retry=retry_if_exception_type(_RetryableError),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise _RetryableError(
                        f"HTTP {resp.status_code} from OpenRouter for model '{model_id}': "
                        f"{resp.text[:300]}"
                    )
                if resp.status_code >= 400:
                    # Non-retryable client error — surface the model name and the body
                    body_excerpt: str
                    try:
                        body_excerpt = json.dumps(resp.json())[:500]
                    except Exception:
                        body_excerpt = resp.text[:500]
                    raise OpenRouterAPIError(
                        f"OpenRouter returned HTTP {resp.status_code} for model "
                        f"'{model_id}'. Response body: {body_excerpt}"
                    )
                body = resp.json()
                if "choices" not in body or not body["choices"]:
                    raise OpenRouterAPIError(
                        f"OpenRouter response for model '{model_id}' had no choices. "
                        f"Body: {json.dumps(body)[:500]}"
                    )
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                return CompletionResult(
                    text=text,
                    tokens_in=int(usage.get("prompt_tokens", 0)),
                    tokens_out=int(usage.get("completion_tokens", 0)),
                    model_id=model_id,
                    raw=body,
                )

        raise RuntimeError("Retry loop exited unexpectedly")  # pragma: no cover

    async def aclose(self) -> None:
        await self._client.aclose()


class _RetryableError(Exception):
    """Marker for tenacity-retried errors."""


class OpenRouterAPIError(Exception):
    """Non-retryable OpenRouter API error with model context."""


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON response, tolerating common LLM quirks.

    Some models wrap JSON in ```json ... ``` fences or add a sentence before
    the JSON object. Strip those and fall back to extracting the first {...}.

    For fine-tuned models that loop / ramble past max_tokens producing a
    truncated `{"contributions": [...]}` array, a final fallback recovers the
    complete contribution objects from the partial array.
    """

    text = text.strip()

    # Strip ```json or ``` fences
    if text.startswith("```"):
        # remove fence
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: first balanced {...} block (find first '{', scan to its matching '}')
    balanced = _first_balanced_json(text)
    if balanced is not None:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

    # Fallback 2: from-first-{ to-last-}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback 3: truncated top-level array (`contributions`, `verdicts`) — recover complete objects
    for key in ("contributions", "verdicts"):
        recovered = _recover_truncated_array(text, key)
        if recovered is not None:
            return recovered

    raise ValueError(f"Could not parse JSON from response: {text[:200]!r}")


def _first_balanced_json(text: str) -> str | None:
    """Return the first balanced `{...}` substring (respects strings/escapes)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _recover_truncated_array(text: str, key: str) -> dict[str, Any] | None:
    """Extract complete top-level objects from a truncated
    `{"<key>": [ {...}, {...}, <truncated>` payload.

    Used by extractor (`contributions`) and critic (`verdicts`) when the
    model rambles past max_tokens and emits a partial array.

    Returns `{"<key>": [...complete objects...]}` or None if recovery
    isn't applicable.
    """
    # Find the array opening
    key_idx = text.find(f'"{key}"')
    if key_idx == -1:
        return None
    bracket_idx = text.find("[", key_idx)
    if bracket_idx == -1:
        return None

    # Walk the array, collecting complete top-level {...} objects
    objects: list[dict[str, Any]] = []
    i = bracket_idx + 1
    n = len(text)
    while i < n:
        # Skip whitespace and commas
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        if text[i] == "]":
            break
        if text[i] != "{":
            # Unexpected token — stop
            break
        # Scan a balanced object
        depth = 0
        in_str = False
        esc = False
        start = i
        end = -1
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1
        if end == -1:
            # Truncated object — stop recovery here
            break
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            break
        objects.append(obj)
        i = end

    if not objects:
        return None
    return {key: objects}
