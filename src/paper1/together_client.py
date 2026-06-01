"""Together AI inference client implementing OpenRouterClientProtocol.

Routes chat-completions to Together's OpenAI-compatible endpoint
(https://api.together.xyz/v1/chat/completions). Same retry / error-handling
shape as the existing OpenRouterClient so any caller that expected
OpenRouterClientProtocol can use TogetherClient transparently.

Used for the fine-tuned SciREX extractor (PaperCardv5 / FT-3). Critic and
Consolidator continue to call OpenRouter — see RoutingClient below for the
combined wrapper.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from paper1.openrouter import (
    CompletionResult,
    OpenRouterAPIError,
    OpenRouterClientProtocol,
    _RetryableError,
)


def _first_balanced_json(text: str) -> str:
    """Return the substring of `text` from the first '{' through its matching
    '}', respecting nested braces and JSON string escapes. If no balanced
    object is found, return `text` unchanged.

    Used for fine-tuned-model outputs that loop and emit multiple
    back-to-back JSON objects (a known artifact of LoRA-tuned chat models).
    """
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
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
    return text  # unbalanced; let the parser fail with a clear error


class TogetherClient:
    """Together AI client. Same protocol as OpenRouterClient."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.together.xyz/v1",
        timeout_s: float = 120.0,
        max_retries: int = 4,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
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
        # Fine-tuned Llama-3.1 chat models on Together sometimes continue past
        # the assistant turn (emitting `<|im_end|>` then a fake follow-up turn).
        # Pass these as stop sequences AND post-trim the response below to be safe.
        stop_tokens = ["<|im_end|>", "<|eot_id|>", "<|im_start|>"]
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop_tokens,
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
                        f"HTTP {resp.status_code} from Together for model '{model_id}': "
                        f"{resp.text[:300]}"
                    )
                if resp.status_code >= 400:
                    try:
                        body_excerpt = json.dumps(resp.json())[:500]
                    except Exception:
                        body_excerpt = resp.text[:500]
                    raise OpenRouterAPIError(
                        f"Together returned HTTP {resp.status_code} for model "
                        f"'{model_id}'. Response body: {body_excerpt}"
                    )
                body = resp.json()
                if "choices" not in body or not body["choices"]:
                    raise OpenRouterAPIError(
                        f"Together response for model '{model_id}' had no choices. "
                        f"Body: {json.dumps(body)[:500]}"
                    )
                text = body["choices"][0]["message"]["content"]
                # Defensive trim — even if `stop` was honoured, some servers
                # echo the stop token back. Cut at the first chat-template
                # marker so downstream JSON parsing doesn't see a second turn.
                for tok in stop_tokens:
                    idx = text.find(tok)
                    if idx >= 0:
                        text = text[:idx]
                text = text.rstrip()
                # If the FT model has looped past the first record (multiple
                # back-to-back JSON objects), keep only the first balanced
                # `{...}` block. This is robust whether the model emitted
                # one object or many.
                text = _first_balanced_json(text)
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


class RoutingClient:
    """Multi-provider router that satisfies OpenRouterClientProtocol.

    Routes a request based on `model_id`: if `model_id` is in `together_models`,
    the call goes through the TogetherClient; otherwise it goes through the
    underlying OpenRouterClient. This lets us keep the FT extractor on Together
    while Critic and Consolidator continue to use the standard OpenRouter
    Llama 3.3 70B endpoint.
    """

    def __init__(
        self,
        openrouter: OpenRouterClientProtocol,
        together: TogetherClient,
        *,
        together_models: set[str],
    ) -> None:
        self._or = openrouter
        self._tg = together
        self._tg_models = set(together_models)

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
        client = self._tg if model_id in self._tg_models else self._or
        return await client.complete(
            model_id=model_id,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    async def aclose(self) -> None:
        await self._or.aclose()
        await self._tg.aclose()
