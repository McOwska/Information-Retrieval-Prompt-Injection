from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from openai import InternalServerError, RateLimitError, APIConnectionError


load_dotenv()


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough estimate: ~4 chars per token for English text."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


def _get_client() -> OpenAI:
    api_key = (
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("BERGET_API_KEY", "").strip()
        or os.getenv("KIMI_API_KEY", "").strip()
    )

    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY, OPENAI_API_KEY or BERGET_API_KEY in environment"
        )

    base_url = os.getenv("BASE_URL", "").strip()

    if not base_url:
        base_url = "https://api.groq.com/openai/v1"

    timeout = float(os.getenv("LLM_TIMEOUT", "60"))

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


client = _get_client()


def call_llm(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    label: str | None = None,
    use_streaming: bool | None = None,
) -> str:
    model_name = model or os.getenv("MODEL", "llama-3.1-8b-instant")

    if temperature is None:
        temperature = float(os.getenv("TEMPERATURE", "0.0"))

    if max_tokens is None:
        max_tokens_env = os.getenv("MAX_TOKENS", "256").strip()
        max_tokens = int(max_tokens_env) if max_tokens_env else 256

    if use_streaming is None:
        use_streaming = os.getenv("USE_STREAMING", "false").lower() == "true"

    disable_thinking = os.getenv("DISABLE_THINKING", "false").lower() == "true"
    extra_body = {"thinking": {"type": "disabled"}} if disable_thinking else None

    estimated_tokens = estimate_tokens(messages)
    label_prefix = f"[{label}] " if label else ""
    stream_indicator = " (streaming)" if use_streaming else ""
    thinking_indicator = " (no-think)" if disable_thinking else ""
    print(f"{label_prefix}[LLM] Model: {model_name}, Est. input tokens: {estimated_tokens}, max_tokens: {max_tokens}{stream_indicator}{thinking_indicator}")

    try:
        if use_streaming:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body=extra_body,
            )
            content = ""
            reasoning = ""
            chunk_count = 0
            finish_reason = None
            for chunk in response:
                chunk_count += 1
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content
                    if hasattr(delta, 'reasoning') and delta.reasoning:
                        reasoning += delta.reasoning
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
            
            if not content and reasoning:
                print(f"{label_prefix}[LLM DEBUG] Model used thinking mode, extracting from reasoning ({len(reasoning)} chars)")
                content = reasoning
            elif not content:
                print(f"{label_prefix}[LLM DEBUG] Streaming: {chunk_count} chunks, no content, finish_reason={finish_reason}")
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            content = response.choices[0].message.content
    except (InternalServerError, RateLimitError, APIConnectionError) as e:
        print(f"{label_prefix}[LLM ERROR] Est. tokens: {estimated_tokens}, Error: {type(e).__name__}")
        print(e)
        return ""

    if content is None:
        print(f"{label_prefix}[LLM WARNING] Model returned None content, Est. tokens: {estimated_tokens}")
        return ""

    return content.strip()