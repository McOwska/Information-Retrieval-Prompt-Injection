from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from openai import InternalServerError, RateLimitError, APIConnectionError


load_dotenv()


def _get_client() -> OpenAI:
    api_key = (
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("BERGET_API_KEY", "").strip()
    )

    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY, OPENAI_API_KEY or BERGET_API_KEY in environment"
        )

    base_url = os.getenv("BASE_URL", "").strip()

    if not base_url:
        base_url = "https://api.groq.com/openai/v1"

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


client = _get_client()


def call_llm(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    model_name = model or os.getenv("MODEL", "llama-3.1-8b-instant")

    if temperature is None:
        temperature = float(os.getenv("TEMPERATURE", "0.0"))

    if max_tokens is None:
        max_tokens_env = os.getenv("MAX_TOKENS", "256").strip()
        max_tokens = int(max_tokens_env) if max_tokens_env else 256

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (InternalServerError, RateLimitError, APIConnectionError) as e:
        print(f"LLM request failed: {type(e).__name__}")
        print(e)
        return ""

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content.strip()