"""Generation providers behind one interface.

Every provider failure is wrapped in LLMError so the API can answer 502 with the
provider's own message. A decommissioned model or an exhausted quota is not a
bug in this service, and reporting it as 500 sends the reader hunting in the
wrong codebase.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings


class LLMError(RuntimeError):
    """The generation provider refused, failed, or timed out."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class GroqLLM:
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Put it in .env or export it, then restart the API."
            )
        from groq import Groq

        self.client = Groq(api_key=settings.groq_api_key)
        self.settings = settings

    def complete(self, system: str, user: str) -> Completion:
        import groq
        import httpx

        s = self.settings
        try:
            resp = self.client.chat.completions.create(
                model=s.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=s.temperature,
                max_tokens=s.max_tokens,
            )
        except (groq.GroqError, httpx.HTTPError) as exc:
            raise LLMError(str(exc)) from exc

        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            model=s.llm_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, system: str, user: str) -> Completion:
        import httpx

        s = self.settings
        try:
            resp = httpx.post(
                f"{s.ollama_url}/api/chat",
                json={
                    "model": s.ollama_model,
                    "stream": False,
                    "options": {"temperature": s.temperature},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=300,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama at {s.ollama_url}: {exc}") from exc

        body = resp.json()
        return Completion(
            text=body.get("message", {}).get("content", ""),
            model=s.ollama_model,
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
        )


def build_llm(settings: Settings):
    provider = settings.llm_provider.strip().lower()
    if provider == "groq":
        return GroqLLM(settings)
    if provider == "ollama":
        return OllamaLLM(settings)
    raise LLMError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}; use 'groq' or 'ollama'.")
