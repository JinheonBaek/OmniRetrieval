"""
LLM client supporting OpenAI, Anthropic, Google (Gemini), and vLLM.

Usage:
    client = LLMClient(provider="openai", model="gpt-5.4-mini")
    response = client.generate("What is 2+2?")
    response = client.generate(
        "Translate to French.",
        system="You are a translator.",
        temperature=0.3,
        max_tokens=256,
    )
"""

import json
import time
from typing import Optional

DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-3-flash-preview",
    "vllm": "Qwen/Qwen3.5-4B",
}

MAX_RETRIES = 5


class LLMClient:
    def __init__(
        self,
        provider: str = "openai",  # "openai" | "anthropic" | "google" | "vllm"
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,  # falls back to env var if None
        extra_kwargs: Optional[dict] = None,
    ):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider!r}. Use 'openai', 'anthropic', 'google', or 'vllm'.")

        self.provider = provider
        self.model = model or DEFAULT_MODELS[self.provider]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.extra_kwargs = extra_kwargs or {}
        self._init_client()

    # ------------------------------------------------------------------
    # Provider init
    # ------------------------------------------------------------------

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key) if self.api_key else Anthropic()
        elif self.provider == "google":
            from google import genai
            self._client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        elif self.provider == "vllm":
            from vllm import LLM
            self._client = LLM(model=self.model, max_num_seqs=8)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            fn = self._generate_openai
        elif self.provider == "anthropic":
            fn = self._generate_anthropic
        elif self.provider == "google":
            fn = self._generate_google
        elif self.provider == "vllm":
            fn = self._generate_vllm
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Use 'openai', 'anthropic', 'google', or 'vllm'.")

        return self._retry_generate(fn, prompt, system, temperature, max_tokens)

    def _retry_generate(self, fn, prompt, system, temperature, max_tokens) -> str:
        for attempt in range(MAX_RETRIES - 1):
            try:
                return fn(prompt, system, temperature, max_tokens)
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
        return fn(prompt, system, temperature, max_tokens)

    def _generate_openai(self, prompt: str, system: str | None, temperature: float, max_tokens: int) -> str:
        from openai import OpenAI
        assert isinstance(self._client, OpenAI)
        client = self._client

        kwargs = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **self.extra_kwargs,
        }
        if system:
            kwargs["instructions"] = system

        response = client.responses.create(**kwargs)
        return response.output_text

    def _generate_anthropic(self, prompt: str, system: str | None, temperature: float, max_tokens: int) -> str:
        from anthropic import Anthropic
        assert isinstance(self._client, Anthropic)
        client = self._client

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.extra_kwargs,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _generate_google(self, prompt: str, system: str | None, temperature: float, max_tokens: int) -> str:
        from google import genai
        from google.genai import types
        assert isinstance(self._client, genai.Client)
        client = self._client

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            **self.extra_kwargs,
        ) if system else types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            **self.extra_kwargs,
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def _generate_vllm(self, prompt: str, system: str | None, temperature: float, max_tokens: int) -> str:
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
        assert isinstance(self._client, LLM)

        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        outputs = self._client.chat(
            messages,
            sampling_params=SamplingParams(temperature=temperature, max_tokens=max_tokens),
            use_tqdm=False,
            **self.extra_kwargs,
        )
        return outputs[0].outputs[0].text

    def generate_json(
            self, 
            prompt: str, 
            system: Optional[str] = None, 
            temperature: Optional[float] = None, 
            max_tokens: Optional[int] = None
    ) -> dict:
        """Generate and parse as JSON."""
        instruction = "Respond with valid JSON only. No markdown, no extra text."
        full_system = f"{system}\n\n{instruction}" if system else instruction
        raw = self.generate(prompt, system=full_system, temperature=temperature, max_tokens=max_tokens)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting from code fences
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
