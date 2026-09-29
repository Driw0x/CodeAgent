import json
import os
from urllib import error, request

DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = os.getenv("CODEAGENT_OLLAMA_KEEP_ALIVE", "30m")


class LocalLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 512,
        keep_alive: str | int = DEFAULT_KEEP_ALIVE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.last_stats = None
        self.last_request_stats = None
        self.usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset_usage_stats(self):
        self.usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _request(self, endpoint: str, payload: dict) -> dict:
        req = request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.base_url} or the request timed out."
            ) from exc

    def _record_stats(self, data: dict):
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        self.last_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.last_request_stats = {
            "load_duration_ms": (data.get("load_duration", 0) or 0) / 1_000_000,
            "prompt_eval_duration_ms": (
                data.get("prompt_eval_duration", 0) or 0
            )
            / 1_000_000,
            "eval_duration_ms": (
                data.get("eval_duration", 0) or 0
            )
            / 1_000_000,
            "total_duration_ms": (
                data.get("total_duration", 0) or 0
            )
            / 1_000_000,
        }
        self.usage_stats["prompt_tokens"] += prompt_tokens
        self.usage_stats["completion_tokens"] += completion_tokens
        self.usage_stats["total_tokens"] += prompt_tokens + completion_tokens

    def preload(self) -> dict:
        data = self._request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [],
                "stream": False,
                "keep_alive": self.keep_alive,
            },
        )
        self.last_request_stats = {
            "load_duration_ms": (data.get("load_duration", 0) or 0) / 1_000_000,
            "total_duration_ms": (data.get("total_duration", 0) or 0) / 1_000_000,
        }
        return self.last_request_stats

    def unload(self):
        self._request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [],
                "stream": False,
                "keep_alive": 0,
            },
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        data = self._request("/api/generate", payload)
        generated_text = data.get("response")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("Ollama returned an empty response.")

        self._record_stats(data)

        return generated_text.strip()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        if not messages:
            raise ValueError("Messages cannot be empty.")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }

        if tools:
            payload["tools"] = tools

        data = self._request("/api/chat", payload)
        message = data.get("message")

        if not isinstance(message, dict):
            raise RuntimeError("Ollama returned an invalid chat response.")

        content = message.get("content")
        tool_calls = message.get("tool_calls", [])

        if (not isinstance(content, str) or not content.strip()) and not tool_calls:
            raise RuntimeError("Ollama returned an empty chat response.")

        self._record_stats(data)

        return message
