import json
from urllib import error, request


DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_BASE_URL = "http://localhost:11434"


class LocalLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 512,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict
        self.last_stats = None

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        req = request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.base_url} or the request timed out."
            ) from exc

        generated_text = data.get("response")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("Ollama returned an empty response.")
        
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        self.last_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        
        return generated_text.strip()