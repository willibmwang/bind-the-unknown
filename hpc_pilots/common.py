import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "hpc_pilots/configs/today.json").read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "2", "--max-time", "180", "--output", str(tmp), url], check=True)
        tmp.replace(path)
    return {"url": url, "path": str(path.relative_to(ROOT)),
            "sha256": digest(path), "bytes": path.stat().st_size,
            "verified_at_utc": datetime.now(timezone.utc).isoformat()}


# Hosted OpenAI-compatible backends selected with PILOT_LLM_PROVIDER: (url, key variable, default model).
PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", None),
}
# Current Claude models reject `temperature`; current OpenAI models want `max_completion_tokens`.
NO_TEMPERATURE = {"anthropic"}
MAX_TOKENS_KEY = {"openai": "max_completion_tokens"}


def load_env(path=ROOT / ".env"):
    """Fill unset variables from a gitignored KEY=value file; real environment variables win."""
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class Agent:
    """OpenAI-compatible inference (local vLLM or a hosted provider); failed inference never substitutes a baseline."""

    def __init__(self):
        load_env()
        self.provider = os.environ.get("PILOT_LLM_PROVIDER", "local")
        self.url = os.environ.get("PILOT_LLM_URL", "").rstrip("/")
        self.model = os.environ.get("PILOT_LLM_MODEL", "")
        self.token_file = os.environ.get("PILOT_LLM_TOKEN_FILE", "")
        # Reasoning models spend part of this budget thinking, so allow headroom when needed.
        self.max_tokens = int(os.environ.get("PILOT_LLM_MAX_TOKENS", "192"))
        if self.provider != "local":
            if self.provider not in PROVIDERS:
                raise RuntimeError(f"Unknown PILOT_LLM_PROVIDER {self.provider!r}")
            url, key_var, default_model = PROVIDERS[self.provider]
            self.url = self.url or url
            self.model = self.model or default_model or ""
            if not os.environ.get(key_var):
                raise RuntimeError(f"PILOT_LLM_PROVIDER={self.provider} requires {key_var}")
            os.environ.setdefault("PILOT_LLM_TOKEN", os.environ[key_var])
        endpoint = os.environ.get("PILOT_LLM_ENDPOINT_FILE", "")
        if endpoint:
            spec = json.loads(Path(endpoint).read_text())
            self.url, self.model = spec["url"], spec["model"]
            self.token_file = spec.get("token_file", self.token_file)
        if not self.url or not self.model:
            raise RuntimeError("Agent requires PILOT_LLM_URL and PILOT_LLM_MODEL")
        # Hosted runs get their own policy name so they never overwrite or pool with local-model runs.
        self.label = "agent" if self.provider == "local" else \
            f"agent-{self.provider}-{self.model.replace('/', '-').replace(':', '-')}"
        self.calls = 0
        self.seconds = 0.0
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        self.history = []

    def choose(self, task, state, allowed):
        import requests
        system = ("You allocate a limited scientific evidence budget. Use only the supplied state. "
                  "Do not infer hidden experimental outcomes. Return one JSON object containing "
                  "action (one of the supplied allowed actions) and rationale (a short string). "
                  "Choose an action that improves decision quality per unit of evidence cost.")
        payload = {"model": self.model, MAX_TOKENS_KEY.get(self.provider, "max_tokens"): self.max_tokens,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": json.dumps({"task": task,
                                 "state": state, "allowed_actions": allowed})}]}
        if self.provider not in NO_TEMPERATURE:
            payload["temperature"] = 0
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "scientific_action", "strict": True, "schema": {"type": "object",
            "properties": {"action": {"type": "string", "enum": allowed},
                           "rationale": {"type": "string"}},
            "required": ["action", "rationale"], "additionalProperties": False}}}
        headers = {}
        token = os.environ.get("PILOT_LLM_TOKEN")
        if not token and self.token_file:
            token = Path(self.token_file).read_text().strip()
        if token:
            headers["Authorization"] = "Bearer " + token
        start = time.perf_counter()
        response = requests.post(self.url + "/chat/completions", json=payload,
                                 headers=headers, timeout=90)
        response.raise_for_status()
        body = response.json()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        for k in self.tokens:
            self.tokens[k] += int(body.get("usage", {}).get(k, 0))
        content = body["choices"][0]["message"]["content"]
        # Accept fenced JSON or surrounding text, but not an invented/fallback action.
        first, last = content.find("{"), content.rfind("}")
        if first < 0 or last < first:
            raise ValueError("Agent did not return a JSON object")
        answer = json.loads(content[first:last + 1])
        if answer.get("action") not in allowed:
            raise ValueError("Agent returned an action outside the allowed set")
        self.history.append({"state": state, "response": answer})
        return answer["action"]

    def report(self):
        return {"provider": self.provider, "model": self.model, "calls": self.calls, "seconds": self.seconds,
                "tokens": self.tokens, "history": self.history}
