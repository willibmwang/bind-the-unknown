import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# `stage_model` writes the first; the repository carries the second as the committed pin.
STAGED_MODEL_MANIFEST = ROOT / "models/manifest.json"
COMMITTED_MODEL_MANIFEST = ROOT / "hpc_pilots/model.manifest.json"

# Everything whose content can change a result. Hashed into source_manifest.json.
SOURCE_PATTERNS = ("hpc_pilots/*.py", "hpc_pilots/tests/*.py", "hpc_pilots/configs/*.json",
                   "hpc_pilots/slurm/*.sbatch", "hpc_pilots/requirements*.txt")


def config():
    return json.loads((ROOT / "hpc_pilots/configs/today.json").read_text())


def model_manifest():
    """The staged manifest if weights were downloaded here, else the committed pin.

    Both `stage_model` and `submit` previously named `models/manifest.json` directly,
    which does not exist in a clean checkout, so the GPU submission guard refused a
    submission whose pin was in fact committed.
    """
    return STAGED_MODEL_MANIFEST if STAGED_MODEL_MANIFEST.exists() else COMMITTED_MODEL_MANIFEST


def _git(*argv):
    try:
        done = subprocess.run(("git", "-C", str(ROOT)) + argv, capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def provenance():
    """Where a run was produced and from which source state.

    Run records carried no host or commit, so whether a recorded sweep came from the
    cluster or a local re-run could not be recovered from the artifact. `git_dirty`
    is the one to read first: a run from a modified tree is not reproducible from the
    commit it names.
    """
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    return {"host": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "git_commit": commit,
            "git_dirty": None if dirty is None else bool(dirty),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat()}


def sources():
    """Digest every source file that can change a result, in a stable order."""
    files = {}
    for pattern in SOURCE_PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(ROOT))] = digest(path)
    return files


def source_manifest(check=False):
    """Rewrite `source_manifest.json`, or report how far it has drifted from the tree.

    Nothing regenerated this file, so seven of its digests no longer matched their
    files: a record that looks like provenance and is not is worse than none. `check`
    returns the drift without writing, for use in CI or before a submission.
    """
    path = ROOT / "hpc_pilots/source_manifest.json"
    files = sources()
    recorded = json.loads(path.read_text()).get("files", {}) if path.exists() else {}
    drift = {"changed": sorted(k for k, v in files.items() if k in recorded and recorded[k] != v),
             "added": sorted(set(files) - set(recorded)),
             "removed": sorted(set(recorded) - set(files))}
    if not check:
        write_json(path, {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                          "git_commit": _git("rev-parse", "HEAD"),
                          "git_dirty": bool(_git("status", "--porcelain")),
                          "patterns": list(SOURCE_PATTERNS), "files": files})
    return path, drift


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

# The original domain-free prompt for the budgeted-acquisition pilots. Kept verbatim so the
# recorded binding replays remain reproducible; the emergency policy supplies its own.
BUDGET_SYSTEM = ("You allocate a limited scientific evidence budget. Use only the supplied state. "
                 "Do not infer hidden experimental outcomes. Return one JSON object containing "
                 "action (one of the supplied allowed actions) and rationale (a short string). "
                 "Choose an action that improves decision quality per unit of evidence cost.")


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

    def act(self, task, state, allowed, fields=None, system=None):
        """Return the validated action object. `fields` adds enum-constrained arguments.

        An action or argument outside the allowed set raises; it never degrades into a
        default or baseline choice.
        """
        import requests
        properties = {"action": {"type": "string", "enum": list(allowed)},
                      "rationale": {"type": "string"}}
        for name, values in (fields or {}).items():
            properties[name] = {"type": "string", "enum": list(values)}
        payload = {"model": self.model, MAX_TOKENS_KEY.get(self.provider, "max_tokens"): self.max_tokens,
                   "messages": [{"role": "system", "content": system or BUDGET_SYSTEM},
                                {"role": "user", "content": json.dumps({"task": task,
                                 "state": state, "allowed_actions": allowed})}]}
        if self.provider not in NO_TEMPERATURE:
            payload["temperature"] = 0
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "scientific_action", "strict": True, "schema": {"type": "object",
            "properties": properties,
            "required": list(properties), "additionalProperties": False}}}
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
        choice = body["choices"][0]
        content = choice["message"]["content"]
        # Accept fenced JSON or surrounding text, but not an invented/fallback action.
        first, last = content.find("{"), content.rfind("}")
        if first < 0 or last < first:
            # The usual cause is truncation at the output-token ceiling, which leaves an
            # opening brace and no closing one. Say so, rather than only that parsing failed.
            reason = choice.get("finish_reason")
            # The two causes need different responses, and treating them alike wasted a
            # debugging session: a ceiling is raised, a filter is recorded and retried.
            hint = {"length": f" The response hit the {self.max_tokens}-token output ceiling; "
                              "raise PILOT_LLM_MAX_TOKENS.",
                    "content_filter": " The provider stopped this response on a content filter, "
                                      "so the output ceiling is not the cause and raising it will "
                                      "not help. This fires on some evidence packets and not "
                                      "others, which makes it a reproducibility hazard rather "
                                      "than a bug: an episode can fail for reasons unrelated to "
                                      "the policy, so record which packets filtered and report "
                                      "them alongside the episodes that completed."}.get(reason, "")
            raise ValueError(f"Agent did not return a JSON object (finish_reason={reason!r},"
                             f" {len(content)} chars).{hint} Content began: {content[:200]!r}")
        answer = json.loads(content[first:last + 1])
        if answer.get("action") not in allowed:
            raise ValueError("Agent returned an action outside the allowed set")
        for name, values in (fields or {}).items():
            if answer.get(name) not in values:
                raise ValueError(f"Agent returned {name} outside the allowed set")
        self.history.append({"state": state, "response": answer})
        return answer

    def choose(self, task, state, allowed):
        """Action-only interface retained so the existing binding replays reproduce exactly."""
        return self.act(task, state, allowed)["action"]

    def report(self):
        return {"provider": self.provider, "model": self.model, "calls": self.calls, "seconds": self.seconds,
                "tokens": self.tokens, "history": self.history}
