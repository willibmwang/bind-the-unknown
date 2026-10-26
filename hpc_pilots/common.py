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


# Fields recording *when* something was observed rather than *what* it is. They belong in a
# manifest -- knowing when a download was last verified is useful -- but they must not
# participate in the manifest's identity.
VOLATILE_KEYS = ("verified_at_utc", "recorded_at_utc", "staged_at_utc")


def content_digest(obj):
    """A hash of what a manifest says, ignoring when it said it.

    `digest` over manifest.json cannot serve as a data identity, because `download` stamps
    `verified_at_utc` on every call including a cache hit. Re-staging byte-identical inputs
    therefore produced a different manifest hash every time, which means the
    `dataset_manifest_sha256` recorded in a result could never be checked by re-staging --
    only by finding the one manifest still on disk. A provenance record that cannot be
    recomputed is a label, not a record.

    This strips the timestamps and hashes the rest with sorted keys, so the result depends
    on the content and not on key order or the moment of observation.
    """
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()
                    if k not in VOLATILE_KEYS and k != "content_sha256"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return hashlib.sha256(
        json.dumps(strip(obj), sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode()).hexdigest()


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

# Provider-side stop reasons worth another attempt. A content filter is intermittent and
# content-dependent: measured on one evidence packet, two of four attempts were stopped at
# 666 and 795 characters while rationales elsewhere complete at 919, so the output ceiling
# is not the cause and the same request can succeed unchanged.
#
# `length` is deliberately absent. A request that truncated once will truncate again, and
# retrying it would spend money to produce the same failure; it needs a raised ceiling.
#
# An action outside the allowed set is also absent, and must stay absent. That is a policy
# result, not a provider failure, and resampling until the policy returns something legal
# would quietly filter its behaviour out of the record.
# How a tool's parameters reach the model. `flat` emits one object whose properties are the
# union over the shown tools, with a parameter the chosen tool does not declare arriving null;
# `nested` emits an `arguments` branch per tool; `two_call` chooses the tool and then fills its
# arguments in a second request. `flat` is the default because, when every shown tool declares
# every parameter as required, it emits the payload this method has always emitted -- byte for
# byte -- so the recorded episodes replay with no mode flag at all. Its limit is not a tool
# count but the first parameter-name collision, at which point `_union` raises rather than
# letting one tool's bounds constrain another tool's decoding.
TOOL_SCHEMA_MODES = ("flat", "nested", "two_call")
# Off by default, and the default is the one that stays comparable with the roughly forty
# recorded episodes. `1` puts the registry's policy-facing projection into the state.
TOOL_STATE_MODES = ("0", "1")

RETRYABLE_FINISH = {"content_filter"}


class ContentFiltered(ValueError):
    """Every attempt was stopped by the provider's content filter.

    Its own type so a caller running a batch can record the episode as filtered and carry
    on to the next one, while an action outside the allowed set -- a plain ValueError --
    still aborts. A filtered episode is missing data with a known cause, not a result and
    not a licence to substitute anything for it.
    """

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


def _union(tools):
    """The ordered parameter union over the shown tools, raising on a real collision.

    Declaration order is preserved, first tool first, because the order of the emitted
    properties is part of what the model sees and reordering it would be a silent change to
    the prompt.

    Two tools may share a parameter name only if they declare it identically. Otherwise the
    flat encoding cannot represent both -- whichever is emitted last constrains the other's
    decoding, and nothing but post-hoc validation would catch the mismatch, which is the thing
    constrained decoding exists to avoid. So the limit on this mode is stated as a condition
    rather than guessed as a tool count, and it raises with the remedy.
    """
    union, seen = [], {}
    for spec in tools:
        for parameter in spec.parameters:
            identity = (parameter.kind, tuple(parameter.enum), parameter.enum_from,
                        parameter.minimum, parameter.maximum, parameter.max_length)
            if parameter.name in seen:
                if seen[parameter.name] != identity:
                    raise ValueError(
                        f"Tools declare {parameter.name!r} differently, which a flat schema "
                        f"cannot carry: one tool's bounds would constrain another tool's "
                        f"decoding. Set PILOT_TOOL_SCHEMA_MODE=nested or two_call.")
                continue
            seen[parameter.name] = identity
            union.append(parameter)
    return union


def _nullable(parameter, choices, required_everywhere):
    """One flat property. Null is the encoding of `not a parameter of the chosen tool`.

    When every shown tool declares the parameter as required, no null branch is emitted and
    the property is exactly what the `fields` form emits -- which is what makes an all-required
    tool set replay the recorded payload byte for byte.

    Otherwise the enum carries None explicitly. With `type: ["string", "null"]` and an enum
    that omits null, null is not a permitted value, so a conforming decoder cannot emit the
    one thing this encoding needs it to emit. That failure presents as a provider that "does
    not support schemas" rather than as a schema bug, which is why it is spelled out here.
    """
    values = parameter.values(choices)
    if required_everywhere:
        return {"type": parameter.json_type(), "enum": list(values)} if values \
            else {"type": parameter.json_type()}
    if values:
        return {"type": [parameter.json_type(), "null"], "enum": list(values) + [None]}
    return {"type": [parameter.json_type(), "null"]}


def _object(spec, choices):
    """One tool's arguments as a strict object, for the `nested` mode."""
    properties = {p.name: ({"type": p.json_type(), "enum": list(p.values(choices))}
                           if p.values(choices) else {"type": p.json_type()})
                  for p in spec.parameters}
    return {"type": "object", "title": spec.name, "properties": properties,
            "required": list(properties), "additionalProperties": False}


class Agent:
    """OpenAI-compatible inference (local vLLM or a hosted provider); failed inference never substitutes a baseline."""

    def __init__(self, tool_state_applies=False):
        load_env()
        self.provider = os.environ.get("PILOT_LLM_PROVIDER", "local")
        self.url = os.environ.get("PILOT_LLM_URL", "").rstrip("/")
        self.model = os.environ.get("PILOT_LLM_MODEL", "")
        self.token_file = os.environ.get("PILOT_LLM_TOKEN_FILE", "")
        # Reasoning models spend part of this budget thinking before emitting anything, so
        # the ceiling is not a bound on the answer's length. 192 was sized for the local 4B
        # model and silently made a hosted reasoning model unusable: every call failed on
        # finish_reason='length', and raising to 512 made the emitted content *shorter*,
        # because the extra budget went to reasoning that still did not reach the JSON.
        # 4096 is what actually completed. The default is per provider rather than global,
        # since the local model does not need the headroom and paying for it would change
        # the recorded binding replays.
        self.max_tokens = int(os.environ.get(
            "PILOT_LLM_MAX_TOKENS", "192" if self.provider == "local" else "4096"))
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
        self.retries = int(os.environ.get("PILOT_LLM_RETRIES", "4"))
        self.tool_schema_mode = os.environ.get("PILOT_TOOL_SCHEMA_MODE", "flat")
        if self.tool_schema_mode not in TOOL_SCHEMA_MODES:
            raise RuntimeError(f"Unknown PILOT_TOOL_SCHEMA_MODE {self.tool_schema_mode!r}; "
                               f"known modes are {list(TOOL_SCHEMA_MODES)}")
        # A non-default mode writes its own artifact, for the reason the hosted label exists:
        # `two_call` spends two requests per action, so its token and call counts are not
        # comparable with a single-call run and the two must never pool.
        if self.tool_schema_mode != "flat":
            self.label = f"{self.label}-{self.tool_schema_mode}"
        # Whether the tool specs themselves reach the policy, as opposed to being the shape
        # the caller validates against. The environment states the request and the caller
        # states whether it can honour it, and the effective value is the conjunction.
        #
        # Both halves are needed, and getting this wrong was a real defect. Reading the
        # variable alone put `-toolstate` on every Agent, but only `emergency` and `twins`
        # build a state that can carry the block: `binding` and `structure` construct the
        # same class and never consult it. So one stray variable relabelled arms that had
        # shown the policy nothing, splitting forty recorded binding artifacts off from runs
        # identical to them. The label lied in the direction nobody was watching -- claiming
        # specs were shown where there was no block to show.
        #
        # An inapplicable request is recorded rather than raised, because `PILOT_TOOL_STATE`
        # describes an episode's state and `binding` has no such state to describe; refusing
        # would let a flag for one experiment break three others. It is not silent either:
        # `report()` carries the request and the effective value side by side, and an arm
        # that showed nothing keeps its plain label and so still pools with the runs it is
        # genuinely comparable to.
        self.tool_state = os.environ.get("PILOT_TOOL_STATE", "0")
        if self.tool_state not in TOOL_STATE_MODES:
            raise RuntimeError(f"Unknown PILOT_TOOL_STATE {self.tool_state!r}; known values "
                               f"are {list(TOOL_STATE_MODES)}. A typo must not read as off, "
                               f"because the off state is the one that pools with the record.")
        self.tool_state_applies = tool_state_applies
        self.shows_tools = tool_state_applies and self.tool_state == "1"
        # The input text changes, so no episode recorded before this is comparable with one
        # after it even at identical payload bytes -- the schema is unmoved and the prompt is
        # not. The suffix follows the effective value, never the request.
        if self.shows_tools:
            self.label = f"{self.label}-toolstate"
        self.calls = 0
        # Requests per action: two under `two_call`, one otherwise. Recorded so that
        # `calls / tool_turns` still recovers the retry count, which `calls` alone stops
        # meaning the moment an action costs more than one request.
        self.tool_turns = 0
        self.seconds = 0.0
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        self.history = []
        # Recovery is recorded, not hidden. An episode that needed four attempts to get a
        # parseable answer is not the same observation as one that needed one, and a
        # filtered attempt still costs tokens, so both have to reach the artifact.
        self.filtered = 0
        self.transport_retries = 0
        self.incidents = []

    def _properties(self, allowed, fields, tools, choices):
        """The schema's properties, by mode, so that the three modes are a table.

        The discriminant key stays `action` and is deliberately not renamed to `tool`.
        Renaming it would change the shape of every entry in every recorded `agent.history`
        and break `answer["action"]` at its call sites, and buy nothing: the enum holds a tool
        name either way.

        `tools=None` returns exactly what this method has always returned.
        """
        properties = {"action": {"type": "string", "enum": list(allowed)},
                      "rationale": {"type": "string"}}
        for name, values in (fields or {}).items():
            properties[name] = {"type": "string", "enum": list(values)}
        if tools is None:
            return properties
        if self.tool_schema_mode == "nested":
            properties["arguments"] = {"anyOf": [_object(spec, choices) for spec in tools]}
            return properties
        for parameter in _union(tools):
            required_everywhere = all(
                any(p.name == parameter.name and p.required for p in spec.parameters)
                for spec in tools)
            properties[parameter.name] = _nullable(parameter, choices, required_everywhere)
        return properties

    def act(self, task, state, allowed, fields=None, system=None, tools=None, choices=None):
        """Return the validated action object. `fields` adds enum-constrained arguments.

        An action or argument outside the allowed set raises; it never degrades into a
        default or baseline choice.

        `tools` is an ordered sequence of tool spec records and `choices` supplies the
        vocabulary a parameter declares late. Late binding is not a convenience: a spec whose
        enum hard-coded target names would survive `relabel_packet`, which rewrites packet
        content and not source constants, so the identity orbit would read a renamed packet
        while the model still saw the real names.
        """
        import requests
        properties = self._properties(allowed, fields, tools, choices)
        self.tool_turns += 1
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
        for attempt in range(1, self.retries + 1):
            body = self._post(payload, headers, requests)
            choice = body["choices"][0]
            content = choice["message"]["content"]
            reason = choice.get("finish_reason")
            # Accept fenced JSON or surrounding text, but not an invented/fallback action.
            first, last = content.find("{"), content.rfind("}")
            if first >= 0 and last > first:
                break
            self.incidents.append({"attempt": attempt, "finish_reason": reason,
                                   "chars": len(content), "content_head": content[:160]})
            if reason in RETRYABLE_FINISH and attempt < self.retries:
                self.filtered += 1
                continue
            # The causes need different responses, and treating them alike wasted a
            # debugging session: a ceiling is raised, a filter is recorded and retried.
            hint = {"length": f" The response hit the {self.max_tokens}-token output ceiling; "
                              "raise PILOT_LLM_MAX_TOKENS.",
                    "content_filter": f" The provider stopped this response on a content filter "
                                      f"on all {attempt} attempts, so the output ceiling is not "
                                      "the cause and raising it will not help. The episode is "
                                      "not substitutable: record the packet as filtered and "
                                      "report it alongside the episodes that completed."
                    }.get(reason, "")
            error = ContentFiltered if reason == "content_filter" else ValueError
            raise error(f"Agent did not return a JSON object (finish_reason={reason!r},"
                        f" {len(content)} chars).{hint} Content began: {content[:200]!r}")
        answer = json.loads(content[first:last + 1])
        if answer.get("action") not in allowed:
            raise ValueError("Agent returned an action outside the allowed set")
        for name, values in (fields or {}).items():
            if answer.get(name) not in values:
                raise ValueError(f"Agent returned {name} outside the allowed set")
        if tools is not None:
            # Only what the emitted schema constrained. A violation here means the decoder did
            # not honour the grammar, which is an instrument failure; everything the spec
            # declares but the schema could not carry is the caller's to charge and record.
            # Imported inside the call so that this module keeps importing nothing but the
            # standard library, which is what lets `dossier` and `tools` stay dependency-free.
            from .chief import schema_conformance
            chosen = next(spec for spec in tools if spec.name == answer["action"])
            supplied = {p.name: answer[p.name] for p in chosen.parameters if p.name in answer}
            schema_conformance(chosen, supplied, choices)
        self.history.append({"state": state, "response": answer})
        return answer

    def _post(self, payload, headers, requests):
        """One completion, retrying only transport failures that could plausibly clear.

        A 429 or a 5xx is worth waiting out; a long sweep should not die on one. Any
        other 4xx is a malformed request and will fail identically however often it is
        sent, so it is raised immediately rather than retried for a minute first.
        """
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            start = time.perf_counter()
            try:
                response = requests.post(self.url + "/chat/completions", json=payload,
                                         headers=headers, timeout=90)
                response.raise_for_status()
            except requests.RequestException as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if attempt == self.retries or not retryable:
                    raise
                self.transport_retries += 1
                self.incidents.append({"attempt": attempt, "status": status,
                                       "transport_error": str(error)[:160]})
                time.sleep(delay)
                delay *= 2
                continue
            self.calls += 1
            self.seconds += time.perf_counter() - start
            body = response.json()
            for key in self.tokens:
                self.tokens[key] += int(body.get("usage", {}).get(key, 0))
            return body

    def choose(self, task, state, allowed):
        """Action-only interface retained so the existing binding replays reproduce exactly."""
        return self.act(task, state, allowed)["action"]

    def report(self):
        return {"provider": self.provider, "model": self.model, "calls": self.calls,
                "seconds": self.seconds, "tokens": self.tokens,
                "filtered_attempts": self.filtered, "transport_retries": self.transport_retries,
                "incidents": self.incidents, "history": self.history,
                "tool_schema_mode": self.tool_schema_mode, "tool_turns": self.tool_turns,
                # The request and what came of it. An arm where the request does not apply
                # records "1" beside False, which is the honest pair: asked for, nothing to
                # show, label unchanged, still comparable with its own recorded runs.
                "tool_state": self.tool_state,
                "tool_state_applies": self.tool_state_applies,
                "shows_tools": self.shows_tools}
