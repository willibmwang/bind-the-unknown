"""Task 2: choosing the next tool to call, scored against a shortest legal path.

The environment is `selector.REQUIREMENTS` read as an API contract rather than as a preference:
a call is legal in a state when the state holds every fact the tool needs, and a successful call
adds the facts the tool declares it provides. Nothing here consults the selector's ordering.

An episode is a seed state, a goal, and a budget. A policy is asked for one tool at a time and
the environment applies it. Three things are then measurable, and they come apart in practice:

* whether the chosen call was *legal* -- the failure mode a tool-description retriever has no way
  to avoid, because the obstruction is in what the state knows and not in the words;
* whether it was *optimal* -- one of the moves that keeps the policy on a shortest path to the
  goal, which is a question about option value and is where a legal-but-greedy policy loses;
* whether the goal was reached at all inside the budget.

**The goals are deliberately different shapes.** `unlock_all` rewards the one call that widens
the action set and is solved in a single move, so it separates policies that notice a bottleneck
from policies that do not. `gate_all` needs a specific pair of calls in either order and cannot
be short-cut. `call_all` has a minimum length equal to the registry size and so measures waste
rather than insight: every illegal or repeated call is pure regret. A policy can top one and fail
another, which is the point of running all three.
"""

from __future__ import annotations

import collections
import json
import math
import random as _random
import re
from dataclasses import dataclass, field

from hpc_pilots import selector, tools

# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

TOOL_NAMES = tuple(sorted(selector.REQUIREMENTS))

# Axes any registered tool can speak to at all. Five of the seven dossier axes are gated by no
# tool, which is a fact about the registry and not about this benchmark, so `gate_all` is defined
# over what is reachable. Scoring a policy against axes nothing can gate would make every arm
# fail identically and measure nothing.
GATEABLE = tuple(sorted({axis for r in selector.REQUIREMENTS.values() for axis in r.gates}))

GOALS = ("unlock_all", "gate_all", "call_all")


@dataclass(frozen=True)
class State:
    """What is known and what has been called. The whole episode state."""

    facts: frozenset = frozenset()
    called: frozenset = frozenset()

    def key(self):
        return (self.facts, self.called)


def legal(tool, state):
    """True when every fact the tool needs is in the state.

    Delegates to `selector.blocked_on` on purpose: a second copy of the precondition check could
    disagree with the one the system uses, and then the benchmark would be grading a policy
    against an environment the policy never ran in.
    """
    situation = selector.Situation(known=state.facts, unresolved=())
    return not selector.blocked_on(tool, situation)


def step(tool, state):
    """The state after a successful call. Illegal calls are the caller's to check.

    `provides` is taken at face value, which is the same assumption `selector.unlocked_by`
    documents: identification against a genome with no recognisable relative returns rows that
    name no reviewed accession, so a call can succeed and provide nothing. Modelling that would
    need a per-genome success model, and the honest place for it is Task 1, where the genome is
    real. Here every arm gets the same optimistic environment, so the comparison is unaffected.
    """
    gained = set(state.facts) | set(selector.REQUIREMENTS[tool].provides)
    return State(facts=frozenset(gained), called=state.called | {tool})


def reached(goal, state):
    """Whether the goal holds in this state."""
    if goal == "unlock_all":
        return all(legal(t, state) for t in TOOL_NAMES)
    if goal == "call_all":
        return set(state.called) >= set(TOOL_NAMES)
    if goal == "gate_all":
        gated = {axis for t in state.called for axis in selector.REQUIREMENTS[t].gates}
        return set(gated) >= set(GATEABLE)
    raise ValueError(f"Unknown goal {goal!r}; goals are {GOALS}")


# ---------------------------------------------------------------------------
# The oracle: shortest legal path, by breadth-first search
# ---------------------------------------------------------------------------

def distances(goal, seed, limit=None):
    """Distance from every state reachable from the seed to the nearest goal state.

    Computed backwards from a forward sweep: the forward pass enumerates what is reachable, and
    the distance is then relaxed until it stops changing. The graph is a DAG in `facts` -- a call
    only ever adds facts -- but not in `called`, so a repeated call is a self-loop in `facts` that
    advances `called`. That is why the relaxation is iterated rather than read off a topological
    order.
    """
    seen, frontier = {seed.key(): seed}, [seed]
    while frontier:
        nxt = []
        for state in frontier:
            for tool in TOOL_NAMES:
                if not legal(tool, state):
                    continue
                after = step(tool, state)
                if after.key() not in seen:
                    seen[after.key()] = after
                    nxt.append(after)
        frontier = nxt
        if limit is not None and len(seen) > limit:
            raise RuntimeError(f"State space exceeded {limit} states")

    dist = {k: (0 if reached(goal, s) else math.inf) for k, s in seen.items()}
    changed = True
    while changed:
        changed = False
        for key, state in seen.items():
            if dist[key] == 0:
                continue
            best = min((dist[step(t, state).key()] for t in TOOL_NAMES if legal(t, state)),
                       default=math.inf)
            if best + 1 < dist[key]:
                dist[key], changed = best + 1, True
    return dist, seen


def optimal_moves(goal, state, dist):
    """Every legal call that keeps the policy on a shortest path.

    A set and not a single tool, because ties are real: `gate_all` from the onset state is
    reached in three calls whether the precedent lookup comes first or second, and grading
    against one arbitrary member of the tie would penalise a correct choice.
    """
    here = dist.get(state.key(), math.inf)
    if here in (0, math.inf):
        return frozenset()
    return frozenset(t for t in TOOL_NAMES
                     if legal(t, state) and dist.get(step(t, state).key(), math.inf) == here - 1)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

# The states a campaign can actually begin in, each one a different claim about what arrived
# from the task statement. `onset` is the real one and the others are the ablations that say
# whether a policy has learned the graph or has memorised the first move.
SEEDS = {
    # One nucleotide accession and nothing else: the state this project is named for.
    "onset": frozenset({"genome_accession"}),
    # A translated reading frame nobody has catalogued. No accession, so every accession-keyed
    # lane is shut and the only structural route is folding. The case a novel pathogen presents,
    # and the one where calling the obvious tool first is wrong.
    "novel_sequence": frozenset({"protein_sequence"}),
    # A named, reviewed protein handed over directly, which is how most tool-use benchmarks
    # implicitly start and is why they never test the obstruction that matters here.
    "known_protein": frozenset({"protein_accession"}),
    # Nothing at all. Two tools need no facts, so the state is not dead, but no goal is
    # reachable -- a policy that claims otherwise is hallucinating reachability.
    "empty": frozenset(),
}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

class Arm:
    """A policy. Given a state and a goal, name the next tool to call."""

    name = "arm"

    def reset(self, goal, seed):
        """Called once per episode. Stateless arms need not override."""

    def choose(self, state, goal):  # pragma: no cover - interface
        raise NotImplementedError


class Oracle(Arm):
    """Plays a shortest path. The ceiling, and the definition of the metric."""

    name = "oracle"

    def __init__(self):
        self._dist = None

    def reset(self, goal, seed):
        self._dist, _ = distances(goal, State(facts=seed))

    def choose(self, state, goal):
        moves = optimal_moves(goal, state, self._dist)
        return min(moves) if moves else None


class RepoSelector(Arm):
    """`selector.first` -- the system under test.

    The goal is passed as the unresolved axis set, which is the only channel the selector has
    for being told what it is working towards. For `unlock_all` and `call_all` there is no axis
    to name, so it runs with nothing unresolved and decides on unblocking alone. That is a real
    limitation of the interface and is reported rather than patched around here: a policy that
    cannot be told the goal will lose the goals that need telling.
    """

    name = "selector"

    def reset(self, goal, seed):
        self._unresolved = tuple(GATEABLE) if goal == "gate_all" else ()

    def choose(self, state, goal):
        situation = selector.Situation(known=state.facts, unresolved=self._unresolved)
        proposal = selector.first(situation)
        return proposal.tool if proposal else None


class RandomAny(Arm):
    """Uniform over the whole registry, preconditions ignored. The floor for legality."""

    name = "random_any"

    def __init__(self, seed=0):
        self._rng = _random.Random(seed)

    def choose(self, state, goal):
        return self._rng.choice(TOOL_NAMES)


class RandomLegal(Arm):
    """Uniform over the legal calls. The floor for *ordering* with legality given free.

    The fair floor, and the one that matters: an arm that is handed the precondition check and
    still loses is losing on judgement, which is the thing being measured.
    """

    name = "random_legal"

    def __init__(self, seed=0):
        self._rng = _random.Random(seed)

    def choose(self, state, goal):
        allowed = [t for t in TOOL_NAMES if legal(t, state)]
        return self._rng.choice(allowed) if allowed else None


_WORD = re.compile(r"[a-z]+")


def _tokens(text):
    return [w for w in _WORD.findall(text.lower()) if len(w) > 2]


# What each goal would be phrased as, if a retriever were asked to serve it from text alone.
# Written to be fair rather than to fail: each one names the thing the goal is about in the
# vocabulary the tool descriptions use.
GOAL_TEXT = {
    "unlock_all": "make every tool callable, produce the facts other tools need, "
                  "identify candidate proteins from the pathogen genome",
    "gate_all": "measure the exposed binding surface and the deposited structural precedent "
                "for the target, accessibility and prior structures",
    "call_all": "run every available tool on the target: genome, literature, family, "
                "similarity, structure, surface, bioactivity, chemistry",
}


class Similarity(Arm):
    """Lexical TF-IDF between the goal statement and each tool's own summary and description.

    The baseline that stands in for description-based tool retrieval, and the one the whole
    precondition design is an argument against: it ranks on what a tool is *about* and has no
    representation of whether the tool can run. Lexical rather than neural so that it runs with
    no key and no model download; `embed=` takes a real encoder for anyone who wants the stronger
    version, and the ranking is otherwise identical.

    It never repeats a call inside an episode. That is a concession in its favour -- a pure
    retriever would return the same top hit for ever, and scoring that would flatter the
    comparison rather than inform it.
    """

    name = "similarity"

    def __init__(self, embed=None):
        self._embed = embed
        self._docs = {t: f"{tools.TOOLS[t].summary} {tools.TOOLS[t].description}"
                      for t in TOOL_NAMES if t in tools.TOOLS}
        counts = collections.Counter()
        for text in self._docs.values():
            counts.update(set(_tokens(text)))
        n = len(self._docs)
        self._idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in counts.items()}
        self._used = set()

    def reset(self, goal, seed):
        self._used = set()

    def _score(self, tool, goal):
        if self._embed is not None:
            return self._embed(GOAL_TEXT[goal], self._docs.get(tool, tool))
        want = collections.Counter(_tokens(GOAL_TEXT[goal]))
        have = collections.Counter(_tokens(self._docs.get(tool, tool)))
        if not want or not have:
            return 0.0
        dot = sum(want[w] * have[w] * self._idf.get(w, 1.0) ** 2 for w in want if w in have)
        nw = math.sqrt(sum((c * self._idf.get(w, 1.0)) ** 2 for w, c in want.items()))
        nh = math.sqrt(sum((c * self._idf.get(w, 1.0)) ** 2 for w, c in have.items()))
        return dot / (nw * nh) if nw and nh else 0.0

    def choose(self, state, goal):
        pool = [t for t in TOOL_NAMES if t not in self._used] or list(TOOL_NAMES)
        pick = max(pool, key=lambda t: (self._score(t, goal), t))
        self._used.add(pick)
        return pick


class PreconditionAwareSimilarity(Similarity):
    """Similarity, but filtered to the legal calls. The ablation that isolates the two ideas.

    Between this and `Similarity` lies exactly the value of knowing what a tool needs; between
    this and `Oracle` lies the value of knowing what a call unlocks. Running both says which of
    the two claims the system's design rests on is doing the work, which neither arm alone can.
    """

    name = "similarity_legal"

    def choose(self, state, goal):
        allowed = [t for t in TOOL_NAMES if legal(t, state)]
        if not allowed:
            return None
        pool = [t for t in allowed if t not in self._used] or allowed
        pick = max(pool, key=lambda t: (self._score(t, goal), t))
        self._used.add(pick)
        return pick


class Scripted(Arm):
    """Replays a fixed sequence. How an LLM transcript is scored without calling one live.

    `benchmarks.task2 --emit-prompts` writes the prompt for each decision point; an answer file
    of tool names per episode is replayed through this arm. Keeping the model out of the harness
    is deliberate: an LLM arm that re-runs on every invocation makes the benchmark unreproducible
    and bills for it, and the thing worth recording is the transcript, not the API call.
    """

    name = "scripted"

    def __init__(self, moves, name="scripted"):
        self._moves, self.name = dict(moves), name
        self._queue = []

    def reset(self, goal, seed):
        self._queue = list(self._moves.get(goal, []))

    def choose(self, state, goal):
        return self._queue.pop(0) if self._queue else None


# ---------------------------------------------------------------------------
# Running an episode
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One arm, one goal, one seed, and everything the run is scored on."""

    arm: str
    goal: str
    seed: str
    optimal: float
    calls: list = field(default_factory=list)
    illegal: list = field(default_factory=list)
    repeats: list = field(default_factory=list)
    suboptimal: list = field(default_factory=list)
    decisions: int = 0
    on_path: int = 0
    reached: bool = False

    def record(self):
        used = len(self.calls)
        return {
            "arm": self.arm, "goal": self.goal, "seed": self.seed,
            "optimal_calls": self.optimal, "calls_used": used,
            "reached": self.reached,
            # Regret only means anything once the goal was reached. Reported as None rather than
            # as a large number otherwise, because an arm that never finished and an arm that
            # finished slowly are different results and a sentinel would merge them.
            "regret": (used - self.optimal) if (self.reached and self.optimal != math.inf)
                      else None,
            "illegal_calls": len(self.illegal), "repeated_calls": len(self.repeats),
            "decisions": self.decisions, "on_shortest_path": self.on_path,
            "optimal_move_rate": (self.on_path / self.decisions) if self.decisions else None,
            "legal_move_rate": ((self.decisions - len(self.illegal)) / self.decisions)
                               if self.decisions else None,
            "sequence": list(self.calls),
            "first_illegal": self.illegal[0] if self.illegal else None,
        }


def episode(arm, goal, seed_name, budget=None, strict=False):
    """Run one arm to the goal or to the budget.

    An illegal call costs a turn and changes nothing, which is what a real dispatcher does when
    a precondition is missing: the call is attempted, it fails, the budget is spent. With
    `strict`, it ends the episode instead -- the harsher reading, where a policy that emits an
    impossible call has produced an unusable plan. Both are reported because they answer
    different questions, and an arm that looks fine under one can look very different under the
    other.
    """
    seed = SEEDS[seed_name]
    state = State(facts=seed)
    dist, _ = distances(goal, state)
    optimal = dist.get(state.key(), math.inf)
    # Enough room to be slow without running for ever: twice the registry, or twice the optimal.
    if budget is None:
        budget = max(2 * len(TOOL_NAMES), 2 * (optimal if optimal != math.inf else 0)) or 4

    run = Episode(arm=arm.name, goal=goal, seed=seed_name, optimal=optimal)
    arm.reset(goal, seed)
    if reached(goal, state):
        run.reached = True
        return run

    while len(run.calls) + len(run.illegal) < budget:
        best = optimal_moves(goal, state, dist)
        pick = arm.choose(state, goal)
        if pick is None:
            break
        run.decisions += 1
        if pick in best:
            run.on_path += 1
        elif legal(pick, state):
            run.suboptimal.append(pick)
        if not legal(pick, state):
            run.illegal.append(pick)
            if strict:
                break
            continue
        if pick in state.called:
            run.repeats.append(pick)
        state = step(pick, state)
        run.calls.append(pick)
        if reached(goal, state):
            run.reached = True
            break
    return run


def arms():
    """The comparison ladder, in the order it should be read."""
    return [Oracle(), RepoSelector(), PreconditionAwareSimilarity(), Similarity(),
            RandomLegal(seed=0), RandomAny(seed=0)]


def run(goals=GOALS, seeds=tuple(SEEDS), built=None, strict=False, repeats=20):
    """Every arm on every goal and seed.

    The two random arms are averaged over `repeats` seeds and the deterministic arms are run
    once, because a single draw from a random policy is a number with no error bar and would
    invite exactly the over-reading this benchmark exists to prevent.
    """
    rows = []
    for goal in goals:
        for seed_name in seeds:
            for arm in (arms() if built is None else built):
                if isinstance(arm, (RandomAny, RandomLegal)):
                    draws = []
                    for i in range(repeats):
                        twin = type(arm)(seed=i)
                        draws.append(episode(twin, goal, seed_name, strict=strict).record())
                    rows.append(_mean(draws, arm.name, repeats))
                else:
                    rows.append(episode(arm, goal, seed_name, strict=strict).record())
    return {"benchmark": "task2-tool-selection",
            "environment": {"tools": len(TOOL_NAMES), "facts": len(selector.FACTS),
                            "gateable_axes": list(GATEABLE),
                            "spec_digest": tools.spec_set_digest()},
            "strict": strict, "random_draws": repeats, "rows": rows}


def _mean(draws, name, repeats):
    """Average a random arm's draws, keeping the spread on the headline number."""
    def avg(key):
        vals = [d[key] for d in draws if d[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    rates = [d["optimal_move_rate"] for d in draws if d["optimal_move_rate"] is not None]
    sd = (math.sqrt(sum((r - sum(rates) / len(rates)) ** 2 for r in rates) / len(rates))
          if len(rates) > 1 else 0.0)
    first = draws[0]
    return {"arm": name, "goal": first["goal"], "seed": first["seed"],
            "optimal_calls": first["optimal_calls"],
            "calls_used": avg("calls_used"), "reached": avg("reached"),
            "regret": avg("regret"), "illegal_calls": avg("illegal_calls"),
            "repeated_calls": avg("repeated_calls"),
            "decisions": avg("decisions"), "on_shortest_path": avg("on_shortest_path"),
            "optimal_move_rate": avg("optimal_move_rate"),
            "optimal_move_rate_sd": sd,
            "legal_move_rate": avg("legal_move_rate"),
            "sequence": None, "first_illegal": None, "draws": repeats}


# ---------------------------------------------------------------------------
# Per-state scoring: every decision point, not just the ones a policy walked into
# ---------------------------------------------------------------------------

def decision_points(goal, seed_name):
    """Every reachable state where a choice exists, with its optimal moves.

    Episode scoring only ever grades the states an arm reached, so a policy that goes wrong at
    move one is never graded on the rest of the graph, and two arms are compared on different
    instances. This enumerates the whole reachable set instead, which makes the instances
    identical for every arm and turns the benchmark from a race into a labelled test set.
    """
    dist, seen = distances(goal, State(facts=SEEDS[seed_name]))
    out = []
    for key, state in seen.items():
        if dist[key] in (0, math.inf):
            continue
        best = optimal_moves(goal, state, dist)
        if not best:
            continue
        out.append({"facts": tuple(sorted(state.facts)),
                    "called": tuple(sorted(state.called)),
                    "distance": dist[key], "optimal": tuple(sorted(best)),
                    "legal": tuple(sorted(t for t in TOOL_NAMES if legal(t, state)))})
    out.sort(key=lambda r: (r["distance"], r["facts"], r["called"]))
    return out


def score_states(arm, goal, seed_name, points=None):
    """An arm's accuracy over every decision point, each state judged independently."""
    points = decision_points(goal, seed_name) if points is None else points
    arm.reset(goal, SEEDS[seed_name])
    legal_hits = optimal_hits = 0
    for point in points:
        state = State(facts=frozenset(point["facts"]), called=frozenset(point["called"]))
        arm.reset(goal, SEEDS[seed_name])  # each state judged cold, so order cannot help
        pick = arm.choose(state, goal)
        if pick in point["optimal"]:
            optimal_hits += 1
        if pick in point["legal"]:
            legal_hits += 1
    n = len(points)
    return {"arm": arm.name, "goal": goal, "seed": seed_name, "states": n,
            "optimal_move_rate": (optimal_hits / n) if n else None,
            "legal_move_rate": (legal_hits / n) if n else None}


def run_states(goals=GOALS, seeds=tuple(SEEDS), built=None):
    """Per-state accuracy for every arm, over the whole reachable graph."""
    rows = []
    for goal in goals:
        for seed_name in seeds:
            points = decision_points(goal, seed_name)
            if not points:
                continue
            for arm in (arms() if built is None else built):
                rows.append(score_states(arm, goal, seed_name, points))
    return {"benchmark": "task2-tool-selection-per-state", "rows": rows}


# ---------------------------------------------------------------------------
# Blocked-fact reporting: the thing only this system claims to do
# ---------------------------------------------------------------------------

def blocked_report(goal="unlock_all", seed_name="onset"):
    """Ground truth for "which tools cannot run here, and what does each one lack".

    Scored apart from move choice because it is a different competence and the system's own
    documentation rests on it: a blocked tool is reported rather than dropped, so that a policy
    can tell a useless tool from one that is a single call away. An arm naming the wrong missing
    fact has produced a plausible, confident, wrong account of why it is stuck -- which is worse
    than being stuck, and is invisible to any benchmark that only grades the chosen call.
    """
    _, seen = distances(goal, State(facts=SEEDS[seed_name]))
    # Keyed on the fact set alone and de-duplicated: what is blocked depends only on what is
    # known, so the `called` component of a state is irrelevant here and reporting it would
    # multiply one question into a hundred identical ones.
    out = {}
    for state in seen.values():
        facts = tuple(sorted(state.facts))
        if facts in out:
            continue
        situation = selector.Situation(known=state.facts, unresolved=())
        out[facts] = {t: selector.blocked_on(t, situation) for t in TOOL_NAMES
                      if selector.blocked_on(t, situation)}
    return [{"facts": facts, "blocked": out[facts]} for facts in sorted(out)]


def main(argv=None):
    """`python -m benchmarks.task2` -- the whole thing, as JSON on stdout."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="end an episode on the first illegal call")
    parser.add_argument("--repeats", type=int, default=20,
                        help="draws for each random arm")
    parser.add_argument("--states", action="store_true",
                        help="per-state accuracy over the whole reachable graph")
    parser.add_argument("--points", metavar="GOAL",
                        help="dump the labelled decision points for one goal")
    parser.add_argument("--seed", default="onset", choices=sorted(SEEDS))
    args = parser.parse_args(argv)

    if args.points:
        payload = {"goal": args.points, "seed": args.seed,
                   "points": decision_points(args.points, args.seed)}
    elif args.states:
        payload = run_states()
    else:
        payload = run(strict=args.strict, repeats=args.repeats)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
