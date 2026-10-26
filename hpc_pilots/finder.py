"""Which tools to show a policy this round, and the record of why.

At the scale this repository actually runs, the honest answer is all of them, and the
measurement says so. The action block is 511 characters against a 26,198-character
coronaviridae state -- 2.0% -- and six tools carrying parameter blocks takes it to roughly
9%. It would take on the order of sixty-five tools before the block matched the dossiers
sitting beside it. Below that, a retrieval stage can only spend reachability to buy nothing,
so `all` is the default and it writes down that it showed everything.

**Withholding a tool is a result-affecting change, not an optimisation.** The property this
project exists to guarantee is that a policy cannot reach an answer it has not earned, and a
retriever that hides a tool changes what the policy can reach. So a restricted Finder is a
named arm that writes its own artifact, never a silent default, and `withheld_the_chosen_tool`
is reported against a paired unrestricted run rather than left to be inferred.

**The query is built only from vocabulary a relabelling cannot move, and this is the
constraint that shapes the module.** Two tests pin `attributable_to_identity` at exactly zero
for the rubric on all three packets. A keyword Finder whose query came from dossier text would
make the shown set a function of the target names, so even the rubric -- which reads no state
-- would have its reachable set move under a rename, and those tests would fail. Correctly.
So `_query` draws on `AXES` and `ASSESSMENTS`, which are module constants that
`relabel_packet` does not touch, and on the round and action counters. Never target names,
never pathogen names, never claim text, never source strings. The terms are sorted, because
the axis-order nuisance is a measured quantity -- 0.116 -- and a query that depended on the
order of a list would be reading it.

**Each stage states the condition under which it earns its place**, rather than a guess:

* `all` -- up to around sixty-five tools, where the block reaches the size of the state it
  sits in.
* `keyword` -- from around fifteen. Not prompt size but the flat schema: past that, the
  property union carries parameters no shown tool needs on most rounds, and every surplus
  nullable property is a branch the decoder can take wrongly.
* `tfidf` -- from around sixty. scikit-learn is already pinned, so this is the embedding
  stage done honestly; a sentence-transformer is a 400 MB dependency and a *second model* in
  a repository whose entire LLM story is that exactly one client exists. Sixty because below
  it the inverse document frequency over a handful of descriptions is too sparse to mean
  anything, and a ranking over fifteen documents is noise dressed as similarity.
* `shuffle` -- never a candidate. It is the floor: it shows as many tools as the real Finder
  showed, drawn from a declared stream. A keyword stage that does not beat it at equal count
  is doing nothing, which is the same move `_coverage_action` makes as a counting control.
* An LLM-semantic stage is deliberately absent. Three conditions would all have to hold: the
  registry exceeds what a prompt can carry; the call is charged to the action budget, because
  a model request that decides what the policy can reach is itself a policy move and an
  unscored one is a hole in the accounting; and it is certified against the identity orbit.

What this is not: it does not validate or execute anything, and it does not rank tools by
cost. It decides what is visible, which is why every path through it is recorded.
"""
from .dossier import ASSESSMENTS, AXES

DEFAULT = "all"

# Tools that may never be withheld. A Finder that hid `stop` would make `terminal_condition`,
# `deadline_met` and `actions_unspent_at_exit` properties of the retriever; one that hid
# `recommend_experiment` would make `recommendation_issued` one. Both would be read as policy
# behaviour, so withholding either raises rather than being reported.
PINNED = ("recommend_experiment", "stop")


def _query(state):
    """The retrieval query, from vocabulary no relabelling can move.

    Drawn from `AXES` and `ASSESSMENTS` -- source constants -- plus the round and action
    counters. Sorted, so that the order of any list in the state cannot reach the query.
    """
    terms = set()
    for dossier in state.get("dossiers", ()) or ():
        for axis, assessment in (dossier.get("axes", {}) or {}).items():
            if axis not in AXES:
                continue
            verdict = assessment.get("assessment") if isinstance(assessment, dict) else None
            if verdict in ASSESSMENTS and verdict in ("unknown", "weak"):
                terms.add(axis)
    for pair in state.get("retrievable_axes", ()) or ():
        if pair.get("axis") in AXES:
            terms.add(pair["axis"])
    if state.get("rounds_remaining") == 0:
        terms.add("stop")
    return tuple(sorted(terms))


def _all(names, descriptions, state, limit, seed):
    return list(names), {"query": (), "note": "showed every registered tool"}


def _keyword(names, descriptions, state, limit, seed):
    query = _query(state)
    scored = []
    for name in names:
        text = f"{name} {descriptions.get(name, '')}".lower()
        scored.append((sum(term.lower() in text for term in query), name))
    ranked = [name for score, name in
              sorted(scored, key=lambda pair: (-pair[0], list(names).index(pair[1])))
              if score > 0]
    return ranked[:limit] if limit else ranked, {"query": query, "note": "substring match"}


def _tfidf(names, descriptions, state, limit, seed):
    query = _query(state)
    if not query:
        return list(names), {"query": query, "note": "empty query; showed everything"}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [f"{name} {descriptions.get(name, '')}" for name in names]
    matrix = TfidfVectorizer().fit_transform(corpus + [" ".join(query)])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
    keep = [list(names)[i] for i in order if scores[i] > 0]
    return keep[:limit] if limit else keep, {"query": query, "note": "tf-idf cosine"}


def _shuffle(names, descriptions, state, limit, seed):
    """The floor: as many tools as the real Finder showed, chosen without reading anything."""
    from .streams import stream

    if seed is None:
        raise ValueError("The shuffle floor needs a seed, so that what it showed is replayable")
    pool = [n for n in names if n not in PINNED]
    generator = stream(seed, "finder", str(state.get("decision_round", 0)))
    order = list(generator.permutation(len(pool)))
    want = max(0, (limit or len(names)) - len(PINNED))
    kept = [pool[i] for i in order[:want]]
    return kept + [n for n in PINNED if n in names], {"query": (),
                                                      "note": "declared-stream floor"}


FINDERS = {"all": _all, "keyword": _keyword, "tfidf": _tfidf, "shuffle": _shuffle}


def find(names, descriptions=None, state=None, mode=None, limit=None, seed=None):
    """Return `(shown, record)`: which tools to offer, and why.

    `shown` becomes the policy's `allowed`, so the two can never diverge. `Agent.act` raises on
    an action outside `allowed`, and a Finder decision surfacing as a crash attributed to the
    model is the worst available outcome.

    Declaration order is preserved regardless of how a stage ranked, because the order of the
    offered set is part of what the model reads and the axis-order nuisance already shows this
    project can measure an order effect.
    """
    names = list(names)
    chosen = mode or DEFAULT
    if chosen not in FINDERS:
        raise ValueError(f"Unknown finder {chosen!r}; registered finders are {sorted(FINDERS)}")
    ranked, record = FINDERS[chosen](names, descriptions or {}, state or {}, limit, seed)
    # A ranking stage that matched nothing shows everything, and says so. Measured on the
    # action space this is the common case rather than the edge: the query is axis names and
    # the action descriptions do not contain them, so `keyword` over the six actions ranks
    # nothing and would leave only the two pinned actions -- making `retrieve_evidence`
    # unreachable, and changing an episode's outcome for a reason that has nothing to do with
    # the policy. Showing everything is the conservative direction, not a substituted default:
    # it is what `all` does, which is the behaviour the recorded runs already have. What must
    # never be silent is the blackout, so it is recorded and an arm that reports
    # `empty_ranking` on every round is a Finder doing nothing, which is a finding about the
    # Finder.
    empty_ranking = not ranked
    if empty_ranking:
        ranked = list(names)
    # Checked against the stage's own output and not against the repaired set. Unioning the
    # pinned names back in would hold the invariant while making this unreachable, so a stage
    # that tried to hide `stop` would be silently corrected -- and a stage that tries to hide a
    # terminal action is a bug in the stage, not a runtime condition to absorb.
    omitted = [name for name in PINNED if name in names and name not in set(ranked)]
    if omitted:
        raise ValueError(f"The finder withheld {omitted}, which are pinned; withholding a "
                         f"terminal action makes the episode's end a property of the "
                         f"retriever rather than of the policy")
    shown = [name for name in names if name in set(ranked)]
    if not set(shown) <= set(names):
        raise ValueError("The finder showed a tool that was not offered")
    return shown, {**record, "mode": chosen, "offered": len(names), "shown": shown,
                   "empty_ranking": empty_ranking,
                   "withheld": [name for name in names if name not in shown],
                   "pinned": [name for name in PINNED if name in names], "limit": limit,
                   "interpretation": "What the policy could reach this round. A withheld tool "
                                     "is a reachability change and not a cost saving."}
