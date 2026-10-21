"""The evidence lattice: a decision state with nowhere to put a target's name.

Everything measured in this project says the same thing twice. A score on a named-target
decision problem splits into the part the evidence state supports and the part the target's
name supplies, and the second part is invisible to every readout except an orbit built to see
it: a planted memoriser holding the answer key attains the ceiling on the graded probe while
scoring +0.65 to +1.26 of pure identity effect. A score on a monomer library splits the same
way, and a tree ensemble over one-hot monomers has a tied-prediction rate of exactly 1.0 on
blocks it has never seen, because a vocabulary has no coefficient for a symbol absent from
training.

Both failures are representational. A model that *can* key on an instance's identity will,
and no amount of evaluation discipline prevents it -- discipline only measures it afterwards.
So this module takes the other route and removes the place where identity would live.

**The state.** A decision problem becomes a `(targets x axes)` lattice of cells. A cell is six
numbers: whether the claim is observed, its strength as an ordinal, whether it is cited,
whether something contradicts it, whether it is purchasable now, and whether it currently
counts as substantiated. An axis carries its *gating role*, not its index. **A target carries
nothing at all** -- it is exactly its row of cells. There is no target embedding, no target
index, no slot in which a name could be stored.

Three properties follow, and they are properties of the representation rather than of a
trained model's behaviour:

* **Identity-invariance by construction.** Renaming every target cannot change the lattice, so
  it cannot change any function of the lattice. The certified orbit must return an identity
  effect of exactly zero, and `tests` checks that it does. This is not evidence that the
  policy is good; it is evidence that the measurement and the representation agree, and it is
  the one claim here that needs no experiment.
* **Permutation-equivariance on targets.** Rows carry no order, so a policy over the lattice
  must commute with permuting them. `rank` resolves ties by the caller's list order, which is
  how the oracle acquired an order-dependence that took 26-versus-28 relevant pairs to find;
  a representation with no row order cannot reacquire it.
* **Schema generality.** Axes are typed by whether they gate rather than by position, so a
  rubric with a different gate, or a different number of axes, is the same kind of object.
  `monitor.RULE_VARIANTS` already moves the gate across four axes and removes it entirely, so
  this is exercised rather than hoped for.

**Why this shape and not a bigger one.** The scoring function below is hand-specified, and
that is deliberate for two reasons. It is the baseline the research brief pre-registered and
the project never built -- S3 names "a simple uncertainty-reduction-per-cost heuristic" and
says the policy contribution is only worth pursuing if gains survive it. And it fixes the
interface a learned policy has to honour, so the architectural guarantees above are verified
before any pretraining is paid for. A learned policy over the same lattice inherits all three
properties; what it adds is skill, and that is the only thing left to measure.

**What it must not do, enforced elsewhere.** The score is computed from the lattice alone,
under a prior on what an unobserved claim will say. It never consults `graded_moves` or
`decision_relevant`. The evaluator's oracle marginalises over evidence it can see and a policy
cannot, so a policy reading it would be scoring against its own answer key; `improve`'s
`state_is_oracle_free` guard builds the state with the oracle patched to raise, which is the
check that this stayed true.
"""
import numpy as np

from .dossier import ASSESSMENTS, AXES
from .streams import stream

# One row per cell feature, in order. Declared as data because a learned policy has to agree
# with the heuristic about the layout, and a silent disagreement would be a rewiring nobody
# could see in a number.
CELL_FEATURES = ("observed", "strength", "cited", "contested", "offered", "substantiated")

# What an axis carries. Its gating role and nothing else -- no index, no name, no position --
# so a rubric with a different gate or a different number of axes is the same kind of object.
AXIS_FEATURES = ("hard_gate",)

# The prior that an unobserved claim, once bought, will turn out cited and at least moderate,
# i.e. that it will substantiate its axis. Measured over the three packets' 59 base entries:
# 40 of 59 qualify, so the base rate is 0.678 rather than a guess. A single scalar on purpose
# -- a policy needing a per-axis prior would carry a fitted parameter per axis, and a new
# rubric arrives with none, which is the thing this representation exists to survive.
#
# The value is also deliberately inert: the score is `prior x indicator`, so a constant prior
# scales every cell equally and the argmax does not depend on it at all. It is carried because
# the magnitude is reported, not because the choice turns on it. `tests` pins that invariance,
# which is what stops this becoming a tuned knob.
SUBSTANTIATION_PRIOR = 0.678


def _substantiated(entries):
    """Whether an axis counts, by the same rule `readiness` uses.

    Reimplemented over observed entries rather than called on the dossier, because the lattice
    has to be constructible from what a policy can see. `Dossier.readiness` is the same
    arithmetic and the two are checked against each other in the tests; if they ever disagree
    the lattice is describing a different decision problem from the one being scored.
    """
    if not entries:
        return False
    cited = [e for e in entries if e["verified"]]
    if not cited:
        return False
    best = max(ASSESSMENTS.index(e["strength"]) for e in cited)
    contested = any(e["conflicts_with"] for e in entries)
    return best >= ASSESSMENTS.index("moderate") and not contested


def lattice(dossiers, pool, retrieved, now, axes=None):
    """Build the policy-visible state. Returns cells, the offered mask, and the axis features.

    `order` is returned so a caller can map a chosen cell back to a target, and it is the only
    place a name appears. Nothing downstream of here reads it except to name the action, which
    is why renaming the targets cannot change what is chosen.
    """
    axes = list(axes or AXES)
    order = [d.target for d in dossiers]
    cells = np.zeros((len(dossiers), len(axes), len(CELL_FEATURES)), dtype=np.float64)
    offered = np.zeros((len(dossiers), len(axes)), dtype=bool)

    for row, dossier in enumerate(dossiers):
        for column, axis in enumerate(axes):
            entries = [e for e in dossier.observed if e["axis"] == axis]
            key = (dossier.target, axis)
            purchasable = key in pool and key not in retrieved \
                and pool[key]["available_from"] <= now
            strength = (max(ASSESSMENTS.index(e["strength"]) for e in entries)
                        / (len(ASSESSMENTS) - 1)) if entries else 0.0
            cells[row, column] = (
                float(bool(entries)),
                strength,
                float(any(e["verified"] for e in entries)),
                float(any(e["conflicts_with"] for e in entries)),
                float(purchasable),
                float(_substantiated(entries)))
            offered[row, column] = purchasable

    axis_features = np.array([[float(AXES[a]["hard_gate"]) if a in AXES else 0.0]
                              for a in axes], dtype=np.float64)
    return cells, offered, axis_features, order


def _lead_key(cells, axis_features):
    """The ranking key for every row, computed from the lattice alone.

    Deliberately the same key `dossier._rank_key` uses, minus its third term. The third term
    breaks ties on how much of a target remains uninvestigated, which is a judgement about how
    to treat ignorance; a policy asserting it would be asserting the authored rule rather than
    inferring from the state. Dropping it leaves more rows tied, which makes the policy
    *less* decisive and is the conservative direction.
    """
    substantiated = cells[:, :, CELL_FEATURES.index("substantiated")]
    gates = axis_features[:, AXIS_FEATURES.index("hard_gate")].astype(bool)
    gate_met = substantiated[:, gates].all(axis=1) if gates.any() \
        else np.ones(len(cells), dtype=bool)
    return gate_met, substantiated.sum(axis=1)


def _leaders(gate_met, counts):
    """The rows tied for the lead, as a frozenset of row indices.

    Row indices, not names. The whole point of the representation is that the caller's naming
    never enters, and returning a set rather than a first element is what stops a tie being
    resolved by whichever row happened to be listed first.
    """
    best = max(zip(gate_met, counts), key=lambda pair: (pair[0], pair[1]))
    return frozenset(row for row, pair in enumerate(zip(gate_met, counts)) if pair == best)


def score(cells, offered, axis_features, prior=SUBSTANTIATION_PRIOR):
    """Expected reduction in who-leads uncertainty, per action, from the lattice alone.

    This is the heuristic the brief pre-registered. Every action costs one, so cost-per-action
    is constant and the score is the expected value of the information: the probability that
    buying a claim substantiates its axis, times whether that would change which rows are
    still contending for the lead.

    The second factor is what makes it not leader-first, and that is the whole point. A
    purchase on the leading row usually extends a lead already held and changes nothing about
    who is contending; a purchase on a row one claim short of tying can change it. The fixed
    rubric cannot express that preference -- it walks the leader's axes in declaration order
    and is pinned at the ceiling a leader-only policy cannot cross -- so a heuristic that can
    is the comparator the policy contribution has to beat.

    Computed under a prior because the claim's strength is hidden until it is bought. That is
    the honest asymmetry between a policy and the evaluator: the oracle knows what each
    purchase would reveal, and a policy knows only the base rate.
    """
    gate_met, counts = _lead_key(cells, axis_features)
    before = _leaders(gate_met, counts)
    substantiated = cells[:, :, CELL_FEATURES.index("substantiated")]
    gates = axis_features[:, AXIS_FEATURES.index("hard_gate")].astype(bool)

    before_key = max(zip(gate_met, counts), key=lambda pair: (pair[0], pair[1]))
    out = np.full(offered.shape, -np.inf)
    for row, column in zip(*np.nonzero(offered)):
        if substantiated[row, column]:
            # Already counted, so buying it again cannot move the contest.
            out[row, column] = 0.0
            continue
        hypothetical = substantiated.copy()
        hypothetical[row, column] = 1.0
        moved_gate = hypothetical[:, gates].all(axis=1) if gates.any() else gate_met
        moved_counts = hypothetical.sum(axis=1)
        after = _leaders(moved_gate, moved_counts)
        after_key = max(zip(moved_gate, moved_counts), key=lambda pair: (pair[0], pair[1]))
        out[row, column] = prior * _magnitude(before, before_key, after, after_key)
    return out


def _magnitude(before, before_key, after, after_key):
    """How much a purchase would move the contest, graded as the evaluator grades it.

    The first version of this scored only whether the *tied set* changed, and that made the
    policy the mirror image of the one it was built to beat. A purchase that better
    substantiates an already-clear leader does not change who is contending, so it scored
    zero -- and in the stratum where the best purchase *is* on the leader the policy therefore
    avoided the optimum systematically. Measured, it attained 0.88 and 0.93 on the hard
    stratum against a leader-first ceiling of 0.26 and 0.05, and 0.30 and 0.18 on the easy one
    against the rubric's 1.0. Trading one fixation for its opposite is not a better policy.

    The grades mirror `twins.GRADES` because a policy is allowed to know the decision *rule* --
    the rubric is authored and public -- and is not allowed to know the hidden evidence. That
    is the real asymmetry: the oracle knows what each unbought claim says, and the policy
    knows only the base rate at which such a claim substantiates anything.
    """
    if after == before and after_key == before_key:
        return 0.0
    if after != before:
        return 3.0 if len(after) == 1 < len(before) else 2.0
    return 1.0


def lattice_action(dossiers, pool, retrieved, now, ranking=None, prior=SUBSTANTIATION_PRIOR,
                   axes=None, seed=20261021):
    """Choose the purchase with the highest expected value of information.

    Ties are broken by the lowest row and column index, which is a property of the lattice and
    not of the caller's ordering: the rows arrive in whatever order the caller built them, so
    the tie-break is only well defined because the score is permutation-equivariant and the
    argmax over a permuted lattice is the permuted argmax. The test checks exactly that.

    Returns the episode action triple, so this drops into the same policy slot as the fixed
    rubric and is measured by the same instrument.
    """
    columns = list(axes or AXES)
    cells, offered, axis_features, order = lattice(dossiers, pool, retrieved, now, columns)
    if not offered.any():
        # Nothing purchasable. Recommend if the lead is held and the gate is met, else stop --
        # the same two terminal moves the rubric has, chosen from the lattice rather than from
        # a ranking the caller supplied.
        gate_met, counts = _lead_key(cells, axis_features)
        leaders = _leaders(gate_met, counts)
        # Content-keyed for the same reason the purchase branch is: `min(leaders)` is a row
        # position, and a position is exactly what this representation refuses to carry.
        tied = sorted((tuple(cells[r, :, :].ravel()), int(r)) for r in leaders)
        row = tied[int(stream(seed, "lattice-lead", len(tied)).integers(len(tied)))][1]
        action = "recommend_experiment" if gate_met[row] and len(leaders) == 1 else "stop"
        return action, order[row], None

    values = score(cells, offered, axis_features, prior)
    best = values.max()
    # Ties are broken on the cell's *content*, never on its position. Taking the lowest row
    # index looked equivalent and was not: permuting the rows then changes which target wins a
    # tie, so the policy was not equivariant after all -- the same defect, in the same shape,
    # as `rank` resolving ties by the caller's list order. The test caught it, which is the
    # only reason the architectural claim is worth stating.
    #
    # Then taking the lowest *column* index was wrong the same way, and worse than it sounds.
    # Measured, it gave the policy a 0.291 axis-order effect against the fixed rubric's 0.116:
    # nearly a third of its headline attainment was the authored axis declaration order rather
    # than anything in the lattice. An axis carries only whether it gates, so six of the seven
    # are feature-identical and content cannot separate them either.
    #
    # Cells the lattice genuinely cannot distinguish are therefore chosen between at random,
    # reproducibly -- the same resolution `pick_batch` uses for the surrogate's tied plateaus
    # and for the same reason. An arbitrary order is not information, and a policy that reads
    # one is reporting the author's keystrokes as skill.
    candidates = sorted((tuple(cells[row, :, :].ravel()), tuple(axis_features[column]),
                         int(row), int(column))
                        for row, column in zip(*np.nonzero(values == best)))
    _, _, row, column = candidates[
        int(stream(seed, "lattice-tie", len(candidates)).integers(len(candidates)))]
    # Named from the presented order, not from the declaration order. A column index means
    # whatever the caller listed at that position, and reading the name from `AXES` instead
    # would silently return a different axis whenever the two differ.
    return "retrieve_evidence", order[row], columns[column]
