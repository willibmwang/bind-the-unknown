"""What a surrogate is, in one place, so swapping one is a config change and not a code edit.

The acquisition replay has fitted a random forest from scratch at every round since the first
pilot, and the model was named nowhere: `predictions` constructed a `RandomForestRegressor`
directly, so trying a different one meant editing the function every policy calls. The arms
went through the same thing -- four `run_*` bodies differing in a handful of strings until
`arms.ARMS` made the difference data -- and the same argument applies here. A registry rather
than a class hierarchy, because what varies between surrogates is a handful of facts and one
function, which is a record and not a subclass.

Three of those facts are recorded because results depend on them and a reader cannot recover
them from a number.

**Whether the spread is a posterior.** The forest's is not. It is the dispersion of the
per-tree predictions, and the docstring in `binding` has always said so. That matters because
the acquisition rule is `mean + ucb_beta * sigma`, and `ucb_beta` is only interpretable as a
confidence multiplier when sigma is a predictive standard deviation. With a dispersion
heuristic it is a tuning constant whose units are the heuristic's, which is a reasonable thing
to do and a different thing from what the formula appears to say. Carrying the distinction as
a field means every result says which it had rather than relying on whoever reads it knowing.

**Whether the model wants a design matrix.** The forest wants the one-hot encoding, so an
unseen category arrives as a column with no training variance and no split can use it -- which
is why a tree ensemble's predictions for two library members differing only in an unseen
monomer are bit-identical, measured at a tied rate of exactly 1.0 in `monomers`. A model that
conditions on rows of raw feature values has no per-column coefficient to be missing, so the
question has a different answer and has to be asked separately. The flag is what lets the
held-out-monomer protocol put the same question to both.

**What it needs installed.** The scientific environment has no torch, deliberately: it is kept
able to resolve without CUDA pins, and the smoke suite runs in it in under three seconds. A
surrogate requiring torch therefore belongs behind a named dependency that raises with its
cause, rather than an `ImportError` surfacing from inside the fifteenth round of a replay.

The forest remains the default and its path is unchanged, so every recorded replay reproduces
bit-for-bit. A config without a `surrogate` key selects it, which is every config currently on
disk.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor

DEFAULT = "forest"


@dataclass(frozen=True)
class Surrogate:
    """One surrogate: what it needs, what its spread means, and how to call it."""

    name: str
    fit_predict: Callable
    """(design, raw, observed_indices, observed_values, seed, block) -> (mean, spread)."""
    takes_design_matrix: bool
    """True for a model over one-hot columns, False for one conditioning on raw values."""
    spread_is_posterior: bool
    """Whether the second return value may be read as a predictive standard deviation."""
    spread_note: str
    """What the spread actually is, carried into every result that used it."""
    requires: tuple = field(default_factory=tuple)
    """Import names that must be available. Checked before use so a failure names its cause."""

    def available(self):
        import importlib.util

        return [name for name in self.requires if importlib.util.find_spec(name) is None]


def _forest(design, raw, observed_indices, observed_values, seed, block):
    """The pilot's surrogate, unchanged.

    Every argument is passed exactly as it was before the registry existed, including the
    default of 64 trees when a config omits `forest_trees`, because the recorded replays
    depend on all of it and reproducing them bit-for-bit is the only way to tell a new
    surrogate's effect from a refactor's.
    """
    trees = (block or {}).get("forest_trees", 64)
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                  max_features=1.0, n_jobs=1, random_state=seed)
    model.fit(design[observed_indices], observed_values)
    estimates = np.array([tree.predict(design) for tree in model.estimators_])
    return estimates.mean(axis=0), estimates.std(axis=0)


SURROGATES = {
    "forest": Surrogate(
        name="forest", fit_predict=_forest, takes_design_matrix=True,
        spread_is_posterior=False,
        spread_note="Dispersion of the per-tree predictions. Not a predictive standard "
                    "deviation, so ucb_beta multiplies a heuristic rather than a confidence "
                    "interval."),
}


def surrogate(name=None):
    if name is None:
        name = DEFAULT
    if name not in SURROGATES:
        raise ValueError(f"Unknown surrogate {name!r}; registered surrogates are "
                         f"{sorted(SURROGATES)}")
    model = SURROGATES[name]
    missing = model.available()
    if missing:
        raise RuntimeError(
            f"Surrogate {name!r} needs {missing}, which is not importable here. The "
            f"scientific environment is deliberately kept free of torch so that it resolves "
            f"without CUDA pins; install a surrogate's dependencies into their own "
            f"environment rather than into that one.")
    return model


def register(model):
    """Add a surrogate to the registry.

    Exists so a surrogate carrying a heavy dependency can be declared in its own module and
    registered on import, instead of this module importing torch to describe a model nobody
    selected.
    """
    if not isinstance(model, Surrogate):
        raise TypeError("A surrogate must be a Surrogate record")
    if model.name in SURROGATES and SURROGATES[model.name] is not model:
        raise ValueError(f"Surrogate {model.name!r} is already registered")
    SURROGATES[model.name] = model
    return model
