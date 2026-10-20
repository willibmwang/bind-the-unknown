"""What distinguishes one acquisition arm from another, in one place.

Four arms share a replay engine and differ only in where their data lives, which column
holds the outcome, and what they stamp onto a result. That difference was previously
expressed four times -- once per `run_*` function in `binding`, once per `prepare_*` in
`data`, and again as an ad hoc dict in `sweep` -- so adding the display arm meant editing
three files and a fourth to summarise it, and the copies were already drifting: `sweep`
knew three arms, `summarize` five experiment names, and only `binding` knew that the
display arm's outcome column is not called `DMS_score`.

A registry rather than a class hierarchy, because the arms differ in *data* and not in
behaviour. There is one replay, one staging routine and one summariser; what varies is a
handful of strings and two small functions, which is a record, not a subclass.

Nothing here reads a file or imports numpy. It is the description of the arms, so that a
module needing only a path does not pay for the scientific stack to get one.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


def _target_fields(spec, manifest):
    return {"target": spec["target"], "pathogen": spec["pathogen"],
            "family": spec["family"], "endpoint": spec["endpoint"]}


def _method_fields(spec, manifest):
    return {"protein": spec["protein"], "interface": spec["interface"],
            "endpoint": spec["endpoint"],
            "role": "method development substrate; not a pathogen target proposal"}


def _display_fields(spec, manifest):
    return {"target": spec["target"], "modality": spec["modality"],
            "endpoint": spec["endpoint"],
            "noncanonical_features": bool(spec.get("monomer_separator"))}


@dataclass(frozen=True)
class Arm:
    """One acquisition arm: where its data is, and what a result from it must carry."""

    name: str
    processed: str
    runs: str
    outcome_column: str
    config: Optional[str] = None
    """Config filename under `hpc_pilots/configs/`. None means the shared `today.json`."""
    section: Optional[str] = None
    """Key within the config holding the replay parameters, when the config is shared."""
    per_landscape: bool = True
    """False for the single staged GB1 pilot, which has no landscape to name."""
    fraction_from_spec: bool = True
    """Whether `top_fraction` is per-landscape or one value for the whole arm."""
    describe: Optional[Callable] = None
    """(spec, manifest) -> extra fields stamped onto every result. None for the pilot."""
    warning: Optional[str] = None
    """Carried in every result, so a caveat travels with the number rather than the paper."""
    extra_hashes: tuple = field(default_factory=tuple)
    """(result key, config filename) pairs hashed into every result for provenance."""
    identity: tuple = field(default_factory=tuple)
    """Result keys carried into each summary row beyond the common ones, so a row says
    which substrate and which endpoint produced it without a lookup."""

    def staged(self, landscape=None):
        return f"{self.processed}/{landscape}" if self.per_landscape else self.processed

    def config_path(self):
        return f"hpc_pilots/configs/{self.config or 'today.json'}"

    def filename(self, landscape, policy, pool, delay, smoke=False, control=None):
        """Result filename. A control arm gets its own, so it can never overwrite a real run."""
        parts = [f"{landscape}_" if self.per_landscape else "", f"{policy}",
                 f"_pool{pool:02d}", f"_delay{delay}", f"_{control}" if control else ""]
        return f"{'smoke_' if smoke else ''}{''.join(parts)}.json"


ARMS = {
    "binding": Arm(
        name="binding", processed="data/processed/binding", runs="runs/binding",
        outcome_column="DMS_score", section="binding", per_landscape=False,
        fraction_from_spec=False, identity=(),
        extra_hashes=(("config_sha256", "today.json"),)),
    "target": Arm(
        name="target", processed="data/processed/targets", runs="runs/targets",
        outcome_column="DMS_score", config="targets.json", fraction_from_spec=False,
        describe=_target_fields, identity=("target", "endpoint"),
        warning="Do not pool metrics across landscapes with different assay endpoints, and "
                "do not compare these top-fraction figures with the GB1 1% figures.",
        extra_hashes=(("targets_config_sha256", "targets.json"),
                      ("config_sha256", "today.json"))),
    "method": Arm(
        name="method", processed="data/processed/method", runs="runs/method",
        outcome_column="DMS_score", config="method.json", describe=_method_fields,
        identity=("protein", "interface", "endpoint"),
        warning="Do not pool metrics across landscapes with different assay endpoints or "
                "different pool geometries; the multiple over random is bounded by the "
                "labelled fraction of the pool.",
        extra_hashes=(("method_config_sha256", "method.json"),)),
    "display": Arm(
        name="display", processed="data/processed/display", runs="runs/display",
        outcome_column="outcome", config="display.json", describe=_display_fields,
        identity=("target", "modality", "endpoint", "noncanonical_features"),
        warning="Enrichment is not affinity, functional inhibition or antiviral activity. "
                "Do not pool metrics across landscapes with different endpoints or geometries.",
        extra_hashes=(("display_config_sha256", "display.json"),)),
}


def arm(name):
    if name not in ARMS:
        raise ValueError(f"Unknown arm {name!r}; known arms are {sorted(ARMS)}")
    return ARMS[name]
