"""Calibration and surrogate-skill audits of the binding pilot's random forest.

Two separate questions, deliberately kept apart:

1. `rf_calibration` — along an *acquisition trajectory*, is the forest's tree-dispersion
   sigma a usable uncertainty estimate? Reported as magnitude calibration
   (overconfidence ratio, interval coverage) and as error ranking.
2. `surrogate_skill` — as a function of *label count alone*, how much predictive skill
   does the surrogate have, and does a linear model do as well? This deliberately
   extends past the pilot's 128-label budget so the pilot's operating point can be
   placed on the curve.

Both evaluate on the held-out complement of the observed set, never on observed labels.
Sigma is a tree-dispersion heuristic, not a calibrated posterior; coverage figures
describe that heuristic and are not confidence intervals. Spearman correlations are
descriptive. Pool-to-pool spread is variation across sub-landscapes of one assay, not
biological replication.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from .common import ROOT, config, digest, provenance, write_json

SURROGATE_BUDGETS = (8, 16, 32, 64, 128, 256, 512, 1024)
# Log-spaced grid for the linear comparator. RidgeCV selects per budget by leave-one-out,
# so the small-label end is not forced to share a penalty with the 1024-label end.
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def load_pool(pool):
    path = ROOT / "data/processed/binding"
    split = np.load(path / "splits.npz")["evaluation"][pool]
    features = np.load(path / "features.npy")[split]
    outcomes = pd.read_csv(path / "outcomes.csv", usecols=["DMS_score"]).DMS_score.to_numpy()[split]
    design = np.eye(20, dtype=np.float32)[features].reshape(len(features), -1)
    return design, outcomes


def forest(design, outcomes, observed, seed, trees):
    """Same estimator the pilot's acquisition uses, so the audit describes that model."""
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                  max_features=1.0, n_jobs=1, random_state=seed)
    observed = np.asarray(observed)
    model.fit(design[observed], outcomes[observed])
    estimates = np.array([tree.predict(design) for tree in model.estimators_])
    return estimates.mean(axis=0), estimates.std(axis=0)


def calibration_metrics(outcomes, mean, sigma, observed):
    held_out = np.setdiff1d(np.arange(len(outcomes)), observed)
    error = np.abs(mean[held_out] - outcomes[held_out])
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_sigma = float(sigma[held_out].mean())
    return {"n_eval": len(held_out), "rmse": rmse, "mean_sigma": mean_sigma,
            # Above 1 means sigma understates typical error; it is a ratio, not a test.
            "overconfidence_ratio": rmse / mean_sigma,
            "coverage_1sigma": float(np.mean(error <= sigma[held_out])),
            "coverage_2sigma": float(np.mean(error <= 2 * sigma[held_out])),
            "spearman_sigma_abserr": float(spearmanr(sigma[held_out], error).statistic),
            "spearman_pred_truth": float(spearmanr(mean[held_out], outcomes[held_out]).statistic)}


def trajectory(design, outcomes, regime, seed, trees, budgets, batch):
    """Acquire to each budget under one regime, auditing the forest at every checkpoint.

    The greedy regime takes the highest predicted candidates in index order, without the
    reproducible tie-break the pilot's `pick_batch` applies. On this assay that matters:
    at low budget the forest produces large exactly-tied plateaus, so this regime is
    biased toward low indices where the pilot's is not. Kept deliberately, so the audit
    reports the acquisition rule in its plainest form.
    """
    rng = np.random.default_rng(seed)
    observed = rng.choice(len(design), size=budgets[0], replace=False).tolist()
    rows = []
    for budget in budgets:
        while len(observed) < budget:
            available = np.setdiff1d(np.arange(len(design)), observed)
            count = min(batch, budget - len(observed))
            if regime == "random":
                observed += rng.choice(available, size=count, replace=False).tolist()
            elif regime == "greedy":
                mean, _ = forest(design, outcomes, observed, seed, trees)
                order = np.argsort(-mean[available], kind="stable")[:count]
                observed += available[order].tolist()
            else:
                raise ValueError(f"Unknown acquisition regime {regime!r}")
        mean, sigma = forest(design, outcomes, observed, seed, trees)
        rows.append({"budget": budget, "regime": regime,
                     **calibration_metrics(outcomes, mean, sigma, observed)})
    return rows


def rf_calibration():
    c = config()
    bc = c["binding"]
    records = []
    for pool in range(bc["evaluation_pools"]):
        design, outcomes = load_pool(pool)
        for regime in ("random", "greedy"):
            for row in trajectory(design, outcomes, regime, c["seed"] + pool,
                                  bc["forest_trees"], bc["checkpoints"], bc["batch_size"]):
                records.append({"pool": pool, **row})
    return pd.DataFrame(records)


def surrogate_skill():
    c = config()
    bc = c["binding"]
    records = []
    for pool in range(bc["evaluation_pools"]):
        design, outcomes = load_pool(pool)
        # Threshold membership matches the pilot's evaluator: a quantile cut with ties kept.
        threshold = np.quantile(outcomes, 1 - bc["top_fraction"])
        hits = outcomes >= threshold
        k = int(hits.sum())
        # One stream, an independent draw per budget. These are label-count curves, so a
        # budget must not inherit the particular labels an earlier budget happened to draw.
        rng = np.random.default_rng(c["seed"] + pool)
        for budget in SURROGATE_BUDGETS:
            if budget >= len(design):
                continue
            observed = rng.choice(len(design), size=budget, replace=False)
            held_out = np.setdiff1d(np.arange(len(design)), observed)
            for name in ("forest", "ridge"):
                if name == "forest":
                    mean, sigma = forest(design, outcomes, observed, c["seed"] + pool,
                                         bc["forest_trees"])
                    ranking = float(spearmanr(sigma[held_out],
                                              np.abs(mean[held_out] - outcomes[held_out])).statistic)
                else:
                    model = RidgeCV(alphas=RIDGE_ALPHAS)
                    model.fit(design[observed], outcomes[observed])
                    mean = model.predict(design)
                    ranking = float("nan")  # A point predictor supplies no dispersion.
                top = held_out[np.argsort(-mean[held_out], kind="stable")[:k]]
                records.append({"pool": pool, "budget": budget, "model": name, "k": k,
                                "spearman_pred_truth": float(spearmanr(mean[held_out],
                                                             outcomes[held_out]).statistic),
                                "spearman_sigma_abserr": ranking,
                                "top1pct_precision": float(hits[top].mean())})
    return pd.DataFrame(records)


def run_calibration():
    directory = ROOT / "runs/calibration"
    directory.mkdir(parents=True, exist_ok=True)
    rf = rf_calibration()
    rf.to_csv(directory / "rf_calibration.csv", index=False)
    rf.groupby(["regime", "budget"])[
        ["overconfidence_ratio", "coverage_1sigma", "coverage_2sigma",
         "spearman_sigma_abserr", "spearman_pred_truth", "rmse", "mean_sigma"]
    ].agg(["mean", "std"]).to_csv(directory / "rf_calibration_summary.csv")
    skill = surrogate_skill()
    skill.to_csv(directory / "surrogate_skill.csv", index=False)
    skill.groupby(["model", "budget"])[
        ["spearman_pred_truth", "spearman_sigma_abserr", "top1pct_precision"]
    ].agg(["mean", "std"]).to_csv(directory / "surrogate_skill_summary.csv")
    write_json(directory / "status.json", {
        "rf_calibration_rows": len(rf), "surrogate_skill_rows": len(skill),
        "surrogate_budgets": list(SURROGATE_BUDGETS),
        "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
        "dataset_manifest_sha256": digest(ROOT / "data/processed/binding/manifest.json"),
        "provenance": provenance(),
        "interpretation": "Audits of the pilot surrogate only; no biological or efficacy claim"})
    print(f"rf_calibration rows={len(rf)} surrogate_skill rows={len(skill)} -> {directory}")
