"""Analysis of experiment E3 (docs/design.md, section 13.3): amplitude CRPS, H3.2, amplitude intervals, growth of the
speed standard deviation along the platoon, collisions.

    python scripts/analyze_e3.py                                  # artifacts/e3
    python scripts/analyze_e3.py --root artifacts/e3_pilot
    python scripts/analyze_e3.py --root artifacts/e3 --n-boot 2000 --no-figures

Writes ``<root>/tables/``: profile_index.csv (one row per variant, model, profile and vehicle index: the per-profile
scores every other table is aggregated from), amplitude_crps.csv, h32.csv, amplitude_coverage.csv, growth.csv,
collisions.csv, analysis_info.json, summary.md (the printed summary as markdown); the figures of
``cf_unc.report.figures_e3`` into ``<root>/figures`` (for the set of ``--set``); prints a summary: per variant the
overview of the set and the H3.2 block per driver mode (ACC, Human, all; a mode with fewer than
``h32.min_profiles`` profiles is listed but not judged, design.md 14.1). Settings (profiles file, sets, calibration,
bootstrap, H3.2 thresholds) come from ``<root>/config.yaml`` (written by the runner) with the defaults of
``configs/experiment/e3.yaml`` for missing keys. Statistical conventions: ``cf_unc.platoon.evaluate``. Windows:
``response`` = profile + tail (design.md 13.3, the primary definition), ``profile`` = the window of the measured data
(sensitivity). Subsets (column ``mode``): all profiles and the driver modes of the followers; an H3.2 verdict needs
``h32.min_profiles`` (5) profiles, else it is inconclusive in h32.csv. Filled series of
``configs/empirical_growth.yaml`` are added to growth.csv (variant ``literature``).

Signal variants: every variant with units (``raw``, ``harmonised`` and the E3-pooled variants ``pool`` and
``lodo_openacc`` of design.md 17.1, or any other) flows through every table and figure. A variant whose leader speed
was smoothed again (``profile_window``) has the file ``<root>/profiles__<variant>.yaml`` written by the runner: its
leader amplitude ``A_0`` (ratio quantity, column ``leader_amplitude``) comes from there; the measured follower
statistics and the driver modes are those of the profiles file for every variant. The leader amplitude stored in
every unit table must agree with the profile's (ValueError otherwise: the units were simulated with another leader).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from cf_unc.constants import DT
from cf_unc.experiments.e3 import available_units, select_profile_ids, variant_profiles_path
from cf_unc.platoon.evaluate import (
    boot_seed,
    coverage_by_method,
    fair_crps,
    growth_curve,
    growth_exponent,
    second_difference_share,
    h32_verdict,
    mean_with_ci,
    profile_mean,
    profile_order,
    relative_reduction,
    sample_quantiles,
)
from cf_unc.platoon.profiles import MODES, Profile, load_profiles, profiles_digest
from cf_unc.report.figures_e3 import MODEL_LABEL_E3, VARIANT_LABEL, VARIANT_ORDER, make_e3_figures

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS: dict[str, Any] = {  # mirror of configs/experiment/e3.yaml
    "profiles": "configs/platoon/profiles.yaml",
    "empirical_growth": "configs/empirical_growth.yaml",
    "platoon": {"burn_in_s": 30.0},
    "sets": {"headline": 10, "extended": None},
    "calibration": {"alpha": 0.1, "gamma": 0.01, "n_min": 5},
    "bootstrap": {"n_boot": 1000, "ci": 0.95, "seed": 0},
    "h32": {"support_min": 0.20, "refute_max": 0.05, "reference": "idm_white_noise", "references_extra": ["lstm_const_sigma"],
            "min_profiles": 5},  # fewer profiles than this in a subset: the verdict is "inconclusive"
}
WINDOWS = {"response": ("amplitude", "sd"), "profile": ("amplitude_profile", "sd_profile")}
SUBSETS = ("all", *MODES)
METHODS = ("raw", "static_loo", "aci_online")
EMPIRICAL_MIN_PROFILES = 3  # an index of the empirical growth curve needs this many profiles
EMPIRICAL_FIT_SHARE = 0.5  # the empirical exponent is fitted over the indices measured by this share of the profiles
LEADER_TOL = 1e-3  # m/s: leader amplitude of a unit table against its profile's (stored rounded to 0.1 mm/s)
COLUMNS: dict[str, list[str]] = {
    "profile_index": [
        "variant", "model", "profile", "index", "n_seeds", "n_samples", "emp_amplitude", "emp_sd", "leader_amplitude",
        "amp_mean", "q05", "q50", "q95", "crps_response", "crps_profile", "crps_ratio_response", "crps_ratio_profile",
        "sd_response", "sd_profile", "collided", "collided_burn_in", "min_gap_mean", "collided_any", "collided_burn_in_any",
    ],
    "amplitude_crps": [
        "variant", "model", "set", "mode", "window", "index", "crps", "crps_lo", "crps_hi", "crps_ratio", "crps_ratio_lo",
        "crps_ratio_hi", "n_profiles",
    ],
    "h32": [
        "variant", "set", "mode", "window", "quantity", "reference", "model", "reduction", "lo", "hi", "verdict",
        "verdict_point", "crps_model", "crps_reference", "n_profiles",
    ],
    "amplitude_coverage": [
        "variant", "model", "set", "index", "method", "coverage", "coverage_lo", "coverage_hi", "nominal", "width_median",
        "q_mean", "n_profiles",
    ],
    "growth": [
        "variant", "model", "set", "window", "index", "sd", "sd_lo", "sd_hi", "n_profiles", "beta", "beta_lo", "beta_hi",
        "log_sd1", "share_neg_d2", "fitted",
    ],
    "collisions": ["variant", "model", "set", "index", "collided", "collided_burn_in", "profiles_with_collision", "min_gap_mean",
                   "n_rollouts"],
}  # fmt: skip


def _resolve(path: str | Path) -> Path:
    """A relative path is taken from the working directory if it exists there, else from the repository root."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


def load_settings(root: Path, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """DEFAULTS <- the same keys of ``<root>/config.yaml`` <- ``overrides`` (per section)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    path = root / "config.yaml"
    stored = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    for key in out:
        if key in stored and stored[key] is not None:
            out[key] = {**out[key], **stored[key]} if isinstance(out[key], dict) else stored[key]
    for key, values in (overrides or {}).items():
        out[key].update({k: v for k, v in values.items() if v is not None})
    return out


def _frame(rows: list[dict[str, Any]], name: str) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS[name])


# ------------------------------------------------------------------------------------------- loading
def _unit_arrays(path: str, burn_in: int) -> dict[str, np.ndarray]:
    """(V, R) arrays of a unit table (rows vehicle-major), the collision flags (V, R) - any time, during the burn-in
    (first collision at step <= ``burn_in``) - and the per-rollout platoon collision flag (R,); ``leader_amplitude``
    (1,): the amplitude of the simulated leader."""
    t = pd.read_parquet(path)
    n_v = int(t["vehicle"].max())
    n_r = len(t) // n_v
    out = {c: t[c].to_numpy(np.float64).reshape(n_v, n_r) for c in ("amplitude", "sd", "amplitude_profile", "sd_profile",
                                                                     "min_gap")}  # fmt: skip
    first = t["first_collision"].to_numpy().reshape(n_v, n_r)
    out["collided"] = (first >= 0).astype(np.float64)
    out["collided_burn_in"] = ((first >= 1) & (first <= burn_in)).astype(np.float64)
    out["collided_any"] = (first >= 0).any(axis=0).astype(np.float64)
    out["collided_burn_in_any"] = out["collided_burn_in"].any(axis=0).astype(np.float64)
    out["leader_amplitude"] = t["leader_amplitude"].to_numpy(np.float64)[:1]
    return out


def variant_profiles(root: Path, variants: Sequence[str], profiles: Sequence[Profile]) -> dict[str, dict[str, Profile]]:
    """id -> profile per variant with a ``profiles__<variant>.yaml`` (re-smoothed leader, design.md 17.1). The file
    must derive from the profiles file of the analysis (``source_digest``, ids, follower statistics, modes)."""
    base = {p.id: p for p in profiles}
    digest = profiles_digest(profiles)
    out: dict[str, dict[str, Profile]] = {}
    for variant in variants:
        path = variant_profiles_path(root, variant)
        if not path.exists():
            continue
        loaded, meta = load_profiles(path)
        if meta.get("source_digest") != digest or [p.id for p in loaded] != list(base):
            raise ValueError(f"{path} was derived from other profiles (source_digest {meta.get('source_digest')}, the "
                             f"profiles file of the analysis has {digest})")  # fmt: skip
        for p in loaded:
            b = base[p.id]
            if (p.mode, p.n_followers) != (b.mode, b.n_followers) or not (
                    np.array_equal(p.follower_amplitude, b.follower_amplitude) and np.array_equal(p.follower_sd, b.follower_sd)):
                raise ValueError(f"{path}: profile {p.id} differs from the profiles file outside the leader speed")
        out[str(variant)] = {p.id: p for p in loaded}
    return out


def profile_scores(listing: pd.DataFrame, profiles: Mapping[str, Profile], burn_in: int, log=print,  # noqa: ANN001
                   by_variant: Mapping[str, Mapping[str, Profile]] | None = None) -> pd.DataFrame:  # fmt: skip
    """The profile_index table: per (variant, model, profile, vehicle index) the scores of the pooled model law;
    ``burn_in`` = burn-in steps of the runs (collisions before the profile starts); ``by_variant``: the profiles of
    a variant whose leader was smoothed again (``variant_profiles``), ``profiles`` for every other variant."""
    rows: list[dict[str, Any]] = []
    for (variant, model), grp in listing.groupby(["variant", "model"], sort=False):
        own = (by_variant or {}).get(str(variant), profiles)
        for profile_id, units in grp.groupby("profile", sort=False):
            p = own.get(str(profile_id))
            if p is None:
                continue
            parts = [_unit_arrays(path, burn_in) for path in units.sort_values("seed")["path"]]
            arr = {k: np.concatenate([u[k] for u in parts], axis=-1) for k in parts[0]}
            n_v, n_s = arr["amplitude"].shape
            emp_amp, emp_sd = np.full(n_v, np.nan), np.full(n_v, np.nan)
            m = min(n_v, p.n_followers)
            emp_amp[:m], emp_sd[:m] = p.follower_amplitude[:m], p.follower_sd[:m]
            a0 = float(p.leader_amplitude)
            simulated = arr.pop("leader_amplitude")
            if np.any(np.abs(simulated - a0) > LEADER_TOL):
                raise ValueError(
                    f"{variant}/{model}/{p.id}: the leader amplitude of the unit tables ({simulated.min():.4f} m/s) differs "
                    f"from the profile's ({a0:.4f} m/s): the units were simulated with another leader (a variant with "
                    f"profile_window needs its {variant_profiles_path('<root>', str(variant)).name})"
                )  # fmt: skip
            crps = {w: fair_crps(arr[cols[0]], emp_amp) for w, cols in WINDOWS.items()}
            q = sample_quantiles(arr["amplitude"])
            for i in range(n_v):
                rows.append({
                    "variant": variant, "model": model, "profile": p.id, "index": i + 1, "n_seeds": len(parts),
                    "n_samples": n_s, "emp_amplitude": emp_amp[i], "emp_sd": emp_sd[i], "leader_amplitude": a0,
                    "amp_mean": float(arr["amplitude"][i].mean()), "q05": q[i, 0], "q50": q[i, 1], "q95": q[i, 2],
                    "crps_response": crps["response"][i], "crps_profile": crps["profile"][i],
                    "crps_ratio_response": crps["response"][i] / a0, "crps_ratio_profile": crps["profile"][i] / a0,
                    "sd_response": float(arr["sd"][i].mean()), "sd_profile": float(arr["sd_profile"][i].mean()),
                    "collided": float(arr["collided"][i].mean()), "collided_burn_in": float(arr["collided_burn_in"][i].mean()),
                    "min_gap_mean": float(arr["min_gap"][i].mean()), "collided_any": float(arr["collided_any"].mean()),
                    "collided_burn_in_any": float(arr["collided_burn_in_any"].mean()),
                })  # fmt: skip
        log(f"  {variant} / {model}: {grp['profile'].nunique()} profiles, seeds {sorted(grp['seed'].unique().tolist())}")
    return _frame(rows, "profile_index")


def _pivot(pi: pd.DataFrame, column: str, ids: Sequence[str], n_v: int) -> np.ndarray:
    """(P, V) values of ``column`` for the profiles ``ids`` (rows) and indices 1..V (NaN where absent)."""
    out = np.full((len(ids), n_v), np.nan)
    pos = {p: k for k, p in enumerate(ids)}
    sub = pi[pi["profile"].isin(pos)]
    out[sub["profile"].map(pos).to_numpy(), sub["index"].to_numpy() - 1] = sub[column].to_numpy(np.float64)
    return out


# -------------------------------------------------------------------------------------------- tables
def subsets(ids: Sequence[str], profiles: Mapping[str, Profile]) -> dict[str, list[str]]:
    out = {"all": list(ids)}
    for mode in MODES:
        sel = [i for i in ids if profiles[i].mode == mode]
        if sel:
            out[mode] = sel
    return out


def crps_tables(pi: pd.DataFrame, sets: Mapping[str, list[str]], profiles: Mapping[str, Profile], s: Mapping[str, Any]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """amplitude_crps and h32 tables."""
    b, h = s["bootstrap"], s["h32"]
    n_boot, ci, base = int(b["n_boot"]), float(b["ci"]), int(b["seed"])
    references = [str(h["reference"]), *[str(r) for r in (h.get("references_extra") or [])]]
    crps_rows, h32_rows = [], []
    for variant in pi["variant"].unique():
        pv = pi[pi["variant"] == variant]
        n_v = int(pv["index"].max())
        max_emp = int(pv.loc[pv["emp_amplitude"].notna(), "index"].max()) if pv["emp_amplitude"].notna().any() else 0
        models = list(dict.fromkeys(pv["model"]))
        for set_name, set_ids in sets.items():
            for mode, ids in subsets(set_ids, profiles).items():
                for window in WINDOWS:
                    per_model = {}
                    for model in models:
                        pm = pv[pv["model"] == model]
                        amp = _pivot(pm, f"crps_{window}", ids, n_v)
                        ratio = _pivot(pm, f"crps_ratio_{window}", ids, n_v)
                        per_model[model] = (amp, ratio)
                        scopes: list[tuple[str, np.ndarray, np.ndarray]] = [
                            (str(i), amp[:, i - 1], ratio[:, i - 1]) for i in range(1, max_emp + 1)
                        ]
                        scopes += [("all", profile_mean(amp), profile_mean(ratio)),
                                   ("1-4", profile_mean(amp, range(1, 5)), profile_mean(ratio, range(1, 5)))]  # fmt: skip
                        for index, a, r in scopes:
                            seed = boot_seed(base, "crps", set_name, mode, window, index)
                            ea, er = mean_with_ci(a, ids, n_boot, seed, ci), mean_with_ci(r, ids, n_boot, seed, ci)
                            if ea["n"] == 0:
                                continue
                            crps_rows.append({
                                "variant": variant, "model": model, "set": set_name, "mode": mode, "window": window,
                                "index": index, "crps": ea["estimate"], "crps_lo": ea["lo"], "crps_hi": ea["hi"],
                                "crps_ratio": er["estimate"], "crps_ratio_lo": er["lo"], "crps_ratio_hi": er["hi"],
                                "n_profiles": ea["n"],
                            })  # fmt: skip
                    for quantity, k in (("amplitude", 0), ("ratio", 1)):
                        seed = boot_seed(base, "h32", set_name, mode, window, quantity)
                        for ref in references:
                            if ref not in per_model:
                                continue
                            ref_scores = profile_mean(per_model[ref][k])
                            for model in models:
                                if model == ref:
                                    continue
                                red = relative_reduction(profile_mean(per_model[model][k]), ref_scores, ids, n_boot, seed, ci)
                                verdict, point = h32_verdict(red["estimate"], red["lo"], red["hi"], float(h["support_min"]),
                                                             float(h["refute_max"]))  # fmt: skip
                                if red["n_profiles"] < int(h["min_profiles"]):  # too few profiles for a verdict
                                    verdict = point = "inconclusive"
                                h32_rows.append({
                                    "variant": variant, "set": set_name, "mode": mode, "window": window,
                                    "quantity": quantity, "reference": ref, "model": model, "reduction": red["estimate"],
                                    "lo": red["lo"], "hi": red["hi"], "verdict": verdict, "verdict_point": point,
                                    "crps_model": red["crps_model"], "crps_reference": red["crps_reference"],
                                    "n_profiles": red["n_profiles"],
                                })  # fmt: skip
    return _frame(crps_rows, "amplitude_crps"), _frame(h32_rows, "h32")


def coverage_table(pi: pd.DataFrame, sets: Mapping[str, list[str]], extended: list[str], s: Mapping[str, Any]) -> pd.DataFrame:
    """amplitude_coverage: raw, leave-one-profile-out static and ACI-online coverage of the 90 % amplitude interval;
    the corrections are calibrated over the profiles of the extended set (per index)."""
    c, b = s["calibration"], s["bootstrap"]
    alpha, gamma, n_min = float(c["alpha"]), float(c["gamma"]), int(c["n_min"])
    n_boot, ci, base = int(b["n_boot"]), float(b["ci"]), int(b["seed"])
    order_ids = profile_order(extended, boot_seed(base, "aci_order"))
    order = [extended.index(i) for i in order_ids]
    rows = []
    for (variant, model), pm in pi.groupby(["variant", "model"], sort=False):
        n_v = int(pm["index"].max())
        cols = {k: _pivot(pm, k, extended, n_v) for k in ("emp_amplitude", "q05", "q50", "q95")}
        max_emp = int(np.max(np.nonzero(np.isfinite(cols["emp_amplitude"]).any(axis=0))[0], initial=-1)) + 1
        cov = {m: np.full((len(extended), n_v), np.nan) for m in METHODS}
        width = {m: np.full((len(extended), n_v), np.nan) for m in METHODS}
        qs = {m: np.full((len(extended), n_v), np.nan) for m in METHODS}
        for i in range(max_emp):
            res = coverage_by_method(cols["emp_amplitude"][:, i], cols["q05"][:, i], cols["q50"][:, i], cols["q95"][:, i],
                                     order, alpha, gamma, n_min)  # fmt: skip
            for m in METHODS:
                cov[m][:, i], width[m][:, i], qs[m][:, i] = res[m]["covered"], res[m]["width"], res[m]["q"]
        for set_name, ids in sets.items():
            rows_of = [extended.index(i) for i in ids if i in extended]
            set_ids = [extended[k] for k in rows_of]
            for m in METHODS:
                scopes = [(str(i + 1), cov[m][rows_of, i], width[m][rows_of, i], qs[m][rows_of, i]) for i in range(max_emp)]
                scopes.append(("all", profile_mean(cov[m][rows_of]), profile_mean(width[m][rows_of]),
                               profile_mean(qs[m][rows_of])))  # fmt: skip
                for index, cv, wd, qq in scopes:
                    est = mean_with_ci(cv, set_ids, n_boot, boot_seed(base, "coverage", set_name, index), ci)
                    if est["n"] == 0:
                        continue
                    finite = wd[np.isfinite(wd)]
                    rows.append({
                        "variant": variant, "model": model, "set": set_name, "index": index, "method": m,
                        "coverage": est["estimate"], "coverage_lo": est["lo"], "coverage_hi": est["hi"], "nominal": 1 - alpha,
                        "width_median": float(np.median(finite)) if finite.size else np.inf,
                        "q_mean": float(np.mean(qq[np.isfinite(qq)])) if np.isfinite(qq).any() else np.nan,
                        "n_profiles": est["n"],
                    })  # fmt: skip
    return _frame(rows, "amplitude_coverage")


def growth_table(pi: pd.DataFrame, sets: Mapping[str, list[str]], s: Mapping[str, Any]) -> pd.DataFrame:
    """growth: mean sd by vehicle index per model and window, the empirical curve (variant ``data``), exponents."""
    b = s["bootstrap"]
    n_boot, ci, base = int(b["n_boot"]), float(b["ci"]), int(b["seed"])
    rows = []

    def emit(variant: str, model: str, set_name: str, window: str, res: Mapping[str, Any]) -> None:
        for i, v in enumerate(res["curve"]):
            if res["n_profiles"][i] == 0:
                continue
            rows.append({
                "variant": variant, "model": model, "set": set_name, "window": window, "index": i + 1, "sd": v,
                "sd_lo": res["lo"][i], "sd_hi": res["hi"][i], "n_profiles": int(res["n_profiles"][i]),
                "beta": res["beta"], "beta_lo": res["beta_lo"], "beta_hi": res["beta_hi"], "log_sd1": res["log_sd1"],
                "share_neg_d2": res["share_neg_d2"], "fitted": bool(res["fitted"][i]),
            })  # fmt: skip

    for (variant, model), pm in pi.groupby(["variant", "model"], sort=False):
        n_v = int(pm["index"].max())
        for set_name, ids in sets.items():
            for window in WINDOWS:
                curves = _pivot(pm, f"sd_{window}", ids, n_v)
                curves = curves[np.isfinite(curves).any(axis=1)]
                if curves.size:
                    emit(variant, model, set_name, window, growth_curve(curves, n_boot, boot_seed(base, "growth", set_name, window), ci))
    first = pi.drop_duplicates(["profile", "index"])
    if len(first):
        n_v = int(first["index"].max())
        for set_name, ids in sets.items():
            curves = _pivot(first, "emp_sd", ids, n_v)
            curves = curves[np.isfinite(curves).any(axis=1)]
            if curves.size:
                fit_min = max(EMPIRICAL_MIN_PROFILES, int(math.ceil(EMPIRICAL_FIT_SHARE * curves.shape[0])))
                res = growth_curve(curves, n_boot, boot_seed(base, "growth", set_name, "profile"), ci,
                                   EMPIRICAL_MIN_PROFILES, fit_min_profiles=fit_min)  # fmt: skip
                emit("data", "empirical", set_name, "profile", res)
    return _frame(rows, "growth")


def literature_rows(path: Path) -> list[dict[str, Any]]:
    """growth rows of the filled series of ``configs/empirical_growth.yaml`` (variant ``literature``, set
    ``literature``, window ``literature:<quantity>``); none when the file is absent or empty."""
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    rows = []
    for name, spec in (data.get("series") or {}).items():
        index = np.asarray(spec.get("index") or [], dtype=np.float64)
        value = np.asarray(spec.get("value") or [], dtype=np.float64)
        if index.size == 0 or index.size != value.size:
            raise ValueError(f"{path}: series {name!r} needs index and value lists of equal length")
        beta, log_sd1 = growth_exponent(value, index)
        for i, v in zip(index, value):
            rows.append({
                "variant": "literature", "model": str(name), "set": "literature",
                "window": f"literature:{spec.get('quantity', 'sd')}", "index": int(i), "sd": float(v), "sd_lo": np.nan,
                "sd_hi": np.nan, "n_profiles": 0, "beta": beta, "beta_lo": np.nan, "beta_hi": np.nan, "log_sd1": log_sd1,
                "share_neg_d2": second_difference_share(value[np.argsort(index)]),
            })  # fmt: skip
    return rows


def collision_table(pi: pd.DataFrame, sets: Mapping[str, list[str]]) -> pd.DataFrame:
    """collisions: share of rollouts with a collision by vehicle index and of any pair of the platoon."""
    rows = []
    for (variant, model), pm in pi.groupby(["variant", "model"], sort=False):
        for set_name, ids in sets.items():
            sub = pm[pm["profile"].isin(ids)]
            if sub.empty:
                continue
            weights = sub["n_samples"].to_numpy(np.float64)
            for index, grp in sub.groupby("index"):
                w = grp["n_samples"].to_numpy(np.float64)
                rows.append({
                    "variant": variant, "model": model, "set": set_name, "index": str(index),
                    "collided": float(np.average(grp["collided"], weights=w)),
                    "collided_burn_in": float(np.average(grp["collided_burn_in"], weights=w)),
                    "profiles_with_collision": float((grp["collided"] > 0).mean()),
                    "min_gap_mean": float(np.average(grp["min_gap_mean"], weights=w)), "n_rollouts": int(w.sum()),
                })  # fmt: skip
            per_profile = sub.drop_duplicates("profile")
            w = per_profile["n_samples"].to_numpy(np.float64)
            rows.append({
                "variant": variant, "model": model, "set": set_name, "index": "any",
                "collided": float(np.average(per_profile["collided_any"], weights=w)),
                "collided_burn_in": float(np.average(per_profile["collided_burn_in_any"], weights=w)),
                "profiles_with_collision": float((per_profile["collided_any"] > 0).mean()),
                "min_gap_mean": float(np.average(sub["min_gap_mean"], weights=weights)), "n_rollouts": int(w.sum()),
            })  # fmt: skip
    return _frame(rows, "collisions")


# ------------------------------------------------------------------------------------------- entry
def analyze_e3(
    root: str | Path = "artifacts/e3",
    *,
    settings: Mapping[str, Mapping[str, Any]] | None = None,
    figures: bool = True,
    verbose: bool = True,
    set_name: str = "extended",
) -> dict[str, pd.DataFrame]:
    """Every E3 table under ``<root>/tables`` (module docstring), ``summary.md`` and the figures of the profile set
    ``set_name`` (the first set of the tables when it is absent); returns the tables by name."""
    started = time.perf_counter()
    root = _resolve(root)
    log = print if verbose else (lambda *a, **k: None)
    s = load_settings(root, settings)
    listing = available_units(root)
    if listing.empty:
        raise FileNotFoundError(f"no unit tables under {root}")
    profiles_list, meta = load_profiles(_resolve(s["profiles"]))
    profiles = {p.id: p for p in profiles_list}
    sets = {name: select_profile_ids(profiles_list, name, s["sets"]) for name in s["sets"]}
    present = set(listing["profile"])
    sets = {k: [i for i in v if i in present] for k, v in sets.items()}
    sets = {k: v for k, v in sets.items() if v}
    extended = sets.get("extended", sorted(present, key=[p.id for p in profiles_list].index))
    log(f"E3 analysis of {root}: {listing['variant'].nunique()} variant(s), {listing['model'].nunique()} models, "
        f"{len(present)} profiles, sets { {k: len(v) for k, v in sets.items()} }")  # fmt: skip
    by_variant = variant_profiles(root, list(dict.fromkeys(listing["variant"])), profiles_list)
    if by_variant:
        log(f"  leader smoothed again (profiles__<variant>.yaml): {sorted(by_variant)}")
    pi = profile_scores(listing, profiles, int(round(float(s["platoon"]["burn_in_s"]) / DT)), log, by_variant)
    crps, h32 = crps_tables(pi, sets, profiles, s)
    out = {
        "profile_index": pi,
        "amplitude_crps": crps,
        "h32": h32,
        "amplitude_coverage": coverage_table(pi, sets, extended, s),
        "growth": growth_table(pi, sets, s),
        "collisions": collision_table(pi, sets),
    }
    literature = literature_rows(_resolve(s["empirical_growth"]))
    if literature:
        out["growth"] = pd.concat([out["growth"], _frame(literature, "growth")], ignore_index=True)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
    reference, min_profiles = str(s["h32"]["reference"]), int(s["h32"]["min_profiles"])
    summary = format_summary(out, set_name, reference, min_profiles, markdown=True, title=f"E3 summary: {root.name}")
    (table_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    paths = make_e3_figures(root, reference=reference, set_name=set_name) if figures else []
    info = {"root": str(root), "settings": s, "sets": sets, "profiles_digest": meta.get("digest"), "summary_set": set_name,
            "figures": [p.name for p in paths], "seconds": time.perf_counter() - started}  # fmt: skip
    (table_dir / "analysis_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"tables written to {table_dir} ({len(paths)} figures) in {info['seconds']:.0f} s")
    return out


def _fmt(x: Any, digits: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def _rows(frame: pd.DataFrame, **keys: Any) -> pd.DataFrame:
    """Rows of ``frame`` whose columns equal the given values."""
    mask = np.ones(len(frame), dtype=bool)
    for column, value in keys.items():
        mask &= (frame[column] == value).to_numpy()
    return frame[mask]


def _interval(row: pd.Series | None, est: str, lo: str, hi: str) -> str:
    return "-" if row is None else f"{_fmt(row[est])} [{_fmt(row[lo])}, {_fmt(row[hi])}]"


SUMMARY_MODES = ("ACC", "Human", "all", "mixed")  # order of the per-mode H3.2 block (design.md 14.1)
NOT_JUDGED = "not judged"  # verdict cell of a mode with fewer than h32.min_profiles profiles


def _model_order(names: Sequence[str], reference: str) -> list[str]:
    """The reference first, then the models in MODEL_LABEL_E3 order, then the rest sorted."""
    present = list(dict.fromkeys(str(n) for n in names))
    order = [m for m in MODEL_LABEL_E3 if m in present] + sorted(m for m in present if m not in MODEL_LABEL_E3)
    return [reference, *[m for m in order if m != reference]] if reference in order else order


def _crps_cells(rows: pd.DataFrame) -> tuple[str, str]:
    """Cells 'CRPS 1-4' and 'CRPS all' (estimate [95 % interval]) of one model's amplitude_crps rows."""
    m = rows.set_index(rows["index"].astype(str))
    c14 = m.loc["1-4"] if "1-4" in m.index else None
    call = m.loc["all"] if "all" in m.index else None
    return _interval(c14, "crps", "crps_lo", "crps_hi"), _interval(call, "crps", "crps_lo", "crps_hi")


def summary_blocks(
    out: Mapping[str, pd.DataFrame], set_name: str = "extended", reference: str = "idm_white_noise", min_profiles: int = 5
) -> tuple[list[dict[str, Any]], list[str]]:
    """The tables of the summary and its closing lines. Per variant: the overview of the set (amplitude CRPS over the
    indices 1-4 and over all measured indices, H3.2 against the reference, growth exponent of the response window,
    collision share of any pair of the platoon) and the H3.2 block per driver mode (ACC, Human, all, mixed: same CRPS,
    reduction and verdict columns; a mode with fewer than ``min_profiles`` profiles is listed, not judged). Closing
    lines: the growth exponent of the measured platoons per set. A table is ``{"title", "level", "columns", "rows"}``
    with ``columns`` = (header, width, alignment) of the text rendering."""
    crps, h32, growth, col = out["amplitude_crps"], out["h32"], out["growth"], out["collisions"]
    h32_head = "H3.2 vs " + reference
    overview_cols = [("model", 18, "<"), ("CRPS 1-4", 20, ">"), ("CRPS all", 20, ">"), (h32_head, 27, ">"),
                     ("verdict", 12, ">"), ("beta", 20, ">"), ("d2<0", 5, ">"), ("collided", 8, ">")]  # fmt: skip
    mode_cols = [("mode", 6, "<"), ("profiles", 8, ">"), ("model", 18, "<"), ("CRPS 1-4", 20, ">"), ("CRPS all", 20, ">"),
                 (h32_head, 27, ">"), ("verdict", 12, ">")]  # fmt: skip
    blocks: list[dict[str, Any]] = []
    sets = [set_name] if set_name in set(crps["set"]) else sorted(set(crps["set"]))
    present = list(dict.fromkeys(crps["variant"]))
    variants = [v for v in VARIANT_ORDER if v in present] + [v for v in present if v not in VARIANT_ORDER]
    for variant in variants:  # recorded signals first: the headline variant (design.md 14.1)
        named = f"{variant} ({VARIANT_LABEL[variant]})" if variant in VARIANT_LABEL else str(variant)
        for sn in sets:
            sel = _rows(crps, variant=variant, set=sn, mode="all", window="response")
            n_prof = int(sel["n_profiles"].max()) if len(sel) else 0
            rows = []
            for model in dict.fromkeys(sel["model"]):
                c14, call = _crps_cells(sel[sel["model"] == model])
                hr = _rows(h32, variant=variant, set=sn, mode="all", window="response", quantity="amplitude",
                           reference=reference, model=model)  # fmt: skip
                g = _rows(growth, variant=variant, model=model, set=sn, window="response")
                cc = _rows(col, variant=variant, model=model, set=sn, index="any")
                h_row = hr.iloc[0] if len(hr) else None
                g_row = g.iloc[0] if len(g) else None
                rows.append([
                    model, c14, call, _interval(h_row, "reduction", "lo", "hi") if h_row is not None else "reference",
                    h_row["verdict"] if h_row is not None else "-", _interval(g_row, "beta", "beta_lo", "beta_hi"),
                    _fmt(g_row["share_neg_d2"]) if g_row is not None else "-", _fmt(cc.iloc[0]["collided"], 3) if len(cc) else "-",
                ])  # fmt: skip
            blocks.append({"title": f"Variant {named}, set {sn} ({n_prof} profiles), window profile + tail", "level": 2,
                           "columns": overview_cols, "rows": rows})  # fmt: skip
            rows = []
            for mode in SUMMARY_MODES:
                msel = _rows(crps, variant=variant, set=sn, mode=mode, window="response")
                n_mode = msel.loc[msel["index"].astype(str) == "all", "n_profiles"]
                if msel.empty or n_mode.empty:
                    continue
                n_mode = int(n_mode.max())
                for model in _model_order(msel["model"], reference):
                    c14, call = _crps_cells(msel[msel["model"] == model])
                    hr = _rows(h32, variant=variant, set=sn, mode=mode, window="response", quantity="amplitude",
                               reference=reference, model=model)  # fmt: skip
                    if model == reference:
                        reduction, verdict = "reference", "-"
                    elif len(hr):
                        reduction = _interval(hr.iloc[0], "reduction", "lo", "hi")
                        verdict = str(hr.iloc[0]["verdict"]) if n_mode >= min_profiles else NOT_JUDGED
                    else:
                        reduction, verdict = "-", "-"
                    rows.append([mode, str(n_mode), model, c14, call, reduction, verdict])
            if rows:
                blocks.append({
                    "title": f"H3.2 by driver mode: variant {variant}, set {sn}, window profile + tail (a verdict needs "
                             f">= {min_profiles} profiles; 'all' = pooled, secondary)",
                    "level": 3, "columns": mode_cols, "rows": rows,
                })  # fmt: skip
    notes: list[str] = []
    emp = growth[(growth["model"] == "empirical")]
    for sn in dict.fromkeys(emp["set"]):
        e = emp[(emp["set"] == sn) & emp["sd"].notna()]
        if e.empty:
            notes.append(f"empirical ({sn}): no index measured in {EMPIRICAL_MIN_PROFILES} or more profiles")
            continue
        f = e[e["fitted"].astype(bool)] if "fitted" in e else e
        notes.append(f"empirical ({sn}): beta {_interval(e.iloc[0], 'beta', 'beta_lo', 'beta_hi')} fitted over indices "
                     f"{int(f['index'].min())}..{int(f['index'].max())} (measured by >= {EMPIRICAL_FIT_SHARE:.0%} of the "
                     f"profiles); curve to index {int(e['index'].max())} (profile window)")  # fmt: skip
    return blocks, notes


def format_summary(
    out: Mapping[str, pd.DataFrame],
    set_name: str = "extended",
    reference: str = "idm_white_noise",
    min_profiles: int = 5,
    *,
    markdown: bool = False,
    title: str | None = None,
) -> str:
    """The summary of :func:`summary_blocks` as fixed-width text (printed) or as markdown (``tables/summary.md``)."""
    blocks, notes = summary_blocks(out, set_name, reference, min_profiles)
    if not markdown:
        lines: list[str] = []
        for b in blocks:
            lines.append("\n" + b["title"])
            table = [[c[0] for c in b["columns"]], *b["rows"]]
            widths = [max(w, *(len(r[k]) for r in table)) for k, (_, w, _) in enumerate(b["columns"])]
            for cells in table:
                lines.append(" ".join(f"{cell:{c[2]}{w}s}" for cell, c, w in zip(cells, b["columns"], widths)).rstrip())
        return "\n".join([*lines, *notes])
    lines = [f"# {title or 'E3 summary'}", "",
             "Window profile + tail. CRPS 1-4 / CRPS all: amplitude CRPS in m/s, mean over the vehicle indices 1-4 / over "
             "every measured index of a profile, then over profiles, [95 % bootstrap interval over profiles]. H3.2: "
             f"reduction 1 - CRPS_model / CRPS_reference against {reference}, paired by profile, with its verdict "
             f"(docs/design.md 13.3, 14.1; '{NOT_JUDGED}': fewer than {min_profiles} profiles). beta: exponent of "
             "sd_i = sd_1 i^beta; d2<0: share of negative second differences of the sd curve; collided: share of "
             "rollouts with a collision of any pair of the platoon."]  # fmt: skip
    for b in blocks:
        lines += ["", "#" * int(b["level"]) + " " + b["title"], "",
                  "| " + " | ".join(c[0] for c in b["columns"]) + " |",
                  "|" + "|".join("---:" if c[2] == ">" else "---" for c in b["columns"]) + "|"]  # fmt: skip
        lines += ["| " + " | ".join(cells) + " |" for cells in b["rows"]]
    if notes:
        lines += ["", "## Growth exponent of the measured platoons", "", *[f"* {n}" for n in notes]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e3", help="E3 root (unit tables, config.yaml)")
    parser.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default: config)")
    parser.add_argument("--set", default="extended", help="profile set of the summary and of the figures")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    settings = {"bootstrap": {"n_boot": args.n_boot}} if args.n_boot else None
    out = analyze_e3(args.root, settings=settings, figures=not args.no_figures, verbose=not args.quiet, set_name=args.set)
    h32 = load_settings(_resolve(args.root))["h32"]
    print(format_summary(out, args.set, str(h32["reference"]), int(h32["min_profiles"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
