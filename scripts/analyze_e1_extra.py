"""Supplementary analysis of experiment E1 (milestone M2).

Three questions that the (s, dv) maps of the specification leave open:

1. signal   How predictable is the evaluated target (column y of the windows tables: the smoothed acceleration
            a[t], or the mean acceleration over the next second for target mean_1s) in each data set, whatever
            the model? (std of y, RMSE of "y = 0", RMSE of persistence y[t] = y_prev, the same target one step
            of its own length earlier; tables written before y_prev existed use a_prev = a[t-1], target a)
2. calib    How far is the predicted variance from the squared error under shift? (MSE / predicted variance)
3. rare3d   The map grid ignores the speed. With the training density on (s, dv, v): where do the shifted
            windows lie, how much of the epistemic variance sits in states that are rare in three dimensions,
            and what is left of the shift ratios when the states are matched in three dimensions? (training
            windows of --config, cut for the target of the root's windows tables)

    python scripts/analyze_e1_extra.py --root artifacts/e1
    python scripts/analyze_e1_extra.py --root artifacts/e1_mean1s --config configs/experiment/e1_mean1s.yaml
    python scripts/analyze_e1_extra.py --root artifacts/e1h --config configs/experiment/e1h.yaml   # harmonised signals
    python scripts/analyze_e1_extra.py --root artifacts/e1m --config configs/experiment/e1m.yaml   # matched signals (E5)
    python scripts/analyze_e1_extra.py --root artifacts/e6_pool --config configs/experiment/e6_pool.yaml   # E6, pool

The signal variant of --config (design.md, section 12) must be that of the windows tables of --root (checked). Every
data set of the root is analysed (the known ones in a fixed order, then the others, e.g. the pool ``pool_test`` of
E6). The in-domain reference defaults to the one of the root's last E1 analysis (``tables/analysis_info.json``,
``pool_test`` for an E6 root analysed with ``--in-domain pool_test``), else ``highd``. A pooled training domain
(design.md 16.1): the rare3d training states are the pooled training windows at stride 1, capped per data set.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from cf_unc.maps.io import available, load_preds, load_windows, windows_target  # noqa: E402
from cf_unc.maps.shift import concentration, event_sums, matched_shift_ratio, shift_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("pool_test", "highd", "highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc")  # table order
S_STEP, DV_STEP, V_STEP = 5.0, 0.5, 4.0
N_S, N_DV, N_V = 24, 20, 12  # s 0..120 m, dv -5..5 m/s, v 0..48 m/s
N_CELLS = N_S * N_DV * N_V


def bins3d(s: np.ndarray, dv: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Flat cell of (s, dv, v); windows outside the grid get N_CELLS."""
    i = np.floor(s / S_STEP).astype(np.int64)
    j = np.floor((dv + 5.0) / DV_STEP).astype(np.int64)
    k = np.floor(v / V_STEP).astype(np.int64)
    inside = (i >= 0) & (i < N_S) & (j >= 0) & (j < N_DV) & (k >= 0) & (k < N_V)
    return np.where(inside, (i * N_DV + j) * N_V + k, N_CELLS)


def rare_by_mass(train_bins: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean flag per cell (N_CELLS + 1,): least frequent cells holding at most ``mass`` of the training windows."""
    counts = np.bincount(train_bins, minlength=N_CELLS + 1).astype(np.float64)
    freq = counts / counts.sum()
    order = np.argsort(freq[:N_CELLS], kind="stable")
    cum = np.cumsum(freq[:N_CELLS][order]) + freq[N_CELLS]
    rare = np.zeros(N_CELLS + 1, dtype=bool)
    rare[order[cum <= mass]] = True
    rare[N_CELLS] = True
    return rare, freq


def training_states(config: str, target: str | None = None) -> np.ndarray:
    """(n, 3) last states (s, dv, v) of the highD training windows at stride 1 (cut for ``target``, default: the
    target of the config; a pooled domain: the pooled windows at stride 1, capped per data set as configured)."""
    from omegaconf import OmegaConf

    from cf_unc.experiments.e1 import Data, load_config

    cfg = load_config(config)
    OmegaConf.update(cfg, "train.stride", 1, force_add=True)
    if target is not None:
        OmegaConf.update(cfg, "train.target", target, force_add=True)
    bundle = Data(cfg).bundle(sequences=False)
    return bundle.windows["train"].X[:, -1, :].astype(np.float64)


def check_signal_variant(root: Path, metas: dict[str, dict], config: str) -> str:
    """Signal variant of the windows tables of ``root`` (``spec.signal`` of their meta, absent: raw); SystemExit when
    the tables mix variants or the experiment config ``config`` (training windows of rare3d) has another one. The
    variant of the config is the processing of its training data (signal ``matched``: harmonised with the reference
    window, design.md 15.1; the tables of a matched root are harmonised with one window per data set)."""
    from cf_unc.experiments.e1 import load_config, training_signal_key

    signals = {d: str((meta.get("spec") or {}).get("signal", "raw")) for d, meta in metas.items()}
    if len(set(signals.values())) > 1:
        raise SystemExit(f"windows tables of {root} have different signal variants: {signals}")
    cfg = load_config(config)
    key = training_signal_key(cfg)
    keys = key if isinstance(key, list) else [key]  # a pooled domain: the per-set keys (design.md 16.1)
    variants = {"raw" if k is None else str(k["signal"]) for k in keys}
    if len(variants) > 1:
        raise SystemExit(f"--config {config} pools training data sets of different signal variants: {sorted(variants)}")
    wanted = variants.pop()
    found = next(iter(signals.values()), wanted)
    if found != wanted:
        raise SystemExit(f"windows tables of {root} hold {found} signals, --config {config} has {wanted} training data; "
                         "pass the config of the root (e.g. configs/experiment/e1h.yaml)")  # fmt: skip
    return found


def signal_row(dataset: str, w: pd.DataFrame, target: str) -> dict[str, object]:
    """Model-free predictability of the evaluated target ``y`` of one windows table: std, RMSE of "y = 0", RMSE of
    persistence (``y_prev`` when the table has it, else ``a_prev``: tables written before, target a)."""
    column = "y_prev" if "y_prev" in w.columns else "a_prev"
    y, prev = w["y"].to_numpy(np.float64), w[column].to_numpy(np.float64)
    return {
        "dataset": dataset,
        "target": target,
        "persistence": column,
        "n_windows": len(w),
        "n_events": int(w["event"].nunique()),
        "std_a": float(y.std()),
        "rmse_zero": float(np.sqrt(np.mean(y**2))),
        "rmse_persistence": float(np.sqrt(np.nanmean((y - prev) ** 2))),
        "share_v_below_8": float(np.mean(w["v"] < 8.0)),
        "share_s_below_15": float(np.mean(w["s"] < 15.0)),
        "share_absdv_above_3": float(np.mean(np.abs(w["dv"]) > 3.0)),
    }


def geometric_ratio(
    x_shift: np.ndarray, ev_shift: np.ndarray, x_in: np.ndarray, ev_in: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Ratio of the geometric means of a positive quantity, shifted / in-domain, two-sample event bootstrap.

    x_*: (n_seeds, n) values; the logarithm is averaged over the seeds per window, then over the windows.
    A ratio of geometric means is not dominated by a few windows with extreme values, unlike a ratio of means.
    """
    rng = np.random.default_rng(seed)

    def prepare(x: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(np.where(x > 0, x, np.nan))
        value = np.nanmean(logs, axis=0) if np.isfinite(logs).any() else np.full(x.shape[1], np.nan)
        codes, names = pd.factorize(events)
        return event_sums(value, codes, len(names))

    def replicates(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
        keep = counts > 0
        sums, counts = sums[keep], counts[keep]
        n = sums.shape[0]
        out = np.empty(n_boot)
        for start in range(0, n_boot, 100):
            w = rng.multinomial(n, np.full(n, 1.0 / n), size=min(100, n_boot - start)).astype(np.float64)
            out[start : start + w.shape[0]] = (w @ sums) / (w @ counts)
        return out

    s_sums, s_counts = prepare(x_shift, ev_shift)
    i_sums, i_counts = prepare(x_in, ev_in)
    if s_counts.sum() == 0 or i_counts.sum() == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan}
    estimate = s_sums.sum() / s_counts.sum() - i_sums.sum() / i_counts.sum()
    reps = replicates(s_sums, s_counts) - replicates(i_sums, i_counts)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"estimate": float(np.exp(estimate)), "lo": float(np.exp(lo)), "hi": float(np.exp(hi))}


def stack(root: Path, model: str, seeds: list[int], dataset: str, column: str) -> np.ndarray:
    return np.stack([load_preds(root, model, k, dataset, columns=[column])[column].to_numpy(np.float64) for k in seeds])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e1")
    parser.add_argument("--config", default="configs/experiment/e1.yaml",
                        help="experiment config of the training windows (rare3d; cut for the target of --root)")  # fmt: skip
    parser.add_argument("--in-domain", default=None,
                        help="in-domain reference (default: that of tables/analysis_info.json, else highd)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--mass", type=float, default=0.10)
    parser.add_argument("--min-count", type=int, default=30)
    args = parser.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    av = available(root)
    datasets = [d for d in DATASETS if d in set(av.dataset)] + sorted(set(av.dataset) - set(DATASETS))
    if args.in_domain is None:
        info_path = out_dir / "analysis_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        args.in_domain = str(info.get("in_domain") or "highd")
    models = sorted(set(av.model))
    loaded = {d: load_windows(root, d) for d in datasets}
    windows = {d: frame for d, (frame, _) in loaded.items()}
    targets = {d: windows_target(meta) for d, (_, meta) in loaded.items()}
    if len(set(targets.values())) > 1:
        raise ValueError(f"windows tables of {root} have different targets: {targets}")
    target = next(iter(targets.values()), None)
    check_signal_variant(root, {d: meta for d, (_, meta) in loaded.items()}, args.config)

    # ------------------------------------------------------------------ 1. signal
    signal = pd.DataFrame([signal_row(d, w, targets[d]) for d, w in windows.items()])
    signal.to_csv(out_dir / "extra_signal.csv", index=False)

    # ------------------------------------------------------------------- 2. calib
    rows = []
    for model in models:
        for d in datasets:
            seeds = sorted(av[(av.model == model) & (av.dataset == d)].seed)
            if not seeds:
                continue
            mean, var = stack(root, model, seeds, d, "mean"), stack(root, model, seeds, d, "var")
            y = windows[d]["y"].to_numpy(np.float64)[None, :]
            ok = np.isfinite(mean) & np.isfinite(var)
            mse = float(np.mean(((y - mean) ** 2)[ok]))
            pv = float(np.mean(var[ok]))
            z2 = ((y - mean) ** 2 / var)[ok]
            rows.append(
                {
                    "model": model,
                    "dataset": d,
                    "n_seeds": len(seeds),
                    "mse": mse,
                    "predicted_var": pv,
                    "mse_over_var": mse / pv,
                    "median_z2": float(np.median(z2)),  # 0.455 for a calibrated Gaussian law
                    "share_z_above_3": float(np.mean(z2 > 9.0)),  # 0.0027 for a calibrated Gaussian law
                }
            )
    calib = pd.DataFrame(rows)
    calib.to_csv(out_dir / "extra_calibration.csv", index=False)

    # ------------------------------------------------------------------ 3. rare3d
    train = training_states(args.config, target)
    rare, freq = rare_by_mass(bins3d(train[:, 0], train[:, 1], train[:, 2]), args.mass)
    cells = {d: bins3d(*(w[c].to_numpy(np.float64) for c in ("s", "dv", "v"))) for d, w in windows.items()}
    occupancy = []
    for d in datasets:
        c = cells[d]
        occupancy.append(
            {
                "dataset": d,
                "share_rare3d": float(rare[c].mean()),
                "share_unseen3d": float(np.mean(freq[c] == 0.0)),
                "share_outside3d": float(np.mean(c == N_CELLS)),
            }
        )
    occupancy = pd.DataFrame(occupancy)
    occupancy.to_csv(out_dir / "extra_rare3d_occupancy.csv", index=False)

    rows = []
    ref = args.in_domain
    for model in models:
        seeds_ref = sorted(av[(av.model == model) & (av.dataset == ref)].seed)
        if not seeds_ref:
            continue
        epi_ref = stack(root, model, seeds_ref, ref, "epistemic_var")
        if not np.isfinite(epi_ref).any() or np.nanmax(epi_ref) <= 0.0:
            continue  # no epistemic output of its own
        ev_ref = windows[ref]["event"].to_numpy()
        for d in datasets:
            seeds = sorted(av[(av.model == model) & (av.dataset == d)].seed)
            if seeds != seeds_ref:
                continue
            epi = epi_ref if d == ref else stack(root, model, seeds, d, "epistemic_var")
            ev = windows[d]["event"].to_numpy()
            conc = concentration(epi, rare[cells[d]], ev, n_boot=args.n_boot, seed=0)
            row = {
                "model": model,
                "dataset": d,
                "share_windows_rare3d": conc["share_windows"],
                "share_epistemic_in_rare3d": conc["share_mass"],
                "ratio_rare3d": conc["estimate"],
                "ratio_rare3d_lo": conc["lo"],
                "ratio_rare3d_hi": conc["hi"],
            }
            if d != ref:
                plain = shift_ratio(epi, ev, epi_ref, ev_ref, n_boot=args.n_boot, seed=0)
                m3 = matched_shift_ratio(
                    epi, cells[d], ev, epi_ref, cells[ref], ev_ref, N_CELLS, min_count=args.min_count, n_boot=args.n_boot, seed=0
                )
                row.update(
                    r_epi=plain["estimate"],
                    r_epi_lo=plain["lo"],
                    r_epi_hi=plain["hi"],
                    r_epi_matched3d=m3["estimate"],
                    r_epi_matched3d_lo=m3["lo"],
                    r_epi_matched3d_hi=m3["hi"],
                    n_cells_matched=m3["n_bins_matched"],
                    mass_in_matched=m3["mass_in"],
                    mass_shift_matched=m3["mass_shift"],
                )
            rows.append(row)
    rare3d = pd.DataFrame(rows)
    rare3d.to_csv(out_dir / "extra_rare3d.csv", index=False)

    # ------------------------------------------------- 4. robust (geometric) shift ratios
    rows = []
    for model in models:
        seeds_ref = sorted(av[(av.model == model) & (av.dataset == ref)].seed)
        if not seeds_ref:
            continue
        ev_ref = windows[ref]["event"].to_numpy()
        y_ref = windows[ref]["y"].to_numpy(np.float64)[None, :]
        cols = {c: stack(root, model, seeds_ref, ref, c) for c in ("epistemic_var", "aleatoric_var", "var", "mean")}
        cols["sq_err"] = (y_ref - cols.pop("mean")) ** 2
        for d in datasets:
            seeds = sorted(av[(av.model == model) & (av.dataset == d)].seed)
            if d == ref or seeds != seeds_ref:
                continue
            ev = windows[d]["event"].to_numpy()
            y = windows[d]["y"].to_numpy(np.float64)[None, :]
            for quantity in ("epistemic_var", "aleatoric_var", "var", "sq_err"):
                x = (y - stack(root, model, seeds, d, "mean")) ** 2 if quantity == "sq_err" else stack(root, model, seeds, d, quantity)
                if not (np.isfinite(cols[quantity]) & (cols[quantity] > 0)).any():
                    continue
                g = geometric_ratio(x, ev, cols[quantity], ev_ref, n_boot=args.n_boot, seed=0)
                m = shift_ratio(x, ev, cols[quantity], ev_ref, n_boot=args.n_boot, seed=0)
                rows.append(
                    {
                        "model": model, "dataset": d, "quantity": quantity,
                        "ratio_geometric": g["estimate"], "geo_lo": g["lo"], "geo_hi": g["hi"],
                        "ratio_means": m["estimate"], "means_lo": m["lo"], "means_hi": m["hi"],
                    }
                )  # fmt: skip
    robust = pd.DataFrame(rows)
    robust.to_csv(out_dir / "extra_shift_geometric.csv", index=False)

    info = {"grid": {"s_step": S_STEP, "dv_step": DV_STEP, "v_step": V_STEP, "n_cells": N_CELLS}, "mass": args.mass,
            "n_train_windows": int(len(train)), "target": target, "config": str(args.config), "in_domain": ref}  # fmt: skip
    (out_dir / "extra_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    for title, table in (("signal", signal), ("rare3d occupancy", occupancy), ("calibration", calib), ("rare3d", rare3d),
                         ("geometric shift ratios", robust)):  # fmt: skip
        print(f"\n== {title}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
