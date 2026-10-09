"""Figures: colour ramps are monotone in lightness and every figure function runs on small tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from cf_unc.report import figures


def luminance(colour: str) -> float:
    rgb = np.array(to_rgb(colour))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(lin @ np.array([0.2126, 0.7152, 0.0722]))


def test_sequential_ramps_go_from_light_to_dark():
    for ramp in (figures.BLUE_RAMP, figures.ORANGE_RAMP):
        lum = [luminance(c) for c in ramp]
        assert all(a > b for a, b in zip(lum, lum[1:]))  # more is darker, no reversal
        assert lum[0] > 0.6 and lum[-1] < 0.08


def test_series_colours_differ_from_ink_and_surface():
    assert figures.BLUE != figures.ORANGE
    for colour in (figures.BLUE, figures.ORANGE):
        assert 0.1 < luminance(colour) < 0.5 < luminance(figures.SURFACE)


def small_tables(datasets: tuple[str, ...] = ("highd", "ngsim")) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    cells = [(i, j) for i in range(24) for j in range(20)]
    maps = []
    for d in datasets:
        for c, (i, j) in enumerate(cells):
            maps.append(
                dict(model="deep_ensemble", kind="model", dataset=d, subset="all", cell=c, i_s=i, i_dv=j, s_lo=5.0 * i, s_hi=5.0 * i + 5,
                     dv_lo=-5 + 0.5 * j, dv_hi=-4.5 + 0.5 * j, count=int(rng.integers(0, 200)), rare_freq=bool(i > 12), rare_mass=bool(i > 15),
                     epistemic=float(rng.uniform(1e-4, 1e-2)), aleatoric=float(rng.uniform(1e-3, 1e-1)))
            )  # fmt: skip
        maps.append(dict(model="deep_ensemble", kind="model", dataset=d, subset="all", cell=480, count=3, rare_freq=True, rare_mass=True,
                         epistemic=0.01, aleatoric=0.1))  # fmt: skip
    verdicts = pd.DataFrame(
        [
            dict(model=m, kind=k, dataset=d, subset="all", r_epi=e, r_epi_lo=e * 0.8, r_epi_hi=e * 1.3, r_alea=a, r_alea_lo=a * 0.9,
                 r_alea_hi=a * 1.1, r_epi_matched=e / 2, r_epi_matched_lo=e / 3, r_epi_matched_hi=e, r_alea_matched=a, r_alea_matched_lo=a * 0.9,
                 r_alea_matched_hi=a * 1.2)
            for m, k, e, a in (("deep_ensemble", "model", 5.0, 2.0), ("hetero_mdn", "seed_ensemble", 3.0, 1.2))
            for d in datasets if d != "highd"
        ]
    )  # fmt: skip
    regimes = ["all", *figures.REGIMES]
    stats = pd.DataFrame(
        [
            dict(model=m, dataset=d, subset="all", regime=r, nll=float(k), nll_lo=float(k) - 0.1, nll_hi=float(k) + 0.1)
            for m in ("hetero_mdn", "lstm_const_sigma") for d in datasets for k, r in enumerate(regimes)
        ]
    )  # fmt: skip
    ordering = pd.DataFrame(
        [
            dict(model="deep_ensemble", dataset=d, subset="all", quantity=q, regime=r, estimate=1.0 + k, lo=0.8 + k, hi=1.3 + k)
            for d in datasets for q in ("var", "mse") for k, r in enumerate(figures.REGIMES) if r != "steady"
        ]
    )  # fmt: skip
    return {"maps_long": pd.DataFrame(maps), "h31_verdicts": verdicts, "regime_stats": stats, "regime_ordering": ordering}


def test_every_figure_is_written(tmp_path):
    tables = small_tables()
    (tmp_path / "tables").mkdir()
    for name, table in tables.items():
        table.to_csv(tmp_path / "tables" / f"{name}.csv", index=False)
    paths = figures.make_e1_figures(tmp_path, datasets=("highd", "ngsim"), shifted=("ngsim",))
    names = {p.name for p in paths}
    assert {"deep_ensemble__model.png", "shift_ratios.png", "shift_ratios_matched.png", "nll_by_regime.png", "variance__deep_ensemble.png"} <= names
    assert all(p.exists() and p.stat().st_size > 5000 for p in paths)


def test_default_panels_include_us101(tmp_path):
    """Default panels: highD, NGSIM I-80, NGSIM US-101, Waymo, OpenACC (5 panels; 4 shifted in the ratio figures)."""
    datasets = ("highd", "ngsim", "ngsim_us101", "waymo", "openacc")
    assert figures.DATASET_LABEL["ngsim_us101"] == "NGSIM US-101 (reconstructed)"
    tables = small_tables(datasets)
    (tmp_path / "tables").mkdir()
    for name, table in tables.items():
        table.to_csv(tmp_path / "tables" / f"{name}.csv", index=False)
    paths = figures.make_e1_figures(tmp_path, target="mean_1s")
    assert {"deep_ensemble__model.png", "shift_ratios.png", "nll_by_regime.png", "variance__deep_ensemble.png"} <= {p.name for p in paths}
    assert all(p.exists() and p.stat().st_size > 5000 for p in paths)
