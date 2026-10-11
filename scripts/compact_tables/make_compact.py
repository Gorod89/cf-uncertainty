#!/usr/bin/env python3
r"""Public copy of the compact-table generator of the article (the 19 tables c1-c19 of the manuscript), kept in
sync with the manuscript tool. It reads the experiment artefacts in the layout of the development repository:
artifacts/report/ (the report assembly, published as results/report/) and artifacts/<root>/tables/ (the analysis
tables of the experiment roots, published as results/controls/<root>/tables/). Set CF_UNC_ARTIFACTS to a directory
with that layout (for example after copying results/report to <dir>/report and results/controls/<root> to
<dir>/<root>), then

    PYTHONUTF8=1 python scripts/compact_tables/make_compact.py --out results/compact_tables --no-copy

The original module docstring follows.

Compact tables and figure environments of version v4 of the manuscript (paper/v4/: the control experiments of
the review response, docs/design.md 18 and docs/review_fixes_v4.md).

Usage, from the repository root (after the analyses of every root and scripts/make_report.py):

    PYTHONUTF8=1 python scripts/compact_tables/make_compact.py
    PYTHONUTF8=1 python scripts/compact_tables/make_compact.py --no-copy   # keep figures/
    PYTHONUTF8=1 python scripts/compact_tables/make_compact.py --out paper/v4   # the default

Reads the CSVs of artifacts/report/tables (their columns are described in artifacts/report/index.md) and, for the
tables of the controls (c13-c18), the tables of the experiment roots themselves (artifacts/<root>/tables: the control
roots e1_ctrl, e2_ctrl and e1m_causal are not report roots), the matching file of e1m_causal and the fit logs of
idm_ar_noise; writes, below ``--out`` (default: the paper directory of this tool, paper/v4; paper/submission and
paper/v3, the earlier versions, are refused):

* tables/<name>.tex      the compact tables c1-c8, c10-c12 (review fixes of docs/review_fixes_v3.md) and c13-c18
                         (controls of docs/review_fixes_v4.md), each a complete MDPI float: [H], caption, label
                         tab:<name>, booktabs rules, tabularx at \textwidth or (full page width) at \fulllength inside
                         adjustwidth{-\extralength}{0cm}, as in the MDPI template.tex (c9_e3pooled.tex is written by
                         hand and left alone);
* generated/tables.tex   the \input lines of the tables, as comments (the file inputs nothing);
* generated/figures.tex  one MDPI figure environment per figure, label fig:<file stem> (the two map panels as one
                         figure with panels (a) and (b));
* generated/LABELS.md    labels, files and captions of the tables and figures;
* figures/               copies of the PNG files used, from artifacts/report/figures.

Rounding as in scripts/make_report.py (num, ci), with at most two decimals: three significant digits for values
of 1 and more (NLL, variance ratios, CRPS, widths, differences, gaps, reductions, exponents), two decimals below
1 (0.220 of Supplementary File S1 is 0.22 here), interval bounds with the decimals of their estimate; coverages
(PICP) with two decimals; shares of windows and of rollouts in per cent with one decimal; one-step CRPS values below
0.1 (m/s^2) with two significant digits (c14, c15). Model labels: cf_unc.report.figures.MODEL_LABEL, in the platoon
tables cf_unc.report.figures_e3.MODEL_LABEL_E3 (c13: short labels, explained in its notes). The model order puts
IDM + AR(1) noise right after IDM + white noise; the history-only Diffusion control of the platoons is reported in
c14 only. Figure captions: artifacts/report/index.md.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import os

REPO = Path(__file__).resolve().parents[2]  # scripts/compact_tables -> repository root
SUB = REPO / "results" / "compact_tables"  # default output directory of the public copy; --out changes it
SUBMITTED = REPO / "paper" / "submission"  # the submitted version: never written
FROZEN = (SUBMITTED, REPO / "paper" / "v3")  # earlier versions: never written by this tool
ART = Path(os.environ.get("CF_UNC_ARTIFACTS", str(REPO / "artifacts")))  # the experiment artefacts (see README)
REPORT = ART / "report"
CSV_DIR = REPORT / "tables"
FIG_DIR = REPORT / "figures"

sys.path.insert(0, str(REPO))
from cf_unc.report.figures import MODEL_LABEL  # noqa: E402
from cf_unc.report.figures_e3 import MODEL_LABEL_E3  # noqa: E402

# Models of the compact tables, in this order where they appear. The one-step IDM fit and the three-component
# mixture are left out (they are in Supplementary File S1).
MODELS = ("idm_white_noise", "idm_ar_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout", "gp_residual_idm",
          "quantile_pidl", "diffusion_cf")
# The history-only Diffusion of the review control R2 (artifacts/e1_ctrl) runs in the recorded-signal platoons (E3 variant raw):
# it is reported in tab:c14_information, not in the platoon tables of the main model set.
CONTROL_MODELS = ("diffusion_cf_hist",)
DATASET_LABEL = {"highd": "highD", "ngsim": "NGSIM I-80", "ngsim_us101": "NGSIM US-101", "waymo": "Waymo",
                 "openacc": "OpenACC"}
SHIFTED = ("ngsim", "ngsim_us101", "waymo", "openacc")
REGIMES = ("congested", "braking", "approaching", "free", "steady", "other")  # order of cf_unc.report.figures
# tab:c2_shift, across-seed decomposition: the report has no geometric epistemic ratio for it
# (scripts/analyze_e1_extra.py computes geometric ratios of a model's own epistemic output only).
# "means": the epistemic cell shows the ratio of means (r_epi, the basis of the verdict) in italics, explained
# in a footnote; "dash": the cell stays empty.
ACROSS_SEED_EPISTEMIC = "means"
DASH = "--"

# Figures of the submission, in the order of the text: (figure, panels). A directory of artifacts/report/figures
# with the panel files to combine (panels (a), (b), ...), or a PNG file (panels None).
FIGURES: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("f1_maps__e1", ("deep_ensemble__model.png", "gp_residual_idm__model.png")),
    ("f2_shift_ratios__e1.png", None),
    ("f3_nll_by_regime__e1.png", None),
    ("f4_coverage_by_horizon__a__e2.png", None),
    ("f4_coverage_by_horizon__a__ngsim__e2.png", None),
    ("f5_crps_bars__raw__e3.png", None),
    ("f6_growth__raw__e3.png", None),
    ("f7_crps_by_index__raw__e3.png", None),
    ("f2_shift_ratios__e1m.png", None),
    ("f4_coverage_by_horizon__a__ngsim__e2m.png", None),
    ("f2_shift_ratios__e6_lodo_ngsim_all.png", None),
)
FIG_PX_PER_CM = 170 / 2.54  # the report figures are saved at 170 dpi
# Figure widths: designed wider than WIDE_CM -> the full page width (\fulllength = 18.46 cm, scale <= 0.84);
# wider than the text column (13.86 cm) -> the text column; narrower -> their design width (scale 1).
WIDE_CM = 22.0
TEXT_CM = 13.86


# ------------------------------------------------------------------------------------------ numbers
# The rounding of scripts/make_report.py (_decimals, _fixed, num, ci), so that a number reads as in Supplementary
# File S1 or as its rounding there.
def _decimals(x: float, sig: int) -> int:
    """Decimals that show x with sig significant digits (negative: rounded to tens, hundreds, ...)."""
    if not np.isfinite(x) or x == 0:
        return sig - 1
    return sig - 1 - int(math.floor(math.log10(abs(x))))


def _fixed(x: float, d: int) -> str:
    """x with d decimals (d < 0: rounded to tens, hundreds, ...); no negative zero."""
    s = f"{round(x, d):.0f}" if d < 0 else f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _sign(s: str) -> str:
    """A leading minus or plus as a math sign."""
    if s.startswith("-"):
        return "$-$" + s[1:]
    if s.startswith("+"):
        return "$+$" + s[1:]
    return s


def _dec(x: float, sig: int, max_dec: int | None) -> int:
    """Decimals of x at sig significant digits and at most max_dec decimals; a rounding that carries into the
    next power of ten (0.9996 -> 1.000) is shown with the decimals of the rounded value (1.00)."""
    d = _decimals(x, sig) if max_dec is None else min(max_dec, _decimals(x, sig))
    rounded = float(_fixed(x, d))
    if rounded != 0 and math.floor(math.log10(abs(rounded))) > math.floor(math.log10(abs(x))):
        d = _decimals(rounded, sig) if max_dec is None else min(max_dec, _decimals(rounded, sig))
    return d


def num(x: float, sig: int = 3, max_dec: int | None = 2, sign: bool = False) -> str:
    """x with sig significant digits and at most max_dec decimals; sign: an explicit plus."""
    x = float(x)
    if not np.isfinite(x):
        return DASH
    d = _dec(x, sig, max_dec)
    s = _fixed(x, d)
    if sign and not s.startswith("-") and float(s) != 0:
        s = "+" + s
    return _sign(s)


def ci(est: float, lo: float, hi: float, sig: int = 3, max_dec: int | None = 2) -> tuple[str, str]:
    """('estimate', '[lo, hi]'): the estimate with sig significant digits (at most max_dec decimals), the bounds
    with the same decimals."""
    est, lo, hi = float(est), float(lo), float(hi)
    if not all(np.isfinite(v) for v in (est, lo, hi)):
        raise SystemExit(f"interval with a missing value: {est}, {lo}, {hi}")
    d = _dec(est, sig, max_dec)
    return _sign(_fixed(est, d)), f"[{_sign(_fixed(lo, d))}, {_sign(_fixed(hi, d))}]"


def ci1(est: float, lo: float, hi: float, **kw) -> str:
    """'estimate [lo, hi]' on one line."""
    a, b = ci(est, lo, hi, **kw)
    return f"{a} {b}"


def prob(x: float, d: int = 2) -> str:
    """A coverage with d decimals."""
    x = float(x)
    if not np.isfinite(x):
        return DASH
    return _fixed(x, d)


def pct(x: float) -> str:
    """A share in per cent with one decimal; '0' for none, '<0.1' for a share below 0.05 %."""
    p = 100.0 * float(x)
    if not np.isfinite(p):
        return DASH
    if p == 0:
        return "0"
    if p < 0.05:
        return "$<$0.1"
    return _fixed(p, 1)


def thousands(n: int) -> str:
    """10000 -> 10\\,000 (thin space as the thousands separator of MDPI)."""
    return f"{int(n):,}".replace(",", r"\,")


def stack(top: str, bottom: str) -> str:
    """Two lines in one cell (a nested tabular; '\\{}' so that a line starting with '[' is not read as the
    optional argument of the line break)."""
    return rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{top}\\{{}}{bottom}\end{{tabular}}"


def fn(k: int) -> str:
    """Footnote mark of a table (MDPI: superscript numbers, explained below the table)."""
    return rf"\textsuperscript{{{k}}}"


# ------------------------------------------------------------------------------------------ LaTeX text
# Escaping as in scripts/make_report.py and paper/tools/make_generated.py (tex, caption_tex), so that a caption
# taken from index.md reads as in Supplementary File S1.
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"$<$",
        ">": r"$>$", "|": r"$|$", "±": r"$\pm$", "→": r"$\rightarrow$", "≥": r"$\geq$",
        "≤": r"$\leq$", "–": "--", "—": "---", "×": r"$\times$", "β": r"$\beta$", "²": r"$^2$",
        "∞": r"$\infty$", "·": r"$\cdot$", "−": "-"}
_MATH = {
    "|y - mu| / sigma": r"$|y - \mu| / \sigma$",
    "sd_i = sd_1 i^beta": r"$\mathrm{sd}_i = \mathrm{sd}_1\, i^{\beta}$",
    "(d2 < 0)": r"($d_2 < 0$)",
    "p < 0.05": r"$p < 0.05$",
    "1 - CRPS/CRPS(IDM + white noise)": r"$1 - \mathrm{CRPS}/\mathrm{CRPS}$(IDM + white noise)",
    # submission only: the variables of the captions as math
    "(dv, s)": r"$(\Delta v, s)$",
    "target a (": r"target $a$ (",
}


def tex(text: str) -> str:
    text = str(text).replace(">=", "≥").replace("<=", "≤")
    out = "".join(_TEX.get(ch, ch) for ch in text)
    out = re.sub(r"(\d)e([+-]?)0*(\d+)",
                 lambda m: f"{m.group(1)}$\\times10^{{{'-' if m.group(2) == '-' else ''}{m.group(3)}}}$", out)
    return re.sub(r"(?:(?<=[\s\[(/])|^)-(?=[\d.])", "$-$", out)


def caption_tex(text: str) -> str:
    """LaTeX of a caption of index.md: the formulas of _MATH as math, the rest escaped as tex(); '90 %' as
    '90\\%' (MDPI style)."""
    parts: list[str] = []
    rest = text
    while rest:
        hits = [(rest.find(k), k) for k in _MATH if k in rest]
        if not hits:
            parts.append(tex(rest))
            break
        pos, key = min(hits)
        parts.append(tex(rest[:pos]))
        parts.append(_MATH[key])
        rest = rest[pos + len(key):]
    return re.sub(r"(\d) \\%", r"\1\\%", "".join(parts))


def detex(text: str) -> str:
    """Plain text of a caption or a cell (LABELS.md, comments)."""
    t = text.replace(r"$\bar a_{1\,\mathrm{s}}$", "mean_1s").replace(r"\sigma_\eta", "σ_η").replace(r"\rho", "ρ")
    t = re.sub(r"\\(?:textit|textbf|emph|mathrm)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\textsuperscript\{([^{}]*)\}", r"^\1", t)
    for a, b in ((r"\%", "%"), (r"\,", " "), (r"\beta", "β"), (r"\Delta ", "Δ"), (r"\Delta", "Δ"), (r"\geq", "≥"),
                 (r"\leq", "≤"),
                 (r"\times", "×"), (r"\pm", "±"), (r"\mu", "μ"), (r"\sigma", "σ"), (r"\rightarrow", "→"),
                 (r"\&", "&"), (r"\_", "_"), ("---", "—"), ("--", "–"), ("~", " "), ("$", "")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def label(model: str, e3: bool = False) -> str:
    return tex((MODEL_LABEL_E3 if e3 else MODEL_LABEL)[model])


# ------------------------------------------------------------------------------------------ tables
@dataclass
class Head:
    """One row of column heads: (text, number of columns spanned); cmidrules under the given column ranges."""

    cells: list[tuple[str, int]]
    rules: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Group:
    """A row that names the group of rows below it (italic, across all columns)."""

    text: str


RULE = r"\midrule"


@dataclass
class Table:
    name: str
    caption: str  # LaTeX
    sources: list[str]  # CSV files of artifacts/report/tables
    spec: str  # tabularx column spec
    heads: list[Head]
    body: list  # rows (lists of cells), Group or RULE
    notes: list[str] = field(default_factory=list)  # footnotes below the table (LaTeX)
    wide: bool = False  # full page width: adjustwidth{-\extralength}{0cm} and tabularx{\fulllength}
    font: str = "small"
    tabcolsep: float | None = None

    @property
    def ncols(self) -> int:
        return sum(span for _, span in self.heads[-1].cells)


# Column specs: "C" (the centred tabularx column of the MDPI class) for the entries that share the table width
# equally, "c" for a column at its natural width (intervals, verdicts); no \hsize weights, whose rounding leaves
# the final tabular* of tabularx a few sp short of its width ("Underfull \hbox in alignment").
def head2(top: str, bottom: str) -> str:
    """A column head on two lines (for a natural-width column whose head is wider than its entries)."""
    return rf"\begin{{tabular}}[b]{{@{{}}c@{{}}}}\textbf{{{top}}}\\{{}}\textbf{{{bottom}}}\end{{tabular}}"


def head_lines(h: Head) -> list[str]:
    cells = []
    for text, span in h.cells:
        cell = text if text.startswith(r"\begin{tabular}") else (rf"\textbf{{{text}}}" if text else "")
        cells.append(rf"\multicolumn{{{span}}}{{c}}{{{cell}}}" if span > 1 else cell)
    lines = [" & ".join(cells) + r" \\"]
    if h.rules:
        lines.append(" ".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b in h.rules))
    return lines


def render(t: Table) -> str:
    n = t.ncols
    for h in t.heads:
        if sum(span for _, span in h.cells) != n:
            raise SystemExit(f"{t.name}: head row with {sum(s for _, s in h.cells)} columns, expected {n}")
    out = [f"% tab:{t.name}: {detex(t.caption)}",
           "% Sources: " + ", ".join(source_path(s) for s in t.sources),
           "% Written by paper/v4/tools/make_compact.py; do not edit by hand. Insert with",
           f"% \\input{{tables/{t.name}}} at paragraph level (this file holds the complete float).",
           r"\begin{table}[H]",
           rf"\caption{{{t.caption}\label{{tab:{t.name}}}}}"]
    if t.wide:
        out.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
    out.append("\\" + t.font)
    if t.tabcolsep is not None:
        out.append(rf"\setlength{{\tabcolsep}}{{{t.tabcolsep:g}pt}}")
    width = r"\fulllength" if t.wide else r"\textwidth"
    out += [rf"\begin{{tabularx}}{{{width}}}{{{t.spec}}}", r"\toprule"]
    for h in t.heads:
        out += head_lines(h)
    out.append(r"\midrule")
    first_group = True
    for item in t.body:
        if isinstance(item, Group):
            if not first_group:
                out.append(r"\midrule")
            first_group = False
            out.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{item.text}}}}} \\")
        elif isinstance(item, str):
            out.append(item)
        else:
            if len(item) != n:
                raise SystemExit(f"{t.name}: row with {len(item)} cells, expected {n}: {item}")
            out.append(" & ".join(item) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}"]
    if t.wide:
        out.append(r"\end{adjustwidth}")
    if t.notes:
        if not t.wide:
            out.append("")
        out.append(r"\noindent{\footnotesize{" + " ".join(t.notes) + "}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------ data
def source_path(name: str) -> str:
    """Path of a source below the repository: a report CSV by its name, a table of an experiment root (c13-c18) as
    given (``artifacts/<root>/...``)."""
    return name if name.startswith("artifacts/") else f"artifacts/report/tables/{name}"


def load(name: str, **eq) -> pd.DataFrame:
    """A report CSV, restricted to the rows with the given column values (every one must exist)."""
    df = pd.read_csv(CSV_DIR / name)
    for k, v in eq.items():
        if k not in df.columns:
            raise SystemExit(f"{name}: no column {k!r}")
        df = df[df[k] == v]
    if df.empty:
        raise SystemExit(f"{name}: no rows for {eq}")
    return df


def pick(df: pd.DataFrame, **eq) -> pd.Series:
    sel = df
    for k, v in eq.items():
        sel = sel[sel[k] == v]
    if len(sel) != 1:
        raise SystemExit(f"expected one row for {eq}, found {len(sel)}")
    return sel.iloc[0]


def check_labels(df: pd.DataFrame, e3: bool = False) -> None:
    """The model labels of a report CSV are those of MODEL_LABEL (MODEL_LABEL_E3)."""
    labels = MODEL_LABEL_E3 if e3 else MODEL_LABEL
    for m, lab in df[["model", "model_label"]].drop_duplicates().itertuples(index=False):
        if m in labels and labels[m] != lab:
            raise SystemExit(f"model label of {m}: {lab!r} in the CSV, {labels[m]!r} in cf_unc.report")


def check_seeds(df: pd.DataFrame, n: int = 5) -> None:
    if "n_seeds" in df.columns and set(df["n_seeds"].astype(int)) != {n}:
        raise SystemExit(f"seeds: {sorted(set(df['n_seeds']))}, expected {n}")


# ------------------------------------------------------------------------------------------ the eight tables
def c1_onestep() -> Table:
    src = "t1_one_step__raw__a.csv"
    df = load(src, variant="raw", target="a")
    check_labels(df)
    sets = ("highd",) + SHIFTED
    check_seeds(df[df.dataset.isin(sets) & df.model.isin(MODELS)])
    n_diff = {int(pick(df, model="diffusion_cf", dataset=d).n_windows) for d in sets}
    if len(n_diff) != 1:
        raise SystemExit(f"Diffusion: different numbers of windows {n_diff}")
    for m in MODELS:
        if m != "diffusion_cf":
            for d in sets:
                if int(pick(df, model=m, dataset=d).n_windows) <= max(n_diff):
                    raise SystemExit(f"{m} on {d}: not more windows than Diffusion")
    body = []
    for m in MODELS:
        cells = [label(m) + (fn(1) if m == "diffusion_cf" else "")]
        for d in sets:
            r = pick(df, model=m, dataset=d)
            cells += [num(r.nll_mean), prob(r.picp90_mean)]
        body.append(cells)
    counts = []  # events and followers of the evaluated windows: the dependence structure of the intervals
    for d in sets:
        r = pick(df, model="idm_white_noise", dataset=d)
        if "n_followers" not in df.columns or not np.isfinite(float(r.n_followers)):
            raise SystemExit(f"{src}: no n_followers for {d} (rerun scripts/analyze_e1.py and scripts/make_report.py)")
        counts.append(rf"\multicolumn{{2}}{{c}}{{{thousands(int(r.n_events))} / {thousands(int(r.n_followers))}}}")
    body += [RULE, r"Events / followers" + fn(2) + " & " + " & ".join(counts) + r" \\"]
    heads = [Head([("", 1)] + [(DATASET_LABEL[d], 2) for d in sets],
                  [(2 + 2 * i, 3 + 2 * i) for i in range(len(sets))]),
             Head([("Model", 1)] + [("NLL", 1), ("PICP", 1)] * len(sets))]
    notes = [fn(1) + rf" Evaluated on {thousands(n_diff.pop())} windows of each data set; the other models on all"
             " windows.",
             fn(2) + r" Evaluated events and distinct followers per data set: the windows of an event, and the events "
             r"of a follower, are dependent; the bootstrap intervals resample events (followers: Supplementary File S1)."]
    caption = (r"One-step metrics, recorded signals, target $a$ (one-step acceleration): negative log-likelihood "
               r"(NLL, nats) and coverage of the 90\% prediction interval (PICP) per model on the highD test set and "
               r"the four shifted data sets; mean over five training seeds.")
    return Table("c1_onestep", caption, [src], "l" + "C" * 10, heads, body, notes, wide=True, font="small")


def c2_shift() -> tuple[Table, list[str]]:
    src, src_s = "t2_h31__raw__a.csv", "t2_h31_summary.csv"
    r = load(src, variant="raw", target="a")
    s = load(src_s, variant="raw", target="a")
    check_labels(r)
    models = [m for m in MODELS if m in set(r.model)]
    log: list[str] = []
    body = []
    same_rule = True
    for m in models:
        kind = "model" if "model" in set(r[r.model == m].kind) else "seed_ensemble"
        own = kind == "model"
        log.append(f"{m}: {'own' if own else 'across-seed'} decomposition")
        cells = [label(m) + fn(1 if own else 2)]
        for d in SHIFTED:
            x = pick(r, model=m, kind=kind, dataset=d, pair="shift")
            if own:
                epi = num(x.geo_epi)
            else:
                if np.isfinite(x.geo_epi):
                    raise SystemExit(f"{m}, {d}: geometric epistemic ratio for the across-seed decomposition")
                epi = rf"\textit{{{num(x.r_epi)}}}" if ACROSS_SEED_EPISTEMIC == "means" else DASH
            cells += [epi, num(x.geo_alea)]
        y = pick(s, model=m, kind=kind)
        if int(y.h31_n_pairs) != len(SHIFTED):
            raise SystemExit(f"{m}: {y.h31_n_pairs} counted pairs, expected {len(SHIFTED)}")
        same_rule &= (y.h31_overall == y.h31_overall_point) and (y.h31e_overall == y.h31e_overall_point)
        cells += [str(y.h31_overall), str(y.h31e_overall)]
        body.append(cells)
    heads = [Head([("", 1)] + [(DATASET_LABEL[d], 2) for d in SHIFTED] + [("Verdict", 2)],
                  [(2 + 2 * i, 3 + 2 * i) for i in range(len(SHIFTED) + 1)]),
             Head([("Model", 1)] + [("Epi.", 1), ("Alea.", 1)] * len(SHIFTED) + [("H3.1", 1), ("H3.1e", 1)])]
    across = (r" The report gives no geometric epistemic ratio for this decomposition; the epistemic entries in "
              r"italics are ratios of means." if ACROSS_SEED_EPISTEMIC == "means" else
              r" The report gives no geometric epistemic ratio for this decomposition.")
    notes = [fn(1) + r" The model's own decomposition: variance of the member means (deep ensemble), of the dropout "
                     r"passes (MC dropout), GP posterior variance (GP residual).",
             fn(2) + r" Across-seed decomposition over the five training seeds (epistemic: variance of the seeds' "
                     r"predictive means; aleatoric: their predictive variance)." + across]
    rule = r" (interval rule; the point rule gives the same verdicts)" if same_rule else r" (interval rule)"
    caption = (r"H3.1 under dataset shift, recorded signals, target $a$: ratio of the geometric means of the "
               r"epistemic (Epi.) and the aleatoric (Alea.) variance, shifted data set over highD test, per model, "
               r"and the overall H3.1 and H3.1e verdicts over the four shifted data sets" + rule + ". IDM + white "
               r"noise, without an epistemic part, is not listed.")
    spec = "l" + "C" * (2 * len(SHIFTED)) + "cc"
    return Table("c2_shift", caption, [src, src_s], spec, heads, body, notes, wide=True, font="footnotesize"), log


def c3_regimes() -> Table:
    src_n, src_v = "t3_h33_nll__raw__a.csv", "t3_regime_variance__raw__a.csv"
    n = load(src_n, variant="raw", target="a", dataset="highd")
    v = load(src_v, variant="raw", target="a", dataset="highd", model="hetero_mdn")
    if set(n.model_a) != {"hetero_mdn"} or set(n.model_b) != {"lstm_const_sigma"}:
        raise SystemExit(f"{src_n}: models {set(n.model_a)} / {set(n.model_b)}")
    total = int(pick(n, regime="all").n_windows)
    if sum(int(pick(n, regime=g).n_windows) for g in REGIMES) != total:
        raise SystemExit(f"{src_n}: the regimes do not partition the windows")
    var, mse = pick(v, quantity="var"), pick(v, quantity="mse")
    ref = str(var.reference)
    if ref != "steady" or str(mse.reference) != ref:
        raise SystemExit(f"{src_v}: reference {var.reference} / {mse.reference}")
    others = [g for g in REGIMES if g != ref]
    rejects = [bool(q[f"reject_{g}"]) for q in (var, mse) for g in others]
    star_all = all(rejects)

    def ratio(q: pd.Series, g: str) -> str:
        if g == ref:
            return "1"
        return num(q[f"estimate_{g}"]) + ("" if star_all or not bool(q[f"reject_{g}"]) else "*")

    body = []
    for g in REGIMES:
        x = pick(n, regime=g)
        body.append([g.capitalize(), pct(x.n_windows / total), num(x.nll_a), num(x.nll_b),
                     ci1(x["diff"], x["diff_lo"], x["diff_hi"]), ratio(var, g), ratio(mse, g)])
    x = pick(n, regime="all")
    body += [RULE, ["All windows", "100", num(x.nll_a), num(x.nll_b), ci1(x["diff"], x["diff_lo"], x["diff_hi"]), DASH,
                    DASH]]
    heads = [Head([("", 1), ("", 1), ("NLL (nats)", 3), ("Ratio to steady following", 2)], [(3, 5), (6, 7)]),
             Head([("Regime", 1), (r"Share (\%)", 1), ("HeteroMDN", 1), ("LSTM", 1),
                   (head2("Difference", r"[95\% CI]"), 1), ("Predicted variance", 1), ("Squared residual", 1)])]
    sig = (r"All ratios differ from 1 (Holm-adjusted $p < 0.05$)." if star_all else
           r"* Holm-adjusted $p < 0.05$ for a ratio of 1.")
    caption = (r"H3.3, recorded signals, target $a$, highD test set: share of the windows, NLL of HeteroMDN and of "
               r"the LSTM with constant sigma per driving regime and their paired difference with its 95\% "
               r"event-bootstrap interval (negative: HeteroMDN better), and HeteroMDN's mean predicted variance and "
               r"mean squared residual per regime relative to steady following. " + sig)
    spec = "lCCCcCC"  # the interval column at its natural width, the others share the rest equally
    return Table("c3_regimes", caption, [src_n, src_v], spec, heads, body, [], wide=True, font="small")


def c4_coverage() -> Table:
    src = "t4_coverage__raw__a.csv"
    df = load(src, variant="raw", target="a")
    check_labels(df)
    body: list = []
    for d, title in (("highd", "highD test"), ("ngsim", "NGSIM I-80")):
        body.append(Group(title))
        for m in MODELS:
            r = pick(df, dataset=d, model=m)
            infinite = max(float(r[f"share_infinite_aci_online_{h}s"]) for h in (1, 3, 5, 10))
            body.append([label(m), prob(r.picp_raw_1s), prob(r.picp_raw_10s), prob(r.picp_aci_online_1s),
                         prob(r.picp_aci_online_10s), num(r.width_median_aci_online_10s), pct(infinite), pct(r.collided_10s)])
    heads = [Head([("", 1), ("PICP, raw", 2), ("PICP, ACI online", 2), ("", 1), ("", 1), ("", 1)], [(2, 3), (4, 5)]),
             Head([("Model", 1), ("1 s", 1), ("10 s", 1), ("1 s", 1), ("10 s", 1), ("ACI width at 10 s (m)", 1),
                   (r"Whole line\textsuperscript{1} (\%)", 1), (r"Collided by 10 s (\%)", 1)])]
    notes = [fn(1) + r" Share of the ACI online intervals at a level $1 - \alpha_t \geq 1$, which are the whole line "
             r"(Gibbs and Cand\`es; counted as covering), largest over the horizons 1, 3, 5 and 10~s."]
    caption = (r"H3.4 in closed-loop rollouts, recorded signals, target $a$: coverage (PICP) of the 90\% interval of "
               r"the gap at horizons of 1 and 10~s, raw and with ACI online (calibrated on highD validation), median "
               r"width of the ACI online interval at 10~s, share of the ACI intervals that are the whole line and "
               r"share of the rollouts collided by 10~s, on the highD test set and NGSIM I-80.")
    return Table("c4_coverage", caption, [src], "l" + "C" * 7, heads, body, notes, wide=True, font="small")


def _e3(df: pd.DataFrame) -> pd.DataFrame:
    """Main run, recorded signals, extended set of the platoon tables."""
    out = df[(df.run == "e3") & (df.variant == "raw") & (df.set == "extended")]
    if out.empty:
        raise SystemExit("no rows of the main platoon run (e3, raw, extended)")
    return out


def c5_platoon() -> Table:
    src = "t5_h32__e3__raw.csv"
    df = _e3(pd.read_csv(CSV_DIR / src))
    check_labels(df, e3=True)
    refs = set(df.reference)
    if len(refs) != 1:
        raise SystemExit(f"{src}: references {refs}")
    ref = refs.pop()
    models = [ref] + [m for m in MODELS if m in set(df.model) and m != ref]
    if not set(df.model) <= set(models) | set(CONTROL_MODELS):
        raise SystemExit(f"{src}: models {sorted(set(df.model) - set(models) - set(CONTROL_MODELS))} not in MODELS")
    df = df[df.model.isin(models)]  # the history-only Diffusion control: tab:c14_information
    n_prof = {mode: int(pick(df, mode=mode, model=ref).n_profiles) for mode in ("ACC", "Human", "all", "mixed")}

    def reduction(r: pd.Series) -> str:
        return DASH if r.model == ref else stack(*ci(r.reduction, r.reduction_lo, r.reduction_hi))

    body = []
    for m in models:
        cells = [label(m, e3=True)]
        for mode in ("ACC", "Human"):
            r = pick(df, mode=mode, model=m)
            if m != ref and not bool(r.judged):
                raise SystemExit(f"{src}: {m}, {mode} not judged")
            cells += [num(r.crps), reduction(r), str(r.verdict)]
        cells.append(reduction(pick(df, mode="all", model=m)))
        body.append(cells)
    heads = [Head([("", 1), (f"ACC platoons ({n_prof['ACC']} profiles)", 3),
                   (f"Human-driven platoons ({n_prof['Human']})", 3), (f"All profiles ({n_prof['all']}){fn(1)}", 1)],
                  [(2, 4), (5, 7), (8, 8)]),
             Head([("Model", 1)] + [("CRPS (m/s)", 1), (head2("Reduction", r"[95\% CI]"), 1), ("Verdict", 1)] * 2
                  + [(head2("Reduction", r"[95\% CI]"), 1)])]
    notes = [fn(1) + rf" The {n_prof['ACC']} ACC, {n_prof['Human']} human-driven and {n_prof['mixed']} mixed "
             r"platoon profiles, pooled."]
    caption = (r"H3.2 per driver mode, recorded signals, extended set of leader profiles: amplitude CRPS (m/s) and "
               r"its reduction $1 - \mathrm{CRPS}/\mathrm{CRPS}_{\mathrm{ref}}$ against IDM + white noise (the "
               r"reference), paired by profile, with its 95\% bootstrap interval over profiles and the verdict "
               r"(supported: lower bound $\geq 0.20$; refuted: upper bound $< 0.05$) for the ACC and the "
               r"human-driven platoons, and the pooled reduction over all profiles.")
    spec = "lCccCccc"  # the two-line reductions and the verdicts at their natural width
    return Table("c5_platoon", caption, [src], spec, heads, body, notes, wide=True, font="footnotesize")


def c6_platoon_diag() -> Table:
    srcs = ["t6_collisions.csv", "t6_amplitude_coverage.csv", "t6_growth.csv"]
    col, cov, gro = (_e3(pd.read_csv(CSV_DIR / s)) for s in srcs)
    for df in (col, cov, gro):
        check_labels(df, e3=True)
    col, cov, gro = (df[~df.model.isin(CONTROL_MODELS)] for df in (col, cov, gro))  # R2 control: tab:c14_information
    models = [m for m in MODELS if m in set(col.model)]
    if set(models) != set(col.model) or set(models) != set(cov.model) or set(models) | {"empirical"} != set(gro.model):
        raise SystemExit("t6 tables: different model sets")
    n_profiles = int(pick(cov, model=models[0]).n_profiles)
    per = {m: int(pick(col, model=m).n_rollouts) / (n_profiles * 5) for m in models}
    if any(p != int(p) for p in per.values()):
        raise SystemExit(f"rollouts per profile and seed not whole: {per}")
    usual = max(per.values())
    body: list = []
    notes = [fn(1) + r" Vehicle 1 during the 30~s constant-speed burn-in before the profile."]
    for m in models:
        c, a, g = pick(col, model=m), pick(cov, model=m), pick(gro, model=m)
        if (int(g.fit_first_profile), int(g.fit_last_profile)) != (1, 20):
            raise SystemExit(f"{m}: growth fitted over {g.fit_first_profile}-{g.fit_last_profile}")
        name = label(m, e3=True)
        if per[m] != usual:
            k_roll = len(notes) + 1
            name += fn(k_roll)
            notes.append(fn(k_roll) + rf" {thousands(pick(col, model=m).n_rollouts)} rollouts ({int(per[m])} per "
                         rf"profile and training seed); the other models {thousands(round(usual * n_profiles * 5))} "
                         rf"({int(usual)}).")
        body.append([name, pct(c.collided_1), pct(c.collided_5), pct(c.collided_20), pct(c.collided_burn_in_1),
                     prob(a.coverage_raw), prob(a.coverage_static_loo), prob(a.coverage_aci_online),
                     stack(*ci(g.beta_profile, g.beta_lo_profile, g.beta_hi_profile)),
                     stack(*ci(g.beta_common_profile, g.beta_common_lo_profile, g.beta_common_hi_profile))])
    e = pick(gro, model="empirical")
    k_meas = len(notes) + 1
    notes.append(fn(k_meas) + rf" Fitted over vehicles {int(e.fit_first_profile)}--{int(e.fit_last_profile)}, the "
                 r"positions measured in at least half of the profiles (the models: vehicles 1--20, extrapolated beyond "
                 r"the measured positions).")
    n_common, n_files = int(e.n_profiles_common_profile), int(e.n_files_common_profile)
    k_common = len(notes) + 1
    notes.append(fn(k_common) + rf" The {n_common} profiles with at least four measured followers ({n_files} recording "
                 r"files), vehicles 1--4, window of the measured data, for every model and the measurements alike; "
                 r"intervals over files in Supplementary File S1.")
    body += [RULE, ["Measured platoons" + fn(k_meas)] + [DASH] * 7
             + [stack(*ci(e.beta_profile, e.beta_lo_profile, e.beta_hi_profile)),
                stack(*ci(e.beta_common_profile, e.beta_common_lo_profile, e.beta_common_hi_profile))]]
    heads = [Head([("", 1), (r"Collisions (\% of rollouts)", 4), (r"Coverage of the 90\% amplitude interval", 3),
                   (r"Growth exponent $\beta$ [95\% CI]", 2)], [(2, 5), (6, 8), (9, 10)]),
             Head([("Model", 1), ("Veh.~1", 1), ("Veh.~5", 1), ("Veh.~20", 1), ("Burn-in" + fn(1), 1),
                   ("Raw", 1), ("Static", 1), ("ACI online", 1), ("Vehicles 1--20", 1),
                   ("Common support" + fn(k_common), 1)])]
    caption = (r"Platoon diagnostics, main run, recorded signals, extended set of leader profiles: share of the "
               r"rollouts in which vehicle 1, 5 or 20 collided (whole time line) and in which vehicle 1 collided "
               r"during the burn-in; coverage of the 90\% interval of the speed amplitude over all measured "
               r"vehicles, raw, with the leave-one-profile-out static correction and with ACI online; growth "
               r"exponent $\beta$ of the speed standard deviation along the platoon "
               r"($\mathrm{sd}_i = \mathrm{sd}_1\, i^{\beta}$, profile window) with its 95\% bootstrap interval over "
               r"profiles, over vehicles 1--20 and on the common support of the measured platoons.")
    spec = "l" + "C" * 7 + "cc"  # the exponents with their intervals at their natural width
    return Table("c6_platoon_diag", caption, srcs, spec, heads, body, notes, wide=True, font="footnotesize", tabcolsep=3)


def c7_e5() -> Table:
    variants = ("raw", "harmonised", "matched")
    srcs = [f"t1_one_step__{v}__a.csv" for v in variants]
    frames = {v: load(s, variant=v, target="a") for v, s in zip(variants, srcs)}
    for df in frames.values():
        check_labels(df)
        check_seeds(df[df.dataset.isin(["highd", "ngsim"]) & df.model.isin(MODELS)])
    body = []
    for m in MODELS:
        cells = [label(m)]
        for v in variants:
            r = pick(frames[v], model=m, dataset="ngsim")
            cells += [num(r.nll_mean), prob(r.picp90_mean)]
        h_raw, h_harm, h_match = (float(pick(frames[v], model=m, dataset="highd").nll_mean) for v in variants)
        if not math.isclose(h_harm, h_match, rel_tol=1e-9, abs_tol=1e-12):
            raise SystemExit(f"{m}: highD NLL harmonised {h_harm} != matched {h_match}")
        cells += [num(h_raw), num(h_harm)]
        body.append(cells)
    heads = [Head([("", 1), ("NGSIM I-80", 6), ("highD test", 2)], [(2, 7), (8, 9)]),
             Head([("", 1), ("Recorded", 2), ("Harmonised", 2), ("Matched", 2), ("NLL", 2)],
                  [(2, 3), (4, 5), (6, 7), (8, 9)]),
             Head([("Model", 1)] + [("NLL", 1), ("PICP", 1)] * 3 + [("Recorded", 1), ("Harmonised" + fn(1), 1)])]
    notes = [fn(1) + r" Also the in-domain reference of the matched signals: they are evaluated with the models "
             r"trained on harmonised highD, without retraining."]
    caption = (r"E5, NGSIM I-80, target $a$: one-step NLL (nats) and PICP of the 90\% interval per model for the "
               r"recorded, the harmonised and the roughness-matched signals, with the in-domain NLL on the highD "
               r"test set; mean over five training seeds.")
    return Table("c7_e5", caption, srcs, "l" + "C" * 6 + "cc", heads, body, notes, wide=True, font="small")


def c8_e6() -> Table:
    roots = (("lodo_ngsim", ("ngsim",)), ("lodo_ngsim_all", ("ngsim", "ngsim_us101")), ("lodo_openacc", ("openacc",)))
    srcs = [f"t1_one_step__{root}__a.csv" for root, _ in roots] + ["t1_one_step__matched__a.csv"]
    frames = {root: load(s, variant=root, target="a") for (root, _), s in zip(roots, srcs)}
    matched = load(srcs[-1], variant="matched", target="a")
    for df in (*frames.values(), matched):
        check_labels(df)
        check_seeds(df[df.model.isin(MODELS)])

    def cell(gap: float, picp: float) -> str:
        return f"{num(gap, sig=3, max_dec=2, sign=True)} ({prob(picp)})"

    body = []
    for m in MODELS:
        cells = [label(m)]
        for root, held in roots:
            df = frames[root]
            ref = float(pick(df, model=m, dataset="pool_test").nll_mean)
            for d in held:
                r = pick(df, model=m, dataset=d)
                cells.append(cell(float(r.nll_mean) - ref, r.picp90_mean))
        ref = float(pick(matched, model=m, dataset="highd").nll_mean)
        for d in ("ngsim", "openacc"):
            r = pick(matched, model=m, dataset=d)
            cells.append(cell(float(r.nll_mean) - ref, r.picp90_mean))
        body.append(cells)
    heads = [Head([("", 1), ("Pooled domain, held-out data set", 4), ("highD only (E5)", 2)], [(2, 5), (6, 7)]),
             Head([("", 1), ("Without NGSIM I-80", 1), ("Without both NGSIM sets", 2), ("Without OpenACC", 1),
                   ("", 1), ("", 1)], [(2, 2), (3, 4), (5, 5)]),
             Head([("Model", 1), ("NGSIM I-80", 1), ("NGSIM I-80", 1), ("NGSIM US-101", 1), ("OpenACC", 1),
                   ("NGSIM I-80", 1), ("OpenACC", 1)])]
    caption = (r"E6 transfer gaps, target $a$, matched signals: one-step NLL (nats) on a held-out data set minus the "
               r"NLL on the pooled test folds of the training domain, for the three leave-one-domain-out trainings "
               r"(the five data sets pooled, without the held-out ones), next to the gap to the highD test set of "
               r"the models trained on highD only (E5); PICP of the 90\% interval on the held-out set in "
               r"parentheses; mean over five training seeds.")
    return Table("c8_e6", caption, srcs, "l" + "C" * 6, heads, body, [], wide=True, font="small")


# ------------------------------------------------------------------------------- review fixes: c10 - c12
E3_VARIANTS = (("raw", "Recorded signals, highD training"), ("harmonised", "Harmonised signals, highD training"),
               ("lodo_openacc", "Matched signals, pooled training without OpenACC (OpenACC held out)"),
               ("pool", "Matched signals, pooled training (five data sets)"))  # fmt: skip
E3_LEARNED = ("diffusion_cf", "gp_residual_idm", "quantile_pidl", "deep_ensemble", "lstm_const_sigma")
E3_BASELINES = ("idm_ar_noise",)  # the second baseline (review control R10), right after IDM + white noise
VERDICT_SHORT = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc.", "not judged": "n.j."}


def _bounds(lo: float, hi: float, like: float) -> str:
    """'[lo, hi]' with the decimals of the estimate ``like`` (as ci())."""
    return ci(like, lo, hi)[1]


def c10_h32_cluster() -> Table:
    src = "t5_h32_cluster__e3.csv"
    df = pd.read_csv(CSV_DIR / src)
    body: list = []
    n = {}
    for mode in ("ACC", "Human"):
        sel = df[(df["mode"] == mode) & (df.variant == "raw")]
        n[mode] = (int(sel.n_profiles.max()), int(sel.n_files.max()))
    for variant, title in E3_VARIANTS:
        if variant not in set(df.variant):
            raise SystemExit(f"{src}: no rows of variant {variant}")
        body.append(Group(title))
        for m in (*E3_BASELINES, *E3_LEARNED):
            if m in E3_BASELINES and df[(df.variant == variant) & (df.model == m)].empty:
                continue  # IDM + AR(1) noise: not run with the harmonised signals (no checkpoint in artifacts/e1h)
            a, h = pick(df, variant=variant, mode="ACC", model=m), pick(df, variant=variant, mode="Human", model=m)
            est_a, ci_a = ci(a.reduction, a.lo, a.hi)
            est_h, _ = ci(h.reduction, h.lo, h.hi)
            body.append([tex(str(a.model_label)), est_a, ci_a, _bounds(a.lo_cluster, a.hi_cluster, a.reduction),
                         f"{VERDICT_SHORT[a.verdict]} / {VERDICT_SHORT[a.verdict_cluster]}", est_h,
                         _bounds(h.lo_cluster, h.hi_cluster, h.reduction),
                         f"{VERDICT_SHORT[h.verdict]} / {VERDICT_SHORT[h.verdict_cluster]}"])
    heads = [Head([("", 1), (f"ACC platoons ({n['ACC'][0]} profiles, {n['ACC'][1]} files)", 4),
                   (f"Human-driven platoons ({n['Human'][0]} profiles, {n['Human'][1]} files)", 3)], [(2, 5), (6, 8)]),
             Head([("Model", 1), ("Reduction", 1), (head2(r"95\% CI", "profiles"), 1), (head2(r"95\% CI", "files"), 1),
                   (head2("Verdict", "profiles / files"), 1), ("Reduction", 1),
                   (head2(r"95\% CI", "files"), 1), (head2("Verdict", "profiles / files"), 1)])]
    notes = [r"Verdict: supp.\ = supported (lower bound $\geq 0.20$), ref.\ = refuted (upper bound $< 0.05$), inc.\ = "
             r"inconclusive. Files: paired bootstrap over the recording files with all their profiles (10\,000 "
             r"replicates; the profile-weighted estimate is unchanged). The reductions of the five training seeds alone "
             r"are given in Supplementary File S1."]
    caption = (r"H3.2 with the dependence of the platoon profiles (extended set, window profile + tail): reduction "
                r"$1 - \mathrm{CRPS}/\mathrm{CRPS}_{\mathrm{ref}}$ of the amplitude CRPS against IDM + white noise of "
                r"the same variant, with the 95\% interval of the bootstrap over profiles and over recording files, the "
                r"verdict by each interval, per signal and training variant.")
    return Table("c10_h32_cluster", caption, [src], "lCcccCcc", heads, body, notes, wide=True, font="footnotesize",
                 tabcolsep=3)


def c11_collisions() -> Table:
    src = "t6_collision_diagnostics.csv"
    df = pd.read_csv(CSV_DIR / src)
    df = df[df["mode"] == "all"]
    models = ("idm_white_noise", *E3_BASELINES, *E3_LEARNED)
    body: list = []
    for variant, title in (E3_VARIANTS[0], E3_VARIANTS[2], E3_VARIANTS[3]):
        body.append(Group(title))
        for m in models:
            if m in E3_BASELINES and df[(df.variant == variant) & (df.model == m)].empty:
                continue
            r = pick(df, variant=variant, model=m)
            first = DASH if not np.isfinite(r.ttfc_median_s) else \
                f"{r.ttfc_median_s:.0f} [{r.ttfc_p10_s:.0f}, {r.ttfc_p90_s:.0f}]"  # whole seconds  # fmt: skip
            body.append([tex(str(r.model_label)), pct(r.collision_free_share), first, pct(r.collided_in_burn_in_share),
                         pct(r.collided_v1), pct(r.collided_v20), num(r.crps_1_4), num(r.crps_1_4_survivors_biased),
                         pct(r.kept_share_1_4)])
    heads = [Head([("", 1), ("Platoon rollouts", 3), (r"Collided (\%)", 2), (r"Amplitude CRPS, vehicles 1--4 (m/s)", 3)],
                  [(2, 4), (5, 6), (7, 9)]),
             Head([("Model", 1), (head2("Collision-", r"free (\%)"), 1), (head2("First", "collision (s)"), 1),
                   (head2("In the", r"burn-in (\%)"), 1), (head2("Vehicle", "1"), 1), (head2("Vehicle", "20"), 1),
                   (head2("All", "rollouts"), 1), (head2("Survivors", "(biased)"), 1), (head2("Kept", r"(\%)"), 1)])]
    notes = [r"The engine flags a collision (gap $\leq 0$) and continues the motion; nothing is removed. Collision-free: "
             r"no vehicle of the platoon collides over the whole time line. First collision: time of the first collision "
             r"of any vehicle, median [10\%, 90\%] over the colliding rollouts, from the start of the time line (the first 30~s are the "
             r"constant-speed burn-in). Survivors: the CRPS over the rollouts in which neither the scored vehicle nor "
             r"one ahead of it collided, a survivor-biased score, with the share of rollouts kept."]
    caption = (r"Collision diagnostics of the platoons, extended set of leader profiles, for the recorded signals with "
               r"highD training and the two pooled trainings (matched signals): collision-free platoon rollouts, time to "
               r"the first collision, collisions of vehicles 1 and 20, and the amplitude CRPS before and after removing "
               r"the rollouts with a collision.")
    return Table("c11_collisions", caption, [src], "lCcCCCCCC", heads, body, notes, wide=True, font="footnotesize",
                 tabcolsep=2)


def c12_info_budget() -> Table:
    src = "t4_info_budget__raw__a.csv"
    df = load(src, variant="raw", target="a")
    check_labels(df)
    cols = (("static", "Static"), ("static_prequential", "Static preq."), ("aci_online", "ACI"),
            ("aci_online_chrono", "ACI chrono."), ("aci_online_delayed", "ACI delayed"))
    body: list = []
    delays = {}
    for m in MODELS:
        cells = [label(m)]
        for d in ("highd", "ngsim"):
            r = pick(df, dataset=d, model=m)
            cells += [prob(r[f"picp_{v}_10s"]) for v, _ in cols]
            cells.append(prob(r["picp_first100_aci_online_delayed_10s"]))
            delays[d] = float(r["delay_events_mean_aci_online_delayed_10s"])
        body.append(cells)
    sub = [(head2(*lab.split(" ", 1)) if " " in lab else lab, 1) for _, lab in cols] + [(head2("First", "100"), 1)]
    heads = [Head([("", 1), ("highD test", 6), ("NGSIM I-80", 6)], [(2, 7), (8, 13)]), Head([("Model", 1)] + sub * 2)]
    notes = [r"Static: the conformal correction of highD validation; static preq.\ (prequential): recalibrated on every label of the "
             r"evaluated stream seen so far (the labels ACI uses, no learning rate); ACI: online in a random order; "
             r"chrono.: in chronological order; delayed: chronological, a label used only 10~s after its interval was "
             rf"issued (NGSIM I-80: event start times, {num(delays['ngsim'], 2)} events in flight on average; highD: no "
             rf"comparable times, {num(delays['highd'], 2)} events). First 100: coverage of delayed ACI over the first 100 "
             r"events of the stream."]
    caption = (r"Information budget of the corrections, recorded signals, target $a$: coverage of the 90\% gap interval at "
               r"10~s of the static correction, of the static correction recalibrated on the same labels as ACI, and of "
               r"ACI online in a random order, in chronological order, and with delayed labels, on the highD test set and "
               r"NGSIM I-80.")
    return Table("c12_info_budget", caption, [src], "l" + "C" * 12, heads, body, notes, wide=True, font="footnotesize",
                 tabcolsep=2)


# ------------------------------------------------------------------------- review controls (v4): c13 - c18
# The tables of the control experiments of the v4 revision (docs/design.md 18, docs/review_fixes_v4.md). They read
# the tables of the experiment roots (artifacts/<root>/tables): the control roots e1_ctrl, e2_ctrl and e1m_causal are
# not roots of scripts/make_report.py. Every number is the one of the root's analysis: one-step metrics as T1 (mean
# over the five training seeds of one_step_metrics.csv; checked against the report where the root is a report root),
# rollouts from coverage_by_horizon.csv as T4, platoons from h32.csv, amplitude_crps.csv, collision_diagnostics.csv
# and growth.csv of artifacts/e3 (extended set, window profile + tail, reference IDM + white noise of the variant).
ART = ART  # set above from CF_UNC_ARTIFACTS
_ART: dict[str, pd.DataFrame] = {}
_E3_INFO: dict = {}
E3_SET, E3_WINDOW, E3_REF = "extended", "response", "idm_white_noise"
ONESTEP_SETS = ("highd", "ngsim", "openacc")  # the one-step data sets of c14 and c15


def art_name(root: str, name: str) -> str:
    return f"artifacts/{root}/tables/{name}"


def art(root: str, name: str) -> pd.DataFrame:
    """A table of an experiment root (read once)."""
    key = art_name(root, name)
    if key not in _ART:
        path = REPO / key
        if not path.exists():
            raise SystemExit(f"{key}: missing (run the analysis of artifacts/{root})")
        _ART[key] = pd.read_csv(path, low_memory=False)
    return _ART[key]


def onestep(root: str, model: str, dataset: str, n_seeds: int = 5) -> dict[str, float]:
    """One-step metrics of an E1 root: the mean over the training seeds of the per-seed rows of one_step_metrics.csv
    (subset all), as T1 of scripts/make_report.py."""
    df = art(root, "one_step_metrics.csv")
    g = df[(df["subset"].astype(str) == "all") & (df["seed"].astype(str) != "mean") & (df["model"] == model)
           & (df["dataset"] == dataset)]
    if len(g) != n_seeds or g["seed"].astype(str).nunique() != n_seeds:
        raise SystemExit(f"{art_name(root, 'one_step_metrics.csv')}: {len(g)} seed rows of {model} on {dataset}, "
                         f"expected {n_seeds}")
    out = {m: float(np.mean(pd.to_numeric(g[m], errors="coerce").to_numpy(np.float64)))
           for m in ("nll", "crps", "picp90", "width90", "rmse")}
    out.update(n_windows=int(g["n_windows"].iloc[0]), n_events=int(g["n_events"].iloc[0]))
    return out


def check_report_onestep(root_variant: str, target: str, model: str, dataset: str, x: dict[str, float]) -> None:
    """The one-step numbers of a report root are those of the report (T1)."""
    r = pick(load(f"t1_one_step__{root_variant}__{target}.csv", variant=root_variant, target=target),
             model=model, dataset=dataset)
    for key in ("nll", "crps", "picp90"):
        if not math.isclose(x[key], float(r[f"{key}_mean"]), rel_tol=1e-9, abs_tol=1e-12):
            raise SystemExit(f"{model} on {dataset}: {key} {x[key]} differs from the report ({r[f'{key}_mean']})")


def rollout(root: str, model: str, dataset: str, target: str = "a") -> dict[str, float]:
    """Rollout numbers of an E2 root (coverage_by_horizon.csv, 500 events x 5 seeds): raw PICP of the 90 % gap
    interval at 1 and 10 s, ACI online PICP and median width at 10 s, CRPS of the gap and share of the rollouts
    collided by 10 s (raw rows), as T4 of scripts/make_report.py."""
    c = art(root, "coverage_by_horizon.csv")
    g = c[(c["model"] == model) & (c["target"] == target) & (c["dataset"] == dataset)]
    if g.empty or set(g["n_seeds"].astype(int)) != {5} or set(g["n_events"].astype(int)) != {500}:
        raise SystemExit(f"{art_name(root, 'coverage_by_horizon.csv')}: no complete rows of {model} on {dataset}")

    def one(variant: str, h: float, col: str) -> float:
        return float(pick(g, variant=variant, horizon_s=h)[col])

    return {"picp_raw_1s": one("raw", 1.0, "picp"), "picp_raw_10s": one("raw", 10.0, "picp"),
            "picp_aci_10s": one("aci_online", 10.0, "picp"), "width_aci_10s": one("aci_online", 10.0, "width_median"),
            "crps_10s": one("raw", 10.0, "crps"), "collided_10s": one("raw", 10.0, "collided")}


def check_report_rollout(model: str, dataset: str, x: dict[str, float]) -> None:
    """The rollout numbers of artifacts/e2 are those of the report (T4, tab:c4_coverage)."""
    r = pick(load("t4_coverage__raw__a.csv", variant="raw", target="a"), model=model, dataset=dataset)
    for key, col in (("picp_raw_1s", "picp_raw_1s"), ("picp_raw_10s", "picp_raw_10s"),
                     ("picp_aci_10s", "picp_aci_online_10s"), ("width_aci_10s", "width_median_aci_online_10s"),
                     ("collided_10s", "collided_10s")):
        if not math.isclose(x[key], float(r[col]), rel_tol=1e-9, abs_tol=1e-12):
            raise SystemExit(f"{model} on {dataset}: {key} {x[key]} differs from the report ({r[col]})")


def e3_info() -> dict:
    """analysis_info.json of artifacts/e3: settings (bootstrap, profiles file) and the profile sets."""
    if not _E3_INFO:
        import json

        _E3_INFO.update(json.loads((ART / "e3" / "tables" / "analysis_info.json").read_text(encoding="utf-8")))
    return _E3_INFO


def e3_has(variant: str, model: str) -> bool:
    h = art("e3", "h32.csv")
    if model == E3_REF:
        return bool((h["variant"] == variant).any())
    return bool(((h["variant"] == variant) & (h["model"] == model)).any())


def e3_h32(variant: str, model: str, mode: str) -> pd.Series:
    return pick(art("e3", "h32.csv"), variant=variant, set=E3_SET, mode=mode, window=E3_WINDOW, quantity="amplitude",
                reference=E3_REF, model=model)


def e3_crps(variant: str, model: str, mode: str) -> pd.Series:
    return pick(art("e3", "amplitude_crps.csv"), variant=variant, model=model, set=E3_SET, mode=mode,
                window=E3_WINDOW, index="all")


def e3_coll(variant: str, model: str, mode: str) -> pd.Series:
    return pick(art("e3", "collision_diagnostics.csv"), variant=variant, model=model, set=E3_SET, mode=mode)


def e3_growth(variant: str, model: str) -> pd.Series:
    """Growth exponents of a model (growth.csv, window of the measured data: the common support of R11)."""
    g = art("e3", "growth.csv")
    w = g[(g["variant"] == variant) & (g["model"] == model) & (g["set"] == E3_SET) & (g["window"] == "profile")]
    if w.empty:
        raise SystemExit(f"{art_name('e3', 'growth.csv')}: no rows of {model} in variant {variant}")
    return w.sort_values("index").iloc[0]


def e3_mode_ids(mode: str) -> list[str]:
    """Profile ids of a driver mode in the extended set (profiles file of the E3 analysis, file order)."""
    from cf_unc.platoon.profiles import load_profiles

    info = e3_info()
    profiles, _ = load_profiles(REPO / info["settings"]["profiles"])
    extended = set(info["sets"][E3_SET])
    return [p.id for p in profiles if p.id in extended and p.mode == mode]


def e3_cell(variant: str, model: str, mode: str) -> dict[str, Any]:
    """The H3.2 reduction of one platoon cell recomputed from the per-profile scores of profile_index.csv (mean over
    the measured vehicles of the amplitude CRPS, window profile + tail, as scripts/analyze_e3.py), with the replicates
    of its paired bootstraps over files and over profiles; the seeds and replicate counts are those of the analysis,
    so that cells with the same profiles share their draws. The estimate and both intervals of h32.csv are
    reproduced (checked)."""
    from cf_unc.eval.bootstrap import bootstrap_ratio_by_group
    from cf_unc.platoon.evaluate import boot_seed

    b = e3_info()["settings"]["bootstrap"]
    ids = e3_mode_ids(mode)
    pi = art("e3", "profile_index.csv")

    def scores(m: str) -> np.ndarray:
        sub = pi[(pi["variant"] == variant) & (pi["model"] == m)]
        return sub.groupby("profile")["crps_response"].mean().reindex(ids).to_numpy(np.float64)

    m, r = scores(model), scores(E3_REF)
    if not (np.isfinite(m).all() and np.isfinite(r).all()):
        raise SystemExit(f"e3 {variant} / {model} / {mode}: a profile without a score (the paired draws would differ)")
    files = pi.drop_duplicates("profile").set_index("profile")["source_file"].reindex(ids).to_numpy()
    by_file = bootstrap_ratio_by_group(m, r, files, n_boot=int(b["n_boot_cluster"]), seed=int(b["cluster_seed"]),
                                       ci=float(b["ci"]))
    by_prof = bootstrap_ratio_by_group(m, r, np.asarray(ids), n_boot=int(b["n_boot"]),
                                       seed=boot_seed(int(b["seed"]), "h32", E3_SET, mode, E3_WINDOW, "amplitude"),
                                       ci=float(b["ci"]))
    row = e3_h32(variant, model, mode)
    got = (1.0 - by_file["estimate"], 1.0 - by_prof["hi"], 1.0 - by_prof["lo"], 1.0 - by_file["hi"], 1.0 - by_file["lo"])
    want = (row.reduction, row.lo, row.hi, row.lo_cluster, row.hi_cluster)
    if not all(math.isclose(a, float(w), rel_tol=1e-9, abs_tol=1e-12) for a, w in zip(got, want)):
        raise SystemExit(f"e3 {variant} / {model} / {mode}: recomputed reduction {got} differs from h32.csv {want}")
    return {"estimate": got[0], "profiles": 1.0 - by_prof["replicates"], "files": 1.0 - by_file["replicates"],
            "ci": float(b["ci"])}


def stack_lines(*lines: str) -> str:
    """Several lines in one cell (a nested tabular, as stack())."""
    return r"\begin{tabular}[c]{@{}c@{}}" + r"\\{}".join(lines) + r"\end{tabular}"


def reduction_cell(r: pd.Series, verdict: bool = True, collided: float | None = None) -> str:
    """H3.2 reduction with the verdict by the file interval (and, given ``collided``, the share of the rollouts with
    a collision of vehicle 1 in parentheses), its 95 % interval over profiles and over files (three lines)."""
    est, prof = ci(r.reduction, r.lo, r.hi)
    top = f"{est} {VERDICT_SHORT[str(r.verdict_cluster)]}" if verdict else est
    if collided is not None:
        top += f" ({pct(collided)})"
    return stack_lines(top, prof, _bounds(r.lo_cluster, r.hi_cluster, r.reduction))


def small(x: float, sig: int = 2) -> str:
    """A small positive score (one-step CRPS in m/s^2): ``sig`` significant digits below 0.1, else as num()."""
    x = float(x)
    return num(x, sig=sig, max_dec=None) if np.isfinite(x) and abs(x) < 0.1 else num(x)


def pval(p: float) -> str:
    """A (Holm-adjusted) p-value with two significant digits; below 0.001 as '<0.001'."""
    p = float(p)
    if not np.isfinite(p):
        return DASH
    return r"$<$0.001" if p < 0.001 else num(p, sig=2, max_dec=None)


VERDICT_NOTE = (r"supp.\ = supported (lower bound $\geq 0.20$), ref.\ = refuted (upper bound $< 0.05$), inc.\ = "
                r"inconclusive")


# ------------------------------------------------------------------------------------- c13: the R3 matrix
C13_MODELS = ("diffusion_cf", "gp_residual_idm", "quantile_pidl", "deep_ensemble", "lstm_const_sigma", "idm_ar_noise")
C13_SHORT = {"diffusion_cf": "Diffusion", "gp_residual_idm": "GP residual", "quantile_pidl": "Quantile PIDL",
             "deep_ensemble": "Deep ensemble", "lstm_const_sigma": "LSTM", "idm_ar_noise": "IDM + AR(1)"}
# the cells (variant of artifacts/e3, training, leader window); QuantilePIDL on target a in every cell, in the
# highD x 11 cell from the same-target control run raw_a (design.md 18.2)
C13_CELLS = (("raw", "highd", 11), ("raw_w61", "highd", 61), ("pool_w11", "pool", 11), ("pool", "pool", 61),
             ("lodo_openacc_w11", "lodo", 11), ("lodo_openacc", "lodo", 61))
C13_SAME_TARGET = {("raw", "quantile_pidl"): "raw_a"}


def c13_variant(variant: str, model: str) -> str:
    return C13_SAME_TARGET.get((variant, model), variant)


def c13_effects(model: str = "diffusion_cf", mode: str = "ACC") -> dict[str, dict[str, tuple[float, ...]]]:
    """Paired differences of the H3.2 reduction of ``model`` between the cells of the matrix: per effect and cell
    (estimate, profile interval lo, hi, file interval lo, hi). Domain: pooled minus highD training at the same leader
    window; smoothing: window 61 minus 11 with the same training; interaction: the smoothing effect of the pooled
    training minus that of the highD training. The intervals are percentiles of the differences of the replicates
    (the same draws in every cell)."""
    cell = {v: e3_cell(c13_variant(v, model), model, mode) for v, _, _ in C13_CELLS}
    level = cell["raw"]["ci"]
    q = (0.5 * (1.0 - level), 1.0 - 0.5 * (1.0 - level))

    def diff(plus: Sequence[str], minus: Sequence[str]) -> tuple[float, ...]:
        out = [sum(cell[v]["estimate"] for v in plus) - sum(cell[v]["estimate"] for v in minus)]
        for kind in ("profiles", "files"):
            reps = sum(cell[v][kind] for v in plus) - sum(cell[v][kind] for v in minus)
            lo, hi = np.nanquantile(np.where(np.isfinite(reps), reps, np.nan), q)
            out += [float(lo), float(hi)]
        return tuple(out)

    return {
        "domain": {"pool_w11": diff(["pool_w11"], ["raw"]), "pool": diff(["pool"], ["raw_w61"]),
                   "lodo_openacc_w11": diff(["lodo_openacc_w11"], ["raw"]),
                   "lodo_openacc": diff(["lodo_openacc"], ["raw_w61"])},
        "smoothing": {"raw_w61": diff(["raw_w61"], ["raw"]), "pool": diff(["pool"], ["pool_w11"]),
                      "lodo_openacc": diff(["lodo_openacc"], ["lodo_openacc_w11"])},
        "interaction": {"pool": diff(["pool", "raw"], ["pool_w11", "raw_w61"]),
                        "lodo_openacc": diff(["lodo_openacc", "raw"], ["lodo_openacc_w11", "raw_w61"])},
    }


def effect_cell(t: tuple[float, ...]) -> str:
    """Effect with its interval over profiles and over files (three lines, as reduction_cell)."""
    return stack_lines(num(t[0], sign=True), _bounds(t[1], t[2], t[0]), _bounds(t[3], t[4], t[0]))


def c13_matrix() -> tuple[Table, list[str]]:
    srcs = [art_name("e3", n) for n in ("h32.csv", "collision_diagnostics.csv", "profile_index.csv")]
    for v, _, _ in C13_CELLS:
        if not e3_has(v, E3_REF):
            raise SystemExit(f"artifacts/e3: no rows of variant {v} (run scripts/analyze_e3.py)")
    n = {mode: (int(e3_h32("raw", "diffusion_cf", mode).n_profiles), int(e3_h32("raw", "diffusion_cf", mode).n_files))
         for mode in ("ACC", "Human")}
    body: list = []
    for mode, title in (("ACC", "ACC platoons"), ("Human", "human-driven platoons")):
        body.append(Group(f"H3.2 reduction against IDM + white noise of the cell, {title} ({n[mode][0]} profiles, "
                          f"{n[mode][1]} files)"))
        for m in C13_MODELS:
            cells = [C13_SHORT[m] + (fn(1) if m == "quantile_pidl" else "")]
            for v, _, _ in C13_CELLS:
                vv = c13_variant(v, m)
                cells.append(reduction_cell(e3_h32(vv, m, mode), collided=e3_coll(vv, m, mode).collided_v1)
                             if e3_has(vv, m) else DASH)
            body.append(cells)
    eff = c13_effects("diffusion_cf", "ACC")
    body.append(Group(r"Effects on Diffusion's reduction, ACC platoons: paired differences between the cells" + fn(3)))
    for name, key in (("Domain", "domain"), ("Smoothing", "smoothing"), ("Interaction", "interaction")):
        body.append([name] + [effect_cell(eff[key][v]) if v in eff[key] else DASH for v, _, _ in C13_CELLS])
    log = []
    for key, cells in eff.items():
        for v, t in cells.items():
            log.append(f"{key:11s} {v:17s} {t[0]:+.3f} profiles [{t[1]:+.3f}, {t[2]:+.3f}] files [{t[3]:+.3f}, {t[4]:+.3f}]")
    heads = [Head([("", 1), ("highD training", 2), ("Pooled training, OpenACC seen", 2),
                   ("Pooled training, OpenACC held out", 2)], [(2, 3), (4, 5), (6, 7)]),
             Head([("Model" + fn(2), 1)] + [(head2("Leader", "window 11"), 1), (head2("Leader", "window 61"), 1)] * 3)]
    notes = [fn(1) + r" On its one-step target $a$ in every cell (highD training, window 11: the same-target run; the "
                     r"main platoon run with recorded signals uses its 1~s target).",
             fn(2) + r" Each reduction cell: the reduction $1 - \mathrm{CRPS}/\mathrm{CRPS}_{\mathrm{ref}}$ of the "
                     r"amplitude CRPS (extended set, window profile + tail) against IDM + white noise of the same cell "
                     r"and the verdict by the file interval: " + VERDICT_NOTE + r"; second line: 95\% interval of "
                     r"the bootstrap over profiles; third line: over recording files. Leader window 11: the leader "
                     r"speed of the profiles file; 61: smoothed again from the raw OpenACC file with the matched "
                     r"OpenACC window of E5. In parentheses: share of the rollouts in which vehicle 1 collided "
                     r"(\%, whole time line). Deep ensemble: the member kept per rollout.",
             fn(3) + r" Domain: pooled minus highD training at the same leader window; smoothing: window 61 minus 11 "
                     r"with the same training; interaction: the smoothing effect of the pooled training minus that of "
                     r"the highD training. Intervals (profiles, then files) from the paired draws of the cells."]
    caption = (r"The platoon matrix of training domain $\times$ leader smoothing, all models on "
               r"their target-$a$ checkpoints: H3.2 reduction of the amplitude CRPS against IDM + white noise of the "
               r"same cell for the ACC and the human-driven platoons, with the intervals over profiles and over "
               r"recording files and the first-follower collisions, and the effects of the training domain and of "
               r"the leader smoothing on Diffusion's reduction for the ACC platoons.")
    return Table("c13_matrix", caption, srcs, "lCCCCCC", heads, body, notes, wide=True, font="footnotesize",
                 tabcolsep=3), log


# ------------------------------------------------------------------------------- c14: information (R2)
C14_MODELS = ("diffusion_cf", "diffusion_cf_hist", "lstm_const_sigma", "hetero_mdn", "deep_ensemble")
C14_HEAD = {"diffusion_cf": ("Diffusion", ""), "diffusion_cf_hist": ("Diffusion,", "history only"),
            "lstm_const_sigma": ("LSTM,", "constant sigma"), "hetero_mdn": ("HeteroMDN", ""),
            "deep_ensemble": ("Deep", "ensemble")}
C14_E1 = {"diffusion_cf_hist": "e1_ctrl"}  # the other models: artifacts/e1
C14_E2 = {"diffusion_cf_hist": "e2_ctrl"}  # the other models: artifacts/e2


def _head_model(top: str, bottom: str) -> str:
    return head2(top, bottom) if bottom else top


def c14_information() -> Table:
    srcs = [art_name("e1", "one_step_metrics.csv"), art_name("e1_ctrl", "one_step_metrics.csv"),
            art_name("e2", "coverage_by_horizon.csv"), art_name("e2_ctrl", "coverage_by_horizon.csv"),
            art_name("e3", "h32.csv"), art_name("e3", "amplitude_crps.csv"), art_name("e3", "collision_diagnostics.csv")]
    body: list = [Group(r"One-step prediction (E1), mean over five training seeds")]
    for key, name, f in (("nll", "NLL (nats)", num), ("crps", r"CRPS (m/s$^2$)", small), ("picp90", "PICP", prob)):
        for i, d in enumerate(ONESTEP_SETS):
            cells = [name if i == 0 else "", DATASET_LABEL[d]]
            for m in C14_MODELS:
                root = C14_E1.get(m, "e1")
                x = onestep(root, m, d)
                if root == "e1":
                    check_report_onestep("raw", "a", m, d, x)
                cells.append(f(x[key]))
            body.append(cells)
    body.append(Group(r"Closed-loop rollouts (E2), 90\% interval of the gap, 500 events per data set"))
    rows = (("picp_raw_1s", "PICP raw, 1~s", prob), ("picp_raw_10s", "PICP raw, 10~s", prob),
            ("picp_aci_10s", "PICP ACI online, 10~s", prob), ("crps_10s", "CRPS of the gap, 10~s (m)", num),
            ("collided_10s", r"Collided by 10~s (\%)", pct))
    for key, name, f in rows:
        for i, d in enumerate(("highd", "ngsim")):
            cells = [name if i == 0 else "", DATASET_LABEL[d]]
            for m in C14_MODELS:
                root = C14_E2.get(m, "e2")
                x = rollout(root, m, d)
                if root == "e2":
                    check_report_rollout(m, d, x)
                cells.append(f(x[key]))
            body.append(cells)
    ref = e3_h32("raw", "diffusion_cf", "ACC")
    body.append(Group(f"Platoons (E3), recorded signals, ACC platoons ({int(ref.n_profiles)} profiles, "
                      f"{int(ref.n_files)} files)"))
    in_e3 = {m: e3_has("raw", m) for m in C14_MODELS}
    body.append(["Amplitude CRPS (m/s)", "ACC"] + [num(e3_crps("raw", m, "ACC").crps) if in_e3[m] else DASH
                                                   for m in C14_MODELS])
    body.append([r"Reduction" + fn(2), "ACC"] + [reduction_cell(e3_h32("raw", m, "ACC")) if in_e3[m] else DASH
                                                 for m in C14_MODELS])
    body.append([r"Vehicle 1 collided (\%)", "ACC"] + [pct(e3_coll("raw", m, "ACC").collided_v1) if in_e3[m] else DASH
                                                       for m in C14_MODELS])
    # V7-R01 (design.md 18.8): the same comparison on the pooled roots, where the history-only model was trained
    # on the pooled domain (configs e6_pool_ctrl, e6_lodo_openacc_ctrl) and runs in the E3 variants pool / lodo_openacc
    for variant, title in (("lodo_openacc", "pooled training, OpenACC held out"), ("pool", "pooled training, OpenACC seen")):
        if not e3_has(variant, "diffusion_cf_hist"):
            continue
        refv = e3_h32(variant, "diffusion_cf", "ACC")
        body.append(Group(f"Platoons (E3), {title}, ACC platoons ({int(refv.n_profiles)} profiles, {int(refv.n_files)} files)"))
        in_v = {m: e3_has(variant, m) for m in C14_MODELS}
        body.append(["Amplitude CRPS (m/s)", "ACC"] + [num(e3_crps(variant, m, "ACC").crps) if in_v[m] else DASH
                                                       for m in C14_MODELS])
        body.append([r"Reduction" + fn(2), "ACC"] + [reduction_cell(e3_h32(variant, m, "ACC")) if in_v[m] else DASH
                                                     for m in C14_MODELS])
        body.append([r"Vehicle 1 collided (\%)", "ACC"] + [pct(e3_coll(variant, m, "ACC").collided_v1) if in_v[m] else DASH
                                                           for m in C14_MODELS])
    heads = [Head([("Quantity", 1), ("Data set" + fn(1), 1)] + [(_head_model(*C14_HEAD[m]), 1) for m in C14_MODELS])]
    notes = [fn(1) + r" One-step: Diffusion models on 10\,000 sampled windows per data set, the others on all windows; "
                     r"rollouts: 32 rollouts per event for the Diffusion models, 200 for the others; deep ensemble: "
                     r"per-step draws in the rollouts, the member kept per rollout in the platoons; HeteroMDN was not "
                     r"run in the platoons.",
             fn(2) + r" Against IDM + white noise, with the verdict by the file interval: " + VERDICT_NOTE + r"; "
                     r"second line: 95\% interval over profiles; third line: over recording files."]
    caption = (r"Equal information: Diffusion conditions its 5~s plan on the 3~s history and the "
               r"leader's speed over the next 5~s (in E1 and E2 the recorded leader, in the platoon cascade the "
               r"simulated trajectory of its predecessor); Diffusion, history only, the same DDPM without the leader "
               r"channel, sees the 3~s history only, as the one-step networks (LSTM with constant sigma, HeteroMDN, "
               r"deep ensemble). Recorded signals, target $a$: one-step metrics, rollout coverage, CRPS and collisions, "
               r"and the H3.2 reduction on the ACC platoons; the last groups repeat the platoon comparison with the "
               r"models trained on the pooled domains (matched signals, leader window 61), where the history-only "
               r"model was trained on the same pooled data as Diffusion.")
    return Table("c14_information", caption, srcs, "ll" + "C" * len(C14_MODELS), heads, body, notes, wide=True,
                 font="footnotesize")


# -------------------------------------------------------------------------------- c15: AR(1) noise (R10)
C15_MODELS = ("idm_white_noise", "idm_ar_noise", "gp_residual_idm", "diffusion_cf")
C15_HEAD = {"idm_white_noise": ("IDM +", "white noise"), "idm_ar_noise": ("IDM +", "AR(1) noise"),
            "gp_residual_idm": ("GP residual", "on IDM"), "diffusion_cf": ("Diffusion", "")}
C15_E3 = (("raw", "highD training, recorded signals", "e1"), ("pool", "pooled training, OpenACC seen", "e6_pool"),
          ("lodo_openacc", "pooled training, OpenACC held out", "e6_lodo_openacc"))


def ar_fit(root: str) -> dict[str, float]:
    """The AR(1) fit of idm_ar_noise in an E1 root (train_log.json of the checkpoints; seed independent: checked)."""
    import json

    paths = sorted((ART / root / "checkpoints" / "idm_ar_noise").glob("seed*/train_log.json"))
    if not paths:
        raise SystemExit(f"artifacts/{root}/checkpoints/idm_ar_noise: no fit log")
    fits = [json.loads(p.read_text(encoding="utf-8"))["log"] for p in paths]
    if len({(round(f["rho"], 12), round(f["sigma_eta"], 12)) for f in fits}) != 1:
        raise SystemExit(f"artifacts/{root}: the AR(1) fit differs between the seeds")
    return {k: float(fits[0][k]) for k in ("rho", "sigma_eta", "sigma_stationary", "conditional_rms")}


def c15_ar() -> tuple[Table, list[str]]:
    srcs = [art_name("e1", "one_step_metrics.csv"), art_name("e2", "coverage_by_horizon.csv"),
            art_name("e3", "h32.csv"), art_name("e3", "amplitude_crps.csv"), art_name("e3", "growth.csv"),
            art_name("e3", "collision_diagnostics.csv")]
    body: list = [Group(r"One-step prediction (E1), recorded signals, target $a$")]
    for key, name, f in (("nll", "NLL (nats)", num), ("picp90", "PICP", prob)):
        for i, d in enumerate(ONESTEP_SETS):
            cells = [name if i == 0 else "", DATASET_LABEL[d]]
            for m in C15_MODELS:
                x = onestep("e1", m, d)
                check_report_onestep("raw", "a", m, d, x)
                cells.append(f(x[key]))
            body.append(cells)
    body.append(Group(r"Closed-loop rollouts (E2), 90\% interval of the gap"))
    for key, name, f in (("picp_raw_1s", "PICP raw, 1~s", prob), ("picp_raw_10s", "PICP raw, 10~s", prob),
                         ("width_aci_10s", "ACI width, 10~s (m)", num)):
        for i, d in enumerate(("highd", "ngsim")):
            cells = [name if i == 0 else "", DATASET_LABEL[d]]
            for m in C15_MODELS:
                x = rollout("e2", m, d)
                check_report_rollout(m, d, x)
                cells.append(f(x[key]))
            body.append(cells)
    for variant, title, _ in C15_E3:
        body.append(Group(f"Platoons (E3), {title}"))
        for mode, mname in (("ACC", "ACC"), ("Human", "human")):
            body.append([f"Amplitude CRPS (m/s)", mname] + [num(e3_crps(variant, m, mode).crps) for m in C15_MODELS])
            body.append(["Reduction" + fn(1), mname] + [DASH if m == E3_REF else reduction_cell(e3_h32(variant, m, mode))
                                                       for m in C15_MODELS])
        body.append([r"$\beta$, common support" + fn(2), "all"]
                    + [stack(*ci(g.beta_common, g.beta_common_lo, g.beta_common_hi))
                       for g in (e3_growth(variant, m) for m in C15_MODELS)])
        body.append([r"Collision-free platoons (\%)", "all"] + [pct(e3_coll(variant, m, "all").collision_free_share)
                                                              for m in C15_MODELS])
    fits = {root: ar_fit(root) for _, _, root in C15_E3}
    log = [f"{root}: rho {f['rho']:.6f}, sigma_eta {f['sigma_eta']:.5f}, stationary {f['sigma_stationary']:.4f}, "
           f"conditional rms {f['conditional_rms']:.5f}" for root, f in fits.items()]
    e1f, pf, lf = fits["e1"], fits["e6_pool"], fits["e6_lodo_openacc"]
    heads = [Head([("Quantity", 1), ("Data set", 1)] + [(_head_model(*C15_HEAD[m]), 1) for m in C15_MODELS])]
    acc, hum = (e3_h32("raw", "diffusion_cf", mode) for mode in ("ACC", "Human"))
    measured = num(e3_growth("data", "empirical").beta_common, sig=2, max_dec=None)
    notes = [fn(1) + r" Against IDM + white noise of the same variant, with the verdict by the file interval: "
                     + VERDICT_NOTE + r"; second line: 95\% interval over profiles; third line: over recording files "
                     rf"(ACC: {int(acc.n_profiles)} profiles from {int(acc.n_files)} files; human-driven: "
                     rf"{int(hum.n_profiles)} from {int(hum.n_files)}).",
             fn(2) + r" Exponent of the speed standard deviation over vehicles 1--4 of the profiles with at least four "
                     rf"measured followers (measured platoons: {measured}), with its 95\% interval over profiles. "
                     r"Pooled trainings: the leader of E3-pooled (window 61); the cells with window 11 are in the matrix "
                     r"of control R3."]
    caption = (r"IDM with AR(1) acceleration noise against IDM + white noise (GP residual and "
               r"Diffusion for reference): one-step NLL and PICP, rollout coverage and ACI width, and the platoon "
               r"amplitude CRPS, H3.2 reduction, growth exponent and collision-free share. The AR(1) noise is fitted on "
               r"the training residuals of the frozen IDM: per step $\rho$ = " + f"{e1f['rho']:.4f}" + r", $\sigma_\eta$ "
               r"= " + f"{e1f['sigma_eta']:.3f}" + r"~m/s$^2$ (stationary " + f"{e1f['sigma_stationary']:.2f}"
               + r"~m/s$^2$, the white-noise sigma) with highD training; pooled training, OpenACC seen: $\rho$ = "
               + f"{pf['rho']:.4f}" + r", $\sigma_\eta$ = " + f"{pf['sigma_eta']:.3f}" + r"~m/s$^2$; OpenACC held "
               r"out: $\rho$ = " + f"{lf['rho']:.4f}" + r", $\sigma_\eta$ = " + f"{lf['sigma_eta']:.3f}" + r"~m/s$^2$.")
    return Table("c15_ar", caption, srcs, "ll" + "C" * len(C15_MODELS), heads, body, notes, wide=True,
                 font="footnotesize"), log


# ---------------------------------------------------------------------------- c16: frozen mean (R13)
def c16_frozen() -> Table:
    src_n, src_v = art_name("e1_ctrl", "h33_nll.csv"), art_name("e1_ctrl", "regime_ordering.csv")
    h = art("e1_ctrl", "h33_nll.csv")
    h = h[(h["dataset"] == "highd") & (h["subset"].astype(str) == "all")]
    fz = h[(h["model_a"] == "lstm_hetero_frozen") & (h["model_b"] == "lstm_const_sigma")]
    md = h[(h["model_a"] == "hetero_mdn") & (h["model_b"] == "lstm_const_sigma")]
    c3 = load("t3_h33_nll__raw__a.csv", variant="raw", target="a", dataset="highd")
    total = int(pick(fz, group="all").n_windows)
    if sum(int(pick(fz, group=g).n_windows) for g in REGIMES) != total:
        raise SystemExit(f"{src_n}: the regimes do not partition the windows")
    # the HeteroMDN pair has the point estimates of tab:c3_regimes (artifacts/e1; its intervals here come from the
    # larger replicate count of the larger Holm family of two pairs), and both pairs share the constant-sigma NLL
    for g in (*REGIMES, "all"):
        a, b, c = pick(md, group=g), pick(fz, group=g), pick(c3, regime=g)
        for col in ("nll_a", "nll_b", "diff"):
            if not math.isclose(float(a[col]), float(c[col]), rel_tol=1e-9, abs_tol=1e-12):
                raise SystemExit(f"{src_n}: HeteroMDN pair, {g}, {col} differs from tab:c3_regimes")
        if not math.isclose(float(a.nll_b), float(b.nll_b), rel_tol=1e-12, abs_tol=1e-12):
            raise SystemExit(f"{src_n}: {g}: the constant-sigma NLL differs between the pairs")
    v = art("e1_ctrl", "regime_ordering.csv")
    v = v[(v["dataset"] == "highd") & (v["subset"].astype(str) == "all") & (v["model"] == "lstm_hetero_frozen")]
    var, mse = v[v["quantity"] == "var"], v[v["quantity"] == "mse"]
    if set(var.reference) != {"steady"} or set(mse.reference) != {"steady"}:
        raise SystemExit(f"{src_v}: reference {set(var.reference)} / {set(mse.reference)}")
    rejects = [bool(r.reject) for q in (var, mse) for r in q.itertuples()]
    star_all = all(rejects)

    def ratio(q: pd.DataFrame, g: str) -> str:
        if g == "steady":
            return "1"
        r = pick(q, regime=g)
        return num(r.estimate) + ("" if star_all or not bool(r.reject) else "*")

    body = []
    for g in (*REGIMES, "all"):
        a, b = pick(fz, group=g), pick(md, group=g)
        row = [("All windows" if g == "all" else g.capitalize()), "100" if g == "all" else pct(a.n_windows / total),
               num(a.nll_b), num(a.nll_a), num(b.nll_a), stack(*ci(a["diff"], a["diff_lo"], a["diff_hi"])),
               pval(a.p_holm), stack(*ci(b["diff"], b["diff_lo"], b["diff_hi"])), pval(b.p_holm)]
        row += [DASH, DASH] if g == "all" else [ratio(var, g), ratio(mse, g)]
        if g == "all":
            body.append(RULE)
        body.append(row)
    heads = [Head([("", 1), ("", 1), ("NLL (nats)", 3), ("Frozen $-$ constant", 2), ("HeteroMDN $-$ constant", 2),
                   ("Frozen: ratio to steady", 2)], [(3, 5), (6, 7), (8, 9), (10, 11)]),
             Head([("Regime", 1), (head2("Share", r"(\%)"), 1), (head2("Constant", "sigma"), 1),
                   (head2("Frozen", "mean"), 1), (head2("Hetero-", "MDN"), 1),
                   (head2("Difference", r"[95\% CI]"), 1), (r"$p_{\mathrm{Holm}}$", 1),
                   (head2("Difference", r"[95\% CI]"), 1), (r"$p_{\mathrm{Holm}}$", 1),
                   (head2("Predicted", "variance"), 1), (head2("Squared", "residual"), 1)])]
    sig = (r"All ratios differ from 1 (Holm-adjusted $p < 0.05$)." if star_all else
           r"* Holm-adjusted $p < 0.05$ for a ratio of 1.")
    notes = [r"Constant sigma: the LSTM with constant sigma; frozen mean: its mean and features frozen (same "
             r"seed), a variance head trained on them; the squared residual is therefore the same for both. "
             r"Differences: paired, negative = the first model better; 95\% event-bootstrap intervals; Holm over all "
             r"data sets, regimes and both pairs of the root (the smallest attainable adjusted $p$ is set by the "
             r"bootstrap replicates). Ratios: mean predicted variance and mean squared residual of the frozen-mean "
             r"model per regime relative to steady following. " + sig]
    caption = (r"H3.3 with the frozen mean (recorded signals, target $a$, highD test set): NLL per "
               r"driving regime of the LSTM with constant sigma, of the same LSTM with its mean frozen and a learned "
               r"state-dependent sigma(x), and of HeteroMDN; the paired differences against the constant-sigma model "
               r"with their 95\% intervals and Holm-adjusted $p$; and the variance ratios of the frozen-mean model.")
    return Table("c16_frozen", caption, [src_n, src_v], "lCCCCcCcCCC", heads, body, notes, wide=True,
                 font="footnotesize", tabcolsep=3)


# --------------------------------------------------------------------------- c17: causal control (R6)
C17_MODELS = ("lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout", "gp_residual_idm")
C17_SETS = ("highd", "ngsim", "ngsim_us101", "waymo", "openacc")
C17_VARIANTS = (("e1h_mean1s", "Harmonised"), ("e1m_mean1s", "Matched, centred"), ("e1m_causal", "Matched, causal"),
                # V7-R02 (design.md 18.9): the cells on the events of the causal control (shifted sets without their
                # validation fold): centred inputs with the processed target (the e1h_mean1s checkpoints) and centred
                # inputs with the recorded target (retrained); skipped while their roots do not exist
                ("e1m_mean1s_abv", "Centred, processed target, common windows"), ("e1m_rt", "Centred, recorded target, common windows"))
# the causal cell restricted to the windows the three cells share (scripts/common_support_cells.py; third review):
# the causal processing keeps 1-7 % more events than the centred one, so its own-sample row above is not paired
C17_COMMON = ("e1m_causal", "Causal, recorded target, common windows")
C17_PERSISTENCE = (("e1_mean1s", "recorded"), ("e1h_mean1s", "harmonised"), ("e1m_mean1s", "matched, centred"),
                   ("e1m_causal", "causal inputs, recorded target"))


def matching(path: Path) -> dict[str, dict[str, float]]:
    """Window and achieved roughness ratio per data set of a matching file (scripts/match_roughness.py)."""
    import json

    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: {"window": int(v["window"]), "ratio": float(v["ratio"]), "flags": list(v.get("flags") or [])}
            for k, v in d["datasets"].items()}


def c17_causal() -> Table:
    srcs = [art_name(r, "one_step_metrics.csv") for r, _ in C17_VARIANTS] + \
           [art_name(r, "extra_signal.csv") for r, _ in C17_PERSISTENCE] + \
           ["artifacts/e1m/matching.json", "artifacts/e1m_causal/matching.json"]
    common_path = ART / C17_COMMON[0] / "tables" / "common_support.csv"
    common = pd.read_csv(common_path) if common_path.exists() else None
    if common is not None:
        srcs.append(art_name(C17_COMMON[0], "common_support.csv"))
    body: list = []
    for m in C17_MODELS:
        body.append(Group(label(m)))
        for root, name in C17_VARIANTS:
            cells = [name]
            for d in C17_SETS:
                x = onestep(root, m, d)
                cells += [num(x["nll"]), prob(x["picp90"])]
            body.append(cells)
        if common is not None:
            cells = [C17_COMMON[1]]
            for d in C17_SETS:
                r = pick(common, dataset=d, model=m, root=C17_COMMON[0])
                cells += [num(r.nll), prob(r.picp90)]
            body.append(cells)
    centred, causal = matching(ART / "e1m" / "matching.json"), matching(ART / "e1m_causal" / "matching.json")

    def span_row(name: str, values: list[str]) -> str:
        """A row whose entries span the two columns of each data set (one LaTeX line)."""
        return " & ".join([name] + [rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in values]) + r" \\"

    body.append(Group(r"Signal processing per data set: Savitzky--Golay window (samples) and achieved roughness ratio "
                      r"to highD" + fn(1)))
    for name, mm in (("Centred (E5)", centred), ("Causal", causal)):
        body.append(span_row(name, [f"{mm[d]['window']} ({num(mm[d]['ratio'])})" for d in C17_SETS]))
    body.append(Group(r"Persistence RMSE of the target $\bar a_{1\,\mathrm{s}}$ (m/s$^2$)" + fn(2)))
    for root, name in C17_PERSISTENCE:
        sig = art(root, "extra_signal.csv")
        cells = [name.capitalize()]
        for d in C17_SETS:
            r = pick(sig, dataset=d)
            if str(r.target) != "mean_1s":
                raise SystemExit(f"{art_name(root, 'extra_signal.csv')}: target {r.target}")
            cells.append(num(r.rmse_persistence))
        body.append(span_row(cells[0], cells[1:]))
    heads = [Head([("", 1)] + [(DATASET_LABEL[d] + (" test" if d == "highd" else ""), 2) for d in C17_SETS],
                  [(2 + 2 * i, 3 + 2 * i) for i in range(len(C17_SETS))]),
             Head([("Signals", 1)] + [("NLL", 1), ("PICP", 1)] * len(C17_SETS))]
    notes = [fn(1) + r" Windows of the roughness rule of E5 (the follower speed within 10\% of highD's at window 11); "
                     r"the largest window of the grid (101) did not reach it except for OpenACC with the centred filter "
                     r"(61). Centred: chosen on all events; causal: on the validation folds only, and the shifted data "
                     r"sets are then evaluated on their other folds.",
             fn(2) + r" Persistence: the same target one target length earlier, on the evaluated windows of each "
                     r"variant (recorded: the stored speed, E1 with recorded signals)."]
    caption = (r"Causal inputs and a fixed physical target, target $\bar a_{1\,\mathrm{s}}$ (1~s "
               r"mean acceleration): one-step NLL (nats) and PICP of the 90\% interval of the five history-based models "
               r"for the harmonised signals and the centred matched signals of E5 (models trained on harmonised highD, "
               r"target from the processed speed) and for the causal matched signals (one-sided filter, models "
               r"retrained on causal highD, target from the recorded speed); mean over five training seeds; with the "
               r"windows, the achieved roughness ratios and the persistence RMSE of the target. The rows ``common windows'' hold "
               r"the cells evaluated on the windows the three cells share (the shifted sets without their validation fold, "
               r"after the processing of each cell; the two centred cells share every window, the causal processing keeps "
               r"0.4--7\% more events): the harmonised checkpoints with the processed target and the models retrained on highD "
               r"with the recorded target, both on the centred matched signals, and the causal cell restricted to the same "
               r"windows; the row ``Matched, causal'' is the causal cell on its own evaluated sample.")
    return Table("c17_causal", caption, srcs, "l" + "C" * (2 * len(C17_SETS)), heads, body, notes, wide=True,
                 font="footnotesize", tabcolsep=3)


# ----------------------------------------------------------------------- c18: density sensitivity (R14)
def c18_density() -> Table:
    src = art_name("e1", "density_sensitivity.csv")
    df = art("e1", "density_sensitivity.csv")
    df = df[df["model"] == "diffusion_cf"]
    seeds = sorted(set(df["seed"].astype(int)))
    rows_per = sorted(set(df["n_rows"].astype(int)))
    samples, bands = (16, 32, 64), (0.5, 1.0, 2.0)
    sets = ("highd", "ngsim")
    body: list = []
    for s in samples:
        cells = [str(s)]
        for d in sets:
            sub = df[(df["dataset"] == d) & (df["n_samples"] == s)]
            cells += [num(pick(sub, bandwidth_factor=b).nll) for b in bands]
            one = pick(sub, bandwidth_factor=1.0)
            for b in bands:  # CRPS and PICP do not depend on the bandwidth: checked
                x = pick(sub, bandwidth_factor=b)
                if (x.crps, x.crps_fair, x.picp90) != (one.crps, one.crps_fair, one.picp90):
                    raise SystemExit(f"{src}: CRPS or PICP depend on the bandwidth ({d}, S = {s})")
            cells += [num(one.crps, sig=3, max_dec=None), num(one.crps_fair, sig=3, max_dec=None), prob(one.picp90)]
        body.append(cells)
    stored = [r"32, E1 table" + fn(1)]
    for d in sets:
        x = pick(df[(df["dataset"] == d)], n_samples=32, bandwidth_factor=1.0)
        stored += [DASH, num(x.stored_nll), DASH, num(x.stored_crps, sig=3, max_dec=None), DASH, prob(x.stored_picp90)]
    body += [RULE, stored]
    med = {d: (df[df["dataset"] == d]["nll_median"].min(), df[df["dataset"] == d]["nll_median"].max()) for d in sets}
    heads = [Head([("", 1), ("highD test", 6), ("NGSIM I-80", 6)], [(2, 7), (8, 13)]),
             Head([("", 1)] + [("NLL (nats), bandwidth", 3), ("", 1), ("", 1), ("", 1)] * 2,
                  [(2, 4), (8, 10)]),
             Head([("Samples $S$", 1)] + [(r"0.5$\times$", 1), (r"1$\times$", 1), (r"2$\times$", 1), ("CRPS", 1),
                                         (head2("CRPS", "(fair)"), 1), ("PICP", 1)] * 2)]
    notes = [fn(1) + r" The stored values of the E1 tables (an independent 32-sample draw, bandwidth 1$\times$)."
             + rf" Training seed {', '.join(map(str, seeds))}, {thousands(rows_per[0])} windows per data set (the rows "
             r"E1 evaluates); $S$ = 16 and 32 are nested subsets of one 64-sample draw; bandwidth in multiples of "
             r"Silverman's rule; CRPS (m/s$^2$): plain and fair sample CRPS; PICP of the 90\% interval of the sample "
             r"quantiles. Median NLL over the windows: highD " + f"{num(med['highd'][0])} to {num(med['highd'][1])}"
             + r", NGSIM I-80 " + f"{num(med['ngsim'][0])} to {num(med['ngsim'][1])}" + r" nats."]
    caption = (r"Density sensitivity of the sampled law of Diffusion (recorded signals, target "
               r"$a$): mean NLL of the kernel density estimate for $S$ = 16, 32 and 64 first-step samples and "
               r"bandwidths of 0.5, 1 and 2 times Silverman's rule, sample CRPS and PICP of the 90\% interval, on "
               r"the highD test set and NGSIM I-80.")
    return Table("c18_density", caption, [src], "l" + "C" * 12, heads, body, notes, wide=True, font="footnotesize",
                 tabcolsep=3)


# ------------------------------------------------------------------------------------------ figures
_RE_FIG = re.compile(r"^\* `figures/(?P<file>[^`/]+\.png)`: (?P<cap>.*?) Source: (?P<src>.*)$")
_RE_DIR = re.compile(r"^\* `figures/(?P<dir>[^`]+)/` \((?P<n>\d+) figures\): (?P<cap>.*?) Source: (?P<src>.*)$")
_RE_PANEL = re.compile(r"^  \* `(?P<file>[^`]+\.png)`: (?P<cap>.*)$")


def parse_index() -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str, dict[str, str]]]]:
    """Figures of index.md: file -> (caption, sources); directory -> (caption, sources, panel file -> caption)."""
    files: dict[str, tuple[str, str]] = {}
    dirs: dict[str, tuple[str, str, dict[str, str]]] = {}
    current = None
    for line in (REPORT / "index.md").read_text(encoding="utf-8").splitlines():
        if m := _RE_FIG.match(line):
            files[m["file"]] = (m["cap"], m["src"].replace("`", "").rstrip(". "))
            current = None
        elif m := _RE_DIR.match(line):
            dirs[m["dir"]] = (m["cap"], m["src"].replace("`", "").rstrip(". "), {})
            current = m["dir"]
        elif (m := _RE_PANEL.match(line)) and current is not None:
            dirs[current][2][m["file"]] = m["cap"]
        elif not line.startswith("  "):
            current = None
    return files, dirs


def png_cm(path: Path) -> float:
    """Design width of a report figure (cm at 170 dpi)."""
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big") / FIG_PX_PER_CM


@dataclass
class FigureEnv:
    label: str
    files: list[str]  # paths below figures/
    caption: str  # LaTeX
    tex: str


def figure_envs() -> list[FigureEnv]:
    files, dirs = parse_index()
    out = []
    for name, panels in FIGURES:
        if panels is None:
            if name not in files:
                raise SystemExit(f"{name}: not in artifacts/report/index.md")
            cap, src = files[name]
            stem = name.removesuffix(".png")
            caption = caption_tex(cap)
            design = png_cm(FIG_DIR / name)
            body = [f"% figures/{name} (source: {src}; designed {design:.1f} cm wide)", r"\begin{figure}[H]"]
            if design > WIDE_CM:
                body += [r"\begin{adjustwidth}{-\extralength}{0cm}", r"\centering",
                         rf"\includegraphics[width=\linewidth]{{figures/{name}}}", r"\end{adjustwidth}"]
            elif design > TEXT_CM:
                body += [r"\centering", rf"\includegraphics[width=\linewidth]{{figures/{name}}}"]
            else:
                body += [r"\centering", rf"\includegraphics[width={design:.1f}cm]{{figures/{name}}}"]
            body += [rf"\caption{{{caption}\label{{fig:{stem}}}}}", r"\end{figure}"]
            out.append(FigureEnv(f"fig:{stem}", [name], caption, "\n".join(body) + "\n"))
            continue
        if name not in dirs:
            raise SystemExit(f"{name}/: not in artifacts/report/index.md")
        _, src, members = dirs[name]
        caps = [members[p] for p in panels]
        names = [c.split(": ", 1)[0] for c in caps]
        common = {c.split(": ", 1)[1] for c in caps}
        if len(common) != 1:
            raise SystemExit(f"{name}: the panels have different captions after the model name: {common}")
        lead = "Uncertainty maps: " + common.pop()
        letters = "abcdefghij"
        panel_text = "; ".join(rf"(\textbf{{{letters[i]}}}) {caption_tex(n)}" for i, n in enumerate(names))
        caption = f"{caption_tex(lead)} {panel_text}."
        paths = [f"{name}/{p}" for p in panels]
        body = [f"% figures/{name}/: {', '.join(panels)} (source: {src})", r"\begin{figure}[H]",
                r"\begin{adjustwidth}{-\extralength}{0cm}", r"\centering"]
        for i, p in enumerate(paths):
            body.append(rf"\subfloat[\centering]{{\includegraphics[width=0.49\linewidth]{{figures/{p}}}}}")
            if i % 2 == 0 and i != len(paths) - 1:
                body.append(r"\hfill")
            elif i != len(paths) - 1:
                body.append(r"\\[4pt]")
        body += [r"\end{adjustwidth}", rf"\caption{{{caption}\label{{fig:{name}}}}}", r"\end{figure}"]
        out.append(FigureEnv(f"fig:{name}", paths, caption, "\n".join(body) + "\n"))
    return out


def copy_figures(envs: list[FigureEnv]) -> None:
    dst = SUB / "figures"
    dst.mkdir(exist_ok=True)
    used = set()
    for env in envs:
        for p in env.files:
            (dst / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(FIG_DIR / p, dst / p)
            used.add(p)
    on_disk = sorted(str(p.relative_to(dst)).replace("\\", "/") for p in dst.rglob("*.png"))
    extra = [p for p in on_disk if p not in used]
    if extra:
        print(f"note: figures/ holds PNG files that no figure environment uses: {extra}")


# ------------------------------------------------------------------------------------------ main
def md(text: str) -> str:
    return text.replace("|", "\\|")


def main() -> None:
    global SUB
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-copy", action="store_true", help="do not copy the figures to <out>/figures/")
    ap.add_argument("--out", default=str(SUB), help="paper directory to write into (default: %(default)s)")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    for frozen in FROZEN:
        if out == frozen.resolve() or frozen.resolve() in out.parents:
            raise SystemExit(f"{out}: {frozen.name} is an earlier, frozen version; write into paper/v4 (or a later one)")
    SUB = out

    c2, c2_log = c2_shift()
    c13, c13_log = c13_matrix()
    c15, c15_log = c15_ar()
    tables = [c1_onestep(), c2, c3_regimes(), c4_coverage(), c5_platoon(), c6_platoon_diag(), c7_e5(), c8_e6(),
              c10_h32_cluster(), c11_collisions(), c12_info_budget(), c13, c14_information(), c15, c16_frozen(),
              c17_causal(), c18_density()]
    (SUB / "tables").mkdir(exist_ok=True)
    gen = SUB / "generated"
    gen.mkdir(exist_ok=True)
    for t in tables:
        (SUB / "tables" / f"{t.name}.tex").write_text(render(t), encoding="utf-8", newline="\n")

    envs = figure_envs()
    if not args.no_copy:
        copy_figures(envs)
    for env in envs:
        for p in env.files:
            if not (SUB / "figures" / p).exists():
                raise SystemExit(f"figures/{p} missing (run without --no-copy)")

    idx = ["% Compact tables of the submission: one complete MDPI float ([H], caption, label) per file in tables/.",
           "% This file inputs nothing. Place each table at paragraph level (never inside a table environment) in",
           "% the section where it belongs, with the line given here:",
           "% Written by paper/v4/tools/make_compact.py; do not edit by hand.", "%"]
    for t in tables:
        idx.append(f"% \\input{{tables/{t.name}}}   % tab:{t.name}: {detex(t.caption)}")
    (gen / "tables.tex").write_text("\n".join(idx) + "\n", encoding="utf-8", newline="\n")

    figs = ["% Figures of the submission: one MDPI figure environment per figure, label fig:<file stem> (the two",
            "% map panels of figures/f1_maps__e1/ as one figure with panels (a) and (b), label fig:f1_maps__e1).",
            "% Copy each environment into the section where the figure belongs; main.tex does not input this file.",
            "% Written by paper/v4/tools/make_compact.py; do not edit by hand.", ""]
    figs += [env.tex for env in envs]
    (gen / "figures.tex").write_text("\n".join(figs), encoding="utf-8", newline="\n")

    labels = ["# Labels of the compact tables and the figures (version v4)", "",
              "Written by `paper/v4/tools/make_compact.py` from the CSVs of `artifacts/report/tables` (c13-c18: the",
              "tables of the experiment roots, see the sources column) and the captions of `artifacts/report/index.md`;",
              "do not edit by hand. `tables/c9_e3pooled.tex` (`tab:c9_e3pooled`) is written by hand and not listed;",
              "c10-c12 are the tables of the review fixes (docs/review_fixes_v3.md), c13-c18 those of the control",
              "experiments (docs/design.md 18, docs/review_fixes_v4.md).", "",
              "* Tables: `\\input{tables/<name>}` at paragraph level, never inside a `table` environment (each file",
              "  holds the complete `[H]` float); the lines are collected, commented out, in `generated/tables.tex`.",
              "* Figures: copy the environment from `generated/figures.tex` into the section.", "",
              "## Tables", "", "| label | file | layout | caption | sources |", "|---|---|---|---|---|"]
    for t in tables:
        layout = f"{'full width' if t.wide else 'text column'}, \\{t.font}"
        labels.append(f"| `tab:{t.name}` | `tables/{t.name}.tex` | {layout} | {md(detex(t.caption))} | "
                      + ", ".join(f"`{source_path(s)}`" for s in t.sources) + " |")
    labels += ["", "## Figures", "", "| label | file(s) | caption |", "|---|---|---|"]
    for env in envs:
        labels.append(f"| `{env.label}` | " + ", ".join(f"`figures/{p}`" for p in env.files)
                      + f" | {md(detex(env.caption))} |")
    (gen / "LABELS.md").write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")

    print(f"{len(tables)} tables in {SUB / 'tables'}; {len(envs)} figure environments "
          f"({sum(len(e.files) for e in envs)} PNG) in {gen / 'figures.tex'}")
    for line in c2_log:
        print(f"  tab:c2_shift  {line}")
    for line in c13_log:
        print(f"  tab:c13_matrix  Diffusion, ACC, effect {line}")
    for line in c15_log:
        print(f"  tab:c15_ar  AR(1) fit {line}")


if __name__ == "__main__":
    main()
