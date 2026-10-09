"""Raw OpenACC platoon runs on a uniform 10 Hz grid (docs/design.md, section 13.1).

Input (JRC OpenACC campaigns, read-only): one csv file per platoon run. Metadata lines before the table
(any order, trailing commas allowed, each optional)::

    Date,DD,MM,YYYY
    Vehicle_order,<name of vehicle 1>,<name of vehicle 2>,...      platoon order, vehicle 1 drives in front
    Number_of_vehicles,<n>
    ACC,<0 | 1 | 2>                                                 0 manual, 1 ACC, 2 mixed
    Distance_setting,<text>

then a table whose header line starts with ``Time`` [s] (arbitrary origin) and holds, per vehicle i (1-based platoon
order), ``Speed<i>`` [m/s], positions (``Lat/Lon/Alt/E/N/U/VE/VN/VU<i>``, ignored), ``Driver<i>`` ("ACC" / "Human",
optional) and ``IVS<i>`` [m], the bumper-to-bumper gap between vehicle i and vehicle i + 1 (the gap of follower i + 1).
Column sets differ between files and campaigns; a vehicle listed in the header may have no columns at all; cells can
be empty; the time can have gaps. Columns are identified by name, never by position.

Output: :class:`PlatoonRun`, every quantity on the grid ``t[0] + k DT`` (``DT`` = 0.1 s) from the first to the last
source time. Numeric columns are interpolated linearly between finite source samples that are at most ``max_gap_s``
apart (NaN inside longer gaps and outside the observed range); the driver mode is the nearest source sample (unknown
when it is farther than ``max_gap_s / 2``). The driver mode of a sample comes from ``Driver<i>`` where that column
exists (an empty or unrecognised cell is unknown), otherwise from the header flag (0 Human, 1 ACC, 2 / missing
unknown). No signal processing beyond the resampling happens here (harmonisation: :mod:`cf_unc.platoon.profiles`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cf_unc.constants import DT

__all__ = [
    "CAMPAIGNS",
    "HUMAN",
    "ACC",
    "UNKNOWN",
    "PlatoonRun",
    "data_files",
    "load_runs",
    "parse_header",
    "read_run",
    "resample_linear",
    "resample_nearest",
]

CAMPAIGNS = ("AstaZero", "ZalaZone", "Vicolungo")  # campaign folders of design.md 13.1
HUMAN, ACC, UNKNOWN = 0, 1, -1  # driver mode codes per sample
MODE_NAMES = {HUMAN: "Human", ACC: "ACC", UNKNOWN: "unknown"}
_HEADER_MODE = {0: HUMAN, 1: ACC}  # header flag -> per-sample mode where Driver<i> is absent (2 = mixed -> unknown)
_COLUMN = re.compile(r"^(Speed|IVS|Driver)(\d+)$")
_MAX_HEADER_LINES = 20
_TIME_TOL = 1e-6  # s, a grid point this close to a source sample lies on it


@dataclass
class PlatoonRun:
    """One OpenACC file on the uniform grid. Vehicle ``i`` (1-based platoon order) is row ``i - 1``.

    t            (T,) grid times [s], source origin, step DT
    speed        (N, T) speed [m/s], NaN where not observed
    ivs          (N, T) gap [m] between vehicle i and vehicle i + 1 (row i - 1 = ``IVS<i>``), NaN where not observed
    mode         (N, T) int8 driver mode per sample: 1 ACC, 0 Human, -1 unknown
    has_speed, has_ivs, has_driver    (N,) bool: the column exists in the file
    """

    path: Path
    campaign: str
    date: str | None
    vehicle_order: list[str]
    acc_flag: int | None
    distance_setting: str | None
    t: np.ndarray
    speed: np.ndarray
    ivs: np.ndarray
    mode: np.ndarray
    has_speed: np.ndarray
    has_ivs: np.ndarray
    has_driver: np.ndarray

    @property
    def name(self) -> str:
        """``<campaign>/<file stem>``, the identifier of the run."""
        return f"{self.campaign}/{self.path.stem}"

    @property
    def n_vehicles(self) -> int:
        return int(self.speed.shape[0])

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def chain_length(self) -> int:
        """Number k of vehicles 1..k whose speed columns exist and whose gaps to their predecessor (``IVS<i-1>``)
        exist: the recorded platoon behind vehicle 1 (0 when vehicle 1 has no speed column)."""
        k = 0
        while k < self.n_vehicles and self.has_speed[k] and (k == 0 or self.has_ivs[k - 1]):
            k += 1
        return k


# ------------------------------------------------------------------------------------------- header
def parse_header(path: str | Path) -> tuple[dict[str, list[str]], list[str], int]:
    """Metadata (key -> non-empty cells), the table column names and the line index of the table header."""
    header: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as fh:
        for number, line in enumerate(fh):
            cells = line.rstrip("\r\n").split(",")
            key = cells[0].strip()
            if key == "Time":
                return header, [c.strip() for c in cells], number
            header[key] = [cell.strip() for cell in cells[1:] if cell.strip()]
            if number >= _MAX_HEADER_LINES:
                break
    raise ValueError(f"{path}: no table header line starting with 'Time'")


def _int_or_none(cells: list[str] | None) -> int | None:
    try:
        return int(float(cells[0])) if cells else None
    except ValueError:
        return None


def _iso_date(cells: list[str] | None) -> str | None:
    """``Date,DD,MM,YYYY`` -> ``YYYY-MM-DD``."""
    try:
        day, month, year = (int(float(c)) for c in (cells or [])[:3])
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


# ------------------------------------------------------------------------------------------ resampling
def _sorted_unique(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source rows in time order with finite, strictly increasing times (a repeated time keeps its first row)."""
    order = np.argsort(t, kind="stable")
    ts = t[order]
    keep = np.isfinite(ts)
    keep[1:] &= ts[1:] != ts[:-1]
    return order[keep], ts[keep]


def resample_linear(t: np.ndarray, y: np.ndarray, grid: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Linear interpolation of ``y`` (n,) at source times ``t`` (n,) onto ``grid`` (T,) -> (T,).

    NaN source values are missing; grid points outside the observed range or strictly inside a gap of more than
    ``max_gap_s`` seconds between two finite samples are NaN.
    """
    rows, ts = _sorted_unique(np.asarray(t, dtype=np.float64))
    ys = np.asarray(y, dtype=np.float64)[rows]
    ok = np.isfinite(ys)
    ts, ys = ts[ok], ys[ok]
    out = np.full(grid.shape, np.nan)
    if ts.size == 0:
        return out
    i = np.searchsorted(ts, grid + _TIME_TOL, side="right") - 1  # last finite sample at or before each grid point
    left = ts[np.maximum(i, 0)]
    right = ts[np.minimum(i + 1, ts.size - 1)]
    in_gap = (right - left > max_gap_s) & (grid - left > _TIME_TOL)
    use = (i >= 0) & (grid <= ts[-1] + _TIME_TOL) & ~in_gap
    out[use] = np.interp(grid[use], ts, ys)
    return out


def resample_nearest(t: np.ndarray, codes: np.ndarray, grid: np.ndarray, max_dist_s: float, fill: int) -> np.ndarray:
    """Integer codes (n,) at source times ``t`` onto ``grid`` (T,): the nearest source sample, ``fill`` where the
    nearest one is farther than ``max_dist_s``."""
    rows, ts = _sorted_unique(np.asarray(t, dtype=np.float64))
    cs = np.asarray(codes)[rows]
    out = np.full(grid.shape, fill, dtype=np.int8)
    if ts.size == 0:
        return out
    right = np.clip(np.searchsorted(ts, grid), 0, ts.size - 1)
    left = np.clip(right - 1, 0, ts.size - 1)
    pick = np.where(np.abs(grid - ts[left]) <= np.abs(ts[right] - grid), left, right)
    near = np.abs(ts[pick] - grid) <= max_dist_s + _TIME_TOL
    out[near] = cs[pick[near]]
    return out


def _driver_codes(values: pd.Series) -> np.ndarray:
    text = values.fillna("").astype(str).str.strip().str.lower()
    return np.where(text == "acc", ACC, np.where(text == "human", HUMAN, UNKNOWN)).astype(np.int8)


# ------------------------------------------------------------------------------------------------ run
def read_run(path: str | Path, campaign: str | None = None, dt: float = DT, max_gap_s: float = 0.35) -> PlatoonRun:
    """Parse one OpenACC csv file onto the uniform grid (module docstring); ``campaign`` defaults to the folder."""
    path = Path(path)
    header, names, line = parse_header(path)
    kinds: dict[str, tuple[str, int]] = {}
    for name in names:
        match = _COLUMN.match(name)
        if match:
            kinds[name] = (match.group(1), int(match.group(2)))
    order = [re.sub(r"\s+", "", n) for n in header.get("Vehicle_order", [])]
    n_listed = max(len(order), _int_or_none(header.get("Number_of_vehicles")) or 0)
    n_veh = max([n_listed, *(i for _, i in kinds.values())]) if (kinds or n_listed) else 0
    frame = pd.read_csv(
        path,
        skiprows=line,
        header=0,
        index_col=False,  # tolerates trailing commas in the data rows
        usecols=lambda c: c.strip() == "Time" or c.strip() in kinds,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        skipinitialspace=True,
    )
    frame.columns = [c.strip() for c in frame.columns]
    t_src = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(np.float64)
    finite_t = t_src[np.isfinite(t_src)]
    if finite_t.size == 0:
        grid = np.empty(0)
    else:
        start, stop = float(finite_t.min()), float(finite_t.max())
        grid = start + np.arange(int(np.floor((stop - start) / dt + 1e-9)) + 1) * dt
    speed = np.full((n_veh, grid.size), np.nan)
    ivs = np.full((n_veh, grid.size), np.nan)
    flag = _int_or_none(header.get("ACC"))
    mode = np.full((n_veh, grid.size), _HEADER_MODE.get(flag, UNKNOWN), dtype=np.int8)
    has = {k: np.zeros(n_veh, dtype=bool) for k in ("Speed", "IVS", "Driver")}
    for name, (kind, index) in kinds.items():
        row = index - 1
        has[kind][row] = True
        if kind == "Driver":
            mode[row] = resample_nearest(t_src, _driver_codes(frame[name]), grid, max_gap_s / 2, UNKNOWN)
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
        (speed if kind == "Speed" else ivs)[row] = resample_linear(t_src, values, grid, max_gap_s)
    return PlatoonRun(
        path=path,
        campaign=campaign or path.parent.name,
        date=_iso_date(header.get("Date")),
        vehicle_order=order,
        acc_flag=flag,
        distance_setting=(header.get("Distance_setting") or [None])[0],
        t=grid,
        speed=speed,
        ivs=ivs,
        mode=mode,
        has_speed=has["Speed"],
        has_ivs=has["IVS"],
        has_driver=has["Driver"],
    )


def data_files(folder: str | Path) -> list[Path]:
    """The run files of a campaign folder, sorted by name: csv files whose first line starts with ``Date``
    (vehicle specifications and other tables are skipped)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"OpenACC campaign folder not found: {folder}")
    out = []
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8", errors="replace") as fh:
            if fh.readline().startswith("Date"):
                out.append(path)
    return out


def load_runs(raw_dir: str | Path, campaigns: Iterable[str] = CAMPAIGNS) -> list[PlatoonRun]:
    """Every run of the campaign folders under ``raw_dir``, campaign by campaign, files sorted by name."""
    return [read_run(p, campaign) for campaign in campaigns for p in data_files(Path(raw_dir) / campaign)]
