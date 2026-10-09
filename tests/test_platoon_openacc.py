"""OpenACC parser (cf_unc.platoon.openacc) on a hand-written file with the quirks of the raw campaigns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cf_unc.constants import DT
from cf_unc.platoon.openacc import ACC, HUMAN, UNKNOWN, data_files, load_runs, read_run, resample_linear

# vehicle 1 LEAD (no Driver column: header flag ACC = 1), vehicle 2 CAR_A (Driver column with an empty cell),
# vehicle 3 CAR_B listed but without any column, vehicle 4 CAR_C (speed and gap columns, no Driver column);
# trailing commas everywhere, empty cells, a time gap of 1.0 s after t = 100.9, columns not in vehicle order.
HEADER = """Date,8,10,2019,,,,,,
Vehicle_order,LEAD,CAR_A,CAR_B,CAR_C,,,,,
Number_of_vehicles,4,,,,,,,,
ACC,1,,,,,,,,
Distance_setting,S,,,,,,,,
Time,Speed1,E1,Speed2,Driver2,IVS1,Speed4,IVS3,Lat4,
"""


def _rows() -> list[str]:
    lines = []
    times = [100.0 + 0.1 * k for k in range(10)] + [101.9 + 0.1 * k for k in range(10)]  # gap 100.9 -> 101.9
    for k, t in enumerate(times):
        speed1 = 10.0 + 0.1 * k
        speed2 = "" if k in (3, 12, 13, 14, 15) else f"{9.0 + 0.1 * k:.3f}"  # one isolated and a 0.4 s run of empties
        driver2 = "" if k == 5 else ("Human" if k < 15 else "ACC")
        ivs1 = f"{20.0 + 0.05 * k:.3f}"
        speed4 = f"{8.0 + 0.1 * k:.3f}"
        ivs3 = "" if k == 0 else f"{30.0:.3f}"
        lines.append(f"{t:.1f},{speed1:.3f},{1000 + k},{speed2},{driver2},{ivs1},{speed4},{ivs3},0.8,")
    return lines


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "raw" / "TestCampaign"
    folder.mkdir(parents=True)
    (folder / "run_a.csv").write_text(HEADER + "\n".join(_rows()) + "\n", encoding="utf-8")
    (folder / "TestCampaign_veh_specifications.csv").write_text("Vehicles,Max power (kW),\nA,100,\n", encoding="utf-8")
    return tmp_path / "raw"


def test_data_files_skip_tables_without_header(raw_dir: Path) -> None:
    files = data_files(raw_dir / "TestCampaign")
    assert [f.name for f in files] == ["run_a.csv"]
    with pytest.raises(FileNotFoundError):
        data_files(raw_dir / "missing")


def test_header_and_columns(raw_dir: Path) -> None:
    run = read_run(raw_dir / "TestCampaign" / "run_a.csv")
    assert run.campaign == "TestCampaign" and run.name == "TestCampaign/run_a" and run.date == "2019-10-08"
    assert run.vehicle_order == ["LEAD", "CAR_A", "CAR_B", "CAR_C"] and run.acc_flag == 1 and run.distance_setting == "S"
    assert run.n_vehicles == 4
    assert run.has_speed.tolist() == [True, True, False, True]
    assert run.has_ivs.tolist() == [True, False, True, False]
    assert run.has_driver.tolist() == [False, True, False, False]
    assert run.chain_length() == 2  # vehicle 3 has no columns: the recorded platoon behind the leader is 1 -> 2
    assert np.isnan(run.speed[2]).all() and np.isnan(run.ivs[1]).all()


def test_uniform_grid_time_gap_and_empty_cells(raw_dir: Path) -> None:
    run = read_run(raw_dir / "TestCampaign" / "run_a.csv")
    np.testing.assert_allclose(np.diff(run.t), DT, atol=1e-9)
    assert run.t[0] == pytest.approx(100.0) and run.t[-1] == pytest.approx(102.8) and len(run) == 29
    gap = (run.t > 100.9 + 1e-6) & (run.t < 101.9 - 1e-6)  # nine grid points strictly inside the 1.0 s gap
    assert gap.sum() == 9
    for row in (run.speed[0], run.speed[1], run.ivs[0], run.speed[3]):
        assert np.isnan(row[gap]).all()
    np.testing.assert_allclose(run.speed[0][~gap], 10.0 + 0.1 * np.arange(20), atol=1e-9)  # observed values kept
    # the isolated empty cell (t = 100.3) lies between samples 0.2 s apart: interpolated linearly
    k3 = int(np.argmin(np.abs(run.t - 100.3)))
    assert run.speed[1][k3] == pytest.approx(9.3, abs=1e-9)
    # four empty cells in a row (t = 102.1 .. 102.4): the finite neighbours are 0.5 s apart, more than 0.35 s
    inner = (run.t > 102.0 + 1e-6) & (run.t < 102.5 - 1e-6)
    assert inner.sum() == 4 and np.isnan(run.speed[1][inner]).all()
    assert np.isnan(run.ivs[2][0]) and np.isfinite(run.ivs[2][1])  # leading empty cell: outside the observed range


def test_driver_modes(raw_dir: Path) -> None:
    run = read_run(raw_dir / "TestCampaign" / "run_a.csv")
    t = run.t
    gap = (t > 100.9 + 1e-6) & (t < 101.9 - 1e-6)
    assert (run.mode[0] == ACC).all() and (run.mode[3] == ACC).all()  # no Driver column: header flag 1 = ACC
    m2 = run.mode[1]
    assert m2[int(np.argmin(np.abs(t - 100.0)))] == HUMAN
    assert m2[int(np.argmin(np.abs(t - 100.5)))] == UNKNOWN  # empty Driver cell
    far = (t > 101.0 + 1e-6) & (t < 101.8 - 1e-6)  # no source sample within max_gap / 2 = 0.175 s
    assert far.sum() == 7 and (m2[far] == UNKNOWN).all()
    assert m2[gap & ~far].tolist() == [HUMAN, HUMAN]  # 0.1 s from the samples at 100.9 and 101.9: nearest sample
    assert m2[int(np.argmin(np.abs(t - 102.8)))] == ACC
    assert run.mode.dtype == np.int8


def test_header_flag_variants(tmp_path: Path) -> None:
    folder = tmp_path / "C"
    folder.mkdir()
    rows = "\n".join(f"{k * 0.1:.1f},5.0,10.0,4.0" for k in range(5))
    for name, flag, expected in (("human.csv", "0", HUMAN), ("mixed.csv", "2", UNKNOWN), ("none.csv", "", UNKNOWN)):
        head = f"Date,1,2,2019\nVehicle_order,A,B\nNumber_of_vehicles,2\nACC,{flag}\nDistance_setting,none\nTime,Speed1,IVS1,Speed2\n"
        (folder / name).write_text(head + rows + "\n", encoding="utf-8")
        run = read_run(folder / name)
        assert (run.mode == expected).all(), name
    runs = load_runs(tmp_path, ["C"])
    assert [r.path.name for r in runs] == ["human.csv", "mixed.csv", "none.csv"] and all(r.chain_length() == 2 for r in runs)


def test_resample_linear_ignores_duplicates_and_order() -> None:
    t = np.array([0.2, 0.0, 0.1, 0.1, 0.3])
    y = np.array([2.0, 0.0, 1.0, 99.0, 3.0])  # the repeated time 0.1 keeps its first sample (in time order: 1.0)
    grid = np.arange(4) * DT
    np.testing.assert_allclose(resample_linear(t, y, grid, 0.35), [0.0, 1.0, 2.0, 3.0])
