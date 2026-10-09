"""GPU audit (marker ``gpu``): every model configuration fits, predicts, samples and round-trips on CUDA.

Tiny synthetic data (``synth_small`` / ``synth_small_seq``) and budgets; float32 work in IEEE precision (as in
the E1 runner, ``configure_torch``), so that predictions of one checkpoint agree across CPU and GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import torch

from cf_unc.constants import MEAN_TARGET_STEPS
from cf_unc.data.pipeline import DataBundle
from cf_unc.dist import PredDist
from cf_unc.experiments.e1 import configure_torch, load_config, model_spec, seed_everything
from cf_unc.models import SAMPLER_MODES, build_model, load_model
from cf_unc.physics.idm import IDMParams

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")]

LABELS = [
    "idm_white_noise",
    "idm_white_noise_mle",
    "lstm_const_sigma",
    "hetero_mdn",
    "hetero_mdn_mix3",
    "deep_ensemble",
    "mc_dropout",
    "gp_residual_idm",
    "quantile_pidl",
    "diffusion_cf",
]
N_ROWS = 256  # predicted rows (DiffusionCF: 64)
N_TRAJ = 32


@pytest.fixture(scope="module", autouse=True)
def ieee_float32() -> Iterator[None]:
    """IEEE float32 on the GPU for this module; the previous flags are restored (legacy flags first)."""
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    legacy = (cudnn.allow_tf32, matmul.allow_tf32)
    precision = (cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision, matmul.fp32_precision)
    flags = (cudnn.deterministic, cudnn.benchmark)
    configure_torch(tf32=False)
    yield
    cudnn.allow_tf32, matmul.allow_tf32 = legacy
    cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision, matmul.fp32_precision = precision
    cudnn.deterministic, cudnn.benchmark = flags


def _spec(label: str, idm: IDMParams):
    """Kwargs of the model configuration as the E1 runner builds them, with a tiny training budget."""
    cfg = load_config()
    cfg.train.overrides = {"max_epochs": 2, "batch_size": 256}
    cfg.model_overrides = {"gp_residual_idm": {"max_epochs": 2, "batch_size": 512}}
    return model_spec(cfg, label, lambda name: (idm, {"name": name}))  # every named calibration: the true IDM


def _tensors(obj: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _tensors(value)


def _check_law(dist: PredDist, n: int, epistemic: bool) -> None:
    assert isinstance(dist, PredDist) and len(dist) == n
    for values in (dist.mean(), dist.var()):
        assert isinstance(values, np.ndarray) and values.dtype == np.float64 and values.shape == (n,)
        assert np.isfinite(values).all()
    assert (dist.var() > 0).all()
    for name in ("aleatoric_var", "epistemic_var"):
        value = getattr(dist, name)
        assert value is None or (isinstance(value, np.ndarray) and value.shape == (n,) and np.isfinite(value).all())
    assert (dist.epistemic_var is not None) or not epistemic


def _assert_close(a: PredDist, b: PredDist, y: np.ndarray, tol: float) -> None:
    """Same law up to ``tol`` (absolute and relative): moments, decomposition, quantiles, CRPS at ``y``.

    The log density is not compared: far in an exponential tail (QuantilePIDL after two epochs has knots
    1e-6 apart) it amplifies float32 rounding of the network outputs to relative differences of ~1e-3.
    """
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    pairs = [(a.mean(), b.mean()), (a.var(), b.var()), (a.quantile(levels), b.quantile(levels)), (a.crps(y), b.crps(y))]
    for name in ("aleatoric_var", "epistemic_var"):
        x, z = getattr(a, name), getattr(b, name)
        assert (x is None) == (z is None), name
        if x is not None:
            pairs.append((x, z))
    for x, z in pairs:
        np.testing.assert_allclose(x, z, rtol=tol, atol=tol)


@pytest.mark.parametrize("label", LABELS)
def test_model_on_cuda(label: str, synth_small: DataBundle, synth_small_seq: DataBundle, true_idm: IDMParams,
                       tmp_path: Path) -> None:  # fmt: skip
    spec = _spec(label, true_idm)
    data = synth_small_seq if spec.sequences else synth_small
    train, val, test = data.windows["train"], data.windows["val"], data.windows["test"]
    seed_everything(0)
    model = build_model(spec.name, **spec.kwargs).to("cuda")
    model.fit(train, val, seed=0)
    assert model.device.type == "cuda"

    n = 64 if spec.sequences else N_ROWS
    x, y = test.X[:n], test.y[:n].astype(np.float64)
    future = test.v_lead_future[:n] if model.needs_leader_future else None
    dist = model.predict_dist(x, v_lead_future=future)  # numpy in, numpy-based law out
    _check_law(dist, n, model.has_epistemic)

    hist = torch.as_tensor(x[:N_TRAJ], device="cuda")
    for mode in SAMPLER_MODES:
        gen = torch.Generator(device="cuda").manual_seed(0)
        state = model.init_sampler(N_TRAJ, gen, mode)
        draws = torch.stack([model.sample_step(hist, state, gen) for _ in range(3)])
        assert draws.device.type == "cuda" and draws.shape == (3, N_TRAJ) and torch.isfinite(draws).all()

    path = model.save(tmp_path / "model.pt")
    with open(path, "rb") as fh:  # no map_location: every stored tensor must come back on the CPU
        payload = torch.load(fh, weights_only=True)
    assert {t.device.type for t in _tensors(payload)} <= {"cpu"}
    on_gpu = load_model(path, device="cuda").predict_dist(x, v_lead_future=future)
    on_cpu = load_model(path, device="cpu").predict_dist(x, v_lead_future=future)
    _assert_close(on_gpu, dist, y, 1e-4)  # same device: the checkpoint reproduces the fitted model
    _assert_close(on_cpu, dist, y, 1e-4)
    _assert_close(on_cpu, on_gpu, y, 1e-3)


def test_diffusion_mean_1s_law_on_cuda(synth_small_seq: DataBundle, true_idm: IDMParams, tmp_path: Path) -> None:
    """Target mean_1s on CUDA: mean of the first 10 sampled steps, the same law as on the CPU (design.md 11.1)."""
    spec = _spec("diffusion_cf", true_idm)
    train, val, test = (synth_small_seq.windows[k] for k in ("train", "val", "test"))
    seed_everything(0)
    model = build_model(spec.name, **spec.kwargs).to("cuda")
    model.fit(train, val, seed=0)
    x, future = test.X[:64], test.v_lead_future[:64]
    dist = model.predict_dist(x, v_lead_future=future, target="mean_1s")
    seq = model.sample_sequences(x, future)
    np.testing.assert_array_equal(dist.samples, seq[:, :, :MEAN_TARGET_STEPS].astype(np.float64).mean(axis=2))
    _check_law(dist, 64, epistemic=False)
    path = model.save(tmp_path / "model.pt")
    on_cpu = load_model(path, device="cpu").predict_dist(x, v_lead_future=future, target="mean_1s")
    y = (test.X[:64, -1, 2] * 0.0).astype(np.float64)  # any common evaluation point of the CRPS
    _assert_close(on_cpu, dist, y, 1e-3)
    first = model.predict_dist(x, v_lead_future=future)  # the default target is unchanged: first sampled step
    np.testing.assert_array_equal(first.samples, seq[:, :, 0].astype(np.float64))
