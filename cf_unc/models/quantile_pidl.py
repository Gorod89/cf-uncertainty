"""Physics-informed quantile regression (docs/design.md, section 7: ``QuantilePIDL``).

A ``SeqRegressor`` (LSTM + MLP head) maps the normalised history to Q quantiles of a[t] in normalised
target units. Training loss, every term in normalised target units::

    L = mean_{b,j} rho_tau_j(y_b - q_j(x_b))                                  pinball (data)
      + w_phys * mean_c (q_med(x_c) - a_IDM(x_c))^2                            physics (IDM frozen)
      + w_nc   * mean_c sum_j relu(q_j(x_c) - q_{j+1}(x_c) + margin)^2         non-crossing

``q_med`` is the output at the level closest to 0.5 and ``a_IDM`` is evaluated at the LAST sample of a
window (then normalised like the target). The collocation windows ``x_c`` are the observed batch
windows plus ``n_colloc`` synthetic windows (``synthetic_windows``): last sample uniform in
``colloc_box``, extended backwards at constant ``dv`` and ``v`` with ``s_{t-k} = s_t + k DT dv`` clipped at
0.5 m, drawn from a seeded ``torch.Generator`` owned by the loss closure. Early stopping uses the
pinball term on the validation windows only. No public code of Liu et al. (2023) exists; this is a
re-implementation of the described recipe.

The predictive law is ``cf_unc.dist.QuantileDist`` (piecewise-linear quantile function between the
knots, exponential tails); ``sample_step`` draws from the same law by inverse CDF in torch
(``quantile_icdf``).
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from cf_unc.constants import DT, HIST_LEN, N_FEATURES
from cf_unc.data.windows import WindowSet
from cf_unc.dist import MIN_SPACING_REL, QuantileDist  # same knot spacing rule as the law
from cf_unc.models.base import CFModel, as_history_tensor, register
from cf_unc.models.nets import Normalizer, SeqRegressor
from cf_unc.physics.idm import IDMParams, idm_accel
from cf_unc.train.trainer import Batch, TrainConfig, fit_module

__all__ = [
    "DEFAULT_COLLOC_BOX",
    "DEFAULT_LEVELS",
    "QuantilePIDL",
    "noncrossing_penalty",
    "pinball_loss",
    "quantile_icdf",
    "synthetic_windows",
]

DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
DEFAULT_COLLOC_BOX: dict[str, tuple[float, float]] = {"s": (2.0, 120.0), "dv": (-5.0, 5.0), "v": (0.0, 40.0)}
SYNTH_MIN_GAP = 0.5  # m, lower clip of the backward-extended synthetic gaps
_BOX_KEYS = ("s", "dv", "v")
_F64 = torch.float64
_EVAL_ROWS = 8192  # histories per network call in predict / sample (bounds the LSTM activations)


# --------------------------------------------------------------------------- loss terms
def pinball_loss(y: torch.Tensor, q: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over rows and levels: ``mean rho_tau(y - q)``, ``rho_tau(u) = max(tau u, (tau - 1) u)``.

    y (B,), q (B, Q), levels (Q,) -> scalar.
    """
    u = y[:, None] - q
    return torch.maximum(levels * u, (levels - 1.0) * u).mean()


def noncrossing_penalty(q: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """``mean_b sum_j relu(q_j - q_{j+1} + margin)^2`` for q (B, Q) -> scalar.

    Zero when every row is increasing with gaps of at least ``margin``, positive otherwise.
    """
    return torch.relu(q[:, :-1] - q[:, 1:] + margin).square().sum(dim=1).mean()


def _resolve_box(box: Mapping[str, Sequence[float]] | None) -> dict[str, tuple[float, float]]:
    """Validated collocation box ``{"s": (lo, hi), "dv": (lo, hi), "v": (lo, hi)}`` (defaults for missing keys)."""
    out = dict(DEFAULT_COLLOC_BOX)
    if box is not None:
        unknown = sorted(set(box) - set(_BOX_KEYS))
        if unknown:
            raise ValueError(f"unknown collocation box keys {unknown}; expected {_BOX_KEYS}")
        out.update({k: (float(v[0]), float(v[1])) for k, v in box.items()})
    for key, (lo, hi) in out.items():
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"collocation box {key}: need finite lo < hi, got {(lo, hi)}")
    if out["s"][0] < SYNTH_MIN_GAP or out["v"][0] < 0.0:
        raise ValueError(f"collocation box needs s >= {SYNTH_MIN_GAP} m and v >= 0")
    return out


def synthetic_windows(
    n: int,
    generator: torch.Generator,
    box: Mapping[str, Sequence[float]] | None = None,
    hist_len: int = HIST_LEN,
    dt: float = DT,
) -> torch.Tensor:
    """``n`` synthetic histories (n, hist_len, 3) float32 on the generator's device.

    The last sample ``(s_t, dv, v)`` is uniform in ``box`` (default ``DEFAULT_COLLOC_BOX``); earlier
    samples keep ``dv`` and ``v`` and have ``s_{t-k} = max(s_t + k dt dv, 0.5 m)``, i.e. constant
    speeds of both vehicles (kinematically consistent: ``s[j+1] - s[j] = -dt dv`` where unclipped).
    """
    box = _resolve_box(box)
    lo = torch.tensor([box[k][0] for k in _BOX_KEYS], dtype=_F64, device=generator.device)
    hi = torch.tensor([box[k][1] for k in _BOX_KEYS], dtype=_F64, device=generator.device)
    last = lo + (hi - lo) * torch.rand((int(n), N_FEATURES), generator=generator, dtype=_F64, device=generator.device)
    back = torch.arange(hist_len - 1, -1, -1, dtype=_F64, device=generator.device) * dt  # k dt for j = 0 .. H-1
    s = (last[:, 0:1] + back[None, :] * last[:, 1:2]).clamp_min(SYNTH_MIN_GAP)
    x = torch.stack([s, last[:, 1:2].expand(-1, hist_len), last[:, 2:3].expand(-1, hist_len)], dim=-1)
    return x.to(torch.float32)


# --------------------------------------------------------------------------- the law in torch
def _sorted_knots(values: torch.Tensor) -> torch.Tensor:
    """Knots as ``QuantileDist`` stores them: sorted along Q, pushed apart to >= 1e-6 max(1, range)."""
    q = torch.sort(values, dim=1).values
    delta = MIN_SPACING_REL * torch.clamp(q[:, -1] - q[:, 0], min=1.0)
    bad = torch.any(torch.diff(q, dim=1) < delta[:, None], dim=1)
    if bool(bad.any()):
        shift = delta[bad, None] * torch.arange(q.shape[1], dtype=q.dtype, device=q.device)
        q[bad] = torch.cummax(q[bad] - shift, dim=1).values + shift
    return q


def quantile_icdf(levels: torch.Tensor, values: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Quantile function of the ``QuantileDist`` law at probabilities ``u`` (B,) -> (B,), float64.

    levels (Q,) increasing in (0, 1); values (B, Q) knots (sorted here). Linear between the knots,
    exponential tails ``q_1 + beta_L log(u / tau_1)`` below ``tau_1`` and ``q_Q - beta_U log((1 - u) /
    (1 - tau_Q))`` above ``tau_Q`` with ``beta_L = tau_1 (q_2 - q_1) / (tau_2 - tau_1)`` and ``beta_U =
    (1 - tau_Q) (q_Q - q_{Q-1}) / (tau_Q - tau_{Q-1})``. ``u`` is floored at the smallest normal float.
    """
    tau = levels.to(_F64)
    q = _sorted_knots(values.to(_F64))
    u = u.to(_F64).clamp_min(torch.finfo(_F64).tiny)
    nq = tau.shape[0]
    beta_l = tau[0] * (q[:, 1] - q[:, 0]) / (tau[1] - tau[0])
    beta_u = (1.0 - tau[-1]) * (q[:, -1] - q[:, -2]) / (tau[-1] - tau[-2])
    seg = (torch.searchsorted(tau, u, right=True) - 1).clamp(0, nq - 2)
    q0 = q.gather(1, seg[:, None])[:, 0]
    q1 = q.gather(1, seg[:, None] + 1)[:, 0]
    inside = q0 + (q1 - q0) * (u - tau[seg]) / (tau[seg + 1] - tau[seg])
    lower = q[:, 0] + beta_l * torch.log(torch.clamp(u, max=float(tau[0])) / tau[0])
    upper = q[:, -1] - beta_u * torch.log(torch.clamp(1.0 - u, max=float(1.0 - tau[-1])) / (1.0 - tau[-1]))
    return torch.where(u < tau[0], lower, torch.where(u > tau[-1], upper, inside))


# --------------------------------------------------------------------------- model
@register("quantile_pidl")
class QuantilePIDL(CFModel):
    """LSTM quantile regressor with an IDM physics term and a non-crossing penalty (module docstring).

    ``train`` holds overrides of ``TrainConfig`` (unknown keys are an error); ``margin`` is in normalised
    target units; ``n_colloc = None`` draws as many synthetic windows as the observed batch has rows.
    """

    uses_history = True
    has_epistemic = False

    def __init__(
        self,
        idm: Mapping[str, Any] | IDMParams,
        levels: Sequence[float] = DEFAULT_LEVELS,
        w_phys: float = 0.3,
        w_nc: float = 10.0,
        margin: float = 0.0,
        colloc_box: Mapping[str, Sequence[float]] | None = None,
        n_colloc: int | None = None,
        hidden: int = 64,
        head_hidden: int = 64,
        train: Mapping[str, Any] | None = None,
    ) -> None:
        idm_dict = idm.to_dict() if isinstance(idm, IDMParams) else dict(idm)
        box = _resolve_box(colloc_box)
        super().__init__(
            idm=idm_dict,
            levels=[float(t) for t in levels],
            w_phys=float(w_phys),
            w_nc=float(w_nc),
            margin=float(margin),
            colloc_box={k: list(v) for k, v in box.items()},
            n_colloc=None if n_colloc is None else int(n_colloc),
            hidden=int(hidden),
            head_hidden=int(head_hidden),
            train=None if train is None else dict(train),
        )
        self.idm = IDMParams.from_dict(idm_dict)
        if not all(isinstance(getattr(self.idm, name), float) for name in idm_dict):
            raise ValueError("QuantilePIDL needs scalar (population) IDM parameters")
        self.levels = np.array(self.init_kwargs["levels"], dtype=np.float64)
        if self.levels.ndim != 1 or self.levels.size < 3:
            raise ValueError("need at least 3 quantile levels")
        if not (self.levels[0] > 0.0 and self.levels[-1] < 1.0 and np.all(np.diff(self.levels) > 0)):
            raise ValueError("levels must be strictly increasing inside (0, 1)")
        if min(w_phys, w_nc, margin) < 0.0:
            raise ValueError("w_phys, w_nc and margin must be >= 0")
        if n_colloc is not None and int(n_colloc) < 0:
            raise ValueError("n_colloc must be None or >= 0")
        unknown = sorted(set(train or {}) - set(TrainConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown TrainConfig fields {unknown}")
        self.train_cfg = TrainConfig(**dict(train or {}))
        self.w_phys, self.w_nc, self.margin = float(w_phys), float(w_nc), float(margin)
        self.colloc_box = box
        self.n_colloc = None if n_colloc is None else int(n_colloc)
        self.median_index = int(np.argmin(np.abs(self.levels - 0.5)))
        self.net_spec = dict(out_dim=int(self.levels.size), hidden=int(hidden), head_hidden=int(head_hidden))
        self.norm = Normalizer()
        self.net: SeqRegressor | None = None

    # ------------------------------------------------------------------ helpers
    def _levels_t(self) -> torch.Tensor:
        return torch.as_tensor(self.levels, dtype=torch.float32, device=self.device)

    def _require_net(self) -> SeqRegressor:
        if self.net is None:
            raise RuntimeError("QuantilePIDL is not fitted")
        return self.net

    def loss_terms(
        self, module: nn.Module, x: torch.Tensor, y: torch.Tensor, x_colloc: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(pinball, physics, non-crossing)`` scalars in normalised target units.

        x (B, H, 3) observed histories and y (B,) targets (physical); x_colloc (C, H, 3) synthetic
        histories or None. The pinball term uses the observed rows, physics (median vs frozen IDM at
        the last sample) and non-crossing use observed and synthetic rows together.
        """
        x_all = x if x_colloc is None else torch.cat([x, x_colloc.to(x.device)])
        q = module(self.norm.norm_x(x_all))
        pin = pinball_loss(self.norm.norm_y(y), q[: x.shape[0]], self._levels_t())
        last = x_all[:, -1, :]
        a_idm = self.norm.norm_y(idm_accel(last[:, 0], last[:, 1], last[:, 2], self.idm))
        phys = (q[:, self.median_index] - a_idm).square().mean()
        return pin, phys, noncrossing_penalty(q, self.margin)

    def _n_colloc(self, n_obs: int) -> int:
        if self.w_phys == 0.0 and self.w_nc == 0.0:
            return 0  # synthetic windows would not enter the loss
        return n_obs if self.n_colloc is None else self.n_colloc

    def make_loss(self, generator: torch.Generator) -> Callable[[nn.Module, Batch], torch.Tensor]:
        """Training loss closure; it owns ``generator`` (CPU), the source of the synthetic windows."""

        def loss(module: nn.Module, batch: Batch) -> torch.Tensor:
            x, y = batch["X"], batch["y"]
            n_c = self._n_colloc(int(x.shape[0]))
            x_c = synthetic_windows(n_c, generator, self.colloc_box, hist_len=int(x.shape[1])) if n_c > 0 else None
            pin, phys, nc = self.loss_terms(module, x, y, x_c)
            return pin + self.w_phys * phys + self.w_nc * nc

        return loss

    def val_loss(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        """Pinball term only (early stopping)."""
        q = module(self.norm.norm_x(batch["X"]))
        return pinball_loss(self.norm.norm_y(batch["y"]), q, self._levels_t())

    @torch.no_grad()
    def evaluate_terms(self, windows: WindowSet, seed: int = 0, batch_size: int | None = None) -> dict[str, float]:
        """Row-weighted means of the three loss terms and of the total over ``windows`` (eval mode).

        Synthetic windows are drawn as in training (``_n_colloc`` per ``batch_size`` observed rows,
        default the training batch size) from a generator seeded with ``seed``.
        """
        net = self._require_net().eval()
        bs = int(batch_size or self.train_cfg.batch_size)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        sums, n = np.zeros(3), len(windows)
        for a in range(0, n, bs):
            x = as_history_tensor(windows.X[a : a + bs], self.device)
            y = torch.as_tensor(windows.y[a : a + bs], dtype=torch.float32, device=self.device)
            n_c = self._n_colloc(int(x.shape[0]))
            x_c = synthetic_windows(n_c, gen, self.colloc_box, hist_len=int(x.shape[1])) if n_c > 0 else None
            sums += np.array([float(t) for t in self.loss_terms(net, x, y, x_c)]) * x.shape[0]
        pin, phys, nc = (float(v) for v in sums / max(n, 1))
        return {
            "pinball": pin,
            "physics": phys,
            "noncrossing": nc,
            "total": pin + self.w_phys * phys + self.w_nc * nc,
        }

    # ------------------------------------------------------------------ training
    def initial_offsets(self, train: WindowSet) -> np.ndarray:
        """Initial output biases (Q,): quantiles of the frozen IDM's training residuals, normalised.

        ``(quantile_tau(r) - mean(r)) / y_std`` with ``r = y - a_IDM(last sample)``: the network starts with
        the quantile spread of the physics model (Adam moves a bias by ~lr per step, so starting from the
        marginal spread of the target would take hundreds of steps to shrink).
        """
        last = np.asarray(train.X[:, -1, :], dtype=np.float64)
        resid = np.asarray(train.y, dtype=np.float64) - idm_accel(last[:, 0], last[:, 1], last[:, 2], self.idm)
        offsets = (np.quantile(resid, self.levels) - resid.mean()) / float(self.norm.y_std)
        return np.maximum.accumulate(offsets)  # quantiles are sorted already; guard against rounding

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Fit normaliser and network (``fit_module``, early stopping on the validation pinball loss).

        Log (JSON-serialisable): ``val_nll`` (physical units, final law), ``val_pinball``,
        ``val_crossing_share`` (validation windows whose raw outputs cross), ``terms`` (final pinball /
        physics / non-crossing values on the training windows, normalised units), ``n_train``,
        ``epochs``, ``seconds`` and the trainer log ``fit``.
        """
        started = time.perf_counter()
        if len(train) == 0:
            raise ValueError("empty training set")
        seed = int(seed)
        self.norm = Normalizer().fit(train.X, train.y).to(self.device)
        forked = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=forked):
            torch.manual_seed(seed)
            net = SeqRegressor(**self.net_spec)
        with torch.no_grad():
            net.fc2.bias.copy_(torch.as_tensor(self.initial_offsets(train), dtype=torch.float32))
        self.net = net.to(self.device)
        colloc_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
        loss_fn = self.make_loss(torch.Generator(device="cpu").manual_seed(colloc_seed))
        has_val = val is not None and len(val) > 0
        fit_log = fit_module(
            self.net,
            loss_fn,
            {"X": train.X, "y": train.y},
            {"X": val.X, "y": val.y} if has_val else None,
            self.train_cfg,
            seed=seed,
            device=self.device,
            verbose=verbose,
            val_loss_fn=self.val_loss,
        )
        self.net.eval()
        self.fitted = True
        log: dict[str, Any] = {
            "val_nll": None,
            "val_pinball": None,
            "val_crossing_share": None,
            "terms": self.evaluate_terms(train, seed=colloc_seed + 1),
            "n_train": int(len(train)),
            "epochs": int(fit_log["epochs"]),
        }
        if has_val:
            dist = self.predict_dist(val.X)
            log["val_nll"] = float(-np.mean(dist.log_prob(val.y)))
            log["val_pinball"] = float(fit_log["best_val"])
            log["val_crossing_share"] = float(np.mean(dist.n_crossings > 0))
        log["seconds"] = time.perf_counter() - started
        log["fit"] = fit_log
        return log

    # ---------------------------------------------------------------- prediction
    def quantiles_t(self, hist: torch.Tensor) -> torch.Tensor:
        """Raw (unsorted) quantiles (B, Q) float64 [m/s^2] for physical histories (B, H, 3) on the device."""
        q_n = self._require_net()(self.norm.norm_x(hist)).to(_F64)
        return float(self.norm.y_mean) + float(self.norm.y_std) * q_n

    @torch.no_grad()
    def quantile_values(self, state_history: np.ndarray | torch.Tensor, batch_size: int = 8192) -> np.ndarray:
        """Raw quantiles (N, Q) float64 [m/s^2] for histories (N, H, 3) or (H, 3), in row chunks."""
        self._require_net().eval()
        x = as_history_tensor(state_history, self.device)
        step = max(1, int(batch_size))
        parts = [self.quantiles_t(x[a : a + step]) for a in range(0, x.shape[0], step)]
        if not parts:
            return np.zeros((0, self.levels.size))
        return torch.cat(parts).cpu().numpy()

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> QuantileDist:
        """``QuantileDist(levels, values)`` in m/s^2 (it sorts crossing quantiles and counts them)."""
        return QuantileDist(self.levels, self.quantile_values(state_history, batch_size))

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        """One draw (B,) [m/s^2] per history (B, H, 3) from the ``predict_dist`` law (inverse CDF).

        Nothing is frozen (``state`` is ignored; both sampler modes are identical); one uniform per row
        comes from ``generator``. The result has the dtype of ``hist`` (float32 for rollout histories).
        """
        self._require_net().eval()
        x = as_history_tensor(hist, self.device)
        u = torch.rand(x.shape[0], generator=generator, dtype=_F64, device=self.device)
        levels = torch.as_tensor(self.levels, dtype=_F64, device=self.device)
        out = torch.empty(x.shape[0], dtype=_F64, device=self.device)
        for a in range(0, x.shape[0], _EVAL_ROWS):
            b = a + _EVAL_ROWS
            out[a:b] = quantile_icdf(levels, self.quantiles_t(x[a:b]), u[a:b])
        return out.to(hist.dtype if torch.is_floating_point(hist) else torch.float32)

    # --------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        net = self._require_net()
        return {
            "net": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "norm": {k: v.detach().cpu().clone() for k, v in self.norm.state_dict().items()},
        }

    def set_state(self, state: dict) -> None:
        norm = Normalizer()
        norm.load_state_dict(state["norm"])
        net = SeqRegressor(**self.net_spec)
        net.load_state_dict(state["net"])
        self.norm = norm.to(self.device)
        self.net = net.to(self.device).eval()
        self.fitted = True
