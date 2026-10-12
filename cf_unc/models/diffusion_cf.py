"""DiffusionCF: a minimal conditional DDPM over 5 s follower acceleration sequences (docs/design.md, section 7).

In the spirit of FollowGen: ``eps_theta(x_k, k, cond)`` removes noise from a standardised acceleration
sequence ``a[t : t+L]`` given the normalised 3 s history and the leader's future speed relative to the current
follower speed. Cosine schedule (Nichol and Dhariwal 2021) with ``t_train`` steps, epsilon prediction, MSE
loss, EMA weights; DDIM sampling (Song et al. 2021) on a strided sub-sequence of ``n_sampling_steps`` steps
with stochasticity ``eta`` (1: ancestral-like, 0: deterministic). Written from scratch (no diffusers).

Diffusion step ``k = 0 .. t_train-1`` is the noise level after ``k + 1`` forward steps:
``x_k = sqrt(ab_k) x_0 + sqrt(1 - ab_k) eps`` with ``ab_k = alpha_bar[k]``.

``DiffusionCFHist`` (registry name ``diffusion_cf_hist``, review control R2): the same model without the leader-future
input, conditioned on the history only.

Preview budget (extension X1, extension/docs/PLAN.md): ``preview_steps`` (constructor) holds the leader's future
constant beyond that many steps at training and prediction time (stage B, a law trained at a budget); a
``PreviewSpec`` given to ``init_sampler`` truncates, perturbs or delays the preview handed at every re-plan of a
rollout (stage A, stored checkpoints). Without either, nothing changes.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cf_unc.constants import A_MAX, A_MIN, MEAN_TARGET_STEPS, N_FEATURES, SEQ_LEN
from cf_unc.data.windows import WindowSet
from cf_unc.dist import SampleDist
from cf_unc.models.base import CFModel, as_history_tensor, register
from cf_unc.models.nets import Normalizer
from cf_unc.preview import PreviewSpec
from cf_unc.train.trainer import Batch, TrainConfig, fit_module

MAX_SEQUENCES = 4096  # sequences denoised at once (contract: <= ~16 000; 2-4k chunks are fastest on CPU)
TARGET_SAMPLES = {"a": 1, "mean_1s": MEAN_TARGET_STEPS}  # sampled steps that make the law of a target (predict_dist)
RNG_BLOCK = 16  # rows per noise stream of the seeded sampler (makes predict_dist independent of batch_size)
VAL_NLL_WINDOWS = 500  # validation windows scored by the approximate (KDE) first-step NLL in ``fit``
EMA_WARMUP_EPOCHS = 1  # epochs trained without EMA before the EMA phase (if ema_decay > 0)
DEFAULT_TRAIN: dict[str, Any] = dict(  # 1 EMA warm-up + 11 EMA epochs: 1.5-2.5 min on 4 shared CPU threads (E4)
    max_epochs=12, batch_size=64, lr=1e-3, weight_decay=0.0, patience=4, lr_patience=2, ema_decay=0.999,
    min_delta=1e-5,  # the validation loss is deterministic (fixed steps and noises): small improvements are real
)

EpsFn = Callable[[torch.Tensor, int], torch.Tensor]  # (x_k (B, L), k) -> predicted noise (B, L)
DrawFn = Callable[[], torch.Tensor]  # standard normal noise shaped like the sampled batch


def hold_beyond(v: torch.Tensor, steps: int | None) -> torch.Tensor:
    """(N, L) leader speeds with the columns from ``steps`` on replaced by column ``steps - 1`` (the last seen value
    held constant); ``steps`` None or >= L: ``v`` itself."""
    if steps is None or int(steps) >= v.shape[1]:
        return v
    k = int(steps)
    return torch.cat([v[:, :k], v[:, k - 1 : k].expand(-1, v.shape[1] - k)], dim=1)


def hold_beyond_np(v: np.ndarray, steps: int | None) -> np.ndarray:
    """NumPy twin of :func:`hold_beyond`."""
    if steps is None or int(steps) >= v.shape[1]:
        return v
    k = int(steps)
    return np.concatenate([v[:, :k], np.repeat(v[:, k - 1 : k], v.shape[1] - k, axis=1)], axis=1)


# ------------------------------------------------------------------------ diffusion maths
def cosine_alpha_bar(t_train: int, s: float = 0.008, max_beta: float = 0.999) -> np.ndarray:
    """(t_train,) float64 ``alpha_bar_k = prod_{i<=k} (1 - beta_i)`` of the cosine schedule.

    ``beta_i = min(1 - f(i+1) / f(i), max_beta)`` with ``f(u) = cos((u / T + s) / (1 + s) pi / 2)^2``;
    for T = 1000: alpha_bar_0 = 1 - 4.1e-5, alpha_bar_{T-1} = 2.4e-9 (the last beta is clipped).
    """
    u = np.arange(t_train + 1, dtype=np.float64) / t_train
    f = np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2
    return np.cumprod(1.0 - np.minimum(1.0 - f[1:] / f[:-1], max_beta))


def sampling_steps(t_train: int, n_steps: int) -> np.ndarray:
    """(n_steps,) distinct decreasing diffusion steps, evenly spaced from t_train-1 down to 0."""
    if not 2 <= n_steps <= t_train:
        raise ValueError(f"need 2 <= n_sampling_steps <= t_train, got {n_steps}, {t_train}")
    return np.floor(np.linspace(t_train - 1.0, 0.0, n_steps) + 0.5).astype(np.int64)


def q_sample(x0: torch.Tensor, k: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Forward process ``x_k = sqrt(ab_k) x0 + sqrt(1 - ab_k) eps``; x0, eps (B, L), k (B,) int, alpha_bar (T,)."""
    ab = alpha_bar[k].unsqueeze(-1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def ddim_step(
    x: torch.Tensor,
    eps: torch.Tensor,
    ab_t: float,
    ab_prev: float,
    eta: float,
    noise: torch.Tensor | None,
    x0_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DDIM update x_t -> x_prev (Song et al. 2021, eq. 12); returns ``(x_prev, x0_hat)``.

    ``x0_hat = (x - sqrt(1 - ab_t) eps) / sqrt(ab_t)``, clipped to ``x0_range`` (eps is then re-derived from
    the clipped x0); ``sigma = eta sqrt((1 - ab_prev) / (1 - ab_t) (1 - ab_t / ab_prev))`` (eta = 1: the
    DDPM posterior); ``x_prev = sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev - sigma^2) eps + sigma noise``.
    """
    sq_ab, sq_1m = math.sqrt(ab_t), math.sqrt(1.0 - ab_t)
    x0 = (x - sq_1m * eps) / sq_ab
    if x0_range is not None:
        x0 = x0.clamp(*x0_range)
        eps = (x - sq_ab * x0) / sq_1m
    sigma = eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev))
    x_prev = math.sqrt(ab_prev) * x0 + math.sqrt(max(1.0 - ab_prev - sigma * sigma, 0.0)) * eps
    if sigma > 0.0:
        if noise is None:
            raise ValueError("eta > 0 needs noise")
        x_prev = x_prev + sigma * noise
    return x_prev, x0


def ddim_sample(
    eps_fn: EpsFn,
    x: torch.Tensor,
    alpha_bar: np.ndarray,
    steps: np.ndarray,
    eta: float,
    draw: DrawFn | None = None,
    x0_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Run DDIM from ``x`` (noise at ``steps[0]``) through the decreasing ``steps``; returns x0 (same shape).

    ``draw()`` supplies the fresh noise of every stochastic update (not called when ``eta = 0``).
    """
    for i, k in enumerate(steps):
        ab_prev = float(alpha_bar[steps[i + 1]]) if i + 1 < len(steps) else 1.0
        noise = draw() if eta > 0.0 and i + 1 < len(steps) and draw is not None else None
        x, _ = ddim_step(x, eps_fn(x, int(k)), float(alpha_bar[k]), ab_prev, eta, noise, x0_range)
    return x


# ------------------------------------------------------------------------------ denoiser
def step_embedding(k: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) diffusion steps -> (B, dim) sinusoidal embedding (sin | cos, periods up to 2 pi 10^4)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=k.device, dtype=torch.float32) / half)
    arg = k.to(torch.float32).unsqueeze(-1) * freqs
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class _ResBlock(nn.Module):
    """Pre-activation residual block ``h + conv(SiLU(GN(h) (1 + scale) + shift))``, (scale, shift) = FiLM(cond)."""

    def __init__(self, channels: int, kernel: int, cond_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(8, channels), channels, affine=False)
        self.film = nn.Linear(cond_dim, 2 * channels)
        self.conv = nn.Conv1d(channels, channels, kernel, padding="same", dilation=dilation)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        return h + self.conv(F.silu(torch.addcmul(shift, self.norm(h), 1.0 + scale)))


class Denoiser(nn.Module):
    """``eps_theta(x_k, k, cond)`` on standardised sequences.

    LSTM(hidden) over the normalised history (B, H, 3) -> context; sinusoidal step embedding -> MLP; both
    -> MLP -> per-block FiLM scale / shift of a residual 1-D conv stack over the sequence axis with input
    channels ``[x_k, v_lead_rel]`` (B, 2, L); block b is dilated by 3^(b mod 4), so that with 4 blocks of
    kernel 5 every output step sees the whole 50-step sequence; zero-initialised 1x1 output convolution.
    Holds the input normaliser (history, pooled acceleration sequence) and ``vl_scale`` [m/s] as buffers.

    ``leader=False`` (DiffusionCFHist, review control R2): the leader-future channel is removed, the input
    convolution sees ``[x_k]`` only (B, 1, L), so the denoiser is conditioned on the history alone; everything else
    (encoder, FiLM, conv stack, step embedding) is the same network.
    """

    def __init__(self, seq_len: int, channels: int, n_blocks: int, kernel: int, hidden: int, leader: bool = True) -> None:
        super().__init__()
        self.seq_len, self.channels, self.leader = seq_len, channels, bool(leader)
        self.norm = Normalizer(N_FEATURES)
        self.register_buffer("vl_scale", torch.ones(()))
        self.encoder = nn.LSTM(N_FEATURES, hidden, batch_first=True)
        self.step_mlp = nn.Sequential(nn.Linear(channels, 2 * channels), nn.SiLU(), nn.Linear(2 * channels, channels))
        cond_dim = 2 * channels
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden + channels, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim), nn.SiLU()
        )
        self.inp = nn.Conv1d(2 if self.leader else 1, channels, kernel, padding="same")
        self.blocks = nn.ModuleList(_ResBlock(channels, kernel, cond_dim, 3 ** (b % 4)) for b in range(n_blocks))
        self.out_norm = nn.GroupNorm(math.gcd(8, channels), channels)
        self.out = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def context(self, hist_n: torch.Tensor) -> torch.Tensor:
        """(B, H, 3) normalised history -> (B, hidden) last LSTM output."""
        return self.encoder(hist_n)[0][:, -1]

    def denoise(
        self, x_k: torch.Tensor, k: torch.Tensor, ctx: torch.Tensor, vl_rel: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x_k (B, L), k (B,) or (1,) int, ctx (B, hidden), vl_rel (B, L) -> predicted noise (B, L).

        ``vl_rel`` is ignored (may be None) by a denoiser without the leader channel (``leader=False``)."""
        temb = self.step_mlp(step_embedding(k, self.channels)).expand(ctx.shape[0], -1)
        cond = self.cond_mlp(torch.cat([ctx, temb], dim=-1))
        if self.leader:
            if vl_rel is None:
                raise ValueError("this denoiser is conditioned on the leader's future: vl_rel is required")
            h = self.inp(torch.stack([x_k, vl_rel], dim=1))
        else:
            h = self.inp(x_k.unsqueeze(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(F.silu(self.out_norm(h))).squeeze(1)

    def forward(
        self, x_k: torch.Tensor, k: torch.Tensor, hist_n: torch.Tensor, vl_rel: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.denoise(x_k, k, self.context(hist_n), vl_rel)


def _seed(*keys: int) -> int:
    """63-bit torch seed derived from integer keys (independent streams for distinct keys)."""
    return int(np.random.SeedSequence([abs(int(k)) for k in keys]).generate_state(1, np.uint64)[0] >> np.uint64(1))


# --------------------------------------------------------------------------------- model
@register("diffusion_cf")
class DiffusionCF(CFModel):
    """Conditional DDPM of the follower's next ``seq_len`` accelerations given history and leader future.

    ``predict_dist`` returns the ``SampleDist`` of the first step of ``n_samples`` sampled sequences (target
    ``"mean_1s"``: of the mean of their first 10 steps; aleatoric = sample variance, no epistemic part).
    ``train`` overrides ``DEFAULT_TRAIN`` (TrainConfig).
    """

    needs_leader_future = True
    plan_sampler = True

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        n_samples: int = 32,
        n_sampling_steps: int = 50,
        t_train: int = 1000,
        eta: float = 1.0,
        channels: int = 64,
        n_blocks: int = 4,
        kernel: int = 5,
        hidden: int = 64,
        replan_every: int = 50,
        predict_seed: int = 0,
        clip_x0: bool = True,
        preview_steps: int | None = None,
        train: dict | None = None,
    ) -> None:
        super().__init__(
            seq_len=seq_len, n_samples=n_samples, n_sampling_steps=n_sampling_steps, t_train=t_train, eta=eta,
            channels=channels, n_blocks=n_blocks, kernel=kernel, hidden=hidden, replan_every=replan_every,
            predict_seed=predict_seed, clip_x0=clip_x0, preview_steps=preview_steps,
            train=None if train is None else dict(train),
        )  # fmt: skip
        if seq_len < 1 or n_samples < 2 or eta < 0.0 or min(channels, n_blocks, kernel, hidden) < 1:
            raise ValueError("need seq_len >= 1, n_samples >= 2, eta >= 0 and positive network sizes")
        if not 1 <= replan_every <= seq_len:
            raise ValueError(f"need 1 <= replan_every <= seq_len, got {replan_every}, {seq_len}")
        if preview_steps is not None:
            if isinstance(preview_steps, bool) or not isinstance(preview_steps, (int, np.integer)) or \
                    not 1 <= int(preview_steps) <= seq_len:  # fmt: skip
                raise ValueError(f"preview_steps must be None or an integer in [1, seq_len = {seq_len}], got {preview_steps!r}")
            if not self.needs_leader_future:
                raise ValueError(f"{type(self).__name__} has no leader input: preview_steps must be None")
        self.preview_steps = None if preview_steps is None else int(preview_steps)
        unknown = sorted(set(train or {}) - set(TrainConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown train options {unknown}")
        self.seq_len, self.n_samples, self.t_train, self.eta = int(seq_len), int(n_samples), int(t_train), float(eta)
        self.arch = dict(seq_len=self.seq_len, channels=channels, n_blocks=n_blocks, kernel=kernel, hidden=hidden)
        if not self.needs_leader_future:  # DiffusionCFHist: no leader-future channel (the denoiser default keeps it)
            self.arch["leader"] = False
        self.replan_every, self.predict_seed, self.clip_x0 = int(replan_every), int(predict_seed), bool(clip_x0)
        self.train_cfg = TrainConfig.from_any({**DEFAULT_TRAIN, **(train or {})})
        self.alpha_bar = cosine_alpha_bar(self.t_train)
        self.steps = sampling_steps(self.t_train, int(n_sampling_steps))
        self.net: Denoiser | None = None

    # -------------------------------------------------------------------- helpers
    def _require_net(self) -> Denoiser:
        if self.net is None:
            raise RuntimeError("DiffusionCF is not fitted")
        return self.net

    def _leader(self, hist: torch.Tensor, v_lead_future: np.ndarray | torch.Tensor | None) -> torch.Tensor | None:
        """(N, L) leader speed [m/s] at t+1 .. t+L, held constant beyond ``preview_steps``; constant ``v_t - dv_t``
        if None. A model without the leader input (``needs_leader_future`` False, DiffusionCFHist) returns None
        whatever is passed."""
        if not self.needs_leader_future:
            return None
        if v_lead_future is None:
            return (hist[:, -1, 2] - hist[:, -1, 1]).unsqueeze(-1).expand(-1, self.seq_len)
        vlf = torch.as_tensor(v_lead_future if torch.is_tensor(v_lead_future) else np.asarray(v_lead_future))
        vlf = vlf.to(device=self.device, dtype=torch.float32)
        vlf = vlf.unsqueeze(0) if vlf.ndim == 1 else vlf
        if tuple(vlf.shape) != (hist.shape[0], self.seq_len):
            raise ValueError(f"v_lead_future must have shape ({hist.shape[0]}, {self.seq_len}), got {tuple(vlf.shape)}")
        return hold_beyond(vlf, self.preview_steps)

    def _arrays(self, ws: WindowSet) -> dict[str, np.ndarray]:
        """Normalised training arrays: hist (N, H, 3), vl (N, L) relative leader speed (only with the leader input),
        y (N, L)."""
        norm = self._require_net().norm
        x_mean, x_std = norm.x_mean.cpu().numpy(), norm.x_std.cpu().numpy()
        out = {
            "hist": ((ws.X - x_mean) / x_std).astype(np.float32),
            "y": ((ws.y_seq - float(norm.y_mean)) / float(norm.y_std)).astype(np.float32),
        }
        if self.needs_leader_future:
            rel = hold_beyond_np(ws.v_lead_future.astype(np.float64), self.preview_steps) - ws.X[:, -1:, 2]
            out["vl"] = (rel / float(self.net.vl_scale)).astype(np.float32)
        return out

    @torch.no_grad()
    def _denoise_chunk(
        self, hist: torch.Tensor, vlf: torch.Tensor | None, n_samples: int, draw: DrawFn
    ) -> torch.Tensor:
        """Sample (N, S, L) accelerations [m/s^2] for histories (N, H, 3) and leader speeds (N, L) (None: a model
        without the leader input)."""
        net = self._require_net()
        rel = None if vlf is None else ((vlf - hist[:, -1, 2:3]) / net.vl_scale).repeat_interleave(n_samples, dim=0)
        ctx = net.context(net.norm.norm_x(hist)).repeat_interleave(n_samples, dim=0)
        y_mean, y_std = float(net.norm.y_mean), float(net.norm.y_std)
        x0_range = ((A_MIN - y_mean) / y_std, (A_MAX - y_mean) / y_std) if self.clip_x0 else None

        def eps_fn(x: torch.Tensor, k: int) -> torch.Tensor:
            return net.denoise(x, torch.full((1,), k, dtype=torch.long, device=x.device), ctx, rel)

        x0 = ddim_sample(eps_fn, draw(), self.alpha_bar, self.steps, self.eta, draw, x0_range)
        return (y_mean + y_std * x0).clamp(A_MIN, A_MAX).reshape(hist.shape[0], n_samples, self.seq_len)

    def _sample(
        self, hist: torch.Tensor, vlf: torch.Tensor | None, n_samples: int, generator: torch.Generator | None,
        batch_size: int,
    ) -> torch.Tensor:  # fmt: skip
        """Chunked sampling, (N, S, L) tensor. ``generator=None``: one seeded noise stream per RNG_BLOCK rows.

        The seeded streams are CPU generators on every device (the noise is moved to the model's device), so that
        a checkpoint gives the same ``predict_dist`` samples on CPU and GPU up to floating-point arithmetic. A given
        ``generator`` draws on its own device.
        """
        n, length = hist.shape[0], self.seq_len
        rows = max(1, min(int(batch_size), MAX_SEQUENCES // n_samples))
        if generator is None:
            rows = max(1, rows // RNG_BLOCK) * RNG_BLOCK
        out = []
        for a in range(0, n, rows):
            b = min(a + rows, n)
            streams = [(generator, b - a, b - a)]  # (generator, rows used, rows drawn)
            if generator is None:  # one stream per block of RNG_BLOCK rows, seeded by the block index; full blocks
                streams = []  # are drawn so that the noise of a row does not depend on the rows in its block
                for r in range(a, b, RNG_BLOCK):
                    g = torch.Generator(device="cpu").manual_seed(_seed(self.predict_seed, r // RNG_BLOCK))
                    streams.append((g, min(r + RNG_BLOCK, b) - r, RNG_BLOCK))

            def draw(streams: list = streams) -> torch.Tensor:
                z = [torch.randn((full * n_samples, length), generator=g, device=g.device) for g, _, full in streams]
                return torch.cat([zi[: m * n_samples] for zi, (_, m, _) in zip(z, streams)]).to(self.device)

            out.append(self._denoise_chunk(hist[a:b], None if vlf is None else vlf[a:b], n_samples, draw))
        return torch.cat(out) if out else torch.empty((0, n_samples, length), device=self.device)

    # ------------------------------------------------------------------- training
    def _train_loss(self, seed: int) -> Callable[[nn.Module, Batch], torch.Tensor]:
        """eps-MSE with uniform steps and noise from a generator owned by the closure."""
        gen = torch.Generator(device=self.device).manual_seed(_seed(seed, 1))
        ab = torch.as_tensor(self.alpha_bar, dtype=torch.float32, device=self.device)

        def loss(net: nn.Module, batch: Batch) -> torch.Tensor:
            x0 = batch["y"]
            k = torch.randint(0, self.t_train, (x0.shape[0],), generator=gen, device=x0.device)
            eps = torch.randn(x0.shape, generator=gen, device=x0.device)
            return F.mse_loss(net(q_sample(x0, k, eps, ab), k, batch["hist"], batch.get("vl")), eps)

        return loss

    def _val_loss(self, net: nn.Module, batch: Batch) -> torch.Tensor:
        """Deterministic eps-MSE with the steps ``k`` and noises ``eps`` stored in the validation arrays."""
        ab = torch.as_tensor(self.alpha_bar, dtype=torch.float32, device=batch["y"].device)
        x_k = q_sample(batch["y"], batch["k"], batch["eps"], ab)
        return F.mse_loss(net(x_k, batch["k"], batch["hist"], batch.get("vl")), batch["eps"])

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Train on windows cut with ``seq_len = L`` (needs ``y_seq``, and ``v_lead_future`` with the leader input);
        JSON-able log."""
        for name, ws in (("train", train), ("val", val)):
            if ws is not None and (ws.y_seq is None or (self.needs_leader_future and ws.v_lead_future is None)):
                what = "y_seq and v_lead_future" if self.needs_leader_future else "y_seq"
                raise ValueError(f"{type(self).__name__} needs {name} windows with {what} (seq_len > 0)")
            if ws is not None and ws.y_seq.shape[1] != self.seq_len:
                raise ValueError(f"{name} windows have seq_len {ws.y_seq.shape[1]}, the model expects {self.seq_len}")
        started = time.perf_counter()
        torch.manual_seed(_seed(seed, 0))
        self.net = net = Denoiser(**self.arch).to(self.device)
        net.norm.fit(train.X, train.y_seq)
        if self.needs_leader_future:  # the history-only model keeps vl_scale = 1 (unused)
            rel = hold_beyond_np(train.v_lead_future.astype(np.float64), self.preview_steps) - train.X[:, -1:, 2]
            net.vl_scale.fill_(max(float(rel.std()), 1e-6))
        val_arrays = None
        if val is not None and len(val) > 0:
            n, rng = len(val), np.random.default_rng([abs(int(seed)), 2])
            val_arrays = self._arrays(val)
            val_arrays["k"] = np.floor((rng.permutation(n) + rng.random(n)) * self.t_train / n).astype(np.int64)
            val_arrays["eps"] = rng.standard_normal((n, self.seq_len)).astype(np.float32)
        # fit_module starts its EMA at the current weights: a short phase without EMA first (EMA warm-up),
        # then the EMA phase, whose best EMA weights are returned; one wall-clock budget for both phases
        cfg, loss, arrays = self.train_cfg, self._train_loss(seed), self._arrays(train)
        warm = min(EMA_WARMUP_EPOCHS, cfg.max_epochs - 1) if cfg.ema_decay > 0.0 else 0
        phases = ([dataclasses.replace(cfg, max_epochs=warm, ema_decay=0.0)] if warm > 0 else []) + [
            dataclasses.replace(cfg, max_epochs=cfg.max_epochs - warm)
        ]
        logs: list[dict] = []
        for i, phase in enumerate(phases):
            if cfg.max_seconds is not None:
                phase.max_seconds = max(cfg.max_seconds - sum(lg["seconds"] for lg in logs), 0.0)
            logs.append(
                fit_module(
                    net, loss, arrays, val_arrays, phase,
                    seed=_seed(seed, 10 + i), device=self.device, verbose=verbose, val_loss_fn=self._val_loss,
                )  # fmt: skip
            )
        history = [dict(h, ema=lg["config"]["ema_decay"] > 0.0) for lg in logs for h in lg["history"]]
        for epoch, h in enumerate(history):
            h["epoch"] = epoch
        log = dict(logs[-1], config=dataclasses.asdict(cfg), history=history, epochs=len(history))
        log.update(best_epoch=len(history) - logs[-1]["epochs"] + logs[-1]["best_epoch"], ema_warmup_epochs=warm)
        log["seconds"] = sum(lg["seconds"] for lg in logs)
        net.eval()
        self.fitted = True
        val_nll = None
        if val_arrays is not None:
            m = min(VAL_NLL_WINDOWS, len(val))
            idx = np.sort(np.random.default_rng([abs(int(seed)), 3]).choice(len(val), m, replace=False))
            future = None if val.v_lead_future is None else val.v_lead_future[idx]
            dist = self.predict_dist(val.X[idx], v_lead_future=future)
            val_nll = float(-np.mean(dist.log_prob(val.y[idx])))
        log.update(
            n_params=int(sum(p.numel() for p in net.parameters())),
            val_loss=None if val_arrays is None else float(log["best_val"]),
            val_nll=val_nll,
            val_nll_method=f"approximate: KDE of {self.n_samples} first-step samples, <= {VAL_NLL_WINDOWS} windows",
            total_seconds=time.perf_counter() - started,
        )
        return log

    # ----------------------------------------------------------------- prediction
    def sample_sequences(
        self,
        state_history: np.ndarray | torch.Tensor,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """(N, S, L) float32 sampled accelerations [m/s^2] a[t : t+L], clipped to [A_MIN, A_MAX].

        state_history (N, H, 3); v_lead_future (N, L) leader speed [m/s] at t+1 .. t+L (None: constant leader;
        ignored by a model without the leader input). ``generator=None`` draws from ``predict_seed`` (reproducible,
        independent of ``batch_size`` rows per chunk).
        """
        hist = as_history_tensor(state_history, self.device)
        s = self.n_samples if n_samples is None else int(n_samples)
        return self._sample(hist, self._leader(hist, v_lead_future), s, generator, batch_size).cpu().numpy()

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
        target: str = "a",
    ) -> SampleDist:
        """SampleDist (N, n_samples) of the target [m/s^2] from the sampled sequences, seeded by ``predict_seed``.

        ``target="a"``: first-step acceleration a[t]; ``"mean_1s"``: mean of the first ``MEAN_TARGET_STEPS``
        accelerations of every sampled sequence, i.e. ``(v[t+10] - v[t]) / (10 DT)`` of the sampled trajectory
        (design.md, section 11.1; the model is not retrained per target).
        """
        steps = TARGET_SAMPLES.get(target)
        if steps is None:
            raise ValueError(f"DiffusionCF predicts the targets {sorted(TARGET_SAMPLES)}, got {target!r}")
        if steps > self.seq_len:
            raise ValueError(f"target {target!r} needs {steps} sampled steps, the model generates {self.seq_len}")
        seq = self.sample_sequences(state_history, v_lead_future, batch_size=batch_size)
        if target == "a":
            return SampleDist(seq[:, :, 0].astype(np.float64), epistemic_var=None)
        return SampleDist(seq[:, :, :steps].astype(np.float64).mean(axis=2), epistemic_var=None)

    # ------------------------------------------------------------------- rollouts
    def init_sampler(
        self,
        n_traj: int,
        generator: torch.Generator,
        mode: str = "per_step",
        v_lead: np.ndarray | torch.Tensor | None = None,
        preview: PreviewSpec | Mapping[str, Any] | None = None,
    ) -> dict:
        """Rollout state: step counter ``k``, current plan (n_traj, L), leader speeds ``v_lead`` (n_traj, T_total)
        and the ``preview`` spec.

        ``v_lead[:, k]`` is the leader speed [m/s] at rollout step k (step 0 = last history sample of the first
        call); None: constant-leader assumption at every re-plan. ``preview`` (``PreviewSpec`` or its mapping,
        extension X1): the preview handed at every re-plan is truncated to its horizon, perturbed by its noise and
        stale by its delay; None (or an identity spec): the stored behaviour. ``mode`` is validated only (no
        epistemic part). A model without the leader input (DiffusionCFHist) ignores ``v_lead`` and ``preview``
        (stored as None).
        """
        super().init_sampler(n_traj, generator, mode)
        spec = PreviewSpec.from_any(preview)
        if not self.needs_leader_future:
            v_lead, spec = None, None
        if v_lead is not None:
            v_lead = torch.as_tensor(v_lead if torch.is_tensor(v_lead) else np.asarray(v_lead))
            v_lead = v_lead.to(device=self.device, dtype=torch.float32)
            if v_lead.ndim != 2 or v_lead.shape[0] != n_traj or v_lead.shape[1] < 1:
                raise ValueError(f"v_lead must have shape ({n_traj}, T_total), got {tuple(v_lead.shape)}")
        if spec is not None and spec.identity:
            spec = None
        return {"k": 0, "plan": torch.zeros((n_traj, self.seq_len), device=self.device), "v_lead": v_lead,
                "preview": spec}  # fmt: skip

    def _preview_window(
        self, v_lead: torch.Tensor, k: int, spec: PreviewSpec | None, generator: torch.Generator
    ) -> torch.Tensor:
        """(B, L) leader speeds handed to the re-plan at rollout step k: ``v_lead[:, k+1 : k+1+L]`` padded with the
        last available value (the stored behaviour). With a spec the window starts ``delay_steps`` earlier, the
        columns beyond ``horizon_steps`` repeat the last seen value, and i.i.d. Gaussian noise of ``noise_sigma``
        [m/s] is drawn from ``generator`` (nothing is drawn at noise 0). The training budget ``preview_steps``
        applies on top, as in ``_leader``."""
        start = k + 1 - (0 if spec is None else spec.delay_steps)
        cols = torch.arange(start, start + self.seq_len, device=self.device)
        if spec is not None and spec.horizon_steps is not None:
            cols = cols.clamp(max=start - 1 + spec.horizon_steps)
        vlf = v_lead[:, cols.clamp(min=0, max=v_lead.shape[1] - 1)]
        if spec is not None and spec.noise_sigma > 0.0:
            vlf = vlf + spec.noise_sigma * torch.randn(vlf.shape, generator=generator, device=vlf.device, dtype=vlf.dtype)
        return hold_beyond(vlf, self.preview_steps)

    def sample_step(self, hist: torch.Tensor, state: dict, generator: torch.Generator) -> torch.Tensor:
        """(B,) acceleration [m/s^2] at rollout step k: ``plan[:, k % replan_every]``, re-planned when that is 0.

        A re-plan draws one sequence per trajectory given the current ``hist`` (B, H, 3) and the leader speeds
        ``v_lead[:, k+1 : k+1+L]`` (padded with the last available value; see ``_preview_window`` for a preview spec
        and the training budget).
        """
        k = int(state["k"])
        j = k % self.replan_every
        if j == 0:
            h = as_history_tensor(hist, self.device)
            if state["v_lead"] is None:
                vlf = self._leader(h, None)  # None for a model without the leader input
            else:
                vlf = self._preview_window(state["v_lead"], k, state.get("preview"), generator)
            state["plan"] = self._sample(h, vlf, 1, generator, MAX_SEQUENCES)[:, 0]
        state["k"] = k + 1
        return state["plan"][:, j]

    # ---------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        """EMA weights used for sampling plus the normaliser buffers (all tensors)."""
        return {"net": {k: v.detach().cpu().clone() for k, v in self._require_net().state_dict().items()}}

    def set_state(self, state: dict) -> None:
        self.net = Denoiser(**self.arch).to(self.device)
        self.net.load_state_dict(state["net"])
        self.net.eval()
        self.fitted = True


@register("diffusion_cf_hist")
class DiffusionCFHist(DiffusionCF):
    """DiffusionCF conditioned on the 3 s history only (review control R2, docs/design.md 18.1).

    The same DDPM (schedule, denoiser, EMA training, DDIM sampler with ``n_samples`` = 32 and 50 steps, re-planning
    every ``replan_every`` steps) with the leader-future branch of the conditioning removed: the input convolution
    of the denoiser sees the noisy sequence only (``Denoiser(leader=False)``), so the generated 5 s plan depends on
    the history alone, exactly the information of the one-step networks. Removal (not a zeroed channel or a learned
    null token) keeps no unused weights and makes the information budget explicit. ``v_lead_future`` arguments and
    the ``v_lead`` of a rollout are accepted and ignored; it is still a plan sampler (``plan_sampler``: sampled-row
    evaluation in E1, ``n_rollouts_sampled`` rollouts and hold 1 in E2 / E3). Constructor and training budget as
    DiffusionCF (configs/model/diffusion_cf_hist.yaml mirrors configs/model/diffusion_cf.yaml).
    """

    needs_leader_future = False
