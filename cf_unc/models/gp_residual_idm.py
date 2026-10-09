"""Sparse variational GP on the residual of a frozen IDM (docs/design.md, section 7: ``GPResidualIDM``).

Model, with ``x`` the standardised ``(s, dv, v)`` of the LAST history sample (physical units elsewhere)::

    a = a_IDM(s, dv, v) + r_mean + r_std * (f(x) + eps),   f ~ GP(c, k),   eps ~ N(0, noise)
    k(x, x') = outputscale * exp(-|(x - x') / lengthscale|^2 / 2)                      (RBF-ARD)

The residual ``r = a - a_IDM`` (IDM clipped as in ``idm_accel``) is centred by its training mean
``r_mean`` (subtracted before the GP sees it) and scaled by its training std ``r_std``; the GP also
learns a constant mean ``c`` (initialised at 0) in these standardised units, which absorbs whatever
offset the centring leaves. Inputs are standardised with the training mean / std of the last samples.

Training: GPyTorch SVGP in float64 - ``ApproximateGP`` with a whitened ``VariationalStrategy`` over a
``CholeskyVariationalDistribution`` of M inducing values, ``ScaleKernel(RBFKernel(ard_num_dims=3))``,
``GaussianLikelihood``, ``VariationalELBO(num_data=N)``, Adam on minibatches, early stopping on the
validation negative log predictive density (best state kept). Inducing inputs start at k-means++
centres refined by a few Lloyd iterations (``kmeans_inducing``) or at a random subset of the inputs.

After training the posterior is exported to plain float64 tensors (``GPPosterior``); prediction,
sampling, ``save`` and ``load`` use only these::

    A      = L^{-1} K_zx                        (M, B),   L = chol(K_zz + jitter I)
    E[f]   = c + A^T m
    Var[f] = k_xx - |A|^2 + |L_S^T A|^2         k_xx = outputscale + jitter

``L`` is computed like GPyTorch's own factor (``psd_safe_cholesky`` of ``K_zz + jitter I`` with the
strategy's ``jitter_val``), and - as in GPyTorch, which adds the same jitter to the prior covariance of
the test inputs - ``k_xx`` carries the jitter too, so the exported law equals ``model(x)`` /
``likelihood(model(x))`` of the trained GPyTorch objects.
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np
import torch

with warnings.catch_warnings():
    # gpytorch / linear_operator decorate helpers with torch.jit.script, deprecated in recent torch (harmless)
    warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    import gpytorch
    from linear_operator.utils.cholesky import psd_safe_cholesky

from cf_unc.constants import N_FEATURES
from cf_unc.data.windows import WindowSet
from cf_unc.dist import GaussianDist
from cf_unc.models.base import CFModel, register
from cf_unc.physics.idm import IDMParams, idm_accel

__all__ = ["GPPosterior", "GPResidualIDM", "INDUCING_INITS", "export_posterior", "kmeans_inducing"]

INDUCING_INITS = ("kmeans", "random")
_F64 = torch.float64
_CHUNK_ELEMS = 1 << 21  # elements of one (M, chunk) float64 temporary (16 MiB); a chunk holds <= ~5 of them
_LOG_2PI = math.log(2.0 * math.pi)
_MIN_DELTA = 1e-4  # nats per window: smallest validation NLPD improvement that resets the patience


# --------------------------------------------------------------------------- inducing-point initialisation
def kmeans_inducing(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_lloyd: int = 10,
    max_points: int = 20_000,
) -> np.ndarray:
    """``k`` distinct centres (k, D) of the rows of ``x`` (N, D): k-means++ seeding plus Lloyd iterations.

    Runs on a random subsample (``rng``) of at most ``max_points`` distinct rows. Seeds are data points
    drawn with probability proportional to the squared distance to the nearest seed; ``n_lloyd`` Lloyd
    iterations then move every centre to the mean of its cluster (empty clusters keep their centre).
    Every centre is a data point or a cluster mean, hence inside the bounding box of ``x``.
    """
    pts = np.unique(np.asarray(x, dtype=np.float64), axis=0)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError(f"x must be a non-empty (N, D) array, got shape {np.shape(x)}")
    if pts.shape[0] > max_points:
        pts = pts[np.sort(rng.choice(pts.shape[0], size=max_points, replace=False))]
    n, d = pts.shape
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= {n} (distinct points), got k = {k}")
    centres = np.empty((k, d))
    first = int(rng.integers(n))
    centres[0] = pts[first]
    d2 = np.sum((pts - pts[first]) ** 2, axis=1)
    for j in range(1, k):  # k-means++ seeding (inverse cdf of the D^2 weights)
        cum = np.cumsum(d2)
        i = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), n - 1)
        if d2[i] <= 0.0:  # only through rounding: fall back to the farthest point
            i = int(np.argmax(d2))
        centres[j] = pts[i]
        np.minimum(d2, np.sum((pts - pts[i]) ** 2, axis=1), out=d2)
    sq_norm = np.sum(pts * pts, axis=1)
    for _ in range(n_lloyd):  # Lloyd iterations
        dist = sq_norm[:, None] - 2.0 * pts @ centres.T + np.sum(centres * centres, axis=1)[None, :]
        assign = np.argmin(dist, axis=1)
        counts = np.bincount(assign, minlength=k)
        filled = counts > 0
        new = centres.copy()
        for c in range(d):
            new[filled, c] = np.bincount(assign, weights=pts[:, c], minlength=k)[filled] / counts[filled]
        if np.array_equal(new, centres):
            break
        centres = new
    return _make_distinct(centres, pts, rng)


def _make_distinct(centres: np.ndarray, pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace repeated centres (a measure-zero event) by random data points that are not centres yet."""
    _, keep = np.unique(centres, axis=0, return_index=True)
    dup = np.setdiff1d(np.arange(centres.shape[0]), keep)
    if dup.size:
        taken = {tuple(row) for row in centres[keep]}
        fresh = [i for i in rng.permutation(pts.shape[0]) if tuple(pts[i]) not in taken][: dup.size]
        centres = centres.copy()
        centres[dup] = pts[fresh]
    return centres


# --------------------------------------------------------------------------- exported posterior
@dataclass
class GPPosterior:
    """Exported SVGP posterior; float64 tensors in standardised input / residual units.

    Z (M, 3) inducing inputs, lengthscale (3,), outputscale / noise / const / jitter 0-d tensors,
    m (M,) whitened variational mean, L_S (M, M) lower Cholesky factor of the whitened variational
    covariance S, L (M, M) lower Cholesky factor of ``K_zz + jitter I``.
    """

    Z: torch.Tensor
    lengthscale: torch.Tensor
    outputscale: torch.Tensor
    noise: torch.Tensor
    const: torch.Tensor
    jitter: torch.Tensor
    m: torch.Tensor
    L_S: torch.Tensor
    L: torch.Tensor

    @property
    def n_inducing(self) -> int:
        return int(self.Z.shape[0])

    @property
    def prior_var(self) -> torch.Tensor:
        """Prior variance ``k_xx`` of f at any input (0-d), jitter included as in GPyTorch."""
        return self.outputscale + self.jitter

    def to(self, device: torch.device) -> "GPPosterior":
        return GPPosterior(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def state(self) -> dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name).detach().cpu().clone() for f in fields(self)}

    @classmethod
    def from_state(cls, state: Mapping[str, torch.Tensor]) -> "GPPosterior":
        return cls(**{f.name: torch.as_tensor(state[f.name]).to(_F64) for f in fields(cls)})

    def cross_cov(self, xn: torch.Tensor) -> torch.Tensor:
        """``K_zx`` (M, B) for standardised inputs ``xn`` (B, 3) (elementwise, so batch-independent)."""
        zs = self.Z / self.lengthscale
        xs = xn / self.lengthscale
        d2 = torch.zeros((zs.shape[0], xs.shape[0]), dtype=_F64, device=xn.device)
        diff = torch.empty_like(d2)
        for c in range(zs.shape[1]):
            torch.sub(zs[:, c, None], xs[None, :, c], out=diff)
            d2.addcmul_(diff, diff)
        return d2.mul_(-0.5).exp_().mul_(self.outputscale)

    def interp(self, xn: torch.Tensor) -> torch.Tensor:
        """``A = L^{-1} K_zx`` (M, B)."""
        return torch.linalg.solve_triangular(self.L, self.cross_cov(xn), upper=False)

    def moments(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean ``c + A^T m`` and variance ``k_xx - |A|^2 + |L_S^T A|^2`` of f, each (B,), from ``A`` (M, B)."""
        mean = self.const + (a * self.m[:, None]).sum(dim=0)
        var = self.prior_var - a.square().sum(dim=0) + (self.L_S.mT @ a).square_().sum(dim=0)
        return mean, var

    def latent(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance (B,) of the latent f at standardised inputs ``xn`` (B, 3)."""
        return self.moments(self.interp(xn))


def _svgp(inducing: torch.Tensor, learn_inducing: bool, jitter: float | None) -> "gpytorch.models.ApproximateGP":
    """GPyTorch SVGP: whitened VariationalStrategy, Cholesky q(u), constant mean, RBF-ARD kernel (float64)."""

    class _SVGP(gpytorch.models.ApproximateGP):
        def __init__(self) -> None:
            q_u = gpytorch.variational.CholeskyVariationalDistribution(inducing.shape[0])
            strategy = gpytorch.variational.VariationalStrategy(
                self, inducing, q_u, learn_inducing_locations=learn_inducing, jitter_val=jitter
            )
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=N_FEATURES))

        def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    return _SVGP().to(_F64)


@torch.no_grad()
def _gpytorch_nlpd(gp: Any, likelihood: Any, x: torch.Tensor, y: torch.Tensor, chunk: int = 4096) -> float:
    """Mean negative log predictive density of ``y`` under ``likelihood(gp(x))`` (standardised units)."""
    gp.eval()
    likelihood.eval()
    total = 0.0
    for a in range(0, x.shape[0], chunk):
        pred = likelihood(gp(x[a : a + chunk]))
        var = pred.variance
        total += float((0.5 * (_LOG_2PI + var.log()) + 0.5 * (y[a : a + chunk] - pred.mean).square() / var).sum())
    return total / max(int(x.shape[0]), 1)


@torch.no_grad()
def export_posterior(gp: Any, likelihood: Any) -> GPPosterior:
    """Plain-tensor copy of a trained SVGP (see ``GPPosterior``); puts both modules in eval mode."""
    gp.eval()
    likelihood.eval()
    strategy = gp.variational_strategy
    q_u = strategy._variational_distribution
    kernel = gp.covar_module
    z = strategy.inducing_points.detach().to(_F64).clone()
    post = GPPosterior(
        Z=z,
        lengthscale=kernel.base_kernel.lengthscale.detach().to(_F64).reshape(-1).clone(),
        outputscale=kernel.outputscale.detach().to(_F64).reshape(()).clone(),
        noise=likelihood.noise.detach().to(_F64).reshape(()).clone(),
        const=gp.mean_module.constant.detach().to(_F64).reshape(()).clone(),
        jitter=torch.tensor(float(strategy.jitter_val), dtype=_F64, device=z.device),
        m=q_u.variational_mean.detach().to(_F64).clone(),
        L_S=q_u.chol_variational_covar.detach().to(_F64).tril().clone(),  # q_u masks the upper triangle
        L=torch.empty(0, dtype=_F64, device=z.device),
    )
    kzz = post.cross_cov(z)
    kzz.diagonal().add_(post.jitter)
    post.L = psd_safe_cholesky(kzz)  # the routine and jitter of VariationalStrategy._cholesky_factor
    return post


# --------------------------------------------------------------------------- model
@register("gp_residual_idm")
class GPResidualIDM(CFModel):
    """Frozen IDM plus an SVGP on the standardised acceleration residual (see the module docstring).

    ``predict_dist`` returns a ``GaussianDist`` with ``aleatoric_var = r_std^2 noise`` (constant) and
    ``epistemic_var = r_std^2 Var[f(x)]``. Sampler modes: ``"per_trajectory"`` freezes whitened inducing
    values ``v_i = m + L_S xi_i`` per trajectory (joint-posterior approximation, design.md section 7),
    ``"per_step"`` draws independently from the marginal predictive law at every step.
    """

    uses_history = False
    has_epistemic = True

    def __init__(
        self,
        idm: Mapping[str, Any] | IDMParams,
        n_inducing: int = 300,
        inducing_init: str = "kmeans",
        learn_inducing: bool = True,
        lr: float = 0.01,
        max_epochs: int = 30,
        batch_size: int = 2048,
        patience: int = 5,
        max_train: int | None = None,
        jitter: float | None = None,
    ) -> None:
        idm_dict = idm.to_dict() if isinstance(idm, IDMParams) else dict(idm)
        super().__init__(
            idm=idm_dict,
            n_inducing=int(n_inducing),
            inducing_init=str(inducing_init),
            learn_inducing=bool(learn_inducing),
            lr=float(lr),
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            patience=int(patience),
            max_train=None if max_train is None else int(max_train),
            jitter=None if jitter is None else float(jitter),
        )
        self.idm = IDMParams.from_dict(idm_dict)
        if not all(isinstance(getattr(self.idm, name), float) for name in idm_dict):
            raise ValueError("GPResidualIDM needs scalar (population) IDM parameters")
        if inducing_init not in INDUCING_INITS:
            raise ValueError(f"inducing_init must be one of {INDUCING_INITS}, got {inducing_init!r}")
        if int(n_inducing) < 1 or int(max_epochs) < 1 or int(batch_size) < 1 or int(patience) < 1:
            raise ValueError("n_inducing, max_epochs, batch_size and patience must be >= 1")
        if not lr > 0.0:
            raise ValueError("lr must be positive")
        if max_train is not None and int(max_train) < 1:
            raise ValueError("max_train must be None or >= 1")
        if jitter is not None and not jitter > 0.0:
            raise ValueError("jitter must be None or positive")
        self.n_inducing = int(n_inducing)
        self.inducing_init = str(inducing_init)
        self.learn_inducing = bool(learn_inducing)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.max_train = None if max_train is None else int(max_train)
        self.jitter = None if jitter is None else float(jitter)
        self.posterior: GPPosterior | None = None
        self.scalers: dict[str, torch.Tensor] | None = None  # x_mean (3,), x_std (3,), r_mean, r_std (0-d)
        self.gp_model: Any = None  # trained GPyTorch objects (after fit only; never saved)
        self.gp_likelihood: Any = None

    # ---------------------------------------------------------------- features
    def last_states(self, state_history: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Last samples ``(s, dv, v)`` (N, 3) float64 on the model device from histories (N, H, 3) or (H, 3).

        Values are rounded to float32 first (the precision of histories, cf. ``as_history_tensor``), so
        numpy and torch inputs, float32 or float64, give identical features.
        """
        x = state_history if torch.is_tensor(state_history) else np.asarray(state_history)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-1] != N_FEATURES:
            raise ValueError(f"state history must have shape (N, H, 3) or (H, 3), got {tuple(x.shape)}")
        last = x[:, -1, :]
        last = last if torch.is_tensor(last) else torch.as_tensor(np.ascontiguousarray(last))
        return last.to(device=self.device, dtype=torch.float32).to(_F64)

    def standardize(self, last: torch.Tensor) -> torch.Tensor:
        """GP inputs (N, 3): ``(last - x_mean) / x_std``."""
        sc = self._require_scalers()
        return (last - sc["x_mean"]) / sc["x_std"]

    def idm_mean(self, last: torch.Tensor) -> torch.Tensor:
        """``a_IDM(s, dv, v) + r_mean`` (N,) [m/s^2]: the predictive mean where the GP is at its constant."""
        return idm_accel(last[:, 0], last[:, 1], last[:, 2], self.idm) + self._require_scalers()["r_mean"]

    def _require_posterior(self) -> GPPosterior:
        if self.posterior is None:
            raise RuntimeError("GPResidualIDM is not fitted")
        return self.posterior

    def _require_scalers(self) -> dict[str, torch.Tensor]:
        if self.scalers is None:
            raise RuntimeError("GPResidualIDM is not fitted")
        return self.scalers

    def _chunk(self, batch_size: int | None = None) -> int:
        step = max(1, _CHUNK_ELEMS // self._require_posterior().n_inducing)
        return step if batch_size is None else max(1, min(int(batch_size), step))

    # ---------------------------------------------------------------- training
    def _residual_data(self, windows: WindowSet, rows: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Last samples (n, 3) float64 and residuals ``y - a_IDM`` (n,) of the windows ``rows``."""
        last = windows.X[:, -1, :] if rows is None else windows.X[rows, -1, :]
        y = windows.y if rows is None else windows.y[rows]
        x = self.last_states(np.ascontiguousarray(last)[:, None, :])
        r = torch.as_tensor(np.asarray(y, dtype=np.float32)).to(device=self.device, dtype=_F64)
        return x, r - idm_accel(x[:, 0], x[:, 1], x[:, 2], self.idm)

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Train the SVGP on the IDM residuals of ``train`` (early stopping on ``val``) and export it.

        Returns a JSON-serialisable log: ``val_nll`` (mean NLL of ``val.y`` under the exported law,
        physical units), ``n_train``, ``n_inducing``, ``epochs``, ``best_epoch``, ``best_val_nlpd``
        (standardised units), ``seconds``, learned hyperparameters and the per-epoch ``history``.
        """
        started = time.perf_counter()
        if len(train) == 0:
            raise ValueError("empty training set")
        rng = np.random.default_rng(int(seed))
        rows = np.arange(len(train))
        if self.max_train is not None and len(train) > self.max_train:
            rows = np.sort(rng.choice(len(train), size=self.max_train, replace=False))
        last, resid = self._residual_data(train, rows)
        x_std = last.std(dim=0, unbiased=False).clamp_min(1e-6)
        r_std = resid.std(unbiased=False).clamp_min(1e-6)
        self.scalers = {"x_mean": last.mean(dim=0), "x_std": x_std, "r_mean": resid.mean(), "r_std": r_std}
        xn = self.standardize(last)
        yn = (resid - self.scalers["r_mean"]) / r_std
        xv = yv = None
        if val is not None and len(val) > 0:
            last_v, resid_v = self._residual_data(val)
            xv, yv = self.standardize(last_v), (resid_v - self.scalers["r_mean"]) / r_std

        # inducing inputs
        t0 = time.perf_counter()
        pts = np.unique(xn.cpu().numpy(), axis=0)
        n_ind = min(self.n_inducing, pts.shape[0])
        if self.inducing_init == "kmeans":
            z0 = kmeans_inducing(pts, n_ind, rng)
        else:
            z0 = pts[np.sort(rng.choice(pts.shape[0], size=n_ind, replace=False))]
        init_seconds = time.perf_counter() - t0

        forked = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=forked):
            torch.manual_seed(int(seed))  # GPyTorch draws the initial variational mean from the global RNG
            gp = _svgp(torch.as_tensor(z0, dtype=_F64), self.learn_inducing, self.jitter).to(self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=self.device, dtype=_F64)
            history, best_epoch, best_val = self._train(gp, likelihood, xn, yn, xv, yv, int(seed), verbose)
        self.posterior = export_posterior(gp, likelihood)
        self.gp_model, self.gp_likelihood = gp, likelihood
        self.fitted = True

        post, sc = self.posterior, self.scalers
        val_nll = None
        if val is not None and len(val) > 0:
            val_nll = float(-np.mean(self.predict_dist(val.X).log_prob(val.y)))
        r_var = float(sc["r_std"]) ** 2
        return {
            "val_nll": val_nll,
            "n_train": int(rows.size),
            "n_inducing": post.n_inducing,
            "epochs": len(history),
            "best_epoch": best_epoch,
            "best_val_nlpd": best_val,
            "seconds": time.perf_counter() - started,
            "init_seconds": init_seconds,
            "hyper": {
                "lengthscale": post.lengthscale.tolist(),
                "outputscale": float(post.outputscale),
                "noise": float(post.noise),
                "const": float(post.const),
                "jitter": float(post.jitter),
                "r_mean": float(sc["r_mean"]),
                "r_std": float(sc["r_std"]),
                "x_mean": sc["x_mean"].tolist(),
                "x_std": sc["x_std"].tolist(),
                "aleatoric_var": r_var * float(post.noise),
                "prior_epistemic_var": r_var * float(post.prior_var),
            },
            "history": history,
        }

    def _train(
        self,
        gp: Any,
        likelihood: Any,
        xn: torch.Tensor,
        yn: torch.Tensor,
        xv: torch.Tensor | None,
        yv: torch.Tensor | None,
        seed: int,
        verbose: bool,
    ) -> tuple[list[dict], int, float]:
        """Adam on the minibatch ELBO; early stopping on the validation NLPD (training loss without val)."""
        n = int(xn.shape[0])
        elbo = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=n)
        optimizer = torch.optim.Adam([*gp.parameters(), *likelihood.parameters()], lr=self.lr)
        shuffle = torch.Generator(device="cpu").manual_seed(seed)
        history: list[dict] = []
        best_val, best_epoch, best_state = math.inf, -1, None
        for epoch in range(self.max_epochs):
            gp.train()
            likelihood.train()
            perm = torch.randperm(n, generator=shuffle).to(xn.device)
            running = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = -elbo(gp(xn[idx]), yn[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite ELBO at epoch {epoch}")
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * int(idx.numel())
            train_loss = running / n
            score = _gpytorch_nlpd(gp, likelihood, xv, yv) if xv is not None else train_loss
            history.append({"epoch": epoch, "neg_elbo": train_loss, "val_nlpd": score if xv is not None else None})
            if verbose:
                print(f"epoch {epoch:3d}  -elbo {train_loss:.5f}  val nlpd {score:.5f}")
            if score < best_val - _MIN_DELTA:
                best_val, best_epoch = score, epoch
                best_state = (copy.deepcopy(gp.state_dict()), copy.deepcopy(likelihood.state_dict()))
            elif epoch - best_epoch >= self.patience:
                break
        if best_state is not None:
            gp.load_state_dict(best_state[0])
            likelihood.load_state_dict(best_state[1])
        gp.eval()
        likelihood.eval()
        return history, best_epoch, best_val

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> GaussianDist:
        """``GaussianDist`` (N rows, float64, m/s^2) with the variance split into aleatoric and epistemic parts.

        ``mu = a_IDM + r_mean + r_std E[f]``, ``aleatoric_var = r_std^2 noise``, ``epistemic_var =
        r_std^2 Var[f]``, ``sigma = sqrt(aleatoric_var + epistemic_var)``; ``v_lead_future`` is ignored.
        """
        post, sc = self._require_posterior(), self._require_scalers()
        last = self.last_states(state_history)
        n = int(last.shape[0])
        mu = torch.empty(n, dtype=_F64, device=self.device)
        var_f = torch.empty(n, dtype=_F64, device=self.device)
        step = self._chunk(batch_size)
        for a in range(0, n, step):
            chunk = last[a : a + step]
            mean_f, var_chunk = post.latent(self.standardize(chunk))
            mu[a : a + step] = self.idm_mean(chunk) + sc["r_std"] * mean_f
            var_f[a : a + step] = var_chunk
        r_var = sc["r_std"].square()
        epistemic = (r_var * var_f).cpu().numpy()
        aleatoric = np.full(n, float(r_var * post.noise))
        return GaussianDist(
            mu.cpu().numpy(), np.sqrt(aleatoric + epistemic), aleatoric_var=aleatoric, epistemic_var=epistemic
        )

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def init_sampler(self, n_traj: int, generator: torch.Generator, mode: str = "per_step") -> torch.Tensor | None:
        """``None`` for ``"per_step"``; for ``"per_trajectory"`` the frozen whitened inducing values
        ``v_i = m + L_S xi_i``, xi_i ~ N(0, I), as a float64 tensor (n_traj, M) (row i = trajectory i)."""
        super().init_sampler(n_traj, generator, mode)
        post = self._require_posterior()
        if mode == "per_step":
            return None
        n_traj, m = int(n_traj), post.n_inducing
        v = torch.empty((n_traj, m), dtype=_F64, device=self.device)
        step = self._chunk()
        for a in range(0, n_traj, step):
            b = min(n_traj, a + step)
            xi = torch.randn((b - a, m), generator=generator, dtype=_F64, device=self.device)
            v[a:b] = torch.addmm(post.m, xi, post.L_S.mT)
        return v

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: torch.Tensor | None, generator: torch.Generator) -> torch.Tensor:
        """One acceleration draw (B,) [m/s^2] per trajectory from histories ``hist`` (B, H, 3).

        ``state`` from ``init_sampler``: ``None`` = per-step draw from the marginal predictive law;
        (B, M) frozen inducing values = ``f_i = A_i^T v_i + sqrt(max(k_xx - |A_i|^2, 0)) xi'`` and
        ``a_i = a_IDM + r_mean + r_std (f_i + c + sqrt(noise) xi'')``. Draws use only ``generator``;
        rows are processed in chunks (a few (M, chunk) float64 temporaries of 16 MiB). The result has
        the dtype of ``hist`` (float32 for rollout histories), no clipping.
        """
        post, sc = self._require_posterior(), self._require_scalers()
        last = self.last_states(hist)
        b = int(last.shape[0])
        if state is not None and tuple(state.shape) != (b, post.n_inducing):
            raise ValueError(f"sampler state has shape {tuple(state.shape)}, expected ({b}, {post.n_inducing})")
        xi = torch.randn((1 if state is None else 2, b), generator=generator, dtype=_F64, device=self.device)
        out = torch.empty(b, dtype=_F64, device=self.device)
        noise_sd = post.noise.sqrt()
        step = self._chunk()
        for a in range(0, b, step):
            e = min(b, a + step)
            chunk = last[a:e]
            interp = post.interp(self.standardize(chunk))
            if state is None:
                mean_f, var_f = post.moments(interp)
                f_obs = mean_f + (var_f + post.noise).sqrt() * xi[0, a:e]
            else:
                cond_sd = (post.prior_var - interp.square().sum(dim=0)).clamp_min(0.0).sqrt()
                f = torch.einsum("mb,bm->b", interp, state[a:e]) + cond_sd * xi[0, a:e]
                f_obs = f + post.const + noise_sd * xi[1, a:e]
            out[a:e] = self.idm_mean(chunk) + sc["r_std"] * f_obs
        return out.to(hist.dtype if torch.is_floating_point(hist) else torch.float32)

    # --------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        post, sc = self._require_posterior(), self._require_scalers()
        return {"posterior": post.state(), "scalers": {k: v.detach().cpu().clone() for k, v in sc.items()}}

    def set_state(self, state: dict) -> None:
        self.posterior = GPPosterior.from_state(state["posterior"]).to(self.device)
        self.scalers = {k: torch.as_tensor(v).to(device=self.device, dtype=_F64) for k, v in state["scalers"].items()}
        self.gp_model = self.gp_likelihood = None
        self.fitted = True

    def to(self, device: str | torch.device) -> "GPResidualIDM":
        super().to(device)
        if self.posterior is not None:
            self.posterior = self.posterior.to(self.device)
        if self.scalers is not None:
            self.scalers = {k: v.to(self.device) for k, v in self.scalers.items()}
        return self
