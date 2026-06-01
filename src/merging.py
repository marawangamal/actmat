from argparse import Namespace
from pyexpat import model
import os
import sys
import time
from pathlib import Path
import torch
from typing import Callable, Optional, Sequence, Union
from tqdm import tqdm

from src.task_vectors import _TaskVector


def _load_stats_dict(path: str) -> dict:
    """Load a statistics dict from a .pt file or a directory of per-layer .pt files."""
    if os.path.isdir(path):
        result = {}
        for fname in os.listdir(path):
            if fname.endswith(".pt"):
                key = fname[:-3]  # strip .pt
                result[key] = torch.load(
                    os.path.join(path, fname), map_location="cpu", weights_only=False
                )
        return result
    else:
        return torch.load(path, map_location="cpu", weights_only=False)


# Type: (key, [w1, w2, ...]) -> merged tensor
TensorMergeFn = Callable[[str, Sequence[torch.Tensor]], torch.Tensor]


def combine_task_vectors(
    vectors: Sequence[_TaskVector], merge_fn_name: str, **kwargs
) -> _TaskVector:
    """Generic combiner for task vectors.

    Args:
        vectors: list of task vectors (same logical type)
        merge: name of the function defined in this module to merge the task vectors

    Returns:
        A new task vector with the merged task vectors.
    """
    vectors = list(vectors)
    assert len(vectors) > 0, "Need at least one task vector"

    # Get the function (must be defined in this module)
    if merge_fn_name == "mix":
        _primary_fn = getattr(
            sys.modules[__name__], "merge_" + kwargs.pop("mix_primary")
        )
        _fallback_fn = getattr(
            sys.modules[__name__], "merge_" + kwargs.pop("mix_fallback")
        )
        _mix_targets = kwargs.pop("mix_targets", [])
        merge_fn = None  # resolved per-key
    else:
        _mix_targets = None
        merge_fn = getattr(sys.modules[__name__], "merge_" + merge_fn_name)

    # Prefer GPU for merging if available; results are moved back to CPU so they
    # stay compatible with the rest of the pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = vectors[0]
    # Cast others to same type as base
    casted = [base] + [base._cast_to_same_type(v) for v in vectors[1:]]

    start_time = time.time()

    with torch.no_grad():
        new_vector = {}
        keys = casted[0].lazy_keys()
        all_key_sets = [set(v.lazy_keys()) for v in casted]

        ignore_keys = kwargs.pop("ignore_keys", None) or []

        for key in tqdm(keys, desc="Merging task vectors", leave=False):
            if any(key not in ks for ks in all_key_sets):
                # Skip keys that are not present in all vectors
                continue
            if ignore_keys and any(ik in key for ik in ignore_keys):
                # Skip entirely — caller is responsible for filling these keys
                # (typically from pretrained values). Required for the wizardlm
                # benchmark, where embed_tokens/lm_head have differing vocab
                # shapes across experts and cannot be stacked.
                print(f"[ignore_keys] skipping {key}")
                continue
            # Stack on the merge device
            # NOTE: use get_vector_element to speedup lazy mode with caching
            taus = torch.stack([v.get_vector_element(key).to(device) for v in casted])

            if (
                taus[0].ndim == 2
                and "text_projection" not in key
                and max(taus[0].shape) < 20_000
            ):
                # Only matrices can be merged using the merge function
                fn = merge_fn
                if _mix_targets is not None:
                    fn = (
                        _primary_fn
                        if any(t in key for t in _mix_targets)
                        else _fallback_fn
                    )
                print(
                    f"[{key}] merging layer with shape {taus[0].shape} ({fn.__name__})"
                )
                merged = fn(taus, key=key, vectors=vectors, **kwargs)
            else:
                # For all other tensors, we average the values
                merged = taus.mean(dim=0)

            # Keep merged parameters on CPU for compatibility with checkpoint loading.
            # i.e. for when we do `task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)`
            new_vector[key] = merged.to("cpu")

    print(f"Merging task vectors took {(time.time() - start_time) / 60:.2f} minutes")
    return base.__class__(vector=new_vector)


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------


def merge_sum(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    # Shape: (N, Do, Di) -> (Do, Di)
    return taus.sum(dim=0)


def merge_sum04(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    # Shape: (N, Do, Di) -> (Do, Di)
    return taus.sum(dim=0) * 0.4


def merge_mean(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    return taus.mean(dim=0)


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------


def _compute_procrustes(x: torch.Tensor) -> torch.Tensor:
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    return u @ vt


def merge_tsv(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    """Computes the TSV merge of the given tensors.

    Computes: Uo  Dc Vto

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """

    N_tasks = len(taus)
    u, s, vt = torch.linalg.svd(taus, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = _compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = _compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    return tau_l


# ---------------------------------------------------------------------------
# ISOC
# ---------------------------------------------------------------------------
def merge_isoc(taus: torch.Tensor, mode="mean", **kwargs):
    m = taus.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    if mode == "mean":
        s_iso = s.mean() * torch.ones_like(s)
    elif mode == "unity":
        s_iso = torch.ones_like(s)
    elif mode == "rms":
        s_iso = torch.sqrt((s**2).mean()) * torch.ones_like(s)
    elif mode == "spectral":
        s_iso = s[0] * torch.ones_like(s)  # Use largest singular value
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return torch.einsum("ik,k,kj->ij", u, s_iso, vt)


merge_isoc2 = lambda *args, **kwargs: merge_isoc(*args, mode="mean", **kwargs) * 2.0
merge_isoc3 = lambda *args, **kwargs: merge_isoc(*args, mode="mean", **kwargs) * 3.0


# ---------------------------------------------------------------------------
# KNOTS
# ---------------------------------------------------------------------------
def _merge_knots(taus: torch.Tensor, merge_fn: Callable, **kwargs) -> torch.Tensor:
    # print device
    N, Do, Di = taus.shape
    d = taus.permute(1, 0, 2).reshape(Do, N * Di)
    # [Do, R], [R, R], [R, N*Di]
    u, s, vt = torch.linalg.svd(d, full_matrices=False)
    # taus_tilde = torch.einsum("ij,jnk->nik", torch.diag(s), vt.reshape(-1, N, Di))
    # tau_tilde = merge_fn(taus_tilde, **kwargs)
    # return u @ tau_tilde
    # [R, N*Di] -> [N, R, Di] -> [R, Di]
    tau_tilde = merge_fn(vt.reshape(-1, N, Di).permute(1, 0, 2), **kwargs)
    return torch.einsum("or,r,ri->oi", u, s, tau_tilde)


merge_knots_isoc = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc, **kwargs
)
merge_knots_tsv = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_tsv, **kwargs
)


pinv = torch.linalg.pinv


# ---------------------------------------------------------------------------
# RegMean
# ---------------------------------------------------------------------------


def merge_regmean(
    taus: torch.Tensor,  # Shape: (N, Do, Di)
    key: str,
    vectors: Sequence[_TaskVector],
    scale_coef=None,
    max_dim=None,  # default to mean for dim(x) > max_dim
    sample_cov=False,  # sample covariance instead of population covariance
    merge_mode="d",
    **kwargs,
):
    c = []
    for v in vectors:
        km = v.param_key_to_cov_key(key)
        cpath = v.covariance_path
        if cpath is None:
            raise ValueError(f"No covariance path provided for task vector {v}")
        cdict = _load_stats_dict(cpath)
        if km not in cdict:
            print(f"[skipped] {km} not found in {cpath}")
            return taus.mean(dim=0)
        ct = cdict[km]
        if not isinstance(ct, torch.Tensor):
            ct = torch.as_tensor(ct)
        if max_dim is not None and ct.shape[1] > max_dim:
            print(f"[skipped] {km} has shape {ct.shape} > {max_dim}")
            return taus.mean(dim=0)
        if sample_cov:
            n = cdict[f"{km}_n"]
            if isinstance(n, torch.Tensor):
                n = n.item()
            ct = (n * ct) / (n - 1)
        c.append(ct)
    c = torch.stack([x.to(device=taus.device, dtype=taus.dtype) for x in c])

    if scale_coef is not None:
        m_diag = (
            torch.eye(c.shape[1], device=c.device, dtype=c.dtype)
            .unsqueeze(0)
            .expand(c.shape[0], -1, -1)
        )
        c = scale_coef * c + (1 - scale_coef) * m_diag * c

    if merge_mode == "w":
        w0 = vectors[0].pt_vector[key].to(device=taus.device, dtype=taus.dtype)
        ws = taus + w0.unsqueeze(0)
        return ((ws @ c).sum(dim=0) @ pinv(c.sum(dim=0))) - w0

    return (taus @ c).sum(dim=0) @ pinv(c.sum(dim=0))


# ---------------------------------------------------------------------------
# Fisher Merging
# ---------------------------------------------------------------------------
def _dinv(x, thresh=1e-8):
    return torch.where(x.abs() > thresh, 1 / x, 0)


def merge_fisher(
    tau: torch.Tensor,
    key: str,
    vectors: Sequence[_TaskVector],
    **kwargs,
):
    N, Do, Di = tau.shape
    f = []
    for v in vectors:
        km = v.param_key_to_cov_key(key)
        fpath = v.fisher_path
        if fpath is None:
            raise ValueError(f"No fisher path provided for task vector {v}")
        fdict = _load_stats_dict(fpath)
        if km not in fdict:
            print(f"[skipped] {km} not found in {fpath}")
            return tau.mean(dim=0)
        ft = fdict[km]
        if not isinstance(ft, torch.Tensor):
            ft = torch.as_tensor(ft)
        f.append(ft)

    # Shape: (N, Do*Di)
    f = torch.stack([x.reshape(-1).to(device=tau.device, dtype=tau.dtype) for x in f])
    return (_dinv(f.sum(dim=0)) * (f * tau.reshape(N, Do * Di)).sum(dim=0)).reshape(
        Do, Di
    )


# ---------------------------------------------------------------------------
# ACTMat
# ---------------------------------------------------------------------------
def merge_actmat(d: torch.Tensor, *args, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_actmat_herm(d: torch.Tensor, *args, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0), hermitian=True)


def merge_actmat_herm_10ki(d: torch.Tensor, *args, **kwargs):
    if d.shape[-1] > 10_000:
        return d.mean(dim=0)
    return merge_actmat_herm(d)


def merge_actmat_double(d: torch.Tensor, *args, **kwargs):
    """ACTMat solved entirely in float64, result cast back to the input dtype.

    Identical objective/closed form to :func:`merge_actmat`, but the covariance
    accumulation, projection, and pinv solve all run in double precision. The
    estimator c = Σ_t d_tᵀd_t and the pinv are sensitive to round-off on large,
    ill-conditioned layers (e.g. 13B models loaded in bf16); fp64 keeps the
    solve numerically stable.
    """
    orig_dtype = d.dtype
    d = d.double()
    c = d.transpose(1, 2) @ d
    out = (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))
    return out.to(orig_dtype)


def merge_actmat_10k(d: torch.Tensor, *args, dim_threshold: int = 10_000, **kwargs):
    """ACTMat on narrow layers, plain mean on any layer with a dim > dim_threshold.

    The closed-form ACTMat solve is O(Di³) in the input dimension and is most
    ill-conditioned on the widest layers (e.g. the 13824-wide MLP projections of
    Llama-2-13B). Falling back to a simple mean there skips the expensive/unstable
    pinv while keeping the ACTMat solve on the narrower attention/projection
    layers (≤ dim_threshold on both dims).
    """
    if max(d.shape[-2], d.shape[-1]) > dim_threshold:
        return d.mean(dim=0)
    return merge_actmat(d)


def merge_actmat_10ki(d: torch.Tensor, *args, dim_threshold: int = 10_000, **kwargs):
    """ACTMat, but mean-merge only layers whose INPUT dim exceeds dim_threshold.

    delta d is (T, Do, Di). The ACTMat covariance c = dᵀd and its pinv are
    (Di, Di), so both the O(Di³) cost and the conditioning scale with the input
    dimension Di alone — not the output dim Do. A wide-output/narrow-input layer
    (e.g. gate/up_proj, 13824×5120) is therefore still cheap and well-posed and
    keeps the ACTMat solve; only wide-input layers (e.g. down_proj, 5120×13824,
    Di > dim_threshold) fall back to mean.
    """
    if d.shape[-1] > dim_threshold:
        return d.mean(dim=0)
    return merge_actmat(d)


def merge_actmat_norm_weight(d: torch.Tensor, *args, **kwargs):
    """ACTMat with the merge *target* (weight projection) per-task normalized.

    Estimator stays unnormalized (c = d_tᵀd_t), but the RHS uses the Frobenius-
    normalized dn_t = d_t/‖d_t‖_F. With all task norms equal to μ this reduces
    to (1/μ)·merge_actmat(d).
    """
    c = d.transpose(1, 2) @ d
    dn = d / d.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    c_sum = c.sum(dim=0)
    # fp64 promotion: c_sum can be ill-conditioned where fp32 cusolver SVD fails.
    return (dn @ c).sum(dim=0) @ pinv(c_sum.double()).to(c_sum.dtype)


def merge_actmat_norm_estimator(d: torch.Tensor, *args, **kwargs):
    """ACTMat with the *estimator* built from per-task normalized dn, target on raw d.

    c_t = dn_tᵀdn_t = ‖d_t‖⁻²·d_tᵀd_t, target uses the unnormalized d. This is
    exactly merge_actmat_p(p=1) ≡ merge_actmat_mons (the per-task ‖d_t‖⁻² weighting
    matches; any common scalar cancels in the pinv solve).
    """
    dn = d / d.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    c = dn.transpose(1, 2) @ dn
    c_sum = c.sum(dim=0)
    return (d @ c).sum(dim=0) @ pinv(c_sum.double()).to(c_sum.dtype)


def merge_actmat_norm_weight_and_estimator(d: torch.Tensor, *args, **kwargs):
    """ACTMat with BOTH the estimator and the target per-task normalized.

    dn_t = d_t/‖d_t‖_F; c_t = dn_tᵀdn_t; target uses dn. Same norm-equalizing
    estimator as merge_actmat_mons, but the projection is normalized too — so
    with all task norms equal to μ this reduces to (1/μ)·merge_actmat(d).
    """
    dn = d / d.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    c = dn.transpose(1, 2) @ dn
    c_sum = c.sum(dim=0)
    return (dn @ c).sum(dim=0) @ pinv(c_sum.double()).to(c_sum.dtype)


def merge_actmat_l1(
    d: torch.Tensor,
    lr: float = 1e-5,
    max_iters: int = 3000,
    eps: float = 1e-8,
    thresh: float = 1e-6,
    patience: int = 10,
    **kwargs,
) -> torch.Tensor:
    """L1-residual ACTMat: minimize Σ_t E_z ||Δz - Δ_t z||_1 with C_t = Δ_tᵀΔ_t.

    Under Gaussian z, the per-task expectation equals (up to √(2/π)):
        Σ_i ||Δ_t (Δ - Δ_t)_iᵀ||_2  =  ||Δ_t (Δ - Δ_t)ᵀ||_{2,1}

    Smoothed with ε under the sqrt to keep gradients finite at zero residual.
    Exits early when relative loss change < `thresh` for `patience` consecutive steps.
    """
    with torch.enable_grad():
        delta = d.mean(dim=0).clone().requires_grad_(True)  # (Do, Di)
        opt = torch.optim.Adam([delta], lr=lr)

        prev_loss = float("inf")
        n_stable = 0
        pbar = tqdm(range(max_iters), desc="ACTMat-L1", leave=False)
        for i in pbar:
            opt.zero_grad()
            resid = delta.unsqueeze(0) - d  # (T, Do, Di)
            mapped = d @ resid.transpose(-1, -2)  # (T, Do, Do)
            col_norms = mapped.pow(2).sum(dim=-2).add(eps).sqrt()  # (T, Do)
            loss = col_norms.sum()
            loss.backward()
            opt.step()

            cur_loss = loss.item()
            rel = abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12)
            n_stable = n_stable + 1 if rel < thresh else 0
            pbar.set_postfix(loss=cur_loss, rel=rel)
            if n_stable >= patience:
                print(f"[converged] iter={i} loss={cur_loss:.4e} rel={rel:.1e}")
                break
            prev_loss = cur_loss
        else:
            print(f"[not converged] loss={cur_loss:.4e} after {max_iters} iters")

    return delta.detach()


def merge_actmat_isoc(d: torch.Tensor, *args, **kwargs):
    """ACTMat on iso-spectrum task vectors: SVD each d_t, replace its singular
    values with the per-position mean across tasks (isotropy à la IsoC), then
    run vanilla ACTMat on the reconstructed d_tilde_t = U_t diag(s_iso) V_t^T."""
    u, s, vt = torch.linalg.svd(d, full_matrices=False)  # (T,Do,r), (T,r), (T,r,Di)
    s_iso = s.mean(dim=0).unsqueeze(0).expand_as(s)  # (T,r) — same spectrum every task
    dtilde = torch.einsum("tik,tk,tkj->tij", u, s_iso, vt)  # (T,Do,Di)
    c = dtilde.transpose(1, 2) @ dtilde
    return (dtilde @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_actmat_norm_softmax_bias(d: torch.Tensor, *args, **kwargs):
    """Per-task Frobenius-normalized ACTMat with softmax-biased C_t blend.

    dn_t = d_t / ‖d_t‖_F; C_t = α_t·(d_tᵀd_t) + (1−α_t)·I with
    α_t = 1 − softmax(‖d_t‖)_t. Merge basis stays in the original d-space
    (kernel uses unnormalized d), but each task's RHS contribution is unit-norm.
    """
    mags = d.norm(dim=(-2, -1))
    alpha = 1 - mags.softmax(dim=0)  # (T,)
    cov = d.transpose(1, 2) @ d  # (T, Di, Di)
    eye = torch.eye(d.shape[-1], device=d.device, dtype=d.dtype).expand_as(cov)
    a = alpha.view(-1, 1, 1)
    c = a * cov + (1 - a) * eye
    dn = d / mags.view(-1, 1, 1).clamp_min(1e-12)
    return (dn @ c).sum(dim=0) @ pinv(c.sum(dim=0).double()).to(c.dtype)


def _get_actmat_cov(d: torch.Tensor, mode="standard"):
    if mode == "standard":
        return d.transpose(1, 2) @ d
    elif mode == "softmax_bias":
        mags = d.norm(dim=(-2, -1))
        alpha = 1 - mags.softmax(dim=0)  # (T,)
        cov = d.transpose(1, 2) @ d  # (T, Di, Di)
        eye = torch.eye(d.shape[-1], device=d.device, dtype=d.dtype).expand_as(cov)
        a = alpha.view(-1, 1, 1)
        return a * cov + (1 - a) * eye
    elif mode == "softmax_bias_noident":
        mags = d.norm(dim=(-2, -1))
        alpha = 1 - mags.softmax(dim=0)  # (T,)
        cov = d.transpose(1, 2) @ d  # (T, Di, Di)
        a = alpha.view(-1, 1, 1)
        return a * cov
    else:
        raise ValueError(f"Unsupported actmat cov mode {mode}")


def merge_actmat_p(d: torch.Tensor, *args, p=1.0, **kwargs):
    # γ_t = 1/‖d_t‖^(2p). p=0 → vanilla actmat; p=1/2 → smons; p=1 → mons
    # (up to a scalar μ^(2p) that cancels in the pinv solve).
    mags = d.norm(dim=(-2, -1))
    # Zero-delta tasks (e.g. LoRA-frozen params) would give inf*0 → NaN under
    # the rescale; leave them at zero so they contribute nothing.
    scale = torch.where(mags > 0, mags.clamp_min(1e-12).pow(-2 * p), mags.new_zeros(()))
    c = (d.transpose(1, 2) @ d) * scale.view(-1, 1, 1)
    c_sum = c.sum(dim=0)
    # fp64 promotion: rescaled C can be ill-conditioned where fp32 cusolver SVD
    # fails to converge.
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return (d @ c).sum(dim=0) @ c_sum_pinv


merge_actmat_p05 = lambda *a, **kw: merge_actmat_p(*a, p=0.5, **kw)
merge_actmat_p025 = lambda *a, **kw: merge_actmat_p(*a, p=0.25, **kw)
merge_actmat_p09 = lambda *a, **kw: merge_actmat_p(*a, p=0.9, **kw)
merge_actmat_p03 = lambda *a, **kw: merge_actmat_p(*a, p=0.3, **kw)
merge_actmat_p02 = lambda *a, **kw: merge_actmat_p(*a, p=0.2, **kw)
merge_actmat_p01 = lambda *a, **kw: merge_actmat_p(*a, p=0.1, **kw)
merge_actmat_p001 = lambda *a, **kw: merge_actmat_p(*a, p=0.01, **kw)


def merge_actmat_softmax_bias(d: torch.Tensor, *args, **kwargs):
    """ACTMat ↔ mean interpolation with softmax-biased per-task strength.

    Per task: C_t = α_t · (d_tᵀ d_t) + (1 − α_t) · I, with
    α_t = 1 − softmax(‖d_t‖)_t. The dominant-norm task gets α ≈ 0, so its
    C_t ≈ I (uniform weighting → its contribution becomes plain d_t instead of
    the covariance-weighted version). Smaller-norm tasks keep α ≈ 1 and behave
    like vanilla ACTMat.

    Edge cases (all-α=0 → simple mean of d; all-α=1 → vanilla ACTMat) are
    exercised in scripts/tests/test_merging.py.
    """
    mags = d.norm(dim=(-2, -1))
    alpha = 1 - mags.softmax(dim=0)  # (T,)
    cov = d.transpose(1, 2) @ d  # (T, Di, Di)
    eye = torch.eye(d.shape[-1], device=d.device, dtype=d.dtype).expand_as(cov)
    a = alpha.view(-1, 1, 1)
    c = a * cov + (1 - a) * eye
    c_sum = c.sum(dim=0)
    # fp64 promotion: rescaled C can be ill-conditioned where fp32 cusolver SVD
    # fails to converge.
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return (d @ c).sum(dim=0) @ c_sum_pinv


def _excess_ridge_merge(d: torch.Tensor, threshold) -> torch.Tensor:
    """Helper: ridge tasks where ‖d_t‖ > threshold via α_t = min(1, threshold/‖d_t‖)."""
    mags = d.norm(dim=(-2, -1))
    alpha = (threshold / mags.clamp_min(1e-12)).clamp(max=1.0)
    cov = d.transpose(1, 2) @ d
    eye = torch.eye(d.shape[-1], device=d.device, dtype=d.dtype).expand_as(cov)
    a = alpha.view(-1, 1, 1)
    c = a * cov + (1 - a) * eye
    c_sum = c.sum(dim=0)
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return (d @ c).sum(dim=0) @ c_sum_pinv


def merge_actmat_excess_ridge(d: torch.Tensor, *args, **kwargs):
    """ACTMat with identity-blend ridge applied ONLY to above-mean-norm tasks.

    α_t = min(1, μ / ‖d_t‖); C_t = α_t · (d_tᵀ d_t) + (1−α_t) · I.

    Tasks at or below the mean norm get α=1 (pure ACTMat — no ridge).
    Tasks above get α = μ/‖d_t‖ < 1 → partial identity ridge proportional to
    their excess. When all norms are nearly equal (FFT-like regime), α ≈ 1
    everywhere → reduces to vanilla ACTMat (so it should not regress FFT).
    When one task has a wildly larger norm (T5 LoRA layers), that task gets a
    significant ridge → softer than mons (which would rescale it) and softer
    than softmax_bias (which uniformly ridges every task).
    """
    return _excess_ridge_merge(d, d.norm(dim=(-2, -1)).mean())


def merge_actmat_excess_ridge_2x(d: torch.Tensor, *args, **kwargs):
    """As excess_ridge but threshold = 2μ: only tasks >2× mean norm get ridge."""
    return _excess_ridge_merge(d, 2 * d.norm(dim=(-2, -1)).mean())


def merge_actmat_excess_ridge_med(d: torch.Tensor, *args, **kwargs):
    """As excess_ridge but threshold = median (robust to outliers in μ)."""
    return _excess_ridge_merge(d, d.norm(dim=(-2, -1)).median())


def merge_actmat_uniform_ridge(d: torch.Tensor, *args, ridge=0.01, **kwargs):
    """Vanilla ACTMat with a tiny uniform identity ridge on c_sum.

    Baseline check: if simply adding λ·I before the pinv beats vanilla ACTMat,
    then the win is just regularization, not per-task weighting.
    """
    c = d.transpose(1, 2) @ d
    c_sum = c.sum(dim=0)
    lam = ridge * c_sum.diagonal().abs().mean()
    eye = torch.eye(c_sum.shape[0], device=c_sum.device, dtype=c_sum.dtype)
    c_reg = (c_sum + lam * eye).double()
    return (d @ c).sum(dim=0) @ pinv(c_reg).to(c_sum.dtype)


def merge_actmat_softmax_bias_solve(
    d: torch.Tensor, *args, jitter: float = 1e-6, **kwargs
):
    """Same as [[merge_actmat_softmax_bias]] but solves with LU on fp32 + jitter.

    The closed-form `(d @ c).sum(0) @ pinv(c_sum)` is replaced with
    `solve(c_sum, RHS.T).T`, which avoids SVD. With α_t = 1 − softmax(‖d_t‖),
    c_sum = Σ α_t · d_tᵀd_t + I is symmetric PD, so a small ridge `jitter·avg(diag)`
    plus LU solve is numerically safe and ~50–100× faster than the fp64 pinv on
    5120×5120 matrices. Result is the same closed-form merge.
    """
    mags = d.norm(dim=(-2, -1))
    alpha = 1 - mags.softmax(dim=0)
    cov = d.transpose(1, 2) @ d
    eye = torch.eye(d.shape[-1], device=d.device, dtype=d.dtype).expand_as(cov)
    a = alpha.view(-1, 1, 1)
    c = a * cov + (1 - a) * eye
    c_sum = c.sum(dim=0)
    rhs = (d @ c).sum(dim=0)  # (Do, Di)
    lam = jitter * c_sum.diagonal().abs().mean()
    c_sum = c_sum + lam * torch.eye(
        c_sum.shape[0], device=c_sum.device, dtype=c_sum.dtype
    )
    return torch.linalg.solve(c_sum, rhs.transpose(-1, -2)).transpose(-1, -2)


def merge_actmat_softmax_bias_noident(d: torch.Tensor, *args, **kwargs):
    """ACTMat with softmax-biased per-task weighting (no identity shrinkage).

    C_t = α_t · (d_tᵀ d_t), with α_t = 1 − softmax(‖d_t‖)_t. The dominant-norm
    task gets α ≈ 0 → it drops out of the merge entirely. Smaller-norm tasks
    keep α ≈ 1 (vanilla ACTMat). When the softmax saturates (wide-spread norms,
    as in T5 LoRA) this is effectively a leave-one-out merge that excludes the
    dominant task; when norms are flat (FFT) the per-task α scalars are nearly
    uniform and cancel in the pinv solve → close to vanilla ACTMat.
    """
    mags = d.norm(dim=(-2, -1))
    alpha = 1 - mags.softmax(dim=0)  # (T,)
    c = (d.transpose(1, 2) @ d) * alpha.view(-1, 1, 1)
    c_sum = c.sum(dim=0)
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return (d @ c).sum(dim=0) @ c_sum_pinv


# # NOTE: scalar gamma cancels in the pinv solve, so this variant is mathematically
# # equivalent to vanilla merge_actmat — kept as a sanity-check baseline.
# actmat_sgeo05_nnone = lambda d, *a, **kw: merge_actmat_cscale(
#     d, *a, gamma=d.norm(dim=(-2, -1)).pow(0.5).mean(), **kw
# )

# actmat_sgeo05_np1 = lambda d, *a, **kw: merge_actmat_cscale(
#     d,
#     *a,
#     gamma=d.norm(dim=(-2, -1)).pow(0.5).mean() / d.norm(dim=(-2, -1), keepdim=True),
#     **kw,
# )

# actmat_sgeo05_np2 = lambda d, *a, **kw: merge_actmat_cscale(
#     d,
#     *a,
#     gamma=d.norm(dim=(-2, -1)).pow(0.5).mean()
#     / d.norm(dim=(-2, -1), keepdim=True).pow(2),
#     **kw,
# )

# actmat_sinv_nnone = lambda d, *a, **kw: merge_actmat_cscale(
#     d,
#     *a,
#     gamma=(1 - d.norm(dim=(-2, -1)).softmax(dim=0)).view(-1, 1, 1),
#     **kw,
# )


def merge_actmat_mons(d: torch.Tensor, *args, **kwargs):
    mags = d.norm(dim=(-2, -1))
    mu_mag = mags.mean()
    # Zero-delta tasks (e.g. a LoRA-frozen param) would give 0/0 → NaN under the
    # rescale; leave them at zero so they contribute nothing to C / target.
    scale = torch.where(mags > 0, mu_mag / mags.clamp_min(1e-12), mags.new_zeros(()))
    d_tilde = d * scale.view(-1, 1, 1)
    c = d_tilde.transpose(1, 2) @ d_tilde
    c_sum = c.sum(dim=0)
    target = (d @ c).sum(dim=0)
    # Norm-equalizing rescale can produce an ill-conditioned C where the
    # cusolver SVD fails to converge in fp32; promote to fp64 for the pinv.
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return target @ c_sum_pinv


def merge_actmat_smons(d: torch.Tensor, *args, **kwargs):
    mu_mag = d.norm(dim=(-2, -1)).mean()
    d_tilde = d * mu_mag / d.norm(dim=(-2, -1), keepdim=True)
    c = d_tilde.transpose(1, 2) @ d_tilde
    c_sum = c.sum(dim=0)
    target = (d @ c).sum(dim=0)
    # Norm-equalizing rescale can produce an ill-conditioned C where the
    # cusolver SVD fails to converge in fp32; promote to fp64 for the pinv.
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return target @ c_sum_pinv


def merge_actmat_selective(
    d: torch.Tensor,
    *args,
    top_frac: float = 1.0 / 3.0,
    top_k: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """ACTMat with C_t = I for the top-K highest-norm experts.

    Vanilla ACTMat weights each task by C_t = d_tᵀd_t, so per-task weight
    scales as ‖d_t‖². When one expert's delta dominates (e.g. WizardLM's
    LM > Math >> Code: 32× spread), Σ C_t ≈ C_dominant and the closed-form
    solve collapses to ≈ d_dominant. Replacing the top-K experts' C with
    identity removes their structural dominance over Σ C_t and lets the
    smaller-norm experts contribute their data-aware shaping.

    Args:
        top_frac: fraction of experts (by ‖d_t‖) to give C=I. Default 1/3.
        top_k: absolute count; overrides top_frac if set.
    """
    T = d.shape[0]
    Di = d.shape[-1]
    mags = d.norm(dim=(-2, -1))
    if top_k is None:
        top_k = max(1, int(round(top_frac * T)))
    top_k = min(top_k, T)
    _, top_idx = torch.topk(mags, top_k)

    c = d.transpose(1, 2) @ d  # (T, Di, Di)
    eye = torch.eye(Di, device=d.device, dtype=d.dtype)
    for t in top_idx.tolist():
        c[t] = eye
    c_sum = c.sum(dim=0)
    # Identity contribution can mix scales drastically with d_tᵀd_t terms;
    # promote to fp64 to keep the pinv stable.
    c_sum_pinv = pinv(c_sum.double()).to(c_sum.dtype)
    return (d @ c).sum(dim=0) @ c_sum_pinv


def merge_actmat_5k(d: torch.Tensor, *args, **kwargs):
    if d.shape[-1] > 5_000:
        return d.mean(dim=0)
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


merge_actmat_gd_5k = lambda d, **kwargs: (
    merge_actmat_gd(d, **kwargs) if d.shape[-1] <= 5_000 else d.mean(0)
)


def merge_actmat_gd_softmax_bias(
    d: torch.Tensor,
    lam=0.0,
    lr=1e-5,
    max_iters=300,
    thresh=-float("inf"),
    **kwargs,
) -> torch.Tensor:
    """GD solver for the ACTMat objective with softmax-biased per-task C_t.

    Per task: C_t = α_t · (d_tᵀ d_t) + (1 − α_t) · I, with α_t = 1 − softmax(‖d_t‖)_t
    (same per-task ridge schedule as [[merge_actmat_softmax_bias]]). The dominant-
    norm task gets α ≈ 0 → its C_t ≈ I (uniform), so its loss contribution behaves
    like plain mean-fitting. Smaller-norm tasks keep α ≈ 1 (vanilla ACTMat).

    Solves Adam on the same loss as merge_actmat_gd, but with the rescaled C_t:
        L(W) = Σ_t tr((W - d_t) C_t (W - d_t)^T) + λ ‖W‖_F²
    """
    mags = d.norm(dim=(-2, -1))
    alpha = 1 - mags.softmax(dim=0)  # (T,)
    cov = d.transpose(1, 2) @ d  # (T, Di, Di)
    eye = torch.eye(d.shape[-1], device=d.device, dtype=d.dtype).expand_as(cov)
    a = alpha.view(-1, 1, 1)
    C = a * cov + (1 - a) * eye

    W = d.mean(dim=0).clone().requires_grad_(True)  # (Do, Di)
    optimizer = torch.optim.Adam([W], lr=lr, weight_decay=0.0)

    cur_loss = float("inf")
    with torch.enable_grad():
        prev_loss = float("inf")
        pbar = tqdm(range(int(max_iters)), desc="Gradient descent", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            diff = W.unsqueeze(0) - d  # (T, Do, Di)
            loss = (diff @ C).mul_(diff).sum()
            if lam > 0:
                loss = loss + lam * W.square().sum()
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            if abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12) < thresh:
                print(f"[converged] loss={cur_loss:.1e} < {thresh:.1e}")
                break
            prev_loss = cur_loss
            pbar.set_postfix(loss=cur_loss)

    if i == int(max_iters) - 1:
        print(f"[not converged] loss={cur_loss:.1e} after {max_iters} iters")

    return W.detach()


def merge_actmat_gd(
    d: torch.Tensor,
    lam=0.0,
    alpha_weighted=False,
    cov_weighted=False,
    lr=1e-5,
    max_iters=300,
    thresh=-float("inf"),
    **kwargs,
) -> torch.Tensor:
    """Gradient-descent solver for the ACTMat objective.

    Minimizes the same weighted least-squares loss as merge_actmat_general:
        L(W) = Σ_t tr((W - d_t) C_t (W - d_t)^T) + λ ‖W‖_F²
    where C_t = d_t^T @ d_t.
    """
    C = d.transpose(1, 2) @ d  # (T, Di, Di)

    if cov_weighted:
        C = C / (torch.linalg.norm(C, ord="fro", dim=(-2, -1), keepdim=True) ** 2)

    if alpha_weighted:
        alpha = 1.0 / d.flatten(1).norm(dim=1)  # (T,)
        C = alpha[:, None, None] * C  # (T, Di, Di)

    W = d.mean(dim=0).clone().requires_grad_(True)  # (Do, Di)
    optimizer = torch.optim.Adam([W], lr=lr, weight_decay=0.0)

    # Re-enable gradients inside the torch.no_grad() context of combine_task_vectors
    with torch.enable_grad():
        prev_loss = float("inf")
        pbar = tqdm(range(int(max_iters)), desc="Gradient descent", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            diff = W.unsqueeze(0) - d  # (T, Do, Di)
            loss = (diff @ C).mul_(diff).sum()
            if lam > 0:
                loss = loss + lam * W.square().sum()
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            if abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12) < thresh:
                print(f"[converged] loss={cur_loss:.1e} < {thresh:.1e}")
                break
            prev_loss = cur_loss
            pbar.set_postfix(loss=cur_loss)

    if i == int(max_iters) - 1:
        print(f"[not converged] loss={cur_loss:.1e} after {max_iters} iters")

    return W.detach()


def _per_layer_topk_mask(d: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """Per-layer top-k magnitude mask over (N, Do, Di) stack of task vectors."""
    n, do, di = d.shape
    numel = do * di
    n_keep = max(1, int(round(keep_frac * numel)))
    if n_keep >= numel:
        return torch.ones_like(d, dtype=torch.bool)
    flat = d.abs().reshape(n, numel)
    _, idx = flat.topk(n_keep, dim=1)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape(n, do, di)


def merge_tact(d: torch.Tensor, tact_k: float = 0.5, **kwargs):
    """TACT: ACTMat with a per-layer magnitude-trimmed covariance estimator.

    Builds C_t = trim(d_t)^T @ trim(d_t) from the top-tact_k fraction of |d_t|
    entries (per layer), but keeps the full d_t in the merge target. Implements
    Algorithm 1 of the TACT paper with the per-layer trim variant (Appendix C).
    """
    mask = _per_layer_topk_mask(d, tact_k)
    d_tilde = d * mask
    c = d_tilde.transpose(1, 2) @ d_tilde
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


# ---------------------------------------------------------------------------
# ACE-Merging (https://arxiv.org/abs/2603.02945)
# Reference: https://github.com/unravel-xu/ACE-Merging/blob/main/src/merge/strategy.py
# ---------------------------------------------------------------------------
def merge_ace(
    d: torch.Tensor,
    ace_eps: float = 1e-5,
    ace_k_frac: float = 0.3,
    **kwargs,
) -> torch.Tensor:
    """ACE-Merging: adaptive covariance estimation, data-free.

    Per-layer closed-form merge of task-vector stack ``d`` of shape ``(N, Do, Di)``:
      1. For each W_t, center column-wise and build Σ_t = W_t^T W_t.
      2. Detect heterogeneity via γ = Var(log tr Σ_t) / Mean(log tr Σ_t)^2;
         if γ > 0.3, normalize each Σ_t by its trace (and scale ε accordingly).
      3. Closed form: W_0 = (Σ_t W_t (Σ_t + εI)) @ inv(Σ_t (Σ_t + εI) + C_agg),
         where C_agg is a column-mean broadcast term over Σ_t.
      4. Under heterogeneity, add a low-rank residual-fusion term reconstructed
         from the top ``ace_k_frac`` singular components.
    """
    N, Do, Di = d.shape
    device, dtype = d.device, d.dtype

    traces = torch.stack([torch.trace(W_t.T @ W_t) for W_t in d])
    log_traces = torch.log(traces + 1e-12)
    gamma = torch.var(log_traces) / (torch.mean(log_traces).pow(2) + 1e-12)
    flag = bool(gamma > 0.3)
    avg_trace = traces.mean()

    eye_di = torch.eye(Di, device=device, dtype=dtype)
    sigmas: list[torch.Tensor] = []
    w_sigma_sum = torch.zeros(Do, Di, device=device, dtype=dtype)
    sigma_sum = torch.zeros(Di, Di, device=device, dtype=dtype)

    for W_t in d:
        W_c = W_t - W_t.mean(dim=0, keepdim=True)
        sigma_raw = W_c.T @ W_c
        tr = torch.trace(sigma_raw) + 1e-12
        if flag:
            sigma_t = sigma_raw / tr
            eps_t = ace_eps / tr
        else:
            sigma_t = sigma_raw
            eps_t = ace_eps
        sigmas.append(sigma_t)
        sigma_reg = sigma_t + eps_t * eye_di
        w_sigma_sum = w_sigma_sum + W_c @ sigma_reg
        sigma_sum = sigma_sum + sigma_reg

    # NOTE: reference computes mean(sum(Σ_t), dim=0, keepdim=True) — a row vector
    # broadcast over A; reproduced verbatim.
    c_agg = torch.mean(sum(sigmas), dim=0, keepdim=True)
    if flag:
        c_agg = c_agg / (avg_trace + 1e-12)
    A = sigma_sum + c_agg
    B = w_sigma_sum

    try:
        A_inv = torch.linalg.inv(A)
    except RuntimeError:
        A_inv = pinv(A)
    merging_vector = B @ A_inv

    if flag:
        sigma_mean = sigma_sum / N
        delta_res = torch.zeros(Do, Di, device=device, dtype=dtype)
        # Reference uses the uncentered originals here (loop variable shadowing
        # in the first pass left task_vectors untouched); reproduced verbatim.
        for W_t, S_t in zip(d, sigmas):
            delta_res = delta_res + W_t @ (S_t - sigma_mean)
        delta_fused = delta_res + merging_vector
        U, S, Vh = torch.linalg.svd(delta_fused, full_matrices=False)
        k = int(S.numel() * ace_k_frac)
        if k > 0:
            sigma_iso = S[:k].mean()
            U_k = U[:, :k]
            V_k = Vh[:k, :].T
            merging_vector = merging_vector + sigma_iso * (U_k @ V_k.T)
    return merging_vector


# ---------------------------------------------------------------------------
# TIES (Yadav et al., 2023 — https://arxiv.org/abs/2306.01708)
# ---------------------------------------------------------------------------
def merge_ties(d: torch.Tensor, ties_k: float = 0.2, **kwargs) -> torch.Tensor:
    """TIES-Merging: Trim, Elect Sign, Disjoint Merge.

    Per layer:
      1. Trim   — zero out all but the top-`ties_k` fraction of |d_t| entries.
      2. Elect  — per parameter, pick the sign of Σ_t d̂_t (sign with larger
                  total magnitude).
      3. Merge  — average only the task vectors whose sign matches the elected
                  sign at each parameter.
    """
    mask = _per_layer_topk_mask(d, ties_k)
    d_trim = d * mask

    elected_sign = torch.sign(d_trim.sum(dim=0))  # (Do, Di)
    sign_match = torch.sign(d_trim) == elected_sign.unsqueeze(0)  # (N, Do, Di)
    kept = d_trim * sign_match
    n_contrib = sign_match.sum(dim=0).clamp(min=1).to(d.dtype)
    return kept.sum(dim=0) / n_contrib


# ---------------------------------------------------------------------------
# WUDI (Li et al., ICML 2025 — https://arxiv.org/abs/2503.08099)
# ---------------------------------------------------------------------------
def merge_wudi(
    d: torch.Tensor,
    wudi_iters: int = 300,
    wudi_lr: float = 1e-5,
    wudi_weighted: bool = True,
    **kwargs,
) -> torch.Tensor:
    """WUDI: data-free merging by minimizing per-task interference.

    Optimizes M (init: Σ_i τ_i) for `wudi_iters` Adam steps at `wudi_lr` to
    minimize Σ_i ‖(M − τ_i) τ_iᵀ‖_F² / ‖τ_i‖_F².

    When `wudi_weighted=False`, drops the 1/‖τ_i‖² normalization — tasks with
    larger task-vector norm then dominate the objective (`merge_wudi_unweighted`).
    """
    N = d.shape[0]
    d_det = d.detach()
    if wudi_weighted:
        l2_sq = d_det.reshape(N, -1).pow(2).sum(dim=-1).view(N, 1, 1).clamp_min(1e-12)
    with torch.enable_grad():
        M = torch.nn.Parameter(d_det.sum(dim=0).clone())
        optimizer = torch.optim.Adam([M], lr=wudi_lr, weight_decay=0.0)
        for _ in tqdm(range(wudi_iters), desc="WUDI", leave=False):
            disturbing = M.unsqueeze(0) - d_det
            inner = torch.matmul(disturbing, d_det.transpose(1, 2))
            sq = inner.pow(2)
            loss = (sq / l2_sq).sum() if wudi_weighted else sq.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return M.detach()


def merge_wudi_unweighted(d: torch.Tensor, **kwargs) -> torch.Tensor:
    """WUDI without the 1/‖τ_i‖² per-task normalization."""
    kwargs.pop("wudi_weighted", None)
    return merge_wudi(d, wudi_weighted=False, **kwargs)


merge_dare_ties = lambda *args, **kwargs: merge_dare(*args, base_merge="ties", **kwargs)
merge_dare_actmat_gd = lambda *args, **kwargs: merge_dare(
    *args, base_merge="actmat_gd", **kwargs
)


def merge_dare(
    d: torch.Tensor,
    drop_rate: float = 0.3,
    rescale: bool = True,
    seed: int = 0,
    base_merge: str = "sum",
    **kwargs,
) -> torch.Tensor:
    """DARE: Drop And REscale (Yu et al. 2024, https://arxiv.org/abs/2311.03099 §3.1).

    For each task vector independently, drop each entry with probability
    `drop_rate`, rescale survivors by 1/(1 - drop_rate), then delegate to
    `base_merge` (default "sum" — DARE+TA). Set `base_merge="ties"` for DARE-TIES.
    """
    gen = torch.Generator(device=d.device).manual_seed(seed)
    keep = torch.rand(d.shape, generator=gen, device=d.device) >= drop_rate
    d_sparse = d * keep
    if rescale and drop_rate < 1.0:
        d_sparse = d_sparse / (1.0 - drop_rate)
    fn = getattr(sys.modules[__name__], "merge_" + base_merge)
    return fn(d_sparse, **kwargs)


def merge_dare_ties(d: torch.Tensor, **kwargs) -> torch.Tensor:
    """DARE-TIES: DARE drop-and-rescale with TIES as the base merge."""
    return merge_dare(d, base_merge="ties", **kwargs)
