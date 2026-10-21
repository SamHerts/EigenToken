"""Geometry over a token embedding matrix.

Everything here takes plain ``torch`` tensors so it works identically on the
static embedding table, the unembedding matrix, or contextual hidden states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# pairwise measurements
# --------------------------------------------------------------------------

@dataclass
class PairStats:
    cosine: float
    angle_deg: float
    euclidean: float
    dot: float
    norm_a: float
    norm_b: float
    # Euclidean distance after unit-normalising both vectors: direction-only.
    unit_euclidean: float


def pair_stats(a: torch.Tensor, b: torch.Tensor) -> PairStats:
    a = a.float()
    b = b.float()
    na, nb = a.norm().item(), b.norm().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    cos = max(-1.0, min(1.0, cos))
    return PairStats(
        cosine=cos,
        angle_deg=math.degrees(math.acos(cos)),
        euclidean=(a - b).norm().item(),
        dot=torch.dot(a, b).item(),
        norm_a=na,
        norm_b=nb,
        unit_euclidean=(F.normalize(a, dim=0) - F.normalize(b, dim=0)).norm().item(),
    )


def cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """[n, d] -> [n, n] cosine similarities."""
    u = F.normalize(vectors.float(), dim=-1)
    return u @ u.T


def cross_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[n, d], [m, d] -> [n, m] cosine similarities between two different sets."""
    ua = F.normalize(a.float(), dim=-1)
    ub = F.normalize(b.float(), dim=-1)
    return ua @ ub.T


# --------------------------------------------------------------------------
# neighbourhoods
# --------------------------------------------------------------------------

@dataclass
class Neighbor:
    token_id: int
    token: str
    score: float


def _unit(E: torch.Tensor) -> torch.Tensor:
    return F.normalize(E.float(), dim=-1)


def neighbors(
    vec: torch.Tensor,
    E: torch.Tensor,
    *,
    k: int = 15,
    exclude: Sequence[int] = (),
    metric: str = "cosine",
    chunk: int = 262_144,
) -> list[tuple[int, float]]:
    """Top-k rows of `E` closest to `vec`.  Chunked to stay within GPU memory."""
    vec = vec.float().to(E.device)
    if metric == "cosine":
        q = F.normalize(vec, dim=0)
    elif metric == "dot":
        q = vec
    elif metric == "euclidean":
        q = vec
    else:
        raise ValueError(f"unknown metric {metric!r}")

    scores = torch.empty(E.shape[0], device=E.device, dtype=torch.float32)
    for i in range(0, E.shape[0], chunk):
        block = E[i : i + chunk].float()
        if metric == "cosine":
            scores[i : i + chunk] = F.normalize(block, dim=-1) @ q
        elif metric == "dot":
            scores[i : i + chunk] = block @ q
        else:
            scores[i : i + chunk] = -torch.cdist(q.unsqueeze(0), block).squeeze(0)

    if exclude:
        idx = torch.tensor(list(exclude), device=E.device, dtype=torch.long)
        scores[idx] = float("-inf")

    k = min(k, scores.numel())
    vals, idxs = torch.topk(scores, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((i, -v if metric == "euclidean" else v))
    return out


# --------------------------------------------------------------------------
# analogy
# --------------------------------------------------------------------------

def analogy_vector(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """a : b :: c : ?   ->   b - a + c"""
    return b.float() - a.float() + c.float()


def pair_direction_similarity(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> float:
    """How parallel is (b - a) to (d - c)?

    This is the honest version of an analogy test: it asks whether the two
    offsets point the same way, instead of letting the answer be carried by
    `c` simply being near `d` already.
    """
    return F.cosine_similarity(
        (b.float() - a.float()), (d.float() - c.float()), dim=0
    ).item()


# --------------------------------------------------------------------------
# semantic axes built from example pairs
# --------------------------------------------------------------------------

@dataclass
class Axis:
    direction: torch.Tensor          # unit vector
    pair_labels: list[str]
    deltas: torch.Tensor             # [n_pairs, d] raw offsets
    coherence: float                 # mean pairwise cosine between offsets
    pc1_ratio: float                 # share of offset energy lying along PC1
    pc1_alignment: float             # |cos| between mean direction and PC1
    delta_norms: list[float]

    @property
    def is_linear(self) -> bool:
        """Whether treating this contrast as one direction is defensible."""
        return self.coherence > 0.25 and self.pc1_ratio > 0.4


def build_axis(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], labels: Sequence[str]) -> Axis:
    """Derive a direction from (from, to) pairs, with diagnostics.

    A "gender direction" is only meaningful if the individual offsets
    (woman-man, queen-king, ...) actually agree.  `coherence` and `pc1_ratio`
    report whether they do; low values mean the axis is an artefact of
    averaging noise and any projection built on it should be discarded.

    With n offsets of mean pairwise cosine c, a perfectly rank-one set gives
    pc1_ratio 1.0 and random offsets give roughly 1/n, so the two diagnostics
    are read together: pc1_ratio well above 1/n is the signal.
    """
    deltas = torch.stack([(b.float() - a.float()) for a, b in pairs])
    unit = F.normalize(deltas, dim=-1)

    n = unit.shape[0]
    if n > 1:
        sim = unit @ unit.T
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        coherence = off.mean().item()
    else:
        coherence = 1.0

    if n > 1:
        # Economy SVD on the *uncentered* unit offsets ([n, d] with n << d).
        # Centering first would subtract the shared component -- precisely the
        # thing being measured -- and leave PC1 describing the disagreement.
        _, S, Vh = torch.linalg.svd(unit, full_matrices=False)
        energy = S ** 2
        pc1_ratio = (energy[0] / energy.sum().clamp_min(1e-12)).item()
        pc1 = Vh[0]
    else:
        pc1_ratio = 1.0
        pc1 = F.normalize(deltas[0], dim=0)

    direction = F.normalize(unit.mean(dim=0), dim=0)
    pc1_align = abs(F.cosine_similarity(direction, pc1, dim=0).item())

    return Axis(
        direction=direction,
        pair_labels=list(labels),
        deltas=deltas,
        coherence=coherence,
        pc1_ratio=pc1_ratio,
        pc1_alignment=pc1_align,
        delta_norms=[d.norm().item() for d in deltas],
    )


def project(vectors: torch.Tensor, axis: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Signed position of each row along `axis`.

    With `normalize`, rows are unit-length first, so the score is a cosine and
    is not dominated by token frequency (rare tokens have large norms).
    """
    v = vectors.float()
    if normalize:
        v = F.normalize(v, dim=-1)
    return v @ F.normalize(axis.float(), dim=0)


def reject(vectors: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Remove the `axis` component -- the standard 'debiasing' projection."""
    u = F.normalize(axis.float(), dim=0)
    v = vectors.float()
    return v - torch.outer(v @ u, u) if v.dim() == 2 else v - (v @ u) * u


# --------------------------------------------------------------------------
# per-dimension attribution
# --------------------------------------------------------------------------

@dataclass
class DimContribution:
    index: int
    a: float
    b: float
    delta: float
    cosine_share: float   # share of the (normalised) dot product
    delta_share: float    # share of squared distance


def dim_attribution(a: torch.Tensor, b: torch.Tensor, k: int = 15) -> dict:
    """Which coordinates carry the similarity, and which carry the difference?

    Individual dimensions of an LLM embedding are not interpretable features,
    but a heavily concentrated attribution is itself a finding -- it usually
    points at 'rogue dimensions' (a handful of outlier coordinates present in
    almost every token) rather than anything semantic.
    """
    a = a.float()
    b = b.float()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    cos_share = (a * b) / denom
    delta = b - a
    dsq = delta ** 2
    dshare = dsq / dsq.sum().clamp_min(1e-12)

    def pack(idxs) -> list[DimContribution]:
        return [
            DimContribution(
                index=int(i),
                a=a[i].item(),
                b=b[i].item(),
                delta=delta[i].item(),
                cosine_share=cos_share[i].item(),
                delta_share=dshare[i].item(),
            )
            for i in idxs
        ]

    k = min(k, a.numel())
    top_sim = torch.topk(cos_share, k).indices.tolist()
    top_opp = torch.topk(-cos_share, k).indices.tolist()
    top_diff = torch.topk(dshare, k).indices.tolist()

    # How concentrated is the difference?  1.0 = one dimension explains it all.
    sorted_share = torch.sort(dshare, descending=True).values
    top10_mass = sorted_share[: min(10, sorted_share.numel())].sum().item()
    effective_dims = (1.0 / (dshare ** 2).sum().clamp_min(1e-12)).item()

    return {
        "aligned": pack(top_sim),
        "opposed": pack(top_opp),
        "divergent": pack(top_diff),
        "top10_delta_mass": top10_mass,
        "effective_dims": effective_dims,
        "total_dims": int(a.numel()),
    }


# --------------------------------------------------------------------------
# dimensionality reduction for plotting / export
# --------------------------------------------------------------------------

def pca(vectors: torch.Tensor, n_components: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (coords [n, k], explained_variance_ratio [k], components [k, d])."""
    X = vectors.float()
    X = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    k = min(n_components, Vh.shape[0])
    var = S ** 2
    ratio = (var[:k] / var.sum().clamp_min(1e-12))
    coords = U[:, :k] * S[:k]
    return coords, ratio, Vh[:k]


def isotropy(E: torch.Tensor, sample: int = 20000, seed: int = 0) -> dict:
    """How spherical is this embedding cloud?

    LLM embedding spaces are famously anisotropic: a large shared mean vector
    makes every pair of tokens look similar.  `mean_cosine` near 0 means cosine
    comparisons are trustworthy as-is; near 0.5 means you should pass --center.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = E.shape[0]
    idx = torch.randperm(n, generator=g)[: min(sample, n)].to(E.device)
    X = E[idx].float()
    u = F.normalize(X, dim=-1)
    m = u.shape[0]
    # mean off-diagonal cosine, computed without forming the full matrix
    s = u.sum(dim=0)
    total = (s @ s - m) / (m * (m - 1))
    mean_vec = X.mean(dim=0)
    return {
        "sampled": m,
        "mean_cosine": total.item(),
        "mean_norm": X.norm(dim=-1).mean().item(),
        "mean_vector_norm": mean_vec.norm().item(),
        # >1 means the cloud's centre of mass is further from the origin than a
        # typical token is from the centre: cosine similarity is badly skewed.
        "offset_ratio": (mean_vec.norm() / (X - mean_vec).norm(dim=-1).mean()).item(),
    }


# --------------------------------------------------------------------------
# neighbourhood cloud layout
# --------------------------------------------------------------------------

def kmeans(vectors: torch.Tensor, k: int, *, iters: int = 40, seed: int = 0) -> torch.Tensor:
    """Spherical k-means (cosine) via k-means++ on unit vectors.

    Used to colour a token cloud by sub-group, which often separates the
    distinct senses or registers of a word.
    """
    X = F.normalize(vectors.float(), dim=-1)
    n = X.shape[0]
    k = max(1, min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)

    # k-means++ seeding
    first = torch.randint(n, (1,), generator=g).item()
    centers = [X[first]]
    for _ in range(k - 1):
        d = 1.0 - (X @ torch.stack(centers).T).max(dim=1).values
        d = d.clamp_min(0)
        if d.sum() <= 0:
            centers.append(X[torch.randint(n, (1,), generator=g).item()])
            continue
        probs = (d / d.sum()).cpu()
        centers.append(X[torch.multinomial(probs, 1, generator=g).item()])
    C = torch.stack(centers)

    labels = torch.zeros(n, dtype=torch.long, device=X.device)
    for _ in range(iters):
        new = (X @ C.T).argmax(dim=1)
        if torch.equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = F.normalize(X[m].mean(dim=0), dim=0)
    return labels


def cloud_layout(
    query: torch.Tensor,
    vectors: torch.Tensor,
    *,
    mode: str = "radial",
    n_clusters: int = 0,
    seed: int = 0,
) -> dict:
    """Place a token's neighbourhood on a 2D plane.

    ``radial`` (default) is the honest layout: the **radius** of every point is
    its true cosine distance from the query, so distance from the centre always
    means what it looks like it means, and only the **angle** comes from a 2D
    projection of the neighbourhood.  A plain PCA scatter distorts both axes at
    once and invites reading structure into what is mostly projection error.

    ``pca`` gives the conventional scatter for comparison, with the query
    included in the fit.
    """
    q = query.float()
    V = vectors.float()
    sims = F.cosine_similarity(V, q.unsqueeze(0), dim=-1)

    if mode == "pca":
        coords, ratio, _ = pca(torch.cat([q.unsqueeze(0), V]), 2)
        qxy = coords[0]
        xy = coords[1:] - qxy
        scale = xy.norm(dim=1).max().clamp_min(1e-9)
        xy = xy / scale
        explained = ratio.tolist()
    elif mode == "radial":
        # Angle from a PCA of the offsets; radius from the real cosine.
        offsets = V - q.unsqueeze(0)
        coords, ratio, _ = pca(offsets, 2)
        theta = torch.atan2(coords[:, 1], coords[:, 0])
        r = (1.0 - sims).clamp_min(0.0)
        r = r / r.max().clamp_min(1e-9)
        xy = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        explained = ratio.tolist()
    else:
        raise ValueError(f"unknown layout mode {mode!r}")

    labels = (
        kmeans(V, n_clusters, seed=seed).tolist()
        if n_clusters and n_clusters > 1
        else [0] * V.shape[0]
    )

    return {
        "coords": xy.tolist(),
        "similarities": sims.tolist(),
        "clusters": labels,
        "explained_variance_ratio": explained,
        "mode": mode,
    }
