"""Comparing two models whose embedding spaces have different dimensions.

Vector coordinates are not comparable across models -- there is no reason
dimension 417 means the same thing in two independently trained networks, and
the dimensionalities usually differ anyway.  What *is* comparable is the shape
of the relationships:

* **RSA** (representational similarity analysis) -- build the word x word
  cosine matrix inside each model, then correlate those matrices.  Two models
  that organise meaning the same way agree here even at different widths.
* **Procrustes** -- find the best rotation mapping one space onto the other
  (after reducing both to a common dimension) and report the residual.
* **Neighbourhood overlap** -- for each word, how much do the two models'
  nearest-neighbour lists agree?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .geometry import cosine_matrix, neighbors


def _upper(mat: torch.Tensor) -> torch.Tensor:
    n = mat.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=mat.device)
    return mat[iu[0], iu[1]]


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks, matching scipy's tie handling."""
    x = x.flatten().double()
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.double, device=x.device)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.double, device=x.device)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


@dataclass
class RSAResult:
    words: list[str]
    pearson: float
    spearman: float
    per_word: list[tuple[str, float]]   # word -> correlation of its own sim profile


def rsa(words: Sequence[str], va: torch.Tensor, vb: torch.Tensor) -> RSAResult:
    """Correlate the internal similarity structure of two models."""
    ca = cosine_matrix(va.cpu())
    cb = cosine_matrix(vb.cpu())
    ua, ub = _upper(ca), _upper(cb)

    per_word = []
    n = len(words)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        ra = ca[i, keep]
        rb = cb[i, keep]
        per_word.append((words[i], _spearman(ra, rb) if n > 2 else float("nan")))
    per_word.sort(key=lambda t: t[1])

    return RSAResult(
        words=list(words),
        pearson=_pearson(ua, ub),
        spearman=_spearman(ua, ub),
        per_word=per_word,
    )


@dataclass
class ProcrustesResult:
    dim: int
    disparity: float        # 0 = perfect rotation exists, 1 = no better than nothing
    scale: float
    per_word_error: list[tuple[str, float]]


def procrustes(
    words: Sequence[str], va: torch.Tensor, vb: torch.Tensor
) -> ProcrustesResult:
    """Best orthogonal map from space A to space B, after PCA to a common dim.

    Both clouds are centred and Frobenius-normalised first, so `disparity` is
    scale-free: it is the residual fraction of variance the rotation cannot
    account for.
    """
    A = va.float().cpu()
    B = vb.float().cpu()
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)

    d = min(A.shape[1], B.shape[1], A.shape[0])
    if A.shape[1] > d:
        A = A @ torch.linalg.svd(A, full_matrices=False).Vh[:d].T
    if B.shape[1] > d:
        B = B @ torch.linalg.svd(B, full_matrices=False).Vh[:d].T

    A = A / A.norm().clamp_min(1e-12)
    B = B / B.norm().clamp_min(1e-12)

    U, S, Vh = torch.linalg.svd(A.T @ B)
    R = U @ Vh
    scale = S.sum().item()
    A_rot = A @ R
    disparity = ((A_rot - B) ** 2).sum().item()

    err = (A_rot - B).norm(dim=1)
    per_word = sorted(zip(words, err.tolist()), key=lambda t: -t[1])
    return ProcrustesResult(dim=d, disparity=disparity, scale=scale, per_word_error=per_word)


@dataclass
class OverlapResult:
    k: int
    mean_jaccard: float
    per_word: list[tuple[str, float, list[str], list[str]]]


def neighborhood_overlap(
    words: Sequence[str],
    model_a,
    model_b,
    vecs_a: torch.Tensor,
    vecs_b: torch.Tensor,
    *,
    k: int = 10,
    exclude_a: Sequence[Sequence[int]] = (),
    exclude_b: Sequence[Sequence[int]] = (),
) -> OverlapResult:
    """Do the two models put the same words next to each word?

    Compared on decoded surface strings, since token ids are model-specific.
    """
    from .vocab import SPACE_MARK, token_label

    rows = []
    scores = []
    for i, w in enumerate(words):
        ex_a = exclude_a[i] if i < len(exclude_a) else ()
        ex_b = exclude_b[i] if i < len(exclude_b) else ()
        na = neighbors(vecs_a[i], model_a.embeddings, k=k, exclude=ex_a)
        nb = neighbors(vecs_b[i], model_b.embeddings, k=k, exclude=ex_b)
        sa = {token_label(model_a.tokenizer, t).lstrip(SPACE_MARK).lower() for t, _ in na}
        sb = {token_label(model_b.tokenizer, t).lstrip(SPACE_MARK).lower() for t, _ in nb}
        j = len(sa & sb) / max(1, len(sa | sb))
        scores.append(j)
        rows.append((w, j, sorted(sa), sorted(sb)))

    rows.sort(key=lambda r: -r[1])
    mean = sum(scores) / max(1, len(scores))
    return OverlapResult(k=k, mean_jaccard=mean, per_word=rows)
