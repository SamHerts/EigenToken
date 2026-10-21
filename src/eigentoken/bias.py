"""Word Embedding Association Test (WEAT) -- Caliskan, Bryson & Narayanan
(2017), "Semantics derived automatically from language corpora contain
human-like biases".

Given two *target* word sets X and Y (e.g. career vs. family words) and two
*attribute* word sets A and B (e.g. male vs. female terms), WEAT measures
how much more strongly a model associates X with A (and Y with B) than
chance would predict, purely from cosine similarity in the embedding space.
It is a statistical test, not a definitive judgement: small word lists and
a single embedding matrix are weak evidence on their own, and the built-in
presets below are illustrative, not an audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from . import geometry


@dataclass
class WeatResult:
    x_label: str
    y_label: str
    a_label: str
    b_label: str
    statistic: float              # s(X,Y,A,B), the raw sum-of-differences
    effect_size: float            # Cohen's d
    p_value: float                # one-sided permutation-test estimate
    n_permutations: int
    word_scores: list[tuple[str, str, float]]  # (word, 'x'|'y', s(w,A,B)), sorted desc

    @property
    def magnitude(self) -> str:
        """Cohen's rule-of-thumb reading of |effect_size|."""
        d = abs(self.effect_size)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"


def weat(
    X: torch.Tensor, Y: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
    x_words: Sequence[str], y_words: Sequence[str],
    x_label: str, y_label: str, a_label: str, b_label: str,
    *, n_permutations: int = 10_000, seed: int = 0,
) -> WeatResult:
    """Run WEAT on four groups of already-resolved embedding vectors.

    ``p_value`` is a Monte Carlo estimate: it draws ``n_permutations`` random
    relabellings of X union Y (keeping the group sizes fixed) and reports the
    fraction at least as extreme as the real split. It is not exact -- an
    exhaustive permutation test is only tractable for tiny word lists -- but
    sampling is standard practice for WEAT.
    """
    sXA = geometry.cross_cosine(X, A).mean(dim=1)
    sXB = geometry.cross_cosine(X, B).mean(dim=1)
    sYA = geometry.cross_cosine(Y, A).mean(dim=1)
    sYB = geometry.cross_cosine(Y, B).mean(dim=1)
    sX = sXA - sXB
    sY = sYA - sYB

    nx, ny = sX.shape[0], sY.shape[0]
    pooled = torch.cat([sX, sY])
    statistic = (sX.sum() - sY.sum()).item()
    std = pooled.std(unbiased=True).item()
    effect_size = 0.0 if std == 0 else (sX.mean().item() - sY.mean().item()) / std

    # Permutations only ever reshuffle these `nx + ny` scalar scores, never
    # recompute cosine similarities, so 10k+ draws are cheap even though the
    # embeddings themselves may be large.
    g = torch.Generator().manual_seed(seed)
    n = nx + ny
    idx = torch.argsort(torch.rand(n_permutations, n, generator=g), dim=1)
    grp = pooled[idx.to(pooled.device)]
    perm_stats = grp[:, :nx].sum(dim=1) - grp[:, nx:].sum(dim=1)
    p_value = (perm_stats >= statistic).float().mean().item()

    word_scores = (
        [(w, "x", s) for w, s in zip(x_words, sX.tolist())]
        + [(w, "y", s) for w, s in zip(y_words, sY.tolist())]
    )
    word_scores.sort(key=lambda t: -t[2])

    return WeatResult(
        x_label=x_label, y_label=y_label, a_label=a_label, b_label=b_label,
        statistic=statistic, effect_size=effect_size, p_value=p_value,
        n_permutations=n_permutations, word_scores=word_scores,
    )


# --------------------------------------------------------------------------
# built-in test batteries
#
# Word lists, not names, to sidestep any specific-identity association and
# keep the defaults simple. They are deliberately short (WEAT's original
# lists ran to 25 words a side): enough to get a real signal, small enough
# to read at a glance. Custom lists via --x/--y/--a/--b (CLI) or the four
# text boxes (web) are the way to do a serious test.
# --------------------------------------------------------------------------

_MALE = ["male", "man", "boy", "brother", "he", "him", "his", "son", "father", "uncle"]
_FEMALE = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter", "mother", "aunt"]
_ARTS = ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"]

PRESETS: dict[str, dict] = {
    "flowers_insects": {
        "description": (
            "The original Greenwald IAT baseline (flowers vs. insects, "
            "pleasant vs. unpleasant words) -- not a social bias, but a "
            "strong, well-known effect useful as a sanity check."
        ),
        "x_label": "flowers", "y_label": "insects",
        "a_label": "pleasant", "b_label": "unpleasant",
        "x": ["rose", "tulip", "daisy", "orchid", "lily",
              "daffodil", "iris", "violet", "peony", "carnation"],
        "y": ["ant", "spider", "wasp", "cockroach", "mosquito",
              "beetle", "tick", "maggot", "moth", "flea"],
        "a": ["love", "peace", "freedom", "joy", "honest",
              "gentle", "friend", "paradise", "happy", "rainbow"],
        "b": ["death", "hatred", "poison", "filth", "murder",
              "disease", "ugly", "vomit", "prison", "agony"],
    },
    "career_family": {
        "description": "Career vs. family words, associated with male vs. female terms.",
        "x_label": "career", "y_label": "family",
        "a_label": "male", "b_label": "female",
        "x": ["executive", "management", "professional", "corporation",
              "salary", "office", "business", "career"],
        "y": ["home", "parents", "children", "family",
              "cousins", "marriage", "wedding", "relatives"],
        "a": _MALE, "b": _FEMALE,
    },
    "science_arts": {
        "description": "Science vs. arts words, associated with male vs. female terms.",
        "x_label": "science", "y_label": "arts",
        "a_label": "male", "b_label": "female",
        "x": ["science", "technology", "physics", "chemistry", "engineering",
              "astronomy", "biology", "laboratory"],
        "y": _ARTS,
        "a": _MALE, "b": _FEMALE,
    },
    "math_arts": {
        "description": "Math vs. arts words, associated with male vs. female terms.",
        "x_label": "math", "y_label": "arts",
        "a_label": "male", "b_label": "female",
        "x": ["math", "algebra", "geometry", "calculus",
              "equations", "computation", "numbers", "addition"],
        "y": _ARTS,
        "a": _MALE, "b": _FEMALE,
    },
}
