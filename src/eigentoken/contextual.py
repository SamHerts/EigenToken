"""Contextual embeddings: a word's vector at a given layer, inside a sentence.

The static embedding table is only the model's starting point.  By the middle
layers a token's representation has absorbed its context, and that is usually
where the interesting structure lives -- 'bank' near a river and 'bank' near a
loan are the same row of the embedding table but very different vectors at
layer 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .vocab import Pooling, pool


@dataclass
class ContextVector:
    word: str
    context: str
    layer: int
    token_ids: list[int]
    token_strs: list[str]
    char_span: tuple[int, int]
    vector: torch.Tensor

    @property
    def label(self) -> str:
        return f"{self.word!r} @ layer {self.layer} in {self.context[:48]!r}"


def _find_span(context: str, word: str) -> tuple[int, int]:
    lo = context.lower().find(word.lower())
    if lo < 0:
        raise ValueError(f"{word!r} does not occur in the context {context!r}")
    return lo, lo + len(word)


def embed_in_context(
    model,
    word: str,
    context: str,
    *,
    layer: int = -1,
    pooling: Pooling = "mean",
    occurrence: int = 0,
) -> ContextVector:
    """Hidden state for `word` inside `context`, taken from `layer`.

    `layer` indexes ``hidden_states``: 0 is the embedding table output, 1 is
    after the first block, -1 is the final layer.  Requires the model to have
    been loaded with ``full=True``.
    """
    if model.model is None:
        raise RuntimeError(
            "contextual embeddings need the full network; load with full=True "
            "(the CLI does this automatically when --layer is given)"
        )

    tok = model.tokenizer
    start = 0
    for _ in range(occurrence + 1):
        lo = context.lower().find(word.lower(), start)
        if lo < 0:
            raise ValueError(
                f"occurrence {occurrence} of {word!r} not found in {context!r}"
            )
        start = lo + 1
    span = (lo, lo + len(word))

    enc = tok(context, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}

    positions = [
        i
        for i, (s, e) in enumerate(offsets)
        if e > s and s < span[1] and e > span[0]
    ]
    if not positions:
        raise ValueError(
            f"could not align {word!r} to any token in {context!r}; "
            "this tokenizer may not provide offset mappings"
        )

    with torch.no_grad():
        out = model.model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    idx = layer if layer >= 0 else len(hs) + layer
    if not 0 <= idx < len(hs):
        raise IndexError(f"layer {layer} out of range (model has {len(hs)} hidden states)")

    states = hs[idx][0][positions].float()
    vec = pool(states, pooling) if len(positions) > 1 else states[0]

    ids = enc["input_ids"][0][positions].tolist()
    return ContextVector(
        word=word,
        context=context,
        layer=idx,
        token_ids=ids,
        token_strs=tok.convert_ids_to_tokens(ids),
        char_span=span,
        vector=vec,
    )


def layer_trajectory(
    model, word: str, context: str, *, pooling: Pooling = "mean"
) -> list[torch.Tensor]:
    """The word's vector at every layer -- useful for seeing where meaning settles."""
    if model.model is None:
        raise RuntimeError("layer trajectories need the full network (full=True)")

    tok = model.tokenizer
    span = _find_span(context, word)
    enc = tok(context, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    positions = [
        i for i, (s, e) in enumerate(offsets) if e > s and s < span[1] and e > span[0]
    ]
    with torch.no_grad():
        out = model.model(**enc, output_hidden_states=True)
    return [
        pool(h[0][positions].float(), pooling) if len(positions) > 1 else h[0][positions][0].float()
        for h in out.hidden_states
    ]
