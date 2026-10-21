"""Resolving human words to rows of a token embedding matrix.

This is where most of the honesty in the tool lives.  "king" is not a vector in
a modern LLM -- ``king``, ``ĠKing`` and ``KING`` are different rows, and plenty
of words are not single tokens at all.  Every resolution returns a
:class:`Resolution` recording exactly which token ids were used and whether
pooling was needed, so results can be read with the right amount of trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch

Pooling = Literal["mean", "sum", "first", "last", "norm-mean"]


@dataclass
class Resolution:
    query: str
    token_ids: list[int]
    token_strs: list[str]
    surface: str            # the variant that was actually encoded
    vector: torch.Tensor
    single_token: bool
    pooling: Optional[str] = None
    candidates: list[tuple[str, list[int]]] = None  # all variants tried

    @property
    def label(self) -> str:
        if self.single_token:
            return f"{self.query!r} -> {self.token_strs[0]!r} (id {self.token_ids[0]})"
        return (
            f"{self.query!r} -> {len(self.token_ids)} tokens "
            f"{self.token_strs} ({self.pooling}-pooled)"
        )

    @property
    def warning(self) -> Optional[str]:
        if not self.single_token:
            return (
                f"{self.query!r} is not a single token in this vocabulary; "
                f"using the {self.pooling} of {len(self.token_ids)} subword vectors. "
                "Treat the geometry as approximate."
            )
        return None


def _variants(word: str, prefer_space: bool) -> list[str]:
    """Surface forms worth trying, best first."""
    base = word
    forms: list[str] = []

    def add(s: str) -> None:
        if s not in forms:
            forms.append(s)

    if prefer_space and not base.startswith((" ", "\t", "\n")):
        add(" " + base)
        add(base)
    else:
        add(base)
        if not base.startswith(" "):
            add(" " + base)

    stripped = base.strip()
    for cased in (stripped.capitalize(), stripped.lower(), stripped.upper()):
        if cased != stripped:
            if prefer_space:
                add(" " + cased)
                add(cased)
            else:
                add(cased)
                add(" " + cased)
    return forms


def pool(vectors: torch.Tensor, how: Pooling) -> torch.Tensor:
    if how == "mean":
        return vectors.mean(dim=0)
    if how == "sum":
        return vectors.sum(dim=0)
    if how == "first":
        return vectors[0]
    if how == "last":
        return vectors[-1]
    if how == "norm-mean":
        # Average of unit vectors: direction-only pooling, ignores that rare
        # subwords often have much larger norms than common ones.
        return torch.nn.functional.normalize(vectors, dim=-1).mean(dim=0)
    raise ValueError(f"unknown pooling {how!r}")


def resolve(
    model,
    query: str,
    *,
    pooling: Pooling = "mean",
    prefer_space: bool = True,
    allow_multi: bool = True,
) -> Resolution:
    """Turn `query` into a single vector from ``model.embeddings``.

    Accepts three forms:

    * ``id:1234``  -- a raw token id
    * ``tok:Ġking`` -- an exact vocabulary entry, no variant search
    * ``king``     -- a word; leading-space and case variants are tried and the
      first one that is a *single* token wins
    """
    tok = model.tokenizer
    E = model.embeddings

    if query.startswith("id:"):
        tid = int(query[3:])
        return Resolution(
            query, [tid], tok.convert_ids_to_tokens([tid]), query,
            E[tid], True, candidates=[],
        )

    if query.startswith("tok:"):
        piece = query[4:]
        tid = tok.convert_tokens_to_ids(piece)
        unk = getattr(tok, "unk_token_id", None)
        if tid is None or (unk is not None and tid == unk and piece != tok.unk_token):
            raise KeyError(f"{piece!r} is not in the vocabulary of {model.name}")
        return Resolution(query, [tid], [piece], piece, E[tid], True, candidates=[])

    tried: list[tuple[str, list[int]]] = []
    for form in _variants(query, prefer_space):
        ids = tok.encode(form, add_special_tokens=False)
        tried.append((form, ids))
        if len(ids) == 1:
            return Resolution(
                query, ids, tok.convert_ids_to_tokens(ids), form,
                E[ids[0]], True, candidates=tried,
            )

    if not allow_multi:
        raise KeyError(
            f"{query!r} has no single-token form in {model.name} "
            f"(tried {[f for f, _ in tried]})"
        )

    form, ids = tried[0]
    if not ids:
        raise KeyError(f"{query!r} encodes to nothing in {model.name}")
    vec = pool(E[torch.tensor(ids, device=E.device)], pooling)
    return Resolution(
        query, list(ids), tok.convert_ids_to_tokens(ids), form,
        vec, False, pooling=pooling, candidates=tried,
    )


def resolve_all(model, queries: Sequence[str], **kw) -> list[Resolution]:
    return [resolve(model, q, **kw) for q in queries]


def case_variant_ids(tokenizer, word: str) -> list[int]:
    """Every distinct single-token id for `word` across leading-space and
    casing variants -- 'king', ' king', 'King', ' King', 'KING', ' KING'.

    Without this, a neighbour search for a common word is dominated by its
    own casing/spacing variants (they are the closest possible vectors,
    trivially), crowding out the words that are actually related.
    """
    ids = set()
    for form in _variants(word, True):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return sorted(ids)


# The word-initial space is part of the token in byte-level BPE, so it has to
# be visible -- ``water`` and `` water`` are different rows.  U+2423 OPEN BOX is
# the Unicode symbol for a space and, unlike an underscore, collides with
# nothing: Qwen alone has 5555 tokens that literally begin with "_".
SPACE_MARK = "\u2423"


def display_token(tok_str: str) -> str:
    """Make BPE space markers legible in terminal output."""
    return (
        tok_str.replace("Ġ", SPACE_MARK)
        .replace("▁", SPACE_MARK)
        .replace("</w>", "")
    )


def token_label(tokenizer, token_id: int) -> str:
    """A readable name for a token id.

    Byte-level BPE stores non-ASCII text as mojibake (``åĽ½çİĭ``), so the raw
    vocabulary string is useless for anything but English.  Decoding the id
    recovers the real text (``国王``); a leading ``␣`` marks the word-initial
    space form.  Falls back to the raw piece for fragments that are not valid
    text on their own.
    """
    raw = tokenizer.convert_ids_to_tokens(int(token_id))
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    if text and text.strip() and "\ufffd" not in text:
        lead = SPACE_MARK if text[:1].isspace() or raw.startswith(("Ġ", "▁")) else ""
        return lead + text.strip()
    return display_token(raw)


def decode_ids(model, ids: Sequence[int]) -> list[str]:
    return [display_token(t) for t in model.tokenizer.convert_ids_to_tokens(list(ids))]
