"""Model loading and extraction of token embedding matrices.

Two paths are supported:

* **lazy** (default) -- read only the tensor holding the embedding table out of
  the model's ``safetensors`` shards.  A 70B model's input embedding is a few
  hundred MB, so this avoids materialising the whole network just to look at
  token geometry.
* **full** -- load the model with ``AutoModel``.  Required for contextual
  embeddings (``--layer``) and used as a fallback when the lazy path cannot
  identify the embedding tensor.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch

MatrixKind = Literal["input", "output"]

# Ordered by how commonly they appear; first match in the checkpoint wins.
INPUT_EMBED_PATTERNS = [
    r"^model\.embed_tokens\.weight$",
    r"^model\.decoder\.embed_tokens\.weight$",
    r"^transformer\.wte\.weight$",
    r"^transformer\.word_embeddings\.weight$",
    r"^gpt_neox\.embed_in\.weight$",
    r"^embeddings\.word_embeddings\.weight$",
    r"^\w+\.embeddings\.word_embeddings\.weight$",
    r"^tok_embeddings\.weight$",
    r"^embed_tokens\.weight$",
    r"^language_model\.model\.embed_tokens\.weight$",
    r"(^|\.)embed_tokens\.weight$",
    r"(^|\.)wte\.weight$",
    r"(^|\.)word_embeddings\.weight$",
]

OUTPUT_EMBED_PATTERNS = [
    r"^lm_head\.weight$",
    r"^embed_out\.weight$",
    r"^output\.weight$",
    r"^language_model\.lm_head\.weight$",
    r"(^|\.)lm_head\.weight$",
]

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": None,
}


@dataclass
class LoadedModel:
    """An embedding matrix plus the tokenizer that indexes it."""

    name: str
    tokenizer: object
    embeddings: torch.Tensor          # [vocab, dim], float32, on `device`
    kind: MatrixKind
    device: str
    source: str                        # human readable provenance
    model: Optional[object] = None     # populated only on the full path
    transforms: list[str] = field(default_factory=list)
    n_layers: Optional[int] = None     # transformer blocks, for --layer bounds

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def vocab_size(self) -> int:
        return self.embeddings.shape[0]

    def describe(self) -> str:
        t = f" [{', '.join(self.transforms)}]" if self.transforms else ""
        return (
            f"{self.name}  {self.kind} embeddings  "
            f"{self.vocab_size} x {self.dim}  ({self.source}){t}"
        )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _match(keys, patterns) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat)
        hits = [k for k in keys if rx.search(k)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            # Prefer the shortest key: avoids picking an adapter/vision tower.
            return sorted(hits, key=len)[0]
    return None


class ModelRefused(Exception):
    """A model this caller's limits rule out: wrong weight format, or too big."""


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n} B"


@dataclass
class Limits:
    """What a server will agree to load. The defaults refuse nothing.

    The checks run against the Hub's file listing *before* any weights are
    downloaded, so a refusal costs one metadata request, not a download.
    """

    safetensors_only: bool = False          # refuse pickle (.bin/.pt) checkpoints
    max_download_bytes: Optional[int] = None  # per load, counting cached files too
    max_matrix_bytes: Optional[int] = None  # the float32 matrix kept in memory
    notify: Optional[Callable[[str], None]] = None  # progress notes for the caller
    # Called before downloading this many bytes, or reading this many into
    # memory, so a server can make room first. Either may raise ModelRefused.
    before_fetch: Optional[Callable[[int], None]] = None
    before_matrix: Optional[Callable[[int], None]] = None

    @property
    def active(self) -> bool:
        return bool(self.safetensors_only or self.max_download_bytes
                    or self.max_matrix_bytes or self.notify
                    or self.before_fetch or self.before_matrix)


class _Guard:
    """Applies :class:`Limits` to one load, given the repo's file sizes."""

    def __init__(self, model_id: str, revision: Optional[str], token: Optional[str],
                 limits: Limits, cache_dir: Optional[str] = None):
        from huggingface_hub import HfApi

        self.model_id = model_id
        self.revision = revision
        self.limits = limits
        self.cache_dir = cache_dir
        info = HfApi().model_info(model_id, revision=revision, token=token,
                                  files_metadata=True)
        self.sizes = {f.rfilename: (f.size or 0) for f in info.siblings or []}
        self.has_safetensors = any(f.endswith(".safetensors") for f in self.sizes)
        if limits.safetensors_only and not self.has_safetensors:
            raise ModelRefused(
                f"{model_id} has no safetensors weights. This server only loads "
                "safetensors files, which hold plain data; the older .bin format "
                "is a Python pickle that can carry code."
            )

    def before_download(self, files: list[str]) -> None:
        """Refuse, or announce, fetching `files`."""
        from huggingface_hub import try_to_load_from_cache

        total = sum(self.sizes.get(f, 0) for f in files)
        cap = self.limits.max_download_bytes
        if cap and total > cap:
            raise ModelRefused(
                f"{self.model_id} needs {_fmt_bytes(total)} of weights; this "
                f"server's limit is {_fmt_bytes(cap)} per model."
            )
        fresh = sum(
            self.sizes.get(f, 0) for f in files
            if not isinstance(try_to_load_from_cache(self.model_id, f, revision=self.revision,
                                                     cache_dir=self.cache_dir), str)
        )
        if fresh and self.limits.before_fetch:
            self.limits.before_fetch(fresh)
        if fresh and self.limits.notify:
            self.limits.notify(f"downloading {_fmt_bytes(fresh)}")

    def check_matrix(self, shape) -> None:
        cap = self.limits.max_matrix_bytes
        size = 4
        for d in shape:
            size *= int(d)
        if cap and size > cap:
            raise ModelRefused(
                f"{self.model_id}'s embedding matrix is {' x '.join(map(str, shape))}, "
                f"{_fmt_bytes(size)} in memory; this server's limit is {_fmt_bytes(cap)}."
            )
        return size

    def reserve_memory(self, nbytes: int) -> None:
        if self.limits.before_matrix:
            self.limits.before_matrix(nbytes)

    def weight_files(self) -> list[str]:
        """What a full load downloads: transformers prefers safetensors when present."""
        ext = (".safetensors",) if self.has_safetensors else (".bin", ".pt", ".pth", ".ckpt")
        return [f for f in self.sizes if f.endswith(ext) and "/" not in f]


def _weight_map(
    model_id: str,
    revision: Optional[str],
    token: Optional[str],
    guard: Optional[_Guard] = None,
    cache_dir: Optional[str] = None,
) -> dict[str, str]:
    """Map tensor name -> file, for either sharded or single-file checkpoints."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    local = os.path.isdir(model_id)

    def fetch(fn: str) -> Optional[str]:
        if local:
            p = os.path.join(model_id, fn)
            return p if os.path.exists(p) else None
        try:
            return hf_hub_download(model_id, fn, revision=revision, token=token,
                                   cache_dir=cache_dir)
        except EntryNotFoundError:
            return None
        except Exception:
            return None

    index = fetch("model.safetensors.index.json")
    if index:
        with open(index) as fh:
            return json.load(fh)["weight_map"]

    if guard is not None and "model.safetensors" in guard.sizes:
        # Listing a single-file checkpoint's tensors means downloading all of it.
        guard.before_download(["model.safetensors"])
    single = fetch("model.safetensors")
    if single:
        from safetensors import safe_open

        with safe_open(single, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    return {}


def _load_lazy(
    model_id: str,
    kind: MatrixKind,
    revision: Optional[str],
    token: Optional[str],
    guard: Optional[_Guard] = None,
    cache_dir: Optional[str] = None,
) -> Optional[tuple[torch.Tensor, str]]:
    """Pull just the embedding tensor from safetensors shards."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_map = _weight_map(model_id, revision, token, guard, cache_dir)
    if not weight_map:
        return None

    patterns = INPUT_EMBED_PATTERNS if kind == "input" else OUTPUT_EMBED_PATTERNS
    key = _match(weight_map.keys(), patterns)
    if key is None:
        if kind == "output":
            # Tied embeddings: the unembedding *is* the input table.
            key = _match(weight_map.keys(), INPUT_EMBED_PATTERNS)
            if key is None:
                return None
            note = "tied to input embeddings"
        else:
            return None
    else:
        note = ""

    shard = weight_map[key]
    if guard is not None:
        guard.before_download([shard])
    path = (
        os.path.join(model_id, shard)
        if os.path.isdir(model_id)
        else hf_hub_download(model_id, shard, revision=revision, token=token,
                             cache_dir=cache_dir)
    )
    with safe_open(path, framework="pt") as f:
        if guard is not None:
            guard.reserve_memory(guard.check_matrix(f.get_slice(key).get_shape()))
        tensor = f.get_tensor(key)
    src = f"safetensors:{key}" + (f" ({note})" if note else "")
    return tensor.to(torch.float32), src


def _load_full(
    model_id: str,
    kind: MatrixKind,
    dtype,
    device: str,
    revision: Optional[str],
    token: Optional[str],
    trust_remote_code: bool,
    need_hidden_states: bool,
    guard: Optional[_Guard] = None,
    cache_dir: Optional[str] = None,
):
    from transformers import AutoModel, AutoModelForCausalLM

    cls = AutoModelForCausalLM if (kind == "output" or need_hidden_states) else AutoModel
    kwargs = dict(
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    if guard is not None:
        files = guard.weight_files()
        guard.before_download(files)
        # The whole network lands in memory, at about its checkpoint size.
        guard.reserve_memory(sum(guard.sizes.get(f, 0) for f in files))
        if guard.limits.safetensors_only:
            # Belt and braces: transformers refuses rather than fall back to .bin.
            kwargs["use_safetensors"] = True
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        model = cls.from_pretrained(model_id, **kwargs)
    except Exception:
        if cls is AutoModelForCausalLM:
            raise
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model.eval()
    model.to(device)

    if kind == "input":
        mat = model.get_input_embeddings().weight
        src = "model.get_input_embeddings()"
    else:
        head = model.get_output_embeddings()
        if head is None:
            mat = model.get_input_embeddings().weight
            src = "get_input_embeddings() (no separate output head)"
        else:
            mat = head.weight
            src = "model.get_output_embeddings()"
    return model, mat.detach().to(torch.float32), src


def load(
    model_id: str,
    *,
    kind: MatrixKind = "input",
    device: str = "auto",
    dtype: str = "auto",
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    full: bool = False,
    center: bool = False,
    whiten: bool = False,
    limits: Optional[Limits] = None,
    cache_dir: Optional[str] = None,
) -> LoadedModel:
    """Load a tokenizer + token embedding matrix for `model_id`.

    `center` subtracts the vocabulary mean.  Do this before reading much into
    cosine similarities: LLM embedding clouds sit far off the origin, so raw
    cosines between arbitrary tokens are often 0.3-0.9 regardless of meaning.

    `whiten` additionally divides each coordinate by its std, equalising
    dimensions that differ by orders of magnitude in scale.

    `limits` lets a server refuse models before downloading their weights; see
    :class:`Limits`.  They apply to Hub repos only, not to local directories.
    `cache_dir` keeps downloads out of the shared Hugging Face cache.
    """
    from transformers import AutoTokenizer

    device = resolve_device(device)
    torch_dtype = DTYPES.get(dtype, None)
    guard = (
        _Guard(model_id, revision, hf_token, limits, cache_dir)
        if limits is not None and limits.active and not os.path.isdir(model_id)
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=hf_token, trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )

    n_layers = None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_id, revision=revision, token=hf_token,
            trust_remote_code=trust_remote_code, cache_dir=cache_dir,
        )
        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    except Exception:
        pass

    model = None
    matrix = None
    source = ""
    if not full:
        got = _load_lazy(model_id, kind, revision, hf_token, guard, cache_dir)
        if got is not None:
            matrix, source = got
    if matrix is None:
        model, matrix, source = _load_full(
            model_id, kind, torch_dtype, device, revision, hf_token,
            trust_remote_code, need_hidden_states=full, guard=guard,
            cache_dir=cache_dir,
        )
        if guard is not None:
            guard.check_matrix(matrix.shape)

    matrix = matrix.to(device)
    transforms: list[str] = []
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        transforms.append("centered")
    if whiten:
        matrix = matrix / matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
        transforms.append("whitened")

    return LoadedModel(
        name=model_id,
        tokenizer=tokenizer,
        embeddings=matrix.contiguous(),
        kind=kind,
        device=device,
        source=source,
        model=model,
        transforms=transforms,
        n_layers=n_layers,
    )
