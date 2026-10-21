"""A process-wide cache of loaded embedding matrices.

Loading is the expensive part of every request -- a first-time download can
take minutes and a resident matrix can be gigabytes -- so models are loaded
once in a background thread, shared by all users, and evicted least-recently-
used.  Requests never block on a download: they get ``status: loading`` and the
client polls.

Two budgets, both least-recently-used:

* **memory** -- resident matrices, by count (`max_models`) and optionally by
  bytes (`max_memory_bytes`).  Room is made before a matrix is read, not after.
* **disk** -- downloaded weights in a cache directory of eigentoken's own
  (`cache_dir` + `max_disk_bytes`).  Room is made before a download starts.
  A model that is loading or resident is never deleted.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from .. import loader


class ModelNotAllowed(PermissionError):
    """The operator's policy rules this model out (HTTP 403)."""


class LoadsBusy(Exception):
    """Too many loads already running or queued; try again shortly (HTTP 503)."""


class LoadRateLimited(Exception):
    """This client has started too many loads recently (HTTP 429)."""


@dataclass(frozen=True)
class Spec:
    """A fully qualified request for one embedding matrix.

    Two specs may name the same model -- that is how a model is compared with
    itself across, say, its input and output matrices, or centered vs raw.
    """

    model_id: str
    matrix: str = "input"
    center: bool = False
    whiten: bool = False
    layer: Optional[int] = None      # not None => the full network is needed

    @property
    def key(self) -> str:
        parts = [self.model_id, self.matrix]
        if self.center:
            parts.append("centered")
        if self.whiten:
            parts.append("whitened")
        if self.layer is not None:
            parts.append("full")
        return "|".join(parts)

    @property
    def label(self) -> str:
        bits = [self.model_id, self.matrix]
        if self.center:
            bits.append("centered")
        if self.whiten:
            bits.append("whitened")
        return " · ".join(bits)

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        mid = (d.get("model_id") or "").strip()
        if not mid:
            raise ValueError("model_id is required")
        matrix = d.get("matrix", "input")
        if matrix not in ("input", "output"):
            raise ValueError("matrix must be 'input' or 'output'")
        layer = d.get("layer")
        return cls(
            model_id=mid,
            matrix=matrix,
            center=bool(d.get("center", False)),
            whiten=bool(d.get("whiten", False)),
            layer=None if layer is None else int(layer),
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "matrix": self.matrix,
            "center": self.center,
            "whiten": self.whiten,
            "layer": self.layer,
            "key": self.key,
            "label": self.label,
        }


@dataclass
class Entry:
    spec: Spec
    status: str = "loading"          # loading | ready | error
    model: object = None
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    last_used: float = field(default_factory=time.time)
    cancelled: bool = False          # dropped while its load thread was still in flight
    queued: bool = False             # waiting for a free load slot
    refused: bool = False            # error is a policy refusal, not a failure
    nbytes: int = 0                  # memory held once ready
    note: str = ""                   # progress while loading, e.g. the download size

    def info(self) -> dict:
        d = {
            "key": self.spec.key,
            "spec": self.spec.to_dict(),
            "status": self.status,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }
        if self.status == "loading" and (self.queued or self.note):
            d["note"] = "queued behind other loads" if self.queued else self.note
        if self.status == "ready" and self.model is not None:
            d.update(
                bytes=self.nbytes,
                vocab_size=self.model.vocab_size,
                dim=self.model.dim,
                source=self.model.source,
                device=self.model.device,
                tokenizer=type(self.model.tokenizer).__name__,
                n_layers=self.model.n_layers,
                transforms=self.model.transforms,
                full_model_loaded=self.model.model is not None,
            )
        if self.status == "error":
            d["error"] = self.error
        return d


class ModelRegistry:
    """LRU cache of :class:`~eigentoken.loader.LoadedModel`, loaded off-thread."""

    def __init__(
        self,
        *,
        max_models: int = 3,
        device: str = "auto",
        dtype: str = "auto",
        allowlist: Optional[list[str]] = None,
        trust_remote_code: bool = False,
        hf_token: Optional[str] = None,
        allow_local_paths: bool = False,
        safetensors_only: bool = False,
        max_download_bytes: Optional[int] = None,
        max_matrix_bytes: Optional[int] = None,
        max_concurrent_loads: int = 0,
        max_queued_loads: int = 0,
        loads_per_client_hour: int = 0,
        max_memory_bytes: Optional[int] = None,
        cache_dir: Optional[str] = None,
        max_disk_bytes: Optional[int] = None,
        allow_unload: bool = False,
    ):
        """Zero or None for any limit means unlimited.

        `max_concurrent_loads` caps loads running at once; further loads wait in
        a queue of at most `max_queued_loads`, and beyond that requests are
        refused with :class:`LoadsBusy`.  `loads_per_client_hour` counts only
        requests that start a new load, so polling and resident models are free.

        `max_disk_bytes` needs `cache_dir`: the disk budget deletes whatever is
        least recently used in that directory, so it must not be a cache other
        tools share.
        """
        if max_disk_bytes and not cache_dir:
            raise ValueError("a disk budget needs a cache directory of eigentoken's own")
        self.max_models = max(1, max_models)
        self.device = device
        self.dtype = dtype
        self.allowlist = allowlist
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token
        self.allow_local_paths = allow_local_paths
        self.safetensors_only = safetensors_only
        self.max_download_bytes = max_download_bytes or None
        self.max_matrix_bytes = max_matrix_bytes or None
        self.max_concurrent_loads = max(0, max_concurrent_loads)
        self.max_queued_loads = max(0, max_queued_loads)
        self.loads_per_client_hour = max(0, loads_per_client_hour)
        self._slots = (
            threading.BoundedSemaphore(self.max_concurrent_loads)
            if self.max_concurrent_loads else None
        )
        self._client_loads: dict[str, deque] = {}
        self.max_memory_bytes = max_memory_bytes or None
        self.cache_dir = cache_dir
        self.max_disk_bytes = max_disk_bytes or None
        self.allow_unload = allow_unload
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        # Disk bookkeeping has its own lock: scanning the cache is slow, and
        # holding the main lock through it would stall every request.
        self._disk_lock = threading.Lock()
        self._disk_reserved: dict[str, int] = {}   # spec key -> bytes promised
        self._last_used: dict[str, float] = {}     # model id -> last request
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        # Tracks in-flight load threads by spec key, independent of `_entries` --
        # a drop() removes the visible entry immediately but the thread keeps
        # running, and this is what stops request() from starting a second one.
        self._loading: dict[str, Entry] = {}
        self._lock = threading.RLock()

    # -- policy ----------------------------------------------------------

    def check_allowed(self, model_id: str) -> None:
        if self.allowlist is not None:
            if model_id not in self.allowlist:
                raise ModelNotAllowed(
                    f"{model_id!r} is not in this server's allowed model list. "
                    "Ask the operator to add it, or start the server without "
                    "--allow to permit any model."
                )
            return                    # the operator listed it, local path or not
        if self.allow_local_paths:
            return
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        # A name that exists on disk would be loaded from there by transformers,
        # so it is refused even when it also looks like a repo id.
        try:
            validate_repo_id(model_id)
            ok = not os.path.exists(model_id)
        except HFValidationError:
            ok = False
        if not ok:
            raise ModelNotAllowed(
                f"{model_id!r} is not a Hugging Face model name. Use a repo id "
                "such as 'gpt2' or 'EleutherAI/pythia-160m'; this server does not "
                "load models from its own disk unless the operator lists them."
            )

    def _limits(self, entry: Entry) -> loader.Limits:
        def notify(msg: str) -> None:
            entry.note = msg

        return loader.Limits(
            safetensors_only=self.safetensors_only,
            max_download_bytes=self.max_download_bytes,
            max_matrix_bytes=self.max_matrix_bytes,
            notify=notify,
            before_fetch=(lambda n: self._make_disk_room(entry, n))
            if self.max_disk_bytes else None,
            before_matrix=(lambda n: self._make_memory_room(entry, n))
            if self.max_memory_bytes else None,
        )

    # -- memory budget ---------------------------------------------------

    @staticmethod
    def _model_bytes(m) -> int:
        n = m.embeddings.numel() * m.embeddings.element_size()
        if m.model is not None:
            n += sum(p.numel() * p.element_size() for p in m.model.parameters())
        return n

    def resident_bytes(self) -> int:
        with self._lock:
            return sum(e.nbytes for e in self._entries.values() if e.status == "ready")

    def _make_memory_room(self, entry: Entry, nbytes: int) -> None:
        cap = self.max_memory_bytes
        if nbytes > cap:
            raise loader.ModelRefused(
                f"{entry.spec.model_id} needs {loader._fmt_bytes(nbytes)} of memory; "
                f"this server keeps at most {loader._fmt_bytes(cap)} of models resident."
            )
        with self._lock:
            self._evict_locked(need=nbytes, keep=entry.spec.key)

    # -- disk budget -----------------------------------------------------

    def _make_disk_room(self, entry: Entry, nbytes: int) -> None:
        """Delete least recently used downloads until `nbytes` more fit."""
        from huggingface_hub import scan_cache_dir

        cap = self.max_disk_bytes
        if nbytes > cap:
            raise loader.ModelRefused(
                f"{entry.spec.model_id} needs {loader._fmt_bytes(nbytes)} of storage; "
                f"this server keeps at most {loader._fmt_bytes(cap)} of models on disk."
            )
        with self._disk_lock:
            repos = [r for r in scan_cache_dir(self.cache_dir).repos if r.repo_type == "model"]
            with self._lock:
                busy = {e.spec.model_id for e in self._entries.values()
                        if e.status in ("loading", "ready")}
                busy |= {e.spec.model_id for e in self._loading.values()}
                last = dict(self._last_used)
            used = sum(r.size_on_disk for r in repos)
            promised = sum(self._disk_reserved.values())
            # Last use: this process's own record, else the files' timestamps,
            # which carry the order across restarts.
            idle = sorted(
                (r for r in repos if r.repo_id not in busy),
                key=lambda r: max(last.get(r.repo_id, 0), r.last_accessed, r.last_modified),
            )
            while used + promised + nbytes > cap and idle:
                r = idle.pop(0)
                shutil.rmtree(r.repo_path, ignore_errors=True)
                used -= r.size_on_disk
                print(f"eigentoken: freed {loader._fmt_bytes(r.size_on_disk)} of storage "
                      f"by deleting {r.repo_id}", flush=True)
            if used + promised + nbytes > cap:
                raise loader.ModelRefused(
                    f"this server's model storage ({loader._fmt_bytes(cap)}) is full of "
                    "models in use or still downloading; try again in a few minutes."
                )
            self._disk_reserved[entry.spec.key] = nbytes

    def _admit_locked(self, client: Optional[str]) -> None:
        """Called with the lock held, before starting a new load."""
        if self.max_concurrent_loads:
            cap = self.max_concurrent_loads + self.max_queued_loads
            if len(self._loading) >= cap:
                raise LoadsBusy(
                    f"the server is already loading {len(self._loading)} models; "
                    "try again in a minute."
                )
        if self.loads_per_client_hour and client:
            now = time.time()
            recent = self._client_loads.setdefault(client, deque())
            while recent and now - recent[0] > 3600:
                recent.popleft()
            if len(recent) >= self.loads_per_client_hour:
                wait = int(3600 - (now - recent[0])) // 60 + 1
                raise LoadRateLimited(
                    f"you have started {len(recent)} model loads in the last hour, "
                    f"this server's limit; try again in about {wait} min, or use "
                    "a model that is already resident."
                )
            recent.append(now)
            # Forget clients with nothing recent, so the table can't grow forever.
            for c in [c for c, q in self._client_loads.items()
                      if not q or now - q[-1] > 3600]:
                del self._client_loads[c]

    # -- lifecycle -------------------------------------------------------

    def request(self, spec: Spec, *, retry: bool = False,
                client: Optional[str] = None) -> Entry:
        """Ensure `spec` is loading or loaded; return its entry immediately.

        A failure is **sticky**.  Clients poll this endpoint while a model
        loads, so clearing the error automatically would restart the download
        on every poll and never let the failure surface.  Pass ``retry`` (the
        UI's reload button) to try again.

        `client` identifies the caller for the per-client load limit.
        """
        self.check_allowed(spec.model_id)
        with self._lock:
            self._last_used[spec.model_id] = time.time()
            entry = self._entries.get(spec.key)
            if entry is not None:
                entry.last_used = time.time()
                self._entries.move_to_end(spec.key)
                if entry.status != "error" or not retry:
                    return entry
                del self._entries[spec.key]

            # A drop() may have hidden the entry for this key while its load
            # thread was still running.  Revive that entry instead of racing
            # a second thread against it -- the original download keeps going
            # and lands here when it finishes.
            revived = self._loading.get(spec.key)
            if revived is not None:
                revived.cancelled = False
                revived.status = "loading"
                revived.last_used = time.time()
                self._entries[spec.key] = revived
                self._entries.move_to_end(spec.key)
                self._evict_locked()
                return revived

            self._admit_locked(client)
            entry = Entry(spec=spec)
            if self._slots is not None:
                running = sum(1 for e in self._loading.values() if not e.queued)
                entry.queued = running >= self.max_concurrent_loads
            self._entries[spec.key] = entry
            self._loading[spec.key] = entry
            self._evict_locked()

        t = threading.Thread(target=self._load, args=(entry,), daemon=True)
        t.start()
        return entry

    def _load(self, entry: Entry) -> None:
        if self._slots is not None:
            self._slots.acquire()
        try:
            with self._lock:
                entry.queued = False
                if entry.cancelled:
                    return            # dropped while still queued: never start it
            try:
                m = loader.load(
                    entry.spec.model_id,
                    kind=entry.spec.matrix,
                    device=self.device,
                    dtype=self.dtype,
                    hf_token=self.hf_token,
                    trust_remote_code=self.trust_remote_code,
                    full=entry.spec.layer is not None,
                    center=entry.spec.center,
                    whiten=entry.spec.whiten,
                    limits=self._limits(entry),
                    cache_dir=self.cache_dir,
                )
            except loader.ModelRefused as e:  # a policy answer, not a crash
                with self._lock:
                    if not entry.cancelled:
                        entry.status = "error"
                        entry.refused = True
                        entry.error = str(e)
                        entry.finished = time.time()
                return
            except Exception as e:  # surfaced to the client verbatim
                with self._lock:
                    if not entry.cancelled:
                        entry.status = "error"
                        entry.error = f"{type(e).__name__}: {e}"
                        entry.finished = time.time()
                traceback.print_exc()
                return
            with self._lock:
                if entry.cancelled:
                    # Dropped while in flight and never re-requested: the
                    # download finished for nothing, so free it immediately
                    # instead of leaving it to be garbage-collected on its
                    # own time.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return
                entry.model = m
                entry.nbytes = self._model_bytes(m)
                entry.status = "ready"
                entry.finished = time.time()
                self._evict_locked(keep=entry.spec.key)
        finally:
            with self._lock:
                self._loading.pop(entry.spec.key, None)
            with self._disk_lock:
                self._disk_reserved.pop(entry.spec.key, None)
            if self._slots is not None:
                self._slots.release()

    def entry(self, key: str) -> Optional[Entry]:
        with self._lock:
            return self._entries.get(key)

    def listing(self) -> list[dict]:
        with self._lock:
            return [e.info() for e in reversed(self._entries.values())]

    def drop(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None and entry.status == "loading":
                # The download keeps running; `_loading` (left untouched)
                # is what stops request() from starting a second one, and
                # what discards the result if nobody asks for it again.
                entry.cancelled = True
        if entry is None:
            return False
        self._release(entry)
        return True

    def _over_locked(self, need: int) -> bool:
        if len(self._entries) > self.max_models:
            return True
        if self.max_memory_bytes:
            held = sum(e.nbytes for e in self._entries.values() if e.status == "ready")
            return held + need > self.max_memory_bytes
        return False

    def _evict_locked(self, need: int = 0, keep: Optional[str] = None) -> None:
        """Called with the lock held; frees the least recently used entries
        until the count and memory budgets hold, with `need` bytes to spare."""
        while self._over_locked(need):
            for k, e in list(self._entries.items()):
                if e.status == "loading" or k == keep:
                    continue          # never evict an in-flight load
                self._entries.pop(k)
                self._release(e)
                break
            else:
                return                # everything is still loading

    @staticmethod
    def _release(entry: Entry) -> None:
        entry.model = None
        entry.status = "evicted"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
