"""Load-limit tests for the server's model registry. Run: python tests/limits.py

Needs no running server, but does talk to the Hub: the refusals are checked
against real repos, and the size checks must happen *before* any weights are
downloaded, which only a real repo can show.
"""

import os
import sys
import tempfile
import time

from huggingface_hub import try_to_load_from_cache

from eigentoken.server.registry import (
    LoadRateLimited, LoadsBusy, ModelNotAllowed, ModelRegistry, Spec,
)

failures = []

PICKLE_ONLY = "sshleifer/tiny-gpt2"      # pytorch_model.bin and nothing else
HUGE = "Qwen/Qwen2.5-72B"                # its embedding sits in a ~3.8 GB shard
WIDE = "Qwen/Qwen3-1.7B"                 # 151936 x 2048 float32 = 1.24 GB


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def raises(exc, fn):
    try:
        fn()
    except exc as e:
        return str(e)
    except Exception as e:
        return f"wrong exception {type(e).__name__}: {e}"
    return None


def wait(reg, spec, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        e = reg.request(spec)
        if e.status != "loading":
            return e
        time.sleep(0.5)
    raise TimeoutError(spec.key)


print("model names")
reg = ModelRegistry()
for bad in ["/etc", "../secrets", "~/models/x", "a/b/c", "src", "tests"]:
    msg = raises(ModelNotAllowed, lambda: reg.check_allowed(bad))
    check(f"refuses {bad!r}", msg is not None and "not a Hugging Face model name" in msg)
for good in ["gpt2", "EleutherAI/pythia-160m"]:
    check(f"accepts {good!r}", raises(Exception, lambda: reg.check_allowed(good)) is None)
with tempfile.TemporaryDirectory() as d:
    listed = ModelRegistry(allowlist=[d])
    check("an --allow'ed local path is accepted",
          raises(Exception, lambda: listed.check_allowed(d)) is None)
    check("--allow-local-paths accepts a directory",
          raises(Exception, lambda: ModelRegistry(allow_local_paths=True).check_allowed(d)) is None)

print("weight format")
strict = ModelRegistry(safetensors_only=True)
# The next check loads it with pickles allowed, so a re-run finds it cached.
was_cached = isinstance(try_to_load_from_cache(PICKLE_ONLY, "pytorch_model.bin"), str)
e = wait(strict, Spec(PICKLE_ONLY))
check("a pickle-only repo is refused", e.status == "error" and e.refused, e.error)
check("refused before downloading its weights",
      was_cached or not isinstance(try_to_load_from_cache(PICKLE_ONLY, "pytorch_model.bin"), str),
      "(cached by an earlier run)" if was_cached else "")
e = wait(ModelRegistry(), Spec(PICKLE_ONLY))
check("loads when pickles are allowed", e.status == "ready", e.error)
e = wait(strict, Spec("gpt2", layer=0))
check("full-model path loads safetensors", e.status == "ready", e.error)

print("sizes")
small = ModelRegistry(max_download_bytes=int(1e9))
e = wait(small, Spec(HUGE))
check("an oversized shard is refused", e.status == "error" and e.refused, e.error)
shard = "model-00001-of-00037.safetensors"
check("refused before downloading it",
      not isinstance(try_to_load_from_cache(HUGE, shard), str))
e = wait(ModelRegistry(max_download_bytes=int(0.1e9)), Spec("gpt2"))
check("a single-file checkpoint is size-checked too", e.refused, e.error)
e = wait(ModelRegistry(max_matrix_bytes=int(1e9)), Spec(WIDE))
check("an oversized matrix is refused", e.refused and "151936 x 2048" in e.error, e.error)
noted = []
reg = ModelRegistry(max_download_bytes=int(10e9))
spec = Spec("google/bert_uncased_L-2_H-128_A-2")
cached = isinstance(try_to_load_from_cache(spec.model_id, "model.safetensors"), str)
entry = reg.request(spec)
while entry.status == "loading":
    if entry.info().get("note"):
        noted.append(entry.info()["note"])
    time.sleep(0.02)
check("a fresh download reports its size",
      cached or any("downloading" in n for n in noted),
      "(already cached, nothing to report)" if cached else str(noted[:1]))

print("load caps")
reg = ModelRegistry(max_concurrent_loads=1, max_queued_loads=1)
a = reg.request(Spec("gpt2", center=True))
b = reg.request(Spec("gpt2", center=False))
check("a second load waits in the queue", b.queued and b.info().get("note") == "queued behind other loads")
check("a third is refused while both are pending",
      raises(LoadsBusy, lambda: reg.request(Spec("distilgpt2"))) is not None)
check("polling a pending load is not refused", reg.request(Spec("gpt2", center=True)) is a)
wait(reg, Spec("gpt2", center=False))
check("the queue drains", a.status == "ready" and b.status == "ready")
check("a new load is accepted once there is room",
      raises(Exception, lambda: reg.request(Spec("distilgpt2"))) is None)

print("per-client rate")
reg = ModelRegistry(loads_per_client_hour=2)
reg.request(Spec("gpt2", center=True), client="1.2.3.4")
reg.request(Spec("gpt2", center=False), client="1.2.3.4")
check("a client's third new load is refused",
      raises(LoadRateLimited, lambda: reg.request(Spec("distilgpt2"), client="1.2.3.4")) is not None)
check("re-requesting a model it started is free",
      raises(Exception, lambda: reg.request(Spec("gpt2", center=True), client="1.2.3.4")) is None)
check("another client is unaffected",
      raises(Exception, lambda: reg.request(Spec("distilgpt2"), client="5.6.7.8")) is None)

print("memory budget")
GPT2_BYTES = 50257 * 768 * 4
reg = ModelRegistry(max_models=10, max_memory_bytes=int(2.5 * GPT2_BYTES))
first, second, third = Spec("gpt2", center=True), Spec("gpt2"), Spec("distilgpt2")
wait(reg, first)
wait(reg, second)
check("two matrices fit", reg.resident_bytes() == 2 * GPT2_BYTES,
      f"{reg.resident_bytes()} bytes")
reg.request(first)                               # now `second` is the LRU
wait(reg, third)
check("a third evicts the least recently used",
      reg.entry(second.key) is None and reg.entry(first.key) is not None)
check("and the budget holds", reg.resident_bytes() <= reg.max_memory_bytes)
e = wait(ModelRegistry(max_memory_bytes=int(1e8)), Spec("gpt2"))
check("a matrix bigger than the whole budget is refused", e.refused, e.error)

print("disk budget")
# bert-tiny's weights are 17.7 MB; tiny-random-gpt2's are 0.5 MB.
BERT, TINY = "google/bert_uncased_L-2_H-128_A-2", "hf-internal-testing/tiny-random-gpt2"
with tempfile.TemporaryDirectory() as cache:
    held = lambda: {r.name for r in os.scandir(cache) if r.name.startswith("models--")}
    reg = ModelRegistry(max_models=1, cache_dir=cache, max_disk_bytes=int(18e6))
    check("bert-tiny downloads into its own cache", wait(reg, Spec(BERT)).status == "ready")
    e = wait(reg, Spec(TINY))
    check("the next download deletes the least recently used model",
          e.status == "ready" and held() == {"models--hf-internal-testing--tiny-random-gpt2"},
          e.error or str(held()))
    reg = ModelRegistry(max_models=2, cache_dir=cache, max_disk_bytes=int(18e6))
    wait(reg, Spec(TINY))
    e = wait(reg, Spec(BERT))
    check("a resident model is never deleted to make room",
          e.refused and "models--hf-internal-testing--tiny-random-gpt2" in held(), e.error)
check("a disk budget without a cache dir of its own is refused",
      raises(ValueError, lambda: ModelRegistry(max_disk_bytes=int(1e9))) is not None)

print("unloading")
from fastapi.testclient import TestClient
from eigentoken.server import create_app

client = TestClient(create_app(ModelRegistry()))
r = client.post("/api/models/unload", json={"key": "gpt2|input"})
check("visitors can't unload by default", r.status_code == 403)
r = TestClient(create_app(ModelRegistry(allow_unload=True))).post(
    "/api/models/unload", json={"key": "gpt2|input"})
check("--allow-unload turns it back on", r.status_code == 200)

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("all limit checks passed")
sys.stdout.flush()
os._exit(0)  # don't wait on daemon load threads left running (skips the usual flush)
