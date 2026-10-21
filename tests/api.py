"""API tests against a live server. Run: python tests/api.py [base_url]

Starts nothing itself -- point it at a running `eigentoken serve`, which listens
on port 8077 by default. Uses gpt2 and pythia-160m, small enough for CI.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077").rstrip("/")
GPT2 = {"model_id": "gpt2", "matrix": "input", "center": True}
GPT2_OUT = {"model_id": "gpt2", "matrix": "output", "center": True}
GPT2_RAW = {"model_id": "gpt2", "matrix": "input", "center": False}
PYTHIA = {"model_id": "EleutherAI/pythia-160m", "matrix": "input", "center": True}

failures = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def call(path, body=None, *, wait=True, timeout=600):
    """POST (or GET), transparently waiting out 202 model loads."""
    deadline = time.time() + timeout
    while True:
        req = urllib.request.Request(
            BASE + "/api" + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read())
                code = r.status
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
        if code == 202 and wait:
            if time.time() > deadline:
                raise TimeoutError(f"{path} never finished loading")
            time.sleep(1.5)
            continue
        return code, data


print(f"testing {BASE}\n")

print("service")
code, health = call("/health")
check("health responds", code == 200 and health["ok"], f"cuda={health.get('cuda')}")
code, models = call("/models")
check("model listing has suggestions", code == 200 and len(models["suggested"]) > 0)

print("\ntokens")
code, d = call("/tokens", {"spec": GPT2, "words": ["king", "antidisestablishmentarianism"]})
check("single-token word is flagged single", d["words"][0]["single_token"])
check("long word is split and flagged", not d["words"][1]["single_token"],
      f"{len(d['words'][1]['token_ids'])} pieces")
check("multi-token word carries a warning", bool(d["words"][1]["warning"]))
check("variants are reported", len(d["words"][0]["variants"]) >= 1)

print("\ncompare")
code, d = call("/compare", {"specs": [GPT2], "words": ["king", "queen", "man", "woman"], "dims": 5})
pairs = {(p["a"], p["b"]): p for p in d["results"][0]["pairs"]}
check("all pairs returned", len(pairs) == 6)
check("king~queen beats king~woman",
      pairs[("king", "queen")]["cosine"] > pairs[("king", "woman")]["cosine"],
      f"{pairs[('king', 'queen')]['cosine']:.3f} vs {pairs[('king', 'woman')]['cosine']:.3f}")
check("cosine matrix has a unit diagonal",
      all(abs(row[i] - 1) < 1e-4 for i, row in enumerate(d["results"][0]["cosine_matrix"])))
check("dimension attribution present", "attribution" in d["results"][0]["pairs"][0])

print("\ncentering changes the numbers")
code, d = call("/compare", {"specs": [GPT2, GPT2_RAW], "words": ["king", "queen"]})
cen, raw = (r["pairs"][0]["cosine"] for r in d["results"])
check("raw cosine is inflated by the shared mean", raw > cen, f"raw {raw:.3f} > centered {cen:.3f}")

print("\nsame model against itself")
code, d = call("/compare", {"specs": [GPT2, GPT2_OUT], "words": ["king", "queen"]})
check("two slots of one model both resolve", code == 200 and len(d["results"]) == 2)
check("gpt2's output matrix is reported as tied", d["results"][1]["meta"]["tied"])
code, d = call("/crossmodel", {
    "specs": [GPT2, GPT2_OUT],
    "words": ["king", "queen", "man", "woman", "paris", "tokyo"],
    "single_token_only": True,
})
check("identical matrices are flagged", d["identical_matrices"])
check("a tied model agrees with itself exactly",
      abs(d["comparisons"][0]["rsa_spearman"] - 1) < 1e-9,
      f"rsa={d['comparisons'][0]['rsa_spearman']}")

print("\nneighbours")
code, d = call("/neighbors", {"specs": [GPT2], "words": ["king"], "k": 12})
nb = d["results"][0]["words"][0]["neighbors"]
toks = [x["token"].lstrip("\u2423").lower() for x in nb]
check("returns k neighbours", len(nb) == 12)
check("scores descend", all(nb[i]["score"] >= nb[i + 1]["score"] for i in range(len(nb) - 1)))
check("the query token is excluded", "king" not in toks[:1] or nb[0]["token_id"] != 5822)
check("royal words dominate king's neighbourhood",
      len({"queen", "kings", "monarch", "prince", "royal", "throne"} & set(toks)) >= 3,
      " ".join(toks[:6]))

code, d = call("/neighbors", {
    "specs": [GPT2], "words": ["king"], "k": 12, "exclude_variants": True,
})
toks_v = [x["token"].lstrip("␣").lower() for x in d["results"][0]["words"][0]["neighbors"]]
check("exclude_variants drops KING/King/' King' from king's neighbours",
      "king" not in toks_v, " ".join(toks_v[:6]))

print("\ncloud (associated words)")
code, d = call("/cloud", {"spec": GPT2, "word": "king", "k": 100, "clusters": 5, "mode": "radial"})
pts = d["points"]
check("returns k points", len(pts) == 100)
check("every point carries a group", all(0 <= p["cluster"] < 5 for p in pts))
check("radius equals cosine distance, rescaled", all(
    abs((p["x"] ** 2 + p["y"] ** 2) ** 0.5
        - (1 - p["cosine"]) / max(1 - q["cosine"] for q in pts)) < 1e-4
    for p in pts))
check("similarities descend with rank",
      all(pts[i]["cosine"] >= pts[i + 1]["cosine"] - 1e-6 for i in range(len(pts) - 1)))
groups = {p["cluster"] for p in pts}
check("clustering actually splits the neighbourhood", len(groups) > 1, f"{len(groups)} groups")

code, d2 = call("/cloud", {"spec": GPT2, "word": "id:5822", "k": 10})
check("id: addresses a token row directly", d2["center"]["token_ids"] == [5822])
code, d3 = call("/cloud", {"spec": GPT2, "word": "tok:Ġking", "k": 10})
check("tok: addresses a vocabulary entry directly", d3["center"]["token_ids"] == [5822])

code, d4 = call("/cloud", {"spec": GPT2, "word": "king", "k": 10, "exclude_variants": True})
cloud_toks = [p["token"].lstrip("␣").lower() for p in d4["points"]]
check("exclude_variants keeps king's own casing/spacing out of its own cloud",
      "king" not in cloud_toks, " ".join(cloud_toks[:6]))

print("\nanalogy")
code, d = call("/analogy", {"specs": [GPT2], "a": "man", "b": "king", "c": "woman",
                            "expect": "queen", "k": 5})
r = d["results"][0]
check("queen is the top candidate", r["candidates"][0]["token"].lstrip("\u2423").lower() == "queen",
      r["candidates"][0]["token"])
check("expected answer ranks first", r["expected"]["rank"] == 1)
check("offsets are parallel", r["expected"]["offset_parallelism"] > 0.3,
      f"{r['expected']['offset_parallelism']:.3f}")

print("\naxis")
code, d = call("/axis", {
    "specs": [GPT2],
    "pairs": ["man:woman", "king:queen", "father:mother", "actor:actress", "brother:sister"],
    "probes": ["nurse", "engineer"], "extremes": 10,
})
r = d["results"][0]
check("axis is judged usable", r["usable"], f"coherence={r['coherence']:.3f}")
check("mean aligns with PC1", r["pc1_alignment"] > 0.9, f"{r['pc1_alignment']:.3f}")
check("PC1 share beats chance", r["pc1_ratio"] > r["chance_pc1"],
      f"{r['pc1_ratio']:.3f} vs {r['chance_pc1']:.3f}")
hi = [x["token"].lstrip("\u2423").lower() for x in r["extremes"]["high"]]
lo = [x["token"].lstrip("\u2423").lower() for x in r["extremes"]["low"]]
check("the positive end is female-coded", "woman" in hi or "women" in hi, " ".join(hi[:4]))
check("the negative end is male-coded", "man" in lo or "he" in lo, " ".join(lo[:4]))

print("\nbias (WEAT)")
code, d = call("/bias/presets")
keys = {p["key"] for p in d["presets"]}
check("all four built-in presets are listed",
      keys == {"flowers_insects", "career_family", "science_arts", "math_arts"}, str(keys))

code, d = call("/bias", {"specs": [GPT2], "preset": "flowers_insects"})
r = d["results"][0]
check("flowers/insects is a large, well-known effect", r["effect_size"] > 0.8,
      f"d={r['effect_size']:.3f} ({r['magnitude']})")
check("the effect is significant", r["p_value"] < 0.05, f"p={r['p_value']:.4f}")
check("every word gets a score", len(r["word_scores"]) == 20, str(len(r["word_scores"])))

code, d = call("/bias", {
    "specs": [GPT2],
    "x": ["dog", "cat"], "y": ["truck", "car"], "a": ["good", "nice"], "b": ["bad", "awful"],
    "x_label": "animals", "y_label": "vehicles",
})
check("custom word lists override the preset", code == 200)
check("custom labels round-trip", d["results"][0]["x_label"] == "animals")

print("\ncross-model")
code, d = call("/crossmodel", {
    "specs": [GPT2, PYTHIA],
    "words": ["king", "queen", "man", "woman", "paris", "tokyo", "dog", "cat",
              "red", "blue", "one", "two"],
    "neighbors": 10, "single_token_only": True,
})
c = d["comparisons"][0]
check("two differently-tokenized models compare", code == 200)
check("RSA is in range", -1 <= c["rsa_spearman"] <= 1, f"{c['rsa_spearman']:.3f}")
check("unrelated models still agree somewhat", c["rsa_spearman"] > 0.2)
check("neighbour overlap reported", 0 <= c["overlap"]["mean_jaccard"] <= 1,
      f"{c['overlap']['mean_jaccard']:.3f}")
check("divergent words listed", len(c["divergent_words"]) > 0)

print("\nvalidation")
cases = [
    ("missing model id", "/compare", {"specs": [{"model_id": ""}], "words": ["a", "b"]}, 400),
    ("bad matrix", "/compare",
     {"specs": [{"model_id": "gpt2", "matrix": "middle"}], "words": ["a", "b"]}, 400),
    ("one word", "/compare", {"specs": [GPT2], "words": ["king"]}, 400),
    ("one model for crossmodel", "/crossmodel", {"specs": [GPT2], "words": list("abcd")}, 400),
    ("too few words", "/crossmodel", {"specs": [GPT2, PYTHIA], "words": ["a", "b"]}, 400),
    ("one axis pair", "/axis", {"specs": [GPT2], "pairs": ["man:woman"]}, 400),
    ("malformed pair", "/axis", {"specs": [GPT2], "pairs": ["manwoman", "a:b"]}, 400),
    ("no specs for bias", "/bias", {"preset": "flowers_insects"}, 400),
    ("unknown preset", "/bias", {"specs": [GPT2], "preset": "nope"}, 400),
    ("no preset and no custom lists", "/bias", {"specs": [GPT2]}, 400),
    ("too few words in a bias set", "/bias",
     {"specs": [GPT2], "x": ["a"], "y": ["b", "c"], "a": ["d", "e"], "b": ["f", "g"]}, 400),
    ("bad metric", "/neighbors", {"specs": [GPT2], "words": ["king"], "metric": "cosmic"}, 400),
    ("unknown token", "/cloud", {"spec": GPT2, "word": "tok:NOTATOKEN"}, 400),
    ("empty word", "/cloud", {"spec": GPT2, "word": "  "}, 400),
]
for name, path, body, want in cases:
    code, data = call(path, body, wait=False)
    check(f"{name} → {want}", code == want, f"got {code} {str(data)[:70]}")

code, data = call("/cloud", {"spec": {"model_id": "nope/does-not-exist"}, "word": "king"},
                  wait=False)
check("a bad model id starts loading, not a crash", code in (202, 502))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all API checks passed")
