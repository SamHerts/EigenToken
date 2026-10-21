"""End-to-end smoke tests. Run: python tests/smoke.py [model_id]

Uses a real model (gpt2 by default) because every interesting failure in this
tool involves a real tokenizer or a real checkpoint layout.
"""

import sys

import torch

import eigentoken as et
from eigentoken.cli import main

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
failures = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


print(f"loading {MODEL}")
m = et.load(MODEL, center=True)
print(m.describe())

print("\nresolution")
king = et.resolve(m, "king")
queen = et.resolve(m, "queen")
check("king is a single token", king.single_token, str(king.token_ids))
check("id: form addresses a row directly",
      et.resolve(m, f"id:{king.token_ids[0]}").token_ids == king.token_ids)
check("multi-token words pool without error",
      et.resolve(m, "antidisestablishmentarianism").vector.shape == (m.dim,))

print("\ngeometry")
s = et.pair_stats(king.vector, queen.vector)
check("cosine in range", -1 <= s.cosine <= 1, f"{s.cosine:.4f}")
check("self-similarity is 1", abs(et.pair_stats(king.vector, king.vector).cosine - 1) < 1e-5)
check("king closer to queen than to a random token",
      s.cosine > et.pair_stats(king.vector, m.embeddings[42]).cosine)

print("\nneighbours")
nb = et.neighbors(king.vector, m.embeddings, k=5, exclude=king.token_ids)
check("returns k results", len(nb) == 5)
check("scores descend", all(nb[i][1] >= nb[i + 1][1] for i in range(len(nb) - 1)))
check("excluded ids absent", all(t not in king.token_ids for t, _ in nb))

print("\ncase/space variant filtering")
variant_ids = et.case_variant_ids(m.tokenizer, "king")
check("finds king's own casing/space variants", king.token_ids[0] in variant_ids,
      str(variant_ids))
check("finds more than just the base form", len(variant_ids) > 1, str(len(variant_ids)))
nb_filtered = et.neighbors(king.vector, m.embeddings, k=5,
                           exclude=king.token_ids + variant_ids)
check("filtered neighbours contain no casing/space variant of king",
      all(t not in variant_ids for t, _ in nb_filtered),
      str([t for t, _ in nb_filtered]))

print("\nanalogy")
man, woman = et.resolve(m, "man"), et.resolve(m, "woman")
target = et.analogy_vector(man.vector, king.vector, woman.vector)
hits = et.neighbors(target, m.embeddings, k=5,
                    exclude=man.token_ids + king.token_ids + woman.token_ids)
top = [m.tokenizer.decode([t]).strip().lower() for t, _ in hits]
check("queen ranks top-5 for man:king::woman:?", "queen" in top, str(top))
par = et.pair_direction_similarity(man.vector, king.vector, woman.vector, queen.vector)
check("gender offsets are parallel", par > 0.3, f"{par:.3f}")

print("\naxis")
pairs = [("man", "woman"), ("king", "queen"), ("father", "mother"), ("actor", "actress")]
axis = et.build_axis(
    [(et.resolve(m, a).vector, et.resolve(m, b).vector) for a, b in pairs],
    [f"{a}->{b}" for a, b in pairs],
)
check("axis direction is unit length", abs(axis.direction.norm().item() - 1) < 1e-5)
check("mean aligns with PC1 when coherent", axis.pc1_alignment > 0.9,
      f"{axis.pc1_alignment:.3f}")
check("woman projects positive, man negative",
      et.project(woman.vector.unsqueeze(0), axis.direction).item()
      > et.project(man.vector.unsqueeze(0), axis.direction).item())
check("rejection removes the axis component",
      abs(torch.dot(et.reject(queen.vector, axis.direction), axis.direction).item()) < 1e-4)

print("\nattribution + pca")
att = et.dim_attribution(king.vector, queen.vector, 5)
check("delta shares sum to 1",
      abs(sum(d.delta_share for d in att["divergent"])) <= 1.0)
check("effective dims within range", 1 <= att["effective_dims"] <= m.dim,
      f"{att['effective_dims']:.0f}/{m.dim}")
X = torch.stack([et.resolve(m, w).vector for w in
                 ["king", "queen", "man", "woman", "paris", "tokyo"]])
coords, ratio, _ = et.pca(X, 2)
check("pca shape", coords.shape == (6, 2))
check("explained variance descends and is bounded",
      ratio[0] >= ratio[1] and ratio.sum() <= 1.0 + 1e-5)

print("\ncross-model self-comparison (a model must agree with itself)")
words = ["king", "queen", "man", "woman", "paris", "tokyo", "dog", "cat"]
V = torch.stack([et.resolve(m, w).vector for w in words]).cpu()
r = et.rsa(words, V, V)
check("RSA with itself is 1", abs(r.spearman - 1) < 1e-6, f"{r.spearman:.6f}")
pr = et.procrustes(words, V, V)
check("procrustes with itself is 0", pr.disparity < 1e-6, f"{pr.disparity:.2e}")

print("\nbias (WEAT)")
preset = et.BIAS_PRESETS["flowers_insects"]


def resolve_group(words):
    return torch.stack([et.resolve(m, w).vector for w in words])


wr = et.weat(
    resolve_group(preset["x"]), resolve_group(preset["y"]),
    resolve_group(preset["a"]), resolve_group(preset["b"]),
    preset["x"], preset["y"],
    preset["x_label"], preset["y_label"], preset["a_label"], preset["b_label"],
)
check("flowers/insects is a large, well-known effect", wr.effect_size > 0.8,
      f"d={wr.effect_size:.3f} ({wr.magnitude})")
check("the effect is significant", wr.p_value < 0.05, f"p={wr.p_value:.4f}")
check("every word gets a score", len(wr.word_scores) == len(preset["x"]) + len(preset["y"]))

print("\nCLI")
check("compare exits 0", main(["compare", "-m", MODEL, "--center", "king", "queen"]) == 0)
check("json exits 0", main(["info", "-m", MODEL, "--json"]) == 0)
check("bias exits 0", main(["bias", "-m", MODEL, "--preset", "flowers_insects", "--json"]) == 0)
check("neighbors --exclude-variants exits 0",
      main(["neighbors", "-m", MODEL, "king", "--exclude-variants", "--json"]) == 0)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
