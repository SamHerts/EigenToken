"""Command line interface for eigentoken."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Optional, Sequence

import torch
from dotenv import load_dotenv

from . import bias, crossmodel, geometry, loader, vocab
from .report import bar, heading, num, sub, table
from .vocab import token_label

# Picks up HF_TOKEN (among others) from a .env file in the working directory
# or any parent of it, before anything reads os.environ. huggingface_hub
# already falls back to $HF_TOKEN on its own when no --hf-token is given, so
# this is the only wiring gated models need.
load_dotenv()

# Not 8000: that collides with the default port of nearly every local model
# server, which is exactly what tends to be running alongside this.
DEFAULT_PORT = 8077


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def add_model_args(p: argparse.ArgumentParser, *, multi: bool = False) -> None:
    if multi:
        p.add_argument(
            "-m", "--model", action="append", required=True, metavar="ID",
            help="model id or local path; repeat to compare several models",
        )
    else:
        p.add_argument(
            "-m", "--model", required=True, metavar="ID",
            help="HuggingFace model id or local path",
        )
    p.add_argument(
        "--matrix", choices=["input", "output"], default="input",
        help="input embedding table (default) or the output/unembedding matrix",
    )
    p.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto (default)")
    p.add_argument(
        "--dtype", default="auto", choices=list(loader.DTYPES),
        help="dtype used when the full model must be loaded",
    )
    p.add_argument(
        "--center", action="store_true",
        help="subtract the vocabulary mean before measuring (see `eigentoken info`)",
    )
    p.add_argument(
        "--whiten", action="store_true",
        help="also divide each coordinate by its standard deviation",
    )
    p.add_argument(
        "--pooling", default="mean", choices=["mean", "sum", "first", "last", "norm-mean"],
        help="how to combine subwords when a word is not a single token",
    )
    p.add_argument(
        "--no-space", dest="prefer_space", action="store_false",
        help="do not prefer the leading-space form (' king') when resolving words",
    )
    p.add_argument("--revision", default=None, help="model revision / branch / commit")
    p.add_argument("--hf-token", default=None, help="HuggingFace access token")
    p.add_argument(
        "--trust-remote-code", action="store_true",
        help="allow the model repo to execute its own modelling code",
    )
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="show model loading detail")


def add_context_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--layer", type=int, default=None, metavar="N",
        help="use contextual hidden states from layer N instead of the static "
             "embedding table (0 = embeddings, -1 = final layer). Requires --context "
             "and loads the full model.",
    )
    p.add_argument(
        "--context", action="append", default=None, metavar="SENTENCE",
        help="sentence containing the word(s); give once for all, or once per word",
    )


def open_model(args, model_id: Optional[str] = None, *, full: bool = False):
    return loader.load(
        model_id or args.model,
        kind=args.matrix,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        full=full or getattr(args, "layer", None) is not None,
        center=args.center,
        whiten=args.whiten,
    )


def contexts_for(args, words: Sequence[str]) -> list[str]:
    ctx = args.context or []
    if not ctx:
        raise SystemExit("--layer requires --context (once for all words, or once per word)")
    if len(ctx) == 1:
        return list(ctx) * len(words)
    if len(ctx) != len(words):
        raise SystemExit(
            f"got {len(ctx)} --context values for {len(words)} words; "
            "give exactly one, or one per word"
        )
    return list(ctx)


class Vec:
    """A resolved word: its vector plus how it was obtained."""

    def __init__(self, query, vector, label, token_ids, warning=None, detail=None):
        self.query = query
        self.name = query          # disambiguated when a word is repeated
        self.vector = vector
        self.label = label
        self.token_ids = list(token_ids)
        self.warning = warning
        self.detail = detail or {}


def _disambiguate(vecs: list["Vec"]) -> list["Vec"]:
    """Repeating a word (two contexts for 'bank') must not collapse the labels."""
    counts: dict[str, int] = {}
    for v in vecs:
        counts[v.query] = counts.get(v.query, 0) + 1
    seen: dict[str, int] = {}
    for v in vecs:
        if counts[v.query] > 1:
            seen[v.query] = seen.get(v.query, 0) + 1
            v.name = f"{v.query}#{seen[v.query]}"
    return vecs


def resolve_words(args, model, words: Sequence[str]) -> list[Vec]:
    """Static-table or contextual resolution, depending on --layer."""
    if getattr(args, "layer", None) is None:
        out = []
        for w in words:
            r = vocab.resolve(
                model, w, pooling=args.pooling, prefer_space=args.prefer_space
            )
            out.append(
                Vec(
                    w, r.vector, r.label, r.token_ids, r.warning,
                    {
                        "surface": r.surface,
                        "tokens": [token_label(model.tokenizer, i) for i in r.token_ids],
                        "single_token": r.single_token,
                    },
                )
            )
        return out

    from .contextual import embed_in_context

    ctxs = contexts_for(args, words)
    out = []
    for w, c in zip(words, ctxs):
        cv = embed_in_context(model, w, c, layer=args.layer, pooling=args.pooling)
        out.append(
            Vec(
                w, cv.vector, cv.label, cv.token_ids, None,
                {
                    "context": c,
                    "layer": cv.layer,
                    "tokens": [token_label(model.tokenizer, i) for i in cv.token_ids],
                },
            )
        )
    return _disambiguate(out)


def emit(args, payload: dict, text: str) -> None:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(text)


def warn_all(vecs: Sequence[Vec]) -> None:
    for v in vecs:
        if v.warning:
            print(f"  ! {v.warning}", file=sys.stderr)


def search_space(args, model):
    """Which matrix nearest-neighbour lookups should search.

    In contextual mode the query vector lives in the residual stream, so the
    right comparison set is the unembedding matrix -- this is the 'logit lens'
    reading of the hidden state, not a lookup in the input table.
    """
    if getattr(args, "layer", None) is None:
        return model.embeddings, model.kind + " embeddings"
    head = model.model.get_output_embeddings() if model.model is not None else None
    if head is None:
        return model.embeddings, model.kind + " embeddings (no output head)"
    E = head.weight.detach().float().to(model.device)
    return E, "output embeddings (logit lens)"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_info(args) -> None:
    m = open_model(args)
    iso = geometry.isotropy(m.embeddings)
    tok = m.tokenizer
    payload = {
        "model": m.name,
        "matrix": m.kind,
        "source": m.source,
        "vocab_size": m.vocab_size,
        "dim": m.dim,
        "tokenizer": type(tok).__name__,
        "transforms": m.transforms,
        "isotropy": iso,
    }
    lines = [heading(f"{m.name}")]
    lines.append(
        table(
            ["property", "value"],
            [
                ["matrix", f"{m.kind} ({m.source})"],
                ["shape", f"{m.vocab_size} x {m.dim}"],
                ["tokenizer", type(tok).__name__],
                ["device", m.device],
                ["transforms", ", ".join(m.transforms) or "none"],
            ],
        )
    )
    lines.append(sub("embedding cloud geometry"))
    lines.append(
        table(
            ["statistic", "value", "meaning"],
            [
                ["mean pairwise cosine", num(iso["mean_cosine"]),
                 "0 = isotropic; high = every token looks similar"],
                ["mean token norm", num(iso["mean_norm"], 3), "typical vector length"],
                ["mean-vector norm", num(iso["mean_vector_norm"], 3),
                 "how far the cloud sits from the origin"],
                ["offset ratio", num(iso["offset_ratio"], 3),
                 "> 1: pass --center before trusting cosines"],
            ],
        )
    )
    if iso["mean_cosine"] > 0.15 and not m.transforms:
        lines.append(
            "\nNote: this space is anisotropic. Re-run with --center so that "
            "similarities reflect meaning rather than the shared mean vector."
        )
    emit(args, payload, "\n".join(lines))


def cmd_tokens(args) -> None:
    m = open_model(args)
    payload = {"model": m.name, "words": []}
    lines = [heading(f"tokenization in {m.name}")]
    rows = []
    for w in args.words:
        r = vocab.resolve(m, w, pooling=args.pooling, prefer_space=args.prefer_space)
        labels = [token_label(m.tokenizer, i) for i in r.token_ids]
        rows.append([
            w,
            "yes" if r.single_token else f"no ({len(r.token_ids)})",
            repr(r.surface),
            " ".join(labels),
            ",".join(str(i) for i in r.token_ids),
            num(r.vector.norm().item(), 3),
        ])
        payload["words"].append({
            "query": w,
            "single_token": r.single_token,
            "surface": r.surface,
            "tokens": labels,
            "token_ids": r.token_ids,
            "norm": r.vector.norm().item(),
            "variants_tried": [
                {"form": f, "n_tokens": len(i)} for f, i in (r.candidates or [])
            ],
        })
    lines.append(table(["word", "single token", "surface used", "pieces", "ids", "norm"], rows))
    emit(args, payload, "\n".join(lines))


def cmd_compare(args) -> None:
    m = open_model(args)
    vecs = resolve_words(args, m, args.words)
    warn_all(vecs)

    payload = {
        "model": m.name, "matrix": m.kind, "transforms": m.transforms,
        "layer": getattr(args, "layer", None),
        "words": [{"query": v.query, "name": v.name, **v.detail} for v in vecs],
        "pairs": [],
    }
    lines = [heading(f"pairwise geometry in {m.name}")]
    lines.append(f"{m.describe()}")
    if getattr(args, "layer", None) is not None:
        lines.append(f"contextual hidden states, layer {args.layer}")
    lines.append("")
    lines.append(table(["word", "resolved as"], [[v.name, v.label] for v in vecs]))

    rows = []
    for a, b in itertools.combinations(vecs, 2):
        s = geometry.pair_stats(a.vector, b.vector)
        rows.append([
            f"{a.name} ~ {b.name}",
            num(s.cosine), f"{s.angle_deg:.1f}", num(s.euclidean, 3),
            num(s.unit_euclidean, 3), num(s.norm_a, 2), num(s.norm_b, 2),
        ])
        payload["pairs"].append({
            "a": a.name, "b": b.name,
            "cosine": s.cosine, "angle_deg": s.angle_deg,
            "euclidean": s.euclidean, "unit_euclidean": s.unit_euclidean,
            "dot": s.dot, "norm_a": s.norm_a, "norm_b": s.norm_b,
        })

    lines.append(sub("pairwise"))
    lines.append(table(
        ["pair", "cosine", "angle", "euclid", "unit euclid", "|a|", "|b|"], rows
    ))

    if len(vecs) > 2:
        lines.append(sub("cosine matrix"))
        M = geometry.cosine_matrix(torch.stack([v.vector for v in vecs]))
        header = [""] + [v.name for v in vecs]
        mrows = [
            [vecs[i].name] + [num(M[i, j].item(), 3) for j in range(len(vecs))]
            for i in range(len(vecs))
        ]
        lines.append(table(header, mrows))
        payload["cosine_matrix"] = M.tolist()

    if args.dims:
        for a, b in itertools.combinations(vecs, 2):
            att = geometry.dim_attribution(a.vector, b.vector, args.dims)
            lines.append(sub(f"dimension attribution: {a.name} vs {b.name}"))
            lines.append(
                f"the difference is spread over ~{att['effective_dims']:.0f} of "
                f"{att['total_dims']} dimensions; the top 10 carry "
                f"{att['top10_delta_mass'] * 100:.1f}% of it"
            )
            lines.append("")
            lines.append("dimensions where they agree most (drive the similarity):")
            lines.append(table(
                ["dim", a.name, b.name, "share of cosine"],
                [[d.index, num(d.a, 3), num(d.b, 3), num(d.cosine_share, 4)]
                 for d in att["aligned"]],
                indent="  ",
            ))
            lines.append("")
            lines.append("dimensions where they differ most (drive the distance):")
            lines.append(table(
                ["dim", a.name, b.name, "delta", "share of dist^2"],
                [[d.index, num(d.a, 3), num(d.b, 3), num(d.delta, 3), num(d.delta_share, 4)]
                 for d in att["divergent"]],
                indent="  ",
            ))
            payload.setdefault("attribution", []).append({
                "a": a.name, "b": b.name,
                "effective_dims": att["effective_dims"],
                "top10_delta_mass": att["top10_delta_mass"],
                "aligned": [vars(d) for d in att["aligned"]],
                "opposed": [vars(d) for d in att["opposed"]],
                "divergent": [vars(d) for d in att["divergent"]],
            })

    emit(args, payload, "\n".join(lines))


def cmd_neighbors(args) -> None:
    m = open_model(args)
    vecs = resolve_words(args, m, args.words)
    warn_all(vecs)
    E, space = search_space(args, m)

    payload = {"model": m.name, "space": space, "metric": args.metric, "results": []}
    lines = [heading(f"nearest neighbours in {m.name}"), f"searching {space}"]

    for v in vecs:
        ex = list(v.token_ids) if getattr(args, "layer", None) is None else []
        if ex and args.exclude_variants:
            base = token_label(m.tokenizer, v.token_ids[0]) if v.query.startswith(("id:", "tok:")) else v.query
            ex += vocab.case_variant_ids(m.tokenizer, base)
        hits = geometry.neighbors(v.vector, E, k=args.k, exclude=ex, metric=args.metric)
        rows = [
            [i + 1, token_label(m.tokenizer, t), t, num(s)]
            for i, (t, s) in enumerate(hits)
        ]
        lines.append(sub(v.label))
        lines.append(table(["#", "token", "id", args.metric], rows))
        payload["results"].append({
            "query": v.name,
            "resolved": v.label,
            "neighbors": [
                {
                    "rank": i + 1,
                    "token": token_label(m.tokenizer, t),
                    "token_id": t,
                    "score": s,
                }
                for i, (t, s) in enumerate(hits)
            ],
        })

    emit(args, payload, "\n".join(lines))


def cmd_analogy(args) -> None:
    m = open_model(args)
    a, b, c = args.a, args.b, args.c
    vecs = resolve_words(args, m, [a, b, c])
    warn_all(vecs)
    va, vb, vc = (v.vector for v in vecs)

    target = geometry.analogy_vector(va, vb, vc)
    E, space = search_space(args, m)
    exclude = [] if getattr(args, "layer", None) is not None else [
        t for v in vecs for t in v.token_ids
    ]
    hits = geometry.neighbors(target, E, k=args.k, exclude=exclude)

    lines = [heading(f"{a} : {b} :: {c} : ?"), f"{m.describe()}", f"searching {space}"]
    lines.append("")
    rows = [
        [i + 1, token_label(m.tokenizer, t), t, num(s)]
        for i, (t, s) in enumerate(hits)
    ]
    lines.append(table(["#", "token", "id", "cosine"], rows))

    payload = {
        "model": m.name, "a": a, "b": b, "c": c, "space": space,
        "candidates": [
            {"rank": i + 1,
             "token": token_label(m.tokenizer, t),
             "token_id": t, "score": s}
            for i, (t, s) in enumerate(hits)
        ],
    }

    if args.expect:
        vd = resolve_words(args, m, [args.expect])[0]
        par = geometry.pair_direction_similarity(va, vb, vc, vd.vector)
        s = geometry.pair_stats(target, vd.vector)
        # Where does the expected answer actually rank?
        full = geometry.neighbors(target, E, k=E.shape[0], exclude=exclude)
        rank = next(
            (i + 1 for i, (t, _) in enumerate(full) if t in vd.token_ids), None
        )
        lines.append(sub(f"against the expected answer {args.expect!r}"))
        lines.append(table(
            ["measure", "value", "reading"],
            [
                ["offset parallelism", num(par),
                 f"cos({b}-{a}, {args.expect}-{c}); 1 = identical offset"],
                ["cosine to b-a+c", num(s.cosine), "how close the arithmetic lands"],
                ["rank of expected", str(rank) if rank else "not found",
                 "1 = the analogy resolves correctly"],
            ],
        ))
        payload["expected"] = {
            "word": args.expect,
            "offset_parallelism": par,
            "cosine_to_target": s.cosine,
            "rank": rank,
        }
        if par < 0.2:
            lines.append(
                "\nOffset parallelism is low: this contrast is not a consistent "
                "direction in this model, so a good top-1 here would be luck."
            )

    emit(args, payload, "\n".join(lines))


def cmd_axis(args) -> None:
    m = open_model(args)
    pairs_raw = []
    for spec in args.pairs:
        if ":" not in spec:
            raise SystemExit(f"--pairs expects FROM:TO, got {spec!r}")
        lo, hi = spec.split(":", 1)
        pairs_raw.append((lo.strip(), hi.strip()))

    flat = [w for p in pairs_raw for w in p]
    vecs = {v.query: v for v in resolve_words(args, m, flat)}
    warn_all(list(vecs.values()))

    pairs = [(vecs[a].vector, vecs[b].vector) for a, b in pairs_raw]
    labels = [f"{a} -> {b}" for a, b in pairs_raw]
    axis = geometry.build_axis(pairs, labels)

    lines = [heading(f"semantic axis in {m.name}"), m.describe(), ""]
    lines.append(table(
        ["defining pair", "offset norm", "cos to axis"],
        [
            [lab, num(n, 3),
             num(torch.nn.functional.cosine_similarity(
                 axis.deltas[i], axis.direction, dim=0).item())]
            for i, (lab, n) in enumerate(zip(axis.pair_labels, axis.delta_norms))
        ],
    ))
    lines.append(sub("is this actually one direction?"))
    lines.append(table(
        ["diagnostic", "value", "reading"],
        [
            ["offset coherence", num(axis.coherence),
             "mean cosine between the defining offsets"],
            ["PC1 energy share", num(axis.pc1_ratio),
             f"1.0 = perfectly rank-one; {1 / max(1, len(pairs)):.2f} = chance here"],
            ["mean/PC1 alignment", num(axis.pc1_alignment), "1 = the mean is the main axis"],
            ["verdict", "usable" if axis.is_linear else "WEAK", ""],
        ],
    ))
    if not axis.is_linear:
        lines.append(
            "\nThe defining offsets disagree with each other, so this contrast is "
            "not a single linear direction in this model. Projections below are "
            "reported but should not be interpreted as measuring the concept."
        )

    payload = {
        "model": m.name,
        "pairs": axis.pair_labels,
        "coherence": axis.coherence,
        "pc1_ratio": axis.pc1_ratio,
        "pc1_alignment": axis.pc1_alignment,
        "usable": axis.is_linear,
        "probes": [],
        "extremes": {},
    }

    if args.probe:
        probes = resolve_words(args, m, args.probe)
        warn_all(probes)
        P = torch.stack([p.vector for p in probes])
        scores = geometry.project(P, axis.direction).tolist()
        order = sorted(zip(args.probe, scores), key=lambda t: -t[1])
        lines.append(sub("probe words projected onto the axis"))
        lo_label, hi_label = pairs_raw[0]
        lines.append(f"(negative = toward {lo_label!r}, positive = toward {hi_label!r})")
        lines.append("")
        lines.append(table(
            ["word", "projection", "position"],
            [[w, num(s), bar(s, -0.5, 0.5)] for w, s in order],
        ))
        payload["probes"] = [{"word": w, "projection": s} for w, s in order]

    if args.extremes:
        scores = geometry.project(m.embeddings, axis.direction)
        k = args.extremes
        top = torch.topk(scores, k)
        bot = torch.topk(-scores, k)
        lines.append(sub(f"vocabulary extremes (top {k} each end)"))
        rows = []
        for i in range(k):
            hi_id = top.indices[i].item()
            lo_id = bot.indices[i].item()
            rows.append([
                i + 1,
                token_label(m.tokenizer, hi_id),
                num(top.values[i].item(), 3),
                token_label(m.tokenizer, lo_id),
                num(-bot.values[i].item(), 3),
            ])
        lines.append(table(
            ["#", f"toward {pairs_raw[0][1]}", "score", f"toward {pairs_raw[0][0]}", "score"],
            rows,
        ))
        payload["extremes"] = {
            "positive": [
                {"token": token_label(m.tokenizer, top.indices[i].item()),
                 "score": top.values[i].item()} for i in range(k)
            ],
            "negative": [
                {"token": token_label(m.tokenizer, bot.indices[i].item()),
                 "score": -bot.values[i].item()} for i in range(k)
            ],
        }

    emit(args, payload, "\n".join(lines))


def cmd_bias(args) -> None:
    if args.list_presets:
        lines = [heading("built-in bias test batteries")]
        lines.append(table(
            ["preset", "target sets", "attribute sets", "description"],
            [
                [key, f"{p['x_label']} vs {p['y_label']}",
                 f"{p['a_label']} vs {p['b_label']}", p["description"]]
                for key, p in bias.PRESETS.items()
            ],
        ))
        print("\n".join(lines))
        return

    if not args.model:
        raise SystemExit("give at least one -m/--model (or use --list-presets)")

    if args.preset:
        if any([args.x, args.y, args.a, args.b]):
            raise SystemExit("--preset and --x/--y/--a/--b are mutually exclusive")
        p = bias.PRESETS[args.preset]
        x_words, y_words, a_words, b_words = p["x"], p["y"], p["a"], p["b"]
        x_label = args.x_label or p["x_label"]
        y_label = args.y_label or p["y_label"]
        a_label = args.a_label or p["a_label"]
        b_label = args.b_label or p["b_label"]
    else:
        if not all([args.x, args.y, args.a, args.b]):
            raise SystemExit("give --preset, or all four of --x/--y/--a/--b")
        x_words, y_words, a_words, b_words = args.x, args.y, args.a, args.b
        x_label, y_label = args.x_label or "x", args.y_label or "y"
        a_label, b_label = args.a_label or "a", args.b_label or "b"

    for name, words in (("x", x_words), ("y", y_words), ("a", a_words), ("b", b_words)):
        if len(words) < 2:
            raise SystemExit(f"--{name} needs at least two words")

    lines = [heading("bias test (WEAT)")]
    payload = {
        "x_label": x_label, "y_label": y_label, "a_label": a_label, "b_label": b_label,
        "models": [],
    }

    for mid in args.model:
        m = open_model(args, mid)
        vx = resolve_words(args, m, x_words)
        vy = resolve_words(args, m, y_words)
        va = resolve_words(args, m, a_words)
        vb = resolve_words(args, m, b_words)
        warn_all(vx + vy + va + vb)

        r = bias.weat(
            torch.stack([v.vector for v in vx]), torch.stack([v.vector for v in vy]),
            torch.stack([v.vector for v in va]), torch.stack([v.vector for v in vb]),
            x_words, y_words, x_label, y_label, a_label, b_label,
            n_permutations=args.permutations, seed=args.seed,
        )

        lines.append(sub(f"{m.name}  ({x_label} vs {y_label}  /  {a_label} vs {b_label})"))
        lines.append(table(
            ["metric", "value", "reading"],
            [
                ["effect size (Cohen's d)", num(r.effect_size), r.magnitude],
                ["p-value", num(r.p_value, 4),
                 f"Monte Carlo estimate, {r.n_permutations} permutations"],
                ["statistic", num(r.statistic, 4), "raw s(X,Y,A,B)"],
            ],
        ))
        lines.append(sub("per-word association  (+ = toward A, - = toward B)"))
        lines.append(table(
            ["word", "group", "s(word, A, B)"],
            [[w, x_label if g == "x" else y_label, num(s, 4)] for w, g, s in r.word_scores],
        ))

        payload["models"].append({
            "model": m.name,
            "effect_size": r.effect_size, "magnitude": r.magnitude, "p_value": r.p_value,
            "statistic": r.statistic, "n_permutations": r.n_permutations,
            "word_scores": [{"word": w, "group": g, "score": s} for w, g, s in r.word_scores],
        })

    emit(args, payload, "\n".join(lines))


def cmd_pca(args) -> None:
    m = open_model(args)
    vecs = resolve_words(args, m, args.words)
    warn_all(vecs)
    X = torch.stack([v.vector for v in vecs])
    coords, ratio, _ = geometry.pca(X, args.components)

    lines = [heading(f"PCA of {len(vecs)} words in {m.name}"), m.describe(), ""]
    lines.append("explained variance: " + ", ".join(
        f"PC{i + 1} {r * 100:.1f}%" for i, r in enumerate(ratio.tolist())
    ))
    lines.append("")
    headers = ["word"] + [f"PC{i + 1}" for i in range(coords.shape[1])]
    lines.append(table(
        headers,
        [[v.name] + [num(c, 3) for c in coords[i].tolist()] for i, v in enumerate(vecs)],
    ))

    payload = {
        "model": m.name,
        "explained_variance_ratio": ratio.tolist(),
        "points": [
            {"word": v.name, "coords": coords[i].tolist()} for i, v in enumerate(vecs)
        ],
    }

    if args.out:
        if args.out.endswith(".csv"):
            with open(args.out, "w") as fh:
                fh.write("word," + ",".join(f"pc{i + 1}" for i in range(coords.shape[1])) + "\n")
                for i, v in enumerate(vecs):
                    fh.write(v.name + "," + ",".join(f"{c:.6f}" for c in coords[i].tolist()) + "\n")
        else:
            _scatter(coords, [v.name for v in vecs], ratio, args.out, m.name)
        lines.append(f"\nwrote {args.out}")

    emit(args, payload, "\n".join(lines))


def _scatter(coords, labels, ratio, path, title) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("plotting needs matplotlib: pip install 'eigentoken[plot]'")

    xy = coords[:, :2].tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter([p[0] for p in xy], [p[1] for p in xy], s=28)
    for (x, y), lab in zip(xy, labels):
        ax.annotate(lab, (x, y), fontsize=9, xytext=(4, 4), textcoords="offset points")
    r = ratio.tolist()
    ax.set_xlabel(f"PC1 ({r[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({r[1] * 100:.1f}%)" if len(r) > 1 else "PC2")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def cmd_crossmodel(args) -> None:
    if len(args.model) < 2:
        raise SystemExit("crossmodel needs at least two -m/--model arguments")

    words = list(args.words)
    if args.words_file:
        with open(args.words_file) as fh:
            words += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if len(words) < 4:
        raise SystemExit("give at least 4 words (RSA over fewer is meaningless)")

    models = []
    vectors = []
    token_ids = []
    dropped: dict[str, list[str]] = {}

    for mid in args.model:
        m = open_model(args, mid)
        res = [
            vocab.resolve(m, w, pooling=args.pooling, prefer_space=args.prefer_space)
            for w in words
        ]
        models.append(m)
        vectors.append(res)
        token_ids.append([r.token_ids for r in res])
        if args.single_token_only:
            dropped[mid] = [r.query for r in res if not r.single_token]

    if args.single_token_only:
        bad = {w for lst in dropped.values() for w in lst}
        if bad:
            print(
                f"  ! dropping {len(bad)} word(s) that are multi-token in at least "
                f"one model: {sorted(bad)}", file=sys.stderr,
            )
        keep = [i for i, w in enumerate(words) if w not in bad]
        if len(keep) < 4:
            raise SystemExit("too few shared single-token words left; drop --single-token-only")
        words = [words[i] for i in keep]
        vectors = [[v[i] for i in keep] for v in vectors]
        token_ids = [[t[i] for i in keep] for t in token_ids]

    stacks = [torch.stack([r.vector for r in v]).cpu() for v in vectors]

    lines = [heading("cross-model comparison")]
    lines.append(table(
        ["model", "matrix", "dim", "vocab", "multi-token words"],
        [
            [m.name, m.kind, m.dim, m.vocab_size,
             sum(1 for r in vectors[i] if not r.single_token)]
            for i, m in enumerate(models)
        ],
    ))
    lines.append(f"\ncomparing {len(words)} words")

    payload = {
        "words": words,
        "models": [{"name": m.name, "dim": m.dim, "vocab": m.vocab_size} for m in models],
        "comparisons": [],
    }

    for i, j in itertools.combinations(range(len(models)), 2):
        a, b = models[i], models[j]
        r = crossmodel.rsa(words, stacks[i], stacks[j])
        p = crossmodel.procrustes(words, stacks[i], stacks[j])

        lines.append(sub(f"{a.name}  vs  {b.name}"))
        lines.append(table(
            ["measure", "value", "reading"],
            [
                ["RSA spearman", num(r.spearman),
                 "rank agreement of all pairwise similarities"],
                ["RSA pearson", num(r.pearson), "linear agreement"],
                ["procrustes disparity", num(p.disparity),
                 f"residual after best rotation in {p.dim}d; 0 = same shape"],
            ],
        ))
        if len(words) <= p.dim:
            lines.append(
                f"  (only {len(words)} words for a {p.dim}d rotation: the "
                "disparity reflects the pairwise-distance mismatch alone. "
                "Add words for a stronger test.)"
            )
        entry = {
            "a": a.name, "b": b.name,
            "rsa_spearman": r.spearman, "rsa_pearson": r.pearson,
            "procrustes_disparity": p.disparity, "procrustes_dim": p.dim,
        }

        lines.append("\nwords the two models disagree about most:")
        lines.append(table(
            ["word", "profile agreement"],
            [[w, num(s)] for w, s in r.per_word[: args.top]],
            indent="  ",
        ))
        entry["most_divergent_words"] = [
            {"word": w, "profile_agreement": s} for w, s in r.per_word[: args.top]
        ]

        if args.neighbors:
            ov = crossmodel.neighborhood_overlap(
                words, a, b, stacks[i].to(a.device), stacks[j].to(b.device),
                k=args.neighbors,
                exclude_a=token_ids[i], exclude_b=token_ids[j],
            )
            lines.append(
                f"\nnearest-neighbour agreement (top {ov.k}): "
                f"mean jaccard {num(ov.mean_jaccard)}"
            )
            lines.append(table(
                ["word", "jaccard", f"{a.name} neighbours", f"{b.name} neighbours"],
                [
                    [w, num(j2, 2), " ".join(na[:6]), " ".join(nb[:6])]
                    for w, j2, na, nb in ov.per_word[-args.top:]
                ],
                indent="  ",
            ))
            entry["neighbor_overlap"] = {
                "k": ov.k,
                "mean_jaccard": ov.mean_jaccard,
                "per_word": [
                    {"word": w, "jaccard": j2, "a": na, "b": nb}
                    for w, j2, na, nb in ov.per_word
                ],
            }

        payload["comparisons"].append(entry)

    emit(args, payload, "\n".join(lines))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eigentoken",
        description="Compare and analyze token embedding geometry in LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  eigentoken info -m gpt2\n"
            "  eigentoken compare -m gpt2 king queen man woman --dims 10 --center\n"
            "  eigentoken neighbors -m gpt2 king -k 20 --center\n"
            "  eigentoken analogy -m gpt2 man king woman --expect queen\n"
            "  eigentoken axis -m gpt2 --pairs man:woman king:queen --probe doctor nurse\n"
            "  eigentoken crossmodel -m gpt2 -m EleutherAI/pythia-160m --words-file examples/words.txt\n"
            "  eigentoken compare -m gpt2 bank bank --layer 8 \\\n"
            "      --context 'I sat on the river bank' --context 'I withdrew cash at the bank'\n"
        ),
    )
    subs = p.add_subparsers(dest="command", required=True)

    q = subs.add_parser("info", help="matrix shape and how trustworthy raw cosines are")
    add_model_args(q)
    q.set_defaults(func=cmd_info)

    q = subs.add_parser("tokens", help="show how words map onto the vocabulary")
    add_model_args(q)
    q.add_argument("words", nargs="+")
    q.set_defaults(func=cmd_tokens)

    q = subs.add_parser("compare", help="pairwise geometry between words")
    add_model_args(q)
    add_context_args(q)
    q.add_argument("words", nargs="+", help="words, or id:1234 / tok:Ġking")
    q.add_argument(
        "--dims", type=int, default=0, metavar="K",
        help="also break the comparison down over the top K dimensions",
    )
    q.set_defaults(func=cmd_compare)

    q = subs.add_parser("neighbors", help="nearest tokens to a word")
    add_model_args(q)
    add_context_args(q)
    q.add_argument("words", nargs="+")
    q.add_argument("-k", type=int, default=15)
    q.add_argument("--metric", default="cosine", choices=["cosine", "dot", "euclidean"])
    q.add_argument(
        "--exclude-variants", action="store_true",
        help="drop casing/leading-space forms of the query word itself "
             "(KING, King, ' King', ...) from the results",
    )
    q.set_defaults(func=cmd_neighbors)

    q = subs.add_parser("analogy", help="a : b :: c : ?")
    add_model_args(q)
    add_context_args(q)
    q.add_argument("a")
    q.add_argument("b")
    q.add_argument("c")
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--expect", default=None, help="score a known answer instead of guessing")
    q.set_defaults(func=cmd_analogy)

    q = subs.add_parser("axis", help="build a semantic direction from example pairs")
    add_model_args(q)
    add_context_args(q)
    q.add_argument(
        "--pairs", nargs="+", required=True, metavar="FROM:TO",
        help="e.g. man:woman king:queen actor:actress",
    )
    q.add_argument("--probe", nargs="*", default=None, help="words to project onto the axis")
    q.add_argument(
        "--extremes", type=int, default=0, metavar="K",
        help="also show the K most extreme tokens in the whole vocabulary",
    )
    q.set_defaults(func=cmd_axis)

    q = subs.add_parser(
        "bias", help="WEAT-style test: how strongly does the model associate two "
                      "word sets with two attribute sets"
    )
    add_model_args(q, multi=True)
    add_context_args(q)
    for act in q._actions:
        if act.dest == "model":
            act.required = False  # --list-presets doesn't need a model
    q.add_argument(
        "--preset", choices=sorted(bias.PRESETS), default=None,
        help="use a built-in test battery instead of --x/--y/--a/--b",
    )
    q.add_argument("--list-presets", action="store_true", help="list built-in presets and exit")
    q.add_argument("--x", nargs="+", default=None, metavar="WORD", help="target set X")
    q.add_argument("--y", nargs="+", default=None, metavar="WORD", help="target set Y")
    q.add_argument("--a", nargs="+", default=None, metavar="WORD", help="attribute set A")
    q.add_argument("--b", nargs="+", default=None, metavar="WORD", help="attribute set B")
    q.add_argument("--x-label", default=None)
    q.add_argument("--y-label", default=None)
    q.add_argument("--a-label", default=None)
    q.add_argument("--b-label", default=None)
    q.add_argument(
        "--permutations", type=int, default=10_000,
        help="Monte Carlo permutations for the p-value estimate (default 10000)",
    )
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=cmd_bias)

    q = subs.add_parser("pca", help="project words to 2-3 dimensions")
    add_model_args(q)
    add_context_args(q)
    q.add_argument("words", nargs="+")
    q.add_argument("-c", "--components", type=int, default=2)
    q.add_argument("-o", "--out", default=None, help="write .csv or .png")
    q.set_defaults(func=cmd_pca)

    q = subs.add_parser("crossmodel", help="compare how two or more models organise the same words")
    add_model_args(q, multi=True)
    q.add_argument("--words", nargs="*", default=[])
    q.add_argument("--words-file", default=None, help="one word per line, # for comments")
    q.add_argument("--top", type=int, default=10, help="rows of detail to show")
    q.add_argument(
        "--neighbors", type=int, default=0, metavar="K",
        help="also compare each model's top-K neighbour lists",
    )
    q.add_argument(
        "--single-token-only", action="store_true",
        help="drop words that are not a single token in every model",
    )
    q.set_defaults(func=cmd_crossmodel)

    q = subs.add_parser("serve", help="run the web app")
    q.add_argument("--host", default=os.environ.get("EIGENTOKEN_HOST", "127.0.0.1"),
                   help="bind address; use 0.0.0.0 to expose on a server "
                        "($EIGENTOKEN_HOST, default 127.0.0.1)")
    q.add_argument("--port", type=int,
                   default=int(os.environ.get("EIGENTOKEN_PORT", DEFAULT_PORT)),
                   help=f"port to listen on ($EIGENTOKEN_PORT, default {DEFAULT_PORT}); "
                        "use 0 to have the OS pick any free port")
    q.add_argument("--device", default="auto")
    q.add_argument("--dtype", default="auto", choices=list(loader.DTYPES))
    q.add_argument("--max-models", type=int, default=3,
                   help="how many embedding matrices stay resident (LRU beyond this)")
    q.add_argument("--max-memory-gb", type=float, default=0,
                   help="also cap resident matrices by size; the least recently "
                        "used are freed to make room before a new one is read "
                        "(default 0: count only)")
    q.add_argument("--cache-dir", default=None,
                   help="download models here instead of the shared Hugging Face "
                        "cache; required by --max-disk-gb")
    q.add_argument("--max-disk-gb", type=float, default=0,
                   help="cap the downloads kept in --cache-dir; the least recently "
                        "used models are deleted to make room before a download "
                        "(default 0: keep everything)")
    q.add_argument("--allow-unload", action="store_true",
                   help="let visitors free resident models by hand. Off by default, "
                        "since models are shared: the LRU budgets free them instead")
    q.add_argument("--allow", nargs="*", default=None, metavar="MODEL_ID",
                   help="restrict users to these models; omit to allow any")
    q.add_argument("--allow-file", default=None,
                   help="file of allowed model ids, one per line")
    q.add_argument("--trust-remote-code", action="store_true",
                   help="let model repos execute their own code (off by default)")
    q.add_argument("--hf-token", default=None, help="HuggingFace token for gated models")
    q.add_argument("--preload", nargs="*", default=None, metavar="SPEC",
                   help="load at boot so the first request is instant. SPEC is "
                        "MODEL[:output][:center][:whiten] -- the web UI defaults to "
                        "centered, so 'gpt2:center' is usually what you want")
    g = q.add_argument_group(
        "load limits",
        "what visitors may load. They matter most without --allow, when anyone "
        "can name any model on the Hub; 0 turns a limit off.")
    g.add_argument("--max-download-gb", type=float, default=10,
                   help="refuse models whose weights to fetch exceed this, checked "
                        "before downloading (default 10)")
    g.add_argument("--max-matrix-gb", type=float, default=6,
                   help="refuse embedding matrices bigger than this in memory, as "
                        "float32 (default 6)")
    g.add_argument("--max-concurrent-loads", type=int, default=2,
                   help="loads running at once; more wait in a queue (default 2)")
    g.add_argument("--max-queued-loads", type=int, default=4,
                   help="loads waiting for a slot; beyond this, requests are "
                        "refused until one finishes (default 4)")
    g.add_argument("--loads-per-hour", type=int, default=30,
                   help="new loads one client address may start per hour; using "
                        "resident models is not counted (default 30)")
    g.add_argument("--allow-pickle-weights", action="store_true",
                   help="also load .bin/.pt checkpoints, which are Python pickles "
                        "that can carry code; by default only safetensors load")
    g.add_argument("--allow-local-paths", action="store_true",
                   help="let visitors name directories on this machine; by default "
                        "only Hub repo ids load, plus anything listed in --allow")
    q.add_argument("--cors-origin", nargs="*", metavar="ORIGIN",
                   default=os.environ.get("EIGENTOKEN_CORS", "").split() or None,
                   help="let pages on these origins call the API, e.g. the GitHub "
                        "Pages copy of the UI: https://USER.github.io "
                        "($EIGENTOKEN_CORS, space separated)")
    q.add_argument("--api-only", action="store_true",
                   help="serve the API without the web interface, for a server "
                        "behind a copy of the UI hosted elsewhere")
    q.set_defaults(func=cmd_serve)

    return p


def bind_socket(host: str, port: int):
    """Claim the port up front so a clash is a clear message, not a traceback.

    The socket is handed to uvicorn rather than closed and reopened, which also
    removes the window where something else could take the port in between.
    Port 0 asks the OS for any free port; the chosen one is printed.
    """
    import socket

    try:
        info = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )[0]
    except socket.gaierror as e:
        raise SystemExit(f"cannot resolve --host {host!r}: {e}")

    sock = socket.socket(info[0], socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(info[4])
    except OSError as e:
        sock.close()
        import errno

        if e.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"port {port} on {host} is already in use -- something else is "
                f"listening there. Pick another with --port N (or --port 0 to "
                f"let the OS choose), or set $EIGENTOKEN_PORT."
            )
        if e.errno in (errno.EACCES, errno.EPERM):
            raise SystemExit(
                f"not allowed to bind port {port} (ports below 1024 need root). "
                "Use --port N with N above 1024 and put a reverse proxy in front."
            )
        raise SystemExit(f"cannot bind {host}:{port} -- {e}")
    return sock


def cmd_serve(args) -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "the web app needs fastapi and uvicorn: uv pip install -e '.[web]'"
        )

    from .server import ModelRegistry, create_app
    from .server.registry import Spec

    allow = list(args.allow) if args.allow else None
    if args.allow_file:
        with open(args.allow_file) as fh:
            allow = (allow or []) + [
                ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")
            ]

    if args.max_disk_gb and not args.cache_dir:
        raise SystemExit(
            "--max-disk-gb deletes the least recently used models in its cache, so "
            "it needs a directory of eigentoken's own: pass --cache-dir as well."
        )

    registry = ModelRegistry(
        max_models=args.max_models,
        device=args.device,
        dtype=args.dtype,
        allowlist=allow,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token,
        allow_local_paths=args.allow_local_paths,
        safetensors_only=not args.allow_pickle_weights,
        max_download_bytes=int(args.max_download_gb * 1e9),
        max_matrix_bytes=int(args.max_matrix_gb * 1e9),
        max_concurrent_loads=args.max_concurrent_loads,
        max_queued_loads=args.max_queued_loads,
        loads_per_client_hour=args.loads_per_hour,
        max_memory_bytes=int(args.max_memory_gb * 1e9),
        cache_dir=args.cache_dir,
        max_disk_bytes=int(args.max_disk_gb * 1e9),
        allow_unload=args.allow_unload,
    )

    for item in args.preload or []:
        bits = item.split(":")
        mid = bits[0]
        flags = {b.strip().lower() for b in bits[1:]}
        unknown = flags - {"input", "output", "center", "centered", "whiten", "whitened"}
        if unknown or not mid:
            raise SystemExit(
                f"cannot parse --preload {item!r}; "
                "expected MODEL[:output][:center][:whiten]"
            )
        try:
            registry.request(Spec(
                model_id=mid,
                matrix="output" if "output" in flags else "input",
                center=bool(flags & {"center", "centered"}),
                whiten=bool(flags & {"whiten", "whitened"}),
            ))
        except Exception as e:  # the load limits apply to preloads too
            raise SystemExit(f"cannot preload {item!r}: {e}")

    if args.host == "0.0.0.0":
        print(
            "note: binding 0.0.0.0 exposes this with no authentication. Put it "
            "behind a reverse proxy, and pass --allow to stop users pulling "
            "arbitrary models onto the server.",
            file=sys.stderr,
        )

    sock = bind_socket(args.host, args.port)
    port = sock.getsockname()[1]

    app = create_app(registry, cors_origins=args.cors_origin, serve_ui=not args.api_only)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    # stdout is block-buffered when redirected to a file or a pipe, and
    # uvicorn logs to stderr; without a flush the URL surfaces after the
    # startup lines, or not at all until exit.
    print(f"eigentoken on http://{shown}:{port}", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))
    server.run(sockets=[sock])


def _quiet() -> None:
    """The loading progress bar belongs on stderr at best; keep output clean."""
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "verbose", False):
        _quiet()
    if getattr(args, "layer", None) is not None and not getattr(args, "context", None):
        raise SystemExit("--layer requires --context")
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
