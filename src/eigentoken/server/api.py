"""HTTP API and static file serving for the eigentoken web app.

Design notes:

* Every analysis endpoint takes one or more **specs** rather than a bare model
  name.  A spec is (model, matrix, center, whiten, layer), so the same model
  can appear twice in one comparison -- its input matrix against its output
  matrix, or centered against raw.
* Nothing blocks on a model download.  If a spec is not resident the endpoint
  returns HTTP 202 with ``status: loading`` and the client polls
  ``/api/models``.
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import bias, crossmodel, geometry, vocab
from ..vocab import token_label
from .registry import LoadRateLimited, LoadsBusy, ModelNotAllowed, ModelRegistry, Spec

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

SUGGESTED = [
    {"id": "gpt2", "note": "124M · fast, the classic probe target"},
    {"id": "gpt2-medium", "note": "355M"},
    {"id": "distilgpt2", "note": "82M · smallest useful"},
    {"id": "EleutherAI/pythia-160m", "note": "160M · different tokenizer"},
    {"id": "EleutherAI/pythia-410m", "note": "410M"},
    {"id": "bert-base-uncased", "note": "110M · encoder, WordPiece"},
    {"id": "roberta-base", "note": "125M · encoder, BPE"},
    {"id": "HuggingFaceTB/SmolLM2-360M", "note": "360M · modern recipe"},
    {"id": "Qwen/Qwen3-0.6B", "note": "0.6B · 151k multilingual vocab"},
    {"id": "Qwen/Qwen3-1.7B", "note": "1.7B"},
    {"id": "meta-llama/Llama-3.2-1B", "note": "1B · gated, needs a token"},
]


def _spec(d: dict) -> Spec:
    try:
        return Spec.from_dict(d)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _request(registry: ModelRegistry, spec: Spec, request: Request, retry: bool = False):
    """registry.request(), with its refusals turned into HTTP errors."""
    client = request.client.host if request.client else None
    try:
        return registry.request(spec, retry=retry, client=client)
    except ModelNotAllowed as e:
        raise HTTPException(403, str(e))
    except LoadRateLimited as e:
        raise HTTPException(429, str(e))
    except LoadsBusy as e:
        raise HTTPException(503, str(e))


def _acquire(registry: ModelRegistry, specs: list[Spec], request: Request):
    """Kick off any missing loads; return (wait_response, models).

    The loaded models are handed back directly rather than re-fetched later.
    Holding a reference for the life of the request means an LRU eviction
    triggered by another user cannot pull the matrix out from under us
    mid-computation -- the entry leaves the cache, the tensor stays alive.
    """
    waiting = []
    models = []
    for s in specs:
        entry = _request(registry, s, request)
        if entry.status == "error" and entry.refused:
            raise HTTPException(403, entry.error)
        if entry.status == "error":
            raise HTTPException(
                502,
                f"loading {s.model_id} failed -- {entry.error}. "
                "Fix the model id, then use the retry button beside it in the "
                "Resident panel.",
            )
        if entry.status != "ready":
            waiting.append(entry.info())
        models.append(entry.model)
    if waiting:
        return JSONResponse({"status": "loading", "waiting": waiting}, status_code=202), []
    return None, models


def _resolve(model, words: list[str], pooling: str) -> list[vocab.Resolution]:
    out = []
    for w in words:
        try:
            out.append(vocab.resolve(model, w, pooling=pooling))
        except KeyError as e:
            # str(KeyError) adds repr quotes around the message; unwrap it.
            raise HTTPException(400, e.args[0] if e.args else str(e))
    return out


def _base_word(model, r: vocab.Resolution) -> str:
    """The plain word text behind a resolution, for variant lookup.

    id:/tok: queries carry the raw address string as ``query``, not text --
    decode the token itself instead.
    """
    if r.query.startswith(("id:", "tok:")):
        return token_label(model.tokenizer, r.token_ids[0])
    return r.query


def _res_json(model, r: vocab.Resolution) -> dict:
    return {
        "query": r.query,
        "surface": r.surface,
        "single_token": r.single_token,
        "token_ids": r.token_ids,
        "tokens": [token_label(model.tokenizer, t) for t in r.token_ids],
        "pooling": r.pooling,
        "warning": r.warning,
        "norm": r.vector.norm().item(),
    }


def _model_meta(m) -> dict:
    """Provenance a caller needs in order to read the numbers correctly.

    ``tied`` matters: most decoder LMs reuse the input table as the unembedding,
    so asking for the output matrix hands back the same rows.  Comparisons then
    agree perfectly, which is correct but looks like a bug unless it is said.
    """
    return {
        "source": m.source,
        "tied": "tied to" in m.source,
        "dim": m.dim,
        "vocab_size": m.vocab_size,
        "n_layers": m.n_layers,
        "device": m.device,
    }


def _clean_words(words) -> list[str]:
    seen, out = set(), []
    for w in words or []:
        w = str(w).strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def build_router(registry: ModelRegistry) -> APIRouter:
    r = APIRouter(prefix="/api")

    # -- models ---------------------------------------------------------

    @r.get("/health")
    def health():
        return {
            "ok": True,
            "device": registry.device,
            "cuda": torch.cuda.is_available(),
            "max_models": registry.max_models,
            "restricted": registry.allowlist is not None,
            "can_unload": registry.allow_unload,
            "limits": {
                "safetensors_only": registry.safetensors_only,
                "max_download_bytes": registry.max_download_bytes,
                "max_matrix_bytes": registry.max_matrix_bytes,
                "max_concurrent_loads": registry.max_concurrent_loads or None,
                "loads_per_client_hour": registry.loads_per_client_hour or None,
            },
        }

    @r.get("/models")
    def models():
        return {
            "suggested": (
                [{"id": m, "note": "allowed by the operator"} for m in registry.allowlist]
                if registry.allowlist is not None
                else SUGGESTED
            ),
            "loaded": registry.listing(),
            "restricted": registry.allowlist is not None,
            "memory": {"used": registry.resident_bytes(), "max": registry.max_memory_bytes},
            "can_unload": registry.allow_unload,
        }

    @r.post("/models/load")
    def load_model(request: Request, body: dict = Body(...)):
        return _request(registry, _spec(body), request, retry=bool(body.get("retry"))).info()

    @r.post("/models/unload")
    def unload_model(body: dict = Body(...)):
        # Models are shared by every visitor, so by default only the LRU budgets
        # free them; one visitor must not be able to unload another's.
        if not registry.allow_unload:
            raise HTTPException(
                403, "this server frees models itself, least recently used first, "
                     "when it needs the room; unloading by hand is turned off.")
        return {"dropped": registry.drop(body.get("key", ""))}

    # -- single-model analysis -------------------------------------------

    @r.post("/tokens")
    def tokens(request: Request, body: dict = Body(...)):
        spec = _spec(body.get("spec", {}))
        pending, models = _acquire(registry, [spec], request)
        if pending:
            return pending
        m = models[0]
        words = _clean_words(body.get("words"))
        if not words:
            raise HTTPException(400, "give at least one word")
        pooling = body.get("pooling", "mean")
        out = []
        for res in _resolve(m, words, pooling):
            d = _res_json(m, res)
            d["variants"] = [
                {"form": f, "n_tokens": len(i)} for f, i in (res.candidates or [])
            ]
            out.append(d)
        return {
            "status": "ok", "spec": spec.to_dict(), "meta": _model_meta(m), "words": out,
        }

    @r.post("/compare")
    def compare(request: Request, body: dict = Body(...)):
        specs = [_spec(s) for s in body.get("specs", [])]
        if not specs:
            raise HTTPException(400, "select at least one model")
        pending, models = _acquire(registry, specs, request)
        if pending:
            return pending
        words = _clean_words(body.get("words"))
        if len(words) < 2:
            raise HTTPException(400, "give at least two words to compare")
        pooling = body.get("pooling", "mean")
        top_dims = int(body.get("dims", 0) or 0)

        results = []
        for spec, m in zip(specs, models):
            res = _resolve(m, words, pooling)
            vecs = [x.vector for x in res]
            pairs = []
            for (i, a), (j, b) in itertools.combinations(enumerate(res), 2):
                s = geometry.pair_stats(vecs[i], vecs[j])
                entry = {
                    "a": a.query, "b": b.query,
                    "cosine": s.cosine, "angle_deg": s.angle_deg,
                    "euclidean": s.euclidean, "unit_euclidean": s.unit_euclidean,
                    "dot": s.dot, "norm_a": s.norm_a, "norm_b": s.norm_b,
                }
                if top_dims:
                    att = geometry.dim_attribution(vecs[i], vecs[j], top_dims)
                    entry["attribution"] = {
                        "effective_dims": att["effective_dims"],
                        "total_dims": att["total_dims"],
                        "top10_delta_mass": att["top10_delta_mass"],
                        "aligned": [vars(d) for d in att["aligned"]],
                        "divergent": [vars(d) for d in att["divergent"]],
                    }
                pairs.append(entry)

            M = geometry.cosine_matrix(torch.stack(vecs))
            results.append({
                "spec": spec.to_dict(),
                "meta": _model_meta(m),
                "dim": m.dim,
                "vocab_size": m.vocab_size,
                "words": [_res_json(m, x) for x in res],
                "pairs": pairs,
                "cosine_matrix": M.tolist(),
            })

        return {"status": "ok", "words": words, "results": results}

    @r.post("/neighbors")
    def neighbors(request: Request, body: dict = Body(...)):
        specs = [_spec(s) for s in body.get("specs", [])]
        if not specs:
            raise HTTPException(400, "select at least one model")
        pending, models = _acquire(registry, specs, request)
        if pending:
            return pending
        words = _clean_words(body.get("words"))
        if not words:
            raise HTTPException(400, "give at least one word")
        k = max(1, min(int(body.get("k", 20)), 500))
        metric = body.get("metric", "cosine")
        if metric not in ("cosine", "dot", "euclidean"):
            raise HTTPException(400, "metric must be cosine, dot or euclidean")
        pooling = body.get("pooling", "mean")
        exclude_variants = bool(body.get("exclude_variants"))

        results = []
        for spec, m in zip(specs, models):
            per_word = []
            for res in _resolve(m, words, pooling):
                ex = list(res.token_ids)
                if exclude_variants:
                    ex += vocab.case_variant_ids(m.tokenizer, _base_word(m, res))
                hits = geometry.neighbors(
                    res.vector, m.embeddings, k=k, exclude=ex, metric=metric
                )
                per_word.append({
                    "word": _res_json(m, res),
                    "neighbors": [
                        {"rank": i + 1, "token_id": t,
                         "token": token_label(m.tokenizer, t), "score": s}
                        for i, (t, s) in enumerate(hits)
                    ],
                })
            results.append({
                "spec": spec.to_dict(), "meta": _model_meta(m), "words": per_word,
            })
        return {"status": "ok", "metric": metric, "results": results}

    @r.post("/cloud")
    def cloud(request: Request, body: dict = Body(...)):
        """The associated-words tool: a token's neighbourhood as a 2D cloud."""
        spec = _spec(body.get("spec", {}))
        pending, models = _acquire(registry, [spec], request)
        if pending:
            return pending
        m = models[0]
        word = str(body.get("word", "")).strip()
        if not word:
            raise HTTPException(400, "give a word or token")
        k = max(5, min(int(body.get("k", 120)), 600))
        mode = body.get("mode", "radial")
        n_clusters = max(0, min(int(body.get("clusters", 6) or 0), 12))
        pooling = body.get("pooling", "mean")
        exclude_variants = bool(body.get("exclude_variants"))

        res = _resolve(m, [word], pooling)[0]
        ex = list(res.token_ids)
        if exclude_variants:
            ex += vocab.case_variant_ids(m.tokenizer, _base_word(m, res))
        hits = geometry.neighbors(
            res.vector, m.embeddings, k=k, exclude=ex, metric="cosine"
        )
        ids = [t for t, _ in hits]
        V = m.embeddings[torch.tensor(ids, device=m.embeddings.device)]

        try:
            layout = geometry.cloud_layout(
                res.vector, V, mode=mode, n_clusters=n_clusters
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        points = [
            {
                "token_id": ids[i],
                "token": token_label(m.tokenizer, ids[i]),
                "x": layout["coords"][i][0],
                "y": layout["coords"][i][1],
                "cosine": layout["similarities"][i],
                "cluster": layout["clusters"][i],
                "norm": V[i].norm().item(),
            }
            for i in range(len(ids))
        ]
        return {
            "status": "ok",
            "spec": spec.to_dict(),
            "meta": _model_meta(m),
            "center": _res_json(m, res),
            "points": points,
            "mode": layout["mode"],
            "explained_variance_ratio": layout["explained_variance_ratio"],
            "n_clusters": n_clusters,
        }

    @r.post("/analogy")
    def analogy(request: Request, body: dict = Body(...)):
        specs = [_spec(s) for s in body.get("specs", [])]
        if not specs:
            raise HTTPException(400, "select at least one model")
        pending, models = _acquire(registry, specs, request)
        if pending:
            return pending
        a, b, c = (str(body.get(x, "")).strip() for x in ("a", "b", "c"))
        if not (a and b and c):
            raise HTTPException(400, "analogy needs a, b and c")
        expect = str(body.get("expect", "")).strip() or None
        k = max(1, min(int(body.get("k", 10)), 100))
        pooling = body.get("pooling", "mean")

        results = []
        for spec, m in zip(specs, models):
            ra, rb, rc = _resolve(m, [a, b, c], pooling)
            target = geometry.analogy_vector(ra.vector, rb.vector, rc.vector)
            exclude = ra.token_ids + rb.token_ids + rc.token_ids
            hits = geometry.neighbors(target, m.embeddings, k=k, exclude=exclude)
            entry = {
                "spec": spec.to_dict(),
                "meta": _model_meta(m),
                "words": [_res_json(m, x) for x in (ra, rb, rc)],
                "candidates": [
                    {"rank": i + 1, "token_id": t,
                     "token": token_label(m.tokenizer, t), "score": s}
                    for i, (t, s) in enumerate(hits)
                ],
            }
            if expect:
                rd = _resolve(m, [expect], pooling)[0]
                full = geometry.neighbors(
                    target, m.embeddings, k=m.vocab_size, exclude=exclude
                )
                rank = next(
                    (i + 1 for i, (t, _) in enumerate(full) if t in rd.token_ids), None
                )
                entry["expected"] = {
                    "word": expect,
                    "resolved": _res_json(m, rd),
                    "offset_parallelism": geometry.pair_direction_similarity(
                        ra.vector, rb.vector, rc.vector, rd.vector
                    ),
                    "cosine_to_target": geometry.pair_stats(target, rd.vector).cosine,
                    "rank": rank,
                }
            results.append(entry)
        return {"status": "ok", "a": a, "b": b, "c": c, "results": results}

    @r.post("/axis")
    def axis(request: Request, body: dict = Body(...)):
        specs = [_spec(s) for s in body.get("specs", [])]
        if not specs:
            raise HTTPException(400, "select at least one model")
        pending, models = _acquire(registry, specs, request)
        if pending:
            return pending

        raw_pairs = []
        for item in body.get("pairs", []):
            if isinstance(item, dict):
                lo, hi = item.get("from", ""), item.get("to", "")
            else:
                if ":" not in str(item):
                    raise HTTPException(400, f"pair {item!r} must be FROM:TO")
                lo, hi = str(item).split(":", 1)
            lo, hi = lo.strip(), hi.strip()
            if lo and hi:
                raw_pairs.append((lo, hi))
        if len(raw_pairs) < 2:
            raise HTTPException(400, "give at least two defining pairs")

        probes = _clean_words(body.get("probes"))
        extremes = max(0, min(int(body.get("extremes", 0) or 0), 50))
        pooling = body.get("pooling", "mean")

        flat = _clean_words(w for p_ in raw_pairs for w in p_)

        results = []
        for spec, m in zip(specs, models):
            rv = {r.query: r for r in _resolve(m, flat, pooling)}
            ax = geometry.build_axis(
                [(rv[x].vector, rv[y].vector) for x, y in raw_pairs],
                [f"{x} -> {y}" for x, y in raw_pairs],
            )
            entry = {
                "spec": spec.to_dict(),
                "meta": _model_meta(m),
                "pairs": [
                    {
                        "label": lab,
                        "norm": ax.delta_norms[i],
                        "cos_to_axis": torch.nn.functional.cosine_similarity(
                            ax.deltas[i], ax.direction, dim=0
                        ).item(),
                    }
                    for i, lab in enumerate(ax.pair_labels)
                ],
                "coherence": ax.coherence,
                "pc1_ratio": ax.pc1_ratio,
                "pc1_alignment": ax.pc1_alignment,
                "chance_pc1": 1.0 / len(raw_pairs),
                "usable": ax.is_linear,
                "low_label": raw_pairs[0][0],
                "high_label": raw_pairs[0][1],
            }
            if probes:
                pr = _resolve(m, probes, pooling)
                scores = geometry.project(
                    torch.stack([x.vector for x in pr]), ax.direction
                ).tolist()
                entry["probes"] = sorted(
                    [{"word": pr[i].query, "projection": scores[i],
                      "single_token": pr[i].single_token}
                     for i in range(len(pr))],
                    key=lambda d: -d["projection"],
                )
            if extremes:
                sc = geometry.project(m.embeddings, ax.direction)
                top = torch.topk(sc, extremes)
                bot = torch.topk(-sc, extremes)
                entry["extremes"] = {
                    "high": [
                        {"token": token_label(m.tokenizer, top.indices[i].item()),
                         "score": top.values[i].item()}
                        for i in range(extremes)
                    ],
                    "low": [
                        {"token": token_label(m.tokenizer, bot.indices[i].item()),
                         "score": -bot.values[i].item()}
                        for i in range(extremes)
                    ],
                }
            results.append(entry)
        return {"status": "ok", "results": results}

    @r.get("/bias/presets")
    def bias_presets():
        return {"presets": [{"key": k, **v} for k, v in bias.PRESETS.items()]}

    @r.post("/bias")
    def run_bias(request: Request, body: dict = Body(...)):
        specs = [_spec(s) for s in body.get("specs", [])]
        if not specs:
            raise HTTPException(400, "select at least one model")
        pending, models = _acquire(registry, specs, request)
        if pending:
            return pending

        preset_key = body.get("preset")
        preset = None
        if preset_key:
            preset = bias.PRESETS.get(preset_key)
            if preset is None:
                raise HTTPException(400, f"unknown preset {preset_key!r}")

        x = _clean_words(body.get("x")) or (preset["x"] if preset else [])
        y = _clean_words(body.get("y")) or (preset["y"] if preset else [])
        a = _clean_words(body.get("a")) or (preset["a"] if preset else [])
        b = _clean_words(body.get("b")) or (preset["b"] if preset else [])
        if not (x and y and a and b):
            raise HTTPException(400, "give a preset, or all four of x/y/a/b")
        for name, words in (("x", x), ("y", y), ("a", a), ("b", b)):
            if len(words) < 2:
                raise HTTPException(400, f"give at least two {name} words")

        x_label = body.get("x_label") or (preset["x_label"] if preset else "x")
        y_label = body.get("y_label") or (preset["y_label"] if preset else "y")
        a_label = body.get("a_label") or (preset["a_label"] if preset else "a")
        b_label = body.get("b_label") or (preset["b_label"] if preset else "b")
        permutations = max(100, min(int(body.get("permutations", 10_000) or 10_000), 50_000))
        pooling = body.get("pooling", "mean")

        results = []
        for spec, m in zip(specs, models):
            vx, vy, va, vb = (_resolve(m, ws, pooling) for ws in (x, y, a, b))
            r = bias.weat(
                torch.stack([v.vector for v in vx]), torch.stack([v.vector for v in vy]),
                torch.stack([v.vector for v in va]), torch.stack([v.vector for v in vb]),
                x, y, x_label, y_label, a_label, b_label,
                n_permutations=permutations,
            )
            results.append({
                "spec": spec.to_dict(),
                "meta": _model_meta(m),
                "x_label": x_label, "y_label": y_label,
                "a_label": a_label, "b_label": b_label,
                "effect_size": r.effect_size, "magnitude": r.magnitude,
                "p_value": r.p_value, "statistic": r.statistic,
                "n_permutations": r.n_permutations,
                "word_scores": [
                    {"word": w, "group": g, "score": s} for w, g, s in r.word_scores
                ],
            })
        return {"status": "ok", "results": results}

    @r.post("/crossmodel")
    def cross(request: Request, body: dict = Body(...)):
        specs = [_spec(s) for s in body.get("specs", [])]
        if len(specs) < 2:
            raise HTTPException(400, "cross-model comparison needs two or more models")
        pending, models = _acquire(registry, specs, request)
        if pending:
            return pending
        words = _clean_words(body.get("words"))
        if len(words) < 4:
            raise HTTPException(
                400, "give at least four words -- similarity structure over fewer is noise"
            )
        pooling = body.get("pooling", "mean")
        nb_k = max(0, min(int(body.get("neighbors", 0) or 0), 50))
        single_only = bool(body.get("single_token_only", False))
        top = max(1, min(int(body.get("top", 10)), 50))

        resolutions = [_resolve(m, words, pooling) for m in models]

        if single_only:
            bad = {
                r.query
                for lst in resolutions
                for r in lst
                if not r.single_token
            }
            keep = [i for i, w in enumerate(words) if w not in bad]
            if len(keep) < 4:
                raise HTTPException(
                    400,
                    f"only {len(keep)} words are single tokens in every model; "
                    "add words or turn off the single-token filter",
                )
            words = [words[i] for i in keep]
            resolutions = [[lst[i] for i in keep] for lst in resolutions]
            dropped = sorted(bad)
        else:
            dropped = []

        stacks = [torch.stack([r.vector for r in lst]).cpu() for lst in resolutions]

        comparisons = []
        for i, j in itertools.combinations(range(len(specs)), 2):
            rs = crossmodel.rsa(words, stacks[i], stacks[j])
            pc = crossmodel.procrustes(words, stacks[i], stacks[j])
            entry = {
                "a": specs[i].to_dict(), "b": specs[j].to_dict(),
                "rsa_spearman": rs.spearman, "rsa_pearson": rs.pearson,
                "procrustes_disparity": pc.disparity, "procrustes_dim": pc.dim,
                "underdetermined": len(words) <= pc.dim,
                "divergent_words": [
                    {"word": w, "agreement": s} for w, s in rs.per_word[:top]
                ],
            }
            if nb_k:
                ov = crossmodel.neighborhood_overlap(
                    words, models[i], models[j],
                    stacks[i].to(models[i].device), stacks[j].to(models[j].device),
                    k=nb_k,
                    exclude_a=[r.token_ids for r in resolutions[i]],
                    exclude_b=[r.token_ids for r in resolutions[j]],
                )
                entry["overlap"] = {
                    "k": ov.k,
                    "mean_jaccard": ov.mean_jaccard,
                    "per_word": [
                        {"word": w, "jaccard": jj, "a": na, "b": nb}
                        for w, jj, na, nb in ov.per_word
                    ],
                }
            comparisons.append(entry)

        # Two slots are the same data when they resolve to the same checkpoint
        # tensor under the same transforms.  The source string carries a
        # "(tied to input embeddings)" note, so strip the parenthetical before
        # comparing -- otherwise a tied output slot never matches its input.
        fingerprints = [
            (
                specs[i].model_id,
                re.sub(r"\s*\(.*\)\s*$", "", models[i].source),
                specs[i].center,
                specs[i].whiten,
            )
            for i in range(len(specs))
        ]
        identical = len(set(fingerprints)) < len(fingerprints)

        return {
            "status": "ok",
            "words": words,
            "dropped": dropped,
            "identical_matrices": identical,
            "models": [
                {"spec": specs[i].to_dict(), "meta": _model_meta(models[i]),
                 "dim": models[i].dim, "vocab_size": models[i].vocab_size,
                 "multi_token": sum(1 for r in resolutions[i] if not r.single_token)}
                for i in range(len(specs))
            ],
            "comparisons": comparisons,
        }

    return r


def create_app(
    registry: ModelRegistry, cors_origins: Optional[list] = None, serve_ui: bool = True
) -> FastAPI:
    app = FastAPI(title="eigentoken", docs_url="/api/docs", redoc_url=None)
    if cors_origins:
        # Lets a copy of the UI hosted elsewhere (GitHub Pages) call this server.
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.include_router(build_router(registry))
    if not serve_ui:
        # API only: the interface is hosted elsewhere (GitHub Pages). The root still
        # answers, so a browser or a host's health check finds something there.
        @app.get("/", include_in_schema=False)
        def root():
            return {"service": "eigentoken API", "health": "/api/health",
                    "docs": "/api/docs"}
    elif WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app
