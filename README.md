# EigenToken

A research tool for reading the geometry of an LLM's token embeddings.

A model encodes `king` as a point in an N-dimensional space and `queen` as
another. `eigentoken` measures where those points sit relative to each other,
what direction separates them, whether that direction generalises to other
word pairs, and whether two different models agree about any of it.

It is a **web app** you can run on a server — pick models, type words, and
explore a token's neighbourhood as an interactive cloud — and the same
functionality as a **command line tool** and a **Python library**.

Works with any model on the HuggingFace hub or on disk — GPT-2, Llama, Qwen,
Pythia, Mistral, Gemma — plus BERT-style encoders.

## Install

```bash
uv venv --python 3.12
uv pip install -e '.[web,plot]'
source .venv/bin/activate
```

The `pyproject.toml` pins the CUDA 12.8 PyTorch index. For a CPU-only machine,
drop the `[tool.uv.sources]` and `[[tool.uv.index]]` blocks at the bottom of
that file before installing.

## Web app

```bash
uv pip install -e '.[web]'
eigentoken serve --preload gpt2:center
# → http://localhost:8077
```

The default port is **8077**, chosen to stay clear of 8000 — the default for
most local model servers, which is exactly what tends to be running alongside
this. Change it with `--port N`, or `$EIGENTOKEN_PORT`; `--port 0` takes any
free port and prints the one it got. `--host`/`$EIGENTOKEN_HOST` do the same for
the bind address. A port that is already taken fails immediately with a message
saying so, rather than a traceback.

Everything the CLI does is in the browser, plus a **word cloud** explorer. Model
slots live in the sidebar; add as many as you like, and **add the same model
twice** to compare it with itself — its input matrix against its output matrix,
or centered against raw. Slots are remembered between visits.

### Word cloud — the associated-words tool

Give it one token and it draws that token's neighbourhood.

The default **radial** layout is the honest one: a point's **distance from the
centre is its true cosine distance** from the query, and only the angle comes
from a 2D projection. Labelled rings mark the cosine values, so nothing about
the radius is distorted. (A plain PCA scatter squeezes both axes at once and
invites reading structure into what is mostly projection error — it is offered
as a second mode for comparison, without the rings.)

Points are grouped by spherical k-means, which tends to split a word into its
distinct senses and registers. GPT-2's neighbourhood of `king` comes apart into
four clean groups:

| group | members |
|---|---|
| royalty | `␣kings ␣queen ␣prince ␣monarch ␣royal ␣emperor ␣throne` |
| secular authority | `␣leader ␣president ␣master ␣chief ␣mayor ␣governor` |
| feudal & religious titles | `␣lord ␣pope ␣bishop ␣knight ␣captain` |
| capitalised & place forms | `␣King ␣kingdom ␣KING ␣Kings ␣castle` |

(The `␣` prefix means the token includes the preceding space — see
[Why so many tokens start with `␣`](#why-so-many-tokens-start-with-).)

`␣King` and `␣KING` in that last group are not a different sense of the word —
they're `king` itself, just a different row in the vocabulary (see above).
**Hide case/space variants**, checked by default, drops every casing and
leading-space form of the query word (`KING`, `King`, `␣King`, ...) from the
results, so the neighbourhood shows only genuinely different words. The
[`neighbors`](#neighbors--what-the-model-thinks-a-word-is-near) tab has the
same toggle; the CLI equivalent is `--exclude-variants`.

Scroll to zoom, drag to pan — more labels appear as points spread out, because
labels are placed greedily and only where they fit. Click any token to recentre
the cloud on it — by token id, so you land on the exact row you clicked rather
than on whichever form the resolver would prefer for its text. A breadcrumb
trail lets you walk back. Click a group in the
legend to hide it.

### Serving it to other people

```bash
eigentoken serve --host 0.0.0.0 --port 8077 \
  --allow gpt2 gpt2-medium EleutherAI/pythia-160m bert-base-uncased \
  --max-models 6 --preload gpt2:center
```

- **There is no authentication.** Put it behind a reverse proxy that provides it.
- `--port`/`$EIGENTOKEN_PORT` and `--host`/`$EIGENTOKEN_HOST` set the bind
  address. Binding below port 1024 needs root — terminate TLS at the proxy and
  keep this on a high port.
- `--allow` (or `--allow-file`) restricts users to a fixed model list. Without
  it, any visitor can make the server download any model on the hub — fine on a
  workstation, not on a shared box.
- `--api-only` serves the API without the web interface, for a server that
  sits behind a copy of the UI hosted elsewhere; pair it with `--cors-origin`.
- Without `--allow`, load limits keep "any model" from becoming "any
  amount of disk, bandwidth and memory". They are on by default; 0 turns a
  numeric one off:
  - only **safetensors** weights load. `.bin` checkpoints are Python pickles
    that can carry code, so repos without safetensors are refused before
    anything is downloaded (`--allow-pickle-weights` to permit them);
  - model names must be **Hub repo ids**. Anything that looks like a path, or
    exists on the server's disk, is refused unless listed in `--allow`
    (`--allow-local-paths` to permit them);
  - `--max-download-gb` (default 10) and `--max-matrix-gb` (default 6) are
    checked against the Hub's file listing and the tensor's header **before**
    the weights download. While a model downloads, the UI shows its size;
  - `--max-concurrent-loads` (2) loads run at once, `--max-queued-loads` (4)
    more wait, and beyond that new loads get a 503 until one finishes;
  - `--loads-per-hour` (30) caps the new loads one client address can start.
    Using a resident model doesn't count. Behind a reverse proxy on the same
    machine the real address comes from `X-Forwarded-For`; elsewhere, set
    uvicorn's `FORWARDED_ALLOW_IPS` to the proxy's address.

  For a public server, lower the size limits to what the machine can hold:
  the defaults admit a 70B model's embedding table.
- Loaded models are shared by every visitor and stay resident until the
  server needs the room; then the least recently used go first.
  `--max-models` caps how many matrices stay resident, and `--max-memory-gb`
  caps their total size, which suits models of mixed sizes better. Room is
  made before a new matrix is read. Each `(model, matrix, center, whiten)`
  combination is its own entry, so a centered and an uncentered view of one
  model count as two. Visitors can't unload models by hand, since that would
  pull them out from under everyone else; `--allow-unload` brings the × back
  for a single-user server.
- `--max-disk-gb` does the same for downloaded weights: before a download, the
  least recently used models are deleted from the cache until it fits. A
  model that is loading or resident is never deleted. It needs
  `--cache-dir`, a directory of eigentoken's own, so it can't delete models
  other tools put in the shared Hugging Face cache.
- `--preload` warms the cache at boot. The syntax is
  `MODEL[:output][:center][:whiten]` — the UI defaults to centered, so
  `gpt2:center` is usually the useful one.
- `--trust-remote-code` is **off**; turn it on only for models you trust, since
  it lets the repo execute its own Python on your server.
- Set `HF_HOME` to put the model cache on a disk with room for it.
- Gated models (like `meta-llama/*`) need a token. Set `HF_TOKEN` (or put
  `HF_TOKEN=hf_...` in a `.env` file in the working directory — eigentoken
  loads it automatically on startup) rather than passing `--hf-token` on the
  command line, where it would sit in shell history.

Nothing blocks on a download. A request for a model that is not resident
returns `202` with a progress note, the browser polls, and the result appears
when the matrix lands. Concurrent requests for the same model share one load. A
failed load is sticky — it reports the error rather than silently retrying on
every poll — and the Resident panel has a retry button.

A minimal systemd unit:

```ini
[Service]
User=eigentoken
Environment=HF_HOME=/srv/eigentoken/hf
WorkingDirectory=/srv/eigentoken
ExecStart=/srv/eigentoken/.venv/bin/eigentoken serve --host 127.0.0.1 --port 8077 \
  --allow gpt2 EleutherAI/pythia-160m --max-models 4 --preload gpt2:center
Restart=on-failure
```

### API

The browser UI is a client of a plain JSON API, documented at `/api/docs` (so
`http://localhost:8077/api/docs` with the default port):

| endpoint | purpose |
|---|---|
| `GET /api/health`, `GET /api/models` | service state, resident matrices |
| `POST /api/models/load`, `/unload` | warm or free a matrix |
| `POST /api/cloud` | a token's neighbourhood, laid out in 2D |
| `POST /api/compare` | pairwise geometry, optional dimension attribution |
| `POST /api/neighbors` | nearest tokens |
| `POST /api/analogy` | `a : b :: c : ?` |
| `POST /api/axis` | build a direction from pairs, project probes |
| `GET /api/bias/presets`, `POST /api/bias` | WEAT bias test |
| `POST /api/crossmodel` | RSA, Procrustes, neighbour overlap |
| `POST /api/tokens` | tokenization diagnostics |

Every analysis endpoint takes **specs**, not bare model names:

```bash
curl -s localhost:8077/api/cloud -H 'Content-Type: application/json' -d '{
  "spec": {"model_id": "gpt2", "matrix": "input", "center": true},
  "word": "king", "k": 120, "clusters": 6, "mode": "radial"
}'
```

## Command line

Every analysis above is also a subcommand, for scripting and for reproducible
runs.

```bash
eigentoken info      -m gpt2
eigentoken tokens    -m gpt2 king queen man woman
eigentoken compare   -m gpt2 --center king queen man woman --dims 8
eigentoken neighbors -m gpt2 --center king -k 20
eigentoken analogy   -m gpt2 --center man king woman --expect queen
```

`examples/king_queen.sh` runs that sequence end to end;
`examples/gender_axis.sh` runs the axis analysis below;
`examples/bias_test.sh` runs the WEAT bias test below.

## What each command tells you

### `info` — read this first

Prints the matrix shape and, more importantly, how anisotropic the embedding
cloud is. LLM embeddings sit far off the origin, which inflates every cosine
similarity: in GPT-2 two unrelated tokens routinely score 0.4. If `offset
ratio` is above 1, pass `--center` to every other command so that similarity
reflects meaning rather than the shared mean vector.

### `tokens` — how the word actually enters the model

`king` is not one thing. `king`, ` king`, `King` and `KING` are separate rows
of the embedding table, and many words are not single tokens at all.
`eigentoken` prefers the leading-space form (` king`), which is how a word
normally appears mid-sentence; `--no-space` reverses that. Words with no
single-token form are mean-pooled across their subwords and every command
prints a warning when that happens — the geometry is approximate in that case.

You can also address rows directly: `id:1337` or `tok:Ġking`.

Tokens are printed decoded, not raw: byte-level BPE stores non-ASCII text as
mojibake in the vocabulary (`åĽ½çİĭ`), so `eigentoken` decodes each id back to
real text (`国王`). Fragments that are not valid text on their own fall back to
the raw piece.

### Why so many tokens start with `␣`

In byte-level BPE the space **before** a word is part of the token, so a word
has two unrelated rows in the embedding matrix:

| shown | raw vocab entry | id (gpt2) | where it appears |
|---|---|---|---|
| `␣water` | `Ġwater` | 1660 | mid-sentence: `the ␣water is cold` |
| `water` | `water` | 7050 | after a prefix: `fresh|water`, or at the very start |

They are genuinely different vectors — cosine 0.58 in gpt2, about as related as
`king` and `queen`. Most words in running text carry the leading space, so the
`␣` forms dominate any neighbour list. Seeing both `␣water` and `water` in one
cloud is a real result, not a duplicate.

`eigentoken` prefers the `␣` form when resolving a bare word, since that is how
a word normally occurs; `--no-space` reverses that, and `tok:` or `id:` name a
row exactly.

The marker is `␣` (U+2423, the Unicode symbol for a space) rather than an
underscore because underscores collide: 5555 tokens in Qwen's vocabulary
literally begin with `_`, and 22 do in gpt2's.

### `compare` — the core measurement

Cosine, angle, Euclidean distance, and unit-normalised distance for every pair,
plus a full cosine matrix for three or more words. `--dims K` breaks the
comparison down coordinate by coordinate: which dimensions carry the
similarity, which carry the difference, and how concentrated that is. A
difference spread over ~3 effective dimensions out of 768 usually means you
have found rogue outlier dimensions rather than anything semantic.

### `neighbors` — what the model thinks a word is near

Nearest tokens by cosine, dot product, or Euclidean distance. Dot product
weights by norm, which tracks token frequency; cosine on centered vectors is
the safer default.

`--exclude-variants` drops casing and leading-space forms of the query word
itself (`KING`, `King`, `␣King`, ...) — otherwise they're often the top hits,
crowding out words that are actually related.

### `analogy` — `a : b :: c : ?`

Computes `b - a + c` and ranks the vocabulary against it. With `--expect`, it
skips the guessing and reports the two numbers that actually matter:

- **offset parallelism** — `cos(b - a, expected - c)`. Whether the two offsets
  genuinely point the same way.
- **rank of expected** — where the true answer lands in the ranking.

Top-1 accuracy on analogy tasks is easy to fake, because `c` is often already
near the answer. Low parallelism with a correct top-1 means the model got it
right by accident.

### `axis` — build a direction from example pairs

```bash
eigentoken axis -m gpt2 --center \
  --pairs man:woman king:queen father:mother actor:actress \
  --probe doctor nurse engineer secretary \
  --extremes 15
```

Averages the offsets into a candidate direction, then checks whether that was a
legitimate thing to do: **offset coherence** (do the defining offsets agree
with each other?) and **PC1 variance share** (is the spread one direction or
many?). Both low means the contrast is not linear in this model and the
projections below are noise. The command says so rather than letting you read
meaning into an average of unrelated vectors.

`--extremes K` scans the whole vocabulary for the tokens furthest along the
axis in each direction — often the most revealing output in the tool.

### `bias` — measure association strength (WEAT)

```bash
eigentoken bias -m gpt2 --preset career_family
eigentoken bias --list-presets
eigentoken bias -m gpt2 -m bert-base-uncased --preset science_arts   # compare models
```

The Word Embedding Association Test (Caliskan, Bryson & Narayanan, 2017):
given a target-word split X vs. Y (e.g. career words vs. family words) and an
attribute-word split A vs. B (e.g. male vs. female terms), it measures how
much more strongly the model associates X with A — and Y with B — than chance
would predict, purely from cosine similarity in the embedding space. Two
numbers matter:

- **effect size** (Cohen's *d*) — how large the association is: below 0.2 is
  negligible, 0.5 is medium, above 0.8 is large.
- **p-value** — how likely an association this strong is to arise from
  randomly relabelling the words, estimated by a 10,000-draw Monte Carlo
  permutation test (not exact, but standard practice for WEAT).

Four built-in presets: `flowers_insects` (the original, non-social IAT
baseline — a strong, well-known effect, useful as a sanity check),
`career_family`, `science_arts`, and `math_arts` (all male vs. female terms).
Pass multiple `-m` to run the same test across several models in one go and
compare how strongly each one shows the effect. The built-in word lists are
short and illustrative, not a rigorous audit — use `--x`/`--y`/`--a`/`--b` to
run your own test with word lists suited to what you're actually trying to
measure.

### `pca` — the picture

Projects a word set to 2–3 dimensions with explained-variance ratios. `-o
out.png` writes a labelled scatter, `-o out.csv` writes the coordinates.

### `crossmodel` — do two models agree?

Coordinates are not comparable across models: dimension 417 means nothing in
common between two independently trained networks, and the widths usually
differ. What is comparable is the *shape* of the relationships.

```bash
eigentoken crossmodel -m gpt2 -m EleutherAI/pythia-160m \
  --words-file examples/words.txt --neighbors 10 --center
```

- **RSA** — build the word × word cosine matrix inside each model, then
  correlate the two matrices. Dimension-agnostic.
- **Procrustes disparity** — the residual after the best possible rotation
  aligning one space to the other (both reduced to a common dimension first).
  0 means the two clouds are the same shape.
- **Neighbourhood overlap** — Jaccard agreement between the models' top-K
  neighbour lists, compared on decoded strings.
- **Most divergent words** — which specific words the two models place
  differently. Usually the interesting output.

`--single-token-only` restricts the comparison to words that are one token in
every model, which removes pooling as a confound.

## Contextual embeddings

The static embedding table is only the model's starting point. `--layer N` with
`--context` uses the hidden state at layer N instead, so a word's vector
reflects the sentence it is in:

```bash
eigentoken compare -m gpt2 --layer 8 bank bank \
  --context 'I sat on the river bank' \
  --context 'I withdrew cash from the bank'
```

Layer 0 is the embedding table output, `-1` is the final layer. Two identical
words in different contexts is a valid and useful query.

In contextual mode, `neighbors` and `analogy` search the **output/unembedding**
matrix rather than the input table, since that is the space a residual-stream
vector actually lives in. This is a logit-lens readout, and it is labelled as
such in the output.

Contextual mode loads the full network; the static path reads only the
embedding tensor out of the safetensors shards, so a 70B model costs a few
hundred MB rather than 140GB.

## Options worth knowing

| flag | effect |
|---|---|
| `--center` | subtract the vocabulary mean; almost always what you want |
| `--whiten` | also normalise per-coordinate scale |
| `--matrix output` | analyse the unembedding matrix instead of the input table |
| `--pooling` | `mean` (default), `sum`, `first`, `last`, `norm-mean` for multi-token words |
| `--json` | machine-readable output from every command |
| `--device` | `cuda`, `cpu`, `mps`, or `auto` |
| `--verbose` | show checkpoint loading detail |

The same flags exist in the web UI as per-slot checkboxes.

Input and output embeddings often encode different things — a model can place
a token one way for reading and another for writing. Running the same analysis
with `--matrix input` and `--matrix output` is a cheap and informative
comparison in models where the two are not tied.

## Python API

```python
import eigentoken as et

m = et.load("gpt2", center=True)
king  = et.resolve(m, "king").vector
queen = et.resolve(m, "queen").vector

print(et.pair_stats(king, queen).cosine)

axis = et.build_axis(
    [(et.resolve(m, a).vector, et.resolve(m, b).vector)
     for a, b in [("man", "woman"), ("king", "queen")]],
    ["man->woman", "king->queen"],
)
print(axis.coherence, axis.is_linear)

for tid, score in et.neighbors(king, m.embeddings, k=10):
    print(m.tokenizer.convert_ids_to_tokens(tid), score)
```

## Tests

```bash
python tests/smoke.py                      # library + CLI, on gpt2
python tests/smoke.py bert-base-uncased    # and on other architectures
python tests/smoke.py EleutherAI/pythia-160m

eigentoken serve &                          # the API and UI tests need a server
python tests/api.py                        # 62 endpoint checks
python tests/limits.py                     # load limits, against the real Hub
node   tests/render.js                     # every UI render path
```

Real checkpoints, not mocks — every interesting failure here involves a real
tokenizer or a real checkpoint layout. Some checks worth knowing about: a
model's RSA against itself must be exactly 1 and its Procrustes disparity
exactly 0; raw cosines must come out higher than centered ones; and the cloud's
plotted radius must equal the real cosine distance for every point.

`tests/render.js` runs `app.js` under a small DOM shim in Node and asserts on
the serialised output, so the front end has real coverage without a headless
browser.

## Caveats

- Static embeddings are a weak signal in large models. Most of what an LLM
  "knows" about a word is built up across layers, not stored in its embedding
  row. Use `--layer` for the deeper picture.
- Distances between subword-pooled vectors are approximate. Prefer
  single-token words when the comparison has to be precise.
- A handful of coordinates in most LLMs have enormous magnitude and dominate
  every dot product. `--center` helps; `--whiten` helps more; `compare --dims`
  will show you when it is happening.
- `bias`'s built-in presets are short word lists on one embedding matrix: a
  real signal, but not a rigorous audit of a model. Treat a result as a
  starting point for investigation, not a conclusion about the model or its
  training data.
