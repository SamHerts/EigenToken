#!/usr/bin/env bash
# The canonical probe: how does the model place king, queen, man and woman?
set -euo pipefail
MODEL="${1:-gpt2}"

echo "== is this space safe to measure with cosines? =="
eigentoken info -m "$MODEL"

echo "== how do these words tokenize? =="
eigentoken tokens -m "$MODEL" king queen man woman

echo "== pairwise geometry, with per-dimension breakdown =="
eigentoken compare -m "$MODEL" --center king queen man woman --dims 8

echo "== does the arithmetic work? =="
eigentoken analogy -m "$MODEL" --center man king woman --expect queen
