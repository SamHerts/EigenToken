#!/usr/bin/env bash
# Run the web app for a small group, with a fixed model list so visitors
# cannot make the server download arbitrary checkpoints.
#
#   PORT=9000 ./examples/serve.sh      # or --port 0 for any free port
set -euo pipefail

exec eigentoken serve \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8077}" \
  --max-models 6 \
  --preload gpt2:center \
  --allow \
    gpt2 gpt2-medium distilgpt2 \
    EleutherAI/pythia-160m EleutherAI/pythia-410m \
    bert-base-uncased roberta-base \
    HuggingFaceTB/SmolLM2-360M
