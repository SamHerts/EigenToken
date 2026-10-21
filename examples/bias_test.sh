#!/usr/bin/env bash
# How strongly does the model associate two word sets? Run with a model id,
# e.g.  ./examples/bias_test.sh gpt2
set -euo pipefail
MODEL="${1:-gpt2}"

eigentoken bias -m "$MODEL" --preset career_family
eigentoken bias -m "$MODEL" --preset science_arts
