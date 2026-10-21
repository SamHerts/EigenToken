#!/usr/bin/env bash
# Does the model encode gender as a single direction, and what does it place
# along it?  Run with a model id, e.g.  ./examples/gender_axis.sh gpt2
set -euo pipefail
MODEL="${1:-gpt2}"

eigentoken axis -m "$MODEL" --center \
  --pairs man:woman king:queen father:mother actor:actress brother:sister \
  --probe doctor nurse engineer teacher scientist secretary programmer \
  --extremes 15
