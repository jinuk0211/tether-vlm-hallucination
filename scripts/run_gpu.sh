#!/usr/bin/env bash
set -euo pipefail

config="${1:?Usage: bash scripts/run_gpu.sh CONFIG DEVELOPMENT_JSONL EVALUATION_JSONL OUTPUT_DIR [tune|test]}"
development="${2:?development manifest required}"
evaluation="${3:?evaluation manifest required}"
output="${4:?output directory required}"
split="${5:-tune}"
if [[ "$split" != "tune" && "$split" != "test" ]]; then
  echo "Split must be tune or test" >&2
  exit 2
fi

python -m tether verify-splits --manifests "$development"
if [[ "$development" != "$evaluation" ]]; then
  python -m tether verify-splits --manifests "$development" "$evaluation"
fi
python -m tether discover --config "$config" --manifest "$development" --output "$output/selection.json"
python -m tether fit --config "$config" --manifest "$development" --selection "$output/selection.json" --output "$output/ephs.json"
for method in revis grains vti; do
  extra=()
  if [[ "$method" == "revis" ]]; then
    extra=(--calibration "$development")
  fi
  python -m tether extract --method "$method" --config "$config" --manifest "$development" --output "$output/$method.pt" "${extra[@]}"
done
for method in base ephs revis rudder grains vti; do
  extra=()
  if [[ "$method" == "ephs" ]]; then
    extra=(--artifact "$output/ephs.json")
  elif [[ "$method" != "base" && "$method" != "rudder" ]]; then
    extra=(--artifact "$output/$method.pt")
  fi
  python -m tether run --method "$method" --config "$config" --manifest "$evaluation" --split "$split" --output "$output/$method.jsonl" "${extra[@]}"
done
echo "Predictions are complete. Run 'tether compare' with the actual CHAIR/Causal scorers."
