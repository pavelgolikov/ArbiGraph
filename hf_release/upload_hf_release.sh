#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_REPO_ID="${HF_REPO_ID:-PavelGolikov/arbigraph}"

python -B hf_release/build_hf_release.py
python -B hf_release/validate_hf_release.py
hf upload "$HF_REPO_ID" hf_release/build/arbigraph . --repo-type=dataset --commit-message "Release ArbiGraph benchmark datasets"
