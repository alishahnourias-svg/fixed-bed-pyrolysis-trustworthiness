#!/usr/bin/env bash
set -euo pipefail
REPO_NAME="${1:-fixed-bed-pyrolysis-trustworthiness}"
REMOTE_URL="${2:-}"
cd "$(dirname "$0")/.."
python tools/audit_public_repository.py

git config user.name >/dev/null 2>&1 || { echo "Set git user.name first"; exit 1; }
git config user.email >/dev/null 2>&1 || { echo "Set git user.email first"; exit 1; }

[ -d .git ] || git init -b main
git add .
if ! git diff --cached --quiet; then
  git commit -m "Prepare public reproducibility repository v1.0.0-rc4"
fi

if [ -n "$REMOTE_URL" ]; then
  git remote get-url origin >/dev/null 2>&1 && git remote set-url origin "$REMOTE_URL" || git remote add origin "$REMOTE_URL"
  git push -u origin main
elif command -v gh >/dev/null 2>&1; then
  gh auth status
  gh repo create "$REPO_NAME" --public --source . --remote origin --push \
    --description "Code and aggregate reproducibility package for trustworthiness-aware ML analysis of fixed-bed biomass pyrolysis yields."
else
  echo "Create an empty public GitHub repository named $REPO_NAME, then rerun with its HTTPS URL as the second argument."
fi
