#!/usr/bin/env bash
# Fails when the staged changes exceed the per-commit line limit.
# Usage: check-commit-size.sh [limit]   (default limit: 200)
set -euo pipefail

limit="${1:-200}"
numstat=$(git diff --cached --numstat)

if [[ -z "$numstat" ]]; then
  echo "Nothing staged." >&2
  exit 1
fi

total=0
while IFS=$'\t' read -r added deleted path; do
  if [[ "$added" == "-" ]]; then
    printf '%6s  %s (binary, not counted)\n' "-" "$path"
    continue
  fi
  changed=$((added + deleted))
  total=$((total + changed))
  printf '%6d  %s\n' "$changed" "$path"
done <<< "$numstat"

echo "Total: $total changed lines (limit $limit)"

if (( total > limit )); then
  echo "Commit too large: split it into smaller logical commits (git restore --staged, then git add -p)." >&2
  exit 1
fi
