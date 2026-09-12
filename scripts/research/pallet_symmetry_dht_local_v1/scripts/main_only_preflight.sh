#!/usr/bin/env bash
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
cd "$repo"
test "$(git branch --show-current)" = main || { echo "FAIL: current branch is not main" >&2; exit 1; }
git fetch origin main
git merge-base --is-ancestor origin/main HEAD || { echo "FAIL: main diverged from origin/main" >&2; exit 1; }
if test -n "$(git status --porcelain)"; then
  echo "FAIL: working tree is dirty; preserve changes and inspect before proceeding" >&2
  git status --short >&2
  exit 1
fi
echo "PASS: main; origin/main is an ancestor; worktree clean"
