#!/usr/bin/env bash
set -euo pipefail
test "$#" -ge 2 || { echo "usage: $0 MESSAGE PATH..." >&2; exit 2; }
message="$1"; shift
repo="$(git rev-parse --show-toplevel)"; cd "$repo"
test "$(git branch --show-current)" = main || { echo "FAIL: not on main" >&2; exit 1; }
git add -- "$@"
git diff --cached --check
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|npy|npz|bin|zip)$' >/dev/null; then
  echo "FAIL: large/raw binary extension staged" >&2; exit 1
fi
if git diff --cached | grep -Ei '(password|api[_-]?key|secret|token|webhook)[[:space:]]*[:=][[:space:]]*["'\''][^"'\'']+' >/dev/null; then
  echo "FAIL: possible secret in staged diff" >&2; exit 1
fi
git commit -m "$message"
git fetch origin main
if ! git merge-base --is-ancestor origin/main HEAD; then
  git rebase origin/main
fi
git push origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
echo "PASS: local main == origin/main at $(git rev-parse HEAD)"
