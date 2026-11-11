#!/usr/bin/env bash
# Codex educational review (read-only report).  tools/agents/codex_review.sh <name> "<dirs>"
set -euo pipefail
name="$1"; dirs="$2"
repo="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
wt_root="$(dirname "$repo")/opengles3-demos-worktrees"; wt="$wt_root/$name"
mkdir -p "$wt_root"
[ -d "$wt" ] || git -C "$repo" worktree add -q "$wt" -b "codex/$name" main
prompt="You are reviewing part of the opengles3-demos project (current directory is a git worktree) for educational
quality. Read docs/agent-briefs/EDUCATION_REVIEW_BRIEF.md and follow it exactly. Do NOT modify files except writing your
report to ./REVIEW_REPORT.md; do not commit. CPU LIMIT: if you build at all, use -j 1.

YOUR GROUP:
$dirs"
codex exec -C "$wt" -s workspace-write -c model_reasoning_effort='"medium"' --add-dir "$wt" \
    -o "$wt_root/$name.final.md" "$prompt" > "$wt_root/$name.log" 2>&1
echo "codex review '$name' finished: $wt/REVIEW_REPORT.md"
