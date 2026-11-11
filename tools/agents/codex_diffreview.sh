#!/usr/bin/env bash
# Codex independent review of a diff range.  tools/agents/codex_diffreview.sh <name> "<dirs>"
set -euo pipefail
name="$1"; dirs="$2"
repo="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
wt_root="$(dirname "$repo")/opengles3-demos-worktrees"; wt="$wt_root/$name"
mkdir -p "$wt_root"
[ -d "$wt" ] || git -C "$repo" worktree add -q "$wt" -b "codex/$name" main
prompt="You are an independent reviewer for the opengles3-demos project (current directory is a git worktree). Read
docs/agent-briefs/DIFF_REVIEW_BRIEF.md and follow it exactly. meta-gl is at /rv/data/development/github.com/libcna/meta-gl
(configure with -DOGLES3_METAGL_SOURCE_DIR=that path). CPU LIMIT: -j 1. Never run anything on the GPU (llvmpipe only).
Write ./DIFF_REVIEW.md; modify nothing else; do not commit.

YOUR DIRECTORIES:
$dirs"
codex exec -C "$wt" -s workspace-write -c model_reasoning_effort='"high"' --add-dir "$wt" \
    -o "$wt_root/$name.final.md" "$prompt" > "$wt_root/$name.log" 2>&1
echo "codex diff review '$name' finished: $wt/DIFF_REVIEW.md"
