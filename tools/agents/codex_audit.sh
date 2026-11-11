#!/usr/bin/env bash
# Run one Codex audit task (phase 6) in its own worktree. Read-mostly: the report is the deliverable.
#   tools/agents/codex_audit.sh <name> "<space-separated demo dirs>"
set -euo pipefail
name="$1"; dirs="$2"
repo="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
wt_root="$(dirname "$repo")/opengles3-demos-worktrees"
wt="$wt_root/$name"
mkdir -p "$wt_root"
[ -d "$wt" ] || git -C "$repo" worktree add -q "$wt" -b "codex/$name" main
prompt="You are auditing part of the opengles3-demos project (current directory is a git worktree).
Read docs/agent-briefs/AUDIT_BRIEF.md completely and follow it exactly. meta-gl lives at
/rv/data/development/github.com/libcna/meta-gl (read-only). CPU LIMIT: build with -j 2 only.
Do NOT modify any files except that you may write your report; do not commit.

YOUR GROUP (audit every one of these directories):
$dirs"
codex exec -C "$wt" -s workspace-write -c model_reasoning_effort='"high"' --add-dir "$wt" \
    -o "$wt_root/$name.report.md" "$prompt" > "$wt_root/$name.log" 2>&1
echo "codex audit '$name' finished; report: $wt_root/$name.report.md"
