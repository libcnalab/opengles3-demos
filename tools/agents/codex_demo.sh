#!/usr/bin/env bash
# Run one Codex CLI task in its own git worktree of opengles3-demos.
#
#   tools/agents/codex_demo.sh <task-name> <task-description-file>
#
# Creates ../opengles3-demos-worktrees/<task-name> on branch codex/<task-name> (from main), runs
# `codex exec` there non-interactively with the shared brief, and writes Codex's final message to
# ../opengles3-demos-worktrees/<task-name>.report.md. The integrating agent reviews, re-runs the
# full matrix and merges. Nothing is pushed.
set -euo pipefail
name="$1"; task_file="$2"
repo="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
wt_root="$(dirname "$repo")/opengles3-demos-worktrees"
wt="$wt_root/$name"
mkdir -p "$wt_root"
if [ ! -d "$wt" ]; then
    git -C "$repo" worktree add -q "$wt" -b "codex/$name" main
fi
prompt="You are working in a git worktree of the opengles3-demos project (current directory).
Read docs/agent-briefs/DEMO_BRIEF.md completely and follow it exactly (deliverables, validation
rules, build/test commands, report format). meta-gl lives at
/rv/data/development/github.com/libcna/meta-gl (read-only for you). Configure with
-DOGLES3_METAGL_SOURCE_DIR=/rv/data/development/github.com/libcna/meta-gl. If the GPU is not
accessible inside your sandbox, run the llvmpipe and zink tests (ctest -L llvmpipe / -L zink) and say
so in your report.
CPU LIMIT: the owner allows at most 8 CPU cores for compilation machine-wide and several agents build
at the same time, so ALWAYS build with -j 2 (cmake --build --preset debug -j 2), never more.

YOUR TASK:
$(cat "$task_file")"
codex exec -C "$wt" -s workspace-write \
    -c model_reasoning_effort='"high"' \
    --add-dir "$wt" \
    -o "$wt_root/$name.report.md" \
    "$prompt" > "$wt_root/$name.log" 2>&1
echo "codex task '$name' finished; report: $wt_root/$name.report.md"
