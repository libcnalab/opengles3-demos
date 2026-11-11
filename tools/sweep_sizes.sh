#!/usr/bin/env bash
# Window-size sweep: run every executable headless with --validate at several surface sizes on llvmpipe (software,
# never the GPU) and list every run that is not PASS or SKIP_UNSUPPORTED.
#   tools/sweep_sizes.sh [build-dir] [sizes...]     e.g. tools/sweep_sizes.sh build/debug 333x777 1000x200 200x1000
# Parallelism: 4 processes (running demos is not compilation, but keep the machine usable).
set -euo pipefail
build="${1:-build/debug}"; shift || true
sizes=("${@:-333x777 1000x200 200x1000}"); [ $# -eq 0 ] && sizes=(333x777 1000x200 200x1000)
out="$(mktemp -d)"
for size in "${sizes[@]}"; do
    w="${size%x*}"; h="${size#*x}"
    ls "$build/bin" | grep -v '^test-driver-crash' | xargs -P 4 -I{} sh -c \
        "EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe timeout 120 '$build/bin/{}' --headless --validate --width $w --height $h > '$out/{}_$size.log' 2>&1;
         r=\$(grep -a -o '^RESULT: [A-Z_]*' '$out/{}_$size.log' | tail -1); echo \"{} $size \${r:-RESULT: CRASHED}\""
done | grep -vE 'RESULT: (PASS|SKIP_UNSUPPORTED)$' | sort || true
echo "logs: $out"
