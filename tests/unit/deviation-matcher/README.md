# unit-deviation-matcher

A CPU-only regression test for `demo::deviation_matches` (`common/include/demo/check.hpp`), the rule that decides on
which driver a documented deviation is reported as XFAIL instead of FAIL. It creates no GL context, has no
`demo.json` and claims no API coverage; it runs once in CTest (label `unit`).

It pins the rules with renderer and version strings copied from real result files: "Mesa" selects every Mesa driver
and never ANGLE; Mesa driver names never match ANGLE whatever its Vulkan backend is; "llvmpipe" also matches zink on
lavapipe; "ANGLE" matches only ANGLE; an empty name matches nothing. The first rule guards a regression found in the
2026-11-11 review: ANGLE's GL_RENDERER names its backend ("ANGLE (Mesa, Vulkan ... (llvmpipe ...))"), so plain
substring matching let Mesa deviations excuse ANGLE failures.

Run: `ctest --preset debug -R unit-deviation-matcher` or `./build/debug/bin/unit-deviation-matcher` (exit 0 = PASS).
