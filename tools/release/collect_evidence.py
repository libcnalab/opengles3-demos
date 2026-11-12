#!/usr/bin/env python3
"""Run one CTest matrix and record machine-readable evidence of it.

    tools/release/collect_evidence.py --build build/debug --label debug --out <dir> [-- <extra ctest args>]

Writes into <dir>:
  manifest.json   source and meta-gl revisions, build configuration, compiler, tool and driver versions, the exact
                  ctest command, start/end time, totals, and the kernel-log check for amdgpu faults during the run
  results.csv     one row per registered test: CTest status, outcome from the result JSON, ES/GLSL version,
                  renderer, checks passed/failed, known deviations (XFAIL)
  ctest.log       the full CTest output        junit.xml   CTest's JUnit report
  SUMMARY.md      a short human-readable digest of the above

manifest.json, results.csv and SUMMARY.md are small enough to commit (docs/evidence/); ctest.log and junit.xml are
not, and belong in a release archive.

The working trees of opengles3-demos and meta-gl must be clean (pass --allow-dirty to record a dirty run anyway;
the manifest then says so). A result JSON counts only if it was written during this run, so leftovers of earlier
runs are never reported as evidence.
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def git_state(path):
    return {
        "path": os.path.abspath(path),
        "commit": sh(["git", "-C", path, "rev-parse", "HEAD"]),
        "describe": sh(["git", "-C", path, "describe", "--tags", "--always", "--long"]),
        "branch": sh(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(sh(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"])),
    }


def cmake_cache(build):
    cache = {}
    with open(os.path.join(build, "CMakeCache.txt")) as f:
        for line in f:
            m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line.rstrip("\n"))
            if m:
                cache[m.group(1)] = m.group(2)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="configured and built CMake build directory")
    ap.add_argument("--label", required=True, help="name of this matrix, e.g. debug, sanitize, headless-ci")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("ctest_args", nargs="*", help="extra ctest arguments (after --)")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build = os.path.abspath(args.build)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    cache = cmake_cache(build)
    metagl_dir = os.path.normpath(cache.get("OGLES3_METAGL_SOURCE_DIR", os.path.join(repo, "..", "..", "libcna", "meta-gl")))

    src, mgl = git_state(repo), git_state(metagl_dir)
    if (src["dirty"] or mgl["dirty"]) and not args.allow_dirty:
        sys.exit("error: a working tree is dirty (opengles3-demos: %s, meta-gl: %s); commit first or pass --allow-dirty"
                 % (src["dirty"], mgl["dirty"]))

    compiler = cache.get("CMAKE_CXX_COMPILER", "c++")
    results_dir = os.path.join(build, "test-results")
    junit = os.path.join(out, "junit.xml")
    cmd = ["ctest", "--test-dir", build, "-j", str(args.jobs), "--output-on-failure", "--output-junit", junit] + args.ctest_args
    start = time.time()
    start_iso = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(os.path.join(out, "ctest.log"), "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    end = time.time()
    end_iso = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    # CTest status per test, from the JUnit report.
    ctest_status = {}
    try:
        import xml.etree.ElementTree as ET
        for tc in ET.parse(junit).getroot().iter("testcase"):
            status = tc.get("status", "")
            if not status:
                status = "failed" if tc.find("failure") is not None else ("skipped" if tc.find("skipped") is not None else "passed")
            ctest_status[tc.get("name")] = status
    except (OSError, ET.ParseError):
        pass

    # Result JSONs written by this run only.
    by_test = {}
    for fn in os.listdir(results_dir) if os.path.isdir(results_dir) else []:
        if not fn.endswith(".json"):
            continue
        path = os.path.join(results_dir, fn)
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("run_time", 0) + 1 >= start and os.path.getmtime(path) >= start - 1:
            by_test[r.get("test_id", "")] = r

    rows, drivers = [], {}
    outcome_counts, status_counts = {}, {}
    for name in sorted(ctest_status):
        r = by_test.get(name)
        st = ctest_status[name]
        status_counts[st] = status_counts.get(st, 0) + 1
        outcome = r["outcome"] if r else ("NO_RESULT_JSON" if ":" in name else "N/A (no result JSON by design)")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        xfail = sum(1 for c in (r or {}).get("demo_checks", []) if "XFAIL known driver deviation" in c.get("detail", "")
                    or "XFAIL: known driver deviation" in c.get("detail", ""))
        rows.append([name, st, outcome, (r or {}).get("reason", ""), (r or {}).get("context_version", ""),
                     (r or {}).get("glsl_version", ""), (r or {}).get("gl_renderer", ""), (r or {}).get("gl_version", ""),
                     (r or {}).get("checks_passed", ""), (r or {}).get("checks_failed", ""), xfail,
                     (r or {}).get("version_override", "")])
        if r and r.get("gl_renderer"):
            key = r["gl_renderer"] + " | " + r.get("gl_version", "")
            d = drivers.setdefault(key, {"gl_renderer": r["gl_renderer"], "gl_version": r.get("gl_version", ""),
                                         "glsl_versions": set(), "platforms": set(), "runs": 0})
            d["glsl_versions"].add(r.get("glsl_version", ""))
            d["platforms"].add(r.get("platform", ""))
            d["runs"] += 1

    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["test", "ctest_status", "outcome", "reason", "context_es", "glsl_version", "gl_renderer", "gl_version",
                    "checks_passed", "checks_failed", "xfail", "version_override"])
        w.writerows(rows)

    since = dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
    until = dt.datetime.fromtimestamp(end + 5).strftime("%Y-%m-%d %H:%M:%S")
    klog = sh(["journalctl", "-k", "--no-pager", "--since", since, "--until", until])
    gpu_faults = [l for l in klog.splitlines() if "amdgpu" in l and re.search(r"fault|timeout|reset|hang", l, re.I)]

    pkgs = sh(["dpkg-query", "-W", "-f", "${Package} ${Version}\\n", "libgl1-mesa-dri", "libegl-mesa0", "mesa-vulkan-drivers"])
    manifest = {
        "label": args.label,
        "opengles3_demos": src,
        "meta_gl": mgl,
        "build": {
            "dir": build,
            "CMAKE_BUILD_TYPE": cache.get("CMAKE_BUILD_TYPE", ""),
            "options": {k: v for k, v in sorted(cache.items()) if k.startswith("OGLES3_")},
            "compiler": compiler,
            "compiler_version": sh([compiler, "--version"]).splitlines()[0] if sh([compiler, "--version"]) else "",
            "cmake_version": sh(["cmake", "--version"]).splitlines()[0] if sh(["cmake", "--version"]) else "",
        },
        "host": {"kernel": platform.release(), "machine": platform.machine(), "mesa_packages": pkgs.splitlines()},
        "drivers_seen": [{**d, "glsl_versions": sorted(d["glsl_versions"]), "platforms": sorted(d["platforms"])}
                         for d in sorted(drivers.values(), key=lambda d: d["gl_renderer"] + d["gl_version"])],
        "command": shlex.join(cmd),
        "ctest_exit_code": rc,
        "start": start_iso,
        "end": end_iso,
        "tests_registered": len(ctest_status),
        "ctest_status_counts": status_counts,
        "outcome_counts": outcome_counts,
        "gpu_safety": {
            "OGLES3_RUN_GPU_FAULT_CASES": os.environ.get("OGLES3_RUN_GPU_FAULT_CASES", ""),
            "angle_vulkan_icd": cache.get("OGLES3_ANGLE_VULKAN_ICD", ""),
            "kernel_log_checked": bool(klog) or klog == "",
            "amdgpu_fault_lines_during_run": gpu_faults,
        },
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write(f"# Evidence: {args.label}\n\n")
        f.write(f"- opengles3-demos `{src['commit']}` ({src['describe']}){' **dirty**' if src['dirty'] else ''}\n")
        f.write(f"- meta-gl `{mgl['commit']}` ({mgl['describe']}){' **dirty**' if mgl['dirty'] else ''}\n")
        f.write(f"- build `{manifest['build']['CMAKE_BUILD_TYPE']}`, {manifest['build']['compiler_version']}, "
                f"{manifest['build']['cmake_version']}\n")
        f.write(f"- run {start_iso} .. {end_iso}; `{manifest['command']}`; ctest exit code {rc}\n")
        f.write(f"- registered tests: {len(ctest_status)}; CTest status: "
                + ", ".join(f"{k} {v}" for k, v in sorted(status_counts.items())) + "\n")
        f.write("- outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcome_counts.items())) + "\n")
        f.write(f"- amdgpu fault/timeout/reset lines in the kernel log during the run: {len(gpu_faults)}\n\n")
        f.write("## Drivers seen\n\n| GL_RENDERER | GL_VERSION | GLSL | runs |\n|---|---|---|---|\n")
        for d in manifest["drivers_seen"]:
            f.write(f"| {d['gl_renderer']} | {d['gl_version']} | {', '.join(d['glsl_versions'])} | {d['runs']} |\n")
        not_pass = [r for r in rows if r[2] != "PASS"]
        f.write("\n## Runs that did not PASS\n\n| test | ctest | outcome | reason |\n|---|---|---|---|\n")
        for r in not_pass:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {str(r[3]).replace('|', '/')[:160]} |\n")
    print(f"{args.label}: ctest exit {rc}; " + ", ".join(f"{k} {v}" for k, v in sorted(outcome_counts.items())) + f"; evidence in {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
