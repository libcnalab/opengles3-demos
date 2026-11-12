#!/usr/bin/env python3
"""Generate coverage documents from manifests and test results.

Inputs
  inventory/es_core_commands.csv     registry-derived core entry points (358)
  inventory/features.csv             semantic feature inventory
  inventory/glsl_features.csv        shader-language inventory
  demos/**/demo.json, tests/**/demo.json, shader-tests/**/demo.json   per-executable manifests
  <results>/*.json                   result files written by the executables under CTest
  <metagl>/include/metagl/Functions.hpp   to confirm wrapper presence

Outputs (all generated - do not edit by hand)
  docs/API_COVERAGE.csv
  docs/generated/COVERAGE_SUMMARY.md
  docs/generated/FEATURE_STATUS.md
  docs/generated/GLSL_STATUS.md
  docs/generated/DEMO_STATUS.md

Status rules (see docs/VALIDATION.md):
  NOT_STARTED            no executable claims it
  DOCUMENTED_ONLY        inventory method is "doc" (undefined behaviour): documented, never asserted
  IMPLEMENTED_UNVERIFIED claimed by a manifest, no executable built
  STALE_RESULTS          result files exist, but none matches the current executable or sources (rebuild + rerun)
  BUILD_PASSED           executable built, no result files
  FAILED                 any run FAILED
  BLOCKED_BY_META_GL     every run reported BLOCKED_META_GL
  BLOCKED_BY_PLATFORM    every run reported BLOCKED_ENVIRONMENT
  UNSUPPORTED_BY_CONTEXT every run reported SKIP_UNSUPPORTED
  RUNTIME_PASSED         >=1 validated PASS
  VISUALLY_VERIFIED      RUNTIME_PASSED + manifest records a human/agent review of the capture
  FULLY_VALIDATED        every registered matrix run is a validated PASS or a SKIP_UNSUPPORTED, gpu and
                         llvmpipe pass in every version mode, and the manifest declares at least one
                         non-trivial validation method

Registered runs: the CTest registration of each build directory (`ctest --show-only=json-v1`) is the list of runs
an executable must have. A registered run without a fresh result (never run, filtered out, or stale) prevents
FULLY_VALIDATED, and a result file whose test is no longer registered is ignored. Without a build directory the
check is skipped and the summary says so.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

RANK = ["NOT_STARTED", "DOCUMENTED_ONLY", "IMPLEMENTED_UNVERIFIED", "STALE_RESULTS", "FAILED", "BLOCKED_BY_PLATFORM", "BLOCKED_BY_META_GL",
        "UNSUPPORTED_BY_CONTEXT", "BUILD_PASSED", "RUNTIME_PASSED", "VISUALLY_VERIFIED", "FULLY_VALIDATED"]


def best(statuses):
    statuses = list(statuses)
    if not statuses:
        return "NOT_STARTED"
    return max(statuses, key=RANK.index)


def load_manifests(repo):
    manifests = []
    for pattern in ("demos/*/*/demo.json", "tests/*/*/demo.json", "shader-tests/demo.json", "shader-tests/*/demo.json", "tests/*/demo.json"):
        for path in sorted(glob.glob(os.path.join(repo, pattern))):
            with open(path) as f:
                m = json.load(f)
            m["_dir"] = os.path.relpath(os.path.dirname(path), repo)
            # Data-driven test executables: feature IDs come from "// @feature:" annotations.
            for case_dir in m.get("case_dirs", []):
                for case in sorted(glob.glob(os.path.join(os.path.dirname(path), case_dir, "*"))):
                    with open(case, errors="replace") as cf:
                        for line in cf:
                            mt = re.match(r"\s*//\s*@feature:\s*(\S+)", line)
                            if not mt:
                                continue
                            key = "glsl_features" if mt.group(1).startswith("GLSL") else "features"
                            if mt.group(1) not in m.setdefault(key, []):
                                m[key].append(mt.group(1))
            manifests.append(m)
    return manifests


def load_results(result_dirs):
    by_demo = defaultdict(list)
    for d in result_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                with open(path) as f:
                    r = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            by_demo[r.get("demo", "")].append(r)
    return by_demo


CODE_EXT = (".cpp", ".hpp", ".h", ".cc", ".inl")


def newest_mtime(paths, code=None):
    """Newest mtime of the files under paths. code=True: only compiled sources; code=False: everything else
    that affects a run (shaders, data, CMake glue); None: both. Documentation and manifests never count."""
    newest = 0.0
    for root in paths:
        files = [root] if os.path.isfile(root) else [os.path.join(d, fn) for d, _, fs in os.walk(root) for fn in fs]
        for path in files:
            if path.endswith((".md", ".json")):
                continue
            is_code = path.endswith(CODE_EXT)
            if code is None or code == is_code:
                newest = max(newest, os.path.getmtime(path))
    return newest


def fresh_runs(m, runs, repo):
    """Keep only results produced by the current executable from current sources.

    A result counts as evidence only if (1) it records the executable that produced it, (2) that executable
    still has the recorded mtime (not rebuilt since), (3) no compiled source of the demo or of common/ is newer
    than the executable, and (4) no run-time input (shaders, data, test registration in cmake/) is newer than the
    run itself."""
    roots = [os.path.join(repo, m["_dir"]), os.path.join(repo, "common")]
    code_src = newest_mtime(roots, code=True)
    run_src = max(newest_mtime(roots, code=False), newest_mtime([os.path.join(repo, "cmake", "OglesDemo.cmake")]))
    fresh, stale = [], []
    for r in runs:
        exe = r.get("executable", "")
        ok = bool(exe) and os.path.exists(exe)
        if ok:
            exe_mtime = os.path.getmtime(exe)
            ok = (int(exe_mtime) == int(r.get("executable_mtime", -1)) and code_src <= exe_mtime + 1
                  and run_src <= r.get("run_time", 0) + 1)
        (fresh if ok else stale).append(r)
    return fresh, stale


def registered_tests(ctest_dirs):
    """Test names registered by CTest in the given build directories, grouped by executable (the part before the
    first ':'). Returns None when no directory could be queried."""
    import subprocess
    by_demo = defaultdict(set)
    queried = False
    for d in ctest_dirs:
        if not os.path.exists(os.path.join(d, "CTestTestfile.cmake")):
            continue
        try:
            out = subprocess.run(["ctest", "--show-only=json-v1", "--test-dir", d], capture_output=True, text=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        queried = True
        for t in json.loads(out).get("tests", []):
            name = t.get("name", "")
            by_demo[name.split(":", 1)[0]].add(name)
    return by_demo if queried else None


def demo_status(m, results, bin_dirs):
    runs = results.get(m["name"], [])
    built = any(os.path.exists(os.path.join(b, m["name"])) for b in bin_dirs)
    if not runs:
        if m.get("_stale"):
            return "STALE_RESULTS", runs
        return ("BUILD_PASSED" if built else "IMPLEMENTED_UNVERIFIED"), runs
    outcomes = [r["outcome"] for r in runs]
    if "FAIL" in outcomes or "CRASHED" in outcomes:
        return "FAILED", runs
    passed = [r for r in runs if r["outcome"] == "PASS" and r.get("validated")]
    if not passed:
        if all(o == "BLOCKED_META_GL" for o in outcomes):
            return "BLOCKED_BY_META_GL", runs
        if all(o == "BLOCKED_ENVIRONMENT" for o in outcomes):
            return "BLOCKED_BY_PLATFORM", runs
        if all(o == "SKIP_UNSUPPORTED" for o in outcomes):
            return "UNSUPPORTED_BY_CONTEXT", runs
        return "BUILD_PASSED", runs
    methods = set(m.get("validation", [])) - {"gl-error"}
    ids = {r.get("test_id", "") for r in runs}
    drivers = {i.split(":")[1] for i in ids if i.count(":") == 2}
    modes = {i.split(":")[2] for i in ids if i.count(":") == 2}
    expected_modes = {"es-max"} if m.get("min_es") == "3.2" else {"es-max", "es-min"}
    # Every run is a validated PASS or a SKIP_UNSUPPORTED (context below min_es, e.g. softpipe for
    # ES 3.2 demos); gpu and llvmpipe must both pass in every version mode.
    ok_runs = all((r["outcome"] == "PASS" and r.get("validated")) or r["outcome"] == "SKIP_UNSUPPORTED" for r in runs)
    pass_ids = {r.get("test_id", "") for r in passed}
    need = {f"{m['name']}:{d}:{md}" for d in ("gpu", "llvmpipe") for md in expected_modes}
    full = ok_runs and bool(methods) and need <= pass_ids and not m.get("_missing")
    if full:
        return "FULLY_VALIDATED", runs
    if m.get("visual_review"):
        return "VISUALLY_VERIFIED", runs
    return "RUNTIME_PASSED", runs


def common_helper_calls(repo, extra_files=()):
    """Map each function defined in common/src/{gl_util,check,image}.cpp (and in extra_files, e.g. meta-gl's
    inline convenience overloads such as glTexParameter -> glTexParameteri/f) to the GL entry points it calls,
    including through other helpers (fixed point). Name-based, so overloads share one entry: a heuristic lint."""
    bodies = {}
    paths = [os.path.join(repo, "common", "src", n) for n in ("gl_util.cpp", "check.cpp", "image.cpp")]
    for path in list(paths) + list(extra_files):
        try:
            src = open(path).read()
        except OSError:
            continue
        for mt in re.finditer(r"(?:\w+::)?(\w+)\s*\([^;{}]*\)\s*(?:const\s*)?\{", src):
            fname = mt.group(1)
            if fname in ("if", "for", "while", "switch", "catch", "return"):
                continue
            depth, i = 0, mt.end() - 1
            while i < len(src):
                depth += {"{": 1, "}": -1}.get(src[i], 0)
                if depth == 0:
                    break
                i += 1
            bodies[fname] = bodies.get(fname, "") + src[mt.end():i]
    calls = {f: set(re.findall(r"\b(gl[A-Z]\w*)\s*[(<]", b)) for f, b in bodies.items()}
    changed = True
    while changed:
        changed = False
        for f, b in bodies.items():
            for g in bodies:
                if g != f and re.search(r"\b" + g + r"\s*\(", b) and not calls[g] <= calls[f]:
                    calls[f] |= calls[g]
                    changed = True
    return {f: c for f, c in calls.items() if c}


def git_revision(repo):
    import subprocess
    try:
        rev = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return rev + (" (with uncommitted changes)" if dirty else "")
    except OSError:
        return "unknown"


def metagl_wrappers(metagl_dir):
    path = os.path.join(metagl_dir, "include", "metagl", "Functions.hpp")
    try:
        src = open(path).read()
    except OSError:
        return None
    return set(re.findall(r"METAGL_API\s+[^;(]*?\b(gl\w+)\s*\(", src))


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--results", action="append", default=[], help="test-results directory (repeatable)")
    ap.add_argument("--metagl", default=None)
    ap.add_argument("--ctest-dir", action="append", default=[],
                    help="build directory whose CTest registration lists the required runs (repeatable; default: "
                         "the parent of each results directory)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    result_dirs = args.results or sorted(glob.glob(os.path.join(repo, "build", "*", "test-results")))
    bin_dirs = [os.path.join(os.path.dirname(d), "bin") for d in result_dirs]
    metagl_dir = args.metagl or os.path.join(repo, "..", "..", "libcna", "meta-gl")

    commands = list(csv.DictReader(open(os.path.join(repo, "inventory", "es_core_commands.csv"))))
    features = list(csv.DictReader(open(os.path.join(repo, "inventory", "features.csv"))))
    glsl = list(csv.DictReader(open(os.path.join(repo, "inventory", "glsl_features.csv"))))
    manifests = load_manifests(repo)
    results = load_results(result_dirs)
    wrappers = metagl_wrappers(metagl_dir)
    problems_early = []

    stale_report = []
    for m in manifests:
        fresh, stale = fresh_runs(m, results.get(m["name"], []), repo)
        results[m["name"]] = fresh
        if stale:
            m["_stale"] = len(stale)
            stale_report.append(f"{m['name']}: {len(stale)} stale result file(s) ignored")

    # Completeness: every registered run must have a fresh result; results of unregistered tests do not count.
    ctest_dirs = args.ctest_dir or [os.path.dirname(d) for d in result_dirs]
    registered = registered_tests(ctest_dirs)
    missing_report, unregistered_report = [], []
    n_registered = 0
    if registered is not None:
        for m in manifests:
            reg = registered.get(m["name"], set())
            n_registered += len(reg)
            kept = [r for r in results[m["name"]] if r.get("test_id", "") in reg]
            dropped = len(results[m["name"]]) - len(kept)
            if dropped:
                unregistered_report.append(f"{m['name']}: {dropped} result file(s) of unregistered tests ignored")
            results[m["name"]] = kept
            missing = sorted(reg - {r.get("test_id", "") for r in kept})
            if missing:
                m["_missing"] = missing
                missing_report.append(f"{m['name']}: {', '.join(missing)}")

    # Manifest honesty lint: every claimed API entry point must be called by the executable's own sources,
    # directly or through a common/ helper (demo::get_integer -> glGetIntegerv, Check::expect_pixel -> glReadPixels).
    helpers = common_helper_calls(repo, glob.glob(os.path.join(metagl_dir, "include", "metagl", "*.hpp")))
    for m in manifests:
        srcs = ""
        for d, _, files in os.walk(os.path.join(repo, m["_dir"])):
            for fn in files:
                if fn.endswith((".cpp", ".hpp", ".h")):
                    srcs += open(os.path.join(d, fn), errors="replace").read()
        called = set(re.findall(r"\b(gl[A-Z]\w*)\b", srcs))
        for h, gls in helpers.items():
            if h in called or re.search(r"\b" + re.escape(h) + r"\s*[(<]", srcs):
                called |= gls
        for fn in m.get("api", {}):
            if fn not in called:
                problems_early.append(f"{m['name']}: api '{fn}' is claimed but never called (directly or via common/)")

    status_of = {}
    for m in manifests:
        status_of[m["name"]], _ = demo_status(m, results, bin_dirs)

    fn_demos = defaultdict(list)
    feat_demos = defaultdict(list)
    glsl_demos = defaultdict(list)
    for m in manifests:
        for fn, how in m.get("api", {}).items():
            fn_demos[fn].append((m["name"], how))
        for fid in m.get("features", []):
            feat_demos[fid].append(m["name"])
        for gid in m.get("glsl_features", []):
            glsl_demos[gid].append(m["name"])

    known_fns = {c["name"] for c in commands}
    known_feats = {f["id"] for f in features}
    known_glsl = {g["id"] for g in glsl}
    problems = list(problems_early)
    for m in manifests:
        for fn in m.get("api", {}):
            if fn not in known_fns:
                problems.append(f"{m['name']}: api '{fn}' is not an ES core command")
        for fid in m.get("features", []):
            if fid not in known_feats:
                problems.append(f"{m['name']}: unknown feature id '{fid}'")
        for gid in m.get("glsl_features", []):
            if gid not in known_glsl:
                problems.append(f"{m['name']}: unknown glsl feature id '{gid}'")

    blockers = {}
    for m in manifests:
        for fn, why in m.get("blocked_by_metagl", {}).items():
            blockers[fn] = why

    gen = os.path.join(repo, "docs", "generated")
    os.makedirs(gen, exist_ok=True)

    # ---- API_COVERAGE.csv -------------------------------------------------------------------
    api_rows = []
    with open(os.path.join(repo, "docs", "API_COVERAGE.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["function", "introduced", "registry_feature", "in_metagl", "demos", "verification",
                    "runtime_status", "limitations", "evidence"])
        for c in commands:
            name = c["name"]
            demos = fn_demos.get(name, [])
            st = best(status_of[d] for d, _ in demos)
            if name in blockers and st in ("NOT_STARTED", "IMPLEMENTED_UNVERIFIED"):
                st = "BLOCKED_BY_META_GL"
            in_mgl = "unknown" if wrappers is None else ("yes" if name in wrappers else "NO")
            verification = " / ".join(how for _, how in demos)
            evidence = " ".join(f"{d}:{status_of[d]}" for d, _ in demos)
            api_rows.append((c["introduced"], st))
            w.writerow([name, c["introduced"], c["feature"], in_mgl, " ".join(d for d, _ in demos), verification,
                        st, blockers.get(name, ""), evidence])

    # ---- summaries --------------------------------------------------------------------------
    def summary(rows, key_name):
        by = defaultdict(Counter)
        for k, st in rows:
            by[k][st] += 1
        cols = [s for s in RANK if any(by[k][s] for k in by)]
        table = []
        for k in sorted(by):
            total = sum(by[k].values())
            done = sum(by[k][s] for s in ("RUNTIME_PASSED", "VISUALLY_VERIFIED", "FULLY_VALIDATED"))
            table.append([k, total, done, f"{100.0 * done / total:.0f}%"] + [by[k][s] for s in cols])
        return md_table(table, [key_name, "total", "passed", "passed %"] + cols)

    feat_rows, feat_table = [], []
    for ft in features:
        st = best(status_of[d] for d in feat_demos.get(ft["id"], []))
        if st == "NOT_STARTED" and ft.get("verification") == "doc":
            st = "DOCUMENTED_ONLY"
        feat_rows.append((ft["introduced"], st))
        feat_table.append([ft["id"], ft["introduced"], ft["category"], ft["title"], ft["spec_ref"], st,
                           ", ".join(feat_demos.get(ft["id"], []))])
    glsl_rows, glsl_table = [], []
    for g in glsl:
        st = best(status_of[d] for d in glsl_demos.get(g["id"], []))
        if st == "NOT_STARTED" and g.get("strategy") == "doc":
            st = "DOCUMENTED_ONLY"
        glsl_rows.append((g["glsl"], st))
        glsl_table.append([g["id"], g["glsl"], g["category"], g["title"], g["spec_ref"], g["strategy"], st,
                           ", ".join(glsl_demos.get(g["id"], []))])

    header = "<!-- GENERATED by tools/coverage/generate_coverage.py - do not edit -->\n"
    with open(os.path.join(gen, "COVERAGE_SUMMARY.md"), "w") as f:
        f.write(header + "# Coverage summary\n\n")
        f.write(f"Result directories: {', '.join(os.path.relpath(d, repo) for d in result_dirs) or '(none)'}\n\n")
        f.write(f"Source revision: {git_revision(repo)}\n\n")
        f.write(f"meta-gl revision: {git_revision(metagl_dir)} ({os.path.relpath(os.path.abspath(metagl_dir), repo)})\n\n")
        if registered is None:
            f.write("Registered runs: not checked (no CTest build directory found).\n\n")
        else:
            f.write(f"Registered runs: {n_registered}; with a fresh result: {n_registered - sum(len(m.get('_missing', [])) for m in manifests)}"
                    f"; missing: {sum(len(m.get('_missing', [])) for m in manifests)}.\n\n")
        f.write("Only results from the current executables count. A result is ignored when its executable was\n"
                "rebuilt after the run, or when the demo's sources, `common/` or the CMake glue are newer than the\n"
                "executable. `es-min` runs use `MESA_GLES_VERSION_OVERRIDE`: they prove API-version detection on an\n"
                "ES 3.2 driver, not conformance of a native ES 3.0/3.1 implementation (see VALIDATION.md).\n\n")
        f.write("## Core API entry points (by introducing ES version)\n\n" + summary(api_rows, "ES") + "\n\n")
        f.write("## Semantic features (by introducing ES version)\n\n" + summary(feat_rows, "ES") + "\n\n")
        f.write("## GLSL ES features (by language version)\n\n" + summary(glsl_rows, "GLSL") + "\n\n")
        f.write("## Executables\n\n" + md_table(sorted(Counter(status_of.values()).items()), ["status", "count"]) + "\n")
        if stale_report:
            f.write("\n## Stale results (ignored)\n\n" + "\n".join(f"- {x}" for x in stale_report) + "\n")
        if missing_report:
            f.write("\n## Registered runs without a fresh result\n\n" + "\n".join(f"- {x}" for x in missing_report) + "\n")
        if unregistered_report:
            f.write("\n## Results of unregistered tests (ignored)\n\n" + "\n".join(f"- {x}" for x in unregistered_report) + "\n")
        if problems:
            f.write("\n## Manifest problems\n\n" + "\n".join(f"- {p}" for p in problems) + "\n")
    # Evidence for every known-deviation check: where it actually failed (XFAIL) or passed although matched (XPASS).
    dev = defaultdict(lambda: {"match": "", "XFAIL": set(), "XPASS": set()})
    for name, runs in results.items():
        for r in runs:
            parts = r.get("test_id", "").split(":")
            if len(parts) != 3:
                continue
            run = f"{parts[1]}:{parts[2]}"
            for c in r.get("demo_checks", []):
                mt = re.search(r"(XFAIL|XPASS)[^']* on '([^']+)'", c.get("detail", ""))
                if mt:
                    e = dev[(name, c["what"])]
                    e["match"] = mt.group(2)
                    e[mt.group(1)].add(run)
    dev_rows = []
    for (name, what), e in sorted(dev.items()):
        drivers_xfail = {x.split(":")[0] for x in e["XFAIL"]}
        note = ""
        if e["match"] == "Mesa" and not {"gpu", "llvmpipe", "zink"} <= drivers_xfail:
            note = "matched 'Mesa' but not observed on every Mesa driver"
        if e["XFAIL"] and all(x.endswith(":es-min") for x in e["XFAIL"]):
            note = "es-min only: version-override artifact, not a driver deviation"
        dev_rows.append([name, what.replace("|", "/")[:140], e["match"], " ".join(sorted(e["XFAIL"])),
                         " ".join(sorted(e["XPASS"])), note])
    with open(os.path.join(gen, "DEVIATIONS.md"), "w") as f:
        f.write(header + "# Known-deviation evidence\n\n"
                "Every check reported through `expect_or_known_deviation`, with the runs where it failed (XFAIL)\n"
                "and the runs where the matched driver passed it (XPASS). Spec quotes and explanations:\n"
                "[KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md).\n\n"
                + md_table(dev_rows, ["executable", "check", "match", "XFAIL runs", "XPASS runs", "note"]) + "\n")
    with open(os.path.join(gen, "FEATURE_STATUS.md"), "w") as f:
        f.write(header + "# Feature status\n\n" + md_table(feat_table, ["id", "ES", "category", "title", "spec", "status", "demos"]) + "\n")
    with open(os.path.join(gen, "GLSL_STATUS.md"), "w") as f:
        f.write(header + "# GLSL ES feature status\n\n" + md_table(glsl_table, ["id", "GLSL", "category", "title", "spec", "strategy", "status", "demos"]) + "\n")
    demo_table = []
    for m in sorted(manifests, key=lambda m: (m.get("category", ""), m.get("order", 0), m["name"])):
        runs = results.get(m["name"], [])
        runs_s = " ".join(f"{r.get('test_id', '?').split(':', 1)[-1]}={r['outcome']}" for r in sorted(runs, key=lambda r: r.get("test_id", "")))
        demo_table.append([m["name"], m.get("min_es", ""), m.get("category", ""), m.get("title", ""), status_of[m["name"]], runs_s])
    with open(os.path.join(gen, "DEMO_STATUS.md"), "w") as f:
        f.write(header + "# Executable status\n\n" + md_table(demo_table, ["executable", "min ES", "category", "title", "status", "runs"]) + "\n")

    for p in problems:
        print("manifest problem:", p, file=sys.stderr)
    n_done = sum(1 for _, st in api_rows if st in ("RUNTIME_PASSED", "VISUALLY_VERIFIED", "FULLY_VALIDATED"))
    print(f"API: {n_done}/{len(api_rows)} entry points with passing evidence; {len(manifests)} manifests; "
          f"{sum(len(v) for v in results.values())} result files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
