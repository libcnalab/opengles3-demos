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
  IMPLEMENTED_UNVERIFIED claimed by a manifest, no executable built
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
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

RANK = ["NOT_STARTED", "IMPLEMENTED_UNVERIFIED", "FAILED", "BLOCKED_BY_PLATFORM", "BLOCKED_BY_META_GL",
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


def demo_status(m, results, bin_dirs):
    runs = results.get(m["name"], [])
    built = any(os.path.exists(os.path.join(b, m["name"])) for b in bin_dirs)
    if not runs:
        return ("BUILD_PASSED" if built else "IMPLEMENTED_UNVERIFIED"), runs
    outcomes = [r["outcome"] for r in runs]
    if "FAIL" in outcomes:
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
    full = ok_runs and bool(methods) and need <= pass_ids
    if full:
        return "FULLY_VALIDATED", runs
    if m.get("visual_review"):
        return "VISUALLY_VERIFIED", runs
    return "RUNTIME_PASSED", runs


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
    problems = []
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
        feat_rows.append((ft["introduced"], st))
        feat_table.append([ft["id"], ft["introduced"], ft["category"], ft["title"], ft["spec_ref"], st,
                           ", ".join(feat_demos.get(ft["id"], []))])
    glsl_rows, glsl_table = [], []
    for g in glsl:
        st = best(status_of[d] for d in glsl_demos.get(g["id"], []))
        glsl_rows.append((g["glsl"], st))
        glsl_table.append([g["id"], g["glsl"], g["category"], g["title"], g["spec_ref"], g["strategy"], st,
                           ", ".join(glsl_demos.get(g["id"], []))])

    header = "<!-- GENERATED by tools/coverage/generate_coverage.py - do not edit -->\n"
    with open(os.path.join(gen, "COVERAGE_SUMMARY.md"), "w") as f:
        f.write(header + "# Coverage summary\n\n")
        f.write(f"Result directories: {', '.join(os.path.relpath(d, repo) for d in result_dirs) or '(none)'}\n\n")
        f.write("## Core API entry points (by introducing ES version)\n\n" + summary(api_rows, "ES") + "\n\n")
        f.write("## Semantic features (by introducing ES version)\n\n" + summary(feat_rows, "ES") + "\n\n")
        f.write("## GLSL ES features (by language version)\n\n" + summary(glsl_rows, "GLSL") + "\n\n")
        f.write("## Executables\n\n" + md_table(sorted(Counter(status_of.values()).items()), ["status", "count"]) + "\n")
        if problems:
            f.write("\n## Manifest problems\n\n" + "\n".join(f"- {p}" for p in problems) + "\n")
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
