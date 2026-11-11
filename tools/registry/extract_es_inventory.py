#!/usr/bin/env python3
"""Derive the OpenGL ES 2.0-3.2 core inventory from the Khronos registry (gl.xml).

Outputs (committed, regenerated on demand):
  inventory/es_core_commands.csv   name,introduced,feature,params
  inventory/es_core_enums.csv      name,value,introduced,groups
  inventory/registry_source.txt    provenance (URL, sha256, fetch date)

Only <feature api="gles2"> blocks GL_ES_VERSION_2_0..3_2 are used, so extension-only
entry points are never counted as core. A name is attributed to the first version that
requires it. <remove> blocks are honoured (none exist for ES 2.0-3.2 at the time of writing).

Usage:
  tools/registry/extract_es_inventory.py --gl-xml path/to/gl.xml [--repo .]
  tools/registry/extract_es_inventory.py --fetch [--repo .]       # downloads gl.xml to a temp dir
"""
import argparse
import csv
import datetime
import hashlib
import os
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

GL_XML_URL = "https://registry.khronos.org/OpenGL/xml/gl.xml"
FEATURES = ["GL_ES_VERSION_2_0", "GL_ES_VERSION_3_0", "GL_ES_VERSION_3_1", "GL_ES_VERSION_3_2"]
VERSION = {"GL_ES_VERSION_2_0": "2.0", "GL_ES_VERSION_3_0": "3.0",
           "GL_ES_VERSION_3_1": "3.1", "GL_ES_VERSION_3_2": "3.2"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gl-xml")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    args = ap.parse_args()

    if args.fetch:
        tmp = tempfile.mkdtemp(prefix="glxml-")
        path = os.path.join(tmp, "gl.xml")
        urllib.request.urlretrieve(GL_XML_URL, path)
    elif args.gl_xml:
        path = args.gl_xml
    else:
        ap.error("give --gl-xml or --fetch")

    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    root = ET.fromstring(data)

    # Command prototypes (for the params column).
    protos = {}
    for cmd in root.find("commands").findall("command"):
        name = cmd.find("proto").find("name").text
        params = []
        for p in cmd.findall("param"):
            params.append("".join(p.itertext()).strip())
        protos[name] = "; ".join(params)

    enum_values, enum_groups = {}, {}
    for enums in root.findall("enums"):
        for e in enums.findall("enum"):
            if e.get("api") not in (None, "gles2"):
                continue
            enum_values.setdefault(e.get("name"), e.get("value"))
            enum_groups.setdefault(e.get("name"), e.get("group") or "")

    cmd_ver, enum_ver = {}, {}
    for feat in root.findall("feature"):
        if feat.get("api") != "gles2" or feat.get("name") not in FEATURES:
            continue
        fname = feat.get("name")
        for req in feat.findall("require"):
            for c in req.findall("command"):
                cmd_ver.setdefault(c.get("name"), fname)
            for e in req.findall("enum"):
                enum_ver.setdefault(e.get("name"), fname)
        for rem in feat.findall("remove"):
            for c in rem.findall("command"):
                cmd_ver.pop(c.get("name"), None)
            for e in rem.findall("enum"):
                enum_ver.pop(e.get("name"), None)

    out = os.path.join(args.repo, "inventory")
    os.makedirs(out, exist_ok=True)
    order = {f: i for i, f in enumerate(FEATURES)}
    with open(os.path.join(out, "es_core_commands.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["name", "introduced", "feature", "params"])
        for name, fe in sorted(cmd_ver.items(), key=lambda kv: (order[kv[1]], kv[0])):
            w.writerow([name, VERSION[fe], fe, protos.get(name, "")])
    with open(os.path.join(out, "es_core_enums.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["name", "value", "introduced", "groups"])
        for name, fe in sorted(enum_ver.items(), key=lambda kv: (order[kv[1]], kv[0])):
            w.writerow([name, enum_values.get(name, ""), VERSION[fe], enum_groups.get(name, "")])
    with open(os.path.join(out, "registry_source.txt"), "w") as f:
        f.write(f"source: {GL_XML_URL}\nsha256: {sha}\nextracted: {datetime.date.today().isoformat()}\n")

    counts = {VERSION[fe]: sum(1 for v in cmd_ver.values() if v == fe) for fe in FEATURES}
    print(f"commands: {len(cmd_ver)} {counts}; enums: {len(enum_ver)}; gl.xml sha256 {sha[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
