# Feature coverage

Live status: **[generated/FEATURE_STATUS.md](generated/FEATURE_STATUS.md)** and
**[generated/COVERAGE_SUMMARY.md](generated/COVERAGE_SUMMARY.md)** (regenerate with
`cmake --build --preset debug --target coverage`). Entry-point level: **[API_COVERAGE.csv](API_COVERAGE.csv)**.

## Sources of the inventory

| Inventory | Source | Count |
|---|---|---|
| `inventory/es_core_commands.csv` | Khronos gl.xml, `<feature api="gles2">` 2.0–3.2 (`tools/registry/extract_es_inventory.py`) | 358 (142/104/68/44) |
| `inventory/es_core_enums.csv` | same | 1001 |
| `inventory/features.csv` | curated from the ES 3.0 / 3.2 specification chapters and the project brief | see summary |

The command and enum lists are mechanical. The **feature** list is semantic and curated by hand. It
cannot be derived from the registry, because one entry point usually has several behaviors that
need separate evidence (e.g. `glMapBufferRange` read access, write access, each flag, error cases).
Section references use ES 3.0 numbering for 2.0/3.0 features and ES 3.2 numbering for 3.1/3.2 features.

## Verification kinds (column `verification`)

| kind | meaning |
|---|---|
| `pixel` | rendered output compared with a CPU reference |
| `readback` | buffer / texture / framebuffer contents read back and compared |
| `state` | `glGet*`/`glIs*` queries compared with expected values |
| `negative` | a defined error (`glGetError`, status, info log) is provoked and observed |
| `doc` | behavior is undefined or implementation-dependent; explained, not asserted |

## Rules

- Every row must eventually map to at least one executable or carry an explicit blocker.
- A demo lists a feature only when `validate()` checks that feature's observable effect.
- New features found during audits are appended (do not renumber IDs).
- Periodic audit (Phase 6): re-read each spec chapter, compare against this list, correct
  claims, and add missing rows.
