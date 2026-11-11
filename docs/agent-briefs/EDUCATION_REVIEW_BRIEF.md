# Brief: educational review (priority 6)

opengles3-demos must stay an excellent resource for developers **learning** OpenGL ES, not only a test suite. You
review a group of demos as a learner would meet them, after reading `AGENTS.md` and `docs/DEMO_CATALOG.md`.
You do not change code; the report is the deliverable. Build nothing, unless you need to run a demo to understand it
(then `-j 1`; machine-wide CPU limit).

For every demo in your group, assess:
1. **Purpose and entry.** Can a learner tell from the README's first lines what the demo teaches and what to look at?
2. **README accuracy.** Do the README claims match main.cpp and demo.json? Check API names, check counts if stated,
   spec section numbers (ES 2.0/3.0 features cite ES 3.0 numbering, 3.1/3.2 features cite ES 3.2 numbering), controls
   and expected image. List every stale or wrong statement with file:line.
3. **Readability of the GL path.** Are the meta-gl calls that teach the topic easy to find, or buried in validation
   code? If validation dominates, suggest concrete structure, e.g. a clearly marked "the technique" section, moving
   CPU-model code below, or a short "read this first" pointer. Do not suggest an engine or a shared abstraction.
4. **Comments.** Do they explain *why* (spec rules, pitfalls) rather than restate the code? Flag misleading ones.
5. **Shaders.** Correct `#version`, readable, commented where the technique happens.
6. **Learning path.** Does the README point to the prerequisite and the next demo where useful?

Report per demo: up to 5 concrete, actionable items, ranked: WRONG (factually incorrect docs), CONFUSING (likely to
mislead a learner), POLISH. A demo with nothing to fix gets "good" plus one sentence on why. Be concise and specific.
