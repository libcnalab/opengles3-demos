Backlog item 20: `tests/negative/driver-crash-readpixels-none/`, executable
`test-driver-crash-readpixels-none`, min_es 3.0. Look at `tests/negative/metagl-contracts/` for how a
test forks a child process and inspects how it ended, and at docs/KNOWN_LIMITATIONS.md (row "crash (not
testable in-process)") for the known Mesa case: glReadPixels with READ_BUFFER = NONE on an FBO with an
integer colour attachment segfaults on radeonsi/llvmpipe/zink instead of raising INVALID_OPERATION
(ES 3.0 §4.3.2.1 — verify the exact rule text).
- Run the offending call in a forked child that creates its own state (the GL context must be usable in
  the child: verify this works with the EGL context created by the runtime, otherwise create a fresh EGL
  context in the child via the runtime's platform — do NOT modify common/; if impossible, report it).
- PASS on drivers where the child observes INVALID_OPERATION and exits normally; on Mesa the crash is
  reported through expect_or_known_deviation(..., "Mesa", ...) with the signal printed.
- Also test the non-crashing related rules in the parent: READ_BUFFER NONE + normalized attachment →
  INVALID_OPERATION; reading from an FBO whose read buffer is missing.
