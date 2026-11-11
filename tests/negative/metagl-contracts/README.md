# test-metagl-contracts

Tests meta-gl behaviors that ordinary GL error checks cannot observe.

- **Invalid-input contract.** meta-gl terminates the process (`std::terminate`, also in Release
  builds) when an argument would otherwise produce undefined behavior or an ES-illegal call.
  Examples are a uniform array whose length is not a whole number of vectors or matrices, and calling any wrapper after
  the context was marked lost. Each case runs in a `fork()`ed child, and the parent requires the
  child to die with `SIGABRT`. A control child with no violation must exit normally. Children never
  call into the GL driver: every case is rejected inside meta-gl first.
- **Row-major uploads.** `glUniformMatrix<4>(..., GL_TRUE)` is legal in ES 3.0+ (§2.12.6.1) and must read back
  transposed. meta-gl used to terminate on it (gap G8, fixed).
- **Context-loss life cycle.** `NotifyContextLost` must notify listeners once, mark the status as
  `Lost`, hide all entry points and capabilities, and keep the generation counter.
  `RestoreCurrentContext(loader)` must reload, notify listeners once (with status `Restored` during
  the callback), return to `Current`, increment the generation, and re-detect capabilities.
  Rendering must then work again, checked with a pixel.

Run: `test-metagl-contracts --headless --validate`. Linux-only (`fork`/`waitpid`).
