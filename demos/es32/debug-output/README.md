# es32-debug-output

**Minimum:** OpenGL ES 3.2 · **Shaders:** GLSL ES 3.20 (`#version 320 es`) · **Context:** a *debug context*
(`demo::Info::debug_context`, which the runtime turns into `EGL_CONTEXT_OPENGL_DEBUG`) ·
**Run:** `es32-debug-output [--headless --validate]`

## Purpose

Chapter 18 of the ES 3.2 specification gives applications a channel to the GL: a stream of *debug messages* (source,
type, id, severity, text) that the GL emits while it executes commands, that the application can add its own messages to,
and that can be received either through a callback or by polling a message log. On top of that there are *debug groups*
(a stack that brackets a section of the command stream and carries its own filter settings) and *labels* (a text on any
object, so that a debugger or a message can say "the shadow-map texture" instead of "texture 7"). This is the API that
tools such as RenderDoc, apitrace or a driver's validation layer use, and the one an application should use to keep its own
render passes legible in them.

The demo uses every part of it and checks what the specification says about each: what is delivered, what is filtered,
what is logged, what is rejected, and what the state queries report. A small scene is drawn inside a debug group so that
the annotations also sit in a real command stream.

## What the demo does

`init()` records the debug state of the freshly created context before anything touches it, observes which messages are
enabled by default (first through the message log, then through a callback), builds the scene (a program, a vertex array,
a vertex buffer, a 2x2 texture and a sampler, each given a label with `glObjectLabel`) and leaves a callback installed.
`render()` draws the scene inside `glPushDebugGroup("Scene")`, with two `glDebugMessageInsert` markers and a
`glPopDebugGroup()`. `validate()` then runs 30 *areas*, each starting from a known baseline (output on, synchronous,
callback set, groups popped, only APPLICATION and THIRD_PARTY messages enabled so that anything the driver itself says
cannot disturb the expected sequences):

| Area | What is checked |
|---|---|
| context and initial state | `CONTEXT_FLAGS` has `CONTEXT_FLAG_DEBUG_BIT`; `DEBUG_OUTPUT` is initially TRUE, `DEBUG_OUTPUT_SYNCHRONOUS` FALSE, callback and user pointer NULL, group depth 1, log empty; the four implementation limits against the Table 21.54 minimums |
| initial message state | with no `glDebugMessageControl` call yet, HIGH, MEDIUM and NOTIFICATION messages are delivered and LOW is not (§18.1), in the log and through the callback |
| callback + GetPointerv | `glGetPointerv` returns the registered function and user pointer; the callback receives that user pointer; a second `glDebugMessageCallback` replaces both; `NULL` clears it and messages go to the log |
| DebugMessageInsert | exact source, type, id, severity, length and text of the callback arguments; explicit lengths; ids 0 and 0xFFFFFFFF; all 72 combinations of {APPLICATION, THIRD_PARTY} x 9 types x 4 severities in order; nothing in the log while a callback is set |
| Insert errors and limits | `INVALID_ENUM` for every other source, for type or severity `DONT_CARE`; `INVALID_VALUE` at `MAX_DEBUG_MESSAGE_LENGTH` characters, accepted at one less |
| DEBUG_OUTPUT / SYNC | disabling discards inserts without error and stops callback and log; `IsEnabled` follows both toggles; with synchronous output the callback has run, on the calling thread, when the command returns |
| Control by source/type/sev | nine successive `glDebugMessageControl` calls (with `DONT_CARE` wildcards, both directions, 32 messages each) compared with a CPU model of the per-message enabled flags |
| Control by id lists | ids are per (source, type) namespace and independent of severity; unknown ids are ignored; a later severity-wide control overrides earlier id settings; with `count == 0` the ids array is ignored |
| Control errors | `INVALID_OPERATION` (count > 0 with `DONT_CARE` source or type, or a severity), `INVALID_VALUE` (negative count); the failed calls change nothing |
| message log | `DEBUG_LOGGED_MESSAGES`, `DEBUG_NEXT_LOGGED_MESSAGE_LENGTH`, oldest-first fetching, count and bufSize limits, lengths including the terminator, concatenated strings, `NULL` arrays, each array optional on its own |
| log overflow | inserting `MAX_DEBUG_LOGGED_MESSAGES + 5` messages keeps exactly the oldest `MAX` ones; the log accepts messages again once emptied |
| debug group messages | PUSH_GROUP / POP_GROUP messages with the right source, id, severity NOTIFICATION; nesting; `DEBUG_GROUP_STACK_DEPTH`; errors of `glPushDebugGroup` |
| debug group filters | a new group inherits its parent's filter state; changes inside a group are undone by the pop; three levels deep |
| group depth limits | `STACK_UNDERFLOW` on the default group; `MAX - 1` pushes succeed, the next one gives `STACK_OVERFLOW`; the pops return the pushed ids in reverse order |
| GL error messages | four deliberately provoked errors: the code is set; any ERROR message that arrives has source API; with `DEBUG_OUTPUT` off no message is generated |
| scene stream + pixels | the frame's annotations arrived in order under the default filter state; triangle, four texel quadrants and background match the CPU expectation |
| scene object labels | the five labels set in `init()` are still there after the objects were used |
| labels: buffer, shader, program, vertex array, query, program pipeline, sampler, texture, renderbuffer, framebuffer, transform feedback | the label battery below, once per namespace of Table 18.4 |
| labels: sync object | the same battery with `glObjectPtrLabel` / `glGetObjectPtrLabel` |
| negative controls | see Validation |

The **label battery** (per namespace): a new object has the empty label; set and read back a null-terminated label;
explicit length truncates; a second object keeps its own label and labels need not be unique; `bufSize` truncation
(`bufSize` 4, 6, 7 and 1 on a 6-character label: at most `bufSize - 1` characters plus the terminator, nothing written
beyond the buffer); `label = NULL` in `glGetObjectLabel` returns only the length, `length = NULL` is accepted; `label = NULL`
in `glObjectLabel` removes the label; `MAX_LABEL_LENGTH` (the label must be *shorter*: `MAX` characters give
`INVALID_VALUE`, `MAX - 1` are accepted and read back whole); a negative `bufSize` and the name of a never generated or of
a deleted object give `INVALID_VALUE` and write nothing through the output pointers.

## Reading the image

The 256 x 256 capture has three parts (rows are numbered from the bottom):

* **Rows 144-255: the scene.** An orange flat-coloured triangle on the left and a quad on the right textured with a
  2x2 checker (lower row red, green; upper row blue, yellow). The quad's four texel colours are probed at the quadrant
  centres. The scene is drawn inside the debug group "Scene"; the group and the two markers change no pixel.
* **Rows 128-143: five squares**, one per scene object (program, vertex array, vertex buffer, texture, sampler), green when
  `glGetObjectLabel` still returns the label from `init()`.
* **Rows 0-127: 32 cells of 32 x 32 pixels**, one per area in the order of the table above (area 1 at the lower left, eight
  per row). Green = every check of the area passed, red = at least one failed, dark = unused (cells 31 and 32).
  The expected capture is 30 green cells; a red cell means the area named in the log failed. The panel is drawn at the end
  of `validate()` (after the pixel probes), so a run without `--validate` shows the scene on a dark background.

## Central calls

| Call | Where |
|---|---|
| `glDebugMessageCallback`, `glGetPointerv` | `init()`, `reset_state()`, `area_pointers()` |
| `glDebugMessageInsert` | `render()`, `area_insert()`, `area_insert_errors()`, most other areas |
| `glDebugMessageControl` | `reset_state()`, `area_control_filters()`, `area_control_ids()`, `area_control_errors()`, `area_group_filters()` |
| `glGetDebugMessageLog` | `fetch_log()`, `drain_log()`, `area_log()`, `area_log_overflow()` |
| `glPushDebugGroup`, `glPopDebugGroup` | `render()`, `area_group_messages()`, `area_group_filters()`, `area_group_limits()` |
| `glObjectLabel`, `glGetObjectLabel`, `glObjectPtrLabel`, `glGetObjectPtrLabel` | `init()`, `label_battery()`, `area_labels()`, `area_sync_labels()` |
| `glEnable` / `glDisable` / `glIsEnabled` with `DEBUG_OUTPUT` and `DEBUG_OUTPUT_SYNCHRONOUS` | `area_output_enable()` |

## Spec sections

OpenGL ES 3.2: §18 Debug Output, §18.1 Debug Messages (Tables 18.1, 18.2, 18.3), §18.2 Debug Message Callback, §18.3 Debug
Message Log, §18.4 Controlling Debug Messages, §18.5 Externally Generated Messages, §18.6 Debug Groups, §18.7 Debug Labels
(Table 18.4), §18.8 Asynchronous and Synchronous Debug Output, §18.9 Debug Output Queries, §20.2 Pointer, String, and Related
Context Queries (GetPointerv, `CONTEXT_FLAGS`), §2.3.1 Errors (Table 2.3, and "the GL may also generate a debug output
message"), Tables 21.53 and 21.54. GLSL ES 3.10 §4.4.3 and §4.4.4 for the `layout(location)` uniforms and `layout(binding)`
sampler of the scene shaders.

## Shader concepts

The scene shaders are deliberately plain; the subject of the demo is the API. They are `#version 320 es` and use two
features that the pixel checks verify: default-block uniforms placed with `layout(location = N)` (`u_color`, `u_mode`) and a
sampler placed with `layout(binding = 0)`, so the C++ side never calls `glGetUniformLocation` nor sets the sampler uniform.

## Validation

Every claim is checked against the specification text, not against a particular driver:

* **Delivery** is observed through a callback that records every argument (source, type, id, severity, length, text, user
  pointer, calling thread) in a locked vector. `DEBUG_OUTPUT_SYNCHRONOUS` is enabled, which §18.8 defines as "each call will be
  made before the GL command that generated the debug message is allowed to return", so the vector is complete when an
  insert returns. The callback itself makes no GL call (§18.2: undefined).
* **Filtering** is compared with a CPU model: one enabled flag per (source, type, severity) message, updated by each
  `glDebugMessageControl` according to §18.4 (`DONT_CARE` = every value). After each call all 32 messages are inserted and the
  delivered ids must equal the model's enabled ids, in order. The scenario goes from "nothing delivered" to "everything
  delivered" and back through partial states, so a filter that did nothing, or too much, fails.
* **Messages of the implementation are kept out** of the expected sequences by the baseline (only APPLICATION and
  THIRD_PARTY enabled). The scene stream check filters on source for the same reason and prints how many other messages
  arrived.
* **Errors** are compared with the defined code through `check.expect_gl_error`, and, per §2.3.1 / Table 2.3 ("the command
  which generates the error is ignored ... if the generating command modifies values through a pointer argument, no change is
  made to these values"), the failed commands are checked to leave messages, depth, enabled flags, labels and output buffers
  untouched.
* **Tolerances:** the scene pixels are compared with ±1 per channel (8-bit quantisation of float colours that are exact in
  8 bits except the 0.10/0.12 clear colour, 25.5 and 30.6); the probes are quadrant centres of the texels and points well
  inside the triangle.
* **Negative controls (the checks can fail):** (1) with the callback removed, the same insert that was just delivered is not
  seen by the sink, so every delivery check would fail, and the message appears in the log instead; (2) one
  `glDebugMessageControl(HIGH, FALSE)` flips the very same insert from delivered to blocked; (3) relabelling the vertex buffer
  makes the comparison with the original label fail; (4) the triangle pixel is shown to differ from a wrong expectation;
  (5) in the id-list area the expectation "the Control call had no effect" is shown to be contradicted. In addition, two
  mutations of the demo were run once during development and failed as they should: a callback that reports `id + 1`
  fails more than 30 checks in at least eight areas, and one push too many in the depth test fails the "MAX - 1 pushes raise
  no error" check.

## Pitfalls and findings

* **Errors are not required to produce messages.** §2.3.1: "the GL *may* also generate a debug output message". The demo
  therefore does *not* require one. It provokes four errors, requires the error code, and checks every ERROR-typed message that
  does arrive (source API, non-empty well-formed string, shorter than `MAX_DEBUG_MESSAGE_LENGTH`); Mesa generates one each
  time and the log prints them. What *is* required is the opposite case: with `DEBUG_OUTPUT` disabled nothing is generated.
* **Initial state.** §18.1: "All messages are initially enabled unless their assigned severity is DEBUG_SEVERITY_LOW." That
  includes NOTIFICATION, which is what PUSH_GROUP, POP_GROUP and typical markers use. Mesa follows the text; many programs
  assume NOTIFICATION is silent by default.
* **Debug group depth.** §18.6 says `STACK_OVERFLOW` is generated "if PushDebugGroup is called and the stack contains the
  value of MAX_DEBUG_GROUP_STACK_DEPTH minus one elements". `DEBUG_GROUP_STACK_DEPTH` starts at 1 (the default group) and
  `MAX_DEBUG_GROUP_STACK_DEPTH` is documented as the "max group stack depth", so the demo reads "elements" as pushed groups: the
  depth may reach exactly `MAX`, i.e. `MAX - 1` pushes succeed. This is the only reading in which the depth limit and the
  overflow rule agree, and what Mesa does.
* **Filter state of the POP message.** §18.6 says a pop "restores the debug output volume control of the parent" and also that
  the GL generates a message "after popping". Which state governs the POP_GROUP message is not unambiguous, and the demo
  avoids depending on it (it never disables NOTIFICATION inside a group). The text of the POP_GROUP message is only
  "based on" the push message and is printed, not asserted.
* **Known Mesa deviations (strict check kept, see `docs/KNOWN_LIMITATIONS.md`).**
  1. When `glDebugMessageInsert` or `glPushDebugGroup` gets an explicit `length` shorter than the string, Mesa hands the
     callback the caller's own pointer with that `length`: the text is correct but is not null-terminated at `length`.
     §18.1 ("null-terminated string representation") and §18.2 ("its length (excluding the null-terminator)") ask for a
     terminated string. Always read the message by `length` in a callback.
  2. A label of `MAX_LABEL_LENGTH` characters is rejected with `INVALID_VALUE` but still stored. Table 2.3 says an
     `INVALID_VALUE` command is ignored. (One XFAIL line per namespace, 12 in total.)
* `glGetObjectLabel` with `bufSize` 0 and a non-NULL label returns the complete length in Mesa and writes nothing; §18.9 does
  not say what `length` should be when not even the terminator fits, so the demo prints it as a note and asserts nothing.
* `INVALID_ENUM` for an object identifier outside Table 18.4, or an invalid enum in `glDebugMessageControl`, can only be
  produced with a raw cast through the typed meta-gl enums and is not tested here (the `INVALID_ENUM` checks use enumerators
  that are valid in general but not for that command: source `API` for insert and push, `DONT_CARE` as insert type or severity).
* `glDebugMessageCallback` needs a callback declared with the platform's `GL_APIENTRY` calling convention (§18.2); on Linux it
  is empty, elsewhere (Windows) it matters. The macro comes from the GL headers that meta-gl includes. The arguments arrive as
  raw `GLenum` values and are converted to the typed meta-gl enums with `static_cast` (they are always members of Tables
  18.1-18.3).
* Messages generated *inside* a callback, and any GL call inside a callback, are undefined (§18.2); the callback only stores
  its arguments. With `DEBUG_OUTPUT_SYNCHRONOUS` disabled the GL may call the callback from another thread, which is why the sink
  is locked and the demo enables synchronous output.
* **softpipe.** The ES 3.2 context cannot be created on softpipe (a native ES 3.1 driver): `eglCreateContext` fails with
  `EGL_BAD_MATCH` before the demo is reached, so the run is reported as `BLOCKED_ENVIRONMENT` (exit 77, CTest: skipped) and not
  as `SKIP_UNSUPPORTED`. This happens with or without the debug flag; the runtime in `common/` decides it.
