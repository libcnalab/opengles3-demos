// es30-sync-objects
//
// Fence sync objects (ES 3.0 sections 5.1 and 5.2, queries in 6.1.8) and what glFlush / glFinish promise, observed
// through them. The image is a 4x4 grid of 64x64 cells; cell f shows a rectangle whose position, size and color
// depend only on f. The rectangles are drawn from TWO vertex buffers that the CPU rewrites in turn (a ring of 2):
//
//   for each update f:  slot = f % 2
//       if slot has a fence:  glClientWaitSync(fence, SYNC_FLUSH_COMMANDS_BIT, timeout)   // GPU finished the last draw that read the slot
//                             glDeleteSync(fence)
//       glMapBufferRange(slot, MAP_WRITE_BIT | MAP_UNSYNCHRONIZED_BIT)  ... write the 4 vertices ... glUnmapBuffer
//       glDrawArrays(...)                                               // draws cell f from the slot
//       fence = glFenceSync(SYNC_GPU_COMMANDS_COMPLETE, 0)
//
// MAP_UNSYNCHRONIZED_BIT tells the GL not to wait for the draw that may still be reading the buffer, so the fence is
// the only protection against overwriting data in use. Every pixel of every cell is compared with the model.
//
// validate() then looks at the sync object API itself: glGetSynciv properties, the legal results of
// glClientWaitSync, glWaitSync, MAX_SERVER_WAIT_TIMEOUT, the effect of glFlush (a fence becomes signaled in finite
// time) and glFinish (every fence is signaled when it returns), an asynchronous glReadPixels into a pixel pack
// buffer guarded by a fence, object lifetime (glIsSync / glDeleteSync) and the errors the specification defines.
// A fence cannot be made to time out deterministically, so TIMEOUT_EXPIRED is only ever accepted, never required.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    using Clock = std::chrono::steady_clock;
    constexpr int kGrid = 4;
    constexpr int kCells = kGrid * kGrid;
    constexpr int kRing = 2;
    constexpr std::array<std::uint8_t, 4> kClear8 = {20, 30, 50, 255};
    constexpr GLuint64 kTimeoutIgnored = GL_TIMEOUT_IGNORED;
    constexpr auto kBudget = std::chrono::seconds(10);  // bound of every wait that the specification guarantees to end "eventually"
    constexpr GLuint64 kSliceNs = 50'000'000;           // one finite ClientWaitSync timeout: 50 ms
    constexpr int kHeavyIterations = 200;
    constexpr std::array<float, 4> kHeavyColor = {0.25f, 0.4f, 0.75f, 1.0f};  // 8-bit (64, 102, 191, 255)

    struct Rect
    {
        int x, y, w, h;
    };
    Rect cell_rect(int w, int h, int id)  // cell 0 is the top left, reading order
    {
        const int cw = w / kGrid, ch = h / kGrid;
        return {(id % kGrid) * cw, (kGrid - 1 - id / kGrid) * ch, cw, ch};
    }

    // What update f draws (cell pixel coordinates, origin bottom left). The rectangle always fits a 64x64 cell:
    // x0 + w <= 4 + 19 + 16 + 23 = 62.
    struct CellSpec
    {
        int x0, y0, w, h;
        std::array<std::uint8_t, 3> rgb;
    };
    CellSpec cell_spec(int f)
    {
        return {4 + (f * 7) % 20,
                4 + (f * 5) % 20,
                16 + (f * 3) % 24,
                16 + (f * 11) % 24,
                {static_cast<std::uint8_t>(90 + (f * 53) % 150), static_cast<std::uint8_t>(90 + (f * 97) % 150), static_cast<std::uint8_t>(90 + (f * 29) % 150)}};
    }

    struct RingVertex  // 20 bytes: position (NDC), flat color
    {
        float x, y, r, g, b;
    };
    // Corner coordinates are integer pixel positions, so in NDC they are small dyadic fractions (exact in a float) and every
    // rectangle edge lies exactly between pixel centers: the coverage is unambiguous.
    std::array<RingVertex, 4> make_vertices(const CellSpec& s, int cell_w, int cell_h)
    {
        const float hx = static_cast<float>(cell_w) / 2.0f, hy = static_cast<float>(cell_h) / 2.0f;
        const float x0 = static_cast<float>(s.x0) / hx - 1.0f, x1 = static_cast<float>(s.x0 + s.w) / hx - 1.0f;
        const float y0 = static_cast<float>(s.y0) / hy - 1.0f, y1 = static_cast<float>(s.y0 + s.h) / hy - 1.0f;
        const float r = static_cast<float>(s.rgb[0]) / 255.0f, g = static_cast<float>(s.rgb[1]) / 255.0f, b = static_cast<float>(s.rgb[2]) / 255.0f;
        return {{{x0, y0, r, g, b}, {x1, y0, r, g, b}, {x0, y1, r, g, b}, {x1, y1, r, g, b}}};  // TRIANGLE_STRIP order
    }

    Rgba8 expected_pixel(const CellSpec& s, int x, int y)
    {
        const bool inside = x >= s.x0 && x < s.x0 + s.w && y >= s.y0 && y < s.y0 + s.h;
        return inside ? Rgba8{s.rgb[0], s.rgb[1], s.rgb[2], 255} : Rgba8{kClear8[0], kClear8[1], kClear8[2], kClear8[3]};
    }

    struct Compare
    {
        long covered = 0, background = 0, bad = 0;
        int max_dev = 0;
        std::string first_bad;
    };
    // Compares one cell of an image with the model of update `f`. Tolerance +-1: the flat float color is converted to 8 bits
    // (ES 3.0 section 2.1.6, equation 2.3); everything else is exact because the edges lie between pixel centers.
    Compare compare_cell(const std::vector<std::uint8_t>& image, int image_w, const Rect& r, int f)
    {
        Compare c;
        const CellSpec s = cell_spec(f);
        for (int y = 0; y < r.h; ++y)
            for (int x = 0; x < r.w; ++x)
            {
                const Rgba8 want = expected_pixel(s, x, y);
                const std::uint8_t* got = &image[static_cast<std::size_t>(((r.y + y) * image_w + r.x + x) * 4)];
                const bool inside = x >= s.x0 && x < s.x0 + s.w && y >= s.y0 && y < s.y0 + s.h;
                ++(inside ? c.covered : c.background);
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int{got[k]} - int{want[k]});
                    c.max_dev = std::max(c.max_dev, dev);
                    ok = ok && dev <= 1;
                }
                if (!ok && c.bad++ == 0)
                    c.first_bad = std::format("first mismatch at cell {} pixel ({},{}): got {} want {}", f, x, y, demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return c;
    }

    struct WaitStats  // what glClientWaitSync returned during a series of waits
    {
        int waits = 0;        // waits started (each loops over finite timeouts until the fence is signaled)
        int calls = 0;        // glClientWaitSync calls
        int already = 0, condition = 0, timeout = 0, failed = 0;
        int not_signaled = 0; // a wait said "signaled" but glGetSynciv disagreed, or the wait gave up
    };

    GLint sync_param(GLsync s, mgl::SyncParameter pname)
    {
        GLint value = -12345;
        GLsizei length = -1;
        mgl::glGetSynciv(s, pname, 1, &length, &value);
        return value;
    }
    bool is_signaled(GLsync s) { return sync_param(s, mgl::SyncParameter::Status) == GL_SIGNALED; }

    // Waits until `fence` is signaled, in slices of 50 ms with SYNC_FLUSH_COMMANDS_BIT (the flush makes sure that the fence
    // command is on its way to the GPU; without it the wait might never end, section 5.2.2). Every result of the series is
    // counted. Returns true when the fence was signaled within the budget.
    bool wait_until_signaled(GLsync fence, WaitStats& stats)
    {
        ++stats.waits;
        const auto start = Clock::now();
        for (;;)
        {
            const mgl::SyncWaitResult r = mgl::glClientWaitSync(fence, mgl::SyncFlushMask::Commands, kSliceNs);
            ++stats.calls;
            if (r == mgl::SyncWaitResult::AlreadySignaled || r == mgl::SyncWaitResult::ConditionSatisfied)
            {
                ++(r == mgl::SyncWaitResult::AlreadySignaled ? stats.already : stats.condition);
                if (!is_signaled(fence)) ++stats.not_signaled;  // the result promised a signaled fence
                return true;
            }
            if (r != mgl::SyncWaitResult::TimeoutExpired)
            {
                ++stats.failed;
                return false;
            }
            ++stats.timeout;
            if (Clock::now() - start > kBudget)
            {
                ++stats.not_signaled;
                return false;
            }
        }
    }

    void note(const std::string& text) { std::cout << "   [note] " << text << "\n"; }

    // A documented deviation of Mesa's Gallium drivers (they share the front end): strict on every other driver.
    bool expect_mesa_deviation(demo::Check& check, bool ok, std::string_view what, std::string_view detail, std::string_view reason)
    {
        const GLubyte* r = mgl::glGetString(mgl::StringName::Renderer);
        const std::string renderer = r != nullptr ? reinterpret_cast<const char*>(r) : "";
        for (const std::string_view name : {"zink", "radeonsi", "llvmpipe", "softpipe"})
            if (renderer.find(name) != std::string::npos) return check.expect_or_known_deviation(ok, what, detail, name, reason);
        return check.expect(ok, what, detail);
    }

    class SyncObjects final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            prog_ring_ = demo::build_program(env, "shaders/ring.vert", "shaders/ring.frag");
            prog_heavy_ = demo::build_program(env, "shaders/heavy.vert", "shaders/heavy.frag");
            u_iterations_ = mgl::glGetUniformLocation(prog_heavy_, "u_iterations");
            u_zero_ = mgl::glGetUniformLocation(prog_heavy_, "u_zero");
            u_color_ = mgl::glGetUniformLocation(prog_heavy_, "u_color");

            mgl::glGenBuffers(kRing, ring_.data());
            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            for (std::size_t i = 0; i < kRing; ++i)
            {
                mgl::glBindVertexArray(vao_[i]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, ring_[i]);
                mgl::glBufferData(mgl::BufferTarget::Array, static_cast<GLsizeiptr>(4 * sizeof(RingVertex)), nullptr, mgl::BufferUsage::StreamDraw);
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
                mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(RingVertex), demo::buffer_offset(offsetof(RingVertex, x)));
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
                mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(RingVertex), demo::buffer_offset(offsetof(RingVertex, r)));
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glGenBuffers(1, &pbo_);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            ring_stats_ = {};
            map_failures_ = 0;
            mgl::glUseProgram(prog_ring_);
            for (int f = 0; f < kCells; ++f)
            {
                const std::size_t slot = static_cast<std::size_t>(f % kRing);
                if (fence_[slot] != nullptr)
                {
                    // The draw that last read this slot may still be running: wait for ITS fence before writing the slot again.
                    (void)wait_until_signaled(fence_[slot], ring_stats_);
                    mgl::glDeleteSync(fence_[slot]);
                    fence_[slot] = nullptr;
                }
                const Rect r = cell_rect(env.width, env.height, f);
                const std::array<RingVertex, 4> vertices = make_vertices(cell_spec(f), r.w, r.h);
                mgl::glBindBuffer(mgl::BufferTarget::Array, ring_[slot]);
                void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::Array, 0, static_cast<GLsizeiptr>(sizeof(vertices)),
                                                     mgl::MapBufferAccessMask::Write | mgl::MapBufferAccessMask::Unsynchronized);
                if (mapped != nullptr)
                {
                    std::memcpy(mapped, vertices.data(), sizeof(vertices));
                    if (mgl::glUnmapBuffer(mgl::BufferTarget::Array) != GL_TRUE) ++map_failures_;
                }
                else ++map_failures_;
                mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
                mgl::glViewport(r.x, r.y, r.w, r.h);
                mgl::glBindVertexArray(vao_[slot]);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
                fence_[slot] = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_image(env, check);
            check_ring_protocol(check);
            check_properties_and_results(env, check);
            check_flush_and_finish(env, check);
            check_async_readback(env, check);
            check_server_wait(env, check);
            check_lifecycle_and_errors(env, check);
            check_many_fences(env, check);
            render(env, 0);  // restore the capture image (and run the ring protocol once more)
        }

        void shutdown(demo::Env& /*env*/) override
        {
            for (GLsync& f : fence_)
            {
                if (f != nullptr) mgl::glDeleteSync(f);
                f = nullptr;
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(kRing, ring_.data());
            mgl::glDeleteBuffers(1, &pbo_);
            mgl::glDeleteProgram(prog_ring_);
            mgl::glDeleteProgram(prog_heavy_);
        }

    private:
        // ---------------------------------------------------------------------------------------------
        // helpers
        // ---------------------------------------------------------------------------------------------
        // A full-viewport draw that keeps the GPU busy for a while. The color is exactly `color` (see heavy.frag).
        void submit_heavy(const demo::Env& env, int iterations, const std::array<float, 4>& color) const
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glUseProgram(prog_heavy_);
            mgl::glUniform1i(u_iterations_, iterations);
            mgl::glUniform1f(u_zero_, 0.0f);
            mgl::glUniform4f(u_color_, color[0], color[1], color[2], color[3]);
            mgl::glBindVertexArray(vao_[kRing]);  // no attributes: the vertex shader works from gl_VertexID
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---------------------------------------------------------------------------------------------
        // validate: the image drawn through the ring
        // ---------------------------------------------------------------------------------------------
        void check_image(const demo::Env& env, demo::Check& check) const
        {
            const std::vector<std::uint8_t> image = demo::read_rgba8(0, 0, env.width, env.height);
            long covered = 0, background = 0, bad = 0;
            int max_dev = 0;
            std::string first_bad;
            int cells_that_differ_from_slot_mate = 0;
            for (int f = 0; f < kCells; ++f)
            {
                const Rect r = cell_rect(env.width, env.height, f);
                const Compare c = compare_cell(image, env.width, r, f);
                covered += c.covered;
                background += c.background;
                bad += c.bad;
                max_dev = std::max(max_dev, c.max_dev);
                if (c.bad != 0 && first_bad.empty()) first_bad = c.first_bad;
                // Negative control: the same pixels against the model of the update that used the same ring slot before or after (f + 2).
                if (compare_cell(image, env.width, r, (f + kRing) % kCells).bad > 0) ++cells_that_differ_from_slot_mate;
            }
            const long expected_covered = [&] {
                long n = 0;
                for (int f = 0; f < kCells; ++f) n += static_cast<long>(cell_spec(f).w) * cell_spec(f).h;
                return n;
            }();
            check.expect(bad == 0 && covered == expected_covered && background == static_cast<long>(env.width) * env.height - expected_covered,
                         std::format("all {} cells match the model pixel for pixel: {} covered + {} background pixels (max deviation {}/255)", kCells, covered, background, max_dev),
                         bad ? std::format("{} mismatching pixels; {}", bad, first_bad) : std::string{});
            check.expect(cells_that_differ_from_slot_mate == kCells,
                         "negative control: every cell disagrees with the model of the update that shared its ring slot (f + 2), so stale slot data would be noticed",
                         std::format("{} of {} cells differ", cells_that_differ_from_slot_mate, kCells));
        }

        // ---------------------------------------------------------------------------------------------
        // validate: what the ring protocol observed
        // ---------------------------------------------------------------------------------------------
        void check_ring_protocol(demo::Check& check)
        {
            const WaitStats& s = ring_stats_;
            check.expect(s.waits == kCells - kRing && s.failed == 0 && s.not_signaled == 0 && s.already + s.condition == s.waits,
                         std::format("ring protocol: {} fence waits (updates 2..15), every one ended ALREADY_SIGNALED or CONDITION_SATISFIED with the fence signaled; no WAIT_FAILED", kCells - kRing),
                         std::format("waits {} calls {} already {} condition {} timeout {} failed {} not signaled {}", s.waits, s.calls, s.already, s.condition, s.timeout, s.failed, s.not_signaled));
            note(std::format("ring waits: {} x ALREADY_SIGNALED, {} x CONDITION_SATISFIED, {} x TIMEOUT_EXPIRED (a {} ms slice that ran out before the fence was signaled)", s.already, s.condition, s.timeout, kSliceNs / 1'000'000));
            check.expect_eq(map_failures_, 0, "every glMapBufferRange(MAP_WRITE | MAP_UNSYNCHRONIZED) and glUnmapBuffer of the ring succeeded");
            check.expect(fence_[0] != nullptr && fence_[1] != nullptr && mgl::glIsSync(fence_[0]) && mgl::glIsSync(fence_[1]) && fence_[0] != fence_[1],
                         "the two fences of the last updates are distinct, valid sync objects");
            mgl::glFinish();
            check.expect(is_signaled(fence_[0]) && is_signaled(fence_[1]), "after glFinish both outstanding ring fences have SYNC_STATUS = SIGNALED");
            check.expect_no_gl_error("the ring protocol raised no GL error");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: glGetSynciv and the results of glClientWaitSync
        // ---------------------------------------------------------------------------------------------
        void check_properties_and_results(const demo::Env& env, demo::Check& check)
        {
            using R = mgl::SyncWaitResult;
            submit_heavy(env, kHeavyIterations, kHeavyColor);
            GLsync f = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            check.expect(f != nullptr && mgl::glIsSync(f), "glFenceSync(SYNC_GPU_COMMANDS_COMPLETE, 0) returns a non-zero name; glIsSync is TRUE");

            const GLint first_status = sync_param(f, mgl::SyncParameter::Status);
            check.expect(first_status == GL_UNSIGNALED || first_status == GL_SIGNALED, "SYNC_STATUS right after creation is UNSIGNALED or SIGNALED (initially UNSIGNALED, Table 5.1; the GPU may be quicker than the query)");
            note(std::format("SYNC_STATUS immediately after the fence of a heavy draw: {}", first_status == GL_UNSIGNALED ? "UNSIGNALED" : "SIGNALED"));
            check.expect_eq(sync_param(f, mgl::SyncParameter::ObjectType), static_cast<GLint>(GL_SYNC_FENCE), "OBJECT_TYPE = SYNC_FENCE (the only type)");
            check.expect_eq(sync_param(f, mgl::SyncParameter::Condition), static_cast<GLint>(GL_SYNC_GPU_COMMANDS_COMPLETE), "SYNC_CONDITION = SYNC_GPU_COMMANDS_COMPLETE (the condition passed to glFenceSync)");
            check.expect_eq(sync_param(f, mgl::SyncParameter::Flags), 0, "SYNC_FLAGS = 0 (the flags passed to glFenceSync)");
            {
                GLint values[2] = {777, 777};
                GLsizei length = -1;
                mgl::glGetSynciv(f, mgl::SyncParameter::ObjectType, 2, &length, values);
                check.expect(length == 1 && values[0] == static_cast<GLint>(GL_SYNC_FENCE) && values[1] == 777, "glGetSynciv with bufSize 2 writes one value and reports length 1 (\"up to bufSize integers\")");
                length = -1;
                values[0] = 777;
                mgl::glGetSynciv(f, mgl::SyncParameter::ObjectType, 0, &length, values);
                check.expect(values[0] == 777, "glGetSynciv with bufSize 0 writes nothing (\"up to bufSize integers\")");
                expect_mesa_deviation(check, length == 0, "glGetSynciv with bufSize 0 reports length 0 (\"the actual number of integers replaced is returned in length\")", std::format("length = {}", length),
                                      "Mesa reports the size of the property (1) as length even when bufSize is 0 and nothing was written");
                GLint alone = 0;
                mgl::glGetSynciv(f, mgl::SyncParameter::Status, 1, nullptr, &alone);
                check.expect(alone == GL_UNSIGNALED || alone == GL_SIGNALED, "glGetSynciv accepts a null length pointer (\"if length is NULL, no length is returned\")");
            }
            check.expect_no_gl_error("the property queries raised no error");

            // Polling with timeout 0 never blocks and has exactly two legal results (section 5.2.1).
            const R r0 = mgl::glClientWaitSync(f, mgl::SyncFlushMask::None, 0);
            const bool r0_legal = r0 == R::AlreadySignaled || r0 == R::TimeoutExpired;
            check.expect(r0_legal && (r0 != R::AlreadySignaled || is_signaled(f)),
                         std::format("glClientWaitSync(timeout 0, no flush) returned {}: legal results are ALREADY_SIGNALED and TIMEOUT_EXPIRED (never CONDITION_SATISFIED or WAIT_FAILED)", mgl::to_string(r0)));
            const R r1 = mgl::glClientWaitSync(f, mgl::SyncFlushMask::Commands, 0);
            check.expect(r1 == R::AlreadySignaled || r1 == R::TimeoutExpired, std::format("glClientWaitSync(timeout 0, SYNC_FLUSH_COMMANDS_BIT) returned {}: same two legal results", mgl::to_string(r1)));
            note(std::format("polling a fresh fence with timeout 0: {} (no flush), then {} (with flush)", mgl::to_string(r0), mgl::to_string(r1)));

            // Waiting with a finite timeout and the flush bit ends with the fence signaled; every individual result is one of the three legal ones.
            WaitStats stats;
            const bool signaled = wait_until_signaled(f, stats);
            check.expect(signaled && stats.failed == 0 && stats.not_signaled == 0,
                         std::format("glClientWaitSync(SYNC_FLUSH_COMMANDS_BIT, 50 ms) repeated until it stops timing out ends with ALREADY_SIGNALED or CONDITION_SATISFIED ({} calls, {} TIMEOUT_EXPIRED on the way)", stats.calls, stats.timeout));

            // Once signaled, a fence stays signaled and every wait is "already signaled", whatever the timeout (section 5.2.1).
            check.expect(mgl::glClientWaitSync(f, mgl::SyncFlushMask::None, 0) == R::AlreadySignaled, "a signaled fence: glClientWaitSync with timeout 0 returns ALREADY_SIGNALED (\"even if the value of timeout is zero\")");
            check.expect(mgl::glClientWaitSync(f, mgl::SyncFlushMask::Commands, 1'000'000'000) == R::AlreadySignaled, "... with a 1 s timeout and the flush bit: ALREADY_SIGNALED at once");
            check.expect(mgl::glClientWaitSync(f, mgl::SyncFlushMask::Commands, kTimeoutIgnored) == R::AlreadySignaled, "... with the largest timeout (2^64 - 1 ns): ALREADY_SIGNALED at once");
            check.expect(is_signaled(f), "SYNC_STATUS = SIGNALED, and it stays SIGNALED on every later query");
            check.expect(sync_param(f, mgl::SyncParameter::ObjectType) == static_cast<GLint>(GL_SYNC_FENCE) && sync_param(f, mgl::SyncParameter::Condition) == static_cast<GLint>(GL_SYNC_GPU_COMMANDS_COMPLETE) && sync_param(f, mgl::SyncParameter::Flags) == 0,
                         "signaling changes only SYNC_STATUS: type, condition and flags are unchanged");
            mgl::glDeleteSync(f);
            check.expect_no_gl_error("the wait sections raised no error");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: glFlush and glFinish (ES 3.0 section 5.1), observed through fences
        // ---------------------------------------------------------------------------------------------
        void check_flush_and_finish(const demo::Env& env, demo::Check& check)
        {
            using R = mgl::SyncWaitResult;
            // glFlush: all commands "must complete in finite time". Nothing but glFlush and status queries is used here.
            submit_heavy(env, kHeavyIterations, kHeavyColor);
            GLsync flushed = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            mgl::glFlush();
            long polls = 0;
            const auto start = Clock::now();
            while (!is_signaled(flushed) && Clock::now() - start < kBudget)
            {
                ++polls;
                std::this_thread::yield();
            }
            const bool reached = is_signaled(flushed);
            check.expect(reached, std::format("glFlush: the fence of a heavy draw became SIGNALED in finite time (glGetSynciv polled {} times; no wait call was used)", polls));
            check.expect(is_signaled(flushed) && is_signaled(flushed) && is_signaled(flushed), "... and it stays SIGNALED");

            // glFinish: when it returns, every earlier command is complete, so every earlier fence is signaled.
            submit_heavy(env, kHeavyIterations, kHeavyColor);
            GLsync finished = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            mgl::glFinish();
            const bool signaled_at_once = is_signaled(finished);
            check.expect(signaled_at_once, "glFinish: the fence created just before it has SYNC_STATUS = SIGNALED on the very first query");
            check.expect(mgl::glClientWaitSync(finished, mgl::SyncFlushMask::None, 0) == R::AlreadySignaled, "glFinish: glClientWaitSync(timeout 0, no flush) on that fence returns ALREADY_SIGNALED");
            mgl::glDeleteSync(flushed);
            mgl::glDeleteSync(finished);

            // A fence completes only after everything before it, including earlier fences (section 5.2: "all preceding commands in the same command stream").
            GLsync first = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            submit_heavy(env, kHeavyIterations, kHeavyColor);
            GLsync second = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            WaitStats stats;
            const bool second_done = wait_until_signaled(second, stats);
            check.expect(second_done && is_signaled(first), "fences are ordered: when the second fence (after a heavy draw) is signaled, the first one (before it) is signaled too");
            mgl::glDeleteSync(first);
            mgl::glDeleteSync(second);
            check.expect_no_gl_error("the flush/finish sections raised no error");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: asynchronous readback into a pixel pack buffer, guarded by a fence
        // ---------------------------------------------------------------------------------------------
        void check_async_readback(const demo::Env& env, demo::Check& check)
        {
            using R = mgl::SyncWaitResult;
            const std::size_t bytes = static_cast<std::size_t>(env.width) * static_cast<std::size_t>(env.height) * 4;
            const std::vector<std::uint8_t> sentinel(bytes, 0xA5);
            auto compare_pbo = [&] {
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, pbo_);
                const std::uint8_t* mapped = static_cast<const std::uint8_t*>(mgl::glMapBufferRange(mgl::BufferTarget::PixelPack, 0, static_cast<GLsizeiptr>(bytes), mgl::MapBufferAccessMask::Read));
                long bad = -1;
                int max_dev = 0;
                if (mapped != nullptr)
                {
                    bad = 0;
                    const std::array<int, 4> want = {64, 102, 191, 255};  // kHeavyColor in 8 bits
                    for (std::size_t i = 0; i < bytes; ++i)
                    {
                        const int dev = std::abs(int{mapped[i]} - want[i % 4]);
                        max_dev = std::max(max_dev, dev);
                        if (dev > 1) ++bad;
                    }
                    (void)mgl::glUnmapBuffer(mgl::BufferTarget::PixelPack);
                }
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
                return std::pair{bad, max_dev};
            };

            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, pbo_);
            mgl::glBufferData(mgl::BufferTarget::PixelPack, static_cast<GLsizeiptr>(bytes), sentinel.data(), mgl::BufferUsage::StreamRead);
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            const auto before = compare_pbo();
            check.expect(before.first == static_cast<long>(bytes), "negative control: the pixel pack buffer filled with 0xA5 disagrees with the expected image in every byte", std::format("{} of {} bytes differ", before.first, bytes));

            submit_heavy(env, kHeavyIterations, kHeavyColor);
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, pbo_);
            mgl::glReadPixels(0, 0, env.width, env.height, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);  // nullptr = offset 0 into the pixel pack buffer; returns before the copy finished
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            GLsync fence = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            mgl::glFlush();  // a fence that is polled must be on its way to the GPU
            long polls = 0, timeouts = 0;
            bool legal = true;
            R result = R::TimeoutExpired;
            const auto start = Clock::now();
            while (Clock::now() - start < kBudget)
            {
                result = mgl::glClientWaitSync(fence, mgl::SyncFlushMask::None, 0);  // poll: never blocks
                ++polls;
                if (result == R::AlreadySignaled) break;
                if (result != R::TimeoutExpired) legal = false;
                ++timeouts;
                std::this_thread::yield();
            }
            check.expect(legal && result == R::AlreadySignaled, std::format("polling the fence with glClientWaitSync(timeout 0) reached ALREADY_SIGNALED after {} polls ({} TIMEOUT_EXPIRED), no other result", polls, timeouts));
            const auto after = compare_pbo();
            check.expect(after.first == 0, std::format("once the fence is signaled the pixel pack buffer holds the whole {}x{} heavy-draw image (max deviation {}/255)", env.width, env.height, after.second), after.first < 0 ? "buffer could not be mapped" : std::format("{} bytes differ", after.first));
            mgl::glDeleteSync(fence);
            check.expect_no_gl_error("the asynchronous readback raised no error");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: glWaitSync and MAX_SERVER_WAIT_TIMEOUT
        // ---------------------------------------------------------------------------------------------
        void check_server_wait(const demo::Env& env, demo::Check& check)
        {
            GLint64 timeout64 = -1;
            mgl::glGetInteger64v(mgl::GetParameter::MaxServerWaitTimeout, &timeout64);
            GLint timeout32 = -1;
            mgl::glGetIntegerv(mgl::GetParameter::MaxServerWaitTimeout, &timeout32);
            check.expect(timeout64 >= 0, std::format("MAX_SERVER_WAIT_TIMEOUT = {} ns (minimum 0, Table 6.29)", timeout64));
            check.expect(timeout32 == static_cast<GLint>(std::min<GLint64>(timeout64, INT_MAX)), "glGetIntegerv returns the 64-bit value clamped to the int range (section 6.1.2)", std::format("32-bit {} vs 64-bit {}", timeout32, timeout64));
            note(std::format("MAX_SERVER_WAIT_TIMEOUT = {} ns", timeout64));

            // A server-side wait does not block the client and does not change the order of commands: the draw after it still lands.
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            submit_heavy(env, kHeavyIterations, kHeavyColor);
            GLsync fence = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            mgl::glWaitSync(fence, mgl::SyncFlag::None, kTimeoutIgnored);
            check.expect_no_gl_error("glWaitSync(fence, 0, TIMEOUT_IGNORED) raises no error");
            submit_heavy(env, 1, {0.8f, 0.2f, 0.1f, 1.0f});  // queued behind the WaitSync
            mgl::glFinish();
            check.expect_pixel(env.width / 2, env.height / 2, {204, 51, 26, 255}, 1, "the draw queued after glWaitSync executed: the center pixel has its color (0.8, 0.2, 0.1)");
            check.expect(is_signaled(fence), "after glFinish the fence that the server waited for is SIGNALED");
            mgl::glWaitSync(fence, mgl::SyncFlag::None, kTimeoutIgnored);
            check.expect_no_gl_error("glWaitSync on an already signaled fence is legal and returns at once");
            mgl::glDeleteSync(fence);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
        }

        // ---------------------------------------------------------------------------------------------
        // validate: object lifetime and the defined errors (ES 3.0 sections 5.2, 5.2.1, 6.1.8)
        // ---------------------------------------------------------------------------------------------
        void check_lifecycle_and_errors(const demo::Env& env, demo::Check& check)
        {
            using Err = mgl::ErrorCode;
            using R = mgl::SyncWaitResult;
            const GLsync none = nullptr;

            // Deleting a sync that is probably still unsignaled is legal: the name becomes invalid at once, the object lives on until it is signaled.
            submit_heavy(env, kHeavyIterations, kHeavyColor);
            GLsync busy = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            check.expect(mgl::glIsSync(busy), "glIsSync is TRUE for a fence that was just created");
            mgl::glDeleteSync(busy);
            check.expect_no_gl_error("glDeleteSync of a fence that is (most likely) not signaled yet raises no error");
            check.expect(!mgl::glIsSync(busy), "glIsSync is FALSE immediately after glDeleteSync (\"the sync name is invalid\", section 5.2)");
            mgl::glFinish();
            check.expect_no_gl_error("glFinish after the deletion raises no error");

            // The name zero is not a sync object.
            check.expect(!mgl::glIsSync(none), "glIsSync(0) is FALSE (zero is not the name of a sync object)");
            mgl::glDeleteSync(none);
            check.expect_no_gl_error("glDeleteSync(0) is silently ignored");
            check.expect(mgl::glClientWaitSync(none, mgl::SyncFlushMask::None, 0) == R::WaitFailed, "glClientWaitSync(0, ...) returns WAIT_FAILED");
            check.expect_gl_error(Err::InvalidValue, "... and generates INVALID_VALUE (sync is not the name of a sync object)");
            mgl::glWaitSync(none, mgl::SyncFlag::None, kTimeoutIgnored);
            check.expect_gl_error(Err::InvalidValue, "glWaitSync(0, 0, TIMEOUT_IGNORED) -> INVALID_VALUE");
            {
                GLint value = 4242;
                GLsizei length = 4242;
                mgl::glGetSynciv(none, mgl::SyncParameter::Status, 1, &length, &value);
                check.expect_gl_error(Err::InvalidValue, "glGetSynciv(0, SYNC_STATUS, ...) -> INVALID_VALUE");
                check.expect(value == 4242 && length == 4242, "... and nothing is written to values or length (\"if an error occurs, nothing will be written\")");
            }

            // A deleted name is no longer valid, even though the memory behind it may still exist.
            GLsync gone = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            mgl::glDeleteSync(gone);
            check.expect(!mgl::glIsSync(gone), "a deleted name: glIsSync is FALSE");
            check.expect(mgl::glClientWaitSync(gone, mgl::SyncFlushMask::Commands, 0) == R::WaitFailed, "a deleted name: glClientWaitSync returns WAIT_FAILED");
            check.expect_gl_error(Err::InvalidValue, "... and generates INVALID_VALUE");
            mgl::glWaitSync(gone, mgl::SyncFlag::None, kTimeoutIgnored);
            check.expect_gl_error(Err::InvalidValue, "a deleted name: glWaitSync -> INVALID_VALUE");
            {
                GLint value = 4242;
                GLsizei length = 4242;
                mgl::glGetSynciv(gone, mgl::SyncParameter::Status, 1, &length, &value);
                check.expect_gl_error(Err::InvalidValue, "a deleted name: glGetSynciv -> INVALID_VALUE");
                check.expect(value == 4242 && length == 4242, "... and nothing is written to values or length");
            }
            mgl::glDeleteSync(gone);
            check.expect_gl_error(Err::InvalidValue, "a deleted name: glDeleteSync -> INVALID_VALUE (\"neither zero nor the name of a sync object\")");

            // glWaitSync: timeout must be TIMEOUT_IGNORED (flags must be 0; the only flags value of meta-gl's SyncFlag is None).
            GLsync fence = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            mgl::glWaitSync(fence, mgl::SyncFlag::None, 0);
            check.expect_gl_error(Err::InvalidValue, "glWaitSync with timeout 0 -> INVALID_VALUE (timeout must be TIMEOUT_IGNORED)");
            mgl::glWaitSync(fence, mgl::SyncFlag::None, 1'000'000'000);
            check.expect_gl_error(Err::InvalidValue, "glWaitSync with timeout 1 s -> INVALID_VALUE");
            mgl::glWaitSync(fence, mgl::SyncFlag::None, kTimeoutIgnored - 1);
            check.expect_gl_error(Err::InvalidValue, "glWaitSync with timeout 2^64 - 2 -> INVALID_VALUE (only the exact value TIMEOUT_IGNORED is accepted)");
            mgl::glWaitSync(fence, mgl::SyncFlag::None, kTimeoutIgnored);
            check.expect_no_gl_error("positive control: glWaitSync with TIMEOUT_IGNORED on the same fence succeeds");
            mgl::glDeleteSync(fence);
            note("not exercised: glFenceSync with a non-zero flags value or another condition (INVALID_VALUE / INVALID_ENUM), glClientWaitSync with flag bits other than SYNC_FLUSH_COMMANDS_BIT (INVALID_VALUE), "
                 "glWaitSync with non-zero flags (INVALID_VALUE) and glGetSynciv with an unknown pname (INVALID_ENUM): meta-gl's SyncFlag, SyncCondition, SyncFlushMask and SyncParameter have no such values, "
                 "so the calls would need a raw cast");
            mgl::glFinish();
            check.expect_no_gl_error("the lifetime and error section ends with an empty error queue");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: many fences
        // ---------------------------------------------------------------------------------------------
        void check_many_fences(const demo::Env& env, demo::Check& check)
        {
            std::array<GLsync, 32> fences{};
            for (GLsync& f : fences)
            {
                submit_heavy(env, 1, kHeavyColor);  // a (cheap) command between the fences
                f = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            }
            std::vector<GLsync> sorted(fences.begin(), fences.end());
            std::sort(sorted.begin(), sorted.end());
            const bool distinct = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end() && sorted.front() != nullptr;
            check.expect(distinct && std::all_of(fences.begin(), fences.end(), [](GLsync f) { return mgl::glIsSync(f); }), "32 fences created back to back have distinct non-zero names and all are sync objects");
            WaitStats stats;
            const bool last_done = wait_until_signaled(fences.back(), stats);
            check.expect(last_done && std::all_of(fences.begin(), fences.end(), [](GLsync f) { return is_signaled(f); }), "when the last of the 32 fences is signaled, all 32 are (fences complete in order)");
            for (GLsync f : fences) mgl::glDeleteSync(f);
            check.expect(std::none_of(fences.begin(), fences.end(), [](GLsync f) { return mgl::glIsSync(f); }), "after glDeleteSync none of the 32 names is a sync object any more");
            check.expect_no_gl_error("the many-fences section raised no error");
        }

        mgl::ProgramId prog_ring_{}, prog_heavy_{};
        mgl::UniformLocation u_iterations_{}, u_zero_{}, u_color_{};
        std::array<mgl::BufferId, kRing> ring_{};
        mgl::BufferId pbo_{};
        std::array<mgl::VertexArrayId, kRing + 1> vao_{};  // [0..1] ring slots, [2] no attributes
        std::array<GLsync, kRing> fence_{};
        WaitStats ring_stats_;
        int map_failures_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-sync-objects",
        .title = "Sync objects: fences, client and server waits, flush and finish",
        .min_es = {3, 0},
        .summary = "Fence sync objects: a two-slot buffer ring whose unsynchronized rewrites are guarded by fences (every pixel checked), "
                   "glClientWaitSync / glWaitSync results, glGetSynciv, glFlush / glFinish observed through fences, an asynchronous "
                   "readback into a pixel pack buffer, object lifetime and the defined errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<SyncObjects>(argc, argv, info);
}
