// es30-buffer-mapping
//
// glMapBufferRange with every access flag, glFlushMappedBufferRange, glUnmapBuffer, the mapped-state
// queries (BUFFER_MAPPED, BUFFER_ACCESS_FLAGS, BUFFER_MAP_OFFSET, BUFFER_MAP_LENGTH, BUFFER_MAP_POINTER),
// glCopyBufferSubData and the errors the spec defines for all of them (ES 3.0 sections 2.10.1 - 2.10.5, 6.1.9).
//
// The 256x256 surface is a 4x4 grid of 64x64 cells (reading order, row 0 on top). Every cell draws the
// SAME kind of content - a 2x2 arrangement of four colored quads, 6 vertices of 12 bytes each, 288 bytes -
// but the bytes get into the buffer by a different mechanism:
//
//   [0] MAP_WRITE_BIT                          [1] WRITE | INVALIDATE_BUFFER (orphaning, whole store)
//   [2] READ | WRITE, in-place edit            [3] WRITE | INVALIDATE_RANGE on quads 1-2, quads 0 and 3 survive
//   [4] WRITE | FLUSH_EXPLICIT, 3 flush calls  [5] FLUSH_EXPLICIT at mapping offset 72, quad 2 never flushed (not drawn)
//   [6] WRITE | INVALIDATE_RANGE | UNSYNCH.    [7] glCopyBufferSubData COPY_READ -> COPY_WRITE (two buffers)
//   [8] glCopyBufferSubData inside ONE buffer   [9] failed Map/Flush/Unmap calls change nothing
//   [10] glBufferData unmaps implicitly         [11] one buffer, two bindings: map through one, query through another
//   [12] WRITE | INVALIDATE_BUFFER on quads 1-2 (rest undefined, not drawn)
//   [13] rejected glCopyBufferSubData calls change nothing
//   [14] glBufferSubData rejected while mapped; a draw from a mapped buffer is only observed, never asserted
//   [15] READ | WRITE at a non-zero offset: the two top quads are read and swapped in place
//
// Two kinds of evidence are collected while the frame is rendered and reported by validate():
//   * exact bytes: after each step the whole buffer is mapped with MAP_READ_BIT and compared with a CPU
//     copy. The CPU copy tracks which bytes the specification DEFINES; bytes that are undefined by the
//     flags used (invalidated, written but never flushed) are skipped, so the demo asserts only what the
//     spec guarantees and never that an unflushed range did NOT arrive;
//   * pixels: each cell is drawn from the buffer and the quads are probed against the colors the CPU
//     says they must have.
// Further checks (all 64 combinations of the six access flags against the spec's rule table, error cases,
// state queries) are made against the buffers directly.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using Bytes = std::vector<std::uint8_t>;

namespace
{
    constexpr int kGrid = 4;
    constexpr int kCells = kGrid * kGrid;
    constexpr Rgba8 kClear = {20, 30, 50, 255};   // inside a cell
    constexpr Rgba8 kBorder = {70, 78, 96, 255};   // the 2-pixel grid lines between cells
    constexpr int kBorderPx = 2;

    // -----------------------------------------------------------------------------------------
    // Vertex data: a quad is two triangles, six vertices. The buffers hold four quads (one per slot).
    // -----------------------------------------------------------------------------------------
    struct Vertex
    {
        float x, y;
        std::uint8_t r, g, b, a;
    };
    static_assert(sizeof(Vertex) == 12 && offsetof(Vertex, r) == 8, "vertex layout is part of the demo");

    constexpr std::ptrdiff_t kQuadVertices = 6;
    constexpr std::ptrdiff_t kQuadBytes = kQuadVertices * static_cast<std::ptrdiff_t>(sizeof(Vertex));  // 72
    constexpr std::ptrdiff_t kQuads = 4;
    constexpr std::ptrdiff_t kSceneBytes = kQuads * kQuadBytes;  // 288

    constexpr Rgba8 kRed{220, 50, 50, 255}, kGreen{50, 200, 70, 255}, kBlue{60, 90, 235, 255},
        kYellow{235, 210, 50, 255}, kCyan{50, 210, 215, 255}, kMagenta{210, 70, 205, 255},
        kOrange{245, 140, 40, 255}, kWhite{235, 235, 235, 255}, kGray{110, 110, 110, 255},
        kSlate{70, 85, 120, 255}, kOlive{120, 130, 50, 255}, kPlum{120, 60, 110, 255}, kTeal{40, 120, 120, 255};

    /// Position s: 0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right of the cell (NDC centers +-0.5).
    Bytes quad_bytes(int position, Rgba8 c)
    {
        const float cx = (position % 2 == 0) ? -0.5f : 0.5f;
        const float cy = (position / 2 == 0) ? -0.5f : 0.5f;
        constexpr float h = 0.4f;
        const float x0 = cx - h, x1 = cx + h, y0 = cy - h, y1 = cy + h;
        const std::array<Vertex, 6> v = {{
            {x0, y0, c[0], c[1], c[2], c[3]}, {x1, y0, c[0], c[1], c[2], c[3]}, {x0, y1, c[0], c[1], c[2], c[3]},
            {x1, y0, c[0], c[1], c[2], c[3]}, {x1, y1, c[0], c[1], c[2], c[3]}, {x0, y1, c[0], c[1], c[2], c[3]},
        }};
        Bytes out(static_cast<std::size_t>(kQuadBytes));
        std::memcpy(out.data(), v.data(), out.size());
        return out;
    }

    /// Four quads, slot s holds the quad for position s.
    Bytes scene_bytes(const std::array<Rgba8, 4>& colors)
    {
        Bytes out;
        for (int s = 0; s < 4; ++s)
        {
            const Bytes q = quad_bytes(s, colors[static_cast<std::size_t>(s)]);
            out.insert(out.end(), q.begin(), q.end());
        }
        return out;
    }

    Bytes concat(const Bytes& a, const Bytes& b)
    {
        Bytes out = a;
        out.insert(out.end(), b.begin(), b.end());
        return out;
    }

    Rgba8 invert(Rgba8 c)
    {
        return {static_cast<std::uint8_t>(255 - c[0]), static_cast<std::uint8_t>(255 - c[1]),
                static_cast<std::uint8_t>(255 - c[2]), c[3]};
    }

    /// Color stored in vertex v of a scene/quad byte array.
    Rgba8 color_at(const Bytes& data, std::size_t vertex)
    {
        const std::size_t o = vertex * sizeof(Vertex) + offsetof(Vertex, r);
        return {data[o], data[o + 1], data[o + 2], data[o + 3]};
    }

    // -----------------------------------------------------------------------------------------
    // CPU copy of a buffer that knows which bytes the specification defines.
    // -----------------------------------------------------------------------------------------
    class Shadow
    {
    public:
        explicit Shadow(std::size_t size = static_cast<std::size_t>(kSceneBytes)) : bytes_(size, 0), known_(size, 0) {}

        /// The GL buffer now holds `data` at `offset` (a glBufferData/BufferSubData/written-and-flushed range).
        void set(std::ptrdiff_t offset, const Bytes& data)
        {
            for (std::size_t i = 0; i < data.size(); ++i)
            {
                bytes_[static_cast<std::size_t>(offset) + i] = data[i];
                known_[static_cast<std::size_t>(offset) + i] = 1;
            }
        }
        /// The contents of [offset, offset+length) are undefined by the spec (invalidated, unflushed, ...).
        void forget(std::ptrdiff_t offset, std::ptrdiff_t length)
        {
            for (std::ptrdiff_t i = 0; i < length; ++i) known_[static_cast<std::size_t>(offset + i)] = 0;
        }
        void forget_all() { std::fill(known_.begin(), known_.end(), std::uint8_t{0}); }

        struct Result
        {
            std::size_t compared = 0, skipped = 0, mismatches = 0;
            std::ptrdiff_t first = -1;
        };
        [[nodiscard]] Result compare(const Bytes& actual) const
        {
            Result r;
            for (std::size_t i = 0; i < bytes_.size(); ++i)
            {
                if (known_[i] == 0)
                {
                    ++r.skipped;
                    continue;
                }
                ++r.compared;
                if (i >= actual.size() || actual[i] != bytes_[i])
                {
                    ++r.mismatches;
                    if (r.first < 0) r.first = static_cast<std::ptrdiff_t>(i);
                }
            }
            return r;
        }

    private:
        Bytes bytes_;
        std::vector<std::uint8_t> known_;
    };

    using Slots = std::array<std::optional<Rgba8>, 4>;

    /// One deferred observation: made while rendering, reported by validate().
    struct Obs
    {
        bool ok;
        std::string what;
        std::string detail;
        std::string deviation_renderer;  ///< non-empty: a documented driver deviation (see Check::expect_or_known_deviation)
        std::string deviation_reason;
    };

    class BufferMapping final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/quads.vert", "shaders/quads.frag");
            u_shift_ = mgl::glGetUniformLocation(program_, "u_shift");
            if (u_shift_.value < 0) throw demo::Failure("uniform u_shift is not active");

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glEnableVertexAttribArray(kLocPosition);
            mgl::glEnableVertexAttribArray(kLocColor);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            mgl::glGenBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glGenBuffers(static_cast<GLsizei>(aux_.size()), aux_.data());
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            obs_.clear();
            model_ = {};
            mapped_draw_note_.clear();

            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glClearColor(kBorder[0] / 255.0f, kBorder[1] / 255.0f, kBorder[2] / 255.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            for (int cell = 0; cell < kCells; ++cell) clear_cell(env, cell);
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);

            cell0_plain_write(env);
            cell1_invalidate_buffer(env);
            cell2_read_write(env);
            cell3_invalidate_range(env);
            cell4_flush_explicit(env);
            cell5_flush_partial(env);
            cell6_unsynchronized(env);
            cell7_copy_two_buffers(env);
            cell8_copy_same_buffer(env);
            cell9_failed_calls(env);
            cell10_buffer_data_unmaps(env);
            cell11_two_bindings(env);
            cell12_invalidate_buffer_range(env);
            cell13_failed_copies(env);
            cell14_subdata_while_mapped(env);
            cell15_read_write_offset(env);

            flag_matrix();
            delete_while_mapped();

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            // 1. Everything observed while rendering: exact bytes, states, defined errors.
            for (const Obs& o : obs_)
            {
                if (o.deviation_renderer.empty()) check.expect(o.ok, o.what, o.detail);
                else check.expect_or_known_deviation(o.ok, o.what, o.detail, o.deviation_renderer, o.deviation_reason);
            }
            if (!mapped_draw_note_.empty()) std::cout << "   [note] " << mapped_draw_note_ << "\n";

            // 2. What the GL drew from the buffers.
            check_cells(env, check);

            // 3. Negative controls: the comparisons must be able to fail.
            negative_controls(env, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(aux_.size()), aux_.data());
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteProgram(program_);
        }

    private:
        static constexpr mgl::AttribLocation kLocPosition{0u}, kLocColor{1u};
        static constexpr auto kArray = mgl::BufferTarget::Array;
        static constexpr auto kCopyRead = mgl::BufferTarget::CopyRead;
        static constexpr auto kCopyWrite = mgl::BufferTarget::CopyWrite;
        using Access = mgl::MapBufferAccessMask;

        // ----------------------------------------------------------------------------------
        // Small helpers (GL calls stay visible in the cells; these only record or draw)
        // ----------------------------------------------------------------------------------
        void record(bool ok, std::string what, std::string detail = {}, std::string deviation_renderer = {}, std::string deviation_reason = {})
        {
            obs_.push_back({ok, std::move(what), std::move(detail), std::move(deviation_renderer), std::move(deviation_reason)});
        }

        void record_error(mgl::ErrorCode expected, std::string what)
        {
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == expected, std::move(what),
                   e == expected ? std::string{}
                                 : std::format("glGetError() = {}, expected {}", demo::error_name(e), demo::error_name(expected)));
        }

        void record_unmap_true(mgl::BufferTarget target, const std::string& cell)
        {
            const GLboolean r = mgl::glUnmapBuffer(target);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(r == GL_TRUE && e == mgl::ErrorCode::NoError, cell + ": glUnmapBuffer returns TRUE and raises no error",
                   std::format("returned {}, glGetError() = {}", static_cast<int>(r), demo::error_name(e)));
        }

        /// UnmapBuffer on a buffer that is not mapped: "returns FALSE, and an INVALID_OPERATION error is generated"
        /// (sec 2.10.3.1). The error and the return value are separate checks because Mesa's glthread (on by default
        /// for radeonsi) returns TRUE without looking, while still generating the error (docs/KNOWN_LIMITATIONS.md).
        void unmap_fails(mgl::BufferTarget target, const std::string& what)
        {
            const GLboolean r = mgl::glUnmapBuffer(target);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::InvalidOperation, what + ": INVALID_OPERATION",
                   std::format("glGetError() = {}", demo::error_name(e)));
            record(r == GL_FALSE, what + ": returns FALSE", std::format("returned {}", static_cast<int>(r)),
                   "radeonsi",
                   "Mesa glthread marshals glUnmapBuffer asynchronously and returns TRUE without checking that the buffer is mapped; "
                   "ES 3.0 sec 2.10.3.1 requires FALSE. Passes with mesa_glthread=false");
        }

        /// Maps [offset, offset+length) and requires success.
        std::uint8_t* map_ok(mgl::BufferTarget target, std::ptrdiff_t offset, std::ptrdiff_t length, Access access, const std::string& what)
        {
            void* p = mgl::glMapBufferRange(target, offset, length, access);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(p != nullptr && e == mgl::ErrorCode::NoError, what,
                   std::format("pointer {}, glGetError() = {}", p != nullptr ? "non-NULL" : "NULL", demo::error_name(e)));
            return static_cast<std::uint8_t*>(p);
        }

        /// A mapping that must fail: NULL result and the given error; the buffer must not be left mapped.
        void map_fails(mgl::BufferTarget target, std::ptrdiff_t offset, std::ptrdiff_t length, Access access,
                       mgl::ErrorCode expected, const std::string& what)
        {
            void* p = mgl::glMapBufferRange(target, offset, length, access);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(p == nullptr && e == expected, what + " -> NULL + " + std::string(demo::error_name(expected)),
                   std::format("pointer {}, glGetError() = {}", p != nullptr ? "non-NULL" : "NULL", demo::error_name(e)));
        }

        static void write_bytes(std::uint8_t* dst, std::ptrdiff_t offset, const Bytes& data)
        {
            if (dst != nullptr) std::memcpy(dst + offset, data.data(), data.size());
        }

        /// Maps the WHOLE buffer through COPY_READ_BUFFER with MAP_READ_BIT and compares every byte the shadow
        /// defines (ES 3.0 sec 2.10.3: the pointer is returned once all pending operations have completed).
        void verify(mgl::BufferId buf, const Shadow& shadow, const std::string& what)
        {
            mgl::glBindBuffer(kCopyRead, buf);
            GLint64 size = 0;
            mgl::glGetBufferParameteri64v(kCopyRead, mgl::BufferParameter::Size, &size);
            const void* p = mgl::glMapBufferRange(kCopyRead, 0, size, Access::Read);
            Bytes actual;
            if (p != nullptr)
            {
                actual.resize(static_cast<std::size_t>(size));
                std::memcpy(actual.data(), p, actual.size());
            }
            const GLboolean unmapped = p != nullptr ? mgl::glUnmapBuffer(kCopyRead) : GL_FALSE;
            mgl::glBindBuffer(kCopyRead, mgl::BufferId{0});
            const Shadow::Result r = shadow.compare(actual);
            record(p != nullptr && unmapped == GL_TRUE && r.mismatches == 0 && r.compared > 0,
                   std::format("{}: mapping the buffer with MAP_READ_BIT returns exactly the {} bytes the spec defines ({} undefined bytes not compared)",
                               what, r.compared, r.skipped),
                   p == nullptr ? std::string("MapBufferRange(READ) failed")
                                : std::format("{} mismatching bytes, first at offset {}", r.mismatches, r.first));
        }

        /// Draws quad slots [first, first+count) of `buf` into `cell`. `shift_y` moves the draw up in NDC.
        void draw_slots(const demo::Env& env, int cell, mgl::BufferId buf, int first, int count, float shift_y = 0.0f)
        {
            const int cw = env.width / kGrid, ch = env.height / kGrid;
            mgl::glViewport((cell % kGrid) * cw, (kGrid - 1 - cell / kGrid) * ch, cw, ch);
            mgl::glBindBuffer(kArray, buf);
            mgl::glVertexAttribPointer(kLocPosition, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glVertexAttribPointer(kLocColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glUniform2f(u_shift_, 0.0f, shift_y);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, first * static_cast<GLint>(kQuadVertices),
                              count * static_cast<GLsizei>(kQuadVertices));
        }

        /// Clears the inside of one cell (everything but its 2-pixel border) to the cell color.
        static void clear_cell(const demo::Env& env, int cell)
        {
            const int cw = env.width / kGrid, ch = env.height / kGrid;
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor((cell % kGrid) * cw + kBorderPx, (kGrid - 1 - cell / kGrid) * ch + kBorderPx, cw - 2 * kBorderPx, ch - 2 * kBorderPx);
            mgl::glClearColor(kClear[0] / 255.0f, kClear[1] / 255.0f, kClear[2] / 255.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        static std::string cell_name(int cell, const char* what) { return std::format("cell {} ({})", cell, what); }

        // ----------------------------------------------------------------------------------
        // [0] MAP_WRITE_BIT: the plain way to fill a store.
        // ----------------------------------------------------------------------------------
        void cell0_plain_write(const demo::Env& env)
        {
            const std::string c = cell_name(0, "MAP_WRITE_BIT");
            const Bytes data = scene_bytes({kRed, kGreen, kBlue, kYellow});
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[0]);
            mgl::glBufferData(kArray, kSceneBytes, nullptr, mgl::BufferUsage::DynamicDraw);  // contents undefined
            std::uint8_t* p = map_ok(kArray, 0, kSceneBytes, Access::Write, c + ": glMapBufferRange(0, 288, WRITE) returns a pointer");
            write_bytes(p, 0, data);
            record_unmap_true(kArray, c);
            shadow.set(0, data);
            verify(buf_[0], shadow, c);
            draw_slots(env, 0, buf_[0], 0, 4);
            model_[0] = Slots{kRed, kGreen, kBlue, kYellow};
        }

        // ----------------------------------------------------------------------------------
        // [1] MAP_INVALIDATE_BUFFER over the whole store: the "orphan and refill" idiom.
        // ----------------------------------------------------------------------------------
        void cell1_invalidate_buffer(const demo::Env& env)
        {
            const std::string c = cell_name(1, "WRITE | INVALIDATE_BUFFER");
            const Bytes old_data = scene_bytes({kGray, kGray, kGray, kGray});
            const Bytes data = scene_bytes({kCyan, kMagenta, kOrange, kWhite});
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[1]);
            mgl::glBufferData(kArray, kSceneBytes, old_data.data(), mgl::BufferUsage::StreamDraw);
            shadow.set(0, old_data);
            std::uint8_t* p = map_ok(kArray, 0, kSceneBytes, Access::Write | Access::InvalidateBuffer,
                                     c + ": glMapBufferRange(WRITE | INVALIDATE_BUFFER) returns a pointer");
            shadow.forget_all();  // the previous contents of the entire buffer may be discarded
            write_bytes(p, 0, data);
            record_unmap_true(kArray, c);
            shadow.set(0, data);  // ... except what was written afterwards
            verify(buf_[1], shadow, c);
            draw_slots(env, 1, buf_[1], 0, 4);
            model_[1] = Slots{kCyan, kMagenta, kOrange, kWhite};
        }

        // ----------------------------------------------------------------------------------
        // [2] READ | WRITE: read what is there, change it in place.
        // ----------------------------------------------------------------------------------
        void cell2_read_write(const demo::Env& env)
        {
            const std::string c = cell_name(2, "READ | WRITE");
            const std::array<Rgba8, 4> original = {kOrange, kCyan, kMagenta, kSlate};
            const Bytes initial = scene_bytes(original);
            Bytes expected = initial;
            for (std::size_t v = 0; v < 24; ++v)
                for (std::size_t k = 0; k < 3; ++k) expected[v * sizeof(Vertex) + offsetof(Vertex, r) + k] = static_cast<std::uint8_t>(255 - initial[v * sizeof(Vertex) + offsetof(Vertex, r) + k]);
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[2]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            std::uint8_t* p = map_ok(kArray, 0, kSceneBytes, Access::Read | Access::Write, c + ": glMapBufferRange(READ | WRITE) returns a pointer");
            if (p != nullptr)
            {
                record(std::memcmp(p, initial.data(), initial.size()) == 0,
                       c + ": reading through the mapped pointer returns the 288 bytes glBufferData stored");
                for (std::size_t v = 0; v < 24; ++v)  // invert r, g, b of every vertex in place; alpha stays
                    for (std::size_t k = 0; k < 3; ++k)
                    {
                        std::uint8_t& byte = p[v * sizeof(Vertex) + offsetof(Vertex, r) + k];
                        byte = static_cast<std::uint8_t>(255 - byte);
                    }
            }
            record_unmap_true(kArray, c);
            shadow.set(0, expected);
            verify(buf_[2], shadow, c);
            draw_slots(env, 2, buf_[2], 0, 4);
            model_[2] = Slots{invert(original[0]), invert(original[1]), invert(original[2]), invert(original[3])};
        }

        // ----------------------------------------------------------------------------------
        // [3] MAP_INVALIDATE_RANGE: only the mapped range may be discarded; the rest of the store is untouched.
        // ----------------------------------------------------------------------------------
        void cell3_invalidate_range(const demo::Env& env)
        {
            const std::string c = cell_name(3, "WRITE | INVALIDATE_RANGE");
            const Bytes initial = scene_bytes({kSlate, kOlive, kPlum, kTeal});
            const Bytes fresh = concat(quad_bytes(1, kYellow), quad_bytes(2, kGreen));
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[3]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            shadow.set(0, initial);
            std::uint8_t* p = map_ok(kArray, kQuadBytes, 2 * kQuadBytes, Access::Write | Access::InvalidateRange,
                                     c + ": glMapBufferRange(offset 72, length 144, WRITE | INVALIDATE_RANGE) returns a pointer");
            shadow.forget(kQuadBytes, 2 * kQuadBytes);
            write_bytes(p, 0, fresh);
            record_unmap_true(kArray, c);
            shadow.set(kQuadBytes, fresh);
            verify(buf_[3], shadow, c + " (quads 0 and 3, outside the range, keep their bytes)");
            draw_slots(env, 3, buf_[3], 0, 4);
            model_[3] = Slots{kSlate, kYellow, kGreen, kTeal};
        }

        // ----------------------------------------------------------------------------------
        // [4] MAP_FLUSH_EXPLICIT_BIT with every modified byte flushed.
        // ----------------------------------------------------------------------------------
        void cell4_flush_explicit(const demo::Env& env)
        {
            const std::string c = cell_name(4, "WRITE | FLUSH_EXPLICIT");
            const Bytes initial = scene_bytes({kGray, kGray, kGray, kGray});
            const Bytes data = scene_bytes({kBlue, kRed, kYellow, kCyan});
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[4]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            shadow.set(0, initial);
            std::uint8_t* p = map_ok(kArray, 0, kSceneBytes, Access::Write | Access::FlushExplicit,
                                     c + ": glMapBufferRange(WRITE | FLUSH_EXPLICIT) returns a pointer");
            shadow.forget(0, kSceneBytes);  // until flushed, nothing written is guaranteed to arrive
            write_bytes(p, 0, data);
            mgl::glFlushMappedBufferRange(kArray, 0, kQuadBytes);                   // quad 0
            mgl::glFlushMappedBufferRange(kArray, kQuadBytes, 2 * kQuadBytes);      // quads 1 and 2
            mgl::glFlushMappedBufferRange(kArray, 3 * kQuadBytes, kQuadBytes);      // quad 3
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::NoError, c + ": three glFlushMappedBufferRange calls (72 + 144 + 72 bytes) raise no error",
                   std::format("glGetError() = {}", demo::error_name(e)));
            record_unmap_true(kArray, c);
            shadow.set(0, data);  // everything written was flushed, so every byte arrived
            verify(buf_[4], shadow, c + " (all flushed ranges)");
            draw_slots(env, 4, buf_[4], 0, 4);
            model_[4] = Slots{kBlue, kRed, kYellow, kCyan};
        }

        // ----------------------------------------------------------------------------------
        // [5] Flush offsets are relative to the START OF THE MAPPING; a modified range that is not
        //     flushed has undefined contents, so it is neither asserted nor drawn.
        // ----------------------------------------------------------------------------------
        void cell5_flush_partial(const demo::Env& env)
        {
            const std::string c = cell_name(5, "FLUSH_EXPLICIT at offset 72, partial flush");
            const Bytes initial = scene_bytes({kGray, kGray, kGray, kGray});
            const Bytes q1 = quad_bytes(1, kMagenta), q2 = quad_bytes(2, kOrange), q3 = quad_bytes(3, kGreen);
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[5]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            shadow.set(0, initial);
            // The mapping covers quads 1..3 (buffer offsets 72..288); mapping offset 0 is buffer offset 72.
            std::uint8_t* p = map_ok(kArray, kQuadBytes, 3 * kQuadBytes, Access::Write | Access::FlushExplicit,
                                     c + ": glMapBufferRange(offset 72, length 216, WRITE | FLUSH_EXPLICIT) returns a pointer");
            shadow.forget(kQuadBytes, 3 * kQuadBytes);
            write_bytes(p, 0, q1);
            write_bytes(p, kQuadBytes, q2);       // modified but NOT flushed: undefined afterwards
            write_bytes(p, 2 * kQuadBytes, q3);
            mgl::glFlushMappedBufferRange(kArray, 0, kQuadBytes);                // mapping offset 0   = buffer offset 72  (quad 1)
            mgl::glFlushMappedBufferRange(kArray, 2 * kQuadBytes, kQuadBytes);   // mapping offset 144 = buffer offset 216 (quad 3)
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::NoError, c + ": flushing mapping ranges [0,72) and [144,216) raises no error",
                   std::format("glGetError() = {}", demo::error_name(e)));
            record_unmap_true(kArray, c);
            shadow.set(kQuadBytes, q1);          // flushed: guaranteed
            shadow.set(3 * kQuadBytes, q3);      // flushed: guaranteed
            // quad 2 stays "unknown" in the shadow. Quad 0 lies outside the mapping and keeps its gray.
            verify(buf_[5], shadow, c + " (quad 0 outside the mapping, quads 1 and 3 flushed; quad 2 not compared)");
            draw_slots(env, 5, buf_[5], 0, 2);   // quads 0, 1
            draw_slots(env, 5, buf_[5], 3, 1);   // quad 3 - quad 2 is deliberately not drawn
            model_[5] = Slots{kGray, kMagenta, std::nullopt, kGreen};
        }

        // ----------------------------------------------------------------------------------
        // [6] MAP_UNSYNCHRONIZED: write the next region while draws that read other regions are still pending.
        // ----------------------------------------------------------------------------------
        void cell6_unsynchronized(const demo::Env& env)
        {
            const std::string c = cell_name(6, "WRITE | INVALIDATE_RANGE | UNSYNCHRONIZED");
            const std::array<Rgba8, 4> colors = {kTeal, kYellow, kRed, kBlue};
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[6]);
            mgl::glBufferData(kArray, kSceneBytes, nullptr, mgl::BufferUsage::StreamDraw);
            for (int q = 0; q < 4; ++q)
            {
                const Bytes quad = quad_bytes(q, colors[static_cast<std::size_t>(q)]);
                // The ranges are disjoint from everything earlier draws read, so skipping synchronization is safe.
                std::uint8_t* p = map_ok(kArray, q * kQuadBytes, kQuadBytes,
                                         Access::Write | Access::InvalidateRange | Access::Unsynchronized,
                                         std::format("{}: quad {} mapped with the UNSYNCHRONIZED bit", c, q));
                write_bytes(p, 0, quad);
                record_unmap_true(kArray, std::format("{} quad {}", c, q));
                shadow.set(q * kQuadBytes, quad);
                draw_slots(env, 6, buf_[6], q, 1);  // pending while the next quad is written
            }
            verify(buf_[6], shadow, c);
            model_[6] = Slots{kTeal, kYellow, kRed, kBlue};
        }

        // ----------------------------------------------------------------------------------
        // [7] glCopyBufferSubData between two buffers through COPY_READ_BUFFER / COPY_WRITE_BUFFER.
        //     The source holds two quads with the geometry of the BOTTOM row; the copy puts them in
        //     slots 2 and 3 of the destination, which is drawn shifted up by 1.0 NDC.
        // ----------------------------------------------------------------------------------
        void cell7_copy_two_buffers(const demo::Env& env)
        {
            const std::string c = cell_name(7, "glCopyBufferSubData, two buffers");
            const Bytes src_data = concat(quad_bytes(0, kOrange), quad_bytes(1, kMagenta));
            const Bytes dst_initial = scene_bytes({kGray, kGray, kGray, kGray});
            Shadow src_shadow(static_cast<std::size_t>(2 * kQuadBytes)), dst_shadow;
            // A canary on ARRAY_BUFFER: copying through the COPY targets must not disturb it.
            mgl::glBindBuffer(kArray, buf_[0]);
            mgl::glBindBuffer(kCopyRead, aux_[0]);
            mgl::glBufferData(kCopyRead, 2 * kQuadBytes, src_data.data(), mgl::BufferUsage::StaticCopy);
            mgl::glBindBuffer(kCopyWrite, buf_[7]);
            mgl::glBufferData(kCopyWrite, kSceneBytes, dst_initial.data(), mgl::BufferUsage::DynamicDraw);
            src_shadow.set(0, src_data);
            dst_shadow.set(0, dst_initial);

            mgl::glCopyBufferSubData(kCopyRead, kCopyWrite, 0, 2 * kQuadBytes, 2 * kQuadBytes);  // src[0,144) -> dst[144,288)
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::NoError, c + ": glCopyBufferSubData(COPY_READ -> COPY_WRITE, 0, 144, 144) raises no error",
                   std::format("glGetError() = {}", demo::error_name(e)));
            dst_shadow.set(2 * kQuadBytes, src_data);
            record(demo::get_integer(mgl::GetParameter::ArrayBufferBinding) == static_cast<int>(buf_[0].value),
                   c + ": ARRAY_BUFFER_BINDING is untouched by the copy (that is what the COPY targets are for)");
            record(demo::get_integer(mgl::GetParameter::CopyReadBufferBinding) == static_cast<int>(aux_[0].value) &&
                       demo::get_integer(mgl::GetParameter::CopyWriteBufferBinding) == static_cast<int>(buf_[7].value),
                   c + ": COPY_READ_BUFFER_BINDING / COPY_WRITE_BUFFER_BINDING name the two buffers");
            mgl::glBindBuffer(kCopyRead, mgl::BufferId{0});
            mgl::glBindBuffer(kCopyWrite, mgl::BufferId{0});
            verify(aux_[0], src_shadow, c + " (source unchanged)");
            verify(buf_[7], dst_shadow, c + " (destination: slots 0-1 old, slots 2-3 copied)");
            draw_slots(env, 7, buf_[7], 0, 2);
            draw_slots(env, 7, buf_[7], 2, 2, 1.0f);  // the copied geometry is bottom-row geometry: lift it to the top row
            model_[7] = Slots{kGray, kGray, kOrange, kMagenta};
        }

        // ----------------------------------------------------------------------------------
        // [8] The same buffer bound to both copy targets: allowed when the ranges do not overlap.
        //     [0,144) and [144,288) touch but do not overlap.
        // ----------------------------------------------------------------------------------
        void cell8_copy_same_buffer(const demo::Env& env)
        {
            const std::string c = cell_name(8, "glCopyBufferSubData inside one buffer");
            const Bytes data = concat(concat(quad_bytes(0, kGreen), quad_bytes(1, kCyan)),
                                      concat(quad_bytes(0, kGray), quad_bytes(1, kGray)));
            Shadow shadow;
            mgl::glBindBuffer(kCopyRead, buf_[8]);
            mgl::glBufferData(kCopyRead, kSceneBytes, data.data(), mgl::BufferUsage::DynamicDraw);
            shadow.set(0, data);
            mgl::glBindBuffer(kCopyWrite, buf_[8]);
            mgl::glCopyBufferSubData(kCopyRead, kCopyWrite, 0, 2 * kQuadBytes, 2 * kQuadBytes);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::NoError, c + ": copying [0,144) to [144,288) of the same buffer (adjacent ranges) raises no error",
                   std::format("glGetError() = {}", demo::error_name(e)));
            Bytes expected = data;
            std::copy(data.begin(), data.begin() + 2 * kQuadBytes, expected.begin() + 2 * kQuadBytes);
            shadow.set(0, expected);
            mgl::glBindBuffer(kCopyRead, mgl::BufferId{0});
            mgl::glBindBuffer(kCopyWrite, mgl::BufferId{0});
            verify(buf_[8], shadow, c);
            draw_slots(env, 8, buf_[8], 0, 2);
            draw_slots(env, 8, buf_[8], 2, 2, 1.0f);
            model_[8] = Slots{kGreen, kCyan, kGreen, kCyan};
        }

        // ----------------------------------------------------------------------------------
        // [9] Calls the spec makes fail: nothing may be mapped, flushed or changed afterwards.
        // ----------------------------------------------------------------------------------
        void cell9_failed_calls(const demo::Env& env)
        {
            const std::string c = cell_name(9, "failed Map / Flush / Unmap calls");
            const Bytes data = scene_bytes({kYellow, kBlue, kOrange, kGreen});
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[9]);
            mgl::glBufferData(kArray, kSceneBytes, data.data(), mgl::BufferUsage::StaticDraw);
            shadow.set(0, data);
            const auto E = mgl::ErrorCode::InvalidOperation;
            const auto V = mgl::ErrorCode::InvalidValue;

            // ES 3.0 sec 2.10.3, INVALID_OPERATION conditions.
            map_fails(kArray, 0, kSceneBytes, Access::Read | Access::InvalidateRange, E, c + ": READ | INVALIDATE_RANGE");
            map_fails(kArray, 0, kSceneBytes, Access::Read | Access::InvalidateBuffer, E, c + ": READ | INVALIDATE_BUFFER");
            map_fails(kArray, 0, kSceneBytes, Access::Read | Access::Unsynchronized, E, c + ": READ | UNSYNCHRONIZED");
            map_fails(kArray, 0, kSceneBytes, Access::Read | Access::FlushExplicit, E, c + ": READ | FLUSH_EXPLICIT (no WRITE)");
            map_fails(kArray, 0, kSceneBytes, Access::FlushExplicit, E, c + ": FLUSH_EXPLICIT alone");
            map_fails(kArray, 0, kSceneBytes, Access{}, E, c + ": access = 0 (neither READ nor WRITE)");
            map_fails(kArray, 0, kSceneBytes, Access::Unsynchronized, E, c + ": UNSYNCHRONIZED alone (neither READ nor WRITE)");
            map_fails(kArray, 0, 0, Access::Write, E, c + ": length 0");
            // INVALID_VALUE conditions.
            map_fails(kArray, -1, 8, Access::Write, V, c + ": negative offset");
            map_fails(kArray, 0, -1, Access::Write, V, c + ": negative length");
            map_fails(kArray, 0, kSceneBytes + 1, Access::Write, V, c + ": offset + length = 289 > BUFFER_SIZE 288");
            map_fails(kArray, kSceneBytes, 1, Access::Write, V, c + ": offset 288, length 1");
            map_fails(kArray, 1, kSceneBytes, Access::Write, V, c + ": offset 1, length 288");
            // No buffer bound: "client attempts to modify or query buffer object state for a target bound to zero" (sec 2.10.1).
            mgl::glBindBuffer(kArray, mgl::BufferId{0});
            map_fails(kArray, 0, 8, Access::Write, E, c + ": no buffer bound to the target");
            mgl::glFlushMappedBufferRange(kArray, 0, 8);
            record_error(E, c + ": glFlushMappedBufferRange with no buffer bound -> INVALID_OPERATION");
            unmap_fails(kArray, c + ": glUnmapBuffer with no buffer bound");
            GLint zero_query = -1;
            mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Mapped, &zero_query);
            record_error(E, c + ": glGetBufferParameteriv with no buffer bound -> INVALID_OPERATION");
            void* zero_ptr = nullptr;
            mgl::glGetBufferPointerv(kArray, mgl::BufferPointerParameter::MapPointer, &zero_ptr);
            record_error(E, c + ": glGetBufferPointerv with no buffer bound -> INVALID_OPERATION");
            mgl::glBindBuffer(kArray, buf_[9]);

            GLint mapped = -1;
            mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Mapped, &mapped);
            record(mapped == GL_FALSE, c + ": none of the failed glMapBufferRange calls left the buffer mapped (BUFFER_MAPPED FALSE)");

            // Unmapped buffer: Flush and Unmap are invalid.
            mgl::glFlushMappedBufferRange(kArray, 0, 8);
            record_error(E, c + ": glFlushMappedBufferRange on an unmapped buffer -> INVALID_OPERATION");
            unmap_fails(kArray, c + ": glUnmapBuffer on an unmapped buffer");

            // Already mapped: a second map fails and the first mapping survives unchanged.
            std::uint8_t* p1 = map_ok(kArray, kQuadBytes, 2 * kQuadBytes, Access::Read | Access::Write, c + ": first mapping (offset 72, length 144, READ | WRITE)");
            map_fails(kArray, 0, 8, Access::Read, E, c + ": mapping a buffer that is already mapped");
            void* queried = nullptr;
            mgl::glGetBufferPointerv(kArray, mgl::BufferPointerParameter::MapPointer, &queried);
            GLint64 off = -1, len = -1;
            mgl::glGetBufferParameteri64v(kArray, mgl::BufferParameter::MapOffset, &off);
            mgl::glGetBufferParameteri64v(kArray, mgl::BufferParameter::MapLength, &len);
            record(queried == p1 && off == kQuadBytes && len == 2 * kQuadBytes,
                   c + ": the failed second map did not disturb the first (same pointer, MAP_OFFSET 72, MAP_LENGTH 144)",
                   std::format("pointer {}, offset {}, length {}", queried == p1 ? "same" : "different", off, len));
            mgl::glFlushMappedBufferRange(kArray, 0, 8);
            record_error(E, c + ": glFlushMappedBufferRange on a mapping without FLUSH_EXPLICIT -> INVALID_OPERATION");
            record_unmap_true(kArray, c);

            // Flush range errors need a FLUSH_EXPLICIT mapping: offset 72, length 144.
            std::uint8_t* p2 = map_ok(kArray, kQuadBytes, 2 * kQuadBytes, Access::Write | Access::FlushExplicit, c + ": mapping (offset 72, length 144, WRITE | FLUSH_EXPLICIT)");
            mgl::glFlushMappedBufferRange(kArray, -1, 8);
            record_error(V, c + ": glFlushMappedBufferRange with a negative offset -> INVALID_VALUE");
            mgl::glFlushMappedBufferRange(kArray, 0, -1);
            record_error(V, c + ": glFlushMappedBufferRange with a negative length -> INVALID_VALUE");
            mgl::glFlushMappedBufferRange(kArray, 100, 100);
            record_error(V, c + ": glFlushMappedBufferRange(100, 100) exceeds the 144-byte mapping -> INVALID_VALUE");
            if (p2 != nullptr) std::memcpy(p2, data.data() + kQuadBytes, static_cast<std::size_t>(2 * kQuadBytes));  // rewrite the same bytes
            mgl::glFlushMappedBufferRange(kArray, 0, 2 * kQuadBytes);
            record_error(mgl::ErrorCode::NoError, c + ": flushing exactly the whole mapping (0, 144) is valid");
            record_unmap_true(kArray, c);

            verify(buf_[9], shadow, c + " (data unchanged by every failed call)");
            draw_slots(env, 9, buf_[9], 0, 4);
            model_[9] = Slots{kYellow, kBlue, kOrange, kGreen};
        }

        // ----------------------------------------------------------------------------------
        // [10] glBufferData on a mapped buffer unmaps it implicitly (sec 2.10.2, 2.10.3.1).
        // ----------------------------------------------------------------------------------
        void cell10_buffer_data_unmaps(const demo::Env& env)
        {
            const std::string c = cell_name(10, "glBufferData unmaps implicitly");
            const Bytes initial = scene_bytes({kGray, kGray, kGray, kGray});
            const Bytes data = scene_bytes({kMagenta, kTeal, kWhite, kRed});
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[10]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::StaticDraw);
            map_ok(kArray, 0, kSceneBytes, Access::Write, c + ": mapped before the re-specification");
            GLint mapped = -1;
            mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Mapped, &mapped);
            record(mapped == GL_TRUE, c + ": BUFFER_MAPPED is TRUE while mapped");
            mgl::glBufferData(kArray, kSceneBytes, data.data(), mgl::BufferUsage::DynamicDraw);  // "as though UnmapBuffer was executed"
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::NoError, c + ": glBufferData on a mapped buffer is not an error", std::format("glGetError() = {}", demo::error_name(e)));
            check_unmapped_state(kArray, c + " after glBufferData");
            unmap_fails(kArray, c + ": glUnmapBuffer afterwards (nothing is mapped any more)");
            shadow.set(0, data);
            verify(buf_[10], shadow, c);
            draw_slots(env, 10, buf_[10], 0, 4);
            model_[10] = Slots{kMagenta, kTeal, kWhite, kRed};
        }

        // ----------------------------------------------------------------------------------
        // [11] Mapping state belongs to the buffer object, not to a target.
        // ----------------------------------------------------------------------------------
        void cell11_two_bindings(const demo::Env& env)
        {
            const std::string c = cell_name(11, "one buffer, two bindings");
            const Bytes data = scene_bytes({kOlive, kPlum, kCyan, kOrange});
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[11]);
            mgl::glBindBuffer(kCopyWrite, buf_[11]);
            mgl::glBufferData(kArray, kSceneBytes, nullptr, mgl::BufferUsage::DynamicDraw);
            std::uint8_t* p = map_ok(kCopyWrite, 0, kSceneBytes, Access::Write | Access::InvalidateBuffer,
                                     c + ": mapped through COPY_WRITE_BUFFER");
            GLint mapped = -1;
            mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Mapped, &mapped);
            void* queried = nullptr;
            mgl::glGetBufferPointerv(kArray, mgl::BufferPointerParameter::MapPointer, &queried);
            record(mapped == GL_TRUE && queried == p && p != nullptr,
                   c + ": through ARRAY_BUFFER the same buffer reports BUFFER_MAPPED TRUE and the same MAP_POINTER",
                   std::format("mapped {}, pointer {}", mapped, queried == p ? "same" : "different"));
            write_bytes(p, 0, data);
            record_unmap_true(kArray, c + " (unmapped through ARRAY_BUFFER)");
            mgl::glGetBufferParameteriv(kCopyWrite, mgl::BufferParameter::Mapped, &mapped);
            record(mapped == GL_FALSE, c + ": ... and COPY_WRITE_BUFFER now reports BUFFER_MAPPED FALSE");
            mgl::glBindBuffer(kCopyWrite, mgl::BufferId{0});
            shadow.set(0, data);
            verify(buf_[11], shadow, c);
            draw_slots(env, 11, buf_[11], 0, 4);
            model_[11] = Slots{kOlive, kPlum, kCyan, kOrange};
        }

        // ----------------------------------------------------------------------------------
        // [12] MAP_INVALIDATE_BUFFER on a sub-range: everything outside it becomes undefined too.
        // ----------------------------------------------------------------------------------
        void cell12_invalidate_buffer_range(const demo::Env& env)
        {
            const std::string c = cell_name(12, "WRITE | INVALIDATE_BUFFER on quads 1-2");
            const Bytes initial = scene_bytes({kGray, kGray, kGray, kGray});
            const Bytes fresh = concat(quad_bytes(1, kBlue), quad_bytes(2, kYellow));
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[12]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            std::uint8_t* p = map_ok(kArray, kQuadBytes, 2 * kQuadBytes, Access::Write | Access::InvalidateBuffer,
                                     c + ": glMapBufferRange(offset 72, length 144, WRITE | INVALIDATE_BUFFER) returns a pointer");
            shadow.forget_all();
            write_bytes(p, 0, fresh);
            record_unmap_true(kArray, c);
            shadow.set(kQuadBytes, fresh);
            verify(buf_[12], shadow, c + " (only the written range; quads 0 and 3 are undefined and not compared)");
            draw_slots(env, 12, buf_[12], 1, 2);  // quads 0 and 3 are undefined: not drawn
            model_[12] = Slots{std::nullopt, kBlue, kYellow, std::nullopt};
        }

        // ----------------------------------------------------------------------------------
        // [13] glCopyBufferSubData errors (ES 3.0 sec 2.10.5): the destination must not change.
        // ----------------------------------------------------------------------------------
        void cell13_failed_copies(const demo::Env& env)
        {
            const std::string c = cell_name(13, "rejected glCopyBufferSubData calls");
            const Bytes a_data = scene_bytes({kRed, kRed, kRed, kRed});
            const Bytes b_data = scene_bytes({kCyan, kOrange, kMagenta, kWhite});
            Shadow a_shadow, b_shadow;
            const auto E = mgl::ErrorCode::InvalidOperation;
            const auto V = mgl::ErrorCode::InvalidValue;
            mgl::glBindBuffer(kCopyRead, aux_[1]);
            mgl::glBufferData(kCopyRead, kSceneBytes, a_data.data(), mgl::BufferUsage::StaticCopy);
            mgl::glBindBuffer(kCopyWrite, buf_[13]);
            mgl::glBufferData(kCopyWrite, kSceneBytes, b_data.data(), mgl::BufferUsage::DynamicDraw);
            a_shadow.set(0, a_data);
            b_shadow.set(0, b_data);

            auto copy_fails = [&](std::ptrdiff_t roff, std::ptrdiff_t woff, std::ptrdiff_t size, mgl::ErrorCode expected, const char* what) {
                mgl::glCopyBufferSubData(kCopyRead, kCopyWrite, roff, woff, size);
                record_error(expected, std::format("{}: {} -> {}", c, what, demo::error_name(expected)));
            };

            // Same buffer bound to both targets: overlapping ranges are INVALID_VALUE.
            mgl::glBindBuffer(kCopyWrite, aux_[1]);
            copy_fails(0, kQuadBytes, 2 * kQuadBytes, V, "same buffer, [0,144) -> [72,216) overlap");
            copy_fails(0, 0, kQuadBytes, V, "same buffer, identical ranges [0,72) -> [0,72)");
            copy_fails(0, 143, 144, V, "same buffer, [0,144) -> [143,287) overlap by one byte");
            mgl::glBindBuffer(kCopyWrite, buf_[13]);

            // Two buffers: range and sign errors.
            copy_fails(200, 0, 100, V, "read range [200,300) exceeds the 288-byte source");
            copy_fails(0, 200, 100, V, "write range [200,300) exceeds the 288-byte destination");
            copy_fails(-1, 0, 8, V, "negative read offset");
            copy_fails(0, -1, 8, V, "negative write offset");
            copy_fails(0, 0, -1, V, "negative size");

            // Mapped buffers.
            map_ok(kCopyWrite, 0, kSceneBytes, Access::Read, c + ": destination mapped for the next attempt");
            copy_fails(0, 0, 8, E, "destination buffer is mapped");
            record_unmap_true(kCopyWrite, c);
            map_ok(kCopyRead, 0, kSceneBytes, Access::Read, c + ": source mapped for the next attempt");
            copy_fails(0, 0, 8, E, "source buffer is mapped");
            record_unmap_true(kCopyRead, c);

            // Zero bound.
            mgl::glBindBuffer(kCopyRead, mgl::BufferId{0});
            copy_fails(0, 0, 8, E, "no buffer bound to COPY_READ_BUFFER");
            mgl::glBindBuffer(kCopyRead, aux_[1]);
            mgl::glBindBuffer(kCopyWrite, mgl::BufferId{0});
            copy_fails(0, 0, 8, E, "no buffer bound to COPY_WRITE_BUFFER");

            mgl::glBindBuffer(kCopyRead, mgl::BufferId{0});
            verify(aux_[1], a_shadow, c + " (source)");
            verify(buf_[13], b_shadow, c + " (destination unchanged by every rejected copy)");
            draw_slots(env, 13, buf_[13], 0, 4);
            model_[13] = Slots{kCyan, kOrange, kMagenta, kWhite};
        }

        // ----------------------------------------------------------------------------------
        // [14] glBufferSubData on a mapped range is INVALID_OPERATION (sec 2.10.2). A draw that reads the
        //      mapped buffer is only OBSERVED: sec 2.10.3.2 says such commands "may generate" the error,
        //      so no outcome is asserted. The cell is cleared and redrawn after the buffer is unmapped.
        // ----------------------------------------------------------------------------------
        void cell14_subdata_while_mapped(const demo::Env& env)
        {
            const std::string c = cell_name(14, "glBufferSubData while mapped");
            const Bytes initial = scene_bytes({kBlue, kGray, kGray, kGreen});
            const Bytes fresh = concat(quad_bytes(1, kOrange), quad_bytes(2, kMagenta));
            const Bytes patch = quad_bytes(3, kYellow);
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[14]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            shadow.set(0, initial);
            std::uint8_t* p = map_ok(kArray, kQuadBytes, 2 * kQuadBytes, Access::Write, c + ": mapped quads 1-2 (offset 72, length 144)");

            const Bytes junk(16, 0xEE);
            mgl::glBufferSubData(kArray, 100, 16, junk.data());  // [100,116) lies inside the mapped [72,216)
            record_error(mgl::ErrorCode::InvalidOperation, c + ": glBufferSubData over a mapped range -> INVALID_OPERATION");

            // Observation only: draw while the vertex buffer is mapped.
            draw_slots(env, 14, buf_[14], 0, 4);
            const mgl::ErrorCode mapped_draw = demo::drain_gl_errors();
            mapped_draw_note_ = std::format(
                "drawing from a buffer that is mapped: glGetError() = {} (ES 3.0 sec 2.10.3.2: the error is optional, so the demo asserts nothing; "
                "the cell is cleared and redrawn from the unmapped buffer below)", demo::error_name(mapped_draw));
            clear_cell(env, 14);

            write_bytes(p, 0, fresh);
            record_unmap_true(kArray, c);
            shadow.set(kQuadBytes, fresh);
            verify(buf_[14], shadow, c + " (the rejected glBufferSubData wrote nothing)");

            // Unmapped again, the buffer accepts glBufferSubData.
            mgl::glBufferSubData(kArray, 3 * kQuadBytes, kQuadBytes, patch.data());
            record_error(mgl::ErrorCode::NoError, c + ": glBufferSubData on the unmapped buffer works");
            shadow.set(3 * kQuadBytes, patch);
            verify(buf_[14], shadow, c + " (after the valid glBufferSubData)");
            draw_slots(env, 14, buf_[14], 0, 4);
            model_[14] = Slots{kBlue, kOrange, kMagenta, kYellow};
        }

        // ----------------------------------------------------------------------------------
        // [15] A mapping that starts in the middle: the pointer addresses buffer offset 144.
        // ----------------------------------------------------------------------------------
        void cell15_read_write_offset(const demo::Env& env)
        {
            const std::string c = cell_name(15, "READ | WRITE at offset 144");
            const Bytes initial = scene_bytes({kSlate, kTeal, kRed, kYellow});
            const Bytes top_row = concat(quad_bytes(2, kRed), quad_bytes(3, kYellow));
            const Bytes swapped = concat(quad_bytes(2, kYellow), quad_bytes(3, kRed));
            Shadow shadow;
            mgl::glBindBuffer(kArray, buf_[15]);
            mgl::glBufferData(kArray, kSceneBytes, initial.data(), mgl::BufferUsage::DynamicDraw);
            std::uint8_t* p = map_ok(kArray, 2 * kQuadBytes, 2 * kQuadBytes, Access::Read | Access::Write,
                                     c + ": glMapBufferRange(offset 144, length 144, READ | WRITE) returns a pointer");
            if (p != nullptr)
            {
                record(std::memcmp(p, top_row.data(), top_row.size()) == 0,
                       c + ": mapped byte 0 is buffer byte 144: the pointer reads the two top quads");
                // Swap the colors of the two quads through the pointer (read each color, write the other's).
                const Rgba8 first = color_at(top_row, 0), second = color_at(top_row, 6);
                for (std::size_t v = 0; v < 12; ++v)
                {
                    const Rgba8 c_new = v < 6 ? second : first;
                    std::memcpy(p + v * sizeof(Vertex) + offsetof(Vertex, r), c_new.data(), 4);
                }
            }
            record_unmap_true(kArray, c);
            shadow.set(0, initial);
            shadow.set(2 * kQuadBytes, swapped);
            verify(buf_[15], shadow, c);
            draw_slots(env, 15, buf_[15], 0, 4);
            model_[15] = Slots{kSlate, kTeal, kYellow, kRed};
        }

        // ----------------------------------------------------------------------------------
        // Queries
        // ----------------------------------------------------------------------------------
        /// Table 2.8 values: what every buffer reports when nothing is mapped.
        void check_unmapped_state(mgl::BufferTarget target, const std::string& what)
        {
            GLint mapped = -1, flags = -1;
            GLint64 off = -1, len = -1;
            void* ptr = &mapped;  // anything but NULL
            mgl::glGetBufferParameteriv(target, mgl::BufferParameter::Mapped, &mapped);
            mgl::glGetBufferParameteriv(target, mgl::BufferParameter::AccessFlags, &flags);
            mgl::glGetBufferParameteri64v(target, mgl::BufferParameter::MapOffset, &off);
            mgl::glGetBufferParameteri64v(target, mgl::BufferParameter::MapLength, &len);
            mgl::glGetBufferPointerv(target, mgl::BufferPointerParameter::MapPointer, &ptr);
            record(mapped == GL_FALSE && flags == 0 && off == 0 && len == 0 && ptr == nullptr,
                   what + ": BUFFER_MAPPED FALSE, BUFFER_ACCESS_FLAGS 0, BUFFER_MAP_OFFSET 0, BUFFER_MAP_LENGTH 0, BUFFER_MAP_POINTER NULL (Table 2.8)",
                   std::format("mapped {}, flags {}, offset {}, length {}, pointer {}", mapped, flags, off, len, ptr == nullptr ? "NULL" : "non-NULL"));
        }

        // ----------------------------------------------------------------------------------
        // All 64 combinations of the six access flags against the rule table of sec 2.10.3.
        // ----------------------------------------------------------------------------------
        void flag_matrix()
        {
            struct Flag { Access bit; const char* name; };
            static constexpr std::array<Flag, 6> flags = {{
                {Access::Read, "READ"}, {Access::Write, "WRITE"}, {Access::InvalidateRange, "INVALIDATE_RANGE"},
                {Access::InvalidateBuffer, "INVALIDATE_BUFFER"}, {Access::FlushExplicit, "FLUSH_EXPLICIT"},
                {Access::Unsynchronized, "UNSYNCHRONIZED"},
            }};
            const Bytes zeros(static_cast<std::size_t>(kSceneBytes), 0);
            mgl::glBindBuffer(kArray, aux_[0]);
            mgl::glBufferData(kArray, kSceneBytes, zeros.data(), mgl::BufferUsage::DynamicDraw);

            constexpr std::ptrdiff_t offset = 24, length = 100;  // neither is a multiple of anything useful on purpose
            int valid_count = 0, invalid_count = 0;
            std::string bad_outcome, bad_state;
            for (unsigned m = 0; m < 64; ++m)
            {
                Access mask{};
                std::string name;
                for (std::size_t i = 0; i < flags.size(); ++i)
                    if ((m >> i) & 1u)
                    {
                        mask = mask | flags[i].bit;
                        name += (name.empty() ? "" : "|") + std::string(flags[i].name);
                    }
                if (name.empty()) name = "0";
                const bool rd = m & 1u, wr = m & 2u, ir = m & 4u, ib = m & 8u, fe = m & 16u, us = m & 32u;
                // The spec's INVALID_OPERATION table, verbatim: neither READ nor WRITE; READ with any of
                // INVALIDATE_RANGE / INVALIDATE_BUFFER / UNSYNCHRONIZED; FLUSH_EXPLICIT without WRITE.
                const bool valid = (rd || wr) && !(rd && (ir || ib || us)) && !(fe && !wr);

                void* p = mgl::glMapBufferRange(kArray, offset, length, mask);
                const mgl::ErrorCode e = demo::drain_gl_errors();
                if (!valid)
                {
                    ++invalid_count;
                    GLint mapped = -1;
                    mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Mapped, &mapped);
                    if (p != nullptr || e != mgl::ErrorCode::InvalidOperation || mapped != GL_FALSE)
                        bad_outcome += std::format("{} (ptr {}, {}) ", name, p != nullptr ? "non-NULL" : "NULL", demo::error_name(e));
                    continue;
                }
                ++valid_count;
                if (p == nullptr || e != mgl::ErrorCode::NoError)
                {
                    bad_outcome += std::format("{} (ptr {}, {}) ", name, p != nullptr ? "non-NULL" : "NULL", demo::error_name(e));
                    continue;
                }
                GLint mapped = -1, got_flags = -1, off32 = -1, len32 = -1, size32 = -1;
                GLint64 off64 = -1, len64 = -1, size64 = -1;
                void* queried = nullptr;
                mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Mapped, &mapped);
                mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::AccessFlags, &got_flags);
                mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::MapOffset, &off32);
                mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::MapLength, &len32);
                mgl::glGetBufferParameteriv(kArray, mgl::BufferParameter::Size, &size32);
                mgl::glGetBufferParameteri64v(kArray, mgl::BufferParameter::MapOffset, &off64);
                mgl::glGetBufferParameteri64v(kArray, mgl::BufferParameter::MapLength, &len64);
                mgl::glGetBufferParameteri64v(kArray, mgl::BufferParameter::Size, &size64);
                mgl::glGetBufferPointerv(kArray, mgl::BufferPointerParameter::MapPointer, &queried);
                const bool state_ok = mapped == GL_TRUE && static_cast<GLbitfield>(got_flags) == static_cast<GLbitfield>(mask) &&
                                      off32 == offset && len32 == length && size32 == kSceneBytes && off64 == offset &&
                                      len64 == length && size64 == kSceneBytes && queried == p;
                if (!state_ok)
                    bad_state += std::format("{} (mapped {}, flags 0x{:X}, offset {}/{}, length {}/{}, size {}/{}, pointer {}) ", name, mapped,
                                             static_cast<unsigned>(got_flags), off32, off64, len32, len64, size32, size64,
                                             queried == p ? "same" : "different");
                const GLboolean un = mgl::glUnmapBuffer(kArray);
                const mgl::ErrorCode ue = demo::drain_gl_errors();
                if (un != GL_TRUE || ue != mgl::ErrorCode::NoError) bad_outcome += std::format("{} (unmap {}, {}) ", name, static_cast<int>(un), demo::error_name(ue));
            }
            record(bad_outcome.empty(),
                   std::format("matrix: all 64 combinations of the six MAP_*_BIT flags: {} valid ones map and unmap, {} invalid ones return NULL with INVALID_OPERATION (ES 3.0 sec 2.10.3)",
                               valid_count, invalid_count),
                   bad_outcome);
            record(bad_state.empty(),
                   std::format("matrix: each of the {} valid mappings reports BUFFER_MAPPED TRUE, BUFFER_ACCESS_FLAGS = the requested bits, BUFFER_MAP_OFFSET 24, BUFFER_MAP_LENGTH 100 "
                               "(glGetBufferParameteriv and glGetBufferParameteri64v), BUFFER_SIZE 288 and MAP_POINTER = the returned pointer",
                               valid_count),
                   bad_state);
            check_unmapped_state(kArray, "matrix: after the last unmap");
        }

        // ----------------------------------------------------------------------------------
        // Deleting a mapped buffer unmaps it implicitly (sec 2.10).
        // ----------------------------------------------------------------------------------
        void delete_while_mapped()
        {
            const std::string c = "delete: ";
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(kArray, b);
            mgl::glBufferData(kArray, 64, nullptr, mgl::BufferUsage::StaticDraw);
            map_ok(kArray, 0, 64, Access::Write, c + "a scratch buffer is mapped");
            mgl::glDeleteBuffers(1, &b);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == mgl::ErrorCode::NoError && !mgl::glIsBuffer(b),
                   c + "glDeleteBuffers on a mapped buffer is not an error and the name is no longer a buffer",
                   std::format("glGetError() = {}", demo::error_name(e)));
            record(demo::get_integer(mgl::GetParameter::ArrayBufferBinding) == 0, c + "the deletion reset the ARRAY_BUFFER binding to 0");
        }

        // ----------------------------------------------------------------------------------
        // Pixels
        // ----------------------------------------------------------------------------------
        struct Rect
        {
            int x0, y0, x1, y1;  // pixel bounds of a quad inside the whole surface, [x0,x1) x [y0,y1)
        };

        static Rect quad_rect(const demo::Env& env, int cell, int position)
        {
            const int cw = env.width / kGrid, ch = env.height / kGrid;
            const int ox = (cell % kGrid) * cw, oy = (kGrid - 1 - cell / kGrid) * ch;
            // Quad centers are +-0.5 NDC with half-width 0.4: [0.1, 0.9] of the half cell on each side.
            const float fx0 = position % 2 == 0 ? 0.05f : 0.55f, fx1 = position % 2 == 0 ? 0.45f : 0.95f;
            const float fy0 = position / 2 == 0 ? 0.05f : 0.55f, fy1 = position / 2 == 0 ? 0.45f : 0.95f;
            return {ox + static_cast<int>(fx0 * static_cast<float>(cw)), oy + static_cast<int>(fy0 * static_cast<float>(ch)),
                    ox + static_cast<int>(fx1 * static_cast<float>(cw)), oy + static_cast<int>(fy1 * static_cast<float>(ch))};
        }

        /// Probe points of one quad: its center and four points 6 px inside its corners (never on a quad edge).
        static std::array<std::pair<int, int>, 5> quad_probes(const Rect& r)
        {
            constexpr int inset = 6;
            return {{{(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2},
                     {r.x0 + inset, r.y0 + inset},
                     {r.x1 - 1 - inset, r.y0 + inset},
                     {r.x0 + inset, r.y1 - 1 - inset},
                     {r.x1 - 1 - inset, r.y1 - 1 - inset}}};
        }

        static bool near(Rgba8 got, Rgba8 want, int tol)
        {
            for (std::size_t i = 0; i < 4; ++i)
                if (std::abs(int{got[i]} - int{want[i]}) > tol) return false;
            return true;
        }

        /// Tolerance 1: the byte becomes c/255 as a normalized attribute, is interpolated (all three vertices
        /// carry the same value, so the result is that value up to float rounding) and is quantized back to
        /// 8 bits; ES 3.0 sec 2.1.2 allows an error of 1/255 in such conversions.
        static constexpr int kTolerance = 1;

        void check_cells(const demo::Env& env, demo::Check& check)
        {
            static constexpr std::array<const char*, kCells> labels = {
                "MAP_WRITE", "INVALIDATE_BUFFER", "READ | WRITE in place", "INVALIDATE_RANGE", "FLUSH_EXPLICIT", "partial flush",
                "UNSYNCHRONIZED", "copy, two buffers", "copy, same buffer", "failed Map/Flush/Unmap", "glBufferData unmaps",
                "two bindings", "INVALIDATE_BUFFER range", "rejected copies", "SubData while mapped", "READ | WRITE offset"};
            const int cw = env.width / kGrid, ch = env.height / kGrid;
            for (int cell = 0; cell < kCells; ++cell)
            {
                int probes = 0;
                std::string bad;
                for (int pos = 0; pos < 4; ++pos)
                {
                    const Rect r = quad_rect(env, cell, pos);
                    const auto& want = model_[static_cast<std::size_t>(cell)][static_cast<std::size_t>(pos)];
                    for (const auto& [x, y] : quad_probes(r))
                    {
                        const Rgba8 got = demo::read_pixel(x, y);
                        ++probes;
                        if (!near(got, want.value_or(kClear), want ? kTolerance : 0))
                            bad += std::format("quad {} at ({},{}) is {} expected {}; ", pos, x, y, demo::format_rgba(got), demo::format_rgba(want.value_or(kClear)));
                    }
                }
                // The cell center lies in the gap between the quads, two corner pixels inside the border outside all
                // quads, and the border pixel itself must still be the grid-line color.
                const int ox = (cell % kGrid) * cw, oy = (kGrid - 1 - cell / kGrid) * ch;
                const std::array<std::pair<std::pair<int, int>, Rgba8>, 4> background = {{
                    {{ox + cw / 2, oy + ch / 2}, kClear}, {{ox + kBorderPx, oy + kBorderPx}, kClear},
                    {{ox + cw - 1 - kBorderPx, oy + ch - 1 - kBorderPx}, kClear}, {{ox, oy}, kBorder}}};
                for (const auto& [xy, want] : background)
                {
                    ++probes;
                    const Rgba8 got = demo::read_pixel(xy.first, xy.second);
                    if (got != want) bad += std::format("background at ({},{}) is {} expected {}; ", xy.first, xy.second, demo::format_rgba(got), demo::format_rgba(want));
                }
                check.expect(bad.empty(), std::format("cell {} ({}): {} probes match the CPU model (quad colors = the written bytes; absent quads, gaps and borders = background)", cell, labels[static_cast<std::size_t>(cell)], probes), bad);
            }
        }

        // ----------------------------------------------------------------------------------
        // Negative controls
        // ----------------------------------------------------------------------------------
        void negative_controls(const demo::Env& env, demo::Check& check)
        {
            // (a) Pixel comparison: write a wrong color into quad 3 of cell 0 through a mapping, redraw, and require
            //     the probe to see it, then restore.
            const Rect r3 = quad_rect(env, 0, 3), r0 = quad_rect(env, 0, 0);
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);  // render() left both unbound
            mgl::glBindBuffer(kArray, buf_[0]);
            void* p = mgl::glMapBufferRange(kArray, 3 * kQuadBytes, kQuadBytes, Access::Write | Access::InvalidateRange);
            write_bytes(static_cast<std::uint8_t*>(p), 0, quad_bytes(3, kMagenta));
            (void)mgl::glUnmapBuffer(kArray);
            draw_slots(env, 0, buf_[0], 0, 4);
            const Rgba8 got3 = demo::read_pixel(quad_probes(r3)[0].first, quad_probes(r3)[0].second);
            const Rgba8 got0 = demo::read_pixel(quad_probes(r0)[0].first, quad_probes(r0)[0].second);
            check.expect(p != nullptr && !near(got3, kYellow, kTolerance) && near(got3, kMagenta, kTolerance),
                         "negative control: a wrong color written through a mapping makes the quad-3 probe of cell 0 fail its yellow expectation",
                         std::format("pixel {}", demo::format_rgba(got3)));
            check.expect(near(got0, kRed, kTolerance), "negative control: the neighbouring quad 0 of cell 0 is unaffected", std::format("pixel {}", demo::format_rgba(got0)));
            p = mgl::glMapBufferRange(kArray, 3 * kQuadBytes, kQuadBytes, Access::Write | Access::InvalidateRange);
            write_bytes(static_cast<std::uint8_t*>(p), 0, quad_bytes(3, kYellow));
            (void)mgl::glUnmapBuffer(kArray);
            draw_slots(env, 0, buf_[0], 0, 4);
            const Rgba8 restored = demo::read_pixel(quad_probes(r3)[0].first, quad_probes(r3)[0].second);
            check.expect(near(restored, kYellow, kTolerance), "negative control: restoring the bytes restores the pixel (cell 0 is correct again)", std::format("pixel {}", demo::format_rgba(restored)));
            mgl::glBindBuffer(kArray, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            // (b) Byte comparison: the shadow must flag a corrupted byte, and must skip bytes it was told are undefined.
            Shadow s(8);
            s.set(0, Bytes{1, 2, 3, 4, 5, 6, 7, 8});
            Bytes actual{1, 2, 3, 4, 5, 6, 7, 8};
            const auto exact = s.compare(actual);
            actual[3] = 99;
            const auto corrupted = s.compare(actual);
            s.forget(3, 1);
            const auto skipped = s.compare(actual);
            check.expect(exact.mismatches == 0 && exact.compared == 8 && corrupted.mismatches == 1 && corrupted.first == 3 &&
                             skipped.mismatches == 0 && skipped.skipped == 1 && skipped.compared == 7,
                         "negative control: the byte comparison reports a flipped byte (1 mismatch at offset 3) and skips a byte declared undefined",
                         std::format("exact {}/{} , corrupted {} mismatches, undefined {} skipped", exact.mismatches, exact.compared, corrupted.mismatches, skipped.skipped));
            check.expect_no_gl_error("the negative controls leave no GL error behind");
        }

        // ----------------------------------------------------------------------------------
        // State
        // ----------------------------------------------------------------------------------
        mgl::ProgramId program_{};
        mgl::UniformLocation u_shift_{};
        mgl::VertexArrayId vao_{};
        std::array<mgl::BufferId, kCells> buf_{};
        std::array<mgl::BufferId, 2> aux_{};
        std::vector<Obs> obs_;
        std::array<Slots, kCells> model_{};
        std::string mapped_draw_note_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-buffer-mapping",
        .title = "Buffer mapping and copying",
        .min_es = {3, 0},
        .summary = "glMapBufferRange with all six access flags, glFlushMappedBufferRange, glUnmapBuffer, mapped-state queries "
                   "(32- and 64-bit) and glCopyBufferSubData, with every defined error; results checked byte for byte "
                   "through MAP_READ_BIT mappings and by drawing from the buffers.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<BufferMapping>(argc, argv, info);
}
