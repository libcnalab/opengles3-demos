// es31-indirect-draw
//
// Indirect drawing (OpenGL ES 3.1; section numbers follow ES 3.2): glDrawArraysIndirect and
// glDrawElementsIndirect take their arguments from a command structure stored in the buffer bound to
// DRAW_INDIRECT_BUFFER, at the byte offset passed as the `indirect` parameter (sections 10.3.9, 10.5):
//
//   DrawArraysIndirectCommand   { uint count, instanceCount, first, reservedMustBeZero; }           16 bytes
//   DrawElementsIndirectCommand { uint count, instanceCount, firstIndex; int baseVertex;
//                                 uint reservedMustBeZero; }                                        20 bytes
//
// One buffer holds 18 commands of both kinds at offsets that are multiples of 4 bytes but not of the
// command size. Every command is drawn with different vertex data, primitive modes, index types and viewport
// placements; a CPU model that interprets the same commands (vertex IDs, instance counts, divisors,
// firstIndex scaled by the index size, baseVertex added after the index is read) predicts every pixel.
//
// The 256x256 surface is a 4x4 grid of 64x64 cells (row 3 is the top row). Cells with two parts show the
// state before (left or top) and after (right or bottom) a change of the command buffer:
//
//   row 3:  [A0] one triangle        [A1] strip, first = 11, 3 inst.  [A2] fan, 2 instances     [A3] left, EMPTY, right
//   row 2:  [E0] USHORT, full quad   [E1] firstIndex 6 | baseVertex   [E2] baseVertex 4         [E3] 3 inst., divisor 2
//   row 1:  [T0] UBYTE firstIndex    [T1] UINT firstIndex + base 4    [V0] gl_VertexID arrays   [V1] gl_VertexID elements
//   row 0:  [D0] glBufferSubData of   [D1] glBufferSubData of one      [D2] compute-written      [D3] compute-written
//           a whole command           word (instanceCount)             arrays command            elements command
//
// A3 draws one command twice at different places with a zero-instance command in between (instanceCount 0
// draws nothing and is not an error). The two compute-written cells are drawn after each of two dispatches
// that count different numbers of visible flags (top: 3 instances; bottom: 5), ordered by
// glMemoryBarrier(COMMAND_BARRIER_BIT).
//
// validate(): all 21 regions pixel by pixel, the compute output read back word by word, DRAW_INDIRECT_BUFFER
// state, every error of section 10.5, and mutation checks that prove the comparisons can fail.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kGrid = 4;
    constexpr std::array<std::uint8_t, 4> kClear8 = {20, 30, 50, 255};

    // -----------------------------------------------------------------------------------------
    // Command layouts (ES 3.2 section 10.5). "Tightly packed 32-bit values" means no padding.
    // -----------------------------------------------------------------------------------------
    struct DrawArraysIndirectCommand
    {
        std::uint32_t count;
        std::uint32_t instanceCount;
        std::uint32_t first;
        std::uint32_t reservedMustBeZero;
    };
    struct DrawElementsIndirectCommand
    {
        std::uint32_t count;
        std::uint32_t instanceCount;
        std::uint32_t firstIndex;
        std::int32_t baseVertex;
        std::uint32_t reservedMustBeZero;
    };
    static_assert(sizeof(DrawArraysIndirectCommand) == 16);
    static_assert(sizeof(DrawElementsIndirectCommand) == 20);

    // -----------------------------------------------------------------------------------------
    // Source data. The CPU model is built from these same arrays.
    // -----------------------------------------------------------------------------------------
    struct Rgb
    {
        std::uint8_t r, g, b;
    };
    constexpr std::array<Rgb, 8> kPalette = {{{230, 60, 60}, {60, 200, 80}, {70, 110, 240}, {240, 200, 50},
                                              {200, 80, 220}, {60, 210, 210}, {240, 140, 40}, {150, 150, 150}}};

    struct Vec2
    {
        float x, y;
    };
    // One vertex pool, several shapes (the vertex ID of a vertex is its index in this array):
    //   0..3 square (strip order BL BR TL TR)   4..7 diamond (bottom, right, left, top)
    //   8..10 skewed triangle   11..14 quad strip (BL BR TL TR)   15..20 fan (center + 5 rim points, 4 triangles)
    constexpr std::array<Vec2, 21> kPool = {{
        {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
        {0, -1}, {1, 0}, {-1, 0}, {0, 1},
        {-1, -0.8f}, {1, -0.4f}, {-0.2f, 1},
        {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
        {0, 0}, {1, 0}, {0.5f, 0.866f}, {-0.5f, 0.866f}, {-1, 0}, {-0.5f, -0.866f},
    }};

    // The index lists, shared by the UBYTE, USHORT and UINT element buffers (21 entries each):
    //   [0..5] two triangles = a whole square/diamond   [6..8] BL BR TL   [9..11] BR TR TL
    //   [12..14] BL BR TR   [15..17] BL TR TL   [18..20] {10, 11, 12}: needs baseVertex -2 to reach the triangle
    constexpr std::array<std::uint32_t, 21> kIndices = {0, 1, 2, 2, 1, 3, 0, 1, 2, 1, 3, 2, 0, 1, 3, 0, 3, 2, 10, 11, 12};

    constexpr std::size_t kFlagCount = 8;
    using Flags = std::array<std::uint32_t, kFlagCount>;
    constexpr Flags kFlags1 = {1, 0, 1, 0, 0, 1, 0, 0};  // 3 visible
    constexpr Flags kFlags2 = {1, 1, 0, 1, 1, 0, 1, 0};  // 5 visible

    // Commands written by the CPU. Offsets are assigned in init() by appending them to one buffer.
    constexpr DrawArraysIndirectCommand kA0 = {3, 1, 8, 0};
    constexpr DrawArraysIndirectCommand kA1 = {4, 3, 11, 0};
    constexpr DrawArraysIndirectCommand kA2 = {6, 2, 15, 0};
    constexpr DrawArraysIndirectCommand kAZ = {3, 0, 8, 0};  // instanceCount 0
    constexpr DrawElementsIndirectCommand kE0 = {6, 1, 0, 0, 0};
    constexpr DrawElementsIndirectCommand kE1a = {3, 1, 6, 0, 0};
    constexpr DrawElementsIndirectCommand kE1b = {3, 1, 18, -2, 0};  // negative baseVertex
    constexpr DrawElementsIndirectCommand kE2 = {6, 1, 0, 4, 0};
    constexpr DrawElementsIndirectCommand kE3 = {3, 3, 9, 4, 0};
    constexpr DrawElementsIndirectCommand kT0 = {3, 2, 12, 0, 0};
    constexpr DrawElementsIndirectCommand kT1 = {3, 2, 15, 4, 0};
    constexpr DrawArraysIndirectCommand kV0 = {6, 2, 4, 0};
    constexpr DrawElementsIndirectCommand kV1 = {6, 1, 0, 7, 0};
    // Commands that change after they were written (D0, D1) and their initial/changed values.
    constexpr DrawArraysIndirectCommand kD0Before = {3, 1, 8, 0};
    constexpr DrawArraysIndirectCommand kD0After = {4, 2, 11, 0};
    constexpr DrawArraysIndirectCommand kD1Before = {4, 1, 11, 0};
    constexpr std::uint32_t kD1InstanceAfter = 3;  // written alone with a 4-byte glBufferSubData at offset + 4
    // Initial content of the two compute-written commands: harmless (instanceCount 0).
    constexpr DrawArraysIndirectCommand kC0Initial = {4, 0, 11, 0};
    constexpr DrawElementsIndirectCommand kC1Initial = {6, 0, 0, 4, 0};
    // Tail: an elements command with instanceCount 0. Its last four words are {0,0,0,0}, a valid empty
    // arrays command, so both an arrays and an elements command fit exactly at the end of the buffer.
    constexpr DrawElementsIndirectCommand kTail = {3, 0, 0, 0, 0};
    constexpr std::uint32_t kGuard = 0xDEADBEEFu;

    // CPU twin of commands.comp.
    struct ComputeResult
    {
        DrawArraysIndirectCommand arrays;
        DrawElementsIndirectCommand elements;
    };
    ComputeResult emulate_compute(const Flags& flags)
    {
        std::uint32_t n = 0;
        for (const std::uint32_t f : flags)
            if (f != 0) ++n;
        const bool many = n > 3;
        return {{4, n, 11, 0}, {many ? 3u : 6u, n, many ? 9u : 0u, 4, 0}};
    }

    // -----------------------------------------------------------------------------------------
    // What is drawn where
    // -----------------------------------------------------------------------------------------
    enum class Kind { Arrays, Elements };
    enum Vao : std::size_t { kVaoMain, kVaoUByte, kVaoUInt, kVaoDiv2, kVaoZero, kVaoNoElem, kVaoCount };

    struct Place  // uniforms: shape space -> NDC of the viewport; mode 1 = position from gl_VertexID
    {
        double sx, sy, ox, oy, px, py;
        int mode;
    };

    struct Draw
    {
        Kind kind;
        mgl::PrimitiveType primitive;
        mgl::IndexType type;  // elements only
        Vao vao;
        std::size_t offset;   // byte offset of the command: the only thing the GL is told
        DrawArraysIndirectCommand arrays;      // what the command contains at draw time (CPU copy)
        DrawElementsIndirectCommand elements;  // (only the member of the right kind is used)
        Place place;
    };

    struct Rect
    {
        int x, y, w, h;
    };

    struct Region
    {
        std::string name;
        Rect rect;
        std::vector<Draw> draws;
    };

    enum RegionId : std::size_t
    {
        kRA0, kRA1, kRA2, kRA3, kRE0, kRE1L, kRE1R, kRE2, kRE3, kRT0, kRT1, kRV0, kRV1,
        kRD0L, kRD0R, kRD1L, kRD1R, kRD2T, kRD2B, kRD3T, kRD3B, kRegionCount
    };

    // -----------------------------------------------------------------------------------------
    // CPU model: flat-colored triangles in NDC, drawn in order.
    // -----------------------------------------------------------------------------------------
    struct Pt
    {
        double x, y;
    };
    struct Tri
    {
        std::array<Pt, 3> p;
        Rgb color;
    };
    enum class Cover { Outside, Inside, Unsure };

    Pt vertex_position(long id, int mode)
    {
        if (mode == 1)
        {
            const double angle = static_cast<double>(id) * 1.3;
            return {0.8 * std::cos(angle), 0.8 * std::sin(angle)};
        }
        if (id < 0 || id >= static_cast<long>(kPool.size())) return {0, 0};
        const Vec2& v = kPool[static_cast<std::size_t>(id)];
        return {static_cast<double>(v.x), static_cast<double>(v.y)};
    }

    // Interprets one draw: the vertex IDs of ES 3.2 section 10.5 (first + i, or baseVertex + index), instances,
    // the attribute divisor of the VAO, and primitive assembly.
    std::vector<Tri> model_draw(const Draw& d)
    {
        std::vector<long> ids;
        std::uint32_t instances = 0;
        if (d.kind == Kind::Arrays)
        {
            instances = d.arrays.instanceCount;
            for (std::uint32_t k = 0; k < d.arrays.count; ++k) ids.push_back(static_cast<long>(d.arrays.first) + k);
        }
        else
        {
            instances = d.elements.instanceCount;
            for (std::uint32_t k = 0; k < d.elements.count; ++k)
                ids.push_back(static_cast<long>(d.elements.baseVertex) + static_cast<long>(kIndices[d.elements.firstIndex + k]));
        }
        const std::uint32_t divisor = d.vao == kVaoDiv2 ? 2u : 1u;
        std::vector<Tri> out;
        for (std::uint32_t inst = 0; inst < instances; ++inst)
        {
            const Rgb color = kPalette[(inst / divisor) % kPalette.size()];
            std::vector<Pt> pts;
            for (const long id : ids)
            {
                const Pt s = vertex_position(id, d.place.mode);
                pts.push_back({s.x * d.place.sx + d.place.ox + inst * d.place.px, s.y * d.place.sy + d.place.oy + inst * d.place.py});
            }
            auto emit = [&](std::size_t a, std::size_t b, std::size_t c) { out.push_back({{pts[a], pts[b], pts[c]}, color}); };
            switch (d.primitive)
            {
            case mgl::PrimitiveType::Triangles:
                for (std::size_t i = 0; i + 2 < pts.size(); i += 3) emit(i, i + 1, i + 2);
                break;
            case mgl::PrimitiveType::TriangleStrip:
                for (std::size_t i = 0; i + 2 < pts.size(); ++i) emit(i, i + 1, i + 2);
                break;
            case mgl::PrimitiveType::TriangleFan:
                for (std::size_t i = 1; i + 1 < pts.size(); ++i) emit(0, i, i + 1);
                break;
            default: break;
            }
        }
        return out;
    }

    // Classifies the pixel center (px, py) of a cw x ch viewport. Within `margin` pixels of an edge the answer depends
    // on the rasterizer's tie-breaking rules and is reported as Unsure (never compared).
    Cover classify(const Tri& t, double px, double py, double cw, double ch, double margin)
    {
        std::array<double, 3> X{}, Y{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            X[i] = (t.p[i].x + 1.0) * 0.5 * cw;
            Y[i] = (t.p[i].y + 1.0) * 0.5 * ch;
        }
        auto edge = [](double ax, double ay, double bx, double by, double x, double y) { return (bx - ax) * (y - ay) - (by - ay) * (x - ax); };
        const double area = edge(X[0], Y[0], X[1], Y[1], X[2], Y[2]);
        if (std::abs(area) < 1e-9) return Cover::Outside;
        const double sign = area > 0 ? 1.0 : -1.0;
        double min_distance = 1e30;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const std::size_t j = (i + 1) % 3, k = (i + 2) % 3;
            const double e = edge(X[j], Y[j], X[k], Y[k], px, py);
            min_distance = std::min(min_distance, sign * e / std::hypot(X[k] - X[j], Y[k] - Y[j]));
        }
        if (min_distance < -margin) return Cover::Outside;
        if (min_distance <= margin) return Cover::Unsure;
        return Cover::Inside;
    }

    struct Sample
    {
        Cover cover = Cover::Outside;
        Rgb color{};
    };
    // Later triangles overwrite earlier ones (no depth test): the last definitely-covering triangle wins, and a pixel
    // that a later, uncertain triangle might also have touched is Unsure.
    Sample sample_scene(const std::vector<Tri>& tris, double px, double py, double cw, double ch)
    {
        Sample result;
        bool unsure_after = false;
        for (const Tri& t : tris)
        {
            const Cover c = classify(t, px, py, cw, ch, 1.0);
            if (c == Cover::Inside) { result = {Cover::Inside, t.color}; unsure_after = false; }
            else if (c == Cover::Unsure) unsure_after = true;
        }
        return unsure_after ? Sample{Cover::Unsure, {}} : result;
    }

    struct Scan
    {
        int inside = 0, outside = 0, unsure = 0, bad = 0, max_dev = 0;
        std::string first_bad;
    };

    // Reads the whole region and compares every pixel the model is certain about. +-1 is the 8-bit quantization of the
    // float color (the palette values are exact multiples of 1/255).
    Scan scan_region(const std::vector<Tri>& tris, Rect r, int tolerance = 1)
    {
        Scan s;
        const std::vector<std::uint8_t> pixels = demo::read_rgba8(r.x, r.y, r.w, r.h);
        for (int y = 0; y < r.h; ++y)
            for (int x = 0; x < r.w; ++x)
            {
                const Sample m = sample_scene(tris, x + 0.5, y + 0.5, r.w, r.h);
                if (m.cover == Cover::Unsure) { ++s.unsure; continue; }
                const Rgba8 want = m.cover == Cover::Inside ? Rgba8{m.color.r, m.color.g, m.color.b, 255} : Rgba8{kClear8[0], kClear8[1], kClear8[2], kClear8[3]};
                ++(m.cover == Cover::Inside ? s.inside : s.outside);
                const std::uint8_t* got = &pixels[static_cast<std::size_t>((y * r.w + x) * 4)];
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int(got[k]) - int(want[k]));
                    s.max_dev = std::max(s.max_dev, dev);
                    ok = ok && dev <= tolerance;
                }
                if (!ok && s.bad++ == 0)
                    s.first_bad = std::format("first mismatch at region pixel ({},{}): got {} want {}", x, y,
                                              demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return s;
    }

    std::vector<Tri> model_region(const Region& region)
    {
        std::vector<Tri> all;
        for (const Draw& d : region.draws)
        {
            const std::vector<Tri> t = model_draw(d);
            all.insert(all.end(), t.begin(), t.end());
        }
        return all;
    }

    // -----------------------------------------------------------------------------------------
    // The demo
    // -----------------------------------------------------------------------------------------
    class IndirectDraw final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/indirect.vert", "shaders/flat.frag");
            u_mode_ = mgl::glGetUniformLocation(program_, "u_mode");
            u_scale_ = mgl::glGetUniformLocation(program_, "u_scale");
            u_origin_ = mgl::glGetUniformLocation(program_, "u_origin");
            u_pitch_ = mgl::glGetUniformLocation(program_, "u_pitch");
            build_compute_program(env);
            build_tf_program(env);

            // DRAW_INDIRECT_BUFFER_BINDING must start at zero (ES 3.2 Table 21.4); validate() reports it.
            initial_indirect_binding_ = demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding);

            layout_commands();
            create_buffers();
            create_vaos();
            create_regions(env);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);  // exact color writes
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(program_);
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, indirect_);
            reset_dynamic_commands();

            for (const RegionId id : {kRA0, kRA1, kRA2, kRA3, kRE0, kRE1L, kRE1R, kRE2, kRE3, kRT0, kRT1, kRV0, kRV1}) draw_region(regions_[id]);

            // D0: the command is rewritten as a whole with glBufferSubData between two draws at the same offset.
            draw_region(regions_[kRD0L]);
            mgl::glBufferSubData(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(off_d0_), sizeof kD0After, &kD0After);
            draw_region(regions_[kRD0R]);

            // D1: only the 4-byte instanceCount word changes (offset + 4).
            draw_region(regions_[kRD1L]);
            mgl::glBufferSubData(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(off_d1_ + 4), sizeof kD1InstanceAfter, &kD1InstanceAfter);
            draw_region(regions_[kRD1R]);

            // D2/D3: a compute shader writes the commands; COMMAND_BARRIER_BIT orders its stores before the draws.
            dispatch_commands(kFlags1);
            draw_region(regions_[kRD2T]);
            draw_region(regions_[kRD3T]);
            dispatch_commands(kFlags2);
            draw_region(regions_[kRD2B]);
            draw_region(regions_[kRD3B]);

            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_pixels(check);
            check_compute_output(check);
            check_negative_controls(check);
            check_indirect_buffer_state(check);
            check_errors(check, env.context_version);
            render(env, 0);  // the error checks draw nothing visible, but leave the capture image exactly as rendered
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            mgl::glDeleteProgram(program_);
            mgl::glDeleteProgram(compute_);
            mgl::glDeleteProgram(tf_program_);
        }

    private:
        // ---------------------------------------------------------------------------------------------
        // Setup
        // ---------------------------------------------------------------------------------------------
        void build_compute_program(const demo::Env& env)
        {
            std::string log;
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, env.load_text("shaders/commands.comp"), log);
            if (cs.value == 0) throw demo::Failure("commands.comp failed to compile:\n" + log);
            const std::array<mgl::ShaderId, 1> shaders = {cs};
            compute_ = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            if (compute_.value == 0) throw demo::Failure("commands.comp failed to link:\n" + log);
            u_arrays_word_ = mgl::glGetUniformLocation(compute_, "u_arrays_word");
            u_elements_word_ = mgl::glGetUniformLocation(compute_, "u_elements_word");
        }

        // Same shaders as program_, linked with a captured varying: used only to make transform feedback active.
        void build_tf_program(const demo::Env& env)
        {
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, env.load_text("shaders/indirect.vert"), log);
            if (vs.value == 0) throw demo::Failure("indirect.vert failed to compile:\n" + log);
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, env.load_text("shaders/flat.frag"), log);
            if (fs.value == 0) throw demo::Failure("flat.frag failed to compile:\n" + log);
            tf_program_ = mgl::glCreateProgram();
            mgl::glAttachShader(tf_program_, vs);
            mgl::glAttachShader(tf_program_, fs);
            const std::array<const GLchar*, 1> varyings = {"v_color"};
            mgl::glTransformFeedbackVaryings(tf_program_, 1, varyings.data(), mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(tf_program_);
            GLint status = GL_FALSE;
            mgl::glGetProgramiv(tf_program_, mgl::ProgramParameter::LinkStatus, &status);
            const std::string link_log = demo::program_info_log(tf_program_);
            mgl::glDetachShader(tf_program_, vs);
            mgl::glDetachShader(tf_program_, fs);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (status != GL_TRUE) throw demo::Failure("transform feedback program failed to link:\n" + link_log);
        }

        std::size_t append(const DrawArraysIndirectCommand& c)
        {
            const std::size_t offset = words_.size() * sizeof(std::uint32_t);
            const std::array<std::uint32_t, 4> w = {c.count, c.instanceCount, c.first, c.reservedMustBeZero};
            words_.insert(words_.end(), w.begin(), w.end());
            return offset;
        }
        std::size_t append(const DrawElementsIndirectCommand& c)
        {
            const std::size_t offset = words_.size() * sizeof(std::uint32_t);
            const std::array<std::uint32_t, 5> w = {c.count, c.instanceCount, c.firstIndex, static_cast<std::uint32_t>(c.baseVertex), c.reservedMustBeZero};
            words_.insert(words_.end(), w.begin(), w.end());
            return offset;
        }

        // All commands live in one buffer. Arrays commands are 16 bytes, elements commands 20 bytes; a guard word
        // after the arrays commands shifts the following offsets off multiples of 16 and 20.
        void layout_commands()
        {
            off_a0_ = append(kA0);
            off_a1_ = append(kA1);
            off_a2_ = append(kA2);
            off_az_ = append(kAZ);
            words_.push_back(kGuard);
            off_e0_ = append(kE0);
            off_e1a_ = append(kE1a);
            off_e1b_ = append(kE1b);
            off_e2_ = append(kE2);
            off_e3_ = append(kE3);
            off_t0_ = append(kT0);
            off_t1_ = append(kT1);
            off_v0_ = append(kV0);
            off_v1_ = append(kV1);
            off_d0_ = append(kD0Before);
            off_d1_ = append(kD1Before);
            words_.push_back(kGuard);
            off_c0_ = append(kC0Initial);
            words_.push_back(kGuard);
            off_c1_ = append(kC1Initial);
            off_tail_ = append(kTail);
            buffer_size_ = words_.size() * sizeof(std::uint32_t);
        }

        template <typename T>
        mgl::BufferId make_buffer(mgl::BufferTarget target, std::span<const T> data)
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(target, b);
            mgl::glBufferData(target, data, mgl::BufferUsage::StaticDraw);
            buffers_.push_back(b);
            return b;
        }

        void create_buffers()
        {
            indirect_ = make_buffer(mgl::BufferTarget::DrawIndirect, std::span<const std::uint32_t>(words_));
            // The same buffer object is the destination of the compute shader's stores (SSBO binding 0).
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, indirect_);
            flags_ = make_buffer(mgl::BufferTarget::ShaderStorage, std::span<const std::uint32_t>(kFlags1));
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, flags_);

            pool_ = make_buffer(mgl::BufferTarget::Array, std::span<const Vec2>(kPool));
            std::array<float, 24> palette{};
            for (std::size_t i = 0; i < kPalette.size(); ++i)
            {
                palette[i * 3 + 0] = kPalette[i].r / 255.0f;
                palette[i * 3 + 1] = kPalette[i].g / 255.0f;
                palette[i * 3 + 2] = kPalette[i].b / 255.0f;
            }
            palette_ = make_buffer(mgl::BufferTarget::Array, std::span<const float>(palette));

            std::array<std::uint8_t, kIndices.size()> ub{};
            std::array<std::uint16_t, kIndices.size()> us{};
            std::array<std::uint32_t, kIndices.size()> ui{};
            for (std::size_t i = 0; i < kIndices.size(); ++i)
            {
                ub[i] = static_cast<std::uint8_t>(kIndices[i]);
                us[i] = static_cast<std::uint16_t>(kIndices[i]);
                ui[i] = kIndices[i];
            }
            ebo_ubyte_ = make_buffer(mgl::BufferTarget::ElementArray, std::span<const std::uint8_t>(ub));
            ebo_ushort_ = make_buffer(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(us));
            ebo_uint_ = make_buffer(mgl::BufferTarget::ElementArray, std::span<const std::uint32_t>(ui));

            tf_buffer_ = make_buffer(mgl::BufferTarget::TransformFeedback, std::span<const std::uint32_t>(std::array<std::uint32_t, 256>{}));
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
        }

        // The VAOs differ in the element array buffer (index type) and in the divisor of the color attribute.
        void create_vaos()
        {
            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            struct Setup
            {
                Vao slot;
                mgl::BufferId ebo;
                GLuint divisor;
            };
            const Setup setups[] = {{kVaoMain, ebo_ushort_, 1}, {kVaoUByte, ebo_ubyte_, 1}, {kVaoUInt, ebo_uint_, 1}, {kVaoDiv2, ebo_ushort_, 2}};
            for (const Setup& s : setups)
            {
                mgl::glBindVertexArray(vao_[s.slot]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, pool_);
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
                mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
                mgl::glBindBuffer(mgl::BufferTarget::Array, palette_);
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
                mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
                mgl::glVertexAttribDivisor(mgl::AttribLocation{1u}, s.divisor);
                mgl::glBindBuffer(mgl::BufferTarget::ElementArray, s.ebo);  // VAO state
            }

            // kVaoNoElem: valid arrays, but no element array buffer (DrawElementsIndirect must fail).
            mgl::glBindVertexArray(vao_[kVaoNoElem]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, pool_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, palette_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));

            // kVaoZero: element buffer present, attribute 1 sourced from a buffer, attribute 0 ENABLED with buffer 0.
            // A NULL pointer is allowed on a non-default VAO (section 10.3.1), but "any enabled vertex array" with
            // buffer 0 makes the indirect draws INVALID_OPERATION (section 10.5).
            mgl::glBindVertexArray(vao_[kVaoZero]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, palette_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ebo_ushort_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 0, nullptr);

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
        }

        static Rect cell(const demo::Env& env, int row, int col)
        {
            const int cw = env.width / kGrid, ch = env.height / kGrid;
            return {col * cw, row * ch, cw, ch};
        }
        static Rect left_half(Rect r) { return {r.x, r.y, r.w / 2, r.h}; }
        static Rect right_half(Rect r) { return {r.x + r.w / 2, r.y, r.w / 2, r.h}; }
        static Rect top_half(Rect r) { return {r.x, r.y + r.h / 2, r.w, r.h / 2}; }
        static Rect bottom_half(Rect r) { return {r.x, r.y, r.w, r.h / 2}; }

        Draw arrays_draw(mgl::PrimitiveType prim, Vao vao, std::size_t offset, const DrawArraysIndirectCommand& c, Place p) const
        {
            return {Kind::Arrays, prim, mgl::IndexType::UnsignedShort, vao, offset, c, {}, p};
        }
        Draw elements_draw(mgl::IndexType type, Vao vao, std::size_t offset, const DrawElementsIndirectCommand& c, Place p) const
        {
            return {Kind::Elements, mgl::PrimitiveType::Triangles, type, vao, offset, {}, c, p};
        }

        void create_regions(const demo::Env& env)
        {
            constexpr auto kTris = mgl::PrimitiveType::Triangles;
            constexpr auto kStrip = mgl::PrimitiveType::TriangleStrip;
            constexpr auto kFan = mgl::PrimitiveType::TriangleFan;
            constexpr auto kUS = mgl::IndexType::UnsignedShort;
            regions_.resize(kRegionCount);
            auto set = [&](RegionId id, const char* name, Rect r, std::vector<Draw> draws) { regions_[id] = {name, r, std::move(draws)}; };

            set(kRA0, "A0 DrawArraysIndirect, triangle", cell(env, 3, 0), {arrays_draw(kTris, kVaoMain, off_a0_, kA0, {0.6, 0.6, 0, 0, 0, 0, 0})});
            set(kRA1, "A1 strip first=11, 3 instances", cell(env, 3, 1), {arrays_draw(kStrip, kVaoMain, off_a1_, kA1, {0.2, 0.2, -0.62, 0, 0.62, 0, 0})});
            set(kRA2, "A2 fan, 2 instances", cell(env, 3, 2), {arrays_draw(kFan, kVaoMain, off_a2_, kA2, {0.4, 0.4, -0.45, 0, 0.9, 0, 0})});
            set(kRA3, "A3 command / instanceCount 0 / command", cell(env, 3, 3),
                {arrays_draw(kTris, kVaoMain, off_a0_, kA0, {0.3, 0.3, -0.55, 0, 0, 0, 0}),
                 arrays_draw(kTris, kVaoMain, off_az_, kAZ, {0.3, 0.3, 0, 0, 0, 0, 0}),
                 arrays_draw(kTris, kVaoMain, off_a0_, kA0, {0.3, 0.3, 0.55, 0, 0, 0, 0})});

            set(kRE0, "E0 DrawElementsIndirect USHORT", cell(env, 2, 0), {elements_draw(kUS, kVaoMain, off_e0_, kE0, {0.6, 0.6, 0, 0, 0, 0, 0})});
            set(kRE1L, "E1 left: firstIndex 6", left_half(cell(env, 2, 1)), {elements_draw(kUS, kVaoMain, off_e1a_, kE1a, {0.8, 0.8, 0, 0, 0, 0, 0})});
            set(kRE1R, "E1 right: baseVertex -2", right_half(cell(env, 2, 1)), {elements_draw(kUS, kVaoMain, off_e1b_, kE1b, {0.8, 0.8, 0, 0, 0, 0, 0})});
            set(kRE2, "E2 baseVertex 4 (diamond)", cell(env, 2, 2), {elements_draw(kUS, kVaoMain, off_e2_, kE2, {0.7, 0.7, 0, 0, 0, 0, 0})});
            set(kRE3, "E3 3 instances, divisor 2", cell(env, 2, 3), {elements_draw(kUS, kVaoDiv2, off_e3_, kE3, {0.25, 0.5, -0.6, 0, 0.6, 0, 0})});

            set(kRT0, "T0 UNSIGNED_BYTE firstIndex 12", cell(env, 1, 0),
                {elements_draw(mgl::IndexType::UnsignedByte, kVaoUByte, off_t0_, kT0, {0.4, 0.4, -0.45, 0, 0.9, 0, 0})});
            set(kRT1, "T1 UNSIGNED_INT firstIndex 15, baseVertex 4", cell(env, 1, 1),
                {elements_draw(mgl::IndexType::UnsignedInt, kVaoUInt, off_t1_, kT1, {0.4, 0.4, -0.45, 0, 0.9, 0, 0})});
            set(kRV0, "V0 gl_VertexID, arrays first=4", cell(env, 1, 2), {arrays_draw(kTris, kVaoMain, off_v0_, kV0, {0.5, 0.5, -0.45, 0, 0.9, 0, 1})});
            set(kRV1, "V1 gl_VertexID, elements baseVertex 7", cell(env, 1, 3), {elements_draw(kUS, kVaoMain, off_v1_, kV1, {0.9, 0.9, 0, 0, 0, 0, 1})});

            set(kRD0L, "D0 before glBufferSubData", left_half(cell(env, 0, 0)), {arrays_draw(kTris, kVaoMain, off_d0_, kD0Before, {0.8, 0.8, 0, 0, 0, 0, 0})});
            set(kRD0R, "D0 after glBufferSubData", right_half(cell(env, 0, 0)), {arrays_draw(kStrip, kVaoMain, off_d0_, kD0After, {0.8, 0.35, 0, -0.5, 0, 1.0, 0})});
            const Place d1 = {0.8, 0.2, 0, -0.6, 0, 0.6, 0};
            set(kRD1L, "D1 before 4-byte update", left_half(cell(env, 0, 1)), {arrays_draw(kStrip, kVaoMain, off_d1_, kD1Before, d1)});
            DrawArraysIndirectCommand d1_after = kD1Before;
            d1_after.instanceCount = kD1InstanceAfter;
            set(kRD1R, "D1 after 4-byte update", right_half(cell(env, 0, 1)), {arrays_draw(kStrip, kVaoMain, off_d1_, d1_after, d1)});

            const ComputeResult c1 = emulate_compute(kFlags1), c2 = emulate_compute(kFlags2);
            const Place pa = {0.14, 0.7, -0.8, 0, 0.4, 0, 0};
            const Place pe = {0.16, 0.7, -0.8, 0, 0.4, 0, 0};
            set(kRD2T, "D2 compute arrays, 3 visible", top_half(cell(env, 0, 2)), {arrays_draw(kStrip, kVaoMain, off_c0_, c1.arrays, pa)});
            set(kRD2B, "D2 compute arrays, 5 visible", bottom_half(cell(env, 0, 2)), {arrays_draw(kStrip, kVaoMain, off_c0_, c2.arrays, pa)});
            set(kRD3T, "D3 compute elements, 3 visible", top_half(cell(env, 0, 3)), {elements_draw(kUS, kVaoMain, off_c1_, c1.elements, pe)});
            set(kRD3B, "D3 compute elements, 5 visible", bottom_half(cell(env, 0, 3)), {elements_draw(kUS, kVaoMain, off_c1_, c2.elements, pe)});
        }

        // ---------------------------------------------------------------------------------------------
        // Drawing
        // ---------------------------------------------------------------------------------------------
        void issue(const Draw& d) const
        {
            mgl::glUniform1i(u_mode_, d.place.mode);
            mgl::glUniform2f(u_scale_, static_cast<float>(d.place.sx), static_cast<float>(d.place.sy));
            mgl::glUniform2f(u_origin_, static_cast<float>(d.place.ox), static_cast<float>(d.place.oy));
            mgl::glUniform2f(u_pitch_, static_cast<float>(d.place.px), static_cast<float>(d.place.py));
            mgl::glBindVertexArray(vao_[d.vao]);
            // The command itself is only addressed by its byte offset into the DRAW_INDIRECT_BUFFER.
            if (d.kind == Kind::Arrays)
                mgl::glDrawArraysIndirect(d.primitive, demo::buffer_offset(d.offset));
            else
                mgl::glDrawElementsIndirect(d.primitive, d.type, demo::buffer_offset(d.offset));
        }

        void draw_region(const Region& r) const
        {
            mgl::glViewport(r.rect.x, r.rect.y, r.rect.w, r.rect.h);
            for (const Draw& d : r.draws) issue(d);
        }

        // Puts the commands that the frame modifies back to their initial values, so every frame starts equal.
        void reset_dynamic_commands() const
        {
            mgl::glBufferSubData(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(off_d0_), sizeof kD0Before, &kD0Before);
            mgl::glBufferSubData(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(off_d1_), sizeof kD1Before, &kD1Before);
            mgl::glBufferSubData(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(off_c0_), sizeof kC0Initial, &kC0Initial);
            mgl::glBufferSubData(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(off_c1_), sizeof kC1Initial, &kC1Initial);
        }

        // Compute pass: count the visible flags and write both commands. The barrier after the dispatch makes those buffer
        // stores visible to the commands that Draw*Indirect sources from DRAW_INDIRECT_BUFFER (COMMAND_BARRIER_BIT) and to
        // the map-and-read of validate() (BUFFER_UPDATE_BARRIER_BIT), ES 3.2 section 7.11.2.
        //
        // The barrier BEFORE the dispatch is just as necessary: the dispatch overwrites the commands that the previous
        // Draw*Indirect calls read. Section 7.11.2: "Explicit synchronization is required to ensure that the effects of
        // buffer and texture data stores performed by shaders will be visible to subsequent operations using the same
        // objects and will not overwrite data still to be read by previously requested operations." SHADER_STORAGE_-
        // BARRIER_BIT provides that: "assignments to ... shader buffer variables after the barrier will not execute until
        // all memory accesses (e.g., loads, stores, texture fetches, vertex fetches) initiated prior to the barrier
        // complete." Without it the second dispatch may rewrite the command before the first draws have read it (ANGLE on
        // Vulkan does exactly that: both "3 visible" regions were drawn with the "5 visible" commands).
        void dispatch_commands(const Flags& flags) const
        {
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, flags_);
            mgl::glBufferSubData(mgl::BufferTarget::ShaderStorage, 0, std::span<const std::uint32_t>(flags));
            mgl::glUseProgram(compute_);
            mgl::glUniform1ui(u_arrays_word_, static_cast<GLuint>(off_c0_ / sizeof(std::uint32_t)));
            mgl::glUniform1ui(u_elements_word_, static_cast<GLuint>(off_c1_ / sizeof(std::uint32_t)));
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::Command | mgl::MemoryBarrierMask::BufferUpdate);
            mgl::glUseProgram(program_);
        }

        // ---------------------------------------------------------------------------------------------
        // Validation
        // ---------------------------------------------------------------------------------------------
        void check_pixels(demo::Check& check) const
        {
            for (const Region& r : regions_)
            {
                const Scan s = scan_region(model_region(r), r.rect);
                const bool enough = s.inside >= 60 && s.outside >= 60;
                check.expect(s.bad == 0 && enough,
                             std::format("{}: {} covered + {} background pixels match the CPU model (max deviation {}/255, {} edge pixels skipped)",
                                         r.name, s.inside, s.outside, s.max_dev, s.unsure),
                             s.bad ? std::format("{} mismatches; {}", s.bad, s.first_bad)
                                   : !enough ? std::format("only {} covered / {} background pixels were compared", s.inside, s.outside)
                                             : std::string{});
            }
        }

        std::vector<std::uint32_t> read_command_words(std::size_t first_word, std::size_t count) const
        {
            std::vector<std::uint32_t> out(count, 0);
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, indirect_);
            const void* p = mgl::glMapBufferRange(mgl::BufferTarget::DrawIndirect, static_cast<GLintptr>(first_word * sizeof(std::uint32_t)),
                                                  static_cast<GLsizeiptr>(count * sizeof(std::uint32_t)), mgl::MapBufferAccessMask::Read);
            if (p != nullptr) std::memcpy(out.data(), p, count * sizeof(std::uint32_t));
            (void)mgl::glUnmapBuffer(mgl::BufferTarget::DrawIndirect);
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            return out;
        }

        // The compute shader's output is read back word by word, independent of any drawing. The guard words around the two
        // commands and the first word of the tail command (which follows the elements command) must not change.
        void check_compute_output(demo::Check& check) const
        {
            struct Case
            {
                const Flags& flags;
                const char* name;
            };
            const std::size_t w0 = off_c0_ / sizeof(std::uint32_t);
            for (const Case& c : {Case{kFlags1, "3 visible flags"}, Case{kFlags2, "5 visible flags"}})
            {
                dispatch_commands(c.flags);
                const ComputeResult want = emulate_compute(c.flags);
                // guard, DrawArraysIndirectCommand (4 words), guard, DrawElementsIndirectCommand (5 words), first word of the tail
                const std::vector<std::uint32_t> expected = {kGuard, want.arrays.count, want.arrays.instanceCount, want.arrays.first,
                                                             want.arrays.reservedMustBeZero, kGuard, want.elements.count, want.elements.instanceCount,
                                                             want.elements.firstIndex, static_cast<std::uint32_t>(want.elements.baseVertex),
                                                             want.elements.reservedMustBeZero, kTail.count};
                const std::vector<std::uint32_t> got = read_command_words(w0 - 1, expected.size());
                std::string detail;
                for (std::size_t i = 0; i < expected.size() && detail.empty(); ++i)
                    if (got[i] != expected[i]) detail = std::format("word {} is {:#x}, expected {:#x}", i, got[i], expected[i]);
                check.expect(detail.empty(),
                             std::format("compute-written commands ({}): {{4, {}, 11, 0}} and {{{}, {}, {}, 4, 0}} (with their guard words) read back word for word",
                                         c.name, want.arrays.instanceCount, want.elements.count, want.elements.instanceCount, want.elements.firstIndex),
                             detail);
            }
        }

        // Mutations of the CPU model: if the pixels equal the real model they cannot equal these, otherwise the
        // comparison would be vacuous.
        void check_negative_controls(demo::Check& check) const
        {
            auto mutated = [&](RegionId id, auto&& change) {
                Region r = regions_[id];
                change(r.draws[0]);
                return scan_region(model_region(r), r.rect);
            };
            auto expect_differs = [&](const Scan& s, std::string_view what) {
                const bool ok = s.bad >= 20;
                check.expect(ok, std::format("negative control: {} ({} mismatching pixels)", what, s.bad),
                             ok ? std::string{} : "the mutated model still matches the pixels, so the comparison cannot detect this error");
            };
            expect_differs(mutated(kRA1, [](Draw& d) { d.arrays.first = 12; }),
                           "model of A1 with first = 12 instead of 11 does not match");
            expect_differs(mutated(kRA1, [](Draw& d) { d.arrays.instanceCount = 2; }), "model of A1 with 2 instead of 3 instances does not match");
            expect_differs(mutated(kRE1R, [](Draw& d) { d.elements.baseVertex = 0; }),
                           "model of E1 (right) without its negative baseVertex does not match");
            expect_differs(mutated(kRE3, [](Draw& d) { d.vao = kVaoMain; }), "model of E3 with attribute divisor 1 instead of 2 does not match");
            expect_differs(mutated(kRT0, [](Draw& d) { d.elements.firstIndex = 6; }), "model of T0 with firstIndex 6 instead of 12 (a wrong index-size scaling would read another list) does not match");
            expect_differs(mutated(kRV0, [](Draw& d) { d.arrays.first = 5; }), "model of V0 with first = 5: the gl_VertexID triangle rotates");
            expect_differs(mutated(kRD2T, [](Draw& d) { d.arrays = kC0Initial; }),
                           "model of D2 (top) with the stale command from before the dispatch does not match");
            expect_differs(mutated(kRD0R, [](Draw& d) { d.arrays = kD0Before; }), "model of D0 (right) with the command from before glBufferSubData does not match");

            // A3: if instanceCount 0 drew one instance, the middle command (same place as the empty gap) would show a triangle.
            Region a3 = regions_[kRA3];
            a3.draws[1].arrays.instanceCount = 1;
            expect_differs(scan_region(model_region(a3), a3.rect), "model of A3 with instanceCount 1 in the middle command (a triangle in the gap) does not match");
        }

        void check_indirect_buffer_state(demo::Check& check) const
        {
            check.expect_eq(initial_indirect_binding_, 0, "DRAW_INDIRECT_BUFFER_BINDING is 0 in the initial state (Table 21.4)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding), 0, "DRAW_INDIRECT_BUFFER_BINDING is 0 after the frame");
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, indirect_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding), static_cast<int>(indirect_.value),
                            "DRAW_INDIRECT_BUFFER_BINDING returns the buffer after glBindBuffer(DRAW_INDIRECT_BUFFER)");
            // The binding is context state, not VAO state (Table 21.4 "not in vertex array objects"): VAO switches keep it.
            mgl::glBindVertexArray(vao_[kVaoMain]);
            const int with_vao = demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding);
            mgl::glBindVertexArray(vao_[kVaoNoElem]);
            const int with_other_vao = demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            const int with_default_vao = demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding);
            check.expect(with_vao == static_cast<int>(indirect_.value) && with_other_vao == with_vao && with_default_vao == with_vao,
                         "DRAW_INDIRECT_BUFFER_BINDING is not VAO state: unchanged by switching between two VAOs and the default VAO",
                         std::format("{} / {} / {}", with_vao, with_other_vao, with_default_vao));
            GLint size = 0;
            mgl::glGetBufferParameteriv(mgl::BufferTarget::DrawIndirect, mgl::BufferParameter::Size, &size);
            check.expect_eq(static_cast<std::size_t>(size), buffer_size_, "the command buffer holds 18 commands + guards, BUFFER_SIZE as uploaded");
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            check.expect_eq(demo::get_integer(mgl::GetParameter::DrawIndirectBufferBinding), 0, "DRAW_INDIRECT_BUFFER_BINDING is 0 after binding buffer 0");
            check.expect_no_gl_error("the state queries raise no error");
        }

        void expect_arrays(demo::Check& check, mgl::ErrorCode code, std::size_t offset, std::string_view what) const
        {
            (void)demo::drain_gl_errors();
            mgl::glDrawArraysIndirect(mgl::PrimitiveType::Triangles, demo::buffer_offset(offset));
            if (code == mgl::ErrorCode::NoError) check.expect_no_gl_error(what);
            else check.expect_gl_error(code, what);
        }
        void expect_elements(demo::Check& check, mgl::ErrorCode code, std::size_t offset, std::string_view what) const
        {
            (void)demo::drain_gl_errors();
            mgl::glDrawElementsIndirect(mgl::PrimitiveType::Triangles, mgl::IndexType::UnsignedShort, demo::buffer_offset(offset));
            if (code == mgl::ErrorCode::NoError) check.expect_no_gl_error(what);
            else check.expect_gl_error(code, what);
        }

        // Every error condition is isolated: all other requirements of the command are met. Valid control calls use the
        // zero-instance commands, so nothing visible is drawn.
        void check_errors(demo::Check& check, demo::Version version) const
        {
            using Ec = mgl::ErrorCode;
            mgl::glUseProgram(program_);
            (void)demo::drain_gl_errors();

            // -- baseline: a valid draw of an empty command from the main VAO raises no error ---------------------------------
            mgl::glBindVertexArray(vao_[kVaoMain]);
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, indirect_);
            expect_arrays(check, Ec::NoError, off_az_, "glDrawArraysIndirect with a zero-instance command (instanceCount 0) raises no error");
            expect_elements(check, Ec::NoError, off_tail_, "glDrawElementsIndirect with a zero-instance command raises no error");

            // -- DRAW_INDIRECT_BUFFER is zero ----------------------------------------------------------------------------------
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            expect_arrays(check, Ec::InvalidOperation, 0, "glDrawArraysIndirect with DRAW_INDIRECT_BUFFER = 0 -> INVALID_OPERATION");
            expect_elements(check, Ec::InvalidOperation, 0, "glDrawElementsIndirect with DRAW_INDIRECT_BUFFER = 0 -> INVALID_OPERATION");
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, indirect_);

            // -- indirect is not a multiple of sizeof(uint) --------------------------------------------------------------------
            for (const std::size_t bad : {std::size_t{1}, std::size_t{2}, std::size_t{3}})
            {
                expect_arrays(check, Ec::InvalidValue, off_az_ + bad, std::format("glDrawArraysIndirect with indirect = {} (valid range, misaligned) -> INVALID_VALUE", off_az_ + bad));
                expect_elements(check, Ec::InvalidValue, off_e0_ + bad, std::format("glDrawElementsIndirect with indirect = {} (misaligned) -> INVALID_VALUE", off_e0_ + bad));
            }

            // -- the command would be read beyond the end of the buffer ----------------------------------------------------------
            expect_arrays(check, Ec::NoError, buffer_size_ - 16, "glDrawArraysIndirect with the command exactly at the end (size - 16) raises no error");
            expect_arrays(check, Ec::InvalidOperation, buffer_size_ - 12, "glDrawArraysIndirect with only 12 bytes left (size - 12) -> INVALID_OPERATION");
            expect_arrays(check, Ec::InvalidOperation, buffer_size_, "glDrawArraysIndirect at indirect = size -> INVALID_OPERATION");
            expect_arrays(check, Ec::InvalidOperation, buffer_size_ + 1000 * 4, "glDrawArraysIndirect far beyond the buffer -> INVALID_OPERATION");
            expect_elements(check, Ec::NoError, buffer_size_ - 20, "glDrawElementsIndirect with the command exactly at the end (size - 20) raises no error");
            expect_elements(check, Ec::InvalidOperation, buffer_size_ - 16, "glDrawElementsIndirect with only 16 of its 20 bytes inside the buffer -> INVALID_OPERATION");
            expect_elements(check, Ec::InvalidOperation, buffer_size_, "glDrawElementsIndirect at indirect = size -> INVALID_OPERATION");

            // -- the default vertex array object is bound -------------------------------------------------------------------------
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            expect_arrays(check, Ec::InvalidOperation, off_az_, "glDrawArraysIndirect with the default VAO bound -> INVALID_OPERATION");
            expect_elements(check, Ec::InvalidOperation, off_tail_, "glDrawElementsIndirect with the default VAO bound -> INVALID_OPERATION");

            // -- an enabled vertex array whose buffer binding is zero ---------------------------------------------------------------
            mgl::glBindVertexArray(vao_[kVaoZero]);
            check.expect_no_gl_error("glVertexAttribPointer with a NULL pointer and ARRAY_BUFFER 0 on a non-default VAO is allowed (VAO setup)");
            expect_arrays(check, Ec::InvalidOperation, off_az_, "glDrawArraysIndirect with an enabled attribute array without a buffer -> INVALID_OPERATION");
            expect_elements(check, Ec::InvalidOperation, off_tail_, "glDrawElementsIndirect with an enabled attribute array without a buffer -> INVALID_OPERATION");
            mgl::glDisableVertexAttribArray(mgl::AttribLocation{0u});
            expect_arrays(check, Ec::NoError, off_az_, "the same glDrawArraysIndirect after glDisableVertexAttribArray(0) raises no error");
            expect_elements(check, Ec::NoError, off_tail_, "the same glDrawElementsIndirect after glDisableVertexAttribArray(0) raises no error");

            // -- DrawElementsIndirect without an ELEMENT_ARRAY_BUFFER ----------------------------------------------------------------
            mgl::glBindVertexArray(vao_[kVaoNoElem]);
            expect_elements(check, Ec::InvalidOperation, off_tail_, "glDrawElementsIndirect with ELEMENT_ARRAY_BUFFER = 0 in the VAO -> INVALID_OPERATION");
            expect_arrays(check, Ec::NoError, off_az_, "glDrawArraysIndirect needs no element buffer: the same VAO raises no error");

            // -- invalid index type (INVALID_ENUM through the DrawElementsInstancedBaseVertex equivalence) --------------------------
            mgl::glBindVertexArray(vao_[kVaoMain]);
            (void)demo::drain_gl_errors();
            mgl::glDrawElementsIndirect(mgl::PrimitiveType::Triangles, mgl::DataType::Float, demo::buffer_offset(off_tail_));
            check.expect_gl_error(Ec::InvalidEnum, "glDrawElementsIndirect with type FLOAT -> INVALID_ENUM");

            // -- transform feedback active and not paused ---------------------------------------------------------------------------
            check_transform_feedback_error(check, version);

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            check.expect_no_gl_error("no stray error remains after the error checks");
        }

        // Transform feedback and indirect draws. ES 3.1 section 10.5 (and 12.1.2): "An INVALID_OPERATION error is generated if
        // transform feedback is active and not paused", whatever the mode. ES 3.2 relaxed this (section 12.2.2): a draw is an error
        // only if its mode is not allowed by Table 12.1 for the primitiveMode of BeginTransformFeedback; any mode may be used while
        // paused. The mismatching-mode and paused cases hold in both versions. The matching-mode case depends on the version of the
        // context: Mesa applies the ES 3.2 behaviour in every ES context (also a native ES 3.1 one and under MESA_GLES_VERSION_OVERRIDE=3.1), reported as a known deviation.
        void check_transform_feedback_error(demo::Check& check, demo::Version version) const
        {
            using Ec = mgl::ErrorCode;
            mgl::glUseProgram(tf_program_);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Triangles);

            (void)demo::drain_gl_errors();
            mgl::glDrawArraysIndirect(mgl::PrimitiveType::Points, demo::buffer_offset(off_az_));
            check.expect_gl_error(Ec::InvalidOperation, "glDrawArraysIndirect(POINTS) while transform feedback records TRIANGLES -> INVALID_OPERATION");
            mgl::glDrawElementsIndirect(mgl::PrimitiveType::Points, mgl::IndexType::UnsignedShort, demo::buffer_offset(off_tail_));
            check.expect_gl_error(Ec::InvalidOperation, "glDrawElementsIndirect(POINTS) while transform feedback records TRIANGLES -> INVALID_OPERATION");

            const bool es31 = version.major == 3 && version.minor == 1;
            (void)demo::drain_gl_errors();
            mgl::glDrawArraysIndirect(mgl::PrimitiveType::Triangles, demo::buffer_offset(off_az_));
            const mgl::ErrorCode arrays_error = mgl::glGetError();
            mgl::glDrawElementsIndirect(mgl::PrimitiveType::Triangles, mgl::IndexType::UnsignedShort, demo::buffer_offset(off_tail_));
            const mgl::ErrorCode elements_error = mgl::glGetError();
            (void)demo::drain_gl_errors();
            const Ec want = es31 ? Ec::InvalidOperation : Ec::NoError;
            const char* rule = es31 ? "ES 3.1 section 10.5: error while transform feedback is active and not paused"
                                    : "ES 3.2 section 12.2.2: TRIANGLES is allowed while recording TRIANGLES";
            // Under ES 3.1 the strict rule is asserted, with the documented Mesa deviation; under ES 3.2 the ES 3.2 rule is asserted
            // without any exception.
            auto expect_rule = [&](bool ok, std::string_view call, mgl::ErrorCode got) {
                const std::string what = std::format("{} with TRIANGLES feedback active: {}", call, rule);
                const std::string detail = std::format("glGetError() = {}, expected {}", demo::error_name(got), demo::error_name(want));
                if (es31)
                    check.expect_or_known_deviation(ok, what, detail, "Mesa", "Mesa applies the ES 3.2 rule in every ES context, also in a native ES 3.1 one (softpipe) and under MESA_GLES_VERSION_OVERRIDE=3.1");
                else
                    check.expect(ok, what, ok ? std::string{} : detail);
            };
            expect_rule(arrays_error == want, "glDrawArraysIndirect(TRIANGLES)", arrays_error);
            expect_rule(elements_error == want, "glDrawElementsIndirect(TRIANGLES)", elements_error);

            mgl::glPauseTransformFeedback();
            expect_arrays(check, Ec::NoError, off_az_, "glDrawArraysIndirect while transform feedback is paused raises no error");
            expect_elements(check, Ec::NoError, off_tail_, "glDrawElementsIndirect while transform feedback is paused raises no error");
            mgl::glResumeTransformFeedback();
            mgl::glEndTransformFeedback();
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
            mgl::glUseProgram(program_);
            (void)demo::drain_gl_errors();
        }

        // ---------------------------------------------------------------------------------------------
        // Members
        // ---------------------------------------------------------------------------------------------
        mgl::ProgramId program_{}, compute_{}, tf_program_{};
        mgl::UniformLocation u_mode_{}, u_scale_{}, u_origin_{}, u_pitch_{}, u_arrays_word_{}, u_elements_word_{};
        std::array<mgl::VertexArrayId, kVaoCount> vao_{};
        std::vector<mgl::BufferId> buffers_;
        mgl::BufferId indirect_{}, flags_{}, pool_{}, palette_{}, ebo_ubyte_{}, ebo_ushort_{}, ebo_uint_{}, tf_buffer_{};
        std::vector<std::uint32_t> words_;
        std::size_t buffer_size_ = 0;
        std::size_t off_a0_ = 0, off_a1_ = 0, off_a2_ = 0, off_az_ = 0, off_e0_ = 0, off_e1a_ = 0, off_e1b_ = 0, off_e2_ = 0, off_e3_ = 0, off_t0_ = 0,
                    off_t1_ = 0, off_v0_ = 0, off_v1_ = 0, off_d0_ = 0, off_d1_ = 0, off_c0_ = 0, off_c1_ = 0, off_tail_ = 0;
        std::vector<Region> regions_;
        int initial_indirect_binding_ = -1;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-indirect-draw",
        .title = "Indirect drawing: glDrawArraysIndirect / glDrawElementsIndirect",
        .min_es = {3, 1},
        .summary = "Draw commands are read from a DRAW_INDIRECT_BUFFER at byte offsets: arrays and elements commands with first, "
                   "firstIndex, baseVertex and instanceCount, rewritten by glBufferSubData and by a compute shader, compared with a "
                   "CPU model, plus every error the specification defines.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<IndirectDraw>(argc, argv, info);
}
