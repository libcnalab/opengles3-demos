// es31-compute-basics
//
// OpenGL ES 3.1 compute shaders from the ground up (ES 3.2 chapter 17, GLSL ES 3.10 sections 4.4.1.1 and 7.1.3):
// glDispatchCompute with 1-D, 2-D and 3-D work group grids and local sizes, every compute built-in written into a
// shader storage buffer by every invocation and compared exactly with a CPU model of the specification's formulas,
// the work group limits and the errors around dispatching, and three image-free parallel array computations
// (vector addition, a dependent second pass, a 2-D transpose).
//
// The image shows the SSBO contents. Each of the four panels is one dispatch configuration; every invocation of it
// is one cell, positioned by the recorded gl_GlobalInvocationID and colored by the recorded gl_WorkGroupID (hue)
// and gl_LocalInvocationIndex (brightness). The cells are drawn from the very buffer the compute shader wrote,
// fetched as per-instance vertex attributes after glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT).
//
// validate() reads the buffers back with glMapBufferRange after glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT) and
// compares them exactly; it then samples the center of every cell of the image against the model.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr std::uint32_t kSentinel = 0xDEADBEEFu;  // pre-fill of every result buffer: "never written"
    constexpr std::uint32_t kMaxExercisedCount = 1u << 20;  // largest work group count dispatched at the limit
    constexpr std::uint32_t kMaxLimitInvocations = 4096;    // cap of the "all invocations of a group" test

    using U3 = std::array<std::uint32_t, 3>;

    std::string to_string(const U3& v) { return std::format("({}, {}, {})", v[0], v[1], v[2]); }

    // ---------------------------------------------------------------------------------------------------------
    // The record every invocation writes (std430: four uvec4, 64 bytes) and the CPU model of it.
    // ---------------------------------------------------------------------------------------------------------
    struct Rec
    {
        std::array<std::uint32_t, 4> num_groups;  // gl_NumWorkGroups, gl_LocalInvocationIndex
        std::array<std::uint32_t, 4> group_id;    // gl_WorkGroupID, gl_WorkGroupSize.x
        std::array<std::uint32_t, 4> local_id;    // gl_LocalInvocationID, gl_WorkGroupSize.y
        std::array<std::uint32_t, 4> global_id;   // gl_GlobalInvocationID, gl_WorkGroupSize.z
        friend bool operator==(const Rec&, const Rec&) = default;
    };
    static_assert(sizeof(Rec) == 64);
    constexpr Rec kSentinelRec = {{kSentinel, kSentinel, kSentinel, kSentinel},
                                  {kSentinel, kSentinel, kSentinel, kSentinel},
                                  {kSentinel, kSentinel, kSentinel, kSentinel},
                                  {kSentinel, kSentinel, kSentinel, kSentinel}};

    // One dispatch configuration: the grid of work groups, and the local size the shader's layout qualifier
    // must produce (dimensions that are not named in the qualifier default to 1).
    struct Config
    {
        std::string name;
        U3 groups;
        U3 local;
        std::string local_macro;  // text of "#define LOCAL_SIZE ..."
        [[nodiscard]] std::size_t invocations() const
        {
            return std::size_t{groups[0]} * groups[1] * groups[2] * local[0] * local[1] * local[2];
        }
        [[nodiscard]] std::size_t local_count() const { return std::size_t{local[0]} * local[1] * local[2]; }
        [[nodiscard]] U3 global_size() const
        {
            return {groups[0] * local[0], groups[1] * local[1], groups[2] * local[2]};
        }
    };

    // CPU model of what builtins.comp records, straight from the definitions in GLSL ES 3.10 section 7.1.3:
    //   gl_GlobalInvocationID    = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID
    //   gl_LocalInvocationIndex  = id.z * size.x * size.y + id.y * size.x + id.x
    // The record is stored at the flat position of the global invocation ID (x fastest, then y, then z).
    std::vector<Rec> model_records(const Config& c)
    {
        std::vector<Rec> out(c.invocations(), kSentinelRec);
        const U3 gs = c.global_size();
        for (std::uint32_t wz = 0; wz < c.groups[2]; ++wz)
            for (std::uint32_t wy = 0; wy < c.groups[1]; ++wy)
                for (std::uint32_t wx = 0; wx < c.groups[0]; ++wx)
                    for (std::uint32_t lz = 0; lz < c.local[2]; ++lz)
                        for (std::uint32_t ly = 0; ly < c.local[1]; ++ly)
                            for (std::uint32_t lx = 0; lx < c.local[0]; ++lx)
                            {
                                const U3 global = {wx * c.local[0] + lx, wy * c.local[1] + ly, wz * c.local[2] + lz};
                                const std::uint32_t index = lz * c.local[0] * c.local[1] + ly * c.local[0] + lx;
                                const std::size_t slot = std::size_t{global[0]} + std::size_t{global[1]} * gs[0] +
                                                         std::size_t{global[2]} * gs[0] * gs[1];
                                out[slot] = Rec{{c.groups[0], c.groups[1], c.groups[2], index},
                                                {wx, wy, wz, c.local[0]},
                                                {lx, ly, lz, c.local[1]},
                                                {global[0], global[1], global[2], c.local[2]}};
                            }
        return out;
    }

    std::string to_string(const Rec& r)
    {
        auto v4 = [](const std::array<std::uint32_t, 4>& a) {
            return std::format("({}, {}, {}, {})", a[0], a[1], a[2], a[3]);
        };
        return std::format("{{num_groups {} group_id {} local_id {} global_id {}}}", v4(r.num_groups), v4(r.group_id),
                           v4(r.local_id), v4(r.global_id));
    }

    struct Diff
    {
        std::size_t bad = 0;
        std::string first;
    };
    Diff compare_records(const std::vector<Rec>& got, const std::vector<Rec>& want)
    {
        Diff d;
        if (got.size() != want.size())
        {
            d.bad = std::max(got.size(), want.size());
            d.first = std::format("size {} vs {}", got.size(), want.size());
            return d;
        }
        for (std::size_t i = 0; i < got.size(); ++i)
            if (!(got[i] == want[i]) && d.bad++ == 0)
                d.first = std::format("record {}: got {} want {}", i, to_string(got[i]), to_string(want[i]));
        return d;
    }

    // Compares uint32 words; returns the count of mismatches and describes the first.
    Diff compare_words(const std::vector<std::uint32_t>& got, const std::vector<std::uint32_t>& want)
    {
        Diff d;
        if (got.size() != want.size())
        {
            d.bad = std::max(got.size(), want.size());
            d.first = std::format("size {} vs {}", got.size(), want.size());
            return d;
        }
        for (std::size_t i = 0; i < got.size(); ++i)
            if (got[i] != want[i] && d.bad++ == 0)
                d.first = std::format("word {}: got {:#x} want {:#x}", i, got[i], want[i]);
        return d;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Shader helpers
    // ---------------------------------------------------------------------------------------------------------

    // Inserts "#define <text>" lines directly below the "#version 310 es" line (which must stay the first line).
    std::string with_defines(const std::string& source, std::initializer_list<std::string> defines)
    {
        const std::size_t eol = source.find('\n');
        std::string head = source.substr(0, eol + 1);
        for (const std::string& d : defines) head += "#define " + d + "\n";
        return head + source.substr(eol + 1);
    }

    mgl::ProgramId build_compute(const std::string& source, const std::string& label)
    {
        std::string log;
        const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, source, log);
        if (cs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", label, log));
        const mgl::ShaderId shaders[] = {cs};
        const mgl::ProgramId program = demo::link_program(shaders, log);
        mgl::glDeleteShader(cs);
        if (program.value == 0) throw demo::Failure(std::format("{} failed to link:\n{}", label, log));
        return program;
    }

    // First line of an info log, for the detail column of a check.
    std::string first_line(std::string log)
    {
        const std::size_t eol = log.find('\n');
        if (eol != std::string::npos) log.resize(eol);
        return log;
    }

    mgl::UniformLocation uniform(mgl::ProgramId program, const char* name)
    {
        const mgl::UniformLocation loc = mgl::glGetUniformLocation(program, name);
        if (loc.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
        return loc;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Buffer helpers (SHADER_STORAGE_BUFFER is only the binding used to create / map them)
    // ---------------------------------------------------------------------------------------------------------
    template <typename T>
    mgl::BufferId create_ssbo(const std::vector<T>& data)
    {
        mgl::BufferId id{};
        mgl::glGenBuffers(1, &id);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, id);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const T>(data), mgl::BufferUsage::DynamicCopy);
        return id;
    }

    // Maps the buffer for reading and copies `count` elements. The caller must have issued
    // glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT) after the last shader write (ES 3.2 section 7.11.2).
    template <typename T>
    std::vector<T> read_ssbo(mgl::BufferId id, std::size_t count)
    {
        std::vector<T> out(count);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, id);
        const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0,
                                                   static_cast<GLsizeiptr>(count * sizeof(T)), mgl::MapBufferAccessMask::Read);
        if (mapped == nullptr) return {};
        std::memcpy(out.data(), mapped, count * sizeof(T));
        (void)mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage);
        return out;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Presentation: where a cell goes and what color it gets (mirrors display.vert)
    // ---------------------------------------------------------------------------------------------------------
    constexpr std::array<std::array<unsigned, 3>, 12> kPalette = {{{235, 64, 64},   {240, 140, 40}, {235, 220, 60},
                                                                    {150, 220, 60},  {60, 200, 90},   {50, 200, 190},
                                                                    {60, 160, 235},  {80, 90, 240},   {150, 80, 235},
                                                                    {225, 80, 205},  {235, 235, 235}, {150, 110, 70}}};
    constexpr Rgba8 kBackground = {8, 8, 12, 255};
    constexpr Rgba8 kPanel = {30, 32, 42, 255};

    struct Layout
    {
        int tiles_x = 1;                // z slices per row of tiles
        int gx = 0, gy = 0;             // cells per slice
        int wrap = 0;                   // row length of a 1-D grid (each row is one work group), 0 = none
        int cell = 2;                   // pixels per cell
        int origin_x = 0, origin_y = 0; // window position of cell (0, 0)
        int panel_x = 0, panel_y = 0, panel_w = 0, panel_h = 0;
    };

    Layout make_layout(const Config& c, int panel_x, int panel_y, int panel_w, int panel_h)
    {
        const U3 g = c.global_size();
        Layout l;
        l.panel_x = panel_x;
        l.panel_y = panel_y;
        l.panel_w = panel_w;
        l.panel_h = panel_h;
        int w = 0, h = 0;
        if (g[1] == 1 && g[2] == 1)
        {
            l.wrap = static_cast<int>(c.local[0]);
            w = l.wrap;
            h = (static_cast<int>(g[0]) + l.wrap - 1) / l.wrap;
        }
        else
        {
            l.gx = static_cast<int>(g[0]);
            l.gy = static_cast<int>(g[1]);
            while (l.tiles_x * l.tiles_x < static_cast<int>(g[2])) ++l.tiles_x;
            const int tiles_y = (static_cast<int>(g[2]) + l.tiles_x - 1) / l.tiles_x;
            w = l.tiles_x * l.gx + (l.tiles_x - 1);
            h = tiles_y * l.gy + (tiles_y - 1);
        }
        l.cell = std::max(3, std::min((panel_w - 8) / w, (panel_h - 8) / h));
        l.origin_x = panel_x + (panel_w - w * l.cell) / 2;
        l.origin_y = panel_y + (panel_h - h * l.cell) / 2;
        return l;
    }

    std::array<int, 2> cell_of(const Layout& l, const std::array<std::uint32_t, 4>& global_id)
    {
        const int x = static_cast<int>(global_id[0]), y = static_cast<int>(global_id[1]), z = static_cast<int>(global_id[2]);
        if (l.wrap > 0) return {x % l.wrap, x / l.wrap};
        return {x + (z % l.tiles_x) * (l.gx + 1), y + (z / l.tiles_x) * (l.gy + 1)};
    }

    // hue_shift != 0 builds the "mutated" model of the negative control.
    Rgba8 expected_color(const Rec& r, std::size_t local_count, unsigned hue_shift = 0)
    {
        const unsigned group_flat = r.group_id[0] + r.num_groups[0] * (r.group_id[1] + r.num_groups[1] * r.group_id[2]);
        const auto& pal = kPalette[(group_flat + hue_shift) % 12];
        const unsigned brightness =
            96u + (159u * r.num_groups[3]) / static_cast<unsigned>(std::max<std::size_t>(local_count - 1, 1));
        return {static_cast<std::uint8_t>(pal[0] * brightness / 255u), static_cast<std::uint8_t>(pal[1] * brightness / 255u),
                static_cast<std::uint8_t>(pal[2] * brightness / 255u), 255};
    }

    // ---------------------------------------------------------------------------------------------------------
    // The demo
    // ---------------------------------------------------------------------------------------------------------
    struct Run
    {
        Config cfg;
        mgl::ProgramId program{};
        mgl::UniformLocation u_capacity{};
        mgl::BufferId records{};
        mgl::VertexArrayId vao{};
        Layout layout{};
        std::array<GLint, 3> queried_size{};
    };

    class ComputeBasics final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // Limits first: the maximum-invocation configuration depends on them.
            for (GLuint d = 0; d < 3; ++d)
            {
                mgl::glGetIntegeri_v(mgl::GetParameter::MaxComputeWorkGroupCount, d, &count_limit_[d]);
                mgl::glGetIntegeri_v(mgl::GetParameter::MaxComputeWorkGroupSize, d, &size_limit_[d]);
            }
            invocation_limit_ = demo::get_integer(mgl::GetParameter::MaxComputeWorkGroupInvocations);

            const std::string builtins_src = env.load_text("shaders/builtins.comp");
            std::vector<Config> configs = {
                {"1-D: 5 groups of 8", {5, 1, 1}, {8, 1, 1}, "local_size_x = 8"},
                {"2-D: 3x2 groups of 4x4", {3, 2, 1}, {4, 4, 1}, "local_size_x = 4, local_size_y = 4"},
                {"3-D: 2x2x2 groups of 4x2x2", {2, 2, 2}, {4, 2, 2}, "local_size_x = 4, local_size_y = 2, local_size_z = 2"},
                {"3-D odd: 2x2x3 groups of 3x5x2", {2, 2, 3}, {3, 5, 2},
                 "local_size_x = 3, local_size_y = 5, local_size_z = 2"},
            };
            for (const Config& c : configs) runs_.push_back(make_run(builtins_src, c));

            // One more configuration: a work group with as many invocations as the implementation allows
            // (capped), spread over the dimensions within the per-dimension limits.
            if (invocation_limit_ >= 1 && size_limit_[0] >= 1)
            {
                const std::uint32_t inv = std::min<std::uint32_t>(static_cast<std::uint32_t>(invocation_limit_), kMaxLimitInvocations);
                const std::uint32_t x = std::min<std::uint32_t>(static_cast<std::uint32_t>(size_limit_[0]), inv);
                const std::uint32_t y = std::max<std::uint32_t>(1, std::min<std::uint32_t>(static_cast<std::uint32_t>(size_limit_[1]), inv / x));
                const std::uint32_t z = std::max<std::uint32_t>(1, std::min<std::uint32_t>(static_cast<std::uint32_t>(size_limit_[2]), inv / (x * y)));
                limit_cfg_ = Config{"maximum work group", {3, 1, 1}, {x, y, z},
                                    std::format("local_size_x = {}, local_size_y = {}, local_size_z = {}", x, y, z)};
                limit_run_ = make_run(builtins_src, limit_cfg_);
                has_limit_run_ = true;
            }

            // Programs of the array computations.
            vecadd_ = build_compute(env.load_text("shaders/vecadd.comp"), "vecadd.comp");
            transform_ = build_compute(env.load_text("shaders/transform.comp"), "transform.comp");
            transpose_ = build_compute(env.load_text("shaders/transpose.comp"), "transpose.comp");
            hits_ = build_compute(env.load_text("shaders/hits.comp"), "hits.comp");
            display_ = demo::build_program(env, "shaders/display.vert", "shaders/display.frag");
            u_target_ = uniform(display_, "u_target");
            u_origin_ = uniform(display_, "u_origin");
            u_cell_ = uniform(display_, "u_cell");
            u_layout_ = uniform(display_, "u_layout");
            u_local_count_ = uniform(display_, "u_local_count");

            // The display VAOs: the three attributes of an instance are views into the 64-byte record.
            for (Run& r : runs_)
            {
                mgl::glGenVertexArrays(1, &r.vao);
                mgl::glBindVertexArray(r.vao);
                mgl::glBindBuffer(mgl::BufferTarget::Array, r.records);
                const struct { GLuint location; std::uintptr_t offset; } attributes[] = {
                    {0, offsetof(Rec, num_groups)}, {1, offsetof(Rec, group_id)}, {2, offsetof(Rec, global_id)}};
                for (const auto& a : attributes)
                {
                    const mgl::AttribLocation loc{a.location};
                    mgl::glEnableVertexAttribArray(loc);
                    mgl::glVertexAttribIPointer(loc, 4, mgl::DataType::UnsignedInt, sizeof(Rec), demo::buffer_offset(a.offset));
                    mgl::glVertexAttribDivisor(loc, 1);  // one record per instance
                }
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            // 1. The compute passes: one dispatch per panel.
            for (const Run& r : runs_) dispatch_builtins(r);

            // 2. Make the shader writes visible to the vertex attribute fetch of the draws below.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::VertexAttribArray);

            // 3. Panels (scissored clears) and cells.
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(f(kBackground[0]), f(kBackground[1]), f(kBackground[2]), 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            const int pw = env.width / 2 - 4, ph = env.height / 2 - 4;
            for (std::size_t i = 0; i < runs_.size(); ++i)
            {
                const int qx = static_cast<int>(i % 2), qy = 1 - static_cast<int>(i / 2);  // reading order, top row first
                runs_[i].layout = make_layout(runs_[i].cfg, qx * (env.width / 2) + 2, qy * (env.height / 2) + 2, pw, ph);
            }
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glClearColor(f(kPanel[0]), f(kPanel[1]), f(kPanel[2]), 1.0f);
            for (const Run& r : runs_)
            {
                mgl::glScissor(r.layout.panel_x, r.layout.panel_y, r.layout.panel_w, r.layout.panel_h);
                mgl::glClear(mgl::ClearBufferBit::Color);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);

            mgl::glUseProgram(display_);
            mgl::glUniform2i(u_target_, env.width, env.height);
            for (const Run& r : runs_)
            {
                const Layout& l = r.layout;
                mgl::glUniform2i(u_origin_, l.origin_x, l.origin_y);
                mgl::glUniform1i(u_cell_, l.cell);
                mgl::glUniform4i(u_layout_, l.tiles_x, l.gx, l.gy, l.wrap);
                // The brightness ramp uses the work group size as the API reports it, not the CPU's copy.
                mgl::glUniform1ui(u_local_count_, static_cast<GLuint>(r.queried_size[0] * r.queried_size[1] * r.queried_size[2]));
                mgl::glBindVertexArray(r.vao);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, static_cast<GLsizei>(r.cfg.invocations()));
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_limits(check);
            check_builtins(check);
            check_program_queries(check);
            check_dispatch_errors(check);
            check_negative_shaders(env, check);
            check_array_computations(check);
            check_image(env, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            for (const Run& r : runs_) delete_run(r);
            if (has_limit_run_) delete_run(limit_run_);
            for (const mgl::ProgramId p : {vecadd_, transform_, transpose_, hits_, display_})
                if (p.value != 0) mgl::glDeleteProgram(p);
            if (!extra_buffers_.empty()) mgl::glDeleteBuffers(std::span<const mgl::BufferId>(extra_buffers_));
            extra_buffers_.clear();
            // Indexed binding points keep a reference until they are reset: unbind what the demo bound.
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, i, mgl::BufferId{0});
        }

    private:
        static float f(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

        Run make_run(const std::string& builtins_src, const Config& c)
        {
            Run r;
            r.cfg = c;
            r.program = build_compute(with_defines(builtins_src, {"LOCAL_SIZE " + c.local_macro}), "builtins.comp [" + c.name + "]");
            r.u_capacity = uniform(r.program, "u_capacity");
            // One record per invocation plus one guard record that no invocation may touch.
            const std::vector<Rec> init(c.invocations() + 1, kSentinelRec);
            r.records = create_ssbo(init);
            mgl::glGetProgramiv(r.program, mgl::ProgramParameter::ComputeWorkGroupSize, r.queried_size.data());
            return r;
        }

        static void delete_run(const Run& r)
        {
            if (r.vao.value != 0) mgl::glDeleteVertexArrays(1, &r.vao);
            if (r.records.value != 0) mgl::glDeleteBuffers(1, &r.records);
            if (r.program.value != 0) mgl::glDeleteProgram(r.program);
        }

        // Refills the record buffer with the sentinel and dispatches the configuration's grid.
        static void dispatch_builtins(const Run& r)
        {
            const std::vector<Rec> init(r.cfg.invocations() + 1, kSentinelRec);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, r.records);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const Rec>(init), mgl::BufferUsage::DynamicCopy);
            mgl::glUseProgram(r.program);
            mgl::glUniform1ui(r.u_capacity, static_cast<GLuint>(r.cfg.invocations()));
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, r.records);
            mgl::glDispatchCompute(r.cfg.groups[0], r.cfg.groups[1], r.cfg.groups[2]);
        }

        // Reads the record buffer of a run (invocations + the guard record).
        static std::vector<Rec> read_records(const Run& r)
        {
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            return read_ssbo<Rec>(r.records, r.cfg.invocations() + 1);
        }

        // ---- limits ---------------------------------------------------------------------------------------
        void check_limits(demo::Check& check)
        {
            constexpr std::array<int, 3> kMinCount = {65535, 65535, 65535};
            constexpr std::array<int, 3> kMinSize = {128, 128, 64};
            const char axis[] = {'x', 'y', 'z'};
            for (std::size_t d = 0; d < 3; ++d)
            {
                check.expect(count_limit_[d] >= kMinCount[d],
                             std::format("MAX_COMPUTE_WORK_GROUP_COUNT[{}] = {} >= {} (ES 3.2 Table 21.49)", axis[d], count_limit_[d], kMinCount[d]));
                check.expect(size_limit_[d] >= kMinSize[d],
                             std::format("MAX_COMPUTE_WORK_GROUP_SIZE[{}] = {} >= {} (ES 3.2 Table 21.49)", axis[d], size_limit_[d], kMinSize[d]));
            }
            check.expect(invocation_limit_ >= 128,
                         std::format("MAX_COMPUTE_WORK_GROUP_INVOCATIONS = {} >= 128 (ES 3.2 Table 21.49)", invocation_limit_));

            // glGetIntegeri_v with an index outside [0, 2]: INVALID_VALUE, and "no change is made" to the output.
            demo::drain_gl_errors();
            GLint untouched = -77;
            mgl::glGetIntegeri_v(mgl::GetParameter::MaxComputeWorkGroupCount, 3, &untouched);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(MAX_COMPUTE_WORK_GROUP_COUNT, index 3) is INVALID_VALUE (ES 3.2 section 20.1)");
            check.expect_eq(untouched, -77, "the failed query left its output untouched (ES 3.2 section 2.3.1)");
            std::cout << std::format("   note: limits count=({}, {}, {}) size=({}, {}, {}) invocations={}\n", count_limit_[0], count_limit_[1],
                                     count_limit_[2], size_limit_[0], size_limit_[1], size_limit_[2], invocation_limit_);
        }

        // ---- built-ins ------------------------------------------------------------------------------------
        void check_one_config(demo::Check& check, const Run& r, const char* label)
        {
            const std::vector<Rec> got = read_records(r);
            const std::vector<Rec> want = model_records(r.cfg);
            // got has the guard record at the end.
            const std::vector<Rec> body(got.begin(), got.end() - 1);
            const Diff d = compare_records(body, want);
            check.expect(d.bad == 0,
                         std::format("{}: all {} records equal the model (gl_NumWorkGroups, gl_WorkGroupID, gl_LocalInvocationID, "
                                     "gl_GlobalInvocationID, gl_LocalInvocationIndex, gl_WorkGroupSize), grid {} x local {}",
                                     label, want.size(), to_string(r.cfg.groups), to_string(r.cfg.local)),
                         d.bad == 0 ? "" : std::format("{} bad; {}", d.bad, d.first));
            check.expect(got.back() == kSentinelRec, std::format("{}: the guard record past the last invocation is untouched", label));
        }

        void check_builtins(demo::Check& check)
        {
            for (const Run& r : runs_) check_one_config(check, r, r.cfg.name.c_str());

            if (has_limit_run_)
            {
                dispatch_builtins(limit_run_);
                check_one_config(check, limit_run_,
                                 std::format("maximum work group ({} invocations as {})", limit_cfg_.local_count(), to_string(limit_cfg_.local)).c_str());
                check.expect(limit_cfg_.local_count() <= static_cast<std::size_t>(invocation_limit_),
                             "the maximum work group stays within MAX_COMPUTE_WORK_GROUP_INVOCATIONS");
                std::cout << std::format("   note: maximum work group used {} invocations per group ({} requested limit, cap {})\n",
                                         limit_cfg_.local_count(), invocation_limit_, kMaxLimitInvocations);
            }

            // Negative controls: the comparison must be able to fail. They use CPU data only, so a wrong GPU result
            // cannot turn them into a second, confusing failure.
            // (a) one corrupted field in one record is found, and only that record.
            const Run& r0 = runs_[1];
            std::vector<Rec> corrupted = model_records(r0.cfg);
            corrupted[corrupted.size() / 2].local_id[1] += 1;
            const Diff d1 = compare_records(corrupted, model_records(r0.cfg));
            check.expect_eq(d1.bad, std::size_t{1}, "negative control: one flipped gl_LocalInvocationID.y is reported as exactly one bad record");
            // (b) a never-dispatched buffer (every record still the sentinel) does not match the model anywhere.
            const std::vector<Rec> untouched(r0.cfg.invocations(), kSentinelRec);
            const Diff d2 = compare_records(untouched, model_records(r0.cfg));
            check.expect_eq(d2.bad, r0.cfg.invocations(), "negative control: an undispatched buffer mismatches the model in every record");
        }

        // ---- glGetProgramiv(COMPUTE_WORK_GROUP_SIZE) -------------------------------------------------------
        void check_program_queries(demo::Check& check)
        {
            for (const Run& r : runs_)
            {
                GLint size[3] = {-1, -1, -1};
                mgl::glGetProgramiv(r.program, mgl::ProgramParameter::ComputeWorkGroupSize, size);
                const bool ok = size[0] == static_cast<GLint>(r.cfg.local[0]) && size[1] == static_cast<GLint>(r.cfg.local[1]) &&
                                size[2] == static_cast<GLint>(r.cfg.local[2]);
                check.expect(ok,
                             std::format("{}: glGetProgramiv(COMPUTE_WORK_GROUP_SIZE) = ({}, {}, {}) equals the layout qualifier {} (ES 3.2 section 17)",
                                         r.cfg.name, size[0], size[1], size[2], to_string(r.cfg.local)));
            }
            const GLint one_d[3] = {runs_[0].queried_size[0], runs_[0].queried_size[1], runs_[0].queried_size[2]};
            check.expect(one_d[1] == 1 && one_d[2] == 1,
                         "a shader that names only local_size_x gets local size 1 in y and z (GLSL ES 3.10 section 4.4.1.1)");

            // INVALID_OPERATION for a program that has no compute shader.
            demo::drain_gl_errors();
            GLint untouched[3] = {-5, -5, -5};
            mgl::glGetProgramiv(display_, mgl::ProgramParameter::ComputeWorkGroupSize, untouched);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glGetProgramiv(COMPUTE_WORK_GROUP_SIZE) on a program without a compute shader is INVALID_OPERATION (ES 3.2 section 7.12)");
            check.expect(untouched[0] == -5 && untouched[1] == -5 && untouched[2] == -5, "the failed program query left its output untouched");
        }

        // ---- dispatch limits and errors -------------------------------------------------------------------
        void check_dispatch_errors(demo::Check& check)
        {
            // One buffer for all "hits" tests, big enough for the largest dimension we dare to dispatch.
            std::uint32_t capacity = 16;
            for (const GLint c : count_limit_)
                if (c > 0 && static_cast<std::uint32_t>(c) <= kMaxExercisedCount) capacity = std::max(capacity, static_cast<std::uint32_t>(c));
            const std::vector<std::uint32_t> sentinel(capacity + 1, kSentinel);
            const mgl::BufferId hits_buffer = create_ssbo(sentinel);
            extra_buffers_.push_back(hits_buffer);

            mgl::glUseProgram(hits_);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, hits_buffer);
            demo::drain_gl_errors();

            auto all_sentinel = [&]() {
                mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
                const std::vector<std::uint32_t> now = read_ssbo<std::uint32_t>(hits_buffer, capacity + 1);
                return compare_words(now, sentinel).bad == 0;
            };
            auto reset = [&]() {
                mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, hits_buffer);
                mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const std::uint32_t>(sentinel), mgl::BufferUsage::DynamicCopy);
                mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, hits_buffer);
            };

            const char axis[] = {'x', 'y', 'z'};
            mgl::glUniform1ui(uniform(hits_, "u_capacity"), capacity);
            for (std::size_t d = 0; d < 3; ++d)
            {
                const std::uint32_t limit = static_cast<std::uint32_t>(std::max(count_limit_[d], 0));
                if (limit == 0) continue;
                std::array<GLuint, 3> counts = {1, 1, 1};
                if (limit <= kMaxExercisedCount)
                {
                    // Exactly the limit is legal: every group runs (hits[id] = id + 1) and nothing past it is written.
                    reset();
                    counts[d] = limit;
                    mgl::glDispatchCompute(counts[0], counts[1], counts[2]);
                    check.expect_no_gl_error(std::format("dispatching exactly MAX_COMPUTE_WORK_GROUP_COUNT[{}] = {} groups raises no error", axis[d], limit));
                    mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
                    const std::vector<std::uint32_t> got = read_ssbo<std::uint32_t>(hits_buffer, capacity + 1);
                    std::vector<std::uint32_t> want(capacity + 1, kSentinel);
                    for (std::uint32_t i = 0; i < limit; ++i) want[i] = i + 1;
                    const Diff diff = compare_words(got, want);
                    check.expect(diff.bad == 0, std::format("... and all {} work groups of dimension {} ran, none past the limit", limit, axis[d]),
                                 diff.bad == 0 ? "" : std::format("{} bad; {}", diff.bad, diff.first));
                }
                else
                {
                    std::cout << std::format("   note: MAX_COMPUTE_WORK_GROUP_COUNT[{}] = {} is too large to dispatch in full; only the error above it is tested\n", axis[d], limit);
                }

                // One above the limit: INVALID_VALUE, and the command has no effect (the shader guards its writes,
                // so even a non-conforming driver cannot write out of bounds).
                if (limit < std::numeric_limits<std::uint32_t>::max())
                {
                    reset();
                    counts[d] = limit + 1;
                    mgl::glDispatchCompute(counts[0], counts[1], counts[2]);
                    check.expect_gl_error(mgl::ErrorCode::InvalidValue,
                                          std::format("dispatching {} groups in {} (limit + 1) is INVALID_VALUE (ES 3.2 section 17)", limit + 1, axis[d]));
                    check.expect(all_sentinel(), std::format("... and the rejected dispatch ran nothing ({}), ES 3.2 section 2.3.1", axis[d]));
                }
            }

            // A zero count in any dimension dispatches nothing and is not an error.
            reset();
            for (std::size_t d = 0; d < 3; ++d)
            {
                std::array<GLuint, 3> counts = {4, 4, 4};
                counts[d] = 0;
                mgl::glDispatchCompute(counts[0], counts[1], counts[2]);
                check.expect_no_gl_error(std::format("a zero work group count in {} raises no error", axis[d]));
            }
            check.expect(all_sentinel(), "... and a zero count dispatches no work groups at all (ES 3.2 section 17)");

            // No active program for the compute stage.
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDispatchCompute(1, 1, 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDispatchCompute with no current program is INVALID_OPERATION (ES 3.2 section 17)");
            mgl::glUseProgram(display_);
            mgl::glDispatchCompute(1, 1, 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glDispatchCompute with a vertex+fragment program current is INVALID_OPERATION: that program has no executable for the compute stage (ES 3.2 section 7.3)");
            mgl::glUseProgram(hits_);
            mgl::glDispatchCompute(1, 1, 1);
            check.expect_no_gl_error("positive control: the same dispatch with the compute program current raises no error");
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---- shaders that must not compile or link ---------------------------------------------------------
        void check_negative_shaders(demo::Env& env, demo::Check& check)
        {
            const std::string builtins_src = env.load_text("shaders/builtins.comp");
            std::string log;

            // Local size above the per-dimension maximum: compile-time error (GLSL ES 3.10 section 4.4.1.1).
            if (size_limit_[0] < std::numeric_limits<GLint>::max())
            {
                const std::string too_big = with_defines(builtins_src, {std::format("LOCAL_SIZE local_size_x = {}", size_limit_[0] + 1)});
                const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, too_big, log);
                check.expect(cs.value == 0,
                             std::format("local_size_x = {} (MAX_COMPUTE_WORK_GROUP_SIZE[0] + 1) is a compile-time error (GLSL ES 3.10 section 4.4.1.1)", size_limit_[0] + 1),
                             first_line(log));
                if (cs.value != 0) mgl::glDeleteShader(cs);
                // Positive control: exactly the limit compiles.
                const std::string at_limit = with_defines(builtins_src, {std::format("LOCAL_SIZE local_size_x = {}", std::min(size_limit_[0], invocation_limit_))});
                const mgl::ShaderId ok = demo::compile_shader(mgl::ShaderType::Compute, at_limit, log);
                check.expect(ok.value != 0, "positive control: the largest legal local_size_x compiles", log);
                if (ok.value != 0) mgl::glDeleteShader(ok);
            }

            // No local size: compiles, does not link.
            {
                const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, env.load_text("shaders/no_local_size.comp"), log);
                check.expect(cs.value != 0, "a compute shader without a local size still compiles", log);
                if (cs.value != 0)
                {
                    const mgl::ShaderId shaders[] = {cs};
                    const mgl::ProgramId p = demo::link_program(shaders, log);
                    check.expect(p.value == 0, "... but a program containing it fails to link (GLSL ES 3.10 section 4.4.1.1)", first_line(log));
                    if (p.value != 0) mgl::glDeleteProgram(p);
                    mgl::glDeleteShader(cs);
                }
            }

            // gl_WorkGroupSize before any local size declaration: compile-time error.
            {
                const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, env.load_text("shaders/workgroup_size_without_layout.comp"), log);
                check.expect(cs.value == 0, "gl_WorkGroupSize without a local size declaration is a compile-time error (GLSL ES 3.10 section 7.1.3)", first_line(log));
                if (cs.value != 0) mgl::glDeleteShader(cs);
            }

            // A compute shader cannot be linked with any other shader type (ES 3.2 section 7.3).
            {
                const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, with_defines(builtins_src, {"LOCAL_SIZE local_size_x = 8"}), log);
                const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, env.load_text("shaders/trivial.vert"), log);
                if (cs.value != 0 && vs.value != 0)
                {
                    const mgl::ShaderId shaders[] = {cs, vs};
                    const mgl::ProgramId p = demo::link_program(shaders, log);
                    check.expect(p.value == 0, "a program with a compute shader and a vertex shader fails to link (ES 3.2 section 7.3)", first_line(log));
                    if (p.value != 0) mgl::glDeleteProgram(p);
                }
                else
                {
                    check.expect(false, "setup of the compute+vertex link test", log);
                }
                if (cs.value != 0) mgl::glDeleteShader(cs);
                if (vs.value != 0) mgl::glDeleteShader(vs);
            }
        }

        // ---- image-free parallel array computations ----------------------------------------------------------
        void check_array_computations(demo::Check& check)
        {
            // Vector addition, then a dependent transform pass. n is not a multiple of the local size (64).
            constexpr std::uint32_t n = 1000, pad = 24;
            std::vector<float> a(n), b(n);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                a[i] = static_cast<float>(i);
                b[i] = 3.0f * static_cast<float>(i) + 0.5f;  // a + b = 4i + 0.5: exactly representable, so the sum is exact
            }
            const mgl::BufferId buf_a = create_ssbo(a);
            const mgl::BufferId buf_b = create_ssbo(b);
            const mgl::BufferId buf_c = create_ssbo(std::vector<std::uint32_t>(n + pad, kSentinel));
            const mgl::BufferId buf_d = create_ssbo(std::vector<std::uint32_t>(n + pad, kSentinel));
            extra_buffers_.insert(extra_buffers_.end(), {buf_a, buf_b, buf_c, buf_d});

            mgl::glUseProgram(vecadd_);
            mgl::glUniform1ui(uniform(vecadd_, "u_count"), n);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_a);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, buf_b);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 2, buf_c);
            mgl::glDispatchCompute((n + 63) / 64, 1, 1);

            // The second dispatch reads c: its loads must see the first dispatch's stores (ES 3.2 section 7.11.2).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glUseProgram(transform_);
            mgl::glUniform1ui(uniform(transform_, "u_count"), n);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_c);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, buf_d);
            mgl::glDispatchCompute((n + 63) / 64, 1, 1);

            // Reading the results with the CPU (glMapBufferRange) needs BUFFER_UPDATE_BARRIER_BIT.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const std::vector<std::uint32_t> c = read_ssbo<std::uint32_t>(buf_c, n + pad);
            const std::vector<std::uint32_t> d = read_ssbo<std::uint32_t>(buf_d, n + pad);

            std::vector<std::uint32_t> want_c(n + pad, kSentinel), want_d(n + pad, kSentinel);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                const float sum = a[i] + b[i];
                want_c[i] = std::bit_cast<std::uint32_t>(sum);
                const std::uint32_t v = static_cast<std::uint32_t>(sum);
                want_d[i] = (v * 2654435761u) ^ (v >> 5);
            }
            const Diff dc = compare_words(c, want_c);
            check.expect(dc.bad == 0, std::format("vector addition: all {} sums are bit-exact; the {} guard words past the end are untouched (1000 invocations are not a multiple of 64)", n, pad),
                         dc.bad == 0 ? "" : std::format("{} bad; {}", dc.bad, dc.first));
            const Diff dd = compare_words(d, want_d);
            check.expect(dd.bad == 0, "dependent second dispatch (transform of the first one's output, SHADER_STORAGE_BARRIER_BIT between) is exact",
                         dd.bad == 0 ? "" : std::format("{} bad; {}", dd.bad, dd.first));
            // Negative control: the comparison against the sums of a different input fails everywhere in range.
            std::vector<std::uint32_t> wrong_c = want_c;
            for (std::uint32_t i = 0; i < n; ++i) wrong_c[i] = std::bit_cast<std::uint32_t>(a[i] + b[i] + 1.0f);
            check.expect_eq(compare_words(c, wrong_c).bad, std::size_t{n}, "negative control: sums compared with a+b+1 mismatch in exactly the n element positions");

            // 2-D transpose, 100 x 70, 16 x 8 work groups -> 7 x 9 groups (112 x 72 invocations).
            constexpr std::uint32_t w = 100, h = 70;
            std::vector<std::uint32_t> src(w * h);
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x) src[y * w + x] = y * 1000u + x;
            const mgl::BufferId buf_src = create_ssbo(src);
            const mgl::BufferId buf_dst = create_ssbo(std::vector<std::uint32_t>(w * h + pad, kSentinel));
            extra_buffers_.insert(extra_buffers_.end(), {buf_src, buf_dst});
            mgl::glUseProgram(transpose_);
            mgl::glUniform2ui(uniform(transpose_, "u_size"), w, h);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_src);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, buf_dst);
            mgl::glDispatchCompute((w + 15) / 16, (h + 7) / 8, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const std::vector<std::uint32_t> out = read_ssbo<std::uint32_t>(buf_dst, w * h + pad);
            std::vector<std::uint32_t> want_out(w * h + pad, kSentinel);
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x) want_out[x * h + y] = src[y * w + x];
            const Diff dt = compare_words(out, want_out);
            check.expect(dt.bad == 0, "2-D transpose of a 100x70 matrix with a 7x9 grid of 16x8 work groups is exact, guard words untouched",
                         dt.bad == 0 ? "" : std::format("{} bad; {}", dt.bad, dt.first));
            std::vector<std::uint32_t> not_transposed = want_out;
            for (std::uint32_t i = 0; i < w * h; ++i) not_transposed[i] = src[i];
            check.expect(compare_words(out, not_transposed).bad > 0, "negative control: the result is not the untransposed matrix");
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---- the image ---------------------------------------------------------------------------------------
        void check_image(demo::Env& env, demo::Check& check)
        {
            const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, env.width, env.height);
            auto pixel = [&](int x, int y) {
                const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4;
                return Rgba8{img[o], img[o + 1], img[o + 2], img[o + 3]};
            };

            std::size_t cells = 0, color_bad = 0, gap_bad = 0, mutated_hits = 0;
            std::string first_bad;
            for (const Run& r : runs_)
            {
                const std::vector<Rec> model = model_records(r.cfg);
                const Layout& l = r.layout;
                for (const Rec& rec : model)
                {
                    const auto [col, row] = cell_of(l, rec.global_id);
                    const int x0 = l.origin_x + col * l.cell, y0 = l.origin_y + row * l.cell;
                    const int mid = (l.cell - 1) / 2;
                    const Rgba8 got = pixel(x0 + mid, y0 + mid);
                    const Rgba8 want = expected_color(rec, r.cfg.local_count());
                    ++cells;
                    if (got != want && color_bad++ == 0)
                        first_bad = std::format("{} cell ({}, {}): got {} want {}", r.cfg.name, col, row, demo::format_rgba(got), demo::format_rgba(want));
                    // The quad is cell - 1 pixels wide: the last pixel column of the cell shows the panel.
                    if (pixel(x0 + l.cell - 1, y0 + mid) != kPanel) ++gap_bad;
                    if (got == expected_color(rec, r.cfg.local_count(), 1)) ++mutated_hits;
                }
                // Outside the cells, the panel keeps its clear color; between the panels the background shows.
                const int corner_x = l.panel_x + 1, corner_y = l.panel_y + 1;
                if (pixel(corner_x, corner_y) != kPanel) ++gap_bad;
            }
            check.expect(color_bad == 0,
                         std::format("image: the center pixel of all {} cells has the color the model predicts for that invocation (exact)", cells),
                         color_bad == 0 ? "" : std::format("{} bad; {}", color_bad, first_bad));
            check.expect(gap_bad == 0, "image: the one pixel gap of every cell and the panel margins show the panel color (the cells are not larger than modeled)");
            check.expect(pixel(0, 0) == kBackground && pixel(env.width - 1, env.height - 1) == kBackground && pixel(env.width / 2, env.height / 2) == kBackground,
                         "image: the margins between the panels keep the clear color");
            check.expect_eq(mutated_hits, std::size_t{0}, "negative control: no cell would also match a model whose hues are shifted by one palette entry");
        }

        // Limits queried in init().
        std::array<GLint, 3> count_limit_{};
        std::array<GLint, 3> size_limit_{};
        GLint invocation_limit_ = 0;

        std::vector<Run> runs_;
        Config limit_cfg_;
        Run limit_run_;
        bool has_limit_run_ = false;

        mgl::ProgramId vecadd_{}, transform_{}, transpose_{}, hits_{}, display_{};
        mgl::UniformLocation u_target_{}, u_origin_{}, u_cell_{}, u_layout_{}, u_local_count_{};
        std::vector<mgl::BufferId> extra_buffers_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-compute-basics",
        .title = "Compute shaders: dispatch, built-ins, limits",
        .min_es = {3, 1},
        .summary = "glDispatchCompute with 1-D/2-D/3-D grids, every compute built-in recorded into an SSBO and compared with a CPU model, "
                   "the work group limits and dispatch errors, and parallel array computations; the buffer contents are drawn.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ComputeBasics>(argc, argv, info);
}
