// es30-transform-feedback
//
// Transform feedback (ES 3.0 sections 2.15 and 2.16): the vertex shader's outputs are recorded into buffer
// objects before rasterization, then used as vertex data of a later pass. Four cells show four captures
// that are drawn in a second pass (the 256x256 image is a 2x2 grid of 128x128 cells):
//
//   top left:     [A] 16 quads warped by a vertex shader, captured as TRIANGLES (INTERLEAVED_ATTRIBS, default object)
//   top right:    [B] 32 particles stepped 11 times on the GPU, two transform feedback objects ping-pong
//   bottom left:  [C] 8 points drawn with a pause in the middle: the 2 points drawn while paused are missing
//   bottom right: [D] 12 points captured in SEPARATE_ATTRIBS mode into three ranges of ONE buffer (glBindBufferRange)
//
// Every capture pass runs with RASTERIZER_DISCARD enabled (nothing may be rasterized); the number of
// primitives to draw in the second pass comes from a TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN query. validate()
// then reads every buffer back and compares it bit for bit with a CPU model (all shader arithmetic uses
// values that are exact in a float), compares every pixel of every cell with a CPU rendering model, and goes
// through introspection, object state, pause/resume rules, query rules, RASTERIZER_DISCARD and the errors that
// the specification defines for transform feedback.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr std::array<std::uint8_t, 4> kClear8 = {20, 30, 50, 255};
    constexpr std::uint8_t kSentinel = 0xA5;  // fill value of capture buffers: untouched bytes stay recognizable

    // -----------------------------------------------------------------------------------------
    // Cells
    // -----------------------------------------------------------------------------------------
    enum CellId : int { kCellMesh, kCellParticles, kCellPause, kCellStream, kCellCount };
    constexpr std::array<const char*, kCellCount> kCellName = {
        "A: warped quads captured as TRIANGLES (INTERLEAVED, default object)",
        "B: particles, 11 ping-pong steps with two transform feedback objects",
        "C: 8 points, 2 drawn while paused are missing",
        "D: SEPARATE_ATTRIBS into three ranges of one buffer"};

    struct Rect
    {
        int x, y, w, h;
    };
    Rect cell_rect(int w, int h, int id)  // id 0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right
    {
        const int cw = w / 2, ch = h / 2;
        return {(id % 2) * cw, id < 2 ? ch : 0, cw, ch};
    }

    // -----------------------------------------------------------------------------------------
    // Source data and CPU models of the shaders
    // -----------------------------------------------------------------------------------------
    struct Particle  // the INTERLEAVED capture of step.vert has exactly this layout
    {
        float x, y, vx, vy, r, g, b, size;
    };
    static_assert(sizeof(Particle) == 32);
    struct MeshVertex  // the INTERLEAVED capture of warp.vert: v_pos then v_color
    {
        float x, y, r, g, b;
    };
    static_assert(sizeof(MeshVertex) == 20);
    struct ProbeRecord  // the INTERLEAVED capture of probe.vert
    {
        std::int32_t vertex, instance;
        float trail[4];
        float position[4];
    };
    static_assert(sizeof(ProbeRecord) == 40);
    struct StreamPoint  // the three SEPARATE streams of stream.vert, put back together
    {
        float x, y, r, g, b, size;
    };

    constexpr int kParticles = 32;
    constexpr int kSteps = 11;  // odd: the final state ends up in the second buffer
    constexpr float kDt = 0.0625f;
    constexpr float kGravityY = -2.0f;
    constexpr float kFloor = -0.75f;
    constexpr int kPauseParticles = 8;
    constexpr int kMeshQuads = 16;
    constexpr int kMeshVertices = kMeshQuads * 6;
    constexpr int kMeshPadded = kMeshVertices + 4;  // 4 extra vertices feed the "incomplete primitive" test
    constexpr int kStreamPoints = 12;
    // Byte layout of the three SEPARATE streams inside one buffer. Offsets and sizes are multiples of 4
    // (the rule of ES 3.0 section 2.15.2) and deliberately not aligned to anything larger.
    constexpr std::array<GLintptr, 3> kStreamOffset = {0, 100, 260};
    constexpr std::array<GLsizeiptr, 3> kStreamBytes = {kStreamPoints * 8, kStreamPoints * 12, kStreamPoints * 4};
    constexpr std::size_t kStreamBufferBytes = 320;

    std::vector<Particle> initial_particles()
    {
        std::vector<Particle> v;
        for (int i = 0; i < kParticles; ++i)
        {
            const float col = static_cast<float>(i % 8), row = static_cast<float>(i / 8);
            v.push_back({-0.1875f + 0.125f * row, -0.5f, (2.0f * col - 7.0f) * 0.09375f, 0.25f + 0.375f * row,
                         0.25f + 0.09375f * col, 0.25f + 0.1875f * row, 1.0f - 0.09375f * col, 6.0f});
        }
        return v;
    }

    // The pause cell's input: a row of 8 points, no motion (the step shader runs with dt = 0 and copies them).
    std::vector<Particle> pause_particles()
    {
        std::vector<Particle> v;
        for (int i = 0; i < kPauseParticles; ++i)
        {
            const float k = static_cast<float>(i);
            v.push_back({-0.875f + 0.25f * k, 0.25f, 0.0f, 0.0f, 0.25f + 0.09375f * k, 0.9375f - 0.09375f * k, 0.3125f + 0.0625f * k, 8.0f});
        }
        return v;
    }

    // Mirrors step.vert statement by statement. Every value is a small dyadic fraction, so the float results are exact
    // whether or not the GPU fuses multiply-adds.
    void step_particle(Particle& p, float dt, int& bounces)
    {
        p.vx = p.vx + 0.0f * dt;  // u_gravity = (0, -2)
        p.vy = p.vy + kGravityY * dt;
        p.x = p.x + p.vx * dt;
        p.y = p.y + p.vy * dt;
        if (p.y < kFloor)
        {
            p.y = 2.0f * kFloor - p.y;
            p.vy = -0.5f * p.vy;
            ++bounces;
        }
    }
    std::vector<Particle> simulate(int steps, int* bounces = nullptr)
    {
        std::vector<Particle> v = initial_particles();
        int count = 0;
        for (int s = 0; s < steps; ++s)
            for (Particle& p : v) step_particle(p, kDt, count);
        if (bounces != nullptr) *bounces = count;
        return v;
    }

    std::vector<MeshVertex> make_mesh()  // 16 quads of 2 triangles; 4 spare vertices at the end
    {
        std::vector<MeshVertex> v;
        for (int q = 0; q < kMeshQuads; ++q)
        {
            const float i = static_cast<float>(q % 4), j = static_cast<float>(q / 4);
            const float x0 = -0.6875f + 0.375f * i, x1 = x0 + 0.25f, y0 = -0.6875f + 0.375f * j, y1 = y0 + 0.25f;
            const float r = 0.25f + 0.1875f * i, g = 0.25f + 0.1875f * j, b = 0.9375f - 0.046875f * static_cast<float>(q);
            const MeshVertex bl{x0, y0, r, g, b}, br{x1, y0, r, g, b}, tl{x0, y1, r, g, b}, tr{x1, y1, r, g, b};
            for (const MeshVertex& m : {bl, br, tl, tl, br, tr}) v.push_back(m);
        }
        while (v.size() < static_cast<std::size_t>(kMeshPadded)) v.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        return v;
    }
    MeshVertex warp(MeshVertex m)  // mirrors warp.vert
    {
        const float px = m.x, py = m.y;
        m.x = px + 0.25f * py * py;
        m.y = py + -0.125f * px * px;
        return m;
    }

    StreamPoint stream_point(int id)  // mirrors stream.vert
    {
        const int col = id % 4, row = id / 4;
        const float f = static_cast<float>(id);
        return {-0.75f + 0.5f * static_cast<float>(col), -0.5f + 0.5f * static_cast<float>(row), 0.25f + 0.046875f * f,
                0.9375f - 0.046875f * f, 0.3125f * static_cast<float>(1 + id % 3), static_cast<float>(6 + 2 * (id % 3))};
    }

    // -----------------------------------------------------------------------------------------
    // CPU rendering model: flat-colored triangles and point squares in cell pixel coordinates
    // -----------------------------------------------------------------------------------------
    using Rgba = std::array<double, 4>;
    struct Prim
    {
        bool triangle = false;
        std::array<double, 6> t{};  // triangle corners (x0 y0 x1 y1 x2 y2), cell pixels
        double cx = 0, cy = 0, size = 0;  // point: center and side, cell pixels
        Rgba color{};
    };
    enum class Cover { Outside, Inside, Unsure };
    struct Sample
    {
        Cover cover = Cover::Outside;
        Rgba color{};
    };

    double to_px(double ndc, int cell) { return (ndc + 1.0) * 0.5 * cell; }

    Prim make_triangle(const std::array<MeshVertex, 3>& v, int cw, int ch)
    {
        Prim p;
        p.triangle = true;
        for (std::size_t i = 0; i < 3; ++i)
        {
            p.t[2 * i] = to_px(static_cast<double>(v[i].x), cw);
            p.t[2 * i + 1] = to_px(static_cast<double>(v[i].y), ch);
        }
        p.color = {static_cast<double>(v[0].r), static_cast<double>(v[0].g), static_cast<double>(v[0].b), 1.0};
        return p;
    }
    Prim make_square(float x, float y, float size, float r, float g, float b, int cw, int ch)
    {
        Prim p;
        p.cx = to_px(static_cast<double>(x), cw);
        p.cy = to_px(static_cast<double>(y), ch);
        p.size = static_cast<double>(size);
        p.color = {static_cast<double>(r), static_cast<double>(g), static_cast<double>(b), 1.0};
        return p;
    }

    // Within `margin` pixels of a triangle edge the answer depends on the rasterizer's tie-breaking and sub-pixel snapping
    // rules (ES 3.0 section 3.5): Unsure, never compared.
    Sample sample_triangle(const Prim& prim, double px, double py, double margin)
    {
        const auto& t = prim.t;
        auto edge = [](double ax, double ay, double bx, double by, double x, double y) { return (bx - ax) * (y - ay) - (by - ay) * (x - ax); };
        const double area = edge(t[0], t[1], t[2], t[3], t[4], t[5]);
        if (std::abs(area) < 1e-9) return {};
        const double sign = area > 0 ? 1.0 : -1.0;
        double min_distance = 1e30;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const std::size_t j = (i + 1) % 3, k = (i + 2) % 3;
            const double e = edge(t[2 * j], t[2 * j + 1], t[2 * k], t[2 * k + 1], px, py);
            min_distance = std::min(min_distance, sign * e / std::hypot(t[2 * k] - t[2 * j], t[2 * k + 1] - t[2 * j + 1]));
        }
        if (min_distance < -margin) return {Cover::Outside, {}};
        if (min_distance <= margin) return {Cover::Unsure, {}};
        return {Cover::Inside, prim.color};
    }
    // A point is a square of side `size` around its window position; a fragment is produced for every pixel center inside it
    // (ES 3.0 section 3.4). Pixel centers within 0.25 px of an edge are Unsure (point position snapping, tie-breaking).
    Sample sample_square(const Prim& prim, double px, double py)
    {
        const double dx = std::abs(px - prim.cx), dy = std::abs(py - prim.cy), half = prim.size / 2.0, margin = 0.25;
        if (dx > half + margin || dy > half + margin) return {Cover::Outside, {}};
        if (dx < half - margin && dy < half - margin) return {Cover::Inside, prim.color};
        return {Cover::Unsure, {}};
    }
    // Later primitives overwrite earlier ones; a pixel that a later, uncertain primitive might touch is Unsure.
    Sample sample_scene(const std::vector<Prim>& scene, double px, double py)
    {
        Sample result;
        bool unsure_after = false;
        for (const Prim& p : scene)
        {
            const Sample s = p.triangle ? sample_triangle(p, px, py, 1.0) : sample_square(p, px, py);
            if (s.cover == Cover::Inside) { result = s; unsure_after = false; }
            else if (s.cover == Cover::Unsure) unsure_after = true;
        }
        return unsure_after ? Sample{Cover::Unsure, {}} : result;
    }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    struct Scan
    {
        int inside = 0, outside = 0, unsure = 0, bad = 0, max_dev = 0;
        std::string first_bad;
    };

    // Compares every pixel of a cell with the model. Tolerance +-1: float -> 8-bit quantization of flat colors
    // (ES 3.0 section 2.1.6, equation 2.3: round to nearest; implementations may differ by one unit in the last place).
    Scan scan_cell(const std::vector<std::uint8_t>& image, int image_w, const Rect& r, const std::vector<Prim>& scene)
    {
        Scan s;
        for (int y = 0; y < r.h; ++y)
            for (int x = 0; x < r.w; ++x)
            {
                const Sample m = sample_scene(scene, x + 0.5, y + 0.5);
                if (m.cover == Cover::Unsure) { ++s.unsure; continue; }
                ++(m.cover == Cover::Inside ? s.inside : s.outside);
                const std::uint8_t* got = &image[static_cast<std::size_t>(((r.y + y) * image_w + r.x + x) * 4)];
                Rgba8 want{};
                for (std::size_t k = 0; k < 4; ++k) want[k] = m.cover == Cover::Inside ? to8(m.color[k]) : kClear8[k];
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int{got[k]} - int{want[k]});
                    s.max_dev = std::max(s.max_dev, dev);
                    ok = ok && dev <= 1;
                }
                if (!ok && s.bad++ == 0)
                    s.first_bad = std::format("first mismatch at cell pixel ({},{}): got {} want {}", x, y,
                                              demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return s;
    }

    void expect_scan(demo::Check& check, const Scan& s, int min_inside, int min_outside, const std::string& what)
    {
        const bool enough = s.inside >= min_inside && s.outside >= min_outside;
        check.expect(s.bad == 0 && enough,
                     std::format("{}: {} covered + {} background pixels match the model (max deviation {}/255, {} edge pixels skipped)", what, s.inside,
                                 s.outside, s.max_dev, s.unsure),
                     s.bad ? std::format("{} mismatches; {}", s.bad, s.first_bad)
                     : enough ? std::string{} : std::format("only {} covered / {} background pixels compared", s.inside, s.outside));
    }

    // -----------------------------------------------------------------------------------------
    // Small helpers on byte vectors
    // -----------------------------------------------------------------------------------------
    template <typename T>
    std::vector<T> from_bytes(const std::vector<std::uint8_t>& bytes, std::size_t offset, std::size_t count)
    {
        std::vector<T> out(count);
        if (bytes.size() >= offset + count * sizeof(T)) std::memcpy(out.data(), bytes.data() + offset, count * sizeof(T));
        return out;
    }
    bool all_sentinel(const std::vector<std::uint8_t>& bytes, std::size_t begin, std::size_t end)
    {
        if (bytes.size() < end) return false;
        return std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(begin), bytes.begin() + static_cast<std::ptrdiff_t>(end),
                           [](std::uint8_t b) { return b == kSentinel; });
    }
    bool same_particles(const std::vector<Particle>& a, const std::vector<Particle>& b, std::size_t& first_diff)
    {
        if (a.size() != b.size()) { first_diff = 0; return false; }
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].vx != b[i].vx || a[i].vy != b[i].vy || a[i].r != b[i].r || a[i].g != b[i].g ||
                a[i].b != b[i].b || a[i].size != b[i].size)
            {
                first_diff = i;
                return false;
            }
        return true;
    }
    bool same_mesh(const std::vector<MeshVertex>& a, const std::vector<MeshVertex>& b, std::size_t& first_diff)
    {
        if (a.size() != b.size()) { first_diff = 0; return false; }
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].r != b[i].r || a[i].g != b[i].g || a[i].b != b[i].b)
            {
                first_diff = i;
                return false;
            }
        return true;
    }
    template <typename T>
    std::vector<T> select(const std::vector<T>& v, std::initializer_list<std::size_t> indices)
    {
        std::vector<T> out;
        for (const std::size_t i : indices) out.push_back(v.at(i));
        return out;
    }

    bool get_bool(mgl::GetParameter pname)
    {
        GLboolean b = GL_FALSE;
        mgl::glGetBooleanv(pname, &b);
        return b == GL_TRUE;
    }
    GLint get_indexed(mgl::GetParameter pname, GLuint index)
    {
        GLint v = -1;
        mgl::glGetIntegeri_v(pname, index, &v);
        return v;
    }
    GLint64 get_indexed64(mgl::GetParameter pname, GLuint index)
    {
        GLint64 v = -1;
        mgl::glGetInteger64i_v(pname, index, &v);
        return v;
    }
    GLuint query_result(mgl::QueryId q)  // QUERY_RESULT forces the query to complete (ES 3.0 section 6.1.7)
    {
        GLuint v = 0xFFFFFFFFu;
        mgl::glGetQueryObjectuiv(q, mgl::QueryObjectParameter::Result, &v);
        return v;
    }

    struct VaryingInfo
    {
        std::string name;
        mgl::UniformType type{};
        GLsizei size = 0;
        GLsizei length = 0;
    };
    VaryingInfo get_varying(mgl::ProgramId program, GLuint index, GLsizei buf_size = 64)
    {
        std::string name(static_cast<std::size_t>(buf_size), '\0');
        VaryingInfo v;
        mgl::glGetTransformFeedbackVarying(program, index, buf_size, &v.length, &v.size, &v.type, name.data());
        name.resize(static_cast<std::size_t>(std::max(v.length, 0)));
        v.name = name;
        return v;
    }

    // A documented deviation of Mesa's Gallium drivers: strict on every other driver (docs/KNOWN_LIMITATIONS.md).
    bool expect_mesa_deviation(demo::Check& check, bool ok, std::string_view what, std::string_view detail, std::string_view reason)
    {
        const GLubyte* r = mgl::glGetString(mgl::StringName::Renderer);
        const std::string renderer = r != nullptr ? reinterpret_cast<const char*>(r) : "";
        for (const std::string_view name : {"zink", "radeonsi", "llvmpipe", "softpipe"})
            if (renderer.find(name) != std::string::npos) return check.expect_or_known_deviation(ok, what, detail, name, reason);
        return check.expect(ok, what, detail);
    }

    class TransformFeedbackDemo final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            GLfloat point_range[2] = {};
            mgl::glGetFloatv(mgl::GetParameter::AliasedPointSizeRange, point_range);
            if (point_range[1] < 10.0f) throw demo::Skip("ALIASED_POINT_SIZE_RANGE maximum is below the 10 px points of the demo");

            // --- programs: transform feedback varyings must be declared BEFORE linking (ES 3.0 section 2.12.8) --------
            {
                const GLchar* v[] = {"v_pos", "v_vel", "v_color", "v_size"};
                prog_step_ = link_tf_program(env, "shaders/step.vert", v, mgl::TransformFeedbackBufferMode::InterleavedAttribs, nullptr, nullptr);
            }
            {
                const GLchar* v[] = {"v_pos", "v_color"};
                prog_warp_ = link_tf_program(env, "shaders/warp.vert", v, mgl::TransformFeedbackBufferMode::InterleavedAttribs, nullptr, nullptr);
            }
            {
                const GLchar* v[] = {"v_pos", "v_color", "v_size"};
                prog_stream_ = link_tf_program(env, "shaders/stream.vert", v, mgl::TransformFeedbackBufferMode::SeparateAttribs, nullptr, nullptr);
            }
            {
                const GLchar* v[] = {"v_vertex", "v_instance", "v_trail", "gl_Position"};
                prog_probe_ = link_tf_program(env, "shaders/probe.vert", v, mgl::TransformFeedbackBufferMode::InterleavedAttribs, nullptr, nullptr);
            }
            prog_render_ = demo::build_program(env, "shaders/render.vert", "shaders/render.frag");  // no varyings: not usable for capture

            for (const mgl::ProgramId p : {prog_step_, prog_warp_, prog_stream_, prog_probe_})
            {
                mgl::glUseProgram(p);
                mgl::glUniform4f(mgl::glGetUniformLocation(p, "u_color"), 1.0f, 0.0f, 0.0f, 1.0f);  // the leak color of the capture passes
            }
            mgl::glUseProgram(prog_step_);
            u_dt_ = mgl::glGetUniformLocation(prog_step_, "u_dt");
            u_color_ = mgl::glGetUniformLocation(prog_step_, "u_color");
            mgl::glUniform2f(mgl::glGetUniformLocation(prog_step_, "u_gravity"), 0.0f, kGravityY);
            mgl::glUniform1f(mgl::glGetUniformLocation(prog_step_, "u_floor"), kFloor);
            mgl::glUseProgram(mgl::ProgramId{0});

            // --- initial state, recorded before any demo code touches it (ES 3.0 Tables 6.24, 6.33) --------------------
            init_.max_separate_attribs = demo::get_integer(mgl::GetParameter::MaxTransformFeedbackSeparateAttribs);
            init_.tf_binding = demo::get_integer(mgl::GetParameter::TransformFeedbackBinding);
            for (int i = 0; i < init_.max_separate_attribs; ++i) init_.bindings.push_back(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, static_cast<GLuint>(i)));
            init_.discard_enabled = mgl::glIsEnabled(mgl::Capability::RasterizerDiscard);
            init_.active = get_bool(mgl::GetParameter::TransformFeedbackActive);

            // --- buffers --------------------------------------------------------------------------------------------
            vbo_mesh_ = make_buffer();
            upload(vbo_mesh_, std::span<const MeshVertex>(make_mesh()));
            tf_mesh_ = make_buffer();
            allocate(tf_mesh_, kMeshPadded * sizeof(MeshVertex), kSentinel);
            for (mgl::BufferId& b : particles_)
            {
                b = make_buffer();
                allocate(b, kParticles * sizeof(Particle), kSentinel);
            }
            pause_in_ = make_buffer();
            upload(pause_in_, std::span<const Particle>(pause_particles()));
            pause_out_ = make_buffer();
            allocate(pause_out_, kPauseParticles * sizeof(Particle), kSentinel);
            stream_buf_ = make_buffer();
            allocate(stream_buf_, kStreamBufferBytes, kSentinel);
            const std::array<float, 6> probe_positions = {-0.5f, -0.25f, 0.25f, 0.5f, 0.75f, -0.75f};
            probe_in_ = make_buffer();
            upload(probe_in_, std::span<const float>(probe_positions));
            for (mgl::BufferId& b : scratch_) b = make_buffer();
            const std::array<std::uint8_t, kPauseParticles> indices = {0, 1, 2, 3, 4, 5, 6, 7};
            pause_ebo_ = make_buffer();

            // --- vertex arrays ---------------------------------------------------------------------------------------
            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glBindVertexArray(vao_[kVaoMeshIn]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_mesh_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(MeshVertex), demo::buffer_offset(offsetof(MeshVertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(MeshVertex), demo::buffer_offset(offsetof(MeshVertex, r)));
            mgl::glBindVertexArray(vao_[kVaoMeshOut]);  // reads what pass 1 captured
            mgl::glBindBuffer(mgl::BufferTarget::Array, tf_mesh_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(MeshVertex), demo::buffer_offset(offsetof(MeshVertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(MeshVertex), demo::buffer_offset(offsetof(MeshVertex, r)));
            particle_vao(vao_[kVaoParticles0], particles_[0]);
            particle_vao(vao_[kVaoParticles1], particles_[1]);
            particle_vao(vao_[kVaoPauseIn], pause_in_);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, pause_ebo_);  // attached to the pause-input VAO
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint8_t>(indices), mgl::BufferUsage::StaticDraw);
            particle_vao(vao_[kVaoPauseOut], pause_out_);
            mgl::glBindVertexArray(vao_[kVaoStreamRender]);  // three attribute streams from three ranges of one buffer
            mgl::glBindBuffer(mgl::BufferTarget::Array, stream_buf_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 8, demo::buffer_offset(static_cast<std::uintptr_t>(kStreamOffset[0])));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 3, mgl::DataType::Float, GL_FALSE, 12, demo::buffer_offset(static_cast<std::uintptr_t>(kStreamOffset[1])));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{3u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{3u}, 1, mgl::DataType::Float, GL_FALSE, 4, demo::buffer_offset(static_cast<std::uintptr_t>(kStreamOffset[2])));
            mgl::glBindVertexArray(vao_[kVaoProbeIn]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, probe_in_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // --- transform feedback objects (names exist, but state only after the first bind: section 2.15.1) --------
            mgl::glGenTransformFeedbacks(static_cast<GLsizei>(tf_.size()), tf_.data());
            mgl::glGenTransformFeedbacks(1, &tf_misc_);
            for (const mgl::TransformFeedbackId t : tf_) init_.is_tf_before_bind.push_back(mgl::glIsTransformFeedback(t));
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, tf_[0]);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, particles_[0]);  // the whole buffer
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, tf_[1]);
            mgl::glBindBufferRange(mgl::BufferTarget::TransformFeedback, 0, particles_[1], 0, kParticles * sizeof(Particle));  // an explicit range
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, mgl::TransformFeedbackId{0});
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});

            // --- a query object (no object exists until the first BeginQuery: section 2.14) ------------------------
            mgl::glGenQueries(1, &query_);
            init_.is_query_before_begin = mgl::glIsQuery(query_);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::RasterizerDiscard);  // glClear is ignored while it is enabled
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            draw_mesh_cell(env);
            draw_particle_cell(env);
            draw_pause_cell(env);
            draw_stream_cell(env);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_cells(env, check);
            check_captured_data(check);
            check_modes_and_introspection(check);
            check_objects(check);
            check_pause_resume(check);
            check_queries(check);
            check_rasterizer_discard(env, check);
            check_errors(env, check);
            render(env, 0);  // restore the capture image
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, mgl::TransformFeedbackId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteTransformFeedbacks(static_cast<GLsizei>(tf_.size()), tf_.data());
            mgl::glDeleteTransformFeedbacks(1, &tf_misc_);
            mgl::glDeleteQueries(1, &query_);
            mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            for (const mgl::ProgramId p : {prog_step_, prog_warp_, prog_stream_, prog_probe_, prog_render_}) mgl::glDeleteProgram(p);
        }

    private:
        static constexpr auto kTF = mgl::BufferTarget::TransformFeedback;
        static constexpr auto kTFObject = mgl::TransformFeedbackTarget::TransformFeedback;
        static constexpr auto kPrimQuery = mgl::QueryTarget::TransformFeedbackPrimitivesWritten;
        enum VaoSlot : std::size_t { kVaoMeshIn, kVaoMeshOut, kVaoParticles0, kVaoParticles1, kVaoPauseIn, kVaoPauseOut, kVaoStreamRender, kVaoProbeIn, kVaoEmpty, kVaoCount };

        // ---------------------------------------------------------------------------------------------
        // Program and buffer helpers
        // ---------------------------------------------------------------------------------------------
        // Compiles vs_file + solid.frag, declares the varyings, links. With `linked` == nullptr a failure throws; otherwise
        // the link result and log are reported and the (possibly unlinked) program is returned for the caller to delete.
        mgl::ProgramId link_tf_program(const demo::Env& env, std::string_view vs_file, std::span<const GLchar* const> varyings,
                                       mgl::TransformFeedbackBufferMode mode, bool* linked, std::string* log_out) const
        {
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, env.load_text(vs_file), log);
            if (vs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", vs_file, log));
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, env.load_text("shaders/solid.frag"), log);
            if (fs.value == 0) throw demo::Failure("solid.frag failed to compile:\n" + log);
            const mgl::ProgramId program = mgl::glCreateProgram();
            mgl::glAttachShader(program, vs);
            mgl::glAttachShader(program, fs);
            mgl::glTransformFeedbackVaryings(program, static_cast<GLsizei>(varyings.size()), varyings.data(), mode);
            mgl::glLinkProgram(program);
            GLint status = GL_FALSE;
            mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &status);
            if (log_out != nullptr) *log_out = demo::program_info_log(program);
            const std::string link_log = demo::program_info_log(program);
            mgl::glDetachShader(program, vs);
            mgl::glDetachShader(program, fs);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (linked != nullptr) *linked = status == GL_TRUE;
            else if (status != GL_TRUE) throw demo::Failure(std::format("{} failed to link:\n{}", vs_file, link_log));
            return program;
        }

        mgl::BufferId make_buffer()
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            buffers_.push_back(b);
            return b;
        }
        // Buffers are filled through COPY_WRITE_BUFFER so that no transform feedback or vertex array binding is disturbed.
        template <typename T>
        void upload(mgl::BufferId b, std::span<const T> data) const
        {
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, b);
            mgl::glBufferData(mgl::BufferTarget::CopyWrite, data, mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
        }
        void allocate(mgl::BufferId b, std::size_t bytes, std::uint8_t fill) const  // a fresh data store full of `fill`
        {
            const std::vector<std::uint8_t> data(bytes, fill);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, b);
            mgl::glBufferData(mgl::BufferTarget::CopyWrite, std::span<const std::uint8_t>(data), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
        }
        // Reads a captured buffer through the TRANSFORM_FEEDBACK_BUFFER target. Only called while no capture is active
        // (ES 3.0 section 2.15.2: MapBufferRange "may need to block pending the completion of a previous transform feedback operation").
        std::vector<std::uint8_t> read_buffer(mgl::BufferId b, std::size_t bytes) const
        {
            std::vector<std::uint8_t> out;
            mgl::glBindBuffer(kTF, b);
            const void* mapped = mgl::glMapBufferRange(kTF, 0, static_cast<GLsizeiptr>(bytes), mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                out.resize(bytes);
                std::memcpy(out.data(), mapped, bytes);
                (void)mgl::glUnmapBuffer(kTF);
            }
            mgl::glBindBuffer(kTF, mgl::BufferId{0});
            return out;
        }

        void particle_vao(mgl::VertexArrayId vao, mgl::BufferId buffer) const  // the 32-byte Particle layout
        {
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buffer);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(Particle), demo::buffer_offset(offsetof(Particle, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(Particle), demo::buffer_offset(offsetof(Particle, vx)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Particle), demo::buffer_offset(offsetof(Particle, r)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{3u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{3u}, 1, mgl::DataType::Float, GL_FALSE, sizeof(Particle), demo::buffer_offset(offsetof(Particle, size)));
        }

        // ---------------------------------------------------------------------------------------------
        // The four cells: pass 1 captures with RASTERIZER_DISCARD enabled, pass 2 draws what was captured
        // ---------------------------------------------------------------------------------------------
        void draw_mesh_cell(const demo::Env& env)
        {
            mgl::glUseProgram(prog_warp_);
            mgl::glBindVertexArray(vao_[kVaoMeshIn]);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});  // the default object
            allocate(tf_mesh_, kMeshPadded * sizeof(MeshVertex), kSentinel);
            mgl::glBindBufferBase(kTF, 0, tf_mesh_);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Triangles);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, kMeshVertices);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mesh_prims_ = query_result(query_);

            const Rect r = cell_rect(env.width, env.height, kCellMesh);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUseProgram(prog_render_);
            mgl::glBindVertexArray(vao_[kVaoMeshOut]);
            mgl::glVertexAttrib1f(mgl::AttribLocation{3u}, 1.0f);  // the size attribute has no array here
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, static_cast<GLsizei>(mesh_prims_ * 3));  // count from the query
        }

        void draw_particle_cell(const demo::Env& env)
        {
            upload(particles_[0], std::span<const Particle>(initial_particles()));
            allocate(particles_[1], kParticles * sizeof(Particle), kSentinel);
            mgl::glUseProgram(prog_step_);
            mgl::glUniform1f(u_dt_, kDt);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(kPrimQuery, query_);  // one query accumulates over all steps and both objects
            for (int s = 0; s < kSteps; ++s)
            {
                const std::size_t src = static_cast<std::size_t>(s % 2), dst = 1 - src;
                mgl::glBindTransformFeedback(kTFObject, tf_[dst]);  // its buffer binding is part of the object: nothing else to set
                mgl::glBindVertexArray(vao_[kVaoParticles0 + src]);
                mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
                mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kParticles);
                mgl::glEndTransformFeedback();
            }
            mgl::glEndQuery(kPrimQuery);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            particle_prims_ = query_result(query_);

            const Rect r = cell_rect(env.width, env.height, kCellParticles);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUseProgram(prog_render_);
            mgl::glBindVertexArray(vao_[kVaoParticles0 + static_cast<std::size_t>(kSteps % 2)]);  // the last step wrote buffer 1
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kParticles);
        }

        void draw_pause_cell(const demo::Env& env)
        {
            allocate(pause_out_, kPauseParticles * sizeof(Particle), kSentinel);
            mgl::glUseProgram(prog_step_);
            mgl::glUniform1f(u_dt_, 0.0f);  // identity step: the capture is a copy of the input
            mgl::glBindVertexArray(vao_[kVaoPauseIn]);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glBindBufferBase(kTF, 0, pause_out_);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 3);  // particles 0, 1, 2: recorded
            mgl::glPauseTransformFeedback();
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 3, 1);                                                  // particle 3: not recorded
            mgl::glDrawElements(mgl::PrimitiveType::Lines, 2, mgl::IndexType::UnsignedByte, demo::buffer_offset(4));  // elements 4, 5 as a LINE: any primitive type is allowed while paused
            mgl::glResumeTransformFeedback();
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 5, 3);  // particles 5, 6, 7: recorded right after particle 2
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            pause_prims_ = query_result(query_);

            const Rect r = cell_rect(env.width, env.height, kCellPause);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUseProgram(prog_render_);
            mgl::glBindVertexArray(vao_[kVaoPauseOut]);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, static_cast<GLsizei>(pause_prims_));
        }

        void draw_stream_cell(const demo::Env& env)
        {
            allocate(stream_buf_, kStreamBufferBytes, kSentinel);
            mgl::glUseProgram(prog_stream_);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);  // no vertex attributes: the shader works from gl_VertexID
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferRange(kTF, i, stream_buf_, kStreamOffset[i], kStreamBytes[i]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kStreamPoints);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            stream_prims_ = query_result(query_);

            const Rect r = cell_rect(env.width, env.height, kCellStream);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUseProgram(prog_render_);
            mgl::glBindVertexArray(vao_[kVaoStreamRender]);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, static_cast<GLsizei>(stream_prims_));
        }

        // ---------------------------------------------------------------------------------------------
        // CPU models of what the cells must contain
        // ---------------------------------------------------------------------------------------------
        std::vector<MeshVertex> expected_mesh() const
        {
            std::vector<MeshVertex> v = make_mesh();
            v.resize(kMeshVertices);
            for (MeshVertex& m : v) m = warp(m);
            return v;
        }
        std::vector<Prim> scene_mesh(const Rect& r, bool apply_warp) const
        {
            std::vector<MeshVertex> v = make_mesh();
            v.resize(kMeshVertices);
            std::vector<Prim> scene;
            for (std::size_t i = 0; i + 2 < v.size(); i += 3)
                scene.push_back(make_triangle({apply_warp ? warp(v[i]) : v[i], apply_warp ? warp(v[i + 1]) : v[i + 1], apply_warp ? warp(v[i + 2]) : v[i + 2]}, r.w, r.h));
            return scene;
        }
        static std::vector<Prim> scene_particles(const std::vector<Particle>& ps, const Rect& r)
        {
            std::vector<Prim> scene;
            for (const Particle& p : ps) scene.push_back(make_square(p.x, p.y, p.size, p.r, p.g, p.b, r.w, r.h));
            return scene;
        }
        static std::vector<Prim> scene_stream(const Rect& r, float forced_size = 0.0f)
        {
            std::vector<Prim> scene;
            for (int i = 0; i < kStreamPoints; ++i)
            {
                const StreamPoint p = stream_point(i);
                scene.push_back(make_square(p.x, p.y, forced_size > 0.0f ? forced_size : p.size, p.r, p.g, p.b, r.w, r.h));
            }
            return scene;
        }

        // ---------------------------------------------------------------------------------------------
        // validate: pixels
        // ---------------------------------------------------------------------------------------------
        void check_cells(const demo::Env& env, demo::Check& check) const
        {
            const std::vector<std::uint8_t> image = demo::read_rgba8(0, 0, env.width, env.height);
            std::array<Rect, kCellCount> rect{};
            for (int id = 0; id < kCellCount; ++id) rect[static_cast<std::size_t>(id)] = cell_rect(env.width, env.height, id);
            const std::vector<Particle> final_state = simulate(kSteps);
            const std::vector<Particle> short_state = simulate(kSteps - 1);
            std::vector<Particle> eight = pause_particles();  // what the picture would show if pausing had no effect
            std::vector<Particle> recorded = select(pause_particles(), {0, 1, 2, 5, 6, 7});

            struct CellCase
            {
                CellId id;
                std::vector<Prim> model;
                int min_inside, min_outside;
            };
            std::vector<CellCase> cases;
            cases.push_back({kCellMesh, scene_mesh(rect[kCellMesh], true), 2000, 8000});
            cases.push_back({kCellParticles, scene_particles(final_state, rect[kCellParticles]), 900, 10000});
            cases.push_back({kCellPause, scene_particles(recorded, rect[kCellPause]), 300, 14000});
            cases.push_back({kCellStream, scene_stream(rect[kCellStream]), 700, 12000});
            for (const CellCase& c : cases)
                expect_scan(check, scan_cell(image, env.width, rect[static_cast<std::size_t>(c.id)], c.model), c.min_inside, c.min_outside,
                            std::format("cell {}", kCellName[static_cast<std::size_t>(c.id)]));

            // Negative controls: the same pixels must DISAGREE with a model of a different configuration, so the scans above can fail.
            auto must_differ = [&](CellId id, const std::vector<Prim>& wrong, const char* what) {
                const Scan s = scan_cell(image, env.width, rect[static_cast<std::size_t>(id)], wrong);
                check.expect(s.bad >= 50, std::format("negative control: cell {} disagrees with {} ({} mismatching pixels)", static_cast<int>(id), what, s.bad));
            };
            must_differ(kCellMesh, scene_mesh(rect[kCellMesh], false), "the model without the warp");
            must_differ(kCellParticles, scene_particles(short_state, rect[kCellParticles]), "a model with one step less (10 instead of 11)");
            must_differ(kCellPause, scene_particles(eight, rect[kCellPause]), "all 8 points, as if pausing had no effect");
            must_differ(kCellStream, scene_stream(rect[kCellStream], 8.0f), "a model where every point has size 8");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: captured bytes
        // ---------------------------------------------------------------------------------------------
        void check_captured_data(demo::Check& check) const
        {
            std::size_t diff = 0;
            // A: INTERLEAVED capture of TRIANGLES, 96 vertices of 20 bytes, compared bit for bit with the warped mesh.
            const std::vector<std::uint8_t> mesh = read_buffer(tf_mesh_, kMeshPadded * sizeof(MeshVertex));
            const std::vector<MeshVertex> mesh_got = from_bytes<MeshVertex>(mesh, 0, kMeshVertices);
            check.expect(mesh_got.size() == kMeshVertices && !mesh.empty() && same_mesh(mesh_got, expected_mesh(), diff),
                         "A: the 96 captured vertices (v_pos, v_color interleaved, stride 20) equal the CPU-warped mesh exactly",
                         std::format("first difference at vertex {}", diff));
            check.expect(!mesh.empty() && all_sentinel(mesh, kMeshVertices * sizeof(MeshVertex), kMeshPadded * sizeof(MeshVertex)),
                         "A: the 4 spare vertices' space in the buffer was not written (only 32 triangles = 96 vertices recorded)");
            check.expect_eq(mesh_prims_, static_cast<GLuint>(kMeshPrimsExpected), "A: TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 32 triangles (and the pass-2 draw used it)");

            // B: two ping-pong buffers: buffer 1 holds step 11, buffer 0 holds step 10.
            int bounces = 0;
            const std::vector<Particle> state11 = simulate(kSteps, &bounces);
            const std::vector<Particle> state10 = simulate(kSteps - 1);
            const std::vector<Particle> p1 = from_bytes<Particle>(read_buffer(particles_[1], kParticles * sizeof(Particle)), 0, kParticles);
            const std::vector<Particle> p0 = from_bytes<Particle>(read_buffer(particles_[0], kParticles * sizeof(Particle)), 0, kParticles);
            check.expect(same_particles(p1, state11, diff), "B: buffer 1 (written by the 11th step) equals the CPU simulation after 11 steps, all 32 x 8 floats exactly",
                         std::format("first difference at particle {}", diff));
            check.expect(same_particles(p0, state10, diff), "B: buffer 0 (written by the 10th step) equals the CPU simulation after 10 steps exactly",
                         std::format("first difference at particle {}", diff));
            check.expect(bounces > 0, std::format("B: the floor bounce branch of step.vert was exercised ({} bounces in the model)", bounces));
            check.expect(!same_particles(p0, p1, diff) && !same_particles(p1, initial_particles(), diff),
                         "B: negative control: the two buffers differ from each other and from the initial state");
            check.expect_eq(particle_prims_, static_cast<GLuint>(kSteps * kParticles), "B: one query over 11 captures into two different objects counts 11 x 32 = 352 points");

            // C: the pause cell: records 0, 1, 2, 5, 6, 7 contiguous, the remaining space untouched.
            const std::vector<std::uint8_t> pause = read_buffer(pause_out_, kPauseParticles * sizeof(Particle));
            check.expect(same_particles(from_bytes<Particle>(pause, 0, 6), select(pause_particles(), {0, 1, 2, 5, 6, 7}), diff),
                         "C: the buffer holds particles 0,1,2,5,6,7 back to back: the draws while paused (3 and the line 4-5) were not recorded",
                         std::format("first difference at record {}", diff));
            check.expect(all_sentinel(pause, 6 * sizeof(Particle), kPauseParticles * sizeof(Particle)),
                         "C: the last two record slots are untouched (resume appended right after the last record before the pause)");
            check.expect(!same_particles(from_bytes<Particle>(pause, 0, 6), select(pause_particles(), {0, 1, 2, 3, 4, 5}), diff),
                         "C: negative control: the buffer is NOT the first 6 particles, which is what a capture that ignored the pause would hold");
            check.expect_eq(pause_prims_, 6u, "C: TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 6 (paused draws are not counted)");

            // D: three SEPARATE streams in one buffer, gaps between the ranges untouched.
            const std::vector<std::uint8_t> stream = read_buffer(stream_buf_, kStreamBufferBytes);
            std::vector<float> pos, color, size;
            for (int i = 0; i < kStreamPoints; ++i)
            {
                const StreamPoint p = stream_point(i);
                pos.insert(pos.end(), {p.x, p.y});
                color.insert(color.end(), {p.r, p.g, p.b});
                size.push_back(p.size);
            }
            auto stream_equal = [&](std::size_t stream_index, const std::vector<float>& want) {
                const std::vector<float> got = from_bytes<float>(stream, static_cast<std::size_t>(kStreamOffset[stream_index]), want.size());
                return !stream.empty() && got == want;
            };
            check.expect(stream_equal(0, pos), "D: binding point 0 (offset 0) holds the 12 v_pos vec2 values, tightly packed (stride 8)");
            check.expect(stream_equal(1, color), "D: binding point 1 (offset 100) holds the 12 v_color vec3 values, tightly packed (stride 12)");
            check.expect(stream_equal(2, size), "D: binding point 2 (offset 260) holds the 12 v_size floats, tightly packed (stride 4)");
            check.expect(all_sentinel(stream, 96, 100) && all_sentinel(stream, 244, 260) && all_sentinel(stream, 308, 320),
                         "D: the bytes between and after the three ranges (96-99, 244-259, 308-319) were never written");
            check.expect_eq(stream_prims_, static_cast<GLuint>(kStreamPoints), "D: TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 12");
        }
        static constexpr int kMeshPrimsExpected = kMeshVertices / 3;

        // ---------------------------------------------------------------------------------------------
        // validate: buffer modes, introspection, the order of recorded values
        // ---------------------------------------------------------------------------------------------
        void check_modes_and_introspection(demo::Check& check)
        {
            auto program_int = [](mgl::ProgramId p, mgl::ProgramParameter pname) {
                GLint v = -1;
                mgl::glGetProgramiv(p, pname, &v);
                return v;
            };
            using Mode = mgl::TransformFeedbackBufferMode;
            auto mode_of = [&](mgl::ProgramId p) { return static_cast<GLenum>(program_int(p, mgl::ProgramParameter::TransformFeedbackBufferMode)); };
            check.expect(mode_of(prog_step_) == static_cast<GLenum>(Mode::InterleavedAttribs) && mode_of(prog_stream_) == static_cast<GLenum>(Mode::SeparateAttribs),
                         "TRANSFORM_FEEDBACK_BUFFER_MODE: INTERLEAVED_ATTRIBS for the particle program, SEPARATE_ATTRIBS for the stream program");
            check.expect(program_int(prog_step_, mgl::ProgramParameter::TransformFeedbackVaryings) == 4 && program_int(prog_warp_, mgl::ProgramParameter::TransformFeedbackVaryings) == 2 &&
                             program_int(prog_stream_, mgl::ProgramParameter::TransformFeedbackVaryings) == 3 && program_int(prog_probe_, mgl::ProgramParameter::TransformFeedbackVaryings) == 4 &&
                             program_int(prog_render_, mgl::ProgramParameter::TransformFeedbackVaryings) == 0,
                         "TRANSFORM_FEEDBACK_VARYINGS: 4 / 2 / 3 / 4 for step / warp / stream / probe, 0 for the program without varyings");
            check.expect_eq(program_int(prog_probe_, mgl::ProgramParameter::TransformFeedbackVaryingMaxLength), 12, "TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH = 12 (\"gl_Position\" and 11 characters + terminator)");

            // glGetTransformFeedbackVarying: name, type and size, in the order of the glTransformFeedbackVaryings list.
            const VaryingInfo v0 = get_varying(prog_probe_, 0), v1 = get_varying(prog_probe_, 1), v2 = get_varying(prog_probe_, 2), v3 = get_varying(prog_probe_, 3);
            check.expect(v0.name == "v_vertex" && v0.type == mgl::UniformType::Int && v0.size == 1, "varying 0 is v_vertex: INT, size 1",
                         std::format("got '{}' size {}", v0.name, v0.size));
            check.expect(v1.name == "v_instance" && v1.type == mgl::UniformType::Int && v1.size == 1, "varying 1 is v_instance: INT, size 1",
                         std::format("got '{}' size {}", v1.name, v1.size));
            check.expect(v2.name == "v_trail" && v2.type == mgl::UniformType::FloatVec2 && v2.size == 2, "varying 2 is the array v_trail: FLOAT_VEC2, size 2 (units of the type)",
                         std::format("got '{}' size {}", v2.name, v2.size));
            check.expect(v3.name == "gl_Position" && v3.type == mgl::UniformType::FloatVec4 && v3.size == 1, "varying 3 is the built-in gl_Position: FLOAT_VEC4, size 1",
                         std::format("got '{}' size {}", v3.name, v3.size));
            const VaryingInfo cut = get_varying(prog_probe_, 3, 4);
            check.expect(cut.name == "gl_" && cut.length == 3, "bufSize 4 truncates \"gl_Position\" to 3 characters + terminator and reports length 3", std::format("got '{}' length {}", cut.name, cut.length));
            {
                char name[8] = "keep";
                GLsizei length = 77, size = 77;
                mgl::UniformType type = mgl::UniformType::Bool;
                mgl::glGetTransformFeedbackVarying(prog_probe_, 4, 8, &length, &size, &type, name);
                check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetTransformFeedbackVarying with index = TRANSFORM_FEEDBACK_VARYINGS -> INVALID_VALUE");
                check.expect(length == 77 && size == 77 && type == mgl::UniformType::Bool && std::string_view(name) == "keep", "... and the return parameters are not modified");
            }

            // The recording order and layout, with integer outputs, an array output and a built-in: 3 vertices x 2 instances.
            allocate(scratch_[0], 320, kSentinel);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glUseProgram(prog_probe_);
            mgl::glBindVertexArray(vao_[kVaoProbeIn]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::Points, 0, 3, 2);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            const GLuint probe_prims = query_result(query_);
            const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], 320);
            const std::vector<ProbeRecord> rec = from_bytes<ProbeRecord>(bytes, 0, 6);
            const std::array<std::array<float, 2>, 3> pos = {{{-0.5f, -0.25f}, {0.25f, 0.5f}, {0.75f, -0.75f}}};
            bool ok = rec.size() == 6 && !bytes.empty();
            std::string why;
            for (std::size_t i = 0; ok && i < 6; ++i)
            {
                const std::size_t vtx = i % 3, inst = i / 3;
                const ProbeRecord& r = rec[i];
                const bool match = r.vertex == static_cast<std::int32_t>(vtx) && r.instance == static_cast<std::int32_t>(inst) && r.trail[0] == pos[vtx][0] && r.trail[1] == pos[vtx][1] &&
                                   r.trail[2] == 2.0f * pos[vtx][0] && r.trail[3] == 2.0f * pos[vtx][1] && r.position[0] == pos[vtx][0] && r.position[1] == pos[vtx][1] &&
                                   r.position[2] == 0.0f && r.position[3] == 1.0f;
                if (!match) { ok = false; why = std::format("record {} is wrong: vertex {} instance {} trail ({},{},{},{}) position ({},{},{},{})", i, r.vertex, r.instance, r.trail[0], r.trail[1], r.trail[2], r.trail[3], r.position[0], r.position[1], r.position[2], r.position[3]); }
            }
            check.expect(ok, "probe: 6 records of 40 bytes: gl_VertexID/gl_InstanceID as ints, both v_trail elements in order, gl_Position (instance 0's vertices first)", why);
            check.expect(!bytes.empty() && all_sentinel(bytes, 240, 320) && probe_prims == 6, "probe: 240 bytes written, the other 80 untouched; the query counts 6 points");

            // Incomplete primitives are not recorded (section 2.15.2): 98 vertices as TRIANGLES -> 32 triangles = 96 vertices.
            allocate(scratch_[1], (kMeshPadded + 2) * sizeof(MeshVertex), kSentinel);
            mgl::glBindBufferBase(kTF, 0, scratch_[1]);
            mgl::glUseProgram(prog_warp_);
            mgl::glBindVertexArray(vao_[kVaoMeshIn]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Triangles);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, kMeshVertices + 2);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            const GLuint tri_prims = query_result(query_);
            const std::vector<std::uint8_t> tri_bytes = read_buffer(scratch_[1], (kMeshPadded + 2) * sizeof(MeshVertex));
            check.expect(tri_prims == 32 && !tri_bytes.empty() && all_sentinel(tri_bytes, kMeshVertices * sizeof(MeshVertex), tri_bytes.size()),
                         "incomplete primitives are not recorded: a TRIANGLES draw of 98 vertices captures 32 triangles (96 vertices) and nothing after them",
                         std::format("primitives written {}", tri_prims));

            // SEPARATE_ATTRIBS with three buffers of their own (glBindBufferBase 0, 1, 2) instead of three ranges of one buffer.
            for (std::size_t i = 0; i < 3; ++i) allocate(scratch_[i], 2048, kSentinel);
            mgl::glUseProgram(prog_stream_);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferBase(kTF, i, scratch_[i]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kStreamPoints);
            mgl::glEndTransformFeedback();
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            std::vector<float> pos_want, color_want, size_want;
            for (int i = 0; i < kStreamPoints; ++i)
            {
                const StreamPoint p = stream_point(i);
                pos_want.insert(pos_want.end(), {p.x, p.y});
                color_want.insert(color_want.end(), {p.r, p.g, p.b});
                size_want.push_back(p.size);
            }
            const std::vector<std::uint8_t> b0 = read_buffer(scratch_[0], 2048), b1 = read_buffer(scratch_[1], 2048), b2 = read_buffer(scratch_[2], 2048);
            check.expect(!b0.empty() && from_bytes<float>(b0, 0, pos_want.size()) == pos_want && all_sentinel(b0, 96, 2048),
                         "SEPARATE_ATTRIBS, three buffers: buffer 0 holds only the 12 v_pos values (96 bytes), the rest untouched");
            check.expect(!b1.empty() && from_bytes<float>(b1, 0, color_want.size()) == color_want && all_sentinel(b1, 144, 2048), "... buffer 1 holds only the 12 v_color values (144 bytes)");
            check.expect(!b2.empty() && from_bytes<float>(b2, 0, size_want.size()) == size_want && all_sentinel(b2, 48, 2048), "... buffer 2 holds only the 12 v_size values (48 bytes)");

            // INTERLEAVED with extra bindings: only binding point 0 is written ("in interleaved mode, only the first buffer object binding point is ever written to").
            allocate(scratch_[0], 640, kSentinel);
            allocate(scratch_[1], 640, kSentinel);
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glBindBufferBase(kTF, 1, scratch_[1]);
            mgl::glUseProgram(prog_step_);
            mgl::glUniform1f(u_dt_, 0.0f);
            mgl::glBindVertexArray(vao_[kVaoPauseIn]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kPauseParticles);
            mgl::glEndTransformFeedback();
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            std::size_t diff = 0;
            const std::vector<std::uint8_t> i0 = read_buffer(scratch_[0], 640), i1 = read_buffer(scratch_[1], 640);
            check.expect(!i0.empty() && same_particles(from_bytes<Particle>(i0, 0, 8), pause_particles(), diff) && all_sentinel(i0, 256, 640) && !i1.empty() && all_sentinel(i1, 0, 640),
                         "INTERLEAVED_ATTRIBS: all four varyings go to binding point 0 (8 x 32 bytes); a buffer at binding point 1 stays untouched");
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferBase(kTF, i, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            check.expect_no_gl_error("the capture sections raised no GL error");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: transform feedback objects (per-object state)
        // ---------------------------------------------------------------------------------------------
        void check_objects(demo::Check& check)
        {
            const GLuint max = static_cast<GLuint>(init_.max_separate_attribs);
            check.expect(init_.max_separate_attribs >= 4, std::format("MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS = {} (minimum 4, Table 6.34)", init_.max_separate_attribs));
            check.expect(init_.tf_binding == 0 && !init_.active, "initially the default object (name 0) is bound and transform feedback is inactive (Tables 6.33, 6.24)");
            check.expect(std::all_of(init_.bindings.begin(), init_.bindings.end(), [](GLint b) { return b == 0; }) && init_.bindings.size() == max,
                         std::format("initially none of the {} indexed TRANSFORM_FEEDBACK_BUFFER bindings has a buffer", max));
            check.expect(!init_.discard_enabled, "RASTERIZER_DISCARD is initially disabled (Table 6.9)");
            check.expect(!mgl::glIsTransformFeedback(mgl::TransformFeedbackId{0}), "glIsTransformFeedback(0) is FALSE (section 6.1.11)");

            // A name from glGenTransformFeedbacks "does not acquire transform feedback state until it is first bound" (section 2.15.1).
            check.expect(!init_.is_tf_before_bind[0] && !init_.is_tf_before_bind[1], "a generated but never bound name is not (yet) a transform feedback object: glIsTransformFeedback = FALSE");
            // Per-object buffer bindings: the two ping-pong objects were set up in init() with different buffers.
            for (std::size_t i = 0; i < 2; ++i)
            {
                mgl::glBindTransformFeedback(kTFObject, tf_[i]);
                check.expect(mgl::glIsTransformFeedback(tf_[i]) && demo::get_integer(mgl::GetParameter::TransformFeedbackBinding) == static_cast<int>(tf_[i].value),
                             std::format("object {}: glIsTransformFeedback is TRUE after the first bind and TRANSFORM_FEEDBACK_BINDING names it", i));
                check.expect(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 0) == static_cast<GLint>(particles_[i].value),
                             std::format("object {}: binding point 0 holds its own buffer (ping-pong buffer {})", i, i));
                check.expect(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 1) == 0, std::format("object {}: binding point 1 is empty", i));
            }
            // BindBufferBase leaves START and SIZE at 0 ("not specified"); BindBufferRange records them.
            mgl::glBindTransformFeedback(kTFObject, tf_[0]);
            check.expect(get_indexed64(mgl::GetParameter::TransformFeedbackBufferStart, 0) == 0 && get_indexed64(mgl::GetParameter::TransformFeedbackBufferSize, 0) == 0,
                         "object 0 was bound with glBindBufferBase: TRANSFORM_FEEDBACK_BUFFER_START = SIZE = 0");
            mgl::glBindTransformFeedback(kTFObject, tf_[1]);
            check.expect(get_indexed64(mgl::GetParameter::TransformFeedbackBufferStart, 0) == 0 &&
                             get_indexed64(mgl::GetParameter::TransformFeedbackBufferSize, 0) == static_cast<GLint64>(kParticles * sizeof(Particle)),
                         "object 1 was bound with glBindBufferRange(0, 1024): START = 0, SIZE = 1024");
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            check.expect(demo::get_integer(mgl::GetParameter::TransformFeedbackBinding) == 0, "glBindTransformFeedback(0) selects the default object again");

            // The default object has bindings of its own: set a range at index 1 and see that the other objects do not change.
            allocate(scratch_[0], 256, kSentinel);
            mgl::glBindBufferRange(kTF, 1, scratch_[0], 36, 64);
            check.expect(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 1) == static_cast<GLint>(scratch_[0].value) &&
                             get_indexed64(mgl::GetParameter::TransformFeedbackBufferStart, 1) == 36 && get_indexed64(mgl::GetParameter::TransformFeedbackBufferSize, 1) == 64,
                         "default object: glBindBufferRange(index 1, offset 36, size 64) is reported by the indexed queries (BINDING, START = 36, SIZE = 64)");
            mgl::glBindTransformFeedback(kTFObject, tf_[0]);
            check.expect(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 1) == 0, "... and object 0 still has nothing at index 1 (the binding state belongs to the object)");
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glBindBufferBase(kTF, 1, scratch_[0]);
            check.expect(get_indexed64(mgl::GetParameter::TransformFeedbackBufferStart, 1) == 0 && get_indexed64(mgl::GetParameter::TransformFeedbackBufferSize, 1) == 0,
                         "glBindBufferBase on a range binding resets START and SIZE to 0");
            mgl::glBindBufferBase(kTF, 1, mgl::BufferId{0});
            check.expect(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 1) == 0, "glBindBufferBase(index, 0) removes the binding");
            (void)get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, max);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "an indexed query with index = MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS -> INVALID_VALUE (section 6.1.9)");

            // Lifecycle: bind, delete, use after delete.
            mgl::TransformFeedbackId t{};
            mgl::glGenTransformFeedbacks(1, &t);
            mgl::glBindTransformFeedback(kTFObject, t);
            check.expect(mgl::glIsTransformFeedback(t), "a new object is a transform feedback object after its first bind");
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glDeleteTransformFeedbacks(1, &t);
            check.expect_no_gl_error("glDeleteTransformFeedbacks of an inactive object raises no error");
            check.expect(!mgl::glIsTransformFeedback(t), "a deleted object is no longer a transform feedback object");
            mgl::glBindTransformFeedback(kTFObject, t);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBindTransformFeedback of a deleted name -> INVALID_OPERATION (section 2.15.1)");
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0x7ffffff0u});
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBindTransformFeedback of a name never returned by glGenTransformFeedbacks -> INVALID_OPERATION");
            const mgl::TransformFeedbackId zero{0};
            mgl::glDeleteTransformFeedbacks(1, &zero);
            const mgl::TransformFeedbackId unused{0x7ffffff1u};
            mgl::glDeleteTransformFeedbacks(1, &unused);
            check.expect_no_gl_error("deleting the default object name 0 and an unused name is silently ignored (section 2.15.1)");
            check.expect(demo::get_integer(mgl::GetParameter::TransformFeedbackBinding) == 0, "the failed binds left the default object bound");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: pause / resume rules (the data side is checked in check_captured_data)
        // ---------------------------------------------------------------------------------------------
        void check_pause_resume(demo::Check& check)
        {
            using Obj = mgl::TransformFeedbackId;
            auto state = [](bool active, bool paused) { return std::pair{active, paused}; };
            auto current = [] { return std::pair{get_bool(mgl::GetParameter::TransformFeedbackActive), get_bool(mgl::GetParameter::TransformFeedbackPaused)}; };

            allocate(scratch_[0], 256, kSentinel);  // object tf_[0]'s buffer is particles_[0]; the default object gets scratch_[0]
            mgl::glBindTransformFeedback(kTFObject, Obj{0});
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glUseProgram(prog_step_);
            mgl::glUniform1f(u_dt_, 0.0f);
            mgl::glBindVertexArray(vao_[kVaoPauseIn]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);

            check.expect(current() == state(false, false), "TRANSFORM_FEEDBACK_ACTIVE and _PAUSED are FALSE before glBeginTransformFeedback");
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            check.expect(current() == state(true, false), "after glBeginTransformFeedback: ACTIVE = TRUE, PAUSED = FALSE");
            mgl::glPauseTransformFeedback();
            check.expect(current() == state(true, true), "after glPauseTransformFeedback: still ACTIVE (\"paused is still considered active\"), PAUSED = TRUE");
            mgl::glResumeTransformFeedback();
            check.expect(current() == state(true, false), "after glResumeTransformFeedback: ACTIVE = TRUE, PAUSED = FALSE");
            mgl::glEndTransformFeedback();
            check.expect(current() == state(false, false), "after glEndTransformFeedback both are FALSE");
            check.expect_no_gl_error("the begin/pause/resume/end sequence raised no error");

            // EndTransformFeedback performs an implicit resume (section 2.15.2).
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glPauseTransformFeedback();
            mgl::glEndTransformFeedback();
            check.expect(current() == state(false, false), "glEndTransformFeedback while paused ends the capture (implicit resume): ACTIVE = PAUSED = FALSE");
            check.expect_no_gl_error("... without an error");

            // While paused, another transform feedback object can be bound, used for a complete capture, and the first one resumed.
            allocate(particles_[0], kParticles * sizeof(Particle), kSentinel);
            allocate(particles_[1], kParticles * sizeof(Particle), kSentinel);
            mgl::glBindTransformFeedback(kTFObject, tf_[0]);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 2);  // 0, 1 -> object 0
            mgl::glPauseTransformFeedback();
            mgl::glBindTransformFeedback(kTFObject, tf_[1]);
            check.expect_no_gl_error("glBindTransformFeedback while the current object is paused is allowed (section 2.15.2)");
            check.expect(current() == state(false, false), "the newly bound object reports its own state: ACTIVE = PAUSED = FALSE");
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 2, 3);  // 2, 3, 4 -> object 1
            mgl::glEndTransformFeedback();
            const bool second_ok = demo::drain_gl_errors() == mgl::ErrorCode::NoError;
            mgl::glBindTransformFeedback(kTFObject, tf_[0]);
            check.expect(current() == state(true, true), "rebinding the first object: it is still ACTIVE and PAUSED");
            mgl::glResumeTransformFeedback();
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 5, 2);  // 5, 6 -> object 0, after 0 and 1
            mgl::glEndTransformFeedback();
            check.expect(second_ok && demo::drain_gl_errors() == mgl::ErrorCode::NoError, "a complete capture on a second object while the first is paused, then resume of the first: no error");
            std::size_t diff = 0;
            const std::vector<std::uint8_t> a = read_buffer(particles_[0], kParticles * sizeof(Particle)), b = read_buffer(particles_[1], kParticles * sizeof(Particle));
            check.expect(!a.empty() && same_particles(from_bytes<Particle>(a, 0, 4), select(pause_particles(), {0, 1, 5, 6}), diff) && all_sentinel(a, 4 * sizeof(Particle), a.size()),
                         "object 0 recorded particles 0, 1 before the pause and 5, 6 after the resume, back to back (4 records, nothing else)", std::format("first difference at record {}", diff));
            check.expect(!b.empty() && same_particles(from_bytes<Particle>(b, 0, 3), select(pause_particles(), {2, 3, 4}), diff) && all_sentinel(b, 3 * sizeof(Particle), b.size()),
                         "object 1 recorded particles 2, 3, 4 into its own buffer meanwhile (3 records, nothing else)", std::format("first difference at record {}", diff));

            // Program rules while paused (section 2.15.2): UseProgram is allowed, but resuming with another program is an error.
            mgl::glBindTransformFeedback(kTFObject, Obj{0});
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glPauseTransformFeedback();
            mgl::glUseProgram(prog_render_);
            check.expect_no_gl_error("glUseProgram of another program while paused is allowed");
            mgl::glResumeTransformFeedback();
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glResumeTransformFeedback while another program is current -> INVALID_OPERATION");
            check.expect(current() == state(true, true), "... and the capture is still paused");
            mgl::glUseProgram(prog_step_);
            mgl::glResumeTransformFeedback();
            check.expect_no_gl_error("with the original program current again, glResumeTransformFeedback succeeds");
            mgl::glEndTransformFeedback();

            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindBufferBase(kTF, 0, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---------------------------------------------------------------------------------------------
        // validate: primitive queries (ES 3.0 sections 2.14, 2.16, 6.1.7)
        // ---------------------------------------------------------------------------------------------
        void check_queries(demo::Check& check)
        {
            check.expect(!init_.is_query_before_begin, "a generated query name is not a query object before its first glBeginQuery (section 2.14)");
            GLint current = -1;
            mgl::glGetQueryiv(kPrimQuery, mgl::QueryParameter::CurrentQuery, &current);
            check.expect_eq(current, 0, "CURRENT_QUERY of TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN is 0 when no query is active");

            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            allocate(scratch_[0], 1024, kSentinel);
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glUseProgram(prog_step_);
            mgl::glUniform1f(u_dt_, 0.0f);
            mgl::glBindVertexArray(vao_[kVaoPauseIn]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);

            // Count of a query that is active while transform feedback is not: the counter does not move.
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glGetQueryiv(kPrimQuery, mgl::QueryParameter::CurrentQuery, &current);
            check.expect_eq(current, static_cast<GLint>(query_.value), "CURRENT_QUERY is the query's name while it is active");
            check.expect(mgl::glIsQuery(query_), "glIsQuery is TRUE after the first glBeginQuery");
            GLuint probe = 0;
            mgl::glGetQueryObjectuiv(query_, mgl::QueryObjectParameter::Result, &probe);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetQueryObjectuiv of the active query -> INVALID_OPERATION (section 6.1.7)");
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 5);  // transform feedback inactive: nothing is recorded
            mgl::glEndQuery(kPrimQuery);
            check.expect_eq(query_result(query_), 0u, "TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN is 0 for draws made while transform feedback was inactive (section 2.16)");
            mgl::glGetQueryiv(kPrimQuery, mgl::QueryParameter::CurrentQuery, &current);
            check.expect_eq(current, 0, "CURRENT_QUERY is 0 again after glEndQuery");

            // Two draws in one capture add up; the next glBeginQuery starts again from zero.
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 3);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 3, 5);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            GLuint available = 0;
            const auto start = std::chrono::steady_clock::now();
            while (available == 0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(5))
                mgl::glGetQueryObjectuiv(query_, mgl::QueryObjectParameter::ResultAvailable, &available);
            check.expect(available == 1, "QUERY_RESULT_AVAILABLE becomes TRUE (guaranteed eventually, section 6.1.7)");
            check.expect_eq(query_result(query_), 8u, "two draws of 3 and 5 points in one capture: the query result is 8");

            mgl::glBeginQuery(kPrimQuery, query_);  // same query object, new count
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 2);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            check.expect_eq(query_result(query_), 2u, "glBeginQuery resets the count: a second use of the query object reports 2, not 10");

            // A query opened inside the capture and closed after it counts the same.
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 4);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            check.expect_eq(query_result(query_), 4u, "a query that starts after glBeginTransformFeedback and ends after glEndTransformFeedback counts the 4 recorded points");
            check.expect_no_gl_error("the query sections raised no unexpected error");

            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindBufferBase(kTF, 0, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---------------------------------------------------------------------------------------------
        // validate: RASTERIZER_DISCARD (ES 3.0 section 3.1)
        // ---------------------------------------------------------------------------------------------
        void check_rasterizer_discard(const demo::Env& env, demo::Check& check)
        {
            const Rect r = cell_rect(env.width, env.height, kCellPause);
            const std::vector<Particle> row = pause_particles();
            const std::vector<Prim> red_row = [&] {
                std::vector<Prim> v;
                for (const Particle& p : row) v.push_back(make_square(p.x, p.y, p.size, 1.0f, 0.0f, 0.0f, r.w, r.h));
                return v;
            }();

            auto clear_cell = [&] {
                mgl::glEnable(mgl::Capability::ScissorTest);
                mgl::glScissor(r.x, r.y, r.w, r.h);
                mgl::glClear(mgl::ClearBufferBit::Color);
                mgl::glDisable(mgl::Capability::ScissorTest);
            };
            auto capture_row = [&](bool discard) {
                allocate(scratch_[0], kPauseParticles * sizeof(Particle), kSentinel);
                mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
                mgl::glBindBufferBase(kTF, 0, scratch_[0]);
                mgl::glUseProgram(prog_step_);
                mgl::glUniform1f(u_dt_, 0.0f);
                mgl::glBindVertexArray(vao_[kVaoPauseIn]);
                mgl::glViewport(r.x, r.y, r.w, r.h);
                if (discard) mgl::glEnable(mgl::Capability::RasterizerDiscard);
                else mgl::glDisable(mgl::Capability::RasterizerDiscard);
                mgl::glBeginQuery(kPrimQuery, query_);
                mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
                mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kPauseParticles);
                mgl::glEndTransformFeedback();
                mgl::glEndQuery(kPrimQuery);
                mgl::glDisable(mgl::Capability::RasterizerDiscard);
                return query_result(query_);
            };
            auto captured_ok = [&] {
                std::size_t diff = 0;
                const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], kPauseParticles * sizeof(Particle));
                return !bytes.empty() && same_particles(from_bytes<Particle>(bytes, 0, 8), row, diff);
            };

            check.expect(!mgl::glIsEnabled(mgl::Capability::RasterizerDiscard), "RASTERIZER_DISCARD is disabled outside the capture passes");
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            check.expect(mgl::glIsEnabled(mgl::Capability::RasterizerDiscard) && get_bool(mgl::GetParameter::RasterizerDiscard), "glEnable(RASTERIZER_DISCARD): glIsEnabled and glGetBooleanv report TRUE");
            mgl::glDisable(mgl::Capability::RasterizerDiscard);

            // Discard OFF: the same capture also rasterizes (8 red squares from gl_PointSize); the recorded data does not change.
            mgl::glUseProgram(prog_step_);
            mgl::glUniform4f(u_color_, 1.0f, 0.0f, 0.0f, 1.0f);
            clear_cell();
            const GLuint prims_visible = capture_row(false);
            const std::vector<std::uint8_t> visible = demo::read_rgba8(0, 0, env.width, env.height);
            const bool visible_data = captured_ok();
            expect_scan(check, scan_cell(visible, env.width, r, red_row), 400, 14000, "RASTERIZER_DISCARD disabled: the capture pass also draws the 8 red points");
            check.expect(prims_visible == 8 && visible_data, "RASTERIZER_DISCARD disabled: the 8 points are recorded as well (query 8, data equal)");

            // Discard ON: the identical capture leaves the framebuffer untouched; the recorded data is the same.
            clear_cell();
            const GLuint prims_hidden = capture_row(true);
            const std::vector<std::uint8_t> hidden = demo::read_rgba8(0, 0, env.width, env.height);
            const bool hidden_data = captured_ok();
            expect_scan(check, scan_cell(hidden, env.width, r, {}), 0, 16000, "RASTERIZER_DISCARD enabled: the same capture pass produces no pixel at all");
            check.expect(prims_hidden == 8 && hidden_data, "RASTERIZER_DISCARD enabled: the same 8 points are recorded (query 8, data equal): capture happens before the discard");
            check.expect(scan_cell(hidden, env.width, r, red_row).bad >= 100, "negative control: the discarded cell disagrees with the model of the visible one");

            // glClear and glClearBufferfv are ignored while RASTERIZER_DISCARD is enabled (section 3.1).
            mgl::glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glClear(mgl::ClearBufferBit::Color);
            const std::array<float, 4> blue = {0.0f, 0.0f, 1.0f, 1.0f};
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, blue.data());
            mgl::glDisable(mgl::Capability::ScissorTest);
            const std::vector<std::uint8_t> ignored = demo::read_rgba8(0, 0, env.width, env.height);
            expect_scan(check, scan_cell(ignored, env.width, r, {}), 0, 16000, "RASTERIZER_DISCARD enabled: glClear (green) and glClearBufferfv (blue) leave the cell at the old clear color");
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glClear(mgl::ClearBufferBit::Color);  // now it works: the same call clears to green
            mgl::glDisable(mgl::Capability::ScissorTest);
            check.expect_pixel(r.x + r.w / 2, r.y + r.h / 2, {0, 255, 0, 255}, 1, "negative control: with RASTERIZER_DISCARD disabled the same glClear paints the cell green");
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);

            mgl::glBindBufferBase(kTF, 0, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            check.expect_no_gl_error("the RASTERIZER_DISCARD sections raised no error");
        }

        // ---------------------------------------------------------------------------------------------
        // validate: the errors defined for transform feedback (ES 3.0 sections 2.12.8, 2.15)
        // ---------------------------------------------------------------------------------------------
        // Leaves transform feedback inactive whatever a failed expectation left behind, so one failure does not cascade.
        void settle()
        {
            if (get_bool(mgl::GetParameter::TransformFeedbackPaused)) mgl::glResumeTransformFeedback();
            if (get_bool(mgl::GetParameter::TransformFeedbackActive)) mgl::glEndTransformFeedback();
            (void)demo::drain_gl_errors();
        }
        // The default object with `buffer` (`bytes` long) at binding point 0, the step program in identity mode, the 8-particle VAO.
        void arm(mgl::BufferId buffer, std::size_t bytes)
        {
            allocate(buffer, bytes, kSentinel);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glBindBufferBase(kTF, 0, buffer);
            mgl::glUseProgram(prog_step_);
            mgl::glUniform1f(u_dt_, 0.0f);
            mgl::glBindVertexArray(vao_[kVaoPauseIn]);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            (void)demo::drain_gl_errors();
        }
        void disarm()
        {
            settle();
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferBase(kTF, i, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            (void)demo::drain_gl_errors();
        }

        void check_errors(const demo::Env& env, demo::Check& check)
        {
            using Err = mgl::ErrorCode;
            using Cap = mgl::TransformFeedbackPrimitive;
            using Mode = mgl::PrimitiveType;
            const auto drawElements = [](Mode mode) { mgl::glDrawElements(mode, 3, mgl::IndexType::UnsignedByte, demo::buffer_offset(0)); };
            const bool es32 = env.context_version >= demo::Version{3, 2};  // the capture rules changed in ES 3.2 (see the blocks below)
            check.expect_no_gl_error("the error section starts with an empty error queue");

            // --- pairing of begin / end / pause / resume (section 2.15.2) ------------------------------
            arm(scratch_[0], 256);
            mgl::glEndTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "glEndTransformFeedback while inactive -> INVALID_OPERATION");
            mgl::glPauseTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "glPauseTransformFeedback while inactive -> INVALID_OPERATION");
            mgl::glResumeTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "glResumeTransformFeedback while inactive -> INVALID_OPERATION");
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_no_gl_error("positive control: glBeginTransformFeedback(POINTS) with a buffer at binding point 0 succeeds");
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_gl_error(Err::InvalidOperation, "glBeginTransformFeedback while already active -> INVALID_OPERATION");
            mgl::glResumeTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "glResumeTransformFeedback while active but not paused -> INVALID_OPERATION");
            mgl::glPauseTransformFeedback();
            mgl::glPauseTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "glPauseTransformFeedback while already paused -> INVALID_OPERATION (the first pause succeeded)");
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_gl_error(Err::InvalidOperation, "glBeginTransformFeedback while paused (still active) -> INVALID_OPERATION");
            settle();
            mgl::glBeginTransformFeedback(Mode::TriangleStrip);  // the broad PrimitiveType overload of meta-gl: no cast needed
            check.expect_gl_error(Err::InvalidEnum, "glBeginTransformFeedback(TRIANGLE_STRIP): primitiveMode must be POINTS, LINES or TRIANGLES -> INVALID_ENUM");
            mgl::glEndTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "... and it did not start a capture (glEndTransformFeedback -> INVALID_OPERATION)");

            // --- the draw's primitive type must be allowed by primitiveMode (section 2.15.2) --------------------
            // ES 3.0 and 3.1: DrawArrays/DrawArraysInstanced need mode == primitiveMode; every other drawing command is
            // INVALID_OPERATION while capturing, "regardless of mode". ES 3.2 (Table 12.1) allows strips, fans and loops
            // of the captured primitive type and indexed draws, and records strips/fans/loops as separate primitives.
            auto ids_of = [&](const std::vector<std::uint8_t>& bytes, std::size_t first, std::size_t count) {
                std::vector<int> ids;
                for (const Particle& p : from_bytes<Particle>(bytes, first * sizeof(Particle), count)) ids.push_back(static_cast<int>(std::lround((p.x + 0.875f) / 0.25f)));
                return ids;
            };
            auto runs_equal = [](const std::vector<int>& got, std::vector<std::vector<int>> want) {  // per primitive, vertex order ignored
                std::size_t at = 0;
                for (auto& prim : want)
                {
                    if (at + prim.size() > got.size()) return false;
                    std::vector<int> have(got.begin() + static_cast<std::ptrdiff_t>(at), got.begin() + static_cast<std::ptrdiff_t>(at + prim.size()));
                    std::sort(have.begin(), have.end());
                    std::sort(prim.begin(), prim.end());
                    if (have != prim) return false;
                    at += prim.size();
                }
                return at == got.size();
            };
            const char* strict_note = "Mesa applies the ES 3.2 rule (Table 12.1) in every context version: MESA_GLES_VERSION_OVERRIDE changes only the reported version";

            arm(scratch_[0], 1024);  // room for 32 particles: nothing below overflows
            mgl::glBeginTransformFeedback(Cap::Points);
            mgl::glDrawArrays(Mode::Points, 0, 3);
            check.expect_no_gl_error("positive control: glDrawArrays(POINTS) while capturing POINTS succeeds");
            mgl::glDrawArrays(Mode::Triangles, 0, 3);
            check.expect_gl_error(Err::InvalidOperation, "glDrawArrays(TRIANGLES) while capturing POINTS -> INVALID_OPERATION");
            mgl::glDrawArrays(Mode::Lines, 0, 2);
            check.expect_gl_error(Err::InvalidOperation, "glDrawArrays(LINES) while capturing POINTS -> INVALID_OPERATION");
            mgl::glDrawArraysInstanced(Mode::Triangles, 0, 3, 2);
            check.expect_gl_error(Err::InvalidOperation, "glDrawArraysInstanced(TRIANGLES) while capturing POINTS -> INVALID_OPERATION");
            if (es32)
            {
                drawElements(Mode::Points);
                check.expect_no_gl_error("ES 3.2: glDrawElements(POINTS) while capturing POINTS is allowed (Table 12.1)");
                mgl::glDrawElementsInstanced(Mode::Points, 3, mgl::IndexType::UnsignedByte, demo::buffer_offset(0), 1);
                check.expect_no_gl_error("ES 3.2: glDrawElementsInstanced(POINTS) while capturing POINTS is allowed");
                mgl::glDrawRangeElements(Mode::Points, 0, 2, 3, mgl::IndexType::UnsignedByte, demo::buffer_offset(0));
                check.expect_no_gl_error("ES 3.2: glDrawRangeElements(POINTS) while capturing POINTS is allowed");
                drawElements(Mode::Triangles);
                check.expect_gl_error(Err::InvalidOperation, "ES 3.2: glDrawElements(TRIANGLES) while capturing POINTS -> INVALID_OPERATION (not allowed by Table 12.1)");
            }
            else
            {
                drawElements(Mode::Points);
                check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: glDrawElements(POINTS) while capturing POINTS -> INVALID_OPERATION (indexed draws are rejected \"regardless of mode\")");
                mgl::glDrawElementsInstanced(Mode::Points, 3, mgl::IndexType::UnsignedByte, demo::buffer_offset(0), 1);
                check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: glDrawElementsInstanced(POINTS) while capturing POINTS -> INVALID_OPERATION");
                mgl::glDrawRangeElements(Mode::Points, 0, 2, 3, mgl::IndexType::UnsignedByte, demo::buffer_offset(0));
                check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: glDrawRangeElements(POINTS) while capturing POINTS -> INVALID_OPERATION");
            }
            mgl::glEndTransformFeedback();
            {
                const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], 1024);
                const std::size_t records = es32 ? 12 : 3;  // 3 from DrawArrays, and 3 more from each of the three indexed draws on ES 3.2
                std::vector<int> expected_ids;
                for (std::size_t i = 0; i < records; ++i) expected_ids.push_back(static_cast<int>(i % 3));
                check.expect(!bytes.empty() && ids_of(bytes, 0, records) == expected_ids && all_sentinel(bytes, records * sizeof(Particle), 1024),
                             std::format("POINTS capture: exactly the {} records of the accepted draws were written (the rejected draws recorded nothing)", records));
            }
            disarm();

            arm(scratch_[0], 1024);
            mgl::glBeginTransformFeedback(Cap::Triangles);
            mgl::glDrawArrays(Mode::Triangles, 0, 3);
            check.expect_no_gl_error("positive control: glDrawArrays(TRIANGLES) while capturing TRIANGLES succeeds");
            mgl::glDrawArrays(Mode::Points, 0, 3);
            check.expect_gl_error(Err::InvalidOperation, "glDrawArrays(POINTS) while capturing TRIANGLES -> INVALID_OPERATION");
            mgl::glDrawArrays(Mode::Lines, 0, 2);
            check.expect_gl_error(Err::InvalidOperation, "glDrawArrays(LINES) while capturing TRIANGLES -> INVALID_OPERATION");
            mgl::glDrawArrays(Mode::TriangleStrip, 0, 4);
            const bool strip_ok = demo::drain_gl_errors() == (es32 ? Err::NoError : Err::InvalidOperation);
            mgl::glDrawArrays(Mode::TriangleFan, 0, 4);
            const bool fan_ok = demo::drain_gl_errors() == (es32 ? Err::NoError : Err::InvalidOperation);
            mgl::glEndTransformFeedback();
            if (es32)
            {
                check.expect(strip_ok && fan_ok, "ES 3.2: glDrawArrays(TRIANGLE_STRIP) and (TRIANGLE_FAN) while capturing TRIANGLES are allowed (Table 12.1)");
                const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], 1024);
                check.expect(!bytes.empty() && runs_equal(ids_of(bytes, 0, 15), {{0, 1, 2}, {0, 1, 2}, {1, 2, 3}, {0, 1, 2}, {0, 2, 3}}) && all_sentinel(bytes, 15 * sizeof(Particle), 1024),
                             "ES 3.2: a 4-vertex strip is recorded as triangles (0,1,2) and (1,2,3), a 4-vertex fan as (0,1,2) and (0,2,3): 5 triangles, 15 records in all");
            }
            else
            {
                expect_mesa_deviation(check, strip_ok, "ES 3.0/3.1: glDrawArrays(TRIANGLE_STRIP) while capturing TRIANGLES -> INVALID_OPERATION (\"mode is not identical to primitiveMode\")",
                                      "glGetError() did not report INVALID_OPERATION", strict_note);
                expect_mesa_deviation(check, fan_ok, "ES 3.0/3.1: glDrawArrays(TRIANGLE_FAN) while capturing TRIANGLES -> INVALID_OPERATION",
                                      "glGetError() did not report INVALID_OPERATION", strict_note);
            }
            disarm();

            arm(scratch_[0], 1024);
            mgl::glBeginTransformFeedback(Cap::Lines);
            mgl::glDrawArrays(Mode::Lines, 0, 2);
            check.expect_no_gl_error("positive control: glDrawArrays(LINES) while capturing LINES succeeds");
            mgl::glDrawArrays(Mode::Triangles, 0, 3);
            check.expect_gl_error(Err::InvalidOperation, "glDrawArrays(TRIANGLES) while capturing LINES -> INVALID_OPERATION");
            mgl::glDrawArrays(Mode::LineStrip, 0, 3);
            const bool line_strip_ok = demo::drain_gl_errors() == (es32 ? Err::NoError : Err::InvalidOperation);
            mgl::glDrawArrays(Mode::LineLoop, 0, 3);
            const bool line_loop_ok = demo::drain_gl_errors() == (es32 ? Err::NoError : Err::InvalidOperation);
            mgl::glEndTransformFeedback();
            if (es32)
            {
                check.expect(line_strip_ok && line_loop_ok, "ES 3.2: glDrawArrays(LINE_STRIP) and (LINE_LOOP) while capturing LINES are allowed (Table 12.1)");
                const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], 1024);
                check.expect(!bytes.empty() && runs_equal(ids_of(bytes, 0, 12), {{0, 1}, {0, 1}, {1, 2}, {0, 1}, {1, 2}, {2, 0}}) && all_sentinel(bytes, 12 * sizeof(Particle), 1024),
                             "ES 3.2: a 3-vertex line strip is recorded as lines (0,1) and (1,2), a 3-vertex loop as (0,1), (1,2) and (2,0): 6 lines, 12 records in all");
            }
            else
            {
                expect_mesa_deviation(check, line_strip_ok, "ES 3.0/3.1: glDrawArrays(LINE_STRIP) while capturing LINES -> INVALID_OPERATION", "glGetError() did not report INVALID_OPERATION", strict_note);
                expect_mesa_deviation(check, line_loop_ok, "ES 3.0/3.1: glDrawArrays(LINE_LOOP) while capturing LINES -> INVALID_OPERATION", "glGetError() did not report INVALID_OPERATION", strict_note);
            }
            disarm();

            // --- insufficient buffer space (ES 3.0 / 3.1 section 2.15.2 / 12.1; ES 3.2 section 12.2.2 and 12.3) ----------
            // ES 3.0/3.1: DrawArrays and DrawArraysInstanced are INVALID_OPERATION when the capture would exceed the buffer
            // (or the BindBufferRange limit offset + size - 1), and nothing is recorded. ES 3.2 removed that error: a
            // primitive that does not fit is simply not recorded and not counted ("does not fit in a buffer object").
            arm(scratch_[0], 4 * sizeof(Particle));  // room for 4 particles
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(Cap::Points);
            mgl::glDrawArrays(Mode::Points, 0, 5);
            if (es32) check.expect_no_gl_error("ES 3.2: glDrawArrays of 5 points into a buffer with room for 4 raises no error");
            else check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: glDrawArrays of 5 points into a buffer with room for 4 -> INVALID_OPERATION");
            mgl::glDrawArraysInstanced(Mode::Points, 0, 2, 3);
            if (es32) check.expect_no_gl_error("ES 3.2: glDrawArraysInstanced of 2 x 3 points into a (full) buffer raises no error");
            else check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: glDrawArraysInstanced of 2 x 3 = 6 points into a buffer with room for 4 -> INVALID_OPERATION");
            if (!es32)
            {
                mgl::glDrawArrays(Mode::Points, 0, 4);
                check.expect_no_gl_error("ES 3.0/3.1: a draw of exactly 4 points fits: no error (so the two failed draws above recorded nothing)");
                mgl::glDrawArrays(Mode::Points, 0, 1);
                check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: one more point into the full buffer -> INVALID_OPERATION");
            }
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            {
                std::size_t diff = 0;
                const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], 4 * sizeof(Particle));
                check.expect(!bytes.empty() && same_particles(from_bytes<Particle>(bytes, 0, 4), select(pause_particles(), {0, 1, 2, 3}), diff),
                             es32 ? "ES 3.2: the buffer holds the first 4 of the 5 points; the fifth, which did not fit, was dropped"
                                  : "the buffer holds exactly the 4 particles of the successful draw",
                             std::format("first difference at record {}", diff));
                check.expect_eq(query_result(query_), 4u, "TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 4: primitives that did not fit are not counted");
            }
            disarm();

            arm(scratch_[0], 256);
            mgl::glBindBufferRange(kTF, 0, scratch_[0], 32, 3 * sizeof(Particle));  // the range [32, 128): room for 3 particles although the buffer is larger
            mgl::glBeginQuery(kPrimQuery, query_);
            mgl::glBeginTransformFeedback(Cap::Points);
            mgl::glDrawArrays(Mode::Points, 0, 4);
            if (es32) check.expect_no_gl_error("ES 3.2: glDrawArrays of 4 points into a 96-byte range of a 256-byte buffer raises no error");
            else check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: glDrawArrays of 4 points into a 96-byte range of a 256-byte buffer -> INVALID_OPERATION (limit is offset + size - 1)");
            if (!es32)
            {
                mgl::glDrawArrays(Mode::Points, 0, 3);
                check.expect_no_gl_error("ES 3.0/3.1: a draw of exactly 3 points fits the range");
                mgl::glDrawArrays(Mode::Points, 0, 1);
                check.expect_gl_error(Err::InvalidOperation, "ES 3.0/3.1: one more point -> INVALID_OPERATION although the buffer has room beyond the range");
            }
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(kPrimQuery);
            {
                std::size_t diff = 0;
                const std::vector<std::uint8_t> bytes = read_buffer(scratch_[0], 256);
                check.expect(!bytes.empty() && all_sentinel(bytes, 0, 32) && same_particles(from_bytes<Particle>(bytes, 32, 3), select(pause_particles(), {0, 1, 2}), diff) && all_sentinel(bytes, 128, 256),
                             "the capture started at byte 32 (BindBufferRange offset) and wrote nothing before it or after the range end (byte 128)",
                             std::format("first difference at record {}", diff));
                check.expect_eq(query_result(query_), 3u, "TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 3 for the 3 points that fit the range");
            }
            disarm();

            // --- changing state while active (section 2.15.2) ---------------------------------------------
            arm(scratch_[0], 256);
            allocate(scratch_[1], 256, kSentinel);
            mgl::glBeginTransformFeedback(Cap::Points);
            mgl::glUseProgram(prog_render_);
            check.expect_gl_error(Err::InvalidOperation, "glUseProgram while active and not paused -> INVALID_OPERATION");
            check.expect_or_known_deviation(demo::get_integer(mgl::GetParameter::CurrentProgram) == static_cast<int>(prog_step_.value), "... and the current program did not change (a command that generates an error has no effect, section 2.5)",
                                            std::format("CURRENT_PROGRAM = {}, expected {}", demo::get_integer(mgl::GetParameter::CurrentProgram), prog_step_.value), "radeonsi",
                                            "radeonsi raises INVALID_OPERATION but still makes the rejected program current");
            mgl::glBindBufferBase(kTF, 0, scratch_[1]);
            check.expect_gl_error(Err::InvalidOperation, "glBindBufferBase(TRANSFORM_FEEDBACK_BUFFER) while active -> INVALID_OPERATION");
            mgl::glBindBufferRange(kTF, 0, scratch_[1], 0, 64);
            check.expect_gl_error(Err::InvalidOperation, "glBindBufferRange(TRANSFORM_FEEDBACK_BUFFER) while active -> INVALID_OPERATION");
            check.expect_eq(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 0), static_cast<GLint>(scratch_[0].value), "... and binding point 0 still has the original buffer");
            mgl::glLinkProgram(prog_step_);
            check.expect_gl_error(Err::InvalidOperation, "glLinkProgram of the program in use by the active capture -> INVALID_OPERATION");
            GLint linked = GL_FALSE;
            mgl::glGetProgramiv(prog_step_, mgl::ProgramParameter::LinkStatus, &linked);
            check.expect_eq(linked, GL_TRUE, "... and the program is still linked");
            mgl::glBindTransformFeedback(kTFObject, tf_[0]);
            check.expect_gl_error(Err::InvalidOperation, "glBindTransformFeedback while the current object is active and not paused -> INVALID_OPERATION");
            check.expect_eq(demo::get_integer(mgl::GetParameter::TransformFeedbackBinding), 0, "... and the default object is still bound");
            mgl::glDrawArrays(Mode::Points, 0, 2);
            check.expect_no_gl_error("the capture still works after all the rejected changes");
            mgl::glEndTransformFeedback();
            mgl::glUseProgram(prog_render_);
            mgl::glUseProgram(prog_step_);
            mgl::glBindBufferBase(kTF, 0, scratch_[1]);
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            check.expect_no_gl_error("after glEndTransformFeedback, glUseProgram and glBindBufferBase are allowed again");
            disarm();

            // The object that is active cannot be deleted; while paused, buffer bindings still cannot change.
            arm(scratch_[0], 256);
            mgl::glBindTransformFeedback(kTFObject, tf_misc_);
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glBeginTransformFeedback(Cap::Points);
            mgl::glDeleteTransformFeedbacks(1, &tf_misc_);
            check.expect_gl_error(Err::InvalidOperation, "glDeleteTransformFeedbacks of the active object -> INVALID_OPERATION");
            check.expect(mgl::glIsTransformFeedback(tf_misc_), "... and the object still exists");
            mgl::glPauseTransformFeedback();
            mgl::glBindBufferBase(kTF, 0, scratch_[1]);
            check.expect_gl_error(Err::InvalidOperation, "glBindBufferBase(TRANSFORM_FEEDBACK_BUFFER) while paused (still active) -> INVALID_OPERATION");
            mgl::glUseProgram(prog_render_);
            mgl::glUseProgram(prog_step_);
            mgl::glResumeTransformFeedback();
            mgl::glEndTransformFeedback();
            check.expect_no_gl_error("pause, UseProgram round trip, resume and end succeed");
            mgl::glBindBufferBase(kTF, 0, mgl::BufferId{0});
            disarm();

            // --- glBeginTransformFeedback preconditions (section 2.15.2) ------------------------------------
            allocate(scratch_[0], 256, kSentinel);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glUseProgram(prog_step_);
            mgl::glBindBufferBase(kTF, 0, mgl::BufferId{0});
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_gl_error(Err::InvalidOperation, "glBeginTransformFeedback with no buffer bound at binding point 0 -> INVALID_OPERATION");
            mgl::glEndTransformFeedback();
            check.expect_gl_error(Err::InvalidOperation, "... and no capture was started (glEndTransformFeedback -> INVALID_OPERATION)");
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_gl_error(Err::InvalidOperation, "glBeginTransformFeedback with no current program -> INVALID_OPERATION");
            mgl::glUseProgram(prog_render_);
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_gl_error(Err::InvalidOperation, "glBeginTransformFeedback with a program that specified no varyings -> INVALID_OPERATION");
            mgl::glUseProgram(prog_stream_);  // SEPARATE_ATTRIBS with three varyings: all three binding points must have buffers
            allocate(scratch_[1], 256, kSentinel);
            allocate(scratch_[2], 256, kSentinel);
            mgl::glBindBufferBase(kTF, 0, scratch_[0]);
            mgl::glBindBufferBase(kTF, 1, scratch_[1]);
            mgl::glBindBufferBase(kTF, 2, mgl::BufferId{0});
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_gl_error(Err::InvalidOperation, "SEPARATE_ATTRIBS with 3 varyings and buffers at binding points 0 and 1 only -> INVALID_OPERATION");
            mgl::glBindBufferBase(kTF, 2, scratch_[2]);
            mgl::glBeginTransformFeedback(Cap::Points);
            check.expect_no_gl_error("positive control: with a buffer at binding point 2 as well, glBeginTransformFeedback succeeds");
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            mgl::glDrawArrays(Mode::Points, 0, 2);
            mgl::glEndTransformFeedback();
            check.expect_no_gl_error("... and a capture of 2 points ends without error");
            disarm();

            // --- binding point arguments (sections 2.10.1.1, 2.15.2) --------------------------------------
            allocate(scratch_[0], 256, kSentinel);
            mgl::glBindTransformFeedback(kTFObject, mgl::TransformFeedbackId{0});
            mgl::glBindBufferRange(kTF, 0, scratch_[0], 8, 64);
            check.expect_no_gl_error("positive control: glBindBufferRange(offset 8, size 64) succeeds");
            const GLuint max = static_cast<GLuint>(init_.max_separate_attribs);
            mgl::glBindBufferRange(kTF, 0, scratch_[0], 2, 64);
            check.expect_gl_error(Err::InvalidValue, "glBindBufferRange with offset 2 (not a multiple of 4) -> INVALID_VALUE");
            mgl::glBindBufferRange(kTF, 0, scratch_[0], 4, 6);
            check.expect_gl_error(Err::InvalidValue, "glBindBufferRange with size 6 (not a multiple of 4) -> INVALID_VALUE");
            mgl::glBindBufferRange(kTF, 0, scratch_[0], 0, 0);
            check.expect_gl_error(Err::InvalidValue, "glBindBufferRange with size 0 and a non-zero buffer -> INVALID_VALUE");
            mgl::glBindBufferRange(kTF, 0, scratch_[0], -4, 64);
            check.expect_gl_error(Err::InvalidValue, "glBindBufferRange with a negative offset -> INVALID_VALUE");
            mgl::glBindBufferRange(kTF, max, scratch_[0], 0, 64);
            check.expect_gl_error(Err::InvalidValue, "glBindBufferRange with index = MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS -> INVALID_VALUE");
            mgl::glBindBufferBase(kTF, max, scratch_[0]);
            check.expect_gl_error(Err::InvalidValue, "glBindBufferBase with index = MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS -> INVALID_VALUE");
            check.expect(get_indexed(mgl::GetParameter::TransformFeedbackBufferBinding, 0) == static_cast<GLint>(scratch_[0].value) &&
                             get_indexed64(mgl::GetParameter::TransformFeedbackBufferStart, 0) == 8 && get_indexed64(mgl::GetParameter::TransformFeedbackBufferSize, 0) == 64,
                         "all six failed calls left the binding of the first call unchanged (START 8, SIZE 64)");
            mgl::glBindBufferBase(kTF, 0, mgl::BufferId{0});

            // --- glTransformFeedbackVaryings and linking (section 2.12.8) ---------------------------------
            {
                std::vector<const GLchar*> names(static_cast<std::size_t>(max) + 1, "v_pos");
                mgl::glTransformFeedbackVaryings(prog_stream_, static_cast<GLsizei>(names.size()), names.data(), mgl::TransformFeedbackBufferMode::SeparateAttribs);
                check.expect_gl_error(Err::InvalidValue, "glTransformFeedbackVaryings with SEPARATE_ATTRIBS and count = MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS + 1 -> INVALID_VALUE");
                bool linked_ok = true;
                std::string log;
                const GLchar* unknown[] = {"v_pos", "v_does_not_exist"};
                mgl::ProgramId p = link_tf_program(env, "shaders/step.vert", unknown, mgl::TransformFeedbackBufferMode::InterleavedAttribs, &linked_ok, &log);
                check.expect(!linked_ok, "a varying name that is not a vertex shader output makes the program fail to link", log);
                mgl::glDeleteProgram(p);
                const GLchar* twice[] = {"v_pos", "v_vel", "v_pos"};
                p = link_tf_program(env, "shaders/step.vert", twice, mgl::TransformFeedbackBufferMode::InterleavedAttribs, &linked_ok, &log);
                check.expect(!linked_ok, "naming the same output twice makes the program fail to link", log);
                mgl::glDeleteProgram(p);
                const GLchar* fine[] = {"v_vel", "v_pos"};
                p = link_tf_program(env, "shaders/step.vert", fine, mgl::TransformFeedbackBufferMode::SeparateAttribs, &linked_ok, &log);
                check.expect(linked_ok, "positive control: two valid, distinct names (in a different order) link", log);
                check.expect(get_varying(p, 0).name == "v_vel" && get_varying(p, 1).name == "v_pos", "... and the list keeps the order given to glTransformFeedbackVaryings (v_vel first)");
                mgl::glDeleteProgram(p);
            }
            check.expect_no_gl_error("the error section ends with an empty error queue");
        }

        // ---------------------------------------------------------------------------------------------
        // State
        // ---------------------------------------------------------------------------------------------
        struct InitState
        {
            int max_separate_attribs = 0;
            int tf_binding = -1;
            std::vector<GLint> bindings;
            bool discard_enabled = true;
            bool active = true;
            std::vector<bool> is_tf_before_bind;
            bool is_query_before_begin = true;
        } init_;

        mgl::ProgramId prog_step_{}, prog_warp_{}, prog_stream_{}, prog_probe_{}, prog_render_{};
        mgl::UniformLocation u_dt_{}, u_color_{};
        mgl::BufferId vbo_mesh_{}, tf_mesh_{}, pause_in_{}, pause_ebo_{}, pause_out_{}, stream_buf_{}, probe_in_{};
        std::array<mgl::BufferId, 2> particles_{};
        std::array<mgl::BufferId, 3> scratch_{};
        std::vector<mgl::BufferId> buffers_;
        std::array<mgl::VertexArrayId, kVaoCount> vao_{};
        std::array<mgl::TransformFeedbackId, 2> tf_{};
        mgl::TransformFeedbackId tf_misc_{};
        mgl::QueryId query_{};
        GLuint mesh_prims_ = 0, particle_prims_ = 0, pause_prims_ = 0, stream_prims_ = 0;

    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-transform-feedback",
        .title = "Transform feedback: capture, objects, pause/resume, queries",
        .min_es = {3, 0},
        .summary = "Records vertex shader outputs into buffers (INTERLEAVED and SEPARATE modes, buffer ranges, transform feedback objects, "
                   "pause/resume, primitive queries, RASTERIZER_DISCARD) and draws the captured data in a second pass; buffers and pixels are "
                   "compared with a CPU model and the defined errors are checked.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<TransformFeedbackDemo>(argc, argv, info);
}
