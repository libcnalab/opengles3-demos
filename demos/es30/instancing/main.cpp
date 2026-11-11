// es30-instancing
//
// Instanced drawing (ES 3.0 section 2.9.3), per-instance attributes (glVertexAttribDivisor), the
// built-in vertex inputs gl_VertexID / gl_InstanceID and the point size output, plus the indexed
// drawing features of the same section: glDrawRangeElements, PRIMITIVE_RESTART_FIXED_INDEX and
// MAX_ELEMENT_INDEX.
//
// The 256x256 surface is a 4x4 grid of 64x64 cells (row 3 is the top row, reading order is id order):
//
//   row 3:  [0] div 1 / div 1      [1] elements, color div 2    [2] color div 3        [3] gl_InstanceID only
//   row 2:  [4] offset div 0       [5] glDrawArrays (1 inst.)   [6] instanceCount 5    [7] points, gl_PointSize
//   row 1:  [8] strip, no marker   [9] USHORT 0xFFFF restart    [10] UBYTE 0xFF restart [11] UBYTE 0xFF, restart off
//   row 0:  [12] UINT restart      [13] instanced + restart     [14] DrawRangeElements  [15] DrawRangeElements subset
//
// Every cell is compared pixel by pixel with a CPU model: a list of triangles (or points) built from
// the same source data using the spec's rules (element = floor(instance / divisor), strip assembly
// with restart). Instance centers, the gaps between instances and the gaps of the restarted strips
// are probed one by one as well.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
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
    // Cells
    // -----------------------------------------------------------------------------------------
    enum CellId : int
    {
        kDiv1, kDiv2Elements, kDiv3, kInstanceId,
        kDiv0, kNotInstanced, kCount5, kPoints,
        kRestartNone, kRestartUs, kRestartUb, kRestartOffUb,
        kRestartUi, kRestartInstanced, kRange, kRangeSubset,
        kCellCount
    };
    constexpr std::array<const char*, kCellCount> kCellName = {
        "glDrawArraysInstanced, offset divisor 1, color divisor 1",
        "glDrawElementsInstanced (USHORT elements), offset divisor 1, color divisor 2",
        "glDrawArraysInstanced, color divisor 3 (groups of 3 do not align with the rows of 4)",
        "glDrawArraysInstanced with no attributes: gl_VertexID + gl_InstanceID",
        "offset attribute with divisor 0 in an instanced draw (advances per vertex)",
        "glDrawArrays (not instanced): divisor-1 VAO shows instance 0 only; the gl_InstanceID shader sees 0 (small square)",
        "glDrawArraysInstanced with instanceCount 5",
        "POINTS: gl_InstanceID selects position, gl_PointSize and color",
        "TRIANGLE_STRIP, USHORT, restart enabled but no 0xFFFF in the indices: no gap",
        "TRIANGLE_STRIP, USHORT, 0xFFFF restart index: gap",
        "TRIANGLE_STRIP, UBYTE, 0xFF restart index: gap",
        "TRIANGLE_STRIP, UBYTE, 0xFF with restart DISABLED: vertex 255 is an ordinary vertex",
        "TRIANGLE_STRIP, UINT, 0xFFFFFFFF restart index: gap",
        "glDrawElementsInstanced x3 with restart: every instance has the gap",
        "glDrawRangeElements(0, 11) with a restart index in the list",
        "glDrawRangeElements(6, 11): second strip only (offset into the element buffer)"};

    struct Rect
    {
        int x, y, w, h;
    };
    Rect cell_rect(int w, int h, int id)
    {
        const int cw = w / kGrid, ch = h / kGrid;
        return {(id % kGrid) * cw, (3 - id / kGrid) * ch, cw, ch};
    }

    // -----------------------------------------------------------------------------------------
    // Source data: the 4x4 instance grid
    // -----------------------------------------------------------------------------------------
    constexpr int kInstances = 16;
    constexpr double kSize = 0.36;  // side of an instance quad (NDC); slots are 0.5 apart
    std::array<double, 2> slot_center(int i) { return {-0.75 + 0.5 * (i % 4), -0.75 + 0.5 * (i / 4)}; }

    constexpr std::array<float, 8> kCorner = {0, 0, 1, 0, 0, 1, 1, 1};  // TRIANGLE_STRIP order
    constexpr std::array<std::uint16_t, 6> kQuadIndices = {0, 1, 2, 2, 1, 3};

    std::vector<float> make_offsets()  // lower-left corner of slot i
    {
        std::vector<float> v;
        for (int i = 0; i < kInstances; ++i)
        {
            v.push_back(static_cast<float>(slot_center(i)[0] - kSize / 2));
            v.push_back(static_cast<float>(slot_center(i)[1] - kSize / 2));
        }
        return v;
    }
    std::vector<float> make_colors()  // 16 different colors; neighbours differ strongly
    {
        std::vector<float> v;
        for (int i = 0; i < kInstances; ++i)
        {
            v.push_back(0.25f + 0.75f * static_cast<float>((i * 7) % 16) / 15.0f);
            v.push_back(0.25f + 0.75f * static_cast<float>((i * 3 + 5) % 16) / 15.0f);
            v.push_back(0.25f + 0.75f * static_cast<float>((i * 11 + 2) % 16) / 15.0f);
        }
        return v;
    }
    const std::vector<float> kOffsets = make_offsets();
    const std::vector<float> kColors = make_colors();

    // -----------------------------------------------------------------------------------------
    // Source data: the strip ribbons (two ribbons of 3 columns each, separated by a gap)
    // -----------------------------------------------------------------------------------------
    struct RibbonVertex
    {
        float x, y;
        std::uint8_t r, g, b, pad;
    };  // 12 bytes
    constexpr std::array<double, 6> kRibbonX = {-0.85, -0.60, -0.35, 0.35, 0.60, 0.85};
    constexpr std::array<std::array<int, 3>, 6> kRibbonBase = {{{255, 0, 0}, {255, 128, 0}, {255, 255, 0}, {0, 255, 128}, {0, 160, 255}, {128, 0, 255}}};
    constexpr std::size_t kSpike = 255;  // the vertex that index 0xFF refers to when restart is disabled
    constexpr std::uint32_t kMarker = 0xFFFFFFFFu;  // placeholder in the index templates

    std::vector<RibbonVertex> make_ribbon()
    {
        std::vector<RibbonVertex> v(256, RibbonVertex{});
        for (std::size_t k = 0; k < 6; ++k)
        {
            const auto& c = kRibbonBase[k];
            const auto lighten = [](int ch) { return static_cast<std::uint8_t>(ch + (255 - ch) / 2); };
            v[2 * k] = {static_cast<float>(kRibbonX[k]), -0.4f, static_cast<std::uint8_t>(c[0]), static_cast<std::uint8_t>(c[1]), static_cast<std::uint8_t>(c[2]), 0};
            v[2 * k + 1] = {static_cast<float>(kRibbonX[k]), 0.4f, lighten(c[0]), lighten(c[1]), lighten(c[2]), 0};
        }
        v[kSpike] = {0.0f, 0.95f, 255, 255, 255, 0};
        return v;
    }
    const std::vector<RibbonVertex> kRibbon = make_ribbon();

    // Indices 0..5 (first ribbon), restart marker, 6..11 (second ribbon). `marker` is the maximum value of the index type.
    template <typename T>
    std::vector<T> marked_indices(T marker)
    {
        return {0, 1, 2, 3, 4, 5, marker, 6, 7, 8, 9, 10, 11};
    }
    template <typename T>
    std::vector<T> plain_indices()
    {
        return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    }

    // -----------------------------------------------------------------------------------------
    // CPU model: triangles in NDC with Gouraud colors, drawn in order (later primitives overwrite)
    // -----------------------------------------------------------------------------------------
    using Rgba = std::array<double, 4>;
    struct Vtx
    {
        double x, y;
        Rgba c;
    };
    using Tri = std::array<Vtx, 3>;
    enum class Cover { Outside, Inside, Unsure };
    struct Sample
    {
        Cover cover = Cover::Outside;
        Rgba color{};
    };

    // Within `margin` pixels of an edge the answer depends on the rasterizer's tie-breaking rules: Unsure (never compared).
    Sample sample_triangle(const Tri& t, double px, double py, double cw, double ch, double margin)
    {
        std::array<double, 3> X{}, Y{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            X[i] = (t[i].x + 1.0) * 0.5 * cw;
            Y[i] = (t[i].y + 1.0) * 0.5 * ch;
        }
        auto edge = [](double ax, double ay, double bx, double by, double x, double y) { return (bx - ax) * (y - ay) - (by - ay) * (x - ax); };
        const double area = edge(X[0], Y[0], X[1], Y[1], X[2], Y[2]);
        if (std::abs(area) < 1e-9) return {};  // degenerate: no fragments
        const double sign = area > 0 ? 1.0 : -1.0;
        double min_distance = 1e30;
        std::array<double, 3> w{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            const std::size_t j = (i + 1) % 3, k = (i + 2) % 3;
            const double e = edge(X[j], Y[j], X[k], Y[k], px, py);
            min_distance = std::min(min_distance, sign * e / std::hypot(X[k] - X[j], Y[k] - Y[j]));
            w[i] = e / area;
        }
        if (min_distance < -margin) return {Cover::Outside, {}};
        if (min_distance <= margin) return {Cover::Unsure, {}};
        Rgba c{};
        for (std::size_t k = 0; k < 4; ++k) c[k] = w[0] * t[0].c[k] + w[1] * t[1].c[k] + w[2] * t[2].c[k];
        return {Cover::Inside, c};
    }

    // The last definitely-covering triangle wins; a pixel that a later, uncertain triangle might also touch is Unsure.
    Sample sample_scene(const std::vector<Tri>& tris, double px, double py, double cw, double ch)
    {
        Sample result;
        bool unsure_after = false;
        for (const Tri& t : tris)
        {
            const Sample s = sample_triangle(t, px, py, cw, ch, 1.0);
            if (s.cover == Cover::Inside) { result = s; unsure_after = false; }
            else if (s.cover == Cover::Unsure) unsure_after = true;
        }
        return unsure_after ? Sample{Cover::Unsure, {}} : result;
    }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }
    Rgba8 want8(const Sample& s)
    {
        Rgba8 w{};
        for (std::size_t k = 0; k < 4; ++k) w[k] = s.cover == Cover::Inside ? to8(s.color[k]) : kClear8[k];
        return w;
    }

    struct Scan
    {
        int inside = 0, outside = 0, unsure = 0, bad = 0, max_dev = 0;
        std::string first_bad;
    };

    Scan scan_pixels(const std::vector<std::uint8_t>& pixels, const std::vector<Tri>& scene, int cw, int ch, int tolerance)
    {
        Scan r;
        for (int y = 0; y < ch; ++y)
            for (int x = 0; x < cw; ++x)
            {
                const Sample s = sample_scene(scene, x + 0.5, y + 0.5, cw, ch);
                if (s.cover == Cover::Unsure) { ++r.unsure; continue; }
                const Rgba8 want = want8(s);
                ++(s.cover == Cover::Inside ? r.inside : r.outside);
                const std::uint8_t* got = &pixels[static_cast<std::size_t>((y * cw + x) * 4)];
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int(got[k]) - int(want[k]));
                    r.max_dev = std::max(r.max_dev, dev);
                    ok = ok && dev <= tolerance;
                }
                if (!ok && r.bad++ == 0)
                    r.first_bad = std::format("first mismatch at cell pixel ({},{}): got {} want {}", x, y,
                                              demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return r;
    }

    // Instanced quads: element index of an attribute = floor(instance / divisor), or the vertex index for divisor 0
    // (DrawArrays with first = 0, so the vertex index is the position in the quad).
    std::vector<Tri> model_instanced(int instance_count, unsigned offset_divisor, unsigned color_divisor)
    {
        std::vector<Tri> out;
        for (int i = 0; i < instance_count; ++i)
        {
            std::array<Vtx, 4> v{};
            for (std::size_t k = 0; k < 4; ++k)
            {
                const std::size_t oi = offset_divisor ? static_cast<std::size_t>(i) / offset_divisor : k;
                const std::size_t ci = color_divisor ? static_cast<std::size_t>(i) / color_divisor : k;
                v[k] = {double(kOffsets[2 * oi]) + double(kCorner[2 * k]) * kSize, double(kOffsets[2 * oi + 1]) + double(kCorner[2 * k + 1]) * kSize,
                        {double(kColors[3 * ci]), double(kColors[3 * ci + 1]), double(kColors[3 * ci + 2]), 1.0}};
            }
            out.push_back({v[0], v[1], v[2]});  // TRIANGLE_STRIP: (0,1,2), (1,2,3)
            out.push_back({v[1], v[2], v[3]});
        }
        return out;
    }

    // The gl_InstanceID shader: slot from instance % 4 and instance / 4, size 0.2 + 0.01 * instance, color from the id bits.
    std::vector<Tri> model_instance_id(int count, double origin_x = -0.75)
    {
        std::vector<Tri> out;
        for (int i = 0; i < count; ++i)
        {
            const std::array<double, 2> c = {origin_x + 0.5 * (i % 4), -0.75 + 0.5 * (i / 4)};
            const double size = 0.2 + 0.01 * i;
            const Rgba color = {double(i & 3) / 3.0, double((i >> 2) & 3) / 3.0, 0.5, 1.0};
            std::array<Vtx, 4> v{};
            for (std::size_t k = 0; k < 4; ++k) v[k] = {c[0] + (double(kCorner[2 * k]) - 0.5) * size, c[1] + (double(kCorner[2 * k + 1]) - 0.5) * size, color};
            out.push_back({v[0], v[1], v[2]});
            out.push_back({v[1], v[2], v[3]});
        }
        return out;
    }

    // Strip assembly with primitive restart (ES 3.0 section 2.9.1): an index equal to `restart` ends the current strip.
    std::vector<std::array<std::uint32_t, 3>> strip_triangles(const std::vector<std::uint32_t>& idx, std::uint32_t restart, bool restart_on)
    {
        std::vector<std::array<std::uint32_t, 3>> out;
        std::vector<std::uint32_t> run;
        auto flush = [&] {
            for (std::size_t i = 0; i + 2 < run.size(); ++i) out.push_back({run[i], run[i + 1], run[i + 2]});
            run.clear();
        };
        for (const std::uint32_t v : idx)
        {
            if (restart_on && v == restart) flush();
            else run.push_back(v);
        }
        flush();
        return out;
    }

    struct StripParams  // the strip.vert uniforms
    {
        double sx = 1, sy = 1, ox = 0, oy = 0, step_x = 0, step_y = 0, fade = 0;
        int instances = 1;
    };
    std::vector<Tri> model_strip(const std::vector<std::uint32_t>& idx, std::uint32_t restart, bool restart_on, const StripParams& p = {})
    {
        std::vector<Tri> out;
        const auto tris = strip_triangles(idx, restart, restart_on);
        for (int inst = 0; inst < p.instances; ++inst)
            for (const auto& t : tris)
            {
                Tri tri{};
                for (std::size_t k = 0; k < 3; ++k)
                {
                    const RibbonVertex& v = kRibbon.at(t[k]);
                    const double f = 1.0 - p.fade * inst;
                    tri[k] = {p.ox + double(v.x) * p.sx + p.step_x * inst, p.oy + double(v.y) * p.sy + p.step_y * inst,
                              {v.r / 255.0 * f, v.g / 255.0 * f, v.b / 255.0 * f, 1.0}};
                }
                out.push_back(tri);
            }
        return out;
    }

    // Index lists as the model sees them (restart marker = maximum of the type).
    std::vector<std::uint32_t> model_indices(bool marked, std::uint32_t marker)
    {
        std::vector<std::uint32_t> v;
        for (std::uint32_t i = 0; i < 12; ++i)
        {
            if (marked && i == 6) v.push_back(marker);
            v.push_back(i);
        }
        return v;
    }

    // Points: size-clamped squares at window coordinates; a pixel is covered when its center is inside the square.
    struct PointModel
    {
        std::array<double, 4> size{};
        std::array<Rgba, 4> color{};
        double range_max = 1;
        double center_x(int i, int cw) const { return cw * (0.125 + 0.25 * i) + 0.5; }
        double center_y(int ch) const { return ch * 0.5 + 0.5; }
    };
    PointModel make_point_model(double range_max)
    {
        PointModel m;
        m.range_max = range_max;
        for (int i = 0; i < 4; ++i)
        {
            m.size[static_cast<std::size_t>(i)] = 3.0 + 2.0 * i;
            m.color[static_cast<std::size_t>(i)] = {double(i & 1), double((i >> 1) & 1), 1.0 - 0.25 * i, 1.0};
        }
        return m;
    }
    Scan scan_points(const std::vector<std::uint8_t>& pixels, const PointModel& m, int cw, int ch)
    {
        Scan r;
        for (int y = 0; y < ch; ++y)
            for (int x = 0; x < cw; ++x)
            {
                Sample s;
                bool unsure = false;
                for (int i = 0; i < 4; ++i)
                {
                    const double size = m.size[static_cast<std::size_t>(i)];
                    const double dx = std::abs(x + 0.5 - m.center_x(i, cw)), dy = std::abs(y + 0.5 - m.center_y(ch));
                    if (size > m.range_max)  // clamped to the supported range: the exact size is not modelled
                    {
                        if (dx <= size / 2 + 1 && dy <= size / 2 + 1) unsure = true;
                    }
                    else if (dx < size / 2 && dy < size / 2) s = {Cover::Inside, m.color[static_cast<std::size_t>(i)]};
                }
                if (unsure) { ++r.unsure; continue; }
                const Rgba8 want = want8(s);
                ++(s.cover == Cover::Inside ? r.inside : r.outside);
                const std::uint8_t* got = &pixels[static_cast<std::size_t>((y * cw + x) * 4)];
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int(got[k]) - int(want[k]));
                    r.max_dev = std::max(r.max_dev, dev);
                    ok = ok && dev <= 1;
                }
                if (!ok && r.bad++ == 0)
                    r.first_bad = std::format("first mismatch at cell pixel ({},{}): got {} want {}", x, y,
                                              demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return r;
    }

    void expect_scan(demo::Check& check, const Scan& s, int min_inside, int min_outside, const std::string& what)
    {
        const bool enough = s.inside >= min_inside && s.outside >= min_outside;
        check.expect(s.bad == 0 && enough,
                     std::format("{}: {} covered + {} background pixels match the model (max deviation {}/255, {} edge pixels skipped)", what,
                                 s.inside, s.outside, s.max_dev, s.unsure),
                     s.bad ? std::format("{} mismatches; {}", s.bad, s.first_bad)
                     : enough ? std::string{} : std::format("only {} covered / {} background pixels compared", s.inside, s.outside));
    }

    // Pixel under an NDC point of a cell, expected from the model (a probe on a model edge is a bug in the demo).
    void expect_model_pixel(demo::Check& check, const Rect& r, const std::vector<Tri>& scene, double nx, double ny, int tolerance, const std::string& what)
    {
        const int px = static_cast<int>(std::floor((nx + 1.0) * 0.5 * r.w)), py = static_cast<int>(std::floor((ny + 1.0) * 0.5 * r.h));
        const Sample s = sample_scene(scene, px + 0.5, py + 0.5, r.w, r.h);
        if (s.cover == Cover::Unsure)
        {
            check.expect(false, what, "probe lies within 1 px of a model edge (demo bug)");
            return;
        }
        check.expect_pixel(r.x + px, r.y + py, want8(s), tolerance, what);
    }

    class Instancing final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            prog_instanced_ = demo::build_program(env, "shaders/instanced.vert", "shaders/color.frag");
            prog_id_ = demo::build_program(env, "shaders/instance_id.vert", "shaders/color.frag");
            prog_points_ = demo::build_program(env, "shaders/points.vert", "shaders/color.frag");
            prog_strip_ = demo::build_program(env, "shaders/strip.vert", "shaders/color.frag");
            u_size_ = mgl::glGetUniformLocation(prog_instanced_, "u_size");
            u_columns_ = mgl::glGetUniformLocation(prog_id_, "u_columns");
            u_id_origin_ = mgl::glGetUniformLocation(prog_id_, "u_origin");
            u_id_step_ = mgl::glGetUniformLocation(prog_id_, "u_step");
            u_size0_ = mgl::glGetUniformLocation(prog_id_, "u_size0");
            u_dsize_ = mgl::glGetUniformLocation(prog_id_, "u_dsize");
            u_cell_ = mgl::glGetUniformLocation(prog_points_, "u_cell");
            u_scale_ = mgl::glGetUniformLocation(prog_strip_, "u_scale");
            u_origin_ = mgl::glGetUniformLocation(prog_strip_, "u_origin");
            u_step_ = mgl::glGetUniformLocation(prog_strip_, "u_step");
            u_fade_ = mgl::glGetUniformLocation(prog_strip_, "u_fade");

            // The initial value of the restart flag (Table 6.3: FALSE) is recorded before anything touches it.
            restart_initially_enabled_ = mgl::glIsEnabled(mgl::Capability::PrimitiveRestartFixedIndex);

            // --- the shared instance buffers ---------------------------------------------------------
            const mgl::BufferId corner = make_array_buffer(std::span<const float>(kCorner));
            const mgl::BufferId offsets = make_array_buffer(std::span<const float>(kOffsets));
            const mgl::BufferId colors = make_array_buffer(std::span<const float>(kColors));
            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());

            // One VAO per divisor configuration. Divisors are VAO state (Table 6.2).
            auto instanced_vao = [&](mgl::VertexArrayId vao, GLuint offset_divisor, GLuint color_divisor) {
                mgl::glBindVertexArray(vao);
                mgl::glBindBuffer(mgl::BufferTarget::Array, corner);
                mgl::glEnableVertexAttribArray(kLocCorner);
                mgl::glVertexAttribPointer(kLocCorner, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
                mgl::glBindBuffer(mgl::BufferTarget::Array, offsets);
                mgl::glEnableVertexAttribArray(kLocOffset);
                mgl::glVertexAttribPointer(kLocOffset, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
                mgl::glVertexAttribDivisor(kLocOffset, offset_divisor);
                mgl::glBindBuffer(mgl::BufferTarget::Array, colors);
                mgl::glEnableVertexAttribArray(kLocColor);
                mgl::glVertexAttribPointer(kLocColor, 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
                mgl::glVertexAttribDivisor(kLocColor, color_divisor);
            };
            instanced_vao(vao_[kVaoDiv11], 1, 1);
            instanced_vao(vao_[kVaoDiv12], 1, 2);
            ebo_quad_ = make_element_buffer(std::span<const std::uint16_t>(kQuadIndices));  // attached to the VAO bound above
            instanced_vao(vao_[kVaoDiv13], 1, 3);
            instanced_vao(vao_[kVaoDiv01], 0, 1);  // the offsets advance per VERTEX here

            // --- strip VAO: a 256-vertex ribbon buffer (vertex 255 exists so that UBYTE index 0xFF is a valid vertex) ----
            vbo_ribbon_ = make_array_buffer(std::span<const RibbonVertex>(kRibbon));
            mgl::glBindVertexArray(vao_[kVaoStrip]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_ribbon_);
            mgl::glEnableVertexAttribArray(kLocPos);
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, sizeof(RibbonVertex), demo::buffer_offset(offsetof(RibbonVertex, x)));
            mgl::glEnableVertexAttribArray(kLocStripColor);
            mgl::glVertexAttribPointer(kLocStripColor, 3, mgl::DataType::UnsignedByte, GL_TRUE, sizeof(RibbonVertex), demo::buffer_offset(offsetof(RibbonVertex, r)));
            ebo_plain_ = make_element_buffer(std::span<const std::uint16_t>(plain_indices<std::uint16_t>()));
            ebo_us_ = make_element_buffer(std::span<const std::uint16_t>(marked_indices<std::uint16_t>(0xFFFF)));
            ebo_ub_ = make_element_buffer(std::span<const std::uint8_t>(marked_indices<std::uint8_t>(0xFF)));
            ebo_ui_ = make_element_buffer(std::span<const std::uint32_t>(marked_indices<std::uint32_t>(0xFFFFFFFFu)));

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            for (int id = 0; id < kCellCount; ++id) draw_cell(env, static_cast<CellId>(id));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_cells(env, check);
            check_controls(env, check);
            check_gaps(env, check);
            check_special_probes(env, check);
            check_state(check);
            check_limits(check);
            check_large_range(env, check);
            render(env, 0);  // restore the capture image
            check_errors(env, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            mgl::glDeleteProgram(prog_instanced_);
            mgl::glDeleteProgram(prog_id_);
            mgl::glDeleteProgram(prog_points_);
            mgl::glDeleteProgram(prog_strip_);
        }

    private:
        static constexpr mgl::AttribLocation kLocCorner{0u}, kLocOffset{1u}, kLocColor{2u};  // instanced.vert
        static constexpr mgl::AttribLocation kLocPos{0u}, kLocStripColor{1u};                  // strip.vert
        enum VaoSlot : std::size_t { kVaoDiv11, kVaoDiv12, kVaoDiv13, kVaoDiv01, kVaoStrip, kVaoEmpty, kVaoCount };

        template <typename T>
        mgl::BufferId make_array_buffer(std::span<const T> data)
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::Array, b);
            mgl::glBufferData(mgl::BufferTarget::Array, data, mgl::BufferUsage::StaticDraw);
            buffers_.push_back(b);
            return b;
        }
        template <typename T>
        mgl::BufferId make_element_buffer(std::span<const T> data)  // binds to the currently bound VAO
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, b);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, data, mgl::BufferUsage::StaticDraw);
            buffers_.push_back(b);
            return b;
        }

        // ---------------------------------------------------------------------------------------------
        // Drawing
        // ---------------------------------------------------------------------------------------------
        void set_strip_uniforms(const StripParams& p) const
        {
            mgl::glUniform2f(u_scale_, static_cast<float>(p.sx), static_cast<float>(p.sy));
            mgl::glUniform2f(u_origin_, static_cast<float>(p.ox), static_cast<float>(p.oy));
            mgl::glUniform2f(u_step_, static_cast<float>(p.step_x), static_cast<float>(p.step_y));
            mgl::glUniform1f(u_fade_, static_cast<float>(p.fade));
        }

        void draw_instanced_cell(VaoSlot vao) const
        {
            mgl::glUseProgram(prog_instanced_);
            mgl::glUniform1f(u_size_, static_cast<float>(kSize));
            mgl::glBindVertexArray(vao_[vao]);
        }

        void use_instance_id_program(float origin_x) const
        {
            mgl::glUseProgram(prog_id_);
            mgl::glUniform1i(u_columns_, 4);
            mgl::glUniform2f(u_id_origin_, origin_x, -0.75f);
            mgl::glUniform1f(u_id_step_, 0.5f);
            mgl::glUniform1f(u_size0_, 0.2f);
            mgl::glUniform1f(u_dsize_, 0.01f);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);  // no attribute arrays at all
        }

        void draw_cell(const demo::Env& env, CellId id) const
        {
            const Rect r = cell_rect(env.width, env.height, id);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            switch (id)
            {
            case kDiv1:
                draw_instanced_cell(kVaoDiv11);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, kInstances);
                break;
            case kDiv2Elements:
                draw_instanced_cell(kVaoDiv12);
                mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, 6, mgl::IndexType::UnsignedShort, demo::buffer_offset(0), kInstances);
                break;
            case kDiv3:
                draw_instanced_cell(kVaoDiv13);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, kInstances);
                break;
            case kInstanceId:
                use_instance_id_program(-0.75f);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, kInstances);
                break;
            case kDiv0:
                draw_instanced_cell(kVaoDiv01);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, kInstances);
                break;
            case kNotInstanced:
                draw_instanced_cell(kVaoDiv11);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);  // non-instanced: only element 0 of the divisor-1 arrays
                use_instance_id_program(-0.25f);  // the shader's "slot 0" is the grid slot 1 position: beside the first quad
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);  // gl_InstanceID is 0 outside instanced draws (GLSL ES 3.00 section 7.1)
                break;
            case kCount5:
                draw_instanced_cell(kVaoDiv11);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, 5);
                break;
            case kPoints:
                mgl::glUseProgram(prog_points_);
                mgl::glUniform2f(u_cell_, static_cast<float>(r.w), static_cast<float>(r.h));
                mgl::glBindVertexArray(vao_[kVaoEmpty]);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::Points, 0, 1, 4);
                break;
            case kRestartNone: draw_strip(mgl::IndexType::UnsignedShort, ebo_plain_, 12, true, [&] {
                    mgl::glDrawElements(mgl::PrimitiveType::TriangleStrip, 12, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
                }); break;
            case kRestartUs: draw_strip(mgl::IndexType::UnsignedShort, ebo_us_, 13, true, [&] {
                    mgl::glDrawElements(mgl::PrimitiveType::TriangleStrip, 13, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
                }); break;
            case kRestartUb: draw_strip(mgl::IndexType::UnsignedByte, ebo_ub_, 13, true, [&] {
                    mgl::glDrawElements(mgl::PrimitiveType::TriangleStrip, 13, mgl::IndexType::UnsignedByte, demo::buffer_offset(0));
                }); break;
            case kRestartOffUb: draw_strip(mgl::IndexType::UnsignedByte, ebo_ub_, 13, false, [&] {
                    mgl::glDrawElements(mgl::PrimitiveType::TriangleStrip, 13, mgl::IndexType::UnsignedByte, demo::buffer_offset(0));
                }); break;
            case kRestartUi: draw_strip(mgl::IndexType::UnsignedInt, ebo_ui_, 13, true, [&] {
                    mgl::glDrawElements(mgl::PrimitiveType::TriangleStrip, 13, mgl::IndexType::UnsignedInt, demo::buffer_offset(0));
                }); break;
            case kRestartInstanced: draw_strip(mgl::IndexType::UnsignedShort, ebo_us_, 13, true, [&] {
                    set_strip_uniforms(kInstancedStrip);
                    mgl::glDrawElementsInstanced(mgl::PrimitiveType::TriangleStrip, 13, mgl::IndexType::UnsignedShort, demo::buffer_offset(0), 3);
                }); break;
            case kRange: draw_strip(mgl::IndexType::UnsignedShort, ebo_us_, 13, true, [&] {
                    mgl::glDrawRangeElements(mgl::PrimitiveType::TriangleStrip, 0, 11, 13, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
                }); break;
            case kRangeSubset: draw_strip(mgl::IndexType::UnsignedShort, ebo_us_, 6, false, [&] {
                    // indices 6..11 start at element 7 of the buffer: a byte offset of 14
                    mgl::glDrawRangeElements(mgl::PrimitiveType::TriangleStrip, 6, 11, 6, mgl::IndexType::UnsignedShort, demo::buffer_offset(14));
                }); break;
            default: break;
            }
        }

        // The strip cells share the VAO; only the bound ELEMENT_ARRAY_BUFFER and the restart flag change.
        template <typename Draw>
        void draw_strip(mgl::IndexType /*type*/, mgl::BufferId ebo, int /*count*/, bool restart, Draw&& draw) const
        {
            mgl::glUseProgram(prog_strip_);
            set_strip_uniforms({});
            mgl::glBindVertexArray(vao_[kVaoStrip]);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ebo);  // re-points the element buffer binding of THIS VAO
            if (restart) mgl::glEnable(mgl::Capability::PrimitiveRestartFixedIndex);
            else mgl::glDisable(mgl::Capability::PrimitiveRestartFixedIndex);
            draw();
            mgl::glDisable(mgl::Capability::PrimitiveRestartFixedIndex);
        }

        static constexpr StripParams kInstancedStrip = {1.0, 0.3, 0.0, -0.6, 0.0, 0.6, 0.3, 3};

        // ---------------------------------------------------------------------------------------------
        // Models
        // ---------------------------------------------------------------------------------------------
        static std::vector<Tri> scene_for(CellId id)
        {
            switch (id)
            {
            case kDiv1: return model_instanced(kInstances, 1, 1);
            case kDiv2Elements: return model_instanced(kInstances, 1, 2);
            case kDiv3: return model_instanced(kInstances, 1, 3);
            case kInstanceId: return model_instance_id(kInstances);
            case kDiv0: return model_instanced(kInstances, 0, 1);
            case kNotInstanced:
            {
                std::vector<Tri> v = model_instanced(1, 1, 1);  // instance 0 of the attribute shader ...
                const std::vector<Tri> small = model_instance_id(1, -0.25);  // ... then the gl_InstanceID shader (its origin uniform is slot 1), gl_InstanceID = 0
                v.insert(v.end(), small.begin(), small.end());
                return v;
            }
            case kCount5: return model_instanced(5, 1, 1);
            case kRestartNone: return model_strip(model_indices(false, 0), 0xFFFF, true);
            case kRestartUs: return model_strip(model_indices(true, 0xFFFF), 0xFFFF, true);
            case kRestartUb: return model_strip(model_indices(true, 0xFF), 0xFF, true);
            case kRestartOffUb: return model_strip(std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5, 0xFF, 6, 7, 8, 9, 10, 11}, 0xFF, false);
            case kRestartUi: return model_strip(model_indices(true, 0xFFFFFFFFu), 0xFFFFFFFFu, true);
            case kRestartInstanced: return model_strip(model_indices(true, 0xFFFF), 0xFFFF, true, kInstancedStrip);
            case kRange: return model_strip(model_indices(true, 0xFFFF), 0xFFFF, true);
            case kRangeSubset: return model_strip(std::vector<std::uint32_t>{6, 7, 8, 9, 10, 11}, 0xFFFF, false);
            default: return {};
            }
        }

        static std::vector<std::uint8_t> read_cell(const Rect& r) { return demo::read_rgba8(r.x, r.y, r.w, r.h); }

        // ---------------------------------------------------------------------------------------------
        // Validation: cells
        // ---------------------------------------------------------------------------------------------
        void check_cells(const demo::Env& env, demo::Check& check) const
        {
            for (int id = 0; id < kCellCount; ++id)
            {
                const Rect r = cell_rect(env.width, env.height, id);
                const std::string what = std::format("cell {}: {}", id, kCellName[static_cast<std::size_t>(id)]);
                if (id == kPoints)
                {
                    GLfloat range[2] = {1, 1};
                    mgl::glGetFloatv(mgl::GetParameter::AliasedPointSizeRange, range);
                    const PointModel m = make_point_model(double(range[1]));
                    int probed = 0;
                    for (const double s : m.size) probed += s <= m.range_max ? 1 : 0;
                    const Scan s = scan_points(read_cell(r), m, r.w, r.h);
                    check.expect(s.bad == 0 && probed >= 1 && s.inside >= 9,
                                 std::format("{}: {} covered + {} background pixels match the model ({} of 4 point sizes within ALIASED_POINT_SIZE_RANGE [{}, {}], {} pixels near clamped points skipped)",
                                             what, s.inside, s.outside, probed, range[0], range[1], s.unsure),
                                 s.bad ? s.first_bad : std::string{});
                    continue;
                }
                const bool gradient = id >= kRestartNone;
                const int min_inside = 50;  // every cell shows at least one drawn primitive; the model must not be vacuous
                expect_scan(check, scan_pixels(read_cell(r), scene_for(static_cast<CellId>(id)), r.w, r.h, gradient ? 3 : 1), min_inside, 800, what);
            }

            // One probe per instance slot: the instance center shows the model's color, and instances that
            // were not drawn leave the background.
            const std::array<std::pair<CellId, unsigned>, 6> grids = {{{kDiv1, 0xFFFFu}, {kDiv2Elements, 0xFFFFu}, {kDiv3, 0xFFFFu}, {kInstanceId, 0xFFFFu},
                                                                {kNotInstanced, 0x3u}, {kCount5, 0x1Fu}}};
            for (const auto& [id, mask] : grids)
            {
                const int present = std::popcount(mask);
                const Rect r = cell_rect(env.width, env.height, id);
                const std::vector<Tri> scene = scene_for(id);
                std::string wrong;
                for (int i = 0; i < kInstances; ++i)
                {
                    // 0.05 NDC down-left of the slot center: inside the quad, away from the diagonal shared by its two triangles
                    const auto c = slot_center(i);
                    const int px = static_cast<int>(std::floor((c[0] - 0.05 + 1.0) * 0.5 * r.w)), py = static_cast<int>(std::floor((c[1] - 0.05 + 1.0) * 0.5 * r.h));
                    const Sample s = sample_scene(scene, px + 0.5, py + 0.5, r.w, r.h);
                    const bool drawn = ((mask >> i) & 1u) != 0;
                    const Rgba8 got = demo::read_pixel(r.x + px, r.y + py);
                    bool ok = (s.cover == (drawn ? Cover::Inside : Cover::Outside));  // the model itself must agree with the intent
                    const Rgba8 want = want8(s);
                    for (std::size_t k = 0; k < 4; ++k) ok = ok && std::abs(int(got[k]) - int(want[k])) <= 1;
                    if (!ok) wrong += std::format("{}:{}vs{} ", i, demo::format_rgba(got), demo::format_rgba(want));
                }
                check.expect(wrong.empty(), std::format("cell {}: the {} drawn slots show the model colors and the other {} slots stay background", static_cast<int>(id), present, kInstances - present),
                             wrong);
            }

            // Primitive restart: probe both ribbons and the gap region of each strip cell.
            for (const CellId id : {kRestartNone, kRestartUs, kRestartUb, kRestartUi, kRange})
            {
                const Rect r = cell_rect(env.width, env.height, id);
                const std::vector<Tri> scene = scene_for(id);
                const std::string name = std::format("cell {}", int(id));
                // One probe in each of the four quads of the ribbons (off the quads' diagonals).
                for (const double x : {-0.80, -0.55, 0.40, 0.65})
                    expect_model_pixel(check, r, scene, x, 0.0, 3, std::format("{}: ribbon quad at x = {} is drawn", name, x));
                const bool restarts = id != kRestartNone;
                // (-0.2, -0.25) and (0.2, 0.25) lie in the two "glue" triangles that connect the ribbons when no restart happens.
                for (const auto& p : {std::array<double, 2>{-0.2, -0.25}, std::array<double, 2>{0.2, 0.25}})
                {
                    const Sample s = sample_scene(scene, std::floor((p[0] + 1) * 0.5 * r.w) + 0.5, std::floor((p[1] + 1) * 0.5 * r.h) + 0.5, r.w, r.h);
                    check.expect(s.cover == (restarts ? Cover::Outside : Cover::Inside), std::format("{}: model expects the gap point ({}, {}) to be {}", name, p[0], p[1], restarts ? "background" : "covered"));
                    expect_model_pixel(check, r, scene, p[0], p[1], 3,
                                       std::format("{}: gap point ({}, {}) shows {}", name, p[0], p[1], restarts ? "the background (the strip was restarted)" : "the connecting triangle (the strip continues)"));
                }
            }
        }

        // Cell 11 (restart disabled) must show the spike toward vertex 255; cell 15 draws only the second ribbon.
        void check_special_probes(const demo::Env& env, demo::Check& check) const
        {
            {
                const Rect r = cell_rect(env.width, env.height, kRestartOffUb);
                const std::vector<Tri> scene = scene_for(kRestartOffUb);
                expect_model_pixel(check, r, scene, 0.0, 0.3, 3, "cell 11: the middle of the gap, (0, 0.3), is covered by the spike triangle (5, 255, 6)");
            }
            {
                // The small square drawn by the gl_InstanceID shader sits on top of the big attribute-driven one: its color
                // (0, 0, 0.5) is the color the shader computes for gl_InstanceID = 0.
                const Rect r = cell_rect(env.width, env.height, kNotInstanced);
                const Sample s = sample_scene(scene_for(kNotInstanced), std::floor((-0.3 + 1.0) * 0.5 * r.w) + 0.5, std::floor((-0.8 + 1.0) * 0.5 * r.h) + 0.5, r.w, r.h);
                check.expect(s.cover == Cover::Inside, "cell 5: the model has the small square beside the first quad (sanity)");
                expect_model_pixel(check, r, scene_for(kNotInstanced), -0.3, -0.8, 1, "cell 5: non-instanced draw: gl_InstanceID = 0 gives the instance-0 color (0, 0, 128) at the shader's slot 0");
            }
            {
                // Cell 13: three instances of the restarted strip, stacked at y = -0.6, 0, 0.6. Each has two ribbons and an empty gap.
                const Rect r = cell_rect(env.width, env.height, kRestartInstanced);
                const std::vector<Tri> scene = scene_for(kRestartInstanced);
                for (int i = 0; i < 3; ++i)
                {
                    const double y = kInstancedStrip.oy + kInstancedStrip.step_y * i;
                    expect_model_pixel(check, r, scene, -0.80, y, 3, std::format("cell 13: instance {}: first ribbon is drawn (faded by {})", i, kInstancedStrip.fade * i));
                    expect_model_pixel(check, r, scene, 0.40, y, 3, std::format("cell 13: instance {}: second ribbon is drawn", i));
                    const Sample gap = sample_scene(scene, std::floor(0.5 * r.w) + 0.5, std::floor((y + 1.0) * 0.5 * r.h) + 0.5, r.w, r.h);
                    check.expect(gap.cover == Cover::Outside, std::format("cell 13: the model has an empty gap in instance {} (sanity)", i));
                    expect_model_pixel(check, r, scene, 0.0, y, 3, std::format("cell 13: instance {}: the gap between its ribbons is background", i));
                }
            }
            {
                const Rect r = cell_rect(env.width, env.height, kRangeSubset);
                const std::vector<Tri> scene = scene_for(kRangeSubset);
                for (const double x : {-0.80, -0.55})
                    expect_model_pixel(check, r, scene, x, 0.0, 3, std::format("cell 15: first ribbon (indices 0..5) at x = {} is NOT drawn", x));
                for (const double x : {0.40, 0.65})
                    expect_model_pixel(check, r, scene, x, 0.0, 3, std::format("cell 15: second ribbon (indices 6..11) at x = {} is drawn", x));
            }
        }

        // The comparison must be able to fail: pixels of one cell scanned against the model of another configuration.
        void check_controls(const demo::Env& env, demo::Check& check) const
        {
            auto bad_against = [&](CellId pixels_of, const std::vector<Tri>& model) {
                const Rect r = cell_rect(env.width, env.height, pixels_of);
                return scan_pixels(read_cell(r), model, r.w, r.h, 3).bad;
            };
            check.expect(bad_against(kDiv1, model_instanced(kInstances, 1, 2)) > 100, "negative control: divisor-1 pixels do not match the divisor-2 model");
            check.expect(bad_against(kDiv2Elements, model_instanced(kInstances, 1, 1)) > 100, "negative control: divisor-2 pixels do not match the divisor-1 model");
            check.expect(bad_against(kDiv1, model_instanced(15, 1, 1)) > 50, "negative control: 16 instances do not match a 15-instance model (the last slot differs)");
            check.expect(bad_against(kNotInstanced, model_instanced(kInstances, 1, 1)) > 500, "negative control: the glDrawArrays cell does not match a 16-instance model");
            check.expect(bad_against(kRestartUs, scene_for(kRestartNone)) > 100, "negative control: the restarted strip does not match the model without restart (gap vs connecting triangles)");
            check.expect(bad_against(kRestartNone, scene_for(kRestartUs)) > 100, "negative control: the continuous strip does not match the model with restart");
            check.expect(bad_against(kRestartOffUb, scene_for(kRestartUb)) > 100, "negative control: with restart disabled 0xFF is a vertex (spike), unlike the restarted strip");
        }

        // Counts pixels that differ from the clear color inside a window rectangle of a cell.
        static int covered_in(const Rect& r, int x0, int y0, int x1, int y1)
        {
            const std::vector<std::uint8_t> px = read_cell(r);
            int n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x)
                {
                    const std::uint8_t* p = &px[static_cast<std::size_t>((y * r.w + x) * 4)];
                    if (!(p[0] == kClear8[0] && p[1] == kClear8[1] && p[2] == kClear8[2])) ++n;
                }
            return n;
        }

        // Aggregate evidence about the gaps, independent of the triangle model: pixel counts of the restarted strips.
        void check_gaps(const demo::Env& env, demo::Check& check) const
        {
            auto whole = [&](CellId id) {
                const Rect r = cell_rect(env.width, env.height, id);
                return covered_in(r, 0, 0, r.w, r.h);
            };
            const int none = whole(kRestartNone), us = whole(kRestartUs), ub = whole(kRestartUb), ui = whole(kRestartUi), range = whole(kRange);
            check.expect(us < none && us == ub && us == ui && us == range,
                         "restarted strips (USHORT, UBYTE, UINT, glDrawRangeElements) cover identical pixel sets, fewer than the unbroken strip (the gap is empty)",
                         std::format("covered pixels: no marker {} / ushort {} / ubyte {} / uint {} / range {}", none, us, ub, ui, range));
            // Above the ribbons (y > 0.4 NDC, i.e. window rows 48..63) only the spike triangles of cell 11 can cover pixels.
            const Rect none_r = cell_rect(env.width, env.height, kRestartNone), off_r = cell_rect(env.width, env.height, kRestartOffUb);
            const int top_none = covered_in(none_r, 0, none_r.h * 3 / 4 + 2, none_r.w, none_r.h);
            const int top_off = covered_in(off_r, 0, off_r.h * 3 / 4 + 2, off_r.w, off_r.h);
            check.expect(top_none == 0 && top_off > 20, "restart disabled: index 0xFF is vertex 255 (0, 0.95): the spike covers pixels above the ribbons, which the other strips never touch",
                         std::format("pixels above the ribbons: continuous strip {} / restart disabled {}", top_none, top_off));
        }

        // ---------------------------------------------------------------------------------------------
        // Validation: state
        // ---------------------------------------------------------------------------------------------
        static int divisor_of(mgl::AttribLocation loc)
        {
            GLint v = -1;
            mgl::glGetVertexAttribiv(loc, mgl::VertexAttribParameter::ArrayDivisor, &v);
            return v;
        }

        void check_state(demo::Check& check) const
        {
            struct Expect
            {
                VaoSlot vao;
                int d0, d1, d2;
                const char* what;
            };
            for (const Expect& e : {Expect{kVaoDiv11, 0, 1, 1, "offset 1, color 1"}, Expect{kVaoDiv12, 0, 1, 2, "offset 1, color 2"},
                                    Expect{kVaoDiv13, 0, 1, 3, "offset 1, color 3"}, Expect{kVaoDiv01, 0, 0, 1, "offset 0, color 1"}})
            {
                mgl::glBindVertexArray(vao_[e.vao]);
                const int a = divisor_of(kLocCorner), b = divisor_of(kLocOffset), c = divisor_of(kLocColor);
                check.expect(a == e.d0 && b == e.d1 && c == e.d2,
                             std::format("VERTEX_ATTRIB_ARRAY_DIVISOR of attributes 0/1/2 = {}/{}/{} ({})", e.d0, e.d1, e.d2, e.what),
                             std::format("read {}/{}/{}", a, b, c));
            }

            // The divisor is part of the VAO: changing it in one VAO leaves the others alone.
            mgl::VertexArrayId scratch{};
            mgl::glGenVertexArrays(1, &scratch);
            mgl::glBindVertexArray(scratch);
            check.expect_eq(divisor_of(kLocOffset), 0, "a fresh VAO has divisor 0 on every attribute");
            mgl::glVertexAttribDivisor(kLocOffset, 7);
            check.expect_eq(divisor_of(kLocOffset), 7, "glVertexAttribDivisor(1, 7) reads back 7 in the bound VAO");
            mgl::glBindVertexArray(vao_[kVaoDiv11]);
            check.expect_eq(divisor_of(kLocOffset), 1, "... and VAO div 1/1 still has 1 (the divisor is VAO state)");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDeleteVertexArrays(1, &scratch);

            // The element array buffer binding is per VAO too: the strip VAO currently points at the last buffer bound by draw_cell.
            mgl::glBindVertexArray(vao_[kVaoDiv12]);
            check.expect_eq(demo::get_integer(mgl::GetParameter::ElementArrayBufferBinding), static_cast<int>(ebo_quad_.value), "VAO div 1/2 keeps its own quad element buffer");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            // PRIMITIVE_RESTART_FIXED_INDEX (ES 3.0 Table 6.3: initially FALSE, queried with IsEnabled).
            check.expect(!restart_initially_enabled_, "PRIMITIVE_RESTART_FIXED_INDEX is initially disabled (glIsEnabled)");
            check.expect(!mgl::glIsEnabled(mgl::Capability::PrimitiveRestartFixedIndex), "... and disabled again after the draws");
            mgl::glEnable(mgl::Capability::PrimitiveRestartFixedIndex);
            GLboolean flag = GL_FALSE;
            mgl::glGetBooleanv(mgl::GetParameter::PrimitiveRestartFixedIndex, &flag);
            check.expect(mgl::glIsEnabled(mgl::Capability::PrimitiveRestartFixedIndex) && flag == GL_TRUE, "glEnable(PRIMITIVE_RESTART_FIXED_INDEX): glIsEnabled and glGetBooleanv report TRUE");
            mgl::glBindVertexArray(vao_[kVaoStrip]);  // not VAO state: the flag survives a VAO switch
            check.expect(mgl::glIsEnabled(mgl::Capability::PrimitiveRestartFixedIndex), "the flag is not VAO state (still enabled with another VAO bound)");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDisable(mgl::Capability::PrimitiveRestartFixedIndex);
            check.expect(!mgl::glIsEnabled(mgl::Capability::PrimitiveRestartFixedIndex), "glDisable(PRIMITIVE_RESTART_FIXED_INDEX) clears it");
            check.expect_no_gl_error("no GL error from the state queries");
        }

        void check_limits(demo::Check& check) const
        {
            GLint64 max_index = 0;
            mgl::glGetInteger64v(mgl::GetParameter::MaxElementIndex, &max_index);
            check.expect(max_index >= (1LL << 24) - 1, "glGetInteger64v(MAX_ELEMENT_INDEX) >= 2^24 - 1 (ES 3.0 Table 6.28 minimum)", std::format("{}", max_index));
            GLint max_index32 = 0;
            mgl::glGetIntegerv(mgl::GetParameter::MaxElementIndex, &max_index32);
            check.expect_eq(static_cast<long long>(max_index32), std::min<long long>(max_index, std::numeric_limits<GLint>::max()),
                            "glGetIntegerv(MAX_ELEMENT_INDEX) is the 64-bit value clamped to INT_MAX (ES 3.0 section 6.1.2)");
            GLint max_vertices = -1, max_indices = -1;
            mgl::glGetIntegerv(mgl::GetParameter::MaxElementsVertices, &max_vertices);
            mgl::glGetIntegerv(mgl::GetParameter::MaxElementsIndices, &max_indices);
            check.expect(max_vertices >= 0 && max_indices >= 0, "MAX_ELEMENTS_VERTICES and MAX_ELEMENTS_INDICES are queryable (recommended maxima for glDrawRangeElements)",
                         std::format("vertices {} indices {}", max_vertices, max_indices));
            check.expect_no_gl_error("no GL error from the limit queries");
        }

        // MAX_ELEMENTS_VERTICES / MAX_ELEMENTS_INDICES are only *recommended* maxima (ES 3.0 section 2.9.3): a call that exceeds
        // both must still draw correctly. Redraw cell 15 (second ribbon only) from a buffer that is larger than both limits.
        void check_large_range(const demo::Env& env, demo::Check& check) const
        {
            GLint max_vertices = 0, max_indices = 0;
            mgl::glGetIntegerv(mgl::GetParameter::MaxElementsVertices, &max_vertices);
            mgl::glGetIntegerv(mgl::GetParameter::MaxElementsIndices, &max_indices);
            // USHORT indices address at most 65536 vertices; keep the test small. (Skipped when the implementation reports huge limits.)
            if (max_vertices + 4 > 60000 || max_indices > 1000000) return;

            const GLuint end = static_cast<GLuint>(max_vertices) + 3;  // end - start + 1 = MAX_ELEMENTS_VERTICES + 4
            std::vector<RibbonVertex> vertices(end + 1);                // the range really lies inside the buffer
            std::copy(kRibbon.begin(), kRibbon.begin() + 12, vertices.begin());
            std::vector<std::uint16_t> indices = {6, 7, 8, 9, 10, 11};
            indices.resize(static_cast<std::size_t>(max_indices) + 6, 11);  // count = MAX_ELEMENTS_INDICES + 6; repeating 11 adds only degenerate triangles

            mgl::VertexArrayId vao{};
            mgl::BufferId buffers[2] = {};
            mgl::glGenVertexArrays(1, &vao);
            mgl::glGenBuffers(2, buffers);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buffers[0]);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const RibbonVertex>(vertices), mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(kLocPos);
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, sizeof(RibbonVertex), demo::buffer_offset(offsetof(RibbonVertex, x)));
            mgl::glEnableVertexAttribArray(kLocStripColor);
            mgl::glVertexAttribPointer(kLocStripColor, 3, mgl::DataType::UnsignedByte, GL_TRUE, sizeof(RibbonVertex), demo::buffer_offset(offsetof(RibbonVertex, r)));
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buffers[1]);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(indices), mgl::BufferUsage::StaticDraw);

            const Rect r = cell_rect(env.width, env.height, kRangeSubset);
            mgl::glEnable(mgl::Capability::ScissorTest);  // clear only this cell so that the draw below is the only thing that can fill it
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUseProgram(prog_strip_);
            set_strip_uniforms({});
            mgl::glDrawRangeElements(mgl::PrimitiveType::TriangleStrip, 0, end, static_cast<GLsizei>(indices.size()), mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
            mgl::glDisable(mgl::Capability::ScissorTest);
            const Scan scan = scan_pixels(read_cell(r), scene_for(kRangeSubset), r.w, r.h, 3);
            expect_scan(check, scan, 50, 800,
                        std::format("glDrawRangeElements with end - start + 1 = {} > MAX_ELEMENTS_VERTICES ({}) and count = {} > MAX_ELEMENTS_INDICES ({}) still draws the ribbon",
                                    end + 1, max_vertices, indices.size(), max_indices));

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteVertexArrays(1, &vao);
            mgl::glDeleteBuffers(2, buffers);
        }

        void check_errors(const demo::Env& env, demo::Check& check) const
        {
            const int max_attribs = demo::get_integer(mgl::GetParameter::MaxVertexAttribs);
            check.expect_no_gl_error("error queue empty before the error cases");
            mgl::glUseProgram(prog_instanced_);
            mgl::glUniform1f(u_size_, static_cast<float>(kSize));
            mgl::glBindVertexArray(vao_[kVaoDiv12]);  // has an element array buffer
            const Rect r = cell_rect(env.width, env.height, kDiv2Elements);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            const std::vector<std::uint8_t> before = read_cell(r);

            mgl::glVertexAttribDivisor(mgl::AttribLocation{static_cast<GLuint>(max_attribs)}, 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribDivisor with index = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawArraysInstanced with a negative instanceCount -> INVALID_VALUE");
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, -4, 3);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawArraysInstanced with a negative count -> INVALID_VALUE");
            mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, 6, mgl::IndexType::UnsignedShort, demo::buffer_offset(0), -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawElementsInstanced with a negative instanceCount -> INVALID_VALUE");
            mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, 6, mgl::DataType::Float, demo::buffer_offset(0), 2);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glDrawElementsInstanced with type FLOAT -> INVALID_ENUM");
            mgl::glDrawRangeElements(mgl::PrimitiveType::Triangles, 5, 2, 6, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawRangeElements with end < start -> INVALID_VALUE");
            mgl::glDrawRangeElements(mgl::PrimitiveType::Triangles, 0, 3, 6, mgl::DataType::Int, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glDrawRangeElements with type INT -> INVALID_ENUM");

            // instanceCount 0 is legal and draws nothing (the loop of section 2.9.3 runs zero times).
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, 0);
            mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, 6, mgl::IndexType::UnsignedShort, demo::buffer_offset(0), 0);
            check.expect_no_gl_error("instanceCount 0 raises no error");
            check.expect(read_cell(r) == before, "... and neither draw changed a single pixel of the cell");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        mgl::ProgramId prog_instanced_{}, prog_id_{}, prog_points_{}, prog_strip_{};
        mgl::UniformLocation u_size_{}, u_columns_{}, u_id_origin_{}, u_id_step_{}, u_size0_{}, u_dsize_{}, u_cell_{}, u_scale_{}, u_origin_{}, u_step_{}, u_fade_{};
        std::array<mgl::VertexArrayId, kVaoCount> vao_{};
        std::vector<mgl::BufferId> buffers_;
        mgl::BufferId ebo_quad_{}, vbo_ribbon_{}, ebo_plain_{}, ebo_us_{}, ebo_ub_{}, ebo_ui_{};
        bool restart_initially_enabled_ = true;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-instancing",
        .title = "Instancing, divisors, primitive restart",
        .min_es = {3, 0},
        .summary = "glDrawArraysInstanced / glDrawElementsInstanced with per-instance attributes (glVertexAttribDivisor) and gl_InstanceID; "
                   "glDrawRangeElements and PRIMITIVE_RESTART_FIXED_INDEX. Every instance and strip is compared with a CPU model.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Instancing>(argc, argv, info);
}
