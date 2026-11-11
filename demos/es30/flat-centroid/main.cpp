// es30-flat-centroid
//
// Three rasterization-stage features that only show up at the level of single fragments and samples:
//
//   * flat shading and the provoking vertex (OpenGL ES 3.0 §2.17, Table 2.12: the LAST vertex of each
//     triangle for TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN), with integer and float flat varyings;
//   * centroid interpolation on a multisample renderbuffer (GLSL ES 3.00 §4.3.9, ES 3.0 §3.6.3);
//   * the multisample fragment operations SAMPLE_COVERAGE / glSampleCoverage and
//     SAMPLE_ALPHA_TO_COVERAGE (ES 3.0 §4.1.3), on 4x multisample renderbuffers and on a single-sample FBO.
//
// All multisampling happens in application-created multisample renderbuffers that are resolved with
// glBlitFramebuffer. The default framebuffer stays single-sampled: on the surfaceless EGL platform the
// llvmpipe and zink pbuffers of a multisample EGLConfig come up as 0 x 0, unusable surfaces.
//
// Picture (a 2 x 2 grid of panels that always fills the surface):
//   top-left      a low-poly dusk landscape: a TRIANGLE_STRIP mountain ridge, a TRIANGLE_FAN sun and two
//                 kites made of independent TRIANGLES; every facet takes palette index and shade from
//                 its provoking (last) vertex
//   top-right     coverage bands for value 0, 1/8, ..., 1, rendered into a 4x multisample renderbuffer,
//                 resolved and magnified: SAMPLE_COVERAGE, the same inverted, both drawn on top of each
//                 other (no gaps), and SAMPLE_ALPHA_TO_COVERAGE with alpha = value
//   bottom-left   a rotated texture-atlas tile drawn into a 4x multisample renderbuffer WITHOUT centroid,
//                 resolved and magnified: magenta neighbour cells bleed in along the edges
//   bottom-right  the same tile with `centroid in`: no bleed
//
// Validation:
//   * flat: the scene is rendered into an RGBA32UI attachment that receives gl_VertexID, the flat uint,
//     the flat float's bits and the smooth float's bits. Every triangle of every topology is probed
//     and must hold the values of its provoking vertex exactly; the smooth copy is the negative control
//     (it must equal the CPU barycentric interpolation, which differs from the provoking value). The
//     displayed panel is probed against palette * shade as well.
//   * centroid: ANY_SAMPLES_PASSED occlusion queries with a fragment shader that discards every fragment
//     NOT having the property under test, so no conclusion depends on the implementation-dependent
//     resolve. With centroid no fragment may be evaluated outside the primitive (edge distance) or
//     sample a neighbouring atlas cell. A positive control proves that the query can return TRUE, and
//     on a single-sample FBO the centroid value must equal the pixel-centre value.
//   * coverage: value 0 writes no sample, value 1 every sample, invert flips both, and (v, FALSE)
//     followed by (v, TRUE) writes every sample exactly once (stencil INCR, then a query for stencil != 1);
//     drawing (0.5, FALSE) twice is the negative control. Alpha-to-coverage endpoints and its AND with
//     SAMPLE_COVERAGE. State queries (initial values, clamping, invert, enables). With SAMPLE_BUFFERS = 0
//     none of it has any effect. Every panel is probed against the read-back resolved texels it magnifies.
//     The resolved values themselves (endpoint bands, coverage curves) are only printed: ES 3.0 4.3.3
//     makes the resolve of normalized formats implementation-dependent, even for pixels whose samples
//     are all equal. Defined errors: blitting into a multisample FBO and ReadPixels from a multisample
//     FBO generate INVALID_OPERATION.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Low-poly scene (NDC of the panel viewport)
    // ---------------------------------------------------------------------------------------------
    struct FlatVertex
    {
        float x, y;
        std::uint32_t palette;
        float shade;
    };

    struct Topology
    {
        const char* name;
        mgl::PrimitiveType mode;
        int first;
        int count;
    };

    struct Tri
    {
        std::array<int, 3> v;
        int provoking;
    };

    constexpr std::array<std::array<float, 3>, 8> kPalette = {{
        {0.33f, 0.38f, 0.58f},  // 0 slate
        {0.50f, 0.42f, 0.66f},  // 1 violet grey
        {0.20f, 0.52f, 0.58f},  // 2 teal
        {1.00f, 0.95f, 0.80f},  // 3 warm white
        {1.00f, 0.80f, 0.25f},  // 4 yellow
        {1.00f, 0.52f, 0.20f},  // 5 orange
        {0.35f, 0.85f, 0.95f},  // 6 cyan
        {0.95f, 0.45f, 0.70f},  // 7 pink
    }};

    std::vector<FlatVertex> build_scene(std::array<Topology, 3>& topologies)
    {
        std::vector<FlatVertex> v;

        // TRIANGLE_STRIP mountain ridge: b0 t0 b1 t1 ... t4 b5 (11 vertices, 9 triangles). The palette
        // cycles through three rock colours, so any three consecutive vertices differ.
        const std::array<float, 5> peaks = {0.05f, 0.30f, -0.05f, 0.26f, 0.12f};
        const std::array<float, 11> strip_shade = {0.80f, 0.95f, 0.70f, 1.00f, 0.75f, 0.90f,
                                                   0.65f, 0.98f, 0.72f, 0.88f, 0.78f};
        for (int k = 0; k < 11; ++k)
        {
            const bool top = (k % 2) == 1;
            const int i = k / 2;
            const float x = top ? -0.768f + 0.384f * static_cast<float>(i) : -0.96f + 0.384f * static_cast<float>(i);
            const float y = top ? peaks[static_cast<std::size_t>(i)] : -0.92f;
            v.push_back({x, y, static_cast<std::uint32_t>(k % 3), strip_shade[static_cast<std::size_t>(k)]});
        }
        topologies[0] = {"TRIANGLE_STRIP", mgl::PrimitiveType::TriangleStrip, 0, 11};

        // TRIANGLE_FAN sun: centre + 11 ring vertices (the last repeats the first position) = 10 triangles.
        const int fan_first = static_cast<int>(v.size());
        v.push_back({0.55f, 0.62f, 3u, 1.0f});
        for (int k = 0; k <= 10; ++k)
        {
            const double a = (90.0 + 36.0 * (k % 10)) * 3.14159265358979323846 / 180.0;
            const float x = 0.55f + 0.27f * static_cast<float>(std::cos(a));
            const float y = 0.62f + 0.27f * static_cast<float>(std::sin(a));
            v.push_back({x, y, (k % 2) == 0 ? 4u : 5u, (k % 2) == 0 ? 0.96f : 0.84f - 0.01f * static_cast<float>(k)});
        }
        topologies[1] = {"TRIANGLE_FAN", mgl::PrimitiveType::TriangleFan, fan_first, 12};

        // Independent TRIANGLES: two kites, each an upper and a lower triangle with their own vertices.
        const int tri_first = static_cast<int>(v.size());
        struct Kite { float cx, cy, hw, hh; };
        const Kite kites[2] = {{-0.62f, 0.62f, 0.18f, 0.30f}, {-0.16f, 0.66f, 0.13f, 0.24f}};
        // palette/shade per vertex: L, R, T for the upper triangle, R, L, B for the lower one (provoking last).
        const std::uint32_t pal[2][6] = {{7u, 2u, 6u, 6u, 2u, 7u}, {1u, 7u, 4u, 4u, 1u, 5u}};
        const float shd[2][6] = {{0.70f, 0.60f, 1.00f, 0.90f, 0.55f, 0.80f}, {0.65f, 0.75f, 0.95f, 0.85f, 0.60f, 0.92f}};
        for (int k = 0; k < 2; ++k)
        {
            const Kite& q = kites[k];
            const float L[2] = {q.cx - q.hw, q.cy}, R[2] = {q.cx + q.hw, q.cy};
            const float T[2] = {q.cx, q.cy + q.hh}, B[2] = {q.cx, q.cy - q.hh};
            const float* pos[6] = {L, R, T, R, L, B};
            for (int j = 0; j < 6; ++j) v.push_back({pos[j][0], pos[j][1], pal[k][j], shd[k][j]});
        }
        topologies[2] = {"TRIANGLES", mgl::PrimitiveType::Triangles, tri_first, 12};
        return v;
    }

    // ES 3.0 Table 2.12, vertices numbered from 0 here: independent triangle j -> 3j+2,
    // strip triangle j -> j+2, fan triangle j -> j+2 (fan triangles are {0, j+1, j+2}).
    std::vector<Tri> triangles_of(const Topology& t)
    {
        std::vector<Tri> out;
        if (t.mode == mgl::PrimitiveType::Triangles)
        {
            for (int j = 0; j < t.count / 3; ++j)
                out.push_back({{t.first + 3 * j, t.first + 3 * j + 1, t.first + 3 * j + 2}, t.first + 3 * j + 2});
        }
        else if (t.mode == mgl::PrimitiveType::TriangleStrip)
        {
            for (int j = 0; j + 2 < t.count; ++j)
                out.push_back({{t.first + j, t.first + j + 1, t.first + j + 2}, t.first + j + 2});
        }
        else
        {
            for (int j = 0; j + 2 < t.count; ++j)
                out.push_back({{t.first, t.first + j + 1, t.first + j + 2}, t.first + j + 2});
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // Small geometry helpers (window coordinates)
    // ---------------------------------------------------------------------------------------------
    struct Vec2
    {
        double x, y;
    };

    struct Rect
    {
        int x, y, w, h;
    };

    double cross(Vec2 a, Vec2 b, Vec2 c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }

    Vec2 to_window(const FlatVertex& v, Rect vp)
    {
        return {vp.x + (static_cast<double>(v.x) + 1.0) * 0.5 * vp.w, vp.y + (static_cast<double>(v.y) + 1.0) * 0.5 * vp.h};
    }

    /// A probe pixel inside a triangle: barycentric weights at its centre and its distance to the nearest edge.
    struct Probe
    {
        int px, py;
        std::array<double, 3> w;
        double edge_distance;
    };

    Probe make_probe(const std::array<Vec2, 3>& t, Vec2 p)
    {
        Probe r{static_cast<int>(std::floor(p.x)), static_cast<int>(std::floor(p.y)), {}, 0.0};
        const Vec2 c{r.px + 0.5, r.py + 0.5};
        const double area2 = cross(t[0], t[1], t[2]);
        r.w = {cross(t[1], t[2], c) / area2, cross(t[2], t[0], c) / area2, cross(t[0], t[1], c) / area2};
        const double len[3] = {std::hypot(t[1].x - t[2].x, t[1].y - t[2].y), std::hypot(t[2].x - t[0].x, t[2].y - t[0].y),
                               std::hypot(t[0].x - t[1].x, t[0].y - t[1].y)};
        r.edge_distance = 1e9;
        for (int k = 0; k < 3; ++k) r.edge_distance = std::min(r.edge_distance, r.w[static_cast<std::size_t>(k)] * std::abs(area2) / len[k]);
        return r;
    }

    /// Centroid of the triangle and the points halfway between the centroid and each vertex.
    std::vector<Probe> probes_for(const std::array<Vec2, 3>& t, double min_edge_distance)
    {
        const Vec2 g{(t[0].x + t[1].x + t[2].x) / 3.0, (t[0].y + t[1].y + t[2].y) / 3.0};
        std::vector<Probe> out;
        const Vec2 candidates[4] = {g, {(g.x + t[0].x) / 2, (g.y + t[0].y) / 2}, {(g.x + t[1].x) / 2, (g.y + t[1].y) / 2},
                                    {(g.x + t[2].x) / 2, (g.y + t[2].y) / 2}};
        for (const Vec2& c : candidates)
        {
            const Probe p = make_probe(t, c);
            if (p.edge_distance >= min_edge_distance) out.push_back(p);
        }
        return out;
    }

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

    bool near_rgb(const std::uint8_t* px, std::array<std::uint8_t, 3> e, int tol)
    {
        for (int c = 0; c < 3; ++c)
            if (std::abs(static_cast<int>(px[c]) - static_cast<int>(e[static_cast<std::size_t>(c)])) > tol) return false;
        return true;
    }

    // ---------------------------------------------------------------------------------------------
    // Offscreen targets and the centroid tile
    // ---------------------------------------------------------------------------------------------
    constexpr int kView = 24;       // one centroid comparison view (the tile nearly fills it)
    constexpr int kOffW = 2 * kView; // multisample / single-sample scratch targets: two views side by side
    constexpr int kOffH = kView;
    constexpr int kIdSize = 128;    // RGBA32UI target of the flat-shading test
    constexpr int kAtlas = 48;      // 3 x 3 cells of 16 x 16 texels
    constexpr int kCell = 16;
    constexpr int kCenterCell = 4;  // the tile shows the centre cell; its neighbours are magenta
    constexpr std::uint8_t cell_tag(int cell) { return static_cast<std::uint8_t>(20 + 25 * cell); }

    // Edge-distance tolerance of the "evaluated outside the primitive" test, in pixels. ES 3.0 Table 6.28
    // guarantees SUBPIXEL_BITS >= 4, i.e. vertex positions may be snapped to a 1/16 pixel grid; a sample that
    // the rasterizer counts as inside may thus lie up to about 1/32 px outside the unsnapped edge line that
    // the interpolated distance refers to. 1/16 px is twice that and still well below the distance (up to
    // about half a pixel) by which the centre of a partially covered pixel lies outside the primitive.
    // The atlas texture coordinates are inset by 1/8 texel at about 0.92 texel per pixel, so a value
    // evaluated within 1/16 px of the tile (<= 0.06 texel) never reaches a neighbouring cell.
    constexpr float kEdgeEpsilon = 1.0f / 16.0f;

    struct TileVertex
    {
        float x, y, u, v;
        float d[4];
    };

    std::array<TileVertex, 4> build_tile()
    {
        // A square of side 17.2 px (about one texel per pixel) rotated by 17 degrees, centred off the pixel grid, so many edge pixels are
        // partially covered with their centres outside. Corners counter-clockwise; edge k runs corner k -> k+1.
        const double cx = 12.3, cy = 11.6, h = 8.6, a = 17.0 * 3.14159265358979323846 / 180.0;
        const double ca = std::cos(a), sa = std::sin(a);
        const double local[4][2] = {{-h, -h}, {h, -h}, {h, h}, {-h, h}};
        Vec2 corner[4];
        for (int k = 0; k < 4; ++k)
            corner[k] = {cx + ca * local[k][0] - sa * local[k][1], cy + sa * local[k][0] + ca * local[k][1]};
        // Texture coordinates span the centre cell inset by 1/8 texel: a value evaluated inside the tile
        // can never select a texel of another cell with NEAREST filtering.
        const double u0 = (kCell + 0.125) / kAtlas, u1 = (2 * kCell - 0.125) / kAtlas;
        std::array<TileVertex, 4> out{};
        for (int k = 0; k < 4; ++k)
        {
            TileVertex& t = out[static_cast<std::size_t>(k)];
            t.x = static_cast<float>(corner[k].x);
            t.y = static_cast<float>(corner[k].y);
            t.u = static_cast<float>(local[k][0] < 0 ? u0 : u1);
            t.v = static_cast<float>(local[k][1] < 0 ? u0 : u1);
            for (int e = 0; e < 4; ++e)
            {
                const Vec2 p = corner[e], q = corner[(e + 1) % 4];
                const double len = std::hypot(q.x - p.x, q.y - p.y);
                const double nx = -(q.y - p.y) / len, ny = (q.x - p.x) / len;  // inward normal (CCW)
                t.d[e] = static_cast<float>(nx * (corner[k].x - p.x) + ny * (corner[k].y - p.y));
            }
        }
        // TRIANGLE_STRIP order 0, 1, 3, 2.
        return {out[0], out[1], out[3], out[2]};
    }

    std::vector<std::uint8_t> build_atlas()
    {
        std::vector<std::uint8_t> px(static_cast<std::size_t>(kAtlas * kAtlas * 4));
        for (int y = 0; y < kAtlas; ++y)
            for (int x = 0; x < kAtlas; ++x)
            {
                const int cell = (y / kCell) * 3 + (x / kCell);
                std::uint8_t* p = &px[static_cast<std::size_t>((y * kAtlas + x) * 4)];
                p[3] = cell_tag(cell);
                if (cell != kCenterCell)
                {
                    p[0] = 255; p[1] = 0; p[2] = 255;  // neighbouring atlas entries: loud magenta
                    continue;
                }
                const double dx = (x % kCell) - 7.5, dy = (y % kCell) - 7.5, r = std::hypot(dx, dy);
                if (r < 5.6)
                {   // a golden disc with a soft highlight
                    p[0] = 255;
                    p[1] = static_cast<std::uint8_t>(215 - 14 * r + (dx < 0 && dy > 0 ? 25 : 0));
                    p[2] = static_cast<std::uint8_t>(70 + (r < 2.0 ? 90 : 0));
                }
                else
                {   // teal checker
                    const bool odd = ((x + y) & 1) != 0;
                    p[0] = odd ? 28 : 22; p[1] = odd ? 118 : 96; p[2] = odd ? 138 : 116;
                }
            }
        return px;
    }

    struct QuadVertex
    {
        float x, y, u, v;
    };

    // Coverage panel colours.
    constexpr std::array<float, 3> kSlot = {0.14f, 0.15f, 0.24f};
    constexpr std::array<float, 3> kColA = {0.30f, 0.85f, 0.95f};
    constexpr std::array<float, 3> kColB = {1.00f, 0.70f, 0.20f};
    constexpr std::array<float, 3> kColC = {0.95f, 0.40f, 0.65f};
    constexpr std::array<float, 3> kSky = {0.10f, 0.12f, 0.27f};
    constexpr std::array<float, 3> kPanelBg = {0.06f, 0.07f, 0.12f};
    constexpr int kBands = 9;  // values 0, 1/8, ..., 1
    constexpr int kRows = 4;
    constexpr int kCellW = 12, kCellH = 24;                         // one band cell in the band target
    constexpr int kBandW = kBands * kCellW, kBandH = kRows * kCellH;  // 108 x 96 multisample band target

    std::array<std::uint8_t, 3> rgb8(std::array<float, 3> c) { return {to8(c[0]), to8(c[1]), to8(c[2])}; }

    struct Layout
    {
        Rect flat, coverage, plain, centroid;
    };

    Layout layout_for(int w, int h)
    {
        const int m = std::max(2, std::min(w, h) / 64);
        const int pw = (w - 3 * m) / 2, ph = (h - 3 * m) / 2;
        return {{m, 2 * m + ph, pw, ph}, {2 * m + pw, 2 * m + ph, pw, ph}, {m, m, pw, ph}, {2 * m + pw, m, pw, ph}};
    }

    /// Band rectangle in the band target (pixels): row 0 is the top row.
    Rect band_rect(int row, int col) { return {col * kCellW + 2, (kRows - 1 - row) * kCellH + 3, kCellW - 4, kCellH - 6}; }

    /// Where the resolved band image is shown inside the coverage panel (stretched, NEAREST).
    Rect coverage_view_rect(Rect panel)
    {
        const int pad = std::max(2, std::min(panel.w, panel.h) / 40);
        return {panel.x + pad, panel.y + pad, panel.w - 2 * pad, panel.h - 2 * pad};
    }

    /// Square image area of a zoom panel (the frame is drawn around it).
    Rect zoom_rect(Rect panel)
    {
        const int frame = std::max(2, std::min(panel.w, panel.h) / 40);
        const int s = std::min(panel.w, panel.h) - 4 * frame;
        return {panel.x + (panel.w - s) / 2, panel.y + (panel.h - s) / 2, s, s};
    }

    class FlatCentroid final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 128 || env.height < 128)
                throw demo::Skip(std::format("surface {} x {} smaller than 128 x 128: the coverage bands and facets would be too small to probe", env.width, env.height));

            // Initial state of the multisample fragment operations (ES 3.0 Table 6.7), before anything changes it.
            initial_cov_enabled_ = mgl::glIsEnabled(mgl::Capability::SampleCoverage);
            initial_a2c_enabled_ = mgl::glIsEnabled(mgl::Capability::SampleAlphaToCoverage);
            mgl::glGetFloatv(mgl::GetParameter::SampleCoverageValue, &initial_cov_value_);
            mgl::glGetBooleanv(mgl::GetParameter::SampleCoverageInvert, &initial_cov_invert_);

            max_samples_ = demo::get_integer(mgl::GetParameter::MaxSamples);
            ms_ok_ = max_samples_ >= 2;
            ms_request_ = std::min(4, max_samples_);
            if (!ms_ok_)
                std::cout << std::format("   note: MAX_SAMPLES = {} < 2: the multisample renderbuffer checks are skipped\n", max_samples_);

            flat_color_ = demo::build_program(env, "shaders/flat.vert", "shaders/flat_color.frag");
            flat_ids_ = demo::build_program(env, "shaders/flat.vert", "shaders/flat_ids.frag");
            centroid_ = demo::build_program(env, "shaders/centroid.vert", "shaders/centroid.frag");
            solid_ = demo::build_program(env, "shaders/quad.vert", "shaders/solid.frag");
            textured_ = demo::build_program(env, "shaders/quad.vert", "shaders/texture.frag");
            u_palette_ = mgl::glGetUniformLocation(flat_color_, "u_palette");
            u_view_size_ = mgl::glGetUniformLocation(centroid_, "u_view_size");
            u_atlas_ = mgl::glGetUniformLocation(centroid_, "u_atlas");
            u_use_centroid_ = mgl::glGetUniformLocation(centroid_, "u_use_centroid");
            u_mode_ = mgl::glGetUniformLocation(centroid_, "u_mode");
            u_epsilon_ = mgl::glGetUniformLocation(centroid_, "u_epsilon");
            u_cell_alpha_ = mgl::glGetUniformLocation(centroid_, "u_cell_alpha");
            u_solid_vp_ = mgl::glGetUniformLocation(solid_, "u_viewport_size");
            u_solid_color_ = mgl::glGetUniformLocation(solid_, "u_color");
            u_tex_vp_ = mgl::glGetUniformLocation(textured_, "u_viewport_size");
            u_tex_image_ = mgl::glGetUniformLocation(textured_, "u_image");

            // --- scene ---------------------------------------------------------------------------
            scene_ = build_scene(topologies_);
            // The data must discriminate between conventions: in every triangle the provoking vertex differs
            // from the other two in palette index and shade, and the smooth average differs from it.
            for (const Topology& t : topologies_)
                for (const Tri& tri : triangles_of(t))
                {
                    const FlatVertex& p = scene_[static_cast<std::size_t>(tri.provoking)];
                    double mean = 0.0;
                    for (int k : tri.v)
                    {
                        const FlatVertex& q = scene_[static_cast<std::size_t>(k)];
                        mean += static_cast<double>(q.shade) / 3.0;
                        if (k != tri.provoking && (q.palette == p.palette || q.shade == p.shade))
                            throw demo::Failure(std::format("scene data: a {} triangle does not discriminate its provoking vertex", t.name));
                    }
                    if (std::abs(mean - static_cast<double>(p.shade)) < 0.01)
                        throw demo::Failure(std::format("scene data: a {} triangle's mean shade equals its provoking shade", t.name));
                }

            mgl::glGenVertexArrays(1, &flat_vao_);
            mgl::glGenBuffers(1, &flat_vbo_);
            mgl::glBindVertexArray(flat_vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, flat_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const FlatVertex>(scene_), mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(FlatVertex),
                                       demo::buffer_offset(offsetof(FlatVertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribIPointer(mgl::AttribLocation{1u}, 1, mgl::DataType::UnsignedInt, sizeof(FlatVertex),
                                        demo::buffer_offset(offsetof(FlatVertex, palette)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 1, mgl::DataType::Float, GL_FALSE, sizeof(FlatVertex),
                                       demo::buffer_offset(offsetof(FlatVertex, shade)));

            // --- centroid tile -------------------------------------------------------------------
            const std::array<TileVertex, 4> tile = build_tile();
            mgl::glGenVertexArrays(1, &tile_vao_);
            mgl::glGenBuffers(1, &tile_vbo_);
            mgl::glBindVertexArray(tile_vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, tile_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const TileVertex>(tile), mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(TileVertex),
                                       demo::buffer_offset(offsetof(TileVertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(TileVertex),
                                       demo::buffer_offset(offsetof(TileVertex, u)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(TileVertex),
                                       demo::buffer_offset(offsetof(TileVertex, d)));

            // --- streamed quads ------------------------------------------------------------------
            mgl::glGenVertexArrays(1, &quad_vao_);
            mgl::glGenBuffers(1, &quad_vbo_);
            mgl::glBindVertexArray(quad_vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, 4 * sizeof(QuadVertex), nullptr, mgl::BufferUsage::StreamDraw);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(QuadVertex),
                                       demo::buffer_offset(offsetof(QuadVertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(QuadVertex),
                                       demo::buffer_offset(offsetof(QuadVertex, u)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // --- textures ------------------------------------------------------------------------
            const std::vector<std::uint8_t> atlas = build_atlas();
            mgl::glGenTextures(1, &atlas_tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, atlas_tex_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kAtlas, kAtlas);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kAtlas, kAtlas, mgl::PixelFormat::Rgba,
                                 mgl::PixelType::UnsignedByte, atlas.data());
            set_nearest_clamp();

            mgl::glGenTextures(1, &view_tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, view_tex_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kOffW, kOffH);
            set_nearest_clamp();
            mgl::glGenTextures(1, &bands_tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, bands_tex_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kBandW, kBandH);
            set_nearest_clamp();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            // --- framebuffers --------------------------------------------------------------------
            const auto color0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
            const auto fb = mgl::FramebufferTarget::Framebuffer;
            const auto rb = mgl::RenderbufferTarget::Renderbuffer;

            mgl::glGenFramebuffers(1, &view_fbo_);
            mgl::glBindFramebuffer(fb, view_fbo_);
            mgl::glFramebufferTexture2D(fb, color0, mgl::TextureTarget::Texture2D, view_tex_, 0);
            require_complete("view FBO (RGBA8 texture)");
            mgl::glGenFramebuffers(1, &bands_fbo_);
            mgl::glBindFramebuffer(fb, bands_fbo_);
            mgl::glFramebufferTexture2D(fb, color0, mgl::TextureTarget::Texture2D, bands_tex_, 0);
            require_complete("band FBO (RGBA8 texture)");

            std::array<mgl::RenderbufferId, 6> rbs{};
            mgl::glGenRenderbuffers(rbs);
            ss_color_ = rbs[0]; ss_ds_ = rbs[1]; ids_rb_ = rbs[2]; ms_color_ = rbs[3]; ms_ds_ = rbs[4]; bands_ms_rb_ = rbs[5];

            mgl::glBindRenderbuffer(rb, ss_color_);
            mgl::glRenderbufferStorage(rb, mgl::InternalFormat::Rgba8, kOffW, kOffH);
            mgl::glBindRenderbuffer(rb, ss_ds_);
            mgl::glRenderbufferStorage(rb, mgl::InternalFormat::Depth24Stencil8, kOffW, kOffH);
            mgl::glGenFramebuffers(1, &ss_fbo_);
            mgl::glBindFramebuffer(fb, ss_fbo_);
            mgl::glFramebufferRenderbuffer(fb, color0, rb, ss_color_);
            mgl::glFramebufferRenderbuffer(fb, mgl::FramebufferAttachment::DepthStencil, rb, ss_ds_);
            require_complete("single-sample FBO (RGBA8 + DEPTH24_STENCIL8)");

            mgl::glBindRenderbuffer(rb, ids_rb_);
            mgl::glRenderbufferStorage(rb, mgl::InternalFormat::Rgba32UI, kIdSize, kIdSize);
            mgl::glGenFramebuffers(1, &ids_fbo_);
            mgl::glBindFramebuffer(fb, ids_fbo_);
            mgl::glFramebufferRenderbuffer(fb, color0, rb, ids_rb_);
            require_complete("flat-id FBO (RGBA32UI)");

            if (ms_ok_)
            {
                mgl::glBindRenderbuffer(rb, ms_color_);
                mgl::glRenderbufferStorageMultisample(rb, ms_request_, mgl::InternalFormat::Rgba8, kOffW, kOffH);
                mgl::glGetRenderbufferParameteriv(rb, mgl::RenderbufferParameter::Samples, &ms_color_samples_);
                mgl::glBindRenderbuffer(rb, ms_ds_);
                mgl::glRenderbufferStorageMultisample(rb, ms_request_, mgl::InternalFormat::Depth24Stencil8, kOffW, kOffH);
                mgl::glGetRenderbufferParameteriv(rb, mgl::RenderbufferParameter::Samples, &ms_ds_samples_);
                mgl::glGenFramebuffers(1, &ms_fbo_);
                mgl::glBindFramebuffer(fb, ms_fbo_);
                mgl::glFramebufferRenderbuffer(fb, color0, rb, ms_color_);
                mgl::glFramebufferRenderbuffer(fb, mgl::FramebufferAttachment::DepthStencil, rb, ms_ds_);
                require_complete("multisample FBO (RGBA8 + DEPTH24_STENCIL8)");

                mgl::glBindRenderbuffer(rb, bands_ms_rb_);
                mgl::glRenderbufferStorageMultisample(rb, ms_request_, mgl::InternalFormat::Rgba8, kBandW, kBandH);
                mgl::glGenFramebuffers(1, &bands_ms_fbo_);
                mgl::glBindFramebuffer(fb, bands_ms_fbo_);
                mgl::glFramebufferRenderbuffer(fb, color0, rb, bands_ms_rb_);
                require_complete("multisample band FBO (RGBA8)");
            }
            mgl::glBindRenderbuffer(rb, mgl::RenderbufferId{0});
            mgl::glBindFramebuffer(fb, mgl::FramebufferId{0});

            mgl::glGenQueries(1, &query_);

            // Dithering may alter the low bit of written colours; the exact pixel checks want it off.
            mgl::glDisable(mgl::Capability::Dither);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            const auto fb = mgl::FramebufferTarget::Framebuffer;

            // 1. Centroid comparison: left view without centroid, right view with centroid, in the 4x
            //    multisample renderbuffer; resolved into the view texture by glBlitFramebuffer.
            mgl::glBindFramebuffer(fb, ms_ok_ ? ms_fbo_ : view_fbo_);
            mgl::glViewport(0, 0, kOffW, kOffH);
            mgl::glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            for (int view = 0; view < 2; ++view)
            {
                mgl::glViewport(view * kView, 0, kView, kView);
                draw_tile(view == 1, 0);
            }
            if (ms_ok_)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, ms_fbo_);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, view_fbo_);
                mgl::glBlitFramebuffer(0, 0, kOffW, kOffH, 0, 0, kOffW, kOffH, mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            }

            // 2. Coverage bands for value 0, 1/8, ..., 1 in a second 4x multisample renderbuffer, resolved into
            //    the band texture. Rows (top to bottom): SAMPLE_COVERAGE; the same inverted; both on top of
            //    each other; SAMPLE_ALPHA_TO_COVERAGE with alpha = value.
            mgl::glBindFramebuffer(fb, ms_ok_ ? bands_ms_fbo_ : bands_fbo_);
            mgl::glViewport(0, 0, kBandW, kBandH);
            mgl::glClearColor(kPanelBg[0], kPanelBg[1], kPanelBg[2], 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(solid_);
            mgl::glUniform2f(u_solid_vp_, static_cast<float>(kBandW), static_cast<float>(kBandH));
            for (int row = 0; row < kRows; ++row)
                for (int col = 0; col < kBands; ++col)
                {
                    const Rect r = band_rect(row, col);
                    const float v = static_cast<float>(col) / (kBands - 1);
                    solid_rect(r, kSlot, 1.0f);
                    switch (row)
                    {
                    case 0: coverage_rect(r, kColA, 1.0f, true, v, GL_FALSE, false); break;
                    case 1: coverage_rect(r, kColB, 1.0f, true, v, GL_TRUE, false); break;
                    case 2:
                        coverage_rect(r, kColA, 1.0f, true, v, GL_FALSE, false);
                        coverage_rect(r, kColB, 1.0f, true, v, GL_TRUE, false);
                        break;
                    default: coverage_rect(r, kColC, v, false, 1.0f, GL_FALSE, true); break;
                    }
                }
            if (ms_ok_)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, bands_ms_fbo_);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, bands_fbo_);
                mgl::glBlitFramebuffer(0, 0, kBandW, kBandH, 0, 0, kBandW, kBandH, mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            }

            // 3. The default framebuffer.
            mgl::glBindFramebuffer(fb, mgl::FramebufferId{0});
            const Layout lay = layout_for(env.width, env.height);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);
            clear_rect(lay.flat, kSky);
            clear_rect(lay.coverage, kPanelBg);
            clear_rect(lay.plain, kPanelBg);
            clear_rect(lay.centroid, kPanelBg);

            // Top-left: flat-shaded low-poly scene.
            mgl::glViewport(lay.flat.x, lay.flat.y, lay.flat.w, lay.flat.h);
            draw_scene(flat_color_);

            // Top-right: the resolved coverage bands, magnified with NEAREST.
            mgl::glViewport(0, 0, env.width, env.height);
            textured_rect(env, coverage_view_rect(lay.coverage), bands_tex_, 0.0f, 1.0f);

            // Bottom: magnified resolved views (red frame: without centroid, green frame: with centroid).
            zoom_panel(env, lay.plain, 0, {0.85f, 0.30f, 0.35f});
            zoom_panel(env, lay.centroid, 1, {0.30f, 0.80f, 0.45f});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            validate_panels(env, check);
            validate_flat_ids(check);
            validate_state(check);
            if (ms_ok_)
            {
                validate_ms_coverage(check);
                validate_ms_centroid(check);
                validate_errors(check);
                measure_resolved_coverage();
            }
            else
            {
                std::cout << "   note: multisample renderbuffer checks skipped (MAX_SAMPLES < 2)\n";
            }
            validate_single_sample(check);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteQueries(1, &query_);
            const std::array<mgl::FramebufferId, 6> fbos = {view_fbo_, ss_fbo_, ids_fbo_, ms_fbo_, bands_fbo_, bands_ms_fbo_};
            mgl::glDeleteFramebuffers(fbos);
            const std::array<mgl::RenderbufferId, 6> rbs = {ss_color_, ss_ds_, ids_rb_, ms_color_, ms_ds_, bands_ms_rb_};
            mgl::glDeleteRenderbuffers(rbs);
            const std::array<mgl::TextureId, 3> texs = {atlas_tex_, view_tex_, bands_tex_};
            mgl::glDeleteTextures(texs);
            const std::array<mgl::VertexArrayId, 3> vaos = {flat_vao_, tile_vao_, quad_vao_};
            mgl::glDeleteVertexArrays(3, vaos.data());
            const std::array<mgl::BufferId, 3> vbos = {flat_vbo_, tile_vbo_, quad_vbo_};
            mgl::glDeleteBuffers(3, vbos.data());
            for (mgl::ProgramId p : {flat_color_, flat_ids_, centroid_, solid_, textured_}) mgl::glDeleteProgram(p);
        }

    private:
        // ------------------------------------------------------------------------------------------
        // Drawing helpers
        // ------------------------------------------------------------------------------------------
        static void set_nearest_clamp()
        {
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
        }

        static void require_complete(const char* what)
        {
            const mgl::FramebufferStatus s = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (s != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{} is incomplete: {}", what, demo::framebuffer_status_name(s)));
        }

        static void clear_rect(Rect r, std::array<float, 3> c)
        {
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glClearColor(c[0], c[1], c[2], 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        void draw_scene(mgl::ProgramId program)
        {
            mgl::glUseProgram(program);
            if (program == flat_color_)
            {
                std::array<float, 24> pal{};
                for (std::size_t i = 0; i < 8; ++i)
                    for (std::size_t c = 0; c < 3; ++c) pal[i * 3 + c] = kPalette[i][c];
                mgl::glUniform3fv(u_palette_, 8, pal.data());
            }
            mgl::glBindVertexArray(flat_vao_);
            for (const Topology& t : topologies_) mgl::glDrawArrays(t.mode, t.first, t.count);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        /// Uploads one quad (pixels of the current viewport) and draws it with the bound program.
        void draw_quad(float x0, float y0, float x1, float y1, float u0 = 0, float v0 = 0, float u1 = 1, float v1 = 1)
        {
            const std::array<QuadVertex, 4> q = {{{x0, y0, u0, v0}, {x1, y0, u1, v0}, {x0, y1, u0, v1}, {x1, y1, u1, v1}}};
            mgl::glBindVertexArray(quad_vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const QuadVertex>(q), mgl::BufferUsage::StreamDraw);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void solid_rect(Rect r, std::array<float, 3> c, float alpha)
        {
            mgl::glUniform4f(u_solid_color_, c[0], c[1], c[2], alpha);
            draw_quad(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.x + r.w), static_cast<float>(r.y + r.h));
        }

        /// One quad with SAMPLE_COVERAGE (value, invert) and/or SAMPLE_ALPHA_TO_COVERAGE enabled.
        void coverage_rect(Rect r, std::array<float, 3> c, float alpha, bool coverage, float value, GLboolean invert, bool a2c)
        {
            if (coverage) mgl::glEnable(mgl::Capability::SampleCoverage);
            if (a2c) mgl::glEnable(mgl::Capability::SampleAlphaToCoverage);
            mgl::glSampleCoverage(value, invert);
            solid_rect(r, c, alpha);
            mgl::glDisable(mgl::Capability::SampleCoverage);
            mgl::glDisable(mgl::Capability::SampleAlphaToCoverage);
            mgl::glSampleCoverage(1.0f, GL_FALSE);
        }

        /// Full-target quad (larger than the target, so every sample is inside) for the scratch FBOs.
        void cover_all(int w, int h, std::array<float, 3> c, float alpha, bool coverage, float value, GLboolean invert, bool a2c)
        {
            mgl::glUseProgram(solid_);
            mgl::glUniform2f(u_solid_vp_, static_cast<float>(w), static_cast<float>(h));
            coverage_rect({-4, -4, w + 8, h + 8}, c, alpha, coverage, value, invert, a2c);
        }

        void draw_tile(bool use_centroid, int mode)
        {
            mgl::glUseProgram(centroid_);
            mgl::glUniform2f(u_view_size_, static_cast<float>(kView), static_cast<float>(kView));
            mgl::glUniform1i(u_atlas_, 0);
            mgl::glUniform1i(u_use_centroid_, use_centroid ? 1 : 0);
            mgl::glUniform1i(u_mode_, mode);
            mgl::glUniform1f(u_epsilon_, kEdgeEpsilon);
            mgl::glUniform1f(u_cell_alpha_, static_cast<float>(cell_tag(kCenterCell)) / 255.0f);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, atlas_tex_);
            mgl::glBindVertexArray(tile_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void zoom_panel(const demo::Env& env, Rect panel, int view, std::array<float, 3> frame_color)
        {
            const Rect z = zoom_rect(panel);
            const int f = std::max(2, std::min(panel.w, panel.h) / 40);
            mgl::glUseProgram(solid_);
            mgl::glUniform2f(u_solid_vp_, static_cast<float>(env.width), static_cast<float>(env.height));
            solid_rect({z.x - f, z.y - f, z.w + 2 * f, z.h + 2 * f}, frame_color, 1.0f);
            const float u0 = view == 0 ? 0.0f : 0.5f;
            textured_rect(env, z, view_tex_, u0, u0 + 0.5f);
        }

        /// Draws texture columns [u0, u1] (all rows) into window rectangle r, NEAREST-magnified.
        void textured_rect(const demo::Env& env, Rect r, mgl::TextureId tex, float u0, float u1)
        {
            mgl::glUseProgram(textured_);
            mgl::glUniform2f(u_tex_vp_, static_cast<float>(env.width), static_cast<float>(env.height));
            mgl::glUniform1i(u_tex_image_, 0);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            draw_quad(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.x + r.w), static_cast<float>(r.y + r.h),
                      u0, 0.0f, u1, 1.0f);
        }

        /// ANY_SAMPLES_PASSED around `draw`.
        template <typename F>
        bool any_samples(F&& draw)
        {
            mgl::glBeginQuery(mgl::QueryTarget::AnySamplesPassed, query_);
            draw();
            mgl::glEndQuery(mgl::QueryTarget::AnySamplesPassed);
            GLuint result = 0;
            mgl::glGetQueryObjectuiv(query_, mgl::QueryObjectParameter::Result, &result);
            return result != 0;
        }

        /// Stencil counting: clear stencil to 0 and make every following draw increment it per written sample.
        static void begin_stencil_count()
        {
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilMask(0xFF);
            mgl::glClearStencil(0);
            mgl::glClear(mgl::ClearBufferBit::Stencil);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFF);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Incr);
        }

        /// Does any sample of the target have a stencil value s with (ref FUNC s)? Full quad, colour writes off.
        bool any_stencil(int w, int h, mgl::CompareFunc func, int ref)
        {
            mgl::glStencilFunc(func, ref, 0xFF);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            const bool r = any_samples([&] { cover_all(w, h, {1, 1, 1}, 1.0f, false, 1.0f, GL_FALSE, false); });
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Incr);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFF);
            return r;
        }

        static void end_stencil_count() { mgl::glDisable(mgl::Capability::StencilTest); }

        // ------------------------------------------------------------------------------------------
        // Validation
        // ------------------------------------------------------------------------------------------
        /// Checks that window rectangle z shows texture columns [col0, col0 + cols) x all rows of `tex`
        /// (a tex_w-texel-wide RGBA8 readback) through a NEAREST-filtered textured quad. The default
        /// framebuffer is single-sampled, so the texture coordinate is evaluated at the pixel centre
        /// (GLSL ES 3.00 4.3.9) and NEAREST selects floor(u * size) (ES 3.0 3.8.10): one exact texel per
        /// pixel. Only where the centre maps within 1/1000 texel of a texel boundary is the neighbour also
        /// accepted. Probes a 7 x 7 grid spread over the rectangle.
        static void check_magnified(demo::Check& check, const std::vector<std::uint8_t>& img, int img_w, Rect z,
                                    const std::vector<std::uint8_t>& tex, int tex_w, int col0, int cols, int rows, const std::string& what)
        {
            auto candidates = [](int p, int size, int texels) {
                const double t = (p + 0.5) * texels / size;
                const int i = static_cast<int>(std::floor(t));
                const double f = t - i;
                return std::pair<int, int>{std::clamp(f < 1e-3 ? i - 1 : i, 0, texels - 1), std::clamp(f > 1.0 - 1e-3 ? i + 1 : i, 0, texels - 1)};
            };
            int n = 0, bad = 0;
            std::string first_bad;
            for (int gy = 1; gy <= 7; ++gy)
                for (int gx = 1; gx <= 7; ++gx)
                {
                    const int x = z.x + z.w * gx / 8 + gx % 3, y = z.y + z.h * gy / 8 + gy % 3;
                    const auto [tx0, tx1] = candidates(x - z.x, z.w, cols);
                    const auto [ty0, ty1] = candidates(y - z.y, z.h, rows);
                    const std::uint8_t* p = &img[static_cast<std::size_t>((y * img_w + x) * 4)];
                    bool ok = false;
                    for (int ty = ty0; ty <= ty1 && !ok; ++ty)
                        for (int tx = tx0; tx <= tx1 && !ok; ++tx)
                        {
                            const std::uint8_t* t = &tex[static_cast<std::size_t>((ty * tex_w + col0 + tx) * 4)];
                            ok = p[0] == t[0] && p[1] == t[1] && p[2] == t[2];
                        }
                    ++n;
                    if (!ok && bad++ == 0)
                    {
                        const std::uint8_t* t = &tex[static_cast<std::size_t>((ty0 * tex_w + col0 + tx0) * 4)];
                        first_bad = std::format("pixel ({},{}) = {},{},{}, texel ({},{}) = {},{},{}", x, y, p[0], p[1], p[2], col0 + tx0, ty0, t[0],
                                                t[1], t[2]);
                    }
                }
            check.expect(bad == 0, what, std::format("{} probes, {} wrong {}", n, bad, first_bad));
        }

        void validate_panels(demo::Env& env, demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const Layout lay = layout_for(env.width, env.height);
            const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, env.width, env.height);
            auto px = [&](int x, int y) { return &img[static_cast<std::size_t>((y * env.width + x) * 4)]; };
            // Every window probe below needs a single-sample window: there a pixel is the fragment color at its centre, while a
            // multisampled window would put its own implementation-dependent resolve (ES 3.0 4.1.10) in between. The runtime
            // asks for a config without samples.
            const bool single_sample_window = demo::get_integer(mgl::GetParameter::SampleBuffers) == 0;
            if (!single_sample_window) std::cout << "   note: the default framebuffer is multisampled here; the window probes are not made\n";

            // --- flat panel: every probed facet shows palette[provoking] * shade[provoking] ---------
            for (const Topology& t : topologies_)
            {
                if (!single_sample_window) break;
                int probes = 0, bad = 0;
                std::string first_bad;
                for (const Tri& tri : triangles_of(t))
                {
                    std::array<Vec2, 3> w{};
                    for (std::size_t k = 0; k < 3; ++k) w[k] = to_window(scene_[static_cast<std::size_t>(tri.v[k])], lay.flat);
                    const FlatVertex& p = scene_[static_cast<std::size_t>(tri.provoking)];
                    const auto& pal = kPalette[p.palette];
                    const std::array<std::uint8_t, 3> e = {to8(pal[0] * p.shade), to8(pal[1] * p.shade), to8(pal[2] * p.shade)};
                    for (const Probe& pr : probes_for(w, 1.0))
                    {
                        ++probes;
                        if (!near_rgb(px(pr.px, pr.py), e, 1) && bad++ == 0)
                        {
                            const std::uint8_t* a = px(pr.px, pr.py);
                            first_bad = std::format("({},{}) = {},{},{} expected {},{},{} (vertex {})", pr.px, pr.py, a[0], a[1], a[2], e[0], e[1],
                                                    e[2], tri.provoking);
                        }
                    }
                }
                check.expect(probes > 0 && bad == 0,
                             std::format("displayed {} facets show palette * shade of their provoking (last) vertex", t.name),
                             std::format("{} probes, {} wrong {}", probes, bad, first_bad));
            }

            // --- the resolved band image: observed, not asserted ---------------------------------------
            // ES 3.0 4.3.3: "If the source formats are floating-point or normalized types, the sample values for each pixel are
            // resolved in an implementation-dependent manner." The specification does not even say that a pixel whose samples
            // all hold one value resolves to that value, so the endpoint bands are only printed. What the endpoint bands show is
            // asserted per sample, without any resolve, in validate_ms_coverage() (occlusion queries and stencil INCR counts).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, bands_fbo_);
            const std::vector<std::uint8_t> bands = demo::read_rgba8(0, 0, kBandW, kBandH);
            auto bpx = [&](int x, int y) { return &bands[static_cast<std::size_t>((y * kBandW + x) * 4)]; };
            if (ms_ok_)
            {
                struct Band { int row, col; std::array<float, 3> color; const char* what; };
                const Band list[] = {
                    {0, 0, kSlot, "SAMPLE_COVERAGE value 0 writes no sample: the slot colour remains"},
                    {0, kBands - 1, kColA, "SAMPLE_COVERAGE value 1 writes every sample"},
                    {1, 0, kColB, "SAMPLE_COVERAGE value 0 inverted writes every sample"},
                    {1, kBands - 1, kSlot, "SAMPLE_COVERAGE value 1 inverted writes no sample"},
                    {2, 0, kColB, "complementary pair at value 0: only the inverted draw is visible"},
                    {2, kBands - 1, kColA, "complementary pair at value 1: only the non-inverted draw is visible"},
                    {3, 0, kSlot, "SAMPLE_ALPHA_TO_COVERAGE alpha 0 writes no sample"},
                    {3, kBands - 1, kColC, "SAMPLE_ALPHA_TO_COVERAGE alpha 1 writes every sample"},
                };
                int exact_bands = 0;
                std::string off;
                for (const Band& b : list)
                {
                    const Rect r = band_rect(b.row, b.col);
                    const auto e = rgb8(b.color);
                    int bad = 0;
                    for (int y = r.y; y < r.y + r.h; ++y)
                        for (int x = r.x; x < r.x + r.w; ++x) bad += near_rgb(bpx(x, y), e, 1) ? 0 : 1;
                    exact_bands += bad == 0 ? 1 : 0;
                    if (bad != 0) off += std::format("; '{}': {} of {} pixels off", b.what, bad, r.w * r.h);
                }
                std::cout << std::format("   note: observed resolve of the endpoint bands (implementation-dependent, not asserted): {} of {} bands show the "
                                         "color their samples were given (+-1){}\n",
                                         exact_bands, std::size(list), off);
                // Intermediate bands: the resolve is implementation-dependent (ES 3.0 4.3.3), so only print.
                std::string row0;
                for (int col = 0; col < kBands; ++col)
                {
                    const Rect r = band_rect(0, col);
                    double sum = 0;
                    for (int y = r.y; y < r.y + r.h; ++y)
                        for (int x = r.x; x < r.x + r.w; ++x)
                            sum += (bpx(x, y)[1] - 255.0 * static_cast<double>(kSlot[1])) / (255.0 * static_cast<double>(kColA[1] - kSlot[1]));
                    row0 += std::format(" {:.2f}", std::max(0.0, sum / (r.w * r.h)));
                }
                std::cout << std::format("   note: resolved SAMPLE_COVERAGE band coverage for value 0..1 step 1/8:{}\n", row0);
            }

            // --- the panels show the resolved images, magnified with NEAREST --------------------------
            // These compare window pixels with the *read-back* resolved texels, so they hold whatever the resolve produced.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, view_fbo_);
            const std::vector<std::uint8_t> view = demo::read_rgba8(0, 0, kOffW, kOffH);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            if (!single_sample_window) return;
            check_magnified(check, img, env.width, coverage_view_rect(lay.coverage), bands, kBandW, 0, kBandW, kBandH,
                            "coverage panel shows the resolved band image (NEAREST magnification)");
            check_magnified(check, img, env.width, zoom_rect(lay.plain), view, kOffW, 0, kView, kView,
                            "non-centroid zoom panel shows the resolved texels (NEAREST magnification)");
            check_magnified(check, img, env.width, zoom_rect(lay.centroid), view, kOffW, kView, kView, kView,
                            "centroid zoom panel shows the resolved texels (NEAREST magnification)");
        }

        void validate_flat_ids(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ids_fbo_);
            mgl::glViewport(0, 0, kIdSize, kIdSize);
            const GLuint sentinel[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, sentinel);
            draw_scene(flat_ids_);
            std::vector<GLuint> ids(static_cast<std::size_t>(kIdSize * kIdSize * 4));
            mgl::glReadPixels(0, 0, kIdSize, kIdSize, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, ids.data());
            check.expect_no_gl_error("RGBA32UI readback with RGBA_INTEGER / UNSIGNED_INT");

            const Rect vp{0, 0, kIdSize, kIdSize};
            for (const Topology& t : topologies_)
            {
                int triangles = 0, unprobed = 0, probes = 0, bad = 0, control_ok = 0, control_n = 0;
                double worst_smooth_err = 0.0;
                std::string first_bad;
                for (const Tri& tri : triangles_of(t))
                {
                    ++triangles;
                    std::array<Vec2, 3> w{};
                    for (std::size_t k = 0; k < 3; ++k) w[k] = to_window(scene_[static_cast<std::size_t>(tri.v[k])], vp);
                    const FlatVertex& p = scene_[static_cast<std::size_t>(tri.provoking)];
                    const std::vector<Probe> ps = probes_for(w, 1.0);
                    if (ps.empty()) ++unprobed;
                    // Shade gradient (per pixel) for the smooth control's tolerance: vertex snapping to a 1/16 px
                    // grid (SUBPIXEL_BITS >= 4) moves the plane by at most ~1/16 px times the gradient.
                    const double s0 = scene_[static_cast<std::size_t>(tri.v[0])].shade, s1 = scene_[static_cast<std::size_t>(tri.v[1])].shade,
                                 s2 = scene_[static_cast<std::size_t>(tri.v[2])].shade;
                    const double a2 = cross(w[0], w[1], w[2]);
                    const double gx = ((s1 - s0) * (w[2].y - w[0].y) - (s2 - s0) * (w[1].y - w[0].y)) / a2;
                    const double gy = ((s2 - s0) * (w[1].x - w[0].x) - (s1 - s0) * (w[2].x - w[0].x)) / a2;
                    const double tol = std::hypot(gx, gy) / 8.0 + 1e-4;
                    for (std::size_t i = 0; i < ps.size(); ++i)
                    {
                        const Probe& pr = ps[i];
                        ++probes;
                        const GLuint* o = &ids[static_cast<std::size_t>((pr.py * kIdSize + pr.px) * 4)];
                        const bool ok = o[0] == static_cast<GLuint>(tri.provoking) && o[1] == p.palette &&
                                        o[2] == std::bit_cast<GLuint>(p.shade);
                        if (!ok && bad++ == 0)
                            first_bad = std::format("pixel ({},{}): vertex id {} palette {} shade {} - expected vertex {} palette {} shade {}",
                                                    pr.px, pr.py, o[0], o[1], std::bit_cast<float>(o[2]), tri.provoking, p.palette, p.shade);
                        // Negative control at the centroid probe: the smooth copy is interpolated (CPU barycentric
                        // reference) and therefore differs from the provoking value.
                        if (i == 0)
                        {
                            ++control_n;
                            const double smooth = std::bit_cast<float>(o[3]);
                            const double expect = pr.w[0] * s0 + pr.w[1] * s1 + pr.w[2] * s2;
                            worst_smooth_err = std::max(worst_smooth_err, std::abs(smooth - expect));
                            if (std::abs(smooth - expect) <= tol && std::abs(smooth - static_cast<double>(p.shade)) > tol) ++control_ok;
                        }
                    }
                }
                check.expect(unprobed == 0 && bad == 0,
                             std::format("{}: every triangle carries gl_VertexID, flat uint and flat float (bit-exact) of its provoking vertex "
                                         "(ES 3.0 Table 2.12)", t.name),
                             std::format("{} triangles, {} without probe, {} probes, {} wrong; {}", triangles, unprobed, probes, bad, first_bad));
                check.expect(control_n > 0 && control_ok == control_n,
                             std::format("{}: negative control - the smooth copy matches barycentric interpolation, not the provoking value",
                                         t.name),
                             std::format("{} of {} centroid probes; worst smooth error {:.2e}", control_ok, control_n, worst_smooth_err));
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void validate_state(demo::Check& check)
        {
            check.expect(!initial_cov_enabled_ && !initial_a2c_enabled_ && initial_cov_value_ == 1.0f && initial_cov_invert_ == GL_FALSE,
                         "initial state: SAMPLE_COVERAGE and SAMPLE_ALPHA_TO_COVERAGE disabled, value 1, invert FALSE (ES 3.0 Table 6.7)",
                         std::format("enabled {} / {}, value {}, invert {}", initial_cov_enabled_, initial_a2c_enabled_, initial_cov_value_,
                                     initial_cov_invert_));
            struct Case { float in; GLboolean inv; float expect; };
            const Case cases[] = {{1.75f, GL_TRUE, 1.0f}, {-0.5f, GL_FALSE, 0.0f}, {0.375f, GL_TRUE, 0.375f}};
            for (const Case& c : cases)
            {
                mgl::glSampleCoverage(c.in, c.inv);
                GLfloat value = -1.0f;
                GLboolean inv = 2;
                mgl::glGetFloatv(mgl::GetParameter::SampleCoverageValue, &value);
                mgl::glGetBooleanv(mgl::GetParameter::SampleCoverageInvert, &inv);
                check.expect(value == c.expect && inv == c.inv,
                             std::format("glSampleCoverage({}, {}) stores SAMPLE_COVERAGE_VALUE {} (clamped to [0,1]) and the invert flag",
                                         c.in, c.inv == GL_TRUE ? "TRUE" : "FALSE", c.expect),
                             std::format("value {}, invert {}", value, inv));
            }
            mgl::glSampleCoverage(1.0f, GL_FALSE);

            for (auto [cap, pname, name] : {std::tuple{mgl::Capability::SampleCoverage, mgl::GetParameter::SampleCoverage, "SAMPLE_COVERAGE"},
                                            std::tuple{mgl::Capability::SampleAlphaToCoverage, mgl::GetParameter::SampleAlphaToCoverage,
                                                       "SAMPLE_ALPHA_TO_COVERAGE"}})
            {
                mgl::glEnable(cap);
                GLboolean on = GL_FALSE;
                mgl::glGetBooleanv(pname, &on);
                const bool enabled = mgl::glIsEnabled(cap);
                mgl::glDisable(cap);
                GLboolean off = GL_TRUE;
                mgl::glGetBooleanv(pname, &off);
                check.expect(enabled && on == GL_TRUE && off == GL_FALSE && !mgl::glIsEnabled(cap),
                             std::format("glEnable/glDisable({}) reported by glIsEnabled and glGetBooleanv", name));
            }
        }

        void validate_ms_coverage(demo::Check& check)
        {
            const auto fb = mgl::FramebufferTarget::Framebuffer;
            mgl::glBindFramebuffer(fb, ms_fbo_);
            mgl::glViewport(0, 0, kOffW, kOffH);
            const int sb = demo::get_integer(mgl::GetParameter::SampleBuffers);
            const int s = demo::get_integer(mgl::GetParameter::Samples);
            check.expect(ms_color_samples_ >= ms_request_ && ms_ds_samples_ == ms_color_samples_ && sb == 1 && s == ms_color_samples_,
                         std::format("glRenderbufferStorageMultisample({} samples): RENDERBUFFER_SAMPLES >= request; FBO SAMPLE_BUFFERS 1, "
                                     "SAMPLES = RENDERBUFFER_SAMPLES (ES 3.0 4.4.2.1, 4.4.4.2)", ms_request_),
                         std::format("colour {}, depth/stencil {}, SAMPLE_BUFFERS {}, SAMPLES {}", ms_color_samples_, ms_ds_samples_, sb, s));
            std::cout << std::format("   note: multisample renderbuffer has {} samples\n", ms_color_samples_);

            const std::array<float, 3> white = {1, 1, 1};
            auto draw = [&](float alpha, bool cov, float value, GLboolean inv, bool a2c) {
                cover_all(kOffW, kOffH, white, alpha, cov, value, inv, a2c);
            };
            mgl::glClearColor(0, 0, 0, 1);
            mgl::glClear(mgl::ClearBufferBit::Color);

            // Single draws observed by the occlusion query.
            check.expect(!any_samples([&] { draw(1, true, 0.0f, GL_FALSE, false); }), "SAMPLE_COVERAGE value 0: no sample is written (query FALSE)");
            check.expect(!any_samples([&] { draw(1, true, 1.0f, GL_TRUE, false); }), "SAMPLE_COVERAGE value 1 inverted: no sample is written");
            check.expect(any_samples([&] { draw(1, true, 0.5f, GL_FALSE, false); }), "positive control: SAMPLE_COVERAGE value 0.5 writes samples (query TRUE)");
            check.expect(!any_samples([&] { draw(0, false, 1.0f, GL_FALSE, true); }), "SAMPLE_ALPHA_TO_COVERAGE alpha 0: no sample is written");
            check.expect(!any_samples([&] { draw(1, true, 0.0f, GL_FALSE, true); }),
                         "alpha-to-coverage (alpha 1) ANDed with SAMPLE_COVERAGE value 0: no sample is written");
            check.expect(!any_samples([&] { draw(0, true, 0.0f, GL_TRUE, true); }),
                         "alpha-to-coverage (alpha 0) ANDed with SAMPLE_COVERAGE all ones (0 inverted): no sample is written");

            // Every sample written exactly once: stencil INCR, then a query for stencil != 1.
            auto exactly_once = [&](auto&& draws) {
                begin_stencil_count();
                draws();
                const bool other = any_stencil(kOffW, kOffH, mgl::CompareFunc::Notequal, 1);
                end_stencil_count();
                return !other;
            };
            check.expect(exactly_once([&] { draw(1, true, 1.0f, GL_FALSE, false); }), "SAMPLE_COVERAGE value 1: every sample is written");
            check.expect(exactly_once([&] { draw(1, true, 0.0f, GL_TRUE, false); }), "SAMPLE_COVERAGE value 0 inverted: every sample is written");
            check.expect(exactly_once([&] { draw(1, false, 1.0f, GL_FALSE, true); }), "SAMPLE_ALPHA_TO_COVERAGE alpha 1: every sample is written");
            std::string failing;
            for (float v : {0.125f, 0.25f, 0.3f, 0.375f, 0.5f, 0.625f, 0.7f, 0.75f, 0.875f})
                if (!exactly_once([&] { draw(1, true, v, GL_FALSE, false); draw(1, true, v, GL_TRUE, false); }))
                    failing += std::format(" {}", v);
            check.expect(failing.empty(),
                         "glSampleCoverage(v, FALSE) then (v, TRUE) write complementary masks: every sample exactly once (9 values)",
                         "failing values:" + failing);
            check.expect(!exactly_once([&] { draw(1, true, 0.5f, GL_FALSE, false); draw(1, true, 0.5f, GL_FALSE, false); }),
                         "negative control: (0.5, FALSE) twice does NOT write every sample exactly once");
            {
                begin_stencil_count();
                draw(1, true, 0.5f, GL_FALSE, false);
                const bool written = any_stencil(kOffW, kOffH, mgl::CompareFunc::Equal, 1);
                const bool skipped = any_stencil(kOffW, kOffH, mgl::CompareFunc::Equal, 0);
                end_stencil_count();
                check.expect(written && skipped, "SAMPLE_COVERAGE value 0.5: a partial mask (some samples written, some not)",
                             std::format("written {}, skipped {}", written, skipped));
            }
            mgl::glBindFramebuffer(fb, mgl::FramebufferId{0});
        }

        void validate_ms_centroid(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ms_fbo_);
            mgl::glViewport(0, 0, kView, kView);
            mgl::glClearColor(0, 0, 0, 1);
            mgl::glClear(mgl::ClearBufferBit::Color);
            auto q = [&](bool centroid, int mode) { return any_samples([&] { draw_tile(centroid, mode); }); };

            check.expect(!q(true, 1), "centroid: no fragment is evaluated outside the primitive (edge distance >= -1/16 px) (GLSL ES 3.00 4.3.9)");
            check.expect(!q(true, 2), "centroid: no fragment samples a neighbouring atlas cell (no texture bleed)");
            check.expect(q(true, 3), "positive control: fragments within 2 px inside an edge are detected (query TRUE)");
            const bool plain_outside = q(false, 1), plain_bleed = q(false, 2), differs = q(true, 4);
            // Without centroid the value may be taken at the pixel centre, anywhere in the pixel or at any sample
            // (GLSL ES 3.00 4.3.9), so extrapolation is permitted, not required: reported, not asserted.
            std::cout << std::format("   note: without centroid: fragments evaluated outside the tile: {}; atlas bleed: {}; "
                                     "centroid value differs from the default somewhere: {}\n",
                                     plain_outside ? "yes" : "no", plain_bleed ? "yes" : "no", differs ? "yes" : "no");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void validate_errors(demo::Check& check)
        {
            demo::drain_gl_errors();
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, ss_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, ms_fbo_);
            mgl::glBlitFramebuffer(0, 0, 8, 8, 0, 0, 8, 8, mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glBlitFramebuffer into a multisample FBO (draw SAMPLE_BUFFERS > 0) -> INVALID_OPERATION (ES 3.0 4.3.3)");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, ms_fbo_);
            std::array<std::uint8_t, 4> sink{};
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, sink.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glReadPixels from a multisample FBO -> INVALID_OPERATION (ES 3.0 4.3.2)");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        /// Prints the resolved coverage curve. The resolve of normalized formats is implementation-dependent
        /// (ES 3.0 4.3.3), so monotonicity of the resolved values is reported, not asserted.
        void measure_resolved_coverage()
        {
            auto curve = [&](bool a2c) {
                std::string out;
                double prev = -1.0;
                bool monotonic = true;
                for (int i = 0; i <= 16; ++i)
                {
                    const float v = static_cast<float>(i) / 16.0f;
                    mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ms_fbo_);
                    mgl::glViewport(0, 0, kOffW, kOffH);
                    mgl::glClearColor(0, 0, 0, 1);
                    mgl::glClear(mgl::ClearBufferBit::Color);
                    if (a2c) cover_all(kOffW, kOffH, {1, 1, 1}, v, false, 1.0f, GL_FALSE, true);
                    else cover_all(kOffW, kOffH, {1, 1, 1}, 1.0f, true, v, GL_FALSE, false);
                    mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, ms_fbo_);
                    mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, ss_fbo_);
                    mgl::glBlitFramebuffer(0, 0, kOffW, kOffH, 0, 0, kOffW, kOffH, mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
                    mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, ss_fbo_);
                    const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, kOffW, kOffH);
                    double sum = 0;
                    for (std::size_t p = 0; p < img.size(); p += 4) sum += img[p] / 255.0;
                    const double mean = sum / (kOffW * kOffH);
                    monotonic = monotonic && mean >= prev;
                    prev = mean;
                    out += std::format(" {:.3f}", mean);
                }
                std::cout << std::format("   note: resolved {} coverage for 0..1 step 1/16:{} ({})\n", a2c ? "alpha-to-coverage" : "SAMPLE_COVERAGE",
                                         out, monotonic ? "monotonic" : "NOT monotonic");
            };
            curve(false);
            curve(true);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void validate_single_sample(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ss_fbo_);
            mgl::glViewport(0, 0, kOffW, kOffH);
            const int sb = demo::get_integer(mgl::GetParameter::SampleBuffers);
            const int s = demo::get_integer(mgl::GetParameter::Samples);
            check.expect(sb == 0 && s == 0, "single-sample FBO: SAMPLE_BUFFERS 0, SAMPLES 0",
                         std::format("SAMPLE_BUFFERS {}, SAMPLES {}", sb, s));
            {
                // The attachments themselves were allocated with glRenderbufferStorage, i.e. RENDERBUFFER_SAMPLES 0.
                GLint cs = -1, ds = -1;
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, ss_color_);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &cs);
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, ss_ds_);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &ds);
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
                check.expect(cs == 0 && ds == 0, "single-sample FBO: both attachments report RENDERBUFFER_SAMPLES 0",
                             std::format("colour {}, depth/stencil {}", cs, ds));
            }

            // ANGLE (Vulkan backend) applies the coverage mask to the single sample although it reports SAMPLE_BUFFERS 0.
            constexpr std::string_view kAngleReason =
                "ANGLE applies SAMPLE_COVERAGE / SAMPLE_ALPHA_TO_COVERAGE to a framebuffer whose SAMPLE_BUFFERS is 0 "
                "(ES 3.0 4.1.3: no change is made if SAMPLE_BUFFERS is not one)";

            // With SAMPLE_BUFFERS = 0 the multisample fragment operations change nothing (ES 3.0 4.1.3).
            struct Case { const char* what; std::array<float, 3> c; float alpha; bool cov; float v; GLboolean inv; bool a2c; };
            const Case cases[] = {
                {"SAMPLE_COVERAGE value 0", {0.2f, 0.6f, 0.9f}, 1.0f, true, 0.0f, GL_FALSE, false},
                {"SAMPLE_COVERAGE value 1 inverted", {0.9f, 0.5f, 0.1f}, 1.0f, true, 1.0f, GL_TRUE, false},
                {"SAMPLE_ALPHA_TO_COVERAGE alpha 0", {0.4f, 0.9f, 0.3f}, 0.0f, false, 1.0f, GL_FALSE, true},
            };
            for (const Case& c : cases)
            {
                mgl::glClearColor(0, 0, 0, 1);
                mgl::glClear(mgl::ClearBufferBit::Color);
                cover_all(kOffW, kOffH, c.c, c.alpha, c.cov, c.v, c.inv, c.a2c);
                const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, kOffW, kOffH);
                const auto e = rgb8(c.c);
                const std::uint8_t ea = to8(c.alpha);
                int bad = 0;
                for (std::size_t p = 0; p < img.size(); p += 4)
                    if (!near_rgb(&img[p], e, 1) || std::abs(img[p + 3] - ea) > 1) ++bad;
                check.expect_or_known_deviation(bad == 0, std::format("single-sample FBO: {} has no effect - every pixel is written", c.what),
                                                std::format("{} of {} pixels differ, pixel (0,0) = ({},{},{},{})", bad, kOffW * kOffH,
                                                            img[0], img[1], img[2], img[3]),
                                                "ANGLE", kAngleReason);
            }
            begin_stencil_count();
            cover_all(kOffW, kOffH, {1, 1, 1}, 1.0f, true, 0.5f, GL_FALSE, false);
            const bool other = any_stencil(kOffW, kOffH, mgl::CompareFunc::Notequal, 1);
            end_stencil_count();
            check.expect_or_known_deviation(!other, "single-sample FBO: SAMPLE_COVERAGE 0.5 has no effect - every pixel's stencil incremented exactly once",
                                            "", "ANGLE", kAngleReason);

            // Centroid on a single-sample target: interpolation at the pixel centre, the qualifier is ignored.
            mgl::glViewport(0, 0, kView, kView);
            mgl::glClearColor(0, 0, 0, 1);
            mgl::glClear(mgl::ClearBufferBit::Color);
            auto q = [&](bool centroid, int mode) { return any_samples([&] { draw_tile(centroid, mode); }); };
            check.expect(!q(false, 1) && !q(true, 1),
                         "single-sample: no fragment is evaluated outside the primitive, with or without centroid (pixel centre is inside)");
            check.expect(!q(true, 4), "single-sample: centroid value equals the pixel-centre value (qualifier ignored, GLSL ES 3.00 4.3.9)");
            check.expect(q(true, 3), "single-sample positive control: near-edge fragments are detected (query TRUE)");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // ------------------------------------------------------------------------------------------
        std::array<Topology, 3> topologies_{};
        std::vector<FlatVertex> scene_;

        bool initial_cov_enabled_ = true, initial_a2c_enabled_ = true;
        GLfloat initial_cov_value_ = -1.0f;
        GLboolean initial_cov_invert_ = GL_TRUE;

        int max_samples_ = 0, ms_request_ = 0;
        bool ms_ok_ = false;
        GLint ms_color_samples_ = 0, ms_ds_samples_ = 0;

        mgl::ProgramId flat_color_{}, flat_ids_{}, centroid_{}, solid_{}, textured_{};
        mgl::UniformLocation u_palette_{}, u_view_size_{}, u_atlas_{}, u_use_centroid_{}, u_mode_{}, u_epsilon_{}, u_cell_alpha_{};
        mgl::UniformLocation u_solid_vp_{}, u_solid_color_{}, u_tex_vp_{}, u_tex_image_{};
        mgl::VertexArrayId flat_vao_{}, tile_vao_{}, quad_vao_{};
        mgl::BufferId flat_vbo_{}, tile_vbo_{}, quad_vbo_{};
        mgl::TextureId atlas_tex_{}, view_tex_{}, bands_tex_{};
        mgl::FramebufferId view_fbo_{}, ss_fbo_{}, ids_fbo_{}, ms_fbo_{}, bands_fbo_{}, bands_ms_fbo_{};
        mgl::RenderbufferId ss_color_{}, ss_ds_{}, ids_rb_{}, ms_color_{}, ms_ds_{}, bands_ms_rb_{};
        mgl::QueryId query_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-flat-centroid",
        .title = "Flat shading, centroid and sample coverage",
        .min_es = {3, 0},
        .summary = "Provoking-vertex flat shading (int and float), centroid vs. default interpolation on a 4x multisample "
                   "renderbuffer (texture-atlas bleed), and glSampleCoverage / alpha-to-coverage on multisample and "
                   "single-sample targets, checked with exact readback, stencil counting and occlusion queries.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<FlatCentroid>(argc, argv, info);
}
