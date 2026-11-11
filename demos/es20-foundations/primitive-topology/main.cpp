// es20-primitive-topology
//
// What the `mode` argument of glDrawArrays means: how a list of vertices is cut into points,
// line segments and triangles (ES 3.0 section 2.7.1), and how those primitives are rasterized
// (sections 3.4 points, 3.5 line segments, 3.6 polygons).
//
// The image is a gallery of 16 cells of 64x64 pixels (4x4, cell 0 top-left, cell 15 bottom-right),
// every cell isolated from its neighbours by its own viewport and scissor rectangle:
//
//    0 POINTS (sizes)       1 POINTS + gl_PointCoord   2 LINES               3 LINE_STRIP
//    4 LINE_LOOP            5 TRIANGLES (+2 leftover)  6 TRIANGLE_STRIP      7 TRIANGLE_FAN
//    8 wide lines           9 largest point            10 line rules          11 glDrawArrays(first, count)
//   12 too few vertices    13 fan as a disc           14 loop as an octagon  15 all seven types in one scene
//
// Cells 2-4 draw the SAME six vertices with the three line modes, so the only difference between
// the pictures is the topology. The vertex data lives in one buffer; each draw selects its range
// with `first` and `count`.
//
// Validation. For every cell a CPU reference is painted from the specification alone:
//   * the vertex list is cut into primitives with the rules of section 2.7.1 (independent of the GPU);
//   * a point covers the pixels whose centers lie inside its square (section 3.4.1);
//   * a triangle covers the pixels whose centers lie inside it;
//   * a line through pixel centers, axis-aligned or at 45 degrees, covers exactly one pixel per
//     column (row) of its major axis (section 3.5.1, diamond-exit rule and rule 3).
// Rasterization of edges and of general line slopes is implementation-dependent within the rules of
// section 3.5.1, so pixels near an edge are not compared ("don't care"). Everything else, both lit and
// unlit pixels, must match; a full cell is one check, and named probes explain the key facts.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <numbers>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Layout, palette, vertex data types
    // ---------------------------------------------------------------------------------------------
    constexpr int kCell = 64;
    constexpr int kCellCount = 16;
    // The 16 cells are arranged as 4x4, 8x2 or 2x8, whichever fits the surface (chosen in init()).
    // The cell content is defined in exact cell-local pixels, so a cell is never scaled.
    int g_columns = 4;
    int g_rows = 4;

    struct Rect
    {
        int x, y, w, h;
    };
    struct Color
    {
        float r, g, b, a;
    };
    struct Vec2
    {
        float x, y;
    };

    enum class Cell : int
    {
        Points, PointCoord, Lines, LineStrip,
        LineLoop, Triangles, TriangleStrip, TriangleFan,
        WideLines, MaxPoint, LineRules, FirstAndCount,
        Degenerate, FanDisc, Octagon, Scene,
    };

    constexpr int index_of(Cell c) { return static_cast<int>(c); }

    Rect cell_rect(Cell c)
    {
        const int i = index_of(c);
        return {(i % g_columns) * kCell, (g_rows - 1 - i / g_columns) * kCell, kCell, kCell};
    }

    constexpr Color kSurface{0.02f, 0.02f, 0.03f, 1.0f};
    constexpr Color kBg{0.06f, 0.07f, 0.10f, 1.0f};

    /// Vertex in cell-local window coordinates (pixels, origin lower-left) plus gl_PointSize.
    struct Vtx
    {
        float x, y, size;
    };

    /// One glDrawArrays call. `first` is relative to the cell's own vertices.
    struct Draw
    {
        mgl::PrimitiveType mode;
        int first;
        int count;
        Color color;
        float line_width = 1.0f;
        bool point_coord = false;  // use the gl_PointCoord program instead of the solid-color one
    };

    struct CellData
    {
        std::vector<Vtx> verts;
        std::vector<Draw> draws;
        int base = 0;       // index of verts[0] in the shared vertex buffer
        int tolerance = 1;  // per-channel comparison tolerance (gl_PointCoord is mediump)
    };

    /// Implementation-dependent ranges (ES 3.0 sections 3.4 and 3.5), queried once.
    struct Limits
    {
        float point_lo = 1.0f, point_hi = 1.0f;
        float line_lo = 1.0f, line_hi = 1.0f;
    };

    struct GpuVertex
    {
        float x, y, size;
    };

    constexpr float kPi = std::numbers::pi_v<float>;

    // ---------------------------------------------------------------------------------------------
    // Section 2.7.1: cutting a vertex list into primitives (the CPU reference of glDrawArrays)
    // ---------------------------------------------------------------------------------------------
    struct Segment
    {
        int a, b;
    };
    using Triangle = std::array<int, 3>;

    std::vector<Segment> segments_of(mgl::PrimitiveType mode, int first, int count)
    {
        std::vector<Segment> out;
        switch (mode)
        {
        case mgl::PrimitiveType::Lines:  // "If the number of specified vertices is odd, then the last one is ignored."
            for (int i = 0; i + 1 < count; i += 2) out.push_back({first + i, first + i + 1});
            break;
        case mgl::PrimitiveType::LineStrip:  // "If only one vertex is specified, then no primitive is generated."
        case mgl::PrimitiveType::LineLoop:
            for (int i = 0; i + 1 < count; ++i) out.push_back({first + i, first + i + 1});
            if (mode == mgl::PrimitiveType::LineLoop && count >= 2) out.push_back({first + count - 1, first});
            break;
        default:
            break;
        }
        return out;
    }

    std::vector<Triangle> triangles_of(mgl::PrimitiveType mode, int first, int count)
    {
        std::vector<Triangle> out;
        switch (mode)
        {
        case mgl::PrimitiveType::Triangles:  // "the final k vertices are ignored"
            for (int i = 0; i + 2 < count; i += 3) out.push_back({first + i, first + i + 1, first + i + 2});
            break;
        case mgl::PrimitiveType::TriangleStrip:  // "If fewer than three vertices are specified, no primitive is produced."
            for (int n = 2; n < count; ++n) out.push_back({first + n - 2, first + n - 1, first + n});
            break;
        case mgl::PrimitiveType::TriangleFan:  // "each vertex after the first always replaces vertex B"
            for (int n = 2; n < count; ++n) out.push_back({first, first + n - 1, first + n});
            break;
        default:
            break;
        }
        return out;
    }

    constexpr bool is_line(mgl::PrimitiveType m)
    {
        return m == mgl::PrimitiveType::Lines || m == mgl::PrimitiveType::LineStrip || m == mgl::PrimitiveType::LineLoop;
    }
    constexpr bool is_triangle(mgl::PrimitiveType m)
    {
        return m == mgl::PrimitiveType::Triangles || m == mgl::PrimitiveType::TriangleStrip ||
               m == mgl::PrimitiveType::TriangleFan;
    }

    // ---------------------------------------------------------------------------------------------
    // Geometry helpers
    // ---------------------------------------------------------------------------------------------
    float distance_to_segment(float px, float py, Vec2 a, Vec2 b)
    {
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float len2 = dx * dx + dy * dy;
        float t = len2 > 0.0f ? ((px - a.x) * dx + (py - a.y) * dy) / len2 : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        return std::hypot(px - (a.x + t * dx), py - (a.y + t * dy));
    }

    /// Signed distances from a point to the three edge lines of a triangle (positive = inside).
    std::array<float, 3> edge_margins(const std::array<Vec2, 3>& t, float px, float py)
    {
        const float area2 = (t[1].x - t[0].x) * (t[2].y - t[0].y) - (t[1].y - t[0].y) * (t[2].x - t[0].x);
        std::array<float, 3> m{-1e9f, -1e9f, -1e9f};
        if (std::fabs(area2) < 1e-3f) return m;  // degenerate: covers nothing
        const float sign = area2 > 0.0f ? 1.0f : -1.0f;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const Vec2 a = t[i], b = t[(i + 1) % 3];
            m[i] = sign * ((b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)) / std::hypot(b.x - a.x, b.y - a.y);
        }
        return m;
    }

    float triangle_margin(const std::array<Vec2, 3>& t, float px, float py)
    {
        const auto m = edge_margins(t, px, py);
        return std::min({m[0], m[1], m[2]});
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference: painting what a draw must (and must not) cover
    // ---------------------------------------------------------------------------------------------
    // Per pixel of one cell: kNone = not covered, kCovered = covered, kUnknown = rasterization-dependent.
    using Field = std::vector<std::uint8_t>;
    constexpr std::uint8_t kNone = 0, kCovered = 1, kUnknown = 2;
    constexpr float kTie = 0.05f;  // point squares: pixels closer than this to the square's edge are not compared

    std::size_t at(int x, int y) { return static_cast<std::size_t>(y) * kCell + static_cast<std::size_t>(x); }
    bool inside(int x, int y) { return x >= 0 && y >= 0 && x < kCell && y < kCell; }

    /// Covered beats unknown beats none.
    void raise(Field& f, int x, int y, std::uint8_t s)
    {
        std::uint8_t& d = f[at(x, y)];
        if (s == kCovered || d == kCovered) d = kCovered;
        else if (s == kUnknown || d == kUnknown) d = kUnknown;
    }

    void paint_points(Field& f, const std::vector<Vtx>& verts, const Draw& d)
    {
        for (int i = d.first; i < d.first + d.count; ++i)
        {
            const Vtx& p = verts[static_cast<std::size_t>(i)];
            const float half = p.size * 0.5f;
            // Section 3.4.1: a fragment for each pixel whose center lies inside the square of side `size` around the vertex.
            for (int y = 0; y < kCell; ++y)
                for (int x = 0; x < kCell; ++x)
                {
                    const float margin = std::min(half - std::fabs(static_cast<float>(x) + 0.5f - p.x),
                                                  half - std::fabs(static_cast<float>(y) + 0.5f - p.y));
                    if (margin > kTie) raise(f, x, y, kCovered);
                    else if (margin > -kTie) raise(f, x, y, kUnknown);
                }
        }
    }

    /// `rows` is the effective line width in pixels (an integer, see section 3.5.2.1).
    void paint_segment(Field& f, Vec2 a, Vec2 b, int rows)
    {
        Field seg(static_cast<std::size_t>(kCell * kCell), kNone);
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float half = static_cast<float>(rows) * 0.5f;

        // Everything within `half + 2.5` pixels of the segment depends on the rasterization algorithm
        // (rules 1-4 of section 3.5.1 allow one pixel of deviation from the diamond-exit result).
        for (int y = 0; y < kCell; ++y)
            for (int x = 0; x < kCell; ++x)
                if (distance_to_segment(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, a, b) < half + 2.5f)
                    seg[at(x, y)] = kUnknown;

        // Exactly known case: both endpoints on pixel centers, axis-aligned (any odd width) or 45 degrees (width 1).
        auto on_center = [](float v) { return std::fabs(v - std::floor(v) - 0.5f) < 1e-4f; };
        const bool centers = on_center(a.x) && on_center(a.y) && on_center(b.x) && on_center(b.y);
        const bool axis = std::fabs(dx) < 1e-4f || std::fabs(dy) < 1e-4f;
        const bool diagonal = std::fabs(std::fabs(dx) - std::fabs(dy)) < 1e-4f;
        if (centers && rows % 2 == 1 && (axis || (diagonal && rows == 1)))
        {
            const int x0 = static_cast<int>(std::floor(a.x)), y0 = static_cast<int>(std::floor(a.y));
            const int sx = (dx > 0.0f) - (dx < 0.0f), sy = (dy > 0.0f) - (dy < 0.0f);
            const int n = static_cast<int>(std::lround(std::max(std::fabs(dx), std::fabs(dy))));
            const bool x_major = std::fabs(dx) >= std::fabs(dy);
            const int k = (rows - 1) / 2;
            // Interior fragments only: the first and last pixels are where implementations may differ.
            for (int i = 2; i <= n - 3; ++i)
                for (int m = -(k + 2); m <= k + 2; ++m)
                {
                    const int x = x0 + i * sx + (x_major ? 0 : m), y = y0 + i * sy + (x_major ? m : 0);
                    if (!inside(x, y)) continue;
                    // |m| <= k: the column (row) of `rows` fragments. k+1..k+2: rule 3 / section 3.5.2.1 say
                    // there is exactly one such column per major-axis position, so these must be empty.
                    seg[at(x, y)] = std::abs(m) <= k ? kCovered : kNone;
                }
        }
        for (int y = 0; y < kCell; ++y)
            for (int x = 0; x < kCell; ++x) raise(f, x, y, seg[at(x, y)]);
    }

    /// Merges one triangle into a primitive's field: a pixel is covered if any triangle covers it for sure,
    /// unknown if none does but one might, and uncovered only if every triangle is clearly away from it.
    void paint_triangle(Field& f, const std::array<Vec2, 3>& t)
    {
        for (int y = 0; y < kCell; ++y)
            for (int x = 0; x < kCell; ++x)
            {
                const float m = triangle_margin(t, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                // Edge pixels (|margin| < 1) depend on the fill rule and on subpixel precision.
                if (m >= 1.0f) raise(f, x, y, kCovered);
                else if (m > -1.0f) raise(f, x, y, kUnknown);
            }
    }

    /// Expected content of one cell; pixels with care == 0 are not compared.
    struct Model
    {
        std::vector<demo::Rgba8> color = std::vector<demo::Rgba8>(static_cast<std::size_t>(kCell * kCell));
        std::vector<char> care = std::vector<char>(static_cast<std::size_t>(kCell * kCell), 1);
    };

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }
    demo::Rgba8 rgba8(Color c) { return {to8(c.r), to8(c.g), to8(c.b), to8(c.a)}; }

    int effective_width(float requested, const Limits& lim)
    {
        // Section 3.5.2.1: round to the nearest integer, 0 acts as 1, then clamp to the maximum.
        const int rounded = std::max(1, static_cast<int>(std::lround(requested)));
        return std::min(rounded, std::max(1, static_cast<int>(std::floor(lim.line_hi))));
    }

    Model build_model(const CellData& cell, const Limits& lim)
    {
        Model m;
        for (auto& px : m.color) px = rgba8(kBg);
        for (const Draw& d : cell.draws)
        {
            Field f(static_cast<std::size_t>(kCell * kCell), kNone);
            if (d.mode == mgl::PrimitiveType::Points)
                paint_points(f, cell.verts, d);
            else if (is_line(d.mode))
                for (const Segment s : segments_of(d.mode, d.first, d.count))
                {
                    const Vtx& va = cell.verts[static_cast<std::size_t>(s.a)];
                    const Vtx& vb = cell.verts[static_cast<std::size_t>(s.b)];
                    paint_segment(f, {va.x, va.y}, {vb.x, vb.y}, effective_width(d.line_width, lim));
                }
            else if (is_triangle(d.mode))
            {
                for (const Triangle t : triangles_of(d.mode, d.first, d.count))
                {
                    std::array<Vec2, 3> p{};
                    for (std::size_t i = 0; i < 3; ++i)
                    {
                        const Vtx& v = cell.verts[static_cast<std::size_t>(t[i])];
                        p[i] = {v.x, v.y};
                    }
                    paint_triangle(f, p);
                }
            }

            for (int y = 0; y < kCell; ++y)
                for (int x = 0; x < kCell; ++x)
                {
                    const std::size_t i = at(x, y);
                    if (f[i] == kCovered)
                    {
                        m.care[i] = 1;
                        if (d.point_coord)
                        {
                            // Equations 3.1 / 3.2 for the (single) point of this draw.
                            const Vtx& p = cell.verts[static_cast<std::size_t>(d.first)];
                            const float s = 0.5f + (static_cast<float>(x) + 0.5f - p.x) / p.size;
                            const float t = 0.5f - (static_cast<float>(y) + 0.5f - p.y) / p.size;
                            m.color[i] = {to8(s), to8(t), 0, 255};
                        }
                        else
                            m.color[i] = rgba8(d.color);
                    }
                    else if (f[i] == kUnknown)
                        m.care[i] = 0;
                }
        }
        return m;
    }

    // ---------------------------------------------------------------------------------------------
    // The 16 cells' geometry
    // ---------------------------------------------------------------------------------------------
    constexpr Color kWhite{0.95f, 0.95f, 0.90f, 1.0f};
    constexpr Color kCyan{0.35f, 0.85f, 0.95f, 1.0f};
    constexpr Color kLime{0.60f, 0.90f, 0.30f, 1.0f};
    constexpr Color kOrange{0.95f, 0.55f, 0.15f, 1.0f};
    constexpr Color kPink{0.95f, 0.45f, 0.65f, 1.0f};
    constexpr Color kViolet{0.65f, 0.50f, 0.95f, 1.0f};
    constexpr Color kYellow{0.98f, 0.88f, 0.25f, 1.0f};
    constexpr Color kRed{0.90f, 0.30f, 0.30f, 1.0f};
    constexpr Color kGreen{0.30f, 0.80f, 0.45f, 1.0f};
    constexpr Color kBlue{0.35f, 0.55f, 0.95f, 1.0f};

    // The six vertices of cells 2-4: a staircase whose segments alternate horizontal / vertical, all on pixel centers.
    // LINES draws (0,1) (2,3) (4,5); LINE_STRIP adds the vertical connectors (1,2) (3,4); LINE_LOOP adds (5,0).
    constexpr std::array<Vec2, 6> kStaircase = {{{10.5f, 10.5f}, {50.5f, 10.5f}, {50.5f, 30.5f},
                                                  {30.5f, 30.5f}, {30.5f, 50.5f}, {10.5f, 50.5f}}};

    // Cell 8: widths requested for the four horizontal lines (rows y = kWideLineY) and the vertical one. Only
    // widths up to ALIASED_LINE_WIDTH_RANGE's maximum are requested; 3 and 2.6 need a maximum of at least 3.
    constexpr std::array<float, 4> kWideLineY = {52.5f, 40.5f, 28.5f, 16.5f};
    struct WideLineWidths
    {
        std::array<float, 4> horizontal;
        float vertical;
    };
    WideLineWidths wide_line_widths(const Limits& lim)
    {
        const bool wide = lim.line_hi >= 3.0f;
        return {{wide ? 3.0f : 1.0f, wide ? 2.6f : 1.0f, 1.4f, 0.4f}, wide ? 3.0f : 1.0f};
    }

    std::array<CellData, kCellCount> build_cells(const Limits& lim)
    {
        std::array<CellData, kCellCount> cells;
        auto cell = [&](Cell c) -> CellData& { return cells[static_cast<std::size_t>(index_of(c))]; };
        auto add = [](CellData& c, float x, float y, float size = 1.0f) { c.verts.push_back({x, y, size}); };
        const float hi = std::floor(lim.point_hi);

        // 0: POINTS. Size-1 points at fractional positions light exactly the pixel that contains them;
        //    larger sizes (those the implementation supports) cover the pixels whose centers are inside the square.
        {
            CellData& c = cell(Cell::Points);
            for (const Vec2 p : {Vec2{8.3f, 54.7f}, {14.6f, 54.2f}, {20.5f, 54.5f}, {26.2f, 54.9f}, {40.2f, 54.4f},
                                 {46.8f, 54.6f}, {52.5f, 54.5f}})
                add(c, p.x, p.y, 1.0f);
            // Even sizes are centered on integer coordinates, odd sizes on half-integers, so that the square's edges
            // fall between pixel centers; two entries are deliberately off-grid (still unambiguous).
            for (const Vtx v : {Vtx{8, 38, 2}, {18.5f, 38.5f, 3}, {30, 38, 4}, {42.5f, 38.5f, 5}, {54, 38, 6}, {12, 20, 8},
                                {30.5f, 20.5f, 11}, {52.3f, 20.2f, 3}, {54.3f, 8.7f, 2}})
                if (v.size <= hi) add(c, v.x, v.y, v.size);
            c.draws.push_back({mgl::PrimitiveType::Points, 0, static_cast<int>(c.verts.size()), kWhite});
        }

        // 1: POINTS with gl_PointCoord. The center is 0.25 pixel off the pixel grid on purpose.
        {
            CellData& c = cell(Cell::PointCoord);
            const float size = std::min(hi, 40.0f);
            const float corner = std::floor((static_cast<float>(kCell) - size) / 2.0f);
            add(c, corner + size / 2.0f + 0.25f, corner + size / 2.0f - 0.25f, size);
            c.draws.push_back({mgl::PrimitiveType::Points, 0, 1, kWhite, 1.0f, true});
            c.tolerance = 3;  // gl_PointCoord is mediump
        }

        // 2-4: the same six vertices, three line topologies.
        for (const auto& [which, mode, color] :
             {std::tuple{Cell::Lines, mgl::PrimitiveType::Lines, kCyan}, std::tuple{Cell::LineStrip, mgl::PrimitiveType::LineStrip, kLime},
              std::tuple{Cell::LineLoop, mgl::PrimitiveType::LineLoop, kOrange}})
        {
            CellData& c = cell(which);
            for (const Vec2 v : kStaircase) add(c, v.x, v.y);
            c.draws.push_back({mode, 0, 6, color});
        }

        // 5: TRIANGLES. Eight vertices: two triangles, then two leftover vertices that must be ignored.
        {
            CellData& c = cell(Cell::Triangles);
            for (const Vec2 v : {Vec2{6, 10}, {30, 12}, {18, 40}, {36, 26}, {58, 28}, {50, 58}, {32, 60}, {60, 6}}) add(c, v.x, v.y);
            c.draws.push_back({mgl::PrimitiveType::Triangles, 0, 8, kRed});
        }

        // 6: TRIANGLE_STRIP: a wavy ribbon of four triangles (v0 v1 v2), (v2 v1 v3), (v2 v3 v4), (v4 v3 v5): the strip
        //    swaps the roles of vertices A and B for every other triangle so that all keep the winding of the first.
        //    Vertices alternate between the lower edge (even) and the upper edge (odd) of the ribbon.
        {
            CellData& c = cell(Cell::TriangleStrip);
            for (const Vec2 v : {Vec2{6, 10}, {12, 40}, {26, 16}, {30, 54}, {46, 10}, {52, 42}}) add(c, v.x, v.y);
            c.draws.push_back({mgl::PrimitiveType::TriangleStrip, 0, 6, kGreen});
        }

        // 7: TRIANGLE_FAN: a pivot and five rim vertices spanning 240 degrees; the fan does not close.
        {
            CellData& c = cell(Cell::TriangleFan);
            add(c, 28.0f, 28.0f);
            for (int k = 0; k < 5; ++k)
            {
                const float a = (30.0f + 60.0f * static_cast<float>(k)) * kPi / 180.0f;
                add(c, 28.0f + 24.0f * std::cos(a), 28.0f + 24.0f * std::sin(a));
            }
            c.draws.push_back({mgl::PrimitiveType::TriangleFan, 0, 6, kBlue});
        }

        // 8: wide lines. Endpoints on pixel centers; requested widths 3, 2.6 (rounds to 3), 1.4 and 0.4 (both act as 1).
        {
            CellData& c = cell(Cell::WideLines);
            const WideLineWidths w = wide_line_widths(lim);
            for (int i = 0; i < 4; ++i)
            {
                add(c, 22.5f, kWideLineY[static_cast<std::size_t>(i)]);
                add(c, 58.5f, kWideLineY[static_cast<std::size_t>(i)]);
                c.draws.push_back({mgl::PrimitiveType::Lines, 2 * i, 2, kCyan, w.horizontal[static_cast<std::size_t>(i)]});
            }
            add(c, 10.5f, 8.5f);
            add(c, 10.5f, 58.5f);
            c.draws.push_back({mgl::PrimitiveType::Lines, 8, 2, kPink, w.vertical});
        }

        // 9: the largest point the implementation supports (up to 60 pixels), edges on the pixel grid.
        {
            CellData& c = cell(Cell::MaxPoint);
            const float size = std::min(hi, 60.0f);
            const float corner = std::floor((static_cast<float>(kCell) - size) / 2.0f);
            add(c, corner + size / 2.0f, corner + size / 2.0f, size);
            c.draws.push_back({mgl::PrimitiveType::Points, 0, 1, kOrange});
        }

        // 10: segments of general slope, checked against the rules of section 3.5.1 (not against exact pixels).
        {
            CellData& c = cell(Cell::LineRules);
            for (const Vec2 v : {Vec2{6.5f, 8.5f}, {57.5f, 22.5f}, {12.5f, 58.5f}, {26.5f, 30.5f}, {36.5f, 30.5f}, {58.5f, 52.5f}})
                add(c, v.x, v.y);
            c.draws.push_back({mgl::PrimitiveType::Lines, 0, 6, kYellow});
        }

        // 11: first and count. Six staircase vertices; the first draw starts at vertex 1, the second at vertex 4.
        {
            CellData& c = cell(Cell::FirstAndCount);
            for (const Vec2 v : {Vec2{8.5f, 8.5f}, {24.5f, 8.5f}, {24.5f, 24.5f}, {40.5f, 24.5f}, {40.5f, 40.5f}, {56.5f, 40.5f}})
                add(c, v.x, v.y);
            c.draws.push_back({mgl::PrimitiveType::LineStrip, 1, 3, kLime});
            c.draws.push_back({mgl::PrimitiveType::Lines, 4, 2, kPink});
        }

        // 12: too few vertices. A: LINES with 3 (one segment), B: TRIANGLES with 4 (one triangle), C: LINE_STRIP with 1,
        //     D: TRIANGLE_STRIP with 2, E: TRIANGLE_FAN with 2 (nothing), F: LINE_LOOP with 2 (one segment), and
        //     count = 0 draws of all seven modes (nothing, and no error).
        {
            CellData& c = cell(Cell::Degenerate);
            for (const Vec2 v : {Vec2{6.5f, 56.5f}, {24.5f, 56.5f}, {6.5f, 46.5f},          // A: 0 1 2
                                 Vec2{34, 58}, {58, 58}, {46, 40}, {58, 42},               // B: 3 4 5 6
                                 Vec2{10.5f, 30.5f},                                        // C: 7
                                 Vec2{24.5f, 36}, {30, 22},                                 // D: 8 9
                                 Vec2{40, 34}, {52, 22},                                    // E: 10 11
                                 Vec2{8.5f, 14.5f}, {30.5f, 14.5f}})                        // F: 12 13
                add(c, v.x, v.y);
            c.draws.push_back({mgl::PrimitiveType::Lines, 0, 3, kCyan});
            c.draws.push_back({mgl::PrimitiveType::Triangles, 3, 4, kRed});
            c.draws.push_back({mgl::PrimitiveType::LineStrip, 7, 1, kWhite});
            c.draws.push_back({mgl::PrimitiveType::TriangleStrip, 8, 2, kWhite});
            c.draws.push_back({mgl::PrimitiveType::TriangleFan, 10, 2, kWhite});
            c.draws.push_back({mgl::PrimitiveType::LineLoop, 12, 2, kOrange});
            for (const auto mode : {mgl::PrimitiveType::Points, mgl::PrimitiveType::Lines, mgl::PrimitiveType::LineStrip,
                                    mgl::PrimitiveType::LineLoop, mgl::PrimitiveType::Triangles, mgl::PrimitiveType::TriangleStrip,
                                    mgl::PrimitiveType::TriangleFan})
                c.draws.push_back({mode, 7, 0, kWhite});
        }

        // 13: a disc as a TRIANGLE_FAN: pivot + 24 rim segments; the 25th rim vertex repeats the first to close it.
        {
            CellData& c = cell(Cell::FanDisc);
            add(c, 32.0f, 32.0f);
            for (int k = 0; k <= 24; ++k)
            {
                const float a = static_cast<float>(k % 24) * (2.0f * kPi / 24.0f);
                add(c, 32.0f + 27.0f * std::cos(a), 32.0f + 27.0f * std::sin(a));
            }
            c.draws.push_back({mgl::PrimitiveType::TriangleFan, 0, 26, kViolet});
        }

        // 14: LINE_LOOP octagon; every edge is horizontal, vertical or exactly 45 degrees, on pixel centers.
        {
            CellData& c = cell(Cell::Octagon);
            for (const Vec2 v : {Vec2{21.5f, 8.5f}, {43.5f, 8.5f}, {55.5f, 20.5f}, {55.5f, 42.5f}, {43.5f, 54.5f}, {21.5f, 54.5f},
                                 {9.5f, 42.5f}, {9.5f, 20.5f}})
                add(c, v.x, v.y);
            c.draws.push_back({mgl::PrimitiveType::LineLoop, 0, 8, kLime});
        }

        // 15: a house using all seven primitive types, each in its own draw call from one buffer.
        {
            CellData& c = cell(Cell::Scene);
            const float sun = std::min(hi, 6.0f);
            const int ground = static_cast<int>(c.verts.size());
            for (const Vec2 v : {Vec2{3.5f, 12.5f}, {30.5f, 12.5f}, {60.5f, 12.5f}}) add(c, v.x, v.y);
            const int body = static_cast<int>(c.verts.size());
            for (const Vec2 v : {Vec2{14, 14}, {50, 14}, {50, 38}, {14, 38}}) add(c, v.x, v.y);
            const int roof = static_cast<int>(c.verts.size());
            for (const Vec2 v : {Vec2{10, 38}, {54, 38}, {32, 56}}) add(c, v.x, v.y);
            const int door = static_cast<int>(c.verts.size());
            for (const Vec2 v : {Vec2{28, 14}, {36, 14}, {28, 28}, {36, 28}}) add(c, v.x, v.y);
            const int window = static_cast<int>(c.verts.size());
            for (const Vec2 v : {Vec2{18.5f, 22.5f}, {24.5f, 22.5f}, {24.5f, 30.5f}, {18.5f, 30.5f}}) add(c, v.x, v.y);
            const int chimney = static_cast<int>(c.verts.size());
            for (const Vec2 v : {Vec2{44.5f, 40.5f}, {44.5f, 52.5f}, {50.5f, 40.5f}, {50.5f, 50.5f}}) add(c, v.x, v.y);
            const int sun_index = static_cast<int>(c.verts.size());
            add(c, 54.0f + sun / 2.0f, 56.0f + sun / 2.0f, sun);

            c.draws.push_back({mgl::PrimitiveType::LineStrip, ground, 3, {0.55f, 0.40f, 0.25f, 1.0f}});
            c.draws.push_back({mgl::PrimitiveType::TriangleFan, body, 4, {0.85f, 0.75f, 0.55f, 1.0f}});
            c.draws.push_back({mgl::PrimitiveType::Triangles, roof, 3, {0.80f, 0.25f, 0.20f, 1.0f}});
            c.draws.push_back({mgl::PrimitiveType::TriangleStrip, door, 4, {0.45f, 0.28f, 0.15f, 1.0f}});
            c.draws.push_back({mgl::PrimitiveType::LineLoop, window, 4, {0.40f, 0.70f, 0.95f, 1.0f}});
            c.draws.push_back({mgl::PrimitiveType::Lines, chimney, 4, {0.70f, 0.70f, 0.75f, 1.0f}});
            c.draws.push_back({mgl::PrimitiveType::Points, sun_index, 1, kYellow});
        }
        return cells;
    }

    // ---------------------------------------------------------------------------------------------
    // Framebuffer snapshot
    // ---------------------------------------------------------------------------------------------
    class Image
    {
    public:
        Image(int w, int h) : width_(w), pixels_(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u)
        {
            // RGBA / UNSIGNED_BYTE is the combination every ES implementation must accept (ES 3.0 §4.3.2).
            mgl::glReadPixels(0, 0, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pixels_.data());
        }
        [[nodiscard]] demo::Rgba8 at(Cell c, int x, int y) const
        {
            const Rect r = cell_rect(c);
            const std::size_t i = (static_cast<std::size_t>(r.y + y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(r.x + x)) * 4u;
            return {pixels_[i], pixels_[i + 1], pixels_[i + 2], pixels_[i + 3]};
        }

    private:
        int width_;
        std::vector<std::uint8_t> pixels_;
    };

    bool near_color(demo::Rgba8 got, demo::Rgba8 want, int tolerance)
    {
        for (std::size_t i = 0; i < 4; ++i)
            if (std::abs(int{got[i]} - int{want[i]}) > tolerance) return false;
        return true;
    }

    bool lit(demo::Rgba8 px)  // differs clearly from the cell background
    {
        return !near_color(px, rgba8(kBg), 12);
    }

    // ---------------------------------------------------------------------------------------------
    // The demo
    // ---------------------------------------------------------------------------------------------
    class PrimitiveTopology final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // Pick the arrangement that fits the surface and whose aspect ratio is closest to it. The
            // expected images are exact per-pixel rasterization models of 64x64 cells; a surface that
            // fits none of 256x256, 512x128 and 128x512 cannot show them, so it is skipped rather than shrunk.
            bool fits = false;
            double best = 0.0;
            for (const int columns : {4, 8, 2})
            {
                const int rows = kCellCount / columns;
                if (env.width < kCell * columns || env.height < kCell * rows) continue;
                const double mismatch = std::abs(std::log(static_cast<double>(env.width) / env.height *
                                                          static_cast<double>(rows) / columns));
                if (!fits || mismatch < best)
                {
                    fits = true;
                    best = mismatch;
                    g_columns = columns;
                    g_rows = rows;
                }
            }
            if (!fits)
                throw demo::Skip(std::format("the 16 cells of 64x64 px need a 256x256, 512x128 or 128x512 surface (got {}x{})",
                                             env.width, env.height));

            // Two programs share the vertex shader: flat color, and gl_PointCoord visualization.
            solid_ = demo::build_program(env, "shaders/primitive.vert", "shaders/solid.frag");
            point_coord_ = demo::build_program(env, "shaders/primitive.vert", "shaders/point_coord.frag");
            u_color_ = mgl::glGetUniformLocation(solid_, "u_color");
            if (u_color_.value < 0) throw demo::Failure("solid.frag: u_color is not active");

            // Implementation-dependent limits (ES 3.0 sections 3.4, 3.5; Table 6.x): nothing is assumed beyond 1.
            std::array<GLfloat, 2> range{};
            mgl::glGetFloatv(mgl::GetParameter::AliasedPointSizeRange, range.data());
            limits_.point_lo = range[0];
            limits_.point_hi = range[1];
            mgl::glGetFloatv(mgl::GetParameter::AliasedLineWidthRange, range.data());
            limits_.line_lo = range[0];
            limits_.line_hi = range[1];
            mgl::glGetFloatv(mgl::GetParameter::LineWidth, &initial_line_width_);
            std::cout << std::format("     ALIASED_POINT_SIZE_RANGE = [{}, {}], ALIASED_LINE_WIDTH_RANGE = [{}, {}]\n",
                                     limits_.point_lo, limits_.point_hi, limits_.line_lo, limits_.line_hi);

            cells_ = build_cells(limits_);

            // All cells' vertices in one buffer, converted to clip space for a 64x64 viewport:
            // x_ndc = 2 * x_window / 64 - 1 (the viewport transform of section 2.13.1 inverted).
            std::vector<GpuVertex> gpu;
            for (CellData& c : cells_)
            {
                c.base = static_cast<int>(gpu.size());
                for (const Vtx& v : c.verts)
                    gpu.push_back({v.x / 32.0f - 1.0f, v.y / 32.0f - 1.0f, v.size});
            }
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const GpuVertex>(gpu), mgl::BufferUsage::StaticDraw);

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            const mgl::AttribLocation a_position{0u};    // layout(location = 0) in primitive.vert
            const mgl::AttribLocation a_point_size{1u};  // layout(location = 1)
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 2, mgl::DataType::Float, GL_FALSE, sizeof(GpuVertex),
                                       demo::buffer_offset(offsetof(GpuVertex, x)));
            mgl::glEnableVertexAttribArray(a_point_size);
            mgl::glVertexAttribPointer(a_point_size, 1, mgl::DataType::Float, GL_FALSE, sizeof(GpuVertex),
                                       demo::buffer_offset(offsetof(GpuVertex, size)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            // Fixed-function state the drawing relies on (all of it is the initial state except DITHER).
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::CullFace);   // initial state; the demo must not depend on triangle winding
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::StencilTest);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::Dither);     // flat colors are then written exactly
            mgl::glLineWidth(1.0f);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kSurface.r, kSurface.g, kSurface.b, kSurface.a);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glBindVertexArray(vao_);
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (int i = 0; i < kCellCount; ++i)
            {
                const Rect r = cell_rect(static_cast<Cell>(i));
                mgl::glViewport(r.x, r.y, r.w, r.h);
                mgl::glScissor(r.x, r.y, r.w, r.h);  // keeps big points and wide lines inside their cell
                mgl::glClearColor(kBg.r, kBg.g, kBg.b, kBg.a);
                mgl::glClear(mgl::ClearBufferBit::Color);
                const CellData& c = cells_[static_cast<std::size_t>(i)];
                for (const Draw& d : c.draws)
                {
                    mgl::glUseProgram(d.point_coord ? point_coord_ : solid_);
                    if (!d.point_coord) mgl::glUniform4f(u_color_, d.color.r, d.color.g, d.color.b, d.color.a);
                    mgl::glLineWidth(d.line_width);
                    mgl::glDrawArrays(d.mode, c.base + d.first, d.count);
                }
                mgl::glLineWidth(1.0f);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            const Image image(env.width, env.height);
            for (int i = 0; i < kCellCount; ++i) models_[static_cast<std::size_t>(i)] = build_model(cells_[static_cast<std::size_t>(i)], limits_);

            check_points(check, image);
            check_point_coord(check, image);
            check_line_topologies(check, image);
            check_triangle_topologies(check, image);
            check_wide_lines(check, image);
            check_max_point(check, image);
            check_line_rules(check, image);
            check_first_and_count(check, image);
            check_degenerate(check, image);
            check_fan_disc(check, image);
            check_octagon(check, image);
            check_scene(check, image);

            check_limits_and_errors(check);
            render(env, 0);  // the error checks above issued extra draws; restore the canonical image
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteProgram(solid_);
            mgl::glDeleteProgram(point_coord_);
        }

    private:
        // =========================================================================================
        // Helpers over the CPU reference
        // =========================================================================================
        [[nodiscard]] const CellData& data(Cell c) const { return cells_[static_cast<std::size_t>(index_of(c))]; }
        [[nodiscard]] const Model& model(Cell c) const { return models_[static_cast<std::size_t>(index_of(c))]; }

        /// Compares the cared-about pixels of `region` with the CPU model; one check.
        void compare(demo::Check& check, const Image& image, Cell c, Rect region, std::string_view what) const
        {
            const Model& m = model(c);
            const int tolerance = data(c).tolerance;
            int compared = 0, bad = 0;
            std::string first_bad;
            for (int y = std::max(region.y, 0); y < std::min(region.y + region.h, kCell); ++y)
                for (int x = std::max(region.x, 0); x < std::min(region.x + region.w, kCell); ++x)
                {
                    if (!m.care[at(x, y)]) continue;
                    ++compared;
                    const demo::Rgba8 got = image.at(c, x, y), want = m.color[at(x, y)];
                    if (!near_color(got, want, tolerance) && bad++ == 0)
                        first_bad = std::format("first at cell ({},{}): got {} expected {}", x, y, demo::format_rgba(got),
                                                demo::format_rgba(want));
                }
            check.expect(compared > 0 && bad == 0, what,
                         bad == 0 ? std::format("{} pixels compared", compared)
                                  : std::format("{} of {} pixels differ; {}", bad, compared, first_bad));
        }

        void compare(demo::Check& check, const Image& image, Cell c, std::string_view what) const
        {
            compare(check, image, c, {0, 0, kCell, kCell}, what);
        }

        /// Named probes at window positions inside a cell. A probe is only taken where the CPU model has a definite
        /// expectation of the requested kind (lit = some primitive's color, clear = the background); the image must agree.
        struct ProbeResult
        {
            int probed = 0;
            std::string failures;
        };
        enum class Want { Lit, Clear };

        void probe(const Image& image, Cell c, Vec2 p, Want want, ProbeResult& out, std::string_view name) const
        {
            const int x = static_cast<int>(std::floor(p.x)), y = static_cast<int>(std::floor(p.y));
            if (!inside(x, y) || !model(c).care[at(x, y)]) return;
            const demo::Rgba8 expected = model(c).color[at(x, y)];
            const bool expect_clear = near_color(expected, rgba8(kBg), 0);
            if (expect_clear != (want == Want::Clear)) return;
            ++out.probed;
            if (!near_color(image.at(c, x, y), expected, data(c).tolerance))
                out.failures += std::format(" {}@({},{})", name, x, y);
        }

        void report(demo::Check& check, const ProbeResult& r, std::string_view what) const
        {
            check.expect(r.probed > 0 && r.failures.empty(), what,
                         r.failures.empty() ? std::format("{} probes", r.probed) : "mismatch:" + r.failures);
        }

        [[nodiscard]] static Vec2 position(const CellData& c, int i)
        {
            return {c.verts[static_cast<std::size_t>(i)].x, c.verts[static_cast<std::size_t>(i)].y};
        }

        // =========================================================================================
        // Cells 0, 1, 9: points
        // =========================================================================================
        void check_points(demo::Check& check, const Image& image) const
        {
            const CellData& c = data(Cell::Points);
            compare(check, image, Cell::Points,
                    "POINTS (cell 0): every point lights exactly the pixels whose centers lie inside its square (sizes 1.." +
                        std::to_string(static_cast<int>(std::min(std::floor(limits_.point_hi), 11.0f))) + ")");

            // A size-1 point lights exactly the one pixel that contains its window coordinates (section 3.4.1).
            int points = 0;
            bool all_ok = true;
            for (const Vtx& v : c.verts)
            {
                if (v.size != 1.0f) continue;
                ++points;
                const int px = static_cast<int>(std::floor(v.x)), py = static_cast<int>(std::floor(v.y));
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        if (lit(image.at(Cell::Points, px + dx, py + dy)) != (dx == 0 && dy == 0)) all_ok = false;
            }
            check.expect(points > 0 && all_ok,
                         std::format("POINTS: each of the {} size-1 points at a fractional position lights its own pixel and none of its 8 neighbours", points));
        }

        void check_point_coord(demo::Check& check, const Image& image) const
        {
            const CellData& c = data(Cell::PointCoord);
            const float size = c.verts[0].size;
            compare(check, image, Cell::PointCoord,
                    std::format("gl_PointCoord (cell 1): a size-{} point shows (s,t) = (1/2 + (x_f+1/2-x_w)/size, 1/2 - (y_f+1/2-y_w)/size) (eq. 3.1, 3.2; +-3)", size));
            if (size >= 4.0f)
            {
                // t is 0 at the TOP of the point and 1 at the bottom; s grows to the right.
                const int left = static_cast<int>(std::floor((static_cast<float>(kCell) - size) / 2.0f));  // lower-left pixel
                const int bottom = left;
                const int right = left + static_cast<int>(size) - 1, top = bottom + static_cast<int>(size) - 1;
                const demo::Rgba8 tl = image.at(Cell::PointCoord, left, top), br = image.at(Cell::PointCoord, right, bottom);
                const demo::Rgba8 tr = image.at(Cell::PointCoord, right, top), bl = image.at(Cell::PointCoord, left, bottom);
                check.expect(tl[1] < bl[1] && tr[1] < br[1] && tl[0] < tr[0] && bl[0] < br[0],
                             "gl_PointCoord: t grows from the top edge to the bottom edge of the point, s from the left edge to the right",
                             std::format("TL={} TR={} BL={} BR={}", demo::format_rgba(tl), demo::format_rgba(tr),
                                         demo::format_rgba(bl), demo::format_rgba(br)));
            }
        }

        void check_max_point(demo::Check& check, const Image& image) const
        {
            const float size = data(Cell::MaxPoint).verts[0].size;
            compare(check, image, Cell::MaxPoint,
                    std::format("largest point (cell 9): a size-{} point (min(ALIASED_POINT_SIZE_RANGE max, 60)) covers its square", size));
            int count = 0;
            for (int y = 0; y < kCell; ++y)
                for (int x = 0; x < kCell; ++x)
                    if (lit(image.at(Cell::MaxPoint, x, y))) ++count;
            const int expected = static_cast<int>(size) * static_cast<int>(size);
            check.expect_eq(count, expected, std::format("largest point: exactly size^2 = {} pixels are lit", expected));
        }

        // =========================================================================================
        // Cells 2-4, 11, 14: line topologies
        // =========================================================================================
        void check_segment_probes(demo::Check& check, const Image& image, Cell c, std::string_view label) const
        {
            const CellData& d = data(c);
            std::set<std::pair<int, int>> drawn, others;
            auto key = [](Segment s) { return std::pair{std::min(s.a, s.b), std::max(s.a, s.b)}; };
            for (const Draw& dr : d.draws)
                for (const Segment s : segments_of(dr.mode, dr.first, dr.count)) drawn.insert(key(s));
            const int n = static_cast<int>(d.verts.size());
            for (const auto mode : {mgl::PrimitiveType::Lines, mgl::PrimitiveType::LineStrip, mgl::PrimitiveType::LineLoop})
                for (const Segment s : segments_of(mode, 0, n))
                    if (!drawn.contains(key(s))) others.insert(key(s));

            ProbeResult lit_probes, clear_probes;
            for (const auto& [a, b] : drawn)
            {
                const Vec2 pa = position(d, a), pb = position(d, b);
                probe(image, c, {(pa.x + pb.x) / 2.0f, (pa.y + pb.y) / 2.0f}, Want::Lit, lit_probes, std::format("v{}-v{}", a, b));
            }
            for (const auto& [a, b] : others)
            {
                const Vec2 pa = position(d, a), pb = position(d, b);
                probe(image, c, {(pa.x + pb.x) / 2.0f, (pa.y + pb.y) / 2.0f}, Want::Clear, clear_probes, std::format("v{}-v{}", a, b));
            }
            report(check, lit_probes, std::format("{}: the midpoint of each of the {} segments this topology builds is lit", label, drawn.size()));
            if (clear_probes.probed > 0)  // (no such segment may exist, or all of them may overlap drawn ones)
                report(check, clear_probes,
                       std::format("{}: the midpoints of the segments only another line topology would add are clear", label));
        }

        void expect_pixels(demo::Check& check, const Image& image, Cell c, std::initializer_list<std::array<int, 2>> px, bool want_lit,
                           std::string_view what) const
        {
            bool ok = true;
            std::string detail;
            for (const auto& p : px)
            {
                const demo::Rgba8 got = image.at(c, p[0], p[1]);
                if (lit(got) != want_lit)
                {
                    ok = false;
                    detail += std::format(" ({},{})={}", p[0], p[1], demo::format_rgba(got));
                }
            }
            check.expect(ok, what, ok ? "" : "wrong:" + detail);
        }

        void check_line_topologies(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::Lines, "LINES (cell 2): 6 vertices = 3 separate segments; every pixel matches the CPU reference");
            compare(check, image, Cell::LineStrip, "LINE_STRIP (cell 3): the same 6 vertices = 5 connected segments");
            compare(check, image, Cell::LineLoop, "LINE_LOOP (cell 4): the same 6 vertices = 6 segments, closed");

            // The vertical connectors v1->v2 (x=50) and v3->v4 (x=30) and the closing segment v5->v0 (x=10).
            expect_pixels(check, image, Cell::Lines, {{30, 10}, {40, 30}, {20, 50}}, true,
                          "LINES: the three separate horizontal segments are drawn (v0-v1, v2-v3, v4-v5)");
            expect_pixels(check, image, Cell::Lines, {{50, 20}, {30, 40}, {10, 30}}, false,
                          "LINES: the connectors v1-v2, v3-v4 and the closing v5-v0 are NOT drawn (vertex pairs, not a path)");
            expect_pixels(check, image, Cell::LineStrip, {{50, 20}, {30, 40}}, true,
                          "LINE_STRIP: each vertex ends one segment and starts the next (connectors v1-v2, v3-v4 are drawn)");
            expect_pixels(check, image, Cell::LineStrip, {{10, 30}}, false,
                          "LINE_STRIP: the path is open - no segment from the last vertex back to the first");
            expect_pixels(check, image, Cell::LineLoop, {{50, 20}, {30, 40}, {10, 30}}, true,
                          "LINE_LOOP: connectors AND the closing segment v5-v0 are drawn");

            check_segment_probes(check, image, Cell::Lines, "LINES");
            check_segment_probes(check, image, Cell::LineStrip, "LINE_STRIP");
            check_segment_probes(check, image, Cell::LineLoop, "LINE_LOOP");
        }

        void check_first_and_count(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::FirstAndCount,
                    "glDrawArrays(first, count) (cell 11): LINE_STRIP first=1 count=3 and LINES first=4 count=2 draw exactly their vertex ranges");
            check_segment_probes(check, image, Cell::FirstAndCount, "first/count");
            // Vertices 0 and 5 (or the segment 0-1) are outside the first draw's range: the bottom segment at y=8 is absent.
            expect_pixels(check, image, Cell::FirstAndCount, {{16, 8}, {40, 32}}, false,
                          "first/count: segment v0-v1 (before `first`) and the connector v3-v4 (never drawn) are absent");
            expect_pixels(check, image, Cell::FirstAndCount, {{24, 16}, {32, 24}, {48, 40}}, true,
                          "first/count: v1-v2, v2-v3 (strip from vertex 1) and v4-v5 (LINES from vertex 4) are drawn");
        }

        void check_octagon(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::Octagon, "LINE_LOOP octagon (cell 14): axis-aligned and 45-degree edges are exactly one pixel wide");
            check_segment_probes(check, image, Cell::Octagon, "octagon");
            expect_pixels(check, image, Cell::Octagon, {{15, 14}, {32, 8}, {49, 14}, {55, 31}, {49, 48}, {32, 54}, {15, 48}, {9, 31}}, true,
                          "octagon: the midpoint of each of the 8 edges, including the closing edge (9,20)-(21,8), is lit");
        }

        // =========================================================================================
        // Cell 8: wide lines
        // =========================================================================================
        void check_wide_lines(demo::Check& check, const Image& image) const
        {
            const WideLineWidths w = wide_line_widths(limits_);
            compare(check, image, Cell::WideLines,
                    std::format("wide lines (cell 8): glLineWidth({}, {}, {}, {}) with ALIASED_LINE_WIDTH_RANGE max {} -> {}, {}, {}, {} rows; vertical line width {}",
                                w.horizontal[0], w.horizontal[1], w.horizontal[2], w.horizontal[3], limits_.line_hi,
                                effective_width(w.horizontal[0], limits_), effective_width(w.horizontal[1], limits_),
                                effective_width(w.horizontal[2], limits_), effective_width(w.horizontal[3], limits_), w.vertical));
            // Count lit pixels in a column across each horizontal line, and in a row across the vertical one.
            for (std::size_t i = 0; i < 4; ++i)
            {
                const int rows = effective_width(w.horizontal[i], limits_);
                int count = 0;
                const int cy = static_cast<int>(kWideLineY[i]);
                for (int y = cy - 6; y <= cy + 6; ++y)
                    if (lit(image.at(Cell::WideLines, 40, y))) ++count;
                check.expect_eq(count, rows,
                                std::format("glLineWidth({}): rounded to the nearest integer (0 acts as 1), a vertical cut through the line crosses {} lit pixels (section 3.5.2.1)",
                                            w.horizontal[i], rows));
            }
            const int vrows = effective_width(w.vertical, limits_);
            int vcount = 0;
            for (int x = 3; x <= 18; ++x)
                if (lit(image.at(Cell::WideLines, x, 30))) ++vcount;
            check.expect_eq(vcount, vrows, std::format("glLineWidth({}) on a vertical (y-major) line: a horizontal cut crosses {} lit pixels",
                                                       w.vertical, vrows));
        }

        // =========================================================================================
        // Cell 10: rules of section 3.5.1 for general slopes
        // =========================================================================================
        void check_line_rules(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::LineRules,
                    "line rules (cell 10): no pixel is lit farther than 2.5 pixels from any of the three segments");
            const CellData& d = data(Cell::LineRules);
            const char* names[3] = {"shallow (slope 0.27, x-major)", "steep (slope -2, y-major)", "45 degrees"};
            for (int s = 0; s < 3; ++s)
            {
                const Vec2 a = position(d, 2 * s), b = position(d, 2 * s + 1);
                const bool x_major = std::fabs(b.x - a.x) >= std::fabs(b.y - a.y);
                // u = major axis, v = minor axis; the line is v(u) = v_a + slope * (u - u_a).
                const float ua = x_major ? a.x : a.y, ub = x_major ? b.x : b.y;
                const float va = x_major ? a.y : a.x, vb = x_major ? b.y : b.x;
                const float slope = (vb - va) / (ub - ua);
                const float lo = std::min(ua, ub), hi = std::max(ua, ub);
                int columns = 0, missing = 0, duplicated = 0, max_dev = 0;
                for (int u = static_cast<int>(std::ceil(lo + 2.0f - 0.5f)); static_cast<float>(u) + 0.5f <= hi - 2.0f; ++u)
                {
                    ++columns;
                    const float v_line = va + slope * (static_cast<float>(u) + 0.5f - ua);
                    const int v_ideal = static_cast<int>(std::floor(v_line));  // pixel whose center is nearest to the line
                    int found = 0;
                    for (int v = std::max(v_ideal - 3, 0); v <= std::min(v_ideal + 3, kCell - 1); ++v)
                    {
                        if (!lit(image.at(Cell::LineRules, x_major ? u : v, x_major ? v : u))) continue;
                        ++found;
                        max_dev = std::max(max_dev, std::abs(v - v_ideal));
                    }
                    if (found == 0) ++missing;
                    if (found > 1) ++duplicated;
                }
                // Rule 3: at most one fragment per column (row for y-major). Rule 2: the total may differ from the
                // diamond-exit count by one, so at most one interior column may be empty. Rule 1: each fragment is
                // within one unit in x and y of an ideal fragment, and the ideal row changes by at most one between
                // neighbouring columns, so it is at most 2 rows from the line's own ideal row.
                check.expect(columns > 0 && duplicated == 0 && missing <= 1 && max_dev <= 2,
                             std::format("line rules: {} segment: at most one fragment per interior column and none farther from the line than rules 1-3 allow (section 3.5.1)", names[s]),
                             std::format("{} interior columns, {} empty, {} with several fragments, max deviation {} px", columns,
                                         missing, duplicated, max_dev));
            }
        }

        // =========================================================================================
        // Cells 5-7, 13: triangle topologies
        // =========================================================================================
        void check_triangle_probes(demo::Check& check, const Image& image, Cell c, std::string_view label) const
        {
            const CellData& d = data(c);
            std::set<std::array<int, 3>> drawn, others;
            auto key = [](Triangle t) { std::sort(t.begin(), t.end()); return t; };
            std::vector<Triangle> drawn_list;
            for (const Draw& dr : d.draws)
                for (const Triangle t : triangles_of(dr.mode, dr.first, dr.count))
                {
                    drawn.insert(key(t));
                    drawn_list.push_back(t);
                }
            const int n = static_cast<int>(d.verts.size());
            for (const auto mode : {mgl::PrimitiveType::Triangles, mgl::PrimitiveType::TriangleStrip, mgl::PrimitiveType::TriangleFan})
                for (const Triangle t : triangles_of(mode, 0, n))
                    if (!drawn.contains(key(t))) others.insert(key(t));

            auto centroid = [&](Triangle t) {
                Vec2 s{0, 0};
                for (const int i : t)
                {
                    s.x += position(d, i).x / 3.0f;
                    s.y += position(d, i).y / 3.0f;
                }
                return s;
            };
            ProbeResult lit_probes, clear_probes, edge_probes;
            for (const Triangle t : drawn_list) probe(image, c, centroid(t), Want::Lit, lit_probes, std::format("({},{},{})", t[0], t[1], t[2]));
            for (const Triangle t : others) probe(image, c, centroid(t), Want::Clear, clear_probes, std::format("({},{},{})", t[0], t[1], t[2]));

            // No holes along edges shared by two triangles of the same primitive: probe points on the shared edge that
            // are at least 3 pixels away from every other edge of both triangles (so the whole neighbourhood is inside
            // the union of the two).
            for (std::size_t i = 0; i < drawn_list.size(); ++i)
                for (std::size_t j = i + 1; j < drawn_list.size(); ++j)
                {
                    std::vector<int> shared;
                    for (const int a : drawn_list[i])
                        if (std::find(drawn_list[j].begin(), drawn_list[j].end(), a) != drawn_list[j].end()) shared.push_back(a);
                    if (shared.size() != 2) continue;
                    const Vec2 pa = position(d, shared[0]), pb = position(d, shared[1]);
                    for (const float t : {0.3f, 0.5f, 0.7f})
                    {
                        const Vec2 p{pa.x + t * (pb.x - pa.x), pa.y + t * (pb.y - pa.y)};
                        bool far_from_boundary = true;
                        for (const Triangle& tri : {drawn_list[i], drawn_list[j]})
                        {
                            std::array<Vec2, 3> corners{};
                            for (std::size_t k = 0; k < 3; ++k) corners[k] = position(d, tri[k]);
                            const auto m = edge_margins(corners, p.x, p.y);
                            for (std::size_t k = 0; k < 3; ++k)
                            {
                                const bool is_shared = (tri[k] == shared[0] || tri[k] == shared[1]) &&
                                                       (tri[(k + 1) % 3] == shared[0] || tri[(k + 1) % 3] == shared[1]);
                                if (!is_shared && m[k] < 3.0f) far_from_boundary = false;
                            }
                        }
                        if (!far_from_boundary) continue;
                        const int px = static_cast<int>(std::floor(p.x)), py = static_cast<int>(std::floor(p.y));
                        ++edge_probes.probed;
                        if (!near_color(image.at(c, px, py), rgba8(d.draws.front().color), 1))
                            edge_probes.failures += std::format(" edge v{}-v{} t={} @({},{})", shared[0], shared[1], t, px, py);
                    }
                }

            report(check, lit_probes, std::format("{}: the centroid of each of the {} triangles this topology builds is covered", label, drawn_list.size()));
            if (clear_probes.probed > 0)  // strips and fans usually overlap their alternatives: then there is nothing to probe
                report(check, clear_probes, std::format("{}: the centroids of the triangles only another topology would build from these vertices are NOT covered", label));
            if (edge_probes.probed > 0)
                report(check, edge_probes, std::format("{}: no hole along edges shared by consecutive triangles (probes on the shared edges)", label));
        }

        void check_triangle_topologies(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::Triangles, "TRIANGLES (cell 5): 8 vertices = 2 separate triangles + 2 ignored vertices");
            compare(check, image, Cell::TriangleStrip, "TRIANGLE_STRIP (cell 6): 6 vertices = 4 triangles, each reusing the previous two vertices");
            compare(check, image, Cell::TriangleFan, "TRIANGLE_FAN (cell 7): pivot + 5 rim vertices = 4 triangles sharing v0");

            check_triangle_probes(check, image, Cell::Triangles, "TRIANGLES");
            check_triangle_probes(check, image, Cell::TriangleStrip, "TRIANGLE_STRIP");
            check_triangle_probes(check, image, Cell::TriangleFan, "TRIANGLE_FAN");

            // TRIANGLES: the two leftover vertices v6=(32,60) and v7=(60,6) start no triangle.
            const Model& m = model(Cell::Triangles);
            bool leftover_clear = true;
            for (const Rect r : {Rect{27, 55, 11, 9}, Rect{55, 0, 9, 11}})
                for (int y = r.y; y < r.y + r.h; ++y)
                    for (int x = r.x; x < r.x + r.w; ++x)
                        if (m.care[at(x, y)] && lit(image.at(Cell::Triangles, x, y))) leftover_clear = false;
            check.expect(leftover_clear, "TRIANGLES: the 2 leftover vertices (count = 3n + 2) produce nothing around them");

            // TRIANGLE_FAN: the sector between the last and the first rim vertex stays open.
            expect_pixels(check, image, Cell::TriangleFan, {{40, 21}, {44, 18}}, false,
                          "TRIANGLE_FAN: the wedge between the last rim vertex and the first is clear (a fan does not close)");
            expect_pixels(check, image, Cell::TriangleFan, {{20, 36}, {34, 44}, {14, 26}, {24, 12}}, true,
                          "TRIANGLE_FAN: pixels inside each of the four triangles (angles 69, 135, 188, 256 degrees around the pivot) are covered");
        }

        void check_fan_disc(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::FanDisc, "fan disc (cell 13): 24 triangles around a pivot; every pixel clearly inside / outside matches");
            // Ring of probes at radius 22 (inside) and 31 (outside) every 7.5 degrees - independent of the vertex angles.
            ProbeResult inside_probes, outside_probes;
            for (int k = 0; k < 48; ++k)
            {
                const float a = static_cast<float>(k) * kPi / 24.0f;
                probe(image, Cell::FanDisc, {32.0f + 22.0f * std::cos(a), 32.0f + 22.0f * std::sin(a)}, Want::Lit, inside_probes, std::to_string(k));
                probe(image, Cell::FanDisc, {32.0f + 31.0f * std::cos(a), 32.0f + 31.0f * std::sin(a)}, Want::Clear, outside_probes, std::to_string(k));
            }
            report(check, inside_probes, "fan disc: 48 probes on a ring of radius 22 are all covered (the last rim vertex repeats the first and closes the fan)");
            report(check, outside_probes, "fan disc: 48 probes on a ring of radius 31 (outside the radius-27 rim) are all clear");
            check_triangle_probes(check, image, Cell::FanDisc, "fan disc");
        }

        // =========================================================================================
        // Cell 12: too few vertices
        // =========================================================================================
        void check_degenerate(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::Degenerate, "too few vertices (cell 12): the whole cell matches the CPU reference");
            compare(check, image, Cell::Degenerate, {0, 40, 32, 24},
                    "LINES with count 3: one segment is drawn, the third vertex is ignored");
            compare(check, image, Cell::Degenerate, {32, 36, 32, 28},
                    "TRIANGLES with count 4: one triangle is drawn, the fourth vertex is ignored");
            compare(check, image, Cell::Degenerate, {0, 22, 24, 16},
                    "LINE_STRIP with count 1 draws nothing (also: count 0 draws of all 7 modes draw nothing)");
            compare(check, image, Cell::Degenerate, {16, 18, 48, 20},
                    "TRIANGLE_STRIP / TRIANGLE_FAN with count 2 draw nothing (no triangle can be formed)");
            compare(check, image, Cell::Degenerate, {0, 8, 40, 12},
                    "LINE_LOOP with count 2: the segment is drawn (there and back)");
            expect_pixels(check, image, Cell::Degenerate, {{15, 56}, {20, 14}}, true, "degenerate: the control segments are visible");
            expect_pixels(check, image, Cell::Degenerate, {{10, 30}, {27, 29}, {46, 28}, {6, 46}, {15, 51}, {6, 50}, {58, 42}}, false,
                          "degenerate: nothing at the lone vertex, between the 2-vertex strips/fans, or around the ignored 3rd and 4th vertices (no v1-v2 / v2-v0 segment)");
        }

        void check_scene(demo::Check& check, const Image& image) const
        {
            compare(check, image, Cell::Scene,
                    "scene (cell 15): POINTS, LINES, LINE_STRIP, LINE_LOOP, TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN in one image match the CPU reference");
        }

        // =========================================================================================
        // State queries and errors
        // =========================================================================================
        void check_limits_and_errors(demo::Check& check)
        {
            check.expect(limits_.point_lo <= 1.0f && limits_.point_hi >= 1.0f && limits_.point_lo <= limits_.point_hi,
                         "ALIASED_POINT_SIZE_RANGE contains 1 (only size 1 is guaranteed, ES 3.0 §3.4)",
                         std::format("[{}, {}]", limits_.point_lo, limits_.point_hi));
            check.expect(limits_.line_lo <= 1.0f && limits_.line_hi >= 1.0f && limits_.line_lo <= limits_.line_hi,
                         "ALIASED_LINE_WIDTH_RANGE contains 1 (only width 1 is guaranteed, ES 3.0 §3.5)",
                         std::format("[{}, {}]", limits_.line_lo, limits_.line_hi));
            check.expect_near(initial_line_width_, 1.0, 0.0, "initial GL_LINE_WIDTH = 1.0 (ES 3.0 §3.5.3)");

            // glLineWidth(width <= 0) -> INVALID_VALUE and the width is unchanged.
            check.expect_no_gl_error("no GL error before the negative tests");
            mgl::glLineWidth(0.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glLineWidth(0) -> INVALID_VALUE (ES 3.0 §3.5)");
            mgl::glLineWidth(-2.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glLineWidth(-2) -> INVALID_VALUE (ES 3.0 §3.5)");
            GLfloat width = 0.0f;
            mgl::glGetFloatv(mgl::GetParameter::LineWidth, &width);
            check.expect_near(width, 1.0, 0.0, "a rejected glLineWidth leaves GL_LINE_WIDTH unchanged");
            if (limits_.line_hi >= 3.0f)
            {
                mgl::glLineWidth(3.0f);
                mgl::glGetFloatv(mgl::GetParameter::LineWidth, &width);
                check.expect_near(width, 3.0, 0.0, "glLineWidth(3) is returned by GL_LINE_WIDTH when the width range allows it");
                mgl::glLineWidth(1.0f);
            }

            // ES 3.0 §2.5: "If a negative number is provided where an argument of type sizei ... is specified, the
            // error INVALID_VALUE is generated" - count is a sizei. §2.9.3: with count == 0 no elements are transferred.
            mgl::glBindVertexArray(vao_);
            mgl::glUseProgram(solid_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, -3);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawArrays with a negative count -> INVALID_VALUE (ES 3.0 §2.5: negative sizei)");
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 0);
            check.expect_no_gl_error("glDrawArrays with count 0 is legal and draws nothing");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        // ---------------------------------------------------------------------------------------
        mgl::ProgramId solid_{}, point_coord_{};
        mgl::UniformLocation u_color_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};
        Limits limits_{};
        GLfloat initial_line_width_ = -1.0f;
        std::array<CellData, kCellCount> cells_{};
        std::array<Model, kCellCount> models_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-primitive-topology",
        .title = "Primitive topology",
        .min_es = {3, 0},
        .summary = "All seven glDrawArrays primitive types on shared vertex data: how vertices form points, lines and triangles, "
                   "point size / gl_PointCoord, line width, and the first/count arguments, checked against a CPU reference.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<PrimitiveTopology>(argc, argv, info);
}
