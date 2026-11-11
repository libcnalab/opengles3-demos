// es32-tessellation-domains
//
// The fixed-function tessellation primitive generator of OpenGL ES 3.2 (section 11.2.2): the three domains
// (triangles, quads, isolines), the three edge spacings (equal_spacing, fractional_odd_spacing, fractional_even_spacing),
// the winding order (cw / ccw) and point_mode, driven by a tessellation control shader that writes gl_TessLevelOuter /
// gl_TessLevelInner, per-patch outputs, gl_InvocationID and barrier(), and by glPatchParameteri(PATCH_VERTICES).
//
// What the 256 x 256 image shows: nine cells, one tessellated patch each, bent into a bump so that the 3D structure is
// visible. Rows are the domains (triangles, quads, isolines), columns the spacings (equal, fractional odd, fractional
// even). A cell is drawn in three passes: the filled triangles (colour by gl_FrontFacing: teal = counter-clockwise,
// orange = clockwise), a wireframe made by a geometry shader (core ES has no polygon mode) and the generated vertices as
// white dots (point_mode, captured by transform feedback and re-drawn as fat points, because core ES 3.2 gives points of
// tessellation and geometry shaders a size of 1.0, section 13.5).
//
// Validation, the heart of the demo: transform feedback captures the output of the tessellation evaluation shader
// (ES 3.2 section 12.1: the TES is the last vertex processing stage) and the values are compared with a CPU
// implementation of the subdivision rules of section 11.2.2:
//   * equal_spacing: the exact vertex set (gl_TessCoord) and the exact number of vertices and primitives, derived
//     independently from the text (rounding of the levels, the "1 + epsilon" rule, concentric inner triangles, the inner
//     rectangle mesh of the quad, the isolines) for many level combinations;
//   * fractional spacings: only what the spec defines - the counts as functions of the rounded levels, n-2 equal middle
//     segments with two equal end segments placed symmetrically, e == m when n == f, and that the end segments shrink
//     monotonically with n - f;
//   * winding (signed area in (u,v) space against cw/ccw), complete non-overlapping coverage of the domain, point mode
//     emitting one point per distinct vertex, discarded patches, level clamping to MAX_TESS_GEN_LEVEL, gl_PrimitiveID,
//     gl_PatchVerticesIn, patch outputs, barrier(), and the program queries TESS_CONTROL_OUTPUT_VERTICES / TESS_GEN_*.
// Pixels: the dots of the equal-spacing cells at CPU-projected positions, the orientation colours, the isolines, and a
// cull-face probe (generated winding x glFrontFace).

#include <demo/check.hpp>
#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    // =============================================================================================================
    // 1. The CPU model of ES 3.2 section 11.2.2 "Tessellation Primitive Generation"
    // =============================================================================================================
    enum class Domain { Triangles, Quads, Isolines };
    enum class Spacing { Equal, Odd, Even };
    enum class Order { Ccw, Cw };

    struct Variant
    {
        Domain domain = Domain::Triangles;
        Spacing spacing = Spacing::Equal;
        Order order = Order::Ccw;
        bool point = false;
        bool defaults = false;  // the layout names only the domain: spacing and winding are the defaults (equal_spacing, ccw)
        friend auto operator<=>(const Variant&, const Variant&) = default;
    };

    struct Levels
    {
        std::array<float, 4> outer{1.0f, 1.0f, 1.0f, 1.0f};
        std::array<float, 2> inner{1.0f, 1.0f};
    };

    // "If equal_spacing is used, the floating-point tessellation level is first clamped to the range [1, max] ... rounded up
    // to the nearest integer n. fractional_even_spacing: clamped to [2, max] and then rounded up to the nearest even
    // integer n. fractional_odd_spacing: clamped to [1, max - 1] and then rounded up to the nearest odd integer n."
    int round_level(double f, Spacing spacing, int max_level)
    {
        const double mx = static_cast<double>(max_level);
        switch (spacing)
        {
        case Spacing::Equal: return static_cast<int>(std::ceil(std::clamp(f, 1.0, mx)));
        case Spacing::Even:
        {
            const int n = static_cast<int>(std::ceil(std::clamp(f, 2.0, mx)));
            return n % 2 == 0 ? n : n + 1;
        }
        case Spacing::Odd:
        {
            const int n = static_cast<int>(std::ceil(std::clamp(f, 1.0, mx - 1.0)));
            return n % 2 == 1 ? n : n + 1;
        }
        }
        return 1;
    }

    // What the primitive generator makes of a set of levels.
    struct Plan
    {
        bool discarded = false;
        std::array<int, 4> outer{};  // rounded segment counts of the edges; isolines: [0] = number of isolines, [1] = segments per isoline
        std::array<int, 2> inner{};  // rounded inner levels, after the "1 + epsilon" rule
    };

    Plan plan_for(Domain d, Spacing s, const Levels& levels, int max_level)
    {
        Plan p;
        const int relevant = d == Domain::Triangles ? 3 : d == Domain::Quads ? 4 : 2;
        // "A patch is discarded ... if any relevant outer tessellation level is less than or equal to zero." (or NaN)
        for (int i = 0; i < relevant; ++i)
            if (!(levels.outer[static_cast<std::size_t>(i)] > 0.0f)) p.discarded = true;
        if (p.discarded) return p;
        if (d == Domain::Isolines)
        {
            // Section 11.2.2.3: the tessellation spacing is ignored (equal_spacing) for the number of isolines.
            p.outer[0] = round_level(levels.outer[0], Spacing::Equal, max_level);
            p.outer[1] = round_level(levels.outer[1], s, max_level);
            return p;
        }
        for (int i = 0; i < relevant; ++i)
            p.outer[static_cast<std::size_t>(i)] = round_level(levels.outer[static_cast<std::size_t>(i)], s, max_level);
        const int inner_count = d == Domain::Quads ? 2 : 1;
        bool all_one = true;
        for (int i = 0; i < relevant; ++i) all_one = all_one && p.outer[static_cast<std::size_t>(i)] == 1;
        for (int i = 0; i < inner_count; ++i)
        {
            p.inner[static_cast<std::size_t>(i)] = round_level(levels.inner[static_cast<std::size_t>(i)], s, max_level);
            all_one = all_one && p.inner[static_cast<std::size_t>(i)] == 1;
        }
        // "If the inner tessellation level is one and any of the outer tessellation levels is greater than one, the inner
        // level is treated as though it were originally specified as 1 + epsilon and will be rounded up to result in a
        // two- or three-segment subdivision according to the tessellation spacing."
        if (!all_one)
            for (int i = 0; i < inner_count; ++i)
                if (p.inner[static_cast<std::size_t>(i)] == 1) p.inner[static_cast<std::size_t>(i)] = s == Spacing::Odd ? 3 : 2;
        return p;
    }

    struct Counts
    {
        int vertices = 0;    // distinct vertices produced by the generator (= points in point mode)
        int primitives = 0;  // triangles, or line segments for isolines
    };

    // The number of vertices on the concentric inner triangles of a triangle patch whose first inner level rounds to n:
    // ring d has n - 2d segments per edge and 3 (n - 2d) vertices; a ring of 0 segments is the single centre point.
    int triangle_inner_vertices(int n)
    {
        int count = 0;
        for (int s = n - 2; s >= 1; s -= 2) count += 3 * s;
        if (n >= 2 && n % 2 == 0) count += 1;
        return count;
    }

    // Vertices come from the text. The number of triangles follows from Euler's formula for a triangulated disk whose
    // vertices are all used and which has B vertices on its boundary: F = 2 V - B - 2. The spec requires the triangles to
    // cover the domain completely and without overlap and to be built from the subdivided edges.
    Counts counts_for(Domain d, const Plan& p)
    {
        if (p.discarded) return {};
        if (d == Domain::Isolines) return {p.outer[0] * (p.outer[1] + 1), p.outer[0] * p.outer[1]};
        const int boundary = d == Domain::Triangles ? p.outer[0] + p.outer[1] + p.outer[2]
                                                    : p.outer[0] + p.outer[1] + p.outer[2] + p.outer[3];
        const int inner = d == Domain::Triangles ? triangle_inner_vertices(p.inner[0])
                                                 : (p.inner[0] - 1) * (p.inner[1] - 1);
        const int vertices = boundary + inner;
        return {vertices, 2 * vertices - boundary - 2};
    }

    using Coord = std::array<double, 3>;  // (u, v, w)

    void add_unique(std::vector<Coord>& set, const Coord& c)
    {
        for (const Coord& q : set)
            if (std::abs(q[0] - c[0]) < 1e-9 && std::abs(q[1] - c[1]) < 1e-9 && std::abs(q[2] - c[2]) < 1e-9) return;
        set.push_back(c);
    }

    // The exact vertex set of the generator for equal_spacing (Figure 11.1 for the position of the corners and edges).
    std::vector<Coord> expected_vertices(Domain d, const Plan& p)
    {
        std::vector<Coord> set;
        auto edge = [&](Coord a, Coord b, int segments) {
            for (int k = 0; k <= segments; ++k)
            {
                const double t = static_cast<double>(k) / static_cast<double>(segments);
                add_unique(set, {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t});
            }
        };
        if (p.discarded) return set;
        if (d == Domain::Triangles)
        {
            // u = 0 (left) edge: OL0, from (0,0,1) to (0,1,0); v = 0 (bottom): OL1, (0,0,1) to (1,0,0); w = 0 (right): OL2.
            edge({0, 0, 1}, {0, 1, 0}, p.outer[0]);
            edge({0, 0, 1}, {1, 0, 0}, p.outer[1]);
            edge({1, 0, 0}, {0, 1, 0}, p.outer[2]);
            // Concentric inner triangles: with first inner level n the three outer edges are temporarily cut into n
            // segments; the corner of the first inner triangle is the intersection of the perpendiculars to the two
            // adjacent edges through the vertices nearest the corner (a = 2 / (3 n) in barycentric coordinates). Each
            // further ring is one step deeper (a = 2 r / (3 n)) and has n - 2 r segments per edge; 0 segments = centre.
            const int n = p.inner[0];
            for (int r = 1; n - 2 * r >= 0; ++r)
            {
                const int s = n - 2 * r;
                if (s == 0)
                {
                    add_unique(set, {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
                    break;
                }
                const double a = 2.0 * r / (3.0 * n);
                const Coord cw = {a, a, 1.0 - 2.0 * a};  // near the (0,0,1) corner
                const Coord cu = {1.0 - 2.0 * a, a, a};  // near (1,0,0)
                const Coord cv = {a, 1.0 - 2.0 * a, a};  // near (0,1,0)
                edge(cw, cu, s);
                edge(cu, cv, s);
                edge(cv, cw, s);
            }
        }
        else if (d == Domain::Quads)
        {
            // u = 0 (left) OL0, v = 0 (bottom) OL1, u = 1 (right) OL2, v = 1 (top) OL3.
            edge({0, 0, 0}, {0, 1, 0}, p.outer[0]);
            edge({0, 0, 0}, {1, 0, 0}, p.outer[1]);
            edge({1, 0, 0}, {1, 1, 0}, p.outer[2]);
            edge({0, 1, 0}, {1, 1, 0}, p.outer[3]);
            // The inner mesh. Figure 11.3: inner levels (4, 2) give 4 segments along the horizontal edges (u) and 2 along
            // the vertical ones (v), so IL0 divides u and IL1 divides v. The grid points not on the outer edges are the
            // inner vertices.
            const int a = p.inner[0];
            const int b = p.inner[1];
            for (int i = 1; i < a; ++i)
                for (int j = 1; j < b; ++j)
                    add_unique(set, {static_cast<double>(i) / a, static_cast<double>(j) / b, 0.0});
        }
        else
        {
            const int n = p.outer[0];
            const int m = p.outer[1];
            for (int k = 0; k < n; ++k)
                for (int j = 0; j <= m; ++j)
                    add_unique(set, {static_cast<double>(j) / m, static_cast<double>(k) / n, 0.0});
        }
        return set;
    }

    // =============================================================================================================
    // 2. Analysis of captured data
    // =============================================================================================================
    // One captured vertex: the interleaved transform feedback layout (see domain.tese).
    struct Record
    {
        float coord[4];   // gl_TessCoord.xyz, gl_PrimitiveID
        float outer[4];   // gl_TessLevelOuter as seen by the TES
        float misc[4];    // gl_TessLevelInner[0..1], gl_PatchVerticesIn, t_tag[0]
        float center[4];  // p_center
        float pos[4];     // gl_Position (clip space)
    };
    static_assert(sizeof(Record) == 80);

    constexpr double kDedupe = 1e-4;  // distinct vertices are at least ~1/(3 x level) apart, far above this

    std::vector<Coord> distinct_coords(const std::vector<Record>& records)
    {
        std::vector<Coord> out;
        for (const Record& r : records)
        {
            const Coord c = {r.coord[0], r.coord[1], r.coord[2]};
            bool found = false;
            for (const Coord& q : out)
                if (std::abs(q[0] - c[0]) < kDedupe && std::abs(q[1] - c[1]) < kDedupe && std::abs(q[2] - c[2]) < kDedupe)
                {
                    found = true;
                    break;
                }
            if (!found) out.push_back(c);
        }
        return out;
    }

    // The largest distance from an expected vertex to the nearest captured one (infinity when `have` is empty).
    double max_match_error(const std::vector<Coord>& expected, const std::vector<Coord>& have)
    {
        double worst = 0.0;
        for (const Coord& e : expected)
        {
            double best = 1e9;
            for (const Coord& h : have)
                best = std::min(best, std::max({std::abs(e[0] - h[0]), std::abs(e[1] - h[1]), std::abs(e[2] - h[2])}));
            worst = std::max(worst, best);
        }
        return worst;
    }

    struct TriStats
    {
        int nondegenerate = 0;
        int degenerate = 0;
        int positive = 0;     // counter-clockwise in (u, v) space
        int negative = 0;     // clockwise
        double signed_sum = 0.0;
        double abs_sum = 0.0;
    };

    // Orientation in the (u, v) plane of Figure 11.1. For the triangle domain the map (u,v,w) -> (u,v) preserves
    // orientation and halves nothing: the domain has area 1/2 there, the quad domain area 1.
    TriStats triangle_stats(const std::vector<Record>& records)
    {
        TriStats s;
        for (std::size_t i = 0; i + 2 < records.size(); i += 3)
        {
            const Record& a = records[i];
            const Record& b = records[i + 1];
            const Record& c = records[i + 2];
            const double area = 0.5 * (static_cast<double>(b.coord[0] - a.coord[0]) * static_cast<double>(c.coord[1] - a.coord[1]) -
                                       static_cast<double>(c.coord[0] - a.coord[0]) * static_cast<double>(b.coord[1] - a.coord[1]));
            if (std::abs(area) < 1e-10)
            {
                ++s.degenerate;
                continue;
            }
            ++s.nondegenerate;
            (area > 0.0 ? s.positive : s.negative) += 1;
            s.signed_sum += area;
            s.abs_sum += std::abs(area);
        }
        return s;
    }

    // Do all non-degenerate triangles have the winding the layout asked for? (cw = negative signed area in (u, v) space.)
    bool winding_matches(const TriStats& s, Order order)
    {
        return s.nondegenerate > 0 && (order == Order::Ccw ? s.negative == 0 : s.positive == 0);
    }

    // The positions of the vertices on one outer edge, sorted ascending: `coordinate` selects the varying coordinate,
    // `fixed` the coordinate that is constant along the edge.
    std::vector<double> edge_positions(const std::vector<Coord>& distinct, int fixed_index, double fixed_value, int along_index)
    {
        std::vector<double> out;
        for (const Coord& c : distinct)
            if (std::abs(c[static_cast<std::size_t>(fixed_index)] - fixed_value) < 2e-4) out.push_back(c[static_cast<std::size_t>(along_index)]);
        std::sort(out.begin(), out.end());
        return out;
    }

    // Structure of a subdivided edge (section 11.2.2): `positions` are the n + 1 vertex positions in [0,1].
    struct EdgeShape
    {
        bool segment_count_ok = false;
        bool symmetric = false;        // the set of positions maps onto itself under p -> 1 - p
        bool lengths_ok = false;       // n - 2 segments of one length and two additional segments of equal length
        double end_to_middle = 0.0;    // e / m, additional segment length over the others (1 for equal spacing)
        double max_asym = 0.0;
    };

    // The spec fixes the number of segments (n), that n - 2 of them have equal length and the two additional ones have
    // equal length, and that the additional ones are placed symmetrically ("The relative location of these two segments is
    // undefined"). So the check does not look at where the short segments are: it clusters the n lengths.
    EdgeShape edge_shape(const std::vector<double>& p, int n, bool fractional)
    {
        EdgeShape shape;
        shape.segment_count_ok = static_cast<int>(p.size()) == n + 1;
        if (!shape.segment_count_ok) return shape;
        const double tol = 1e-3;
        for (std::size_t i = 0; i < p.size(); ++i) shape.max_asym = std::max(shape.max_asym, std::abs(p[i] + p[p.size() - 1 - i] - 1.0));
        shape.symmetric = shape.max_asym < tol;
        std::vector<double> len;
        for (std::size_t i = 0; i + 1 < p.size(); ++i) len.push_back(p[i + 1] - p[i]);
        std::vector<std::pair<double, int>> clusters;  // (length, how many segments have it)
        for (const double l : len)
        {
            bool found = false;
            for (auto& c : clusters)
                if (std::abs(c.first - l) < tol)
                {
                    ++c.second;
                    found = true;
                    break;
                }
            if (!found) clusters.emplace_back(l, 1);
        }
        shape.end_to_middle = 1.0;
        if (!fractional)
        {
            shape.lengths_ok = clusters.size() == 1 && std::abs(clusters[0].first - 1.0 / n) < tol;
            return shape;
        }
        if (clusters.size() == 1) shape.lengths_ok = true;
        else if (clusters.size() == 2)
        {
            // m is the length shared by n - 2 segments; with n = 4 both groups have two, then m is the longer one.
            auto m = clusters[0];
            auto e = clusters[1];
            if (e.second == n - 2 && (m.second != n - 2 || e.first > m.first)) std::swap(m, e);
            shape.lengths_ok = m.second == n - 2 && e.second == 2;
            shape.end_to_middle = e.first / m.first;
        }
        return shape;
    }

    // =============================================================================================================
    // 3. Scene description
    // =============================================================================================================
    struct PatchVertex
    {
        float x, y, z, tag;
    };

    struct Rect
    {
        int x, y, w, h;
    };

    struct CellSpec
    {
        Variant variant;
        Levels levels;
    };

    // Row 0 triangles, row 1 quads, row 2 isolines; columns: equal, fractional odd, fractional even spacing.
    std::array<CellSpec, 9> make_cells()
    {
        auto lv = [](std::array<float, 4> o, std::array<float, 2> i) { return Levels{o, i}; };
        return {{
            {{Domain::Triangles, Spacing::Equal, Order::Ccw, false}, lv({2, 3, 4, 1}, {4, 1})},
            {{Domain::Triangles, Spacing::Odd, Order::Cw, false}, lv({2.6f, 3.4f, 4.2f, 1}, {3.4f, 1})},
            {{Domain::Triangles, Spacing::Even, Order::Ccw, false}, lv({2.5f, 3.5f, 4.6f, 1}, {3.4f, 1})},
            {{Domain::Quads, Spacing::Equal, Order::Cw, false}, lv({2, 3, 4, 3}, {4, 3})},
            {{Domain::Quads, Spacing::Odd, Order::Ccw, false}, lv({2.6f, 3.4f, 3.2f, 4.4f}, {3.4f, 2.6f})},
            {{Domain::Quads, Spacing::Even, Order::Cw, false}, lv({2.5f, 3.5f, 2.6f, 3.6f}, {3.4f, 2.5f})},
            {{Domain::Isolines, Spacing::Equal, Order::Ccw, false}, lv({4, 6, 1, 1}, {1, 1})},
            {{Domain::Isolines, Spacing::Odd, Order::Ccw, false}, lv({4, 4.2f, 1, 1}, {1, 1})},
            {{Domain::Isolines, Spacing::Even, Order::Ccw, false}, lv({4, 4.2f, 1, 1}, {1, 1})},
        }};
    }

    constexpr float kDome = 0.75f;     // height of the bump the patches are bent into
    constexpr float kRadius = 1.5f;
    constexpr int kRecordCapacity = 1 << 16;

    const std::array<float, 3> kFront = {0.10f, 0.62f, 0.85f};   // counter-clockwise triangles (teal)
    const std::array<float, 3> kBack = {0.95f, 0.50f, 0.12f};    // clockwise triangles (orange)
    const std::array<float, 3> kLine = {0.55f, 0.95f, 0.45f};    // isolines
    constexpr demo::Rgba8 kDotColor = {255, 255, 255, 255};
    constexpr demo::Rgba8 kBackground = {26, 29, 43, 255};

    // Direction of a pixel: front-facing (teal), back-facing (orange) or neither (background, wire, dots, lines).
    enum class Tint { None, Front, Back };
    Tint classify(const std::uint8_t* px)
    {
        const int r = px[0], b = px[2];
        if (b > r + 30 && b > 50) return Tint::Front;
        if (r > b + 30 && r > 50) return Tint::Back;
        return Tint::None;
    }

    struct Prog
    {
        mgl::ProgramId id{};
        mgl::UniformLocation u_outer, u_inner, u_bias, u_mvp, u_dome, u_radius, u_front, u_back, u_height, u_lines, u_color, u_size;
    };

    struct ProgramKey
    {
        Variant variant;
        int out_vertices = 3;
        bool wire = false;
        friend auto operator<=>(const ProgramKey&, const ProgramKey&) = default;
    };

    struct PatchSet
    {
        std::vector<PatchVertex> verts;
        int patch_size = 3;      // PATCH_VERTICES for the draw
        int out_vertices = 3;    // the TCS output patch size the set is meant for
        mgl::BufferId vbo{};
    };

    struct Capture
    {
        std::vector<Record> records;
        GLuint written = 0;
        GLuint generated = 0;
    };

    std::string_view domain_name(Domain d)
    {
        return d == Domain::Triangles ? "triangles" : d == Domain::Quads ? "quads" : "isolines";
    }
    std::string_view spacing_name(Spacing s)
    {
        return s == Spacing::Equal ? "equal_spacing" : s == Spacing::Odd ? "fractional_odd_spacing" : "fractional_even_spacing";
    }
    std::string label_of(const Variant& v)
    {
        return std::format("{}/{}/{}{}{}", domain_name(v.domain), spacing_name(v.spacing), v.order == Order::Ccw ? "ccw" : "cw",
                           v.point ? "/point_mode" : "", v.defaults ? "/layout-defaults" : "");
    }

    std::string layout_of(const Variant& v)
    {
        if (v.defaults) return std::string(domain_name(v.domain)) + (v.point ? ", point_mode" : "");
        return std::format("{}, {}, {}{}", domain_name(v.domain), spacing_name(v.spacing), v.order == Order::Ccw ? "ccw" : "cw",
                           v.point ? ", point_mode" : "");
    }

    int standard_out_vertices(Domain d) { return d == Domain::Triangles ? 3 : 4; }

    // Inserts `defines` after the "#version" line (which must stay first, GLSL ES 3.20 section 3.4).
    std::string with_defines(const std::string& source, const std::string& defines)
    {
        const std::size_t nl = source.find('\n');
        return source.substr(0, nl + 1) + defines + source.substr(nl + 1);
    }

    int count_per_primitive(const Variant& v) { return v.point ? 1 : v.domain == Domain::Isolines ? 2 : 3; }

    // The window-space position (pixels, GL origin bottom-left) of a clip-space position in a viewport.
    std::array<double, 2> to_window(const dm::Vec4& clip, const Rect& r)
    {
        const double nx = static_cast<double>(clip.x) / static_cast<double>(clip.w);
        const double ny = static_cast<double>(clip.y) / static_cast<double>(clip.w);
        return {r.x + (nx + 1.0) * 0.5 * r.w, r.y + (ny + 1.0) * 0.5 * r.h};
    }

    class TessellationDomains final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            max_level_ = demo::get_integer(mgl::GetParameter::MaxTessGenLevel);
            max_patch_vertices_ = demo::get_integer(mgl::GetParameter::MaxPatchVertices);
            GLfloat psize[2] = {};
            mgl::glGetFloatv(mgl::GetParameter::AliasedPointSizeRange, psize);
            max_point_size_ = psize[1];
            cells_ = make_cells();

            load_sources(env);

            // The patch data. Display patches are symmetric; capture patches are irregular so that the centroid computed
            // in the TCS and the interpolation in the TES are non-trivial.
            const float k = 0.90933f;  // sqrt(3) / 2 * 1.05
            display_tri_ = make_patch({{k, -0.525f, 0, 1}, {0, 1.05f, 0, 2}, {-k, -0.525f, 0, 3}}, 3, 3);
            display_quad_ = make_patch({{-0.9f, -0.9f, 0, 10}, {0.9f, -0.9f, 0, 11}, {-0.9f, 0.9f, 0, 12}, {0.9f, 0.9f, 0, 13}}, 4, 4);
            irregular_tri_ = make_patch({{0.85f, -0.55f, 0.10f, 21}, {-0.10f, 0.95f, 0.30f, 22}, {-0.80f, -0.40f, -0.20f, 23}}, 3, 3);
            irregular_quad_ = make_patch({{-0.8f, -0.6f, 0.10f, 31}, {0.9f, -0.7f, 0.30f, 32}, {-0.7f, 0.8f, -0.20f, 33}, {0.6f, 0.9f, 0.40f, 34}}, 4, 4);
            {   // three quad patches (tag[0] = 100 + 10 p) and two stray vertices: "if k is not zero, the final k vertices are ignored"
                std::vector<PatchVertex> v;
                for (int p = 0; p < 3; ++p)
                {
                    const float ox = 0.25f * static_cast<float>(p);
                    for (int i = 0; i < 4; ++i)
                        v.push_back({-0.8f + ox + 0.05f * static_cast<float>(i * p), -0.7f + 0.3f * static_cast<float>(i % 2), 0.1f * static_cast<float>(p) + 0.05f * static_cast<float>(i),
                                     static_cast<float>(100 + 10 * p + i)});
                }
                v.push_back({0.1f, 0.2f, 0.3f, 900});
                v.push_back({0.3f, 0.2f, 0.1f, 901});
                multi_ = make_patch(std::move(v), 4, 4);
            }
            {   // two patches of five vertices, drawn with a TCS whose output patch has three vertices, plus three strays
                std::vector<PatchVertex> v;
                for (int p = 0; p < 2; ++p)
                    for (int i = 0; i < 5; ++i)
                        v.push_back({-0.7f + 0.3f * static_cast<float>(i) + 0.1f * static_cast<float>(p), 0.4f * static_cast<float>((i * 3 + p) % 4) - 0.5f,
                                     0.05f * static_cast<float>(i), static_cast<float>(200 + 10 * p + i)});
                for (int i = 0; i < 3; ++i) v.push_back({0.0f, 0.0f, 0.0f, 990.0f + static_cast<float>(i)});
                resample_ = make_patch(std::move(v), 5, 3);
            }
            {   // one patch with the largest legal number of vertices, input and output
                std::vector<PatchVertex> v;
                for (int i = 0; i < max_patch_vertices_; ++i)
                {
                    const float a = 6.2831853f * static_cast<float>(i) / static_cast<float>(max_patch_vertices_);
                    v.push_back({0.8f * std::cos(a), 0.7f * std::sin(a), 0.02f * static_cast<float>(i % 5), static_cast<float>(300 + i)});
                }
                max_patch_ = make_patch(std::move(v), max_patch_vertices_, max_patch_vertices_);
            }

            for (PatchSet* ps : {&display_tri_, &display_quad_, &irregular_tri_, &irregular_quad_, &multi_, &resample_, &max_patch_})
            {
                mgl::glGenBuffers(1, &ps->vbo);
                mgl::glBindBuffer(mgl::BufferTarget::Array, ps->vbo);
                mgl::glBufferData(mgl::BufferTarget::Array, std::span<const PatchVertex>(ps->verts), mgl::BufferUsage::StaticDraw);
            }
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // Transform feedback buffer: the TES output of whatever is drawn.
            mgl::glGenBuffers(1, &tf_buffer_);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tf_buffer_);
            mgl::glBufferData(mgl::BufferTarget::TransformFeedback, static_cast<GLsizeiptr>(kRecordCapacity * sizeof(Record)), nullptr,
                              mgl::BufferUsage::StreamRead);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});

            mgl::glGenVertexArrays(1, &vao_patch_);
            mgl::glGenVertexArrays(1, &vao_dots_);
            mgl::glGenQueries(1, &q_written_);
            mgl::glGenQueries(1, &q_generated_);

            dots_program_ = link_simple(env.load_text("shaders/dots.vert"), env.load_text("shaders/solid.frag"));
            dots_u_size_ = mgl::glGetUniformLocation(dots_program_, "u_size");
            dots_u_color_ = mgl::glGetUniformLocation(dots_program_, "u_color");

            // The gallery programs are built up front: a failure here is a failure of the demo, not of a check.
            for (const CellSpec& c : cells_)
            {
                (void)program({c.variant, standard_out_vertices(c.variant.domain), false});
                Variant pv = c.variant;
                pv.point = true;
                (void)program({pv, standard_out_vertices(pv.domain), false});
                if (c.variant.domain != Domain::Isolines) (void)program({c.variant, standard_out_vertices(c.variant.domain), true});
            }
        }

        void render(demo::Env& env, int frame) override
        {
            if (env.interactive && auto_rotate_) angle_ = 0.012f * static_cast<float>(frame);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.02f, 0.022f, 0.03f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (int i = 0; i < 9; ++i) draw_cell(env, i);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::DepthTest);
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
            case 'a': auto_rotate_ = !auto_rotate_; break;
            case 'w': wire_ = !wire_; break;
            case 'p': dots_on_ = !dots_on_; break;
            case '+':
            case '=': level_scale_ = std::min(level_scale_ * 1.25f, 8.0f); break;
            case '-': level_scale_ = std::max(level_scale_ / 1.25f, 0.25f); break;
            case 'r': level_scale_ = 1.0f; angle_ = 0.0f; wire_ = dots_on_ = true; break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (auto& [key, prog] : programs_) mgl::glDeleteProgram(prog.id);
            programs_.clear();
            mgl::glDeleteProgram(dots_program_);
            for (PatchSet* ps : {&display_tri_, &display_quad_, &irregular_tri_, &irregular_quad_, &multi_, &resample_, &max_patch_})
                mgl::glDeleteBuffers(1, &ps->vbo);
            mgl::glDeleteBuffers(1, &tf_buffer_);
            mgl::glDeleteVertexArrays(1, &vao_patch_);
            mgl::glDeleteVertexArrays(1, &vao_dots_);
            mgl::glDeleteQueries(1, &q_written_);
            mgl::glDeleteQueries(1, &q_generated_);
        }

    private:
        // -----------------------------------------------------------------------------------------------------
        // Programs
        // -----------------------------------------------------------------------------------------------------
        void load_sources(const demo::Env& env)
        {
            src_vert_ = env.load_text("shaders/patch.vert");
            src_tesc_ = env.load_text("shaders/domain.tesc");
            src_tese_ = env.load_text("shaders/domain.tese");
            src_geom_ = env.load_text("shaders/wire.geom");
            src_surface_ = env.load_text("shaders/surface.frag");
            src_solid_ = env.load_text("shaders/solid.frag");
        }

        mgl::ProgramId link_simple(const std::string& vs_src, const std::string& fs_src)
        {
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, vs_src, log);
            if (vs.value == 0) throw demo::Failure("vertex shader failed to compile:\n" + log);
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, fs_src, log);
            if (fs.value == 0) throw demo::Failure("fragment shader failed to compile:\n" + log);
            const mgl::ShaderId both[] = {vs, fs};
            const mgl::ProgramId p = demo::link_program(both, log);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (p.value == 0) throw demo::Failure("program failed to link:\n" + log);
            return p;
        }

        // Builds vertex + TCS + TES (+ geometry) + fragment. Returns 0 and fills `log` when any stage fails.
        mgl::ProgramId try_build_tess(const ProgramKey& key, const std::string& tesc_defines, const std::string& tese_defines,
                                      bool with_transform_feedback, std::string& log)
        {
            const std::string tesc_src = with_defines(src_tesc_, tesc_defines);
            const std::string tese_src = with_defines(src_tese_, tese_defines);
            std::vector<mgl::ShaderId> shaders;
            auto compile = [&](mgl::ShaderType type, const std::string& src) {
                std::string l;
                const mgl::ShaderId s = demo::compile_shader(type, src, l);
                if (s.value == 0) log += l;
                else shaders.push_back(s);
                return s.value != 0;
            };
            bool ok = compile(mgl::ShaderType::Vertex, src_vert_) && compile(mgl::ShaderType::TessControl, tesc_src) &&
                      compile(mgl::ShaderType::TessEvaluation, tese_src);
            if (ok && key.wire) ok = compile(mgl::ShaderType::Geometry, src_geom_);
            if (ok) ok = compile(mgl::ShaderType::Fragment, key.wire ? src_solid_ : src_surface_);
            mgl::ProgramId program{};
            if (ok)
            {
                program = mgl::glCreateProgram();
                for (const mgl::ShaderId s : shaders) mgl::glAttachShader(program, s);
                if (with_transform_feedback)
                {
                    static const GLchar* const varyings[] = {"tf_coord", "tf_outer", "tf_misc", "tf_center", "gl_Position"};
                    mgl::glTransformFeedbackVaryings(program, 5, varyings, mgl::TransformFeedbackBufferMode::InterleavedAttribs);
                }
                mgl::glLinkProgram(program);
                GLint status = GL_FALSE;
                mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &status);
                if (status != GL_TRUE)
                {
                    log += demo::program_info_log(program);
                    ok = false;
                }
                for (const mgl::ShaderId s : shaders) mgl::glDetachShader(program, s);
                if (!ok)
                {
                    mgl::glDeleteProgram(program);
                    program = mgl::ProgramId{0};
                }
            }
            for (const mgl::ShaderId s : shaders) mgl::glDeleteShader(s);
            return program;
        }

        std::string standard_tesc_defines(int out_vertices) const { return std::format("#define OUT_VERTICES {}\n", out_vertices); }
        static std::string standard_tese_defines(const Variant& v)
        {
            const char* dom = v.domain == Domain::Triangles ? "DOMAIN_TRIANGLES" : v.domain == Domain::Quads ? "DOMAIN_QUADS" : "DOMAIN_ISOLINES";
            return std::format("#define TESS_LAYOUT {}\n#define {} 1\n", layout_of(v), dom);
        }

        Prog make_prog(mgl::ProgramId id) const
        {
            Prog p;
            p.id = id;
            auto loc = [&](const char* name) { return mgl::glGetUniformLocation(id, name); };
            p.u_outer = loc("u_outer");
            p.u_inner = loc("u_inner");
            p.u_bias = loc("u_bias");
            p.u_mvp = loc("u_mvp");
            p.u_dome = loc("u_dome");
            p.u_radius = loc("u_radius");
            p.u_front = loc("u_front");
            p.u_back = loc("u_back");
            p.u_height = loc("u_height");
            p.u_lines = loc("u_lines");
            p.u_color = loc("u_color");
            p.u_size = loc("u_size");
            return p;
        }

        // Built on first use. The programs that carry transform feedback varyings are the ones that rasterize the
        // surfaces; the wireframe programs (with a geometry shader, whose output would be captured instead) have none.
        const Prog& program(const ProgramKey& key)
        {
            const auto it = programs_.find(key);
            if (it != programs_.end()) return it->second;
            std::string log;
            const mgl::ProgramId id = try_build_tess(key, standard_tesc_defines(key.out_vertices), standard_tese_defines(key.variant), !key.wire, log);
            if (id.value == 0) throw demo::Failure(std::format("tessellation program {} failed:\n{}", label_of(key.variant), log));
            return programs_.emplace(key, make_prog(id)).first->second;
        }

        // -----------------------------------------------------------------------------------------------------
        // Scene
        // -----------------------------------------------------------------------------------------------------
        static PatchSet make_patch(std::vector<PatchVertex> verts, int patch_size, int out_vertices)
        {
            PatchSet ps;
            ps.verts = std::move(verts);
            ps.patch_size = patch_size;
            ps.out_vertices = out_vertices;
            return ps;
        }

        static Rect cell_rect(int width, int height, int col, int row)
        {
            const int cw = width / 3;
            const int ch = height / 3;
            const int x0 = (width - 3 * cw) / 2;
            const int y0 = (height - 3 * ch) / 2;
            return {x0 + col * cw + 2, y0 + (2 - row) * ch + 2, cw - 4, ch - 4};
        }

        // The camera of a cell: looks down at the patch from 50 degrees above the horizon, optionally orbiting. The triangle's
        // centre of area is above the origin of its patch, so its camera aims a little higher.
        dm::Mat4 cell_mvp(Domain d = Domain::Quads) const
        {
            const float c = std::cos(angle_), s = std::sin(angle_);
            const dm::Vec3 eye = {2.0f * s, -2.0f * c, 3.2f};
            const dm::Vec3 target = {0.0f, d == Domain::Triangles ? 0.14f : 0.0f, 0.2f};
            return dm::perspective(dm::radians(38.0f), 1.0f, 0.5f, 20.0f) * dm::look_at(eye, target, {0.0f, 0.0f, 1.0f});
        }

        Levels scaled(const Levels& l) const
        {
            Levels out = l;
            for (float& f : out.outer) f = std::max(f * level_scale_, 1.0f);
            for (float& f : out.inner) f = std::max(f * level_scale_, 1.0f);
            return out;
        }

        void bind_patches(const PatchSet& ps) const
        {
            mgl::glBindVertexArray(vao_patch_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, ps.vbo);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(PatchVertex), demo::buffer_offset(offsetof(PatchVertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 1, mgl::DataType::Float, GL_FALSE, sizeof(PatchVertex), demo::buffer_offset(offsetof(PatchVertex, tag)));
        }

        static void set_levels(const Prog& p, const Levels& l, float bias, const dm::Mat4& mvp)
        {
            mgl::glUniform4f(p.u_outer, l.outer[0], l.outer[1], l.outer[2], l.outer[3]);
            mgl::glUniform2f(p.u_inner, l.inner[0], l.inner[1]);
            mgl::glUniform1f(p.u_bias, bias);
            mgl::glUniformMatrix4fv(p.u_mvp, 1, GL_FALSE, mvp.data());
            mgl::glUniform1f(p.u_dome, kDome);
            mgl::glUniform1f(p.u_radius, kRadius);
        }

        void draw_patches(const PatchSet& ps, int vertex_count) const
        {
            bind_patches(ps);
            mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, ps.patch_size);
            mgl::glDrawArrays(mgl::PrimitiveType::Patches, 0, vertex_count);
        }

        const PatchSet& display_patch(Domain d) const { return d == Domain::Triangles ? display_tri_ : display_quad_; }

        // Runs a program in point mode / triangle / line mode with RASTERIZER_DISCARD and transform feedback.
        Capture run_capture(const Prog& prog, const PatchSet& ps, int vertex_count, const Levels& levels, float bias,
                            mgl::TransformFeedbackPrimitive mode, bool read_back, bool set_patch_size = true,
                            std::optional<dm::Mat4> mvp = std::nullopt)
        {
            Capture cap;
            mgl::glUseProgram(prog.id);
            set_levels(prog, levels, bias, mvp.value_or(cell_mvp()));
            bind_patches(ps);
            if (set_patch_size) mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, ps.patch_size);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(mgl::QueryTarget::PrimitivesGenerated, q_generated_);
            mgl::glBeginQuery(mgl::QueryTarget::TransformFeedbackPrimitivesWritten, q_written_);
            mgl::glBeginTransformFeedback(mode);
            mgl::glDrawArrays(mgl::PrimitiveType::Patches, 0, vertex_count);
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(mgl::QueryTarget::TransformFeedbackPrimitivesWritten);
            mgl::glEndQuery(mgl::QueryTarget::PrimitivesGenerated);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glGetQueryObjectuiv(q_written_, mgl::QueryObjectParameter::Result, &cap.written);
            mgl::glGetQueryObjectuiv(q_generated_, mgl::QueryObjectParameter::Result, &cap.generated);
            if (read_back && cap.written > 0)
            {
                const std::size_t verts = static_cast<std::size_t>(cap.written) * (mode == mgl::TransformFeedbackPrimitive::Points ? 1u : mode == mgl::TransformFeedbackPrimitive::Lines ? 2u : 3u);
                const std::size_t n = std::min<std::size_t>(verts, kRecordCapacity);
                mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tf_buffer_);
                const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::TransformFeedback, 0, static_cast<GLsizeiptr>(n * sizeof(Record)), mgl::MapBufferAccessMask::Read);
                if (mapped != nullptr)
                {
                    cap.records.resize(n);
                    std::memcpy(cap.records.data(), mapped, n * sizeof(Record));
                    (void)mgl::glUnmapBuffer(mgl::BufferTarget::TransformFeedback);
                }
                mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            }
            return cap;
        }

        static mgl::TransformFeedbackPrimitive capture_mode(const Variant& v)
        {
            if (v.point) return mgl::TransformFeedbackPrimitive::Points;
            return v.domain == Domain::Isolines ? mgl::TransformFeedbackPrimitive::Lines : mgl::TransformFeedbackPrimitive::Triangles;
        }

        // One gallery cell: the surface, its wireframe, the generated vertices.
        void draw_cell(const demo::Env& env, int index)
        {
            const CellSpec& cell = cells_[static_cast<std::size_t>(index)];
            const Variant& v = cell.variant;
            const Rect r = cell_rect(env.width, env.height, index % 3, index / 3);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glClearColor(kBackground[0] / 255.0f, kBackground[1] / 255.0f, kBackground[2] / 255.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            const Levels levels = scaled(cell.levels);
            const dm::Mat4 mvp = cell_mvp(v.domain);
            const PatchSet& ps = display_patch(v.domain == Domain::Triangles ? Domain::Triangles : Domain::Quads);
            const int out_vertices = standard_out_vertices(v.domain);
            const bool iso = v.domain == Domain::Isolines;

            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Lequal);
            if (iso)
            {   // A dim copy of the domain (a finely tessellated quad) under the lines, to show the shape the isolines lie on.
                const Prog& p = program({{Domain::Quads, Spacing::Equal, Order::Ccw, false}, 4, false});
                mgl::glUseProgram(p.id);
                set_levels(p, Levels{{12, 12, 12, 12}, {12, 12}}, 0.0f, mvp);
                mgl::glUniform3f(p.u_front, 0.12f, 0.17f, 0.27f);
                mgl::glUniform3f(p.u_back, 0.12f, 0.17f, 0.27f);
                mgl::glUniform1f(p.u_height, kDome);
                mgl::glUniform1i(p.u_lines, 1);
                mgl::glDepthMask(GL_FALSE);  // the lines lie on this surface: it must not hide them
                draw_patches(display_quad_, 4);
                mgl::glDepthMask(GL_TRUE);
            }
            // Pass 1: the generated triangles (or lines), depth-tested.
            {
                const Prog& p = program({v, out_vertices, false});
                mgl::glUseProgram(p.id);
                set_levels(p, levels, 0.0f, mvp);
                const auto& c = iso ? kLine : kFront;
                mgl::glUniform3f(p.u_front, c[0], c[1], c[2]);
                mgl::glUniform3f(p.u_back, kBack[0], kBack[1], kBack[2]);
                mgl::glUniform1f(p.u_height, kDome);
                mgl::glUniform1i(p.u_lines, iso ? 1 : 0);
                draw_patches(ps, ps.patch_size);
            }
            // Pass 2: wireframe of the generated triangles (geometry shader).
            if (wire_ && !iso)
            {
                const Prog& p = program({v, out_vertices, true});
                mgl::glUseProgram(p.id);
                set_levels(p, levels, 0.0f, mvp);
                mgl::glUniform4f(p.u_color, 0.02f, 0.03f, 0.05f, 1.0f);
                draw_patches(ps, ps.patch_size);
            }
            // Pass 3: the generated vertices (point_mode), captured and drawn as dots.
            mgl::glDisable(mgl::Capability::DepthTest);
            if (dots_on_)
            {
                Variant pv = v;
                pv.point = true;
                const Prog& p = program({pv, out_vertices, false});
                const Capture cap = run_capture(p, ps, ps.patch_size, levels, 0.0f, mgl::TransformFeedbackPrimitive::Points, false, true, mvp);
                mgl::glUseProgram(dots_program_);
                mgl::glUniform1f(dots_u_size_, std::min(std::max(static_cast<float>(r.w) / 24.0f, 3.0f), std::max(max_point_size_, 1.0f)));
                mgl::glUniform4f(dots_u_color_, 1.0f, 1.0f, 1.0f, 1.0f);
                mgl::glBindVertexArray(vao_dots_);
                mgl::glBindBuffer(mgl::BufferTarget::Array, tf_buffer_);
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
                mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(Record), demo::buffer_offset(offsetof(Record, pos)));
                mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, static_cast<GLsizei>(cap.written));
                mgl::glBindVertexArray(mgl::VertexArrayId{0});
                mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            }
        }

        // -----------------------------------------------------------------------------------------------------
        // CPU model of the shaders
        // -----------------------------------------------------------------------------------------------------
        // The clip-space position the TES computes for a tessellation coordinate (domain.tese).
        dm::Vec4 cpu_clip(const PatchSet& ps, std::size_t first_vertex, Domain d, const Coord& c, const dm::Mat4& mvp) const
        {
            auto at = [&](std::size_t i) { return ps.verts[first_vertex + i]; };
            dm::Vec3 p{};
            if (d == Domain::Triangles)
            {
                const auto a = at(0), b = at(1), cc = at(2);
                const float u = static_cast<float>(c[0]), v = static_cast<float>(c[1]), w = static_cast<float>(c[2]);
                p = {u * a.x + v * b.x + w * cc.x, u * a.y + v * b.y + w * cc.y, u * a.z + v * b.z + w * cc.z};
            }
            else
            {
                const auto p0 = at(0), p1 = at(1), p2 = at(2), p3 = at(3);
                const float u = static_cast<float>(c[0]), v = static_cast<float>(c[1]);
                auto mix = [](float a, float b, float t) { return a * (1.0f - t) + b * t; };
                auto m3 = [&](float f0, float f1, float f2, float f3) { return mix(mix(f0, f1, u), mix(f2, f3, u), v); };
                p = {m3(p0.x, p1.x, p2.x, p3.x), m3(p0.y, p1.y, p2.y, p3.y), m3(p0.z, p1.z, p2.z, p3.z)};
            }
            const dm::Vec3 center = patch_center(ps, first_vertex);
            const float dx = (p.x - center.x) / kRadius, dy = (p.y - center.y) / kRadius;
            const float t = std::max(0.0f, 1.0f - (dx * dx + dy * dy));
            const float h = kDome * t * t;
            return mvp * dm::Vec4{p.x, p.y, p.z + h, 1.0f};
        }

        static dm::Vec3 patch_center(const PatchSet& ps, std::size_t first_vertex)
        {
            dm::Vec3 sum{};
            for (int i = 0; i < ps.out_vertices; ++i)
            {
                const auto& v = ps.verts[first_vertex + static_cast<std::size_t>(i)];
                sum = sum + dm::Vec3{v.x, v.y, v.z};
            }
            return sum * (1.0f / static_cast<float>(ps.out_vertices));
        }

        // -----------------------------------------------------------------------------------------------------
        // Checks
        // -----------------------------------------------------------------------------------------------------
        struct Verdict
        {
            bool ok = true;
            std::string detail;
            void fail(const std::string& why)
            {
                ok = false;
                if (!detail.empty()) detail += "; ";
                detail += why;
            }
        };

        // Verifies one capture against the CPU model. `exact` = equal_spacing: the vertex set is compared coordinate by
        // coordinate; for fractional spacings only counts and structure are asserted.
        void verify_capture(demo::Check& check, const std::string& label, const Variant& v, const Levels& levels, const PatchSet& ps,
                            float bias, const Capture& cap, std::size_t patch_count)
        {
            // ---- counts -------------------------------------------------------------------------------------
            Counts total{};
            std::vector<Plan> plans;
            for (std::size_t k = 0; k < patch_count; ++k)
            {
                Levels l = levels;
                const float b = bias * static_cast<float>(k);
                for (float& f : l.outer) f += b;
                for (float& f : l.inner) f += b;
                plans.push_back(plan_for(v.domain, v.spacing, l, max_level_));
                const Counts c = counts_for(v.domain, plans.back());
                total.vertices += c.vertices;
                total.primitives += c.primitives;
            }
            const int per = count_per_primitive(v);
            const int expect_primitives = v.point ? total.vertices : total.primitives;
            {
                Verdict vd;
                if (static_cast<int>(cap.written) != expect_primitives) vd.fail(std::format("TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN {} != CPU {}", cap.written, expect_primitives));
                if (static_cast<int>(cap.generated) != expect_primitives) vd.fail(std::format("PRIMITIVES_GENERATED {} != CPU {}", cap.generated, expect_primitives));
                if (static_cast<int>(cap.records.size()) != expect_primitives * per) vd.fail(std::format("{} records, expected {}", cap.records.size(), expect_primitives * per));
                check.expect(vd.ok, std::format("{}: primitives written = generated = CPU count ({} {})", label, expect_primitives,
                                                v.point ? "points" : v.domain == Domain::Isolines ? "lines" : "triangles"), vd.detail);
            }
            if (expect_primitives == 0 || static_cast<int>(cap.records.size()) != expect_primitives * per) return;

            // ---- per record metadata: patch id, levels seen by the TES, patch size, centroid, position -----------
            {
                Verdict vd;
                const dm::Mat4 mvp = cell_mvp();
                double worst_pos = 0.0;
                for (const Record& r : cap.records)
                {
                    const std::size_t k = static_cast<std::size_t>(std::lround(r.coord[3]));
                    if (k >= patch_count) { vd.fail(std::format("gl_PrimitiveID {} out of range", r.coord[3])); break; }
                    const float b = bias * static_cast<float>(k);
                    const int relevant_outer = v.domain == Domain::Triangles ? 3 : v.domain == Domain::Quads ? 4 : 2;
                    for (int i = 0; i < relevant_outer; ++i)
                        if (std::abs(r.outer[i] - (levels.outer[static_cast<std::size_t>(i)] + b)) > 1e-5f) { vd.fail("gl_TessLevelOuter seen by the TES differs"); break; }
                    const int relevant_inner = v.domain == Domain::Triangles ? 1 : v.domain == Domain::Quads ? 2 : 0;
                    for (int i = 0; i < relevant_inner; ++i)
                        if (std::abs(r.misc[i] - (levels.inner[static_cast<std::size_t>(i)] + b)) > 1e-5f) { vd.fail("gl_TessLevelInner seen by the TES differs"); break; }
                    // The TES's input patch is the TCS output patch (section 11.2.3: "equal to the tessellation control shader
                    // output patch size"); the TCS's gl_PatchVerticesIn is PATCH_VERTICES and arrives through p_center.w.
                    if (static_cast<int>(std::lround(r.misc[2])) != ps.out_vertices) vd.fail(std::format("TES gl_PatchVerticesIn {} != TCS output patch size {}", r.misc[2], ps.out_vertices));
                    const std::size_t first = k * static_cast<std::size_t>(ps.patch_size);
                    const dm::Vec3 c = patch_center(ps, first);
                    if (std::abs(r.center[0] - c.x) > 1e-5f || std::abs(r.center[1] - c.y) > 1e-5f || std::abs(r.center[2] - c.z) > 1e-5f)
                        vd.fail("patch centroid (written by invocation 0 after barrier()) differs");
                    if (static_cast<int>(std::lround(r.center[3])) != ps.patch_size) vd.fail("per-patch output p_center.w differs");
                    if (std::abs(r.misc[3] - ps.verts[first].tag) > 0.0f) vd.fail("t_tag[0] differs");
                    const dm::Vec4 want = cpu_clip(ps, first, v.domain, {r.coord[0], r.coord[1], r.coord[2]}, mvp);
                    worst_pos = std::max({worst_pos, static_cast<double>(std::abs(r.pos[0] - want.x)), static_cast<double>(std::abs(r.pos[1] - want.y)),
                                          static_cast<double>(std::abs(r.pos[2] - want.z)), static_cast<double>(std::abs(r.pos[3] - want.w))});
                    if (!vd.ok) break;
                }
                if (worst_pos > 2e-4) vd.fail(std::format("gl_Position differs from the CPU TES by {:.3g}", worst_pos));
                check.expect(vd.ok, std::format("{}: gl_PrimitiveID, TES-visible tess levels, gl_PatchVerticesIn, barrier()-computed centroid and gl_Position match the CPU", label),
                             vd.ok ? std::format("max |gl_Position error| {:.2g}", worst_pos) : vd.detail);
            }

            // ---- every patch of a multi-patch draw is tessellated on its own levels --------------------------------
            if (patch_count > 1)
            {
                std::vector<int> got(patch_count, 0);
                for (const Record& r : cap.records) ++got[static_cast<std::size_t>(std::lround(r.coord[3]))];
                Verdict vd;
                std::string list;
                for (std::size_t k = 0; k < patch_count; ++k)
                {
                    const Counts c = counts_for(v.domain, plans[k]);
                    const int want = (v.point ? c.vertices : c.primitives) * per;
                    list += std::format("{}{}", k == 0 ? "" : ", ", want / per);
                    if (got[k] != want) vd.fail(std::format("patch {}: {} records, expected {}", k, got[k], want));
                }
                check.expect(vd.ok, std::format("{}: patch k is tessellated with the levels of its gl_PrimitiveID: {} primitives per patch", label, list), vd.detail);
            }

            // ---- tessellation coordinates -------------------------------------------------------------------
            {
                Verdict vd;
                int bad_w = 0, bad_sum = 0, bad_range = 0, bad_inv = 0;
                for (const Record& r : cap.records)
                {
                    const float u = r.coord[0], vv = r.coord[1], w = r.coord[2];
                    if (v.domain == Domain::Triangles) { if (std::abs(u + vv + w - 1.0f) > 1e-5f) ++bad_sum; }
                    else if (w != 0.0f) ++bad_w;
                    for (const float x : {u, vv, w})
                    {
                        if (!(x >= 0.0f && x <= 1.0f)) ++bad_range;
                        const float twice = 1.0f - (1.0f - x);  // GLSL ES 3.20 section 7.1.3: gl_TessCoord obeys x == 1.0 - (1.0 - x)
                        if (twice != x) ++bad_inv;
                    }
                }
                if (bad_w) vd.fail(std::format("{} records with w != 0 (quads and isolines: w is always zero)", bad_w));
                if (bad_sum) vd.fail(std::format("{} records with u + v + w != 1", bad_sum));
                if (bad_range) vd.fail(std::format("{} coordinates outside [0, 1]", bad_range));
                check.expect(vd.ok, std::format("{}: gl_TessCoord in [0,1], {}", label, v.domain == Domain::Triangles ? "u + v + w = 1" : "w = 0"), vd.detail);
                if (bad_inv != 0)
                    std::cout << std::format("note: {} {} gl_TessCoord values violate x == 1.0 - (1.0 - x) in float arithmetic\n", label, bad_inv);
                tess_coord_inverse_violations_ += bad_inv;
                tess_coord_values_ += static_cast<int>(cap.records.size()) * 3;
            }

            // ---- vertex set ---------------------------------------------------------------------------------
            const std::vector<Coord> have = distinct_coords(cap.records);
            if (patch_count == 1)
            {
                Verdict vd;
                if (static_cast<int>(have.size()) != total.vertices) vd.fail(std::format("{} distinct vertices, CPU model {}", have.size(), total.vertices));
                double err = 0.0;
                if (v.spacing == Spacing::Equal)
                {
                    const std::vector<Coord> want = expected_vertices(v.domain, plans[0]);
                    if (static_cast<int>(want.size()) != total.vertices) vd.fail("internal: vertex formula and vertex set of the CPU model disagree");
                    err = max_match_error(want, have);
                    if (err > kPositionTolerance) vd.fail(std::format("an expected vertex is missing (nearest captured vertex {:.3g} away)", err));
                    const double back = max_match_error(have, want);
                    if (back > kPositionTolerance) vd.fail(std::format("a captured vertex is not in the expected set ({:.3g} away)", back));
                    max_vertex_error_ = std::max({max_vertex_error_, err, back});
                }
                check.expect(vd.ok, std::format("{}: {} distinct vertices{}", label, total.vertices, v.spacing == Spacing::Equal ? " and every one at its CPU-computed gl_TessCoord" : " (count from the rounded levels)"),
                             vd.ok ? (v.spacing == Spacing::Equal ? std::format("max deviation {:.2g}", err) : std::string{}) : vd.detail);
            }

            // ---- triangles: winding and coverage --------------------------------------------------------------
            if (!v.point && v.domain != Domain::Isolines && patch_count == 1)
            {
                const TriStats s = triangle_stats(cap.records);
                const double domain_area = v.domain == Domain::Triangles ? 0.5 : 1.0;
                const bool ccw = v.order == Order::Ccw;
                Verdict vd;
                if (s.nondegenerate != total.primitives) vd.fail(std::format("{} non-degenerate triangles, CPU model {} ({} degenerate)", s.nondegenerate, total.primitives, s.degenerate));
                if (!winding_matches(s, v.order)) vd.fail(std::format("{} triangles with the wrong winding", ccw ? s.negative : s.positive));
                if (std::abs(s.abs_sum - domain_area) > 1e-3 * domain_area) vd.fail(std::format("triangle areas sum to {:.6f}, domain area {}", s.abs_sum, domain_area));
                check.expect(vd.ok, std::format("{}: {} triangles, all {} in (u,v) space, areas cover the domain exactly once", label, total.primitives, ccw ? "counter-clockwise" : "clockwise"),
                             vd.ok ? std::format("area sum {:.6f} of {}, {} degenerate", s.abs_sum, domain_area, s.degenerate) : vd.detail);
            }
        }

        // Fractional spacing structure of the outer edges of a captured triangle/quad patch or of an isoline.
        void verify_edges(demo::Check& check, const std::string& label, const Variant& v, const Plan& plan, const Capture& cap)
        {
            const bool fractional = v.spacing != Spacing::Equal;
            std::vector<std::pair<std::string, EdgeShape>> shapes;
            if (v.domain == Domain::Isolines)
            {
                // Every isoline: the u positions of its segments' end points (grouped by v).
                std::map<long, std::vector<double>> lines;
                for (const Record& r : cap.records) lines[std::lround(static_cast<double>(r.coord[1]) * 1e4)].push_back(r.coord[0]);
                bool lines_ok = static_cast<int>(lines.size()) == plan.outer[0];
                EdgeShape worst;
                worst.segment_count_ok = worst.symmetric = worst.lengths_ok = true;
                worst.end_to_middle = 1.0;
                for (auto& [key, us] : lines)
                {
                    std::sort(us.begin(), us.end());
                    us.erase(std::unique(us.begin(), us.end(), [](double a, double b) { return std::abs(a - b) < kDedupe; }), us.end());
                    const EdgeShape s = edge_shape(us, plan.outer[1], fractional);
                    worst.segment_count_ok = worst.segment_count_ok && s.segment_count_ok;
                    worst.symmetric = worst.symmetric && s.symmetric;
                    worst.lengths_ok = worst.lengths_ok && s.lengths_ok;
                    worst.end_to_middle = s.end_to_middle;
                }
                shapes.emplace_back(std::format("{} isolines with {} segments", plan.outer[0], plan.outer[1]), worst);
                check.expect(lines_ok, label + ": number of isolines (distinct v) equals the rounded first outer level", std::format("{} vs {}", lines.size(), plan.outer[0]));
            }
            else
            {
                const std::vector<Coord> have = distinct_coords(cap.records);
                if (v.domain == Domain::Triangles)
                {
                    shapes.emplace_back("u = 0 edge", edge_shape(edge_positions(have, 0, 0.0, 1), plan.outer[0], fractional));
                    shapes.emplace_back("v = 0 edge", edge_shape(edge_positions(have, 1, 0.0, 0), plan.outer[1], fractional));
                    shapes.emplace_back("w = 0 edge", edge_shape(edge_positions(have, 2, 0.0, 0), plan.outer[2], fractional));
                }
                else
                {
                    shapes.emplace_back("u = 0 edge", edge_shape(edge_positions(have, 0, 0.0, 1), plan.outer[0], fractional));
                    shapes.emplace_back("v = 0 edge", edge_shape(edge_positions(have, 1, 0.0, 0), plan.outer[1], fractional));
                    shapes.emplace_back("u = 1 edge", edge_shape(edge_positions(have, 0, 1.0, 1), plan.outer[2], fractional));
                    shapes.emplace_back("v = 1 edge", edge_shape(edge_positions(have, 1, 1.0, 0), plan.outer[3], fractional));
                }
            }
            Verdict vd;
            std::string measured;
            for (const auto& [name, s] : shapes)
            {
                if (!s.segment_count_ok) vd.fail(name + ": wrong number of segments");
                if (!s.symmetric) vd.fail(std::format("{}: not symmetric (max asymmetry {:.3g})", name, s.max_asym));
                if (!s.lengths_ok) vd.fail(name + ": not n - 2 segments of one length plus two additional segments of equal length");
                measured += std::format(" {}={:.3f}", name.substr(0, 6), s.end_to_middle);
            }
            check.expect(vd.ok, std::format("{}: edges are cut into n segments, symmetric, {}", label, fractional ? "n - 2 equal ones plus two equal end segments" : "all equal"),
                         vd.ok ? "end/middle length:" + measured : vd.detail);
        }

        void verify_variant_queries(demo::Check& check, const Variant& v, int out_vertices, const Prog& prog)
        {
            auto geti = [&](mgl::ProgramParameter p) {
                GLint x = -1;
                mgl::glGetProgramiv(prog.id, p, &x);
                return x;
            };
            const GLenum mode = static_cast<GLenum>(v.domain == Domain::Triangles ? mgl::TessGenMode::Triangles : v.domain == Domain::Quads ? mgl::TessGenMode::Quads : mgl::TessGenMode::Isolines);
            const GLenum spacing = static_cast<GLenum>(v.spacing == Spacing::Equal ? mgl::TessGenSpacing::Equal : v.spacing == Spacing::Odd ? mgl::TessGenSpacing::FractionalOdd : mgl::TessGenSpacing::FractionalEven);
            const GLenum order = static_cast<GLenum>(v.order == Order::Ccw ? mgl::FrontFace::CCW : mgl::FrontFace::CW);
            Verdict vd;
            if (geti(mgl::ProgramParameter::TessControlOutputVertices) != out_vertices) vd.fail("TESS_CONTROL_OUTPUT_VERTICES");
            if (static_cast<GLenum>(geti(mgl::ProgramParameter::TessGenMode)) != mode) vd.fail("TESS_GEN_MODE");
            if (static_cast<GLenum>(geti(mgl::ProgramParameter::TessGenSpacing)) != spacing) vd.fail("TESS_GEN_SPACING");
            if (static_cast<GLenum>(geti(mgl::ProgramParameter::TessGenVertexOrder)) != order) vd.fail("TESS_GEN_VERTEX_ORDER");
            if (geti(mgl::ProgramParameter::TessGenPointMode) != (v.point ? GL_TRUE : GL_FALSE)) vd.fail("TESS_GEN_POINT_MODE");
            check.expect(vd.ok, std::format("program queries for {}: TESS_CONTROL_OUTPUT_VERTICES = {}, TESS_GEN_MODE/SPACING/VERTEX_ORDER/POINT_MODE match the layout", label_of(v), out_vertices),
                         vd.detail);
        }

        // Runs a draw of a variant with the given levels and verifies it.
        void capture_and_verify(demo::Check& check, const std::string& what, const Variant& v, const Levels& levels, const PatchSet& ps,
                                int vertex_count, std::size_t patches, float bias, bool verify_structure)
        {
            const Prog& p = program({v, ps.out_vertices, false});
            const Capture cap = run_capture(p, ps, vertex_count, levels, bias, capture_mode(v), true);
            const std::string label = std::format("{} [{}]", what, label_of(v));
            verify_capture(check, label, v, levels, ps, bias, cap, patches);
            if (verify_structure && patches == 1 && !v.point)
            {
                const Plan plan = plan_for(v.domain, v.spacing, levels, max_level_);
                if (!plan.discarded) verify_edges(check, label, v, plan, cap);
            }
        }

        void check_gallery_pixels(demo::Env& env, demo::Check& check);
        void check_queries(demo::Check& check);
        void check_equal_spacing(demo::Check& check);
        void check_fractional_spacing(demo::Check& check);
        void check_level_rules(demo::Check& check);
        void check_point_mode(demo::Check& check);
        void check_patches(demo::Check& check);
        void check_errors(demo::Check& check);
        void check_culling(demo::Env& env, demo::Check& check);

        // The line (isoline) end-segment ratio e / m of the first isoline for one second outer level.
        double isoline_end_ratio(Spacing spacing, float outer1)
        {
            const Variant v{Domain::Isolines, spacing, Order::Ccw, false};
            const Prog& p = program({v, 4, false});
            Levels l;
            l.outer = {1.0f, outer1, 1.0f, 1.0f};
            const Capture cap = run_capture(p, irregular_quad_, 4, l, 0.0f, capture_mode(v), true);
            std::vector<double> us;
            for (const Record& r : cap.records) us.push_back(r.coord[0]);
            std::sort(us.begin(), us.end());
            us.erase(std::unique(us.begin(), us.end(), [](double a, double b) { return std::abs(a - b) < kDedupe; }), us.end());
            if (us.size() < 4) return -1.0;
            return edge_shape(us, static_cast<int>(us.size()) - 1, true).end_to_middle;
        }

        static constexpr double kPositionTolerance = 1e-3;  // see validate(): gl_TessCoord precision is not specified

        int max_level_ = 64;
        int max_patch_vertices_ = 32;
        float max_point_size_ = 1.0f;
        float angle_ = 0.0f;
        float level_scale_ = 1.0f;
        bool wire_ = true;
        bool dots_on_ = true;
        bool auto_rotate_ = true;
        double max_vertex_error_ = 0.0;
        int tess_coord_inverse_violations_ = 0;
        int tess_coord_values_ = 0;

        std::array<CellSpec, 9> cells_{};
        std::string src_vert_, src_tesc_, src_tese_, src_geom_, src_surface_, src_solid_;
        std::map<ProgramKey, Prog> programs_;
        mgl::ProgramId dots_program_{};
        mgl::UniformLocation dots_u_size_, dots_u_color_;

        PatchSet display_tri_, display_quad_, irregular_tri_, irregular_quad_, multi_, resample_, max_patch_;
        mgl::BufferId tf_buffer_{};
        mgl::VertexArrayId vao_patch_{}, vao_dots_{};
        mgl::QueryId q_written_{}, q_generated_{};
    };

    // =============================================================================================================
    // validate()
    // =============================================================================================================
    void TessellationDomains::validate(demo::Env& env, demo::Check& check)
    {
        // The implementation limits behind the clamping rules (ES 3.2 Table 21.45).
        check.expect(max_level_ >= 64, std::format("MAX_TESS_GEN_LEVEL = {} (Table 21.45: at least 64)", max_level_));
        check.expect(max_patch_vertices_ >= 32, std::format("MAX_PATCH_VERTICES = {} (Table 21.45: at least 32)", max_patch_vertices_));

        check_gallery_pixels(env, check);  // first: it reads the frame render() has just drawn
        check_queries(check);
        check_equal_spacing(check);
        check_fractional_spacing(check);
        check_level_rules(check);
        check_point_mode(check);
        check_patches(check);
        check_errors(check);
        check_culling(env, check);

        std::cout << std::format("note: largest gl_TessCoord deviation from the exact equal_spacing positions: {:.3g} "
                                 "(sanity tolerance {}; the spec gives no precision)\n", max_vertex_error_, kPositionTolerance);
        std::cout << std::format("note: gl_TessCoord values violating x == 1.0 - (1.0 - x) (GLSL ES 3.20 section 7.1.3): {} of {}\n",
                                 tess_coord_inverse_violations_, tess_coord_values_);
        check.expect(tess_coord_inverse_violations_ == 0,
                     "gl_TessCoord obeys x == 1.0 - (1.0 - x) for every captured value (GLSL ES 3.20 section 7.1.3)",
                     std::format("{} of {} values violate it", tess_coord_inverse_violations_, tess_coord_values_));

        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 3);
        render(env, 0);  // leave the gallery in the default framebuffer for the capture
    }

    // ---- pixels of the gallery ---------------------------------------------------------------------------------
    void TessellationDomains::check_gallery_pixels(demo::Env& env, demo::Check& check)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        const int W = env.width;
        const int H = env.height;
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, W, H);
        auto at = [&](int x, int y) { return &px[(static_cast<std::size_t>(y) * static_cast<std::size_t>(W) + static_cast<std::size_t>(x)) * 4u]; };
        auto is = [](const std::uint8_t* p, const demo::Rgba8& c, int tol) {
            for (int i = 0; i < 4; ++i)
                if (std::abs(static_cast<int>(p[i]) - static_cast<int>(c[static_cast<std::size_t>(i)])) > tol) return false;
            return true;
        };
        auto is_line = [](const std::uint8_t* p) { return p[1] > p[0] + 30 && p[1] > p[2] + 50 && p[1] > 80; };
        for (int i = 0; i < 9; ++i)
        {
            const CellSpec& c = cells_[static_cast<std::size_t>(i)];
            const Rect r = cell_rect(W, H, i % 3, i / 3);
            const dm::Mat4 mvp = cell_mvp(c.variant.domain);
            const std::string cell = std::format("cell {} ({})", i, label_of(c.variant));
            const bool iso = c.variant.domain == Domain::Isolines;

            {   // The patch does not reach the corners of its cell: they keep the cell's clear colour. This is also the
                // negative control of every "is drawn here" probe below.
                bool ok = true;
                for (const int x : {r.x + 1, r.x + r.w - 2})
                    for (const int y : {r.y + 1, r.y + r.h - 2}) ok = ok && is(at(x, y), kBackground, 1);
                check.expect(ok, cell + ": the four corner pixels of the cell show the clear colour");
            }
            if (!iso)
            {
                int front = 0, back = 0;
                for (int y = r.y; y < r.y + r.h; ++y)
                    for (int x = r.x; x < r.x + r.w; ++x)
                    {
                        const Tint t = classify(at(x, y));
                        front += t == Tint::Front;
                        back += t == Tint::Back;
                    }
                const bool ccw = c.variant.order == Order::Ccw;
                const int enough = r.w * r.h / 16;  // a visible part of the cell shows the surface
                check.expect(ccw ? (front >= enough && back == 0) : (back >= enough && front == 0),
                             cell + std::format(": the filled triangles are {} (the generated triangles are {}, gl_FrontFacing = {})",
                                                ccw ? "teal" : "orange", ccw ? "counter-clockwise" : "clockwise", ccw ? "true" : "false"),
                             std::format("{} front-facing pixels, {} back-facing pixels", front, back));
            }
            if (c.variant.spacing != Spacing::Equal) continue;

            // equal_spacing: the exact vertices are known, so the dots (point_mode, TF capture, re-drawn as points) must
            // sit on their CPU-projected positions.
            const PatchSet& ps = c.variant.domain == Domain::Triangles ? display_tri_ : display_quad_;
            const Plan plan = plan_for(c.variant.domain, c.variant.spacing, c.levels, max_level_);
            const std::vector<Coord> want = expected_vertices(c.variant.domain, plan);
            int missing = 0;
            for (const Coord& w : want)
            {
                const auto win = to_window(cpu_clip(ps, 0, c.variant.domain, w, mvp), r);
                const int x = static_cast<int>(std::floor(win[0]));
                const int y = static_cast<int>(std::floor(win[1]));
                if (x < r.x || x >= r.x + r.w || y < r.y || y >= r.y + r.h || !is(at(x, y), kDotColor, 2)) ++missing;
            }
            check.expect(missing == 0, cell + std::format(": all {} generated vertices appear as white dots at their CPU-projected pixels", want.size()),
                         std::format("{} of {} missing", missing, want.size()));

            if (iso)
            {
                // Isolines: n horizontal lines at v = k / n, each cut into m segments. The middle of one segment of every
                // line must be on a drawn line (3 x 3 neighbourhood: diamond-exit rasterization of a one pixel wide line),
                // the pixel in the middle between two lines must not be a line pixel.
                const int n = plan.outer[0];
                const int m = plan.outer[1];
                auto near_line = [&](double wx, double wy) {
                    const int cx = static_cast<int>(std::floor(wx));
                    const int cy = static_cast<int>(std::floor(wy));
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            if (is_line(at(cx + dx, cy + dy))) return true;
                    return false;
                };
                int on_missing = 0, off_hits = 0;
                const double um = (static_cast<double>(m / 2) + 0.5) / m;  // the middle of a segment, away from the dots
                for (int k = 0; k < n; ++k)
                {
                    const auto on = to_window(cpu_clip(ps, 0, c.variant.domain, {um, static_cast<double>(k) / n, 0.0}, mvp), r);
                    if (!near_line(on[0], on[1])) ++on_missing;
                    if (k + 1 < n)
                    {
                        const auto off = to_window(cpu_clip(ps, 0, c.variant.domain, {um, (static_cast<double>(k) + 0.5) / n, 0.0}, mvp), r);
                        if (is_line(at(static_cast<int>(std::floor(off[0])), static_cast<int>(std::floor(off[1]))))) ++off_hits;
                    }
                }
                check.expect(on_missing == 0 && off_hits == 0,
                             cell + std::format(": {} horizontal isolines at v = k/{} are drawn, and nothing is drawn halfway between them", n, n),
                             std::format("{} lines missing, {} spurious hits", on_missing, off_hits));
            }
        }
    }

    // ---- program queries ---------------------------------------------------------------------------------------
    void TessellationDomains::check_queries(demo::Check& check)
    {
        for (const CellSpec& c : cells_)
        {
            const int out = standard_out_vertices(c.variant.domain);
            verify_variant_queries(check, c.variant, out, program({c.variant, out, false}));
            Variant pv = c.variant;
            pv.point = true;
            verify_variant_queries(check, pv, out, program({pv, out, false}));
        }
        // A layout that names only the domain gets the defaults: equal_spacing and ccw (GLSL ES 3.20 section 4.4.1).
        const Variant d{Domain::Triangles, Spacing::Equal, Order::Ccw, false, true};
        verify_variant_queries(check, d, 3, program({d, 3, false}));
    }

    // ---- equal_spacing: exact ----------------------------------------------------------------------------------
    void TessellationDomains::check_equal_spacing(demo::Check& check)
    {
        struct Case
        {
            const char* name;
            Domain domain;
            Levels levels;
        };
        auto lv = [](std::array<float, 4> o, std::array<float, 2> i) { return Levels{o, i}; };
        const Case cases[] = {
            {"triangle: outer (3,4,5), inner 5", Domain::Triangles, lv({3, 4, 5, 1}, {5, 1})},
            {"triangle: every level 1 gives a single triangle", Domain::Triangles, lv({1, 1, 1, 1}, {1, 1})},
            {"triangle: inner 1 with outer 2 is treated as 1 + epsilon", Domain::Triangles, lv({2, 2, 2, 1}, {1, 1})},
            {"triangle: levels round up (2.1, 3.0, 3.5 / 2.2)", Domain::Triangles, lv({2.1f, 3.0f, 3.5f, 1}, {2.2f, 1})},
            {"triangle: inner 3 leaves a non-degenerate inner triangle", Domain::Triangles, lv({6, 5, 4, 1}, {3, 1})},
            {"triangle: inner 4 (ring of two segments, then the centre)", Domain::Triangles, lv({4, 4, 4, 1}, {4, 1})},
            {"triangle: outer 0.5 / 0.2 clamp to 1 (not discarded), inner 6.2", Domain::Triangles, lv({0.5f, 1, 0.2f, 1}, {6.2f, 1})},
            {"triangle: a negative inner level is clamped, not discarded", Domain::Triangles, lv({4, 4, 4, 1}, {-3, 1})},
            {"triangle: outer[3] = 0 is irrelevant", Domain::Triangles, lv({4, 4, 4, 0}, {4, 0})},
            {"triangle: outer (9,2,5), inner 8", Domain::Triangles, lv({9, 2, 5, 1}, {8, 1})},
            {"quad: outer (2,3,4,5), inner (4,6)", Domain::Quads, lv({2, 3, 4, 5}, {4, 6})},
            {"quad: every level 1 gives one triangle pair", Domain::Quads, lv({1, 1, 1, 1}, {1, 1})},
            {"quad: inner (1,3): the 1 is treated as 1 + epsilon", Domain::Quads, lv({4, 4, 4, 4}, {1, 3})},
            {"quad: inner (2,2), a single inner point", Domain::Quads, lv({3, 5, 2, 6}, {2, 2})},
            {"quad: inner (3,5), outer (4,1,4,1)", Domain::Quads, lv({4, 1, 4, 1}, {3, 5})},
            {"quad: inner (7,4), Figure 11.3(b)", Domain::Quads, lv({7, 7, 7, 7}, {7, 4})},
            {"quad: inner (4,2), Figure 11.3(a), degenerate inner rectangle", Domain::Quads, lv({4, 4, 4, 4}, {4, 2})},
            {"quad: outer 1, inner (1,5)", Domain::Quads, lv({1, 1, 1, 1}, {1, 5})},
            {"quad: outer (2.2,3.7,1.1,4.5), inner (-1,0.5)", Domain::Quads, lv({2.2f, 3.7f, 1.1f, 4.5f}, {-1, 0.5f})},
            {"isolines: (1,3), Figure 11.4(a)", Domain::Isolines, lv({1, 3, 1, 1}, {1, 1})},
            {"isolines: (4,6), Figure 11.4(b)", Domain::Isolines, lv({4, 6, 1, 1}, {1, 1})},
            {"isolines: (3,1)", Domain::Isolines, lv({3, 1, 1, 1}, {1, 1})},
            {"isolines: (2.5, 4.2) round up to (3, 5)", Domain::Isolines, lv({2.5f, 4.2f, 1, 1}, {1, 1})},
            {"isolines: outer[2] and outer[3] are irrelevant", Domain::Isolines, lv({3, 4, 0, 0}, {0, 0})},
        };
        for (const Case& cs : cases)
        {
            const Variant v{cs.domain, Spacing::Equal, cs.domain == Domain::Quads ? Order::Cw : Order::Ccw, false};
            const PatchSet& ps = cs.domain == Domain::Triangles ? irregular_tri_ : irregular_quad_;
            capture_and_verify(check, cs.name, v, cs.levels, ps, ps.patch_size, 1, 0.0f, false);
        }

        // The winding layout: the same patch with cw instead of ccw gives the same vertices and the opposite winding.
        {
            const Levels l = lv({3, 4, 5, 1}, {5, 1});
            capture_and_verify(check, "triangle: the winding layout flips", {Domain::Triangles, Spacing::Equal, Order::Cw, false}, l, irregular_tri_, 3, 1, 0.0f, false);
            capture_and_verify(check, "quad: the winding layout flips", {Domain::Quads, Spacing::Equal, Order::Ccw, false}, lv({2, 3, 4, 5}, {4, 6}), irregular_quad_, 4, 1, 0.0f, false);
        }

        // IL0 divides the u axis and IL1 the v axis (Figure 11.1 and 11.3). Positive control and mutation (swapped levels).
        {
            const Variant v{Domain::Quads, Spacing::Equal, Order::Cw, false};
            const Levels l = lv({4, 4, 4, 4}, {7, 3});
            const Capture cap = run_capture(program({v, 4, false}), irregular_quad_, 4, l, 0.0f, capture_mode(v), true);
            const std::vector<Coord> have = distinct_coords(cap.records);
            Plan plan = plan_for(Domain::Quads, Spacing::Equal, l, max_level_);
            const double right = max_match_error(expected_vertices(Domain::Quads, plan), have);
            std::swap(plan.inner[0], plan.inner[1]);
            const double wrong = max_match_error(expected_vertices(Domain::Quads, plan), have);
            check.expect(right < kPositionTolerance, "quad inner levels (7,3): the inner grid has 7 segments along u and 3 along v (Figure 11.3: IL0 horizontal, IL1 vertical)",
                         std::format("max deviation {:.3g}", right));
            check.expect(wrong > 0.05, "negative control: the CPU model with IL0 and IL1 swapped does not match the captured vertices",
                         std::format("deviation {:.3g}", wrong));
        }
        {
            const Variant v{Domain::Triangles, Spacing::Equal, Order::Ccw, false};
            const Capture cap = run_capture(program({v, 3, false}), irregular_tri_, 3, lv({3, 4, 5, 1}, {5, 1}), 0.0f, capture_mode(v), true);
            const TriStats s = triangle_stats(cap.records);
            check.expect(winding_matches(s, Order::Ccw) && !winding_matches(s, Order::Cw),
                         "negative control: a counter-clockwise capture is accepted as ccw and rejected as cw",
                         std::format("{} positive, {} negative", s.positive, s.negative));
        }
    }

    // ---- fractional spacings: only what the text defines -------------------------------------------------------
    void TessellationDomains::check_fractional_spacing(demo::Check& check)
    {
        // The gallery's own cells: counts from the rounded levels, edges cut into n segments with two equal, symmetric
        // end segments.
        for (std::size_t i = 0; i < cells_.size(); ++i)
        {
            const CellSpec& c = cells_[i];
            if (c.variant.spacing == Spacing::Equal) continue;
            const PatchSet& ps = c.variant.domain == Domain::Triangles ? irregular_tri_ : irregular_quad_;
            capture_and_verify(check, std::format("gallery cell {}", i), c.variant, c.levels, ps, ps.patch_size, 1, 0.0f, true);
        }

        // Rounding of the level to n: written out by hand from the text (isolines with one line: n segments = n lines written).
        {
            struct Round
            {
                Spacing spacing;
                float level;
                int n;
            };
            const Round table[] = {
                {Spacing::Equal, 0.2f, 1}, {Spacing::Equal, 1.0f, 1}, {Spacing::Equal, 1.01f, 2}, {Spacing::Equal, 2.5f, 3}, {Spacing::Equal, 3.0f, 3},
                {Spacing::Odd, 1.0f, 1}, {Spacing::Odd, 1.5f, 3}, {Spacing::Odd, 2.0f, 3}, {Spacing::Odd, 3.0f, 3}, {Spacing::Odd, 3.01f, 5}, {Spacing::Odd, 4.9f, 5},
                {Spacing::Even, 1.0f, 2}, {Spacing::Even, 2.0f, 2}, {Spacing::Even, 2.01f, 4}, {Spacing::Even, 3.9f, 4}, {Spacing::Even, 4.0f, 4}, {Spacing::Even, 5.0f, 6},
            };
            for (const Spacing sp : {Spacing::Equal, Spacing::Odd, Spacing::Even})
            {
                const Variant v{Domain::Isolines, sp, Order::Ccw, false};
                std::string bad;
                int rows = 0;
                for (const Round& r : table)
                {
                    if (r.spacing != sp) continue;
                    ++rows;
                    Levels l;
                    l.outer = {1.0f, r.level, 1.0f, 1.0f};
                    const Capture cap = run_capture(program({v, 4, false}), irregular_quad_, 4, l, 0.0f, capture_mode(v), false);
                    if (static_cast<int>(cap.written) != r.n) bad += std::format(" f={}: {} segments, expected {};", r.level, cap.written, r.n);
                }
                check.expect(bad.empty(), std::format("{}: second outer level -> number of segments of an isoline for {} levels, rounded as section 11.2.2 prescribes", spacing_name(sp), rows), bad);
            }
        }

        // When n - f = 0 the additional segments have the length of the others; the ratio falls monotonically with n - f.
        {
            struct Series
            {
                Spacing spacing;
                std::array<float, 5> levels;  // n - f = 0, 0.5, 1, 1.5, 1.8 for n = 7 (odd) / 6 (even)
            };
            for (const Series& s : {Series{Spacing::Odd, {7.0f, 6.5f, 6.0f, 5.5f, 5.2f}}, Series{Spacing::Even, {6.0f, 5.5f, 5.0f, 4.5f, 4.2f}}})
            {
                std::array<double, 5> ratio{};
                for (std::size_t i = 0; i < 5; ++i) ratio[i] = isoline_end_ratio(s.spacing, s.levels[i]);
                bool decreasing = true;
                for (std::size_t i = 0; i + 1 < 5; ++i) decreasing = decreasing && ratio[i + 1] < ratio[i] - 1e-3;
                check.expect(std::abs(ratio[0] - 1.0) < 1e-3, std::format("{}: n - f = 0 (f = {}): the two additional segments have the length of the others", spacing_name(s.spacing), s.levels[0]),
                             std::format("e/m = {:.4f}", ratio[0]));
                check.expect(decreasing && ratio[4] > 0.0 && ratio[4] < 0.5,
                             std::format("{}: the additional/other segment length ratio decreases monotonically with n - f (0, .5, 1, 1.5, 1.8) and stays positive", spacing_name(s.spacing)),
                             std::format("e/m = {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}", ratio[0], ratio[1], ratio[2], ratio[3], ratio[4]));
            }
        }

        // "... must be identical for any pair of subdivided edges with identical values of f": opposite quad edges, equal levels.
        {
            const Variant v{Domain::Quads, Spacing::Odd, Order::Ccw, false};
            const Capture cap = run_capture(program({v, 4, false}), irregular_quad_, 4, Levels{{5.4f, 4.6f, 5.4f, 3.3f}, {4.4f, 5.6f}}, 0.0f, capture_mode(v), true);
            const std::vector<Coord> have = distinct_coords(cap.records);
            const std::vector<double> left = edge_positions(have, 0, 0.0, 1);
            const std::vector<double> right = edge_positions(have, 0, 1.0, 1);
            double worst = 0.0;
            for (std::size_t i = 0; i < std::min(left.size(), right.size()); ++i) worst = std::max(worst, std::abs(left[i] - right[i]));
            check.expect(left.size() == right.size() && left.size() == 8 && worst < 1e-3,
                         "fractional_odd quad, outer levels 5.4 on both the u = 0 and u = 1 edges: both edges are cut at identical positions (7 segments)",
                         std::format("{} and {} vertices, max difference {:.3g}", left.size(), right.size(), worst));
        }
    }

    // ---- rules for the levels ----------------------------------------------------------------------------------
    void TessellationDomains::check_level_rules(demo::Check& check)
    {
        auto lv = [](std::array<float, 4> o, std::array<float, 2> i) { return Levels{o, i}; };
        {
            // "A patch is discarded ... if any relevant outer tessellation level is less than or equal to zero."
            struct Case
            {
                const char* name;
                Domain d;
                Levels l;
            };
            const Case cases[] = {
                {"triangle outer[0] = 0", Domain::Triangles, lv({0, 4, 4, 1}, {4, 1})},
                {"triangle outer[2] = -1", Domain::Triangles, lv({4, 4, -1, 1}, {4, 1})},
                {"quad outer[2] = 0", Domain::Quads, lv({4, 4, 0, 4}, {4, 4})},
                {"quad outer[3] = -2", Domain::Quads, lv({4, 4, 4, -2}, {4, 4})},
                {"isolines outer[0] = 0", Domain::Isolines, lv({0, 4, 1, 1}, {1, 1})},
                {"isolines outer[1] = 0", Domain::Isolines, lv({3, 0, 1, 1}, {1, 1})},
            };
            std::string bad;
            for (const Case& cs : cases)
            {
                const Variant v{cs.d, Spacing::Equal, Order::Ccw, false};
                const PatchSet& ps = cs.d == Domain::Triangles ? irregular_tri_ : irregular_quad_;
                const Capture cap = run_capture(program({v, ps.out_vertices, false}), ps, ps.patch_size, cs.l, 0.0f, capture_mode(v), false);
                if (cap.written != 0 || cap.generated != 0) bad += std::format(" {}: {} written, {} generated;", cs.name, cap.written, cap.generated);
            }
            check.expect(bad.empty(), "a patch with a relevant outer level <= 0 is discarded: no primitives generated, none written (6 cases)", bad);
        }
        // Irrelevant levels never discard (triangles: outer[3], inner[1]; isolines: outer[2], outer[3], inner).
        capture_and_verify(check, "outer[3] = 0 and inner[1] = 0 do not discard a triangle", {Domain::Triangles, Spacing::Equal, Order::Ccw, false}, lv({4, 4, 4, 0}, {4, 0}), irregular_tri_, 3, 1, 0.0f, false);
        capture_and_verify(check, "outer[2] = outer[3] = 0 do not discard isolines", {Domain::Isolines, Spacing::Equal, Order::Ccw, false}, lv({3, 4, 0, 0}, {0, 0}), irregular_quad_, 4, 1, 0.0f, false);
        // A negative inner level "will not cause a patch to be discarded; ... clamped".
        capture_and_verify(check, "a negative inner level of a quad is clamped", {Domain::Quads, Spacing::Equal, Order::Cw, false}, lv({4, 4, 4, 4}, {-5, -0.5f}), irregular_quad_, 4, 1, 0.0f, false);

        // Clamping to MAX_TESS_GEN_LEVEL: [1, max] equal, [1, max - 1] odd, [2, max] even. Expected values come from the
        // queried maximum, independently of plan_for().
        {
            struct Clamp
            {
                Spacing spacing;
                Levels l;
                int expected_lines;
                const char* what;
            };
            const int mx = max_level_;
            const Clamp cases[] = {
                {Spacing::Equal, lv({static_cast<float>(mx) + 100.5f, 2, 1, 1}, {1, 1}), mx * 2, "equal: MAX_TESS_GEN_LEVEL + 100.5 lines clamp to MAX_TESS_GEN_LEVEL lines of 2 segments"},
                {Spacing::Odd, lv({1, static_cast<float>(mx) + 10.0f, 1, 1}, {1, 1}), mx - 1, "fractional_odd: segments clamp to MAX_TESS_GEN_LEVEL - 1"},
                {Spacing::Even, lv({1, static_cast<float>(mx) + 10.0f, 1, 1}, {1, 1}), mx, "fractional_even: segments clamp to MAX_TESS_GEN_LEVEL"},
            };
            for (const Clamp& c : cases)
            {
                const Variant v{Domain::Isolines, c.spacing, Order::Ccw, false};
                const Capture cap = run_capture(program({v, 4, false}), irregular_quad_, 4, c.l, 0.0f, capture_mode(v), false);
                check.expect(static_cast<int>(cap.written) == c.expected_lines && static_cast<int>(cap.generated) == c.expected_lines,
                             std::string("isolines, ") + c.what, std::format("{} written, expected {} (MAX_TESS_GEN_LEVEL = {})", cap.written, c.expected_lines, mx));
            }
        }
    }

    // ---- point mode --------------------------------------------------------------------------------------------
    void TessellationDomains::check_point_mode(demo::Check& check)
    {
        for (std::size_t i = 0; i < cells_.size(); ++i)
        {
            const CellSpec& c = cells_[i];
            Variant pv = c.variant;
            pv.point = true;
            const PatchSet& ps = c.variant.domain == Domain::Triangles ? irregular_tri_ : irregular_quad_;
            const Capture pts = run_capture(program({pv, ps.out_vertices, false}), ps, ps.patch_size, c.levels, 0.0f, capture_mode(pv), true);
            verify_capture(check, std::format("gallery cell {} [{}]", i, label_of(pv)), pv, c.levels, ps, 0.0f, pts, 1);
            if (c.variant.spacing != Spacing::Equal) continue;
            // "... one point for each distinct vertex produced by tessellation": the same vertices as the triangles/lines.
            const Capture prim = run_capture(program({c.variant, ps.out_vertices, false}), ps, ps.patch_size, c.levels, 0.0f, capture_mode(c.variant), true);
            const std::vector<Coord> a = distinct_coords(prim.records);
            const std::vector<Coord> b = distinct_coords(pts.records);
            const double ab = max_match_error(a, b);
            const double ba = max_match_error(b, a);
            check.expect(a.size() == b.size() && ab < kPositionTolerance && ba < kPositionTolerance,
                         std::format("gallery cell {}: point_mode emits exactly the {} distinct vertices of the triangles/lines, one point each", i, a.size()),
                         std::format("{} vs {} vertices, deviation {:.3g}", a.size(), b.size(), std::max(ab, ba)));
        }
    }

    // ---- patches, patch sizes, per-patch data ------------------------------------------------------------------
    void TessellationDomains::check_patches(demo::Check& check)
    {
        // Three patches in one draw, levels grow with gl_PrimitiveID (all levels 1, 2, 3); the two vertices after the last
        // complete patch are ignored (section 10.1.12). The per-patch outputs (centroid, tag) differ per patch.
        capture_and_verify(check, "3 patches + 2 stray vertices (PATCH_VERTICES = 4, 14 vertices)", {Domain::Quads, Spacing::Equal, Order::Cw, false},
                           Levels{{1, 1, 1, 1}, {1, 1}}, multi_, static_cast<int>(multi_.verts.size()), 3, 1.0f, false);
        // Input patch of 5 vertices, output patch of 3 (the TCS's layout(vertices = 3)), 13 vertices = 2 patches + 3 strays.
        capture_and_verify(check, "PATCH_VERTICES = 5 with a TCS output patch of 3 (13 vertices)", {Domain::Triangles, Spacing::Equal, Order::Ccw, false},
                           Levels{{2, 2, 2, 1}, {2, 1}}, resample_, static_cast<int>(resample_.verts.size()), 2, 0.0f, false);
        // The largest legal patch, input and output: PATCH_VERTICES = MAX_PATCH_VERTICES, layout(vertices = MAX_PATCH_VERTICES).
        {
            const Variant v{Domain::Quads, Spacing::Equal, Order::Cw, false};
            const Prog& p = program({v, max_patch_vertices_, false});
            GLint out_vertices = -1;
            mgl::glGetProgramiv(p.id, mgl::ProgramParameter::TessControlOutputVertices, &out_vertices);
            check.expect(out_vertices == max_patch_vertices_, std::format("layout(vertices = MAX_PATCH_VERTICES) links: TESS_CONTROL_OUTPUT_VERTICES = {}", out_vertices));
            capture_and_verify(check, std::format("PATCH_VERTICES = MAX_PATCH_VERTICES = {} (one patch)", max_patch_vertices_), v,
                               Levels{{2, 2, 2, 2}, {2, 2}}, max_patch_, max_patch_vertices_, 1, 0.0f, false);
        }
        // Layout declarations that name only the domain: equal_spacing, ccw.
        capture_and_verify(check, "layout(triangles) only: the defaults are equal_spacing and ccw", {Domain::Triangles, Spacing::Equal, Order::Ccw, false, true},
                           cells_[0].levels, irregular_tri_, 3, 1, 0.0f, false);
    }

    // ---- defined errors ----------------------------------------------------------------------------------------
    void TessellationDomains::check_errors(demo::Check& check)
    {
        const Variant tri{Domain::Triangles, Spacing::Equal, Order::Ccw, false};
        const Prog& tess = program({tri, 3, false});
        mgl::glUseProgram(tess.id);
        set_levels(tess, cells_[0].levels, 0.0f, cell_mvp());
        bind_patches(irregular_tri_);
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 3);

        // ES 3.2 section 11.2: tessellation active and the mode is not PATCHES / not active and the mode is PATCHES.
        (void)demo::drain_gl_errors();
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDrawArrays(TRIANGLES) with a tessellation program: INVALID_OPERATION (section 11.2)");
        mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 3);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDrawArrays(POINTS) with a tessellation program: INVALID_OPERATION");
        mgl::glUseProgram(dots_program_);
        mgl::glDrawArrays(mgl::PrimitiveType::Patches, 0, 3);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDrawArrays(PATCHES) without a tessellation program: INVALID_OPERATION (section 11.2)");

        // PATCH_VERTICES range (section 10.1.12). A failed call changes nothing: the next draw still has 4-vertex patches.
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 4);
        check.expect_no_gl_error("glPatchParameteri(PATCH_VERTICES, 4) is legal");
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, max_patch_vertices_ + 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glPatchParameteri(PATCH_VERTICES, MAX_PATCH_VERTICES + 1): INVALID_VALUE");
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 0);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glPatchParameteri(PATCH_VERTICES, 0): INVALID_VALUE");
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, -1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glPatchParameteri(PATCH_VERTICES, -1): INVALID_VALUE");
        {
            const Variant quad{Domain::Quads, Spacing::Equal, Order::Cw, false};
            const Capture cap = run_capture(program({quad, 4, false}), irregular_quad_, 4, Levels{{2, 2, 2, 2}, {2, 2}}, 0.0f, capture_mode(quad), true, false);
            check.expect(!cap.records.empty() && std::lround(cap.records.front().center[3]) == 4,
                         "after the failed glPatchParameteri calls PATCH_VERTICES is still 4 (gl_PatchVerticesIn in the TCS)",
                         cap.records.empty() ? "no records" : std::format("gl_PatchVerticesIn = {}", cap.records.front().center[3]));
        }
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, max_patch_vertices_);
        check.expect_no_gl_error("glPatchParameteri(PATCH_VERTICES, MAX_PATCH_VERTICES) is legal");
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 3);

        // GetProgramiv: tessellation queries need the matching shader (section 7.12).
        {
            GLint x = 0;
            mgl::glGetProgramiv(dots_program_, mgl::ProgramParameter::TessControlOutputVertices, &x);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TESS_CONTROL_OUTPUT_VERTICES queried on a program without a tessellation control shader: INVALID_OPERATION");
            mgl::glGetProgramiv(dots_program_, mgl::ProgramParameter::TessGenMode, &x);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TESS_GEN_MODE queried on a program without a tessellation evaluation shader: INVALID_OPERATION");
            mgl::glGetProgramiv(dots_program_, mgl::ProgramParameter::TessGenPointMode, &x);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TESS_GEN_POINT_MODE queried on a program without a tessellation evaluation shader: INVALID_OPERATION");
        }

        // Transform feedback: "If a tessellation evaluation shader is active, the type of primitive emitted by that shader is
        // used instead of the mode parameter ... for the purposes of this error check" (section 12.2.2, Table 12.1).
        {
            struct Mismatch
            {
                Variant v;
                mgl::TransformFeedbackPrimitive mode;
                const char* what;
            };
            const Mismatch cases[] = {
                {{Domain::Isolines, Spacing::Equal, Order::Ccw, false}, mgl::TransformFeedbackPrimitive::Triangles, "isolines program, BeginTransformFeedback(TRIANGLES)"},
                {{Domain::Triangles, Spacing::Equal, Order::Ccw, false}, mgl::TransformFeedbackPrimitive::Lines, "triangles program, BeginTransformFeedback(LINES)"},
                {{Domain::Quads, Spacing::Equal, Order::Cw, true}, mgl::TransformFeedbackPrimitive::Triangles, "quads program in point_mode, BeginTransformFeedback(TRIANGLES)"},
                {{Domain::Quads, Spacing::Equal, Order::Cw, false}, mgl::TransformFeedbackPrimitive::Points, "quads program, BeginTransformFeedback(POINTS)"},
            };
            for (const Mismatch& m : cases)
            {
                const PatchSet& ps = m.v.domain == Domain::Triangles ? irregular_tri_ : irregular_quad_;
                const Prog& p = program({m.v, ps.out_vertices, false});
                mgl::glUseProgram(p.id);
                set_levels(p, cells_[0].levels, 0.0f, cell_mvp());
                bind_patches(ps);
                mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, ps.patch_size);
                mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
                mgl::glEnable(mgl::Capability::RasterizerDiscard);
                (void)demo::drain_gl_errors();
                mgl::glBeginTransformFeedback(m.mode);
                mgl::glDrawArrays(mgl::PrimitiveType::Patches, 0, ps.patch_size);
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, std::string(m.what) + ": the draw raises INVALID_OPERATION (Table 12.1)");
                mgl::glEndTransformFeedback();
                mgl::glDisable(mgl::Capability::RasterizerDiscard);
                (void)demo::drain_gl_errors();
            }
        }

        // Programs that must not build (GLSL ES 3.20 sections 4.4.1 and 4.4.2). The failing stage is not prescribed: compile or link.
        {
            auto builds = [&](const std::string& tesc_defines, const std::string& tese_defines) {
                std::string log;
                const mgl::ProgramId id = try_build_tess({tri, 3, false}, tesc_defines, tese_defines, false, log);
                if (id.value != 0) mgl::glDeleteProgram(id);
                return id.value != 0;
            };
            const std::string tese_ok = standard_tese_defines(tri);
            check.expect(!builds(standard_tesc_defines(max_patch_vertices_ + 1), tese_ok), "layout(vertices = MAX_PATCH_VERTICES + 1) does not build (compile- or link-time error)");
            check.expect(!builds(standard_tesc_defines(0), tese_ok), "layout(vertices = 0) does not build (compile- or link-time error)");
            check.expect(!builds(standard_tesc_defines(3), "#define TESS_LAYOUT equal_spacing, ccw\n#define DOMAIN_TRIANGLES 1\n"),
                         "a tessellation evaluation shader without a primitive mode (triangles/quads/isolines) does not build");
            check.expect(builds(standard_tesc_defines(3), tese_ok), "positive control: the same sources with a valid layout build");
        }
        (void)demo::drain_gl_errors();
    }

    // ---- the generated winding as seen by the rasterizer ---------------------------------------------------------
    void TessellationDomains::check_culling(demo::Env& env, demo::Check& check)
    {
        const dm::Mat4 mvp = cell_mvp();
        const Rect full{0, 0, env.width, env.height};
        const auto center = to_window(cpu_clip(display_quad_, 0, Domain::Quads, {0.5, 0.5, 0.0}, mvp), full);
        const int cx = static_cast<int>(std::floor(center[0]));
        const int cy = static_cast<int>(std::floor(center[1]));
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glDisable(mgl::Capability::ScissorTest);
        mgl::glViewport(0, 0, env.width, env.height);
        for (const Order order : {Order::Ccw, Order::Cw})
        {
            for (const mgl::FrontFace front : {mgl::FrontFace::CCW, mgl::FrontFace::CW})
            {
                const Variant v{Domain::Quads, Spacing::Equal, order, false};
                const Prog& p = program({v, 4, false});
                mgl::glClearColor(kBackground[0] / 255.0f, kBackground[1] / 255.0f, kBackground[2] / 255.0f, 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
                mgl::glEnable(mgl::Capability::DepthTest);
                mgl::glEnable(mgl::Capability::CullFace);
                mgl::glCullFace(mgl::CullFace::Back);
                mgl::glFrontFace(front);
                mgl::glUseProgram(p.id);
                set_levels(p, Levels{{4, 4, 4, 4}, {4, 4}}, 0.0f, mvp);
                mgl::glUniform3f(p.u_front, kFront[0], kFront[1], kFront[2]);
                mgl::glUniform3f(p.u_back, kBack[0], kBack[1], kBack[2]);
                mgl::glUniform1f(p.u_height, kDome);
                mgl::glUniform1i(p.u_lines, 0);
                draw_patches(display_quad_, 4);
                const demo::Rgba8 px = demo::read_pixel(cx, cy);
                const bool visible = classify(px.data()) != Tint::None;
                const bool expect_visible = (order == Order::Ccw) == (front == mgl::FrontFace::CCW);
                check.expect(visible == expect_visible,
                             std::format("CULL_FACE(BACK), FrontFace({}), generated triangles {}: the patch is {}", front == mgl::FrontFace::CCW ? "CCW" : "CW",
                                         order == Order::Ccw ? "ccw" : "cw", expect_visible ? "drawn" : "culled"),
                             std::format("pixel {}", demo::format_rgba(px)));
            }
        }
        mgl::glFrontFace(mgl::FrontFace::CCW);
        mgl::glDisable(mgl::Capability::CullFace);
        mgl::glDisable(mgl::Capability::DepthTest);
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-tessellation-domains",
        .title = "Tessellation domains, spacings and winding",
        .min_es = {3, 2},
        .summary = "Triangle, quad and isoline tessellation with equal / fractional odd / fractional even spacing, cw/ccw and point_mode; "
                   "the TES output is captured with transform feedback and compared with a CPU implementation of ES 3.2 section 11.2.2.",
        .controls = "  a  toggle the camera rotation\n  w  toggle the wireframe\n  p  toggle the vertex dots\n  +/- scale all tessellation levels\n  r  reset",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<TessellationDomains>(argc, argv, info);
}
