// es30-multisample-default-framebuffer
//
// A multisampled *default* framebuffer (OpenGL ES 3.0 sections 3.3, 4.1.10, 4.3.2, 4.3.3). The runtime asks
// EGL for a pbuffer config with 4 samples (demo::Info::samples = 4). The window then has a multisample buffer
// next to its color buffer: every pixel stores SAMPLES color samples, rasterization computes per-sample
// coverage, and "for window system-provided framebuffers, [the resolve] occurs each time a pixel is updated,
// so the antialiasing appears to be automatic at the application level" (3.3).
//
// The picture is a rotating cube with six flat-colored faces on a dark background, drawn with back-face
// culling. Every face color has the same channel sum (R + G + B = 450), so if the resolve averages the samples, a
// resolved pixel's channel sum says how much of it the cube covers, whichever faces those are.
//
// ES 3.0 does not say how the samples are combined: "The method of combination is not specified" (4.1.10), and a
// resolve blit of normalized colors is "implementation-dependent" (4.3.3). No resolved color is asserted, not even
// for pixels whose samples are all equal. Validation, against a CPU projection of the same cube:
//   * SAMPLE_BUFFERS = 1 and SAMPLES >= 1 for the default framebuffer, 0 / 0 while a single-sample framebuffer
//     object is the draw framebuffer; the back buffer is RGBA8 (attachment queries); with an ES 3.1+ context,
//     glGetMultisamplefv returns SAMPLES positions in [0, 1] and INVALID_VALUE for index = SAMPLES;
//   * glReadPixels from the multisampled default framebuffer raises no error (the INVALID_OPERATION rule of
//     4.3.2 is for framebuffer *objects*);
//   * per sample, on the window, without any resolve: the cube is drawn with stencil INCR and ANY_SAMPLES_PASSED
//     queries with stencil-tested polygons show that no sample is covered twice (3.6.1 shared edges), that no
//     sample inside the silhouette is missed, and that none outside it is covered; positive controls show that
//     each query can return TRUE;
//   * the same cube in a single-sample framebuffer object (fully specified): exact background and face pixels,
//     every seam pixel exactly one face, every silhouette pixel the background or one face;
//   * glBlitFramebuffer from the default framebuffer into an RGBA8 framebuffer object with identical rectangles
//     raises no error; an identical-bounds sub-rectangle writes only that rectangle, with the values of the
//     whole resolve; source and destination both mirrored give the whole resolve, not its mirror image;
//   * printed only: the window and blit images classified against the projection (bounds, seam sums of 450,
//     blended silhouette pixels, coverage against the silhouette area), and how the two resolves compare;
//   * INVALID_OPERATION (and nothing copied) for a resolve with a translated, scaled or mirrored destination
//     rectangle, into RGB565 (formats not identical), and for a blit *into* the multisampled default
//     framebuffer; glReadPixels from a multisample framebuffer object is INVALID_OPERATION, from the default
//     framebuffer it is not.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;
using demo::Rgba8;

namespace
{
    constexpr int kMinSize = 96;          // smaller surfaces make the edge statistics meaningless
    constexpr int kRequestedSamples = 4;  // demo::Info::samples
    constexpr mgl::ClearBufferBit kColorBit = mgl::ClearBufferBit::Color;
    constexpr mgl::BlitFilter kNearest = mgl::BlitFilter::Nearest;
    const mgl::FramebufferId kDefaultFbo{0};

    constexpr int gl(GLenum e) { return static_cast<int>(e); }

    // ---------------------------------------------------------------------------------------------
    // Colors. All values are k / 255, so they are exact in an 8-bit buffer. Every face has R + G + B = 450,
    // the background 84: a blend of any faces keeps the sum 450, a blend with the background lies between.
    // ---------------------------------------------------------------------------------------------
    constexpr std::array<Rgba8, 6> kFace = {{
        {230, 140, 80, 255},  // +X
        {80, 230, 140, 255},  // -X
        {140, 80, 230, 255},  // +Y
        {220, 200, 30, 255},  // -Y
        {30, 220, 200, 255},  // +Z
        {200, 30, 220, 255},  // -Z
    }};
    constexpr Rgba8 kBackground = {20, 24, 40, 255};
    constexpr Rgba8 kSentinel = {255, 0, 255, 255};  // blit destination before a blit: no cube pixel is magenta
    constexpr int kFaceSum = 450;
    constexpr int kBackgroundSum = 84;
    // Converting a resolved (averaged) value to 8 bits errs by less than one step per channel, so a channel sum
    // errs by less than 3 (2 after rounding to an integer with round-to-nearest; 3 allows truncation).
    constexpr int kSumTolerance = 3;

    int channel_sum(const Rgba8& c) { return int{c[0]} + int{c[1]} + int{c[2]}; }

    float unorm(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

    // ---------------------------------------------------------------------------------------------
    // The cube: 6 faces x 2 triangles, counter-clockwise seen from outside (front faces under FrontFace CCW).
    // ---------------------------------------------------------------------------------------------
    using Corner = std::array<float, 3>;
    constexpr std::array<std::array<Corner, 4>, 6> kFaces = {{
        {{{1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}}},        // +X
        {{{-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}}},    // -X
        {{{-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, 1, -1}}},        // +Y
        {{{-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}}},    // -Y
        {{{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}}},        // +Z
        {{{-1, -1, -1}, {-1, 1, -1}, {1, 1, -1}, {1, -1, -1}}},    // -Z
    }};

    struct Vertex
    {
        float x, y, z;
        float r, g, b, a;
    };

    std::vector<Vertex> cube_vertices()
    {
        std::vector<Vertex> v;
        for (std::size_t f = 0; f < kFaces.size(); ++f)
            for (const std::size_t i : {0u, 1u, 2u, 0u, 2u, 3u})
            {
                const Corner& p = kFaces[f][i];
                const Rgba8& c = kFace[f];
                v.push_back({p[0], p[1], p[2], unorm(c[0]), unorm(c[1]), unorm(c[2]), unorm(c[3])});
            }
        return v;
    }

    /// The rotation of frame `frame` (deterministic animation time frame / 60 s).
    float cube_angle(int frame) { return 0.55f + 0.6f * static_cast<float>(frame) / 60.0f; }

    /// projection * view * model. The projection keeps the cube's size relative to the smaller surface side, so
    /// it fills any surface shape without being cut off.
    dm::Mat4 cube_mvp(int width, int height, int frame)
    {
        const float m = static_cast<float>(std::min(width, height));
        const dm::Mat4 fit = dm::scale({m / static_cast<float>(width), m / static_cast<float>(height), 1.0f});
        const dm::Mat4 projection = fit * dm::perspective(dm::radians(40.0f), 1.0f, 1.0f, 20.0f);
        const dm::Mat4 view = dm::translate({0.0f, 0.0f, -7.0f});
        const dm::Mat4 model = dm::rotate(cube_angle(frame), {1.0f, 0.7f, 0.25f}) * dm::rotate(0.4f, {0.0f, 1.0f, 0.0f});
        return projection * view * model;
    }

    // ---------------------------------------------------------------------------------------------
    // 2D geometry in window coordinates (pixel (x, y) covers [x, x + 1) x [y, y + 1)).
    // ---------------------------------------------------------------------------------------------
    struct P2
    {
        double x, y;
    };
    using Poly = std::vector<P2>;  // convex, counter-clockwise

    double cross(P2 o, P2 a, P2 b) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); }

    double signed_area(const Poly& p)
    {
        double a = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i)
        {
            const P2 u = p[i], v = p[(i + 1) % p.size()];
            a += u.x * v.y - v.x * u.y;
        }
        return 0.5 * a;
    }

    bool point_inside(const Poly& p, P2 q)
    {
        for (std::size_t i = 0; i < p.size(); ++i)
            if (cross(p[i], p[(i + 1) % p.size()], q) < 0.0) return false;
        return true;
    }

    struct Box
    {
        double x0, y0, x1, y1;
        [[nodiscard]] std::array<P2, 4> corners() const { return {{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}}}; }
    };

    bool box_inside(const Poly& p, const Box& b)
    {
        for (const P2& c : b.corners())
            if (!point_inside(p, c)) return false;
        return true;
    }

    /// Separating-axis test of two convex shapes: the polygon's edge lines and the box's axes.
    bool box_disjoint(const Poly& p, const Box& b)
    {
        const auto corners = b.corners();
        for (std::size_t i = 0; i < p.size(); ++i)
        {
            bool all_outside = true;
            for (const P2& c : corners) all_outside = all_outside && cross(p[i], p[(i + 1) % p.size()], c) < 0.0;
            if (all_outside) return true;
        }
        double min_x = 1e30, max_x = -1e30, min_y = 1e30, max_y = -1e30;
        for (const P2& v : p)
        {
            min_x = std::min(min_x, v.x);
            max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y);
            max_y = std::max(max_y, v.y);
        }
        return b.x1 <= min_x || b.x0 >= max_x || b.y1 <= min_y || b.y0 >= max_y;
    }

    Poly convex_hull(std::vector<P2> pts)
    {
        std::ranges::sort(pts, [](P2 a, P2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
        Poly hull(2 * pts.size());
        std::size_t k = 0;
        for (std::size_t i = 0; i < pts.size(); ++i)
        {
            while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.0) --k;
            hull[k++] = pts[i];
        }
        for (std::size_t i = pts.size() - 1, t = k + 1; i > 0; --i)
        {
            while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i - 1]) <= 0.0) --k;
            hull[k++] = pts[i - 1];
        }
        hull.resize(k - 1);
        return hull;
    }

    /// The convex counter-clockwise polygon `p` with every edge moved by `d` pixels along its inward normal (d < 0 grows it):
    /// each new corner is the intersection of two neighbouring moved edge lines.
    Poly offset_polygon(const Poly& p, double d)
    {
        const std::size_t n = p.size();
        std::vector<P2> base(n), dir(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const P2 a = p[i], b = p[(i + 1) % n];
            const double len = std::hypot(b.x - a.x, b.y - a.y);
            const P2 u{(b.x - a.x) / len, (b.y - a.y) / len};
            dir[i] = u;
            base[i] = {a.x - u.y * d, a.y + u.x * d};  // the inward normal of a CCW edge is (-u.y, u.x)
        }
        Poly out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t j = (i + n - 1) % n;  // the edge ending at corner i
            // base[j] + s * dir[j] = base[i] + t * dir[i]
            const double den = dir[j].x * dir[i].y - dir[j].y * dir[i].x;
            const double s = ((base[i].x - base[j].x) * dir[i].y - (base[i].y - base[j].y) * dir[i].x) / den;
            out[i] = {base[j].x + s * dir[j].x, base[j].y + s * dir[j].y};
        }
        return out;
    }

    double distance_to_segment(P2 p, P2 a, P2 b)
    {
        const double dx = b.x - a.x, dy = b.y - a.y;
        const double t = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy), 0.0, 1.0);
        return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
    }

    // ---------------------------------------------------------------------------------------------
    // The CPU projection of the cube and the per-pixel expectation derived from it.
    // ---------------------------------------------------------------------------------------------
    struct Geometry
    {
        std::vector<Poly> faces;       // front faces in window coordinates
        std::vector<std::size_t> ids;  // their index into kFace
        std::vector<P2> corners;       // all 8 projected cube corners
        Poly hull;                     // the silhouette
        double area = 0.0, perimeter = 0.0;
    };

    Geometry project(const dm::Mat4& mvp, int width, int height)
    {
        auto window = [&](const Corner& c) {
            const dm::Vec4 clip = mvp * dm::Vec4{c[0], c[1], c[2], 1.0f};
            const double w = static_cast<double>(clip.w);
            return P2{(static_cast<double>(clip.x) / w + 1.0) * width * 0.5, (static_cast<double>(clip.y) / w + 1.0) * height * 0.5};
        };
        Geometry g;
        for (std::size_t f = 0; f < kFaces.size(); ++f)
        {
            Poly p;
            for (const Corner& c : kFaces[f]) p.push_back(window(c));
            for (const P2& q : p) g.corners.push_back(q);
            if (signed_area(p) > 0.0)  // counter-clockwise in window coordinates: front-facing (3.6.1, FrontFace CCW)
            {
                g.faces.push_back(p);
                g.ids.push_back(f);
            }
        }
        g.hull = convex_hull(g.corners);
        g.area = signed_area(g.hull);
        for (std::size_t i = 0; i < g.hull.size(); ++i)
            g.perimeter += std::hypot(g.hull[(i + 1) % g.hull.size()].x - g.hull[i].x, g.hull[(i + 1) % g.hull.size()].y - g.hull[i].y);
        return g;
    }

    enum class Kind : std::uint8_t
    {
        Exterior,    // no sample can be covered: the background exactly
        Face,        // every sample is covered by one face: that face's color exactly
        Seam,        // every sample is covered by some face: channel sum 450, channels between the faces' colors
        Silhouette,  // the rest: channels between the background and the faces the pixel touches
    };

    struct Expect
    {
        Kind kind = Kind::Exterior;
        Rgba8 lo{}, hi{};       // per-channel bounds of the colors that can reach the pixel
        bool centred = false;   // the pixel centre is within 0.1 pixel of the silhouette, far from corners
        std::uint8_t faces = 0; // bit f: face f touches the (grown) pixel square
        bool near_corner = false;  // the pixel centre is within 1 pixel of a projected cube corner
    };

    /// The sample points of a pixel may lie outside its unit square (3.3), and vertex positions are snapped to the
    /// subpixel grid (SUBPIXEL_BITS >= 4): every classification uses the pixel square grown by half a pixel.
    constexpr double kMargin = 0.5;

    std::vector<Expect> expectations(const Geometry& g, int width, int height)
    {
        std::vector<Expect> out(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                Expect& e = out[static_cast<std::size_t>(y * width + x)];
                const Box b{x - kMargin, y - kMargin, x + 1 + kMargin, y + 1 + kMargin};
                if (box_disjoint(g.hull, b))
                {
                    e.kind = Kind::Exterior;
                    e.lo = e.hi = kBackground;
                    continue;
                }
                std::vector<std::size_t> touching;
                int inside_face = -1;
                for (std::size_t i = 0; i < g.faces.size(); ++i)
                {
                    if (box_disjoint(g.faces[i], b)) continue;
                    touching.push_back(g.ids[i]);
                    e.faces = static_cast<std::uint8_t>(e.faces | (1u << g.ids[i]));
                    if (box_inside(g.faces[i], b)) inside_face = static_cast<int>(g.ids[i]);
                }
                for (const P2& v : g.corners) e.near_corner = e.near_corner || std::hypot(v.x - (x + 0.5), v.y - (y + 0.5)) < 1.0;
                const bool in_hull = box_inside(g.hull, b);
                e.kind = inside_face >= 0 ? Kind::Face : in_hull ? Kind::Seam : Kind::Silhouette;
                e.lo = {255, 255, 255, 255};
                e.hi = {0, 0, 0, 0};
                auto include = [&](const Rgba8& c) {
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        e.lo[k] = std::min(e.lo[k], c[k]);
                        e.hi[k] = std::max(e.hi[k], c[k]);
                    }
                };
                if (inside_face >= 0)
                    include(kFace[static_cast<std::size_t>(inside_face)]);
                else
                    for (const std::size_t f : touching) include(kFace[f]);
                if (e.kind == Kind::Silhouette)
                {
                    include(kBackground);
                    // Centred on the silhouette: the edge halves the pixel, so a spread of sample points has samples on
                    // both sides. Corners are avoided (3 pixels), where an edge's neighbourhood is not a half-plane.
                    const P2 c{x + 0.5, y + 0.5};
                    bool near_corner = false;
                    for (const P2& v : g.corners) near_corner = near_corner || std::hypot(v.x - c.x, v.y - c.y) < 3.0;
                    if (!near_corner)
                        for (std::size_t i = 0; i < g.hull.size(); ++i)
                            e.centred = e.centred || distance_to_segment(c, g.hull[i], g.hull[(i + 1) % g.hull.size()]) <= 0.1;
                }
            }
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // Images in GL order (row 0 is the bottom row).
    // ---------------------------------------------------------------------------------------------
    struct Image
    {
        int w = 0, h = 0;
        std::vector<Rgba8> px;
        [[nodiscard]] const Rgba8& at(int x, int y) const { return px[static_cast<std::size_t>(y * w + x)]; }
    };

    Image read_image(int w, int h)
    {
        const std::vector<std::uint8_t> bytes = demo::read_rgba8(0, 0, w, h);
        Image im{w, h, std::vector<Rgba8>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h))};
        for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = {bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]};
        return im;
    }

    bool within(const Rgba8& c, const Rgba8& lo, const Rgba8& hi, int tolerance)
    {
        for (std::size_t k = 0; k < 4; ++k)
            if (int{c[k]} < int{lo[k]} - tolerance || int{c[k]} > int{hi[k]} + tolerance) return false;
        return true;
    }

    /// A pixel blended from background and cube: its channel sum lies strictly between theirs, beyond the
    /// quantization error of the sum.
    bool mixed(const Rgba8& c)
    {
        const int s = channel_sum(c);
        return s > kBackgroundSum + kSumTolerance && s < kFaceSum - kSumTolerance;
    }

    struct Region
    {
        int x0, y0, x1, y1;  // half-open
        [[nodiscard]] bool contains(int x, int y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; }
    };

    struct Counter
    {
        int total = 0, bad = 0;
        std::string first;
        void add(bool ok, int x, int y, const Rgba8& got, std::string_view want)
        {
            ++total;
            if (!ok && bad++ == 0) first = std::format("first at ({},{}): got {}, expected {}", x, y, demo::format_rgba(got), want);
        }
        [[nodiscard]] std::string detail() const
        {
            return bad == 0 ? std::format("{} pixels", total) : std::format("{} of {} pixels wrong, {}", bad, total, first);
        }
    };

    struct Stats
    {
        Counter exterior, face, seam_sum, seam_range, silhouette_range, centred;
        int mixed = 0;
        double covered = 0.0;  // Face/Seam pixels count 1, silhouette pixels their measured fraction
    };

    Stats analyze(const Image& img, const std::vector<Expect>& ex, const Region& r)
    {
        Stats s;
        for (int y = r.y0; y < r.y1; ++y)
            for (int x = r.x0; x < r.x1; ++x)
            {
                const Rgba8& got = img.at(x, y);
                const Expect& e = ex[static_cast<std::size_t>(y * img.w + x)];
                switch (e.kind)
                {
                case Kind::Exterior:
                    s.exterior.add(within(got, e.lo, e.hi, 1), x, y, got, demo::format_rgba(kBackground));
                    break;
                case Kind::Face:
                    s.face.add(within(got, e.lo, e.hi, 1), x, y, got, demo::format_rgba(e.lo));
                    s.covered += 1.0;
                    break;
                case Kind::Seam:
                    s.seam_sum.add(std::abs(channel_sum(got) - kFaceSum) <= kSumTolerance, x, y, got, std::format("R+G+B = {}", kFaceSum));
                    s.seam_range.add(within(got, e.lo, e.hi, 1), x, y, got, std::format("between {} and {}", demo::format_rgba(e.lo), demo::format_rgba(e.hi)));
                    s.covered += 1.0;
                    break;
                case Kind::Silhouette:
                    s.silhouette_range.add(within(got, e.lo, e.hi, 1) && channel_sum(got) >= kBackgroundSum - kSumTolerance &&
                                               channel_sum(got) <= kFaceSum + kSumTolerance,
                                           x, y, got, std::format("between {} and {}", demo::format_rgba(e.lo), demo::format_rgba(e.hi)));
                    s.covered += static_cast<double>(channel_sum(got) - kBackgroundSum) / (kFaceSum - kBackgroundSum);
                    s.mixed += mixed(got) ? 1 : 0;
                    if (e.centred) s.centred.add(mixed(got), x, y, got, "a blend of background and cube");
                    break;
                }
            }
        return s;
    }

    /// Prints what a weighted-average resolve would satisfy, measured on a resolved image. ES 3.0 says only that the samples
    /// are "combined" (4.1.10) or "resolved in an implementation-dependent manner" (4.3.3), so none of this is asserted.
    void print_resolved(std::string_view label, const Stats& s, double area)
    {
        std::cout << std::format("   note: observed {} (implementation-dependent resolve, not asserted): exterior pixels off the background: {} of {}; "
                                 "single-face pixels off their face color: {} of {}; seam pixels with R+G+B != 450 (+-3): {} of {}; seam pixels outside "
                                 "the range of their faces: {} of {}; silhouette pixels outside the range of background and faces: {} of {}; pixels "
                                 "centred on the silhouette not strictly blended: {} of {}; summed coverage {:.1f} px for an area of {:.1f} px ({:+.3f}%)\n",
                                 label, s.exterior.bad, s.exterior.total, s.face.bad, s.face.total, s.seam_sum.bad, s.seam_sum.total, s.seam_range.bad,
                                 s.seam_range.total, s.silhouette_range.bad, s.silhouette_range.total, s.centred.bad, s.centred.total, s.covered, area,
                                 100.0 * (s.covered - area) / area);
    }

    bool near(const Rgba8& a, const Rgba8& b) { return within(a, b, b, 1); }

    /// Single-sample rasterization is fully specified (ES 3.0 3.6.1): a pixel is produced when its centre lies inside a
    /// polygon, and a centre on an edge shared by two polygons is produced by exactly one of them. With flat, opaque colors a
    /// pixel is therefore exactly the background or exactly one face, never a blend. The shared-edge rule says nothing about a
    /// centre that lies exactly on a corner where several faces meet, so pixels within 1 pixel of a projected corner are left
    /// out of the seam and silhouette counts.
    struct SingleStats
    {
        Counter exterior, face, seam, silhouette;
        int corner_skipped = 0;
    };

    SingleStats analyze_single(const Image& img, const std::vector<Expect>& ex)
    {
        SingleStats s;
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x)
            {
                const Rgba8& got = img.at(x, y);
                const Expect& e = ex[static_cast<std::size_t>(y * img.w + x)];
                bool one_face = false;
                for (std::size_t f = 0; f < kFace.size(); ++f) one_face = one_face || (((e.faces >> f) & 1u) != 0 && near(got, kFace[f]));
                switch (e.kind)
                {
                case Kind::Exterior: s.exterior.add(near(got, kBackground), x, y, got, demo::format_rgba(kBackground)); break;
                case Kind::Face: s.face.add(near(got, e.lo), x, y, got, demo::format_rgba(e.lo)); break;
                case Kind::Seam:
                    if (e.near_corner) ++s.corner_skipped;
                    else s.seam.add(one_face, x, y, got, "exactly one of the faces it touches");
                    break;
                case Kind::Silhouette:
                    if (e.near_corner) ++s.corner_skipped;
                    else s.silhouette.add(one_face || near(got, kBackground), x, y, got, "the background or exactly one of the faces it touches");
                    break;
                }
            }
        return s;
    }

    template <typename F>
    bool expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // a stale error must not be mistaken for the one under test
        call();
        return check.expect_gl_error(code, what);
    }

    void bind_read(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, f); }
    void bind_draw(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, f); }
    void bind_both(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, f); }

    void blit(const Region& s, const Region& d)
    {
        mgl::glBlitFramebuffer(s.x0, s.y0, s.x1, s.y1, d.x0, d.y0, d.x1, d.y1, kColorBit, kNearest);
    }

    void clear_to(const Rgba8& c)
    {
        mgl::glClearColor(unorm(c[0]), unorm(c[1]), unorm(c[2]), unorm(c[3]));
        mgl::glClear(kColorBit);
    }

    int count_differences(const Image& a, const Image& b, int& max_diff)
    {
        int n = 0;
        max_diff = 0;
        for (std::size_t i = 0; i < a.px.size(); ++i)
        {
            int d = 0;
            for (std::size_t k = 0; k < 4; ++k) d = std::max(d, std::abs(int{a.px[i][k]} - int{b.px[i][k]}));
            max_diff = std::max(max_diff, d);
            n += d > 0 ? 1 : 0;
        }
        return n;
    }

    class MultisampleDefault final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kMinSize || env.height < kMinSize)
                throw demo::Skip(std::format("the edge statistics need a surface of at least {0}x{0} pixels", kMinSize));

            program_ = demo::build_program(env, "shaders/cube.vert", "shaders/cube.frag");
            u_mvp_ = mgl::glGetUniformLocation(program_, "u_mvp");
            if (u_mvp_.value < 0) throw demo::Failure("u_mvp not found");

            const std::vector<Vertex> vertices = cube_vertices();
            vertex_count_ = static_cast<GLsizei>(vertices.size());
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(vertices), mgl::BufferUsage::StaticDraw);
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            // The probing polygons of the per-sample checks: same vertex layout, contents uploaded per draw.
            mgl::glGenBuffers(1, &poly_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, poly_vbo_);
            mgl::glGenVertexArrays(1, &poly_vao_);
            mgl::glBindVertexArray(poly_vao_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glGenQueries(1, &query_);

            // Framebuffer objects for the checks; their storage is sized in validate() to the current surface.
            std::array<mgl::FramebufferId, 3> f{};
            mgl::glGenFramebuffers(f);
            single_ = f[0];
            rgb565_ = f[1];
            ms_fbo_ = f[2];
            std::array<mgl::RenderbufferId, 3> r{};
            mgl::glGenRenderbuffers(r);
            single_rb_ = r[0];
            rgb565_rb_ = r[1];
            ms_rb_ = r[2];

            // The cube is a closed convex solid: back-face culling alone removes the hidden faces.
            mgl::glEnable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);
            mgl::glFrontFace(mgl::FrontFace::CCW);
            // Dithering (on by default, 4.1.9) cannot change the exactly representable k / 255 colors; off for clarity.
            mgl::glDisable(mgl::Capability::Dither);
        }

        void render(demo::Env& env, int frame) override
        {
            frame_ = frame;
            bind_both(kDefaultFbo);
            draw_cube(env.width, env.height, frame);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            const int w = env.width, h = env.height;
            const Geometry geo = project(cube_mvp(w, h, frame_), w, h);
            const std::vector<Expect> ex = expectations(geo, w, h);
            const Region all{0, 0, w, h};
            allocate_targets(w, h);

            const int samples = check_queries(env, check);

            // ---- glReadPixels from the multisampled default framebuffer (4.3.2) --------------------------------
            bind_both(kDefaultFbo);
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadFramebufferBinding), 0, "READ_FRAMEBUFFER_BINDING is 0 (the default framebuffer)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), gl(GL_BACK), "READ_BUFFER of the default framebuffer is BACK");
            (void)demo::drain_gl_errors();
            const Image rp = read_image(w, h);
            check.expect_no_gl_error("glReadPixels from the multisampled default framebuffer raises no error (READ_FRAMEBUFFER_BINDING is zero)");
            check.expect(geo.area > 100.0 && count_kind(ex, Kind::Exterior) > 0 && count_kind(ex, Kind::Face) > 0 && count_kind(ex, Kind::Seam) > 0 &&
                             count_kind(ex, Kind::Silhouette) > 0,
                         "the projection yields exterior, single-face, seam and silhouette pixels (no check below is vacuous)");
            // The window holds resolved pixels. ES 3.0 4.1.10: "the sample values for each color in the multisample buffer are
            // combined to produce a single color value [...] The method of combination is not specified." Not even a pixel whose
            // samples are all equal is guaranteed to come back unchanged, so everything measured on the image is printed only.
            const Stats s_rp = analyze(rp, ex, all);
            std::cout << std::format("   note: {}x{}, SAMPLES {}; silhouette area {:.1f} px, perimeter {:.1f} px; {} exterior, {} face, {} seam, {} silhouette pixels "
                                     "({} centred), {} blended pixels in glReadPixels\n",
                                     w, h, samples, geo.area, geo.perimeter, s_rp.exterior.total, s_rp.face.total, s_rp.seam_sum.total,
                                     s_rp.silhouette_range.total, s_rp.centred.total, s_rp.mixed);
            print_resolved("glReadPixels of the window", s_rp, geo.area);

            // ---- the same cube in a single-sample framebuffer object: fully specified ------------------------------
            bind_both(single_);
            check.expect(demo::get_integer(mgl::GetParameter::SampleBuffers) == 0, "control framebuffer object: SAMPLE_BUFFERS = 0 (single-sample rasterization)");
            draw_cube(w, h, frame_);
            const Image control = read_image(w, h);
            const SingleStats s_ss = analyze_single(control, ex);
            check.expect(s_ss.exterior.bad == 0 && s_ss.face.bad == 0, "single-sample control: exterior pixels equal the background, single-face pixels their face (+-1)",
                         s_ss.exterior.detail() + "; " + s_ss.face.detail());
            check.expect(s_ss.seam.total > 0 && s_ss.seam.bad == 0,
                         "single-sample control: every seam pixel is exactly one of the faces it touches (shared edges: no crack, no blend)", s_ss.seam.detail());
            check.expect(s_ss.silhouette.total > 0 && s_ss.silhouette.bad == 0,
                         "single-sample control: every silhouette pixel is the background or exactly one face (no blend)", s_ss.silhouette.detail());
            std::cout << std::format("   note: single-sample control: {} pixels within 1 px of a projected corner left out; {} blended silhouette pixels "
                                     "(multisampled window: {})\n",
                                     s_ss.corner_skipped, analyze(control, ex, all).mixed, s_rp.mixed);

            // ---- per-sample checks on the window itself, with the stencil buffer and occlusion queries -------------------
            check_samples_with_stencil(check, geo, w, h);

            // ---- glBlitFramebuffer resolve of the default framebuffer (4.3.3) --------------------------------------
            prepare_single();
            bind_read(kDefaultFbo);
            bind_draw(single_);
            (void)demo::drain_gl_errors();
            blit(all, all);
            check.expect_no_gl_error("glBlitFramebuffer from the multisampled default framebuffer into an RGBA8 framebuffer object, identical rectangles: no error");
            const Image full = read_single(w, h);
            print_resolved("blit resolve", analyze(full, ex, all), geo.area);
            int max_diff = 0;
            const int differ = count_differences(full, rp, max_diff);
            std::cout << std::format("   note: observed: the blit resolve differs from glReadPixels in {} pixels (max {} per channel); both resolves are "
                                     "implementation-dependent and need not agree\n",
                                     differ, max_diff);

            // The resolves below are compared with `full`, the resolve of the same samples by the same command with the whole
            // rectangle. 4.3.3: "the samples corresponding to each pixel location in the source are converted to a single sample
            // before being written to the destination": a resolved pixel is a function of that pixel's samples, and Appendix
            // A.1 (repeatability) makes it the same function each time. So a pixel resolves to the same value however large
            // the identical-bounds rectangle is, whatever that value is.
            //
            // Sub-rectangle with identical bounds: only pixels whose centres lie in the rectangle are written.
            const Region part{w / 5, h / 4, w - w / 3, h - h / 5};
            prepare_single();
            bind_read(kDefaultFbo);
            bind_draw(single_);
            (void)demo::drain_gl_errors();
            blit(part, part);
            check.expect_no_gl_error("resolve of a sub-rectangle with identical bounds raises no error");
            const Image sub = read_single(w, h);
            check.expect(outside_is_sentinel(sub, part), "sub-rectangle resolve: pixels outside the rectangle keep the sentinel");
            check.expect(same_inside(sub, full, part), "sub-rectangle resolve: every pixel inside it equals the whole-surface resolve");

            // Both rectangles mirrored in x the same way: the bounds are identical (allowed), and "if both the source and
            // destination rectangles specify a negative width or height for the same direction, no reversal is performed".
            prepare_single();
            bind_read(kDefaultFbo);
            bind_draw(single_);
            (void)demo::drain_gl_errors();
            blit({w, 0, 0, h}, {w, 0, 0, h});
            check.expect_no_gl_error("resolve with source and destination both mirrored in x (identical bounds) raises no error");
            const Image both = read_single(w, h);
            check.expect(same_inside(both, full, all), "...and it does not mirror: every pixel equals the whole-surface resolve");
            // The comparison can fail: the single-sample image of the (asymmetric) cube differs from its own mirror image.
            check.expect(count_differences(control, mirrored_x(control), max_diff) > 100,
                         "negative control: the cube's single-sample image differs from its mirror image in more than 100 pixels (a mirrored copy is detectable)");
            std::cout << std::format("   note: observed: the whole-surface resolve differs from its mirror image in {} pixels\n",
                                     count_differences(full, mirrored_x(full), max_diff));

            check_errors(check, w, h, rp);

            // Leave the default framebuffer bound for the capture.
            bind_both(kDefaultFbo);
            mgl::glViewport(0, 0, w, h);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            bind_both(kDefaultFbo);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            const std::array<mgl::FramebufferId, 3> f = {single_, rgb565_, ms_fbo_};
            mgl::glDeleteFramebuffers(f);
            const std::array<mgl::RenderbufferId, 3> r = {single_rb_, rgb565_rb_, ms_rb_};
            mgl::glDeleteRenderbuffers(r);
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteVertexArrays(1, &poly_vao_);
            mgl::glDeleteBuffers(1, &poly_vbo_);
            mgl::glDeleteQueries(1, &query_);
            mgl::glDeleteProgram(program_);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glEnable(mgl::Capability::Dither);
        }

    private:
        void draw_cube(int w, int h, int frame) const
        {
            mgl::glViewport(0, 0, w, h);
            clear_to(kBackground);
            const dm::Mat4 mvp = cube_mvp(w, h, frame);
            mgl::glUseProgram(program_);
            mgl::glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
            mgl::glBindVertexArray(vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, vertex_count_);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void attach(mgl::FramebufferId f, mgl::RenderbufferId rb, std::string_view what) const
        {
            bind_both(f);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rb);
            const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (st != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{} framebuffer incomplete: {}", what, demo::framebuffer_status_name(st)));
        }

        /// Single-sample RGBA8 and RGB565 targets of the surface size, and a 16 x 16 four-sample RGBA8 target.
        void allocate_targets(int w, int h) const
        {
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, single_rb_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, w, h);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rgb565_rb_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgb565, w, h);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, ms_rb_);
            // MAX_SAMPLES is at least 4 (Table 6.35) and RGBA8 supports every count up to it (4.4.2.2).
            mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, kRequestedSamples, mgl::InternalFormat::Rgba8, 16, 16);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            attach(single_, single_rb_, "RGBA8");
            attach(rgb565_, rgb565_rb_, "RGB565");
            attach(ms_fbo_, ms_rb_, "multisample RGBA8");
            bind_both(ms_fbo_);
            clear_to(kBackground);
            bind_both(kDefaultFbo);
        }

        /// Fills the single-sample target with the sentinel so that "written" and "kept" cannot be confused.
        void prepare_single() const
        {
            bind_both(single_);
            clear_to(kSentinel);
        }

        Image read_single(int w, int h) const
        {
            bind_read(single_);
            Image im = read_image(w, h);
            bind_read(kDefaultFbo);
            return im;
        }

        static bool outside_is_sentinel(const Image& img, const Region& r)
        {
            for (int y = 0; y < img.h; ++y)
                for (int x = 0; x < img.w; ++x)
                    if (!r.contains(x, y) && img.at(x, y) != kSentinel) return false;
            return true;
        }

        static bool same_inside(const Image& a, const Image& b, const Region& r)
        {
            for (int y = r.y0; y < r.y1; ++y)
                for (int x = r.x0; x < r.x1; ++x)
                    if (a.at(x, y) != b.at(x, y)) return false;
            return true;
        }

        static Image mirrored_x(const Image& a)
        {
            Image m = a;
            for (int y = 0; y < a.h; ++y)
                for (int x = 0; x < a.w; ++x) m.px[static_cast<std::size_t>(y * a.w + x)] = a.at(a.w - 1 - x, y);
            return m;
        }

        static int count_kind(const std::vector<Expect>& ex, Kind k)
        {
            return static_cast<int>(std::ranges::count_if(ex, [k](const Expect& e) { return e.kind == k; }));
        }

        // ------------------------------------------------------------------------------------------
        // Per-sample checks on the multisampled window, without any resolve. The stencil buffer of the window lives in its
        // multisample buffer (3.3), the stencil test and stencil operations run per sample (4.1.10), and an ANY_SAMPLES_PASSED
        // query is TRUE "if any fragment or sample passes the depth test" (4.1.6). So "does any sample satisfy the stencil
        // condition C?" is answered exactly by drawing a polygon with the stencil test C inside a query. Every comparison
        // below uses the same sample points for the cube and for the probing polygon, whatever and wherever they are (3.3:
        // they "may be located inside or outside of the unit square"; ES 3.2 13.4: they "cannot be queried").
        void draw_cube_only(int w, int h) const
        {
            mgl::glViewport(0, 0, w, h);
            const dm::Mat4 mvp = cube_mvp(w, h, frame_);
            mgl::glUseProgram(program_);
            mgl::glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
            mgl::glBindVertexArray(vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, vertex_count_);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        /// A convex counter-clockwise polygon given in window coordinates (front-facing, so culling keeps it).
        void draw_polygon(const Poly& p, int w, int h) const
        {
            std::vector<Vertex> v;
            for (const P2& q : p)
                v.push_back({static_cast<float>(q.x / w * 2.0 - 1.0), static_cast<float>(q.y / h * 2.0 - 1.0), 0.0f, 1.0f, 1.0f, 1.0f, 1.0f});
            mgl::glViewport(0, 0, w, h);
            mgl::glUseProgram(program_);
            const dm::Mat4 identity = dm::Mat4::identity();
            mgl::glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, identity.data());
            mgl::glBindVertexArray(poly_vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, poly_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(v), mgl::BufferUsage::StreamDraw);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleFan, 0, static_cast<GLsizei>(v.size()));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        /// TRUE if any sample of polygon `p` passes the stencil test `func`(ref, stored).
        bool any_sample(const Poly& p, int w, int h, mgl::CompareFunc func, int ref) const
        {
            mgl::glStencilFunc(func, ref, 0xFFu);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glBeginQuery(mgl::QueryTarget::AnySamplesPassed, query_);
            draw_polygon(p, w, h);
            mgl::glEndQuery(mgl::QueryTarget::AnySamplesPassed);
            GLuint result = 0;
            mgl::glGetQueryObjectuiv(query_, mgl::QueryObjectParameter::Result, &result);
            return result != 0;
        }

        /// Clears the window's stencil samples to 0 and adds 1 to every sample the cube covers, `times` times.
        void count_cube_coverage(int w, int h, int times) const
        {
            mgl::glClearStencil(0);
            mgl::glClear(mgl::ClearBufferBit::Stencil);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFu);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Incr);
            for (int i = 0; i < times; ++i) draw_cube_only(w, h);
        }

        void check_samples_with_stencil(demo::Check& check, const Geometry& geo, int w, int h) const
        {
            bind_both(kDefaultFbo);
            GLint stencil_bits = 0;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::DrawFramebuffer, mgl::DefaultFramebufferQueryAttachment::Stencil,
                                                       mgl::FramebufferAttachmentParameter::StencilSize, &stencil_bits);
            if (stencil_bits < 2)
            {
                std::cout << std::format("   note: the window has {} stencil bits; the per-sample stencil checks need 2 and are not made\n", stencil_bits);
                return;
            }
            // Margins against vertex snapping (SUBPIXEL_BITS >= 4: a vertex moves by at most 1/32 pixel) and CPU/GPU rounding of
            // the projection: the probing polygons stay 0.75 pixel inside or outside the silhouette.
            const Poly inset = offset_polygon(geo.hull, 0.75), grown = offset_polygon(geo.hull, -0.75);
            const Poly screen = {{0.0, 0.0}, {static_cast<double>(w), 0.0}, {static_cast<double>(w), static_cast<double>(h)}, {0.0, static_cast<double>(h)}};
            // Color writes off (the picture stays), depth test on with ALWAYS and no depth writes: every sample "passes the depth
            // test", which both the query (4.1.6) and the stencil dppass operation (4.1.5) refer to.
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            mgl::glDepthMask(GL_FALSE);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilMask(0xFFu);
            (void)demo::drain_gl_errors();

            count_cube_coverage(w, h, 0);
            const bool control_uncovered = any_sample(inset, w, h, mgl::CompareFunc::Equal, 0);
            count_cube_coverage(w, h, 1);
            const bool twice = any_sample(screen, w, h, mgl::CompareFunc::Less, 1);       // 1 < stored: covered more than once
            const bool crack = any_sample(inset, w, h, mgl::CompareFunc::Equal, 0);       // inside the silhouette, never covered
            const bool covered = any_sample(screen, w, h, mgl::CompareFunc::Notequal, 0); // positive control
            // Zero every sample inside the grown silhouette; a sample the cube covered outside it would keep its 1.
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFu);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Zero);
            draw_polygon(grown, w, h);
            const bool outside = any_sample(screen, w, h, mgl::CompareFunc::Notequal, 0);
            count_cube_coverage(w, h, 2);
            const bool control_twice = any_sample(screen, w, h, mgl::CompareFunc::Less, 1);

            mgl::glClear(mgl::ClearBufferBit::Stencil);
            mgl::glDisable(mgl::Capability::StencilTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_TRUE);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            std::cout << std::format("   note: window stencil bits {}\n", stencil_bits);
            check.expect_no_gl_error("the stencil counting passes and occlusion queries on the window raise no error");
            check.expect(!twice, "per sample (stencil INCR + ANY_SAMPLES_PASSED): no sample of the window is covered by two front faces (shared edges, 3.6.1/3.6.3)");
            check.expect(!crack, "per sample: every sample 0.75 px or more inside the silhouette is covered by a front face (no crack along the seams)");
            check.expect(!outside, "per sample: the cube covers no sample 0.75 px or more outside its silhouette");
            check.expect(covered && control_uncovered && control_twice,
                         "positive controls: the queries return TRUE for covered samples, for the inset silhouette before the cube is drawn, and for "
                         "samples covered twice when the cube is drawn twice",
                         std::format("covered {}, uncovered before drawing {}, drawn twice {}", covered, control_uncovered, control_twice));
        }

        // ------------------------------------------------------------------------------------------
        // SAMPLE_BUFFERS / SAMPLES (3.3, 4.4.4.2), the back buffer's format (6.1.13), sample positions (ES 3.1+).
        int check_queries(const demo::Env& env, demo::Check& check) const
        {
            bind_both(kDefaultFbo);
            const int sample_buffers = demo::get_integer(mgl::GetParameter::SampleBuffers);
            const int samples = demo::get_integer(mgl::GetParameter::Samples);
            std::cout << std::format("   note: default framebuffer SAMPLE_BUFFERS {}, SAMPLES {} ({} requested from EGL), SUBPIXEL_BITS {}\n", sample_buffers,
                                     samples, kRequestedSamples, demo::get_integer(mgl::GetParameter::SubpixelBits));
            check.expect_eq(sample_buffers, 1, "default framebuffer: SAMPLE_BUFFERS = 1 (multisample rasterization)");
            check.expect(samples >= 1, std::format("default framebuffer: SAMPLES = {} >= 1", samples));

            // SAMPLE_BUFFERS and SAMPLES are framebuffer-dependent: they follow the draw framebuffer.
            bind_draw(single_);
            check.expect(demo::get_integer(mgl::GetParameter::SampleBuffers) == 0 && demo::get_integer(mgl::GetParameter::Samples) == 0,
                         "single-sample RGBA8 framebuffer object as DRAW_FRAMEBUFFER: SAMPLE_BUFFERS = 0 and SAMPLES = 0");
            bind_draw(ms_fbo_);
            check.expect(demo::get_integer(mgl::GetParameter::SampleBuffers) == 1 && demo::get_integer(mgl::GetParameter::Samples) >= kRequestedSamples,
                         "4-sample framebuffer object as DRAW_FRAMEBUFFER: SAMPLE_BUFFERS = 1, SAMPLES >= 4");
            bind_draw(kDefaultFbo);
            check.expect(demo::get_integer(mgl::GetParameter::SampleBuffers) == 1 && demo::get_integer(mgl::GetParameter::Samples) == samples,
                         "default framebuffer bound again: SAMPLE_BUFFERS = 1 and the same SAMPLES");

            // The back buffer is RGBA8, unsigned normalized, linear: the format of the RGBA8 blit target (4.3.3 requires
            // identical formats for a resolve).
            std::array<GLint, 6> v{};
            const std::array<mgl::FramebufferAttachmentParameter, 6> pn = {
                mgl::FramebufferAttachmentParameter::RedSize,   mgl::FramebufferAttachmentParameter::GreenSize,     mgl::FramebufferAttachmentParameter::BlueSize,
                mgl::FramebufferAttachmentParameter::AlphaSize, mgl::FramebufferAttachmentParameter::ComponentType, mgl::FramebufferAttachmentParameter::ColorEncoding};
            for (std::size_t i = 0; i < pn.size(); ++i)
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::DrawFramebuffer, mgl::DefaultFramebufferQueryAttachment::Back, pn[i], &v[i]);
            check.expect(v[0] == 8 && v[1] == 8 && v[2] == 8 && v[3] == 8 && v[4] == gl(GL_UNSIGNED_NORMALIZED) && v[5] == gl(GL_LINEAR),
                         "the default framebuffer's BACK buffer is 8/8/8/8 bits, UNSIGNED_NORMALIZED, LINEAR encoding (RGBA8)",
                         std::format("sizes {}/{}/{}/{}, component type 0x{:04X}, encoding 0x{:04X}", v[0], v[1], v[2], v[3], v[4], v[5]));
            check.expect_no_gl_error("default framebuffer queries raise no error");

            if (env.context_version >= demo::Version{3, 1})
            {
                // ES 3.2 13.4: "x and y each lie in the range [0, 1]"; index >= SAMPLES is INVALID_VALUE.
                std::string positions;
                bool in_range = true;
                for (int i = 0; i < samples; ++i)
                {
                    std::array<GLfloat, 2> p = {-1.0f, -1.0f};
                    mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(i), p.data());
                    in_range = in_range && p[0] >= 0.0f && p[0] <= 1.0f && p[1] >= 0.0f && p[1] <= 1.0f;
                    positions += std::format(" ({:.4f}, {:.4f})", p[0], p[1]);
                }
                std::cout << "   note: sample positions:" << positions << "\n";
                check.expect(in_range, "glGetMultisamplefv(SAMPLE_POSITION, i) for every i < SAMPLES lies in [0, 1] x [0, 1] (ES 3.1+)", positions);
                check.expect_no_gl_error("glGetMultisamplefv for valid indices raises no error");
                std::array<GLfloat, 2> p{};
                expect_error(check, mgl::ErrorCode::InvalidValue, "glGetMultisamplefv(SAMPLE_POSITION, SAMPLES) -> INVALID_VALUE (ES 3.1+)",
                             [&] { mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(samples), p.data()); });
            }
            else
            {
                std::cout << "   note: ES 3.0 context: glGetMultisamplefv is ES 3.1 and is not called\n";
            }
            return samples;
        }

        // ------------------------------------------------------------------------------------------
        // The INVALID_OPERATION cases of 4.3.2 and 4.3.3 that involve the multisampled default framebuffer.
        void check_errors(demo::Check& check, int w, int h, const Image& rp) const
        {
            const Region all{0, 0, w, h};
            const int hw = w / 2, hh = h / 2;
            prepare_single();
            bind_read(kDefaultFbo);
            bind_draw(single_);
            expect_error(check, mgl::ErrorCode::InvalidOperation,
                         "resolve from the default framebuffer to a rectangle of the same size at another position -> INVALID_OPERATION",
                         [&] { blit({0, 0, hw, hh}, {w / 4, h / 4, w / 4 + hw, h / 4 + hh}); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "resolve from the default framebuffer with scaling (destination 2x) -> INVALID_OPERATION",
                         [&] { blit({0, 0, hw, hh}, {0, 0, 2 * hw, 2 * hh}); });
            expect_error(check, mgl::ErrorCode::InvalidOperation,
                         "resolve from the default framebuffer with only the destination mirrored (X0 and X1 differ) -> INVALID_OPERATION",
                         [&] { blit(all, {w, 0, 0, h}); });
            check.expect(outside_is_sentinel(read_single(w, h), {0, 0, 0, 0}), "the rejected resolves copied nothing (the target still holds the sentinel)");

            bind_read(kDefaultFbo);
            bind_draw(rgb565_);
            expect_error(check, mgl::ErrorCode::InvalidOperation,
                         "resolve from the RGBA8 default framebuffer into RGB565: formats not identical -> INVALID_OPERATION", [&] { blit(all, all); });

            // The multisampled default framebuffer as the *destination*: SAMPLE_BUFFERS of the draw framebuffer is one.
            prepare_single();
            bind_read(single_);
            bind_draw(kDefaultFbo);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "blit from a single-sample framebuffer object into the multisampled default framebuffer -> INVALID_OPERATION",
                         [&] { blit(all, all); });
            bind_read(kDefaultFbo);
            int max_diff = 0;
            check.expect(count_differences(read_image(w, h), rp, max_diff) == 0, "...and the default framebuffer is unchanged (glReadPixels gives the same image)");

            // Contrast for 4.3.2: the same read from a multisample framebuffer *object* is an error.
            bind_read(ms_fbo_);
            std::array<std::uint8_t, 4> px{};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glReadPixels from a complete 4-sample framebuffer object -> INVALID_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data()); });
            bind_read(kDefaultFbo);
            (void)demo::drain_gl_errors();
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            check.expect_no_gl_error("the same glReadPixels from the multisampled default framebuffer raises no error");
        }

        mgl::ProgramId program_{};
        mgl::UniformLocation u_mvp_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};
        mgl::BufferId poly_vbo_{};
        mgl::VertexArrayId poly_vao_{};
        mgl::QueryId query_{};
        GLsizei vertex_count_ = 0;
        mgl::FramebufferId single_{}, rgb565_{}, ms_fbo_{};
        mgl::RenderbufferId single_rb_{}, rgb565_rb_{}, ms_rb_{};
        int frame_ = 0;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es30-multisample-default-framebuffer",
        .title = "Multisampled default framebuffer",
        .min_es = {3, 0},
        .summary = "A 4-sample default framebuffer: SAMPLE_BUFFERS / SAMPLES, an anti-aliased rotating cube read back with glReadPixels (allowed for the "
                   "default framebuffer) and resolved with glBlitFramebuffer, checked against a CPU projection, and the INVALID_OPERATION cases of ES 3.0 4.3.2 / 4.3.3.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
        .samples = kRequestedSamples,
    };
    return demo::run<MultisampleDefault>(argc, argv, info);
}
