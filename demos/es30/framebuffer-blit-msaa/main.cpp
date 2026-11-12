// es30-framebuffer-blit-msaa
//
// glBlitFramebuffer, separate READ / DRAW framebuffer bindings, multisample renderbuffers and the
// internal-format sample-count query (OpenGL ES 3.0 sections 4.3.3, 4.4.1, 4.4.2, 6.1.15).
//
// The window is four quarters (any window size), each one a blit that stretches an image onto it:
//   top-left     the 64 x 64 chart stretched with NEAREST      | top-right the same chart with LINEAR
//   bottom-left  a depth+stencil scene copied by a blit and then *shown* by depth- and stencil-tested
//                draws (red where the copied depth failed the test, white where it passed, black where
//                the copied stencil is 3)                      | bottom-right a triangle rendered into a
//                multisample renderbuffer and resolved by a blit: the pixels along its edges are blends.
//
// Validation, all against CPU models:
//   * color blits: one generic CPU model of the spec's rule "a destination pixel is written when its
//     centre lies in the destination rectangle, and receives the source at the matching point" covers
//     1:1 copies, sub-rectangles, 2x and 3:1 NEAREST, 2x and 2:1 LINEAR, mirroring by swapped
//     coordinates, clipping on either side, the scissor test, and write masks / fragment tests being
//     ignored; pixels outside the written region must keep their previous values exactly;
//   * glReadBuffer picks the source attachment and every draw buffer receives the copy;
//   * depth and stencil blits are proven through depth- and stencil-tested draws on the copy;
//   * the multisample resolve, asserted only where the specification defines it: the filter is ignored, an
//     identical-bounds sub-rectangle resolves just that rectangle, uniformly cleared depth and stencil resolve to
//     their values ("between the minimum and maximum depth", "a single sample's value is selected"), and - in
//     ES 3.1+ contexts - every pixel of an RGBA8UI resolve equals one of its own samples, which are read back
//     one by one with texelFetch(usampler2DMS). The RGBA8 resolve of the triangle is implementation-dependent
//     (ES 3.0 4.3.3): its blends, bounds and coverage are printed as observations, not asserted;
//   * the errors ES 3.0 4.3.3 defines (LINEAR with depth/stencil, format mismatches, resolve with
//     scaling / translation / mirroring, multisample draw framebuffer, identical buffers, ...);
//   * glGetInternalformativ (NUM_SAMPLE_COUNTS / SAMPLES) and allocation with the largest count.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kSrc = 64;      // edge of the chart (and of the depth/stencil scene)
    constexpr int kDst = 128;     // edge of the destination targets and of the multisample target
    constexpr int kMinWindow = 64;  // the window is 2 x 2 quarters of any size; this is the smallest
    constexpr int kSmall = 32;    // edge of the integer targets

    constexpr mgl::ClearBufferBit kColorBit = mgl::ClearBufferBit::Color;
    constexpr mgl::ClearBufferBit kDepthBit = mgl::ClearBufferBit::Depth;
    constexpr mgl::ClearBufferBit kStencilBit = mgl::ClearBufferBit::Stencil;
    constexpr mgl::BlitFilter kNearest = mgl::BlitFilter::Nearest;
    constexpr mgl::BlitFilter kLinear = mgl::BlitFilter::Linear;

    constexpr std::array<mgl::ColorAttachment, 3> kAtt = {mgl::ColorAttachment::Color0, mgl::ColorAttachment::Color1,
                                                          mgl::ColorAttachment::Color2};
    constexpr mgl::DrawBuffer kDraw0 = mgl::to_draw_buffer(mgl::ColorAttachment::Color0);
    constexpr mgl::DrawBuffer kDraw1 = mgl::to_draw_buffer(mgl::ColorAttachment::Color1);
    constexpr mgl::DrawBuffer kDraw2 = mgl::to_draw_buffer(mgl::ColorAttachment::Color2);
    constexpr mgl::DrawBuffer kDrawNone = mgl::DrawBuffer::None;
    const mgl::FramebufferId kDefaultFbo{0};

    constexpr int gl(GLenum e) { return static_cast<int>(e); }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    // ---------------------------------------------------------------------------------------------
    // Colors. The destination sentinels differ from every chart pixel (the chart's red channel is a
    // multiple of 4, the sentinels' is not), so "written" and "kept" can never be confused.
    // ---------------------------------------------------------------------------------------------
    using Rgbaf = std::array<float, 4>;
    constexpr std::array<Rgba8, 3> kSentinel = {{{30, 61, 90, 255}, {91, 62, 31, 255}, {61, 90, 33, 255}}};
    constexpr Rgbaf kSceneBackground = {0.15f, 0.15f, 0.15f, 1.0f};
    constexpr Rgbaf kSceneRed = {0.9f, 0.2f, 0.2f, 1.0f};
    constexpr Rgbaf kSceneGreen = {0.2f, 0.8f, 0.3f, 1.0f};
    constexpr Rgbaf kSceneBlue = {0.2f, 0.4f, 0.9f, 1.0f};
    constexpr Rgbaf kWhite = {1.0f, 1.0f, 1.0f, 1.0f};
    constexpr Rgbaf kBlack = {0.0f, 0.0f, 0.0f, 1.0f};
    constexpr Rgbaf kTriBackground = {0.10f, 0.12f, 0.30f, 1.0f};  // A: what is outside the triangle
    constexpr Rgbaf kTriColor = {0.95f, 0.55f, 0.10f, 1.0f};       // B: what is inside it

    // ---------------------------------------------------------------------------------------------
    // Images in GL order (row 0 is the bottom row, like glReadPixels returns them).
    // ---------------------------------------------------------------------------------------------
    struct Rect
    {
        int x0, y0, x1, y1;  // blit coordinates: x1 < x0 or y1 < y0 mirrors
    };

    struct Image
    {
        int w = 0, h = 0;
        std::vector<Rgba8> px;
        Image() = default;
        Image(int width, int height, Rgba8 fill = {0, 0, 0, 0})
            : w(width), h(height), px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), fill)
        {
        }
        Rgba8& at(int x, int y) { return px[static_cast<std::size_t>(y * w + x)]; }
        const Rgba8& at(int x, int y) const { return px[static_cast<std::size_t>(y * w + x)]; }
    };

    Image read_image(int w, int h)
    {
        const std::vector<std::uint8_t> bytes = demo::read_rgba8(0, 0, w, h);
        Image im(w, h);
        for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = {bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]};
        return im;
    }

    Image crop(const Image& src, int x0, int y0, int w, int h)
    {
        Image im(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) im.at(x, y) = src.at(x0 + x, y0 + y);
        return im;
    }

    struct Diff
    {
        int bad = 0;
        int max_error = 0;
        std::string first;
    };

    Diff compare(const Image& got, const Image& want, int tolerance)
    {
        Diff d;
        if (got.w != want.w || got.h != want.h)
        {
            d.bad = 1;
            d.first = "image sizes differ";
            return d;
        }
        for (int y = 0; y < got.h; ++y)
            for (int x = 0; x < got.w; ++x)
            {
                bool bad = false;
                for (std::size_t c = 0; c < 4; ++c)
                {
                    const int err = std::abs(int{got.at(x, y)[c]} - int{want.at(x, y)[c]});
                    d.max_error = std::max(d.max_error, err);
                    if (err > tolerance) bad = true;
                }
                if (bad && d.bad++ == 0)
                    d.first = std::format("first at ({},{}): got {} expected {}", x, y, demo::format_rgba(got.at(x, y)),
                                          demo::format_rgba(want.at(x, y)));
            }
        return d;
    }

    bool expect_image(demo::Check& check, std::string_view what, const Image& got, const Image& want, int tolerance)
    {
        const Diff d = compare(got, want, tolerance);
        return check.expect(d.bad == 0, what,
                            d.bad == 0 ? std::format("{}x{} pixels, max error {} (tolerance {})", got.w, got.h, d.max_error, tolerance)
                                       : std::format("{} of {} pixels differ, {} (tolerance {})", d.bad, got.w * got.h, d.first, tolerance));
    }

    // ---------------------------------------------------------------------------------------------
    // CPU model of glBlitFramebuffer (ES 3.0 4.3.3): the destination pixel (x, y) is written when its
    // centre (x + 0.5, y + 0.5) lies inside the destination rectangle. It receives the source image at
    // the point the centre maps to; a point outside the source buffer is clipped away (the scaling is
    // computed as though no clipping took place). Swapped coordinates mirror.
    // ---------------------------------------------------------------------------------------------
    struct SourcePoint
    {
        double u, v;  // continuous source pixel coordinates; pixel i covers [i, i + 1)
    };

    std::optional<SourcePoint> map_pixel(const Rect& s, const Rect& d, int x, int y, int src_w, int src_h)
    {
        if (d.x1 == d.x0 || d.y1 == d.y0) return std::nullopt;
        const double tx = (x + 0.5 - d.x0) / static_cast<double>(d.x1 - d.x0);
        const double ty = (y + 0.5 - d.y0) / static_cast<double>(d.y1 - d.y0);
        if (tx < 0.0 || tx > 1.0 || ty < 0.0 || ty > 1.0) return std::nullopt;
        const double u = s.x0 + tx * static_cast<double>(s.x1 - s.x0);
        const double v = s.y0 + ty * static_cast<double>(s.y1 - s.y0);
        if (u < 0.0 || u >= src_w || v < 0.0 || v >= src_h) return std::nullopt;
        return SourcePoint{u, v};
    }

    int floor_index(double v) { return static_cast<int>(std::floor(v)); }

    Rgba8 sample_nearest(const Image& s, const SourcePoint& p)
    {
        return s.at(std::clamp(floor_index(p.u), 0, s.w - 1), std::clamp(floor_index(p.v), 0, s.h - 1));
    }

    // LINEAR: bilinear interpolation of the four texels around (u, v) - 0.5, edge texels repeated beyond
    // the buffer ("as though CLAMP_TO_EDGE texture sampling were being performed").
    Rgba8 sample_linear(const Image& s, const SourcePoint& p)
    {
        const double u = p.u - 0.5, v = p.v - 0.5;
        const int i0 = floor_index(u), j0 = floor_index(v);
        const double fx = u - i0, fy = v - j0;
        auto texel = [&](int i, int j) { return s.at(std::clamp(i, 0, s.w - 1), std::clamp(j, 0, s.h - 1)); };
        const Rgba8 t00 = texel(i0, j0), t10 = texel(i0 + 1, j0), t01 = texel(i0, j0 + 1), t11 = texel(i0 + 1, j0 + 1);
        Rgba8 out{};
        for (std::size_t c = 0; c < 4; ++c)
        {
            const double bottom = t00[c] * (1.0 - fx) + t10[c] * fx;
            const double top = t01[c] * (1.0 - fx) + t11[c] * fx;
            out[c] = to8((bottom * (1.0 - fy) + top * fy) / 255.0);
        }
        return out;
    }

    Image model_color_blit(const Image& before, const Image& src, const Rect& s, const Rect& d, mgl::BlitFilter filter,
                           std::optional<Rect> scissor = std::nullopt)
    {
        Image out = before;
        for (int y = 0; y < out.h; ++y)
            for (int x = 0; x < out.w; ++x)
            {
                if (scissor && (x < scissor->x0 || x >= scissor->x1 || y < scissor->y0 || y >= scissor->y1)) continue;
                const auto p = map_pixel(s, d, x, y, src.w, src.h);
                if (!p) continue;
                out.at(x, y) = filter == kNearest ? sample_nearest(src, *p) : sample_linear(src, *p);
            }
        return out;
    }

    /// The source pixel index (or indices) a NEAREST blit may read along one axis for source coordinate `c`: the
    /// pixel whose square contains it, plus the neighbour when `c` lies within 0.05 pixel of a pixel edge. A stretch by a
    /// non-integer factor maps some destination centres exactly onto source pixel edges (e.g. 64 -> 160 pixels), where
    /// the implementation's rounding decides; elsewhere the choice is unique.
    std::vector<int> nearest_candidates(double c, int size)
    {
        const int i = floor_index(c);
        const double f = c - i;
        std::vector<int> out{std::clamp(i, 0, size - 1)};
        if (f < 0.05 && i - 1 >= 0) out.push_back(i - 1);
        if (f > 0.95 && i + 1 < size) out.push_back(i + 1);
        return out;
    }

    /// Compares `got` (a whole destination rectangle) with a NEAREST blit that stretches all of `src` onto it: every
    /// pixel must equal (within `tolerance`) one of the source pixels nearest_candidates() allows for its centre.
    Diff compare_nearest_stretch(const Image& got, const Image& src, int tolerance)
    {
        Diff d;
        const Rect s{0, 0, src.w, src.h}, dst{0, 0, got.w, got.h};
        for (int y = 0; y < got.h; ++y)
            for (int x = 0; x < got.w; ++x)
            {
                const auto p = map_pixel(s, dst, x, y, src.w, src.h);
                if (!p)
                {
                    if (d.bad++ == 0) d.first = std::format("pixel ({},{}) has no source point", x, y);
                    continue;
                }
                int best = 1 << 30;
                for (const int j : nearest_candidates(p->v, src.h))
                    for (const int i : nearest_candidates(p->u, src.w))
                    {
                        int err = 0;
                        for (std::size_t c = 0; c < 4; ++c) err = std::max(err, std::abs(int{got.at(x, y)[c]} - int{src.at(i, j)[c]}));
                        best = std::min(best, err);
                    }
                d.max_error = std::max(d.max_error, best);
                if (best > tolerance && d.bad++ == 0)
                    d.first = std::format("first at ({},{}): got {} expected {}", x, y, demo::format_rgba(got.at(x, y)),
                                          demo::format_rgba(sample_nearest(src, *p)));
            }
        return d;
    }

    bool expect_nearest_stretch(demo::Check& check, std::string_view what, const Image& got, const Image& src, int tolerance)
    {
        const Diff d = compare_nearest_stretch(got, src, tolerance);
        return check.expect(d.bad == 0, what,
                            d.bad == 0 ? std::format("{}x{} pixels from a {}x{} source, max error {} (tolerance {})", got.w, got.h, src.w, src.h, d.max_error, tolerance)
                                       : std::format("{} of {} pixels differ, {} (tolerance {})", d.bad, got.w * got.h, d.first, tolerance));
    }

    int count_differences(const Image& a, const Image& b)
    {
        int n = 0;
        for (std::size_t i = 0; i < a.px.size(); ++i) n += a.px[i] != b.px[i] ? 1 : 0;
        return n;
    }

    // ---------------------------------------------------------------------------------------------
    // The depth/stencil scene (64 x 64): four regions with distinct depth and stencil values. Edges lie
    // on pixel boundaries so every pixel, and every multisample of it, belongs to exactly one region.
    // ---------------------------------------------------------------------------------------------
    struct SceneRegion
    {
        Rect rect;       // pixels [x0, x1) x [y0, y1)
        double depth;    // window-space depth that ends up in the buffer
        int stencil;
        Rgbaf color;
    };
    constexpr std::array<SceneRegion, 3> kRegions = {{
        {{0, 0, 32, 48}, 0.25, 1, kSceneRed},
        {{32, 32, 64, 64}, 0.75, 2, kSceneGreen},
        {{32, 0, 64, 32}, 0.90, 3, kSceneBlue},
    }};  // the remaining pixels (x < 32, y >= 48) keep the clear values: depth 1, stencil 0

    // The four draws that make a copied depth/stencil buffer visible (render_ds_scene_blit, in this order):
    // a draw at window depth 0.5 / 0.8 passes DEPTH_TEST LESS where the stored depth is greater; a draw with
    // STENCIL_TEST EQUAL passes where the stored stencil value equals the reference.
    struct Overlay
    {
        double depth_ref;  // >= 0: depth-tested draw at this window depth; < 0: stencil-tested draw
        int stencil_ref;
        Rgbaf color;
    };
    constexpr Rgbaf kGray = {170.0f / 255.0f, 170.0f / 255.0f, 170.0f / 255.0f, 1.0f};
    constexpr Rgbaf kCyan = {0.0f, 1.0f, 1.0f, 1.0f};
    constexpr std::array<Overlay, 4> kOverlays = {{{0.5, 0, kWhite}, {-1.0, 2, kCyan}, {0.8, 0, kGray}, {-1.0, 3, kBlack}}};

    struct DsPixel
    {
        float depth = 1.0f;
        int stencil = 0;
    };

    DsPixel scene_pixel(int x, int y)
    {
        for (const SceneRegion& r : kRegions)
            if (x >= r.rect.x0 && x < r.rect.x1 && y >= r.rect.y0 && y < r.rect.y1) return {static_cast<float>(r.depth), r.stencil};
        return {};
    }

    struct DsImage
    {
        int w, h;
        std::vector<DsPixel> px;
        DsImage(int width, int height) : w(width), h(height), px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {}
        DsPixel& at(int x, int y) { return px[static_cast<std::size_t>(y * w + x)]; }
        const DsPixel& at(int x, int y) const { return px[static_cast<std::size_t>(y * w + x)]; }
    };

    /// What a NEAREST depth and/or stencil blit of the scene into a cleared (depth 1, stencil 0) buffer gives.
    DsImage model_ds_blit(int w, int h, const Rect& s, const Rect& d, bool depth, bool stencil)
    {
        DsImage out(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const auto p = map_pixel(s, d, x, y, kSrc, kSrc);
                if (!p) continue;
                const DsPixel src = scene_pixel(floor_index(p->u), floor_index(p->v));
                if (depth) out.at(x, y).depth = src.depth;
                if (stencil) out.at(x, y).stencil = src.stencil;
            }
        return out;
    }

    // The probe passes read the depth/stencil buffer back through draws: bit 0 = a draw at window depth
    // 0.5 passes DEPTH_TEST LESS (stored depth > 0.5), bit 1 = a draw at 0.8 passes (stored > 0.8),
    // bit 2 + k = a draw passes STENCIL_TEST EQUAL k (stored stencil == k).
    std::uint8_t probe_bits(const DsPixel& p)
    {
        std::uint8_t bits = 0;
        if (p.depth > 0.5f) bits |= 1u;
        if (p.depth > 0.8f) bits |= 2u;
        if (p.stencil >= 0 && p.stencil <= 3) bits |= static_cast<std::uint8_t>(1u << (2 + p.stencil));
        return bits;
    }

    // ---------------------------------------------------------------------------------------------
    // The triangle (pixels of the 128 x 128 multisample target) and its exact geometry.
    // ---------------------------------------------------------------------------------------------
    constexpr std::array<float, 6> kTriangle = {10.3f, 8.6f, 118.7f, 33.4f, 46.2f, 120.1f};

    struct Triangle
    {
        std::array<std::array<double, 2>, 3> p;
        Triangle()
        {
            for (std::size_t i = 0; i < 3; ++i) p[i] = {static_cast<double>(kTriangle[2 * i]), static_cast<double>(kTriangle[2 * i + 1])};
        }
        /// > 0 on the inner side of edge i (the triangle is counter-clockwise).
        double edge(std::size_t i, double x, double y) const
        {
            const auto& a = p[i];
            const auto& b = p[(i + 1) % 3];
            return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        }
        double length(std::size_t i) const
        {
            const auto& a = p[i];
            const auto& b = p[(i + 1) % 3];
            return std::hypot(b[0] - a[0], b[1] - a[1]);
        }
        bool inside(double x, double y) const { return edge(0, x, y) >= 0.0 && edge(1, x, y) >= 0.0 && edge(2, x, y) >= 0.0; }
        double area() const
        {
            return 0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]));
        }
    };

    enum class PixelClass
    {
        Exterior,  // the pixel square and the triangle are disjoint
        Edge,      // anything else
        Interior   // the pixel square lies inside the triangle
    };

    PixelClass classify(const Triangle& t, int x, int y)
    {
        const std::array<std::array<double, 2>, 4> corners = {{{double(x), double(y)}, {x + 1.0, double(y)}, {double(x), y + 1.0}, {x + 1.0, y + 1.0}}};
        int inside = 0;
        for (const auto& c : corners) inside += t.inside(c[0], c[1]) ? 1 : 0;
        if (inside == 4) return PixelClass::Interior;
        // Separating axis test: the three edge lines and the two axes (bounding box).
        for (std::size_t i = 0; i < 3; ++i)
        {
            bool all_outside = true;
            for (const auto& c : corners) all_outside = all_outside && t.edge(i, c[0], c[1]) < 0.0;
            if (all_outside) return PixelClass::Exterior;
        }
        double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (const auto& v : t.p)
        {
            min_x = std::min(min_x, v[0]);
            max_x = std::max(max_x, v[0]);
            min_y = std::min(min_y, v[1]);
            max_y = std::max(max_y, v[1]);
        }
        if (x + 1.0 <= min_x || x >= max_x || y + 1.0 <= min_y || y >= max_y) return PixelClass::Exterior;
        return PixelClass::Edge;
    }

    /// An edge pixel whose centre lies within 0.1 pixel of an edge line and more than 3 pixels from every
    /// corner: the edge splits the pixel into two parts of almost equal area, so every sample pattern
    /// that spreads its samples around the pixel centre has samples on both sides.
    bool centered_on_edge(const Triangle& t, int x, int y)
    {
        const double cx = x + 0.5, cy = y + 0.5;
        for (const auto& v : t.p)
            if (std::hypot(v[0] - cx, v[1] - cy) < 3.0) return false;
        for (std::size_t i = 0; i < 3; ++i)
            if (std::abs(t.edge(i, cx, cy)) / t.length(i) <= 0.1) return true;
        return false;
    }

    Rgba8 to_rgba8(const Rgbaf& c)
    {
        return {to8(static_cast<double>(c[0])), to8(static_cast<double>(c[1])), to8(static_cast<double>(c[2])), to8(static_cast<double>(c[3]))};
    }
    Rgba8 triangle_background8() { return to_rgba8(kTriBackground); }
    Rgba8 triangle_color8() { return to_rgba8(kTriColor); }

    bool near_color(const Rgba8& got, const Rgba8& want, int tolerance)
    {
        for (std::size_t c = 0; c < 4; ++c)
            if (std::abs(int{got[c]} - int{want[c]}) > tolerance) return false;
        return true;
    }

    /// The result of the two pixel-class checks on one resolved (or control) image.
    struct TriangleStats
    {
        int interior = 0, interior_bad = 0;
        int exterior = 0, exterior_bad = 0;
        int edge = 0, edge_outside_range = 0;
        int centered = 0, centered_strictly_between = 0;
        int intermediate = 0;  // edge pixels strictly between A and B on the red channel
        double covered_area = 0.0;
    };

    TriangleStats analyze_triangle(const Image& img)
    {
        const Triangle tri;
        const Rgba8 a = triangle_background8(), b = triangle_color8();
        TriangleStats st;
        const int lo = std::min(int{a[0]}, int{b[0]}) + 1, hi = std::max(int{a[0]}, int{b[0]}) - 1;  // "strictly between" in red
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x)
            {
                const Rgba8 got = img.at(x, y);
                const double fraction = (static_cast<double>(got[0]) - a[0]) / (static_cast<double>(b[0]) - a[0]);
                const bool between = got[0] >= lo && got[0] <= hi;
                switch (classify(tri, x, y))
                {
                case PixelClass::Interior:
                    ++st.interior;
                    st.covered_area += 1.0;
                    st.interior_bad += near_color(got, b, 1) ? 0 : 1;
                    break;
                case PixelClass::Exterior:
                    ++st.exterior;
                    st.exterior_bad += near_color(got, a, 1) ? 0 : 1;
                    break;
                case PixelClass::Edge:
                {
                    ++st.edge;
                    st.covered_area += fraction;  // interior pixels count 1, exterior pixels 0: only the edge pixels are blends
                    bool outside = false;
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        const int l = std::min(int{a[c]}, int{b[c]}) - 1, h = std::max(int{a[c]}, int{b[c]}) + 1;
                        outside = outside || int{got[c]} < l || int{got[c]} > h;
                    }
                    st.edge_outside_range += outside ? 1 : 0;
                    st.intermediate += between ? 1 : 0;
                    if (centered_on_edge(tri, x, y))
                    {
                        ++st.centered;
                        st.centered_strictly_between += between ? 1 : 0;
                    }
                    break;
                }
                }
            }
        return st;
    }

    // ---------------------------------------------------------------------------------------------
    template <typename F>
    bool expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // a stale error must not be mistaken for the one under test
        call();
        return check.expect_gl_error(code, what);
    }

    /// Strict check; on the Mesa Gallium drivers of the test matrix (radeonsi, llvmpipe, and zink on lavapipe, whose
    /// GL_RENDERER contains "llvmpipe") a failure is recorded as a labelled XFAIL (docs/KNOWN_LIMITATIONS.md).
    void expect_known_mesa_deviation(demo::Check& check, bool ok, std::string_view what, const std::string& detail, std::string_view reason)
    {
        const GLubyte* r = mgl::glGetString(mgl::StringName::Renderer);
        const std::string_view renderer = r != nullptr ? reinterpret_cast<const char*>(r) : "";
        std::string_view driver = "no-known-driver";
        for (const std::string_view candidate : {"radeonsi", "zink", "llvmpipe"})
            if (renderer.find(candidate) != std::string_view::npos)
            {
                driver = candidate;
                break;
            }
        check.expect_or_known_deviation(ok, what, detail, driver, reason);
    }

    void blit(const Rect& s, const Rect& d, mgl::ClearBufferBit mask, mgl::BlitFilter filter)
    {
        mgl::glBlitFramebuffer(s.x0, s.y0, s.x1, s.y1, d.x0, d.y0, d.x1, d.y1, mask, filter);
    }

    void bind_read(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, f); }
    void bind_draw(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, f); }
    void bind_both(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, f); }
    void bind_pair(mgl::FramebufferId read, mgl::FramebufferId draw)
    {
        bind_read(read);
        bind_draw(draw);
    }

    /// Back to the initial fragment state: nothing that a blit could (wrongly) be affected by is active.
    void reset_fragment_state()
    {
        mgl::glDisable(mgl::Capability::ScissorTest);
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glDisable(mgl::Capability::StencilTest);
        mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        mgl::glDepthMask(GL_TRUE);
        mgl::glStencilMask(0xFFFFFFFFu);
        mgl::glDepthFunc(mgl::CompareFunc::Less);
        mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFu);
        mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
    }

    Image read_attachment(mgl::FramebufferId f, mgl::ColorAttachment a, int w, int h)
    {
        bind_read(f);
        mgl::glReadBuffer(mgl::to_read_buffer(a));
        return read_image(w, h);
    }

    class BlitMsaa final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kMinWindow || env.height < kMinWindow)
                throw demo::Skip(std::format("this demo needs a default framebuffer of at least {0}x{0} pixels", kMinWindow));
            width_ = env.width;
            height_ = env.height;

            quad_ = demo::build_program(env, "shaders/quad.vert", "shaders/flat.frag");
            chart_ = demo::build_program(env, "shaders/quad.vert", "shaders/chart.frag");
            tri_ = demo::build_program(env, "shaders/tri.vert", "shaders/flat.frag");
            quad_rect_ = mgl::glGetUniformLocation(quad_, "u_rect");
            quad_z_ = mgl::glGetUniformLocation(quad_, "u_z");
            quad_color_ = mgl::glGetUniformLocation(quad_, "u_color");
            chart_rect_ = mgl::glGetUniformLocation(chart_, "u_rect");
            chart_z_ = mgl::glGetUniformLocation(chart_, "u_z");
            tri_tri_ = mgl::glGetUniformLocation(tri_, "u_tri");
            tri_size_ = mgl::glGetUniformLocation(tri_, "u_size");
            tri_color_ = mgl::glGetUniformLocation(tri_, "u_color");

            // The sample counts the implementation offers for the formats of the multisample targets.
            rgba8_counts_ = sample_counts(mgl::InternalFormat::Rgba8);
            const std::vector<GLint> d24s8_counts = sample_counts(mgl::InternalFormat::Depth24Stencil8);
            if (rgba8_counts_.empty() || rgba8_counts_.front() < 2 || d24s8_counts.empty() || d24s8_counts.front() < 2)
                throw demo::Skip("the implementation reports no multisample renderbuffer support (ES 3.0 requires MAX_SAMPLES >= 4)");
            ds_samples_ = std::min(rgba8_counts_.front(), d24s8_counts.front());

            // chart source: three RGBA8 textures
            src_ = make_fbo();
            for (std::size_t i = 0; i < 3; ++i)
            {
                src_tex_[i] = make_texture(mgl::SizedInternalFormat::Rgba8, kSrc, kSrc);
                attach_texture(kAtt[i], src_tex_[i]);
            }
            require_complete("chart");

            // destination: three RGBA8 textures
            dst_ = make_fbo();
            for (std::size_t i = 0; i < 3; ++i)
            {
                dst_tex_[i] = make_texture(mgl::SizedInternalFormat::Rgba8, kDst, kDst);
                attach_texture(kAtt[i], dst_tex_[i]);
            }
            require_complete("destination");

            // depth+stencil scene: source, destination, depth-only destination, resolve target
            ds_src_ = make_color_ds_fbo(kSrc, 0, mgl::InternalFormat::Depth24Stencil8, true);
            ds_dst_ = make_color_ds_fbo(kDst, 0, mgl::InternalFormat::Depth24Stencil8, true);
            ds_dst16_ = make_color_ds_fbo(kDst, 0, mgl::InternalFormat::DepthComponent16, false);
            ds_res_ = make_color_ds_fbo(kSrc, 0, mgl::InternalFormat::Depth24Stencil8, true);
            ds_ms_ = make_color_ds_fbo(kSrc, ds_samples_, mgl::InternalFormat::Depth24Stencil8, true, true);
            ds_ms_other_ = make_color_ds_fbo(kSrc, ds_samples_, mgl::InternalFormat::Depth24Stencil8, true, true);

            // multisample color target, its resolve target and the single-sample control
            ms_ = make_fbo();
            ms_rb_ = make_renderbuffer(mgl::InternalFormat::Rgba8, rgba8_counts_.front(), kDst, kDst);
            attach_color_renderbuffer(ms_rb_);
            require_complete("multisample color");
            res_ = make_fbo();
            res_tex_ = make_texture(mgl::SizedInternalFormat::Rgba8, kDst, kDst);
            attach_texture(mgl::ColorAttachment::Color0, res_tex_);
            require_complete("resolve target");
            ss_ = make_fbo();
            ss_tex_ = make_texture(mgl::SizedInternalFormat::Rgba8, kDst, kDst);
            attach_texture(mgl::ColorAttachment::Color0, ss_tex_);
            require_complete("single-sample control");

            // format-conversion and integer targets
            conv_ = make_fbo();
            conv_rb_ = make_renderbuffer(mgl::InternalFormat::Rgb565, 0, kDst, kDst);
            attach_color_renderbuffer(conv_rb_);
            require_complete("RGB565");
            ui_src_ = make_integer_fbo(mgl::SizedInternalFormat::Rgba8UI);
            ui_dst_ = make_integer_fbo(mgl::SizedInternalFormat::Rgba8UI);
            si_ = make_integer_fbo(mgl::SizedInternalFormat::Rgba8I);
            empty_ = make_fbo();  // nothing attached: incomplete

            bind_both(kDefaultFbo);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            width_ = env.width;
            height_ = env.height;
            render_chart();
            render_ds_scene_blit();
            render_multisample_triangle();
            compose_window();
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            // Pull the results of render() into CPU images before anything below modifies the targets.
            for (std::size_t i = 0; i < 3; ++i) chart_img_[i] = read_attachment(src_, kAtt[i], kSrc, kSrc);
            ds_src_img_ = read_attachment(ds_src_, mgl::ColorAttachment::Color0, kSrc, kSrc);
            res_img_ = read_attachment(res_, mgl::ColorAttachment::Color0, kDst, kDst);
            ss_img_ = read_attachment(ss_, mgl::ColorAttachment::Color0, kDst, kDst);

            check_window(check);
            check_chart(check);
            check_read_draw(check);
            check_color_blits(check);
            check_read_buffer(check);
            check_scissor_and_masks(check);
            check_conversion_and_integers(check);
            check_depth_stencil_blits(check);
            check_internalformat_queries(env, check);
            check_multisample_resolve(check);
            check_multisample_depth_stencil(check);
            check_integer_resolve(env, check);
            check_blit_errors(check);

            bind_both(kDefaultFbo);
            reset_fragment_state();
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            bind_both(kDefaultFbo);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(framebuffers_.size()), framebuffers_.data());
            mgl::glDeleteRenderbuffers(static_cast<GLsizei>(renderbuffers_.size()), renderbuffers_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
            mgl::glDeleteProgram(quad_);
            mgl::glDeleteProgram(chart_);
            mgl::glDeleteProgram(tri_);
        }

    private:
        // ==========================================================================================
        // Object construction. Every object is recorded and deleted in shutdown().
        // ==========================================================================================
        static std::vector<GLint> sample_counts(mgl::InternalFormat format)
        {
            GLint n = 0;
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, format, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
            std::vector<GLint> counts(static_cast<std::size_t>(std::max(n, 0)));
            if (n > 0)
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, format, mgl::InternalFormatParameter::Samples, n, counts.data());
            return counts;
        }

        mgl::FramebufferId make_fbo()
        {
            mgl::FramebufferId f;
            mgl::glGenFramebuffers(1, &f);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, f);  // the name becomes an object on its first bind
            framebuffers_.push_back(f);
            return f;
        }

        mgl::TextureId make_texture(mgl::SizedInternalFormat format, int w, int h)
        {
            mgl::TextureId t;
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, w, h);
            textures_.push_back(t);
            return t;
        }

        mgl::RenderbufferId make_renderbuffer(mgl::InternalFormat format, int samples, int w, int h)
        {
            mgl::RenderbufferId r;
            mgl::glGenRenderbuffers(1, &r);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, r);
            if (samples > 0) mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, samples, format, w, h);
            else mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, format, w, h);
            renderbuffers_.push_back(r);
            return r;
        }

        // Attach to the framebuffer bound as FRAMEBUFFER.
        static void attach_texture(mgl::ColorAttachment a, mgl::TextureId t)
        {
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(a), mgl::TextureTarget::Texture2D, t, 0);
        }

        static void attach_color_renderbuffer(mgl::RenderbufferId r)
        {
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, r);
        }

        static void require_complete(std::string_view what)
        {
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer '{}' is not complete: {}", what, demo::framebuffer_status_name(status)));
        }

        /// RGBA8 color (a texture, or a multisample renderbuffer) + a depth(+stencil) renderbuffer.
        mgl::FramebufferId make_color_ds_fbo(int size, int samples, mgl::InternalFormat ds_format, bool stencil, bool color_renderbuffer = false)
        {
            const mgl::FramebufferId f = make_fbo();
            if (color_renderbuffer || samples > 0)
            {
                attach_color_renderbuffer(make_renderbuffer(mgl::InternalFormat::Rgba8, samples, size, size));
            }
            else
            {
                attach_texture(mgl::ColorAttachment::Color0, make_texture(mgl::SizedInternalFormat::Rgba8, size, size));
            }
            const mgl::RenderbufferId ds = make_renderbuffer(ds_format, samples, size, size);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer,
                                           stencil ? mgl::FramebufferAttachment::DepthStencil : mgl::FramebufferAttachment::Depth,
                                           mgl::RenderbufferTarget::Renderbuffer, ds);
            require_complete("depth/stencil target");
            return f;
        }

        mgl::FramebufferId make_integer_fbo(mgl::SizedInternalFormat format)
        {
            const mgl::FramebufferId f = make_fbo();
            attach_texture(mgl::ColorAttachment::Color0, make_texture(format, kSmall, kSmall));
            require_complete("integer target");
            return f;
        }

        // ==========================================================================================
        // Drawing helpers
        // ==========================================================================================
        /// Fills the rectangle of pixels [x0, x1) x [y0, y1) of a size x size target at NDC depth z. The
        /// "quad" program must be in use.
        void draw_rect(const Rect& r, int size, float z, const Rgbaf& color) const
        {
            auto ndc = [size](int p) { return static_cast<float>(p) / static_cast<float>(size) * 2.0f - 1.0f; };
            mgl::glUniform4f(quad_rect_, ndc(r.x0), ndc(r.y0), ndc(r.x1), ndc(r.y1));
            mgl::glUniform1f(quad_z_, z);
            mgl::glUniform4f(quad_color_, color[0], color[1], color[2], color[3]);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        static float ndc_depth(double window_depth) { return static_cast<float>(window_depth * 2.0 - 1.0); }

        /// Renders the depth/stencil scene into the framebuffer bound as FRAMEBUFFER (a size x size target).
        void draw_ds_scene(int size) const
        {
            reset_fragment_state();
            mgl::glViewport(0, 0, size, size);
            mgl::glClearColor(kSceneBackground[0], kSceneBackground[1], kSceneBackground[2], kSceneBackground[3]);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            // Depth and stencil are only written while their test is enabled: ALWAYS / REPLACE it is.
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Replace);
            mgl::glUseProgram(quad_);
            for (const SceneRegion& r : kRegions)
            {
                mgl::glStencilFunc(mgl::CompareFunc::Always, r.stencil, 0xFFu);
                draw_rect(r.rect, kSrc, ndc_depth(r.depth), r.color);
            }
            reset_fragment_state();
        }

        /// The same scene as draw_ds_scene(), written with scissored clears: every sample of a pixel gets the value of the
        /// region that contains the pixel (ES 3.0 4.2.3.1).
        void clear_ds_scene(int size) const
        {
            reset_fragment_state();
            mgl::glViewport(0, 0, size, size);
            mgl::glClearColor(kSceneBackground[0], kSceneBackground[1], kSceneBackground[2], kSceneBackground[3]);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (const SceneRegion& r : kRegions)
            {
                mgl::glScissor(r.rect.x0, r.rect.y0, r.rect.x1 - r.rect.x0, r.rect.y1 - r.rect.y0);
                mgl::glClearColor(r.color[0], r.color[1], r.color[2], r.color[3]);
                mgl::glClearDepthf(static_cast<float>(r.depth));
                mgl::glClearStencil(r.stencil);
                mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            }
            reset_fragment_state();
        }

        void render_chart() const
        {
            bind_both(src_);
            reset_fragment_state();
            const std::array<mgl::DrawBuffer, 3> all = {kDraw0, kDraw1, kDraw2};
            mgl::glDrawBuffers(all);
            mgl::glViewport(0, 0, kSrc, kSrc);
            mgl::glUseProgram(chart_);
            mgl::glUniform4f(chart_rect_, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glUniform1f(chart_z_, 0.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        // The depth/stencil scene is rendered into ds_src_, copied 2x by a blit (color, depth and
        // stencil at once) into ds_dst_, and the copy is then made visible by two tested draws.
        void render_ds_scene_blit() const
        {
            bind_both(ds_src_);
            draw_ds_scene(kSrc);

            bind_both(ds_dst_);
            reset_fragment_state();
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            bind_pair(ds_src_, ds_dst_);
            blit({0, 0, kSrc, kSrc}, {0, 0, kDst, kDst}, kColorBit | kDepthBit | kStencilBit, kNearest);

            // Make the copied depth and stencil visible. Four full-target draws, each limited by one test, paint
            // over the copied color; the CPU model (kOverlays) applies the same four steps.
            bind_both(ds_dst_);
            mgl::glViewport(0, 0, kDst, kDst);
            mgl::glUseProgram(quad_);
            mgl::glDepthMask(GL_FALSE);  // the probes must not change what they probe
            for (const Overlay& o : kOverlays)
            {
                if (o.depth_ref >= 0.0)
                {
                    mgl::glDisable(mgl::Capability::StencilTest);
                    mgl::glEnable(mgl::Capability::DepthTest);
                    mgl::glDepthFunc(mgl::CompareFunc::Less);
                }
                else
                {
                    mgl::glDisable(mgl::Capability::DepthTest);
                    mgl::glEnable(mgl::Capability::StencilTest);
                    mgl::glStencilFunc(mgl::CompareFunc::Equal, o.stencil_ref, 0xFFu);
                }
                draw_rect({0, 0, kDst, kDst}, kDst, o.depth_ref >= 0.0 ? ndc_depth(o.depth_ref) : 0.0f, o.color);
            }
            reset_fragment_state();
        }

        void draw_triangle(int size) const
        {
            reset_fragment_state();
            mgl::glViewport(0, 0, size, size);
            mgl::glClearColor(kTriBackground[0], kTriBackground[1], kTriBackground[2], kTriBackground[3]);
            mgl::glClear(kColorBit);
            mgl::glUseProgram(tri_);
            mgl::glUniform2fv(tri_tri_, 3, kTriangle.data());
            mgl::glUniform1f(tri_size_, static_cast<float>(size));
            mgl::glUniform4f(tri_color_, kTriColor[0], kTriColor[1], kTriColor[2], kTriColor[3]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        // The triangle goes into the multisample renderbuffer; a blit with identical source and
        // destination rectangles resolves it into a texture. The same triangle is also drawn into a
        // single-sample target: the control without any blended pixel.
        void render_multisample_triangle() const
        {
            bind_both(ms_);
            draw_triangle(kDst);
            bind_pair(ms_, res_);
            blit({0, 0, kDst, kDst}, {0, 0, kDst, kDst}, kColorBit, kNearest);
            bind_both(ss_);
            draw_triangle(kDst);
        }

        /// The four quarters of the window (they partition it at width / 2 and height / 2):
        /// 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right.
        [[nodiscard]] Rect panel(int i) const
        {
            const int xm = width_ / 2, ym = height_ / 2;
            const bool right = i % 2 == 1, top = i < 2;
            return {right ? xm : 0, top ? ym : 0, right ? width_ : xm, top ? height_ : ym};
        }

        /// Each image is stretched onto its quarter of the window: the chart (64 x 64) with NEAREST and with LINEAR,
        /// the depth/stencil scene's color target and the resolve target (128 x 128) with NEAREST.
        void compose_window() const
        {
            reset_fragment_state();
            bind_both(kDefaultFbo);
            mgl::glViewport(0, 0, width_, height_);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(kColorBit);
            bind_read(src_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            blit({0, 0, kSrc, kSrc}, panel(0), kColorBit, kNearest);
            blit({0, 0, kSrc, kSrc}, panel(1), kColorBit, kLinear);
            bind_read(ds_dst_);
            blit({0, 0, kDst, kDst}, panel(2), kColorBit, kNearest);
            bind_read(res_);
            blit({0, 0, kDst, kDst}, panel(3), kColorBit, kNearest);
        }

        // ==========================================================================================
        // Fixtures for the checks
        // ==========================================================================================
        void set_draw_buffers(std::initializer_list<mgl::DrawBuffer> list) const
        {
            std::vector<mgl::DrawBuffer> v(list);
            mgl::glDrawBuffers(v);
        }

        /// Every attachment of dst_ gets its sentinel color; only COLOR_ATTACHMENT0 stays a draw buffer.
        void reset_dst() const
        {
            reset_fragment_state();
            bind_both(dst_);
            set_draw_buffers({kDraw0, kDraw1, kDraw2});
            for (std::size_t i = 0; i < 3; ++i)
            {
                const std::array<GLfloat, 4> v = {kSentinel[i][0] / 255.0f, kSentinel[i][1] / 255.0f, kSentinel[i][2] / 255.0f, kSentinel[i][3] / 255.0f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, static_cast<GLint>(i), v.data());
            }
            set_draw_buffers({kDraw0});
        }

        /// ds_dst_ (kDst x kDst): sentinel color, depth 1, stencil 0.
        void reset_ds_target(mgl::FramebufferId f) const
        {
            reset_fragment_state();
            bind_both(f);
            mgl::glClearColor(kSentinel[0][0] / 255.0f, kSentinel[0][1] / 255.0f, kSentinel[0][2] / 255.0f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
        }

        /// Reads the depth and stencil buffer of `f` back through six draws (see probe_bits()). The
        /// framebuffer's color attachment is overwritten; depth and stencil are not modified.
        std::vector<std::uint8_t> probe_ds(mgl::FramebufferId f, int size) const
        {
            reset_fragment_state();
            bind_both(f);
            mgl::glViewport(0, 0, size, size);
            mgl::glUseProgram(quad_);
            mgl::glDepthMask(GL_FALSE);
            std::vector<std::uint8_t> bits(static_cast<std::size_t>(size * size), 0);
            for (int pass = 0; pass < 6; ++pass)
            {
                mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                mgl::glClear(kColorBit);
                float z = 0.0f;
                if (pass < 2)
                {
                    mgl::glEnable(mgl::Capability::DepthTest);
                    mgl::glDisable(mgl::Capability::StencilTest);
                    mgl::glDepthFunc(mgl::CompareFunc::Less);
                    z = pass == 0 ? ndc_depth(0.5) : ndc_depth(0.8);
                }
                else
                {
                    mgl::glDisable(mgl::Capability::DepthTest);
                    mgl::glEnable(mgl::Capability::StencilTest);
                    mgl::glStencilFunc(mgl::CompareFunc::Equal, pass - 2, 0xFFu);
                }
                draw_rect({0, 0, size, size}, size, z, kWhite);
                const Image im = read_image(size, size);
                for (std::size_t i = 0; i < im.px.size(); ++i)
                    if (im.px[i][0] > 127) bits[i] = static_cast<std::uint8_t>(bits[i] | (1u << pass));
            }
            reset_fragment_state();
            return bits;
        }

        static std::vector<std::uint8_t> expected_bits(const DsImage& ds)
        {
            std::vector<std::uint8_t> bits(ds.px.size());
            for (std::size_t i = 0; i < ds.px.size(); ++i) bits[i] = probe_bits(ds.px[i]);
            return bits;
        }

        static int count_mismatches(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
        {
            int n = 0;
            for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i] ? 1 : 0;
            return n;
        }

        static bool expect_bits(demo::Check& check, std::string_view what, const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want)
        {
            const int bad = count_mismatches(got, want);
            return check.expect(bad == 0, what, bad == 0 ? std::format("{} pixels", got.size()) : std::format("{} of {} pixels differ", bad, got.size()));
        }

        // ==========================================================================================
        // Checks
        // ==========================================================================================

        // The window: four blits into the default framebuffer, each stretching an image onto a quarter of the window.
        void check_window(demo::Check& check) const
        {
            bind_read(kDefaultFbo);
            const Image win = read_image(width_, height_);
            auto quarter = [&](int i) {
                const Rect r = panel(i);
                return crop(win, r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
            };
            const Image top_left = quarter(0), top_right = quarter(1);

            const Image blank_tr(top_right.w, top_right.h);
            const Image linear = model_color_blit(blank_tr, chart_img_[0], {0, 0, kSrc, kSrc}, {0, 0, top_right.w, top_right.h}, kLinear);
            const Image nearest_tr = model_color_blit(blank_tr, chart_img_[0], {0, 0, kSrc, kSrc}, {0, 0, top_right.w, top_right.h}, kNearest);
            expect_nearest_stretch(check,
                                   std::format("window top-left: the 64x64 chart stretched onto {}x{} pixels with NEAREST equals the CPU model exactly "
                                               "(either neighbour where a pixel centre maps within 0.05 of a source pixel edge)",
                                               top_left.w, top_left.h),
                                   top_left, chart_img_[0], 0);
            // LINEAR: the spec gives no precision for the filter weights; an 8-bit weight is off by at most 1/256
            // of a channel difference (at most 153 for the checkerboard), plus rounding of the 8-bit result: 2 steps.
            expect_image(check, std::format("window top-right: the chart stretched onto {}x{} pixels with LINEAR matches bilinear interpolation (tolerance 2)",
                                            top_right.w, top_right.h),
                         top_right, linear, 2);
            // At 2x (a 256 x 256 window) the two models differ in about 2/3 of the pixels; 1/8 is required at every size.
            const int differ = count_differences(nearest_tr, linear);
            check.expect(differ > top_right.w * top_right.h / 8,
                         "negative control: the NEAREST and LINEAR models of the top-right quarter differ in more than 1/8 of its pixels",
                         std::format("{} of {} pixels differ", differ, top_right.w * top_right.h));

            // bottom-left: the color copied by the blit, painted over by the four overlay draws where their test passes
            Image ds_panel(kDst, kDst);
            for (int y = 0; y < kDst; ++y)
                for (int x = 0; x < kDst; ++x)
                {
                    const auto p = map_pixel({0, 0, kSrc, kSrc}, {0, 0, kDst, kDst}, x, y, kSrc, kSrc);
                    const DsPixel ds = scene_pixel(floor_index(p->u), floor_index(p->v));
                    Rgba8 c = sample_nearest(ds_src_img_, *p);
                    for (const Overlay& o : kOverlays)
                    {
                        const bool passes = o.depth_ref >= 0.0 ? static_cast<double>(ds.depth) > o.depth_ref : ds.stencil == o.stencil_ref;
                        if (passes) c = to_rgba8(o.color);
                    }
                    ds_panel.at(x, y) = c;
                }
            // flat colors go through float -> 8-bit conversion once more (tolerance 1)
            // The 128 x 128 target is stretched onto the quarter with NEAREST.
            expect_nearest_stretch(check,
                                   "window bottom-left: depth+stencil blit shown through depth- and stencil-tested draws equals the CPU model (tolerance 1), "
                                   "stretched onto the quarter with NEAREST",
                                   quarter(2), ds_panel, 1);

            expect_nearest_stretch(check, "window bottom-right: the resolved multisample image stretched onto the quarter with NEAREST equals the resolve target exactly",
                                   quarter(3), res_img_, 0);
        }

        // The chart itself: the shader's formulas, 8-bit conversion tolerance 1 (equation 2.3 allows either of the two closest integers).
        void check_chart(demo::Check& check) const
        {
            for (std::size_t a = 0; a < 3; ++a)
            {
                Image want(kSrc, kSrc);
                for (int y = 0; y < kSrc; ++y)
                    for (int x = 0; x < kSrc; ++x)
                    {
                        const std::uint8_t r = static_cast<std::uint8_t>(4 * x), g = static_cast<std::uint8_t>(4 * y);
                        const std::uint8_t b = (((x >> 3) + (y >> 3)) & 1) == 1 ? 200 : 60;
                        const std::array<std::array<std::uint8_t, 3>, 3> perm = {{{r, g, b}, {b, r, g}, {g, b, r}}};
                        want.at(x, y) = {perm[a][0], perm[a][1], perm[a][2], 255};
                    }
                expect_image(check, std::format("chart attachment {} holds the three permuted channel patterns (tolerance 1)", a), chart_img_[a], want, 1);
            }
        }

        // READ_FRAMEBUFFER and DRAW_FRAMEBUFFER are separate bindings (ES 3.0 4.4.1).
        void check_read_draw(demo::Check& check)
        {
            reset_dst();
            bind_pair(src_, dst_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadFramebufferBinding), static_cast<int>(src_.value), "READ_FRAMEBUFFER_BINDING is the chart framebuffer");
            check.expect_eq(demo::get_integer(mgl::GetParameter::DrawFramebufferBinding), static_cast<int>(dst_.value), "DRAW_FRAMEBUFFER_BINDING is the destination framebuffer");
            check.expect_eq(demo::get_integer(mgl::GetParameter::FramebufferBinding), static_cast<int>(dst_.value), "FRAMEBUFFER_BINDING reports the draw binding");

            // glReadPixels reads the READ framebuffer: the chart pixel, not the destination sentinel.
            check.expect_pixel(5, 7, chart_img_[0].at(5, 7), 0, "glReadPixels reads the READ framebuffer (a chart pixel)");
            // glClear writes the DRAW framebuffer: the destination turns magenta and the chart is untouched.
            mgl::glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
            mgl::glClear(kColorBit);
            expect_image(check, "glClear with separate bindings clears the DRAW framebuffer", read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst),
                         Image(kDst, kDst, {255, 0, 255, 255}), 0);
            expect_image(check, "...and leaves the READ framebuffer untouched", read_attachment(src_, mgl::ColorAttachment::Color0, kSrc, kSrc), chart_img_[0], 0);

            // Binding FRAMEBUFFER sets both targets at once.
            bind_both(src_);
            check.expect(demo::get_integer(mgl::GetParameter::ReadFramebufferBinding) == static_cast<int>(src_.value) &&
                             demo::get_integer(mgl::GetParameter::DrawFramebufferBinding) == static_cast<int>(src_.value),
                         "binding FRAMEBUFFER sets the READ and the DRAW binding");
        }

        // ------------------------------------------------------------------------------------------
        struct ColorCase
        {
            std::string_view name;
            Rect s, d;
            mgl::BlitFilter filter;
            int tolerance;
        };

        /// Blits chart 0 into a freshly reset destination and compares the whole destination with the model.
        Image run_color_case(demo::Check& check, const ColorCase& c)
        {
            reset_dst();
            const Image before = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
            bind_pair(src_, dst_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            set_draw_buffers({kDraw0});
            (void)demo::drain_gl_errors();
            blit(c.s, c.d, kColorBit, c.filter);
            const bool no_error = check.expect_no_gl_error(std::format("blit '{}' raises no error", c.name));
            const Image after = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
            const Image want = model_color_blit(before, chart_img_[0], c.s, c.d, c.filter);
            const int written = count_differences(before, want);
            check.expect(no_error && written > 0, std::format("blit '{}': the model predicts {} written pixels (not vacuous)", c.name, written));
            expect_image(check, std::format("blit '{}': destination equals the CPU model, pixels outside the region keep their value", c.name), after, want, c.tolerance);
            return after;
        }

        void check_color_blits(demo::Check& check)
        {
            const std::array<ColorCase, 14> cases = {{
                {"1:1 full copy", {0, 0, 64, 64}, {0, 0, 64, 64}, kNearest, 0},
                {"1:1 sub-rectangle (8,16)-(40,48) to (70,60)", {8, 16, 40, 48}, {70, 60, 102, 92}, kNearest, 0},
                {"2x NEAREST", {0, 0, 64, 64}, {0, 0, 128, 128}, kNearest, 0},
                {"3:1 NEAREST downscale picks the central texel", {0, 0, 48, 48}, {10, 10, 26, 26}, kNearest, 0},
                {"mirrored in x (dstX0 > dstX1)", {0, 0, 64, 64}, {64, 0, 0, 64}, kNearest, 0},
                {"mirrored in y (dstY0 > dstY1)", {0, 0, 64, 64}, {0, 64, 64, 0}, kNearest, 0},
                {"mirrored in x and y, 2x", {0, 0, 64, 64}, {128, 128, 0, 0}, kNearest, 0},
                {"source and destination both swapped: no reversal", {64, 0, 0, 64}, {64, 0, 0, 64}, kNearest, 0},
                {"destination partly outside the buffer", {0, 0, 64, 64}, {-16, -16, 48, 48}, kNearest, 0},
                {"source partly outside the buffer", {-8, -8, 56, 56}, {0, 0, 64, 64}, kNearest, 0},
                {"2x NEAREST with the destination clipped on two sides", {0, 0, 64, 64}, {-64, -64, 64, 64}, kNearest, 0},
                {"2x LINEAR", {0, 0, 64, 64}, {0, 0, 128, 128}, kLinear, 2},
                {"2:1 LINEAR downscale", {0, 0, 64, 64}, {0, 0, 32, 32}, kLinear, 2},
                {"2x LINEAR mirrored in x", {0, 0, 64, 64}, {128, 0, 0, 128}, kLinear, 2},
            }};
            std::array<Image, 14> results;
            for (std::size_t i = 0; i < cases.size(); ++i) results[i] = run_color_case(check, cases[i]);

            // Negative control: the mirrored result is not the identity result, and the comparison says so.
            const Diff wrong = compare(results[4], results[0], 0);
            check.expect(wrong.bad > 1000, "negative control: the mirrored blit differs from the unmirrored one in more than 1000 pixels (the comparison can fail)",
                         std::format("{} pixels differ", wrong.bad));
            // mask == 0: "no buffers are copied" and it is not an error. meta-gl has no zero enumerator, but the
            // intersection of two different mask bits is zero.
            {
                reset_dst();
                const Image before = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
                bind_pair(src_, dst_);
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                (void)demo::drain_gl_errors();
                blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit & kDepthBit, kNearest);
                check.expect_no_gl_error("blit with mask 0 raises no error");
                expect_image(check, "blit with mask 0 copies nothing", read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst), before, 0);
            }
            // The whole-chart 1:1 copy equals the chart (the model is anchored to the source).
            Image block(kSrc, kSrc);
            for (int y = 0; y < kSrc; ++y)
                for (int x = 0; x < kSrc; ++x) block.at(x, y) = results[0].at(x, y);
            expect_image(check, "the 64 x 64 corner of the 1:1 copy is the chart itself", block, chart_img_[0], 0);
        }

        // glReadBuffer selects the blit source; every draw buffer is a destination (ES 3.0 4.3.3).
        void check_read_buffer(demo::Check& check)
        {
            reset_dst();
            bind_pair(src_, dst_);
            set_draw_buffers({kDraw0, kDrawNone, kDraw2});
            for (std::size_t source : {1u, 2u})
            {
                bind_read(src_);
                mgl::glReadBuffer(mgl::to_read_buffer(kAtt[source]));
                check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), gl(static_cast<GLenum>(kAtt[source])), std::format("READ_BUFFER is COLOR_ATTACHMENT{}", source));
                bind_draw(dst_);
                (void)demo::drain_gl_errors();
                blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit, kNearest);
                check.expect_no_gl_error("blit with draw buffers {C0, NONE, C2} raises no error");

                const Image d0 = read_attachment(dst_, kAtt[0], kDst, kDst), d1 = read_attachment(dst_, kAtt[1], kDst, kDst), d2 = read_attachment(dst_, kAtt[2], kDst, kDst);
                const Image sentinel0(kDst, kDst, kSentinel[0]), sentinel1(kDst, kDst, kSentinel[1]), sentinel2(kDst, kDst, kSentinel[2]);
                const Rect full = {0, 0, kSrc, kSrc};
                const Image want = model_color_blit(sentinel0, chart_img_[source], full, full, kNearest);
                expect_image(check, std::format("read buffer C{} is the source: draw buffer C0 receives chart {}", source, source), d0, want, 0);
                expect_image(check, std::format("read buffer C{} is the source: draw buffer C2 receives chart {} as well", source, source), d2,
                             model_color_blit(sentinel2, chart_img_[source], full, full, kNearest), 0);
                expect_image(check, "the NONE draw buffer (C1) keeps its sentinel", d1, sentinel1, 0);
            }
            // READ_BUFFER is per framebuffer object: another framebuffer still has its initial value.
            bind_read(ds_src_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), gl(GL_COLOR_ATTACHMENT0), "another framebuffer's READ_BUFFER is still COLOR_ATTACHMENT0");
            bind_read(src_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), gl(GL_COLOR_ATTACHMENT2), "the chart framebuffer kept its READ_BUFFER (COLOR_ATTACHMENT2)");
            set_draw_buffers({kDraw0});
        }

        // Blits ignore the write masks and the fragment tests but not the scissor test (ES 3.0 4.3.3).
        void check_scissor_and_masks(demo::Check& check)
        {
            const Rect s = {0, 0, kSrc, kSrc}, d = {0, 0, kDst, kDst};
            const Rect box = {20, 30, 60, 55};  // x, y, x + width, y + height
            {
                reset_dst();
                const Image before = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
                bind_pair(src_, dst_);
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                mgl::glEnable(mgl::Capability::ScissorTest);
                mgl::glScissor(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
                (void)demo::drain_gl_errors();
                blit(s, d, kColorBit, kNearest);
                check.expect_no_gl_error("blit with the scissor test enabled raises no error");
                mgl::glDisable(mgl::Capability::ScissorTest);
                const Image after = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
                expect_image(check, "scissor test: only the scissor box (20,30)-(60,55) of the 2x blit is written", after,
                             model_color_blit(before, chart_img_[0], s, d, kNearest, box), 0);
                check.expect(count_differences(after, model_color_blit(before, chart_img_[0], s, d, kNearest)) > 1000,
                             "negative control: without the scissor the model would write more than 1000 further pixels");
            }
            {
                reset_dst();
                const Image before = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
                bind_pair(src_, dst_);
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                (void)demo::drain_gl_errors();
                blit(s, d, kColorBit, kNearest);
                check.expect_no_gl_error("blit with all color writes masked raises no error");
                mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                const Image after = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
                expect_image(check, "glColorMask(FALSE x4) is ignored: the blit still writes every pixel", after, model_color_blit(before, chart_img_[0], s, d, kNearest), 0);
            }
            reset_fragment_state();
        }

        // Format conversion and integer buffers (ES 3.0 4.3.3).
        void check_conversion_and_integers(demo::Check& check)
        {
            // RGBA8 -> RGB565: "pixel groups are converted to match the destination format".
            {
                bind_both(conv_);
                reset_fragment_state();
                mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                mgl::glClear(kColorBit);
                bind_pair(src_, conv_);
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                (void)demo::drain_gl_errors();
                blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit, kNearest);
                check.expect_no_gl_error("blit RGBA8 -> RGB565 (fixed point to fixed point) raises no error");
                const Image got = read_attachment(conv_, mgl::ColorAttachment::Color0, kSrc, kSrc);
                // Equation 2.3 (ES 3.0 2.1.6.2) converts to "one of the two integers closest to f * (2^b - 1)": the
                // result is the level just below or just above the exact value. The levels are read back as 8-bit
                // values (round(level * 255 / (2^b - 1))), so the two neighbours are allowed, +-1 for that rounding.
                int violations = 0, alpha_wrong = 0;
                std::string first;
                for (int y = 0; y < kSrc; ++y)
                    for (int x = 0; x < kSrc; ++x)
                    {
                        const Rgba8 c = chart_img_[0].at(x, y), g = got.at(x, y);
                        const std::array<int, 3> bits = {5, 6, 5};
                        for (std::size_t ch = 0; ch < 3; ++ch)
                        {
                            const double levels = static_cast<double>((1 << bits[ch]) - 1);
                            const double exact = c[ch] * levels / 255.0;
                            const double lo = std::floor(exact) * 255.0 / levels, hi = std::ceil(exact) * 255.0 / levels;
                            if (g[ch] < lo - 1.0 || g[ch] > hi + 1.0)
                            {
                                if (violations++ == 0) first = std::format("({},{}) channel {}: source {} read back {}, allowed [{:.1f}, {:.1f}]", x, y, ch, c[ch], g[ch], lo, hi);
                            }
                        }
                        alpha_wrong += g[3] == 255 ? 0 : 1;
                    }
                check.expect(violations == 0 && alpha_wrong == 0, "RGBA8 -> RGB565: every channel is one of the two 5/6/5-bit levels next to the source value, alpha reads back 1",
                             violations == 0 ? std::string{} : std::format("{} violations, {}", violations, first));
                check.expect(compare(got, chart_img_[0], 1).bad > 1000, "negative control: the converted copy differs from the unconverted chart in over 1000 pixels");
            }
            // Integer buffers: copies are exact; mixing integer and fixed-point or unsigned and signed is an error.
            fill_integer(ui_src_, false);
            fill_integer(ui_dst_, true);
            fill_integer_signed(si_);
            const auto src_values = read_uint(ui_src_);
            const auto dst_before = read_uint(ui_dst_);
            check.expect(src_values != dst_before, "integer fixture: source and destination start different");

            bind_pair(ui_src_, ui_dst_);
            (void)demo::drain_gl_errors();
            blit({0, 0, kSmall, kSmall}, {0, 0, kSmall, kSmall}, kColorBit, kNearest);
            check.expect_no_gl_error("blit RGBA8UI -> RGBA8UI with NEAREST raises no error");
            check.expect(read_uint(ui_dst_) == src_values, "RGBA8UI -> RGBA8UI copies every unsigned integer exactly");

            fill_integer(ui_dst_, true);
            bind_pair(ui_src_, ui_dst_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "LINEAR with an integer read buffer -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSmall, kSmall}, {0, 0, kSmall, kSmall}, kColorBit, kLinear); });
            bind_pair(ui_src_, si_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "unsigned integer read buffer, signed integer draw buffer -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSmall, kSmall}, {0, 0, kSmall, kSmall}, kColorBit, kNearest); });
            bind_pair(si_, ui_dst_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "signed integer read buffer, unsigned integer draw buffer -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSmall, kSmall}, {0, 0, kSmall, kSmall}, kColorBit, kNearest); });
            bind_pair(src_, ui_dst_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            expect_error(check, mgl::ErrorCode::InvalidOperation, "fixed-point read buffer, unsigned integer draw buffer -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSmall, kSmall}, {0, 0, kSmall, kSmall}, kColorBit, kNearest); });
            bind_pair(ui_src_, dst_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "unsigned integer read buffer, fixed-point draw buffer -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSmall, kSmall}, {0, 0, kSmall, kSmall}, kColorBit, kNearest); });
            check.expect(read_uint(ui_dst_) == dst_before, "failed blits changed nothing: the integer destination is unchanged");
        }

        // The integer fixtures: a source is (200,100,50,25) with a scissored block of (1,2,3,4); a destination is all 9.
        void fill_integer(mgl::FramebufferId f, bool destination) const
        {
            reset_fragment_state();
            bind_both(f);
            const std::array<GLuint, 4> whole = destination ? std::array<GLuint, 4>{9, 9, 9, 9} : std::array<GLuint, 4>{200, 100, 50, 25};
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, whole.data());
            if (!destination)
            {
                const std::array<GLuint, 4> block = {1, 2, 3, 4};
                mgl::glEnable(mgl::Capability::ScissorTest);
                mgl::glScissor(8, 8, 16, 16);
                mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, block.data());
                mgl::glDisable(mgl::Capability::ScissorTest);
            }
        }

        void fill_integer_signed(mgl::FramebufferId f) const
        {
            reset_fragment_state();
            bind_both(f);
            const std::array<GLint, 4> v = {-5, -6, -7, -8};
            mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 0, v.data());
        }

        static std::vector<std::uint32_t> read_uint(mgl::FramebufferId f)
        {
            bind_read(f);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            std::vector<std::uint32_t> px(static_cast<std::size_t>(kSmall * kSmall * 4));
            mgl::glReadPixels(0, 0, kSmall, kSmall, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, px.data());
            return px;
        }

        // ------------------------------------------------------------------------------------------
        /// Reads ds_dst_-like targets: color first (the probes overwrite it), then depth and stencil through draws.
        struct DsResult
        {
            Image color;
            std::vector<std::uint8_t> bits;
        };

        DsResult observe(mgl::FramebufferId f, int size) const
        {
            DsResult r;
            r.color = read_attachment(f, mgl::ColorAttachment::Color0, size, size);
            r.bits = probe_ds(f, size);
            return r;
        }

        void run_ds_case(demo::Check& check, std::string_view name, const Rect& s, const Rect& d, mgl::ClearBufferBit mask, bool hostile_state = false)
        {
            reset_ds_target(ds_dst_);
            const Image before = read_attachment(ds_dst_, mgl::ColorAttachment::Color0, kDst, kDst);
            bind_pair(ds_src_, ds_dst_);
            if (hostile_state)
            {
                // Masks and tests that would stop every fragment: a blit bypasses the fragment pipeline.
                mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                mgl::glDepthMask(GL_FALSE);
                mgl::glStencilMask(0u);
                mgl::glEnable(mgl::Capability::DepthTest);
                mgl::glDepthFunc(mgl::CompareFunc::Never);
                mgl::glEnable(mgl::Capability::StencilTest);
                mgl::glStencilFunc(mgl::CompareFunc::Never, 0, 0xFFu);
            }
            (void)demo::drain_gl_errors();
            blit(s, d, mask, kNearest);
            check.expect_no_gl_error(std::format("blit '{}' raises no error", name));
            reset_fragment_state();

            const bool color = (mask & kColorBit) == kColorBit;
            const bool depth = (mask & kDepthBit) == kDepthBit;
            const bool stencil = (mask & kStencilBit) == kStencilBit;
            const DsResult got = observe(ds_dst_, kDst);
            const Image want_color = color ? model_color_blit(before, ds_src_img_, s, d, kNearest) : before;
            const DsImage want_ds = model_ds_blit(kDst, kDst, s, d, depth, stencil);
            expect_image(check, std::format("'{}': color {}", name, color ? "equals the CPU model of the blit" : "is untouched (COLOR_BUFFER_BIT not in the mask)"),
                         got.color, want_color, 0);
            expect_bits(check, std::format("'{}': depth {} and stencil {} (probed with tested draws)", name, depth ? "copied" : "untouched", stencil ? "copied" : "untouched"),
                        got.bits, expected_bits(want_ds));
        }

        void check_depth_stencil_blits(demo::Check& check)
        {
            const Rect s = {0, 0, kSrc, kSrc};
            run_ds_case(check, "color+depth+stencil 1:1", s, {0, 0, kSrc, kSrc}, kColorBit | kDepthBit | kStencilBit);
            run_ds_case(check, "depth+stencil 2x NEAREST", s, {0, 0, kDst, kDst}, kDepthBit | kStencilBit);
            run_ds_case(check, "depth+stencil mirrored in x and y", s, {kSrc, kSrc, 0, 0}, kDepthBit | kStencilBit);
            run_ds_case(check, "depth only", s, {0, 0, kDst, kDst}, kDepthBit);
            run_ds_case(check, "stencil only", s, {0, 0, kDst, kDst}, kStencilBit);
            run_ds_case(check, "everything while write masks and failing tests are active", s, {0, 0, kDst, kDst}, kColorBit | kDepthBit | kStencilBit, true);

            // Negative control for the probes: an untouched target must look different from a blitted one.
            reset_ds_target(ds_dst_);
            const DsResult pristine = observe(ds_dst_, kDst);
            const std::vector<std::uint8_t> blitted = expected_bits(model_ds_blit(kDst, kDst, s, {0, 0, kDst, kDst}, true, true));
            check.expect(count_mismatches(pristine.bits, blitted) > 1000, "negative control: a cleared target and a blitted one differ in over 1000 probed pixels",
                         std::format("{} differ", count_mismatches(pristine.bits, blitted)));
            expect_bits(check, "a cleared target probes as depth 1, stencil 0 everywhere", pristine.bits, expected_bits(DsImage(kDst, kDst)));

            // LINEAR is allowed only for color: no copy at all when depth or stencil is in the mask.
            const std::vector<std::uint8_t> cleared = expected_bits(DsImage(kDst, kDst));
            struct LinearCase
            {
                std::string_view name;
                mgl::ClearBufferBit mask;
            };
            for (const LinearCase& c : {LinearCase{"DEPTH_BUFFER_BIT", kDepthBit}, LinearCase{"STENCIL_BUFFER_BIT", kStencilBit},
                                        LinearCase{"COLOR + DEPTH", kColorBit | kDepthBit}, LinearCase{"COLOR + STENCIL", kColorBit | kStencilBit}})
            {
                reset_ds_target(ds_dst_);
                const Image before = read_attachment(ds_dst_, mgl::ColorAttachment::Color0, kDst, kDst);
                bind_pair(ds_src_, ds_dst_);
                expect_error(check, mgl::ErrorCode::InvalidOperation, std::format("LINEAR with {} -> INVALID_OPERATION", c.name),
                             [&] { blit(s, {0, 0, kDst, kDst}, c.mask, kLinear); });
                const DsResult got = observe(ds_dst_, kDst);
                expect_image(check, std::format("LINEAR with {}: no copy is performed (color unchanged)", c.name), got.color, before, 0);
                expect_bits(check, std::format("LINEAR with {}: no copy is performed (depth and stencil unchanged)", c.name), got.bits, cleared);
            }
            // The same rectangle with NEAREST works, so the error above is about the filter alone.
            run_ds_case(check, "NEAREST with the mask that LINEAR rejects", s, {0, 0, kDst, kDst}, kColorBit | kDepthBit | kStencilBit);

            // Mismatching depth formats, and buffers missing from one framebuffer.
            reset_ds_target(ds_dst16_);
            bind_pair(ds_src_, ds_dst16_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "DEPTH24_STENCIL8 -> DEPTH_COMPONENT16 depth blit: the formats do not match -> INVALID_OPERATION",
                         [&] { blit(s, {0, 0, kSrc, kSrc}, kDepthBit, kNearest); });
            reset_ds_target(ds_dst16_);
            const Image before16 = read_attachment(ds_dst16_, mgl::ColorAttachment::Color0, kDst, kDst);
            bind_pair(ds_src_, ds_dst16_);
            (void)demo::drain_gl_errors();
            blit(s, {0, 0, kSrc, kSrc}, kColorBit | kStencilBit, kNearest);
            check.expect_no_gl_error("STENCIL_BUFFER_BIT with no stencil buffer in the draw framebuffer is silently ignored");
            expect_image(check, "...and the COLOR_BUFFER_BIT of the same call is still copied", read_attachment(ds_dst16_, mgl::ColorAttachment::Color0, kDst, kDst),
                         model_color_blit(before16, ds_src_img_, s, {0, 0, kSrc, kSrc}, kNearest), 0);

            reset_dst();
            const Image before = read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst);
            bind_pair(ds_src_, dst_);
            (void)demo::drain_gl_errors();
            blit(s, {0, 0, kSrc, kSrc}, kColorBit | kDepthBit | kStencilBit, kNearest);
            check.expect_no_gl_error("DEPTH and STENCIL bits with no depth/stencil buffer in the draw framebuffer are silently ignored");
            expect_image(check, "...and the color of the same call is copied", read_attachment(dst_, mgl::ColorAttachment::Color0, kDst, kDst),
                         model_color_blit(before, ds_src_img_, s, {0, 0, kSrc, kSrc}, kNearest), 0);
        }

        // ------------------------------------------------------------------------------------------
        // glGetInternalformativ and glRenderbufferStorageMultisample (ES 3.0 4.4.2.1, 6.1.15).
        void check_internalformat_queries(demo::Env& env, demo::Check& check)
        {
            const int max_samples = demo::get_integer(mgl::GetParameter::MaxSamples);
            struct Fmt
            {
                mgl::InternalFormat format;
                std::string_view name;
            };
            const std::array<Fmt, 15> formats = {{{mgl::InternalFormat::Rgba8, "RGBA8"}, {mgl::InternalFormat::Srgb8Alpha8, "SRGB8_ALPHA8"}, {mgl::InternalFormat::Rgb8, "RGB8"},
                                                  {mgl::InternalFormat::Rgb565, "RGB565"}, {mgl::InternalFormat::Rgba4, "RGBA4"}, {mgl::InternalFormat::Rgb5A1, "RGB5_A1"},
                                                  {mgl::InternalFormat::Rgb10A2, "RGB10_A2"}, {mgl::InternalFormat::R8, "R8"}, {mgl::InternalFormat::Rg8, "RG8"},
                                                  {mgl::InternalFormat::DepthComponent16, "DEPTH_COMPONENT16"}, {mgl::InternalFormat::DepthComponent24, "DEPTH_COMPONENT24"},
                                                  {mgl::InternalFormat::DepthComponent32F, "DEPTH_COMPONENT32F"}, {mgl::InternalFormat::Depth24Stencil8, "DEPTH24_STENCIL8"},
                                                  {mgl::InternalFormat::Depth32FStencil8, "DEPTH32F_STENCIL8"}, {mgl::InternalFormat::StencilIndex8, "STENCIL_INDEX8"}}};
            for (const Fmt& f : formats)
            {
                (void)demo::drain_gl_errors();
                GLint n = -1;
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, f.format, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
                std::vector<GLint> counts(static_cast<std::size_t>(std::max(n, 0)), -1);
                if (n > 0)
                    mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, f.format, mgl::InternalFormatParameter::Samples, n, counts.data());
                const bool no_error = check.expect_no_gl_error(std::format("{}: both queries raise no error", f.name));
                bool descending = !counts.empty(), positive = true;
                for (std::size_t i = 0; i < counts.size(); ++i)
                {
                    positive = positive && counts[i] > 0;
                    if (i > 0) descending = descending && counts[i - 1] > counts[i];
                }
                std::string list;
                for (const GLint c : counts) list += std::format("{} ", c);
                check.expect(no_error && n >= 1 && descending && positive && counts.front() >= max_samples,
                             std::format("{}: NUM_SAMPLE_COUNTS = {} >= 1, SAMPLES strictly descending and positive, largest >= MAX_SAMPLES ({})", f.name, n, max_samples),
                             std::format("SAMPLES = {}", list));
            }

            // How much is written: bufSize bounds the output, the rest is not touched (ES 3.0 6.1.15).
            const std::vector<GLint>& counts = rgba8_counts_;
            const GLint n = static_cast<GLint>(counts.size());
            std::array<GLint, 12> buffer{};
            auto fill = [&] { buffer.fill(-7); };
            fill();
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, 1, buffer.data());
            check.expect(buffer[0] == counts.front() && buffer[1] == -7, "SAMPLES with bufSize 1 returns just the maximum and writes nothing else", std::format("max = {}", buffer[0]));
            fill();
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, 0, buffer.data());
            check.expect(buffer[0] == -7, "SAMPLES with bufSize 0 writes nothing");
            fill();
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, 10, buffer.data());
            bool tail_untouched = true, head_ok = true;
            for (GLint i = 0; i < n; ++i) head_ok = head_ok && buffer[static_cast<std::size_t>(i)] == counts[static_cast<std::size_t>(i)];
            for (std::size_t i = static_cast<std::size_t>(n); i < buffer.size(); ++i) tail_untouched = tail_untouched && buffer[i] == -7;
            check.expect(head_ok && tail_untouched, "SAMPLES with a larger bufSize writes exactly NUM_SAMPLE_COUNTS values and nothing after them");
            check.expect_no_gl_error("the partial queries raise no error");

            // Errors: a negative bufSize is INVALID_VALUE, a format that is not renderable INVALID_ENUM.
            expect_error(check, mgl::ErrorCode::InvalidValue, "glGetInternalformativ with a negative bufSize -> INVALID_VALUE", [&] {
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, -1, buffer.data());
            });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "glGetInternalformativ for RGB9_E5 (not renderable) -> INVALID_ENUM", [&] {
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, mgl::InternalFormat::Rgb9E5, mgl::InternalFormatParameter::NumSampleCounts, 1, buffer.data());
            });

            // Integer formats: ES 3.0 gives them no multisample support at all (NUM_SAMPLE_COUNTS is zero and a
            // multisample allocation is INVALID_OPERATION); ES 3.1 allows up to MAX_INTEGER_SAMPLES (an allocation with
            // the largest reported count must work). The rule that applies is the one of the context's version.
            const bool at_least_31 = env.context_version >= demo::Version{3, 1};
            mgl::RenderbufferId int_rb;
            mgl::glGenRenderbuffers(1, &int_rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, int_rb);
            for (const Fmt& f : {Fmt{mgl::InternalFormat::Rgba8UI, "RGBA8UI"}, Fmt{mgl::InternalFormat::Rgba8I, "RGBA8I"}, Fmt{mgl::InternalFormat::Rgba32UI, "RGBA32UI"},
                                 Fmt{mgl::InternalFormat::Rg32I, "RG32I"}})
            {
                const std::vector<GLint> c = sample_counts(f.format);
                if (at_least_31)
                {
                    check.expect(!c.empty(), std::format("{}: an ES 3.1+ context offers multisample integer renderbuffers (NUM_SAMPLE_COUNTS = {})", f.name, c.size()));
                    if (c.empty()) continue;
                    (void)demo::drain_gl_errors();
                    mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, c.front(), f.format, 16, 16);
                    check.expect_no_gl_error(std::format("{}: allocating with the largest reported count ({}) works", f.name, c.front()));
                }
                else
                {
                    check.expect(c.empty(), std::format("{}: ES 3.0 reports NUM_SAMPLE_COUNTS = 0 for integer formats (got {})", f.name, c.size()));
                    expect_error(check, mgl::ErrorCode::InvalidOperation, std::format("{}: glRenderbufferStorageMultisample with samples > 0 -> INVALID_OPERATION (ES 3.0)", f.name),
                                 [&] { mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, 4, f.format, 16, 16); });
                }
            }
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glDeleteRenderbuffers(1, &int_rb);
            check.expect_no_gl_error("the integer-format queries raise no further error");

            // Allocate with every reported count: no error, RENDERBUFFER_SAMPLES >= the request and no more than the next larger supported count.
            mgl::RenderbufferId rb;
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            bool all_ok = true;
            std::string detail;
            for (std::size_t i = 0; i < counts.size(); ++i)
            {
                (void)demo::drain_gl_errors();
                mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, counts[i], mgl::InternalFormat::Rgba8, 16, 16);
                GLint actual = -1;
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &actual);
                const GLint upper = i == 0 ? counts[0] : counts[i - 1];
                const bool ok = demo::drain_gl_errors() == mgl::ErrorCode::NoError && actual >= counts[i] && actual <= upper;
                all_ok = all_ok && ok;
                detail += std::format("{}->{} ", counts[i], actual);
            }
            check.expect(all_ok, "RGBA8 renderbuffers allocate with every reported count and RENDERBUFFER_SAMPLES is between the request and the next larger count", detail);

            mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, counts.front(), mgl::InternalFormat::Rgba8, 16, 16);
            GLint largest = -1, width = -1;
            mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &largest);
            mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Width, &width);
            check.expect(largest == counts.front() && width == 16, "allocating with the largest reported count gives exactly that many samples (no larger count exists)",
                         std::format("requested {}, got {}", counts.front(), largest));
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 16, 16);
            GLint single = -1;
            mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &single);
            check.expect_eq(single, 0, "glRenderbufferStorage (the single-sample call) leaves RENDERBUFFER_SAMPLES at 0");
            for (const Fmt& f : {Fmt{mgl::InternalFormat::Depth24Stencil8, "DEPTH24_STENCIL8"}, Fmt{mgl::InternalFormat::DepthComponent16, "DEPTH_COMPONENT16"},
                                 Fmt{mgl::InternalFormat::StencilIndex8, "STENCIL_INDEX8"}, Fmt{mgl::InternalFormat::Srgb8Alpha8, "SRGB8_ALPHA8"}})
            {
                const std::vector<GLint> c = sample_counts(f.format);
                (void)demo::drain_gl_errors();
                mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, c.front(), f.format, 16, 16);
                GLint actual = -1;
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &actual);
                check.expect(demo::drain_gl_errors() == mgl::ErrorCode::NoError && actual == c.front(),
                             std::format("{}: allocating with the largest reported count ({}) gives {} samples", f.name, c.front(), actual));
            }
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glDeleteRenderbuffers(1, &rb);
        }

        // ------------------------------------------------------------------------------------------
        // Multisample resolve (ES 3.0 4.3.3, 4.4.4.2).
        void check_multisample_resolve(demo::Check& check)
        {
            // SAMPLES and SAMPLE_BUFFERS come from the draw framebuffer's attachments (4.4.4.2).
            bind_draw(ms_);
            const int samples = demo::get_integer(mgl::GetParameter::Samples);
            check.expect(samples >= rgba8_counts_.front() && demo::get_integer(mgl::GetParameter::SampleBuffers) == 1,
                         std::format("the multisample framebuffer as DRAW_FRAMEBUFFER: SAMPLE_BUFFERS = 1, SAMPLES = {} >= {} requested", samples, rgba8_counts_.front()));
            GLint rb_samples = -1;
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, ms_rb_);
            mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &rb_samples);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            check.expect_eq(rb_samples, rgba8_counts_.front(), "RENDERBUFFER_SAMPLES of the color renderbuffer equals the largest reported count");
            check.expect_eq(samples, rb_samples, "SAMPLES equals RENDERBUFFER_SAMPLES of the attached image");
            bind_draw(res_);
            check.expect(demo::get_integer(mgl::GetParameter::Samples) == 0 && demo::get_integer(mgl::GetParameter::SampleBuffers) == 0,
                         "a texture-backed framebuffer has SAMPLE_BUFFERS = 0 and SAMPLES = 0");

            // The resolved image. ES 3.0 4.3.3: "If the source formats are floating-point or normalized types, the sample values
            // for each pixel are resolved in an implementation-dependent manner." Nothing about the value of a resolved RGBA8
            // pixel is guaranteed - not even that a pixel whose samples are all equal resolves to that value - so everything
            // measured on the resolved triangle is printed as an observation, not asserted.
            const TriangleStats ms = analyze_triangle(res_img_);
            const TriangleStats control = analyze_triangle(ss_img_);
            const Triangle tri;
            std::cout << std::format("   note: observed RGBA8 resolve ({} samples, implementation-dependent, not asserted): interior pixels off the triangle color "
                                     "by more than 1: {} of {}; exterior pixels off the background: {} of {}; edge pixels outside the range of the two colors: "
                                     "{} of {}; edge pixels strictly between the colors: {} ({} of {} centred on an edge); summed coverage {:.1f} for a "
                                     "triangle area of {:.1f} ({:+.3f}%)\n",
                                     samples, ms.interior_bad, ms.interior, ms.exterior_bad, ms.exterior, ms.edge_outside_range, ms.edge, ms.intermediate,
                                     ms.centered_strictly_between, ms.centered, ms.covered_area, tri.area(), 100.0 * (ms.covered_area - tri.area()) / tri.area());
            // The single-sample control is fully specified: point sampling at the pixel centre (ES 3.0 3.6.1) and a flat color.
            check.expect(control.interior > 5000 && control.exterior > 5000 && control.edge > 200,
                         "the triangle splits the 128 x 128 target into plenty of interior, exterior and edge pixels (CPU classification)",
                         std::format("{} / {} / {}", control.interior, control.exterior, control.edge));
            check.expect(control.interior_bad == 0 && control.exterior_bad == 0,
                         "single-sample render of the triangle: interior pixels equal the triangle color, exterior pixels the background (tolerance 1)",
                         std::format("{} interior and {} exterior pixels differ", control.interior_bad, control.exterior_bad));
            check.expect(control.intermediate == 0 && control.centered_strictly_between == 0,
                         "single-sample render of the triangle: no pixel is a blend (each pixel centre is either covered or not)",
                         std::format("{} blended pixels", control.intermediate));

            // The filter is ignored when resolving (4.3.3): LINEAR gives the same image as NEAREST.
            clear_target(res_);
            bind_pair(ms_, res_);
            (void)demo::drain_gl_errors();
            blit({0, 0, kDst, kDst}, {0, 0, kDst, kDst}, kColorBit, kLinear);
            check.expect_no_gl_error("a resolve with filter LINEAR raises no error");
            expect_image(check, "resolving with LINEAR gives the same image as NEAREST (the filter is ignored)", read_attachment(res_, mgl::ColorAttachment::Color0, kDst, kDst),
                         res_img_, 0);
            // A resolve of a sub-rectangle with identical bounds is allowed and resolves just that rectangle.
            clear_target(res_);
            bind_pair(ms_, res_);
            const Rect part = {20, 30, 100, 90};
            blit(part, part, kColorBit, kNearest);
            check.expect_no_gl_error("a resolve of the sub-rectangle (20,30)-(100,90) with identical bounds raises no error");
            Image want(kDst, kDst, {0, 0, 0, 255});
            for (int y = part.y0; y < part.y1; ++y)
                for (int x = part.x0; x < part.x1; ++x) want.at(x, y) = res_img_.at(x, y);
            // ES 3.0 4.3.3: "Only pixels whose centers lie within the destination rectangle are written by BlitFramebuffer." That
            // part is normative and asserted exactly. The values inside the rectangle are an RGBA8 resolve, which is
            // implementation-dependent - nothing requires a sub-rectangle resolve to reproduce the full resolve bit for bit - so
            // the inside is only compared as an observation. The integer resolve check (ES 3.1) asserts the inside normatively.
            {
                const Image got = read_attachment(res_, mgl::ColorAttachment::Color0, kDst, kDst);
                const demo::Rgba8 cleared = {0, 0, 0, 255};
                int inside = 0, inside_same = 0, outside = 0, outside_written = 0, outside_as_resolved = 0;
                std::string first_written;
                for (int y = 0; y < kDst; ++y)
                    for (int x = 0; x < kDst; ++x)
                    {
                        if (x >= part.x0 && x < part.x1 && y >= part.y0 && y < part.y1)
                        {
                            ++inside;
                            inside_same += got.at(x, y) == res_img_.at(x, y) ? 1 : 0;
                            continue;
                        }
                        ++outside;
                        if (got.at(x, y) == cleared) continue;
                        outside_as_resolved += got.at(x, y) == res_img_.at(x, y) ? 1 : 0;
                        if (outside_written++ == 0) first_written = std::format("first at ({},{}): {}", x, y, demo::format_rgba(got.at(x, y)));
                    }
                check.expect_or_known_deviation(
                    outside_written == 0, "...and writes no pixel outside that rectangle (every other pixel keeps the clear color)",
                    outside_written == 0 ? std::format("{} pixels outside", outside)
                                         : std::format("{} of {} pixels outside were written ({} of them equal the full resolve), {}", outside_written, outside,
                                                       outside_as_resolved, first_written),
                    "ANGLE", "ANGLE writes resolved pixels outside the destination rectangle of a multisample resolve blit (ES 3.0 4.3.3: only pixels whose centers lie within the destination rectangle are written)");
                std::cout << std::format("   note: observed sub-rectangle RGBA8 resolve (implementation-dependent, not asserted): {} of {} pixels inside the "
                                         "rectangle equal the full resolve\n", inside_same, inside);
            }

            // Reading the samples of a multisample framebuffer directly is an error; a resolve is the only way (4.3.2).
            bind_read(ms_);
            std::array<std::uint8_t, 4> px{};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glReadPixels from a multisample framebuffer -> INVALID_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data()); });
        }

        // A depth/stencil resolve (ES 3.0 4.3.3): "If the source formats are integer types or stencil values, a single sample's
        // value is selected for each pixel. [...] If the source formats are depth values, sample values are resolved in an
        // implementation-dependent manner where the result will be between the minimum and maximum depth values in the pixel."
        // Both rules pin the result down exactly when all samples of a pixel are equal. The scene is therefore written with
        // scissored clears, not with draws: the scissor test selects whole pixels, and "masking and scissoring affect clearing
        // the multisample buffer in the same way as they affect clearing the corresponding color, depth, and stencil buffers"
        // (4.2.3.1), so every sample of a pixel receives its region's value whatever the sample locations are (3.3 allows them
        // outside the pixel square, which would make a drawn rectangle's border pixels mixed).
        void check_multisample_depth_stencil(demo::Check& check)
        {
            bind_both(ds_ms_);
            clear_ds_scene(kSrc);
            reset_ds_target(ds_res_);
            bind_pair(ds_ms_, ds_res_);
            (void)demo::drain_gl_errors();
            blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit | kDepthBit | kStencilBit, kNearest);
            check.expect_no_gl_error("resolving color, depth and stencil of a multisample framebuffer raises no error");
            const DsResult got = observe(ds_res_, kSrc);
            DsImage want(kSrc, kSrc);
            for (int y = 0; y < kSrc; ++y)
                for (int x = 0; x < kSrc; ++x) want.at(x, y) = scene_pixel(x, y);
            expect_bits(check, "resolved depth and stencil of the uniformly cleared regions equal the scene (probed with tested draws)", got.bits, expected_bits(want));
            check.expect(count_mismatches(got.bits, expected_bits(DsImage(kSrc, kSrc))) > 1500,
                         "negative control: the resolved buffer differs from a cleared one in over 1500 pixels");
            // The RGBA8 color of the same pixels is a normalized format: its resolve is implementation-dependent, even here.
            const Diff color = compare(got.color, ds_src_img_, 1);
            std::cout << std::format("   note: observed RGBA8 resolve of uniformly cleared pixels (implementation-dependent, not asserted): {} of {} pixels differ "
                                     "from the region color by more than 1 (max {})\n",
                                     color.bad, kSrc * kSrc, color.max_error);
        }

        // ------------------------------------------------------------------------------------------
        // The integer resolve, with every sample read back (ES 3.1+ contexts).
        //
        // ES 3.0 4.3.3 (ES 3.2 16.2.1): "If the source formats are integer types or stencil values, a single sample's value is
        // selected for each pixel." This is the one color resolve the specification pins down. ES 3.0 has no multisample integer
        // images, so the check needs ES 3.1: the triangle goes into an RGBA8UI multisample *texture*, texelFetch(usampler2DMS,
        // texel, sample) reads every stored sample into a single-sample target (no resolve involved), and the blit resolve of the
        // same texture must give, in every pixel, the value of one of that pixel's samples. Nothing here depends on where the
        // samples lie, how many of them an edge covers, or how normalized colors are combined.
        void check_integer_resolve(const demo::Env& env, demo::Check& check)
        {
            if (env.context_version < demo::Version{3, 1})
            {
                std::cout << "   note: ES 3.0 context: multisample integer images and texelFetch(usampler2DMS) are ES 3.1, the integer resolve check is not made\n";
                return;
            }
            GLint n = 0;
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8UI, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
            std::vector<GLint> counts(static_cast<std::size_t>(std::max(n, 0)));
            if (n > 0)
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8UI, mgl::InternalFormatParameter::Samples, n,
                                           counts.data());
            if (counts.empty() || counts.front() < 2)
            {
                std::cout << "   note: RGBA8UI multisample textures offer fewer than 2 samples here, the integer resolve check is not made\n";
                return;
            }
            const int requested = counts.front();

            // Targets: the multisample texture, its resolve target and the target that receives one sample per pass.
            const mgl::FramebufferId ms = make_fbo();
            mgl::TextureId ms_tex;
            mgl::glGenTextures(1, &ms_tex);
            textures_.push_back(ms_tex);
            mgl::glBindTexture(mgl::TextureBindingTarget::Texture2DMultisample, ms_tex);
            mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2DMultisample, requested, mgl::SizedInternalFormat::Rgba8UI, kDst, kDst, GL_TRUE);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2DMultisample, ms_tex, 0);
            require_complete("RGBA8UI multisample texture");
            const int samples = demo::get_integer(mgl::GetParameter::Samples);
            check.expect(demo::get_integer(mgl::GetParameter::SampleBuffers) == 1 && samples >= requested,
                         std::format("RGBA8UI multisample texture framebuffer: SAMPLE_BUFFERS = 1, SAMPLES = {} >= {} requested", samples, requested));
            const mgl::FramebufferId resolved = make_fbo();
            attach_texture(mgl::ColorAttachment::Color0, make_texture(mgl::SizedInternalFormat::Rgba8UI, kDst, kDst));
            require_complete("RGBA8UI resolve target");
            const mgl::FramebufferId fetched = make_fbo();
            attach_texture(mgl::ColorAttachment::Color0, make_texture(mgl::SizedInternalFormat::Rgba8UI, kDst, kDst));
            require_complete("RGBA8UI per-sample target");

            // Background and triangle values differ by far more than 1 in every channel (the negative control relies on it).
            const std::array<GLuint, 4> bg = {10, 20, 30, 40};
            const std::array<GLuint, 4> fg = {200, 150, 100, 250};
            bind_both(ms);
            reset_fragment_state();
            mgl::glViewport(0, 0, kDst, kDst);
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, bg.data());
            const mgl::ProgramId tri_uint = demo::build_program(env, "shaders/tri.vert", "shaders/flat_uint.frag");
            mgl::glUseProgram(tri_uint);
            mgl::glUniform2fv(mgl::glGetUniformLocation(tri_uint, "u_tri"), 3, kTriangle.data());
            mgl::glUniform1f(mgl::glGetUniformLocation(tri_uint, "u_size"), static_cast<float>(kDst));
            mgl::glUniform4ui(mgl::glGetUniformLocation(tri_uint, "u_value"), fg[0], fg[1], fg[2], fg[3]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(tri_uint);

            // The resolve.
            bind_pair(ms, resolved);
            (void)demo::drain_gl_errors();
            blit({0, 0, kDst, kDst}, {0, 0, kDst, kDst}, kColorBit, kNearest);
            check.expect_no_gl_error("resolve blit of the RGBA8UI multisample texture into an RGBA8UI texture (NEAREST) raises no error");
            const std::vector<std::uint32_t> res = read_uint_image(resolved, kDst);

            // Every sample, one pass per sample index.
            const mgl::ProgramId fetch = demo::build_program(env, "shaders/fetch_ms.vert", "shaders/fetch_ms.frag");
            mgl::glUseProgram(fetch);
            mgl::glUniform1i(mgl::glGetUniformLocation(fetch, "u_ms"), 0);
            const mgl::UniformLocation u_sample = mgl::glGetUniformLocation(fetch, "u_sample");
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureBindingTarget::Texture2DMultisample, ms_tex);
            std::vector<std::vector<std::uint32_t>> per_sample;
            for (int s = 0; s < samples; ++s)
            {
                bind_both(fetched);
                mgl::glViewport(0, 0, kDst, kDst);
                mgl::glUniform1i(u_sample, s);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                per_sample.push_back(read_uint_image(fetched, kDst));
            }
            mgl::glBindTexture(mgl::TextureBindingTarget::Texture2DMultisample, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(fetch);
            check.expect_no_gl_error("drawing the triangle into the RGBA8UI multisample texture and fetching its samples raise no error");

            using U4 = std::array<std::uint32_t, 4>;
            auto value_at = [](const std::vector<std::uint32_t>& img, std::size_t pixel) {
                return U4{img[4 * pixel], img[4 * pixel + 1], img[4 * pixel + 2], img[4 * pixel + 3]};
            };
            const U4 bg32 = {bg[0], bg[1], bg[2], bg[3]}, fg32 = {fg[0], fg[1], fg[2], fg[3]};
            // "Is v the value of one of the samples of pixel p?" - the property the specification requires of the resolve.
            auto is_a_sample = [&](std::size_t p, const U4& v) {
                for (const auto& img : per_sample)
                    if (value_at(img, p) == v) return true;
                return false;
            };

            const std::size_t pixels = static_cast<std::size_t>(kDst) * static_cast<std::size_t>(kDst);
            int foreign = 0, fg_samples = 0, bg_samples = 0, mixed = 0, uniform = 0, bad = 0;
            std::vector<int> picked(per_sample.size(), 0);
            std::string first_bad;
            for (std::size_t p = 0; p < pixels; ++p)
            {
                int covered = 0;
                for (const auto& img : per_sample)
                {
                    const U4 v = value_at(img, p);
                    covered += v == fg32 ? 1 : 0;
                    fg_samples += v == fg32 ? 1 : 0;
                    bg_samples += v == bg32 ? 1 : 0;
                    foreign += v != fg32 && v != bg32 ? 1 : 0;
                }
                const bool is_mixed = covered > 0 && covered < samples;
                mixed += is_mixed ? 1 : 0;
                uniform += is_mixed ? 0 : 1;
                const U4 r = value_at(res, p);
                if (!is_a_sample(p, r) && bad++ == 0)
                    first_bad = std::format("pixel ({},{}) resolved to ({},{},{},{})", p % kDst, p / kDst, r[0], r[1], r[2], r[3]);
                if (is_mixed)
                    for (std::size_t s = 0; s < per_sample.size(); ++s) picked[s] += value_at(per_sample[s], p) == r ? 1 : 0;
            }
            check.expect(foreign == 0 && fg_samples > 0 && bg_samples > 0,
                         "every stored sample (texelFetch per sample) is exactly the cleared background or the triangle's value, and both occur",
                         std::format("{} triangle samples, {} background samples, {} other", fg_samples, bg_samples, foreign));
            check.expect(bad == 0, "integer resolve: every resolved pixel equals the value of one of its own samples ('a single sample's value is selected')",
                         bad == 0 ? std::format("{} pixels ({} with mixed samples, {} with all samples equal)", pixels, mixed, uniform)
                                  : std::format("{} of {} pixels violate it, {}", bad, pixels, first_bad));
            // Negative control: the same test rejects a value that is not one of the samples. Every resolved value shifted by one
            // in red is neither the background nor the triangle value, so the test must reject every pixel.
            int rejected = 0;
            for (std::size_t p = 0; p < pixels; ++p)
            {
                U4 r = value_at(res, p);
                r[0] += 1;
                rejected += is_a_sample(p, r) ? 0 : 1;
            }
            check.expect(rejected == static_cast<int>(pixels), "negative control: resolved values shifted by one in red are rejected in every pixel",
                         std::format("{} of {}", rejected, pixels));
            // An averaging resolve would be rejected wherever the samples differ. How many such pixels exist depends on the
            // implementation-dependent sample locations, so this is printed, not asserted.
            int mean_rejected = 0;
            for (std::size_t p = 0; p < pixels; ++p)
            {
                U4 mean{};
                for (std::size_t c = 0; c < 4; ++c)
                {
                    std::uint32_t sum = 0;
                    for (const auto& img : per_sample) sum += img[4 * p + c];
                    mean[c] = (sum + static_cast<std::uint32_t>(samples) / 2) / static_cast<std::uint32_t>(samples);
                }
                mean_rejected += is_a_sample(p, mean) ? 0 : 1;
            }
            // A resolve of a sub-rectangle (identical source and destination bounds). Integer samples make the inside normative too:
            // every pixel whose centre lies within the rectangle is one of its own samples, every other pixel keeps the sentinel.
            {
                const Rect part = {20, 30, 100, 90};
                const std::array<GLuint, 4> sentinel = {1, 2, 3, 4};
                const U4 sentinel32 = {sentinel[0], sentinel[1], sentinel[2], sentinel[3]};
                bind_both(resolved);
                reset_fragment_state();
                mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, sentinel.data());
                bind_pair(ms, resolved);
                (void)demo::drain_gl_errors();
                blit(part, part, kColorBit, kNearest);
                check.expect_no_gl_error("integer resolve of the sub-rectangle (20,30)-(100,90) with identical bounds raises no error");
                const std::vector<std::uint32_t> sub = read_uint_image(resolved, kDst);
                int in_bad = 0, out_written = 0, in_count = 0;
                std::string first_in, first_out;
                for (std::size_t p = 0; p < pixels; ++p)
                {
                    const int x = static_cast<int>(p % kDst), y = static_cast<int>(p / kDst);
                    const U4 v = value_at(sub, p);
                    if (x >= part.x0 && x < part.x1 && y >= part.y0 && y < part.y1)
                    {
                        ++in_count;
                        if (!is_a_sample(p, v) && in_bad++ == 0) first_in = std::format("({},{}) = ({},{},{},{})", x, y, v[0], v[1], v[2], v[3]);
                    }
                    else if (v != sentinel32 && out_written++ == 0)
                        first_out = std::format("({},{}) = ({},{},{},{})", x, y, v[0], v[1], v[2], v[3]);
                }
                check.expect(in_bad == 0, "integer sub-rectangle resolve: every pixel inside the rectangle equals one of its own samples",
                             in_bad == 0 ? std::format("{} pixels", in_count) : std::format("{} of {} violate it, first {}", in_bad, in_count, first_in));
                check.expect_or_known_deviation(
                    out_written == 0, "integer sub-rectangle resolve: no pixel outside the rectangle is written (all keep the sentinel)",
                    out_written == 0 ? std::format("{} pixels outside", pixels - static_cast<std::size_t>(in_count))
                                     : std::format("{} pixels written, first {}", out_written, first_out),
                    "ANGLE", "ANGLE writes resolved pixels outside the destination rectangle of a multisample resolve blit (ES 3.0 4.3.3: only pixels whose centers lie within the destination rectangle are written)");
            }

            std::string picks;
            for (const int k : picked) picks += std::format(" {}", k);
            std::cout << std::format("   note: observed integer resolve ({} samples): {} pixels with mixed samples; a per-channel mean would be rejected in {} "
                                     "of them; in the mixed pixels the resolved value equals sample 0..{} this often:{}\n",
                                     samples, mixed, mean_rejected, samples - 1, picks);
            bind_both(kDefaultFbo);
        }

        static std::vector<std::uint32_t> read_uint_image(mgl::FramebufferId f, int size)
        {
            bind_read(f);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            std::vector<std::uint32_t> px(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
            mgl::glReadPixels(0, 0, size, size, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, px.data());
            return px;
        }

        // ------------------------------------------------------------------------------------------
        // The errors of ES 3.0 4.3.3 that are not about filters or depth/stencil.
        void check_blit_errors(demo::Check& check)
        {
            const Rect r = {0, 0, kDst, kDst};
            const Image zero(kDst, kDst, {0, 0, 0, 255});

            // Multisample source: formats and bounds must be identical, the destination must not be multisampled.
            clear_target(res_);
            bind_pair(ms_, res_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "multisample resolve with scaling (destination 2x larger) -> INVALID_OPERATION",
                         [&] { blit({0, 0, 64, 64}, {0, 0, 128, 128}, kColorBit, kNearest); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "multisample resolve of a 64 x 64 rectangle placed elsewhere (same size, other position) -> INVALID_OPERATION",
                         [&] { blit({0, 0, 64, 64}, {32, 32, 96, 96}, kColorBit, kNearest); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "multisample resolve with mirroring (swapped destination coordinates) -> INVALID_OPERATION",
                         [&] { blit(r, {kDst, 0, 0, kDst}, kColorBit, kNearest); });
            expect_image(check, "the rejected resolves copied nothing", read_attachment(res_, mgl::ColorAttachment::Color0, kDst, kDst), zero, 0);

            // Format mismatch: RGBA8 multisample -> RGB565 renderbuffer / SRGB8_ALPHA8.
            bind_pair(ms_, conv_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "multisample RGBA8 -> RGB565: the formats are not identical -> INVALID_OPERATION",
                         [&] { blit(r, r, kColorBit, kNearest); });
            bind_pair(ms_, srgb_target());
            (void)demo::drain_gl_errors();
            blit(r, r, kColorBit, kNearest);
            const mgl::ErrorCode srgb_error = demo::drain_gl_errors();
            // ES 3.0 4.3.3: "If SAMPLE_BUFFERS for the read framebuffer is one, no copy is performed and an INVALID_OPERATION
            // error is generated if the formats of the read and draw framebuffers are not identical". RGBA8 and SRGB8_ALPHA8
            // are different formats. Mesa's radeonsi, llvmpipe and zink accept the resolve without an error.
            expect_known_mesa_deviation(check, srgb_error == mgl::ErrorCode::InvalidOperation,
                                        "multisample RGBA8 -> SRGB8_ALPHA8: the formats are not identical -> INVALID_OPERATION",
                                        std::format("glGetError() = {}, expected GL_INVALID_OPERATION", demo::error_name(srgb_error)),
                                        "ES 3.0 4.3.3 requires identical formats for a multisample read framebuffer; Mesa resolves RGBA8 into SRGB8_ALPHA8 without an error");

            // A multisampled draw framebuffer is always an error.
            bind_pair(ds_ms_, ds_ms_other_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "multisample -> multisample: the draw framebuffer is multisampled -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit, kNearest); });
            bind_pair(ds_src_, ds_ms_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "single-sample -> multisample: the draw framebuffer is multisampled -> INVALID_OPERATION",
                         [&] { blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit, kNearest); });

            // Identical buffers.
            bind_pair(res_, res_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "source and destination are the same framebuffer -> INVALID_OPERATION",
                         [&] { blit({0, 0, 64, 64}, {64, 64, 128, 128}, kColorBit, kNearest); });

            // Incomplete framebuffers.
            bind_pair(src_, empty_);
            expect_error(check, mgl::ErrorCode::InvalidFramebufferOperation, "incomplete DRAW framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit, kNearest); });
            bind_pair(empty_, dst_);
            expect_error(check, mgl::ErrorCode::InvalidFramebufferOperation, "incomplete READ framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { blit({0, 0, kSrc, kSrc}, {0, 0, kSrc, kSrc}, kColorBit, kNearest); });
            bind_both(kDefaultFbo);
        }

        /// Clears the color attachment 0 of `f` to opaque black.
        static void clear_target(mgl::FramebufferId f)
        {
            bind_both(f);
            reset_fragment_state();
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(kColorBit);
        }

        /// An SRGB8_ALPHA8 framebuffer, created on first use (a different format from RGBA8).
        mgl::FramebufferId srgb_target()
        {
            if (srgb_fbo_.value == 0)
            {
                srgb_fbo_ = make_fbo();
                attach_color_renderbuffer(make_renderbuffer(mgl::InternalFormat::Srgb8Alpha8, 0, kDst, kDst));
                require_complete("SRGB8_ALPHA8");
            }
            return srgb_fbo_;
        }

        // ------------------------------------------------------------------------------------------
        mgl::ProgramId quad_{}, chart_{}, tri_{};
        mgl::UniformLocation quad_rect_{}, quad_z_{}, quad_color_{}, chart_rect_{}, chart_z_{}, tri_tri_{}, tri_size_{}, tri_color_{};
        int width_ = 0, height_ = 0;  // window size: the 2 x 2 quarters partition it
        std::vector<GLint> rgba8_counts_;
        int ds_samples_ = 0;

        mgl::FramebufferId src_{}, dst_{}, ds_src_{}, ds_dst_{}, ds_dst16_{}, ds_res_{}, ds_ms_{}, ds_ms_other_{}, ms_{}, res_{}, ss_{}, conv_{}, ui_src_{}, ui_dst_{},
            si_{}, empty_{}, srgb_fbo_{};
        std::array<mgl::TextureId, 3> src_tex_{}, dst_tex_{};
        mgl::TextureId res_tex_{}, ss_tex_{};
        mgl::RenderbufferId ms_rb_{}, conv_rb_{};
        std::vector<mgl::FramebufferId> framebuffers_;
        std::vector<mgl::RenderbufferId> renderbuffers_;
        std::vector<mgl::TextureId> textures_;

        std::array<Image, 3> chart_img_;
        Image ds_src_img_, res_img_, ss_img_;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es30-framebuffer-blit-msaa",
        .title = "Framebuffer blits and multisample resolve",
        .min_es = {3, 0},
        .summary = "glBlitFramebuffer between separate READ and DRAW framebuffers: color (NEAREST/LINEAR, scaling, mirroring, clipping), depth and stencil, "
                   "multisample renderbuffer resolve, the errors ES 3.0 defines, and glGetInternalformativ sample counts.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<BlitMsaa>(argc, argv, info);
}
