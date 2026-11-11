// es30-depth-textures
//
// The five depth formats of OpenGL ES 3.0 (Table 3.14) as textures, systematically:
//
//   * DEPTH_COMPONENT16 / 24 / 32F, DEPTH24_STENCIL8, DEPTH32F_STENCIL8 uploaded with the matching
//     pixel types of Table 3.2 (UNSIGNED_SHORT, UNSIGNED_INT, FLOAT, UNSIGNED_INT_24_8,
//     FLOAT_32_UNSIGNED_INT_24_8_REV) and read back through a sampler with compare mode NONE;
//     invalid format/type/internalformat combinations raise INVALID_OPERATION;
//   * the same formats as render targets: a 3D scene rendered into each, gl_FragDepth values written
//     into 16x1 strips, and the stored values compared against the conversion rules of §2.1.6 and
//     §3.9.2.3 (fixed point: one of the two nearest b-bit values; floating point: exact);
//   * COMPARE_REF_TO_TEXTURE through sampler2DShadow, sampler2DArrayShadow and samplerCubeShadow for
//     all eight compare functions (Table 3.23), including the clamping of Dref for fixed-point
//     formats only (§3.8.15.1), and LINEAR filtering of comparisons;
//   * texture completeness of depth textures (§3.8.13: compare mode NONE requires NEAREST filters),
//     what incomplete textures return (§3.9.2.1: (0,0,0,1), or 0 for shadow samplers), and
//     GenerateMipmap on depth formats (INVALID_OPERATION, §3.8.10.5);
//   * the stencil half of the depth-stencil formats, which ES 3.0 cannot sample: it is uploaded,
//     cleared, written and tested through a DEPTH_STENCIL_ATTACHMENT and read back with the stencil
//     test itself (one glColorMask'ed pass per stencil bit).
//
// The window shows the scene depth texture of every format as a relief-shaded depth map (top row;
// DEPTH_COMPONENT16 shows stair-stepped normals on the far boxes) and a grid of the eight compare
// functions (columns) for the three shadow sampler types (rows). See README.md.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    using SF = mgl::SizedInternalFormat;
    using IF = mgl::InternalFormat;
    using PF = mgl::PixelFormat;
    using PT = mgl::PixelType;
    using TT = mgl::TextureTarget;
    using FT = mgl::FramebufferTarget;
    using FA = mgl::FramebufferAttachment;
    using FAP = mgl::FramebufferAttachmentParameter;
    using CF = mgl::CompareFunc;
    using TPS = mgl::TextureParameterSetter;

    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    // ------------------------------------------------------------------------------------------
    // Formats and compare functions
    // ------------------------------------------------------------------------------------------

    struct DepthFormat
    {
        const char* name;
        SF sized;
        IF internal;
        PF format;      // matching external format (Table 3.2)
        PT type;        // matching external type (Table 3.2)
        int min_bits;   // Table 3.14 minimum depth resolution (32 = f32)
        bool is_float;
        bool has_stencil;
    };

    constexpr std::array<DepthFormat, 5> kFormats = {{
        {"DEPTH_COMPONENT16", SF::DepthComponent16, IF::DepthComponent16, PF::DepthComponent, PT::UnsignedShort, 16, false, false},
        {"DEPTH_COMPONENT24", SF::DepthComponent24, IF::DepthComponent24, PF::DepthComponent, PT::UnsignedInt, 24, false, false},
        {"DEPTH_COMPONENT32F", SF::DepthComponent32F, IF::DepthComponent32F, PF::DepthComponent, PT::Float, 32, true, false},
        {"DEPTH24_STENCIL8", SF::Depth24Stencil8, IF::Depth24Stencil8, PF::DepthStencil, PT::UnsignedInt248, 24, false, true},
        {"DEPTH32F_STENCIL8", SF::Depth32FStencil8, IF::Depth32FStencil8, PF::DepthStencil, PT::Float32UnsignedInt248Rev, 32, true, true},
    }};
    constexpr int kF16 = 0, kF24 = 1, kF32F = 2, kF24S8 = 3, kF32FS8 = 4;

    struct FuncInfo
    {
        CF func;
        const char* name;
        std::array<float, 3> color;
    };

    // GL enum order NEVER (0x200) .. ALWAYS (0x207): the columns of the window grid.
    constexpr std::array<FuncInfo, 8> kFuncs = {{
        {CF::Never, "NEVER", {0.66f, 0.68f, 0.82f}},
        {CF::Less, "LESS", {1.00f, 0.47f, 0.36f}},
        {CF::Equal, "EQUAL", {0.55f, 0.95f, 0.45f}},
        {CF::Lequal, "LEQUAL", {1.00f, 0.76f, 0.28f}},
        {CF::Greater, "GREATER", {0.42f, 0.58f, 1.00f}},
        {CF::Notequal, "NOTEQUAL", {1.00f, 0.50f, 0.85f}},
        {CF::Gequal, "GEQUAL", {0.35f, 0.86f, 1.00f}},
        {CF::Always, "ALWAYS", {0.95f, 0.95f, 0.92f}},
    }};
    constexpr int kNever = 0, kLess = 1, kEqual = 2, kLequal = 3, kGreater = 4, kNotequal = 5, kGequal = 6, kAlways = 7;

    // ES 3.0 Table 3.23 (after the fixed-point clamping of §3.8.15.1, done by the caller).
    bool table_3_23(CF f, double ref, double dt)
    {
        switch (f)
        {
        case CF::Lequal: return ref <= dt;
        case CF::Gequal: return ref >= dt;
        case CF::Less: return ref < dt;
        case CF::Greater: return ref > dt;
        case CF::Equal: return ref == dt;
        case CF::Notequal: return ref != dt;
        case CF::Always: return true;
        case CF::Never: return false;
        }
        return false;
    }

    float bits_to_float(std::uint32_t u) { return std::bit_cast<float>(u); }
    std::uint32_t float_to_bits(float f) { return std::bit_cast<std::uint32_t>(f); }

    // Float32 tolerance for converted fixed-point values: one float32 ulp in [0.5, 1) is 2^-24; the
    // conversion into a float (eq. 2.1) and the value returned to the shader may each round once.
    constexpr double kFloatSlack = 1.0 / 8388608.0;  // 2^-23

    /// Interval of values a b-bit fixed-point depth may return for the real value f (§2.1.6.2:
    /// convert_float_uint returns one of the two closest b-bit integers; §2.1.6.1: c / (2^b - 1)).
    struct FixedInterval
    {
        double lo, hi;
        [[nodiscard]] bool contains(double v) const { return v >= lo && v <= hi; }
    };

    FixedInterval fixed_interval(double f, int bits)
    {
        const double scale = std::ldexp(1.0, bits) - 1.0;
        const double r = std::clamp(f, 0.0, 1.0) * scale;
        return {std::floor(r) / scale - kFloatSlack, std::ceil(r) / scale + kFloatSlack};
    }

    // ------------------------------------------------------------------------------------------
    // The scene: a ground plane and six boxes, all planar so the CPU depth is exact.
    // ------------------------------------------------------------------------------------------

    struct V3
    {
        double x = 0, y = 0, z = 0;
        friend V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
        friend V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
        friend V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    };
    double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    V3 normalize(V3 v) { return v * (1.0 / std::sqrt(dot(v, v))); }

    struct Box
    {
        V3 center, half;
        double yaw_deg;
        int id;  // stencil value written by the depth-stencil formats
    };

    constexpr int kGroundId = 1;
    constexpr double kGroundX0 = -30.0, kGroundX1 = 30.0, kGroundZ0 = -45.0, kGroundZ1 = 14.0;
    const std::array<Box, 6> kBoxes = {{
        {{-2.4, 1.0, 0.0}, {1.0, 1.0, 1.0}, 30.0, 2},
        {{2.2, 0.7, -0.8}, {1.3, 0.7, 0.8}, -20.0, 3},
        {{0.3, 1.9, -4.5}, {0.45, 1.9, 2.2}, 0.0, 4},
        {{0.9, 2.5, 1.4}, {0.5, 0.5, 0.5}, 45.0, 5},
        {{-6.0, 2.0, -16.0}, {2.0, 2.0, 2.0}, 15.0, 6},
        {{7.0, 3.0, -26.0}, {2.5, 3.0, 2.5}, -30.0, 7},
    }};

    // Rotation about +Y by `deg` degrees (right-handed), and its inverse.
    V3 rot_y(V3 p, double deg)
    {
        const double a = deg * std::numbers::pi / 180.0, c = std::cos(a), s = std::sin(a);
        return {c * p.x + s * p.z, p.y, -s * p.x + c * p.z};
    }

    struct Hit
    {
        double t = std::numeric_limits<double>::infinity();
        int id = 0;  // 0 = nothing (sky)
    };

    Hit trace(V3 o, V3 d)
    {
        Hit h;
        if (d.y < 0.0)
        {
            const double t = -o.y / d.y;
            const V3 p = o + d * t;
            if (t > 0.0 && p.x >= kGroundX0 && p.x <= kGroundX1 && p.z >= kGroundZ0 && p.z <= kGroundZ1)
                h = {t, kGroundId};
        }
        for (const Box& b : kBoxes)
        {
            const V3 lo = rot_y(o - b.center, -b.yaw_deg);
            const V3 ld = rot_y(d, -b.yaw_deg);
            const double oo[3] = {lo.x, lo.y, lo.z}, dd[3] = {ld.x, ld.y, ld.z}, hh[3] = {b.half.x, b.half.y, b.half.z};
            double tmin = -std::numeric_limits<double>::infinity(), tmax = std::numeric_limits<double>::infinity();
            bool miss = false;
            for (int k = 0; k < 3 && !miss; ++k)
            {
                if (std::abs(dd[k]) < 1e-12)
                {
                    miss = std::abs(oo[k]) > hh[k];
                    continue;
                }
                const double t1 = (-hh[k] - oo[k]) / dd[k], t2 = (hh[k] - oo[k]) / dd[k];
                tmin = std::max(tmin, std::min(t1, t2));
                tmax = std::min(tmax, std::max(t1, t2));
                miss = tmin > tmax;
            }
            if (!miss && tmin > 1e-9 && tmin < h.t)
                h = {tmin, b.id};
        }
        return h;
    }

    struct Camera
    {
        V3 eye, fwd, right, up;
        double fovy_deg, aspect, n, f;
        V3 target, up_hint;

        [[nodiscard]] double tan_half() const { return std::tan(fovy_deg * std::numbers::pi / 360.0); }
        /// Ray direction through the center of texel (i, j) of a w x h target (dot(dir, fwd) == 1).
        [[nodiscard]] V3 texel_dir(int i, int j, int w, int h) const
        {
            const double nx = (i + 0.5) / w * 2.0 - 1.0, ny = (j + 0.5) / h * 2.0 - 1.0;
            return fwd + right * (nx * tan_half() * aspect) + up * (ny * tan_half());
        }
        /// Window depth (DepthRangef(0, 1)) of a point at eye distance z along the view axis.
        [[nodiscard]] double window_depth(double z) const { return 0.5 + 0.5 * ((f + n) / (f - n) - 2.0 * f * n / ((f - n) * z)); }

        [[nodiscard]] dm::Mat4 view_proj() const
        {
            auto v3 = [](V3 v) { return dm::Vec3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; };
            return dm::perspective(static_cast<float>(fovy_deg * std::numbers::pi / 180.0), static_cast<float>(aspect),
                                   static_cast<float>(n), static_cast<float>(f)) *
                   dm::look_at(v3(eye), v3(target), v3(up_hint));
        }
    };

    Camera make_camera(V3 eye, V3 target, V3 up_hint, double fovy_deg, double aspect, double n, double f)
    {
        Camera c;
        c.eye = eye;
        c.target = target;
        c.up_hint = up_hint;
        c.fwd = normalize(target - eye);
        c.right = normalize(cross(c.fwd, up_hint));
        c.up = cross(c.right, c.fwd);
        c.fovy_deg = fovy_deg;
        c.aspect = aspect;
        c.n = n;
        c.f = f;
        return c;
    }

    constexpr int kSceneW = 256, kSceneH = 160;  // scene depth textures (all five formats and the array)
    constexpr int kCubeSize = 128;               // cube depth texture faces
    constexpr double kNearA = 0.25, kFarA = 80.0;
    constexpr double kCubeNear = 0.1, kCubeFar = 80.0;
    const V3 kCubeCenter{0.0, 1.2, 3.0};

    Camera camera_a() { return make_camera({0.0, 2.6, 7.5}, {0.0, 1.2, -2.0}, {0, 1, 0}, 52.0, double(kSceneW) / kSceneH, kNearA, kFarA); }
    Camera camera_b() { return make_camera({-8.0, 5.5, -2.0}, {0.5, 0.6, -1.5}, {0, 1, 0}, 48.0, double(kSceneW) / kSceneH, kNearA, kFarA); }

    // Cube face cameras in GL face order +X, -X, +Y, -Y, +Z, -Z. The up vectors reproduce the
    // (sc, tc) conventions of ES 3.0 Table 3.21, so texel (i, j) of a rendered face is the texel the
    // sampler selects for the direction through its center.
    Camera cube_camera(int face)
    {
        static const V3 axes[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        static const V3 ups[6] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
        return make_camera(kCubeCenter, kCubeCenter + axes[face], ups[face], 90.0, 1.0, kCubeNear, kCubeFar);
    }
    constexpr std::array<TT, 6> kCubeFaces = {TT::TextureCubeMapPositiveX, TT::TextureCubeMapNegativeX, TT::TextureCubeMapPositiveY,
                                              TT::TextureCubeMapNegativeY, TT::TextureCubeMapPositiveZ, TT::TextureCubeMapNegativeZ};

    struct SceneMesh
    {
        std::vector<float> positions;
        struct Range { int first, count, id; };
        std::vector<Range> ranges;
    };

    SceneMesh make_scene_mesh()
    {
        SceneMesh m;
        auto push = [&](V3 p) {
            m.positions.push_back(static_cast<float>(p.x));
            m.positions.push_back(static_cast<float>(p.y));
            m.positions.push_back(static_cast<float>(p.z));
        };
        const V3 g[4] = {{kGroundX0, 0, kGroundZ0}, {kGroundX1, 0, kGroundZ0}, {kGroundX1, 0, kGroundZ1}, {kGroundX0, 0, kGroundZ1}};
        for (int k : {0, 1, 2, 0, 2, 3}) push(g[k]);
        m.ranges.push_back({0, 6, kGroundId});
        static const int quads[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
        for (const Box& b : kBoxes)
        {
            V3 c[8];
            for (int k = 0; k < 8; ++k)
            {
                const V3 local{(k & 1) ? b.half.x : -b.half.x, (k & 2) ? b.half.y : -b.half.y, (k & 4) ? b.half.z : -b.half.z};
                c[k] = b.center + rot_y(local, b.yaw_deg);
            }
            const int first = static_cast<int>(m.positions.size() / 3);
            for (const auto& q : quads)
                for (int k : {0, 1, 2, 0, 2, 3}) push(c[q[k]]);
            m.ranges.push_back({first, 36, b.id});
        }
        return m;
    }

    /// A texel whose 3x3 (or wider) neighbourhood sees the same object, with its CPU depth.
    struct SceneProbe
    {
        int i, j;
        int id;
        double depth;  // window depth, 1.0 for sky
    };

    std::vector<SceneProbe> pick_scene_probes(const Camera& cam, int w, int h, int max_count, int step)
    {
        std::vector<SceneProbe> all;
        for (int j = 3; j < h - 3; j += step)
            for (int i = 3; i < w - 3; i += step)
            {
                const Hit c = trace(cam.eye, cam.texel_dir(i, j, w, h));
                bool uniform = true;
                for (int dj = -1; dj <= 1 && uniform; ++dj)
                    for (int di = -1; di <= 1 && uniform; ++di)
                        uniform = trace(cam.eye, cam.texel_dir(i + di, j + dj, w, h)).id == c.id;
                if (!uniform) continue;
                // t is measured along a direction with dot(dir, fwd) == 1, so it is the eye-space distance.
                all.push_back({i, j, c.id, c.id == 0 ? 1.0 : cam.window_depth(c.t)});
            }
        std::vector<SceneProbe> out;
        // Every object first (one probe each), then a uniform stride through the rest.
        for (int id = 0; id <= 7; ++id)
            for (const SceneProbe& p : all)
                if (p.id == id)
                {
                    out.push_back(p);
                    break;
                }
        const std::size_t stride = std::max<std::size_t>(1, all.size() / static_cast<std::size_t>(max_count));
        for (std::size_t k = stride / 2; k < all.size() && static_cast<int>(out.size()) < max_count; k += stride)
            if (std::none_of(out.begin(), out.end(), [&](const SceneProbe& p) { return p.i == all[k].i && p.j == all[k].j; }))
                out.push_back(all[k]);
        return out;
    }

    // ------------------------------------------------------------------------------------------
    // Window layout: five panels on top, an 8 x 3 grid below; equal integer cell sizes so that
    // the same local pixel of every cell in a row samples the same texel with the same Dref.
    // ------------------------------------------------------------------------------------------

    struct Rect
    {
        int x = 0, y = 0, w = 0, h = 0;
    };

    /// The panel shows the scene texture "cover"-fitted: uv = offset + local * scale, cropping the
    /// texture symmetrically so it keeps its aspect ratio and still fills the panel.
    struct PanelUv
    {
        double sx = 1, sy = 1, ox = 0, oy = 0;
    };

    struct Layout
    {
        int margin = 0;
        std::array<Rect, 5> panel{};
        std::array<PanelUv, 5> uv{};
        int cell_w = 0, cell_h = 0;
        std::array<int, 8> cell_x{};
        std::array<int, 3> cell_y{};  // row 0 = sampler2DShadow (top), 1 = array, 2 = cube (bottom)
    };

    constexpr float kBand = 0.25f;  // height fraction of the exact band in every grid cell
    constexpr int kMinWindow = 128;
    constexpr double kSceneAspect = 256.0 / 160.0;

    Layout make_layout(int w, int h)
    {
        Layout l;
        l.margin = std::max(2, std::min(w, h) / 90);
        const int m = l.margin;
        const int top_h = static_cast<int>((h - 5 * m) * 0.46);
        l.cell_h = (h - 5 * m - top_h) / 3;
        const int y0 = m + (h - 5 * m - top_h - 3 * l.cell_h) / 2;
        l.cell_y = {y0 + 2 * (l.cell_h + m), y0 + l.cell_h + m, y0};
        const int top_y = y0 + 3 * (l.cell_h + m);

        // Panels: 1 row of 5, rows of 3 + 2, or 2 + 2 + 1, whichever keeps the panels closest to the
        // texture's aspect ratio.
        const std::array<std::vector<int>, 3> arrangements = {{{5}, {3, 2}, {2, 2, 1}}};
        double best = 1e30;
        for (const std::vector<int>& rows : arrangements)
        {
            const int nr = static_cast<int>(rows.size());
            const int ph = (top_h - (nr - 1) * m) / nr;
            if (ph < 8) continue;
            std::array<Rect, 5> rects{};
            double score = 0.0;
            int idx = 0;
            for (int r = 0; r < nr; ++r)
            {
                const int n = rows[static_cast<std::size_t>(r)];
                const int pw = (w - (n + 1) * m) / n;
                const int x0 = m + (w - (n + 1) * m - n * pw) / 2;
                for (int c = 0; c < n; ++c, ++idx)
                {
                    rects[static_cast<std::size_t>(idx)] = {x0 + c * (pw + m), top_y + (nr - 1 - r) * (ph + m), pw, ph};
                    score += std::abs(std::log(double(pw) / ph / kSceneAspect));
                }
            }
            if (score < best)
            {
                best = score;
                l.panel = rects;
            }
        }
        for (std::size_t k = 0; k < 5; ++k)
        {
            const double a = double(l.panel[k].w) / l.panel[k].h;
            PanelUv& uv = l.uv[k];
            if (a > kSceneAspect)
                uv.sy = kSceneAspect / a;  // wider than the texture: crop top and bottom
            else
                uv.sx = a / kSceneAspect;  // narrower: crop left and right
            uv.ox = (1.0 - uv.sx) / 2.0;
            uv.oy = (1.0 - uv.sy) / 2.0;
        }
        l.cell_w = (w - 9 * m) / 8;
        const int cx0 = m + (w - 9 * m - 8 * l.cell_w) / 2;
        for (int k = 0; k < 8; ++k) l.cell_x[static_cast<std::size_t>(k)] = cx0 + k * (l.cell_w + m);
        return l;
    }

    constexpr std::array<float, 3> kSky = {0.16f, 0.22f, 0.32f};
    constexpr std::array<float, 3> kBackground = {0.06f, 0.065f, 0.08f};
    constexpr std::array<std::array<float, 3>, 8> kIdTint = {{
        {0, 0, 0}, {0, 0, 0}, {0.2f, 1.0f, 0.6f}, {0.3f, 0.6f, 1.0f}, {1.0f, 0.9f, 0.2f}, {1.0f, 0.3f, 0.3f}, {0.8f, 0.4f, 1.0f}, {0.2f, 1.0f, 1.0f},
    }};

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

    // ------------------------------------------------------------------------------------------
    // Uploaded 8x2 textures (16 texels each) for the Table 3.2 combinations.
    // ------------------------------------------------------------------------------------------

    constexpr int kUpW = 8, kUpH = 2, kUpN = kUpW * kUpH;

    constexpr std::array<std::uint16_t, kUpN> kU16 = {0, 65535, 32768, 1, 12345, 54321, 40000, 32767, 100, 65534, 30000, 7, 60000, 21845, 43690, 1000};
    constexpr std::array<std::uint32_t, kUpN> kU32For16 = {0u, 0xFFFFFFFFu, 0x80000000u, 0x40000000u, 0x12345678u, 0xDEADBEEFu, 0x00010000u, 0x0000FFFFu,
                                                           0x7FFFFFFFu, 0xC0000000u, 0x00008000u, 0xFFFF0000u, 0x30398000u, 0x00018000u, 0xA0000000u, 0x55555555u};
    constexpr std::array<std::uint32_t, kUpN> kU32For24 = {0u, 0xFFFFFFFFu, 0x80000000u, 0x00000100u, 0x12345678u, 0xDEADBEEFu, 0x00000080u, 0xFFFFFF00u,
                                                           0x7FFFFFFFu, 0xC0000000u, 0x00ABCDEFu, 0x55555555u, 0x33333333u, 0x00000001u, 0xFEDCBA98u, 0x01000000u};
    constexpr std::array<float, kUpN> kF32 = {0.0f, 1.0f, 0.5f, -0.25f, 1.75f, 0.1f, 1.0f / 3.0f, 0.987654321f,
                                              1e-6f, 0.999999f, 0.25f, 0.75f, 0x1p-20f, 0.3f, 0.6180339887f, 0.0001f};
    constexpr std::array<std::uint32_t, kUpN> kD24 = {0x000000u, 0xFFFFFFu, 0x800000u, 0x000001u, 0x123456u, 0xABCDEFu, 0x7FFFFFu, 0xFFFFFEu,
                                                      0x400000u, 0xC00000u, 0x0F0F0Fu, 0x333333u, 0xDDDDDDu, 0x000100u, 0x5A5A5Au, 0xA5A5A5u};
    constexpr std::array<float, kUpN> kF32S = {1.0f, 0.0f, -0.5f, 2.0f, 0.125f, 0.2f, 0.4f, 0.6f, 0.8f, 0.9f, 0.95f, 0.05f, 0.33f, 0.66f, 0.999f, 0.001f};

    int stencil_24_8(int t) { return (t * 37 + 11) & 0xFF; }
    int stencil_32f_8(int t) { return (t * 53 + 200) & 0xFF; }

    enum Upload { kUp16Short, kUp16Int, kUp24Int, kUp32Float, kUp24S8, kUp32FS8, kUploadCount };
    struct UploadInfo
    {
        const char* label;
        int format;  // index into kFormats
        PT type;
    };
    constexpr std::array<UploadInfo, kUploadCount> kUploads = {{
        {"DEPTH_COMPONENT16 / UNSIGNED_SHORT", kF16, PT::UnsignedShort},
        {"DEPTH_COMPONENT16 / UNSIGNED_INT", kF16, PT::UnsignedInt},
        {"DEPTH_COMPONENT24 / UNSIGNED_INT", kF24, PT::UnsignedInt},
        {"DEPTH_COMPONENT32F / FLOAT", kF32F, PT::Float},
        {"DEPTH24_STENCIL8 / UNSIGNED_INT_24_8", kF24S8, PT::UnsignedInt248},
        {"DEPTH32F_STENCIL8 / FLOAT_32_UNSIGNED_INT_24_8_REV", kF32FS8, PT::Float32UnsignedInt248Rev},
    }};

    /// The real value the spec defines for texel t of upload u before it is stored: the unpacked
    /// component converted by eq. 2.1 (normalized integers) or taken as is (float), clamped to [0, 1]
    /// (§3.8.3: "For depth component groups, the depth value is clamped to [0, 1]").
    double upload_value(int u, int t)
    {
        const auto ut = static_cast<std::size_t>(t);
        switch (u)
        {
        case kUp16Short: return kU16[ut] / 65535.0;
        case kUp16Int: return kU32For16[ut] / 4294967295.0;
        case kUp24Int: return kU32For24[ut] / 4294967295.0;
        case kUp32Float: return std::clamp(static_cast<double>(kF32[ut]), 0.0, 1.0);
        case kUp24S8: return kD24[ut] / 16777215.0;
        case kUp32FS8: return std::clamp(static_cast<double>(kF32S[ut]), 0.0, 1.0);
        default: return 0.0;
        }
    }

    // gl_FragDepth values written into the 16x1 strips (index 15 keeps the clear value).
    constexpr float kClearDepth = 0.8125f;
    constexpr int kClearStencil = 0xA5;
    const std::array<float, 15> kFragDepths = {
        0.0f, 1.0f, 0.5f, 1.0f / 3.0f, 0.1f, 0.9f,
        static_cast<float>(1000.3 / 65535.0), static_cast<float>(20000.3 / 65535.0), static_cast<float>(40000.7 / 65535.0),
        static_cast<float>(65000.3 / 65535.0), static_cast<float>(50000.7 / 65535.0),
        static_cast<float>(5000000.4 / 16777215.0), 0.987654321f, 1.5f, -0.25f,
    };
    // Texels that the DEPTH_COMPONENT16 negative control expects to be visibly quantized.
    constexpr std::array<int, 5> kOff16Grid = {6, 7, 8, 9, 10};

    struct Lookup
    {
        float x, y, z, ref;
    };
    using Probe = std::array<std::uint32_t, 4>;

    constexpr std::array<mgl::TextureUnit, 6> kUnits = {mgl::TextureUnit::Texture0, mgl::TextureUnit::Texture1, mgl::TextureUnit::Texture2,
                                                       mgl::TextureUnit::Texture3, mgl::TextureUnit::Texture4, mgl::TextureUnit::Texture5};
    constexpr int kUnitRaw2D = 0, kUnitShadow2D = 1, kUnitShadowArray = 2, kUnitShadowCube = 3, kUnitRawArray = 4, kUnitRawCube = 5;
    enum ProbeMode { kModeRaw2D = 0, kModeShadow2D = 1, kModeShadowArray = 2, kModeShadowCube = 3, kModeRawArray = 4, kModeRawCube = 5 };

    mgl::UniformLocation uloc(mgl::ProgramId program, const char* name)
    {
        const mgl::UniformLocation l = mgl::glGetUniformLocation(program, name);
        if (l.value < 0) throw demo::Failure(std::format("uniform {} not found", name));
        return l;
    }

    class DepthTextures final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override;
        void render(demo::Env& env, int frame) override;
        void validate(demo::Env& env, demo::Check& check) override;
        void shutdown(demo::Env& env) override;

    private:
        // -- helpers ---------------------------------------------------------------------------
        static void bind_unit(int unit, TT target, mgl::TextureId tex, mgl::SamplerId sampler)
        {
            mgl::glActiveTexture(kUnits[static_cast<std::size_t>(unit)]);
            mgl::glBindTexture(target, tex);
            mgl::glBindSampler(static_cast<GLuint>(unit), sampler);
        }
        static void attach_depth(FT target, int format, TT textarget, mgl::TextureId tex)
        {
            mgl::glFramebufferTexture2D(target, kFormats[static_cast<std::size_t>(format)].has_stencil ? FA::DepthStencil : FA::Depth, textarget, tex, 0);
        }
        static void set_filters(TT target, mgl::TextureMinFilter min, mgl::TextureMagFilter mag)
        {
            mgl::glTexParameter(target, min);
            mgl::glTexParameter(target, mag);
        }
        static int fb_param(FA attachment, FAP pname)
        {
            GLint v = -1;
            mgl::glGetFramebufferAttachmentParameteriv(FT::Framebuffer, attachment, pname, &v);
            return v;
        }

        void render_depth_targets();
        void render_window(const demo::Env& env);
        void draw_scene(const dm::Mat4& view_proj, bool stencil_ids);
        std::vector<Probe> probe(int mode, std::span<const Lookup> lookups);
        std::vector<int> read_stencil(mgl::TextureId ds_texture, int format, int w, int h);
        void fill_frag_depth_strip(int format, float clear_depth, int clear_stencil);

        void check_window(const demo::Env& env, demo::Check& check);
        void check_attachments(demo::Check& check);
        void check_uploads(demo::Check& check);
        void check_format_errors(demo::Check& check);
        void check_quantization(demo::Check& check);
        void check_scene(demo::Check& check);
        void check_compare_2d(demo::Check& check);
        void check_compare_array_cube(demo::Check& check);
        void check_linear(demo::Check& check);
        void check_completeness(demo::Check& check);
        void check_stencil(demo::Check& check);

        // -- objects ---------------------------------------------------------------------------
        mgl::ProgramId scene_prog_{}, frag_depth_prog_{}, probe_prog_{}, solid_prog_{}, vis_prog_{}, show_prog_{}, grid_prog_{};
        mgl::VertexArrayId scene_vao_{}, empty_vao_{};
        mgl::BufferId scene_vbo_{};
        SceneMesh mesh_;

        std::array<mgl::TextureId, 5> scene_tex_{};   // kSceneW x kSceneH, one per format (camera A)
        std::array<mgl::FramebufferId, 5> scene_fbo_{};
        mgl::TextureId array_tex_{};                  // DEPTH32F 2 layers: camera A, camera B
        std::array<mgl::FramebufferId, 2> array_fbo_{};
        mgl::TextureId cube_tex_{};                   // DEPTH32F cube around kCubeCenter
        std::array<mgl::FramebufferId, 6> cube_fbo_{};
        std::array<mgl::TextureId, 5> vis_tex_{};     // RGBA8 visualizations
        std::array<mgl::FramebufferId, 5> vis_fbo_{};
        std::array<mgl::FramebufferId, 5> tint_fbo_{};  // vis color + depth-stencil scene texture (DS formats only)

        std::array<mgl::TextureId, 5> q_tex_{};       // 16x1 gl_FragDepth strips
        std::array<mgl::FramebufferId, 5> q_fbo_{};
        std::array<mgl::TextureId, kUploadCount> up_tex_{};
        mgl::TextureId probe_tex_{};
        mgl::FramebufferId probe_fbo_{};

        mgl::SamplerId raw_sampler_{}, linear_sampler_{};
        std::array<mgl::SamplerId, 8> cmp_sampler_{};

        // uniforms
        mgl::UniformLocation u_scene_vp_{}, u_fd_depth_{}, u_solid_color_{};
        mgl::UniformLocation u_probe_mode_{}, u_probe_coord_{}, u_probe_ref_{};
        mgl::UniformLocation u_vis_near_far_{}, u_vis_tan_{}, u_vis_size_{}, u_vis_sky_{};
        mgl::UniformLocation u_show_rect_{}, u_show_uv_{};
        mgl::UniformLocation u_grid_kind_{}, u_grid_cell_{}, u_grid_band_{}, u_grid_pass_{}, u_grid_near_far_{}, u_grid_slice_{};

        std::vector<SceneProbe> probes_a_, probes_b_;
        std::array<std::vector<SceneProbe>, 6> probes_cube_;
        std::array<int, 5> depth_bits_{};  // FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE per format
        int width_ = 0, height_ = 0;
    };

    // ==========================================================================================
    // init
    // ==========================================================================================

    void DepthTextures::init(demo::Env& env)
    {
        if (env.width < kMinWindow || env.height < kMinWindow)
            throw demo::Skip(std::format("the layout needs at least {}x{} pixels", kMinWindow, kMinWindow));

        scene_prog_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
        frag_depth_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/frag_depth.frag");
        probe_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/probe.frag");
        solid_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/solid.frag");
        vis_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/depth_vis.frag");
        show_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/show.frag");
        grid_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/compare_grid.frag");

        u_scene_vp_ = uloc(scene_prog_, "u_view_proj");
        u_fd_depth_ = uloc(frag_depth_prog_, "u_depth");
        u_solid_color_ = uloc(solid_prog_, "u_color");
        u_probe_mode_ = uloc(probe_prog_, "u_mode");
        u_probe_coord_ = uloc(probe_prog_, "u_coord");
        u_probe_ref_ = uloc(probe_prog_, "u_ref");
        u_vis_near_far_ = uloc(vis_prog_, "u_near_far");
        u_vis_tan_ = uloc(vis_prog_, "u_tan");
        u_vis_size_ = uloc(vis_prog_, "u_size");
        u_vis_sky_ = uloc(vis_prog_, "u_sky");
        u_show_rect_ = uloc(show_prog_, "u_rect");
        u_show_uv_ = uloc(show_prog_, "u_uv");
        u_grid_kind_ = uloc(grid_prog_, "u_kind");
        u_grid_cell_ = uloc(grid_prog_, "u_cell");
        u_grid_band_ = uloc(grid_prog_, "u_band");
        u_grid_pass_ = uloc(grid_prog_, "u_pass");
        u_grid_near_far_ = uloc(grid_prog_, "u_near_far");
        u_grid_slice_ = uloc(grid_prog_, "u_slice");

        // Sampler uniforms: every sampler type on its own unit (two types on one unit is a draw-time error).
        const char* sampler_names[6] = {"u_raw2d", "u_shadow2d", "u_shadow_array", "u_shadow_cube", "u_raw_array", "u_raw_cube"};
        for (mgl::ProgramId p : {probe_prog_, grid_prog_})
        {
            mgl::glUseProgram(p);
            for (int k = 0; k < 6; ++k) mgl::glUniform1i(uloc(p, sampler_names[k]), k);
        }
        mgl::glUseProgram(vis_prog_);
        mgl::glUniform1i(uloc(vis_prog_, "u_depth"), 0);
        mgl::glUseProgram(show_prog_);
        mgl::glUniform1i(uloc(show_prog_, "u_image"), 0);
        mgl::glUseProgram(mgl::ProgramId{0});

        // Scene geometry.
        mesh_ = make_scene_mesh();
        mgl::glGenVertexArrays(1, &scene_vao_);
        mgl::glGenVertexArrays(1, &empty_vao_);
        mgl::glGenBuffers(1, &scene_vbo_);
        mgl::glBindVertexArray(scene_vao_);
        mgl::glBindBuffer(mgl::BufferTarget::Array, scene_vbo_);
        mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(mesh_.positions), mgl::BufferUsage::StaticDraw);
        mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
        mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, 3 * sizeof(float), demo::buffer_offset(0));
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

        // Sampler objects: raw depth (compare NONE + NEAREST, the only complete filtering for a depth
        // texture without comparison, §3.8.13), LINEAR for the RGBA8 panels, and one per compare function.
        mgl::glGenSamplers(1, &raw_sampler_);
        mgl::glGenSamplers(1, &linear_sampler_);
        mgl::glGenSamplers(8, cmp_sampler_.data());
        auto clamp_sampler = [](mgl::SamplerId s) {
            mgl::glSamplerParameter(s, mgl::SamplerWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glSamplerParameter(s, mgl::SamplerWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            mgl::glSamplerParameter(s, mgl::SamplerWrapParameter::WrapR, mgl::TextureWrapMode::ClampToEdge);
        };
        for (mgl::SamplerId s : {raw_sampler_, linear_sampler_}) clamp_sampler(s);
        mgl::glSamplerParameter(raw_sampler_, mgl::TextureMinFilter::Nearest);
        mgl::glSamplerParameter(raw_sampler_, mgl::TextureMagFilter::Nearest);
        mgl::glSamplerParameter(raw_sampler_, mgl::TextureCompareMode::None);
        mgl::glSamplerParameter(linear_sampler_, mgl::TextureMinFilter::Linear);
        mgl::glSamplerParameter(linear_sampler_, mgl::TextureMagFilter::Linear);
        for (std::size_t f = 0; f < 8; ++f)
        {
            clamp_sampler(cmp_sampler_[f]);
            mgl::glSamplerParameter(cmp_sampler_[f], mgl::TextureMinFilter::Nearest);
            mgl::glSamplerParameter(cmp_sampler_[f], mgl::TextureMagFilter::Nearest);
            mgl::glSamplerParameter(cmp_sampler_[f], mgl::TextureCompareMode::CompareRefToTexture);
            mgl::glSamplerParameter(cmp_sampler_[f], kFuncs[f].func);
        }

        // Scene depth textures, one per format, immutable single level.
        mgl::glGenTextures(5, scene_tex_.data());
        mgl::glGenFramebuffers(5, scene_fbo_.data());
        mgl::glGenTextures(5, vis_tex_.data());
        mgl::glGenFramebuffers(5, vis_fbo_.data());
        mgl::glGenFramebuffers(5, tint_fbo_.data());
        for (int k = 0; k < 5; ++k)
        {
            const auto uk = static_cast<std::size_t>(k);
            mgl::glBindTexture(TT::Texture2D, scene_tex_[uk]);
            mgl::glTexStorage2D(TT::Texture2D, 1, kFormats[uk].sized, kSceneW, kSceneH);
            mgl::glBindFramebuffer(FT::Framebuffer, scene_fbo_[uk]);
            attach_depth(FT::Framebuffer, k, TT::Texture2D, scene_tex_[uk]);
            if (mgl::glCheckFramebufferStatus(FT::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{} scene framebuffer is not complete", kFormats[uk].name));
            depth_bits_[uk] = fb_param(FA::Depth, FAP::DepthSize);

            mgl::glBindTexture(TT::Texture2D, vis_tex_[uk]);
            mgl::glTexStorage2D(TT::Texture2D, 1, SF::Rgba8, kSceneW, kSceneH);
            mgl::glBindFramebuffer(FT::Framebuffer, vis_fbo_[uk]);
            mgl::glFramebufferTexture2D(FT::Framebuffer, kColor0, TT::Texture2D, vis_tex_[uk], 0);
            if (kFormats[uk].has_stencil)
            {
                // Same color texture plus the depth-stencil scene texture: the stencil ids tint the panel.
                // The depth-stencil texture is never sampled while this framebuffer is bound (§4.4.3).
                mgl::glBindFramebuffer(FT::Framebuffer, tint_fbo_[uk]);
                mgl::glFramebufferTexture2D(FT::Framebuffer, kColor0, TT::Texture2D, vis_tex_[uk], 0);
                attach_depth(FT::Framebuffer, k, TT::Texture2D, scene_tex_[uk]);
                if (mgl::glCheckFramebufferStatus(FT::Framebuffer) != mgl::FramebufferStatus::Complete)
                    throw demo::Failure(std::format("{} tint framebuffer is not complete", kFormats[uk].name));
            }
        }

        // 2D array (layer 0 = camera A, layer 1 = camera B) and cube map, both DEPTH_COMPONENT32F.
        mgl::glGenTextures(1, &array_tex_);
        mgl::glBindTexture(TT::Texture2DArray, array_tex_);
        mgl::glTexStorage3D(TT::Texture2DArray, 1, SF::DepthComponent32F, kSceneW, kSceneH, 2);
        mgl::glGenFramebuffers(2, array_fbo_.data());
        for (int layer = 0; layer < 2; ++layer)
        {
            mgl::glBindFramebuffer(FT::Framebuffer, array_fbo_[static_cast<std::size_t>(layer)]);
            mgl::glFramebufferTextureLayer(FT::Framebuffer, FA::Depth, array_tex_, 0, layer);
            if (mgl::glCheckFramebufferStatus(FT::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("array layer framebuffer is not complete");
        }
        mgl::glGenTextures(1, &cube_tex_);
        mgl::glBindTexture(TT::TextureCubeMap, cube_tex_);
        mgl::glTexStorage2D(TT::TextureCubeMap, 1, SF::DepthComponent32F, kCubeSize, kCubeSize);
        mgl::glGenFramebuffers(6, cube_fbo_.data());
        for (std::size_t face = 0; face < 6; ++face)
        {
            mgl::glBindFramebuffer(FT::Framebuffer, cube_fbo_[face]);
            mgl::glFramebufferTexture2D(FT::Framebuffer, FA::Depth, kCubeFaces[face], cube_tex_, 0);
            if (mgl::glCheckFramebufferStatus(FT::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("cube face framebuffer is not complete");
        }

        // 16x1 gl_FragDepth strips.
        mgl::glGenTextures(5, q_tex_.data());
        mgl::glGenFramebuffers(5, q_fbo_.data());
        for (std::size_t k = 0; k < 5; ++k)
        {
            mgl::glBindTexture(TT::Texture2D, q_tex_[k]);
            mgl::glTexStorage2D(TT::Texture2D, 1, kFormats[k].sized, 16, 1);
            mgl::glBindFramebuffer(FT::Framebuffer, q_fbo_[k]);
            attach_depth(FT::Framebuffer, static_cast<int>(k), TT::Texture2D, q_tex_[k]);
        }

        // Uploaded textures (mutable, glTexImage2D with the matching format/type of Table 3.2).
        mgl::glGenTextures(kUploadCount, up_tex_.data());
        mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
        for (int u = 0; u < kUploadCount; ++u)
        {
            const UploadInfo& info = kUploads[static_cast<std::size_t>(u)];
            const DepthFormat& fmt = kFormats[static_cast<std::size_t>(info.format)];
            mgl::glBindTexture(TT::Texture2D, up_tex_[static_cast<std::size_t>(u)]);
            std::vector<std::uint32_t> words;
            const void* data = nullptr;
            switch (u)
            {
            case kUp16Short: data = kU16.data(); break;
            case kUp16Int: data = kU32For16.data(); break;
            case kUp24Int: data = kU32For24.data(); break;
            case kUp32Float: data = kF32.data(); break;
            case kUp24S8:
                // UNSIGNED_INT_24_8 (Table 3.8): depth in bits 31..8, stencil in bits 7..0.
                for (int t = 0; t < kUpN; ++t) words.push_back((kD24[static_cast<std::size_t>(t)] << 8) | static_cast<std::uint32_t>(stencil_24_8(t)));
                data = words.data();
                break;
            case kUp32FS8:
                // FLOAT_32_UNSIGNED_INT_24_8_REV (Table 3.9): word 0 = float depth, word 1 = 24 unused bits + 8-bit stencil.
                for (int t = 0; t < kUpN; ++t)
                {
                    words.push_back(float_to_bits(kF32S[static_cast<std::size_t>(t)]));
                    words.push_back(0xABCDEF00u | static_cast<std::uint32_t>(stencil_32f_8(t)));
                }
                data = words.data();
                break;
            default: break;
            }
            mgl::glTexImage2D(TT::Texture2D, 0, fmt.internal, kUpW, kUpH, 0, fmt.format, info.type, data);
            set_filters(TT::Texture2D, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest);
        }

        // Probe target: 64x1 RGBA32UI.
        mgl::glGenTextures(1, &probe_tex_);
        mgl::glBindTexture(TT::Texture2D, probe_tex_);
        mgl::glTexStorage2D(TT::Texture2D, 1, SF::Rgba32UI, 64, 1);
        mgl::glGenFramebuffers(1, &probe_fbo_);
        mgl::glBindFramebuffer(FT::Framebuffer, probe_fbo_);
        mgl::glFramebufferTexture2D(FT::Framebuffer, kColor0, TT::Texture2D, probe_tex_, 0);
        if (mgl::glCheckFramebufferStatus(FT::Framebuffer) != mgl::FramebufferStatus::Complete)
            throw demo::Failure("probe framebuffer is not complete");

        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});

        // CPU probe sets (static scene).
        probes_a_ = pick_scene_probes(camera_a(), kSceneW, kSceneH, 60, 7);
        probes_b_ = pick_scene_probes(camera_b(), kSceneW, kSceneH, 60, 7);
        for (int face = 0; face < 6; ++face) probes_cube_[static_cast<std::size_t>(face)] = pick_scene_probes(cube_camera(face), kCubeSize, kCubeSize, 10, 9);
    }

    // ==========================================================================================
    // render
    // ==========================================================================================

    void DepthTextures::draw_scene(const dm::Mat4& view_proj, bool stencil_ids)
    {
        mgl::glUseProgram(scene_prog_);
        mgl::glUniformMatrix4fv(u_scene_vp_, 1, GL_FALSE, view_proj.data());
        mgl::glBindVertexArray(scene_vao_);
        for (const SceneMesh::Range& r : mesh_.ranges)
        {
            if (stencil_ids) mgl::glStencilFunc(CF::Always, r.id, 0xFF);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, r.first, r.count);
        }
    }

    void DepthTextures::render_depth_targets()
    {
        mgl::glEnable(mgl::Capability::DepthTest);
        mgl::glDepthFunc(CF::Less);
        mgl::glDepthMask(GL_TRUE);
        const float one = 1.0f;

        // Camera A into all five formats. The depth-stencil formats also store the object id.
        const dm::Mat4 vp_a = camera_a().view_proj();
        for (std::size_t k = 0; k < 5; ++k)
        {
            mgl::glBindFramebuffer(FT::Framebuffer, scene_fbo_[k]);
            mgl::glViewport(0, 0, kSceneW, kSceneH);
            if (kFormats[k].has_stencil)
            {
                mgl::glStencilMask(0xFF);
                mgl::glClearBufferfi(1.0f, 0);
                mgl::glEnable(mgl::Capability::StencilTest);
                mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Replace);
            }
            else
            {
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Depth, 0, &one);
            }
            draw_scene(vp_a, kFormats[k].has_stencil);
            mgl::glDisable(mgl::Capability::StencilTest);
        }

        // Array layers.
        const dm::Mat4 vp_layers[2] = {vp_a, camera_b().view_proj()};
        for (std::size_t layer = 0; layer < 2; ++layer)
        {
            mgl::glBindFramebuffer(FT::Framebuffer, array_fbo_[layer]);
            mgl::glViewport(0, 0, kSceneW, kSceneH);
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Depth, 0, &one);
            draw_scene(vp_layers[layer], false);
        }

        // Cube faces.
        for (int face = 0; face < 6; ++face)
        {
            mgl::glBindFramebuffer(FT::Framebuffer, cube_fbo_[static_cast<std::size_t>(face)]);
            mgl::glViewport(0, 0, kCubeSize, kCubeSize);
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Depth, 0, &one);
            draw_scene(cube_camera(face).view_proj(), false);
        }
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glBindVertexArray(empty_vao_);

        // Visualizations (RGBA8, texture resolution).
        const Camera cam = camera_a();
        for (std::size_t k = 0; k < 5; ++k)
        {
            mgl::glBindFramebuffer(FT::Framebuffer, vis_fbo_[k]);
            mgl::glViewport(0, 0, kSceneW, kSceneH);
            mgl::glUseProgram(vis_prog_);
            bind_unit(0, TT::Texture2D, scene_tex_[k], raw_sampler_);
            mgl::glUniform2f(u_vis_near_far_, static_cast<float>(cam.n), static_cast<float>(cam.f));
            mgl::glUniform2f(u_vis_tan_, static_cast<float>(cam.tan_half() * cam.aspect), static_cast<float>(cam.tan_half()));
            mgl::glUniform2f(u_vis_size_, float(kSceneW), float(kSceneH));
            mgl::glUniform3f(u_vis_sky_, kSky[0], kSky[1], kSky[2]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            if (!kFormats[k].has_stencil) continue;

            // Tint every box by its stencil id: stencil test against the depth-stencil texture's stencil.
            mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
            mgl::glBindFramebuffer(FT::Framebuffer, tint_fbo_[k]);
            mgl::glUseProgram(solid_prog_);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glEnable(mgl::Capability::Blend);
            mgl::glBlendFunc(mgl::BlendFactor::SrcAlpha, mgl::BlendFactor::OneMinusSrcAlpha);
            for (const Box& b : kBoxes)
            {
                const auto& c = kIdTint[static_cast<std::size_t>(b.id)];
                mgl::glStencilFunc(CF::Equal, b.id, 0xFF);
                mgl::glUniform4f(u_solid_color_, c[0], c[1], c[2], 0.30f);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::StencilTest);
        }
    }

    void DepthTextures::render_window(const demo::Env& env)
    {
        const Layout l = make_layout(env.width, env.height);
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        mgl::glViewport(0, 0, env.width, env.height);
        mgl::glClearColor(kBackground[0], kBackground[1], kBackground[2], 1.0f);
        mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);
        mgl::glBindVertexArray(empty_vao_);

        mgl::glUseProgram(show_prog_);
        for (std::size_t k = 0; k < 5; ++k)
        {
            const Rect& r = l.panel[k];
            mgl::glViewport(r.x, r.y, r.w, r.h);
            bind_unit(0, TT::Texture2D, vis_tex_[k], linear_sampler_);
            mgl::glUniform4f(u_show_rect_, float(r.x), float(r.y), float(r.w), float(r.h));
            mgl::glUniform4f(u_show_uv_, static_cast<float>(l.uv[k].sx), static_cast<float>(l.uv[k].sy), static_cast<float>(l.uv[k].ox),
                             static_cast<float>(l.uv[k].oy));
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        // Compare grid: raw and comparing views of the same texture on two units.
        mgl::glUseProgram(grid_prog_);
        mgl::glUniform1f(u_grid_band_, kBand);
        bind_unit(kUnitRaw2D, TT::Texture2D, scene_tex_[kF32F], raw_sampler_);
        bind_unit(kUnitRawArray, TT::Texture2DArray, array_tex_, raw_sampler_);
        bind_unit(kUnitRawCube, TT::TextureCubeMap, cube_tex_, raw_sampler_);
        bind_unit(kUnitShadow2D, TT::Texture2D, scene_tex_[kF32F], mgl::SamplerId{0});
        bind_unit(kUnitShadowArray, TT::Texture2DArray, array_tex_, mgl::SamplerId{0});
        bind_unit(kUnitShadowCube, TT::TextureCubeMap, cube_tex_, mgl::SamplerId{0});
        for (int row = 0; row < 3; ++row)
        {
            mgl::glUniform1i(u_grid_kind_, row);
            if (row < 2)
            {
                mgl::glUniform2f(u_grid_near_far_, static_cast<float>(kNearA), static_cast<float>(kFarA));
                mgl::glUniform2f(u_grid_slice_, 3.0f, 30.0f);
            }
            else
            {
                mgl::glUniform2f(u_grid_near_far_, static_cast<float>(kCubeNear), static_cast<float>(kCubeFar));
                mgl::glUniform2f(u_grid_slice_, 1.0f, 14.0f);
            }
            for (std::size_t f = 0; f < 8; ++f)
            {
                const int x = l.cell_x[f], y = l.cell_y[static_cast<std::size_t>(row)];
                mgl::glViewport(x, y, l.cell_w, l.cell_h);
                for (int unit : {kUnitShadow2D, kUnitShadowArray, kUnitShadowCube}) mgl::glBindSampler(static_cast<GLuint>(unit), cmp_sampler_[f]);
                mgl::glUniform4f(u_grid_cell_, float(x), float(y), float(l.cell_w), float(l.cell_h));
                mgl::glUniform3f(u_grid_pass_, kFuncs[f].color[0], kFuncs[f].color[1], kFuncs[f].color[2]);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
        }
        for (int unit = 0; unit < 6; ++unit) bind_unit(unit, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glViewport(0, 0, env.width, env.height);
    }

    void DepthTextures::render(demo::Env& env, int /*frame*/)
    {
        width_ = env.width;
        height_ = env.height;
        render_depth_targets();
        render_window(env);
    }

    // ==========================================================================================
    // probe / stencil read-out helpers
    // ==========================================================================================

    std::vector<Probe> DepthTextures::probe(int mode, std::span<const Lookup> lookups)
    {
        const int n = static_cast<int>(lookups.size());
        if (n == 0 || n > 64) throw demo::Failure("probe count out of range");
        std::vector<float> coords(256, 0.0f), refs(64, 0.0f);
        for (int k = 0; k < n; ++k)
        {
            const Lookup& lk = lookups[static_cast<std::size_t>(k)];
            coords[static_cast<std::size_t>(4 * k)] = lk.x;
            coords[static_cast<std::size_t>(4 * k + 1)] = lk.y;
            coords[static_cast<std::size_t>(4 * k + 2)] = lk.z;
            refs[static_cast<std::size_t>(k)] = lk.ref;
        }
        mgl::glBindFramebuffer(FT::Framebuffer, probe_fbo_);
        mgl::glViewport(0, 0, n, 1);
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glDisable(mgl::Capability::StencilTest);
        mgl::glDisable(mgl::Capability::Blend);
        mgl::glUseProgram(probe_prog_);
        mgl::glUniform1i(u_probe_mode_, mode);
        mgl::glUniform4fv(u_probe_coord_, 64, coords.data());
        mgl::glUniform1fv(u_probe_ref_, 64, refs.data());
        mgl::glBindVertexArray(empty_vao_);
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        std::vector<std::uint32_t> raw(static_cast<std::size_t>(4 * n));
        mgl::glReadPixels(0, 0, n, 1, PF::RgbaInteger, PT::UnsignedInt, raw.data());
        std::vector<Probe> out(static_cast<std::size_t>(n));
        for (std::size_t k = 0; k < out.size(); ++k) out[k] = {raw[4 * k], raw[4 * k + 1], raw[4 * k + 2], raw[4 * k + 3]};
        return out;
    }

    /// Reads the stencil component of a depth-stencil texture through the stencil test: the texture
    /// is the DEPTH_STENCIL_ATTACHMENT of a scratch framebuffer with an RGBA8 color buffer, and pass b
    /// writes 1.0 into channel (b mod 4) where (stencil & (1 << b)) != 0 (StencilFunc EQUAL, ref = mask
    /// = 1 << b). Two rounds of four bits recover all eight bits of every texel.
    std::vector<int> DepthTextures::read_stencil(mgl::TextureId ds_texture, int format, int w, int h)
    {
        mgl::TextureId color{};
        mgl::FramebufferId fbo{};
        mgl::glGenTextures(1, &color);
        mgl::glBindTexture(TT::Texture2D, color);
        mgl::glTexStorage2D(TT::Texture2D, 1, SF::Rgba8, w, h);
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
        mgl::glGenFramebuffers(1, &fbo);
        mgl::glBindFramebuffer(FT::Framebuffer, fbo);
        mgl::glFramebufferTexture2D(FT::Framebuffer, kColor0, TT::Texture2D, color, 0);
        attach_depth(FT::Framebuffer, format, TT::Texture2D, ds_texture);
        std::vector<int> result(static_cast<std::size_t>(w * h), 0);
        if (mgl::glCheckFramebufferStatus(FT::Framebuffer) == mgl::FramebufferStatus::Complete)
        {
            mgl::glViewport(0, 0, w, h);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glUseProgram(solid_prog_);
            mgl::glUniform4f(u_solid_color_, 1.0f, 1.0f, 1.0f, 1.0f);
            mgl::glBindVertexArray(empty_vao_);
            const float zero[4] = {0, 0, 0, 0};
            for (int round = 0; round < 2; ++round)
            {
                mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, zero);
                for (int c = 0; c < 4; ++c)
                {
                    const int bit = round * 4 + c;
                    mgl::glColorMask(c == 0, c == 1, c == 2, c == 3);
                    mgl::glStencilFunc(CF::Equal, 1 << bit, static_cast<GLuint>(1 << bit));
                    mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                }
                mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, w, h);
                for (std::size_t p = 0; p < result.size(); ++p)
                    for (int c = 0; c < 4; ++c)
                        if (px[4 * p + static_cast<std::size_t>(c)] > 127) result[p] |= 1 << (round * 4 + c);
            }
            mgl::glDisable(mgl::Capability::StencilTest);
        }
        else
        {
            std::fill(result.begin(), result.end(), -1);
        }
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        mgl::glDeleteFramebuffers(1, &fbo);
        mgl::glDeleteTextures(1, &color);
        return result;
    }

    /// Clears strip `format` and writes kFragDepths into texels 0..14 through gl_FragDepth.
    void DepthTextures::fill_frag_depth_strip(int format, float clear_depth, int clear_stencil)
    {
        const auto k = static_cast<std::size_t>(format);
        mgl::glBindFramebuffer(FT::Framebuffer, q_fbo_[k]);
        mgl::glViewport(0, 0, 15, 1);
        if (kFormats[k].has_stencil)
        {
            mgl::glStencilMask(0xFF);
            mgl::glClearBufferfi(clear_depth, clear_stencil);
        }
        else
        {
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Depth, 0, &clear_depth);
        }
        // Depth writes happen only with the depth test enabled (§4.1.5); ALWAYS keeps every fragment.
        mgl::glEnable(mgl::Capability::DepthTest);
        mgl::glDepthFunc(CF::Always);
        mgl::glDepthMask(GL_TRUE);
        mgl::glDisable(mgl::Capability::StencilTest);
        mgl::glUseProgram(frag_depth_prog_);
        std::array<float, 16> values{};
        std::copy(kFragDepths.begin(), kFragDepths.end(), values.begin());
        mgl::glUniform1fv(u_fd_depth_, 16, values.data());
        mgl::glBindVertexArray(empty_vao_);
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glDepthFunc(CF::Less);
    }

    // ==========================================================================================
    // validate
    // ==========================================================================================

    void DepthTextures::validate(demo::Env& env, demo::Check& check)
    {
        check_window(env, check);  // first: everything below draws offscreen
        check_attachments(check);
        check_uploads(check);
        check_format_errors(check);
        check_quantization(check);
        check_scene(check);
        check_compare_2d(check);
        check_compare_array_cube(check);
        check_linear(check);
        check_completeness(check);
        check_stencil(check);
        for (int unit = 0; unit < 6; ++unit) bind_unit(unit, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        check.expect_no_gl_error("no GL error after all validation passes");
    }

    // -- window ----------------------------------------------------------------------------------

    void DepthTextures::check_window(const demo::Env& env, demo::Check& check)
    {
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        const int w = env.width, h = env.height;
        const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, w, h);
        auto at = [&](int x, int y) {
            const std::size_t o = 4 * (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x));
            return demo::Rgba8{img[o], img[o + 1], img[o + 2], img[o + 3]};
        };
        const Layout l = make_layout(w, h);

        // Panels: in the visible (cover-cropped) part of each panel, a sky texel whose 5x5 neighbourhood is
        // sky in the CPU model shows exactly the sky color, and a ground texel does not. LINEAR display
        // filtering of equal texels is exact.
        const Camera cam = camera_a();
        std::vector<int> ids(static_cast<std::size_t>(kSceneW * kSceneH));
        for (int j = 0; j < kSceneH; ++j)
            for (int i = 0; i < kSceneW; ++i) ids[static_cast<std::size_t>(j * kSceneW + i)] = trace(cam.eye, cam.texel_dir(i, j, kSceneW, kSceneH)).id;
        auto uniform_id = [&](int i, int j, int id) {
            for (int dj = -2; dj <= 2; ++dj)
                for (int di = -2; di <= 2; ++di)
                    if (ids[static_cast<std::size_t>((j + dj) * kSceneW + i + di)] != id) return false;
            return true;
        };
        const demo::Rgba8 sky{to8(kSky[0]), to8(kSky[1]), to8(kSky[2]), 255};
        int panel_bad = 0;
        std::string panel_detail;
        for (std::size_t k = 0; k < 5; ++k)
        {
            const Rect& r = l.panel[k];
            const PanelUv& uv = l.uv[k];
            const int i_lo = static_cast<int>(std::ceil(uv.ox * kSceneW)) + 3, i_hi = static_cast<int>((uv.ox + uv.sx) * kSceneW) - 4;
            const int j_lo = static_cast<int>(std::ceil(uv.oy * kSceneH)) + 3, j_hi = static_cast<int>((uv.oy + uv.sy) * kSceneH) - 4;
            auto to_px = [&](int i, int j) {
                const double lx = ((i + 0.5) / kSceneW - uv.ox) / uv.sx, ly = ((j + 0.5) / kSceneH - uv.oy) / uv.sy;
                return std::array<int, 2>{r.x + static_cast<int>(lx * r.w), r.y + static_cast<int>(ly * r.h)};
            };
            std::array<int, 2> sky_t{-1, -1}, ground_t{-1, -1};
            for (int j = j_hi; j >= j_lo && sky_t[0] < 0; --j)
                for (int i = (i_lo + i_hi) / 2; i <= i_hi && sky_t[0] < 0; ++i)
                    if (uniform_id(i, j, 0)) sky_t = {i, j};
            for (int j = j_lo; j <= j_hi && ground_t[0] < 0; ++j)
                for (int i = (i_lo + i_hi) / 2; i <= i_hi && ground_t[0] < 0; ++i)
                    if (uniform_id(i, j, kGroundId)) ground_t = {i, j};
            if (sky_t[0] < 0 || ground_t[0] < 0)
            {
                ++panel_bad;
                panel_detail += std::format("{}: no sky / ground texel in the visible region; ", kFormats[k].name);
                continue;
            }
            const auto sp = to_px(sky_t[0], sky_t[1]);
            const demo::Rgba8 got = at(sp[0], sp[1]);
            bool ok = true;
            for (int c = 0; c < 3; ++c) ok = ok && std::abs(int(got[static_cast<std::size_t>(c)]) - int(sky[static_cast<std::size_t>(c)])) <= 1;
            const auto gp = to_px(ground_t[0], ground_t[1]);
            const demo::Rgba8 g = at(gp[0], gp[1]);
            int diff = 0;
            for (int c = 0; c < 3; ++c) diff = std::max(diff, std::abs(int(g[static_cast<std::size_t>(c)]) - int(sky[static_cast<std::size_t>(c)])));
            ok = ok && diff > 12;
            if (!ok)
            {
                ++panel_bad;
                panel_detail += std::format("{}: sky px({},{})={} ground px({},{})={}; ", kFormats[k].name, sp[0], sp[1], demo::format_rgba(got), gp[0],
                                            gp[1], demo::format_rgba(g));
            }
        }
        check.expect(panel_bad == 0, "format panels: cleared depth 1.0 shows the sky color (+-1), the ground does not", panel_detail);

        // Grid. Classify a pixel: lit (function passed) or dark (failed).
        auto classify = [&](int x, int y) {
            const demo::Rgba8 c = at(x, y);
            const int mx = std::max({int(c[0]), int(c[1]), int(c[2])});
            return mx >= 115 ? 1 : (mx <= 64 ? 0 : -1);
        };
        const char* row_names[3] = {"sampler2DShadow", "sampler2DArrayShadow (layer 1)", "samplerCubeShadow"};
        const int band_row = static_cast<int>(std::floor(0.125 * l.cell_h));
        const int upper_first = static_cast<int>(std::ceil(kBand * l.cell_h)) + 1;
        for (std::size_t row = 0; row < 3; ++row)
        {
            // Bottom band: Dref == Dt exactly (DEPTH_COMPONENT32F, the raw value of the same texel).
            std::string got, want;
            bool band_ok = true;
            for (std::size_t f = 0; f < 8; ++f)
            {
                const int expect = table_3_23(kFuncs[f].func, 0.5, 0.5) ? 1 : 0;
                const int c = classify(l.cell_x[f] + l.cell_w / 2, l.cell_y[row] + band_row);
                band_ok = band_ok && c == expect;
                got += std::format("{}={} ", kFuncs[f].name, c);
                want += std::format("{}={} ", kFuncs[f].name, expect);
            }
            check.expect(band_ok, std::format("grid {}: band with Dref == Dt is lit exactly for LEQUAL, GEQUAL, EQUAL, ALWAYS", row_names[row]),
                         band_ok ? std::string{} : "got " + got + "expected " + want);

            // Upper part: identical cell sizes -> identical (texel, Dref) at the same local pixel in
            // every column, so Table 3.23 relations between the functions must hold pixel by pixel.
            int violations = 0, unclassified = 0, lequal_pass = 0, lequal_fail = 0, samples = 0;
            std::string first;
            for (int b = 0; b < 4; ++b)
                for (int a = 0; a < 6; ++a)
                {
                    const int lx = static_cast<int>((a + 0.5) / 6.0 * l.cell_w);
                    const int ly = upper_first + static_cast<int>((b + 0.5) / 4.0 * (l.cell_h - upper_first - 1));
                    std::array<int, 8> r{};
                    for (std::size_t f = 0; f < 8; ++f) r[f] = classify(l.cell_x[f] + lx, l.cell_y[row] + ly);
                    ++samples;
                    if (std::any_of(r.begin(), r.end(), [](int v) { return v < 0; }))
                    {
                        ++unclassified;
                        continue;
                    }
                    const bool ok = r[kNever] == 0 && r[kAlways] == 1 && r[kLequal] != r[kGreater] && r[kGequal] != r[kLess] &&
                                    r[kEqual] != r[kNotequal] && r[kLequal] == (r[kLess] | r[kEqual]) && r[kGequal] == (r[kGreater] | r[kEqual]);
                    if (!ok)
                    {
                        ++violations;
                        if (first.empty())
                            first = std::format("local ({},{}): N{} L{} E{} LE{} G{} NE{} GE{} A{}", lx, ly, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
                    }
                    (r[kLequal] ? lequal_pass : lequal_fail) += 1;
                }
            check.expect(violations == 0 && unclassified == 0,
                         std::format("grid {}: at {} pixels NEVER=0, ALWAYS=1, LEQUAL=!GREATER, GEQUAL=!LESS, EQUAL=!NOTEQUAL, "
                                     "LEQUAL=LESS|EQUAL, GEQUAL=GREATER|EQUAL",
                                     row_names[row], samples),
                         std::format("{} violations, {} unclassifiable; {}", violations, unclassified, first));
            check.expect(lequal_pass > 0 && lequal_fail > 0, std::format("grid {}: the slicing surface splits LEQUAL into lit and dark pixels", row_names[row]),
                         std::format("pass {} fail {}", lequal_pass, lequal_fail));
        }
    }

    // -- attachments / state -------------------------------------------------------------------

    void DepthTextures::check_attachments(demo::Check& check)
    {
        for (std::size_t k = 0; k < 5; ++k)
        {
            const DepthFormat& fmt = kFormats[k];
            mgl::glBindFramebuffer(FT::Framebuffer, scene_fbo_[k]);
            const bool complete = mgl::glCheckFramebufferStatus(FT::Framebuffer) == mgl::FramebufferStatus::Complete;
            const int depth_bits = fb_param(FA::Depth, FAP::DepthSize);
            const int type = fb_param(FA::Depth, FAP::ComponentType);
            const int obj_type = fb_param(FA::Depth, FAP::ObjectType);
            const int obj_name = fb_param(FA::Depth, FAP::ObjectName);
            const int want_type = fmt.is_float ? GL_FLOAT : GL_UNSIGNED_NORMALIZED;
            const bool ok = complete && depth_bits >= fmt.min_bits && (!fmt.is_float || depth_bits == 32) && type == want_type &&
                            obj_type == GL_TEXTURE && obj_name == static_cast<int>(scene_tex_[k].value);
            check.expect(ok, std::format("{}: depth-only texture framebuffer complete, DEPTH_SIZE >= {} (Table 3.14), COMPONENT_TYPE {}", fmt.name,
                                         fmt.min_bits, fmt.is_float ? "FLOAT" : "UNSIGNED_NORMALIZED"),
                         std::format("complete={} DEPTH_SIZE={} COMPONENT_TYPE=0x{:X} OBJECT_TYPE=0x{:X} OBJECT_NAME={}", complete, depth_bits, type,
                                     obj_type, obj_name));
            std::cout << std::format("   [info] {}: FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE = {}\n", fmt.name, depth_bits);

            const int stencil_bits = fb_param(FA::Stencil, FAP::ObjectType) == GL_NONE ? 0 : fb_param(FA::Stencil, FAP::StencilSize);
            if (fmt.has_stencil)
            {
                const int stencil_name = fb_param(FA::Stencil, FAP::ObjectName);
                const int ds_name = fb_param(FA::DepthStencil, FAP::ObjectName);
                check.expect(stencil_bits >= 8 && stencil_name == static_cast<int>(scene_tex_[k].value) && ds_name == stencil_name,
                             std::format("{} at DEPTH_STENCIL_ATTACHMENT: the texture is bound to DEPTH and STENCIL, STENCIL_SIZE >= 8", fmt.name),
                             std::format("STENCIL_SIZE={} STENCIL name={} DEPTH_STENCIL name={}", stencil_bits, stencil_name, ds_name));
                GLint dummy = 0;
                mgl::glGetFramebufferAttachmentParameteriv(FT::Framebuffer, FA::DepthStencil, FAP::ComponentType, &dummy);
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                      std::format("{}: COMPONENT_TYPE of DEPTH_STENCIL_ATTACHMENT -> INVALID_OPERATION (§6.1.13)", fmt.name));
            }
            else
            {
                check.expect_eq(fb_param(FA::Stencil, FAP::ObjectType), GL_NONE, std::format("{}: nothing attached at STENCIL_ATTACHMENT", fmt.name));
            }
        }
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        check.expect_no_gl_error("attachment queries raise no other error");

        // Initial comparison state of a texture and the sampler object state (state tables 6.10 / 6.11).
        mgl::TextureId fresh{};
        mgl::glGenTextures(1, &fresh);
        mgl::glBindTexture(TT::Texture2D, fresh);
        GLint mode = -1, func = -1;
        mgl::glGetTexParameteriv(TT::Texture2D, mgl::TextureParameterQuery::CompareMode, &mode);
        mgl::glGetTexParameteriv(TT::Texture2D, mgl::TextureParameterQuery::CompareFunc, &func);
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
        mgl::glDeleteTextures(1, &fresh);
        check.expect(mode == GL_NONE && func == GL_LEQUAL, "initial TEXTURE_COMPARE_MODE = NONE, TEXTURE_COMPARE_FUNC = LEQUAL",
                     std::format("mode=0x{:X} func=0x{:X}", mode, func));
        bool samplers_ok = true;
        for (std::size_t f = 0; f < 8; ++f)
        {
            GLint m = -1, fn = -1;
            mgl::glGetSamplerParameteriv(cmp_sampler_[f], mgl::SamplerParameter::CompareMode, &m);
            mgl::glGetSamplerParameteriv(cmp_sampler_[f], mgl::SamplerParameter::CompareFunc, &fn);
            samplers_ok = samplers_ok && m == GL_COMPARE_REF_TO_TEXTURE && fn == static_cast<GLint>(kFuncs[f].func);
        }
        check.expect(samplers_ok, "the eight comparing sampler objects read back COMPARE_REF_TO_TEXTURE and their function");
    }

    // -- uploads -------------------------------------------------------------------------------

    void DepthTextures::check_uploads(demo::Check& check)
    {
        std::vector<Lookup> lk;
        for (int t = 0; t < kUpN; ++t) lk.push_back({(t % kUpW + 0.5f) / kUpW, (t / kUpW + 0.5f) / kUpH, 0.0f, 0.0f});
        for (int u = 0; u < kUploadCount; ++u)
        {
            const UploadInfo& info = kUploads[static_cast<std::size_t>(u)];
            const DepthFormat& fmt = kFormats[static_cast<std::size_t>(info.format)];
            const int bits = depth_bits_[static_cast<std::size_t>(info.format)];
            bind_unit(kUnitRaw2D, TT::Texture2D, up_tex_[static_cast<std::size_t>(u)], raw_sampler_);
            const std::vector<Probe> got = probe(kModeRaw2D, lk);
            int bad = 0, bad_gba = 0;
            double max_err = 0.0;
            std::string first;
            for (int t = 0; t < kUpN; ++t)
            {
                const Probe& p = got[static_cast<std::size_t>(t)];
                const double f = upload_value(u, t);
                const double s = bits_to_float(p[0]);
                bool ok;
                if (fmt.is_float)
                    ok = p[0] == float_to_bits(static_cast<float>(f));  // stored without conversion: exact
                else
                    ok = fixed_interval(f, bits).contains(s);
                max_err = std::max(max_err, std::abs(s - f));
                if (!ok)
                {
                    ++bad;
                    if (first.empty()) first = std::format("texel {}: got {:.9g} expected {:.9g} ({} bits)", t, s, f, bits);
                }
                // §3.9.2.1: depth expands to the base color (Rt, 0, 0, 1).
                if (p[1] != 0u || p[2] != 0u || p[3] != float_to_bits(1.0f)) ++bad_gba;
            }
            check.expect(bad == 0,
                         std::format("upload {}: all 16 texels sample back {}", info.label,
                                     fmt.is_float ? "bit-exactly (clamped to [0, 1], no conversion)" : "within one step of the stored precision of c/(2^n-1)"),
                         std::format("{} mismatches; {}; max |error| {:.3g}", bad, first, max_err));
            check.expect(bad_gba == 0, std::format("upload {}: lookups through sampler2D return (D, 0, 0, 1)", info.label),
                         std::format("{} texels with other g/b/a", bad_gba));
            std::cout << std::format("   [info] {}: max |sampled - c/(2^n-1)| = {:.3g}\n", info.label, max_err);
        }

        // The clamp on upload: FLOAT -0.25 and 1.75 into DEPTH_COMPONENT32F become 0 and 1.
        bind_unit(kUnitRaw2D, TT::Texture2D, up_tex_[kUp32Float], raw_sampler_);
        const std::vector<Probe> clamp = probe(kModeRaw2D, std::span<const Lookup>(lk).subspan(3, 2));
        check.expect(clamp[0][0] == float_to_bits(0.0f) && clamp[1][0] == float_to_bits(1.0f),
                     "DEPTH_COMPONENT32F upload of -0.25 / 1.75 is clamped to 0 / 1 (§3.8.3)",
                     std::format("got {} / {}", bits_to_float(clamp[0][0]), bits_to_float(clamp[1][0])));

        // TexSubImage2D: DEPTH_COMPONENT16 accepts UNSIGNED_INT data too (Table 3.2), DEPTH_COMPONENT24
        // rejects UNSIGNED_SHORT and keeps its contents.
        mgl::glBindTexture(TT::Texture2D, up_tex_[kUp16Short]);
        const std::uint32_t sub32 = 0x20000000u;  // 0.125
        mgl::glTexSubImage2D(TT::Texture2D, 0, 7, 1, 1, 1, PF::DepthComponent, PT::UnsignedInt, &sub32);
        check.expect_no_gl_error("TexSubImage2D DEPTH_COMPONENT / UNSIGNED_INT into DEPTH_COMPONENT16: no error");
        mgl::glBindTexture(TT::Texture2D, up_tex_[kUp24Int]);
        const std::uint16_t sub16 = 0x1234;
        mgl::glTexSubImage2D(TT::Texture2D, 0, 7, 1, 1, 1, PF::DepthComponent, PT::UnsignedShort, &sub16);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TexSubImage2D DEPTH_COMPONENT / UNSIGNED_SHORT into DEPTH_COMPONENT24 -> INVALID_OPERATION");
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
        const Lookup last[1] = {lk[15]};
        bind_unit(kUnitRaw2D, TT::Texture2D, up_tex_[kUp16Short], raw_sampler_);
        const double v16 = bits_to_float(probe(kModeRaw2D, last)[0][0]);
        bind_unit(kUnitRaw2D, TT::Texture2D, up_tex_[kUp24Int], raw_sampler_);
        const double v24 = bits_to_float(probe(kModeRaw2D, last)[0][0]);
        check.expect(fixed_interval(sub32 / 4294967295.0, depth_bits_[kF16]).contains(v16), "DEPTH_COMPONENT16 texel (7,1) now holds the UNSIGNED_INT value 0.125",
                     std::format("got {:.9g}", v16));
        check.expect(fixed_interval(kU32For24[15] / 4294967295.0, depth_bits_[kF24]).contains(v24),
                     "DEPTH_COMPONENT24 texel (7,1) is unchanged after the rejected TexSubImage2D", std::format("got {:.9g}", v24));
    }

    void DepthTextures::check_format_errors(demo::Check& check)
    {
        struct Bad
        {
            IF internal;
            PF format;
            PT type;
            const char* what;
        };
        const Bad bad[] = {
            {IF::DepthComponent16, PF::DepthComponent, PT::Float, "DEPTH_COMPONENT16 with FLOAT"},
            {IF::DepthComponent24, PF::DepthComponent, PT::UnsignedShort, "DEPTH_COMPONENT24 with UNSIGNED_SHORT"},
            {IF::DepthComponent32F, PF::DepthComponent, PT::UnsignedInt, "DEPTH_COMPONENT32F with UNSIGNED_INT"},
            {IF::DepthComponent24, PF::DepthStencil, PT::UnsignedInt248, "DEPTH_COMPONENT24 with DEPTH_STENCIL / UNSIGNED_INT_24_8"},
            {IF::Depth24Stencil8, PF::DepthComponent, PT::UnsignedInt, "DEPTH24_STENCIL8 with DEPTH_COMPONENT / UNSIGNED_INT"},
            {IF::Depth24Stencil8, PF::DepthStencil, PT::Float32UnsignedInt248Rev, "DEPTH24_STENCIL8 with FLOAT_32_UNSIGNED_INT_24_8_REV"},
            {IF::Depth32FStencil8, PF::DepthStencil, PT::UnsignedInt248, "DEPTH32F_STENCIL8 with UNSIGNED_INT_24_8"},
            {IF::Depth32FStencil8, PF::DepthComponent, PT::Float, "DEPTH32F_STENCIL8 with DEPTH_COMPONENT / FLOAT"},
            {IF::DepthComponent32F, PF::DepthComponent, PT::Float32UnsignedInt248Rev, "DEPTH_COMPONENT with the two-component FLOAT_32_UNSIGNED_INT_24_8_REV"},
        };
        mgl::TextureId scratch{};
        mgl::glGenTextures(1, &scratch);
        mgl::glBindTexture(TT::Texture2D, scratch);
        for (const Bad& b : bad)
        {
            mgl::glTexImage2D(TT::Texture2D, 0, b.internal, 4, 4, 0, b.format, b.type, nullptr);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, std::format("TexImage2D {} -> INVALID_OPERATION (Table 3.2)", b.what));
        }
        // Positive control on the same texture: a Table 3.2 combination that is not the "obvious" one.
        mgl::glTexImage2D(TT::Texture2D, 0, IF::DepthComponent16, 4, 4, 0, PF::DepthComponent, PT::UnsignedInt, nullptr);
        check.expect_no_gl_error("TexImage2D DEPTH_COMPONENT16 with UNSIGNED_INT is valid (Table 3.2)");
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
        mgl::glDeleteTextures(1, &scratch);

        // Depth formats only for 2D, 2D array and cube textures (§3.8.3).
        mgl::TextureId tex3d{}, arr{};
        mgl::glGenTextures(1, &tex3d);
        mgl::glGenTextures(1, &arr);
        mgl::glBindTexture(TT::Texture3D, tex3d);
        mgl::glTexImage3D(TT::Texture3D, 0, IF::DepthComponent24, 4, 4, 2, 0, PF::DepthComponent, PT::UnsignedInt, nullptr);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TexImage3D TEXTURE_3D with DEPTH_COMPONENT24 -> INVALID_OPERATION (§3.8.3)");
        mgl::glBindTexture(TT::Texture2DArray, arr);
        mgl::glTexImage3D(TT::Texture2DArray, 0, IF::DepthComponent24, 4, 4, 2, 0, PF::DepthComponent, PT::UnsignedInt, nullptr);
        check.expect_no_gl_error("TexImage3D TEXTURE_2D_ARRAY with DEPTH_COMPONENT24 is valid (positive control)");
        mgl::glBindTexture(TT::Texture3D, mgl::TextureId{0});
        mgl::glBindTexture(TT::Texture2DArray, mgl::TextureId{0});
        mgl::glDeleteTextures(1, &tex3d);
        mgl::glDeleteTextures(1, &arr);
    }

    // -- gl_FragDepth quantization ---------------------------------------------------------------

    void DepthTextures::check_quantization(demo::Check& check)
    {
        std::vector<Lookup> lk;
        for (int t = 0; t < 16; ++t) lk.push_back({(t + 0.5f) / 16.0f, 0.5f, 0.0f, 0.0f});
        std::array<std::vector<double>, 5> sampled;
        for (int k = 0; k < 5; ++k)
        {
            const auto uk = static_cast<std::size_t>(k);
            const DepthFormat& fmt = kFormats[uk];
            const int bits = depth_bits_[uk];
            fill_frag_depth_strip(k, kClearDepth, kClearStencil);
            bind_unit(kUnitRaw2D, TT::Texture2D, q_tex_[uk], raw_sampler_);
            const std::vector<Probe> got = probe(kModeRaw2D, lk);
            int bad = 0;
            std::string first, values;
            for (int t = 0; t < 16; ++t)
            {
                const float written = t < 15 ? kFragDepths[static_cast<std::size_t>(t)] : kClearDepth;
                const double f = std::clamp(static_cast<double>(written), 0.0, 1.0);  // §3.9.2.3: clamped to [0, 1] first
                const double s = bits_to_float(got[static_cast<std::size_t>(t)][0]);
                sampled[uk].push_back(s);
                const bool ok = fmt.is_float ? got[static_cast<std::size_t>(t)][0] == float_to_bits(static_cast<float>(f)) : fixed_interval(f, bits).contains(s);
                if (!ok)
                {
                    ++bad;
                    if (first.empty()) first = std::format("texel {}: wrote {:.9g} got {:.9g}", t, written, s);
                }
                values += std::format("{:.9g} ", s);
            }
            check.expect(bad == 0,
                         std::format("{}: 15 gl_FragDepth values and the {} clear value are stored {}", fmt.name,
                                     fmt.has_stencil ? "ClearBufferfi" : "ClearBufferfv",
                                     fmt.is_float ? "exactly (only clamped, §3.9.2.3)" : std::format("as one of the two nearest {}-bit values (§2.1.6.2)", bits)),
                         std::format("{} mismatches; {}", bad, first));
            std::cout << std::format("   [info] {} strip: {}\n", fmt.name, values);
        }

        // Negative controls: the precision is real. A 24-bit expectation fails on DEPTH_COMPONENT16 at the
        // values placed 0.3 / 0.7 of a 16-bit step off the grid, and DEPTH_COMPONENT24 is not bit-exact
        // where DEPTH_COMPONENT32F is.
        if (depth_bits_[kF16] == 16)
        {
            int off = 0;
            for (int t : kOff16Grid)
                if (!fixed_interval(kFragDepths[static_cast<std::size_t>(t)], 24).contains(sampled[kF16][static_cast<std::size_t>(t)])) ++off;
            check.expect(off == static_cast<int>(kOff16Grid.size()),
                         "negative control: DEPTH_COMPONENT16 misses a 24-bit expectation at all 5 off-grid values (16-bit quantization is visible)",
                         std::format("{} of 5", off));
        }
        else
        {
            std::cout << std::format("   [note] DEPTH_COMPONENT16 is stored with {} bits; the 16-bit negative control does not apply\n", depth_bits_[kF16]);
        }
        int differ = 0;
        for (int t = 0; t < 15; ++t)
            if (float_to_bits(static_cast<float>(sampled[kF24][static_cast<std::size_t>(t)])) != float_to_bits(static_cast<float>(sampled[kF32F][static_cast<std::size_t>(t)])))
                ++differ;
        // Not asserted: the float returned for a 24-bit value may round to the same float as the 32F value
        // (radeonsi returns identical bits for all 15 values), so a difference is not guaranteed.
        std::cout << std::format("   [info] DEPTH_COMPONENT24 vs DEPTH_COMPONENT32F: {} of 15 sampled values differ in their float bits\n", differ);
    }

    // -- render-to-depth: scene, array layers, cube faces ---------------------------------------------

    void DepthTextures::check_scene(demo::Check& check)
    {
        // Sanity tolerance: the spec gives no bound for depth interpolation precision. 1e-4 in window depth
        // is far below the depth difference between neighbouring objects and still catches a wrong texel
        // row on the far ground; one step of the stored precision is added for the fixed-point formats.
        auto run = [&](const std::vector<SceneProbe>& probes, int mode, auto lookup_of, const std::string& what, int bits) {
            std::vector<Lookup> lk;
            for (const SceneProbe& p : probes) lk.push_back(lookup_of(p));
            const std::vector<Probe> got = probe(mode, lk);
            const double tol = 1e-4 + (bits >= 32 ? 0.0 : 1.0 / (std::ldexp(1.0, bits) - 1.0));
            int bad = 0, sky = 0;
            double max_err = 0.0;
            std::string first;
            for (std::size_t k = 0; k < probes.size(); ++k)
            {
                const double s = bits_to_float(got[k][0]);
                const bool is_sky = probes[k].id == 0;
                sky += is_sky ? 1 : 0;
                const bool ok = is_sky ? s == 1.0 : std::abs(s - probes[k].depth) <= tol;
                if (!is_sky) max_err = std::max(max_err, std::abs(s - probes[k].depth));
                if (!ok)
                {
                    ++bad;
                    if (first.empty())
                        first = std::format("texel ({},{}) id {}: got {:.9g} expected {:.9g}", probes[k].i, probes[k].j, probes[k].id, s, probes[k].depth);
                }
            }
            check.expect(bad == 0 && !probes.empty(),
                         std::format("{}: {} texels match the CPU ray-cast depth ({} cleared to exactly 1.0)", what, probes.size(), sky),
                         std::format("{} mismatches; {}", bad, first));
            std::cout << std::format("   [info] {}: max |depth - CPU| = {:.3g}\n", what, max_err);
        };

        for (std::size_t k = 0; k < 5; ++k)
        {
            bind_unit(kUnitRaw2D, TT::Texture2D, scene_tex_[k], raw_sampler_);
            run(probes_a_, kModeRaw2D,
                [](const SceneProbe& p) { return Lookup{(p.i + 0.5f) / kSceneW, (p.j + 0.5f) / kSceneH, 0.0f, 0.0f}; },
                std::format("scene in {}", kFormats[k].name), kFormats[k].is_float ? 32 : depth_bits_[k]);
        }
        bind_unit(kUnitRawArray, TT::Texture2DArray, array_tex_, raw_sampler_);
        run(probes_a_, kModeRawArray, [](const SceneProbe& p) { return Lookup{(p.i + 0.5f) / kSceneW, (p.j + 0.5f) / kSceneH, 0.0f, 0.0f}; },
            "2D array layer 0 (camera A, FramebufferTextureLayer)", 32);
        run(probes_b_, kModeRawArray, [](const SceneProbe& p) { return Lookup{(p.i + 0.5f) / kSceneW, (p.j + 0.5f) / kSceneH, 1.0f, 0.0f}; },
            "2D array layer 1 (camera B, FramebufferTextureLayer)", 32);
        bind_unit(kUnitRawCube, TT::TextureCubeMap, cube_tex_, raw_sampler_);
        for (int face = 0; face < 6; ++face)
        {
            const Camera cam = cube_camera(face);
            run(probes_cube_[static_cast<std::size_t>(face)], kModeRawCube,
                [&](const SceneProbe& p) {
                    const V3 d = cam.texel_dir(p.i, p.j, kCubeSize, kCubeSize);
                    return Lookup{static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z), 0.0f};
                },
                std::format("cube face {} (rendered face texel == texel selected by its direction, Table 3.21)", face), 32);
        }
    }

    // -- comparison on 2D textures -----------------------------------------------------------------

    void DepthTextures::check_compare_2d(demo::Check& check)
    {
        std::vector<Lookup> raw_lk;
        for (int t = 0; t < kUpN; ++t) raw_lk.push_back({(t % kUpW + 0.5f) / kUpW, (t / kUpW + 0.5f) / kUpH, 0.0f, 0.0f});
        bool func_readback = true;
        for (int u = 0; u < kUploadCount; ++u)
        {
            const UploadInfo& info = kUploads[static_cast<std::size_t>(u)];
            const DepthFormat& fmt = kFormats[static_cast<std::size_t>(info.format)];
            const mgl::TextureId tex = up_tex_[static_cast<std::size_t>(u)];
            bind_unit(kUnitRaw2D, TT::Texture2D, tex, raw_sampler_);
            const std::vector<Probe> raw = probe(kModeRaw2D, raw_lk);

            // Lookups: Dref below / equal to / above each texel's own value. 0 and 1 get +-0.5 so that the
            // reference leaves [0, 1]: clamped for fixed-point formats, not for floating-point (§3.8.15.1).
            struct Item
            {
                Lookup lk;
                double ref_c, dt;
                bool clamp_case;
            };
            std::vector<Item> items;
            for (int t = 0; t < kUpN; ++t)
            {
                const float dt = bits_to_float(raw[static_cast<std::size_t>(t)][0]);
                const bool edge = dt == 0.0f || dt == 1.0f;
                const float delta = edge ? 0.5f : 0.02f;
                for (float ref : {dt - delta, dt, dt + delta})
                {
                    const double ref_c = fmt.is_float ? static_cast<double>(ref) : std::clamp(static_cast<double>(ref), 0.0, 1.0);
                    // Dref == Dt is only well defined where Dt is exactly representable as the float the
                    // shader passes: every value of a float format, 0 and 1 of a fixed-point format.
                    if (ref == dt && !fmt.is_float && !edge) continue;
                    items.push_back({{raw_lk[static_cast<std::size_t>(t)].x, raw_lk[static_cast<std::size_t>(t)].y, 0.0f, ref}, ref_c, dt, ref < 0.0f || ref > 1.0f});
                }
            }
            std::vector<Lookup> lk;
            for (const Item& it : items) lk.push_back(it.lk);

            int clamp_bad = 0, clamp_total = 0;
            std::string clamp_first;
            bind_unit(kUnitShadow2D, TT::Texture2D, tex, mgl::SamplerId{0});  // texture state decides
            mgl::glTexParameter(TT::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
            for (std::size_t f = 0; f < 8; ++f)
            {
                mgl::glActiveTexture(kUnits[kUnitShadow2D]);
                mgl::glTexParameter(TT::Texture2D, kFuncs[f].func);
                GLint readback = -1;
                mgl::glGetTexParameteriv(TT::Texture2D, mgl::TextureParameterQuery::CompareFunc, &readback);
                func_readback = func_readback && readback == static_cast<GLint>(kFuncs[f].func);
                const std::vector<Probe> got = probe(kModeShadow2D, lk);
                int bad = 0, total = 0;
                std::string first;
                for (std::size_t k = 0; k < items.size(); ++k)
                {
                    const bool pass = table_3_23(kFuncs[f].func, items[k].ref_c, items[k].dt);
                    const bool ok = got[k][0] == float_to_bits(pass ? 1.0f : 0.0f);
                    if (items[k].clamp_case)
                    {
                        ++clamp_total;
                        if (!ok)
                        {
                            ++clamp_bad;
                            if (clamp_first.empty())
                                clamp_first = std::format("{} Dref {} vs Dt {}: got {} expected {}", kFuncs[f].name, items[k].lk.ref, items[k].dt,
                                                          bits_to_float(got[k][0]), pass ? 1 : 0);
                        }
                        continue;
                    }
                    ++total;
                    if (!ok)
                    {
                        ++bad;
                        if (first.empty())
                            first = std::format("Dref {:.9g} vs Dt {:.9g}: got {} expected {}", items[k].lk.ref, items[k].dt, bits_to_float(got[k][0]), pass ? 1 : 0);
                    }
                }
                check.expect(bad == 0, std::format("sampler2DShadow {} on {}: {} lookups follow Table 3.23 exactly", kFuncs[f].name, info.label, total),
                             std::format("{} mismatches; {}", bad, first));
            }
            mgl::glTexParameter(TT::Texture2D, mgl::TextureCompareMode::None);
            mgl::glTexParameter(TT::Texture2D, CF::Lequal);
            const std::string clamp_what = std::format("{}: Dref outside [0, 1] is {} before the comparison (§3.8.15.1), {} lookups", info.label,
                                                       fmt.is_float ? "NOT clamped" : "clamped", clamp_total);
            const std::string clamp_detail = std::format("{} mismatches; {}", clamp_bad, clamp_first);
            if (fmt.is_float || fmt.min_bits != 24)  // the deviation is limited to the 24-bit fixed-point formats
                check.expect(clamp_bad == 0, clamp_what, clamp_detail);
            else
                check.expect_or_known_deviation(clamp_bad == 0, clamp_what, clamp_detail, "ANGLE",
                                                "ANGLE (Vulkan backend) does not clamp Dref for DEPTH_COMPONENT24 and DEPTH24_STENCIL8 (DEPTH_COMPONENT16 "
                                                "is clamped): \"If the texture's internal format indicates a fixed-point depth texture, then Dt "
                                                "and Dref are clamped to the range [0, 1]\" (ES 3.0 §3.8.15.1, ES 3.2 §8.20.1)");
        }
        check.expect(func_readback, "glGetTexParameteriv(TEXTURE_COMPARE_FUNC) reads back each of the eight functions");
    }

    void DepthTextures::check_compare_array_cube(demo::Check& check)
    {
        // --- 2D array, DEPTH_COMPONENT32F 4x2x2 uploaded with TexImage3D FLOAT; comparison state from
        // sampler objects (the texture's own compare mode stays NONE: the sampler supersedes it, §3.8.2).
        constexpr std::array<float, 8> layer0 = {0.05f, 0.15f, 0.3f, 0.35f, 0.6f, 0.7f, 0.85f, 0.95f};
        std::vector<float> data;
        for (float v : layer0) data.push_back(v);
        for (float v : layer0) data.push_back(1.0f - v);
        mgl::TextureId arr{};
        mgl::glGenTextures(1, &arr);
        mgl::glBindTexture(TT::Texture2DArray, arr);
        mgl::glTexImage3D(TT::Texture2DArray, 0, IF::DepthComponent32F, 4, 2, 2, 0, PF::DepthComponent, PT::Float, data.data());
        set_filters(TT::Texture2DArray, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest);
        mgl::glBindTexture(TT::Texture2DArray, mgl::TextureId{0});
        check.expect_no_gl_error("TexImage3D DEPTH_COMPONENT32F 2D array upload");

        std::vector<Lookup> raw_lk;
        for (int layer = 0; layer < 2; ++layer)
            for (int t = 0; t < 8; ++t) raw_lk.push_back({(t % 4 + 0.5f) / 4.0f, (t / 4 + 0.5f) / 2.0f, float(layer), 0.0f});
        bind_unit(kUnitRawArray, TT::Texture2DArray, arr, raw_sampler_);
        const std::vector<Probe> raw = probe(kModeRawArray, raw_lk);
        bool raw_ok = true;
        for (std::size_t k = 0; k < 16; ++k) raw_ok = raw_ok && raw[k][0] == float_to_bits(data[k]);
        check.expect(raw_ok, "sampler2DArray reads both layers of the DEPTH_COMPONENT32F array exactly (layer = P.z)");

        std::vector<Lookup> lk;
        std::vector<std::array<float, 2>> ref_dt;
        for (std::size_t k = 0; k < 16; ++k)
            for (float ref : {data[k] - 0.02f, data[k], data[k] + 0.02f})
            {
                lk.push_back({raw_lk[k].x, raw_lk[k].y, raw_lk[k].z, ref});
                ref_dt.push_back({ref, data[k]});
            }
        for (std::size_t f = 0; f < 8; ++f)
        {
            bind_unit(kUnitShadowArray, TT::Texture2DArray, arr, cmp_sampler_[f]);
            const std::vector<Probe> got = probe(kModeShadowArray, lk);
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < lk.size(); ++k)
            {
                const bool pass = table_3_23(kFuncs[f].func, ref_dt[k][0], ref_dt[k][1]);
                if (got[k][0] != float_to_bits(pass ? 1.0f : 0.0f))
                {
                    ++bad;
                    if (first.empty()) first = std::format("layer {} Dref {:.9g} vs Dt {:.9g}: got {}", lk[k].z, ref_dt[k][0], ref_dt[k][1], bits_to_float(got[k][0]));
                }
            }
            check.expect(bad == 0, std::format("sampler2DArrayShadow {} (sampler object): 48 lookups on 2 layers follow Table 3.23", kFuncs[f].name),
                         std::format("{} mismatches; {}", bad, first));
        }
        bind_unit(kUnitShadowArray, TT::Texture2DArray, mgl::TextureId{0}, mgl::SamplerId{0});
        bind_unit(kUnitRawArray, TT::Texture2DArray, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glDeleteTextures(1, &arr);

        // --- cube map, DEPTH_COMPONENT32F 4x4 faces with one value per face; comparison state from the
        // texture parameters (no sampler object on that unit).
        constexpr std::array<float, 6> face_value = {0.2f, 0.3f, 0.45f, 0.55f, 0.7f, 0.85f};
        mgl::TextureId cube{};
        mgl::glGenTextures(1, &cube);
        mgl::glBindTexture(TT::TextureCubeMap, cube);
        for (std::size_t face = 0; face < 6; ++face)
        {
            const std::vector<float> texels(16, face_value[face]);
            mgl::glTexImage2D(kCubeFaces[face], 0, IF::DepthComponent32F, 4, 4, 0, PF::DepthComponent, PT::Float, texels.data());
        }
        set_filters(TT::TextureCubeMap, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest);
        mgl::glTexParameter(TT::TextureCubeMap, mgl::TextureCompareMode::CompareRefToTexture);
        check.expect_no_gl_error("TexImage2D DEPTH_COMPONENT32F on the six cube faces");
        const std::array<std::array<float, 3>, 12> dirs = {{
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
            {1, 0.3f, -0.4f}, {-1, -0.5f, 0.2f}, {0.25f, 1, -0.6f}, {-0.7f, -1, 0.1f}, {0.45f, -0.2f, 1}, {-0.3f, 0.6f, -1},
        }};
        std::vector<Lookup> cube_raw_lk, cube_lk;
        std::vector<std::array<float, 2>> cube_ref_dt;
        for (std::size_t k = 0; k < dirs.size(); ++k)
        {
            const float dt = face_value[k % 6];
            cube_raw_lk.push_back({dirs[k][0], dirs[k][1], dirs[k][2], 0.0f});
            for (float ref : {dt - 0.02f, dt, dt + 0.02f})
            {
                cube_lk.push_back({dirs[k][0], dirs[k][1], dirs[k][2], ref});
                cube_ref_dt.push_back({ref, dt});
            }
        }
        bind_unit(kUnitRawCube, TT::TextureCubeMap, cube, raw_sampler_);
        const std::vector<Probe> cube_raw = probe(kModeRawCube, cube_raw_lk);
        bool cube_raw_ok = true;
        for (std::size_t k = 0; k < dirs.size(); ++k) cube_raw_ok = cube_raw_ok && cube_raw[k][0] == float_to_bits(face_value[k % 6]);
        check.expect(cube_raw_ok, "samplerCube (raw sampler object) selects the face of the major axis for 12 directions (Table 3.21)");

        bind_unit(kUnitShadowCube, TT::TextureCubeMap, cube, mgl::SamplerId{0});
        for (std::size_t f = 0; f < 8; ++f)
        {
            mgl::glActiveTexture(kUnits[kUnitShadowCube]);
            mgl::glTexParameter(TT::TextureCubeMap, kFuncs[f].func);
            const std::vector<Probe> got = probe(kModeShadowCube, cube_lk);
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < cube_lk.size(); ++k)
            {
                const bool pass = table_3_23(kFuncs[f].func, cube_ref_dt[k][0], cube_ref_dt[k][1]);
                if (got[k][0] != float_to_bits(pass ? 1.0f : 0.0f))
                {
                    ++bad;
                    if (first.empty()) first = std::format("lookup {} Dref {:.9g} vs Dt {:.9g}: got {}", k, cube_ref_dt[k][0], cube_ref_dt[k][1], bits_to_float(got[k][0]));
                }
            }
            check.expect(bad == 0, std::format("samplerCubeShadow {} (texture state): 36 lookups follow Table 3.23", kFuncs[f].name),
                         std::format("{} mismatches; {}", bad, first));
        }
        bind_unit(kUnitShadowCube, TT::TextureCubeMap, mgl::TextureId{0}, mgl::SamplerId{0});
        bind_unit(kUnitRawCube, TT::TextureCubeMap, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glDeleteTextures(1, &cube);
    }

    // -- LINEAR filtering of comparisons -------------------------------------------------------------

    void DepthTextures::check_linear(demo::Check& check)
    {
        // 4x4 textures: columns 0, 1 hold 0.25 and columns 2, 3 hold 0.75.
        std::array<float, 16> f32{};
        std::array<std::uint16_t, 16> u16{};
        for (std::size_t t = 0; t < 16; ++t)
        {
            const bool right = (t % 4) >= 2;
            f32[t] = right ? 0.75f : 0.25f;
            u16[t] = right ? 49151 : 16384;
        }
        mgl::TextureId tex[2]{};
        mgl::glGenTextures(2, tex);
        mgl::glBindTexture(TT::Texture2D, tex[0]);
        mgl::glTexImage2D(TT::Texture2D, 0, IF::DepthComponent32F, 4, 4, 0, PF::DepthComponent, PT::Float, f32.data());
        mgl::glBindTexture(TT::Texture2D, tex[1]);
        mgl::glTexImage2D(TT::Texture2D, 0, IF::DepthComponent16, 4, 4, 0, PF::DepthComponent, PT::UnsignedShort, u16.data());
        for (mgl::TextureId t : tex)
        {
            mgl::glBindTexture(TT::Texture2D, t);
            mgl::glTexParameter(TT::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(TT::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(TT::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
            mgl::glTexParameter(TT::Texture2D, CF::Lequal);
        }
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});

        const std::array<float, 11> us = {0.06f, 0.125f, 0.3f, 0.4f, 0.45f, 0.5f, 0.55f, 0.6f, 0.7f, 0.875f, 0.94f};
        const std::array<float, 3> refs = {0.1f, 0.5f, 0.9f};
        std::vector<Lookup> lk;
        for (float ref : refs)
            for (float u : us) lk.push_back({u, 0.375f, 0.0f, ref});

        const char* names[2] = {"DEPTH_COMPONENT32F", "DEPTH_COMPONENT16"};
        for (int k = 0; k < 2; ++k)
        {
            // LINEAR + COMPARE_REF_TO_TEXTURE is complete (§3.8.13 only restricts compare mode NONE).
            bind_unit(kUnitShadow2D, TT::Texture2D, tex[k], mgl::SamplerId{0});
            set_filters(TT::Texture2D, mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Linear);
            const std::vector<Probe> lin = probe(kModeShadow2D, lk);
            set_filters(TT::Texture2D, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest);
            const std::vector<Probe> nearest = probe(kModeShadow2D, lk);

            // Dref 0.1 passes LEQUAL against every texel, 0.9 against none: whatever texels the implementation
            // combines, r is "proportional to the number of comparison passes" (§3.8.15.1) and therefore 1 / 0.
            int bad_all = 0, out_of_range = 0, bad_nearest = 0;
            std::string mixed;
            for (std::size_t r = 0; r < refs.size(); ++r)
                for (std::size_t i = 0; i < us.size(); ++i)
                {
                    const std::size_t idx = r * us.size() + i;
                    const float v = bits_to_float(lin[idx][0]);
                    if (refs[r] == 0.1f && v != 1.0f) ++bad_all;
                    if (refs[r] == 0.9f && v != 0.0f) ++bad_all;
                    if (!(v >= 0.0f && v <= 1.0f)) ++out_of_range;
                    if (refs[r] == 0.5f) mixed += std::format("u={:.3f}:{:.3f} ", us[i], v);
                    // NEAREST picks column floor(4u); at u == 0.5 (texel boundary) either neighbour is accepted.
                    const float n = bits_to_float(nearest[idx][0]);
                    const double dt = (std::floor(4.0 * static_cast<double>(us[i])) >= 2.0) ? 0.75 : 0.25;
                    const float want = static_cast<double>(refs[r]) <= dt ? 1.0f : 0.0f;
                    const float want_left = refs[r] <= 0.25f ? 1.0f : 0.0f;  // column 1 at the u == 0.5 boundary
                    const bool ok = us[i] == 0.5f ? (n == want || n == want_left) : n == want;
                    if (!ok) ++bad_nearest;
                }
            check.expect(bad_all == 0, std::format("{} LINEAR shadow lookups: all-pass (Dref 0.1) give 1.0 and all-fail (Dref 0.9) give 0.0 at 11 positions", names[k]),
                         std::format("{} mismatches", bad_all));
            check.expect(out_of_range == 0, std::format("{} LINEAR shadow lookups stay in [0, 1]", names[k]));
            check.expect(bad_nearest == 0, std::format("{} NEAREST shadow lookups are exactly 0 or 1 as the selected texel predicts", names[k]),
                         std::format("{} mismatches", bad_nearest));
            std::cout << std::format("   [info] {} LINEAR LEQUAL, Dref 0.5 across the 0.25 | 0.75 edge (implementation-dependent weights): {}\n", names[k], mixed);
        }
        bind_unit(kUnitShadow2D, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glDeleteTextures(2, tex);
    }

    // -- completeness, GenerateMipmap ------------------------------------------------------------

    void DepthTextures::check_completeness(demo::Check& check)
    {
        // Immutable single-level textures holding 0.75: DEPTH_COMPONENT32F and DEPTH_COMPONENT24.
        mgl::TextureId imm[2]{};
        mgl::glGenTextures(2, imm);
        const std::array<float, 4> f075 = {0.75f, 0.75f, 0.75f, 0.75f};
        const std::array<std::uint32_t, 4> u075 = {0xC0000000u, 0xC0000000u, 0xC0000000u, 0xC0000000u};
        mgl::glBindTexture(TT::Texture2D, imm[0]);
        mgl::glTexStorage2D(TT::Texture2D, 1, SF::DepthComponent32F, 2, 2);
        mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, PF::DepthComponent, PT::Float, f075.data());
        mgl::glBindTexture(TT::Texture2D, imm[1]);
        mgl::glTexStorage2D(TT::Texture2D, 1, SF::DepthComponent24, 2, 2);
        mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, PF::DepthComponent, PT::UnsignedInt, u075.data());
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});

        const Lookup one[2] = {{0.25f, 0.25f, 0.0f, 0.5f}, {0.75f, 0.75f, 0.0f, 0.5f}};
        struct Case
        {
            mgl::TextureMinFilter min;
            mgl::TextureMagFilter mag;
            bool complete;
            const char* what;
        };
        const Case cases[] = {
            {mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest, true, "NEAREST / NEAREST"},
            {mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Linear, false, "min NEAREST, mag LINEAR"},
            {mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Nearest, false, "min LINEAR, mag NEAREST"},
            {mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Linear, false, "LINEAR / LINEAR"},
            {mgl::TextureMinFilter::NearestMipmapNearest, mgl::TextureMagFilter::Nearest, true, "min NEAREST_MIPMAP_NEAREST (1 immutable level)"},
            {mgl::TextureMinFilter::NearestMipmapLinear, mgl::TextureMagFilter::Nearest, false, "min NEAREST_MIPMAP_LINEAR"},
            {mgl::TextureMinFilter::LinearMipmapNearest, mgl::TextureMagFilter::Nearest, false, "min LINEAR_MIPMAP_NEAREST"},
        };
        const char* names[2] = {"DEPTH_COMPONENT32F", "DEPTH_COMPONENT24"};
        for (int k = 0; k < 2; ++k)
        {
            bind_unit(kUnitRaw2D, TT::Texture2D, imm[k], mgl::SamplerId{0});  // texture state decides
            for (const Case& c : cases)
            {
                set_filters(TT::Texture2D, c.min, c.mag);
                const std::vector<Probe> got = probe(kModeRaw2D, one);
                bool ok = true;
                std::string detail;
                for (const Probe& p : got)
                {
                    const double r = bits_to_float(p[0]);
                    const bool r_ok = c.complete ? std::abs(r - 0.75) <= 1e-6 : p[0] == 0u;
                    ok = ok && r_ok && p[1] == 0u && p[2] == 0u && p[3] == float_to_bits(1.0f);
                    detail += std::format("({:.6g},{},{},{}) ", r, bits_to_float(p[1]), bits_to_float(p[2]), bits_to_float(p[3]));
                }
                const std::string what =
                    std::format("{} compare NONE, {}: {} -> sampler2D returns {}", names[k], c.what, c.complete ? "complete" : "incomplete (§3.8.13)",
                                c.complete ? "(0.75, 0, 0, 1)" : "(0, 0, 0, 1) (§3.9.2.1)");
                if (c.complete)
                    check.expect(ok, what, detail);
                else
                    // ES 3.0 §3.8.13 (last bullet, added for Bug 9481): a sized depth format with compare mode
                    // NONE is incomplete unless mag == NEAREST and min is NEAREST or NEAREST_MIPMAP_NEAREST.
                    // No exposed extension relaxes it (OES_depth_texture covers only unsized DEPTH_COMPONENT
                    // and says nothing about filters). Mesa filters such textures instead of treating them as
                    // incomplete; the strict check stays for every other implementation.
                    check.expect_or_known_deviation(ok, what, detail, "Mesa",
                                                    "Mesa does not apply the ES 3.0 depth-texture completeness rule for compare mode NONE "
                                                    "with non-NEAREST filters; it returns the (filtered) depth instead of (0, 0, 0, 1)");
            }
            // With comparison enabled LINEAR is allowed: ALWAYS gives 1, NEVER gives 0.
            set_filters(TT::Texture2D, mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Linear);
            mgl::glTexParameter(TT::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
            bind_unit(kUnitRaw2D, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
            bind_unit(kUnitShadow2D, TT::Texture2D, imm[k], mgl::SamplerId{0});
            mgl::glTexParameter(TT::Texture2D, CF::Always);
            const float always = bits_to_float(probe(kModeShadow2D, one)[0][0]);
            mgl::glTexParameter(TT::Texture2D, CF::Never);
            const float never = bits_to_float(probe(kModeShadow2D, one)[0][0]);
            check.expect(always == 1.0f && never == 0.0f,
                         std::format("{} LINEAR / LINEAR with COMPARE_REF_TO_TEXTURE is complete: ALWAYS = 1, NEVER = 0", names[k]),
                         std::format("ALWAYS {} NEVER {}", always, never));
            bind_unit(kUnitShadow2D, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
        }
        mgl::glDeleteTextures(2, imm);

        // Mutable texture with only level 0: NEAREST_MIPMAP_NEAREST needs levels 1.. (max level 1000) -> incomplete.
        mgl::TextureId mut{};
        mgl::glGenTextures(1, &mut);
        mgl::glBindTexture(TT::Texture2D, mut);
        mgl::glTexImage2D(TT::Texture2D, 0, IF::DepthComponent32F, 2, 2, 0, PF::DepthComponent, PT::Float, f075.data());
        mgl::glTexParameter(TT::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
        mgl::glTexParameter(TT::Texture2D, CF::Always);
        bind_unit(kUnitShadow2D, TT::Texture2D, mut, mgl::SamplerId{0});
        set_filters(TT::Texture2D, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest);
        const float complete_always = bits_to_float(probe(kModeShadow2D, one)[0][0]);
        set_filters(TT::Texture2D, mgl::TextureMinFilter::NearestMipmapNearest, mgl::TextureMagFilter::Nearest);
        const float incomplete_always = bits_to_float(probe(kModeShadow2D, one)[0][0]);
        check.expect(complete_always == 1.0f, "control: complete DEPTH_COMPONENT32F with compare func ALWAYS returns 1.0",
                     std::format("got {}", complete_always));
        // ES 3.0 §3.9.2.1: "(0, 0, 0, 1) will be returned for a non-shadow sampler and 0 for a shadow sampler".
        const char* zink_reason = "zink returns 1.0 (the ALWAYS result) for a shadow lookup on a mipmap-incomplete depth texture instead of 0";
        check.expect_or_known_deviation(incomplete_always == 0.0f,
                                        "mipmap-incomplete DEPTH_COMPONENT32F (level 0 only, NEAREST_MIPMAP_NEAREST): shadow lookup returns 0 even with ALWAYS (§3.9.2.1)",
                                        std::format("got {}", incomplete_always), "zink", zink_reason);

        // GenerateMipmap: depth formats are neither color-renderable nor texture-filterable (§3.8.10.5).
        for (int u : {kUp16Short, kUp24Int, kUp32Float, kUp24S8, kUp32FS8})
        {
            mgl::glBindTexture(TT::Texture2D, up_tex_[static_cast<std::size_t>(u)]);
            mgl::glGenerateMipmap(TT::Texture2D);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  std::format("GenerateMipmap on {} -> INVALID_OPERATION", kFormats[static_cast<std::size_t>(kUploads[static_cast<std::size_t>(u)].format)].name));
        }
        // ... and it really generated nothing: the mutable texture stays mipmap-incomplete.
        mgl::glActiveTexture(kUnits[kUnitShadow2D]);
        mgl::glGenerateMipmap(TT::Texture2D);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "GenerateMipmap on the level-0-only DEPTH_COMPONENT32F texture -> INVALID_OPERATION");
        const float after = bits_to_float(probe(kModeShadow2D, one)[0][0]);
        check.expect_or_known_deviation(after == 0.0f, "after the rejected GenerateMipmap the texture is still mipmap-incomplete (shadow lookup 0)",
                                        std::format("got {}", after), "zink", zink_reason);
        bind_unit(kUnitShadow2D, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glDeleteTextures(1, &mut);

        // Positive control: GenerateMipmap works on RGBA8 (color-renderable and filterable).
        mgl::TextureId rgba{};
        mgl::glGenTextures(1, &rgba);
        mgl::glBindTexture(TT::Texture2D, rgba);
        mgl::glTexImage2D(TT::Texture2D, 0, IF::Rgba8, 4, 4, 0, PF::Rgba, PT::UnsignedByte, nullptr);
        mgl::glGenerateMipmap(TT::Texture2D);
        check.expect_no_gl_error("control: GenerateMipmap on RGBA8 raises no error");
        mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
        mgl::glDeleteTextures(1, &rgba);
    }

    // -- stencil -----------------------------------------------------------------------------------

    void DepthTextures::check_stencil(demo::Check& check)
    {
        // Uploaded stencil values (UNSIGNED_INT_24_8 bits 7..0, FLOAT_32_UNSIGNED_INT_24_8_REV word 1 bits 7..0).
        for (int u : {kUp24S8, kUp32FS8})
        {
            const UploadInfo& info = kUploads[static_cast<std::size_t>(u)];
            const std::vector<int> s = read_stencil(up_tex_[static_cast<std::size_t>(u)], info.format, kUpW, kUpH);
            int bad = 0;
            std::string first;
            for (int t = 0; t < kUpN; ++t)
            {
                const int want = u == kUp24S8 ? stencil_24_8(t) : stencil_32f_8(t);
                if (s[static_cast<std::size_t>(t)] != want)
                {
                    ++bad;
                    if (first.empty()) first = std::format("texel {}: 0x{:02X} expected 0x{:02X}", t, s[static_cast<std::size_t>(t)], want);
                }
            }
            check.expect(bad == 0, std::format("upload {}: the stencil test sees the 16 uploaded stencil values", info.label),
                         std::format("{} mismatches; {}", bad, first));
        }

        // Strips: ClearBufferfi stencil 0xA5 survives the gl_FragDepth draw (stencil test off); then a draw
        // with depth test LESS at 0.55 and StencilOp(KEEP, INCR, REPLACE 0x3C) writes 0x3C where the stored
        // depth is greater than 0.55 and 0xA6 where it is not.
        for (int k : {kF24S8, kF32FS8})
        {
            const auto uk = static_cast<std::size_t>(k);
            fill_frag_depth_strip(k, kClearDepth, kClearStencil);
            const std::vector<int> cleared = read_stencil(q_tex_[uk], k, 16, 1);
            check.expect(std::all_of(cleared.begin(), cleared.end(), [](int v) { return v == kClearStencil; }),
                         std::format("{}: ClearBufferfi(depth {}, stencil 0x{:02X}) sets the stencil of all 16 texels", kFormats[uk].name, kClearDepth,
                                     kClearStencil),
                         std::format("texel 0 = 0x{:02X}", cleared[0]));

            mgl::glBindFramebuffer(FT::Framebuffer, q_fbo_[uk]);
            mgl::glViewport(0, 0, 16, 1);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(CF::Less);
            mgl::glDepthMask(GL_FALSE);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilMask(0xFF);
            mgl::glStencilFunc(CF::Always, 0x3C, 0xFF);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Incr, mgl::StencilOp::Replace);
            mgl::glUseProgram(frag_depth_prog_);
            const std::array<float, 16> half = {0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f};
            mgl::glUniform1fv(u_fd_depth_, 16, half.data());
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glDepthMask(GL_TRUE);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::StencilTest);

            const std::vector<int> s = read_stencil(q_tex_[uk], k, 16, 1);
            int bad = 0, passed = 0;
            std::string got;
            for (int t = 0; t < 16; ++t)
            {
                const double stored = t < 15 ? std::clamp(static_cast<double>(kFragDepths[static_cast<std::size_t>(t)]), 0.0, 1.0) : static_cast<double>(kClearDepth);
                const bool depth_pass = 0.55 < stored;  // LESS: incoming 0.55 < stored
                passed += depth_pass ? 1 : 0;
                const int want = depth_pass ? 0x3C : kClearStencil + 1;
                if (s[static_cast<std::size_t>(t)] != want) ++bad;
                got += std::format("{:02X} ", s[static_cast<std::size_t>(t)]);
            }
            check.expect(bad == 0,
                         std::format("{}: depth test against the texture's depth + StencilOp(KEEP, INCR, REPLACE): {} texels 0x3C (depth pass), {} texels 0xA6 (depth fail)",
                                     kFormats[uk].name, passed, 16 - passed),
                         got);
            mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        }

        // Scene stencil ids written with StencilOp(KEEP, KEEP, REPLACE) under depth test LESS: the id of
        // the visible object, 0 for sky.
        for (int k : {kF24S8, kF32FS8})
        {
            const std::vector<int> s = read_stencil(scene_tex_[static_cast<std::size_t>(k)], k, kSceneW, kSceneH);
            int bad = 0;
            std::string first;
            for (const SceneProbe& p : probes_a_)
            {
                const int v = s[static_cast<std::size_t>(p.j * kSceneW + p.i)];
                if (v != p.id)
                {
                    ++bad;
                    if (first.empty()) first = std::format("texel ({},{}): {} expected {}", p.i, p.j, v, p.id);
                }
            }
            check.expect(bad == 0, std::format("scene in {}: stencil holds the id of the visible object at {} texels", kFormats[static_cast<std::size_t>(k)].name, probes_a_.size()),
                         std::format("{} mismatches; {}", bad, first));
        }

        // Attachment completeness (§4.4.4.1): a depth-only format cannot be a stencil or color attachment,
        // a depth-stencil format alone at STENCIL_ATTACHMENT is fine.
        mgl::FramebufferId fbo{};
        mgl::glGenFramebuffers(1, &fbo);
        mgl::glBindFramebuffer(FT::Framebuffer, fbo);
        mgl::glFramebufferTexture2D(FT::Framebuffer, FA::Stencil, TT::Texture2D, q_tex_[kF24], 0);
        const mgl::FramebufferStatus st1 = mgl::glCheckFramebufferStatus(FT::Framebuffer);
        mgl::glFramebufferTexture2D(FT::Framebuffer, FA::Stencil, TT::Texture2D, mgl::TextureId{0}, 0);
        mgl::glFramebufferTexture2D(FT::Framebuffer, kColor0, TT::Texture2D, q_tex_[kF24], 0);
        const mgl::FramebufferStatus st2 = mgl::glCheckFramebufferStatus(FT::Framebuffer);
        mgl::glFramebufferTexture2D(FT::Framebuffer, kColor0, TT::Texture2D, mgl::TextureId{0}, 0);
        mgl::glFramebufferTexture2D(FT::Framebuffer, FA::Stencil, TT::Texture2D, q_tex_[kF24S8], 0);
        const mgl::FramebufferStatus st3 = mgl::glCheckFramebufferStatus(FT::Framebuffer);
        check.expect(st1 == mgl::FramebufferStatus::IncompleteAttachment, "DEPTH_COMPONENT24 at STENCIL_ATTACHMENT -> FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
                     std::string(demo::framebuffer_status_name(st1)));
        check.expect(st2 == mgl::FramebufferStatus::IncompleteAttachment, "DEPTH_COMPONENT24 at COLOR_ATTACHMENT0 -> FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
                     std::string(demo::framebuffer_status_name(st2)));
        check.expect(st3 == mgl::FramebufferStatus::Complete, "DEPTH24_STENCIL8 alone at STENCIL_ATTACHMENT is complete (stencil-renderable)",
                     std::string(demo::framebuffer_status_name(st3)));

        // DEPTH_STENCIL_ATTACHMENT queries need the same object at both points (§6.1.13).
        mgl::glFramebufferTexture2D(FT::Framebuffer, FA::DepthStencil, TT::Texture2D, q_tex_[kF24S8], 0);
        const int same = fb_param(FA::DepthStencil, FAP::ObjectName);
        mgl::glFramebufferTexture2D(FT::Framebuffer, FA::Depth, TT::Texture2D, q_tex_[kF24], 0);
        check.expect_no_gl_error("re-attaching DEPTH_ATTACHMENT alone");
        GLint dummy = 0;
        mgl::glGetFramebufferAttachmentParameteriv(FT::Framebuffer, FA::DepthStencil, FAP::ObjectName, &dummy);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "DEPTH_STENCIL_ATTACHMENT query with different depth and stencil objects -> INVALID_OPERATION");
        check.expect_eq(same, static_cast<int>(q_tex_[kF24S8].value), "DEPTH_STENCIL_ATTACHMENT query names the texture bound to both points");
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        mgl::glDeleteFramebuffers(1, &fbo);
    }

    // ==========================================================================================
    // shutdown
    // ==========================================================================================

    void DepthTextures::shutdown(demo::Env& /*env*/)
    {
        mgl::glBindFramebuffer(FT::Framebuffer, mgl::FramebufferId{0});
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glUseProgram(mgl::ProgramId{0});
        for (int unit = 0; unit < 6; ++unit) bind_unit(unit, TT::Texture2D, mgl::TextureId{0}, mgl::SamplerId{0});
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);

        mgl::glDeleteFramebuffers(5, scene_fbo_.data());
        mgl::glDeleteFramebuffers(5, vis_fbo_.data());
        mgl::glDeleteFramebuffers(5, tint_fbo_.data());
        mgl::glDeleteFramebuffers(2, array_fbo_.data());
        mgl::glDeleteFramebuffers(6, cube_fbo_.data());
        mgl::glDeleteFramebuffers(5, q_fbo_.data());
        mgl::glDeleteFramebuffers(1, &probe_fbo_);
        mgl::glDeleteTextures(5, scene_tex_.data());
        mgl::glDeleteTextures(5, vis_tex_.data());
        mgl::glDeleteTextures(5, q_tex_.data());
        mgl::glDeleteTextures(kUploadCount, up_tex_.data());
        mgl::glDeleteTextures(1, &array_tex_);
        mgl::glDeleteTextures(1, &cube_tex_);
        mgl::glDeleteTextures(1, &probe_tex_);
        mgl::glDeleteSamplers(1, &raw_sampler_);
        mgl::glDeleteSamplers(1, &linear_sampler_);
        mgl::glDeleteSamplers(8, cmp_sampler_.data());
        mgl::glDeleteBuffers(1, &scene_vbo_);
        mgl::glDeleteVertexArrays(1, &scene_vao_);
        mgl::glDeleteVertexArrays(1, &empty_vao_);
        for (mgl::ProgramId p : {scene_prog_, frag_depth_prog_, probe_prog_, solid_prog_, vis_prog_, show_prog_, grid_prog_}) mgl::glDeleteProgram(p);
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-depth-textures",
        .title = "Depth Textures",
        .min_es = {3, 0},
        .summary = "All five ES 3.0 depth formats as textures: Table 3.2 uploads and errors, render-to-depth with exact "
                   "quantization rules, the eight compare functions through sampler2DShadow / sampler2DArrayShadow / "
                   "samplerCubeShadow, LINEAR comparisons, depth-texture completeness, and the stencil half of "
                   "DEPTH24_STENCIL8 / DEPTH32F_STENCIL8 through the stencil test.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<DepthTextures>(argc, argv, info);
}
