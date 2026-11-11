// es30-glsl-texture-functions
//
// The GLSL ES 3.00 texture lookup functions (§8.8) and derivative functions (§8.9), run on every sampler kind ES 3.0
// has: sampler2D, sampler3D, sampler2DArray, samplerCube, the integer usampler2D / isampler3D / usamplerCube and the
// shadow samplers sampler2DShadow / samplerCubeShadow / sampler2DArrayShadow.
//
// How the numbers are made exact:
//  * Every texel of every level carries its own address. RGBA8 texels hold (32 level + 16, 8 i, 8 j, 255 - 16 a)
//    where a is the slice, layer or cube face; integer textures hold (level, i, j, a) (negated for the signed one);
//    depth textures hold a distinct dyadic depth per (face/layer, level, i, j). A lookup result therefore names the
//    exact level and texel the implementation chose.
//  * All textures use NEAREST_MIPMAP_NEAREST / NEAREST, so a lookup is a single texel of a single level and the
//    result is fully specified once lambda is known (ES 3.0 §3.8.10.1-§3.8.10.4).
//  * Each lookup runs in one 4x4 block of a 128x64 RGBA32UI framebuffer (colour-renderable in core ES 3.0); float
//    results are written as floatBitsToUint, so the CPU reads exactly what the shader got. The vertex-stage forms are
//    captured with transform feedback.
//  * The coordinate is P = u_p + u_ddx f.x + u_ddy f.y (f = position inside the block). All inputs are short dyadic
//    fractions, so P is exact and its implicit derivatives are exactly u_ddx / u_ddy.
//  * Level selection has latitude only through rho (§3.8.10.1: "an implementation may approximate the ideal rho with a
//    function f(x, y)" bounded by max{mu, mv, mw} <= f <= mu + mv + mw) and the two rounding rules of eq. 3.23 (and its
//    footnote). The CPU model evaluates lambda at both ends of that interval and only uses inputs for which every
//    allowed choice selects the same level, at least 0.1 away from a rounding boundary; sample positions stay at least
//    1/32 texel away from texel boundaries. The table is checked against these rules in init().
//
// Validation: every pixel of every block (and every captured vertex) is compared with the CPU model; for every case a
// mutated model (offset dropped, bias ignored, one level off, no projective divide, layer counted in rho, s mirrored)
// must disagree with the measurement, showing that the comparison can fail. Derivatives of functions that are affine
// in window coordinates are compared with their slopes. textureSize is checked per level for all ten samplers.
// Compile-time rules (no bias and no dFdx in a vertex shader, offsets must be constant, no offset or texelFetch on
// cube maps) are checked with negative shaders and positive controls. The picture shows a mip-level visualisation on
// a ground plane and one cell per lookup block; the cells are checked against the same model.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
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
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;
using Wrap = mgl::TextureWrapMode;

namespace
{
    constexpr int kBlock = 4;                 // a lookup case covers kBlock x kBlock pixels
    constexpr int kTargetW = 128;             // RGBA32UI probe target
    constexpr int kTargetH = 64;
    constexpr int kBlocksPerRow = kTargetW / kBlock;
    constexpr int kMaxBlocks = kBlocksPerRow * (kTargetH / kBlock);
    constexpr int kVertexCases = 13;          // must match kCases in vertex_lookup.vert
    constexpr std::uint32_t kSentinel = 0xDEADBEEFu;

    constexpr double kDepthScale = 16384.0;   // depth texel spacing 1 / 16384
    constexpr double kRefHalfGap = 0.25 / kDepthScale;
    constexpr double kTexelMargin = 1.0 / 32.0;   // minimum distance of a sample from a texel boundary (texels)
    constexpr double kLambdaMargin = 0.1;         // minimum distance of lambda from a level rounding boundary

    // ---------------------------------------------------------------------------------------------------------------
    // Textures
    // ---------------------------------------------------------------------------------------------------------------
    enum class Fmt { Rgba8, Rgba32ui, Rgba32i, Rgba8ui, Depth32f };
    enum class Dim { D2, D3, Array, Cube };
    enum Tex : int { T2D, T3D, TArr, TCube, TU2D, TI3D, TUCube, TS2D, TSCube, TSArr, kTexCount };

    struct TexInfo
    {
        const char* glsl;
        int w, h, d;  // base level; d = depth (3D) or layers (array)
        int levels;
        Dim dim;
        Fmt fmt;
        Wrap ws, wt, wr;
    };

    constexpr Wrap R = Wrap::Repeat;
    constexpr Wrap C = Wrap::ClampToEdge;
    const std::array<TexInfo, kTexCount> kTex = {{
        {"sampler2D", 32, 16, 1, 6, Dim::D2, Fmt::Rgba8, R, C, R},
        {"sampler3D", 16, 8, 8, 5, Dim::D3, Fmt::Rgba8, R, R, R},
        {"sampler2DArray", 16, 16, 4, 5, Dim::Array, Fmt::Rgba8, R, C, R},
        {"samplerCube", 16, 16, 1, 5, Dim::Cube, Fmt::Rgba8, C, C, C},
        {"usampler2D", 16, 16, 1, 5, Dim::D2, Fmt::Rgba32ui, R, R, R},
        {"isampler3D", 8, 8, 4, 4, Dim::D3, Fmt::Rgba32i, R, R, R},
        {"usamplerCube", 8, 8, 1, 4, Dim::Cube, Fmt::Rgba8ui, C, C, C},
        {"sampler2DShadow", 16, 16, 1, 5, Dim::D2, Fmt::Depth32f, R, C, R},
        {"samplerCubeShadow", 8, 8, 1, 4, Dim::Cube, Fmt::Depth32f, C, C, C},
        {"sampler2DArrayShadow", 8, 8, 3, 4, Dim::Array, Fmt::Depth32f, R, C, R},
    }};

    struct Dims
    {
        int w, h, d;
    };

    // §3.8.10.4: level i has max(1, w / 2^i) x max(1, h / 2^i), depth halves only for 3D; array layers stay.
    Dims level_dims(const TexInfo& t, int level)
    {
        const int w = std::max(1, t.w >> level);
        const int h = std::max(1, t.h >> level);
        const int d = t.dim == Dim::D3 ? std::max(1, t.d >> level) : (t.dim == Dim::Array ? t.d : 1);
        return {w, h, d};
    }

    /// The address of one texel: level, column, row, and slice / layer / cube face.
    struct Texel
    {
        int level = 0, i = 0, j = 0, a = 0;
        bool valid = true;
        friend bool operator==(const Texel&, const Texel&) = default;
    };

    std::array<std::uint8_t, 4> rgba8_of(const Texel& t)
    {
        return {static_cast<std::uint8_t>(32 * t.level + 16), static_cast<std::uint8_t>(8 * t.i), static_cast<std::uint8_t>(8 * t.j),
                static_cast<std::uint8_t>(255 - 16 * t.a)};
    }
    float depth_of(const Texel& t)  // exact in binary32: an odd multiple of 1/32768
    {
        return static_cast<float>((((t.a * 8 + t.level) * 256) + t.j * 16 + t.i + 0.5) / kDepthScale);
    }
    std::array<std::uint32_t, 4> uint_of(Tex tex, const Texel& t)
    {
        const auto u = [](int v) { return static_cast<std::uint32_t>(v); };
        if (kTex[tex].fmt == Fmt::Rgba32ui) return {u(t.level), u(t.i), u(t.j), 0xC0DE0000u + u(t.a)};
        if (kTex[tex].fmt == Fmt::Rgba32i) return {u(-1 - t.level), u(-1 - t.i), u(-1 - t.j), u(-1 - t.a)};
        return {u(t.level), u(t.i), u(t.j), u(t.a)};  // RGBA8UI
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Cases
    // ---------------------------------------------------------------------------------------------------------------
    struct V4
    {
        double x = 0, y = 0, z = 0, w = 0;
        friend V4 operator+(V4 a, V4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
        friend V4 operator*(V4 a, double s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
    };

    enum class Mode { Implicit, Explicit, Grad, Fetch, Size, Deriv };
    enum class Mut { DropOffset, IgnoreBias, LevelUp, NoDivide, LayerInRho, MirrorS };

    std::string_view mut_name(Mut m)
    {
        switch (m)
        {
        case Mut::DropOffset: return "model without the texel offset";
        case Mut::IgnoreBias: return "model ignoring the bias";
        case Mut::LevelUp: return "model one level off";
        case Mut::NoDivide: return "model without the projective divide";
        case Mut::LayerInRho: return "model counting the array layer in rho";
        case Mut::MirrorS: return "model with s mirrored on the cube face";
        }
        return "?";
    }

    struct Case
    {
        int id = 0;           // shader case number (probe.frag) or gl_VertexID (vertex_lookup.vert)
        std::string fn;       // GLSL call, for messages
        Tex tex = T2D;
        Mode mode = Mode::Explicit;
        bool proj = false;    // coordinates are divided by P.w (P.w constant over the block)
        bool bias = false;    // implicit lookup with the bias argument (u_lod)
        std::array<int, 3> off{};
        V4 p, ddx, ddy, gx, gy;
        double lod = 0;       // explicit lod, or the bias
        std::array<int, 4> ip{};
        std::vector<Mut> controls;
        bool vertex = false;
        bool optional = false;  // dropped (with a note) when the queried limits make it ambiguous
        // filled in init()
        int slot = -1;                                  // block index or transform-feedback vertex
        std::array<std::array<float, 2>, 16> refs{};    // depth references per pixel (shadow samplers)
    };

    // ---------------------------------------------------------------------------------------------------------------
    // CPU model of the lookups (ES 3.0 §3.8.9 cube selection, §3.8.10.1 lambda, §3.8.10.2 wrapping and texel
    // selection, §3.8.10.4 mipmapping, §2.12.9.2 texel fetches, §2.12.9.3 size query, §2.12.9.4 vertex stage).
    // ---------------------------------------------------------------------------------------------------------------
    double frac_margin(double u)
    {
        const double f = u - std::floor(u);
        return std::min(f, 1.0 - f);
    }

    // Eq. 3.23 for NEAREST_MIPMAP_NEAREST with level_base = 0 (lambda <= 0 is magnification: level 0 as well).
    int select_level(double lambda, int q)
    {
        if (lambda <= 0.5) return 0;
        return std::min(q, static_cast<int>(std::ceil(lambda + 0.5)) - 1);
    }
    double level_margin(double lambda, int q)  // distance to the nearest rounding boundary n + 1/2, n < q
    {
        double m = 1e9;
        for (int n = 0; n < q; ++n) m = std::min(m, std::abs(lambda - (n + 0.5)));
        return m;
    }

    int wrap_coord(int c, int size, Wrap mode)  // Table 3.22
    {
        if (mode == Wrap::Repeat) return ((c % size) + size) % size;
        return std::clamp(c, 0, size - 1);
    }

    // Table 3.21: sc = a . dir, tc = b . dir, ma = the major component.
    struct Face
    {
        int index;
        std::array<double, 3> a, b;
        int major;  // 0, 1, 2 = x, y, z
    };
    Face cube_face(double x, double y, double z)
    {
        const double ax = std::abs(x), ay = std::abs(y), az = std::abs(z);
        if (ax >= ay && ax >= az) return x > 0 ? Face{0, {0, 0, -1}, {0, -1, 0}, 0} : Face{1, {0, 0, 1}, {0, -1, 0}, 0};
        if (ay >= az) return y > 0 ? Face{2, {1, 0, 0}, {0, 0, 1}, 1} : Face{3, {1, 0, 0}, {0, 0, -1}, 1};
        return z > 0 ? Face{4, {1, 0, 0}, {0, -1, 0}, 2} : Face{5, {-1, 0, 0}, {0, -1, 0}, 2};
    }
    // The direction whose projection on `face` is (s, t), with |ma| = 1; and the direction in which sc grows.
    V4 cube_dir(int face, double s, double t)
    {
        const double sc = 2 * s - 1, tc = 2 * t - 1;
        switch (face)
        {
        case 0: return {1, -tc, -sc, 0};
        case 1: return {-1, -tc, sc, 0};
        case 2: return {sc, 1, tc, 0};
        case 3: return {sc, -1, -tc, 0};
        case 4: return {sc, -tc, 1, 0};
        default: return {-sc, -tc, -1, 0};
        }
    }
    V4 cube_sc_axis(int face)
    {
        static const std::array<V4, 6> axes = {{{0, 0, -1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {-1, 0, 0, 0}}};
        return axes[static_cast<std::size_t>(face)];
    }
    V4 cube_tc_axis(int face)
    {
        static const std::array<V4, 6> axes = {{{0, -1, 0, 0}, {0, -1, 0, 0}, {0, 0, 1, 0}, {0, 0, -1, 0}, {0, -1, 0, 0}, {0, -1, 0, 0}}};
        return axes[static_cast<std::size_t>(face)];
    }

    struct Eval
    {
        Texel t;
        std::string problem;  // non-empty: the case is not robustly specified at this pixel
        double lambda_lo = 0, lambda_hi = 0;
    };

    Eval model(const Case& c, int px, int py, std::optional<Mut> mut, double bias_max)
    {
        Eval e;
        const TexInfo& T = kTex[c.tex];
        const int q = T.levels - 1;
        const auto has = [&](Mut m) { return mut && *mut == m; };
        const std::array<int, 3> off = has(Mut::DropOffset) ? std::array<int, 3>{} : c.off;
        auto problem = [&](std::string s) {
            if (e.problem.empty()) e.problem = std::move(s);
        };

        if (c.mode == Mode::Fetch)
        {
            // §2.12.9.2: integer coordinates (i, j, k) of level base + lod; no wrapping, no LOD clamping.
            int level = c.ip[3];
            if (has(Mut::LevelUp)) level = level < q ? level + 1 : level - 1;
            const Dims D = level_dims(T, level);
            const int i = c.ip[0] + px + off[0];
            const int j = c.ip[1] + py + off[1];
            const int k = c.ip[2] + (T.dim == Dim::D3 ? off[2] : 0);
            e.t = {level, i, j, k};
            if (i < 0 || i >= D.w || j < 0 || j >= D.h || k < 0 || k >= D.d)
            {
                e.t.valid = false;
                problem(std::format("texelFetch ({}, {}, {}) is outside level {} ({}x{}x{})", i, j, k, level, D.w, D.h, D.d));
            }
            return e;
        }

        const double fx = c.vertex ? 0.0 : px + 0.5;
        const double fy = c.vertex ? 0.0 : py + 0.5;
        const V4 P = c.vertex ? c.p : c.p + c.ddx * fx + c.ddy * fy;
        const double qd = (c.proj && !has(Mut::NoDivide)) ? P.w : 1.0;
        if (c.proj && (c.ddx.w != 0 || c.ddy.w != 0)) problem("q must be constant over the block");

        // Projected coordinate and its derivatives per window pixel (textureProjGrad: "already projected").
        double s = 0, t = 0, r = 0, layer = 0;
        std::array<double, 3> dx{}, dy{};  // d(s, t, r)/dx, d(s, t, r)/dy
        const V4 Dx = c.mode == Mode::Grad ? c.gx : c.ddx * (1.0 / qd);
        const V4 Dy = c.mode == Mode::Grad ? c.gy : c.ddy * (1.0 / qd);
        int face = 0;
        if (T.dim == Dim::Cube)
        {
            const Face f = cube_face(P.x, P.y, P.z);
            const std::array<double, 3> dir = {P.x, P.y, P.z};
            const double ma = std::abs(dir[static_cast<std::size_t>(f.major)]);
            for (std::size_t n = 0; n < 3; ++n)
                if (static_cast<int>(n) != f.major && ma - std::abs(dir[n]) < 0.05) problem("cube direction close to a face edge");
            const auto dot3 = [](const std::array<double, 3>& a, double x, double y, double z) { return a[0] * x + a[1] * y + a[2] * z; };
            const double sc = dot3(f.a, P.x, P.y, P.z), tc = dot3(f.b, P.x, P.y, P.z);
            s = 0.5 * (sc / ma + 1.0);
            t = 0.5 * (tc / ma + 1.0);
            const std::array<double, 3> dxv = {Dx.x, Dx.y, Dx.z}, dyv = {Dy.x, Dy.y, Dy.z};
            if (dxv[static_cast<std::size_t>(f.major)] != 0 || dyv[static_cast<std::size_t>(f.major)] != 0)
                problem("the derivative along the major axis must be zero for an exact model");
            // Chain rule with d(ma) = 0: ds = dsc / (2 |ma|).
            dx = {0.5 * dot3(f.a, Dx.x, Dx.y, Dx.z) / ma, 0.5 * dot3(f.b, Dx.x, Dx.y, Dx.z) / ma, 0};
            dy = {0.5 * dot3(f.a, Dy.x, Dy.y, Dy.z) / ma, 0.5 * dot3(f.b, Dy.x, Dy.y, Dy.z) / ma, 0};
            face = f.index;
            if (has(Mut::MirrorS)) s = 1.0 - s;
        }
        else
        {
            s = P.x / qd;
            t = P.y / qd;
            if (T.dim == Dim::D3) r = P.z / qd;
            if (T.dim == Dim::Array) layer = P.z;
            dx = {Dx.x, Dx.y, T.dim == Dim::D3 || (T.dim == Dim::Array && has(Mut::LayerInRho)) ? Dx.z : 0.0};
            dy = {Dy.x, Dy.y, T.dim == Dim::D3 || (T.dim == Dim::Array && has(Mut::LayerInRho)) ? Dy.z : 0.0};
        }

        // Lambda (§3.8.10.1). Equation 3.19 scales by the base level size; for an array the (wrong) mutation uses the layer count.
        double lo = 0, hi = 0;
        if (c.mode == Mode::Explicit)
            lo = hi = c.lod;
        else if (c.vertex)
            lo = hi = 0.0;  // §2.12.9.4: no explicit lod in a vertex shader -> lambda_base = 0
        else
        {
            const double W = T.w, H = T.h, Dd = T.d;
            const double mu = std::max(std::abs(W * dx[0]), std::abs(W * dy[0]));
            const double mv = std::max(std::abs(H * dx[1]), std::abs(H * dy[1]));
            const double mw = std::max(std::abs(Dd * dx[2]), std::abs(Dd * dy[2]));
            const double f_lo = std::max({mu, mv, mw}), f_hi = mu + mv + mw;
            const double ninf = -std::numeric_limits<double>::infinity();
            lo = f_lo > 0 ? std::log2(f_lo) : ninf;
            hi = f_hi > 0 ? std::log2(f_hi) : ninf;
            if (c.bias && !has(Mut::IgnoreBias))
            {
                const double b = std::clamp(c.lod, -bias_max, bias_max);  // eq. 3.17
                lo += b;
                hi += b;
            }
        }
        e.lambda_lo = lo;
        e.lambda_hi = hi;
        const int d_lo = select_level(lo, q), d_hi = select_level(hi, q);
        if (d_lo != d_hi || level_margin(lo, q) < kLambdaMargin || level_margin(hi, q) < kLambdaMargin)
            problem(std::format("lambda in [{:.3f}, {:.3f}] is too close to a level boundary", lo, hi));
        int level = d_lo;
        if (has(Mut::LevelUp)) level = level < q ? level + 1 : level - 1;

        // Texel selection (§3.8.10.2, eq. 3.19 with the size of the selected level, offsets in texels).
        const Dims D = level_dims(T, level);
        const double u = D.w * s + off[0];
        const double v = D.h * t + off[1];
        const double w = T.dim == Dim::D3 ? D.d * r + off[2] : 0.0;
        if (frac_margin(u) < kTexelMargin || frac_margin(v) < kTexelMargin || (T.dim == Dim::D3 && frac_margin(w) < kTexelMargin))
            problem(std::format("sample ({:.4f}, {:.4f}, {:.4f}) of level {} is close to a texel boundary", u, v, w, level));
        // Cube maps with NEAREST always clamp to the edge (§3.8.9.1).
        const Wrap ws = T.dim == Dim::Cube ? Wrap::ClampToEdge : T.ws;
        const Wrap wt = T.dim == Dim::Cube ? Wrap::ClampToEdge : T.wt;
        e.t.level = level;
        e.t.i = wrap_coord(static_cast<int>(std::floor(u)), D.w, ws);
        e.t.j = wrap_coord(static_cast<int>(std::floor(v)), D.h, wt);
        if (T.dim == Dim::D3) e.t.a = wrap_coord(static_cast<int>(std::floor(w)), D.d, T.wr);
        if (T.dim == Dim::Cube) e.t.a = face;
        if (T.dim == Dim::Array)
        {
            // §3.8.10.2: l = clamp(RNE(r), 0, dt - 1) (or floor(r + 1/2)); the offset never applies to the layer.
            if (frac_margin(layer + 0.5) < kTexelMargin) problem("array layer coordinate close to a rounding boundary");
            e.t.a = std::clamp(static_cast<int>(std::floor(layer + 0.5)), 0, T.d - 1);
        }
        return e;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Expected results
    // ---------------------------------------------------------------------------------------------------------------
    using Words = std::array<std::uint32_t, 4>;

    float as_float(std::uint32_t u) { return std::bit_cast<float>(u); }

    struct Expect
    {
        enum class Kind { Unorm, Exact } kind = Kind::Exact;
        std::array<double, 4> f{};  // Unorm: expected values (byte / 255)
        Words u{};                  // Exact: expected words
        bool valid = true;
    };

    Expect expect_texel(const Case& c, const Texel& t, std::array<float, 2> ref)
    {
        Expect e;
        e.valid = t.valid;
        const Fmt fmt = kTex[c.tex].fmt;
        if (fmt == Fmt::Rgba8)
        {
            e.kind = Expect::Kind::Unorm;
            const auto b = rgba8_of(t);
            for (std::size_t k = 0; k < 4; ++k) e.f[k] = b[k] / 255.0;
        }
        else if (fmt == Fmt::Depth32f)
        {
            // Table 3.23 LEQUAL: r = 1.0 when Dref <= Dt, else 0.0. With NEAREST filtering r is exactly 0 or 1 (§3.8.15.1).
            const float d = depth_of(t);
            e.u = {std::bit_cast<std::uint32_t>(ref[0] <= d ? 1.0f : 0.0f), std::bit_cast<std::uint32_t>(ref[1] <= d ? 1.0f : 0.0f), 0u, 0u};
        }
        else
            e.u = uint_of(c.tex, t);
        return e;
    }

    bool matches(const Expect& e, const Words& got)
    {
        if (!e.valid) return false;
        if (e.kind == Expect::Kind::Exact) return got == e.u;
        // An RGBA8 lookup returns c / 255 (§2.1.6.1); a quarter of a step separates the right byte from its neighbours.
        for (std::size_t k = 0; k < 4; ++k)
        {
            const double f = static_cast<double>(as_float(got[k]));
            if (!(std::abs(f - e.f[k]) <= 0.25 / 255.0)) return false;
        }
        return true;
    }

    std::string describe(const Words& w, Fmt fmt)
    {
        if (fmt == Fmt::Rgba8 || fmt == Fmt::Depth32f)
            return std::format("({:.4f}, {:.4f}, {:.4f}, {:.4f})", as_float(w[0]), as_float(w[1]), as_float(w[2]), as_float(w[3]));
        if (fmt == Fmt::Rgba32i)
            return std::format("({}, {}, {}, {})", static_cast<std::int32_t>(w[0]), static_cast<std::int32_t>(w[1]), static_cast<std::int32_t>(w[2]),
                               static_cast<std::int32_t>(w[3]));
        return std::format("({}, {}, {}, {:#x})", w[0], w[1], w[2], w[3]);
    }
    std::string describe(const Texel& t) { return std::format("level {} texel ({}, {}) slice/layer/face {}", t.level, t.i, t.j, t.a); }

    Words size_words(int tex, int level)
    {
        const TexInfo& T = kTex[static_cast<std::size_t>(tex)];
        const Dims D = level_dims(T, level);
        const int third = (T.dim == Dim::D3 || T.dim == Dim::Array) ? D.d : 1;  // probe.frag pads ivec2 sizes with 1
        return {static_cast<std::uint32_t>(D.w), static_cast<std::uint32_t>(D.h), static_cast<std::uint32_t>(third), 0u};
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The case table
    // ---------------------------------------------------------------------------------------------------------------
    constexpr double k3_128 = 3.0 / 128.0, k13_32 = 13.0 / 32.0, k19_64 = 19.0 / 64.0, k5_16 = 5.0 / 16.0, k3_64 = 3.0 / 64.0;

    std::vector<Case> build_cases(int off_min, int off_max)
    {
        std::vector<Case> v;
        using M = Mut;
        auto add = [&](Case c) { v.push_back(std::move(c)); };

        // ---- sampler2D: 32x16, levels 32x16 .. 1x1, S = REPEAT, T = CLAMP_TO_EDGE ----------------------------------
        add({.id = 1, .fn = "texture(sampler2D, P)", .tex = T2D, .mode = Mode::Implicit, .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 8, 0, 0, 0},
             .controls = {M::LevelUp}});
        add({.id = 2, .fn = "texture(sampler2D, P, bias = +1)", .tex = T2D, .mode = Mode::Implicit, .bias = true, .p = {k3_128, k13_32, 0, 1},
             .ddx = {1.0 / 16, 0, 0, 0}, .lod = 1.0, .controls = {M::IgnoreBias, M::LevelUp}});
        add({.id = 2, .fn = "texture(sampler2D, P, bias = -3) (magnified)", .tex = T2D, .mode = Mode::Implicit, .bias = true,
             .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 16, 0, 0, 0}, .lod = -3.0, .controls = {M::IgnoreBias}});
        add({.id = 2, .fn = "texture(sampler2D, P, bias = +100) (clamped to MAX_TEXTURE_LOD_BIAS)", .tex = T2D, .mode = Mode::Implicit, .bias = true,
             .p = {k19_64, k13_32, 0, 1}, .ddx = {1.0 / 16, 0, 0, 0}, .lod = 100.0, .controls = {M::IgnoreBias}, .optional = true});
        add({.id = 3, .fn = "textureProj(sampler2D, vec3 P)", .tex = T2D, .mode = Mode::Implicit, .proj = true, .p = {3.0 / 64, 13.0 / 16, 0, 2},
             .ddx = {1.0 / 8, 0, 0, 0}, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 4, .fn = "textureProj(sampler2D, vec4 P, bias = +1)", .tex = T2D, .mode = Mode::Implicit, .proj = true, .bias = true,
             .p = {3.0 / 32, 13.0 / 8, 77, 4}, .ddx = {1.0 / 4, 0, 0, 0}, .lod = 1.0, .controls = {M::NoDivide, M::IgnoreBias}});
        for (const double lod : {0.3, 2.4, 2.6, -1.0, 9.0})
            add({.id = 5, .fn = std::format("textureLod(sampler2D, P, {})", lod), .tex = T2D, .mode = Mode::Explicit, .p = {k3_128, k13_32, 0, 1},
                 .ddx = {1.0 / 32, 0, 0, 0}, .lod = lod, .controls = {M::LevelUp}});
        add({.id = 6, .fn = "textureOffset(sampler2D, P, ivec2(MAX_PROGRAM_TEXEL_OFFSET, MIN_PROGRAM_TEXEL_OFFSET))", .tex = T2D, .mode = Mode::Implicit,
             .off = {off_max, off_min, 0}, .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 16, 0, 0, 0}, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 7, .fn = "textureOffset(sampler2D, P, ivec2(-3, 2), bias = +2)", .tex = T2D, .mode = Mode::Implicit, .bias = true, .off = {-3, 2, 0},
             .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 2.0, .controls = {M::DropOffset, M::IgnoreBias}});
        add({.id = 8, .fn = "textureLodOffset(sampler2D, P, 1.0, ivec2(3, -2))", .tex = T2D, .mode = Mode::Explicit, .off = {3, -2, 0},
             .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 1.0, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 9, .fn = "textureGrad(sampler2D, P, (1/4, 0), (0, 0))", .tex = T2D, .mode = Mode::Grad, .p = {k3_128, k13_32, 0, 1},
             .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 4, 0, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 9, .fn = "textureGrad(sampler2D, P, (0, 0), (0, 1/4))", .tex = T2D, .mode = Mode::Grad, .p = {k3_128, k13_32, 0, 1},
             .ddx = {1.0 / 32, 0, 0, 0}, .gy = {0, 1.0 / 4, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 9, .fn = "textureGrad(sampler2D, P, (1/4, 0), (0, 1/8)) (rho in [8, 10])", .tex = T2D, .mode = Mode::Grad,
             .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 4, 0, 0, 0}, .gy = {0, 1.0 / 8, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 10, .fn = "textureGradOffset(sampler2D, P, (1/8, 0), (0, 0), ivec2(-1, 2))", .tex = T2D, .mode = Mode::Grad, .off = {-1, 2, 0},
             .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 8, 0, 0, 0}, .controls = {M::DropOffset}});
        add({.id = 11, .fn = "textureProjLod(sampler2D, vec3 P, 1.0)", .tex = T2D, .mode = Mode::Explicit, .proj = true, .p = {3.0 / 64, 13.0 / 16, 0, 2},
             .ddx = {1.0 / 16, 0, 0, 0}, .lod = 1.0, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 12, .fn = "textureProjLod(sampler2D, vec4 P, 1.0) (q = 1/2)", .tex = T2D, .mode = Mode::Explicit, .proj = true,
             .p = {3.0 / 256, 13.0 / 64, -5, 0.5}, .ddx = {1.0 / 64, 0, 0, 0}, .lod = 1.0, .controls = {M::NoDivide}});
        add({.id = 13, .fn = "textureProjOffset(sampler2D, vec4 P, ivec2(2, 1))", .tex = T2D, .mode = Mode::Implicit, .proj = true, .off = {2, 1, 0},
             .p = {3.0 / 64, 13.0 / 16, 9, 2}, .ddx = {1.0 / 8, 0, 0, 0}, .controls = {M::NoDivide, M::DropOffset}});
        add({.id = 14, .fn = "textureProjLodOffset(sampler2D, vec3 P, 2.0, ivec2(-2, -1))", .tex = T2D, .mode = Mode::Explicit, .proj = true,
             .off = {-2, -1, 0}, .p = {3.0 / 32, 13.0 / 8, 0, 4}, .ddx = {1.0 / 8, 0, 0, 0}, .lod = 2.0, .controls = {M::DropOffset, M::NoDivide}});
        add({.id = 15, .fn = "textureProjGrad(sampler2D, vec3 P, (1/8, 0), (0, 0))", .tex = T2D, .mode = Mode::Grad, .proj = true,
             .p = {3.0 / 64, 13.0 / 16, 0, 2}, .ddx = {1.0 / 16, 0, 0, 0}, .gx = {1.0 / 8, 0, 0, 0}, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 16, .fn = "textureProjGradOffset(sampler2D, vec4 P, (0, 0), (0, 1/2), ivec2(1, -1))", .tex = T2D, .mode = Mode::Grad, .proj = true,
             .off = {1, -1, 0}, .p = {3.0 / 64, 13.0 / 16, 33, 2}, .ddx = {1.0 / 16, 0, 0, 0}, .gy = {0, 1.0 / 2, 0, 0}, .controls = {M::DropOffset}});
        for (const auto& ip : {std::array<int, 4>{4, 5, 0, 0}, std::array<int, 4>{10, 3, 0, 1}, std::array<int, 4>{4, 0, 0, 2}})
            add({.id = 17, .fn = std::format("texelFetch(sampler2D, ({}, {}) + pixel, {})", ip[0], ip[1], ip[3]), .tex = T2D, .mode = Mode::Fetch,
                 .ip = ip, .controls = {M::LevelUp}});
        for (const auto& ip : {std::array<int, 4>{9, 2, 0, 0}, std::array<int, 4>{6, 0, 0, 1}})
            add({.id = 18, .fn = std::format("texelFetchOffset(sampler2D, ({}, {}) + pixel, {}, ivec2(-5, 3))", ip[0], ip[1], ip[3]), .tex = T2D,
                 .mode = Mode::Fetch, .off = {-5, 3, 0}, .ip = ip, .controls = {M::DropOffset}});

        // ---- sampler3D: 16x8x8, 5 levels (depth halves too), REPEAT -------------------------------------------------
        const V4 p3{k13_32, k5_16, k3_64, 0};
        add({.id = 20, .fn = "texture(sampler3D, P, bias = +1) (rho from dr/dx)", .tex = T3D, .mode = Mode::Implicit, .bias = true, .p = p3,
             .ddx = {0, 0, 1.0 / 4, 0}, .lod = 1.0, .controls = {M::IgnoreBias, M::LevelUp}});
        for (const double lod : {1.0, 3.3})
            add({.id = 21, .fn = std::format("textureLod(sampler3D, P, {})", lod), .tex = T3D, .mode = Mode::Explicit, .p = p3, .ddx = {1.0 / 64, 0, 0, 0},
                 .lod = lod, .controls = {M::LevelUp}});
        add({.id = 22, .fn = "textureOffset(sampler3D, P, ivec3(-2, 1, -1))", .tex = T3D, .mode = Mode::Implicit, .off = {-2, 1, -1}, .p = p3,
             .ddx = {1.0 / 64, 0, 0, 0}, .controls = {M::DropOffset}});
        for (const double lod : {0.0, 2.0})
            add({.id = 23, .fn = std::format("textureLodOffset(sampler3D, P, {}, ivec3(1, -1, 2))", lod), .tex = T3D, .mode = Mode::Explicit,
                 .off = {1, -1, 2}, .p = p3, .ddx = {1.0 / 64, 0, 0, 0}, .lod = lod, .controls = {M::DropOffset}});
        add({.id = 24, .fn = "textureGrad(sampler3D, P, (0, 0, 1/2), 0)", .tex = T3D, .mode = Mode::Grad, .p = p3, .ddx = {1.0 / 64, 0, 0, 0},
             .gx = {0, 0, 1.0 / 2, 0}, .controls = {M::LevelUp}});
        add({.id = 24, .fn = "textureGrad(sampler3D, P, 0, (0, 1/4, 0))", .tex = T3D, .mode = Mode::Grad, .p = p3, .ddx = {1.0 / 64, 0, 0, 0},
             .gy = {0, 1.0 / 4, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 25, .fn = "textureGradOffset(sampler3D, P, (1/8, 0, 0), 0, ivec3(1, 0, -1))", .tex = T3D, .mode = Mode::Grad, .off = {1, 0, -1},
             .p = p3, .ddx = {1.0 / 64, 0, 0, 0}, .gx = {1.0 / 8, 0, 0, 0}, .controls = {M::DropOffset}});
        const V4 p3q{13.0 / 16, 5.0 / 8, 3.0 / 32, 2};
        add({.id = 26, .fn = "textureProj(sampler3D, P)", .tex = T3D, .mode = Mode::Implicit, .proj = true, .p = p3q, .ddx = {0, 0, 1.0 / 2, 0},
             .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 27, .fn = "textureProjLod(sampler3D, P, 2.0) (q = 4)", .tex = T3D, .mode = Mode::Explicit, .proj = true,
             .p = {13.0 / 8, 5.0 / 4, 3.0 / 16, 4}, .ddx = {1.0 / 16, 0, 0, 0}, .lod = 2.0, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 28, .fn = "textureProjOffset(sampler3D, P, ivec3(0, 1, 1))", .tex = T3D, .mode = Mode::Implicit, .proj = true, .off = {0, 1, 1},
             .p = p3q, .ddx = {1.0 / 32, 0, 0, 0}, .controls = {M::DropOffset, M::NoDivide}});
        add({.id = 29, .fn = "textureProjLodOffset(sampler3D, P, 1.0, ivec3(-1, 0, 1))", .tex = T3D, .mode = Mode::Explicit, .proj = true,
             .off = {-1, 0, 1}, .p = p3q, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 1.0, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 30, .fn = "textureProjGrad(sampler3D, P, (1/4, 0, 0), 0)", .tex = T3D, .mode = Mode::Grad, .proj = true, .p = p3q,
             .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 4, 0, 0, 0}, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 31, .fn = "textureProjGradOffset(sampler3D, P, (0, 0, 1/4), 0, ivec3(2, -1, 0))", .tex = T3D, .mode = Mode::Grad, .proj = true,
             .off = {2, -1, 0}, .p = p3q, .ddx = {1.0 / 32, 0, 0, 0}, .gx = {0, 0, 1.0 / 4, 0}, .controls = {M::DropOffset}});
        for (const auto& ip : {std::array<int, 4>{8, 2, 5, 0}, std::array<int, 4>{3, 0, 2, 1}})
            add({.id = 32, .fn = std::format("texelFetch(sampler3D, ({}, {}, {}) + pixel, {})", ip[0], ip[1], ip[2], ip[3]), .tex = T3D, .mode = Mode::Fetch,
                 .ip = ip, .controls = {M::LevelUp}});
        add({.id = 33, .fn = "texelFetchOffset(sampler3D, (5, 1, 3) + pixel, 0, ivec3(-1, 2, 1))", .tex = T3D, .mode = Mode::Fetch, .off = {-1, 2, 1},
             .ip = {5, 1, 3, 0}, .controls = {M::DropOffset}});

        // ---- sampler2DArray: 16x16, 4 layers, 5 levels, S = REPEAT, T = CLAMP_TO_EDGE ------------------------------
        add({.id = 40, .fn = "texture(sampler2DArray, P, bias = +1) (layer changes per pixel)", .tex = TArr, .mode = Mode::Implicit, .bias = true,
             .p = {k3_128, k13_32, -0.25, 0}, .ddx = {1.0 / 8, 0, 1, 0}, .lod = 1.0, .controls = {M::IgnoreBias, M::LayerInRho}});
        for (const double layer : {2.0, 7.25, -2.0})
            add({.id = 41, .fn = std::format("textureLod(sampler2DArray, (P.xy, {}), 1.0)", layer), .tex = TArr, .mode = Mode::Explicit,
                 .p = {k3_128, k13_32, layer, 0}, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 1.0, .controls = {M::LevelUp}});
        add({.id = 42, .fn = "textureOffset(sampler2DArray, P, ivec2(2, -1))", .tex = TArr, .mode = Mode::Implicit, .off = {2, -1, 0},
             .p = {k3_128, k13_32, 1, 0}, .ddx = {1.0 / 128, 0, 0, 0}, .controls = {M::DropOffset}});
        add({.id = 43, .fn = "textureLodOffset(sampler2DArray, P, 2.0, ivec2(-1, 1))", .tex = TArr, .mode = Mode::Explicit, .off = {-1, 1, 0},
             .p = {k3_128, k13_32, 3, 0}, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 2.0, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 44, .fn = "textureGrad(sampler2DArray, P, (1/4, 0), 0)", .tex = TArr, .mode = Mode::Grad, .p = {k3_128, k13_32, 2, 0},
             .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 4, 0, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 44, .fn = "textureGrad(sampler2DArray, P, 0, (0, 1/2))", .tex = TArr, .mode = Mode::Grad, .p = {k3_128, k13_32, 2, 0},
             .ddx = {1.0 / 32, 0, 0, 0}, .gy = {0, 1.0 / 2, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 45, .fn = "textureGradOffset(sampler2DArray, P, (1/8, 0), 0, ivec2(1, 1))", .tex = TArr, .mode = Mode::Grad, .off = {1, 1, 0},
             .p = {k3_128, k13_32, 0, 0}, .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 8, 0, 0, 0}, .controls = {M::DropOffset}});
        for (const auto& ip : {std::array<int, 4>{6, 9, 2, 0}, std::array<int, 4>{0, 0, 3, 2}})
            add({.id = 46, .fn = std::format("texelFetch(sampler2DArray, ({}, {}, {}) + pixel, {})", ip[0], ip[1], ip[2], ip[3]), .tex = TArr,
                 .mode = Mode::Fetch, .ip = ip, .controls = {M::LevelUp}});
        add({.id = 47, .fn = "texelFetchOffset(sampler2DArray, (3, 6, 1) + pixel, 0, ivec2(2, -2))", .tex = TArr, .mode = Mode::Fetch, .off = {2, -2, 0},
             .ip = {3, 6, 1, 0}, .controls = {M::DropOffset}});

        // ---- samplerCube: 16x16 faces, 5 levels -----------------------------------------------------------------------
        for (const int face : {0, 3})
            add({.id = 50, .fn = std::format("texture(samplerCube, P, bias = +1) (face {})", face), .tex = TCube, .mode = Mode::Implicit, .bias = true,
                 .p = cube_dir(face, k19_64, k13_32), .ddx = cube_sc_axis(face) * (2.0 / 8), .lod = 1.0, .controls = {M::IgnoreBias, M::LevelUp}});
        const std::array<double, 6> cube_lods = {0.0, 1.0, 2.0, 3.0, 4.0, 1.4};
        for (int face = 0; face < 6; ++face)
        {
            const double lod = cube_lods[static_cast<std::size_t>(face)];
            Case c{.id = 51, .fn = std::format("textureLod(samplerCube, P, {}) (face {})", lod, face), .tex = TCube, .mode = Mode::Explicit,
                   .p = cube_dir(face, k19_64, k13_32), .ddx = cube_sc_axis(face) * (2.0 / 64), .lod = lod, .controls = {M::LevelUp}};
            if (lod <= 2.0) c.controls.push_back(M::MirrorS);
            add(c);
        }
        add({.id = 52, .fn = "textureGrad(samplerCube, P, dPdx along sc, 0) (face 4)", .tex = TCube, .mode = Mode::Grad, .p = cube_dir(4, k19_64, k13_32),
             .ddx = cube_sc_axis(4) * (2.0 / 64), .gx = cube_sc_axis(4) * (2.0 / 4), .controls = {M::LevelUp, M::MirrorS}});
        add({.id = 52, .fn = "textureGrad(samplerCube, P, 0, dPdy along tc) (face 2)", .tex = TCube, .mode = Mode::Grad, .p = cube_dir(2, k19_64, k13_32),
             .ddx = cube_sc_axis(2) * (2.0 / 64), .gy = cube_tc_axis(2) * (2.0 / 8), .controls = {M::LevelUp}});
        add({.id = 53, .fn = "texture(samplerCube, P) (face 1)", .tex = TCube, .mode = Mode::Implicit, .p = cube_dir(1, 1.0 / 32, k13_32),
             .ddx = cube_sc_axis(1) * (2.0 / 4), .controls = {M::LevelUp, M::MirrorS}});

        // ---- integer samplers ---------------------------------------------------------------------------------------
        for (const auto& ip : {std::array<int, 4>{5, 7, 0, 0}, std::array<int, 4>{2, 1, 0, 1}, std::array<int, 4>{0, 0, 0, 2}})
            add({.id = 60, .fn = std::format("texelFetch(usampler2D, ({}, {}) + pixel, {})", ip[0], ip[1], ip[3]), .tex = TU2D, .mode = Mode::Fetch,
                 .ip = ip, .controls = {M::LevelUp}});
        add({.id = 61, .fn = "textureLod(usampler2D, P, 2.0)", .tex = TU2D, .mode = Mode::Explicit, .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 32, 0, 0, 0},
             .lod = 2.0, .controls = {M::LevelUp}});
        add({.id = 62, .fn = "textureOffset(usampler2D, P, ivec2(2, -3), bias = +1)", .tex = TU2D, .mode = Mode::Implicit, .bias = true, .off = {2, -3, 0},
             .p = {k3_128, k13_32, 0, 1}, .ddx = {1.0 / 16, 0, 0, 0}, .lod = 1.0, .controls = {M::DropOffset, M::IgnoreBias}});
        add({.id = 63, .fn = "textureGrad(usampler2D, P, (1/2, 0), 0)", .tex = TU2D, .mode = Mode::Grad, .p = {k3_128, k13_32, 0, 1},
             .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 2, 0, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 64, .fn = "textureProj(usampler2D, vec3 P)", .tex = TU2D, .mode = Mode::Implicit, .proj = true, .p = {3.0 / 64, 13.0 / 16, 0, 2},
             .ddx = {1.0 / 4, 0, 0, 0}, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 65, .fn = "texelFetchOffset(isampler3D, (2, 1, 1) + pixel, 0, ivec3(-1, 2, 1))", .tex = TI3D, .mode = Mode::Fetch, .off = {-1, 2, 1},
             .ip = {2, 1, 1, 0}, .controls = {M::DropOffset}});
        add({.id = 66, .fn = "textureLodOffset(isampler3D, P, 1.0, ivec3(1, -1, 1))", .tex = TI3D, .mode = Mode::Explicit, .off = {1, -1, 1},
             .p = {k13_32, k5_16, 3.0 / 16, 0}, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 1.0, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 67, .fn = "textureProjLod(isampler3D, P, 0.0)", .tex = TI3D, .mode = Mode::Explicit, .proj = true, .p = {13.0 / 16, 5.0 / 8, 3.0 / 8, 2},
             .ddx = {1.0 / 16, 0, 0, 0}, .lod = 0.0, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 68, .fn = "texture(isampler3D, P, bias = +2)", .tex = TI3D, .mode = Mode::Implicit, .bias = true, .p = {k13_32, k5_16, 3.0 / 16, 0},
             .ddx = {0, 0, 1.0 / 4, 0}, .lod = 2.0, .controls = {M::IgnoreBias, M::LevelUp}});
        for (const auto& [face, lod] : {std::pair{2, 1.0}, std::pair{5, 2.0}})
            add({.id = 69, .fn = std::format("textureLod(usamplerCube, P, {}) (face {})", lod, face), .tex = TUCube, .mode = Mode::Explicit,
                 .p = cube_dir(face, k19_64, k13_32), .ddx = cube_sc_axis(face) * (2.0 / 64), .lod = lod, .controls = {M::LevelUp, M::MirrorS}});
        add({.id = 70, .fn = "texture(usamplerCube, P, bias = +2) (face 4)", .tex = TUCube, .mode = Mode::Implicit, .bias = true,
             .p = cube_dir(4, 1.0 / 32, k13_32), .ddx = cube_sc_axis(4) * (2.0 / 8), .lod = 2.0, .controls = {M::IgnoreBias, M::MirrorS}});

        // ---- shadow samplers (LEQUAL); the result pair is (ref just below, ref just above) the expected depth ----------
        const V4 ps{k3_128, k13_32, 0, 1};
        add({.id = 80, .fn = "texture(sampler2DShadow, P, bias = +1)", .tex = TS2D, .mode = Mode::Implicit, .bias = true, .p = ps,
             .ddx = {1.0 / 16, 0, 0, 0}, .lod = 1.0, .controls = {M::IgnoreBias, M::LevelUp}});
        add({.id = 81, .fn = "textureLod(sampler2DShadow, P, 2.0)", .tex = TS2D, .mode = Mode::Explicit, .p = ps, .ddx = {1.0 / 32, 0, 0, 0}, .lod = 2.0,
             .controls = {M::LevelUp}});
        add({.id = 82, .fn = "textureOffset(sampler2DShadow, P, ivec2(-2, 3))", .tex = TS2D, .mode = Mode::Implicit, .off = {-2, 3, 0}, .p = ps,
             .ddx = {1.0 / 8, 0, 0, 0}, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 83, .fn = "textureLodOffset(sampler2DShadow, P, 0.0, ivec2(1, -2))", .tex = TS2D, .mode = Mode::Explicit, .off = {1, -2, 0}, .p = ps,
             .ddx = {1.0 / 32, 0, 0, 0}, .lod = 0.0, .controls = {M::DropOffset}});
        add({.id = 84, .fn = "textureGrad(sampler2DShadow, P, (1/4, 0), 0)", .tex = TS2D, .mode = Mode::Grad, .p = ps, .ddx = {1.0 / 32, 0, 0, 0},
             .gx = {1.0 / 4, 0, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 85, .fn = "textureGradOffset(sampler2DShadow, P, 0, (0, 1/8), ivec2(-1, -1))", .tex = TS2D, .mode = Mode::Grad, .off = {-1, -1, 0},
             .p = ps, .ddx = {1.0 / 32, 0, 0, 0}, .gy = {0, 1.0 / 8, 0, 0}, .controls = {M::DropOffset, M::LevelUp}});
        const V4 psq{3.0 / 64, 13.0 / 16, 0, 2};
        add({.id = 86, .fn = "textureProj(sampler2DShadow, P)", .tex = TS2D, .mode = Mode::Implicit, .proj = true, .p = psq, .ddx = {1.0 / 8, 0, 0, 0},
             .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 87, .fn = "textureProjLod(sampler2DShadow, P, 1.0)", .tex = TS2D, .mode = Mode::Explicit, .proj = true, .p = psq,
             .ddx = {1.0 / 16, 0, 0, 0}, .lod = 1.0, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 88, .fn = "textureProjOffset(sampler2DShadow, P, ivec2(1, 2)) (q = 4)", .tex = TS2D, .mode = Mode::Implicit, .proj = true,
             .off = {1, 2, 0}, .p = {3.0 / 32, 13.0 / 8, 0, 4}, .ddx = {1.0 / 2, 0, 0, 0}, .controls = {M::DropOffset, M::NoDivide}});
        add({.id = 89, .fn = "textureProjGrad(sampler2DShadow, P, (1/2, 0), 0)", .tex = TS2D, .mode = Mode::Grad, .proj = true, .p = psq,
             .ddx = {1.0 / 16, 0, 0, 0}, .gx = {1.0 / 2, 0, 0, 0}, .controls = {M::NoDivide, M::LevelUp}});
        add({.id = 90, .fn = "textureProjLodOffset(sampler2DShadow, P, 2.0, ivec2(-1, 1))", .tex = TS2D, .mode = Mode::Explicit, .proj = true,
             .off = {-1, 1, 0}, .p = psq, .ddx = {1.0 / 16, 0, 0, 0}, .lod = 2.0, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 91, .fn = "textureProjGradOffset(sampler2DShadow, P, (1/8, 0), 0, ivec2(2, -1))", .tex = TS2D, .mode = Mode::Grad, .proj = true,
             .off = {2, -1, 0}, .p = psq, .ddx = {1.0 / 16, 0, 0, 0}, .gx = {1.0 / 8, 0, 0, 0}, .controls = {M::DropOffset, M::LevelUp}});
        add({.id = 92, .fn = "texture(samplerCubeShadow, P, bias = +1) (face 5)", .tex = TSCube, .mode = Mode::Implicit, .bias = true,
             .p = cube_dir(5, 1.0 / 32, k13_32), .ddx = cube_sc_axis(5) * (2.0 / 8), .lod = 1.0, .controls = {M::IgnoreBias, M::MirrorS}});
        add({.id = 93, .fn = "textureGrad(samplerCubeShadow, P, dPdx along tc, 0) (face 2)", .tex = TSCube, .mode = Mode::Grad,
             .p = cube_dir(2, k19_64, k13_32), .ddx = cube_sc_axis(2) * (2.0 / 64), .gx = cube_tc_axis(2) * 1.0, .controls = {M::LevelUp}});
        add({.id = 94, .fn = "texture(sampler2DArrayShadow, P) (layer 2)", .tex = TSArr, .mode = Mode::Implicit, .p = {k3_128, k13_32, 2, 0},
             .ddx = {1.0 / 4, 0, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 95, .fn = "textureGrad(sampler2DArrayShadow, P, (1/2, 0), 0) (layer 1)", .tex = TSArr, .mode = Mode::Grad, .p = {k3_128, k13_32, 1, 0},
             .ddx = {1.0 / 32, 0, 0, 0}, .gx = {1.0 / 2, 0, 0, 0}, .controls = {M::LevelUp}});
        add({.id = 96, .fn = "textureGradOffset(sampler2DArrayShadow, P, 0, (0, 1/4), ivec2(1, -1)) (layer 0)", .tex = TSArr, .mode = Mode::Grad,
             .off = {1, -1, 0}, .p = {k3_128, k13_32, 0, 0}, .ddx = {1.0 / 32, 0, 0, 0}, .gy = {0, 1.0 / 4, 0, 0}, .controls = {M::DropOffset}});

        // ---- textureSize of every level of every sampler ------------------------------------------------------------
        for (int tex = 0; tex < kTexCount; ++tex)
            for (int level = 0; level < kTex[static_cast<std::size_t>(tex)].levels; ++level)
                add({.id = 100, .fn = std::format("textureSize({}, {})", kTex[static_cast<std::size_t>(tex)].glsl, level), .tex = static_cast<Tex>(tex),
                     .mode = Mode::Size, .ip = {tex, level, 0, 0}, .controls = {M::LevelUp}});

        // ---- derivatives ----------------------------------------------------------------------------------------------
        add({.id = 110, .fn = "dFdx(affine vec4 of gl_FragCoord)", .mode = Mode::Deriv});
        add({.id = 111, .fn = "dFdy(affine vec4 of gl_FragCoord)", .mode = Mode::Deriv});
        add({.id = 112, .fn = "fwidth(affine vec4 of gl_FragCoord)", .mode = Mode::Deriv});
        add({.id = 113, .fn = "dFdx / dFdy of an interpolated input", .mode = Mode::Deriv});
        add({.id = 114, .fn = "dFdx(P.x), dFdy(P.y), fwidth(P.x + P.y) of the uniform-built coordinate", .mode = Mode::Deriv,
             .ddx = {1.0 / 4, 1.0 / 8, 0, 0}, .ddy = {1.0 / 2, -1.0, 0, 0}});
        add({.id = 115, .fn = "dFdx(x^2), dFdy(y^2) (informational)", .mode = Mode::Deriv, .p = {0, 0, 0, 0}});

        // ---- vertex stage (transform feedback); P is a single point, no derivatives ---------------------------------
        const V4 pv{k19_64, k13_32, 0, 1};
        auto vadd = [&](Case c) {
            c.vertex = true;
            add(std::move(c));
        };
        vadd({.id = 0, .fn = "vertex texture(sampler2D, P) -> base level", .tex = T2D, .mode = Mode::Implicit, .p = pv, .controls = {M::LevelUp}});
        vadd({.id = 1, .fn = "vertex textureLod(sampler2D, P, 2.4)", .tex = T2D, .mode = Mode::Explicit, .p = pv, .lod = 2.4, .controls = {M::LevelUp}});
        vadd({.id = 2, .fn = "vertex textureOffset(sampler2D, P, ivec2(2, -1))", .tex = T2D, .mode = Mode::Implicit, .off = {2, -1, 0}, .p = pv,
              .controls = {M::DropOffset}});
        vadd({.id = 3, .fn = "vertex textureProj(sampler2D, vec3 P)", .tex = T2D, .mode = Mode::Implicit, .proj = true, .p = {19.0 / 32, 13.0 / 16, 0, 2},
              .controls = {M::NoDivide}});
        vadd({.id = 4, .fn = "vertex textureProjLod(sampler3D, P, 1.0)", .tex = T3D, .mode = Mode::Explicit, .proj = true, .p = p3q, .lod = 1.0,
              .controls = {M::NoDivide, M::LevelUp}});
        vadd({.id = 5, .fn = "vertex textureLodOffset(sampler2DArray, P, 1.0, ivec2(-1, 1))", .tex = TArr, .mode = Mode::Explicit, .off = {-1, 1, 0},
              .p = {k19_64, k13_32, 2, 0}, .lod = 1.0, .controls = {M::DropOffset, M::LevelUp}});
        vadd({.id = 6, .fn = "vertex texelFetch(sampler3D, (1, 0, 0), 3)", .tex = T3D, .mode = Mode::Fetch, .ip = {1, 0, 0, 3}, .controls = {M::LevelUp}});
        vadd({.id = 7, .fn = "vertex texelFetchOffset(sampler2D, (6, 1), 2, ivec2(-3, 2))", .tex = T2D, .mode = Mode::Fetch, .off = {-3, 2, 0},
              .ip = {6, 1, 0, 2}, .controls = {M::DropOffset}});
        vadd({.id = 8, .fn = "vertex textureSize(samplerCube, 3), textureSize(sampler2DArray, 3)", .tex = TCube, .mode = Mode::Size, .ip = {0, 0, 0, 3},
              .controls = {M::LevelUp}});
        vadd({.id = 9, .fn = "vertex textureLod(samplerCube, P, 4.0) (face 3)", .tex = TCube, .mode = Mode::Explicit, .p = cube_dir(3, k19_64, k13_32),
              .lod = 4.0, .controls = {M::LevelUp}});
        vadd({.id = 10, .fn = "vertex textureLod(sampler2DShadow, P, 3.0)", .tex = TS2D, .mode = Mode::Explicit, .p = pv, .lod = 3.0,
              .controls = {M::LevelUp}});
        vadd({.id = 11, .fn = "vertex textureLod(usampler2D, P, 1.0)", .tex = TU2D, .mode = Mode::Explicit, .p = pv, .lod = 1.0, .controls = {M::LevelUp}});
        vadd({.id = 12, .fn = "vertex texelFetch(isampler3D, (0, 1, 0), 2)", .tex = TI3D, .mode = Mode::Fetch, .ip = {0, 1, 0, 2}, .controls = {M::LevelUp}});
        return v;
    }

    // Derivative expectations (exact slopes of affine functions).
    std::array<double, 4> deriv_expect(const Case& c)
    {
        switch (c.id)
        {
        case 110: return {3.0, -2.0, 0.75, 0.0};
        case 111: return {-5.0, 0.5, 0.75, 0.25};
        case 112: return {8.0, 2.5, 1.5, 0.25};
        case 113: return {1.0, 0.25, 0.0, 1.0};
        case 114: return {c.ddx.x, c.ddy.y, std::abs(c.ddx.x + c.ddx.y) + std::abs(c.ddy.x + c.ddy.y), 0.0};
        default: return {};
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Display layout (relative to the window size)
    // ---------------------------------------------------------------------------------------------------------------
    struct Rect
    {
        int x, y, w, h;
    };
    struct Layout
    {
        Rect plane{}, grid{};
        int cols = 1, rows = 1;
    };
    Layout make_layout(int width, int height, int cells)
    {
        Layout l;
        if (width >= height)
        {
            const int pw = width * 9 / 20;
            l.plane = {0, 0, pw, height};
            l.grid = {pw, 0, width - pw, height};
        }
        else
        {
            const int ph = height * 2 / 5;
            l.plane = {0, height - ph, width, ph};
            l.grid = {0, 0, width, height - ph};
        }
        const double aspect = static_cast<double>(l.grid.w) / std::max(1, l.grid.h);
        l.cols = std::max(1, static_cast<int>(std::ceil(std::sqrt(cells * aspect))));
        l.rows = (cells + l.cols - 1) / l.cols;
        return l;
    }
    Rect cell_rect(const Layout& l, int k)  // row 0 at the top
    {
        const int c = k % l.cols, r = k / l.cols;
        const int x0 = l.grid.x + c * l.grid.w / l.cols, x1 = l.grid.x + (c + 1) * l.grid.w / l.cols;
        const int y1 = l.grid.y + l.grid.h - r * l.grid.h / l.rows, y0 = l.grid.y + l.grid.h - (r + 1) * l.grid.h / l.rows;
        return {x0, y0, x1 - x0, y1 - y0};
    }

    constexpr std::array<std::array<double, 3>, 8> kPalette = {{{0.902, 0.235, 0.204}, {0.957, 0.612, 0.180}, {0.941, 0.886, 0.290},
                                                                 {0.384, 0.769, 0.329}, {0.231, 0.702, 0.769}, {0.290, 0.416, 0.875},
                                                                 {0.588, 0.341, 0.839}, {0.878, 0.878, 0.878}}};

    int display_kind(const Case& c)
    {
        if (c.mode == Mode::Size) return 4;
        if (c.mode == Mode::Deriv) return 5;
        switch (kTex[c.tex].fmt)
        {
        case Fmt::Rgba8: return 0;
        case Fmt::Rgba32ui:
        case Fmt::Rgba8ui: return 1;
        case Fmt::Rgba32i: return 2;
        case Fmt::Depth32f: return 3;
        }
        return 0;
    }

    template <typename T>
    std::vector<T> pixels_of_block(const std::vector<T>& image, int block)
    {
        std::vector<T> out;
        const int bx = (block % kBlocksPerRow) * kBlock, by = (block / kBlocksPerRow) * kBlock;
        for (int y = 0; y < kBlock; ++y)
            for (int x = 0; x < kBlock; ++x) out.push_back(image[static_cast<std::size_t>((by + y) * kTargetW + bx + x)]);
        return out;
    }

    class GlslTextureFunctions final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            off_min_ = demo::get_integer(mgl::GetParameter::MinProgramTexelOffset);
            off_max_ = demo::get_integer(mgl::GetParameter::MaxProgramTexelOffset);
            mgl::glGetFloatv(mgl::GetParameter::MaxTextureLodBias, &bias_max_);

            cases_ = build_cases(off_min_, off_max_);
            plan_cases();
            const Layout l = make_layout(env.width, env.height, static_cast<int>(fs_cases_.size()));
            const Rect smallest = cell_rect(l, static_cast<int>(fs_cases_.size()) - 1);
            if (cell_rect(l, 0).w < 8 || cell_rect(l, 0).h < 8 || smallest.w < 8 || smallest.h < 8)
                throw demo::Skip(std::format("the {} result cells need at least 8x8 pixels each; {}x{} is too small", fs_cases_.size(), env.width, env.height));

            build_programs(env);
            create_textures();
            create_targets();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);
            run_probes();
            run_vertex_stage();
            draw_picture(env);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_limits(check);
            check_fragment_cases(check);
            check_vertex_cases(check);
            check_derivatives(check);
            check_compile_rules(check);
            check_picture(env, check);
            check.expect_no_gl_error("no GL error during validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, mgl::TransformFeedbackId{0});
            mgl::glDeleteTransformFeedbacks(1, &tfo_);
            mgl::glDeleteBuffers(1, &tf_buffer_);
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteTextures(1, &results_);
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteVertexArrays(1, &vao_);
            for (const ProbeProgram& p : probe_) mgl::glDeleteProgram(p.id);
            for (const mgl::ProgramId p : {vertex_.id, plane_.id, cell_.id}) mgl::glDeleteProgram(p);
        }

    private:
        struct ProbeProgram
        {
            mgl::ProgramId id{};
            mgl::UniformLocation kase, origin, p, ddx, ddy, gx, gy, lod, ip, ref;
        };
        struct VertexProgram
        {
            mgl::ProgramId id{};
            mgl::UniformLocation vp, vlod, vip, vref;
        };
        struct PlaneProgram
        {
            mgl::ProgramId id{};
            mgl::UniformLocation mvp;
        };
        struct CellProgram
        {
            mgl::ProgramId id{};
            mgl::UniformLocation block, kind, rect;
        };

        static constexpr std::array<const char*, kTexCount> kSamplerNames = {"u_t2d", "u_t3d", "u_tarr", "u_tcube", "u_tu2d",
                                                                            "u_ti3d", "u_tucube", "u_s2d", "u_scube", "u_sarr"};
        static constexpr std::array<mgl::TextureUnit, 11> kUnits = {
            mgl::TextureUnit::Texture0, mgl::TextureUnit::Texture1, mgl::TextureUnit::Texture2, mgl::TextureUnit::Texture3,
            mgl::TextureUnit::Texture4, mgl::TextureUnit::Texture5, mgl::TextureUnit::Texture6, mgl::TextureUnit::Texture7,
            mgl::TextureUnit::Texture8, mgl::TextureUnit::Texture9, mgl::TextureUnit::Texture10};
        static constexpr int kResultsUnit = 10;
        static constexpr int kGroups = 6;  // probe.frag is compiled once per GROUP (see the comment there)
        static int group_of(int id) { return id < 20 ? 0 : id < 40 ? 1 : id < 60 ? 2 : id < 80 ? 3 : id < 100 ? 4 : 5; }

        // ---- planning: slots, references, self-check of the table ----------------------------------------------------
        void plan_cases()
        {
            std::vector<std::string> problems;
            std::vector<Case> kept;
            for (Case& c : cases_)
            {
                if (c.mode == Mode::Deriv || c.mode == Mode::Size)
                {
                    kept.push_back(c);
                    continue;
                }
                const int n = c.vertex ? 1 : kBlock;
                std::string problem;
                for (int py = 0; py < n && problem.empty(); ++py)
                    for (int px = 0; px < n && problem.empty(); ++px)
                    {
                        const Eval e = model(c, px, py, std::nullopt, static_cast<double>(bias_max_));
                        if (!e.problem.empty()) problem = std::format("pixel ({}, {}): {}", px, py, e.problem);
                        const float d = depth_of(e.t);
                        c.refs[static_cast<std::size_t>(py * kBlock + px)] = {static_cast<float>(static_cast<double>(d) - kRefHalfGap),
                                                                              static_cast<float>(static_cast<double>(d) + kRefHalfGap)};
                    }
                if (!problem.empty())
                {
                    if (c.optional)
                    {
                        std::cout << std::format("     note: '{}' is not tested on this implementation: {} (MAX_TEXTURE_LOD_BIAS = {})\n", c.fn, problem,
                                                 bias_max_);
                        continue;
                    }
                    problems.push_back(std::format("{}: {}", c.fn, problem));
                }
                // Every control must change the prediction somewhere, or it would prove nothing.
                for (const Mut m : c.controls)
                    if (!control_differs(c, m)) problems.push_back(std::format("{}: the {} predicts the same result", c.fn, mut_name(m)));
                kept.push_back(c);
            }
            for (Case& c : kept)
                if (c.mode == Mode::Size)
                    for (const Mut m : c.controls)
                        if (!control_differs(c, m)) problems.push_back(std::format("{}: the {} predicts the same result", c.fn, mut_name(m)));
            if (!problems.empty())
            {
                std::string all;
                for (const std::string& p : problems) all += "\n  " + p;
                throw demo::Failure("the case table is not robustly specified:" + all);
            }
            cases_ = std::move(kept);
            int block = 0, vertex_slot = 0;
            for (std::size_t n = 0; n < cases_.size(); ++n)
            {
                Case& c = cases_[n];
                if (c.vertex)
                {
                    c.slot = vertex_slot++;
                    if (c.slot != c.id) throw demo::Failure("vertex cases must be listed in gl_VertexID order");
                    vs_cases_.push_back(n);
                }
                else
                {
                    c.slot = block++;
                    fs_cases_.push_back(n);
                }
            }
            if (block > kMaxBlocks || vertex_slot != kVertexCases) throw demo::Failure("case count does not match the probe target / vertex shader");
        }

        // Expected result words of a case at one pixel (correct model or a mutation).
        Expect expected(const Case& c, int px, int py, std::optional<Mut> m) const
        {
            if (c.mode == Mode::Size)
            {
                int level = c.vertex ? c.ip[3] : c.ip[1];
                const int tex = c.vertex ? static_cast<int>(TCube) : c.ip[0];
                const int q = kTex[static_cast<std::size_t>(tex)].levels - 1;
                if (m == Mut::LevelUp) level = level < q ? level + 1 : level - 1;
                Expect e;
                if (c.vertex)
                {
                    const Words cube = size_words(TCube, level), arr = size_words(TArr, level);
                    e.u = {cube[0], cube[1], arr[0], arr[2]};
                }
                else
                    e.u = size_words(tex, level);
                return e;
            }
            const Eval ev = model(c, px, py, m, static_cast<double>(bias_max_));
            return expect_texel(c, ev.t, c.refs[static_cast<std::size_t>(py * kBlock + px)]);
        }

        bool control_differs(const Case& c, Mut m) const
        {
            const int n = c.vertex ? 1 : kBlock;
            for (int py = 0; py < n; ++py)
                for (int px = 0; px < n; ++px)
                {
                    const Expect a = expected(c, px, py, std::nullopt), b = expected(c, px, py, m);
                    if (!b.valid) return true;
                    if (a.kind == Expect::Kind::Exact ? a.u != b.u : a.f != b.f) return true;
                }
            return false;
        }

        // ---- GL objects -------------------------------------------------------------------------------------------
        static std::string with_defines(const std::string& source, int min_offset, int max_offset, int group)
        {
            const std::size_t eol = source.find('\n');
            return source.substr(0, eol + 1) + std::format("#define OFF_MIN ({})\n#define OFF_MAX ({})\n#define GROUP {}\n", min_offset, max_offset, group) +
                   source.substr(eol + 1);
        }

        void set_samplers(mgl::ProgramId program) const
        {
            mgl::glUseProgram(program);
            for (std::size_t n = 0; n < kSamplerNames.size(); ++n)
            {
                const mgl::UniformLocation loc = mgl::glGetUniformLocation(program, kSamplerNames[n]);
                if (loc.value >= 0) mgl::glUniform1i(loc, static_cast<GLint>(n));
            }
        }

        void build_programs(const demo::Env& env)
        {
            for (int group = 0; group < kGroups; ++group)
            {
                ProbeProgram& pr = probe_[static_cast<std::size_t>(group)];
                pr.id = demo::build_program_from_source(env.load_text("shaders/probe.vert"),
                                                        with_defines(env.load_text("shaders/probe.frag"), off_min_, off_max_, group),
                                                        std::format("probe.frag GROUP {}", group));
                const auto u = [&](const char* name) { return mgl::glGetUniformLocation(pr.id, name); };  // -1 when optimised away
                pr.kase = u("u_case");
                if (pr.kase.value < 0) throw demo::Failure("probe.frag: u_case not found");
                pr.origin = u("u_origin");
                pr.p = u("u_p");
                pr.ddx = u("u_ddx");
                pr.ddy = u("u_ddy");
                pr.gx = u("u_gx");
                pr.gy = u("u_gy");
                pr.lod = u("u_lod");
                pr.ip = u("u_ip");
                pr.ref = u("u_ref");
                set_samplers(pr.id);
            }

            // Vertex-stage program with one captured varying.
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, env.load_text("shaders/vertex_lookup.vert"), log);
            if (vs.value == 0) throw demo::Failure("vertex_lookup.vert failed to compile:\n" + log);
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, env.load_text("shaders/discard.frag"), log);
            if (fs.value == 0) throw demo::Failure("discard.frag failed to compile:\n" + log);
            vertex_.id = mgl::glCreateProgram();
            mgl::glAttachShader(vertex_.id, vs);
            mgl::glAttachShader(vertex_.id, fs);
            const std::array<const GLchar*, 1> varyings = {"v_result"};
            mgl::glTransformFeedbackVaryings(vertex_.id, 1, varyings.data(), mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(vertex_.id);
            GLint linked = GL_FALSE;
            mgl::glGetProgramiv(vertex_.id, mgl::ProgramParameter::LinkStatus, &linked);
            mgl::glDetachShader(vertex_.id, vs);
            mgl::glDetachShader(vertex_.id, fs);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (linked != GL_TRUE) throw demo::Failure("vertex_lookup program failed to link:\n" + demo::program_info_log(vertex_.id));
            vertex_.vp = mgl::glGetUniformLocation(vertex_.id, "u_vp");
            vertex_.vlod = mgl::glGetUniformLocation(vertex_.id, "u_vlod");
            vertex_.vip = mgl::glGetUniformLocation(vertex_.id, "u_vip");
            vertex_.vref = mgl::glGetUniformLocation(vertex_.id, "u_vref");
            set_samplers(vertex_.id);
            // The per-case inputs are uniform arrays indexed by gl_VertexID.
            std::array<float, 4 * kVertexCases> vp{};
            std::array<float, kVertexCases> vlod{};
            std::array<GLint, 4 * kVertexCases> vip{};
            std::array<float, 2 * kVertexCases> vref{};
            for (const std::size_t n : vs_cases_)
            {
                const Case& c = cases_[n];
                const auto s = static_cast<std::size_t>(c.slot);
                vp[4 * s] = static_cast<float>(c.p.x);
                vp[4 * s + 1] = static_cast<float>(c.p.y);
                vp[4 * s + 2] = static_cast<float>(c.p.z);
                vp[4 * s + 3] = static_cast<float>(c.p.w);
                vlod[s] = static_cast<float>(c.lod);
                for (std::size_t k = 0; k < 4; ++k) vip[4 * s + k] = c.ip[k];
                vref[2 * s] = c.refs[0][0];
                vref[2 * s + 1] = c.refs[0][1];
            }
            mgl::glUniform4fv(vertex_.vp, kVertexCases, vp.data());
            mgl::glUniform1fv(vertex_.vlod, kVertexCases, vlod.data());
            mgl::glUniform4iv(vertex_.vip, kVertexCases, vip.data());
            mgl::glUniform2fv(vertex_.vref, kVertexCases, vref.data());

            plane_.id = demo::build_program(env, "shaders/plane.vert", "shaders/plane.frag");
            plane_.mvp = mgl::glGetUniformLocation(plane_.id, "u_mvp");
            set_samplers(plane_.id);
            cell_.id = demo::build_program(env, "shaders/cell.vert", "shaders/cell.frag");
            cell_.block = mgl::glGetUniformLocation(cell_.id, "u_block");
            cell_.kind = mgl::glGetUniformLocation(cell_.id, "u_kind");
            cell_.rect = mgl::glGetUniformLocation(cell_.id, "u_rect");
            mgl::glUseProgram(cell_.id);
            mgl::glUniform1i(mgl::glGetUniformLocation(cell_.id, "u_results"), kResultsUnit);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glGenVertexArrays(1, &vao_);
        }

        static mgl::TextureTarget bind_target(Dim d)
        {
            switch (d)
            {
            case Dim::D2: return mgl::TextureTarget::Texture2D;
            case Dim::D3: return mgl::TextureTarget::Texture3D;
            case Dim::Array: return mgl::TextureTarget::Texture2DArray;
            case Dim::Cube: return mgl::TextureTarget::TextureCubeMap;
            }
            return mgl::TextureTarget::Texture2D;
        }

        void create_textures()
        {
            using IF = mgl::InternalFormat;
            using PF = mgl::PixelFormat;
            using PT = mgl::PixelType;
            static constexpr std::array<mgl::TextureTarget, 6> faces = {
                mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX, mgl::TextureTarget::TextureCubeMapPositiveY,
                mgl::TextureTarget::TextureCubeMapNegativeY, mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            for (int n = 0; n < kTexCount; ++n)
            {
                const TexInfo& T = kTex[static_cast<std::size_t>(n)];
                const mgl::TextureTarget target = bind_target(T.dim);
                mgl::glActiveTexture(kUnits[static_cast<std::size_t>(n)]);
                mgl::glBindTexture(target, tex_[static_cast<std::size_t>(n)]);
                IF internal = IF::Rgba8;
                PF format = PF::Rgba;
                PT type = PT::UnsignedByte;
                switch (T.fmt)
                {
                case Fmt::Rgba8: break;
                case Fmt::Rgba32ui: internal = IF::Rgba32UI; format = PF::RgbaInteger; type = PT::UnsignedInt; break;
                case Fmt::Rgba32i: internal = IF::Rgba32I; format = PF::RgbaInteger; type = PT::Int; break;
                case Fmt::Rgba8ui: internal = IF::Rgba8UI; format = PF::RgbaInteger; break;
                case Fmt::Depth32f: internal = IF::DepthComponent32F; format = PF::DepthComponent; type = PT::Float; break;
                }
                const int slices = T.dim == Dim::Cube ? 6 : 1;
                for (int level = 0; level < T.levels; ++level)
                {
                    const Dims D = level_dims(T, level);
                    for (int f = 0; f < slices; ++f)
                    {
                        std::vector<std::uint8_t> bytes;
                        for (int k = 0; k < D.d; ++k)
                            for (int j = 0; j < D.h; ++j)
                                for (int i = 0; i < D.w; ++i)
                                {
                                    const Texel t{level, i, j, T.dim == Dim::Cube ? f : k};
                                    if (T.fmt == Fmt::Rgba8)
                                    {
                                        const auto b = rgba8_of(t);
                                        bytes.insert(bytes.end(), b.begin(), b.end());
                                    }
                                    else if (T.fmt == Fmt::Depth32f)
                                    {
                                        const float d = depth_of(t);
                                        const auto b = std::bit_cast<std::array<std::uint8_t, 4>>(d);
                                        bytes.insert(bytes.end(), b.begin(), b.end());
                                    }
                                    else if (T.fmt == Fmt::Rgba8ui)
                                    {
                                        const Words w = uint_of(static_cast<Tex>(n), t);
                                        for (const std::uint32_t x : w) bytes.push_back(static_cast<std::uint8_t>(x));
                                    }
                                    else
                                    {
                                        const auto b = std::bit_cast<std::array<std::uint8_t, 16>>(uint_of(static_cast<Tex>(n), t));
                                        bytes.insert(bytes.end(), b.begin(), b.end());
                                    }
                                }
                        if (T.dim == Dim::D3 || T.dim == Dim::Array)
                            mgl::glTexImage3D(target, level, internal, D.w, D.h, D.d, 0, format, type, bytes.data());
                        else
                            mgl::glTexImage2D(T.dim == Dim::Cube ? faces[static_cast<std::size_t>(f)] : target, level, internal, D.w, D.h, 0, format, type,
                                              bytes.data());
                    }
                }
                // NEAREST_MIPMAP_NEAREST / NEAREST: one texel of one level per lookup, and legal for every format here
                // (integer and depth formats are not filterable, §3.8.13).
                mgl::glTexParameter(target, mgl::TextureMinFilter::NearestMipmapNearest);
                mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);
                mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapS, T.ws);
                mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapT, T.wt);
                mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapR, T.wr);
                if (T.fmt == Fmt::Depth32f)
                {
                    mgl::glTexParameter(target, mgl::TextureCompareMode::CompareRefToTexture);
                    mgl::glTexParameter(target, mgl::CompareFunc::Lequal);
                }
            }
        }

        void create_targets()
        {
            // The probe target: an RGBA32UI texture (colour-renderable in core ES 3.0) also sampled by the result cells.
            mgl::glGenTextures(1, &results_);
            mgl::glActiveTexture(kUnits[kResultsUnit]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, results_);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba32UI, kTargetW, kTargetH, 0, mgl::PixelFormat::RgbaInteger,
                              mgl::PixelType::UnsignedInt, nullptr);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glActiveTexture(kUnits[0]);
            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2D, results_, 0);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("RGBA32UI probe target incomplete: {}", demo::framebuffer_status_name(status)));

            mgl::glGenBuffers(1, &tf_buffer_);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tf_buffer_);
            mgl::glBufferData(mgl::BufferTarget::TransformFeedback, kVertexCases * 16, nullptr, mgl::BufferUsage::DynamicRead);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            mgl::glGenTransformFeedbacks(1, &tfo_);
        }

        // ---- rendering --------------------------------------------------------------------------------------------
        void run_probes()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glViewport(0, 0, kTargetW, kTargetH);
            const std::array<GLuint, 4> sentinel = {kSentinel, kSentinel, kSentinel, kSentinel};
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, sentinel.data());
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glBindVertexArray(vao_);
            for (const std::size_t n : fs_cases_)
            {
                const Case& c = cases_[n];
                const ProbeProgram& probe = probe_[static_cast<std::size_t>(group_of(c.id))];
                mgl::glUseProgram(probe.id);
                const int bx = (c.slot % kBlocksPerRow) * kBlock, by = (c.slot / kBlocksPerRow) * kBlock;
                mgl::glScissor(bx, by, kBlock, kBlock);
                const auto f = [](double d) { return static_cast<float>(d); };
                mgl::glUniform1i(probe.kase, c.id);
                mgl::glUniform2f(probe.origin, f(bx), f(by));
                mgl::glUniform4f(probe.p, f(c.p.x), f(c.p.y), f(c.p.z), f(c.p.w));
                mgl::glUniform4f(probe.ddx, f(c.ddx.x), f(c.ddx.y), f(c.ddx.z), f(c.ddx.w));
                mgl::glUniform4f(probe.ddy, f(c.ddy.x), f(c.ddy.y), f(c.ddy.z), f(c.ddy.w));
                mgl::glUniform4f(probe.gx, f(c.gx.x), f(c.gx.y), f(c.gx.z), f(c.gx.w));
                mgl::glUniform4f(probe.gy, f(c.gy.x), f(c.gy.y), f(c.gy.z), f(c.gy.w));
                mgl::glUniform1f(probe.lod, f(c.lod));
                mgl::glUniform4i(probe.ip, c.ip[0], c.ip[1], c.ip[2], c.ip[3]);
                mgl::glUniform2fv(probe.ref, 16, c.refs[0].data());
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void run_vertex_stage()
        {
            mgl::glUseProgram(vertex_.id);
            mgl::glBindVertexArray(vao_);
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, tfo_);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kVertexCases);
            mgl::glEndTransformFeedback();
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
            mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, mgl::TransformFeedbackId{0});
        }

        void draw_picture(const demo::Env& env)
        {
            const Layout l = make_layout(env.width, env.height, static_cast<int>(fs_cases_.size()));
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            // Mip-level visualisation: a ground plane in perspective sampled with texture().
            mgl::glViewport(l.plane.x, l.plane.y, l.plane.w, l.plane.h);
            using namespace demo::math;
            const Mat4 mvp = perspective(radians(55.0f), static_cast<float>(l.plane.w) / static_cast<float>(std::max(1, l.plane.h)), 0.1f, 200.0f) *
                             look_at({0.0f, 6.0f, 2.0f}, {0.0f, 0.0f, -8.0f}, {0.0f, 1.0f, 0.0f});
            mgl::glUseProgram(plane_.id);
            mgl::glUniformMatrix4fv(plane_.mvp, 1, GL_FALSE, mvp.data());
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);

            // One cell per lookup block.
            mgl::glUseProgram(cell_.id);
            for (std::size_t k = 0; k < fs_cases_.size(); ++k)
            {
                const Case& c = cases_[fs_cases_[k]];
                const Rect r = cell_rect(l, static_cast<int>(k));
                mgl::glViewport(r.x, r.y, r.w, r.h);
                mgl::glUniform2i(cell_.block, (c.slot % kBlocksPerRow) * kBlock, (c.slot / kBlocksPerRow) * kBlock);
                mgl::glUniform1i(cell_.kind, display_kind(c));
                mgl::glUniform4f(cell_.rect, static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.w), static_cast<float>(r.h));
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---- validation -------------------------------------------------------------------------------------------
        void check_limits(demo::Check& check) const
        {
            // ES 3.0 Table 6.32 (minimum maxima) and Table 6.28.
            check.expect(off_min_ <= -8, "MIN_PROGRAM_TEXEL_OFFSET <= -8", std::format("{}", off_min_));
            check.expect(off_max_ >= 7, "MAX_PROGRAM_TEXEL_OFFSET >= 7", std::format("{}", off_max_));
            check.expect(bias_max_ >= 2.0f, "MAX_TEXTURE_LOD_BIAS >= 2.0", std::format("{}", bias_max_));
            check.expect(demo::get_integer(mgl::GetParameter::MaxTextureImageUnits) >= 11, "MAX_TEXTURE_IMAGE_UNITS >= 11 (10 samplers + results)");
            std::cout << std::format("     MIN/MAX_PROGRAM_TEXEL_OFFSET = {}/{}, MAX_TEXTURE_LOD_BIAS = {}\n", off_min_, off_max_, bias_max_);
        }

        std::vector<Words> read_target() const
        {
            std::vector<Words> image(static_cast<std::size_t>(kTargetW * kTargetH));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glReadPixels(0, 0, kTargetW, kTargetH, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, image.data());
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return image;
        }

        // Compares one case (16 pixels or 1 vertex) with the model and runs its negative controls.
        void check_case(demo::Check& check, const Case& c, const std::vector<Words>& got)
        {
            const int n = c.vertex ? 1 : kBlock;
            const Fmt fmt = c.mode == Mode::Size ? Fmt::Rgba32ui : kTex[c.tex].fmt;
            int bad = 0;
            std::string first;
            for (int py = 0; py < n; ++py)
                for (int px = 0; px < n; ++px)
                {
                    const Words& w = got[static_cast<std::size_t>(py * n + px)];
                    const Expect e = expected(c, px, py, std::nullopt);
                    if (matches(e, w)) continue;
                    if (bad++ == 0)
                    {
                        std::string want = e.kind == Expect::Kind::Unorm ? std::format("({:.4f}, {:.4f}, {:.4f}, {:.4f})", e.f[0], e.f[1], e.f[2], e.f[3])
                                                                         : describe(e.u, fmt);
                        if (c.mode != Mode::Size) want += " = " + describe(model(c, px, py, std::nullopt, static_cast<double>(bias_max_)).t);
                        first = std::format("pixel ({}, {}): got {}, expected {}", px, py, describe(w, fmt), want);
                    }
                }
            const std::string where = c.vertex ? "vertex stage" : std::format("block {}", c.slot);
            check.expect(bad == 0, std::format("{} matches the CPU model ({})", c.fn, where), bad == 0 ? "" : std::format("{} mismatches; {}", bad, first));
            ++compared_;
            if (bad != 0) return;  // a control is only meaningful when the case itself passed
            for (const Mut m : c.controls)
            {
                int differ = 0;
                for (int py = 0; py < n; ++py)
                    for (int px = 0; px < n; ++px)
                        if (!matches(expected(c, px, py, m), got[static_cast<std::size_t>(py * n + px)])) ++differ;
                check.expect(differ > 0, std::format("negative control: the {} disagrees with {}", mut_name(m), c.fn),
                             std::format("{} of {} samples differ", differ, n * n));
                ++controls_;
            }
        }

        void check_fragment_cases(demo::Check& check)
        {
            const std::vector<Words> image = read_target();
            for (const std::size_t n : fs_cases_)
            {
                const Case& c = cases_[n];
                if (c.mode == Mode::Deriv) continue;
                check_case(check, c, pixels_of_block(image, c.slot));
            }
            image_ = image;
        }

        void check_vertex_cases(demo::Check& check)
        {
            std::vector<Words> captured(kVertexCases);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tf_buffer_);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::TransformFeedback, 0, kVertexCases * 16, mgl::MapBufferAccessMask::Read);
            const bool ok = mapped != nullptr;
            if (ok)
            {
                std::memcpy(captured.data(), mapped, kVertexCases * 16);
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::TransformFeedback);
            }
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            if (!check.expect(ok, "transform feedback buffer of the vertex-stage lookups can be mapped for reading")) return;
            for (const std::size_t n : vs_cases_)
            {
                const Case& c = cases_[n];
                check_case(check, c, {captured[static_cast<std::size_t>(c.slot)]});
            }
            std::cout << std::format("     {} lookup cases compared with the CPU model, {} negative controls\n", compared_, controls_);
        }

        void check_derivatives(demo::Check& check)
        {
            // §8.9: forward/backward differencing; for a function that is affine in window coordinates every such
            // difference is exactly the slope. The spec gives no precision, so the tolerance is a sanity bound
            // (1e-3 relative, larger than float rounding of the inputs) and the measured values are printed.
            std::array<std::vector<std::array<float, 4>>, 6> measured;
            for (const std::size_t n : fs_cases_)
            {
                const Case& c = cases_[n];
                if (c.mode != Mode::Deriv) continue;
                const std::vector<Words> px = pixels_of_block(image_, c.slot);
                auto& m = measured[static_cast<std::size_t>(c.id - 110)];
                for (const Words& w : px) m.push_back({as_float(w[0]), as_float(w[1]), as_float(w[2]), as_float(w[3])});
                if (c.id == 115) continue;
                const double tol_rel = c.id == 113 ? 1e-2 : 1e-3;  // 113 goes through attribute interpolation
                const std::array<double, 4> want = deriv_expect(c);
                int bad = 0;
                double worst = 0;
                for (const auto& v : m)
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        const double dev = std::abs(static_cast<double>(v[k]) - want[k]);
                        worst = std::max(worst, dev);
                        if (dev > tol_rel * std::max(1.0, std::abs(want[k]))) ++bad;
                    }
                check.expect(bad == 0, std::format("{} equals the slopes ({}, {}, {}, {})", c.fn, want[0], want[1], want[2], want[3]),
                             std::format("{} components off, largest deviation {:.3g}", bad, worst));
                std::cout << std::format("     {}: pixel (0,0) = ({}, {}, {}, {}), largest deviation {:.3g}\n", c.fn, m[0][0], m[0][1], m[0][2], m[0][3], worst);
            }
            const auto& dx = measured[0];
            const auto& dy = measured[1];
            const auto& fw = measured[2];
            if (dx.size() == 16 && dy.size() == 16 && fw.size() == 16)
            {
                // fwidth(p) = abs(dFdx(p)) + abs(dFdy(p)) (§8.9), here with the measured dFdx / dFdy of the same pixels.
                int bad = 0;
                for (std::size_t i = 0; i < 16; ++i)
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        const double sum = std::abs(static_cast<double>(dx[i][k])) + std::abs(static_cast<double>(dy[i][k]));
                        if (std::abs(static_cast<double>(fw[i][k]) - sum) > 1e-3 * std::max(1.0, sum)) ++bad;
                    }
                check.expect(bad == 0, "fwidth = |dFdx| + |dFdy| of the measured derivatives", std::format("{} components differ", bad));
                // Negative control: dFdx and dFdy of this function differ, so swapping them must be detected.
                int same = 0;
                const std::array<double, 4> want_dy = {-5.0, 0.5, 0.75, 0.25};
                for (const auto& v : dx)
                    if (std::abs(static_cast<double>(v[0]) - want_dy[0]) < 1e-3 && std::abs(static_cast<double>(v[1]) - want_dy[1]) < 1e-3) ++same;
                check.expect(same == 0, "negative control: the dFdx results do not match the dFdy slopes", std::format("{} pixels match", same));
            }
            // Informational: dFdx(x^2) is 2x + 1 (forward), 2x - 1 (backward), or one of them for a whole 2x2 quad
            // (coarse). §8.9 allows all of these, so nothing is asserted.
            const auto& sq = measured[5];
            if (sq.size() == 16)
            {
                std::string row;
                for (std::size_t i = 0; i < 4; ++i)
                    row += std::format(" x={}: {}", sq[i][2], static_cast<double>(sq[i][0]) - 2.0 * static_cast<double>(sq[i][2]));
                std::cout << std::format("     dFdx(x^2) - 2x on the bottom row of its block:{} (+1 = forward difference, -1 = backward; "
                                         "alternating +1/-1 = one difference per pixel pair of a 2x2 quad)\n",
                                         row);
            }
        }

        void check_compile_rules(demo::Check& check) const
        {
            // Programs that must not compile, each with a positive control that differs only in the rule.
            struct Rule
            {
                const char* what;
                mgl::ShaderType type;
                const char* bad;
                const char* good;
            };
            static constexpr const char* kVsHead = "#version 300 es\nuniform highp sampler2D s;\nuniform highp samplerCube c;\nuniform int i;\nout vec4 o;\nvoid main() {\n";
            static constexpr const char* kFsHead =
                "#version 300 es\nprecision highp float;\nuniform highp sampler2D s;\nuniform highp samplerCube c;\nuniform int i;\nout vec4 o;\nvoid main() {\n";
            const std::array<Rule, 5> rules = {{
                {"texture() with a bias argument in a vertex shader (§8.8: 'The bias parameter is not accepted in a vertex shader')",
                 mgl::ShaderType::Vertex, "o = texture(s, vec2(0.5), 1.0); gl_Position = o; }\n", "o = texture(s, vec2(0.5)); gl_Position = o; }\n"},
                {"dFdx in a vertex shader (§8.9: only available in fragment shaders)", mgl::ShaderType::Vertex,
                 "o = vec4(dFdx(float(i))); gl_Position = o; }\n", "o = vec4(float(i)); gl_Position = o; }\n"},
                {"textureOffset with a non-constant offset (§8.8: 'The offset value must be a constant expression')", mgl::ShaderType::Fragment,
                 "o = textureOffset(s, vec2(0.5), ivec2(i, 0)); }\n", "o = textureOffset(s, vec2(0.5), ivec2(1, 0)); }\n"},
                {"textureOffset on a samplerCube (§8.8: 'texel offsets are also not supported for cube maps')", mgl::ShaderType::Fragment,
                 "o = textureOffset(c, vec3(1.0), ivec2(1, 0)); }\n", "o = texture(c, vec3(1.0)); }\n"},
                {"texelFetch on a samplerCube (§8.8 defines it for 2D, 3D and 2D array samplers only)", mgl::ShaderType::Fragment,
                 "o = texelFetch(c, ivec3(0), 0); }\n", "o = texelFetch(s, ivec2(0), 0); }\n"},
            }};
            for (const Rule& r : rules)
            {
                const std::string head = r.type == mgl::ShaderType::Vertex ? kVsHead : kFsHead;
                std::string log_bad, log_good;
                const mgl::ShaderId bad = demo::compile_shader(r.type, head + r.bad, log_bad);
                const mgl::ShaderId good = demo::compile_shader(r.type, head + r.good, log_good);
                check.expect(bad.value == 0, std::format("rejected at compile time: {}", r.what));
                check.expect(good.value != 0, std::format("positive control compiles: {}", r.what), log_good);
                if (bad.value != 0) mgl::glDeleteShader(bad);
                if (good.value != 0) mgl::glDeleteShader(good);
            }
        }

        void check_picture(const demo::Env& env, demo::Check& check) const
        {
            // Each texel cell shows the palette colour of the level its block returned, with a parity shade. One pixel per
            // cell (inside texel (1, 1) of the block, away from the frame) is compared with the model's texel.
            const Layout l = make_layout(env.width, env.height, static_cast<int>(fs_cases_.size()));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            int probes = 0, bad = 0;
            std::string first;
            for (std::size_t k = 0; k < fs_cases_.size(); ++k)
            {
                const Case& c = cases_[fs_cases_[k]];
                const int kind = display_kind(c);
                if (kind > 2) continue;
                const Rect r = cell_rect(l, static_cast<int>(k));
                // A pixel whose block coordinate local * 4 is well inside texel 1 in x and y.
                auto pick = [](int size) {
                    for (int p = 1; p < size - 1; ++p)
                    {
                        const double v = (p + 0.5) * 4.0 / size;
                        if (v > 1.15 && v < 1.85) return p;
                    }
                    return -1;
                };
                const int ox = pick(r.w), oy = pick(r.h);
                if (ox < 0 || oy < 0) continue;
                const Texel t = model(c, 1, 1, std::nullopt, static_cast<double>(bias_max_)).t;
                const auto& pal = kPalette[static_cast<std::size_t>(std::clamp(t.level, 0, 7))];
                const double shade = ((t.i + t.j) & 1) == 0 ? 1.0 : 0.75;
                demo::Rgba8 want{};
                for (std::size_t ch = 0; ch < 3; ++ch) want[ch] = static_cast<std::uint8_t>(std::lround(pal[ch] * shade * 255.0));
                want[3] = 255;
                const demo::Rgba8 got = demo::read_pixel(r.x + ox, r.y + oy);
                ++probes;
                int dev = 0;
                for (std::size_t ch = 0; ch < 4; ++ch) dev = std::max(dev, std::abs(int{got[ch]} - int{want[ch]}));
                if (dev > 2 && bad++ == 0)
                    first = std::format("{}: got {} want {}", c.fn, demo::format_rgba(got), demo::format_rgba(want));
            }
            check.expect(probes > 0 && bad == 0, std::format("the result cells show the model's level colours ({} cells probed)", probes),
                         bad == 0 ? "" : std::format("{} cells differ; {}", bad, first));
        }

        std::vector<Case> cases_;
        std::vector<std::size_t> fs_cases_, vs_cases_;
        std::vector<Words> image_;
        int off_min_ = 0, off_max_ = 0;
        float bias_max_ = 0.0f;
        int compared_ = 0, controls_ = 0;

        std::array<ProbeProgram, kGroups> probe_;
        VertexProgram vertex_;
        PlaneProgram plane_;
        CellProgram cell_;
        std::array<mgl::TextureId, kTexCount> tex_{};
        mgl::TextureId results_{};
        mgl::FramebufferId fbo_{};
        mgl::BufferId tf_buffer_{};
        mgl::TransformFeedbackId tfo_{};
        mgl::VertexArrayId vao_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-glsl-texture-functions",
        .title = "GLSL ES 3.00 texture lookup and derivative functions",
        .min_es = {3, 0},
        .summary = "Every GLSL ES 3.00 texture lookup form (texture with bias, Lod, Grad, Offset, Proj and their combinations, texelFetch(Offset), "
                   "textureSize) on 2D, 3D, 2D array, cube, integer and shadow samplers, in the fragment and the vertex stage, compared texel by "
                   "texel with a CPU model of ES 3.0 level selection; dFdx/dFdy/fwidth of affine functions.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<GlslTextureFunctions>(argc, argv, info);
}
