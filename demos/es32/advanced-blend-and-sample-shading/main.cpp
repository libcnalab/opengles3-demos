// es32-advanced-blend-and-sample-shading
//
// Two ES 3.2 fragment-stage features that were extensions in ES 3.1 (KHR_blend_equation_advanced, OES_sample_shading,
// OES_sample_variables, OES_shader_multisample_interpolation):
//
//   1. Advanced blend equations (ES 3.2 section 15.1.5.3, Tables 15.3 and 15.4). The fifteen equations MULTIPLY ...
//      HSL_LUMINOSITY combine a premultiplied source and destination as
//          RGB = f(Cs', Cd') As Ad + Cs' As (1 - Ad) + Cd' Ad (1 - As),     A = As + Ad - As Ad
//      where Cs' and Cd' are the colors divided by their alpha. A fragment shader must declare the matching
//      layout(blend_support_*) out qualifier (GLSL ES 3.20 section 4.4.2), only one color buffer may be written, and
//      blending into a sample that the same "pass" already touched is undefined: glBlendBarrier separates passes.
//   2. Sample shading and per-sample interpolation (ES 3.2 sections 13.4, 13.4.1 and 14.1; GLSL ES 3.20 sections
//      4.5, 7.1.5 and 8.14.2): glMinSampleShading + SAMPLE_SHADING, gl_SampleID, gl_SamplePosition, gl_SampleMaskIn,
//      gl_NumSamples, the "sample" qualifier and interpolateAtCentroid / interpolateAtSample / interpolateAtOffset.
//
// Image (fills the window at any size):
//   * top 5/8: the advanced-blend grid. 4 x 4 cells (each 64 x 40 texels of an RGBA8 texture); every cell has the
//     same premultiplied gradient backdrop with alpha 0.45 .. 0.95, then disc A, glBlendBarrier, then disc B,
//     both with alpha 0.45 .. 0.9. Cells 0 .. 14 use MULTIPLY, SCREEN, OVERLAY, DARKEN, LIGHTEN, COLORDODGE,
//     COLORBURN, HARDLIGHT, SOFTLIGHT, DIFFERENCE, EXCLUSION, HSL_HUE, HSL_SATURATION, HSL_COLOR, HSL_LUMINOSITY
//     (row by row from the top left); cell 15 is the control: BLEND disabled, the discs simply replace. The
//     premultiplied result is shown over a checkerboard, so the alpha is visible.
//   * bottom 3/8: three panels of a 32 x 32 multisample texture with concentric rings that get thinner than a pixel:
//     left, shaded once per pixel (aliased); middle, the same with SAMPLE_SHADING and MinSampleShading(1.0)
//     (antialiased by the per-sample evaluation); right, every texel of the middle texture split into one sub-cell
//     per sample.
//
// Validation (see README.md for the tolerance arguments):
//   * every texel of the 16 cells against a CPU implementation of section 15.1.5.3 (Tables 15.3, 15.4 and the HSL
//     pseudocode), the backdrop read back from the same framebuffer, the blend functions set to (ZERO,
//     ONE_MINUS_SRC_COLOR) to show that advanced equations ignore them; a negative control compares each cell with the
//     reference of the next equation;
//   * the INVALID_OPERATION of section 15.1.5.3 (no matching blend_support qualifier, a qualifier for another equation,
//     two draw buffers, draw buffer 0 NONE and buffer 1 used) and its absence when blending is disabled, when the
//     equation is FUNC_ADD, with blend_support_all_equations and with draw buffers {COLOR0, NONE}; the INVALID_ENUM of
//     BlendEquationSeparate with an advanced equation (section 15.1.5.1); BLEND_EQUATION_RGB / _ALPHA for all 15;
//   * per-sample values read with texelFetch(sampler2DMS) in a second pass: gl_SampleID, gl_NumSamples,
//     gl_SampleMaskIn, gl_SamplePosition against glGetMultisamplefv, "sample in" inputs and MinSampleShading(1.0) at
//     the sample positions, MinSampleShading(0.5) giving at least ceil(0.5 N) distinct inputs, interpolateAtSample for
//     every sample, interpolateAtOffset up to the MIN/MAX_FRAGMENT_INTERPOLATION_OFFSET limits, interpolateAtCentroid
//     equal to a "centroid in" input and inside pixel and triangle; the ring pattern per sample; the state of
//     SAMPLE_SHADING and MIN_SAMPLE_SHADING_VALUE (initial values, clamping to [0, 1]);
//   * every window pixel of the grid region against the CPU composite of the grid texel it shows.
//
// Known deviation: radeonsi (Mesa 25.0.7) does not shade per sample when the draw follows a single-sample draw without a
// flush in between (kRadeonsiSampleShading). The demo checks that order strictly (XFAIL on radeonsi) and also with glFlush;
// the displayed rings are drawn after a glFlush so that the picture is right on that driver too.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    using BE = mgl::BlendEquation;

    constexpr int kGridW = 256;      // advanced-blend target
    constexpr int kGridH = 160;
    constexpr int kCellW = 64;
    constexpr int kCellH = 40;
    constexpr int kGridCols = 4;
    constexpr int kControlCell = 15;
    constexpr int kProbe = 16;       // edge of the multisample texture that the probes are drawn into
    constexpr int kVis = 32;         // edge of the two displayed multisample textures
    constexpr int kMaxSamples = 8;   // gl_SampleMaskIn[0] is stored in 8 bits
    constexpr int kStripW = kVis * kMaxSamples;
    constexpr int kScratch = 8;      // error-test framebuffer
    constexpr double kHalfLsb = 0.5 / 255.0;

    constexpr std::string_view kRadeonsiSampleShading =
        "radeonsi (Mesa 25.0.7): a draw with SAMPLE_SHADING and MinSampleShading(1.0) that follows a draw into a single-sample framebuffer "
        "without an intervening flush is shaded once per pixel (every sample receives the inputs at sample 0's location), against ES 3.2 "
        "section 13.4.1; with glFlush in between it is shaded per sample";

    struct ModeInfo
    {
        BE eq;
        std::string_view name;
        std::string_view layout;
    };

    constexpr std::array<ModeInfo, 15> kModes = {{
        {BE::Multiply, "MULTIPLY", "blend_support_multiply"},
        {BE::Screen, "SCREEN", "blend_support_screen"},
        {BE::Overlay, "OVERLAY", "blend_support_overlay"},
        {BE::Darken, "DARKEN", "blend_support_darken"},
        {BE::Lighten, "LIGHTEN", "blend_support_lighten"},
        {BE::ColorDodge, "COLORDODGE", "blend_support_colordodge"},
        {BE::ColorBurn, "COLORBURN", "blend_support_colorburn"},
        {BE::HardLight, "HARDLIGHT", "blend_support_hardlight"},
        {BE::SoftLight, "SOFTLIGHT", "blend_support_softlight"},
        {BE::Difference, "DIFFERENCE", "blend_support_difference"},
        {BE::Exclusion, "EXCLUSION", "blend_support_exclusion"},
        {BE::HslHue, "HSL_HUE", "blend_support_hsl_hue"},
        {BE::HslSaturation, "HSL_SATURATION", "blend_support_hsl_saturation"},
        {BE::HslColor, "HSL_COLOR", "blend_support_hsl_color"},
        {BE::HslLuminosity, "HSL_LUMINOSITY", "blend_support_hsl_luminosity"},
    }};
    constexpr int kModeCount = static_cast<int>(kModes.size());

    constexpr GLint value_of(BE e) { return static_cast<GLint>(static_cast<GLenum>(e)); }

    struct Disc
    {
        double cx, cy, r;   // relative to the cell, in pixels
    };
    constexpr Disc kDiscA{25.0, 22.0, 14.0};
    constexpr Disc kDiscB{40.0, 17.0, 13.0};

    /// Cell i in the grid texture: row 0 of the cells is the top row of the picture (GL rows count upwards).
    constexpr int cell_x(int i) { return (i % kGridCols) * kCellW; }
    constexpr int cell_y(int i) { return (3 - i / kGridCols) * kCellH; }

    // ---------------------------------------------------------------------------------------------
    // CPU model of the shaders (backdrop.frag, source.frag) and of ES 3.2 section 15.1.5.3.
    // ---------------------------------------------------------------------------------------------
    using V3 = std::array<double, 3>;
    using V4 = std::array<double, 4>;   // premultiplied RGBA in [0, 1]

    double glsl_mod(double x, double y) { return x - y * std::floor(x / y); }
    double fract(double x) { return x - std::floor(x); }
    double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }

    V3 hsv(double h, double s, double v)
    {
        constexpr std::array<double, 3> off = {0.0, 4.0, 2.0};
        V3 c{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            const double k = std::clamp(std::abs(glsl_mod(h * 6.0 + off[i], 6.0) - 3.0) - 1.0, 0.0, 1.0);
            c[i] = v * (1.0 + (k - 1.0) * s);   // v * mix(1, k, s)
        }
        return c;
    }

    V4 premul(const V3& base, double a) { return {base[0] * a, base[1] * a, base[2] * a, a}; }

    /// Premultiplied source color of disc 0 (A) or 1 (B) at cell-local position (u, v) in [0, 1].
    V4 source_color(int disc, double u, double v)
    {
        if (disc == 0) return premul(hsv(fract(0.55 + 0.35 * v), 0.85, 0.3 + 0.7 * u), 0.9 - 0.45 * v);
        return premul(hsv(fract(0.05 + 0.15 * u), 0.9, 1.0 - 0.6 * v), 0.5 + 0.4 * u);
    }

    /// f(Cs, Cd) of Table 15.3 for the separable equations 0 .. 10.
    double separable(int m, double s, double d)
    {
        switch (m)
        {
            case 0: return s * d;
            case 1: return s + d - s * d;
            case 2: return d <= 0.5 ? 2.0 * s * d : 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
            case 3: return std::min(s, d);
            case 4: return std::max(s, d);
            case 5:
                if (d <= 0.0) return 0.0;
                if (s < 1.0) return std::min(1.0, d / (1.0 - s));
                return 1.0;
            case 6:
                if (d >= 1.0) return 1.0;
                if (s > 0.0) return 1.0 - std::min(1.0, (1.0 - d) / s);
                return 0.0;
            case 7: return s <= 0.5 ? 2.0 * s * d : 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
            case 8:
                if (s <= 0.5) return d - (1.0 - 2.0 * s) * d * (1.0 - d);
                if (d <= 0.25) return d + (2.0 * s - 1.0) * d * ((16.0 * d - 12.0) * d + 3.0);
                return d + (2.0 * s - 1.0) * (std::sqrt(d) - d);
            case 9: return std::abs(d - s);
            default: return s + d - 2.0 * s * d;
        }
    }

    // The HSL pseudocode of section 15.1.5.3, transcribed.
    double minv3(const V3& c) { return std::min(std::min(c[0], c[1]), c[2]); }
    double maxv3(const V3& c) { return std::max(std::max(c[0], c[1]), c[2]); }
    double lumv3(const V3& c) { return 0.30 * c[0] + 0.59 * c[1] + 0.11 * c[2]; }
    double satv3(const V3& c) { return maxv3(c) - minv3(c); }

    V3 clip_color(V3 color)
    {
        const double lum = lumv3(color);
        const double mincol = minv3(color);
        const double maxcol = maxv3(color);
        if (mincol < 0.0)
            for (double& c : color) c = lum + ((c - lum) * lum) / (lum - mincol);
        if (maxcol > 1.0)
            for (double& c : color) c = lum + ((c - lum) * (1.0 - lum)) / (maxcol - lum);
        return color;
    }

    V3 set_lum(const V3& cbase, const V3& clum)
    {
        const double ldiff = lumv3(clum) - lumv3(cbase);
        return clip_color({cbase[0] + ldiff, cbase[1] + ldiff, cbase[2] + ldiff});
    }

    V3 set_lum_sat(const V3& cbase, const V3& csat, const V3& clum)
    {
        const double minbase = minv3(cbase);
        const double sbase = satv3(cbase);
        const double ssat = satv3(csat);
        V3 color{};
        if (sbase > 0.0)
            for (std::size_t i = 0; i < 3; ++i) color[i] = (cbase[i] - minbase) * ssat / sbase;
        return set_lum(color, clum);
    }

    /// The blend result of equation m for base (non-premultiplied) colors and alphas, premultiplied.
    V4 advanced(int m, const V3& cs, double as, const V3& cd, double ad)
    {
        V3 f{};
        if (m <= 10)
            for (std::size_t i = 0; i < 3; ++i) f[i] = separable(m, cs[i], cd[i]);
        else if (m == 11) f = set_lum_sat(cs, cd, cd);
        else if (m == 12) f = set_lum_sat(cd, cs, cd);
        else if (m == 13) f = set_lum(cs, cd);
        else f = set_lum(cd, cs);
        const double p0 = as * ad, p1 = as * (1.0 - ad), p2 = ad * (1.0 - as);   // (X, Y, Z) = (1, 1, 1) for all 15
        return {f[0] * p0 + cs[0] * p1 + cd[0] * p2, f[1] * p0 + cs[1] * p1 + cd[1] * p2, f[2] * p0 + cs[2] * p1 + cd[2] * p2, p0 + p1 + p2};
    }

    V3 unpremul(const V4& c)
    {
        if (c[3] <= 0.0) return {0.0, 0.0, 0.0};
        return {c[0] / c[3], c[1] / c[3], c[2] / c[3]};
    }

    V4 advanced_premul(int m, const V4& s, const V4& d) { return advanced(m, unpremul(s), s[3], unpremul(d), d[3]); }

    V4 unit(const demo::Rgba8& c)
    {
        return {c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, c[3] / 255.0};
    }

    /// Range of the result of equation m when the base colors and the source alpha are known only to half an 8-bit
    /// step (section 15.1.5.1 allows "the smaller of" the destination precision (8 bits here) and 16-bit float), and
    /// the destination lies anywhere in [dlo, dhi] (premultiplied). Evaluated at the 256 corners of the input box and
    /// at the exact point.
    void advanced_range(int m, const V4& src, const V4& dlo, const V4& dhi, const V4& dcenter, V4& lo, V4& hi)
    {
        const V3 cs = unpremul(src);
        std::array<std::array<double, 2>, 8> box{};
        for (std::size_t i = 0; i < 3; ++i) box[i] = {clamp01(cs[i] - kHalfLsb), clamp01(cs[i] + kHalfLsb)};
        box[3] = {clamp01(src[3] - kHalfLsb), clamp01(src[3] + kHalfLsb)};
        for (std::size_t i = 0; i < 3; ++i)
        {
            const double l = dhi[3] > 0.0 ? dlo[i] / dhi[3] : 0.0;
            const double h = dlo[3] > 0.0 ? dhi[i] / dlo[3] : 1.0;
            box[4 + i] = {clamp01(l - kHalfLsb), clamp01(h + kHalfLsb)};
        }
        box[7] = {dlo[3], dhi[3]};
        lo = hi = advanced_premul(m, src, dcenter);
        for (unsigned corner = 0; corner < 256; ++corner)
        {
            std::array<double, 8> p{};
            for (std::size_t i = 0; i < 8; ++i) p[i] = box[i][(corner >> i) & 1u];
            const V4 r = advanced(m, {p[0], p[1], p[2]}, p[3], {p[4], p[5], p[6]}, p[7]);
            for (std::size_t i = 0; i < 4; ++i)
            {
                lo[i] = std::min(lo[i], r[i]);
                hi[i] = std::max(hi[i], r[i]);
            }
        }
    }

    /// Coverage of a disc for the pixel centre (px, py) in cell-local pixels: 0 outside, 1 inside, 2 on the
    /// boundary (the shader's float length() may decide either way: both outcomes are accepted).
    int disc_cover(const Disc& d, double px, double py)
    {
        const double dist = std::hypot(px - d.cx, py - d.cy);
        if (std::abs(dist - d.r) < 1e-3) return 2;
        return dist <= d.r ? 1 : 0;
    }

    struct PixelVerdict
    {
        bool ok = false;
        double dev = 0.0;      // largest |got - exact| over the channels (8-bit units)
        bool used_band = false;
    };

    /// Is `got` a valid result of: destination dst0, then disc A (if a) blended with equation m, glBlendBarrier, then
    /// disc B (if b)? Exact reference +-1 first; otherwise the precision range of advanced_range() +-1 per step.
    PixelVerdict consistent(int m, bool a, bool b, const V4& sa, const V4& sb, const demo::Rgba8& dst0, const demo::Rgba8& got)
    {
        PixelVerdict v;
        const V4 d0 = unit(dst0);
        V4 c = d0;
        if (a) c = advanced_premul(m, sa, c);
        if (b) c = advanced_premul(m, sb, c);
        for (std::size_t i = 0; i < 4; ++i) v.dev = std::max(v.dev, std::abs(got[i] - c[i] * 255.0));
        if (v.dev <= 1.0)
        {
            v.ok = true;
            return v;
        }
        if (!a && !b) return v;
        v.used_band = true;
        V4 lo = d0, hi = d0, center = d0;
        auto step = [&](const V4& s) {
            V4 rlo{}, rhi{};
            advanced_range(m, s, lo, hi, center, rlo, rhi);
            center = advanced_premul(m, s, center);
            for (std::size_t i = 0; i < 4; ++i)
            {
                // a stored 8-bit result: rounding (+-1 step) of a value in [rlo, rhi]
                lo[i] = std::max(0.0, std::ceil(rlo[i] * 255.0 - 1.0)) / 255.0;
                hi[i] = std::min(255.0, std::floor(rhi[i] * 255.0 + 1.0)) / 255.0;
            }
        };
        if (a) step(sa);
        if (b) step(sb);
        v.ok = true;
        for (std::size_t i = 0; i < 4; ++i)
            if (got[i] < std::lround(lo[i] * 255.0) || got[i] > std::lround(hi[i] * 255.0)) v.ok = false;
        return v;
    }

    std::string fmt4(const V4& c) { return std::format("({:.1f}, {:.1f}, {:.1f}, {:.1f})", c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0); }

    /// 16-bit position encoding of probe.frag.
    double decode(std::uint8_t hi, std::uint8_t lo) { return (hi + lo / 255.0) / 255.0; }

    std::string replace_marker(std::string source, std::string_view marker, std::string_view text)
    {
        const std::size_t at = source.find(marker);
        if (at == std::string::npos) throw demo::Failure(std::format("marker {} missing in a shader", marker));
        source.replace(at, marker.size(), text);
        return source;
    }

    std::string with_define(std::string source, std::string_view define)
    {
        source.insert(source.find('\n') + 1, std::format("#define {}\n", define));
        return source;
    }

    struct Program
    {
        mgl::ProgramId id{};
        mgl::UniformLocation corner{}, size{}, cell{}, origin{}, center{}, radius{}, disc{};
        mgl::UniformLocation scale{}, index{}, offset{};
        mgl::UniformLocation ms{}, texels{}, grid{}, mode{}, extent{}, samples{}, cols{};
    };

    Program make_program(const std::string& vs, const std::string& fs, std::string_view label)
    {
        Program p;
        p.id = demo::build_program_from_source(vs, fs, label);
        auto loc = [&](const char* n) { return mgl::glGetUniformLocation(p.id, n); };
        p.corner = loc("u_corner");
        p.size = loc("u_size");
        p.cell = loc("u_cell");
        p.origin = loc("u_origin");
        p.center = loc("u_center");
        p.radius = loc("u_radius");
        p.disc = loc("u_disc");
        p.scale = loc("u_scale");
        p.index = loc("u_index");
        p.offset = loc("u_offset");
        p.ms = loc("u_ms");
        p.texels = loc("u_texels");
        p.grid = loc("u_grid");
        p.mode = loc("u_mode");
        p.extent = loc("u_extent");
        p.samples = loc("u_samples");
        p.cols = loc("u_cols");
        return p;
    }

    /// Sets u_corner to the rectangle [x0, x1] x [y0, y1] (TRIANGLE_STRIP order) and u_size.
    void set_rect(const Program& p, double x0, double y0, double x1, double y1, double w, double h)
    {
        const std::array<float, 8> c = {static_cast<float>(x0), static_cast<float>(y0), static_cast<float>(x1), static_cast<float>(y0),
                                        static_cast<float>(x0), static_cast<float>(y1), static_cast<float>(x1), static_cast<float>(y1)};
        mgl::glUniform2fv(p.corner, 4, c.data());
        mgl::glUniform2f(p.size, static_cast<float>(w), static_cast<float>(h));
    }

    struct Pos
    {
        double x = 0.0, y = 0.0;
    };

    /// Triangle of the centroid test, in pixels of the 16 x 16 probe texture.
    constexpr std::array<Pos, 3> kTriangle = {{{1.3, 2.2}, {15.1, 5.7}, {6.4, 14.6}}};

    /// Signed distance to edge i of kTriangle (positive inside; the corners are counter-clockwise).
    double edge_distance(std::size_t i, double x, double y)
    {
        const Pos a = kTriangle[i], b = kTriangle[(i + 1) % 3];
        return ((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)) / std::hypot(b.x - a.x, b.y - a.y);
    }

    class AdvancedBlendAndSampleShading final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 96 || env.height < 96) throw demo::Skip("this demo needs a default framebuffer of at least 96 x 96 pixels");

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glDisable(mgl::Capability::Dither);   // 8-bit results are compared with a CPU model

            // Initial state of section 13.4.1 (Table 21.8: SAMPLE_SHADING FALSE, MIN_SAMPLE_SHADING_VALUE 0).
            initial_sample_shading_ = mgl::glIsEnabled(mgl::Capability::SampleShading);
            mgl::glGetFloatv(mgl::GetParameter::MinSampleShadingValue, &initial_min_sample_shading_);

            build_programs(env);
            create_grid_targets();
            create_multisample_targets();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            draw_grid();
            grid_error_ = demo::drain_gl_errors();
            if (ms_ok_)
            {
                draw_rings(vis_fbo_[0], false);
                draw_rings(vis_fbo_[1], true);
                rings_error_ = demo::drain_gl_errors();
            }
            draw_window(env);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            validate_grid(env, check);
            validate_blend_errors(check);
            validate_sample_state(check);
            if (ms_ok_)
                validate_multisample(check);
            else
                std::cout << "  multisample checks skipped: " << ms_reason_ << "\n";
            check.expect_no_gl_error("no GL error after validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::SampleShading);
            mgl::glBlendEquation(BE::FuncAdd);
            mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::Zero);
            for (Program* p : all_programs())
                if (p->id.value != 0) mgl::glDeleteProgram(p->id);
            for (mgl::FramebufferId* f : {&grid_fbo_, &scratch_fbo_, &probe_fbo_, &vis_fbo_[0], &vis_fbo_[1], &strip_fbo_})
                if (f->value != 0) mgl::glDeleteFramebuffers(1, f);
            for (mgl::TextureId* t : {&grid_tex_, &scratch_tex_[0], &scratch_tex_[1], &probe_tex_, &vis_tex_[0], &vis_tex_[1], &strip_tex_})
                if (t->value != 0) mgl::glDeleteTextures(1, t);
            mgl::glDeleteVertexArrays(1, &vao_);
        }

    private:
        // ==========================================================================================
        // Setup
        // ==========================================================================================
        std::vector<Program*> all_programs()
        {
            std::vector<Program*> v = {&backdrop_, &source_none_, &source_all_, &rings_, &fetch_, &composite_};
            for (Program& p : source_mode_) v.push_back(&p);
            for (Program& p : probe_) v.push_back(&p);
            return v;
        }

        void build_programs(const demo::Env& env)
        {
            const std::string vs = env.load_text("shaders/quad.vert");
            const std::string source = env.load_text("shaders/source.frag");
            constexpr std::string_view marker = "//@BLEND_SUPPORT@";
            backdrop_ = make_program(vs, env.load_text("shaders/backdrop.frag"), "backdrop");
            for (int m = 0; m < kModeCount; ++m)
            {
                const ModeInfo& info = kModes[static_cast<std::size_t>(m)];
                source_mode_[static_cast<std::size_t>(m)] =
                    make_program(vs, replace_marker(source, marker, std::format("layout({}) out;", info.layout)), std::format("source {}", info.layout));
            }
            source_all_ = make_program(vs, replace_marker(source, marker, "layout(blend_support_all_equations) out;"), "source blend_support_all_equations");
            source_none_ = make_program(vs, replace_marker(source, marker, "// (no blend_support qualifier)"), "source without blend_support");
            rings_ = make_program(vs, env.load_text("shaders/pattern.frag"), "rings");
            fetch_ = make_program(vs, env.load_text("shaders/fetch.frag"), "fetch");
            composite_ = make_program(vs, env.load_text("shaders/composite.frag"), "composite");
            const std::string probe = env.load_text("shaders/probe.frag");
            for (int k = 0; k < static_cast<int>(probe_.size()); ++k)
                probe_[static_cast<std::size_t>(k)] = make_program(vs, with_define(probe, std::format("MODE {}", k)), std::format("probe MODE {}", k));
        }

        static void make_texture_fbo(mgl::TextureId& tex, mgl::FramebufferId& fbo, int w, int h)
        {
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, w, h);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2D, tex, 0);
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("RGBA8 framebuffer incomplete");
        }

        void create_grid_targets()
        {
            make_texture_fbo(grid_tex_, grid_fbo_, kGridW, kGridH);
            // The error tests need a framebuffer with two color attachments.
            make_texture_fbo(scratch_tex_[0], scratch_fbo_, kScratch, kScratch);
            mgl::glGenTextures(1, &scratch_tex_[1]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, scratch_tex_[1]);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kScratch, kScratch);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1),
                                        mgl::TextureTarget::Texture2D, scratch_tex_[1], 0);
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("two-attachment framebuffer incomplete");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void create_multisample_targets()
        {
            max_samples_ = demo::get_integer(mgl::GetParameter::MaxSamples);
            max_color_texture_samples_ = demo::get_integer(mgl::GetParameter::MaxColorTextureSamples);
            GLint n = 0;
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
            std::vector<GLint> counts(static_cast<std::size_t>(std::max(n, 0)));
            if (n > 0)
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, n,
                                           counts.data());
            int chosen = 0;
            if (std::find(counts.begin(), counts.end(), 4) != counts.end())
                chosen = 4;
            else
                for (GLint c : counts)
                    if (c >= 2 && c <= kMaxSamples) chosen = std::max(chosen, static_cast<int>(c));
            if (chosen == 0)
            {
                ms_reason_ = std::format("no RGBA8 multisample texture with 2 .. {} samples (MAX_SAMPLES = {}, MAX_COLOR_TEXTURE_SAMPLES = {}, {} RGBA8 counts)", kMaxSamples,
                                         max_samples_, max_color_texture_samples_, counts.size());
                return;
            }
            auto make_ms = [&](mgl::TextureId& tex, mgl::FramebufferId& fbo, int edge) {
                mgl::glGenTextures(1, &tex);
                mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, tex);
                mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2DMultisample, chosen, mgl::SizedInternalFormat::Rgba8, edge, edge, GL_TRUE);
                mgl::glGenFramebuffers(1, &fbo);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2DMultisample, tex, 0);
                if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                    throw demo::Failure("multisample framebuffer incomplete");
                return demo::get_integer(mgl::GetParameter::Samples);
            };
            n_ = make_ms(probe_tex_, probe_fbo_, kProbe);
            // The shading sample locations of section 13.4 (glGetMultisamplefv), queried with the framebuffer bound.
            positions_.assign(static_cast<std::size_t>(n_), Pos{});
            for (int s = 0; s < n_; ++s)
            {
                std::array<GLfloat, 2> v{};
                mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(s), v.data());
                positions_[static_cast<std::size_t>(s)] = {static_cast<double>(v[0]), static_cast<double>(v[1])};
            }
            const int n_vis0 = make_ms(vis_tex_[0], vis_fbo_[0], kVis);
            const int n_vis1 = make_ms(vis_tex_[1], vis_fbo_[1], kVis);
            vis_positions_.assign(static_cast<std::size_t>(n_vis1), Pos{});
            for (int s = 0; s < n_vis1; ++s)
            {
                std::array<GLfloat, 2> v{};
                mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(s), v.data());
                vis_positions_[static_cast<std::size_t>(s)] = {static_cast<double>(v[0]), static_cast<double>(v[1])};
            }
            make_texture_fbo(strip_tex_, strip_fbo_, kStripW, kVis);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, mgl::TextureId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (n_ < 2 || n_ > kMaxSamples || n_vis0 != n_ || n_vis1 != n_)
            {
                ms_reason_ = std::format("requested {} samples, framebuffers report SAMPLES = {} / {} / {}", chosen, n_, n_vis0, n_vis1);
                return;
            }
            ms_ok_ = true;
            std::cout << std::format("  multisample: {} samples (MAX_SAMPLES {}, MAX_COLOR_TEXTURE_SAMPLES {}); positions", n_, max_samples_, max_color_texture_samples_);
            for (const Pos& p : positions_) std::cout << std::format(" ({:.4f}, {:.4f})", p.x, p.y);
            std::cout << "\n";
        }

        // ==========================================================================================
        // The advanced-blend grid
        // ==========================================================================================
        void draw_disc(const Program& p, int cell, int disc)
        {
            mgl::glUseProgram(p.id);
            const double x0 = cell_x(cell), y0 = cell_y(cell);
            set_rect(p, x0, y0, x0 + kCellW, y0 + kCellH, kGridW, kGridH);
            const Disc& d = disc == 0 ? kDiscA : kDiscB;
            mgl::glUniform2f(p.origin, static_cast<float>(x0), static_cast<float>(y0));
            mgl::glUniform2f(p.cell, static_cast<float>(kCellW), static_cast<float>(kCellH));
            mgl::glUniform2f(p.center, static_cast<float>(d.cx), static_cast<float>(d.cy));
            mgl::glUniform1f(p.radius, static_cast<float>(d.r));
            mgl::glUniform1i(p.disc, disc);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        void draw_backdrop()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, grid_fbo_);
            mgl::glViewport(0, 0, kGridW, kGridH);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(backdrop_.id);
            set_rect(backdrop_, 0, 0, kGridW, kGridH, kGridW, kGridH);
            mgl::glUniform2f(backdrop_.cell, static_cast<float>(kCellW), static_cast<float>(kCellH));
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        void draw_grid()
        {
            draw_backdrop();
            // The backdrop touched every sample: start a new blending pass (section 15.1.5.3).
            mgl::glBlendBarrier();
            mgl::glEnable(mgl::Capability::Blend);
            // Advanced equations do not use the blend functions; these factors would visibly change any Table 15.1 result.
            mgl::glBlendFunc(mgl::BlendFactor::Zero, mgl::BlendFactor::OneMinusSrcColor);
            for (int disc = 0; disc < 2; ++disc)
            {
                if (disc == 1) mgl::glBlendBarrier();   // disc B overlaps disc A in every cell
                // Pass A uses one program per equation with its own qualifier, pass B blend_support_all_equations.
                for (int m = 0; m < kModeCount; ++m)
                {
                    mgl::glBlendEquation(kModes[static_cast<std::size_t>(m)].eq);
                    draw_disc(disc == 0 ? source_mode_[static_cast<std::size_t>(m)] : source_all_, m, disc);
                }
                // Control cell: an advanced equation is current, but blending is disabled, so the program without any
                // blend_support qualifier is legal and the disc replaces the destination.
                mgl::glDisable(mgl::Capability::Blend);
                mgl::glBlendEquation(BE::HslLuminosity);
                draw_disc(source_none_, kControlCell, disc);
                mgl::glEnable(mgl::Capability::Blend);
            }
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glBlendEquation(BE::FuncAdd);
            mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::Zero);
        }

        // ==========================================================================================
        // Multisample rendering
        // ==========================================================================================
        void draw_rings(mgl::FramebufferId fbo, bool per_sample)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glViewport(0, 0, kVis, kVis);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            if (per_sample)
            {
                // Not required by the spec: radeonsi ignores sample shading for a draw that follows a single-sample draw
                // (the grid) in the same command stream (see kRadeonsiSampleShading; validate() checks both orders).
                mgl::glFlush();
                mgl::glEnable(mgl::Capability::SampleShading);
                mgl::glMinSampleShading(1.0f);
            }
            mgl::glUseProgram(rings_.id);
            set_rect(rings_, 0, 0, kVis, kVis, kVis, kVis);
            mgl::glUniform2f(rings_.center, kVis / 2.0f, kVis / 2.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glDisable(mgl::Capability::SampleShading);
            mgl::glMinSampleShading(0.0f);
        }

        void draw_window(const demo::Env& env)
        {
            const int w = env.width, h = env.height;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, w, h);
            mgl::glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(composite_.id);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, grid_tex_);
            mgl::glUniform1i(composite_.grid, 0);
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glUniform1i(composite_.ms, 1);
            mgl::glUniform2i(composite_.texels, kVis, kVis);
            mgl::glUniform1i(composite_.samples, std::max(n_, 1));
            mgl::glUniform1i(composite_.cols, n_ <= 2 ? n_ : n_ <= 4 ? 2 : n_ <= 9 ? 3 : 4);
            mgl::glUniform2i(composite_.cell, kCellW, kCellH);
            auto region = [&](int mode, double x0, double y0, double x1, double y1, mgl::TextureId ms) {
                mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, ms);
                mgl::glUniform1i(composite_.mode, mode);
                mgl::glUniform2f(composite_.origin, static_cast<float>(x0), static_cast<float>(y0));
                mgl::glUniform2f(composite_.extent, static_cast<float>(x1 - x0), static_cast<float>(y1 - y0));
                set_rect(composite_, x0, y0, x1, y1, w, h);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            };
            const double split = std::round(0.375 * h);
            region(0, 0, split, w, h, mgl::TextureId{0});
            const double margin = std::max(2.0, std::round(0.02 * std::min(w, h)));
            const double pw = (w - 4.0 * margin) / 3.0;
            for (int k = 0; k < 3; ++k)
            {
                const double x0 = std::round(margin + k * (pw + margin));
                const int mode = !ms_ok_ ? 3 : k == 2 ? 2 : 1;
                region(mode, x0, margin, std::round(x0 + pw), split - margin, ms_ok_ ? vis_tex_[k == 0 ? 0 : 1] : mgl::TextureId{0});
            }
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        /// Copies every sample of a multisample texture (edge x edge) with texelFetch and reads them back:
        /// result[(s * edge + y) * edge + x].
        std::vector<demo::Rgba8> read_samples(mgl::TextureId ms, int edge)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, strip_fbo_);
            mgl::glViewport(0, 0, edge * n_, edge);
            mgl::glUseProgram(fetch_.id);
            set_rect(fetch_, 0, 0, edge * n_, edge, edge * n_, edge);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, ms);
            mgl::glUniform1i(fetch_.ms, 0);
            mgl::glUniform2i(fetch_.texels, edge, edge);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, mgl::TextureId{0});
            const std::vector<std::uint8_t> raw = demo::read_rgba8(0, 0, edge * n_, edge);
            std::vector<demo::Rgba8> out(static_cast<std::size_t>(n_ * edge * edge));
            for (int s = 0; s < n_; ++s)
                for (int y = 0; y < edge; ++y)
                    for (int x = 0; x < edge; ++x)
                    {
                        const std::size_t src = 4 * static_cast<std::size_t>(y * edge * n_ + s * edge + x);
                        out[static_cast<std::size_t>((s * edge + y) * edge + x)] = {raw[src], raw[src + 1], raw[src + 2], raw[src + 3]};
                    }
            return out;
        }

        /// Draws probe MODE `mode` into the 16 x 16 multisample texture (cleared to 0) and returns all samples.
        std::vector<demo::Rgba8> probe(int mode, const std::function<void(const Program&)>& setup, bool triangle = false)
        {
            const Program& p = probe_[static_cast<std::size_t>(mode)];
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, probe_fbo_);
            mgl::glViewport(0, 0, kProbe, kProbe);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(p.id);
            mgl::glUniform1f(p.scale, static_cast<float>(kProbe));
            if (triangle)
            {
                std::array<float, 8> c{};
                for (std::size_t i = 0; i < 3; ++i)
                {
                    c[2 * i] = static_cast<float>(kTriangle[i].x);
                    c[2 * i + 1] = static_cast<float>(kTriangle[i].y);
                }
                mgl::glUniform2fv(p.corner, 4, c.data());
                mgl::glUniform2f(p.size, static_cast<float>(kProbe), static_cast<float>(kProbe));
            }
            else
                set_rect(p, 0, 0, kProbe, kProbe, kProbe, kProbe);
            if (setup) setup(p);
            mgl::glDrawArrays(triangle ? mgl::PrimitiveType::Triangles : mgl::PrimitiveType::TriangleStrip, 0, triangle ? 3 : 4);
            return read_samples(probe_tex_, kProbe);
        }

        // ==========================================================================================
        // Validation: the grid
        // ==========================================================================================
        void validate_grid(const demo::Env& env, demo::Check& check)
        {
            check.expect(grid_error_ == mgl::ErrorCode::NoError, "the grid (15 advanced equations with their blend_support qualifiers, glBlendBarrier, control cell) raises no GL error",
                         demo::error_name(grid_error_));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, grid_fbo_);
            const std::vector<std::uint8_t> grid = demo::read_rgba8(0, 0, kGridW, kGridH);
            validate_window_grid(env, check, grid);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, grid_fbo_);
            // The destination that every cell started from: the same backdrop draw into the same framebuffer
            // (repeatability and invariance, ES 3.2 Appendix A.1 and A.3 rule 1), read back instead of being predicted.
            draw_backdrop();
            const std::vector<std::uint8_t> backdrop = demo::read_rgba8(0, 0, kGridW, kGridH);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            auto px = [](const std::vector<std::uint8_t>& img, int x, int y) {
                const std::size_t i = 4 * static_cast<std::size_t>(y * kGridW + x);
                return demo::Rgba8{img[i], img[i + 1], img[i + 2], img[i + 3]};
            };

            // The backdrop itself: within 1 of the CPU model and with alpha 0.45 .. 0.95 (non-trivial).
            {
                int bad = 0;
                std::string first;
                for (int y = 0; y < kGridH; ++y)
                    for (int x = 0; x < kGridW; ++x)
                    {
                        const double u = glsl_mod(x + 0.5, kCellW) / kCellW, v = glsl_mod(y + 0.5, kCellH) / kCellH;
                        const V4 want = premul(hsv(0.8 * u, 0.75, 0.5 + 0.45 * v), 0.95 - 0.5 * u);
                        const demo::Rgba8 got = px(backdrop, x, y);
                        for (std::size_t i = 0; i < 4; ++i)
                            if (std::abs(got[i] - want[i] * 255.0) > 1.0)
                            {
                                if (bad++ == 0) first = std::format("({}, {}): got {} expected {}", x, y, demo::format_rgba(got), fmt4(want));
                                break;
                            }
                    }
                check.expect(bad == 0, "backdrop: premultiplied gradient with alpha 0.45 .. 0.95 in every texel (+-1)", first);
            }

            for (int m = 0; m < kModeCount; ++m) check_cell(check, m, grid, backdrop, px);
            check_control_cell(check, grid, backdrop, px);
        }

        /// The upper part of the window shows the grid texture composited over a checkerboard (composite.frag, mode 0).
        /// Every pixel of that region is compared with the CPU composite of the grid texel it maps to; where the mapping
        /// lands on a texel boundary both neighbours are accepted.
        void validate_window_grid(const demo::Env& env, demo::Check& check, const std::vector<std::uint8_t>& grid)
        {
            const int w = env.width, h = env.height;
            const int split = static_cast<int>(std::round(0.375 * h));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const std::vector<std::uint8_t> win = demo::read_rgba8(0, split, w, h - split);
            auto candidates = [](double t, int size) {
                t = std::clamp(t, 0.0, 0.99999);
                const int a = std::clamp(static_cast<int>(std::floor(t * size - 1e-3)), 0, size - 1);
                const int b = std::clamp(static_cast<int>(std::floor(t * size + 1e-3)), 0, size - 1);
                return std::array<int, 2>{a, b};
            };
            auto composite = [&](int tx, int ty) {
                const std::size_t i = 4 * static_cast<std::size_t>(ty * kGridW + tx);
                const double a = grid[i + 3] / 255.0;
                const double board = (((tx / 8) % 2) ^ ((ty / 8) % 2)) == 0 ? 0.86 : 0.70;
                const double dark = (tx % kCellW == 0 || ty % kCellH == 0) ? 0.35 : 1.0;
                return V3{(grid[i] / 255.0 + (1.0 - a) * board) * dark, (grid[i + 1] / 255.0 + (1.0 - a) * board) * dark,
                          (grid[i + 2] / 255.0 + (1.0 - a) * board) * dark};
            };
            int bad = 0;
            std::string first;
            for (int y = split; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const std::size_t i = 4 * static_cast<std::size_t>((y - split) * w + x);
                    const auto cx = candidates((x + 0.5) / w, kGridW);
                    const auto cy = candidates((y + 0.5 - split) / (h - split), kGridH);
                    bool ok = false;
                    for (int tx : cx)
                        for (int ty : cy)
                        {
                            const V3 want = composite(tx, ty);
                            bool match = true;
                            for (std::size_t c = 0; c < 3; ++c) match = match && std::abs(win[i + c] - want[c] * 255.0) <= 1.0;
                            ok = ok || match;
                        }
                    if (!ok && bad++ == 0) first = std::format("window ({}, {}) = ({}, {}, {}), grid texel ({}, {})", x, y, win[i], win[i + 1], win[i + 2], cx[0], cy[0]);
                }
            check.expect(bad == 0, std::format("window: all {} x {} pixels of the grid region show the grid texel over the checkerboard (+-1)", w, h - split), first);
        }

        template <typename Px>
        void check_cell(demo::Check& check, int m, const std::vector<std::uint8_t>& grid, const std::vector<std::uint8_t>& backdrop, Px px)
        {
            const ModeInfo& info = kModes[static_cast<std::size_t>(m)];
            const int wrong = (m + 1) % kModeCount;
            int bad = 0, banded = 0, count_a = 0, count_b = 0, count_ab = 0, blended = 0, wrong_mismatch = 0;
            double max_dev = 0.0;
            std::string first;
            const int x0 = cell_x(m), y0 = cell_y(m);
            for (int y = 0; y < kCellH; ++y)
                for (int x = 0; x < kCellW; ++x)
                {
                    const double lx = x + 0.5, ly = y + 0.5;
                    const double u = lx / kCellW, v = ly / kCellH;
                    const int ca = disc_cover(kDiscA, lx, ly), cb = disc_cover(kDiscB, lx, ly);
                    const V4 sa = source_color(0, u, v), sb = source_color(1, u, v);
                    const demo::Rgba8 dst0 = px(backdrop, x0 + x, y0 + y);
                    const demo::Rgba8 got = px(grid, x0 + x, y0 + y);
                    bool ok = false;
                    PixelVerdict primary;
                    bool have_primary = false;
                    for (int a = 0; a < 2 && !ok; ++a)
                        for (int b = 0; b < 2 && !ok; ++b)
                        {
                            if ((ca != 2 && a != ca) || (cb != 2 && b != cb)) continue;
                            const PixelVerdict pv = consistent(m, a == 1, b == 1, sa, sb, dst0, got);
                            if (!have_primary)
                            {
                                primary = pv;
                                have_primary = true;
                            }
                            ok = pv.ok;
                        }
                    if (ca == 1 && cb == 0) ++count_a;
                    if (ca == 0 && cb == 1) ++count_b;
                    if (ca == 1 && cb == 1) ++count_ab;
                    if (ca != 2 && cb != 2)
                    {
                        max_dev = std::max(max_dev, primary.dev);
                        if (primary.used_band) ++banded;
                        if (ca == 1 || cb == 1)
                        {
                            ++blended;
                            // Negative control: the same texel judged against the next equation's reference.
                            if (!consistent(wrong, ca == 1, cb == 1, sa, sb, dst0, got).ok) ++wrong_mismatch;
                        }
                    }
                    if (!ok && bad++ == 0)
                    {
                        V4 c = unit(dst0);
                        if (ca == 1) c = advanced_premul(m, sa, c);
                        if (cb == 1) c = advanced_premul(m, sb, c);
                        first = std::format("cell texel ({}, {}) A={} B={}: got {} exact {} (destination {})", x, y, ca, cb, demo::format_rgba(got), fmt4(c),
                                            demo::format_rgba(dst0));
                    }
                }
            std::cout << std::format("  {:<15} A only {:4}, B only {:4}, A+B {:4}: max |got - exact| = {:.2f}, {} texels needed the precision range, {} failed\n", info.name,
                                     count_a, count_b, count_ab, max_dev, banded, bad);
            check.expect(bad == 0 && count_a > 50 && count_b > 50 && count_ab > 50,
                         std::format("{}: all {} texels of the cell match section 15.1.5.3 (backdrop, disc A, glBlendBarrier, disc B)", info.name, kCellW * kCellH), first);
            const double fraction = blended > 0 ? static_cast<double>(wrong_mismatch) / blended : 0.0;
            check.expect(fraction >= 0.2,
                         std::format("negative control: the {} cell does not match the {} reference ({:.0f}% of {} blended texels differ)", info.name,
                                     kModes[static_cast<std::size_t>(wrong)].name, 100.0 * fraction, blended));
        }

        template <typename Px>
        void check_control_cell(demo::Check& check, const std::vector<std::uint8_t>& grid, const std::vector<std::uint8_t>& backdrop, Px px)
        {
            int bad = 0;
            std::string first;
            const int x0 = cell_x(kControlCell), y0 = cell_y(kControlCell);
            for (int y = 0; y < kCellH; ++y)
                for (int x = 0; x < kCellW; ++x)
                {
                    const double lx = x + 0.5, ly = y + 0.5;
                    const int ca = disc_cover(kDiscA, lx, ly), cb = disc_cover(kDiscB, lx, ly);
                    const V4 sa = source_color(0, lx / kCellW, ly / kCellH), sb = source_color(1, lx / kCellW, ly / kCellH);
                    const V4 d0 = unit(px(backdrop, x0 + x, y0 + y));
                    const demo::Rgba8 got = px(grid, x0 + x, y0 + y);
                    bool ok = false;
                    for (int a = 0; a < 2 && !ok; ++a)
                        for (int b = 0; b < 2 && !ok; ++b)
                        {
                            if ((ca != 2 && a != ca) || (cb != 2 && b != cb)) continue;
                            const V4& want = b ? sb : a ? sa : d0;
                            ok = true;
                            for (std::size_t i = 0; i < 4; ++i)
                                if (std::abs(got[i] - want[i] * 255.0) > 1.0) ok = false;
                        }
                    if (!ok && bad++ == 0) first = std::format("cell texel ({}, {}): got {}", x, y, demo::format_rgba(got));
                }
            check.expect(bad == 0, "control cell: BLEND disabled with HSL_LUMINOSITY current: the discs replace the destination (+-1), no blend_support needed", first);
        }

        // ==========================================================================================
        // Validation: errors and state of advanced blending
        // ==========================================================================================
        void draw_scratch(const Program& p)
        {
            mgl::glUseProgram(p.id);
            set_rect(p, 0, 0, kScratch, kScratch, kScratch, kScratch);
            mgl::glUniform2f(p.origin, 0.0f, 0.0f);
            mgl::glUniform2f(p.cell, static_cast<float>(kScratch), static_cast<float>(kScratch));
            mgl::glUniform2f(p.center, 4.0f, 4.0f);
            mgl::glUniform1f(p.radius, 100.0f);
            mgl::glUniform1i(p.disc, 0);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        void validate_blend_errors(demo::Check& check)
        {
            using mgl::ErrorCode;
            demo::drain_gl_errors();
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, scratch_fbo_);
            mgl::glViewport(0, 0, kScratch, kScratch);
            const std::array<mgl::DrawBuffer, 2> c0_none = {mgl::to_draw_buffer(mgl::ColorAttachment::Color0), mgl::DrawBuffer::None};
            const std::array<mgl::DrawBuffer, 2> c0_c1 = {mgl::to_draw_buffer(mgl::ColorAttachment::Color0), mgl::to_draw_buffer(mgl::ColorAttachment::Color1)};
            const std::array<mgl::DrawBuffer, 2> none_c1 = {mgl::DrawBuffer::None, mgl::to_draw_buffer(mgl::ColorAttachment::Color1)};
            mgl::glDrawBuffers(c0_none);

            // BLEND_EQUATION_RGB / _ALPHA after glBlendEquation with each advanced equation (section 15.1.5.5).
            {
                int bad = 0;
                std::string first;
                for (const ModeInfo& info : kModes)
                {
                    mgl::glBlendEquation(info.eq);
                    GLint rgb = 0, alpha = 0;
                    mgl::glGetIntegerv(mgl::GetParameter::BlendEquationRgb, &rgb);
                    mgl::glGetIntegerv(mgl::GetParameter::BlendEquationAlpha, &alpha);
                    if ((rgb != value_of(info.eq) || alpha != value_of(info.eq)) && bad++ == 0)
                        first = std::format("{}: RGB 0x{:X} ALPHA 0x{:X}", info.name, rgb, alpha);
                }
                check.expect(bad == 0, "glBlendEquation(advanced): BLEND_EQUATION_RGB and BLEND_EQUATION_ALPHA both report the equation, all 15", first);
                check.expect_no_gl_error("glBlendEquation accepts all 15 advanced equations (Tables 15.3, 15.4)");
            }

            // BlendEquationSeparate accepts only Table 15.1 (section 15.1.5.1).
            mgl::glBlendEquation(BE::FuncReverseSubtract);
            mgl::glBlendEquationSeparate(BE::Multiply, BE::FuncAdd);
            check.expect_gl_error(ErrorCode::InvalidEnum, "glBlendEquationSeparate(MULTIPLY, FUNC_ADD): INVALID_ENUM (advanced equations are not in Table 15.1)");
            mgl::glBlendEquationSeparate(BE::FuncAdd, BE::HslColor);
            check.expect_gl_error(ErrorCode::InvalidEnum, "glBlendEquationSeparate(FUNC_ADD, HSL_COLOR): INVALID_ENUM");
            {
                GLint rgb = 0, alpha = 0;
                mgl::glGetIntegerv(mgl::GetParameter::BlendEquationRgb, &rgb);
                mgl::glGetIntegerv(mgl::GetParameter::BlendEquationAlpha, &alpha);
                check.expect(rgb == value_of(BE::FuncReverseSubtract) && alpha == value_of(BE::FuncReverseSubtract),
                             "the rejected glBlendEquationSeparate calls leave the equation unchanged (FUNC_REVERSE_SUBTRACT)");
            }

            // Draw-time errors of section 15.1.5.3 with blending enabled.
            mgl::glEnable(mgl::Capability::Blend);
            {
                const std::array<GLfloat, 4> fill = {0.2f, 0.4f, 0.6f, 0.8f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, fill.data());
                mgl::glBlendEquation(BE::Multiply);
                draw_scratch(source_none_);
                check.expect_gl_error(ErrorCode::InvalidOperation, "MULTIPLY + a fragment shader without blend_support: the draw raises INVALID_OPERATION");
                const demo::Rgba8 after = demo::read_pixel(3, 3);
                check.expect(after == demo::Rgba8{51, 102, 153, 204}, "the rejected draw leaves the color buffer unchanged", demo::format_rgba(after));
            }
            {
                int bad_none = 0, bad_own = 0, bad_other = 0, bad_all = 0;
                std::string first;
                for (int m = 0; m < kModeCount; ++m)
                {
                    const ModeInfo& info = kModes[static_cast<std::size_t>(m)];
                    const ModeInfo& next = kModes[static_cast<std::size_t>((m + 1) % kModeCount)];
                    mgl::glBlendEquation(info.eq);
                    draw_scratch(source_none_);
                    if (const ErrorCode e = demo::drain_gl_errors(); e != ErrorCode::InvalidOperation && bad_none++ == 0)
                        first = std::format("{} without qualifier: {}", info.name, demo::error_name(e));
                    draw_scratch(source_mode_[static_cast<std::size_t>(m)]);
                    if (const ErrorCode e = demo::drain_gl_errors(); e != ErrorCode::NoError && bad_own++ == 0)
                        first = std::format("{} with layout({}): {}", info.name, info.layout, demo::error_name(e));
                    mgl::glBlendBarrier();
                    draw_scratch(source_mode_[static_cast<std::size_t>((m + 1) % kModeCount)]);
                    if (const ErrorCode e = demo::drain_gl_errors(); e != ErrorCode::InvalidOperation && bad_other++ == 0)
                        first = std::format("{} with layout({}): {}", info.name, next.layout, demo::error_name(e));
                    draw_scratch(source_all_);
                    if (const ErrorCode e = demo::drain_gl_errors(); e != ErrorCode::NoError && bad_all++ == 0)
                        first = std::format("{} with blend_support_all_equations: {}", info.name, demo::error_name(e));
                    mgl::glBlendBarrier();
                }
                check.expect(bad_none == 0, "every advanced equation + no blend_support qualifier: INVALID_OPERATION at the draw (15 equations)", first);
                check.expect(bad_own == 0, "every advanced equation + its own layout(blend_support_<equation>): no error (15 qualifiers)", first);
                check.expect(bad_other == 0, "every advanced equation + the qualifier of a different equation: INVALID_OPERATION (15 pairs)", first);
                check.expect(bad_all == 0, "every advanced equation + layout(blend_support_all_equations): no error", first);
            }
            mgl::glBlendEquation(BE::FuncAdd);
            draw_scratch(source_none_);
            check.expect_no_gl_error("control: FUNC_ADD + a shader without blend_support draws without error");
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glBlendEquation(BE::Multiply);
            draw_scratch(source_none_);
            check.expect_no_gl_error("control: MULTIPLY current but BLEND disabled + a shader without blend_support draws without error");
            mgl::glEnable(mgl::Capability::Blend);

            // Only one color buffer, written by fragment color zero (section 15.1.5.3).
            mgl::glDrawBuffers(c0_c1);
            draw_scratch(source_all_);
            check.expect_gl_error(ErrorCode::InvalidOperation, "MULTIPLY with draw buffers {COLOR_ATTACHMENT0, COLOR_ATTACHMENT1}: INVALID_OPERATION");
            mgl::glDrawBuffers(none_c1);
            draw_scratch(source_all_);
            check.expect_gl_error(ErrorCode::InvalidOperation, "MULTIPLY with draw buffers {NONE, COLOR_ATTACHMENT1} (buffer 1 is not NONE): INVALID_OPERATION");
            mgl::glBlendEquation(BE::FuncAdd);
            mgl::glDrawBuffers(c0_c1);
            draw_scratch(source_all_);
            check.expect_no_gl_error("control: FUNC_ADD with two draw buffers draws without error");
            mgl::glBlendEquation(BE::Screen);
            mgl::glDrawBuffers(c0_none);
            draw_scratch(source_all_);
            check.expect_no_gl_error("SCREEN with draw buffers {COLOR_ATTACHMENT0, NONE} draws without error");
            mgl::glBlendBarrier();
            check.expect_no_gl_error("glBlendBarrier raises no error");

            mgl::glDisable(mgl::Capability::Blend);
            mgl::glBlendEquation(BE::FuncAdd);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // ==========================================================================================
        // Validation: sample shading state
        // ==========================================================================================
        void validate_sample_state(demo::Check& check)
        {
            check.expect(!initial_sample_shading_, "SAMPLE_SHADING is initially disabled");
            check.expect_near(static_cast<double>(initial_min_sample_shading_), 0.0, 0.0, "MIN_SAMPLE_SHADING_VALUE is initially 0");
            auto value = [] {
                GLfloat v = -1.0f;
                mgl::glGetFloatv(mgl::GetParameter::MinSampleShadingValue, &v);
                return static_cast<double>(v);
            };
            mgl::glMinSampleShading(0.25f);
            check.expect_near(value(), 0.25, 0.0, "glMinSampleShading(0.25): MIN_SAMPLE_SHADING_VALUE = 0.25");
            mgl::glMinSampleShading(1.5f);
            check.expect_near(value(), 1.0, 0.0, "glMinSampleShading(1.5) is clamped to 1.0");
            mgl::glMinSampleShading(-0.5f);
            check.expect_near(value(), 0.0, 0.0, "glMinSampleShading(-0.5) is clamped to 0.0");
            mgl::glEnable(mgl::Capability::SampleShading);
            check.expect(mgl::glIsEnabled(mgl::Capability::SampleShading), "glEnable(SAMPLE_SHADING): IsEnabled = TRUE");
            mgl::glDisable(mgl::Capability::SampleShading);
            check.expect(!mgl::glIsEnabled(mgl::Capability::SampleShading), "glDisable(SAMPLE_SHADING): IsEnabled = FALSE");

            const double min_offset = [] {
                GLfloat v = 0.0f;
                mgl::glGetFloatv(mgl::GetParameter::MinFragmentInterpolationOffset, &v);
                return static_cast<double>(v);
            }();
            const double max_offset = [] {
                GLfloat v = 0.0f;
                mgl::glGetFloatv(mgl::GetParameter::MaxFragmentInterpolationOffset, &v);
                return static_cast<double>(v);
            }();
            offset_bits_ = demo::get_integer(mgl::GetParameter::FragmentInterpolationOffsetBits);
            min_offset_ = min_offset;
            max_offset_ = max_offset;
            std::cout << std::format("  interpolateAtOffset limits: [{}, {}], {} fraction bits\n", min_offset, max_offset, offset_bits_);
            check.expect(offset_bits_ >= 4, std::format("FRAGMENT_INTERPOLATION_OFFSET_BITS = {} >= 4 (Table 21.41)", offset_bits_));
            check.expect(min_offset <= -0.5, std::format("MIN_FRAGMENT_INTERPOLATION_OFFSET = {} <= -0.5 (Table 21.41)", min_offset));
            check.expect(max_offset >= 0.5 - std::ldexp(1.0, -std::max(offset_bits_, 1)),
                         std::format("MAX_FRAGMENT_INTERPOLATION_OFFSET = {} >= 0.5 - 1 ULP (2^-{}) (Table 21.41)", max_offset, offset_bits_));
            check.expect(max_samples_ >= 4, std::format("MAX_SAMPLES = {} >= 4 (Table 21.41)", max_samples_));
        }

        // ==========================================================================================
        // Validation: per-sample values
        // ==========================================================================================
        Pos sample_pos(int x, int y, int s) const
        {
            const Pos& p = positions_[static_cast<std::size_t>(s)];
            return {x + p.x, y + p.y};
        }

        static Pos decoded(const demo::Rgba8& c, double scale)
        {
            return {decode(c[0], c[1]) * scale, decode(c[2], c[3]) * scale};
        }

        static std::size_t at(int s, int x, int y, int edge) { return static_cast<std::size_t>((s * edge + y) * edge + x); }

        /// Compares the decoded positions of every sample with want(x, y, s); returns the largest error.
        template <typename Want>
        double expect_positions(demo::Check& check, const std::vector<demo::Rgba8>& got, Want want, double tol, std::string_view what,
                                std::string_view deviation_renderer = {}, std::string_view deviation_reason = {})
        {
            int bad = 0;
            double worst = 0.0;
            std::string first;
            for (int s = 0; s < n_; ++s)
                for (int y = 0; y < kProbe; ++y)
                    for (int x = 0; x < kProbe; ++x)
                    {
                        const Pos g = decoded(got[at(s, x, y, kProbe)], kProbe);
                        const Pos w = want(x, y, s);
                        const double e = std::max(std::abs(g.x - w.x), std::abs(g.y - w.y));
                        worst = std::max(worst, e);
                        if (e > tol && bad++ == 0) first = std::format("sample {} of ({}, {}): got ({:.4f}, {:.4f}) expected ({:.4f}, {:.4f})", s, x, y, g.x, g.y, w.x, w.y);
                    }
            std::cout << std::format("  {}: max error {:.5f} px\n", what, worst);
            const std::string label = std::format("{} (all {} samples, tolerance {:.4f} px)", what, n_ * kProbe * kProbe, tol);
            if (deviation_renderer.empty())
                check.expect(bad == 0, label, first);
            else
                check.expect_or_known_deviation(bad == 0, label, first, deviation_renderer, deviation_reason);
            return worst;
        }

        void validate_multisample(demo::Check& check)
        {
            check.expect(rings_error_ == mgl::ErrorCode::NoError, "multisample ring passes (SAMPLE_SHADING on/off) raise no GL error", demo::error_name(rings_error_));
            const auto n = static_cast<std::size_t>(n_);

            // glGetMultisamplefv (section 13.4): positions in [0, 1]; INVALID_VALUE for index = SAMPLES.
            {
                bool in_range = true;
                for (const Pos& p : positions_) in_range = in_range && p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0;
                check.expect(in_range, "glGetMultisamplefv(SAMPLE_POSITION): every position lies in [0, 1]");
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, probe_fbo_);
                std::array<GLfloat, 2> v{};
                mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(n_), v.data());
                check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetMultisamplefv(SAMPLE_POSITION, index = SAMPLES): INVALID_VALUE");
            }
            // Smallest distance between two sample positions: the position checks must tell samples apart.
            double dmin = 1.0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = i + 1; j < n; ++j)
                    dmin = std::min(dmin, std::max(std::abs(positions_[i].x - positions_[j].x), std::abs(positions_[i].y - positions_[j].y)));
            // Interpolation precision is not specified; 1/32 pixel is a sanity bound that still separates the samples.
            const double tol = std::min(1.0 / 32.0, dmin / 4.0);
            check.expect(dmin > 2.0 * tol, std::format("the {} sample positions are distinct (closest pair {:.4f} px apart)", n_, dmin));

            // MODE 0: gl_SampleID, gl_NumSamples, gl_SampleMaskIn. Static use of gl_SampleID makes the shader run per
            // sample (GLSL ES 3.20 section 7.1.5), so each invocation covers exactly its own sample.
            {
                const auto got = probe(0, nullptr);
                int bad_id = 0, bad_num = 0, bad_mask = 0;
                std::string first;
                for (int s = 0; s < n_; ++s)
                    for (int y = 0; y < kProbe; ++y)
                        for (int x = 0; x < kProbe; ++x)
                        {
                            const demo::Rgba8& c = got[at(s, x, y, kProbe)];
                            if (c[0] != s && bad_id++ == 0) first = std::format("sample {} of ({}, {}): {}", s, x, y, demo::format_rgba(c));
                            if (c[1] != n_) ++bad_num;
                            if (c[2] != (1 << s) && bad_mask++ == 0) first += std::format(" mask of sample {} of ({}, {}) = 0x{:X}", s, x, y, c[2]);
                        }
                check.expect(bad_id == 0, std::format("gl_SampleID of the invocation that wrote sample s is s (texelFetch sample index, all {} samples)", n_ * kProbe * kProbe), first);
                check.expect(bad_num == 0, std::format("gl_NumSamples = {} (SAMPLES of the framebuffer) in every invocation", n_));
                check.expect(bad_mask == 0, "gl_SampleMaskIn[0] = 1 << gl_SampleID: a per-sample invocation covers only its own sample", first);
            }

            // MODE 1: gl_SamplePosition = the glGetMultisamplefv position (mediump: relative precision 2^-10, section 4.7.1).
            {
                const auto got = probe(1, nullptr);
                int bad = 0;
                double worst = 0.0;
                std::string first;
                for (int s = 0; s < n_; ++s)
                    for (int y = 0; y < kProbe; ++y)
                        for (int x = 0; x < kProbe; ++x)
                        {
                            const Pos g = decoded(got[at(s, x, y, kProbe)], 1.0);
                            const Pos& w = positions_[static_cast<std::size_t>(s)];
                            const double e = std::max(std::abs(g.x - w.x), std::abs(g.y - w.y));
                            worst = std::max(worst, e);
                            if (e > 1.0 / 512.0 && bad++ == 0) first = std::format("sample {} of ({}, {}): ({:.4f}, {:.4f})", s, x, y, g.x, g.y);
                        }
                std::cout << std::format("  gl_SamplePosition: max error {:.5f}\n", worst);
                check.expect(bad == 0, "gl_SamplePosition equals glGetMultisamplefv(SAMPLE_POSITION, gl_SampleID) at every sample (+-1/512)", first);
            }

            // MODE 2: a "sample in" input is evaluated at the sample's location (ES 3.2 section 13.4).
            {
                const auto got = probe(2, nullptr);
                expect_positions(check, got, [&](int x, int y, int s) { return sample_pos(x, y, s); }, tol, "\"sample in\" input = pixel + sample position");
                // Negative control: the same values compared with the next sample's position must all fail.
                int wrong_ok = 0;
                for (int s = 0; s < n_; ++s)
                    for (int y = 0; y < kProbe; ++y)
                        for (int x = 0; x < kProbe; ++x)
                        {
                            const Pos g = decoded(got[at(s, x, y, kProbe)], kProbe);
                            const Pos w = sample_pos(x, y, (s + 1) % n_);
                            if (std::max(std::abs(g.x - w.x), std::abs(g.y - w.y)) <= tol) ++wrong_ok;
                        }
                check.expect(wrong_ok == 0, std::format("negative control: no sample matches the position of sample s + 1 ({} matches)", wrong_ok));
            }

            // MODE 3 with SAMPLE_SHADING and MinSampleShading(1.0): an ordinary input is evaluated per sample, at the
            // sample location (section 13.4.1).
            {
                // (a) Directly after a draw into a single-sample framebuffer, in the same command stream.
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, scratch_fbo_);
                mgl::glViewport(0, 0, kScratch, kScratch);
                draw_scratch(source_none_);
                mgl::glEnable(mgl::Capability::SampleShading);
                mgl::glMinSampleShading(1.0f);
                const auto after_single = probe(3, nullptr);
                expect_positions(check, after_single, [&](int x, int y, int s) { return sample_pos(x, y, s); }, tol,
                                 "SAMPLE_SHADING + MinSampleShading(1.0) right after a single-sample draw: an ordinary input = pixel + sample position", "radeonsi",
                                 kRadeonsiSampleShading);
                // (b) The same with glFlush between the single-sample draw and the multisample draw.
                mgl::glDisable(mgl::Capability::SampleShading);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, scratch_fbo_);
                mgl::glViewport(0, 0, kScratch, kScratch);
                draw_scratch(source_none_);
                mgl::glFlush();
                mgl::glEnable(mgl::Capability::SampleShading);
                const auto got = probe(3, nullptr);
                expect_positions(check, got, [&](int x, int y, int s) { return sample_pos(x, y, s); }, tol,
                                 "SAMPLE_SHADING + MinSampleShading(1.0): an ordinary input = pixel + sample position");

                // MinSampleShading(0.5): at least max(ceil(0.5 * N), 1) unique sets of inputs per fragment.
                mgl::glMinSampleShading(0.5f);
                const auto half = probe(3, nullptr);
                const int need = std::max(static_cast<int>(std::ceil(0.5 * n_)), 1);
                int bad = 0, fewest = n_;
                for (int y = 0; y < kProbe; ++y)
                    for (int x = 0; x < kProbe; ++x)
                    {
                        const int d = distinct(half, x, y, tol);
                        fewest = std::min(fewest, d);
                        if (d < need) ++bad;
                    }
                std::cout << std::format("  MinSampleShading(0.5): at least {} distinct inputs per pixel\n", fewest);
                check.expect(bad == 0, std::format("SAMPLE_SHADING + MinSampleShading(0.5): every pixel has >= ceil(0.5 * {}) = {} distinct input values", n_, need),
                             std::format("{} pixels below", bad));
                mgl::glDisable(mgl::Capability::SampleShading);
                mgl::glMinSampleShading(0.0f);
                const auto off = probe(3, nullptr);
                int most = 0;
                for (int y = 0; y < kProbe; ++y)
                    for (int x = 0; x < kProbe; ++x) most = std::max(most, distinct(off, x, y, tol));
                std::cout << std::format("  SAMPLE_SHADING disabled: at most {} distinct input values per pixel (not specified, not checked)\n", most);
            }

            // MODE 4: interpolateAtSample(v, i) from a pixel-rate shader, for every sample number i.
            {
                int bad = 0;
                double worst = 0.0;
                std::string first;
                for (int i = 0; i < n_; ++i)
                {
                    const auto got = probe(4, [&](const Program& p) { mgl::glUniform1i(p.index, i); });
                    for (int s = 0; s < n_; ++s)
                        for (int y = 0; y < kProbe; ++y)
                            for (int x = 0; x < kProbe; ++x)
                            {
                                const Pos g = decoded(got[at(s, x, y, kProbe)], kProbe);
                                const Pos w = sample_pos(x, y, i);
                                const double e = std::max(std::abs(g.x - w.x), std::abs(g.y - w.y));
                                worst = std::max(worst, e);
                                if (e > tol && bad++ == 0) first = std::format("i={} sample {} of ({}, {}): ({:.4f}, {:.4f})", i, s, x, y, g.x, g.y);
                            }
                }
                std::cout << std::format("  interpolateAtSample: max error {:.5f} px\n", worst);
                check.expect(bad == 0, std::format("interpolateAtSample(v, i) = v at pixel + position of sample i, for i = 0 .. {} (tolerance {:.4f} px)", n_ - 1, tol), first);
            }

            // MODE 5: interpolateAtOffset(v, offset) = v at the pixel centre + offset, for offsets on the
            // 2^-FRAGMENT_INTERPOLATION_OFFSET_BITS grid inside [MIN, MAX] (ES 3.2 section 14.1).
            {
                const double q = std::ldexp(1.0, -std::max(offset_bits_, 1));
                const double lo = std::ceil(min_offset_ / q) * q, hi = std::floor(max_offset_ / q) * q;
                const std::array<Pos, 6> offsets = {{{0.0, 0.0}, {lo, lo}, {hi, hi}, {lo, hi}, {0.25, -0.375}, {-0.125, 0.4375}}};
                for (const Pos& o : offsets)
                {
                    const Pos oc = {std::clamp(o.x, lo, hi), std::clamp(o.y, lo, hi)};
                    const auto got = probe(5, [&](const Program& p) { mgl::glUniform2f(p.offset, static_cast<float>(oc.x), static_cast<float>(oc.y)); });
                    expect_positions(check, got, [&](int x, int y, int) { return Pos{x + 0.5 + oc.x, y + 0.5 + oc.y}; }, tol,
                                     std::format("interpolateAtOffset(v, ({}, {})) = v at pixel centre + offset", oc.x, oc.y));
                }
            }

            // MODES 6, 7, 8 on a triangle: interpolateAtCentroid, "centroid in", and gl_SampleMaskIn of a pixel-rate shader.
            {
                const auto iac = probe(6, nullptr, true);
                const auto cin = probe(7, nullptr, true);
                const auto mask = probe(8, nullptr, true);
                int partial = 0, center_outside = 0, bad_loc = 0, bad_same = 0, bad_mask = 0, covered_pixels = 0;
                double worst_same = 0.0;
                std::string first;
                for (int y = 0; y < kProbe; ++y)
                    for (int x = 0; x < kProbe; ++x)
                    {
                        unsigned covered = 0;
                        for (int s = 0; s < n_; ++s)
                            if (mask[at(s, x, y, kProbe)][3] == 255) covered |= 1u << s;
                        if (covered == 0) continue;
                        ++covered_pixels;
                        const bool is_partial = covered != (1u << n_) - 1u;
                        if (is_partial) ++partial;
                        bool center_in = true;
                        for (std::size_t e = 0; e < 3; ++e) center_in = center_in && edge_distance(e, x + 0.5, y + 0.5) > 0.0;
                        if (!center_in) ++center_outside;
                        for (int s = 0; s < n_; ++s)
                        {
                            if (!(covered & (1u << s))) continue;
                            const unsigned m = mask[at(s, x, y, kProbe)][0];
                            if (((m & (1u << s)) == 0 || (m & ~covered) != 0) && bad_mask++ == 0)
                                first += std::format(" mask 0x{:X} at sample {} of ({}, {}) covered 0x{:X};", m, s, x, y, covered);
                            for (const auto* img : {&iac, &cin})
                            {
                                const Pos g = decoded((*img)[at(s, x, y, kProbe)], kProbe);
                                bool inside = g.x >= x - tol && g.x <= x + 1 + tol && g.y >= y - tol && g.y <= y + 1 + tol;
                                for (std::size_t e = 0; e < 3; ++e) inside = inside && edge_distance(e, g.x, g.y) >= -tol;
                                if (!inside && bad_loc++ == 0)
                                    first += std::format(" {} at sample {} of ({}, {}) = ({:.4f}, {:.4f}) outside pixel or triangle;", img == &iac ? "interpolateAtCentroid" : "centroid in", s, x,
                                                         y, g.x, g.y);
                            }
                            const Pos a = decoded(iac[at(s, x, y, kProbe)], kProbe), b = decoded(cin[at(s, x, y, kProbe)], kProbe);
                            const double e = std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
                            worst_same = std::max(worst_same, e);
                            if (e > tol) ++bad_same;
                        }
                    }
                std::cout << std::format("  triangle: {} covered pixels, {} partially covered, {} with the centre outside; |interpolateAtCentroid - centroid in| <= {:.5f} px\n",
                                         covered_pixels, partial, center_outside, worst_same);
                check.expect(partial >= 10 && center_outside >= 5, std::format("the triangle has partially covered pixels ({}) and covered pixels whose centre is outside ({})", partial, center_outside));
                check.expect(bad_loc == 0, "interpolateAtCentroid and \"centroid in\" lie inside both the pixel and the triangle for every covered sample", first);
                check.expect(bad_same == 0, "interpolateAtCentroid(v) equals the same input declared \"centroid in\"", first);
                check.expect(bad_mask == 0, "gl_SampleMaskIn[0] of a pixel-rate shader contains the written sample and only covered samples", first);
            }

            // The displayed rings: with SAMPLE_SHADING and MinSampleShading(1.0) each sample holds the pattern at its own
            // position; where the pattern changes within half a tolerance of a sample both colors are accepted.
            {
                const auto got = read_samples(vis_tex_[1], kVis);
                const auto flat = read_samples(vis_tex_[0], kVis);
                constexpr demo::Rgba8 gold = {250, 204, 64, 255}, blue = {26, 64, 140, 255};
                auto near = [](const demo::Rgba8& a, const demo::Rgba8& b) {
                    for (std::size_t i = 0; i < 4; ++i)
                        if (std::abs(a[i] - b[i]) > 1) return false;
                    return true;
                };
                int bad = 0, mixed = 0, mixed_flat = 0;
                std::string first;
                for (int y = 0; y < kVis; ++y)
                    for (int x = 0; x < kVis; ++x)
                    {
                        bool has_gold = false, has_blue = false, flat_gold = false, flat_blue = false;
                        for (int s = 0; s < n_; ++s)
                        {
                            const Pos& p = vis_positions_[static_cast<std::size_t>(s)];
                            const double dx = x + p.x - kVis / 2.0, dy = y + p.y - kVis / 2.0;
                            const double r2 = dx * dx + dy * dy;
                            const double g = fract(r2 / 48.0);
                            const double eps = 2.0 * std::sqrt(r2) / 48.0 * tol + 1e-4;   // |d g / d p| * tolerance
                            const bool either = std::abs(g - 0.5) < eps || g < eps || g > 1.0 - eps;
                            const demo::Rgba8& c = got[static_cast<std::size_t>((s * kVis + y) * kVis + x)];
                            const bool ok = either ? near(c, gold) || near(c, blue) : near(c, g < 0.5 ? gold : blue);
                            if (!ok && bad++ == 0) first = std::format("sample {} of ({}, {}): {} (g = {:.3f})", s, x, y, demo::format_rgba(c), g);
                            has_gold = has_gold || near(c, gold);
                            has_blue = has_blue || near(c, blue);
                            const demo::Rgba8& f = flat[static_cast<std::size_t>((s * kVis + y) * kVis + x)];
                            flat_gold = flat_gold || near(f, gold);
                            flat_blue = flat_blue || near(f, blue);
                        }
                        if (has_gold && has_blue) ++mixed;
                        if (flat_gold && flat_blue) ++mixed_flat;
                    }
                std::cout << std::format("  rings: {} of {} pixels have both colors among their samples with sample shading, {} without\n", mixed, kVis * kVis, mixed_flat);
                check.expect(bad == 0 && mixed >= 20,
                             std::format("rings with MinSampleShading(1.0): every sample shows the pattern at its own position ({} pixels with distinct per-sample values)", mixed), first);
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        int distinct(const std::vector<demo::Rgba8>& img, int x, int y, double tol) const
        {
            std::vector<Pos> seen;
            for (int s = 0; s < n_; ++s)
            {
                const Pos g = decoded(img[at(s, x, y, kProbe)], kProbe);
                bool found = false;
                for (const Pos& p : seen) found = found || std::max(std::abs(p.x - g.x), std::abs(p.y - g.y)) <= tol;
                if (!found) seen.push_back(g);
            }
            return static_cast<int>(seen.size());
        }

        // ==========================================================================================
        mgl::VertexArrayId vao_{};
        Program backdrop_, source_none_, source_all_, rings_, fetch_, composite_;
        std::array<Program, 15> source_mode_{};
        std::array<Program, 9> probe_{};
        mgl::TextureId grid_tex_{}, probe_tex_{}, strip_tex_{};
        std::array<mgl::TextureId, 2> scratch_tex_{}, vis_tex_{};
        mgl::FramebufferId grid_fbo_{}, scratch_fbo_{}, probe_fbo_{}, strip_fbo_{};
        std::array<mgl::FramebufferId, 2> vis_fbo_{};
        bool ms_ok_ = false;
        std::string ms_reason_;
        int n_ = 0;
        int max_samples_ = 0, max_color_texture_samples_ = 0;
        std::vector<Pos> positions_, vis_positions_;
        bool initial_sample_shading_ = true;
        GLfloat initial_min_sample_shading_ = -1.0f;
        int offset_bits_ = 0;
        double min_offset_ = 0.0, max_offset_ = 0.0;
        mgl::ErrorCode grid_error_ = mgl::ErrorCode::NoError;
        mgl::ErrorCode rings_error_ = mgl::ErrorCode::NoError;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es32-advanced-blend-and-sample-shading",
        .title = "Advanced blend equations and sample shading",
        .min_es = {3, 2},
        .summary = "The 15 advanced blend equations (MULTIPLY ... HSL_LUMINOSITY) with layout(blend_support_*) and glBlendBarrier, compared texel by texel with a CPU "
                   "model of ES 3.2 section 15.1.5.3 and their draw-time errors; sample shading (glMinSampleShading, gl_SampleID, gl_SamplePosition, "
                   "gl_SampleMaskIn, gl_NumSamples, the sample qualifier, interpolateAtCentroid/Sample/Offset) read back per sample with texelFetch(sampler2DMS).",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<AdvancedBlendAndSampleShading>(argc, argv, info);
}
