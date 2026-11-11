// es32-float-targets-and-indexed-blend
//
// Two ES 3.2 ideas that belong together, because blending decides what a floating-point color buffer is good for:
//
//   1. Floating-point color targets. ES 3.2 makes R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F and R11F_G11F_B10F
//      color-renderable and required (Table 8.10, "CR" and "Req. rend."; framebuffer formats in 9.4.3). They hold values
//      outside [0, 1] (HDR 3.75, -2.5, 65504, 2^100) that are read back exactly with glReadPixels(RGBA, FLOAT)
//      (16.1.2, 16.1.3). Blending into them does not clamp (15.1.5), except that blending into a 32-bit float target
//      is an error in core ES 3.2 unless GL_EXT_float_blend is present: the demo computes that expectation from
//      glGet / HasExtension and reports the branch it took.
//   2. Indexed blend state (15.1.5, 15.2.2, 20.1): glEnablei / glDisablei / glIsEnabledi(BLEND, i),
//      glBlendFunci / glBlendFuncSeparatei / glBlendEquationi / glBlendEquationSeparatei / glColorMaski give every
//      draw buffer of one MRT framebuffer its own blend state; glGetIntegeri_v / glGetBooleani_v read it back, the
//      non-indexed commands reset ALL buffers, and an index >= MAX_DRAW_BUFFERS raises INVALID_VALUE.
//
// Image (256 x 256 window, default framebuffer RGBA8):
//   * top 160 rows: a small 3D scene (checkered floor, three boxes, two emissive spheres with additive glow halos,
//     one directional and two point lights with intensities up to 6) rendered into an RGBA16F target with a depth
//     buffer. Left half of the picture: what an 8-bit target would hold (min(hdr, 1): the lamps and highlights clip to
//     flat white). Right half: Reinhard tone mapping of the same HDR values.
//   * 32 rows: seven 16 x 16 float-format textures (R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F, R11F_G11F_B10F)
//     filled with a pattern of signed, HDR values, shown through a signed tone curve (grey = 0).
//   * 64 rows: the four attachments (RGBA8, RGBA16F, R11F_G11F_B10F, RGBA16F) of one MRT framebuffer after a single
//     draw with four different per-buffer blend states.
//
// Validation: every float texel is compared with a CPU model (exact for representable values); the MRT results of
// eight blend scenarios are compared with a CPU implementation of Table 15.1 / 15.2 per buffer; all indexed state is
// read back; selected pixels of the HDR scene are compared with a small CPU ray tracer; the tone-mapped window pixels
// are compared with the tone-mapping formula; errors are checked.

#include <demo/check.hpp>
#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;
using demo::Rgba8;
using Rgbaf = std::array<float, 4>;

namespace
{
    constexpr GLboolean gl_bool(bool b) { return b ? GL_TRUE : GL_FALSE; }
    constexpr GLint value_of(mgl::BlendFactor f) { return static_cast<GLint>(static_cast<GLenum>(f)); }
    constexpr GLint value_of(mgl::BlendEquation e) { return static_cast<GLint>(static_cast<GLenum>(e)); }

    // =============================================================================================
    // Numeric helpers
    // =============================================================================================

    /// True when `v` is exactly representable in a floating-point format with a 5-bit exponent (bias 15) and
    /// `mantissa_bits` mantissa bits, in the normal range up to `max_finite` (0 is representable, denormals are
    /// avoided on purpose: ES 2.3.4.2 allows them to be flushed).
    bool fits_float(float v, int mantissa_bits, float max_finite)
    {
        if (v == 0.0f) return true;
        const float a = std::fabs(v);
        if (!(a >= 0x1p-14f) || a > max_finite) return false;
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(a);
        const std::uint32_t low = (1u << (23 - mantissa_bits)) - 1u;
        return (bits & low) == 0u;
    }
    bool half_exact(float v) { return fits_float(v, 10, 65504.0f); }

    float clamp01(float v) { return std::clamp(v, 0.0f, 1.0f); }

    // =============================================================================================
    // Part 1: the seven float formats
    // =============================================================================================
    constexpr int kPatternSize = 16;

    enum class FloatClass
    {
        Half,    // 16-bit floats: 1 sign, 5 exponent, 10 mantissa bits (2.3.4.2)
        Single,  // IEEE 32-bit floats
        Packed   // R11F_G11F_B10F: unsigned 11/11/10-bit floats (2.3.4.3, 2.3.4.4), no alpha
    };

    struct FormatCase
    {
        std::string_view name;
        mgl::SizedInternalFormat format;
        mgl::InternalFormat renderbuffer_format;
        int stored;                // number of leading components that exist; A exists only when stored == 4
        FloatClass cls;
        std::array<int, 4> bits;   // Table 8.10 minimum bits per component (0 = absent)
    };

    constexpr std::array<FormatCase, 7> kFormats = {{
        {"R16F", mgl::SizedInternalFormat::R16F, mgl::InternalFormat::R16F, 1, FloatClass::Half, {16, 0, 0, 0}},
        {"RG16F", mgl::SizedInternalFormat::Rg16F, mgl::InternalFormat::Rg16F, 2, FloatClass::Half, {16, 16, 0, 0}},
        {"RGBA16F", mgl::SizedInternalFormat::Rgba16F, mgl::InternalFormat::Rgba16F, 4, FloatClass::Half, {16, 16, 16, 16}},
        {"R32F", mgl::SizedInternalFormat::R32F, mgl::InternalFormat::R32F, 1, FloatClass::Single, {32, 0, 0, 0}},
        {"RG32F", mgl::SizedInternalFormat::Rg32F, mgl::InternalFormat::Rg32F, 2, FloatClass::Single, {32, 32, 0, 0}},
        {"RGBA32F", mgl::SizedInternalFormat::Rgba32F, mgl::InternalFormat::Rgba32F, 4, FloatClass::Single, {32, 32, 32, 32}},
        {"R11F_G11F_B10F", mgl::SizedInternalFormat::R11FG11FB10F, mgl::InternalFormat::R11FG11FB10F, 3, FloatClass::Packed,
         {11, 11, 10, 0}},
    }};

    bool is_32bit(const FormatCase& f) { return f.cls == FloatClass::Single; }

    /// pattern.frag: component c of pixel (x, y).
    float pattern_value(int c, int x, int y)
    {
        switch (c)
        {
        case 0:
        {
            float r = 0.09375f;  // 1.5 * 2^-4
            for (int i = 0; i < x; ++i) r *= 2.0f;
            return (y & 1) ? -r : r;
        }
        case 1: return static_cast<float>(x - 8) * 0.5f + static_cast<float>(y - 8) * 0.125f;
        case 2: return (y & 1) ? -2.5f : 3.75f;
        default: return 0.25f * static_cast<float>(x + 1);
        }
    }

    struct ProbeValues
    {
        Rgbaf exact;     // pixel (15,15): range extremes that the format can hold
        Rgbaf rounding;  // pixel (14,15): 1.4, which no format represents exactly
    };

    ProbeValues probes_for(FloatClass cls)
    {
        constexpr Rgbaf kRounding = {1.4f, 1.4f, 1.4f, 1.4f};
        switch (cls)
        {
        case FloatClass::Half: return {{65504.0f, -65504.0f, 0x1p-14f, 6.0f}, kRounding};  // largest finite half, smallest normal
        case FloatClass::Single: return {{0x1p100f, -0x1p100f, 1000000.5f, 0x1p-100f}, kRounding};  // far outside half range
        case FloatClass::Packed: return {{65024.0f, 100000.0f, 100000.0f, 0.0f}, kRounding};  // largest R11, then values above the maxima
        }
        return {};
    }

    /// The value a component reads back as when the pattern shader writes `raw` to the format.
    float stored_component(const FormatCase& fc, int c, float raw)
    {
        if (c >= fc.stored) return c == 3 ? 1.0f : 0.0f;  // missing G, B, A read as 0, 0, 1 (ES 3.2 16.1.3)
        if (fc.cls == FloatClass::Packed)
        {
            // Unsigned formats: negative values are converted to zero, finite values above the maximum to the maximum (2.3.4.3 / 2.3.4.4).
            raw = std::max(raw, 0.0f);
            return std::min(raw, c == 2 ? 64512.0f : 65024.0f);
        }
        return raw;
    }

    float raw_pattern(const FormatCase& fc, int c, int x, int y)
    {
        const ProbeValues p = probes_for(fc.cls);
        if (x == 15 && y == 15) return p.exact[static_cast<std::size_t>(c)];
        if (x == 14 && y == 15) return p.rounding[static_cast<std::size_t>(c)];
        return pattern_value(c, x, y);
    }

    /// Does `got` agree with the model for component c of pixel (x, y)? Exact everywhere except the "rounding" probe.
    bool pattern_component_ok(const FormatCase& fc, int c, int x, int y, float got)
    {
        const float want = stored_component(fc, c, raw_pattern(fc, c, x, y));
        const bool rounding_probe = x == 14 && y == 15 && c < fc.stored;
        if (!rounding_probe) return got == want;
        switch (fc.cls)
        {
        case FloatClass::Single: return got == want;
        // 16-bit floats: the spec names no rounding rule for fragment outputs (2.3.4.2); accept the two neighbours of 1.4
        // (one half ulp at 1.4 is 2^-11, so +-2^-10 covers round-to-nearest and either directed rounding).
        case FloatClass::Half: return std::fabs(got - 1.4f) <= 0x1p-10f;
        // Packed: "rounded to the closest representable finite value ... implementations are allowed to always round
        // in the direction of zero" (2.3.4.3, 2.3.4.4): 1.4 lies between 89/64 and 90/64 (11-bit) or 44/32 and 45/32 (10-bit).
        case FloatClass::Packed:
            return got == 1.40625f || got == (c == 2 ? 1.375f : 1.390625f);
        }
        return false;
    }

    // =============================================================================================
    // Part 2: blending, CPU model, scenarios
    // =============================================================================================
    constexpr int kMrtSize = 64;      // each MRT attachment is 64 x 64 = four 32 x 32 quadrants
    constexpr int kQuadSize = 32;
    constexpr int kBuffers = 4;

    enum class BufKind
    {
        Unorm8,  // RGBA8
        Half,    // RGBA16F
        Packed   // R11F_G11F_B10F (no alpha)
    };

    constexpr std::array<BufKind, kBuffers> kBufKinds = {BufKind::Unorm8, BufKind::Half, BufKind::Packed, BufKind::Half};
    constexpr std::array<std::string_view, kBuffers> kBufNames = {"RGBA8", "RGBA16F", "R11F_G11F_B10F", "RGBA16F"};
    constexpr std::array<mgl::SizedInternalFormat, kBuffers> kBufFormats = {
        mgl::SizedInternalFormat::Rgba8, mgl::SizedInternalFormat::Rgba16F, mgl::SizedInternalFormat::R11FG11FB10F,
        mgl::SizedInternalFormat::Rgba16F};

    /// Blend state of one draw buffer as the CPU model knows it.
    struct BufState
    {
        bool enable = false;
        mgl::BlendEquation eq_rgb = mgl::BlendEquation::FuncAdd;
        mgl::BlendEquation eq_alpha = mgl::BlendEquation::FuncAdd;
        mgl::BlendFactor src_rgb = mgl::BlendFactor::One;
        mgl::BlendFactor dst_rgb = mgl::BlendFactor::Zero;
        mgl::BlendFactor src_alpha = mgl::BlendFactor::One;
        mgl::BlendFactor dst_alpha = mgl::BlendFactor::Zero;
        std::array<bool, 4> mask = {true, true, true, true};
        friend bool operator==(const BufState&, const BufState&) = default;
    };

    std::string describe(const BufState& s)
    {
        return std::format("{} {}/{}({},{} | {},{}) mask={}{}{}{}", s.enable ? "on" : "off", mgl::to_string(s.eq_rgb),
                           mgl::to_string(s.eq_alpha), mgl::to_string(s.src_rgb), mgl::to_string(s.dst_rgb),
                           mgl::to_string(s.src_alpha), mgl::to_string(s.dst_alpha), s.mask[0] ? 'T' : 'F',
                           s.mask[1] ? 'T' : 'F', s.mask[2] ? 'T' : 'F', s.mask[3] ? 'T' : 'F');
    }

    /// The same state as glGet* returns it: raw enumerator values (so that no integer is cast back to an enum).
    struct RawState
    {
        bool enable = false;
        GLint eq_rgb = 0, eq_alpha = 0, src_rgb = 0, dst_rgb = 0, src_alpha = 0, dst_alpha = 0;
        std::array<bool, 4> mask = {};
        friend bool operator==(const RawState&, const RawState&) = default;
    };

    RawState raw(const BufState& s)
    {
        return {s.enable, value_of(s.eq_rgb), value_of(s.eq_alpha), value_of(s.src_rgb), value_of(s.dst_rgb),
                value_of(s.src_alpha), value_of(s.dst_alpha), s.mask};
    }

    std::string describe(const RawState& s)
    {
        return std::format("{} eq={:#x}/{:#x} src={:#x}/{:#x} dst={:#x}/{:#x} mask={}{}{}{}", s.enable ? "on" : "off", s.eq_rgb,
                           s.eq_alpha, s.src_rgb, s.src_alpha, s.dst_rgb, s.dst_alpha, s.mask[0] ? 'T' : 'F',
                           s.mask[1] ? 'T' : 'F', s.mask[2] ? 'T' : 'F', s.mask[3] ? 'T' : 'F');
    }

    /// One batch of blend-state commands for a draw buffer (indexed form) or for all buffers (non-indexed form).
    struct Op
    {
        std::optional<bool> enable;
        int func = 0;  // 1: BlendFunc[i], 2: BlendFuncSeparate[i]
        mgl::BlendFactor src_rgb = mgl::BlendFactor::One, dst_rgb = mgl::BlendFactor::Zero;
        mgl::BlendFactor src_alpha = mgl::BlendFactor::One, dst_alpha = mgl::BlendFactor::Zero;
        int eq = 0;    // 1: BlendEquation[i], 2: BlendEquationSeparate[i]
        mgl::BlendEquation eq_rgb = mgl::BlendEquation::FuncAdd, eq_alpha = mgl::BlendEquation::FuncAdd;
        std::optional<std::array<bool, 4>> mask;

        Op& on(bool e = true) { enable = e; return *this; }
        Op& f(mgl::BlendFactor s, mgl::BlendFactor d) { func = 1; src_rgb = src_alpha = s; dst_rgb = dst_alpha = d; return *this; }
        Op& fs(mgl::BlendFactor sr, mgl::BlendFactor dr, mgl::BlendFactor sa, mgl::BlendFactor da)
        {
            func = 2; src_rgb = sr; dst_rgb = dr; src_alpha = sa; dst_alpha = da; return *this;
        }
        Op& e(mgl::BlendEquation m) { eq = 1; eq_rgb = eq_alpha = m; return *this; }
        Op& es(mgl::BlendEquation r, mgl::BlendEquation a) { eq = 2; eq_rgb = r; eq_alpha = a; return *this; }
        Op& m(bool r, bool g, bool b, bool a) { mask = std::array<bool, 4>{r, g, b, a}; return *this; }
    };

    /// The state after the commands of `op` ran (each field changes only when its command is part of the op).
    BufState apply_to(BufState s, const Op& op)
    {
        if (op.enable) s.enable = *op.enable;
        if (op.func != 0) { s.src_rgb = op.src_rgb; s.dst_rgb = op.dst_rgb; s.src_alpha = op.src_alpha; s.dst_alpha = op.dst_alpha; }
        if (op.eq != 0) { s.eq_rgb = op.eq_rgb; s.eq_alpha = op.eq_alpha; }
        if (op.mask) s.mask = *op.mask;
        return s;
    }

    /// Indexed commands: ES 3.2 15.1.5 (Enablei/Disablei), 15.1.5.1 (BlendEquationi), 15.1.5.2 (BlendFunci), 15.2.2 (ColorMaski).
    void issue_indexed(GLuint i, const Op& op)
    {
        if (op.enable)
        {
            if (*op.enable) mgl::glEnablei(mgl::Capability::Blend, i);
            else mgl::glDisablei(mgl::Capability::Blend, i);
        }
        if (op.func == 1) mgl::glBlendFunci(i, op.src_rgb, op.dst_rgb);
        else if (op.func == 2) mgl::glBlendFuncSeparatei(i, op.src_rgb, op.dst_rgb, op.src_alpha, op.dst_alpha);
        if (op.eq == 1) mgl::glBlendEquationi(i, op.eq_rgb);
        else if (op.eq == 2) mgl::glBlendEquationSeparatei(i, op.eq_rgb, op.eq_alpha);
        if (op.mask)
            mgl::glColorMaski(i, gl_bool((*op.mask)[0]), gl_bool((*op.mask)[1]), gl_bool((*op.mask)[2]), gl_bool((*op.mask)[3]));
    }

    /// Non-indexed commands: they set the state of ALL draw buffers.
    void issue_global(const Op& op)
    {
        if (op.enable)
        {
            if (*op.enable) mgl::glEnable(mgl::Capability::Blend);
            else mgl::glDisable(mgl::Capability::Blend);
        }
        if (op.func == 1) mgl::glBlendFunc(op.src_rgb, op.dst_rgb);
        else if (op.func == 2) mgl::glBlendFuncSeparate(op.src_rgb, op.dst_rgb, op.src_alpha, op.dst_alpha);
        if (op.eq == 1) mgl::glBlendEquation(op.eq_rgb);
        else if (op.eq == 2) mgl::glBlendEquationSeparate(op.eq_rgb, op.eq_alpha);
        if (op.mask)
            mgl::glColorMask(gl_bool((*op.mask)[0]), gl_bool((*op.mask)[1]), gl_bool((*op.mask)[2]), gl_bool((*op.mask)[3]));
    }

    // ---- CPU model of one blended draw: Tables 15.1 and 15.2, 15.1.5.4, 15.1.6, 15.2.2 -----------------------------
    Rgbaf weights(mgl::BlendFactor f, const Rgbaf& s, const Rgbaf& d, const Rgbaf& c)
    {
        const float saturate = std::min(s[3], 1.0f - d[3]);  // Table 15.2 footnote: f = min(As, 1 - Ad)
        switch (f)
        {
        case mgl::BlendFactor::Zero: return {0.0f, 0.0f, 0.0f, 0.0f};
        case mgl::BlendFactor::One: return {1.0f, 1.0f, 1.0f, 1.0f};
        case mgl::BlendFactor::SrcColor: return s;
        case mgl::BlendFactor::OneMinusSrcColor: return {1.0f - s[0], 1.0f - s[1], 1.0f - s[2], 1.0f - s[3]};
        case mgl::BlendFactor::DstColor: return d;
        case mgl::BlendFactor::OneMinusDstColor: return {1.0f - d[0], 1.0f - d[1], 1.0f - d[2], 1.0f - d[3]};
        case mgl::BlendFactor::SrcAlpha: return {s[3], s[3], s[3], s[3]};
        case mgl::BlendFactor::OneMinusSrcAlpha: return {1.0f - s[3], 1.0f - s[3], 1.0f - s[3], 1.0f - s[3]};
        case mgl::BlendFactor::DstAlpha: return {d[3], d[3], d[3], d[3]};
        case mgl::BlendFactor::OneMinusDstAlpha: return {1.0f - d[3], 1.0f - d[3], 1.0f - d[3], 1.0f - d[3]};
        case mgl::BlendFactor::ConstantColor: return c;
        case mgl::BlendFactor::OneMinusConstantColor: return {1.0f - c[0], 1.0f - c[1], 1.0f - c[2], 1.0f - c[3]};
        case mgl::BlendFactor::ConstantAlpha: return {c[3], c[3], c[3], c[3]};
        case mgl::BlendFactor::OneMinusConstantAlpha: return {1.0f - c[3], 1.0f - c[3], 1.0f - c[3], 1.0f - c[3]};
        case mgl::BlendFactor::SrcAlphaSaturate: return {saturate, saturate, saturate, 1.0f};
        }
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    /// Table 15.1 for one component. For the half-float target, `exact` is cleared when a product or the sum is not
    /// representable in 16 bits: the scenario table is built so that this never happens (every intermediate value is a
    /// small dyadic number), which makes an exact comparison with a hardware blender legitimate.
    float combine(mgl::BlendEquation e, float s, float fs, float d, float fd, BufKind kind, bool& exact)
    {
        float a = s * fs, b = d * fd, r = 0.0f;
        switch (e)
        {
        case mgl::BlendEquation::FuncAdd: r = a + b; break;
        case mgl::BlendEquation::FuncSubtract: r = a - b; break;
        case mgl::BlendEquation::FuncReverseSubtract: r = b - a; break;
        case mgl::BlendEquation::Min: return std::min(s, d);  // MIN and MAX ignore the factors
        case mgl::BlendEquation::Max: return std::max(s, d);
        default: break;
        }
        if (kind == BufKind::Half && !(half_exact(a) && half_exact(b) && half_exact(r))) exact = false;
        return r;
    }

    /// What one draw buffer holds after a blended draw, in native units: bytes for RGBA8, floats otherwise.
    /// `dst` is what the buffer held before (a stored value), `src` the fragment output.
    std::array<double, 4> model_draw(BufKind kind, const BufState& st, const Rgbaf& constant, Rgbaf src, Rgbaf dst, bool& exact)
    {
        const bool fixed = kind == BufKind::Unorm8;
        Rgbaf d = dst, s = src, c = constant;
        if (kind == BufKind::Packed) d[3] = 1.0f;  // "If a color buffer has no A value, then Ad is taken to be 1" (15.1.5.5)
        Rgbaf res = s;
        if (st.enable)
        {
            // Fixed-point: source, destination and factors are clamped to [0,1]; the constant color is clamped for an
            // unsigned normalized destination. Floating-point: no clamping (15.1.5, 15.1.5.4).
            if (fixed)
                for (Rgbaf* v : {&s, &d, &c})
                    for (float& x : *v) x = clamp01(x);
            Rgbaf sr = weights(st.src_rgb, s, d, c), dr = weights(st.dst_rgb, s, d, c);
            Rgbaf sa = weights(st.src_alpha, s, d, c), da = weights(st.dst_alpha, s, d, c);
            if (fixed)
                for (Rgbaf* v : {&sr, &dr, &sa, &da})
                    for (float& x : *v) x = clamp01(x);
            for (std::size_t i = 0; i < 3; ++i) res[i] = combine(st.eq_rgb, s[i], sr[i], d[i], dr[i], kind, exact);
            res[3] = combine(st.eq_alpha, s[3], sa[3], d[3], da[3], kind, exact);
        }
        // Storing: fixed point clamps to [0,1] (15.1.6); the unsigned packed floats turn negative values into 0 (2.3.4.3).
        if (fixed)
            for (float& x : res) x = clamp01(x);
        if (kind == BufKind::Packed)
            for (std::size_t i = 0; i < 3; ++i) res[i] = std::clamp(res[i], 0.0f, i == 2 ? 64512.0f : 65024.0f);
        std::array<double, 4> out{};
        for (std::size_t i = 0; i < 4; ++i)
        {
            const float v = st.mask[i] ? res[i] : dst[i];  // ColorMaski: masked components keep the old value (15.2.2)
            out[i] = fixed ? static_cast<double>(std::lround(clamp01(v) * 255.0f)) : static_cast<double>(v);
        }
        if (kind == BufKind::Packed)
        {
            out[3] = 1.0;  // no alpha channel: reads back as 1
            for (std::size_t i = 0; i < 3; ++i)
                if (!fits_float(static_cast<float>(out[i]), i == 2 ? 5 : 6, i == 2 ? 64512.0f : 65024.0f)) exact = false;
        }
        else if (kind == BufKind::Half)
        {
            for (double v : out)
                if (!half_exact(static_cast<float>(v))) exact = false;
        }
        return out;
    }

    // ---- the quadrant data: dst values are drawn first (blending off), src values are blended on top --------------
    constexpr float k255(int v) { return static_cast<float>(v) / 255.0f; }

    struct QuadData
    {
        std::array<Rgbaf, kBuffers> dst;
        std::array<Rgbaf, kBuffers> src;
    };

    // Buffer 0 (RGBA8) uses multiples of 1/255 (exact in the buffer), the float buffers dyadic numbers (exact in 16-bit
    // floats), buffer 2 (R11F_G11F_B10F) non-negative numbers for its destination. Sources may be negative or above 1.
    const std::array<QuadData, 4> kQuads = {{
        {{Rgbaf{k255(51), k255(102), k255(204), k255(179)}, Rgbaf{1.0f, 2.0f, 0.5f, 0.5f}, Rgbaf{1.0f, 2.0f, 0.5f, 0.5f},
          Rgbaf{0.25f, 0.5f, 4.0f, 1.0f}},
         {Rgbaf{k255(204), k255(102), k255(51), k255(153)}, Rgbaf{3.0f, -2.0f, 0.5f, 2.5f}, Rgbaf{2.0f, 1.0f, -0.5f, 0.25f},
          Rgbaf{0.5f, 4.0f, 2.0f, 0.75f}}},
        {{Rgbaf{k255(230), k255(77), k255(26), k255(128)}, Rgbaf{0.5f, 0.25f, 8.0f, 2.0f}, Rgbaf{0.25f, 4.0f, 0.125f, 0.5f},
          Rgbaf{2.0f, 1.0f, 0.5f, 0.25f}},
         {Rgbaf{k255(255), k255(0), k255(128), k255(255)}, Rgbaf{0.25f, 8.0f, 0.125f, 1.0f}, Rgbaf{5.0f, 0.25f, 3.0f, 0.5f},
          Rgbaf{-1.0f, 2.0f, 6.0f, 3.0f}}},
        {{Rgbaf{k255(26), k255(179), k255(128), k255(255)}, Rgbaf{4.0f, 0.5f, 1.0f, 1.0f}, Rgbaf{6.0f, 0.5f, 2.0f, 1.0f},
          Rgbaf{0.125f, 3.0f, 0.75f, 0.5f}},
         {Rgbaf{1.5f, -0.5f, 0.2f, 2.5f}, Rgbaf{2.0f, 2.0f, 2.0f, 2.0f}, Rgbaf{0.5f, 0.5f, 0.5f, 0.5f}, Rgbaf{1.0f, 1.0f, 1.0f, 0.0f}}},
        {{Rgbaf{k255(128), k255(128), k255(128), k255(77)}, Rgbaf{0.0f, 0.0f, 0.0f, 0.0f}, Rgbaf{10.0f, 0.5f, 1.5f, 1.0f},
          Rgbaf{1.5f, 1.5f, 1.5f, 1.5f}},
         {Rgbaf{k255(26), k255(230), k255(153), k255(102)}, Rgbaf{1.0f, 0.5f, 0.25f, 0.125f}, Rgbaf{8.0f, 4.0f, 2.0f, 1.0f},
          Rgbaf{0.75f, 0.375f, 3.0f, 5.0f}}},
    }};

    constexpr std::array<std::pair<int, int>, 4> kQuadOrigin = {{{0, 0}, {kQuadSize, 0}, {0, kQuadSize}, {kQuadSize, kQuadSize}}};

    struct Scenario
    {
        std::string name;
        Rgbaf constant = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<std::optional<Op>, kBuffers> junk;          // indexed commands issued first; the global ones must undo them
        std::optional<Op> global;                              // non-indexed commands (all buffers)
        std::array<std::optional<Op>, kBuffers> per_buffer;    // indexed commands issued last
        /// zink on lavapipe (Mesa) blends and stores channels that glColorMask masked when blending is enabled and R and A are written but
        /// G or B is masked (every buffer with the mask TFTT, TTFT or TFFT; verified with an experiment on all four buffers). ES 3.2 15.2.2:
        /// "r, g, b, and a indicate whether R, G, B, or A values, respectively, are written": a masked component is not written.
        bool zink_ignores_mask = false;
    };

    /// State of the buffers no indexed command touches, and of buffers >= 4: the initial state, or the global op's result.
    BufState base_state(const Scenario& sc)
    {
        BufState s;
        if (sc.global) s = apply_to(s, *sc.global);
        return s;
    }

    std::array<BufState, kBuffers> effective_states(const Scenario& sc)
    {
        std::array<BufState, kBuffers> out;
        for (std::size_t i = 0; i < kBuffers; ++i)
        {
            out[i] = base_state(sc);
            if (sc.per_buffer[i]) out[i] = apply_to(out[i], *sc.per_buffer[i]);
        }
        return out;
    }

    constexpr auto kZero = mgl::BlendFactor::Zero;
    constexpr auto kOne = mgl::BlendFactor::One;
    constexpr auto kAdd = mgl::BlendEquation::FuncAdd;

    std::vector<Scenario> build_scenarios()
    {
        using BF = mgl::BlendFactor;
        using BE = mgl::BlendEquation;
        std::vector<Scenario> v;

        {   // 0: the picture. Four buffers, four different results from one draw.
            Scenario s;
            s.name = "showcase: replace / additive / alpha-over / reverse-subtract";
            s.per_buffer[0] = Op{}.on(false).f(kOne, kZero);
            s.per_buffer[1] = Op{}.on(true).f(kOne, kOne).e(kAdd);
            s.per_buffer[2] = Op{}.on(true).f(BF::SrcAlpha, BF::OneMinusSrcAlpha).e(kAdd);
            s.per_buffer[3] = Op{}.on(true).f(kOne, kOne).e(BE::FuncReverseSubtract);
            v.push_back(std::move(s));
        }
        {   // 1: one non-indexed setup for all buffers, then indexed overrides of single fields
            Scenario s;
            s.name = "global blend state, then indexed overrides of some fields";
            s.global = Op{}.on(true).f(BF::SrcAlpha, BF::OneMinusSrcAlpha).e(kAdd);
            s.per_buffer[2] = Op{}.f(kOne, kOne).es(BE::Max, BE::Min);
            s.per_buffer[3] = Op{}.on(false);
            v.push_back(std::move(s));
        }
        {   // 2: separate RGB/alpha functions and equations, different on every buffer
            Scenario s;
            s.name = "separate functions and equations, different per buffer";
            s.per_buffer[0] = Op{}.on().fs(BF::SrcAlpha, BF::OneMinusSrcAlpha, kOne, BF::OneMinusSrcAlpha).e(kAdd);
            s.per_buffer[1] = Op{}.on().f(kOne, kOne).es(BE::FuncReverseSubtract, BE::Max);
            s.per_buffer[2] = Op{}.on().f(BF::SrcColor, BF::DstColor).e(BE::Min);
            s.per_buffer[3] = Op{}.on().fs(BF::DstColor, kZero, kZero, BF::SrcAlpha).es(BE::FuncSubtract, BE::Max);
            v.push_back(std::move(s));
        }
        {   // 3: write masks (ColorMaski) per buffer
            Scenario s;
            s.name = "per-buffer write masks";
            s.per_buffer[0] = Op{}.m(true, false, true, false);
            s.per_buffer[1] = Op{}.on().f(kOne, kOne).m(false, false, true, true);
            s.per_buffer[2] = Op{}.m(false, true, false, true);
            s.per_buffer[3] = Op{}.m(true, true, true, false);
            v.push_back(std::move(s));
        }
        {   // 3b: write masks combined with blending: masked components keep the destination value, the others are blended
            Scenario s;
            s.name = "write masks with blending enabled";
            s.per_buffer[0] = Op{}.on().f(kOne, kOne).e(kAdd).m(true, false, true, true);
            s.per_buffer[1] = Op{}.on().f(BF::SrcAlpha, BF::OneMinusSrcAlpha).e(kAdd).m(false, false, true, true);
            s.per_buffer[2] = Op{}.on().f(kOne, kOne).e(kAdd).m(true, true, false, true);
            s.per_buffer[3] = Op{}.on().f(kOne, kOne).e(BE::FuncReverseSubtract).m(false, true, true, true);
            v.push_back(std::move(s));
        }
        {   // 3c: the same partial mask on every buffer (zink/lavapipe does not honor it while blending)
            Scenario s;
            s.name = "write mask TFTT on every buffer with blending enabled";
            s.zink_ignores_mask = true;
            s.global = Op{}.on().f(kOne, kOne).e(kAdd).m(true, false, true, true);
            v.push_back(std::move(s));
        }
        {   // 4: disabled buffers ignore their functions; enabled ones use them
            Scenario s;
            s.name = "enable flag per buffer, functions kept while disabled";
            s.per_buffer[0] = Op{}.f(kOne, kOne);
            s.per_buffer[1] = Op{}.on().f(BF::DstColor, kZero);
            s.per_buffer[2] = Op{}.e(BE::FuncSubtract).f(kOne, kOne);
            s.per_buffer[3] = Op{}.on().f(BF::OneMinusDstAlpha, BF::DstAlpha);
            v.push_back(std::move(s));
        }
        {   // 5: the constant blend color is clamped only for the fixed-point buffer
            Scenario s;
            s.name = "constant color factors: clamped for RGBA8, unclamped for the float buffers";
            s.constant = {0.5f, 2.0f, -0.25f, 0.75f};
            s.global = Op{}.on().fs(BF::ConstantColor, BF::OneMinusConstantColor, BF::ConstantAlpha, BF::OneMinusConstantAlpha)
                           .es(kAdd, kAdd);
            v.push_back(std::move(s));
        }
        {   // 6: non-indexed commands undo any per-buffer state
            Scenario s;
            s.name = "non-indexed commands reset every buffer";
            s.junk[0] = Op{}.on().f(kOne, kOne).m(false, false, false, false);
            s.junk[1] = Op{}.on().e(BE::Max).m(true, false, false, false);
            s.junk[2] = Op{}.on(false).fs(BF::DstColor, kZero, BF::SrcAlpha, kOne).es(BE::Min, BE::FuncSubtract).m(false, true, true, true);
            s.junk[3] = Op{}.on().f(BF::ConstantAlpha, BF::SrcColor).m(true, true, false, true);
            s.global = Op{}.on().f(BF::SrcAlpha, kOne).e(BE::FuncReverseSubtract).m(true, true, true, true);
            v.push_back(std::move(s));
        }
        {   // 7: SRC_ALPHA_SATURATE and DST_ALPHA with and without a destination alpha channel
            Scenario s;
            s.name = "dst-alpha factors: a buffer without alpha has Ad = 1";
            s.global = Op{}.on().f(BF::SrcAlphaSaturate, BF::OneMinusDstAlpha).e(kAdd);
            v.push_back(std::move(s));
        }
        return v;
    }

    // =============================================================================================
    // Part 3: HDR scene (CPU ray tracer used as reference)
    // =============================================================================================
    constexpr int kSceneW = 256, kSceneH = 160, kSceneY = 96;  // scene region in the window: x 0..255, y 96..255
    constexpr double kExposure = 1.2;

    struct D3
    {
        double x = 0, y = 0, z = 0;
    };
    D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    D3 operator*(D3 a, D3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
    double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    D3 normalize(D3 a) { return a * (1.0 / std::sqrt(dot(a, a))); }
    double axis(const D3& a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }

    struct BoxDef
    {
        D3 center, half, albedo;
    };
    struct SphereDef
    {
        D3 center;
        double radius;
        D3 emission;     // emissive radiance (HDR)
        D3 light_color;  // point light placed at the center
        D3 glow_color;   // additive halo
        double glow_radius;
    };

    constexpr std::array<BoxDef, 3> kBoxes = {{
        {{-1.7, 0.5, 0.2}, {0.5, 0.5, 0.5}, {0.9, 0.45, 0.15}},
        {{0.2, 0.8, -1.0}, {0.5, 0.8, 0.5}, {0.25, 0.4, 0.9}},
        {{1.9, 0.4, 0.5}, {0.4, 0.4, 0.4}, {0.3, 0.8, 0.35}},
    }};
    constexpr std::array<SphereDef, 2> kSpheres = {{
        {{-0.7, 1.55, 1.4}, 0.2, {12.0, 6.0, 2.5}, {6.0, 3.0, 1.0}, {2.5, 1.2, 0.4}, 1.1},
        {{1.0, 1.0, 1.9}, 0.18, {3.0, 8.0, 12.0}, {1.5, 3.5, 6.0}, {0.6, 1.6, 2.4}, 0.9},
    }};
    constexpr D3 kAmbient = {0.05, 0.06, 0.08};
    constexpr D3 kSunColor = {0.9, 0.85, 0.75};
    constexpr D3 kSunDirRaw = {-0.35, 0.8, 0.45};
    constexpr D3 kClear = {0.015, 0.02, 0.035};
    constexpr double kFloorX = 10.0, kFloorZMin = -30.0, kFloorZMax = 6.0;

    struct Camera
    {
        D3 eye, f, r, u;
        double tan_half = 0, aspect = 1;
    };

    Camera make_camera()
    {
        Camera c;
        c.eye = {0.0, 2.4, 6.2};
        c.f = normalize(D3{0.0, 0.8, 0.0} - c.eye);
        c.r = normalize(cross(c.f, D3{0, 1, 0}));
        c.u = cross(c.r, c.f);
        c.tan_half = std::tan(static_cast<double>(dm::radians(42.0f)) * 0.5);
        c.aspect = static_cast<double>(kSceneW) / kSceneH;
        return c;
    }

    D3 ray_dir(const Camera& c, double px, double py)
    {
        const double nx = px / kSceneW * 2.0 - 1.0, ny = py / kSceneH * 2.0 - 1.0;
        return normalize(c.f + c.r * (nx * c.tan_half * c.aspect) + c.u * (ny * c.tan_half));
    }

    bool project(const Camera& c, D3 p, double& px, double& py)
    {
        const D3 v = p - c.eye;
        const double z = dot(v, c.f);
        if (z <= 0.0) return false;
        px = (dot(v, c.r) / (z * c.tan_half * c.aspect) * 0.5 + 0.5) * kSceneW;
        py = (dot(v, c.u) / (z * c.tan_half) * 0.5 + 0.5) * kSceneH;
        return true;
    }

    struct Hit
    {
        double t = 0;
        int id = -1;  // 0 floor, 1..3 boxes, 4..5 spheres
        D3 pos, normal, albedo, emission;
        bool floor = false;
    };

    std::optional<Hit> trace(const D3& o, const D3& d)
    {
        std::optional<Hit> best;
        auto consider = [&](const Hit& h) {
            if (h.t > 0.0 && (!best || h.t < best->t)) best = h;
        };
        if (d.y < -1e-12)
        {
            const double t = -o.y / d.y;
            const D3 p = o + d * t;
            if (std::fabs(p.x) <= kFloorX && p.z >= kFloorZMin && p.z <= kFloorZMax)
                consider({t, 0, p, {0, 1, 0}, {}, {}, true});
        }
        for (std::size_t b = 0; b < kBoxes.size(); ++b)
        {
            double tmin = -1e30, tmax = 1e30;
            int hit_axis = -1;
            double hit_sign = 0;
            bool miss = false;
            for (int a = 0; a < 3; ++a)
            {
                const double oa = axis(o, a), da = axis(d, a);
                const double lo = axis(kBoxes[b].center, a) - axis(kBoxes[b].half, a);
                const double hi = axis(kBoxes[b].center, a) + axis(kBoxes[b].half, a);
                if (std::fabs(da) < 1e-12)
                {
                    if (oa < lo || oa > hi) miss = true;
                    continue;
                }
                double t1 = (lo - oa) / da, t2 = (hi - oa) / da;
                if (t1 > t2) std::swap(t1, t2);
                if (t1 > tmin) { tmin = t1; hit_axis = a; hit_sign = da > 0 ? -1.0 : 1.0; }
                tmax = std::min(tmax, t2);
            }
            if (!miss && tmin <= tmax && hit_axis >= 0)
            {
                D3 n{};
                (hit_axis == 0 ? n.x : (hit_axis == 1 ? n.y : n.z)) = hit_sign;
                consider({tmin, 1 + static_cast<int>(b), o + d * tmin, n, kBoxes[b].albedo, {}, false});
            }
        }
        for (std::size_t s = 0; s < kSpheres.size(); ++s)
        {
            const D3 oc = o - kSpheres[s].center;
            const double b = dot(oc, d), c = dot(oc, oc) - kSpheres[s].radius * kSpheres[s].radius;
            const double disc = b * b - c;
            if (disc <= 0.0) continue;
            const double t = -b - std::sqrt(disc);
            const D3 p = o + d * t;
            consider({t, 4 + static_cast<int>(s), p, normalize(p - kSpheres[s].center), {}, kSpheres[s].emission, false});
        }
        return best;
    }

    D3 shade(const Hit& h)
    {
        D3 albedo = h.albedo;
        if (h.floor)
        {
            const int tile = (static_cast<int>(std::floor(h.pos.x)) + static_cast<int>(std::floor(h.pos.z))) & 1;
            albedo = D3{tile == 0 ? 0.65 : 0.15, tile == 0 ? 0.65 : 0.15, tile == 0 ? 0.65 : 0.15};
        }
        const D3 sun_dir = normalize(kSunDirRaw);
        D3 lit = kAmbient + kSunColor * std::max(dot(h.normal, sun_dir), 0.0);
        for (const SphereDef& s : kSpheres)
        {
            const D3 to_light = s.center - h.pos;
            const double d2 = dot(to_light, to_light);
            lit = lit + s.light_color * (std::max(dot(h.normal, normalize(to_light)), 0.0) / (1.0 + d2));
        }
        return albedo * lit + h.emission;
    }

    struct PixelRef
    {
        D3 color;
        int signature = 0;
        bool ambiguous = false;  // a glow billboard and the opaque surface are almost at the same depth
    };

    PixelRef reference_pixel(const Camera& cam, double px, double py)
    {
        const D3 d = ray_dir(cam, px, py);
        const std::optional<Hit> hit = trace(cam.eye, d);
        PixelRef r;
        r.color = hit ? shade(*hit) : kClear;
        r.signature = hit ? hit->id : 15;
        if (hit && hit->floor)
            r.signature |= (((static_cast<int>(std::floor(hit->pos.x)) + static_cast<int>(std::floor(hit->pos.z))) & 1) << 4);
        const double z_opaque = hit ? dot(hit->pos - cam.eye, cam.f) : 1e30;
        for (std::size_t g = 0; g < kSpheres.size(); ++g)
        {
            const SphereDef& s = kSpheres[g];
            const double z_b = dot(s.center - cam.eye, cam.f);
            const double t = z_b / dot(d, cam.f);
            const D3 p = cam.eye + d * t - s.center;
            const double u = dot(p, cam.r) / s.glow_radius, v = dot(p, cam.u) / s.glow_radius;
            const double r2 = u * u + v * v;
            if (r2 >= 1.0) continue;
            if (std::fabs(z_opaque - z_b) < 0.05) r.ambiguous = true;
            if (z_b < z_opaque)  // depth test LESS: the billboard is in front of the opaque surface
            {
                const double f = 1.0 - r2;
                r.color = r.color + s.glow_color * (f * f);
                r.signature |= 1 << (8 + static_cast<int>(g));
            }
        }
        return r;
    }

    /// A pixel is a usable probe when the same objects (and the same glow visibility) are seen at the pixel center and at
    /// eight points 1.2 pixels around it: then rasterization details (edges, tessellated sphere silhouettes, checker
    /// borders) cannot make the CPU reference and the GPU disagree about which surface is at the pixel.
    bool stable_probe(const Camera& cam, int px, int py)
    {
        const PixelRef center = reference_pixel(cam, px + 0.5, py + 0.5);
        if (center.ambiguous) return false;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                const PixelRef n = reference_pixel(cam, px + 0.5 + 1.2 * dx, py + 0.5 + 1.2 * dy);
                if (n.ambiguous || n.signature != center.signature) return false;
            }
        return true;
    }

    // =============================================================================================
    // The demo
    // =============================================================================================
    struct Probe
    {
        int x = 0, y = 0;
        std::string label;
    };

    class FloatTargetsAndIndexedBlend final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 256 || env.height < 256) throw demo::Skip("this demo composes a 256 x 256 picture; the surface is smaller");

            // --- state of a fresh context, read before any command of this demo changes it ----------------------------
            max_draw_buffers_ = demo::get_integer(mgl::GetParameter::MaxDrawBuffers);
            max_color_attachments_ = demo::get_integer(mgl::GetParameter::MaxColorAttachments);
            for (int i = 0; i < max_draw_buffers_; ++i) initial_.push_back(query_raw(static_cast<GLuint>(i)));
            initial_enabled_global_ = mgl::glIsEnabled(mgl::Capability::Blend);
            initial_src_rgb_global_ = demo::get_integer(mgl::GetParameter::BlendSrcRgb);
            mgl::glGetFloatv(mgl::GetParameter::BlendColor, initial_blend_color_.data());

            float_blend_ = mgl::HasExtension("GL_EXT_float_blend");
            half_float_ext_ = mgl::HasExtension("GL_EXT_color_buffer_half_float");
            color_buffer_float_ext_ = mgl::HasExtension("GL_EXT_color_buffer_float");

            pattern_ = demo::build_program(env, "shaders/quad.vert", "shaders/pattern.frag");
            outputs_ = demo::build_program(env, "shaders/quad.vert", "shaders/outputs.frag");
            scene_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            glow_ = demo::build_program(env, "shaders/glow.vert", "shaders/glow.frag");
            compose_ = demo::build_program(env, "shaders/quad.vert", "shaders/compose.frag");

            build_scene_geometry();
            create_format_targets();
            create_hdr_target();
            create_mrt_target();
            scenarios_ = build_scenarios();
            check_scenario_design();
            camera_ = make_camera();
            choose_probes();
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            // Dithering (initially enabled, 15.1.7) may add +-1 to 8-bit results; it is no part of this demo's subject.
            mgl::glDisable(mgl::Capability::Dither);

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            draw_patterns();
            draw_hdr_scene();
            showcase_state_ = run_scenario(scenarios_[0]);
            reset_blend_state();
            compose_window();
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            // The window holds the result of render(); read it before anything else.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            window_ = demo::read_rgba8(0, 0, 256, 256);

            std::cout << std::format("     extensions: GL_EXT_float_blend={} GL_EXT_color_buffer_float={} GL_EXT_color_buffer_half_float={}\n",
                                     float_blend_, color_buffer_float_ext_, half_float_ext_);
            check_float_formats(check);
            check_float_blending(check);
            check_renderbuffer_formats(check);
            check_hdr_scene(check);
            check_composite_panels(check);
            check_initial_indexed_state(check);
            check_scenarios(check);
            check_nonindexed_commands(check);
            check_indexed_errors(check);

            reset_blend_state();
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            reset_blend_state();
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                mgl::glDeleteFramebuffers(1, &fmt_fbo_[i]);
                mgl::glDeleteTextures(1, &fmt_tex_[i]);
            }
            mgl::glDeleteFramebuffers(1, &ctrl_fbo_);
            mgl::glDeleteTextures(1, &ctrl_tex_);
            mgl::glDeleteFramebuffers(1, &hdr_fbo_);
            mgl::glDeleteTextures(1, &hdr_tex_);
            mgl::glDeleteRenderbuffers(1, &hdr_depth_);
            mgl::glDeleteFramebuffers(1, &mrt_fbo_);
            mgl::glDeleteTextures(static_cast<GLsizei>(mrt_tex_.size()), mrt_tex_.data());
            mgl::glDeleteVertexArrays(1, &scene_vao_);
            mgl::glDeleteBuffers(1, &scene_vbo_);
            mgl::glDeleteProgram(pattern_);
            mgl::glDeleteProgram(outputs_);
            mgl::glDeleteProgram(scene_);
            mgl::glDeleteProgram(glow_);
            mgl::glDeleteProgram(compose_);
        }

    private:
        // =========================================================================================
        // Object creation
        // =========================================================================================
        static mgl::TextureId make_texture(mgl::SizedInternalFormat format, int w, int h)
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, w, h);
            // NEAREST: the default MIN_FILTER needs a mipmap chain, and R32F is not texture-filterable (Table 8.10).
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            return t;
        }

        static void attach_color(mgl::ColorAttachment where, mgl::TextureId tex)
        {
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(where),
                                        mgl::TextureTarget::Texture2D, tex, 0);
        }

        void create_format_targets()
        {
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                fmt_tex_[i] = make_texture(kFormats[i].format, kPatternSize, kPatternSize);
                mgl::glGenFramebuffers(1, &fmt_fbo_[i]);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fmt_fbo_[i]);
                attach_color(mgl::ColorAttachment::Color0, fmt_tex_[i]);
                fmt_status_[i] = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            }
            // The RGBA8 control (negative control: an 8-bit target cannot hold the pattern).
            ctrl_tex_ = make_texture(mgl::SizedInternalFormat::Rgba8, kPatternSize, kPatternSize);
            mgl::glGenFramebuffers(1, &ctrl_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ctrl_fbo_);
            attach_color(mgl::ColorAttachment::Color0, ctrl_tex_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void create_hdr_target()
        {
            hdr_tex_ = make_texture(mgl::SizedInternalFormat::Rgba16F, kSceneW, kSceneH);
            mgl::glGenRenderbuffers(1, &hdr_depth_);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, hdr_depth_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent24, kSceneW, kSceneH);
            mgl::glGenFramebuffers(1, &hdr_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, hdr_fbo_);
            attach_color(mgl::ColorAttachment::Color0, hdr_tex_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                           mgl::RenderbufferTarget::Renderbuffer, hdr_depth_);
            const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (st != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("RGBA16F + DEPTH_COMPONENT24 framebuffer incomplete: {}", demo::framebuffer_status_name(st)));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void create_mrt_target()
        {
            if (max_draw_buffers_ < kBuffers || max_color_attachments_ < kBuffers)
                throw demo::Failure(std::format("MAX_DRAW_BUFFERS = {} / MAX_COLOR_ATTACHMENTS = {} are below the required 4", max_draw_buffers_,
                                                max_color_attachments_));
            for (std::size_t i = 0; i < kBuffers; ++i) mrt_tex_[i] = make_texture(kBufFormats[i], kMrtSize, kMrtSize);
            mgl::glGenFramebuffers(1, &mrt_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mrt_fbo_);
            constexpr std::array<mgl::ColorAttachment, kBuffers> where = {mgl::ColorAttachment::Color0, mgl::ColorAttachment::Color1,
                                                                          mgl::ColorAttachment::Color2, mgl::ColorAttachment::Color3};
            for (std::size_t i = 0; i < kBuffers; ++i) attach_color(where[i], mrt_tex_[i]);
            const std::array<mgl::DrawBuffer, kBuffers> draw = {
                mgl::to_draw_buffer(where[0]), mgl::to_draw_buffer(where[1]), mgl::to_draw_buffer(where[2]), mgl::to_draw_buffer(where[3])};
            mgl::glDrawBuffers(draw);
            const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (st != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("RGBA8 + RGBA16F + R11F_G11F_B10F + RGBA16F framebuffer incomplete: {}", demo::framebuffer_status_name(st)));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // ---- scene geometry -----------------------------------------------------------------------
        void add_vertex(std::vector<float>& v, D3 p, D3 n)
        {
            for (double c : {p.x, p.y, p.z, n.x, n.y, n.z}) v.push_back(static_cast<float>(c));
        }

        void add_quad(std::vector<float>& v, D3 a, D3 b, D3 c, D3 d, D3 n)
        {
            for (D3 p : {a, b, c, a, c, d}) add_vertex(v, p, n);
        }

        void build_scene_geometry()
        {
            std::vector<float> v;
            // floor: y = 0
            add_quad(v, {-kFloorX, 0, kFloorZMax}, {kFloorX, 0, kFloorZMax}, {kFloorX, 0, kFloorZMin}, {-kFloorX, 0, kFloorZMin}, {0, 1, 0});
            floor_count_ = static_cast<GLsizei>(v.size() / 6);
            box_first_ = static_cast<GLint>(v.size() / 6);
            for (const BoxDef& b : kBoxes)
            {
                const D3 c = b.center, h = b.half;
                const D3 p[8] = {{c.x - h.x, c.y - h.y, c.z - h.z}, {c.x + h.x, c.y - h.y, c.z - h.z}, {c.x + h.x, c.y + h.y, c.z - h.z},
                                 {c.x - h.x, c.y + h.y, c.z - h.z}, {c.x - h.x, c.y - h.y, c.z + h.z}, {c.x + h.x, c.y - h.y, c.z + h.z},
                                 {c.x + h.x, c.y + h.y, c.z + h.z}, {c.x - h.x, c.y + h.y, c.z + h.z}};
                add_quad(v, p[4], p[5], p[6], p[7], {0, 0, 1});    // +z
                add_quad(v, p[1], p[0], p[3], p[2], {0, 0, -1});   // -z
                add_quad(v, p[5], p[1], p[2], p[6], {1, 0, 0});    // +x
                add_quad(v, p[0], p[4], p[7], p[3], {-1, 0, 0});   // -x
                add_quad(v, p[7], p[6], p[2], p[3], {0, 1, 0});    // +y
                add_quad(v, p[0], p[1], p[5], p[4], {0, -1, 0});   // -y
            }
            box_vertices_ = 36;
            sphere_first_ = static_cast<GLint>(v.size() / 6);
            constexpr int kStacks = 12, kSlices = 20;
            for (const SphereDef& s : kSpheres)
            {
                auto point = [&](int i, int j) {
                    const double theta = std::numbers::pi * i / kStacks, phi = 2.0 * std::numbers::pi * j / kSlices;
                    return D3{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
                };
                for (int i = 0; i < kStacks; ++i)
                    for (int j = 0; j < kSlices; ++j)
                    {
                        const D3 q[4] = {point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)};
                        for (int k : {0, 1, 2, 0, 2, 3}) add_vertex(v, s.center + q[k] * s.radius, q[k]);
                    }
            }
            sphere_vertices_ = kStacks * kSlices * 6;

            mgl::glGenVertexArrays(1, &scene_vao_);
            mgl::glBindVertexArray(scene_vao_);
            mgl::glGenBuffers(1, &scene_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, scene_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(v), mgl::BufferUsage::StaticDraw);
            const mgl::AttribLocation position{0}, normal{1};
            mgl::glEnableVertexAttribArray(position);
            mgl::glVertexAttribPointer(position, 3, mgl::DataType::Float, GL_FALSE, 6 * sizeof(float), demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(normal);
            mgl::glVertexAttribPointer(normal, 3, mgl::DataType::Float, GL_FALSE, 6 * sizeof(float), demo::buffer_offset(3 * sizeof(float)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void choose_probes()
        {
            auto add = [&](int x, int y, std::string label) {
                if (x < 0 || y < 0 || x >= kSceneW || y >= kSceneH) return;
                if (std::abs(x - 127) <= 1 || std::abs(x - 128) <= 1) return;  // the split columns
                if (!stable_probe(camera_, x, y)) return;
                for (const Probe& p : probes_)
                    if (p.x == x && p.y == y) return;
                // The label names what the CPU ray tracer finds at the pixel, not what the probe was aimed at (a box may stand in front).
                const int sig = reference_pixel(camera_, x + 0.5, y + 0.5).signature;
                const int id = sig & 0xF;
                std::string what = id == 15 ? "background" : (id == 0 ? "floor" : (id <= 3 ? std::format("box {}", id - 1) : std::format("sphere {}", id - 4)));
                if (sig & 0x300) what += " + glow";
                probes_.push_back({x, y, std::format("{} at ({},{}) [aimed at: {}]", what, x, y, label)});
            };
            auto add_world = [&](D3 p, std::string label) {
                double px = 0, py = 0;
                if (project(camera_, p, px, py)) add(static_cast<int>(std::floor(px)), static_cast<int>(std::floor(py)), std::move(label));
            };
            for (std::size_t s = 0; s < kSpheres.size(); ++s)
            {
                const SphereDef& sp = kSpheres[s];
                add_world(sp.center, std::format("emissive sphere {} center", s));
                add_world(sp.center + camera_.r * (0.55 * sp.glow_radius) + camera_.u * (0.45 * sp.glow_radius), std::format("glow halo {} (upper right)", s));
                add_world(sp.center - camera_.r * (0.5 * sp.glow_radius) + camera_.u * (0.3 * sp.glow_radius), std::format("glow halo {} (left)", s));
                add_world(sp.center + camera_.r * (0.3 * sp.glow_radius) - camera_.u * (0.6 * sp.glow_radius), std::format("glow halo {} (below)", s));
            }
            for (std::size_t b = 0; b < kBoxes.size(); ++b)
            {
                const BoxDef& bx = kBoxes[b];
                add_world({bx.center.x, bx.center.y, bx.center.z + bx.half.z}, std::format("box {} front face", b));
                add_world({bx.center.x, bx.center.y + bx.half.y, bx.center.z}, std::format("box {} top face", b));
                add_world({bx.center.x + bx.half.x * 0.6, bx.center.y + bx.half.y * 0.3, bx.center.z + bx.half.z}, std::format("box {} front face (right)", b));
            }
            const double floor_pts[][2] = {{-3.5, 3.5}, {2.5, 4.2}, {-0.5, 3.0}, {0.7, 1.3}, {-2.4, -0.8}, {3.2, -0.6}, {0.5, -3.0}, {-0.8, 2.3}};
            for (const auto& f : floor_pts) add_world({f[0], 0.0, f[1]}, std::format("floor ({:.1f}, {:.1f})", f[0], f[1]));
            add(20, 150, "background");
            add(235, 148, "background");
            add(60, 130, "background");
            for (int y = 8; y < kSceneH; y += 16)
                for (int x = 8; x < kSceneW; x += 16) add(x, y, std::format("grid ({},{})", x, y));
        }

        // =========================================================================================
        // Rendering
        // =========================================================================================
        mgl::UniformLocation uniform(mgl::ProgramId program, const char* name) const
        {
            const mgl::UniformLocation loc = mgl::glGetUniformLocation(program, name);
            if (loc.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
            return loc;
        }

        void draw_quad() const { mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4); }

        /// Blend state of every draw buffer back to its initial values, with the non-indexed commands (which set all buffers).
        static void reset_blend_state()
        {
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::Zero);
            mgl::glBlendEquation(mgl::BlendEquation::FuncAdd);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glBlendColor(0.0f, 0.0f, 0.0f, 0.0f);
        }

        void draw_patterns()
        {
            mgl::glUseProgram(pattern_);
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                const ProbeValues p = probes_for(kFormats[i].cls);
                mgl::glUniform4fv(uniform(pattern_, "u_probe0"), 1, p.exact.data());
                mgl::glUniform4fv(uniform(pattern_, "u_probe1"), 1, p.rounding.data());
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fmt_fbo_[i]);
                mgl::glViewport(0, 0, kPatternSize, kPatternSize);
                draw_quad();
            }
        }

        void draw_pattern_into_control() const
        {
            const ProbeValues p = probes_for(FloatClass::Half);
            mgl::glUseProgram(pattern_);
            mgl::glUniform4fv(uniform(pattern_, "u_probe0"), 1, p.exact.data());
            mgl::glUniform4fv(uniform(pattern_, "u_probe1"), 1, p.rounding.data());
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ctrl_fbo_);
            mgl::glViewport(0, 0, kPatternSize, kPatternSize);
            draw_quad();
        }

        void draw_hdr_scene()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, hdr_fbo_);
            mgl::glViewport(0, 0, kSceneW, kSceneH);
            // The clear value (0.015, 0.02, 0.035) is not an 8-bit color: it only exists because the target is a float.
            mgl::glClearColor(static_cast<float>(kClear.x), static_cast<float>(kClear.y), static_cast<float>(kClear.z), 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);

            const dm::Mat4 proj = dm::perspective(dm::radians(42.0f), static_cast<float>(camera_.aspect), 0.1f, 60.0f);
            const dm::Mat4 view = dm::look_at({0.0f, 2.4f, 6.2f}, {0.0f, 0.8f, 0.0f}, {0.0f, 1.0f, 0.0f});
            const dm::Mat4 viewproj = proj * view;

            mgl::glUseProgram(scene_);
            mgl::glUniformMatrix4fv(uniform(scene_, "u_viewproj"), 1, GL_FALSE, viewproj.data());
            const D3 sun = normalize(kSunDirRaw);
            mgl::glUniform3f(uniform(scene_, "u_ambient"), static_cast<float>(kAmbient.x), static_cast<float>(kAmbient.y), static_cast<float>(kAmbient.z));
            mgl::glUniform3f(uniform(scene_, "u_sun_dir"), static_cast<float>(sun.x), static_cast<float>(sun.y), static_cast<float>(sun.z));
            mgl::glUniform3f(uniform(scene_, "u_sun_color"), static_cast<float>(kSunColor.x), static_cast<float>(kSunColor.y), static_cast<float>(kSunColor.z));
            std::array<float, 6> light_pos{}, light_color{};
            for (std::size_t i = 0; i < kSpheres.size(); ++i)
            {
                light_pos[3 * i + 0] = static_cast<float>(kSpheres[i].center.x);
                light_pos[3 * i + 1] = static_cast<float>(kSpheres[i].center.y);
                light_pos[3 * i + 2] = static_cast<float>(kSpheres[i].center.z);
                light_color[3 * i + 0] = static_cast<float>(kSpheres[i].light_color.x);
                light_color[3 * i + 1] = static_cast<float>(kSpheres[i].light_color.y);
                light_color[3 * i + 2] = static_cast<float>(kSpheres[i].light_color.z);
            }
            mgl::glUniform3fv(uniform(scene_, "u_light_pos"), 2, light_pos.data());
            mgl::glUniform3fv(uniform(scene_, "u_light_color"), 2, light_color.data());

            mgl::glBindVertexArray(scene_vao_);
            mgl::glUniform3f(uniform(scene_, "u_albedo"), 0.0f, 0.0f, 0.0f);
            mgl::glUniform3f(uniform(scene_, "u_emission"), 0.0f, 0.0f, 0.0f);
            mgl::glUniform1i(uniform(scene_, "u_checker"), 1);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, floor_count_);
            mgl::glUniform1i(uniform(scene_, "u_checker"), 0);
            for (std::size_t b = 0; b < kBoxes.size(); ++b)
            {
                mgl::glUniform3f(uniform(scene_, "u_albedo"), static_cast<float>(kBoxes[b].albedo.x), static_cast<float>(kBoxes[b].albedo.y),
                                 static_cast<float>(kBoxes[b].albedo.z));
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, box_first_ + static_cast<GLint>(b) * box_vertices_, box_vertices_);
            }
            mgl::glUniform3f(uniform(scene_, "u_albedo"), 0.0f, 0.0f, 0.0f);
            for (std::size_t s = 0; s < kSpheres.size(); ++s)
            {
                mgl::glUniform3f(uniform(scene_, "u_emission"), static_cast<float>(kSpheres[s].emission.x), static_cast<float>(kSpheres[s].emission.y),
                                 static_cast<float>(kSpheres[s].emission.z));
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, sphere_first_ + static_cast<GLint>(s) * sphere_vertices_, sphere_vertices_);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            // Additive halos: ONE, ONE blending into the float target. The sums exceed 1 and are kept.
            mgl::glEnable(mgl::Capability::Blend);
            mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::One);
            mgl::glDepthMask(GL_FALSE);
            mgl::glUseProgram(glow_);
            mgl::glUniformMatrix4fv(uniform(glow_, "u_viewproj"), 1, GL_FALSE, viewproj.data());
            mgl::glUniform3f(uniform(glow_, "u_right"), static_cast<float>(camera_.r.x), static_cast<float>(camera_.r.y), static_cast<float>(camera_.r.z));
            mgl::glUniform3f(uniform(glow_, "u_up"), static_cast<float>(camera_.u.x), static_cast<float>(camera_.u.y), static_cast<float>(camera_.u.z));
            for (const SphereDef& s : kSpheres)
            {
                mgl::glUniform3f(uniform(glow_, "u_center"), static_cast<float>(s.center.x), static_cast<float>(s.center.y), static_cast<float>(s.center.z));
                mgl::glUniform1f(uniform(glow_, "u_radius"), static_cast<float>(s.glow_radius));
                mgl::glUniform3f(uniform(glow_, "u_color"), static_cast<float>(s.glow_color.x), static_cast<float>(s.glow_color.y), static_cast<float>(s.glow_color.z));
                draw_quad();
            }
            mgl::glDepthMask(GL_TRUE);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::DepthTest);
            reset_blend_state();
        }

        /// Runs one scenario on the MRT framebuffer: the destination colors are drawn with blending off, then the
        /// scenario's state commands are issued and the source colors are drawn. Returns the indexed state read back
        /// right after the commands (before the draws), for every draw buffer.
        std::vector<RawState> run_scenario(const Scenario& sc)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mrt_fbo_);
            mgl::glUseProgram(outputs_);
            const mgl::UniformLocation u_src = uniform(outputs_, "u_src");

            reset_blend_state();
            for (std::size_t q = 0; q < 4; ++q) draw_quadrant(u_src, q, true);

            mgl::glBlendColor(sc.constant[0], sc.constant[1], sc.constant[2], sc.constant[3]);
            for (std::size_t i = 0; i < kBuffers; ++i)
                if (sc.junk[i]) issue_indexed(static_cast<GLuint>(i), *sc.junk[i]);
            if (sc.global) issue_global(*sc.global);
            for (std::size_t i = 0; i < kBuffers; ++i)
                if (sc.per_buffer[i]) issue_indexed(static_cast<GLuint>(i), *sc.per_buffer[i]);

            std::vector<RawState> states;
            for (int i = 0; i < max_draw_buffers_; ++i) states.push_back(query_raw(static_cast<GLuint>(i)));

            for (std::size_t q = 0; q < 4; ++q) draw_quadrant(u_src, q, false);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return states;
        }

        void draw_quadrant(mgl::UniformLocation u_src, std::size_t q, bool destination) const
        {
            std::array<float, 16> values{};
            for (std::size_t b = 0; b < kBuffers; ++b)
            {
                const Rgbaf& c = destination ? kQuads[q].dst[b] : kQuads[q].src[b];
                for (std::size_t k = 0; k < 4; ++k) values[b * 4 + k] = c[k];
            }
            mgl::glUniform4fv(u_src, 4, values.data());
            mgl::glViewport(kQuadOrigin[q].first, kQuadOrigin[q].second, kQuadSize, kQuadSize);
            draw_quad();
        }

        void compose_window()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, 256, 256);
            mgl::glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(compose_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glUniform1i(uniform(compose_, "u_tex"), 0);  // texture unit index
            mgl::glUniform1f(uniform(compose_, "u_exposure"), static_cast<float>(kExposure));

            auto region = [&](mgl::TextureId tex, int mode, int x, int y, int w, int h) {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
                mgl::glUniform1i(uniform(compose_, "u_mode"), mode);
                mgl::glUniform2i(uniform(compose_, "u_origin"), x, y);
                mgl::glUniform2i(uniform(compose_, "u_size"), w, h);
                mgl::glViewport(x, y, w, h);
                draw_quad();
            };
            region(hdr_tex_, 0, 0, kSceneY, kSceneW, kSceneH);
            for (std::size_t i = 0; i < kFormats.size(); ++i)
                region(fmt_tex_[i], 1, kSwatchX0 + static_cast<int>(i) * kSwatchW, 64, kSwatchW, 32);
            for (std::size_t b = 0; b < kBuffers; ++b)
                region(mrt_tex_[b], b == 0 ? 2 : 1, static_cast<int>(b) * kMrtSize, 0, kMrtSize, kMrtSize);
        }

        // =========================================================================================
        // State queries
        // =========================================================================================
        static RawState query_raw(GLuint i)
        {
            RawState s;
            s.enable = mgl::glIsEnabledi(mgl::Capability::Blend, i);
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendEquationRgb, i, &s.eq_rgb);
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendEquationAlpha, i, &s.eq_alpha);
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendSrcRgb, i, &s.src_rgb);
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendDstRgb, i, &s.dst_rgb);
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendSrcAlpha, i, &s.src_alpha);
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendDstAlpha, i, &s.dst_alpha);
            std::array<GLboolean, 4> m{};
            mgl::glGetBooleani_v(mgl::GetParameter::ColorWritemask, i, m.data());
            for (std::size_t k = 0; k < 4; ++k) s.mask[k] = m[k] != GL_FALSE;
            return s;
        }

        /// Compares `got` (one RawState per draw buffer index) with the expectation `want(i)` for every index.
        template <typename WantFn>
        void expect_states(demo::Check& check, const std::vector<RawState>& got, WantFn&& want, std::string_view what) const
        {
            std::string first;
            int bad = 0;
            for (std::size_t i = 0; i < got.size(); ++i)
            {
                const RawState w = want(i);
                if (!(got[i] == w))
                {
                    if (bad++ == 0) first = std::format("first at index {}: got [{}] expected [{}]", i, describe(got[i]), describe(w));
                }
            }
            check.expect(bad == 0, what, bad == 0 ? std::format("{} indices", got.size()) : std::format("{} of {} indices differ, {}", bad, got.size(), first));
        }

        std::vector<RawState> query_all() const
        {
            std::vector<RawState> v;
            for (int i = 0; i < max_draw_buffers_; ++i) v.push_back(query_raw(static_cast<GLuint>(i)));
            return v;
        }

        // =========================================================================================
        // Validation: float formats
        // =========================================================================================
        std::vector<float> read_float(int x, int y, int w, int h) const
        {
            std::vector<float> px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u);
            // ES 3.2 16.1.2: "For floating-point rendering surfaces, the combination format RGBA and type FLOAT is accepted."
            mgl::glReadPixels(x, y, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::Float, px.data());
            return px;
        }

        void check_float_formats(demo::Check& check)
        {
            std::size_t total_outside = 0;
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                const FormatCase& fc = kFormats[i];
                check.expect(fmt_status_[i] == mgl::FramebufferStatus::Complete,
                             std::format("{} texture attachment: framebuffer complete (Table 8.10 CR + Req. rend., 9.4.3)", fc.name),
                             std::format("status {}", demo::framebuffer_status_name(fmt_status_[i])));

                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fmt_fbo_[i]);
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                GLint type = 0, enc = 0;
                const auto att = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, att, mgl::FramebufferAttachmentParameter::ComponentType, &type);
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, att, mgl::FramebufferAttachmentParameter::ColorEncoding, &enc);
                std::array<GLint, 4> size{};
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, att, mgl::FramebufferAttachmentParameter::RedSize, &size[0]);
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, att, mgl::FramebufferAttachmentParameter::GreenSize, &size[1]);
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, att, mgl::FramebufferAttachmentParameter::BlueSize, &size[2]);
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, att, mgl::FramebufferAttachmentParameter::AlphaSize, &size[3]);
                bool sizes_ok = true;
                for (std::size_t c = 0; c < 4; ++c)
                    sizes_ok = sizes_ok && (fc.bits[c] == 0 ? size[c] == 0 : size[c] >= fc.bits[c]);
                check.expect(type == static_cast<GLint>(GL_FLOAT) && enc == static_cast<GLint>(GL_LINEAR) && sizes_ok,
                             std::format("{}: attachment COMPONENT_TYPE = FLOAT, COLOR_ENCODING = LINEAR, R/G/B/A sizes >= {}/{}/{}/{} (absent = 0)", fc.name,
                                         fc.bits[0], fc.bits[1], fc.bits[2], fc.bits[3]),
                             std::format("type {:#x} encoding {:#x} sizes {}/{}/{}/{}", type, enc, size[0], size[1], size[2], size[3]));

                // The pattern, read back as floats and compared with the CPU model.
                const std::vector<float> px = read_float(0, 0, kPatternSize, kPatternSize);
                int bad = 0;
                std::string first;
                std::size_t outside = 0;
                for (int y = 0; y < kPatternSize; ++y)
                    for (int x = 0; x < kPatternSize; ++x)
                        for (int c = 0; c < 4; ++c)
                        {
                            const float got = px[static_cast<std::size_t>((y * kPatternSize + x) * 4 + c)];
                            if (c < fc.stored && (got < 0.0f || got > 1.0f)) ++outside;
                            if (!pattern_component_ok(fc, c, x, y, got) && bad++ == 0)
                                first = std::format("first at ({},{}) component {}: got {} expected {}", x, y, c, got,
                                                    stored_component(fc, c, raw_pattern(fc, c, x, y)));
                        }
                total_outside += outside;
                check.expect(bad == 0,
                             std::format("{}: all {} texels read back with glReadPixels(RGBA, FLOAT) equal the CPU model ({} stored components outside [0,1])",
                                         fc.name, kPatternSize * kPatternSize, outside),
                             bad == 0 ? "" : std::format("{} components differ, {}", bad, first));

                // The extreme probes: print what was read (the range extremes are part of the model above).
                const std::size_t p0 = static_cast<std::size_t>((15 * kPatternSize + 15) * 4);
                std::cout << std::format("     {:>14}: texel (15,15) = ({}, {}, {}, {})\n", fc.name, px[p0], px[p0 + 1], px[p0 + 2], px[p0 + 3]);

                // The implementation-chosen read format (16.1.2): for a floating-point buffer ReadPixels rejects every type but FLOAT,
                // HALF_FLOAT and UNSIGNED_INT_10F_11F_11F_REV (16.1.3), so the chosen type must be one of them.
                const int read_type = demo::get_integer(mgl::GetParameter::ImplementationColorReadType);
                const int read_format = demo::get_integer(mgl::GetParameter::ImplementationColorReadFormat);
                const bool type_ok = read_type == static_cast<int>(GL_FLOAT) || read_type == static_cast<int>(GL_HALF_FLOAT) ||
                                     read_type == static_cast<int>(GL_UNSIGNED_INT_10F_11F_11F_REV);
                check.expect(type_ok, std::format("{}: IMPLEMENTATION_COLOR_READ_TYPE is a type that floating-point buffers accept", fc.name),
                             std::format("format {:#x} type {:#x}", read_format, read_type));
            }
            check.expect(total_outside > 600, "the float targets hold values outside [0, 1]", std::format("{} stored components", total_outside));

            // RGBA/UNSIGNED_BYTE is not accepted for a floating-point color buffer (16.1.2, 16.1.3).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fmt_fbo_[2]);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            std::array<std::uint8_t, 4> byte_px{};
            demo::drain_gl_errors();
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, byte_px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glReadPixels(RGBA, UNSIGNED_BYTE) from an RGBA16F buffer: INVALID_OPERATION (16.1.3)");

            // Negative control: the same pattern drawn into an RGBA8 target cannot be the float pattern.
            draw_pattern_into_control();
            const std::vector<std::uint8_t> bytes = demo::read_rgba8(0, 0, kPatternSize, kPatternSize);
            int float_model_misses = 0, clamped_model_misses = 0;
            const FormatCase& rgba16f = kFormats[2];
            for (int y = 0; y < kPatternSize; ++y)
                for (int x = 0; x < kPatternSize; ++x)
                    for (int c = 0; c < 4; ++c)
                    {
                        const float got = static_cast<float>(bytes[static_cast<std::size_t>((y * kPatternSize + x) * 4 + c)]) / 255.0f;
                        if (!pattern_component_ok(rgba16f, c, x, y, got)) ++float_model_misses;
                        const float clamped = clamp01(stored_component(rgba16f, c, raw_pattern(rgba16f, c, x, y)));
                        if (std::fabs(got - clamped) > 1.0f / 255.0f + 1e-6f) ++clamped_model_misses;
                    }
            check.expect(float_model_misses > 300, "negative control: the pattern in an RGBA8 target does NOT match the float model (the comparison can fail)",
                         std::format("{} of 1024 components differ", float_model_misses));
            check.expect(clamped_model_misses == 0, "RGBA8 control: every component equals the pattern clamped to [0,1] (+-1/255, 15.1.6)",
                         std::format("{} components differ", clamped_model_misses));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        /// Float blending (15.1.5): no clamping, results beyond [0,1] are stored; 32-bit float targets are an error
        /// without GL_EXT_float_blend.
        void check_float_blending(demo::Check& check)
        {
            std::cout << std::format("     32-bit float blending: GL_EXT_float_blend is {} -> blending into R32F/RG32F/RGBA32F is expected to {}\n",
                                     float_blend_ ? "present" : "absent", float_blend_ ? "work" : "raise INVALID_OPERATION");
            mgl::glUseProgram(outputs_);
            const mgl::UniformLocation u_src = uniform(outputs_, "u_src");
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                const FormatCase& fc = kFormats[i];
                const bool expect_error = is_32bit(fc) && !float_blend_;
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fmt_fbo_[i]);
                mgl::glViewport(0, 0, kPatternSize, kPatternSize);
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));

                // (a) "over" with a source alpha of 2.5: SRC_ALPHA = 2.5 and ONE_MINUS_SRC_ALPHA = -1.5 are NOT clamped for a float buffer.
                //     src = (3, -2, 0.5, 2.5), dst = (1, 2, 0.5, 0.5):  R = 7.5 - 1.5 = 6,  G = -5 - 3 = -8,  B = 1.25 - 0.75 = 0.5,  A = 6.25 - 0.75 = 5.5.
                const Rgbaf dst = {1.0f, 2.0f, 0.5f, 0.5f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, dst.data());
                const std::array<float, 16> src_over = {3.0f, -2.0f, 0.5f, 2.5f, 3.0f, -2.0f, 0.5f, 2.5f, 3.0f, -2.0f, 0.5f, 2.5f, 3.0f, -2.0f, 0.5f, 2.5f};
                mgl::glUniform4fv(u_src, 4, src_over.data());
                mgl::glEnable(mgl::Capability::Blend);
                mgl::glBlendFunc(mgl::BlendFactor::SrcAlpha, mgl::BlendFactor::OneMinusSrcAlpha);
                demo::drain_gl_errors();
                draw_quad();
                const mgl::ErrorCode err_a = demo::drain_gl_errors();
                const Rgbaf want_over = {6.0f, -8.0f, 0.5f, 5.5f};
                const Rgbaf want_unchanged = dst;
                report_blend(check, fc, "SRC_ALPHA / ONE_MINUS_SRC_ALPHA with As = 2.5 (unclamped factors)", err_a, expect_error,
                             expect_error ? want_unchanged : want_over);

                // (b) HDR accumulation: four additive draws of (3.75, 0.5, 0.25, 1) onto 0 give (15, 2, 1, 4).
                const Rgbaf zero = {0.0f, 0.0f, 0.0f, 0.0f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, zero.data());
                const std::array<float, 16> src_add = {3.75f, 0.5f, 0.25f, 1.0f, 3.75f, 0.5f, 0.25f, 1.0f, 3.75f, 0.5f, 0.25f, 1.0f, 3.75f, 0.5f, 0.25f, 1.0f};
                mgl::glUniform4fv(u_src, 4, src_add.data());
                mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::One);
                mgl::ErrorCode err_b = mgl::ErrorCode::NoError;
                for (int n = 0; n < 4; ++n)
                {
                    demo::drain_gl_errors();
                    draw_quad();
                    const mgl::ErrorCode e = demo::drain_gl_errors();
                    if (e != mgl::ErrorCode::NoError) err_b = e;
                }
                report_blend(check, fc, "4 x additive ONE, ONE of (3.75, 0.5, 0.25, 1) (HDR accumulation)", err_b, expect_error,
                             expect_error ? zero : Rgbaf{15.0f, 2.0f, 1.0f, 4.0f});
                mgl::glDisable(mgl::Capability::Blend);
            }
            reset_blend_state();
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void report_blend(demo::Check& check, const FormatCase& fc, std::string_view what, mgl::ErrorCode err, bool expect_error, const Rgbaf& want_raw)
        {
            if (expect_error)
                check.expect(err == mgl::ErrorCode::InvalidOperation,
                             std::format("{}: {} -> INVALID_OPERATION (blending enabled, 32-bit float draw buffer, no GL_EXT_float_blend; 15.1.5)", fc.name, what),
                             std::format("glGetError() = {}", demo::error_name(err)));
            else
                check.expect(err == mgl::ErrorCode::NoError, std::format("{}: {} raises no error", fc.name, what),
                             std::format("glGetError() = {}", demo::error_name(err)));
            const std::vector<float> px = read_float(0, 0, kPatternSize, kPatternSize);
            int bad = 0;
            std::string first;
            for (std::size_t p = 0; p < px.size() / 4; ++p)
                for (int c = 0; c < 4; ++c)
                {
                    const float want = stored_component(fc, c, want_raw[static_cast<std::size_t>(c)]);
                    const float got = px[p * 4 + static_cast<std::size_t>(c)];
                    if (got != want && bad++ == 0) first = std::format("pixel {} component {}: got {} expected {}", p, c, got, want);
                }
            check.expect(bad == 0,
                         std::format("{}: {}: all texels = ({}, {}, {}, {}){}", fc.name, what, stored_component(fc, 0, want_raw[0]),
                                     stored_component(fc, 1, want_raw[1]), stored_component(fc, 2, want_raw[2]), stored_component(fc, 3, want_raw[3]),
                                     expect_error ? " (the rejected draw changed nothing)" : ""),
                         bad == 0 ? "" : std::format("{} components differ, {}", bad, first));
        }

        /// The seven formats as renderbuffers, the formats that are NOT color-renderable, glClearBufferfv with HDR values.
        void check_renderbuffer_formats(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            demo::drain_gl_errors();
            for (const FormatCase& fc : kFormats)
            {
                mgl::RenderbufferId rb{};
                mgl::glGenRenderbuffers(1, &rb);
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, fc.renderbuffer_format, 8, 8);
                const mgl::ErrorCode storage_error = demo::drain_gl_errors();
                mgl::FramebufferId fbo{};
                mgl::glGenFramebuffers(1, &fbo);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
                mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                               mgl::RenderbufferTarget::Renderbuffer, rb);
                const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
                std::array<GLint, 4> size{};
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::RedSize, &size[0]);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::GreenSize, &size[1]);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::BlueSize, &size[2]);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::AlphaSize, &size[3]);
                bool sizes_ok = true;
                for (std::size_t c = 0; c < 4; ++c) sizes_ok = sizes_ok && (fc.bits[c] == 0 ? size[c] == 0 : size[c] >= fc.bits[c]);

                // glClearBufferfv takes the HDR value as it is; read it back.
                const Rgbaf clear = {3.75f, -2.5f, 0.5f, 2.0f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, clear.data());
                const std::vector<float> px = read_float(0, 0, 8, 8);
                int bad = 0;
                std::string first;
                for (std::size_t p = 0; p < px.size() / 4; ++p)
                    for (int c = 0; c < 4; ++c)
                    {
                        const float want = stored_component(fc, c, clear[static_cast<std::size_t>(c)]);
                        if (px[p * 4 + static_cast<std::size_t>(c)] != want && bad++ == 0)
                            first = std::format("component {}: got {} expected {}", c, px[p * 4 + static_cast<std::size_t>(c)], want);
                    }
                check.expect(storage_error == mgl::ErrorCode::NoError && st == mgl::FramebufferStatus::Complete && sizes_ok && bad == 0,
                             std::format("{} renderbuffer: storage accepted, complete, sizes >= table, glClearBufferfv(3.75, -2.5, 0.5, 2) reads back as ({}, {}, {}, {})",
                                         fc.name, stored_component(fc, 0, clear[0]), stored_component(fc, 1, clear[1]), stored_component(fc, 2, clear[2]),
                                         stored_component(fc, 3, clear[3])),
                             std::format("storage error {}, status {}, sizes {}/{}/{}/{}{}{}", demo::error_name(storage_error), demo::framebuffer_status_name(st),
                                         size[0], size[1], size[2], size[3], bad == 0 ? "" : ", ", first));
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                mgl::glDeleteFramebuffers(1, &fbo);
                mgl::glDeleteRenderbuffers(1, &rb);
            }

            // Formats that are texture formats but not color-renderable in core ES 3.2 (Table 8.10: CR column empty).
            struct NotRenderable
            {
                std::string_view name;
                mgl::SizedInternalFormat texture;
                mgl::InternalFormat renderbuffer;
                bool amended_by_extension;       // GL_EXT_color_buffer_half_float makes RGB16F renderable
                bool mesa_reports_unsupported;   // known deviation of every Mesa driver of the matrix
            };
            const std::array<NotRenderable, 3> not_renderable = {{
                {"RGB32F", mgl::SizedInternalFormat::Rgb32F, mgl::InternalFormat::Rgb32F, false, true},
                {"RGB9_E5", mgl::SizedInternalFormat::Rgb9E5, mgl::InternalFormat::Rgb9E5, false, false},
                {"RGB16F", mgl::SizedInternalFormat::Rgb16F, mgl::InternalFormat::Rgb16F, true, false},
            }};
            for (const NotRenderable& nr : not_renderable)
            {
                const bool strict = !(nr.amended_by_extension && half_float_ext_);
                mgl::TextureId tex = make_texture(nr.texture, 4, 4);
                mgl::FramebufferId fbo{};
                mgl::glGenFramebuffers(1, &fbo);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
                attach_color(mgl::ColorAttachment::Color0, tex);
                const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
                mgl::RenderbufferId rb{};
                mgl::glGenRenderbuffers(1, &rb);
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
                demo::drain_gl_errors();
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, nr.renderbuffer, 4, 4);
                const mgl::ErrorCode rb_error = demo::drain_gl_errors();
                if (strict)
                {
                    // 9.4.1: "If attachment is COLOR_ATTACHMENTi, then image must have a color-renderable internal format", else the
                    // attachment point is not attachment complete: FRAMEBUFFER_INCOMPLETE_ATTACHMENT (9.4.2). An implementation may
                    // also reject the format combination under FRAMEBUFFER_UNSUPPORTED, and "if more than one clause is violated, it
                    // is implementation-dependent which value will be returned" (9.4.2). Both statuses are therefore correct; the
                    // framebuffer must not be complete. (Mesa returns UNSUPPORTED for RGB32F.)
                    const std::string what = std::format("{} texture as COLOR_ATTACHMENT0: incomplete (INCOMPLETE_ATTACHMENT or UNSUPPORTED, 9.4.1/9.4.2)", nr.name);
                    const std::string detail = std::format("status {}", demo::framebuffer_status_name(st));
                    const bool ok = st == mgl::FramebufferStatus::IncompleteAttachment || st == mgl::FramebufferStatus::Unsupported;
                    (void)nr.mesa_reports_unsupported;
                    check.expect(ok, what, detail);
                    check.expect(rb_error == mgl::ErrorCode::InvalidEnum,
                                 std::format("glRenderbufferStorage({}): INVALID_ENUM (not color-, depth- or stencil-renderable, 9.2.4)", nr.name),
                                 std::format("glGetError() = {}", demo::error_name(rb_error)));
                }
                else
                {
                    // GL_EXT_color_buffer_half_float adds RGB16F to the color-renderable formats: the core rule is amended, nothing to assert.
                    std::cout << std::format("     {}: GL_EXT_color_buffer_half_float is present, so the core rule does not apply (texture status {}, renderbuffer error {})\n",
                                             nr.name, demo::framebuffer_status_name(st), demo::error_name(rb_error));
                }
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                mgl::glDeleteFramebuffers(1, &fbo);
                mgl::glDeleteRenderbuffers(1, &rb);
                mgl::glDeleteTextures(1, &tex);
            }
        }

        // =========================================================================================
        // Validation: HDR scene and composition
        // =========================================================================================
        /// Sanity tolerance for the HDR probes. The spec gives no accuracy for varying interpolation or shader arithmetic (GLSL ES 3.20
        /// 4.7.1 only bounds individual operations), only the storage in RGBA16F costs a defined 2^-11 relative rounding error. The bound is
        /// 0.5 % + 2^-12: far below any wrong light, missing glow or clamped value (those change a probe by 10 % to 1000 %), far above
        /// the measured deviation, which is printed.
        static double sanity_tolerance(double expected) { return std::fabs(expected) * 0.005 + 0x1p-12; }

        void check_hdr_scene(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, hdr_fbo_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            hdr_ = read_float(0, 0, kSceneW, kSceneH);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            std::size_t above_one = 0;
            float max_value = 0.0f;
            for (std::size_t p = 0; p < hdr_.size() / 4; ++p)
            {
                const float m = std::max({hdr_[p * 4], hdr_[p * 4 + 1], hdr_[p * 4 + 2]});
                if (m > 1.0f) ++above_one;
                max_value = std::max(max_value, m);
            }
            check.expect(probes_.size() >= 24, "enough stable probe pixels were found for the HDR scene", std::format("{} probes", probes_.size()));
            check.expect(above_one > 300 && max_value >= 12.0f, "the RGBA16F scene target stores values above 1 (nothing was clamped)",
                         std::format("{} pixels with a color component > 1, maximum {}", above_one, max_value));

            int bad = 0, over_one = 0, under_one = 0;
            double worst = 0.0;
            std::string first, worst_label;
            for (const Probe& pr : probes_)
            {
                const PixelRef ref = reference_pixel(camera_, pr.x + 0.5, pr.y + 0.5);
                const double want[3] = {ref.color.x, ref.color.y, ref.color.z};
                (std::max({want[0], want[1], want[2]}) > 1.0 ? over_one : under_one)++;
                for (int c = 0; c < 3; ++c)
                {
                    const double got = hdr_[static_cast<std::size_t>((pr.y * kSceneW + pr.x) * 4 + c)];
                    const double err = std::fabs(got - want[c]);
                    if (err / (std::fabs(want[c]) + 0x1p-6) > worst)
                    {
                        worst = err / (std::fabs(want[c]) + 0x1p-6);
                        worst_label = pr.label;
                    }
                    if (err > sanity_tolerance(want[c]) && bad++ == 0)
                        first = std::format("{} at ({},{}) channel {}: got {} expected {}", pr.label, pr.x, pr.y, c, got, want[c]);
                }
            }
            check.expect(bad == 0, std::format("{} probe pixels of the HDR scene equal the CPU ray tracer (floor, boxes, lights, emission, glow) within 0.5 % + 2^-12", probes_.size()),
                         bad == 0 ? std::format("{} probes with a value > 1, {} <= 1; largest relative deviation {:.2e} ({})", over_one, under_one, worst, worst_label)
                                  : std::format("{} channels differ, {}", bad, first));
            check.expect(over_one >= 4 && under_one >= 4, "the probes cover HDR (> 1) and LDR (<= 1) pixels", std::format("{} / {}", over_one, under_one));

            // Window pixels: left half = min(hdr, 1), right half = Reinhard.
            int bad_px = 0, saturated_left = 0, bright_right = 0;
            std::string first_px;
            for (const Probe& pr : probes_)
            {
                const int wx = pr.x, wy = pr.y + kSceneY;
                for (int c = 0; c < 3; ++c)
                {
                    const double h = hdr_[static_cast<std::size_t>((pr.y * kSceneW + pr.x) * 4 + c)];
                    const double v = pr.x < kSceneW / 2 ? std::min(h, 1.0) : (h * kExposure) / (1.0 + h * kExposure);
                    const int want = static_cast<int>(std::lround(255.0 * v));
                    const int got = window_[static_cast<std::size_t>((wy * 256 + wx) * 4 + c)];
                    if (std::abs(got - want) > 1 && bad_px++ == 0)
                        first_px = std::format("{} at window ({},{}) channel {}: got {} expected {} (hdr {})", pr.label, wx, wy, c, got, want, h);
                    if (pr.x < kSceneW / 2 && h > 1.0 && want == 255) ++saturated_left;
                    if (pr.x >= kSceneW / 2 && h > 1.0) ++bright_right;
                }
            }
            check.expect(bad_px == 0, "window pixels of the scene: left half min(hdr, 1), right half Reinhard m / (1 + m) with m = 1.2 * hdr, each within 1/255",
                         bad_px == 0 ? std::format("{} channels clipped to 255 on the left, {} HDR channels tone-mapped below 255 on the right", saturated_left, bright_right)
                                     : std::format("{} channels differ, {}", bad_px, first_px));
            check.expect(saturated_left >= 3 && bright_right >= 3, "the clipped half really saturates where the tone-mapped half keeps detail",
                         std::format("{} / {}", saturated_left, bright_right));
        }

        static double signed_map(double v) { return v / (1.0 + std::fabs(v)) * 0.5 + 0.5; }

        void check_composite_panels(demo::Check& check)
        {
            // Float-format swatches: window pixel -> texel with the shader's integer arithmetic.
            int bad = 0, checked = 0;
            std::string first;
            for (std::size_t i = 0; i < kFormats.size(); ++i)
                for (const auto& pxy : {std::pair{4, 4}, std::pair{12, 9}, std::pair{20, 17}, std::pair{27, 28}})
                {
                    const int lx = pxy.first, ly = pxy.second;
                    const int tx = lx * kPatternSize / kSwatchW, ty = ly * kPatternSize / 32;
                    for (int c = 0; c < 3; ++c)
                    {
                        const double want = signed_map(stored_component(kFormats[i], c, raw_pattern(kFormats[i], c, tx, ty)));
                        const int got = window_[static_cast<std::size_t>(((64 + ly) * 256 + kSwatchX0 + static_cast<int>(i) * kSwatchW + lx) * 4 + c)];
                        ++checked;
                        // The shader divides in fp32 and converts to 8 bits with rounding: +-1 (the 16-bit pattern value itself is exact).
                        if (std::abs(got - static_cast<int>(std::lround(255.0 * want))) > 1 && bad++ == 0)
                            first = std::format("{} swatch pixel ({},{}) texel ({},{}) channel {}: got {} expected {}", kFormats[i].name, lx, ly, tx, ty, c, got,
                                                static_cast<int>(std::lround(255.0 * want)));
                    }
                }
            check.expect(bad == 0, std::format("the seven format swatches of the window show their texels through the signed tone curve ({} channels sampled)", checked),
                         bad == 0 ? "" : std::format("{} differ, {}", bad, first));

            // MRT panels of the showcase scenario: the centre of every quadrant.
            const auto states = effective_states(scenarios_[0]);
            int bad_mrt = 0, checked_mrt = 0;
            std::string first_mrt;
            for (std::size_t b = 0; b < kBuffers; ++b)
                for (std::size_t q = 0; q < 4; ++q)
                {
                    bool exact = true;
                    const std::array<double, 4> want = model_draw(kBufKinds[b], states[b], scenarios_[0].constant, kQuads[q].src[b], kQuads[q].dst[b], exact);
                    const int lx = kQuadOrigin[q].first + kQuadSize / 2, ly = kQuadOrigin[q].second + kQuadSize / 2;
                    for (int c = 0; c < 3; ++c)
                    {
                        const double v = kBufKinds[b] == BufKind::Unorm8 ? want[static_cast<std::size_t>(c)] / 255.0 : signed_map(want[static_cast<std::size_t>(c)]);
                        const int expect8 = static_cast<int>(std::lround(255.0 * v));
                        const int got = window_[static_cast<std::size_t>((ly * 256 + static_cast<int>(b) * kMrtSize + lx) * 4 + c)];
                        ++checked_mrt;
                        // RGBA8 panel: the 8-bit texel is shown unchanged (exact up to 8-bit round trip +-1); float panels: curve in fp32, +-1.
                        if (std::abs(got - expect8) > 1 && bad_mrt++ == 0)
                            first_mrt = std::format("buffer {} quadrant {} channel {}: got {} expected {}", b, q, c, got, expect8);
                    }
                }
            check.expect(bad_mrt == 0, std::format("the four MRT panels of the window show the showcase scenario ({} channels sampled)", checked_mrt),
                         bad_mrt == 0 ? "" : std::format("{} differ, {}", bad_mrt, first_mrt));
        }

        // =========================================================================================
        // Validation: indexed blend state and MRT results
        // =========================================================================================
        void check_scenario_design() const
        {
            // Every expected float result must be exactly representable, so that exact comparison is legitimate.
            std::string problems;
            for (const Scenario& sc : scenarios_)
            {
                const auto states = effective_states(sc);
                for (std::size_t b = 0; b < kBuffers; ++b)
                    for (std::size_t q = 0; q < 4; ++q)
                    {
                        bool exact = true;
                        (void)model_draw(kBufKinds[b], states[b], sc.constant, kQuads[q].src[b], kQuads[q].dst[b], exact);
                        if (!exact)
                            problems += std::format("[{}: buffer {} quadrant {}: {}] ", sc.name, b, q, describe(states[b]));
                    }
            }
            if (!problems.empty()) throw demo::Failure("scenario table error: results not exactly representable: " + problems);
        }

        /// Reads attachment b of the MRT framebuffer in native units (bytes for RGBA8, floats otherwise).
        std::vector<double> read_attachment(std::size_t b) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mrt_fbo_);
            constexpr std::array<mgl::ColorAttachment, kBuffers> where = {mgl::ColorAttachment::Color0, mgl::ColorAttachment::Color1,
                                                                          mgl::ColorAttachment::Color2, mgl::ColorAttachment::Color3};
            mgl::glReadBuffer(mgl::to_read_buffer(where[b]));
            std::vector<double> out;
            if (kBufKinds[b] == BufKind::Unorm8)
            {
                for (std::uint8_t v : demo::read_rgba8(0, 0, kMrtSize, kMrtSize)) out.push_back(static_cast<double>(v));
            }
            else
            {
                for (float v : read_float(0, 0, kMrtSize, kMrtSize)) out.push_back(static_cast<double>(v));
            }
            return out;
        }

        /// Compares one attachment with the model of `state`; returns the number of differing pixels (and the largest deviation).
        int compare_attachment(const std::vector<double>& px, std::size_t b, const BufState& state, const Scenario& sc, double& max_dev, std::string& first) const
        {
            // 8-bit buffer: tolerance 2 steps. Blending in fixed point rounds each factor product (llvmpipe does), which costs up to one
            // step per product, a sum of two products two steps (the same bound as es20-blending; the spec only demands a precision no
            // lower than the destination components, 15.1.5.1). Float buffers: every value is exact, so the tolerance is 0.
            const double tol = kBufKinds[b] == BufKind::Unorm8 ? 2.0 : 0.0;
            int bad = 0;
            max_dev = 0.0;
            for (int y = 0; y < kMrtSize; ++y)
                for (int x = 0; x < kMrtSize; ++x)
                {
                    const std::size_t q = static_cast<std::size_t>((x >= kQuadSize ? 1 : 0) + (y >= kQuadSize ? 2 : 0));
                    bool exact = true;
                    const std::array<double, 4> want = model_draw(kBufKinds[b], state, sc.constant, kQuads[q].src[b], kQuads[q].dst[b], exact);
                    bool pixel_bad = false;
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        const double got = px[static_cast<std::size_t>((y * kMrtSize + x) * 4) + c];
                        const double dev = std::fabs(got - want[c]);
                        max_dev = std::max(max_dev, dev);
                        if (dev > tol)
                        {
                            if (!pixel_bad && bad == 0)
                            {
                                const std::size_t base = static_cast<std::size_t>((y * kMrtSize + x) * 4);
                                first = std::format("first at ({},{}) quadrant {} component {}: got ({}, {}, {}, {}) expected ({}, {}, {}, {})", x, y, q, c,
                                                    px[base], px[base + 1], px[base + 2], px[base + 3], want[0], want[1], want[2], want[3]);
                            }
                            pixel_bad = true;
                        }
                    }
                    if (pixel_bad) ++bad;
                }
            return bad;
        }

        void check_scenarios(demo::Check& check)
        {
            for (std::size_t s = 0; s < scenarios_.size(); ++s)
            {
                const Scenario& sc = scenarios_[s];
                const auto states = effective_states(sc);
                std::vector<RawState> queried;
                if (s == 0)
                {
                    queried = showcase_state_;  // the picture's own draw (render()); the buffers still hold its result
                }
                else
                {
                    queried = run_scenario(sc);
                    reset_blend_state();
                }
                std::cout << std::format("     scenario {}: {}\n", s, sc.name);
                const BufState base = base_state(sc);
                expect_states(check, queried,
                              [&](std::size_t i) { return raw(i < kBuffers ? states[i] : base); },
                              std::format("scenario {}: glIsEnabledi / glGetIntegeri_v / glGetBooleani_v of all {} draw buffer indices equal the commands issued", s, max_draw_buffers_));

                for (std::size_t b = 0; b < kBuffers; ++b)
                {
                    const std::vector<double> px = read_attachment(b);
                    double max_dev = 0.0;
                    std::string first;
                    const int bad = compare_attachment(px, b, states[b], sc, max_dev, first);
                    const std::string what = std::format("scenario {} buffer {} ({}): [{}] all {} pixels equal the CPU blend model", s, b, kBufNames[b],
                                                         describe(states[b]), kMrtSize * kMrtSize);
                    const std::string detail = bad == 0 ? std::format("largest deviation {}{}", max_dev, kBufKinds[b] == BufKind::Unorm8 ? "/255" : "")
                                                        : std::format("{} pixels differ, {}", bad, first);
                    if (sc.zink_ignores_mask)
                        check.expect_or_known_deviation(bad == 0, what, detail, "zink",
                                                        "zink (lavapipe) blends and stores the green channel although the write mask (TFTT) masks it while blending is enabled; "
                                                        "ES 3.2 15.2.2: a masked component is not written");
                    else
                        check.expect(bad == 0, what, detail);
                }
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

                if (s == 2)
                {
                    // Negative control: the buffers do not all follow one state. Compare every buffer with the model of the NEXT buffer's state.
                    int mismatching = 0;
                    for (std::size_t b = 0; b < kBuffers; ++b)
                    {
                        const std::vector<double> px = read_attachment(b);
                        double max_dev = 0.0;
                        std::string first;
                        if (compare_attachment(px, b, states[(b + 1) % kBuffers], sc, max_dev, first) > 0) ++mismatching;
                    }
                    mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                    check.expect(mismatching == kBuffers, "negative control: every buffer's result differs from the result its neighbour's blend state would give (the per-buffer comparison can fail)",
                                 std::format("{} of {} buffers differ", mismatching, kBuffers));
                }
            }
        }

        void check_initial_indexed_state(demo::Check& check)
        {
            check.expect(max_draw_buffers_ >= 4, "MAX_DRAW_BUFFERS >= 4", std::format("MAX_DRAW_BUFFERS = {}", max_draw_buffers_));
            // ES 3.2 15.1.5.5: equations FUNC_ADD, functions ONE / ZERO, blending disabled for all draw buffers; 15.2.2: all color values enabled.
            expect_states(check, initial_, [](std::size_t) { return raw(BufState{}); },
                          std::format("initial state of all {} draw buffer indices: blending off, FUNC_ADD, ONE / ZERO, write mask TTTT (15.1.5.5, 15.2.2)", max_draw_buffers_));
            check.expect(!initial_enabled_global_ && initial_src_rgb_global_ == value_of(mgl::BlendFactor::One),
                         "initially glIsEnabled(BLEND) = FALSE and glGetIntegerv(BLEND_SRC_RGB) = ONE (draw buffer zero)");
            check.expect(initial_blend_color_ == std::array<float, 4>{0, 0, 0, 0}, "initial BLEND_COLOR is (0, 0, 0, 0) (15.1.5.5)");
        }

        /// A distinct, index-dependent state for every draw buffer index.
        static Op junk_op(std::size_t i)
        {
            using BF = mgl::BlendFactor;
            constexpr std::array<BF, 7> f = {BF::One, BF::SrcAlpha, BF::DstColor, BF::OneMinusConstantAlpha, BF::ConstantColor, BF::OneMinusSrcColor, BF::DstAlpha};
            constexpr std::array<mgl::BlendEquation, 5> e = {mgl::BlendEquation::FuncAdd, mgl::BlendEquation::FuncSubtract, mgl::BlendEquation::FuncReverseSubtract,
                                                              mgl::BlendEquation::Min, mgl::BlendEquation::Max};
            return Op{}.on((i & 1) != 0).fs(f[i % 7], f[(i + 1) % 7], f[(i + 2) % 7], f[(i + 3) % 7]).es(e[i % 5], e[(i + 2) % 5]).m((i & 1) == 0, (i & 2) == 0, (i & 4) == 0, (i & 3) != 3);
        }

        void check_nonindexed_commands(demo::Check& check)
        {
            using BF = mgl::BlendFactor;
            using BE = mgl::BlendEquation;
            const std::size_t n = static_cast<std::size_t>(max_draw_buffers_);
            auto set_junk = [&]() {
                reset_blend_state();
                for (std::size_t i = 0; i < n; ++i) issue_indexed(static_cast<GLuint>(i), junk_op(i));
            };
            auto junk_states = [&]() {
                std::vector<BufState> v(n);
                for (std::size_t i = 0; i < n; ++i) v[i] = apply_to(BufState{}, junk_op(i));
                return v;
            };

            set_junk();
            expect_states(check, query_all(), [&](std::size_t i) { return raw(junk_states()[i]); },
                          std::format("{} different indexed states set with glEnablei/glDisablei/glBlendFuncSeparatei/glBlendEquationSeparatei/glColorMaski are read back per index", n));

            struct Case
            {
                std::string name;
                Op op;
            };
            const std::vector<Case> cases = {
                {"glEnable(BLEND)", Op{}.on(true)},
                {"glDisable(BLEND)", Op{}.on(false)},
                {"glBlendFunc(DST_COLOR, ONE_MINUS_SRC_ALPHA)", Op{}.f(BF::DstColor, BF::OneMinusSrcAlpha)},
                {"glBlendFuncSeparate(SRC_ALPHA, ONE, DST_ALPHA, ONE_MINUS_DST_COLOR)", Op{}.fs(BF::SrcAlpha, kOne, BF::DstAlpha, BF::OneMinusDstColor)},
                {"glBlendEquation(MAX)", Op{}.e(BE::Max)},
                {"glBlendEquationSeparate(FUNC_SUBTRACT, MIN)", Op{}.es(BE::FuncSubtract, BE::Min)},
                {"glColorMask(FALSE, TRUE, FALSE, TRUE)", Op{}.m(false, true, false, true)},
            };
            for (const Case& cs : cases)
            {
                set_junk();
                issue_global(cs.op);
                const std::vector<BufState> before = junk_states();
                expect_states(check, query_all(), [&](std::size_t i) { return raw(apply_to(before[i], cs.op)); },
                              std::format("{} sets that state of all {} draw buffers and leaves every other field of each buffer alone (15.1.5, 15.2.2)", cs.name, n));
            }

            // The non-indexed queries describe draw buffer zero (15.1.5.5, 15.2.2).
            reset_blend_state();
            issue_indexed(0, Op{}.on(false).fs(BF::SrcColor, BF::DstAlpha, BF::ConstantColor, BF::OneMinusDstColor).es(BE::Min, BE::FuncSubtract).m(true, false, true, false));
            issue_indexed(1, Op{}.on(true).f(BF::One, BF::One).e(BE::Max).m(false, true, false, true));
            const bool enabled0 = mgl::glIsEnabled(mgl::Capability::Blend);
            std::array<GLint, 6> g{};
            mgl::glGetIntegerv(mgl::GetParameter::BlendSrcRgb, &g[0]);
            mgl::glGetIntegerv(mgl::GetParameter::BlendDstRgb, &g[1]);
            mgl::glGetIntegerv(mgl::GetParameter::BlendSrcAlpha, &g[2]);
            mgl::glGetIntegerv(mgl::GetParameter::BlendDstAlpha, &g[3]);
            mgl::glGetIntegerv(mgl::GetParameter::BlendEquationRgb, &g[4]);
            mgl::glGetIntegerv(mgl::GetParameter::BlendEquationAlpha, &g[5]);
            std::array<GLboolean, 4> mask{};
            mgl::glGetBooleanv(mgl::GetParameter::ColorWritemask, mask.data());
            check.expect(!enabled0 && g[0] == value_of(BF::SrcColor) && g[1] == value_of(BF::DstAlpha) && g[2] == value_of(BF::ConstantColor) &&
                             g[3] == value_of(BF::OneMinusDstColor) && g[4] == value_of(BE::Min) && g[5] == value_of(BE::FuncSubtract) &&
                             mask[0] == GL_TRUE && mask[1] == GL_FALSE && mask[2] == GL_TRUE && mask[3] == GL_FALSE,
                         "glIsEnabled(BLEND), glGetIntegerv(BLEND_*) and glGetBooleanv(COLOR_WRITEMASK) report draw buffer zero, not buffer 1 (15.1.5.5, 15.2.2)",
                         std::format("enabled {} src/dst rgb {:#x}/{:#x} src/dst alpha {:#x}/{:#x} eq {:#x}/{:#x} mask {}{}{}{}", enabled0, g[0], g[1], g[2], g[3], g[4], g[5],
                                     mask[0] != GL_FALSE, mask[1] != GL_FALSE, mask[2] != GL_FALSE, mask[3] != GL_FALSE));
            issue_indexed(0, Op{}.on(true));
            issue_indexed(1, Op{}.on(false));
            check.expect(mgl::glIsEnabledi(mgl::Capability::Blend, 0) && !mgl::glIsEnabledi(mgl::Capability::Blend, 1),
                         "glIsEnabledi(BLEND, 0) = TRUE and glIsEnabledi(BLEND, 1) = FALSE after glEnablei(0) / glDisablei(1)");
            // 15.1.5.5: "The value of the blend enable ... for draw buffer zero may also be queried by calling IsEnabled".
            check.expect_or_known_deviation(mgl::glIsEnabled(mgl::Capability::Blend),
                                            "glIsEnabled(BLEND) is TRUE when buffer zero is enabled, although buffer 1 is disabled (15.1.5.5)", "", "radeonsi",
                                            "radeonsi's glIsEnabled(BLEND) is TRUE only when blending is enabled for all draw buffers, not the value of draw buffer zero that ES 3.2 15.1.5.5 specifies");
            reset_blend_state();
            expect_states(check, query_all(), [](std::size_t) { return raw(BufState{}); },
                          "the non-indexed reset (glDisable, glBlendFunc, glBlendEquation, glColorMask) restores the initial state of all indices");
        }

        void check_indexed_errors(demo::Check& check)
        {
            using BF = mgl::BlendFactor;
            using BE = mgl::BlendEquation;
            const GLuint max = static_cast<GLuint>(max_draw_buffers_);
            reset_blend_state();
            demo::drain_gl_errors();
            const std::vector<RawState> before = query_all();

            // "An INVALID_VALUE error is generated by Enablei, Disablei and IsEnabledi if target is BLEND and index >= MAX_DRAW_BUFFERS" (15.1.5).
            mgl::glEnablei(mgl::Capability::Blend, max);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glEnablei(BLEND, MAX_DRAW_BUFFERS): INVALID_VALUE");
            mgl::glDisablei(mgl::Capability::Blend, max);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDisablei(BLEND, MAX_DRAW_BUFFERS): INVALID_VALUE");
            (void)mgl::glIsEnabledi(mgl::Capability::Blend, max);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glIsEnabledi(BLEND, MAX_DRAW_BUFFERS): INVALID_VALUE");
            mgl::glEnablei(mgl::Capability::Blend, 0xFFFFFFFFu);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glEnablei(BLEND, 0xFFFFFFFF): INVALID_VALUE");
            // "An INVALID_VALUE error is generated if buf is not in the range zero to MAX_DRAW_BUFFERS minus one" (15.1.5.1, 15.1.5.2).
            mgl::glBlendEquationi(max, BE::Max);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBlendEquationi(MAX_DRAW_BUFFERS, ...): INVALID_VALUE");
            mgl::glBlendEquationSeparatei(max, BE::Max, BE::Min);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBlendEquationSeparatei(MAX_DRAW_BUFFERS, ...): INVALID_VALUE");
            mgl::glBlendFunci(max, BF::SrcAlpha, BF::OneMinusSrcAlpha);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBlendFunci(MAX_DRAW_BUFFERS, ...): INVALID_VALUE");
            mgl::glBlendFuncSeparatei(max, BF::SrcAlpha, BF::One, BF::One, BF::Zero);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBlendFuncSeparatei(MAX_DRAW_BUFFERS, ...): INVALID_VALUE");
            // "An INVALID_VALUE error is generated by ColorMaski if buf is greater than the value of MAX_DRAW_BUFFERS minus one" (15.2.2).
            mgl::glColorMaski(max, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glColorMaski(MAX_DRAW_BUFFERS, ...): INVALID_VALUE");
            // "An INVALID_VALUE error is generated if index is outside the valid range for the indexed state target" (20.1).
            GLint scratch = 0;
            mgl::glGetIntegeri_v(mgl::GetParameter::BlendSrcRgb, max, &scratch);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(BLEND_SRC_RGB, MAX_DRAW_BUFFERS): INVALID_VALUE");
            std::array<GLboolean, 4> mask{};
            mgl::glGetBooleani_v(mgl::GetParameter::ColorWritemask, max, mask.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetBooleani_v(COLOR_WRITEMASK, MAX_DRAW_BUFFERS): INVALID_VALUE");
            // "An INVALID_ENUM error is generated if target is not indexed enable state queriable with IsEnabledi" (20.1) / "not indexed state" (20.1).
            (void)mgl::glIsEnabledi(mgl::Capability::DepthTest, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glIsEnabledi(DEPTH_TEST, 0): INVALID_ENUM (DEPTH_TEST is not indexed state)");
            mgl::glGetIntegeri_v(mgl::GetParameter::DepthFunc, 0, &scratch);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glGetIntegeri_v(DEPTH_FUNC, 0): INVALID_ENUM (not indexed state)");

            expect_states(check, query_all(), [&](std::size_t i) { return before[i]; }, "none of the rejected commands changed any blend state");
        }

        // =========================================================================================
        // Members
        // =========================================================================================
        static constexpr int kSwatchW = 36;
        static constexpr int kSwatchX0 = 2;

        int max_draw_buffers_ = 0;
        int max_color_attachments_ = 0;
        bool float_blend_ = false, half_float_ext_ = false, color_buffer_float_ext_ = false;

        std::vector<RawState> initial_;
        bool initial_enabled_global_ = false;
        int initial_src_rgb_global_ = 0;
        std::array<float, 4> initial_blend_color_{};

        mgl::ProgramId pattern_{}, outputs_{}, scene_{}, glow_{}, compose_{};

        std::array<mgl::TextureId, 7> fmt_tex_{};
        std::array<mgl::FramebufferId, 7> fmt_fbo_{};
        std::array<mgl::FramebufferStatus, 7> fmt_status_{};
        mgl::TextureId ctrl_tex_{};
        mgl::FramebufferId ctrl_fbo_{};

        mgl::TextureId hdr_tex_{};
        mgl::RenderbufferId hdr_depth_{};
        mgl::FramebufferId hdr_fbo_{};
        mgl::VertexArrayId scene_vao_{};
        mgl::BufferId scene_vbo_{};
        GLsizei floor_count_ = 0;
        GLint box_first_ = 0, sphere_first_ = 0;
        GLsizei box_vertices_ = 0, sphere_vertices_ = 0;

        std::array<mgl::TextureId, kBuffers> mrt_tex_{};
        mgl::FramebufferId mrt_fbo_{};

        std::vector<Scenario> scenarios_;
        std::vector<RawState> showcase_state_;
        Camera camera_;
        std::vector<Probe> probes_;
        std::vector<float> hdr_;
        std::vector<std::uint8_t> window_;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es32-float-targets-and-indexed-blend",
        .title = "Float render targets and indexed blending",
        .min_es = {3, 2},
        .summary = "Color-renderable float formats of ES 3.2 (R16F ... RGBA32F, R11F_G11F_B10F) hold HDR values that read back exactly, with an HDR "
                   "tone-mapped 3D scene; per-draw-buffer blend state (glEnablei, glBlendFunci, glColorMaski, ...) on an MRT framebuffer against a CPU blend model.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
        .depth_bits = 24,
        .stencil_bits = 8,
    };
    return demo::run<FloatTargetsAndIndexedBlend>(argc, argv, info);
}
