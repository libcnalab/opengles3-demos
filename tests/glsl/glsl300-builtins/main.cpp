// test-glsl300-builtins
//
// Numerical checks of the GLSL ES 3.00 built-in functions (§8.1-§8.7) and integer operators.
//
// Method: ES 3.0 does not require float render targets, and an RGBA8 framebuffer would quantize
// results, so every expression is evaluated in a *vertex shader* and its outputs (4 x vec4/ivec4/
// uvec4) are captured with transform feedback into a buffer (exact 32-bit values), drawn with
// RASTERIZER_DISCARD, read back with glMapBufferRange, and compared with a CPU reference.
//
// Tolerances follow GLSL ES 3.00 §4.5.1 "Range and Precision" (highp):
//   Exact      - results defined exactly (abs, floor, min, bit casts, packing of exact inputs, ...)
//   Ulp(n)     - n ULP, from the precision table (+,-,* correctly rounded; 1/x 2.5; inversesqrt 2;
//                exp/exp2 3+2|x|; derived bounds for functions built from those operations)
//   Log        - log/log2: 3 ULP outside [0.5, 2], absolute 2^-21 inside
//   Quality    - the spec leaves precision UNDEFINED (trigonometric, hyperbolic, determinant,
//                inverse). Only a sanity bound (1e-3, catches a wrong function) is asserted; the
//                measured maximum error is printed as an implementation-quality observation.
// Undefined results (asin(2), log(0), % with negative operands, ...) are never asserted.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    using V4 = std::array<float, 4>;
    using I4 = std::array<GLint, 4>;
    using U4 = std::array<GLuint, 4>;
    using Out = std::array<double, 16>;

    struct Row
    {
        V4 a{}, b{}, c{};
        I4 ia{}, ib{};
        U4 ua{}, ub{};
    };

    enum class OutType { Float, Int, Uint };
    enum class TolKind { Exact, Ulp, ExpUlp, Log, Abs, Rel, QualityAbs, QualityRel };

    struct Tol
    {
        TolKind kind = TolKind::Exact;
        double value = 0.0;
    };

    struct Case
    {
        const char* name;
        const char* feature;
        OutType type;
        int components;      // how many of the 16 output components are meaningful
        const char* body;    // GLSL: reads a, b, c, ia, ib, ua, ub; writes r0..r3
        std::function<Out(const Row&)> ref;
        Tol tol;
        std::vector<Row> rows;
    };

    double ulp_of(double r)
    {
        const double m = std::max(std::abs(r), std::ldexp(1.0, -126));
        return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(m))) - 23);
    }

    // Applies f lane-wise to a (and b, c) and returns the 4 results in r0.
    template <typename F>
    std::function<Out(const Row&)> lanes(F f)
    {
        return [f](const Row& r) {
            Out o{};
            for (int i = 0; i < 4; ++i) o[static_cast<std::size_t>(i)] = f(r, i);
            return o;
        };
    }

    Row A(V4 a, V4 b = {}, V4 c = {}) { return Row{a, b, c}; }
    Row IA(I4 ia, I4 ib = {}) { Row r; r.ia = ia; r.ib = ib; return r; }
    Row UA(U4 ua, U4 ub = {}) { Row r; r.ua = ua; r.ub = ub; return r; }

    double fa(const Row& r, int i) { return r.a[static_cast<std::size_t>(i)]; }
    double fb(const Row& r, int i) { return r.b[static_cast<std::size_t>(i)]; }
    double fc(const Row& r, int i) { return r.c[static_cast<std::size_t>(i)]; }

    std::uint32_t f2u(float f) { return std::bit_cast<std::uint32_t>(f); }
    float u2f(std::uint32_t u) { return std::bit_cast<float>(u); }

    // IEEE half from float with round-to-nearest-even (inputs are chosen exactly representable).
    std::uint16_t to_half(float f)
    {
        const std::uint32_t x = f2u(f);
        const std::uint32_t sign = (x >> 16) & 0x8000u;
        const int exp = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
        const std::uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFF) == 0) return static_cast<std::uint16_t>(sign);
        if (exp <= 0) return static_cast<std::uint16_t>(sign | ((mant | 0x800000u) >> (14 - exp)));
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
    }

    std::vector<Case> make_cases()
    {
        const double pi = std::numbers::pi;
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<Case> c;

        // ---------------- §8.1 angle and trigonometry ---------------------------------------------
        c.push_back({"radians", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = radians(a);",
                     lanes([pi](const Row& r, int i) { return fa(r, i) * pi / 180.0; }), {TolKind::Ulp, 2},
                     {A({0, 90, -180, 45}), A({360, 1, -30, 720})}});
        c.push_back({"degrees", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = degrees(a);",
                     lanes([pi](const Row& r, int i) { return fa(r, i) * 180.0 / pi; }), {TolKind::Ulp, 2},
                     {A({0, 1, -3.14159265f, 0.5f})}});
        const std::vector<Row> angles = {A({0, 0.5f, -1, 2}), A({3, -2.5f, 0.125f, 1.5f})};
        c.push_back({"sin", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = sin(a);", lanes([](const Row& r, int i) { return std::sin(fa(r, i)); }), {TolKind::QualityAbs, 1e-3}, angles});
        c.push_back({"cos", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = cos(a);", lanes([](const Row& r, int i) { return std::cos(fa(r, i)); }), {TolKind::QualityAbs, 1e-3}, angles});
        c.push_back({"tan", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = tan(a);", lanes([](const Row& r, int i) { return std::tan(fa(r, i)); }), {TolKind::QualityRel, 1e-3},
                     {A({0.1f, 0.5f, -1, 1.2f})}});
        const std::vector<Row> unit = {A({0, 0.5f, -0.5f, 1}), A({-1, 0.25f, 0.9f, -0.75f})};
        c.push_back({"asin", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = asin(a);", lanes([](const Row& r, int i) { return std::asin(fa(r, i)); }), {TolKind::QualityAbs, 1e-3}, unit});
        c.push_back({"acos", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = acos(a);", lanes([](const Row& r, int i) { return std::acos(fa(r, i)); }), {TolKind::QualityAbs, 1e-3}, unit});
        c.push_back({"atan(y_over_x)", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = atan(a);", lanes([](const Row& r, int i) { return std::atan(fa(r, i)); }), {TolKind::QualityAbs, 1e-3},
                     {A({0, 1, -1, 100})}});
        c.push_back({"atan(y, x) all quadrants", "GLSL300.FN.TRIG", OutType::Float, 4, "r0 = atan(a, b);",
                     lanes([](const Row& r, int i) { return std::atan2(fa(r, i), fb(r, i)); }), {TolKind::QualityAbs, 1e-3},
                     {A({1, 1, -1, -1}, {1, -1, -1, 1}), A({0.5f, 2, -3, 0}, {-2, 0.25f, 4, 1})}});
        c.push_back({"sinh/cosh/tanh", "GLSL300.FN.TRIG", OutType::Float, 12, "r0 = sinh(a); r1 = cosh(a); r2 = tanh(a);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = std::sinh(fa(r, i)); o[4 + i] = std::cosh(fa(r, i)); o[8 + i] = std::tanh(fa(r, i)); } return o; },
                     {TolKind::QualityRel, 1e-3}, {A({0, 0.5f, -1, 2})}});
        c.push_back({"asinh/acosh/atanh", "GLSL300.FN.TRIG", OutType::Float, 12, "r0 = asinh(a); r1 = acosh(b); r2 = atanh(c);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = std::asinh(fa(r, i)); o[4 + i] = std::acosh(fb(r, i)); o[8 + i] = std::atanh(fc(r, i)); } return o; },
                     {TolKind::QualityRel, 1e-3}, {A({0.5f, -1, 2, 0.1f}, {1.5f, 2, 10, 1.25f}, {0.5f, -0.25f, 0.75f, 0.1f})}});

        // ---------------- §8.2 exponential --------------------------------------------------------
        c.push_back({"pow", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = pow(a, b);",
                     lanes([](const Row& r, int i) { return std::pow(fa(r, i), fb(r, i)); }), {TolKind::Rel, 1e-5},  // inherited from exp2(y*log2(x))
                     {A({2, 9, 0.5f, 10}, {3, 0.5f, -2, 1.5f})}});
        c.push_back({"exp", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = exp(a);", lanes([](const Row& r, int i) { return std::exp(fa(r, i)); }), {TolKind::ExpUlp, 0},
                     {A({0, 1, -2, 3.5f})}});
        c.push_back({"exp2", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = exp2(a);", lanes([](const Row& r, int i) { return std::exp2(fa(r, i)); }), {TolKind::ExpUlp, 0},
                     {A({0, 1, -3, 10}), A({0.5f, -0.5f, 7.25f, 20})}});
        const std::vector<Row> positive = {A({1, 2, 0.5f, 10}), A({0.75f, 100, 1.5f, 0.001f})};
        c.push_back({"log", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = log(a);", lanes([](const Row& r, int i) { return std::log(fa(r, i)); }), {TolKind::Log, 0}, positive});
        c.push_back({"log2", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = log2(a);", lanes([](const Row& r, int i) { return std::log2(fa(r, i)); }), {TolKind::Log, 0}, positive});
        c.push_back({"sqrt", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = sqrt(a);", lanes([](const Row& r, int i) { return std::sqrt(fa(r, i)); }), {TolKind::Ulp, 5},  // 1/inversesqrt: 2 + 2.5 ULP
                     {A({4, 2, 0.25f, 1e6f}), A({9, 3, 0.5f, 123.456f})}});
        c.push_back({"inversesqrt", "GLSL300.FN.EXP", OutType::Float, 4, "r0 = inversesqrt(a);", lanes([](const Row& r, int i) { return 1.0 / std::sqrt(fa(r, i)); }), {TolKind::Ulp, 2},
                     {A({4, 2, 0.25f, 1e6f}), A({9, 3, 0.5f, 123.456f})}});

        // ---------------- §8.3 common -------------------------------------------------------------
        const std::vector<Row> mixed = {A({1.75f, -1.75f, 0, -0.25f}), A({2.5f, -3.5f, 1e7f, -0.0f})};
        c.push_back({"abs/sign (float)", "GLSL300.FN.COMMON", OutType::Float, 8, "r0 = abs(a); r1 = sign(a);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = std::abs(fa(r, i)); o[4 + i] = fa(r, i) > 0 ? 1 : fa(r, i) < 0 ? -1 : 0; } return o; },
                     {TolKind::Exact, 0}, mixed});
        c.push_back({"floor/trunc/ceil/fract", "GLSL300.FN.COMMON", OutType::Float, 16, "r0 = floor(a); r1 = trunc(a); r2 = ceil(a); r3 = fract(a);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { const double x = fa(r, i); o[i] = std::floor(x); o[4 + i] = std::trunc(x); o[8 + i] = std::ceil(x); o[12 + i] = x - std::floor(x); } return o; },
                     {TolKind::Exact, 0}, mixed});
        c.push_back({"round (no .5 ties) / roundEven (ties to even)", "GLSL300.FN.COMMON", OutType::Float, 8, "r0 = round(a); r1 = roundEven(b);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = std::round(fa(r, i)); o[4 + i] = std::nearbyint(fb(r, i)); } return o; },
                     {TolKind::Exact, 0}, {A({1.4f, 1.6f, -2.3f, -2.7f}, {0.5f, 1.5f, 2.5f, -3.5f}), A({0.2f, -0.6f, 7.9f, 100.1f}, {4.5f, -0.5f, 5.5f, 6.5f})}});
        c.push_back({"mod(x, y) = x - y*floor(x/y)", "GLSL300.FN.COMMON", OutType::Float, 8, "r0 = mod(a, b); r1 = mod(a, 3.0);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = fa(r, i) - fb(r, i) * std::floor(fa(r, i) / fb(r, i)); o[4 + i] = fa(r, i) - 3.0 * std::floor(fa(r, i) / 3.0); } return o; },
                     {TolKind::Abs, 1e-5}, {A({5.5f, -5.5f, 7, -7.25f}, {2, 2, -3, -2})}});
        c.push_back({"modf", "GLSL300.FN.COMMON", OutType::Float, 8, "vec4 ip; r0 = modf(a, ip); r1 = ip;",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { double ip; o[i] = std::modf(fa(r, i), &ip); o[4 + i] = ip; } return o; },
                     {TolKind::Exact, 0}, {A({3.25f, -3.25f, 0.5f, 100})}});
        c.push_back({"min/max/clamp (float, incl. scalar forms)", "GLSL300.FN.COMMON", OutType::Float, 16,
                     "r0 = min(a, b); r1 = max(a, b); r2 = clamp(a, -1.0, 1.0); r3 = clamp(a, b, c);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = std::min(fa(r, i), fb(r, i)); o[4 + i] = std::max(fa(r, i), fb(r, i)); o[8 + i] = std::clamp(fa(r, i), -1.0, 1.0); o[12 + i] = std::min(std::max(fa(r, i), fb(r, i)), fc(r, i)); } return o; },
                     {TolKind::Exact, 0}, {A({-2, 0.5f, 3, -0.25f}, {1, 0.25f, -3, 0}, {2, 0.375f, 5, 0})}});
        c.push_back({"mix(x, y, a) and mix(x, y, bvec)", "GLSL300.FN.COMMON", OutType::Float, 8, "r0 = mix(a, b, c); r1 = mix(a, b, greaterThan(c, vec4(0.5)));",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = fa(r, i) * (1 - fc(r, i)) + fb(r, i) * fc(r, i); o[4 + i] = fc(r, i) > 0.5 ? fb(r, i) : fa(r, i); } return o; },
                     {TolKind::Ulp, 2}, {A({0, 10, -4, 1}, {8, 20, 4, 3}, {0.25f, 0.5f, 1, 0})}});
        c.push_back({"step/smoothstep", "GLSL300.FN.COMMON", OutType::Float, 8, "r0 = step(b, a); r1 = smoothstep(0.0, 2.0, a);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = fa(r, i) < fb(r, i) ? 0 : 1; const double t = std::clamp(fa(r, i) / 2.0, 0.0, 1.0); o[4 + i] = t * t * (3 - 2 * t); } return o; },
                     {TolKind::Abs, 1e-6}, {A({1, 0.5f, -1, 3}, {1, 1, 0, 2.5f})}});
        c.push_back({"isnan/isinf", "GLSL300.FN.COMMON", OutType::Uint, 8, "r0 = uvec4(isnan(a)); r1 = uvec4(isinf(a));",
                     [](const Row&) { return Out{1, 0, 0, 0, 0, 1, 1, 0}; }, {TolKind::Exact, 0}, {A({nan, inf, -inf, 1})}});
        c.push_back({"floatBitsToInt/Uint", "GLSL300.FN.COMMON", OutType::Uint, 8, "r0 = uvec4(floatBitsToInt(a)); r1 = floatBitsToUint(a);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) o[i] = o[4 + i] = f2u(r.a[static_cast<std::size_t>(i)]); return o; },
                     {TolKind::Exact, 0}, {A({1, -2.5f, 0, -0.0f}), A({inf, 1e-30f, 3.1415927f, -65504})}});
        c.push_back({"intBitsToFloat/uintBitsToFloat", "GLSL300.FN.COMMON", OutType::Uint, 8,
                     "r0 = floatBitsToUint(intBitsToFloat(ia)); r1 = floatBitsToUint(uintBitsToFloat(ua));",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { o[i] = static_cast<GLuint>(r.ia[static_cast<std::size_t>(i)]); o[4 + i] = r.ua[static_cast<std::size_t>(i)]; } return o; },
                     {TolKind::Exact, 0}, {[] { Row r; r.ia = {0x3F800000, static_cast<GLint>(0xC0200000u), 0x7F800000, 1}; r.ua = {0x40490FDBu, 0x80000000u, 0x00800000u, 0x3EAAAAABu}; return r; }()}});

        // ---------------- §8.4 floating-point pack and unpack ---------------------------------------
        c.push_back({"packSnorm2x16/packUnorm2x16", "GLSL300.FN.PACKING", OutType::Uint, 4,
                     "r0 = uvec4(packSnorm2x16(a.xy), packSnorm2x16(a.zw), packUnorm2x16(b.xy), packUnorm2x16(b.zw));",
                     [](const Row& r) {
                         auto sn = [](double v) { return static_cast<std::uint32_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(std::round(std::clamp(v, -1.0, 1.0) * 32767.0)))); };
                         auto un = [](double v) { return static_cast<std::uint32_t>(std::round(std::clamp(v, 0.0, 1.0) * 65535.0)); };
                         return Out{double(sn(fa(r, 0)) | (sn(fa(r, 1)) << 16)), double(sn(fa(r, 2)) | (sn(fa(r, 3)) << 16)),
                                    double(un(fb(r, 0)) | (un(fb(r, 1)) << 16)), double(un(fb(r, 2)) | (un(fb(r, 3)) << 16))};
                     },
                     {TolKind::Exact, 0}, {A({1, -1, 0.5f, -2}, {0, 1, 0.5f, 3})}});
        c.push_back({"unpackSnorm2x16/unpackUnorm2x16", "GLSL300.FN.PACKING", OutType::Float, 8,
                     "r0 = vec4(unpackSnorm2x16(ua.x), unpackSnorm2x16(ua.y)); r1 = vec4(unpackUnorm2x16(ua.z), unpackUnorm2x16(ua.w));",
                     [](const Row& r) {
                         auto sn = [](std::uint32_t h) { return std::clamp(static_cast<std::int16_t>(h) / 32767.0, -1.0, 1.0); };
                         auto un = [](std::uint32_t h) { return h / 65535.0; };
                         Out o{};
                         o[0] = sn(r.ua[0] & 0xFFFF); o[1] = sn(r.ua[0] >> 16); o[2] = sn(r.ua[1] & 0xFFFF); o[3] = sn(r.ua[1] >> 16);
                         o[4] = un(r.ua[2] & 0xFFFF); o[5] = un(r.ua[2] >> 16); o[6] = un(r.ua[3] & 0xFFFF); o[7] = un(r.ua[3] >> 16);
                         return o;
                     },
                     {TolKind::Ulp, 3}, {UA({0x80017FFFu, 0x40000000u, 0xFFFF0000u, 0x00018000u})}});
        c.push_back({"packHalf2x16/unpackHalf2x16", "GLSL300.FN.PACKING", OutType::Uint, 4,
                     "r0 = uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), floatBitsToUint(unpackHalf2x16(ua.x)));",
                     [](const Row& r) {
                         return Out{double(to_half(r.a[0]) | (std::uint32_t(to_half(r.a[1])) << 16)), double(to_half(r.a[2]) | (std::uint32_t(to_half(r.a[3])) << 16)),
                                    double(f2u(1.0f)), double(f2u(-2.5f))};
                     },
                     {TolKind::Exact, 0}, {[] { Row r = A({1, -2.5f, 0.0009765625f, 65504}); r.ua = {0xC1003C00u, 0, 0, 0}; return r; }()}});

        // ---------------- §8.5 geometric -------------------------------------------------------------
        c.push_back({"dot/cross (exact inputs)", "GLSL300.FN.GEOMETRIC", OutType::Float, 8, "r0 = vec4(dot(a, b), dot(a.xy, b.xy), 0.0, 0.0); r1 = vec4(cross(a.xyz, b.xyz), 0.0);",
                     [](const Row& r) { Out o{}; o[0] = fa(r, 0) * fb(r, 0) + fa(r, 1) * fb(r, 1) + fa(r, 2) * fb(r, 2) + fa(r, 3) * fb(r, 3); o[1] = fa(r, 0) * fb(r, 0) + fa(r, 1) * fb(r, 1);
                                        o[4] = fa(r, 1) * fb(r, 2) - fa(r, 2) * fb(r, 1); o[5] = fa(r, 2) * fb(r, 0) - fa(r, 0) * fb(r, 2); o[6] = fa(r, 0) * fb(r, 1) - fa(r, 1) * fb(r, 0); return o; },
                     {TolKind::Exact, 0}, {A({1, 2, 3, 4}, {5, -6, 7, 0.5f}), A({1, 0, 0, 0}, {0, 1, 0, 0})}});
        c.push_back({"length/distance/normalize", "GLSL300.FN.GEOMETRIC", OutType::Float, 8, "r0 = vec4(length(a), distance(a, b), length(a.xy), 0.0); r1 = normalize(a);",
                     [](const Row& r) { Out o{}; double l = 0, d = 0; for (int i = 0; i < 4; ++i) { l += fa(r, i) * fa(r, i); d += (fa(r, i) - fb(r, i)) * (fa(r, i) - fb(r, i)); }
                                        o[0] = std::sqrt(l); o[1] = std::sqrt(d); o[2] = std::hypot(fa(r, 0), fa(r, 1)); for (int i = 0; i < 4; ++i) o[4 + i] = fa(r, i) / std::sqrt(l); return o; },
                     {TolKind::Rel, 1e-6}, {A({3, 4, 0, 0}, {0, 0, 0, 0}), A({1, 2, 2, 4}, {-1, 0, 3, 1})}});
        c.push_back({"faceforward/reflect/refract", "GLSL300.FN.GEOMETRIC", OutType::Float, 12,
                     "r0 = faceforward(a, b, c); r1 = reflect(b, normalize(c)); r2 = vec4(refract(b.xyz, normalize(c.xyz), 0.5), 0.0);",
                     [](const Row& r) {
                         Out o{};
                         double d = 0; for (int i = 0; i < 4; ++i) d += fc(r, i) * fb(r, i);
                         for (int i = 0; i < 4; ++i) o[i] = d < 0 ? fa(r, i) : -fa(r, i);
                         double ln = 0; for (int i = 0; i < 4; ++i) ln += fc(r, i) * fc(r, i); ln = std::sqrt(ln);
                         double nd = 0; for (int i = 0; i < 4; ++i) nd += fc(r, i) / ln * fb(r, i);
                         for (int i = 0; i < 4; ++i) o[4 + i] = fb(r, i) - 2 * nd * fc(r, i) / ln;
                         double ln3 = std::sqrt(fc(r, 0) * fc(r, 0) + fc(r, 1) * fc(r, 1) + fc(r, 2) * fc(r, 2)), nd3 = 0;
                         for (int i = 0; i < 3; ++i) nd3 += fc(r, i) / ln3 * fb(r, i);
                         const double eta = 0.5, k = 1 - eta * eta * (1 - nd3 * nd3);
                         for (int i = 0; i < 3; ++i) o[8 + i] = k < 0 ? 0 : eta * fb(r, i) - (eta * nd3 + std::sqrt(k)) * fc(r, i) / ln3;
                         return o;
                     },
                     {TolKind::Rel, 1e-5}, {A({1, 2, 3, 0}, {0.6f, -0.8f, 0, 0}, {0, 1, 0, 0}), A({-1, 0.5f, 2, 1}, {0.5f, 0.5f, -0.70710678f, 0}, {0, 0, 1, 0})}});

        // ---------------- §8.6 matrix -----------------------------------------------------------------
        c.push_back({"matrixCompMult/outerProduct/transpose", "GLSL300.FN.MATRIX", OutType::Float, 16,
                     "mat2 m = mat2(a); mat2 n = mat2(b); mat2 p = matrixCompMult(m, n); r0 = vec4(p[0], p[1]);"
                     " mat2x3 o = outerProduct(c.xyz, a.xy); r1 = vec4(o[0], o[1].x); r2 = vec4(o[1].yz, 0.0, 0.0);"
                     " mat2 t = transpose(m); r3 = vec4(t[0], t[1]);",
                     [](const Row& r) {
                         Out o{};
                         for (int i = 0; i < 4; ++i) o[i] = fa(r, i) * fb(r, i);
                         // outerProduct(c, a.xy): column j = c * a[j] -> mat2x3 columns (c*a0), (c*a1)
                         o[4] = fc(r, 0) * fa(r, 0); o[5] = fc(r, 1) * fa(r, 0); o[6] = fc(r, 2) * fa(r, 0); o[7] = fc(r, 0) * fa(r, 1);
                         o[8] = fc(r, 1) * fa(r, 1); o[9] = fc(r, 2) * fa(r, 1);
                         o[12] = fa(r, 0); o[13] = fa(r, 2); o[14] = fa(r, 1); o[15] = fa(r, 3);  // transpose of column-major (a0 a1 | a2 a3)
                         return o;
                     },
                     {TolKind::Exact, 0}, {A({1, 2, 3, 4}, {5, 6, -7, 0.5f}, {2, -3, 0.25f, 0})}});
        c.push_back({"determinant/inverse (mat2, mat3)", "GLSL300.FN.MATRIX", OutType::Float, 12,
                     "mat2 m = mat2(a); mat2 mi = inverse(m); r0 = vec4(determinant(m), determinant(mat3(a.xyz, b.xyz, c.xyz)), 0.0, 0.0); r1 = vec4(mi[0], mi[1]);"
                     " mat3 n = mat3(a.xyz, b.xyz, c.xyz); mat3 ni = inverse(n); r2 = vec4(ni[0][0], ni[1][1], ni[2][2], ni[2][0]);",
                     [](const Row& r) {
                         Out o{};
                         const double d2 = fa(r, 0) * fa(r, 3) - fa(r, 2) * fa(r, 1);
                         // mat3 columns: a.xyz, b.xyz, c.xyz -> element (row i, col j) = col_j[i]
                         auto m = [&](int i, int j) { return j == 0 ? fa(r, i) : j == 1 ? fb(r, i) : fc(r, i); };
                         const double d3 = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
                                           m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
                         o[0] = d2; o[1] = d3;
                         o[4] = fa(r, 3) / d2; o[5] = -fa(r, 1) / d2; o[6] = -fa(r, 2) / d2; o[7] = fa(r, 0) / d2;
                         auto cof = [&](int i, int j) {  // inverse(i, j) = cofactor(j, i) / det
                             int r0 = (j + 1) % 3, r1 = (j + 2) % 3, c0 = (i + 1) % 3, c1 = (i + 2) % 3;
                             return (m(r0, c0) * m(r1, c1) - m(r0, c1) * m(r1, c0)) / d3;
                         };
                         // ni[c][r] = inverse(row r, col c)
                         o[8] = cof(0, 0); o[9] = cof(1, 1); o[10] = cof(2, 2); o[11] = cof(0, 2);
                         return o;
                     },
                     {TolKind::QualityRel, 1e-3}, {A({4, 2, 7, 6}, {1, 3, 2, 0}, {0, 1, 5, 0})}});

        // ---------------- §8.7 vector relational ------------------------------------------------------
        c.push_back({"lessThan/lessThanEqual/greaterThan/greaterThanEqual", "GLSL300.FN.VECREL", OutType::Uint, 16,
                     "r0 = uvec4(lessThan(a, b)); r1 = uvec4(lessThanEqual(a, b)); r2 = uvec4(greaterThan(ia, ib)); r3 = uvec4(greaterThanEqual(ua, ub));",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { const auto k = static_cast<std::size_t>(i);
                                            o[i] = r.a[k] < r.b[k]; o[4 + i] = r.a[k] <= r.b[k]; o[8 + i] = r.ia[k] > r.ib[k]; o[12 + i] = r.ua[k] >= r.ub[k]; } return o; },
                     {TolKind::Exact, 0}, {[] { Row r = A({1, 2, 3, -4}, {2, 2, 1, -5}); r.ia = {1, -2, 3, 0}; r.ib = {0, -2, 4, 0}; r.ua = {1, 5, 0xFFFFFFFFu, 0}; r.ub = {2, 5, 7, 0}; return r; }()}});
        c.push_back({"equal/notEqual/any/all/not", "GLSL300.FN.VECREL", OutType::Uint, 12,
                     "r0 = uvec4(equal(a, b)); r1 = uvec4(notEqual(ia, ib)); r2 = uvec4(any(equal(a, b)), all(equal(a, b)), all(not(notEqual(ia, ia))), any(bvec4(false)));",
                     [](const Row& r) { Out o{}; bool an = false, al = true; for (int i = 0; i < 4; ++i) { const auto k = static_cast<std::size_t>(i); const bool e = r.a[k] == r.b[k];
                                            o[i] = e; o[4 + i] = r.ia[k] != r.ib[k]; an = an || e; al = al && e; } o[8] = an; o[9] = al; o[10] = 1; o[11] = 0; return o; },
                     {TolKind::Exact, 0}, {[] { Row r = A({1, 2, 3, 4}, {1, 0, 3, 5}); r.ia = {1, 2, 3, 4}; r.ib = {1, 0, 3, 5}; return r; }()}});

        // ---------------- §8.3 integer forms + §5 integer operators ----------------------------------
        c.push_back({"abs/sign/min/max/clamp (int)", "GLSL300.FN.COMMON", OutType::Int, 16,
                     "r0 = abs(ia); r1 = sign(ia); r2 = min(ia, ib); r3 = clamp(ia, ivec4(-5), max(ib, ivec4(5)));",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { const auto k = static_cast<std::size_t>(i); const int x = r.ia[k], y = r.ib[k];
                                            o[i] = std::abs(x); o[4 + i] = (x > 0) - (x < 0); o[8 + i] = std::min(x, y); o[12 + i] = std::clamp(x, -5, std::max(y, 5)); } return o; },
                     {TolKind::Exact, 0}, {IA({-7, 0, 9, -2147483647}, {3, -1, 20, 0})}});
        c.push_back({"min/max/clamp (uint)", "GLSL300.FN.COMMON", OutType::Uint, 12, "r0 = min(ua, ub); r1 = max(ua, ub); r2 = clamp(ua, 2u, 100u);",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { const auto k = static_cast<std::size_t>(i);
                                            o[i] = std::min(r.ua[k], r.ub[k]); o[4 + i] = std::max(r.ua[k], r.ub[k]); o[8 + i] = std::clamp(r.ua[k], 2u, 100u); } return o; },
                     {TolKind::Exact, 0}, {UA({1, 0xFFFFFFFFu, 50, 0}, {3, 7, 50, 1})}});
        c.push_back({"integer / % & | ^ ~ << >> (defined operand ranges)", "GLSL300.OPERATORS.BITWISE", OutType::Uint, 16,
                     "r0 = uvec4(ua / ub); r1 = ua % ub; r2 = uvec4((ua & ub).x, ua.y | ub.y, ua.z ^ ub.z, ~ua.w);"
                     " r3 = uvec4(ua.x << 4u, ua.y >> 3u, uint(ia.x / ib.x), uint(ia.y << 2));",
                     [](const Row& r) { Out o{}; for (int i = 0; i < 4; ++i) { const auto k = static_cast<std::size_t>(i); o[i] = r.ua[k] / r.ub[k]; o[4 + i] = r.ua[k] % r.ub[k]; }
                                        o[8] = r.ua[0] & r.ub[0]; o[9] = r.ua[1] | r.ub[1]; o[10] = r.ua[2] ^ r.ub[2]; o[11] = ~r.ua[3];
                                        o[12] = static_cast<GLuint>(r.ua[0] << 4u); o[13] = r.ua[1] >> 3u; o[14] = static_cast<GLuint>(r.ia[0] / r.ib[0]); o[15] = static_cast<GLuint>(r.ia[1] << 2); return o; },
                     {TolKind::Exact, 0}, {[] { Row r = UA({100, 0xF0F0F0F0u, 0x0000FFFFu, 0x12345678u}, {7, 0x0F0F0000u, 0x00FF00FFu, 13}); r.ia = {-17, 5, 0, 0}; r.ib = {5, 1, 1, 1}; return r; }()}});
        return c;
    }

    // Vertex layout: 3 vec4 + 2 ivec4 + 2 uvec4 per vertex.
    struct GpuRow
    {
        V4 a, b, c;
        I4 ia, ib;
        U4 ua, ub;
    };

    std::string vertex_source(const Case& cs)
    {
        const char* t = cs.type == OutType::Float ? "vec4" : cs.type == OutType::Int ? "ivec4" : "uvec4";
        const char* q = cs.type == OutType::Float ? "" : "flat ";
        return std::format(
            "#version 300 es\nprecision highp float;\nprecision highp int;\n"
            "layout(location = 0) in vec4 a;\nlayout(location = 1) in vec4 b;\nlayout(location = 2) in vec4 c;\n"
            "layout(location = 3) in ivec4 ia;\nlayout(location = 4) in ivec4 ib;\n"
            "layout(location = 5) in uvec4 ua;\nlayout(location = 6) in uvec4 ub;\n"
            "{0}out {1} r0;\n{0}out {1} r1;\n{0}out {1} r2;\n{0}out {1} r3;\n"
            "void main()\n{{\n    r0 = {1}(0); r1 = {1}(0); r2 = {1}(0); r3 = {1}(0);\n    {2}\n    gl_Position = vec4(0.0);\n}}\n",
            q, t, cs.body);
    }

    class Glsl300Builtins final : public demo::Demo
    {
    public:
        void init(demo::Env& /*env*/) override
        {
            cases_ = make_cases();
            std::string log;
            fs_ = demo::compile_shader(mgl::ShaderType::Fragment,
                                       "#version 300 es\nprecision mediump float;\nout vec4 o;\nvoid main() { o = vec4(0.0); }\n", log);
            if (fs_.value == 0) throw demo::Failure("fragment shader: " + log);
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glGenBuffers(1, &vbo_);
            mgl::glGenBuffers(1, &tfb_);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            for (const auto& cs : cases_) run_case(cs, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteBuffers(1, &tfb_);
            mgl::glDeleteShader(fs_);
        }

    private:
        void run_case(const Case& cs, demo::Check& check)
        {
            // --- program with transform feedback varyings (must be declared before linking) --------
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, vertex_source(cs), log);
            if (vs.value == 0)
            {
                check.expect(false, std::format("{}: compiles", cs.name), log);
                return;
            }
            const mgl::ProgramId program = mgl::glCreateProgram();
            mgl::glAttachShader(program, vs);
            mgl::glAttachShader(program, fs_);
            const GLchar* varyings[] = {"r0", "r1", "r2", "r3"};
            mgl::glTransformFeedbackVaryings(program, 4, varyings, mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(program);
            mgl::glDeleteShader(vs);
            GLint linked = GL_FALSE;
            mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &linked);
            if (linked != GL_TRUE)
            {
                check.expect(false, std::format("{}: links", cs.name), demo::program_info_log(program));
                mgl::glDeleteProgram(program);
                return;
            }

            // --- inputs -----------------------------------------------------------------------------
            std::vector<GpuRow> rows;
            for (const auto& r : cs.rows) rows.push_back({r.a, r.b, r.c, r.ia, r.ib, r.ua, r.ub});
            mgl::glBindVertexArray(vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const GpuRow>(rows), mgl::BufferUsage::StreamDraw);
            const GLsizei stride = sizeof(GpuRow);
            for (GLuint i = 0; i < 3; ++i)
            {
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{i});
                mgl::glVertexAttribPointer(mgl::AttribLocation{i}, 4, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(i * 16));
            }
            for (GLuint i = 3; i < 7; ++i)
            {
                mgl::glEnableVertexAttribArray(mgl::AttribLocation{i});
                mgl::glVertexAttribIPointer(mgl::AttribLocation{i}, 4, i < 5 ? mgl::DataType::Int : mgl::DataType::UnsignedInt, stride,
                                            demo::buffer_offset(i * 16));
            }

            // --- capture ----------------------------------------------------------------------------
            const GLsizeiptr bytes = static_cast<GLsizeiptr>(rows.size() * 64);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tfb_);
            mgl::glBufferData(mgl::BufferTarget::TransformFeedback, bytes, nullptr, mgl::BufferUsage::StreamRead);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tfb_);
            mgl::glUseProgram(program);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, static_cast<GLsizei>(rows.size()));
            mgl::glEndTransformFeedback();
            mgl::glDisable(mgl::Capability::RasterizerDiscard);

            std::vector<std::uint32_t> raw(rows.size() * 16);
            void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::TransformFeedback, 0, bytes, mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                std::memcpy(raw.data(), mapped, static_cast<std::size_t>(bytes));
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::TransformFeedback);
            }
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program);
            if (mapped == nullptr)
            {
                check.expect(false, std::format("{}: transform feedback buffer mapped", cs.name));
                return;
            }

            // --- compare ----------------------------------------------------------------------------
            int bad = 0;
            double quality_err = 0.0;
            std::string first;
            for (std::size_t row = 0; row < cs.rows.size(); ++row)
            {
                const Out ref = cs.ref(cs.rows[row]);
                for (int k = 0; k < cs.components; ++k)
                {
                    const std::uint32_t bits = raw[row * 16 + static_cast<std::size_t>(k)];
                    const double r = ref[static_cast<std::size_t>(k)];
                    double got = 0;
                    bool ok = false;
                    if (cs.type == OutType::Uint) { got = bits; ok = got == r; }
                    else if (cs.type == OutType::Int) { got = static_cast<std::int32_t>(bits); ok = got == r; }
                    else
                    {
                        got = u2f(bits);
                        const double diff = std::abs(got - r);
                        const double x = cs.rows[row].a[static_cast<std::size_t>(k % 4)];
                        switch (cs.tol.kind)
                        {
                        case TolKind::Exact: ok = got == static_cast<double>(static_cast<float>(r)) || (std::isnan(r) && std::isnan(got)); break;
                        case TolKind::Ulp: ok = diff <= cs.tol.value * ulp_of(r); break;
                        case TolKind::ExpUlp: ok = diff <= (3.0 + 2.0 * std::abs(x)) * ulp_of(r); break;
                        case TolKind::Log: ok = (x >= 0.5 && x <= 2.0) ? diff < std::ldexp(1.0, -21) : diff <= 3.0 * ulp_of(r); break;
                        case TolKind::Abs:
                        case TolKind::QualityAbs: ok = diff <= cs.tol.value; quality_err = std::max(quality_err, diff); break;
                        case TolKind::Rel:
                        case TolKind::QualityRel:
                            ok = diff <= cs.tol.value * std::max(1.0, std::abs(r));
                            quality_err = std::max(quality_err, diff / std::max(1.0, std::abs(r)));
                            break;
                        }
                    }
                    if (!ok && bad++ == 0) first = std::format("row {} component {}: got {} expected {}", row, k, got, r);
                }
            }
            const bool quality = cs.tol.kind == TolKind::QualityAbs || cs.tol.kind == TolKind::QualityRel;
            check.expect(bad == 0,
                         std::format("{}{}", cs.name, quality ? std::format(" [precision undefined by spec: sanity bound {}, measured max error {:.2e}]", cs.tol.value, quality_err) : ""),
                         first);
        }

        std::vector<Case> cases_;
        mgl::ShaderId fs_{};
        mgl::VertexArrayId vao_{};
        mgl::BufferId vbo_{}, tfb_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "test-glsl300-builtins",
        .title = "GLSL ES 3.00 built-in functions (numeric)",
        .min_es = {3, 0},
        .summary = "Evaluates the GLSL ES 3.00 built-in functions in a vertex shader, captures the results with "
                   "transform feedback and compares them with CPU references using the §4.5.1 precision rules.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Glsl300Builtins>(argc, argv, info);
}
