// test-glsl-language
//
// Run-time checks of GLSL ES language rules (as opposed to built-in functions):
//
//   1. GLSL ES 1.00 shaders (attribute / varying / gl_FragColor / gl_FragData / texture2D /
//      texture2DProj / textureCube / texture2DLod... in the vertex shader / bias in the fragment
//      shader, and a shader with NO #version directive) render on an ES 3.x context; the picture is
//      compared with a CPU reference.
//   2. Linking shaders of different language versions fails (100 + 300 es; 310 es + older).
//   3. Language cases (operators, parameter passing, constant folding, scoping, comments/tokens)
//      are evaluated in BOTH shader languages: in a vertex shader captured with transform feedback,
//      and in a fragment shader writing four RGBA32UI colour attachments. Both paths deliver exact
//      32-bit values (floats as bit patterns), compared with CPU references.
//   4. gl_Max* built-in constants are at least the specified minimums (GLSL ES 3.00 §7.3; 3.10 §7.2 on
//      >= 3.1); gl_MaxVertexUniformVectors equals the API limit in GLSL ES 1.00 and 3.00 (§1.5). Equality
//      of the other constants with the API limits is printed, not asserted.
//   5. gl_DepthRange equals the clamped glDepthRangef values (GLSL ES 3.00 §7.4).
//   6. invariant / #pragma STDGL invariant(all): the same invariant output computed by two different
//      programs is bit-identical (GLSL ES 3.00 §4.6.1); a constant expression evaluates identically in
//      the vertex and the fragment language (§4.6.3).
//
// Inputs come from uniform arrays so that the compiler cannot fold the run-time expressions.
// Undefined results (integer division rounding, % with negative operands, out-of-range shifts, division
// by zero, copy-out order of aliased out parameters, ...) are never asserted; see README.md.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int kSlots = 16;  // four uvec4 outputs per case
    using Slots = std::array<std::uint32_t, kSlots>;

    // ---- run-time inputs: uniform vec4 uf[4], ivec4 ui[4], uvec4 uu[4] ---------------------------
    constexpr GLfloat kUF[16] = {1.5f, -2.25f, 3.0f, 0.1f,        // uf[0]
                                 16777216.0f, 1.0f, 0.5f, 0.0f,   // uf[1]
                                 7.0f, 2.0f, 1.0e30f, 3.0e-5f,    // uf[2]
                                 0.0f, 1.0f, 2.0f, 3.0f};         // uf[3]
    constexpr GLint kUI[16] = {7, -7, INT_MAX, INT_MIN,           // ui[0]
                               2, 3, 1, 12,                       // ui[1]
                               0, 1, -1, 5,                       // ui[2]
                               46341, 4, -12, 16};                // ui[3]
    constexpr GLuint kUU[16] = {7u, 0xFFFFFFFFu, 0x80000000u, 12u,  // uu[0]
                                2u, 3u, 1u, 0u,                     // uu[1]
                                0xF0F0F0F0u, 0xFFFFu, 5u, 10u,      // uu[2]
                                0u, 1u, 2u, 3u};                    // uu[3]

    double uf(int v, int c) { return static_cast<double>(kUF[v * 4 + c]); }

    // ---- expectations ------------------------------------------------------------------------------
    enum class Kind
    {
        Ignore,
        Bits,     // exact 32-bit pattern (integers, booleans as 0/1)
        Float,    // exact float; +0 and -0 are interchangeable (GLSL ES 3.00 §4.5.1)
        Rounded,  // "correctly rounded" with an undefined rounding mode: one of the two floats bracketing the exact value
        Ulp,      // within n ULP of the reference (precision table of §4.5.1)
        Quality,  // precision undefined by the spec: relative sanity bound only
    };

    struct Exp
    {
        Kind kind = Kind::Ignore;
        std::uint32_t bits = 0;
        double value = 0.0;
        double tol = 0.0;
    };

    Exp U(std::uint32_t v) { return {Kind::Bits, v, 0.0, 0.0}; }
    Exp I(std::int32_t v) { return {Kind::Bits, std::bit_cast<std::uint32_t>(v), 0.0, 0.0}; }
    Exp B(bool b) { return U(b ? 1u : 0u); }
    Exp F(float f) { return {Kind::Float, std::bit_cast<std::uint32_t>(f), f, 0.0}; }
    Exp R(double exact) { return {Kind::Rounded, 0, exact, 0.0}; }
    Exp Ulp(double ref, double n) { return {Kind::Ulp, 0, ref, n}; }
    Exp Q(double ref, double rel) { return {Kind::Quality, 0, ref, rel}; }

    double ulp_of(double r)
    {
        const double m = std::max(std::abs(r), std::ldexp(1.0, -126));
        return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(m))) - 23);
    }

    std::string show(const Exp& e, std::uint32_t got)
    {
        if (e.kind == Kind::Bits) return std::format("got 0x{:08X} ({}) expected 0x{:08X} ({})", got, static_cast<std::int32_t>(got), e.bits,
                                                     static_cast<std::int32_t>(e.bits));
        return std::format("got {:.9g} (0x{:08X}) expected {:.9g}", std::bit_cast<float>(got), got, e.value);
    }

    bool matches(const Exp& e, std::uint32_t got)
    {
        const float f = std::bit_cast<float>(got);
        switch (e.kind)
        {
        case Kind::Ignore: return true;
        case Kind::Bits: return got == e.bits;
        case Kind::Float: return f == std::bit_cast<float>(e.bits);  // == treats +0 and -0 as equal
        case Kind::Rounded:
        {
            const float rn = static_cast<float>(e.value);
            float lo = rn, hi = rn;
            if (static_cast<double>(rn) < e.value) hi = std::nextafter(rn, std::numeric_limits<float>::infinity());
            else if (static_cast<double>(rn) > e.value) lo = std::nextafter(rn, -std::numeric_limits<float>::infinity());
            return f == lo || f == hi;
        }
        case Kind::Ulp: return std::abs(static_cast<double>(f) - e.value) <= e.tol * ulp_of(e.value);
        case Kind::Quality: return std::abs(static_cast<double>(f) - e.value) <= e.tol * std::max(1.0, std::abs(e.value));
        }
        return false;
    }

    // ---- language cases ----------------------------------------------------------------------------
    struct Case
    {
        std::string name;
        std::string globals;  // declarations before main()
        std::string body;     // writes uvec4 r[0..3]; may use uf/ui/uu and fb()/ib()/bb()
        std::vector<Exp> expected;
        bool cross_stage_identical = false;  // GLSL ES 3.00 §4.6.3: VS and FS results must be bit-identical
        std::string mesa_deviation;          // documented Mesa deviation (strict check elsewhere, XFAIL on Mesa)
    };

    std::string preamble(std::string_view version)
    {
        return std::format(
            "#version {}\n"
            "precision highp float;\nprecision highp int;\n"
            "uniform vec4 uf[4];\nuniform ivec4 ui[4];\nuniform uvec4 uu[4];\n"
            "uint fb(float x) {{ return floatBitsToUint(x); }}\n"
            "uvec4 fb(vec4 x) {{ return floatBitsToUint(x); }}\n"
            "uint ib(int x) {{ return uint(x); }}\n"
            "uvec4 ib(ivec4 x) {{ return uvec4(x); }}\n"
            "uint bb(bool b) {{ return b ? 1u : 0u; }}\n",
            version);
    }

    std::string main_function(const Case& c, std::string_view store)
    {
        return std::format("{}\nvoid main()\n{{\n    uvec4 r[4];\n    r[0] = uvec4(0u); r[1] = uvec4(0u); r[2] = uvec4(0u); r[3] = uvec4(0u);\n"
                           "    {{\n{}\n    }}\n{}}}\n",
                           c.globals, c.body, store);
    }

    std::string vertex_case_source(const Case& c, std::string_view version)
    {
        return preamble(version) + "flat out uvec4 o0;\nflat out uvec4 o1;\nflat out uvec4 o2;\nflat out uvec4 o3;\n" +
               main_function(c, "    o0 = r[0]; o1 = r[1]; o2 = r[2]; o3 = r[3];\n    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n");
    }

    std::string fragment_case_source(const Case& c, std::string_view version)
    {
        return preamble(version) +
               "layout(location = 0) out uvec4 o0;\nlayout(location = 1) out uvec4 o1;\n"
               "layout(location = 2) out uvec4 o2;\nlayout(location = 3) out uvec4 o3;\n" +
               main_function(c, "    o0 = r[0]; o1 = r[1]; o2 = r[2]; o3 = r[3];\n");
    }

    std::vector<Case> language_cases()
    {
        constexpr std::uint32_t kIntMinBits = 0x80000000u;
        std::vector<Case> c;

        // ---------------- GLSL ES 3.00 §3.1, §3.6, §3.8: comments and tokens ------------------------
        c.push_back({"comments are one space; // comment continued by backslash-newline; token spliced by line continuation",
                     "",
                     "int a = 3/**/-/**/1;          // a comment between tokens is a single space\n"
                     "int b = 7; // trailing comment: r[0].w = 99u;\n"
                     "/* r[0].w = 98u; */\n"
                     "uint c = 1u;\n"
                     "// this comment ends with a backslash, so the next line belongs to it (phase 6 before phase 7) \\\n"
                     "c = 42u;\n"
                     "in\\\n"
                     "t spliced = 5;                // phase 6 joins 'in' and 't' into the keyword int\n"
                     "int my__value = 6;            // names containing __ are reserved but legal (§3.8)\n"
                     "r[0] = uvec4(ib(a), ib(b), c, ib(spliced));\n"
                     "r[1].x = ib(my__value);\n",
                     {I(2), I(7), U(1), I(5), I(6)}});

        // ---------------- GLSL ES 3.00 §5.9 / §4.5.1: float arithmetic ------------------------------
        const double a = uf(0, 3), b = uf(2, 3), cc = uf(0, 0), d = uf(0, 1), big = uf(1, 0);
        c.push_back({"float + - * correctly rounded, unary - and + (§5.9, §4.5.1 'Correctly rounded')",
                     "",
                     "float a = uf[0].w, b = uf[2].w, c = uf[0].x, d = uf[0].y, big = uf[1].x, one = uf[1].y;\n"
                     "r[0] = uvec4(fb(a + b), fb(a - b), fb(a * b), fb(-a));\n"
                     "r[1] = uvec4(fb(c + d), fb(c - d), fb(c * d), fb(+d));\n"
                     "r[2] = uvec4(fb(big + one), fb(big - one), fb(uf[2].z * uf[2].z), fb(b * b));\n",
                     {R(a + b), R(a - b), R(a * b), F(static_cast<float>(-a)),            //
                      R(cc + d), R(cc - d), R(cc * d), F(static_cast<float>(d)),          //
                      R(big + 1.0), R(big - 1.0), R(static_cast<double>(1.0e30f) * static_cast<double>(1.0e30f)),  // overflow: FLT_MAX or +Inf
                      R(b * b)}});
        c.push_back({"float division within 2.5 ULP (§4.5.1)",
                     "",
                     "r[0] = uvec4(fb(uf[0].x / uf[0].y), fb(1.0 / uf[2].w), fb(uf[2].x / uf[0].w), fb(uf[0].x / uf[3].z));\n",
                     {Ulp(cc / d, 2.5), Ulp(1.0 / b, 2.5), Ulp(uf(2, 0) / a, 2.5), Ulp(cc / 2.0, 2.5)}});

        // ---------------- §5.9 relational / equality / logical ---------------------------------------
        c.push_back({"relational and equality operators; ==/!= on vectors, structures and arrays (§5.9, §5.7)",
                     "struct S { int k; vec2 p; };\n",
                     "float a = uf[0].x, b = uf[0].z; int i = ui[0].x, j = ui[0].y; uint u = uu[0].x, v = uu[0].y;\n"
                     "r[0] = uvec4(bb(a < b), bb(a <= a), bb(a > b), bb(b >= a));\n"
                     "r[1] = uvec4(bb(i < j), bb(j <= i), bb(u > v), bb(v >= u));\n"
                     "r[2] = uvec4(bb(a == uf[0].x), bb(a != b), bb(i == -j), bb(u != v));\n"
                     "S s1 = S(i, vec2(a, b)); S s2 = S(7, vec2(1.5, 3.0)); S s3 = S(i, vec2(a, a));\n"
                     "float x1[3] = float[3](a, b, 0.5); float x2[3] = float[3](1.5, 3.0, uf[1].z);\n"
                     "r[3] = uvec4(bb(vec3(a, b, 1.0) == vec3(1.5, 3.0, uf[1].y)), bb(vec2(a) != vec2(a, b)),\n"
                     "             bb(s1 == s2 && s1 != s3), bb(x1 == x2 && x1 != float[3](a, b, 0.25)));\n",
                     {B(true), B(true), B(false), B(true),     // 1.5<3, 1.5<=1.5, 1.5>3, 3>=1.5
                      B(false), B(true), B(false), B(true),    // 7<-7, -7<=7, 7u>0xFFFFFFFFu (unsigned!), 0xFFFFFFFFu>=7u
                      B(true), B(true), B(true), B(true),      //
                      B(true), B(true), B(true), B(true)}});
        c.push_back({"logical && || ^^ ! truth tables (§5.9)",
                     "",
                     "bool t = ui[2].y == 1; bool f = ui[2].x == 1;\n"
                     "r[0] = uvec4(bb(t && t), bb(t && f), bb(f && t), bb(f && f));\n"
                     "r[1] = uvec4(bb(t || t), bb(t || f), bb(f || t), bb(f || f));\n"
                     "r[2] = uvec4(bb(t ^^ t), bb(t ^^ f), bb(f ^^ t), bb(f ^^ f));\n"
                     "r[3] = uvec4(bb(!t), bb(!f), bb(!!t), bb(t && !f || f));\n",
                     {B(true), B(false), B(false), B(false), B(true), B(true), B(true), B(false),  //
                      B(false), B(true), B(true), B(false), B(false), B(true), B(true), B(true)}});
        c.push_back({"&& and || short-circuit, ^^ evaluates both operands, ?: evaluates exactly one of its branches (§5.9)",
                     "",
                     "bool t = ui[2].y == 1; bool f = !t;\n"
                     "int n_and = 0, n_or = 0, n_xor = 0;\n"
                     "bool x1 = f && (++n_and > 0);\n"
                     "bool x2 = t && (++n_and > 0);\n"
                     "bool x3 = t || (++n_or > 0);\n"
                     "bool x4 = f || (++n_or > 0);\n"
                     "bool x5 = t ^^ (++n_xor > 0);\n"
                     "bool x6 = f ^^ (++n_xor > 0);\n"
                     "int p = 0, q = 0;\n"
                     "int z1 = t ? ++p : ++q;\n"
                     "int z2 = f ? ++p : (q += 5);\n"
                     "r[0] = uvec4(ib(n_and), ib(n_or), ib(n_xor), bb(x1) | bb(x2) << 1 | bb(x3) << 2 | bb(x4) << 3 | bb(x5) << 4 | bb(x6) << 5);\n"
                     "r[1] = uvec4(ib(p), ib(q), ib(z1), ib(z2));\n",
                     {I(1), I(1), I(2), U(0b101110), I(1), I(5), I(1), I(5)}});
        c.push_back({"sequence (comma) operator: left to right, value and type of the right-most operand (§5.9)",
                     "",
                     "int i = ui[1].x;\n"
                     "int k = (i += 3, i * 10);\n"
                     "int m = 0; int j; int q;\n"
                     "for (j = 0, q = 10; j < 3; j++, q--) m += q;\n"
                     "int last = (ui[1].y, ui[1].w);\n"
                     "vec2 v = (uf[0].x, vec2(uf[0].y));\n"
                     "r[0] = uvec4(ib(i), ib(k), ib(m), ib(last));\n"
                     "r[1] = uvec4(fb(v.x), fb(v.y), ib(j), ib(q));\n",
                     {I(5), I(50), I(27), I(12), F(-2.25f), F(-2.25f), I(3), I(7)}});
        c.push_back({"++/-- prefix and postfix on int, uint (wraps), float, vector and matrix (§5.9)",
                     "",
                     "int i = ui[3].w;\n"
                     "int a = i++; int b = ++i; int c = i--; int d = --i;\n"
                     "uint u = uu[1].w; uint e = u--;\n"
                     "float f = uf[0].x; float g = ++f;\n"
                     "vec2 v = vec2(uf[0].y); vec2 w = v++;\n"
                     "mat2 m = mat2(uf[3].z); m--;\n"
                     "r[0] = uvec4(ib(a), ib(b), ib(c), ib(d));\n"
                     "r[1] = uvec4(ib(i), e, u, fb(g));\n"
                     "r[2] = uvec4(fb(w.x), fb(v.y), fb(m[0][0]), fb(m[0][1]));\n"
                     "r[3] = uvec4(fb(m[1][0]), fb(m[1][1]), fb(f), 0u);\n",
                     {I(16), I(18), I(18), I(16), I(16), U(0), U(0xFFFFFFFFu), F(2.5f),  //
                      F(-2.25f), F(-1.25f), F(1.0f), F(-1.0f), F(-1.0f), F(1.0f), F(2.5f), U(0)}});
        c.push_back({"precedence and associativity of the §5.1 operator table",
                     "",
                     "int two = ui[1].x, three = ui[1].y, four = ui[3].y, one = ui[1].z;\n"
                     "bool t = one == 1; bool f = !t;\n"
                     "r[0] = uvec4(ib(two + three * four), ib((two + three) * four), ib(one << two + one), ib(6 ^ three & 5));\n"
                     "r[1] = uvec4(ib(10 - four - three), ib(64 / four / two), ib(-two * -three), ib(~one + 1));\n"
                     "r[2] = uvec4(bb(t || f && f), bb(!f == t), bb(two < three == t), ib(t ? 1 : t ? 2 : 3));\n"
                     "int p, q; p = q = three;\n"
                     "r[3] = uvec4(ib(f ? 1 : t ? 2 : 3), ib(p), ib(q), ib(two * three % four));\n",
                     {I(14), I(20), I(8), I(7), I(3), I(8), I(6), I(-1),  //
                      B(true), B(true), B(true), I(1), I(2), I(3), I(3), I(2)}});
        c.push_back({"assignment operators = += -= *= /= %= <<= >>= &= |= ^= on int, vector and matrix (§5.8)",
                     "",
                     "int i = ui[0].x;\n"
                     "i += 5; int s0 = i; i -= 2; int s1 = i; i *= 3; int s2 = i; i /= 6; int s3 = i; i %= 3; int s4 = i;\n"
                     "i <<= 4; int s5 = i; i >>= 2; int s6 = i; i &= 12; int s7 = i; i |= 3; int s8 = i; i ^= 6; int s9 = i;\n"
                     "vec3 v = vec3(uf[0].x); v *= 2.0; v += vec3(1.0, 2.0, 3.0); v *= uf[1].z;\n"
                     "mat2 m = mat2(uf[3].z); m *= mat2(1.0, 2.0, 3.0, 4.0);\n"
                     "r[0] = uvec4(ib(s0), ib(s1), ib(s2), ib(s3));\n"
                     "r[1] = uvec4(ib(s4), ib(s5), ib(s6), ib(s7));\n"
                     "r[2] = uvec4(ib(s8), ib(s9), fb(v.x), fb(v.y));\n"
                     "r[3] = uvec4(fb(v.z), fb(m[0][0]), fb(m[0][1]), fb(m[1][1]));\n",
                     {I(12), I(10), I(30), I(5), I(2), I(32), I(8), I(8), I(11), I(13), F(2.0f), F(2.5f), F(3.0f), F(2.0f), F(4.0f), F(8.0f)}});
        c.push_back({"integer wrap-around (§4.1.3) and exactly divisible / non-negative / and % (§5.9)",
                     "",
                     "int imax = ui[0].z, imin = ui[0].w, one = ui[1].z, big = ui[3].x;\n"
                     "uint umax = uu[0].y, uone = uu[1].z, uzero = uu[1].w;\n"
                     "r[0] = uvec4(ib(imax + one), ib(imin - one), ib(big * big), ib(-imin));\n"
                     "r[1] = uvec4(umax + uone, uzero - uone, umax * umax, uu[0].z * 2u);\n"
                     "int twelve = ui[1].w, four = ui[3].y;\n"
                     "r[2] = uvec4(ib(twelve / four), ib(-twelve / four), ib(twelve / -four), ib(ui[0].x % ui[1].y));\n"
                     "r[3] = uvec4(umax / 5u, umax % 10u, uu[0].w / uu[1].y, ib(0 % ui[2].w));\n",
                     {U(kIntMinBits), I(INT_MAX), I(-2147479015), U(kIntMinBits),  //
                      U(0), U(0xFFFFFFFFu), U(1), U(0),                            //
                      I(3), I(-3), I(-3), I(1),                                    //
                      U(858993459u), U(5), U(4), I(0)}});
        c.push_back({"scalar-vector and component-wise arithmetic (§5.9, §5.10)",
                     "",
                     "vec4 a = uf[0]; float s = uf[1].z; ivec4 ia = ui[1]; int is = ui[1].x;\n"
                     "r[0] = fb(a * s); r[1] = fb(s - a); r[2] = ib(ia * is + 1); r[3] = ib(ia * ia);\n",
                     {R(uf(0, 0) * 0.5), R(uf(0, 1) * 0.5), R(uf(0, 2) * 0.5), R(uf(0, 3) * 0.5),                    //
                      R(0.5 - uf(0, 0)), R(0.5 - uf(0, 1)), R(0.5 - uf(0, 2)), R(0.5 - uf(0, 3)),  //
                      I(5), I(7), I(3), I(25), I(4), I(9), I(1), I(144)}});
        c.push_back({"the left side of an assignment is evaluated before the right side: x[i++] = i (§5.8)",
                     "",
                     "int x[3] = int[3](0, 0, 0); int i = ui[2].x;\n"
                     "x[i++] = i;\n"
                     "r[0] = uvec4(ib(x[0]), ib(x[1]), ib(i), 0u);\n",
                     {I(1), I(0), I(1)}});
        c.push_back({"the left side of an assignment is evaluated before the right side: y[j] = ++j (§5.8)",
                     "",
                     "int y[3] = int[3](0, 0, 0); int j = ui[2].x;\n"
                     "y[j] = ++j;\n"
                     "r[0] = uvec4(ib(y[0]), ib(y[1]), ib(j), 0u);\n",
                     {I(1), I(0), I(1)},
                     false,
                     "Mesa reads the index of y[j] after evaluating the right-hand side ++j (stores into y[1])"});

        // ---------------- GLSL ES 3.00 §6.1.1: parameter qualifiers ----------------------------------
        c.push_back({"in / out / inout copy semantics incl. arrays and structures (§6.1.1)",
                     "struct P { int k; };\n"
                     "float set_in(float x) { x = 99.0; return x; }\n"
                     "void set_out(out float y) { y = 4.0; }\n"
                     "void twice_plus_one(inout vec2 v) { v = v * 2.0 + 1.0; }\n"
                     "void no_write_back(in ivec2 v) { v.x = 100; }\n"
                     "void fill(out float a[2]) { a[0] = 5.0; a[1] = 6.0; }\n"
                     "void clear_copy(float a[2]) { a[0] = 0.0; }\n"
                     "void bump(inout P p) { p.k += 10; }\n",
                     "float a = uf[0].x; float ra = set_in(a);\n"
                     "float b = uf[3].x; set_out(b);\n"
                     "vec2 v = vec2(uf[0].x, uf[0].y); twice_plus_one(v);\n"
                     "ivec2 iv = ui[1].xy; no_write_back(iv);\n"
                     "float arr[2]; fill(arr);\n"
                     "float brr[2] = float[2](uf[0].x, 2.0); clear_copy(brr);\n"
                     "P p = P(ui[1].w); bump(p);\n"
                     "r[0] = uvec4(fb(a), fb(ra), fb(b), fb(v.x));\n"
                     "r[1] = uvec4(fb(v.y), ib(iv.x), ib(iv.y), fb(arr[0]));\n"
                     "r[2] = uvec4(fb(arr[1]), fb(brr[0]), fb(brr[1]), ib(p.k));\n",
                     {F(1.5f), F(99.0f), F(4.0f), F(4.0f), F(-3.5f), I(2), I(3), F(5.0f), F(6.0f), F(1.5f), F(2.0f), I(22)}});
        c.push_back({"aliasing: parameters are copies taken at call time (§6.1.1); aliased copy-out order is undefined and not asserted",
                     "float in_then_out(in float a, out float b) { b = 10.0; return a; }\n"
                     "float inout_pair(inout float a, inout float b) { a += 1.0; return a + b; }\n"
                     "int write_then_read(inout int a[2], in int b[2]) { a[0] = 50; return b[0]; }\n",
                     "float x = uf[0].z; float rx = in_then_out(x, x);\n"
                     "float y = uf[0].z; float ry = inout_pair(y, y);\n"
                     "int arr[2] = int[2](ui[1].x, ui[1].y); int rb = write_then_read(arr, arr);\n"
                     "r[0] = uvec4(fb(rx), fb(x), fb(ry), ib(rb));\n"
                     "r[1] = uvec4(ib(arr[0]), ib(arr[1]), 0u, 0u);\n",
                     {F(3.0f), F(10.0f), F(7.0f), I(2), I(50), I(3)}});
        c.push_back({"arguments are evaluated left to right, exactly once; an inout l-value with a side effect is evaluated once (§6.1.1)",
                     "int g_calls = 0;\n"
                     "int next_call() { g_calls++; return g_calls; }\n"
                     "int digits(int a, int b, int c) { return a * 100 + b * 10 + c; }\n"
                     "void add_ten(inout int v) { v += 10; }\n",
                     "int i = ui[1].z;\n"
                     "int d = digits(i++, i++, i++);\n"
                     "int e = digits(next_call(), next_call(), next_call());\n"
                     "int arr[3] = int[3](1, 2, 3); int j = ui[2].x;\n"
                     "add_ten(arr[j++]);\n"
                     "r[0] = uvec4(ib(d), ib(i), ib(e), ib(g_calls));\n"
                     "r[1] = uvec4(ib(arr[0]), ib(arr[1]), ib(arr[2]), ib(j));\n",
                     {I(123), I(4), I(123), I(3), I(11), I(2), I(3), I(1)}});
        c.push_back({"the l-value of an out argument is fixed when the argument is evaluated: store(arr[j], ++j) (§6.1.1)",
                     "void store(out int dst, int v) { dst = v; }\n",
                     "int arr[3] = int[3](0, 0, 0); int j = ui[2].x;\n"
                     "store(arr[j], ++j);\n"
                     "r[0] = uvec4(ib(arr[0]), ib(arr[1]), ib(j), 0u);\n",
                     {I(1), I(0), I(1)},
                     false,
                     "Mesa evaluates the index of the out argument arr[j] at copy-out time, after ++j (stores into arr[1])"});

        // ---------------- GLSL ES 3.00 §4.3.2 / §4.3.3: constant expressions -------------------------
        c.push_back({"constant folding equals run-time evaluation for exactly defined operations (§4.3.3, §4.1.3, §5.9)",
                     "const int kWrap = 2147483647 + 1;\n"
                     "const uint kUWrap = 0u - 1u;\n"
                     "const int kDiv = -12 / 4;\n"
                     "const int kMod = 7 % 3;\n"
                     "const float kMul = 1.5 * 2.25;\n"
                     "const int kShift = -16 >> 2;\n"
                     "const uint kBits = (0xF0F0F0F0u ^ 0xFFFFu) & ~0x0F00u;\n"
                     "const float kBuiltins = floor(2.75) + abs(-0.5) + max(1.0, 3.0) + clamp(9.0, 0.0, 4.0) + fract(2.75);\n",
                     "float t = uf[0].x + 1.25;\n"
                     "r[0] = uvec4(ib(kWrap), ib(ui[0].z + ui[1].z), kUWrap, uu[1].w - uu[1].z);\n"
                     "r[1] = uvec4(ib(kDiv), ib(-ui[1].w / ui[3].y), ib(kMod), ib(ui[0].x % ui[1].y));\n"
                     "r[2] = uvec4(fb(kMul), fb(uf[0].x * 2.25), ib(kShift), ib(-ui[3].w >> 2));\n"
                     "r[3] = uvec4(kBits, (uu[2].x ^ uu[2].y) & ~0x0F00u, fb(kBuiltins),\n"
                     "             fb(floor(t) + abs(-uf[1].z) + max(uf[1].y, uf[0].z) + clamp(9.0 * uf[1].y, uf[3].x, 4.0) + fract(t)));\n",
                     {U(kIntMinBits), U(kIntMinBits), U(0xFFFFFFFFu), U(0xFFFFFFFFu), I(-3), I(-3), I(1), I(1),  //
                      F(3.375f), F(3.375f), I(-4), I(-4), U(0xF0F0000Fu), U(0xF0F0000Fu), F(10.25f), F(10.25f)}});
        c.push_back({"constant expressions: built-in calls, constructors, matrix product, structure field, array element, length() (§4.3.3)",
                     "struct S { int a; float b; };\n"
                     "const S kS = S(3, 4.5);\n"
                     "const float kArr[3] = float[3](1.0, 2.0, 3.0);\n"
                     "const uint kPack = packHalf2x16(vec2(1.0, -2.0));\n"
                     "const uint kOneBits = floatBitsToUint(1.0);\n"
                     "const mat2 kMat = mat2(1.0, 2.0, 3.0, 4.0) * mat2(0.0, 1.0, 1.0, 0.0);\n"
                     "float g_data[5];\n",
                     "const int kLocal = kS.a * 2;\n"
                     "const int kLen = g_data.length() + kArr.length();\n"
                     "mat2 m = mat2(uf[1].y, uf[3].z, uf[0].z, 4.0) * mat2(0.0, uf[1].y, 1.0, 0.0);\n"
                     "r[0] = uvec4(kPack, packHalf2x16(vec2(uf[1].y, -uf[3].z)), kOneBits, floatBitsToUint(uf[1].y));\n"
                     "r[1] = uvec4(fb(kMat[0][0]), fb(kMat[0][1]), fb(kMat[1][0]), fb(kMat[1][1]));\n"
                     "r[2] = uvec4(fb(m[0][0]), fb(m[0][1]), fb(m[1][0]), fb(m[1][1]));\n"
                     "r[3] = uvec4(ib(kLocal), fb(kS.b), fb(kArr[2]), ib(kLen));\n",
                     {U(0xC0003C00u), U(0xC0003C00u), U(0x3F800000u), U(0x3F800000u),  //
                      F(3.0f), F(4.0f), F(1.0f), F(2.0f), F(3.0f), F(4.0f), F(1.0f), F(2.0f),  //
                      I(6), F(4.5f), F(3.0f), I(8)}});
        {
            // §4.6.3: the same constant expression gives the same value everywhere. sin/exp/pow have no
            // defined precision (§4.5.1), so only a sanity bound against the exact value is asserted;
            // the invariance itself (slot 0 == slot 1, vertex == fragment) is checked bit-exactly.
            const double k = std::sin(1.0) * std::exp(0.5) + std::sqrt(2.0) / 3.0;
            const double n = 1.0 / std::sqrt(9.0 + 0.49);
            const double p = std::pow(2.0, 0.3);
            c.push_back({"a constant expression evaluates identically in one shader, in two shaders and in both languages (§4.6.3)",
                         "const highp float kA = sin(1.0) * exp(0.5) + sqrt(2.0) / 3.0;\n",
                         "const highp float kB = sin(1.0) * exp(0.5) + sqrt(2.0) / 3.0;\n"
                         "const highp vec2 kC = normalize(vec2(3.0, 0.7)) * pow(2.0, 0.3);\n"
                         "r[0] = uvec4(fb(kA), fb(kB), fb(kC.x), fb(kC.y));\n",
                         {Q(k, 1e-3), Q(k, 1e-3), Q(3.0 * n * p, 1e-3), Q(0.7 * n * p, 1e-3)},
                         true});
        }

        // ---------------- GLSL ES 3.00 §4.2: scoping ------------------------------------------------
        c.push_back({"scoping: nesting, hiding, initializer scope, do-while body scope, for-init scope, struct/variable names (§4.2)",
                     "struct S { int x; };\n"
                     "int shadow_param(int k) { { int k = k + 3; return k; } return -1; }\n",
                     "int x = ui[1].z;\n"
                     "int y1, y2, y3, y4;\n"
                     "{ int x = 2, y = x; y1 = y; }\n"
                     "{ int x = x + 10; y2 = x; }\n"
                     "y3 = x;\n"
                     "int i = ui[2].x; int n = 0;\n"
                     "do { int i = 5; n += i; } while (i++ < 3);\n"
                     "int k = 100;\n"
                     "for (int k = 0; k < 3; k++) { n += k; }\n"
                     "{ S S = S(ui[1].y); y4 = S.x; }\n"
                     "float hidden;\n"
                     "{ vec4 uf = vec4(9.0); hidden = uf.x; }\n"
                     "r[0] = uvec4(ib(y1), ib(y2), ib(y3), ib(shadow_param(ui[1].x)));\n"
                     "r[1] = uvec4(ib(n), ib(i), ib(k), ib(y4));\n"
                     "r[2] = uvec4(fb(hidden), fb(uf[0].x), 0u, 0u);\n",
                     {I(2), I(11), I(1), I(5), I(23), I(4), I(100), I(3), F(9.0f), F(1.5f)}});
        return c;
    }

    // ---- GLSL ES 1.00 picture -----------------------------------------------------------------------
    struct Rect
    {
        int x, y, w, h;
    };

    Rect sub_rect(Rect r, int i, int cols, int rows)
    {
        const int cx = i % cols, cy = i / cols;
        const int x0 = r.x + r.w * cx / cols, x1 = r.x + r.w * (cx + 1) / cols;
        const int y0 = r.y + r.h * cy / rows, y1 = r.y + r.h * (cy + 1) / rows;
        return {x0, y0, x1 - x0, y1 - y0};
    }

    constexpr int kTex = 4;  // 4x4 texture for texture2D / texture2DProj
    demo::Rgba8 texel(int i, int j)
    {
        return {static_cast<std::uint8_t>(20 + i * 60), static_cast<std::uint8_t>(20 + j * 60),
                static_cast<std::uint8_t>(255 - (i + j) * 25), 255};
    }
    // Cube faces in GL order +X -X +Y -Y +Z -Z.
    constexpr std::array<demo::Rgba8, 6> kFace = {{{200, 40, 40, 255}, {40, 200, 40, 255}, {40, 40, 200, 255},
                                                   {200, 200, 40, 255}, {200, 40, 200, 255}, {40, 200, 200, 255}}};
    constexpr std::array<std::array<float, 3>, 6> kFaceDir = {{{1.0f, 0.2f, -0.3f}, {-1.0f, -0.4f, 0.1f}, {0.3f, 1.0f, 0.2f},
                                                               {-0.2f, -1.0f, 0.4f}, {0.1f, 0.3f, 1.0f}, {-0.4f, 0.2f, -1.0f}}};
    // One solid colour per mipmap level of the 8x8 texture.
    constexpr std::array<demo::Rgba8, 4> kLevel = {{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}}};

    struct LodDraw
    {
        int mode;
        float lod;
        demo::Rgba8 expected;
    };
    // texture2DLod lod 0/1/3, texture2DProjLod (vec3) lod 2, texture2DProjLod (vec4) lod 1, textureCubeLod (-Z face).
    const std::array<LodDraw, 6> kLodDraws = {{{0, 0.0f, kLevel[0]}, {0, 1.0f, kLevel[1]}, {0, 3.0f, kLevel[3]},
                                               {1, 2.0f, kLevel[2]}, {2, 1.0f, kLevel[1]}, {3, 0.0f, kFace[5]}}};
    // texture2D(s, uv, bias) with lambda_base = -1: bias 0 -> magnification (level 0), bias 2 -> lambda 1 (level 1).
    constexpr std::array<float, 2> kBias = {0.0f, 2.0f};

    class GlslLanguage final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 144 || env.height < 96) throw demo::Skip("the GLSL ES 1.00 picture needs at least 144x96 pixels");
            es31_ = env.context_version >= demo::Version{3, 1};

            // GLSL ES 1.00 programs (attribute locations bound before linking: ES 3.0 §2.12.5).
            grad_ = build100(env, "shaders/gradient100.vert", "shaders/gradient100.frag");
            noversion_ = build100(env, "shaders/noversion.vert", "shaders/noversion.frag");
            cube_prog_ = build100(env, "shaders/cube100.vert", "shaders/cube100.frag");
            lod_ = build100(env, "shaders/lod100.vert", "shaders/lod100.frag");
            bias_ = build100(env, "shaders/bias100.vert", "shaders/bias100.frag");

            // Quad: position, uv, colour = (u, v, 1 - u) - affine, so both triangles interpolate one plane.
            const std::array<GLfloat, 28> quad = {-1, -1, 0, 0, 0, 0, 1,  //
                                                  1,  -1, 1, 0, 1, 0, 0,  //
                                                  -1, 1,  0, 1, 0, 1, 1,  //
                                                  1,  1,  1, 1, 1, 1, 0};
            mgl::glGenVertexArrays(1, &quad_vao_);
            mgl::glBindVertexArray(quad_vao_);
            mgl::glGenBuffers(1, &quad_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const GLfloat>(quad), mgl::BufferUsage::StaticDraw);
            const GLsizei stride = 7 * sizeof(GLfloat);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0}, 2, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1}, 2, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(8));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2}, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(16));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            // Textures: unit 0 = 4x4 RGBA8, unit 1 = cube map with 1x1 faces, unit 2 = 8x8 with 4 solid levels.
            mgl::glGenTextures(1, &tex4_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex4_);
            std::vector<std::uint8_t> pixels;
            for (int j = 0; j < kTex; ++j)
                for (int i = 0; i < kTex; ++i)
                    for (const auto ch : texel(i, j)) pixels.push_back(ch);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, kTex, kTex, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, pixels.data());
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);

            mgl::glGenTextures(1, &cube_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube_);
            constexpr std::array<mgl::TextureTarget, 6> faces = {
                mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
                mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
                mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
            for (std::size_t f = 0; f < faces.size(); ++f)
                mgl::glTexImage2D(faces[f], 0, mgl::InternalFormat::Rgba, 1, 1, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                  kFace[f].data());
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Nearest);

            mgl::glGenTextures(1, &mip_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture2);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mip_);
            for (int level = 0; level < 4; ++level)
            {
                const int size = 8 >> level;
                std::vector<std::uint8_t> solid;
                for (int k = 0; k < size * size; ++k)
                    for (const auto ch : kLevel[static_cast<std::size_t>(level)]) solid.push_back(ch);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgba, size, size, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, solid.data());
            }
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::NearestMipmapNearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            // Transform feedback capture (vertex cases) and an empty VAO for gl_VertexID-only draws.
            mgl::glGenVertexArrays(1, &empty_vao_);
            mgl::glGenBuffers(1, &tfb_);
            mgl::glGenBuffers(1, &inv_vbo_);

            // 1x1 framebuffer with four RGBA32UI attachments (fragment cases). RGBA32UI is colour-renderable
            // in ES 3.0 (Table 3.13); MAX_DRAW_BUFFERS and MAX_COLOR_ATTACHMENTS are at least 4.
            mgl::glGenRenderbuffers(4, rb_.data());
            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            for (std::size_t i = 0; i < 4; ++i)
            {
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_[i]);
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba32UI, 1, 1);
                mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(kAttach[i]),
                                               mgl::RenderbufferTarget::Renderbuffer, rb_[i]);
            }
            const std::array<mgl::DrawBuffer, 4> draw = {mgl::to_draw_buffer(kAttach[0]), mgl::to_draw_buffer(kAttach[1]),
                                                         mgl::to_draw_buffer(kAttach[2]), mgl::to_draw_buffer(kAttach[3])};
            mgl::glDrawBuffers(draw);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("RGBA32UI framebuffer incomplete: {}", demo::framebuffer_status_name(status)));
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glBindVertexArray(quad_vao_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex4_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture2);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mip_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            // Cell 0: attribute + varying gradient, gl_FragColor.
            viewport(cell(env, 0));
            mgl::glUseProgram(grad_);
            draw_quad();

            // Cell 1: no #version directive, texture2D, gl_FragData[0].
            mgl::glUseProgram(noversion_);
            mgl::glUniform1i(mgl::glGetUniformLocation(noversion_, "u_tex"), 0);
            mgl::glUniform1f(mgl::glGetUniformLocation(noversion_, "u_q"), kQ);
            mgl::glUniform1i(mgl::glGetUniformLocation(noversion_, "u_mode"), 0);
            viewport(cell(env, 1));
            draw_quad();
            // Cell 2: texture2DProj, vec3 form (left half) and vec4 form (right half).
            for (int half = 0; half < 2; ++half)
            {
                mgl::glUniform1i(mgl::glGetUniformLocation(noversion_, "u_mode"), 1 + half);
                viewport(sub_rect(cell(env, 2), half, 2, 1));
                draw_quad();
            }

            // Cell 3: textureCube, one sub-rectangle per face.
            mgl::glUseProgram(cube_prog_);
            mgl::glUniform1i(mgl::glGetUniformLocation(cube_prog_, "u_cube"), 1);
            for (int f = 0; f < 6; ++f)
            {
                const auto& d = kFaceDir[static_cast<std::size_t>(f)];
                mgl::glUniform3f(mgl::glGetUniformLocation(cube_prog_, "u_dir"), d[0], d[1], d[2]);
                viewport(sub_rect(cell(env, 3), f, 3, 2));
                draw_quad();
            }

            // Cell 4: texture2DLod / texture2DProjLod / textureCubeLod in the vertex shader.
            mgl::glUseProgram(lod_);
            mgl::glUniform1i(mgl::glGetUniformLocation(lod_, "u_mip"), 2);
            mgl::glUniform1i(mgl::glGetUniformLocation(lod_, "u_cube"), 1);
            for (std::size_t k = 0; k < kLodDraws.size(); ++k)
            {
                mgl::glUniform1i(mgl::glGetUniformLocation(lod_, "u_mode"), kLodDraws[k].mode);
                mgl::glUniform1f(mgl::glGetUniformLocation(lod_, "u_lod"), kLodDraws[k].lod);
                viewport(sub_rect(cell(env, 4), static_cast<int>(k), 3, 2));
                draw_quad();
            }

            // Cell 5: texture2D with a bias argument in the fragment shader.
            mgl::glUseProgram(bias_);
            mgl::glUniform1i(mgl::glGetUniformLocation(bias_, "u_mip"), 2);
            for (std::size_t k = 0; k < kBias.size(); ++k)
            {
                const Rect r = sub_rect(cell(env, 5), static_cast<int>(k), 2, 1);
                // One pixel = 1/16 of the texture in s and t, i.e. 0.5 base-level texels: lambda_base = -1.
                mgl::glUniform2f(mgl::glGetUniformLocation(bias_, "u_uv_scale"), static_cast<float>(r.w) / 16.0f, static_cast<float>(r.h) / 16.0f);
                mgl::glUniform1f(mgl::glGetUniformLocation(bias_, "u_bias"), kBias[k]);
                viewport(r);
                draw_quad();
            }
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            validate_glsl100_picture(env, check);
            validate_version_mixing(check);
            comparator_negative_control(check);
            for (const auto& c : language_cases()) run_case(c, "300 es", check);
            validate_builtin_constants(check);
            validate_depth_range(check);
            validate_invariance(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            for (const auto p : {grad_, noversion_, cube_prog_, lod_, bias_}) mgl::glDeleteProgram(p);
            mgl::glDeleteVertexArrays(1, &quad_vao_);
            mgl::glDeleteVertexArrays(1, &empty_vao_);
            mgl::glDeleteBuffers(1, &quad_vbo_);
            mgl::glDeleteBuffers(1, &tfb_);
            mgl::glDeleteBuffers(1, &inv_vbo_);
            mgl::glDeleteTextures(1, &tex4_);
            mgl::glDeleteTextures(1, &cube_);
            mgl::glDeleteTextures(1, &mip_);
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteRenderbuffers(4, rb_.data());
        }

    private:
        static constexpr float kQ = 2.5f;  // projective q for texture2DProj
        static constexpr std::array<mgl::ColorAttachment, 4> kAttach = {mgl::ColorAttachment::Color0, mgl::ColorAttachment::Color1,
                                                                        mgl::ColorAttachment::Color2, mgl::ColorAttachment::Color3};

        static Rect cell(const demo::Env& env, int i) { return sub_rect({0, 0, env.width, env.height}, i, 3, 2); }
        static void viewport(Rect r) { mgl::glViewport(r.x, r.y, r.w, r.h); }
        static void draw_quad() { mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4); }

        // ---- shader building ------------------------------------------------------------------------
        static mgl::ProgramId build100(const demo::Env& env, std::string_view vs_file, std::string_view fs_file)
        {
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, env.load_text(vs_file), log);
            if (vs.value == 0) throw demo::Failure(std::format("{}: {}", vs_file, log));
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, env.load_text(fs_file), log);
            if (fs.value == 0) throw demo::Failure(std::format("{}: {}", fs_file, log));
            const mgl::ProgramId p = mgl::glCreateProgram();
            mgl::glAttachShader(p, vs);
            mgl::glAttachShader(p, fs);
            mgl::glBindAttribLocation(p, mgl::AttribLocation{0}, "a_position");
            mgl::glBindAttribLocation(p, mgl::AttribLocation{1}, "a_uv");
            mgl::glBindAttribLocation(p, mgl::AttribLocation{2}, "a_color");
            mgl::glLinkProgram(p);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            GLint ok = GL_FALSE;
            mgl::glGetProgramiv(p, mgl::ProgramParameter::LinkStatus, &ok);
            if (ok != GL_TRUE) throw demo::Failure(std::format("{} + {}: {}", vs_file, fs_file, demo::program_info_log(p)));
            return p;
        }

        struct Built
        {
            mgl::ProgramId program{};
            std::string log;
        };

        /// Compiles vs + fs and links; with `tf` the outputs o0..o3 (or `tf_names`) are captured.
        static Built build(const std::string& vs_src, const std::string& fs_src, std::span<const GLchar* const> tf_names = {})
        {
            Built b;
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, vs_src, log);
            if (vs.value == 0)
            {
                b.log = "vertex shader: " + log;
                return b;
            }
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, fs_src, log);
            if (fs.value == 0)
            {
                mgl::glDeleteShader(vs);
                b.log = "fragment shader: " + log;
                return b;
            }
            const mgl::ProgramId p = mgl::glCreateProgram();
            mgl::glAttachShader(p, vs);
            mgl::glAttachShader(p, fs);
            if (!tf_names.empty())
                mgl::glTransformFeedbackVaryings(p, static_cast<GLsizei>(tf_names.size()), tf_names.data(),
                                                 mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(p);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            GLint ok = GL_FALSE;
            mgl::glGetProgramiv(p, mgl::ProgramParameter::LinkStatus, &ok);
            if (ok != GL_TRUE)
            {
                b.log = "link: " + demo::program_info_log(p);
                mgl::glDeleteProgram(p);
                return b;
            }
            b.program = p;
            return b;
        }

        static std::string trivial_fs(std::string_view version)
        {
            return std::format("#version {}\nprecision mediump float;\nout vec4 c;\nvoid main() {{ c = vec4(0.0); }}\n", version);
        }
        static std::string vertex_id_triangle_vs(std::string_view version)
        {
            // gl_VertexID 0, 1, 2 -> (-1,-1), (3,-1), (-1,3): one triangle covering the viewport.
            return std::format("#version {}\nvoid main() {{ gl_Position = vec4(float((gl_VertexID & 1) << 2) - 1.0, "
                               "float((gl_VertexID & 2) << 1) - 1.0, 0.0, 1.0); }}\n",
                               version);
        }

        static void set_inputs(mgl::ProgramId p)
        {
            mgl::glUniform4fv(mgl::glGetUniformLocation(p, "uf"), 4, kUF);
            mgl::glUniform4iv(mgl::glGetUniformLocation(p, "ui"), 4, kUI);
            mgl::glUniform4uiv(mgl::glGetUniformLocation(p, "uu"), 4, kUU);
        }

        /// Draws `vertices` points with RASTERIZER_DISCARD and returns the captured interleaved outputs.
        std::vector<std::uint32_t> capture(mgl::ProgramId p, int vertices, int words_per_vertex, mgl::VertexArrayId vao)
        {
            const GLsizeiptr bytes = static_cast<GLsizeiptr>(vertices) * words_per_vertex * 4;
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tfb_);
            mgl::glBufferData(mgl::BufferTarget::TransformFeedback, bytes, nullptr, mgl::BufferUsage::StreamRead);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tfb_);
            mgl::glBindVertexArray(vao);
            mgl::glUseProgram(p);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Points);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, vertices);
            mgl::glEndTransformFeedback();
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            std::vector<std::uint32_t> out(static_cast<std::size_t>(vertices * words_per_vertex), 0xCDCDCDCDu);
            void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::TransformFeedback, 0, bytes, mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                std::memcpy(out.data(), mapped, static_cast<std::size_t>(bytes));
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::TransformFeedback);
            }
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            return out;
        }

        Slots run_vertex(mgl::ProgramId p)
        {
            mgl::glUseProgram(p);
            set_inputs(p);
            const auto v = capture(p, 1, kSlots, empty_vao_);
            Slots s{};
            std::copy(v.begin(), v.end(), s.begin());
            return s;
        }

        Slots run_fragment(mgl::ProgramId p)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glViewport(0, 0, 1, 1);
            const std::array<GLuint, 4> sentinel = {0xCDCDCDCDu, 0xCDCDCDCDu, 0xCDCDCDCDu, 0xCDCDCDCDu};
            for (GLint i = 0; i < 4; ++i) mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, i, sentinel.data());
            mgl::glUseProgram(p);
            set_inputs(p);
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            Slots s{};
            for (std::size_t i = 0; i < 4; ++i)
            {
                mgl::glReadBuffer(mgl::to_read_buffer(kAttach[i]));
                mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, s.data() + 4 * i);
            }
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[0]));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return s;
        }

        /// Builds the case for one language, runs it and returns the 16 output words (nullopt-like flag on failure).
        bool run_stage(const Case& c, std::string_view version, bool fragment, Slots& out, demo::Check& check)
        {
            static constexpr std::array<const GLchar*, 4> kOut = {"o0", "o1", "o2", "o3"};
            const Built b = fragment ? build(vertex_id_triangle_vs(version), fragment_case_source(c, version))
                                     : build(vertex_case_source(c, version), trivial_fs(version), kOut);
            if (b.program.value == 0)
            {
                check.expect(false, std::format("[{}] {}: builds", fragment ? "fragment" : "vertex", c.name), b.log);
                return false;
            }
            out = fragment ? run_fragment(b.program) : run_vertex(b.program);
            mgl::glDeleteProgram(b.program);
            return true;
        }

        void compare(const Case& c, bool fragment, const Slots& got, demo::Check& check)
        {
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < c.expected.size(); ++k)
                if (!matches(c.expected[k], got[k]) && bad++ == 0)
                    first = std::format("slot {} (r[{}].{}): {}", k, k / 4, "xyzw"[k % 4], show(c.expected[k], got[k]));
            const std::string what = std::format("[{}] {}", fragment ? "fragment" : "vertex", c.name);
            const std::string detail = bad == 0 ? std::string{} : std::format("{} mismatches; first: {}", bad, first);
            if (c.mesa_deviation.empty()) check.expect(bad == 0, what, detail);
            else check.expect_or_known_deviation(bad == 0, what, detail, "Mesa", c.mesa_deviation);
        }

        void run_case(const Case& c, std::string_view version, demo::Check& check)
        {
            Slots vs{}, fs{};
            const bool vs_ok = run_stage(c, version, false, vs, check);
            const bool fs_ok = run_stage(c, version, true, fs, check);
            if (vs_ok) compare(c, false, vs, check);
            if (fs_ok) compare(c, true, fs, check);
            if (c.cross_stage_identical && vs_ok && fs_ok)
            {
                bool same = true;
                for (std::size_t k = 0; k < c.expected.size(); ++k) same = same && vs[k] == fs[k];
                check.expect(same && vs[0] == vs[1] && fs[0] == fs[1],
                             "constant expression bit-identical: twice in one shader, and in the vertex vs the fragment language (§4.6.3)",
                             std::format("vertex 0x{:08X}/0x{:08X} fragment 0x{:08X}/0x{:08X}", vs[0], vs[1], fs[0], fs[1]));
            }
        }

        // ---- 1. GLSL ES 1.00 picture -----------------------------------------------------------------
        void probe(demo::Check& check, std::string_view what, const std::vector<std::array<int, 2>>& points,
                   const std::vector<demo::Rgba8>& expected, int tolerance)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            int bad = 0;
            std::string first;
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                const demo::Rgba8 got = demo::read_pixel(points[i][0], points[i][1]);
                bool ok = true;
                for (std::size_t ch = 0; ch < 4; ++ch) ok = ok && std::abs(int(got[ch]) - int(expected[i][ch])) <= tolerance;
                if (!ok && bad++ == 0)
                    first = std::format("({}, {}) got {} expected {}", points[i][0], points[i][1], demo::format_rgba(got),
                                        demo::format_rgba(expected[i]));
            }
            check.expect(bad == 0, std::format("{} ({} probes, tolerance {})", what, points.size(), tolerance),
                         bad == 0 ? std::string{} : std::format("{} bad; first {}", bad, first));
        }

        void validate_glsl100_picture(const demo::Env& env, demo::Check& check)
        {
            // Cell 0: v_color = (u, v, 1 - u) interpolated; mediump varyings (GLSL ES 1.00 §4.5.2, relative 2^-10
            // = 0.25 of an 8-bit step) plus RGBA8 quantization (±1) -> tolerance 2.
            {
                const Rect r = cell(env, 0);
                std::vector<std::array<int, 2>> pts;
                std::vector<demo::Rgba8> exp;
                for (int j = 1; j <= 5; ++j)
                    for (int i = 1; i <= 5; ++i)
                    {
                        const int x = r.x + r.w * i / 6, y = r.y + r.h * j / 6;
                        const double u = (x - r.x + 0.5) / r.w, v = (y - r.y + 0.5) / r.h;
                        auto q = [](double c) { return static_cast<std::uint8_t>(std::lround(c * 255.0)); };
                        pts.push_back({x, y});
                        exp.push_back({q(u), q(v), q(1.0 - u), 255});
                    }
                probe(check, "#version 100: attribute + varying interpolation + gl_FragColor", pts, exp, 2);
            }
            // Cells 1 and 2: texel centres of the 4x4 NEAREST texture. Texels are exact RGBA8 values: ±1.
            auto texel_probes = [&](Rect r, std::string_view what) {
                std::vector<std::array<int, 2>> pts;
                std::vector<demo::Rgba8> exp;
                for (int j = 0; j < kTex; ++j)
                    for (int i = 0; i < kTex; ++i)
                    {
                        pts.push_back({r.x + (2 * i + 1) * r.w / (2 * kTex), r.y + (2 * j + 1) * r.h / (2 * kTex)});
                        exp.push_back(texel(i, j));
                    }
                probe(check, what, pts, exp, 1);
            };
            texel_probes(cell(env, 1), "no #version directive (treated as 100): texture2D + gl_FragData[0]");
            texel_probes(sub_rect(cell(env, 2), 0, 2, 1), "texture2DProj(sampler2D, vec3): (s, t) divided by q");
            texel_probes(sub_rect(cell(env, 2), 1, 2, 1), "texture2DProj(sampler2D, vec4): (s, t) divided by q, third component ignored");

            // Cell 3: textureCube - centre of each sub-rectangle shows the face selected by the major axis.
            auto centres = [](Rect r, int n, int cols, int rows) {
                std::vector<std::array<int, 2>> pts;
                for (int k = 0; k < n; ++k)
                {
                    const Rect s = sub_rect(r, k, cols, rows);
                    pts.push_back({s.x + s.w / 2, s.y + s.h / 2});
                }
                return pts;
            };
            probe(check, "textureCube: +X -X +Y -Y +Z -Z faces", centres(cell(env, 3), 6, 3, 2),
                  std::vector<demo::Rgba8>(kFace.begin(), kFace.end()), 1);

            // Cell 4: vertex-shader lookups with explicit LOD; NEAREST_MIPMAP_NEAREST picks level lod (ES 3.0 §3.8.10.4).
            std::vector<demo::Rgba8> lod_exp;
            for (const auto& d : kLodDraws) lod_exp.push_back(d.expected);
            probe(check, "vertex shader texture2DLod (lod 0/1/3), texture2DProjLod vec3/vec4, textureCubeLod", centres(cell(env, 4), 6, 3, 2),
                  lod_exp, 1);

            // Cell 5: lambda_base = -1 exactly (axis-aligned, rho = 0.5); bias 0 -> magnification -> level 0,
            // bias 2 (<= MAX_TEXTURE_LOD_BIAS, which is at least 2) -> lambda = 1 -> level 1 (ES 3.0 eq. 3.17-3.21).
            const auto bias_pts = centres(cell(env, 5), 2, 2, 1);
            probe(check, "fragment shader texture2D(sampler, coord, bias): bias 0 -> level 0, bias 2 -> level 1", bias_pts,
                  {kLevel[0], kLevel[1]}, 1);

            // Negative control: the probes can tell levels apart (the two bias halves and the LOD draws differ).
            const demo::Rgba8 b0 = demo::read_pixel(bias_pts[0][0], bias_pts[0][1]);
            const demo::Rgba8 b1 = demo::read_pixel(bias_pts[1][0], bias_pts[1][1]);
            check.expect(b0 != b1, "negative control: changing only the bias changes the sampled level",
                         std::format("{} vs {}", demo::format_rgba(b0), demo::format_rgba(b1)));
        }

        // ---- 2. version mixing -----------------------------------------------------------------------
        void validate_version_mixing(demo::Check& check)
        {
            const std::string vs100 = "#version 100\nattribute vec4 a_position;\nvoid main() { gl_Position = a_position; }\n";
            const std::string fs100 = "#version 100\nprecision mediump float;\nvoid main() { gl_FragColor = vec4(1.0); }\n";
            const std::string vs300 = "#version 300 es\nin vec4 a_position;\nvoid main() { gl_Position = a_position; }\n";
            const std::string fs300 = trivial_fs("300 es");
            const std::string vs310 = "#version 310 es\n#if __VERSION__ != 310\n#error \"__VERSION__ must be 310\"\n#endif\n"
                                      "in vec4 a_position;\nvoid main() { gl_Position = a_position; }\n";
            const std::string fs310 = trivial_fs("310 es");
            auto links = [&](const std::string& vs, const std::string& fs, std::string& log) {
                const Built b = build(vs, fs);
                log = b.log;
                if (b.program.value != 0) mgl::glDeleteProgram(b.program);
                return b.program.value != 0;
            };
            std::string log;
            check.expect(links(vs100, fs100, log), "positive control: #version 100 VS + #version 100 FS link", log);
            check.expect(links(vs300, fs300, log), "positive control: #version 300 es VS + FS link", log);
            check.expect(!links(vs100, fs300, log), "100 VS + 300 es FS: link fails (GLSL ES 3.00 §1.5, §3.4)", log);
            check.expect(!links(vs300, fs100, log), "300 es VS + 100 FS: link fails (GLSL ES 3.00 §1.5, §3.4)", log);
            if (!es31_)
            {
                std::cout << "     note: context < ES 3.1 - the #version 310 es checks are skipped\n";
                return;
            }
            check.expect(links(vs310, fs310, log), "#version 310 es VS (with __VERSION__ == 310) + 310 es FS link (GLSL ES 3.10 §3.3)", log);
            check.expect(!links(vs310, fs300, log), "310 es VS + 300 es FS: link fails (GLSL ES 3.10 §3.3)", log);
            check.expect(!links(vs300, fs310, log), "300 es VS + 310 es FS: link fails (GLSL ES 3.10 §3.3)", log);
            check.expect(!links(vs310, fs100, log), "310 es VS + 100 FS: link fails (GLSL ES 3.10 §3.3)", log);
        }

        // ---- comparator self-test --------------------------------------------------------------------
        static void comparator_negative_control(demo::Check& check)
        {
            const auto bits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
            const bool rejects = !matches(F(3.375f), bits(std::nextafter(3.375f, 4.0f))) &&
                                 !matches(R(16777217.0), bits(16777220.0f)) && matches(R(16777217.0), bits(16777216.0f)) &&
                                 matches(R(16777217.0), bits(16777218.0f)) && !matches(I(-3), static_cast<std::uint32_t>(-2)) &&
                                 matches(F(0.0f), bits(-0.0f)) && !matches(Ulp(1.0, 2.5), bits(1.0f + 3.0f * std::ldexp(1.0f, -23)));
            check.expect(rejects, "negative control: the comparator rejects values one step outside each tolerance class");
        }

        // ---- 4. built-in constants -------------------------------------------------------------------
        void validate_builtin_constants(demo::Check& check)
        {
            using P = mgl::GetParameter;
            const auto q = [](P p) { return demo::get_integer(p); };
            struct Constant
            {
                const char* glsl;
                int api;
                int minimum;
                bool at_most;  // gl_MinProgramTexelOffset: the limit is an upper bound
            };
            // GLSL ES 3.00 §7.3 (names, minimums). The API value of the same name is only reported, except for
            // gl_MaxVertexUniformVectors (§1.5).
            const std::vector<Constant> c300 = {
                {"gl_MaxVertexAttribs", q(P::MaxVertexAttribs), 16, false},
                {"gl_MaxVertexUniformVectors", q(P::MaxVertexUniformVectors), 256, false},
                {"gl_MaxVertexOutputVectors", q(P::MaxVertexOutputComponents) / 4, 16, false},
                {"gl_MaxFragmentInputVectors", q(P::MaxFragmentInputComponents) / 4, 15, false},
                {"gl_MaxVertexTextureImageUnits", q(P::MaxVertexTextureImageUnits), 16, false},
                {"gl_MaxCombinedTextureImageUnits", q(P::MaxCombinedTextureImageUnits), 32, false},
                {"gl_MaxTextureImageUnits", q(P::MaxTextureImageUnits), 16, false},
                {"gl_MaxFragmentUniformVectors", q(P::MaxFragmentUniformVectors), 224, false},
                {"gl_MaxDrawBuffers", q(P::MaxDrawBuffers), 4, false},
                {"gl_MinProgramTexelOffset", q(P::MinProgramTexelOffset), -8, true},
                {"gl_MaxProgramTexelOffset", q(P::MaxProgramTexelOffset), 7, false},
            };
            run_constants(c300, "300 es", "GLSL ES 3.00 §7.3", check);
            validate_vertex_uniform_vectors_100(check);
            if (!es31_)
            {
                std::cout << "     note: context < ES 3.1 - the GLSL ES 3.10 §7.2 constants are skipped\n";
                return;
            }
            const auto qi = [](P p, GLuint i) {
                GLint v = 0;
                mgl::glGetIntegeri_v(p, i, &v);
                return v;
            };
            const std::vector<Constant> c310a = {
                {"gl_MaxImageUnits", q(P::MaxImageUnits), 4, false},
                {"gl_MaxVertexImageUniforms", q(P::MaxVertexImageUniforms), 0, false},
                {"gl_MaxFragmentImageUniforms", q(P::MaxFragmentImageUniforms), 0, false},
                {"gl_MaxComputeImageUniforms", q(P::MaxComputeImageUniforms), 4, false},
                {"gl_MaxCombinedImageUniforms", q(P::MaxCombinedImageUniforms), 4, false},
                {"gl_MaxCombinedShaderOutputResources", q(P::MaxCombinedShaderOutputResources), 4, false},
                {"gl_MaxComputeUniformComponents", q(P::MaxComputeUniformComponents), 512, false},
                {"gl_MaxComputeTextureImageUnits", q(P::MaxComputeTextureImageUnits), 16, false},
                {"gl_MaxComputeAtomicCounters", q(P::MaxComputeAtomicCounters), 8, false},
                {"gl_MaxComputeAtomicCounterBuffers", q(P::MaxComputeAtomicCounterBuffers), 1, false},
                {"gl_MaxComputeWorkGroupCount.x", qi(P::MaxComputeWorkGroupCount, 0), 65535, false},
                {"gl_MaxComputeWorkGroupCount.y", qi(P::MaxComputeWorkGroupCount, 1), 65535, false},
                {"gl_MaxComputeWorkGroupCount.z", qi(P::MaxComputeWorkGroupCount, 2), 65535, false},
            };
            const std::vector<Constant> c310b = {
                {"gl_MaxComputeWorkGroupSize.x", qi(P::MaxComputeWorkGroupSize, 0), 128, false},
                {"gl_MaxComputeWorkGroupSize.y", qi(P::MaxComputeWorkGroupSize, 1), 128, false},
                {"gl_MaxComputeWorkGroupSize.z", qi(P::MaxComputeWorkGroupSize, 2), 64, false},
                {"gl_MaxVertexAtomicCounters", q(P::MaxVertexAtomicCounters), 0, false},
                {"gl_MaxFragmentAtomicCounters", q(P::MaxFragmentAtomicCounters), 0, false},
                {"gl_MaxCombinedAtomicCounters", q(P::MaxCombinedAtomicCounters), 8, false},
                {"gl_MaxAtomicCounterBindings", q(P::MaxAtomicCounterBufferBindings), 1, false},
                {"gl_MaxFragmentAtomicCounterBuffers", q(P::MaxFragmentAtomicCounterBuffers), 0, false},
                {"gl_MaxVertexAtomicCounterBuffers", q(P::MaxVertexAtomicCounterBuffers), 0, false},
                {"gl_MaxCombinedAtomicCounterBuffers", q(P::MaxCombinedAtomicCounterBuffers), 1, false},
                {"gl_MaxAtomicCounterBufferSize", q(P::MaxAtomicCounterBufferSize), 32, false},
            };
            run_constants(c310a, "310 es", "GLSL ES 3.10 §7.2", check);
            run_constants(c310b, "310 es", "GLSL ES 3.10 §7.2", check);
        }

        template <typename Constant>
        void run_constants(const std::vector<Constant>& list, std::string_view version, std::string_view section, demo::Check& check)
        {
            // §7.3 / 3.10 §7.2: "The actual values used are implementation dependent, but must be at least the
            // value shown." Only the minimums are normative for every constant; the API limit is printed.
            Case c;
            c.name = std::format("built-in constants ({}, {} constants)", section, list.size());
            for (std::size_t k = 0; k < list.size(); ++k)
                c.body += std::format("r[{}].{} = ib({});\n", k / 4, "xyzw"[k % 4], list[k].glsl);
            for (const bool fragment : {false, true})
            {
                const char* lang = fragment ? "fragment" : "vertex";
                Slots got{};
                if (!run_stage(c, version, fragment, got, check)) continue;
                int bad = 0;
                std::string first, equal, differ;
                for (std::size_t k = 0; k < list.size(); ++k)
                {
                    const int v = static_cast<std::int32_t>(got[k]);
                    const bool ok = list[k].at_most ? v <= list[k].minimum : v >= list[k].minimum;
                    if (!ok && bad++ == 0) first = std::format("{} = {} (spec minimum {})", list[k].glsl, v, list[k].minimum);
                    (v == list[k].api ? equal : differ) += std::format(" {}={}", list[k].glsl, v);
                    if (v != list[k].api) differ += std::format("(API {})", list[k].api);
                }
                check.expect(bad == 0, std::format("[{}] {}: every constant is at least the specified minimum", lang, section), first);
                std::cout << std::format("     note: [{}] {} constants equal to the API limit (not asserted):{}\n", lang, section,
                                         equal.empty() ? " none" : equal);
                if (!differ.empty())
                    std::cout << std::format("     note: [{}] {} constants different from the API limit (not asserted):{}\n", lang, section, differ);
                // GLSL ES 3.00 §1.5: "The number of uniforms specified by gl_MaxVertexUniformVectors and returned by the
                // corresponding API query is the same for GLSL ES versions 1.00 and 3.00" - the one normative tie to the API.
                for (std::size_t k = 0; k < list.size(); ++k)
                    if (std::string_view(list[k].glsl) == "gl_MaxVertexUniformVectors")
                        check.expect_eq(static_cast<std::int32_t>(got[k]), list[k].api,
                                        std::format("[{}] gl_MaxVertexUniformVectors equals MAX_VERTEX_UNIFORM_VECTORS (GLSL ES 3.00 §1.5)", lang));
            }
        }

        /// GLSL ES 3.00 §1.5: gl_MaxVertexUniformVectors is the same number in a GLSL ES 1.00 shader.
        void validate_vertex_uniform_vectors_100(demo::Check& check)
        {
            static constexpr std::array<const GLchar*, 1> kOut = {"v_n"};
            const Built b = build("#version 100\nvarying highp float v_n;\n"
                                  "void main() { v_n = float(gl_MaxVertexUniformVectors); gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n",
                                  "#version 100\nprecision mediump float;\nvoid main() { gl_FragColor = vec4(0.0); }\n", kOut);
            if (b.program.value == 0)
            {
                check.expect(false, "#version 100 program reading gl_MaxVertexUniformVectors builds", b.log);
                return;
            }
            const auto v = capture(b.program, 1, 1, empty_vao_);
            mgl::glDeleteProgram(b.program);
            const float got = std::bit_cast<float>(v[0]);  // small integers are exact in float
            const int api = demo::get_integer(mgl::GetParameter::MaxVertexUniformVectors);
            check.expect(got == static_cast<float>(api),
                         "#version 100 gl_MaxVertexUniformVectors equals MAX_VERTEX_UNIFORM_VECTORS and the 3.00 value (GLSL ES 3.00 §1.5)",
                         std::format("GLSL ES 1.00 value {} API {}", got, api));
            check.expect(got >= 128.0f, "#version 100 gl_MaxVertexUniformVectors is at least 128 (GLSL ES 1.00 §7.4)", std::format("{}", got));
        }

        // ---- 5. gl_DepthRange ------------------------------------------------------------------------
        void validate_depth_range(demo::Check& check)
        {
            struct Range
            {
                float n, f;      // passed to glDepthRangef
                float cn, cf;    // clamped to [0, 1] (ES 3.0 §2.13.1)
            };
            const std::array<Range, 4> ranges = {{{0.0f, 1.0f, 0.0f, 1.0f}, {0.25f, 0.75f, 0.25f, 0.75f}, {0.75f, 0.125f, 0.75f, 0.125f},
                                                  {1.5f, -0.5f, 1.0f, 0.0f}}};
            for (const auto& r : ranges)
            {
                mgl::glDepthRangef(r.n, r.f);
                std::array<GLfloat, 2> state{};
                mgl::glGetFloatv(mgl::GetParameter::DepthRange, state.data());
                check.expect(state[0] == r.cn && state[1] == r.cf, std::format("glDepthRangef({}, {}) stores the clamped range", r.n, r.f),
                             std::format("DEPTH_RANGE = ({}, {})", state[0], state[1]));
                Case c;
                c.name = std::format("gl_DepthRange.near/far/diff after glDepthRangef({}, {}) (GLSL ES 3.00 §7.4)", r.n, r.f);
                c.body = "r[0] = uvec4(fb(gl_DepthRange.near), fb(gl_DepthRange.far), fb(gl_DepthRange.diff), 0u);\n";
                c.expected = {F(r.cn), F(r.cf), F(r.cf - r.cn)};  // the differences are exact in float
                run_case(c, "300 es", check);
            }
            mgl::glDepthRangef(0.0f, 1.0f);
        }

        // ---- 6. invariance ---------------------------------------------------------------------------
        static std::string invariance_vs(bool pragma, bool extra, bool perturb)
        {
            const std::string expr =
                "vec4(sin(p.x) * exp(p.y) + p.z * p.w,\n"
                "         inversesqrt(abs(p.x) + 1.0) * p.y - p.z / (abs(p.w) + 2.0),\n"
                "         pow(abs(p.z) + 0.5, p.w) + log2(abs(p.y) + 1.0),\n"
                "         dot(p.xyz, p.yzw) * 0.25 + 1.0)";
            std::string s = "#version 300 es\n";
            s += pragma ? "#pragma STDGL invariant(all)\n" : "invariant gl_Position;\n";
            s += "layout(location = 0) in highp vec4 a_in;\nout highp vec4 v_res;\nout highp vec4 v_extra;\nvoid main()\n{\n"
                 "    highp vec4 p = a_in;\n";
            // The other outputs re-use sub-expressions of the invariant one, which invites different optimisation.
            s += extra ? "    v_extra = vec4(sin(p.x) * 3.0, exp(p.y) + p.z * p.w, p.z / (abs(p.w) + 2.0), dot(p.xyz, p.yzw));\n"
                       : "    v_extra = vec4(0.0);\n";
            const std::string value = expr + (perturb ? " + vec4(1.0e-3, 0.0, 0.0, 0.0)" : "");
            if (pragma) s += "    v_res = " + value + ";\n    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n";
            else s += "    gl_Position = " + value + ";\n    v_res = vec4(0.0);\n";
            return s + "}\n";
        }

        void validate_invariance(demo::Check& check)
        {
            constexpr int kVerts = 64;
            std::vector<GLfloat> inputs;
            for (int i = 0; i < kVerts; ++i)
            {
                const double t = i;
                inputs.push_back(static_cast<GLfloat>(2.0 * std::sin(t * 1.3)));
                inputs.push_back(static_cast<GLfloat>(1.5 * std::cos(t * 0.7)));
                inputs.push_back(static_cast<GLfloat>(std::sin(t * 2.1 + 0.5) * 3.0));
                inputs.push_back(static_cast<GLfloat>(std::cos(t * 0.37) * 2.0));
            }
            mgl::VertexArrayId vao{};
            mgl::glGenVertexArrays(1, &vao);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, inv_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const GLfloat>(inputs), mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0}, 4, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            auto run = [&](bool pragma, bool extra, bool perturb, std::vector<std::uint32_t>& out) {
                static constexpr std::array<const GLchar*, 2> kQualifier = {"gl_Position", "v_extra"};
                static constexpr std::array<const GLchar*, 2> kPragma = {"v_res", "v_extra"};
                const Built b = build(invariance_vs(pragma, extra, perturb), trivial_fs("300 es"), pragma ? kPragma : kQualifier);
                if (b.program.value == 0)
                {
                    check.expect(false, std::format("invariance program (pragma={}, extra={}) builds", pragma, extra), b.log);
                    return false;
                }
                const auto all = capture(b.program, kVerts, 8, vao);
                mgl::glDeleteProgram(b.program);
                out.clear();
                for (int v = 0; v < kVerts; ++v)
                    for (int k = 0; k < 4; ++k) out.push_back(all[static_cast<std::size_t>(v * 8 + k)]);  // the invariant output
                return true;
            };
            auto differences = [](const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
                int n = 0;
                for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i] ? 1 : 0;
                return n;
            };
            std::vector<std::uint32_t> a, b, c, d, e;
            if (run(false, false, false, a) && run(false, true, false, b))
                check.expect(differences(a, b) == 0,
                             "invariant gl_Position: two programs with the same data flow give bit-identical values (GLSL ES 3.00 §4.6.1)",
                             std::format("{} of {} components differ", differences(a, b), a.size()));
            if (run(true, false, false, c) && run(true, true, false, d))
                check.expect(differences(c, d) == 0,
                             "#pragma STDGL invariant(all): a user output is bit-identical across two programs (GLSL ES 3.00 §4.6.1)",
                             std::format("{} of {} components differ", differences(c, d), c.size()));
            if (!a.empty() && run(false, false, true, e))
                check.expect(differences(a, e) > 0, "negative control: a perturbed expression is detected by the bitwise comparison",
                             differences(a, e) > 0 ? std::format("{} of {} components differ", differences(a, e), a.size()) : "no component differs");
            mgl::glDeleteVertexArrays(1, &vao);
        }

        bool es31_ = false;
        mgl::ProgramId grad_{}, noversion_{}, cube_prog_{}, lod_{}, bias_{};
        mgl::VertexArrayId quad_vao_{}, empty_vao_{};
        mgl::BufferId quad_vbo_{}, tfb_{}, inv_vbo_{};
        mgl::TextureId tex4_{}, cube_{}, mip_{};
        mgl::FramebufferId fbo_{};
        std::array<mgl::RenderbufferId, 4> rb_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "test-glsl-language",
        .title = "GLSL ES language rules at run time",
        .min_es = {3, 0},
        .summary = "GLSL ES 1.00 shaders rendered on ES 3.x and compared with a CPU reference; version mixing; operators, "
                   "parameter passing, constant folding and scoping evaluated in the vertex and the fragment language; "
                   "gl_Max* constants vs the specified minimums; gl_DepthRange; invariance.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<GlslLanguage>(argc, argv, info);
}
