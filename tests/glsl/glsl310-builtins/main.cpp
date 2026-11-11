// test-glsl310-builtins
//
// Checks the built-in functions that GLSL ES 3.10 added (§8.8 integer functions, §8.3 frexp/ldexp,
// §8.4 packUnorm4x8/packSnorm4x8/unpackUnorm4x8/unpackSnorm4x8) and arrays of arrays (§4.1.9).
//
// Method: every case is a compute shader. Each invocation reads one input row (4 x uvec4) from a
// shader storage buffer, evaluates the case body and writes one output row (4 x uvec4) to a second
// SSBO. Floats travel as their bit patterns (uintBitsToFloat / floatBitsToUint), so nothing is
// quantized on the way. The output is read back with glMapBufferRange after
// glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT) and compared with a CPU reference.
//
// Exactness (GLSL ES 3.10 §4.7.1 and the function descriptions):
//   - Integer functions are exact by definition; compared bit for bit. Undefined argument ranges
//     (negative offset/bits, offset + bits > 32) are never used; offset + bits == 32 and bits == 0
//     (including offset == 32) are defined and are tested.
//   - frexp/ldexp: defined exactly for zero and finite normal values; results compared bit for bit,
//     with +0/-0 interchangeable (§4.7.1). Inf/NaN (undefined for frexp), subnormals (may be
//     flushed) and ldexp exponents outside [-126, 128] (undefined / may be flushed) are not used.
//   - packUnorm4x8/packSnorm4x8 take mediump vec4 and use round(), whose 0.5 direction is
//     implementation-chosen (§8.3). Inputs are multiples of 2^-10 (exact in any mediump) whose
//     scaled value stays farther than mediump error from a .5 tie, so the packed byte is unique;
//     compared bit for bit.
//   - unpackUnorm4x8/unpackSnorm4x8 return mediump vec4 computed with a division (2.5 ULP highp,
//     relative 2^-10 mediump), so they are compared with relative tolerance 2^-10 (measured error
//     printed), plus the exact clamp unpackSnorm4x8(0x80) == -1 and an exact pack(unpack(x))
//     round trip.
// Negative controls run the CPU references with plausible mistakes (truncation instead of rounding,
// 256 instead of 255, no sign extension, unsigned findMSB, [1,2) significands, outer-fastest
// array-of-array layout) and require that they disagree with the GPU.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    // One row = 4 x uvec4 = 16 x 32-bit words. Inputs u0..u3, outputs r0..r3.
    using U16 = std::array<std::uint32_t, 16>;

    constexpr std::uint32_t kSeed = 0x310B17u;  // fixed: the random set is identical on every run
    constexpr std::uint32_t kSentinel = 0xDEADBEEFu;
    constexpr GLuint kLocalSize = 64;

    // ---------------- CPU references: GLSL ES 3.10 §8.8 -----------------------------------------
    std::uint32_t low_mask(int bits) { return bits >= 32 ? 0xFFFFFFFFu : (1u << bits) - 1u; }  // bits in [0, 32]

    std::uint32_t bfe_u(std::uint32_t v, int offset, int bits)
    {
        if (bits == 0) return 0;  // "If bits is zero, the result will be zero" (also for offset 32)
        return static_cast<std::uint32_t>(static_cast<std::uint64_t>(v) >> offset) & low_mask(bits);
    }

    // Signed: the bits above the field copy bit offset+bits-1.
    std::uint32_t bfe_i(std::uint32_t v, int offset, int bits)
    {
        std::uint32_t x = bfe_u(v, offset, bits);
        if (bits > 0 && bits < 32 && ((x >> (bits - 1)) & 1u) != 0) x |= ~low_mask(bits);
        return x;
    }

    std::uint32_t bfi(std::uint32_t base, std::uint32_t insert, int offset, int bits)
    {
        if (bits == 0) return base;
        const auto m = static_cast<std::uint32_t>(static_cast<std::uint64_t>(low_mask(bits)) << offset);
        return (base & ~m) | (static_cast<std::uint32_t>(static_cast<std::uint64_t>(insert) << offset) & m);
    }

    std::uint32_t reverse_bits(std::uint32_t v)
    {
        std::uint32_t r = 0;
        for (int i = 0; i < 32; ++i)
            if ((v >> i) & 1u) r |= 1u << (31 - i);
        return r;
    }

    std::uint32_t as_u(std::int32_t v) { return std::bit_cast<std::uint32_t>(v); }
    std::int32_t as_i(std::uint32_t v) { return std::bit_cast<std::int32_t>(v); }
    std::uint32_t find_lsb(std::uint32_t v) { return as_u(v == 0 ? -1 : std::countr_zero(v)); }
    std::uint32_t find_msb_u(std::uint32_t v) { return as_u(v == 0 ? -1 : 31 - std::countl_zero(v)); }
    // Negative values: the most significant ZERO bit; 0 and -1 give -1.
    std::uint32_t find_msb_i(std::uint32_t v) { return as_i(v) < 0 ? find_msb_u(~v) : find_msb_u(v); }

    // ---------------- CPU references: §8.3 frexp/ldexp, §8.4 pack/unpack ------------------------
    float f(std::uint32_t bits) { return std::bit_cast<float>(bits); }
    std::uint32_t b(float v) { return std::bit_cast<std::uint32_t>(v); }

    enum class Rounding { Nearest, Truncate };  // Truncate exists only for the negative controls

    std::uint32_t to_fixed(double scaled, Rounding r)
    {
        const double q = r == Rounding::Nearest ? std::round(scaled) : std::trunc(scaled);  // inputs avoid .5 ties
        return static_cast<std::uint32_t>(static_cast<std::int32_t>(q)) & 0xFFu;
    }

    // fixed = round(clamp(c, 0, 1) * 255.0), first component in the least significant byte.
    std::uint32_t pack_unorm8(const std::array<float, 4>& v, double scale = 255.0, Rounding r = Rounding::Nearest)
    {
        std::uint32_t p = 0;
        for (int i = 0; i < 4; ++i) p |= to_fixed(std::clamp(double(v[static_cast<std::size_t>(i)]), 0.0, 1.0) * scale, r) << (8 * i);
        return p;
    }

    // fixed = round(clamp(c, -1, 1) * 127.0) as a two's complement byte.
    std::uint32_t pack_snorm8(const std::array<float, 4>& v, double scale = 127.0, Rounding r = Rounding::Nearest)
    {
        std::uint32_t p = 0;
        for (int i = 0; i < 4; ++i) p |= to_fixed(std::clamp(double(v[static_cast<std::size_t>(i)]), -1.0, 1.0) * scale, r) << (8 * i);
        return p;
    }

    double unpack_unorm8(std::uint32_t p, int i) { return double((p >> (8 * i)) & 0xFFu) / 255.0; }
    double unpack_snorm8(std::uint32_t p, int i)
    {
        return std::clamp(double(static_cast<std::int8_t>((p >> (8 * i)) & 0xFFu)) / 127.0, -1.0, 1.0);
    }

    std::array<float, 4> lane_floats(const U16& in, int vec)
    {
        const auto o = static_cast<std::size_t>(vec * 4);
        return {f(in[o]), f(in[o + 1]), f(in[o + 2]), f(in[o + 3])};
    }

    // ---------------- cases -----------------------------------------------------------------------
    struct Case
    {
        std::string name;
        std::string feature;
        std::string body;               // GLSL: reads u0..u3 / i0..i3 / f0..f3, writes r0..r3
        int components = 16;            // output words compared: [0, components)
        std::uint16_t float_mask = 0;   // words holding float bits: +0 and -0 interchangeable (§4.7.1)
        std::uint16_t mediump_mask = 0; // words holding mediump results: relative 2^-10
        std::vector<U16> rows;
        std::function<U16(const U16&)> ref;
    };

    // Mediump tolerance: the minimum mediump precision of GLSL ES 3.10 §4.7.1 (relative 2^-10).
    const double kMediumpRel = std::ldexp(1.0, -10);

    std::string compute_source(const std::string& body)
    {
        return std::format(
            "#version 310 es\n"
            "layout(local_size_x = {0}) in;\n"
            "precision highp float;\nprecision highp int;\n"
            "layout(std430, binding = 0) readonly buffer Inputs {{ uvec4 inp[]; }};\n"
            "layout(std430, binding = 1) writeonly buffer Outputs {{ uvec4 outp[]; }};\n"
            "void main()\n{{\n"
            "    uint row = gl_GlobalInvocationID.x;\n"
            "    if (row >= uint(inp.length()) / 4u) return;\n"
            "    uvec4 u0 = inp[4u * row + 0u], u1 = inp[4u * row + 1u], u2 = inp[4u * row + 2u], u3 = inp[4u * row + 3u];\n"
            "    ivec4 i0 = ivec4(u0), i1 = ivec4(u1), i2 = ivec4(u2), i3 = ivec4(u3);\n"
            "    vec4 f0 = uintBitsToFloat(u0), f1 = uintBitsToFloat(u1);\n"
            "    uvec4 r0 = uvec4(0u), r1 = uvec4(0u), r2 = uvec4(0u), r3 = uvec4(0u);\n"
            "    {1}\n"
            "    outp[4u * row + 0u] = r0; outp[4u * row + 1u] = r1; outp[4u * row + 2u] = r2; outp[4u * row + 3u] = r3;\n"
            "}}\n",
            kLocalSize, body);
    }

    class Rng
    {
    public:
        std::uint32_t next() { return static_cast<std::uint32_t>(gen_()); }  // raw mt19937 output: portable
        std::uint32_t below(std::uint32_t n) { return next() % n; }

    private:
        std::mt19937 gen_{kSeed};
    };

    const std::vector<std::uint32_t> kEdges = {0u,          1u,          2u,          0x7FFFFFFFu, 0x80000000u, 0x80000001u,
                                               0xFFFFFFFFu, 0xFFFFFFFEu, 0x55555555u, 0xAAAAAAAAu, 0x0000FFFFu, 0xFFFF0000u,
                                               0x00010000u, 0x12345678u, 0x87654321u, 0x00000100u};

    // Every ordered pair of edge values (64 rows) plus 256 random rows: lanes u0 (x) and u1 (y).
    std::vector<U16> pair_rows(Rng& rng)
    {
        std::vector<U16> rows;
        U16 r{};
        int lane = 0;
        for (std::uint32_t x : kEdges)
            for (std::uint32_t y : kEdges)
            {
                r[static_cast<std::size_t>(lane)] = x;
                r[static_cast<std::size_t>(4 + lane)] = y;
                if (++lane == 4) { rows.push_back(r); r = {}; lane = 0; }
            }
        for (int i = 0; i < 256; ++i)
        {
            U16 q{};
            for (int k = 0; k < 8; ++k) q[static_cast<std::size_t>(k)] = rng.next();
            rows.push_back(q);
        }
        return rows;
    }

    // Values with every bit length, single bits and edges, for reverse/count/findLSB/findMSB.
    std::vector<U16> value_rows(Rng& rng)
    {
        std::vector<std::uint32_t> v = kEdges;
        for (int k = 0; k < 32; ++k)
        {
            v.push_back(1u << k);
            v.push_back(~(1u << k));
            v.push_back(low_mask(k));
            v.push_back(~low_mask(k));
        }
        for (int i = 0; i < 512; ++i) v.push_back(rng.next() >> rng.below(32));
        std::vector<U16> rows;
        for (std::size_t i = 0; i + 4 <= v.size(); i += 4)
        {
            U16 r{};
            for (std::size_t k = 0; k < 4; ++k) r[k] = v[i + k];
            rows.push_back(r);
        }
        return rows;
    }

    // u0, u1 = values, u2.x = offset, u2.y = bits; only the defined range offset, bits >= 0 and
    // offset + bits <= 32 (GLSL ES 3.10 §8.8).
    std::vector<U16> bitfield_rows(Rng& rng)
    {
        const std::vector<std::array<int, 2>> edges = {{0, 0},  {0, 1},  {0, 31}, {0, 32}, {1, 31}, {31, 1},  {16, 16}, {8, 8},  {31, 0},
                                                       {32, 0}, {5, 0},  {24, 8}, {4, 28}, {28, 4}, {15, 17}, {0, 16},  {16, 0}, {30, 2}};
        std::vector<U16> rows;
        std::size_t e = 0;
        for (const auto& [offset, bits] : edges)
            for (int rep = 0; rep < 4; ++rep)
            {
                U16 r{};
                for (std::size_t k = 0; k < 4; ++k)
                {
                    r[k] = kEdges[(e + k) % kEdges.size()];
                    r[4 + k] = kEdges[(e + 7 + 3 * k) % kEdges.size()];
                }
                e += 4;
                r[8] = static_cast<std::uint32_t>(offset);
                r[9] = static_cast<std::uint32_t>(bits);
                rows.push_back(r);
            }
        for (int i = 0; i < 256; ++i)
        {
            U16 r{};
            for (std::size_t k = 0; k < 8; ++k) r[k] = rng.next();
            const std::uint32_t offset = rng.below(33);
            r[8] = offset;
            r[9] = rng.below(33 - offset);
            rows.push_back(r);
        }
        return rows;
    }

    // Random finite normal float (exponent field 1..254), random sign.
    float random_normal(Rng& rng)
    {
        const std::uint32_t exp = 1 + rng.below(254);
        return f((rng.next() & 0x807FFFFFu) | (exp << 23));
    }

    std::vector<U16> frexp_rows(Rng& rng)
    {
        const float mx = std::numeric_limits<float>::max(), mn = std::numeric_limits<float>::min();
        std::vector<float> v = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.75f, 3.0f, -6.5f, 1024.0f, mn, -mn, 2.0f * mn,
                                mx,   -mx,   std::nextafter(1.0f, 0.0f), std::nextafter(1.0f, 2.0f), 1e-30f, 1e30f, 0.1f};
        while (v.size() % 4 != 0) v.push_back(1.5f);
        for (int i = 0; i < 1024; ++i) v.push_back(random_normal(rng));
        std::vector<U16> rows;
        for (std::size_t i = 0; i < v.size(); i += 4)
        {
            U16 r{};
            for (std::size_t k = 0; k < 4; ++k) r[k] = b(v[i + k]);
            rows.push_back(r);
        }
        return rows;
    }

    // u0 = x (zero or finite normal), u1 = exponent with -126 <= exponent <= 128 and a normal result.
    std::vector<U16> ldexp_rows(Rng& rng)
    {
        const float mn = std::numeric_limits<float>::min();
        const std::vector<std::pair<float, int>> edges = {
            {0.5f, 128}, {0.75f, 128}, {1.0f, 127}, {-1.0f, 127}, {1.0f, -126}, {0.5f, -125}, {-1.5f, 10}, {3.0f, 0},
            {0.0f, 50},  {-0.0f, -3},  {mn, 126},   {mn, 0},      {std::numeric_limits<float>::max(), 0}, {1.0f, 0}, {0.0f, 128}, {0.0f, -126},
        };
        std::vector<U16> rows;
        for (std::size_t i = 0; i < edges.size(); i += 4)
        {
            U16 r{};
            for (std::size_t k = 0; k < 4; ++k)
            {
                r[k] = b(edges[i + k].first);
                r[4 + k] = as_u(edges[i + k].second);
            }
            rows.push_back(r);
        }
        for (int i = 0; i < 256; ++i)
        {
            U16 r{};
            for (std::size_t k = 0; k < 4; ++k)
            {
                const float x = random_normal(rng);
                int ex = 0;
                (void)std::frexp(x, &ex);
                int e = 0;
                do
                {
                    const int target = -125 + static_cast<int>(rng.below(254));  // result exponent: x' = s * 2^target, normal
                    e = target - ex;
                } while (e < -126 || e > 128);
                r[k] = b(x);
                r[4 + k] = as_u(e);
            }
            rows.push_back(r);
        }
        return rows;
    }

    // Inputs are multiples of 2^-10 with at most 10 significant bits inside [-1, 1], so they are exact
    // in any mediump representation (values beyond +-1 clamp to the same result whatever their
    // rounding). A pack input is "safe" when every implementation that computes clamp(v) * scale
    // with mediump precision (relative 2^-10, §4.7.1) must produce the same byte:
    // |p - round(p)| + 2^-10 * |p| < 0.5, i.e. no .5 tie is reachable within the error.
    bool pack_safe(double p) { return std::abs(p - std::round(p)) + std::ldexp(std::abs(p), -10) < 0.5; }
    bool pack_input_safe(float v)
    {
        return pack_safe(std::clamp(double(v), 0.0, 1.0) * 255.0) && pack_safe(std::clamp(double(v), -1.0, 1.0) * 127.0);
    }

    std::vector<U16> pack_rows(Rng& rng, int& rejected)
    {
        std::vector<float> v = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 1.5f, -1.5f, 0.25f, -0.25f, 0.125f, -0.125f, 0.75f, -0.75f,
                                0.0625f, -0.0625f, 0.9990234375f, -0.9990234375f, 0.875f, -0.875f, 0.00390625f, -0.00390625f, -0.0f};
        for (float x : v)
            if (!pack_input_safe(x)) throw demo::Failure(std::format("pack edge input {} is not tie-safe", x));
        rejected = 0;
        while (v.size() < 1024)
        {
            const float x = static_cast<float>(static_cast<int>(rng.below(3201)) - 1600) / 1024.0f;  // multiples of 2^-10 in [-1.5625, 1.5625]
            if (pack_input_safe(x)) v.push_back(x);
            else ++rejected;
        }
        std::vector<U16> rows;
        for (std::size_t i = 0; i < v.size(); i += 8)
        {
            U16 r{};
            for (std::size_t k = 0; k < 8; ++k) r[k] = b(v[i + k]);
            rows.push_back(r);
        }
        return rows;
    }

    std::vector<U16> unpack_rows(Rng& rng)
    {
        std::vector<U16> rows;
        for (std::uint32_t i = 0; i < 64; ++i)
        {
            U16 r{};
            r[0] = (4 * i) | ((4 * i + 1) << 8) | ((4 * i + 2) << 16) | ((4 * i + 3) << 24);  // every byte value once
            r[1] = rng.next();
            r[2] = i == 0 ? 0x80808080u : i == 1 ? 0x7F81FF01u : rng.next();
            rows.push_back(r);
        }
        return rows;
    }

    // Integer constant expressions (§4.3.3): built-in calls with constant arguments are folded by the
    // compiler, a separate code path from the run-time evaluation above.
    struct ConstExpr
    {
        std::string type;  // "uint" or "int"
        std::string expr;
        std::uint32_t expected;
    };

    std::vector<ConstExpr> const_exprs()
    {
        std::vector<ConstExpr> c;
        auto lit = [](std::uint32_t v) { return std::format("0x{:08X}u", v); };
        auto ilit = [&](std::uint32_t v) { return std::format("int({})", lit(v)); };
        const std::vector<std::array<int, 2>> fields = {{4, 8}, {16, 16}, {0, 32}, {31, 1}, {32, 0}, {7, 0}, {1, 31}};
        const std::uint32_t vals[] = {0x87654321u, 0xF0F0F0F0u, 0x80000000u};
        for (std::uint32_t v : vals)
            for (const auto& [o, n] : fields)
            {
                c.push_back({"uint", std::format("bitfieldExtract({}, {}, {})", lit(v), o, n), bfe_u(v, o, n)});
                c.push_back({"int", std::format("bitfieldExtract({}, {}, {})", ilit(v), o, n), bfe_i(v, o, n)});
            }
        for (const auto& [o, n] : fields)
        {
            c.push_back({"uint", std::format("bitfieldInsert({}, {}, {}, {})", lit(0x12345678u), lit(0xFEDCBA98u), o, n), bfi(0x12345678u, 0xFEDCBA98u, o, n)});
            c.push_back({"int", std::format("bitfieldInsert({}, {}, {}, {})", ilit(0xFFFFFFFFu), ilit(0x0u), o, n), bfi(0xFFFFFFFFu, 0u, o, n)});
        }
        for (std::uint32_t v : {0u, 1u, 0x12345678u, 0x80000000u, 0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFEu, 0x00F00000u})
        {
            c.push_back({"uint", std::format("bitfieldReverse({})", lit(v)), reverse_bits(v)});
            c.push_back({"int", std::format("bitfieldReverse({})", ilit(v)), reverse_bits(v)});
            c.push_back({"int", std::format("bitCount({})", lit(v)), static_cast<std::uint32_t>(std::popcount(v))});
            c.push_back({"int", std::format("bitCount({})", ilit(v)), static_cast<std::uint32_t>(std::popcount(v))});
            c.push_back({"int", std::format("findLSB({})", lit(v)), find_lsb(v)});
            c.push_back({"int", std::format("findLSB({})", ilit(v)), find_lsb(v)});
            c.push_back({"int", std::format("findMSB({})", lit(v)), find_msb_u(v)});
            c.push_back({"int", std::format("findMSB({})", ilit(v)), find_msb_i(v)});
        }
        return c;
    }

    std::vector<Case> make_cases(int& pack_rejected)
    {
        Rng rng;
        std::vector<Case> c;

        // ---------------- §8.8 integer functions (run-time arguments) ------------------------------
        c.push_back({"uaddCarry (genUType and scalar)", "GLSL310.FN.INTEGER",
                     "uvec4 c; r0 = uaddCarry(u0, u1, c); r1 = c; uint cs; r2.x = uaddCarry(u0.x, u1.x, cs); r2.y = cs;", 10, 0, 0, pair_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             const std::uint64_t s = std::uint64_t(in[k]) + in[4 + k];
                             o[k] = static_cast<std::uint32_t>(s);
                             o[4 + k] = static_cast<std::uint32_t>(s >> 32);
                         }
                         o[8] = o[0]; o[9] = o[4];
                         return o;
                     }});
        c.push_back({"usubBorrow (genUType and scalar)", "GLSL310.FN.INTEGER",
                     "uvec4 c; r0 = usubBorrow(u0, u1, c); r1 = c; uint cs; r2.x = usubBorrow(u0.x, u1.x, cs); r2.y = cs;", 10, 0, 0, pair_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             o[k] = in[k] - in[4 + k];  // modulo 2^32
                             o[4 + k] = in[k] < in[4 + k] ? 1u : 0u;
                         }
                         o[8] = o[0]; o[9] = o[4];
                         return o;
                     }});
        c.push_back({"umulExtended (msb, lsb)", "GLSL310.FN.INTEGER",
                     "umulExtended(u0, u1, r0, r1); uint m, l; umulExtended(u0.x, u1.x, m, l); r2.xy = uvec2(m, l);", 10, 0, 0, pair_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             const std::uint64_t p = std::uint64_t(in[k]) * in[4 + k];
                             o[k] = static_cast<std::uint32_t>(p >> 32);
                             o[4 + k] = static_cast<std::uint32_t>(p);
                         }
                         o[8] = o[0]; o[9] = o[4];
                         return o;
                     }});
        c.push_back({"imulExtended (msb, lsb)", "GLSL310.FN.INTEGER",
                     "ivec4 m, l; imulExtended(i0, i1, m, l); r0 = uvec4(m); r1 = uvec4(l);"
                     " int ms, ls; imulExtended(i0.x, i1.x, ms, ls); r2.xy = uvec2(ivec2(ms, ls));",
                     10, 0, 0, pair_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             const auto p = static_cast<std::uint64_t>(std::int64_t(as_i(in[k])) * std::int64_t(as_i(in[4 + k])));
                             o[k] = static_cast<std::uint32_t>(p >> 32);
                             o[4 + k] = static_cast<std::uint32_t>(p);
                         }
                         o[8] = o[0]; o[9] = o[4];
                         return o;
                     }});
        c.push_back({"bitfieldExtract (uint/int; vector, scalar, vec2 forms; bits = 0, offset + bits = 32)", "GLSL310.FN.INTEGER",
                     "r0 = bitfieldExtract(u0, i2.x, i2.y); ivec4 s = bitfieldExtract(i0, i2.x, i2.y); r1 = uvec4(s);"
                     " int sy = bitfieldExtract(i0.y, i2.x, i2.y);"
                     " r2 = uvec4(bitfieldExtract(u0.x, i2.x, i2.y), uint(sy), bitfieldExtract(u0.zw, i2.x, i2.y));",
                     12, 0, 0, bitfield_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         const int off = as_i(in[8]), bits = as_i(in[9]);
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             o[k] = bfe_u(in[k], off, bits);
                             o[4 + k] = bfe_i(in[k], off, bits);
                         }
                         o[8] = bfe_u(in[0], off, bits); o[9] = bfe_i(in[1], off, bits); o[10] = bfe_u(in[2], off, bits); o[11] = bfe_u(in[3], off, bits);
                         return o;
                     }});
        c.push_back({"bitfieldInsert (uint/int; vector and scalar forms; bits = 0, offset + bits = 32)", "GLSL310.FN.INTEGER",
                     "r0 = bitfieldInsert(u0, u1, i2.x, i2.y); ivec4 s = bitfieldInsert(i0, i1, i2.x, i2.y); r1 = uvec4(s);"
                     " int sy = bitfieldInsert(i0.y, i1.y, i2.x, i2.y); r2.xy = uvec2(bitfieldInsert(u0.x, u1.x, i2.x, i2.y), uint(sy));",
                     10, 0, 0, bitfield_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         const int off = as_i(in[8]), bits = as_i(in[9]);
                         for (std::size_t k = 0; k < 4; ++k) o[k] = o[4 + k] = bfi(in[k], in[4 + k], off, bits);
                         o[8] = bfi(in[0], in[4], off, bits); o[9] = bfi(in[1], in[5], off, bits);
                         return o;
                     }});
        c.push_back({"bitfieldReverse/bitCount (uint/int)", "GLSL310.FN.INTEGER",
                     "r0 = bitfieldReverse(u0); ivec4 rv = bitfieldReverse(i0); r1 = uvec4(rv);"
                     " ivec4 cu = bitCount(u0); ivec4 ci = bitCount(i0); r2 = uvec4(cu); r3 = uvec4(ci);",
                     16, 0, 0, value_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             o[k] = o[4 + k] = reverse_bits(in[k]);
                             o[8 + k] = o[12 + k] = static_cast<std::uint32_t>(std::popcount(in[k]));
                         }
                         return o;
                     }});
        c.push_back({"findLSB/findMSB (uint/int; 0 and -1 give -1)", "GLSL310.FN.INTEGER",
                     "ivec4 lu = findLSB(u0); ivec4 li = findLSB(i0); ivec4 mu = findMSB(u0); ivec4 mi = findMSB(i0);"
                     " r0 = uvec4(lu); r1 = uvec4(li); r2 = uvec4(mu); r3 = uvec4(mi);",
                     16, 0, 0, value_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             o[k] = o[4 + k] = find_lsb(in[k]);
                             o[8 + k] = find_msb_u(in[k]);
                             o[12 + k] = find_msb_i(in[k]);
                         }
                         return o;
                     }});

        // ---------------- §8.8 constant-folded forms ----------------------------------------------
        const auto consts = const_exprs();
        for (std::size_t first = 0; first < consts.size(); first += 16)
        {
            const std::size_t count = std::min<std::size_t>(16, consts.size() - first);
            std::string body;
            U16 expected{};
            for (std::size_t k = 0; k < count; ++k)
            {
                const auto& ce = consts[first + k];
                body += std::format("const highp {} k{} = {}; ", ce.type, k, ce.expr);
                expected[k] = ce.expected;
            }
            for (std::size_t k = 0; k < count; ++k) body += std::format("r{}[{}] = uint(k{}); ", k / 4, k % 4, k);
            c.push_back({std::format("constant expressions {}-{} ({} ... )", first, first + count - 1, consts[first].expr), "GLSL310.FN.INTEGER", body,
                         static_cast<int>(count), 0, 0, {U16{}}, [expected](const U16&) { return expected; }});
        }

        // ---------------- §8.3 frexp / ldexp -------------------------------------------------------
        c.push_back({"frexp (zero, +-0, normal range incl. FLT_MIN/FLT_MAX)", "GLSL310.FN.FLOAT_EXT",
                     "ivec4 e; r0 = floatBitsToUint(frexp(f0, e)); r1 = uvec4(e); int es; r2.x = floatBitsToUint(frexp(f0.x, es)); r2.y = uint(es);",
                     10, 0x010F, 0, frexp_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             int e = 0;
                             const float s = std::frexp(f(in[k]), &e);  // significand in [0.5, 1), 0 -> (0, 0)
                             o[k] = b(s);
                             o[4 + k] = as_u(e);
                         }
                         o[8] = o[0]; o[9] = o[4];
                         return o;
                     }});
        c.push_back({"ldexp (normal results, exponent in [-126, 128]) and ldexp(frexp(x)) == x", "GLSL310.FN.FLOAT_EXT",
                     "r0 = floatBitsToUint(ldexp(f0, i1)); ivec4 e; r1 = floatBitsToUint(ldexp(frexp(f0, e), e)); r2.x = floatBitsToUint(ldexp(f0.x, i1.x));",
                     9, 0x01FF, 0, ldexp_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (std::size_t k = 0; k < 4; ++k)
                         {
                             o[k] = b(std::ldexp(f(in[k]), as_i(in[4 + k])));  // exact: the result is a normal float
                             o[4 + k] = in[k];
                         }
                         o[8] = o[0];
                         return o;
                     }});

        // ---------------- §8.4 4x8 packing ----------------------------------------------------------
        c.push_back({"packUnorm4x8/packSnorm4x8 (clamp, round, byte order)", "GLSL310.FN.PACKING_8",
                     "r0 = uvec4(packUnorm4x8(f0), packSnorm4x8(f0), packUnorm4x8(f1), packSnorm4x8(f1));", 4, 0, 0,
                     pack_rows(rng, pack_rejected),
                     [](const U16& in) {
                         const auto a = lane_floats(in, 0), b1 = lane_floats(in, 1);
                         U16 o{};
                         o[0] = pack_unorm8(a); o[1] = pack_snorm8(a); o[2] = pack_unorm8(b1); o[3] = pack_snorm8(b1);
                         return o;
                     }});
        c.push_back({"unpackUnorm4x8/unpackSnorm4x8 (all 256 bytes) and exact pack(unpack(x)) round trip", "GLSL310.FN.PACKING_8",
                     "r0 = floatBitsToUint(unpackUnorm4x8(u0.x)); r1 = floatBitsToUint(unpackSnorm4x8(u0.x));"
                     " r2 = uvec4(packUnorm4x8(unpackUnorm4x8(u0.y)), packSnorm4x8(unpackSnorm4x8(u0.y)),"
                     " packUnorm4x8(unpackUnorm4x8(u0.z)), packSnorm4x8(unpackSnorm4x8(u0.z)));",
                     12, 0x00FF, 0x00FF, unpack_rows(rng),
                     [](const U16& in) {
                         U16 o{};
                         for (int k = 0; k < 4; ++k)
                         {
                             o[static_cast<std::size_t>(k)] = b(static_cast<float>(unpack_unorm8(in[0], k)));
                             o[static_cast<std::size_t>(4 + k)] = b(static_cast<float>(unpack_snorm8(in[0], k)));
                         }
                         // Snorm round trip: 0x80 (-128) unpacks to -1.0 (clamp) and packs to 0x81 (-127).
                         auto snorm_rt = [](std::uint32_t p) {
                             std::uint32_t r = 0;
                             for (int k = 0; k < 4; ++k)
                             {
                                 std::uint32_t byte = (p >> (8 * k)) & 0xFFu;
                                 r |= (byte == 0x80u ? 0x81u : byte) << (8 * k);
                             }
                             return r;
                         };
                         o[8] = in[1]; o[9] = snorm_rt(in[1]); o[10] = in[2]; o[11] = snorm_rt(in[2]);
                         return o;
                     }});
        return c;
    }

    // Expected values of shaders/arrays_of_arrays.comp, index = res[] slot.
    struct AoaExpect
    {
        int index;
        const char* label;
        std::int32_t value;
    };

    int src_value(int i, int j, int k) { return (i + 1) * 1000 + j * 37 - k * 5; }

    std::vector<AoaExpect> aoa_expectations()
    {
        int sum = 0;
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k) sum += src_value(1, j, k);
        return {
            {0, "float a[2][3]: a.length() == 2", 2},
            {1, "float a[2][3]: a[0].length() == 3", 3},
            {2, "float[3] b[2]: b.length() == 2", 2},
            {3, "float[3] b[2]: b[1].length() == 3", 3},
            {4, "float[2][3] c: c.length() == 2", 2},
            {5, "float[2][3] c: c[1].length() == 3", 3},
            {6, "vec4[2] w[3]: w.length() == 3", 3},
            {7, "vec4[2] w[3]: w[0].length() == 2", 2},
            {8, "int d[2][3][4]: d.length() == 2", 2},
            {9, "d[1].length() == 3", 3},
            {10, "d[1][2].length() == 4", 4},
            {11, "length() is a constant expression: int q[d[0][0].length() * 2] has length 8", 8},
            {12, "a[1][2] (float, name-first form)", 5},
            {13, "b[1][2] (mixed form indexes [2][3])", 12},
            {14, "c[1][2] (type-first form indexes [2][3])", 102},
            {15, "w[2][1].z", 5},
            {16, "d[sel[0]][sel[1]][sel[2]] with run-time indices (1, 2, 3)", 123},
            {17, "const int k[2][3] = int[2][3](...): k[1][2]", 6},
            {18, "sum23(k): array of arrays passed by value", 21},
            {19, "int m[][3] = int[][3](3 rows): outer size inferred, m.length() == 3", 3},
            {20, "m[2][0]", 13},
            {21, "int[][](int[](...), ...): sizes 2 and 3 inferred", 23},
            {22, "n[1][0]", 4},
            {23, "whole-array assignment e = k, e == k", 1},
            {24, "after e[sel[3]][sel[0]] = 0, e != k", 1},
            {25, "inout int[2][3] parameter scaled by 3, sum", 63},
            {26, "int[3][2] return type: transpose(k)[2][0] == k[0][2]", 3},
            {27, "transpose(k)[0][1] == k[1][0]", 4},
            {28, "returned int[3][2]: length() 3 and [0].length() 2", 32},
            {29, "Cell cells[2][2] (struct with int v[2]): cells[1][0].v[1]", 5},
            {30, "cells.length(), cells[0].length(), cells[0][0].v.length()", 222},
            {31, "SSBO int src[2][3][4] read (sum of src[1])", sum},
            {32, "SSBO src[sel[0]][sel[1]][sel[2]]", src_value(1, 2, 3)},
            {33, "SSBO src lengths 2, 3, 4", 234},
            {34, "SSBO uint rt[][3]: run-time length() == 5 (60 bytes / 12)", 5},
            {35, "SSBO rt[0].length() == 3", 3},
            {36, "SSBO grid[3][4] lengths", 34},
            {37, "SSBO pairs[2][2] lengths", 22},
            {38, "int[2][2](...)[1][0] (constructor indexed directly)", 3},
            {39, "array-of-array equality of different arrays is false", 0},
        };
    }

    constexpr int kAoaResults = 48;
    constexpr int kRtRows = 5;
    constexpr std::size_t kGridBytes = 80 + kRtRows * 12;  // std430: grid 48 B, pairs 32 B at 48, rt at 80

    class Glsl310Builtins final : public demo::Demo
    {
    public:
        void init(demo::Env& /*env*/) override
        {
            cases_ = make_cases(pack_rejected_);
            mgl::glGenBuffers(1, &in_);
            mgl::glGenBuffers(1, &out_);
            mgl::glGenBuffers(1, &grid_);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            std::vector<std::vector<U16>> outputs;
            for (const auto& cs : cases_) outputs.push_back(run_case(cs, check));
            negative_controls(outputs, check);
            arrays_of_arrays(env, check);
            compile_negatives(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, i, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            mgl::glDeleteBuffers(1, &in_);
            mgl::glDeleteBuffers(1, &out_);
            mgl::glDeleteBuffers(1, &grid_);
        }

    private:
        static mgl::ProgramId build_compute(const std::string& source, std::string& log)
        {
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, source, log);
            if (cs.value == 0) return mgl::ProgramId{0};
            const mgl::ShaderId shaders[] = {cs};
            const mgl::ProgramId program = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            return program;
        }

        template <typename T>
        static bool read_back(mgl::BufferId buffer, std::vector<T>& out)
        {
            const auto bytes = static_cast<GLsizeiptr>(out.size() * sizeof(T));
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, buffer);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, bytes, mgl::MapBufferAccessMask::Read);
            if (mapped == nullptr) return false;
            std::memcpy(out.data(), mapped, static_cast<std::size_t>(bytes));
            (void)mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage);
            return true;
        }

        std::vector<U16> run_case(const Case& cs, demo::Check& check)
        {
            std::string log;
            const mgl::ProgramId program = build_compute(compute_source(cs.body), log);
            if (program.value == 0)
            {
                check.expect(false, std::format("{}: compiles and links", cs.name), log);
                return {};
            }

            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, in_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const U16>(cs.rows), mgl::BufferUsage::StaticDraw);
            const std::vector<U16> init(cs.rows.size(), [] { U16 s; s.fill(kSentinel); return s; }());
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, out_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const U16>(init), mgl::BufferUsage::DynamicRead);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, in_);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, out_);

            mgl::glUseProgram(program);
            mgl::glDispatchCompute(static_cast<GLuint>((cs.rows.size() + kLocalSize - 1) / kLocalSize), 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);  // shader writes visible to MapBufferRange
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program);

            std::vector<U16> got(cs.rows.size());
            if (!read_back(out_, got))
            {
                check.expect(false, std::format("{}: output buffer mapped", cs.name));
                return {};
            }

            int bad = 0, compared = 0;
            double worst_rel = 0.0;
            std::string first;
            for (std::size_t row = 0; row < cs.rows.size(); ++row)
            {
                const U16 ref = cs.ref(cs.rows[row]);
                for (int k = 0; k < cs.components; ++k)
                {
                    const auto ku = static_cast<std::size_t>(k);
                    const std::uint32_t g = got[row][ku], e = ref[ku];
                    bool ok = g == e;
                    if (!ok && (cs.float_mask >> k) & 1u)
                    {
                        if ((cs.mediump_mask >> k) & 1u)
                        {
                            const double diff = std::abs(double(f(g)) - double(f(e)));
                            const double rel = diff / std::max(std::abs(double(f(e))), std::numeric_limits<double>::min());
                            ok = diff <= kMediumpRel * std::abs(double(f(e)));
                            if (ok) worst_rel = std::max(worst_rel, rel);
                        }
                        else
                        {
                            ok = (g & 0x7FFFFFFFu) == 0 && (e & 0x7FFFFFFFu) == 0;  // +0 / -0 interchangeable
                        }
                    }
                    ++compared;
                    if (!ok && bad++ == 0)
                        first = std::format("row {} word {}: got 0x{:08X}{} expected 0x{:08X}{} (inputs u0 = {:08X} {:08X} {:08X} {:08X}, u1 = {:08X} {:08X} {:08X} {:08X}, u2 = {:08X} {:08X})",
                                            row, k, g, ((cs.float_mask >> k) & 1u) ? std::format(" ({})", f(g)) : "", e,
                                            ((cs.float_mask >> k) & 1u) ? std::format(" ({})", f(e)) : "", cs.rows[row][0], cs.rows[row][1], cs.rows[row][2],
                                            cs.rows[row][3], cs.rows[row][4], cs.rows[row][5], cs.rows[row][6], cs.rows[row][7], cs.rows[row][8],
                                            cs.rows[row][9]);
                }
            }
            std::string what = std::format("{} [{}; {} rows, {} words]", cs.name, cs.feature, cs.rows.size(), compared);
            if (cs.mediump_mask != 0) what += std::format(" [mediump results: tolerance 2^-10 relative, measured max {:.2e}]", worst_rel);
            check.expect(bad == 0, what, bad == 0 ? std::string{} : std::format("{} mismatches; first: {}", bad, first));
            return got;
        }

        const Case* find_case(std::string_view prefix, std::size_t& index) const
        {
            for (std::size_t i = 0; i < cases_.size(); ++i)
                if (cases_[i].name.starts_with(prefix))
                {
                    index = i;
                    return &cases_[i];
                }
            return nullptr;
        }

        // Each control runs a deliberately wrong CPU model against the GPU output of a passing case
        // and requires at least one disagreement: the exact comparison above would have caught it.
        void negative_controls(const std::vector<std::vector<U16>>& outputs, demo::Check& check)
        {
            auto control = [&](std::string_view case_prefix, std::string_view what, int word_begin, int word_end,
                               const std::function<std::uint32_t(const U16& in, int word)>& wrong) {
                std::size_t idx = 0;
                const Case* cs = find_case(case_prefix, idx);
                if (cs == nullptr || outputs[idx].empty())
                {
                    check.expect(false, std::format("negative control: {}", what), "case output unavailable");
                    return;
                }
                int differ = 0, total = 0;
                for (std::size_t row = 0; row < cs->rows.size(); ++row)
                    for (int k = word_begin; k < word_end; ++k)
                    {
                        ++total;
                        if (wrong(cs->rows[row], k) != outputs[idx][row][static_cast<std::size_t>(k)]) ++differ;
                    }
                check.expect(differ > 0, std::format("negative control: {} disagrees with the GPU ({} of {} words)", what, differ, total));
            };

            control("packUnorm4x8", "packSnorm4x8 with truncation instead of round()", 1, 2,
                    [](const U16& in, int) { return pack_snorm8(lane_floats(in, 0), 127.0, Rounding::Truncate); });
            control("packUnorm4x8", "packUnorm4x8 with truncation instead of round()", 0, 1,
                    [](const U16& in, int) { return pack_unorm8(lane_floats(in, 0), 255.0, Rounding::Truncate); });
            control("packUnorm4x8", "packUnorm4x8 scaled by 256 instead of 255", 0, 1,
                    [](const U16& in, int) { return pack_unorm8(lane_floats(in, 0), 256.0); });
            control("packUnorm4x8", "packSnorm4x8 scaled by 128 instead of 127", 1, 2,
                    [](const U16& in, int) { return pack_snorm8(lane_floats(in, 0), 128.0); });
            control("bitfieldExtract", "signed bitfieldExtract without sign extension", 4, 8,
                    [](const U16& in, int k) { return bfe_u(in[static_cast<std::size_t>(k - 4)], as_i(in[8]), as_i(in[9])); });
            control("findLSB/findMSB", "findMSB(int) treating negative values as unsigned", 12, 16,
                    [](const U16& in, int k) { return find_msb_u(in[static_cast<std::size_t>(k - 12)]); });
            control("frexp", "frexp with a [1, 2) significand", 4, 8, [](const U16& in, int k) {
                int e = 0;
                (void)std::frexp(f(in[static_cast<std::size_t>(k - 4)]), &e);
                return as_u(f(in[static_cast<std::size_t>(k - 4)]) == 0.0f ? 0 : e - 1);
            });

            // unpack: the 2^-10 tolerance still rejects a /256 implementation for every nonzero byte.
            std::size_t idx = 0;
            if (const Case* cs = find_case("unpackUnorm4x8", idx); cs != nullptr && !outputs[idx].empty())
            {
                int rejected = 0, nonzero = 0;
                for (std::size_t row = 0; row < cs->rows.size(); ++row)
                    for (int k = 0; k < 4; ++k)
                    {
                        const std::uint32_t byte = (cs->rows[row][0] >> (8 * k)) & 0xFFu;
                        if (byte == 0) continue;
                        ++nonzero;
                        const double wrong = byte / 256.0;
                        const double gpu = f(outputs[idx][row][static_cast<std::size_t>(k)]);
                        if (std::abs(gpu - wrong) > kMediumpRel * wrong) ++rejected;
                    }
                check.expect(rejected == nonzero && nonzero == 255,
                             std::format("negative control: unpackUnorm4x8 modelled as byte / 256 is outside the 2^-10 tolerance for all {} nonzero bytes", nonzero),
                             rejected == nonzero ? std::string{} : std::format("{} of {} rejected", rejected, nonzero));
                // The clamp is exact: -128 / 127 < -1 at any precision, clamp returns exactly -1.0.
                int found = 0, exact = 0;
                for (std::size_t row = 0; row < cs->rows.size(); ++row)
                    for (int k = 0; k < 4; ++k)
                        if (((cs->rows[row][0] >> (8 * k)) & 0xFFu) == 0x80u)
                        {
                            ++found;
                            exact += outputs[idx][row][static_cast<std::size_t>(4 + k)] == b(-1.0f);
                        }
                check.expect(found == 1 && exact == 1, "unpackSnorm4x8(0x80): clamp(-128 / 127, -1, 1) is exactly -1.0");
            }
            // The tie filter itself: 0.5 gives 127.5 (unorm) and 63.5 (snorm), whose rounding direction
            // the spec leaves to the implementation, so it must be excluded.
            check.expect(!pack_input_safe(0.5f) && pack_input_safe(0.25f),
                         std::format("pack input filter excludes .5 ties such as 0.5 and keeps 0.25 ({} random candidates excluded)", pack_rejected_));
        }

        void arrays_of_arrays(demo::Env& env, demo::Check& check)
        {
            std::string log;
            const mgl::ProgramId program = build_compute(env.load_text("shaders/arrays_of_arrays.comp"), log);
            if (program.value == 0)
            {
                check.expect(false, "arrays_of_arrays.comp compiles and links", log);
                return;
            }

            // Source block: int src[2][3][4] (std430 stride 4, inner index fastest) then int sel[4].
            std::vector<std::int32_t> src;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 4; ++k) src.push_back(src_value(i, j, k));
            for (int s : {1, 2, 3, 0}) src.push_back(s);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, in_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const std::int32_t>(src), mgl::BufferUsage::StaticDraw);
            const std::vector<std::int32_t> res_init(kAoaResults, static_cast<std::int32_t>(kSentinel));
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, out_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const std::int32_t>(res_init), mgl::BufferUsage::DynamicRead);
            const std::vector<std::uint32_t> grid_init(kGridBytes / 4, kSentinel);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, grid_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const std::uint32_t>(grid_init), mgl::BufferUsage::DynamicRead);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, in_);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, out_);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 2, grid_);

            mgl::glUseProgram(program);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program);

            std::vector<std::int32_t> res(kAoaResults);
            std::vector<std::uint32_t> grid(kGridBytes / 4);
            if (!read_back(out_, res) || !read_back(grid_, grid))
            {
                check.expect(false, "arrays of arrays: result buffers mapped");
                return;
            }
            for (const auto& e : aoa_expectations())
                check.expect(res[static_cast<std::size_t>(e.index)] == e.value, std::format("arrays of arrays [{}]: {}", e.index, e.label),
                             std::format("got {} expected {}", res[static_cast<std::size_t>(e.index)], e.value));

            // SSBO members written by the shader, read with the §4.1.9 order (inner index fastest).
            int bad = 0, outer_fast_differs = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 4; ++j)
                {
                    const std::uint32_t expect = 0xA000u + static_cast<std::uint32_t>(i * 16 + j);
                    bad += grid[static_cast<std::size_t>(i * 4 + j)] != expect;
                    outer_fast_differs += grid[static_cast<std::size_t>(j * 3 + i)] != expect;
                }
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                {
                    const auto w = static_cast<std::size_t>(12 + (i * 2 + j) * 2);
                    bad += grid[w] != static_cast<std::uint32_t>(i * 2 + j);
                    bad += grid[w + 1] != 0xB000u + static_cast<std::uint32_t>(i * 2 + j);
                }
            for (int i = 0; i < kRtRows; ++i)
                for (int j = 0; j < 3; ++j) bad += grid[static_cast<std::size_t>(20 + i * 3 + j)] != 0xC000u + static_cast<std::uint32_t>(i * 16 + j);
            check.expect(bad == 0,
                         "SSBO members uint grid[3][4], uvec2 pairs[2][2], uint rt[][3] (all 5 run-time rows) written by the shader match the std430 "
                         "array-of-arrays layout (inner index fastest)",
                         bad == 0 ? std::string{} : std::format("{} words differ", bad));
            check.expect(outer_fast_differs > 0,
                         std::format("negative control: reading grid with the outer index fastest disagrees ({} of 12 words)", outer_fast_differs));
        }

        void compile_negatives(demo::Check& check)
        {
            struct Src
            {
                mgl::ShaderType type;
                const char* what;
                const char* source;
                bool compiles;
            };
            const Src sources[] = {
                {mgl::ShaderType::Vertex, "#version 300 es: float a[3] compiles (positive control)",
                 "#version 300 es\nvoid main() { float a[3]; a[0] = 1.0; gl_Position = vec4(a[0]); }\n", true},
                {mgl::ShaderType::Vertex, "#version 300 es: float a[2][3] is a compile error (GLSL ES 3.00 §4.1.9: no arrays of arrays)",
                 "#version 300 es\nvoid main() { float a[2][3]; a[0][0] = 1.0; gl_Position = vec4(a[0][0]); }\n", false},
                {mgl::ShaderType::Fragment, "#version 310 es: out vec4 o[2] compiles (positive control)",
                 "#version 310 es\nprecision mediump float;\nlayout(location = 0) out vec4 o[2];\nvoid main() { o[0] = vec4(1.0); o[1] = vec4(0.0); }\n", true},
                {mgl::ShaderType::Fragment, "#version 310 es: fragment output vec4 o[2][2] is a compile error (GLSL ES 3.10 §4.3.6)",
                 "#version 310 es\nprecision mediump float;\nlayout(location = 0) out vec4 o[2][2];\n"
                 "void main() { o[0][0] = vec4(1.0); o[0][1] = vec4(0.0); o[1][0] = vec4(0.0); o[1][1] = vec4(0.0); }\n",
                 false},
                {mgl::ShaderType::Vertex, "#version 310 es: vertex output float v[2][2] is a compile error (GLSL ES 3.10 §4.3.6)",
                 "#version 310 es\nout float v[2][2];\nvoid main() { v[0][0] = 1.0; v[0][1] = 1.0; v[1][0] = 1.0; v[1][1] = 1.0; gl_Position = vec4(0.0); }\n",
                 false},
            };
            for (const auto& s : sources)
            {
                std::string log;
                const mgl::ShaderId sh = demo::compile_shader(s.type, s.source, log);
                const bool ok = sh.value != 0;
                if (ok) mgl::glDeleteShader(sh);
                check.expect(ok == s.compiles, s.what, ok == s.compiles ? std::string{} : (ok ? std::string{"compiled"} : log));
            }
        }

        std::vector<Case> cases_;
        int pack_rejected_ = 0;
        mgl::BufferId in_{}, out_{}, grid_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "test-glsl310-builtins",
        .title = "GLSL ES 3.10 built-in functions and arrays of arrays",
        .min_es = {3, 1},
        .summary = "Evaluates the GLSL ES 3.10 integer, frexp/ldexp and 4x8 packing built-ins in compute shaders, reads the "
                   "results back from a shader storage buffer and compares them bit-exactly with CPU references; checks "
                   "arrays of arrays (declarations, length(), initializers, functions, SSBO members).",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Glsl310Builtins>(argc, argv, info);
}
