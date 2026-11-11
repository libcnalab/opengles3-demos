// test-glsl320-builtins
//
// Checks three GLSL ES 3.20 additions with compute shaders and buffer readback:
//
//   1. Image atomics (GLSL ES 3.20 §8.12): imageAtomicAdd/Min/Max/And/Or/Xor/Exchange/CompSwap on
//      r32ui and r32i images, imageAtomicExchange on an r32f image, under heavy contention (1024
//      invocations per texel), plus a single-invocation sequence with exact results, invalid
//      (non-existent texel) atomics, and contended adds through the other image types.
//   2. fma() (§8.3, precision rules of §4.7.1).
//   3. The precise qualifier (§4.9).
//
// What is asserted, and why (spec text quoted in README.md):
//   - Atomics: for add/min/max/and/or/xor the final texel is order independent and compared exactly.
//     For every operation, including exchange and compSwap, the returned values must be explained by
//     SOME serial order of the operations: the edges old_i -> op(old_i, data_i) must form one trail
//     from the initial to the final value that uses every edge (an Euler trail). That is exactly the
//     "read, compute, write without intervening modification" guarantee of §8.12. A compSwap counter
//     must hand out every value base..base+N-1 exactly once; exchange results plus the final value
//     must be a permutation of the initial value plus the data.
//   - fma(): §4.7.1 says its precision is "inherited from a * b + c", which is "correctly rounded
//     single operation or sequence of two correctly rounded operations". The rounding mode is
//     undefined, so "correctly rounded" allows either neighbour of the exact value. Results must lie
//     in that set. Single rounding is NOT required. What §8.3 requires for fma() consumed by a precise
//     variable is invariance: the same inputs give the same bits in every such use, also across
//     programs. That is checked bit-exactly with inputs where single and double rounding differ.
//   - precise (§4.9): "a * b + c is not allowed to become a single operation fma(a, b, c)",
//     "a + (b + c) is not allowed to become (a + b) + c", "a * (b * c) is not allowed to become
//     (a * b) * c", and a * b + c * d must equal b * a + c * d = d * c + b * a. Inputs are chosen so
//     that the forbidden transformation produces a value outside the allowed set.
// Negative controls run the checkers on deliberately wrong models (lost updates, duplicated counter
// values, unsigned min on int data, fused results) and require rejection, and compile checks show
// the 3.20-only features are rejected by #version 310 es and the wrong overloads by 320 es.

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
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr std::uint32_t kSeed = 0x320A70u;  // fixed: inputs are identical on every run
    constexpr std::uint32_t kSentinel = 0xDEADBEEFu;

    std::uint32_t as_u(std::int32_t v) { return std::bit_cast<std::uint32_t>(v); }
    std::int32_t as_i(std::uint32_t v) { return std::bit_cast<std::int32_t>(v); }
    float f(std::uint32_t bits) { return std::bit_cast<float>(bits); }
    std::uint32_t b(float v) { return std::bit_cast<std::uint32_t>(v); }

    class Rng
    {
    public:
        std::uint32_t next() { return static_cast<std::uint32_t>(gen_()); }  // raw mt19937 output: portable
        std::uint32_t below(std::uint32_t n) { return next() % n; }

    private:
        std::mt19937 gen_{kSeed};
    };

    // =============================================================================================
    // Part 1: image atomics
    // =============================================================================================
    constexpr std::uint32_t kInvocations = 1024;  // 16 work groups of 64
    constexpr std::uint32_t kLocal = 64;
    constexpr std::size_t kStride = 32;          // words per invocation in Args / Olds
    constexpr std::uint32_t kUBase = 1000;       // counter bases, also in image_atomics.comp
    constexpr std::int32_t kIBase = -512;
    constexpr int kMaxAttempts = 8192;
    constexpr int kTexels = 16;                  // 4 x 4 images

    enum class Op { Add, Min, Max, And, Or, Xor, Exchange, CompSwap };

    // One atomic on a 32-bit word. Signed selects the int (r32i) interpretation of min / max.
    std::uint32_t apply(Op op, bool is_signed, std::uint32_t old, std::uint32_t data, std::uint32_t compare)
    {
        switch (op)
        {
        case Op::Add: return old + data;  // two's complement wrap-around (GLSL ES 3.20 §4.7.1 "low-order 32 bits")
        case Op::Min: return is_signed ? as_u(std::min(as_i(old), as_i(data))) : std::min(old, data);
        case Op::Max: return is_signed ? as_u(std::max(as_i(old), as_i(data))) : std::max(old, data);
        case Op::And: return old & data;
        case Op::Or: return old | data;
        case Op::Xor: return old ^ data;
        case Op::Exchange: return data;
        case Op::CompSwap: return old == compare ? data : old;
        }
        return old;
    }

    struct Slot
    {
        const char* name;
        Op op;
        bool order_independent;  // final value = fold of all data regardless of order
    };

    // Slots 0..8 of both integer images (see image_atomics.comp).
    const std::array<Slot, 9> kSlots = {{
        {"imageAtomicAdd (small data, values chain)", Op::Add, true},
        {"imageAtomicAdd (32-bit data, wraps)", Op::Add, true},
        {"imageAtomicMin", Op::Min, true},
        {"imageAtomicMax", Op::Max, true},
        {"imageAtomicAnd", Op::And, true},
        {"imageAtomicOr", Op::Or, true},
        {"imageAtomicXor", Op::Xor, true},
        {"imageAtomicExchange", Op::Exchange, false},
        {"imageAtomicCompSwap (one attempt per invocation)", Op::CompSwap, false},
    }};

    struct AtomicInputs
    {
        std::vector<std::uint32_t> args;   // kInvocations * kStride
        std::array<std::uint32_t, kTexels> uinit{}, iinit{}, finit{};
        std::array<std::uint32_t, 40> seq_args{};
    };

    AtomicInputs make_atomic_inputs()
    {
        Rng rng;
        AtomicInputs in;
        in.args.assign(kInvocations * kStride, 0u);
        for (std::uint32_t i = 0; i < kInvocations; ++i)
        {
            std::uint32_t* a = &in.args[i * kStride];
            // r32ui data
            a[0] = (i % 7u) + 1u;                    // positive: the values chain upwards
            a[1] = rng.next();                       // the sum wraps many times
            a[2] = rng.next();
            a[3] = rng.next() >> 1;
            a[4] = ~(1u << (i % 24u));               // clears bits 0..23
            a[5] = 1u << (i % 24u);                  // sets bits 0..23
            a[6] = rng.next();
            a[7] = 0x40000000u + i * 7919u;          // distinct values
            a[21] = i % 16u;                         // compSwap: compare
            a[8] = (i % 16u) + 1u;                   //           data
            // r32i data
            a[10] = as_u(-static_cast<std::int32_t>(i % 5u) - 1);  // negative: the values chain downwards
            a[11] = rng.next();
            a[12] = rng.next();                      // half of them negative
            a[13] = rng.next();
            a[14] = ~(1u << (i % 24u + 4u));
            a[15] = 1u << (i % 24u + 4u);
            a[16] = rng.next();
            a[17] = as_u(-1000000 - static_cast<std::int32_t>(i) * 13);
            a[22] = as_u(-8 + static_cast<std::int32_t>(i % 16u));
            a[18] = as_u(-7 + static_cast<std::int32_t>(i % 16u));
            // r32f exchange data: distinct, exact, normal floats of both signs
            const float v = static_cast<float>(i + 1) * 0.375f;
            a[20] = b(i % 2 == 0 ? v : -v);
        }
        for (int t = 0; t < kTexels; ++t)
        {
            const auto k = static_cast<std::size_t>(t);
            in.uinit[k] = 0x5EED0000u + static_cast<std::uint32_t>(t);
            in.iinit[k] = as_u(-0x5EED0 - t);
            in.finit[k] = b(100.0f + static_cast<float>(t));
        }
        in.uinit = {0x100u, 0xFFFFF000u, 0xF0000000u, 0x10u, 0xDEADBEEFu, 0x01000000u, 0x13579BDFu, 0xABCDEF01u, 0u, kUBase,
                    0x0F0F0F0Fu, in.uinit[11], in.uinit[12], in.uinit[13], in.uinit[14], in.uinit[15]};
        in.iinit = {as_u(100), as_u(0x7FFFFFFF - 1000), as_u(1000), as_u(-1000000), as_u(-1), 0u, as_u(12345), as_u(-7777), as_u(-8),
                    as_u(kIBase), as_u(-5), in.iinit[11], in.iinit[12], in.iinit[13], in.iinit[14], in.iinit[15]};
        in.finit[0] = b(0.5f);
        in.finit[1] = b(1.5f);

        auto& s = in.seq_args;
        s = {0xF0F0F0F2u, 0x7FFFFFFFu, 0x7FFFFFFFu, 0x80000001u, 0x7F00FF0Fu, 0x0A0B0C00u, 0xFFFF0000u, 0x12345678u,
             0x12345679u, 0xDEAD0001u, 0x12345678u, 0xCAFEF00Du, 0x55u,
             as_u(-10), 0x80000000u, as_u(-3), as_u(100), as_u(-4), as_u(-256), 0x0F0F0F0Fu, as_u(-123456789),
             as_u(-123456788), 1u, as_u(-123456789), 0x7FFFFFFFu, 7u,
             b(-2.75f), b(std::numeric_limits<float>::max()), b(std::numeric_limits<float>::min()), b(0.1f), b(9.0f)};
        return in;
    }

    // Expected results of the invocation-0 sequence (seq_out[0..32]) and the final texels.
    struct SeqExpect
    {
        std::array<std::uint32_t, 33> out{};
        std::uint32_t ufinal = 0, ifinal = 0, ffinal = 0;
    };

    // swap_compsw: models compSwap with compare and data exchanged (a negative control).
    // unsigned_int_minmax: models min/max on the int texel with unsigned comparison (a negative control).
    SeqExpect expected_sequence(const AtomicInputs& in, bool swap_compsw = false, bool unsigned_int_minmax = false)
    {
        SeqExpect e;
        const auto& s = in.seq_args;
        auto cs = [&](std::uint32_t old, std::uint32_t cmp, std::uint32_t data) {
            return swap_compsw ? apply(Op::CompSwap, false, old, cmp, data) : apply(Op::CompSwap, false, old, data, cmp);
        };
        std::uint32_t u = in.uinit[10];
        const Op ops[] = {Op::Add, Op::Add, Op::Min, Op::Max, Op::And, Op::Or, Op::Xor, Op::Exchange};
        for (std::size_t k = 0; k < 8; ++k)
        {
            e.out[k] = u;
            u = apply(ops[k], false, u, s[k], 0);
        }
        e.out[8] = u;
        u = cs(u, s[8], s[9]);
        e.out[9] = u;
        u = cs(u, s[10], s[11]);
        e.ufinal = u;
        for (std::size_t k = 10; k < 14; ++k) e.out[k] = 0;  // invalid accesses return zero

        std::uint32_t v = in.iinit[10];
        for (std::size_t k = 0; k < 8; ++k)
        {
            e.out[14 + k] = v;
            v = apply(ops[k], !unsigned_int_minmax, v, s[13 + k], 0);
        }
        e.out[22] = v;
        v = cs(v, s[21], s[22]);
        e.out[23] = v;
        v = cs(v, s[23], s[24]);
        e.ifinal = v;
        for (std::size_t k = 24; k < 28; ++k) e.out[k] = 0;

        std::uint32_t w = in.finit[1];
        for (std::size_t k = 0; k < 4; ++k)
        {
            e.out[28 + k] = w;
            w = s[26 + k];
        }
        e.out[32] = 0;
        e.ffinal = w;
        return e;
    }

    struct Edge
    {
        std::uint32_t from, to;
    };

    // True when the edges (old -> new of every atomic on one texel) can be ordered into a single
    // trail starting at `init` and ending at `final`, i.e. some serial execution of the operations
    // reproduces every returned value and the final texel. Adding the virtual edge final -> init
    // turns this into the Euler-circuit condition: every vertex balanced and all edges connected.
    bool serializable(std::uint32_t init, std::uint32_t final_value, std::span<const Edge> edges, std::string& why)
    {
        std::map<std::uint32_t, long> balance;  // out-degree minus in-degree
        std::map<std::uint32_t, std::uint32_t> parent;
        std::function<std::uint32_t(std::uint32_t)> find = [&](std::uint32_t x) {
            auto it = parent.find(x);
            if (it == parent.end())
            {
                parent[x] = x;
                return x;
            }
            if (it->second == x) return x;
            const std::uint32_t r = find(it->second);
            parent[x] = r;
            return r;
        };
        auto unite = [&](std::uint32_t x, std::uint32_t y) { parent[find(x)] = find(y); };
        auto add = [&](std::uint32_t from, std::uint32_t to) {
            ++balance[from];
            --balance[to];
            unite(from, to);
        };
        for (const Edge& e : edges) add(e.from, e.to);
        add(final_value, init);
        for (const auto& [vertex, bal] : balance)
            if (bal != 0)
            {
                why = bal > 0 ? std::format("value 0x{:08X} is returned {} time(s) more often than a serial order can produce it", vertex, bal)
                              : std::format("value 0x{:08X} is produced {} time(s) more often than it is returned", vertex, -bal);
                return false;
            }
        const std::uint32_t root = find(init);
        for (const auto& entry : balance)
            if (find(entry.first) != root)
            {
                why = std::format("value 0x{:08X} is disconnected from the initial value", entry.first);
                return false;
            }
        return true;
    }

    // =============================================================================================
    // Part 2: fma() and precise
    // =============================================================================================
    using U16 = std::array<std::uint32_t, 16>;  // one input row: u0..u3 (a, b, c, d lanes)
    using U20 = std::array<std::uint32_t, 20>;  // one output row: r0..r3 + the precise block pv[row]
    constexpr GLuint kRowsPerGroup = 64;

    // Neighbours of an exact value (RD, RU); equal when x is representable. x must be exact in double.
    std::vector<float> bracket(double x)
    {
        const auto fx = static_cast<float>(x);
        if (static_cast<double>(fx) == x) return {fx};
        if (static_cast<double>(fx) < x) return {fx, std::nextafter(fx, std::numeric_limits<float>::infinity())};
        return {std::nextafter(fx, -std::numeric_limits<float>::infinity()), fx};
    }

    // x + y if it is exact in double (TwoSum error is zero).
    bool exact_sum(double x, double y, double& s)
    {
        s = x + y;
        const double bb = s - x;
        const double err = (x - (s - bb)) + (y - bb);
        return err == 0.0;
    }

    bool normal_nonzero(float v) { return std::isnormal(v); }

    bool contains(const std::vector<float>& set, float v)
    {
        return std::any_of(set.begin(), set.end(), [&](float s) { return b(s) == b(v); });
    }

    bool disjoint(const std::vector<float>& x, const std::vector<float>& y)
    {
        return std::none_of(x.begin(), x.end(), [&](float v) { return contains(y, v); });
    }

    std::vector<float> merge(std::vector<float> x, const std::vector<float>& y)
    {
        for (float v : y)
            if (!contains(x, v)) x.push_back(v);
        return x;
    }

    // Allowed results of a*b+c (§4.7.1). fused: one correctly rounded operation; twice: a*b rounded,
    // then + c rounded. "Correctly rounded" with an undefined rounding mode = either neighbour.
    struct Triple
    {
        float a, b, c;
        std::vector<float> fused, twice;
        float fused_rn = 0.0f, twice_rn = 0.0f;  // round-to-nearest-even values (reported, not required)
    };

    bool analyse(Triple& t)
    {
        const double ab = double(t.a) * double(t.b);  // exact: 48-bit significand
        double exact = 0.0;
        if (!exact_sum(ab, double(t.c), exact)) return false;
        t.fused = bracket(exact);
        t.twice.clear();
        for (float p : bracket(ab))
        {
            double s = 0.0;
            if (!exact_sum(double(p), double(t.c), s)) return false;
            t.twice = merge(t.twice, bracket(s));
        }
        t.fused_rn = static_cast<float>(exact);
        t.twice_rn = static_cast<float>(double(static_cast<float>(ab)) + double(t.c));  // the sum of two floats here is exact (checked above)
        for (float v : merge(t.fused, t.twice))
            if (!normal_nonzero(v)) return false;
        return true;
    }

    // Allowed results of precise a*b + c*d: each product rounded, the sum rounded.
    struct Quad
    {
        float a, b, c, d;
        std::vector<float> allowed;
        std::vector<float> contracted_ab, contracted_cd;  // fma(a,b,RN(c*d)) and fma(c,d,RN(a*b)) neighbours
    };

    bool analyse(Quad& q)
    {
        const double ab = double(q.a) * double(q.b), cd = double(q.c) * double(q.d);
        q.allowed.clear();
        for (float p1 : bracket(ab))
            for (float p2 : bracket(cd))
            {
                double s = 0.0;
                if (!exact_sum(double(p1), double(p2), s)) return false;
                q.allowed = merge(q.allowed, bracket(s));
            }
        double s1 = 0.0, s2 = 0.0;
        if (!exact_sum(ab, double(static_cast<float>(cd)), s1) || !exact_sum(cd, double(static_cast<float>(ab)), s2)) return false;
        q.contracted_ab = bracket(s1);
        q.contracted_cd = bracket(s2);
        for (float v : merge(merge(q.allowed, q.contracted_ab), q.contracted_cd))
            if (!normal_nonzero(v)) return false;
        return true;
    }

    // Random float with `bits` significant bits, exponent in [emin, emax], random sign.
    float random_float(Rng& rng, int bits, int emin, int emax)
    {
        const std::uint32_t m = (1u << (bits - 1)) | (rng.next() & ((1u << (bits - 1)) - 1u));
        const int e = emin + static_cast<int>(rng.below(static_cast<std::uint32_t>(emax - emin + 1)));
        const float v = std::ldexp(static_cast<float>(m), e - (bits - 1));
        return (rng.next() & 1u) ? -v : v;
    }

    float step_ulps(float v, int n)
    {
        for (int k = 0; k < std::abs(n); ++k) v = std::nextafter(v, n > 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
        return v;
    }

    // a*b+c exact in float: then every allowed set is the single exact value.
    std::vector<Triple> exact_triples(Rng& rng, std::size_t count)
    {
        std::vector<Triple> out;
        out.push_back({3.0f, 5.0f, -7.0f, {}, {}});
        out.push_back({-1.5f, 2.25f, 0.125f, {}, {}});
        out.push_back({1024.0f, 0.75f, -768.5f, {}, {}});
        out.push_back({-0.0625f, -16.0f, 2.0f, {}, {}});
        while (out.size() < count)
        {
            Triple t{random_float(rng, 12, -6, 6), random_float(rng, 12, -6, 6), random_float(rng, 12, -6, 6), {}, {}};
            if (analyse(t) && t.fused.size() == 1 && t.twice.size() == 1 && b(t.fused[0]) == b(t.twice[0])) out.push_back(t);
        }
        for (auto& t : out)
            if (!analyse(t) || t.fused.size() != 1 || t.twice.size() != 1) throw demo::Failure("exact fma inputs are not exact");
        return out;
    }

    // Single and double rounding disagree for every rounding mode: fused set and twice set disjoint.
    std::vector<Triple> discriminating_triples(Rng& rng, std::size_t count, int& tried)
    {
        std::vector<Triple> out;
        tried = 0;
        while (out.size() < count)
        {
            if (++tried > 1000000) throw demo::Failure("could not generate discriminating fma inputs");
            Triple t{random_float(rng, 24, -3, 3), random_float(rng, 24, -3, 3), 0.0f, {}, {}};
            const auto p = static_cast<float>(double(t.a) * double(t.b));
            if (rng.below(4) != 0)
            {
                // c cancels the rounded product up to a few ulps: the product's rounding error decides.
                const int m = static_cast<int>(rng.below(7)) - 3;
                t.c = -step_ulps(p, m);
            }
            else
            {
                int e = 0;
                (void)std::frexp(p, &e);
                t.c = random_float(rng, 24, e - 1 - static_cast<int>(rng.below(6)), e - 1);
            }
            if (analyse(t) && disjoint(t.fused, t.twice)) out.push_back(t);
        }
        return out;
    }

    // a*b + c*d where both one-sided contractions fall outside the allowed (two products, one sum) set.
    std::vector<Quad> discriminating_quads(Rng& rng, std::size_t count, int& tried)
    {
        std::vector<Quad> out;
        tried = 0;
        while (out.size() < count)
        {
            if (++tried > 1000000) throw demo::Failure("could not generate discriminating a*b+c*d inputs");
            Quad q{random_float(rng, 24, -2, 2), random_float(rng, 24, -2, 2), 0.0f, random_float(rng, 24, -2, 2), {}, {}, {}};
            const auto p = static_cast<float>(double(q.a) * double(q.b));
            const auto c0 = static_cast<float>(double(p) / double(q.d));
            q.c = -step_ulps(c0, static_cast<int>(rng.below(5)) - 2);
            if (analyse(q) && disjoint(q.contracted_ab, q.allowed) && disjoint(q.contracted_cd, q.allowed)) out.push_back(q);
        }
        return out;
    }

    template <typename T, typename Fn>
    std::vector<U16> pack_rows(const std::vector<T>& items, Fn lanes)
    {
        std::vector<U16> rows((items.size() + 3) / 4, U16{});
        for (std::size_t k = 0; k < items.size(); ++k)
        {
            const std::array<float, 4> v = lanes(items[k]);
            for (std::size_t j = 0; j < 4; ++j) rows[k / 4][4 * j + k % 4] = b(v[j]);
        }
        return rows;
    }

    std::string compute_source(const std::string& decls, const std::string& body)
    {
        return std::format(
            "#version 320 es\n"
            "layout(local_size_x = {0}) in;\n"
            "precision highp float;\nprecision highp int;\n"
            "layout(std430, binding = 0) readonly buffer Inputs {{ uvec4 inp[]; }};\n"
            "layout(std430, binding = 1) writeonly buffer Outputs {{ uvec4 outp[]; }};\n"
            "{1}\n"
            "void main()\n{{\n"
            "    uint row = gl_GlobalInvocationID.x;\n"
            "    if (row >= uint(inp.length()) / 4u) return;\n"
            "    uvec4 u0 = inp[4u * row + 0u], u1 = inp[4u * row + 1u], u2 = inp[4u * row + 2u], u3 = inp[4u * row + 3u];\n"
            "    vec4 f0 = uintBitsToFloat(u0), f1 = uintBitsToFloat(u1), f2 = uintBitsToFloat(u2), f3 = uintBitsToFloat(u3);\n"
            "    uvec4 r0 = uvec4(0u), r1 = uvec4(0u), r2 = uvec4(0u), r3 = uvec4(0u);\n"
            "    {2}\n"
            "    outp[4u * row + 0u] = r0; outp[4u * row + 1u] = r1; outp[4u * row + 2u] = r2; outp[4u * row + 3u] = r3;\n"
            "}}\n",
            kRowsPerGroup, decls, body);
    }

    // =============================================================================================
    // The test
    // =============================================================================================
    class Glsl320Builtins final : public demo::Demo
    {
    public:
        void init(demo::Env& /*env*/) override
        {
            atomics_ = make_atomic_inputs();
            Rng rng;
            exact_ = exact_triples(rng, 256);
            disc_ = discriminating_triples(rng, 512, disc_tried_);
            quads_ = discriminating_quads(rng, 512, quad_tried_);
            mgl::glGenBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            mgl::glGenTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            image_atomics(env, check);
            image_targets(check);
            fma_and_precise(check);
            compile_checks(check);
            check.expect_no_gl_error("no GL error during the test");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 4; ++i) mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, i, mgl::BufferId{0});
            for (GLuint u = 0; u < 3; ++u)
                mgl::glBindImageTexture(mgl::ImageUnit{u}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
        }

    private:
        // ---------------------------------------------------------------- helpers
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
        static void upload(mgl::BufferId buffer, std::span<const T> data, GLuint binding)
        {
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, buffer);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, data, mgl::BufferUsage::DynamicRead);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, binding, buffer);
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

        mgl::TextureId make_image_2d(std::size_t index, mgl::SizedInternalFormat format, mgl::PixelFormat pf, mgl::PixelType type,
                                     const std::array<std::uint32_t, kTexels>& texels)
        {
            const mgl::TextureId tex = textures_[index];
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, 4, 4);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, pf, type, texels.data());
            return tex;
        }

        // ---------------------------------------------------------------- part 1: 2D image atomics
        void image_atomics(demo::Env& env, demo::Check& check)
        {
            const AtomicInputs& in = atomics_;
            const mgl::TextureId utex = make_image_2d(0, mgl::SizedInternalFormat::R32UI, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, in.uinit);
            const mgl::TextureId itex = make_image_2d(1, mgl::SizedInternalFormat::R32I, mgl::PixelFormat::RedInteger, mgl::PixelType::Int, in.iinit);
            const mgl::TextureId ftex = make_image_2d(2, mgl::SizedInternalFormat::R32F, mgl::PixelFormat::Red, mgl::PixelType::Float, in.finit);
            mgl::glBindImageTexture(mgl::ImageUnit{0}, utex, 0, GL_FALSE, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32UI);
            mgl::glBindImageTexture(mgl::ImageUnit{1}, itex, 0, GL_FALSE, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32I);
            mgl::glBindImageTexture(mgl::ImageUnit{2}, ftex, 0, GL_FALSE, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32F);

            std::string log;
            const mgl::ProgramId atomics = build_compute(env.load_text("shaders/image_atomics.comp"), log);
            if (atomics.value == 0)
            {
                check.expect(false, "image_atomics.comp compiles and links (r32ui / r32i / r32f image atomics, #version 320 es)", log);
                return;
            }
            const mgl::ProgramId dump = build_compute(env.load_text("shaders/image_dump.comp"), log);
            if (dump.value == 0)
            {
                mgl::glDeleteProgram(atomics);
                check.expect(false, "image_dump.comp compiles and links", log);
                return;
            }

            const std::vector<std::uint32_t> olds_init(kInvocations * kStride, kSentinel);
            const std::vector<std::uint32_t> seq_init(40, kSentinel);
            upload(buffers_[0], std::span<const std::uint32_t>(in.args), 0);
            upload(buffers_[1], std::span<const std::uint32_t>(olds_init), 1);
            upload(buffers_[2], std::span<const std::uint32_t>(in.seq_args), 2);
            upload(buffers_[3], std::span<const std::uint32_t>(seq_init), 3);
            mgl::glUseProgram(atomics);
            mgl::glDispatchCompute(kInvocations / kLocal, 1, 1);
            // The atomics' image writes must be visible to the imageLoad of the next dispatch, and the
            // buffer writes to glMapBufferRange (ES 3.2 §7.11.2).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderImageAccess | mgl::MemoryBarrierMask::BufferUpdate);

            const std::vector<std::uint32_t> dump_init(kTexels * 4, kSentinel);
            upload(buffers_[4], std::span<const std::uint32_t>(dump_init), 0);
            mgl::glUseProgram(dump);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(atomics);
            mgl::glDeleteProgram(dump);

            std::vector<std::uint32_t> olds(kInvocations * kStride), seq(40), texels(kTexels * 4);
            if (!read_back(buffers_[1], olds) || !read_back(buffers_[3], seq) || !read_back(buffers_[4], texels))
            {
                check.expect(false, "image atomics: result buffers mapped");
                return;
            }
            auto final_u = [&](int t) { return texels[static_cast<std::size_t>(t) * 4]; };
            auto final_i = [&](int t) { return texels[static_cast<std::size_t>(t) * 4 + 1]; };
            auto final_f = [&](int t) { return texels[static_cast<std::size_t>(t) * 4 + 2]; };
            auto arg = [&](std::uint32_t i, std::size_t w) { return in.args[i * kStride + w]; };
            auto old = [&](std::uint32_t i, std::size_t w) { return olds[i * kStride + w]; };

            // ---- contended slots 0..8, both integer formats
            for (int type = 0; type < 2; ++type)
            {
                const bool is_signed = type == 1;
                const char* fmt = is_signed ? "r32i" : "r32ui";
                const std::size_t base = is_signed ? 10 : 0;
                for (std::size_t s = 0; s < kSlots.size(); ++s)
                {
                    const Slot& slot = kSlots[s];
                    const std::uint32_t init = is_signed ? in.iinit[s] : in.uinit[s];
                    const std::uint32_t fin = is_signed ? final_i(static_cast<int>(s)) : final_u(static_cast<int>(s));
                    std::vector<Edge> edges;
                    std::uint32_t fold = init;
                    for (std::uint32_t i = 0; i < kInvocations; ++i)
                    {
                        const std::uint32_t data = arg(i, base + s);
                        const std::uint32_t cmp = arg(i, is_signed ? 22 : 21);
                        const std::uint32_t o = old(i, base + s);
                        edges.push_back({o, apply(slot.op, is_signed, o, data, cmp)});
                        fold = apply(slot.op, is_signed, fold, data, cmp);
                    }
                    if (slot.order_independent)
                        check.expect(fin == fold, std::format("{} {} [GLSL320.FN.IMAGE_ATOMIC]: final texel after {} contending invocations equals the order-independent result 0x{:08X}", fmt, slot.name, kInvocations, fold),
                                     std::format("got 0x{:08X}", fin));
                    std::string why;
                    const bool ok = serializable(init, fin, edges, why);
                    check.expect(ok, std::format("{} {}: the {} returned values and the final texel 0x{:08X} are explained by one serial order of the atomics", fmt, slot.name, kInvocations, fin), why);
                    if (s == 7)  // exchange: returned values + final == initial value + data (as multisets)
                    {
                        std::vector<std::uint32_t> got{fin}, want{init};
                        for (std::uint32_t i = 0; i < kInvocations; ++i)
                        {
                            got.push_back(old(i, base + s));
                            want.push_back(arg(i, base + s));
                        }
                        std::sort(got.begin(), got.end());
                        std::sort(want.begin(), want.end());
                        check.expect(got == want, std::format("{} imageAtomicExchange: returned values plus the final texel are a permutation of the initial value plus the {} data values", fmt, kInvocations));
                    }
                }

                // ---- compSwap counter (slot 9)
                std::vector<std::uint32_t> got;
                std::uint32_t max_attempts = 0;
                for (std::uint32_t i = 0; i < kInvocations; ++i)
                {
                    got.push_back(old(i, base + 9));
                    max_attempts = std::max(max_attempts, old(i, is_signed ? 24 : 23));
                }
                std::vector<std::uint32_t> want;
                for (std::uint32_t k = 0; k < kInvocations; ++k) want.push_back(is_signed ? as_u(kIBase + static_cast<std::int32_t>(k)) : kUBase + k);
                std::sort(got.begin(), got.end());
                std::sort(want.begin(), want.end());
                const std::uint32_t want_final = is_signed ? as_u(kIBase + static_cast<std::int32_t>(kInvocations)) : kUBase + kInvocations;
                const std::uint32_t fin9 = is_signed ? final_i(9) : final_u(9);
                check.expect(got == want && fin9 == want_final && max_attempts <= static_cast<std::uint32_t>(kMaxAttempts),
                             std::format("{} imageAtomicCompSwap counter (retry until the swap succeeds): the {} invocations obtained each value {}..{} exactly once and the final texel is {} "
                                         "(contention: up to {} attempts per invocation)",
                                         fmt, kInvocations, is_signed ? kIBase : static_cast<std::int64_t>(kUBase),
                                         is_signed ? kIBase + static_cast<std::int32_t>(kInvocations) - 1 : static_cast<std::int64_t>(kUBase + kInvocations - 1),
                                         is_signed ? std::to_string(as_i(want_final)) : std::to_string(want_final), max_attempts),
                             std::format("final {} attempts {}", fin9, max_attempts));
            }

            // ---- r32f contended exchange
            {
                std::vector<Edge> edges;
                std::vector<std::uint32_t> got{final_f(0)}, want{in.finit[0]};
                for (std::uint32_t i = 0; i < kInvocations; ++i)
                {
                    edges.push_back({old(i, 20), arg(i, 20)});
                    got.push_back(old(i, 20));
                    want.push_back(arg(i, 20));
                }
                std::string why;
                const bool ser = serializable(in.finit[0], final_f(0), edges, why);
                std::sort(got.begin(), got.end());
                std::sort(want.begin(), want.end());
                check.expect(ser && got == want,
                             std::format("r32f imageAtomicExchange [GLSL320.FN.IMAGE_ATOMIC]: {} contending exchanges return bit-exact floats; returned values plus the final texel are a permutation "
                                         "of the initial value plus the data and form one serial order",
                                         kInvocations),
                             why);
            }

            // ---- invocation-0 sequence: exact results
            const SeqExpect e = expected_sequence(in);
            auto seq_ok = [&](std::size_t from, std::size_t to) {
                for (std::size_t k = from; k < to; ++k)
                    if (seq[k] != e.out[k]) return std::format("seq_out[{}] = 0x{:08X}, expected 0x{:08X}", k, seq[k], e.out[k]);
                return std::string{};
            };
            {
                const std::string d = seq_ok(0, 10);
                check.expect(d.empty() && final_u(10) == e.ufinal,
                             "r32ui single-invocation sequence add (wraps), add, min, max (unsigned order), and, or, xor, exchange, compSwap (compare differs: no store), compSwap (compare equal: store): "
                             "every returned value and the final texel exact",
                             d.empty() ? std::format("final 0x{:08X} expected 0x{:08X}", final_u(10), e.ufinal) : d);
            }
            {
                const std::string d = seq_ok(14, 24);
                check.expect(d.empty() && final_i(10) == e.ifinal,
                             "r32i single-invocation sequence add, add (wraps), min / max (signed order), and, or, xor, exchange, compSwap (no store / store): every returned value and the final texel exact",
                             d.empty() ? std::format("final 0x{:08X} expected 0x{:08X}", final_i(10), e.ifinal) : d);
            }
            {
                const std::string d = seq_ok(28, 32);
                check.expect(d.empty() && final_f(1) == e.ffinal,
                             "r32f single-invocation imageAtomicExchange chain 1.5 -> -2.75 -> FLT_MAX -> FLT_MIN -> 0.1: returned values and the final texel bit-exact",
                             d.empty() ? std::format("final 0x{:08X}", final_f(1)) : d);
            }
            {
                const std::string d = seq_ok(10, 14) + seq_ok(24, 28);
                const bool fzero = (seq[32] & 0x7FFFFFFFu) == 0;  // the float zero may carry either sign
                bool untouched = true;
                for (int t = 11; t < kTexels; ++t)
                    untouched = untouched && final_u(t) == in.uinit[static_cast<std::size_t>(t)] && final_i(t) == in.iinit[static_cast<std::size_t>(t)];
                for (int t = 2; t < kTexels; ++t) untouched = untouched && final_f(t) == in.finit[static_cast<std::size_t>(t)];
                const bool ok = d.empty() && fzero && untouched;
                check.expect(ok,
                             "ES 3.2 §8.23 invalid image atomics (texel (4,0), (0,4), (-1,1), (4,4) does not exist) return zero for add/min/max/exchange/compSwap on r32ui, r32i and r32f "
                             "and update no texel (texels never addressed keep their values)",
                             ok ? std::string{} : !d.empty() ? d : !fzero ? std::format("float invalid exchange returned 0x{:08X}", seq[32]) : std::string{"an unaddressed texel changed"});
            }

            atomic_negative_controls(in, olds, texels, seq, check);
        }

        void atomic_negative_controls(const AtomicInputs& in, const std::vector<std::uint32_t>& olds, const std::vector<std::uint32_t>& texels,
                                      const std::vector<std::uint32_t>& seq, demo::Check& check)
        {
            auto old = [&](std::uint32_t i, std::size_t w) { return olds[i * kStride + w]; };
            auto arg = [&](std::uint32_t i, std::size_t w) { return in.args[i * kStride + w]; };

            // 1. Lost update: two invocations read the same value (a non-atomic read-modify-write).
            {
                std::vector<Edge> edges;
                std::uint32_t lo = 0, hi = 0;  // the invocations that saw the smallest and second smallest value
                std::vector<std::pair<std::uint32_t, std::uint32_t>> by_old;
                for (std::uint32_t i = 0; i < kInvocations; ++i) by_old.emplace_back(old(i, 0), i);
                std::sort(by_old.begin(), by_old.end());
                lo = by_old[0].second;
                hi = by_old[1].second;
                std::uint32_t final_lost = in.uinit[0];
                for (std::uint32_t i = 0; i < kInvocations; ++i)
                {
                    const std::uint32_t o = i == hi ? old(lo, 0) : old(i, 0);
                    edges.push_back({o, o + arg(i, 0)});
                    final_lost += arg(i, 0);
                }
                final_lost -= arg(lo, 0);  // the overwritten update is lost
                std::string why;
                check.expect(!serializable(in.uinit[0], final_lost, edges, why),
                             "negative control: the serial-order checker rejects a lost update (two invocations of imageAtomicAdd return the same value)", why);
            }
            // 2. Exchange modelled as "store, then return the new value": rejected.
            {
                std::vector<Edge> edges;
                for (std::uint32_t i = 0; i < kInvocations; ++i) edges.push_back({arg(i, 7), arg(i, 7)});
                std::string why;
                check.expect(!serializable(in.uinit[7], texels[7 * 4], edges, why),
                             "negative control: the serial-order checker rejects an imageAtomicExchange that returns the new value instead of the old one");
            }
            // 3. Counter with one duplicated value.
            {
                std::vector<std::uint32_t> got;
                for (std::uint32_t i = 0; i < kInvocations; ++i) got.push_back(old(i, 9));
                got[1] = got[0];
                std::sort(got.begin(), got.end());
                bool perm = true;
                for (std::uint32_t k = 0; k < kInvocations; ++k) perm = perm && got[k] == kUBase + k;
                check.expect(!perm, "negative control: the counter check rejects a duplicated compSwap result");
            }
            // 4. Signed min / max folded with unsigned order.
            {
                std::uint32_t umin = in.iinit[2], umax = in.iinit[3];
                for (std::uint32_t i = 0; i < kInvocations; ++i)
                {
                    umin = apply(Op::Min, false, umin, arg(i, 12), 0);
                    umax = apply(Op::Max, false, umax, arg(i, 13), 0);
                }
                check.expect(umin != texels[2 * 4 + 1] && umax != texels[3 * 4 + 1],
                             std::format("negative control: r32i min / max computed with unsigned order (0x{:08X} / 0x{:08X}) disagree with the GPU (0x{:08X} / 0x{:08X})", umin, umax,
                                         texels[2 * 4 + 1], texels[3 * 4 + 1]));
            }
            // 5. Sequence models with compare/data exchanged, or unsigned int min/max.
            {
                const SeqExpect swapped = expected_sequence(in, true, false);
                const SeqExpect unsig = expected_sequence(in, false, true);
                bool swapped_differs = swapped.ufinal != texels[10 * 4] || swapped.ifinal != texels[10 * 4 + 1];
                bool unsig_differs = unsig.ifinal != texels[10 * 4 + 1];
                for (std::size_t k = 0; k < 33; ++k)
                {
                    swapped_differs = swapped_differs || swapped.out[k] != seq[k];
                    unsig_differs = unsig_differs || unsig.out[k] != seq[k];
                }
                check.expect(swapped_differs && unsig_differs,
                             "negative control: sequence models with imageAtomicCompSwap compare/data swapped, and with unsigned min/max on r32i, disagree with the GPU");
            }
        }

        // ---------------------------------------------------------------- part 1b: other image types
        void image_targets(demo::Check& check)
        {
            struct Kind
            {
                const char* type;
                mgl::TextureTarget target;
                int w, h, layers;   // layers: array layers, depth, 6 faces, 6*cubes, or 1
                std::array<int, 3> coord;
                bool layered;
            };
            const Kind kinds[] = {
                {"uimage2DArray", mgl::TextureTarget::Texture2DArray, 2, 2, 3, {1, 0, 2}, true},
                {"uimage3D", mgl::TextureTarget::Texture3D, 2, 2, 2, {0, 1, 1}, true},
                {"uimageCube", mgl::TextureTarget::TextureCubeMap, 2, 2, 6, {1, 1, 4}, true},
                {"uimageCubeArray", mgl::TextureTarget::TextureCubeMapArray, 2, 2, 12, {0, 1, 9}, true},
                {"uimageBuffer", mgl::TextureTarget::TextureBuffer, 8, 1, 1, {5, 0, 0}, false},
            };
            std::size_t tex_index = 3;
            for (const Kind& k : kinds)
            {
                const int count = k.w * k.h * k.layers;
                std::vector<std::uint32_t> init(static_cast<std::size_t>(count));
                for (int t = 0; t < count; ++t) init[static_cast<std::size_t>(t)] = 0x7000u * static_cast<std::uint32_t>(tex_index) + 3u * static_cast<std::uint32_t>(t);
                const mgl::TextureId tex = textures_[tex_index++];
                const bool is_buffer = k.target == mgl::TextureTarget::TextureBuffer;
                mgl::glBindTexture(k.target, tex);
                if (is_buffer)
                {
                    mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, buffers_[5]);
                    mgl::glBufferData(mgl::BufferTarget::TextureBuffer, std::span<const std::uint32_t>(init), mgl::BufferUsage::DynamicDraw);
                    mgl::glTexBuffer(mgl::TextureTarget::TextureBuffer, mgl::InternalFormat::R32UI, buffers_[5]);
                }
                else
                {
                    if (k.target == mgl::TextureTarget::TextureCubeMap)
                    {
                        mgl::glTexStorage2D(k.target, 1, mgl::SizedInternalFormat::R32UI, k.w, k.h);
                        const mgl::TextureTarget faces[] = {mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
                                                            mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
                                                            mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
                        for (std::size_t face = 0; face < 6; ++face)
                            mgl::glTexSubImage2D(faces[face], 0, 0, 0, k.w, k.h, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt,
                                                 init.data() + face * static_cast<std::size_t>(k.w * k.h));
                    }
                    else
                    {
                        mgl::glTexStorage3D(k.target, 1, mgl::SizedInternalFormat::R32UI, k.w, k.h, k.layers);
                        mgl::glTexSubImage3D(k.target, 0, 0, 0, 0, k.w, k.h, k.layers, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, init.data());
                    }
                    mgl::glTexParameter(k.target, mgl::TextureMinFilter::Nearest);
                    mgl::glTexParameter(k.target, mgl::TextureMagFilter::Nearest);
                }
                mgl::glBindImageTexture(mgl::ImageUnit{0}, tex, 0, k.layered ? GL_TRUE : GL_FALSE, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32UI);

                const std::string coord = is_buffer ? std::format("{}", k.coord[0]) : std::format("ivec3({}, {}, {})", k.coord[0], k.coord[1], k.coord[2]);
                const std::string dump_coord = is_buffer ? "int(idx)" : std::format("ivec3(int(idx) % {0}, (int(idx) / {0}) % {1}, int(idx) / {2})", k.w, k.h, k.w * k.h);
                const std::string head = std::format("#version 320 es\nlayout(local_size_x = 64) in;\nlayout(r32ui, binding = 0) uniform highp {} img;\n"
                                                     "layout(std430, binding = 1) writeonly buffer Out {{ uint words[]; }};\n",
                                                     k.type);
                const std::string atomic_src = head + std::format("void main() {{ words[gl_GlobalInvocationID.x] = imageAtomicAdd(img, {}, 1u); }}\n", coord);
                const std::string dump_src = head + std::format("void main() {{ uint idx = gl_GlobalInvocationID.x; if (idx < {}u) words[idx] = imageLoad(img, {}).x; }}\n", count, dump_coord);
                std::string log;
                const mgl::ProgramId atomic = build_compute(atomic_src, log);
                const mgl::ProgramId dumper = atomic.value != 0 ? build_compute(dump_src, log) : mgl::ProgramId{0};
                if (atomic.value == 0 || dumper.value == 0)
                {
                    if (atomic.value != 0) mgl::glDeleteProgram(atomic);
                    check.expect(false, std::format("{} atomic / dump shaders compile and link", k.type), log);
                    continue;
                }
                const std::vector<std::uint32_t> olds_init(kInvocations, kSentinel), dump_init(static_cast<std::size_t>(count), kSentinel);
                upload(buffers_[1], std::span<const std::uint32_t>(olds_init), 1);
                mgl::glUseProgram(atomic);
                mgl::glDispatchCompute(kInvocations / kLocal, 1, 1);
                mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderImageAccess | mgl::MemoryBarrierMask::BufferUpdate);
                std::vector<std::uint32_t> olds(kInvocations);
                const bool got_olds = read_back(buffers_[1], olds);
                upload(buffers_[4], std::span<const std::uint32_t>(dump_init), 1);
                mgl::glUseProgram(dumper);
                mgl::glDispatchCompute(static_cast<GLuint>((count + 63) / 64), 1, 1);
                mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
                mgl::glUseProgram(mgl::ProgramId{0});
                mgl::glDeleteProgram(atomic);
                mgl::glDeleteProgram(dumper);
                std::vector<std::uint32_t> texels(static_cast<std::size_t>(count));
                if (!got_olds || !read_back(buffers_[4], texels))
                {
                    check.expect(false, std::format("{}: result buffers mapped", k.type));
                    continue;
                }
                // Table 8.25: (x, y, layer/face/z) -> linear index used by the upload order.
                const auto target = static_cast<std::size_t>(is_buffer ? k.coord[0] : (k.coord[2] * k.h + k.coord[1]) * k.w + k.coord[0]);
                std::sort(olds.begin(), olds.end());
                bool perm = true;
                for (std::uint32_t i = 0; i < kInvocations; ++i) perm = perm && olds[i] == init[target] + i;
                int others_changed = 0;
                for (std::size_t t = 0; t < texels.size(); ++t)
                    if (t != target && texels[t] != init[t]) ++others_changed;
                check.expect(perm && texels[target] == init[target] + kInvocations && others_changed == 0,
                             std::format("{} (r32ui, binding layered = {}) imageAtomicAdd(img, {}, 1u) from {} invocations: returned values are {}..{} once each, the addressed texel "
                                         "(linear {}) ends at +{}, the other {} texels are unchanged",
                                         k.type, k.layered, coord, kInvocations, init[target], init[target] + kInvocations - 1, target, kInvocations, count - 1),
                             std::format("permutation {} final 0x{:08X} other texels changed {}", perm, texels[target], others_changed));
            }
            mgl::glBindImageTexture(mgl::ImageUnit{0}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
        }

        // ---------------------------------------------------------------- part 2: fma and precise
        struct Case
        {
            std::string name;
            std::string decls;
            std::string body;
            std::vector<U16> rows;
            std::vector<U20> out;  // filled by run_case
            const char* mesa_compile_gap = nullptr;  // set: a compile failure on Mesa is a documented deviation
        };

        bool run_case(Case& c, demo::Check& check)
        {
            std::string log;
            const mgl::ProgramId program = build_compute(compute_source(c.decls, c.body), log);
            if (c.mesa_compile_gap != nullptr)
                check.expect_or_known_deviation(program.value != 0, std::format("{}: compiles and links", c.name), log, "Mesa", c.mesa_compile_gap);
            if (program.value == 0)
            {
                if (c.mesa_compile_gap == nullptr) check.expect(false, std::format("{}: compiles and links", c.name), log);
                return false;
            }
            const std::vector<U16> init(c.rows.size(), [] { U16 s; s.fill(kSentinel); return s; }());
            const std::vector<std::array<std::uint32_t, 4>> pinit(c.rows.size(), {kSentinel, kSentinel, kSentinel, kSentinel});
            upload(buffers_[0], std::span<const U16>(c.rows), 0);
            upload(buffers_[1], std::span<const U16>(init), 1);
            upload(buffers_[2], std::span<const std::array<std::uint32_t, 4>>(pinit), 2);
            mgl::glUseProgram(program);
            mgl::glDispatchCompute(static_cast<GLuint>((c.rows.size() + kRowsPerGroup - 1) / kRowsPerGroup), 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program);
            std::vector<U16> got(c.rows.size());
            std::vector<std::array<std::uint32_t, 4>> pgot(c.rows.size());
            if (!read_back(buffers_[1], got) || !read_back(buffers_[2], pgot))
            {
                check.expect(false, std::format("{}: output mapped", c.name));
                return false;
            }
            c.out.resize(c.rows.size());
            for (std::size_t r = 0; r < c.rows.size(); ++r)
            {
                std::copy(got[r].begin(), got[r].end(), c.out[r].begin());
                std::copy(pgot[r].begin(), pgot[r].end(), c.out[r].begin() + 16);
            }
            return true;
        }

        // Output word (vector rk, lane) of item `k` (items are packed 4 per row, lane = k % 4).
        static std::uint32_t word(const Case& c, std::size_t k, std::size_t vec) { return c.out[k / 4][4 * vec + k % 4]; }

        // Checks every item's word in vector `vec` against an allowed set; returns mismatch text.
        template <typename T>
        static std::string check_sets(const Case& c, const std::vector<T>& items, std::size_t vec,
                                      const std::function<const std::vector<float>&(const T&)>& allowed, int& checked)
        {
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < items.size(); ++k)
            {
                ++checked;
                const std::uint32_t g = word(c, k, vec);
                const auto& set = allowed(items[k]);
                if (!contains(set, f(g)) && bad++ == 0)
                {
                    std::string s;
                    for (float v : set) s += std::format(" {:a}", v);
                    first = std::format("item {} r{}: got {:a} (0x{:08X}), allowed{}", k, vec, f(g), g, s);
                }
            }
            return bad == 0 ? std::string{} : std::format("{} mismatches; first: {}", bad, first);
        }

        static std::string same_bits(const Case& x, std::size_t vx, const Case& y, std::size_t vy, std::size_t items)
        {
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < items; ++k)
                if (word(x, k, vx) != word(y, k, vy) && bad++ == 0)
                    first = std::format("item {}: {:a} vs {:a}", k, f(word(x, k, vx)), f(word(y, k, vy)));
            return bad == 0 ? std::string{} : std::format("{} differ; first: {}", bad, first);
        }

        void fma_and_precise(demo::Check& check)
        {
            auto triple_lanes = [](const Triple& t) { return std::array<float, 4>{t.a, t.b, t.c, 0.0f}; };
            auto quad_lanes = [](const Quad& q) { return std::array<float, 4>{q.a, q.b, q.c, q.d}; };
            // For exact inputs the fused and twice-rounded sets are the same single value.
            const std::function<const std::vector<float>&(const Triple&)> exact_value = [](const Triple& t) -> const std::vector<float>& { return t.fused; };
            std::vector<std::vector<float>> union_sets;
            for (const auto& t : disc_) union_sets.push_back(merge(t.fused, t.twice));

            // ---- fma, exact inputs
            Case exact{"fma exact", "",
                       "precise vec4 p = fma(f0, f1, f2); r0 = floatBitsToUint(p);"
                       " r1 = floatBitsToUint(fma(f0, f1, f2));"
                       " precise float s = fma(f0.y, f1.y, f2.y); r2 = uvec4(floatBitsToUint(s));"
                       " r3 = floatBitsToUint(f0 * f1 + f2);",
                       pack_rows(exact_, triple_lanes), {}};
            if (run_case(exact, check))
            {
                int n = 0;
                std::string d = check_sets(exact, exact_, 0, exact_value, n) + check_sets(exact, exact_, 1, exact_value, n) +
                                check_sets(exact, exact_, 3, exact_value, n);
                for (std::size_t k = 0; k < exact_.size(); ++k)
                    if (word(exact, k, 2) != b(exact_[(k / 4) * 4 + 1].fused[0]) && d.empty()) d = std::format("scalar fma lane y of item {}", k);
                check.expect(d.empty(), std::format("fma(a, b, c) [GLSL320.FN.FMA] where a*b+c is exactly representable: precise vec4, plain vec4, precise scalar and a*b+c all return the exact value ({} values)", n + static_cast<int>(exact_.size())), d);
            }

            // ---- fma, discriminating inputs, program A
            Case fa{"fma program A", "",
                    "precise vec4 p = fma(f0, f1, f2); r0 = floatBitsToUint(p);"
                    " vec4 q = fma(f0, f1, f2); r1 = floatBitsToUint(q);"
                    " precise float s0 = fma(f0.x, f1.x, f2.x); precise float s1 = fma(f0.y, f1.y, f2.y);"
                    " precise float s2 = fma(f0.z, f1.z, f2.z); precise float s3 = fma(f0.w, f1.w, f2.w);"
                    " r2 = floatBitsToUint(vec4(s0, s1, s2, s3));"
                    " r3 = floatBitsToUint(f0 * f1 + f2);",
                    pack_rows(disc_, triple_lanes), {}};
            // ---- program B: different code, lanes reversed, fma inside a function with a precise local
            Case fb{"fma program B",
                    "float helper(float a, float b, float c) { precise float t = fma(a, b, c); return t; }\n",
                    "precise vec4 p = fma(f0.wzyx, f1.wzyx, f2.wzyx); r0 = floatBitsToUint(p.wzyx);"
                    " r1 = floatBitsToUint(vec4(helper(f0.x, f1.x, f2.x), helper(f0.y, f1.y, f2.y), helper(f0.z, f1.z, f2.z), helper(f0.w, f1.w, f2.w)));"
                    " precise vec4 extra = p * 3.0 - f2.wzyx; r2 = floatBitsToUint(extra);",
                    pack_rows(disc_, triple_lanes), {}};
            const bool ran_a = run_case(fa, check), ran_b = run_case(fb, check);
            const std::function<const std::vector<float>&(const Triple&)> any_rounding = [&](const Triple& t) -> const std::vector<float>& {
                return union_sets[static_cast<std::size_t>(&t - disc_.data())];
            };
            if (ran_a)
            {
                for (std::size_t vec : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}})
                {
                    int n = 0;
                    const std::string d = check_sets(fa, disc_, vec, any_rounding, n);
                    const char* what[] = {"precise vec4 p = fma(a, b, c)", "vec4 q = fma(a, b, c) (not precise)", "precise float s = fma(a, b, c) (scalar)", "a * b + c (not precise)"};
                    check.expect(d.empty(),
                                 std::format("{} ({} inputs where single and double rounding differ): result is a correctly rounded single operation or two correctly rounded operations "
                                             "(GLSL ES 3.20 §4.7.1 'a*b+c' row, fma inherited from it) [GLSL320.FN.FMA]",
                                             what[vec], n),
                                 d);
                }
                // What this implementation does is reported, not asserted: both roundings are allowed.
                int fused = 0, twice = 0, plain_fused = 0, mad_fused = 0;
                for (std::size_t k = 0; k < disc_.size(); ++k)
                {
                    fused += word(fa, k, 0) == b(disc_[k].fused_rn);
                    twice += word(fa, k, 0) == b(disc_[k].twice_rn);
                    plain_fused += word(fa, k, 1) == b(disc_[k].fused_rn);
                    mad_fused += word(fa, k, 3) == b(disc_[k].fused_rn);
                }
                const std::string inv = same_bits(fa, 0, fa, 2, disc_.size());
                check.expect(inv.empty(),
                             std::format("fma() consumed by precise variables is invariant within a program: vec4 and per-component scalar evaluation give identical bits (§8.3) "
                                         "[measured, not asserted: of {} inputs precise fma() matched single rounding (nearest even) {}x and double rounding {}x; plain fma() "
                                         "single-rounded {}x, plain a*b+c {}x]",
                                         disc_.size(), fused, twice, plain_fused, mad_fused),
                             inv);
            }
            if (ran_a && ran_b)
            {
                check.expect(same_bits(fa, 0, fb, 0, disc_.size()).empty(),
                             "fma() consumed by a precise variable is invariant across programs: program B (lanes reversed, other surrounding code) returns the same bits as program A (§8.3 "
                             "'computed with the same precision as any other fma() consumed by a precise variable')",
                             same_bits(fa, 0, fb, 0, disc_.size()));
                check.expect(same_bits(fa, 0, fb, 1, disc_.size()).empty(),
                             "fma() assigned to a precise local inside a function (§4.9 func2 pattern) returns the same bits as the precise fma() in main of the other program",
                             same_bits(fa, 0, fb, 1, disc_.size()));
            }

            // ---- precise a * b + c: never fused
            const std::function<const std::vector<float>&(const Triple&)> twice_only = [](const Triple& t) -> const std::vector<float>& { return t.twice; };
            Case pa{"precise a*b+c forms A", "struct PS { vec4 v; };\nvec4 gv;\nprecise gv;\n",
                    "precise vec4 v = f0 * f1 + f2; r0 = floatBitsToUint(v);"
                    " vec4 m = f0 * f1; precise vec4 w = m + f2; r1 = floatBitsToUint(w);"
                    " precise PS ps; ps.v = f0 * f1 + f2; r2 = floatBitsToUint(ps.v);"
                    " gv = f0 * f1 + f2; r3 = floatBitsToUint(gv);",
                    pack_rows(disc_, triple_lanes), {}};
            Case pb{"precise a*b+c forms B", "",
                    "precise float s0 = f0.x * f1.x + f2.x; precise float s1 = f0.y * f1.y + f2.y;"
                    " precise float s2 = f0.z * f1.z + f2.z; precise float s3 = f0.w * f1.w + f2.w;"
                    " r0 = floatBitsToUint(vec4(s0, s1, s2, s3));",
                    pack_rows(disc_, triple_lanes), {}};
            // §4.3.9: an interface block's type-qualifier "may optionally include ... the precise qualifier";
            // §4.9: on a block, precise applies to each member.
            Case pblock{"precise buffer block", "layout(std430, binding = 2) precise writeonly buffer PreciseOut { vec4 pv[]; };\n",
                        "pv[row] = f0 * f1 + f2;", pack_rows(disc_, triple_lanes), {},
                        "Mesa's GLSL parser rejects precise as an interface-block qualifier (syntax error at the block name), which GLSL ES 3.20 §4.3.9 allows"};
            if (run_case(pa, check) && run_case(pb, check))
            {
                struct Form
                {
                    const Case* c;
                    std::size_t vec;
                    const char* what;
                };
                const Form forms[] = {
                    {&pa, 0, "precise vec4 v = a * b + c"},
                    {&pa, 1, "vec4 m = a * b; precise vec4 w = m + c (operands of a precise assignment are precise, §4.9 example)"},
                    {&pa, 2, "precise PS ps; ps.v = a * b + c (precise on a structure variable applies to its members)"},
                    {&pa, 3, "vec4 gv; precise gv; gv = a * b + c (precise redeclaration of a global)"},
                    {&pb, 0, "precise float s = a * b + c (scalar)"},
                };
                for (const Form& form : forms)
                {
                    int n = 0;
                    const std::string d = check_sets(*form.c, disc_, form.vec, twice_only, n);
                    check.expect(d.empty(),
                                 std::format("{} [GLSL320.PRECISE]: {} inputs where a fused multiply-add gives a different value; the result is two correctly rounded operations "
                                             "(§4.9 'a * b + c is not allowed to become a single operation fma(a, b, c)')",
                                             form.what, n),
                                 d);
                }
                std::string d;
                for (const Form& form : forms)
                    if (d.empty()) d = same_bits(pa, 0, *form.c, form.vec, disc_.size());
                check.expect(d.empty(), "operator consistency (§4.9): all five precise forms of a * b + c return identical bits", d);
                if (run_case(pblock, check))
                {
                    int n = 0;
                    const std::string e = check_sets(pblock, disc_, 4, twice_only, n) + same_bits(pa, 0, pblock, 4, disc_.size());
                    check.expect(e.empty(),
                                 std::format("pv[row] = a * b + c with pv a member of a precise buffer block [GLSL320.PRECISE]: {} results are two correctly rounded operations and equal "
                                             "the other precise forms",
                                             n),
                                 e);
                }
            }

            // ---- precise a*b + c*d in every order, two programs
            const std::function<const std::vector<float>&(const Quad&)> quad_allowed = [](const Quad& q) -> const std::vector<float>& { return q.allowed; };
            Case qa{"precise a*b+c*d program A", "",
                    "precise vec4 x0 = f0 * f1 + f2 * f3; r0 = floatBitsToUint(x0);"
                    " precise vec4 x1 = f3 * f2 + f1 * f0; r1 = floatBitsToUint(x1);"
                    " precise vec4 x2 = f1 * f0 + f2 * f3; r2 = floatBitsToUint(x2);",
                    pack_rows(quads_, quad_lanes), {}};
            Case qb{"precise a*b+c*d program B", "",
                    "precise float y0 = f2.x * f3.x + f0.x * f1.x; precise float y1 = f2.y * f3.y + f0.y * f1.y;"
                    " precise float y2 = f2.z * f3.z + f0.z * f1.z; precise float y3 = f2.w * f3.w + f0.w * f1.w;"
                    " r0 = floatBitsToUint(vec4(y0, y1, y2, y3));",
                    pack_rows(quads_, quad_lanes), {}};
            if (run_case(qa, check) && run_case(qb, check))
            {
                int n = 0;
                const std::string d = check_sets(qa, quads_, 0, quad_allowed, n) + check_sets(qa, quads_, 1, quad_allowed, n) + check_sets(qa, quads_, 2, quad_allowed, n) +
                                      check_sets(qb, quads_, 0, quad_allowed, n);
                check.expect(d.empty(),
                             std::format("precise a*b + c*d [GLSL320.PRECISE] ({} results, inputs where contracting either product into an fma lands outside the set): both products and the sum "
                                         "correctly rounded",
                                         n),
                             d);
                const std::string s = same_bits(qa, 0, qa, 1, quads_.size()) + same_bits(qa, 0, qa, 2, quads_.size()) + same_bits(qa, 0, qb, 0, quads_.size());
                check.expect(s.empty(),
                             "§4.9 identities: precise a*b + c*d, d*c + b*a, b*a + c*d (program A) and c*d + a*b (program B, scalar) give identical bits", s);
            }

            // ---- precise forbids reassociation
            {
                const float p24 = std::ldexp(1.0f, 24), e24 = std::ldexp(1.0f, -24), e25 = std::ldexp(1.0f, -25);
                const float big = std::ldexp(1.0f, 100), small = std::ldexp(1.0f, -100), top = std::ldexp(1.0f, 127);
                // lanes x..w of a, b, c for addition (u0..u2) and multiplication (u3 and the second row)
                const std::array<std::array<float, 3>, 4> add = {{{p24, 1.0f, 1.0f}, {1.0f, e24, e24}, {-p24, -1.0f, -1.0f}, {0.5f, e25, e25}}};
                const std::array<std::array<float, 3>, 4> mul = {{{big, big, small}, {small, small, big}, {top, 4.0f, 0.25f}, {-top, 2.0f, 0.5f}}};
                U16 r0{}, r1{};
                for (std::size_t k = 0; k < 4; ++k)
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        r0[4 * j + k] = b(add[k][j]);
                        r1[4 * j + k] = b(mul[k][j]);
                    }
                Case ra{"precise reassociation", "",
                        "precise vec4 s = f0 + (f1 + f2); r0 = floatBitsToUint(s);"
                        " precise vec4 m = f0 * (f1 * f2); r1 = floatBitsToUint(m);",
                        {r0, r1}, {}};
                std::array<float, 4> want_add{}, want_mul{}, wrong_add{}, wrong_mul{};
                bool inputs_ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    // b + c and a + (b + c) are exact (checked), so no rounding mode matters.
                    const double bc = double(add[k][1]) + double(add[k][2]);
                    want_add[k] = static_cast<float>(double(add[k][0]) + bc);
                    inputs_ok = inputs_ok && double(static_cast<float>(bc)) == bc && double(want_add[k]) == double(add[k][0]) + bc;
                    volatile float t = add[k][0] + add[k][1];  // volatile: no contraction, plain float rounding
                    wrong_add[k] = t + add[k][2];
                    const double mc = double(mul[k][1]) * double(mul[k][2]);
                    want_mul[k] = static_cast<float>(double(mul[k][0]) * mc);
                    inputs_ok = inputs_ok && double(static_cast<float>(mc)) == mc && double(want_mul[k]) == double(mul[k][0]) * mc;
                    volatile float u = mul[k][0] * mul[k][1];
                    wrong_mul[k] = u * mul[k][2];
                }
                bool differ = true;
                for (std::size_t k = 0; k < 4; ++k) differ = differ && b(wrong_add[k]) != b(want_add[k]) && b(wrong_mul[k]) != b(want_mul[k]);
                check.expect(inputs_ok && differ,
                             "negative control: for the reassociation inputs the exact a + (b + c) and a * (b * c) differ from (a + b) + c and (a * b) * c in float (2^24 + 2 vs 2^24, 2^100 vs inf, ...)");
                if (run_case(ra, check))
                {
                    std::string d;
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        if (ra.out[0][k] != b(want_add[k]) && d.empty()) d = std::format("a + (b + c) lane {}: got {:a} expected {:a}", k, f(ra.out[0][k]), want_add[k]);
                        // row 0 holds the addition inputs (read r0), row 1 the multiplication inputs (read r1)
                        if (ra.out[1][4 + k] != b(want_mul[k]) && d.empty()) d = std::format("a * (b * c) lane {}: got {:a} expected {:a}", k, f(ra.out[1][4 + k]), want_mul[k]);
                    }
                    check.expect(d.empty(),
                                 "precise a + (b + c) and a * (b * c) [GLSL320.PRECISE] are evaluated in the stated order (§4.9: not (a + b) + c, not (a * b) * c), although §5.12 allows "
                                 "reassociation without precise",
                                 d);
                }
            }

            // ---- negative controls for the rounding sets
            {
                int fused_rejected = 0, contracted_rejected = 0;
                for (const auto& t : disc_) fused_rejected += !contains(t.twice, t.fused_rn);
                for (const auto& q : quads_) contracted_rejected += !contains(q.allowed, static_cast<float>(double(q.a) * double(q.b) + double(static_cast<float>(double(q.c) * double(q.d)))));
                check.expect(fused_rejected == static_cast<int>(disc_.size()) && contracted_rejected == static_cast<int>(quads_.size()),
                             std::format("negative control: a contracting compiler would fail: the single-rounding fma value is outside the precise a*b+c set for all {} inputs ({} candidates "
                                         "tried), and fma(a, b, c*d) is outside the a*b+c*d set for all {} inputs ({} tried)",
                                         disc_.size(), disc_tried_, quads_.size(), quad_tried_));
            }
        }

        // ---------------------------------------------------------------- compile checks
        void compile_checks(demo::Check& check)
        {
            struct Src
            {
                const char* what;
                std::string source;
                bool compiles;
            };
            auto cs = [](const char* version, const char* decls, const char* body) {
                return std::format("#version {}\nlayout(local_size_x = 1) in;\nprecision highp float;\n{}\n"
                                   "layout(std430, binding = 0) buffer B {{ uint u; int i; float x; }};\nvoid main() {{ {} }}\n",
                                   version, decls, body);
            };
            const char* uimg = "layout(r32ui, binding = 0) uniform highp uimage2D img;";
            const char* fimg = "layout(r32f, binding = 0) uniform highp image2D img;";
            const std::vector<Src> sources = {
                {"#version 320 es: imageAtomicAdd(uimage2D r32ui, ivec2, uint) compiles (positive control)", cs("320 es", uimg, "u = imageAtomicAdd(img, ivec2(0), 1u);"), true},
                {"#version 310 es: imageAtomicAdd is not a GLSL ES 3.10 built-in (compile error)", cs("310 es", uimg, "u = imageAtomicAdd(img, ivec2(0), 1u);"), false},
                {"#version 320 es: imageAtomicAdd(uimage2D, ivec2, int) has no overload (no implicit int -> uint conversion), compile error",
                 cs("320 es", uimg, "u = imageAtomicAdd(img, ivec2(0), 1);"), false},
                {"#version 320 es: imageAtomicExchange(image2D r32f, ivec2, float) compiles (positive control)", cs("320 es", fimg, "x = imageAtomicExchange(img, ivec2(0), 1.0);"), true},
                {"#version 320 es: imageAtomicAdd(image2D r32f, ivec2, float) does not exist (only exchange takes float, §8.12), compile error",
                 cs("320 es", fimg, "x = imageAtomicAdd(img, ivec2(0), 1.0);"), false},
                {"#version 320 es: fma(float, float, float) compiles (positive control)", cs("320 es", "", "x = fma(x, x, x);"), true},
                {"#version 310 es: fma is not a GLSL ES 3.10 built-in (compile error)", cs("310 es", "", "x = fma(x, x, x);"), false},
                {"#version 320 es: precise float compiles (positive control)", cs("320 es", "", "precise float y = x * x + x; x = y;"), true},
                {"#version 310 es: precise is a reserved keyword in GLSL ES 3.10 (compile error)", cs("310 es", "", "precise float y = x * x + x; x = y;"), false},
                {"#version 320 es: highp on a function return type compiles (positive control)", cs("320 es", "highp float sq(float a) { return a * a; }", "x = sq(x);"), true},
                {"#version 320 es: precise on a function return type is a compile error (§6.1: only a precision qualifier is allowed on the return type)",
                 cs("320 es", "precise float sq(float a) { return a * a; }", "x = sq(x);"), false},
            };
            for (const auto& s : sources)
            {
                std::string log;
                const mgl::ShaderId sh = demo::compile_shader(mgl::ShaderType::Compute, s.source, log);
                const bool ok = sh.value != 0;
                if (ok) mgl::glDeleteShader(sh);
                check.expect(ok == s.compiles, s.what, ok == s.compiles ? std::string{} : (ok ? std::string{"compiled"} : log));
            }
        }

        AtomicInputs atomics_;
        std::vector<Triple> exact_, disc_;
        std::vector<Quad> quads_;
        int disc_tried_ = 0, quad_tried_ = 0;
        std::array<mgl::BufferId, 6> buffers_{};
        std::array<mgl::TextureId, 8> textures_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "test-glsl320-builtins",
        .title = "GLSL ES 3.20 image atomics, fma and precise",
        .min_es = {3, 2},
        .summary = "Runs the GLSL ES 3.20 image atomics on r32ui / r32i / r32f images with 1024 contending compute invocations and checks "
                   "order-independent final values and that one serial order explains every returned value; checks fma() against the "
                   "single-or-double rounding rule and its invariance under precise, and that precise forbids fusion and reassociation.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Glsl320Builtins>(argc, argv, info);
}
