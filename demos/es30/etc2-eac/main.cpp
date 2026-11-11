// es30-etc2-eac
//
// The ten ETC2/EAC compressed formats that every ES 3.0 implementation must support (ES 3.0 §3.8.6,
// Table 3.19, Appendix C). Every block is hand-encoded here, field by field, from the bit layouts of
// Appendix C: the individual, differential, T, H and planar modes of COMPRESSED_RGB8_ETC2, the
// punchthrough-alpha variant with opaque and transparent blocks, the EAC alpha channel of
// RGBA8_ETC2_EAC, unsigned and signed R11/RG11 EAC, and the sRGB variants.
//
// The textures are uploaded with all four compressed upload commands:
//   glCompressedTexImage2D      most 2D textures, and the 6 x 6 texture whose size is not a multiple of 4
//   glCompressedTexSubImage2D   one block of the RGB8 texture, the whole immutable SRGB8 texture
//                               (glTexStorage2D), and the partial 2 x 2 edge block of the 6 x 6 texture
//   glCompressedTexImage3D      an RGB8_ETC2 2D array texture with three layers
//   glCompressedTexSubImage3D   one block column of layers 1 and 2 of that array
//
// What is checked:
//   * a CPU decoder written from the text of Appendix C (separate from the encoder) predicts every
//     texel; each texture is drawn 1:1 with NEAREST into an RGBA32UI target with floatBitsToUint and
//     compared exactly (sRGB formats: within 1/255 after the equation 3.26 decode);
//   * the numbers worked out in the Appendix C text itself (e.g. 103 / table 13 / multiplier 2 -> 83)
//     are reproduced, independently of the decoder;
//   * the magnified picture in the window is compared texel by texel with the same prediction;
//   * negative controls: decoders with a wrong modifier table, a negated differential, swapped index
//     bits, a reversed H-mode ordering, the "B + dG" misprint, truncating planar rounding, a zero
//     EAC multiplier, -128 not mapped to -127, and non-zeroed transparent texels all disagree;
//   * NUM_COMPRESSED_TEXTURE_FORMATS / COMPRESSED_TEXTURE_FORMATS list all ten formats, and
//     (ES 3.1+ contexts) TEXTURE_COMPRESSED / TEXTURE_INTERNAL_FORMAT level queries;
//   * the errors of ES 3.0 §3.8.6 (ES 3.2 §8.7): imageSize, block alignment, format mismatch, the
//     TEXTURE_3D restriction, immutable textures, sub-image bounds, and that a rejected call leaves
//     the texels unchanged.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using CIF = mgl::CompressedInternalFormat;

namespace
{
    // =============================================================================================
    // Encoder: places field values at the bit positions of Appendix C (Tables C.3, C.9, C.11).
    // It knows nothing about decoding except the overflow rule that selects T, H and planar
    // blocks, which an encoder needs to produce them at all.
    // =============================================================================================

    /// Pixel indices of one block: row y = 0..3 (v direction), one digit per x = 0..3 (u direction).
    using Pattern = std::array<const char*, 4>;

    struct Word
    {
        std::uint64_t w = 0;
        Word& put(int hi, int lo, std::uint64_t value)
        {
            const std::uint64_t mask = (std::uint64_t{1} << (hi - lo + 1)) - 1;
            w = (w & ~(mask << lo)) | ((value & mask) << lo);
            return *this;
        }
    };

    /// Table C.3f: pixel k (a = 0 ... p = 15, column by column: k = 4x + y) has its index LSB in
    /// bit k and its MSB in bit k + 16.
    Word etc_indices(const Pattern& p)
    {
        Word word;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                const int idx = p[static_cast<std::size_t>(y)][x] - '0';
                const int k = 4 * x + y;
                word.put(k, k, static_cast<std::uint64_t>(idx & 1));
                word.put(k + 16, k + 16, static_cast<std::uint64_t>(idx >> 1));
            }
        return word;
    }

    /// The 5-bit base + 3-bit delta pair that starts at bit `hi` (R: 63, G: 55, B: 47) leaves [0, 31].
    bool enc_overflows(std::uint64_t w, int hi)
    {
        const int base = static_cast<int>((w >> (hi - 4)) & 31);
        int delta = static_cast<int>((w >> (hi - 7)) & 7);
        if (delta >= 4) delta -= 8;
        return base + delta < 0 || base + delta > 31;
    }

    /// Sets the bits that T, H and planar blocks leave unused so that the overflow rule selects the mode.
    std::uint64_t settle(std::uint64_t w, std::initializer_list<int> free_bits, const std::function<bool(std::uint64_t)>& wanted)
    {
        const unsigned combos = 1u << free_bits.size();
        for (unsigned combo = 0; combo < combos; ++combo)
        {
            std::uint64_t c = w;
            unsigned i = 0;
            for (const int b : free_bits)
            {
                if ((combo >> i) & 1u) c |= std::uint64_t{1} << b;
                else c &= ~(std::uint64_t{1} << b);
                ++i;
            }
            if (wanted(c)) return c;
        }
        throw demo::Failure("encoder: no assignment of the free bits selects the intended ETC2 mode");
    }

    std::uint64_t enc_individual(int r1, int g1, int b1, int r2, int g2, int b2, int t1, int t2, int flip, const Pattern& p)
    {
        Word w = etc_indices(p);
        w.put(63, 60, static_cast<std::uint64_t>(r1)).put(59, 56, static_cast<std::uint64_t>(r2));
        w.put(55, 52, static_cast<std::uint64_t>(g1)).put(51, 48, static_cast<std::uint64_t>(g2));
        w.put(47, 44, static_cast<std::uint64_t>(b1)).put(43, 40, static_cast<std::uint64_t>(b2));
        w.put(39, 37, static_cast<std::uint64_t>(t1)).put(36, 34, static_cast<std::uint64_t>(t2));
        w.put(33, 33, 0).put(32, 32, static_cast<std::uint64_t>(flip));
        return w.w;
    }

    /// bit33 is the 'diff' bit (1) of RGB8_ETC2 or the 'opaque' bit of the punchthrough formats.
    std::uint64_t enc_differential(int r, int g, int b, int dr, int dg, int db, int t1, int t2, int flip, int bit33, const Pattern& p)
    {
        for (const auto& [c, d] : {std::pair{r, dr}, std::pair{g, dg}, std::pair{b, db}})
            if (c + d < 0 || c + d > 31 || d < -4 || d > 3) throw demo::Failure("encoder: differential base + delta out of range");
        Word w = etc_indices(p);
        w.put(63, 59, static_cast<std::uint64_t>(r)).put(58, 56, static_cast<std::uint64_t>(dr & 7));
        w.put(55, 51, static_cast<std::uint64_t>(g)).put(50, 48, static_cast<std::uint64_t>(dg & 7));
        w.put(47, 43, static_cast<std::uint64_t>(b)).put(42, 40, static_cast<std::uint64_t>(db & 7));
        w.put(39, 37, static_cast<std::uint64_t>(t1)).put(36, 34, static_cast<std::uint64_t>(t2));
        w.put(33, 33, static_cast<std::uint64_t>(bit33)).put(32, 32, static_cast<std::uint64_t>(flip));
        return w.w;
    }

    std::uint64_t enc_t(int r1, int g1, int b1, int r2, int g2, int b2, int dist, int bit33, const Pattern& p)
    {
        Word w = etc_indices(p);
        w.put(60, 59, static_cast<std::uint64_t>(r1 >> 2)).put(57, 56, static_cast<std::uint64_t>(r1 & 3));
        w.put(55, 52, static_cast<std::uint64_t>(g1)).put(51, 48, static_cast<std::uint64_t>(b1));
        w.put(47, 44, static_cast<std::uint64_t>(r2)).put(43, 40, static_cast<std::uint64_t>(g2)).put(39, 36, static_cast<std::uint64_t>(b2));
        w.put(35, 34, static_cast<std::uint64_t>(dist >> 1)).put(33, 33, static_cast<std::uint64_t>(bit33)).put(32, 32, static_cast<std::uint64_t>(dist & 1));
        return settle(w.w, {63, 62, 61, 58}, [](std::uint64_t c) { return enc_overflows(c, 63); });
    }

    std::uint64_t enc_h(int r1, int g1, int b1, int r2, int g2, int b2, int da, int db, int bit33, const Pattern& p)
    {
        Word w = etc_indices(p);
        w.put(62, 59, static_cast<std::uint64_t>(r1));
        w.put(58, 56, static_cast<std::uint64_t>(g1 >> 1)).put(52, 52, static_cast<std::uint64_t>(g1 & 1));
        w.put(51, 51, static_cast<std::uint64_t>(b1 >> 3)).put(49, 47, static_cast<std::uint64_t>(b1 & 7));
        w.put(46, 43, static_cast<std::uint64_t>(r2)).put(42, 39, static_cast<std::uint64_t>(g2)).put(38, 35, static_cast<std::uint64_t>(b2));
        w.put(34, 34, static_cast<std::uint64_t>(da)).put(33, 33, static_cast<std::uint64_t>(bit33)).put(32, 32, static_cast<std::uint64_t>(db));
        return settle(w.w, {63, 55, 54, 53, 50}, [](std::uint64_t c) { return !enc_overflows(c, 63) && enc_overflows(c, 55); });
    }

    /// RGB 676: ro/bo/rh/bh/rv/bv are 6-bit, go/gh/gv 7-bit.
    std::uint64_t enc_planar(int ro, int go, int bo, int rh, int gh, int bh, int rv, int gv, int bv, int bit33)
    {
        Word w;
        w.put(62, 57, static_cast<std::uint64_t>(ro));
        w.put(56, 56, static_cast<std::uint64_t>(go >> 6)).put(54, 49, static_cast<std::uint64_t>(go & 63));
        w.put(48, 48, static_cast<std::uint64_t>(bo >> 5)).put(44, 43, static_cast<std::uint64_t>((bo >> 3) & 3)).put(41, 39, static_cast<std::uint64_t>(bo & 7));
        w.put(38, 34, static_cast<std::uint64_t>(rh >> 1)).put(33, 33, static_cast<std::uint64_t>(bit33)).put(32, 32, static_cast<std::uint64_t>(rh & 1));
        w.put(31, 25, static_cast<std::uint64_t>(gh)).put(24, 19, static_cast<std::uint64_t>(bh));
        w.put(18, 13, static_cast<std::uint64_t>(rv)).put(12, 6, static_cast<std::uint64_t>(gv)).put(5, 0, static_cast<std::uint64_t>(bv));
        return settle(w.w, {63, 55, 47, 46, 45, 42},
                      [](std::uint64_t c) { return !enc_overflows(c, 63) && !enc_overflows(c, 55) && enc_overflows(c, 47); });
    }

    /// Table C.9: base codeword 63-56, multiplier 55-52, table index 51-48, then 3-bit indices with
    /// pixel a in bits 47-45, b in 44-42, ... (pixel k = 4x + y in bits 47-3k .. 45-3k).
    std::uint64_t enc_eac(int base, int mult, int table, const Pattern& p)
    {
        Word w;
        w.put(63, 56, static_cast<std::uint64_t>(base & 0xFF)).put(55, 52, static_cast<std::uint64_t>(mult)).put(51, 48, static_cast<std::uint64_t>(table));
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                const int k = 4 * x + y;
                w.put(47 - 3 * k, 45 - 3 * k, static_cast<std::uint64_t>(p[static_cast<std::size_t>(y)][x] - '0'));
            }
        return w.w;
    }

    /// q0 is the most significant byte (Appendix C.1: int64bit = 256 x (... 256 x q0 + q1 ...) + q7).
    void append(std::vector<std::uint8_t>& out, std::uint64_t w)
    {
        for (int i = 7; i >= 0; --i) out.push_back(static_cast<std::uint8_t>(w >> (8 * i)));
    }

    std::vector<std::uint8_t> bytes(std::initializer_list<std::uint64_t> words)
    {
        std::vector<std::uint8_t> out;
        for (const std::uint64_t w : words) append(out, w);
        return out;
    }

    // =============================================================================================
    // Decoder: written from the prose of Appendix C, independently of the encoder above.
    // Mutations are deliberate mistakes for the negative controls.
    // =============================================================================================

    enum class Mut
    {
        None,
        WrongModifierTable,      // Table C.6 row of the next codeword
        NegatedDifferential,     // dR/dG/dB subtracted instead of added
        SwappedIndexBits,        // MSB/LSB of the 2-bit pixel index exchanged
        ReversedHOrder,          // H mode: distance LSB = (col1 < col2)
        BlueUsesDG,              // the misprint "B + dG" in the subblock-2 formula of C.1.1 / C.1.9
        PlanarTruncate,          // planar: no "+2" rounding term
        EacZeroMultiplierIsZero, // R11: multiplier 0 taken as 0 instead of 1/8
        SignedKeepsMinus128,     // signed R11: base -128 not treated as -127
        TransparentKeepsRgb,     // punchthrough: RGB not zeroed where alpha = 0
    };

    using Rgba = std::array<int, 4>;

    int field(std::uint64_t w, int hi, int lo) { return static_cast<int>((w >> lo) & ((std::uint64_t{1} << (hi - lo + 1)) - 1)); }
    int twos3(int v) { return v >= 4 ? v - 8 : v; }
    int clamp255(int v) { return std::clamp(v, 0, 255); }
    int extend4(int v) { return (v << 4) | v; }         // "replicates the four bits twice"
    int extend5(int v) { return (v << 3) | (v >> 2); }  // top three bits copied to the bottom
    int extend6(int v) { return (v << 2) | (v >> 4); }  // top two bits copied to the bottom
    int extend7(int v) { return (v << 1) | (v >> 6); }

    // Table C.6 / C.12a: {a, b} of codewords 0..7 (the modifier sets are [-b, -a, a, b]).
    constexpr std::array<std::array<int, 2>, 8> kTableC6 = {{{2, 8}, {5, 17}, {9, 29}, {13, 42}, {18, 60}, {24, 80}, {33, 106}, {47, 183}}};
    // Table C.8: distances of the T and H modes.
    constexpr std::array<int, 8> kTableC8 = {3, 6, 11, 16, 23, 32, 41, 64};
    // Table C.10: EAC modifier tables.
    constexpr std::array<std::array<int, 8>, 16> kTableC10 = {{
        {-3, -6, -9, -15, 2, 5, 8, 14}, {-3, -7, -10, -13, 2, 6, 9, 12}, {-2, -5, -8, -13, 1, 4, 7, 12}, {-2, -4, -6, -13, 1, 3, 5, 12},
        {-3, -6, -8, -12, 2, 5, 7, 11}, {-3, -7, -9, -11, 2, 6, 8, 10}, {-4, -7, -8, -11, 3, 6, 7, 10}, {-3, -5, -8, -11, 2, 4, 7, 10},
        {-2, -6, -8, -10, 1, 5, 7, 9},  {-2, -5, -8, -10, 1, 4, 7, 9},  {-2, -4, -8, -10, 1, 3, 7, 9},  {-2, -5, -7, -10, 1, 4, 6, 9},
        {-3, -4, -7, -10, 2, 3, 6, 9},  {-1, -2, -3, -10, 0, 1, 2, 9},  {-4, -6, -8, -9, 3, 5, 7, 8},   {-3, -5, -7, -9, 2, 4, 6, 8},
    }};

    /// One texel (x along u, y along v, both 0..3) of a COMPRESSED_RGB8_ETC2 block (punch = false) or a
    /// COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 block (punch = true), as 8-bit RGBA (C.1.1, C.1.9).
    Rgba decode_etc2(std::uint64_t w, bool punch, int x, int y, Mut m)
    {
        // Pixel index: pixel d (x = 0, y = 3) has its MSB in bit 19 and its LSB in bit 3 (Table C.3f).
        const int k = 4 * x + y;
        int msb = field(w, 16 + k, 16 + k), lsb = field(w, k, k);
        if (m == Mut::SwappedIndexBits) std::swap(msb, lsb);
        const int index = 2 * msb + lsb;

        const bool opaque = !punch || field(w, 33, 33) == 1;
        // C.1.9: "if( opaque == 0 && MSB == 1 && LSB == 0) red = green = blue = alpha = 0".
        const bool transparent = !opaque && msb == 1 && lsb == 0;
        auto out = [&](int r, int g, int b) -> Rgba {
            if (transparent) return m == Mut::TransparentKeepsRgb ? Rgba{clamp255(r), clamp255(g), clamp255(b), 0} : Rgba{0, 0, 0, 0};
            return {clamp255(r), clamp255(g), clamp255(b), 255};
        };

        const int table_shift = m == Mut::WrongModifierTable ? 1 : 0;
        // Table C.7 (opaque) and Table C.13b (punchthrough, opaque bit 0).
        auto modifier = [&](int codeword) {
            const auto& t = kTableC6[static_cast<std::size_t>((codeword + table_shift) & 7)];
            if (!opaque) return index == 1 ? t[1] : index == 3 ? -t[1] : 0;
            switch (index)
            {
            case 0: return t[0];
            case 1: return t[1];
            case 2: return -t[0];
            default: return -t[1];
            }
        };
        const bool flip = field(w, 32, 32) == 1;
        const bool second = flip ? y >= 2 : x >= 2;  // Tables C.4 / C.5

        if (!punch && field(w, 33, 33) == 0)
        {
            // 'individual': two RGB444 base colors (Table C.3b).
            const int r = extend4(field(w, second ? 59 : 63, second ? 56 : 60));
            const int g = extend4(field(w, second ? 51 : 55, second ? 48 : 52));
            const int b = extend4(field(w, second ? 43 : 47, second ? 40 : 44));
            const int mod = modifier(field(w, second ? 36 : 39, second ? 34 : 37));
            return out(r + mod, g + mod, b + mod);
        }

        const int R = field(w, 63, 59), G = field(w, 55, 51), B = field(w, 47, 43);
        int dR = twos3(field(w, 58, 56)), dG = twos3(field(w, 50, 48)), dB = twos3(field(w, 42, 40));
        if (R + dR < 0 || R + dR > 31)
        {
            // 'T' (Table C.3d).
            const std::array<int, 3> c1 = {extend4((field(w, 60, 59) << 2) | field(w, 57, 56)), extend4(field(w, 55, 52)), extend4(field(w, 51, 48))};
            const std::array<int, 3> c2 = {extend4(field(w, 47, 44)), extend4(field(w, 43, 40)), extend4(field(w, 39, 36))};
            const int d = kTableC8[static_cast<std::size_t>((field(w, 35, 34) << 1) | field(w, 32, 32))];
            const std::array<std::array<int, 3>, 4> paint = {{c1, {c2[0] + d, c2[1] + d, c2[2] + d}, c2, {c2[0] - d, c2[1] - d, c2[2] - d}}};
            const auto& c = paint[static_cast<std::size_t>(index)];
            return out(c[0], c[1], c[2]);
        }
        if (G + dG < 0 || G + dG > 31)
        {
            // 'H' (Table C.3e).
            const std::array<int, 3> c1 = {extend4(field(w, 62, 59)), extend4((field(w, 58, 56) << 1) | field(w, 52, 52)),
                                           extend4((field(w, 51, 51) << 3) | field(w, 49, 47))};
            const std::array<int, 3> c2 = {extend4(field(w, 46, 43)), extend4(field(w, 42, 39)), extend4(field(w, 38, 35))};
            const int v1 = (c1[0] << 16) + (c1[1] << 8) + c1[2], v2 = (c2[0] << 16) + (c2[1] << 8) + c2[2];
            const int order = m == Mut::ReversedHOrder ? (v1 < v2 ? 1 : 0) : (v1 >= v2 ? 1 : 0);
            const int d = kTableC8[static_cast<std::size_t>((field(w, 34, 34) << 2) | (field(w, 32, 32) << 1) | order)];
            const std::array<std::array<int, 3>, 4> paint = {
                {{c1[0] + d, c1[1] + d, c1[2] + d}, {c1[0] - d, c1[1] - d, c1[2] - d}, {c2[0] + d, c2[1] + d, c2[2] + d}, {c2[0] - d, c2[1] - d, c2[2] - d}}};
            const auto& c = paint[static_cast<std::size_t>(index)];
            return out(c[0], c[1], c[2]);
        }
        if (B + dB < 0 || B + dB > 31)
        {
            // 'planar' (Table C.3g): RGB676 colors O, H, V; alpha always 255, the opaque bit is ignored.
            const int ro = extend6(field(w, 62, 57)), go = extend7((field(w, 56, 56) << 6) | field(w, 54, 49));
            const int bo = extend6((field(w, 48, 48) << 5) | (field(w, 44, 43) << 3) | field(w, 41, 39));
            const int rh = extend6((field(w, 38, 34) << 1) | field(w, 32, 32)), gh = extend7(field(w, 31, 25)), bh = extend6(field(w, 24, 19));
            const int rv = extend6(field(w, 18, 13)), gv = extend7(field(w, 12, 6)), bv = extend6(field(w, 5, 0));
            const int round = m == Mut::PlanarTruncate ? 0 : 2;
            // "(x (H - O) + y (V - O) + 4 O + 2) >> 2", i.e. rounded to nearest with ties up.
            auto plane = [&](int o, int h, int v) { return clamp255(static_cast<int>(std::floor((x * (h - o) + y * (v - o) + 4 * o + round) / 4.0))); };
            return {plane(ro, rh, rv), plane(go, gh, gv), plane(bo, bh, bv), 255};
        }
        // 'differential' (Table C.3c / C.11b): RGB555 base plus a 3-bit two's-complement delta.
        if (m == Mut::NegatedDifferential) dR = -dR, dG = -dG, dB = -dB;
        const int blue_delta = m == Mut::BlueUsesDG ? dG : dB;
        // (The clamps only matter for the mutated decoders; the real sums stay in [0, 31] by mode selection.)
        const int r = extend5(std::clamp(second ? R + dR : R, 0, 31)), g = extend5(std::clamp(second ? G + dG : G, 0, 31));
        const int b = extend5(std::clamp(second ? B + blue_delta : B, 0, 31));
        const int mod = modifier(field(w, second ? 36 : 39, second ? 34 : 37));
        return out(r + mod, g + mod, b + mod);
    }

    int eac_index(std::uint64_t w, int x, int y)
    {
        const int k = 4 * x + y;
        return field(w, 47 - 3 * k, 45 - 3 * k);
    }

    /// Alpha of COMPRESSED_RGBA8_ETC2_EAC (C.1.3, equation C.1): clamp255(base + modifier x multiplier).
    int decode_eac8(std::uint64_t w, int x, int y)
    {
        const int base = field(w, 63, 56), mult = field(w, 55, 52), table = field(w, 51, 48);
        return clamp255(base + kTableC10[static_cast<std::size_t>(table)][static_cast<std::size_t>(eac_index(w, x, y))] * mult);
    }

    /// 11-bit value of R11 EAC (C.1.5, equations C.4 / C.5: 0..2047) or signed R11 EAC (C.1.7,
    /// equations C.8 / C.9: -1023..1023).
    int decode_r11(std::uint64_t w, bool is_signed, int x, int y, Mut m)
    {
        const int mult = field(w, 55, 52), table = field(w, 51, 48);
        const int mod = kTableC10[static_cast<std::size_t>(table)][static_cast<std::size_t>(eac_index(w, x, y))];
        const int term = mult != 0 ? mod * mult * 8 : (m == Mut::EacZeroMultiplierIsZero ? 0 : mod);
        if (!is_signed) return std::clamp(field(w, 63, 56) * 8 + 4 + term, 0, 2047);
        int base = field(w, 63, 56);
        if (base >= 128) base -= 256;
        if (base == -128 && m != Mut::SignedKeepsMinus128) base = -127;
        return std::clamp(base * 8 + term, -1023, 1023);
    }

    double srgb_to_linear(double cs) { return cs <= 0.04045 ? cs / 12.92 : std::pow((cs + 0.055) / 1.055, 2.4); }

    std::uint64_t word_at(const std::vector<std::uint8_t>& data, std::size_t offset)
    {
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < 8; ++i) v = v * 256 + data[offset + i];
        return v;
    }

    // =============================================================================================
    // Textures
    // =============================================================================================

    enum class Kind { Rgb, Punch, RgbaEac, R11, Rg11, SignedR11, SignedRg11 };

    int block_bytes(Kind k) { return k == Kind::RgbaEac || k == Kind::Rg11 || k == Kind::SignedRg11 ? 16 : 8; }

    struct TexDesc
    {
        const char* name;
        CIF format;
        Kind kind;
        bool srgb;
        int w, h, layers;
        bool array;
        std::vector<std::uint8_t> data;  ///< final contents (after every sub-image update), layer after layer
    };

    /// Expected sampled value of one texel: value, the unit in which it is compared, and the tolerance.
    struct Expect
    {
        std::array<double, 4> v{};
        std::array<double, 4> unit{255, 255, 255, 255};
        std::array<double, 4> tol{};
    };

    // Exact formats: the sampled float must be the stored code divided by 2^b - 1 (ES 3.0 §2.1.6.1),
    // within float rounding; 0.45 code units accepts that and nothing else. sRGB: equation 3.26 has no
    // stated accuracy and real decoders use lookup tables; 1/255 is the suite's 8-bit policy
    // (docs/VALIDATION.md, see es30-srgb for the measured decode error of each driver).
    constexpr double kExact = 0.45;
    constexpr double kSrgbTol = 1.0;

    Expect expect_texel(const TexDesc& t, int layer, int i, int j, Mut m)
    {
        const int bw = (t.w + 3) / 4, bh = (t.h + 3) / 4;
        const std::size_t off = static_cast<std::size_t>(((layer * bh + j / 4) * bw + i / 4) * block_bytes(t.kind));
        const int x = i % 4, y = j % 4;
        Expect e;
        e.tol = {kExact, kExact, kExact, kExact};
        auto set_rgb = [&](const Rgba& c) {
            for (std::size_t n = 0; n < 3; ++n)
            {
                const double cs = c[n] / 255.0;
                e.v[n] = t.srgb ? srgb_to_linear(cs) : cs;
                if (t.srgb) e.tol[n] = kSrgbTol;
            }
        };
        switch (t.kind)
        {
        case Kind::Rgb:
        case Kind::Punch:
        {
            const Rgba c = decode_etc2(word_at(t.data, off), t.kind == Kind::Punch, x, y, m);
            set_rgb(c);
            e.v[3] = c[3] / 255.0;  // Rgb: always 255
            break;
        }
        case Kind::RgbaEac:
        {
            // C.1.3: the first 8 bytes are the alpha word, the second 8 bytes the RGB8_ETC2 color word.
            set_rgb(decode_etc2(word_at(t.data, off + 8), false, x, y, m));
            e.v[3] = decode_eac8(word_at(t.data, off), x, y) / 255.0;
            break;
        }
        case Kind::R11:
        case Kind::Rg11:
        case Kind::SignedR11:
        case Kind::SignedRg11:
        {
            const bool is_signed = t.kind == Kind::SignedR11 || t.kind == Kind::SignedRg11;
            const double scale = is_signed ? 1023.0 : 2047.0;
            e.v[0] = decode_r11(word_at(t.data, off), is_signed, x, y, m) / scale;
            e.unit[0] = scale;
            if (t.kind == Kind::Rg11 || t.kind == Kind::SignedRg11)
            {
                e.v[1] = decode_r11(word_at(t.data, off + 8), is_signed, x, y, m) / scale;  // C.1.6: second word = green
                e.unit[1] = scale;
            }
            e.v[3] = 1.0;  // RED / RG base formats sample as (r, 0, 0, 1) / (r, g, 0, 1) (ES 3.0 Table 3.24)
            break;
        }
        }
        return e;
    }

    // ---------------------------------------------------------------------------------------------
    // Block contents. Each pattern lists the pixel indices row by row (v = 0 first).
    // ---------------------------------------------------------------------------------------------
    enum : std::size_t { kRgb, kSrgb, kPunch, kSrgbPunch, kRgba, kSrgba, kR11, kSignedR11, kRg11, kSignedRg11, kArray, kPad, kTexCount };

    // RGB8_ETC2 blocks shared by several textures.
    const std::uint64_t kIndividual = enc_individual(14, 3, 8, 2, 9, 13, 0, 5, 0, {"0123", "1230", "2301", "3012"});
    // Differential: the base color (231, 8, 16) and modifier 29 of the C.1.1 example; dB (+3) != dG (+2).
    const std::uint64_t kDifferential = enc_differential(28, 1, 2, -4, 2, 3, 2, 6, 1, 1, {"1103", "1230", "3201", "0312"});
    // T: distance index 101b = 5 (the C.1.1 example: "da 10 binary, db 1 binary ... distance 32").
    const std::uint64_t kTMode = enc_t(12, 3, 1, 2, 8, 14, 5, 1, {"0011", "0123", "2233", "3210"});
    const std::uint64_t kHModeGe = enc_h(13, 10, 2, 3, 6, 12, 0, 1, 1, {"0123", "3210", "0213", "1302"});  // base 1 >= base 2
    const std::uint64_t kHModeLt = enc_h(2, 5, 9, 12, 4, 1, 1, 1, 1, {"3120", "0312", "2031", "1203"});   // base 1 < base 2
    // Planar: red overshoots 255, green undershoots 0 towards the far corner, blue is a smooth ramp.
    const std::uint64_t kPlanar = enc_planar(40, 100, 5, 63, 10, 40, 63, 0, 20, 1);
    const std::uint64_t kPlaceholder = enc_individual(8, 8, 8, 8, 8, 8, 0, 0, 0, {"0000", "0000", "0000", "0000"});
    const std::uint64_t kWhite = enc_individual(15, 15, 15, 15, 15, 15, 0, 0, 0, {"0000", "0000", "0000", "0000"});
    // Written by glCompressedTexSubImage3D into the right block of array layers 1 and 2.
    const std::uint64_t kArraySub0 = kHModeLt;
    const std::uint64_t kArraySub1 = kHModeGe;

    std::array<TexDesc, kTexCount> make_textures()
    {
        std::array<TexDesc, kTexCount> t{};
        // 12 x 8 RGB8: blocks (bx, by) in memory order by * 3 + bx; block (2, 1) is replaced by
        // glCompressedTexSubImage2D with the second H block.
        t[kRgb] = {"RGB8_ETC2", CIF::Rgb8Etc2, Kind::Rgb, false, 12, 8, 1, false, bytes({kIndividual, kDifferential, kTMode, kHModeGe, kPlanar, kHModeLt})};
        t[kSrgb] = {"SRGB8_ETC2", CIF::Srgb8Etc2, Kind::Rgb, true, 8, 4, 1, false,
                    bytes({enc_differential(20, 9, 26, 3, -2, 1, 4, 1, 0, 1, {"0123", "0123", "3210", "2301"}), enc_planar(12, 20, 60, 50, 120, 10, 30, 70, 33, 1)})};
        t[kPunch] = {"RGB8_PUNCHTHROUGH_ALPHA1_ETC2", CIF::Rgb8PunchthroughAlpha1Etc2, Kind::Punch, false, 12, 8, 1, false,
                     bytes({
                         enc_differential(10, 20, 25, 3, -3, -1, 1, 4, 0, 1, {"0123", "2301", "1032", "3210"}),  // opaque
                         // The C.1.1/C.1.9 example R = 28, dR = -4, G = 4, dG = 2, B = 3, dB = 0: index 0 gives the
                         // base colors themselves, (231, 33, 24) and (198, 49, 24); index 2 is transparent.
                         enc_differential(28, 4, 3, -4, 2, 0, 2, 3, 1, 0, {"0102", "2013", "0320", "1002"}),
                         enc_t(15, 9, 0, 3, 7, 11, 6, 0, {"0123", "1230", "2301", "3012"}),
                         enc_h(4, 12, 7, 11, 3, 9, 1, 0, 0, {"0213", "2130", "1302", "3021"}),
                         enc_planar(20, 30, 40, 60, 100, 10, 5, 120, 63, 0),  // opaque bit 0 is treated as 1
                         enc_t(1, 14, 6, 9, 2, 13, 2, 1, {"3300", "2211", "1122", "0033"}),
                     })};
        t[kSrgbPunch] = {"SRGB8_PUNCHTHROUGH_ALPHA1_ETC2", CIF::Srgb8PunchthroughAlpha1Etc2, Kind::Punch, true, 4, 4, 1, false,
                         bytes({enc_differential(20, 12, 28, 1, 3, -2, 5, 1, 0, 0, {"0123", "1203", "3021", "2310"})})};
        t[kRgba] = {"RGBA8_ETC2_EAC", CIF::Rgba8Etc2Eac, Kind::RgbaEac, false, 12, 4, 1, false,
                    bytes({
                        // The C.1.3 example: base 103, table 13, multiplier 2; pixel b (x 0, y 1) has index 3 -> 83.
                        enc_eac(103, 2, 13, {"0457", "3612", "7035", "1264"}), enc_planar(30, 60, 50, 10, 127, 63, 60, 0, 20, 1),
                        enc_eac(240, 0, 4, {"0123", "4567", "7654", "3210"}), kIndividual,  // multiplier 0: modifier x 0
                        enc_eac(10, 15, 0, {"7777", "0123", "4567", "3333"}), kHModeGe,  // clamps at 0 and stays below 255
                    })};
        t[kSrgba] = {"SRGB8_ALPHA8_ETC2_EAC", CIF::Srgb8Alpha8Etc2Eac, Kind::RgbaEac, true, 4, 4, 1, false,
                     bytes({enc_eac(128, 5, 3, {"0167", "2345", "5432", "7610"}), kTMode})};
        t[kR11] = {"R11_EAC", CIF::R11Eac, Kind::R11, false, 12, 4, 1, false,
                   bytes({
                       enc_eac(103, 2, 13, {"0457", "3612", "7035", "1264"}),  // C.1.5 example: pixel b -> 668
                       enc_eac(103, 0, 13, {"0457", "3612", "7035", "1264"}),  // multiplier 0 -> 1/8: pixel b -> 818
                       enc_eac(255, 1, 15, {"0123", "4567", "4567", "0123"}),  // the C.1.5 "reaches 1.0" example
                   })};
        t[kSignedR11] = {"SIGNED_R11_EAC", CIF::SignedR11Eac, Kind::SignedR11, false, 12, 4, 1, false,
                         bytes({
                             enc_eac(60, 2, 13, {"0457", "3612", "7035", "1264"}),  // C.1.7 example: pixel b -> 320
                             enc_eac(60, 0, 13, {"0457", "3612", "7035", "1264"}),  // multiplier 0: pixel b -> 470
                             enc_eac(-128, 15, 0, {"0123", "4567", "7654", "3210"}),  // -128 is read as -127; clamps at -1023
                         })};
        t[kRg11] = {"RG11_EAC", CIF::Rg11Eac, Kind::Rg11, false, 4, 4, 1, false,
                    bytes({enc_eac(128, 15, 0, {"0123", "4567", "7654", "3210"}), enc_eac(90, 9, 3, {"4567", "0123", "3210", "7654"})})};  // both clamp at 0 and 2047
        t[kSignedRg11] = {"SIGNED_RG11_EAC", CIF::SignedRg11Eac, Kind::SignedRg11, false, 4, 4, 1, false,
                          bytes({enc_eac(0, 15, 1, {"0123", "4567", "7654", "3210"}), enc_eac(-40, 11, 4, {"4567", "0123", "3210", "7654"})})};  // red clamps at +-1023
        // 8 x 4 x 3 array: layer 0 [individual, differential], layer 1 [T, H], layer 2 [planar, individual];
        // glCompressedTexSubImage3D replaces the right block of layers 1 and 2.
        t[kArray] = {"RGB8_ETC2 2D array", CIF::Rgb8Etc2, Kind::Rgb, false, 8, 4, 3, true,
                     bytes({kIndividual, kDifferential, kTMode, kArraySub0, kPlanar, kArraySub1})};
        // 6 x 6: 2 x 2 blocks, the texels beyond 6 are padding; the top-right 2 x 2 texels are respecified.
        t[kPad] = {"RGB8_ETC2 (padded)", CIF::Rgb8Etc2, Kind::Rgb, false, 6, 6, 1, false, bytes({kTMode, kPlanar, kDifferential, kHModeLt})};
        return t;
    }

    // =============================================================================================
    // Picture layout: one cell per format and mode, each showing a block (or a whole small texture)
    // magnified with NEAREST. Cells are laid out from the surface size.
    // =============================================================================================
    struct Cell
    {
        std::size_t tex;
        int layer, tx, ty, tw, th;  // texel rectangle shown
    };

    std::vector<Cell> make_cells()
    {
        std::vector<Cell> c;
        auto blocks = [&](std::size_t tex, int bw, int bh) {
            for (int by = 0; by < bh; ++by)
                for (int bx = 0; bx < bw; ++bx) c.push_back({tex, 0, 4 * bx, 4 * by, 4, 4});
        };
        blocks(kRgb, 3, 2);
        blocks(kSrgb, 2, 1);
        blocks(kPunch, 3, 2);
        blocks(kSrgbPunch, 1, 1);
        blocks(kRgba, 3, 1);
        blocks(kSrgba, 1, 1);
        blocks(kR11, 3, 1);
        blocks(kSignedR11, 3, 1);
        blocks(kRg11, 1, 1);
        blocks(kSignedRg11, 1, 1);
        for (int layer = 0; layer < 3; ++layer) c.push_back({kArray, layer, 0, 0, 8, 4});
        c.push_back({kPad, 0, 0, 0, 6, 6});
        return c;
    }

    struct Quad
    {
        double x0, y0, x1, y1;  // window pixels
    };

    struct Grid
    {
        int cols = 1, rows = 1;
        double cw = 0, ch = 0;
    };

    Grid grid_for(int width, int height, int n)
    {
        Grid best;
        double best_size = -1;
        for (int cols = 1; cols <= n; ++cols)
        {
            const int rows = (n + cols - 1) / cols;
            const double size = std::min(static_cast<double>(width) / cols, static_cast<double>(height) / rows);
            if (size > best_size) best_size = size, best = {cols, rows, static_cast<double>(width) / cols, static_cast<double>(height) / rows};
        }
        return best;
    }

    Quad quad_for(const Grid& g, std::size_t index, int height)
    {
        const int col = static_cast<int>(index) % g.cols, row = static_cast<int>(index) / g.cols;
        const double x0 = col * g.cw, y1 = height - row * g.ch;
        const double margin = 0.07 * std::min(g.cw, g.ch);
        return {x0 + margin, y1 - g.ch + margin, x0 + g.cw - margin, y1 - margin};
    }

    int display_mode(Kind k)
    {
        switch (k)
        {
        case Kind::Rgb: return 0;
        case Kind::Punch:
        case Kind::RgbaEac: return 1;
        case Kind::R11:
        case Kind::Rg11: return 2;
        default: return 3;
        }
    }

    /// CPU model of show.frag for pixel (px, py).
    std::array<double, 3> display_value(const Expect& e, int mode, int px, int py)
    {
        if (mode == 0) return {e.v[0], e.v[1], e.v[2]};
        if (mode == 1)
        {
            const double checker = ((px / 6 + py / 6) % 2 == 0 ? 77.0 : 158.0) / 255.0;
            std::array<double, 3> r{};
            for (std::size_t n = 0; n < 3; ++n) r[n] = checker * (1.0 - e.v[3]) + e.v[n] * e.v[3];
            return r;
        }
        const double sx = mode == 3 ? 0.5 + 0.5 * e.v[0] : e.v[0], sy = mode == 3 ? 0.5 + 0.5 * e.v[1] : e.v[1];
        return {sx, sy, 1.0 - sx};
    }

    constexpr int kMinPixelsPerTexel = 3;

    class Etc2Eac final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            textures_ = make_textures();
            cells_ = make_cells();
            version_31_ = env.context_version >= demo::Version{3, 1};
            check_surface(env.width, env.height);

            show_ = demo::build_program(env, "shaders/cell.vert", "shaders/show.frag");
            probe_ = demo::build_program(env, "shaders/cell.vert", "shaders/probe.frag");
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glGenTextures(1, &scratch_2d_);
            mgl::glGenTextures(1, &scratch_3d_);

            using TT = mgl::TextureTarget;
            auto record = [&](std::string what) { upload_errors_.emplace_back(std::move(what), demo::drain_gl_errors()); };
            auto data_of = [&](std::size_t n) { return textures_[n].data.data(); };
            auto size_of = [&](std::size_t n) { return static_cast<GLsizei>(textures_[n].data.size()); };
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            // --- RGB8_ETC2 12 x 8: six blocks; block (2, 1) starts as a grey placeholder and is replaced by
            //     glCompressedTexSubImage2D (offsets and size are multiples of 4).
            {
                std::vector<std::uint8_t> initial(textures_[kRgb].data.begin(), textures_[kRgb].data.end() - 8);
                append(initial, kPlaceholder);
                mgl::glBindTexture(TT::Texture2D, tex_[kRgb]);
                mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Rgb8Etc2, 12, 8, 0, static_cast<GLsizei>(initial.size()), initial.data());
                record("glCompressedTexImage2D RGB8_ETC2 12x8 (48 bytes)");
                const std::vector<std::uint8_t> block = bytes({kHModeLt});
                mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 8, 4, 4, 4, CIF::Rgb8Etc2, 8, block.data());
                record("glCompressedTexSubImage2D RGB8_ETC2 block at (8, 4)");
            }
            // --- SRGB8_ETC2 8 x 4: immutable storage, then the whole level through glCompressedTexSubImage2D.
            mgl::glBindTexture(TT::Texture2D, tex_[kSrgb]);
            mgl::glTexStorage2D(TT::Texture2D, 1, CIF::Srgb8Etc2, 8, 4);
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 8, 4, CIF::Srgb8Etc2, size_of(kSrgb), data_of(kSrgb));
            record("glTexStorage2D SRGB8_ETC2 + glCompressedTexSubImage2D of the whole level");
            // --- the other 2D textures: one glCompressedTexImage2D each. imageSize = blocks x 8 or 16 bytes.
            for (const std::size_t n : {kPunch, kSrgbPunch, kRgba, kSrgba, kR11, kSignedR11, kRg11, kSignedRg11})
            {
                const TexDesc& t = textures_[n];
                mgl::glBindTexture(TT::Texture2D, tex_[n]);
                mgl::glCompressedTexImage2D(TT::Texture2D, 0, t.format, t.w, t.h, 0, size_of(n), data_of(n));
                record(std::format("glCompressedTexImage2D {} {}x{} ({} bytes)", t.name, t.w, t.h, t.data.size()));
            }
            // --- RGB8_ETC2 2D array 8 x 4 x 3 with glCompressedTexImage3D, then glCompressedTexSubImage3D
            //     replaces the right block (xoffset 4) of layers 1 and 2 (zoffset 1, depth 2).
            {
                std::vector<std::uint8_t> initial = textures_[kArray].data;
                const std::vector<std::uint8_t> before = bytes({kPlaceholder, kWhite});
                std::copy(before.begin(), before.begin() + 8, initial.begin() + 24);  // layer 1, block 1
                std::copy(before.begin() + 8, before.end(), initial.begin() + 40);    // layer 2, block 1
                mgl::glBindTexture(TT::Texture2DArray, tex_[kArray]);
                mgl::glCompressedTexImage3D(TT::Texture2DArray, 0, CIF::Rgb8Etc2, 8, 4, 3, 0, static_cast<GLsizei>(initial.size()), initial.data());
                record("glCompressedTexImage3D TEXTURE_2D_ARRAY RGB8_ETC2 8x4x3 (48 bytes)");
                const std::vector<std::uint8_t> sub = bytes({kArraySub0, kArraySub1});
                mgl::glCompressedTexSubImage3D(TT::Texture2DArray, 0, 4, 0, 1, 4, 4, 2, CIF::Rgb8Etc2, 16, sub.data());
                record("glCompressedTexSubImage3D TEXTURE_2D_ARRAY (4, 0, 1) 4x4x2");
            }
            // --- RGB8_ETC2 6 x 6 (not a multiple of 4): 2 x 2 blocks = 32 bytes. Then a 2 x 2 update at (4, 4):
            //     width 2 is not a multiple of 4 but xoffset + width equals the level width (ES 3.0 §3.8.6).
            {
                std::vector<std::uint8_t> initial(textures_[kPad].data.begin(), textures_[kPad].data.end() - 8);
                append(initial, kPlaceholder);
                mgl::glBindTexture(TT::Texture2D, tex_[kPad]);
                mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Rgb8Etc2, 6, 6, 0, 32, initial.data());
                record("glCompressedTexImage2D RGB8_ETC2 6x6 (32 bytes)");
                const std::vector<std::uint8_t> block = bytes({kHModeLt});
                mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 4, 4, 2, 2, CIF::Rgb8Etc2, 8, block.data());
                record("glCompressedTexSubImage2D RGB8_ETC2 2x2 at (4, 4) of the 6x6 texture");
            }
            // NEAREST, no mipmaps (the default NEAREST_MIPMAP_LINEAR would make the textures incomplete).
            for (std::size_t n = 0; n < kTexCount; ++n)
            {
                const TT target = textures_[n].array ? TT::Texture2DArray : TT::Texture2D;
                mgl::glBindTexture(target, tex_[n]);
                mgl::glTexParameter(target, mgl::TextureMinFilter::Nearest);
                mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);
                mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
                mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            }
            // Scratch textures for the error cases: an uncompressed RGBA8 3D texture and an empty 2D name.
            mgl::glBindTexture(TT::Texture3D, scratch_3d_);
            mgl::glTexImage3D(TT::Texture3D, 0, mgl::InternalFormat::Rgba8, 4, 4, 4, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            record("glTexImage3D RGBA8 4x4x4 scratch");
            mgl::glBindTexture(TT::Texture2D, scratch_2d_);

            // The array texture stays on unit 1 for the whole run; 2D textures are bound on unit 0.
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(TT::Texture2DArray, tex_[kArray]);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            // Exact readback target: RGBA32UI (color-renderable in ES 3.0, Table 3.13), as large as the
            // largest texture level.
            mgl::glGenRenderbuffers(1, &rbo_);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rbo_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba32UI, 12, 8);
            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rbo_);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("RGBA32UI readback target incomplete: {}", demo::framebuffer_status_name(status)));

            for (const mgl::ProgramId p : {show_, probe_})
            {
                mgl::glUseProgram(p);
                mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_tex2d"), 0);
                mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_array"), 1);
            }
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glClearColor(0.07f, 0.075f, 0.09f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(show_);
            const mgl::UniformLocation rect = mgl::glGetUniformLocation(show_, "u_rect");
            const mgl::UniformLocation uv = mgl::glGetUniformLocation(show_, "u_uv");
            const mgl::UniformLocation is_array = mgl::glGetUniformLocation(show_, "u_is_array");
            const mgl::UniformLocation layer = mgl::glGetUniformLocation(show_, "u_layer");
            const mgl::UniformLocation mode = mgl::glGetUniformLocation(show_, "u_mode");
            const Grid g = grid_for(env.width, env.height, static_cast<int>(cells_.size()));
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (std::size_t c = 0; c < cells_.size(); ++c)
            {
                const Cell& cell = cells_[c];
                const TexDesc& t = textures_[cell.tex];
                const Quad q = quad_for(g, c, env.height);
                if (!t.array) mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[cell.tex]);
                mgl::glUniform4f(rect, static_cast<float>(2.0 * q.x0 / env.width - 1.0), static_cast<float>(2.0 * q.y0 / env.height - 1.0),
                                 static_cast<float>(2.0 * q.x1 / env.width - 1.0), static_cast<float>(2.0 * q.y1 / env.height - 1.0));
                mgl::glUniform4f(uv, static_cast<float>(cell.tx) / static_cast<float>(t.w), static_cast<float>(cell.ty) / static_cast<float>(t.h),
                                 static_cast<float>(cell.tx + cell.tw) / static_cast<float>(t.w), static_cast<float>(cell.ty + cell.th) / static_cast<float>(t.h));
                mgl::glUniform1i(is_array, t.array ? 1 : 0);
                mgl::glUniform1f(layer, static_cast<float>(cell.layer));
                mgl::glUniform1i(mode, display_mode(t.kind));
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            for (const auto& [what, error] : upload_errors_)
                check.expect(error == mgl::ErrorCode::NoError, std::format("{} -> no error", what), demo::error_name(error));

            check_picture(env, check);  // reads the window before anything else is drawn
            check_format_queries(check);
            check_level_queries(env, check);
            check_errors(check);         // every rejected call would have changed texels ...
            read_all_textures();         // ... so the exact comparison below also proves they had no effect
            check_exact(check);
            check_spec_examples(check);
            check_negative_controls(check);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteRenderbuffers(1, &rbo_);
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteTextures(1, &scratch_2d_);
            mgl::glDeleteTextures(1, &scratch_3d_);
            mgl::glDeleteProgram(show_);
            mgl::glDeleteProgram(probe_);
        }

    private:
        void check_surface(int width, int height) const
        {
            const Grid g = grid_for(width, height, static_cast<int>(cells_.size()));
            for (std::size_t c = 0; c < cells_.size(); ++c)
            {
                const Quad q = quad_for(g, c, height);
                if ((q.x1 - q.x0) / cells_[c].tw < kMinPixelsPerTexel || (q.y1 - q.y0) / cells_[c].th < kMinPixelsPerTexel)
                    throw demo::Skip(std::format("surface {}x{} is too small: cells need {} pixels per texel", width, height, kMinPixelsPerTexel));
            }
        }

        // ---------------------------------------------------------------------------------------------
        // The magnified picture: one probe per texel, at the pixel nearest the texel centre. The texel
        // that pixel samples is recomputed from the interpolated coordinate; if it lies within 2% of a
        // texel edge, either neighbour is accepted.
        // ---------------------------------------------------------------------------------------------
        void check_picture(demo::Env& env, demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, env.width, env.height);
            const Grid g = grid_for(env.width, env.height, static_cast<int>(cells_.size()));
            int probes = 0, bad = 0;
            double max_dev = 0;
            std::string first;
            for (std::size_t c = 0; c < cells_.size(); ++c)
            {
                const Cell& cell = cells_[c];
                const TexDesc& t = textures_[cell.tex];
                const Quad q = quad_for(g, c, env.height);
                const double qw = q.x1 - q.x0, qh = q.y1 - q.y0;
                const int mode = display_mode(t.kind);
                for (int tj = 0; tj < cell.th; ++tj)
                    for (int ti = 0; ti < cell.tw; ++ti)
                    {
                        const int x = static_cast<int>(std::floor(q.x0 + (ti + 0.5) * qw / cell.tw));
                        const int y = static_cast<int>(std::floor(q.y0 + (tj + 0.5) * qh / cell.th));
                        const double s = cell.tx + (x + 0.5 - q.x0) / qw * cell.tw;  // texel-space coordinate at the pixel centre
                        const double tt = cell.ty + (y + 0.5 - q.y0) / qh * cell.th;
                        auto candidates = [](double coord, int lo, int hi) {
                            std::vector<int> r{std::clamp(static_cast<int>(std::floor(coord)), lo, hi)};
                            const double f = coord - std::floor(coord);
                            if (f < 0.02) r.push_back(std::clamp(r[0] - 1, lo, hi));
                            if (f > 0.98) r.push_back(std::clamp(r[0] + 1, lo, hi));
                            return r;
                        };
                        const std::size_t at = (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4;
                        bool ok = false;
                        double best = 1e9;
                        std::array<double, 3> best_want{};
                        for (const int i : candidates(s, cell.tx, cell.tx + cell.tw - 1))
                            for (const int j : candidates(tt, cell.ty, cell.ty + cell.th - 1))
                            {
                                const Expect e = expect_texel(t, cell.layer, i, j, Mut::None);
                                const std::array<double, 3> want = display_value(e, mode, x, y);
                                // 8-bit output of an exact code: the code itself; everything computed (sRGB decode,
                                // EAC 11-bit values, checker blend, remapping) is quantized to 8 bits: +-1.
                                const double tol = (mode == 0 && !t.srgb) ? 0.5 : 1.0;
                                double dev = std::abs(px[at + 3] - 255.0);
                                for (std::size_t n = 0; n < 3; ++n) dev = std::max(dev, std::abs(px[at + n] - 255.0 * want[n]));
                                if (dev < best) best = dev, best_want = want;
                                ok = ok || dev <= tol;
                            }
                        ++probes;
                        max_dev = std::max(max_dev, best);
                        if (!ok && bad++ == 0)
                            first = std::format("cell {} ({}), texel ({}, {}) at pixel ({}, {}): got ({}, {}, {}, {}) want ({:.1f}, {:.1f}, {:.1f}, 255)", c, t.name,
                                                cell.tx + ti, cell.ty + tj, x, y, px[at], px[at + 1], px[at + 2], px[at + 3], 255 * best_want[0],
                                                255 * best_want[1], 255 * best_want[2]);
                    }
            }
            std::cout << std::format("   note: picture {}x{}, {} cells, largest deviation from the prediction {:.2f} (8-bit steps)\n", env.width, env.height,
                                     cells_.size(), max_dev);
            check.expect(probes > 400 && bad == 0,
                         std::format("magnified picture: all {} texels of the {} cells (NEAREST) match the CPU decoder (exact codes exact, computed "
                                     "values within 1/255)",
                                     probes, cells_.size()),
                         std::format("{} wrong; first: {}", bad, first));
        }

        void check_format_queries(demo::Check& check)
        {
            const int count = demo::get_integer(mgl::GetParameter::NumCompressedTextureFormats);
            std::vector<GLint> formats(static_cast<std::size_t>(std::max(count, 0)) + 1, -1);
            mgl::glGetIntegerv(mgl::GetParameter::CompressedTextureFormats, formats.data());
            check.expect(count >= 10, "NUM_COMPRESSED_TEXTURE_FORMATS >= 10 (ES 3.0 Table 6.29 minimum)", std::format("reported {}", count));
            check.expect_eq(formats.back(), -1, "COMPRESSED_TEXTURE_FORMATS writes exactly NUM_COMPRESSED_TEXTURE_FORMATS values");
            const std::array<std::pair<CIF, const char*>, 10> required = {{
                {CIF::R11Eac, "R11_EAC"}, {CIF::SignedR11Eac, "SIGNED_R11_EAC"}, {CIF::Rg11Eac, "RG11_EAC"}, {CIF::SignedRg11Eac, "SIGNED_RG11_EAC"},
                {CIF::Rgb8Etc2, "RGB8_ETC2"}, {CIF::Srgb8Etc2, "SRGB8_ETC2"}, {CIF::Rgb8PunchthroughAlpha1Etc2, "RGB8_PUNCHTHROUGH_ALPHA1_ETC2"},
                {CIF::Srgb8PunchthroughAlpha1Etc2, "SRGB8_PUNCHTHROUGH_ALPHA1_ETC2"}, {CIF::Rgba8Etc2Eac, "RGBA8_ETC2_EAC"},
                {CIF::Srgb8Alpha8Etc2Eac, "SRGB8_ALPHA8_ETC2_EAC"},
            }};
            std::string missing;
            for (const auto& [format, name] : required)
                if (std::find(formats.begin(), formats.end() - 1, static_cast<GLint>(format)) == formats.end() - 1) missing += std::format("{} ", name);
            check.expect(missing.empty(), "COMPRESSED_TEXTURE_FORMATS lists all ten formats of Table 3.19", "missing: " + missing);

            // TEXTURE_IMMUTABLE_FORMAT (ES 3.0 §3.8.4): TRUE after glTexStorage2D with a compressed format.
            GLint immutable_srgb = -1, immutable_rgb = -1;
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kSrgb]);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::ImmutableFormat, &immutable_srgb);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kRgb]);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::ImmutableFormat, &immutable_rgb);
            check.expect(immutable_srgb == GL_TRUE && immutable_rgb == GL_FALSE,
                         "TEXTURE_IMMUTABLE_FORMAT: TRUE for glTexStorage2D(SRGB8_ETC2), FALSE for glCompressedTexImage2D",
                         std::format("{} / {}", immutable_srgb, immutable_rgb));
        }

        /// glGetTexLevelParameteriv is ES 3.1 (ES 3.2 §8.11.3); on an ES 3.0 context this part is skipped.
        void check_level_queries(demo::Env& env, demo::Check& check)
        {
            if (env.context_version < demo::Version{3, 1})
            {
                std::cout << std::format("   note: glGetTexLevelParameteriv (TEXTURE_COMPRESSED) is ES 3.1; the context reports ES {}, skipped\n",
                                         demo::to_string(env.context_version));
                return;
            }
            using P = mgl::TextureLevelParameter;
            std::string problems;
            for (std::size_t n = 0; n < kTexCount; ++n)
            {
                const TexDesc& t = textures_[n];
                const mgl::TextureTarget target = t.array ? mgl::TextureTarget::Texture2DArray : mgl::TextureTarget::Texture2D;
                mgl::glBindTexture(target, tex_[n]);
                GLint compressed = -1, format = -1, w = -1, h = -1, d = -1;
                mgl::glGetTexLevelParameteriv(target, 0, P::Compressed, &compressed);
                mgl::glGetTexLevelParameteriv(target, 0, P::InternalFormat, &format);
                mgl::glGetTexLevelParameteriv(target, 0, P::Width, &w);
                mgl::glGetTexLevelParameteriv(target, 0, P::Height, &h);
                mgl::glGetTexLevelParameteriv(target, 0, P::Depth, &d);
                if (compressed != GL_TRUE || format != static_cast<GLint>(t.format) || w != t.w || h != t.h || d != t.layers)
                    problems += std::format("{}: COMPRESSED {} INTERNAL_FORMAT {:#x} size {}x{}x{}; ", t.name, compressed, format, w, h, d);
            }
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kArray]);  // unit 1 keeps the array
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            GLint compressed_3d = -1;
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, scratch_3d_);
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture3D, 0, P::Compressed, &compressed_3d);
            if (compressed_3d != GL_FALSE) problems += std::format("RGBA8 3D: COMPRESSED {}; ", compressed_3d);
            check.expect(problems.empty(),
                         "glGetTexLevelParameteriv: TEXTURE_COMPRESSED TRUE, INTERNAL_FORMAT and WIDTH/HEIGHT/DEPTH as specified for all 12 "
                         "compressed images (6x6 included); COMPRESSED FALSE for RGBA8",
                         problems);
        }

        // ---------------------------------------------------------------------------------------------
        // Errors of ES 3.0 §3.8.6 (ES 3.2 §8.7). Each rejected call carries data that would visibly change
        // the texels; read_all_textures() + check_exact() afterwards prove the commands were ignored.
        // ---------------------------------------------------------------------------------------------
        void check_errors(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            using E = mgl::ErrorCode;
            const std::vector<std::uint8_t> white = bytes({kWhite, kWhite, kWhite, kWhite});
            demo::drain_gl_errors();

            mgl::glBindTexture(TT::Texture2D, tex_[kRgb]);  // 12 x 8 RGB8_ETC2
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 2, 0, 4, 4, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexSubImage2D xoffset 2 (not a multiple of 4) -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 4, 2, 4, 4, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexSubImage2D yoffset 2 (not a multiple of 4) -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 4, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation,
                                  "glCompressedTexSubImage2D width 2 (not a multiple of 4, xoffset + width != 12) -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 4, 4, 2, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation,
                                  "glCompressedTexSubImage2D height 2 at yoffset 4 (yoffset + height = 6 != 8) -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 4, 4, CIF::Srgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation,
                                  "glCompressedTexSubImage2D format SRGB8_ETC2 into an RGB8_ETC2 image (no format conversion) -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 4, 4, CIF::Rgba8Etc2Eac, 16, white.data());
            check.expect_gl_error(E::InvalidOperation,
                                  "glCompressedTexSubImage2D format RGBA8_ETC2_EAC into an RGB8_ETC2 image -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 8, 0, 8, 4, CIF::Rgb8Etc2, 16, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexSubImage2D xoffset 8 + width 8 > 12 -> INVALID_VALUE (TexSubImage2D bounds)");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 4, 4, CIF::Rgb8Etc2, 7, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexSubImage2D imageSize 7 for one 8-byte block -> INVALID_VALUE");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 4, 4, CIF::Rgb8Etc2, 16, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexSubImage2D imageSize 16 for one 8-byte block (too much) -> INVALID_VALUE");
            mgl::glCompressedTexSubImage2D(TT::Texture2D, 0, 0, 0, 4, 4, CIF::Rgb8Etc2, -8, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexSubImage2D negative imageSize -> INVALID_VALUE");
            mgl::glCompressedTexSubImage2D(TT::Texture3D, 0, 0, 0, 4, 4, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidEnum, "glCompressedTexSubImage2D target TEXTURE_3D -> INVALID_ENUM (ES 3.2 §8.7)");

            mgl::glBindTexture(TT::Texture2D, scratch_2d_);  // never specified: any accepted call would leave a level behind
            mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Rgb8Etc2, 4, 4, 0, 7, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexImage2D RGB8_ETC2 4x4 with imageSize 7 (needs 8) -> INVALID_VALUE");
            mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Rgba8Etc2Eac, 8, 4, 0, 16, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexImage2D RGBA8_ETC2_EAC 8x4 with imageSize 16 (needs 32) -> INVALID_VALUE");
            mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::R11Eac, 4, 4, 0, 16, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexImage2D R11_EAC 4x4 with imageSize 16 (needs 8) -> INVALID_VALUE");
            mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Rgb8Etc2, 5, 5, 0, 8, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexImage2D RGB8_ETC2 5x5 with imageSize 8 (padded to 2x2 blocks = 32) -> INVALID_VALUE");
            mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Rgb8Etc2, 4, 4, 1, 8, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexImage2D border 1 -> INVALID_VALUE");
            mgl::glCompressedTexImage2D(TT::Texture2DArray, 0, CIF::Rgb8Etc2, 4, 4, 0, 8, white.data());
            check.expect_gl_error(E::InvalidEnum, "glCompressedTexImage2D target TEXTURE_2D_ARRAY -> INVALID_ENUM (ES 3.2 §8.7)");
            if (version_31_)  // glGetTexLevelParameteriv is ES 3.1
            {
                GLint w = -1;
                mgl::glGetTexLevelParameteriv(TT::Texture2D, 0, mgl::TextureLevelParameter::Width, &w);
                check.expect_eq(w, 0, "the rejected glCompressedTexImage2D calls left level 0 of the scratch texture unspecified (TEXTURE_WIDTH 0)");
            }

            mgl::glBindTexture(TT::Texture2D, tex_[kSrgb]);  // immutable (glTexStorage2D)
            mgl::glCompressedTexImage2D(TT::Texture2D, 0, CIF::Srgb8Etc2, 8, 4, 0, 16, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexImage2D on an immutable texture (same size and format) -> INVALID_OPERATION");

            mgl::glBindTexture(TT::Texture3D, scratch_3d_);
            mgl::glCompressedTexImage3D(TT::Texture3D, 0, CIF::Rgb8Etc2, 4, 4, 1, 0, 8, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexImage3D TEXTURE_3D with RGB8_ETC2 -> INVALID_OPERATION (ETC2/EAC: 2D arrays only)");
            mgl::glCompressedTexImage3D(TT::Texture3D, 0, CIF::R11Eac, 4, 4, 2, 0, 16, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexImage3D TEXTURE_3D with R11_EAC -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage3D(TT::Texture3D, 0, 0, 0, 0, 4, 4, 1, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexSubImage3D TEXTURE_3D with RGB8_ETC2 -> INVALID_OPERATION");

            mgl::glActiveTexture(mgl::TextureUnit::Texture1);  // the array lives on unit 1
            mgl::glCompressedTexSubImage3D(TT::Texture2DArray, 0, 0, 0, 0, 4, 4, 2, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexSubImage3D 4x4x2 with imageSize 8 (needs 16) -> INVALID_VALUE");
            mgl::glCompressedTexSubImage3D(TT::Texture2DArray, 0, 0, 0, 0, 4, 4, 1, CIF::R11Eac, 8, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexSubImage3D format R11_EAC into the RGB8_ETC2 array -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage3D(TT::Texture2DArray, 0, 2, 0, 0, 4, 4, 1, CIF::Rgb8Etc2, 8, white.data());
            check.expect_gl_error(E::InvalidOperation, "glCompressedTexSubImage3D xoffset 2 -> INVALID_OPERATION");
            mgl::glCompressedTexSubImage3D(TT::Texture2DArray, 0, 0, 0, 2, 4, 4, 2, CIF::Rgb8Etc2, 16, white.data());
            check.expect_gl_error(E::InvalidValue, "glCompressedTexSubImage3D zoffset 2 + depth 2 > 3 layers -> INVALID_VALUE");
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            check.expect_no_gl_error("no error left over after the error cases");
        }

        // ---------------------------------------------------------------------------------------------
        // Exact path: every texture (every layer) drawn 1:1 into the RGBA32UI target.
        // ---------------------------------------------------------------------------------------------
        void read_all_textures()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glUseProgram(probe_);
            const mgl::UniformLocation rect = mgl::glGetUniformLocation(probe_, "u_rect");
            const mgl::UniformLocation uv = mgl::glGetUniformLocation(probe_, "u_uv");
            const mgl::UniformLocation is_array = mgl::glGetUniformLocation(probe_, "u_is_array");
            const mgl::UniformLocation layer = mgl::glGetUniformLocation(probe_, "u_layer");
            const mgl::UniformLocation size = mgl::glGetUniformLocation(probe_, "u_size");
            mgl::glUniform4f(rect, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glUniform4f(uv, 0.0f, 0.0f, 1.0f, 1.0f);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (std::size_t n = 0; n < kTexCount; ++n)
            {
                const TexDesc& t = textures_[n];
                readback_[n].assign(static_cast<std::size_t>(t.layers), {});
                if (!t.array) mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[n]);
                mgl::glUniform1i(is_array, t.array ? 1 : 0);
                mgl::glUniform2f(size, static_cast<float>(t.w), static_cast<float>(t.h));
                for (int l = 0; l < t.layers; ++l)
                {
                    mgl::glViewport(0, 0, t.w, t.h);
                    mgl::glUniform1f(layer, static_cast<float>(l));
                    mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
                    std::vector<std::uint32_t> bits(static_cast<std::size_t>(t.w * t.h * 4));
                    mgl::glReadPixels(0, 0, t.w, t.h, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, bits.data());
                    auto& out = readback_[n][static_cast<std::size_t>(l)];
                    out.resize(static_cast<std::size_t>(t.w * t.h));
                    for (std::size_t p = 0; p < out.size(); ++p)
                        for (std::size_t c = 0; c < 4; ++c) out[p][c] = std::bit_cast<float>(bits[p * 4 + c]);
                }
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        std::array<float, 4> got(std::size_t tex, int layer, int i, int j) const
        {
            return readback_[tex][static_cast<std::size_t>(layer)][static_cast<std::size_t>(j * textures_[tex].w + i)];
        }

        /// Number of texels of `tex` whose readback disagrees with the decoder (with mutation m), and the
        /// largest deviation in compare units.
        std::pair<int, double> compare(std::size_t tex, Mut m, std::string* first = nullptr) const
        {
            const TexDesc& t = textures_[tex];
            int bad = 0;
            double max_dev = 0;
            for (int l = 0; l < t.layers; ++l)
                for (int j = 0; j < t.h; ++j)
                    for (int i = 0; i < t.w; ++i)
                    {
                        const Expect e = expect_texel(t, l, i, j, m);
                        const std::array<float, 4> g = got(tex, l, i, j);
                        bool ok = true;
                        for (std::size_t c = 0; c < 4; ++c)
                        {
                            const double dev = std::abs(static_cast<double>(g[c]) - e.v[c]) * e.unit[c];
                            max_dev = std::max(max_dev, dev);
                            ok = ok && dev <= e.tol[c];
                        }
                        if (!ok && bad++ == 0 && first)
                            *first = std::format("layer {} texel ({}, {}): got ({:.5f}, {:.5f}, {:.5f}, {:.5f}) want ({:.5f}, {:.5f}, {:.5f}, {:.5f})", l, i, j,
                                                 g[0], g[1], g[2], g[3], e.v[0], e.v[1], e.v[2], e.v[3]);
                    }
            return {bad, max_dev};
        }

        void check_exact(demo::Check& check)
        {
            for (std::size_t n = 0; n < kTexCount; ++n)
            {
                const TexDesc& t = textures_[n];
                std::string first;
                const auto [bad, max_dev] = compare(n, Mut::None, &first);
                const bool eleven = t.kind == Kind::R11 || t.kind == Kind::Rg11 || t.kind == Kind::SignedR11 || t.kind == Kind::SignedRg11;
                const std::string tolerance = t.srgb ? "RGB within 1/255 after the sRGB decode, alpha exact"
                                              : eleven ? "the exact 11-bit value (never truncated below 11 bits, C.1.5 / C.1.7)"
                                                       : "exactly the 8-bit code / 255";
                check.expect(bad == 0,
                             std::format("{} {}x{}{}: all {} texels match the Appendix C decoder ({}; largest deviation {:.3f} units)", t.name, t.w, t.h,
                                         t.layers > 1 ? std::format("x{}", t.layers) : "", t.w * t.h * t.layers, tolerance, max_dev),
                             std::format("{} texels wrong; first: {}", bad, first));
            }
        }

        // ---------------------------------------------------------------------------------------------
        // Results stated in the text of Appendix C, hard-coded here (independent of the decoder).
        // ---------------------------------------------------------------------------------------------
        void check_spec_examples(demo::Check& check)
        {
            auto code8 = [&](std::size_t tex, int i, int j) {
                const auto g = got(tex, 0, i, j);
                std::array<long, 4> c{};
                for (std::size_t n = 0; n < 4; ++n) c[n] = std::lround(static_cast<double>(g[n]) * 255.0);
                return c;
            };
            auto code11 = [&](std::size_t tex, int i, int j, double scale) { return std::lround(static_cast<double>(got(tex, 0, i, j)[0]) * scale); };
            using A = std::array<long, 4>;
            auto expect_code = [&](const A& have, const A& want, std::string_view what) {
                check.expect(have == want, what, std::format("got ({}, {}, {}, {}) expected ({}, {}, {}, {})", have[0], have[1], have[2], have[3], want[0],
                                                             want[1], want[2], want[3]));
            };
            // C.1.1 individual: R1 = 14, G1 = 3, B1 = 8 -> base (238, 51, 136); texel (0, 0) has index 0 of table 0 (+2).
            expect_code(code8(kRgb, 0, 0), A{240, 53, 138, 255}, "C.1.1 example: individual base (238, 51, 136) + modifier 2 = (240, 53, 138)");
            // C.1.1 differential: base (231, 8, 16) + 29 -> (260, 37, 45) clamped to (255, 37, 45); texel (4, 0) has index 1 of table 2.
            expect_code(code8(kRgb, 4, 0), A{255, 37, 45, 255}, "C.1.1 example: (231, 8, 16) + 29 clamps to (255, 37, 45)");
            // C.1.1 T: distance index 101b -> 32; paint color 1 = base 2 (34, 136, 238) + 32 -> (66, 168, 255); texel (8 + 1, 1) has index 1.
            expect_code(code8(kRgb, 9, 1), A{66, 168, 255, 255}, "C.1.1 example: T distance index 5 = 32, paint color 1 = (66, 168, 255)");
            // C.1.9 / C.1.1: R = 28, dR = -4, G = 4, dG = 2, B = 3, dB = 0 -> (231, 33, 24) and (198, 49, 24); with opaque = 0,
            // index 0 adds 0 (Table C.13b). Texel (4, 0) lies in subblock 1 (flip 1: rows 0-1), texel (4, 2) in subblock 2.
            expect_code(code8(kPunch, 4, 0), A{231, 33, 24, 255}, "C.1.1 example: differential base color of subblock 1 = (231, 33, 24)");
            expect_code(code8(kPunch, 4, 2), A{198, 49, 24, 255},
                            "C.1.1 example: subblock 2 = (198, 49, 24), i.e. blue uses dB (the formula's 'B + dG' is a misprint)");
            check.expect_eq(code8(kPunch, 7, 0)[3], 0L, "C.1.9: index 2 of an opaque = 0 block is transparent");
            // C.1.3: base 103, table 13, multiplier 2, pixel b (x 0, y 1) index 3 -> 83.
            check.expect_eq(code8(kRgba, 0, 1)[3], 83L, "C.1.3 example: alpha 103 + (-10 x 2) = 83");
            // C.1.5: the same codewords as R11 -> 668; with multiplier 0 -> 818.
            check.expect_eq(code11(kR11, 0, 1, 2047.0), 668L, "C.1.5 example: R11 103 x 8 + 4 - 160 = 668 (of 2047)");
            check.expect_eq(code11(kR11, 4, 1, 2047.0), 818L, "C.1.5 example: multiplier 0 acts as 1/8: 828 - 10 = 818");
            // C.1.7: base 60, table 13, multiplier 2 -> 320; multiplier 0 -> 470.
            check.expect_eq(code11(kSignedR11, 0, 1, 1023.0), 320L, "C.1.7 example: signed R11 60 x 8 - 160 = 320 (of 1023)");
            check.expect_eq(code11(kSignedR11, 4, 1, 1023.0), 470L, "C.1.7 example: multiplier 0: 480 - 10 = 470");
            // C.1.5 "reaches 1.0": base 255, multiplier 1, table 15: modifiers 2, 4, 6, 8 give 1.0; -3 gives 0.987.
            check.expect(got(kR11, 0, 8, 1)[0] == 1.0f && code11(kR11, 8, 0, 2047.0) == 2020,
                         "C.1.5 example: base 255 reaches exactly 1.0 with a positive modifier; modifier -3 gives 2020/2047 = 0.987",
                         std::format("{} / {}", got(kR11, 0, 8, 1)[0], code11(kR11, 8, 0, 2047.0)));
        }

        void check_negative_controls(demo::Check& check)
        {
            struct Control
            {
                Mut m;
                std::size_t tex;
                const char* what;
            };
            const std::array<Control, 11> controls = {{
                {Mut::WrongModifierTable, kRgb, "a decoder using the next row of Table C.6"},
                {Mut::NegatedDifferential, kRgb, "a decoder that subtracts dR/dG/dB"},
                {Mut::SwappedIndexBits, kRgb, "a decoder with MSB and LSB of the pixel index swapped"},
                {Mut::ReversedHOrder, kRgb, "a decoder with the H-mode ordering bit reversed"},
                {Mut::BlueUsesDG, kRgb, "a decoder taking the misprinted 'B + dG' literally"},
                {Mut::BlueUsesDG, kPunch, "a decoder taking 'B + dG' literally (punchthrough)"},
                {Mut::PlanarTruncate, kRgb, "a planar decoder without the +2 rounding term"},
                {Mut::EacZeroMultiplierIsZero, kR11, "an R11 decoder treating multiplier 0 as 0 instead of 1/8"},
                {Mut::SignedKeepsMinus128, kSignedR11, "a signed decoder keeping base -128 instead of -127"},
                {Mut::TransparentKeepsRgb, kPunch, "a punchthrough decoder that leaves RGB of transparent texels"},
                {Mut::WrongModifierTable, kArray, "a decoder using the next row of Table C.6 (array layers)"},
            }};
            for (const Control& c : controls)
            {
                const auto [bad, max_dev] = compare(c.tex, c.m);
                check.expect(bad > 0, std::format("negative control: {} disagrees with the {} readback ({} texels differ)", c.what, textures_[c.tex].name, bad));
            }
        }

        std::array<TexDesc, kTexCount> textures_{};
        std::vector<Cell> cells_;
        std::array<std::vector<std::vector<std::array<float, 4>>>, kTexCount> readback_{};
        std::vector<std::pair<std::string, mgl::ErrorCode>> upload_errors_;
        std::array<mgl::TextureId, kTexCount> tex_{};
        mgl::TextureId scratch_2d_{}, scratch_3d_{};
        mgl::ProgramId show_{}, probe_{};
        mgl::RenderbufferId rbo_{};
        mgl::FramebufferId fbo_{};
        bool version_31_ = false;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-etc2-eac",
        .title = "ETC2 / EAC compressed textures",
        .min_es = {3, 0},
        .summary = "Hand-encoded blocks of all ten ES 3.0 ETC2/EAC formats (individual, differential, T, H, planar, punchthrough, EAC "
                   "alpha, R11/RG11 unsigned and signed, sRGB) uploaded with glCompressedTex{Sub}Image{2D,3D}, compared texel by "
                   "texel with an Appendix C decoder; format queries and the compressed-upload errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Etc2Eac>(argc, argv, info);
}
