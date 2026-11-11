// es32-astc-and-copy-image
//
// Two image features that OpenGL ES 3.2 made core:
//
//  1. ASTC LDR compressed textures (ES 3.2 section 8.7, Table 8.17, Appendix C.1, and the "ASTC Compressed Texture Image Formats" chapter
//     of the Khronos Data Format Specification 1.1, which Appendix C.1 refers to). All 28 ASTC LDR formats must be listed by
//     COMPRESSED_TEXTURE_FORMATS. The demo hand-encodes 128-bit blocks: void-extent (constant colour) blocks and single-partition blocks with
//     direct LDR endpoints (luminance, luminance + alpha, RGB with and without blue contraction, RGBA), 2- and 3-bit weights on 4x4 and 3x3
//     weight grids (bilinear weight infill), plus illegal encodings that must decode to the error colour. A CPU decoder written from the
//     Data Format Specification (sections 18.9 - 18.24, only for the modes used here) predicts every texel; every texel of every texture is
//     read back with texelFetch in a compute shader and compared. Uploads and the CompressedTexSubImage2D block-alignment and imageSize rules
//     are exercised on the 4x4, 5x4, 6x6 and 12x12 footprints (including textures whose size is not a multiple of the block size).
//
//  2. glCopyImageSubData (ES 3.2 section 16.2.2): texture to texture, renderbuffer to texture and back, several array layers in one call,
//     2D array layers to cube map faces and a cube face back to a 2D texture, reinterpretation between compatible formats (RGBA8UI <-> R32F),
//     compressed <-> uncompressed copies of the same block size (ASTC and RGBA8_ETC2_EAC <-> RGBA32UI, RGB8_ETC2 <-> RGBA16UI), ASTC linear
//     -> ASTC sRGB, depth and depth-stencil textures, and every defined error.
//
// The picture (laid out from the surface size, at least 128 x 128): two panels side by side (one above the other on tall surfaces),
// each a texture magnified with texelFetch so that every texel is a flat rectangle:
//   left / top       the ASTC 4x4 "atlas", a tile set of 16 hand-encoded blocks (constant colours, RGB / RGBA gradients, blue contraction,
//                    luminance and luminance + alpha clouds);
//   right / bottom   the 32 x 32 ASTC "mosaic", a sunset over the sea assembled block by block with glCopyImageSubData from the atlas (compressed -> compressed)
//                    and from an RGBA32UI texture whose texels hold raw ASTC blocks (uncompressed -> compressed).
// Every pixel of the frame is compared with the CPU decoder's prediction.

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
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    using Block = std::array<std::uint8_t, 16>;  // one 128-bit ASTC block; byte 0 holds bits 0..7 (DFS 18.9)
    using U4 = std::array<std::uint32_t, 4>;
    using Rgb = std::array<int, 3>;
    using Rgba = std::array<int, 4>;

    constexpr int kMinSurface = 128;
    constexpr demo::Rgba8 kClear = {18, 20, 30, 255};

    // =============================================================================================
    // Bit access (DFS 18.9: "The bits in the block are labeled in little-endian order - the byte at the lowest
    // address contains bits 0..7. Bit 0 is the least significant bit in the byte.")
    // =============================================================================================
    void put_bits(Block& b, int pos, int count, std::uint32_t value)
    {
        for (int i = 0; i < count; ++i)
        {
            const int p = pos + i;
            const auto byte = static_cast<std::size_t>(p / 8);
            const auto mask = static_cast<std::uint8_t>(1u << (p % 8));
            if ((value >> i) & 1u)
                b[byte] = static_cast<std::uint8_t>(b[byte] | mask);
            else
                b[byte] = static_cast<std::uint8_t>(b[byte] & ~mask);
        }
    }

    std::uint32_t get_bits(const Block& b, int pos, int count)
    {
        std::uint32_t v = 0;
        for (int i = 0; i < count; ++i)
        {
            const int p = pos + i;
            v |= static_cast<std::uint32_t>((b[static_cast<std::size_t>(p / 8)] >> (p % 8)) & 1u) << i;
        }
        return v;
    }

    // =============================================================================================
    // Encoder for the block types used here.
    // =============================================================================================

    /// 2D void-extent block (DFS 18.23, Table 106): bits 0..8 = 1 1111 1100, bit 9 = dynamic range (0 = LDR, UNORM16 colour),
    /// bits 10..11 reserved = 1, bits 12..63 the extent coordinates (all ones = "no extent", a plain constant-colour block),
    /// bits 64..127 the colour R, G, B, A (16 bits each).
    Block void_extent(std::array<std::uint16_t, 4> c, bool hdr = false)
    {
        Block b{};
        put_bits(b, 0, 9, 0x1FCu);
        put_bits(b, 9, 1, hdr ? 1u : 0u);
        put_bits(b, 10, 2, 3u);
        for (int i = 0; i < 4; ++i) put_bits(b, 12 + 13 * i, 13, 0x1FFFu);
        for (int i = 0; i < 4; ++i) put_bits(b, 64 + 16 * i, 16, c[static_cast<std::size_t>(i)]);
        return b;
    }

    /// A constant 8-bit colour: UNORM16 = v * 257 (both bytes v), so every decode mode returns exactly v.
    Block solid(int r, int g, int bl, int a = 255)
    {
        const auto e = [](int v) { return static_cast<std::uint16_t>(v * 257); };
        return void_extent({e(r), e(g), e(bl), e(a)});
    }

    /// 2D block mode (DFS 18.10, Table 82) for a gw x gh weight grid with `bits` bits per weight (single plane).
    /// Weight range (Table 81): 2 bits = range 0..3 (P = 0, rho = 100), 3 bits = range 0..7 (P = 0, rho = 111).
    ///   bits[3:2] = 00:           W = B + 4 (bits 8..7), H = A + 2 (bits 6..5)
    ///   bits[3:2] = 11, bit 8 = 1: W = bit 7 + 2,       H = A + 2
    /// rho0 is bit 4, rho1 bit 0, rho2 bit 1; P is bit 9, the dual-plane bit 10.
    std::uint32_t block_mode(int gw, int gh, int bits)
    {
        const std::uint32_t rho = bits == 2 ? 0b100u : 0b111u;
        const std::uint32_t r0 = rho & 1u, r1 = (rho >> 1) & 1u, r2 = (rho >> 2) & 1u;
        const std::uint32_t range_bits = (r0 << 4) | (r2 << 1) | r1;
        if (gw >= 4 && gw <= 7 && gh >= 2 && gh <= 5)
            return (static_cast<std::uint32_t>(gw - 4) << 7) | (static_cast<std::uint32_t>(gh - 2) << 5) | range_bits;
        if (gw >= 2 && gw <= 3 && gh >= 2 && gh <= 5)
            return (1u << 8) | (static_cast<std::uint32_t>(gw - 2) << 7) | (static_cast<std::uint32_t>(gh - 2) << 5) | (3u << 2) | range_bits;
        throw demo::Failure(std::format("encoder: no block mode row for a {}x{} grid", gw, gh));
    }

    /// Single-partition block (DFS Table 79): block mode bits 0..10, partition count - 1 = 0 in bits 11..12, CEM in bits 13..16,
    /// endpoint values from bit 17 upwards (range 0..255: 8 bits each, DFS 18.12 / 18.22), weights from bit 127 downwards
    /// ("Bit n in the stream is thus bit 127-n in the block", DFS 18.16), raster order, value i occupies stream bits i*n .. i*n+n-1.
    Block encode_block(int gw, int gh, int bits, int cem, const std::vector<int>& values, const std::vector<int>& weights)
    {
        Block b{};
        put_bits(b, 0, 11, block_mode(gw, gh, bits));
        put_bits(b, 11, 2, 0u);
        put_bits(b, 13, 4, static_cast<std::uint32_t>(cem));
        for (std::size_t i = 0; i < values.size(); ++i) put_bits(b, 17 + 8 * static_cast<int>(i), 8, static_cast<std::uint32_t>(values[i]));
        for (std::size_t i = 0; i < weights.size(); ++i)
            for (int k = 0; k < bits; ++k)
                put_bits(b, 127 - (static_cast<int>(i) * bits + k), 1, (static_cast<std::uint32_t>(weights[i]) >> k) & 1u);
        return b;
    }

    std::vector<int> mirrored(std::vector<int> w, int bits)
    {
        const int max = (1 << bits) - 1;
        for (int& v : w) v = max - v;
        return w;
    }

    /// CEM 8, LDR RGB direct. The decoder swaps to blue contraction when v1+v3+v5 < v0+v2+v4, so for a plain RGB ramp the
    /// endpoints are put in the order that keeps s1 >= s0 (mirroring the weights if they had to be exchanged).
    Block rgb_block(int gw, int gh, int bits, Rgb e0, Rgb e1, std::vector<int> w)
    {
        if (e1[0] + e1[1] + e1[2] < e0[0] + e0[1] + e0[2])
        {
            std::swap(e0, e1);
            w = mirrored(std::move(w), bits);
        }
        return encode_block(gw, gh, bits, 8, {e0[0], e1[0], e0[1], e1[1], e0[2], e1[2]}, w);
    }

    /// CEM 8 with the raw values given directly (used for the blue-contraction case, where s1 < s0 on purpose).
    Block rgb_raw_block(int gw, int gh, int bits, std::array<int, 6> v, const std::vector<int>& w)
    {
        return encode_block(gw, gh, bits, 8, {v.begin(), v.end()}, w);
    }

    Block lum_block(int gw, int gh, int bits, int l0, int l1, const std::vector<int>& w) { return encode_block(gw, gh, bits, 0, {l0, l1}, w); }

    Block lum_alpha_block(int gw, int gh, int bits, int l0, int l1, int a0, int a1, const std::vector<int>& w)
    {
        return encode_block(gw, gh, bits, 4, {l0, l1, a0, a1}, w);
    }

    Block rgba_block(int gw, int gh, int bits, Rgba e0, Rgba e1, std::vector<int> w)
    {
        if (e1[0] + e1[1] + e1[2] < e0[0] + e0[1] + e0[2])
        {
            std::swap(e0, e1);
            w = mirrored(std::move(w), bits);
        }
        return encode_block(gw, gh, bits, 12, {e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]}, w);
    }

    // Weight grid patterns (values 0 .. 2^bits - 1, raster order, row 0 first).
    std::vector<int> pattern(int gw, int gh, int bits, const std::function<double(double, double)>& f)
    {
        const int max = (1 << bits) - 1;
        std::vector<int> w;
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x)
            {
                const double u = gw > 1 ? static_cast<double>(x) / (gw - 1) : 0.0;
                const double v = gh > 1 ? static_cast<double>(y) / (gh - 1) : 0.0;
                w.push_back(std::clamp(static_cast<int>(std::lround(f(u, v) * max)), 0, max));
            }
        return w;
    }
    std::vector<int> ramp_x(int gw, int gh, int bits) { return pattern(gw, gh, bits, [](double u, double) { return u; }); }
    std::vector<int> ramp_y(int gw, int gh, int bits) { return pattern(gw, gh, bits, [](double, double v) { return v; }); }
    std::vector<int> ramp_diag(int gw, int gh, int bits) { return pattern(gw, gh, bits, [](double u, double v) { return (u + v) / 2; }); }
    std::vector<int> bump(int gw, int gh, int bits)
    {
        return pattern(gw, gh, bits, [](double u, double v) { return 1.0 - std::max(std::abs(2 * u - 1), std::abs(2 * v - 1)); });
    }
    std::vector<int> checker(int gw, int gh, int bits)
    {
        std::vector<int> w;
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) w.push_back(((x + y) & 1) != 0 ? (1 << bits) - 1 : 0);
        return w;
    }

    /// Error-colour blocks (DFS 18.24 "Illegal Encodings", 18.23, 18.25).
    Block hdr_void_extent() { return void_extent({0x3C00, 0x3800, 0x3400, 0x3C00}, true); }  // "a void-extent block with HDR values ... decoded in LDR mode"
    Block reserved_block() { return Block{}; }  // block mode bits [3:0] = 0000: "Reserved" row of Table 82
    Block oversized_grid_block()                // a 5 x 4 weight grid in a 4 x 4 footprint: "The size of the weight grid exceeds the size of the block footprint"
    {
        return rgb_block(5, 4, 2, {0, 0, 0}, {192, 192, 192}, ramp_x(5, 4, 2));
    }

    // =============================================================================================
    // CPU decoder, written from the Data Format Specification 1.1 chapter 18 for exactly the subset used here
    // (single partition, single plane, bit-only weight ranges, endpoint range 0..255, CEM 0 / 4 / 8 / 12). Everything
    // outside the subset either is an illegal encoding (-> error colour, as the specification requires) or throws, because
    // a block the demo did not intend to produce is a bug in the demo, not something to compare against.
    // =============================================================================================
    struct Decoded
    {
        bool error = false;
        std::array<std::uint32_t, 4> c{};  // UNORM16 interpolation result per channel (DFS 18.19), before the decode-mode conversion
    };

    std::uint32_t replicate(std::uint32_t v, int from, int to)
    {
        std::uint32_t r = 0;
        int n = 0;
        while (n < to)
        {
            r = (r << from) | v;
            n += from;
        }
        return r >> (n - to);
    }

    Decoded decode_texel(const Block& b, int bw, int bh, int x, int y, bool srgb)
    {
        Decoded out;
        const Decoded error{true, {}};
        // ---- void extent (18.23) ----
        if (get_bits(b, 0, 9) == 0x1FCu)
        {
            if (get_bits(b, 10, 2) != 3u) return error;  // "A 2D void-extent block that has any of the reserved bits not set to 1"
            if (get_bits(b, 9, 1) != 0u) return error;   // HDR void extent in LDR mode -> error colour
            const std::uint32_t s0 = get_bits(b, 12, 13), s1 = get_bits(b, 25, 13), t0 = get_bits(b, 38, 13), t1 = get_bits(b, 51, 13);
            const bool all_ones = s0 == 0x1FFFu && s1 == 0x1FFFu && t0 == 0x1FFFu && t1 == 0x1FFFu;
            if (!all_ones && (s0 >= s1 || t0 >= t1)) return error;
            for (int i = 0; i < 4; ++i) out.c[static_cast<std::size_t>(i)] = get_bits(b, 64 + 16 * i, 16);
            return out;
        }
        // ---- block mode (18.10, Table 82) ----
        const std::uint32_t mode = get_bits(b, 0, 11);
        if ((mode & 3u) == 0u)
        {
            if ((mode & 0xFu) == 0u) return error;  // Reserved row
            throw demo::Failure("decoder: block-mode rows with bits[1:0] = 00 are not used by this demo");
        }
        const std::uint32_t rho = ((mode >> 1) & 1u) << 2 | (mode & 1u) << 1 | ((mode >> 4) & 1u);
        const std::uint32_t p = (mode >> 9) & 1u;
        if (((mode >> 10) & 1u) != 0u) throw demo::Failure("decoder: dual-plane blocks are not used by this demo");
        const int a = static_cast<int>((mode >> 5) & 3u), bb = static_cast<int>((mode >> 7) & 3u);
        int gw = 0, gh = 0;
        switch ((mode >> 2) & 3u)
        {
        case 0: gw = bb + 4; gh = a + 2; break;
        case 1: gw = bb + 8; gh = a + 2; break;
        case 2: gw = a + 2; gh = bb + 8; break;
        default:
            if ((mode & 0x100u) != 0u) { gw = (bb & 1) + 2; gh = a + 2; }
            else { gw = a + 2; gh = (bb & 1) + 6; }
            break;
        }
        int wbits = 0;  // bit-only weight ranges of Table 81
        if (p == 0 && rho == 2) wbits = 1;
        else if (p == 0 && rho == 4) wbits = 2;
        else if (p == 0 && rho == 7) wbits = 3;
        else if (p == 1 && rho == 4) wbits = 4;
        else if (p == 1 && rho == 7) wbits = 5;
        else throw demo::Failure("decoder: trit / quint weight ranges are not used by this demo");
        // ---- illegal encodings (18.24) ----
        if (gw > bw || gh > bh) return error;
        const int nweights = gw * gh;
        const int weight_bits = nweights * wbits;
        if (nweights > 64 || weight_bits > 96 || weight_bits < 24) return error;
        if (get_bits(b, 11, 2) != 0u) throw demo::Failure("decoder: multi-partition blocks are not used by this demo");
        const auto cem = static_cast<int>(get_bits(b, 13, 4));
        if (cem == 2 || cem == 3 || cem == 7 || cem == 11 || cem == 14 || cem == 15) return error;  // HDR endpoint modes: error in LDR
        const int nvalues = ((cem >> 2) + 1) * 2;
        // ---- data size determination (18.22): the largest endpoint range that fits ----
        const int remaining = 128 - 17 - weight_bits;
        if (remaining < (13 * nvalues + 4) / 5) return error;
        if (nvalues * 8 > remaining) throw demo::Failure("decoder: endpoint ranges below 0..255 are not used by this demo");
        std::array<int, 8> v{};
        for (int i = 0; i < nvalues; ++i) v[static_cast<std::size_t>(i)] = static_cast<int>(get_bits(b, 17 + 8 * i, 8));  // 0..255: unquantized as-is
        // ---- LDR endpoint decoding (18.14) ----
        Rgba e0{}, e1{};
        const auto blue_contract = [](int r, int g, int bl, int al) { return Rgba{(r + bl) >> 1, (g + bl) >> 1, bl, al}; };
        switch (cem)
        {
        case 0: e0 = {v[0], v[0], v[0], 0xFF}; e1 = {v[1], v[1], v[1], 0xFF}; break;
        case 4: e0 = {v[0], v[0], v[0], v[2]}; e1 = {v[1], v[1], v[1], v[3]}; break;
        case 8:
        case 12:
        {
            const int a0 = cem == 12 ? v[6] : 0xFF, a1 = cem == 12 ? v[7] : 0xFF;
            if (v[1] + v[3] + v[5] >= v[0] + v[2] + v[4])
            {
                e0 = {v[0], v[2], v[4], a0};
                e1 = {v[1], v[3], v[5], a1};
            }
            else
            {
                e0 = blue_contract(v[1], v[3], v[5], a1);
                e1 = blue_contract(v[0], v[2], v[4], a0);
            }
            break;
        }
        default: throw demo::Failure(std::format("decoder: CEM {} is not used by this demo", cem));
        }
        // ---- weights: decode (18.16), unquantize (18.17), infill (18.18) ----
        const auto weight = [&](int n) -> int {
            if (n >= nweights) return 0;  // only reached with a zero bilinear factor (fs or ft = 0 at the last grid column / row)
            std::uint32_t q = 0;
            for (int k = 0; k < wbits; ++k) q |= get_bits(b, 127 - (n * wbits + k), 1) << k;
            std::uint32_t unq = replicate(q, wbits, 6);
            if (unq > 32) unq += 1;
            return static_cast<int>(unq);
        };
        const int ds = (1024 + bw / 2) / (bw - 1), dt = (1024 + bh / 2) / (bh - 1);
        const int gs = (ds * x * (gw - 1) + 32) >> 6, gt = (dt * y * (gh - 1) + 32) >> 6;
        const int js = gs >> 4, fs = gs & 0x0F, jt = gt >> 4, ft = gt & 0x0F;
        const int v0 = js + jt * gw;
        const int p00 = weight(v0), p01 = weight(v0 + 1), p10 = weight(v0 + gw), p11 = weight(v0 + gw + 1);
        const int w11 = (fs * ft + 8) >> 4, w10 = ft - w11, w01 = fs - w11, w00 = 16 - fs - ft + w11;
        const int i = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4;
        // ---- weight application (18.19) ----
        for (std::size_t ch = 0; ch < 4; ++ch)
        {
            const auto c0 = static_cast<std::uint32_t>(e0[ch]), c1 = static_cast<std::uint32_t>(e1[ch]);
            const bool srgb_channel = srgb && ch < 3;
            const std::uint32_t x0 = srgb_channel ? (c0 << 8) | 0x80u : (c0 << 8) | c0;
            const std::uint32_t x1 = srgb_channel ? (c1 << 8) | 0x80u : (c1 << 8) | c1;
            out.c[ch] = (x0 * static_cast<std::uint32_t>(64 - i) + x1 * static_cast<std::uint32_t>(i) + 32u) / 64u;
        }
        return out;
    }

    /// decode_float16 of DFS 18.19: "if C = 65535 the final result is 1.0 (0x3C00); otherwise C is divided by 65536 and the
    /// infinite-precision result of the division is converted to FP16 with round-to-zero semantics."
    double f16_rtz(std::uint32_t c)
    {
        if (c == 65535u) return 1.0;
        const double x = static_cast<double>(c) / 65536.0;
        if (x == 0.0) return 0.0;
        const int e = static_cast<int>(std::floor(std::log2(x)));
        if (e < -14) return std::floor(x / std::ldexp(1.0, -24)) * std::ldexp(1.0, -24);
        return std::floor(std::ldexp(x, -e) * 1024.0) / 1024.0 * std::ldexp(1.0, e);
    }

    /// An ASTC image held on the CPU: blocks in raster order (DFS 18.9).
    struct AstcImage
    {
        int w = 0, h = 0, bw = 4, bh = 4;
        std::vector<Block> blocks;
        [[nodiscard]] int bx() const { return (w + bw - 1) / bw; }
        [[nodiscard]] int by() const { return (h + bh - 1) / bh; }
        [[nodiscard]] int image_size() const { return bx() * by() * 16; }  // the imageSize the GL expects
        [[nodiscard]] std::vector<std::uint8_t> bytes() const
        {
            std::vector<std::uint8_t> out;
            for (const Block& b : blocks) out.insert(out.end(), b.begin(), b.end());
            return out;
        }
        [[nodiscard]] Decoded decode(int x, int y, bool srgb) const
        {
            return decode_texel(blocks[static_cast<std::size_t>((y / bh) * bx() + x / bw)], bw, bh, x % bw, y % bh, srgb);
        }
    };

    std::vector<std::uint8_t> block_bytes(std::initializer_list<Block> blocks)
    {
        std::vector<std::uint8_t> out;
        for (const Block& b : blocks) out.insert(out.end(), b.begin(), b.end());
        return out;
    }

    U4 block_words(const Block& b)
    {
        U4 w{};
        std::memcpy(w.data(), b.data(), 16);  // the bytes of RGBA32UI client data (RGBA_INTEGER / UNSIGNED_INT) in host order
        return w;
    }

    /// 8-bit result of a texel that every decode mode agrees on: for UNORM8 decoding the top 8 bits of C (DFS 18.19), and the
    /// demo only uses data for which decode_float16 * 255 lies within 0.375 of the same value (checked in init()).
    std::array<std::uint8_t, 4> top8(const Decoded& d)
    {
        return {static_cast<std::uint8_t>(d.c[0] >> 8), static_cast<std::uint8_t>(d.c[1] >> 8), static_cast<std::uint8_t>(d.c[2] >> 8),
                static_cast<std::uint8_t>(d.c[3] >> 8)};
    }

    double srgb_to_linear(double cs) { return cs <= 0.04045 ? cs / 12.92 : std::pow((cs + 0.055) / 1.055, 2.4); }  // ES 3.2 equation 8.14
    double linear_to_srgb(double cl) { return cl <= 0.0031308 ? cl * 12.92 : 1.055 * std::pow(cl, 1.0 / 2.4) - 0.055; }

    std::array<float, 4> as_float(const U4& u)
    {
        return {std::bit_cast<float>(u[0]), std::bit_cast<float>(u[1]), std::bit_cast<float>(u[2]), std::bit_cast<float>(u[3])};
    }

    /// Collects mismatches; `first` describes the first one.
    struct Tally
    {
        int bad = 0;
        int total = 0;
        std::string first;
        template <typename F>
        void add(bool ok, F&& describe)
        {
            ++total;
            if (!ok && bad++ == 0) first = describe();
        }
        [[nodiscard]] std::string detail() const { return bad == 0 ? std::format("{} values", total) : std::format("{} of {} wrong; first: {}", bad, total, first); }
    };

    std::string with_defines(std::string source, std::initializer_list<std::pair<std::string_view, std::string_view>> defines)
    {
        std::string inserted;
        for (const auto& [name, value] : defines) inserted += std::format("#define {} {}\n", name, value);
        source.insert(source.find('\n') + 1, inserted);  // right after "#version 320 es"
        return source;
    }

    mgl::ProgramId build_compute(const std::string& source, std::string_view label)
    {
        std::string log;
        const mgl::ShaderId s = demo::compile_shader(mgl::ShaderType::Compute, source, log);
        if (s.value == 0) throw demo::Failure(std::format("{}: compile failed:\n{}", label, log));
        const std::array<mgl::ShaderId, 1> shaders = {s};
        const mgl::ProgramId program = demo::link_program(shaders, log);
        mgl::glDeleteShader(s);
        if (program.value == 0) throw demo::Failure(std::format("{}: link failed:\n{}", label, log));
        return program;
    }

    // The 28 ASTC LDR formats of Table 8.17 (linear and sRGB per footprint).
    struct AstcFormat
    {
        mgl::CompressedInternalFormat linear, srgb;
        int bw, bh;
    };
    constexpr std::array<AstcFormat, 14> kAstcFormats = {{
        {mgl::CompressedInternalFormat::RgbaAstc4x4, mgl::CompressedInternalFormat::Srgb8Alpha8Astc4x4, 4, 4},
        {mgl::CompressedInternalFormat::RgbaAstc5x4, mgl::CompressedInternalFormat::Srgb8Alpha8Astc5x4, 5, 4},
        {mgl::CompressedInternalFormat::RgbaAstc5x5, mgl::CompressedInternalFormat::Srgb8Alpha8Astc5x5, 5, 5},
        {mgl::CompressedInternalFormat::RgbaAstc6x5, mgl::CompressedInternalFormat::Srgb8Alpha8Astc6x5, 6, 5},
        {mgl::CompressedInternalFormat::RgbaAstc6x6, mgl::CompressedInternalFormat::Srgb8Alpha8Astc6x6, 6, 6},
        {mgl::CompressedInternalFormat::RgbaAstc8x5, mgl::CompressedInternalFormat::Srgb8Alpha8Astc8x5, 8, 5},
        {mgl::CompressedInternalFormat::RgbaAstc8x6, mgl::CompressedInternalFormat::Srgb8Alpha8Astc8x6, 8, 6},
        {mgl::CompressedInternalFormat::RgbaAstc8x8, mgl::CompressedInternalFormat::Srgb8Alpha8Astc8x8, 8, 8},
        {mgl::CompressedInternalFormat::RgbaAstc10x5, mgl::CompressedInternalFormat::Srgb8Alpha8Astc10x5, 10, 5},
        {mgl::CompressedInternalFormat::RgbaAstc10x6, mgl::CompressedInternalFormat::Srgb8Alpha8Astc10x6, 10, 6},
        {mgl::CompressedInternalFormat::RgbaAstc10x8, mgl::CompressedInternalFormat::Srgb8Alpha8Astc10x8, 10, 8},
        {mgl::CompressedInternalFormat::RgbaAstc10x10, mgl::CompressedInternalFormat::Srgb8Alpha8Astc10x10, 10, 10},
        {mgl::CompressedInternalFormat::RgbaAstc12x10, mgl::CompressedInternalFormat::Srgb8Alpha8Astc12x10, 12, 10},
        {mgl::CompressedInternalFormat::RgbaAstc12x12, mgl::CompressedInternalFormat::Srgb8Alpha8Astc12x12, 12, 12},
    }};

    /// The picture's two panels, derived from the surface size (shared by draw_picture() and check_picture()).
    struct Rect
    {
        int x = 0, y = 0, w = 0, h = 0;
        [[nodiscard]] bool contains(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; }
    };
    struct Layout
    {
        Rect atlas, mosaic;
        Layout(int width, int height)
        {
            const int m = std::max(2, std::min(width, height) / 32);  // margin around and between the panels (clear colour)
            if (width >= height)
            {
                atlas = {m, m, (width - 3 * m) / 2, height - 2 * m};
                mosaic = {2 * m + atlas.w, m, 0, height - 2 * m};
                mosaic.w = width - m - mosaic.x;
            }
            else
            {
                mosaic = {m, m, width - 2 * m, (height - 3 * m) / 2};
                atlas = {m, 2 * m + mosaic.h, width - 2 * m, 0};
                atlas.h = height - m - atlas.y;
            }
        }
    };

    // The RGBA8 patterns of the uncompressed copy tests (every channel varies, alpha included: the copy is a memcpy).
    demo::Rgba8 rgba_pattern(int seed, int x, int y)
    {
        return {static_cast<std::uint8_t>((seed * 53 + x * 29 + y * 7 + 11) & 255), static_cast<std::uint8_t>((seed * 19 + x * 3 + y * 41 + 5) & 255),
                static_cast<std::uint8_t>((seed * 71 + x * 13 + y * 17 + 101) & 255), static_cast<std::uint8_t>((seed * 31 + x * 5 + y * 11 + 60) & 255)};
    }
    std::vector<std::uint8_t> rgba_image(int seed, int w, int h)
    {
        std::vector<std::uint8_t> out;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const demo::Rgba8 c = rgba_pattern(seed, x, y);
                out.insert(out.end(), c.begin(), c.end());
            }
        return out;
    }

    std::uint32_t lcg(std::uint32_t& s)
    {
        s = s * 1664525u + 1013904223u;
        return s >> 8;
    }

    class AstcAndCopyImage final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override;
        void render(demo::Env& env, int frame) override;
        void validate(demo::Env& env, demo::Check& check) override;
        void shutdown(demo::Env& env) override;

    private:
        struct Fetched
        {
            std::array<int, 2> size{};
            std::vector<U4> texels;
        };

        mgl::TextureId new_texture();
        mgl::RenderbufferId new_renderbuffer();
        void note(std::string what);  // records the GL error state after a setup step that must succeed
        void build_astc_textures();
        void build_mosaic();
        void build_copy_textures();
        void draw_picture(int width, int height);
        Fetched fetch(bool uint_kind, mgl::TextureId tex, int w, int h);
        std::vector<std::uint8_t> read_texture_rgba8(mgl::TextureTarget target, mgl::TextureId tex, int layer, int w, int h);
        std::vector<std::uint8_t> read_renderbuffer_rgba8(mgl::RenderbufferId rb, int w, int h);

        // validation parts
        void check_picture(demo::Check& check, int width, int height);
        void check_formats_query(demo::Check& check);
        void check_decoder_design(demo::Check& check);
        void check_decode(demo::Check& check, std::string_view label, mgl::TextureId tex, const AstcImage& img, bool srgb);
        void check_astc_decoding(demo::Check& check);
        void check_compressed_uploads(demo::Check& check);
        void check_compressed_copies(demo::Check& check);
        void check_uncompressed_copies(demo::Check& check);
        void check_layers_and_faces(demo::Check& check);
        void check_reinterpretation(demo::Check& check);
        void check_depth_stencil(demo::Check& check);
        void check_completeness(demo::Check& check);
        void check_copy_errors(demo::Check& check);

        mgl::ProgramId fetch_f_{}, fetch_u_{}, panel_{};
        mgl::BufferId ssbo_{};
        mgl::FramebufferId fbo_{};
        std::vector<mgl::TextureId> textures_;
        std::vector<mgl::RenderbufferId> renderbuffers_;
        std::vector<std::pair<std::string, mgl::ErrorCode>> setup_errors_;

        // ASTC
        AstcImage atlas_, mosaic_img_, astc6_, astc5x4_, astc12_, errors_img_;
        std::vector<Block> palette_;           // raw ASTC blocks held in the RGBA32UI palette texture
        std::vector<char> mosaic_src_;         // per mosaic block: 'a'..'p' = atlas block, '0'..'7' = palette texel
        mgl::TextureId atlas_tex_{}, atlas_srgb_tex_{}, srgb_copy_tex_{}, mosaic_tex_{}, palette_tex_{}, atlas_u32_{}, mosaic_u32_{};
        mgl::TextureId astc6_tex_{}, astc5x4_tex_{}, astc12_tex_{}, errors_lin_tex_{}, errors_srgb_tex_{}, astc12_u32_{};
        std::vector<std::pair<std::string, mgl::ErrorCode>> upload_errors_;  // expected-error uploads done in init (checked in validate)
        std::vector<std::uint8_t> etc2_rgba_, etc2_rgb_;
        mgl::TextureId etc2_rgba_tex_{}, etc2_rgb_tex_{}, etc2_rgba_u32_{}, etc2_rgb_u16_{}, etc2_rgb_back_{}, etc2_rgb_u16_back_{};

        // uncompressed copies
        mgl::TextureId rgba8_src_{}, rgba8_dst_{}, rgba8_dst2_{};
        mgl::RenderbufferId rb_{}, ms_rb_{};
        std::vector<std::uint8_t> rb_before_;  // the renderbuffer as read back before the copies
        mgl::TextureId arr_src_{}, arr_dst_{}, arr6_{}, cube_{}, face_out_{};
        mgl::TextureId u8_src_{}, f32_dst_{}, f32_src_{}, u8_dst_{};
        std::vector<std::uint8_t> u8_data_;
        std::vector<float> f32_data_;
        mgl::TextureId d32_src_{}, d32_dst_{}, ds_src_{}, ds_dst_{};
        std::vector<float> d32_src_data_, d32_dst_data_;
        std::vector<std::uint32_t> ds_src_data_, ds_dst_data_;
        mgl::TextureId incomplete_{}, incomplete_dst_{}, rg8_{}, rgba16f_{}, rgba16ui_{};
    };

    mgl::TextureId AstcAndCopyImage::new_texture()
    {
        mgl::TextureId t{};
        mgl::glGenTextures(1, &t);
        textures_.push_back(t);
        return t;
    }

    mgl::RenderbufferId AstcAndCopyImage::new_renderbuffer()
    {
        mgl::RenderbufferId r{};
        mgl::glGenRenderbuffers(1, &r);
        renderbuffers_.push_back(r);
        return r;
    }

    void AstcAndCopyImage::note(std::string what) { setup_errors_.emplace_back(std::move(what), demo::drain_gl_errors()); }

    void nearest(mgl::TextureTarget target)
    {
        mgl::glTexParameter(target, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);
    }

    // =================================================================================================
    // Setup
    // =================================================================================================
    void AstcAndCopyImage::init(demo::Env& env)
    {
        if (env.width < kMinSurface || env.height < kMinSurface)
            throw demo::Skip(std::format("the picture needs a surface of at least {0}x{0} pixels to show every texel (got {1}x{2})", kMinSurface, env.width, env.height));

        const std::string cs = env.load_text("shaders/fetch.comp");
        fetch_f_ = build_compute(with_defines(cs, {{"SAMPLER", "sampler2D"}, {"KIND_FLOAT", "1"}}), "fetch.comp (float)");
        fetch_u_ = build_compute(with_defines(cs, {{"SAMPLER", "usampler2D"}, {"KIND_UINT", "1"}}), "fetch.comp (uint)");
        panel_ = demo::build_program(env, "shaders/panel.vert", "shaders/panel.frag");
        mgl::glGenBuffers(1, &ssbo_);
        mgl::glGenFramebuffers(1, &fbo_);

        build_astc_textures();
        build_mosaic();
        build_copy_textures();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
    }

    void AstcAndCopyImage::build_astc_textures()
    {
        // ---- the atlas: the tile set of 16 blocks of the 4x4 footprint (16 x 16 texels), block i at block column i % 4, block row i / 4 (row 0 at the bottom) ----
        // Endpoint pairs differ by multiples of 64 in every channel, so every interpolated UNORM16 value is exactly v * 257 (v integer):
        // UNORM8 decoding (top 8 bits) and decode_float16 then agree on v (verified for every texel in check_decoder_design()).
        atlas_ = {16, 16, 4, 4, {}};
        // Sky colours, top to bottom, and the water / sand colours of the scene (every step a multiple of 64 per channel).
        const Rgb t0 = {24, 8, 72}, t1 = {88, 8, 136}, t2 = {152, 72, 136}, t3 = {216, 136, 72}, t4 = {216, 200, 8};
        const Rgb deep = {0, 8, 72}, sea = {0, 72, 136}, gold = {192, 200, 8}, sand = {216, 184, 120}, sand_dark = {152, 120, 56};
        atlas_.blocks = {
            rgb_block(4, 4, 2, sand, sand_dark, ramp_x(4, 4, 2)),                       // a  sand ramp (CEM 8; endpoints exchanged by the encoder)
            solid(216, 184, 120),                                                        // b  sand (void extent)
            solid(0, 8, 72),                                                             // c  deep sea (void extent)
            solid(255, 255, 255),                                                        // d  white (void extent)
            rgb_block(4, 4, 2, deep, sea, ramp_y(4, 4, 2)),                              // e  deep water, dark at the bottom
            rgba_block(4, 4, 2, {0, 72, 136, 255}, {64, 136, 200, 63}, ramp_x(4, 4, 2)),  // f  wave (CEM 12, alpha 255 -> 63)
            rgb_block(4, 4, 2, sea, gold, checker(4, 4, 2)),                             // g  glitter of the sun's reflection
            rgb_raw_block(4, 4, 2, {64, 0, 128, 64, 192, 128}, ramp_y(4, 4, 2)),         // h  misty water: blue contraction (s1 = 192 < s0 = 384),
                                                                                         //    e0 = (64, 96, 128), e1 = (128, 160, 192)
            rgb_block(4, 4, 2, sea, gold, ramp_y(4, 4, 2)),                              // i  reflection below the sun
            rgb_block(4, 4, 2, t4, t3, ramp_y(4, 4, 2)),                                 // j  sky at the horizon
            solid(255, 240, 160),                                                        // k  the sun (void extent)
            lum_block(3, 3, 3, 191, 255, bump(3, 3, 3)),                                 // l  cloud (CEM 0, 3 x 3 weights infilled to 4 x 4)
            rgb_block(4, 4, 2, t3, t2, ramp_y(4, 4, 2)),                                 // m  sky
            rgb_block(4, 4, 2, t2, t1, ramp_y(4, 4, 2)),                                 // n  sky
            rgb_block(4, 4, 2, t1, t0, ramp_y(4, 4, 2)),                                 // o  sky at the top
            lum_alpha_block(4, 4, 2, 191, 255, 255, 191, ramp_x(4, 4, 2)),               // p  cloud (CEM 4, luminance + alpha)
        };
        // ---- the palette: raw blocks stored as RGBA32UI texels (one texel = one 128-bit block, ES 3.2 Table 16.3) ----
        palette_ = {
            rgba_block(4, 4, 2, {0, 72, 136, 255}, {64, 136, 200, 63}, mirrored(ramp_x(4, 4, 2), 2)),  // 0  wave, the other way round
            lum_block(3, 3, 3, 63, 255, bump(3, 3, 3)),                                                // 1  foam
            solid(255, 200, 64),                                                                       // 2  glow around the sun
            rgb_block(4, 4, 2, sand, sand_dark, checker(4, 4, 2)),                                     // 3  pebbles
            rgb_block(3, 3, 3, t1, {216, 136, 200}, bump(3, 3, 3)),                                    // 4  star
            lum_block(3, 3, 3, 191, 255, ramp_x(3, 3, 3)),                                             // 5  cloud edge
            solid(0, 24, 96),                                                                          // 6  dark water
            solid(8, 8, 16),                                                                           // 7  rock
        };

        const std::vector<std::uint8_t> atlas_bytes = atlas_.bytes();
        atlas_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, atlas_tex_);
        mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::RgbaAstc4x4, 16, 16, 0, atlas_.image_size(), atlas_bytes.data());
        nearest(mgl::TextureTarget::Texture2D);
        note("glCompressedTexImage2D(RGBA_ASTC_4x4, 16 x 16, imageSize 256)");

        atlas_srgb_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, atlas_srgb_tex_);
        mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::Srgb8Alpha8Astc4x4, 16, 16, 0, atlas_.image_size(), atlas_bytes.data());
        nearest(mgl::TextureTarget::Texture2D);
        note("glCompressedTexImage2D(SRGB8_ALPHA8_ASTC_4x4, 16 x 16)");

        // ---- 6x6 footprint, 17 x 10 texels: 3 x 2 blocks, the last column and row only partly inside the texture ----
        astc6_ = {17, 10, 6, 6, {}};
        astc6_.blocks = {
            lum_block(3, 3, 3, 0, 192, bump(3, 3, 3)),  rgb_block(4, 4, 2, {0, 64, 128}, {192, 128, 64}, ramp_x(4, 4, 2)), solid(200, 40, 120),
            solid(30, 90, 200), rgb_block(3, 3, 3, {255, 255, 0}, {63, 63, 192}, ramp_diag(3, 3, 3)), lum_block(4, 4, 2, 63, 255, checker(4, 4, 2)),
        };
        astc6_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, astc6_tex_);
        {
            const std::vector<std::uint8_t> bytes = astc6_.bytes();
            // A wrong imageSize first (one block short): INVALID_VALUE, and nothing is specified.
            mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::RgbaAstc6x6, 17, 10, 0, astc6_.image_size() - 16, bytes.data());
            upload_errors_.emplace_back("glCompressedTexImage2D(RGBA_ASTC_6x6, 17 x 10, imageSize 80 instead of ceil(17/6) * ceil(10/6) * 16 = 96)", demo::drain_gl_errors());
            mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::RgbaAstc6x6, 17, 10, 0, astc6_.image_size(), bytes.data());
        }
        nearest(mgl::TextureTarget::Texture2D);
        note("glCompressedTexImage2D(RGBA_ASTC_6x6, 17 x 10, imageSize 96)");

        // ---- 5x4 footprint, 15 x 8 texels (3 x 2 blocks): immutable storage, filled and then edited with glCompressedTexSubImage2D ----
        astc5x4_ = {15, 8, 5, 4, {}};
        astc5x4_.blocks = {solid(40, 40, 40), solid(80, 80, 80), solid(120, 120, 120), solid(160, 160, 160), solid(200, 200, 200), solid(240, 240, 240)};
        astc5x4_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, astc5x4_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::CompressedInternalFormat::RgbaAstc5x4, 15, 8);
        {
            const std::vector<std::uint8_t> bytes = astc5x4_.bytes();
            mgl::glCompressedTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 15, 8, mgl::CompressedInternalFormat::RgbaAstc5x4, astc5x4_.image_size(), bytes.data());
        }
        nearest(mgl::TextureTarget::Texture2D);
        note("glTexStorage2D(RGBA_ASTC_5x4, 15 x 8) + glCompressedTexSubImage2D(whole level, imageSize 96)");

        // ---- 12x12 footprint, 30 x 30 texels (3 x 3 blocks, 6-texel partial blocks at the right and top) ----
        astc12_ = {30, 30, 12, 12, {}};
        for (int i = 0; i < 9; ++i) astc12_.blocks.push_back(solid(20 + 25 * i, 200 - 20 * i, 60 + 15 * i));
        astc12_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, astc12_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::CompressedInternalFormat::RgbaAstc12x12, 30, 30);
        {
            const std::vector<std::uint8_t> bytes = astc12_.bytes();
            mgl::glCompressedTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 30, 30, mgl::CompressedInternalFormat::RgbaAstc12x12, astc12_.image_size(), bytes.data());
        }
        nearest(mgl::TextureTarget::Texture2D);
        note("glTexStorage2D(RGBA_ASTC_12x12, 30 x 30) + glCompressedTexSubImage2D(whole level, imageSize 144)");

        // ---- illegal encodings, 12 x 4 texels (3 blocks of 4x4), in a linear and an sRGB texture ----
        errors_img_ = {12, 4, 4, 4, {hdr_void_extent(), reserved_block(), oversized_grid_block()}};
        const std::vector<std::uint8_t> err_bytes = errors_img_.bytes();
        errors_lin_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, errors_lin_tex_);
        mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::RgbaAstc4x4, 12, 4, 0, errors_img_.image_size(), err_bytes.data());
        nearest(mgl::TextureTarget::Texture2D);
        errors_srgb_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, errors_srgb_tex_);
        mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::Srgb8Alpha8Astc4x4, 12, 4, 0, errors_img_.image_size(), err_bytes.data());
        nearest(mgl::TextureTarget::Texture2D);
        note("glCompressedTexImage2D of the illegal-encoding blocks (linear and sRGB)");

        // ---- ETC2 / EAC blocks (arbitrary bits: every 64- or 128-bit pattern is a valid ETC2 / EAC block) ----
        std::uint32_t seed = 0x5EEDu;
        for (int i = 0; i < 32; ++i) etc2_rgba_.push_back(static_cast<std::uint8_t>(lcg(seed)));
        for (int i = 0; i < 16; ++i) etc2_rgb_.push_back(static_cast<std::uint8_t>(lcg(seed)));
        etc2_rgba_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, etc2_rgba_tex_);
        mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::Rgba8Etc2Eac, 8, 4, 0, 32, etc2_rgba_.data());
        nearest(mgl::TextureTarget::Texture2D);
        etc2_rgb_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, etc2_rgb_tex_);
        mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::CompressedInternalFormat::Rgb8Etc2, 8, 4, 0, 16, etc2_rgb_.data());
        nearest(mgl::TextureTarget::Texture2D);
        note("glCompressedTexImage2D(RGBA8_ETC2_EAC 8 x 4 and RGB8_ETC2 8 x 4)");
    }

    void AstcAndCopyImage::build_mosaic()
    {
        // The palette texture: RGBA32UI, one texel per raw block.
        palette_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, palette_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba32UI, static_cast<GLsizei>(palette_.size()), 1);
        std::vector<std::uint32_t> words;
        for (const Block& b : palette_)
        {
            const U4 w = block_words(b);
            words.insert(words.end(), w.begin(), w.end());
        }
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, static_cast<GLsizei>(palette_.size()), 1, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt,
                             words.data());
        nearest(mgl::TextureTarget::Texture2D);
        note("palette: glTexStorage2D(RGBA32UI) + glTexSubImage2D(RGBA_INTEGER, UNSIGNED_INT)");

        // The mosaic: a sunset over the sea, 8 x 8 blocks, top row first ('a'..'p' = atlas block, '0'..'7' = palette texel).
        constexpr std::array<std::string_view, 8> scene = {
            "oo4o5ldo",  // sky with a star and clouds
            "nlpnnn4n",
            "mmm2kk2m",  // the sun, with its glow
            "jjj2kk2j",  // horizon
            "hhhhiihh",  // misty water, the reflection
            "f0f0gg0f",  // waves, glitter
            "6eeegge6",  // deep water
            "a3b17b3a",  // the beach
        };
        mosaic_src_.assign(64, ' ');
        for (int by = 0; by < 8; ++by)
            for (int bx = 0; bx < 8; ++bx)
                mosaic_src_[static_cast<std::size_t>(by * 8 + bx)] = scene[static_cast<std::size_t>(7 - by)][static_cast<std::size_t>(bx)];  // block row 0 is the bottom
        mosaic_img_ = {32, 32, 4, 4, {}};
        for (const char c : mosaic_src_)
            mosaic_img_.blocks.push_back(c >= 'a' ? atlas_.blocks[static_cast<std::size_t>(c - 'a')] : palette_[static_cast<std::size_t>(c - '0')]);

        mosaic_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mosaic_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::CompressedInternalFormat::RgbaAstc4x4, 32, 32);
        nearest(mgl::TextureTarget::Texture2D);
        int from_atlas = 0, from_palette = 0;
        for (int by = 0; by < 8; ++by)
            for (int bx = 0; bx < 8; ++bx)
            {
                const char c = mosaic_src_[static_cast<std::size_t>(by * 8 + bx)];
                if (c >= 'a')
                {
                    // compressed -> compressed: the region is given in texels (one 4 x 4 block)
                    const int i = c - 'a';
                    mgl::glCopyImageSubData(atlas_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, (i % 4) * 4, (i / 4) * 4, 0, mosaic_tex_,
                                            mgl::ImageCopyTextureTarget::Texture2D, 0, bx * 4, by * 4, 0, 4, 4, 1);
                    ++from_atlas;
                }
                else
                {
                    // uncompressed -> compressed: one RGBA32UI texel becomes one block ("each texel of uncompressed data becomes a single block")
                    mgl::glCopyImageSubData(palette_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, c - '0', 0, 0, mosaic_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0,
                                            bx * 4, by * 4, 0, 1, 1, 1);
                    ++from_palette;
                }
            }
        note(std::format("mosaic: {} glCopyImageSubData calls ASTC -> ASTC (4 x 4 texels each) and {} RGBA32UI -> ASTC (1 texel -> 1 block)", from_atlas, from_palette));

        // compressed -> uncompressed: the whole atlas (16 x 16 source texels) and the whole mosaic into RGBA32UI textures of 4 x 4 and 8 x 8 texels.
        atlas_u32_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, atlas_u32_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba32UI, 4, 4);
        nearest(mgl::TextureTarget::Texture2D);
        // TEXTURE_MIN/MAG_FILTER LINEAR makes an integer texture incomplete for sampling ("not texture-filterable", section 8.17), but
        // CopyImageSubData ignores "format-based completeness rules": the copy below must succeed.
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Linear);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Linear);
        mgl::glCopyImageSubData(atlas_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, atlas_u32_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 16, 16, 1);
        note("glCopyImageSubData(ASTC 4x4 16 x 16 texels -> RGBA32UI 4 x 4, destination LINEAR-filtered = format-incomplete)");
        nearest(mgl::TextureTarget::Texture2D);

        mosaic_u32_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mosaic_u32_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba32UI, 8, 8);
        nearest(mgl::TextureTarget::Texture2D);
        mgl::glCopyImageSubData(mosaic_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, mosaic_u32_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 32, 32, 1);
        note("glCopyImageSubData(mosaic ASTC 4x4 32 x 32 -> RGBA32UI 8 x 8)");

        // ASTC linear -> ASTC sRGB (the same VIEW_CLASS_ASTC_4x4_RGBA): the bits are copied, the new format decodes them as sRGB.
        srgb_copy_tex_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, srgb_copy_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::CompressedInternalFormat::Srgb8Alpha8Astc4x4, 16, 16);
        nearest(mgl::TextureTarget::Texture2D);
        mgl::glCopyImageSubData(atlas_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, srgb_copy_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 16, 16, 1);
        note("glCopyImageSubData(RGBA_ASTC_4x4 -> SRGB8_ALPHA8_ASTC_4x4, 16 x 16)");

        // 12x12 footprint <-> RGBA32UI: 2 x 2 aligned blocks, and the partial block at the right edge (x = 24, width 6 = up to the edge).
        astc12_u32_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, astc12_u32_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba32UI, 3, 3);
        nearest(mgl::TextureTarget::Texture2D);

        // ETC2 / EAC <-> uncompressed of the same size.
        etc2_rgba_u32_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, etc2_rgba_u32_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba32UI, 2, 1);
        nearest(mgl::TextureTarget::Texture2D);
        mgl::glCopyImageSubData(etc2_rgba_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, etc2_rgba_u32_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 8, 4, 1);
        etc2_rgb_u16_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, etc2_rgb_u16_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba16UI, 2, 1);
        nearest(mgl::TextureTarget::Texture2D);
        mgl::glCopyImageSubData(etc2_rgb_tex_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, etc2_rgb_u16_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 8, 4, 1);
        // ... and back: RGBA16UI -> a second RGB8_ETC2 texture -> a second RGBA16UI texture (the bits must survive both directions).
        etc2_rgb_back_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, etc2_rgb_back_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::CompressedInternalFormat::Rgb8Etc2, 8, 4);
        nearest(mgl::TextureTarget::Texture2D);
        mgl::glCopyImageSubData(etc2_rgb_u16_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, etc2_rgb_back_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 2, 1, 1);
        etc2_rgb_u16_back_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, etc2_rgb_u16_back_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba16UI, 2, 1);
        nearest(mgl::TextureTarget::Texture2D);
        mgl::glCopyImageSubData(etc2_rgb_back_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, etc2_rgb_u16_back_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 8, 4, 1);
        note("glCopyImageSubData ETC2 / EAC <-> RGBA32UI / RGBA16UI");
    }

    void AstcAndCopyImage::build_copy_textures()
    {
        const auto storage2d = [&](mgl::SizedInternalFormat f, int w, int h) {
            const mgl::TextureId t = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, f, w, h);
            nearest(mgl::TextureTarget::Texture2D);
            return t;
        };
        const auto upload_rgba8 = [](mgl::TextureId t, int seed, int w, int h) {
            const std::vector<std::uint8_t> img = rgba_image(seed, w, h);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());
        };

        // ---- RGBA8 2D textures and renderbuffers ----
        rgba8_src_ = storage2d(mgl::SizedInternalFormat::Rgba8, 8, 8);
        upload_rgba8(rgba8_src_, 1, 8, 8);
        rgba8_dst_ = storage2d(mgl::SizedInternalFormat::Rgba8, 8, 8);
        upload_rgba8(rgba8_dst_, 2, 8, 8);
        rgba8_dst2_ = storage2d(mgl::SizedInternalFormat::Rgba8, 8, 8);
        upload_rgba8(rgba8_dst2_, 3, 8, 8);
        note("RGBA8 8 x 8 textures (glTexStorage2D + glTexSubImage2D)");

        rb_ = new_renderbuffer();
        mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_);
        mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 8, 8);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
        mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), mgl::RenderbufferTarget::Renderbuffer, rb_);
        mgl::glViewport(0, 0, 8, 8);
        const std::array<float, 4> outer = {0.2f, 0.4f, 0.6f, 0.8f}, inner = {1.0f, 0.5f, 0.0f, 1.0f};
        mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, outer.data());
        mgl::glEnable(mgl::Capability::ScissorTest);
        mgl::glScissor(2, 3, 4, 3);
        mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, inner.data());
        mgl::glDisable(mgl::Capability::ScissorTest);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        rb_before_ = read_renderbuffer_rgba8(rb_, 8, 8);
        note("RGBA8 renderbuffer cleared with glClearBufferfv (outer and scissored inner rectangle)");

        // renderbuffer -> texture, and texture -> renderbuffer (the typed RenderbufferId overloads pass target RENDERBUFFER, level 0)
        mgl::glCopyImageSubData(rb_, 1, 2, rgba8_dst2_, mgl::ImageCopyTextureTarget::Texture2D, 0, 2, 1, 0, 6, 5);
        note("glCopyImageSubData(renderbuffer (1, 2) 6 x 5 -> texture (2, 1))");
        mgl::glCopyImageSubData(rgba8_src_, mgl::ImageCopyTextureTarget::Texture2D, 0, 4, 4, 0, rb_, 0, 0, 3, 2);
        note("glCopyImageSubData(texture (4, 4) 3 x 2 -> renderbuffer (0, 0))");

        // texture -> texture
        mgl::glCopyImageSubData(rgba8_src_, mgl::ImageCopyTextureTarget::Texture2D, 0, 1, 2, 0, rgba8_dst_, mgl::ImageCopyTextureTarget::Texture2D, 0, 3, 4, 0, 5, 3, 1);
        note("glCopyImageSubData(RGBA8 texture (1, 2) 5 x 3 -> texture (3, 4))");

        // ---- 2D arrays and cube maps (4 x 4 RGBA8) ----
        const auto storage3d = [&](mgl::TextureTarget target, int layers) {
            const mgl::TextureId t = new_texture();
            mgl::glBindTexture(target, t);
            mgl::glTexStorage3D(target, 1, mgl::SizedInternalFormat::Rgba8, 4, 4, layers);
            nearest(target);
            for (int l = 0; l < layers; ++l)
            {
                const std::vector<std::uint8_t> img = rgba_image(10 + l + (layers == 6 ? 20 : 0), 4, 4);
                mgl::glTexSubImage3D(target, 0, 0, 0, l, 4, 4, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());
            }
            return t;
        };
        arr_src_ = storage3d(mgl::TextureTarget::Texture2DArray, 4);
        arr_dst_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, arr_dst_);
        mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 1, mgl::SizedInternalFormat::Rgba8, 4, 4, 4);
        nearest(mgl::TextureTarget::Texture2DArray);
        for (int l = 0; l < 4; ++l)
        {
            const std::vector<std::uint8_t> img = rgba_image(40 + l, 4, 4);
            mgl::glTexSubImage3D(mgl::TextureTarget::Texture2DArray, 0, 0, 0, l, 4, 4, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());
        }
        arr6_ = storage3d(mgl::TextureTarget::Texture2DArray, 6);
        note("RGBA8 2D array textures (4 x 4 x 4, 4 x 4 x 6)");
        // layers 1 and 2 of arr_src_ -> layers 2 and 3 of arr_dst_ (one call, srcDepth = 2)
        mgl::glCopyImageSubData(arr_src_, mgl::ImageCopyTextureTarget::Texture2DArray, 0, 0, 0, 1, arr_dst_, mgl::ImageCopyTextureTarget::Texture2DArray, 0, 0, 0, 2, 4, 4, 2);
        note("glCopyImageSubData(2D array layers 1..2 -> layers 2..3, srcDepth 2)");

        cube_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube_);
        mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, 4, 4);
        nearest(mgl::TextureTarget::TextureCubeMap);
        // six array layers -> the six cube faces (z = face index in the order of Table 8.20) in one call
        mgl::glCopyImageSubData(arr6_, mgl::ImageCopyTextureTarget::Texture2DArray, 0, 0, 0, 0, cube_, mgl::ImageCopyTextureTarget::TextureCubeMap, 0, 0, 0, 0, 4, 4, 6);
        note("glCopyImageSubData(2D array layers 0..5 -> cube map faces 0..5, srcDepth 6)");
        face_out_ = storage2d(mgl::SizedInternalFormat::Rgba8, 4, 4);
        upload_rgba8(face_out_, 60, 4, 4);
        // cube face 4 (POSITIVE_Z), the 2 x 2 texels at (1, 1) -> 2D texture at (2, 0)
        mgl::glCopyImageSubData(cube_, mgl::ImageCopyTextureTarget::TextureCubeMap, 0, 1, 1, 4, face_out_, mgl::ImageCopyTextureTarget::Texture2D, 0, 2, 0, 0, 2, 2, 1);
        note("glCopyImageSubData(cube face 4 (1, 1) 2 x 2 -> 2D texture (2, 0))");

        // ---- reinterpretation between formats of the same size (VIEW_CLASS_32_BITS): RGBA8UI <-> R32F ----
        // Bytes chosen so that every 32-bit word is a normal float (exponent byte between 0x30 and 0x4F): the copy itself is a memcpy, but
        // the R32F read-back goes through the texture unit, which need not preserve denormals or NaN payloads.
        std::uint32_t seed = 0xC0FFEEu;
        for (int i = 0; i < 16; ++i)
        {
            u8_data_.push_back(static_cast<std::uint8_t>(lcg(seed)));
            u8_data_.push_back(static_cast<std::uint8_t>(lcg(seed)));
            u8_data_.push_back(static_cast<std::uint8_t>(lcg(seed)));
            u8_data_.push_back(static_cast<std::uint8_t>(0x30 + lcg(seed) % 0x20));
        }
        u8_src_ = storage2d(mgl::SizedInternalFormat::Rgba8UI, 4, 4);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedByte, u8_data_.data());
        f32_dst_ = storage2d(mgl::SizedInternalFormat::R32F, 4, 4);
        mgl::glCopyImageSubData(u8_src_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, f32_dst_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 4, 4, 1);
        note("glCopyImageSubData(RGBA8UI 4 x 4 -> R32F)");
        for (int i = 0; i < 16; ++i) f32_data_.push_back(static_cast<float>(i) * 1.375f - 7.0625f + (i == 5 ? 1e-3f : 0.0f));
        f32_src_ = storage2d(mgl::SizedInternalFormat::R32F, 4, 4);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, mgl::PixelFormat::Red, mgl::PixelType::Float, f32_data_.data());
        u8_dst_ = storage2d(mgl::SizedInternalFormat::Rgba8UI, 4, 4);
        mgl::glCopyImageSubData(f32_src_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, u8_dst_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, 4, 4, 1);
        note("glCopyImageSubData(R32F 4 x 4 -> RGBA8UI)");

        // ---- depth and depth-stencil textures (same format only: depth formats are in no class of Table 16.4) ----
        for (int i = 0; i < 16; ++i)
        {
            d32_src_data_.push_back(static_cast<float>(i) / 16.0f + 0.03125f);
            d32_dst_data_.push_back(0.75f);
            ds_src_data_.push_back((static_cast<std::uint32_t>(i) * 0x0F0F0Fu + 0x123456u) << 8 | static_cast<std::uint32_t>(17 * i + 3));
            ds_dst_data_.push_back(0x800000u << 8 | 200u);
        }
        d32_src_ = storage2d(mgl::SizedInternalFormat::DepthComponent32F, 4, 4);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, mgl::PixelFormat::DepthComponent, mgl::PixelType::Float, d32_src_data_.data());
        d32_dst_ = storage2d(mgl::SizedInternalFormat::DepthComponent32F, 4, 4);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, mgl::PixelFormat::DepthComponent, mgl::PixelType::Float, d32_dst_data_.data());
        mgl::glCopyImageSubData(d32_src_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 0, 0, d32_dst_, mgl::ImageCopyTextureTarget::Texture2D, 0, 1, 1, 0, 3, 2, 1);
        note("glCopyImageSubData(DEPTH_COMPONENT32F (0, 0) 3 x 2 -> (1, 1))");
        ds_src_ = storage2d(mgl::SizedInternalFormat::Depth24Stencil8, 4, 4);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, mgl::PixelFormat::DepthStencil, mgl::PixelType::UnsignedInt248, ds_src_data_.data());
        ds_dst_ = storage2d(mgl::SizedInternalFormat::Depth24Stencil8, 4, 4);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 4, 4, mgl::PixelFormat::DepthStencil, mgl::PixelType::UnsignedInt248, ds_dst_data_.data());
        mgl::glCopyImageSubData(ds_src_, mgl::ImageCopyTextureTarget::Texture2D, 0, 1, 0, 0, ds_dst_, mgl::ImageCopyTextureTarget::Texture2D, 0, 0, 2, 0, 3, 2, 1);
        note("glCopyImageSubData(DEPTH24_STENCIL8 (1, 0) 3 x 2 -> (0, 2))");

        // ---- textures for the completeness and error tests ----
        incomplete_ = new_texture();  // mutable, level 0 only, default MIN_FILTER NEAREST_MIPMAP_LINEAR: not mipmap complete -> not complete
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, incomplete_);
        {
            const std::vector<std::uint8_t> img = rgba_image(70, 4, 4);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 4, 4, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());
        }
        incomplete_dst_ = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, incomplete_dst_);
        {
            const std::vector<std::uint8_t> img = rgba_image(71, 4, 4);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 4, 4, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());
        }
        rg8_ = storage2d(mgl::SizedInternalFormat::Rg8, 8, 8);
        rgba16f_ = storage2d(mgl::SizedInternalFormat::Rgba16F, 8, 8);
        rgba16ui_ = storage2d(mgl::SizedInternalFormat::Rgba16UI, 4, 4);
        ms_rb_ = new_renderbuffer();
        mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, ms_rb_);
        mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, 4, mgl::InternalFormat::Rgba8, 8, 8);
        mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
        note("textures for the completeness and error tests");
    }

    // =================================================================================================
    // Picture
    // =================================================================================================
    void AstcAndCopyImage::draw_picture(int width, int height)
    {
        const Layout lay(width, height);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glViewport(0, 0, width, height);
        mgl::glClearColor(static_cast<float>(kClear[0]) / 255.0f, static_cast<float>(kClear[1]) / 255.0f, static_cast<float>(kClear[2]) / 255.0f, 1.0f);
        mgl::glClear(mgl::ClearBufferBit::Color);
        mgl::glUseProgram(panel_);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glUniform1i(mgl::glGetUniformLocation(panel_, "u_tex"), 0);
        const std::array<std::pair<Rect, std::pair<mgl::TextureId, int>>, 2> panels = {{{lay.atlas, {atlas_tex_, 16}}, {lay.mosaic, {mosaic_tex_, 32}}}};
        for (const auto& [r, tex] : panels)
        {
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex.first);
            mgl::glUniform2f(mgl::glGetUniformLocation(panel_, "u_origin"), static_cast<float>(r.x), static_cast<float>(r.y));
            mgl::glUniform2f(mgl::glGetUniformLocation(panel_, "u_panel"), static_cast<float>(r.w), static_cast<float>(r.h));
            mgl::glUniform2i(mgl::glGetUniformLocation(panel_, "u_texels"), tex.second, tex.second);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        mgl::glViewport(0, 0, width, height);
    }

    void AstcAndCopyImage::render(demo::Env& env, int /*frame*/) { draw_picture(env.width, env.height); }

    // =================================================================================================
    // Read-back helpers
    // =================================================================================================
    AstcAndCopyImage::Fetched AstcAndCopyImage::fetch(bool uint_kind, mgl::TextureId tex, int w, int h)
    {
        const std::size_t count = 1 + static_cast<std::size_t>(w * h);
        const std::vector<U4> sentinel(count, U4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu});
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(count * sizeof(U4)), sentinel.data(), mgl::BufferUsage::DynamicCopy);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, ssbo_);
        const mgl::ProgramId program = uint_kind ? fetch_u_ : fetch_f_;
        mgl::glUseProgram(program);
        mgl::glUniform2i(mgl::glGetUniformLocation(program, "u_size"), w, h);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
        mgl::glDispatchCompute(static_cast<GLuint>((w + 7) / 8), static_cast<GLuint>((h + 7) / 8), 1);
        // The mapped read below is a buffer-object access by the commands of section 6 (ES 3.2 section 7.11.2): BUFFER_UPDATE_BARRIER_BIT.
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

        std::vector<U4> raw(count);
        const void* p = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, static_cast<GLsizeiptr>(count * sizeof(U4)), mgl::MapBufferAccessMask::Read);
        if (p == nullptr) throw demo::Failure("glMapBufferRange failed");
        std::memcpy(raw.data(), p, count * sizeof(U4));
        if (mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage) != GL_TRUE) throw demo::Failure("glUnmapBuffer returned FALSE");
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
        Fetched f;
        f.size = {std::bit_cast<std::int32_t>(raw[0][0]), std::bit_cast<std::int32_t>(raw[0][1])};
        f.texels.assign(raw.begin() + 1, raw.end());
        return f;
    }

    std::vector<std::uint8_t> AstcAndCopyImage::read_texture_rgba8(mgl::TextureTarget target, mgl::TextureId tex, int layer, int w, int h)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
        const auto attachment = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
        if (layer >= 0)
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, attachment, tex, 0, layer);
        else
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, attachment, target, tex, 0);
        const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
        if (status != mgl::FramebufferStatus::Complete)
            throw demo::Failure(std::format("read-back framebuffer incomplete: {}", demo::framebuffer_status_name(status)));
        std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, w, h);
        mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, attachment, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        return px;
    }

    std::vector<std::uint8_t> AstcAndCopyImage::read_renderbuffer_rgba8(mgl::RenderbufferId rb, int w, int h)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
        const auto attachment = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
        mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, attachment, mgl::RenderbufferTarget::Renderbuffer, rb);
        const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
        if (status != mgl::FramebufferStatus::Complete)
            throw demo::Failure(std::format("renderbuffer read-back framebuffer incomplete: {}", demo::framebuffer_status_name(status)));
        std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, w, h);
        mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, attachment, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        return px;
    }

    // =================================================================================================
    // Validation
    // =================================================================================================
    void AstcAndCopyImage::validate(demo::Env& env, demo::Check& check)
    {
        check_picture(check, env.width, env.height);  // first, while the default framebuffer still holds the rendered frame
        for (const auto& [what, e] : setup_errors_)
            check.expect(e == mgl::ErrorCode::NoError, std::format("setup without GL error: {}", what), std::format("glGetError() = {}", demo::error_name(e)));
        check_formats_query(check);
        check_decoder_design(check);
        check_astc_decoding(check);
        check_compressed_uploads(check);
        check_compressed_copies(check);
        check_uncompressed_copies(check);
        check_layers_and_faces(check);
        check_reinterpretation(check);
        check_depth_stencil(check);
        check_completeness(check);
        check_copy_errors(check);

        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        check.expect_no_gl_error("no stray GL error left by the validation");
        render(env, 0);  // leave the picture in the framebuffer for the capture
    }

    // ---- COMPRESSED_TEXTURE_FORMATS ----------------------------------------------------------------
    void AstcAndCopyImage::check_formats_query(demo::Check& check)
    {
        const int n = demo::get_integer(mgl::GetParameter::NumCompressedTextureFormats);
        std::vector<GLint> formats(static_cast<std::size_t>(std::max(n, 0)) + 1, -1);
        mgl::glGetIntegerv(mgl::GetParameter::CompressedTextureFormats, formats.data());
        check.expect(formats[static_cast<std::size_t>(std::max(n, 0))] == -1, "COMPRESSED_TEXTURE_FORMATS writes exactly NUM_COMPRESSED_TEXTURE_FORMATS values",
                     std::format("NUM = {}", n));
        formats.resize(static_cast<std::size_t>(std::max(n, 0)));
        const auto has = [&](mgl::CompressedInternalFormat f) { return std::find(formats.begin(), formats.end(), static_cast<GLint>(std::to_underlying(f))) != formats.end(); };
        int missing = 0;
        std::string list;
        for (const AstcFormat& f : kAstcFormats)
            for (const mgl::CompressedInternalFormat v : {f.linear, f.srgb})
                if (!has(v))
                {
                    ++missing;
                    list += std::format(" 0x{:04X}", std::to_underlying(v));
                }
        check.expect(n >= 38 && missing == 0,
                     "COMPRESSED_TEXTURE_FORMATS lists all 28 ASTC LDR formats of Table 8.17 (\"All implementations support at least the formats listed in table 8.17\", section 8.7)",
                     std::format("NUM_COMPRESSED_TEXTURE_FORMATS = {}, missing ASTC:{}", n, missing == 0 ? " none" : list));
        int etc_missing = 0;
        for (const mgl::CompressedInternalFormat f :
             {mgl::CompressedInternalFormat::R11Eac, mgl::CompressedInternalFormat::SignedR11Eac, mgl::CompressedInternalFormat::Rg11Eac, mgl::CompressedInternalFormat::SignedRg11Eac,
              mgl::CompressedInternalFormat::Rgb8Etc2, mgl::CompressedInternalFormat::Srgb8Etc2, mgl::CompressedInternalFormat::Rgb8PunchthroughAlpha1Etc2,
              mgl::CompressedInternalFormat::Srgb8PunchthroughAlpha1Etc2, mgl::CompressedInternalFormat::Rgba8Etc2Eac, mgl::CompressedInternalFormat::Srgb8Alpha8Etc2Eac})
            etc_missing += has(f) ? 0 : 1;
        check.expect(etc_missing == 0, "COMPRESSED_TEXTURE_FORMATS also lists the 10 ETC2 / EAC formats of Table 8.17", std::format("{} missing", etc_missing));
    }

    // ---- the test data itself ----------------------------------------------------------------------
    void AstcAndCopyImage::check_decoder_design(demo::Check& check)
    {
        // ES 3.2 does not say which ASTC decode mode a texture fetch uses (the decode-mode selection is an extension). The data are chosen so
        // that UNORM8 decoding (the top 8 bits of C) and decode_float16 (C / 65536, round to zero; 1.0 for 65535) give the same 8-bit value:
        // |decode_float16 * 255 - (C >> 8)| <= 0.375 for every channel of every non-error texel (sRGB: alpha only; RGB is C >> 8 by definition).
        Tally t;
        double worst = 0.0;
        const std::array<std::pair<const AstcImage*, const char*>, 5> images = {
            {{&atlas_, "atlas"}, {&mosaic_img_, "mosaic"}, {&astc6_, "6x6"}, {&astc5x4_, "5x4"}, {&astc12_, "12x12"}}};
        for (const auto& [img, name] : images)
            for (const bool srgb : {false, true})
                for (int y = 0; y < img->h; ++y)
                    for (int x = 0; x < img->w; ++x)
                    {
                        const Decoded d = img->decode(x, y, srgb);
                        if (d.error) continue;
                        for (std::size_t ch = srgb ? 3 : 0; ch < 4; ++ch)
                        {
                            const double e = std::abs(f16_rtz(d.c[ch]) * 255.0 - static_cast<double>(d.c[ch] >> 8));
                            worst = std::max(worst, e);
                            t.add(e <= 0.375, [&] { return std::format("{}{} texel ({}, {}) channel {}: C = {}", name, srgb ? " (sRGB)" : "", x, y, ch, d.c[ch]); });
                        }
                    }
        check.expect(t.bad == 0, "test data: decode_float16 and UNORM8 decoding agree on the 8-bit value of every texel (margin 0.375)",
                     std::format("{}; worst |f16 * 255 - (C >> 8)| = {:.4f}", t.detail(), worst));
        int errors = 0, valid = 0;
        for (int x = 0; x < errors_img_.w; ++x) (errors_img_.decode(x, 0, false).error ? errors : valid)++;
        check.expect(errors == errors_img_.w && valid == 0, "CPU decoder: the three illegal blocks (HDR void extent, reserved block mode, 5x4 grid in a 4x4 footprint) are error blocks",
                     std::format("{} error / {} valid texels in a row", errors, valid));
    }

    // ---- ASTC decoding -----------------------------------------------------------------------------
    void AstcAndCopyImage::check_decode(demo::Check& check, std::string_view label, mgl::TextureId tex, const AstcImage& img, bool srgb)
    {
        const Fetched f = fetch(false, tex, img.w, img.h);
        check.expect(f.size[0] == img.w && f.size[1] == img.h, std::format("{}: textureSize = {} x {}", label, img.w, img.h),
                     std::format("textureSize = {} x {}", f.size[0], f.size[1]));
        Tally t;
        int exact_unorm8 = 0, exact_unorm16 = 0, exact_f16 = 0, nan_errors = 0, magenta_errors = 0, compared = 0;
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x)
            {
                const Decoded d = img.decode(x, y, srgb);
                const std::array<float, 4> m = as_float(f.texels[static_cast<std::size_t>(y * img.w + x)]);
                ++compared;
                if (d.error)
                {
                    const bool magenta = m[0] == 1.0f && m[1] == 0.0f && m[2] == 1.0f && m[3] == 1.0f;
                    const bool nan = std::isnan(m[0]) && std::isnan(m[1]) && std::isnan(m[2]) && std::isnan(m[3]);
                    magenta_errors += magenta ? 1 : 0;
                    nan_errors += nan ? 1 : 0;
                    // DFS 18.5: the error colour is opaque magenta; "For linear RGB decode, the error color may be either opaque fully-saturated
                    // magenta (R,G,B,A) = (1.0, 0.0, 1.0, 1.0) or a vector of four NaNs". sRGB decoding returns 8-bit values: magenta only.
                    t.add(magenta || (!srgb && nan),
                          [&] { return std::format("texel ({}, {}): error block decoded as ({}, {}, {}, {}), expected the error colour", x, y, m[0], m[1], m[2], m[3]); });
                    continue;
                }
                const std::array<std::uint8_t, 4> want = top8(d);
                bool ok = true;
                bool is_unorm8 = true, is_unorm16 = true, is_f16 = true;
                for (std::size_t ch = 0; ch < 4; ++ch)
                {
                    const double got = static_cast<double>(m[ch]);
                    if (srgb && ch < 3)
                    {
                        // The decoder hands the 8-bit sRGB value to the sRGB conversion (DFS 18.19, ES 3.2 section 8.21): recover the code.
                        ok = ok && std::lround(linear_to_srgb(got) * 255.0) == want[ch];
                        is_unorm8 = is_unorm8 && std::abs(got - srgb_to_linear(want[ch] / 255.0)) < 1e-6;
                        continue;
                    }
                    ok = ok && std::lround(got * 255.0) == want[ch] && std::abs(got * 255.0 - want[ch]) <= 0.5;
                    is_unorm8 = is_unorm8 && got == static_cast<double>(static_cast<float>(want[ch] / 255.0));
                    is_unorm16 = is_unorm16 && got == static_cast<double>(static_cast<float>(d.c[ch] / 65535.0));
                    is_f16 = is_f16 && got == f16_rtz(d.c[ch]);
                }
                exact_unorm8 += is_unorm8 ? 1 : 0;
                exact_f16 += is_f16 ? 1 : 0;
                exact_unorm16 += is_unorm16 ? 1 : 0;
                t.add(ok, [&] {
                    return std::format("texel ({}, {}): got ({:.6f}, {:.6f}, {:.6f}, {:.6f}), expected 8-bit ({}, {}, {}, {})", x, y, m[0], m[1], m[2], m[3], want[0], want[1], want[2],
                                       want[3]);
                });
            }
        check.expect(t.bad == 0, std::format("{}: every texel equals the CPU decoder (DFS chapter 18) at 8-bit precision", label),
                     std::format("{}; texels returned exactly as (C >> 8) / 255: {}, as C / 65535: {}, as decode_float16: {}; error texels magenta / NaN: {} / {}", t.detail(),
                                 exact_unorm8, srgb ? 0 : exact_unorm16, srgb ? 0 : exact_f16, magenta_errors, nan_errors));
    }

    void AstcAndCopyImage::check_astc_decoding(demo::Check& check)
    {
        check_decode(check, "RGBA_ASTC_4x4 atlas (void extents, CEM 0 / 4 / 8 / 12, blue contraction, 3x3 -> 4x4 infill)", atlas_tex_, atlas_, false);
        check_decode(check, "SRGB8_ALPHA8_ASTC_4x4 atlas (same bits; RGB endpoints expanded with 0x80, sRGB -> linear)", atlas_srgb_tex_, atlas_, true);
        check_decode(check, "RGBA_ASTC_6x6, 17 x 10 (partial blocks; 3x3 and 4x4 grids infilled to 6 x 6)", astc6_tex_, astc6_, false);
        check_decode(check, "illegal encodings, RGBA_ASTC_4x4: error colour", errors_lin_tex_, errors_img_, false);
        check_decode(check, "illegal encodings, SRGB8_ALPHA8_ASTC_4x4: error colour (magenta)", errors_srgb_tex_, errors_img_, true);

        // Negative control 1: the sRGB texture holds the same bits as the linear one, so its texels differ from the linear prediction exactly
        // where sRGB decoding changes the value. Negative control 2: a CPU prediction with one block's weights mirrored disagrees with the GPU.
        const Fetched srgb = fetch(false, atlas_srgb_tex_, 16, 16);
        int differ = 0;
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
            {
                const auto want = top8(atlas_.decode(x, y, false));
                const auto m = as_float(srgb.texels[static_cast<std::size_t>(y * 16 + x)]);
                bool same = true;
                for (std::size_t ch = 0; ch < 4; ++ch) same = same && std::lround(static_cast<double>(m[ch]) * 255.0) == want[ch];
                differ += same ? 0 : 1;
            }
        check.expect(differ > 100, "negative control: the sRGB atlas does not match the linear prediction (the comparison sees the sRGB decode)",
                     std::format("{} of 256 texels differ", differ));
        AstcImage mutated = atlas_;
        mutated.blocks[5] = palette_[0];  // the wave block with its weights mirrored
        const Fetched lin = fetch(false, atlas_tex_, 16, 16);
        int caught = 0;
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
            {
                const auto want = top8(mutated.decode(x, y, false));
                const auto m = as_float(lin.texels[static_cast<std::size_t>(y * 16 + x)]);
                bool same = true;
                for (std::size_t ch = 0; ch < 4; ++ch) same = same && std::lround(static_cast<double>(m[ch]) * 255.0) == want[ch];
                caught += same ? 0 : 1;
            }
        check.expect(caught == 16, "negative control: mirroring the weights of one block in the CPU prediction makes all 16 texels of that block mismatch",
                     std::format("{} texels mismatch", caught));
    }

    // ---- CompressedTexImage / CompressedTexSubImage rules ------------------------------------------
    void AstcAndCopyImage::check_compressed_uploads(demo::Check& check)
    {
        for (const auto& [what, e] : upload_errors_)
            check.expect(e == mgl::ErrorCode::InvalidValue,
                         std::format("{}: INVALID_VALUE (\"imageSize parameter is not consistent with the format, dimensions, and contents\", section 8.7)", what),
                         std::format("glGetError() = {}", demo::error_name(e)));

        // ---- 5x4 footprint, 15 x 8 texels ----
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, astc5x4_tex_);
        const auto sub = [](int x, int y, int w, int h, mgl::CompressedInternalFormat f, int size, const std::vector<std::uint8_t>& data) {
            mgl::glCompressedTexSubImage2D(mgl::TextureTarget::Texture2D, 0, x, y, w, h, f, size, data.data());
        };
        const auto f54 = mgl::CompressedInternalFormat::RgbaAstc5x4;
        const std::vector<std::uint8_t> two = block_bytes({rgb_block(4, 4, 2, {0, 64, 192}, {192, 128, 0}, ramp_x(4, 4, 2)), solid(255, 64, 0)});
        sub(5, 4, 10, 4, f54, 32, two);  // blocks (1, 1) and (2, 1)
        check.expect_no_gl_error("glCompressedTexSubImage2D(5x4, offset (5, 4), 10 x 4, imageSize 32): accepted");
        astc5x4_.blocks[4] = rgb_block(4, 4, 2, {0, 64, 192}, {192, 128, 0}, ramp_x(4, 4, 2));
        astc5x4_.blocks[5] = solid(255, 64, 0);
        const std::vector<std::uint8_t> one = block_bytes({solid(0, 0, 255)});
        sub(2, 0, 5, 4, f54, 16, one);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glCompressedTexSubImage2D(5x4, xoffset 2): INVALID_OPERATION (\"xoffset or yoffset is not a multiple of the block width or height\")");
        sub(0, 0, 5, 2, f54, 16, one);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                              "glCompressedTexSubImage2D(5x4, height 2 at yoffset 0 in an 8-texel-high level): INVALID_OPERATION (\"height is not a multiple of the format's block height, and height + yoffset is not equal to the value of TEXTURE_HEIGHT\")");
        sub(0, 0, 10, 4, f54, 16, two);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCompressedTexSubImage2D(5x4, 10 x 4 = 2 blocks, imageSize 16): INVALID_VALUE (\"imageSize is not consistent ... (too little or too much data)\")");
        sub(0, 0, 5, 4, f54, 32, two);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCompressedTexSubImage2D(5x4, 5 x 4 = 1 block, imageSize 32): INVALID_VALUE (too much data)");
        sub(0, 0, 5, 4, mgl::CompressedInternalFormat::Srgb8Alpha8Astc5x4, 16, one);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                              "glCompressedTexSubImage2D(SRGB8_ALPHA8_ASTC_5x4 into an RGBA_ASTC_5x4 level): INVALID_OPERATION (\"format does not match the internal format of the texture image being modified\")");
        check_decode(check, "RGBA_ASTC_5x4, 15 x 8 after the sub-image updates (untouched blocks preserved, rejected calls without effect)", astc5x4_tex_, astc5x4_, false);

        // ---- 12x12 footprint, 30 x 30 texels ----
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, astc12_tex_);
        const auto f12 = mgl::CompressedInternalFormat::RgbaAstc12x12;
        const Block grad12 = rgb_block(4, 4, 2, {255, 192, 0}, {63, 0, 128}, ramp_diag(4, 4, 2));
        const Block lum12 = lum_block(3, 3, 3, 0, 192, bump(3, 3, 3));
        sub(12, 12, 12, 12, f12, 16, block_bytes({grad12}));
        check.expect_no_gl_error("glCompressedTexSubImage2D(12x12, offset (12, 12), 12 x 12, imageSize 16): accepted");
        sub(24, 0, 6, 12, f12, 16, block_bytes({lum12}));
        check.expect_no_gl_error("glCompressedTexSubImage2D(12x12, offset (24, 0), width 6 = up to TEXTURE_WIDTH 30): accepted");
        sub(0, 24, 12, 6, f12, 16, block_bytes({solid(255, 255, 255)}));
        check.expect_no_gl_error("glCompressedTexSubImage2D(12x12, offset (0, 24), height 6 = up to TEXTURE_HEIGHT 30): accepted");
        astc12_.blocks[4] = grad12;
        astc12_.blocks[2] = lum12;
        astc12_.blocks[6] = solid(255, 255, 255);
        sub(6, 0, 12, 12, f12, 16, block_bytes({solid(255, 0, 0)}));
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glCompressedTexSubImage2D(12x12, xoffset 6): INVALID_OPERATION (offset not a multiple of the block width)");
        sub(0, 0, 6, 12, f12, 16, block_bytes({solid(255, 0, 0)}));
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glCompressedTexSubImage2D(12x12, width 6 at xoffset 0): INVALID_OPERATION (not a multiple of 12 and 0 + 6 != TEXTURE_WIDTH)");
        sub(12, 0, 12, 12, f12, 15, block_bytes({solid(255, 0, 0)}));
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCompressedTexSubImage2D(12x12, one block, imageSize 15): INVALID_VALUE (too little data)");
        sub(12, 0, 18, 12, f12, 16, block_bytes({solid(255, 0, 0)}));
        check.expect_gl_error(mgl::ErrorCode::InvalidValue,
                              "glCompressedTexSubImage2D(12x12, offset (12, 0), width 18 up to the edge = 2 blocks, imageSize 16): INVALID_VALUE (imageSize from the block footprint is 32)");
        sub(0, 0, 12, 12, mgl::CompressedInternalFormat::RgbaAstc4x4, 144, std::vector<std::uint8_t>(144, 0));
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glCompressedTexSubImage2D(RGBA_ASTC_4x4 into an RGBA_ASTC_12x12 level): INVALID_OPERATION (format mismatch)");
        check_decode(check, "RGBA_ASTC_12x12, 30 x 30 after the sub-image updates (edge blocks of 6 texels, 4x4 / 3x3 grids infilled to 12 x 12)", astc12_tex_, astc12_, false);

        // ---- the 6x6 texture's failed CompressedTexImage2D (imageSize 80) is checked above; the successful upload is decoded in check_astc_decoding ----
        // ---- imageSize for every footprint: one CompressedTexImage2D per ASTC format with a non-multiple size, and with one byte less ----
        Tally t;
        for (const AstcFormat& f : kAstcFormats)
        {
            const int w = 2 * f.bw + 1, h = f.bh + 3;  // 3 x 2 blocks (the last ones partial)
            const int size = ((w + f.bw - 1) / f.bw) * ((h + f.bh - 1) / f.bh) * 16;
            std::vector<std::uint8_t> data;
            for (int i = 0; i < size / 16; ++i)
            {
                const Block b = solid(30 * i, 255 - 30 * i, 128);
                data.insert(data.end(), b.begin(), b.end());
            }
            const mgl::TextureId tex = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, f.linear, w, h, 0, size - 1, data.data());
            const mgl::ErrorCode short_error = demo::drain_gl_errors();
            mgl::glCompressedTexImage2D(mgl::TextureTarget::Texture2D, 0, f.linear, w, h, 0, size, data.data());
            const mgl::ErrorCode good_error = demo::drain_gl_errors();
            GLint got_w = 0, got_h = 0;
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2D, 0, mgl::TextureLevelParameter::Width, &got_w);
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2D, 0, mgl::TextureLevelParameter::Height, &got_h);
            t.add(short_error == mgl::ErrorCode::InvalidValue && good_error == mgl::ErrorCode::NoError && got_w == w && got_h == h, [&] {
                return std::format("{}x{} ({} x {} texels): imageSize {} -> {}, imageSize {} -> {}, level size {} x {}", f.bw, f.bh, w, h, size - 1, demo::error_name(short_error), size,
                                   demo::error_name(good_error), got_w, got_h);
            });
        }
        check.expect(t.bad == 0,
                     "every ASTC footprint: CompressedTexImage2D of a (2 bw + 1) x (bh + 3) image needs imageSize = ceil(w / bw) * ceil(h / bh) * 16 (one byte less: INVALID_VALUE)",
                     t.detail());
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
    }

    // ---- compressed <-> uncompressed and compressed <-> compressed copies ---------------------------
    void AstcAndCopyImage::check_compressed_copies(demo::Check& check)
    {
        const auto compare_blocks = [&](std::string_view what, mgl::TextureId tex, int w, int h, const std::function<Block(int, int)>& block_at) {
            const Fetched f = fetch(true, tex, w, h);
            Tally t;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const U4 want = block_words(block_at(x, y));
                    const U4& got = f.texels[static_cast<std::size_t>(y * w + x)];
                    t.add(got == want, [&] {
                        return std::format("texel ({}, {}): got {:08X} {:08X} {:08X} {:08X}, expected {:08X} {:08X} {:08X} {:08X}", x, y, got[0], got[1], got[2], got[3], want[0], want[1],
                                           want[2], want[3]);
                    });
                }
            check.expect(t.bad == 0, what, t.detail());
            return f;
        };

        // ASTC -> RGBA32UI: "a block of compressed data becomes a single texel of uncompressed data".
        const Fetched atlas_words = compare_blocks("glCopyImageSubData(ASTC 4x4 atlas, 16 x 16 texels -> RGBA32UI 4 x 4): every texel holds the 128 bits of its block",
                                                   atlas_u32_, 4, 4, [&](int x, int y) { return atlas_.blocks[static_cast<std::size_t>(y * 4 + x)]; });
        compare_blocks("glCopyImageSubData(mosaic ASTC 4x4, 32 x 32 -> RGBA32UI 8 x 8): the 64 copied blocks are bit-exact (ASTC -> ASTC and RGBA32UI -> ASTC copies)",
                       mosaic_u32_, 8, 8, [&](int x, int y) { return mosaic_img_.blocks[static_cast<std::size_t>(y * 8 + x)]; });
        int transposed_mismatch = 0;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                transposed_mismatch += atlas_words.texels[static_cast<std::size_t>(y * 4 + x)] == block_words(atlas_.blocks[static_cast<std::size_t>(x * 4 + y)]) ? 0 : 1;
        check.expect(transposed_mismatch == 12, "negative control: the RGBA32UI block texels do not match the atlas blocks in transposed order (off-diagonal)",
                     std::format("{} of 16 differ", transposed_mismatch));

        // The decoded results of the copies.
        check_decode(check, "mosaic RGBA_ASTC_4x4 32 x 32, assembled with 64 glCopyImageSubData calls", mosaic_tex_, mosaic_img_, false);
        check_decode(check, "glCopyImageSubData(RGBA_ASTC_4x4 -> SRGB8_ALPHA8_ASTC_4x4): the copied bits decode as sRGB", srgb_copy_tex_, atlas_, true);

        // 12x12 footprint -> RGBA32UI, including the partial blocks at the right and top edges: a region whose width or height is not a
        // multiple of the block size is aligned when it ends at the edge of the level (the rule of section 8.7 for CompressedTexSubImage*).
        const auto ic = mgl::ImageCopyTextureTarget::Texture2D;
        mgl::glCopyImageSubData(astc12_tex_, ic, 0, 0, 0, 0, astc12_u32_, ic, 0, 0, 0, 0, 24, 24, 1);
        check.expect_no_gl_error("glCopyImageSubData(ASTC 12x12 (0, 0) 24 x 24 -> RGBA32UI (0, 0)): accepted");
        mgl::glCopyImageSubData(astc12_tex_, ic, 0, 24, 0, 0, astc12_u32_, ic, 0, 2, 0, 0, 6, 24, 1);
        check.expect_no_gl_error("glCopyImageSubData(ASTC 12x12 (24, 0) 6 x 24 (right edge) -> RGBA32UI (2, 0)): accepted");
        mgl::glCopyImageSubData(astc12_tex_, ic, 0, 0, 24, 0, astc12_u32_, ic, 0, 0, 2, 0, 30, 6, 1);
        check.expect_no_gl_error("glCopyImageSubData(ASTC 12x12 (0, 24) 30 x 6 (top edge, whole width) -> RGBA32UI (0, 2)): accepted");
        compare_blocks("ASTC 12x12, 30 x 30 -> RGBA32UI 3 x 3: every block, including the partial edge blocks", astc12_u32_, 3, 3,
                       [&](int x, int y) { return astc12_.blocks[static_cast<std::size_t>(y * 3 + x)]; });

        // ETC2 / EAC (Table 16.3): RGBA8_ETC2_EAC (128-bit blocks) <-> RGBA32UI, RGB8_ETC2 (64-bit blocks) <-> RGBA16UI.
        {
            const Fetched f = fetch(true, etc2_rgba_u32_, 2, 1);
            bool ok = true;
            for (std::size_t i = 0; i < 2; ++i)
            {
                U4 want{};
                std::memcpy(want.data(), etc2_rgba_.data() + 16 * i, 16);
                ok = ok && f.texels[i] == want;
            }
            check.expect(ok, "glCopyImageSubData(RGBA8_ETC2_EAC 8 x 4 -> RGBA32UI 2 x 1): the two 128-bit blocks, bit-exact");
        }
        for (const auto& [tex, what] : {std::pair{etc2_rgb_u16_, "glCopyImageSubData(RGB8_ETC2 8 x 4 -> RGBA16UI 2 x 1): the two 64-bit blocks, bit-exact"},
                                        std::pair{etc2_rgb_u16_back_, "RGBA16UI -> RGB8_ETC2 -> RGBA16UI (both directions): the 64-bit blocks survive unchanged"}})
        {
            const Fetched f = fetch(true, tex, 2, 1);
            bool ok = true;
            std::string got;
            for (std::size_t i = 0; i < 2; ++i)
            {
                std::array<std::uint16_t, 4> want{};
                std::memcpy(want.data(), etc2_rgb_.data() + 8 * i, 8);
                for (std::size_t c = 0; c < 4; ++c) ok = ok && f.texels[i][c] == want[c];
                got += std::format(" ({:04X} {:04X} {:04X} {:04X} vs {:04X} {:04X} {:04X} {:04X})", f.texels[i][0], f.texels[i][1], f.texels[i][2], f.texels[i][3], want[0], want[1],
                                   want[2], want[3]);
            }
            check.expect(ok, what, got);
        }
    }

    // ---- RGBA8 texture / renderbuffer copies -------------------------------------------------------
    void AstcAndCopyImage::check_uncompressed_copies(demo::Check& check)
    {
        const auto compare = [&](std::string_view what, const std::vector<std::uint8_t>& got, int w, int h, const std::function<demo::Rgba8(int, int)>& want_at) {
            Tally t;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const std::size_t o = static_cast<std::size_t>((y * w + x) * 4);
                    const demo::Rgba8 g = {got[o], got[o + 1], got[o + 2], got[o + 3]};
                    const demo::Rgba8 want = want_at(x, y);
                    t.add(g == want, [&] { return std::format("({}, {}): got {} expected {}", x, y, demo::format_rgba(g), demo::format_rgba(want)); });
                }
            check.expect(t.bad == 0, what, t.detail());
        };
        const auto rb_at = [&](const std::vector<std::uint8_t>& px, int x, int y) {
            const std::size_t o = static_cast<std::size_t>((y * 8 + x) * 4);
            return demo::Rgba8{px[o], px[o + 1], px[o + 2], px[o + 3]};
        };
        const auto in = [](int x, int y, int x0, int y0, int w, int h) { return x >= x0 && x < x0 + w && y >= y0 && y < y0 + h; };

        const std::vector<std::uint8_t> dst = read_texture_rgba8(mgl::TextureTarget::Texture2D, rgba8_dst_, -1, 8, 8);
        compare("glCopyImageSubData(RGBA8 texture (1, 2) 5 x 3 -> texture (3, 4)): the region is copied texel for texel, the rest of the destination is unchanged", dst, 8, 8,
                [&](int x, int y) { return in(x, y, 3, 4, 5, 3) ? rgba_pattern(1, x - 3 + 1, y - 4 + 2) : rgba_pattern(2, x, y); });
        int changed = 0;
        for (int y = 4; y < 7; ++y)
            for (int x = 3; x < 8; ++x) changed += rgba_pattern(1, x - 2, y - 2) == rgba_pattern(2, x, y) ? 0 : 1;
        check.expect(changed == 15, "negative control: all 15 copied texels differ from the destination's original content", std::format("{} differ", changed));

        const std::vector<std::uint8_t> dst2 = read_texture_rgba8(mgl::TextureTarget::Texture2D, rgba8_dst2_, -1, 8, 8);
        compare("glCopyImageSubData(RGBA8 renderbuffer (1, 2) 6 x 5 -> texture (2, 1)): the renderbuffer's cleared rectangles appear in the texture", dst2, 8, 8,
                [&](int x, int y) { return in(x, y, 2, 1, 6, 5) ? rb_at(rb_before_, x - 2 + 1, y - 1 + 2) : rgba_pattern(3, x, y); });
        bool two_colours = rb_at(rb_before_, 0, 0) != rb_at(rb_before_, 3, 4);
        check.expect(two_colours, "the source renderbuffer holds two different colours (outer clear and scissored clear)",
                     std::format("{} / {}", demo::format_rgba(rb_at(rb_before_, 0, 0)), demo::format_rgba(rb_at(rb_before_, 3, 4))));

        const std::vector<std::uint8_t> rb = read_renderbuffer_rgba8(rb_, 8, 8);
        compare("glCopyImageSubData(RGBA8 texture (4, 4) 3 x 2 -> renderbuffer (0, 0)): read back from the renderbuffer", rb, 8, 8,
                [&](int x, int y) { return in(x, y, 0, 0, 3, 2) ? rgba_pattern(1, x + 4, y + 4) : rb_at(rb_before_, x, y); });
    }

    void AstcAndCopyImage::check_layers_and_faces(demo::Check& check)
    {
        const auto compare = [&](std::string_view what, const std::vector<std::uint8_t>& got, const std::function<demo::Rgba8(int, int)>& want_at) {
            Tally t;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                {
                    const std::size_t o = static_cast<std::size_t>((y * 4 + x) * 4);
                    const demo::Rgba8 g = {got[o], got[o + 1], got[o + 2], got[o + 3]};
                    const demo::Rgba8 want = want_at(x, y);
                    t.add(g == want, [&] { return std::format("({}, {}): got {} expected {}", x, y, demo::format_rgba(g), demo::format_rgba(want)); });
                }
            check.expect(t.bad == 0, what, t.detail());
        };
        for (int l = 0; l < 4; ++l)
        {
            const std::vector<std::uint8_t> px = read_texture_rgba8(mgl::TextureTarget::Texture2DArray, arr_dst_, l, 4, 4);
            const int seed = l < 2 ? 40 + l : 10 + (l - 1);
            compare(std::format("2D array: glCopyImageSubData(srcZ 1, dstZ 2, srcDepth 2): destination layer {} {}", l,
                                l < 2 ? "is unchanged" : std::format("holds source layer {}", l - 1)),
                    px, [&](int x, int y) { return rgba_pattern(seed, x, y); });
        }
        constexpr std::array<mgl::TextureTarget, 6> faces = {mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
                                                             mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
                                                             mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
        constexpr std::array<std::string_view, 6> face_names = {"POSITIVE_X", "NEGATIVE_X", "POSITIVE_Y", "NEGATIVE_Y", "POSITIVE_Z", "NEGATIVE_Z"};
        for (std::size_t f = 0; f < 6; ++f)
        {
            const std::vector<std::uint8_t> px = read_texture_rgba8(faces[f], cube_, -1, 4, 4);
            compare(std::format("2D array layers 0..5 -> cube map (srcDepth 6): face {} ({}) holds array layer {} (faces in the order of Table 8.20)", f, face_names[f], f), px,
                    [&](int x, int y) { return rgba_pattern(30 + static_cast<int>(f), x, y); });
        }
        const std::vector<std::uint8_t> out = read_texture_rgba8(mgl::TextureTarget::Texture2D, face_out_, -1, 4, 4);
        compare("glCopyImageSubData(cube map, srcZ 4 = POSITIVE_Z, (1, 1) 2 x 2 -> 2D texture (2, 0))", out,
                [&](int x, int y) { return x >= 2 && y < 2 ? rgba_pattern(34, x - 2 + 1, y + 1) : rgba_pattern(60, x, y); });
    }

    // ---- RGBA8UI <-> R32F --------------------------------------------------------------------------
    void AstcAndCopyImage::check_reinterpretation(demo::Check& check)
    {
        // Section 16.2.2: "If the internal format does not exactly match the internal format of the original texture, the contents of the memory
        // are reinterpreted in the same manner as for image bindings described in section 8.23", i.e. the RGBA8UI texel read as RGBA_INTEGER /
        // UNSIGNED_BYTE into memory and written from that memory as RED / FLOAT: the float is the four bytes R, G, B, A in host memory order.
        const Fetched f = fetch(false, f32_dst_, 4, 4);
        Tally t, swapped;
        for (std::size_t i = 0; i < 16; ++i)
        {
            std::uint32_t want = 0;
            std::memcpy(&want, u8_data_.data() + 4 * i, 4);
            const U4& got = f.texels[i];
            t.add(got[0] == want && got[1] == 0u && got[2] == 0u && got[3] == std::bit_cast<std::uint32_t>(1.0f),
                  [&] { return std::format("texel {}: got {:08X} (+ {:08X} {:08X} {:08X}), expected {:08X} (0, 0, 1)", i, got[0], got[1], got[2], got[3], want); });
            swapped.add(got[0] == std::byteswap(want), [] { return std::string{}; });
        }
        check.expect(t.bad == 0, "glCopyImageSubData(RGBA8UI -> R32F): each float is the memory of the four bytes, bit-exact (texelFetch returns (f, 0, 0, 1))", t.detail());
        check.expect(swapped.bad > 0, "negative control: the floats do not match the byte-reversed words", std::format("{} of 16 differ", swapped.bad));

        const Fetched u = fetch(true, u8_dst_, 4, 4);
        Tally t2;
        for (std::size_t i = 0; i < 16; ++i)
        {
            std::array<std::uint8_t, 4> want{};
            std::memcpy(want.data(), &f32_data_[i], 4);
            const U4& got = u.texels[i];
            t2.add(got[0] == want[0] && got[1] == want[1] && got[2] == want[2] && got[3] == want[3], [&] {
                return std::format("texel {} ({}): got ({}, {}, {}, {}), expected ({}, {}, {}, {})", i, f32_data_[i], got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
            });
        }
        check.expect(t2.bad == 0, "glCopyImageSubData(R32F -> RGBA8UI): the four bytes of every float, bit-exact", t2.detail());
    }

    // ---- depth and depth-stencil -------------------------------------------------------------------
    void AstcAndCopyImage::check_depth_stencil(demo::Check& check)
    {
        const auto in = [](int x, int y, int x0, int y0, int w, int h) { return x >= x0 && x < x0 + w && y >= y0 && y < y0 + h; };
        {
            const Fetched f = fetch(false, d32_dst_, 4, 4);
            Tally t;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                {
                    const float want = in(x, y, 1, 1, 3, 2) ? d32_src_data_[static_cast<std::size_t>((y - 1) * 4 + (x - 1))] : 0.75f;
                    const float got = std::bit_cast<float>(f.texels[static_cast<std::size_t>(y * 4 + x)][0]);
                    t.add(got == want, [&] { return std::format("({}, {}): got {} expected {}", x, y, got, want); });
                }
            check.expect(t.bad == 0, "glCopyImageSubData(DEPTH_COMPONENT32F (0, 0) 3 x 2 -> DEPTH_COMPONENT32F (1, 1)): depth values bit-exact (texelFetch, no comparison)", t.detail());
        }
        // DEPTH24_STENCIL8: depth through DEPTH_STENCIL_TEXTURE_MODE = DEPTH_COMPONENT, stencil through STENCIL_INDEX and a usampler.
        const auto set_mode = [](mgl::TextureId tex, GLenum mode) {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::DepthStencilTextureMode, static_cast<GLint>(mode));
        };
        const Fetched src_depth = fetch(false, ds_src_, 4, 4);
        const Fetched dst_depth = fetch(false, ds_dst_, 4, 4);
        set_mode(ds_src_, GL_STENCIL_INDEX);
        set_mode(ds_dst_, GL_STENCIL_INDEX);
        const Fetched src_stencil = fetch(true, ds_src_, 4, 4);
        const Fetched dst_stencil = fetch(true, ds_dst_, 4, 4);
        set_mode(ds_src_, GL_DEPTH_COMPONENT);
        set_mode(ds_dst_, GL_DEPTH_COMPONENT);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        Tally depth, stencil;
        double worst = 0.0;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                const auto i = static_cast<std::size_t>(y * 4 + x);
                const bool copied = in(x, y, 0, 2, 3, 2);
                const auto si = static_cast<std::size_t>((y - 2) * 4 + (x + 1));
                const std::uint32_t uploaded = copied ? ds_src_data_[si] : ds_dst_data_[i];
                const double want_depth = static_cast<double>(uploaded >> 8) / 16777215.0;
                const float got_depth = std::bit_cast<float>(dst_depth.texels[i][0]);
                worst = std::max(worst, std::abs(static_cast<double>(got_depth) - want_depth));
                // The copy is a memcpy: a copied texel samples exactly like its source texel; every texel is the uploaded 24-bit depth
                // (no precision is specified for the fixed-point -> float conversion of a texture fetch: sanity bound 2^-20, measured value printed).
                const bool same_as_source = !copied || dst_depth.texels[i][0] == src_depth.texels[si][0];
                depth.add(same_as_source && std::abs(static_cast<double>(got_depth) - want_depth) <= std::ldexp(1.0, -20),
                          [&] { return std::format("({}, {}): depth {} expected {}{}", x, y, got_depth, want_depth, same_as_source ? "" : " (differs from the source texel)"); });
                const std::uint32_t want_stencil = uploaded & 0xFFu;
                stencil.add(dst_stencil.texels[i][0] == want_stencil && (!copied || src_stencil.texels[si][0] == want_stencil),
                            [&] { return std::format("({}, {}): stencil {} expected {}", x, y, dst_stencil.texels[i][0], want_stencil); });
            }
        check.expect(depth.bad == 0, "glCopyImageSubData(DEPTH24_STENCIL8 (1, 0) 3 x 2 -> (0, 2)): depth copied, the rest unchanged",
                     std::format("{}; max |depth - D / (2^24 - 1)| = {:.3g}", depth.detail(), worst));
        check.expect(stencil.bad == 0, "glCopyImageSubData(DEPTH24_STENCIL8): the stencil bits travel with the depth (read with DEPTH_STENCIL_TEXTURE_MODE = STENCIL_INDEX)",
                     stencil.detail());
    }

    // ---- completeness ------------------------------------------------------------------------------
    void AstcAndCopyImage::check_completeness(demo::Check& check)
    {
        const auto ic = mgl::ImageCopyTextureTarget::Texture2D;
        // "An INVALID_OPERATION error is generated if either object is a texture and the texture is not complete (as defined in section 8.17,
        // but ignoring format-based completeness rules)". A mutable texture with only level 0 and the default MIN_FILTER
        // NEAREST_MIPMAP_LINEAR is not mipmap complete, hence not complete.
        mgl::glCopyImageSubData(incomplete_, ic, 0, 0, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, 4, 4, 1);
        const mgl::ErrorCode e_src = demo::drain_gl_errors();
        check.expect(e_src == mgl::ErrorCode::InvalidOperation,
                     "glCopyImageSubData from a mipmap-incomplete texture (level 0 only, MIN_FILTER NEAREST_MIPMAP_LINEAR): INVALID_OPERATION",
                     std::format("glGetError() = {}", demo::error_name(e_src)));
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 0, 0, 0, incomplete_dst_, ic, 0, 0, 0, 0, 4, 4, 1);
        const mgl::ErrorCode e_dst = demo::drain_gl_errors();
        check.expect(e_dst == mgl::ErrorCode::InvalidOperation, "glCopyImageSubData into a mipmap-incomplete texture: INVALID_OPERATION",
                     std::format("glGetError() = {}", demo::error_name(e_dst)));
        // Positive control: with MIN_FILTER NEAREST the same texture is complete and the copy succeeds.
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, incomplete_);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
        const mgl::TextureId fresh = new_texture();
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, fresh);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, 4, 4);
        mgl::glCopyImageSubData(incomplete_, ic, 0, 0, 0, 0, fresh, ic, 0, 0, 0, 0, 4, 4, 1);
        check.expect_no_gl_error("the same texture with MIN_FILTER NEAREST (complete): glCopyImageSubData accepted");
        const std::vector<std::uint8_t> px = read_texture_rgba8(mgl::TextureTarget::Texture2D, fresh, -1, 4, 4);
        bool ok = true;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                const std::size_t o = static_cast<std::size_t>((y * 4 + x) * 4);
                ok = ok && demo::Rgba8{px[o], px[o + 1], px[o + 2], px[o + 3]} == rgba_pattern(70, x, y);
            }
        check.expect(ok, "... and the copied texels are those of the mutable texture's level 0");
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
    }

    // ---- errors ------------------------------------------------------------------------------------
    void AstcAndCopyImage::check_copy_errors(demo::Check& check)
    {
        const auto ic = mgl::ImageCopyTextureTarget::Texture2D;
        const auto tt = mgl::TextureTarget::Texture2D;
        const std::vector<std::uint8_t> before = read_texture_rgba8(mgl::TextureTarget::Texture2D, rgba8_dst_, -1, 8, 8);

        // INVALID_ENUM: "if either target is not RENDERBUFFER or a valid texture target; is TEXTURE_BUFFER or one of the cubemap face selectors"
        mgl::glCopyImageSubData(rgba8_src_.value, mgl::TextureTarget::TextureBuffer, 0, 0, 0, 0, rgba8_dst_.value, tt, 0, 0, 0, 0, 1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glCopyImageSubData(srcTarget TEXTURE_BUFFER): INVALID_ENUM");
        mgl::glCopyImageSubData(cube_.value, mgl::TextureTarget::TextureCubeMapPositiveX, 0, 0, 0, 0, rgba8_dst_.value, tt, 0, 0, 0, 0, 1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glCopyImageSubData(srcTarget TEXTURE_CUBE_MAP_POSITIVE_X, a cube face selector): INVALID_ENUM");
        // INVALID_VALUE: "if either name does not correspond to a valid renderbuffer or texture object according to the corresponding target parameter"
        // (generated and deleted right here: a GL may hand a deleted name out again from the next glGenTextures)
        mgl::TextureId deleted{};
        mgl::glGenTextures(1, &deleted);
        mgl::glDeleteTextures(1, &deleted);
        mgl::glCopyImageSubData(deleted, ic, 0, 0, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, 1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(srcName = a deleted texture name): INVALID_VALUE");
        // INVALID_VALUE: "if srcLevel and dstLevel are not valid levels for the corresponding images"
        mgl::glCopyImageSubData(rgba8_src_, ic, 1, 0, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, 1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(srcLevel 1 of a one-level texture): INVALID_VALUE");
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 0, 0, 0, rgba8_dst_, ic, 1, 0, 0, 0, 1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(dstLevel 1 of a one-level texture): INVALID_VALUE");
        // INVALID_VALUE: "if srcWidth, srcHeight, or srcDepth is negative"
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 0, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, -1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(srcWidth -1): INVALID_VALUE");
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 0, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, 1, 1, -1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(srcDepth -1): INVALID_VALUE");
        // INVALID_VALUE: "if the dimensions of either subregion exceeds the boundaries of the corresponding image object"
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 6, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, 4, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(source x 6 + width 4 > 8): INVALID_VALUE");
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 0, 0, 0, rgba8_dst_, ic, 0, 0, 7, 0, 1, 2, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(destination y 7 + height 2 > 8): INVALID_VALUE");
        mgl::glCopyImageSubData(rgba8_src_, ic, 0, 0, 0, 0, rgba8_dst_, ic, 0, 0, 0, 1, 1, 1, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(2D destination z 1): INVALID_VALUE");
        mgl::glCopyImageSubData(arr_src_, mgl::ImageCopyTextureTarget::Texture2DArray, 0, 0, 0, 3, arr_dst_, mgl::ImageCopyTextureTarget::Texture2DArray, 0, 0, 0, 0, 4, 4, 2);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(2D array srcZ 3 + srcDepth 2 > 4 layers): INVALID_VALUE");
        mgl::glCopyImageSubData(arr6_, mgl::ImageCopyTextureTarget::Texture2DArray, 0, 0, 0, 0, cube_, mgl::ImageCopyTextureTarget::TextureCubeMap, 0, 0, 0, 5, 4, 4, 2);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(cube map dstZ 5 + depth 2 > 6 faces): INVALID_VALUE");
        // INVALID_VALUE: "if the image format is compressed and the dimensions of the subregion fail to meet the alignment constraints of the format"
        mgl::glCopyImageSubData(atlas_tex_, ic, 0, 2, 0, 0, mosaic_tex_, ic, 0, 0, 0, 0, 4, 4, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(ASTC 4x4 source x 2, not on a block boundary): INVALID_VALUE");
        mgl::glCopyImageSubData(atlas_tex_, ic, 0, 0, 0, 0, mosaic_tex_, ic, 0, 0, 0, 0, 6, 4, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(ASTC 4x4 width 6, neither a multiple of 4 nor up to the edge): INVALID_VALUE");
        mgl::glCopyImageSubData(atlas_tex_, ic, 0, 0, 0, 0, mosaic_tex_, ic, 0, 0, 2, 0, 4, 4, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCopyImageSubData(ASTC 4x4 destination y 2, not on a block boundary): INVALID_VALUE");
        // INVALID_OPERATION: "if the source and destination internal formats are not compatible" / "if the texel size of the uncompressed image
        // is not equal to the block size of the compressed image"
        struct Incompatible
        {
            mgl::TextureId src;
            int w, h;
            mgl::TextureId dst;
            const char* what;
        };
        const std::array<Incompatible, 10> incompatible = {{
            {rgba8_src_, 1, 1, rg8_, "RGBA8 -> RG8 (32-bit vs 16-bit class)"},
            {rgba8_src_, 1, 1, rgba16f_, "RGBA8 -> RGBA16F (VIEW_CLASS_32_BITS vs VIEW_CLASS_64_BITS)"},
            {astc12_tex_, 12, 12, mosaic_tex_, "RGBA_ASTC_12x12 -> RGBA_ASTC_4x4, a 12 x 12 region aligned in both (different ASTC view classes)"},
            {atlas_tex_, 4, 4, etc2_rgba_tex_, "RGBA_ASTC_4x4 -> RGBA8_ETC2_EAC (both 128-bit 4x4 blocks, but neither the same format nor the same class)"},
            {atlas_tex_, 4, 4, rgba16ui_, "RGBA_ASTC_4x4 -> RGBA16UI (64-bit texel vs 128-bit block)"},
            {etc2_rgb_tex_, 4, 4, atlas_u32_, "RGB8_ETC2 -> RGBA32UI (64-bit block vs 128-bit texel)"},
            {d32_src_, 1, 1, f32_dst_, "DEPTH_COMPONENT32F -> R32F (depth formats are in no class of Table 16.4)"},
            {ds_src_, 1, 1, u8_dst_, "DEPTH24_STENCIL8 -> RGBA8UI (32 bits each, but no common class)"},
            {d32_src_, 1, 1, ds_dst_, "DEPTH_COMPONENT32F -> DEPTH24_STENCIL8 (different depth formats)"},
            {rgba8_src_, 1, 1, atlas_tex_, "RGBA8 -> RGBA_ASTC_4x4 (32-bit texel vs 128-bit block)"},
        }};
        for (const Incompatible& c : incompatible)
        {
            mgl::glCopyImageSubData(c.src, ic, 0, 0, 0, 0, c.dst, ic, 0, 0, 0, 0, c.w, c.h, 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, std::format("glCopyImageSubData({}): INVALID_OPERATION", c.what));
        }
        // INVALID_OPERATION: "if the number of samples do not match"
        mgl::glCopyImageSubData(ms_rb_, 0, 0, rgba8_dst_, ic, 0, 0, 0, 0, 2, 2);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glCopyImageSubData(4-sample RGBA8 renderbuffer -> single-sample RGBA8 texture): INVALID_OPERATION");

        // A command that generates an error has no other effect (ES 3.2 section 2.3.1): the destination is unchanged.
        const std::vector<std::uint8_t> after = read_texture_rgba8(mgl::TextureTarget::Texture2D, rgba8_dst_, -1, 8, 8);
        check.expect(after == before, "the rejected copies into the RGBA8 texture left it unchanged");
        const Fetched mosaic_words = fetch(true, mosaic_u32_, 8, 8);  // the mosaic itself is re-copied and compared to make sure nothing touched it
        mgl::glCopyImageSubData(mosaic_tex_, ic, 0, 0, 0, 0, mosaic_u32_, ic, 0, 0, 0, 0, 32, 32, 1);
        const Fetched again = fetch(true, mosaic_u32_, 8, 8);
        check.expect_no_gl_error("re-copy of the mosaic into the RGBA32UI texture");
        check.expect(again.texels == mosaic_words.texels, "the rejected copies into the ASTC mosaic left it unchanged");
    }

    // ---- the picture -------------------------------------------------------------------------------
    void AstcAndCopyImage::check_picture(demo::Check& check, int width, int height)
    {
        const Layout lay(width, height);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, width, height);
        const auto at = [&](int x, int y) {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4;
            return demo::Rgba8{px[o], px[o + 1], px[o + 2], px[o + 3]};
        };
        // Expected texel colours (the CPU decoder; 8-bit values every decode mode agrees on).
        const auto colours = [](const AstcImage& img) {
            std::vector<demo::Rgba8> c;
            for (int y = 0; y < img.h; ++y)
                for (int x = 0; x < img.w; ++x)
                {
                    const auto v = top8(img.decode(x, y, false));
                    c.push_back({v[0], v[1], v[2], 255});
                }
            return c;
        };
        const std::vector<demo::Rgba8> atlas_c = colours(atlas_), mosaic_c = colours(mosaic_img_);
        AstcImage wrong = mosaic_img_;  // negative control: the mosaic with its first two blocks exchanged
        std::swap(wrong.blocks[0], wrong.blocks[1]);
        const std::vector<demo::Rgba8> wrong_c = colours(wrong);

        // A pixel centre maps to the texel coordinate t = (x + 0.5 - origin) * texels / size (panel.frag). Where t lies within 0.002 texel of
        // an integer the float evaluation may land on either side, so both neighbouring texels are accepted there; elsewhere exactly one is.
        const auto candidates = [](int p, int origin, int size, int n) {
            const double t = (static_cast<double>(p - origin) + 0.5) * n / size;
            const int i = std::clamp(static_cast<int>(std::floor(t)), 0, n - 1);
            const double f = t - std::floor(t);
            std::vector<int> c = {i};
            if (f < 0.002 && i > 0) c.push_back(i - 1);
            if (f > 0.998 && i < n - 1) c.push_back(i + 1);
            return c;
        };
        const auto close = [](const demo::Rgba8& a, const demo::Rgba8& b) {
            // +-1: a decoded value need not be exactly v / 255 (decode_float16 rounds toward zero), and the conversion to the 8-bit
            // framebuffer returns one of the two closest values (ES 3.2 section 2.3.5.2).
            for (std::size_t c = 0; c < 4; ++c)
                if (std::abs(static_cast<int>(a[c]) - static_cast<int>(b[c])) > 1) return false;
            return true;
        };
        Tally t;
        std::vector<bool> seen_atlas(256, false), seen_mosaic(1024, false);
        int ambiguous = 0, wrong_hits = 0;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                const demo::Rgba8 got = at(x, y);
                const Rect* r = lay.atlas.contains(x, y) ? &lay.atlas : lay.mosaic.contains(x, y) ? &lay.mosaic : nullptr;
                if (r == nullptr)
                {
                    t.add(close(got, kClear), [&] { return std::format("({}, {}) outside the panels: got {} expected the clear colour", x, y, demo::format_rgba(got)); });
                    continue;
                }
                const bool is_atlas = r == &lay.atlas;
                const int n = is_atlas ? 16 : 32;
                const std::vector<demo::Rgba8>& c = is_atlas ? atlas_c : mosaic_c;
                const std::vector<int> ci = candidates(x, r->x, r->w, n), cj = candidates(y, r->y, r->h, n);
                if (ci.size() * cj.size() > 1) ++ambiguous;
                bool ok = false;
                for (const int i : ci)
                    for (const int j : cj) ok = ok || close(got, c[static_cast<std::size_t>(j * n + i)]);
                const int i0 = ci.front(), j0 = cj.front();
                (is_atlas ? seen_atlas : seen_mosaic)[static_cast<std::size_t>(j0 * n + i0)] = true;
                if (!is_atlas && ci.size() * cj.size() == 1 && !close(got, wrong_c[static_cast<std::size_t>(j0 * n + i0)])) ++wrong_hits;
                t.add(ok, [&] {
                    return std::format("({}, {}) {} texel ({}, {}): got {} expected {}", x, y, is_atlas ? "atlas" : "mosaic", i0, j0, demo::format_rgba(got),
                                       demo::format_rgba(c[static_cast<std::size_t>(j0 * n + i0)]));
                });
            }
        check.expect(t.bad == 0,
                     std::format("picture: every pixel is the clear colour or the decoded texel of its panel (atlas {}x{} px, mosaic {}x{} px; {} pixels near a texel edge "
                                 "accept either neighbour)",
                                 lay.atlas.w, lay.atlas.h, lay.mosaic.w, lay.mosaic.h, ambiguous),
                     t.detail());
        const auto count = [](const std::vector<bool>& v) { return std::count(v.begin(), v.end(), true); };
        check.expect(count(seen_atlas) == 256 && count(seen_mosaic) == 1024, "picture: every texel of the atlas and of the mosaic is on screen",
                     std::format("atlas {} / 256, mosaic {} / 1024", count(seen_atlas), count(seen_mosaic)));
        check.expect(wrong_hits > 0, "negative control: the picture check rejects a mosaic with two blocks exchanged",
                     std::format("{} pixels would mismatch", wrong_hits));
    }

    void AstcAndCopyImage::shutdown(demo::Env& /*env*/)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glUseProgram(mgl::ProgramId{0});
        if (!textures_.empty()) mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
        if (!renderbuffers_.empty()) mgl::glDeleteRenderbuffers(static_cast<GLsizei>(renderbuffers_.size()), renderbuffers_.data());
        textures_.clear();
        renderbuffers_.clear();
        mgl::glDeleteFramebuffers(1, &fbo_);
        mgl::glDeleteBuffers(1, &ssbo_);
        for (const mgl::ProgramId p : {fetch_f_, fetch_u_, panel_}) mgl::glDeleteProgram(p);
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-astc-and-copy-image",
        .title = "ASTC textures and glCopyImageSubData",
        .min_es = {3, 2},
        .summary = "Hand-encoded ASTC LDR blocks (void extent, luminance, luminance + alpha, RGB with blue contraction, RGBA, weight infill, error blocks) decoded texel by "
                   "texel against a CPU decoder in 4x4 / 5x4 / 6x6 / 12x12 footprints and sRGB, with the CompressedTexSubImage alignment and imageSize rules; "
                   "glCopyImageSubData between textures, renderbuffers, array layers, cube faces, compatible formats, compressed <-> uncompressed images and depth formats, "
                   "and its errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<AstcAndCopyImage>(argc, argv, info);
}
