// es30-integer-float-textures
//
// Integer and floating-point textures (OpenGL ES 3.0, section 3.8): what a shader gets back from
// a texture whose texels are not normalized 8-bit colors.
//
//   * Integer textures (R8UI, R16I, RGBA8I, RGBA16UI, R32I, RGBA32UI, RGB10_A2UI) are read with
//     usampler2D / isampler2D and return the stored integers bit for bit, including values that no
//     32-bit float can hold (2^24 + 1, INT_MIN, 0xFFFFFFFF). They are not texture-filterable
//     (Table 3.13), so they are complete only with NEAREST (or NEAREST_MIPMAP_NEAREST) filters
//     (section 3.8.13, 3.8.13.1).
//   * Floating-point textures (R32F, RGBA32F, RGBA16F, R16F) return the stored float exactly; the
//     16-bit formats are filterable, the 32-bit ones are not (unless an extension says so).
//   * Packed floats R11F_G11F_B10F (section 2.1.3/2.1.4) and shared-exponent RGB9_E5 (sections
//     3.8.3.2 and 3.8.17) are decoded exactly as the specification states.
//
// How the values are made observable without rounding: a fragment shader samples a texture and
// writes the result unchanged into an RGBA32UI or RGBA32I render target (floats through
// floatBitsToUint); glReadPixels with RGBA_INTEGER returns the 32-bit words. Nothing in that path
// converts to a normalized 8-bit value.
//
// Image (256 x 256 window): a 4 x 4 contact sheet. Every cell is one 8 x 8 texture, one texel per
// 8 x 8 pixel block (row 0 of a texture is at the bottom of its cell), mapped to a color by
// color = clamp(value * scale + bias, 0, 1) (display only; the exact checks do not use it).
//
//   row 0 (top):  R8UI gray ramp | R16I ramp over the signed 16-bit range | RGBA8I | RGBA16UI
//   row 1:        R32I = 2^24 + k (left half ramp, right half green/red stripes: float(value)
//                 keeps even values, loses odd ones) | R32I over the whole int range | RGBA32UI | RGB10_A2UI
//   row 2:        R32F | RGBA32F | RGBA16F | R16F uploaded as FLOAT (converted by the GL)
//   row 3:        R11F_G11F_B10F | RGB9_E5 raw words | RGB9_E5 uploaded as FLOAT (encoded by the GL) |
//                 the R32F texture again, sampled with a LINEAR sampler object: a picture of the
//                 R32F gradient when the implementation makes 32-bit floats filterable
//                 (GL_OES_texture_float_linear), black = (0, 0, 0, 1) of an incomplete texture otherwise
//
// Validation: every texture is read back through the exact path with texture() and with
// texelFetch() and compared word for word with the CPU model; completeness through sampler objects
// and mip chains, filterability of the 16-bit and packed formats, GenerateMipmap and format-combination
// errors, the one-unit-two-sampler-types error, and the display cells are all checked.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kTex = 8;       // every texture is 8 x 8
    constexpr int kTexels = kTex * kTex;
    constexpr int kGrid = 4;      // 4 x 4 cells partition the window
    constexpr int kMinCell = 16;  // a cell needs 2 pixels per texel so that both halves of the "above 2^24" blocks exist

    struct Rect
    {
        int x, y, w, h;
    };

    /// Cell `cell` of the contact sheet (cell 0 is the top left). The edges sit at i * width / 4 and
    /// j * height / 4, so the cells partition the window exactly (they may differ by one pixel).
    Rect cell_rect(int cell, int width, int height)
    {
        const int col = cell % kGrid, row = kGrid - 1 - cell / kGrid;  // row 0 is the bottom row of the window
        const int x0 = col * width / kGrid, x1 = (col + 1) * width / kGrid;
        const int y0 = row * height / kGrid, y1 = (row + 1) * height / kGrid;
        return {x0, y0, x1 - x0, y1 - y0};
    }

    /// The pixels of a cell n pixels long that show part `i` of `parts` (pixel p shows part p * parts / n, the
    /// integer division in the show_*.frag shaders): [first, last]. Returns their centre.
    int block_centre(int i, int parts, int n)
    {
        const int first = (i * n + parts - 1) / parts;            // smallest p with p * parts / n >= i
        const int last = ((i + 1) * n + parts - 1) / parts - 1;   // largest p with p * parts / n <= i
        return (first + last) / 2;
    }
    constexpr std::uint32_t kOneBits = 0x3F800000u;  // 1.0f

    using Words = std::array<std::uint32_t, 4>;
    enum class Kind : std::size_t { Uint = 0, Int = 1, Float = 2 };

    float as_float(std::uint32_t b) { return std::bit_cast<float>(b); }
    std::uint32_t bits_of(float f) { return std::bit_cast<std::uint32_t>(f); }
    std::uint32_t word_of(std::int32_t v) { return std::bit_cast<std::uint32_t>(v); }
    std::int32_t signed_of(std::uint32_t w) { return std::bit_cast<std::int32_t>(w); }

    std::uint32_t hash32(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    // ---------------------------------------------------------------------------------------------
    // CPU codecs of the special float formats. Everything is computed in double, which represents every
    // value of these formats exactly.
    // ---------------------------------------------------------------------------------------------

    /// Value of a 16-bit half float (ES 3.0 section 2.1.2): sign, 5-bit exponent (bias 15), 10-bit mantissa.
    float half_to_float(std::uint16_t h)
    {
        const int e = (h >> 10) & 31;
        const int m = h & 1023;
        const float sign = (h & 0x8000) ? -1.0f : 1.0f;
        if (e == 0) return sign * std::ldexp(static_cast<float>(m), -24);
        return sign * std::ldexp(static_cast<float>(1024 + m), e - 25);
    }

    /// Nearest half float (ties to even) of a float whose magnitude is a normal half value.
    std::uint16_t float_to_half(float f)
    {
        const std::uint32_t sign = std::signbit(f) ? 0x8000u : 0u;
        const double v = std::fabs(static_cast<double>(f));
        if (v == 0.0) return static_cast<std::uint16_t>(sign);
        int e = 0;
        std::frexp(v, &e);
        e -= 1;  // v = 1.xxx * 2^e
        double m = std::nearbyint((v / std::ldexp(1.0, e) - 1.0) * 1024.0);
        if (m >= 1024.0) { m = 0.0; ++e; }
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e + 15) << 10) | static_cast<std::uint32_t>(m));
    }

    /// Unsigned 11-bit (mant_bits 6) and 10-bit (mant_bits 5) floats, ES 3.0 sections 2.1.3 and 2.1.4.
    double decode_small_float(std::uint32_t n, int mant_bits)
    {
        const std::uint32_t e = n >> mant_bits;
        const std::uint32_t m = n & ((1u << mant_bits) - 1u);
        if (e == 0) return std::ldexp(static_cast<double>(m), -14 - mant_bits);
        return std::ldexp(static_cast<double>((1u << mant_bits) + m), static_cast<int>(e) - 15 - mant_bits);
    }

    /// The nearest small float for a value >= 2^-14 (all values used here are exactly representable).
    std::uint32_t encode_small_float(double v, int mant_bits)
    {
        if (v == 0.0) return 0u;
        int e = 0;
        std::frexp(v, &e);
        e -= 1;
        double m = std::nearbyint((v / std::ldexp(1.0, e) - 1.0) * static_cast<double>(1 << mant_bits));
        if (m >= static_cast<double>(1 << mant_bits)) { m = 0.0; ++e; }
        return (static_cast<std::uint32_t>(e + 15) << mant_bits) | static_cast<std::uint32_t>(m);
    }

    /// RGB9_E5 (ES 3.0 sections 3.8.3.2 and 3.8.17): three 9-bit mantissas and one 5-bit shared exponent.
    struct Rgb9E5
    {
        std::uint32_t r = 0, g = 0, b = 0, e = 0;
        std::uint32_t pack() const { return r | (g << 9) | (b << 18) | (e << 27); }
        /// red = reds * 2^(exps - B - N) with B = 15, N = 9.
        std::array<float, 3> decode() const
        {
            return {std::ldexp(static_cast<float>(r), static_cast<int>(e) - 24), std::ldexp(static_cast<float>(g), static_cast<int>(e) - 24),
                    std::ldexp(static_cast<float>(b), static_cast<int>(e) - 24)};
        }
    };

    /// The encoding procedure of section 3.8.3.2, step by step.
    Rgb9E5 encode_rgb9e5(float red, float green, float blue)
    {
        constexpr double kSharedExpMax = (511.0 / 512.0) * 65536.0;  // (2^N - 1)/2^N * 2^(Emax - B)
        auto clamp_component = [&](float x) {
            const double d = static_cast<double>(x);
            return d > 0.0 ? std::min(d, kSharedExpMax) : 0.0;  // NaN and negative values become 0
        };
        const double rc = clamp_component(red), gc = clamp_component(green), bc = clamp_component(blue);
        const double maxc = std::max({rc, gc, bc});
        const int floor_log2 = maxc > 0.0 ? std::ilogb(maxc) : -1000;
        const int exp_p = std::max(-16, floor_log2) + 1 + 15;
        const double max_s = std::floor(std::ldexp(maxc, -(exp_p - 24)) + 0.5);
        const int exp_s = max_s == 512.0 ? exp_p + 1 : exp_p;
        auto mantissa = [&](double c) { return static_cast<std::uint32_t>(std::floor(std::ldexp(c, -(exp_s - 24)) + 0.5)); };
        return {mantissa(rc), mantissa(gc), mantissa(bc), static_cast<std::uint32_t>(exp_s)};
    }

    // ---------------------------------------------------------------------------------------------
    // The textures. Every texture holds 8 x 8 texels; texel (i, j) has index k = j * 8 + i, j counts from
    // the bottom. The values are a gradient (so that the picture means something) with the low-order bits
    // replaced by a hash (so that the exact comparison covers full-entropy values), plus a few extreme
    // texels.
    // ---------------------------------------------------------------------------------------------
    struct Display
    {
        std::array<float, 3> scale{1.0f, 1.0f, 1.0f};
        std::array<float, 3> bias{0.0f, 0.0f, 0.0f};
        bool gray = false;   // show the first component as gray
        int mode = 0;        // 1: the "above 2^24" cell
    };

    struct Case
    {
        const char* name = "";
        Kind kind = Kind::Uint;
        mgl::InternalFormat internal = mgl::InternalFormat::R8UI;
        mgl::PixelFormat format = mgl::PixelFormat::RedInteger;
        mgl::PixelType type = mgl::PixelType::UnsignedByte;
        std::vector<std::uint8_t> data;       // the bytes handed to glTexImage2D
        std::array<Words, kTexels> expected{};  // what a lookup must return, as 32-bit words
        Display display;
        mgl::TextureId tex{};
    };

    template <typename T>
    void put(std::vector<std::uint8_t>& out, T v)
    {
        std::array<std::uint8_t, sizeof(T)> raw{};
        std::memcpy(raw.data(), &v, sizeof(T));
        out.insert(out.end(), raw.begin(), raw.end());
    }

    /// Replaces the low `noise_bits` mantissa bits of a float by hash bits (zero stays zero).
    float noisy(float base, std::uint32_t h, int noise_bits)
    {
        if (base == 0.0f) return base;
        const std::uint32_t mask = (1u << noise_bits) - 1u;
        return as_float((bits_of(base) & ~mask) | (h & mask));
    }

    constexpr float kScale32 = 2.3283064365386963e-10f;  // 2^-32

    Case make_r8ui()
    {
        Case c{"R8UI", Kind::Uint, mgl::InternalFormat::R8UI, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedByte};
        for (int k = 0; k < kTexels; ++k)
        {
            const auto v = static_cast<std::uint8_t>(k * 4 + 3);  // 3 .. 255
            put(c.data, v);
            c.expected[static_cast<std::size_t>(k)] = {v, 0, 0, 1};
        }
        c.display = {{1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f}, {0, 0, 0}, true, 0};
        return c;
    }

    Case make_r16i()
    {
        Case c{"R16I", Kind::Int, mgl::InternalFormat::R16I, mgl::PixelFormat::RedInteger, mgl::PixelType::Short};
        for (int k = 0; k < kTexels; ++k)
        {
            const auto v = static_cast<std::int16_t>((k - 32) * 1000 + static_cast<int>(hash32(static_cast<std::uint32_t>(k)) & 255u) - 128);
            put(c.data, v);
            c.expected[static_cast<std::size_t>(k)] = {word_of(v), 0, 0, 1};
        }
        c.display = {{1.0f / 65535.0f, 1.0f / 65535.0f, 1.0f / 65535.0f}, {32768.0f / 65535.0f, 32768.0f / 65535.0f, 32768.0f / 65535.0f}, true, 0};
        return c;
    }

    Case make_rgba8i()
    {
        Case c{"RGBA8I", Kind::Int, mgl::InternalFormat::Rgba8I, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Byte};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const std::array<std::int8_t, 4> v = {static_cast<std::int8_t>(i * 36 - 126), static_cast<std::int8_t>(j * 36 - 126),
                                                  static_cast<std::int8_t>(((i + j) & 1) ? 127 : -128), static_cast<std::int8_t>(127 - 4 * k)};
            for (std::size_t n = 0; n < 4; ++n) put(c.data, v[n]);
            c.expected[static_cast<std::size_t>(k)] = {word_of(v[0]), word_of(v[1]), word_of(v[2]), word_of(v[3])};
        }
        c.display = {{1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f}, {128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f}, false, 0};
        return c;
    }

    Case make_rgba16ui()
    {
        Case c{"RGBA16UI", Kind::Uint, mgl::InternalFormat::Rgba16UI, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedShort};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const std::uint32_t h = hash32(static_cast<std::uint32_t>(k));
            const std::array<std::uint16_t, 4> v = {static_cast<std::uint16_t>(i * 9362 + static_cast<int>(h & 1u)),
                                                    static_cast<std::uint16_t>(j * 9362 + static_cast<int>((h >> 1) & 1u)),
                                                    static_cast<std::uint16_t>(i == j ? 65535 : 32768 + k), static_cast<std::uint16_t>(65535)};
            for (std::size_t n = 0; n < 4; ++n) put(c.data, v[n]);
            c.expected[static_cast<std::size_t>(k)] = {v[0], v[1], v[2], v[3]};
        }
        c.display = {{1.0f / 65535.0f, 1.0f / 65535.0f, 1.0f / 65535.0f}, {0, 0, 0}, false, 0};
        return c;
    }

    Case make_r32i_above_2_24()
    {
        Case c{"R32I 2^24 + k", Kind::Int, mgl::InternalFormat::R32I, mgl::PixelFormat::RedInteger, mgl::PixelType::Int};
        for (int k = 0; k < kTexels; ++k)
        {
            const std::int32_t v = 16777216 + k;  // 2^24 + k: odd k are not representable as a 32-bit float
            put(c.data, v);
            c.expected[static_cast<std::size_t>(k)] = {word_of(v), 0, 0, 1};
        }
        c.display.mode = 1;
        return c;
    }

    Case make_r32i_range()
    {
        Case c{"R32I full range", Kind::Int, mgl::InternalFormat::R32I, mgl::PixelFormat::RedInteger, mgl::PixelType::Int};
        for (int k = 0; k < kTexels; ++k)
        {
            std::int64_t v = static_cast<std::int64_t>(INT_MIN) + static_cast<std::int64_t>(k) * 68174084;
            if (k > 0 && k < kTexels - 1) v += static_cast<std::int64_t>(hash32(static_cast<std::uint32_t>(k) + 100u) & 0xFFFFu);
            if (k == kTexels - 1) v = INT_MAX;
            const auto w = static_cast<std::int32_t>(v);
            put(c.data, w);
            c.expected[static_cast<std::size_t>(k)] = {word_of(w), 0, 0, 1};
        }
        c.display = {{kScale32, kScale32, kScale32}, {0.5f, 0.5f, 0.5f}, true, 0};
        return c;
    }

    Case make_rgba32ui()
    {
        Case c{"RGBA32UI", Kind::Uint, mgl::InternalFormat::Rgba32UI, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const auto uk = static_cast<std::uint32_t>(k);
            auto grad = [](int x, std::uint32_t h) { return static_cast<std::uint32_t>(x) * 613566756u + (x < 7 ? (h & 0xFFFFu) : (h & 3u)); };
            const std::array<std::uint32_t, 4> v = {grad(i, hash32(uk)), grad(j, hash32(uk + 200u)),
                                                    uk * 68174084u + (k < kTexels - 1 ? (hash32(uk + 300u) & 0xFFFFu) : 0u), 0xFFFFFFFFu - uk};
            for (std::size_t n = 0; n < 4; ++n) put(c.data, v[n]);
            c.expected[static_cast<std::size_t>(k)] = v;
        }
        c.display = {{kScale32, kScale32, kScale32}, {0, 0, 0}, false, 0};
        return c;
    }

    Case make_rgb10a2ui()
    {
        Case c{"RGB10_A2UI", Kind::Uint, mgl::InternalFormat::Rgb10A2UI, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt2101010Rev};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const std::uint32_t h = hash32(static_cast<std::uint32_t>(k) + 400u);
            const auto r = static_cast<std::uint32_t>(i * 146) + (h & 1u);
            const auto g = static_cast<std::uint32_t>(j * 146) + ((h >> 1) & 1u);
            const auto b = static_cast<std::uint32_t>(1023 - k * 16);
            const auto a = static_cast<std::uint32_t>((i + j) & 3);
            put(c.data, r | (g << 10) | (b << 20) | (a << 30));  // red in the low bits (UNSIGNED_INT_2_10_10_10_REV)
            c.expected[static_cast<std::size_t>(k)] = {r, g, b, a};
        }
        c.display = {{1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 1023.0f}, {0, 0, 0}, false, 0};
        return c;
    }

    Case make_r32f()
    {
        Case c{"R32F", Kind::Float, mgl::InternalFormat::R32F, mgl::PixelFormat::Red, mgl::PixelType::Float};
        for (int k = 0; k < kTexels; ++k)
        {
            float v = noisy(static_cast<float>(k) / 63.0f, hash32(static_cast<std::uint32_t>(k) + 500u), 12);
            if (k == 1) v = FLT_MIN;       // the smallest normal float
            // The two largest magnitudes are NOT neighbors: a bilinear lookup next to both would compute FLT_MAX - (-FLT_MAX),
            // which overflows to infinity in a lerp of the form a + w * (b - a) and turns weight 0 into 0 * inf = NaN (llvmpipe).
            if (k == 56) v = -FLT_MAX;     // texel (0, 7)
            if (k == 63) v = FLT_MAX;      // texel (7, 7)
            put(c.data, v);
            c.expected[static_cast<std::size_t>(k)] = {bits_of(v), 0, 0, kOneBits};
        }
        c.display = {{1.0f, 1.0f, 1.0f}, {0, 0, 0}, true, 0};
        return c;
    }

    Case make_rgba32f()
    {
        Case c{"RGBA32F", Kind::Float, mgl::InternalFormat::Rgba32F, mgl::PixelFormat::Rgba, mgl::PixelType::Float};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const auto uk = static_cast<std::uint32_t>(k);
            const std::array<float, 4> v = {noisy(static_cast<float>(i) / 7.0f, hash32(uk + 600u), 12), noisy(static_cast<float>(j) / 7.0f, hash32(uk + 700u), 12),
                                            noisy(static_cast<float>(7 - i) / 7.0f, hash32(uk + 800u), 12),
                                            as_float(((uk & 1u) << 31) | ((95u + uk) << 23) | (hash32(uk + 900u) & 0x7FFFFFu))};  // sign and exponent vary: 2^-32 .. 2^31
            for (std::size_t n = 0; n < 4; ++n) put(c.data, v[n]);
            c.expected[static_cast<std::size_t>(k)] = {bits_of(v[0]), bits_of(v[1]), bits_of(v[2]), bits_of(v[3])};
        }
        return c;
    }

    Case make_rgba16f()
    {
        Case c{"RGBA16F (HALF_FLOAT data)", Kind::Float, mgl::InternalFormat::Rgba16F, mgl::PixelFormat::Rgba, mgl::PixelType::HalfFloat};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const auto uk = static_cast<std::uint32_t>(k);
            auto half_noisy = [&](float base, std::uint32_t salt) {
                std::uint16_t h = float_to_half(base);
                if (base != 0.0f) h = static_cast<std::uint16_t>((h & ~31u) | (hash32(uk + salt) & 31u));
                return h;
            };
            const std::array<std::uint16_t, 4> v = {half_noisy(static_cast<float>(i) / 7.0f, 1000u), half_noisy(static_cast<float>(j) / 7.0f, 1100u),
                                                    half_noisy(static_cast<float>(7 - i) / 7.0f, 1200u),
                                                    static_cast<std::uint16_t>(((uk & 1u) << 15) | ((1u + hash32(uk + 1300u) % 30u) << 10) | ((hash32(uk + 1400u) >> 8) & 1023u))};
            for (std::size_t n = 0; n < 4; ++n) put(c.data, v[n]);
            c.expected[static_cast<std::size_t>(k)] = {bits_of(half_to_float(v[0])), bits_of(half_to_float(v[1])), bits_of(half_to_float(v[2])),
                                                       bits_of(half_to_float(v[3]))};
        }
        return c;
    }

    Case make_r16f_from_float()
    {
        Case c{"R16F (FLOAT data)", Kind::Float, mgl::InternalFormat::R16F, mgl::PixelFormat::Red, mgl::PixelType::Float};
        for (int k = 0; k < kTexels; ++k)
        {
            float v = static_cast<float>(k) / 64.0f;  // exactly representable as a half: the conversion in the GL must keep it
            if (k == 1) v = 6.103515625e-05f;         // 2^-14, the smallest normal half
            if (k == 62) v = -65504.0f;               // the largest finite half magnitude
            if (k == 63) v = 65504.0f;
            put(c.data, v);
            c.expected[static_cast<std::size_t>(k)] = {bits_of(v), 0, 0, kOneBits};
        }
        c.display = {{1.0f, 1.0f, 1.0f}, {0, 0, 0}, true, 0};
        return c;
    }

    Case make_r11g11b10f()
    {
        Case c{"R11F_G11F_B10F (packed words)", Kind::Float, mgl::InternalFormat::R11FG11FB10F, mgl::PixelFormat::Rgb, mgl::PixelType::UnsignedInt10F11F11FRev};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const std::uint32_t h = hash32(static_cast<std::uint32_t>(k) + 1500u);
            const std::uint32_t r = encode_small_float(static_cast<double>(i + 1) / 8.0, 6) ^ (h & 7u);
            const std::uint32_t g = encode_small_float(static_cast<double>(j + 1) / 8.0, 6) ^ ((h >> 3) & 7u);
            const std::uint32_t b = encode_small_float(static_cast<double>((i + j) % 8 + 1) / 8.0, 5) ^ ((h >> 6) & 3u);
            put(c.data, r | (g << 11) | (b << 22));  // red in the low 11 bits
            c.expected[static_cast<std::size_t>(k)] = {bits_of(static_cast<float>(decode_small_float(r, 6))), bits_of(static_cast<float>(decode_small_float(g, 6))),
                                                       bits_of(static_cast<float>(decode_small_float(b, 5))), kOneBits};
        }
        return c;
    }

    void set_rgb9e5_expected(Case& c, int k, const Rgb9E5& w)
    {
        const std::array<float, 3> v = w.decode();
        c.expected[static_cast<std::size_t>(k)] = {bits_of(v[0]), bits_of(v[1]), bits_of(v[2]), kOneBits};
    }

    Case make_rgb9e5_words()
    {
        Case c{"RGB9_E5 (packed words)", Kind::Float, mgl::InternalFormat::Rgb9E5, mgl::PixelFormat::Rgb, mgl::PixelType::UnsignedInt5999Rev};
        for (int k = 0; k < kTexels; ++k)
        {
            const auto i = static_cast<std::uint32_t>(k % kTex), j = static_cast<std::uint32_t>(k / kTex);
            Rgb9E5 w{i * 73u, j * 73u, (static_cast<std::uint32_t>(k) * 8u + 7u) & 511u, 15u};  // exponent 15: value = mantissa / 512
            if (k == 62) w = {1u, 0u, 0u, 0u};      // 2^-24, the smallest nonzero value
            if (k == 63) w = {511u, 511u, 1u, 31u};  // 511 * 2^7 = 65408
            put(c.data, w.pack());
            set_rgb9e5_expected(c, k, w);
        }
        return c;
    }

    Case make_rgb9e5_from_float()
    {
        Case c{"RGB9_E5 (FLOAT data)", Kind::Float, mgl::InternalFormat::Rgb9E5, mgl::PixelFormat::Rgb, mgl::PixelType::Float};
        for (int k = 0; k < kTexels; ++k)
        {
            const int i = k % kTex, j = k / kTex;
            const float r = static_cast<float>(i + 1) * 0.1357f, g = static_cast<float>(j + 1) * 0.1119f, b = static_cast<float>((i * 3 + j * 5) % 8 + 1) * 0.0937f;
            put(c.data, r);
            put(c.data, g);
            put(c.data, b);
            set_rgb9e5_expected(c, k, encode_rgb9e5(r, g, b));  // the procedure of 3.8.3.2, done on the CPU
        }
        return c;
    }

    // Index of the R32F texture in the case list: the 16th cell samples it again through a LINEAR sampler object.
    constexpr std::size_t kCaseR32f = 8;
    constexpr std::size_t kCaseR32iAbove = 4;
    constexpr std::size_t kCaseR32iRange = 5;
    constexpr std::size_t kCaseRgba16f = 10;
    constexpr std::size_t kCasePacked = 12;
    constexpr std::size_t kCaseRgb9e5 = 13;

    // What a display cell must show for one texel: clamp(value * scale + bias, 0, 1) in 0..255.
    std::array<double, 3> display_color(const Case& c, const Words& w)
    {
        auto value = [&](std::size_t n) {
            switch (c.kind)
            {
            case Kind::Uint: return static_cast<double>(w[n]);
            case Kind::Int: return static_cast<double>(signed_of(w[n]));
            default: return static_cast<double>(as_float(w[n]));
            }
        };
        std::array<double, 3> out{};
        for (std::size_t n = 0; n < 3; ++n)
        {
            const double v = value(c.display.gray ? 0 : n);
            out[n] = std::clamp(v * static_cast<double>(c.display.scale[n]) + static_cast<double>(c.display.bias[n]), 0.0, 1.0) * 255.0;
        }
        return out;
    }

    std::string hex_words(const Words& w, Kind kind)
    {
        if (kind == Kind::Int) return std::format("({}, {}, {}, {})", signed_of(w[0]), signed_of(w[1]), signed_of(w[2]), signed_of(w[3]));
        return std::format("({:#x}, {:#x}, {:#x}, {:#x})", w[0], w[1], w[2], w[3]);
    }

    struct Diff
    {
        int bad_words = 0;
        int bad_texels = 0;
        std::string first;
    };

    Diff diff_words(const std::vector<Words>& got, const std::array<Words, kTexels>& expected, Kind kind)
    {
        Diff d;
        for (int k = 0; k < kTexels; ++k)
        {
            const Words& g = got[static_cast<std::size_t>(k)];
            const Words& e = expected[static_cast<std::size_t>(k)];
            bool texel_bad = false;
            for (std::size_t n = 0; n < 4; ++n)
                if (g[n] != e[n])
                {
                    ++d.bad_words;
                    texel_bad = true;
                }
            if (texel_bad && d.bad_texels++ == 0)
                d.first = std::format("first at texel ({},{}): got {} expected {}", k % kTex, k / kTex, hex_words(g, kind), hex_words(e, kind));
        }
        return d;
    }

    /// The name of the Mesa driver in GL_RENDERER ("zink", "radeonsi", "llvmpipe"), used as the renderer substring of
    /// Check::expect_or_known_deviation for deviations that all Mesa drivers share. Any other driver gets a tag that
    /// never matches, so the strict check applies.
    std::string mesa_driver_tag()
    {
        const GLubyte* r = mgl::glGetString(mgl::StringName::Renderer);
        const std::string_view renderer = r != nullptr ? reinterpret_cast<const char*>(r) : "";
        for (const std::string_view tag : {std::string_view{"zink"}, std::string_view{"radeonsi"}, std::string_view{"llvmpipe"}})
            if (renderer.find(tag) != std::string_view::npos) return std::string{tag};
        return "<not a known Mesa driver>";
    }

    /// Reason of a known deviation that only shows when GL_OES_texture_float_linear is NOT advertised (never in the CTest matrix: all
    /// Mesa drivers advertise it; reproduced by hand with MESA_EXTENSION_OVERRIDE=-GL_OES_texture_float_linear).
    constexpr std::string_view kMesaFloat32Complete =
        "Mesa keeps R32F + LINEAR complete (and sampled) in ES 3.x contexts even when GL_OES_texture_float_linear is not exposed; ES 3.0 3.8.13 / Table 3.13 make it incomplete";

    class IntegerFloatTextures final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kGrid * kMinCell || env.height < kGrid * kMinCell)
                throw demo::Skip(std::format("this demo needs a surface of at least {0}x{0} pixels (4 x 4 cells of {1} pixels)", kGrid * kMinCell, kMinCell));

            exact_[0] = demo::build_program(env, "shaders/quad.vert", "shaders/exact_uint.frag");
            exact_[1] = demo::build_program(env, "shaders/quad.vert", "shaders/exact_int.frag");
            exact_[2] = demo::build_program(env, "shaders/quad.vert", "shaders/exact_float.frag");
            show_[0] = demo::build_program(env, "shaders/quad.vert", "shaders/show_uint.frag");
            show_[1] = demo::build_program(env, "shaders/quad.vert", "shaders/show_int.frag");
            show_[2] = demo::build_program(env, "shaders/quad.vert", "shaders/show_float.frag");
            mixed_ = demo::build_program(env, "shaders/quad.vert", "shaders/mixed_samplers.frag");

            // Cell order (row by row from the top).
            cases_.push_back(make_r8ui());
            cases_.push_back(make_r16i());
            cases_.push_back(make_rgba8i());
            cases_.push_back(make_rgba16ui());
            cases_.push_back(make_r32i_above_2_24());
            cases_.push_back(make_r32i_range());
            cases_.push_back(make_rgba32ui());
            cases_.push_back(make_rgb10a2ui());
            cases_.push_back(make_r32f());
            cases_.push_back(make_rgba32f());
            cases_.push_back(make_rgba16f());
            cases_.push_back(make_r16f_from_float());
            cases_.push_back(make_r11g11b10f());
            cases_.push_back(make_rgb9e5_words());
            cases_.push_back(make_rgb9e5_from_float());

            // One texture per case. NEAREST filters: the initial MIN_FILTER (NEAREST_MIPMAP_LINEAR) needs a mip
            // chain, and the integer formats may not be filtered at all (3.8.13).
            for (Case& c : cases_)
            {
                mgl::glGenTextures(1, &c.tex);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, c.tex);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, c.internal, kTex, kTex, 0, c.format, c.type, c.data.data());
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            }

            // Two sampler objects: NEAREST and LINEAR (clamped), used to change the filters a texture is sampled with.
            mgl::glGenSamplers(2, samplers_.data());
            for (std::size_t n = 0; n < 2; ++n)
            {
                const bool linear = n == 1;
                // The typed overloads call glSamplerParameteri.
                mgl::glSamplerParameter(samplers_[n], linear ? mgl::TextureMinFilter::Linear : mgl::TextureMinFilter::Nearest);
                mgl::glSamplerParameter(samplers_[n], linear ? mgl::TextureMagFilter::Linear : mgl::TextureMagFilter::Nearest);
                mgl::glSamplerParameter(samplers_[n], mgl::SamplerWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
                mgl::glSamplerParameter(samplers_[n], mgl::SamplerWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            }

            // Render targets of the exact path: one RGBA32UI and one RGBA32I renderbuffer of 8 x 8 pixels
            // (both are color-renderable in core ES 3.0, Table 3.13), one framebuffer object each.
            mgl::glGenRenderbuffers(2, rbo_.data());
            mgl::glGenFramebuffers(2, fbo_.data());
            const std::array<mgl::InternalFormat, 2> target_format = {mgl::InternalFormat::Rgba32UI, mgl::InternalFormat::Rgba32I};
            for (std::size_t n = 0; n < 2; ++n)
            {
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rbo_[n]);
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, target_format[n], kTex, kTex);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[n]);
                mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                               mgl::RenderbufferTarget::Renderbuffer, rbo_[n]);
                const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
                if (status != mgl::FramebufferStatus::Complete)
                    throw demo::Failure(std::format("integer render target {} is incomplete: {}", n, demo::framebuffer_status_name(status)));
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            // The core spec makes 32-bit float textures non-filterable; an extension may lift that.
            float_linear_ = mgl::HasExtension("GL_OES_texture_float_linear");
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            width_ = env.width;
            height_ = env.height;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, width_, height_);
            mgl::glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (int cell = 0; cell < kGrid * kGrid; ++cell)
            {
                const bool linear_cell = cell == 15;
                const Case& c = cases_[linear_cell ? kCaseR32f : static_cast<std::size_t>(cell)];
                const mgl::ProgramId program = show_[static_cast<std::size_t>(c.kind)];
                const Rect r = cell_rect(cell, width_, height_);  // cell 0 is the top left
                mgl::glUseProgram(program);
                mgl::glViewport(r.x, r.y, r.w, r.h);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, c.tex);
                mgl::glBindSampler(0, linear_cell ? samplers_[1] : mgl::SamplerId{0});
                mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_tex"), 0);
                mgl::glUniform2i(mgl::glGetUniformLocation(program, "u_origin"), r.x, r.y);
                mgl::glUniform2i(mgl::glGetUniformLocation(program, "u_cell_size"), r.w, r.h);
                mgl::glUniform3f(mgl::glGetUniformLocation(program, "u_scale"), c.display.scale[0], c.display.scale[1], c.display.scale[2]);
                mgl::glUniform3f(mgl::glGetUniformLocation(program, "u_bias"), c.display.bias[0], c.display.bias[1], c.display.bias[2]);
                mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_gray"), c.display.gray ? 1 : 0);
                mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_mode"), c.display.mode);  // only show_int.frag has it (location -1 is ignored)
                mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_use_texture"), linear_cell ? 1 : 0);
                mgl::glUniform1f(mgl::glGetUniformLocation(program, "u_zero"), 0.0f);  // only show_int.frag has it
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
            mgl::glBindSampler(0, mgl::SamplerId{0});
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            // The window is read first; everything below renders into the offscreen targets.
            check_display(check);

            check_exact_cases(check);
            check_negative_controls(check);
            check_integer_completeness(check);
            check_float_filtering(check);
            check_float32_completeness(check);
            check_generate_mipmap_rules(check);
            check_format_errors(check);
            check_sampler_types(check);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindSampler(0, mgl::SamplerId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (Case& c : cases_) mgl::glDeleteTextures(1, &c.tex);
            mgl::glDeleteSamplers(2, samplers_.data());
            mgl::glDeleteFramebuffers(2, fbo_.data());
            mgl::glDeleteRenderbuffers(2, rbo_.data());
            for (const mgl::ProgramId p : exact_) mgl::glDeleteProgram(p);
            for (const mgl::ProgramId p : show_) mgl::glDeleteProgram(p);
            mgl::glDeleteProgram(mixed_);
        }

    private:
        // ------------------------------------------------------------------------------------------
        // The exact path: sample a texture, write the result unchanged into an RGBA32UI / RGBA32I target, read
        // the 32-bit words back. mode 0 texture(), 1 texelFetch(level), 2 textureLod(level).
        // ------------------------------------------------------------------------------------------
        std::vector<Words> run_exact(Kind kind, mgl::TextureId tex, int mode, int lod = 0, float uv_shift_x = 0.0f,
                                     mgl::SamplerId sampler = mgl::SamplerId{0}) const
        {
            const std::size_t k = static_cast<std::size_t>(kind);
            const mgl::ProgramId program = exact_[k];
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, kind == Kind::Int ? fbo_[1] : fbo_[0]);
            mgl::glViewport(0, 0, kTex, kTex);
            mgl::glUseProgram(program);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glBindSampler(0, sampler);
            mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_tex"), 0);
            mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_mode"), mode);
            mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_lod"), lod);
            mgl::glUniform2f(mgl::glGetUniformLocation(program, "u_uv_shift"), uv_shift_x, 0.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glBindSampler(0, mgl::SamplerId{0});

            std::vector<Words> out(kTexels);
            if (kind == Kind::Int)
            {
                std::vector<std::int32_t> px(kTexels * 4);
                mgl::glReadPixels(0, 0, kTex, kTex, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int, px.data());
                for (std::size_t t = 0; t < kTexels; ++t)
                    for (std::size_t n = 0; n < 4; ++n) out[t][n] = word_of(px[t * 4 + n]);
            }
            else
            {
                std::vector<std::uint32_t> px(kTexels * 4);
                mgl::glReadPixels(0, 0, kTex, kTex, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, px.data());
                for (std::size_t t = 0; t < kTexels; ++t)
                    for (std::size_t n = 0; n < 4; ++n) out[t][n] = px[t * 4 + n];
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return out;
        }

        // ------------------------------------------------------------------------------------------
        // The contact sheet
        // ------------------------------------------------------------------------------------------
        void check_display(demo::Check& check) const
        {
            const std::vector<std::uint8_t> window = demo::read_rgba8(0, 0, width_, height_);
            auto pixel = [&](int x, int y) {
                const auto o = static_cast<std::size_t>((y * width_ + x) * 4);
                return Rgba8{window[o], window[o + 1], window[o + 2], window[o + 3]};
            };
            for (int cell = 0; cell < kGrid * kGrid; ++cell)
            {
                const bool linear_cell = cell == 15;
                const Case& c = cases_[linear_cell ? kCaseR32f : static_cast<std::size_t>(cell)];
                const Rect r = cell_rect(cell, width_, height_);
                int probes = 0, bad = 0;
                std::string first;
                auto compare = [&](int x, int y, const std::array<double, 3>& want, std::string_view where) {
                    const Rgba8 got = pixel(x, y);
                    ++probes;
                    bool ok = got[3] == 255;
                    for (std::size_t n = 0; n < 3; ++n) ok = ok && std::abs(static_cast<double>(got[n]) - want[n]) <= 1.0 + 1e-9;
                    // The color is clamp(float(value) * scale + bias) written to an 8-bit target: the result
                    // is rounded to the nearest of 256 steps (equation 2.3), and single-precision arithmetic
                    // may land on the neighbor of a rounding boundary. Hence +-1.
                    if (!ok && bad++ == 0)
                        first = std::format("{} at pixel ({},{}): got {} expected ({:.1f}, {:.1f}, {:.1f})", where, x, y, demo::format_rgba(got), want[0], want[1], want[2]);
                };
                for (int k = 0; k < kTexels; ++k)
                {
                    const int i = k % kTex, j = k / kTex;
                    // The centre pixel of the block that shows texel (i, j); the integer texel mapping of the shaders
                    // is exact, so every pixel of the block shows that texel and the centre is never ambiguous.
                    const int px = r.x + block_centre(i, kTex, r.w), py = r.y + block_centre(j, kTex, r.h);
                    const Words& w = c.expected[static_cast<std::size_t>(k)];
                    if (c.display.mode == 1)
                    {
                        // Left half: (value - 2^24) / 63 as gray = k / 63. Right half: green for even k (float(value)
                        // is exact), red for odd k (an odd integer above 2^24 has no float).
                        const double gray = 255.0 * static_cast<double>(k) / 63.0;
                        compare(r.x + block_centre(2 * i, 2 * kTex, r.w), py, {gray, gray, gray}, "left half");
                        const std::array<double, 3> stripe = (k % 2 == 0) ? std::array<double, 3>{0.0, 255.0, 0.0} : std::array<double, 3>{255.0, 0.0, 0.0};
                        compare(r.x + block_centre(2 * i + 1, 2 * kTex, r.w), py, stripe, "right half");
                    }
                    else if (linear_cell && !float_linear_)
                    {
                        compare(px, py, {0.0, 0.0, 0.0}, "incomplete R32F + LINEAR");  // (0, 0, 0, 1)
                    }
                    else
                    {
                        compare(px, py, display_color(c, w), "texel center");
                    }
                }
                std::string what = linear_cell ? std::format("cell {}: R32F through a LINEAR sampler object: {}", cell,
                                                             float_linear_ ? "GL_OES_texture_float_linear is present, so the texture is complete and shows the gradient"
                                                                           : "no float-linear extension, so the texture is incomplete and every lookup is (0, 0, 0, 1)")
                                                 : std::format("cell {}: {}", cell, c.name);
                const std::string description = std::format("{} - {} probes match the CPU color mapping", what, probes);
                if (linear_cell && !float_linear_)
                    check.expect_or_known_deviation(bad == 0 && probes >= kTexels, description, first, mesa_driver_tag(), kMesaFloat32Complete);
                else
                    check.expect(bad == 0 && probes >= kTexels, description, first);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Every texture, exactly
        // ------------------------------------------------------------------------------------------
        void check_exact_cases(demo::Check& check) const
        {
            for (const Case& c : cases_)
            {
                const Diff by_texture = diff_words(run_exact(c.kind, c.tex, 0), c.expected, c.kind);
                const Diff by_fetch = diff_words(run_exact(c.kind, c.tex, 1), c.expected, c.kind);
                const std::string what_texture =
                    std::format("{}: texture() returns all {} texels as the exact 32-bit words ({} words compared)", c.name, kTexels, kTexels * 4);
                const std::string what_fetch = std::format("{}: texelFetch() returns the same exact words", c.name);
                const std::string detail_texture = std::format("{} texels wrong; {}", by_texture.bad_texels, by_texture.first);
                const std::string detail_fetch = std::format("{} texels wrong; {}", by_fetch.bad_texels, by_fetch.first);
                if (c.internal == mgl::InternalFormat::Rgb9E5 && c.type == mgl::PixelType::Float)
                {
                    // The encoding of 3.8.3.2 is a fixed procedure (expp = max(-B-1, floor(log2(maxc))) + 1 + B, then
                    // round-to-nearest mantissas), and 3.8.17 decodes the stored bits exactly.
                    constexpr std::string_view kAngleRgb9e5 =
                        "ANGLE's CPU conversion of FLOAT data to RGB9_E5 picks a larger shared exponent than the procedure of ES 3.0 "
                        "§3.8.3.2 / ES 3.2 §8.5.2 (expp = max(-B-1, floor(log2(maxc))) + 1 + B) for 4 of the 64 texels, so their "
                        "components lose the least significant mantissa bit; the 4 texels are exactly those where floor(ln(maxc)) "
                        "differs from floor(log2(maxc))";
                    check.expect_or_known_deviation(by_texture.bad_words == 0, what_texture, detail_texture, "ANGLE", kAngleRgb9e5);
                    check.expect_or_known_deviation(by_fetch.bad_words == 0, what_fetch, detail_fetch, "ANGLE", kAngleRgb9e5);
                }
                else
                {
                    check.expect(by_texture.bad_words == 0, what_texture, detail_texture);
                    check.expect(by_fetch.bad_words == 0, what_fetch, detail_fetch);
                }
            }
        }

        // The comparison can fail, and values above 2^24 really are not float-representable: take the texels that
        // the R32I texture holds, convert each through a 32-bit float on the CPU, and compare the GPU's exact result
        // with THAT expectation. Exactly the 32 odd values must differ.
        void check_negative_controls(demo::Check& check) const
        {
            const Case& c = cases_[kCaseR32iAbove];
            const std::vector<Words> got = run_exact(c.kind, c.tex, 1);
            std::array<Words, kTexels> through_float = c.expected;
            for (Words& w : through_float) w[0] = word_of(static_cast<std::int32_t>(static_cast<float>(signed_of(w[0]))));
            const Diff d = diff_words(got, through_float, c.kind);
            check.expect(d.bad_texels == kTexels / 2,
                         "negative control: the exact words of 2^24 + k differ from the same values after a trip through float at exactly the 32 odd k",
                         std::format("{} texels differ; {}", d.bad_texels, d.first));

            // A single flipped bit in the expectation must be reported (the comparator is not vacuous).
            std::array<Words, kTexels> flipped = cases_[kCaseR32f].expected;
            flipped[37][0] ^= 1u;
            const Diff f = diff_words(run_exact(Kind::Float, cases_[kCaseR32f].tex, 0), flipped, Kind::Float);
            check.expect(f.bad_words == 1, "negative control: one flipped bit in the expected R32F words is reported as exactly one differing word",
                         std::format("{} words differ", f.bad_words));
        }

        // ------------------------------------------------------------------------------------------
        // ES30.TEXTURE.COMPLETENESS_INTEGER: which filter state makes an integer texture complete.
        // ------------------------------------------------------------------------------------------
        void check_integer_completeness(demo::Check& check) const
        {
            const Case& range = cases_[kCaseR32iRange];

            // 1. A new texture has MIN_FILTER = NEAREST_MIPMAP_LINEAR and MAG_FILTER = LINEAR (3.8.14): both violate the
            //    rule of 3.8.13 for a not texture-filterable format, so an integer texture in its initial state is
            //    INCOMPLETE. A lookup of an incomplete integer texture has no defined value (ES 3.2 11.1.3.5;
            //    ES 3.0 says only "(0, 0, 0, 1)" without naming the type), so nothing is asserted about what it returns.
            mgl::TextureId fresh{};
            mgl::glGenTextures(1, &fresh);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, fresh);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, range.internal, kTex, kTex, 0, range.format, range.type, range.data.data());
            GLint min_filter = 0, mag_filter = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MinFilter, &min_filter);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MagFilter, &mag_filter);
            check.expect(min_filter == GL_NEAREST_MIPMAP_LINEAR && mag_filter == GL_LINEAR,
                         "a new integer texture has the initial filters NEAREST_MIPMAP_LINEAR / LINEAR, i.e. it is incomplete until NEAREST filters are set (3.8.13, 3.8.14)",
                         std::format("MIN {:#x}, MAG {:#x}", min_filter, mag_filter));

            const std::vector<Words> incomplete = run_exact(Kind::Int, fresh, 0);
            std::cout << std::format("   note: texture() on the incomplete R32I texture returned {} for texel (0,0): undefined by the specification, not asserted\n",
                                     hex_words(incomplete[0], Kind::Int));

            // 2. The filters in effect are those of the sampler object bound to the unit (3.8.13.1): the very same texture
            //    object, still in its incomplete initial state, becomes complete on a unit whose sampler is NEAREST.
            const Diff by_sampler = diff_words(run_exact(Kind::Int, fresh, 0, 0, 0.0f, samplers_[0]), range.expected, Kind::Int);
            check.expect(by_sampler.bad_words == 0,
                         "R32I texture in its incomplete initial state + a NEAREST sampler object on the unit: the texture is complete there, texture() returns all exact values",
                         by_sampler.first);
            mgl::glDeleteTextures(1, &fresh);

            // 3. NEAREST_MIPMAP_NEAREST (with MAG_FILTER NEAREST) is allowed for integer formats: a four-level R32I chain.
            mgl::TextureId chain{};
            mgl::glGenTextures(1, &chain);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, chain);
            for (int level = 0; level < 4; ++level)
            {
                const int size = kTex >> level;
                std::vector<std::int32_t> texels;
                for (int n = 0; n < size * size; ++n) texels.push_back(mip_value(level, n % size, n / size));
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::R32I, size, size, 0, mgl::PixelFormat::RedInteger, mgl::PixelType::Int,
                                  texels.data());
            }
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::NearestMipmapNearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);

            auto expected_level = [](int level) {
                std::array<Words, kTexels> e{};
                for (int k = 0; k < kTexels; ++k) e[static_cast<std::size_t>(k)] = {word_of(mip_value(level, (k % kTex) >> level, (k / kTex) >> level)), 0, 0, 1};
                return e;
            };
            // texture() with one texel per pixel has scale factor 1, lambda 0: level 0 (equation 3.18 rounding for NEAREST_MIPMAP_NEAREST).
            const Diff base = diff_words(run_exact(Kind::Int, chain, 0), expected_level(0), Kind::Int);
            check.expect(base.bad_words == 0, "R32I mip chain with NEAREST_MIPMAP_NEAREST / NEAREST: texture() at one texel per pixel returns level 0 exactly (3.8.13: filterable-or-NEAREST rule met)", base.first);
            int bad_lod = 0, bad_fetch = 0;
            std::string first_lod, first_fetch;
            for (int level = 1; level <= 3; ++level)
            {
                const Diff l = diff_words(run_exact(Kind::Int, chain, 2, level), expected_level(level), Kind::Int);
                const Diff f = diff_words(run_exact(Kind::Int, chain, 1, level), expected_level(level), Kind::Int);
                if (l.bad_words != 0 && bad_lod++ == 0) first_lod = std::format("level {}: {}", level, l.first);
                if (f.bad_words != 0 && bad_fetch++ == 0) first_fetch = std::format("level {}: {}", level, f.first);
            }
            check.expect(bad_lod == 0, "R32I mip chain: textureLod() at lod 1, 2 and 3 returns the exact texels of those levels", first_lod);
            check.expect(bad_fetch == 0, "R32I mip chain: texelFetch() at levels 1, 2 and 3 returns the exact texels of those levels", first_fetch);
            mgl::glDeleteTextures(1, &chain);
        }

        /// A value of the R32I mip chain: unique per (level, i, j), above 2^24, and different in every level.
        static std::int32_t mip_value(int level, int i, int j) { return 100000000 * (level + 1) + 16777217 + 131 * j + 7 * i; }

        // ------------------------------------------------------------------------------------------
        // Filterable formats: the 16-bit, packed and shared-exponent formats can be filtered (Table 3.13).
        // ------------------------------------------------------------------------------------------
        void check_float_filtering(demo::Check& check) const
        {
            struct Filterable
            {
                std::size_t index;
                const char* name;
            };
            const std::array<Filterable, 3> formats = {{{kCaseRgba16f, "RGBA16F"}, {kCasePacked, "R11F_G11F_B10F"}, {kCaseRgb9e5, "RGB9_E5"}}};
            for (const Filterable& f : formats)
            {
                const Case& c = cases_[f.index];
                // At a texel center a bilinear lookup returns that texel; half a texel to the right it returns the mean of
                // two neighbors (weights 1/2, exactly representable in any weight format). The filter arithmetic is not specified
                // beyond "implementation-dependent" (3.8.10.2, 2.1.1), so the tolerance is 2^-10 of the larger magnitude, which
                // is the precision of a half float.
                for (const float shift : {0.0f, 0.0625f})
                {
                    const std::vector<Words> got = run_exact(Kind::Float, c.tex, 0, 0, shift, samplers_[1]);
                    int bad = 0, probes = 0;
                    std::string first;
                    for (int k = 0; k < kTexels; ++k)
                    {
                        const int i = k % kTex;
                        if (shift != 0.0f && i == kTex - 1) continue;  // the neighbor would be outside the image (CLAMP_TO_EDGE)
                        for (std::size_t n = 0; n < 4; ++n)
                        {
                            const double a = static_cast<double>(as_float(c.expected[static_cast<std::size_t>(k)][n]));
                            const double b = static_cast<double>(as_float(c.expected[static_cast<std::size_t>(shift != 0.0f ? k + 1 : k)][n]));
                            const double want = shift != 0.0f ? 0.5 * (a + b) : a;
                            const double have = static_cast<double>(as_float(got[static_cast<std::size_t>(k)][n]));
                            const double tolerance = std::ldexp(std::max(std::fabs(a), std::fabs(b)), -10);
                            ++probes;
                            if (std::fabs(have - want) > tolerance && bad++ == 0)
                                first = std::format("texel ({},{}) component {}: got {} expected {} (+-{})", i, k / kTex, n, have, want, tolerance);
                        }
                    }
                    check.expect(bad == 0, std::format("{} with a LINEAR sampler: {} ({} components)", f.name,
                                                       shift == 0.0f ? "lookups at texel centers return the texel" : "lookups half a texel to the right return the mean of two texels", probes),
                                 first);
                }
            }
        }

        // R32F is NOT texture-filterable in core ES 3.0 (Table 3.13), so with a LINEAR sampler the texture is incomplete and a
        // float lookup returns (0, 0, 0, 1) (3.9.2.1). GL_OES_texture_float_linear makes it filterable; the expectation follows the
        // extension string, and nothing but the presence of that string is taken from the extension.
        void check_float32_completeness(demo::Check& check) const
        {
            const Case& c = cases_[kCaseR32f];
            const std::vector<Words> got = run_exact(Kind::Float, c.tex, 0, 0, 0.0f, samplers_[1]);
            if (float_linear_)
            {
                // Complete: weights at a texel center are (1, 0, 0, 0), the lookup is the texel itself.
                const Diff d = diff_words(got, c.expected, Kind::Float);
                check.expect(d.bad_words == 0, "R32F + LINEAR sampler, GL_OES_texture_float_linear present: the texture is complete, lookups at texel centers return the exact texel",
                             d.first);
            }
            else
            {
                std::array<Words, kTexels> incomplete{};
                for (Words& w : incomplete) w = {0u, 0u, 0u, kOneBits};
                const Diff d = diff_words(got, incomplete, Kind::Float);
                check.expect_or_known_deviation(d.bad_words == 0, "R32F + LINEAR sampler, no float-linear extension: the texture is incomplete, every lookup returns (0, 0, 0, 1) (3.8.13, 3.9.2.1)", d.first,
                                                mesa_driver_tag(), kMesaFloat32Complete);
            }
            // NEAREST on the same texture is always complete.
            const Diff nearest = diff_words(run_exact(Kind::Float, c.tex, 0, 0, 0.0f, samplers_[0]), c.expected, Kind::Float);
            check.expect(nearest.bad_words == 0, "R32F + NEAREST sampler: complete, exact values", nearest.first);
            if (float_linear_) std::cout << "   note: GL_OES_texture_float_linear is advertised; the incompleteness of R32F + LINEAR is therefore not asserted on this driver\n";
        }

        // 3.8.10.5: GenerateMipmap needs a color-renderable AND texture-filterable format (Table 3.13). The integer formats
        // are renderable but never filterable; RGB9_E5 is filterable but never renderable.
        void check_generate_mipmap_rules(demo::Check& check) const
        {
            auto expect_error = [&](const Case& c, std::string_view what) {
                demo::drain_gl_errors();
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, c.tex);
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, what);
            };
            expect_error(cases_[0], "glGenerateMipmap on an R8UI texture -> INVALID_OPERATION (integer formats are not texture-filterable, 3.8.10.5)");
            expect_error(cases_[kCaseR32iRange], "glGenerateMipmap on an R32I texture -> INVALID_OPERATION");
            expect_error(cases_[3], "glGenerateMipmap on an RGBA16UI texture -> INVALID_OPERATION");
            expect_error(cases_[6], "glGenerateMipmap on an RGBA32UI texture -> INVALID_OPERATION");
            expect_error(cases_[kCaseRgb9e5], "glGenerateMipmap on an RGB9_E5 texture -> INVALID_OPERATION (not color-renderable, 3.8.10.5)");

            // Positive control: RGBA8 is renderable and filterable.
            mgl::TextureId rgba8{};
            mgl::glGenTextures(1, &rgba8);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, rgba8);
            const std::vector<std::uint8_t> zeros(kTexels * 4, 0);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kTex, kTex, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, zeros.data());
            demo::drain_gl_errors();
            mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
            check.expect_no_gl_error("positive control: glGenerateMipmap on an RGBA8 texture raises no error");
            mgl::glDeleteTextures(1, &rgba8);
        }

        // 3.8.3 / Tables 3.2 and 3.3: a combination of format, type and internalformat that is not listed generates INVALID_OPERATION.
        // The integer / float split is the point: integer internal formats need an *_INTEGER format.
        void check_format_errors(demo::Check& check) const
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            const std::vector<std::uint8_t> zeros(kTexels * 16, 0);
            auto error_of = [&](mgl::InternalFormat internal, mgl::PixelFormat format, mgl::PixelType type) {
                demo::drain_gl_errors();
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, internal, kTex, kTex, 0, format, type, zeros.data());
                return demo::drain_gl_errors();
            };
            auto expect_error = [&](mgl::InternalFormat internal, mgl::PixelFormat format, mgl::PixelType type, std::string_view what) {
                const mgl::ErrorCode e = error_of(internal, format, type);
                check.expect(e == mgl::ErrorCode::InvalidOperation, what, std::format("glGetError() = {}, expected GL_INVALID_OPERATION", demo::error_name(e)));
            };
            // Mesa raises INVALID_ENUM instead of INVALID_OPERATION when an integer format meets FLOAT (the rule of desktop
            // EXT_texture_integer / GL 3.0). ES 3.0 3.8.3 and ES 3.2 8.5 ("format, type, and internalformat are each valid, but the
            // combination ... is not listed") both say INVALID_OPERATION, so the strict check stays and the deviation is named.
            auto expect_error_known_deviation = [&](mgl::InternalFormat internal, mgl::PixelFormat format, mgl::PixelType type, std::string_view what) {
                const mgl::ErrorCode e = error_of(internal, format, type);
                check.expect_or_known_deviation(e == mgl::ErrorCode::InvalidOperation, what,
                                                std::format("glGetError() = {}, expected GL_INVALID_OPERATION", demo::error_name(e)), mesa_driver_tag(),
                                                "Mesa reports GL_INVALID_ENUM for an integer pixel format with type FLOAT (desktop-GL rule); ES 3.0 3.8.3 / ES 3.2 8.5 require GL_INVALID_OPERATION");
            };
            using F = mgl::PixelFormat;
            using T = mgl::PixelType;
            using I = mgl::InternalFormat;
            expect_error_known_deviation(I::R32I, F::RedInteger, T::Float, "R32I with RED_INTEGER / FLOAT -> INVALID_OPERATION (Table 3.2 lists only INT for R32I)");
            expect_error(I::R32I, F::Red, T::Int, "R32I with RED / INT (a non-integer format) -> INVALID_OPERATION");
            expect_error(I::R8UI, F::RedInteger, T::UnsignedInt, "R8UI with RED_INTEGER / UNSIGNED_INT -> INVALID_OPERATION (R8UI needs UNSIGNED_BYTE)");
            expect_error_known_deviation(I::Rgba32F, F::RgbaInteger, T::Float, "RGBA32F with RGBA_INTEGER / FLOAT -> INVALID_OPERATION");
            expect_error(I::Rgba8, F::RgbaInteger, T::UnsignedByte, "RGBA8 with RGBA_INTEGER / UNSIGNED_BYTE -> INVALID_OPERATION (that pair is for RGBA8UI)");
            demo::drain_gl_errors();
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, I::Rgba8UI, kTex, kTex, 0, F::RgbaInteger, T::UnsignedByte, zeros.data());
            check.expect_no_gl_error("positive control: RGBA8UI with RGBA_INTEGER / UNSIGNED_BYTE is accepted");
            mgl::glDeleteTextures(1, &t);
        }

        // GLSL ES 3.00 4.1.7 / 4.1.10 / 8.8 and ES 3.0 2.12.9.7.
        void check_sampler_types(demo::Check& check) const
        {
            // texture() on a usampler2D returns uvec4; there is no implicit conversion to vec4.
            const std::string bad_source =
                "#version 300 es\nprecision highp float;\nprecision highp usampler2D;\nuniform usampler2D u_tex;\nout vec4 o;\n"
                "void main() { o = texture(u_tex, vec2(0.5)); }\n";
            const std::string good_source =
                "#version 300 es\nprecision highp float;\nprecision highp usampler2D;\nuniform usampler2D u_tex;\nout vec4 o;\n"
                "void main() { o = vec4(texture(u_tex, vec2(0.5))); }\n";
            std::string log;
            const mgl::ShaderId bad = demo::compile_shader(mgl::ShaderType::Fragment, bad_source, log);
            check.expect(bad.value == 0 && !log.empty(), "negative: assigning the uvec4 of texture(usampler2D) to a vec4 does not compile (no implicit conversion)", log);
            if (bad.value != 0) mgl::glDeleteShader(bad);
            log.clear();
            const mgl::ShaderId good = demo::compile_shader(mgl::ShaderType::Fragment, good_source, log);
            check.expect(good.value != 0, "positive control: the same shader with an explicit vec4(...) conversion compiles", log);
            if (good.value != 0) mgl::glDeleteShader(good);

            // Two active samplers of different types on one unit: INVALID_OPERATION at the draw call (2.12.9.7).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            mgl::glViewport(0, 0, kTex, kTex);
            mgl::glUseProgram(mixed_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, cases_[kCaseR32f].tex);
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, cases_[0].tex);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glUniform1i(mgl::glGetUniformLocation(mixed_, "u_float_tex"), 0);
            mgl::glUniform1i(mgl::glGetUniformLocation(mixed_, "u_uint_tex"), 0);
            demo::drain_gl_errors();
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            const mgl::ErrorCode mixed_error = demo::drain_gl_errors();
            // Mesa only compares the texture TARGET of the samplers (TEXTURE_2D for both), not their GLSL type, and raises nothing here.
            check.expect_or_known_deviation(mixed_error == mgl::ErrorCode::InvalidOperation,
                                            "a sampler2D and a usampler2D both on texture unit 0: the draw call fails with INVALID_OPERATION (2.12.9.7)",
                                            std::format("glGetError() = {}, expected GL_INVALID_OPERATION", demo::error_name(mixed_error)), mesa_driver_tag(),
                                            "Mesa validates sampler uniforms by texture target only; sampler2D + usampler2D on one unit is not detected");
            mgl::glUniform1i(mgl::glGetUniformLocation(mixed_, "u_uint_tex"), 1);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            check.expect_no_gl_error("positive control: the same program with the usampler2D on unit 1 draws without error");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        std::array<mgl::ProgramId, 3> exact_{};
        std::array<mgl::ProgramId, 3> show_{};
        mgl::ProgramId mixed_{};
        std::vector<Case> cases_;
        std::array<mgl::SamplerId, 2> samplers_{};   // [0] NEAREST, [1] LINEAR, both CLAMP_TO_EDGE
        int width_ = 0, height_ = 0;                  // window size: the 4 x 4 cells partition it
        std::array<mgl::RenderbufferId, 2> rbo_{};   // RGBA32UI, RGBA32I
        std::array<mgl::FramebufferId, 2> fbo_{};
        bool float_linear_ = false;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-integer-float-textures",
        .title = "Integer, float and packed-float textures",
        .min_es = {3, 0},
        .summary = "usampler2D/isampler2D lookups return exact integers (even above 2^24), float and half-float textures return exact floats, "
                   "RGB9_E5 and R11F_G11F_B10F decode as specified; completeness vs. filters, filterability, GenerateMipmap and format errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<IntegerFloatTextures>(argc, argv, info);
}
