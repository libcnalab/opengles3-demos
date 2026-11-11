// es30-immutable-storage
//
// Immutable-format textures (OpenGL ES 3.0 section 3.8.4: glTexStorage2D / glTexStorage3D) and the sized internal
// formats of Table 3.13 with the format/type pairs of Table 3.2 that may be used to fill them.
//
// A texture created by glTexStorage* has ALL its levels allocated at once, can only change its texel values (TexSubImage*,
// CopyTexSubImage*) and its parameters, and answers every command that would change its format or size with
// INVALID_OPERATION. The demo shows that with pixels (what is sampled before and after the refused commands), with the
// TEXTURE_IMMUTABLE_FORMAT / TEXTURE_IMMUTABLE_LEVELS queries, and with the defined errors of section 3.8.4.
//
// The window is a contact sheet of 8 x 8 panels (32 x 32 pixels each, panel 0 at the lower left, panels are numbered
// left to right and then upwards):
//
//   panels  0-22  13 sized formats x their valid upload pairs; every texture is made with glTexStorage2D(1 level)
//                 and filled with glTexSubImage2D; each panel is compared with decoded values computed on the CPU
//   panels 23-26  an immutable 8x8 texture with 4 levels: textureLod at level 0, 1, 2, 3 (every level exists)
//   panel  27     the same filter state on a mutable texture with only level 0: incomplete, samples black
//   panels 28-29  TEXTURE_BASE_LEVEL = 100: clamped to the last level of an immutable texture (equation 3.14), incomplete
//                 for a mutable one
//   panels 30-32  textureSize of the levels of an immutable 7x5 texture     33-36 ... of a 4x2x8 3D texture
//   panels 37-39  ... of a 6x3x5 array texture

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kGrid = 8;     // 8 x 8 panels
    constexpr int kTexel = 4;    // the format textures are 4 x 4
    constexpr int kTexels = kTexel * kTexel;
    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    using Color = std::array<double, 4>;

    // ---------------------------------------------------------------------------------------------
    // CPU encoders and decoders (ES 3.0 sections 2.1.2 - 2.1.4, 3.8.3.2). Written from the specification, not
    // from the GL: the expected pixels must not depend on the implementation under test.
    // ---------------------------------------------------------------------------------------------
    template <typename T>
    void put(std::vector<std::uint8_t>& out, T value)
    {
        std::uint8_t raw[sizeof(T)];
        std::memcpy(raw, &value, sizeof(T));
        out.insert(out.end(), raw, raw + sizeof(T));
    }

    // 16-bit half float of a non-negative value that is exactly representable (the demo only uses such values).
    std::uint16_t half_from_float(double v)
    {
        if (v == 0.0) return 0;
        int e = 0;
        const double m = std::frexp(v, &e);  // v = m * 2^e with m in [0.5, 1)
        const int biased = e - 1 + 15;
        const int mantissa = static_cast<int>(std::lround((2.0 * m - 1.0) * 1024.0));
        return static_cast<std::uint16_t>((biased << 10) | mantissa);
    }

    // Unsigned 11-bit (5 exponent, 6 mantissa bits) and 10-bit (5, 5) floating point: section 2.1.3.
    std::uint32_t uf_encode(double v, int mantissa_bits)
    {
        if (v == 0.0) return 0;
        int e = 0;
        const double m = std::frexp(v, &e);
        const std::uint32_t biased = static_cast<std::uint32_t>(e - 1 + 15);
        const std::uint32_t mantissa = static_cast<std::uint32_t>(std::lround((2.0 * m - 1.0) * (1 << mantissa_bits)));
        return (biased << mantissa_bits) | mantissa;
    }

    double uf_decode(std::uint32_t bits, int mantissa_bits)
    {
        const int e = static_cast<int>(bits >> mantissa_bits);
        const double m = static_cast<double>(bits & ((1u << mantissa_bits) - 1));
        const double scale = static_cast<double>(1 << mantissa_bits);
        return e == 0 ? std::ldexp(m / scale, -14) : std::ldexp(1.0 + m / scale, e - 15);
    }

    // RGB9_E5: section 3.8.3.2 (encoding) and the matching decode: value = mantissa * 2^(exponent - B - N), B = 15, N = 9.
    struct SharedExp
    {
        std::uint32_t r, g, b, e;
    };

    SharedExp shared_exp_encode(double r, double g, double b)
    {
        constexpr int kN = 9, kB = 15, kEmax = 31;
        const double shared_max = (std::pow(2.0, kN) - 1.0) / std::pow(2.0, kN) * std::pow(2.0, kEmax - kB);
        const auto clamp_c = [&](double c) { return std::max(0.0, std::min(shared_max, c)); };
        const double rc = clamp_c(r), gc = clamp_c(g), bc = clamp_c(b);
        const double max_c = std::max({rc, gc, bc});
        const int exp_p = std::max(-kB - 1, max_c > 0.0 ? static_cast<int>(std::floor(std::log2(max_c))) : -kB - 1) + 1 + kB;
        const double max_s = std::floor(max_c / std::pow(2.0, exp_p - kB - kN) + 0.5);
        const int exp_s = max_s == std::pow(2.0, kN) ? exp_p + 1 : exp_p;
        const auto mantissa = [&](double c) { return static_cast<std::uint32_t>(std::floor(c / std::pow(2.0, exp_s - kB - kN) + 0.5)); };
        return {mantissa(rc), mantissa(gc), mantissa(bc), static_cast<std::uint32_t>(exp_s)};
    }

    double shared_exp_decode(std::uint32_t mantissa, std::uint32_t exponent)
    {
        return std::ldexp(static_cast<double>(mantissa), static_cast<int>(exponent) - 15 - 9);
    }

    // Values that are exactly representable in half float, 11-bit and 10-bit float and (with their neighbours) in a shared
    // exponent: multiples of 1/16 up to 4.
    constexpr std::array<double, 16> kPool{0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    double pool(int texel, int channel) { return kPool[static_cast<std::size_t>((7 * texel + 5 * channel + 3) % 16)]; }

    // An n-bit field value for texel i that covers 0 and the maximum.
    int field(int i, int bits, int mul, int phase)
    {
        const int max = (1 << bits) - 1;
        return i == 0 ? 0 : i == kTexels - 1 ? max : (i * mul + phase) % (max + 1);
    }

    // 8-bit source values for the formats that are uploaded as bytes.
    int src8(int i, int channel)
    {
        switch (channel)
        {
        case 0: return 17 * i;
        case 1: return (53 * i + 7) & 255;
        case 2: return 255 - 17 * i;
        default: return 64 + 8 * i;
        }
    }

    // One FULL step of an n-bit unorm in 8-bit units. Converting 8-bit data into a 5/6/4-bit internal format may land on either of the
    // two nearest representable values: ES 3.0 section 2.1.6.2 defines convert_float_uint as returning "one of the two ... values
    // which are closest to r (where rounding to nearest is preferred)". An implementation that keeps more bits is closer still.
    double lossy_tol(int bits) { return 255.0 / ((1 << bits) - 1); }

    // ---------------------------------------------------------------------------------------------
    // The format cases
    // ---------------------------------------------------------------------------------------------
    struct Case
    {
        std::string name;
        mgl::SizedInternalFormat sized{};
        mgl::PixelFormat format{};
        mgl::PixelType type{};
        std::vector<std::uint8_t> bytes;
        std::array<Color, kTexels> decoded{};  // what texture() must return (before scale and bias), per texel
        Color scale{1, 1, 1, 1};
        Color bias{0, 0, 0, 0};
        Color tol{0, 0, 0, 0};  // allowed deviation in 8-bit units (on top of the 0.5 rounding allowance)
    };

    // The color the default framebuffer must contain for texel i: clamp(decoded * scale + bias) as 8-bit.
    Color expected8(const Case& c, int i)
    {
        Color out{};
        for (std::size_t ch = 0; ch < 4; ++ch)
            out[ch] = 255.0 * std::clamp(c.decoded[static_cast<std::size_t>(i)][ch] * c.scale[ch] + c.bias[ch], 0.0, 1.0);
        return out;
    }

    std::vector<Case> make_cases()
    {
        using F = mgl::PixelFormat;
        using T = mgl::PixelType;
        using S = mgl::SizedInternalFormat;
        std::vector<Case> cases;
        const auto begin = [&](const char* name, S sized, F format, T type) {
            Case c;
            c.name = name;
            c.sized = sized;
            c.format = format;
            c.type = type;
            return c;
        };
        const auto u8 = [](int v) { return static_cast<std::uint8_t>(v); };
        const auto set = [](Case& c, int i, double r, double g, double b, double a) { c.decoded[static_cast<std::size_t>(i)] = {r, g, b, a}; };

        // --- 8-bit unsigned normalized: Table 3.2 gives exactly one pair for each (RED/RG/RGB/RGBA + UNSIGNED_BYTE) ---
        {
            Case c = begin("R8 <- RED / UNSIGNED_BYTE", S::R8, F::Red, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i) { put(c.bytes, u8(src8(i, 0))); set(c, i, src8(i, 0) / 255.0, 0, 0, 1); }
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RG8 <- RG / UNSIGNED_BYTE", S::Rg8, F::Rg, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i)
            {
                put(c.bytes, u8(src8(i, 0)));
                put(c.bytes, u8(src8(i, 1)));
                set(c, i, src8(i, 0) / 255.0, src8(i, 1) / 255.0, 0, 1);
            }
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGB8 <- RGB / UNSIGNED_BYTE", S::Rgb8, F::Rgb, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 3; ++ch) put(c.bytes, u8(src8(i, ch)));
                set(c, i, src8(i, 0) / 255.0, src8(i, 1) / 255.0, src8(i, 2) / 255.0, 1);
            }
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGBA8 <- RGBA / UNSIGNED_BYTE", S::Rgba8, F::Rgba, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 4; ++ch) put(c.bytes, u8(src8(i, ch)));
                set(c, i, src8(i, 0) / 255.0, src8(i, 1) / 255.0, src8(i, 2) / 255.0, src8(i, 3) / 255.0);
            }
            cases.push_back(std::move(c));
        }

        // --- signed normalized: f = max(c / 127, -1) (equation 2.2); mapped into [0, 1] by the shader ---------------
        {
            Case c = begin("R8_SNORM <- RED / BYTE (value * 0.5 + 0.5)", S::R8Snorm, F::Red, T::Byte);
            for (int i = 0; i < kTexels; ++i)
            {
                const int v = -128 + 17 * i;  // -128 ... 127
                put(c.bytes, static_cast<std::int8_t>(v));
                set(c, i, std::max(v / 127.0, -1.0), 0, 0, 1);
            }
            c.scale = {0.5, 1, 1, 1};
            c.bias = {0.5, 0, 0, 0};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }

        // --- 16-bit packed formats ------------------------------------------------------------------------------
        {
            Case c = begin("RGB565 <- RGB / UNSIGNED_SHORT_5_6_5", S::Rgb565, F::Rgb, T::UnsignedShort565);
            for (int i = 0; i < kTexels; ++i)
            {
                const int r = field(i, 5, 7, 3), g = field(i, 6, 11, 5), b = field(i, 5, 5, 9);
                put(c.bytes, static_cast<std::uint16_t>((r << 11) | (g << 5) | b));
                set(c, i, r / 31.0, g / 63.0, b / 31.0, 1);
            }
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGB565 <- RGB / UNSIGNED_BYTE (5/6/5 bits kept)", S::Rgb565, F::Rgb, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 3; ++ch) put(c.bytes, u8(src8(i, ch)));
                set(c, i, src8(i, 0) / 255.0, src8(i, 1) / 255.0, src8(i, 2) / 255.0, 1);
            }
            c.tol = {lossy_tol(5), lossy_tol(6), lossy_tol(5), 0};
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGBA4 <- RGBA / UNSIGNED_SHORT_4_4_4_4", S::Rgba4, F::Rgba, T::UnsignedShort4444);
            for (int i = 0; i < kTexels; ++i)
            {
                const int r = field(i, 4, 3, 1), g = field(i, 4, 5, 2), b = field(i, 4, 7, 3), a = field(i, 4, 9, 4);
                put(c.bytes, static_cast<std::uint16_t>((r << 12) | (g << 8) | (b << 4) | a));
                set(c, i, r / 15.0, g / 15.0, b / 15.0, a / 15.0);
            }
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGBA4 <- RGBA / UNSIGNED_BYTE (4 bits kept)", S::Rgba4, F::Rgba, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 4; ++ch) put(c.bytes, u8(src8(i, ch)));
                set(c, i, src8(i, 0) / 255.0, src8(i, 1) / 255.0, src8(i, 2) / 255.0, src8(i, 3) / 255.0);
            }
            c.tol = {lossy_tol(4), lossy_tol(4), lossy_tol(4), lossy_tol(4)};
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGB5_A1 <- RGBA / UNSIGNED_SHORT_5_5_5_1", S::Rgb5A1, F::Rgba, T::UnsignedShort5551);
            for (int i = 0; i < kTexels; ++i)
            {
                const int r = field(i, 5, 7, 3), g = field(i, 5, 11, 5), b = field(i, 5, 5, 9), a = i & 1;
                put(c.bytes, static_cast<std::uint16_t>((r << 11) | (g << 6) | (b << 1) | a));
                set(c, i, r / 31.0, g / 31.0, b / 31.0, a);
            }
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        {
            // The one alpha bit of RGB5_A1: only 0 and 255 are used as source alpha, so that an implementation that keeps more
            // than one bit (allowed: "match or exceed") produces the same result.
            Case c = begin("RGB5_A1 <- RGBA / UNSIGNED_BYTE (alpha 0 or 255)", S::Rgb5A1, F::Rgba, T::UnsignedByte);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 3; ++ch) put(c.bytes, u8(src8(i, ch)));
                put(c.bytes, u8((i & 1) ? 255 : 0));
                set(c, i, src8(i, 0) / 255.0, src8(i, 1) / 255.0, src8(i, 2) / 255.0, (i & 1) ? 1.0 : 0.0);
            }
            c.tol = {lossy_tol(5), lossy_tol(5), lossy_tol(5), 0};
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGB5_A1 <- RGBA / UNSIGNED_INT_2_10_10_10_REV (alpha 0 or 3)", S::Rgb5A1, F::Rgba, T::UnsignedInt2101010Rev);
            for (int i = 0; i < kTexels; ++i)
            {
                const int r = field(i, 10, 97, 13), g = field(i, 10, 131, 29), b = field(i, 10, 211, 5), a = (i & 1) ? 3 : 0;
                put(c.bytes, static_cast<std::uint32_t>(r | (g << 10) | (b << 20) | (a << 30)));
                set(c, i, r / 1023.0, g / 1023.0, b / 1023.0, a / 3.0);
            }
            c.tol = {lossy_tol(5), lossy_tol(5), lossy_tol(5), 0};
            cases.push_back(std::move(c));
        }
        {
            Case c = begin("RGB10_A2 <- RGBA / UNSIGNED_INT_2_10_10_10_REV", S::Rgb10A2, F::Rgba, T::UnsignedInt2101010Rev);
            for (int i = 0; i < kTexels; ++i)
            {
                const int r = field(i, 10, 97, 13), g = field(i, 10, 131, 29), b = field(i, 10, 211, 5), a = i % 4;
                put(c.bytes, static_cast<std::uint32_t>(r | (g << 10) | (b << 20) | (a << 30)));
                set(c, i, r / 1023.0, g / 1023.0, b / 1023.0, a / 3.0);
            }
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }

        // --- 16-bit floating point (values up to 4 are shown with a scale of 0.25) -----------------------------------
        for (const T type : {T::HalfFloat, T::Float})
        {
            Case c = begin(type == T::HalfFloat ? "R16F <- RED / HALF_FLOAT (value * 0.25)" : "R16F <- RED / FLOAT (value * 0.25)", S::R16F, F::Red, type);
            for (int i = 0; i < kTexels; ++i)
            {
                if (type == T::HalfFloat) put(c.bytes, half_from_float(pool(i, 0))); else put(c.bytes, static_cast<float>(pool(i, 0)));
                set(c, i, pool(i, 0), 0, 0, 1);
            }
            c.scale = {0.25, 1, 1, 1};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        for (const T type : {T::HalfFloat, T::Float})
        {
            Case c = begin(type == T::HalfFloat ? "RGBA16F <- RGBA / HALF_FLOAT (value * 0.25)" : "RGBA16F <- RGBA / FLOAT (value * 0.25)", S::Rgba16F, F::Rgba, type);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 4; ++ch)
                    if (type == T::HalfFloat) put(c.bytes, half_from_float(pool(i, ch))); else put(c.bytes, static_cast<float>(pool(i, ch)));
                set(c, i, pool(i, 0), pool(i, 1), pool(i, 2), pool(i, 3));
            }
            c.scale = {0.25, 0.25, 0.25, 0.25};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }

        // --- packed floating point: R11F_G11F_B10F --------------------------------------------------------------------
        {
            Case c = begin("R11F_G11F_B10F <- RGB / UNSIGNED_INT_10F_11F_11F_REV (value * 0.25)", S::R11FG11FB10F, F::Rgb, T::UnsignedInt10F11F11FRev);
            for (int i = 0; i < kTexels; ++i)
            {
                // exponent fields 12..16, mantissa from the texel index: values between 2^-3 and about 4
                const std::uint32_t r = static_cast<std::uint32_t>(((12 + i % 5) << 6) | ((i * 5) & 63));
                const std::uint32_t g = static_cast<std::uint32_t>(((12 + (i + 1) % 5) << 6) | ((i * 9 + 3) & 63));
                const std::uint32_t b = static_cast<std::uint32_t>(((12 + (i + 3) % 5) << 5) | ((i * 3 + 1) & 31));
                put(c.bytes, r | (g << 11) | (b << 22));
                set(c, i, uf_decode(r, 6), uf_decode(g, 6), uf_decode(b, 5), 1);
            }
            c.scale = {0.25, 0.25, 0.25, 1};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        for (const T type : {T::HalfFloat, T::Float})
        {
            Case c = begin(type == T::HalfFloat ? "R11F_G11F_B10F <- RGB / HALF_FLOAT (value * 0.25)" : "R11F_G11F_B10F <- RGB / FLOAT (value * 0.25)", S::R11FG11FB10F, F::Rgb, type);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 3; ++ch)
                    if (type == T::HalfFloat) put(c.bytes, half_from_float(pool(i, ch))); else put(c.bytes, static_cast<float>(pool(i, ch)));
                // The pool values are exact in 11/10-bit floats, so the stored value is the value.
                set(c, i, uf_decode(uf_encode(pool(i, 0), 6), 6), uf_decode(uf_encode(pool(i, 1), 6), 6), uf_decode(uf_encode(pool(i, 2), 5), 5), 1);
            }
            c.scale = {0.25, 0.25, 0.25, 1};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }

        // --- shared exponent: RGB9_E5 ---------------------------------------------------------------------------------
        {
            Case c = begin("RGB9_E5 <- RGB / UNSIGNED_INT_5_9_9_9_REV (value * 0.5)", S::Rgb9E5, F::Rgb, T::UnsignedInt5999Rev);
            for (int i = 0; i < kTexels; ++i)
            {
                const std::uint32_t r = static_cast<std::uint32_t>(field(i, 9, 37, 11)), g = static_cast<std::uint32_t>(field(i, 9, 91, 3)), b = static_cast<std::uint32_t>(field(i, 9, 53, 29));
                const std::uint32_t e = static_cast<std::uint32_t>(14 + i % 3);  // one exponent for all three components
                put(c.bytes, r | (g << 9) | (b << 18) | (e << 27));
                set(c, i, shared_exp_decode(r, e), shared_exp_decode(g, e), shared_exp_decode(b, e), 1);
            }
            c.scale = {0.5, 0.5, 0.5, 1};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        for (const T type : {T::HalfFloat, T::Float})
        {
            Case c = begin(type == T::HalfFloat ? "RGB9_E5 <- RGB / HALF_FLOAT (value * 0.25)" : "RGB9_E5 <- RGB / FLOAT (value * 0.25)", S::Rgb9E5, F::Rgb, type);
            for (int i = 0; i < kTexels; ++i)
            {
                for (int ch = 0; ch < 3; ++ch)
                    if (type == T::HalfFloat) put(c.bytes, half_from_float(pool(i, ch))); else put(c.bytes, static_cast<float>(pool(i, ch)));
                const SharedExp s = shared_exp_encode(pool(i, 0), pool(i, 1), pool(i, 2));  // the algorithm of section 3.8.3.2
                set(c, i, shared_exp_decode(s.r, s.e), shared_exp_decode(s.g, s.e), shared_exp_decode(s.b, s.e), 1);
            }
            c.scale = {0.25, 0.25, 0.25, 1};
            c.tol = {1, 1, 1, 1};
            cases.push_back(std::move(c));
        }
        return cases;
    }

    // ---------------------------------------------------------------------------------------------
    // Panels
    // ---------------------------------------------------------------------------------------------
    enum class Mode : int { Texture = 0, TextureLod = 1, Size2D = 2, Size3D = 3, SizeArray = 4 };

    struct Panel
    {
        std::string name;
        Mode mode = Mode::Texture;
        mgl::TextureId tex{};  // unit 0 (sampler2D modes), unit 1 (sampler3D), unit 2 (sampler2DArray)
        GLint min_filter = GL_NEAREST;
        GLint base_level = 0;
        float lod_f = 0.0f;
        int lod = 0;
        Color scale{1, 1, 1, 1};
        Color bias{0, 0, 0, 0};
        std::function<Color(double, double)> ref;  // expected 8-bit color at (s, t)
        Color tol{0, 0, 0, 0};
        int boundary = 0;  // NEAREST probes within 0.1 texel of a boundary of this grid are skipped
    };

    struct Probe
    {
        int probes = 0;
        int bad = 0;
        double max_dev = 0.0;
        std::string first;
    };

    bool near_edge(double texel_coord)
    {
        const double f = texel_coord - std::floor(texel_coord);
        return f < 0.1 || f > 0.9;
    }

    // The colors of the mip ladder: every level has its own flat color.
    constexpr std::array<Rgba8, 4> kLadder{{{200, 40, 40, 255}, {40, 200, 40, 255}, {40, 40, 200, 255}, {200, 200, 40, 255}}};
    constexpr Rgba8 kMagenta{220, 40, 200, 255};

    Color to_color(Rgba8 c) { return {double(c[0]), double(c[1]), double(c[2]), double(c[3])}; }

    class ImmutableStorage final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/quad.vert", "shaders/sample.frag");
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glGenFramebuffers(1, &fbo_);

            cases_ = make_cases();
            case_tex_.assign(cases_.size(), mgl::TextureId{});
            mgl::glGenTextures(static_cast<GLsizei>(case_tex_.size()), case_tex_.data());
            build_format_textures();
            build_ladders();
            build_size_textures();
            build_panels();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            sheet_w_ = env.width;
            sheet_h_ = env.height;
            draw_sheet();
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            frame_ = demo::read_rgba8(0, 0, sheet_w_, sheet_h_);

            check_formats(check);
            check_levels(check);
            check_negative_controls(check);
            check_queries(check);
            check_storage_errors(check);
            check_format_validation(check);
            check_updates_on_immutable(check);
            check_cube_and_array_uploads(check);
            check_respecification(check);  // last: it redraws the sheet

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation sequence");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteTextures(static_cast<GLsizei>(case_tex_.size()), case_tex_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(other_tex_.size()), other_tex_.data());
            mgl::glDeleteProgram(program_);
        }

    private:
        static int get_tex_parameter(mgl::TextureTarget target, mgl::TextureParameterQuery pname)
        {
            GLint v = -12345;
            mgl::glGetTexParameteriv(target, pname, &v);
            return v;
        }

        // ------------------------------------------------------------------------------------------
        // Textures (init)
        // ------------------------------------------------------------------------------------------
        void build_format_textures()
        {
            for (std::size_t i = 0; i < cases_.size(); ++i)
            {
                const Case& c = cases_[i];
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, case_tex_[i]);
                (void)demo::drain_gl_errors();
                // ALL of the storage is allocated here, in one call: format, size, one level. No data yet.
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, c.sized, kTexel, kTexel);
                storage_error_.push_back(demo::drain_gl_errors());
                immutable_.push_back({get_tex_parameter(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::ImmutableFormat),
                                      get_tex_parameter(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::ImmutableLevels)});
                (void)demo::drain_gl_errors();
                // Only texel VALUES can be given now: glTexSubImage2D with a valid (format, type) pair of Table 3.2.
                mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kTexel, kTexel, c.format, c.type, c.bytes.data());
                upload_error_.push_back(demo::drain_gl_errors());
            }
        }

        mgl::TextureId new_texture()
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            other_tex_.push_back(t);
            return t;
        }

        void flat_level(mgl::TextureTarget target, int level, int size, Rgba8 color) const
        {
            const std::vector<Rgba8> texels(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), color);
            mgl::glTexSubImage2D(target, level, 0, 0, size, size, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, texels.data());
        }

        void build_ladders()
        {
            // An 8x8 immutable texture with 4 levels (8, 4, 2, 1): every level exists after the single call.
            ladder_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, ladder_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 4, mgl::SizedInternalFormat::Rgba8, 8, 8);
            for (int level = 0; level < 4; ++level) flat_level(mgl::TextureTarget::Texture2D, level, 8 >> level, kLadder[static_cast<std::size_t>(level)]);
            // A sub-rectangle at a nonzero offset of level 1 (4x4): glTexSubImage2D on an immutable texture is allowed.
            const std::array<Rgba8, 4> box{{kMagenta, kMagenta, kMagenta, kMagenta}};
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 1, 1, 1, 2, 2, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, box.data());

            // The same flat colors in ordinary (mutable) textures: one with level 0 only, one with the whole chain.
            mutable_one_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mutable_one_);
            const std::vector<Rgba8> level0(64, kLadder[0]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 8, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, level0.data());

            mutable_full_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mutable_full_);
            for (int level = 0; level < 4; ++level)
            {
                const int size = 8 >> level;
                const std::vector<Rgba8> texels(static_cast<std::size_t>(size * size), kLadder[static_cast<std::size_t>(level)]);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgba8, size, size, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, texels.data());
            }
        }

        void build_size_textures()
        {
            // 7x5 with 3 levels: 7x5, 3x2, 1x1 (width = max(1, floor(width / 2))).
            size2d_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, size2d_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 3, mgl::SizedInternalFormat::Rgba8, 7, 5);
            // 4x2x8 3D with 4 levels: 4x2x8, 2x1x4, 1x1x2, 1x1x1 (floor(log2(8)) + 1 = 4 levels).
            size3d_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, size3d_);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture3D, 4, mgl::SizedInternalFormat::Rgba8, 4, 2, 8);
            // 6x3 array of 5 layers with 3 levels: 6x3x5, 3x1x5, 1x1x5 (the layer count does not shrink).
            size_array_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, size_array_);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 3, mgl::SizedInternalFormat::Rgba8, 6, 3, 5);
        }

        // ------------------------------------------------------------------------------------------
        // Panels
        // ------------------------------------------------------------------------------------------
        void build_panels()
        {
            // ---- panels 0-22: the format cases --------------------------------------------------------------
            for (std::size_t i = 0; i < cases_.size(); ++i)
            {
                const Case& c = cases_[i];
                Panel p;
                p.name = c.name;
                p.tex = case_tex_[i];
                p.scale = c.scale;
                p.bias = c.bias;
                p.tol = c.tol;
                p.boundary = kTexel;
                p.ref = [&c](double s, double t) { return expected8(c, std::min(static_cast<int>(t * kTexel), kTexel - 1) * kTexel + std::min(static_cast<int>(s * kTexel), kTexel - 1)); };
                panels_.push_back(std::move(p));
            }
            first_level_panel_ = panels_.size();

            // ---- the mip ladder: textureLod at exactly chosen levels (NEAREST_MIPMAP_NEAREST) ------------------
            for (int level = 0; level < 4; ++level)
            {
                Panel p;
                p.name = std::format("immutable 8x8 texture, 4 levels: textureLod level {}", level);
                p.mode = Mode::TextureLod;
                p.tex = ladder_;
                p.min_filter = GL_NEAREST_MIPMAP_NEAREST;
                p.lod_f = static_cast<float>(level);
                if (level == 1)
                {
                    p.boundary = 4;  // level 1 is 4x4 with a 2x2 magenta box at (1, 1)
                    p.name += " (green with a magenta box written by glTexSubImage2D)";
                    p.ref = [](double s, double t) {
                        const int x = std::min(static_cast<int>(s * 4), 3), y = std::min(static_cast<int>(t * 4), 3);
                        return to_color((x >= 1 && x <= 2 && y >= 1 && y <= 2) ? kMagenta : kLadder[1]);
                    };
                }
                else
                {
                    const Rgba8 flat = kLadder[static_cast<std::size_t>(level)];
                    p.ref = [flat](double, double) { return to_color(flat); };
                }
                panels_.push_back(std::move(p));
            }
            {
                Panel p;
                p.name = "mutable texture with level 0 only, same filter and textureLod(1.0): incomplete, samples black";
                p.mode = Mode::TextureLod;
                p.tex = mutable_one_;
                p.min_filter = GL_NEAREST_MIPMAP_NEAREST;
                p.lod_f = 1.0f;
                p.ref = [](double, double) { return Color{0, 0, 0, 255}; };
                panels_.push_back(std::move(p));
            }
            {
                Panel p;
                p.name = "immutable texture, TEXTURE_BASE_LEVEL = 100 is clamped to the last level (3): samples its color";
                p.tex = ladder_;
                p.base_level = 100;
                p.ref = [](double, double) { return to_color(kLadder[3]); };
                panels_.push_back(std::move(p));
            }
            {
                Panel p;
                p.name = "mutable texture with 4 levels, TEXTURE_BASE_LEVEL = 100: incomplete, samples black";
                p.tex = mutable_full_;
                p.base_level = 100;
                p.ref = [](double, double) { return Color{0, 0, 0, 255}; };
                panels_.push_back(std::move(p));
            }

            // ---- textureSize of every level ----------------------------------------------------------------------
            const auto size_panel = [&](const char* what, Mode mode, mgl::TextureId tex, int lod, std::array<int, 3> size) {
                Panel p;
                p.name = std::format("textureSize({}, {}) = ({}, {}, {})", what, lod, size[0], size[1], size[2]);
                p.mode = mode;
                p.tex = tex;
                p.lod = lod;
                const Color expected{double(size[0] * 20), double(size[1] * 20), double(size[2] * 20), 255.0};
                p.ref = [expected](double, double) { return expected; };
                panels_.push_back(std::move(p));
            };
            size_panel("immutable 7x5, 3 levels", Mode::Size2D, size2d_, 0, {7, 5, 0});
            size_panel("immutable 7x5, 3 levels", Mode::Size2D, size2d_, 1, {3, 2, 0});
            size_panel("immutable 7x5, 3 levels", Mode::Size2D, size2d_, 2, {1, 1, 0});
            size_panel("immutable 4x2x8 3D, 4 levels", Mode::Size3D, size3d_, 0, {4, 2, 8});
            size_panel("immutable 4x2x8 3D, 4 levels", Mode::Size3D, size3d_, 1, {2, 1, 4});
            size_panel("immutable 4x2x8 3D, 4 levels", Mode::Size3D, size3d_, 2, {1, 1, 2});
            size_panel("immutable 4x2x8 3D, 4 levels", Mode::Size3D, size3d_, 3, {1, 1, 1});
            size_panel("immutable 6x3x5 array, 3 levels", Mode::SizeArray, size_array_, 0, {6, 3, 5});
            size_panel("immutable 6x3x5 array, 3 levels", Mode::SizeArray, size_array_, 1, {3, 1, 5});
            size_panel("immutable 6x3x5 array, 3 levels", Mode::SizeArray, size_array_, 2, {1, 1, 5});
        }

        void draw_sheet()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, sheet_w_, sheet_h_);
            mgl::glClearColor(0.10f, 0.10f, 0.14f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(program_);
            const auto uniform = [&](const char* name) { return mgl::glGetUniformLocation(program_, name); };
            mgl::glUniform1i(uniform("u_tex2d"), 0);  // sampler uniforms hold texture UNIT indices, one unit per sampler type
            mgl::glUniform1i(uniform("u_tex3d"), 1);
            mgl::glUniform1i(uniform("u_tex_array"), 2);

            pw_ = std::min(sheet_w_, sheet_h_) / kGrid;
            for (std::size_t i = 0; i < panels_.size(); ++i)
            {
                const Panel& p = panels_[i];
                mgl::glViewport(static_cast<GLint>(i % kGrid) * pw_, static_cast<GLint>(i / kGrid) * pw_, pw_, pw_);
                bind_for(p);
                mgl::glUniform1i(uniform("u_mode"), static_cast<int>(p.mode));
                mgl::glUniform1f(uniform("u_lod_f"), p.lod_f);
                mgl::glUniform1i(uniform("u_lod"), p.lod);
                mgl::glUniform4f(uniform("u_scale"), static_cast<float>(p.scale[0]), static_cast<float>(p.scale[1]), static_cast<float>(p.scale[2]), static_cast<float>(p.scale[3]));
                mgl::glUniform4f(uniform("u_bias"), static_cast<float>(p.bias[0]), static_cast<float>(p.bias[1]), static_cast<float>(p.bias[2]), static_cast<float>(p.bias[3]));
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }

            mgl::glActiveTexture(mgl::TextureUnit::Texture2);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glViewport(0, 0, sheet_w_, sheet_h_);
        }

        void bind_for(const Panel& p) const
        {
            const bool is_2d = p.mode == Mode::Texture || p.mode == Mode::TextureLod || p.mode == Mode::Size2D;
            mgl::glActiveTexture(mgl::TextureUnit::Texture2);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, p.mode == Mode::SizeArray ? p.tex : mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, p.mode == Mode::Size3D ? p.tex : mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, is_2d ? p.tex : mgl::TextureId{0});
            if (is_2d)
            {
                // The texture parameters of an immutable texture can still be changed (section 3.8.4).
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, p.min_filter);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, GL_NEAREST);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BaseLevel, p.base_level);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Validation helpers
        // ------------------------------------------------------------------------------------------
        [[nodiscard]] Rgba8 pixel(const std::vector<std::uint8_t>& frame, int x, int y) const
        {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(sheet_w_) + static_cast<std::size_t>(x)) * 4;
            return {frame[o], frame[o + 1], frame[o + 2], frame[o + 3]};
        }

        [[nodiscard]] Probe probe(std::size_t idx, const std::function<Color(double, double)>& ref, const Color& tol, int boundary) const
        {
            Probe r;
            const int gx = static_cast<int>(idx) % kGrid, gy = static_cast<int>(idx) / kGrid;
            for (int py = 0; py < pw_; ++py)
                for (int px = 0; px < pw_; ++px)
                {
                    const double s = (px + 0.5) / pw_, t = (py + 0.5) / pw_;
                    if (boundary > 0 && (near_edge(s * boundary) || near_edge(t * boundary))) continue;
                    const Color want = ref(s, t);
                    const Rgba8 got = pixel(frame_, gx * pw_ + px, gy * pw_ + py);
                    bool ok = true;
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        const double dev = std::abs(got[c] - want[c]);
                        r.max_dev = std::max(r.max_dev, dev);
                        if (dev > tol[c] + 0.5) ok = false;
                    }
                    ++r.probes;
                    if (!ok && r.bad++ == 0)
                        r.first = std::format("first at panel pixel ({},{}): got {} want ({:.1f},{:.1f},{:.1f},{:.1f})", px, py, demo::format_rgba(got), want[0], want[1], want[2], want[3]);
                }
            return r;
        }

        void check_panel(demo::Check& check, std::size_t idx, std::string_view what) const
        {
            const Panel& p = panels_[idx];
            const Probe r = probe(idx, p.ref, p.tol, p.boundary);
            check.expect(r.probes >= 100 && r.bad == 0, std::format("panel {}: {}", idx, what),
                         r.bad == 0 ? std::format("{} probes match the CPU model, max deviation {:.2f}/255", r.probes, r.max_dev)
                                    : std::format("{} of {} probes differ, {}", r.bad, r.probes, r.first));
        }

        // Fraction of probes of panel `idx` that do not match `other` (tolerance as the panel's own).
        [[nodiscard]] double mismatch_fraction(std::size_t idx, const std::function<Color(double, double)>& other) const
        {
            const Panel& p = panels_[idx];
            const Probe r = probe(idx, other, p.tol, p.boundary);
            return r.probes > 0 ? static_cast<double>(r.bad) / r.probes : 0.0;
        }

        template <typename F>
        static void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
        {
            (void)demo::drain_gl_errors();  // start from an empty queue so that the error belongs to this call
            call();
            check.expect_gl_error(code, what);
        }

        // ------------------------------------------------------------------------------------------
        // Checks
        // ------------------------------------------------------------------------------------------
        void check_formats(demo::Check& check)
        {
            for (std::size_t i = 0; i < cases_.size(); ++i)
            {
                const Case& c = cases_[i];
                check.expect(storage_error_[i] == mgl::ErrorCode::NoError && immutable_[i].first == GL_TRUE && immutable_[i].second == 1,
                             std::format("{}: glTexStorage2D(1 level) succeeds and TEXTURE_IMMUTABLE_FORMAT = TRUE, TEXTURE_IMMUTABLE_LEVELS = 1", c.name),
                             std::format("error {}, immutable format {}, levels {}", demo::error_name(storage_error_[i]), immutable_[i].first, immutable_[i].second));
                check.expect(upload_error_[i] == mgl::ErrorCode::NoError, std::format("{}: the (format, type) pair of Table 3.2 is accepted by glTexSubImage2D", c.name),
                             std::format("glGetError() = {}", demo::error_name(upload_error_[i])));
                check_panel(check, i, c.name);
            }
        }

        void check_levels(demo::Check& check)
        {
            for (std::size_t i = first_level_panel_; i < panels_.size(); ++i) check_panel(check, i, panels_[i].name);
        }

        void check_negative_controls(demo::Check& check) const
        {
            // The panels of the signed, float and shared-exponent formats must differ from what a wrong decoding would give.
            // R8_SNORM (panel 4): the same bytes read as unsigned normalized.
            const Case& snorm = cases_[4];
            check.expect(mismatch_fraction(4, [&snorm](double s, double t) {
                             const int i = std::min(static_cast<int>(t * kTexel), kTexel - 1) * kTexel + std::min(static_cast<int>(s * kTexel), kTexel - 1);
                             const int v = static_cast<std::int8_t>(snorm.bytes[static_cast<std::size_t>(i)]);
                             return Color{255.0 * std::clamp((v & 255) / 255.0 * 0.5 + 0.5, 0.0, 1.0), 0, 0, 255};
                         }) > 0.5,
                         "negative control: R8_SNORM texels decoded as UNSIGNED normalized would give a different image (the signed decode is observable)");
            // RGB9_E5 (the exact-bits case): decoding the mantissas without the shared exponent.
            const std::size_t e5 = find_case("RGB9_E5 <- RGB / UNSIGNED_INT_5_9_9_9_REV");
            const Case& shared = cases_[e5];
            check.expect(mismatch_fraction(e5, [&shared](double s, double t) {
                             const int i = std::min(static_cast<int>(t * kTexel), kTexel - 1) * kTexel + std::min(static_cast<int>(s * kTexel), kTexel - 1);
                             std::uint32_t bits = 0;
                             std::memcpy(&bits, &shared.bytes[static_cast<std::size_t>(i) * 4], 4);
                             const double r = (bits & 511) / 512.0;  // mantissa as a fraction, exponent ignored
                             return Color{255.0 * std::clamp(r * 0.5, 0.0, 1.0), 0, 0, 255};
                         }) > 0.3,
                         "negative control: RGB9_E5 decoded without the shared exponent would give a different image");
            // R16F with HALF_FLOAT: the same panel without the 0.25 scale of the CPU model (values above 1 clamp).
            const std::size_t r16 = find_case("R16F <- RED / HALF_FLOAT (value * 0.25)");
            const Case& half = cases_[r16];
            check.expect(mismatch_fraction(r16, [&half](double s, double t) {
                             const int i = std::min(static_cast<int>(t * kTexel), kTexel - 1) * kTexel + std::min(static_cast<int>(s * kTexel), kTexel - 1);
                             return Color{255.0 * std::clamp(half.decoded[static_cast<std::size_t>(i)][0], 0.0, 1.0), 0, 0, 255};
                         }) > 0.3,
                         "negative control: the half-float panel does not match the model without its scale (values up to 4 are really stored, not clamped to 1)");
            // RGBA4 read as RGB5_A1: the same bits give a different image.
            const std::size_t rgba4 = find_case("RGBA4 <- RGBA / UNSIGNED_SHORT_4_4_4_4");
            const Case& c4444 = cases_[rgba4];
            check.expect(mismatch_fraction(rgba4, [&c4444](double s, double t) {
                             const int i = std::min(static_cast<int>(t * kTexel), kTexel - 1) * kTexel + std::min(static_cast<int>(s * kTexel), kTexel - 1);
                             std::uint16_t bits = 0;
                             std::memcpy(&bits, &c4444.bytes[static_cast<std::size_t>(i) * 2], 2);
                             return Color{255.0 * ((bits >> 11) & 31) / 31.0, 255.0 * ((bits >> 6) & 31) / 31.0, 255.0 * ((bits >> 1) & 31) / 31.0, 255.0 * (bits & 1)};
                         }) > 0.5,
                         "negative control: RGBA4 bits decoded as 5_5_5_1 would give a different image (the bit layout is observable)");
        }


        [[nodiscard]] std::size_t find_case(std::string_view name) const
        {
            for (std::size_t i = 0; i < cases_.size(); ++i)
                if (cases_[i].name.compare(0, name.size(), name) == 0) return i;
            throw demo::Failure(std::format("internal error: no format case named '{}'", name));
        }

        void check_queries(demo::Check& check)
        {
            using Q = mgl::TextureParameterQuery;
            // A texture made with glTexImage2D is NOT immutable and reports 0 levels (initial values, section 3.8.14).
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mutable_full_);
            check.expect(get_tex_parameter(mgl::TextureTarget::Texture2D, Q::ImmutableFormat) == GL_FALSE && get_tex_parameter(mgl::TextureTarget::Texture2D, Q::ImmutableLevels) == 0,
                         "a texture specified with glTexImage2D: TEXTURE_IMMUTABLE_FORMAT = FALSE, TEXTURE_IMMUTABLE_LEVELS = 0");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, ladder_);
            check.expect(get_tex_parameter(mgl::TextureTarget::Texture2D, Q::ImmutableFormat) == GL_TRUE && get_tex_parameter(mgl::TextureTarget::Texture2D, Q::ImmutableLevels) == 4,
                         "glTexStorage2D(4 levels): TEXTURE_IMMUTABLE_FORMAT = TRUE, TEXTURE_IMMUTABLE_LEVELS = 4");
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, size3d_);
            check.expect(get_tex_parameter(mgl::TextureTarget::Texture3D, Q::ImmutableFormat) == GL_TRUE && get_tex_parameter(mgl::TextureTarget::Texture3D, Q::ImmutableLevels) == 4,
                         "glTexStorage3D(TEXTURE_3D, 4 levels): immutable, 4 levels");
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, size_array_);
            check.expect(get_tex_parameter(mgl::TextureTarget::Texture2DArray, Q::ImmutableFormat) == GL_TRUE && get_tex_parameter(mgl::TextureTarget::Texture2DArray, Q::ImmutableLevels) == 3,
                         "glTexStorage3D(TEXTURE_2D_ARRAY, 3 levels): immutable, 3 levels");

            // The parameters of an immutable texture can be changed and read back.
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, ladder_);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::WrapS, GL_MIRRORED_REPEAT);
            check.expect(get_tex_parameter(mgl::TextureTarget::Texture2D, Q::WrapS) == GL_MIRRORED_REPEAT, "glTexParameteri on an immutable texture is allowed and reads back (WRAP_S = MIRRORED_REPEAT)");
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::WrapS, GL_REPEAT);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, mgl::TextureId{0});
        }

        // Pairs of (levels, size) around the limit floor(log2(max(width, height[, depth]))) + 1 (ES 3.0 section 3.8.4).
        void check_storage_errors(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            using S = mgl::SizedInternalFormat;
            const auto fresh = [&](TT target) {
                const mgl::TextureId t = new_texture();
                mgl::glBindTexture(target, t);
                return t;
            };
            // A refused command has NO effect: afterwards the same texture accepts a valid glTexStorage and becomes immutable.
            const auto still_usable_2d = [&](std::string_view what) {
                (void)demo::drain_gl_errors();
                const bool before = get_tex_parameter(TT::Texture2D, mgl::TextureParameterQuery::ImmutableFormat) == GL_FALSE;
                mgl::glTexStorage2D(TT::Texture2D, 1, S::Rgba8, 4, 4);
                const bool accepted = demo::drain_gl_errors() == mgl::ErrorCode::NoError;
                check.expect(before && accepted && get_tex_parameter(TT::Texture2D, mgl::TextureParameterQuery::ImmutableFormat) == GL_TRUE,
                             std::format("{}: the refused call had no effect (still mutable, a valid glTexStorage2D afterwards succeeds)", what));
            };

            // --- levels versus size, 2D -------------------------------------------------------------------
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage2D 8x8 with 5 levels (> floor(log2(8)) + 1 = 4) -> INVALID_OPERATION", [] { mgl::glTexStorage2D(TT::Texture2D, 5, S::Rgba8, 8, 8); });
            still_usable_2d("glTexStorage2D 8x8, 5 levels");
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::NoError, "glTexStorage2D 8x8 with 4 levels (= floor(log2(8)) + 1) is accepted", [] { mgl::glTexStorage2D(TT::Texture2D, 4, S::Rgba8, 8, 8); });
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage2D 16x4 with 6 levels (the LARGER dimension counts: floor(log2(16)) + 1 = 5) -> INVALID_OPERATION", [] { mgl::glTexStorage2D(TT::Texture2D, 6, S::Rgba8, 16, 4); });
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::NoError, "glTexStorage2D 16x4 with 5 levels is accepted", [] { mgl::glTexStorage2D(TT::Texture2D, 5, S::Rgba8, 16, 4); });
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage2D 1x1 with 2 levels -> INVALID_OPERATION", [] { mgl::glTexStorage2D(TT::Texture2D, 2, S::Rgba8, 1, 1); });
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage2D 5x3 with 4 levels (floor(log2(5)) + 1 = 3) -> INVALID_OPERATION", [] { mgl::glTexStorage2D(TT::Texture2D, 4, S::Rgba8, 5, 3); });
            fresh(TT::TextureCubeMap);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage2D(TEXTURE_CUBE_MAP) 4x4 with 4 levels (> 3) -> INVALID_OPERATION", [] { mgl::glTexStorage2D(TT::TextureCubeMap, 4, S::Rgba8, 4, 4); });

            // --- 3D and array: which dimensions count ---------------------------------------------------------
            fresh(TT::Texture3D);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage3D(TEXTURE_3D) 4x4x16 with 6 levels (the DEPTH counts: floor(log2(16)) + 1 = 5) -> INVALID_OPERATION", [] { mgl::glTexStorage3D(TT::Texture3D, 6, S::Rgba8, 4, 4, 16); });
            fresh(TT::Texture3D);
            expect_error(check, mgl::ErrorCode::NoError, "glTexStorage3D(TEXTURE_3D) 4x4x16 with 5 levels is accepted", [] { mgl::glTexStorage3D(TT::Texture3D, 5, S::Rgba8, 4, 4, 16); });
            fresh(TT::Texture2DArray);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage3D(TEXTURE_2D_ARRAY) 4x4 with 64 layers and 4 levels (the layer count does NOT count: floor(log2(4)) + 1 = 3) -> INVALID_OPERATION", [] { mgl::glTexStorage3D(TT::Texture2DArray, 4, S::Rgba8, 4, 4, 64); });
            fresh(TT::Texture2DArray);
            expect_error(check, mgl::ErrorCode::NoError, "glTexStorage3D(TEXTURE_2D_ARRAY) 4x4x64 with 3 levels is accepted", [] { mgl::glTexStorage3D(TT::Texture2DArray, 3, S::Rgba8, 4, 4, 64); });

            // --- values ------------------------------------------------------------------------------------------
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage2D with levels = 0 -> INVALID_VALUE", [] { mgl::glTexStorage2D(TT::Texture2D, 0, S::Rgba8, 4, 4); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage2D with width = 0 -> INVALID_VALUE", [] { mgl::glTexStorage2D(TT::Texture2D, 1, S::Rgba8, 0, 4); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage2D with height = 0 -> INVALID_VALUE", [] { mgl::glTexStorage2D(TT::Texture2D, 1, S::Rgba8, 4, 0); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage2D with a negative width -> INVALID_VALUE", [] { mgl::glTexStorage2D(TT::Texture2D, 1, S::Rgba8, -4, 4); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage2D with a negative number of levels -> INVALID_VALUE", [] { mgl::glTexStorage2D(TT::Texture2D, -1, S::Rgba8, 4, 4); });
            still_usable_2d("glTexStorage2D with invalid values");
            fresh(TT::Texture3D);
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage3D with depth = 0 -> INVALID_VALUE", [] { mgl::glTexStorage3D(TT::Texture3D, 1, S::Rgba8, 4, 4, 0); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexStorage3D with levels = 0 -> INVALID_VALUE", [] { mgl::glTexStorage3D(TT::Texture3D, 0, S::Rgba8, 4, 4, 4); });

            // --- targets ------------------------------------------------------------------------------------------
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::InvalidEnum, "glTexStorage3D(TEXTURE_2D) -> INVALID_ENUM (target must be TEXTURE_3D or TEXTURE_2D_ARRAY)", [] { mgl::glTexStorage3D(TT::Texture2D, 1, S::Rgba8, 4, 4, 4); });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "glTexStorage3D(TEXTURE_CUBE_MAP) -> INVALID_ENUM", [] { mgl::glTexStorage3D(TT::TextureCubeMap, 1, S::Rgba8, 4, 4, 4); });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "glTexStorage2D(TEXTURE_3D) -> INVALID_ENUM (target must be TEXTURE_2D or TEXTURE_CUBE_MAP)", [] { mgl::glTexStorage2D(TT::Texture3D, 1, S::Rgba8, 4, 4); });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "glTexStorage2D(TEXTURE_2D_ARRAY) -> INVALID_ENUM", [] { mgl::glTexStorage2D(TT::Texture2DArray, 1, S::Rgba8, 4, 4); });

            // --- the default texture object --------------------------------------------------------------------------
            mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage2D with the default texture object (0) bound -> INVALID_OPERATION", [] { mgl::glTexStorage2D(TT::Texture2D, 1, S::Rgba8, 4, 4); });

            // --- depth formats exist only for 2D, array and cube targets -------------------------------------------------
            fresh(TT::Texture2D);
            expect_error(check, mgl::ErrorCode::NoError, "glTexStorage2D with DEPTH_COMPONENT16 is accepted", [] { mgl::glTexStorage2D(TT::Texture2D, 1, S::DepthComponent16, 4, 4); });
            fresh(TT::Texture3D);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexStorage3D(TEXTURE_3D) with DEPTH_COMPONENT16 -> INVALID_OPERATION (section 3.8.3: depth textures are 2D, 2D array or cube)", [] { mgl::glTexStorage3D(TT::Texture3D, 1, S::DepthComponent16, 4, 4, 4); });
            fresh(TT::Texture2DArray);
            expect_error(check, mgl::ErrorCode::NoError, "glTexStorage3D(TEXTURE_2D_ARRAY) with DEPTH_COMPONENT16 is accepted", [] { mgl::glTexStorage3D(TT::Texture2DArray, 1, S::DepthComponent16, 4, 4, 4); });
            mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
            mgl::glBindTexture(TT::Texture3D, mgl::TextureId{0});
            mgl::glBindTexture(TT::Texture2DArray, mgl::TextureId{0});
            mgl::glBindTexture(TT::TextureCubeMap, mgl::TextureId{0});
        }

        // ES 3.0 section 3.8.3: "Specifying a combination of values for format, type, and internalformat that is not listed as a
        // valid combination in tables 3.2 or 3.3 generates the error INVALID_OPERATION."
        void check_format_validation(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            using I = mgl::InternalFormat;
            using F = mgl::PixelFormat;
            using T = mgl::PixelType;
            const mgl::TextureId tex = new_texture();
            mgl::glBindTexture(TT::Texture2D, tex);
            const std::vector<std::uint8_t> zeros(16 * 16, 0);  // large enough for every pair used below (4x4 texels of up to 16 bytes)

            struct Combo
            {
                I internal;
                F format;
                T type;
                const char* what;
                bool valid;
                bool mesa_accepts = false;  // known deviation of Mesa (see below)
            };
            const std::array<Combo, 30> combos{{
                {I::Rgba8, F::Rgba, T::UnsignedByte, "RGBA8 + RGBA + UNSIGNED_BYTE", true},
                {I::Rgba8, F::Rgba, T::UnsignedShort565, "RGBA8 + RGBA + UNSIGNED_SHORT_5_6_5 (packed type has 3 components, format 4)", false},
                {I::Rgba8, F::Rgb, T::UnsignedByte, "RGBA8 + RGB + UNSIGNED_BYTE (an RGBA8 image needs format RGBA)", false},
                {I::Rgba8, F::Rgba, T::Float, "RGBA8 + RGBA + FLOAT (FLOAT is for RGBA32F and RGBA16F)", false},
                {I::Rgba8, F::Rgba, T::UnsignedInt2101010Rev, "RGBA8 + RGBA + UNSIGNED_INT_2_10_10_10_REV (that type is for RGB10_A2 and RGB5_A1)", false},
                {I::Rgba32F, F::Rgba, T::UnsignedByte, "RGBA32F + RGBA + UNSIGNED_BYTE", false},
                {I::Rgba32F, F::Rgba, T::Float, "RGBA32F + RGBA + FLOAT", true},
                {I::R8, F::Red, T::UnsignedByte, "R8 + RED + UNSIGNED_BYTE", true},
                {I::R8UI, F::Red, T::UnsignedByte, "R8UI + RED + UNSIGNED_BYTE (an integer format needs RED_INTEGER)", false},
                {I::R8, F::RedInteger, T::UnsignedByte, "R8 + RED_INTEGER + UNSIGNED_BYTE", false},
                {I::R8UI, F::RedInteger, T::UnsignedByte, "R8UI + RED_INTEGER + UNSIGNED_BYTE", true},
                {I::Rg8, F::Red, T::UnsignedByte, "RG8 + RED + UNSIGNED_BYTE (component count differs)", false},
                {I::R16F, F::Red, T::UnsignedByte, "R16F + RED + UNSIGNED_BYTE", false},
                {I::R16F, F::Red, T::Float, "R16F + RED + FLOAT (accepted: converted to half float)", true},
                {I::Rgb10A2, F::Rgba, T::UnsignedByte, "RGB10_A2 + RGBA + UNSIGNED_BYTE", false},
                {I::Rgb10A2, F::Rgba, T::UnsignedInt2101010Rev, "RGB10_A2 + RGBA + UNSIGNED_INT_2_10_10_10_REV", true},
                {I::Rgb5A1, F::Rgba, T::UnsignedInt2101010Rev, "RGB5_A1 + RGBA + UNSIGNED_INT_2_10_10_10_REV (accepted)", true},
                {I::Rgb9E5, F::Rgb, T::UnsignedByte, "RGB9_E5 + RGB + UNSIGNED_BYTE", false},
                {I::Rgb9E5, F::Rgb, T::Float, "RGB9_E5 + RGB + FLOAT (accepted)", true},
                {I::Rgb565, F::Rgba, T::UnsignedShort565, "RGB565 + RGBA + UNSIGNED_SHORT_5_6_5", false},
                {I::Rgba4, F::Rgba, T::UnsignedShort5551, "RGBA4 + RGBA + UNSIGNED_SHORT_5_5_5_1 (that type is for RGB5_A1)", false},
                {I::Rgba4, F::Rgba, T::UnsignedShort4444, "RGBA4 + RGBA + UNSIGNED_SHORT_4_4_4_4", true},
                {I::Rgba, F::Rgba, T::UnsignedShort4444, "unsized RGBA + RGBA + UNSIGNED_SHORT_4_4_4_4 (Table 3.3)", true},
                {I::Rgba, F::Rgba, T::HalfFloat, "unsized RGBA + RGBA + HALF_FLOAT (not in Table 3.3)", false},
                {I::Rgb, F::Rgb, T::Float, "unsized RGB + RGB + FLOAT (not in Table 3.3)", false, true},
                {I::Rgba, F::Rgba, T::UnsignedShort565, "unsized RGBA + RGBA + UNSIGNED_SHORT_5_6_5", false},
                {I::Luminance, F::Luminance, T::UnsignedByte, "unsized LUMINANCE + LUMINANCE + UNSIGNED_BYTE (Table 3.3)", true},
                {I::Luminance, F::Red, T::UnsignedByte, "unsized LUMINANCE + RED + UNSIGNED_BYTE (the format must be LUMINANCE)", false},
                {I::DepthComponent24, F::DepthComponent, T::UnsignedInt, "DEPTH_COMPONENT24 + DEPTH_COMPONENT + UNSIGNED_INT", true},
                {I::DepthComponent24, F::DepthComponent, T::UnsignedByte, "DEPTH_COMPONENT24 + DEPTH_COMPONENT + UNSIGNED_BYTE", false},
            }};
            for (const Combo& c : combos)
            {
                const std::string what = std::format("glTexImage2D {}: {}", c.what, c.valid ? "valid pair, no error" : "invalid pair -> INVALID_OPERATION");
                if (!c.mesa_accepts)
                {
                    expect_error(check, c.valid ? mgl::ErrorCode::NoError : mgl::ErrorCode::InvalidOperation, what,
                                 [&] { mgl::glTexImage2D(TT::Texture2D, 0, c.internal, 4, 4, 0, c.format, c.type, zeros.data()); });
                    continue;
                }
                // Unsized RGB + RGB + FLOAT is not in core ES 3.0 Table 3.3 (ES 3.2 Table 8.3), but GL_OES_texture_float
                // adds exactly this pair, and an implementation exposing an extension must accept its additions.
                // The expectation therefore follows the context (the extension is not used for anything else).
                const bool oes_float = mgl::HasExtension("GL_OES_texture_float");
                expect_error(check, oes_float ? mgl::ErrorCode::NoError : mgl::ErrorCode::InvalidOperation,
                             std::format("glTexImage2D {}: {}", c.what,
                                         oes_float ? "accepted because GL_OES_texture_float is exposed" : "invalid in core -> INVALID_OPERATION"),
                             [&] { mgl::glTexImage2D(TT::Texture2D, 0, c.internal, 4, 4, 0, c.format, c.type, zeros.data()); });
            }

            // glTexSubImage2D applies the same constraints, against the internal format of the existing level.
            mgl::glTexImage2D(TT::Texture2D, 0, I::Rgba8, 4, 4, 0, F::Rgba, T::UnsignedByte, zeros.data());
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexSubImage2D(RGB + UNSIGNED_BYTE) on an RGBA8 level -> INVALID_OPERATION",
                         [&] { mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, F::Rgb, T::UnsignedByte, zeros.data()); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexSubImage2D(RGBA + FLOAT) on an RGBA8 level -> INVALID_OPERATION",
                         [&] { mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, F::Rgba, T::Float, zeros.data()); });
            expect_error(check, mgl::ErrorCode::NoError, "glTexSubImage2D(RGBA + UNSIGNED_BYTE) on an RGBA8 level is accepted",
                         [&] { mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, F::Rgba, T::UnsignedByte, zeros.data()); });

            // The same rule on an immutable RGB565 texture.
            const mgl::TextureId tex565 = new_texture();
            mgl::glBindTexture(TT::Texture2D, tex565);
            mgl::glTexStorage2D(TT::Texture2D, 1, mgl::SizedInternalFormat::Rgb565, 4, 4);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexSubImage2D(RGBA + UNSIGNED_BYTE) on an immutable RGB565 texture -> INVALID_OPERATION",
                         [&] { mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, F::Rgba, T::UnsignedByte, zeros.data()); });
            expect_error(check, mgl::ErrorCode::NoError, "glTexSubImage2D(RGB + UNSIGNED_BYTE) on an immutable RGB565 texture is accepted (a valid pair for RGB565)",
                         [&] { mgl::glTexSubImage2D(TT::Texture2D, 0, 0, 0, 2, 2, F::Rgb, T::UnsignedByte, zeros.data()); });

            // Depth formats and 3D textures (section 3.8.3).
            const mgl::TextureId tex3d = new_texture();
            mgl::glBindTexture(TT::Texture3D, tex3d);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage3D(TEXTURE_3D) with DEPTH_COMPONENT16 -> INVALID_OPERATION",
                         [&] { mgl::glTexImage3D(TT::Texture3D, 0, I::DepthComponent16, 4, 4, 4, 0, F::DepthComponent, T::UnsignedShort, nullptr); });
            const mgl::TextureId tex_array = new_texture();
            mgl::glBindTexture(TT::Texture2DArray, tex_array);
            expect_error(check, mgl::ErrorCode::NoError, "glTexImage3D(TEXTURE_2D_ARRAY) with DEPTH_COMPONENT16 is accepted",
                         [&] { mgl::glTexImage3D(TT::Texture2DArray, 0, I::DepthComponent16, 4, 4, 4, 0, F::DepthComponent, T::UnsignedShort, nullptr); });
            mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
            mgl::glBindTexture(TT::Texture3D, mgl::TextureId{0});
            mgl::glBindTexture(TT::Texture2DArray, mgl::TextureId{0});
        }

        // glTexSubImage2D, glCopyTexSubImage2D and glTexParameteri are the commands that remain legal on an immutable texture.
        void check_updates_on_immutable(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            const mgl::TextureId tex = new_texture();
            mgl::glBindTexture(TT::Texture2D, tex);
            mgl::glTexStorage2D(TT::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, 4, 4);
            // Fill the default framebuffer with one color and copy a 4x4 rectangle of it.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glClearColor(0.8f, 0.2f, 0.4f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            (void)demo::drain_gl_errors();
            mgl::glCopyTexSubImage2D(TT::Texture2D, 0, 0, 0, 0, 0, 4, 4);
            check.expect_no_gl_error("glCopyTexSubImage2D into an immutable texture is accepted");
            const auto got = read_texture_2d(tex, 4);
            int bad = 0;
            for (std::size_t i = 0; i < got.size(); i += 4)
                if (std::abs(got[i] - 204) > 1 || std::abs(got[i + 1] - 51) > 1 || std::abs(got[i + 2] - 102) > 1 || got[i + 3] != 255) ++bad;
            check.expect(bad == 0, "the copy changed the texels of the immutable texture to the framebuffer color (204, 51, 102, 255)", std::format("{} of 16 texels wrong", bad));
            mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});
            // (the clear above destroyed the contact sheet in the default framebuffer: check_respecification draws it again)
        }

        std::vector<std::uint8_t> read_texture_2d(mgl::TextureId tex, int size)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, tex, 0);
            const auto bytes = demo::read_rgba8(0, 0, size, size);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return bytes;
        }

        // A cube map has six faces with all levels, an array has layers: texel values written with glTexSubImage* are read back.
        void check_cube_and_array_uploads(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            const Rgba8 red{230, 30, 30, 255}, green{30, 220, 40, 255}, blue{40, 60, 240, 255};

            const mgl::TextureId cube = new_texture();
            mgl::glBindTexture(TT::TextureCubeMap, cube);
            mgl::glTexStorage2D(TT::TextureCubeMap, 2, mgl::SizedInternalFormat::Rgba8, 4, 4);
            check.expect(get_tex_parameter(TT::TextureCubeMap, mgl::TextureParameterQuery::ImmutableFormat) == GL_TRUE &&
                             get_tex_parameter(TT::TextureCubeMap, mgl::TextureParameterQuery::ImmutableLevels) == 2,
                         "glTexStorage2D(TEXTURE_CUBE_MAP, 2 levels): immutable, 2 levels");
            const std::vector<Rgba8> level0(16, red), level1(4, green);
            mgl::glTexSubImage2D(TT::TextureCubeMapPositiveX, 0, 0, 0, 4, 4, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, level0.data());
            mgl::glTexSubImage2D(TT::TextureCubeMapNegativeZ, 1, 0, 0, 2, 2, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, level1.data());
            check.expect_no_gl_error("glTexSubImage2D into face +X level 0 and face -Z level 1 of an immutable cube map");
            const auto face = [&](TT target, int level, int size) {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, target, cube, level);
                const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
                const auto bytes = demo::read_rgba8(0, 0, size, size);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, target, mgl::TextureId{0}, 0);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                return std::pair{status, bytes};
            };
            const auto uniform_color = [](const std::vector<std::uint8_t>& bytes, Rgba8 c) {
                for (std::size_t i = 0; i + 3 < bytes.size(); i += 4)
                    if (bytes[i] != c[0] || bytes[i + 1] != c[1] || bytes[i + 2] != c[2] || bytes[i + 3] != c[3]) return false;
                return true;
            };
            const auto px = face(TT::TextureCubeMapPositiveX, 0, 4);
            check.expect(px.first == mgl::FramebufferStatus::Complete && uniform_color(px.second, red), "cube map +X level 0 holds the uploaded texels (read through a framebuffer)");
            const auto nz = face(TT::TextureCubeMapNegativeZ, 1, 2);
            check.expect(nz.first == mgl::FramebufferStatus::Complete && uniform_color(nz.second, green), "cube map -Z level 1 (2x2) holds the uploaded texels: every face has every level");
            const auto last = face(TT::TextureCubeMapNegativeY, 1, 2);
            check.expect(last.first == mgl::FramebufferStatus::Complete, "an unwritten face and level (-Y level 1) of an immutable cube map exists and is framebuffer complete");

            // 2D array: a single layer written with glTexSubImage3D.
            const mgl::TextureId array = new_texture();
            mgl::glBindTexture(TT::Texture2DArray, array);
            mgl::glTexStorage3D(TT::Texture2DArray, 1, mgl::SizedInternalFormat::Rgba8, 4, 4, 6);
            const std::vector<Rgba8> layer(16, blue);
            mgl::glTexSubImage3D(TT::Texture2DArray, 0, 0, 0, 4, 4, 4, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, layer.data());
            check.expect_no_gl_error("glTexSubImage3D into layer 4 of an immutable 6-layer array texture");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, array, 0, 4);
            const auto bytes = demo::read_rgba8(0, 0, 4, 4);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureId{0}, 0, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect(uniform_color(bytes, blue), "layer 4 of the immutable array texture holds the uploaded texels");
            mgl::glBindTexture(TT::TextureCubeMap, mgl::TextureId{0});
            mgl::glBindTexture(TT::Texture2DArray, mgl::TextureId{0});
        }

        // The commands that would change format or size of an immutable texture are refused and change nothing: the contact sheet drawn
        // afterwards is pixel-identical to the one drawn before.
        void check_respecification(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            using I = mgl::InternalFormat;
            using F = mgl::PixelFormat;
            using T = mgl::PixelType;
            using S = mgl::SizedInternalFormat;
            const std::vector<std::uint8_t> data(64 * 64 * 4, 0x55);

            mgl::glBindTexture(TT::Texture2D, ladder_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage2D on an immutable texture with exactly its own format and size -> INVALID_OPERATION",
                         [&] { mgl::glTexImage2D(TT::Texture2D, 0, I::Rgba8, 8, 8, 0, F::Rgba, T::UnsignedByte, data.data()); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage2D on level 1 of an immutable texture with another size and format -> INVALID_OPERATION",
                         [&] { mgl::glTexImage2D(TT::Texture2D, 1, I::Rgb565, 16, 16, 0, F::Rgb, T::UnsignedShort565, data.data()); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "a second glTexStorage2D on an immutable texture -> INVALID_OPERATION",
                         [] { mgl::glTexStorage2D(TT::Texture2D, 2, S::Rgba8, 16, 16); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "a second glTexStorage2D with identical parameters -> INVALID_OPERATION",
                         [] { mgl::glTexStorage2D(TT::Texture2D, 4, S::Rgba8, 8, 8); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glCopyTexImage2D on an immutable texture -> INVALID_OPERATION",
                         [] { mgl::glCopyTexImage2D(TT::Texture2D, 0, I::Rgba8, 0, 0, 8, 8, 0); });
            const std::array<std::uint8_t, 8> etc2_block{};  // one 4x4 ETC2 RGB block
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glCompressedTexImage2D on an immutable texture -> INVALID_OPERATION",
                         [&] { mgl::glCompressedTexImage2D(TT::Texture2D, 0, mgl::CompressedInternalFormat::Rgb8Etc2, 4, 4, 0, 8, etc2_block.data()); });
            check.expect(get_tex_parameter(TT::Texture2D, mgl::TextureParameterQuery::ImmutableLevels) == 4 &&
                             get_tex_parameter(TT::Texture2D, mgl::TextureParameterQuery::ImmutableFormat) == GL_TRUE,
                         "after the refused commands TEXTURE_IMMUTABLE_LEVELS is still 4 and TEXTURE_IMMUTABLE_FORMAT still TRUE");
            mgl::glBindTexture(TT::Texture2D, mgl::TextureId{0});

            mgl::glBindTexture(TT::Texture3D, size3d_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage3D on an immutable 3D texture -> INVALID_OPERATION",
                         [&] { mgl::glTexImage3D(TT::Texture3D, 0, I::Rgba8, 2, 2, 2, 0, F::Rgba, T::UnsignedByte, data.data()); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "a second glTexStorage3D on an immutable 3D texture -> INVALID_OPERATION",
                         [] { mgl::glTexStorage3D(TT::Texture3D, 1, S::Rgba8, 2, 2, 2); });
            mgl::glBindTexture(TT::Texture3D, mgl::TextureId{0});

            mgl::glBindTexture(TT::Texture2DArray, size_array_);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage3D on an immutable array texture -> INVALID_OPERATION",
                         [&] { mgl::glTexImage3D(TT::Texture2DArray, 0, I::Rgba8, 2, 2, 2, 0, F::Rgba, T::UnsignedByte, data.data()); });
            mgl::glBindTexture(TT::Texture2DArray, mgl::TextureId{0});

            // Draw the sheet again: it must be pixel-identical to the one that was drawn (and checked) before any of the refused commands.
            const std::vector<std::uint8_t> before = frame_;
            draw_sheet();
            frame_ = demo::read_rgba8(0, 0, sheet_w_, sheet_h_);
            std::size_t different = 0;
            for (std::size_t i = 0; i < before.size() && i < frame_.size(); ++i)
                if (before[i] != frame_[i]) ++different;
            check.expect(different == 0 && before.size() == frame_.size(),
                         "after all refused commands the contact sheet drawn again is identical to the first one: formats, sizes and texels of the immutable textures are unchanged",
                         std::format("{} bytes differ", different));
        }

        // ------------------------------------------------------------------------------------------
        mgl::ProgramId program_{};
        mgl::FramebufferId fbo_{};
        std::vector<Case> cases_;
        std::vector<mgl::TextureId> case_tex_;
        std::vector<mgl::TextureId> other_tex_;
        std::vector<mgl::ErrorCode> storage_error_, upload_error_;
        std::vector<std::pair<int, int>> immutable_;  // TEXTURE_IMMUTABLE_FORMAT and TEXTURE_IMMUTABLE_LEVELS after glTexStorage2D
        mgl::TextureId ladder_{}, mutable_one_{}, mutable_full_{}, size2d_{}, size3d_{}, size_array_{};
        std::vector<Panel> panels_;
        std::size_t first_level_panel_ = 0;
        int sheet_w_ = 256, sheet_h_ = 256, pw_ = 32;
        std::vector<std::uint8_t> frame_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-immutable-storage",
        .title = "Immutable texture storage and sized formats",
        .min_es = {3, 0},
        .summary = "glTexStorage2D/3D: all levels allocated at once, TEXTURE_IMMUTABLE_FORMAT/LEVELS, respecification errors, level limits, "
                   "and 13 sized internal formats uploaded with their valid format/type pairs and compared with CPU-decoded values.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ImmutableStorage>(argc, argv, info);
}
