// es31-image-load-store
//
// Shader image load/store (OpenGL ES 3.1, ES 3.2 section 8.23 "Texture Image Loads and Stores"):
// a level (or one layer) of an immutable texture is bound to an image unit with glBindImageTexture and a
// shader reads and writes individual texels with imageLoad / imageStore, without a framebuffer.
//
// What the demo does
//   * Format round trip. For every one of the 13 image formats of Table 8.26 (rgba32f rgba16f r32f rgba32ui
//     rgba16ui rgba8ui r32ui rgba32i rgba16i rgba8i r32i rgba8 rgba8_snorm) a compute shader reads a
//     readonly image and writes a transformed copy to a writeonly image. Both textures are then read back
//     through a *sampler* (texelFetch into a shader storage buffer) and compared with a CPU model, word for
//     word (floats, integers) or within the normalization step (rgba8, rgba8_snorm).
//   * Image processing. A 3x3 Gaussian blur and a Sobel edge filter run on a 64 x 64 RGBA8 image and are
//     compared with CPU references over every texel (glReadPixels through a framebuffer object).
//   * Layered and non-layered bindings: layered = TRUE addresses all layers of a 2D array / all slices of a 3D
//     texture with a 3-component coordinate; layered = FALSE binds one layer (or cube face) as an image2D.
//     Table 8.24 (layer number -> cube face) is checked on a cube map.
//   * Level selection (an image is ONE level) and imageSize.
//   * Reading and writing the same texture memory with another format (R32F bits seen as r32ui, RGBA8 bytes
//     seen as one r32ui word and back): the "compatible by size" rule of Table 8.27.
//   * coherent / restrict / readonly / writeonly: a work group communicates through a coherent r32ui image;
//     GLSL ES 3.10 4.9 compile-time errors (store to readonly, load from writeonly, missing readonly/writeonly,
//     missing or mismatched format qualifier) are provoked.
//   * imageStore and imageLoad in a FRAGMENT shader when MAX_FRAGMENT_IMAGE_UNIFORMS > 0.
//   * Image unit state: GetIntegeri_v / GetBooleani_v of IMAGE_BINDING_*, IMAGE_FORMAT_COMPATIBILITY_TYPE,
//     the errors of glBindImageTexture, detach on glDeleteTextures.
//   * Memory barriers between the passes: SHADER_IMAGE_ACCESS (image -> image), TEXTURE_FETCH (image ->
//     sampler), FRAMEBUFFER (image -> ReadPixels), BUFFER_UPDATE (shader -> MapBufferRange).
//
// Image: a 4 x 4 contact sheet that fills the window (each cell is width/4 x height/4 pixels, row 0 on top; a texture
// is stretched over its cell with texelFetch, so every pixel shows one known texel).
//   row 0  source image | blur | Sobel | the source color-inverted by a fragment shader (imageLoad)
//   row 1  layers 0..3 of an 8 x 8 x 4 2D array written with layered = TRUE, layer 2 overwritten through a
//          non-layered binding (blue tint)
//   row 2  slices 0..3 of an 8 x 8 x 4 3D texture, slice 1 overwritten through a non-layered binding
//   row 3  levels 0, 1 (inverted by a compute shader through a level-1 binding), 2 of a 16 x 16 texture |
//          a texture written by a fragment shader (imageStore): a gray ramp 3x + 5y + 1

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <cstdlib>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kImg = 64;  // size of the filtered image
    constexpr int kMinSurface = 64;  // smallest supported surface side: the fragment imageStore pass draws a 64 x 64 viewport

    /// One cell of the 4 x 4 display grid in GL window coordinates. The grid tiles the surface exactly: the cell
    /// boundaries are k * width / 4 and k * height / 4 (k = 0..4), so the picture fills the window at any size.
    struct CellRect
    {
        int x0, y0, w, h;
    };
    CellRect cell_rect(int width, int height, int row, int col)  // row 0 is the top row (GL window y grows upwards)
    {
        const int band = 3 - row;
        const int x0 = col * width / 4, x1 = (col + 1) * width / 4;
        const int y0 = band * height / 4, y1 = (band + 1) * height / 4;
        return {x0, y0, x1 - x0, y1 - y0};
    }
    /// The texel shown by pixel p (0-based inside the cell) of a cell n pixels long for a texture s texels long:
    /// p * s / n, the integer expression of display.frag and frag_load.frag.
    int texel_at(int p, int s, int n) { return p * s / n; }
    /// A pixel near the middle of the pixels that show texel t (when the texture is magnified, that pixel shows t
    /// itself; when it is minified, texel_at() of the result names the neighbour that is shown instead).
    int pixel_for(int t, int s, int n) { return std::min(n - 1, ((2 * t + 1) * n) / (2 * s)); }

    using Words = std::array<std::uint32_t, 4>;
    enum class Kind { Float, Int, Uint };
    enum class Norm { None, Unorm8, Snorm8 };

    float as_float(std::uint32_t b) { return std::bit_cast<float>(b); }
    std::uint32_t bits_of(float f) { return std::bit_cast<std::uint32_t>(f); }

    std::uint32_t hash32(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    std::uint8_t c8(int v) { return static_cast<std::uint8_t>(std::clamp(v, 0, 255)); }
    int max_channel_diff(const Rgba8& a, const Rgba8& b)
    {
        int d = 0;
        for (std::size_t i = 0; i < 4; ++i) d = std::max(d, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        return d;
    }

    // ---------------------------------------------------------------------------------------------
    // The 13 image formats of ES 3.2 Table 8.26.
    // ---------------------------------------------------------------------------------------------
    struct FormatCase
    {
        const char* name;                      // the format layout qualifier, e.g. "rgba8_snorm"
        mgl::SizedInternalFormat storage;      // glTexStorage2D internal format
        mgl::InternalFormat image_format;      // glBindImageTexture format
        Kind kind;
        Norm norm;
        int comps;                             // stored components: 4, or 1 for the r32* formats
        int bits;                              // bits per component
        mgl::PixelFormat pixel_format;         // upload format/type pair (Table 8.27 / ES 3.2 Table 8.3)
        mgl::PixelType pixel_type;
    };

    using SF = mgl::SizedInternalFormat;
    using IF = mgl::InternalFormat;
    using PF = mgl::PixelFormat;
    using PT = mgl::PixelType;
    const std::array<FormatCase, 13> kFormats = {{
        {"rgba32f", SF::Rgba32F, IF::Rgba32F, Kind::Float, Norm::None, 4, 32, PF::Rgba, PT::Float},
        {"rgba16f", SF::Rgba16F, IF::Rgba16F, Kind::Float, Norm::None, 4, 16, PF::Rgba, PT::Float},
        {"r32f", SF::R32F, IF::R32F, Kind::Float, Norm::None, 1, 32, PF::Red, PT::Float},
        {"rgba32ui", SF::Rgba32UI, IF::Rgba32UI, Kind::Uint, Norm::None, 4, 32, PF::RgbaInteger, PT::UnsignedInt},
        {"rgba16ui", SF::Rgba16UI, IF::Rgba16UI, Kind::Uint, Norm::None, 4, 16, PF::RgbaInteger, PT::UnsignedShort},
        {"rgba8ui", SF::Rgba8UI, IF::Rgba8UI, Kind::Uint, Norm::None, 4, 8, PF::RgbaInteger, PT::UnsignedByte},
        {"r32ui", SF::R32UI, IF::R32UI, Kind::Uint, Norm::None, 1, 32, PF::RedInteger, PT::UnsignedInt},
        {"rgba32i", SF::Rgba32I, IF::Rgba32I, Kind::Int, Norm::None, 4, 32, PF::RgbaInteger, PT::Int},
        {"rgba16i", SF::Rgba16I, IF::Rgba16I, Kind::Int, Norm::None, 4, 16, PF::RgbaInteger, PT::Short},
        {"rgba8i", SF::Rgba8I, IF::Rgba8I, Kind::Int, Norm::None, 4, 8, PF::RgbaInteger, PT::Byte},
        {"r32i", SF::R32I, IF::R32I, Kind::Int, Norm::None, 1, 32, PF::RedInteger, PT::Int},
        {"rgba8", SF::Rgba8, IF::Rgba8, Kind::Float, Norm::Unorm8, 4, 8, PF::Rgba, PT::UnsignedByte},
        {"rgba8_snorm", SF::Rgba8Snorm, IF::Rgba8Snorm, Kind::Float, Norm::Snorm8, 4, 8, PF::Rgba, PT::Byte},
    }};

    constexpr int kFmtSize = 8;  // the format textures are 8 x 8
    constexpr int kFmtTexels = kFmtSize * kFmtSize;

    template <typename T>
    void put(std::vector<std::uint8_t>& out, T v)
    {
        std::array<std::uint8_t, sizeof(T)> raw{};
        std::memcpy(raw.data(), &v, sizeof(T));
        out.insert(out.end(), raw.begin(), raw.end());
    }

    std::uint32_t one_of(Kind k) { return k == Kind::Float ? bits_of(1.0f) : 1u; }

    struct Texels
    {
        std::vector<std::uint8_t> bytes;  // what glTexSubImage2D receives
        std::vector<Words> words;         // what imageLoad / texelFetch return, as 32-bit words
    };

    /// Deterministic pseudo-random texel data whose values are exactly representable in the format (and for the
    /// half-float format in the 32-bit float used for the upload), with extreme values in the first texels.
    Texels make_texels(const FormatCase& f, std::uint32_t salt)
    {
        Texels t;
        for (int k = 0; k < kFmtTexels; ++k)
        {
            Words w{0, 0, 0, one_of(f.kind)};
            for (int c = 0; c < f.comps; ++c)
            {
                const std::uint32_t h = hash32(static_cast<std::uint32_t>(k * 4 + c) + salt * 1000u);
                if (f.kind == Kind::Float && f.norm == Norm::None)
                {
                    float v;
                    if (f.bits == 32) v = static_cast<float>(static_cast<int>(h & 0xFFFFFu) - 0x80000) / 1024.0f;  // +-512, steps of 2^-10
                    else v = static_cast<float>(static_cast<int>(h & 0x7FFu) - 0x400) / 64.0f;                       // +-16, exact as a half float
                    put(t.bytes, v);
                    w[static_cast<std::size_t>(c)] = bits_of(v);
                }
                else if (f.norm == Norm::Unorm8)
                {
                    std::uint8_t b = static_cast<std::uint8_t>(h & 255u);
                    if (k == 1) b = 255;
                    if (k == 2) b = 0;
                    put(t.bytes, b);
                    w[static_cast<std::size_t>(c)] = bits_of(static_cast<float>(b) / 255.0f);
                }
                else if (f.norm == Norm::Snorm8)
                {
                    int v = static_cast<int>(h % 255u) - 127;
                    if (k == 1) v = 127;
                    if (k == 2) v = -127;
                    put(t.bytes, static_cast<std::int8_t>(v));
                    w[static_cast<std::size_t>(c)] = bits_of(std::max(static_cast<float>(v) / 127.0f, -1.0f));
                }
                else if (f.kind == Kind::Int)
                {
                    std::int64_t v;
                    if (f.bits == 8) v = static_cast<int>(h & 255u) - 128;
                    else if (f.bits == 16) v = static_cast<int>(h & 0xFFFFu) - 32768;
                    else if (f.comps == 4) v = static_cast<std::int32_t>(h);
                    else v = static_cast<int>(h & 0x1FFFFFFFu) - 0x10000000;  // r32i: x * 2 + 1 must not overflow
                    if (f.comps == 4 && k == 1) v = f.bits == 8 ? 127 : f.bits == 16 ? 32767 : 2147483647;
                    if (f.comps == 4 && k == 2) v = f.bits == 8 ? -128 : f.bits == 16 ? -32768 : -2147483647 - 1;
                    if (f.bits == 8) put(t.bytes, static_cast<std::int8_t>(v));
                    else if (f.bits == 16) put(t.bytes, static_cast<std::int16_t>(v));
                    else put(t.bytes, static_cast<std::int32_t>(v));
                    w[static_cast<std::size_t>(c)] = static_cast<std::uint32_t>(static_cast<std::int32_t>(v));
                }
                else
                {
                    std::uint32_t v = f.bits == 8 ? (h & 255u) : f.bits == 16 ? (h & 0xFFFFu) : h;
                    if (f.comps == 4 && k == 1) v = f.bits == 8 ? 255u : f.bits == 16 ? 65535u : 0xFFFFFFFFu;
                    if (f.comps == 4 && k == 2) v = 0u;
                    if (f.bits == 8) put(t.bytes, static_cast<std::uint8_t>(v));
                    else if (f.bits == 16) put(t.bytes, static_cast<std::uint16_t>(v));
                    else put(t.bytes, v);
                    w[static_cast<std::size_t>(c)] = v;
                }
            }
            t.words.push_back(w);
        }
        return t;
    }

    /// What transform.comp writes for a loaded texel (and what a texelFetch of the destination returns).
    Words transformed(const FormatCase& f, const Words& w)
    {
        if (f.comps == 4) return {w[3], w[2], w[1], w[0]};
        switch (f.kind)
        {
        case Kind::Float: return {bits_of(as_float(w[0]) * 2.0f + 1.0f), 0, 0, bits_of(1.0f)};
        default: return {w[0] * 2u + 1u, 0, 0, 1u};  // int and uint arithmetic wraps modulo 2^32
        }
    }

    struct Compare
    {
        int bad = 0;
        std::string first;
    };

    // Tolerance of the normalized formats, in units of one stored level (1/255 or 1/127). A value that was only UPLOADED
    // and is fetched back is converted by equation 2.1/2.2 (c / (2^b - 1)): exact up to float evaluation, 0.01 level.
    // A value that was STORED by imageStore is converted from float by equation 2.3/2.4 (ES 3.2 section 2.3.5.2): "one of the
    // two integer values closest to f * (2^b - 1) (where rounding to nearest is preferred)", so one level is the bound.
    constexpr double kFetchTolerance = 0.01;
    constexpr double kStoreTolerance = 1.001;

    bool word_equal(const FormatCase& f, std::uint32_t got, std::uint32_t want, double norm_levels)
    {
        if (f.norm != Norm::None)
        {
            const double scale = f.norm == Norm::Unorm8 ? 255.0 : 127.0;
            return std::fabs(static_cast<double>(as_float(got)) * scale - static_cast<double>(as_float(want)) * scale) <= norm_levels;
        }
        if (f.kind == Kind::Float) return as_float(got) == as_float(want);  // numeric: -0 == +0
        return got == want;
    }

    Compare compare_words(const FormatCase& f, const std::vector<std::uint32_t>& got, const std::vector<Words>& want, double norm_levels)
    {
        Compare c;
        for (std::size_t k = 0; k < want.size(); ++k)
            for (std::size_t n = 0; n < 4; ++n)
                if (4 * k + n >= got.size() || !word_equal(f, got[4 * k + n], want[k][n], norm_levels))
                    if (c.bad++ == 0)
                        c.first = std::format("texel {} component {}: got {:#x} expected {:#x}", k, n, 4 * k + n < got.size() ? got[4 * k + n] : 0u, want[k][n]);
        return c;
    }

    // ---------------------------------------------------------------------------------------------
    // The filtered image and its CPU references.
    // ---------------------------------------------------------------------------------------------
    Rgba8 src_pixel(int x, int y)
    {
        Rgba8 c = {c8(x * 4), c8(y * 4), 128, 255};
        const int dx = x - 32, dy = y - 30;
        if (dx * dx + dy * dy < 14 * 14) c = {255, 230, 40, 255};                    // a bright disc
        if (x >= 6 && x < 20 && y >= 40 && y < 58) c = {20, 30, 200, 255};         // a dark blue square
        if (x >= 44 && y < 20 && (((x - 44) / 5 + y / 5) & 1) == 0) c = {240, 240, 240, 255};  // a checker patch
        return c;
    }

    std::vector<Rgba8> make_image()
    {
        std::vector<Rgba8> img(static_cast<std::size_t>(kImg * kImg));
        for (int y = 0; y < kImg; ++y)
            for (int x = 0; x < kImg; ++x) img[static_cast<std::size_t>(y * kImg + x)] = src_pixel(x, y);
        return img;
    }

    const Rgba8& at(const std::vector<Rgba8>& img, int x, int y)
    {
        return img[static_cast<std::size_t>(std::clamp(y, 0, kImg - 1) * kImg + std::clamp(x, 0, kImg - 1))];
    }

    std::vector<Rgba8> blur_reference(const std::vector<Rgba8>& img)
    {
        std::vector<Rgba8> out(img.size());
        for (int y = 0; y < kImg; ++y)
            for (int x = 0; x < kImg; ++x)
            {
                Rgba8 o{};
                for (std::size_t c = 0; c < 4; ++c)
                {
                    double s = 0.0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) s += static_cast<double>((2 - std::abs(dx)) * (2 - std::abs(dy))) * static_cast<double>(at(img, x + dx, y + dy)[c]);
                    o[c] = c8(static_cast<int>(std::lround(s / 16.0)));
                }
                out[static_cast<std::size_t>(y * kImg + x)] = o;
            }
        return out;
    }

    std::vector<Rgba8> sobel_reference(const std::vector<Rgba8>& img)
    {
        auto luma = [&](int x, int y) {
            const Rgba8& c = at(img, x, y);
            return (static_cast<double>(c[0]) * 0.299 + static_cast<double>(c[1]) * 0.587 + static_cast<double>(c[2]) * 0.114) / 255.0;
        };
        std::vector<Rgba8> out(img.size());
        for (int y = 0; y < kImg; ++y)
            for (int x = 0; x < kImg; ++x)
            {
                const double gx = (luma(x + 1, y - 1) + 2.0 * luma(x + 1, y) + luma(x + 1, y + 1)) - (luma(x - 1, y - 1) + 2.0 * luma(x - 1, y) + luma(x - 1, y + 1));
                const double gy = (luma(x - 1, y + 1) + 2.0 * luma(x, y + 1) + luma(x + 1, y + 1)) - (luma(x - 1, y - 1) + 2.0 * luma(x, y - 1) + luma(x + 1, y - 1));
                const double mag = std::clamp(std::sqrt(gx * gx + gy * gy), 0.0, 1.0);
                const auto v = c8(static_cast<int>(std::lround(mag * 255.0)));
                out[static_cast<std::size_t>(y * kImg + x)] = {v, v, v, 255};
            }
        return out;
    }

    // Level-selection textures: texture A holds a different gradient in every level, texture B a flat color.
    constexpr int kLvlSize = 16;
    constexpr int kLvlCount = 5;
    Rgba8 level_a(int level, int x, int y) { return {c8(40 * level + 7 * x), c8(30 * level + 11 * y), c8(255 - 30 * level - 3 * (x + y)), 255}; }
    Rgba8 level_b(int level) { return {c8(200 - 10 * level), c8(100 + level), 50, 255}; }

    // The tinted gray of display.frag for values 1000 * tag + index (modes 1 and 2).
    Rgba8 tagged_color(std::uint32_t v)
    {
        const double idx = static_cast<double>(v % 1000u) / 63.0;
        const std::uint32_t tag = (v / 1000u) % 8u;
        const std::array<double, 3> tint = {static_cast<double>(tag & 1u), static_cast<double>((tag >> 1) & 1u), static_cast<double>((tag >> 2) & 1u)};
        Rgba8 c{0, 0, 0, 255};
        for (std::size_t i = 0; i < 3; ++i) c[i] = c8(static_cast<int>(std::lround(255.0 * (0.5 * idx + 0.5 * tint[i]))));
        return c;
    }

    // Values the layered / non-layered passes write.
    constexpr std::uint32_t kArrBase = 0u;
    constexpr std::uint32_t k3dBase = 50000u;
    constexpr std::uint32_t kSingleBase = 900000u;
    std::uint32_t layered_value(std::uint32_t base, int x, int y, int z) { return base + 1000u * static_cast<std::uint32_t>(z) + static_cast<std::uint32_t>(y * 8 + x); }
    std::uint32_t single_value(int x, int y) { return kSingleBase + static_cast<std::uint32_t>(y * 8 + x); }
    constexpr int kArrSingleLayer = 2;  // layer of the array overwritten through a non-layered binding
    constexpr int k3dSingleSlice = 1;   // slice of the 3D texture overwritten through a non-layered binding
    constexpr int kFragSize = 64;
    std::uint32_t frag_value(int x, int y) { return static_cast<std::uint32_t>(x * 3 + y * 5 + 1); }

    // ---------------------------------------------------------------------------------------------
    // Sampler read-back programs (texelFetch -> shader storage buffer).
    // ---------------------------------------------------------------------------------------------
    enum DumpKind : int { DumpF2, DumpI2, DumpU2, DumpU2A, DumpU3, DumpCount };
    struct DumpDesc
    {
        const char* sampler;
        int dim;
        int kind;  // 0 float, 1 int, 2 uint (the KIND macro of dump.comp)
        mgl::TextureTarget target;
    };
    constexpr std::array<DumpDesc, DumpCount> kDumps = {{
        {"sampler2D", 2, 0, mgl::TextureTarget::Texture2D},
        {"isampler2D", 2, 1, mgl::TextureTarget::Texture2D},
        {"usampler2D", 2, 2, mgl::TextureTarget::Texture2D},
        {"usampler2DArray", 3, 2, mgl::TextureTarget::Texture2DArray},
        {"usampler3D", 3, 2, mgl::TextureTarget::Texture3D},
    }};

    struct DumpProgram
    {
        mgl::ProgramId id{};
        mgl::UniformLocation tex{}, level{}, layer{}, size{};
    };

    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
    constexpr std::size_t kSsboBytes = 128 * 1024;

    // ---------------------------------------------------------------------------------------------
    // Crash isolation (the pattern of tests/negative/driver-crash-readpixels-none). A call that is valid, or
    // that the specification says must fail cleanly, but that crashes or hangs a driver cannot be observed
    // in-process. It runs in a fresh child process instead: this executable re-executed through
    // /proc/self/exe with a hidden first argument. The child creates its own EGL display and context
    // through demo::run (a fork()ed copy of the parent's context is not usable), inherits the environment
    // (driver selection, MESA_GLES_VERSION_OVERRIDE), and prints a marker immediately before and after the
    // isolated call. The parent classifies how the child ended: exit status, signal or timeout, and whether
    // it died between the two markers.
    // ---------------------------------------------------------------------------------------------
    constexpr std::string_view kChildFlag = "--crash-isolated-child";
    constexpr std::string_view kBeforeMarker = "@@ before the isolated call";
    constexpr std::string_view kAfterMarker = "@@ after the isolated call";

    struct ChildResult
    {
        bool spawned = false;
        bool timed_out = false;
        bool exited = false;
        int exit_code = -1;
        bool signaled = false;
        int signal = 0;
        std::string output;

        [[nodiscard]] bool reached_call() const { return output.find(kBeforeMarker) != std::string::npos; }
        [[nodiscard]] bool returned_from_call() const { return output.find(kAfterMarker) != std::string::npos; }
        /// The child passed its own checks.
        [[nodiscard]] bool passed() const { return exited && exit_code == 0 && returned_from_call(); }
        /// The child died (signal or timeout) between the two markers, i.e. inside the isolated call.
        [[nodiscard]] bool died_in_call() const { return (signaled || timed_out) && reached_call() && !returned_from_call(); }
        [[nodiscard]] std::string describe() const
        {
            if (!spawned) return "child could not be spawned";
            const char* where = reached_call() && !returned_from_call() ? " inside the isolated call" : "";
            if (timed_out) return std::format("child timed out{} and was killed", where);
            if (signaled) return std::format("child killed by signal {} ({}){}", signal, sigabbrev_np(signal) ? sigabbrev_np(signal) : "?", where);
            return std::format("child exited with status {}", exit_code);
        }
    };

    /// Environment of the child: ours, plus ASan options that leave SIGSEGV/SIGBUS/SIGABRT to the default action, so that a
    /// crash is reported as a signal also in OGLES3_SANITIZE builds. Harmless without ASan.
    std::vector<std::string> child_environment()
    {
        std::vector<std::string> env;
        std::string asan = "ASAN_OPTIONS=";
        for (char** e = environ; *e != nullptr; ++e)
        {
            const std::string_view kv = *e;
            if (kv.starts_with("ASAN_OPTIONS=")) asan = std::string(kv) + ":";
            else env.emplace_back(kv);
        }
        env.push_back(asan + "handle_segv=0:handle_sigbus=0:handle_abort=0:handle_sigill=0:handle_sigfpe=0");
        return env;
    }

    ChildResult run_child(std::string_view case_name, std::chrono::seconds timeout)
    {
        ChildResult r;
        const std::string name(case_name);
        std::vector<std::string> args = {"/proc/self/exe", std::string(kChildFlag), name};
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        std::vector<std::string> env_strings = child_environment();
        std::vector<char*> envp;
        for (auto& e : env_strings) envp.push_back(e.data());
        envp.push_back(nullptr);

        int fds[2] = {-1, -1};
        if (pipe2(fds, O_CLOEXEC) != 0) return r;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);

        // No core dump for the expected crash (inherited soft limit; restored right after the spawn).
        rlimit core{};
        const bool have_core = getrlimit(RLIMIT_CORE, &core) == 0;
        if (have_core)
        {
            rlimit none = core;
            none.rlim_cur = 0;
            setrlimit(RLIMIT_CORE, &none);
        }
        std::cout << std::flush;
        std::fflush(nullptr);
        pid_t pid = -1;
        const int rc = posix_spawn(&pid, "/proc/self/exe", &actions, nullptr, argv.data(), envp.data());
        if (have_core) setrlimit(RLIMIT_CORE, &core);
        posix_spawn_file_actions_destroy(&actions);
        close(fds[1]);
        if (rc != 0)
        {
            close(fds[0]);
            return r;
        }
        r.spawned = true;

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::array<char, 4096> buf{};
        for (;;)
        {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            if (left.count() <= 0)
            {
                r.timed_out = true;
                kill(pid, SIGKILL);
                break;
            }
            pollfd p{fds[0], POLLIN, 0};
            const int n = poll(&p, 1, static_cast<int>(left.count()));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) continue;
            const ssize_t got = read(fds[0], buf.data(), buf.size());
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;  // EOF: the child (and everything that inherited the pipe) is gone
            r.output.append(buf.data(), static_cast<std::size_t>(got));
        }
        close(fds[0]);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        if (!r.timed_out)
        {
            r.exited = WIFEXITED(status);
            r.exit_code = r.exited ? WEXITSTATUS(status) : -1;
            r.signaled = WIFSIGNALED(status);
            r.signal = r.signaled ? WTERMSIG(status) : 0;
        }

        std::cout << std::format("   --- child '{}' output ---\n", name);
        std::size_t start = 0;
        while (start < r.output.size())
        {
            const std::size_t end = r.output.find('\n', start);
            const std::size_t stop = end == std::string::npos ? r.output.size() : end;
            const std::string_view line = std::string_view(r.output).substr(start, stop - start);
            // "child RESULT:" so that a log scan for the parent's own "RESULT:" line never matches a child.
            std::cout << "   | " << (line.starts_with("RESULT:") ? "child " : "") << line << "\n";
            start = stop + 1;
        }
        std::cout << std::format("   --- child '{}': {} ---\n", name, r.describe());
        return r;
    }

    /// Prints a marker and flushes it to the pipe before the isolated call can crash.
    void marker(std::string_view m) { std::cout << m << std::endl; }


    // Child process: the store through a non-layered binding of one slice of a 3D texture (ES 3.2 8.23: "If layered is
    // FALSE, only the single layer identified by layer will be bound, and the layer is treated as a different texture
    // target for image accesses; [...] three-dimensional [...] texture layers are treated as two-dimensional textures").
    // On ANGLE (Vulkan backend, RADV) this valid store raises a GPU page fault, the context is lost and the next
    // synchronizing call never returns; in-process it would take the whole validation run down.
    constexpr std::string_view kSliceChild = "slice3d-store";
    constexpr std::uint32_t kChildSentinel = 0x5EEDu;

    class SliceStoreChild final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            std::string log;
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, env.load_text("shaders/single_store.comp"), log);
            if (cs.value == 0) throw demo::Failure("single_store.comp failed to compile:\n" + log);
            const std::array<mgl::ShaderId, 1> shaders = {cs};
            prog_ = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            if (prog_.value == 0) throw demo::Failure("single_store.comp failed to link:\n" + log);

            mgl::glGenTextures(1, &tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture3D, 1, mgl::SizedInternalFormat::R32UI, 8, 8, 4);
            const std::vector<std::uint32_t> fill(8 * 8 * 4, kChildSentinel);
            mgl::glTexSubImage3D(mgl::TextureTarget::Texture3D, 0, 0, 0, 0, 8, 8, 4, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, fill.data());
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glGenFramebuffers(1, &fbo_);
        }

        void render(demo::Env& /*env*/, int /*frame*/) override {}

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            mgl::glUseProgram(prog_);
            mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_, 0, GL_FALSE, k3dSingleSlice, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            check.expect_no_gl_error("glBindImageTexture(8x8x4 3D texture, level 0, layered = FALSE, layer = 1, WRITE_ONLY, R32UI)");
            std::array<std::vector<std::uint32_t>, 4> slices;
            marker(kBeforeMarker);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::Framebuffer);  // the slices are read with glReadPixels next
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            for (std::size_t z = 0; z < slices.size(); ++z)
            {
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_, 0, static_cast<GLint>(z));
                slices[z].assign(8 * 8 * 4, 0xDEADBEEFu);
                mgl::glReadPixels(0, 0, 8, 8, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, slices[z].data());
            }
            marker(kAfterMarker);
            check.expect_no_gl_error("glDispatchCompute and the RGBA_INTEGER read-back of the four slices raise no error");
            for (std::size_t z = 0; z < slices.size(); ++z)
            {
                const bool stored = static_cast<int>(z) == k3dSingleSlice;
                int bad = 0;
                std::string first;
                for (int y = 0; y < 8; ++y)
                    for (int x = 0; x < 8; ++x)
                    {
                        const std::uint32_t g = slices[z][static_cast<std::size_t>((y * 8 + x) * 4)], e = stored ? single_value(x, y) : kChildSentinel;
                        if (g != e && bad++ == 0) first = std::format("texel ({},{}) got {} expected {}", x, y, g, e);
                    }
                check.expect(bad == 0, std::format("slice {}: {}", z, stored ? "holds the values stored through the non-layered binding" : "keeps the uploaded sentinel (not bound)"), first);
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindImageTexture(mgl::ImageUnit{0}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteTextures(1, &tex_);
            mgl::glDeleteProgram(prog_);
        }

    private:
        mgl::ProgramId prog_{};
        mgl::TextureId tex_{};
        mgl::FramebufferId fbo_{};
    };


    class ImageLoadStore final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // The fragment imageStore pass writes one texel per pixel of a 64 x 64 viewport of the default framebuffer and
            // the check requires all 4096 writes; pixels outside the surface produce no fragments.
            if (env.width < kMinSurface || env.height < kMinSurface)
                throw demo::Skip(std::format("the surface must be at least {0} x {0} pixels (got {1} x {2})", kMinSurface, env.width, env.height));
            max_image_units_ = demo::get_integer(mgl::GetParameter::MaxImageUnits);
            max_compute_images_ = demo::get_integer(mgl::GetParameter::MaxComputeImageUniforms);
            max_combined_images_ = demo::get_integer(mgl::GetParameter::MaxCombinedImageUniforms);
            max_fragment_images_ = demo::get_integer(mgl::GetParameter::MaxFragmentImageUniforms);
            have_fragment_images_ = max_fragment_images_ > 0;

            // The initial state of every image unit (ES 3.2 Table 21.33), read before anything is bound.
            for (int u = 0; u < std::min(max_image_units_, 32); ++u)
            {
                UnitState s;
                const auto uu = static_cast<GLuint>(u);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, uu, &s.name);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingLevel, uu, &s.level);
                mgl::glGetBooleani_v(mgl::GetParameter::ImageBindingLayered, uu, &s.layered);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingLayer, uu, &s.layer);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingAccess, uu, &s.access);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingFormat, uu, &s.format);
                initial_.push_back(s);
            }

            // Programs.
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                const FormatCase& f = kFormats[i];
                const char* image = f.kind == Kind::Float ? "image2D" : f.kind == Kind::Int ? "iimage2D" : "uimage2D";
                const char* vec = f.kind == Kind::Float ? "vec4" : f.kind == Kind::Int ? "ivec4" : "uvec4";
                prog_transform_[i] = load_compute(env, "shaders/transform.comp",
                                                  std::format("#define FORMAT {}\n#define IMAGE {}\n#define VEC {}\n#define COMPONENTS {}\n", f.name, image, vec, f.comps));
            }
            for (int d = 0; d < DumpCount; ++d)
            {
                const DumpDesc& desc = kDumps[static_cast<std::size_t>(d)];
                DumpProgram& p = dump_[static_cast<std::size_t>(d)];
                p.id = load_compute(env, "shaders/dump.comp", std::format("#define SAMPLER {}\n#define DIM {}\n#define KIND {}\n", desc.sampler, desc.dim, desc.kind));
                p.tex = mgl::glGetUniformLocation(p.id, "u_tex");
                p.level = mgl::glGetUniformLocation(p.id, "u_level");
                p.layer = mgl::glGetUniformLocation(p.id, "u_layer");
                p.size = mgl::glGetUniformLocation(p.id, "u_size");
                mgl::glUseProgram(p.id);
                mgl::glUniform1i(p.tex, 0);
            }
            prog_sizes_ = load_compute(env, "shaders/sizes.comp", {});
            prog_layered_array_ = load_compute(env, "shaders/layered_store.comp", std::format("#define IMAGE uimage2DArray\n#define BASE {}u\n", kArrBase));
            prog_layered_3d_ = load_compute(env, "shaders/layered_store.comp", std::format("#define IMAGE uimage3D\n#define BASE {}u\n", k3dBase));
            prog_single_ = load_compute(env, "shaders/single_store.comp", {});
            prog_face_ = load_compute(env, "shaders/face_color.comp", {});
            prog_blur_ = load_compute(env, "shaders/blur.comp", {});
            prog_sobel_ = load_compute(env, "shaders/sobel.comp", {});
            prog_invert_ = load_compute(env, "shaders/invert.comp", {});
            prog_coherent_[0] = load_compute(env, "shaders/coherent.comp", "#define QUALIFIERS coherent restrict\n");
            prog_coherent_[1] = load_compute(env, "shaders/coherent.comp", "#define QUALIFIERS volatile\n");
            prog_copy_bits_ = load_compute(env, "shaders/copy_bits.comp", {});
            prog_display_ = demo::build_program(env, "shaders/display.vert", "shaders/display.frag");
            if (have_fragment_images_)
            {
                prog_frag_store_ = demo::build_program(env, "shaders/display.vert", "shaders/frag_store.frag");
                prog_frag_load_ = demo::build_program(env, "shaders/display.vert", "shaders/frag_load.frag");
            }

            // Shader storage buffer that receives the sampler read-backs.
            mgl::glGenBuffers(1, &ssbo_);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(kSsboBytes), nullptr, mgl::BufferUsage::DynamicCopy);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            mgl::glGenVertexArrays(1, &vao_);

            // Display program: four samplers on four different units (two sampler types on one unit would be an error).
            mgl::glUseProgram(prog_display_);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_tex2d"), 0);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_uarr"), 1);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_u3d"), 2);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_utex"), 3);

            // The non-layered store into a 3D-texture slice crashes one driver (see SliceStoreChild). It is made first in a
            // child process, on every driver; the parent repeats it for the displayed image only when the child survived it.
            // On ANGLE (Vulkan/RADV) this store raises a GPU page fault and an amdgpu context reset. The GPU is shared with
            // the desktop, so the known-faulting case runs there only on explicit request (OGLES3_RUN_GPU_FAULT_CASES=1).
            const char* renderer = reinterpret_cast<const char*>(mgl::glGetString(mgl::StringName::Renderer));
            const char* opt_in = std::getenv("OGLES3_RUN_GPU_FAULT_CASES");
            slice3d_not_run_ = renderer != nullptr && std::string_view(renderer).find("ANGLE") != std::string_view::npos &&
                               !(opt_in != nullptr && std::string_view(opt_in) == "1");
            if (slice3d_not_run_)
                std::cout << "note: the non-layered 3D-slice store is NOT run on ANGLE (known GPU page fault / context reset, see "
                             "KNOWN_LIMITATIONS); set OGLES3_RUN_GPU_FAULT_CASES=1 to run it\n";
            else
                slice3d_child_ = run_child(kSliceChild, std::chrono::seconds{30});
            slice3d_in_process_ = !slice3d_not_run_ && slice3d_child_.exited;
            if (!slice3d_in_process_)
                std::cout << "note: the non-layered 3D-slice store is not repeated in-process (the child did not survive it); slice "
                          << k3dSingleSlice << " keeps the layered values\n";

            create_textures();
            run_passes();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            win_w_ = env.width;
            win_h_ = env.height;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, win_w_, win_h_);
            mgl::glClearColor(0.10f, 0.10f, 0.14f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glBindVertexArray(vao_);

            // Textures on four units, one per sampler of display.frag.
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_arr_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture2);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_3d_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture3);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_frag_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            mgl::glUseProgram(prog_display_);
            // row 0: source, blur, Sobel (RGBA8, sampler2D)
            show(0, 0, 0, tex_src_, 0, 0);
            show(0, 1, 0, tex_blur_, 0, 0);
            show(0, 2, 0, tex_sobel_, 0, 0);
            // row 1: array layers; row 2: 3D slices
            for (int c = 0; c < 4; ++c)
            {
                show(1, c, 1, mgl::TextureId{}, 0, c);
                show(2, c, 2, mgl::TextureId{}, 0, c);
            }
            // row 3: levels 0, 1, 2 of texture B
            show(3, 0, 0, tex_lvl_b_, 0, 0);
            show(3, 1, 0, tex_lvl_b_, 1, 0);
            show(3, 2, 0, tex_lvl_b_, 2, 0);
            if (have_fragment_images_)
            {
                set_value_scale(1.0f / static_cast<float>(frag_value(kFragSize - 1, kFragSize - 1)), 0.0f);
                show(3, 3, 3, mgl::TextureId{}, 0, 0);

                // row 0, cell 3: imageLoad in a fragment shader (the source image, color-inverted)
                mgl::glUseProgram(prog_frag_load_);
                mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_src_, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
                mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderImageAccess);
                const CellRect c = cell_rect(win_w_, win_h_, 0, 3);
                mgl::glViewport(c.x0, c.y0, c.w, c.h);
                mgl::glUniform2i(mgl::glGetUniformLocation(prog_frag_load_, "u_origin"), c.x0, c.y0);
                mgl::glUniform2i(mgl::glGetUniformLocation(prog_frag_load_, "u_cell"), c.w, c.h);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            for (int u = 0; u < std::min(max_image_units_, 8); ++u)
                mgl::glBindImageTexture(mgl::ImageUnit{static_cast<GLuint>(u)}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (!textures_.empty()) mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
            textures_.clear();
            for (mgl::ProgramId p : programs_) mgl::glDeleteProgram(p);
            programs_.clear();
            if (fbo_.value != 0) mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteBuffers(1, &ssbo_);
            mgl::glDeleteVertexArrays(1, &vao_);
        }

    private:
        struct UnitState
        {
            GLint name = -1, level = -1, layer = -1, access = -1, format = -1;
            GLboolean layered = 2;
        };

        // ---- small helpers -------------------------------------------------------------------------------------
        mgl::ProgramId load_compute(const demo::Env& env, std::string_view file, const std::string& defines)
        {
            std::string src = env.load_text(file);
            if (!defines.empty()) src.insert(src.find('\n') + 1, defines);  // macros go right after "#version 310 es"
            std::string log;
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, src, log);
            if (cs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", file, log));
            const std::array<mgl::ShaderId, 1> shaders = {cs};
            const mgl::ProgramId program = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            if (program.value == 0) throw demo::Failure(std::format("{} failed to link:\n{}", file, log));
            programs_.push_back(program);
            return program;
        }

        mgl::TextureId new_texture_2d(mgl::SizedInternalFormat format, int w, int h, int levels)
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, levels, format, w, h);  // immutable storage: required by glBindImageTexture
            set_nearest(mgl::TextureTarget::Texture2D, levels > 1);
            textures_.push_back(t);
            return t;
        }

        mgl::TextureId new_texture_3d(mgl::TextureTarget target, mgl::SizedInternalFormat format, int w, int h, int d)
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(target, t);
            mgl::glTexStorage3D(target, 1, format, w, h, d);
            set_nearest(target, false);
            textures_.push_back(t);
            return t;
        }

        mgl::TextureId new_texture_cube(int size)
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, t);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, size, size);
            set_nearest(mgl::TextureTarget::TextureCubeMap, false);
            textures_.push_back(t);
            return t;
        }

        static void set_nearest(mgl::TextureTarget target, bool mipmapped)
        {
            mgl::glTexParameter(target, mipmapped ? mgl::TextureMinFilter::NearestMipmapNearest : mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);
        }

        static void upload_2d(mgl::TextureTarget target, mgl::TextureId tex, int level, int w, int h, mgl::PixelFormat pf, mgl::PixelType pt, const void* data)
        {
            mgl::glBindTexture(target, tex);
            mgl::glTexSubImage2D(target, level, 0, 0, w, h, pf, pt, data);
        }

        static void fill_u32(mgl::TextureTarget target, mgl::TextureId tex, int w, int h, int d, std::uint32_t value)
        {
            const std::vector<std::uint32_t> data(static_cast<std::size_t>(w * h * d), value);
            mgl::glBindTexture(target, tex);
            if (d == 1 && target == mgl::TextureTarget::Texture2D)
                mgl::glTexSubImage2D(target, 0, 0, 0, w, h, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, data.data());
            else
                mgl::glTexSubImage3D(target, 0, 0, 0, 0, w, h, d, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, data.data());
        }

        static void bind_image(unsigned unit, mgl::TextureId tex, int level, bool layered, int layer, mgl::ImageAccess access, mgl::InternalFormat format)
        {
            mgl::glBindImageTexture(mgl::ImageUnit{unit}, tex, level, layered ? GL_TRUE : GL_FALSE, layer, access, format);
        }

        void set_value_scale(float scale, float bias)
        {
            mgl::glUniform2f(mgl::glGetUniformLocation(prog_display_, "u_value_scale"), scale, bias);
        }

        /// Draws one display cell: row 0 is the top row (GL window y grows upwards).
        void show(int row, int col, int mode, mgl::TextureId tex2d, int level, int layer)
        {
            if (mode == 0)
            {
                mgl::glActiveTexture(mgl::TextureUnit::Texture0);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex2d);
            }
            const CellRect c = cell_rect(win_w_, win_h_, row, col);
            mgl::glViewport(c.x0, c.y0, c.w, c.h);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_mode"), mode);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_level"), level);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_display_, "u_layer"), layer);
            mgl::glUniform2i(mgl::glGetUniformLocation(prog_display_, "u_origin"), c.x0, c.y0);
            mgl::glUniform2i(mgl::glGetUniformLocation(prog_display_, "u_cell"), c.w, c.h);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        /// Fills the first `words` words of the shader storage buffer with a sentinel (an unwritten word cannot match by
        /// accident) and binds it to SHADER_STORAGE_BUFFER binding point 0.
        void prepare_ssbo(std::size_t words)
        {
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
            const std::vector<std::uint32_t> sentinel(words, 0xDEADBEEFu);
            mgl::glBufferSubData(mgl::BufferTarget::ShaderStorage, 0, std::span<const std::uint32_t>(sentinel));
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, ssbo_);
        }

        /// Reads the first `words` words of the shader storage buffer after a compute pass that wrote them.
        std::vector<std::uint32_t> read_ssbo(std::size_t words)
        {
            // The data is read by MapBufferRange (ES 3.2 section 6.3): BUFFER_UPDATE_BARRIER_BIT.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            std::vector<std::uint32_t> out(words, 0u);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, static_cast<GLsizeiptr>(words * sizeof(std::uint32_t)), mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                std::memcpy(out.data(), mapped, words * sizeof(std::uint32_t));
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage);
            }
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            return out;
        }

        /// Reads a whole texture (level) back through a texelFetch compute shader (one 32-bit word per component).
        std::vector<std::uint32_t> dump(int which, mgl::TextureId tex, int level, int layer, int w, int h)
        {
            const DumpDesc& d = kDumps[static_cast<std::size_t>(which)];
            const DumpProgram& p = dump_[static_cast<std::size_t>(which)];
            const std::size_t words = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u;
            prepare_ssbo(words);
            mgl::glUseProgram(p.id);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(d.target, tex);
            mgl::glUniform1i(p.level, level);
            mgl::glUniform1i(p.layer, layer);
            mgl::glUniform2i(p.size, w, h);
            mgl::glDispatchCompute(static_cast<GLuint>((w + 7) / 8), static_cast<GLuint>((h + 7) / 8), 1);
            return read_ssbo(words);
        }

        /// glReadPixels of level 0 of a 2D texture / one cube face through a framebuffer object.
        std::vector<std::uint8_t> read_texture(mgl::TextureTarget target, mgl::TextureId tex, int w, int h)
        {
            if (fbo_.value == 0) mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, target, tex, 0);
            std::vector<std::uint8_t> bytes;
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete)
                bytes = demo::read_rgba8(0, 0, w, h);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, target, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return bytes;
        }

        // ---- the passes whose results are displayed ------------------------------------------------------------
        void create_textures()
        {
            // Filter chain: source (CPU upload), blur and Sobel outputs.
            tex_src_ = new_texture_2d(mgl::SizedInternalFormat::Rgba8, kImg, kImg, 1);
            tex_blur_ = new_texture_2d(mgl::SizedInternalFormat::Rgba8, kImg, kImg, 1);
            tex_sobel_ = new_texture_2d(mgl::SizedInternalFormat::Rgba8, kImg, kImg, 1);
            const std::vector<Rgba8> img = make_image();
            upload_2d(mgl::TextureTarget::Texture2D, tex_src_, 0, kImg, kImg, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());

            // Layered bindings.
            tex_arr_ = new_texture_3d(mgl::TextureTarget::Texture2DArray, mgl::SizedInternalFormat::R32UI, 8, 8, 4);
            tex_3d_ = new_texture_3d(mgl::TextureTarget::Texture3D, mgl::SizedInternalFormat::R32UI, 8, 8, 4);
            fill_u32(mgl::TextureTarget::Texture2DArray, tex_arr_, 8, 8, 4, 0u);
            fill_u32(mgl::TextureTarget::Texture3D, tex_3d_, 8, 8, 4, 0u);

            // Level selection: A (a gradient per level) and B (flat colors) have all five levels of a 16 x 16 texture.
            tex_lvl_a_ = new_texture_2d(mgl::SizedInternalFormat::Rgba8, kLvlSize, kLvlSize, kLvlCount);
            tex_lvl_b_ = new_texture_2d(mgl::SizedInternalFormat::Rgba8, kLvlSize, kLvlSize, kLvlCount);
            for (int level = 0; level < kLvlCount; ++level)
            {
                const int s = kLvlSize >> level;
                std::vector<Rgba8> a, b;
                for (int y = 0; y < s; ++y)
                    for (int x = 0; x < s; ++x)
                    {
                        a.push_back(level_a(level, x, y));
                        b.push_back(level_b(level));
                    }
                upload_2d(mgl::TextureTarget::Texture2D, tex_lvl_a_, level, s, s, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, a.data());
                upload_2d(mgl::TextureTarget::Texture2D, tex_lvl_b_, level, s, s, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, b.data());
            }

            if (have_fragment_images_)
            {
                tex_frag_ = new_texture_2d(mgl::SizedInternalFormat::R32UI, kFragSize, kFragSize, 1);
                fill_u32(mgl::TextureTarget::Texture2D, tex_frag_, kFragSize, kFragSize, 1, 0xFFFFFFFFu);  // a sentinel no fragment writes
            }
        }

        void run_passes()
        {
            // 1. Blur and Sobel: image -> image. The source was written by glTexSubImage2D, which the GL
            //    synchronizes by itself (section 7.11.2); only shader writes need explicit barriers.
            mgl::glUseProgram(prog_blur_);
            bind_image(0, tex_src_, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
            bind_image(1, tex_blur_, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::Rgba8);
            mgl::glDispatchCompute(kImg / 8, kImg / 8, 1);
            mgl::glUseProgram(prog_sobel_);
            bind_image(1, tex_sobel_, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::Rgba8);
            mgl::glDispatchCompute(kImg / 8, kImg / 8, 1);

            // 2. Layered binding: the whole 2D array / 3D texture with a 3-component coordinate ...
            mgl::glUseProgram(prog_layered_array_);
            bind_image(0, tex_arr_, 0, true, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(1, 1, 4);
            mgl::glUseProgram(prog_layered_3d_);
            bind_image(0, tex_3d_, 0, true, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(1, 1, 4);
            // ... then one layer / slice as an image2D. The second write to the same texels must be ordered after the
            // first: SHADER_IMAGE_ACCESS_BARRIER_BIT (image stores after the barrier do not start before earlier accesses finish).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderImageAccess);
            mgl::glUseProgram(prog_single_);
            bind_image(0, tex_arr_, 0, false, kArrSingleLayer, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(1, 1, 1);
            if (slice3d_in_process_)
            {
                bind_image(0, tex_3d_, 0, false, k3dSingleSlice, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
                mgl::glDispatchCompute(1, 1, 1);
            }

            // 3. Level selection: level 1 of A (8 x 8) inverted into level 1 of B; the other levels of B stay untouched.
            mgl::glUseProgram(prog_invert_);
            bind_image(0, tex_lvl_a_, 1, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
            bind_image(1, tex_lvl_b_, 1, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::Rgba8);
            mgl::glDispatchCompute(1, 1, 1);

            // 4. imageStore in a fragment shader: one write per pixel of a 64 x 64 viewport.
            if (have_fragment_images_)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                mgl::glViewport(0, 0, kFragSize, kFragSize);
                mgl::glUseProgram(prog_frag_store_);
                bind_image(0, tex_frag_, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
                mgl::glBindVertexArray(vao_);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                mgl::glBindVertexArray(mgl::VertexArrayId{0});
            }

            // Everything written above is next read through samplers (TEXTURE_FETCH), by a fragment shader's imageLoad
            // (SHADER_IMAGE_ACCESS) and by glReadPixels through a framebuffer object (FRAMEBUFFER).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch | mgl::MemoryBarrierMask::ShaderImageAccess | mgl::MemoryBarrierMask::Framebuffer);
        }

        // ---- checks (defined below the class) ------------------------------------------------------------------
        void check_limits_and_initial_state(demo::Check& check);
        void check_display(demo::Check& check);
        void check_filters(demo::Check& check);
        void check_formats(demo::Check& check);
        void check_layers(demo::Check& check);
        void check_levels_and_sizes(demo::Check& check);
        void check_reinterpretation(demo::Check& check);
        void check_coherent(demo::Check& check);
        void check_fragment_stage(demo::Check& check);
        void check_binding_queries(demo::Check& check);
        void check_errors(demo::Check& check);
        void check_compile_errors(demo::Check& check);

        int win_w_ = 0, win_h_ = 0;  // surface size of the last render(): the display grid is laid out relative to it
        int max_image_units_ = 0, max_compute_images_ = 0, max_combined_images_ = 0, max_fragment_images_ = 0;
        bool have_fragment_images_ = false;
        std::vector<UnitState> initial_;

        std::vector<mgl::ProgramId> programs_;
        std::vector<mgl::TextureId> textures_;
        std::array<mgl::ProgramId, 13> prog_transform_{};
        std::array<DumpProgram, DumpCount> dump_{};
        std::array<mgl::ProgramId, 2> prog_coherent_{};
        mgl::ProgramId prog_sizes_{}, prog_layered_array_{}, prog_layered_3d_{}, prog_single_{}, prog_face_{}, prog_blur_{}, prog_sobel_{},
            prog_invert_{}, prog_copy_bits_{}, prog_display_{}, prog_frag_store_{}, prog_frag_load_{};
        mgl::BufferId ssbo_{};
        mgl::VertexArrayId vao_{};
        mgl::FramebufferId fbo_{};
        mgl::TextureId tex_src_{}, tex_blur_{}, tex_sobel_{}, tex_arr_{}, tex_3d_{}, tex_lvl_a_{}, tex_lvl_b_{}, tex_frag_{};
        ChildResult slice3d_child_;
        bool slice3d_not_run_ = false;
        bool slice3d_in_process_ = true;  // the 3D slice was overwritten through the non-layered binding in this process

        /// Is layer `l` of the array (row 1) or slice `l` of the 3D texture (row 2) overwritten by the non-layered binding?
        [[nodiscard]] bool overwritten(bool is_3d, int l) const { return is_3d ? slice3d_in_process_ && l == k3dSingleSlice : l == kArrSingleLayer; }
    };

    // ---------------------------------------------------------------------------------------------------------
    // Checks
    // ---------------------------------------------------------------------------------------------------------
    void ImageLoadStore::check_limits_and_initial_state(demo::Check& check)
    {
        std::cout << std::format("note: MAX_IMAGE_UNITS={} MAX_COMPUTE_IMAGE_UNIFORMS={} MAX_COMBINED_IMAGE_UNIFORMS={} MAX_FRAGMENT_IMAGE_UNIFORMS={} MAX_VERTEX_IMAGE_UNIFORMS={}\n",
                                 max_image_units_, max_compute_images_, max_combined_images_, max_fragment_images_,
                                 demo::get_integer(mgl::GetParameter::MaxVertexImageUniforms));
        // ES 3.2 Table 21.40 (ES 3.1 Table 20.43): minimum values.
        check.expect(max_image_units_ >= 4, "MAX_IMAGE_UNITS >= 4", std::format("value {}", max_image_units_));
        check.expect(max_compute_images_ >= 4, "MAX_COMPUTE_IMAGE_UNIFORMS >= 4", std::format("value {}", max_compute_images_));
        check.expect(max_combined_images_ >= 4, "MAX_COMBINED_IMAGE_UNIFORMS >= 4", std::format("value {}", max_combined_images_));
        if (!have_fragment_images_)
            std::cout << "note: MAX_FRAGMENT_IMAGE_UNIFORMS is 0 (allowed by ES 3.1): the fragment-shader image load/store checks are skipped\n";

        // Table 21.33: initial value of every image unit.
        auto all_units = [&](auto&& pred) {
            for (const UnitState& s : initial_)
                if (!pred(s)) return false;
            return !initial_.empty();
        };
        check.expect(all_units([](const UnitState& s) { return s.name == 0; }), "initial IMAGE_BINDING_NAME is 0 on every unit");
        check.expect(all_units([](const UnitState& s) { return s.level == 0; }), "initial IMAGE_BINDING_LEVEL is 0 on every unit");
        check.expect(all_units([](const UnitState& s) { return s.layered == GL_FALSE; }), "initial IMAGE_BINDING_LAYERED is FALSE on every unit");
        check.expect(all_units([](const UnitState& s) { return s.layer == 0; }), "initial IMAGE_BINDING_LAYER is 0 on every unit");
        check.expect(all_units([](const UnitState& s) { return s.access == static_cast<GLint>(mgl::ImageAccess::ReadOnly); }), "initial IMAGE_BINDING_ACCESS is READ_ONLY on every unit");
        check.expect(all_units([](const UnitState& s) { return s.format == static_cast<GLint>(mgl::InternalFormat::R32UI); }), "initial IMAGE_BINDING_FORMAT is R32UI on every unit");
    }

    struct Probe
    {
        int x, y;
        Rgba8 expected;
    };

    /// One check for a set of pixel probes of the default framebuffer (the first failing probe is reported).
    bool probe_cell(demo::Check& check, std::string_view what, const std::vector<Probe>& probes, int tolerance)
    {
        int bad = 0;
        std::string first;
        for (const Probe& p : probes)
        {
            const Rgba8 got = demo::read_pixel(p.x, p.y);
            if (max_channel_diff(got, p.expected) > tolerance && bad++ == 0)
                first = std::format("pixel ({},{}) is {} expected {}", p.x, p.y, demo::format_rgba(got), demo::format_rgba(p.expected));
        }
        return check.expect(bad == 0, what, first);
    }

    Rgba8 inverted(const Rgba8& c) { return {c8(255 - c[0]), c8(255 - c[1]), c8(255 - c[2]), 255}; }
    /// Probes of a cell that shows a size_x x size_y texture: for each wanted texel the pixel near the middle of its
    /// pixel span, with the expected color of the texel that pixel actually shows (the same integer mapping as the
    /// shaders, so the expectation is exact at any cell size; under minification that may be a neighbouring texel).
    template <class F>
    std::vector<Probe> texel_probes(const CellRect& c, int size_x, int size_y, std::span<const std::array<int, 2>> texels, F&& expected)
    {
        std::vector<Probe> probes;
        for (const auto& t : texels)
        {
            const int px = pixel_for(t[0], size_x, c.w), py = pixel_for(t[1], size_y, c.h);
            probes.push_back({c.x0 + px, c.y0 + py, expected(texel_at(px, size_x, c.w), texel_at(py, size_y, c.h))});
        }
        return probes;
    }

    void ImageLoadStore::check_display(demo::Check& check)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        const std::vector<Rgba8> img = make_image();
        const std::vector<Rgba8> blur = blur_reference(img);
        const std::vector<Rgba8> sobel = sobel_reference(img);
        const std::array<std::array<int, 2>, 8> pts = {{{10, 10}, {32, 30}, {12, 48}, {50, 8}, {45, 30}, {31, 16}, {19, 45}, {60, 60}}};
        auto cell = [&](int row, int col) { return cell_rect(win_w_, win_h_, row, col); };

        // Row 0: 64 x 64 images. The display tolerance covers the 8-bit output quantization (+-1); the blur and
        // Sobel textures were themselves rounded to 8 bits by imageStore (+-1 against the CPU value).
        auto image_cell = [&](int col, const std::vector<Rgba8>& ref, bool invert, std::string_view what, int tol) {
            probe_cell(check, what, texel_probes(cell(0, col), kImg, kImg, pts, [&](int x, int y) {
                           const Rgba8& e = ref[static_cast<std::size_t>(y * kImg + x)];
                           return invert ? inverted(e) : e;
                       }),
                       tol);
        };
        image_cell(0, img, false, "cell (0,0): the uploaded source image, fetched by a sampler, matches the CPU image at 8 probes", 1);
        image_cell(1, blur, false, "cell (0,1): the blur output texture, fetched by a sampler, matches the CPU blur at 8 probes", 2);
        image_cell(2, sobel, false, "cell (0,2): the Sobel output texture, fetched by a sampler, matches the CPU Sobel at 8 probes", 2);
        if (have_fragment_images_)
            image_cell(3, img, true, "cell (0,3): imageLoad in a fragment shader shows the inverted source at 8 probes", 1);

        // Negative control: at the edges (disc, square, checker) the blur is far from the unfiltered image, so a probe
        // that passes against the blur must fail against the source. Of the texels the blur cell shows (one candidate
        // pixel per texel row and column, see pixel_for), the one with the largest blur/source difference is probed. A
        // search instead of a fixed texel: a minified cell does not show every texel, and may skip a given edge texel.
        {
            const CellRect c = cell(0, 1);
            int best_px = 0, best_py = 0, best_diff = -1;
            std::size_t k = 0;
            for (int ty = 0; ty < kImg; ++ty)
                for (int tx = 0; tx < kImg; ++tx)
                {
                    const int px = pixel_for(tx, kImg, c.w), py = pixel_for(ty, kImg, c.h);
                    const auto kk = static_cast<std::size_t>(texel_at(py, kImg, c.h) * kImg + texel_at(px, kImg, c.w));
                    if (const int d = max_channel_diff(img[kk], blur[kk]); d > best_diff)
                    {
                        best_diff = d;
                        best_px = px;
                        best_py = py;
                        k = kk;
                    }
                }
            const Rgba8 got = demo::read_pixel(c.x0 + best_px, c.y0 + best_py);
            check.expect(max_channel_diff(img[k], blur[k]) > 8, "negative control setup: at the probed texel the CPU blur differs from the CPU source by more than 8 levels",
                         std::format("texel ({},{}) source {} blur {}", k % kImg, k / kImg, demo::format_rgba(img[k]), demo::format_rgba(blur[k])));
            check.expect(max_channel_diff(got, img[k]) > 8, "negative control: the blur cell at that texel does NOT match the unfiltered source (the probe can fail)",
                         std::format("pixel {} source {}", demo::format_rgba(got), demo::format_rgba(img[k])));
        }

        // Rows 1 and 2: 8 x 8 textures; tagged_color() is the display function of display.frag.
        const std::array<std::array<int, 2>, 5> tex_pts = {{{1, 1}, {6, 2}, {3, 6}, {7, 7}, {0, 0}}};
        for (int l = 0; l < 4; ++l)
        {
            for (int row = 1; row <= 2; ++row)
            {
                const bool single = overwritten(row == 2, l);
                const std::vector<Probe> probes = texel_probes(cell(row, l), 8, 8, tex_pts, [&](int x, int y) {
                    return tagged_color(single ? single_value(x, y) : layered_value(row == 1 ? kArrBase : k3dBase, x, y, l));
                });
                probe_cell(check,
                           std::format("cell ({},{}): {} {} shows the values of {}", row, l, row == 1 ? "array layer" : "3D slice", l,
                                       single ? "the non-layered binding (layered = FALSE)" : "the layered binding (layered = TRUE)"),
                           probes, 1);
            }
        }

        // Row 3: levels 0 (16 x 16), 1 (8 x 8) and 2 (4 x 4) of texture B, then the 64 x 64 fragment-store texture.
        {
            const std::array<std::array<int, 2>, 3> flat_pts = {{{0, 0}, {7, 10}, {15, 2}}};
            const std::array<std::array<int, 2>, 3> flat_pts2 = {{{0, 0}, {1, 2}, {3, 3}}};
            probe_cell(check, "cell (3,0): level 0 of texture B is untouched by the level-1 binding",
                       texel_probes(cell(3, 0), 16, 16, flat_pts, [](int, int) { return level_b(0); }), 1);
            const std::array<std::array<int, 2>, 4> inv_pts = {{{0, 0}, {3, 5}, {7, 7}, {5, 1}}};
            probe_cell(check, "cell (3,1): level 1 of texture B holds the inverted level 1 of texture A (image bound with level = 1)",
                       texel_probes(cell(3, 1), 8, 8, inv_pts, [](int x, int y) { return inverted(level_a(1, x, y)); }), 1);
            probe_cell(check, "cell (3,2): level 2 of texture B is untouched by the level-1 binding",
                       texel_probes(cell(3, 2), 4, 4, flat_pts2, [](int, int) { return level_b(2); }), 1);
            if (have_fragment_images_)
            {
                const double top = static_cast<double>(frag_value(kFragSize - 1, kFragSize - 1));
                probe_cell(check, "cell (3,3): the texture written by imageStore in a fragment shader shows the ramp 3x + 5y + 1",
                           texel_probes(cell(3, 3), kFragSize, kFragSize, pts, [&](int x, int y) {
                               const auto g = c8(static_cast<int>(std::lround(255.0 * static_cast<double>(frag_value(x, y)) / top)));
                               return Rgba8{g, g, g, 255};
                           }),
                           1);
            }
        }
    }

    void ImageLoadStore::check_filters(demo::Check& check)
    {
        const std::vector<Rgba8> img = make_image();
        auto compare = [&](std::string_view name, mgl::TextureId tex, const std::vector<Rgba8>& want) {
            const std::vector<std::uint8_t> got = read_texture(mgl::TextureTarget::Texture2D, tex, kImg, kImg);
            int max_diff = 0, bad = 0;
            std::string first;
            if (got.size() != want.size() * 4) bad = -1;
            for (std::size_t i = 0; bad >= 0 && i < want.size(); ++i)
                for (std::size_t c = 0; c < 4; ++c)
                {
                    const int d = std::abs(static_cast<int>(got[4 * i + c]) - static_cast<int>(want[i][c]));
                    max_diff = std::max(max_diff, d);
                    if (d > 1 && bad++ == 0) first = std::format("texel ({},{}) channel {}: got {} expected {}", i % kImg, i / kImg, c, got[4 * i + c], want[i][c]);
                }
            std::cout << std::format("note: {} filter: max |GPU - CPU| = {} over {} texels\n", name, max_diff, want.size());
            check.expect(bad == 0, std::format("{} output (imageLoad x9 / imageStore, glReadPixels) equals the CPU reference at all {} texels within 1", name, want.size()),
                         first.empty() ? std::format("read {} bytes", got.size()) : first);
            return got;
        };
        // Tolerance: imageStore converts the float result to RGBA8 by rounding to nearest (ES 3.2 8.23: as for TexImage2D with
        // RGBA/FLOAT, section 2.3.5); the CPU rounds the exact value, so ties and float evaluation order may differ by one level.
        const std::vector<std::uint8_t> blur = compare("blur 3x3", tex_blur_, blur_reference(img));
        compare("Sobel", tex_sobel_, sobel_reference(img));

        // Negative control: the filter did something. The unfiltered image would fail the same comparison.
        int max_change = 0;
        for (std::size_t i = 0; i < img.size(); ++i)
            for (std::size_t c = 0; c < 3; ++c) max_change = std::max(max_change, std::abs(static_cast<int>(blur[4 * i + c]) - static_cast<int>(img[i][c])));
        check.expect(max_change > 8, "negative control: the blurred image differs from the unfiltered image (the filter comparison can fail)", std::format("max change {}", max_change));
    }

    void ImageLoadStore::check_formats(demo::Check& check)
    {
        bool controlled_norm = false, controlled_int = false;
        for (std::size_t i = 0; i < kFormats.size(); ++i)
        {
            const FormatCase& f = kFormats[i];
            const Texels src = make_texels(f, static_cast<std::uint32_t>(i) + 1u);
            const mgl::TextureId tsrc = new_texture_2d(f.storage, kFmtSize, kFmtSize, 1);
            const mgl::TextureId tdst = new_texture_2d(f.storage, kFmtSize, kFmtSize, 1);
            upload_2d(mgl::TextureTarget::Texture2D, tsrc, 0, kFmtSize, kFmtSize, f.pixel_format, f.pixel_type, src.bytes.data());

            mgl::glUseProgram(prog_transform_[i]);
            bind_image(0, tsrc, 0, false, 0, mgl::ImageAccess::ReadOnly, f.image_format);
            bind_image(1, tdst, 0, false, 0, mgl::ImageAccess::WriteOnly, f.image_format);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch);  // the destination is read through a sampler next

            const int dk = f.kind == Kind::Float ? DumpF2 : f.kind == Kind::Int ? DumpI2 : DumpU2;
            std::vector<Words> want;
            for (const Words& w : src.words) want.push_back(transformed(f, w));
            const std::vector<std::uint32_t> got_dst = dump(dk, tdst, 0, 0, kFmtSize, kFmtSize);
            const std::vector<std::uint32_t> got_src = dump(dk, tsrc, 0, 0, kFmtSize, kFmtSize);

            const Compare cd = compare_words(f, got_dst, want, kStoreTolerance);
            check.expect(cd.bad == 0, std::format("{}: imageStore of the transformed imageLoad result lands in the destination ({} texels, {})", f.name, kFmtTexels,
                                                  f.comps == 4 ? "components reversed" : "x * 2 + 1"),
                         cd.first);
            const Compare cs = compare_words(f, got_src, src.words, kFetchTolerance);
            check.expect(cs.bad == 0, std::format("{}: the readonly source image is unchanged", f.name), cs.first);

            // Negative controls: a perturbed model must be rejected by the same comparison.
            const bool want_control = (f.norm == Norm::Unorm8 && !controlled_norm) || (f.name == std::string_view("r32i") && !controlled_int);
            if (want_control)
            {
                std::vector<Words> wrong = want;
                wrong[5][1] = f.kind == Kind::Float ? bits_of(as_float(wrong[5][1]) + 1.0f) : wrong[5][1] ^ 1u;
                check.expect(compare_words(f, got_dst, wrong, kStoreTolerance).bad > 0, std::format("negative control ({}): a model with one perturbed word fails the comparison", f.name));
                (f.norm == Norm::Unorm8 ? controlled_norm : controlled_int) = true;
            }
        }
    }

    void ImageLoadStore::check_layers(demo::Check& check)
    {
        auto compare_layer = [&](std::string_view what, int which, mgl::TextureId tex, int layer, auto&& value) {
            const std::vector<std::uint32_t> got = dump(which, tex, 0, layer, 8, 8);
            int bad = 0;
            std::string first;
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x)
                {
                    const std::uint32_t g = got[static_cast<std::size_t>((y * 8 + x) * 4)], e = value(x, y);
                    if (g != e && bad++ == 0) first = std::format("texel ({},{}) got {} expected {}", x, y, g, e);
                }
            check.expect(bad == 0, what, first);
        };
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch);
        for (int l = 0; l < 4; ++l)
        {
            const bool single = l == kArrSingleLayer;
            compare_layer(std::format("2D array layer {}: {}", l, single ? "overwritten by the non-layered binding (layered = FALSE, layer = 2)" : "written by the layered binding (layered = TRUE, z = layer)"),
                          DumpU2A, tex_arr_, l, [&](int x, int y) { return single ? single_value(x, y) : layered_value(kArrBase, x, y, l); });
        }
        for (int l = 0; l < 4; ++l)
        {
            const bool single = overwritten(true, l);
            compare_layer(std::format("3D texture slice {}: {}", l, single ? "overwritten by the non-layered binding (layered = FALSE, layer = 1)" : "written by the layered binding (layered = TRUE, z = slice)"),
                          DumpU3, tex_3d_, l, [&](int x, int y) { return single ? single_value(x, y) : layered_value(k3dBase, x, y, l); });
        }

        // Cube map: with layered = FALSE the layer number selects the face (Table 8.24). Layers 3 and 5 are bound, so
        // NEGATIVE_Y and NEGATIVE_Z receive the gradient; the faces between and before them stay black.
        const mgl::TextureId cube = new_texture_cube(8);
        const std::array<mgl::TextureTarget, 6> faces = {mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
                                                         mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
                                                         mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
        const std::vector<Rgba8> black(64, Rgba8{0, 0, 0, 255});
        mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube);
        for (const mgl::TextureTarget face : faces) mgl::glTexSubImage2D(face, 0, 0, 0, 8, 8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, black.data());
        mgl::glUseProgram(prog_face_);
        for (const int layer : {3, 5})
        {
            bind_image(0, cube, 0, false, layer, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::Rgba8);
            mgl::glDispatchCompute(1, 1, 1);
        }
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::Framebuffer);  // the faces are read with glReadPixels next
        for (int f = 0; f < 6; ++f)
        {
            const std::vector<std::uint8_t> bytes = read_texture(faces[static_cast<std::size_t>(f)], cube, 8, 8);
            const bool written = f == 3 || f == 5;
            int bad = 0;
            std::string first;
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x)
                {
                    const Rgba8 e = written ? Rgba8{c8(static_cast<int>(std::lround(255.0 * x / 7.0))), c8(static_cast<int>(std::lround(255.0 * y / 7.0))), 128, 255} : Rgba8{0, 0, 0, 255};
                    const std::size_t o = static_cast<std::size_t>((y * 8 + x) * 4);
                    if (bytes.size() < o + 4) { bad++; continue; }
                    const Rgba8 g = {bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]};
                    if (max_channel_diff(g, e) > 1 && bad++ == 0) first = std::format("texel ({},{}) is {} expected {}", x, y, demo::format_rgba(g), demo::format_rgba(e));
                }
            static constexpr const char* kFaceNames[6] = {"POSITIVE_X", "NEGATIVE_X", "POSITIVE_Y", "NEGATIVE_Y", "POSITIVE_Z", "NEGATIVE_Z"};
            check.expect(bad == 0, std::format("cube face {} ({}): {}", f, kFaceNames[f], written ? "layer number selects this face (Table 8.24): gradient written" : "not selected by any binding: still black"), first);
        }
    }

    void ImageLoadStore::check_levels_and_sizes(demo::Check& check)
    {
        // Texture B: only level 1 was touched by the level-1 image binding.
        for (int level = 0; level < kLvlCount; ++level)
        {
            const int s = kLvlSize >> level;
            const std::vector<std::uint32_t> got = dump(DumpF2, tex_lvl_b_, level, 0, s, s);
            int bad = 0;
            std::string first;
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x)
                {
                    const Rgba8 e = level == 1 ? inverted(level_a(1, x, y)) : level_b(level);
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        const double g = static_cast<double>(as_float(got[static_cast<std::size_t>((y * s + x) * 4) + c])) * 255.0;
                        // Level 1 was stored by imageStore (float -> unorm, +-1 level, equation 2.3); the other levels were uploaded.
                        if (std::fabs(g - static_cast<double>(e[c])) > (level == 1 ? kStoreTolerance : kFetchTolerance) && bad++ == 0)
                            first = std::format("texel ({},{}) channel {}: got {} expected {}", x, y, c, g, e[c]);
                    }
                }
            check.expect(bad == 0, std::format("level {} ({}x{}) of texture B {}", level, s, s, level == 1 ? "holds the inverted level 1 of A (the only level an image bound with level = 1 can touch)" : "is unchanged"), first);
        }

        // imageSize of images bound to a level / all layers / one layer.
        const mgl::TextureId tex_size = new_texture_2d(mgl::SizedInternalFormat::R32UI, 16, 16, 5);
        auto sizes = [&](int level) {
            bind_image(0, tex_size, level, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(1, tex_3d_, 0, true, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(2, tex_arr_, 0, true, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(3, tex_arr_, 0, false, 1, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            prepare_ssbo(16);
            mgl::glUseProgram(prog_sizes_);
            mgl::glDispatchCompute(1, 1, 1);
            return read_ssbo(16);
        };
        auto expect_size = [&](const std::vector<std::uint32_t>& s, std::size_t i, std::array<std::uint32_t, 3> e, std::string_view what) {
            check.expect(s[4 * i] == e[0] && s[4 * i + 1] == e[1] && s[4 * i + 2] == e[2], what,
                         std::format("got ({}, {}, {}) expected ({}, {}, {})", s[4 * i], s[4 * i + 1], s[4 * i + 2], e[0], e[1], e[2]));
        };
        const std::vector<std::uint32_t> s1 = sizes(1);
        expect_size(s1, 0, {8, 8, 0}, "imageSize(image2D) of level 1 of a 16x16 texture is (8, 8)");
        expect_size(s1, 1, {8, 8, 4}, "imageSize(image3D) of a layered 8x8x4 3D texture is (8, 8, 4)");
        expect_size(s1, 2, {8, 8, 4}, "imageSize(image2DArray) of a layered 8x8x4 2D array is (8, 8, 4): the last component is the number of layers");
        expect_size(s1, 3, {8, 8, 0}, "imageSize(image2D) of ONE layer of that array (layered = FALSE) is (8, 8)");
        const std::vector<std::uint32_t> s2 = sizes(2);
        expect_size(s2, 0, {4, 4, 0}, "imageSize(image2D) of level 2 of the 16x16 texture is (4, 4)");
    }

    void ImageLoadStore::check_reinterpretation(demo::Check& check)
    {
        // ES 3.2 8.23 / Table 8.27: for textures allocated by the GL, an image unit format is compatible with the texture
        // format if both texels have the same size; the memory is then reinterpreted.
        const FormatCase& f32 = kFormats[2];    // r32f
        const FormatCase& rgba8 = kFormats[11];  // rgba8
        const Texels floats = make_texels(f32, 40u);
        const Texels bytes = make_texels(rgba8, 41u);

        // (1) R32F texture seen as r32ui: the IEEE bit patterns.
        {
            const mgl::TextureId tf = new_texture_2d(f32.storage, 8, 8, 1), tu = new_texture_2d(mgl::SizedInternalFormat::R32UI, 8, 8, 1);
            upload_2d(mgl::TextureTarget::Texture2D, tf, 0, 8, 8, f32.pixel_format, f32.pixel_type, floats.bytes.data());
            mgl::glUseProgram(prog_copy_bits_);
            bind_image(0, tf, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(1, tu, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch);
            const std::vector<std::uint32_t> got = dump(DumpU2, tu, 0, 0, 8, 8);
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < 64; ++k)
                if (got[4 * k] != floats.words[k][0] && bad++ == 0) first = std::format("texel {}: got {:#x} expected {:#x}", k, got[4 * k], floats.words[k][0]);
            check.expect(bad == 0, "an R32F texture bound with format R32UI loads the IEEE-754 bit patterns of its floats (64 texels)", first);
        }
        // (2) RGBA8 texture seen as r32ui: the four bytes of a texel as one word (client byte order, as the spec's ReadPixels /
        //     TexSubImage description of the reinterpretation reads and writes scratch memory).
        {
            const mgl::TextureId tr = new_texture_2d(rgba8.storage, 8, 8, 1), tu = new_texture_2d(mgl::SizedInternalFormat::R32UI, 8, 8, 1);
            upload_2d(mgl::TextureTarget::Texture2D, tr, 0, 8, 8, rgba8.pixel_format, rgba8.pixel_type, bytes.bytes.data());
            mgl::glUseProgram(prog_copy_bits_);
            bind_image(0, tr, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(1, tu, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch);
            const std::vector<std::uint32_t> got = dump(DumpU2, tu, 0, 0, 8, 8);
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < 64; ++k)
            {
                std::uint32_t e = 0;
                std::memcpy(&e, &bytes.bytes[4 * k], 4);
                if (got[4 * k] != e && bad++ == 0) first = std::format("texel {}: got {:#x} expected {:#x}", k, got[4 * k], e);
            }
            check.expect(bad == 0, "an RGBA8 texture bound with format R32UI loads each texel's four bytes as one 32-bit word (64 texels)", first);
        }
        // (3) The other direction: r32ui stores into an RGBA8 texture, read back through a sampler as bytes.
        {
            const mgl::TextureId tu = new_texture_2d(mgl::SizedInternalFormat::R32UI, 8, 8, 1), tr = new_texture_2d(rgba8.storage, 8, 8, 1);
            std::vector<std::uint32_t> words(64);
            for (std::size_t k = 0; k < 64; ++k) words[k] = hash32(static_cast<std::uint32_t>(k) + 500u);
            upload_2d(mgl::TextureTarget::Texture2D, tu, 0, 8, 8, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, words.data());
            mgl::glUseProgram(prog_copy_bits_);
            bind_image(0, tu, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(1, tr, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch);
            const std::vector<std::uint32_t> got = dump(DumpF2, tr, 0, 0, 8, 8);
            int bad = 0;
            std::string first;
            for (std::size_t k = 0; k < 64; ++k)
            {
                std::array<std::uint8_t, 4> e{};
                std::memcpy(e.data(), &words[k], 4);
                for (std::size_t c = 0; c < 4; ++c)
                    if (std::fabs(static_cast<double>(as_float(got[4 * k + c])) * 255.0 - static_cast<double>(e[c])) > kFetchTolerance && bad++ == 0)
                        first = std::format("texel {} channel {}: got {} expected {}", k, c, static_cast<double>(as_float(got[4 * k + c])) * 255.0, e[c]);
            }
            check.expect(bad == 0, "r32ui stores into an RGBA8 texture are seen as the four bytes of each word by a sampler (64 texels)", first);
        }
    }

    void ImageLoadStore::check_coherent(demo::Check& check)
    {
        std::vector<std::uint32_t> in(64);
        for (std::size_t k = 0; k < 64; ++k) in[k] = hash32(static_cast<std::uint32_t>(k) + 900u) & 0xFFFFu;
        const std::array<const char*, 2> qualifiers = {"coherent restrict", "volatile"};
        for (std::size_t q = 0; q < qualifiers.size(); ++q)
        {
            // 16 x 4 r32ui textures: scratch (read-write), source (readonly), destination (writeonly).
            const mgl::TextureId scratch = new_texture_2d(mgl::SizedInternalFormat::R32UI, 16, 4, 1);
            const mgl::TextureId src = new_texture_2d(mgl::SizedInternalFormat::R32UI, 16, 4, 1);
            const mgl::TextureId dst = new_texture_2d(mgl::SizedInternalFormat::R32UI, 16, 4, 1);
            upload_2d(mgl::TextureTarget::Texture2D, src, 0, 16, 4, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, in.data());
            fill_u32(mgl::TextureTarget::Texture2D, scratch, 16, 4, 1, 0u);

            mgl::glUseProgram(prog_coherent_[q]);
            bind_image(0, scratch, 0, false, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32UI);
            bind_image(1, src, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            bind_image(2, dst, 0, false, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glDispatchCompute(2, 4, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch);

            const std::vector<std::uint32_t> got = dump(DumpU2, dst, 0, 0, 16, 4);
            const std::vector<std::uint32_t> got_scratch = dump(DumpU2, scratch, 0, 0, 16, 4);
            int bad = 0, bad_plain = 0, bad_scratch = 0;
            std::string first;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 16; ++x)
                {
                    const int rx = (x / 8) * 8 + 7 - (x % 8);
                    const std::uint32_t e = in[static_cast<std::size_t>(y * 16 + rx)] * 3u + 1u;
                    const std::uint32_t plain = in[static_cast<std::size_t>(y * 16 + x)] * 3u + 1u;
                    const std::uint32_t g = got[static_cast<std::size_t>((y * 16 + x) * 4)];
                    if (g != e && bad++ == 0) first = std::format("texel ({},{}): got {} expected {}", x, y, g, e);
                    if (g != plain) ++bad_plain;
                    if (got_scratch[static_cast<std::size_t>((y * 16 + x) * 4)] != plain) ++bad_scratch;
                }
            check.expect(bad == 0, std::format("`{}` r32ui image: each work group of 8 reverses its row segment through imageStore, memoryBarrierImage(), barrier(), imageLoad (64 texels)", qualifiers[q]), first);
            check.expect(bad_scratch == 0, std::format("the `{}` read-write image holds 3 * in + 1 at every texel after the pass (access = READ_WRITE)", qualifiers[q]));
            if (q == 0) check.expect(bad_plain > 0, "negative control: the un-reversed model (dst = 3 * in + 1) does not match the result");
        }
    }

    void ImageLoadStore::check_fragment_stage(demo::Check& check)
    {
        if (!have_fragment_images_)
        {
            std::cout << "note: no fragment-shader image checks (MAX_FRAGMENT_IMAGE_UNIFORMS = 0)\n";
            return;
        }
        const std::vector<std::uint32_t> got = dump(DumpU2, tex_frag_, 0, 0, kFragSize, kFragSize);
        int bad = 0;
        std::string first;
        for (int y = 0; y < kFragSize; ++y)
            for (int x = 0; x < kFragSize; ++x)
            {
                const std::uint32_t g = got[static_cast<std::size_t>((y * kFragSize + x) * 4)], e = frag_value(x, y);
                if (g != e && bad++ == 0) first = std::format("texel ({},{}): got {} expected {}", x, y, g, e);
            }
        check.expect(bad == 0, std::format("imageStore in a fragment shader wrote 3x + 5y + 1 at gl_FragCoord of all {} pixels of the viewport (sentinel 0xFFFFFFFF everywhere else)", kFragSize * kFragSize), first);
    }

    void ImageLoadStore::check_binding_queries(demo::Check& check)
    {
        struct Expected
        {
            const char* label;
            unsigned unit;
            mgl::TextureId tex;
            int level;
            bool layered;
            int layer;
            mgl::ImageAccess access;
            mgl::InternalFormat format;
        };
        const mgl::TextureId f16 = new_texture_2d(mgl::SizedInternalFormat::Rgba16F, 4, 4, 1);
        const std::array<Expected, 4> cases = {{
            {"layered array, READ_WRITE, R32UI", 2, tex_arr_, 0, true, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32UI},
            {"one array layer, WRITE_ONLY, R32UI", 3, tex_arr_, 0, false, 3, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI},
            {"level 2 of a 5-level 2D texture, READ_ONLY, RGBA8", 1, tex_lvl_a_, 2, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8},
            {"RGBA16F texture, READ_WRITE, RGBA16F", 0, f16, 0, false, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::Rgba16F},
        }};
        for (const Expected& e : cases)
            bind_image(e.unit, e.tex, e.level, e.layered, e.layer, e.access, e.format);
        for (const Expected& e : cases)
        {
            GLint v = -1;
            GLboolean b = 2;
            const auto u = static_cast<GLuint>(e.unit);
            mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, u, &v);
            check.expect_eq(v, static_cast<GLint>(e.tex.value), std::format("unit {} ({}): IMAGE_BINDING_NAME", e.unit, e.label));
            mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingLevel, u, &v);
            check.expect_eq(v, e.level, std::format("unit {}: IMAGE_BINDING_LEVEL", e.unit));
            mgl::glGetBooleani_v(mgl::GetParameter::ImageBindingLayered, u, &b);
            check.expect_eq(b == GL_TRUE, e.layered, std::format("unit {}: IMAGE_BINDING_LAYERED (glGetBooleani_v)", e.unit));
            mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingLayer, u, &v);
            check.expect_eq(v, e.layer, std::format("unit {}: IMAGE_BINDING_LAYER", e.unit));
            mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingAccess, u, &v);
            check.expect_eq(v, static_cast<GLint>(e.access), std::format("unit {}: IMAGE_BINDING_ACCESS", e.unit));
            mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingFormat, u, &v);
            check.expect_eq(v, static_cast<GLint>(e.format), std::format("unit {}: IMAGE_BINDING_FORMAT", e.unit));
        }

        // glBindImageTexture with texture 0 unbinds; deleting a bound texture detaches it (section 8.23).
        GLint v = -1;
        bind_image(1, mgl::TextureId{0}, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
        mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, 1, &v);
        check.expect_eq(v, 0, "glBindImageTexture(unit, texture 0, ...) unbinds: IMAGE_BINDING_NAME is 0");
        mgl::TextureId doomed{};
        mgl::glGenTextures(1, &doomed);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, doomed);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::R32UI, 4, 4);
        bind_image(3, doomed, 0, false, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32UI);
        mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, 3, &v);
        check.expect_eq(v, static_cast<GLint>(doomed.value), "a second texture is bound to unit 3 (setup of the detach check)");
        mgl::glDeleteTextures(1, &doomed);
        mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, 3, &v);
        check.expect_eq(v, 0, "glDeleteTextures detaches the texture from the image unit: IMAGE_BINDING_NAME is 0");

        // IMAGE_FORMAT_COMPATIBILITY_TYPE (ES 3.2 8.23): textures allocated by the GL are compatible by size.
        auto compat = [&](mgl::TextureTarget target, mgl::TextureId tex, std::string_view what) {
            GLint t = -1;
            mgl::glBindTexture(target, tex);
            mgl::glGetTexParameteriv(target, mgl::TextureParameterQuery::ImageFormatCompatibilityType, &t);
            check.expect_eq(t, static_cast<GLint>(GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE), what);
        };
        compat(mgl::TextureTarget::Texture2D, tex_src_, "IMAGE_FORMAT_COMPATIBILITY_TYPE of an immutable RGBA8 2D texture is IMAGE_FORMAT_COMPATIBILITY_BY_SIZE");
        compat(mgl::TextureTarget::Texture2DArray, tex_arr_, "IMAGE_FORMAT_COMPATIBILITY_TYPE of an immutable R32UI 2D array texture is IMAGE_FORMAT_COMPATIBILITY_BY_SIZE");
        mgl::TextureId mutable_tex{};
        mgl::glGenTextures(1, &mutable_tex);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mutable_tex);
        mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 4, 4, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
        compat(mgl::TextureTarget::Texture2D, mutable_tex, "IMAGE_FORMAT_COMPATIBILITY_TYPE of a (mutable) glTexImage2D texture is IMAGE_FORMAT_COMPATIBILITY_BY_SIZE");
        mgl::glDeleteTextures(1, &mutable_tex);
    }

    void ImageLoadStore::check_errors(demo::Check& check)
    {
        const mgl::TextureId good = new_texture_2d(mgl::SizedInternalFormat::Rgba8, 8, 8, 1);
        const auto bad_unit = static_cast<unsigned>(max_image_units_);

        // The unit that the failing calls target holds a known binding; errors have no effect (ES 3.2 2.3.1).
        bind_image(0, good, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
        check.expect_no_gl_error("glBindImageTexture with an immutable texture and a Table 8.26 format raises no error");

        mgl::TextureId mutable_tex{};
        mgl::glGenTextures(1, &mutable_tex);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mutable_tex);
        mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 8, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
        bind_image(0, mutable_tex, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBindImageTexture of a texture made with glTexImage2D (mutable) -> INVALID_OPERATION");
        mgl::glDeleteTextures(1, &mutable_tex);

        bind_image(bad_unit, good, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindImageTexture with unit == MAX_IMAGE_UNITS -> INVALID_VALUE");
        bind_image(0, good, -1, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindImageTexture with level = -1 -> INVALID_VALUE");
        bind_image(0, good, 0, false, -1, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindImageTexture with layer = -1 -> INVALID_VALUE");
        bind_image(0, mgl::TextureId{0x3FFFFFF0u}, 0, false, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindImageTexture with a name that is not a texture -> INVALID_VALUE");
        // Formats that are legal ES texture formats but NOT in Table 8.26. No extension changes the table: the extensions
        // whose names mention "image" are printed here (OES_shader_image_atomic and EXT_shader_image_load_formatted change
        // shader-side rules only, not the image unit formats).
        for (const std::string& ext : mgl::GetCapabilities().extensions)
            if (ext.find("image") != std::string::npos && ext.find("shader") != std::string::npos) std::cout << "note: extension " << ext << '\n';
        struct Outside
        {
            mgl::InternalFormat format;
            const char* name;
            bool mesa_accepts;  // observed on radeonsi, llvmpipe and zink (Mesa 25.0.7)
        };
        const std::array<Outside, 7> outside = {{{mgl::InternalFormat::Rgb8, "RGB8", false}, {mgl::InternalFormat::Rg32F, "RG32F", true},
                                                 {mgl::InternalFormat::Rg16F, "RG16F", true}, {mgl::InternalFormat::R16F, "R16F", true},
                                                 {mgl::InternalFormat::R8, "R8", true}, {mgl::InternalFormat::Rg8, "RG8", true},
                                                 {mgl::InternalFormat::R11FG11FB10F, "R11F_G11F_B10F", true}}};
        for (const Outside& o : outside)
        {
            bind_image(0, good, 0, false, 0, mgl::ImageAccess::ReadOnly, o.format);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            const std::string what = std::format("glBindImageTexture with format {} (not in Table 8.26) -> INVALID_VALUE", o.name);
            const std::string detail = std::format("glGetError() = {}", demo::error_name(e));
            if (o.mesa_accepts)
                check.expect_or_known_deviation(e == mgl::ErrorCode::InvalidValue, what, detail, "Mesa",
                                                "Mesa validates the format against the desktop ARB_shader_image_load_store list, so formats outside ES 3.2 Table 8.26 are accepted");
            else
                check.expect(e == mgl::ErrorCode::InvalidValue, what, detail);
        }

        GLint name = -1;
        mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, 0, &name);
        check.expect_eq(name, static_cast<GLint>(good.value), "the failed calls left unit 0 bound to the texture of the first successful call (an erroring command has no effect)");

        mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, bad_unit, &name);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(IMAGE_BINDING_NAME, MAX_IMAGE_UNITS) -> INVALID_VALUE (index out of range, ES 3.2 20.1)");
        GLboolean layered = GL_FALSE;
        mgl::glGetBooleani_v(mgl::GetParameter::ImageBindingLayered, bad_unit, &layered);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetBooleani_v(IMAGE_BINDING_LAYERED, MAX_IMAGE_UNITS) -> INVALID_VALUE (index out of range, ES 3.2 20.1)");
    }

    void ImageLoadStore::check_compile_errors(demo::Check& check)
    {
        auto compiles = [](std::string_view body, std::string& log) {
            const std::string src = std::string("#version 310 es\nlayout(local_size_x = 1) in;\n") + std::string(body);
            const mgl::ShaderId s = demo::compile_shader(mgl::ShaderType::Compute, src, log);
            if (s.value == 0) return false;
            mgl::glDeleteShader(s);
            return true;
        };
        std::string log;
        const bool control = compiles(
            "layout(rgba8, binding = 0) uniform readonly highp image2D a;\n"
            "layout(rgba8, binding = 1) uniform writeonly highp image2D b;\n"
            "layout(r32f, binding = 2) uniform highp image2D c;\n"
            "void main() { imageStore(b, ivec2(0), imageLoad(a, ivec2(0))); imageStore(c, ivec2(0), imageLoad(c, ivec2(0))); }\n", log);
        check.expect(control, "control: readonly load + writeonly store + read-write r32f image compile", log);

        struct Negative
        {
            const char* what;
            const char* body;
        };
        const std::array<Negative, 5> negatives = {{
            {"imageStore to a readonly image is a compile-time error (GLSL ES 3.10 4.9)",
             "layout(rgba8, binding = 0) uniform readonly highp image2D a;\nvoid main() { imageStore(a, ivec2(0), vec4(1.0)); }\n"},
            {"imageLoad from a writeonly image is a compile-time error (GLSL ES 3.10 4.9)",
             "layout(rgba8, binding = 0) uniform writeonly highp image2D a;\nvoid main() { vec4 v = imageLoad(a, ivec2(0)); imageStore(a, ivec2(0), v); }\n"},
            {"an rgba8 image without readonly or writeonly is a compile-time error (GLSL ES 3.10 4.9: only r32f, r32i, r32ui may be read-write)",
             "layout(rgba8, binding = 0) uniform highp image2D a;\nvoid main() { imageStore(a, ivec2(0), vec4(1.0)); }\n"},
            {"an image variable without a format layout qualifier is a compile-time error (GLSL ES 3.10 4.4.7)",
             "layout(binding = 0) uniform writeonly highp image2D a;\nvoid main() { imageStore(a, ivec2(0), vec4(1.0)); }\n"},
            {"a float format qualifier (rgba8) on a uimage2D is a compile-time error (GLSL ES 3.10 4.4.7)",
             "layout(rgba8, binding = 0) uniform writeonly highp uimage2D a;\nvoid main() { imageStore(a, ivec2(0), uvec4(1u)); }\n"},
        }};
        for (const Negative& n : negatives)
        {
            std::string nlog;
            const bool ok = compiles(n.body, nlog);
            check.expect(!ok, n.what, ok ? "the shader compiled" : "");
        }
    }

    void ImageLoadStore::validate(demo::Env& /*env*/, demo::Check& check)
    {
        check_limits_and_initial_state(check);
        check_display(check);  // reads the default framebuffer first: nothing below draws into it
        check_filters(check);
        check_formats(check);
        check_layers(check);
        if (!slice3d_not_run_)
        {
            // The isolated child (see SliceStoreChild): only a death inside the store/read-back is the documented deviation;
            // any other outcome (wrong values, a crash during setup) is a plain failure on every driver.
            const ChildResult& r = slice3d_child_;
            const std::string what = "isolated child: imageStore through a non-layered binding of 3D-texture slice 1 writes exactly that slice "
                                     "(the other slices keep their contents) and the process survives";
            if (r.passed() || r.died_in_call())
                check.expect_or_known_deviation(r.passed(), what, r.describe(), "ANGLE",
                                                "ANGLE (Vulkan, RADV) raises a GPU page fault on imageStore through a non-layered binding of a 3D-texture "
                                                "slice; the context is lost and the next synchronizing call hangs (ES 3.2 8.23: the slice is treated as a 2D texture)");
            else
                check.expect(false, what, r.describe());
        }
        check_levels_and_sizes(check);
        check_reinterpretation(check);
        check_coherent(check);
        check_fragment_stage(check);
        check_binding_queries(check);
        check_errors(check);
        check_compile_errors(check);
    }
}

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string_view(argv[1]) == kChildFlag)
    {
        if (std::string_view(argv[2]) != kSliceChild)
        {
            std::cerr << "unknown child case '" << argv[2] << "'\n";
            return 2;
        }
        std::array<char*, 3> child_argv = {argv[0], const_cast<char*>("--headless"), const_cast<char*>("--validate")};
        const demo::Info child_info{
            .name = "es31-image-load-store(child)",
            .title = "Image store into a 3D-texture slice: isolated child process",
            .min_es = {3, 1},
            .summary = "Internal child process of es31-image-load-store.",
            .source_dir = DEMO_SOURCE_DIR,
        };
        return demo::run<SliceStoreChild>(static_cast<int>(child_argv.size()), child_argv.data(), child_info);
    }

    const demo::Info info{
        .name = "es31-image-load-store",
        .title = "Image load/store",
        .min_es = {3, 1},
        .summary = "glBindImageTexture + imageLoad/imageStore/imageSize in compute (and fragment) shaders: all 13 image formats, a blur and a Sobel "
                   "filter against CPU references, layered and non-layered bindings, level selection, memory barriers, image unit state and errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ImageLoadStore>(argc, argv, info);
}
