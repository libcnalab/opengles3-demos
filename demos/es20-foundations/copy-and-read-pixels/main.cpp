// es20-copy-and-read-pixels
//
// Getting pixels out of a framebuffer (OpenGL ES 3.0 sections 3.8.5 and 4.3.2): glReadPixels into
// client memory (the required RGBA/UNSIGNED_BYTE pair and the implementation-chosen pair, sub-
// rectangles, PACK_ALIGNMENT / PACK_ROW_LENGTH / PACK_SKIP_*, reads that leave the framebuffer) and
// glCopyTexImage2D / glCopyTexSubImage2D into texture images that are then sampled.
//
// Everything starts from a 64 x 64 test chart in an RGBA8 framebuffer object whose pixel (x, y) is
// (4x, 4y, checker, 128 + x + y): each pixel is distinct, so a misplaced, flipped or channel-swapped
// result cannot pass. The 256 x 256 default framebuffer shows (cells of 64 x 64, top row first):
//
//   row 0: the chart | glCopyTexImage2D RGBA, whole chart | two glCopyTexSubImage2D patches | RGBA8 sub-rectangle (2x)
//   row 1: RGB (2x)  | LUMINANCE (2x)                      | ALPHA as gray (2x)               | LUMINANCE_ALPHA (2x)
//   row 2: R8 (2x)   | RG8 (2x)
//
// Validation: the chart is read back and compared with its analytic formula; reads are compared with
// that read-back (exactly: the same stored values), copies are verified twice - by attaching the
// copied texture to a framebuffer and reading it, and by sampling it - and the defined error cases of
// both command families are provoked.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kChart = 64;     // edge of the chart framebuffer
    constexpr int kCellSize = 64;  // edge of a cell on the screen
    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    // A float -> 8-bit conversion returns one of the two integers closest to f * 255 (eq. 2.3, section
    // 2.1.6.2): results of shader output / texture sampling are compared with +-1. Values that are only
    // *moved* (glReadPixels, glCopyTex*, ReadPixels of a texture attached to a framebuffer) are compared
    // exactly: no arithmetic is specified for them beyond the conversion of the stored 8-bit value.
    constexpr int kConvertTolerance = 1;

    std::uint8_t u8(int v) { return static_cast<std::uint8_t>(v); }

    // CPU reference of chart.frag.
    Rgba8 chart_texel(int x, int y)
    {
        const bool odd = (((x >> 3) ^ (y >> 3)) & 1) != 0;
        return {u8(4 * x), u8(4 * y), u8(odd ? 204 : 51), u8(128 + x + y)};
    }

    struct Region
    {
        int x, y, w, h;
    };

    std::vector<Rgba8> to_texels(std::span<const std::uint8_t> bytes)
    {
        std::vector<Rgba8> out(bytes.size() / 4);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = {bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]};
        return out;
    }

    std::vector<Rgba8> read_texels(int x, int y, int w, int h) { return to_texels(demo::read_rgba8(x, y, w, h)); }

    std::vector<Rgba8> block_of(std::span<const Rgba8> image, int image_width, const Region& r)
    {
        std::vector<Rgba8> out;
        out.reserve(static_cast<std::size_t>(r.w) * static_cast<std::size_t>(r.h));
        for (int j = 0; j < r.h; ++j)
            for (int i = 0; i < r.w; ++i)
                out.push_back(image[static_cast<std::size_t>(r.y + j) * static_cast<std::size_t>(image_width) + static_cast<std::size_t>(r.x + i)]);
        return out;
    }

    struct Diff
    {
        int bad = 0;
        int max_dev = 0;
        std::string first;
    };

    Diff compare(std::span<const Rgba8> got, std::span<const Rgba8> want, int width, int tolerance)
    {
        Diff d;
        if (got.size() != want.size()) d.bad = 1 << 20;
        for (std::size_t i = 0; i < want.size() && i < got.size(); ++i)
        {
            int dev = 0;
            for (std::size_t k = 0; k < 4; ++k) dev = std::max(dev, std::abs(int{got[i][k]} - int{want[i][k]}));
            d.max_dev = std::max(d.max_dev, dev);
            if (dev > tolerance && d.bad++ == 0)
                d.first = std::format("first at ({},{}): got {} want {}", static_cast<int>(i) % width, static_cast<int>(i) / width,
                                      demo::format_rgba(got[i]), demo::format_rgba(want[i]));
        }
        return d;
    }

    // ---------------------------------------------------------------------------------------------
    // Pixel transfer layout (ES 3.0 section 3.7.2.1, equation 3.13 with the PACK_* parameters).
    // ---------------------------------------------------------------------------------------------
    struct Layout
    {
        int n;  // elements per group (1 for the packed types)
        int s;  // bytes per element
    };

    struct Pair
    {
        mgl::PixelFormat format;
        mgl::PixelType type;
        Layout layout;
        std::array<int, 4> bits;  // bits per R, G, B, A in this type; 0 = the pair has no such channel
        [[nodiscard]] int group_bytes() const { return layout.n * layout.s; }
    };

    /// Bytes between the starts of two consecutive rows: k * s with k = nl when s >= a, else a/s * ceil(s*n*l / a).
    int row_stride_bytes(const Layout& lay, int groups_per_row, int alignment)
    {
        const int nl = lay.n * groups_per_row;
        const int k = lay.s >= alignment ? nl : alignment / lay.s * ((lay.s * nl + alignment - 1) / alignment);
        return k * lay.s;
    }

    std::optional<Pair> make_pair(mgl::PixelFormat f, mgl::PixelType t)
    {
        using F = mgl::PixelFormat;
        using T = mgl::PixelType;
        int comps = 0;
        std::array<bool, 4> present{};
        switch (f)
        {
        case F::Rgba: case F::Bgra: comps = 4; present = {true, true, true, true}; break;
        case F::Rgb: comps = 3; present = {true, true, true, false}; break;
        case F::Rg: comps = 2; present = {true, true, false, false}; break;
        case F::Red: comps = 1; present = {true, false, false, false}; break;
        default: return std::nullopt;
        }
        Pair p{f, t, {}, {}};
        auto uniform_bits = [&](int bits) { for (std::size_t c = 0; c < 4; ++c) p.bits[c] = present[c] ? bits : 0; };
        switch (t)
        {
        case T::UnsignedByte: p.layout = {comps, 1}; uniform_bits(8); break;
        case T::HalfFloat: if (f == F::Bgra) return std::nullopt; p.layout = {comps, 2}; uniform_bits(10); break;
        case T::Float: if (f == F::Bgra) return std::nullopt; p.layout = {comps, 4}; uniform_bits(23); break;
        case T::UnsignedShort565: if (f != F::Rgb) return std::nullopt; p.layout = {1, 2}; p.bits = {5, 6, 5, 0}; break;
        case T::UnsignedShort4444: if (f != F::Rgba) return std::nullopt; p.layout = {1, 2}; p.bits = {4, 4, 4, 4}; break;
        case T::UnsignedShort5551: if (f != F::Rgba) return std::nullopt; p.layout = {1, 2}; p.bits = {5, 5, 5, 1}; break;
        case T::UnsignedInt2101010Rev: if (f != F::Rgba) return std::nullopt; p.layout = {1, 4}; p.bits = {10, 10, 10, 2}; break;
        default: return std::nullopt;
        }
        return p;
    }

    /// The (format, type) pair IMPLEMENTATION_COLOR_READ_FORMAT / _TYPE currently name, if this demo can decode it.
    /// The values come back as plain integers; they are mapped to the typed enumerators by comparison, not by casting.
    std::optional<Pair> implementation_pair(GLint* raw_format = nullptr, GLint* raw_type = nullptr)
    {
        const GLint f = demo::get_integer(mgl::GetParameter::ImplementationColorReadFormat);
        const GLint t = demo::get_integer(mgl::GetParameter::ImplementationColorReadType);
        if (raw_format) *raw_format = f;
        if (raw_type) *raw_type = t;
        using F = mgl::PixelFormat;
        using T = mgl::PixelType;
        for (const F format : {F::Rgba, F::Rgb, F::Rg, F::Red, F::Bgra})
            for (const T type : {T::UnsignedByte, T::UnsignedShort565, T::UnsignedShort4444, T::UnsignedShort5551,
                                 T::UnsignedInt2101010Rev, T::Float, T::HalfFloat})
                if (static_cast<GLint>(format) == f && static_cast<GLint>(type) == t) return make_pair(format, type);
        return std::nullopt;
    }

    std::string describe(GLint format, GLint type)
    {
        return std::format("format {:#06x} type {:#06x}", static_cast<unsigned>(format), static_cast<unsigned>(type));
    }

    double half_to_double(std::uint16_t h)
    {
        const int e = (h >> 10) & 31, m = h & 1023;
        const double sign = (h >> 15) != 0 ? -1.0 : 1.0;
        if (e == 0) return sign * std::ldexp(static_cast<double>(m), -24);
        if (e == 31) return m != 0 ? std::numeric_limits<double>::quiet_NaN() : sign * std::numeric_limits<double>::infinity();
        return sign * std::ldexp(static_cast<double>(1024 + m), e - 25);
    }

    /// One pixel group -> normalized RGBA. Components the format does not carry are 0, 0, 0 and 1 (section 4.3.2.1).
    std::array<double, 4> decode(const Pair& p, const std::uint8_t* b)
    {
        std::array<double, 4> v{0.0, 0.0, 0.0, 1.0};
        switch (p.type)
        {
        case mgl::PixelType::UnsignedByte:
            for (int c = 0; c < p.layout.n; ++c)
            {
                const int channel = (p.format == mgl::PixelFormat::Bgra && c < 3) ? 2 - c : c;
                v[static_cast<std::size_t>(channel)] = b[c] / 255.0;
            }
            break;
        case mgl::PixelType::Float:
            for (int c = 0; c < p.layout.n; ++c) { float f; std::memcpy(&f, b + 4 * c, 4); v[static_cast<std::size_t>(c)] = static_cast<double>(f); }
            break;
        case mgl::PixelType::HalfFloat:
            for (int c = 0; c < p.layout.n; ++c) { std::uint16_t h; std::memcpy(&h, b + 2 * c, 2); v[static_cast<std::size_t>(c)] = half_to_double(h); }
            break;
        case mgl::PixelType::UnsignedShort565:
        {
            std::uint16_t x; std::memcpy(&x, b, 2);
            v = {((x >> 11) & 31) / 31.0, ((x >> 5) & 63) / 63.0, (x & 31) / 31.0, 1.0};
            break;
        }
        case mgl::PixelType::UnsignedShort4444:
        {
            std::uint16_t x; std::memcpy(&x, b, 2);
            v = {((x >> 12) & 15) / 15.0, ((x >> 8) & 15) / 15.0, ((x >> 4) & 15) / 15.0, (x & 15) / 15.0};
            break;
        }
        case mgl::PixelType::UnsignedShort5551:
        {
            std::uint16_t x; std::memcpy(&x, b, 2);
            v = {((x >> 11) & 31) / 31.0, ((x >> 6) & 31) / 31.0, ((x >> 1) & 31) / 31.0, (x & 1) / 1.0};
            break;
        }
        case mgl::PixelType::UnsignedInt2101010Rev:
        {
            std::uint32_t x; std::memcpy(&x, b, 4);
            v = {(x & 1023u) / 1023.0, ((x >> 10) & 1023u) / 1023.0, ((x >> 20) & 1023u) / 1023.0, ((x >> 30) & 3u) / 3.0};
            break;
        }
        default: break;
        }
        return v;
    }

    double step(int bits) { return bits > 0 ? 1.0 / (std::pow(2.0, bits) - 1.0) : 0.0; }

    // ---------------------------------------------------------------------------------------------
    // Texture copy model (ES 3.0 section 3.8.5, Table 3.24): what sampling a texture gives, as RGBA, when
    // it was made from a framebuffer pixel `s` with a given internal format.
    // ---------------------------------------------------------------------------------------------
    Rgba8 expected_texel(mgl::InternalFormat internal, Rgba8 s)
    {
        switch (internal)
        {
        case mgl::InternalFormat::Rgb: case mgl::InternalFormat::Rgb8: return {s[0], s[1], s[2], 255};
        case mgl::InternalFormat::Luminance: return {s[0], s[0], s[0], 255};  // L = R
        case mgl::InternalFormat::Alpha: return {0, 0, 0, s[3]};
        case mgl::InternalFormat::LuminanceAlpha: return {s[0], s[0], s[0], s[3]};
        case mgl::InternalFormat::R8: return {s[0], 0, 0, 255};
        case mgl::InternalFormat::Rg8: return {s[0], s[1], 0, 255};
        default: return s;  // RGBA, RGBA8
        }
    }

    enum Tex : std::size_t { kTexFull, kTexRgba8, kTexRgb, kTexLum, kTexAlpha, kTexLumAlpha, kTexR8, kTexRg8, kTexSub, kTexCount };

    struct CopyCase
    {
        std::string_view name;
        Tex tex;
        mgl::InternalFormat internal;
        Region src;
        int cell;
        int scale;         // magnification on the screen
        int display_mode;  // show.frag u_mode of the on-screen cell
        bool renderable;   // can the texture be attached to a framebuffer (and read back directly)?
    };

    // LUMINANCE, ALPHA and LUMINANCE_ALPHA are not color-renderable (section 4.4.4): those textures can
    // only be checked by sampling.
    const std::array<CopyCase, 8> kCases = {{
        {"RGBA, whole chart", kTexFull, mgl::InternalFormat::Rgba, {0, 0, 64, 64}, 1, 1, 0, true},
        {"RGBA8 (sized), sub-rectangle", kTexRgba8, mgl::InternalFormat::Rgba8, {8, 16, 32, 24}, 3, 2, 0, true},
        {"RGB, sub-rectangle", kTexRgb, mgl::InternalFormat::Rgb, {8, 16, 32, 24}, 4, 2, 0, true},
        {"LUMINANCE, sub-rectangle", kTexLum, mgl::InternalFormat::Luminance, {8, 16, 32, 24}, 5, 2, 0, false},
        {"ALPHA, sub-rectangle", kTexAlpha, mgl::InternalFormat::Alpha, {8, 16, 32, 24}, 6, 2, 1, false},
        {"LUMINANCE_ALPHA, sub-rectangle", kTexLumAlpha, mgl::InternalFormat::LuminanceAlpha, {8, 16, 32, 24}, 7, 2, 0, false},
        {"R8, sub-rectangle", kTexR8, mgl::InternalFormat::R8, {8, 16, 32, 24}, 8, 2, 0, true},
        {"RG8, sub-rectangle", kTexRg8, mgl::InternalFormat::Rg8, {8, 16, 32, 24}, 9, 2, 0, true},
    }};

    // The glCopyTexSubImage2D scene: a solid texture with two patches copied into it.
    constexpr Rgba8 kBaseColor = {100, 150, 200, 255};
    struct Patch
    {
        int xoffset, yoffset;
        Region src;
    };
    constexpr std::array<Patch, 2> kPatches = {{{20, 12, {8, 16, 16, 16}}, {40, 40, {40, 40, 24, 24}}}};
    constexpr int kSubCell = 2;
    constexpr int kChartCell = 0;

    int cell_x(int cell) { return (cell % 4) * kCellSize; }
    int cell_y(int cell) { return (3 - cell / 4) * kCellSize; }

    template <typename F>
    void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // start from an empty queue so the error belongs to this call
        call();
        check.expect_gl_error(code, what);
    }

    class CopyAndReadPixels final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 4 * kCellSize || env.height < 4 * kCellSize)
                throw demo::Skip(std::format("needs a default framebuffer of at least {0}x{0}", 4 * kCellSize));
            if (demo::get_integer(mgl::GetParameter::RedBits) < 8 || demo::get_integer(mgl::GetParameter::GreenBits) < 8 ||
                demo::get_integer(mgl::GetParameter::BlueBits) < 8)
                throw demo::Skip("needs 8 bits per color channel in the default framebuffer");

            chart_program_ = demo::build_program(env, "shaders/quad.vert", "shaders/chart.frag");
            show_program_ = demo::build_program(env, "shaders/quad.vert", "shaders/show.frag");
            chart_rect_ = mgl::glGetUniformLocation(chart_program_, "u_rect");
            show_rect_ = mgl::glGetUniformLocation(show_program_, "u_rect");
            show_tex_ = mgl::glGetUniformLocation(show_program_, "u_tex");
            show_mode_ = mgl::glGetUniformLocation(show_program_, "u_mode");

            // The chart and a scratch target are RGBA8 renderbuffer framebuffers: a known format on every driver.
            make_target(chart_fbo_, chart_rb_, mgl::InternalFormat::Rgba8, kChart, kChart);
            make_target(scratch_fbo_, scratch_rb_, mgl::InternalFormat::Rgba8, kChart, kChart);
            mgl::glGenFramebuffers(1, &attach_fbo_);  // no attachments yet: textures are attached when they are read back

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            for (const mgl::TextureId t : tex_)
            {
                // Copying creates only level 0: the default MIN_FILTER (NEAREST_MIPMAP_LINEAR) would make the
                // texture incomplete. NEAREST needs no further levels.
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, GL_NEAREST);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, GL_NEAREST);
            }
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);  // dithering could change a stored value by one step

            // ---- the chart --------------------------------------------------------------------------
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, chart_fbo_);
            mgl::glViewport(0, 0, kChart, kChart);
            mgl::glUseProgram(chart_program_);
            mgl::glUniform4f(chart_rect_, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);

            // ---- copies: they read the READ framebuffer (the chart), not the DRAW framebuffer ------------
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, chart_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            for (const CopyCase& c : kCases)
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[c.tex]);
                // x, y, width, height address the framebuffer exactly like glReadPixels (lower left origin);
                // internalformat and border follow glTexImage2D (border must be 0).
                mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, c.internal, c.src.x, c.src.y, c.src.w, c.src.h, 0);
            }
            {
                // glCopyTexSubImage2D respecifies a rectangle of an existing level: allocate it first.
                const std::vector<Rgba8> base(static_cast<std::size_t>(kChart) * kChart, kBaseColor);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexSub]);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kChart, kChart, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, base.data());
                for (const Patch& p : kPatches)
                    mgl::glCopyTexSubImage2D(mgl::TextureTarget::Texture2D, 0, p.xoffset, p.yoffset, p.src.x, p.src.y, p.src.w, p.src.h);
            }
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            // ---- the default framebuffer shows everything -------------------------------------------------
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, 4 * kCellSize, 4 * kCellSize);
            mgl::glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, chart_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            mgl::glBlitFramebuffer(0, 0, kChart, kChart, cell_x(kChartCell), cell_y(kChartCell), cell_x(kChartCell) + kChart, cell_y(kChartCell) + kChart,
                                   mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            for (const CopyCase& c : kCases)
            {
                mgl::glViewport(cell_x(c.cell), cell_y(c.cell), c.src.w * c.scale, c.src.h * c.scale);
                show(tex_[c.tex], c.display_mode);
            }
            mgl::glViewport(cell_x(kSubCell), cell_y(kSubCell), kChart, kChart);
            show(tex_[kTexSub], 0);
            mgl::glViewport(0, 0, 4 * kCellSize, 4 * kCellSize);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            // Ground truth for everything below: the chart, read once with the one pair every ES 3.0
            // implementation must accept for a normalized fixed-point surface.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, chart_fbo_);
            chart_ = read_texels(0, 0, kChart, kChart);

            check_read_pixels(check);
            check_out_of_window(check, env);
            check_pack_alignment(check);
            check_pack_rows(check);
            check_implementation_pair(check);
            check_copy_tex_image(check);
            check_copy_tex_sub_image(check);
            check_copy_errors(check);
            check_read_errors(check);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            check.expect_no_gl_error("no stray GL error left by the validation sequence");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteFramebuffers(1, &attach_fbo_);
            mgl::glDeleteFramebuffers(1, &scratch_fbo_);
            mgl::glDeleteFramebuffers(1, &chart_fbo_);
            mgl::glDeleteRenderbuffers(1, &scratch_rb_);
            mgl::glDeleteRenderbuffers(1, &chart_rb_);
            mgl::glDeleteProgram(show_program_);
            mgl::glDeleteProgram(chart_program_);
        }

    private:
        // ---- helpers ---------------------------------------------------------------------------------
        static void make_target(mgl::FramebufferId& fbo, mgl::RenderbufferId& rb, mgl::InternalFormat format, int w, int h)
        {
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, format, w, h);
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::RenderbufferTarget::Renderbuffer, rb);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer is {}", demo::framebuffer_status_name(status)));
        }

        // Samples `tex` over the whole current viewport.
        void show(mgl::TextureId tex, int mode)
        {
            mgl::glUseProgram(show_program_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glUniform1i(show_tex_, 0);
            mgl::glUniform1i(show_mode_, mode);
            mgl::glUniform4f(show_rect_, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        // Renders `tex` 1:1 into the scratch target and reads it back: what a shader sees in each texel.
        std::vector<Rgba8> sampled(mgl::TextureId tex, int w, int h, int mode)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, scratch_fbo_);
            mgl::glViewport(0, 0, w, h);
            mgl::glClearColor(1.0f, 0.0f, 1.0f, 0.0f);  // magenta, transparent: a texel that was not written is visible
            mgl::glClear(mgl::ClearBufferBit::Color);
            show(tex, mode);
            const std::vector<Rgba8> out = read_texels(0, 0, w, h);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return out;
        }

        // Attaches level 0 of `tex` to a framebuffer and reads it with glReadPixels.
        std::optional<std::vector<Rgba8>> read_attached(mgl::TextureId tex, int w, int h)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, attach_fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, tex, 0);
            std::optional<std::vector<Rgba8>> out;
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete)
                out = read_texels(0, 0, w, h);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return out;
        }

        void bind_read(mgl::FramebufferId fbo) { mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo); }

        // ===========================================================================================
        // glReadPixels
        // ===========================================================================================
        void check_read_pixels(demo::Check& check)
        {
            bind_read(chart_fbo_);
            std::vector<Rgba8> formula;
            for (int y = 0; y < kChart; ++y)
                for (int x = 0; x < kChart; ++x) formula.push_back(chart_texel(x, y));
            const Diff d = compare(chart_, formula, kChart, kConvertTolerance);
            check.expect(d.bad == 0, std::format("glReadPixels(RGBA, UNSIGNED_BYTE) of the whole chart matches the analytic chart (+-1 for the shader's float->8-bit conversion; max deviation {})", d.max_dev), d.first);

            // Sub-rectangles: the origin is the lower left corner and rows run bottom to top (section 4.3.2.1).
            const Region regions[] = {{0, 0, 1, 1}, {63, 63, 1, 1}, {5, 9, 13, 7}, {20, 0, 1, 64}, {0, 33, 64, 1}, {16, 16, 32, 32}};
            for (const Region& r : regions)
            {
                const std::vector<Rgba8> got = read_texels(r.x, r.y, r.w, r.h);
                const std::vector<Rgba8> want = block_of(chart_, kChart, r);
                const Diff sub = compare(got, want, r.w, 0);
                check.expect(sub.bad == 0, std::format("glReadPixels sub-rectangle x={} y={} {}x{} equals that block of the full read (exact)", r.x, r.y, r.w, r.h), sub.first);
            }

            // Negative controls: the comparison must be able to fail.
            const Region r{5, 9, 13, 7};
            const std::vector<Rgba8> got = read_texels(r.x, r.y, r.w, r.h);
            std::vector<Rgba8> flipped;  // the same block with its rows in top-down order
            for (int j = r.h - 1; j >= 0; --j)
                for (int i = 0; i < r.w; ++i) flipped.push_back(got[static_cast<std::size_t>(j * r.w + i)]);
            check.expect(compare(flipped, block_of(chart_, kChart, r), r.w, 0).bad > r.w * r.h / 2,
                         "negative control: a top-down (flipped) reading of the sub-rectangle is rejected by the comparison");
            const Region shifted{r.x + 1, r.y, r.w, r.h};
            check.expect(compare(got, block_of(chart_, kChart, shifted), r.w, 0).bad > r.w * r.h / 2,
                         "negative control: a reference that is one pixel to the right is rejected by the comparison");
        }

        // Reading outside the framebuffer is not an error, but the values of the outside pixels are
        // undefined (section 4.3.2.1): only the pixels inside the framebuffer are compared.
        void check_out_of_window(demo::Check& check, demo::Env& env)
        {
            struct Case { const char* name; mgl::FramebufferId fbo; int size; int x, y, w, h; };
            const Case cases[] = {
                {"chart framebuffer, region over the lower left corner", chart_fbo_, kChart, -8, -8, 24, 24},
                {"chart framebuffer, region over the upper right corner", chart_fbo_, kChart, 56, 56, 24, 24},
                {"default framebuffer, region over its upper right corner", mgl::FramebufferId{0}, env.width, env.width - 16, env.height - 16, 32, 32},
            };
            for (const Case& c : cases)
            {
                bind_read(c.fbo);
                const int ix0 = std::max(c.x, 0), iy0 = std::max(c.y, 0);
                const int ix1 = std::min(c.x + c.w, c.size), iy1 = std::min(c.y + c.h, c.size);
                const std::vector<Rgba8> inside = read_texels(ix0, iy0, ix1 - ix0, iy1 - iy0);  // reference: the part inside, read separately
                std::vector<std::uint8_t> buf(static_cast<std::size_t>(c.w) * static_cast<std::size_t>(c.h) * 4, 0xA5);
                (void)demo::drain_gl_errors();
                mgl::glReadPixels(c.x, c.y, c.w, c.h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, buf.data());
                const mgl::ErrorCode err = demo::drain_gl_errors();
                check.expect(err == mgl::ErrorCode::NoError, std::format("{}: glReadPixels over the edge is not an error", c.name), std::string(demo::error_name(err)));
                int bad = 0, untouched = 0;
                for (int j = 0; j < c.h; ++j)
                    for (int i = 0; i < c.w; ++i)
                    {
                        const int fx = c.x + i, fy = c.y + j;
                        const std::uint8_t* px = &buf[(static_cast<std::size_t>(j) * static_cast<std::size_t>(c.w) + static_cast<std::size_t>(i)) * 4];
                        if (fx >= 0 && fy >= 0 && fx < c.size && fy < c.size)
                        {
                            const Rgba8 want = inside[static_cast<std::size_t>(fy - iy0) * static_cast<std::size_t>(ix1 - ix0) + static_cast<std::size_t>(fx - ix0)];
                            if (std::memcmp(px, want.data(), 4) != 0) ++bad;
                        }
                        else if (px[0] == 0xA5 && px[1] == 0xA5 && px[2] == 0xA5 && px[3] == 0xA5)
                        {
                            ++untouched;  // observation only: outside values are undefined
                        }
                    }
                check.expect(bad == 0, std::format("{}: every pixel inside the framebuffer is correct in the same call ({} outside pixels were left untouched by this driver; their values are undefined)", c.name, untouched));
            }
        }

        // Reads w x h pixels with the pair into a buffer laid out with PACK_ALIGNMENT `alignment`.
        std::vector<std::uint8_t> read_pairs(const Pair& pair, const Region& r, int alignment, std::size_t extra, int* stride)
        {
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, alignment);
            *stride = row_stride_bytes(pair.layout, r.w, alignment);
            std::vector<std::uint8_t> buf(static_cast<std::size_t>(*stride) * static_cast<std::size_t>(r.h) + extra, 0xA5);
            mgl::glReadPixels(r.x, r.y, r.w, r.h, pair.format, pair.type, buf.data());
            return buf;
        }

        // Checks the row layout for PACK_ALIGNMENT 1, 2, 4 and 8 against the tightly packed read, group by group.
        // Returns a description of the observed strides; fills `ok` and the number of padding bytes left untouched.
        bool alignment_layout_ok(const Pair& pair, const Region& r, std::string& detail, int& untouched_padding)
        {
            int stride1 = 0;
            const std::vector<std::uint8_t> tight = read_pairs(pair, r, 1, 0, &stride1);
            bool ok = true;
            untouched_padding = 0;
            detail.clear();
            for (const int a : {1, 2, 4, 8})
            {
                int stride = 0;
                const std::vector<std::uint8_t> buf = read_pairs(pair, r, a, 16, &stride);
                const int gb = pair.group_bytes();
                int bad = 0;
                for (int j = 0; j < r.h; ++j)
                    for (int i = 0; i < r.w; ++i)
                        if (std::memcmp(&buf[static_cast<std::size_t>(j * stride + i * gb)], &tight[static_cast<std::size_t>(j * stride1 + i * gb)], static_cast<std::size_t>(gb)) != 0) ++bad;
                // Nothing may be written behind the last row (the spec places groups only at row starts + i*group).
                for (std::size_t k = static_cast<std::size_t>(stride * r.h); k < buf.size(); ++k)
                    if (buf[k] != 0xA5) ++bad;
                for (int j = 0; j < r.h; ++j)
                    for (int k = r.w * gb; k < stride; ++k)
                        if (buf[static_cast<std::size_t>(j * stride + k)] == 0xA5) ++untouched_padding;
                ok = ok && bad == 0;
                detail += std::format("{}{}:{}B{}", detail.empty() ? "" : ", ", a, stride, bad == 0 ? "" : " WRONG");
            }
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
            return ok;
        }

        void check_pack_alignment(demo::Check& check)
        {
            bind_read(chart_fbo_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::PackAlignment), 4, "the initial PACK_ALIGNMENT is 4 (glGetIntegerv, Table 4.4)");

            // RGBA / UNSIGNED_BYTE is the one pair that is always allowed. A pixel is 4 bytes, so a 3-pixel row is
            // 12 bytes: alignments 1, 2, 4 add no padding, alignment 8 pads each row to 16 bytes.
            const Pair rgba = *make_pair(mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte);
            const Region r{4, 6, 3, 5};
            int stride1 = 0;
            const std::vector<std::uint8_t> tight = read_pairs(rgba, r, 1, 0, &stride1);
            check.expect(stride1 == 12, "3 RGBA/UNSIGNED_BYTE pixels per row are 12 bytes with PACK_ALIGNMENT 1");
            for (const int a : {1, 2, 4, 8})
            {
                int stride = 0;
                const std::vector<std::uint8_t> buf = read_pairs(rgba, r, a, 16, &stride);
                const int expected_stride = a == 8 ? 16 : 12;  // equation 3.13: k = (a/s) * ceil(s*n*l / a) when s < a
                int bad = 0;
                for (int j = 0; j < r.h; ++j)
                    for (int i = 0; i < r.w; ++i)
                    {
                        const Rgba8 want = chart_texel(r.x + i, r.y + j);
                        const std::uint8_t* got = &buf[static_cast<std::size_t>(j * expected_stride + i * 4)];
                        for (std::size_t k = 0; k < 4; ++k)
                            if (std::abs(int{got[k]} - int{want[k]}) > kConvertTolerance) ++bad;
                    }
                check.expect(stride == expected_stride && bad == 0,
                             std::format("PACK_ALIGNMENT {}: rows of 3 RGBA pixels start {} bytes apart (equation 3.13) and all 15 pixels are where that layout puts them", a, expected_stride),
                             std::format("{} wrong values", bad));
                check.expect_eq(demo::get_integer(mgl::GetParameter::PackAlignment), a, std::format("glGetIntegerv(PACK_ALIGNMENT) returns {} after glPixelStorei", a));
            }

            // Negative control: the alignment-8 buffer interpreted as tightly packed does not match.
            {
                int stride8 = 0;
                const std::vector<std::uint8_t> padded = read_pairs(rgba, r, 8, 0, &stride8);
                int bad = 0;
                for (int j = 0; j < r.h; ++j)
                    for (int i = 0; i < r.w; ++i)
                        if (std::memcmp(&padded[static_cast<std::size_t>(j * stride1 + i * 4)], &tight[static_cast<std::size_t>(j * stride1 + i * 4)], 4) != 0) ++bad;
                check.expect(stride8 == 16 && bad > 0, "negative control: reading the alignment-8 buffer as if it were tightly packed gives wrong pixels (the padding really is there)",
                             std::format("{} of 15 pixels differ", bad));
                mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
            }

            // The rule is generic over (format, type): also verify RGB, 565, 4444, 5551 and 2_10_10_10_REV layouts
            // when this implementation chooses them (see check_implementation_pair).
        }

        void check_pack_rows(demo::Check& check)
        {
            bind_read(chart_fbo_);
            // ES 3.0 adds PACK_ROW_LENGTH, PACK_SKIP_PIXELS and PACK_SKIP_ROWS: the rectangle is stored inside a larger one.
            const Region r{10, 20, 5, 4};
            const int row_length = 8, skip_pixels = 2, skip_rows = 1, alignment = 4;
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, alignment);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, row_length);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, skip_pixels);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, skip_rows);
            check.expect(demo::get_integer(mgl::GetParameter::PackRowLength) == row_length && demo::get_integer(mgl::GetParameter::PackSkipPixels) == skip_pixels &&
                             demo::get_integer(mgl::GetParameter::PackSkipRows) == skip_rows,
                         "glGetIntegerv returns PACK_ROW_LENGTH, PACK_SKIP_PIXELS and PACK_SKIP_ROWS as set");
            const Layout rgba{4, 1};
            const int stride = row_stride_bytes(rgba, row_length, alignment);  // 32 bytes
            const int first = skip_pixels * 4 + skip_rows * stride;            // the pointer is advanced by skip_pixels*n + skip_rows*k elements
            std::vector<std::uint8_t> buf(static_cast<std::size_t>(first + stride * r.h), 0xA5);
            mgl::glReadPixels(r.x, r.y, r.w, r.h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, buf.data());
            int bad = 0, wrong_layout = 0;
            for (int j = 0; j < r.h; ++j)
                for (int i = 0; i < r.w; ++i)
                {
                    const Rgba8 want = chart_texel(r.x + i, r.y + j);
                    const std::uint8_t* at = &buf[static_cast<std::size_t>(first + j * stride + i * 4)];
                    for (std::size_t k = 0; k < 4; ++k)
                        if (std::abs(int{at[k]} - int{want[k]}) > kConvertTolerance) ++bad;
                    const std::uint8_t* tight_at = &buf[static_cast<std::size_t>((j * r.w + i) * 4)];  // where a plain tight read would be
                    if (std::memcmp(tight_at, want.data(), 4) != 0) ++wrong_layout;
                }
            check.expect(bad == 0, std::format("PACK_ROW_LENGTH {} / SKIP_PIXELS {} / SKIP_ROWS {}: the 5x4 block lands at byte {} with a {}-byte row pitch", row_length, skip_pixels, skip_rows, first, stride),
                         std::format("{} wrong values", bad));
            check.expect(wrong_layout > r.w * r.h / 2, "negative control: the same bytes read as a tight 5x4 block are wrong (the skip and row length really moved the data)");
            mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
        }

        // ---- IMPLEMENTATION_COLOR_READ_FORMAT / _TYPE ------------------------------------------------------
        // The chart (RGBA8 data) and the default framebuffer are read with the implementation-chosen pair and
        // compared with the RGBA/UNSIGNED_BYTE read; renderbuffers of several formats are cleared to a known
        // color, read with whatever pair each one names, and decoded.
        void compare_pair_with_rgba8(demo::Check& check, std::string_view name, const Region& r)
        {
            GLint raw_f = 0, raw_t = 0;
            const std::optional<Pair> pair = implementation_pair(&raw_f, &raw_t);
            if (!pair)
            {
                demo_note(std::format("{}: the implementation-chosen pair ({}) is not one this demo decodes; not compared", name, describe(raw_f, raw_t)));
                return;
            }
            const std::vector<Rgba8> ref = read_texels(r.x, r.y, r.w, r.h);
            int stride = 0;
            const std::vector<std::uint8_t> buf = read_pairs(*pair, r, 1, 0, &stride);
            int bad = 0;
            for (int j = 0; j < r.h; ++j)
                for (int i = 0; i < r.w; ++i)
                {
                    const std::array<double, 4> v = decode(*pair, &buf[static_cast<std::size_t>(j * stride + i * pair->group_bytes())]);
                    const Rgba8 want = ref[static_cast<std::size_t>(j * r.w + i)];
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        if (pair->bits[c] == 0) continue;  // not carried by this pair
                        // The stored value is an 8-bit one; converting it to the pair's width moves it by at most one step of that width.
                        if (std::abs(v[c] - want[c] / 255.0) > step(pair->bits[c]) + 1e-9) ++bad;
                    }
                }
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
            check.expect(bad == 0, std::format("{}: reading with the implementation-chosen pair ({}) agrees with RGBA/UNSIGNED_BYTE", name, describe(raw_f, raw_t)), std::format("{} channel values differ", bad));
        }

        void check_implementation_pair(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            compare_pair_with_rgba8(check, "default framebuffer (inside the chart cell)", {8, cell_y(kChartCell) + 8, 5, 4});
            bind_read(chart_fbo_);
            compare_pair_with_rgba8(check, "chart framebuffer (RGBA8 renderbuffer)", {3, 5, 11, 9});
            // The pair a normalized RGBA8 surface names is one every ES 3.0 implementation can honour.
            GLint raw_f = 0, raw_t = 0;
            (void)implementation_pair(&raw_f, &raw_t);
            check.expect(raw_f != 0 && raw_t != 0, "an RGBA8 framebuffer reports non-zero IMPLEMENTATION_COLOR_READ_FORMAT and _TYPE", describe(raw_f, raw_t));

            struct Sized { mgl::InternalFormat format; const char* name; };
            const Sized sized[] = {
                {mgl::InternalFormat::Rgba8, "RGBA8"}, {mgl::InternalFormat::Rgb8, "RGB8"}, {mgl::InternalFormat::Rgb565, "RGB565"},
                {mgl::InternalFormat::Rgba4, "RGBA4"}, {mgl::InternalFormat::Rgb5A1, "RGB5_A1"}, {mgl::InternalFormat::Rgb10A2, "RGB10_A2"},
                {mgl::InternalFormat::Rg8, "RG8"}, {mgl::InternalFormat::R8, "R8"},
            };
            const std::array<double, 4> clear = {0.8, 0.4, 0.2, 0.6};
            for (const Sized& s : sized)
            {
                mgl::FramebufferId fbo{};
                mgl::RenderbufferId rb{};
                make_target(fbo, rb, s.format, 16, 16);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
                std::array<GLint, 4> storage{};
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::RedSize, &storage[0]);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::GreenSize, &storage[1]);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::BlueSize, &storage[2]);
                mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::AlphaSize, &storage[3]);
                mgl::glClearColor(static_cast<float>(clear[0]), static_cast<float>(clear[1]), static_cast<float>(clear[2]), static_cast<float>(clear[3]));
                mgl::glClear(mgl::ClearBufferBit::Color);
                GLint raw_f2 = 0, raw_t2 = 0;
                const std::optional<Pair> pair = implementation_pair(&raw_f2, &raw_t2);
                if (!pair)
                {
                    demo_note(std::format("{} renderbuffer names an implementation read pair this demo does not decode ({}); not compared", s.name, describe(raw_f2, raw_t2)));
                }
                else
                {
                    const Region r{2, 3, 3, 2};
                    int stride = 0;
                    const std::vector<std::uint8_t> buf = read_pairs(*pair, r, 1, 0, &stride);
                    int bad = 0;
                    std::string seen;
                    for (int j = 0; j < r.h; ++j)
                        for (int i = 0; i < r.w; ++i)
                        {
                            const std::array<double, 4> v = decode(*pair, &buf[static_cast<std::size_t>(j * stride + i * pair->group_bytes())]);
                            for (std::size_t c = 0; c < 4; ++c)
                            {
                                if (pair->bits[c] == 0) continue;
                                // Absent components read as 0, 0, 0, 1 (section 4.3.2.1); present ones are the clear color moved through
                                // two conversions, each within one step of its width (eq. 2.3): the storage's and the pair's.
                                const bool present = storage[c] > 0;
                                const double want = present ? clear[c] : (c == 3 ? 1.0 : 0.0);
                                const double tol = (present ? step(storage[c]) : 0.0) + step(pair->bits[c]) + 1e-9;
                                if (std::abs(v[c] - want) > tol) ++bad;
                            }
                            if (i == 0 && j == 0) seen = std::format("({:.3f},{:.3f},{:.3f},{:.3f})", v[0], v[1], v[2], v[3]);
                        }
                    check.expect(bad == 0, std::format("{} renderbuffer cleared to (0.8,0.4,0.2,0.6): read with its implementation pair ({}) decodes to the expected color (storage bits {}/{}/{}/{})",
                                                       s.name, describe(raw_f2, raw_t2), storage[0], storage[1], storage[2], storage[3]),
                                 std::format("{} channel values wrong, first pixel {}", bad, seen));
                    std::string detail;
                    int untouched = 0;
                    const bool ok = alignment_layout_ok(*pair, {1, 1, 3, 2}, detail, untouched);
                    check.expect(ok, std::format("{} renderbuffer: with its implementation pair a 3-pixel row has these strides for PACK_ALIGNMENT 1/2/4/8 (equation 3.13): {}", s.name, detail));
                    if (s.format == mgl::InternalFormat::Rgb565 && pair->format == mgl::PixelFormat::Rgb)
                    {
                        // The classic padding case: a 3-pixel RGB row is not a multiple of 4 bytes, so PACK_ALIGNMENT 4 and 1 differ.
                        const int tight = row_stride_bytes(pair->layout, 3, 1), aligned = row_stride_bytes(pair->layout, 3, 4);
                        check.expect(ok && aligned > tight, std::format("RGB565 renderbuffer, RGB pair: a 3-pixel row is {} bytes with PACK_ALIGNMENT 1 and {} with PACK_ALIGNMENT 4 (padding), and every pixel sits where that layout puts it", tight, aligned));
                    }
                    demo_note(std::format("{}: padding bytes left untouched by this driver: {} (their values are not specified)", s.name, untouched));
                }
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
                mgl::glDeleteFramebuffers(1, &fbo);
                mgl::glDeleteRenderbuffers(1, &rb);
            }
            bind_read(chart_fbo_);
        }

        static void demo_note(const std::string& text) { std::cout << "   [note] " << text << "\n"; }

        // ===========================================================================================
        // glCopyTexImage2D / glCopyTexSubImage2D
        // ===========================================================================================
        void check_copy_tex_image(demo::Check& check)
        {
            for (const CopyCase& c : kCases)
            {
                std::vector<Rgba8> want;
                for (int j = 0; j < c.src.h; ++j)
                    for (int i = 0; i < c.src.w; ++i)
                        want.push_back(expected_texel(c.internal, chart_[static_cast<std::size_t>(c.src.y + j) * kChart + static_cast<std::size_t>(c.src.x + i)]));

                // (1) the texels, exactly, through a framebuffer the texture is attached to
                if (c.renderable)
                {
                    const std::optional<std::vector<Rgba8>> direct = read_attached(tex_[c.tex], c.src.w, c.src.h);
                    if (!direct)
                    {
                        check.expect(false, std::format("{}: the copied texture can be attached to a framebuffer", c.name));
                    }
                    else
                    {
                        const Diff d = compare(*direct, want, c.src.w, 0);
                        check.expect(d.bad == 0, std::format("glCopyTexImage2D {}: {}x{} texels read back through an attached framebuffer equal the source pixels (exact)", c.name, c.src.w, c.src.h), d.first);
                    }
                }
                // (2) what a shader sees: NEAREST sampling at every texel center
                const std::vector<Rgba8> seen = sampled(tex_[c.tex], c.src.w, c.src.h, 2);
                const Diff s = compare(seen, want, c.src.w, kConvertTolerance);
                check.expect(s.bad == 0, std::format("glCopyTexImage2D {}: sampling the {}x{} texture returns the expected components ({}; max deviation {})", c.name, c.src.w, c.src.h,
                                                     c.internal == mgl::InternalFormat::Luminance ? "(L,L,L,1), L = R" : c.internal == mgl::InternalFormat::Alpha ? "(0,0,0,A)" :
                                                     c.internal == mgl::InternalFormat::LuminanceAlpha ? "(L,L,L,A)" : "dropped components read as 0,0,0,1", s.max_dev), s.first);
                // (3) the cell on the screen
                std::vector<Rgba8> cell_want;
                for (int j = 0; j < c.src.h * c.scale; ++j)
                    for (int i = 0; i < c.src.w * c.scale; ++i)
                    {
                        const Rgba8 t = want[static_cast<std::size_t>(j / c.scale) * static_cast<std::size_t>(c.src.w) + static_cast<std::size_t>(i / c.scale)];
                        cell_want.push_back(c.display_mode == 1 ? Rgba8{t[3], t[3], t[3], 255} : Rgba8{t[0], t[1], t[2], 255});
                    }
                mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
                const std::vector<Rgba8> cell = read_texels(cell_x(c.cell), cell_y(c.cell), c.src.w * c.scale, c.src.h * c.scale);
                const Diff cd = compare(cell, cell_want, c.src.w * c.scale, kConvertTolerance);
                check.expect(cd.bad == 0, std::format("{}: the cell on the default framebuffer shows the texture (x{} magnified; max deviation {})", c.name, c.scale, cd.max_dev), cd.first);
            }

            // The chart itself was blitted to cell 0.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            const Diff chart_cell = compare(read_texels(cell_x(kChartCell), cell_y(kChartCell), kChart, kChart), chart_, kChart, 0);
            check.expect(chart_cell.bad == 0, "cell 0 on the default framebuffer shows the chart", chart_cell.first);

            // Negative control: a reference with the wrong source rectangle is rejected.
            {
                const CopyCase& c = kCases[2];  // RGB sub-rectangle
                const std::optional<std::vector<Rgba8>> direct = read_attached(tex_[c.tex], c.src.w, c.src.h);
                const Region wrong{c.src.x + 4, c.src.y, c.src.w, c.src.h};
                std::vector<Rgba8> want_wrong;
                for (const Rgba8 px : block_of(chart_, kChart, wrong)) want_wrong.push_back(expected_texel(c.internal, px));
                check.expect(direct && compare(*direct, want_wrong, c.src.w, 0).bad > c.src.w * c.src.h / 2,
                             "negative control: comparing the RGB copy with a block shifted by 4 pixels is rejected");
            }
            // The copy read the READ framebuffer: the DRAW framebuffer (the default one) held different pixels.
            check.expect(chart_ != read_default_block(), "the copies came from the READ framebuffer: the DRAW (default) framebuffer does not hold the chart at the same place, yet the copies equal the chart");
        }

        std::vector<Rgba8> read_default_block()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return read_texels(0, 0, kChart, kChart);
        }

        void check_copy_tex_sub_image(demo::Check& check)
        {
            std::vector<Rgba8> want(static_cast<std::size_t>(kChart) * kChart, kBaseColor);
            for (const Patch& p : kPatches)
                for (int j = 0; j < p.src.h; ++j)
                    for (int i = 0; i < p.src.w; ++i)
                        want[static_cast<std::size_t>(p.yoffset + j) * kChart + static_cast<std::size_t>(p.xoffset + i)] =
                            chart_[static_cast<std::size_t>(p.src.y + j) * kChart + static_cast<std::size_t>(p.src.x + i)];
            const std::optional<std::vector<Rgba8>> direct = read_attached(tex_[kTexSub], kChart, kChart);
            check.expect(direct.has_value(), "the glCopyTexSubImage2D texture can be attached to a framebuffer");
            if (direct)
            {
                const Diff d = compare(*direct, want, kChart, 0);
                check.expect(d.bad == 0, "glCopyTexSubImage2D: two patches replace exactly their rectangles of the 64x64 texture, every other texel keeps its value (exact)", d.first);
            }
            const Diff s = compare(sampled(tex_[kTexSub], kChart, kChart, 2), want, kChart, kConvertTolerance);
            check.expect(s.bad == 0, std::format("glCopyTexSubImage2D: sampling the patched texture matches (max deviation {})", s.max_dev), s.first);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            std::vector<Rgba8> shown = want;  // the cell is drawn with u_mode 0: (r, g, b, 1)
            for (Rgba8& px : shown) px[3] = 255;
            const Diff cd = compare(read_texels(cell_x(kSubCell), cell_y(kSubCell), kChart, kChart), shown, kChart, kConvertTolerance);
            check.expect(cd.bad == 0, "the patched texture on the default framebuffer (cell 2) matches", cd.first);
            // Negative control: pretending the patch had not been applied must be detected.
            const std::vector<Rgba8> unpatched(static_cast<std::size_t>(kChart) * kChart, kBaseColor);
            check.expect(direct && compare(*direct, unpatched, kChart, 0).bad > 400, "negative control: comparing with the unpatched base color is rejected (the patches changed >400 texels)");
        }

        // ===========================================================================================
        // Defined errors
        // ===========================================================================================
        void check_copy_errors(demo::Check& check)
        {
            using E = mgl::ErrorCode;
            using IF = mgl::InternalFormat;
            mgl::TextureId tex{};
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            bind_read(chart_fbo_);

            // Table 3.16: color components may be dropped, never added. A source without alpha cannot make a texture with alpha.
            mgl::FramebufferId rgb_fbo{};
            mgl::RenderbufferId rgb_rb{};
            make_target(rgb_fbo, rgb_rb, IF::Rgb8, 16, 16);
            bind_read(rgb_fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, rgb_fbo);
            mgl::glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            const struct { IF format; const char* name; } adding[] = {{IF::Rgba, "RGBA"}, {IF::Alpha, "ALPHA"}, {IF::LuminanceAlpha, "LUMINANCE_ALPHA"}};
            for (const auto& a : adding)
                expect_error(check, E::InvalidOperation, std::format("glCopyTexImage2D(internalformat {}) from an RGB8 framebuffer -> INVALID_OPERATION (a component would have to be added)", a.name),
                             [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, a.format, 0, 0, 8, 8, 0); });
            mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Rgb, 0, 0, 8, 8, 0);
            mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Luminance, 0, 0, 8, 8, 0);
            check.expect_no_gl_error("control: from the same RGB8 framebuffer, RGB and LUMINANCE are valid (components are only dropped)");
            mgl::glDeleteFramebuffers(1, &rgb_fbo);
            mgl::glDeleteRenderbuffers(1, &rgb_rb);

            // A sized internal format must have exactly the source's component sizes (section 3.8.5).
            bind_read(chart_fbo_);
            expect_error(check, E::InvalidOperation, "glCopyTexImage2D(RGB565) from an RGBA8 framebuffer -> INVALID_OPERATION (sized formats must match the source's component sizes)",
                         [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Rgb565, 0, 0, 8, 8, 0); });
            expect_error(check, E::InvalidOperation, "glCopyTexImage2D(RGBA4) from an RGBA8 framebuffer -> INVALID_OPERATION",
                         [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Rgba4, 0, 0, 8, 8, 0); });
            mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Rgba8, 0, 0, 8, 8, 0);
            check.expect_no_gl_error("control: RGBA8 from an RGBA8 framebuffer is valid");

            // Arguments follow glTexImage2D.
            expect_error(check, E::InvalidValue, "glCopyTexImage2D with border = 1 -> INVALID_VALUE",
                         [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Rgba, 0, 0, 8, 8, 1); });
            expect_error(check, E::InvalidValue, "glCopyTexImage2D to a cube map face with width != height -> INVALID_VALUE",
                         [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::TextureCubeMapPositiveX, 0, IF::Rgba, 0, 0, 8, 4, 0); });
            expect_error(check, E::InvalidValue, "glCopyTexSubImage2D with x + width > the texture level's width -> INVALID_VALUE",
                         [&] { mgl::glCopyTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 4, 0, 0, 0, 8, 8); });
            expect_error(check, E::InvalidValue, "glCopyTexSubImage2D with a negative xoffset -> INVALID_VALUE",
                         [&] { mgl::glCopyTexSubImage2D(mgl::TextureTarget::Texture2D, 0, -1, 0, 0, 0, 4, 4); });

            // READ_BUFFER: nothing to copy from.
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            expect_error(check, E::InvalidOperation, "glCopyTexImage2D with READ_BUFFER = NONE -> INVALID_OPERATION",
                         [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, IF::Rgba, 0, 0, 8, 8, 0); });
            expect_error(check, E::InvalidOperation, "glCopyTexSubImage2D with READ_BUFFER = NONE -> INVALID_OPERATION",
                         [&] { mgl::glCopyTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 0, 0, 4, 4); });
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &tex);
        }

        void check_read_errors(demo::Check& check)
        {
            using E = mgl::ErrorCode;
            bind_read(chart_fbo_);
            std::array<std::uint8_t, 64> buf{};
            // Section 3.7.2.2: a packed type fixes the number of components; format and type must agree.
            expect_error(check, E::InvalidOperation, "glReadPixels(RGB, UNSIGNED_SHORT_4_4_4_4) -> INVALID_OPERATION (the packed type has 4 components, the format 3)",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgb, mgl::PixelType::UnsignedShort4444, buf.data()); });
            expect_error(check, E::InvalidOperation, "glReadPixels(RGBA, UNSIGNED_SHORT_5_6_5) -> INVALID_OPERATION (the packed type has 3 components, the format 4)",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedShort565, buf.data()); });
            // Section 4.3.2.1: an integer format needs an integer color buffer.
            expect_error(check, E::InvalidOperation, "glReadPixels(RGBA_INTEGER, UNSIGNED_INT) from a normalized fixed-point buffer -> INVALID_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, buf.data()); });
            // The accepted pair still works afterwards (the failed calls did not change state).
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, buf.data());
            check.expect_no_gl_error("control: RGBA/UNSIGNED_BYTE is accepted");

            // Pixel storage ranges (Tables 3.1 and 4.4).
            expect_error(check, E::InvalidValue, "glPixelStorei(PACK_ALIGNMENT, 3) -> INVALID_VALUE (only 1, 2, 4, 8)",
                         [&] { mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 3); });
            expect_error(check, E::InvalidValue, "glPixelStorei(PACK_ROW_LENGTH, -1) -> INVALID_VALUE (range [0, inf))",
                         [&] { mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, -1); });
            check.expect_eq(demo::get_integer(mgl::GetParameter::PackAlignment), 4, "a rejected glPixelStorei leaves PACK_ALIGNMENT unchanged");

            // READ_BUFFER problems: nothing to read from.
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            expect_error(check, E::InvalidOperation, "glReadPixels with READ_BUFFER = NONE -> INVALID_OPERATION (section 4.3.2.1)",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, buf.data()); });
            expect_error(check, E::InvalidOperation, "IMPLEMENTATION_COLOR_READ_FORMAT with READ_BUFFER = NONE -> INVALID_OPERATION (section 4.3.2)",
                         [&] { (void)demo::get_integer(mgl::GetParameter::ImplementationColorReadFormat); });
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color1));
            expect_error(check, E::InvalidOperation, "glReadPixels from COLOR_ATTACHMENT1, which has no image attached -> INVALID_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, buf.data()); });
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, buf.data());
            check.expect_no_gl_error("control: with READ_BUFFER = COLOR_ATTACHMENT0 reading works again");
        }

        mgl::ProgramId chart_program_{}, show_program_{};
        mgl::UniformLocation chart_rect_{}, show_rect_{}, show_tex_{}, show_mode_{};
        mgl::FramebufferId chart_fbo_{}, scratch_fbo_{}, attach_fbo_{};
        mgl::RenderbufferId chart_rb_{}, scratch_rb_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        std::vector<Rgba8> chart_;  // the chart as stored in the framebuffer, read in validate()
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-copy-and-read-pixels",
        .title = "Copy and read pixels: glReadPixels, glCopyTexImage2D, glCopyTexSubImage2D",
        .min_es = {3, 0},
        .summary = "Reading framebuffer pixels (required and implementation-chosen format, sub-rectangles, PACK_ALIGNMENT/ROW_LENGTH/SKIP_*, "
                   "reads past the edge) and copying them into textures of several internal formats, checked texel by texel.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<CopyAndReadPixels>(argc, argv, info);
}
