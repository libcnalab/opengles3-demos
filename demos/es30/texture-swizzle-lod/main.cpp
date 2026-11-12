// es30-texture-swizzle-lod
//
// Two things ES 3.0 added to texture access, checked against CPU models:
//
//  * texture SWIZZLE (ES 3.0 3.8.7, 3.9.2.1): TEXTURE_SWIZZLE_R/G/B/A select, for each component of
//    the value a lookup returns, one of RED, GREEN, BLUE, ALPHA, ZERO or ONE. The swizzle is applied to
//    the texture base color Cb of Table 3.24 (so a single-channel R8 texture has G = B = 0 and A = 1 to
//    swizzle from), it belongs to the TEXTURE (a sampler object cannot carry it) and it affects every
//    lookup function including texelFetch.
//
//  * lookups whose level of detail the SHADER chooses: textureLod, textureGrad, textureProjLod,
//    textureSize and texelFetch (levels counted from TEXTURE_BASE_LEVEL, never clamped by MIN/MAX_LOD),
//    texel offsets at MIN/MAX_PROGRAM_TEXEL_OFFSET (textureOffset, textureProjOffset, textureLodOffset,
//    textureGradOffset, texelFetchOffset), and textureProj / textureProjLod. es20-mipmaps already
//    verifies the implicit-LOD side (level selection by screen-space derivatives, all four mipmap
//    filters, the effect of BASE/MAX_LEVEL and MIN/MAX_LOD on texture()); this demo does not repeat it.
//
// The cells are drawn into an offscreen RGBA8 "contact sheet" of 16 x 16 pixel cells, 16 per row, row 0 at the bottom
// (256 pixels wide, as high as the rows in use). Every cell is one draw call whose result is a known color or a known
// pattern; sections start on a new row. The cells measure exact pixel positions, screen-space derivatives and per-pixel
// texelFetch coordinates, so they keep their 16-pixel design at every window size; the window shows the finished sheet
// stretched over the whole surface (shaders/present.frag, integer texel mapping):
//
//   rows 0-1   swizzle      17 cells: R8, RG8, LUMINANCE_ALPHA, ALPHA and RGBA8 textures under different
//                           swizzles, one swizzled texture through four lookup functions, and through a sampler
//   row  2     textureLod   13 cells on a 7-level ladder texture: every cell is one solid level color
//   row  3     textureGrad  7 cells, and textureProj / textureProjLod 4 cells
//   row  4     textureSize  10 cells, each cell encodes (width, height) in its red and green byte
//   row  5     texelFetch   ladder (1 pixel) and 13 x 13 chain (full level) cells with BASE/MAX_LEVEL, MIN/MAX_LOD
//   rows 6-8   offsets      8 lookup functions x 5 offsets, each cell a 13 x 13 (or 6 x 6) pixel image
//
// Validation: every pixel of every cell is compared with the CPU model of that cell (a swizzled
// Table 3.24 base color, a solid level color, a wrapped/mirrored texel of a chain level, a bilinear mix,
// an encoded size); the swizzle state, the limits and the defined errors are queried and provoked; three
// negative controls show that the cell comparisons can fail. Every window pixel is compared with the sheet
// texel the presentation maps it to.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using S = mgl::TextureSwizzle;
using Wrap = mgl::TextureWrapMode;
using Min = mgl::TextureMinFilter;

namespace
{
    constexpr int kCellPx = 16;  // edge of a cell in pixels
    constexpr int kCols = 16;
    constexpr int kSwz = 4;      // swizzle textures are 4 x 4 texels (4 pixels per texel in a cell)

    constexpr int kLadderW = 64;  // ladder: 64 x 16, 7 levels, every level one solid color
    constexpr int kLadderH = 16;
    constexpr int kLadderLevels = 7;
    constexpr int kChain = 13;     // chain: 13 x 13 with levels 13, 6, 3, 1 (a non-power-of-two: floor(13 / 2^k))
    constexpr int kChainLevels = 4;

    constexpr std::uint8_t u8(int v) { return static_cast<std::uint8_t>(v); }

    // ---------------------------------------------------------------------------------------------
    // Source texels of the 4 x 4 swizzle textures. Every texel has four different channel values, none
    // of them 0 or 255 (the values ZERO and ONE produce), so a wrong swizzle cannot hide.
    // ---------------------------------------------------------------------------------------------
    constexpr int channel(int n, int c)
    {
        switch (c)
        {
        case 0: return 16 + 15 * n;
        case 1: return 240 - 14 * n;
        case 2: return 24 + ((n + 1) % 16) * 14;
        default: return 40 + ((n + 3) % 16) * 12;
        }
    }

    constexpr bool channels_distinct()
    {
        for (int n = 0; n < kSwz * kSwz; ++n)
            for (int a = 0; a < 4; ++a)
            {
                if (channel(n, a) <= 0 || channel(n, a) >= 255) return false;
                for (int b = a + 1; b < 4; ++b)
                    if (channel(n, a) == channel(n, b)) return false;
            }
        return true;
    }
    static_assert(channels_distinct(), "all four channels of every source texel differ and avoid 0 / 255");

    constexpr Rgba8 src_texel(int i, int j)
    {
        const int n = j * kSwz + i;
        return {u8(channel(n, 0)), u8(channel(n, 1)), u8(channel(n, 2)), u8(channel(n, 3))};
    }

    // ---------------------------------------------------------------------------------------------
    // ES 3.0 Table 3.24: texture base color Cb (+ Ab) from the filtered texel of a texture whose base
    // internal format is `fmt`. The swizzle of 3.9.2.1 is applied to THIS color, not to the stored texel.
    // ---------------------------------------------------------------------------------------------
    enum class Fmt { R8, Rg8, LuminanceAlpha, Alpha, Rgba8 };

    constexpr Rgba8 base_color(Fmt fmt, Rgba8 s)
    {
        switch (fmt)
        {
        case Fmt::R8: return {s[0], 0, 0, 255};
        case Fmt::Rg8: return {s[0], s[1], 0, 255};
        case Fmt::LuminanceAlpha: return {s[0], s[0], s[0], s[1]};  // L = first byte, A = second byte of the group
        case Fmt::Alpha: return {0, 0, 0, s[3]};
        case Fmt::Rgba8: return s;
        }
        return s;
    }

    using Swz = std::array<S, 4>;
    constexpr Swz kIdentity = {S::Red, S::Green, S::Blue, S::Alpha};

    // 3.9.2.1: Cs[c] = Cb[0], Cb[1], Cb[2], Ab, 0 or 1 according to the swizzle of component c.
    constexpr Rgba8 swizzle(const Swz& z, Rgba8 cb)
    {
        Rgba8 out{};
        for (std::size_t c = 0; c < 4; ++c)
        {
            switch (z[c])
            {
            case S::Red: out[c] = cb[0]; break;
            case S::Green: out[c] = cb[1]; break;
            case S::Blue: out[c] = cb[2]; break;
            case S::Alpha: out[c] = cb[3]; break;
            case S::Zero: out[c] = 0; break;
            case S::One: out[c] = 255; break;
            }
        }
        return out;
    }

    constexpr GLint value_of(S s) { return static_cast<GLint>(static_cast<GLenum>(s)); }

    // ---------------------------------------------------------------------------------------------
    // The ladder (64 x 16, solid levels) and the chain (13 x 13, patterned levels).
    // ---------------------------------------------------------------------------------------------
    constexpr std::array<Rgba8, kLadderLevels> kLevelColor = {{
        {230, 40, 40, 255}, {40, 200, 60, 255}, {50, 70, 230, 255}, {230, 210, 40, 255},
        {200, 60, 200, 255}, {40, 200, 200, 255}, {240, 240, 240, 255}}};

    constexpr int ladder_w(int level) { return std::max(1, kLadderW >> level); }
    constexpr int ladder_h(int level) { return std::max(1, kLadderH >> level); }
    constexpr int chain_size(int level) { return std::max(1, kChain >> level); }

    // Level `level`, texel (i, j): red grows with i, green with j, blue with the level. All distinct.
    constexpr Rgba8 chain_texel(int level, int i, int j) { return {u8(12 + 18 * i), u8(12 + 18 * j), u8(40 + 70 * level), 255}; }

    int wrap_coord(int c, int size, Wrap mode)
    {
        if (mode == Wrap::Repeat) return ((c % size) + size) % size;
        return std::clamp(c, 0, size - 1);  // CLAMP_TO_EDGE
    }

    Rgba8 mix8(Rgba8 a, Rgba8 b, double t)  // (1 - t) a + t b, rounded
    {
        Rgba8 r{};
        for (std::size_t k = 0; k < 4; ++k) r[k] = u8(static_cast<int>(std::lround((1.0 - t) * a[k] + t * b[k])));
        return r;
    }

    int blend_tolerance(Rgba8 a, Rgba8 b)  // 1 + |tau2 - tau1| / 16: the blend weight has >= 4 fractional bits
    {
        int d = 0;
        for (std::size_t k = 0; k < 4; ++k) d = std::max(d, std::abs(int{a[k]} - int{b[k]}));
        return 1 + (d + 15) / 16;
    }

    // ---------------------------------------------------------------------------------------------
    // One cell of the contact sheet.
    // ---------------------------------------------------------------------------------------------
    enum class Prog { Swizzle, Lod, Offset };
    using Expect = std::function<std::optional<Rgba8>(int, int)>;  // expected pixel (x, y) of the cell; nullopt = not probed

    struct Cell
    {
        std::string tag;                 // optional name, used by the negative controls
        std::string what;
        Prog prog = Prog::Lod;
        int vw = kCellPx, vh = kCellPx;  // viewport size inside the 16 x 16 cell
        std::function<void()> prepare;   // texture state and uniforms of this draw
        std::function<void()> after;     // optional cleanup (unbind a sampler)
        Expect expect;
        int tol = 1;
        int col = 0, row = 0;
    };

    struct Compare
    {
        int probes = 0;
        int bad = 0;
        int max_dev = 0;
        std::string first;
    };

    Compare compare_cell(const std::vector<std::uint8_t>& px, int vw, int vh, const Expect& expect, int tol)
    {
        Compare r;
        for (int y = 0; y < vh; ++y)
            for (int x = 0; x < vw; ++x)
            {
                const std::optional<Rgba8> e = expect(x, y);
                if (!e) continue;
                ++r.probes;
                int dev = 0;
                for (std::size_t k = 0; k < 4; ++k) dev = std::max(dev, std::abs(int{px[static_cast<std::size_t>((y * vw + x) * 4) + k]} - int{(*e)[k]}));
                r.max_dev = std::max(r.max_dev, dev);
                if (dev > tol && r.bad++ == 0)
                    r.first = std::format("first at ({},{}): got ({},{},{},{}) want {}", x, y, px[static_cast<std::size_t>((y * vw + x) * 4)],
                                          px[static_cast<std::size_t>((y * vw + x) * 4) + 1], px[static_cast<std::size_t>((y * vw + x) * 4) + 2],
                                          px[static_cast<std::size_t>((y * vw + x) * 4) + 3], demo::format_rgba(*e));
            }
        return r;
    }

    Expect solid(Rgba8 c)
    {
        return [c](int, int) { return std::optional<Rgba8>{c}; };
    }

    // Sampling state that is set on the ladder / chain texture before a draw (all of it texture state in ES 3.0).
    struct TexState
    {
        Min min = Min::NearestMipmapNearest;
        Wrap wrap = Wrap::Repeat;
        int base = 0;
        int max_level = 1000;
        float min_lod = -1000.0f;
        float max_lod = 1000.0f;
    };

    template <typename F>
    void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // start from an empty queue so the error belongs to this call
        call();
        check.expect_gl_error(code, what);
    }

    /// Inserts two #define lines after the #version line of `source`.
    std::string with_offset_limits(const std::string& source, int min_offset, int max_offset)
    {
        const std::size_t eol = source.find('\n');
        return source.substr(0, eol + 1) + std::format("#define OFF_MIN ({})\n#define OFF_MAX ({})\n", min_offset, max_offset) + source.substr(eol + 1);
    }

    class TextureSwizzleLod final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            min_offset_ = demo::get_integer(mgl::GetParameter::MinProgramTexelOffset);
            max_offset_ = demo::get_integer(mgl::GetParameter::MaxProgramTexelOffset);

            swz_.id = demo::build_program(env, "shaders/quad.vert", "shaders/swizzle.frag");
            swz_.tex = mgl::glGetUniformLocation(swz_.id, "u_tex");
            swz_.lookup = mgl::glGetUniformLocation(swz_.id, "u_lookup");

            lod_.id = demo::build_program(env, "shaders/quad.vert", "shaders/lod.frag");
            lod_.tex = mgl::glGetUniformLocation(lod_.id, "u_tex");
            lod_.func = mgl::glGetUniformLocation(lod_.id, "u_func");
            lod_.lod = mgl::glGetUniformLocation(lod_.id, "u_lod");
            lod_.dx = mgl::glGetUniformLocation(lod_.id, "u_dx");
            lod_.dy = mgl::glGetUniformLocation(lod_.id, "u_dy");
            lod_.q = mgl::glGetUniformLocation(lod_.id, "u_q");
            lod_.origin = mgl::glGetUniformLocation(lod_.id, "u_origin");

            // The offset program is compiled from the shader file with the queried limits pasted in (an offset is a constant expression).
            off_.id = demo::build_program_from_source(env.load_text("shaders/quad.vert"), with_offset_limits(env.load_text("shaders/offset.frag"), min_offset_, max_offset_),
                                                      "offset.frag with queried limits");
            off_.tex = mgl::glGetUniformLocation(off_.id, "u_tex");
            off_.func = mgl::glGetUniformLocation(off_.id, "u_func");
            off_.off = mgl::glGetUniformLocation(off_.id, "u_off");
            off_.lod = mgl::glGetUniformLocation(off_.id, "u_lod");
            off_.dx = mgl::glGetUniformLocation(off_.id, "u_dx");
            off_.dy = mgl::glGetUniformLocation(off_.id, "u_dy");
            off_.q = mgl::glGetUniformLocation(off_.id, "u_q");
            off_.origin = mgl::glGetUniformLocation(off_.id, "u_origin");
            off_.size = mgl::glGetUniformLocation(off_.id, "u_size");

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glGenSamplers(1, &sampler_);
            upload_swizzle_textures();
            upload_ladder();
            upload_chain();

            // A sampler with LINEAR filters and CLAMP_TO_EDGE: it will override the texture's NEAREST state on one unit.
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::MinFilter, static_cast<GLint>(static_cast<GLenum>(Min::Linear)));
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::MagFilter, static_cast<GLint>(static_cast<GLenum>(mgl::TextureMagFilter::Linear)));
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::WrapS, static_cast<GLint>(static_cast<GLenum>(Wrap::ClampToEdge)));
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::WrapT, static_cast<GLint>(static_cast<GLenum>(Wrap::ClampToEdge)));

            build_cells();

            // The offscreen sheet: 16 cells wide, as many rows high as the cells use.
            for (const Cell& c : cells_) sheet_rows_ = std::max(sheet_rows_, c.row + 1);
            mgl::glGenTextures(1, &sheet_tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, sheet_tex_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kCols * kCellPx, sheet_rows_ * kCellPx);
            nearest_clamp();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glGenFramebuffers(1, &sheet_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, sheet_fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), mgl::TextureTarget::Texture2D, sheet_tex_, 0);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete) throw demo::Failure(std::format("the contact-sheet framebuffer is incomplete: {}", demo::framebuffer_status_name(status)));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            present_.id = demo::build_program(env, "shaders/quad.vert", "shaders/present.frag");
            present_.sheet = mgl::glGetUniformLocation(present_.id, "u_sheet");
            present_.window = mgl::glGetUniformLocation(present_.id, "u_window");
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            // Pass 1: the cells, into the offscreen sheet.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, sheet_fbo_);
            mgl::glDisable(mgl::Capability::Dither);  // dithering could change a stored 8-bit value by one step
            mgl::glViewport(0, 0, kCols * kCellPx, sheet_rows_ * kCellPx);
            mgl::glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (const Cell& c : cells_) draw_cell(c);

            // Pass 2: the sheet, stretched over the window.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glUseProgram(present_.id);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, sheet_tex_);
            mgl::glUniform1i(present_.sheet, 0);
            mgl::glUniform2i(present_.window, env.width, env.height);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});  // the sheet is the render target of the next frame
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_presentation(env, check);
            // Every cell check reads the sheet (read_cell reads the bound READ framebuffer).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, sheet_fbo_);
            check_cells(check);
            check_equivalences(check);
            check_negative_controls(check);
            check_offsets_are_meaningful(check);
            check_swizzle_state(check);
            check_limits_and_state(check);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation sequence");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindSampler(3, mgl::SamplerId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &sheet_fbo_);
            mgl::glDeleteTextures(1, &sheet_tex_);
            mgl::glDeleteProgram(present_.id);
            mgl::glDeleteSamplers(1, &sampler_);
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(off_.id);
            mgl::glDeleteProgram(lod_.id);
            mgl::glDeleteProgram(swz_.id);
        }

    private:
        enum : std::size_t { kTexR8, kTexRg8, kTexLa, kTexA8, kTexRgba, kTexLadder, kTexChain, kTexCount };

        struct SwizzleProgram { mgl::ProgramId id{}; mgl::UniformLocation tex{}, lookup{}; };
        struct LodProgram { mgl::ProgramId id{}; mgl::UniformLocation tex{}, func{}, lod{}, dx{}, dy{}, q{}, origin{}; };
        struct OffsetProgram { mgl::ProgramId id{}; mgl::UniformLocation tex{}, func{}, off{}, lod{}, dx{}, dy{}, q{}, origin{}, size{}; };
        struct PresentProgram { mgl::ProgramId id{}; mgl::UniformLocation sheet{}, window{}; };

        // ------------------------------------------------------------------------------------------
        // Texture uploads
        // ------------------------------------------------------------------------------------------
        static void nearest_clamp()
        {
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, Min::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, Wrap::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, Wrap::ClampToEdge);
        }

        void upload_swizzle_textures()
        {
            std::vector<std::uint8_t> r8, rg8, la, a8, rgba;
            for (int j = 0; j < kSwz; ++j)
                for (int i = 0; i < kSwz; ++i)
                {
                    const Rgba8 s = src_texel(i, j);
                    r8.push_back(s[0]);
                    rg8.insert(rg8.end(), {s[0], s[1]});
                    la.insert(la.end(), {s[0], s[1]});  // LUMINANCE_ALPHA group: luminance, alpha (same bytes as RG8)
                    a8.push_back(s[3]);
                    rgba.insert(rgba.end(), s.begin(), s.end());
                }
            using IF = mgl::InternalFormat;
            using PF = mgl::PixelFormat;
            const mgl::TextureTarget t2d = mgl::TextureTarget::Texture2D;
            mgl::glBindTexture(t2d, tex_[kTexR8]);
            mgl::glTexImage2D(t2d, 0, IF::R8, kSwz, kSwz, 0, PF::Red, mgl::PixelType::UnsignedByte, r8.data());
            nearest_clamp();
            mgl::glBindTexture(t2d, tex_[kTexRg8]);
            mgl::glTexImage2D(t2d, 0, IF::Rg8, kSwz, kSwz, 0, PF::Rg, mgl::PixelType::UnsignedByte, rg8.data());
            nearest_clamp();
            mgl::glBindTexture(t2d, tex_[kTexLa]);
            mgl::glTexImage2D(t2d, 0, IF::LuminanceAlpha, kSwz, kSwz, 0, PF::LuminanceAlpha, mgl::PixelType::UnsignedByte, la.data());
            nearest_clamp();
            mgl::glBindTexture(t2d, tex_[kTexA8]);
            mgl::glTexImage2D(t2d, 0, IF::Alpha, kSwz, kSwz, 0, PF::Alpha, mgl::PixelType::UnsignedByte, a8.data());
            nearest_clamp();
            mgl::glBindTexture(t2d, tex_[kTexRgba]);
            mgl::glTexImage2D(t2d, 0, IF::Rgba8, kSwz, kSwz, 0, PF::Rgba, mgl::PixelType::UnsignedByte, rgba.data());
            nearest_clamp();
        }

        void upload_ladder()
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexLadder]);
            for (int level = 0; level < kLadderLevels; ++level)
            {
                const std::vector<Rgba8> flat(static_cast<std::size_t>(ladder_w(level) * ladder_h(level)), kLevelColor[static_cast<std::size_t>(level)]);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgba8, ladder_w(level), ladder_h(level), 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, flat.data());
            }
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
        }

        void upload_chain()
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexChain]);
            for (int level = 0; level < kChainLevels; ++level)
            {
                const int size = chain_size(level);
                std::vector<Rgba8> texels;
                for (int j = 0; j < size; ++j)
                    for (int i = 0; i < size; ++i) texels.push_back(chain_texel(level, i, j));
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgba8, size, size, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                  texels.data());
            }
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
        }

        // ------------------------------------------------------------------------------------------
        // State helpers used by the cells' prepare() functions
        // ------------------------------------------------------------------------------------------
        static void bind_2d(mgl::TextureId t) { mgl::glBindTexture(mgl::TextureTarget::Texture2D, t); }

        static void set_swizzle(const Swz& z)
        {
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleR, z[0]);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleG, z[1]);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleB, z[2]);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleA, z[3]);
        }

        /// Binds `tex` to the active unit and sets every sampling parameter the cells vary (an absolute state: no leftovers).
        static void apply_state(mgl::TextureId tex, const TexState& s)
        {
            bind_2d(tex);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, s.min);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, s.wrap);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, s.wrap);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BaseLevel, s.base);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, s.max_level);
            mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinLod, s.min_lod);
            mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLod, s.max_lod);
        }

        void lod_uniforms(int func, float lod, float dx0 = 0, float dx1 = 0, float dy0 = 0, float dy1 = 0, float q = 1) const
        {
            mgl::glUniform1i(lod_.func, func);
            mgl::glUniform1f(lod_.lod, lod);
            mgl::glUniform2f(lod_.dx, dx0, dx1);
            mgl::glUniform2f(lod_.dy, dy0, dy1);
            mgl::glUniform1f(lod_.q, q);
        }

        void draw_cell(const Cell& c) const
        {
            switch (c.prog)
            {
            case Prog::Swizzle:
                mgl::glUseProgram(swz_.id);
                mgl::glUniform1i(swz_.tex, 0);  // sampler uniforms hold a texture UNIT index
                break;
            case Prog::Lod:
                mgl::glUseProgram(lod_.id);
                mgl::glUniform1i(lod_.tex, 0);
                mgl::glUniform2i(lod_.origin, c.col * kCellPx, c.row * kCellPx);
                break;
            case Prog::Offset:
                mgl::glUseProgram(off_.id);
                mgl::glUniform1i(off_.tex, 0);
                mgl::glUniform2i(off_.origin, c.col * kCellPx, c.row * kCellPx);
                break;
            }
            mgl::glViewport(c.col * kCellPx, c.row * kCellPx, c.vw, c.vh);
            c.prepare();
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            if (c.after) c.after();
        }

        // ------------------------------------------------------------------------------------------
        // Building the cells (each holds the GL state of its draw AND the CPU model of its result)
        // ------------------------------------------------------------------------------------------
        void add(Cell c)
        {
            c.col = col_;
            c.row = row_;
            if (++col_ == kCols) next_row();
            cells_.push_back(std::move(c));
        }

        void next_row()
        {
            if (col_ != 0) ++row_;
            col_ = 0;
        }

        [[nodiscard]] int index_of(std::string_view tag) const
        {
            for (std::size_t i = 0; i < cells_.size(); ++i)
                if (cells_[i].tag == tag) return static_cast<int>(i);
            return -1;
        }

        void build_cells()
        {
            add_swizzle_cells();
            next_row();
            add_lod_cells();
            next_row();
            add_grad_and_proj_cells();
            next_row();
            add_size_cells();
            next_row();
            add_fetch_cells();
            next_row();
            add_offset_cells();
        }

        // ---- swizzle ---------------------------------------------------------------------------------
        struct SwzCase
        {
            std::string tag;
            const char* what;
            Fmt fmt;
            Swz swz;
            int lookup;
        };

        [[nodiscard]] mgl::TextureId texture_of(Fmt fmt) const
        {
            switch (fmt)
            {
            case Fmt::R8: return tex_[kTexR8];
            case Fmt::Rg8: return tex_[kTexRg8];
            case Fmt::LuminanceAlpha: return tex_[kTexLa];
            case Fmt::Alpha: return tex_[kTexA8];
            case Fmt::Rgba8: break;
            }
            return tex_[kTexRgba];
        }

        void add_swizzle_cells()
        {
            const Swz bgra = {S::Blue, S::Green, S::Red, S::Alpha};
            const std::vector<SwzCase> cases = {
                {"r8-id", "R8 with the initial swizzle: Cb = (R, 0, 0), Ab = 1", Fmt::R8, kIdentity, 0},
                {"r8-gray", "R8 swizzled (R, R, R, ONE): a gray ramp, the classic use of swizzles for single-channel data", Fmt::R8, {S::Red, S::Red, S::Red, S::One}, 0},
                {"r8-alpha", "R8 swizzled (ALPHA, GREEN, BLUE, RED): the swizzle sees Ab = 1 and G = B = 0, red moves to alpha", Fmt::R8, {S::Alpha, S::Green, S::Blue, S::Red}, 0},
                {"rg8-id", "RG8 with the initial swizzle: Cb = (R, G, 0), Ab = 1", Fmt::Rg8, kIdentity, 0},
                {"rg8-gr", "RG8 swizzled (G, R, ZERO, ONE): red and green exchanged", Fmt::Rg8, {S::Green, S::Red, S::Zero, S::One}, 0},
                {"rg8-la", "RG8 swizzled (R, R, R, G): the ES 3.0 replacement for LUMINANCE_ALPHA", Fmt::Rg8, {S::Red, S::Red, S::Red, S::Green}, 0},
                {"la-id", "legacy LUMINANCE_ALPHA texture, initial swizzle: Cb = (L, L, L), Ab = A", Fmt::LuminanceAlpha, kIdentity, 0},
                {"a8-id", "legacy ALPHA texture, initial swizzle: Cb = (0, 0, 0), Ab = A", Fmt::Alpha, kIdentity, 0},
                {"a8-gray", "ALPHA texture swizzled (ALPHA, ALPHA, ALPHA, ONE): alpha shown as gray", Fmt::Alpha, {S::Alpha, S::Alpha, S::Alpha, S::One}, 0},
                {"rgba-id", "RGBA8 with the initial swizzle (the reference image)", Fmt::Rgba8, kIdentity, 0},
                {"rgba-bgra", "RGBA8 swizzled (B, G, R, A): red and blue exchanged", Fmt::Rgba8, bgra, 0},
                {"rgba-argb", "RGBA8 swizzled (A, R, G, B): all four components rotated", Fmt::Rgba8, {S::Alpha, S::Red, S::Green, S::Blue}, 0},
                {"rgba-const", "RGBA8 swizzled (R, ZERO, ONE, G): two components replaced by constants", Fmt::Rgba8, {S::Red, S::Zero, S::One, S::Green}, 0},
                {"rgba-fetch", "RGBA8 swizzled (B, G, R, A), looked up with texelFetch: the swizzle applies to fetches too (2.12.9.2)", Fmt::Rgba8, bgra, 1},
                {"rgba-lod", "RGBA8 swizzled (B, G, R, A), looked up with textureLod", Fmt::Rgba8, bgra, 2},
                {"rgba-proj", "RGBA8 swizzled (B, G, R, A), looked up with textureProj (q = 2)", Fmt::Rgba8, bgra, 3},
            };
            for (const SwzCase& sc : cases)
            {
                Cell c;
                c.tag = sc.tag;
                c.what = sc.what;
                c.prog = Prog::Swizzle;
                c.prepare = [this, sc] {
                    bind_2d(texture_of(sc.fmt));
                    set_swizzle(sc.swz);
                    mgl::glUniform1i(swz_.lookup, sc.lookup);
                };
                c.expect = [sc](int x, int y) { return std::optional<Rgba8>{swizzle(sc.swz, base_color(sc.fmt, src_texel(x / 4, y / 4)))}; };
                add(std::move(c));
            }

            // The same swizzled RGBA8 texture on unit 3 together with a LINEAR / CLAMP_TO_EDGE sampler object. The sampler
            // replaces the filtering (the picture is smooth), but the swizzle stays: it is texture state, not sampler state.
            Cell c;
            c.tag = "rgba-sampler";
            c.what = "RGBA8 swizzled (B, G, R, A) on unit 3 with a LINEAR sampler: smooth, but still swizzled (swizzle is texture state)";
            c.prog = Prog::Swizzle;
            c.tol = 3;  // bilinear weights are only specified to the implementation's precision
            c.prepare = [this, bgra] {
                mgl::glActiveTexture(mgl::TextureUnit::Texture3);
                bind_2d(tex_[kTexRgba]);
                set_swizzle(bgra);
                mgl::glBindSampler(3, sampler_);
                mgl::glUniform1i(swz_.tex, 3);
                mgl::glUniform1i(swz_.lookup, 0);
                mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            };
            c.after = [] {
                mgl::glBindSampler(3, mgl::SamplerId{0});
                mgl::glActiveTexture(mgl::TextureUnit::Texture3);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
                mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            };
            c.expect = [bgra](int x, int y) {
                // ES 3.0 3.8.10.2 equation for LINEAR with CLAMP_TO_EDGE (Table 3.22); swizzling commutes with the filter.
                const double u = (x + 0.5) / kCellPx * kSwz - 0.5, v = (y + 0.5) / kCellPx * kSwz - 0.5;
                const int i0 = static_cast<int>(std::floor(u)), j0 = static_cast<int>(std::floor(v));
                const double a = u - i0, b = v - j0;
                auto texel = [&](int i, int j) { return swizzle(bgra, src_texel(std::clamp(i, 0, kSwz - 1), std::clamp(j, 0, kSwz - 1))); };
                const Rgba8 top = mix8(texel(i0, j0), texel(i0 + 1, j0), a);
                const Rgba8 bottom = mix8(texel(i0, j0 + 1), texel(i0 + 1, j0 + 1), a);
                Rgba8 r{};
                for (std::size_t k = 0; k < 4; ++k) r[k] = u8(static_cast<int>(std::lround((1.0 - b) * top[k] + b * bottom[k])));
                return std::optional<Rgba8>{r};
            };
            add(std::move(c));
        }

        // ---- explicit LOD ----------------------------------------------------------------------------
        void add_ladder_cell(std::string tag, std::string what, const TexState& state, int func, float lod, Rgba8 expected, int tol = 1, float q = 1.0f,
                             std::array<float, 4> grad = {0, 0, 0, 0}, int vw = kCellPx, int vh = kCellPx)
        {
            Cell c;
            c.tag = std::move(tag);
            c.what = std::move(what);
            c.prog = Prog::Lod;
            c.vw = vw;
            c.vh = vh;
            c.tol = tol;
            c.prepare = [this, state, func, lod, q, grad] {
                apply_state(tex_[kTexLadder], state);
                lod_uniforms(func, lod, grad[0], grad[1], grad[2], grad[3], q);
            };
            c.expect = solid(expected);
            add(std::move(c));
        }

        void add_lod_cells()
        {
            const auto color = [](int level) { return kLevelColor[static_cast<std::size_t>(level)]; };
            // The 16 x 16 pixel quad maps 64 x 16 texels onto 16 x 16 pixels: du/dx = 4 texels per pixel, dv/dy = 1. So
            // mu = 4, mv = 1 and any allowed approximation of rho (3.8.10.1) lies in [4, 5]: lambda in [2, 2.33], level 2.
            add_ladder_cell("lod-implicit", "texture() for reference: this geometry (4 texels per pixel in s) selects level 2 by itself", {}, 0, 0, color(2));
            add_ladder_cell("lod-0", "textureLod(0.0): level 0 although the geometry would say level 2 (explicit LOD replaces the derivatives)", {}, 1, 0.0f, color(0));
            add_ladder_cell("lod-3", "textureLod(3.0): level 3", {}, 1, 3.0f, color(3));
            add_ladder_cell("lod-6", "textureLod(6.0): level 6, the 1 x 1 level", {}, 1, 6.0f, color(6));
            add_ladder_cell("lod-9", "textureLod(9.0): beyond the last level, clamped to level q = 6 (equation 3.23)", {}, 1, 9.0f, color(6));
            {
                TexState s;
                s.min = Min::LinearMipmapLinear;
                const Rgba8 blend = mix8(color(1), color(2), 0.5);
                add_ladder_cell("lod-1.5", "textureLod(1.5) with LINEAR_MIPMAP_LINEAR: half level 1, half level 2 (equations 3.24-3.25)", s, 1, 1.5f, blend,
                                blend_tolerance(color(1), color(2)));
            }
            add_ladder_cell("lod-2.75-nn", "textureLod(2.75) with NEAREST_MIPMAP_NEAREST: the level is rounded, level 3", {}, 1, 2.75f, color(3));
            {
                TexState s;
                s.min = Min::NearestMipmapLinear;
                const Rgba8 blend = mix8(color(2), color(3), 0.75);
                add_ladder_cell("lod-2.75-nl", "textureLod(2.75) with NEAREST_MIPMAP_LINEAR: 1/4 level 2, 3/4 level 3", s, 1, 2.75f, blend, blend_tolerance(color(2), color(3)));
            }
            {
                TexState s;
                s.max_lod = 1.0f;
                add_ladder_cell("lod-maxlod", "textureLod(4.0) with TEXTURE_MAX_LOD = 1: explicit LOD is clamped by MAX_LOD too (equation 3.18), level 1", s, 1, 4.0f, color(1));
            }
            {
                TexState s;
                s.min_lod = 3.0f;
                add_ladder_cell("lod-minlod", "textureLod(0.0) with TEXTURE_MIN_LOD = 3: clamped up, level 3", s, 1, 0.0f, color(3));
            }
            {
                TexState s;
                s.base = 2;
                add_ladder_cell("lod-base-1", "textureLod(1.0) with TEXTURE_BASE_LEVEL = 2: lod counts from the base level, level 3", s, 1, 1.0f, color(3));
            }
            {
                TexState s;
                s.base = 2;
                add_ladder_cell("lod-base-0", "textureLod(0.0) with TEXTURE_BASE_LEVEL = 2: level 2", s, 1, 0.0f, color(2));
            }
            {
                TexState s;
                s.max_level = 4;
                add_ladder_cell("lod-maxlevel", "textureLod(5.0) with TEXTURE_MAX_LEVEL = 4: the chain ends at q = 4, level 4", s, 1, 5.0f, color(4));
            }
        }

        void add_grad_and_proj_cells()
        {
            const auto color = [](int level) { return kLevelColor[static_cast<std::size_t>(level)]; };
            // textureGrad: with only ONE non-zero partial derivative every allowed approximation of rho equals the exact
            // value (3.8.10.1: max{mu, mv} <= f <= mu + mv with the other one 0), so the level is a property of the test.
            // rho = |d(u)/dx| with u = 64 s, or |d(v)/dy| with v = 16 t: s scales with the WIDTH, t with the HEIGHT of the base level.
            add_ladder_cell("grad-1", "textureGrad(dPdx = (2/64, 0), dPdy = 0): rho = 2 texels, level 1", {}, 2, 0, color(1), 1, 1.0f, {2.0f / 64, 0, 0, 0});
            add_ladder_cell("grad-3", "textureGrad(dPdx = (8/64, 0), dPdy = 0): rho = 8, level 3", {}, 2, 0, color(3), 1, 1.0f, {8.0f / 64, 0, 0, 0});
            add_ladder_cell("grad-v-2", "textureGrad(dPdx = 0, dPdy = (0, 4/16)): rho = 4 texels (t scales with the height 16), level 2", {}, 2, 0, color(2), 1, 1.0f,
                            {0, 0, 0, 4.0f / 16});
            add_ladder_cell("grad-v-4", "textureGrad(dPdx = 0, dPdy = (0, 1)): rho = 16, level 4", {}, 2, 0, color(4), 1, 1.0f, {0, 0, 0, 1.0f});
            add_ladder_cell("grad-max", "textureGrad(dPdx = (2/64, 0), dPdy = (16/64, 0)): rho is the larger of the x and y footprints, 16: level 4", {}, 2, 0, color(4), 1, 1.0f,
                            {2.0f / 64, 0, 16.0f / 64, 0});
            {
                TexState s;
                s.base = 2;  // base level 2 is 16 x 4: u = 16 s, v = 4 t
                add_ladder_cell("grad-base-u", "textureGrad(dPdx = (4/16, 0)) with TEXTURE_BASE_LEVEL = 2: rho = 4 texels of level 2, lambda 2 counted from the base: level 4", s, 2, 0, color(4), 1,
                                1.0f, {4.0f / 16, 0, 0, 0});
                add_ladder_cell("grad-base-v", "textureGrad(dPdy = (0, 2/4)) with TEXTURE_BASE_LEVEL = 2: rho = 2 texels of level 2 (height 4), lambda 1: level 3", s, 2, 0, color(3), 1, 1.0f,
                                {0, 0, 0, 2.0f / 4});
            }

            // textureProj / textureProjLod: the coordinates are divided by the last component q. P = (s q, t q, q) therefore
            // is the same lookup as (s, t) for every q; without the division these cells would show other levels.
            add_ladder_cell("projlod-3", "textureProjLod(vec3 (s q, t q, q), 2.0) with q = 4: level 2", {}, 3, 2.0f, color(2), 1, 4.0f);
            add_ladder_cell("projlod-4", "textureProjLod(vec4 (s q, t q, 99, q), 4.0) with q = 0.5: the third component is ignored, level 4", {}, 4, 4.0f, color(4), 1, 0.5f);
            // lambda of the projected lookup is that of (s, t): level 2 as for texture() (q = 4 undivided would give lambda 4).
            add_ladder_cell("proj-3", "textureProj(vec3 (s q, t q, q)) with q = 4: the projected coordinates have the derivatives of (s, t), level 2", {}, 5, 0, color(2), 1, 4.0f);
            add_ladder_cell("proj-4", "textureProj(vec4 (s q, t q, 99, q)) with q = 0.25: level 2 (undivided it would be level 0)", {}, 6, 0, color(2), 1, 0.25f);
        }

        void add_size_cells()
        {
            auto size_cell = [this](std::string tag, std::string what, int base, int lod) {
                Cell c;
                c.tag = std::move(tag);
                c.what = std::move(what);
                c.prog = Prog::Lod;
                TexState s;
                s.base = base;
                c.prepare = [this, s, lod] {
                    apply_state(tex_[kTexLadder], s);
                    lod_uniforms(7, static_cast<float>(lod));
                };
                const int level = base + lod;
                c.expect = solid({u8(ladder_w(level)), u8(ladder_h(level)), 0, 255});
                add(std::move(c));
            };
            for (int lod = 0; lod < kLadderLevels; ++lod)
            {
                const int w = ladder_w(lod), h = ladder_h(lod);
                size_cell(std::format("size-{}", lod), std::format("textureSize({}) of the 64 x 16 ladder = ({}, {}): halved and clamped at 1 per level", lod, w, h), 0, lod);
            }
            for (int lod = 0; lod < 3; ++lod)
            {
                const int level = 2 + lod;
                size_cell(std::format("size-base2-{}", lod),
                          std::format("textureSize({}) with TEXTURE_BASE_LEVEL = 2 = size of level {} = ({}, {})", lod, level, ladder_w(level), ladder_h(level)), 2, lod);
            }
        }

        // ---- texelFetch -------------------------------------------------------------------------------
        void add_fetch_cells()
        {
            const auto color = [](int level) { return kLevelColor[static_cast<std::size_t>(level)]; };
            // texelFetch on the ladder at texel (0, 0): a single pixel per cell is enough because every level is solid.
            auto ladder_fetch = [this, &color](std::string tag, std::string what, const TexState& s, int lod, int level) {
                add_ladder_cell(std::move(tag), std::move(what), s, 8, static_cast<float>(lod), color(level), 1, 1.0f, {0, 0, 0, 0}, 1, 1);
            };
            {
                TexState s;
                s.min_lod = 5.0f;
                s.max_lod = 6.0f;
                ladder_fetch("fetch-minlod", "texelFetch(lod 0) with TEXTURE_MIN_LOD = 5, MAX_LOD = 6: fetches are not LOD-clamped, level 0 (textureLod(0) would give level 5)", s, 0, 0);
            }
            {
                TexState s;
                s.max_lod = 0.0f;
                ladder_fetch("fetch-maxlod", "texelFetch(lod 4) with TEXTURE_MAX_LOD = 0: not clamped, level 4 (textureLod(4) would give level 0)", s, 4, 4);
            }
            {
                TexState s;
                s.base = 2;
                ladder_fetch("fetch-base-ladder", "texelFetch(lod 1) with TEXTURE_BASE_LEVEL = 2: level 3", s, 1, 3);
            }

            // texelFetch on the 13 x 13 chain: the whole level is drawn pixel for pixel (pixel (x, y) = texel (x, y)).
            auto chain_fetch = [this](std::string tag, std::string what, const TexState& s, int lod) {
                const int level = s.base + lod;
                const int size = chain_size(level);
                Cell c;
                c.tag = std::move(tag);
                c.what = std::move(what);
                c.prog = Prog::Lod;
                c.vw = c.vh = size;
                c.prepare = [this, s, lod] {
                    apply_state(tex_[kTexChain], s);
                    lod_uniforms(8, static_cast<float>(lod));
                };
                c.expect = [level](int x, int y) { return std::optional<Rgba8>{chain_texel(level, x, y)}; };
                add(std::move(c));
            };
            for (int lod = 0; lod < kChainLevels; ++lod)
                chain_fetch(std::format("fetch-chain-b0-{}", lod),
                            std::format("texelFetch(lod {0}) on the 13x13 chain, base level 0: level {0}, {1}x{1} texels, drawn pixel for pixel", lod, chain_size(lod)), TexState{}, lod);
            for (int lod = 0; lod < 3; ++lod)
            {
                TexState s;
                s.base = 1;
                chain_fetch(std::format("fetch-chain-b1-{}", lod), std::format("texelFetch(lod {}) with TEXTURE_BASE_LEVEL = 1: level {} (lod is added to the base level)", lod, 1 + lod), s, lod);
            }
            {
                TexState s;
                s.base = 1;
                s.max_level = 2;
                chain_fetch("fetch-chain-b1-m2", "texelFetch(lod 1) with BASE_LEVEL = 1, MAX_LEVEL = 2: the chain is levels 1..2; lod 1 is level 2", s, 1);
            }
        }

        // ---- offsets ------------------------------------------------------------------------------------
        struct OffsetFn
        {
            const char* name;
            int func;
            int level;
            Wrap wrap;
            Min min;
            float lod;
            std::array<float, 4> grad;
            float q;
            bool fetch;
        };

        void add_offset_cells()
        {
            const std::array<std::pair<int, int>, 5> offsets = {{{1, 2}, {min_offset_, min_offset_}, {max_offset_, max_offset_}, {min_offset_, max_offset_}, {max_offset_, min_offset_}}};
            const std::array<const char*, 5> offset_names = {"small", "MIN/MIN", "MAX/MAX", "MIN/MAX", "MAX/MIN"};
            const std::vector<OffsetFn> fns = {
                {"texelFetchOffset", 0, 0, Wrap::Repeat, Min::Nearest, 0.0f, {0, 0, 0, 0}, 1.0f, true},
                {"textureOffset (REPEAT)", 1, 0, Wrap::Repeat, Min::Nearest, 0.0f, {0, 0, 0, 0}, 1.0f, false},
                {"textureOffset (CLAMP_TO_EDGE)", 1, 0, Wrap::ClampToEdge, Min::Nearest, 0.0f, {0, 0, 0, 0}, 1.0f, false},
                {"textureProjOffset (q = 2)", 2, 0, Wrap::Repeat, Min::Nearest, 0.0f, {0, 0, 0, 0}, 2.0f, false},
                {"textureLodOffset (lod 0)", 3, 0, Wrap::Repeat, Min::Nearest, 0.0f, {0, 0, 0, 0}, 1.0f, false},
                {"textureLodOffset (lod 1, level 1 = 6x6)", 3, 1, Wrap::Repeat, Min::NearestMipmapNearest, 1.0f, {0, 0, 0, 0}, 1.0f, false},
                {"textureGradOffset (rho = 2: level 1 = 6x6)", 4, 1, Wrap::Repeat, Min::NearestMipmapNearest, 0.0f, {2.0f / kChain, 0, 0, 0}, 1.0f, false},
                {"texelFetchOffset (lod 1, level 1 = 6x6)", 0, 1, Wrap::Repeat, Min::NearestMipmapNearest, 1.0f, {0, 0, 0, 0}, 1.0f, true},
            };
            for (const OffsetFn& fn : fns)
            {
                if (col_ + 5 > kCols) next_row();  // keep the five offsets of one function together
                for (std::size_t o = 0; o < offsets.size(); ++o)
                {
                    const int ox = offsets[o].first, oy = offsets[o].second;
                    const int size = chain_size(fn.level);
                    Cell c;
                    c.tag = std::format("off-{}-{}", fn.name, o);
                    c.what = std::format("{} with the {} offset ({}, {}) on level {}: {}", fn.name, offset_names[o], ox, oy, fn.level,
                                         fn.fetch ? "the cell shows the horizontally MIRRORED level (P = mirrored - offset, so P + offset is the mirrored texel)"
                                                  : "the level shifted by the offset, wrapped after the offset is added");
                    c.prog = Prog::Offset;
                    c.vw = c.vh = size;
                    TexState state;
                    state.min = fn.min;
                    state.wrap = fn.wrap;
                    c.prepare = [this, state, fn, o, size] {
                        apply_state(tex_[kTexChain], state);
                        mgl::glUniform1i(off_.func, fn.func);
                        mgl::glUniform1i(off_.off, static_cast<GLint>(o));
                        mgl::glUniform1f(off_.lod, fn.lod);
                        mgl::glUniform2f(off_.dx, fn.grad[0], fn.grad[1]);
                        mgl::glUniform2f(off_.dy, fn.grad[2], fn.grad[3]);
                        mgl::glUniform1f(off_.q, fn.q);
                        mgl::glUniform1i(off_.size, size);
                    };
                    c.expect = [fn, ox, oy, size](int x, int y) {
                        if (fn.fetch) return std::optional<Rgba8>{chain_texel(fn.level, size - 1 - x, y)};
                        return std::optional<Rgba8>{chain_texel(fn.level, wrap_coord(x + ox, size, fn.wrap), wrap_coord(y + oy, size, fn.wrap))};
                    };
                    add(std::move(c));
                }
            }
        }

        // ------------------------------------------------------------------------------------------
        // Validation
        // ------------------------------------------------------------------------------------------
        [[nodiscard]] std::vector<std::uint8_t> read_cell(const Cell& c) const { return demo::read_rgba8(c.col * kCellPx, c.row * kCellPx, c.vw, c.vh); }

        /// The window shows the sheet stretched with the integer mapping of shaders/present.frag: every window pixel must be
        /// exactly the sheet texel that mapping names (RGB), and opaque. The sheet's content is what check_cells verifies.
        void check_presentation(const demo::Env& env, demo::Check& check) const
        {
            const int sw = kCols * kCellPx, sh = sheet_rows_ * kCellPx;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, sheet_fbo_);
            const std::vector<std::uint8_t> sheet = demo::read_rgba8(0, 0, sw, sh);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const std::vector<std::uint8_t> window = demo::read_rgba8(0, 0, env.width, env.height);
            int bad = 0;
            std::string first;
            std::vector<bool> shown(static_cast<std::size_t>(sw * sh), false);
            for (int y = 0; y < env.height; ++y)
                for (int x = 0; x < env.width; ++x)
                {
                    const int tx = x * sw / env.width, ty = y * sh / env.height;
                    shown[static_cast<std::size_t>(ty * sw + tx)] = true;
                    const std::size_t w = (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4;
                    const std::size_t t = static_cast<std::size_t>(ty * sw + tx) * 4;
                    const bool ok = window[w] == sheet[t] && window[w + 1] == sheet[t + 1] && window[w + 2] == sheet[t + 2] && window[w + 3] == 255;
                    if (!ok && bad++ == 0)
                        first = std::format("window pixel ({},{}) shows {}, expected sheet texel ({},{}) = ({}, {}, {}) opaque", x, y,
                                            demo::format_rgba({window[w], window[w + 1], window[w + 2], window[w + 3]}), tx, ty, sheet[t], sheet[t + 1], sheet[t + 2]);
                }
            const auto shown_count = std::count(shown.begin(), shown.end(), true);
            check.expect(bad == 0,
                         std::format("the window ({} x {}) shows the {} x {} contact sheet stretched over the whole surface: every window pixel equals sheet texel (x * {} / {}, y * {} / {}), opaque",
                                     env.width, env.height, sw, sh, sw, env.width, sh, env.height),
                         bad == 0 ? std::format("{} pixels compared exactly; {} of {} sheet texels are visible", env.width * env.height, shown_count, sw * sh)
                                  : std::format("{} of {} pixels differ, {}", bad, env.width * env.height, first));
        }

        void check_cells(demo::Check& check) const
        {
            for (std::size_t i = 0; i < cells_.size(); ++i)
            {
                const Cell& c = cells_[i];
                const Compare r = compare_cell(read_cell(c), c.vw, c.vh, c.expect, c.tol);
                check.expect(r.probes > 0 && r.bad == 0, std::format("cell {} (col {}, row {}): {} [{} px, max deviation {}, tolerance {}]", i, c.col, c.row, c.what, r.probes, r.max_dev, c.tol),
                             r.first);
            }
        }

        // RG8 swizzled (R, R, R, G) is the ES 3.0 way to get what LUMINANCE_ALPHA gave: the two cells must be the same picture.
        void check_equivalences(demo::Check& check) const
        {
            const int a = index_of("rg8-la"), b = index_of("la-id");
            const std::vector<std::uint8_t> pa = read_cell(cells_[static_cast<std::size_t>(a)]), pb = read_cell(cells_[static_cast<std::size_t>(b)]);
            int diff = 0;
            for (std::size_t i = 0; i < pa.size(); ++i) diff = std::max(diff, std::abs(int{pa[i]} - int{pb[i]}));
            check.expect(diff <= 1, "RG8 + swizzle (R, R, R, G) renders the same picture as a LUMINANCE_ALPHA texture of the same data", std::format("max difference {}", diff));
        }

        // A comparison that cannot fail proves nothing: re-compare a cell with the model of a DIFFERENT configuration.
        void check_negative_controls(demo::Check& check) const
        {
            // (1) the rotated swizzle (A, R, G, B) against the model of (B, G, R, A)
            {
                const Cell& cell = cells_[static_cast<std::size_t>(index_of("rgba-argb"))];
                const Swz bgra = {S::Blue, S::Green, S::Red, S::Alpha};
                const Compare r = compare_cell(read_cell(cell), cell.vw, cell.vh,
                                               [bgra](int x, int y) { return std::optional<Rgba8>{swizzle(bgra, src_texel(x / 4, y / 4))}; }, 1);
                check.expect(r.bad > r.probes * 3 / 4, "negative control: the (A, R, G, B) cell does not match the CPU model of (B, G, R, A)",
                             std::format("{} of {} pixels differ", r.bad, r.probes));
            }
            // (2) a shifted offset cell against the unshifted level
            {
                const Cell& cell = cells_[static_cast<std::size_t>(index_of("off-textureOffset (REPEAT)-1"))];
                const Compare r = compare_cell(read_cell(cell), cell.vw, cell.vh, [](int x, int y) { return std::optional<Rgba8>{chain_texel(0, x, y)}; }, 1);
                check.expect(r.bad > r.probes * 3 / 4, "negative control: textureOffset at (MIN, MIN) does not equal the unshifted level",
                             std::format("{} of {} pixels differ", r.bad, r.probes));
            }
            // (3) explicit LOD: textureLod(3.0) against the level colors that texture() and textureLod(0.0) produce
            {
                const Cell& cell = cells_[static_cast<std::size_t>(index_of("lod-3"))];
                const Compare vs0 = compare_cell(read_cell(cell), cell.vw, cell.vh, solid(kLevelColor[0]), 1);
                const Compare vs2 = compare_cell(read_cell(cell), cell.vw, cell.vh, solid(kLevelColor[2]), 1);
                check.expect(vs0.bad == vs0.probes && vs2.bad == vs2.probes, "negative control: the textureLod(3.0) cell is neither level 0 (explicit lod 0) nor level 2 (the implicit lambda)",
                             std::format("{} / {} pixels differ", vs0.bad, vs2.bad));
                const Compare implicit = compare_cell(read_cell(cells_[static_cast<std::size_t>(index_of("lod-implicit"))]), kCellPx, kCellPx, solid(kLevelColor[0]), 1);
                const Compare explicit0 = compare_cell(read_cell(cells_[static_cast<std::size_t>(index_of("lod-0"))]), kCellPx, kCellPx, solid(kLevelColor[2]), 1);
                check.expect(implicit.bad == implicit.probes && explicit0.bad == explicit0.probes,
                             "same quad: texture() is not level 0 and textureLod(0.0) is not level 2: the explicit lod ignores the geometry");
            }
            // (4) a swizzle changes the picture of the same texture: R8 default vs R8 (R, R, R, ONE)
            {
                const Cell& id = cells_[static_cast<std::size_t>(index_of("r8-id"))];
                const Cell& gray = cells_[static_cast<std::size_t>(index_of("r8-gray"))];
                const std::vector<std::uint8_t> a = read_cell(id), b = read_cell(gray);
                int differing = 0;
                for (std::size_t i = 0; i < a.size(); i += 4)
                    if (a[i + 1] != b[i + 1]) ++differing;
                check.expect(differing > 200, "the same R8 texture: green is 0 without the swizzle and = red with (R, R, R, ONE) in almost every pixel", std::format("{} of 256 pixels differ in green", differing));
            }
        }

        void check_offsets_are_meaningful(demo::Check& check) const
        {
            // An offset that is a multiple of the level size would be invisible with REPEAT. Prove the five offsets are not.
            const std::array<std::pair<int, int>, 5> offsets = {{{1, 2}, {min_offset_, min_offset_}, {max_offset_, max_offset_}, {min_offset_, max_offset_}, {max_offset_, min_offset_}}};
            bool ok = true;
            std::string detail;
            for (const auto& [ox, oy] : offsets)
                for (int level : {0, 1})
                {
                    const int size = chain_size(level);
                    if (wrap_coord(ox, size, Wrap::Repeat) == 0 && wrap_coord(oy, size, Wrap::Repeat) == 0)
                    {
                        ok = false;
                        detail += std::format("({},{}) is a multiple of {}; ", ox, oy, size);
                    }
                }
            check.expect(ok, std::format("none of the five offsets (limits {} and {}) is a multiple of the 13 / 6 texel level size: every shifted image differs from the unshifted one", min_offset_, max_offset_), detail);
        }

        void check_swizzle_state(demo::Check& check) const
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            bind_2d(t);
            auto get = [](mgl::TextureParameterQuery p) {
                GLint v = 0;
                mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, p, &v);
                return v;
            };
            using Q = mgl::TextureParameterQuery;
            check.expect(get(Q::SwizzleR) == value_of(S::Red) && get(Q::SwizzleG) == value_of(S::Green) && get(Q::SwizzleB) == value_of(S::Blue) &&
                             get(Q::SwizzleA) == value_of(S::Alpha),
                         "initial swizzle of a new texture is (RED, GREEN, BLUE, ALPHA) (3.8.14, Table 6.9)");

            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleR, value_of(S::Blue));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleG, value_of(S::Zero));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleB, value_of(S::One));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleA, value_of(S::Red));
            check.expect(get(Q::SwizzleR) == value_of(S::Blue) && get(Q::SwizzleG) == value_of(S::Zero) && get(Q::SwizzleB) == value_of(S::One) && get(Q::SwizzleA) == value_of(S::Red),
                         "glTexParameteri swizzle (BLUE, ZERO, ONE, RED) is read back with glGetTexParameteriv (every legal value of Table 3.20 incl. ZERO and ONE)");

            const std::array<GLint, 1> alpha = {value_of(S::Alpha)};
            mgl::glTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleG, alpha.data());
            check.expect_eq(get(Q::SwizzleG), value_of(S::Alpha), "glTexParameteriv (vector form) sets TEXTURE_SWIZZLE_G = ALPHA");

            expect_error(check, mgl::ErrorCode::InvalidEnum, "TEXTURE_SWIZZLE_B = LINEAR (not RED/GREEN/BLUE/ALPHA/ZERO/ONE) -> INVALID_ENUM (ES 3.2 8.10)",
                         [] { mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleB, GL_LINEAR); });
            check.expect_eq(get(Q::SwizzleB), value_of(S::One), "the rejected glTexParameteri left TEXTURE_SWIZZLE_B unchanged (ONE)");
            expect_error(check, mgl::ErrorCode::InvalidEnum, "TEXTURE_SWIZZLE_R = REPEAT (a wrap mode) -> INVALID_ENUM",
                         [] { mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::SwizzleR, GL_REPEAT); });
            check.expect_no_gl_error("reading the swizzle state raised no error");

            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &t);
        }

        void check_limits_and_state(demo::Check& check) const
        {
            check.expect(min_offset_ <= -8 && max_offset_ >= 7, "MIN_PROGRAM_TEXEL_OFFSET <= -8 and MAX_PROGRAM_TEXEL_OFFSET >= 7 (Table 6.31)",
                         std::format("MIN = {}, MAX = {}", min_offset_, max_offset_));

            // The LOD parameters round-trip through the texture exactly (set by the last cell that used them).
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            bind_2d(t);
            auto geti = [](mgl::TextureParameterQuery p) {
                GLint v = 0;
                mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, p, &v);
                return v;
            };
            auto getf = [](mgl::TextureParameterQuery p) {
                GLfloat v = 0;
                mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, p, &v);
                return v;
            };
            TexState s;
            s.base = 2;
            s.max_level = 5;
            s.min_lod = 1.5f;
            s.max_lod = 4.25f;
            apply_state(t, s);
            check.expect(geti(mgl::TextureParameterQuery::BaseLevel) == 2 && geti(mgl::TextureParameterQuery::MaxLevel) == 5 && getf(mgl::TextureParameterQuery::MinLod) == 1.5f &&
                             getf(mgl::TextureParameterQuery::MaxLod) == 4.25f,
                         "TEXTURE_BASE_LEVEL = 2, MAX_LEVEL = 5, MIN_LOD = 1.5, MAX_LOD = 4.25 read back exactly");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &t);
        }

        PresentProgram present_;
        mgl::TextureId sheet_tex_{};
        mgl::FramebufferId sheet_fbo_{};
        int sheet_rows_ = 0;  // rows of cells in use: the sheet is kCols * kCellPx x sheet_rows_ * kCellPx pixels
        SwizzleProgram swz_;
        LodProgram lod_;
        OffsetProgram off_;
        std::array<mgl::TextureId, kTexCount> tex_{};
        mgl::SamplerId sampler_{};
        std::vector<Cell> cells_;
        int col_ = 0;
        int row_ = 0;
        int min_offset_ = 0;
        int max_offset_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-texture-swizzle-lod",
        .title = "Texture swizzle and shader-chosen level of detail",
        .min_es = {3, 0},
        .summary = "TEXTURE_SWIZZLE_R/G/B/A applied to Table 3.24 base colors and to every lookup; textureLod, textureGrad, textureProj(Lod), textureSize, "
                   "texelFetch and texel offsets at MIN/MAX_PROGRAM_TEXEL_OFFSET against CPU models.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<TextureSwizzleLod>(argc, argv, info);
}
