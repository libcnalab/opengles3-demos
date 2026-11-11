// es30-framebuffer-invalidate
//
// glInvalidateFramebuffer / glInvalidateSubFramebuffer (OpenGL ES 3.0 section 4.5) and the framebuffer
// attachment queries of section 6.1.13 that the other framebuffer demos do not use: the texture layer and
// the cube-map face of an attachment, the component type of float / integer / normalized attachments and the
// rules for the combined DEPTH_STENCIL_ATTACHMENT point.
//
// Invalidation tells the GL that it need not keep the contents of an attachment: afterwards they are
// *undefined*. So the demo never asserts anything about invalidated pixels. It validates what the spec does
// define: legal calls raise no error, the listed errors are raised (and change nothing), every attachment or
// region that was NOT invalidated keeps its contents bit for bit, and rendering after an invalidation works.
//
// Image (256 x 256 window, 4 x 4 cells of 64 x 64, top row first), every cell a blit of one image:
//   cube faces +X -X +Y -Y | +Z -Z, layers 0..3 of a 2D array texture | slices 0..3 of a 3D texture |
//   color attachments 0 and 1 of a framebuffer whose attachments were ALL invalidated and rendered again.
//   The default framebuffer's depth and stencil are invalidated at the end of the frame (the classic use).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kSize = 64;      // edge of every target image
    constexpr int kWindow = 256;   // 4 x 4 cells
    constexpr int kCells = 16;

    constexpr mgl::ClearBufferBit kColorBit = mgl::ClearBufferBit::Color;
    constexpr mgl::ClearBufferBit kDepthBit = mgl::ClearBufferBit::Depth;
    constexpr mgl::ClearBufferBit kStencilBit = mgl::ClearBufferBit::Stencil;

    // meta-gl keeps the 32 color attachment points in their own enum (ColorAttachment); the helpers convert.
    constexpr std::array<mgl::ColorAttachment, 32> kColor = {
        mgl::ColorAttachment::Color0,  mgl::ColorAttachment::Color1,  mgl::ColorAttachment::Color2,  mgl::ColorAttachment::Color3,
        mgl::ColorAttachment::Color4,  mgl::ColorAttachment::Color5,  mgl::ColorAttachment::Color6,  mgl::ColorAttachment::Color7,
        mgl::ColorAttachment::Color8,  mgl::ColorAttachment::Color9,  mgl::ColorAttachment::Color10, mgl::ColorAttachment::Color11,
        mgl::ColorAttachment::Color12, mgl::ColorAttachment::Color13, mgl::ColorAttachment::Color14, mgl::ColorAttachment::Color15,
        mgl::ColorAttachment::Color16, mgl::ColorAttachment::Color17, mgl::ColorAttachment::Color18, mgl::ColorAttachment::Color19,
        mgl::ColorAttachment::Color20, mgl::ColorAttachment::Color21, mgl::ColorAttachment::Color22, mgl::ColorAttachment::Color23,
        mgl::ColorAttachment::Color24, mgl::ColorAttachment::Color25, mgl::ColorAttachment::Color26, mgl::ColorAttachment::Color27,
        mgl::ColorAttachment::Color28, mgl::ColorAttachment::Color29, mgl::ColorAttachment::Color30, mgl::ColorAttachment::Color31};
    constexpr mgl::FramebufferAttachment kAttC0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
    constexpr mgl::FramebufferAttachment kAttC1 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1);
    constexpr mgl::FramebufferAttachment kAttC2 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color2);
    constexpr mgl::FramebufferAttachment kAttDepth = mgl::FramebufferAttachment::Depth;
    constexpr mgl::FramebufferAttachment kAttStencil = mgl::FramebufferAttachment::Stencil;
    constexpr mgl::FramebufferAttachment kAttDepthStencil = mgl::FramebufferAttachment::DepthStencil;
    constexpr mgl::DefaultFramebufferAttachment kDefColor = mgl::DefaultFramebufferAttachment::Color;
    constexpr mgl::DefaultFramebufferAttachment kDefDepth = mgl::DefaultFramebufferAttachment::Depth;
    constexpr mgl::DefaultFramebufferAttachment kDefStencil = mgl::DefaultFramebufferAttachment::Stencil;
    constexpr mgl::DrawBuffer kDraw0 = mgl::to_draw_buffer(mgl::ColorAttachment::Color0);
    constexpr mgl::DrawBuffer kDraw1 = mgl::to_draw_buffer(mgl::ColorAttachment::Color1);
    constexpr mgl::DrawBuffer kDraw2 = mgl::to_draw_buffer(mgl::ColorAttachment::Color2);
    const mgl::FramebufferId kDefaultFbo{0};

    constexpr std::array<mgl::TextureTarget, 6> kFaces = {mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
                                                          mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
                                                          mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
    constexpr std::array<std::string_view, 6> kFaceNames = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

    constexpr int gl(GLenum e) { return static_cast<int>(e); }
    constexpr int gl(mgl::TextureTarget t) { return static_cast<int>(static_cast<GLenum>(t)); }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    // ---------------------------------------------------------------------------------------------
    // Images (GL order: row 0 is the bottom row) and comparisons.
    // ---------------------------------------------------------------------------------------------
    struct Rect
    {
        int x0, y0, x1, y1;  // pixels [x0, x1) x [y0, y1)
    };

    struct Image
    {
        int w = 0, h = 0;
        std::vector<Rgba8> px;
        Image() = default;
        Image(int width, int height, Rgba8 fill = {0, 0, 0, 0})
            : w(width), h(height), px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), fill)
        {
        }
        Rgba8& at(int x, int y) { return px[static_cast<std::size_t>(y * w + x)]; }
        const Rgba8& at(int x, int y) const { return px[static_cast<std::size_t>(y * w + x)]; }
    };

    Image read_image(int w, int h)
    {
        const std::vector<std::uint8_t> bytes = demo::read_rgba8(0, 0, w, h);
        Image im(w, h);
        for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = {bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]};
        return im;
    }

    Image crop(const Image& src, int x0, int y0, int w, int h)
    {
        Image im(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) im.at(x, y) = src.at(x0 + x, y0 + y);
        return im;
    }

    bool in_rect(const std::optional<Rect>& r, int x, int y) { return r && x >= r->x0 && x < r->x1 && y >= r->y0 && y < r->y1; }

    struct Diff
    {
        int bad = 0;
        int compared = 0;
        int max_error = 0;
        std::string first;
    };

    /// Compares two images; pixels inside `hole` (the region that was invalidated) are not compared.
    Diff compare(const Image& got, const Image& want, int tolerance, std::optional<Rect> hole = std::nullopt)
    {
        Diff d;
        for (int y = 0; y < got.h; ++y)
            for (int x = 0; x < got.w; ++x)
            {
                if (in_rect(hole, x, y)) continue;
                ++d.compared;
                bool bad = false;
                for (std::size_t c = 0; c < 4; ++c)
                {
                    const int err = std::abs(int{got.at(x, y)[c]} - int{want.at(x, y)[c]});
                    d.max_error = std::max(d.max_error, err);
                    if (err > tolerance) bad = true;
                }
                if (bad && d.bad++ == 0)
                    d.first = std::format("first at ({},{}): got {} expected {}", x, y, demo::format_rgba(got.at(x, y)), demo::format_rgba(want.at(x, y)));
            }
        return d;
    }

    bool expect_image(demo::Check& check, std::string_view what, const Image& got, const Image& want, int tolerance, std::optional<Rect> hole = std::nullopt)
    {
        const Diff d = compare(got, want, tolerance, hole);
        return check.expect(d.bad == 0 && d.compared > 0, what,
                            d.bad == 0 ? std::format("{} pixels compared, max error {} (tolerance {})", d.compared, d.max_error, tolerance)
                                       : std::format("{} of {} pixels differ, {} (tolerance {})", d.bad, d.compared, d.first, tolerance));
    }

    // ---------------------------------------------------------------------------------------------
    // CPU models of the two fragment shaders.
    // ---------------------------------------------------------------------------------------------
    using Rgb = std::array<float, 3>;
    constexpr std::array<Rgb, 14> kPalette = {{{1.0f, 0.2f, 0.2f}, {0.2f, 1.0f, 0.2f}, {0.3f, 0.4f, 1.0f}, {1.0f, 1.0f, 0.2f}, {1.0f, 0.2f, 1.0f}, {0.2f, 1.0f, 1.0f},
                                               {1.0f, 0.6f, 0.2f}, {0.6f, 0.3f, 1.0f}, {0.5f, 1.0f, 0.4f}, {1.0f, 0.4f, 0.6f}, {0.4f, 0.8f, 1.0f}, {0.8f, 0.8f, 0.3f},
                                               {0.9f, 0.5f, 0.9f}, {0.3f, 0.9f, 0.7f}}};

    Rgba8 pattern_pixel(const Rgb& base, int x, int y, int size)
    {
        const double shade = 0.35 + 0.65 * (x + 0.5) / size;
        const double stripe = ((y >> 3) & 1) == 1 ? 0.8 : 1.0;
        return {to8(static_cast<double>(base[0]) * shade * stripe), to8(static_cast<double>(base[1]) * shade * stripe),
                to8(static_cast<double>(base[2]) * shade * stripe), 255};
    }

    Image pattern_image(const Rgb& base, int size)
    {
        Image im(size, size);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) im.at(x, y) = pattern_pixel(base, x, y, size);
        return im;
    }

    Rgba8 mrt_pixel(int attachment, int x, int y)
    {
        const std::uint8_t r = static_cast<std::uint8_t>(4 * x), g = static_cast<std::uint8_t>(4 * y);
        const std::uint8_t b = (((x >> 3) + (y >> 3)) & 1) == 1 ? 200 : 60;
        const std::array<std::array<std::uint8_t, 3>, 3> perm = {{{r, g, b}, {b, r, g}, {g, b, r}}};
        return {perm[static_cast<std::size_t>(attachment)][0], perm[static_cast<std::size_t>(attachment)][1], perm[static_cast<std::size_t>(attachment)][2], 255};
    }

    Image mrt_image(int attachment)
    {
        Image im(kSize, kSize);
        for (int y = 0; y < kSize; ++y)
            for (int x = 0; x < kSize; ++x) im.at(x, y) = mrt_pixel(attachment, x, y);
        return im;
    }

    // ---------------------------------------------------------------------------------------------
    // The depth/stencil contents that fill() writes and probe() reads back.
    //   left half  (x < 32): window depth 0.25, stencil 1       right half: window depth 0.75, stencil 2
    // probe(): bit 0 = a draw at depth 0.5 passes DEPTH_TEST LESS (stored depth > 0.5),
    //          bit 1 / 2 / 3 = a draw passes STENCIL_TEST EQUAL 1 / 2 / 0.
    // ---------------------------------------------------------------------------------------------
    constexpr std::uint8_t kBitDepth = 1;
    constexpr std::uint8_t kBitsStencil = 0xE;
    std::uint8_t expected_ds_bits(int x, int /*y*/) { return x < kSize / 2 ? std::uint8_t{0b0010} : std::uint8_t{0b0101}; }

    template <typename F>
    bool expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // a stale error must not be mistaken for the one under test
        call();
        return check.expect_gl_error(code, what);
    }

    /// Strict check; on the Mesa Gallium drivers of the test matrix (radeonsi, zink on lavapipe, llvmpipe) a failure is
    /// recorded as a labelled XFAIL (docs/KNOWN_LIMITATIONS.md). Every other driver must conform.
    void expect_known_mesa_deviation(demo::Check& check, bool ok, std::string_view what, const std::string& detail, std::string_view reason)
    {
        const GLubyte* r = mgl::glGetString(mgl::StringName::Renderer);
        const std::string_view renderer = r != nullptr ? reinterpret_cast<const char*>(r) : "";
        std::string_view driver = "no-known-driver";
        for (const std::string_view candidate : {"radeonsi", "zink", "llvmpipe"})
            if (renderer.find(candidate) != std::string_view::npos)
            {
                driver = candidate;
                break;
            }
        check.expect_or_known_deviation(ok, what, detail, driver, reason);
    }

    void bind_read(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, f); }
    void bind_draw(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, f); }
    void bind_both(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, f); }

    void reset_state()
    {
        mgl::glDisable(mgl::Capability::ScissorTest);
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glDisable(mgl::Capability::StencilTest);
        mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        mgl::glDepthMask(GL_TRUE);
        mgl::glStencilMask(0xFFFFFFFFu);
        mgl::glDepthFunc(mgl::CompareFunc::Less);
        mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFu);
        mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
    }

    Image read_attachment(mgl::FramebufferId f, mgl::ColorAttachment a, int size = kSize)
    {
        bind_read(f);
        mgl::glReadBuffer(mgl::to_read_buffer(a));
        return read_image(size, size);
    }

    /// Everything observable about a framebuffer whose attachments are invalidated.
    struct State
    {
        Image c0, c1, c2;
        std::vector<std::uint8_t> ds;  // probe bits, kSize x kSize
    };

    enum Part : unsigned
    {
        kPartC0 = 1,
        kPartC1 = 2,
        kPartC2 = 4,
        kPartDepth = 8,
        kPartStencil = 16,
        kPartAll = 31
    };

    /// A program drawn with quad.vert: it needs u_rect and u_z.
    struct Quad
    {
        mgl::ProgramId program{};
        mgl::UniformLocation rect{}, z{};
    };

    class FramebufferInvalidate final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            version_ = env.context_version;
            if (env.width < kWindow || env.height < kWindow)
                throw demo::Skip(std::format("this demo draws {0}x{0} pixels and needs a default framebuffer at least that large", kWindow));

            pattern_ = make_quad(env, "shaders/pattern.frag");
            mrt_ = make_quad(env, "shaders/mrt.frag");
            flat_ = make_quad(env, "shaders/flat.frag");
            pattern_base_ = mgl::glGetUniformLocation(pattern_.program, "u_base");
            pattern_size_ = mgl::glGetUniformLocation(pattern_.program, "u_size");
            flat_color_ = mgl::glGetUniformLocation(flat_.program, "u_color");

            // Texture targets that are shown and invalidated: cube map, 2D array, 3D, and a plain 2D texture with a mip chain.
            cube_ = new_texture(mgl::TextureTarget::TextureCubeMap);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, kSize, kSize);
            array_ = new_texture(mgl::TextureTarget::Texture2DArray);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 3, mgl::SizedInternalFormat::Rgba8, kSize, kSize, 4);
            volume_ = new_texture(mgl::TextureTarget::Texture3D);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture3D, 3, mgl::SizedInternalFormat::Rgba8, kSize, kSize, 4);
            plain_ = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 4, mgl::SizedInternalFormat::Rgba8, kSize, kSize);

            work_ = make_fbo();  // the framebuffer the images are rendered through
            view_ = make_fbo();  // the framebuffer the images are read / blitted / queried through

            // The subject: three color attachments (two textures and a renderbuffer) and a DEPTH24_STENCIL8 renderbuffer.
            tex_s0_ = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kSize, kSize);
            tex_s1_ = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kSize, kSize);
            rb_s2_ = new_renderbuffer(mgl::InternalFormat::Rgba8);
            rb_ds_ = new_renderbuffer(mgl::InternalFormat::Depth24Stencil8);
            subj_ = make_fbo();
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, tex_s0_, 0);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC1, mgl::TextureTarget::Texture2D, tex_s1_, 0);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttC2, mgl::RenderbufferTarget::Renderbuffer, rb_s2_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::RenderbufferTarget::Renderbuffer, rb_ds_);
            require_complete("subject");
            // A second, smaller framebuffer (one color attachment + its own depth/stencil) for the READ / DRAW target tests.
            tex_p0_ = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kSize, kSize);
            rb_pds_ = new_renderbuffer(mgl::InternalFormat::Depth24Stencil8);
            peer_ = make_fbo();
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, tex_p0_, 0);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::RenderbufferTarget::Renderbuffer, rb_pds_);
            require_complete("peer");
            // Probe framebuffers: their own color texture, but the SAME depth/stencil renderbuffer as the framebuffer they probe.
            tex_probe_s_ = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kSize, kSize);
            probe_subj_ = make_fbo();
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, tex_probe_s_, 0);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::RenderbufferTarget::Renderbuffer, rb_ds_);
            require_complete("probe of the subject");
            tex_probe_p_ = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kSize, kSize);
            probe_peer_ = make_fbo();
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, tex_probe_p_, 0);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::RenderbufferTarget::Renderbuffer, rb_pds_);
            require_complete("probe of the peer");
            empty_ = make_fbo();  // nothing attached: incomplete

            bind_both(kDefaultFbo);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            render_targets();
            // The subject is filled, then ALL of its attachments are invalidated (their contents are now undefined) and
            // it is filled again: rendering after an invalidation behaves normally. Cells 14 and 15 show the result.
            fill(subj_, 3);
            const std::array<mgl::FramebufferAttachment, 4> all = {kAttC0, kAttC1, kAttC2, kAttDepthStencil};
            mgl::glInvalidateFramebuffer(mgl::FramebufferTarget::Framebuffer, std::span<const mgl::FramebufferAttachment>(all));
            fill(subj_, 3);
            compose_window();
            // The classic use: nothing reads the depth and stencil of the default framebuffer after the frame is
            // done, so tell the GL it need not keep them. The color buffer (what is displayed) is not touched.
            const std::array<mgl::DefaultFramebufferAttachment, 2> depth_stencil = {kDefDepth, kDefStencil};
            mgl::glInvalidateFramebuffer(mgl::FramebufferTarget::Framebuffer, std::span<const mgl::DefaultFramebufferAttachment>(depth_stencil));
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            check_window(check, "after the end-of-frame invalidation of the default framebuffer's depth and stencil");
            check_layer_attachments(check);
            check_component_types(check);
            check_depth_stencil_query_rule(check);
            check_texture_layer_errors(check);
            check_layer_isolation(check);
            check_invalidate_framebuffer_objects(check);
            check_invalidate_texture_images(check);
            check_invalidate_errors(check);
            check_invalidate_default_framebuffer(check);

            // The invalidation tests changed the images and the window: draw everything again (it is what the
            // capture shows) and compare once more.
            render_again();
            check_window(check, "after re-rendering every image (rendering after invalidation)");

            bind_both(kDefaultFbo);
            reset_state();
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            bind_both(kDefaultFbo);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(framebuffers_.size()), framebuffers_.data());
            mgl::glDeleteRenderbuffers(static_cast<GLsizei>(renderbuffers_.size()), renderbuffers_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
            mgl::glDeleteProgram(pattern_.program);
            mgl::glDeleteProgram(mrt_.program);
            mgl::glDeleteProgram(flat_.program);
        }

    private:
        // ==========================================================================================
        // Objects
        // ==========================================================================================
        static Quad make_quad(const demo::Env& env, std::string_view fragment_file)
        {
            Quad q;
            q.program = demo::build_program(env, "shaders/quad.vert", fragment_file);
            q.rect = mgl::glGetUniformLocation(q.program, "u_rect");
            q.z = mgl::glGetUniformLocation(q.program, "u_z");
            return q;
        }

        mgl::TextureId new_texture(mgl::TextureTarget target)
        {
            mgl::TextureId t;
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(target, t);
            textures_.push_back(t);
            return t;
        }

        mgl::RenderbufferId new_renderbuffer(mgl::InternalFormat format)
        {
            mgl::RenderbufferId r;
            mgl::glGenRenderbuffers(1, &r);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, r);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, format, kSize, kSize);
            renderbuffers_.push_back(r);
            return r;
        }

        mgl::FramebufferId make_fbo()
        {
            mgl::FramebufferId f;
            mgl::glGenFramebuffers(1, &f);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, f);  // the name becomes an object on its first bind
            framebuffers_.push_back(f);
            return f;
        }

        static void require_complete(std::string_view what)
        {
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer '{}' is not complete: {}", what, demo::framebuffer_status_name(status)));
        }

        static void attach_face(mgl::FramebufferTarget target, mgl::TextureTarget face, mgl::TextureId tex, int level)
        {
            mgl::glFramebufferTexture2D(target, kAttC0, face, tex, level);
        }
        static void attach_layer(mgl::FramebufferTarget target, mgl::TextureId tex, int level, int layer)
        {
            mgl::glFramebufferTextureLayer(target, kAttC0, tex, level, layer);
        }

        // ==========================================================================================
        // Drawing
        // ==========================================================================================
        /// Draws the pixel rectangle r of a size x size target at NDC depth z (the quad's program must be in use).
        static void draw_rect(const Quad& q, const Rect& r, int size, float z)
        {
            auto ndc = [size](int p) { return static_cast<float>(p) / static_cast<float>(size) * 2.0f - 1.0f; };
            mgl::glUniform4f(q.rect, ndc(r.x0), ndc(r.y0), ndc(r.x1), ndc(r.y1));
            mgl::glUniform1f(q.z, z);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// Covers the whole viewport at NDC depth z.
        static void draw_full(const Quad& q, float z)
        {
            mgl::glUniform4f(q.rect, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glUniform1f(q.z, z);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// Draws the pattern of `base` over a size x size image attached to the bound framebuffer.
        void draw_pattern(const Rgb& base, int size) const
        {
            mgl::glViewport(0, 0, size, size);
            mgl::glUseProgram(pattern_.program);
            mgl::glUniform3f(pattern_base_, base[0], base[1], base[2]);
            mgl::glUniform1f(pattern_size_, static_cast<float>(size));
            draw_full(pattern_, 0.0f);
        }

        static float ndc_depth(double window_depth) { return static_cast<float>(window_depth * 2.0 - 1.0); }

        /// Renders the images of cells 0..13 (cube faces, array layers, 3D slices) through the work framebuffer.
        void render_targets() const
        {
            reset_state();
            bind_both(work_);
            std::size_t cell = 0;
            for (std::size_t f = 0; f < kFaces.size(); ++f)
            {
                attach_face(mgl::FramebufferTarget::Framebuffer, kFaces[f], cube_, 0);
                draw_pattern(kPalette[cell++], kSize);
            }
            for (int layer = 0; layer < 4; ++layer)
            {
                attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, layer);
                draw_pattern(kPalette[cell++], kSize);
            }
            for (int layer = 0; layer < 4; ++layer)
            {
                attach_layer(mgl::FramebufferTarget::Framebuffer, volume_, 0, layer);
                draw_pattern(kPalette[cell++], kSize);
            }
        }

        /// Fills a subject framebuffer: `colors` color attachments get the mrt.frag patterns, depth and stencil get
        /// the two-halves scene (see expected_ds_bits).
        void fill(mgl::FramebufferId f, std::size_t colors) const
        {
            reset_state();
            bind_both(f);
            std::vector<mgl::DrawBuffer> buffers = {kDraw0, kDraw1, kDraw2};
            buffers.resize(colors);
            mgl::glDrawBuffers(buffers);
            mgl::glViewport(0, 0, kSize, kSize);
            mgl::glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            // Depth and stencil are only written while their tests are enabled: ALWAYS / REPLACE.
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Replace);
            mgl::glUseProgram(mrt_.program);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 1, 0xFFu);
            draw_rect(mrt_, {0, 0, kSize / 2, kSize}, kSize, ndc_depth(0.25));
            mgl::glStencilFunc(mgl::CompareFunc::Always, 2, 0xFFu);
            draw_rect(mrt_, {kSize / 2, 0, kSize, kSize}, kSize, ndc_depth(0.75));
            reset_state();
        }

        /// Reads the depth and stencil of the framebuffer `probe` (a framebuffer that shares them) through four draws:
        /// bit 0 = DEPTH_TEST LESS at 0.5 passes, bits 1..3 = STENCIL_TEST EQUAL 1 / 2 / 0 passes. Depth and stencil
        /// are not modified; the probe framebuffer's own color attachment is.
        std::vector<std::uint8_t> probe(mgl::FramebufferId probe_fbo) const
        {
            reset_state();
            bind_both(probe_fbo);
            mgl::glViewport(0, 0, kSize, kSize);
            mgl::glUseProgram(flat_.program);
            mgl::glUniform4f(flat_color_, 1.0f, 1.0f, 1.0f, 1.0f);
            mgl::glDepthMask(GL_FALSE);
            std::vector<std::uint8_t> bits(static_cast<std::size_t>(kSize * kSize), 0);
            for (int pass = 0; pass < 4; ++pass)
            {
                mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                mgl::glClear(kColorBit);
                float z = 0.0f;
                if (pass == 0)
                {
                    mgl::glEnable(mgl::Capability::DepthTest);
                    mgl::glDisable(mgl::Capability::StencilTest);
                    mgl::glDepthFunc(mgl::CompareFunc::Less);
                    z = ndc_depth(0.5);
                }
                else
                {
                    mgl::glDisable(mgl::Capability::DepthTest);
                    mgl::glEnable(mgl::Capability::StencilTest);
                    mgl::glStencilFunc(mgl::CompareFunc::Equal, pass == 3 ? 0 : pass, 0xFFu);
                }
                draw_full(flat_, z);
                const Image im = read_image(kSize, kSize);
                for (std::size_t i = 0; i < im.px.size(); ++i)
                    if (im.px[i][0] > 127) bits[i] = static_cast<std::uint8_t>(bits[i] | (1u << pass));
            }
            reset_state();
            return bits;
        }

        State capture_subject() const
        {
            State s;
            s.c0 = read_attachment(subj_, mgl::ColorAttachment::Color0);
            s.c1 = read_attachment(subj_, mgl::ColorAttachment::Color1);
            s.c2 = read_attachment(subj_, mgl::ColorAttachment::Color2);
            s.ds = probe(probe_subj_);
            return s;
        }

        State capture_peer() const
        {
            State s;
            s.c0 = read_attachment(peer_, mgl::ColorAttachment::Color0);
            s.ds = probe(probe_peer_);
            return s;
        }

        /// Composes the window: 14 images and two attachments of the subject, each a 1:1 blit.
        void compose_window() const
        {
            reset_state();
            bind_both(kDefaultFbo);
            mgl::glViewport(0, 0, kWindow, kWindow);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            bind_draw(kDefaultFbo);
            std::size_t cell = 0;
            auto blit_cell = [&cell] {
                const int cx = static_cast<int>(cell % 4) * kSize, cy = (3 - static_cast<int>(cell / 4)) * kSize;
                mgl::glBlitFramebuffer(0, 0, kSize, kSize, cx, cy, cx + kSize, cy + kSize, kColorBit, mgl::BlitFilter::Nearest);
                ++cell;
            };
            bind_read(view_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            for (std::size_t f = 0; f < kFaces.size(); ++f)
            {
                attach_face(mgl::FramebufferTarget::ReadFramebuffer, kFaces[f], cube_, 0);
                blit_cell();
            }
            for (int layer = 0; layer < 4; ++layer)
            {
                attach_layer(mgl::FramebufferTarget::ReadFramebuffer, array_, 0, layer);
                blit_cell();
            }
            for (int layer = 0; layer < 4; ++layer)
            {
                attach_layer(mgl::FramebufferTarget::ReadFramebuffer, volume_, 0, layer);
                blit_cell();
            }
            bind_read(subj_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            blit_cell();
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color1));
            blit_cell();
            bind_read(view_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, kAttC0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
        }

        void render_again()
        {
            render_targets();
            fill(subj_, 3);
            compose_window();
            const std::array<mgl::DefaultFramebufferAttachment, 2> depth_stencil = {kDefDepth, kDefStencil};
            mgl::glInvalidateFramebuffer(mgl::FramebufferTarget::Framebuffer, std::span<const mgl::DefaultFramebufferAttachment>(depth_stencil));
        }

        // ==========================================================================================
        // Checks: the window
        // ==========================================================================================
        void check_window(demo::Check& check, std::string_view when) const
        {
            bind_read(kDefaultFbo);
            const Image win = read_image(kWindow, kWindow);
            // Cells 11..13 are slices 1..3 of the 3D texture, blitted from a read framebuffer whose attachment is
            // glFramebufferTextureLayer(volume, 0, slice). They are judged separately so that a driver which blits the
            // wrong slice does not hide the result of every other cell.
            auto is_3d_blit = [](int cell) { return cell >= 11 && cell <= 13; };
            std::size_t bad_cells = 0, bad_3d = 0;
            std::string first, first_3d;
            for (int cell = 0; cell < kCells; ++cell)
            {
                const Image got = crop(win, (cell % 4) * kSize, (3 - cell / 4) * kSize, kSize, kSize);
                const Image want = cell < 14 ? pattern_image(kPalette[static_cast<std::size_t>(cell)], kSize) : mrt_image(cell - 14);
                // The patterns are computed in float and stored as 8 bits: equation 2.3 allows either of the two closest values (tolerance 1).
                const Diff d = compare(got, want, 1);
                if (d.bad == 0) continue;
                if (is_3d_blit(cell))
                {
                    if (bad_3d++ == 0) first_3d = std::format("cell {}: {}", cell, d.first);
                }
                else if (bad_cells++ == 0)
                    first = std::format("cell {}: {}", cell, d.first);
            }
            check.expect(bad_cells == 0, std::format("the 13 cells other than 3D slices 1..3 equal their CPU patterns {}", when),
                         bad_cells == 0 ? "tolerance 1 per channel" : std::format("{} bad cells, {}", bad_cells, first));

            // The slices themselves, read with glReadPixels through a framebuffer that has only that slice attached:
            // this separates "the slice holds the right image" from "glBlitFramebuffer read the right slice".
            std::size_t bad_slices = 0;
            std::string first_slice;
            bind_read(view_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            for (int slice = 0; slice < 4; ++slice)
            {
                attach_layer(mgl::FramebufferTarget::ReadFramebuffer, volume_, 0, slice);
                const Diff d = compare(read_image(kSize, kSize), pattern_image(kPalette[static_cast<std::size_t>(10 + slice)], kSize), 1);
                if (d.bad != 0 && bad_slices++ == 0) first_slice = std::format("slice {}: {}", slice, d.first);
            }
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, kAttC0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            check.expect(bad_slices == 0, std::format("glReadPixels of 3D slices 0..3 returns each slice's own pattern {}", when),
                         bad_slices == 0 ? "tolerance 1 per channel" : std::format("{} bad slices, {}", bad_slices, first_slice));

            // ES 3.0 section 4.3.3: BlitFramebuffer copies from the read buffer of the read framebuffer, and section
            // 4.4.2.4 makes that the 2D image `layer` of the 3D texture. Nothing about the source is undefined here (the
            // slices were not invalidated). The ANGLE build in the matrix blits slice 0 for every slice of a 3D texture
            // (glReadPixels of the same attachment is right, 2D array layers blit correctly).
            check.expect_or_known_deviation(bad_3d == 0, std::format("cells 11..13 (3D slices 1..3 copied with glBlitFramebuffer) equal their CPU patterns {}", when),
                                            bad_3d == 0 ? "tolerance 1 per channel" : std::format("{} bad cells, {}", bad_3d, first_3d), "ANGLE",
                                            "ANGLE (Vulkan) glBlitFramebuffer from a 3D texture slice attached with glFramebufferTextureLayer reads "
                                            "slice 0 for every slice (ES 3.0 sections 4.3.3, 4.4.2.4)");
        }

        // ==========================================================================================
        // Checks: attachment queries
        // ==========================================================================================
        static GLint query(mgl::FramebufferAttachment attachment, mgl::FramebufferAttachmentParameter pname)
        {
            GLint v = -12345;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, attachment, pname, &v);
            return v;
        }

        static bool expect_query(demo::Check& check, mgl::FramebufferAttachment attachment, mgl::FramebufferAttachmentParameter pname, int expected,
                                 std::string_view what)
        {
            (void)demo::drain_gl_errors();
            const GLint v = query(attachment, pname);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            return check.expect(e == mgl::ErrorCode::NoError && v == expected, what,
                                std::format("value {} expected {}{}", v, expected, e == mgl::ErrorCode::NoError ? "" : std::format(", glGetError() = {}", demo::error_name(e))));
        }

        static bool expect_query_error(demo::Check& check, mgl::FramebufferAttachment attachment, mgl::FramebufferAttachmentParameter pname, mgl::ErrorCode code,
                                       std::string_view what)
        {
            return expect_error(check, code, what, [&] { (void)query(attachment, pname); });
        }

        // TEXTURE_LAYER, TEXTURE_LEVEL and TEXTURE_CUBE_MAP_FACE of texture attachments (ES 3.0 6.1.13).
        void check_layer_attachments(demo::Check& check)
        {
            using P = mgl::FramebufferAttachmentParameter;
            bind_both(view_);
            for (int layer = 0; layer < 4; ++layer)
            {
                attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, layer);
                expect_query(check, kAttC0, P::TextureLayer, layer, std::format("2D array layer {}: TEXTURE_LAYER = {}", layer, layer));
            }
            attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 1, 2);
            expect_query(check, kAttC0, P::TextureLevel, 1, "2D array, level 1 layer 2: TEXTURE_LEVEL = 1");
            expect_query(check, kAttC0, P::TextureLayer, 2, "...and TEXTURE_LAYER = 2");
            expect_query(check, kAttC0, P::ObjectType, GL_TEXTURE, "...OBJECT_TYPE = TEXTURE");
            expect_query(check, kAttC0, P::ObjectName, static_cast<int>(array_.value), "...OBJECT_NAME = the array texture");
            expect_query(check, kAttC0, P::TextureCubeMapFace, 0, "...TEXTURE_CUBE_MAP_FACE is 0 for a texture that is not a cube map");
            for (int layer = 0; layer < 4; ++layer)
            {
                attach_layer(mgl::FramebufferTarget::Framebuffer, volume_, 0, layer);
                expect_query(check, kAttC0, P::TextureLayer, layer, std::format("3D texture slice {}: TEXTURE_LAYER = {}", layer, layer));
            }
            attach_layer(mgl::FramebufferTarget::Framebuffer, volume_, 1, 1);
            expect_query(check, kAttC0, P::TextureLevel, 1, "3D texture, level 1 slice 1: TEXTURE_LEVEL = 1");
            expect_query(check, kAttC0, P::TextureLayer, 1, "...and TEXTURE_LAYER = 1");
            expect_query(check, kAttC0, P::TextureCubeMapFace, 0, "...TEXTURE_CUBE_MAP_FACE = 0 for a 3D texture");

            for (std::size_t f = 0; f < kFaces.size(); ++f)
            {
                attach_face(mgl::FramebufferTarget::Framebuffer, kFaces[f], cube_, 0);
                expect_query(check, kAttC0, P::TextureCubeMapFace, gl(kFaces[f]), std::format("cube map face {}: TEXTURE_CUBE_MAP_FACE is that face's target", kFaceNames[f]));
                expect_query(check, kAttC0, P::TextureLayer, 0, std::format("cube map face {}: TEXTURE_LAYER = 0 (not a 3D or array texture)", kFaceNames[f]));
            }
            attach_face(mgl::FramebufferTarget::Framebuffer, mgl::TextureTarget::Texture2D, plain_, 2);
            expect_query(check, kAttC0, P::TextureLevel, 2, "2D texture level 2: TEXTURE_LEVEL = 2");
            expect_query(check, kAttC0, P::TextureLayer, 0, "...TEXTURE_LAYER = 0");
            expect_query(check, kAttC0, P::TextureCubeMapFace, 0, "...TEXTURE_CUBE_MAP_FACE = 0");

            // Detached: the attachment point is back to its defaults (Table 6.14); only the name can be queried.
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            expect_query(check, kAttC0, P::ObjectType, GL_NONE, "detached attachment: OBJECT_TYPE = NONE");
            expect_query(check, kAttC0, P::ObjectName, 0, "...OBJECT_NAME = 0");
            expect_query_error(check, kAttC0, P::TextureLayer, mgl::ErrorCode::InvalidOperation, "...TEXTURE_LAYER of an attachment with no image -> INVALID_OPERATION");
            expect_query_error(check, kAttC0, P::TextureLevel, mgl::ErrorCode::InvalidOperation, "...TEXTURE_LEVEL of an attachment with no image -> INVALID_OPERATION");

            // A renderbuffer attachment has no level, face or layer: INVALID_ENUM ("any combination ... not described above").
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::RenderbufferTarget::Renderbuffer, rb_s2_);
            expect_query(check, kAttC0, P::ObjectType, GL_RENDERBUFFER, "renderbuffer attachment: OBJECT_TYPE = RENDERBUFFER");
            expect_query(check, kAttC0, P::ObjectName, static_cast<int>(rb_s2_.value), "...OBJECT_NAME = the renderbuffer");
            expect_query_error(check, kAttC0, P::TextureLevel, mgl::ErrorCode::InvalidEnum, "...TEXTURE_LEVEL of a renderbuffer attachment -> INVALID_ENUM");
            expect_query_error(check, kAttC0, P::TextureCubeMapFace, mgl::ErrorCode::InvalidEnum, "...TEXTURE_CUBE_MAP_FACE of a renderbuffer attachment -> INVALID_ENUM");
            expect_query_error(check, kAttC0, P::TextureLayer, mgl::ErrorCode::InvalidEnum, "...TEXTURE_LAYER of a renderbuffer attachment -> INVALID_ENUM");
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});

            // The effect, not only the state: the attached layer / face is the one that was drawn into (see check_window for the
            // pixels of every cell); here, rendering into layer 1 of a mip level changes level 1 and nothing else.
            bind_both(work_);
            attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 1, 2);
            draw_pattern(kPalette[13], kSize / 2);
            const Image level1 = read_image(kSize / 2, kSize / 2);
            expect_image(check, "rendering through level 1 of the array texture (a 32 x 32 image) draws the pattern at that size", level1, pattern_image(kPalette[13], kSize / 2), 1);
            attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, 2);
            expect_image(check, "...and layer 2 of level 0 still holds its own pattern", read_image(kSize, kSize), pattern_image(kPalette[8], kSize), 1);
            bind_both(kDefaultFbo);
        }

        // ------------------------------------------------------------------------------------------
        struct ComponentCase
        {
            mgl::SizedInternalFormat format;
            std::string_view name;
            int type;
            bool renderable;  // color-renderable in every ES 3.x version (floating point and SNORM are not in ES 3.0)
        };

        // FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE (ES 3.0 6.1.13): FLOAT, INT, UNSIGNED_INT, SIGNED_NORMALIZED or UNSIGNED_NORMALIZED.
        void check_component_types(demo::Check& check)
        {
            constexpr int UN = GL_UNSIGNED_NORMALIZED, SN = GL_SIGNED_NORMALIZED, FL = GL_FLOAT, UI = GL_UNSIGNED_INT, SI = GL_INT;
            using F = mgl::SizedInternalFormat;
            const std::array<ComponentCase, 22> cases = {{
                {F::Rgba8, "RGBA8", UN, true},          {F::Srgb8Alpha8, "SRGB8_ALPHA8", UN, true}, {F::Rgb10A2, "RGB10_A2", UN, true},
                {F::R8, "R8", UN, true},                {F::Rg8, "RG8", UN, true},                  {F::Rgba8Snorm, "RGBA8_SNORM", SN, false},
                {F::R16F, "R16F", FL, false},           {F::Rgba16F, "RGBA16F", FL, false},         {F::Rgba32F, "RGBA32F", FL, false},
                {F::R11FG11FB10F, "R11F_G11F_B10F", FL, false},                                    {F::R8UI, "R8UI", UI, true},
                {F::Rgba8UI, "RGBA8UI", UI, true},      {F::Rgba16UI, "RGBA16UI", UI, true},        {F::Rgba32UI, "RGBA32UI", UI, true},
                {F::Rgb10A2UI, "RGB10_A2UI", UI, true}, {F::R8I, "R8I", SI, true},                 {F::Rg16I, "RG16I", SI, true},
                {F::Rgba8I, "RGBA8I", SI, true},        {F::Rgba32I, "RGBA32I", SI, true},          {F::R32UI, "R32UI", UI, true},
                {F::R32I, "R32I", SI, true},            {F::Rg32UI, "RG32UI", UI, true},
            }};
            bind_both(view_);
            for (const ComponentCase& c : cases)
            {
                const mgl::TextureId t = new_texture(mgl::TextureTarget::Texture2D);
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, c.format, 4, 4);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, t, 0);
                expect_query(check, kAttC0, mgl::FramebufferAttachmentParameter::ComponentType, c.type,
                             std::format("{} texture attachment: COMPONENT_TYPE = {}", c.name,
                                         c.type == UN ? "UNSIGNED_NORMALIZED" : c.type == SN ? "SIGNED_NORMALIZED" : c.type == FL ? "FLOAT" : c.type == UI ? "UNSIGNED_INT" : "INT"));
                if (c.renderable)
                    check.expect(mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete,
                                 std::format("{}: a framebuffer with only this attachment is complete (color-renderable)", c.name));
            }
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);

            // Renderbuffers: the same query on the integer and normalized formats, and the depth formats.
            struct RbCase
            {
                mgl::InternalFormat format;
                std::string_view name;
                mgl::FramebufferAttachment attachment;
                int type;
            };
            const std::array<RbCase, 9> rb_cases = {{{mgl::InternalFormat::Rgba8, "RGBA8", kAttC0, UN},
                                                     {mgl::InternalFormat::Rgba8UI, "RGBA8UI", kAttC0, UI},
                                                     {mgl::InternalFormat::Rgba32UI, "RGBA32UI", kAttC0, UI},
                                                     {mgl::InternalFormat::Rgb10A2UI, "RGB10_A2UI", kAttC0, UI},
                                                     {mgl::InternalFormat::Rgba8I, "RGBA8I", kAttC0, SI},
                                                     {mgl::InternalFormat::Rg32I, "RG32I", kAttC0, SI},
                                                     {mgl::InternalFormat::DepthComponent16, "DEPTH_COMPONENT16", kAttDepth, UN},
                                                     {mgl::InternalFormat::DepthComponent24, "DEPTH_COMPONENT24", kAttDepth, UN},
                                                     {mgl::InternalFormat::DepthComponent32F, "DEPTH_COMPONENT32F", kAttDepth, FL}}};
            for (const RbCase& c : rb_cases)
            {
                const mgl::RenderbufferId rb = new_renderbuffer(c.format);
                mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, c.attachment, mgl::RenderbufferTarget::Renderbuffer, rb);
                expect_query(check, c.attachment, mgl::FramebufferAttachmentParameter::ComponentType, c.type,
                             std::format("{} renderbuffer at {}: COMPONENT_TYPE = {}", c.name, c.attachment == kAttDepth ? "DEPTH_ATTACHMENT" : "COLOR_ATTACHMENT0",
                                         c.type == UN ? "UNSIGNED_NORMALIZED" : c.type == FL ? "FLOAT" : c.type == UI ? "UNSIGNED_INT" : "INT"));
                check.expect(mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete, std::format("{}: the framebuffer is complete", c.name));
                mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, c.attachment, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            }
            // A depth texture: DEPTH_COMPONENT32F is FLOAT, DEPTH_COMPONENT24 is UNSIGNED_NORMALIZED.
            for (const auto& [format, name, type] : {std::tuple{mgl::SizedInternalFormat::DepthComponent32F, std::string_view{"DEPTH_COMPONENT32F"}, FL},
                                                     std::tuple{mgl::SizedInternalFormat::DepthComponent24, std::string_view{"DEPTH_COMPONENT24"}, UN}})
            {
                const mgl::TextureId t = new_texture(mgl::TextureTarget::Texture2D);
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, 4, 4);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttDepth, mgl::TextureTarget::Texture2D, t, 0);
                expect_query(check, kAttDepth, mgl::FramebufferAttachmentParameter::ComponentType, type, std::format("{} texture at DEPTH_ATTACHMENT: COMPONENT_TYPE", name));
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttDepth, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            }
            // No data store at all: COMPONENT_TYPE is NONE.
            const mgl::TextureId empty_texture = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, empty_texture, 0);
            expect_query(check, kAttC0, mgl::FramebufferAttachmentParameter::ComponentType, GL_NONE,
                         "a texture with no image at the attached level: COMPONENT_TYPE = NONE");
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttC0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            bind_both(kDefaultFbo);
        }

        // ------------------------------------------------------------------------------------------
        // The rules for DEPTH_STENCIL_ATTACHMENT and for the attachment argument (ES 3.0 6.1.13).
        void check_depth_stencil_query_rule(demo::Check& check)
        {
            using P = mgl::FramebufferAttachmentParameter;
            // (a) one DEPTH24_STENCIL8 renderbuffer fills both points.
            bind_both(view_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::RenderbufferTarget::Renderbuffer, rb_ds_);
            expect_query(check, kAttDepthStencil, P::ObjectType, GL_RENDERBUFFER, "DEPTH_STENCIL_ATTACHMENT (one renderbuffer): OBJECT_TYPE = RENDERBUFFER");
            expect_query(check, kAttDepthStencil, P::ObjectName, static_cast<int>(rb_ds_.value), "...OBJECT_NAME = that renderbuffer");
            expect_query(check, kAttDepth, P::ObjectName, static_cast<int>(rb_ds_.value), "...DEPTH_ATTACHMENT holds the same renderbuffer");
            expect_query(check, kAttStencil, P::ObjectName, static_cast<int>(rb_ds_.value), "...STENCIL_ATTACHMENT holds the same renderbuffer");
            check.expect(query(kAttDepthStencil, P::DepthSize) >= 24 && query(kAttDepthStencil, P::StencilSize) >= 8,
                         "...DEPTH_SIZE >= 24 and STENCIL_SIZE >= 8 through the combined point (the object's sizes are returned)");
            expect_query(check, kAttDepthStencil, P::RedSize, 0, "...RED_SIZE = 0 (the component is not present)");
            expect_query_error(check, kAttDepthStencil, P::ComponentType, mgl::ErrorCode::InvalidOperation,
                               "...COMPONENT_TYPE of DEPTH_STENCIL_ATTACHMENT -> INVALID_OPERATION (no single format)");
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});

            // (b) a DEPTH24_STENCIL8 texture through DEPTH_STENCIL_ATTACHMENT.
            const mgl::TextureId ds_texture = new_texture(mgl::TextureTarget::Texture2D);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Depth24Stencil8, 4, 4);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::TextureTarget::Texture2D, ds_texture, 0);
            expect_query(check, kAttDepthStencil, P::ObjectType, GL_TEXTURE, "DEPTH_STENCIL_ATTACHMENT (a texture): OBJECT_TYPE = TEXTURE");
            expect_query(check, kAttDepth, P::ObjectName, static_cast<int>(ds_texture.value), "...DEPTH_ATTACHMENT names the same texture");
            expect_query(check, kAttStencil, P::ObjectName, static_cast<int>(ds_texture.value), "...STENCIL_ATTACHMENT names the same texture");
            expect_query(check, kAttDepthStencil, P::TextureLevel, 0, "...TEXTURE_LEVEL = 0");
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kAttDepthStencil, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);

            // (c) two different renderbuffers at the depth and the stencil point: the combined point cannot be queried.
            const mgl::RenderbufferId d16 = new_renderbuffer(mgl::InternalFormat::DepthComponent16);
            const mgl::RenderbufferId s8 = new_renderbuffer(mgl::InternalFormat::StencilIndex8);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepth, mgl::RenderbufferTarget::Renderbuffer, d16);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttStencil, mgl::RenderbufferTarget::Renderbuffer, s8);
            expect_query_error(check, kAttDepthStencil, P::ObjectType, mgl::ErrorCode::InvalidOperation,
                               "different renderbuffers at DEPTH and STENCIL: querying DEPTH_STENCIL_ATTACHMENT -> INVALID_OPERATION");
            expect_query_error(check, kAttDepthStencil, P::DepthSize, mgl::ErrorCode::InvalidOperation,
                               "...also for DEPTH_SIZE");
            check.expect(query(kAttDepth, P::DepthSize) >= 16 && query(kAttDepth, P::StencilSize) == 0, "...DEPTH_ATTACHMENT alone is fine: DEPTH_SIZE >= 16, STENCIL_SIZE = 0");
            check.expect(query(kAttStencil, P::StencilSize) >= 8 && query(kAttStencil, P::DepthSize) == 0, "...STENCIL_ATTACHMENT alone is fine: STENCIL_SIZE >= 8, DEPTH_SIZE = 0");
            check.expect_no_gl_error("...those single-point queries raise no error");
            // Depth only (nothing at the stencil point) is also "different objects".
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttStencil, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            expect_query_error(check, kAttDepthStencil, P::ObjectType, mgl::ErrorCode::InvalidOperation,
                               "a renderbuffer at DEPTH and nothing at STENCIL: querying DEPTH_STENCIL_ATTACHMENT -> INVALID_OPERATION");
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kAttDepth, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            // Nothing at either point. The text does not settle this case: "the same object must be bound to both the depth
            // and stencil attachment points" is not met when no object is bound (ANGLE: INVALID_OPERATION), while the
            // only listed error is for "different objects are bound to the depth and stencil attachment points"
            // (Mesa: no error, OBJECT_TYPE = NONE, like either point queried alone). Both outcomes are accepted; anything
            // else (another error, or no error with a value other than NONE) fails.
            {
                (void)demo::drain_gl_errors();
                const GLint v = query(kAttDepthStencil, P::ObjectType);
                const mgl::ErrorCode e = demo::drain_gl_errors();
                std::cout << std::format("  [query] nothing at DEPTH and STENCIL: DEPTH_STENCIL_ATTACHMENT OBJECT_TYPE -> {}, value {:#x} (not asserted which)\n",
                                         demo::error_name(e), v);
                check.expect((e == mgl::ErrorCode::NoError && v == GL_NONE) || e == mgl::ErrorCode::InvalidOperation,
                             "nothing at DEPTH and STENCIL: DEPTH_STENCIL_ATTACHMENT reports OBJECT_TYPE = NONE or raises INVALID_OPERATION",
                             std::format("glGetError() = {}, value {:#x}", demo::error_name(e), v));
            }

            // (d) which attachment arguments are legal for which framebuffer (the table of 6.1.13's errors).
            const int max_color = demo::get_integer(mgl::GetParameter::MaxColorAttachments);
            expect_query_error(check, mgl::FramebufferAttachment::None, P::ObjectType, mgl::ErrorCode::InvalidEnum, "framebuffer object: attachment GL_NONE is not in Table 4.6 -> INVALID_ENUM");
            // ES 3.0.6 section 6.1.13 / ES 3.2 section 9.2.3 Errors: "An
            // INVALID_ENUM error is generated if a framebuffer object is bound to target, attachment is not one of the
            // attachments in table 4.6, and attachment is not COLOR_ATTACHMENTm where m is greater than or equal to
            // the value of MAX_COLOR_ATTACHMENTS" and "An INVALID_OPERATION error is generated if a framebuffer object
            // is bound to target and attachment is COLOR_ATTACHMENTm where m is greater than or equal to the value of
            // MAX_COLOR_ATTACHMENTS." Both texts are explicit. The ANGLE build in the matrix swaps the two errors
            // (dEQP's negative state test accepts either error for BACK, so it does not catch this).
            {
                (void)demo::drain_gl_errors();
                GLint v = 0;
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::DefaultFramebufferQueryAttachment::Back, P::ObjectType, &v);
                const mgl::ErrorCode e = demo::drain_gl_errors();
                check.expect_or_known_deviation(e == mgl::ErrorCode::InvalidEnum, "framebuffer object: attachment GL_BACK (a default framebuffer name) -> INVALID_ENUM",
                                                std::format("glGetError() = {}, expected GL_INVALID_ENUM", demo::error_name(e)), "ANGLE",
                                                "ANGLE raises INVALID_OPERATION for BACK with a framebuffer object bound (ES 3.2 section 9.2.3: INVALID_ENUM)");
            }
            if (max_color < 32)
            {
                (void)demo::drain_gl_errors();
                const GLint v = query(mgl::to_framebuffer_attachment(kColor[static_cast<std::size_t>(max_color)]), P::ObjectType);
                const mgl::ErrorCode e = demo::drain_gl_errors();
                check.expect_or_known_deviation(e == mgl::ErrorCode::InvalidOperation,
                                                std::format("framebuffer object: COLOR_ATTACHMENT{} (= MAX_COLOR_ATTACHMENTS) -> INVALID_OPERATION", max_color),
                                                std::format("glGetError() = {}, expected GL_INVALID_OPERATION, value {}", demo::error_name(e), v), "ANGLE",
                                                "ANGLE raises INVALID_ENUM for COLOR_ATTACHMENTm, m >= MAX_COLOR_ATTACHMENTS (ES 3.2 section 9.2.3: INVALID_OPERATION)");
            }
            bind_both(kDefaultFbo);
            GLint object_type = -1, depth_size = -1;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::DefaultFramebufferQueryAttachment::Back, P::ObjectType, &object_type);
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::DefaultFramebufferQueryAttachment::Depth, P::DepthSize, &depth_size);
            check.expect(object_type == GL_FRAMEBUFFER_DEFAULT && depth_size >= 0, "default framebuffer: BACK and DEPTH are the legal attachments (OBJECT_TYPE = FRAMEBUFFER_DEFAULT)",
                         std::format("object type {:#x}, depth size {}", object_type, depth_size));
            check.expect_no_gl_error("...and raise no error");
            // ES 3.0 6.1.13 and ES 3.2 9.2.3: "An INVALID_OPERATION error is generated if the default framebuffer is bound to
            // target and attachment is not BACK, DEPTH, or STENCIL." Mesa (radeonsi, llvmpipe, zink) raises INVALID_ENUM.
            const std::array<std::pair<mgl::FramebufferAttachment, std::string_view>, 3> not_default = {{{kAttDepthStencil, "DEPTH_STENCIL_ATTACHMENT"},
                                                                                                          {kAttC0, "COLOR_ATTACHMENT0"},
                                                                                                          {kAttDepth, "DEPTH_ATTACHMENT"}}};
            for (const auto& [attachment, name] : not_default)
            {
                (void)demo::drain_gl_errors();
                (void)query(attachment, P::ObjectType);
                const mgl::ErrorCode e = demo::drain_gl_errors();
                expect_known_mesa_deviation(check, e == mgl::ErrorCode::InvalidOperation,
                                            std::format("default framebuffer: querying {} (not BACK, DEPTH or STENCIL) -> INVALID_OPERATION", name),
                                            std::format("glGetError() = {}, expected GL_INVALID_OPERATION", demo::error_name(e)),
                                            "ES 3.0 6.1.13 / ES 3.2 9.2.3 define INVALID_OPERATION for an attachment that is not BACK, DEPTH or STENCIL on the default framebuffer; Mesa raises INVALID_ENUM");
            }
        }

        // ------------------------------------------------------------------------------------------
        // glFramebufferTextureLayer (ES 3.0 4.4.2.4): the errors, and that a failed call changes nothing.
        void check_texture_layer_errors(demo::Check& check)
        {
            using P = mgl::FramebufferAttachmentParameter;
            bind_both(view_);
            attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, 3);
            const int max_layers = demo::get_integer(mgl::GetParameter::MaxArrayTextureLayers);
            const int max_3d = demo::get_integer(mgl::GetParameter::Max3DTextureSize);
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with a negative layer -> INVALID_VALUE",
                         [&] { attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, -1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with layer = MAX_ARRAY_TEXTURE_LAYERS on an array texture -> INVALID_VALUE",
                         [&] { attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, max_layers); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with layer = MAX_3D_TEXTURE_SIZE on a 3D texture -> INVALID_VALUE",
                         [&] { attach_layer(mgl::FramebufferTarget::Framebuffer, volume_, 0, max_3d); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with a level the texture cannot have (100) -> INVALID_VALUE",
                         [&] { attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 100, 0); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glFramebufferTextureLayer with a plain 2D texture -> INVALID_OPERATION",
                         [&] { attach_layer(mgl::FramebufferTarget::Framebuffer, plain_, 0, 0); });
            // A cube map: ES 3.0 and 3.1 allow only 3D and 2D array textures (INVALID_OPERATION); ES 3.2 9.2.8 translates the layer
            // into a cube map face. The minimum-version runs only change what the driver reports, so the 3.0 rule is a note.
            (void)demo::drain_gl_errors();
            attach_layer(mgl::FramebufferTarget::Framebuffer, cube_, 0, 0);
            const mgl::ErrorCode cube_error = demo::drain_gl_errors();
            if (version_ >= demo::Version{3, 2})
                check.expect(cube_error == mgl::ErrorCode::NoError, "ES 3.2: glFramebufferTextureLayer accepts a cube map texture (the layer selects the face)",
                             std::format("glGetError() = {}", demo::error_name(cube_error)));
            else
                std::cout << std::format("   note: glFramebufferTextureLayer with a cube map raised {} (ES 3.0/3.1 require INVALID_OPERATION; not asserted: the context only reports {})\n",
                                         demo::error_name(cube_error), demo::to_string(version_));
            attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, 3);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glFramebufferTexture2D with a 3D texture -> INVALID_OPERATION",
                         [&] { attach_face(mgl::FramebufferTarget::Framebuffer, mgl::TextureTarget::Texture2D, volume_, 0); });
            expect_query(check, kAttC0, P::TextureLayer, 3, "a failed attachment call changed nothing: the attachment is still layer 3 of the array texture");
            expect_query(check, kAttC0, P::ObjectName, static_cast<int>(array_.value), "...and still names the array texture");
            bind_both(kDefaultFbo);
        }

        // Rendering into one layer / slice / face changes that image only.
        void check_layer_isolation(demo::Check& check)
        {
            struct Target
            {
                std::string_view name;
                int images;
                std::function<void(int)> attach;
            };
            const std::array<Target, 3> targets = {{
                {"2D array texture", 4, [this](int i) { attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, i); }},
                {"3D texture", 4, [this](int i) { attach_layer(mgl::FramebufferTarget::Framebuffer, volume_, 0, i); }},
                {"cube map", 6, [this](int i) { attach_face(mgl::FramebufferTarget::Framebuffer, kFaces[static_cast<std::size_t>(i)], cube_, 0); }},
            }};
            for (const Target& t : targets)
            {
                bind_both(work_);
                reset_state();
                const Image magenta(kSize, kSize, {255, 0, 255, 255});
                for (int i = 0; i < t.images; ++i)
                {
                    t.attach(i);
                    mgl::glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
                    mgl::glClear(kColorBit);
                }
                const int chosen = t.images - 2;
                t.attach(chosen);
                draw_pattern(kPalette[3], kSize);
                bool others_untouched = true;
                std::string detail;
                for (int i = 0; i < t.images; ++i)
                {
                    t.attach(i);
                    const Image im = read_image(kSize, kSize);
                    const Diff d = compare(im, i == chosen ? pattern_image(kPalette[3], kSize) : magenta, 1);
                    if (d.bad != 0)
                    {
                        others_untouched = false;
                        detail += std::format("image {} differs ({}); ", i, d.first);
                    }
                }
                check.expect(others_untouched, std::format("{}: after clearing all {} images and drawing into image {}, that image holds the pattern and the others the clear color", t.name, t.images, chosen),
                             detail);
            }
            bind_both(kDefaultFbo);
        }

        // ==========================================================================================
        // Checks: invalidation
        // ==========================================================================================
        /// One check per part: the part of `got` equals `want` exactly. Parts in `outside` are compared only outside `hole`.
        void expect_parts(demo::Check& check, std::string_view scenario, const State& want, const State& got, unsigned whole, unsigned outside = 0,
                          std::optional<Rect> hole = std::nullopt) const
        {
            auto one = [&](unsigned bit, std::string_view part, auto&& compare_part) {
                if (((whole | outside) & bit) == 0) return;
                const std::optional<Rect> h = (outside & bit) != 0 ? hole : std::nullopt;
                const int bad = compare_part(h);
                check.expect(bad == 0, std::format("{}: {} {}", scenario, part, h ? "outside the invalidated rectangle is unchanged" : "is unchanged"),
                             bad == 0 ? std::string{} : std::format("{} pixels differ", bad));
            };
            auto image_part = [](const Image& a, const Image& b) {
                return [&a, &b](std::optional<Rect> h) { return compare(a, b, 0, h).bad; };
            };
            auto bits_part = [&](std::uint8_t mask) {
                return [&, mask](std::optional<Rect> h) {
                    int bad = 0;
                    for (int y = 0; y < kSize; ++y)
                        for (int x = 0; x < kSize; ++x)
                            if (!in_rect(h, x, y) && ((got.ds[static_cast<std::size_t>(y * kSize + x)] ^ want.ds[static_cast<std::size_t>(y * kSize + x)]) & mask) != 0) ++bad;
                    return bad;
                };
            };
            one(kPartC0, "color attachment 0", image_part(got.c0, want.c0));
            one(kPartC1, "color attachment 1", image_part(got.c1, want.c1));
            one(kPartC2, "color attachment 2 (a renderbuffer)", image_part(got.c2, want.c2));
            one(kPartDepth, "depth", bits_part(kBitDepth));
            one(kPartStencil, "stencil", bits_part(kBitsStencil));
        }

        static void invalidate(mgl::FramebufferTarget target, std::initializer_list<mgl::FramebufferAttachment> list)
        {
            const std::vector<mgl::FramebufferAttachment> v(list);
            mgl::glInvalidateFramebuffer(target, std::span<const mgl::FramebufferAttachment>(v));
        }

        static void invalidate_sub(mgl::FramebufferTarget target, std::initializer_list<mgl::FramebufferAttachment> list, int x, int y, int w, int h)
        {
            const std::vector<mgl::FramebufferAttachment> v(list);
            mgl::glInvalidateSubFramebuffer(target, std::span<const mgl::FramebufferAttachment>(v), x, y, w, h);
        }

        /// Refills the subject, takes a snapshot, runs `act`, checks that it raised no error and compares the parts.
        template <typename Act>
        void scenario(demo::Check& check, std::string_view name, Act&& act, unsigned whole, unsigned outside = 0, std::optional<Rect> hole = std::nullopt)
        {
            fill(subj_, 3);
            const State before = capture_subject();
            bind_both(subj_);
            (void)demo::drain_gl_errors();
            act();
            check.expect_no_gl_error(std::format("{} raises no error", name));
            const State after = capture_subject();
            expect_parts(check, name, before, after, whole, outside, hole);
        }

        void check_invalidate_framebuffer_objects(demo::Check& check)
        {
            const mgl::FramebufferTarget fb = mgl::FramebufferTarget::Framebuffer;
            // The fill itself must match the models, or "unchanged" would only mean "equal to garbage".
            fill(subj_, 3);
            const State fresh = capture_subject();
            expect_image(check, "the subject's color attachment 0 holds the mrt.frag pattern", fresh.c0, mrt_image(0), 1);
            expect_image(check, "the subject's color attachment 1 holds the mrt.frag pattern", fresh.c1, mrt_image(1), 1);
            expect_image(check, "the subject's color attachment 2 (renderbuffer) holds the mrt.frag pattern", fresh.c2, mrt_image(2), 1);
            int ds_bad = 0;
            for (int y = 0; y < kSize; ++y)
                for (int x = 0; x < kSize; ++x) ds_bad += fresh.ds[static_cast<std::size_t>(y * kSize + x)] == expected_ds_bits(x, y) ? 0 : 1;
            check.expect(ds_bad == 0, "the subject's depth and stencil hold the two-halves scene (probed with tested draws)", std::format("{} pixels differ", ds_bad));
            // Negative control: the probe sees a different depth/stencil state when it is different.
            fill(peer_, 1);
            reset_state();
            bind_both(peer_);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kDepthBit | kStencilBit);
            const State cleared = capture_peer();
            int differ = 0;
            for (int y = 0; y < kSize; ++y)
                for (int x = 0; x < kSize; ++x) differ += cleared.ds[static_cast<std::size_t>(y * kSize + x)] != expected_ds_bits(x, y) ? 1 : 0;
            check.expect(differ == kSize * kSize, "negative control: a framebuffer whose depth and stencil were just cleared probes as different from the scene in every pixel",
                         std::format("{} of {} pixels differ", differ, kSize * kSize));

            // Negative control for "unchanged": a real change (a clear) is detected by the very same comparison.
            {
                bind_both(subj_);
                mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                mgl::glClear(kColorBit | kDepthBit | kStencilBit);
                const State cleared_subject = capture_subject();
                check.expect(compare(cleared_subject.c0, fresh.c0, 0).bad > 3000 && compare(cleared_subject.c1, fresh.c1, 0).bad > 3000 &&
                                 compare(cleared_subject.c2, fresh.c2, 0).bad > 3000,
                             "negative control: after a glClear every color attachment differs from the snapshot in more than 3000 of 4096 pixels (a change is detected)");
            }

            // What the implementation actually does with invalidated contents is undefined by the spec: print, never assert.
            {
                fill(subj_, 3);
                const State pre = capture_subject();
                bind_both(subj_);
                invalidate(fb, {kAttC0, kAttDepthStencil});
                const State post = capture_subject();
                int color_changed = compare(post.c0, pre.c0, 0).bad, ds_changed = 0;
                for (std::size_t i = 0; i < pre.ds.size(); ++i) ds_changed += pre.ds[i] != post.ds[i] ? 1 : 0;
                std::cout << std::format("   note: after invalidating COLOR_ATTACHMENT0 and DEPTH_STENCIL_ATTACHMENT this implementation changed {} of 4096 color pixels and {} of 4096 "
                                         "depth/stencil probes (undefined by the spec; nothing is asserted about them)\n", color_changed, ds_changed);
            }

            // --- whole attachments -----------------------------------------------------------------
            scenario(check, "invalidating COLOR_ATTACHMENT0", [&] { invalidate(fb, {kAttC0}); }, kPartC1 | kPartC2 | kPartDepth | kPartStencil);
            scenario(check, "invalidating COLOR_ATTACHMENT1 and 2 (DRAW_FRAMEBUFFER target)", [&] { invalidate(mgl::FramebufferTarget::DrawFramebuffer, {kAttC1, kAttC2}); },
                     kPartC0 | kPartDepth | kPartStencil);
            scenario(check, "invalidating DEPTH_ATTACHMENT of a DEPTH24_STENCIL8 image", [&] { invalidate(fb, {kAttDepth}); }, kPartC0 | kPartC1 | kPartC2 | kPartStencil);
            scenario(check, "invalidating STENCIL_ATTACHMENT of a DEPTH24_STENCIL8 image", [&] { invalidate(fb, {kAttStencil}); }, kPartC0 | kPartC1 | kPartC2 | kPartDepth);
            scenario(check, "invalidating DEPTH_STENCIL_ATTACHMENT", [&] { invalidate(fb, {kAttDepthStencil}); }, kPartC0 | kPartC1 | kPartC2);
            scenario(check, "invalidating DEPTH_ATTACHMENT and STENCIL_ATTACHMENT in one list", [&] { invalidate(fb, {kAttDepth, kAttStencil}); }, kPartC0 | kPartC1 | kPartC2);
            scenario(check, "invalidating an attachment point with no image (COLOR_ATTACHMENT3)", [&] { invalidate(fb, {mgl::to_framebuffer_attachment(kColor[3])}); }, kPartAll);
            scenario(check, "invalidating with an empty list (numAttachments = 0)", [&] { invalidate(fb, {}); }, kPartAll);

            // --- regions -----------------------------------------------------------------------------
            const Rect hole = {8, 8, 24, 24};
            scenario(check, "invalidating the region (8,8) 16x16 of COLOR_ATTACHMENT0", [&] { invalidate_sub(fb, {kAttC0}, 8, 8, 16, 16); },
                     kPartC1 | kPartC2 | kPartDepth | kPartStencil, kPartC0, hole);
            scenario(check, "invalidating the region (8,8) 16x16 of DEPTH_STENCIL_ATTACHMENT", [&] { invalidate_sub(fb, {kAttDepthStencil}, 8, 8, 16, 16); },
                     kPartC0 | kPartC1 | kPartC2, kPartDepth | kPartStencil, hole);
            scenario(check, "invalidating a region that starts left of and below the image: (-10,-10) 30x30", [&] { invalidate_sub(fb, {kAttC0, kAttC1}, -10, -10, 30, 30); },
                     kPartC2 | kPartDepth | kPartStencil, kPartC0 | kPartC1, Rect{0, 0, 20, 20});
            scenario(check, "invalidating a region that extends beyond the image: (40,40) 1000x1000", [&] { invalidate_sub(fb, {kAttC0, kAttDepthStencil}, 40, 40, 1000, 1000); },
                     kPartC1 | kPartC2, kPartC0 | kPartDepth | kPartStencil, Rect{40, 40, kSize, kSize});
            scenario(check, "invalidating a region entirely outside the image: (200,200) 10x10", [&] { invalidate_sub(fb, {kAttC0, kAttDepthStencil}, 200, 200, 10, 10); }, kPartAll);
            scenario(check, "invalidating a region of width 0", [&] { invalidate_sub(fb, {kAttC0, kAttDepthStencil}, 0, 0, 0, 20); }, kPartAll);
            scenario(check, "invalidating a region of height 0", [&] { invalidate_sub(fb, {kAttC0, kAttDepthStencil}, 0, 0, 20, 0); }, kPartAll);

            // --- the target argument: READ_FRAMEBUFFER, DRAW_FRAMEBUFFER and FRAMEBUFFER (= DRAW) ----------
            {
                fill(subj_, 3);
                fill(peer_, 1);
                const State subj_before = capture_subject();
                const State peer_before = capture_peer();
                bind_read(subj_);
                bind_draw(peer_);
                (void)demo::drain_gl_errors();
                invalidate(mgl::FramebufferTarget::ReadFramebuffer, {kAttC0, kAttDepthStencil});
                check.expect_no_gl_error("invalidating READ_FRAMEBUFFER (the subject) while another framebuffer is the draw framebuffer raises no error");
                expect_parts(check, "READ_FRAMEBUFFER invalidation, the subject", subj_before, capture_subject(), kPartC1 | kPartC2);
                expect_parts(check, "READ_FRAMEBUFFER invalidation, the DRAW framebuffer (the peer)", peer_before, capture_peer(), kPartC0 | kPartDepth | kPartStencil);

                fill(subj_, 3);
                fill(peer_, 1);
                const State subj_before2 = capture_subject();
                bind_read(subj_);
                bind_draw(peer_);
                invalidate(mgl::FramebufferTarget::DrawFramebuffer, {kAttC0, kAttDepthStencil});
                check.expect_no_gl_error("invalidating DRAW_FRAMEBUFFER (the peer) raises no error");
                expect_parts(check, "DRAW_FRAMEBUFFER invalidation, the READ framebuffer (the subject)", subj_before2, capture_subject(), kPartAll);

                fill(subj_, 3);
                fill(peer_, 1);
                const State subj_before3 = capture_subject();
                bind_read(subj_);
                bind_draw(peer_);
                invalidate(fb, {kAttC0, kAttDepthStencil});
                check.expect_no_gl_error("invalidating FRAMEBUFFER raises no error");
                expect_parts(check, "FRAMEBUFFER invalidation (it means DRAW), the READ framebuffer (the subject)", subj_before3, capture_subject(), kPartAll);
            }

            // --- nothing to invalidate / incomplete ---------------------------------------------------------
            bind_both(empty_);
            (void)demo::drain_gl_errors();
            invalidate(fb, {kAttC0, kAttDepth, kAttStencil, kAttDepthStencil});
            check.expect_no_gl_error("invalidating attachments of a framebuffer with no images (an incomplete framebuffer) raises no error (the call may be ignored)");
            invalidate_sub(fb, {kAttC0}, 0, 0, 8, 8);
            check.expect_no_gl_error("...glInvalidateSubFramebuffer on it raises no error either");
            bind_both(work_);
            attach_face(fb, mgl::TextureTarget::Texture2D, plain_, 0);
            mgl::glFramebufferTexture2D(fb, kAttDepth, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            invalidate(fb, {kAttDepth, kAttStencil, kAttDepthStencil});
            check.expect_no_gl_error("invalidating depth and stencil of a color-only framebuffer: attachments that do not exist are ignored, no error");
            bind_both(kDefaultFbo);
        }

        // Invalidating the attachment that is one layer / one face of a texture does not touch the other images.
        void check_invalidate_texture_images(demo::Check& check)
        {
            render_targets();
            const mgl::FramebufferTarget fb = mgl::FramebufferTarget::Framebuffer;
            struct Group
            {
                std::string_view name;
                int images;
                int invalidated;
                std::function<void(int)> attach;
            };
            const std::array<Group, 3> groups = {{
                {"2D array texture", 4, 1, [this](int i) { attach_layer(mgl::FramebufferTarget::Framebuffer, array_, 0, i); }},
                {"3D texture", 4, 2, [this](int i) { attach_layer(mgl::FramebufferTarget::Framebuffer, volume_, 0, i); }},
                {"cube map", 6, 3, [this](int i) { attach_face(mgl::FramebufferTarget::Framebuffer, kFaces[static_cast<std::size_t>(i)], cube_, 0); }},
            }};
            for (const Group& g : groups)
            {
                bind_both(view_);
                std::vector<Image> before;
                for (int i = 0; i < g.images; ++i)
                {
                    g.attach(i);
                    before.push_back(read_image(kSize, kSize));
                }
                g.attach(g.invalidated);
                (void)demo::drain_gl_errors();
                invalidate(fb, {kAttC0});
                const bool no_error = demo::drain_gl_errors() == mgl::ErrorCode::NoError;
                int bad_images = 0;
                for (int i = 0; i < g.images; ++i)
                {
                    g.attach(i);
                    if (i == g.invalidated) continue;  // undefined now
                    bad_images += compare(read_image(kSize, kSize), before[static_cast<std::size_t>(i)], 0).bad != 0 ? 1 : 0;
                }
                check.expect(no_error && bad_images == 0,
                             std::format("{}: invalidating the color attachment that is image {} raises no error and the other {} images are unchanged", g.name, g.invalidated, g.images - 1),
                             std::format("{} images changed", bad_images));
            }
            bind_both(kDefaultFbo);
        }

        // ------------------------------------------------------------------------------------------
        void check_invalidate_errors(demo::Check& check)
        {
            const mgl::FramebufferTarget fb = mgl::FramebufferTarget::Framebuffer;
            fill(subj_, 3);
            const State before = capture_subject();
            bind_both(subj_);

            // A framebuffer object is bound: the default framebuffer's tokens are not attachment points (Table 4.6).
            for (const mgl::DefaultFramebufferAttachment token : {kDefColor, kDefDepth, kDefStencil})
            {
                const std::string_view name = token == kDefColor ? "COLOR" : token == kDefDepth ? "DEPTH" : "STENCIL";
                const std::array<mgl::DefaultFramebufferAttachment, 1> list = {token};
                expect_error(check, mgl::ErrorCode::InvalidEnum, std::format("framebuffer object bound, attachment GL_{} (default framebuffer token) -> INVALID_ENUM", name),
                             [&] { mgl::glInvalidateFramebuffer(fb, std::span<const mgl::DefaultFramebufferAttachment>(list)); });
                expect_error(check, mgl::ErrorCode::InvalidEnum, std::format("...glInvalidateSubFramebuffer with GL_{} -> INVALID_ENUM", name),
                             [&] { mgl::glInvalidateSubFramebuffer(fb, std::span<const mgl::DefaultFramebufferAttachment>(list), 0, 0, 8, 8); });
            }
            expect_error(check, mgl::ErrorCode::InvalidEnum, "framebuffer object bound, attachment list {COLOR_ATTACHMENT0, GL_NONE} -> INVALID_ENUM",
                         [&] { invalidate(fb, {kAttC0, mgl::FramebufferAttachment::None}); });
            const int max_color = demo::get_integer(mgl::GetParameter::MaxColorAttachments);
            if (max_color < 32)
            {
                const mgl::FramebufferAttachment too_big = mgl::to_framebuffer_attachment(kColor[static_cast<std::size_t>(max_color)]);
                expect_error(check, mgl::ErrorCode::InvalidOperation, std::format("COLOR_ATTACHMENT{} (= MAX_COLOR_ATTACHMENTS) in the list -> INVALID_OPERATION", max_color),
                             [&] { invalidate(fb, {kAttC0, too_big}); });
                expect_error(check, mgl::ErrorCode::InvalidOperation, "...glInvalidateSubFramebuffer with it -> INVALID_OPERATION",
                             [&] { invalidate_sub(fb, {kAttC0, too_big}, 0, 0, 8, 8); });
            }
            // Negative sizes.
            const std::array<mgl::FramebufferAttachment, 1> c0 = {kAttC0};
            expect_error(check, mgl::ErrorCode::InvalidValue, "numAttachments = -1 -> INVALID_VALUE", [&] { mgl::glInvalidateFramebuffer(fb, -1, c0.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "...glInvalidateSubFramebuffer with numAttachments = -1 -> INVALID_VALUE",
                         [&] { mgl::glInvalidateSubFramebuffer(fb, -1, c0.data(), 0, 0, 8, 8); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "...width = -1 -> INVALID_VALUE", [&] { invalidate_sub(fb, {kAttC0}, 0, 0, -1, 8); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "...height = -1 -> INVALID_VALUE", [&] { invalidate_sub(fb, {kAttC0}, 0, 0, 8, -1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "...width = height = -1 -> INVALID_VALUE", [&] { invalidate_sub(fb, {kAttC0, kAttDepthStencil}, 3, 3, -1, -1); });
            // A command that raises an error has no effect: none of the lists above invalidated anything.
            expect_parts(check, "after the failed calls on the framebuffer object (errors have no effect)", before, capture_subject(), kPartAll);

            // The default framebuffer is bound: only COLOR, DEPTH and STENCIL are attachment points.
            bind_both(kDefaultFbo);
            const std::array<mgl::DefaultFramebufferAttachment, 1> def_color = {kDefColor};
            const std::array<std::pair<mgl::FramebufferAttachment, std::string_view>, 4> wrong = {{{kAttDepth, "DEPTH_ATTACHMENT"},
                                                                                                    {kAttDepthStencil, "DEPTH_STENCIL_ATTACHMENT"},
                                                                                                    {kAttC0, "COLOR_ATTACHMENT0"},
                                                                                                    {mgl::FramebufferAttachment::None, "GL_NONE"}}};
            for (const auto& [attachment, name] : wrong)
            {
                const std::array<mgl::FramebufferAttachment, 1> list = {attachment};
                expect_error(check, mgl::ErrorCode::InvalidEnum, std::format("default framebuffer bound, attachment {} -> INVALID_ENUM", name),
                             [&] { mgl::glInvalidateFramebuffer(fb, std::span<const mgl::FramebufferAttachment>(list)); });
                expect_error(check, mgl::ErrorCode::InvalidEnum, std::format("...glInvalidateSubFramebuffer with {} -> INVALID_ENUM", name),
                             [&] { mgl::glInvalidateSubFramebuffer(fb, std::span<const mgl::FramebufferAttachment>(list), 0, 0, 8, 8); });
            }
            expect_error(check, mgl::ErrorCode::InvalidValue, "default framebuffer, numAttachments = -1 -> INVALID_VALUE",
                         [&] { mgl::glInvalidateFramebuffer(fb, -1, def_color.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "default framebuffer, InvalidateSub width = -1 -> INVALID_VALUE",
                         [&] { mgl::glInvalidateSubFramebuffer(fb, std::span<const mgl::DefaultFramebufferAttachment>(def_color), 0, 0, -1, 8); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "default framebuffer, InvalidateSub height = -1 -> INVALID_VALUE",
                         [&] { mgl::glInvalidateSubFramebuffer(fb, std::span<const mgl::DefaultFramebufferAttachment>(def_color), 0, 0, 8, -1); });
        }

        // ------------------------------------------------------------------------------------------
        void check_invalidate_default_framebuffer(demo::Check& check)
        {
            const mgl::FramebufferTarget fb = mgl::FramebufferTarget::Framebuffer;
            bind_both(kDefaultFbo);
            using D = mgl::DefaultFramebufferAttachment;
            struct Case
            {
                std::string_view name;
                std::vector<D> list;
            };
            const std::array<Case, 5> cases = {{{"COLOR", {kDefColor}}, {"DEPTH", {kDefDepth}}, {"STENCIL", {kDefStencil}},
                                                {"DEPTH and STENCIL", {kDefDepth, kDefStencil}}, {"COLOR, DEPTH and STENCIL", {kDefColor, kDefDepth, kDefStencil}}}};
            for (const Case& c : cases)
            {
                (void)demo::drain_gl_errors();
                mgl::glInvalidateFramebuffer(fb, std::span<const D>(c.list));
                check.expect_no_gl_error(std::format("default framebuffer: glInvalidateFramebuffer({}) raises no error", c.name));
                mgl::glInvalidateSubFramebuffer(fb, std::span<const D>(c.list), 16, 16, 64, 64);
                check.expect_no_gl_error(std::format("default framebuffer: glInvalidateSubFramebuffer({}, a 64x64 region) raises no error", c.name));
            }
            mgl::glInvalidateSubFramebuffer(fb, std::span<const D>(cases[4].list), -50, -50, 100000, 100000);
            check.expect_no_gl_error("default framebuffer: a region far larger than the window is clipped, no error");
            mgl::glInvalidateFramebuffer(fb, std::span<const D>{});
            check.expect_no_gl_error("default framebuffer: an empty list is a no-op, no error");

            // Rendering after the invalidation of color, depth and stencil works: clear everything, then draw two
            // overlapping quads with the depth test; the nearer one (drawn first) must survive the farther one.
            mgl::glInvalidateFramebuffer(fb, std::span<const D>(cases[4].list));
            reset_state();
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(0, 0, 32, 32);
            mgl::glViewport(0, 0, kWindow, kWindow);
            mgl::glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(kColorBit | kDepthBit | kStencilBit);
            check.expect_pixel(10, 10, {64, 128, 191, 255}, 1, "after invalidating the default framebuffer's color, depth and stencil, a clear works (color 0.25, 0.5, 0.75)");
            const int depth_bits = demo::get_integer(mgl::GetParameter::DepthBits);
            if (depth_bits > 0)
            {
                mgl::glEnable(mgl::Capability::DepthTest);
                mgl::glDepthFunc(mgl::CompareFunc::Less);
                mgl::glUseProgram(flat_.program);
                mgl::glUniform4f(flat_color_, 1.0f, 0.0f, 0.0f, 1.0f);
                draw_full(flat_, ndc_depth(0.3));
                mgl::glUniform4f(flat_color_, 0.0f, 1.0f, 0.0f, 1.0f);
                draw_full(flat_, ndc_depth(0.6));
                check.expect_pixel(10, 10, {255, 0, 0, 255}, 1, "...and the depth buffer works: the nearer red quad survives the farther green one");
                mgl::glDisable(mgl::Capability::DepthTest);
                mgl::glUniform4f(flat_color_, 0.0f, 0.0f, 1.0f, 1.0f);
                draw_full(flat_, ndc_depth(0.6));
                check.expect_pixel(10, 10, {0, 0, 255, 255}, 1, "negative control: with the depth test off the farther blue quad is drawn over the red one");
            }
            else
            {
                std::cout << "   note: the default framebuffer has no depth buffer; the depth part of the test is skipped\n";
            }
            reset_state();
            bind_both(kDefaultFbo);
        }

        // ------------------------------------------------------------------------------------------
        demo::Version version_{};
        Quad pattern_, mrt_, flat_;
        mgl::UniformLocation pattern_base_{}, pattern_size_{}, flat_color_{};

        mgl::TextureId cube_{}, array_{}, volume_{}, plain_{}, tex_s0_{}, tex_s1_{}, tex_p0_{}, tex_probe_s_{}, tex_probe_p_{};
        mgl::RenderbufferId rb_s2_{}, rb_ds_{}, rb_pds_{};
        mgl::FramebufferId work_{}, view_{}, subj_{}, peer_{}, probe_subj_{}, probe_peer_{}, empty_{};
        std::vector<mgl::FramebufferId> framebuffers_;
        std::vector<mgl::RenderbufferId> renderbuffers_;
        std::vector<mgl::TextureId> textures_;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es30-framebuffer-invalidate",
        .title = "Framebuffer invalidation and attachment queries",
        .min_es = {3, 0},
        .summary = "glInvalidateFramebuffer / glInvalidateSubFramebuffer on framebuffer objects and the default framebuffer: legal calls, the defined errors, "
                   "what stays untouched; plus attachment queries for texture layer, cube-map face, component type and DEPTH_STENCIL_ATTACHMENT.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<FramebufferInvalidate>(argc, argv, info);
}
