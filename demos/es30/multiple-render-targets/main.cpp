// es30-multiple-render-targets
//
// Multiple render targets (OpenGL ES 3.0): one framebuffer object with four color attachments
// (two RGBA8 textures, one RGBA32UI texture, one RG32I texture) written by a single fragment shader
// whose outputs are bound to draw buffers with layout(location = N). The demo exercises
//
//   * glDrawBuffers (ES 3.0 4.2.1): the mapping of fragment outputs to attachments, NONE gaps
//     (an output routed to NONE is discarded), n = 0, the rule that draw buffer i must be
//     COLOR_ATTACHMENTi or NONE, and the errors of the default framebuffer;
//   * glClearBufferfv / iv / uiv / fi (4.2.3): per-attachment clears of normalized, signed and
//     unsigned integer buffers, and a combined depth+stencil clear that is proven by a depth- and
//     stencil-tested draw;
//   * glReadBuffer + glReadPixels (4.3.1, 4.3.2): reading each attachment, integer formats
//     (RGBA_INTEGER with INT / UNSIGNED_INT) and the error rules;
//   * glGetFragDataLocation (3.9.2.3) and the compile-time rules for fragment outputs (GLSL ES 3.00 4.3.8.2);
//   * MAX_DRAW_BUFFERS, MAX_COLOR_ATTACHMENTS, DRAW_BUFFERi, READ_BUFFER queries.
//
// Image (256 x 256 window): the four attachments of the 128 x 128 offscreen target side by side,
//   top-left albedo (RGBA8) | top-right data (RGBA8)
//   bottom-left ids (RGBA32UI) | bottom-right signed (RG32I)
// The integer attachments are shown through formulas that the CPU repeats (see composite.frag).
//
// Validation: every pixel of every attachment is compared with a CPU model of the fragment shader:
// exactly for the integer attachments, within +-1/255 for the normalized ones.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kSize = 128;        // the offscreen render targets are kSize x kSize
    constexpr int kWindow = 2 * kSize;  // the window (default framebuffer) is 2 x 2 panels
    constexpr int kAttachments = 4;

    constexpr std::array<mgl::ColorAttachment, kAttachments> kAttach = {
        mgl::ColorAttachment::Color0, mgl::ColorAttachment::Color1, mgl::ColorAttachment::Color2, mgl::ColorAttachment::Color3};

    // glDrawBuffers takes DrawBuffer values; ColorAttachment values are converted with to_draw_buffer().
    constexpr mgl::DrawBuffer kC0 = mgl::to_draw_buffer(mgl::ColorAttachment::Color0);
    constexpr mgl::DrawBuffer kC1 = mgl::to_draw_buffer(mgl::ColorAttachment::Color1);
    constexpr mgl::DrawBuffer kC2 = mgl::to_draw_buffer(mgl::ColorAttachment::Color2);
    constexpr mgl::DrawBuffer kC3 = mgl::to_draw_buffer(mgl::ColorAttachment::Color3);
    constexpr mgl::DrawBuffer kNone = mgl::DrawBuffer::None;
    constexpr mgl::DrawBuffer kBack = mgl::DrawBuffer::Back;

    // GetIntegerv(DRAW_BUFFERi) pnames. The enumerators are listed one by one on purpose.
    constexpr std::array<mgl::GetParameter, 16> kDrawBufferPname = {
        mgl::GetParameter::DrawBuffer0,  mgl::GetParameter::DrawBuffer1,  mgl::GetParameter::DrawBuffer2,  mgl::GetParameter::DrawBuffer3,
        mgl::GetParameter::DrawBuffer4,  mgl::GetParameter::DrawBuffer5,  mgl::GetParameter::DrawBuffer6,  mgl::GetParameter::DrawBuffer7,
        mgl::GetParameter::DrawBuffer8,  mgl::GetParameter::DrawBuffer9,  mgl::GetParameter::DrawBuffer10, mgl::GetParameter::DrawBuffer11,
        mgl::GetParameter::DrawBuffer12, mgl::GetParameter::DrawBuffer13, mgl::GetParameter::DrawBuffer14, mgl::GetParameter::DrawBuffer15};

    constexpr GLint value_of(mgl::DrawBuffer d) { return static_cast<GLint>(static_cast<GLenum>(d)); }
    constexpr GLint value_of(mgl::ColorAttachment a) { return static_cast<GLint>(static_cast<GLenum>(a)); }

    std::string buffer_name(GLint v)
    {
        if (v == GL_NONE) return "NONE";
        if (v == GL_BACK) return "BACK";
        if (v >= GL_COLOR_ATTACHMENT0 && v <= GL_COLOR_ATTACHMENT0 + 31) return std::format("COLOR_ATTACHMENT{}", v - GL_COLOR_ATTACHMENT0);
        return std::format("{:#x}", v);
    }

    // ---------------------------------------------------------------------------------------------
    // CPU model of mrt.frag. (x, y) is the pixel coordinate; everything is a function of it.
    // ---------------------------------------------------------------------------------------------
    using Expected = std::array<double, 4>;

    Expected albedo_model(int x, int y)
    {
        const double u = (x + 0.5) / kSize, v = (y + 0.5) / kSize;
        const double dark = (((x >> 4) + (y >> 4)) & 1) ? 0.5 : 1.0;
        return {255.0 * u * dark, 255.0 * v * dark, 255.0 * 0.25 * dark, 255.0};
    }

    Expected data_model(int x, int y)
    {
        const double u = (x + 0.5) / kSize, v = (y + 0.5) / kSize;
        return {255.0 * (1.0 - u), 255.0 * (1.0 - v), 255.0 * 0.5 * (u + v), 255.0 * 0.75};
    }

    std::uint32_t ids_counter(int x, int y) { return 0xFFFFFFFFu - static_cast<std::uint32_t>(y * kSize + x); }

    // What glReadPixels(RGBA_INTEGER, UNSIGNED_INT) returns for the RGBA32UI attachment.
    Expected ids_model(int x, int y)
    {
        const std::uint32_t z = 0xDEAD0000u | (static_cast<std::uint32_t>(x) << 8) | static_cast<std::uint32_t>(y);
        return {double(x), double(y), double(z), double(ids_counter(x, y))};
    }

    // What glReadPixels(RGBA_INTEGER, INT) returns for the RG32I attachment: the framebuffer has no
    // B and A, so they read back as 0 and 1 (ES 3.0 4.3.2.1).
    Expected signed_model(int x, int y)
    {
        return {double(-(x * y) - 1), double(std::numeric_limits<std::int32_t>::min() + x), 0.0, 1.0};
    }

    // The composite shader's colors (composite.frag) for the same pixel, in 0..255.
    Expected composite_model(int window_x, int window_y)
    {
        const int panel = window_x / kSize + 2 * (1 - window_y / kSize);  // panel 0 is the top-left quarter
        const int x = window_x % kSize, y = window_y % kSize;
        switch (panel)
        {
        case 0: { Expected e = albedo_model(x, y); e[3] = 255.0; return e; }  // composite.frag writes alpha 1
        case 1: { Expected e = data_model(x, y); e[3] = 255.0; return e; }
        case 2: return {255.0 * x / 127.0, 255.0 * y / 127.0, 255.0 * (ids_counter(x, y) & 255u) / 255.0, 255.0};
        default: return {255.0 * (x * y) / 16129.0, 255.0 * x / 127.0, 255.0 * 0.4, 255.0};
        }
    }

    struct Diff
    {
        int bad = 0;
        std::string first;
    };

    /// Compares an image of kSize x kSize RGBA texels (or `width` x `height`) with a per-pixel model.
    template <typename T, typename Model>
    Diff compare(const std::vector<T>& data, int width, int height, Model&& model, double tolerance)
    {
        Diff d;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                const Expected e = model(x, y);
                for (int c = 0; c < 4; ++c)
                {
                    const double got = static_cast<double>(data[static_cast<std::size_t>((y * width + x) * 4 + c)]);
                    if (std::abs(got - e[static_cast<std::size_t>(c)]) > tolerance)
                    {
                        if (d.bad++ == 0)
                            d.first = std::format("first at ({},{}) channel {}: got {} expected {}", x, y, c, got, e[static_cast<std::size_t>(c)]);
                        break;
                    }
                }
            }
        return d;
    }

    template <typename T>
    Diff compare_model(const std::vector<T>& data, Expected (*model)(int, int), double tolerance)
    {
        return compare(data, kSize, kSize, model, tolerance);
    }

    template <typename T>
    Diff compare_constant(const std::vector<T>& data, Expected value, double tolerance)
    {
        return compare(data, kSize, kSize, [&](int, int) { return value; }, tolerance);
    }

    std::string describe(const Diff& d) { return d.bad == 0 ? std::string{} : std::format("{} pixels differ, {}", d.bad, d.first); }

    // The sentinel contents written by set_sentinels(): they differ from everything the shader
    // writes, so "unchanged" and "overwritten" are distinguishable at every pixel.
    constexpr Expected kSentinel0 = {10, 20, 30, 40};         // RGBA8 attachment 0
    constexpr Expected kSentinel1 = {50, 60, 70, 80};         // RGBA8 attachment 1
    constexpr Expected kSentinel2 = {11, 22, 33, 44};         // RGBA32UI attachment 2
    constexpr Expected kSentinel3 = {-11, -22, 0, 1};         // RG32I attachment 3 (B/A read as 0/1)

    class MultipleRenderTargets final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kWindow || env.height < kWindow)
                throw demo::Skip(std::format("this demo draws into the lower left {0}x{0} pixels and needs a window at least that large", kWindow));

            vertex_source_ = env.load_text("shaders/quad.vert");
            mrt_ = demo::build_program(env, "shaders/quad.vert", "shaders/mrt.frag");
            single_ = demo::build_program(env, "shaders/quad.vert", "shaders/single.frag");
            composite_ = demo::build_program(env, "shaders/quad.vert", "shaders/composite.frag");

            // State of the default framebuffer, read before any framebuffer object exists (ES 3.0 4.2.1, 4.3.1).
            initial_.default_draw0 = demo::get_integer(mgl::GetParameter::DrawBuffer0);
            initial_.default_draw1 = demo::get_integer(mgl::GetParameter::DrawBuffer1);
            initial_.default_read = demo::get_integer(mgl::GetParameter::ReadBuffer);
            initial_.max_draw_buffers = demo::get_integer(mgl::GetParameter::MaxDrawBuffers);
            initial_.max_color_attachments = demo::get_integer(mgl::GetParameter::MaxColorAttachments);

            // Four texture attachments. Sized formats + glTexStorage2D (immutable storage, ES 3.0 3.8.4).
            // R8/RG8/RGBA8 and the 8/16/32-bit integer formats with 1, 2 or 4 components are
            // color-renderable (ES 3.0 Table 3.13); the three-component integer formats are not.
            const std::array<mgl::SizedInternalFormat, kAttachments> formats = {
                mgl::SizedInternalFormat::Rgba8, mgl::SizedInternalFormat::Rgba8, mgl::SizedInternalFormat::Rgba32UI,
                mgl::SizedInternalFormat::Rg32I};
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            for (std::size_t i = 0; i < tex_.size(); ++i)
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[i]);
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, formats[i], kSize, kSize);
                // NEAREST filters: the default MIN_FILTER needs a full mipmap chain (the texture would be
                // incomplete), and integer textures are not filterable at all.
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            }

            // Depth + stencil: a DEPTH24_STENCIL8 renderbuffer for the glClearBufferfi part.
            mgl::glGenRenderbuffers(1, &depth_stencil_);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, depth_stencil_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Depth24Stencil8, kSize, kSize);

            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            for (std::size_t i = 0; i < tex_.size(); ++i)
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(kAttach[i]),
                                            mgl::TextureTarget::Texture2D, tex_[i], 0);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::DepthStencil,
                                           mgl::RenderbufferTarget::Renderbuffer, depth_stencil_);
            initial_.status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (initial_.status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer is not complete: {}", demo::framebuffer_status_name(initial_.status)));

            // State of a fresh framebuffer object: nothing has called glDrawBuffers / glReadBuffer on it yet.
            for (std::size_t i = 0; i < 4; ++i) initial_.fbo_draw[i] = demo::get_integer(kDrawBufferPname[i]);
            initial_.fbo_read = demo::get_integer(mgl::GetParameter::ReadBuffer);
            for (std::size_t i = 0; i < tex_.size(); ++i)
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(kAttach[i]),
                                                           mgl::FramebufferAttachmentParameter::ComponentType, &initial_.component_type[i]);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            // Pass 1: all four fragment outputs of mrt.frag into the four attachments.
            bind_fbo();
            const std::array<mgl::DrawBuffer, 4> all = {kC0, kC1, kC2, kC3};
            mgl::glDrawBuffers(all);
            draw_mrt(kSize);

            // Pass 2: composite the attachments into the window, one quarter each.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, kWindow, kWindow);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(composite_);
            const std::array<const char*, 4> samplers = {"u_albedo", "u_data", "u_ids", "u_signed"};
            const std::array<mgl::TextureUnit, 4> units = {mgl::TextureUnit::Texture0, mgl::TextureUnit::Texture1,
                                                          mgl::TextureUnit::Texture2, mgl::TextureUnit::Texture3};
            for (std::size_t i = 0; i < 4; ++i)
            {
                mgl::glActiveTexture(units[i]);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[i]);
                mgl::glUniform1i(mgl::glGetUniformLocation(composite_, samplers[i]), static_cast<GLint>(i));  // unit index
            }
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glUniform1f(mgl::glGetUniformLocation(composite_, "u_z"), 0.0f);
            for (int panel = 0; panel < 4; ++panel)
            {
                const int ox = (panel % 2) * kSize, oy = (1 - panel / 2) * kSize;
                mgl::glViewport(ox, oy, kSize, kSize);
                mgl::glUniform1i(mgl::glGetUniformLocation(composite_, "u_panel"), panel);
                mgl::glUniform2i(mgl::glGetUniformLocation(composite_, "u_origin"), ox, oy);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            // The window and the attachments hold the results of render(); read them before any test
            // below modifies the attachments.
            check_composite(check);
            check_initial_state(check);
            check_mrt_draw(check);
            check_none_gaps(check);
            check_single_output(check);
            check_zero_draw_buffers(check);
            check_mapping_errors(check);
            check_readback_rules(check);
            check_clears(check);
            check_depth_stencil_clear(check);
            check_frag_data_location(check);
            check_output_rules(check);

            // Leave the default framebuffer bound, as render() found it.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteRenderbuffers(1, &depth_stencil_);
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(mrt_);
            mgl::glDeleteProgram(single_);
            mgl::glDeleteProgram(composite_);
        }

    private:
        // ------------------------------------------------------------------------------------------
        // Small helpers (every GL call stays visible).
        // ------------------------------------------------------------------------------------------
        void bind_fbo() const { mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_); }

        void draw_quad(mgl::ProgramId program, float ndc_z) const
        {
            mgl::glUniform1f(mgl::glGetUniformLocation(program, "u_z"), ndc_z);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// Draws mrt.frag over a size x size viewport (all outputs). The draw buffers are set by the caller.
        void draw_mrt(int size, float ndc_z = 0.0f) const
        {
            mgl::glViewport(0, 0, size, size);
            mgl::glUseProgram(mrt_);
            mgl::glUniform2i(mgl::glGetUniformLocation(mrt_, "u_size"), kSize, kSize);
            draw_quad(mrt_, ndc_z);
        }

        std::vector<std::uint8_t> read_unorm(std::size_t attachment) const
        {
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[attachment]));
            std::vector<std::uint8_t> px(static_cast<std::size_t>(kSize * kSize * 4));
            mgl::glReadPixels(0, 0, kSize, kSize, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            return px;
        }

        std::vector<std::uint32_t> read_uint(std::size_t attachment) const
        {
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[attachment]));
            std::vector<std::uint32_t> px(static_cast<std::size_t>(kSize * kSize * 4));
            mgl::glReadPixels(0, 0, kSize, kSize, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, px.data());
            return px;
        }

        std::vector<std::int32_t> read_int(std::size_t attachment) const
        {
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[attachment]));
            std::vector<std::int32_t> px(static_cast<std::size_t>(kSize * kSize * 4));
            mgl::glReadPixels(0, 0, kSize, kSize, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int, px.data());
            return px;
        }

        /// Writes a known value into every pixel of every attachment: glClearBuffer{fv,uiv,iv} with each
        /// of the four draw buffers enabled (a clear of DRAW_BUFFERi = NONE would do nothing).
        void set_sentinels() const
        {
            bind_fbo();
            const std::array<mgl::DrawBuffer, 4> all = {kC0, kC1, kC2, kC3};
            mgl::glDrawBuffers(all);
            auto normalized = [](const Expected& bytes) {
                return std::array<GLfloat, 4>{static_cast<GLfloat>(bytes[0] / 255.0), static_cast<GLfloat>(bytes[1] / 255.0),
                                              static_cast<GLfloat>(bytes[2] / 255.0), static_cast<GLfloat>(bytes[3] / 255.0)};
            };
            const std::array<GLfloat, 4> f0 = normalized(kSentinel0);
            const std::array<GLfloat, 4> f1 = normalized(kSentinel1);
            const std::array<GLuint, 4> u2 = {11u, 22u, 33u, 44u};
            const std::array<GLint, 4> i3 = {-11, -22, -33, -44};
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, f0.data());
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 1, f1.data());
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 2, u2.data());
            mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 3, i3.data());
        }

        // The four expectations of an attachment that still holds its sentinel / holds the shader pattern.
        Diff diff_attachment(std::size_t i, bool sentinel) const
        {
            switch (i)
            {
            case 0: return sentinel ? compare_constant(read_unorm(0), kSentinel0, 1.0) : compare_model(read_unorm(0), albedo_model, 1.0);
            case 1: return sentinel ? compare_constant(read_unorm(1), kSentinel1, 1.0) : compare_model(read_unorm(1), data_model, 1.0);
            case 2: return sentinel ? compare_constant(read_uint(2), kSentinel2, 0.0) : compare_model(read_uint(2), ids_model, 0.0);
            default: return sentinel ? compare_constant(read_int(3), kSentinel3, 0.0) : compare_model(read_int(3), signed_model, 0.0);
            }
        }

        static std::vector<GLint> query_draw_buffers(GLint count)
        {
            std::vector<GLint> v;
            for (GLint i = 0; i < count && i < 16; ++i) v.push_back(demo::get_integer(kDrawBufferPname[static_cast<std::size_t>(i)]));
            return v;
        }

        /// DRAW_BUFFER0..3 must equal `first_four`, and every further draw buffer must be NONE (4.2.1).
        void expect_draw_buffers(demo::Check& check, std::array<mgl::DrawBuffer, 4> first_four, std::string_view what) const
        {
            const std::vector<GLint> now = query_draw_buffers(initial_.max_draw_buffers);
            bool ok = now.size() >= 4;
            std::string got;
            for (std::size_t i = 0; i < now.size(); ++i)
            {
                got += buffer_name(now[i]) + (i + 1 < now.size() ? ", " : "");
                const GLint want = i < 4 ? value_of(first_four[i]) : GL_NONE;
                if (now[i] != want) ok = false;
            }
            check.expect(ok, what, ok ? std::string{} : "DRAW_BUFFERi = " + got);
        }

        template <typename Call>
        static void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, Call&& call)
        {
            demo::drain_gl_errors();
            call();
            check.expect_gl_error(code, what);
        }

        // ------------------------------------------------------------------------------------------
        // Checks
        // ------------------------------------------------------------------------------------------
        void check_composite(demo::Check& check) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const std::vector<std::uint8_t> window = demo::read_rgba8(0, 0, kWindow, kWindow);
            static constexpr const char* names[4] = {"albedo RGBA8", "data RGBA8", "ids RGBA32UI", "signed RG32I"};
            for (int panel = 0; panel < 4; ++panel)
            {
                const int ox = (panel % 2) * kSize, oy = (1 - panel / 2) * kSize;
                std::vector<std::uint8_t> sub(static_cast<std::size_t>(kSize * kSize * 4));
                for (int y = 0; y < kSize; ++y)
                    for (int x = 0; x < kSize; ++x)
                        for (int c = 0; c < 4; ++c)
                            sub[static_cast<std::size_t>((y * kSize + x) * 4 + c)] =
                                window[static_cast<std::size_t>(((oy + y) * kWindow + ox + x) * 4 + c)];
                const Diff d = compare(sub, kSize, kSize, [&](int x, int y) { return composite_model(ox + x, oy + y); }, 1.0);
                check.expect(d.bad == 0, std::format("window panel {} ({}) shows the attachment: all {} pixels match the CPU model (+-1)", panel, names[panel], kSize * kSize),
                             describe(d));
            }
        }

        void check_initial_state(demo::Check& check) const
        {
            check.expect(initial_.max_draw_buffers >= 4, "MAX_DRAW_BUFFERS >= 4 (ES 3.0 minimum)", std::format("{}", initial_.max_draw_buffers));
            check.expect(initial_.max_color_attachments >= 4, "MAX_COLOR_ATTACHMENTS >= 4 (ES 3.0 minimum)", std::format("{}", initial_.max_color_attachments));
            check.expect(initial_.default_draw0 == GL_BACK && initial_.default_draw1 == GL_NONE,
                         "default framebuffer: DRAW_BUFFER0 = BACK, DRAW_BUFFER1 = NONE initially (4.2.1)",
                         std::format("{}, {}", buffer_name(initial_.default_draw0), buffer_name(initial_.default_draw1)));
            check.expect_eq(buffer_name(initial_.default_read), std::string("BACK"), "default framebuffer: READ_BUFFER = BACK initially (4.3.1)");
            check.expect(initial_.fbo_draw[0] == GL_COLOR_ATTACHMENT0 && initial_.fbo_draw[1] == GL_NONE && initial_.fbo_draw[2] == GL_NONE &&
                             initial_.fbo_draw[3] == GL_NONE,
                         "fresh framebuffer object: DRAW_BUFFER0 = COLOR_ATTACHMENT0, the others NONE (4.2.1)",
                         std::format("{}, {}, {}, {}", buffer_name(initial_.fbo_draw[0]), buffer_name(initial_.fbo_draw[1]),
                                     buffer_name(initial_.fbo_draw[2]), buffer_name(initial_.fbo_draw[3])));
            check.expect_eq(buffer_name(initial_.fbo_read), std::string("COLOR_ATTACHMENT0"), "fresh framebuffer object: READ_BUFFER = COLOR_ATTACHMENT0 (4.3.1)");
            check.expect(initial_.status == mgl::FramebufferStatus::Complete,
                         "framebuffer with RGBA8, RGBA8, RGBA32UI, RG32I and DEPTH24_STENCIL8 attachments is complete");
            check.expect(initial_.component_type[0] == static_cast<GLint>(GL_UNSIGNED_NORMALIZED) && initial_.component_type[1] == static_cast<GLint>(GL_UNSIGNED_NORMALIZED) &&
                             initial_.component_type[2] == static_cast<GLint>(GL_UNSIGNED_INT) && initial_.component_type[3] == static_cast<GLint>(GL_INT),
                         "FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE: UNSIGNED_NORMALIZED, UNSIGNED_NORMALIZED, UNSIGNED_INT, INT (6.1.13)",
                         std::format("{:#x} {:#x} {:#x} {:#x}", initial_.component_type[0], initial_.component_type[1], initial_.component_type[2], initial_.component_type[3]));
        }

        // The draw of render(): four outputs, four buffers.
        void check_mrt_draw(demo::Check& check) const
        {
            bind_fbo();
            expect_draw_buffers(check, {kC0, kC1, kC2, kC3}, "after glDrawBuffers(C0..C3): DRAW_BUFFER0..3 = COLOR_ATTACHMENT0..3, higher ones NONE");
            const std::vector<std::uint8_t> a0 = read_unorm(0), a1 = read_unorm(1);
            const std::vector<std::uint32_t> a2 = read_uint(2);
            const std::vector<std::int32_t> a3 = read_int(3);
            check.expect_no_gl_error("reading all four attachments produced no GL error");
            const Diff d0 = compare_model(a0, albedo_model, 1.0), d1 = compare_model(a1, data_model, 1.0);
            const Diff d2 = compare_model(a2, ids_model, 0.0), d3 = compare_model(a3, signed_model, 0.0);
            check.expect(d0.bad == 0, "attachment 0 (RGBA8): every pixel equals the CPU model of o_albedo (+-1/255, eq. 2.3 rounding)", describe(d0));
            check.expect(d1.bad == 0, "attachment 1 (RGBA8): every pixel equals the CPU model of o_data (+-1/255)", describe(d1));
            check.expect(d2.bad == 0, "attachment 2 (RGBA32UI): every texel equals the uvec4 written, exactly (incl. values above 2^31)", describe(d2));
            check.expect(d3.bad == 0, "attachment 3 (RG32I): (x, y) exact incl. INT_MIN + x; B, A read as 0, 1 (4.3.2.1)", describe(d3));

            // Negative control: the comparison must be able to fail. Each attachment against the model of
            // the *other* attachment of its type must report many differences.
            const Diff n0 = compare_model(a0, data_model, 1.0);
            const Diff n2 = compare(a2, kSize, kSize, [](int x, int y) { Expected e = ids_model(x, y); e[3] += 1.0; return e; }, 0.0);
            check.expect(n0.bad > kSize * kSize / 2, "negative control: attachment 0 compared with the model of attachment 1 differs nearly everywhere",
                         std::format("{} pixels differ", n0.bad));
            check.expect(n2.bad == kSize * kSize, "negative control: an RGBA32UI model off by one in w differs at every pixel (the comparison is exact)",
                         std::format("{} pixels differ", n2.bad));
        }

        // NONE gaps: the output of a draw buffer that is NONE is discarded.
        void check_none_gaps(demo::Check& check) const
        {
            set_sentinels();
            const std::array<mgl::DrawBuffer, 4> even = {kC0, kNone, kC2, kNone};
            mgl::glDrawBuffers(even);
            expect_draw_buffers(check, {kC0, kNone, kC2, kNone}, "glDrawBuffers(C0, NONE, C2, NONE): DRAW_BUFFERi read back, gaps are NONE");
            draw_mrt(kSize);
            const Diff w0 = diff_attachment(0, false), s1 = diff_attachment(1, true), w2 = diff_attachment(2, false), s3 = diff_attachment(3, true);
            check.expect(w0.bad == 0 && w2.bad == 0, "mapping {C0, NONE, C2, NONE}: attachments 0 and 2 receive outputs 0 and 2", describe(w0) + describe(w2));
            check.expect(s1.bad == 0 && s3.bad == 0, "mapping {C0, NONE, C2, NONE}: outputs 1 and 3 are discarded, attachments 1 and 3 keep their sentinel", describe(s1) + describe(s3));

            set_sentinels();
            const std::array<mgl::DrawBuffer, 4> odd = {kNone, kC1, kNone, kC3};
            mgl::glDrawBuffers(odd);
            draw_mrt(kSize);
            const Diff s0 = diff_attachment(0, true), w1 = diff_attachment(1, false), s2 = diff_attachment(2, true), w3 = diff_attachment(3, false);
            check.expect(s0.bad == 0 && s2.bad == 0, "mapping {NONE, C1, NONE, C3}: attachments 0 and 2 keep their sentinel", describe(s0) + describe(s2));
            check.expect(w1.bad == 0 && w3.bad == 0, "mapping {NONE, C1, NONE, C3}: attachments 1 and 3 receive outputs 1 and 3", describe(w1) + describe(w3));
        }

        // A shader with a single output bound to location 1 and glGetFragDataLocation of it.
        void check_single_output(demo::Check& check) const
        {
            set_sentinels();
            const std::array<mgl::DrawBuffer, 4> only1 = {kNone, kC1, kNone, kNone};
            mgl::glDrawBuffers(only1);
            mgl::glViewport(0, 0, kSize, kSize);
            mgl::glUseProgram(single_);
            draw_quad(single_, 0.0f);
            const Diff c1 = compare_constant(read_unorm(1), {204, 102, 51, 255}, 1.0);
            const Diff s0 = diff_attachment(0, true), s2 = diff_attachment(2, true), s3 = diff_attachment(3, true);
            check.expect(c1.bad == 0, "layout(location = 1) output reaches draw buffer 1 = COLOR_ATTACHMENT1 (0.8, 0.4, 0.2, 1.0)", describe(c1));
            check.expect(s0.bad == 0 && s2.bad == 0 && s3.bad == 0, "the other attachments are untouched by the single-output shader", describe(s0) + describe(s2) + describe(s3));
        }

        // n = 0 is equivalent to setting every draw buffer to NONE.
        void check_zero_draw_buffers(demo::Check& check) const
        {
            set_sentinels();
            mgl::glDrawBuffers(std::span<const mgl::DrawBuffer>{});
            expect_draw_buffers(check, {kNone, kNone, kNone, kNone}, "glDrawBuffers with n = 0: every DRAW_BUFFERi = NONE (4.2.1)");
            draw_mrt(kSize);
            const Diff s0 = diff_attachment(0, true), s1 = diff_attachment(1, true), s2 = diff_attachment(2, true), s3 = diff_attachment(3, true);
            check.expect(s0.bad + s1.bad + s2.bad + s3.bad == 0, "with n = 0 a four-output draw writes no color attachment", describe(s0) + describe(s1) + describe(s2) + describe(s3));
        }

        // ES 3.0 4.2.1 errors and the "no side effect on error" rule.
        void check_mapping_errors(demo::Check& check) const
        {
            bind_fbo();
            const std::array<mgl::DrawBuffer, 4> before = {kC0, kC1, kNone, kNone};
            mgl::glDrawBuffers(before);
            const std::array<mgl::DrawBuffer, 2> swapped = {kC1, kC0};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glDrawBuffers(C1, C0): the i-th buffer must be COLOR_ATTACHMENTi or NONE -> INVALID_OPERATION",
                         [&] { mgl::glDrawBuffers(swapped); });
            const std::array<mgl::DrawBuffer, 2> skipped = {kC0, kC2};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glDrawBuffers(C0, C2): C2 in position 1 -> INVALID_OPERATION",
                         [&] { mgl::glDrawBuffers(skipped); });
            const std::array<mgl::DrawBuffer, 1> back = {kBack};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glDrawBuffers(BACK) with a framebuffer object bound -> INVALID_OPERATION",
                         [&] { mgl::glDrawBuffers(back); });
            const std::vector<mgl::DrawBuffer> too_many(static_cast<std::size_t>(initial_.max_draw_buffers) + 1, kNone);
            expect_error(check, mgl::ErrorCode::InvalidValue, "glDrawBuffers with n = MAX_DRAW_BUFFERS + 1 -> INVALID_VALUE",
                         [&] { mgl::glDrawBuffers(too_many); });
            expect_draw_buffers(check, {kC0, kC1, kNone, kNone}, "failed glDrawBuffers calls leave the draw buffer state unchanged");

            // The default framebuffer: n must be 1 and the value BACK or NONE.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const std::array<mgl::DrawBuffer, 1> att0 = {kC0};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "default framebuffer: glDrawBuffers(COLOR_ATTACHMENT0) -> INVALID_OPERATION",
                         [&] { mgl::glDrawBuffers(att0); });
            const std::array<mgl::DrawBuffer, 2> two = {kBack, kNone};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "default framebuffer: glDrawBuffers with n = 2 -> INVALID_OPERATION",
                         [&] { mgl::glDrawBuffers(two); });
            expect_error(check, mgl::ErrorCode::NoError, "default framebuffer: glDrawBuffers(BACK) is valid",
                         [&] { mgl::glDrawBuffers(back); });
            check.expect_eq(demo::get_integer(mgl::GetParameter::DrawBuffer0), GL_BACK, "default framebuffer: DRAW_BUFFER0 is still BACK");

            // glReadBuffer rules (4.3.1).
            expect_error(check, mgl::ErrorCode::InvalidOperation, "default framebuffer: glReadBuffer(COLOR_ATTACHMENT0) -> INVALID_OPERATION",
                         [&] { mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0)); });
            bind_fbo();
            expect_error(check, mgl::ErrorCode::InvalidOperation, "framebuffer object: glReadBuffer(BACK) -> INVALID_OPERATION",
                         [&] { mgl::glReadBuffer(mgl::ReadBuffer::Back); });
            if (initial_.max_color_attachments < 32)
            {
                // The enumerators exist for 32 attachment points; the first one beyond the limit is invalid.
                static constexpr std::array<mgl::ColorAttachment, 32> all_points = {
                    mgl::ColorAttachment::Color0,  mgl::ColorAttachment::Color1,  mgl::ColorAttachment::Color2,  mgl::ColorAttachment::Color3,
                    mgl::ColorAttachment::Color4,  mgl::ColorAttachment::Color5,  mgl::ColorAttachment::Color6,  mgl::ColorAttachment::Color7,
                    mgl::ColorAttachment::Color8,  mgl::ColorAttachment::Color9,  mgl::ColorAttachment::Color10, mgl::ColorAttachment::Color11,
                    mgl::ColorAttachment::Color12, mgl::ColorAttachment::Color13, mgl::ColorAttachment::Color14, mgl::ColorAttachment::Color15,
                    mgl::ColorAttachment::Color16, mgl::ColorAttachment::Color17, mgl::ColorAttachment::Color18, mgl::ColorAttachment::Color19,
                    mgl::ColorAttachment::Color20, mgl::ColorAttachment::Color21, mgl::ColorAttachment::Color22, mgl::ColorAttachment::Color23,
                    mgl::ColorAttachment::Color24, mgl::ColorAttachment::Color25, mgl::ColorAttachment::Color26, mgl::ColorAttachment::Color27,
                    mgl::ColorAttachment::Color28, mgl::ColorAttachment::Color29, mgl::ColorAttachment::Color30, mgl::ColorAttachment::Color31};
                const mgl::ColorAttachment beyond = all_points[static_cast<std::size_t>(initial_.max_color_attachments)];
                mgl::glReadBuffer(mgl::to_read_buffer(kAttach[1]));
                expect_error(check, mgl::ErrorCode::InvalidOperation,
                             std::format("glReadBuffer(COLOR_ATTACHMENT{}) with MAX_COLOR_ATTACHMENTS = {} -> INVALID_OPERATION", initial_.max_color_attachments, initial_.max_color_attachments),
                             [&] { mgl::glReadBuffer(mgl::to_read_buffer(beyond)); });
                check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), value_of(kAttach[1]), "a failed glReadBuffer leaves READ_BUFFER unchanged");
            }
        }

        // Integer color buffers need an integer format; reading a missing buffer is an error (4.3.2.1).
        void check_readback_rules(demo::Check& check) const
        {
            bind_fbo();
            std::array<std::uint32_t, 4> sink{};
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[2]));
            expect_error(check, mgl::ErrorCode::InvalidOperation, "RGBA / UNSIGNED_BYTE from the RGBA32UI attachment -> INVALID_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, sink.data()); });
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[0]));
            expect_error(check, mgl::ErrorCode::InvalidOperation, "RGBA_INTEGER / UNSIGNED_INT from the RGBA8 attachment -> INVALID_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, sink.data()); });
            if (initial_.max_color_attachments > 4)
            {
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color4));
                expect_error(check, mgl::ErrorCode::InvalidOperation, "glReadPixels from COLOR_ATTACHMENT4, which has no image attached -> INVALID_OPERATION",
                             [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, sink.data()); });
            }
            // glReadBuffer(NONE) is valid and is reported by READ_BUFFER.
            expect_error(check, mgl::ErrorCode::NoError, "glReadBuffer(NONE) on a framebuffer object is valid",
                         [&] { mgl::glReadBuffer(mgl::ReadBuffer::None); });
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), GL_NONE, "READ_BUFFER reads back NONE");
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[0]));

            // glReadPixels with READ_BUFFER = NONE generates INVALID_OPERATION (4.3.2.1). This is tested on a
            // separate framebuffer object that has a single RGBA8 attachment and no integer attachment, and NOT
            // on the four-attachment framebuffer above: Mesa 25.0.7 (radeonsi, llvmpipe and zink) crashes with
            // SIGSEGV inside glReadPixels when READ_BUFFER is NONE and ANY attachment of the framebuffer is an
            // integer format (reproduced with COLOR_ATTACHMENT0 = RGBA8, COLOR_ATTACHMENT3 = RG32I).
            mgl::RenderbufferId rb{};
            mgl::FramebufferId plain{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 4, 4);
            mgl::glGenFramebuffers(1, &plain);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, plain);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(kAttach[0]),
                                           mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glReadPixels with READ_BUFFER = NONE -> INVALID_OPERATION (framebuffer without integer attachments)",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, sink.data()); });
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &plain);
            mgl::glDeleteRenderbuffers(1, &rb);
            bind_fbo();
        }

        // glClearBuffer{fv,uiv,iv}: one attachment at a time, in the format of that attachment.
        void check_clears(demo::Check& check) const
        {
            set_sentinels();
            // Clear only draw buffer 2 (RGBA32UI) with glClearBufferuiv: the others keep their sentinel.
            const std::array<GLuint, 4> u2 = {0xFFFFFFFFu, 0x80000000u, 1u, 0xDEADBEEFu};
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 2, u2.data());
            const Diff only2 = compare_constant(read_uint(2), {double(u2[0]), double(u2[1]), double(u2[2]), double(u2[3])}, 0.0);
            const Diff k0 = diff_attachment(0, true), k1 = diff_attachment(1, true), k3 = diff_attachment(3, true);
            check.expect(only2.bad == 0, "glClearBufferuiv(COLOR, 2, {0xFFFFFFFF, 0x80000000, 1, 0xDEADBEEF}): RGBA32UI reads back exactly", describe(only2));
            check.expect(k0.bad + k1.bad + k3.bad == 0, "clearing draw buffer 2 leaves attachments 0, 1 and 3 untouched", describe(k0) + describe(k1) + describe(k3));

            // glClearBufferiv on the RG32I attachment (draw buffer 3).
            const std::array<GLint, 4> i3 = {std::numeric_limits<GLint>::min(), -1, std::numeric_limits<GLint>::max(), -123456789};
            mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 3, i3.data());
            const Diff c3 = compare_constant(read_int(3), {double(i3[0]), double(i3[1]), 0.0, 1.0}, 0.0);
            check.expect(c3.bad == 0, "glClearBufferiv(COLOR, 3, {INT_MIN, -1, ...}): RG32I reads back (INT_MIN, -1, 0, 1)", describe(c3));

            // glClearBufferfv on the normalized attachments: values are clamped to [0,1] and converted (eq. 2.3).
            const std::array<GLfloat, 4> f0 = {0.2f, 0.4f, 0.6f, 0.8f};
            const std::array<GLfloat, 4> f1 = {1.5f, -0.5f, 0.2f, 2.0f};
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, f0.data());
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 1, f1.data());
            const Diff c0 = compare_constant(read_unorm(0), {51, 102, 153, 204}, 1.0);
            const Diff c1 = compare_constant(read_unorm(1), {255, 0, 51, 255}, 1.0);
            check.expect(c0.bad == 0, "glClearBufferfv(COLOR, 0, {0.2, 0.4, 0.6, 0.8}): RGBA8 reads back (51, 102, 153, 204) +-1", describe(c0));
            check.expect(c1.bad == 0, "glClearBufferfv(COLOR, 1, {1.5, -0.5, 0.2, 2.0}): clamped to (255, 0, 51, 255) +-1", describe(c1));

            // A draw buffer that is NONE is not cleared (stated explicitly in ES 3.2 15.2.3.1; in ES 3.0
            // "DRAW_BUFFERi" is the buffer named by the draw buffer state, and NONE names no buffer).
            const std::array<mgl::DrawBuffer, 4> hole = {kC0, kC1, kNone, kC3};
            mgl::glDrawBuffers(hole);
            const std::array<GLuint, 4> other = {5u, 6u, 7u, 8u};
            mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 2, other.data());
            const Diff still2 = compare_constant(read_uint(2), {double(u2[0]), double(u2[1]), double(u2[2]), double(u2[3])}, 0.0);
            check.expect(still2.bad == 0, "glClearBufferuiv on a draw buffer that is NONE has no effect", describe(still2));

            // Errors (ES 3.0 4.2.3.1). INVALID_ENUM cases cannot be formed with meta-gl's typed buffer enums.
            const std::array<GLfloat, 4> zero4 = {};
            const GLint past = initial_.max_draw_buffers;
            expect_error(check, mgl::ErrorCode::InvalidValue, "glClearBufferfv(COLOR, MAX_DRAW_BUFFERS) -> INVALID_VALUE",
                         [&] { mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, past, zero4.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glClearBufferfv(COLOR, -1) -> INVALID_VALUE",
                         [&] { mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, -1, zero4.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glClearBufferfv(DEPTH, 1) -> INVALID_VALUE (drawbuffer must be 0)",
                         [&] { mgl::glClearBufferfv(mgl::FloatClearBuffer::Depth, 1, zero4.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glClearBufferiv(STENCIL, 1) -> INVALID_VALUE (drawbuffer must be 0)",
                         [&] { mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Stencil, 1, &past); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glClearBufferuiv(COLOR, MAX_DRAW_BUFFERS) -> INVALID_VALUE",
                         [&] { mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, past, other.data()); });
        }

        // glClearBufferfi sets depth and stencil in one call. The values cannot be read back in ES, so they
        // are proven through the tests: a quad passes only if its depth is below the cleared depth AND its
        // stencil reference equals the cleared stencil value.
        unsigned run_quadrants(float clear_depth, int clear_stencil) const
        {
            struct Quad
            {
                int vx, vy;       // 64 x 64 viewport (one quadrant of the target)
                float window_z;   // window-space depth of the quad
                int reference;    // stencil reference of this quad
            };
            const std::array<Quad, 4> quads = {{{0, 0, 0.2f, 5}, {64, 0, 0.3f, 5}, {0, 64, 0.2f, 4}, {64, 64, 0.3f, 6}}};

            bind_fbo();
            const std::array<mgl::DrawBuffer, 1> c0_only = {kC0};
            mgl::glDrawBuffers(c0_only);
            const std::array<GLfloat, 4> transparent = {0, 0, 0, 0};
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, transparent.data());
            mgl::glClearBufferfi(clear_depth, clear_stencil);

            mgl::glEnable(mgl::Capability::DepthTest);  // default DepthFunc is LESS
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glUseProgram(mrt_);
            mgl::glUniform2i(mgl::glGetUniformLocation(mrt_, "u_size"), kSize, kSize);
            for (const Quad& q : quads)
            {
                mgl::glViewport(q.vx, q.vy, 64, 64);
                mgl::glStencilFunc(mgl::CompareFunc::Equal, q.reference, 0xFF);
                draw_quad(mrt_, 2.0f * q.window_z - 1.0f);  // window z = (ndc z + 1) / 2 with the default depth range
            }
            mgl::glDisable(mgl::Capability::StencilTest);
            mgl::glDisable(mgl::Capability::DepthTest);

            unsigned mask = 0;
            mgl::glReadBuffer(mgl::to_read_buffer(kAttach[0]));
            for (std::size_t i = 0; i < quads.size(); ++i)
                if (demo::read_pixel(quads[i].vx + 32, quads[i].vy + 32)[3] == 255) mask |= 1u << i;  // the shader writes alpha 1
            return mask;
        }

        static unsigned expected_quadrants(float clear_depth, int clear_stencil)
        {
            const float z[4] = {0.2f, 0.3f, 0.2f, 0.3f};
            const int ref[4] = {5, 5, 4, 6};
            unsigned mask = 0;
            for (int i = 0; i < 4; ++i)
                if (z[i] < clear_depth && ref[i] == clear_stencil) mask |= 1u << i;
            return mask;
        }

        void check_depth_stencil_clear(demo::Check& check) const
        {
            const unsigned a = run_quadrants(0.25f, 5), b = run_quadrants(0.35f, 6), c = run_quadrants(0.35f, 5);
            check.expect(a == expected_quadrants(0.25f, 5), "glClearBufferfi(0.25, 5): only the quad with z = 0.2 and stencil reference 5 passes both tests",
                         std::format("passing quads {:04b}, expected {:04b}", a, expected_quadrants(0.25f, 5)));
            check.expect(b == expected_quadrants(0.35f, 6), "glClearBufferfi(0.35, 6): a different pair of clear values selects a different quad (z = 0.3, reference 6)",
                         std::format("passing quads {:04b}, expected {:04b}", b, expected_quadrants(0.35f, 6)));
            check.expect(c == expected_quadrants(0.35f, 5), "glClearBufferfi(0.35, 5): the z = 0.3 quad now passes the depth test as well",
                         std::format("passing quads {:04b}, expected {:04b}", c, expected_quadrants(0.35f, 5)));
            check.expect(a != b && b != c && a != c, "negative control: the three clear-value pairs produce three different test outcomes");
        }

        void check_frag_data_location(demo::Check& check) const
        {
            check.expect_eq(mgl::glGetFragDataLocation(mrt_, "o_albedo"), 0, "glGetFragDataLocation(o_albedo) = 0 (layout(location = 0))");
            check.expect_eq(mgl::glGetFragDataLocation(mrt_, "o_data"), 1, "glGetFragDataLocation(o_data) = 1");
            check.expect_eq(mgl::glGetFragDataLocation(mrt_, "o_ids"), 2, "glGetFragDataLocation(o_ids) = 2");
            check.expect_eq(mgl::glGetFragDataLocation(mrt_, "o_signed"), 3, "glGetFragDataLocation(o_signed) = 3");
            check.expect_eq(mgl::glGetFragDataLocation(single_, "o_only"), 1, "single-output program: glGetFragDataLocation(o_only) = 1, not the default 0");
            check.expect_eq(mgl::glGetFragDataLocation(mrt_, "o_missing"), -1, "glGetFragDataLocation of a name that is not an output returns -1");
            check.expect_no_gl_error("... and that is not an error");

            // A program that was never linked, and one whose link failed: INVALID_OPERATION and -1 (3.9.2.3).
            const mgl::ProgramId never = mgl::glCreateProgram();
            GLint r = 0;
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glGetFragDataLocation on a program that was never linked -> INVALID_OPERATION",
                         [&] { r = mgl::glGetFragDataLocation(never, "o_albedo"); });
            check.expect_eq(r, -1, "... and returns -1");
            mgl::glDeleteProgram(never);

            std::string log;
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, "#version 300 es\nprecision highp float;\nout vec4 c;\nvoid main(){c=vec4(1.0);}\n", log);
            const mgl::ProgramId failed = mgl::glCreateProgram();
            mgl::glAttachShader(failed, fs);
            mgl::glLinkProgram(failed);  // no vertex shader: the link fails
            GLint linked = GL_TRUE;
            mgl::glGetProgramiv(failed, mgl::ProgramParameter::LinkStatus, &linked);
            check.expect(linked == GL_FALSE, "a program with only a fragment shader fails to link");
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glGetFragDataLocation on a program whose link failed -> INVALID_OPERATION",
                         [&] { r = mgl::glGetFragDataLocation(failed, "c"); });
            mgl::glDetachShader(failed, fs);
            mgl::glDeleteProgram(failed);
            mgl::glDeleteShader(fs);
        }

        // Rules for fragment outputs (GLSL ES 3.00 4.3.8.2: "It is an error if ..."; ES 3.0 3.9.2.3: "Missing or
        // conflicting binding assignments will cause CompileShader to fail").
        //
        // Observed behaviour of Mesa 25.0.7 (radeonsi, llvmpipe, zink): CompileShader accepts a duplicate or
        // out-of-range location and LinkProgram rejects it. The text of ES 3.0 3.9.2.3 asks for a compile-time
        // failure, so the checks below record at which stage the shader was rejected and accept either stage:
        // what the application can rely on is that no usable program results.
        enum class Stage { CompileFailed, LinkFailed, Linked };

        Stage build_with_fragment_body(const std::string& body, std::string& log) const
        {
            const std::string source = "#version 300 es\nprecision highp float;\n" + body;
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, source, log);
            if (fs.value == 0) return Stage::CompileFailed;
            std::string vlog;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, vertex_source_, vlog);
            const mgl::ShaderId both[] = {vs, fs};
            const mgl::ProgramId program = demo::link_program(both, log);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (program.value == 0) return Stage::LinkFailed;
            mgl::glDeleteProgram(program);
            return Stage::Linked;
        }

        void check_output_rules(demo::Check& check) const
        {
            auto name = [](Stage s) { return s == Stage::CompileFailed ? "CompileShader fails" : s == Stage::LinkFailed ? "LinkProgram fails" : "links"; };
            auto accepted = [&](std::string_view what, const std::string& body) {
                std::string log;
                const Stage s = build_with_fragment_body(body, log);
                check.expect(s == Stage::Linked, what, s == Stage::Linked ? std::string{} : std::format("{}: {}", name(s), log));
            };
            auto rejected = [&](std::string_view what, const std::string& body) {
                std::string log;
                const Stage s = build_with_fragment_body(body, log);
                const std::size_t eol = log.find('\n');
                check.expect(s != Stage::Linked, what, std::format("{}: {}", name(s), log.substr(0, eol)));
            };
            accepted("two outputs with locations 0 and 1 compile and link",
                     "layout(location = 0) out vec4 a; layout(location = 1) out vec4 b; void main(){a = vec4(1.0); b = vec4(0.0);}");
            accepted("outputs may be declared out of order and leave gaps (locations 2 and 0)",
                     "layout(location = 2) out vec4 a; layout(location = 0) out vec4 b; void main(){a = vec4(1.0); b = vec4(0.0);}");
            // NOT checked: "more than one output without explicit locations is an error" (GLSL ES 3.00 4.3.8.2,
            // ES 3.0 3.9.2.3). Mesa 25.0.7 compiles AND links such shaders and assigns the locations itself, as
            // desktop GL does. See the README ("Known driver deviation").
            rejected("two outputs bound to the same location are rejected",
                     "layout(location = 1) out vec4 a; layout(location = 1) out vec4 b; void main(){a = vec4(1.0); b = vec4(0.0);}");
            rejected("an output at location MAX_DRAW_BUFFERS is rejected",
                     std::format("layout(location = {}) out vec4 a; void main(){{a = vec4(1.0);}}", initial_.max_draw_buffers));
            accepted("an output at location MAX_DRAW_BUFFERS - 1 is accepted",
                     std::format("layout(location = {}) out vec4 a; void main(){{a = vec4(1.0);}}", initial_.max_draw_buffers - 1));
        }

        struct InitialState
        {
            GLint default_draw0 = 0, default_draw1 = 0, default_read = 0;
            GLint max_draw_buffers = 0, max_color_attachments = 0;
            std::array<GLint, 4> fbo_draw{};
            GLint fbo_read = 0;
            std::array<GLint, 4> component_type{};
            mgl::FramebufferStatus status = mgl::FramebufferStatus::Undefined;
        };

        std::string vertex_source_;
        mgl::ProgramId mrt_{}, single_{}, composite_{};
        std::array<mgl::TextureId, kAttachments> tex_{};
        mgl::RenderbufferId depth_stencil_{};
        mgl::FramebufferId fbo_{};
        InitialState initial_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-multiple-render-targets",
        .title = "Multiple render targets: glDrawBuffers, glClearBuffer*, integer targets",
        .min_es = {3, 0},
        .summary = "One framebuffer object with RGBA8, RGBA32UI and RG32I color attachments written by one GLSL ES 3.00 shader; "
                   "glDrawBuffers mappings with NONE gaps, per-attachment glClearBuffer*, integer readback, glGetFragDataLocation.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<MultipleRenderTargets>(argc, argv, info);
}
