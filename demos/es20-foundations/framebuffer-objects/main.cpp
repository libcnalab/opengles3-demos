// es20-framebuffer-objects
//
// Framebuffer objects (OpenGL ES 3.0 section 4.4): off-screen rendering into textures and
// renderbuffers, depth and stencil attachments, framebuffer completeness (including the failures the
// spec defines), INVALID_FRAMEBUFFER_OPERATION, and the object queries and lifecycle rules.
//
// The default framebuffer is divided into a grid of cells (4 x 3, top row first, at the default size; 3 x 4,
// 6 x 2 or 2 x 6 when that gives larger cells). Every cell shows an off-screen 64 x 64 render target,
// centred and magnified by the largest integer factor that fits, brought to the screen either by sampling
// its texture in a second pass (render to texture) or, for renderbuffers that cannot be sampled, by
// glBlitFramebuffer.
//
//   row 0: RTT gradient+checker | negative of it (texture -> texture) | mip level 2 (4x) | stencil (S8)
//   row 1: depth test, D16      | depth test, D24                      | D16, test OFF   | stencil OFF
//   row 2: DEPTH24_STENCIL8     | depth *texture*: color               | depth texture as gray
//
// Validation: every target is read back with glReadPixels from its own framebuffer and compared with
// a CPU model (an analytic image for the pattern, a small CPU depth/stencil pipeline for the
// scenes); the cell on screen is compared with the read-back; the "OFF" cells are controls that must
// differ from the "ON" cells. Completeness, errors, queries and the object lifecycle are checked with
// a scratch framebuffer.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kSize = 64;      // edge of every off-screen target
    constexpr int kMipLevel = 2;   // attached mipmap level of the "mip" target (64 -> 32 -> 16)
    constexpr int kMipSize = kSize >> kMipLevel;

    using Rgbaf = std::array<float, 4>;
    constexpr Rgbaf kBackground = {0.2f, 0.2f, 0.2f, 1.0f};  // clear color of every scene
    constexpr Rgbaf kGreen = {0.2f, 0.8f, 0.2f, 1.0f};
    constexpr Rgbaf kRed = {0.8f, 0.2f, 0.2f, 1.0f};
    constexpr Rgbaf kBlue = {0.2f, 0.4f, 0.8f, 1.0f};
    constexpr Rgbaf kYellow = {0.8f, 0.8f, 0.2f, 1.0f};
    constexpr Rgbaf kCyan = {0.2f, 0.8f, 0.8f, 1.0f};
    constexpr Rgbaf kMagenta = {0.8f, 0.2f, 0.8f, 1.0f};
    constexpr Rgbaf kWhite = {1.0f, 1.0f, 1.0f, 1.0f};

    // Tolerances, in 8-bit steps. A float -> 8-bit conversion (equation 2.3, section 2.1.6.2) returns
    // "one of the two integers closest to f * 255, rounding to nearest preferred", so a result that is
    // exactly k + 0.5 or a float that rounds differently on the GPU and on the CPU is off by at most 1.
    constexpr int kConvertTolerance = 1;
    // glBlitFramebuffer between two buffers of the same 8-bit format with NEAREST copies stored values: with an
    // integer magnification m, destination pixel i maps to source position (i + 1/2) / m, which lies inside
    // source pixel i / m and at least 1/(2m) away from its edges (section 4.3.3), so no value is interpolated.
    constexpr int kBlitTolerance = 0;

    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    constexpr int gl(GLenum e) { return static_cast<int>(e); }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    Rgba8 to_rgba8(const Rgbaf& c)
    {
        return {to8(static_cast<double>(c[0])), to8(static_cast<double>(c[1])), to8(static_cast<double>(c[2])),
                to8(static_cast<double>(c[3]))};
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference of the first pass (pattern.frag): pixel (i, j) of a size x size target.
    // ---------------------------------------------------------------------------------------------
    Rgba8 pattern_texel(int i, int j, int size, double divisor_override = 0.0)
    {
        const double d = divisor_override > 0.0 ? divisor_override : static_cast<double>(size - 1);
        const bool odd = (((i >> 3) ^ (j >> 3)) & 1) != 0;
        return {to8(i / d), to8(j / d), to8(odd ? 0.8 : 0.2), 255};
    }

    std::vector<Rgba8> pattern_image(int size)
    {
        std::vector<Rgba8> img;
        img.reserve(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
        for (int j = 0; j < size; ++j)
            for (int i = 0; i < size; ++i) img.push_back(pattern_texel(i, j, size));
        return img;
    }

    std::vector<Rgba8> to_texels(std::span<const std::uint8_t> bytes)
    {
        std::vector<Rgba8> out(bytes.size() / 4);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = {bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]};
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // A scene is a list of axis-aligned rectangles at a constant depth with per-rectangle depth and
    // stencil state. The same list drives the GL calls and a CPU model of the per-fragment pipeline
    // (ES 3.0 sections 4.1.4 stencil test, 4.1.5 depth test, 4.2.2 masking), so the two cannot drift.
    // Rectangle edges lie on pixel boundaries: a pixel is covered when its center is inside, and a
    // center is never on an integer coordinate, so rasterization has no ambiguity.
    // ---------------------------------------------------------------------------------------------
    struct Rect
    {
        int x0, y0, x1, y1;  // pixels, [x0, x1) x [y0, y1)
    };

    struct DrawCmd
    {
        Rect rect;
        float z;  // NDC depth; the window depth is (z + 1) / 2
        Rgbaf color;
        bool color_write = true;
        bool depth_test = false;  // DEPTH_TEST enabled, DepthFunc LESS
        bool stencil_test = false;
        mgl::CompareFunc func = mgl::CompareFunc::Always;
        int ref = 0;
        mgl::StencilOp sfail = mgl::StencilOp::Keep;
        mgl::StencilOp dpfail = mgl::StencilOp::Keep;
        mgl::StencilOp dppass = mgl::StencilOp::Keep;
    };
    using Scene = std::vector<DrawCmd>;

    struct SceneImage
    {
        std::vector<Rgba8> color;
        std::vector<float> depth;  // window-space depth, 1.0 = far (the clear value)
        std::vector<std::uint8_t> stencil;
    };

    bool stencil_passes(mgl::CompareFunc f, int ref, int stored)
    {
        // Section 4.1.4: the test passes when "ref func stored" holds (both masked; the mask is 0xFF).
        switch (f)
        {
        case mgl::CompareFunc::Never: return false;
        case mgl::CompareFunc::Less: return ref < stored;
        case mgl::CompareFunc::Equal: return ref == stored;
        case mgl::CompareFunc::Lequal: return ref <= stored;
        case mgl::CompareFunc::Greater: return ref > stored;
        case mgl::CompareFunc::Notequal: return ref != stored;
        case mgl::CompareFunc::Gequal: return ref >= stored;
        case mgl::CompareFunc::Always: return true;
        }
        return false;
    }

    int stencil_update(mgl::StencilOp op, int stored, int ref)
    {
        switch (op)
        {
        case mgl::StencilOp::Keep: return stored;
        case mgl::StencilOp::Zero: return 0;
        case mgl::StencilOp::Replace: return ref;
        case mgl::StencilOp::Incr: return std::min(stored + 1, 255);
        case mgl::StencilOp::Decr: return std::max(stored - 1, 0);
        case mgl::StencilOp::Invert: return ~stored & 0xFF;
        case mgl::StencilOp::IncrWrap: return (stored + 1) & 0xFF;
        case mgl::StencilOp::DecrWrap: return (stored - 1) & 0xFF;
        }
        return stored;
    }

    SceneImage simulate(const Scene& scene)
    {
        const std::size_t n = static_cast<std::size_t>(kSize) * static_cast<std::size_t>(kSize);
        SceneImage img{std::vector<Rgba8>(n, to_rgba8(kBackground)), std::vector<float>(n, 1.0f), std::vector<std::uint8_t>(n, 0)};
        for (const DrawCmd& c : scene)
            for (int y = c.rect.y0; y < c.rect.y1; ++y)
                for (int x = c.rect.x0; x < c.rect.x1; ++x)
                {
                    const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) + static_cast<std::size_t>(x);
                    const int stored = img.stencil[i];
                    const bool s_ok = !c.stencil_test || stencil_passes(c.func, c.ref & 0xFF, stored);
                    const float zw = (c.z + 1.0f) * 0.5f;
                    // A disabled depth test is bypassed: it passes and the depth buffer is not written.
                    const bool d_ok = s_ok && (!c.depth_test || zw < img.depth[i]);
                    if (c.stencil_test)
                    {
                        const mgl::StencilOp op = !s_ok ? c.sfail : (!d_ok ? c.dpfail : c.dppass);
                        img.stencil[i] = static_cast<std::uint8_t>(stencil_update(op, stored, c.ref & 0xFF));
                    }
                    if (s_ok && d_ok)
                    {
                        if (c.depth_test) img.depth[i] = zw;
                        if (c.color_write) img.color[i] = to_rgba8(c.color);
                    }
                }
        return img;
    }

    // Near green quad first, far red quad second (they overlap), then a blue stripe at middle depth.
    // With a working depth buffer the far quad must NOT overwrite the near one although it is drawn
    // later, and the stripe is only visible where it is in front of what is already there.
    Scene depth_scene(bool depth_test)
    {
        return {
            {.rect = {8, 8, 40, 56}, .z = -0.5f, .color = kGreen, .depth_test = depth_test},
            {.rect = {24, 16, 56, 48}, .z = 0.5f, .color = kRed, .depth_test = depth_test},
            {.rect = {16, 28, 48, 36}, .z = 0.0f, .color = kBlue, .depth_test = depth_test},
        };
    }

    // Stencil as a counter and a mask: write 1 into a square, increment a smaller square to 2, then
    // paint the whole target three times, each time only where the stencil value is 0, 1 or 2.
    Scene stencil_scene(bool stencil_test)
    {
        const Rect all{0, 0, kSize, kSize};
        using Op = mgl::StencilOp;
        using Fn = mgl::CompareFunc;
        return {
            {.rect = {16, 16, 48, 48}, .z = 0.0f, .color = kWhite, .color_write = false, .stencil_test = stencil_test,
             .func = Fn::Always, .ref = 1, .sfail = Op::Keep, .dpfail = Op::Keep, .dppass = Op::Replace},
            {.rect = {24, 24, 40, 40}, .z = 0.0f, .color = kWhite, .color_write = false, .stencil_test = stencil_test,
             .func = Fn::Always, .ref = 0, .sfail = Op::Keep, .dpfail = Op::Keep, .dppass = Op::Incr},
            {.rect = all, .z = 0.0f, .color = kBlue, .stencil_test = stencil_test, .func = Fn::Equal, .ref = 0},
            {.rect = all, .z = 0.0f, .color = kYellow, .stencil_test = stencil_test, .func = Fn::Equal, .ref = 1},
            {.rect = all, .z = 0.0f, .color = kRed, .stencil_test = stencil_test, .func = Fn::Equal, .ref = 2},
        };
    }

    // One DEPTH24_STENCIL8 image has to serve both tests. Two overlapping quads A (near) and B (far)
    // are drawn twice: in the upper half with colors (the depth test resolves the overlap), in the
    // lower half without color but with stencil operations that record what the depth test decided:
    //   A: depth pass -> REPLACE with 1                      B: depth fail -> INCR, depth pass -> REPLACE with 3
    // so the stencil value is 1 where only A is, 2 where B lost against A, 3 where only B is. Three
    // full-width bands, each painted where the stencil value equals 1, 2 or 3, make the values visible.
    Scene packed_scene()
    {
        using Op = mgl::StencilOp;
        using Fn = mgl::CompareFunc;
        auto a = [](int y0, int y1, bool color) {
            return DrawCmd{.rect = {4, y0, 40, y1}, .z = -0.5f, .color = kGreen, .color_write = color, .depth_test = true,
                           .stencil_test = true, .func = Fn::Always, .ref = 1,
                           .sfail = Op::Keep, .dpfail = Op::Keep, .dppass = Op::Replace};
        };
        auto b = [](int y0, int y1, bool color) {
            return DrawCmd{.rect = {24, y0, 60, y1}, .z = 0.5f, .color = kRed, .color_write = color, .depth_test = true,
                           .stencil_test = true, .func = Fn::Always, .ref = 3,
                           .sfail = Op::Keep, .dpfail = Op::Incr, .dppass = Op::Replace};
        };
        auto band = [](int ref, const Rgbaf& color) {
            return DrawCmd{.rect = {0, 4, kSize, 28}, .z = 0.0f, .color = color, .stencil_test = true, .func = Fn::Equal, .ref = ref};
        };
        return {a(36, 60, true), b(36, 60, true), a(4, 28, false), b(4, 28, false),
                band(1, kCyan), band(2, kMagenta), band(3, kYellow)};
    }

    // ---------------------------------------------------------------------------------------------
    // Image comparison.
    // ---------------------------------------------------------------------------------------------
    struct Diff
    {
        int bad = 0;
        int max_dev = 0;
        std::string first;
    };

    Diff compare(std::span<const Rgba8> got, std::span<const Rgba8> want, int width, int tolerance)
    {
        Diff d;
        for (std::size_t i = 0; i < want.size() && i < got.size(); ++i)
        {
            int dev = 0;
            for (std::size_t k = 0; k < 4; ++k) dev = std::max(dev, std::abs(int{got[i][k]} - int{want[i][k]}));
            d.max_dev = std::max(d.max_dev, dev);
            if (dev > tolerance && d.bad++ == 0)
                d.first = std::format("first at ({},{}): got {} want {}", static_cast<int>(i) % width, static_cast<int>(i) / width,
                                      demo::format_rgba(got[i]), demo::format_rgba(want[i]));
        }
        if (got.size() != want.size()) d.bad += 1000000;
        return d;
    }

    std::vector<Rgba8> read_fbo(mgl::FramebufferId fbo, int size)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo);
        const auto bytes = demo::read_rgba8(0, 0, size, size);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        return to_texels(bytes);
    }

    Rgba8 read_fbo_pixel(mgl::FramebufferId fbo, int x, int y)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo);
        const Rgba8 p = demo::read_pixel(x, y);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        return p;
    }

    // Cells: index = 4 * row_from_top + column.
    enum Cell : int
    {
        kCellRtt, kCellPost, kCellMip, kCellStencil,
        kCellDepth16, kCellDepth24, kCellDepth16Off, kCellStencilOff,
        kCellPacked, kCellDepthTexColor, kCellDepthTexGray,
        kCellCount
    };
    // Screen layout: g_columns x g_rows cells partition the g_width x g_height surface (chosen in init()).
    int g_columns = 4, g_rows = 3, g_width = 256, g_height = 256;

    /// Where a 64 x 64 target is shown: centred in its cell, magnified by the integer factor `scale` (>= 1;
    /// cells of 128 pixels or more keep a gutter of 4 pixels between the images).
    struct Display
    {
        int x, y, scale;
        [[nodiscard]] int size() const { return kSize * scale; }
    };
    Display display(int cell)
    {
        const int col = cell % g_columns, row = cell / g_columns;  // row 0 of the grid is at the top
        const int x0 = g_width * col / g_columns, x1 = g_width * (col + 1) / g_columns;
        const int y0 = g_height * (g_rows - 1 - row) / g_rows, y1 = g_height * (g_rows - row) / g_rows;
        const int gutter = std::min(x1 - x0, y1 - y0) >= 128 ? 4 : 0;  // between neighbouring images, in cells of 128+ pixels
        const int scale = std::max(1, (std::min(x1 - x0, y1 - y0) - gutter) / kSize);
        return {x0 + (x1 - x0 - kSize * scale) / 2, y0 + (y1 - y0 - kSize * scale) / 2, scale};
    }

    std::vector<Rgba8> read_cell(int cell)
    {
        const Display d = display(cell);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        return to_texels(demo::read_rgba8(d.x, d.y, d.size(), d.size()));
    }

    /// The window pixel at the center of the block that shows target texel (x, y).
    Rgba8 read_cell_pixel(int cell, int x, int y)
    {
        const Display d = display(cell);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        return demo::read_pixel(d.x + x * d.scale + d.scale / 2, d.y + y * d.scale + d.scale / 2);
    }

    std::vector<Rgba8> magnify(std::span<const Rgba8> img, int size, int factor)
    {
        std::vector<Rgba8> out;
        for (int y = 0; y < size * factor; ++y)
            for (int x = 0; x < size * factor; ++x)
                out.push_back(img[static_cast<std::size_t>(y / factor) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x / factor)]);
        return out;
    }

    template <typename F>
    void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // start from an empty queue so the error belongs to this call
        call();
        check.expect_gl_error(code, what);
    }

    mgl::FramebufferStatus status_of_bound()
    {
        return mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
    }

    bool expect_status(demo::Check& check, mgl::FramebufferStatus want, std::string_view what)
    {
        const mgl::FramebufferStatus got = status_of_bound();
        return check.expect(got == want, what,
                            std::format("glCheckFramebufferStatus = {}, expected {}", demo::framebuffer_status_name(got),
                                        demo::framebuffer_status_name(want)));
    }

    int attachment_param(mgl::FramebufferAttachment attachment, mgl::FramebufferAttachmentParameter pname)
    {
        GLint v = -12345;
        mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, attachment, pname, &v);
        return v;
    }

    int renderbuffer_param(mgl::RenderbufferParameter pname)
    {
        GLint v = -12345;
        mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, pname, &v);
        return v;
    }

    struct Locations
    {
        mgl::UniformLocation rect, z, color, size, tex, lod, lod_size, invert, gray;
    };

    Locations locate(mgl::ProgramId program)
    {
        return {mgl::glGetUniformLocation(program, "u_rect"),     mgl::glGetUniformLocation(program, "u_z"),
                mgl::glGetUniformLocation(program, "u_color"),    mgl::glGetUniformLocation(program, "u_size"),
                mgl::glGetUniformLocation(program, "u_tex"),      mgl::glGetUniformLocation(program, "u_lod"),
                mgl::glGetUniformLocation(program, "u_lod_size"), mgl::glGetUniformLocation(program, "u_invert"),
                mgl::glGetUniformLocation(program, "u_gray")};
    }

    float ndc(int pixel, int size) { return static_cast<float>(pixel) * 2.0f / static_cast<float>(size) - 1.0f; }

    class FramebufferObjects final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // Arrange the 11 cells so that the smallest cell edge is as large as possible; a cell must show its
            // 64 x 64 target at least 1:1.
            int best = -1;
            for (const auto& [columns, rows] : {std::pair{4, 3}, std::pair{3, 4}, std::pair{6, 2}, std::pair{2, 6}})
            {
                if (columns * rows < kCellCount) continue;
                const int edge = std::min(env.width / columns, env.height / rows);
                if (edge > best)
                {
                    best = edge;
                    g_columns = columns;
                    g_rows = rows;
                }
            }
            if (best < kSize)
                throw demo::Skip(std::format("needs a surface whose cells are at least {0}x{0} pixels: 256x192, 192x256, 384x128 or "
                                             "128x384 (got {1}x{2})", kSize, env.width, env.height));
            g_width = env.width;
            g_height = env.height;
            if (demo::get_integer(mgl::GetParameter::RedBits) < 8 || demo::get_integer(mgl::GetParameter::GreenBits) < 8 ||
                demo::get_integer(mgl::GetParameter::BlueBits) < 8)
                throw demo::Skip("needs 8 bits per color channel in the default framebuffer");

            pattern_ = demo::build_program(env, "shaders/quad.vert", "shaders/pattern.frag");
            flat_ = demo::build_program(env, "shaders/quad.vert", "shaders/flat.frag");
            show_ = demo::build_program(env, "shaders/quad.vert", "shaders/show.frag");
            pattern_u_ = locate(pattern_);
            flat_u_ = locate(flat_);
            show_u_ = locate(show_);

            max_renderbuffer_size_ = demo::get_integer(mgl::GetParameter::MaxRenderbufferSize);
            max_samples_ = demo::get_integer(mgl::GetParameter::MaxSamples);

            // --- textures: color targets (RGBA8), the mip chain, and a depth texture ----------------
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            for (const std::size_t t : {kTexRtt, kTexPost, kTexStencil, kTexStencilOff, kTexPacked, kTexDepthColor})
                specify_texture(tex_[t], 0, kSize, mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, GL_NEAREST);
            // A texture is only a valid attachment at a level other than its base level when it is mipmap
            // complete (section 4.4.4.1), so define the whole chain 64, 32, 16, 8, 4, 2, 1.
            for (int level = 0; (kSize >> level) >= 1; ++level)
                specify_texture(tex_[kTexMip], level, kSize >> level, mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba,
                                mgl::PixelType::UnsignedByte, GL_NEAREST_MIPMAP_NEAREST);
            // Depth textures: DEPTH_COMPONENT / UNSIGNED_INT -> DEPTH_COMPONENT24 (Table 3.2/3.3).
            specify_texture(tex_[kTexDepth], 0, kSize, mgl::InternalFormat::DepthComponent24, mgl::PixelFormat::DepthComponent,
                            mgl::PixelType::UnsignedInt, GL_NEAREST);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            // --- renderbuffers ---------------------------------------------------------------------
            mgl::glGenRenderbuffers(static_cast<GLsizei>(rb_.size()), rb_.data());
            rb_storage(rb_[kRbColorD16], mgl::InternalFormat::Rgba8, kSize, kSize);
            rb_storage(rb_[kRbColorD24], mgl::InternalFormat::Rgba8, kSize, kSize);
            rb_storage(rb_[kRbColorD16Off], mgl::InternalFormat::Rgba8, kSize, kSize);
            rb_storage(rb_[kRbD16], mgl::InternalFormat::DepthComponent16, kSize, kSize);
            rb_storage(rb_[kRbD24], mgl::InternalFormat::DepthComponent24, kSize, kSize);
            rb_storage(rb_[kRbS8], mgl::InternalFormat::StencilIndex8, kSize, kSize);
            rb_storage(rb_[kRbD24S8], mgl::InternalFormat::Depth24Stencil8, kSize, kSize);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});

            // --- framebuffer objects ---------------------------------------------------------------
            mgl::glGenFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            using A = mgl::FramebufferAttachment;
            begin_fbo(kFboRtt);                                   // texture color only
            attach_texture(kColor0, tex_[kTexRtt], 0);
            end_fbo(kFboRtt, "render-to-texture");
            begin_fbo(kFboPost);
            attach_texture(kColor0, tex_[kTexPost], 0);
            end_fbo(kFboPost, "post-process");
            begin_fbo(kFboMip);                                   // a non-zero mipmap level of a texture
            attach_texture(kColor0, tex_[kTexMip], kMipLevel);
            end_fbo(kFboMip, "mip level");
            begin_fbo(kFboD16);                                   // color renderbuffer + depth renderbuffer
            attach_renderbuffer(kColor0, rb_[kRbColorD16]);
            attach_renderbuffer(A::Depth, rb_[kRbD16]);
            end_fbo(kFboD16, "RGBA8 + DEPTH_COMPONENT16 renderbuffers");
            begin_fbo(kFboD24);
            attach_renderbuffer(kColor0, rb_[kRbColorD24]);
            attach_renderbuffer(A::Depth, rb_[kRbD24]);
            end_fbo(kFboD24, "RGBA8 + DEPTH_COMPONENT24 renderbuffers");
            begin_fbo(kFboD16Off);                                // shares the depth renderbuffer of kFboD16
            attach_renderbuffer(kColor0, rb_[kRbColorD16Off]);
            attach_renderbuffer(A::Depth, rb_[kRbD16]);
            end_fbo(kFboD16Off, "RGBA8 renderbuffer + shared depth renderbuffer");
            begin_fbo(kFboStencil);                               // color texture + stencil renderbuffer
            attach_texture(kColor0, tex_[kTexStencil], 0);
            attach_renderbuffer(A::Stencil, rb_[kRbS8]);
            end_fbo(kFboStencil, "RGBA8 texture + STENCIL_INDEX8 renderbuffer");
            begin_fbo(kFboStencilOff);
            attach_texture(kColor0, tex_[kTexStencilOff], 0);
            attach_renderbuffer(A::Stencil, rb_[kRbS8]);
            end_fbo(kFboStencilOff, "RGBA8 texture + shared stencil renderbuffer");
            begin_fbo(kFboPacked);                                // one packed image for depth and stencil
            attach_texture(kColor0, tex_[kTexPacked], 0);
            attach_renderbuffer(A::DepthStencil, rb_[kRbD24S8]);
            end_fbo(kFboPacked, "RGBA8 texture + DEPTH24_STENCIL8 renderbuffer");
            begin_fbo(kFboDepthTex);                              // a depth *texture* as the depth attachment
            attach_texture(kColor0, tex_[kTexDepthColor], 0);
            attach_texture(A::Depth, tex_[kTexDepth], 0);
            end_fbo(kFboDepthTex, "RGBA8 texture + DEPTH_COMPONENT24 texture");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            // Dithering may change the stored color by one step; pixel comparisons need it off.
            mgl::glDisable(mgl::Capability::Dither);

            // ---- pass 1: render into texture images -------------------------------------------------
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});  // nothing sampled while attached
            bind_target(kFboRtt, kSize);
            mgl::glUseProgram(pattern_);
            mgl::glUniform1f(pattern_u_.size, static_cast<float>(kSize));
            draw_quad(pattern_u_, -1.0f, -1.0f, 1.0f, 1.0f, 0.0f);

            // ---- pass 2 (texture -> texture): sample pass 1 and render its negative into another texture
            bind_target(kFboPost, kSize);
            show(tex_[kTexRtt], /*lod=*/-1, 0.0f, /*invert=*/1.0f, /*gray=*/0.0f);

            // ---- a mipmap level is an attachable image: render a 16x16 pattern into level 2 ----------
            bind_target(kFboMip, kMipSize);
            mgl::glUseProgram(pattern_);
            mgl::glUniform1f(pattern_u_.size, static_cast<float>(kMipSize));
            draw_quad(pattern_u_, -1.0f, -1.0f, 1.0f, 1.0f, 0.0f);

            // ---- renderbuffer / depth / stencil scenes ----------------------------------------------
            bind_target(kFboD16, kSize);
            run_scene(depth_scene(true), /*depth=*/true, /*stencil=*/false);
            bind_target(kFboD24, kSize);
            run_scene(depth_scene(true), true, false);
            bind_target(kFboD16Off, kSize);
            run_scene(depth_scene(false), true, false);  // control: same draws, DEPTH_TEST disabled
            bind_target(kFboStencil, kSize);
            run_scene(stencil_scene(true), false, true);
            bind_target(kFboStencilOff, kSize);
            run_scene(stencil_scene(false), false, true);  // control: STENCIL_TEST disabled
            bind_target(kFboPacked, kSize);
            run_scene(packed_scene(), true, true);
            bind_target(kFboDepthTex, kSize);
            run_scene(depth_scene(true), true, false);

            // ---- the default framebuffer: show every target -----------------------------------------
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            show_cell(kCellRtt, tex_[kTexRtt], -1, 0.0f, 0.0f, 0.0f);
            show_cell(kCellPost, tex_[kTexPost], -1, 0.0f, 0.0f, 0.0f);
            show_cell(kCellMip, tex_[kTexMip], kMipLevel, static_cast<float>(kMipSize), 0.0f, 0.0f);
            show_cell(kCellStencil, tex_[kTexStencil], -1, 0.0f, 0.0f, 0.0f);
            blit_cell(kCellDepth16, kFboD16);
            blit_cell(kCellDepth24, kFboD24);
            blit_cell(kCellDepth16Off, kFboD16Off);
            show_cell(kCellStencilOff, tex_[kTexStencilOff], -1, 0.0f, 0.0f, 0.0f);
            show_cell(kCellPacked, tex_[kTexPacked], -1, 0.0f, 0.0f, 0.0f);
            show_cell(kCellDepthTexColor, tex_[kTexDepthColor], -1, 0.0f, 0.0f, 0.0f);
            show_cell(kCellDepthTexGray, tex_[kTexDepth], -1, 0.0f, 0.0f, /*gray=*/1.0f);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            (void)env;
            check_render_to_texture(check);
            check_scenes(check);
            check_attachment_queries(check);
            check_renderbuffer_queries(check);
            check_completeness(check);
            check_invalid_framebuffer_operation(check);
            check_lifecycle(check);
            check_attachment_errors(check);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation sequence");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            mgl::glDeleteRenderbuffers(static_cast<GLsizei>(rb_.size()), rb_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(pattern_);
            mgl::glDeleteProgram(flat_);
            mgl::glDeleteProgram(show_);
        }

    private:
        enum : std::size_t { kTexRtt, kTexPost, kTexMip, kTexStencil, kTexStencilOff, kTexPacked, kTexDepthColor, kTexDepth, kTexCount };
        enum : std::size_t { kRbColorD16, kRbColorD24, kRbColorD16Off, kRbD16, kRbD24, kRbS8, kRbD24S8, kRbCount };
        enum : std::size_t {
            kFboRtt, kFboPost, kFboMip, kFboD16, kFboD24, kFboD16Off, kFboStencil, kFboStencilOff, kFboPacked, kFboDepthTex, kFboCount
        };

        // ---- creation helpers (every call visible) --------------------------------------------------
        static void specify_texture(mgl::TextureId tex, int level, int size, mgl::InternalFormat internal, mgl::PixelFormat format,
                                    mgl::PixelType type, GLenum min_filter)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            // data == nullptr: storage is allocated, the texels are undefined until rendered into.
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, internal, size, size, 0, format, type, nullptr);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, static_cast<GLint>(min_filter));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, GL_NEAREST);
        }

        static void rb_storage(mgl::RenderbufferId rb, mgl::InternalFormat format, int w, int h)
        {
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, format, w, h);
        }

        void begin_fbo(std::size_t which) { mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[which]); }

        static void attach_texture(mgl::FramebufferAttachment attachment, mgl::TextureId tex, int level)
        {
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, attachment, mgl::TextureTarget::Texture2D, tex, level);
        }

        static void attach_renderbuffer(mgl::FramebufferAttachment attachment, mgl::RenderbufferId rb)
        {
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, attachment, mgl::RenderbufferTarget::Renderbuffer, rb);
        }

        void end_fbo(std::size_t /*which*/, const char* what)
        {
            const mgl::FramebufferStatus status = status_of_bound();
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer '{}' is {} - rendering would generate INVALID_FRAMEBUFFER_OPERATION", what,
                                                demo::framebuffer_status_name(status)));
        }

        // ---- drawing helpers -------------------------------------------------------------------------
        void bind_target(std::size_t which, int size)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[which]);
            mgl::glViewport(0, 0, size, size);
        }

        static void draw_quad(const Locations& u, float x0, float y0, float x1, float y1, float z)
        {
            mgl::glUniform4f(u.rect, x0, y0, x1, y1);
            mgl::glUniform1f(u.z, z);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        // Samples `tex` (unit 0) over the whole current viewport.
        void show(mgl::TextureId tex, int lod, float lod_size, float invert, float gray)
        {
            mgl::glUseProgram(show_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glUniform1i(show_u_.tex, 0);  // the sampler uniform holds a texture unit index
            mgl::glUniform1i(show_u_.lod, lod);
            mgl::glUniform1f(show_u_.lod_size, lod_size);
            mgl::glUniform1f(show_u_.invert, invert);
            mgl::glUniform1f(show_u_.gray, gray);
            draw_quad(show_u_, -1.0f, -1.0f, 1.0f, 1.0f, 0.0f);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        void show_cell(int cell, mgl::TextureId tex, int lod, float lod_size, float invert, float gray)
        {
            const Display d = display(cell);
            mgl::glViewport(d.x, d.y, d.size(), d.size());
            show(tex, lod, lod_size, invert, gray);
        }

        // A renderbuffer cannot be sampled; glBlitFramebuffer copies it into the cell (integer magnification, NEAREST).
        void blit_cell(int cell, std::size_t src)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo_[src]);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            const Display d = display(cell);
            mgl::glBlitFramebuffer(0, 0, kSize, kSize, d.x, d.y, d.x + d.size(), d.y + d.size(),
                                   mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // Clears the bound framebuffer and executes a scene with GL state; the CPU model in simulate()
        // interprets the same list.
        void run_scene(const Scene& scene, bool has_depth, bool has_stencil)
        {
            mgl::glClearColor(kBackground[0], kBackground[1], kBackground[2], kBackground[3]);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::ClearBufferBit mask = mgl::ClearBufferBit::Color;
            if (has_depth) mask = mask | mgl::ClearBufferBit::Depth;
            if (has_stencil) mask = mask | mgl::ClearBufferBit::Stencil;
            mgl::glClear(mask);

            mgl::glUseProgram(flat_);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            for (const DrawCmd& c : scene)
            {
                const GLboolean cw = c.color_write ? GL_TRUE : GL_FALSE;
                mgl::glColorMask(cw, cw, cw, cw);
                if (c.depth_test) mgl::glEnable(mgl::Capability::DepthTest); else mgl::glDisable(mgl::Capability::DepthTest);
                if (c.stencil_test)
                {
                    mgl::glEnable(mgl::Capability::StencilTest);
                    mgl::glStencilFunc(c.func, c.ref, 0xFF);
                    mgl::glStencilOp(c.sfail, c.dpfail, c.dppass);
                }
                else
                {
                    mgl::glDisable(mgl::Capability::StencilTest);
                }
                mgl::glUniform4f(flat_u_.color, c.color[0], c.color[1], c.color[2], c.color[3]);
                draw_quad(flat_u_, ndc(c.rect.x0, kSize), ndc(c.rect.y0, kSize), ndc(c.rect.x1, kSize), ndc(c.rect.y1, kSize), c.z);
            }
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::StencilTest);
        }

        // =============================================================================================
        // Validation
        // =============================================================================================

        // Compares a target (read back through its own framebuffer) with the CPU image, and the cell
        // on the screen with what was read back (or with `display` when the screen shows a derived image).
        void check_target(demo::Check& check, std::string_view name, std::size_t fbo, int size, std::span<const Rgba8> cpu,
                          std::string_view how)
        {
            const std::vector<Rgba8> got = read_fbo(fbo_[fbo], size);
            const Diff d = compare(got, cpu, size, kConvertTolerance);
            check.expect(d.bad == 0, std::format("{}: the {}x{} target read back through its framebuffer matches the CPU model ({}; max deviation {})", name, size, size, how, d.max_dev),
                         d.first);
        }

        /// `want` is the 64 x 64 target; the cell shows it magnified by an integer factor m, so every window pixel
        /// is compared with the texel it shows.
        void check_cell(demo::Check& check, std::string_view name, int cell, std::span<const Rgba8> want, int tolerance, std::string_view how)
        {
            const Display shown = display(cell);
            const std::vector<Rgba8> got = read_cell(cell);
            const Diff d = compare(got, magnify(want, kSize, shown.scale), shown.size(), tolerance);
            check.expect(d.bad == 0, std::format("{}: the cell on the default framebuffer equals the target ({}; shown x{}; max deviation {})", name, how,
                                                 shown.scale, d.max_dev), d.first);
        }

        void check_render_to_texture(demo::Check& check)
        {
            // --- pass 1: the gradient + checkerboard rendered into the texture ------------------------
            const std::vector<Rgba8> cpu = pattern_image(kSize);
            check_target(check, "render to texture", kFboRtt, kSize, cpu, "gradient + 8x8 checker, +-1 for float->8-bit");
            const std::vector<Rgba8> rtt = read_fbo(fbo_[kFboRtt], kSize);
            // --- pass 2: the same texture sampled by a second pass on the screen ------------------------
            check_cell(check, "render to texture", kCellRtt, rtt, kConvertTolerance, "texture() with NEAREST in a second pass, +-1");

            // --- negative controls: the comparison must be able to fail -------------------------------
            const int wrong_step = count_mismatches(rtt, [](int i, int j) { return pattern_texel(i, j, kSize, 64.0); });
            check.expect(wrong_step > 1000, "negative control: a reference whose gradient step is 1/64 instead of 1/63 is rejected",
                         std::format("{} of {} texels differ by more than the tolerance", wrong_step, kSize * kSize));
            const int shifted = count_mismatches(rtt, [](int i, int j) { return pattern_texel(std::min(i + 1, kSize - 1), j, kSize); });
            check.expect(shifted > 1000, "negative control: a reference shifted by one texel is rejected",
                         std::format("{} of {} texels differ by more than the tolerance", shifted, kSize * kSize));

            // --- texture -> texture: negative of the first pass --------------------------------------
            std::vector<Rgba8> negative = rtt;
            for (Rgba8& t : negative) { t[0] = static_cast<std::uint8_t>(255 - t[0]); t[1] = static_cast<std::uint8_t>(255 - t[1]); t[2] = static_cast<std::uint8_t>(255 - t[2]); t[3] = 255; }
            check_target(check, "texture -> texture", kFboPost, kSize, negative, "255 - the first pass, alpha 1");
            check_cell(check, "texture -> texture", kCellPost, read_fbo(fbo_[kFboPost], kSize), kConvertTolerance, "sampled again, +-1");

            // --- a non-zero mipmap level ------------------------------------------------------------
            const std::vector<Rgba8> mip_cpu = pattern_image(kMipSize);
            check_target(check, std::format("mipmap level {}", kMipLevel), kFboMip, kMipSize, mip_cpu, "16x16 gradient + checker");
            const std::vector<Rgba8> mip = read_fbo(fbo_[kFboMip], kMipSize);
            check_cell(check, std::format("mipmap level {}", kMipLevel), kCellMip, magnify(mip, kMipSize, kSize / kMipSize), kConvertTolerance,
                       "texelFetch(level 2), every texel shown 4x4 pixels");
        }

        template <typename Ref>
        static int count_mismatches(const std::vector<Rgba8>& got, Ref&& ref)
        {
            int bad = 0;
            for (int j = 0; j < kSize; ++j)
                for (int i = 0; i < kSize; ++i)
                {
                    const Rgba8 want = ref(i, j);
                    const Rgba8 g = got[static_cast<std::size_t>(j) * kSize + static_cast<std::size_t>(i)];
                    int dev = 0;
                    for (std::size_t k = 0; k < 4; ++k) dev = std::max(dev, std::abs(int{g[k]} - int{want[k]}));
                    if (dev > kConvertTolerance) ++bad;
                }
            return bad;
        }

        void check_scenes(demo::Check& check)
        {
            const SceneImage depth_on = simulate(depth_scene(true));
            const SceneImage depth_off = simulate(depth_scene(false));
            const SceneImage stencil_on = simulate(stencil_scene(true));
            const SceneImage stencil_off = simulate(stencil_scene(false));
            const SceneImage packed = simulate(packed_scene());

            // Depth: the same list of draws in a depth-tested and an untested framebuffer.
            check_target(check, "depth test, DEPTH_COMPONENT16 renderbuffer", kFboD16, kSize, depth_on.color, "CPU depth pipeline");
            check_cell(check, "depth test, D16", kCellDepth16, read_fbo(fbo_[kFboD16], kSize), kBlitTolerance, "glBlitFramebuffer of the color renderbuffer");
            check_target(check, "depth test, DEPTH_COMPONENT24 renderbuffer", kFboD24, kSize, depth_on.color, "CPU depth pipeline");
            check_cell(check, "depth test, D24", kCellDepth24, read_fbo(fbo_[kFboD24], kSize), kBlitTolerance, "glBlitFramebuffer");
            check_target(check, "depth attached but DEPTH_TEST disabled", kFboD16Off, kSize, depth_off.color, "painter's order wins");
            check_cell(check, "depth test off", kCellDepth16Off, read_fbo(fbo_[kFboD16Off], kSize), kBlitTolerance, "glBlitFramebuffer");

            // Named pixels: where the answer depends on the depth buffer, and the negative controls.
            const Rgba8 green = to_rgba8(kGreen), red = to_rgba8(kRed), blue = to_rgba8(kBlue);
            const struct { int x, y; Rgba8 on, off; const char* what; } probes[] = {
                {32, 22, green, red, "near/far overlap (far quad drawn later)"},
                {48, 22, red, red, "far quad alone"},
                {44, 32, blue, blue, "stripe in front of the far quad"},
                {20, 32, green, blue, "stripe behind the near quad"},
            };
            for (const auto& p : probes)
            {
                const Rgba8 on = read_fbo_pixel(fbo_[kFboD16], p.x, p.y);
                const Rgba8 on24 = read_fbo_pixel(fbo_[kFboD24], p.x, p.y);
                const Rgba8 off = read_fbo_pixel(fbo_[kFboD16Off], p.x, p.y);
                check.expect(near_color(on, p.on) && near_color(on24, p.on),
                             std::format("depth test: {} -> {} on D16 and D24", p.what, demo::format_rgba(p.on)),
                             std::format("D16 {} D24 {}", demo::format_rgba(on), demo::format_rgba(on24)));
                check.expect(near_color(off, p.off), std::format("depth test off: {} -> {}", p.what, demo::format_rgba(p.off)), demo::format_rgba(off));
            }
            check.expect(!near_color(read_fbo_pixel(fbo_[kFboD16Off], 32, 22), green),
                         "negative control: with the depth test disabled the overlap pixel is NOT the depth-test result (the probe can fail)");
            check.expect(!near_color(read_fbo_pixel(fbo_[kFboD16Off], 20, 32), green),
                         "negative control: with the depth test disabled the stripe overwrites the near quad");

            // Stencil.
            check_target(check, "stencil test, STENCIL_INDEX8 renderbuffer", kFboStencil, kSize, stencil_on.color, "CPU stencil pipeline");
            check_cell(check, "stencil test", kCellStencil, read_fbo(fbo_[kFboStencil], kSize), kConvertTolerance, "texture() in a second pass");
            check_target(check, "stencil attached but STENCIL_TEST disabled", kFboStencilOff, kSize, stencil_off.color, "last paint wins");
            check_cell(check, "stencil test off", kCellStencilOff, read_fbo(fbo_[kFboStencilOff], kSize), kConvertTolerance, "texture() in a second pass");
            const struct { int x, y; Rgba8 on, off; const char* what; } sprobes[] = {
                {6, 6, blue, red, "outside every stencil square (value 0)"},
                {20, 20, yellow(), red, "ring of the first square (value 1)"},
                {32, 32, red, red, "inner square after INCR (value 2)"},
            };
            for (const auto& p : sprobes)
            {
                const Rgba8 on = read_fbo_pixel(fbo_[kFboStencil], p.x, p.y);
                const Rgba8 off = read_fbo_pixel(fbo_[kFboStencilOff], p.x, p.y);
                check.expect(near_color(on, p.on), std::format("stencil test: {} -> {}", p.what, demo::format_rgba(p.on)), demo::format_rgba(on));
                check.expect(near_color(off, p.off), std::format("stencil test off: {} -> {}", p.what, demo::format_rgba(p.off)), demo::format_rgba(off));
            }
            check.expect(!near_color(read_fbo_pixel(fbo_[kFboStencilOff], 6, 6), blue),
                         "negative control: without the stencil test the outside pixel is painted over (the probe can fail)");

            // One DEPTH24_STENCIL8 image for both.
            check_target(check, "DEPTH24_STENCIL8 via DEPTH_STENCIL_ATTACHMENT", kFboPacked, kSize, packed.color, "depth-resolved colors above, stencil values 1/2/3 below");
            check_cell(check, "DEPTH24_STENCIL8", kCellPacked, read_fbo(fbo_[kFboPacked], kSize), kConvertTolerance, "texture() in a second pass");
            const struct { int x, y; Rgbaf want; const char* what; } pprobes[] = {
                {10, 48, kGreen, "A alone, depth test passes"},
                {32, 48, kGreen, "overlap: B is behind A and loses the depth test"},
                {50, 48, kRed, "B alone"},
                {2, 16, kBackground, "no geometry, stencil 0"},
                {10, 16, kCyan, "stencil 1: only A"},
                {32, 16, kMagenta, "stencil 2: B failed the DEPTH test (the depth part of the image works)"},
                {50, 16, kYellow, "stencil 3: only B"},
            };
            for (const auto& p : pprobes)
            {
                const Rgba8 got = read_fbo_pixel(fbo_[kFboPacked], p.x, p.y);
                check.expect(near_color(got, to_rgba8(p.want)), std::format("packed depth+stencil: {} -> {}", p.what, demo::format_rgba(to_rgba8(p.want))), demo::format_rgba(got));
            }

            // Depth *texture*: the depth-tested draws land in a color texture and a depth texture.
            check_target(check, "depth texture attachment: color texture", kFboDepthTex, kSize, depth_on.color, "CPU depth pipeline");
            check_cell(check, "depth texture attachment: color", kCellDepthTexColor, read_fbo(fbo_[kFboDepthTex], kSize), kConvertTolerance, "texture() in a second pass");
            std::vector<Rgba8> gray;
            for (const float d : depth_on.depth) { const std::uint8_t g = to8(static_cast<double>(d)); gray.push_back({g, g, g, 255}); }
            check_cell(check, "depth texture attachment: the depth texture sampled as gray", kCellDepthTexGray, gray, kConvertTolerance,
                       "window depth 0.25 / 0.5 / 0.75 / 1.0 as texture().r, +-1 for the 8-bit output");
            const struct { int x, y; int expect; const char* what; } dprobes[] = {
                {20, 12, 64, "near quad: window depth 0.25"}, {48, 22, 191, "far quad alone: 0.75"},
                {44, 32, 128, "stripe in front of the far quad: 0.5"}, {2, 2, 255, "clear value 1.0"}, {32, 22, 64, "overlap keeps the nearer depth"},
            };
            for (const auto& p : dprobes)
            {
                const Rgba8 got = read_cell_pixel(kCellDepthTexGray, p.x, p.y);
                check.expect(std::abs(int{got[0]} - p.expect) <= kConvertTolerance, std::format("depth texture: {} -> gray {}", p.what, p.expect),
                             demo::format_rgba(got));
            }
        }

        static Rgba8 yellow() { return to_rgba8(kYellow); }

        static bool near_color(Rgba8 a, Rgba8 b)
        {
            for (std::size_t k = 0; k < 4; ++k)
                if (std::abs(int{a[k]} - int{b[k]}) > kConvertTolerance) return false;
            return true;
        }

        void check_attachment_queries(demo::Check& check)
        {
            using P = mgl::FramebufferAttachmentParameter;
            using A = mgl::FramebufferAttachment;

            // Texture color attachment (section 6.1.13).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboRtt]);
            check.expect_eq(attachment_param(kColor0, P::ObjectType), gl(GL_TEXTURE), "color attachment: OBJECT_TYPE = TEXTURE");
            check.expect_eq(attachment_param(kColor0, P::ObjectName), static_cast<int>(tex_[kTexRtt].value), "color attachment: OBJECT_NAME = the texture");
            check.expect_eq(attachment_param(kColor0, P::TextureLevel), 0, "color attachment: TEXTURE_LEVEL = 0");
            check.expect_eq(attachment_param(kColor0, P::TextureCubeMapFace), 0, "color attachment: TEXTURE_CUBE_MAP_FACE = 0 for a 2D texture");
            check.expect(attachment_param(kColor0, P::RedSize) >= 8 && attachment_param(kColor0, P::GreenSize) >= 8 &&
                             attachment_param(kColor0, P::BlueSize) >= 8 && attachment_param(kColor0, P::AlphaSize) >= 8,
                         "RGBA8 texture attachment: component sizes are at least 8 bits");
            check.expect_eq(attachment_param(kColor0, P::DepthSize) + attachment_param(kColor0, P::StencilSize), 0, "a color image has no depth or stencil bits");
            check.expect_eq(attachment_param(kColor0, P::ComponentType), gl(GL_UNSIGNED_NORMALIZED), "RGBA8: COMPONENT_TYPE = UNSIGNED_NORMALIZED");
            check.expect_eq(attachment_param(kColor0, P::ColorEncoding), gl(GL_LINEAR), "RGBA8: COLOR_ENCODING = LINEAR");

            // Level 2 of the mip texture.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboMip]);
            check.expect_eq(attachment_param(kColor0, P::TextureLevel), kMipLevel, "mip attachment: TEXTURE_LEVEL = 2");
            check.expect_eq(attachment_param(kColor0, P::ObjectName), static_cast<int>(tex_[kTexMip].value), "mip attachment: OBJECT_NAME = the mip texture");

            // Renderbuffer attachments + an empty attachment point.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboD16]);
            check.expect_eq(attachment_param(kColor0, P::ObjectType), gl(GL_RENDERBUFFER), "color attachment: OBJECT_TYPE = RENDERBUFFER");
            check.expect_eq(attachment_param(kColor0, P::ObjectName), static_cast<int>(rb_[kRbColorD16].value), "color attachment: OBJECT_NAME = the renderbuffer");
            check.expect_eq(attachment_param(A::Depth, P::ObjectType), gl(GL_RENDERBUFFER), "depth attachment: OBJECT_TYPE = RENDERBUFFER");
            check.expect_eq(attachment_param(A::Depth, P::ObjectName), static_cast<int>(rb_[kRbD16].value), "depth attachment: OBJECT_NAME = the renderbuffer");
            check.expect(attachment_param(A::Depth, P::DepthSize) >= 16, "DEPTH_COMPONENT16: FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE >= 16");
            check.expect_eq(attachment_param(A::Depth, P::RedSize), 0, "a depth image has no red bits");
            const mgl::FramebufferAttachment color1 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1);
            check.expect_eq(attachment_param(color1, P::ObjectType), gl(GL_NONE), "unused COLOR_ATTACHMENT1: OBJECT_TYPE = NONE");
            check.expect_eq(attachment_param(color1, P::ObjectName), 0, "unused COLOR_ATTACHMENT1: OBJECT_NAME = 0");
            expect_error(check, mgl::ErrorCode::InvalidOperation, "querying RED_SIZE of an empty attachment point -> INVALID_OPERATION (section 6.1.13)",
                         [&] { (void)attachment_param(color1, P::RedSize); });
            check.expect_eq(attachment_param(A::Stencil, P::ObjectType), gl(GL_NONE), "the stencil attachment point of this framebuffer is empty");

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboD24]);
            check.expect(attachment_param(A::Depth, P::DepthSize) >= 24, "DEPTH_COMPONENT24: FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE >= 24");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboStencil]);
            check.expect(attachment_param(A::Stencil, P::StencilSize) >= 8, "STENCIL_INDEX8: FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE >= 8");

            // DEPTH_STENCIL_ATTACHMENT sets both points to the same renderbuffer.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboPacked]);
            const int rb = static_cast<int>(rb_[kRbD24S8].value);
            check.expect(attachment_param(A::Depth, P::ObjectType) == gl(GL_RENDERBUFFER) && attachment_param(A::Depth, P::ObjectName) == rb,
                         "DEPTH_STENCIL_ATTACHMENT: the depth attachment point holds the renderbuffer");
            check.expect(attachment_param(A::Stencil, P::ObjectType) == gl(GL_RENDERBUFFER) && attachment_param(A::Stencil, P::ObjectName) == rb,
                         "DEPTH_STENCIL_ATTACHMENT: the stencil attachment point holds the same renderbuffer");
            check.expect_eq(attachment_param(A::DepthStencil, P::ObjectName), rb, "querying DEPTH_STENCIL_ATTACHMENT returns the shared object");
            check.expect(attachment_param(A::DepthStencil, P::DepthSize) >= 24 && attachment_param(A::DepthStencil, P::StencilSize) >= 8,
                         "DEPTH24_STENCIL8: depth size >= 24 and stencil size >= 8");
            expect_error(check, mgl::ErrorCode::InvalidOperation,
                         "COMPONENT_TYPE of a combined depth+stencil attachment -> INVALID_OPERATION (section 6.1.13)",
                         [&] { (void)attachment_param(A::DepthStencil, P::ComponentType); });

            // Depth texture.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboDepthTex]);
            check.expect(attachment_param(A::Depth, P::ObjectType) == gl(GL_TEXTURE) && attachment_param(A::Depth, P::ObjectName) == static_cast<int>(tex_[kTexDepth].value),
                         "depth attachment can be a texture: OBJECT_TYPE = TEXTURE, OBJECT_NAME = the depth texture");
            check.expect(attachment_param(A::Depth, P::DepthSize) >= 24, "DEPTH_COMPONENT24 texture: depth size >= 24");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void check_renderbuffer_queries(demo::Check& check)
        {
            using P = mgl::RenderbufferParameter;
            struct Case { std::size_t rb; mgl::InternalFormat format; GLenum gl_format; int r, g, b, a, d, s; const char* name; };
            const Case cases[] = {
                {kRbColorD16, mgl::InternalFormat::Rgba8, GL_RGBA8, 8, 8, 8, 8, 0, 0, "RGBA8"},
                {kRbD16, mgl::InternalFormat::DepthComponent16, GL_DEPTH_COMPONENT16, 0, 0, 0, 0, 16, 0, "DEPTH_COMPONENT16"},
                {kRbD24, mgl::InternalFormat::DepthComponent24, GL_DEPTH_COMPONENT24, 0, 0, 0, 0, 24, 0, "DEPTH_COMPONENT24"},
                {kRbS8, mgl::InternalFormat::StencilIndex8, GL_STENCIL_INDEX8, 0, 0, 0, 0, 0, 8, "STENCIL_INDEX8"},
                {kRbD24S8, mgl::InternalFormat::Depth24Stencil8, GL_DEPTH24_STENCIL8, 0, 0, 0, 0, 24, 8, "DEPTH24_STENCIL8"},
            };
            for (const Case& c : cases)
            {
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_[c.rb]);
                check.expect(renderbuffer_param(P::Width) == kSize && renderbuffer_param(P::Height) == kSize,
                             std::format("{} renderbuffer: RENDERBUFFER_WIDTH/HEIGHT = {}x{}", c.name, kSize, kSize));
                check.expect_eq(renderbuffer_param(P::InternalFormat), gl(c.gl_format), std::format("{} renderbuffer: RENDERBUFFER_INTERNAL_FORMAT", c.name));
                check.expect(renderbuffer_param(P::RedSize) >= c.r && renderbuffer_param(P::GreenSize) >= c.g && renderbuffer_param(P::BlueSize) >= c.b &&
                                 renderbuffer_param(P::AlphaSize) >= c.a && renderbuffer_param(P::DepthSize) >= c.d && renderbuffer_param(P::StencilSize) >= c.s,
                             std::format("{} renderbuffer: component sizes are at least the format's (R{} G{} B{} A{} D{} S{})", c.name, c.r, c.g, c.b, c.a, c.d, c.s),
                             std::format("got R{} G{} B{} A{} D{} S{}", renderbuffer_param(P::RedSize), renderbuffer_param(P::GreenSize), renderbuffer_param(P::BlueSize),
                                         renderbuffer_param(P::AlphaSize), renderbuffer_param(P::DepthSize), renderbuffer_param(P::StencilSize)));
                check.expect_eq(renderbuffer_param(P::Samples), 0, std::format("{} renderbuffer: RENDERBUFFER_SAMPLES = 0 for glRenderbufferStorage", c.name));
            }
            // Color renderbuffers never hold depth or stencil bits; the other sized color formats.
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_[kRbColorD16]);
            check.expect_eq(renderbuffer_param(P::DepthSize) + renderbuffer_param(P::StencilSize), 0, "an RGBA8 renderbuffer has no depth or stencil bits");

            mgl::RenderbufferId scratch{};
            mgl::glGenRenderbuffers(1, &scratch);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, scratch);
            // Table 6.15: a new renderbuffer has no storage and RGBA4 as its internal format.
            check.expect(renderbuffer_param(P::Width) == 0 && renderbuffer_param(P::Height) == 0, "a new renderbuffer is 0x0 (initial state, Table 6.15)");
            check.expect_eq(renderbuffer_param(P::InternalFormat), gl(GL_RGBA4), "a new renderbuffer has internal format RGBA4 (initial state, Table 6.15)");
            // Table 6.15 lists 0 as the initial value of every RENDERBUFFER_*_SIZE: the initial internal format
            // RGBA4 has no image yet, so it has no components.
            check.expect_or_known_deviation(
                renderbuffer_param(P::RedSize) == 0 && renderbuffer_param(P::GreenSize) == 0 && renderbuffer_param(P::BlueSize) == 0 &&
                    renderbuffer_param(P::AlphaSize) == 0 && renderbuffer_param(P::DepthSize) == 0 && renderbuffer_param(P::StencilSize) == 0,
                "a new renderbuffer has zero component sizes (initial state, Table 6.15)",
                std::format("got R{} G{} B{} A{} D{} S{}", renderbuffer_param(P::RedSize), renderbuffer_param(P::GreenSize), renderbuffer_param(P::BlueSize),
                            renderbuffer_param(P::AlphaSize), renderbuffer_param(P::DepthSize), renderbuffer_param(P::StencilSize)),
                "ANGLE", "ANGLE reports the component sizes of the initial format RGBA4 for a renderbuffer without storage; Table 6.15 gives 0 as the "
                "initial value of RENDERBUFFER_{RED,GREEN,BLUE,ALPHA,DEPTH,STENCIL}_SIZE (ES 3.2 Table 21.17)");
            struct Sized { mgl::InternalFormat format; GLenum gl_format; int r, g, b, a; const char* name; };
            const Sized sized[] = {
                {mgl::InternalFormat::Rgba4, GL_RGBA4, 4, 4, 4, 4, "RGBA4"},
                {mgl::InternalFormat::Rgb5A1, GL_RGB5_A1, 5, 5, 5, 1, "RGB5_A1"},
                {mgl::InternalFormat::Rgb565, GL_RGB565, 5, 6, 5, 0, "RGB565"},
                {mgl::InternalFormat::Rgb8, GL_RGB8, 8, 8, 8, 0, "RGB8"},
            };
            for (const Sized& s : sized)
            {
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, s.format, 16, 8);
                check.expect(renderbuffer_param(P::Width) == 16 && renderbuffer_param(P::Height) == 8 && renderbuffer_param(P::InternalFormat) == gl(s.gl_format),
                             std::format("glRenderbufferStorage({}, 16, 8): width, height and internal format read back", s.name));
                check.expect(renderbuffer_param(P::RedSize) >= s.r && renderbuffer_param(P::GreenSize) >= s.g && renderbuffer_param(P::BlueSize) >= s.b &&
                                 renderbuffer_param(P::AlphaSize) >= s.a,
                             std::format("{}: component sizes at least R{} G{} B{} A{} (section 4.4.2.2: \"at least the internal component sizes\")", s.name, s.r, s.g, s.b, s.a));
            }

            // Multisample storage: the sample count is a request for a minimum.
            const int requested = std::min(4, max_samples_);
            if (requested >= 1)
            {
                mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, requested, mgl::InternalFormat::Rgba8, 16, 16);
                const int actual = renderbuffer_param(P::Samples);
                check.expect(actual >= requested, std::format("glRenderbufferStorageMultisample({}): RENDERBUFFER_SAMPLES = {} >= requested (section 4.4.2.1)", requested, actual));
                check.expect(max_samples_ >= 4, std::format("MAX_SAMPLES = {} (ES 3.0 requires at least 4)", max_samples_));
            }
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glGetRenderbufferParameteriv with renderbuffer 0 bound -> INVALID_OPERATION (section 6.1.14)",
                         [&] { (void)renderbuffer_param(P::Width); });
            mgl::glDeleteRenderbuffers(1, &scratch);
        }

        void check_completeness(demo::Check& check)
        {
            using A = mgl::FramebufferAttachment;
            using S = mgl::FramebufferStatus;
            // The default framebuffer is complete whenever it exists (section 4.4.4).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            expect_status(check, S::Complete, "the default framebuffer is FRAMEBUFFER_COMPLETE");
            const struct { std::size_t fbo; const char* what; } complete[] = {
                {kFboRtt, "RGBA8 texture"},
                {kFboMip, "level 2 of a mipmapped texture"},
                {kFboD16, "RGBA8 + DEPTH_COMPONENT16 renderbuffers"},
                {kFboD24, "RGBA8 + DEPTH_COMPONENT24 renderbuffers"},
                {kFboStencil, "RGBA8 texture + STENCIL_INDEX8 renderbuffer"},
                {kFboPacked, "RGBA8 texture + DEPTH24_STENCIL8 renderbuffer (DEPTH_STENCIL_ATTACHMENT)"},
                {kFboDepthTex, "RGBA8 texture + DEPTH_COMPONENT24 texture"},
            };
            for (const auto& c : complete)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[c.fbo]);
                expect_status(check, S::Complete, std::format("framebuffer with {}: FRAMEBUFFER_COMPLETE", c.what));
            }

            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            expect_status(check, S::IncompleteMissingAttachment, "a new framebuffer object with nothing attached: FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");

            // Renderbuffer without storage has width = height = 0 -> attachment incomplete.
            std::array<mgl::RenderbufferId, 4> rb{};
            mgl::glGenRenderbuffers(static_cast<GLsizei>(rb.size()), rb.data());
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb[0]);
            attach_renderbuffer(kColor0, rb[0]);
            expect_status(check, S::IncompleteAttachment, "a zero-sized renderbuffer attached: FRAMEBUFFER_INCOMPLETE_ATTACHMENT (width and height must be non-zero)");
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 32, 32);
            expect_status(check, S::Complete, "after glRenderbufferStorage(RGBA8, 32, 32) the same framebuffer is FRAMEBUFFER_COMPLETE");

            // A depth-format image at a color attachment point (not color-renderable).
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb[1]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent16, 32, 32);
            attach_renderbuffer(kColor0, rb[1]);
            expect_status(check, S::IncompleteAttachment, "a DEPTH_COMPONENT16 renderbuffer at COLOR_ATTACHMENT0: FRAMEBUFFER_INCOMPLETE_ATTACHMENT (not color-renderable)");
            attach_renderbuffer(kColor0, rb[0]);

            // A color-format image at the depth / stencil attachment points.
            attach_renderbuffer(A::Depth, rb[0]);
            expect_status(check, S::IncompleteAttachment, "an RGBA8 renderbuffer at DEPTH_ATTACHMENT: FRAMEBUFFER_INCOMPLETE_ATTACHMENT (not depth-renderable)");
            attach_renderbuffer(A::Depth, mgl::RenderbufferId{0});
            attach_renderbuffer(A::Stencil, rb[0]);
            expect_status(check, S::IncompleteAttachment, "an RGBA8 renderbuffer at STENCIL_ATTACHMENT: FRAMEBUFFER_INCOMPLETE_ATTACHMENT (not stencil-renderable)");
            attach_renderbuffer(A::Stencil, mgl::RenderbufferId{0});
            expect_status(check, S::Complete, "after detaching the bad attachments the framebuffer is complete again");

            // Valid depth and stencil, but two different images: UNSUPPORTED (ES 3.0 requires them to be the same image).
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb[2]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::StencilIndex8, 32, 32);
            attach_renderbuffer(A::Depth, rb[1]);
            expect_status(check, S::Complete, "RGBA8 color + DEPTH_COMPONENT16 depth: FRAMEBUFFER_COMPLETE");
            attach_renderbuffer(A::Stencil, rb[2]);
            expect_status(check, S::Unsupported, "separate depth and stencil renderbuffers: FRAMEBUFFER_UNSUPPORTED (depth and stencil must be the same image, section 4.4.4.2)");
            attach_renderbuffer(A::Stencil, mgl::RenderbufferId{0});
            attach_renderbuffer(A::Depth, mgl::RenderbufferId{0});

            // ES 3.0 allows attachments of different sizes: INCOMPLETE_DIMENSIONS "cannot be generated".
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb[3]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent16, 16, 16);
            attach_renderbuffer(A::Depth, rb[3]);
            expect_status(check, S::Complete, "a 32x32 color and a 16x16 depth attachment: FRAMEBUFFER_COMPLETE (INCOMPLETE_DIMENSIONS cannot occur in ES 3.0)");
            attach_renderbuffer(A::Depth, mgl::RenderbufferId{0});

            // Multisample: all attached renderbuffers must have the same sample count.
            const int samples = std::min(4, max_samples_);
            if (samples >= 1)
            {
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb[3]);
                mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, samples, mgl::InternalFormat::Rgba8, 32, 32);
                attach_renderbuffer(kColor0, rb[3]);  // multisampled color ...
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb[1]);
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent16, 32, 32);
                attach_renderbuffer(A::Depth, rb[1]);  // ... and single-sampled depth
                expect_status(check, S::IncompleteMultisample, "a multisampled color and a single-sampled depth renderbuffer: FRAMEBUFFER_INCOMPLETE_MULTISAMPLE");
                attach_renderbuffer(A::Depth, mgl::RenderbufferId{0});
                attach_renderbuffer(kColor0, rb[0]);
            }

            // A texture that is not color-renderable (RGB9_E5 is texture-only in ES 3.0 / 3.2).
            mgl::TextureId tex{};
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgb9E5, 8, 8, 0, mgl::PixelFormat::Rgb,
                              mgl::PixelType::UnsignedInt5999Rev, nullptr);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, GL_NEAREST);
            attach_texture(kColor0, tex, 0);
            expect_status(check, S::IncompleteAttachment, "an RGB9_E5 texture (not color-renderable) at COLOR_ATTACHMENT0: FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 8, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            expect_status(check, S::Complete, "respecifying the attached texture as RGBA8 makes the framebuffer complete (TexImage2D changes completeness)");

            // Detaching everything returns to the empty state.
            attach_texture(kColor0, mgl::TextureId{0}, 0);
            expect_status(check, S::IncompleteMissingAttachment, "detaching the last image (texture 0): FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");
            check.expect_no_gl_error("the completeness sequence generated no GL error");

            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glDeleteRenderbuffers(static_cast<GLsizei>(rb.size()), rb.data());
            mgl::glDeleteTextures(1, &tex);
        }

        void check_invalid_framebuffer_operation(demo::Check& check)
        {
            using E = mgl::ErrorCode;
            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);  // incomplete: nothing attached
            check.expect(status_of_bound() == mgl::FramebufferStatus::IncompleteMissingAttachment, "setup: the scratch framebuffer is incomplete");

            // `tex` is the copy destination, `target` the framebuffer attachment of the control below
            // (copying into the very texture that is attached for reading would be a feedback loop).
            std::array<mgl::TextureId, 2> textures{};
            mgl::glGenTextures(2, textures.data());
            const mgl::TextureId tex = textures[0], target = textures[1];
            for (const mgl::TextureId t : textures)
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 4, 4, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, GL_NEAREST);
            }
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);

            mgl::glUseProgram(flat_);
            Rgba8 px{};
            const std::array<float, 4> value = {0.0f, 0.0f, 0.0f, 0.0f};
            expect_error(check, E::InvalidFramebufferOperation, "glClear on an incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION (section 4.4.4.4)",
                         [&] { mgl::glClear(mgl::ClearBufferBit::Color); });
            expect_error(check, E::InvalidFramebufferOperation, "glClearBufferfv on an incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, value.data()); });
            expect_error(check, E::InvalidFramebufferOperation, "glDrawArrays on an incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4); });
            expect_error(check, E::InvalidFramebufferOperation, "glReadPixels from an incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data()); });
            expect_error(check, E::InvalidFramebufferOperation, "glCopyTexSubImage2D from an incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { mgl::glCopyTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 0, 0, 1, 1); });
            expect_error(check, E::InvalidFramebufferOperation, "glCopyTexImage2D from an incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, 0, 0, 4, 4, 0); });
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            expect_error(check, E::InvalidFramebufferOperation, "glBlitFramebuffer with an incomplete READ framebuffer -> INVALID_FRAMEBUFFER_OPERATION",
                         [&] { mgl::glBlitFramebuffer(0, 0, 4, 4, 0, 0, 4, 4, mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest); });
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);

            // Positive control: the same calls succeed once an image is attached.
            attach_texture(kColor0, target, 0);
            check.expect(status_of_bound() == mgl::FramebufferStatus::Complete, "control: with a texture attached the framebuffer is complete");
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            mgl::glCopyTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 0, 0, 1, 1);  // copies into `tex`, not into the attached `target`
            check.expect_no_gl_error("control: clear, draw, read and copy on the now-complete framebuffer raise no error");

            // The offending command is ignored (Table 2.3): an incomplete framebuffer is not modified.
            mgl::glClearColor(0.2f, 0.8f, 0.2f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::RenderbufferId empty{};
            mgl::glGenRenderbuffers(1, &empty);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, empty);  // no storage: zero-sized
            attach_renderbuffer(mgl::FramebufferAttachment::Depth, empty);
            check.expect(status_of_bound() == mgl::FramebufferStatus::IncompleteAttachment, "setup: a zero-sized depth renderbuffer makes the framebuffer incomplete");
            mgl::glClearColor(0.8f, 0.2f, 0.2f, 1.0f);
            mgl::glUniform4f(flat_u_.color, 0.8f, 0.2f, 0.2f, 1.0f);
            mgl::glUniform4f(flat_u_.rect, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glUniform1f(flat_u_.z, 0.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            (void)demo::drain_gl_errors();
            attach_renderbuffer(mgl::FramebufferAttachment::Depth, mgl::RenderbufferId{0});
            check.expect(status_of_bound() == mgl::FramebufferStatus::Complete, "setup: after detaching it the framebuffer is complete again");
            const Rgba8 survivor = demo::read_pixel(2, 2);
            check.expect(near_color(survivor, to_rgba8(kGreen)), "a clear and a draw rejected with INVALID_FRAMEBUFFER_OPERATION did not modify the attached texture (the erroneous command is ignored)",
                         demo::format_rgba(survivor));

            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glDeleteTextures(2, textures.data());
            mgl::glDeleteRenderbuffers(1, &empty);
        }

        void check_lifecycle(demo::Check& check)
        {
            using T = mgl::FramebufferTarget;
            using G = mgl::GetParameter;
            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            check.expect(!mgl::glIsFramebuffer(fbo), "a generated framebuffer name is not a framebuffer object until it is bound (section 4.4.1)");
            check.expect(!mgl::glIsFramebuffer(mgl::FramebufferId{0}), "glIsFramebuffer(0) = FALSE");
            mgl::glBindFramebuffer(T::Framebuffer, fbo);
            check.expect(mgl::glIsFramebuffer(fbo), "binding creates the framebuffer object: glIsFramebuffer = TRUE");
            const int name = static_cast<int>(fbo.value);
            check.expect(demo::get_integer(G::FramebufferBinding) == name && demo::get_integer(G::DrawFramebufferBinding) == name &&
                             demo::get_integer(G::ReadFramebufferBinding) == name,
                         "glBindFramebuffer(FRAMEBUFFER) binds draw and read: FRAMEBUFFER_BINDING = DRAW_FRAMEBUFFER_BINDING = READ_FRAMEBUFFER_BINDING = the name");
            mgl::glBindFramebuffer(T::DrawFramebuffer, mgl::FramebufferId{0});
            check.expect(demo::get_integer(G::DrawFramebufferBinding) == 0 && demo::get_integer(G::ReadFramebufferBinding) == name,
                         "binding only DRAW_FRAMEBUFFER leaves READ_FRAMEBUFFER_BINDING untouched");
            mgl::glBindFramebuffer(T::Framebuffer, fbo);

            // Deleting an attached renderbuffer / texture detaches it from the *bound* framebuffer.
            mgl::RenderbufferId rb{};
            mgl::glGenRenderbuffers(1, &rb);
            check.expect(!mgl::glIsRenderbuffer(rb), "a generated renderbuffer name is not a renderbuffer until it is bound");
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            check.expect(mgl::glIsRenderbuffer(rb), "binding creates the renderbuffer: glIsRenderbuffer = TRUE");
            check.expect_eq(demo::get_integer(G::RenderbufferBinding), static_cast<int>(rb.value), "RENDERBUFFER_BINDING = the bound renderbuffer");
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 8, 8);
            attach_renderbuffer(kColor0, rb);
            check.expect_eq(attachment_param(kColor0, mgl::FramebufferAttachmentParameter::ObjectType), gl(GL_RENDERBUFFER), "the renderbuffer is attached");
            mgl::glDeleteRenderbuffers(1, &rb);
            check.expect(!mgl::glIsRenderbuffer(rb), "glDeleteRenderbuffers: glIsRenderbuffer = FALSE");
            check.expect_eq(demo::get_integer(G::RenderbufferBinding), 0, "deleting the bound renderbuffer reverts RENDERBUFFER_BINDING to 0");
            check.expect_eq(attachment_param(kColor0, mgl::FramebufferAttachmentParameter::ObjectType), gl(GL_NONE),
                            "deleting a renderbuffer attached to the bound framebuffer detaches it (section 4.4.2.3)");

            mgl::TextureId tex{};
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 8, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, GL_NEAREST);
            attach_texture(kColor0, tex, 0);
            check.expect_eq(attachment_param(kColor0, mgl::FramebufferAttachmentParameter::ObjectType), gl(GL_TEXTURE), "the texture is attached");
            mgl::glDeleteTextures(1, &tex);
            check.expect_eq(attachment_param(kColor0, mgl::FramebufferAttachmentParameter::ObjectType), gl(GL_NONE),
                            "deleting a texture attached to the bound framebuffer detaches it (section 4.4.2.4)");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            // Deleting the bound framebuffer binds 0 to both targets.
            mgl::glDeleteFramebuffers(1, &fbo);
            check.expect(!mgl::glIsFramebuffer(fbo), "glDeleteFramebuffers: glIsFramebuffer = FALSE");
            check.expect(demo::get_integer(G::DrawFramebufferBinding) == 0 && demo::get_integer(G::ReadFramebufferBinding) == 0,
                         "deleting the bound framebuffer reverts DRAW and READ bindings to 0 (section 4.4.1)");
            const mgl::FramebufferId zero{};
            mgl::glDeleteFramebuffers(1, &zero);
            check.expect_no_gl_error("deleting the name 0 is silently ignored");
        }

        void check_attachment_errors(demo::Check& check)
        {
            using E = mgl::ErrorCode;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            expect_error(check, E::InvalidOperation, "glFramebufferRenderbuffer with the default framebuffer bound -> INVALID_OPERATION (section 4.4.2.3)",
                         [&] { attach_renderbuffer(kColor0, rb_[kRbColorD16]); });
            expect_error(check, E::InvalidOperation, "glFramebufferTexture2D with the default framebuffer bound -> INVALID_OPERATION (section 4.4.2.4)",
                         [&] { attach_texture(kColor0, tex_[kTexRtt], 0); });

            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            expect_error(check, E::InvalidOperation, "glFramebufferRenderbuffer with a name that is not a renderbuffer -> INVALID_OPERATION",
                         [&] { attach_renderbuffer(kColor0, mgl::RenderbufferId{0x7fff0001u}); });
            expect_error(check, E::InvalidOperation, "glFramebufferTexture2D of a 2D texture with textarget = a cube map face -> INVALID_OPERATION",
                         [&] { mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::TextureCubeMapPositiveX, tex_[kTexRtt], 0); });
            check.expect(status_of_bound() == mgl::FramebufferStatus::IncompleteMissingAttachment, "a failed attach leaves the framebuffer unchanged (still empty)");

            mgl::RenderbufferId rb{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            expect_error(check, E::InvalidValue, "glRenderbufferStorage with width > MAX_RENDERBUFFER_SIZE -> INVALID_VALUE (section 4.4.2.1)",
                         [&] { mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, max_renderbuffer_size_ + 1, 1); });
            // The most samples any format supports (the first value of SAMPLES is the largest).
            GLint max_for_rgba8 = 0;
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Renderbuffer, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, 1, &max_for_rgba8);
            check.expect(max_for_rgba8 >= max_samples_, "glGetInternalformativ(RENDERBUFFER, RGBA8, SAMPLES): the largest count is >= MAX_SAMPLES (section 6.1.15)",
                         std::format("{} vs MAX_SAMPLES {}", max_for_rgba8, max_samples_));
            expect_error(check, E::InvalidOperation, "glRenderbufferStorageMultisample with more samples than the format supports -> INVALID_OPERATION",
                         [&] { mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, max_for_rgba8 + 1, mgl::InternalFormat::Rgba8, 8, 8); });
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteRenderbuffers(1, &rb);
            mgl::glDeleteFramebuffers(1, &fbo);
        }

        mgl::ProgramId pattern_{}, flat_{}, show_{};
        Locations pattern_u_{}, flat_u_{}, show_u_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        std::array<mgl::RenderbufferId, kRbCount> rb_{};
        std::array<mgl::FramebufferId, kFboCount> fbo_{};
        int max_renderbuffer_size_ = 0;
        int max_samples_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-framebuffer-objects",
        .title = "Framebuffer objects: render to texture, renderbuffers, completeness",
        .min_es = {3, 0},
        .summary = "Texture and renderbuffer attachments, depth/stencil (also DEPTH24_STENCIL8) proven by depth- and stencil-tested "
                   "draws, completeness incl. the defined failures, INVALID_FRAMEBUFFER_OPERATION, attachment/renderbuffer queries "
                   "and the object lifecycle, all checked against CPU models.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<FramebufferObjects>(argc, argv, info);
}
