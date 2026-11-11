// es30-srgb
//
// sRGB in OpenGL ES 3.0: the two places where the GL converts between the non-linear sRGB color space
// and linear light, and the framebuffer query that says which one applies.
//
//   * Texture side (section 3.8.16, equation 3.26): the SRGB8 and SRGB8_ALPHA8 internal formats hold sRGB-encoded
//     red, green and blue. A lookup returns them linearized; alpha is always linear and returned unchanged.
//   * Framebuffer side (sections 4.1.7, 4.1.8, equation 4.1, 4.2.3): when an attachment's
//     FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING is SRGB (color-renderable SRGB8_ALPHA8 textures and renderbuffers),
//     the destination color is linearized before blending, the blend result is encoded again, and the encoded
//     8-bit value is what is stored. Clears (glClear, glClearBuffer) and glBlitFramebuffer take part in the
//     same conversions (4.2.3, 4.3.3); glCopyTexImage2D refuses a copy that would mix the two encodings (3.8.5).
//
// Image (256 x 256, 8 bands of 32 pixels, band 0 on top; pixel column x of a band is the byte value x):
//   0  RGBA8 ramp, raw                                         the straight reference ramp
//   1  SRGB8_ALPHA8 ramp as the shader sees it                 decoded to linear: the dark curve
//   2  band 1 re-encoded by a shader pow()                     back to the straight ramp
//   3  stored bytes of an SRGB8_ALPHA8 target that a LINEAR ramp was written to       the bright curve (encode on write)
//   4  the same target after a 50% white overlay was blended in (linear space), stored bytes
//   5  an RGBA8 target with the same overlay (blended in the encoded values), stored bytes
//   6  alpha channel of the SRGB8_ALPHA8 ramp                  a straight ramp: alpha is not converted
//   7  SRGB8 (no alpha) ramp as the shader sees it             the same dark curve as band 1
//
// Validation: every lookup is read back bit for bit through an RGBA32UI render target (floatBitsToUint) and
// compared with the CPU implementation of equation 3.26; every band column is compared with a CPU model; stored
// bytes of the sRGB targets are compared with equation 4.1 and with the blending equation applied in linear
// space; clears, blits, the COLOR_ENCODING queries (also of the default framebuffer) and the copy rules are checked.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kRamp = 256;          // textures and targets are 256 texels wide: texel x = byte value x
    constexpr int kBand = 32;
    constexpr int kBands = 8;
    constexpr int kWindow = kRamp;

    using Bytes = std::array<std::uint8_t, 4>;

    // ---------------------------------------------------------------------------------------------
    // The two conversions, in double precision.
    // ---------------------------------------------------------------------------------------------

    /// Equation 3.26 (ES 3.0 3.8.16): sRGB-encoded component -> linear.
    double srgb_to_linear(double cs)
    {
        return cs <= 0.04045 ? cs / 12.92 : std::pow((cs + 0.055) / 1.055, 2.4);
    }

    /// Equation 4.1 (ES 3.0 4.1.8): linear component -> sRGB-encoded, with the exponent 0.41666 exactly as printed.
    double linear_to_srgb(double cl)
    {
        if (!(cl > 0.0)) return 0.0;
        if (cl < 0.0031308) return 12.92 * cl;
        if (cl < 1.0) return 1.055 * std::pow(cl, 0.41666) - 0.055;
        return 1.0;
    }

    /// The byte pattern of texel x. Four different functions of x, so that a mix-up of channels cannot go unnoticed.
    Bytes ramp_texel(int x)
    {
        return {static_cast<std::uint8_t>(x), static_cast<std::uint8_t>(255 - x), static_cast<std::uint8_t>((x * 37) & 255), static_cast<std::uint8_t>(x)};
    }

    /// Two-texel texture for the LINEAR filtering test.
    constexpr std::array<Bytes, 2> kPair = {{{64, 128, 192, 100}, {192, 64, 128, 200}}};

    double unorm(std::uint8_t b) { return static_cast<double>(b) / 255.0; }  // equation 2.1

    // ---------------------------------------------------------------------------------------------
    // Tolerances, in 8-bit steps (1/255) of |stored byte - 255 * exact value|, unless a name says otherwise.
    //
    // kRoundTolerance: results that are integers (or half-integers) before the conversion to 8 bits - copied bytes, 0.5 + 0.5 * d.
    // Equation 2.3 rounds to the nearest of 256 steps, so the stored byte differs from the exact value by at most 0.5 (a tie may
    // go either way); ES 3.0 2.1.1 lets the float arithmetic be off by "about 1 part in 10^5", at most 255 * 10^-5 = 0.003 steps
    // in [0, 1]. kEps leaves a margin of several times that.
    //
    // kConversionTolerance: results that are computed floats (a shader's l * l, 1 - l) or went through one of the sRGB
    // conversions (equation 3.26 or 4.1). docs/VALIDATION.md sets "8-bit quantization +-1" as the policy, and the measurements
    // fit it: real hardware converts a computed float to 8 bits with an error of up to ~0.6 steps instead of 0.5 (radeonsi:
    // 0.604, the "0.6 ULP" class of Direct3D 10), and the sRGB conversions add their own inaccuracy. ES 3.0 states the formulas but
    // no accuracy for them beyond 2.1.1's generic "1 part in 10^5", which every tested driver (radeonsi, llvmpipe, zink) misses
    // by orders of magnitude because the conversions use lookup tables or short polynomials: the unrounded decode result is
    // off by up to 0.49 steps (radeonsi) and 0.31 steps (llvmpipe); the encode by up to 0.12. Largest deviation of any 8-bit
    // result measured here: 0.85 steps (decode of a byte followed by the 8-bit conversion). The bound is 1.0.
    // It is still far tighter than every plausible mistake: a plain gamma-2.2 power curve instead of the piecewise sRGB curve is
    // 6 steps off at the dark end (negative control in check_encode_on_write), no conversion at all up to 100 steps.
    // ---------------------------------------------------------------------------------------------
    constexpr double kEps = 0.02;
    constexpr double kRoundTolerance = 0.5 + kEps;
    constexpr double kConversionTolerance = 1.0;

    /// Relative tolerance of a float computed from an exact formula: ES 3.0 2.1.1, "about 1 part in 10^5" per operation, times
    /// a handful of operations. Used where the result is a 32-bit float that is not rounded to 8 bits: alpha and RGBA8 lookups
    /// (a single division by 255) must meet it; the sRGB decode does not on any tested driver (see above).
    constexpr double kFloatRelTolerance = 1.0e-4;

    struct Cmp
    {
        int probes = 0;
        int bad = 0;
        double max_dev = 0.0;
        std::string first;

        void add(double got, double want, double tolerance, std::string_view where)
        {
            ++probes;
            const double dev = std::fabs(got - want);
            max_dev = std::max(max_dev, dev);
            if (dev > tolerance && bad++ == 0) first = std::format("{}: got {:.4f} expected {:.4f} (tolerance {:.4f})", where, got, want, tolerance);
        }
    };


    class Srgb final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kWindow || env.height < kWindow)
                throw demo::Skip(std::format("this demo draws into the lower left {0}x{0} pixels and needs a window at least that large", kWindow));

            exact_ = demo::build_program(env, "shaders/quad.vert", "shaders/exact_float.frag");
            show_ = demo::build_program(env, "shaders/quad.vert", "shaders/show.frag");
            ramp_ = demo::build_program(env, "shaders/quad.vert", "shaders/ramp.frag");
            color_ = demo::build_program(env, "shaders/quad.vert", "shaders/color.frag");

            // Textures: the same bytes as RGBA8, SRGB8_ALPHA8 and SRGB8, so that the only difference is the internal format.
            std::vector<std::uint8_t> rgba, rgb;
            for (int x = 0; x < kRamp; ++x)
            {
                const Bytes b = ramp_texel(x);
                rgba.insert(rgba.end(), b.begin(), b.end());
                rgb.insert(rgb.end(), b.begin(), b.begin() + 3);
            }
            std::vector<std::uint8_t> pair;
            for (const Bytes& b : kPair) pair.insert(pair.end(), b.begin(), b.end());

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            make_texture(tex_[kRgba8], mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba, kRamp, 1, rgba.data(), false);
            make_texture(tex_[kSrgba8], mgl::InternalFormat::Srgb8Alpha8, mgl::PixelFormat::Rgba, kRamp, 1, rgba.data(), false);
            make_texture(tex_[kSrgb8], mgl::InternalFormat::Srgb8, mgl::PixelFormat::Rgb, kRamp, 1, rgb.data(), false);
            make_texture(tex_[kPairTex], mgl::InternalFormat::Srgb8Alpha8, mgl::PixelFormat::Rgba, 2, 1, pair.data(), true);
            make_texture(tex_[kView], mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba, kRamp, 3, nullptr, false);
            make_texture(tex_[kSrgbTarget], mgl::InternalFormat::Srgb8Alpha8, mgl::PixelFormat::Rgba, kRamp, 2, nullptr, false);
            make_texture(tex_[kLinTarget], mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba, kRamp, 2, nullptr, false);

            // Render targets. 16 x 16 renderbuffers for clears and copies, 256 x 1 ones for blits, RGBA32UI for the exact path.
            mgl::glGenRenderbuffers(static_cast<GLsizei>(rb_.size()), rb_.data());
            make_renderbuffer(rb_[kExactRb], mgl::InternalFormat::Rgba32UI, kRamp, 1);
            make_renderbuffer(rb_[kSrgbRb], mgl::InternalFormat::Srgb8Alpha8, 16, 16);
            make_renderbuffer(rb_[kLinRb], mgl::InternalFormat::Rgba8, 16, 16);
            make_renderbuffer(rb_[kDepthRb], mgl::InternalFormat::DepthComponent16, 16, 16);
            make_renderbuffer(rb_[kScratchSrgbRb], mgl::InternalFormat::Srgb8Alpha8, kRamp, 1);
            make_renderbuffer(rb_[kScratchLinRb], mgl::InternalFormat::Rgba8, kRamp, 1);

            mgl::glGenFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            attach_renderbuffer(fbo_[kExactFbo], rb_[kExactRb], mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0));
            attach_texture(fbo_[kSrgbTargetFbo], tex_[kSrgbTarget]);
            attach_texture(fbo_[kLinTargetFbo], tex_[kLinTarget]);
            attach_renderbuffer(fbo_[kSrgbRbFbo], rb_[kSrgbRb], mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0));
            attach_renderbuffer(fbo_[kSrgbRbFbo], rb_[kDepthRb], mgl::FramebufferAttachment::Depth);
            attach_renderbuffer(fbo_[kLinRbFbo], rb_[kLinRb], mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0));
            attach_renderbuffer(fbo_[kScratchSrgbFbo], rb_[kScratchSrgbRb], mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0));
            attach_renderbuffer(fbo_[kScratchLinFbo], rb_[kScratchLinRb], mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            // The encoding of the default framebuffer is implementation-defined (6.1.13); remember it, because it decides what
            // a shader output looks like after it was written to the window.
            GLint encoding = 0;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::DefaultFramebufferQueryAttachment::Back,
                                                       mgl::FramebufferAttachmentParameter::ColorEncoding, &encoding);
            default_srgb_ = encoding == GL_SRGB;
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            // Dithering (on by default, 4.1.9) may change an 8-bit result by one step; the stored bytes are compared exactly.
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glDisable(mgl::Capability::Blend);

            // Pass 1: the same linear ramp into an SRGB8_ALPHA8 and an RGBA8 target (2 rows), then a 50% white overlay blended
            // into row 1 of each.
            const std::array<mgl::FramebufferId, 2> targets = {fbo_[kSrgbTargetFbo], fbo_[kLinTargetFbo]};
            for (const mgl::FramebufferId target : targets)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, target);
                mgl::glViewport(0, 0, kRamp, 2);
                mgl::glUseProgram(ramp_);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);

                mgl::glEnable(mgl::Capability::Blend);
                mgl::glBlendFunc(mgl::BlendFactor::SrcAlpha, mgl::BlendFactor::OneMinusSrcAlpha);
                mgl::glViewport(0, 1, kRamp, 1);
                mgl::glUseProgram(color_);
                mgl::glUniform4f(mgl::glGetUniformLocation(color_, "u_color"), 1.0f, 1.0f, 1.0f, 0.5f);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
                mgl::glDisable(mgl::Capability::Blend);
            }

            // Pass 2: read the stored bytes (no conversion on read, 4.3.2) and put them into an RGBA8 texture, so that the
            // window can show them raw. Row 0: sRGB target, ramp. Row 1: sRGB target, blended. Row 2: RGBA8 target, blended.
            const std::vector<std::uint8_t> srgb_stored = read_target(fbo_[kSrgbTargetFbo], kRamp, 2);
            const std::vector<std::uint8_t> lin_stored = read_target(fbo_[kLinTargetFbo], kRamp, 2);
            std::vector<std::uint8_t> view;
            view.insert(view.end(), srgb_stored.begin(), srgb_stored.begin() + kRamp * 4);
            view.insert(view.end(), srgb_stored.begin() + kRamp * 4, srgb_stored.end());
            view.insert(view.end(), lin_stored.begin() + kRamp * 4, lin_stored.end());
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kView]);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kRamp, 3, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, view.data());

            // Pass 3: the bands.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, kWindow, kWindow);
            mgl::glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(show_);
            mgl::glUniform1i(mgl::glGetUniformLocation(show_, "u_tex"), 0);
            struct BandSource
            {
                std::size_t texture;
                int row;
                int channel;
                bool encode;
            };
            const std::array<BandSource, kBands> sources = {{{kRgba8, 0, 0, false}, {kSrgba8, 0, 0, false}, {kSrgba8, 0, 0, true}, {kView, 0, 0, false},
                                                             {kView, 1, 0, false}, {kView, 2, 0, false}, {kSrgba8, 0, 3, false}, {kSrgb8, 0, 0, false}}};
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (int band = 0; band < kBands; ++band)
            {
                const BandSource& s = sources[static_cast<std::size_t>(band)];
                mgl::glViewport(0, (kBands - 1 - band) * kBand, kRamp, kBand);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[s.texture]);
                mgl::glUniform1i(mgl::glGetUniformLocation(show_, "u_row"), s.row);
                mgl::glUniform1i(mgl::glGetUniformLocation(show_, "u_channel"), s.channel);
                mgl::glUniform1i(mgl::glGetUniformLocation(show_, "u_encode"), s.encode ? 1 : 0);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_bands(check);   // the window first: everything below draws into other targets (and one patch into the window)

            check_texture_decode(check);
            check_texture_filtering(check);
            check_encode_on_write(check);
            check_blending(check);
            check_clears(check);
            check_blits(check);
            check_color_encoding_queries(check);
            check_copy_rules(check);
            check_not_renderable(check);
            check_default_framebuffer(check);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            render(env, 0);  // restore the window: the capture is taken after validate()
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            mgl::glDeleteRenderbuffers(static_cast<GLsizei>(rb_.size()), rb_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(exact_);
            mgl::glDeleteProgram(show_);
            mgl::glDeleteProgram(ramp_);
            mgl::glDeleteProgram(color_);
        }

    private:
        enum : std::size_t { kRgba8, kSrgba8, kSrgb8, kPairTex, kView, kSrgbTarget, kLinTarget, kTexCount };
        enum : std::size_t { kExactRb, kSrgbRb, kLinRb, kDepthRb, kScratchSrgbRb, kScratchLinRb, kRbCount };
        enum : std::size_t { kExactFbo, kSrgbTargetFbo, kLinTargetFbo, kSrgbRbFbo, kLinRbFbo, kScratchSrgbFbo, kScratchLinFbo, kFboCount };

        // ------------------------------------------------------------------------------------------
        // Setup helpers (every GL call stays visible)
        // ------------------------------------------------------------------------------------------
        static void make_texture(mgl::TextureId tex, mgl::InternalFormat internal, mgl::PixelFormat format, int w, int h, const void* data, bool linear)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, internal, w, h, 0, format, mgl::PixelType::UnsignedByte, data);
            // NEAREST: the initial MIN_FILTER needs a mip chain. CLAMP_TO_EDGE for the two-texel texture.
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, linear ? mgl::TextureMinFilter::Linear : mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, linear ? mgl::TextureMagFilter::Linear : mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
        }

        static void make_renderbuffer(mgl::RenderbufferId rb, mgl::InternalFormat internal, int w, int h)
        {
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, internal, w, h);
        }

        static void check_complete(std::string_view what)
        {
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete) throw demo::Failure(std::format("{} is incomplete: {}", what, demo::framebuffer_status_name(status)));
        }

        static void attach_renderbuffer(mgl::FramebufferId fbo, mgl::RenderbufferId rb, mgl::FramebufferAttachment attachment)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, attachment, mgl::RenderbufferTarget::Renderbuffer, rb);
            // A color-only or color + depth framebuffer is complete once its color attachment is there; check at each attach.
            if (attachment != mgl::FramebufferAttachment::Depth) check_complete("renderbuffer framebuffer");
        }

        static void attach_texture(mgl::FramebufferId fbo, mgl::TextureId tex)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), mgl::TextureTarget::Texture2D, tex, 0);
            check_complete("texture framebuffer");
        }

        /// glReadPixels of a w x h RGBA8 block of the color buffer of `fbo` (bottom-up rows). Reads store no conversion (4.3.2).
        static std::vector<std::uint8_t> read_target(mgl::FramebufferId fbo, int w, int h)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            return demo::read_rgba8(0, 0, w, h);
        }

        // ------------------------------------------------------------------------------------------
        // The window: eight bands, one probe per pixel column
        // ------------------------------------------------------------------------------------------

        /// What a shader output `linear` (0..1) becomes in the default framebuffer, in 0..255: itself, unless the default
        /// framebuffer happens to be sRGB-encoded (6.1.13: implementation-defined), where 4.1.8 encodes it.
        double window_value(double linear) const { return 255.0 * (default_srgb_ ? linear_to_srgb(linear) : linear); }

        void check_bands(demo::Check& check) const
        {
            const std::vector<std::uint8_t> window = demo::read_rgba8(0, 0, kWindow, kWindow);
            // The stored bytes that bands 3-5 show were read back during render(); read them again now (the targets still hold them).
            const std::vector<std::uint8_t> srgb_stored = read_target(fbo_[kSrgbTargetFbo], kRamp, 2);
            const std::vector<std::uint8_t> lin_stored = read_target(fbo_[kLinTargetFbo], kRamp, 2);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            struct Band
            {
                const char* what;
                double tolerance;
            };
            const std::array<Band, kBands> bands = {{
                {"band 0: RGBA8 ramp, raw (the reference ramp)", kRoundTolerance},
                {"band 1: SRGB8_ALPHA8 ramp sampled: decoded to linear, the dark curve (equation 3.26)", kConversionTolerance},
                {"band 2: band 1 re-encoded in the shader: back to the straight ramp", kConversionTolerance},
                {"band 3: stored bytes of the SRGB8_ALPHA8 target after a linear ramp was written (shown raw; the encoding itself is checked in the next section)", kRoundTolerance},
                {"band 4: stored bytes of the SRGB8_ALPHA8 target after the 50% white overlay (shown raw)", kRoundTolerance},
                {"band 5: stored bytes of the RGBA8 target after the same overlay (shown raw)", kRoundTolerance},
                {"band 6: alpha of the SRGB8_ALPHA8 ramp: unchanged, a straight ramp", kRoundTolerance},
                {"band 7: SRGB8 ramp sampled: the same dark curve as band 1", kConversionTolerance},
            }};
            for (int band = 0; band < kBands; ++band)
            {
                Cmp cmp;
                const int y = (kBands - 1 - band) * kBand + kBand / 2;
                for (int x = 0; x < kRamp; ++x)
                {
                    const auto o = static_cast<std::size_t>((y * kWindow + x) * 4);
                    const double got = static_cast<double>(window[o]);
                    double want = 0.0;
                    switch (band)
                    {
                    case 0:
                    case 6: want = window_value(unorm(static_cast<std::uint8_t>(x))); break;
                    case 1:
                    case 7: want = window_value(srgb_to_linear(unorm(static_cast<std::uint8_t>(x)))); break;
                    case 2: want = window_value(unorm(static_cast<std::uint8_t>(x))); break;
                    case 3: want = window_value(unorm(srgb_stored[static_cast<std::size_t>(x * 4)])); break;
                    case 4: want = window_value(unorm(srgb_stored[static_cast<std::size_t>((kRamp + x) * 4)])); break;
                    default: want = window_value(unorm(lin_stored[static_cast<std::size_t>((kRamp + x) * 4)])); break;
                    }
                    cmp.add(got, want, bands[static_cast<std::size_t>(band)].tolerance, std::format("column {}", x));
                    // Green and blue of a band are the same gray; the alpha is 255.
                    if (window[o + 1] != window[o] || window[o + 2] != window[o] || window[o + 3] != 255) cmp.add(1.0, 0.0, 0.0, std::format("column {} is not an opaque gray", x));
                }
                check.expect(cmp.bad == 0, std::format("{} - {} columns, largest deviation {:.3f}", bands[static_cast<std::size_t>(band)].what, cmp.probes, cmp.max_dev), cmp.first);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Texture side: exact decode through an RGBA32UI target
        // ------------------------------------------------------------------------------------------

        /// Samples `tex` (256 x 1 or 2 x 1) for every fragment of a 256 x 1 target and returns the four floats of each.
        std::vector<std::array<float, 4>> sample_exact(mgl::TextureId tex, int mode, float u = 0.5f) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kExactFbo]);
            mgl::glViewport(0, 0, kRamp, 1);
            mgl::glUseProgram(exact_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glUniform1i(mgl::glGetUniformLocation(exact_, "u_tex"), 0);
            mgl::glUniform1i(mgl::glGetUniformLocation(exact_, "u_mode"), mode);
            mgl::glUniform2f(mgl::glGetUniformLocation(exact_, "u_uv"), u, 0.5f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            std::vector<std::uint32_t> words(kRamp * 4);
            mgl::glReadPixels(0, 0, kRamp, 1, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, words.data());
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            std::vector<std::array<float, 4>> out(kRamp);
            for (std::size_t x = 0; x < kRamp; ++x)
                for (std::size_t c = 0; c < 4; ++c) out[x][c] = std::bit_cast<float>(words[x * 4 + c]);
            return out;
        }

        /// Compares float results with a model, tracking the relative error (for the strict check) and the absolute
        /// error in 8-bit steps (for the check with the accuracy class of kConversionTolerance).
        struct FloatCmp
        {
            int probes = 0;
            int bad_rel = 0;    // relative error above kFloatRelTolerance
            int bad_steps = 0;  // absolute error above kConversionTolerance 8-bit steps
            double max_rel = 0.0;
            double max_steps = 0.0;
            std::string first_rel, first_steps;

            void add(double got, double want, std::string_view where)
            {
                ++probes;
                const double rel = want == 0.0 ? std::fabs(got) : std::fabs(got - want) / std::fabs(want);
                const double steps = 255.0 * std::fabs(got - want);
                max_rel = std::max(max_rel, rel);
                max_steps = std::max(max_steps, steps);
                const std::string what = std::format("{}: got {:.9g} expected {:.9g}", where, got, want);
                if (rel > kFloatRelTolerance && bad_rel++ == 0) first_rel = what;
                if (steps > kConversionTolerance && bad_steps++ == 0) first_steps = what;
            }
        };

        void check_texture_decode(demo::Check& check) const
        {
            struct Format
            {
                std::size_t tex;
                const char* name;
                bool decode_rgb;   // sRGB format?
                bool has_alpha;
            };
            const std::array<Format, 3> formats = {{{kRgba8, "RGBA8", false, true}, {kSrgba8, "SRGB8_ALPHA8", true, true}, {kSrgb8, "SRGB8", true, false}}};
            for (const Format& f : formats)
            {
                for (const int mode : {0, 1})
                {
                    const char* lookup = mode == 0 ? "texture()" : "texelFetch()";
                    const std::vector<std::array<float, 4>> got = sample_exact(tex_[f.tex], mode);
                    FloatCmp rgb, alpha;
                    for (int x = 0; x < kRamp; ++x)
                    {
                        const Bytes b = ramp_texel(x);
                        for (std::size_t c = 0; c < 3; ++c)
                        {
                            const double stored = unorm(b[c]);
                            rgb.add(static_cast<double>(got[static_cast<std::size_t>(x)][c]), f.decode_rgb ? srgb_to_linear(stored) : stored, std::format("texel {} channel {}", x, c));
                        }
                        alpha.add(static_cast<double>(got[static_cast<std::size_t>(x)][3]), f.has_alpha ? unorm(b[3]) : 1.0, std::format("texel {} alpha", x));
                    }
                    if (f.decode_rgb)
                    {
                        check.expect(rgb.bad_steps == 0,
                                     std::format("{}: {} returns red, green, blue linearized by equation 3.26 to within {} 8-bit step for all 256 texels (largest error {:.3f} steps, relative {:.1e})", f.name, lookup,
                                                 kConversionTolerance, rgb.max_steps, rgb.max_rel),
                                     rgb.first_steps);
                    }
                    else
                    {
                        check.expect(rgb.bad_rel == 0, std::format("{}: {} returns red, green, blue as stored byte / 255 (equation 2.1) for all 256 texels (largest relative error {:.1e})", f.name, lookup, rgb.max_rel),
                                     rgb.first_rel);
                    }
                    check.expect(alpha.bad_rel == 0,
                                 std::format("{}: {} returns alpha as {} for all 256 texels (largest relative error {:.1e})", f.name, lookup, f.has_alpha ? "stored byte / 255, NOT linearized" : "1 (an RGB format has alpha 1)",
                                             alpha.max_rel),
                                 alpha.first_rel);
                    // ES 3.0 §3.8.16 states no accuracy for the sRGB decode, and §2.1.1 is a general remark about
                    // floating-point computation, not a requirement on this conversion. So the decode accuracy is
                    // reported as an implementation-quality measurement, not asserted (the 1-step check above is).
                    if (f.decode_rgb && f.has_alpha && mode == 1)
                        std::cout << std::format("     note: SRGB8_ALPHA8 texelFetch decode: largest relative error {:.1e} ({:.3f} 8-bit steps)\n",
                                                 rgb.max_rel, rgb.max_steps);
                }
            }

            // Negative controls: the comparisons can fail. The decoded SRGB8_ALPHA8 values are NOT the stored bytes / 255, and the RGBA8 values (same
            // bytes) are NOT decoded; most texels differ from the wrong expectation by far more than any tolerance used above.
            const std::vector<std::array<float, 4>> got = sample_exact(tex_[kSrgba8], 1);
            int differing = 0;
            for (int x = 0; x < kRamp; ++x)
                if (255.0 * std::fabs(static_cast<double>(got[static_cast<std::size_t>(x)][0]) - unorm(ramp_texel(x)[0])) > 4.0 * kConversionTolerance) ++differing;
            check.expect(differing > 200, "negative control: the SRGB8_ALPHA8 red channel differs from the undecoded stored byte / 255 by more than 4 steps for most of the 256 texels",
                         std::format("{} of 256 texels differ", differing));
            const std::vector<std::array<float, 4>> plain = sample_exact(tex_[kRgba8], 1);
            int plain_differing = 0;
            for (int x = 0; x < kRamp; ++x)
                if (255.0 * std::fabs(static_cast<double>(plain[static_cast<std::size_t>(x)][0]) - srgb_to_linear(unorm(ramp_texel(x)[0]))) > 4.0 * kConversionTolerance) ++plain_differing;
            check.expect(plain_differing > 200, "negative control: the RGBA8 texture (same bytes) is NOT decoded: it differs from equation 3.26 by more than 4 steps for most texels",
                         std::format("{} of 256 texels differ", plain_differing));
        }

        // 3.8.16: "Ideally, implementations should perform this color conversion on each sample prior to filtering but
        // implementations are allowed to perform this conversion after filtering." The specification itself allows both
        // orders; the result must be one of the two models, and it must not be the undecoded average.
        void check_texture_filtering(demo::Check& check) const
        {
            const std::vector<std::array<float, 4>> got = sample_exact(tex_[kPairTex], 2, 0.5f);  // exactly between the two texels: weights 1/2, 1/2
            const std::array<float, 4>& v = got[0];
            std::array<double, 4> before{}, after{}, undecoded{};
            for (std::size_t c = 0; c < 3; ++c)
            {
                const double a = unorm(kPair[0][c]), b = unorm(kPair[1][c]);
                before[c] = 0.5 * (srgb_to_linear(a) + srgb_to_linear(b));  // convert each texel, then filter
                after[c] = srgb_to_linear(0.5 * (a + b));                    // filter the stored values, then convert
                undecoded[c] = 0.5 * (a + b);                                // no sRGB handling at all
            }
            before[3] = after[3] = undecoded[3] = 0.5 * (unorm(kPair[0][3]) + unorm(kPair[1][3]));
            auto matches = [&](const std::array<double, 4>& model) {
                bool ok = true;
                for (std::size_t c = 0; c < 4; ++c) ok = ok && 255.0 * std::fabs(static_cast<double>(v[c]) - model[c]) <= kConversionTolerance;
                return ok;
            };
            const bool pre = matches(before), post = matches(after), raw = matches(undecoded);
            check.expect(pre || post, "SRGB8_ALPHA8 with LINEAR filtering between two texels: the result is the average of the linearized texels (preferred) or the linearization of the average (also allowed), 3.8.16",
                         std::format("got ({:.5f}, {:.5f}, {:.5f}, {:.5f}); convert-then-filter ({:.5f}, {:.5f}, {:.5f}), filter-then-convert ({:.5f}, {:.5f}, {:.5f})", v[0], v[1], v[2], v[3],
                                     before[0], before[1], before[2], after[0], after[1], after[2]));
            check.expect(!raw, "SRGB8_ALPHA8 with LINEAR filtering is not the plain average of the stored bytes (the sRGB decode is applied)");
            check.expect(std::fabs(static_cast<double>(v[3]) - before[3]) <= kFloatRelTolerance * before[3], "SRGB8_ALPHA8 with LINEAR filtering: alpha is the plain average of the alpha texels (never linearized)");
            std::cout << std::format("   note: this driver converts {} filtering\n", pre ? "BEFORE" : (post ? "AFTER" : "neither before nor after (?)"));
        }

        // ------------------------------------------------------------------------------------------
        // Framebuffer side
        // ------------------------------------------------------------------------------------------

        /// Writing a linear ramp into an SRGB8_ALPHA8 target stores 255 * equation 4.1 (read back with glReadPixels, which
        /// does not convert); RGBA8 stores the ramp itself; alpha is stored unencoded in both.
        void check_encode_on_write(demo::Check& check) const
        {
            const std::vector<std::uint8_t> srgb = read_target(fbo_[kSrgbTargetFbo], kRamp, 2);
            const std::vector<std::uint8_t> lin = read_target(fbo_[kLinTargetFbo], kRamp, 2);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            Cmp enc_rgb, enc_alpha, lin_rgb;
            for (int x = 0; x < kRamp; ++x)
            {
                const double l = static_cast<double>(x) / 255.0;
                const std::array<double, 3> value = {l, 1.0 - l, l * l};
                for (std::size_t c = 0; c < 3; ++c)
                {
                    const auto idx = static_cast<std::size_t>(x * 4) + c;
                    enc_rgb.add(static_cast<double>(srgb[idx]), 255.0 * linear_to_srgb(value[c]), kConversionTolerance, std::format("x {} channel {}", x, c));
                    lin_rgb.add(static_cast<double>(lin[idx]), 255.0 * value[c], kConversionTolerance, std::format("x {} channel {}", x, c));
                }
                enc_alpha.add(static_cast<double>(srgb[static_cast<std::size_t>(x * 4) + 3]), 255.0 * (1.0 - l), kRoundTolerance, std::format("x {}", x));
            }
            check.expect(enc_rgb.bad == 0, std::format("SRGB8_ALPHA8 target: a linear ramp written by a shader is stored as 255 * equation 4.1 ({} bytes, largest deviation {:.3f})", enc_rgb.probes, enc_rgb.max_dev),
                         enc_rgb.first);
            check.expect(enc_alpha.bad == 0, "SRGB8_ALPHA8 target: alpha is stored as written, not encoded (256 bytes)", enc_alpha.first);
            check.expect(lin_rgb.bad == 0, std::format("RGBA8 target (negative control): the same ramp is stored as written ({} bytes, largest deviation {:.3f})", lin_rgb.probes, lin_rgb.max_dev), lin_rgb.first);
            int differing = 0;
            for (int x = 0; x < kRamp; ++x)
                if (std::abs(static_cast<int>(srgb[static_cast<std::size_t>(x * 4)]) - static_cast<int>(lin[static_cast<std::size_t>(x * 4)])) > 8) ++differing;
            check.expect(differing > 100, "negative control: the stored red bytes of the SRGB8_ALPHA8 and the RGBA8 target differ by more than 8 for most of the 256 columns",
                         std::format("{} of 256 differ", differing));
            // The tolerance can tell the piecewise sRGB curve from a plain gamma-2.2 power curve (6 steps apart at the dark end).
            int gamma_differing = 0;
            for (int x = 0; x < kRamp; ++x)
                if (std::fabs(static_cast<double>(srgb[static_cast<std::size_t>(x * 4)]) - 255.0 * std::pow(static_cast<double>(x) / 255.0, 1.0 / 2.2)) > kConversionTolerance) ++gamma_differing;
            check.expect(gamma_differing >= 5, "negative control: the stored red bytes are NOT those of a gamma-2.2 power curve: they differ by more than the tolerance in at least 5 columns",
                         std::format("{} of 256 columns differ from 255 * l^(1/2.2)", gamma_differing));
        }

        /// 4.1.7.1: the destination of an sRGB target is linearized before blending and the result is encoded again.
        /// Overlay (1, 1, 1, 0.5) with SRC_ALPHA / ONE_MINUS_SRC_ALPHA: result = 0.5 + 0.5 * dest.
        void check_blending(demo::Check& check) const
        {
            const std::vector<std::uint8_t> srgb = read_target(fbo_[kSrgbTargetFbo], kRamp, 2);
            const std::vector<std::uint8_t> lin = read_target(fbo_[kLinTargetFbo], kRamp, 2);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            Cmp in_linear, in_encoded, alpha_srgb, naive;
            for (int x = 0; x < kRamp; ++x)
            {
                for (std::size_t c = 0; c < 3; ++c)
                {
                    const std::size_t below = static_cast<std::size_t>(x * 4) + c, above = static_cast<std::size_t>((kRamp + x) * 4) + c;
                    const double dest_stored = unorm(srgb[below]);
                    in_linear.add(static_cast<double>(srgb[above]), 255.0 * linear_to_srgb(0.5 + 0.5 * srgb_to_linear(dest_stored)), kConversionTolerance, std::format("x {} channel {}", x, c));
                    // What blending in the stored (encoded) values would give: the model that must NOT match.
                    naive.add(static_cast<double>(srgb[above]), 255.0 * (0.5 + 0.5 * dest_stored), kConversionTolerance, std::format("x {} channel {}", x, c));
                    in_encoded.add(static_cast<double>(lin[above]), 255.0 * (0.5 + 0.5 * unorm(lin[below])), kRoundTolerance, std::format("x {} channel {}", x, c));
                }
                const std::size_t a_below = static_cast<std::size_t>(x * 4) + 3, a_above = static_cast<std::size_t>((kRamp + x) * 4) + 3;
                alpha_srgb.add(static_cast<double>(srgb[a_above]), 255.0 * (0.5 * 0.5 + 0.5 * unorm(srgb[a_below])), kRoundTolerance, std::format("x {}", x));
            }
            check.expect(in_linear.bad == 0,
                         std::format("SRGB8_ALPHA8 target: blending 50% white over the stored ramp = encode(0.5 + 0.5 * decode(dest)), i.e. in linear space ({} bytes, largest deviation {:.3f})", in_linear.probes, in_linear.max_dev),
                         in_linear.first);
            check.expect(naive.bad > 200, "negative control: the same pixels do NOT match blending in the stored values (that model is off by up to a third of the range for most columns)",
                         std::format("{} of {} bytes differ from the naive model", naive.bad, naive.probes));
            check.expect(in_encoded.bad == 0, std::format("RGBA8 target: the same overlay blends the stored values directly, 0.5 + 0.5 * dest ({} bytes, largest deviation {:.3f})", in_encoded.probes, in_encoded.max_dev),
                         in_encoded.first);
            check.expect(alpha_srgb.bad == 0, "SRGB8_ALPHA8 target: destination alpha is not linearized for blending (256 bytes)", alpha_srgb.first);
        }

        /// 4.2.3: Clear and ClearBuffer "clamping and conversion ... same fashion", and the list of per-fragment operations that apply to
        /// Clear includes "sRGB conversion (see section 4.1.8)": a cleared sRGB target stores the encoded clear color.
        void check_clears(demo::Check& check) const
        {
            auto read_first = [&](mgl::FramebufferId fbo) {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
                return demo::read_rgba8(0, 0, 16, 16);
            };
            auto expect_cleared = [&](mgl::FramebufferId fbo, bool srgb, std::string_view what) {
                const std::vector<std::uint8_t> px = read_first(fbo);
                const std::array<double, 4> clear = {0.4, 0.2, 0.8, 0.6};
                Cmp cmp;
                for (std::size_t p = 0; p < 16 * 16; ++p)
                    for (std::size_t c = 0; c < 4; ++c)
                        cmp.add(static_cast<double>(px[p * 4 + c]), 255.0 * ((srgb && c < 3) ? linear_to_srgb(clear[c]) : clear[c]), (srgb && c < 3) ? kConversionTolerance : kRoundTolerance, std::format("pixel {} channel {}", p, c));
                check.expect(cmp.bad == 0, std::format("{} (largest deviation {:.3f})", what, cmp.max_dev), cmp.first);
            };
            mgl::glDisable(mgl::Capability::Dither);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kSrgbRbFbo]);
            mgl::glClearColor(0.4f, 0.2f, 0.8f, 0.6f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            expect_cleared(fbo_[kSrgbRbFbo], true, "glClear on an SRGB8_ALPHA8 renderbuffer stores the sRGB-encoded clear color (alpha unchanged)");

            const std::array<GLfloat, 4> value = {0.4f, 0.2f, 0.8f, 0.6f};
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kSrgbRbFbo]);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, value.data());
            expect_cleared(fbo_[kSrgbRbFbo], true, "glClearBufferfv(COLOR) on an SRGB8_ALPHA8 renderbuffer stores the same encoded color");

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kLinRbFbo]);
            mgl::glClearColor(0.4f, 0.2f, 0.8f, 0.6f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            expect_cleared(fbo_[kLinRbFbo], false, "negative control: glClear on an RGBA8 renderbuffer stores the clear color as is");
        }

        /// 4.3.3: a blit decodes what it reads from an SRGB read buffer (equation 3.26) and encodes what it writes to an SRGB draw buffer (4.1.8).
        void check_blits(demo::Check& check) const
        {
            auto blit = [&](mgl::FramebufferId src, mgl::FramebufferId dst) {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, src);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, dst);
                mgl::glBlitFramebuffer(0, 0, kRamp, 1, 0, 0, kRamp, 1, mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            };
            const std::vector<std::uint8_t> srgb = read_target(fbo_[kSrgbTargetFbo], kRamp, 1);
            const std::vector<std::uint8_t> lin = read_target(fbo_[kLinTargetFbo], kRamp, 1);

            struct Case
            {
                mgl::FramebufferId src, dst;
                const std::vector<std::uint8_t>& source;
                int mode;  // 0: SRGB -> RGBA8 (decode), 1: RGBA8 -> SRGB (encode), 2: SRGB -> SRGB (decode, encode)
                const char* what;
            };
            const std::array<Case, 3> cases = {{
                {fbo_[kSrgbTargetFbo], fbo_[kScratchLinFbo], srgb, 0, "blit SRGB8_ALPHA8 -> RGBA8: the stored sRGB bytes arrive linearized (equation 3.26), alpha unchanged"},
                {fbo_[kLinTargetFbo], fbo_[kScratchSrgbFbo], lin, 1, "blit RGBA8 -> SRGB8_ALPHA8: the stored bytes are read as linear and arrive encoded (equation 4.1), alpha unchanged"},
                {fbo_[kSrgbTargetFbo], fbo_[kScratchSrgbFbo], srgb, 2, "blit SRGB8_ALPHA8 -> SRGB8_ALPHA8: decoded and encoded again, the stored bytes arrive unchanged"},
            }};
            for (const Case& c : cases)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, c.dst);
                mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
                blit(c.src, c.dst);
                const std::vector<std::uint8_t> out = read_target(c.dst, kRamp, 1);
                Cmp cmp;
                for (int x = 0; x < kRamp; ++x)
                    for (std::size_t ch = 0; ch < 4; ++ch)
                    {
                        const std::size_t idx = static_cast<std::size_t>(x * 4) + ch;
                        const double s = unorm(c.source[idx]);
                        double want = 255.0 * s;
                        if (ch < 3 && c.mode == 0) want = 255.0 * srgb_to_linear(s);
                        if (ch < 3 && c.mode == 1) want = 255.0 * linear_to_srgb(s);
                        cmp.add(static_cast<double>(out[idx]), want, (ch < 3 && c.mode != 2) ? kConversionTolerance : kRoundTolerance, std::format("x {} channel {}", x, ch));
                    }
                check.expect(cmp.bad == 0, std::format("{} ({} bytes, largest deviation {:.3f})", c.what, cmp.probes, cmp.max_dev), cmp.first);
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // ------------------------------------------------------------------------------------------
        // 6.1.13: FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING
        // ------------------------------------------------------------------------------------------
        static GLint encoding_of(mgl::FramebufferId fbo, mgl::FramebufferAttachment attachment)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            GLint v = -1;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, attachment, mgl::FramebufferAttachmentParameter::ColorEncoding, &v);
            return v;
        }

        void check_color_encoding_queries(demo::Check& check) const
        {
            const mgl::FramebufferAttachment color0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
            check.expect(encoding_of(fbo_[kSrgbTargetFbo], color0) == GL_SRGB, "COLOR_ENCODING of an SRGB8_ALPHA8 texture attachment is SRGB (6.1.13)");
            check.expect(encoding_of(fbo_[kSrgbRbFbo], color0) == GL_SRGB, "COLOR_ENCODING of an SRGB8_ALPHA8 renderbuffer attachment is SRGB");
            check.expect(encoding_of(fbo_[kLinTargetFbo], color0) == GL_LINEAR, "COLOR_ENCODING of an RGBA8 texture attachment is LINEAR");
            check.expect(encoding_of(fbo_[kLinRbFbo], color0) == GL_LINEAR, "COLOR_ENCODING of an RGBA8 renderbuffer attachment is LINEAR");
            check.expect(encoding_of(fbo_[kSrgbRbFbo], mgl::FramebufferAttachment::Depth) == GL_LINEAR,
                         "COLOR_ENCODING of a depth attachment is LINEAR: \"if attachment is not a color attachment ... LINEAR\" (6.1.13)");
            check.expect(encoding_of(fbo_[kExactFbo], color0) == GL_LINEAR, "COLOR_ENCODING of an RGBA32UI renderbuffer attachment is LINEAR");

            // The default framebuffer: "color encoding is determined by the implementation" (6.1.13). The answer must be one of the two
            // values, and the window must behave accordingly (check_default_framebuffer).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            GLint back = -1;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::DefaultFramebufferQueryAttachment::Back,
                                                       mgl::FramebufferAttachmentParameter::ColorEncoding, &back);
            check.expect(back == GL_LINEAR || back == GL_SRGB, "COLOR_ENCODING of the default framebuffer's BACK buffer is LINEAR or SRGB (implementation-defined, 6.1.13)", std::format("{:#x}", back));
            std::cout << std::format("   note: the default framebuffer is {}-encoded on this context\n", back == GL_SRGB ? "sRGB" : "LINEAR");
            check.expect_no_gl_error("the COLOR_ENCODING queries raise no error");
        }

        /// What a shader output written to the window ends up as: the default framebuffer follows its own COLOR_ENCODING.
        void check_default_framebuffer(demo::Check& check) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, 4, 4);
            mgl::glUseProgram(color_);
            mgl::glUniform4f(mgl::glGetUniformLocation(color_, "u_color"), 0.4f, 0.2f, 0.8f, 1.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, 4, 4);
            const std::array<double, 3> color = {0.4, 0.2, 0.8};
            Cmp cmp;
            for (std::size_t p = 0; p < 16; ++p)
                for (std::size_t c = 0; c < 3; ++c) cmp.add(static_cast<double>(px[p * 4 + c]), window_value(color[c]), default_srgb_ ? kConversionTolerance : kRoundTolerance, std::format("pixel {} channel {}", p, c));
            check.expect(cmp.bad == 0,
                         std::format("a linear color drawn into the default framebuffer is stored as {} (its COLOR_ENCODING is {}; largest deviation {:.3f})",
                                     default_srgb_ ? "equation 4.1 of the value" : "the value itself", default_srgb_ ? "SRGB" : "LINEAR", cmp.max_dev),
                         cmp.first);
        }

        // ------------------------------------------------------------------------------------------
        // Rules that mix the two encodings, and formats that cannot be rendered to
        // ------------------------------------------------------------------------------------------

        /// 3.8.5: CopyTexImage2D raises INVALID_OPERATION when the read buffer is LINEAR and internalformat is sRGB, or the read buffer is SRGB
        /// and internalformat is not sRGB.
        void check_copy_rules(demo::Check& check) const
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            auto copy_error = [&](mgl::FramebufferId read, mgl::InternalFormat internal) {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, read);
                demo::drain_gl_errors();
                mgl::glCopyTexImage2D(mgl::TextureTarget::Texture2D, 0, internal, 0, 0, 16, 16, 0);
                return demo::drain_gl_errors();
            };
            const mgl::ErrorCode lin_to_srgb = copy_error(fbo_[kLinRbFbo], mgl::InternalFormat::Srgb8Alpha8);
            check.expect(lin_to_srgb == mgl::ErrorCode::InvalidOperation, "glCopyTexImage2D from a LINEAR read buffer into an sRGB internal format (SRGB8_ALPHA8) -> INVALID_OPERATION (3.8.5)",
                         std::format("glGetError() = {}", demo::error_name(lin_to_srgb)));
            const mgl::ErrorCode srgb_to_lin = copy_error(fbo_[kSrgbRbFbo], mgl::InternalFormat::Rgba8);
            check.expect(srgb_to_lin == mgl::ErrorCode::InvalidOperation, "glCopyTexImage2D from an SRGB read buffer into a non-sRGB internal format (RGBA8) -> INVALID_OPERATION (3.8.5)",
                         std::format("glGetError() = {}", demo::error_name(srgb_to_lin)));
            const mgl::ErrorCode srgb_to_srgb = copy_error(fbo_[kSrgbRbFbo], mgl::InternalFormat::Srgb8Alpha8);
            check.expect(srgb_to_srgb == mgl::ErrorCode::NoError, "positive control: glCopyTexImage2D from an SRGB read buffer into SRGB8_ALPHA8 is accepted", std::format("glGetError() = {}", demo::error_name(srgb_to_srgb)));
            const mgl::ErrorCode lin_to_lin = copy_error(fbo_[kLinRbFbo], mgl::InternalFormat::Rgba8);
            check.expect(lin_to_lin == mgl::ErrorCode::NoError, "positive control: glCopyTexImage2D from a LINEAR read buffer into RGBA8 is accepted", std::format("glGetError() = {}", demo::error_name(lin_to_lin)));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteTextures(1, &t);
        }

        /// Table 3.13: SRGB8 is texture-filterable but not color-renderable; SRGB8_ALPHA8 is both.
        void check_not_renderable(demo::Check& check) const
        {
            mgl::RenderbufferId rb{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            demo::drain_gl_errors();
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Srgb8, 4, 4);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glRenderbufferStorage(SRGB8) -> INVALID_ENUM: SRGB8 is not color-renderable (Table 3.13, 4.4.2.1; ES 3.2 9.2.4 names the error)");
            mgl::glDeleteRenderbuffers(1, &rb);

            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), mgl::TextureTarget::Texture2D, tex_[kSrgb8], 0);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            check.expect(status == mgl::FramebufferStatus::IncompleteAttachment, "a framebuffer with an SRGB8 texture as color attachment is FRAMEBUFFER_INCOMPLETE_ATTACHMENT (not color-renderable, 4.4.4.1 and 4.4.4.2)",
                         std::format("status = {}", demo::framebuffer_status_name(status)));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            demo::drain_gl_errors();
        }

        mgl::ProgramId exact_{}, show_{}, ramp_{}, color_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        std::array<mgl::RenderbufferId, kRbCount> rb_{};
        std::array<mgl::FramebufferId, kFboCount> fbo_{};
        bool default_srgb_ = false;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-srgb",
        .title = "sRGB textures and sRGB render targets",
        .min_es = {3, 0},
        .summary = "SRGB8 / SRGB8_ALPHA8 textures decode to linear on sampling (equation 3.26), SRGB8_ALPHA8 render targets encode on write and blend in linear space "
                   "(4.1.7, 4.1.8), clears and blits follow, FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING tells which applies; a gamma-ramp picture and exact checks.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Srgb>(argc, argv, info);
}
