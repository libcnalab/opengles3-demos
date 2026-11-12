// es31-multisample-textures
//
// Multisample textures and the sample mask (OpenGL ES 3.2 sections 8.8 and 13.8.3, GLSL ES 3.10 sections 8.9.1 and 8.9.2):
// glTexStorage2DMultisample, rendering into a TEXTURE_2D_MULTISAMPLE attachment, reading every sample back with
// texelFetch(sampler2DMS / isampler2DMS / usampler2DMS, texel, sample), textureSize, glSampleMaski + SAMPLE_MASK,
// glGetMultisamplefv, and the limits and queries around them.
//
// The positions of the samples inside a pixel, and which of them a triangle edge covers, are implementation-defined, so nothing here
// depends on them. Every sample is instead given a known content by the sample mask: pattern k (a color that depends on the pixel
// and on k) is drawn over the whole texture with only sample k enabled in the mask. Reading sample k back must then return
// pattern k at every texel, exactly, for every sample count and format.
//
// The window is divided into four equal cells (boundaries at width / 2 and height / 2), one per multisample texture; a cell is
// a grid of sub-cells, sub-cell k showing sample k with texelFetch:
//   top-left   RGBA8, fixedsamplelocations = TRUE   | top-right    RGBA8, fixedsamplelocations = FALSE
//   bottom-left RGBA8UI (usampler2DMS)               | bottom-right  RGBA8I  (isampler2DMS)
//
// Validation:
//   * the limits MAX_SAMPLE_MASK_WORDS, MAX_COLOR_TEXTURE_SAMPLES, MAX_DEPTH_TEXTURE_SAMPLES, MAX_INTEGER_SAMPLES, MAX_SAMPLES and
//     glGetInternalformativ(TEXTURE_2D_MULTISAMPLE) for twelve formats: counts descending and positive, the maximum at least the
//     limit that section 20.3.1 names; allocation with every reported count and with the maximum;
//   * the number of samples of an allocated texture (SAMPLES of a framebuffer that has it attached) between the request and the next
//     supported count; the texture's size and format from glGetTexLevelParameteriv; the sample count and fixedsamplelocations made
//     visible by framebuffer completeness (section 9.4.2);
//   * glGetMultisamplefv: positions in [0, 1], the INVALID_VALUE for index >= SAMPLES, identical positions for two fixedsamplelocations
//     textures of different size and format;
//   * every sample of RGBA8 (every count, both values of fixedsamplelocations), RGBA8UI and RGBA8I, exactly;
//   * the sample mask: initial state, masks 0 / all / alternating / out-of-range bits, the enable flag, the effect of a mask on a
//     triangle's partial coverage, and the rule that a single-sample framebuffer ignores it;
//   * textureSize for three samplers and sizes; depth samples read back with a sampler2DMS;
//   * the errors of ES 3.2 sections 8.1, 8.5, 8.8, 8.10, 8.11, 9.2, 13.8.3 and 20.3.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int kW = 16;           // texels of the main multisample textures
    constexpr int kH = 12;
    constexpr int kTri = 32;         // edge of the texture the triangle is drawn into
    constexpr int kStripW = 1024;    // the single-sample targets that the samples are fetched into: W * samples <= 1024
    constexpr int kStripH = 32;
    constexpr int kMinWindow = 64;   // smallest surface: the cells are then 32 x 32, at least 4 pixels per sub-cell side for up to 64 samples

    /// Window cell `index` (0 = top left, row by row) as a rectangle: the four cells partition the surface exactly.
    struct CellRect
    {
        int x0 = 0, y0 = 0, w = 0, h = 0;
    };
    CellRect cell_rect(int index, int width, int height)
    {
        const int col = index % 2, row = index / 2;
        const int x0 = col * width / 2, x1 = (col + 1) * width / 2;
        const int y0 = (1 - row) * height / 2, y1 = (2 - row) * height / 2;  // row 0 is the top half
        return {x0, y0, x1 - x0, y1 - y0};
    }
    constexpr int kIdA = 40;         // two pattern numbers that no per-sample pattern (0 .. 31) uses
    constexpr int kIdB = 90;

    enum class Kind
    {
        Float,
        Int,
        Uint
    };

    // ---------------------------------------------------------------------------------------------
    // The pattern: the same arithmetic as shaders/pattern.frag.
    // ---------------------------------------------------------------------------------------------
    using Pattern = std::array<int, 4>;

    Pattern pattern(int id, int x, int y)
    {
        return {(16 * x + 3 * id) & 255, (20 * y + 5 * id) & 255, (97 * id + 13 * (x + y)) & 255, (200 + 7 * id + 3 * y) & 255};
    }

    /// A row of strips (see shaders/fetch.frag), as pattern values 0..255 whatever the kind of the texture.
    struct Strips
    {
        int w = 0, h = 0;
        std::vector<Pattern> px;
        Strips() = default;
        Strips(int width, int height) : w(width), h(height), px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {}
        Pattern& at(int x, int y) { return px[static_cast<std::size_t>(y * w + x)]; }
        const Pattern& at(int x, int y) const { return px[static_cast<std::size_t>(y * w + x)]; }
    };

    /// What the strips show when sample j holds pattern ids[j] everywhere.
    Strips expected_strips(int texel_w, int texel_h, const std::vector<int>& ids)
    {
        Strips s(texel_w * static_cast<int>(ids.size()), texel_h);
        for (std::size_t j = 0; j < ids.size(); ++j)
            for (int y = 0; y < texel_h; ++y)
                for (int x = 0; x < texel_w; ++x) s.at(static_cast<int>(j) * texel_w + x, y) = pattern(ids[j], x, y);
        return s;
    }

    struct StripDiff
    {
        int bad = 0;
        std::string first;
    };

    StripDiff compare_strips(const Strips& got, const Strips& want, int texel_w)
    {
        StripDiff d;
        if (got.w != want.w || got.h != want.h)
        {
            d.bad = 1;
            d.first = std::format("sizes differ: {}x{} vs {}x{}", got.w, got.h, want.w, want.h);
            return d;
        }
        for (int y = 0; y < got.h; ++y)
            for (int x = 0; x < got.w; ++x)
                if (got.at(x, y) != want.at(x, y) && d.bad++ == 0)
                {
                    const Pattern& g = got.at(x, y);
                    const Pattern& w = want.at(x, y);
                    d.first = std::format("first at sample {} texel ({},{}): got ({}, {}, {}, {}) expected ({}, {}, {}, {})", x / texel_w, x % texel_w, y, g[0], g[1],
                                          g[2], g[3], w[0], w[1], w[2], w[3]);
                }
        return d;
    }

    // ---------------------------------------------------------------------------------------------
    // Texture formats
    // ---------------------------------------------------------------------------------------------
    struct FormatInfo
    {
        mgl::SizedInternalFormat sized;
        mgl::InternalFormat query;  // the same enumerant for glGetInternalformativ
        const char* name;
        bool integer;
        bool depth;   // depth or depth/stencil
    };

    constexpr std::array<FormatInfo, 12> kFormats = {{
        {mgl::SizedInternalFormat::Rgba8, mgl::InternalFormat::Rgba8, "RGBA8", false, false},
        {mgl::SizedInternalFormat::Rgba8UI, mgl::InternalFormat::Rgba8UI, "RGBA8UI", true, false},
        {mgl::SizedInternalFormat::Rgba8I, mgl::InternalFormat::Rgba8I, "RGBA8I", true, false},
        {mgl::SizedInternalFormat::Rgb10A2, mgl::InternalFormat::Rgb10A2, "RGB10_A2", false, false},
        {mgl::SizedInternalFormat::Rg8, mgl::InternalFormat::Rg8, "RG8", false, false},
        {mgl::SizedInternalFormat::R8, mgl::InternalFormat::R8, "R8", false, false},
        {mgl::SizedInternalFormat::Srgb8Alpha8, mgl::InternalFormat::Srgb8Alpha8, "SRGB8_ALPHA8", false, false},
        {mgl::SizedInternalFormat::DepthComponent16, mgl::InternalFormat::DepthComponent16, "DEPTH_COMPONENT16", false, true},
        {mgl::SizedInternalFormat::DepthComponent24, mgl::InternalFormat::DepthComponent24, "DEPTH_COMPONENT24", false, true},
        {mgl::SizedInternalFormat::DepthComponent32F, mgl::InternalFormat::DepthComponent32F, "DEPTH_COMPONENT32F", false, true},
        {mgl::SizedInternalFormat::Depth24Stencil8, mgl::InternalFormat::Depth24Stencil8, "DEPTH24_STENCIL8", false, true},
        {mgl::SizedInternalFormat::Depth32FStencil8, mgl::InternalFormat::Depth32FStencil8, "DEPTH32F_STENCIL8", false, true},
    }};

    constexpr std::size_t kRgba8 = 0, kRgba8UI = 1, kRgba8I = 2, kDepth24 = 8;

    // ---------------------------------------------------------------------------------------------
    // meta-gl has the typed bitfield SampleMaskValue (the type glSampleMaski takes) but only the enumerators None and All. A
    // mask that has some bits set is another value of the same bitfield type: this is the only place that makes one.
    // (Reported as a meta-gl gap in the final report: an enumerator or helper for single sample bits is missing.)
    // ---------------------------------------------------------------------------------------------
    mgl::SampleMaskValue mask_value(std::uint32_t bits) { return static_cast<mgl::SampleMaskValue>(bits); }

    using Defines = std::vector<std::pair<std::string, std::string>>;

    std::string with_defines(std::string source, const Defines& defines)
    {
        std::string inserted;
        for (const auto& [name, value] : defines) inserted += std::format("#define {} {}\n", name, value);
        const std::size_t end_of_version = source.find('\n');
        source.insert(end_of_version + 1, inserted);
        return source;
    }

    enum class Role
    {
        Fill,
        FillTriangle,
        Fetch,
        Show,
        Size
    };

    struct Program
    {
        mgl::ProgramId id{};
        mgl::UniformLocation ms{}, sample{}, texels{}, z{}, tri{}, size{}, origin{}, cell{}, samples{}, cols{};
    };

    /// One multisample texture with a framebuffer that has it attached (color 0, or the depth attachment).
    struct Ms
    {
        mgl::TextureId tex{};
        mgl::FramebufferId fbo{};
        mgl::SizedInternalFormat format = mgl::SizedInternalFormat::Rgba8;
        Kind kind = Kind::Float;
        bool depth = false;
        bool fixed = true;
        int requested = 0;
        int samples = 0;    // SAMPLES of the framebuffer: the texture's TEXTURE_SAMPLES (section 9.2.3.1)
        int w = 0, h = 0;
        mgl::FramebufferStatus status = mgl::FramebufferStatus::Complete;
    };

    struct TriangleGeometry
    {
        std::array<float, 6> v;
        /// > 0 on the inner side of edge i (the corners are counter-clockwise), in pixels.
        double distance(std::size_t i, double x, double y) const
        {
            const std::size_t j = (i + 1) % 3;
            const double ax = static_cast<double>(v[2 * i]), ay = static_cast<double>(v[2 * i + 1]);
            const double bx = static_cast<double>(v[2 * j]), by = static_cast<double>(v[2 * j + 1]);
            return ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / std::hypot(bx - ax, by - ay);
        }
        double area() const
        {
            return 0.5 * (static_cast<double>(v[2] - v[0]) * static_cast<double>(v[5] - v[1]) - static_cast<double>(v[4] - v[0]) * static_cast<double>(v[3] - v[1]));
        }
    };
    constexpr TriangleGeometry kTriangle = {{3.3f, 2.6f, 29.1f, 7.4f, 12.2f, 30.4f}};

    class MultisampleTextures final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kMinWindow || env.height < kMinWindow)
                throw demo::Skip(std::format("the four sample cells need a default framebuffer of at least {0}x{0} pixels", kMinWindow));

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glDisable(mgl::Capability::Dither);  // 8-bit results are compared exactly

            // The state that must hold before any glSampleMaski call (section 21: SAMPLE_MASK is disabled, every mask word is all ones).
            mask_words_ = demo::get_integer(mgl::GetParameter::MaxSampleMaskWords);
            initial_enabled_ = mgl::glIsEnabled(mgl::Capability::SampleMask);
            for (int w = 0; w < mask_words_; ++w)
            {
                GLint v = 0;
                mgl::glGetIntegeri_v(mgl::GetParameter::SampleMaskValue, static_cast<GLuint>(w), &v);
                initial_mask_.push_back(v);
            }

            // Sample counts per format (descending, as glGetInternalformativ returns them).
            for (const FormatInfo& f : kFormats) counts_.push_back(sample_counts(f.query));
            if (counts_[kRgba8].empty() || counts_[kRgba8UI].empty() || counts_[kRgba8I].empty() || counts_[kDepth24].empty())
                throw demo::Skip("the implementation reports no multisample texture support for RGBA8 / RGBA8UI / RGBA8I / DEPTH_COMPONENT24");

            vertex_fullscreen_ = env.load_text("shaders/fullscreen.vert");
            vertex_triangle_ = env.load_text("shaders/triangle.vert");
            fragment_[0] = env.load_text("shaders/pattern.frag");
            fragment_[1] = env.load_text("shaders/fetch.frag");
            fragment_[2] = env.load_text("shaders/show.frag");
            fragment_[3] = env.load_text("shaders/size.frag");

            // Single-sample targets that the samples are fetched into, one per kind of result.
            const std::array<mgl::SizedInternalFormat, 3> strip_formats = {mgl::SizedInternalFormat::Rgba8, mgl::SizedInternalFormat::Rgba32I,
                                                                           mgl::SizedInternalFormat::Rgba32UI};
            for (std::size_t k = 0; k < 3; ++k)
            {
                mgl::glGenTextures(1, &strip_tex_[k]);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, strip_tex_[k]);
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, strip_formats[k], kStripW, kStripH);
                mgl::glGenFramebuffers(1, &strip_fbo_[k]);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, strip_fbo_[k]);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2D, strip_tex_[k], 0);
                if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                    throw demo::Failure("a single-sample result framebuffer is incomplete");
            }

            // The multisample textures of the window and of the later checks.
            ms_fixed_ = make_ms(kRgba8, pick(kRgba8), true, kW, kH);
            ms_free_ = make_ms(kRgba8, pick(kRgba8), false, kW, kH);
            ms_uint_ = make_ms(kRgba8UI, pick(kRgba8UI), true, kW, kH);
            ms_int_ = make_ms(kRgba8I, pick(kRgba8I), true, kW, kH);
            ms_tri_ = make_ms(kRgba8, pick(kRgba8), true, kTri, kTri);
            for (const Ms* m : {&ms_fixed_, &ms_free_, &ms_uint_, &ms_int_, &ms_tri_})
                if (m->status != mgl::FramebufferStatus::Complete) throw demo::Failure("a multisample texture framebuffer is incomplete");

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            for (Ms* m : {&ms_fixed_, &ms_free_, &ms_uint_, &ms_int_}) fill_per_sample(*m);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            const std::array<const Ms*, 4> cells = {&ms_fixed_, &ms_free_, &ms_uint_, &ms_int_};
            for (std::size_t c = 0; c < cells.size(); ++c) show(*cells[c], cell_rect(static_cast<int>(c), env.width, env.height), static_cast<int>(c));
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_window(env, check);
            check_limits(check);
            check_internalformat_queries(check);
            check_allocation(check);
            check_completeness(check);
            check_sample_positions(check);
            check_every_sample(check);
            check_sample_mask(check);
            check_triangle_coverage(check);
            check_texture_size(check);
            check_depth_samples(check);
            check_errors(check);
            check_negative_controls(check);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (auto& [key, p] : programs_) mgl::glDeleteProgram(p.id);
            programs_.clear();
            for (Ms* m : {&ms_fixed_, &ms_free_, &ms_uint_, &ms_int_, &ms_tri_}) destroy(*m);
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(strip_fbo_.size()), strip_fbo_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(strip_tex_.size()), strip_tex_.data());
            mgl::glDeleteVertexArrays(1, &vao_);
        }

    private:
        // ==========================================================================================
        // Objects
        // ==========================================================================================
        static std::vector<GLint> sample_counts(mgl::InternalFormat format)
        {
            GLint n = 0;
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, format, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
            std::vector<GLint> counts(static_cast<std::size_t>(std::max(n, 0)));
            if (n > 0)
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, format, mgl::InternalFormatParameter::Samples, n, counts.data());
            return counts;
        }

        /// The sample count the demo uses for a format: the smallest reported count that is at least 4 (the maximum if none is).
        int pick(std::size_t format) const
        {
            const std::vector<GLint>& counts = counts_[format];
            int chosen = counts.front();
            for (const GLint c : counts)
                if (c >= 4) chosen = c;  // descending order: the last one >= 4 is the smallest
            return chosen;
        }

        static Kind kind_of(const FormatInfo& f)
        {
            return f.depth ? Kind::Float : f.sized == mgl::SizedInternalFormat::Rgba8UI ? Kind::Uint : f.sized == mgl::SizedInternalFormat::Rgba8I ? Kind::Int : Kind::Float;
        }

        /// glTexStorage2DMultisample plus a framebuffer with the texture attached. Nothing is checked here: the caller looks at
        /// `status`, `samples` and glGetError.
        Ms make_ms(std::size_t format_index, int requested, bool fixed, int w, int h)
        {
            const FormatInfo& f = kFormats[format_index];
            Ms m;
            m.format = f.sized;
            m.kind = kind_of(f);
            m.depth = f.depth;
            m.fixed = fixed;
            m.requested = requested;
            m.w = w;
            m.h = h;
            mgl::glGenTextures(1, &m.tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, m.tex);
            mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2DMultisample, requested, f.sized, w, h, fixed ? GL_TRUE : GL_FALSE);
            mgl::glGenFramebuffers(1, &m.fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m.fbo);
            if (f.depth)
            {
                const mgl::FramebufferAttachment attachment =
                    f.sized == mgl::SizedInternalFormat::Depth24Stencil8 || f.sized == mgl::SizedInternalFormat::Depth32FStencil8
                        ? mgl::FramebufferAttachment::DepthStencil : mgl::FramebufferAttachment::Depth;
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, attachment, mgl::TextureTarget::Texture2DMultisample, m.tex, 0);
                const std::array<mgl::DrawBuffer, 1> none = {mgl::DrawBuffer::None};
                mgl::glDrawBuffers(none);
                mgl::glReadBuffer(mgl::ReadBuffer::None);
            }
            else
            {
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2DMultisample, m.tex, 0);
            }
            m.status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            m.samples = demo::get_integer(mgl::GetParameter::Samples);
            return m;
        }

        static void destroy(Ms& m)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (m.fbo.value != 0) mgl::glDeleteFramebuffers(1, &m.fbo);
            if (m.tex.value != 0) mgl::glDeleteTextures(1, &m.tex);
            m = Ms{};
        }

        const Program& program(Role role, Kind kind)
        {
            const std::string key = std::format("{}:{}", static_cast<int>(role), static_cast<int>(kind));
            auto it = programs_.find(key);
            if (it != programs_.end()) return it->second;

            Defines d;
            d.emplace_back("SAMPLER", kind == Kind::Float ? "highp sampler2DMS" : kind == Kind::Int ? "highp isampler2DMS" : "highp usampler2DMS");
            d.emplace_back("GVEC4", kind == Kind::Float ? "vec4" : kind == Kind::Int ? "ivec4" : "uvec4");
            d.emplace_back(kind == Kind::Float ? "KIND_FLOAT" : kind == Kind::Int ? "KIND_INT" : "KIND_UINT", "1");
            const std::size_t fragment = role == Role::Fill || role == Role::FillTriangle ? 0 : role == Role::Fetch ? 1 : role == Role::Show ? 2 : 3;
            Program p;
            p.id = demo::build_program_from_source(role == Role::FillTriangle ? vertex_triangle_ : vertex_fullscreen_, with_defines(fragment_[fragment], d), key);
            p.ms = mgl::glGetUniformLocation(p.id, "u_ms");
            p.sample = mgl::glGetUniformLocation(p.id, "u_sample");
            p.texels = mgl::glGetUniformLocation(p.id, "u_texels");
            p.z = mgl::glGetUniformLocation(p.id, "u_z");
            p.tri = mgl::glGetUniformLocation(p.id, "u_tri");
            p.size = mgl::glGetUniformLocation(p.id, "u_size");
            p.origin = mgl::glGetUniformLocation(p.id, "u_origin");
            p.cell = mgl::glGetUniformLocation(p.id, "u_cell");
            p.samples = mgl::glGetUniformLocation(p.id, "u_samples");
            p.cols = mgl::glGetUniformLocation(p.id, "u_cols");
            return programs_.emplace(key, p).first->second;
        }

        // ==========================================================================================
        // Writing: pattern draws into a multisample framebuffer
        // ==========================================================================================
        /// Sets the sample mask to word 0 = `bits`, every other word 0.
        void set_mask(std::uint32_t bits) const
        {
            for (int w = 0; w < mask_words_; ++w) mgl::glSampleMaski(static_cast<GLuint>(w), mask_value(w == 0 ? bits : 0u));
        }

        /// The mask that lets exactly sample k through (the sample numbering of section 13.8.3: bit B of word M is sample 32 M + B).
        void set_mask_only(int k) const
        {
            for (int w = 0; w < mask_words_; ++w) mgl::glSampleMaski(static_cast<GLuint>(w), mask_value(w == k / 32 ? 1u << (k % 32) : 0u));
        }

        void restore_mask() const
        {
            for (int w = 0; w < mask_words_; ++w) mgl::glSampleMaski(static_cast<GLuint>(w), mgl::SampleMaskValue::All);
        }

        void draw_pattern(const Ms& m, int id, float depth_ndc = 0.0f)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m.fbo);
            mgl::glViewport(0, 0, m.w, m.h);
            const Program& p = program(Role::Fill, m.kind);
            mgl::glUseProgram(p.id);
            mgl::glUniform1i(p.sample, id);
            mgl::glUniform1f(p.z, depth_ndc);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        void draw_triangle(const Ms& m, int id)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m.fbo);
            mgl::glViewport(0, 0, m.w, m.h);
            const Program& p = program(Role::FillTriangle, m.kind);
            mgl::glUseProgram(p.id);
            mgl::glUniform1i(p.sample, id);
            mgl::glUniform2fv(p.tri, 3, kTriangle.v.data());
            mgl::glUniform1f(p.size, static_cast<float>(m.w));
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        /// Every sample of the texture := pattern `id` (SAMPLE_MASK disabled).
        void fill_all(const Ms& m, int id)
        {
            mgl::glDisable(mgl::Capability::SampleMask);
            draw_pattern(m, id);
        }

        /// Sample k := pattern k, for every sample: pattern k drawn over the whole texture with only sample k in the mask.
        void fill_per_sample(const Ms& m)
        {
            mgl::glEnable(mgl::Capability::SampleMask);
            for (int k = 0; k < m.samples && k < 32 * mask_words_; ++k)
            {
                set_mask_only(k);
                draw_pattern(m, k);
            }
            mgl::glDisable(mgl::Capability::SampleMask);
            restore_mask();
        }

        // ==========================================================================================
        // Reading: texelFetch of every sample into a single-sample target
        // ==========================================================================================
        static std::size_t index_of(Kind k) { return static_cast<std::size_t>(k); }

        Strips fetch(const Ms& m)
        {
            const std::size_t k = index_of(m.kind);
            const int n = m.samples;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, strip_fbo_[k]);
            clear_sentinel(m.kind);
            mgl::glViewport(0, 0, m.w * n, m.h);
            const Program& p = program(Role::Fetch, m.kind);
            mgl::glUseProgram(p.id);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, m.tex);
            mgl::glUniform1i(p.ms, 0);
            mgl::glUniform2i(p.texels, m.w, m.h);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            return read_strips(m.kind, m.w * n, m.h);
        }

        static void clear_sentinel(Kind kind)
        {
            if (kind == Kind::Float)
            {
                const std::array<GLfloat, 4> v = {0.30f, 0.60f, 0.90f, 0.50f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, v.data());
            }
            else if (kind == Kind::Int)
            {
                const std::array<GLint, 4> v = {-7777, 7777, -7777, 7777};
                mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 0, v.data());
            }
            else
            {
                const std::array<GLuint, 4> v = {0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
                mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, v.data());
            }
        }

        /// Reads the bound single-sample framebuffer and converts to pattern values (the signed kind stores pattern - 128).
        static Strips read_strips(Kind kind, int w, int h)
        {
            Strips s(w, h);
            const std::size_t count = static_cast<std::size_t>(w * h);
            if (kind == Kind::Float)
            {
                std::vector<std::uint8_t> raw(count * 4);
                mgl::glReadPixels(0, 0, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, raw.data());
                for (std::size_t i = 0; i < count; ++i) s.px[i] = {raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]};
            }
            else if (kind == Kind::Int)
            {
                std::vector<std::int32_t> raw(count * 4);
                mgl::glReadPixels(0, 0, w, h, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int, raw.data());
                for (std::size_t i = 0; i < count; ++i) s.px[i] = {raw[4 * i] + 128, raw[4 * i + 1] + 128, raw[4 * i + 2] + 128, raw[4 * i + 3] + 128};
            }
            else
            {
                std::vector<std::uint32_t> raw(count * 4);
                mgl::glReadPixels(0, 0, w, h, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, raw.data());
                for (std::size_t i = 0; i < count; ++i)
                    s.px[i] = {static_cast<int>(raw[4 * i]), static_cast<int>(raw[4 * i + 1]), static_cast<int>(raw[4 * i + 2]), static_cast<int>(raw[4 * i + 3])};
            }
            return s;
        }

        /// Window cell `index` (0 = top left, row by row): the samples of `m` as a grid of sub-cells.
        void show(const Ms& m, CellRect r, int index)
        {
            const int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(m.samples))));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(r.x0, r.y0, r.w, r.h);
            const Program& p = program(Role::Show, m.kind);
            mgl::glUseProgram(p.id);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, m.tex);
            mgl::glUniform1i(p.ms, 0);
            mgl::glUniform2i(p.texels, m.w, m.h);
            mgl::glUniform2i(p.origin, r.x0, r.y0);
            mgl::glUniform2i(p.cell, r.w, r.h);
            mgl::glUniform1i(p.samples, m.samples);
            mgl::glUniform1i(p.cols, cols);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            shown_cols_[static_cast<std::size_t>(index)] = cols;
        }

        // ==========================================================================================
        // Checks
        // ==========================================================================================
        template <typename F>
        static bool expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
        {
            (void)demo::drain_gl_errors();  // a stale error must not be mistaken for the one under test
            call();
            return check.expect_gl_error(code, what);
        }

        /// expect_error with a documented driver deviation (Check::expect_or_known_deviation).
        template <typename F>
        static bool expect_error_or_known_deviation(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call,
                                                    std::string_view renderer, std::string_view reason)
        {
            (void)demo::drain_gl_errors();
            call();
            const mgl::ErrorCode e = demo::drain_gl_errors();
            return check.expect_or_known_deviation(e == code, what,
                                                   std::format("glGetError() = {}, expected {}", demo::error_name(e), demo::error_name(code)), renderer,
                                                   reason);
        }

        bool expect_strips(demo::Check& check, std::string_view what, const Ms& m, const std::vector<int>& ids)
        {
            const Strips got = fetch(m);
            const StripDiff d = compare_strips(got, expected_strips(m.w, m.h, ids), m.w);
            return check.expect(d.bad == 0, what,
                                d.bad == 0 ? std::format("{} samples x {} texels, every value exact", m.samples, m.w * m.h)
                                           : std::format("{} of {} values differ, {}", d.bad, got.w * got.h, d.first));
        }

        static std::vector<int> iota_ids(int n)
        {
            std::vector<int> ids(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) ids[static_cast<std::size_t>(i)] = i;
            return ids;
        }

        void check_window(const demo::Env& env, demo::Check& check)
        {
            // Replays the arithmetic of shaders/show.frag on the CPU for every pixel of the four cells.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const int width = env.width;
            const std::vector<std::uint8_t> window = demo::read_rgba8(0, 0, width, env.height);
            const std::array<const Ms*, 4> cells = {&ms_fixed_, &ms_free_, &ms_uint_, &ms_int_};
            int bad = 0, compared = 0;
            std::string first;
            for (std::size_t c = 0; c < cells.size(); ++c)
            {
                const Ms& m = *cells[c];
                const CellRect r = cell_rect(static_cast<int>(c), width, env.height);
                const int cols = shown_cols_[c];
                const int sub_w = r.w / cols, sub_h = r.h / cols;
                for (int ly = 0; ly < r.h; ++ly)
                    for (int lx = 0; lx < r.w; ++lx)
                    {
                        ++compared;
                        const int sx = lx / sub_w, sy = ly / sub_h;
                        const int k = sy * cols + sx;
                        std::array<int, 3> want = {0, 0, 0};
                        if (sx < cols && sy < cols && k < m.samples)
                        {
                            const int tx = std::min(((lx - sx * sub_w) * m.w) / sub_w, m.w - 1);
                            const int ty = std::min(((ly - sy * sub_h) * m.h) / sub_h, m.h - 1);
                            const Pattern pat = pattern(k, tx, ty);
                            want = {pat[0], pat[1], pat[2]};
                        }
                        const std::size_t at = (static_cast<std::size_t>(r.y0 + ly) * static_cast<std::size_t>(width) + static_cast<std::size_t>(r.x0 + lx)) * 4;
                        const bool ok = window[at] == want[0] && window[at + 1] == want[1] && window[at + 2] == want[2] && window[at + 3] == 255;
                        if (!ok && bad++ == 0)
                            first = std::format("cell {} pixel ({},{}): got {} expected ({}, {}, {}, 255)", c, lx, ly,
                                                demo::format_rgba({window[at], window[at + 1], window[at + 2], window[at + 3]}), want[0], want[1], want[2]);
                    }
            }
            check.expect(bad == 0, "the four window cells show, for every sample of every texel, the value that sample was given (texelFetch of sampler2DMS / usampler2DMS / isampler2DMS)",
                         bad == 0 ? std::format("{} pixels compared exactly (every pixel of the window)", compared) : std::format("{} of {} pixels differ, {}", bad, compared, first));
        }

        void check_limits(demo::Check& check)
        {
            const int mask_words = demo::get_integer(mgl::GetParameter::MaxSampleMaskWords);
            const int color = demo::get_integer(mgl::GetParameter::MaxColorTextureSamples);
            const int depth = demo::get_integer(mgl::GetParameter::MaxDepthTextureSamples);
            const int integer = demo::get_integer(mgl::GetParameter::MaxIntegerSamples);
            const int max_samples = demo::get_integer(mgl::GetParameter::MaxSamples);
            std::cout << std::format("  limits: MAX_SAMPLE_MASK_WORDS={} MAX_COLOR_TEXTURE_SAMPLES={} MAX_DEPTH_TEXTURE_SAMPLES={} MAX_INTEGER_SAMPLES={} MAX_SAMPLES={}\n",
                                     mask_words, color, depth, integer, max_samples);
            // ES 3.2 chapter 21 (implementation dependent values): all of them are positive, and the minimum values are 1, 1, 1, 1 and 4.
            check.expect(mask_words >= 1 && color >= 1 && depth >= 1 && integer >= 1 && max_samples >= 4,
                         "MAX_SAMPLE_MASK_WORDS, MAX_COLOR_TEXTURE_SAMPLES, MAX_DEPTH_TEXTURE_SAMPLES, MAX_INTEGER_SAMPLES >= 1 and MAX_SAMPLES >= 4 (ES 3.2 chapter 21, implementation dependent values)",
                         std::format("{}, {}, {}, {}, {}", mask_words, color, depth, integer, max_samples));
            check.expect_no_gl_error("no error from the limit queries");
        }

        void check_internalformat_queries(demo::Check& check)
        {
            const int color = demo::get_integer(mgl::GetParameter::MaxColorTextureSamples);
            const int depth = demo::get_integer(mgl::GetParameter::MaxDepthTextureSamples);
            const int integer = demo::get_integer(mgl::GetParameter::MaxIntegerSamples);

            for (std::size_t f = 0; f < kFormats.size(); ++f)
            {
                const FormatInfo& fmt = kFormats[f];
                const std::vector<GLint>& counts = counts_[f];
                GLint n = -1;
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, fmt.query, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
                bool descending = !counts.empty();
                for (std::size_t i = 0; i < counts.size(); ++i) descending = descending && counts[i] > 0 && (i == 0 || counts[i] < counts[i - 1]);
                // Section 20.3.1: the maximum is at least the lowest of the limits that apply to the format.
                const int bound = fmt.depth ? depth : fmt.integer ? std::min(integer, color) : color;
                const int max_count = counts.empty() ? 0 : counts.front();
                check.expect(n >= 1 && static_cast<std::size_t>(n) == counts.size() && descending && max_count >= bound,
                             std::format("glGetInternalformativ(TEXTURE_2D_MULTISAMPLE, {}): {} sample counts, positive and descending, the largest {} >= {}", fmt.name, n, max_count,
                                         fmt.depth ? "MAX_DEPTH_TEXTURE_SAMPLES" : fmt.integer ? "min(MAX_INTEGER_SAMPLES, MAX_COLOR_TEXTURE_SAMPLES)" : "MAX_COLOR_TEXTURE_SAMPLES"),
                             std::format("{} counts, largest {}, bound {}", n, max_count, bound));
            }

            // count limits how many values are written; the rest of the array is untouched (section 20.3).
            std::array<GLint, 3> sentinel = {-1, -1, -1};
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, 1, sentinel.data());
            check.expect(sentinel[0] == counts_[kRgba8].front() && sentinel[1] == -1 && sentinel[2] == -1,
                         "glGetInternalformativ(SAMPLES) with count 1 writes only the maximum (the first, largest value) and nothing else",
                         std::format("({}, {}, {}), maximum {}", sentinel[0], sentinel[1], sentinel[2], counts_[kRgba8].front()));
            sentinel = {-1, -1, -1};
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, 0, sentinel.data());
            check.expect(sentinel[0] == -1, "glGetInternalformativ with count 0 writes nothing");
            check.expect_no_gl_error("no error from the internal format queries");
        }

        /// Allocation with every reported count and with the maximum, of every format: no error, a complete framebuffer, and the number of
        /// samples between the request and the next larger supported count (section 8.8).
        void check_allocation(demo::Check& check)
        {
            for (std::size_t f = 0; f < kFormats.size(); ++f)
            {
                const std::vector<GLint>& counts = counts_[f];
                int bad = 0;
                std::string first;
                std::string measured;
                for (std::size_t i = 0; i < counts.size(); ++i)
                {
                    (void)demo::drain_gl_errors();
                    Ms m = make_ms(f, counts[i], true, 8, 8);
                    const mgl::ErrorCode error = demo::drain_gl_errors();
                    // "no more than the next larger sample count supported": counts is descending, so that is counts[i - 1] (the maximum if none).
                    const int upper = i == 0 ? counts[0] : counts[i - 1];
                    const bool ok = error == mgl::ErrorCode::NoError && m.status == mgl::FramebufferStatus::Complete && m.samples >= counts[i] && m.samples <= upper;
                    if (!ok && bad++ == 0)
                        first = std::format("request {}: error {}, status {}, SAMPLES {} (allowed {}..{})", counts[i], demo::error_name(error),
                                            demo::framebuffer_status_name(m.status), m.samples, counts[i], upper);
                    measured += std::format("{}{}", measured.empty() ? "" : ",", m.samples);
                    destroy(m);
                }
                check.expect(bad == 0 && !counts.empty(),
                             std::format("glTexStorage2DMultisample({}) with each of its {} reported sample counts: no error, framebuffer complete, TEXTURE_SAMPLES (= SAMPLES) between the request and the next larger count",
                                         kFormats[f].name, counts.size()),
                             bad == 0 ? std::format("requested {}, allocated {}", join(counts), measured) : first);
            }
        }

        static std::string join(const std::vector<GLint>& v)
        {
            std::string s;
            for (const GLint x : v) s += std::format("{}{}", s.empty() ? "" : ",", x);
            return s;
        }

        void check_completeness(demo::Check& check)
        {
            // TEXTURE_SAMPLES and TEXTURE_FIXED_SAMPLE_LOCATIONS cannot be read with glGetTexLevelParameteriv through meta-gl
            // (TextureLevelParameter has no such enumerators). Section 9.4.2 makes both observable: attached images must agree.
            const int s = ms_fixed_.samples;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ms_fixed_.fbo);
            check.expect(demo::get_integer(mgl::GetParameter::Samples) == s && demo::get_integer(mgl::GetParameter::SampleBuffers) == 1 && s >= ms_fixed_.requested,
                         "a framebuffer with a multisample texture attached: SAMPLE_BUFFERS = 1 and SAMPLES = the texture's sample count (section 9.2.3.1), at least the request",
                         std::format("requested {}, SAMPLES {}", ms_fixed_.requested, s));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, strip_fbo_[0]);
            check.expect(demo::get_integer(mgl::GetParameter::Samples) == 0 && demo::get_integer(mgl::GetParameter::SampleBuffers) == 0,
                         "a framebuffer with a single-sample texture: SAMPLES = 0 and SAMPLE_BUFFERS = 0");

            // A multisample renderbuffer with the same request.
            mgl::RenderbufferId rb;
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, s, mgl::InternalFormat::Rgba8, kW, kH);
            GLint rb_samples = 0;
            mgl::glGetRenderbufferParameteriv(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferParameter::Samples, &rb_samples);

            const mgl::FramebufferAttachment c0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
            const mgl::FramebufferAttachment c1 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1);
            mgl::FramebufferId fbo;
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);

            auto attach_tex = [&](mgl::FramebufferAttachment a, const Ms& m) {
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, a, mgl::TextureTarget::Texture2DMultisample, m.tex, 0);
            };
            auto status = [] { return mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer); };
            auto expect_status = [&](mgl::FramebufferStatus want, std::string_view what, std::string_view detail) {
                const mgl::FramebufferStatus got = status();
                check.expect(got == want, what, std::format("{} - got {}, expected {}", detail, demo::framebuffer_status_name(got), demo::framebuffer_status_name(want)));
            };

            // (a) fixedsamplelocations = TRUE texture + renderbuffer: complete exactly when the two sample counts are equal.
            attach_tex(c0, ms_fixed_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, c1, mgl::RenderbufferTarget::Renderbuffer, rb);
            expect_status(rb_samples == s ? mgl::FramebufferStatus::Complete : mgl::FramebufferStatus::IncompleteMultisample,
                          "texture (fixedsamplelocations TRUE) + renderbuffer: FRAMEBUFFER_COMPLETE if RENDERBUFFER_SAMPLES equals TEXTURE_SAMPLES, else INCOMPLETE_MULTISAMPLE (section 9.4.2)",
                          std::format("TEXTURE_SAMPLES {}, RENDERBUFFER_SAMPLES {}", s, rb_samples));
            // (b) fixedsamplelocations = FALSE texture + renderbuffer: always INCOMPLETE_MULTISAMPLE ("must be TRUE for all attached textures").
            attach_tex(c0, ms_free_);
            expect_status(mgl::FramebufferStatus::IncompleteMultisample,
                          "texture (fixedsamplelocations FALSE) + renderbuffer: INCOMPLETE_MULTISAMPLE, whatever the sample counts",
                          std::format("texture samples {}, renderbuffer samples {}", ms_free_.samples, rb_samples));
            // (c) the same texture alone, and two textures that agree in samples and fixedsamplelocations.
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, c1, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            expect_status(mgl::FramebufferStatus::Complete, "a texture with fixedsamplelocations FALSE alone is complete", "");
            Ms other_free = make_ms(kRgba8, ms_free_.samples, false, 5, 7);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            attach_tex(c0, ms_free_);
            attach_tex(c1, other_free);
            expect_status(other_free.samples == ms_free_.samples ? mgl::FramebufferStatus::Complete : mgl::FramebufferStatus::IncompleteMultisample,
                          "two textures with fixedsamplelocations FALSE and different sizes: complete when their sample counts are equal (ES 3.0 and later have no size rule)",
                          std::format("samples {} and {}", ms_free_.samples, other_free.samples));
            // (d) two textures that differ in fixedsamplelocations.
            attach_tex(c0, ms_fixed_);
            Ms same_samples_free = make_ms(kRgba8, ms_fixed_.samples, false, kW, kH);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            attach_tex(c1, same_samples_free);
            expect_status(mgl::FramebufferStatus::IncompleteMultisample, "two multisample textures that differ in fixedsamplelocations: INCOMPLETE_MULTISAMPLE",
                          std::format("samples {} and {}", ms_fixed_.samples, same_samples_free.samples));
            // (e) a multisample texture together with a single-sample texture: TEXTURE_SAMPLES differs.
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, c1, mgl::TextureTarget::Texture2D, strip_tex_[0], 0);
            expect_status(mgl::FramebufferStatus::IncompleteMultisample, "a multisample texture + a single-sample texture: INCOMPLETE_MULTISAMPLE (TEXTURE_SAMPLES differs)", "");

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glDeleteRenderbuffers(1, &rb);
            destroy(other_free);
            destroy(same_samples_free);
            check.expect_no_gl_error("no error from the completeness fixtures");
        }

        void check_sample_positions(demo::Check& check)
        {
            auto positions = [&](const Ms& m) {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m.fbo);
                std::vector<std::array<float, 2>> out;
                for (int i = 0; i < m.samples; ++i)
                {
                    std::array<float, 2> v = {-1.0f, -1.0f};
                    mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(i), v.data());
                    out.push_back(v);
                }
                return out;
            };
            auto in_range = [](const std::vector<std::array<float, 2>>& p) {
                bool ok = !p.empty();
                for (const auto& v : p) ok = ok && v[0] >= 0.0f && v[0] <= 1.0f && v[1] >= 0.0f && v[1] <= 1.0f;
                return ok;
            };
            auto describe = [](const std::vector<std::array<float, 2>>& p) {
                std::string s;
                for (const auto& v : p) s += std::format("({:.4f},{:.4f}) ", v[0], v[1]);
                return s;
            };
            (void)demo::drain_gl_errors();
            const auto fixed = positions(ms_fixed_);
            const bool no_error_fixed = demo::drain_gl_errors() == mgl::ErrorCode::NoError;
            check.expect(no_error_fixed && in_range(fixed), "glGetMultisamplefv(SAMPLE_POSITION, i) for every sample of a fixedsamplelocations texture: x and y in [0, 1] (section 13.4)",
                         describe(fixed));
            const auto free_locations = positions(ms_free_);
            const bool no_error_free = demo::drain_gl_errors() == mgl::ErrorCode::NoError;
            check.expect(no_error_free && in_range(free_locations),
                         "glGetMultisamplefv for every sample of a texture without fixed locations: x and y in [0, 1] (the values may only describe some pixels)", describe(free_locations));

            // "If fixedsamplelocations is TRUE ... the sample locations will not depend on the internalformat or size of the image" (section 8.8).
            Ms other = make_ms(kRgba8I, ms_fixed_.samples, true, 7, 5);
            if (other.samples == ms_fixed_.samples)
                check.expect(positions(other) == fixed,
                             "two fixedsamplelocations textures with the same sample count but different size (7 x 5 vs 16 x 12) and format (RGBA8I vs RGBA8) have identical sample positions (section 8.8)",
                             describe(positions(other)));
            else
                std::cout << std::format("  note: RGBA8I with the request {} gave {} samples (RGBA8: {}), so the identical-positions check is skipped\n", ms_fixed_.samples, other.samples, ms_fixed_.samples);
            destroy(other);

            // INVALID_VALUE for an index that is not smaller than SAMPLES.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ms_fixed_.fbo);
            std::array<float, 2> v = {};
            expect_error(check, mgl::ErrorCode::InvalidValue, "glGetMultisamplefv with index == SAMPLES -> INVALID_VALUE (section 13.4)",
                         [&] { mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(ms_fixed_.samples), v.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glGetMultisamplefv with index 1000 -> INVALID_VALUE",
                         [&] { mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, 1000u, v.data()); });
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            const int default_samples = demo::get_integer(mgl::GetParameter::Samples);
            expect_error(check, mgl::ErrorCode::InvalidValue, std::format("glGetMultisamplefv with the default framebuffer bound (SAMPLES = {}) and index {} -> INVALID_VALUE", default_samples, default_samples),
                         [&] { mgl::glGetMultisamplefv(mgl::MultisampleParameter::SamplePosition, static_cast<GLuint>(default_samples), v.data()); });
        }

        /// Every sample of every texel, exactly, for every format and count.
        void check_every_sample(demo::Check& check)
        {
            // The textures of the window: filled by render(), read back now.
            expect_strips(check, "RGBA8 texture (fixedsamplelocations TRUE): every sample k holds pattern k at every texel (sampler2DMS, texelFetch)", ms_fixed_, iota_ids(ms_fixed_.samples));
            expect_strips(check, "RGBA8 texture (fixedsamplelocations FALSE): every sample k holds pattern k at every texel", ms_free_, iota_ids(ms_free_.samples));
            expect_strips(check, "RGBA8UI texture: every sample holds its pattern as unsigned integers (usampler2DMS, uvec4)", ms_uint_, iota_ids(ms_uint_.samples));
            expect_strips(check, "RGBA8I texture: every sample holds its pattern as signed integers (isampler2DMS, ivec4)", ms_int_, iota_ids(ms_int_.samples));

            // Every reported sample count of RGBA8, written and read in full.
            for (const GLint count : counts_[kRgba8])
            {
                if (count > 32 || count > 32 * mask_words_) continue;  // the strips target and the mask words limit the demo to 32 samples
                Ms m = make_ms(kRgba8, count, true, kW, kH);
                fill_per_sample(m);
                expect_strips(check, std::format("RGBA8, request {} samples ({} allocated): every sample holds its own pattern", count, m.samples), m, iota_ids(m.samples));
                destroy(m);
            }

            // Draw without a sample mask: every sample receives the color (full coverage by one primitive).
            fill_all(ms_fixed_, kIdB);
            expect_strips(check, "SAMPLE_MASK disabled: a draw that covers the texture writes the same pattern into every sample", ms_fixed_, std::vector<int>(static_cast<std::size_t>(ms_fixed_.samples), kIdB));
            fill_per_sample(ms_fixed_);  // restore what the window shows
        }

        void check_sample_mask(demo::Check& check)
        {
            // Initial state, captured in init() before any glSampleMaski call.
            bool all_ones = !initial_mask_.empty();
            for (const GLint v : initial_mask_) all_ones = all_ones && v == -1;  // 32 one bits read back as the signed integer -1
            check.expect(!initial_enabled_ && all_ones, "initially SAMPLE_MASK is disabled and every SAMPLE_MASK_VALUE word is all ones (glIsEnabled, glGetIntegeri_v)",
                         std::format("enabled {}, {} words, first {:#x}", initial_enabled_, initial_mask_.size(), static_cast<unsigned>(initial_mask_.empty() ? 0 : initial_mask_[0])));

            // glSampleMaski is readable with glGetIntegeri_v.
            set_mask(0xA5A5A5A5u);
            GLint v = 0;
            mgl::glGetIntegeri_v(mgl::GetParameter::SampleMaskValue, 0, &v);
            check.expect_eq(static_cast<unsigned>(v), 0xA5A5A5A5u, "glSampleMaski(0, 0xA5A5A5A5) is read back by glGetIntegeri_v(SAMPLE_MASK_VALUE, 0)");
            mgl::glEnable(mgl::Capability::SampleMask);
            check.expect(mgl::glIsEnabled(mgl::Capability::SampleMask), "glEnable(SAMPLE_MASK) is visible in glIsEnabled");
            mgl::glDisable(mgl::Capability::SampleMask);
            check.expect(!mgl::glIsEnabled(mgl::Capability::SampleMask), "glDisable(SAMPLE_MASK) is visible in glIsEnabled");
            restore_mask();

            const Ms& m = ms_fixed_;
            const std::size_t n = static_cast<std::size_t>(m.samples);
            const std::uint32_t valid = m.samples >= 32 ? 0xFFFFFFFFu : (1u << m.samples) - 1u;

            // Sample j holds pattern B if the mask lets it through, pattern A (the clear pattern) otherwise.
            auto run = [&](std::uint32_t mask, bool enabled, std::string_view what) {
                fill_all(m, kIdA);
                if (enabled) mgl::glEnable(mgl::Capability::SampleMask);
                set_mask(mask);
                draw_pattern(m, kIdB);
                mgl::glDisable(mgl::Capability::SampleMask);
                restore_mask();
                std::vector<int> ids(n);
                for (std::size_t j = 0; j < n; ++j) ids[j] = !enabled || ((mask >> j) & 1u) ? kIdB : kIdA;
                expect_strips(check, what, m, ids);
            };
            run(0u, true, "SAMPLE_MASK enabled, mask 0: no sample is written (every sample keeps the clear pattern)");
            run(0xFFFFFFFFu, true, "SAMPLE_MASK enabled, mask all ones: every sample is written");
            run(0u, false, "SAMPLE_MASK disabled with a stored mask of 0: the mask is ignored and every sample is written");
            if (valid != 0xFFFFFFFFu)
                run(~valid, true, "SAMPLE_MASK enabled with only bits >= SAMPLES set: no sample of the texture is written");
            run(0x55555555u, true, "SAMPLE_MASK enabled, mask 0x55555555: exactly the even samples are written");
            run(0xAAAAAAAAu, true, "SAMPLE_MASK enabled, mask 0xAAAAAAAA: exactly the odd samples are written");
            run(valid & ~(1u << (m.samples - 1)), true, "SAMPLE_MASK enabled, every sample but the last: the last keeps the clear pattern");
            run(valid & ~1u, true, "SAMPLE_MASK enabled, every sample but sample 0: sample 0 keeps the clear pattern");

            // A single-sample framebuffer ignores the mask (section 13.8.3: "If the value of SAMPLE_BUFFERS is not one, this step is skipped").
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, strip_fbo_[0]);
            mgl::glViewport(0, 0, kW, kH);
            clear_sentinel(Kind::Float);
            mgl::glEnable(mgl::Capability::SampleMask);
            set_mask(0u);
            const Program& p = program(Role::Fill, Kind::Float);
            mgl::glUseProgram(p.id);
            mgl::glUniform1i(p.sample, kIdB);
            mgl::glUniform1f(p.z, 0.0f);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glDisable(mgl::Capability::SampleMask);
            restore_mask();
            const Strips single = read_strips(Kind::Float, kW, kH);
            const StripDiff d = compare_strips(single, expected_strips(kW, kH, {kIdB}), kW);
            check.expect(d.bad == 0, "SAMPLE_MASK enabled with mask 0 does not affect a single-sample framebuffer (SAMPLE_BUFFERS = 0): the whole texture is written",
                         d.bad == 0 ? std::format("{} texels written", kW * kH) : std::format("{} values differ, {}", d.bad, d.first));
        }

        /// A triangle, whose partial coverage of pixels the sample mask restricts sample by sample.
        void check_triangle_coverage(demo::Check& check)
        {
            const Ms& m = ms_tri_;
            const int n = m.samples;
            const std::size_t nn = static_cast<std::size_t>(n);
            fill_all(m, kIdA);
            mgl::glDisable(mgl::Capability::SampleMask);
            draw_triangle(m, kIdB);
            const Strips all = fetch(m);

            // covered[j][y][x]: sample j of pixel (x, y) shows pattern B. Every sample must show A or B: nothing else was drawn.
            int neither = 0;
            std::vector<std::vector<char>> covered(nn, std::vector<char>(static_cast<std::size_t>(kTri * kTri), 0));
            for (int j = 0; j < n; ++j)
                for (int y = 0; y < kTri; ++y)
                    for (int x = 0; x < kTri; ++x)
                    {
                        const Pattern got = all.at(j * kTri + x, y);
                        const bool is_b = got == pattern(kIdB, x, y);
                        const bool is_a = got == pattern(kIdA, x, y);
                        if (!is_a && !is_b) ++neither;
                        covered[static_cast<std::size_t>(j)][static_cast<std::size_t>(y * kTri + x)] = is_b ? 1 : 0;
                    }
            check.expect(neither == 0, "after one triangle every sample holds either the clear pattern or the triangle's pattern, never anything else",
                         std::format("{} samples x {} pixels examined, {} foreign values", n, kTri * kTri, neither));

            // Pixels well inside / outside the triangle: all samples B / all samples A. The margin of 2 pixels is a sanity margin: the sample
            // points of a pixel are meant to lie in or near it (section 13.4 allows them anywhere, so no tighter rule is asserted).
            int interior = 0, interior_bad = 0, exterior = 0, exterior_bad = 0, partial = 0;
            long covered_samples = 0;
            for (int y = 0; y < kTri; ++y)
                for (int x = 0; x < kTri; ++x)
                {
                    int count = 0;
                    for (std::size_t j = 0; j < nn; ++j) count += covered[j][static_cast<std::size_t>(y * kTri + x)];
                    covered_samples += count;
                    partial += count > 0 && count < n ? 1 : 0;
                    const double cx = x + 0.5, cy = y + 0.5;
                    double min_distance = 1e9, max_outside = -1e9;
                    for (std::size_t e = 0; e < 3; ++e)
                    {
                        const double d = kTriangle.distance(e, cx, cy);
                        min_distance = std::min(min_distance, d);
                        max_outside = std::max(max_outside, -d);
                    }
                    if (min_distance >= 2.0)
                    {
                        ++interior;
                        interior_bad += count == n ? 0 : 1;
                    }
                    else if (max_outside >= 2.0)
                    {
                        ++exterior;
                        exterior_bad += count == 0 ? 0 : 1;
                    }
                }
            check.expect(interior > 20 && interior_bad == 0 && exterior > 20 && exterior_bad == 0,
                         "pixels 2 or more pixels inside the triangle have all samples covered, pixels 2 or more outside have none",
                         std::format("{} interior pixels ({} wrong), {} exterior pixels ({} wrong)", interior, interior_bad, exterior, exterior_bad));
            const double fraction = static_cast<double>(covered_samples) / (static_cast<double>(n) * kTri * kTri);
            std::cout << std::format("  triangle: area {:.1f} px = {:.4f} of the texture, covered samples {:.4f}, {} pixels with partial coverage\n", kTriangle.area(),
                                     kTriangle.area() / (kTri * kTri), fraction, partial);
            if (n >= 2)
                check.expect(partial > 0, "a triangle edge covers some but not all samples of at least one pixel (the texture really is multisampled)", std::format("{} pixels with partial coverage", partial));

            // One draw per sample, with only that sample in the mask: it writes exactly the samples the unmasked draw wrote, restricted to it.
            int mask_bad = 0;
            std::string first;
            for (int k = 0; k < n && k < 32 * mask_words_; ++k)
            {
                fill_all(m, kIdA);
                mgl::glEnable(mgl::Capability::SampleMask);
                set_mask_only(k);
                draw_triangle(m, kIdB);
                mgl::glDisable(mgl::Capability::SampleMask);
                restore_mask();
                const Strips got = fetch(m);
                for (int j = 0; j < n; ++j)
                    for (int y = 0; y < kTri; ++y)
                        for (int x = 0; x < kTri; ++x)
                        {
                            const bool want_b = j == k && covered[static_cast<std::size_t>(j)][static_cast<std::size_t>(y * kTri + x)] != 0;
                            const Pattern want = pattern(want_b ? kIdB : kIdA, x, y);
                            if (got.at(j * kTri + x, y) != want && mask_bad++ == 0)
                                first = std::format("mask sample {}: sample {} pixel ({},{}) differs", k, j, x, y);
                        }
            }
            check.expect(mask_bad == 0, "triangle drawn with only sample k in the mask (every k): sample k gets exactly its unmasked coverage and every other sample keeps the clear pattern",
                         mask_bad == 0 ? std::format("{} draws, {} samples each", n, n) : std::format("{} samples differ, {}", mask_bad, first));
        }

        void check_texture_size(demo::Check& check)
        {
            // textureSize(gsampler2DMS) returns (width, height) with no level argument; three samplers, three sizes.
            Ms f = make_ms(kRgba8, ms_fixed_.requested, true, 5, 3);
            Ms i = make_ms(kRgba8I, ms_int_.requested, true, 37, 1);
            const std::array<const Ms*, 3> textures = {&f, &i, &ms_uint_};
            for (const Ms* m : textures)
            {
                const Program& p = program(Role::Size, m->kind);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, strip_fbo_[index_of(Kind::Int)]);
                mgl::glViewport(0, 0, 1, 1);
                clear_sentinel(Kind::Int);
                mgl::glUseProgram(p.id);
                mgl::glActiveTexture(mgl::TextureUnit::Texture0);
                mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, m->tex);
                mgl::glUniform1i(p.ms, 0);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                std::array<GLint, 4> size = {};
                mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int, size.data());
                check.expect(size[0] == m->w && size[1] == m->h,
                             std::format("textureSize({}) of a {} x {} multisample texture returns ({}, {})",
                                         m->kind == Kind::Float ? "sampler2DMS" : m->kind == Kind::Int ? "isampler2DMS" : "usampler2DMS", m->w, m->h, m->w, m->h),
                             std::format("got ({}, {})", size[0], size[1]));
            }
            // glGetTexLevelParameteriv knows the size and the format of level 0 (section 8.8: they are set by the allocation).
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, i.tex);
            GLint w = 0, h = 0, format = 0;
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2DMultisample, 0, mgl::TextureLevelParameter::Width, &w);
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2DMultisample, 0, mgl::TextureLevelParameter::Height, &h);
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2DMultisample, 0, mgl::TextureLevelParameter::InternalFormat, &format);
            check.expect(w == 37 && h == 1 && format == static_cast<GLint>(GL_RGBA8I),
                         "glGetTexLevelParameteriv(TEXTURE_2D_MULTISAMPLE, 0): TEXTURE_WIDTH, TEXTURE_HEIGHT and TEXTURE_INTERNAL_FORMAT are those of the allocation",
                         std::format("{} x {}, format {:#x}", w, h, static_cast<unsigned>(format)));
            GLint immutable = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2DMultisample, mgl::TextureParameterQuery::ImmutableFormat, &immutable);
            check.expect(immutable == GL_TRUE, "a multisample texture is immutable: TEXTURE_IMMUTABLE_FORMAT is TRUE after glTexStorage2DMultisample (section 8.8)");
            destroy(f);
            destroy(i);
        }

        void check_depth_samples(demo::Check& check)
        {
            // Depth per sample: with the sample mask letting sample k through, a full-texture draw at depth z_k = (k + 1) / (N + 1) and
            // the depth test ALWAYS writes z_k to sample k. A sampler2DMS reads a depth texture as (depth, 0, 0, 1).
            Ms d = make_ms(kDepth24, pick(kDepth24), true, kW, kH);
            check.expect(d.status == mgl::FramebufferStatus::Complete && d.samples >= d.requested, "a DEPTH_COMPONENT24 multisample texture as the only attachment: framebuffer complete",
                         std::format("requested {}, SAMPLES {}", d.requested, d.samples));
            if (d.status != mgl::FramebufferStatus::Complete) return;
            const int n = d.samples;

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, d.fbo);
            mgl::glViewport(0, 0, d.w, d.h);
            mgl::glDisable(mgl::Capability::SampleMask);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            mgl::glEnable(mgl::Capability::SampleMask);
            for (int k = 0; k < n && k < 32 * mask_words_; ++k)
            {
                set_mask_only(k);
                const double z = static_cast<double>(k + 1) / (n + 1);
                draw_pattern(d, 0, static_cast<float>(2.0 * z - 1.0));  // window depth = (ndc + 1) / 2 with the default depth range [0, 1]
            }
            mgl::glDisable(mgl::Capability::SampleMask);
            restore_mask();
            mgl::glDisable(mgl::Capability::DepthTest);

            const Strips s = fetch(d);
            int bad = 0;
            int max_error = 0;
            std::string first;
            for (int k = 0; k < n; ++k)
            {
                const int want = static_cast<int>(std::lround(255.0 * static_cast<double>(k + 1) / (n + 1)));
                for (int y = 0; y < d.h; ++y)
                    for (int x = 0; x < d.w; ++x)
                    {
                        const Pattern& px = s.at(k * d.w + x, y);
                        const int error = std::abs(px[0] - want);
                        max_error = std::max(max_error, error);
                        // The 8-bit result of a 24-bit depth: +-1. The other components of a depth texture are (0, 0, 1).
                        if ((error > 1 || px[1] != 0 || px[2] != 0 || px[3] != 255) && bad++ == 0)
                            first = std::format("sample {} texel ({},{}): got ({}, {}, {}, {}), expected ({}+-1, 0, 0, 255)", k, x, y, px[0], px[1], px[2], px[3], want);
                    }
            }
            check.expect(bad == 0, "depth samples: sample k of a DEPTH_COMPONENT24 texture reads back as depth (k+1)/(N+1) through sampler2DMS, in (d, 0, 0, 1) form",
                         bad == 0 ? std::format("{} samples x {} texels, max error {} of 255 (tolerance 1: 8-bit output)", n, d.w * d.h, max_error) : std::format("{} values differ, {}", bad, first));
            destroy(d);
        }

        void check_errors(demo::Check& check)
        {
            const int max_size = demo::get_integer(mgl::GetParameter::MaxTextureSize);
            const std::vector<GLint>& counts = counts_[kRgba8];
            const mgl::TextureTarget ms = mgl::TextureTarget::Texture2DMultisample;
            const int samples = ms_fixed_.requested;

            mgl::TextureId t;
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(ms, t);
            const mgl::SizedInternalFormat rgba8 = mgl::SizedInternalFormat::Rgba8;
            using Err = mgl::ErrorCode;
            expect_error(check, Err::InvalidEnum, "glTexStorage2DMultisample with target TEXTURE_2D -> INVALID_ENUM",
                         [&] { mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2D, samples, rgba8, 8, 8, GL_TRUE); });
            expect_error(check, Err::InvalidValue, "glTexStorage2DMultisample with samples 0 -> INVALID_VALUE", [&] { mgl::glTexStorage2DMultisample(ms, 0, rgba8, 8, 8, GL_TRUE); });
            expect_error(check, Err::InvalidValue, "glTexStorage2DMultisample with width 0 -> INVALID_VALUE", [&] { mgl::glTexStorage2DMultisample(ms, samples, rgba8, 0, 8, GL_TRUE); });
            expect_error(check, Err::InvalidValue, "glTexStorage2DMultisample with height 0 -> INVALID_VALUE", [&] { mgl::glTexStorage2DMultisample(ms, samples, rgba8, 8, 0, GL_TRUE); });
            expect_error(check, Err::InvalidValue, "glTexStorage2DMultisample with width > MAX_TEXTURE_SIZE -> INVALID_VALUE",
                         [&] { mgl::glTexStorage2DMultisample(ms, samples, rgba8, max_size + 1, 8, GL_TRUE); });
            expect_error(check, Err::InvalidOperation, "glTexStorage2DMultisample with samples greater than the maximum for the format -> INVALID_OPERATION",
                         [&] { mgl::glTexStorage2DMultisample(ms, counts.front() + 1, rgba8, 8, 8, GL_TRUE); });
            expect_error(check, Err::InvalidEnum, "glTexStorage2DMultisample with RGB9_E5, which is not renderable -> INVALID_ENUM",
                         [&] { mgl::glTexStorage2DMultisample(ms, samples, mgl::SizedInternalFormat::Rgb9E5, 8, 8, GL_TRUE); });
            // None of the failed calls allocated anything: the texture can still be given storage once, and then it is immutable.
            expect_error(check, Err::NoError, "glTexStorage2DMultisample of the same texture after all those failures succeeds",
                         [&] { mgl::glTexStorage2DMultisample(ms, samples, rgba8, 8, 8, GL_FALSE); });
            expect_error(check, Err::InvalidOperation, "glTexStorage2DMultisample of a texture that already has storage (TEXTURE_IMMUTABLE_FORMAT) -> INVALID_OPERATION",
                         [&] { mgl::glTexStorage2DMultisample(ms, samples, rgba8, 8, 8, GL_FALSE); });
            expect_error(check, Err::InvalidOperation, "glBindTexture(TEXTURE_2D, a multisample texture) -> INVALID_OPERATION (different dimensionality)",
                         [&] { mgl::glBindTexture(mgl::TextureTarget::Texture2D, t); });

            mgl::glBindTexture(ms, mgl::TextureId{0});
            expect_error(check, Err::InvalidOperation, "glTexStorage2DMultisample with texture 0 bound to TEXTURE_2D_MULTISAMPLE -> INVALID_OPERATION",
                         [&] { mgl::glTexStorage2DMultisample(ms, samples, rgba8, 8, 8, GL_TRUE); });

            // Commands that do not accept the multisample target.
            mgl::glBindTexture(ms, t);
            expect_error(check, Err::InvalidEnum, "glTexImage2D with target TEXTURE_2D_MULTISAMPLE -> INVALID_ENUM (section 8.5)",
                         [&] { mgl::glTexImage2D(ms, 0, mgl::InternalFormat::Rgba8, 8, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, static_cast<const void*>(nullptr)); });
            expect_error(check, Err::InvalidEnum, "glTexStorage2D with target TEXTURE_2D_MULTISAMPLE -> INVALID_ENUM (section 8.18)",
                         [&] { mgl::glTexStorage2D(ms, 1, rgba8, 8, 8); });
            // ES 3.2 8.14.4: "An INVALID_ENUM error is generated if target is not TEXTURE_2D, TEXTURE_3D, TEXTURE_2D_ARRAY,
            // TEXTURE_CUBE_MAP or TEXTURE_CUBE_MAP_ARRAY."
            expect_error_or_known_deviation(check, Err::InvalidEnum, "glGenerateMipmap(TEXTURE_2D_MULTISAMPLE) -> INVALID_ENUM (section 8.14.4)",
                                            [&] { mgl::glGenerateMipmap(ms); }, "ANGLE",
                                            "ANGLE accepts glGenerateMipmap(TEXTURE_2D_MULTISAMPLE) without an error; ES 3.1/3.2 8.14.4 list the "
                                            "valid targets and require INVALID_ENUM for any other");
            expect_error(check, Err::InvalidEnum, "glTexParameteri(TEXTURE_2D_MULTISAMPLE, TEXTURE_MIN_FILTER, NEAREST): sampler state -> INVALID_ENUM (section 8.10)",
                         [&] { mgl::glTexParameteri(ms, mgl::TextureParameterSetter::MinFilter, GL_NEAREST); });
            expect_error(check, Err::InvalidEnum, "glTexParameteri(TEXTURE_2D_MULTISAMPLE, TEXTURE_WRAP_S, REPEAT): sampler state -> INVALID_ENUM",
                         [&] { mgl::glTexParameteri(ms, mgl::TextureParameterSetter::WrapS, GL_REPEAT); });
            expect_error(check, Err::InvalidOperation, "glTexParameteri(TEXTURE_2D_MULTISAMPLE, TEXTURE_BASE_LEVEL, 1) -> INVALID_OPERATION (section 8.10)",
                         [&] { mgl::glTexParameteri(ms, mgl::TextureParameterSetter::BaseLevel, 1); });
            expect_error(check, Err::NoError, "glTexParameteri(TEXTURE_2D_MULTISAMPLE, TEXTURE_BASE_LEVEL, 0) is accepted",
                         [&] { mgl::glTexParameteri(ms, mgl::TextureParameterSetter::BaseLevel, 0); });

            // Level queries: the only level is 0 (section 8.11).
            GLint v = 0;
            // ES 3.2 8.11.3: "For targets TEXTURE_BUFFER, TEXTURE_2D_MULTISAMPLE, and TEXTURE_2D_MULTISAMPLE_ARRAY, which do
            // not support mipmaps, the maximum value is zero"; ES 3.1 8.8: "Multisample textures do not have multiple image levels".
            expect_error_or_known_deviation(check, Err::InvalidValue,
                                            "glGetTexLevelParameteriv(TEXTURE_2D_MULTISAMPLE, level 1) -> INVALID_VALUE (a multisample texture has no mipmaps)",
                                            [&] { mgl::glGetTexLevelParameteriv(ms, 1, mgl::TextureLevelParameter::Width, &v); }, "ANGLE",
                                            "ANGLE accepts level 1 of a TEXTURE_2D_MULTISAMPLE texture without an error; ES 3.2 8.11.3 makes the "
                                            "maximum level of that target zero");
            expect_error(check, Err::InvalidValue, "glGetTexLevelParameteriv(TEXTURE_2D_MULTISAMPLE, level -1) -> INVALID_VALUE",
                         [&] { mgl::glGetTexLevelParameteriv(ms, -1, mgl::TextureLevelParameter::Width, &v); });

            // Framebuffer attachment rules (section 9.2).
            mgl::FramebufferId fbo;
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            const mgl::FramebufferAttachment c0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);
            expect_error(check, Err::InvalidValue, "glFramebufferTexture2D(TEXTURE_2D_MULTISAMPLE, level 1) -> INVALID_VALUE (the level must be zero)",
                         [&] { mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, c0, ms, ms_fixed_.tex, 1); });
            expect_error(check, Err::InvalidOperation, "glFramebufferTexture2D(TEXTURE_2D, a multisample texture) -> INVALID_OPERATION (the texture's type does not match)",
                         [&] { mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, c0, mgl::TextureTarget::Texture2D, ms_fixed_.tex, 0); });
            expect_error(check, Err::InvalidOperation, "glFramebufferTexture2D(TEXTURE_2D_MULTISAMPLE, a 2D texture) -> INVALID_OPERATION",
                         [&] { mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, c0, ms, strip_tex_[0], 0); });
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);

            // Sample mask and internal format queries.
            expect_error(check, Err::InvalidValue, "glSampleMaski(MAX_SAMPLE_MASK_WORDS, mask) -> INVALID_VALUE (section 13.8.3)",
                         [&] { mgl::glSampleMaski(static_cast<GLuint>(mask_words_), mgl::SampleMaskValue::All); });
            expect_error(check, Err::NoError, "glSampleMaski(MAX_SAMPLE_MASK_WORDS - 1, mask) is accepted",
                         [&] { mgl::glSampleMaski(static_cast<GLuint>(mask_words_ - 1), mgl::SampleMaskValue::All); });
            GLint q = 0;
            expect_error(check, Err::InvalidValue, "glGetInternalformativ(TEXTURE_2D_MULTISAMPLE, RGBA8, SAMPLES, count -1) -> INVALID_VALUE (section 20.3)",
                         [&] { mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgba8, mgl::InternalFormatParameter::Samples, -1, &q); });
            expect_error(check, Err::InvalidEnum, "glGetInternalformativ(TEXTURE_2D_MULTISAMPLE, RGB9_E5 (not renderable), SAMPLES) -> INVALID_ENUM",
                         [&] { mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, mgl::InternalFormat::Rgb9E5, mgl::InternalFormatParameter::Samples, 1, &q); });

            mgl::glBindTexture(ms, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &t);
        }

        void check_negative_controls(demo::Check& check)
        {
            // Mutation 1: a model that ignores the sample mask (every sample = pattern B) must disagree with what the hardware produced for
            // "SAMPLE_MASK enabled, mask 0x55555555".
            const Ms& m = ms_fixed_;
            fill_all(m, kIdA);
            mgl::glEnable(mgl::Capability::SampleMask);
            set_mask(0x55555555u);
            draw_pattern(m, kIdB);
            mgl::glDisable(mgl::Capability::SampleMask);
            restore_mask();
            const Strips got = fetch(m);
            const StripDiff ignoring = compare_strips(got, expected_strips(m.w, m.h, std::vector<int>(static_cast<std::size_t>(m.samples), kIdB)), m.w);
            check.expect(m.samples < 2 || ignoring.bad > 0, "negative control: a model that ignores the sample mask disagrees with the observed samples",
                         std::format("{} of {} values differ", ignoring.bad, got.w * got.h));
            // Mutation 2: a model with the sample numbers reversed (sample j <-> N-1-j) must disagree with the per-sample patterns.
            fill_per_sample(m);
            const Strips per_sample = fetch(m);
            std::vector<int> reversed = iota_ids(m.samples);
            std::reverse(reversed.begin(), reversed.end());
            const StripDiff swapped = compare_strips(per_sample, expected_strips(m.w, m.h, reversed), m.w);
            check.expect(m.samples < 2 || swapped.bad > 0, "negative control: a model with the sample numbers reversed disagrees with the observed samples",
                         std::format("{} of {} values differ", swapped.bad, per_sample.w * per_sample.h));
        }

        // ==========================================================================================
        // State
        // ==========================================================================================
        mgl::VertexArrayId vao_{};
        std::array<mgl::TextureId, 3> strip_tex_{};
        std::array<mgl::FramebufferId, 3> strip_fbo_{};
        Ms ms_fixed_, ms_free_, ms_uint_, ms_int_, ms_tri_;
        std::vector<std::vector<GLint>> counts_;
        std::map<std::string, Program> programs_;
        std::string vertex_fullscreen_, vertex_triangle_;
        std::array<std::string, 4> fragment_;
        int mask_words_ = 1;
        bool initial_enabled_ = false;
        std::vector<GLint> initial_mask_;
        std::array<int, 4> shown_cols_{};
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es31-multisample-textures",
        .title = "Multisample textures and the sample mask",
        .min_es = {3, 1},
        .summary = "glTexStorage2DMultisample, rendering into a TEXTURE_2D_MULTISAMPLE attachment, texelFetch of every sample through sampler2DMS / isampler2DMS / "
                   "usampler2DMS, glSampleMaski + SAMPLE_MASK, glGetMultisamplefv, textureSize and the multisample limits, with every sample read back exactly.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<MultisampleTextures>(argc, argv, info);
}
