// es31-stencil-texturing-and-level-queries
//
// Two ES 3.1 texture features:
//   * DEPTH_STENCIL_TEXTURE_MODE: a DEPTH24_STENCIL8 / DEPTH32F_STENCIL8 texture is sampled either as depth
//     (mode DEPTH_COMPONENT, a sampler2D returns the depth) or as stencil (mode STENCIL_INDEX, a usampler2D
//     returns the unsigned stencil value; only NEAREST filtering is supported). ES 3.2 section 8.16.
//   * glGetTexLevelParameteriv / glGetTexLevelParameterfv: the size, format, component resolutions and types
//     of one image of a texture. ES 3.2 section 8.11.3.
//
// Part 1 renders a 32 x 32 depth/stencil scene into depth/stencil textures through a framebuffer object with
// no color attachment (three passes of rectangles with different depth and stencil state, so that the stored
// stencil values are the product of REPLACE, INCR, ZERO, INVERT under a write mask), then reads every texel back
// IN THE SHADER (a compute shader dumps texelFetch and textureLod results into a shader storage buffer):
// depth through sampler2D, stencil through usampler2D, for both formats. A CPU model of the three passes
// predicts every texel. The contact sheet shows the same textures through fragment shaders.
//
// Part 2 allocates textures of every kind the query accepts (2D levels, 3D, 2D array, the six cube faces,
// multisample, compressed, mutable and immutable, 25 internal formats) and compares what the driver reports
// with what was allocated.

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
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int gl(GLenum e) { return static_cast<int>(e); }

    constexpr int kTex = 32;       // edge of the depth/stencil textures
    constexpr int kSmall = 4;      // edge of the small fixture textures
    constexpr int kAny = -1;       // "not asserted"

    // ---------------------------------------------------------------------------------------------
    // The scene: rectangles in texel units, drawn with the depth and stencil state given here. Depths are
    // multiples of 1/64 (exact in float) and no depth comparison is a tie, so GPU and model agree on every
    // test whatever the depth buffer's precision.
    // ---------------------------------------------------------------------------------------------
    struct Rect
    {
        int x0, y0, x1, y1;  // texels [x0, x1) x [y0, y1)
    };

    struct Draw
    {
        Rect rect;
        double depth;                  // window-space depth of the rectangle
        mgl::CompareFunc depth_func;
        mgl::CompareFunc stencil_func;
        int ref;
        unsigned func_mask;
        mgl::StencilOp sfail, dpfail, dppass;
        unsigned write_mask;
    };

    // Pass 1: fifteen 8 x 8 blocks of a 4 x 4 grid (block 0, the lower left one, is not drawn: it keeps the
    // cleared depth 1 and stencil 0) REPLACE their stencil value with a different reference each.
    constexpr std::array<int, 16> kBlockRef = {0, 255, 1, 2, 128, 127, 85, 170, 3, 5, 8, 13, 21, 34, 55, 89};

    std::vector<Draw> make_draws()
    {
        std::vector<Draw> draws;
        for (int b = 1; b < 16; ++b)
        {
            const int x = (b % 4) * 8, y = (b / 4) * 8;
            draws.push_back({{x, y, x + 8, y + 8}, (2.0 * b + 3.0) / 64.0, mgl::CompareFunc::Less, mgl::CompareFunc::Always,
                             kBlockRef[static_cast<std::size_t>(b)], 0xFFu, mgl::StencilOp::Replace, mgl::StencilOp::Replace,
                             mgl::StencilOp::Replace, 0xFFu});
        }
        // Pass 2a: a large rectangle at depth 0.25. Where the stored depth is not greater (blocks 1..6) the depth test
        // fails and INCR (saturating: 255 stays 255) runs; elsewhere the depth test passes, KEEP, and the depth becomes 0.25.
        draws.push_back({{4, 4, 28, 28}, 16.0 / 64.0, mgl::CompareFunc::Less, mgl::CompareFunc::Always, 0, 0xFFu,
                         mgl::StencilOp::Keep, mgl::StencilOp::Incr, mgl::StencilOp::Keep, 0xFFu});
        // Pass 2b: the stencil test LESS 100 (passes where the stored value is above 100). With a write mask of 0xF0 only the
        // upper nibble changes: INVERT where the test passes (the depth test is ALWAYS, the depth becomes 0.125), ZERO where it fails.
        draws.push_back({{12, 12, 20, 24}, 8.0 / 64.0, mgl::CompareFunc::Always, mgl::CompareFunc::Less, 100, 0xFFu,
                         mgl::StencilOp::Zero, mgl::StencilOp::Keep, mgl::StencilOp::Invert, 0xF0u});
        return draws;
    }

    constexpr std::size_t kPassOneDraws = 15;

    struct Pixel
    {
        double depth = 1.0;
        int stencil = 0;
    };

    bool compare(mgl::CompareFunc f, double incoming, double stored)
    {
        switch (f)
        {
        case mgl::CompareFunc::Never: return false;
        case mgl::CompareFunc::Less: return incoming < stored;
        case mgl::CompareFunc::Equal: return incoming == stored;
        case mgl::CompareFunc::Lequal: return incoming <= stored;
        case mgl::CompareFunc::Greater: return incoming > stored;
        case mgl::CompareFunc::Notequal: return incoming != stored;
        case mgl::CompareFunc::Gequal: return incoming >= stored;
        case mgl::CompareFunc::Always: return true;
        }
        return false;
    }

    int apply_op(mgl::StencilOp op, int stored, int ref)
    {
        switch (op)
        {
        case mgl::StencilOp::Keep: return stored;
        case mgl::StencilOp::Zero: return 0;
        case mgl::StencilOp::Replace: return ref;
        case mgl::StencilOp::Incr: return std::min(stored + 1, 255);
        case mgl::StencilOp::Decr: return std::max(stored - 1, 0);
        case mgl::StencilOp::Invert: return (~stored) & 255;
        case mgl::StencilOp::IncrWrap: return (stored + 1) & 255;
        case mgl::StencilOp::DecrWrap: return (stored - 1) & 255;
        }
        return stored;
    }

    /// What the first `count` draws leave in a depth 1 / stencil 0 buffer (ES 3.2 sections 15.1.2 and 15.1.3).
    std::vector<Pixel> model_scene(const std::vector<Draw>& draws, std::size_t count)
    {
        std::vector<Pixel> px(static_cast<std::size_t>(kTex * kTex));
        for (std::size_t i = 0; i < count; ++i)
        {
            const Draw& d = draws[i];
            for (int y = d.rect.y0; y < d.rect.y1; ++y)
                for (int x = d.rect.x0; x < d.rect.x1; ++x)
                {
                    Pixel& p = px[static_cast<std::size_t>(y * kTex + x)];
                    const unsigned ref_masked = static_cast<unsigned>(d.ref) & d.func_mask;
                    const unsigned stored_masked = static_cast<unsigned>(p.stencil) & d.func_mask;
                    const bool stencil_ok = compare(d.stencil_func, static_cast<double>(ref_masked), static_cast<double>(stored_masked));
                    const bool depth_ok = stencil_ok && compare(d.depth_func, d.depth, p.depth);
                    const mgl::StencilOp op = !stencil_ok ? d.sfail : (depth_ok ? d.dppass : d.dpfail);
                    const unsigned updated = static_cast<unsigned>(apply_op(op, p.stencil, d.ref));
                    p.stencil = static_cast<int>((static_cast<unsigned>(p.stencil) & ~d.write_mask) | (updated & d.write_mask));
                    if (depth_ok) p.depth = d.depth;
                }
        }
        return px;
    }

    // ---------------------------------------------------------------------------------------------
    // The picture of a stencil value in the contact sheet (show_stencil.frag).
    // ---------------------------------------------------------------------------------------------
    demo::Rgba8 stencil_color(int s)
    {
        return {static_cast<std::uint8_t>(s), static_cast<std::uint8_t>((s * 37) & 255), static_cast<std::uint8_t>((s * 91) & 255), 255};
    }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    float float_of(std::uint32_t bits)
    {
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }

    // ---------------------------------------------------------------------------------------------
    // Part 2: what the level query must report for each sized internal format.
    // minimum component resolutions come from ES 3.2 tables 8.10 and 8.11 ("Bits/component": the actual size may be
    // larger, section 8.11.3); the type of each color component and of the depth component is one of the tokens of
    // section 8.11.3 (NONE = missing). kAny = not asserted.
    // ---------------------------------------------------------------------------------------------
    struct FormatCase
    {
        std::string_view name;
        mgl::SizedInternalFormat sized;
        GLenum token;
        std::array<int, 6> min_bits;  // R G B A D S
        std::array<int, 5> types;     // R G B A D
    };

    constexpr int kNone = gl(GL_NONE);
    constexpr int kUnorm = gl(GL_UNSIGNED_NORMALIZED);
    constexpr int kSnorm = gl(GL_SIGNED_NORMALIZED);
    constexpr int kFloat = gl(GL_FLOAT);
    constexpr int kInt = gl(GL_INT);
    constexpr int kUint = gl(GL_UNSIGNED_INT);

    using SIF = mgl::SizedInternalFormat;
    const std::array<FormatCase, 25> kFormats = {{
        {"R8", SIF::R8, GL_R8, {8, 0, 0, 0, 0, 0}, {kUnorm, kNone, kNone, kNone, kNone}},
        {"RG8", SIF::Rg8, GL_RG8, {8, 8, 0, 0, 0, 0}, {kUnorm, kUnorm, kNone, kNone, kNone}},
        {"RGB8", SIF::Rgb8, GL_RGB8, {8, 8, 8, 0, 0, 0}, {kUnorm, kUnorm, kUnorm, kNone, kNone}},
        {"RGBA8", SIF::Rgba8, GL_RGBA8, {8, 8, 8, 8, 0, 0}, {kUnorm, kUnorm, kUnorm, kUnorm, kNone}},
        {"RGB565", SIF::Rgb565, GL_RGB565, {5, 6, 5, 0, 0, 0}, {kUnorm, kUnorm, kUnorm, kNone, kNone}},
        {"RGB5_A1", SIF::Rgb5A1, GL_RGB5_A1, {5, 5, 5, 1, 0, 0}, {kUnorm, kUnorm, kUnorm, kUnorm, kNone}},
        {"RGBA4", SIF::Rgba4, GL_RGBA4, {4, 4, 4, 4, 0, 0}, {kUnorm, kUnorm, kUnorm, kUnorm, kNone}},
        {"RGB10_A2", SIF::Rgb10A2, GL_RGB10_A2, {10, 10, 10, 2, 0, 0}, {kUnorm, kUnorm, kUnorm, kUnorm, kNone}},
        {"RGB10_A2UI", SIF::Rgb10A2UI, GL_RGB10_A2UI, {10, 10, 10, 2, 0, 0}, {kUint, kUint, kUint, kUint, kNone}},
        {"SRGB8_ALPHA8", SIF::Srgb8Alpha8, GL_SRGB8_ALPHA8, {8, 8, 8, 8, 0, 0}, {kUnorm, kUnorm, kUnorm, kUnorm, kNone}},
        {"RGBA8_SNORM", SIF::Rgba8Snorm, GL_RGBA8_SNORM, {8, 8, 8, 8, 0, 0}, {kSnorm, kSnorm, kSnorm, kSnorm, kNone}},
        {"R16F", SIF::R16F, GL_R16F, {16, 0, 0, 0, 0, 0}, {kFloat, kNone, kNone, kNone, kNone}},
        {"RGBA16F", SIF::Rgba16F, GL_RGBA16F, {16, 16, 16, 16, 0, 0}, {kFloat, kFloat, kFloat, kFloat, kNone}},
        {"R32F", SIF::R32F, GL_R32F, {32, 0, 0, 0, 0, 0}, {kFloat, kNone, kNone, kNone, kNone}},
        {"R11F_G11F_B10F", SIF::R11FG11FB10F, GL_R11F_G11F_B10F, {11, 11, 10, 0, 0, 0}, {kFloat, kFloat, kFloat, kNone, kNone}},
        {"RGB9_E5", SIF::Rgb9E5, GL_RGB9_E5, {9, 9, 9, 0, 0, 0}, {kAny, kAny, kAny, kNone, kNone}},
        {"R8I", SIF::R8I, GL_R8I, {8, 0, 0, 0, 0, 0}, {kInt, kNone, kNone, kNone, kNone}},
        {"R16UI", SIF::R16UI, GL_R16UI, {16, 0, 0, 0, 0, 0}, {kUint, kNone, kNone, kNone, kNone}},
        {"RG32I", SIF::Rg32I, GL_RG32I, {32, 32, 0, 0, 0, 0}, {kInt, kInt, kNone, kNone, kNone}},
        {"RGBA32UI", SIF::Rgba32UI, GL_RGBA32UI, {32, 32, 32, 32, 0, 0}, {kUint, kUint, kUint, kUint, kNone}},
        {"DEPTH_COMPONENT16", SIF::DepthComponent16, GL_DEPTH_COMPONENT16, {0, 0, 0, 0, 16, 0}, {kNone, kNone, kNone, kNone, kUnorm}},
        {"DEPTH_COMPONENT24", SIF::DepthComponent24, GL_DEPTH_COMPONENT24, {0, 0, 0, 0, 24, 0}, {kNone, kNone, kNone, kNone, kUnorm}},
        {"DEPTH_COMPONENT32F", SIF::DepthComponent32F, GL_DEPTH_COMPONENT32F, {0, 0, 0, 0, 32, 0}, {kNone, kNone, kNone, kNone, kFloat}},
        {"DEPTH24_STENCIL8", SIF::Depth24Stencil8, GL_DEPTH24_STENCIL8, {0, 0, 0, 0, 24, 8}, {kNone, kNone, kNone, kNone, kUnorm}},
        {"DEPTH32F_STENCIL8", SIF::Depth32FStencil8, GL_DEPTH32F_STENCIL8, {0, 0, 0, 0, 32, 8}, {kNone, kNone, kNone, kNone, kFloat}},
    }};

    constexpr std::array<mgl::TextureLevelParameter, 19> kAllPnames = {
        mgl::TextureLevelParameter::Width,     mgl::TextureLevelParameter::Height,      mgl::TextureLevelParameter::Depth,
        mgl::TextureLevelParameter::InternalFormat, mgl::TextureLevelParameter::RedSize, mgl::TextureLevelParameter::GreenSize,
        mgl::TextureLevelParameter::BlueSize,  mgl::TextureLevelParameter::AlphaSize,   mgl::TextureLevelParameter::DepthSize,
        mgl::TextureLevelParameter::StencilSize, mgl::TextureLevelParameter::SharedSize, mgl::TextureLevelParameter::RedType,
        mgl::TextureLevelParameter::GreenType, mgl::TextureLevelParameter::BlueType,    mgl::TextureLevelParameter::AlphaType,
        mgl::TextureLevelParameter::DepthType, mgl::TextureLevelParameter::Compressed, mgl::TextureLevelParameter::Samples,
        mgl::TextureLevelParameter::FixedSampleLocations};

    struct Level
    {
        int w = -1, h = -1, d = -1, format = -1;
        std::array<int, 6> size{};  // R G B A D S
        int shared = -1;
        std::array<int, 5> type{};  // R G B A D
        int compressed = -1;
        int samples = -1, fixed = -1;  // TEXTURE_SAMPLES, TEXTURE_FIXED_SAMPLE_LOCATIONS
        bool float_agrees = true;   // glGetTexLevelParameterfv returned the same numbers
    };

    Level query_level(mgl::TextureTarget target, int level)
    {
        using P = mgl::TextureLevelParameter;
        Level l;
        const auto geti = [&](P p) {
            GLint v = -12345;
            mgl::glGetTexLevelParameteriv(target, level, p, &v);
            return static_cast<int>(v);
        };
        l.w = geti(P::Width);
        l.h = geti(P::Height);
        l.d = geti(P::Depth);
        l.format = geti(P::InternalFormat);
        l.size = {geti(P::RedSize), geti(P::GreenSize), geti(P::BlueSize), geti(P::AlphaSize), geti(P::DepthSize), geti(P::StencilSize)};
        l.shared = geti(P::SharedSize);
        l.type = {geti(P::RedType), geti(P::GreenType), geti(P::BlueType), geti(P::AlphaType), geti(P::DepthType)};
        l.compressed = geti(P::Compressed);
        l.samples = geti(P::Samples);
        l.fixed = geti(P::FixedSampleLocations);
        for (const P p : kAllPnames)
        {
            GLfloat f = -12345.0f;
            mgl::glGetTexLevelParameterfv(target, level, p, &f);
            // Integer and boolean state converted to float is exact (section 2.2.2).
            if (f != static_cast<float>(geti(p))) l.float_agrees = false;
        }
        return l;
    }

    std::string_view type_name(int t)
    {
        switch (t)
        {
        case kNone: return "NONE";
        case kUnorm: return "UNSIGNED_NORMALIZED";
        case kSnorm: return "SIGNED_NORMALIZED";
        case kFloat: return "FLOAT";
        case kInt: return "INT";
        case kUint: return "UNSIGNED_INT";
        default: return "?";
        }
    }

    struct Dim
    {
        int w, h, d;  // d = kAny: not asserted (two-dimensional images: the spec's "depth" is only defined for 3D and arrays)
    };

    /// Problems of `l` against what was allocated; empty when everything matches. Sizes only need to reach the minimum
    /// of the format table; absent components are checked through their type (NONE).
    /// What an allocation with glTexStorage2DMultisample must report: TEXTURE_SAMPLES in [at_least, at_most] (ES 3.2 8.8: not fewer than requested, not more than
    /// the next larger supported count) and the requested TEXTURE_FIXED_SAMPLE_LOCATIONS.
    struct MsExpect
    {
        int at_least, at_most;
        bool fixed;
    };

    std::string verify(const Level& l, const Dim& want, const FormatCase& f, bool compressed_expected = false, const std::optional<MsExpect>& ms = std::nullopt)
    {
        std::string problems;
        const auto bad = [&](const std::string& s) { problems += (problems.empty() ? "" : "; ") + s; };
        if (l.w != want.w) bad(std::format("WIDTH {} != {}", l.w, want.w));
        if (l.h != want.h) bad(std::format("HEIGHT {} != {}", l.h, want.h));
        if (want.d != kAny && l.d != want.d) bad(std::format("DEPTH {} != {}", l.d, want.d));
        if (l.format != gl(f.token)) bad(std::format("INTERNAL_FORMAT 0x{:X} != 0x{:X}", l.format, gl(f.token)));
        if (l.compressed != (compressed_expected ? 1 : 0)) bad(std::format("COMPRESSED {} != {}", l.compressed, compressed_expected ? 1 : 0));
        static constexpr std::array<std::string_view, 6> size_names = {"RED", "GREEN", "BLUE", "ALPHA", "DEPTH", "STENCIL"};
        for (std::size_t i = 0; i < 6; ++i)
            if (!compressed_expected && l.size[i] < f.min_bits[i]) bad(std::format("{}_SIZE {} < {}", size_names[i], l.size[i], f.min_bits[i]));
        for (std::size_t i = 0; i < 5; ++i)
            if (f.types[i] != kAny && l.type[i] != f.types[i])
                bad(std::format("{}_TYPE {} != {}", size_names[i], type_name(l.type[i]), type_name(f.types[i])));
        if (std::string_view(f.name) == "RGB9_E5" && l.shared < 5) bad(std::format("SHARED_SIZE {} < 5", l.shared));
        if (ms)
        {
            if (l.samples < ms->at_least || l.samples > ms->at_most) bad(std::format("TEXTURE_SAMPLES {} not in [{}, {}]", l.samples, ms->at_least, ms->at_most));
            if (l.fixed != (ms->fixed ? 1 : 0)) bad(std::format("TEXTURE_FIXED_SAMPLE_LOCATIONS {} != {}", l.fixed, ms->fixed ? 1 : 0));
        }
        else
        {
            // ES 3.2 8.11.3: "For non-multisample textures, the default values in table 21.11 are returned": samples 0, fixed sample locations TRUE.
            if (l.samples != 0) bad(std::format("TEXTURE_SAMPLES {} != 0 (non-multisample)", l.samples));
            if (l.fixed != 1) bad(std::format("TEXTURE_FIXED_SAMPLE_LOCATIONS {} != TRUE (non-multisample)", l.fixed));
        }
        if (!l.float_agrees) bad("glGetTexLevelParameterfv disagrees with glGetTexLevelParameteriv");
        return problems;
    }

    // ---------------------------------------------------------------------------------------------
    class StencilTexturing final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            scene_prog_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            show_depth_prog_ = demo::build_program(env, "shaders/quad.vert", "shaders/show_depth.frag");
            show_stencil_prog_ = demo::build_program(env, "shaders/quad.vert", "shaders/show_stencil.frag");
            dump_float_prog_ = build_compute(env, "shaders/dump_float.comp");
            dump_uint_prog_ = build_compute(env, "shaders/dump_uint.comp");

            u_rect_ = mgl::glGetUniformLocation(scene_prog_, "u_rect");
            u_size_ = mgl::glGetUniformLocation(scene_prog_, "u_size");
            u_depth_ = mgl::glGetUniformLocation(scene_prog_, "u_depth");

            mgl::glGenBuffers(1, &dump_buffer_);
            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glDisable(mgl::Capability::Dither);

            draws_ = make_draws();

            // Three depth/stencil textures with immutable storage: the scene is rendered into the first two, the third
            // is only cleared (the negative control of the read-back checks).
            d24s8_ = make_ds_texture(mgl::SizedInternalFormat::Depth24Stencil8);
            d32fs8_ = make_ds_texture(mgl::SizedInternalFormat::Depth32FStencil8);
            control_ = make_ds_texture(mgl::SizedInternalFormat::Depth24Stencil8);
            render_scene_into(d24s8_, true);
            render_scene_into(d32fs8_, true);
            render_scene_into(control_, false);

            // Fixtures for "the mode is ignored for textures that have no stencil component".
            build_fixtures();
        }

        void render(demo::Env& env, int /*frame*/) override { draw_contact_sheet(env); }

        void validate(demo::Env& env, demo::Check& check) override
        {
            max_color_samples_ = demo::get_integer(mgl::GetParameter::MaxColorTextureSamples);
            max_depth_samples_ = demo::get_integer(mgl::GetParameter::MaxDepthTextureSamples);
            expected_ = model_scene(draws_, draws_.size());

            check_model_sanity(check);
            check_texture_state(check);
            check_depth_stencil_readback(check);
            check_mode_ignored(check);
            check_contact_sheet(env, check);
            check_level_queries(check);
            check_level_errors(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glDeleteProgram(scene_prog_);
            mgl::glDeleteProgram(show_depth_prog_);
            mgl::glDeleteProgram(show_stencil_prog_);
            mgl::glDeleteProgram(dump_float_prog_);
            mgl::glDeleteProgram(dump_uint_prog_);
            mgl::glDeleteBuffers(1, &dump_buffer_);
            mgl::glDeleteFramebuffers(1, &fbo_);
            if (!textures_.empty()) mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
            if (!query_fbos_.empty()) mgl::glDeleteFramebuffers(static_cast<GLsizei>(query_fbos_.size()), query_fbos_.data());
        }

    private:
        // ------------------------------------------------------------------------------------------
        // Setup helpers.
        // ------------------------------------------------------------------------------------------
        static mgl::ProgramId build_compute(const demo::Env& env, std::string_view file)
        {
            std::string log;
            const mgl::ShaderId shader = demo::compile_shader(mgl::ShaderType::Compute, env.load_text(file), log);
            if (shader.value == 0) throw demo::Failure(std::format("compute shader {} failed to compile:\n{}", file, log));
            const std::array<mgl::ShaderId, 1> shaders = {shader};
            const mgl::ProgramId program = demo::link_program(shaders, log);
            mgl::glDeleteShader(shader);
            if (program.value == 0) throw demo::Failure(std::format("compute program {} failed to link:\n{}", file, log));
            return program;
        }

        mgl::TextureId new_texture()
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            textures_.push_back(t);
            return t;
        }

        mgl::TextureId make_ds_texture(mgl::SizedInternalFormat format)
        {
            const mgl::TextureId t = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, kTex, kTex);
            // NEAREST is required for stencil sampling (ES 3.2 8.16) and, with TEXTURE_COMPARE_MODE NONE, for sampling a
            // depth format at all (8.17: sized depth and depth/stencil formats are not filterable).
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            return t;
        }

        /// Attaches `tex` as the only attachment (DEPTH_STENCIL_ATTACHMENT, no color buffer) of the helper framebuffer.
        void attach_ds(mgl::TextureId tex)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::DepthStencil,
                                        mgl::TextureTarget::Texture2D, tex, 0);
            const std::array<mgl::DrawBuffer, 1> none = {mgl::DrawBuffer::None};
            mgl::glDrawBuffers(none);
            mgl::glReadBuffer(mgl::ReadBuffer::None);
        }

        void render_scene_into(mgl::TextureId tex, bool draw_scene)
        {
            attach_ds(tex);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("a framebuffer with only a depth/stencil texture is not complete: {}", demo::framebuffer_status_name(status)));

            mgl::glViewport(0, 0, kTex, kTex);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glDepthMask(GL_TRUE);
            mgl::glStencilMask(0xFFu);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);

            if (draw_scene)
            {
                mgl::glUseProgram(scene_prog_);
                mgl::glUniform1f(u_size_, static_cast<float>(kTex));
                for (const Draw& d : draws_)
                {
                    mgl::glUniform4f(u_rect_, static_cast<float>(d.rect.x0), static_cast<float>(d.rect.y0), static_cast<float>(d.rect.x1),
                                     static_cast<float>(d.rect.y1));
                    mgl::glUniform1f(u_depth_, static_cast<float>(d.depth));
                    mgl::glDepthFunc(d.depth_func);
                    mgl::glStencilFunc(d.stencil_func, d.ref, d.func_mask);
                    mgl::glStencilOp(d.sfail, d.dpfail, d.dppass);
                    mgl::glStencilMask(d.write_mask);
                    mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
                }
            }

            // Back to the initial state of the tests.
            mgl::glStencilMask(0xFFu);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFu);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDisable(mgl::Capability::StencilTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            // Detach, so the next user of the helper framebuffer starts without attachments (depth and stencil
            // attachments must be the same image, ES 3.2 9.4.2).
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::DepthStencil, mgl::TextureTarget::Texture2D,
                                        mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void build_fixtures()
        {
            // A 4 x 4 RGBA8 texture with known colors (r = 40 x + 10, g = 60 y + 5, b = 200, a = 255 - 30 x).
            rgba_fixture_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, rgba_fixture_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kSmall, kSmall);
            std::vector<std::uint8_t> bytes;
            for (int y = 0; y < kSmall; ++y)
                for (int x = 0; x < kSmall; ++x)
                {
                    bytes.push_back(static_cast<std::uint8_t>(40 * x + 10));
                    bytes.push_back(static_cast<std::uint8_t>(60 * y + 5));
                    bytes.push_back(200);
                    bytes.push_back(static_cast<std::uint8_t>(255 - 30 * x));
                }
            rgba_bytes_ = bytes;
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kSmall, kSmall, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, bytes.data());
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);

            // A DEPTH_COMPONENT32F texture cleared to depth 0.5 through a framebuffer.
            depth_fixture_ = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, depth_fixture_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::DepthComponent32F, kSmall, kSmall);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, mgl::TextureTarget::Texture2D, depth_fixture_, 0);
            const std::array<mgl::DrawBuffer, 1> none = {mgl::DrawBuffer::None};
            mgl::glDrawBuffers(none);
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("depth-only framebuffer is not complete");
            mgl::glDepthMask(GL_TRUE);
            mgl::glClearDepthf(0.5f);
            mgl::glClear(mgl::ClearBufferBit::Depth);
            mgl::glClearDepthf(1.0f);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // ------------------------------------------------------------------------------------------
        // The contact sheet: the four (texture, mode) pairs through fragment shaders.
        // ------------------------------------------------------------------------------------------
        struct Panel
        {
            mgl::TextureId tex;
            bool stencil;
            int col, row;
        };

        [[nodiscard]] std::array<Panel, 4> panels() const
        {
            return {{{d24s8_, false, 0, 0}, {d24s8_, true, 1, 0}, {d32fs8_, false, 0, 1}, {d32fs8_, true, 1, 1}}};
        }

        static int panel_size(const demo::Env& env) { return std::max(1, std::min(env.width, env.height) / 2 / kTex) * kTex; }

        void draw_contact_sheet(const demo::Env& env)
        {
            const int size = panel_size(env);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.10f, 0.10f, 0.14f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (const Panel& p : panels())
            {
                mgl::glUseProgram(p.stencil ? show_stencil_prog_ : show_depth_prog_);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, p.tex);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::DepthStencilTextureMode,
                                     p.stencil ? gl(GL_STENCIL_INDEX) : gl(GL_DEPTH_COMPONENT));
                mgl::glViewport(p.col * size, p.row * size, size, size);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
            mgl::glViewport(0, 0, env.width, env.height);
        }

        // ------------------------------------------------------------------------------------------
        // Shader read-back: dump a texture through a compute shader into the SSBO and map it.
        // ------------------------------------------------------------------------------------------
        std::vector<std::uint32_t> dump(mgl::ProgramId program, mgl::TextureId tex, int w, int h)
        {
            const std::size_t words = static_cast<std::size_t>(8 * w * h);
            const std::vector<std::uint32_t> sentinel(words, 0xDEADBEEFu);  // a texel the shader never wrote stays recognizable
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, dump_buffer_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(words * sizeof(std::uint32_t)), sentinel.data(), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, dump_buffer_);

            mgl::glUseProgram(program);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glDispatchCompute(static_cast<GLuint>((w + 7) / 8), static_cast<GLuint>((h + 7) / 8), 1);
            // The mapped read below is a buffer-object access by the commands of section 6 (ES 3.2 7.11.2): BUFFER_UPDATE_BARRIER_BIT.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);

            std::vector<std::uint32_t> out(words);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, static_cast<GLsizeiptr>(words * sizeof(std::uint32_t)), mgl::MapBufferAccessMask::Read);
            if (mapped == nullptr) throw demo::Failure("glMapBufferRange on the dump buffer returned null");
            std::memcpy(out.data(), mapped, words * sizeof(std::uint32_t));
            (void)mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage);
            return out;
        }

        void set_mode(mgl::TextureId tex, int mode)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::DepthStencilTextureMode, mode);
        }

        static int get_mode()
        {
            GLint v = -1;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::DepthStencilTextureMode, &v);
            return static_cast<int>(v);
        }

        struct DepthResult
        {
            int mismatches = 0;
            double max_error = 0.0;
            std::string first;
        };

        /// Compares a depth dump (both halves: texelFetch and textureLod) with the model.
        DepthResult compare_depth(const std::vector<std::uint32_t>& data, const std::vector<Pixel>& model, double tolerance) const
        {
            DepthResult r;
            const std::size_t count = model.size();
            for (std::size_t half = 0; half < 2; ++half)
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double got = static_cast<double>(float_of(data[half * 4 * count + 4 * i]));
                    const double err = std::abs(got - model[i].depth);
                    r.max_error = std::max(r.max_error, err);
                    if (err > tolerance && r.mismatches++ == 0)
                        r.first = std::format("{} at ({},{}): got {:.8f} expected {:.8f}", half == 0 ? "texelFetch" : "textureLod", i % kTex, i / kTex, got, model[i].depth);
                }
            return r;
        }

        struct StencilResult
        {
            int mismatches = 0;
            std::string first;
        };

        StencilResult compare_stencil(const std::vector<std::uint32_t>& data, const std::vector<Pixel>& model) const
        {
            StencilResult r;
            const std::size_t count = model.size();
            for (std::size_t half = 0; half < 2; ++half)
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::uint32_t got = data[half * 4 * count + 4 * i];
                    if (got != static_cast<std::uint32_t>(model[i].stencil) && r.mismatches++ == 0)
                        r.first = std::format("{} at ({},{}): got {} expected {}", half == 0 ? "texelFetch" : "textureLod", i % kTex, i / kTex, got, model[i].stencil);
                }
            return r;
        }

        // ------------------------------------------------------------------------------------------
        // Checks.
        // ------------------------------------------------------------------------------------------
        void check_model_sanity(demo::Check& check) const
        {
            // The scene must be able to distinguish the bugs the read-backs look for.
            std::set<int> stencils;
            std::set<double> depths;
            int or_all = 0;
            for (const Pixel& p : expected_)
            {
                stencils.insert(p.stencil);
                depths.insert(p.depth);
                or_all |= p.stencil;
            }
            const std::vector<Pixel> after_one = model_scene(draws_, kPassOneDraws);
            int differing = 0;
            for (std::size_t i = 0; i < expected_.size(); ++i)
                differing += (after_one[i].stencil != expected_[i].stencil || after_one[i].depth != expected_[i].depth) ? 1 : 0;
            std::cout << std::format("   note: the scene has {} distinct stencil values, {} distinct depths; passes 2a/2b change {} of {} texels\n", stencils.size(),
                                     depths.size(), differing, expected_.size());
            check.expect(stencils.size() >= 16 && depths.size() >= 10 && or_all == 0xFF && differing > 400,
                         "test design: the CPU model of the scene has >= 16 distinct stencil values using all 8 bits, >= 10 distinct depths, and passes 2a/2b change > 400 texels",
                         std::format("stencil {}, depth {}, OR {:#x}, changed {}", stencils.size(), depths.size(), or_all, differing));
        }

        void check_texture_state(demo::Check& check)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, d24s8_);
            // ES 3.2 8.19: "The value of DEPTH_STENCIL_TEXTURE_MODE is DEPTH_COMPONENT" initially.
            mgl::TextureId fresh{};
            mgl::glGenTextures(1, &fresh);
            textures_.push_back(fresh);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, fresh);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Depth24Stencil8, kSmall, kSmall);
            check.expect_eq(get_mode(), gl(GL_DEPTH_COMPONENT), "a new texture has DEPTH_STENCIL_TEXTURE_MODE = DEPTH_COMPONENT (ES 3.2 8.19)");

            set_mode(fresh, gl(GL_STENCIL_INDEX));
            check.expect_no_gl_error("DEPTH_STENCIL_TEXTURE_MODE = STENCIL_INDEX is accepted");
            check.expect_eq(get_mode(), gl(GL_STENCIL_INDEX), "glGetTexParameteriv returns STENCIL_INDEX after glTexParameteri");

            // Table 8.19: the legal values are DEPTH_COMPONENT and STENCIL_INDEX; anything else is INVALID_ENUM and changes nothing.
            (void)demo::drain_gl_errors();
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::DepthStencilTextureMode, gl(GL_RED));
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "DEPTH_STENCIL_TEXTURE_MODE = RED -> INVALID_ENUM (ES 3.2 8.10 errors)");
            check.expect_eq(get_mode(), gl(GL_STENCIL_INDEX), "the rejected value left the mode unchanged");
            set_mode(fresh, gl(GL_DEPTH_COMPONENT));
            check.expect_eq(get_mode(), gl(GL_DEPTH_COMPONENT), "the mode can be switched back to DEPTH_COMPONENT");

            // Immutable storage: levels / format, the texture is complete for NEAREST sampling.
            GLint immutable = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::ImmutableFormat, &immutable);
            check.expect_eq(immutable, gl(GL_TRUE), "TEXTURE_IMMUTABLE_FORMAT = TRUE for a glTexStorage2D depth/stencil texture");
        }

        void check_depth_stencil_readback(demo::Check& check)
        {
            struct Subject
            {
                std::string_view name;
                mgl::TextureId tex;
                double depth_tolerance;
                std::string_view tolerance_reason;
            };
            // Tolerances. A fixed-point depth buffer of m bits stores k / (2^m - 1) (ES 3.2 12.6.1); how a window z is
            // rounded to k is not specified, so the stored value may be one step away from z, and converting k / (2^24 - 1)
            // to float adds less than half a step: two steps = 2 / (2^24 - 1). A floating-point depth buffer stores the
            // single-precision window z; with the exactly representable depths used here 1e-6 only absorbs the arithmetic
            // of the viewport transform.
            const std::array<Subject, 2> subjects = {{
                {"DEPTH24_STENCIL8", d24s8_, 2.0 / 16777215.0, "two 24-bit steps"},
                {"DEPTH32F_STENCIL8", d32fs8_, 1e-6, "float arithmetic of the viewport transform"},
            }};
            for (const Subject& s : subjects)
            {
                // Depth: mode DEPTH_COMPONENT (the initial value), sampler2D.
                set_mode(s.tex, gl(GL_DEPTH_COMPONENT));
                const std::vector<std::uint32_t> depth_dump = dump(dump_float_prog_, s.tex, kTex, kTex);
                const DepthResult depth = compare_depth(depth_dump, expected_, s.depth_tolerance);
                check.expect(depth.mismatches == 0,
                             std::format("{}: all {} texels of the depth component read in the shader (sampler2D, DEPTH_COMPONENT; texelFetch and textureLod) equal the CPU model (tolerance {})", s.name, kTex * kTex, s.tolerance_reason),
                             depth.mismatches == 0 ? std::format("max error {:.3g}", depth.max_error) : std::format("{} mismatches, {}", depth.mismatches, depth.first));
                bool unwritten = false;
                for (const std::uint32_t w : depth_dump) unwritten = unwritten || w == 0xDEADBEEFu;
                check.expect(!unwritten, std::format("{}: the compute shader wrote every word of the depth dump (no sentinel left)", s.name));

                // Stencil: mode STENCIL_INDEX, usampler2D.
                set_mode(s.tex, gl(GL_STENCIL_INDEX));
                check.expect_eq(get_mode(), gl(GL_STENCIL_INDEX), std::format("{}: the mode reads back as STENCIL_INDEX", s.name));
                const std::vector<std::uint32_t> stencil_dump = dump(dump_uint_prog_, s.tex, kTex, kTex);
                const StencilResult stencil = compare_stencil(stencil_dump, expected_);
                check.expect(stencil.mismatches == 0,
                             std::format("{}: all {} stencil values read in the shader (usampler2D, STENCIL_INDEX; texelFetch and textureLod) equal the CPU model exactly", s.name, kTex * kTex),
                             stencil.mismatches == 0 ? std::string("exact") : std::format("{} mismatches, {}", stencil.mismatches, stencil.first));
                // The stencil dump is a vec4-like (s, 0, 0, 1) for a RED texture: the other components follow the ordinary
                // rules for a one-component texture (GLSL ES 3.10 section 8.9 / ES 3.2 8.20 "treated as RED").
                bool gba_ok = true;
                for (std::size_t i = 0; i < expected_.size(); ++i)
                    gba_ok = gba_ok && stencil_dump[4 * i + 1] == 0u && stencil_dump[4 * i + 2] == 0u;
                // Not asserted: the spec leaves the green/blue/alpha result of an unsigned stencil fetch open in this text; print it.
                std::cout << std::format("   note: {} stencil fetch returns g = b = 0 everywhere: {}; alpha of texel 0 = {}\n", s.name, gba_ok ? "yes" : "no", stencil_dump[3]);

                // Switching back restores the depth view: the mode is texture state, nothing was lost in between.
                set_mode(s.tex, gl(GL_DEPTH_COMPONENT));
                const DepthResult again = compare_depth(dump(dump_float_prog_, s.tex, kTex, kTex), expected_, s.depth_tolerance);
                check.expect(again.mismatches == 0, std::format("{}: after switching back to DEPTH_COMPONENT the depth reads exactly as before", s.name),
                             std::format("{} mismatches", again.mismatches));
            }

            // The depth and the stencil of the two formats are the same pattern: stencil values are exact in both, depths agree within the
            // looser tolerance. (This also shows the formats were rendered by the same passes.)
            // Negative control: a texture that was only cleared reports depth 1 / stencil 0 and must fail the same comparisons.
            set_mode(control_, gl(GL_DEPTH_COMPONENT));
            const DepthResult control_depth = compare_depth(dump(dump_float_prog_, control_, kTex, kTex), expected_, 2.0 / 16777215.0);
            set_mode(control_, gl(GL_STENCIL_INDEX));
            const StencilResult control_stencil = compare_stencil(dump(dump_uint_prog_, control_, kTex, kTex), expected_);
            set_mode(control_, gl(GL_DEPTH_COMPONENT));
            check.expect(control_depth.mismatches > 400 && control_stencil.mismatches > 400,
                         "negative control: a depth/stencil texture that was only cleared fails the same depth and stencil comparisons (> 400 mismatching reads each)",
                         std::format("depth {} stencil {}", control_depth.mismatches, control_stencil.mismatches));
            // Negative control 2: the stencil dump compared against the model of pass 1 only (a stale expectation) must also fail.
            set_mode(d24s8_, gl(GL_STENCIL_INDEX));
            const StencilResult stale = compare_stencil(dump(dump_uint_prog_, d24s8_, kTex, kTex), model_scene(draws_, kPassOneDraws));
            set_mode(d24s8_, gl(GL_DEPTH_COMPONENT));
            check.expect(stale.mismatches > 400, "negative control: the model of pass 1 alone (stale expectation) does not match the final stencil (> 400 mismatching reads)",
                         std::format("{} mismatches", stale.mismatches));
        }

        void check_mode_ignored(demo::Check& check)
        {
            // ES 3.2 8.16: "The DEPTH_STENCIL_TEXTURE_MODE is ignored for non depth/stencil textures." Set the mode to STENCIL_INDEX
            // and read two such textures through a sampler2D: the color texture still returns its colors, the depth texture its depth.
            set_mode(rgba_fixture_, gl(GL_STENCIL_INDEX));
            const std::vector<std::uint32_t> rgba = dump(dump_float_prog_, rgba_fixture_, kSmall, kSmall);
            double worst = 0.0;
            int bad = 0;
            for (int i = 0; i < kSmall * kSmall; ++i)
                for (int c = 0; c < 4; ++c)
                    for (int half = 0; half < 2; ++half)
                    {
                        const double got = static_cast<double>(float_of(rgba[static_cast<std::size_t>(half * 4 * kSmall * kSmall + 4 * i + c)]));
                        const double want = rgba_bytes_[static_cast<std::size_t>(4 * i + c)] / 255.0;
                        worst = std::max(worst, std::abs(got - want));
                        bad += std::abs(got - want) > 1.0 / 255.0 / 4.0 ? 1 : 0;
                    }
            // RGBA8 texels are k / 255 exactly in an UNORM fetch; allow a quarter of a step for filtering-free float conversion.
            check.expect(bad == 0, "an RGBA8 texture with DEPTH_STENCIL_TEXTURE_MODE = STENCIL_INDEX still samples its colors (the mode is ignored; tolerance 1/4 of an 8-bit step)",
                         std::format("max error {:.3g}, {} bad", worst, bad));

            set_mode(depth_fixture_, gl(GL_STENCIL_INDEX));
            const std::vector<std::uint32_t> depth = dump(dump_float_prog_, depth_fixture_, kSmall, kSmall);
            int bad_depth = 0;
            for (int i = 0; i < kSmall * kSmall; ++i)
                for (int half = 0; half < 2; ++half)
                    bad_depth += float_of(depth[static_cast<std::size_t>(half * 4 * kSmall * kSmall + 4 * i)]) != 0.5f ? 1 : 0;
            check.expect(bad_depth == 0, "a DEPTH_COMPONENT32F texture cleared to 0.5 with DEPTH_STENCIL_TEXTURE_MODE = STENCIL_INDEX still samples 0.5 (the mode is ignored)",
                         std::format("{} of {} reads differ", bad_depth, 2 * kSmall * kSmall));
        }

        void check_contact_sheet(const demo::Env& env, demo::Check& check)
        {
            draw_contact_sheet(env);
            const int size = panel_size(env);
            const int scale = size / kTex;
            const std::vector<std::uint8_t> image = demo::read_rgba8(0, 0, env.width, env.height);
            const auto pixel = [&](int x, int y) {
                const std::size_t i = 4 * (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x));
                return demo::Rgba8{image[i], image[i + 1], image[i + 2], image[i + 3]};
            };
            for (const Panel& p : panels())
            {
                int bad = 0;
                int max_error = 0;
                std::string first;
                for (int ty = 0; ty < kTex; ++ty)
                    for (int tx = 0; tx < kTex; ++tx)
                    {
                        const Pixel& m = expected_[static_cast<std::size_t>(ty * kTex + tx)];
                        const demo::Rgba8 want = p.stencil ? stencil_color(m.stencil) : demo::Rgba8{to8(m.depth), to8(m.depth), to8(m.depth), 255};
                        const demo::Rgba8 got = pixel(p.col * size + tx * scale + scale / 2, p.row * size + ty * scale + scale / 2);
                        int worst = 0;
                        for (std::size_t c = 0; c < 4; ++c) worst = std::max(worst, std::abs(int{got[c]} - int{want[c]}));
                        max_error = std::max(max_error, worst);
                        if (worst > 1 && bad++ == 0) first = std::format("texel ({},{}): got {} expected {}", tx, ty, demo::format_rgba(got), demo::format_rgba(want));
                    }
                const bool is24 = p.tex.value == d24s8_.value;
                check.expect(bad == 0,
                             std::format("contact sheet: the {} panel of {} (fragment shader, {}) shows every texel as the model predicts (tolerance 1: 8-bit quantization)",
                                         p.stencil ? "stencil" : "depth", is24 ? "DEPTH24_STENCIL8" : "DEPTH32F_STENCIL8", p.stencil ? "usampler2D" : "sampler2D"),
                             bad == 0 ? std::format("{} texels, max error {}", kTex * kTex, max_error) : std::format("{} bad, {}", bad, first));
            }
        }

        // ------------------------------------------------------------------------------------------
        // Part 2: glGetTexLevelParameter*.
        // ------------------------------------------------------------------------------------------
        const FormatCase& format_case(std::string_view name) const
        {
            for (const FormatCase& f : kFormats)
                if (f.name == name) return f;
            throw demo::Failure("unknown format case");
        }

        void check_level_queries(demo::Check& check)
        {
            check_format_sweep(check);
            check_dimension_cases(check);
            check_mutable_cases(check);
            check_compressed(check);
            check_multisample(check);
        }

        void check_format_sweep(demo::Check& check)
        {
            std::set<std::string> absent_nonzero;
            int formats_ok = 0;
            std::string problems;
            for (const FormatCase& f : kFormats)
            {
                const mgl::TextureId t = new_texture();
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
                (void)demo::drain_gl_errors();
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, f.sized, 8, 4);
                const mgl::ErrorCode storage_error = demo::drain_gl_errors();
                const Level l = query_level(mgl::TextureTarget::Texture2D, 0);
                std::string p = storage_error == mgl::ErrorCode::NoError ? verify(l, {8, 4, kAny}, f) : std::string("glTexStorage2D raised an error");
                if (!p.empty()) problems += std::format("{}{}: {}", problems.empty() ? "" : " | ", f.name, p);
                else ++formats_ok;
                static constexpr std::array<std::string_view, 6> names = {"R", "G", "B", "A", "D", "S"};
                for (std::size_t i = 0; i < 6; ++i)
                    if (f.min_bits[i] == 0 && l.size[i] != 0) absent_nonzero.insert(std::format("{}:{}={}", f.name, names[i], l.size[i]));
            }
            check.expect(problems.empty(),
                         std::format("{} sized internal formats (8 x 4, glTexStorage2D): WIDTH, HEIGHT, INTERNAL_FORMAT, COMPRESSED = FALSE, component sizes >= the minimum of ES 3.2 tables 8.10/8.11, component types, and the float query equal the integer query",
                                     kFormats.size()),
                         problems.empty() ? std::format("{} of {} formats", formats_ok, kFormats.size()) : problems);
            std::cout << std::format("   note: sizes reported for components a format does not have (not asserted; absent components are identified by type NONE): {}\n",
                                     absent_nonzero.empty() ? std::string("all 0") : [&] {
                                         std::string s;
                                         for (const auto& a : absent_nonzero) s += a + " ";
                                         return s;
                                     }());
        }

        void check_dimension_cases(demo::Check& check)
        {
            const FormatCase& rgba8 = format_case("RGBA8");
            const auto level_expect = [&](std::string what, mgl::TextureTarget target, int level, Dim dim) {
                const Level l = query_level(target, level);
                const std::string problems = verify(l, dim, rgba8);
                check.expect(problems.empty(), std::move(what), problems.empty() ? std::format("{} x {} x {}", l.w, l.h, l.d) : problems);
            };

            // 2D, immutable, 3 levels: 16 x 8, 8 x 4, 4 x 2.
            const mgl::TextureId t2d = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t2d);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 3, mgl::SizedInternalFormat::Rgba8, 16, 8);
            for (int level = 0; level < 3; ++level)
                level_expect(std::format("TEXTURE_2D 16 x 8 with 3 levels, level {}: {} x {}", level, 16 >> level, 8 >> level), mgl::TextureTarget::Texture2D, level,
                             {16 >> level, 8 >> level, kAny});

            // 3D, immutable, 3 levels: 8x4x6, 4x2x3, 2x1x1 (depth halves too, ES 3.2 8.14.3 / 8.18).
            const mgl::TextureId t3d = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, t3d);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture3D, 3, mgl::SizedInternalFormat::Rgba8, 8, 4, 6);
            constexpr std::array<Dim, 3> dims3d = {{{8, 4, 6}, {4, 2, 3}, {2, 1, 1}}};
            for (int level = 0; level < 3; ++level)
                level_expect(std::format("TEXTURE_3D 8 x 4 x 6 with 3 levels, level {}: {} x {} x {}", level, dims3d[static_cast<std::size_t>(level)].w, dims3d[static_cast<std::size_t>(level)].h, dims3d[static_cast<std::size_t>(level)].d),
                             mgl::TextureTarget::Texture3D, level, dims3d[static_cast<std::size_t>(level)]);

            // 2D array: the layer count does not shrink with the level.
            const mgl::TextureId tarr = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tarr);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 4, mgl::SizedInternalFormat::Rgba8, 8, 8, 5);
            for (int level = 0; level < 4; ++level)
                level_expect(std::format("TEXTURE_2D_ARRAY 8 x 8 x 5 layers with 4 levels, level {}: {} x {} x 5 (the layer count stays 5)", level, 8 >> level, 8 >> level),
                             mgl::TextureTarget::Texture2DArray, level, {8 >> level, 8 >> level, 5});

            // Cube map: the query takes a face target (TEXTURE_CUBE_MAP itself is not a valid target).
            const mgl::TextureId tcube = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tcube);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 4, mgl::SizedInternalFormat::Rgba8, 8, 8);
            constexpr std::array<std::pair<mgl::TextureTarget, std::string_view>, 6> faces = {{
                {mgl::TextureTarget::TextureCubeMapPositiveX, "+X"}, {mgl::TextureTarget::TextureCubeMapNegativeX, "-X"},
                {mgl::TextureTarget::TextureCubeMapPositiveY, "+Y"}, {mgl::TextureTarget::TextureCubeMapNegativeY, "-Y"},
                {mgl::TextureTarget::TextureCubeMapPositiveZ, "+Z"}, {mgl::TextureTarget::TextureCubeMapNegativeZ, "-Z"}}};
            std::string cube_problems;
            int cube_levels = 0;
            for (const auto& [face, name] : faces)
                for (int level = 0; level < 4; ++level)
                {
                    const std::string p = verify(query_level(face, level), {8 >> level, 8 >> level, kAny}, rgba8);
                    if (!p.empty()) cube_problems += std::format("{} level {}: {}; ", name, level, p);
                    ++cube_levels;
                }
            check.expect(cube_problems.empty(), std::format("cube map 8 x 8 with 4 levels: all {} face/level images report the allocated size, format and RGBA8 resolutions", cube_levels),
                         cube_problems.empty() ? std::string("6 faces x 4 levels") : cube_problems);

            // Negative control: verify() is not vacuous. The very same queries judged against a wrong expectation (the
            // wrong level's size, the wrong format) must report problems.
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t2d);
            const Level l0 = query_level(mgl::TextureTarget::Texture2D, 0);
            const std::string wrong_size = verify(l0, {8, 4, kAny}, rgba8);
            const std::string wrong_format = verify(l0, {16, 8, kAny}, format_case("R8"));
            check.expect(wrong_size.contains("WIDTH") && wrong_size.contains("HEIGHT") && wrong_format.contains("INTERNAL_FORMAT") && wrong_format.contains("GREEN_TYPE"),
                         "negative control: level 0 of the 16 x 8 RGBA8 texture judged as 8 x 4 / as R8 is rejected (WIDTH, HEIGHT, INTERNAL_FORMAT and GREEN_TYPE mismatches)",
                         wrong_size + " | " + wrong_format);

            // A level beyond the allocated ones of an immutable texture: nothing was ever specified there.
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t2d);
            const Level beyond2d = query_level(mgl::TextureTarget::Texture2D, 3);
            check.expect(beyond2d.w == 0 && beyond2d.h == 0, "TEXTURE_2D (3 levels): level 3 was never allocated, WIDTH = HEIGHT = 0 (initial per-image state, ES 3.2 table 21.11)",
                         std::format("{} x {}", beyond2d.w, beyond2d.h));
        }

        void check_mutable_cases(demo::Check& check)
        {
            const FormatCase& rgba8 = format_case("RGBA8");
            const FormatCase& rg8 = format_case("RG8");

            // Mutable 2D texture: images are specified one by one with glTexImage2D and can be replaced.
            const mgl::TextureId t = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 16, 16, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 1, mgl::InternalFormat::Rgba8, 8, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            check.expect(verify(query_level(mgl::TextureTarget::Texture2D, 0), {16, 16, kAny}, rgba8).empty() && verify(query_level(mgl::TextureTarget::Texture2D, 1), {8, 8, kAny}, rgba8).empty(),
                         "mutable TEXTURE_2D: glTexImage2D RGBA8 16 x 16 (level 0) and 8 x 8 (level 1) are reported as specified");
            const Level undefined = query_level(mgl::TextureTarget::Texture2D, 2);
            check.expect(undefined.w == 0 && undefined.h == 0, "mutable TEXTURE_2D: level 2 was never specified, WIDTH = HEIGHT = 0", std::format("{} x {}", undefined.w, undefined.h));
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rg8, 4, 2, 0, mgl::PixelFormat::Rg, mgl::PixelType::UnsignedByte, nullptr);
            const std::string respec = verify(query_level(mgl::TextureTarget::Texture2D, 0), {4, 2, kAny}, rg8);
            check.expect(respec.empty(), "mutable TEXTURE_2D: respecifying level 0 as RG8 4 x 2 changes WIDTH, HEIGHT, INTERNAL_FORMAT and the component resolutions/types (BLUE and ALPHA type NONE)",
                         respec.empty() ? std::string("RG8 4 x 2") : respec);
            check.expect(verify(query_level(mgl::TextureTarget::Texture2D, 1), {8, 8, kAny}, rgba8).empty(), "...and leaves level 1 (RGBA8 8 x 8) as it was");

            // Mutable 3D texture: the depth is part of the specification.
            const mgl::TextureId t3 = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, t3);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 0, mgl::InternalFormat::Rgba8, 4, 4, 4, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            const bool first = verify(query_level(mgl::TextureTarget::Texture3D, 0), {4, 4, 4}, rgba8).empty();
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 0, mgl::InternalFormat::Rgba8, 2, 2, 8, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            const std::string second = verify(query_level(mgl::TextureTarget::Texture3D, 0), {2, 2, 8}, rgba8);
            check.expect(first && second.empty(), "mutable TEXTURE_3D: glTexImage3D 4 x 4 x 4 then 2 x 2 x 8: WIDTH, HEIGHT and DEPTH follow the latest specification", second);
        }

        void check_compressed(demo::Check& check)
        {
            // COMPRESSED_RGB8_ETC2 is a required ES 3.0 format. ES 3.2 8.11.3: "the types returned specify how components are
            // interpreted after decompression, while the resolutions returned ... should be treated only as rough approximations".
            const mgl::TextureId t = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            (void)demo::drain_gl_errors();
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 2, mgl::CompressedInternalFormat::Rgb8Etc2, 8, 8);
            check.expect_no_gl_error("glTexStorage2D with COMPRESSED_RGB8_ETC2 (8 x 8, 2 levels) is accepted");
            std::string problems, type_problems;
            for (int level = 0; level < 2; ++level)
            {
                const Level l = query_level(mgl::TextureTarget::Texture2D, level);
                if (l.w != (8 >> level) || l.h != (8 >> level)) problems += std::format("level {} size {} x {}; ", level, l.w, l.h);
                if (l.format != gl(GL_COMPRESSED_RGB8_ETC2)) problems += std::format("level {} INTERNAL_FORMAT 0x{:X}; ", level, l.format);
                if (l.compressed != 1) problems += std::format("level {} COMPRESSED {}; ", level, l.compressed);
                for (std::size_t i = 0; i < 3; ++i)
                    if (l.type[i] != kUnorm) type_problems += std::format("level {} component {} type {}; ", level, i, type_name(l.type[i]));
                if (!l.float_agrees) problems += "float query differs; ";
                if (level == 0)
                    std::cout << std::format("   note: ETC2 RGB8 resolutions reported (rough approximations, not asserted): R{} G{} B{} A{}\n", l.size[0], l.size[1], l.size[2], l.size[3]);
            }
            check.expect(problems.empty(), "COMPRESSED_RGB8_ETC2 8 x 8, levels 0 and 1: WIDTH, HEIGHT, INTERNAL_FORMAT as specified, COMPRESSED = TRUE",
                         problems.empty() ? std::string("ok") : problems);
            check.expect_or_known_deviation(type_problems.empty(),
                                            "COMPRESSED_RGB8_ETC2 levels 0 and 1: TEXTURE_RED/GREEN/BLUE_TYPE = UNSIGNED_NORMALIZED (the type after decompression)",
                                            type_problems.empty() ? std::string("ok") : type_problems, "ANGLE",
                                            "ANGLE returns NONE for the component types of a compressed format; ES 3.2 8.11.3: \"For texture images "
                                            "with compressed internal formats, the types returned specify how components are interpreted after "
                                            "decompression\"");
        }

        /// The smallest sample count glGetInternalformativ lists for a multisample texture of `format` that is >= `request` (the request itself when none is).
        static int next_supported_samples(mgl::InternalFormat format, int request)
        {
            GLint count = 0;
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, format, mgl::InternalFormatParameter::NumSampleCounts, 1, &count);
            std::vector<GLint> counts(static_cast<std::size_t>(std::max(count, 0)));
            if (!counts.empty())
                mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisample, format, mgl::InternalFormatParameter::Samples, static_cast<GLsizei>(counts.size()), counts.data());
            int best = request;
            bool found = false;
            for (const GLint c : counts)
                if (c >= request && (!found || c < best))
                {
                    best = c;
                    found = true;
                }
            return best;
        }

        void check_multisample(demo::Check& check)
        {
            const int color_samples = std::max(1, std::min(4, max_color_samples_));
            const int depth_samples = std::max(1, std::min(4, max_depth_samples_));
            const int color_limit = next_supported_samples(mgl::InternalFormat::Rgba8, color_samples);
            const int depth_limit = next_supported_samples(mgl::InternalFormat::Depth24Stencil8, depth_samples);

            const mgl::TextureId ms = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, ms);
            (void)demo::drain_gl_errors();
            mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2DMultisample, color_samples, mgl::SizedInternalFormat::Rgba8, 16, 8, GL_TRUE);
            check.expect_no_gl_error(std::format("glTexStorage2DMultisample RGBA8 16 x 8, {} samples is accepted", color_samples));
            const Level rgba_level = query_level(mgl::TextureTarget::Texture2DMultisample, 0);
            const MsExpect rgba_ms{color_samples, color_limit, true};
            const std::string rgba = verify(rgba_level, {16, 8, kAny}, format_case("RGBA8"), false, rgba_ms);
            check.expect(rgba.empty(),
                         std::format("TEXTURE_2D_MULTISAMPLE RGBA8 16 x 8, {} samples, fixed sample locations TRUE: level 0 reports WIDTH, HEIGHT, INTERNAL_FORMAT, resolutions, types, "
                                     "TEXTURE_SAMPLES in [{}, {}] (not fewer than requested, not more than the next larger supported count) and TEXTURE_FIXED_SAMPLE_LOCATIONS = TRUE (ES 3.2 8.8, 8.11.3)",
                                     color_samples, color_samples, color_limit),
                         rgba.empty() ? std::format("TEXTURE_SAMPLES = {}", rgba_level.samples) : rgba);

            const mgl::TextureId msds = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, msds);
            mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2DMultisample, depth_samples, mgl::SizedInternalFormat::Depth24Stencil8, 8, 8, GL_FALSE);
            check.expect_no_gl_error(std::format("glTexStorage2DMultisample DEPTH24_STENCIL8 8 x 8, {} samples (fixed sample locations FALSE) is accepted", depth_samples));
            const Level ds_level = query_level(mgl::TextureTarget::Texture2DMultisample, 0);
            const std::string ds = verify(ds_level, {8, 8, kAny}, format_case("DEPTH24_STENCIL8"), false, MsExpect{depth_samples, depth_limit, false});
            check.expect(ds.empty(),
                         std::format("TEXTURE_2D_MULTISAMPLE DEPTH24_STENCIL8 8 x 8, {} samples, fixed sample locations FALSE: DEPTH_SIZE >= 24, STENCIL_SIZE >= 8, DEPTH_TYPE UNSIGNED_NORMALIZED, "
                                     "TEXTURE_SAMPLES in [{}, {}], TEXTURE_FIXED_SAMPLE_LOCATIONS = FALSE", depth_samples, depth_samples, depth_limit),
                         ds.empty() ? std::format("TEXTURE_SAMPLES = {}", ds_level.samples) : ds);

            // Negative control: the same level judged as a fixed-sample-location texture, or as a single-sample one, is rejected.
            const std::string wrong_fixed = verify(ds_level, {8, 8, kAny}, format_case("DEPTH24_STENCIL8"), false, MsExpect{depth_samples, depth_limit, true});
            const std::string as_single = verify(rgba_level, {16, 8, kAny}, format_case("RGBA8"));
            check.expect(wrong_fixed.contains("FIXED_SAMPLE_LOCATIONS") && as_single.contains("TEXTURE_SAMPLES"),
                         "negative control: the DEPTH24_STENCIL8 level judged as fixed-sample-location, and the multisample RGBA8 level judged as a non-multisample image (samples 0), are rejected",
                         wrong_fixed + " | " + as_single);

            // The same count seen through a framebuffer: SAMPLES of a framebuffer with the texture attached equals TEXTURE_SAMPLES (ES 3.2 9.2.3.1).
            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            query_fbos_.push_back(fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2DMultisample, ms, 0);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            const int samples = demo::get_integer(mgl::GetParameter::Samples);
            const int buffers = demo::get_integer(mgl::GetParameter::SampleBuffers);
            check.expect(status == mgl::FramebufferStatus::Complete && samples == rgba_level.samples && buffers == 1,
                         "the multisample texture attached to a framebuffer: complete, SAMPLE_BUFFERS = 1, and SAMPLES equals the texture's TEXTURE_SAMPLES (ES 3.2 9.2.3.1)",
                         std::format("status {}, SAMPLE_BUFFERS {}, SAMPLES {}, TEXTURE_SAMPLES {}", demo::framebuffer_status_name(status), buffers, samples, rgba_level.samples));
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2DMultisample, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void check_level_errors(demo::Check& check)
        {
            const mgl::TextureId t = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 2, mgl::SizedInternalFormat::Rgba8, 4, 4);
            GLint v = 0;
            (void)demo::drain_gl_errors();
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2D, -1, mgl::TextureLevelParameter::Width, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "level -1 -> INVALID_VALUE (ES 3.2 8.11.3)");
            // The largest valid level of a 2D texture is log2(MAX_TEXTURE_SIZE).
            const int max_size = demo::get_integer(mgl::GetParameter::MaxTextureSize);
            int max_level = 0;
            while ((2 << max_level) <= max_size) ++max_level;
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2D, max_level, mgl::TextureLevelParameter::Width, &v);
            check.expect_no_gl_error(std::format("level log2(MAX_TEXTURE_SIZE) = {} is a valid level (and was never specified)", max_level));
            check.expect_eq(static_cast<int>(v), 0, "...it reports WIDTH 0");
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2D, max_level + 1, mgl::TextureLevelParameter::Width, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, std::format("level log2(MAX_TEXTURE_SIZE) + 1 = {} -> INVALID_VALUE", max_level + 1));
            // TEXTURE_CUBE_MAP is not a valid target: it does not name a face.
            const mgl::TextureId cube = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, 4, 4);
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::TextureCubeMap, 0, mgl::TextureLevelParameter::Width, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "target TEXTURE_CUBE_MAP (not a face) -> INVALID_ENUM (ES 3.2 8.11.3)");
            // Multisample textures have no mipmaps: the maximum level is zero.
            const mgl::TextureId ms = new_texture();
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisample, ms);
            mgl::glTexStorage2DMultisample(mgl::TextureTarget::Texture2DMultisample, 1, mgl::SizedInternalFormat::Rgba8, 4, 4, GL_TRUE);
            (void)demo::drain_gl_errors();
            mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2DMultisample, 1, mgl::TextureLevelParameter::Width, &v);
            const mgl::ErrorCode ms_error = demo::drain_gl_errors();
            check.expect_or_known_deviation(ms_error == mgl::ErrorCode::InvalidValue, "TEXTURE_2D_MULTISAMPLE level 1 -> INVALID_VALUE (the maximum level is zero)",
                                            std::format("glGetError() = {}, expected GL_INVALID_VALUE", demo::error_name(ms_error)), "ANGLE",
                                            "ANGLE accepts level 1 of a TEXTURE_2D_MULTISAMPLE texture without an error; ES 3.2 8.11.3: \"For targets "
                                            "TEXTURE_BUFFER, TEXTURE_2D_MULTISAMPLE, and TEXTURE_2D_MULTISAMPLE_ARRAY, which do not support mipmaps, "
                                            "the maximum value is zero\"");
        }

        // ------------------------------------------------------------------------------------------
        mgl::ProgramId scene_prog_{}, show_depth_prog_{}, show_stencil_prog_{}, dump_float_prog_{}, dump_uint_prog_{};
        mgl::UniformLocation u_rect_{}, u_size_{}, u_depth_{};
        mgl::BufferId dump_buffer_{};
        mgl::FramebufferId fbo_{};
        mgl::TextureId d24s8_{}, d32fs8_{}, control_{}, rgba_fixture_{}, depth_fixture_{};
        std::vector<mgl::TextureId> textures_;
        std::vector<mgl::FramebufferId> query_fbos_;
        std::vector<Draw> draws_;
        std::vector<Pixel> expected_;
        std::vector<std::uint8_t> rgba_bytes_;
        int max_color_samples_ = 1;
        int max_depth_samples_ = 1;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es31-stencil-texturing-and-level-queries",
        .title = "Stencil texturing and texture level queries",
        .min_es = {3, 1},
        .summary = "Depth/stencil textures sampled as depth (sampler2D) and as stencil (usampler2D, DEPTH_STENCIL_TEXTURE_MODE) after rendering through an FBO, "
                   "read back exactly in a compute shader; glGetTexLevelParameteriv/fv compared with what was allocated.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<StencilTexturing>(argc, argv, info);
}
