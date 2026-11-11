// es32-multisample-array-and-border-color
//
// Two texture features that OpenGL ES 3.2 added to the core:
//
//  1. 2D multisample array textures (ES 3.2 section 8.8): glTexStorage3DMultisample(TEXTURE_2D_MULTISAMPLE_ARRAY, ...), rendered into
//     layer by layer (glFramebufferTextureLayer) and all layers at once (glFramebufferTexture + a geometry shader writing gl_Layer),
//     read back with texelFetch(gsampler2DMSArray, ivec3(x, y, layer), sample) for every texel, layer and sample.
//     Every sample is given its own value by a fragment shader that reads gl_SampleID (per-sample shading, section 14.2.2) over a
//     triangle that covers the whole viewport, so nothing depends on the implementation-defined sample positions.
//
//  2. The border color (sections 8.10, 8.14.2, 8.2, 8.3): CLAMP_TO_BORDER and TEXTURE_BORDER_COLOR set with glTexParameterfv / iv / Iiv /
//     Iuiv and glSamplerParameterfv / Iiv / Iuiv, read back with glGetTexParameterfv / Iiv / Iuiv and glGetSamplerParameterfv / Iiv / Iuiv,
//     and looked up outside [0, 1] for normalized (RGBA8, RG8), float (RGBA32F), integer (RGBA8I, RGBA16UI, RGBA32UI) and depth
//     (DEPTH_COMPONENT16, with and without depth comparison) textures, including the clamping of the border to the format's range,
//     the RG -> (R, G, 0, 1) rule, sampler objects overriding the texture's border, and LINEAR filtering across the border.
//
// The picture (laid out from the surface size, see Layout; at least 128 x 128):
//   top half      the RGBA8 multisample array, unresolved: one tile per (layer, sample), layer 0 at the top, sample 0 at the left;
//   bottom half   four square quads (a row, or 2 x 2 on tall surfaces) with texture coordinates -0.5 .. 1.5 (the 4 x 4 texture in the
//                 middle, its border all around):
//                 RGBA8 with the texture's border | the same texture with a sampler object's border | RGBA8I with an integer border |
//                 DEPTH_COMPONENT16 with depth comparison (white = pass), border depth 0.875 > reference 0.5.
//
// Every value is compared with a CPU model; the picture itself is checked pixel by pixel at any surface size.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int kW = 8;          // multisample array textures: kW x kH texels,
    constexpr int kH = 8;
    constexpr int kLayers = 4;     // kLayers layers (shaders/layers.geom emits exactly 4 layers)
    constexpr int kMinSurface = 128;  // smallest width / height with every texel of every tile and quad cell visible (see Layout)
    constexpr int kTex = 4;           // border textures are 4 x 4
    constexpr int kMaxShownSamples = 8;
    constexpr float kShowRef = 0.5f;  // depth reference of the shadow quad

    constexpr demo::Rgba8 kClear = {20, 20, 28, 255};  // exactly representable: k / 255 converts back to k

    enum class Kind
    {
        Float = 0,
        Int = 1,
        Uint = 2,
        Shadow = 3
    };

    using U4 = std::array<std::uint32_t, 4>;
    using I4 = std::array<std::int32_t, 4>;
    using F4 = std::array<float, 4>;

    // ---------------------------------------------------------------------------------------------
    // Multisample content: the same arithmetic as shaders/pattern.frag (32-bit unsigned, wrapping).
    // ---------------------------------------------------------------------------------------------
    U4 pattern(std::uint32_t x, std::uint32_t y, std::uint32_t layer, std::uint32_t s)
    {
        return {(29u * x + 61u * layer + 17u * s + 11u) & 255u, (37u * y + 53u * s + 23u * layer + 5u) & 255u,
                (67u * layer + 41u * s + 5u * x + 3u * y) & 255u, (250u - 23u * s - 13u * layer) & 255u};
    }

    // ---------------------------------------------------------------------------------------------
    // Border-color textures (4 x 4, NEAREST, CLAMP_TO_BORDER), their texels and border values.
    // Texel (i, j) is column i (s = (i + 0.5) / 4) of row j (t = (j + 0.5) / 4); row 0 is uploaded first.
    // ---------------------------------------------------------------------------------------------
    std::array<std::uint8_t, 4> texel_rgba8(int i, int j)
    {
        return {static_cast<std::uint8_t>(40 + 60 * i), static_cast<std::uint8_t>(30 + 60 * j), static_cast<std::uint8_t>(200 - 40 * i - 20 * j),
                static_cast<std::uint8_t>(255 - 30 * j)};
    }
    std::array<std::uint8_t, 2> texel_rg8(int i, int j) { return {static_cast<std::uint8_t>(10 + 70 * i), static_cast<std::uint8_t>(250 - 60 * j)}; }
    I4 texel_rgba8i(int i, int j) { return {20 * i - 60, 25 * j - 40, 50 - 10 * (i + j), 5 * i * j - 100}; }
    F4 texel_rgba32f(int i, int j)
    {
        return {static_cast<float>(i) + 0.25f, static_cast<float>(j) - 0.5f, -static_cast<float>(i * j), 2.0f + static_cast<float>(i)};
    }
    U4 texel_rgba16ui(int i, int j)
    {
        return {1000u * static_cast<std::uint32_t>(i) + 1u, 2000u * static_cast<std::uint32_t>(j) + 2u, 3u + static_cast<std::uint32_t>(i + j), 4u};
    }
    U4 texel_rgba32ui(int i, int j)
    {
        return {0x10000000u * static_cast<std::uint32_t>(i) + static_cast<std::uint32_t>(j), 0xF0000000u + static_cast<std::uint32_t>(j), 7u, 0x12345678u};
    }
    std::uint16_t texel_depth16(int i, int j) { return static_cast<std::uint16_t>((i + 4 * j) * 4096 + 2048); }
    double depth_value(int i, int j) { return static_cast<double>(texel_depth16(i, j)) / 65535.0; }

    constexpr F4 kBorderRgba8 = {1.5f, -0.25f, 0.2f, 0.6f};     // glTexParameterfv; 0.2 and 0.6 are 51/255 and 153/255
    constexpr F4 kBorderSampler = {0.0f, 1.0f, 0.4f, 1.0f};     // glSamplerParameterfv (0.4 = 102/255)
    constexpr F4 kBorderLinear = {0.0f, 0.2f, 1.0f, 1.0f};      // the LINEAR sampler's border
    constexpr I4 kBorderRgba8I = {300, 40, -200, 5};             // glTexParameterIiv; clamped to [-128, 127] when used
    constexpr I4 kBorderSamplerI = {-1, 2, -300, 4};             // glSamplerParameterIiv
    constexpr U4 kBorderRgba16UI = {70000u, 3u, 0xFFFFFFFFu, 65535u};  // glTexParameterIuiv; clamped to [0, 65535] when used
    constexpr U4 kBorderRgba32UI = {0xFFFFFFFFu, 0x80000000u, 7u, 123456789u};
    constexpr U4 kBorderSamplerUI = {1u, 2u, 3u, 70000u};        // glSamplerParameterIuiv
    constexpr F4 kBorderRgba32F = {-2.5f, 3.0f, 1000.0f, 0.125f};
    constexpr float kBorderDepth = 0.875f;

    struct Probe
    {
        float s, t, ref, pad;
    };

    /// Texel selection of NEAREST filtering (ES 3.2 section 8.14.2, Table 8.21) for a 4 x 4 texture.
    struct Selected
    {
        bool border = false;
        int i = 0, j = 0;
    };

    Selected select_nearest(const Probe& p, bool clamp_to_border)
    {
        int i = static_cast<int>(std::floor(static_cast<double>(p.s) * kTex));
        int j = static_cast<int>(std::floor(static_cast<double>(p.t) * kTex));
        if (clamp_to_border)
        {
            i = std::clamp(i, -1, kTex);  // CLAMP_TO_BORDER: clamp(coord, -1, size)
            j = std::clamp(j, -1, kTex);
            return {i < 0 || i >= kTex || j < 0 || j >= kTex, i, j};
        }
        return {false, std::clamp(i, 0, kTex - 1), std::clamp(j, 0, kTex - 1)};  // CLAMP_TO_EDGE: clamp(coord, 0, size - 1)
    }

    /// The picture's layout, derived from the surface size (shared by draw_picture() and check_picture()).
    ///   top half    (rows H/2 .. H-1): the multisample grid, `shown` columns (samples) x kLayers rows (layer 0 at the top),
    ///               tiles of tile_w x tile_h pixels with a `gap` of clear color on the right and at the bottom of each tile;
    ///   bottom half (rows 0 .. H/2-1): four square border quads, 4 x 1 or 2 x 2 cells (whichever gives larger squares),
    ///               each quad 7/8 of its cell's shorter side rounded down to a multiple of 8, centred in the cell.
    struct Layout
    {
        int width = 0, height = 0;
        int grid_y = 0, grid_h = 0;
        int shown = 1, tile_w = 0, tile_h = 0, gap = 1;
        struct Quad
        {
            int x = 0, y = 0, size = 0;
        };
        std::array<Quad, 4> quads{};

        Layout(int w, int h, int samples) : width(w), height(h)
        {
            grid_y = h / 2;
            grid_h = h - grid_y;
            shown = std::clamp(samples, 1, kMaxShownSamples);
            tile_w = w / shown;
            tile_h = grid_h / kLayers;
            gap = std::max(1, std::min(tile_w, tile_h) / 24);
            const int bottom = grid_y;
            const int row_size = std::min(w / 4, bottom);
            const int square_size = std::min(w / 2, bottom / 2);
            const int cols = row_size >= square_size ? 4 : 2;
            const int rows = 4 / cols;
            const int cell_w = w / cols, cell_h = bottom / rows;
            // A multiple of 8 puts every texel boundary (8 cells of size / 8 pixels across the quad) on a pixel edge, never on a pixel centre.
            const int size = std::min(cell_w, cell_h) * 7 / 8 / 8 * 8;
            for (int q = 0; q < 4; ++q)
            {
                const int c = q % cols, r = rows - 1 - q / cols;  // first row at the top
                quads[static_cast<std::size_t>(q)] = {c * cell_w + (cell_w - size) / 2, r * cell_h + (cell_h - size) / 2, size};
            }
        }
    };

    /// The probes: the 16 texel centres and 14 points outside [0, 1] in s, t or both, none of them near a texel boundary.
    std::vector<Probe> border_probes(std::initializer_list<float> refs = {0.0f})
    {
        std::vector<std::pair<float, float>> st;
        for (int j = 0; j < kTex; ++j)
            for (int i = 0; i < kTex; ++i)
                st.emplace_back((static_cast<float>(i) + 0.5f) / kTex, (static_cast<float>(j) + 0.5f) / kTex);
        const std::array<std::pair<float, float>, 14> outside = {{{-0.375f, 0.375f}, {-0.125f, 0.625f}, {1.125f, 0.125f}, {1.625f, 0.875f}, {0.375f, -0.125f},
                                                                  {0.625f, -0.875f}, {0.125f, 1.125f}, {0.875f, 1.375f}, {-0.125f, -0.125f}, {1.125f, 1.125f},
                                                                  {-0.375f, 1.375f}, {1.375f, -0.375f}, {-3.0f, 0.375f}, {0.625f, 7.25f}}};
        for (const auto& o : outside) st.push_back(o);
        std::vector<Probe> probes;
        for (const float r : refs)
            for (const auto& [s, t] : st) probes.push_back({s, t, r, 0.0f});
        return probes;
    }

    double unorm8(int v) { return static_cast<double>(v) / 255.0; }
    float unorm8f(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

    /// Collects mismatches; `first` describes the first one.
    struct Tally
    {
        int bad = 0;
        int total = 0;
        double max_err = 0.0;
        std::string first;
        void add(bool ok, const std::string& what)
        {
            ++total;
            if (!ok && bad++ == 0) first = what;
        }
        [[nodiscard]] std::string detail() const
        {
            return bad == 0 ? std::format("{} values, max |error| {:.3g}", total, max_err) : std::format("{} of {} wrong; first: {}", bad, total, first);
        }
    };

    void t_add_rgb(Tally& t, const demo::Rgba8& got, const demo::Rgba8& want, int tolerance, int x, int y)
    {
        bool ok = true;
        for (std::size_t c = 0; c < 4; ++c)
        {
            const int err = std::abs(static_cast<int>(got[c]) - static_cast<int>(want[c]));
            t.max_err = std::max(t.max_err, static_cast<double>(err));
            ok = ok && err <= tolerance;
        }
        t.add(ok, std::format("pixel ({}, {}): got {} expected {}", x, y, demo::format_rgba(got), demo::format_rgba(want)));
    }

    std::string f4s(const F4& v) { return std::format("({}, {}, {}, {})", v[0], v[1], v[2], v[3]); }
    template <typename T>
    std::string v4s(const std::array<T, 4>& v)
    {
        return std::format("({}, {}, {}, {})", v[0], v[1], v[2], v[3]);
    }

    F4 as_float(const U4& u) { return {std::bit_cast<float>(u[0]), std::bit_cast<float>(u[1]), std::bit_cast<float>(u[2]), std::bit_cast<float>(u[3])}; }
    I4 as_int(const U4& u)
    {
        return {std::bit_cast<std::int32_t>(u[0]), std::bit_cast<std::int32_t>(u[1]), std::bit_cast<std::int32_t>(u[2]), std::bit_cast<std::int32_t>(u[3])};
    }

    std::string with_defines(std::string source, std::initializer_list<std::pair<std::string_view, std::string_view>> defines)
    {
        std::string inserted;
        for (const auto& [name, value] : defines) inserted += std::format("#define {} {}\n", name, value);
        source.insert(source.find('\n') + 1, inserted);  // right after "#version 320 es"
        return source;
    }

    mgl::ProgramId build(std::initializer_list<std::pair<mgl::ShaderType, std::string>> stages, std::string_view label)
    {
        std::vector<mgl::ShaderId> shaders;
        std::string log;
        for (const auto& [type, source] : stages)
        {
            const mgl::ShaderId s = demo::compile_shader(type, source, log);
            if (s.value == 0)
            {
                for (const mgl::ShaderId d : shaders) mgl::glDeleteShader(d);
                throw demo::Failure(std::format("{}: a shader failed to compile:\n{}", label, log));
            }
            shaders.push_back(s);
        }
        const mgl::ProgramId program = demo::link_program(shaders, log);
        for (const mgl::ShaderId d : shaders) mgl::glDeleteShader(d);
        if (program.value == 0) throw demo::Failure(std::format("{}: link failed:\n{}", label, log));
        return program;
    }

    /// One 2D multisample array texture and a framebuffer for rendering into it.
    struct MsArray
    {
        mgl::TextureId tex{};
        mgl::FramebufferId fbo{};
        Kind kind = Kind::Float;
        mgl::SizedInternalFormat format = mgl::SizedInternalFormat::Rgba8;
        const char* name = "";
        int requested = 0;
        int samples = 0;  // TEXTURE_SAMPLES
    };

    struct FormatCounts
    {
        mgl::InternalFormat format;
        const char* name;
        bool integer;
        std::vector<GLint> counts;  // GetInternalformativ(TEXTURE_2D_MULTISAMPLE_ARRAY, SAMPLES): descending
    };

    class MultisampleArrayAndBorderColor final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override;
        void render(demo::Env& env, int frame) override;
        void validate(demo::Env& env, demo::Check& check) override;
        void shutdown(demo::Env& env) override;

    private:
        // objects
        void build_programs(const demo::Env& env);
        MsArray make_ms(Kind kind, mgl::SizedInternalFormat format, const char* name, int requested, bool fixed, int w, int h, int layers);
        static mgl::TextureId make_texture_2d(mgl::SizedInternalFormat format, mgl::PixelFormat pf, mgl::PixelType type, const void* data);
        static mgl::SamplerId make_sampler(mgl::TextureWrapMode wrap, bool linear);
        static std::vector<GLint> sample_counts(mgl::InternalFormat format);
        static int pick(const std::vector<GLint>& counts);
        static int next_count(const std::vector<GLint>& counts, int requested);
        static int level_param(mgl::TextureId tex, mgl::TextureLevelParameter pname);

        // passes
        void fill(const MsArray& m);
        void draw_picture(int width, int height);
        struct Fetched
        {
            std::array<int, 3> size{};
            std::vector<U4> texels;
        };
        Fetched fetch(const MsArray& m, int w, int h, int layers);
        std::vector<U4> lookup(Kind kind, mgl::TextureId tex, mgl::SamplerId sampler, const std::vector<Probe>& probes);

        // checks
        void check_ms_counts(demo::Check& check);
        void check_ms_queries(demo::Check& check);
        void check_ms_contents(demo::Check& check);
        void check_ms_layered_clear(demo::Check& check);
        void check_ms_errors(demo::Check& check);
        void check_border_queries(demo::Check& check);
        void check_border_normalized(demo::Check& check);
        void check_border_float(demo::Check& check);
        void check_border_integer(demo::Check& check);
        void check_border_depth(demo::Check& check);
        void check_sampler_override(demo::Check& check);
        void check_border_linear(demo::Check& check);
        void check_border_errors(demo::Check& check);
        void check_picture(demo::Check& check, int width, int height);

        void check_float_lookups(demo::Check& check, mgl::TextureId tex, mgl::SamplerId sampler, bool clamp_to_border, double tolerance,
                                 F4 (*texel)(int, int), F4 border, std::string_view what);

        // state
        mgl::VertexArrayId vao_{};
        mgl::ProgramId fill_layered_{};                 // VS + GS + FS (RGBA8, layered)
        std::array<mgl::ProgramId, 3> fill_layer_{};    // VS + FS per kind (Float unused)
        std::array<mgl::ProgramId, 3> fetch_{};         // compute, per kind
        std::array<mgl::ProgramId, 4> border_{};        // compute, per kind (incl. Shadow)
        mgl::ProgramId show_ms_{};
        std::array<mgl::ProgramId, 4> show_border_{};   // Float, Int, (unused), Shadow

        std::vector<FormatCounts> formats_;
        MsArray ms_float_, ms_int_, ms_uint_, ms_free_;
        int max_samples_ = 0, max_color_samples_ = 0, max_integer_samples_ = 0;

        mgl::TextureId tex_rgba8_{}, tex_rgba8i_{}, tex_shadow_{};
        mgl::SamplerId sampler_override_{};
        F4 initial_texture_border_{-1.0f, -1.0f, -1.0f, -1.0f};
        F4 initial_sampler_border_{-1.0f, -1.0f, -1.0f, -1.0f};

        mgl::BufferId probe_buf_{}, result_buf_{};
    };

    // =================================================================================================
    // Objects
    // =================================================================================================
    std::vector<GLint> MultisampleArrayAndBorderColor::sample_counts(mgl::InternalFormat format)
    {
        GLint n = 0;
        mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisampleArray, format, mgl::InternalFormatParameter::NumSampleCounts, 1, &n);
        std::vector<GLint> counts(static_cast<std::size_t>(std::max(n, 0)));
        if (n > 0)
            mgl::glGetInternalformativ(mgl::InternalFormatTarget::Texture2DMultisampleArray, format, mgl::InternalFormatParameter::Samples, n, counts.data());
        return counts;
    }

    /// The smallest reported count that is at least 4, or the largest one if none is.
    int MultisampleArrayAndBorderColor::pick(const std::vector<GLint>& counts)
    {
        int chosen = counts.empty() ? 1 : counts.front();
        for (const GLint c : counts)
            if (c >= 4) chosen = c;  // descending order: the last one >= 4 is the smallest
        return chosen;
    }

    /// "the next larger sample count supported by the implementation" (section 8.8): the smallest reported count >= requested.
    int MultisampleArrayAndBorderColor::next_count(const std::vector<GLint>& counts, int requested)
    {
        int best = -1;
        for (const GLint c : counts)
            if (c >= requested && (best < 0 || c < best)) best = c;
        return best;
    }

    int MultisampleArrayAndBorderColor::level_param(mgl::TextureId tex, mgl::TextureLevelParameter pname)
    {
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, tex);
        GLint v = -12345;
        mgl::glGetTexLevelParameteriv(mgl::TextureTarget::Texture2DMultisampleArray, 0, pname, &v);
        return v;
    }

    MsArray MultisampleArrayAndBorderColor::make_ms(Kind kind, mgl::SizedInternalFormat format, const char* name, int requested, bool fixed, int w, int h,
                                                    int layers)
    {
        MsArray m;
        m.kind = kind;
        m.format = format;
        m.name = name;
        m.requested = requested;
        mgl::glGenTextures(1, &m.tex);
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, m.tex);
        mgl::glTexStorage3DMultisample(mgl::TextureTarget::Texture2DMultisampleArray, requested, format, w, h, layers, fixed ? GL_TRUE : GL_FALSE);
        m.samples = level_param(m.tex, mgl::TextureLevelParameter::Samples);
        mgl::glGenFramebuffers(1, &m.fbo);
        return m;
    }

    mgl::TextureId MultisampleArrayAndBorderColor::make_texture_2d(mgl::SizedInternalFormat format, mgl::PixelFormat pf, mgl::PixelType type, const void* data)
    {
        mgl::TextureId t{};
        mgl::glGenTextures(1, &t);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, kTex, kTex);
        mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kTex, kTex, pf, type, data);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToBorder);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToBorder);
        return t;
    }

    mgl::SamplerId MultisampleArrayAndBorderColor::make_sampler(mgl::TextureWrapMode wrap, bool linear)
    {
        mgl::SamplerId s{};
        mgl::glGenSamplers(1, &s);
        mgl::glSamplerParameter(s, linear ? mgl::TextureMinFilter::Linear : mgl::TextureMinFilter::Nearest);
        mgl::glSamplerParameter(s, linear ? mgl::TextureMagFilter::Linear : mgl::TextureMagFilter::Nearest);
        mgl::glSamplerParameter(s, mgl::SamplerWrapParameter::WrapS, wrap);
        mgl::glSamplerParameter(s, mgl::SamplerWrapParameter::WrapT, wrap);
        return s;
    }

    void MultisampleArrayAndBorderColor::build_programs(const demo::Env& env)
    {
        const std::string vs_full = env.load_text("shaders/fullscreen.vert");
        const std::string gs = env.load_text("shaders/layers.geom");
        const std::string fs_pattern = env.load_text("shaders/pattern.frag");
        const std::string cs_fetch = env.load_text("shaders/fetch.comp");
        const std::string cs_border = env.load_text("shaders/border.comp");
        const std::string vs_quad = env.load_text("shaders/quad.vert");
        const std::string fs_show_ms = env.load_text("shaders/show_ms.frag");
        const std::string fs_show_border = env.load_text("shaders/show_border.frag");

        fill_layered_ = build({{mgl::ShaderType::Vertex, vs_full},
                               {mgl::ShaderType::Geometry, gs},
                               {mgl::ShaderType::Fragment, with_defines(fs_pattern, {{"KIND_FLOAT", "1"}, {"FROM_GS", "1"}})}},
                              "fill (layered)");
        fill_layer_[1] = build({{mgl::ShaderType::Vertex, vs_full}, {mgl::ShaderType::Fragment, with_defines(fs_pattern, {{"KIND_INT", "1"}})}}, "fill (int)");
        fill_layer_[2] = build({{mgl::ShaderType::Vertex, vs_full}, {mgl::ShaderType::Fragment, with_defines(fs_pattern, {{"KIND_UINT", "1"}})}}, "fill (uint)");

        fetch_[0] = build({{mgl::ShaderType::Compute, with_defines(cs_fetch, {{"SAMPLER", "sampler2DMSArray"}, {"KIND_FLOAT", "1"}})}}, "fetch (float)");
        fetch_[1] = build({{mgl::ShaderType::Compute, with_defines(cs_fetch, {{"SAMPLER", "isampler2DMSArray"}, {"KIND_INT", "1"}})}}, "fetch (int)");
        fetch_[2] = build({{mgl::ShaderType::Compute, with_defines(cs_fetch, {{"SAMPLER", "usampler2DMSArray"}, {"KIND_UINT", "1"}})}}, "fetch (uint)");

        border_[0] = build({{mgl::ShaderType::Compute, with_defines(cs_border, {{"SAMPLER", "sampler2D"}, {"KIND_FLOAT", "1"}})}}, "border (float)");
        border_[1] = build({{mgl::ShaderType::Compute, with_defines(cs_border, {{"SAMPLER", "isampler2D"}, {"KIND_INT", "1"}})}}, "border (int)");
        border_[2] = build({{mgl::ShaderType::Compute, with_defines(cs_border, {{"SAMPLER", "usampler2D"}, {"KIND_UINT", "1"}})}}, "border (uint)");
        border_[3] = build({{mgl::ShaderType::Compute, with_defines(cs_border, {{"SAMPLER", "sampler2DShadow"}, {"KIND_SHADOW", "1"}})}}, "border (shadow)");

        show_ms_ = build({{mgl::ShaderType::Vertex, vs_quad}, {mgl::ShaderType::Fragment, fs_show_ms}}, "show multisample array");
        show_border_[0] = build({{mgl::ShaderType::Vertex, vs_quad}, {mgl::ShaderType::Fragment, with_defines(fs_show_border, {{"SAMPLER", "sampler2D"}, {"KIND_FLOAT", "1"}})}},
                                "show border (float)");
        show_border_[1] = build({{mgl::ShaderType::Vertex, vs_quad}, {mgl::ShaderType::Fragment, with_defines(fs_show_border, {{"SAMPLER", "isampler2D"}, {"KIND_INT", "1"}})}},
                                "show border (int)");
        show_border_[3] = build(
            {{mgl::ShaderType::Vertex, vs_quad}, {mgl::ShaderType::Fragment, with_defines(fs_show_border, {{"SAMPLER", "sampler2DShadow"}, {"KIND_SHADOW", "1"}})}},
            "show border (shadow)");
    }

    void MultisampleArrayAndBorderColor::init(demo::Env& env)
    {
        // Below 128 x 128 a tile of the grid (8 samples) or a quad cell gets too few pixels to show every texel and border cell.
        if (env.width < kMinSurface || env.height < kMinSurface)
            throw demo::Skip(std::format("the picture needs a surface of at least {0}x{0} pixels to show every texel (got {1}x{2})", kMinSurface, env.width, env.height));

        mgl::glGenVertexArrays(1, &vao_);
        mgl::glBindVertexArray(vao_);
        mgl::glDisable(mgl::Capability::Dither);  // 8-bit results are compared exactly
        mgl::glGenBuffers(1, &probe_buf_);
        mgl::glGenBuffers(1, &result_buf_);
        build_programs(env);

        // ---- limits and sample counts (section 20.3) -------------------------------------------------
        max_samples_ = demo::get_integer(mgl::GetParameter::MaxSamples);
        max_color_samples_ = demo::get_integer(mgl::GetParameter::MaxColorTextureSamples);
        max_integer_samples_ = demo::get_integer(mgl::GetParameter::MaxIntegerSamples);
        formats_ = {{mgl::InternalFormat::Rgba8, "RGBA8", false, {}}, {mgl::InternalFormat::Rgba8I, "RGBA8I", true, {}}, {mgl::InternalFormat::Rgba8UI, "RGBA8UI", true, {}}};
        for (FormatCounts& f : formats_) f.counts = sample_counts(f.format);
        for (const FormatCounts& f : formats_)
            if (f.counts.empty())  // section 20.3.1 guarantees at least one count for these color-renderable formats; validate() reports it
                std::cout << std::format("note: GetInternalformativ(TEXTURE_2D_MULTISAMPLE_ARRAY, {}, NUM_SAMPLE_COUNTS) returned 0\n", f.name);
        std::cout << std::format("MAX_SAMPLES {}, MAX_COLOR_TEXTURE_SAMPLES {}, MAX_INTEGER_SAMPLES {}\n", max_samples_, max_color_samples_, max_integer_samples_);
        for (const FormatCounts& f : formats_)
        {
            std::string list;
            for (const GLint c : f.counts) list += std::format("{} ", c);
            std::cout << std::format("{} sample counts for TEXTURE_2D_MULTISAMPLE_ARRAY: {}(using {})\n", f.name, list, pick(f.counts));
        }

        // ---- the multisample arrays ---------------------------------------------------------------
        ms_float_ = make_ms(Kind::Float, mgl::SizedInternalFormat::Rgba8, "RGBA8", pick(formats_[0].counts), true, kW, kH, kLayers);
        ms_int_ = make_ms(Kind::Int, mgl::SizedInternalFormat::Rgba8I, "RGBA8I", pick(formats_[1].counts), true, kW, kH, kLayers);
        ms_uint_ = make_ms(Kind::Uint, mgl::SizedInternalFormat::Rgba8UI, "RGBA8UI", pick(formats_[2].counts), true, kW, kH, kLayers);
        ms_free_ = make_ms(Kind::Float, mgl::SizedInternalFormat::Rgba8, "RGBA8 (fixedsamplelocations FALSE)", pick(formats_[0].counts), false, 4, 4, 2);
        if (mgl::glGetError() != mgl::ErrorCode::NoError) throw demo::Failure("glTexStorage3DMultisample of the demo's textures failed");

        // RGBA8: the whole array attached (a layered attachment, section 9.2.8); the geometry shader picks the layer.
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, ms_float_.fbo);
        mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), ms_float_.tex, 0);
        // RGBA8I / RGBA8UI: one layer at a time (glFramebufferTextureLayer in fill()); layer 0 for now.
        for (const MsArray* m : {&ms_int_, &ms_uint_})
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m->fbo);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), m->tex, 0, 0);
        }
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

        // ---- border-color textures used by the picture -------------------------------------------
        std::vector<std::uint8_t> rgba8;
        std::vector<std::int8_t> rgba8i;
        std::vector<std::uint16_t> depth;
        for (int j = 0; j < kTex; ++j)
            for (int i = 0; i < kTex; ++i)
            {
                for (const std::uint8_t c : texel_rgba8(i, j)) rgba8.push_back(c);
                for (const std::int32_t c : texel_rgba8i(i, j)) rgba8i.push_back(static_cast<std::int8_t>(c));
                depth.push_back(texel_depth16(i, j));
            }
        tex_rgba8_ = make_texture_2d(mgl::SizedInternalFormat::Rgba8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, rgba8.data());
        mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::BorderColor, initial_texture_border_.data());  // checked in validate()
        mgl::glTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, kBorderRgba8.data());

        tex_rgba8i_ = make_texture_2d(mgl::SizedInternalFormat::Rgba8I, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Byte, rgba8i.data());
        mgl::glTexParameterIiv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, kBorderRgba8I.data());

        tex_shadow_ = make_texture_2d(mgl::SizedInternalFormat::DepthComponent16, mgl::PixelFormat::DepthComponent, mgl::PixelType::UnsignedShort, depth.data());
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::CompareFunc::Lequal);
        const F4 depth_border = {kBorderDepth, 0.0f, 0.0f, 0.0f};  // the first component is the depth value (section 8.14.2)
        mgl::glTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, depth_border.data());
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

        sampler_override_ = make_sampler(mgl::TextureWrapMode::ClampToBorder, false);
        mgl::glGetSamplerParameterfv(sampler_override_, mgl::SamplerParameter::BorderColor, initial_sampler_border_.data());  // checked in validate()
        mgl::glSamplerParameterfv(sampler_override_, mgl::SamplerParameter::BorderColor, kBorderSampler.data());
    }

    void MultisampleArrayAndBorderColor::shutdown(demo::Env& /*env*/)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glUseProgram(mgl::ProgramId{0});
        mgl::glBindSampler(0, mgl::SamplerId{0});
        for (MsArray* m : {&ms_float_, &ms_int_, &ms_uint_, &ms_free_})
        {
            if (m->fbo.value != 0) mgl::glDeleteFramebuffers(1, &m->fbo);
            if (m->tex.value != 0) mgl::glDeleteTextures(1, &m->tex);
            *m = MsArray{};
        }
        for (mgl::TextureId* t : {&tex_rgba8_, &tex_rgba8i_, &tex_shadow_})
            if (t->value != 0) mgl::glDeleteTextures(1, t);
        if (sampler_override_.value != 0) mgl::glDeleteSamplers(1, &sampler_override_);
        std::vector<mgl::ProgramId> programs = {fill_layered_, show_ms_};
        for (const auto* group : {&fill_layer_, &fetch_}) programs.insert(programs.end(), group->begin(), group->end());
        for (const auto* group : {&border_, &show_border_}) programs.insert(programs.end(), group->begin(), group->end());
        for (const mgl::ProgramId p : programs)
            if (p.value != 0) mgl::glDeleteProgram(p);
        mgl::glDeleteBuffers(1, &probe_buf_);
        mgl::glDeleteBuffers(1, &result_buf_);
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glDeleteVertexArrays(1, &vao_);
    }

    // =================================================================================================
    // Rendering
    // =================================================================================================
    void MultisampleArrayAndBorderColor::fill(const MsArray& m)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m.fbo);
        mgl::glViewport(0, 0, kW, kH);
        if (m.kind == Kind::Float)
        {
            // Layered: one draw, the geometry shader emits the triangle into each of the 4 layers.
            mgl::glUseProgram(fill_layered_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            return;
        }
        const mgl::ProgramId program = fill_layer_[static_cast<std::size_t>(m.kind)];
        mgl::glUseProgram(program);
        const mgl::UniformLocation u_layer = mgl::glGetUniformLocation(program, "u_layer");
        for (int layer = 0; layer < kLayers; ++layer)
        {
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), m.tex, 0, layer);
            mgl::glUniform1i(u_layer, layer);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }
    }

    void MultisampleArrayAndBorderColor::draw_picture(int width, int height)
    {
        const Layout lay(width, height, ms_float_.samples);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glViewport(0, 0, width, height);
        mgl::glClearColor(unorm8f(kClear[0]), unorm8f(kClear[1]), unorm8f(kClear[2]), 1.0f);
        mgl::glClear(mgl::ClearBufferBit::Color);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);

        // Top half: the RGBA8 multisample array, one tile per (layer, sample).
        mgl::glViewport(0, lay.grid_y, width, lay.grid_h);
        mgl::glUseProgram(show_ms_);
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, ms_float_.tex);
        mgl::glUniform4f(mgl::glGetUniformLocation(show_ms_, "u_uv"), 0.0f, 0.0f, 1.0f, 1.0f);
        mgl::glUniform2i(mgl::glGetUniformLocation(show_ms_, "u_origin"), 0, lay.grid_y);
        mgl::glUniform2i(mgl::glGetUniformLocation(show_ms_, "u_tile"), lay.tile_w, lay.tile_h);
        mgl::glUniform1i(mgl::glGetUniformLocation(show_ms_, "u_gap"), lay.gap);
        mgl::glUniform1i(mgl::glGetUniformLocation(show_ms_, "u_columns"), lay.shown);
        mgl::glUniform2i(mgl::glGetUniformLocation(show_ms_, "u_texels"), kW, kH);
        mgl::glUniform1i(mgl::glGetUniformLocation(show_ms_, "u_layers"), kLayers);
        mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, mgl::TextureId{0});

        // Bottom half: four square border quads, texture coordinates -0.5 .. 1.5.
        struct Quad
        {
            mgl::ProgramId program;
            mgl::TextureId tex;
            mgl::SamplerId sampler;
        };
        const std::array<Quad, 4> quads = {{{show_border_[0], tex_rgba8_, mgl::SamplerId{0}},
                                            {show_border_[0], tex_rgba8_, sampler_override_},
                                            {show_border_[1], tex_rgba8i_, mgl::SamplerId{0}},
                                            {show_border_[3], tex_shadow_, mgl::SamplerId{0}}}};
        for (std::size_t q = 0; q < quads.size(); ++q)
        {
            const Layout::Quad& r = lay.quads[q];
            mgl::glViewport(r.x, r.y, r.size, r.size);
            mgl::glUseProgram(quads[q].program);
            mgl::glUniform4f(mgl::glGetUniformLocation(quads[q].program, "u_uv"), -0.5f, -0.5f, 1.5f, 1.5f);
            const mgl::UniformLocation ref = mgl::glGetUniformLocation(quads[q].program, "u_ref");
            if (ref.value >= 0) mgl::glUniform1f(ref, kShowRef);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, quads[q].tex);
            mgl::glBindSampler(0, quads[q].sampler);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }
        mgl::glBindSampler(0, mgl::SamplerId{0});
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        mgl::glViewport(0, 0, width, height);
    }

    void MultisampleArrayAndBorderColor::render(demo::Env& env, int /*frame*/)
    {
        for (const MsArray* m : {&ms_float_, &ms_int_, &ms_uint_}) fill(*m);
        draw_picture(env.width, env.height);
    }

    // =================================================================================================
    // Read-back passes (compute shaders writing to a shader storage buffer)
    // =================================================================================================
    template <typename T>
    std::vector<T> read_ssbo(mgl::BufferId buf, std::size_t count)
    {
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, buf);
        const std::size_t bytes = count * sizeof(T);
        const void* p = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, static_cast<GLsizeiptr>(bytes), mgl::MapBufferAccessMask::Read);
        if (p == nullptr) throw demo::Failure("glMapBufferRange failed");
        std::vector<T> out(count);
        std::memcpy(out.data(), p, bytes);
        if (mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage) != GL_TRUE) throw demo::Failure("glUnmapBuffer returned FALSE");
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
        return out;
    }

    MultisampleArrayAndBorderColor::Fetched MultisampleArrayAndBorderColor::fetch(const MsArray& m, int w, int h, int layers)
    {
        const int n = std::max(m.samples, 1);
        const std::size_t count = 1 + static_cast<std::size_t>(w * h * layers * n);  // the size record, then the texels
        const std::vector<U4> sentinel(count, U4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu});
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, result_buf_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(count * sizeof(U4)), sentinel.data(), mgl::BufferUsage::DynamicCopy);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, result_buf_);

        const mgl::ProgramId program = fetch_[static_cast<std::size_t>(m.kind)];
        mgl::glUseProgram(program);
        mgl::glUniform3i(mgl::glGetUniformLocation(program, "u_size"), w, h, layers);
        mgl::glUniform1i(mgl::glGetUniformLocation(program, "u_samples"), m.samples);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, m.tex);
        mgl::glDispatchCompute(static_cast<GLuint>((w + 3) / 4), static_cast<GLuint>((h + 3) / 4), static_cast<GLuint>(layers));
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, mgl::TextureId{0});

        const std::vector<U4> raw = read_ssbo<U4>(result_buf_, count);
        Fetched f;
        f.size = {std::bit_cast<std::int32_t>(raw[0][0]), std::bit_cast<std::int32_t>(raw[0][1]), std::bit_cast<std::int32_t>(raw[0][2])};
        f.texels.assign(raw.begin() + 1, raw.end());
        return f;
    }

    std::vector<U4> MultisampleArrayAndBorderColor::lookup(Kind kind, mgl::TextureId tex, mgl::SamplerId sampler, const std::vector<Probe>& probes)
    {
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, probe_buf_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(probes.size() * sizeof(Probe)), probes.data(), mgl::BufferUsage::DynamicDraw);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, probe_buf_);
        const std::vector<U4> sentinel(probes.size(), U4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu});
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, result_buf_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(sentinel.size() * sizeof(U4)), sentinel.data(), mgl::BufferUsage::DynamicCopy);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, result_buf_);

        const mgl::ProgramId program = border_[static_cast<std::size_t>(kind)];
        mgl::glUseProgram(program);
        mgl::glUniform1ui(mgl::glGetUniformLocation(program, "u_count"), static_cast<GLuint>(probes.size()));
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
        mgl::glBindSampler(0, sampler);
        mgl::glDispatchCompute(static_cast<GLuint>((probes.size() + 63) / 64), 1, 1);
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glBindSampler(0, mgl::SamplerId{0});
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        return read_ssbo<U4>(result_buf_, probes.size());
    }

    // =================================================================================================
    // Validation
    // =================================================================================================
    void MultisampleArrayAndBorderColor::validate(demo::Env& env, demo::Check& check)
    {
        check_picture(check, env.width, env.height);  // first, while the default framebuffer still holds the rendered frame
        check_ms_counts(check);
        check_ms_queries(check);
        check_ms_contents(check);
        check_ms_layered_clear(check);
        check_ms_errors(check);
        check_border_queries(check);
        check_border_normalized(check);
        check_border_float(check);
        check_border_integer(check);
        check_border_depth(check);
        check_sampler_override(check);
        check_border_linear(check);
        check_border_errors(check);

        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        check.expect_no_gl_error("no stray GL error left by the validation");
        render(env, 0);  // leave the picture in the framebuffer for the capture
    }

    // ---- multisample arrays ------------------------------------------------------------------------
    void MultisampleArrayAndBorderColor::check_ms_counts(demo::Check& check)
    {
        for (const FormatCounts& f : formats_)
        {
            const auto& c = f.counts;
            check.expect(!c.empty(), std::format("GetInternalformativ(TEXTURE_2D_MULTISAMPLE_ARRAY, {}, NUM_SAMPLE_COUNTS) > 0 (section 20.3.1)", f.name),
                         c.empty() ? "no sample count reported" : "");
            if (c.empty()) continue;
            bool descending = c.back() > 0;
            for (std::size_t k = 1; k < c.size(); ++k) descending = descending && c[k] < c[k - 1];
            check.expect(descending, std::format("{}: SAMPLES are positive and in descending order (section 20.3.1)", f.name));
            // "The maximum value in SAMPLES is guaranteed to be at least the lowest of the following": the integer, color-texture
            // and MAX_SAMPLES limits that apply. Taking the minimum of all that may apply is the weakest reading.
            int guarantee = std::min(max_color_samples_, max_samples_);
            if (f.integer) guarantee = std::min(guarantee, max_integer_samples_);
            check.expect(c.front() >= guarantee, std::format("{}: largest supported count {} >= {} (section 20.3.1 guarantee)", f.name, c.front(), guarantee));
        }
    }

    void MultisampleArrayAndBorderColor::check_ms_queries(demo::Check& check)
    {
        for (const MsArray* m : {&ms_float_, &ms_int_, &ms_uint_, &ms_free_})
        {
            const bool small = m == &ms_free_;
            const FormatCounts& f = m->kind == Kind::Int ? formats_[1] : m->kind == Kind::Uint ? formats_[2] : formats_[0];
            check.expect_eq(level_param(m->tex, mgl::TextureLevelParameter::Width), small ? 4 : kW, std::format("{}: TEXTURE_WIDTH (section 8.8)", m->name));
            check.expect_eq(level_param(m->tex, mgl::TextureLevelParameter::Height), small ? 4 : kH, std::format("{}: TEXTURE_HEIGHT", m->name));
            check.expect_eq(level_param(m->tex, mgl::TextureLevelParameter::Depth), small ? 2 : kLayers,
                            std::format("{}: TEXTURE_DEPTH = number of layers (section 8.11.3)", m->name));
            check.expect_eq(level_param(m->tex, mgl::TextureLevelParameter::InternalFormat), static_cast<int>(m->format),
                            std::format("{}: TEXTURE_INTERNAL_FORMAT", m->name));
            const int next = next_count(f.counts, m->requested);
            check.expect(m->samples >= m->requested && m->samples <= next,
                         std::format("{}: TEXTURE_SAMPLES {} is >= the request {} and <= the next supported count {} (section 8.8)", m->name, m->samples,
                                     m->requested, next));
            check.expect_eq(level_param(m->tex, mgl::TextureLevelParameter::FixedSampleLocations), small ? 0 : 1,
                            std::format("{}: TEXTURE_FIXED_SAMPLE_LOCATIONS", m->name));
            GLint immutable = -1;
            mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, m->tex);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2DMultisampleArray, mgl::TextureParameterQuery::ImmutableFormat, &immutable);
            check.expect_eq(immutable, 1, std::format("{}: TEXTURE_IMMUTABLE_FORMAT TRUE (multisample textures are immutable, section 8.8)", m->name));
            check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBinding2DMultisampleArray), static_cast<int>(m->tex.value),
                            std::format("{}: TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY = the bound texture", m->name));
        }
        mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, mgl::TextureId{0});
        check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBinding2DMultisampleArray), 0, "TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY = 0 after unbinding");

        // Every request 1 .. max: the allocated count is >= the request and <= the next supported count (section 8.8).
        const std::vector<GLint>& counts = formats_[0].counts;
        if (!counts.empty())
        {
            Tally t;
            for (int r = 1; r <= counts.front(); ++r)
            {
                mgl::TextureId tex{};
                mgl::glGenTextures(1, &tex);
                mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, tex);
                mgl::glTexStorage3DMultisample(mgl::TextureTarget::Texture2DMultisampleArray, r, mgl::SizedInternalFormat::Rgba8, 2, 2, 3, GL_TRUE);
                const mgl::ErrorCode e = demo::drain_gl_errors();
                const int got = level_param(tex, mgl::TextureLevelParameter::Samples);
                const int next = next_count(counts, r);
                t.add(e == mgl::ErrorCode::NoError && got >= r && got <= next,
                      std::format("request {} -> {} (next supported {}), error {}", r, got, next, demo::error_name(e)));
                mgl::glBindTexture(mgl::TextureTarget::Texture2DMultisampleArray, mgl::TextureId{0});
                mgl::glDeleteTextures(1, &tex);
            }
            check.expect(t.bad == 0, "RGBA8: every request 1..max allocates a count in [request, next supported count] (section 8.8)", t.detail());
        }

        // Framebuffers: complete, SAMPLES = TEXTURE_SAMPLES (section 9.2.3.1), layered or not (FRAMEBUFFER_ATTACHMENT_LAYERED).
        for (const MsArray* m : {&ms_float_, &ms_int_, &ms_uint_})
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m->fbo);
            if (m->kind != Kind::Float)
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), m->tex, 0, 2);
            const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            check.expect(st == mgl::FramebufferStatus::Complete, std::format("{}: framebuffer with the multisample array attached is complete", m->name),
                         demo::framebuffer_status_name(st));
            check.expect_eq(demo::get_integer(mgl::GetParameter::Samples), m->samples, std::format("{}: framebuffer SAMPLES = TEXTURE_SAMPLES", m->name));
            GLint layered = -1, layer = -1;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                                       mgl::FramebufferAttachmentParameter::Layered, &layered);
            check.expect_eq(layered, m->kind == Kind::Float ? 1 : 0,
                            std::format("{}: FRAMEBUFFER_ATTACHMENT_LAYERED ({} attachment, section 9.2.8)", m->name,
                                        m->kind == Kind::Float ? "glFramebufferTexture" : "glFramebufferTextureLayer"));
            if (m->kind != Kind::Float)
            {
                // Not checked: section 9.2.3 defines FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER only for "a three-dimensional texture or a
                // two-dimensional array texture" ("Otherwise params will contain zero"); a multisample array is neither, so the
                // normative text and the evident intent (the attached layer) disagree. The value is printed.
                mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                                           mgl::FramebufferAttachmentParameter::TextureLayer, &layer);
                std::cout << std::format("note: {}: FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER of a layer-2 attachment = {} (not checked, see README)\n", m->name, layer);
            }
        }
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
    }

    void MultisampleArrayAndBorderColor::check_ms_contents(demo::Check& check)
    {
        for (const MsArray* m : {&ms_float_, &ms_int_, &ms_uint_})
        {
            const Fetched f = fetch(*m, kW, kH, kLayers);
            check.expect(f.size == std::array<int, 3>{kW, kH, kLayers}, std::format("{}: textureSize(gsampler2DMSArray) = ({}, {}, {})", m->name, kW, kH, kLayers),
                         std::format("got ({}, {}, {})", f.size[0], f.size[1], f.size[2]));
            const int n = m->samples;
            const auto at = [&](int x, int y, int l, int s) { return f.texels[static_cast<std::size_t>(((l * n + s) * kH + y) * kW + x)]; };
            // Compare layer l, sample s with the pattern of layer (l + dl) % L, sample (s + ds) % n.
            const auto compare = [&](int dl, int ds) {
                Tally t;
                for (int l = 0; l < kLayers; ++l)
                    for (int s = 0; s < n; ++s)
                        for (int y = 0; y < kH; ++y)
                            for (int x = 0; x < kW; ++x)
                            {
                                const U4 want = pattern(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>((l + dl) % kLayers),
                                                        static_cast<std::uint32_t>((s + ds) % n));
                                const U4 got = at(x, y, l, s);
                                t.add(got == want, std::format("layer {} sample {} texel ({},{}): got {} expected {}", l, s, x, y, v4s(got), v4s(want)));
                            }
                return t;
            };
            const Tally exact = compare(0, 0);
            check.expect(exact.bad == 0,
                         std::format("{}: texelFetch(ivec3(x, y, layer), sample) returns the value written to every sample of every layer ({} samples, {})",
                                     m->name, n, m->kind == Kind::Float ? "one layered draw, gl_Layer from the geometry shader" : "one glFramebufferTextureLayer draw per layer"),
                         exact.detail());
            // Negative controls: the same comparison against the wrong layer / the wrong sample must fail almost everywhere.
            const Tally wrong_layer = compare(1, 0);
            check.expect(wrong_layer.bad > wrong_layer.total / 2, std::format("{}: negative control: the layers hold different data (comparison with layer + 1 fails)", m->name),
                         std::format("{} of {} differ", wrong_layer.bad, wrong_layer.total));
            if (n >= 2)
            {
                const Tally wrong_sample = compare(0, 1);
                check.expect(wrong_sample.bad > wrong_sample.total / 2,
                             std::format("{}: negative control: the samples of a texel hold different data (comparison with sample + 1 fails)", m->name),
                             std::format("{} of {} differ", wrong_sample.bad, wrong_sample.total));
            }
            else
            {
                std::cout << std::format("note: {} has 1 sample, so the per-sample negative control is skipped\n", m->name);
            }
        }
    }

    void MultisampleArrayAndBorderColor::check_ms_layered_clear(demo::Check& check)
    {
        // A layered framebuffer: Clear / ClearBuffer clears all layers (section 9.8). Then a single-layer attachment clears only that layer.
        MsArray m = make_ms(Kind::Uint, mgl::SizedInternalFormat::Rgba8UI, "RGBA8UI 4x4x3", pick(formats_[2].counts), true, 4, 4, 3);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, m.fbo);
        mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), m.tex, 0);
        const mgl::FramebufferStatus st = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
        check.expect(st == mgl::FramebufferStatus::Complete, "layered RGBA8UI multisample array framebuffer is complete", demo::framebuffer_status_name(st));
        const std::array<GLuint, 4> all = {11, 22, 33, 44};
        mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, all.data());
        mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), m.tex, 0, 1);
        const std::array<GLuint, 4> one = {55, 66, 77, 88};
        mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, one.data());
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        check.expect_no_gl_error("layered and single-layer glClearBufferuiv on a multisample array");

        const Fetched f = fetch(m, 4, 4, 3);
        Tally t;
        for (int l = 0; l < 3; ++l)
            for (int s = 0; s < m.samples; ++s)
                for (int i = 0; i < 16; ++i)
                {
                    const U4 got = f.texels[static_cast<std::size_t>((l * m.samples + s) * 16 + i)];
                    const U4 want = l == 1 ? U4{55, 66, 77, 88} : U4{11, 22, 33, 44};
                    t.add(got == want, std::format("layer {} sample {} texel {}: got {} expected {}", l, s, i, v4s(got), v4s(want)));
                }
        check.expect(t.bad == 0, "ClearBufferuiv on the layered attachment clears every sample of all 3 layers; on the layer-1 attachment only layer 1", t.detail());
        mgl::glDeleteFramebuffers(1, &m.fbo);
        mgl::glDeleteTextures(1, &m.tex);
    }

    void MultisampleArrayAndBorderColor::check_ms_errors(demo::Check& check)
    {
        const mgl::TextureTarget target = mgl::TextureTarget::Texture2DMultisampleArray;
        const mgl::SizedInternalFormat rgba8 = mgl::SizedInternalFormat::Rgba8;
        mgl::TextureId tex{};
        mgl::glGenTextures(1, &tex);
        mgl::glBindTexture(target, tex);

        mgl::glTexStorage3DMultisample(mgl::TextureTarget::Texture2DMultisample, 1, rgba8, 4, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexStorage3DMultisample with target TEXTURE_2D_MULTISAMPLE -> INVALID_ENUM (section 8.8)");
        mgl::glTexStorage3DMultisample(mgl::TextureTarget::Texture2DArray, 1, rgba8, 4, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexStorage3DMultisample with target TEXTURE_2D_ARRAY -> INVALID_ENUM");
        mgl::glTexStorage3DMultisample(target, 1, rgba8, 4, 4, 0, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "depth 0 -> INVALID_VALUE");
        mgl::glTexStorage3DMultisample(target, 1, rgba8, 0, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "width 0 -> INVALID_VALUE");
        mgl::glTexStorage3DMultisample(target, 0, rgba8, 4, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "samples 0 -> INVALID_VALUE");
        const int max_layers = demo::get_integer(mgl::GetParameter::MaxArrayTextureLayers);
        mgl::glTexStorage3DMultisample(target, 1, rgba8, 4, 4, max_layers + 1, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, std::format("depth MAX_ARRAY_TEXTURE_LAYERS + 1 = {} -> INVALID_VALUE", max_layers + 1));
        const int max_size = demo::get_integer(mgl::GetParameter::MaxTextureSize);
        mgl::glTexStorage3DMultisample(target, 1, rgba8, max_size + 1, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "width MAX_TEXTURE_SIZE + 1 -> INVALID_VALUE");
        mgl::glTexStorage3DMultisample(target, 1, mgl::SizedInternalFormat::Rgb8Snorm, 4, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "RGB8_SNORM (not color-, depth- or stencil-renderable) -> INVALID_ENUM");
        for (const FormatCounts& f : formats_)
        {
            if (f.counts.empty()) continue;
            const mgl::SizedInternalFormat sized = f.format == mgl::InternalFormat::Rgba8I ? mgl::SizedInternalFormat::Rgba8I
                                                   : f.format == mgl::InternalFormat::Rgba8UI ? mgl::SizedInternalFormat::Rgba8UI
                                                                                              : rgba8;
            mgl::glTexStorage3DMultisample(target, f.counts.front() + 1, sized, 4, 4, 2, GL_TRUE);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  std::format("{}: samples = max supported ({}) + 1 -> INVALID_OPERATION (section 8.8)", f.name, f.counts.front()));
        }
        check.expect_eq(level_param(tex, mgl::TextureLevelParameter::Samples), 0, "the failed calls allocated nothing: TEXTURE_SAMPLES still 0");
        GLint immutable = -1;
        mgl::glGetTexParameteriv(target, mgl::TextureParameterQuery::ImmutableFormat, &immutable);
        check.expect_eq(immutable, 0, "the failed calls left TEXTURE_IMMUTABLE_FORMAT FALSE");

        mgl::glTexStorage3DMultisample(target, 1, rgba8, 4, 4, 2, GL_TRUE);
        check.expect_no_gl_error("a valid TexStorage3DMultisample after the failed ones");
        mgl::glTexStorage3DMultisample(target, 1, rgba8, 8, 8, 3, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "re-specifying an immutable multisample array -> INVALID_OPERATION");
        check.expect_eq(level_param(tex, mgl::TextureLevelParameter::Depth), 2, "the rejected re-specification kept TEXTURE_DEPTH 2");

        // FramebufferTextureLayer: a multisample array only has level 0 (section 9.2.8); layer >= MAX_ARRAY_TEXTURE_LAYERS is an error.
        mgl::FramebufferId fbo{};
        mgl::glGenFramebuffers(1, &fbo);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
        mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex, 1, 0);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FramebufferTextureLayer of a multisample array with level 1 -> INVALID_VALUE (section 9.2.8)");
        mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex, 0, max_layers);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FramebufferTextureLayer with layer = MAX_ARRAY_TEXTURE_LAYERS -> INVALID_VALUE");
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glDeleteFramebuffers(1, &fbo);

        mgl::glBindTexture(target, mgl::TextureId{0});
        mgl::glTexStorage3DMultisample(target, 1, rgba8, 4, 4, 2, GL_TRUE);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TexStorage3DMultisample with texture 0 bound -> INVALID_OPERATION");
        mgl::glDeleteTextures(1, &tex);
    }

    // ---- border color ------------------------------------------------------------------------------
    void MultisampleArrayAndBorderColor::check_border_queries(demo::Check& check)
    {
        check.expect(initial_texture_border_ == F4{0.0f, 0.0f, 0.0f, 0.0f}, "initial TEXTURE_BORDER_COLOR of a texture is (0, 0, 0, 0) (section 8.19, Table 21.11)",
                     f4s(initial_texture_border_));
        check.expect(initial_sampler_border_ == F4{0.0f, 0.0f, 0.0f, 0.0f}, "initial TEXTURE_BORDER_COLOR of a sampler object is (0, 0, 0, 0) (Table 21.12)",
                     f4s(initial_sampler_border_));

        F4 got{};
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_rgba8_);
        mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::BorderColor, got.data());
        check.expect(got == kBorderRgba8, "GetTexParameterfv returns the TexParameterfv border unmodified, (1.5, -0.25, ...) not clamped (section 8.10)", f4s(got));
        GLint wrap_s = 0, wrap_t = 0;
        mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::WrapS, &wrap_s);
        mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::WrapT, &wrap_t);
        check.expect(wrap_s == static_cast<GLint>(mgl::TextureWrapMode::ClampToBorder) && wrap_t == wrap_s, "TEXTURE_WRAP_S / T read back as CLAMP_TO_BORDER",
                     std::format("0x{:X} 0x{:X}", wrap_s, wrap_t));

        I4 got_i{};
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_rgba8i_);
        mgl::glGetTexParameterIiv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::BorderColor, got_i.data());
        check.expect(got_i == kBorderRgba8I, "GetTexParameterIiv returns the TexParameterIiv border unmodified, (300, 40, -200, 5) not clamped (section 8.11.2)", v4s(got_i));
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

        mgl::glGetSamplerParameterfv(sampler_override_, mgl::SamplerParameter::BorderColor, got.data());
        check.expect(got == kBorderSampler, "GetSamplerParameterfv returns the SamplerParameterfv border (section 8.3)", f4s(got));
        GLint sampler_wrap = 0;
        mgl::glGetSamplerParameteriv(sampler_override_, mgl::SamplerParameter::WrapS, &sampler_wrap);
        check.expect_eq(sampler_wrap, static_cast<GLint>(mgl::TextureWrapMode::ClampToBorder), "sampler TEXTURE_WRAP_S reads back as CLAMP_TO_BORDER");
    }

    void MultisampleArrayAndBorderColor::check_float_lookups(demo::Check& check, mgl::TextureId tex, mgl::SamplerId sampler, bool clamp_to_border,
                                                             double tolerance, F4 (*texel)(int, int), F4 border, std::string_view what)
    {
        const std::vector<Probe> probes = border_probes();
        const std::vector<U4> raw = lookup(Kind::Float, tex, sampler, probes);
        Tally t;
        for (std::size_t k = 0; k < probes.size(); ++k)
        {
            const Selected sel = select_nearest(probes[k], clamp_to_border);
            const F4 want = sel.border ? border : texel(sel.i, sel.j);
            const F4 got = as_float(raw[k]);
            bool ok = true;
            for (std::size_t c = 0; c < 4; ++c)
            {
                const double err = std::abs(static_cast<double>(got[c]) - static_cast<double>(want[c]));
                t.max_err = std::max(t.max_err, err);
                ok = ok && err <= tolerance;
            }
            t.add(ok, std::format("(s, t) = ({}, {}) {}: got {} expected {}", probes[k].s, probes[k].t, sel.border ? "border" : "texel", f4s(got), f4s(want)));
        }
        check.expect(t.bad == 0, what, t.detail());
    }

    F4 rgba8_as_float(int i, int j)
    {
        const auto c = texel_rgba8(i, j);
        return {static_cast<float>(unorm8(c[0])), static_cast<float>(unorm8(c[1])), static_cast<float>(unorm8(c[2])), static_cast<float>(unorm8(c[3]))};
    }
    F4 rg8_as_float(int i, int j)
    {
        const auto c = texel_rg8(i, j);
        return {static_cast<float>(unorm8(c[0])), static_cast<float>(unorm8(c[1])), 0.0f, 1.0f};  // Table 14.1: RG -> (R, G, 0, 1)
    }

    void MultisampleArrayAndBorderColor::check_border_normalized(demo::Check& check)
    {
        // Tolerance for unsigned normalized lookups: the texels are k / 255 and the borders used are 0, 1 or multiples of 1/255
        // (0.2 = 51/255, 0.6 = 153/255, 0.4 = 102/255), so a conversion of the border to the format's precision changes nothing;
        // half an 8-bit step allows only float rounding.
        const double tol = 0.5 / 255.0;

        // 1. RGBA8 with the texture's border (1.5, -0.25, 0.2, 0.6): "For ... unsigned normalized fixed-point formats, border values
        //    are clamped to [0, 1]" (section 8.14.2) -> (1, 0, 0.2, 0.6).
        check_float_lookups(check, tex_rgba8_, mgl::SamplerId{0}, true, tol, rgba8_as_float, F4{1.0f, 0.0f, 0.2f, 0.6f},
                            "RGBA8, CLAMP_TO_BORDER: texels inside, border (1.5, -0.25, 0.2, 0.6) clamped to (1, 0, 0.2, 0.6) outside [0, 1]");

        // 2. A fresh RGBA8 texture: the initial border (0, 0, 0, 0); then glTexParameteriv, converted with equation 2.2 (section 8.10).
        std::vector<std::uint8_t> rgba8;
        std::vector<std::uint8_t> rg8;
        for (int j = 0; j < kTex; ++j)
            for (int i = 0; i < kTex; ++i)
            {
                for (const std::uint8_t c : texel_rgba8(i, j)) rgba8.push_back(c);
                for (const std::uint8_t c : texel_rg8(i, j)) rg8.push_back(c);
            }
        const mgl::TextureId fresh = make_texture_2d(mgl::SizedInternalFormat::Rgba8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, rgba8.data());
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        check_float_lookups(check, fresh, mgl::SamplerId{0}, true, tol, rgba8_as_float, F4{0.0f, 0.0f, 0.0f, 0.0f},
                            "RGBA8, CLAMP_TO_BORDER with the initial border: (0, 0, 0, 0) outside [0, 1]");
        const std::array<GLint, 4> border_iv = {INT_MAX, -INT_MAX, 0, INT_MAX};
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, fresh);
        mgl::glTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, border_iv.data());
        F4 got{};
        mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::BorderColor, got.data());
        // Equation 2.2 (b = 32): f = max(c / (2^31 - 1), -1). INT_MAX and -INT_MAX give 1 and -1 (1e-6: the float division), and 0 gives
        // exactly 0 - the pre-ES 3.0 / GL 4.1 formula (2c + 1) / (2^32 - 1) would give 2^-32 = 2.3e-10 instead.
        const bool conv_ok = std::abs(got[0] - 1.0f) <= 1e-6f && std::abs(got[1] + 1.0f) <= 1e-6f && std::abs(got[3] - 1.0f) <= 1e-6f;
        check.expect(conv_ok, "TexParameteriv border components INT_MAX, -INT_MAX, INT_MAX are stored as 1, -1, 1 (equation 2.2, section 8.10)", f4s(got));
        check.expect_or_known_deviation(got[2] == 0.0f, "TexParameteriv border component 0 is stored as exactly 0.0 (equation 2.2)",
                                        std::format("got {} (the old (2c + 1) / (2^32 - 1) formula gives 2^-32)", got[2]), "Mesa",
                                        "Mesa converts glTexParameteriv(TEXTURE_BORDER_COLOR) integers with (2c + 1) / (2^32 - 1), not ES 3.2 equation 2.2");
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        check_float_lookups(check, fresh, mgl::SamplerId{0}, true, tol, rgba8_as_float, F4{1.0f, 0.0f, 0.0f, 1.0f},
                            "RGBA8, border from TexParameteriv: (1, -1, 0, 1) clamped to (1, 0, 0, 1)");
        mgl::glDeleteTextures(1, &fresh);

        // 3. RG8: "the values of TEXTURE_BORDER_COLOR are interpreted as an RGBA color to match the texture's internal format in a manner
        //    consistent with table 8.8": only R and G exist, and the lookup returns (R, G, 0, 1) (Table 14.1), not the border's B and A.
        const mgl::TextureId rg = make_texture_2d(mgl::SizedInternalFormat::Rg8, mgl::PixelFormat::Rg, mgl::PixelType::UnsignedByte, rg8.data());
        const F4 border_rg = {0.6f, 0.2f, 0.8f, 0.4f};  // 153/255, 51/255, 204/255, 102/255
        mgl::glTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, border_rg.data());
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        check_float_lookups(check, rg, mgl::SamplerId{0}, true, tol, rg8_as_float, F4{0.6f, 0.2f, 0.0f, 1.0f},
                            "RG8: border (0.6, 0.2, 0.8, 0.4) is looked up as (0.6, 0.2, 0, 1) (Table 8.8 components, Table 14.1)");
        mgl::glDeleteTextures(1, &rg);
    }

    void MultisampleArrayAndBorderColor::check_border_float(demo::Check& check)
    {
        std::vector<float> data;
        for (int j = 0; j < kTex; ++j)
            for (int i = 0; i < kTex; ++i)
                for (const float c : texel_rgba32f(i, j)) data.push_back(c);
        const mgl::TextureId tex = make_texture_2d(mgl::SizedInternalFormat::Rgba32F, mgl::PixelFormat::Rgba, mgl::PixelType::Float, data.data());
        mgl::glTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, kBorderRgba32F.data());
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        // "For floating-point ... formats, border values are clamped to the representable range of the format": every component of
        // (-2.5, 3, 1000, 0.125) is representable in a 32-bit float, so the lookup returns it bit for bit.
        check_float_lookups(check, tex, mgl::SamplerId{0}, true, 0.0, texel_rgba32f, kBorderRgba32F,
                            "RGBA32F: border (-2.5, 3, 1000, 0.125) returned exactly (no clamping to [0, 1] for float formats)");
        mgl::glDeleteTextures(1, &tex);
    }

    void MultisampleArrayAndBorderColor::check_border_integer(demo::Check& check)
    {
        const std::vector<Probe> probes = border_probes();
        const auto run = [&](Kind kind, mgl::TextureId tex, mgl::SamplerId sampler, auto texel, auto border, std::string_view what) {
            const std::vector<U4> raw = lookup(kind, tex, sampler, probes);
            Tally t;
            for (std::size_t k = 0; k < probes.size(); ++k)
            {
                const Selected sel = select_nearest(probes[k], true);
                const auto want = sel.border ? border : texel(sel.i, sel.j);
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(want)>, I4>)
                {
                    const I4 got = as_int(raw[k]);
                    t.add(got == want, std::format("({}, {}): got {} expected {}", probes[k].s, probes[k].t, v4s(got), v4s(want)));
                }
                else
                {
                    t.add(raw[k] == want, std::format("({}, {}): got {} expected {}", probes[k].s, probes[k].t, v4s(raw[k]), v4s(want)));
                }
            }
            check.expect(t.bad == 0, what, t.detail());
        };

        // RGBA8I, border (300, 40, -200, 5) from TexParameterIiv: "For ... integer formats, border values are clamped to the representable
        // range of the format" -> (127, 40, -128, 5).
        run(Kind::Int, tex_rgba8i_, mgl::SamplerId{0}, texel_rgba8i, I4{127, 40, -128, 5},
            "RGBA8I (isampler2D): border (300, 40, -200, 5) from TexParameterIiv is clamped to (127, 40, -128, 5)");

        // RGBA16UI, border (70000, 3, 0xFFFFFFFF, 65535) from TexParameterIuiv -> (65535, 3, 65535, 65535).
        std::vector<std::uint16_t> u16;
        std::vector<std::uint32_t> u32;
        for (int j = 0; j < kTex; ++j)
            for (int i = 0; i < kTex; ++i)
            {
                for (const std::uint32_t c : texel_rgba16ui(i, j)) u16.push_back(static_cast<std::uint16_t>(c));
                for (const std::uint32_t c : texel_rgba32ui(i, j)) u32.push_back(c);
            }
        const mgl::TextureId t16 = make_texture_2d(mgl::SizedInternalFormat::Rgba16UI, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedShort, u16.data());
        mgl::glTexParameterIuiv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, kBorderRgba16UI.data());
        U4 got_u{};
        mgl::glGetTexParameterIuiv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::BorderColor, got_u.data());
        check.expect(got_u == kBorderRgba16UI, "GetTexParameterIuiv returns the TexParameterIuiv border unmodified (section 8.11.2)", v4s(got_u));
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        run(Kind::Uint, t16, mgl::SamplerId{0}, texel_rgba16ui, U4{65535u, 3u, 65535u, 65535u},
            "RGBA16UI (usampler2D): border (70000, 3, 0xFFFFFFFF, 65535) from TexParameterIuiv is clamped to (65535, 3, 65535, 65535)");

        // RGBA32UI: every value is representable, so the border is returned unchanged.
        const mgl::TextureId t32 = make_texture_2d(mgl::SizedInternalFormat::Rgba32UI, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, u32.data());
        mgl::glTexParameterIuiv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, kBorderRgba32UI.data());
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        run(Kind::Uint, t32, mgl::SamplerId{0}, texel_rgba32ui, kBorderRgba32UI,
            "RGBA32UI (usampler2D): border (0xFFFFFFFF, 0x80000000, 7, 123456789) returned unchanged");

        // Sampler objects with integer borders: SamplerParameterIiv / Iuiv, read back with GetSamplerParameterIiv / Iuiv.
        const mgl::SamplerId si = make_sampler(mgl::TextureWrapMode::ClampToBorder, false);
        mgl::glSamplerParameterIiv(si, mgl::SamplerParameter::BorderColor, kBorderSamplerI.data());
        I4 got_si{};
        mgl::glGetSamplerParameterIiv(si, mgl::SamplerParameter::BorderColor, got_si.data());
        check.expect(got_si == kBorderSamplerI, "GetSamplerParameterIiv returns the SamplerParameterIiv border unmodified (section 8.3)", v4s(got_si));
        run(Kind::Int, tex_rgba8i_, si, texel_rgba8i, I4{-1, 2, -128, 4},
            "RGBA8I with a sampler object: the sampler's border (-1, 2, -300, 4) replaces the texture's, clamped to (-1, 2, -128, 4)");
        const mgl::SamplerId su = make_sampler(mgl::TextureWrapMode::ClampToBorder, false);
        mgl::glSamplerParameterIuiv(su, mgl::SamplerParameter::BorderColor, kBorderSamplerUI.data());
        U4 got_su{};
        mgl::glGetSamplerParameterIuiv(su, mgl::SamplerParameter::BorderColor, got_su.data());
        check.expect(got_su == kBorderSamplerUI, "GetSamplerParameterIuiv returns the SamplerParameterIuiv border unmodified", v4s(got_su));
        run(Kind::Uint, t16, su, texel_rgba16ui, U4{1u, 2u, 3u, 65535u},
            "RGBA16UI with a sampler object: the sampler's border (1, 2, 3, 70000) replaces the texture's, clamped to (1, 2, 3, 65535)");

        mgl::glDeleteSamplers(1, &si);
        mgl::glDeleteSamplers(1, &su);
        mgl::glDeleteTextures(1, &t16);
        mgl::glDeleteTextures(1, &t32);
    }

    void MultisampleArrayAndBorderColor::check_border_depth(demo::Check& check)
    {
        // DEPTH_COMPONENT16 with COMPARE_REF_TO_TEXTURE / LEQUAL: "If the texture contains depth components, the first component of
        // TEXTURE_BORDER_COLOR is interpreted as a depth value" (section 8.14.2); the result is 1 if Dref <= Dt (Table 8.22).
        const std::vector<Probe> probes = border_probes({0.1f, 0.5f, 0.95f});
        const std::vector<U4> raw = lookup(Kind::Shadow, tex_shadow_, mgl::SamplerId{0}, probes);
        Tally t;
        int border_pass = 0, border_fail = 0;
        for (std::size_t k = 0; k < probes.size(); ++k)
        {
            const Selected sel = select_nearest(probes[k], true);
            const double d = sel.border ? static_cast<double>(kBorderDepth) : depth_value(sel.i, sel.j);
            const float want = static_cast<double>(probes[k].ref) <= d ? 1.0f : 0.0f;
            if (sel.border) (want == 1.0f ? border_pass : border_fail)++;
            const float got = std::bit_cast<float>(raw[k][0]);
            t.add(got == want, std::format("({}, {}) ref {} {}: got {} expected {}", probes[k].s, probes[k].t, probes[k].ref, sel.border ? "border" : "texel", got, want));
        }
        check.expect(t.bad == 0 && border_pass > 0 && border_fail > 0,
                     "DEPTH_COMPONENT16 shadow lookups: border depth 0.875 compared with Dref 0.1 / 0.5 (pass) and 0.95 (fail), texels inside", t.detail());

        // Without comparison the lookup returns the depth (section 8.20.1: r = Dt); only the first component is specified, so only it is checked.
        std::vector<std::uint16_t> depth;
        for (int j = 0; j < kTex; ++j)
            for (int i = 0; i < kTex; ++i) depth.push_back(texel_depth16(i, j));
        const mgl::TextureId tex = make_texture_2d(mgl::SizedInternalFormat::DepthComponent16, mgl::PixelFormat::DepthComponent, mgl::PixelType::UnsignedShort, depth.data());
        const std::vector<Probe> plain = border_probes();
        const std::array<std::pair<float, double>, 3> cases = {{{0.25f, 0.25}, {1.5f, 1.0}, {-0.5f, 0.0}}};
        for (const auto& [border, expected] : cases)
        {
            const F4 b = {border, 0.9f, 0.9f, 0.9f};
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, b.data());
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            const std::vector<U4> r = lookup(Kind::Float, tex, mgl::SamplerId{0}, plain);
            Tally td;
            for (std::size_t k = 0; k < plain.size(); ++k)
            {
                const Selected sel = select_nearest(plain[k], true);
                const double want = sel.border ? expected : depth_value(sel.i, sel.j);
                const double got = static_cast<double>(std::bit_cast<float>(r[k][0]));
                td.max_err = std::max(td.max_err, std::abs(got - want));
                // 1e-4: the border may be stored at the 16-bit precision of the format (one step is 1.5e-5).
                td.add(std::abs(got - want) <= 1e-4, std::format("({}, {}) {}: got {} expected {}", plain[k].s, plain[k].t, sel.border ? "border" : "texel", got, want));
            }
            check.expect(td.bad == 0,
                         std::format("DEPTH_COMPONENT16 without comparison, border ({}, 0.9, 0.9, 0.9): first component outside [0, 1] = {} "
                                     "(the depth value, clamped to [0, 1] for a fixed-point depth format)",
                                     border, expected),
                         td.detail());
        }
        mgl::glDeleteTextures(1, &tex);
    }

    void MultisampleArrayAndBorderColor::check_sampler_override(demo::Check& check)
    {
        const double tol = 0.5 / 255.0;
        // A sampler object bound to the unit replaces all of the texture's sampler state, the border color included (section 8.2).
        check_float_lookups(check, tex_rgba8_, sampler_override_, true, tol, rgba8_as_float, kBorderSampler,
                            "RGBA8 with a sampler object bound: the sampler's border (0, 1, 0.4, 1) replaces the texture's (1.5, -0.25, 0.2, 0.6)");
        // Unbinding the sampler (lookup() binds sampler 0) brings the texture's own border back.
        check_float_lookups(check, tex_rgba8_, mgl::SamplerId{0}, true, tol, rgba8_as_float, F4{1.0f, 0.0f, 0.2f, 0.6f},
                            "after BindSampler(0, 0) the texture's own border (clamped (1, 0, 0.2, 0.6)) is used again");

        // Negative control: a CLAMP_TO_EDGE sampler on the same texture returns the edge texels, which differ from the border at every
        // probe outside [0, 1] - so the border checks above could not pass by returning edge texels.
        const mgl::SamplerId edge = make_sampler(mgl::TextureWrapMode::ClampToEdge, false);
        check_float_lookups(check, tex_rgba8_, edge, false, tol, rgba8_as_float, F4{},
                            "negative control: a CLAMP_TO_EDGE sampler overrides the texture's CLAMP_TO_BORDER, lookups outside [0, 1] return edge texels");
        const std::vector<Probe> probes = border_probes();
        int outside = 0, distinguishable = 0;
        for (const Probe& p : probes)
        {
            const Selected b = select_nearest(p, true);
            if (!b.border) continue;
            ++outside;
            const Selected e = select_nearest(p, false);
            const F4 edge_texel = rgba8_as_float(e.i, e.j);
            for (const F4& border : {F4{1.0f, 0.0f, 0.2f, 0.6f}, kBorderSampler})
            {
                bool differs = false;
                for (std::size_t c = 0; c < 4; ++c) differs = differs || std::abs(static_cast<double>(edge_texel[c]) - static_cast<double>(border[c])) > 2.0 * tol;
                if (differs) ++distinguishable;
            }
        }
        check.expect(outside > 0 && distinguishable == 2 * outside,
                     "negative control: at every probe outside [0, 1] the edge texel differs from both border colors by more than the tolerance",
                     std::format("{} of {} pairs differ", distinguishable, 2 * outside));
        mgl::glDeleteSamplers(1, &edge);
    }

    void MultisampleArrayAndBorderColor::check_border_linear(demo::Check& check)
    {
        // LINEAR across the edge (section 8.14.2): at s = 0 the 2x2 footprint is i0 = floor(0 * 4 - 0.5) = -1 (a border texel) and
        // i1 = 0, with weight alpha = frac(-0.5) = 0.5; with t at a texel centre beta = 0. So the result is (border + texel(0, j)) / 2.
        const mgl::SamplerId linear = make_sampler(mgl::TextureWrapMode::ClampToBorder, true);
        mgl::glSamplerParameterfv(linear, mgl::SamplerParameter::BorderColor, kBorderLinear.data());
        std::vector<Probe> probes;
        std::vector<F4> want;
        for (int j = 0; j < kTex; ++j)
        {
            const float c = (static_cast<float>(j) + 0.5f) / kTex;
            for (const auto& [s, t, i, j2] : {std::tuple{0.0f, c, 0, j}, std::tuple{1.0f, c, kTex - 1, j}, std::tuple{c, 0.0f, j, 0}, std::tuple{c, 1.0f, j, kTex - 1}})
            {
                probes.push_back({s, t, 0.0f, 0.0f});
                const F4 tx = rgba8_as_float(i, j2);
                want.push_back({(tx[0] + kBorderLinear[0]) * 0.5f, (tx[1] + kBorderLinear[1]) * 0.5f, (tx[2] + kBorderLinear[2]) * 0.5f, (tx[3] + kBorderLinear[3]) * 0.5f});
            }
        }
        const std::vector<U4> raw = lookup(Kind::Float, tex_rgba8_, linear, probes);
        Tally t;
        // The weights' precision is not specified; 2/255 is a sanity bound for an exactly representable weight of 0.5, the measured
        // maximum error is printed.
        for (std::size_t k = 0; k < probes.size(); ++k)
        {
            const F4 got = as_float(raw[k]);
            bool ok = true;
            for (std::size_t c = 0; c < 4; ++c)
            {
                const double err = std::abs(static_cast<double>(got[c]) - static_cast<double>(want[k][c]));
                t.max_err = std::max(t.max_err, err);
                ok = ok && err <= 2.0 / 255.0;
            }
            t.add(ok, std::format("({}, {}): got {} expected {}", probes[k].s, probes[k].t, f4s(got), f4s(want[k])));
        }
        check.expect(t.bad == 0, "LINEAR at s or t = 0 / 1: half border, half edge texel (the border takes part in the 2x2 filter)", t.detail());
        mgl::glDeleteSamplers(1, &linear);
    }

    void MultisampleArrayAndBorderColor::check_border_errors(demo::Check& check)
    {
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_rgba8_);
        mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexParameteri(TEXTURE_BORDER_COLOR) -> INVALID_ENUM (scalar form for a non-scalar parameter, section 8.10)");
        mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BorderColor, 1.0f);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexParameterf(TEXTURE_BORDER_COLOR) -> INVALID_ENUM");
        F4 got{};
        mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::BorderColor, got.data());
        check.expect(got == kBorderRgba8, "the rejected calls left the texture's border unchanged", f4s(got));
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

        mgl::glSamplerParameteri(sampler_override_, mgl::SamplerParameter::BorderColor, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "SamplerParameteri(TEXTURE_BORDER_COLOR) -> INVALID_ENUM (section 8.2)");
        mgl::glSamplerParameterf(sampler_override_, mgl::SamplerParameter::BorderColor, 1.0f);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "SamplerParameterf(TEXTURE_BORDER_COLOR) -> INVALID_ENUM");
        mgl::glGetSamplerParameterfv(sampler_override_, mgl::SamplerParameter::BorderColor, got.data());
        check.expect(got == kBorderSampler, "the rejected calls left the sampler's border unchanged", f4s(got));

        mgl::SamplerId gone{};
        mgl::glGenSamplers(1, &gone);
        mgl::glDeleteSamplers(1, &gone);
        mgl::glSamplerParameterIiv(gone, mgl::SamplerParameter::BorderColor, kBorderSamplerI.data());
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "SamplerParameterIiv on a deleted sampler name -> INVALID_OPERATION (section 8.2)");

        // Sampler state is not allowed on multisample targets (section 8.10): the border color and the wrap modes are sampler state (Table 21.12).
        const mgl::TextureTarget ms = mgl::TextureTarget::Texture2DMultisampleArray;
        mgl::glBindTexture(ms, ms_float_.tex);
        mgl::glTexParameterfv(ms, mgl::TextureParameterSetter::BorderColor, kBorderSampler.data());
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexParameterfv(TEXTURE_2D_MULTISAMPLE_ARRAY, TEXTURE_BORDER_COLOR) -> INVALID_ENUM");
        mgl::glTexParameterIiv(ms, mgl::TextureParameterSetter::BorderColor, kBorderSamplerI.data());
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexParameterIiv(TEXTURE_2D_MULTISAMPLE_ARRAY, TEXTURE_BORDER_COLOR) -> INVALID_ENUM");
        mgl::glTexParameter(ms, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToBorder);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexParameteri(TEXTURE_2D_MULTISAMPLE_ARRAY, TEXTURE_WRAP_S) -> INVALID_ENUM");
        mgl::glTexParameter(ms, mgl::TextureMinFilter::Nearest);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "TexParameteri(TEXTURE_2D_MULTISAMPLE_ARRAY, TEXTURE_MIN_FILTER) -> INVALID_ENUM");
        mgl::glTexParameteri(ms, mgl::TextureParameterSetter::BaseLevel, 1);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "TexParameteri(TEXTURE_2D_MULTISAMPLE_ARRAY, TEXTURE_BASE_LEVEL, 1) -> INVALID_OPERATION");
        mgl::glTexParameteri(ms, mgl::TextureParameterSetter::BaseLevel, 0);
        check.expect_no_gl_error("TexParameteri(TEXTURE_2D_MULTISAMPLE_ARRAY, TEXTURE_BASE_LEVEL, 0) is allowed");
        mgl::glBindTexture(ms, mgl::TextureId{0});
    }

    // ---- the picture -------------------------------------------------------------------------------
    void MultisampleArrayAndBorderColor::check_picture(demo::Check& check, int width, int height)
    {
        const Layout lay(width, height, ms_float_.samples);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, width, height);
        const auto at = [&](int x, int y) {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4;
            return demo::Rgba8{px[o], px[o + 1], px[o + 2], px[o + 3]};
        };
        // Which quad (if any) covers a pixel.
        const auto quad_at = [&](int x, int y) {
            for (int q = 0; q < 4; ++q)
            {
                const Layout::Quad& r = lay.quads[static_cast<std::size_t>(q)];
                if (x >= r.x && x < r.x + r.size && y >= r.y && y < r.y + r.size) return q;
            }
            return -1;
        };

        // ---- top half, every pixel: the same integer arithmetic as shaders/show_ms.frag, so the reference is exact ----
        Tally grid;
        std::vector<bool> seen(static_cast<std::size_t>(kLayers * lay.shown * kW * kH), false);
        for (int y = lay.grid_y; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                demo::Rgba8 want = kClear;
                const int py = y - lay.grid_y;
                const int col = x / lay.tile_w, row = py / lay.tile_h;
                const int lx = x - col * lay.tile_w, ly = py - row * lay.tile_h;
                if (col < lay.shown && row < kLayers && lx < lay.tile_w - lay.gap && ly >= lay.gap)
                {
                    const int layer = kLayers - 1 - row;
                    const int tx = lx * kW / (lay.tile_w - lay.gap), ty = (ly - lay.gap) * kH / (lay.tile_h - lay.gap);
                    seen[static_cast<std::size_t>(((layer * lay.shown + col) * kH + ty) * kW + tx)] = true;
                    const U4 p = pattern(static_cast<std::uint32_t>(tx), static_cast<std::uint32_t>(ty), static_cast<std::uint32_t>(layer), static_cast<std::uint32_t>(col));
                    want = {static_cast<std::uint8_t>(p[0]), static_cast<std::uint8_t>(p[1]), static_cast<std::uint8_t>(p[2]), 255};
                }
                t_add_rgb(grid, at(x, y), want, 0, x, y);
            }
        check.expect(grid.bad == 0,
                     std::format("picture, top half: every pixel shows texelFetch of its (layer, sample, texel) of the RGBA8 array ({} samples shown, tiles {}x{}, gap {})",
                                 lay.shown, lay.tile_w, lay.tile_h, lay.gap),
                     grid.detail());
        const auto shown_texels = std::count(seen.begin(), seen.end(), true);
        check.expect(shown_texels == static_cast<std::ptrdiff_t>(seen.size()), "picture, top half: every texel of every shown (layer, sample) appears on screen",
                     std::format("{} of {}", shown_texels, seen.size()));

        // ---- bottom half, every pixel: clear color outside the quads; inside, the texel or border cell the pixel centre selects ----
        // A pixel centre at (l + 0.5) in a quad of `size` pixels has the texel coordinate u = -2 + 8 (l + 0.5) / size (the texture
        // spans [0, 4) of it). The interpolated coordinate is not exact, so where u lies within 0.02 texel of an integer both
        // neighbouring cells (texel / texel or texel / border) are accepted; everywhere else exactly one is.
        const auto candidates = [](int l, int size) {
            const double u = -2.0 + 8.0 * (static_cast<double>(l) + 0.5) / static_cast<double>(size);
            const int i = static_cast<int>(std::floor(u));
            const double f = u - static_cast<double>(i);
            std::vector<int> c = {i};
            if (f < 0.02) c.push_back(i - 1);
            if (f > 0.98) c.push_back(i + 1);
            return c;
        };
        const auto expected = [&](int q, int i, int j) {
            const bool border = i < 0 || i >= kTex || j < 0 || j >= kTex;
            if (q == 0 || q == 1)
            {
                if (border) return q == 0 ? demo::Rgba8{255, 0, 51, 255} : demo::Rgba8{0, 255, 102, 255};
                const auto c = texel_rgba8(i, j);
                return demo::Rgba8{c[0], c[1], c[2], 255};
            }
            if (q == 2)
            {
                const I4 c = border ? I4{127, 40, -128, 5} : texel_rgba8i(i, j);
                return demo::Rgba8{static_cast<std::uint8_t>(c[0] + 128), static_cast<std::uint8_t>(c[1] + 128), static_cast<std::uint8_t>(c[2] + 128), 255};
            }
            const double d = border ? static_cast<double>(kBorderDepth) : depth_value(i, j);
            const std::uint8_t v = static_cast<double>(kShowRef) <= d ? 255 : 0;
            return demo::Rgba8{v, v, v, 255};
        };
        Tally quads;
        int ambiguous = 0;
        std::array<int, 4> border_px{}, texel_px{};
        for (int y = 0; y < lay.grid_y; ++y)
            for (int x = 0; x < width; ++x)
            {
                const int q = quad_at(x, y);
                const demo::Rgba8 got = at(x, y);
                if (q < 0)
                {
                    t_add_rgb(quads, got, kClear, 0, x, y);
                    continue;
                }
                const Layout::Quad& r = lay.quads[static_cast<std::size_t>(q)];
                const std::vector<int> ci = candidates(x - r.x, r.size), cj = candidates(y - r.y, r.size);
                if (ci.size() > 1 || cj.size() > 1) ++ambiguous;
                bool ok = false;
                for (const int i : ci)
                    for (const int j : cj)
                    {
                        const demo::Rgba8 want = expected(q, i, j);
                        bool match = true;
                        // ±1: a border value that is not stored at 8-bit precision may round either way when written to RGBA8 (section 2.3.5).
                        for (std::size_t c = 0; c < 4; ++c) match = match && std::abs(static_cast<int>(got[c]) - static_cast<int>(want[c])) <= 1;
                        ok = ok || match;
                    }
                const int i0 = ci.front(), j0 = cj.front();
                ((i0 < 0 || i0 >= kTex || j0 < 0 || j0 >= kTex) ? border_px : texel_px)[static_cast<std::size_t>(q)]++;
                quads.add(ok, std::format("quad {} pixel ({}, {}): got {} expected {}{}", q, x, y, demo::format_rgba(got), demo::format_rgba(expected(q, i0, j0)),
                                          ci.size() * cj.size() > 1 ? " (or a neighbouring cell)" : ""));
            }
        check.expect(quads.bad == 0,
                     std::format("picture, bottom half: every pixel is the clear color or the texel / border cell of its quad (quads {} px; {} pixels near a "
                                 "texel boundary accept either neighbour)",
                                 lay.quads[0].size, ambiguous),
                     quads.detail());
        bool both = true;
        for (std::size_t q = 0; q < 4; ++q) both = both && border_px[q] > 0 && texel_px[q] > 0;
        check.expect(both, "picture, bottom half: every quad shows both texels and border",
                     std::format("border/texel pixels: {}/{} {}/{} {}/{} {}/{}", border_px[0], texel_px[0], border_px[1], texel_px[1], border_px[2], texel_px[2],
                                 border_px[3], texel_px[3]));
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-multisample-array-and-border-color",
        .title = "Multisample array textures and the border color",
        .min_es = {3, 2},
        .summary = "glTexStorage3DMultisample (TEXTURE_2D_MULTISAMPLE_ARRAY) filled per layer and with layered rendering, every sample of every layer read back with "
                   "texelFetch(sampler2DMSArray, ivec3, sample); CLAMP_TO_BORDER with TEXTURE_BORDER_COLOR set by glTexParameter{fv,iv,Iiv,Iuiv} and "
                   "glSamplerParameter{fv,Iiv,Iuiv} on normalized, float, integer and depth textures.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<MultisampleArrayAndBorderColor>(argc, argv, info);
}
