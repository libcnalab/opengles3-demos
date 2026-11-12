// es31-texture-gather
//
// textureGather and textureGatherOffset (GLSL ES 3.10 section 8.9.3, OpenGL ES 3.2 section 8.14.2 and figure 8.7):
// one call returns one component of each texel of the 2 x 2 footprint of the LINEAR filter, in the order
// (i0 j1, i1 j1, i1 j0, i0 j0) = (x, y, z, w).
//
// Every query is one fragment. The fragment at window pixel (a, b) gathers at the texel-space point (a, b) of an
// 8 x 8 texture, the corner shared by four texels, so the footprint is exactly texels (a-1..a, b-1..b) and the
// result image of 9 x 9 pixels visits every footprint of the texture, including the ones that reach past its edge.
// The textures hold a value in every (texel, component) pair that differs from every other one (a permutation of
// 0..255 for the 8-bit ones), so a wrong texel, a wrong order or a wrong component can never give the right value.
//
// Validation, all exact (no filtering takes place: gather returns texels, not blends):
//   * the limits MIN/MAX_PROGRAM_TEXTURE_GATHER_OFFSET against the relations of section 8.14.2;
//   * the order of figure 8.7 from four independently written texel lookups, and a mutation (the row-major order)
//     that must NOT match;
//   * 2D textures x CLAMP_TO_EDGE / REPEAT / MIRRORED_REPEAT x comp 0..3 (the whole 9 x 9 result against a CPU
//     model of table 8.21), comp omitted, a constant-expression comp, min/mag filters that must be ignored, and
//     a texture swizzle that gather must apply (including the constants ZERO and ONE);
//   * integer samplers (RGBA8I with negative values, RGBA32UI with values above 2^24);
//   * 2D array textures (layers, clamped layer coordinates) and cube maps (all six faces, all four components);
//   * textureGatherOffset with constant offsets up to the queried limits, on 2D, array and integer textures;
//   * shadow samplers (2D, array, cube; four comparison functions): the four comparison results;
//   * an incomplete texture, which gathers (0, 0, 0, 1);
//   * the compile-time rules: a non-constant comp or offset is rejected by a 3.10 compiler;
//   * on an ES 3.2 context only: textureGatherOffsets with a constant offsets array (GLSL ES 3.20).
// The window shows sixteen of the result images in a 4 x 4 grid of cells that tiles the surface (boundaries at k * width / 4 and
// k * height / 4), each image magnified by an integer factor per axis with a NEAREST blit; read back and compared as well.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
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
using demo::Rgba8;

namespace
{
    constexpr int kTex = 8;       // texels per edge of every source texture
    constexpr int kTarget = 16;   // edge of the result targets
    constexpr int kGrid = 4;        // the window is a kGrid x kGrid grid of cells
    constexpr int kMinWindow = 40;  // a cell must hold a 9 x 9 result image at scale 1 plus a 1-pixel gap: 4 * (9 + 1)

    using U8x4 = std::array<std::uint8_t, 4>;
    using I32x4 = std::array<std::int32_t, 4>;
    using U32x4 = std::array<std::uint32_t, 4>;

    // ---------------------------------------------------------------------------------------------
    // The source data. slot(k, c) = 4 * m_c(k) + c, where m_c is a bijection on the 64 texels of an image:
    // every (texel, component) pair gets its own value 0..255, and each component is a ramp that runs in its own
    // direction, so the images are smooth enough to be read by eye.
    // ---------------------------------------------------------------------------------------------
    unsigned ramp_slot(unsigned k, unsigned c, unsigned layer)
    {
        unsigned m = 0;
        switch (c)
        {
        case 0: m = k; break;
        case 1: m = 63u - k; break;
        case 2: m = (k + 21u) & 63u; break;
        default: m = (105u - k) & 63u; break;
        }
        m = (m + 16u * layer) & 63u;
        return 4u * m + c;
    }

    unsigned texel_index(int i, int j) { return static_cast<unsigned>(j * kTex + i); }

    /// RGBA8: texel (i, j) of layer / face `layer` (the 2D texture is layer 0).
    U8x4 texel_u8(int layer, int i, int j)
    {
        U8x4 t{};
        for (unsigned c = 0; c < 4; ++c) t[c] = static_cast<std::uint8_t>(ramp_slot(texel_index(i, j), c, static_cast<unsigned>(layer)));
        return t;
    }

    /// RGBA8I: the same bytes read as signed (-128..127).
    I32x4 texel_i8(int i, int j)
    {
        const U8x4 b = texel_u8(0, i, j);
        return {static_cast<std::int8_t>(b[0]), static_cast<std::int8_t>(b[1]), static_cast<std::int8_t>(b[2]), static_cast<std::int8_t>(b[3])};
    }

    /// RGBA32UI: a multiplicative hash of the slot (odd multiplier: a bijection on 32 bits), values far above 2^24.
    U32x4 texel_u32(int i, int j)
    {
        U32x4 t{};
        for (unsigned c = 0; c < 4; ++c) t[c] = 0x9E3779B1u * (ramp_slot(texel_index(i, j), c, 0) + 1u);
        return t;
    }

    /// DEPTH_COMPONENT16: 64 depth values spaced 1024/65535 apart, in a scrambled order (37 is odd: a bijection).
    std::uint16_t depth16(int layer, int i, int j)
    {
        const unsigned level = (37u * texel_index(i, j) + 11u + 17u * static_cast<unsigned>(layer)) & 63u;
        return static_cast<std::uint16_t>(level * 1024u + 512u);
    }

    // ---------------------------------------------------------------------------------------------
    // CPU model of the gather rules
    // ---------------------------------------------------------------------------------------------
    enum class Wrap
    {
        Clamp,
        Repeat,
        Mirror
    };

    int floor_mod(int a, int b) { return ((a % b) + b) % b; }

    /// Table 8.21 of ES 3.2: the value of wrap(coord) for a texture `size` texels wide.
    int wrap_index(Wrap mode, int coord, int size)
    {
        switch (mode)
        {
        case Wrap::Clamp: return std::clamp(coord, 0, size - 1);
        case Wrap::Repeat: return floor_mod(coord, size);
        case Wrap::Mirror:
        {
            const int a = floor_mod(coord, 2 * size) - size;
            return (size - 1) - (a >= 0 ? a : -(1 + a));
        }
        }
        return 0;
    }

    struct Region
    {
        int x0, y0, x1, y1;  // pixels [x0, x1) x [y0, y1)
        int w() const { return x1 - x0; }
        int h() const { return y1 - y0; }
    };
    constexpr Region kFootprints{0, 0, kTex + 1, kTex + 1};  // 9 x 9: i0 = -1 .. 7, so the edge footprints wrap
    constexpr Region kInterior{1, 1, kTex, kTex};            // 7 x 7: every footprint inside the image

    template <typename T>
    struct Image
    {
        int w = 0, h = 0;
        std::vector<std::array<T, 4>> px;
        Image() = default;
        Image(int width, int height) : w(width), h(height), px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {}
        std::array<T, 4>& at(int x, int y) { return px[static_cast<std::size_t>(y * w + x)]; }
        const std::array<T, 4>& at(int x, int y) const { return px[static_cast<std::size_t>(y * w + x)]; }
    };

    enum class Order
    {
        Spec,      // figure 8.7: i0 j1, i1 j1, i1 j0, i0 j0
        RowMajor   // the mutation: i0 j0, i1 j0, i0 j1, i1 j1
    };

    /// What textureGather / textureGatherOffset returns for every pixel of `region`. `texel(i, j)` is the (post-swizzle,
    /// converted) texel at an in-range position.
    template <typename T, typename TexelFn>
    Image<T> model_gather(const TexelFn& texel, Wrap wrap, int comp, std::array<int, 2> offset, Region region, Order order = Order::Spec)
    {
        Image<T> out(region.w(), region.h());
        for (int y = region.y0; y < region.y1; ++y)
            for (int x = region.x0; x < region.x1; ++x)
            {
                const int i0 = x - 1 + offset[0];  // the footprint of the point (x, y): texels i0, i0+1 / j0, j0+1
                const int j0 = y - 1 + offset[1];
                auto t = [&](int di, int dj) { return texel(wrap_index(wrap, i0 + di, kTex), wrap_index(wrap, j0 + dj, kTex)); };
                const auto c = static_cast<std::size_t>(comp);
                if (order == Order::Spec)
                    out.at(x - region.x0, y - region.y0) = {t(0, 1)[c], t(1, 1)[c], t(1, 0)[c], t(0, 0)[c]};
                else
                    out.at(x - region.x0, y - region.y0) = {t(0, 0)[c], t(1, 0)[c], t(0, 1)[c], t(1, 1)[c]};
            }
        return out;
    }

    /// The comparison of table 8.22: 1 where `dref <func> dt`.
    bool compare_depth(mgl::CompareFunc f, double dref, double dt)
    {
        switch (f)
        {
        case mgl::CompareFunc::Never: return false;
        case mgl::CompareFunc::Less: return dref < dt;
        case mgl::CompareFunc::Equal: return dref == dt;
        case mgl::CompareFunc::Lequal: return dref <= dt;
        case mgl::CompareFunc::Greater: return dref > dt;
        case mgl::CompareFunc::Notequal: return dref != dt;
        case mgl::CompareFunc::Gequal: return dref >= dt;
        case mgl::CompareFunc::Always: return true;
        }
        return false;
    }

    /// The texel function of a shadow gather: the comparison result as every component (0 or 255).
    auto shadow_texel(int layer, mgl::CompareFunc f, double ref)
    {
        return [layer, f, ref](int i, int j) -> U8x4 {
            const double dt = static_cast<double>(depth16(layer, i, j)) / 65535.0;
            const std::uint8_t r = compare_depth(f, ref, dt) ? 255 : 0;
            return {r, r, r, r};
        };
    }

    struct Diff
    {
        int bad = 0;
        std::string first;
    };

    template <typename T>
    Diff compare(const Image<T>& got, const Image<T>& want)
    {
        Diff d;
        if (got.w != want.w || got.h != want.h)
        {
            d.bad = 1;
            d.first = "image sizes differ";
            return d;
        }
        for (int y = 0; y < got.h; ++y)
            for (int x = 0; x < got.w; ++x)
                if (got.at(x, y) != want.at(x, y) && d.bad++ == 0)
                {
                    const auto& g = got.at(x, y);
                    const auto& w = want.at(x, y);
                    d.first = std::format("first at ({},{}): got ({}, {}, {}, {}) expected ({}, {}, {}, {})", x, y, g[0], g[1], g[2], g[3], w[0], w[1],
                                          w[2], w[3]);
                }
        return d;
    }

    template <typename T>
    bool expect_image(demo::Check& check, std::string_view what, const Image<T>& got, const Image<T>& want)
    {
        const Diff d = compare(got, want);
        return check.expect(d.bad == 0, what,
                            d.bad == 0 ? std::format("{}x{} gathers equal the model exactly", got.w, got.h)
                                       : std::format("{} of {} gathers differ, {}", d.bad, got.w * got.h, d.first));
    }

    template <typename T>
    int count_differences(const Image<T>& a, const Image<T>& b)
    {
        int n = 0;
        for (std::size_t i = 0; i < a.px.size() && i < b.px.size(); ++i) n += a.px[i] != b.px[i] ? 1 : 0;
        return n;
    }

    // ---------------------------------------------------------------------------------------------
    // Shader variants: #define lines are injected after the #version line of shaders/gather.frag.
    // ---------------------------------------------------------------------------------------------
    using Defines = std::vector<std::pair<std::string, std::string>>;

    enum class Kind  // type of the result and of the result target
    {
        Float,
        Int,
        Uint
    };
    enum class Target
    {
        Tex2D,
        Array,
        Cube
    };

    std::string with_defines(std::string source, const Defines& defines)
    {
        std::string inserted;
        for (const auto& [name, value] : defines) inserted += std::format("#define {} {}\n", name, value);
        const std::size_t end_of_version = source.find('\n');
        source.insert(end_of_version + 1, inserted);
        return source;
    }

    Defines make_defines(Kind kind, Target target, bool shadow, std::optional<std::string> comp, std::optional<std::string> offset)
    {
        Defines d;
        std::string sampler;
        if (shadow)
            sampler = target == Target::Tex2D ? "highp sampler2DShadow" : target == Target::Array ? "highp sampler2DArrayShadow" : "highp samplerCubeShadow";
        else if (kind == Kind::Float)
            sampler = target == Target::Tex2D ? "highp sampler2D" : target == Target::Array ? "highp sampler2DArray" : "highp samplerCube";
        else
            sampler = kind == Kind::Int ? "highp isampler2D" : "highp usampler2D";
        d.emplace_back("GSAMPLER", sampler);
        d.emplace_back("GVEC4", shadow || kind == Kind::Float ? "vec4" : kind == Kind::Int ? "ivec4" : "uvec4");
        if (target == Target::Array) d.emplace_back("TARGET_ARRAY", "1");
        if (target == Target::Cube) d.emplace_back("TARGET_CUBE", "1");
        if (shadow) d.emplace_back("SHADOW", "1");
        if (comp) d.emplace_back("COMP", *comp);
        if (offset) d.emplace_back("OFFSET", *offset);
        return d;
    }

    std::string offset_expr(std::array<int, 2> o) { return std::format("ivec2({}, {})", o[0], o[1]); }

    struct Program
    {
        mgl::ProgramId id{};
        mgl::UniformLocation tex{}, size{}, layer{}, ref{};
    };

    struct Bound
    {
        mgl::TextureTarget target = mgl::TextureTarget::Texture2D;
        mgl::TextureId id{};
    };

    constexpr std::array<mgl::TextureTarget, 6> kFaces = {
        mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX, mgl::TextureTarget::TextureCubeMapPositiveY,
        mgl::TextureTarget::TextureCubeMapNegativeY, mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
    constexpr std::array<const char*, 6> kFaceNames = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
    constexpr std::array<const char*, 3> kWrapNames = {"CLAMP_TO_EDGE", "REPEAT", "MIRRORED_REPEAT"};
    constexpr std::array<Wrap, 3> kWraps = {Wrap::Clamp, Wrap::Repeat, Wrap::Mirror};

    mgl::TextureWrapMode to_gl(Wrap w)
    {
        return w == Wrap::Clamp ? mgl::TextureWrapMode::ClampToEdge : w == Wrap::Repeat ? mgl::TextureWrapMode::Repeat : mgl::TextureWrapMode::MirroredRepeat;
    }

    // ---------------------------------------------------------------------------------------------
    // A window cell: where one result image is shown, and what it must look like.
    // ---------------------------------------------------------------------------------------------
    struct Cell
    {
        int x = 0, y = 0;  // lower-left pixel of the magnified block in the window
        Region region{};
        int scale_x = 7, scale_y = 7;  // integer magnification per axis (7 x 7 at 256 x 256)
        Image<std::uint8_t> expected;
    };

    class TextureGather final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kMinWindow || env.height < kMinWindow)
                throw demo::Skip(std::format("the 4 x 4 grid of result images needs a default framebuffer of at least {0}x{0} pixels", kMinWindow));

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);  // no attributes: the vertex shader generates its corners from gl_VertexID

            vertex_source_ = env.load_text("shaders/fullscreen.vert");
            fragment_source_ = env.load_text("shaders/gather.frag");
            offsets32_source_ = env.load_text("shaders/gather_offsets.frag");

            // The result targets: one per result type, 16 x 16, attached to a framebuffer each.
            const std::array<mgl::SizedInternalFormat, 3> formats = {mgl::SizedInternalFormat::Rgba8, mgl::SizedInternalFormat::Rgba32I,
                                                                     mgl::SizedInternalFormat::Rgba32UI};
            for (std::size_t k = 0; k < 3; ++k)
            {
                mgl::glGenTextures(1, &result_tex_[k]);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, result_tex_[k]);
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, formats[k], kTarget, kTarget);
                mgl::glGenFramebuffers(1, &result_fbo_[k]);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, result_fbo_[k]);
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2D, result_tex_[k], 0);
                if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                    throw demo::Failure("a result framebuffer is incomplete");
            }

            create_textures();

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDisable(mgl::Capability::Dither);  // dithering may perturb 8-bit results; every comparison below is exact
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            win_w_ = env.width;
            win_h_ = env.height;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            cells_.clear();

            const auto texel2d = [](int i, int j) { return texel_u8(0, i, j); };
            int index = 0;
            // Rows 0..2: CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT; columns: comp 0..3.
            for (std::size_t w = 0; w < kWraps.size(); ++w)
            {
                set_wrap(tex_2d_, kWraps[w]);
                for (int comp = 0; comp < 4; ++comp)
                {
                    show_cell(index++, kFootprints, Kind::Float, make_defines(Kind::Float, Target::Tex2D, false, std::to_string(comp), std::nullopt),
                              {mgl::TextureTarget::Texture2D, tex_2d_}, 0, 0.0f,
                              model_gather<std::uint8_t>(texel2d, kWraps[w], comp, {0, 0}, kFootprints));
                }
            }
            set_wrap(tex_2d_, Wrap::Clamp);

            // Row 3: array layer 1, cube face +Z, a 2D shadow texture, and a gather with an offset.
            const auto texel_layer1 = [](int i, int j) { return texel_u8(1, i, j); };
            show_cell(index++, kFootprints, Kind::Float, make_defines(Kind::Float, Target::Array, false, "0", std::nullopt),
                      {mgl::TextureTarget::Texture2DArray, tex_array_}, 1, 0.0f, model_gather<std::uint8_t>(texel_layer1, Wrap::Clamp, 0, {0, 0}, kFootprints));

            const auto texel_face4 = [](int i, int j) { return texel_u8(4, i, j); };
            show_cell(index++, kInterior, Kind::Float, make_defines(Kind::Float, Target::Cube, false, "0", std::nullopt),
                      {mgl::TextureTarget::TextureCubeMap, tex_cube_}, 4, 0.0f, model_gather<std::uint8_t>(texel_face4, Wrap::Clamp, 0, {0, 0}, kInterior));

            set_compare_func(mgl::TextureTarget::Texture2D, tex_depth_, mgl::CompareFunc::Less);
            show_cell(index++, kFootprints, Kind::Float, make_defines(Kind::Float, Target::Tex2D, true, std::nullopt, std::nullopt),
                      {mgl::TextureTarget::Texture2D, tex_depth_}, 0, 0.5f,
                      model_gather<std::uint8_t>(shadow_texel(0, mgl::CompareFunc::Less, 0.5), Wrap::Clamp, 0, {0, 0}, kFootprints));

            set_wrap(tex_2d_, Wrap::Repeat);
            show_cell(index++, kFootprints, Kind::Float, make_defines(Kind::Float, Target::Tex2D, false, "0", offset_expr({2, -3})),
                      {mgl::TextureTarget::Texture2D, tex_2d_}, 0, 0.0f, model_gather<std::uint8_t>(texel2d, Wrap::Repeat, 0, {2, -3}, kFootprints));
            set_wrap(tex_2d_, Wrap::Clamp);

            // The cells were blitted from an RGBA8 target whose alpha is the gathered w component. Make the window
            // opaque (alpha only) so that image viewers show the colors.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_window(check);
            check_limits(check);
            check_order_and_wrap(check);
            check_comp_filter_swizzle(check);
            check_integer_samplers(check);
            check_array(check);
            check_cube(check);
            check_offsets(check);
            check_shadow(check);
            check_incomplete(check);
            check_compile_rules(check);
            check_gather_offsets32(env, check);
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
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(result_fbo_.size()), result_fbo_.data());
            const std::array<mgl::TextureId, 9> all = {tex_2d_, tex_i8_, tex_u32_, tex_array_, tex_cube_, tex_depth_, tex_depth_array_, tex_depth_cube_, tex_incomplete_};
            mgl::glDeleteTextures(static_cast<GLsizei>(all.size()), all.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(result_tex_.size()), result_tex_.data());
            mgl::glDeleteVertexArrays(1, &vao_);
        }

    private:
        // ==========================================================================================
        // Textures
        // ==========================================================================================
        static void nearest(mgl::TextureTarget t)
        {
            mgl::glTexParameter(t, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(t, mgl::TextureMagFilter::Nearest);
        }

        void create_textures()
        {
            // 2D RGBA8
            std::vector<std::uint8_t> bytes;
            for (int j = 0; j < kTex; ++j)
                for (int i = 0; i < kTex; ++i)
                {
                    const U8x4 t = texel_u8(0, i, j);
                    bytes.insert(bytes.end(), t.begin(), t.end());
                }
            tex_2d_ = make_2d(mgl::TextureTarget::Texture2D, mgl::SizedInternalFormat::Rgba8);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kTex, kTex, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, bytes.data());
            nearest(mgl::TextureTarget::Texture2D);

            // 2D RGBA8I
            std::vector<std::int8_t> signed_bytes;
            for (int j = 0; j < kTex; ++j)
                for (int i = 0; i < kTex; ++i)
                {
                    const I32x4 t = texel_i8(i, j);
                    for (std::int32_t v : t) signed_bytes.push_back(static_cast<std::int8_t>(v));
                }
            tex_i8_ = make_2d(mgl::TextureTarget::Texture2D, mgl::SizedInternalFormat::Rgba8I);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kTex, kTex, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Byte, signed_bytes.data());
            nearest(mgl::TextureTarget::Texture2D);

            // 2D RGBA32UI
            std::vector<std::uint32_t> words;
            for (int j = 0; j < kTex; ++j)
                for (int i = 0; i < kTex; ++i)
                {
                    const U32x4 t = texel_u32(i, j);
                    words.insert(words.end(), t.begin(), t.end());
                }
            tex_u32_ = make_2d(mgl::TextureTarget::Texture2D, mgl::SizedInternalFormat::Rgba32UI);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kTex, kTex, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, words.data());
            nearest(mgl::TextureTarget::Texture2D);

            // 2D array RGBA8, three layers
            mgl::glGenTextures(1, &tex_array_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_array_);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 1, mgl::SizedInternalFormat::Rgba8, kTex, kTex, kLayers);
            for (int layer = 0; layer < kLayers; ++layer)
            {
                bytes.clear();
                for (int j = 0; j < kTex; ++j)
                    for (int i = 0; i < kTex; ++i)
                    {
                        const U8x4 t = texel_u8(layer, i, j);
                        bytes.insert(bytes.end(), t.begin(), t.end());
                    }
                mgl::glTexSubImage3D(mgl::TextureTarget::Texture2DArray, 0, 0, 0, layer, kTex, kTex, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                     bytes.data());
            }
            nearest(mgl::TextureTarget::Texture2DArray);
            mgl::glTexParameter(mgl::TextureTarget::Texture2DArray, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2DArray, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);

            // Cube map RGBA8, six faces (face f uses layer f of the value generator)
            mgl::glGenTextures(1, &tex_cube_);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_cube_);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, kTex, kTex);
            for (int face = 0; face < 6; ++face)
            {
                bytes.clear();
                for (int j = 0; j < kTex; ++j)
                    for (int i = 0; i < kTex; ++i)
                    {
                        const U8x4 t = texel_u8(face, i, j);
                        bytes.insert(bytes.end(), t.begin(), t.end());
                    }
                mgl::glTexSubImage2D(kFaces[static_cast<std::size_t>(face)], 0, 0, 0, kTex, kTex, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, bytes.data());
            }
            nearest(mgl::TextureTarget::TextureCubeMap);

            // Depth textures with the comparison enabled
            tex_depth_ = make_2d(mgl::TextureTarget::Texture2D, mgl::SizedInternalFormat::DepthComponent16);
            upload_depth(mgl::TextureTarget::Texture2D, 0, 0, false);
            setup_depth(mgl::TextureTarget::Texture2D);

            mgl::glGenTextures(1, &tex_depth_array_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_depth_array_);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 1, mgl::SizedInternalFormat::DepthComponent16, kTex, kTex, kLayers);
            for (int layer = 0; layer < kLayers; ++layer) upload_depth(mgl::TextureTarget::Texture2DArray, layer, layer, true);
            setup_depth(mgl::TextureTarget::Texture2DArray);

            mgl::glGenTextures(1, &tex_depth_cube_);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_depth_cube_);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::DepthComponent16, kTex, kTex);
            for (int face = 0; face < 6; ++face) upload_depth(kFaces[static_cast<std::size_t>(face)], face, 0, false);
            setup_depth(mgl::TextureTarget::TextureCubeMap);

            // An incomplete texture: only level 0 of a mutable texture with the default NEAREST_MIPMAP_LINEAR filter.
            mgl::glGenTextures(1, &tex_incomplete_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_incomplete_);
            bytes.clear();
            for (int j = 0; j < kTex; ++j)
                for (int i = 0; i < kTex; ++i)
                {
                    const U8x4 t = texel_u8(0, i, j);
                    bytes.insert(bytes.end(), t.begin(), t.end());
                }
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kTex, kTex, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, bytes.data());
        }

        mgl::TextureId make_2d(mgl::TextureTarget target, mgl::SizedInternalFormat format)
        {
            mgl::TextureId t;
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(target, t);
            mgl::glTexStorage2D(target, 1, format, kTex, kTex);
            return t;
        }

        /// Uploads the depth image of `layer` into slice `slice` (array textures) or the image of `target`.
        static void upload_depth(mgl::TextureTarget target, int layer, int slice, bool array)
        {
            std::vector<std::uint16_t> depth;
            for (int j = 0; j < kTex; ++j)
                for (int i = 0; i < kTex; ++i) depth.push_back(depth16(layer, i, j));
            if (array)
                mgl::glTexSubImage3D(target, 0, 0, 0, slice, kTex, kTex, 1, mgl::PixelFormat::DepthComponent, mgl::PixelType::UnsignedShort, depth.data());
            else
                mgl::glTexSubImage2D(target, 0, 0, 0, kTex, kTex, mgl::PixelFormat::DepthComponent, mgl::PixelType::UnsignedShort, depth.data());
        }

        static void setup_depth(mgl::TextureTarget target)
        {
            nearest(target);
            mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);  // the initial wrap mode is REPEAT
            mgl::glTexParameter(target, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(target, mgl::TextureCompareMode::CompareRefToTexture);
            mgl::glTexParameter(target, mgl::CompareFunc::Less);
        }

        static void set_compare_func(mgl::TextureTarget target, mgl::TextureId tex, mgl::CompareFunc f)
        {
            mgl::glBindTexture(target, tex);
            mgl::glTexParameter(target, f);
        }

        static void set_wrap(mgl::TextureId tex, Wrap w)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, to_gl(w));
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, to_gl(w));
        }

        // ==========================================================================================
        // Running a gather: one program per shader variant, one draw, one read-back
        // ==========================================================================================
        const Program& program(const Defines& defines)
        {
            std::string key;
            for (const auto& [n, v] : defines) key += n + "=" + v + ";";
            auto it = programs_.find(key);
            if (it != programs_.end()) return it->second;

            Program p;
            p.id = demo::build_program_from_source(vertex_source_, with_defines(fragment_source_, defines), "gather.frag [" + key + "]");
            p.tex = mgl::glGetUniformLocation(p.id, "u_tex");
            p.size = mgl::glGetUniformLocation(p.id, "u_size");
            p.layer = mgl::glGetUniformLocation(p.id, "u_layer");
            p.ref = mgl::glGetUniformLocation(p.id, "u_ref");
            return programs_.emplace(key, p).first->second;
        }

        /// Clears the result target of `kind` to a sentinel (so that a draw that wrote nothing cannot look right), then draws
        /// the gather over `region`.
        void draw(const Program& p, Kind kind, Bound texture, int layer, float ref, Region region)
        {
            const std::size_t k = static_cast<std::size_t>(kind);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, result_fbo_[k]);
            if (kind == Kind::Float)
            {
                const std::array<GLfloat, 4> sentinel = {0.30f, 0.60f, 0.90f, 0.50f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, sentinel.data());
            }
            else if (kind == Kind::Int)
            {
                const std::array<GLint, 4> sentinel = {-7777, 7777, -7777, 7777};
                mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 0, sentinel.data());
            }
            else
            {
                const std::array<GLuint, 4> sentinel = {0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
                mgl::glClearBufferuiv(mgl::UnsignedIntegerClearBuffer::Color, 0, sentinel.data());
            }
            mgl::glViewport(region.x0, region.y0, region.w(), region.h());
            mgl::glUseProgram(p.id);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(texture.target, texture.id);
            mgl::glUniform1i(p.tex, 0);
            mgl::glUniform1f(p.size, static_cast<float>(kTex));
            mgl::glUniform1i(p.layer, layer);
            mgl::glUniform1f(p.ref, ref);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        Image<std::uint8_t> gather_f(const Defines& defines, Bound texture, int layer, float ref, Region region)
        {
            draw(program(defines), Kind::Float, texture, layer, ref, region);
            std::vector<std::uint8_t> raw(static_cast<std::size_t>(region.w() * region.h() * 4));
            mgl::glReadPixels(region.x0, region.y0, region.w(), region.h(), mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, raw.data());
            Image<std::uint8_t> im(region.w(), region.h());
            for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = {raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]};
            return im;
        }

        Image<std::int32_t> gather_i(const Defines& defines, Bound texture, int layer, Region region)
        {
            draw(program(defines), Kind::Int, texture, layer, 0.0f, region);
            std::vector<std::int32_t> raw(static_cast<std::size_t>(region.w() * region.h() * 4));
            mgl::glReadPixels(region.x0, region.y0, region.w(), region.h(), mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int, raw.data());
            Image<std::int32_t> im(region.w(), region.h());
            for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = {raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]};
            return im;
        }

        Image<std::uint32_t> gather_u(const Defines& defines, Bound texture, int layer, Region region)
        {
            draw(program(defines), Kind::Uint, texture, layer, 0.0f, region);
            std::vector<std::uint32_t> raw(static_cast<std::size_t>(region.w() * region.h() * 4));
            mgl::glReadPixels(region.x0, region.y0, region.w(), region.h(), mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, raw.data());
            Image<std::uint32_t> im(region.w(), region.h());
            for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = {raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]};
            return im;
        }

        /// render(): draw one float gather and blit its result into the window cell `index` (row 0 = top).
        void show_cell(int index, Region region, Kind kind, const Defines& defines, Bound texture, int layer, float ref, Image<std::uint8_t> expected)
        {
            draw(program(defines), kind, texture, layer, ref, region);
            // Cell (col, row) spans [col * w / 4, (col + 1) * w / 4) x [(3 - row) * h / 4, (4 - row) * h / 4): the cells tile the window.
            // The result image is magnified by the largest integer factor per axis that leaves at least one pixel of gap, and centred.
            const int col = index % kGrid, row = index / kGrid;
            const int cx0 = col * win_w_ / kGrid, cx1 = (col + 1) * win_w_ / kGrid;
            const int cy0 = (kGrid - 1 - row) * win_h_ / kGrid, cy1 = (kGrid - row) * win_h_ / kGrid;
            Cell cell;
            cell.region = region;
            cell.scale_x = (cx1 - cx0 - 1) / region.w();
            cell.scale_y = (cy1 - cy0 - 1) / region.h();
            cell.x = cx0 + (cx1 - cx0 - cell.scale_x * region.w()) / 2;
            cell.y = cy0 + (cy1 - cy0 - cell.scale_y * region.h()) / 2;
            cell.expected = std::move(expected);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, result_fbo_[0]);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            mgl::glBlitFramebuffer(region.x0, region.y0, region.x1, region.y1, cell.x, cell.y, cell.x + cell.scale_x * region.w(), cell.y + cell.scale_y * region.h(),
                                   mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            cells_.push_back(std::move(cell));
        }

        // ==========================================================================================
        // Checks
        // ==========================================================================================
        void check_window(demo::Check& check)
        {
            // The window cells were blitted by render(): every block of every cell must hold the gathered value (alpha 255).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            const std::vector<std::uint8_t> window = demo::read_rgba8(0, 0, win_w_, win_h_);
            int bad = 0;
            int total = 0;
            std::string first;
            for (std::size_t c = 0; c < cells_.size(); ++c)
            {
                const Cell& cell = cells_[c];
                for (int y = 0; y < cell.region.h(); ++y)
                    for (int x = 0; x < cell.region.w(); ++x)
                    {
                        ++total;
                        const int px = cell.x + x * cell.scale_x + cell.scale_x / 2;
                        const int py = cell.y + y * cell.scale_y + cell.scale_y / 2;
                        const std::size_t at = (static_cast<std::size_t>(py) * static_cast<std::size_t>(win_w_) + static_cast<std::size_t>(px)) * 4;
                        const auto& want = cell.expected.at(x, y);
                        const bool ok = window[at] == want[0] && window[at + 1] == want[1] && window[at + 2] == want[2] && window[at + 3] == 255;
                        if (!ok && bad++ == 0)
                            first = std::format("cell {} block ({},{}): got {} expected ({}, {}, {}, 255)", c, x, y,
                                                demo::format_rgba({window[at], window[at + 1], window[at + 2], window[at + 3]}), want[0], want[1], want[2]);
                    }
            }
            check.expect(bad == 0 && cells_.size() == 16, "the window shows 16 cells, each block the gathered RGB of its footprint (alpha made opaque)",
                         bad == 0 ? std::format("{} blocks compared", total) : std::format("{} of {} blocks differ, {}", bad, total, first));
        }

        void check_limits(demo::Check& check)
        {
            const int gather_min = demo::get_integer(mgl::GetParameter::MinProgramTextureGatherOffset);
            const int gather_max = demo::get_integer(mgl::GetParameter::MaxProgramTextureGatherOffset);
            const int texel_min = demo::get_integer(mgl::GetParameter::MinProgramTexelOffset);
            const int texel_max = demo::get_integer(mgl::GetParameter::MaxProgramTexelOffset);
            gather_min_ = gather_min;
            gather_max_ = gather_max;
            // ES 3.2 section 8.14.2: MIN_..._GATHER_OFFSET must be <= MIN_PROGRAM_TEXEL_OFFSET and MAX_... must be >=
            // MAX_PROGRAM_TEXEL_OFFSET; chapter 21 lists MIN_PROGRAM_TEXEL_OFFSET <= -8 and MAX_PROGRAM_TEXEL_OFFSET >= 7.
            check.expect(gather_min <= texel_min && gather_max >= texel_max,
                         "MIN/MAX_PROGRAM_TEXTURE_GATHER_OFFSET are at least as wide as MIN/MAX_PROGRAM_TEXEL_OFFSET (section 8.14.2)",
                         std::format("gather [{}, {}], texel [{}, {}]", gather_min, gather_max, texel_min, texel_max));
            check.expect(texel_min <= -8 && texel_max >= 7 && gather_min <= -8 && gather_max >= 7,
                         "the gather offset range includes the guaranteed [-8, 7]", std::format("gather [{}, {}]", gather_min, gather_max));
            check.expect_no_gl_error("no error from the limit queries");
        }

        /// The order of figure 8.7, once from four independently written lookups and once as the whole 9 x 9 image per wrap mode.
        void check_order_and_wrap(demo::Check& check)
        {
            const auto texel2d = [](int i, int j) { return texel_u8(0, i, j); };
            const Bound tex{mgl::TextureTarget::Texture2D, tex_2d_};

            for (std::size_t w = 0; w < kWraps.size(); ++w)
            {
                set_wrap(tex_2d_, kWraps[w]);
                for (int comp = 0; comp < 4; ++comp)
                {
                    const Image<std::uint8_t> got =
                        gather_f(make_defines(Kind::Float, Target::Tex2D, false, std::to_string(comp), std::nullopt), tex, 0, 0.0f, kFootprints);
                    expect_image(check, std::format("textureGather(sampler2D, P, {}) with {}: all 81 footprints incl. the ones past the edge", comp, kWrapNames[w]),
                                 got, model_gather<std::uint8_t>(texel2d, kWraps[w], comp, {0, 0}, kFootprints));
                    if (comp == 0 && w == 0) observed_clamp0_ = got;
                    if (comp == 0 && w == 1) observed_repeat0_ = got;

                    if (w == 0)  // figure 8.7 from the pixel (6, 6): i0 = 5, j0 = 5 -> X = (5,6) Y = (6,6) Z = (6,5) W = (5,5)
                    {
                        const auto c = static_cast<std::size_t>(comp);
                        const std::array<std::uint8_t, 4> figure = {texel_u8(0, 5, 6)[c], texel_u8(0, 6, 6)[c], texel_u8(0, 6, 5)[c], texel_u8(0, 5, 5)[c]};
                        check.expect(got.at(6, 6) == figure,
                                     std::format("figure 8.7: the footprint of the point (6, 6) returns (texel (5,6), (6,6), (6,5), (5,5)) of component {} in x, y, z, w", comp),
                                     std::format("got ({}, {}, {}, {}) expected ({}, {}, {}, {})", got.at(6, 6)[0], got.at(6, 6)[1], got.at(6, 6)[2], got.at(6, 6)[3],
                                                 figure[0], figure[1], figure[2], figure[3]));
                    }
                }
            }
            set_wrap(tex_2d_, Wrap::Clamp);
        }

        void check_comp_filter_swizzle(demo::Check& check)
        {
            const auto texel2d = [](int i, int j) { return texel_u8(0, i, j); };
            const Bound tex{mgl::TextureTarget::Texture2D, tex_2d_};
            set_wrap(tex_2d_, Wrap::Repeat);

            // comp omitted = component 0
            expect_image(check, "textureGather(sampler2D, P) without comp selects the x component (REPEAT)",
                         gather_f(make_defines(Kind::Float, Target::Tex2D, false, std::nullopt, std::nullopt), tex, 0, 0.0f, kFootprints),
                         model_gather<std::uint8_t>(texel2d, Wrap::Repeat, 0, {0, 0}, kFootprints));
            // comp is a constant integer expression, not only a literal
            expect_image(check, "comp may be a constant integer expression: (1 + 2) selects the w component",
                         gather_f(make_defines(Kind::Float, Target::Tex2D, false, "(1 + 2)", std::nullopt), tex, 0, 0.0f, kFootprints),
                         model_gather<std::uint8_t>(texel2d, Wrap::Repeat, 3, {0, 0}, kFootprints));

            // The filters are ignored: LINEAR / NEAREST_MIPMAP_NEAREST min filters and a LINEAR mag filter give the same result.
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_2d_);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Linear);
            expect_image(check, "gather ignores the filters: MIN_FILTER = MAG_FILTER = LINEAR returns the same texels, not blends",
                         gather_f(make_defines(Kind::Float, Target::Tex2D, false, "1", std::nullopt), tex, 0, 0.0f, kFootprints),
                         model_gather<std::uint8_t>(texel2d, Wrap::Repeat, 1, {0, 0}, kFootprints));
            nearest(mgl::TextureTarget::Texture2D);

            // The swizzle is applied before the component is selected: (R, G, B, A) <- (A, ONE, ZERO, R).
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleR, mgl::TextureSwizzle::Alpha);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleG, mgl::TextureSwizzle::One);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleB, mgl::TextureSwizzle::Zero);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleA, mgl::TextureSwizzle::Red);
            const auto swizzled = [&](int i, int j) {
                const U8x4 t = texel_u8(0, i, j);
                return U8x4{t[3], 255, 0, t[0]};
            };
            for (int comp = 0; comp < 4; ++comp)
                expect_image(check,
                             std::format("gather applies the texture swizzle (A, ONE, ZERO, R) before selecting comp {}: {}", comp,
                                         comp == 0 ? "the alpha texels" : comp == 1 ? "all ones (255)" : comp == 2 ? "all zeros" : "the red texels"),
                             gather_f(make_defines(Kind::Float, Target::Tex2D, false, std::to_string(comp), std::nullopt), tex, 0, 0.0f, kFootprints),
                             model_gather<std::uint8_t>(swizzled, Wrap::Repeat, comp, {0, 0}, kFootprints));
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleR, mgl::TextureSwizzle::Red);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleG, mgl::TextureSwizzle::Green);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleB, mgl::TextureSwizzle::Blue);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureSwizzleParameter::SwizzleA, mgl::TextureSwizzle::Alpha);
            set_wrap(tex_2d_, Wrap::Clamp);
        }

        void check_integer_samplers(demo::Check& check)
        {
            // isampler2D on RGBA8I: values -128..127; the 8-bit signed texels come back as exact ints.
            set_wrap(tex_i8_, Wrap::Clamp);
            for (int comp = 0; comp < 4; ++comp)
                expect_image(check, std::format("textureGather(isampler2D, P, {}) on RGBA8I with CLAMP_TO_EDGE returns the signed texels (ivec4)", comp),
                             gather_i(make_defines(Kind::Int, Target::Tex2D, false, std::to_string(comp), std::nullopt), {mgl::TextureTarget::Texture2D, tex_i8_}, 0, kFootprints),
                             model_gather<std::int32_t>([](int i, int j) { return texel_i8(i, j); }, Wrap::Clamp, comp, {0, 0}, kFootprints));
            // usampler2D on RGBA32UI: values above 2^24 would be damaged by a float path.
            set_wrap(tex_u32_, Wrap::Repeat);
            for (int comp = 0; comp < 4; ++comp)
                expect_image(check, std::format("textureGather(usampler2D, P, {}) on RGBA32UI with REPEAT returns the 32-bit texels (uvec4)", comp),
                             gather_u(make_defines(Kind::Uint, Target::Tex2D, false, std::to_string(comp), std::nullopt), {mgl::TextureTarget::Texture2D, tex_u32_}, 0, kFootprints),
                             model_gather<std::uint32_t>([](int i, int j) { return texel_u32(i, j); }, Wrap::Repeat, comp, {0, 0}, kFootprints));
        }

        void check_array(demo::Check& check)
        {
            const Bound tex{mgl::TextureTarget::Texture2DArray, tex_array_};
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_array_);
            mgl::glTexParameter(mgl::TextureTarget::Texture2DArray, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::Repeat);
            mgl::glTexParameter(mgl::TextureTarget::Texture2DArray, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::Repeat);
            const auto layer_texel = [](int layer) { return [layer](int i, int j) { return texel_u8(layer, i, j); }; };
            for (int layer = 0; layer < kLayers; ++layer)
                for (int comp : {1, 3})
                    expect_image(check, std::format("textureGather(sampler2DArray, P, {}) on layer {} (REPEAT in s and t)", comp, layer),
                                 gather_f(make_defines(Kind::Float, Target::Array, false, std::to_string(comp), std::nullopt), tex, layer, 0.0f, kFootprints),
                                 model_gather<std::uint8_t>(layer_texel(layer), Wrap::Repeat, comp, {0, 0}, kFootprints));
            // The layer coordinate is clamped to [0, layers-1] (section 8.14.2: l = clamp(floor(r + 0.5), 0, d - 1)).
            expect_image(check, "array layer coordinate 7 is clamped to the last layer (2)",
                         gather_f(make_defines(Kind::Float, Target::Array, false, "0", std::nullopt), tex, 7, 0.0f, kFootprints),
                         model_gather<std::uint8_t>(layer_texel(kLayers - 1), Wrap::Repeat, 0, {0, 0}, kFootprints));
            expect_image(check, "array layer coordinate -3 is clamped to layer 0",
                         gather_f(make_defines(Kind::Float, Target::Array, false, "0", std::nullopt), tex, -3, 0.0f, kFootprints),
                         model_gather<std::uint8_t>(layer_texel(0), Wrap::Repeat, 0, {0, 0}, kFootprints));
        }

        void check_cube(demo::Check& check)
        {
            // All six faces, all four components, on the footprints that lie inside a face (the others reach the neighbour
            // faces, where seamless filtering applies; section 8.13.1).
            const Bound tex{mgl::TextureTarget::TextureCubeMap, tex_cube_};
            int images = 0, bad_images = 0;
            std::string first;
            for (int face = 0; face < 6; ++face)
                for (int comp = 0; comp < 4; ++comp)
                {
                    const Image<std::uint8_t> got =
                        gather_f(make_defines(Kind::Float, Target::Cube, false, std::to_string(comp), std::nullopt), tex, face, 0.0f, kInterior);
                    const Image<std::uint8_t> want =
                        model_gather<std::uint8_t>([face](int i, int j) { return texel_u8(face, i, j); }, Wrap::Clamp, comp, {0, 0}, kInterior);
                    const Diff d = compare(got, want);
                    ++images;
                    if (d.bad != 0 && bad_images++ == 0) first = std::format("face {} comp {}: {}", kFaceNames[static_cast<std::size_t>(face)], comp, d.first);
                }
            check.expect(bad_images == 0, "textureGather(samplerCube, direction, comp): 6 faces x 4 components, every footprint inside the face (7 x 7 each)",
                         bad_images == 0 ? std::format("{} images of 49 gathers equal the model exactly", images) : std::format("{} of {} images differ, {}", bad_images, images, first));
        }

        void check_offsets(demo::Check& check)
        {
            const auto texel2d = [](int i, int j) { return texel_u8(0, i, j); };
            const Bound tex{mgl::TextureTarget::Texture2D, tex_2d_};
            const int lo = gather_min_, hi = gather_max_;
            // Constant offsets: small ones, mixed signs, and the extremes of the queried range (they exceed the 8-texel
            // image, so wrapping is part of the test).
            const std::vector<std::array<int, 2>> offsets = {{0, 0}, {1, 0}, {0, 1}, {-1, -1}, {2, -3}, {lo, lo}, {hi, hi}, {lo, hi}, {hi, lo}};
            for (const Wrap w : {Wrap::Clamp, Wrap::Repeat})
            {
                set_wrap(tex_2d_, w);
                for (const auto& o : offsets)
                    expect_image(check,
                                 std::format("textureGatherOffset(sampler2D, P, ivec2({}, {}), 2) with {}", o[0], o[1], kWrapNames[w == Wrap::Clamp ? 0 : 1]),
                                 gather_f(make_defines(Kind::Float, Target::Tex2D, false, "2", offset_expr(o)), tex, 0, 0.0f, kFootprints),
                                 model_gather<std::uint8_t>(texel2d, w, 2, o, kFootprints));
            }
            // The other components, and the form without comp.
            set_wrap(tex_2d_, Wrap::Repeat);
            for (int comp : {0, 1, 3})
                expect_image(check, std::format("textureGatherOffset(sampler2D, P, ivec2(3, -2), {}) with REPEAT", comp),
                             gather_f(make_defines(Kind::Float, Target::Tex2D, false, std::to_string(comp), offset_expr({3, -2})), tex, 0, 0.0f, kFootprints),
                             model_gather<std::uint8_t>(texel2d, Wrap::Repeat, comp, {3, -2}, kFootprints));
            expect_image(check, "textureGatherOffset(sampler2D, P, ivec2(-2, 1)) without comp selects x",
                         gather_f(make_defines(Kind::Float, Target::Tex2D, false, std::nullopt, offset_expr({-2, 1})), tex, 0, 0.0f, kFootprints),
                         model_gather<std::uint8_t>(texel2d, Wrap::Repeat, 0, {-2, 1}, kFootprints));
            set_wrap(tex_2d_, Wrap::Clamp);

            // The offset does not apply to the array layer.
            const Bound array{mgl::TextureTarget::Texture2DArray, tex_array_};
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_array_);
            mgl::glTexParameter(mgl::TextureTarget::Texture2DArray, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::Repeat);
            mgl::glTexParameter(mgl::TextureTarget::Texture2DArray, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::Repeat);
            expect_image(check, "textureGatherOffset(sampler2DArray, P, ivec2(1, 2), 0) on layer 2: the offset moves the footprint, not the layer",
                         gather_f(make_defines(Kind::Float, Target::Array, false, "0", offset_expr({1, 2})), array, 2, 0.0f, kFootprints),
                         model_gather<std::uint8_t>([](int i, int j) { return texel_u8(2, i, j); }, Wrap::Repeat, 0, {1, 2}, kFootprints));

            // An integer sampler with an offset.
            set_wrap(tex_i8_, Wrap::Clamp);
            expect_image(check, "textureGatherOffset(isampler2D, P, ivec2(-3, 2), 3) with CLAMP_TO_EDGE",
                         gather_i(make_defines(Kind::Int, Target::Tex2D, false, "3", offset_expr({-3, 2})), {mgl::TextureTarget::Texture2D, tex_i8_}, 0, kFootprints),
                         model_gather<std::int32_t>([](int i, int j) { return texel_i8(i, j); }, Wrap::Clamp, 3, {-3, 2}, kFootprints));
        }

        void check_shadow(demo::Check& check)
        {
            struct Case
            {
                mgl::CompareFunc func;
                const char* name;
                float ref;
            };
            // The reference depths lie 1/128 or more away from every stored depth (the stored values are (n + 0.5)/64
            // up to 16-bit rounding), so the comparison cannot be decided by rounding.
            const std::array<Case, 4> cases = {{{mgl::CompareFunc::Less, "LESS", 0.5f},
                                                {mgl::CompareFunc::Gequal, "GEQUAL", 0.25f},
                                                {mgl::CompareFunc::Greater, "GREATER", 0.75f},
                                                {mgl::CompareFunc::Lequal, "LEQUAL", 0.5f}}};
            for (const Case& c : cases)
            {
                set_compare_func(mgl::TextureTarget::Texture2D, tex_depth_, c.func);
                expect_image(check, std::format("textureGather(sampler2DShadow, P, {}) with TEXTURE_COMPARE_FUNC = {} returns the four comparison results", c.ref, c.name),
                             gather_f(make_defines(Kind::Float, Target::Tex2D, true, std::nullopt, std::nullopt), {mgl::TextureTarget::Texture2D, tex_depth_}, 0, c.ref, kFootprints),
                             model_gather<std::uint8_t>(shadow_texel(0, c.func, static_cast<double>(c.ref)), Wrap::Clamp, 0, {0, 0}, kFootprints));
            }
            // Complementary comparisons give complementary results (no stored depth equals a reference).
            set_compare_func(mgl::TextureTarget::Texture2D, tex_depth_, mgl::CompareFunc::Less);
            const Image<std::uint8_t> less = gather_f(make_defines(Kind::Float, Target::Tex2D, true, std::nullopt, std::nullopt), {mgl::TextureTarget::Texture2D, tex_depth_}, 0, 0.5f, kFootprints);
            set_compare_func(mgl::TextureTarget::Texture2D, tex_depth_, mgl::CompareFunc::Gequal);
            const Image<std::uint8_t> gequal = gather_f(make_defines(Kind::Float, Target::Tex2D, true, std::nullopt, std::nullopt), {mgl::TextureTarget::Texture2D, tex_depth_}, 0, 0.5f, kFootprints);
            bool complementary = less.px.size() == gequal.px.size();
            int ones = 0;
            for (std::size_t i = 0; complementary && i < less.px.size(); ++i)
                for (std::size_t c = 0; c < 4; ++c)
                {
                    complementary = complementary && (less.px[i][c] != gequal.px[i][c]) && (less.px[i][c] == 0 || less.px[i][c] == 255);
                    ones += less.px[i][c] == 255 ? 1 : 0;
                }
            check.expect(complementary && ones > 0 && ones < static_cast<int>(less.px.size()) * 4,
                         "LESS and GEQUAL at the same reference are complementary: every gathered result is 0 or 255 and differs between them",
                         std::format("{} of {} results are 1.0 under LESS", ones, less.px.size() * 4));

            // Shadow gather on a 2D shadow sampler with an offset, array shadow and cube shadow.
            set_compare_func(mgl::TextureTarget::Texture2D, tex_depth_, mgl::CompareFunc::Less);
            expect_image(check, "textureGatherOffset(sampler2DShadow, P, 0.5, ivec2(2, -1)) (LESS)",
                         gather_f(make_defines(Kind::Float, Target::Tex2D, true, std::nullopt, offset_expr({2, -1})), {mgl::TextureTarget::Texture2D, tex_depth_}, 0, 0.5f, kFootprints),
                         model_gather<std::uint8_t>(shadow_texel(0, mgl::CompareFunc::Less, 0.5), Wrap::Clamp, 0, {2, -1}, kFootprints));

            for (int layer = 0; layer < kLayers; ++layer)
            {
                set_compare_func(mgl::TextureTarget::Texture2DArray, tex_depth_array_, mgl::CompareFunc::Less);
                expect_image(check, std::format("textureGather(sampler2DArrayShadow, P, 0.5) on layer {} (LESS)", layer),
                             gather_f(make_defines(Kind::Float, Target::Array, true, std::nullopt, std::nullopt), {mgl::TextureTarget::Texture2DArray, tex_depth_array_}, layer, 0.5f, kFootprints),
                             model_gather<std::uint8_t>(shadow_texel(layer, mgl::CompareFunc::Less, 0.5), Wrap::Clamp, 0, {0, 0}, kFootprints));
            }
            expect_image(check, "textureGatherOffset(sampler2DArrayShadow, P, 0.5, ivec2(-1, 3)) on layer 1 (LESS)",
                         gather_f(make_defines(Kind::Float, Target::Array, true, std::nullopt, offset_expr({-1, 3})), {mgl::TextureTarget::Texture2DArray, tex_depth_array_}, 1, 0.5f, kFootprints),
                         model_gather<std::uint8_t>(shadow_texel(1, mgl::CompareFunc::Less, 0.5), Wrap::Clamp, 0, {-1, 3}, kFootprints));

            set_compare_func(mgl::TextureTarget::TextureCubeMap, tex_depth_cube_, mgl::CompareFunc::Greater);
            int faces_ok = 0;
            std::string first;
            for (int face = 0; face < 6; ++face)
            {
                const Image<std::uint8_t> got = gather_f(make_defines(Kind::Float, Target::Cube, true, std::nullopt, std::nullopt), {mgl::TextureTarget::TextureCubeMap, tex_depth_cube_}, face, 0.5f, kInterior);
                const Diff d = compare(got, model_gather<std::uint8_t>(shadow_texel(face, mgl::CompareFunc::Greater, 0.5), Wrap::Clamp, 0, {0, 0}, kInterior));
                if (d.bad == 0) ++faces_ok;
                else if (first.empty()) first = std::format("face {}: {}", kFaceNames[static_cast<std::size_t>(face)], d.first);
            }
            check.expect(faces_ok == 6, "textureGather(samplerCubeShadow, direction, 0.5) on all six faces (GREATER)",
                         faces_ok == 6 ? "6 images of 49 comparison gathers equal the model" : first);
        }

        void check_incomplete(demo::Check& check)
        {
            // Section 8.14.2: an incomplete texture gathers (0, 0, 0, 1) from all four texels. The default MIN_FILTER is
            // NEAREST_MIPMAP_LINEAR, and only level 0 of this 8 x 8 texture exists.
            const Bound tex{mgl::TextureTarget::Texture2D, tex_incomplete_};
            for (int comp : {0, 3})
            {
                const Image<std::uint8_t> got = gather_f(make_defines(Kind::Float, Target::Tex2D, false, std::to_string(comp), std::nullopt), tex, 0, 0.0f, kFootprints);
                const std::uint8_t value = comp == 3 ? 255 : 0;
                Image<std::uint8_t> want(kFootprints.w(), kFootprints.h());
                for (auto& px : want.px) px = {value, value, value, value};
                const Diff d = compare(got, want);
                check.expect(d.bad == 0, std::format("an incomplete texture gathers (0, 0, 0, 1): comp {} returns {} in all four components", comp, value),
                             d.bad == 0 ? std::string{} : std::format("{} of {} gathers differ, {}", d.bad, got.w * got.h, d.first));
            }
        }

        void check_compile_rules(demo::Check& check)
        {
            // GLSL ES 3.10 section 8.9.3: comp "must be a constant integer expression"; textureGatherOffset takes its offset "as
            // described in textureOffset", which must be a constant expression (ES 3.20 relaxes that rule for textureGatherOffset).
            // u_comp is a uniform, so neither is constant.
            std::string log;
            const mgl::ShaderId bad_comp = demo::compile_shader(mgl::ShaderType::Fragment, with_defines(fragment_source_, make_defines(Kind::Float, Target::Tex2D, false, "u_comp", std::nullopt)), log);
            check.expect(bad_comp.value == 0, "a non-constant comp argument (a uniform) is a compile error in GLSL ES 3.10",
                         bad_comp.value == 0 ? "rejected: " + first_line(log) : "the shader compiled");
            if (bad_comp.value != 0) mgl::glDeleteShader(bad_comp);

            log.clear();
            const mgl::ShaderId bad_offset = demo::compile_shader(mgl::ShaderType::Fragment, with_defines(fragment_source_, make_defines(Kind::Float, Target::Tex2D, false, "0", "ivec2(u_comp, 1)")), log);
            check.expect(bad_offset.value == 0, "a non-constant textureGatherOffset offset (built from a uniform) is a compile error in GLSL ES 3.10",
                         bad_offset.value == 0 ? "rejected: " + first_line(log) : "the shader compiled");
            if (bad_offset.value != 0) mgl::glDeleteShader(bad_offset);

            log.clear();
            const mgl::ShaderId good = demo::compile_shader(mgl::ShaderType::Fragment, with_defines(fragment_source_, make_defines(Kind::Float, Target::Tex2D, false, "(3 - 1)", offset_expr({1, 1}))), log);
            check.expect(good.value != 0, "the same call with a constant-expression comp and a constant offset compiles", good.value != 0 ? std::string{} : log);
            if (good.value != 0) mgl::glDeleteShader(good);
        }

        static std::string first_line(const std::string& log)
        {
            const std::size_t end = log.find('\n');
            return end == std::string::npos ? log : log.substr(0, end);
        }

        void check_gather_offsets32(demo::Env& env, demo::Check& check)
        {
            if (env.context_version < demo::Version{3, 2})
            {
                std::cout << std::format("  note: textureGatherOffsets is GLSL ES 3.20; the context reports ES {}, so that section is skipped\n", demo::to_string(env.context_version));
                return;
            }
            // The vertex shader is the same text with the version of the fragment shader (all shaders of a program share one).
            std::string vertex32 = vertex_source_;
            vertex32.replace(vertex32.find("310"), 3, "320");
            Program p;
            p.id = demo::build_program_from_source(vertex32, offsets32_source_, "gather_offsets.frag (#version 320 es)");
            p.tex = mgl::glGetUniformLocation(p.id, "u_tex");
            p.size = mgl::glGetUniformLocation(p.id, "u_size");
            programs_.emplace("gather_offsets32", p);  // deleted in shutdown()

            // The four offsets written in the shader. Result texel k = texel (i0, j0) of the footprint moved by offsets[k]:
            // the four result texels no longer come from one 2 x 2 block.
            constexpr std::array<std::array<int, 2>, 4> offsets = {{{-1, 0}, {3, 1}, {0, -2}, {2, 2}}};
            set_wrap(tex_2d_, Wrap::Repeat);
            draw(p, Kind::Float, {mgl::TextureTarget::Texture2D, tex_2d_}, 0, 0.0f, kFootprints);
            std::vector<std::uint8_t> raw(static_cast<std::size_t>(kFootprints.w() * kFootprints.h() * 4));
            mgl::glReadPixels(0, 0, kFootprints.w(), kFootprints.h(), mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, raw.data());
            Image<std::uint8_t> got(kFootprints.w(), kFootprints.h()), want(kFootprints.w(), kFootprints.h());
            for (std::size_t i = 0; i < got.px.size(); ++i) got.px[i] = {raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]};
            for (int y = 0; y < kFootprints.h(); ++y)
                for (int x = 0; x < kFootprints.w(); ++x)
                    for (std::size_t k = 0; k < 4; ++k)
                        want.at(x, y)[k] = texel_u8(0, wrap_index(Wrap::Repeat, x - 1 + offsets[k][0], kTex), wrap_index(Wrap::Repeat, y - 1 + offsets[k][1], kTex))[2];
            expect_image(check, "ES 3.2 only: textureGatherOffsets(sampler2D, P, offsets[4], 2): each result texel comes from its own offset footprint (REPEAT)", got, want);
            set_wrap(tex_2d_, Wrap::Clamp);
        }

        void check_negative_controls(demo::Check& check)
        {
            // Mutation 1: a model that returns the row-major order (i0 j0, i1 j0, i0 j1, i1 j1) instead of figure 8.7's
            // (i0 j1, i1 j1, i1 j0, i0 j0) must NOT match what the hardware returned.
            const auto texel2d = [](int i, int j) { return texel_u8(0, i, j); };
            const Image<std::uint8_t> wrong = model_gather<std::uint8_t>(texel2d, Wrap::Clamp, 0, {0, 0}, kFootprints, Order::RowMajor);
            const int order_diff = count_differences(observed_clamp0_, wrong);
            check.expect(order_diff > 70, "negative control: the row-major order (a mutated model) disagrees with the observed gathers in almost every pixel",
                         std::format("{} of {} pixels differ", order_diff, observed_clamp0_.px.size()));
            // Mutation 2: the wrap mode matters. Only the edge footprints can differ between CLAMP_TO_EDGE and REPEAT.
            const int wrap_diff = count_differences(observed_clamp0_, observed_repeat0_);
            check.expect(wrap_diff == 32, "negative control: CLAMP_TO_EDGE and REPEAT results differ in exactly the 32 edge footprints (and nowhere else)",
                         std::format("{} of 81 pixels differ (the 9 x 9 footprints of an 8 x 8 image have 9*9 - 7*7 = 32 edge ones)", wrap_diff));
        }

        // ==========================================================================================
        // State
        // ==========================================================================================
        static constexpr int kLayers = 3;

        mgl::VertexArrayId vao_{};
        std::array<mgl::TextureId, 3> result_tex_{};
        std::array<mgl::FramebufferId, 3> result_fbo_{};
        mgl::TextureId tex_2d_{}, tex_i8_{}, tex_u32_{}, tex_array_{}, tex_cube_{}, tex_depth_{}, tex_depth_array_{}, tex_depth_cube_{}, tex_incomplete_{};
        std::map<std::string, Program> programs_;
        std::string vertex_source_, fragment_source_, offsets32_source_;
        std::vector<Cell> cells_;
        int win_w_ = 0, win_h_ = 0;  // surface size of the last render()
        int gather_min_ = -8, gather_max_ = 7;
        Image<std::uint8_t> observed_clamp0_, observed_repeat0_;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es31-texture-gather",
        .title = "textureGather and textureGatherOffset",
        .min_es = {3, 1},
        .summary = "textureGather on 2D, 2D array and cube textures with component selection, textureGatherOffset up to the queried limits, "
                   "shadow and integer samplers, checked texel by texel against figure 8.7 of the ES 3.2 specification.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<TextureGather>(argc, argv, info);
}
