// es32-cube-map-array
//
// Cube map array textures (OpenGL ES 3.2 sections 8.5, 8.13, 8.14.3, 8.23 and Table 8.24): an array of N complete
// cube maps stored as one texture of 6 N "layer-faces". Layer-face z belongs to cube layer = z / 6 and face = z mod 6,
// the face order being the order of the cube map face targets (+X -X +Y -Y +Z -Z, Table 8.24). The shader looks the
// texture up with ONE four-component coordinate: the direction (s, t, r) selects face and texel like a cube map
// (Table 8.20), the fourth component q selects the cube.
//
// Every texel of the main texture (5 cubes x 6 faces x 4 x 4 texels, 3 mip levels) has its own colour:
//   R = 24 + 36 * face       G = 24 + 40 * layer       B = 16 + 14 * texel index + 8 * level
// so a lookup that returns the wrong face, layer, texel or level shows up as a different colour, and the colour
// tells which one it was.
//
// Image (256 x 256; other sizes: see Layout / layout_of()):
//   * top 181 rows: five spheres drawn with ONE glDrawElementsInstanced call, a perspective camera and a depth buffer.
//     Sphere k reflects layer k of the array (environment mapping: texture(reflect(view, normal), k)).
//   * bottom 75 rows: 30 tiles, one per layer-face z = 6 layer + face in the order of Table 8.24. Each tile shows the
//     4 x 4 image of that layer-face (a direction on the plane of the face, plain texture() lookup).
//
// validate() compares
//   * every texel of every level, layer and face, read through the framebuffer (glFramebufferTextureLayer with the
//     layer-face number), through imageLoad (layered imageCubeArray and single layer-face image2D bindings) and through
//     samplerCubeArray lookups of the texel centres, with the CPU model of Tables 8.20 / 8.24
//   * 2400 further lookups (random directions, layer coordinates including fractions and out-of-range values, integer
//     and fractional levels of detail) with the CPU model
//   * isamplerCubeArray, usamplerCubeArray and samplerCubeArrayShadow (all 8 comparison functions of Table 8.22)
//   * textureSize of every level (the third component is the number of cubes), the level queries, the errors of
//     glTexStorage3D / glTexImage3D / glTexSubImage3D, completeness and glGenerateMipmap of cube map arrays
//   * GLSL ES 3.20 compile-time rules (no texelFetch for cube maps, precision qualifiers, coordinate forms)
//   * the picture: tiles and spheres against a ray-sphere / reflection model

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
namespace dm = demo::math;

namespace
{
    using Bytes = std::vector<std::uint8_t>;
    using Words = std::array<std::uint32_t, 4>;

    constexpr int kFaces = 6;
    constexpr int kSize = 4;     // level 0 faces are 4 x 4 texels
    constexpr int kLevels = 3;   // 4x4, 2x2, 1x1
    constexpr int kLayers = 5;   // cubes in the main array: 30 layer-faces
    constexpr int kSmall = 2;    // integer and depth arrays: 2 x 2 faces, one level
    constexpr int kSmallLayers = 3;
    constexpr int kTilesPerRow = 10;
    constexpr int kTileRows = 3;

    constexpr std::array<const char*, kFaces> kFaceName = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

    // ---------------------------------------------------------------------------------------------
    // The data: one distinct value per (level, layer, face, texel)
    // ---------------------------------------------------------------------------------------------
    int level_size(int level) { return kSize >> level; }

    Rgba8 texel_color(int level, int layer, int face, int i, int j)
    {
        const int idx = j * level_size(level) + i;
        return {static_cast<std::uint8_t>(24 + 36 * face), static_cast<std::uint8_t>(24 + 40 * layer),
                static_cast<std::uint8_t>(16 + 14 * idx + 8 * level), 255};
    }

    /// RGBA8 image of one layer-face of `level`, rows in upload order (j = 0 first).
    Bytes face_image(int level, int layer, int face)
    {
        Bytes out;
        for (int j = 0; j < level_size(level); ++j)
            for (int i = 0; i < level_size(level); ++i)
            {
                const Rgba8 c = texel_color(level, layer, face, i, j);
                out.insert(out.end(), c.begin(), c.end());
            }
        return out;
    }

    /// RGBA8UI texel of the unsigned integer array: (layer, face, texel index, 100 + 8 z)
    std::array<std::uint8_t, 4> uint_texel(int layer, int face, int i, int j)
    {
        return {static_cast<std::uint8_t>(layer), static_cast<std::uint8_t>(face), static_cast<std::uint8_t>(2 * j + i), static_cast<std::uint8_t>(100 + 8 * (6 * layer + face))};
    }
    /// RGBA16I texel of the signed integer array; negative values and distinct components.
    std::array<std::int16_t, 4> int_texel(int layer, int face, int i, int j)
    {
        return {static_cast<std::int16_t>(-300 - 100 * layer), static_cast<std::int16_t>(face - 3), static_cast<std::int16_t>(i * 1000 - 500),
                static_cast<std::int16_t>(j * -777 + 6 * layer + face)};
    }
    /// DEPTH_COMPONENT32F texel: n / 128 with n = 1 + 4 z + texel index; exactly representable, all different.
    float depth_texel(int layer, int face, int i, int j) { return static_cast<float>(1 + 4 * (6 * layer + face) + 2 * j + i) / 128.0f; }

    // ---------------------------------------------------------------------------------------------
    // CPU model of cube map array lookups
    // ---------------------------------------------------------------------------------------------
    struct V3
    {
        double x = 0, y = 0, z = 0;
    };
    V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    V3 operator*(V3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
    double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    V3 normalize(V3 a) { return a * (1.0 / std::sqrt(dot(a, a))); }
    V3 as_float(V3 a) { return {static_cast<double>(static_cast<float>(a.x)), static_cast<double>(static_cast<float>(a.y)), static_cast<double>(static_cast<float>(a.z))}; }

    struct Sample
    {
        int level = 0, layer = 0, face = 0, i = 0, j = 0;
        bool ambiguous = false;
        bool operator==(const Sample& o) const { return level == o.level && layer == o.layer && face == o.face && i == o.i && j == o.j; }
    };

    constexpr double kTexelMargin = 0.05;  // texel units: probes closer to a texel boundary are skipped
    constexpr double kEdgeMargin = 0.98;   // max(|sc|, |tc|) / |ma| above this: face edge, the major axis may tie
    constexpr double kTieMargin = 0.05;    // distance of a fractional level of detail / layer coordinate to .5

    /// Table 8.20 + equation 8.11 (NEAREST_MIPMAP_NEAREST) + the layer rule of GLSL ES 3.20 8.9.2 for a cube map array whose faces
    /// are base_size x base_size at level 0:
    /// face from the major axis, s = (sc / |ma| + 1) / 2, texel floor(s * size), level nearest(lambda),
    /// layer = max(0, min(d - 1, floor(q + 0.5))) with d the number of cubes.
    Sample model_lookup(V3 r, double q, double lod, int base_size, int levels, int layers)
    {
        Sample S;
        const double ax = std::abs(r.x), ay = std::abs(r.y), az = std::abs(r.z);
        double sc = 0, tc = 0, ma = 0;
        if (ax >= ay && ax >= az)
        {
            ma = ax;
            if (r.x >= 0) { S.face = 0; sc = -r.z; tc = -r.y; }
            else { S.face = 1; sc = r.z; tc = -r.y; }
        }
        else if (ay >= az)
        {
            ma = ay;
            if (r.y >= 0) { S.face = 2; sc = r.x; tc = r.z; }
            else { S.face = 3; sc = r.x; tc = -r.z; }
        }
        else
        {
            ma = az;
            if (r.z >= 0) { S.face = 4; sc = r.x; tc = -r.y; }
            else { S.face = 5; sc = -r.x; tc = -r.y; }
        }
        const double s = 0.5 * (sc / ma + 1.0), t = 0.5 * (tc / ma + 1.0);

        // level of detail (8.14.3, eq. 8.11): lambda <= 0 -> base level; otherwise nearest(lambda) = ceil(lambda + 1/2) - 1, clamped to q
        const int qmax = levels - 1;
        if (lod <= 0.0) S.level = 0;
        else if (lod > qmax + 0.5) S.level = qmax;
        else
        {
            S.level = static_cast<int>(std::ceil(lod + 0.5)) - 1;
            if (std::abs(lod - std::floor(lod) - 0.5) < kTieMargin) S.ambiguous = true;  // the "alternative" formula differs at exact ties
        }
        S.level = std::clamp(S.level, 0, qmax);

        // layer
        if (std::abs(q - std::floor(q) - 0.5) < kTieMargin) S.ambiguous = true;  // round-to-nearest-even and floor(q + 1/2) differ at ties
        S.layer = std::clamp(static_cast<int>(std::floor(q + 0.5)), 0, layers - 1);

        // texel
        const int size = std::max(1, base_size >> S.level);
        const double us = s * size, vt = t * size;
        S.i = std::clamp(static_cast<int>(std::floor(us)), 0, size - 1);
        S.j = std::clamp(static_cast<int>(std::floor(vt)), 0, size - 1);
        for (int k = 1; k < size; ++k)
            if (std::abs(us - k) < kTexelMargin || std::abs(vt - k) < kTexelMargin) S.ambiguous = true;
        if (std::max(std::abs(sc), std::abs(tc)) / ma > kEdgeMargin) S.ambiguous = true;
        return S;
    }

    /// Table 8.20 solved for r with |ma| = 1: the direction on the plane of `face` at face coordinates (s, t). Mirrors tile.frag.
    V3 face_direction(int face, double s, double t)
    {
        const double sc = 2 * s - 1, tc = 2 * t - 1;
        switch (face)
        {
        case 0: return {1, -tc, -sc};
        case 1: return {-1, -tc, sc};
        case 2: return {sc, 1, tc};
        case 3: return {sc, -1, -tc};
        case 4: return {sc, -tc, 1};
        default: return {-sc, -tc, -1};
        }
    }

    Rgba8 sample_color(const Sample& S) { return texel_color(S.level, S.layer, S.face, S.i, S.j); }

    // Deliberately wrong models: each must disagree with the GPU (negative controls).
    Sample wrong_flip_t(Sample S) { S.j = level_size(S.level) - 1 - S.j; return S; }
    Sample wrong_flip_s(Sample S) { S.i = level_size(S.level) - 1 - S.i; return S; }
    Sample wrong_transpose(Sample S) { std::swap(S.i, S.j); return S; }
    Sample wrong_face_layer_swap(Sample S) { std::swap(S.face, S.layer); return S; }   // z = 6 face + layer instead of 6 layer + face
    Sample wrong_layer_plus_one(Sample S) { S.layer = (S.layer + 1) % kLayers; return S; }
    Sample wrong_level_zero(Sample S) { S.level = 0; S.i = std::min(S.i, level_size(0) - 1); return S; }

    struct Rng
    {
        std::uint32_t state = 0x1234567u;
        std::uint32_t next()
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        }
        double uniform(double lo, double hi) { return lo + (hi - lo) * (static_cast<double>(next() >> 8) / 16777216.0); }
        int below(int n) { return static_cast<int>(next() % static_cast<std::uint32_t>(n)); }
    };

    struct Probe
    {
        V3 dir;
        double q = 0;
        double lod = 0;
        double ref = 0;
    };

    struct GpuProbe
    {
        float p[4];
        float extra[4];
    };
    static_assert(sizeof(GpuProbe) == 32);

    // ---------------------------------------------------------------------------------------------
    // The demo
    // ---------------------------------------------------------------------------------------------
    enum class Kind { Float = 0, Int = 1, Uint = 2, Shadow = 3 };

    struct LookupProgram
    {
        mgl::ProgramId lookup{};
        mgl::ProgramId sizes{};
        mgl::UniformLocation lookup_tex{}, lookup_count{}, sizes_tex{}, sizes_levels{};
    };

    struct Mismatch
    {
        int compared = 0, skipped = 0, bad = 0;
        std::string first;
    };

    // The tile strip at the bottom: kTileRows rows of `tile` pixels (at most 3/10 of the height, at most a tenth of the
    // width); its kTilesPerRow columns partition the whole width exactly (column c spans [c*w/10, (c+1)*w/10)). The scene
    // viewport takes the rest of the surface. fovy is the scene camera's vertical field of view: 45 degrees, except on a
    // portrait viewport, which keeps the 45 degree HORIZONTAL field of view so that all five spheres stay in view.
    struct Layout
    {
        int width = 0, height = 0, tile = 0, tile_h = 0, scene_h = 0;
        double fovy = 0.0;
        [[nodiscard]] int col_x(int col) const { return col * width / kTilesPerRow; }
    };
    constexpr int kMinTile = 8;     // smallest tile row height (pixels)
    constexpr int kMinScene = 64;   // smallest scene viewport height (pixels)

    class CubeMapArray final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override;
        void render(demo::Env& env, int frame) override;
        void validate(demo::Env& env, demo::Check& check) override;
        void shutdown(demo::Env& env) override;
        void on_key(demo::Env& env, char key) override;

    private:
        // ---- helpers --------------------------------------------------------------------------------------
        mgl::BufferId new_buffer()
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            buffers_.push_back(b);
            return b;
        }
        mgl::TextureId new_texture()
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            textures_.push_back(t);
            return t;
        }
        mgl::ProgramId compile_compute(const demo::Env& env, std::string_view file, const std::string& defines);
        mgl::UniformLocation locate(mgl::ProgramId program, const char* name) const;
        static Bytes map_read(mgl::BufferTarget target, mgl::BufferId buf, std::size_t offset, std::size_t size);
        static Layout layout_of(const demo::Env& env);

        std::vector<Words> run_lookups(Kind kind, mgl::TextureId tex, const std::vector<Probe>& probes);
        std::vector<std::array<int, 4>> run_sizes(Kind kind, mgl::TextureId tex, int levels);
        void build_textures();
        void bind_cube_array(mgl::TextureId t) const
        {
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMapArray, t);
        }

        // ---- validation sections -------------------------------------------------------------------------
        void check_data_is_distinct(demo::Check& check);
        void check_state_and_queries(demo::Check& check);
        void check_errors(demo::Check& check);
        void check_framebuffer_layer_faces(demo::Check& check);
        void check_image_access(demo::Check& check);
        void check_float_lookups(demo::Env& env, demo::Check& check);
        void check_integer_lookups(demo::Check& check);
        void check_shadow_lookups(demo::Check& check);
        void check_texture_size(demo::Check& check);
        void check_completeness_and_mipmaps(demo::Check& check);
        void check_compile_rules(demo::Check& check);
        void check_picture(demo::Env& env, demo::Check& check);

        // ---- rendering -------------------------------------------------------------------------------------
        [[nodiscard]] dm::Vec3 eye_position() const;
        [[nodiscard]] std::array<V3, 5> sphere_centres(double t) const;

        std::array<LookupProgram, 4> kinds_{};
        mgl::ProgramId env_program_{}, tile_program_{}, image_layered_{}, image_single_{};
        mgl::UniformLocation loc_view_proj_{}, loc_sphere_{}, loc_eye_{}, loc_env_{}, loc_tile_env_{}, loc_tile_z_{};
        mgl::UniformLocation loc_img_extent_layered_{}, loc_img_extent_single_{};

        std::vector<mgl::BufferId> buffers_;
        std::vector<mgl::TextureId> textures_;
        mgl::TextureId tex_color_{}, tex_int_{}, tex_uint_{}, tex_depth_{};
        mgl::BufferId sphere_vbo_{}, sphere_ibo_{}, probe_buf_{}, result_buf_{};
        mgl::VertexArrayId sphere_vao_{}, empty_vao_{};
        mgl::FramebufferId fbo_{};
        int sphere_index_count_ = 0;
        int initial_binding_ = -1;

        double yaw_deg_ = -6.0, pitch_deg_ = 16.0, distance_ = 9.6;
        int shown_frame_ = 0;
    };

    // =================================================================================================
    // init
    // =================================================================================================
    mgl::UniformLocation CubeMapArray::locate(mgl::ProgramId program, const char* name) const
    {
        const mgl::UniformLocation l = mgl::glGetUniformLocation(program, name);
        if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
        return l;
    }

    mgl::ProgramId CubeMapArray::compile_compute(const demo::Env& env, std::string_view file, const std::string& defines)
    {
        std::string src = env.load_text(file);
        if (!defines.empty()) src.insert(src.find('\n') + 1, defines);  // macros go right after "#version 320 es"
        std::string log;
        const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, src, log);
        if (cs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", file, log));
        const std::array<mgl::ShaderId, 1> shaders = {cs};
        const mgl::ProgramId program = demo::link_program(shaders, log);
        mgl::glDeleteShader(cs);
        if (program.value == 0) throw demo::Failure(std::format("{} failed to link:\n{}", file, log));
        return program;
    }

    void CubeMapArray::build_textures()
    {
        const mgl::TextureTarget target = mgl::TextureTarget::TextureCubeMapArray;

        // ---- the main array: RGBA8, 4 x 4, 3 levels, 5 cubes = 30 layer-faces, immutable storage ----------
        tex_color_ = new_texture();
        bind_cube_array(tex_color_);
        initial_binding_ = demo::get_integer(mgl::GetParameter::TextureBindingCubeMapArray);
        mgl::glTexStorage3D(target, kLevels, mgl::SizedInternalFormat::Rgba8, kSize, kSize, kFaces * kLayers);  // depth = 6 * layers
        // Level 0: one layer-face per call (zoffset = z, depth = 1: "zoffset is the first layer-face to update").
        for (int layer = 0; layer < kLayers; ++layer)
            for (int face = 0; face < kFaces; ++face)
            {
                const Bytes img = face_image(0, layer, face);
                mgl::glTexSubImage3D(target, 0, 0, 0, kFaces * layer + face, kSize, kSize, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, img.data());
            }
        // Level 1: one whole cube (6 layer-faces) per call.
        for (int layer = 0; layer < kLayers; ++layer)
        {
            Bytes cube;
            for (int face = 0; face < kFaces; ++face)
            {
                const Bytes img = face_image(1, layer, face);
                cube.insert(cube.end(), img.begin(), img.end());
            }
            mgl::glTexSubImage3D(target, 1, 0, 0, kFaces * layer, level_size(1), level_size(1), kFaces, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, cube.data());
        }
        // Level 2: all 30 layer-faces in one call.
        {
            Bytes all;
            for (int layer = 0; layer < kLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                {
                    const Bytes img = face_image(2, layer, face);
                    all.insert(all.end(), img.begin(), img.end());
                }
            mgl::glTexSubImage3D(target, 2, 0, 0, 0, level_size(2), level_size(2), kFaces * kLayers, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, all.data());
        }
        mgl::glTexParameter(target, mgl::TextureMinFilter::NearestMipmapNearest);
        mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);

        // ---- integer arrays: RGBA8UI and RGBA16I, 2 x 2, one level, 3 cubes (integer textures need NEAREST) -
        tex_uint_ = new_texture();
        bind_cube_array(tex_uint_);
        mgl::glTexStorage3D(target, 1, mgl::SizedInternalFormat::Rgba8UI, kSmall, kSmall, kFaces * kSmallLayers);
        {
            std::vector<std::uint8_t> data;
            for (int layer = 0; layer < kSmallLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                    for (int j = 0; j < kSmall; ++j)
                        for (int i = 0; i < kSmall; ++i)
                        {
                            const auto t = uint_texel(layer, face, i, j);
                            data.insert(data.end(), t.begin(), t.end());
                        }
            mgl::glTexSubImage3D(target, 0, 0, 0, 0, kSmall, kSmall, kFaces * kSmallLayers, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedByte, data.data());
        }
        mgl::glTexParameter(target, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);

        tex_int_ = new_texture();
        bind_cube_array(tex_int_);
        mgl::glTexStorage3D(target, 1, mgl::SizedInternalFormat::Rgba16I, kSmall, kSmall, kFaces * kSmallLayers);
        {
            std::vector<std::int16_t> data;
            for (int layer = 0; layer < kSmallLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                    for (int j = 0; j < kSmall; ++j)
                        for (int i = 0; i < kSmall; ++i)
                        {
                            const auto t = int_texel(layer, face, i, j);
                            data.insert(data.end(), t.begin(), t.end());
                        }
            mgl::glTexSubImage3D(target, 0, 0, 0, 0, kSmall, kSmall, kFaces * kSmallLayers, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Short, data.data());
        }
        mgl::glTexParameter(target, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);

        // ---- depth array for samplerCubeArrayShadow ---------------------------------------------------------
        tex_depth_ = new_texture();
        bind_cube_array(tex_depth_);
        mgl::glTexStorage3D(target, 1, mgl::SizedInternalFormat::DepthComponent32F, kSmall, kSmall, kFaces * kSmallLayers);
        {
            std::vector<float> data;
            for (int layer = 0; layer < kSmallLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                    for (int j = 0; j < kSmall; ++j)
                        for (int i = 0; i < kSmall; ++i) data.push_back(depth_texel(layer, face, i, j));
            mgl::glTexSubImage3D(target, 0, 0, 0, 0, kSmall, kSmall, kFaces * kSmallLayers, mgl::PixelFormat::DepthComponent, mgl::PixelType::Float, data.data());
        }
        mgl::glTexParameter(target, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);
        mgl::glTexParameter(target, mgl::TextureCompareMode::CompareRefToTexture);
        mgl::glTexParameter(target, mgl::CompareFunc::Lequal);
        mgl::glBindTexture(target, mgl::TextureId{0});
    }

    void CubeMapArray::init(demo::Env& env)
    {
        {
            const Layout lay = layout_of(env);
            if (lay.tile < kMinTile || lay.scene_h < kMinScene)
                throw demo::Skip(std::format("{}x{} is too small: the tile strip needs rows of at least {} pixels and the scene at least {} rows", env.width, env.height, kMinTile, kMinScene));
        }
        struct Variant
        {
            Kind kind;
            const char* defines;
            const char* sampler;
        };
        const std::array<Variant, 4> variants = {{
            {Kind::Float, "#define SAMPLER samplerCubeArray\n#define LOOKUP(t, pr) textureLod(t, pr.p, pr.extra.x)\n#define TO_WORDS(v) floatBitsToUint(v)\n", "samplerCubeArray"},
            {Kind::Int, "#define SAMPLER isamplerCubeArray\n#define LOOKUP(t, pr) textureLod(t, pr.p, pr.extra.x)\n#define TO_WORDS(v) uvec4(v)\n", "isamplerCubeArray"},
            {Kind::Uint, "#define SAMPLER usamplerCubeArray\n#define LOOKUP(t, pr) textureLod(t, pr.p, pr.extra.x)\n#define TO_WORDS(v) (v)\n", "usamplerCubeArray"},
            {Kind::Shadow, "#define SAMPLER samplerCubeArrayShadow\n#define LOOKUP(t, pr) vec4(texture(t, pr.p, pr.extra.y), 0.0, 0.0, 1.0)\n#define TO_WORDS(v) floatBitsToUint(v)\n",
             "samplerCubeArrayShadow"},
        }};
        for (const Variant& v : variants)
        {
            LookupProgram& p = kinds_[static_cast<std::size_t>(v.kind)];
            p.lookup = compile_compute(env, "shaders/lookup.comp", v.defines);
            p.lookup_tex = locate(p.lookup, "u_tex");
            p.lookup_count = locate(p.lookup, "u_count");
            p.sizes = compile_compute(env, "shaders/sizes.comp", std::format("#define SAMPLER {}\n", v.sampler));
            p.sizes_tex = locate(p.sizes, "u_tex");
            p.sizes_levels = locate(p.sizes, "u_levels");
        }
        image_layered_ = compile_compute(env, "shaders/image_cube.comp", "#define LAYERED 1\n");
        image_single_ = compile_compute(env, "shaders/image_cube.comp", "#define LAYERED 0\n");
        loc_img_extent_layered_ = locate(image_layered_, "u_extent");
        loc_img_extent_single_ = locate(image_single_, "u_extent");

        env_program_ = demo::build_program(env, "shaders/env.vert", "shaders/env.frag");
        loc_view_proj_ = locate(env_program_, "u_view_proj");
        loc_sphere_ = locate(env_program_, "u_sphere");
        loc_eye_ = locate(env_program_, "u_eye");
        loc_env_ = locate(env_program_, "u_env");
        tile_program_ = demo::build_program(env, "shaders/quad.vert", "shaders/tile.frag");
        loc_tile_env_ = locate(tile_program_, "u_env");
        loc_tile_z_ = locate(tile_program_, "u_layer_face");

        build_textures();

        // Unit sphere mesh: 64 segments x 32 rings.
        constexpr int kSeg = 64, kRing = 32;
        std::vector<float> verts;
        for (int r = 0; r <= kRing; ++r)
            for (int s = 0; s <= kSeg; ++s)
            {
                const double phi = std::numbers::pi * r / kRing, theta = 2.0 * std::numbers::pi * s / kSeg;
                verts.push_back(static_cast<float>(std::sin(phi) * std::cos(theta)));
                verts.push_back(static_cast<float>(std::cos(phi)));
                verts.push_back(static_cast<float>(std::sin(phi) * std::sin(theta)));
            }
        std::vector<std::uint16_t> indices;
        for (int r = 0; r < kRing; ++r)
            for (int s = 0; s < kSeg; ++s)
            {
                const auto a = static_cast<std::uint16_t>(r * (kSeg + 1) + s), b = static_cast<std::uint16_t>(a + kSeg + 1);
                // counter-clockwise seen from outside
                for (std::uint16_t k : {a, static_cast<std::uint16_t>(a + 1), b, static_cast<std::uint16_t>(a + 1), static_cast<std::uint16_t>(b + 1), b}) indices.push_back(k);
            }
        sphere_index_count_ = static_cast<int>(indices.size());
        sphere_vbo_ = new_buffer();
        mgl::glBindBuffer(mgl::BufferTarget::Array, sphere_vbo_);
        mgl::glBufferData(mgl::BufferTarget::Array, static_cast<GLsizeiptr>(verts.size() * sizeof(float)), verts.data(), mgl::BufferUsage::StaticDraw);
        sphere_ibo_ = new_buffer();
        mgl::glGenVertexArrays(1, &sphere_vao_);
        mgl::glBindVertexArray(sphere_vao_);
        mgl::glBindBuffer(mgl::BufferTarget::ElementArray, sphere_ibo_);
        mgl::glBufferData(mgl::BufferTarget::ElementArray, static_cast<GLsizeiptr>(indices.size() * sizeof(std::uint16_t)), indices.data(), mgl::BufferUsage::StaticDraw);
        mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
        mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, 3 * sizeof(float), demo::buffer_offset(0));
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
        mgl::glGenVertexArrays(1, &empty_vao_);

        probe_buf_ = new_buffer();
        result_buf_ = new_buffer();
        mgl::glGenFramebuffers(1, &fbo_);
    }

    // =================================================================================================
    // render
    // =================================================================================================
    Layout CubeMapArray::layout_of(const demo::Env& env)
    {
        Layout l;
        l.width = env.width;
        l.height = env.height;
        l.tile = std::min(env.width / kTilesPerRow, env.height * 3 / 10 / kTileRows);
        l.tile_h = l.tile * kTileRows;
        l.scene_h = env.height - l.tile_h;
        const double aspect = static_cast<double>(l.width) / std::max(l.scene_h, 1);
        const double tan_half = std::tan(static_cast<double>(dm::radians(45.0f)) * 0.5);
        l.fovy = 2.0 * std::atan(aspect < 1.0 ? tan_half / aspect : tan_half);
        return l;
    }

    dm::Vec3 CubeMapArray::eye_position() const
    {
        const float yaw = dm::radians(static_cast<float>(yaw_deg_ + 0.7 * shown_frame_)), pitch = dm::radians(static_cast<float>(pitch_deg_));
        const float d = static_cast<float>(distance_);
        return {d * std::cos(pitch) * std::sin(yaw), d * std::sin(pitch), d * std::cos(pitch) * std::cos(yaw)};
    }

    std::array<V3, 5> CubeMapArray::sphere_centres(double t) const
    {
        std::array<V3, 5> c{};
        for (int k = 0; k < 5; ++k) c[static_cast<std::size_t>(k)] = {1.95 * (k - 2), 0.15 * std::sin(2.0 * t + 1.3 * k), 0.0};
        return c;
    }

    void CubeMapArray::render(demo::Env& env, int frame)
    {
        shown_frame_ = frame;
        const Layout lay = layout_of(env);
        const double t = static_cast<double>(frame) / 60.0;
        mgl::glViewport(0, 0, env.width, env.height);
        mgl::glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
        mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
        bind_cube_array(tex_color_);

        // ---- the 30 layer-face tiles (z = 6 layer + face, in reading order) ----------------------------------
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glUseProgram(tile_program_);
        mgl::glUniform1i(loc_tile_env_, 0);
        mgl::glBindVertexArray(empty_vao_);
        for (int z = 0; z < kFaces * kLayers; ++z)
        {
            const int col = z % kTilesPerRow, row = z / kTilesPerRow;
            mgl::glViewport(lay.col_x(col), (kTileRows - 1 - row) * lay.tile, lay.col_x(col + 1) - lay.col_x(col), lay.tile);
            mgl::glUniform1i(loc_tile_z_, z);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        // ---- the five environment-mapped spheres, one draw call ----------------------------------------------
        mgl::glViewport(0, lay.tile_h, lay.width, lay.scene_h);
        mgl::glEnable(mgl::Capability::DepthTest);
        mgl::glUseProgram(env_program_);
        const dm::Mat4 proj = dm::perspective(static_cast<float>(lay.fovy), static_cast<float>(lay.width) / static_cast<float>(lay.scene_h), 0.5f, 60.0f);
        const dm::Vec3 eye = eye_position();
        const dm::Mat4 vp = proj * dm::look_at(eye, {0, 0, 0}, {0, 1, 0});
        mgl::glUniformMatrix4fv(loc_view_proj_, 1, GL_FALSE, vp.data());
        mgl::glUniform3f(loc_eye_, eye.x, eye.y, eye.z);
        std::array<float, 20> spheres{};
        const auto centres = sphere_centres(t);
        for (std::size_t k = 0; k < 5; ++k)
        {
            spheres[4 * k + 0] = static_cast<float>(centres[k].x);
            spheres[4 * k + 1] = static_cast<float>(centres[k].y);
            spheres[4 * k + 2] = static_cast<float>(centres[k].z);
            spheres[4 * k + 3] = 0.8f;
        }
        mgl::glUniform4fv(loc_sphere_, 5, spheres.data());
        mgl::glUniform1i(loc_env_, 0);
        mgl::glBindVertexArray(sphere_vao_);
        mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, sphere_index_count_, mgl::IndexType::UnsignedShort, demo::buffer_offset(0), 5);
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glDisable(mgl::Capability::DepthTest);
    }

    void CubeMapArray::on_key(demo::Env& /*env*/, char key)
    {
        if (key == 'a') yaw_deg_ -= 5.0;
        else if (key == 'd') yaw_deg_ += 5.0;
        else if (key == 'w') pitch_deg_ = std::min(pitch_deg_ + 3.0, 60.0);
        else if (key == 's') pitch_deg_ = std::max(pitch_deg_ - 3.0, -20.0);
        else if (key == 'q') distance_ = std::max(distance_ - 0.5, 5.0);
        else if (key == 'e') distance_ = std::min(distance_ + 0.5, 20.0);
    }

    void CubeMapArray::shutdown(demo::Env& /*env*/)
    {
        mgl::glUseProgram(mgl::ProgramId{0});
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::TextureCubeMapArray, mgl::TextureId{0});
        mgl::glBindImageTexture(mgl::ImageUnit{0}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
        if (fbo_.value != 0) mgl::glDeleteFramebuffers(1, &fbo_);
        if (!textures_.empty()) mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
        if (!buffers_.empty()) mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
        mgl::glDeleteVertexArrays(1, &sphere_vao_);
        mgl::glDeleteVertexArrays(1, &empty_vao_);
        for (const LookupProgram& p : kinds_)
        {
            mgl::glDeleteProgram(p.lookup);
            mgl::glDeleteProgram(p.sizes);
        }
        mgl::glDeleteProgram(image_layered_);
        mgl::glDeleteProgram(image_single_);
        mgl::glDeleteProgram(env_program_);
        mgl::glDeleteProgram(tile_program_);
    }

    // =================================================================================================
    // Compute helpers
    // =================================================================================================
    Bytes CubeMapArray::map_read(mgl::BufferTarget target, mgl::BufferId buf, std::size_t offset, std::size_t size)
    {
        mgl::glBindBuffer(target, buf);
        const void* p = mgl::glMapBufferRange(target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), mgl::MapBufferAccessMask::Read);
        if (p == nullptr) throw demo::Failure("glMapBufferRange failed");
        Bytes out(size);
        std::memcpy(out.data(), p, size);
        if (mgl::glUnmapBuffer(target) != GL_TRUE) throw demo::Failure("glUnmapBuffer returned FALSE");
        mgl::glBindBuffer(target, mgl::BufferId{0});
        return out;
    }

    std::vector<Words> CubeMapArray::run_lookups(Kind kind, mgl::TextureId tex, const std::vector<Probe>& probes)
    {
        const LookupProgram& p = kinds_[static_cast<std::size_t>(kind)];
        std::vector<GpuProbe> gpu;
        gpu.reserve(probes.size());
        for (const Probe& pr : probes)
            gpu.push_back({{static_cast<float>(pr.dir.x), static_cast<float>(pr.dir.y), static_cast<float>(pr.dir.z), static_cast<float>(pr.q)},
                           {static_cast<float>(pr.lod), static_cast<float>(pr.ref), 0.0f, 0.0f}});
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, probe_buf_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(gpu.size() * sizeof(GpuProbe)), gpu.data(), mgl::BufferUsage::DynamicDraw);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, probe_buf_);
        const std::size_t result_bytes = probes.size() * sizeof(Words);
        const std::vector<std::uint32_t> sentinel(result_bytes / 4, 0xDEADBEEFu);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, result_buf_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(result_bytes), sentinel.data(), mgl::BufferUsage::DynamicCopy);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, result_buf_);

        bind_cube_array(tex);
        mgl::glUseProgram(p.lookup);
        mgl::glUniform1i(p.lookup_tex, 0);
        mgl::glUniform1i(p.lookup_count, static_cast<GLint>(probes.size()));
        mgl::glDispatchCompute(static_cast<GLuint>((probes.size() + 63) / 64), 1, 1);
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        const Bytes raw = map_read(mgl::BufferTarget::ShaderStorage, result_buf_, 0, result_bytes);
        std::vector<Words> out(probes.size());
        std::memcpy(out.data(), raw.data(), raw.size());
        return out;
    }

    std::vector<std::array<int, 4>> CubeMapArray::run_sizes(Kind kind, mgl::TextureId tex, int levels)
    {
        const LookupProgram& p = kinds_[static_cast<std::size_t>(kind)];
        const std::vector<std::int32_t> sentinel(32, -77);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, result_buf_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, sizeof(std::int32_t) * 32, sentinel.data(), mgl::BufferUsage::DynamicCopy);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, result_buf_);
        bind_cube_array(tex);
        mgl::glUseProgram(p.sizes);
        mgl::glUniform1i(p.sizes_tex, 0);
        mgl::glUniform1i(p.sizes_levels, levels);
        mgl::glDispatchCompute(1, 1, 1);
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        const Bytes raw = map_read(mgl::BufferTarget::ShaderStorage, result_buf_, 0, sizeof(std::int32_t) * 32);
        std::vector<std::array<int, 4>> out(8);
        std::memcpy(out.data(), raw.data(), raw.size());
        return out;
    }

    // Compare float lookups with the model; norm_tol covers float rounding of equation 2.1.
    constexpr double kNormTol = 1e-6;

    Rgba8 words_to_rgba8(const Words& w, double& worst)
    {
        Rgba8 out{};
        for (std::size_t c = 0; c < 4; ++c)
        {
            const double v = static_cast<double>(std::bit_cast<float>(w[c])) * 255.0;
            const double r = std::round(v);
            worst = std::max(worst, std::abs(v - r));
            out[c] = static_cast<std::uint8_t>(std::clamp(r, 0.0, 255.0));
        }
        return out;
    }

    std::string describe(const Probe& pr, const Sample& S)
    {
        return std::format("direction ({:.3f}, {:.3f}, {:.3f}) q {:.2f} lod {:.2f} -> level {} layer {} face {} texel ({}, {})", pr.dir.x, pr.dir.y, pr.dir.z, pr.q, pr.lod, S.level, S.layer,
                           kFaceName[static_cast<std::size_t>(S.face)], S.i, S.j);
    }

    // =================================================================================================
    // validate
    // =================================================================================================
    void CubeMapArray::validate(demo::Env& env, demo::Check& check)
    {
        check_data_is_distinct(check);
        check_state_and_queries(check);
        check_framebuffer_layer_faces(check);
        check_image_access(check);
        check_float_lookups(env, check);
        check_integer_lookups(check);
        check_shadow_lookups(check);
        check_texture_size(check);
        check_completeness_and_mipmaps(check);
        check_errors(check);
        check_compile_rules(check);
        check_picture(env, check);
        check.expect_no_gl_error("no GL error after all experiments");
    }

    void CubeMapArray::check_data_is_distinct(demo::Check& check)
    {
        std::set<std::array<std::uint8_t, 3>> colours;
        int total = 0;
        for (int level = 0; level < kLevels; ++level)
            for (int layer = 0; layer < kLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                    for (int j = 0; j < level_size(level); ++j)
                        for (int i = 0; i < level_size(level); ++i)
                        {
                            const Rgba8 c = texel_color(level, layer, face, i, j);
                            colours.insert({c[0], c[1], c[2]});
                            ++total;
                        }
        check.expect(static_cast<int>(colours.size()) == total && total == kLayers * kFaces * (16 + 4 + 1),
                     "test data: all 630 texels of the 3 levels, 5 layers and 6 faces have pairwise different colours",
                     std::format("{} distinct of {}", colours.size(), total));
    }

    void CubeMapArray::check_state_and_queries(demo::Check& check)
    {
        const mgl::TextureTarget target = mgl::TextureTarget::TextureCubeMapArray;
        check.expect_eq(initial_binding_, static_cast<int>(tex_color_.value), "TEXTURE_BINDING_CUBE_MAP_ARRAY == the cube map array just bound with glBindTexture");
        mgl::glBindTexture(target, mgl::TextureId{0});
        check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBindingCubeMapArray), 0, "binding texture 0 clears TEXTURE_BINDING_CUBE_MAP_ARRAY");
        bind_cube_array(tex_color_);
        GLint immutable_format = -1, immutable_levels = -1;
        mgl::glGetTexParameteriv(target, mgl::TextureParameterQuery::ImmutableFormat, &immutable_format);
        mgl::glGetTexParameteriv(target, mgl::TextureParameterQuery::ImmutableLevels, &immutable_levels);
        check.expect(immutable_format == GL_TRUE && immutable_levels == kLevels, "glTexStorage3D made the cube map array immutable: TEXTURE_IMMUTABLE_FORMAT TRUE, TEXTURE_IMMUTABLE_LEVELS 3",
                     std::format("format {} levels {}", immutable_format, immutable_levels));
        for (int level = 0; level < kLevels; ++level)
        {
            GLint w = -1, h = -1, d = -1, fmt = -1;
            mgl::glGetTexLevelParameteriv(target, level, mgl::TextureLevelParameter::Width, &w);
            mgl::glGetTexLevelParameteriv(target, level, mgl::TextureLevelParameter::Height, &h);
            mgl::glGetTexLevelParameteriv(target, level, mgl::TextureLevelParameter::Depth, &d);
            mgl::glGetTexLevelParameteriv(target, level, mgl::TextureLevelParameter::InternalFormat, &fmt);
            check.expect(w == level_size(level) && h == level_size(level) && d == kFaces * kLayers && fmt == static_cast<GLint>(mgl::InternalFormat::Rgba8),
                         std::format("glGetTexLevelParameteriv(TEXTURE_CUBE_MAP_ARRAY, level {}): {} x {} faces, depth {} (= 6 x {} layer-faces; it does not shrink with the level), RGBA8", level, level_size(level),
                                     level_size(level), kFaces * kLayers, kLayers),
                         std::format("width {} height {} depth {} format {:#x}", w, h, d, fmt));
        }
        mgl::glBindTexture(target, mgl::TextureId{0});
    }

    // ---------------------------------------------------------------------------------------------------
    // Framebuffer layers: the layer-face numbering without any sampling
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_framebuffer_layer_faces(demo::Check& check)
    {
        // glFramebufferTextureLayer(layer = z): "for cube map array textures, layer is translated into an array layer
        // and a cube map face as described for layer-face numbers" (ES 3.2 9.2.8). Reading the attached image back with
        // glReadPixels shows the texels glTexSubImage3D stored at zoffset z, so the upload and Table 8.24 agree without
        // any cube map selection math.
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
        int bad_images = 0, bad_texels = 0, incomplete = 0, images = 0;
        std::string first;
        for (int level = 0; level < kLevels; ++level)
            for (int z = 0; z < kFaces * kLayers; ++z)
            {
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex_color_, level, z);
                if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                {
                    ++incomplete;
                    continue;
                }
                ++images;
                const int layer = z / kFaces, face = z % kFaces, size = level_size(level);
                const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, size, size);
                bool image_ok = true;
                for (int j = 0; j < size; ++j)
                    for (int i = 0; i < size; ++i)
                    {
                        const Rgba8 want = texel_color(level, layer, face, i, j);
                        const std::size_t o = (static_cast<std::size_t>(j) * static_cast<std::size_t>(size) + static_cast<std::size_t>(i)) * 4;
                        const Rgba8 got = {px[o], px[o + 1], px[o + 2], px[o + 3]};
                        if (got != want)
                        {
                            ++bad_texels;
                            image_ok = false;
                            if (first.empty())
                                first = std::format("level {} z {} (layer {} face {}) texel ({},{}): got {} expected {}", level, z, layer, kFaceName[static_cast<std::size_t>(face)], i, j,
                                                    demo::format_rgba(got), demo::format_rgba(want));
                        }
                    }
                if (!image_ok) ++bad_images;
            }
        check.expect(incomplete == 0 && bad_texels == 0 && images == kLevels * kFaces * kLayers,
                     "glFramebufferTextureLayer(z) attaches layer-face z = 6 layer + face: all 90 attached images (3 levels x 30 layer-faces) read back exactly the texels uploaded at zoffset z",
                     std::format("{} incomplete, {} wrong images, {} wrong texels; first: {}", incomplete, bad_images, bad_texels, first));
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
    }

    // ---------------------------------------------------------------------------------------------------
    // imageCubeArray
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_image_access(demo::Check& check)
    {
        // Layered binding: one imageCubeArray, coordinate z = layer-face (Table 8.25).
        for (int level = 0; level < kLevels; ++level)
        {
            const int size = level_size(level), zs = kFaces * kLayers;
            const std::size_t words = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * static_cast<std::size_t>(zs);
            const std::vector<std::uint32_t> sentinel(4 + words * 4, 0xDEADBEEFu);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, result_buf_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(sentinel.size() * 4), sentinel.data(), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, result_buf_);
            mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_color_, level, GL_TRUE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
            mgl::glUseProgram(image_layered_);
            mgl::glUniform3i(loc_img_extent_layered_, size, size, zs);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const Bytes raw = map_read(mgl::BufferTarget::ShaderStorage, result_buf_, 0, sentinel.size() * 4);
            std::array<std::int32_t, 4> image_size{};
            std::memcpy(image_size.data(), raw.data(), 16);
            int bad = 0;
            std::string first;
            for (int z = 0; z < zs; ++z)
                for (int j = 0; j < size; ++j)
                    for (int i = 0; i < size; ++i)
                    {
                        Words w{};
                        std::memcpy(w.data(), raw.data() + 16 + ((static_cast<std::size_t>(z) * size + static_cast<std::size_t>(j)) * size + static_cast<std::size_t>(i)) * 16, 16);
                        double worst = 0;
                        const Rgba8 got = words_to_rgba8(w, worst);
                        const Rgba8 want = texel_color(level, z / kFaces, z % kFaces, i, j);
                        if (got != want || worst > 1e-3)
                        {
                            if (bad == 0) first = std::format("z {} texel ({},{}): got {} expected {}", z, i, j, demo::format_rgba(got), demo::format_rgba(want));
                            ++bad;
                        }
                    }
            check.expect(bad == 0, std::format("imageLoad(imageCubeArray, ivec3(x, y, z)) at level {}: all {} texels (z = layer-face) equal the uploaded data", level, size * size * zs),
                         std::format("{} wrong; first: {}", bad, first));
            // GLSL ES 3.20 8.12: imageSize of a cube map array returns the face size and "the number of cubes in the cube map array".
            check.expect(image_size[0] == size && image_size[1] == size && image_size[2] == kLayers,
                         std::format("imageSize(imageCubeArray) at level {} == ({}, {}, {}): face size and the number of cubes (not 6 x cubes)", level, size, size, kLayers),
                         std::format("({}, {}, {})", image_size[0], image_size[1], image_size[2]));
        }

        // Non-layered bindings: layer = n binds ONE layer-face, "treated as a two-dimensional texture": image2D, 2D coordinates.
        int bad = 0, bound = 0;
        std::string first;
        const int size = kSize;
        for (int z = 0; z < kFaces * kLayers; ++z)
        {
            const std::vector<std::uint32_t> sentinel(4 + 16 * 4, 0xDEADBEEFu);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, result_buf_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(sentinel.size() * 4), sentinel.data(), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, result_buf_);
            mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_color_, 0, GL_FALSE, z, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
            mgl::glUseProgram(image_single_);
            mgl::glUniform3i(loc_img_extent_single_, size, size, 1);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const Bytes raw = map_read(mgl::BufferTarget::ShaderStorage, result_buf_, 0, sentinel.size() * 4);
            ++bound;
            for (int j = 0; j < size; ++j)
                for (int i = 0; i < size; ++i)
                {
                    Words w{};
                    std::memcpy(w.data(), raw.data() + 16 + (static_cast<std::size_t>(j) * size + static_cast<std::size_t>(i)) * 16, 16);
                    double worst = 0;
                    const Rgba8 got = words_to_rgba8(w, worst);
                    const Rgba8 want = texel_color(0, z / kFaces, z % kFaces, i, j);
                    if (got != want || worst > 1e-3)
                    {
                        if (bad == 0) first = std::format("layer {} (= layer-face of cube {} face {}) texel ({},{}): got {} expected {}", z, z / kFaces, kFaceName[static_cast<std::size_t>(z % kFaces)], i, j,
                                                          demo::format_rgba(got), demo::format_rgba(want));
                        ++bad;
                    }
                }
        }
        check.expect(bad == 0 && bound == 30,
                     "glBindImageTexture(layered = FALSE, layer = n) for n = 0..29 binds layer-face n as an image2D: layer = floor(n / 6), face = n - 6 layer (8.23); all 480 texels are right",
                     std::format("{} wrong; first: {}", bad, first));
        mgl::glBindImageTexture(mgl::ImageUnit{0}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
    }

    // ---------------------------------------------------------------------------------------------------
    // Float lookups
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_float_lookups(demo::Env& /*env*/, demo::Check& check)
    {
        // (a) texel centres of every level, layer and face
        std::vector<Probe> centres;
        for (int level = 0; level < kLevels; ++level)
            for (int layer = 0; layer < kLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                    for (int j = 0; j < level_size(level); ++j)
                        for (int i = 0; i < level_size(level); ++i)
                        {
                            const double size = level_size(level);
                            centres.push_back({face_direction(face, (i + 0.5) / size, (j + 0.5) / size), static_cast<double>(layer), static_cast<double>(level), 0.0});
                        }
        const std::vector<Words> got = run_lookups(Kind::Float, tex_color_, centres);
        int bad = 0;
        double worst_norm = 0;
        std::string first;
        std::array<int, 6> wrong_model_hits{};  // how often each deliberately wrong model equals the GPU result
        const std::array<Sample (*)(Sample), 6> wrong = {wrong_flip_t, wrong_flip_s, wrong_transpose, wrong_face_layer_swap, wrong_layer_plus_one, wrong_level_zero};
        for (std::size_t n = 0; n < centres.size(); ++n)
        {
            const Sample S = model_lookup(as_float(centres[n].dir), centres[n].q, centres[n].lod, kSize, kLevels, kLayers);
            double w = 0;
            const Rgba8 g = words_to_rgba8(got[n], w);
            worst_norm = std::max(worst_norm, w / 255.0);
            if (g != sample_color(S) || w > 255.0 * kNormTol)
            {
                if (bad == 0) first = std::format("{}: got {} expected {}", describe(centres[n], S), demo::format_rgba(g), demo::format_rgba(sample_color(S)));
                ++bad;
            }
            for (std::size_t k = 0; k < wrong.size(); ++k)
                if (g == sample_color(wrong[k](S))) ++wrong_model_hits[k];
        }
        check.expect(bad == 0, std::format("samplerCubeArray textureLod at the centre of every texel: {} lookups (3 levels, 5 layers, 6 faces) return exactly the colour of that level / layer / face / texel", centres.size()),
                     std::format("{} wrong; first: {}", bad, first));
        std::cout << std::format("   [note] max |fetched - c/255| over the texel-centre lookups: {:.2e}\n", worst_norm);
        // Negative controls: models with a mirrored s or t, a transposed image, the layer and the face swapped, a layer off by one
        // or the base level instead of the requested one must NOT reproduce the GPU result (the lookups would have found them).
        const std::array<const char*, 6> wrong_names = {"t mirrored", "s mirrored", "transposed", "face and layer swapped (z = 6 face + layer)", "layer + 1", "level 0 instead of the requested level"};
        for (std::size_t k = 0; k < wrong.size(); ++k)
        {
            // texels that are symmetric under the mutation cannot detect it, so count how many lookups it gets wrong.
            const int differ = static_cast<int>(centres.size()) - wrong_model_hits[k];
            check.expect(differ > static_cast<int>(centres.size()) / 10,
                         std::format("negative control: a model with {} disagrees with the GPU on a large share of the texel-centre lookups", wrong_names[k]),
                         std::format("{} of {} lookups differ", differ, centres.size()));
        }

        // (b) the layer coordinate: rounding and clamping (GLSL ES 3.20 8.9.2: layer = max(0, min(d - 1, floor(q + 0.5))))
        std::vector<Probe> layer_cases;
        const V3 dir = face_direction(2, 0.3, 0.8);  // a fixed direction on the +Y face, texel (1, 3) at level 0
        const std::array<double, 14> qs = {-3.7, -0.4, 0.0, 0.3, 0.7, 1.0, 1.2, 1.8, 3.3, 3.7, 4.0, 4.4, 4.6, 100.0};
        for (double q : qs) layer_cases.push_back({dir, q, 0.0, 0.0});
        const std::vector<Words> lg = run_lookups(Kind::Float, tex_color_, layer_cases);
        int lbad = 0;
        std::string lfirst;
        for (std::size_t n = 0; n < layer_cases.size(); ++n)
        {
            const Sample S = model_lookup(dir, layer_cases[n].q, 0.0, kSize, kLevels, kLayers);
            double w = 0;
            const Rgba8 g = words_to_rgba8(lg[n], w);
            if (g != sample_color(S))
            {
                if (lbad == 0) lfirst = std::format("q = {}: got {} expected layer {} = {}", layer_cases[n].q, demo::format_rgba(g), S.layer, demo::format_rgba(sample_color(S)));
                ++lbad;
            }
        }
        check.expect(lbad == 0, "the layer coordinate q is rounded to the nearest cube and clamped to 0 .. 4: q = -3.7, -0.4, 0.3, 0.7, 1.2, 1.8, 3.3, 3.7, 4.4, 4.6, 100 and the integers select layer 0..4",
                     std::format("{} wrong; first: {}", lbad, lfirst));

        // (c) fractional levels of detail: NEAREST_MIPMAP_NEAREST picks round(lambda), lambda <= 0 -> base level, large -> last level
        std::vector<Probe> lod_cases;
        const std::array<double, 12> lods = {-2.0, -0.2, 0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 9.0};
        const V3 ldir = face_direction(4, 0.62, 0.30);
        for (double l : lods) lod_cases.push_back({ldir, 3.0, l, 0.0});
        const std::vector<Words> gg = run_lookups(Kind::Float, tex_color_, lod_cases);
        int dbad = 0;
        std::string dfirst;
        for (std::size_t n = 0; n < lod_cases.size(); ++n)
        {
            const Sample S = model_lookup(ldir, 3.0, lod_cases[n].lod, kSize, kLevels, kLayers);
            double w = 0;
            const Rgba8 g = words_to_rgba8(gg[n], w);
            if (g != sample_color(S))
            {
                if (dbad == 0) dfirst = std::format("lod {}: got {} expected level {} = {}", lod_cases[n].lod, demo::format_rgba(g), S.level, demo::format_rgba(sample_color(S)));
                ++dbad;
            }
        }
        check.expect(dbad == 0, "textureLod with the level of detail -2, -0.2, 0, 0.3, 0.7, 1, 1.3, 1.7, 2, 2.3, 2.7, 9 selects level 0,0,0,0,1,1,1,2,2,2,2,2 (equation 8.11)", std::format("{} wrong; first: {}", dbad, dfirst));

        // (d) random directions, layer coordinates and levels of detail
        Rng rng;
        std::vector<Probe> random;
        for (int n = 0; n < 2400; ++n)
        {
            V3 d{rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)};
            if (std::abs(d.x) + std::abs(d.y) + std::abs(d.z) < 0.2) d.x = 0.9;
            double q = 0;
            switch (rng.below(4))
            {
            case 0: q = rng.below(kLayers); break;
            case 1: q = rng.uniform(-1.5, kLayers + 1.5); break;
            default: q = rng.uniform(0.0, kLayers - 1.0); break;
            }
            double lod = rng.below(2) == 0 ? static_cast<double>(rng.below(kLevels)) : rng.uniform(-0.5, kLevels + 0.5);
            random.push_back({d, q, lod, 0.0});
        }
        const std::vector<Words> rg = run_lookups(Kind::Float, tex_color_, random);
        Mismatch m;
        std::array<int, kFaces> faces_seen{};
        std::array<int, kLayers> layers_seen{};
        std::array<int, kLevels> levels_seen{};
        for (std::size_t n = 0; n < random.size(); ++n)
        {
            const Sample S = model_lookup(as_float(random[n].dir), random[n].q, random[n].lod, kSize, kLevels, kLayers);
            if (S.ambiguous)
            {
                ++m.skipped;
                continue;
            }
            ++m.compared;
            ++faces_seen[static_cast<std::size_t>(S.face)];
            ++layers_seen[static_cast<std::size_t>(S.layer)];
            ++levels_seen[static_cast<std::size_t>(S.level)];
            double w = 0;
            const Rgba8 g = words_to_rgba8(rg[n], w);
            if (g != sample_color(S) || w > 255.0 * kNormTol)
            {
                if (m.bad == 0) m.first = std::format("{}: got {} expected {}", describe(random[n], S), demo::format_rgba(g), demo::format_rgba(sample_color(S)));
                ++m.bad;
            }
        }
        const bool all_seen = std::ranges::all_of(faces_seen, [](int c) { return c > 50; }) && std::ranges::all_of(layers_seen, [](int c) { return c > 50; }) &&
                              std::ranges::all_of(levels_seen, [](int c) { return c > 50; });
        check.expect(m.bad == 0 && m.compared > 1000 && all_seen,
                     std::format("2400 random lookups (direction, layer coordinate -1.5 .. 6.5, level of detail -0.5 .. 3.5): {} compared with the model ({} skipped near texel boundaries, face edges and rounding ties), all 6 faces, 5 layers and 3 levels covered",
                                 m.compared, m.skipped),
                     std::format("{} wrong; first: {}", m.bad, m.first));
    }

    // ---------------------------------------------------------------------------------------------------
    // Integer lookups
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_integer_lookups(demo::Check& check)
    {
        std::vector<Probe> probes;
        for (int layer = 0; layer < kSmallLayers; ++layer)
            for (int face = 0; face < kFaces; ++face)
                for (int j = 0; j < kSmall; ++j)
                    for (int i = 0; i < kSmall; ++i) probes.push_back({face_direction(face, (i + 0.5) / kSmall, (j + 0.5) / kSmall), static_cast<double>(layer), 0.0, 0.0});
        Rng rng;
        for (int n = 0; n < 300; ++n)
            probes.push_back({{rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)}, static_cast<double>(rng.below(kSmallLayers)), 0.0, 0.0});
        for (const Kind kind : {Kind::Uint, Kind::Int})
        {
            const mgl::TextureId tex = kind == Kind::Uint ? tex_uint_ : tex_int_;
            const std::vector<Words> got = run_lookups(kind, tex, probes);
            int compared = 0, bad = 0;
            std::string first;
            for (std::size_t n = 0; n < probes.size(); ++n)
            {
                const Sample S = model_lookup(as_float(probes[n].dir), probes[n].q, 0.0, kSmall, 1, kSmallLayers);  // one level of 2 x 2 faces
                if (S.ambiguous) continue;
                ++compared;
                Words want{};
                if (kind == Kind::Uint)
                {
                    const auto t = uint_texel(S.layer, S.face, S.i, S.j);
                    for (std::size_t c = 0; c < 4; ++c) want[c] = t[c];
                }
                else
                {
                    const auto t = int_texel(S.layer, S.face, S.i, S.j);
                    for (std::size_t c = 0; c < 4; ++c) want[c] = static_cast<std::uint32_t>(static_cast<std::int32_t>(t[c]));
                }
                if (got[n] != want)
                {
                    if (bad == 0) first = std::format("{}: got {:#x} {:#x} {:#x} {:#x} expected {:#x} {:#x} {:#x} {:#x}", describe(probes[n], S), got[n][0], got[n][1], got[n][2], got[n][3], want[0], want[1], want[2], want[3]);
                    ++bad;
                }
            }
            check.expect(bad == 0 && compared > 200,
                         std::format("{} on an {} cube map array (2 x 2 faces, 3 cubes): {} lookups (all 72 texel centres + random directions) return exactly the integer texel of that layer / face / texel", kind == Kind::Uint ? "usamplerCubeArray" : "isamplerCubeArray",
                                     kind == Kind::Uint ? "RGBA8UI" : "RGBA16I (negative values)", compared),
                         std::format("{} wrong; first: {}", bad, first));
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Shadow lookups
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_shadow_lookups(demo::Check& check)
    {
        const mgl::TextureTarget target = mgl::TextureTarget::TextureCubeMapArray;
        // Table 8.22. Dref and Dt are exact floats (n / 128, ref = (2n + k) / 256 with k in {-1, 0, 1}).
        struct Func
        {
            mgl::CompareFunc gl;
            const char* name;
            bool (*pass)(double ref, double dt);
        };
        const std::array<Func, 8> funcs = {{
            {mgl::CompareFunc::Lequal, "LEQUAL", [](double r, double d) { return r <= d; }},
            {mgl::CompareFunc::Gequal, "GEQUAL", [](double r, double d) { return r >= d; }},
            {mgl::CompareFunc::Less, "LESS", [](double r, double d) { return r < d; }},
            {mgl::CompareFunc::Greater, "GREATER", [](double r, double d) { return r > d; }},
            {mgl::CompareFunc::Equal, "EQUAL", [](double r, double d) { return r == d; }},
            {mgl::CompareFunc::Notequal, "NOTEQUAL", [](double r, double d) { return r != d; }},
            {mgl::CompareFunc::Always, "ALWAYS", [](double, double) { return true; }},
            {mgl::CompareFunc::Never, "NEVER", [](double, double) { return false; }},
        }};
        std::vector<Probe> probes;
        std::vector<double> dts;
        for (int layer = 0; layer < kSmallLayers; ++layer)
            for (int face = 0; face < kFaces; ++face)
                for (int j = 0; j < kSmall; ++j)
                    for (int i = 0; i < kSmall; ++i)
                        for (int k = -1; k <= 1; ++k)
                        {
                            const double dt = static_cast<double>(depth_texel(layer, face, i, j));
                            probes.push_back({face_direction(face, (i + 0.5) / kSmall, (j + 0.5) / kSmall), static_cast<double>(layer), 0.0, dt + k / 256.0});
                            dts.push_back(dt);
                        }
        bind_cube_array(tex_depth_);
        std::array<int, 8> mismatches_vs_lequal{};
        for (std::size_t f = 0; f < funcs.size(); ++f)
        {
            mgl::glTexParameter(target, funcs[f].gl);
            const std::vector<Words> got = run_lookups(Kind::Shadow, tex_depth_, probes);
            int bad = 0;
            std::string first;
            for (std::size_t n = 0; n < probes.size(); ++n)
            {
                const float r = std::bit_cast<float>(got[n][0]);
                const bool want = funcs[f].pass(probes[n].ref, dts[n]);
                if (r != (want ? 1.0f : 0.0f) || got[n][3] != std::bit_cast<std::uint32_t>(1.0f))
                {
                    if (bad == 0) first = std::format("{} Dref {} Dt {}: got {} expected {}", describe(probes[n], model_lookup(probes[n].dir, probes[n].q, 0.0, kSmall, 1, kSmallLayers)), probes[n].ref, dts[n], r, want ? 1 : 0);
                    ++bad;
                }
                if (want != funcs[0].pass(probes[n].ref, dts[n])) ++mismatches_vs_lequal[f];
            }
            check.expect(bad == 0, std::format("samplerCubeArrayShadow with TEXTURE_COMPARE_FUNC {}: {} lookups (72 texels x Dref = Dt - 1/256, Dt, Dt + 1/256) return 1.0 / 0.0 as Table 8.22 says", funcs[f].name, probes.size()),
                         std::format("{} wrong; first: {}", bad, first));
        }
        // Negative control: the eight functions give different results (the comparison function really is applied).
        check.expect(mismatches_vs_lequal[1] > 0 && mismatches_vs_lequal[2] > 0 && mismatches_vs_lequal[4] > 0 && mismatches_vs_lequal[6] > 0 && mismatches_vs_lequal[7] > 0,
                     "negative control: GEQUAL, LESS, EQUAL, ALWAYS and NEVER give results that LEQUAL does not (the expected values depend on the function)",
                     std::format("differences vs LEQUAL: {} {} {} {} {} {} {} {}", mismatches_vs_lequal[0], mismatches_vs_lequal[1], mismatches_vs_lequal[2], mismatches_vs_lequal[3], mismatches_vs_lequal[4],
                                 mismatches_vs_lequal[5], mismatches_vs_lequal[6], mismatches_vs_lequal[7]));

        // With TEXTURE_COMPARE_MODE NONE a depth texture read through a plain samplerCubeArray returns (Dt, 0, 0, 1) (8.20.1, 11.1.3.5).
        mgl::glTexParameter(target, mgl::TextureCompareMode::None);
        std::vector<Probe> plain;
        for (std::size_t n = 0; n < probes.size(); n += 3) plain.push_back(probes[n]);
        const std::vector<Words> pg = run_lookups(Kind::Float, tex_depth_, plain);
        int pbad = 0;
        std::string pfirst;
        for (std::size_t n = 0; n < plain.size(); ++n)
        {
            const double dt = dts[3 * n];
            if (std::bit_cast<float>(pg[n][0]) != static_cast<float>(dt) || pg[n][1] != 0u || pg[n][2] != 0u || pg[n][3] != std::bit_cast<std::uint32_t>(1.0f))
            {
                if (pbad == 0) pfirst = std::format("probe {}: got {} {} {} {} expected Dt {}", n, std::bit_cast<float>(pg[n][0]), std::bit_cast<float>(pg[n][1]), std::bit_cast<float>(pg[n][2]), std::bit_cast<float>(pg[n][3]), dt);
                ++pbad;
            }
        }
        check.expect(pbad == 0, "TEXTURE_COMPARE_MODE NONE: a depth cube map array read with samplerCubeArray returns (Dt, 0, 0, 1) with the exact depth of every layer-face texel",
                     std::format("{} of {} wrong; first: {}", pbad, plain.size(), pfirst));
        mgl::glTexParameter(target, mgl::TextureCompareMode::CompareRefToTexture);
        mgl::glTexParameter(target, mgl::CompareFunc::Lequal);
        mgl::glBindTexture(target, mgl::TextureId{0});
    }

    // ---------------------------------------------------------------------------------------------------
    // textureSize
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_texture_size(demo::Check& check)
    {
        struct Case
        {
            const char* what;
            Kind kind;
            mgl::TextureId tex;
            int levels, size, layers;
        };
        const std::array<Case, 4> cases = {{
            {"samplerCubeArray (RGBA8, 3 levels)", Kind::Float, tex_color_, kLevels, kSize, kLayers},
            {"isamplerCubeArray (RGBA16I)", Kind::Int, tex_int_, 1, kSmall, kSmallLayers},
            {"usamplerCubeArray (RGBA8UI)", Kind::Uint, tex_uint_, 1, kSmall, kSmallLayers},
            {"samplerCubeArrayShadow (DEPTH_COMPONENT32F)", Kind::Shadow, tex_depth_, 1, kSmall, kSmallLayers},
        }};
        for (const Case& c : cases)
        {
            const auto sizes = run_sizes(c.kind, c.tex, c.levels);
            bool ok = true;
            std::string got;
            for (int level = 0; level < c.levels; ++level)
            {
                const int s = std::max(1, c.size >> level);
                const auto& v = sizes[static_cast<std::size_t>(level)];
                ok = ok && v[0] == s && v[1] == s && v[2] == c.layers && v[3] == 0;
                got += std::format("({},{},{}) ", v[0], v[1], v[2]);
            }
            check.expect(ok, std::format("textureSize({}, lod) is (face size >> lod, face size >> lod, {} cubes) for every level (GLSL ES 3.20 8.9.1: the last component is the number of cubes, not {} layer-faces)", c.what,
                                         c.layers, c.layers * kFaces),
                         "got " + got);
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Completeness, glGenerateMipmap, mutable cube map arrays
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_completeness_and_mipmaps(demo::Check& check)
    {
        const mgl::TextureTarget target = mgl::TextureTarget::TextureCubeMapArray;
        constexpr int kMutLayers = 2;
        const mgl::TextureId tex = new_texture();
        bind_cube_array(tex);
        Bytes level0;
        for (int layer = 0; layer < kMutLayers; ++layer)
            for (int face = 0; face < kFaces; ++face)
            {
                const Bytes img = face_image(0, layer, face);
                level0.insert(level0.end(), img.begin(), img.end());
            }
        // glTexImage3D: mutable storage, depth = 6 * layers.
        mgl::glTexImage3D(target, 0, mgl::InternalFormat::Rgba8, kSize, kSize, kFaces * kMutLayers, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, level0.data());
        check.expect_no_gl_error("glTexImage3D(TEXTURE_CUBE_MAP_ARRAY, 4 x 4, depth 12) with depth a multiple of six: no error");
        GLint depth = -1, immutable = -1;
        mgl::glGetTexLevelParameteriv(target, 0, mgl::TextureLevelParameter::Depth, &depth);
        mgl::glGetTexParameteriv(target, mgl::TextureParameterQuery::ImmutableFormat, &immutable);
        check.expect(depth == 12 && immutable == GL_FALSE, "the glTexImage3D cube map array has depth 12 layer-faces and mutable format (TEXTURE_IMMUTABLE_FORMAT FALSE)", std::format("depth {} immutable {}", depth, immutable));

        const std::vector<Probe> probe = {{face_direction(1, 0.7, 0.2), 1.0, 0.0, 0.0}, {face_direction(4, 0.2, 0.6), 0.0, 0.0, 0.0}};
        const Words black = {0u, 0u, 0u, std::bit_cast<std::uint32_t>(1.0f)};
        // Default MIN_FILTER is NEAREST_MIPMAP_LINEAR: it needs levels 1 and 2, which were never specified -> incomplete.
        std::vector<Words> got = run_lookups(Kind::Float, tex, probe);
        check.expect(got[0] == black && got[1] == black, "not mipmap complete (default NEAREST_MIPMAP_LINEAR, only level 0 specified): lookups return (0, 0, 0, 1) (11.1.3.5)",
                     std::format("got {:#x} {:#x} {:#x} {:#x}", got[0][0], got[0][1], got[0][2], got[0][3]));
        // NEAREST needs no mipmaps: base complete -> the texels appear.
        bind_cube_array(tex);
        mgl::glTexParameter(target, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(target, mgl::TextureMagFilter::Nearest);
        got = run_lookups(Kind::Float, tex, probe);
        int bad = 0;
        for (std::size_t n = 0; n < probe.size(); ++n)
        {
            const Sample S = model_lookup(as_float(probe[n].dir), probe[n].q, 0.0, kSize, 1, kMutLayers);
            double w = 0;
            if (words_to_rgba8(got[n], w) != sample_color(S)) ++bad;
        }
        check.expect(bad == 0, "MIN_FILTER NEAREST makes the same texture complete (cube array complete: every slice is cube complete): both lookups return the texels of their layer and face", std::format("{} wrong", bad));

        // glGenerateMipmap: "for two-dimensional array and cube map array textures, each layer is filtered independently".
        bind_cube_array(tex);
        mgl::glTexParameter(target, mgl::TextureMinFilter::NearestMipmapNearest);
        mgl::glGenerateMipmap(target);
        check.expect_no_gl_error("glGenerateMipmap(TEXTURE_CUBE_MAP_ARRAY) on a cube array complete RGBA8 texture: no error");
        GLint w1 = -1, h1 = -1, d1 = -1, w2 = -1, d2 = -1;
        mgl::glGetTexLevelParameteriv(target, 1, mgl::TextureLevelParameter::Width, &w1);
        mgl::glGetTexLevelParameteriv(target, 1, mgl::TextureLevelParameter::Height, &h1);
        mgl::glGetTexLevelParameteriv(target, 1, mgl::TextureLevelParameter::Depth, &d1);
        mgl::glGetTexLevelParameteriv(target, 2, mgl::TextureLevelParameter::Width, &w2);
        mgl::glGetTexLevelParameteriv(target, 2, mgl::TextureLevelParameter::Depth, &d2);
        check.expect(w1 == 2 && h1 == 2 && d1 == 12 && w2 == 1 && d2 == 12, "glGenerateMipmap created level 1 (2 x 2) and level 2 (1 x 1), each with all 12 layer-faces",
                     std::format("level 1 {}x{}x{}, level 2 width {} depth {}", w1, h1, d1, w2, d2));
        // Every texel of a face has the same R (face code) and G (layer code), only B varies, so a filter that keeps layers and faces
        // apart returns R and G unchanged (a convex combination of equal values) and B between the minimum and maximum of the face.
        std::vector<Probe> mip_probes;
        std::vector<std::array<int, 3>> expect;  // layer, face, level
        for (int level = 1; level <= 2; ++level)
            for (int layer = 0; layer < kMutLayers; ++layer)
                for (int face = 0; face < kFaces; ++face)
                {
                    mip_probes.push_back({face_direction(face, 0.3, 0.3), static_cast<double>(layer), static_cast<double>(level), 0.0});
                    expect.push_back({layer, face, level});
                }
        const std::vector<Words> mg = run_lookups(Kind::Float, tex, mip_probes);
        int mbad = 0;
        std::string mfirst;
        for (std::size_t n = 0; n < mip_probes.size(); ++n)
        {
            double w = 0;
            const Rgba8 g = words_to_rgba8(mg[n], w);
            const auto [layer, face, level] = std::tuple{expect[n][0], expect[n][1], expect[n][2]};
            const Rgba8 any = texel_color(0, layer, face, 0, 0);
            const int bmin = 16, bmax = 16 + 14 * 15;
            const bool ok = std::abs(static_cast<int>(g[0]) - static_cast<int>(any[0])) <= 1 && std::abs(static_cast<int>(g[1]) - static_cast<int>(any[1])) <= 1 && g[2] >= bmin - 1 && g[2] <= bmax + 1 && g[3] == 255;
            if (!ok)
            {
                if (mbad == 0) mfirst = std::format("layer {} face {} level {}: got {} (expected R {} G {})", layer, kFaceName[static_cast<std::size_t>(face)], level, demo::format_rgba(g), any[0], any[1]);
                ++mbad;
            }
        }
        check.expect(mbad == 0, "after glGenerateMipmap every generated texel of levels 1 and 2 keeps the face code (R) and layer code (G) of its own layer-face (+-1: 8-bit rounding of an average): layers and faces are filtered independently",
                     std::format("{} of {} wrong; first: {}", mbad, mip_probes.size(), mfirst));

        // glGenerateMipmap needs a color-renderable AND texture-filterable format (8.14.4): RGBA8UI is not filterable.
        bind_cube_array(tex_uint_);
        demo::drain_gl_errors();
        mgl::glGenerateMipmap(target);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGenerateMipmap on an RGBA8UI cube map array -> INVALID_OPERATION (the format is not texture-filterable, 8.14.4)");
        mgl::glBindTexture(target, mgl::TextureId{0});
    }

    // ---------------------------------------------------------------------------------------------------
    // Errors
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_errors(demo::Check& check)
    {
        const mgl::TextureTarget target = mgl::TextureTarget::TextureCubeMapArray;
        const mgl::SizedInternalFormat rgba8 = mgl::SizedInternalFormat::Rgba8;
        demo::drain_gl_errors();
        {
            // glTexStorage3D is defined by the TexImage3D pseudo-code (8.18), so the TexImage3D rules apply.
            bind_cube_array(new_texture());
            mgl::glTexStorage3D(target, 1, rgba8, 4, 4, 7);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexStorage3D(CUBE_MAP_ARRAY, depth 7): depth is not a multiple of six -> INVALID_VALUE (8.5 via the 8.18 pseudo-code)");
            mgl::glTexStorage3D(target, 1, rgba8, 4, 8, 6);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexStorage3D(CUBE_MAP_ARRAY, 4 x 8): width and height differ -> INVALID_VALUE");
            mgl::glTexStorage3D(target, 1, rgba8, 4, 4, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexStorage3D(CUBE_MAP_ARRAY, depth 0) -> INVALID_VALUE (8.18: width, height, depth and levels must be at least 1)");
            mgl::glTexStorage3D(target, 4, rgba8, 4, 4, 6);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glTexStorage3D(CUBE_MAP_ARRAY, levels 4 for 4 x 4 faces) -> INVALID_OPERATION (at most floor(log2(4)) + 1 = 3 levels)");
            mgl::glBindTexture(target, mgl::TextureId{0});
            mgl::glTexStorage3D(target, 1, rgba8, 4, 4, 6);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glTexStorage3D with texture 0 bound to TEXTURE_CUBE_MAP_ARRAY -> INVALID_OPERATION");
        }
        {
            bind_cube_array(tex_color_);
            const std::vector<std::uint8_t> px(64 * 7, 0);
            mgl::glTexImage3D(target, 0, mgl::InternalFormat::Rgba8, 4, 4, 6, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glTexImage3D on an immutable-format cube map array -> INVALID_OPERATION (8.18: no further changes to dimensions or format)");
            mgl::glTexStorage3D(target, 1, rgba8, 4, 4, 6);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glTexStorage3D on an immutable-format cube map array -> INVALID_OPERATION");
        }
        {
            const mgl::TextureId t = new_texture();
            bind_cube_array(t);
            const std::vector<std::uint8_t> px(4 * 4 * 4 * 7, 0);
            mgl::glTexImage3D(target, 0, mgl::InternalFormat::Rgba8, 4, 4, 7, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexImage3D(CUBE_MAP_ARRAY, depth 7) -> INVALID_VALUE: depth is not a multiple of six (8.5)");
            mgl::glTexImage3D(target, 0, mgl::InternalFormat::Rgba8, 4, 8, 6, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexImage3D(CUBE_MAP_ARRAY, 4 x 8 faces) -> INVALID_VALUE: width and height are not equal (8.5)");
            mgl::glTexImage3D(target, 0, mgl::InternalFormat::Rgba8, 4, 4, 6, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexImage3D(CUBE_MAP_ARRAY, border 1) -> INVALID_VALUE");
        }
        {
            // glTexSubImage3D: the update region is checked against the layer-face count (8.6: z + d > dt).
            bind_cube_array(tex_color_);
            const std::vector<std::uint8_t> px(4 * 4 * 4 * 2, 7);
            const mgl::PixelFormat rgba = mgl::PixelFormat::Rgba;
            const mgl::PixelType ub = mgl::PixelType::UnsignedByte;
            mgl::glTexSubImage3D(target, 0, 0, 0, 29, 4, 4, 2, rgba, ub, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexSubImage3D(zoffset 29, depth 2) on 30 layer-faces: z + d > depth -> INVALID_VALUE");
            mgl::glTexSubImage3D(target, 0, 0, 0, 30, 4, 4, 1, rgba, ub, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexSubImage3D(zoffset 30): the layer-face 30 does not exist (6 x 5 = 30) -> INVALID_VALUE");
            mgl::glTexSubImage3D(target, 0, 2, 0, 0, 4, 4, 1, rgba, ub, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexSubImage3D(xoffset 2, width 4) on 4-texel faces: x + w > width -> INVALID_VALUE");
            mgl::glTexSubImage3D(target, 0, 0, 0, -1, 4, 4, 1, rgba, ub, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexSubImage3D(zoffset -1) -> INVALID_VALUE");
            mgl::glTexSubImage3D(target, -1, 0, 0, 0, 4, 4, 1, rgba, ub, px.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexSubImage3D(level -1) -> INVALID_VALUE");
            // The failed calls changed nothing: layer-face 29 still holds its original texels.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex_color_, 0, 29);
            const std::vector<std::uint8_t> back = demo::read_rgba8(0, 0, 4, 4);
            check.expect(Bytes(back.begin(), back.end()) == face_image(0, 4, 5), "the failed glTexSubImage3D calls left layer-face 29 (cube 4, -Z) unchanged (an erroring command has no effect, 2.3.1)");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindTexture(target, mgl::TextureId{0});
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Compile-time rules
    // ---------------------------------------------------------------------------------------------------
    void CubeMapArray::check_compile_rules(demo::Check& check)
    {
        auto compiles = [](std::string_view version, std::string_view body, std::string& log) {
            const std::string src = std::format("#version {}\nlayout(local_size_x = 1) in;\nlayout(std430, binding = 0) buffer Out {{ vec4 r; }};\n{}\n", version, body);
            const mgl::ShaderId s = demo::compile_shader(mgl::ShaderType::Compute, src, log);
            if (s.value == 0) return false;
            mgl::glDeleteShader(s);
            return true;
        };
        std::string log;
        const bool control = compiles("320 es",
                                      "uniform highp samplerCubeArray a;\nuniform highp isamplerCubeArray ia;\nuniform highp usamplerCubeArray ua;\nuniform highp samplerCubeArrayShadow sh;\n"
                                      "void main() { r = textureLod(a, vec4(1.0, 0.0, 0.0, 2.0), 1.0) + vec4(textureLod(ia, vec4(0.0, 1.0, 0.0, 1.0), 0.0)) + vec4(textureLod(ua, vec4(0.0, 0.0, 1.0, 0.0), 0.0)) +\n"
                                      "  vec4(texture(sh, vec4(1.0, 1.0, 0.0, 1.0), 0.5)) + vec4(textureSize(a, 1), 0.0) + vec4(textureSize(sh, 0), 0.0); }",
                                      log);
        check.expect(control, "control: samplerCubeArray / isamplerCubeArray / usamplerCubeArray / samplerCubeArrayShadow with textureLod, texture(sh, vec4, float) and textureSize compile as GLSL ES 3.20", log);

        struct Negative
        {
            const char* version;
            const char* what;
            const char* body;
        };
        const std::array<Negative, 8> negatives = {{
            {"320 es", "texelFetch(samplerCubeArray, ivec3, int) does not exist: texel fetches cannot access cube map textures (ES 3.2 11.1.3.2, GLSL ES 3.20 8.9.2 has no cube forms)",
             "uniform highp samplerCubeArray a;\nvoid main() { r = texelFetch(a, ivec3(0, 0, 0), 0); }"},
            {"320 es", "texelFetch(samplerCube, ivec2, int) does not exist either (the same rule for plain cube maps)", "uniform highp samplerCube a;\nvoid main() { r = texelFetch(a, ivec2(0, 0), 0); }"},
            {"320 es", "samplerCubeArray without a precision qualifier does not compile: there is no default precision for it (GLSL ES 3.20 4.7.3)",
             "uniform samplerCubeArray a;\nvoid main() { r = textureLod(a, vec4(0.0), 0.0); }"},
            {"320 es", "texture(samplerCubeArray, vec3) does not exist: a cube map array needs the four-component (direction, layer) coordinate",
             "uniform highp samplerCubeArray a;\nvoid main() { r = textureLod(a, vec3(1.0, 0.0, 0.0), 0.0); }"},
            {"320 es", "texture(samplerCubeArrayShadow, vec4) without the separate compare argument does not exist (8.9.2: float texture(samplerCubeArrayShadow sampler, vec4 P, float compare))",
             "uniform highp samplerCubeArrayShadow a;\nvoid main() { r = vec4(texture(a, vec4(1.0))); }"},
            {"320 es", "textureLod(samplerCubeArrayShadow, ...) does not exist (only texture() and textureGather() accept a cube map array shadow sampler)",
             "uniform highp samplerCubeArrayShadow a;\nvoid main() { r = vec4(textureLod(a, vec4(1.0), 0.0)); }"},
            {"320 es", "textureOffset(samplerCubeArray, ...) does not exist: texel offsets are not supported for cube maps (8.9.2)",
             "uniform highp samplerCubeArray a;\nvoid main() { r = textureOffset(a, vec4(1.0), ivec2(1)); }"},
            {"310 es", "samplerCubeArray is not part of GLSL ES 3.10 (new in 3.20, section 4.1.7); no extension is enabled",
             "uniform highp samplerCubeArray a;\nvoid main() { r = vec4(0.0); }"},
        }};
        for (const Negative& n : negatives)
        {
            std::string nlog;
            const bool ok = compiles(n.version, n.body, nlog);
            check.expect(!ok, n.what, ok ? "the shader compiled" : "");
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // The picture
    // ---------------------------------------------------------------------------------------------------
    struct SphereHit
    {
        int id = -1;
        V3 point{}, normal{};
    };

    SphereHit hit_spheres(const std::array<V3, 5>& centres, double radius, V3 origin, V3 dir)
    {
        SphereHit best;
        double best_t = std::numeric_limits<double>::infinity();
        for (int k = 0; k < 5; ++k)
        {
            const V3 oc = origin - centres[static_cast<std::size_t>(k)];
            const double b = dot(oc, dir), c = dot(oc, oc) - radius * radius, disc = b * b - c;
            if (disc < 0) continue;
            const double t = -b - std::sqrt(disc);
            if (t > 0 && t < best_t)
            {
                best_t = t;
                best.id = k;
                best.point = origin + dir * t;
                best.normal = (best.point - centres[static_cast<std::size_t>(k)]) * (1.0 / radius);
            }
        }
        return best;
    }

    V3 reflect(V3 incident, V3 n) { return incident - n * (2.0 * dot(incident, n)); }

    void CubeMapArray::check_picture(demo::Env& env, demo::Check& check)
    {
        render(env, shown_frame_);
        const Layout lay = layout_of(env);
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, lay.width, lay.height);
        auto at = [&](int x, int y_top) -> Rgba8 {
            const std::size_t i = (static_cast<std::size_t>(lay.height - 1 - y_top) * static_cast<std::size_t>(lay.width) + static_cast<std::size_t>(x)) * 4;
            return {px[i], px[i + 1], px[i + 2], px[i + 3]};
        };

        // ---- the 30 tiles ---------------------------------------------------------------------------------
        int tile_bad = 0, tile_probes = 0, tile_area = 0;
        std::string first;
        std::array<int, 6> control_equal{};  // pixels where a wrong model happens to equal the picture
        const std::array<Sample (*)(Sample), 6> wrong = {wrong_flip_t, wrong_flip_s, wrong_transpose, wrong_face_layer_swap, wrong_layer_plus_one, wrong_level_zero};
        for (int z = 0; z < kFaces * kLayers; ++z)
        {
            const int layer = z / kFaces, face = z % kFaces, col = z % kTilesPerRow, row = z / kTilesPerRow;
            const int x0 = lay.col_x(col), y0 = lay.height - lay.tile_h + row * lay.tile, tile_w = lay.col_x(col + 1) - x0;
            tile_area += tile_w * lay.tile;
            for (int py = 0; py < lay.tile; ++py)
                for (int pxx = 0; pxx < tile_w; ++pxx)
                {
                    const double s = (pxx + 0.5) / tile_w, t = (py + 0.5) / lay.tile;  // t grows downwards, like the rows of the upload
                    bool near_boundary = false;
                    for (int k = 1; k < kSize; ++k)
                        near_boundary = near_boundary || std::abs(s * kSize - k) < kTexelMargin || std::abs(t * kSize - k) < kTexelMargin;
                    if (near_boundary) continue;
                    Sample S;
                    S.level = 0;
                    S.layer = layer;
                    S.face = face;
                    S.i = static_cast<int>(std::floor(s * kSize));
                    S.j = static_cast<int>(std::floor(t * kSize));
                    ++tile_probes;
                    const Rgba8 got = at(x0 + pxx, y0 + py);
                    if (got != sample_color(S))
                    {
                        if (tile_bad == 0)
                            first = std::format("tile z {} (layer {} face {}) pixel ({},{}): got {} expected {}", z, layer, kFaceName[static_cast<std::size_t>(face)], pxx, py, demo::format_rgba(got), demo::format_rgba(sample_color(S)));
                        ++tile_bad;
                    }
                    for (std::size_t k = 0; k < wrong.size(); ++k)
                        if (got == sample_color(wrong[k](S))) ++control_equal[k];
                }
        }
        // Texel boundaries take 2 x 3 x 0.05 = 30% of each axis at most: more than half of the tile pixels are compared.
        check.expect(tile_bad == 0 && tile_probes > tile_area / 2,
                     std::format("30 layer-face tiles: {} pixels (texel boundaries excluded) show the upright, unmirrored 4 x 4 image of layer-face z = 6 layer + face, through the plain texture() function", tile_probes),
                     std::format("{} wrong; first: {}", tile_bad, first));
        // Negative control: wrong models (mirror, transpose, face/layer swap, layer + 1, level 1 data) must disagree with most of the tile pixels.
        bool controls_ok = true;
        std::string control_detail;
        for (std::size_t k = 0; k < wrong.size() - 1; ++k)  // the last one (level 0 instead of the lookup's level) is the identity for level-0 tiles
        {
            controls_ok = controls_ok && control_equal[k] < tile_probes / 2;
            control_detail += std::format("{} ", control_equal[k]);
        }
        check.expect(controls_ok, "negative control: models with t mirrored, s mirrored, transposed, face and layer swapped, or layer + 1 reproduce fewer than half of the tile pixels",
                     std::format("equal pixels per wrong model: {} of {}", control_detail, tile_probes));

        // ---- the five spheres -----------------------------------------------------------------------------
        const double t = static_cast<double>(shown_frame_) / 60.0;
        const auto centres = sphere_centres(t);
        constexpr double kRadius = 0.8;
        const dm::Vec3 eye_f = eye_position();
        const V3 eye{eye_f.x, eye_f.y, eye_f.z};
        const V3 forward = normalize(V3{0, 0, 0} - eye);
        const V3 right = normalize(cross(forward, V3{0, 1, 0}));
        const V3 up = cross(right, forward);
        const double aspect = static_cast<double>(lay.width) / lay.scene_h;
        const double tan_half = std::tan(lay.fovy * 0.5);
        int compared = 0, skipped = 0, bad = 0, layer_plus_one_equal = 0;
        std::array<int, 5> per_sphere{}, hits{};
        std::array<int, kFaces> per_face{};
        std::string sfirst;
        constexpr double kPerturb = 0.012;  // radians; the tessellated sphere's interpolated normals deviate from the analytic ones by far less
        for (int y = 1; y < lay.scene_h; y += 2)  // y counted from the top of the scene viewport
            for (int x = 1; x < lay.width; x += 2)
            {
                const double nx = (x + 0.5) / lay.width * 2.0 - 1.0, ny = 1.0 - (y + 0.5) / lay.scene_h * 2.0;
                const V3 dir = normalize(forward + right * (nx * tan_half * aspect) + up * (ny * tan_half));
                const SphereHit h = hit_spheres(centres, kRadius, eye, dir);
                if (h.id < 0) continue;
                ++hits[static_cast<std::size_t>(h.id)];
                const V3 incident = normalize(h.point - eye);
                if (-dot(incident, h.normal) < 0.25)
                {
                    ++skipped;  // grazing: the reflection direction changes too fast
                    continue;
                }
                // The expected texel must be the same for the analytic normal and for normals tilted by +-kPerturb about two axes.
                const V3 t1 = normalize(cross(h.normal, std::abs(h.normal.y) < 0.9 ? V3{0, 1, 0} : V3{1, 0, 0}));
                const V3 t2 = cross(h.normal, t1);
                bool stable = true;
                Sample S{};
                for (int variant = 0; variant < 5 && stable; ++variant)
                {
                    V3 n = h.normal;
                    if (variant == 1) n = normalize(h.normal + t1 * kPerturb);
                    if (variant == 2) n = normalize(h.normal - t1 * kPerturb);
                    if (variant == 3) n = normalize(h.normal + t2 * kPerturb);
                    if (variant == 4) n = normalize(h.normal - t2 * kPerturb);
                    const Sample v = model_lookup(as_float(reflect(incident, n)), static_cast<double>(h.id), 0.0, kSize, kLevels, kLayers);
                    if (v.ambiguous) stable = false;
                    if (variant == 0) S = v;
                    else if (!(v == S)) stable = false;
                }
                if (!stable)
                {
                    ++skipped;
                    continue;
                }
                ++compared;
                ++per_sphere[static_cast<std::size_t>(h.id)];
                ++per_face[static_cast<std::size_t>(S.face)];
                const Rgba8 got = at(x, y);
                if (got != sample_color(S))
                {
                    if (bad == 0) sfirst = std::format("pixel ({}, {}) on sphere {}: got {} expected layer {} face {} texel ({},{}) = {}", x, y, h.id, demo::format_rgba(got), S.layer, kFaceName[static_cast<std::size_t>(S.face)], S.i, S.j, demo::format_rgba(sample_color(S)));
                    ++bad;
                }
                if (got == sample_color(wrong_layer_plus_one(S))) ++layer_plus_one_equal;
            }
        // Every sphere must be in the picture (the ray model hits it at >= 40 probe positions) and at least a quarter of its
        // probes must survive the grazing / texel-boundary filter: a bound that scales with the sphere's size on screen.
        bool every_sphere = true;
        for (std::size_t k = 0; k < 5; ++k) every_sphere = every_sphere && hits[k] >= 40 && per_sphere[k] >= hits[k] / 4;
        int faces_seen = 0;
        for (int c : per_face) faces_seen += c > 20 ? 1 : 0;
        std::cout << std::format("   [note] spheres: {} probes compared, {} skipped (grazing / near texel boundaries); per sphere {}/{} {}/{} {}/{} {}/{} {}/{} (compared/hit); per face {} {} {} {} {} {}\n", compared, skipped,
                                 per_sphere[0], hits[0], per_sphere[1], hits[1], per_sphere[2], hits[2], per_sphere[3], hits[3], per_sphere[4], hits[4],
                                 per_face[0], per_face[1], per_face[2], per_face[3], per_face[4], per_face[5]);
        check.expect(bad == 0 && every_sphere && faces_seen >= 4,
                     std::format("environment-mapped spheres: {} pixel probes equal texture(reflect(view, normal), k) of the model, sphere k showing layer k (all five spheres, {} different faces)", compared, faces_seen),
                     std::format("{} wrong of {}; first: {}", bad, compared, sfirst));
        check.expect(layer_plus_one_equal < compared / 4, "negative control: a model that lets sphere k reflect layer k + 1 reproduces fewer than a quarter of the sphere pixels",
                     std::format("{} of {} equal", layer_plus_one_equal, compared));

        render(env, shown_frame_);  // leave the final picture in the framebuffer for the capture
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-cube-map-array",
        .title = "Cube map array textures",
        .min_es = {3, 2},
        .summary = "TEXTURE_CUBE_MAP_ARRAY from glTexStorage3D (depth = 6 x layers) filled with glTexSubImage3D per layer-face, samplerCubeArray / isamplerCubeArray / usamplerCubeArray / "
                   "samplerCubeArrayShadow lookups with a (direction, layer) coordinate, mip levels, imageCubeArray; five environment-mapped spheres, each reflecting its own layer.",
        .controls = "a/d: orbit, w/s: pitch, q/e: zoom",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 20,
    };
    return demo::run<CubeMapArray>(argc, argv, info);
}
