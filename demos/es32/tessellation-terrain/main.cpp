// es32-tessellation-terrain
//
// A 3D terrain drawn with tessellation shaders (OpenGL ES 3.2 section 11.2): 16 x 16 quad patches cover a 64 x 64 world
// unit landscape. The tessellation control shader chooses the level of detail of every patch edge from its distance to
// the camera (crack-free: an edge's level depends only on the edge), the tessellation evaluation shader displaces the
// generated vertices with a procedural height map (an R16F texture fetched in the TES), the fragment shader lights each pixel
// from the same height map. fractional_odd_spacing makes the mesh refine continuously. A camera flies over the terrain; in
// a window the keys switch a wireframe overlay (made by a geometry shader), a tint that shows the tessellation level,
// a frozen LOD viewpoint and the primitive bounding box mode.
//
// Validation (all against CPU models of the shaders, which only use +, *, mix, smoothstep, normalize and length):
//   * limits: MAX_TESS_GEN_LEVEL, MAX_PATCH_VERTICES and all MAX_TESS_* of ES 3.2 Tables 21.45 / 21.46 against their minimums;
//     TESS_CONTROL_OUTPUT_VERTICES and the TES layout through glGetProgramiv;
//   * transform feedback captures the TES output of the frame: every patch's levels against the CPU LOD formula, the number of
//     triangles (TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = PRIMITIVES_GENERATED) against the section 11.2.2 count for the rounded
//     odd levels, the height of every vertex against the CPU height map, and that neighbouring patches tessellate their
//     shared edge identically (no cracks);
//   * the LOD really depends on the distance: three viewpoints of the same patches give strictly decreasing triangle counts,
//     each equal to the CPU count;
//   * pixels: the captured triangles are ray-cast on the CPU; every pixel away from silhouettes and triangle edges is compared
//     with the CPU lighting model (material by height and slope, sun, fog), sky pixels with the sky gradient, and a mutated model
//     (mirrored sun) must fail;
//   * primitive bounding box (section 13.2): the state is queried, and the image is bit-identical with the default box, with a
//     loose box set by glPrimitiveBoundingBox, and with a tight per-patch box written to gl_BoundingBox by the TCS while the
//     state box is absurdly small (gl_BoundingBox overrides it). A different camera must give a different image (the comparison
//     can fail).

#include <demo/check.hpp>
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
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    // =============================================================================================================
    // World and height map (CPU)
    // =============================================================================================================
    constexpr int kHeightmapN = 256;          // lattice points per edge
    constexpr float kHalfSize = 32.0f;        // the terrain covers x, z in [-32, 32]
    constexpr double kHalfSizeD = 32.0;
    constexpr int kPatchesPerEdge = 16;       // 16 x 16 patches of 4 x 4 world units
    constexpr float kPatchSize = 2.0f * kHalfSize / static_cast<float>(kPatchesPerEdge);
    constexpr int kPatchCount = kPatchesPerEdge * kPatchesPerEdge;
    constexpr float kLodScaleDefault = 80.0f;
    constexpr float kFovDegrees = 60.0f;
    constexpr double kStartTime = 8.0;        // seconds into the fly-over at frame 0 (the captured view)

    using V3 = std::array<double, 3>;
    V3 operator+(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
    V3 operator-(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
    V3 operator*(const V3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
    double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    V3 cross(const V3& a, const V3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
    double length(const V3& a) { return std::sqrt(dot(a, a)); }
    V3 normalize(const V3& a) { return a * (1.0 / length(a)); }
    double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }
    double mixd(double a, double b, double t) { return a * (1.0 - t) + b * t; }
    V3 mixv(const V3& a, const V3& b, double t) { return {mixd(a[0], b[0], t), mixd(a[1], b[1], t), mixd(a[2], b[2], t)}; }
    double smoothstep(double e0, double e1, double x)
    {
        const double t = clamp01((x - e0) / (e1 - e0));
        return t * t * (3.0 - 2.0 * t);
    }

    // IEEE binary16 conversion with round-to-nearest-even; the height map is uploaded as HALF_FLOAT, so the texture holds exactly
    // these bits and the CPU model reads exactly these values.
    std::uint16_t float_to_half(float f)
    {
        const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
        const std::uint32_t sign = (x >> 16) & 0x8000u;
        const std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        const std::uint32_t mant = x & 0x7FFFFFu;
        if (exp <= 0)
        {
            if (exp < -10) return static_cast<std::uint16_t>(sign);
            const std::uint32_t m = mant | 0x800000u;
            const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
            std::uint32_t half = m >> shift;
            const std::uint32_t rem = m & ((1u << shift) - 1u);
            const std::uint32_t mid = 1u << (shift - 1u);
            if (rem > mid || (rem == mid && (half & 1u) != 0u)) ++half;
            return static_cast<std::uint16_t>(sign | half);
        }
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
        std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
        const std::uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (half & 1u) != 0u)) ++half;
        return static_cast<std::uint16_t>(sign | half);
    }

    float half_to_float(std::uint16_t h)
    {
        const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
        const std::uint32_t exp = (h >> 10) & 0x1Fu;
        const std::uint32_t mant = h & 0x3FFu;
        if (exp == 0)
        {
            const float v = std::ldexp(static_cast<float>(mant), -24);
            return (sign != 0) ? -v : v;
        }
        return std::bit_cast<float>(sign | ((exp + 112u) << 23) | (mant << 13));
    }

    // Value noise on an integer hash: deterministic everywhere (no libm functions in the generator except floor).
    std::uint32_t hash2(int x, int y)
    {
        std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393u + static_cast<std::uint32_t>(y) * 668265263u;
        h = (h ^ (h >> 13)) * 1274126177u;
        return h ^ (h >> 16);
    }
    double lattice_noise(int x, int y) { return static_cast<double>(hash2(x, y) & 0xFFFFFFu) / 16777216.0; }
    double value_noise(double x, double y)
    {
        const double fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        auto fade = [](double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); };
        const double tx = fade(x - fx), ty = fade(y - fy);
        return mixd(mixd(lattice_noise(ix, iy), lattice_noise(ix + 1, iy), tx), mixd(lattice_noise(ix, iy + 1), lattice_noise(ix + 1, iy + 1), tx), ty);
    }
    double fbm(double x, double y)
    {
        double sum = 0.0, amp = 0.5, freq = 1.0;
        for (int o = 0; o < 6; ++o)
        {
            sum += amp * value_noise(x * freq + 31.7 * o, y * freq - 17.3 * o);
            freq *= 2.0;
            amp *= 0.5;
        }
        return sum;
    }

    struct Heightmap
    {
        std::vector<float> lattice;  // after the round trip through binary16

        void generate()
        {
            lattice.assign(static_cast<std::size_t>(kHeightmapN) * kHeightmapN, 0.0f);
            for (int j = 0; j < kHeightmapN; ++j)
                for (int i = 0; i < kHeightmapN; ++i)
                {
                    const double x = 4.0 * i / (kHeightmapN - 1), y = 4.0 * j / (kHeightmapN - 1);
                    const double n = fbm(x, y);
                    const double m = smoothstep(0.24, 0.76, n);
                    const double ridge = 1.0 - std::abs(2.0 * clamp01(fbm(x * 0.7 + 11.0, y * 0.7 - 5.0)) - 1.0);
                    const double h = std::max(8.0 * (0.72 * m * m + 0.28 * ridge * ridge * m), 0.6);   // the lowest parts are flat lakes at height 0.6
                    lattice[static_cast<std::size_t>(j) * kHeightmapN + static_cast<std::size_t>(i)] = half_to_float(float_to_half(static_cast<float>(h)));
                }
        }
        std::vector<std::uint16_t> half_bits() const
        {
            std::vector<std::uint16_t> out;
            out.reserve(lattice.size());
            for (const float f : lattice) out.push_back(float_to_half(f));
            return out;
        }
        float at(int i, int j) const
        {
            i = std::clamp(i, 0, kHeightmapN - 1);
            j = std::clamp(j, 0, kHeightmapN - 1);
            return lattice[static_cast<std::size_t>(j) * kHeightmapN + static_cast<std::size_t>(i)];
        }
        // height_at() of terrain_common.glsl.
        double height(double x, double z) const
        {
            const double last = kHeightmapN - 1;
            const double scale = last / (2.0 * kHalfSizeD);
            const double fx = std::clamp((x + kHalfSizeD) * scale, 0.0, last - 0.001);
            const double fz = std::clamp((z + kHalfSizeD) * scale, 0.0, last - 0.001);
            const int ix = static_cast<int>(std::floor(fx)), iz = static_cast<int>(std::floor(fz));
            const double wx = fx - ix, wz = fz - iz;
            const double h0 = mixd(at(ix, iz), at(ix + 1, iz), wx);
            const double h1 = mixd(at(ix, iz + 1), at(ix + 1, iz + 1), wx);
            return mixd(h0, h1, wz);
        }
        // terrain_normal()
        V3 normal(double x, double z) const
        {
            const double d = 2.0 * kHalfSizeD / (kHeightmapN - 1);
            return normalize({height(x - d, z) - height(x + d, z), 2.0 * d, height(x, z - d) - height(x, z + d)});
        }
        // terrain_shade(): `sun` is a parameter only so that the validation can run a mutated model.
        V3 shade(const V3& world, const V3& eye, const V3& sun) const
        {
            const double h = height(world[0], world[2]);
            V3 n = normal(world[0], world[2]);
            const double water = 1.0 - smoothstep(0.62, 0.95, h);
            n = normalize(mixv(n, {0.0, 1.0, 0.0}, water));
            const double slope = 1.0 - n[1];
            V3 albedo = mixv({0.74, 0.68, 0.48}, {0.22, 0.40, 0.13}, smoothstep(0.7, 1.6, h));
            albedo = mixv(albedo, {0.45, 0.40, 0.35}, smoothstep(0.12, 0.30, slope));
            albedo = mixv(albedo, {0.96, 0.97, 1.00}, smoothstep(4.6, 5.8, h) * (1.0 - smoothstep(0.30, 0.55, slope)));
            albedo = mixv(albedo, {0.08, 0.28, 0.42}, water);
            const double diffuse = std::max(dot(n, normalize(sun)), 0.0);
            const V3 lit = albedo * (0.26 + 0.74 * diffuse);
            const double fog = clamp01((length(world - eye) - 30.0) / 90.0);
            return mixv(lit, {0.72, 0.82, 0.93}, fog);
        }
    };

    V3 sky_color(double y01)
    {
        const double t = clamp01(y01 * 1.25 - 0.1);
        return mixv({0.72, 0.82, 0.93}, {0.18, 0.38, 0.72}, t * t);
    }

    constexpr V3 kSun = {0.60, 0.50, 0.35};

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(clamp01(v) * 255.0)); }

    // =============================================================================================================
    // Tessellation model (ES 3.2 section 11.2.2) for quads with fractional_odd_spacing
    // =============================================================================================================
    // "fractional_odd_spacing: the tessellation level is clamped to the range [1, max - 1] and then rounded up to the nearest odd integer."
    int odd_round(double f, int max_level)
    {
        const int n = static_cast<int>(std::ceil(std::clamp(f, 1.0, static_cast<double>(max_level) - 1.0)));
        return n % 2 == 1 ? n : n + 1;
    }

    // Distance of a level to the nearest odd integer: the rounding boundaries. Levels at or below 1 and at or above max - 1 are clamped
    // and never ambiguous.
    double odd_margin(double f, int max_level)
    {
        if (f <= 1.0 || f >= static_cast<double>(max_level) - 1.0) return 1e9;
        const double k = 2.0 * std::round((f - 1.0) / 2.0) + 1.0;
        return std::abs(f - k);
    }

    // Triangles of a quad patch whose outer / inner levels round to these segment counts: with B boundary vertices and V vertices
    // in total, a triangulated disk has 2 V - B - 2 triangles (Euler); V = B + (a-1)(b-1) from the inner mesh (section 11.2.2.2).
    // If every level is one the patch is a single triangle pair; an inner level of one next to a larger outer level counts as
    // 1 + epsilon, which odd spacing rounds up to 3.
    int quad_triangles(std::array<int, 4> outer, std::array<int, 2> inner)
    {
        const bool all_one = outer[0] == 1 && outer[1] == 1 && outer[2] == 1 && outer[3] == 1 && inner[0] == 1 && inner[1] == 1;
        if (!all_one)
            for (int& i : inner)
                if (i == 1) i = 3;
        const int boundary = outer[0] + outer[1] + outer[2] + outer[3];
        const int vertices = boundary + (inner[0] - 1) * (inner[1] - 1);
        return 2 * vertices - boundary - 2;
    }

    struct Pose
    {
        V3 eye;
        V3 target;
    };

    struct PatchLevels
    {
        std::array<double, 4> outer;
        std::array<double, 2> inner;
    };

    // =============================================================================================================
    // GL side
    // =============================================================================================================
    struct Record
    {
        float pos[4];    // world position, gl_PrimitiveID
        float uv[4];     // gl_TessCoord.xy, patch level
        float outer[4];  // gl_TessLevelOuter
        float inner[4];  // gl_TessLevelInner
    };
    static_assert(sizeof(Record) == 64);

    struct Prog
    {
        mgl::ProgramId id{};
        mgl::UniformLocation viewproj, lod_eye, lod_scale, max_level, height_range, cam_eye, tint, heightmap, viewport;
    };

    struct FrameParams
    {
        dm::Mat4 viewproj;
        V3 cam_eye{};
        V3 lod_eye{};
        float lod_scale = kLodScaleDefault;
        float tint = 0.0f;
        int width = 256;
        int height = 256;
        Pose pose{};
    };

    struct Capture
    {
        std::vector<Record> records;
        GLuint written = 0;
        GLuint generated = 0;
    };

    std::string with_inserted(const std::string& source, const std::string& text)
    {
        const std::size_t nl = source.find('\n');
        return source.substr(0, nl + 1) + text + source.substr(nl + 1);
    }

    class TessellationTerrain final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            max_level_gl_ = demo::get_integer(mgl::GetParameter::MaxTessGenLevel);
            max_level_ = std::min(16, max_level_gl_ - 1);
            terrain_.generate();
            const auto [lo, hi] = std::minmax_element(terrain_.lattice.begin(), terrain_.lattice.end());
            height_lo_ = *lo - 0.01f;
            height_hi_ = *hi + 0.01f;

            // Height map: one R16F texel per lattice point, uploaded as HALF_FLOAT (the exact bits of the CPU model).
            // NEAREST filtering: texelFetch ignores the filter, but an incomplete texture (the default mipmapped minification filter with a
            // single level) would fetch zeros.
            mgl::glGenTextures(1, &heightmap_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, heightmap_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::R16F, kHeightmapN, kHeightmapN);
            const std::vector<std::uint16_t> bits = terrain_.half_bits();
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kHeightmapN, kHeightmapN, mgl::PixelFormat::Red, mgl::PixelType::HalfFloat, bits.data());
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);

            // Patch corners: for every patch (pi, pj) the four vertices (x0,z0), (x1,z0), (x0,z1), (x1,z1) = (u,v) = (0,0), (1,0), (0,1), (1,1).
            std::vector<float> corners;
            for (int pj = 0; pj < kPatchesPerEdge; ++pj)
                for (int pi = 0; pi < kPatchesPerEdge; ++pi)
                {
                    const float x0 = -kHalfSize + static_cast<float>(pi) * kPatchSize, x1 = x0 + kPatchSize;
                    const float z0 = -kHalfSize + static_cast<float>(pj) * kPatchSize, z1 = z0 + kPatchSize;
                    for (const float c : {x0, z0, x1, z0, x0, z1, x1, z1}) corners.push_back(c);
                }
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(corners), mgl::BufferUsage::StaticDraw);
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glGenVertexArrays(1, &vao_empty_);   // the sky triangle has no attributes; core ES needs a bound VAO object only for attributes

            mgl::glGenBuffers(1, &tf_buffer_);
            mgl::glGenQueries(1, &q_written_);
            mgl::glGenQueries(1, &q_generated_);

            // Programs.
            const std::string common = env.load_text("shaders/terrain_common.glsl");
            const std::string defs = std::format("#define HEIGHTMAP_N {}\n#define TERRAIN_HALF {:.1f}\n", kHeightmapN, static_cast<double>(kHalfSize));
            const std::string vert = env.load_text("shaders/terrain.vert");
            const std::string tesc = env.load_text("shaders/terrain.tesc");
            const std::string tese = env.load_text("shaders/terrain.tese");
            const std::string frag = env.load_text("shaders/terrain.frag");
            const std::string geom = env.load_text("shaders/wire.geom");
            const std::string wire_frag = env.load_text("shaders/wire.frag");
            const std::string sky_vert = env.load_text("shaders/sky.vert");
            const std::string sky_frag = env.load_text("shaders/sky.frag");
            auto with_common = [&](const std::string& src, const std::string& extra = {}) { return with_inserted(src, defs + extra + common); };
            static const GLchar* const varyings[] = {"tf_pos", "tf_uv", "tf_outer", "tf_inner"};

            main_ = build({{mgl::ShaderType::Vertex, vert}, {mgl::ShaderType::TessControl, with_common(tesc)}, {mgl::ShaderType::TessEvaluation, with_common(tese)},
                           {mgl::ShaderType::Fragment, with_common(frag)}}, varyings, "terrain");
            bbox_ = build({{mgl::ShaderType::Vertex, vert}, {mgl::ShaderType::TessControl, with_common(tesc, "#define WRITE_BOUNDING_BOX 1\n")},
                           {mgl::ShaderType::TessEvaluation, with_common(tese)}, {mgl::ShaderType::Fragment, with_common(frag)}}, {}, "terrain with gl_BoundingBox");
            wire_ = build({{mgl::ShaderType::Vertex, vert}, {mgl::ShaderType::TessControl, with_common(tesc)}, {mgl::ShaderType::TessEvaluation, with_common(tese)},
                           {mgl::ShaderType::Geometry, geom}, {mgl::ShaderType::Fragment, wire_frag}}, {}, "terrain wireframe");
            sky_ = build({{mgl::ShaderType::Vertex, sky_vert}, {mgl::ShaderType::Fragment, with_common(sky_frag)}}, {}, "sky");
        }

        void render(demo::Env& env, int frame) override
        {
            last_frame_ = frame;
            if (env.interactive)
            {
                if (!paused_) anim_time_ += 1.0 / 60.0;
            }
            else anim_time_ = kStartTime + static_cast<double>(frame) / 60.0;
            shown_ = make_params(env.width, env.height, fly_pose(anim_time_), anim_time_);
            if (env.interactive && freeze_lod_) shown_.lod_eye = frozen_lod_eye_;
            else frozen_lod_eye_ = shown_.lod_eye;
            shown_.tint = tint_ ? 1.0f : 0.0f;
            draw_scene(shown_, wire_mode_, bbox_mode_);
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
            case 'w': wire_mode_ = (wire_mode_ + 1) % 3; break;
            case 'l': tint_ = !tint_; break;
            case 'b': bbox_mode_ = (bbox_mode_ + 1) % 3; break;
            case 'f': freeze_lod_ = !freeze_lod_; break;
            case ' ': paused_ = !paused_; break;
            case '+':
            case '=': lod_scale_ *= 1.2f; break;
            case '-': lod_scale_ /= 1.2f; break;
            case 'r': lod_scale_ = kLodScaleDefault; wire_mode_ = 1; tint_ = false; bbox_mode_ = 0; freeze_lod_ = false; break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (const Prog& p : {main_, bbox_, wire_, sky_}) mgl::glDeleteProgram(p.id);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteBuffers(1, &tf_buffer_);
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteVertexArrays(1, &vao_empty_);
            mgl::glDeleteQueries(1, &q_written_);
            mgl::glDeleteQueries(1, &q_generated_);
            mgl::glDeleteTextures(1, &heightmap_);
        }

    private:
        // -----------------------------------------------------------------------------------------------------
        // Programs
        // -----------------------------------------------------------------------------------------------------
        Prog build(std::vector<std::pair<mgl::ShaderType, std::string>> stages, std::span<const GLchar* const> tf_varyings, const char* label)
        {
            std::vector<mgl::ShaderId> shaders;
            for (const auto& [type, src] : stages)
            {
                std::string log;
                const mgl::ShaderId s = demo::compile_shader(type, src, log);
                if (s.value == 0) throw demo::Failure(std::format("{}: a shader failed to compile:\n{}", label, log));
                shaders.push_back(s);
            }
            const mgl::ProgramId program = mgl::glCreateProgram();
            for (const mgl::ShaderId s : shaders) mgl::glAttachShader(program, s);
            if (!tf_varyings.empty())
                mgl::glTransformFeedbackVaryings(program, static_cast<GLsizei>(tf_varyings.size()), tf_varyings.data(), mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(program);
            GLint ok = GL_FALSE;
            mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &ok);
            const std::string log = demo::program_info_log(program);
            for (const mgl::ShaderId s : shaders)
            {
                mgl::glDetachShader(program, s);
                mgl::glDeleteShader(s);
            }
            if (ok != GL_TRUE) throw demo::Failure(std::format("{}: program failed to link:\n{}", label, log));
            Prog p;
            p.id = program;
            auto loc = [&](const char* name) { return mgl::glGetUniformLocation(program, name); };
            p.viewproj = loc("u_viewproj");
            p.lod_eye = loc("u_lod_eye");
            p.lod_scale = loc("u_lod_scale");
            p.max_level = loc("u_max_level");
            p.height_range = loc("u_height_range");
            p.cam_eye = loc("u_cam_eye");
            p.tint = loc("u_tint");
            p.heightmap = loc("u_heightmap");
            p.viewport = loc("u_viewport");
            return p;
        }

        void set_uniforms(const Prog& p, const FrameParams& fp) const
        {
            mgl::glUniformMatrix4fv(p.viewproj, 1, GL_FALSE, fp.viewproj.data());
            mgl::glUniform3f(p.lod_eye, static_cast<float>(fp.lod_eye[0]), static_cast<float>(fp.lod_eye[1]), static_cast<float>(fp.lod_eye[2]));
            mgl::glUniform1f(p.lod_scale, fp.lod_scale);
            mgl::glUniform1f(p.max_level, static_cast<float>(max_level_));
            mgl::glUniform2f(p.height_range, height_lo_, height_hi_);
            mgl::glUniform3f(p.cam_eye, static_cast<float>(fp.cam_eye[0]), static_cast<float>(fp.cam_eye[1]), static_cast<float>(fp.cam_eye[2]));
            mgl::glUniform1f(p.tint, fp.tint);
            mgl::glUniform1i(p.heightmap, 0);   // texture unit 0
        }

        // -----------------------------------------------------------------------------------------------------
        // Camera
        // -----------------------------------------------------------------------------------------------------
        // The fly-over: an orbit around the terrain at a varying radius, always a few units above the ground, looking ahead and down.
        Pose fly_pose(double t) const
        {
            const double a = 0.16 * t;
            const double r = 27.0 + 4.0 * std::sin(0.31 * t);
            const double ex = r * std::cos(a), ez = r * std::sin(a);
            const double ey = terrain_.height(ex, ez) + 9.0 + 1.5 * std::sin(0.7 * t);
            const double b = a + 1.1;
            const double tx = 0.55 * r * std::cos(b), tz = 0.55 * r * std::sin(b);
            return {{ex, ey, ez}, {tx, terrain_.height(tx, tz) + 0.5, tz}};
        }

        FrameParams make_params(int width, int height, const Pose& pose, double /*t*/) const
        {
            FrameParams fp;
            fp.width = width;
            fp.height = height;
            fp.pose = pose;
            fp.cam_eye = pose.eye;
            fp.lod_eye = pose.eye;
            fp.lod_scale = lod_scale_;
            const float aspect = static_cast<float>(width) / static_cast<float>(height);
            auto f3 = [](const V3& v) { return dm::Vec3{static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])}; };
            fp.viewproj = dm::perspective(dm::radians(kFovDegrees), aspect, 0.5f, 400.0f) * dm::look_at(f3(pose.eye), f3(pose.target), {0.0f, 1.0f, 0.0f});
            return fp;
        }

        // -----------------------------------------------------------------------------------------------------
        // Drawing
        // -----------------------------------------------------------------------------------------------------
        void draw_patches() const
        {
            mgl::glBindVertexArray(vao_);
            mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 4);
            mgl::glDrawArrays(mgl::PrimitiveType::Patches, 0, kPatchCount * 4);
        }

        // bbox_mode: 0 = default box state, 1 = a loose correct box from glPrimitiveBoundingBox, 2 = the TCS writes a tight box to
        // gl_BoundingBox (and the box state is absurdly small, which gl_BoundingBox overrides), 3 = the absurdly small box state
        // alone (an incorrect box: only the validation uses this one, to print what the implementation does).
        void draw_scene(const FrameParams& fp, int wire_mode, int bbox_mode)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glViewport(0, 0, fp.width, fp.height);
            mgl::glClearDepthf(1.0f);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClear(mgl::ClearBufferBit::Depth);

            // Sky: a full-window triangle, no depth.
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glUseProgram(sky_.id);
            mgl::glUniform2f(sky_.viewport, static_cast<float>(fp.width), static_cast<float>(fp.height));
            mgl::glBindVertexArray(vao_empty_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            // Terrain.
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, heightmap_);
            if (bbox_mode == 1) mgl::glPrimitiveBoundingBox(-1e4f, -1e4f, -1e4f, 0.0f, 1e4f, 1e4f, 1e4f, 1e4f);
            if (bbox_mode >= 2) mgl::glPrimitiveBoundingBox(0.0f, 0.0f, 0.0f, 1.0f, 1e-3f, 1e-3f, 1e-3f, 1.0f);
            const Prog& p = bbox_mode == 2 ? bbox_ : main_;
            mgl::glUseProgram(p.id);
            set_uniforms(p, fp);
            draw_patches();
            mgl::glPrimitiveBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);   // the initial value (section 13.2, Table 21.6)

            // Wireframe overlay: the whole window, or its left half.
            if (wire_mode != 0)
            {
                mgl::glDepthFunc(mgl::CompareFunc::Lequal);
                if (wire_mode == 1)
                {
                    mgl::glEnable(mgl::Capability::ScissorTest);
                    mgl::glScissor(0, 0, fp.width / 2, fp.height);
                }
                mgl::glEnable(mgl::Capability::Blend);
                mgl::glBlendFunc(mgl::BlendFactor::SrcAlpha, mgl::BlendFactor::OneMinusSrcAlpha);
                mgl::glUseProgram(wire_.id);
                set_uniforms(wire_, fp);
                draw_patches();
                mgl::glDisable(mgl::Capability::Blend);
                mgl::glDisable(mgl::Capability::ScissorTest);
            }
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        // Captures the TES output (transform feedback) of one draw of all patches; no pixels are produced.
        Capture capture(const FrameParams& fp, std::size_t reserve_vertices)
        {
            Capture cap;
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, heightmap_);
            mgl::glUseProgram(main_.id);
            set_uniforms(main_, fp);
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, tf_buffer_);
            mgl::glBufferData(mgl::BufferTarget::TransformFeedback, static_cast<GLsizeiptr>((reserve_vertices + 64) * sizeof(Record)), nullptr, mgl::BufferUsage::StreamRead);
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(mgl::QueryTarget::PrimitivesGenerated, q_generated_);
            mgl::glBeginQuery(mgl::QueryTarget::TransformFeedbackPrimitivesWritten, q_written_);
            mgl::glBeginTransformFeedback(mgl::TransformFeedbackPrimitive::Triangles);
            draw_patches();
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(mgl::QueryTarget::TransformFeedbackPrimitivesWritten);
            mgl::glEndQuery(mgl::QueryTarget::PrimitivesGenerated);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glGetQueryObjectuiv(q_written_, mgl::QueryObjectParameter::Result, &cap.written);
            mgl::glGetQueryObjectuiv(q_generated_, mgl::QueryObjectParameter::Result, &cap.generated);
            const std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(cap.written) * 3u, reserve_vertices + 64);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::TransformFeedback, 0, static_cast<GLsizeiptr>(n * sizeof(Record)), mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                cap.records.resize(n);
                std::memcpy(cap.records.data(), mapped, n * sizeof(Record));
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::TransformFeedback);
            }
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            return cap;
        }

        // -----------------------------------------------------------------------------------------------------
        // CPU model of the TCS
        // -----------------------------------------------------------------------------------------------------
        static V3 patch_corner(int pi, int pj, int k)   // k = 0..3: (u,v) = (0,0), (1,0), (0,1), (1,1)
        {
            const double x = -kHalfSizeD + (pi + (k & 1)) * static_cast<double>(kPatchSize);
            const double z = -kHalfSizeD + (pj + (k >> 1)) * static_cast<double>(kPatchSize);
            return {x, 0.0, z};
        }

        double edge_level(const V3& a, const V3& b, const V3& lod_eye, double lod_scale) const
        {
            const double mx = 0.5 * (a[0] + b[0]), mz = 0.5 * (a[2] + b[2]);
            const V3 mid = {mx, terrain_.height(mx, mz), mz};
            return std::clamp(lod_scale / length(lod_eye - mid), 1.0, static_cast<double>(max_level_));
        }

        PatchLevels cpu_levels(int patch, const V3& lod_eye, double lod_scale) const
        {
            const int pi = patch % kPatchesPerEdge, pj = patch / kPatchesPerEdge;
            const V3 c0 = patch_corner(pi, pj, 0), c1 = patch_corner(pi, pj, 1), c2 = patch_corner(pi, pj, 2), c3 = patch_corner(pi, pj, 3);
            const double left = edge_level(c0, c2, lod_eye, lod_scale), bottom = edge_level(c0, c1, lod_eye, lod_scale);
            const double right = edge_level(c1, c3, lod_eye, lod_scale), top = edge_level(c2, c3, lod_eye, lod_scale);
            return {{left, bottom, right, top}, {std::max(bottom, top), std::max(left, right)}};
        }

        struct Expectation
        {
            std::size_t triangles = 0;
            double min_margin = 1e9;
            int max_patch_level = 0;
            int min_patch_level = 1000;
            std::vector<PatchLevels> levels;
        };

        Expectation expected(const V3& lod_eye, double lod_scale) const
        {
            Expectation e;
            for (int p = 0; p < kPatchCount; ++p)
            {
                const PatchLevels l = cpu_levels(p, lod_eye, lod_scale);
                e.levels.push_back(l);
                std::array<int, 4> outer{};
                std::array<int, 2> inner{};
                for (std::size_t i = 0; i < 4; ++i)
                {
                    outer[i] = odd_round(l.outer[i], max_level_gl_);
                    e.min_margin = std::min(e.min_margin, odd_margin(l.outer[i], max_level_gl_));
                    e.max_patch_level = std::max(e.max_patch_level, outer[i]);
                    e.min_patch_level = std::min(e.min_patch_level, outer[i]);
                }
                for (std::size_t i = 0; i < 2; ++i)
                {
                    inner[i] = odd_round(l.inner[i], max_level_gl_);
                    e.min_margin = std::min(e.min_margin, odd_margin(l.inner[i], max_level_gl_));
                }
                e.triangles += static_cast<std::size_t>(quad_triangles(outer, inner));
            }
            return e;
        }

        void check_limits(demo::Check& check);
        void check_geometry(demo::Check& check);
        void check_pixels(demo::Env& env, demo::Check& check);
        void check_bounding_box(demo::Check& check);

        int max_level_gl_ = 64;   // MAX_TESS_GEN_LEVEL
        int max_level_ = 16;      // the largest level the TCS asks for
        Heightmap terrain_;
        float height_lo_ = 0.0f, height_hi_ = 0.0f;
        float lod_scale_ = kLodScaleDefault;
        double anim_time_ = kStartTime;
        int last_frame_ = 0;
        int wire_mode_ = 1;       // 0 off, 1 left half, 2 whole window
        int bbox_mode_ = 0;
        bool tint_ = false;
        bool paused_ = false;
        bool freeze_lod_ = false;
        V3 frozen_lod_eye_{};
        FrameParams shown_{};

        Prog main_, bbox_, wire_, sky_;
        mgl::TextureId heightmap_{};
        mgl::BufferId vbo_{}, tf_buffer_{};
        mgl::VertexArrayId vao_{}, vao_empty_{};
        mgl::QueryId q_written_{}, q_generated_{};
    };

    // =============================================================================================================
    // validate()
    // =============================================================================================================
    void TessellationTerrain::validate(demo::Env& env, demo::Check& check)
    {
        check_limits(check);
        check_pixels(env, check);   // first: a clean frame of the displayed camera, before anything else is drawn
        check_geometry(check);
        check_bounding_box(check);
        mgl::glPatchParameteri(mgl::TessellationParameter::PatchVertices, 3);
        render(env, last_frame_);   // the displayed frame (wireframe on the left half) for the capture
    }

    // ---- limits and program queries -------------------------------------------------------------------------------
    void TessellationTerrain::check_limits(demo::Check& check)
    {
        struct Limit
        {
            mgl::GetParameter pname;
            const char* name;
            int minimum;
        };
        using P = mgl::GetParameter;
        const Limit generator[] = {
            {P::MaxTessGenLevel, "MAX_TESS_GEN_LEVEL", 64},
            {P::MaxPatchVertices, "MAX_PATCH_VERTICES", 32},
            {P::MaxTessPatchComponents, "MAX_TESS_PATCH_COMPONENTS", 120},
        };
        const Limit control[] = {
            {P::MaxTessControlUniformComponents, "MAX_TESS_CONTROL_UNIFORM_COMPONENTS", 1024},
            {P::MaxTessControlTextureImageUnits, "MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS", 16},
            {P::MaxTessControlOutputComponents, "MAX_TESS_CONTROL_OUTPUT_COMPONENTS", 64},
            {P::MaxTessControlTotalOutputComponents, "MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS", 2048},
            {P::MaxTessControlInputComponents, "MAX_TESS_CONTROL_INPUT_COMPONENTS", 64},
            {P::MaxTessControlUniformBlocks, "MAX_TESS_CONTROL_UNIFORM_BLOCKS", 12},
            {P::MaxTessControlAtomicCounterBuffers, "MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS", 0},
            {P::MaxTessControlAtomicCounters, "MAX_TESS_CONTROL_ATOMIC_COUNTERS", 0},
            {P::MaxTessControlShaderStorageBlocks, "MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS", 0},
            {P::MaxTessControlImageUniforms, "MAX_TESS_CONTROL_IMAGE_UNIFORMS", 0},
        };
        const Limit evaluation[] = {
            {P::MaxTessEvaluationUniformComponents, "MAX_TESS_EVALUATION_UNIFORM_COMPONENTS", 1024},
            {P::MaxTessEvaluationTextureImageUnits, "MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS", 16},
            {P::MaxTessEvaluationOutputComponents, "MAX_TESS_EVALUATION_OUTPUT_COMPONENTS", 64},
            {P::MaxTessEvaluationInputComponents, "MAX_TESS_EVALUATION_INPUT_COMPONENTS", 64},
            {P::MaxTessEvaluationUniformBlocks, "MAX_TESS_EVALUATION_UNIFORM_BLOCKS", 12},
            {P::MaxTessEvaluationAtomicCounterBuffers, "MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS", 0},
            {P::MaxTessEvaluationAtomicCounters, "MAX_TESS_EVALUATION_ATOMIC_COUNTERS", 0},
            {P::MaxTessEvaluationShaderStorageBlocks, "MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS", 0},
            {P::MaxTessEvaluationImageUniforms, "MAX_TESS_EVALUATION_IMAGE_UNIFORMS", 0},
        };
        auto group = [&](std::span<const Limit> limits, const char* what) {
            std::string bad;
            for (const Limit& l : limits)
            {
                const int v = demo::get_integer(l.pname);
                if (v < l.minimum) bad += std::format(" {} = {} < {};", l.name, v, l.minimum);
                std::cout << std::format("limit: {:<46} {:>8}   (minimum {})\n", l.name, v, l.minimum);
            }
            check.expect(bad.empty(), std::format("{} meet the minimums of ES 3.2 Tables 21.45 / 21.46 ({} limits)", what, limits.size()), bad);
        };
        group(generator, "the tessellation generator and patch limits (MAX_TESS_GEN_LEVEL, MAX_PATCH_VERTICES, MAX_TESS_PATCH_COMPONENTS)");
        group(control, "the tessellation control shader limits");
        group(evaluation, "the tessellation evaluation shader limits");

        // MAX_COMBINED_TESS_*_UNIFORM_COMPONENTS has a formula as minimum (Table 21.50): blocks x MAX_UNIFORM_BLOCK_SIZE / 4 + stage
        // components. The formula can exceed what a 32-bit integer holds; the expectation is capped at the largest 32-bit value.
        {
            GLint64 block_size = 0;
            mgl::glGetInteger64v(P::MaxUniformBlockSize, &block_size);
            constexpr std::int64_t kIntMax = 2147483647;
            auto minimum = [&](P blocks, P components) {
                return std::min<std::int64_t>(static_cast<std::int64_t>(demo::get_integer(blocks)) * block_size / 4 + demo::get_integer(components), kIntMax);
            };
            const std::int64_t tc = minimum(P::MaxTessControlUniformBlocks, P::MaxTessControlUniformComponents);
            const std::int64_t te = minimum(P::MaxTessEvaluationUniformBlocks, P::MaxTessEvaluationUniformComponents);
            GLint64 ctc = 0, cte = 0;
            mgl::glGetInteger64v(P::MaxCombinedTessControlUniformComponents, &ctc);
            mgl::glGetInteger64v(P::MaxCombinedTessEvaluationUniformComponents, &cte);
            check.expect(ctc >= tc && cte >= te, "MAX_COMBINED_TESS_{CONTROL,EVALUATION}_UNIFORM_COMPONENTS >= blocks x MAX_UNIFORM_BLOCK_SIZE / 4 + stage components (Table 21.50)",
                         std::format("control {} (min {}), evaluation {} (min {}), MAX_UNIFORM_BLOCK_SIZE {}", ctc, tc, cte, te, block_size));
        }
        // What this demo uses must fit: one texture unit in each of the TCS, TES and FS, and the LOD ceiling below MAX_TESS_GEN_LEVEL.
        check.expect(max_level_ >= 2 && max_level_ <= max_level_gl_ - 1,
                     std::format("the LOD ceiling ({}) is within [2, MAX_TESS_GEN_LEVEL - 1] = [2, {}]", max_level_, max_level_gl_ - 1));

        // glGetProgramiv: the TCS output patch size and the TES layout of the terrain program (section 7.12).
        {
            auto geti = [&](mgl::ProgramId p, mgl::ProgramParameter pn) {
                GLint x = -1;
                mgl::glGetProgramiv(p, pn, &x);
                return x;
            };
            bool ok = true;
            std::string detail;
            for (const mgl::ProgramId p : {main_.id, bbox_.id, wire_.id})
            {
                ok = ok && geti(p, mgl::ProgramParameter::TessControlOutputVertices) == 4 &&
                     static_cast<GLenum>(geti(p, mgl::ProgramParameter::TessGenMode)) == static_cast<GLenum>(mgl::TessGenMode::Quads) &&
                     static_cast<GLenum>(geti(p, mgl::ProgramParameter::TessGenSpacing)) == static_cast<GLenum>(mgl::TessGenSpacing::FractionalOdd) &&
                     static_cast<GLenum>(geti(p, mgl::ProgramParameter::TessGenVertexOrder)) == static_cast<GLenum>(mgl::FrontFace::CCW) &&
                     geti(p, mgl::ProgramParameter::TessGenPointMode) == GL_FALSE;
            }
            check.expect(ok, "the three terrain programs report TESS_CONTROL_OUTPUT_VERTICES = 4, TESS_GEN_MODE = QUADS, TESS_GEN_SPACING = FRACTIONAL_ODD, TESS_GEN_VERTEX_ORDER = CCW, TESS_GEN_POINT_MODE = FALSE");
        }
    }

    // ---- geometry: TF capture of the TES output ---------------------------------------------------------------------
    void TessellationTerrain::check_geometry(demo::Check& check)
    {
        FrameParams fp = shown_;
        fp.tint = 0.0f;
        const Expectation ex = expected(fp.lod_eye, fp.lod_scale);
        std::cout << std::format("note: displayed frame: {} triangles, patch edge segments from {} to {}, smallest distance of a level to an odd-rounding boundary {:.3g}\n",
                                 ex.triangles, ex.min_patch_level, ex.max_patch_level, ex.min_margin);
        check.expect(ex.min_margin > 1e-3, "model precondition: every level of the displayed frame is more than 1e-3 away from a rounding boundary of fractional_odd_spacing, so the CPU rounds like the GPU",
                     std::format("smallest margin {:.3g}", ex.min_margin));
        const Capture cap = capture(fp, ex.triangles * 3);
        check.expect(cap.written == ex.triangles && cap.generated == ex.triangles && cap.records.size() == ex.triangles * 3,
                     std::format("displayed frame: TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = PRIMITIVES_GENERATED = {} triangles, the section 11.2.2 count for the CPU-computed levels", ex.triangles),
                     std::format("{} written, {} generated, {} records", cap.written, cap.generated, cap.records.size()));
        if (cap.records.size() != ex.triangles * 3) return;

        // Per patch: the levels the TCS computed against the CPU LOD formula, and every vertex against the CPU terrain.
        struct PatchData
        {
            bool seen = false;
            std::array<float, 4> outer{};
            std::array<float, 2> inner{};
            float level = 0.0f;
            std::array<std::vector<std::array<double, 4>>, 4> edges;   // left (u=0), right (u=1), bottom (v=0), top (v=1): (param, x, y, z)
        };
        std::vector<PatchData> patches(kPatchCount);
        double worst_level = 0.0, worst_xz = 0.0, worst_y = 0.0, lo_y = 1e9, hi_y = -1e9;
        int bad_patch_id = 0;
        for (const Record& r : cap.records)
        {
            const int id = static_cast<int>(std::lround(r.pos[3]));
            if (id < 0 || id >= kPatchCount)
            {
                ++bad_patch_id;
                continue;
            }
            PatchData& pd = patches[static_cast<std::size_t>(id)];
            if (!pd.seen)
            {
                pd.seen = true;
                for (std::size_t i = 0; i < 4; ++i) pd.outer[i] = r.outer[i];
                pd.inner = {r.inner[0], r.inner[1]};
                pd.level = r.uv[2];
            }
            const int pi = id % kPatchesPerEdge, pj = id / kPatchesPerEdge;
            const V3 c0 = patch_corner(pi, pj, 0), c3 = patch_corner(pi, pj, 3);
            const double u = r.uv[0], v = r.uv[1];
            const double ex_x = mixd(c0[0], c3[0], u), ex_z = mixd(c0[2], c3[2], v);
            worst_xz = std::max({worst_xz, std::abs(static_cast<double>(r.pos[0]) - ex_x), std::abs(static_cast<double>(r.pos[2]) - ex_z)});
            const double ex_y = terrain_.height(r.pos[0], r.pos[2]);
            worst_y = std::max(worst_y, std::abs(static_cast<double>(r.pos[1]) - ex_y));
            lo_y = std::min<double>(lo_y, r.pos[1]);
            hi_y = std::max<double>(hi_y, r.pos[1]);
            const double eps = 1e-6;
            if (u <= eps) pd.edges[0].push_back({v, r.pos[0], r.pos[1], r.pos[2]});
            if (u >= 1.0 - eps) pd.edges[1].push_back({v, r.pos[0], r.pos[1], r.pos[2]});
            if (v <= eps) pd.edges[2].push_back({u, r.pos[0], r.pos[1], r.pos[2]});
            if (v >= 1.0 - eps) pd.edges[3].push_back({u, r.pos[0], r.pos[1], r.pos[2]});
        }
        int unseen = 0;
        for (int p = 0; p < kPatchCount; ++p)
        {
            if (!patches[static_cast<std::size_t>(p)].seen)
            {
                ++unseen;
                continue;
            }
            const PatchLevels& l = ex.levels[static_cast<std::size_t>(p)];
            const PatchData& pd = patches[static_cast<std::size_t>(p)];
            for (std::size_t i = 0; i < 4; ++i) worst_level = std::max(worst_level, std::abs(static_cast<double>(pd.outer[i]) - l.outer[i]) / l.outer[i]);
            for (std::size_t i = 0; i < 2; ++i) worst_level = std::max(worst_level, std::abs(static_cast<double>(pd.inner[i]) - l.inner[i]) / l.inner[i]);
        }
        check.expect(unseen == 0 && bad_patch_id == 0, "every one of the 256 patches produced triangles (no patch was discarded, gl_PrimitiveID covers 0..255)", std::format("{} patches without output, {} bad ids", unseen, bad_patch_id));
        // The shader divides by a distance: GLSL ES gives no accuracy for that, so this is a sanity bound.
        check.expect(worst_level < 1e-4, "the tessellation levels written by the TCS (gl_TessLevelOuter / Inner, seen in the TES) match the CPU formula clamp(scale / distance, 1, max)",
                     std::format("largest relative deviation {:.2g} (bound 1e-4)", worst_level));
        check.expect(worst_xz < 2e-4 && worst_y < 1e-3,
                     "every generated vertex lies at the CPU position: bilinear in the patch corners by gl_TessCoord, height from the CPU height map (displacement is analytic)",
                     std::format("largest xz deviation {:.2g}, height deviation {:.2g} (sanity bounds 2e-4 / 1e-3; float arithmetic, no accuracy is specified)", worst_xz, worst_y));
        check.expect(hi_y - lo_y > 3.0, "the displaced vertices span a real relief (not a flat plane)", std::format("heights {:.2f} .. {:.2f}", lo_y, hi_y));

        // Crack-free: a shared edge gets the same levels and the same vertices from both patches.
        {
            int pairs = 0, bad = 0, level_bad = 0;
            auto compare = [&](std::vector<std::array<double, 4>> a, std::vector<std::array<double, 4>> b) {
                auto order = [](const std::array<double, 4>& x, const std::array<double, 4>& y) { return x[0] < y[0]; };
                std::sort(a.begin(), a.end(), order);
                std::sort(b.begin(), b.end(), order);
                auto dedupe = [](std::vector<std::array<double, 4>>& v) {
                    v.erase(std::unique(v.begin(), v.end(), [](const auto& x, const auto& y) { return std::abs(x[0] - y[0]) < 1e-6; }), v.end());
                };
                dedupe(a);
                dedupe(b);
                if (a.size() != b.size()) return false;
                for (std::size_t i = 0; i < a.size(); ++i)
                    for (std::size_t k = 1; k < 4; ++k)
                        if (std::abs(a[i][k] - b[i][k]) > 1e-4) return false;
                return true;
            };
            for (int pj = 0; pj < kPatchesPerEdge; ++pj)
                for (int pi = 0; pi < kPatchesPerEdge; ++pi)
                {
                    const PatchData& a = patches[static_cast<std::size_t>(pj * kPatchesPerEdge + pi)];
                    if (pi + 1 < kPatchesPerEdge)
                    {
                        const PatchData& b = patches[static_cast<std::size_t>(pj * kPatchesPerEdge + pi + 1)];
                        ++pairs;
                        if (!compare(a.edges[1], b.edges[0])) ++bad;
                        if (a.outer[2] != b.outer[0]) ++level_bad;
                    }
                    if (pj + 1 < kPatchesPerEdge)
                    {
                        const PatchData& b = patches[static_cast<std::size_t>((pj + 1) * kPatchesPerEdge + pi)];
                        ++pairs;
                        if (!compare(a.edges[3], b.edges[2])) ++bad;
                        if (a.outer[3] != b.outer[1]) ++level_bad;
                    }
                }
            check.expect(bad == 0 && level_bad == 0, std::format("no cracks: all {} shared patch edges have equal outer levels and identical vertex positions on both sides", pairs),
                         std::format("{} edges with different vertices, {} with different levels", bad, level_bad));
        }

        // The level of detail depends on the distance: the same patches from three LOD viewpoints (the camera stays).
        {
            struct Eye
            {
                const char* name;
                double lift;
            };
            std::vector<std::size_t> totals;
            std::string list;
            bool counts_ok = true, margins_ok = true;
            int far_min_level = 0;
            for (const Eye& e : {Eye{"camera", 0.0}, Eye{"25 units higher", 25.0}, Eye{"120 units higher", 120.0}})
            {
                FrameParams f = fp;
                f.lod_eye = fp.cam_eye + V3{0.0, e.lift, 0.0};
                const Expectation x = expected(f.lod_eye, f.lod_scale);
                margins_ok = margins_ok && x.min_margin > 1e-3;
                const Capture c = capture(f, x.triangles * 3);
                counts_ok = counts_ok && c.written == x.triangles && c.generated == x.triangles;
                totals.push_back(c.written);
                list += std::format("{}{}: {} triangles", list.empty() ? "" : "; ", e.name, c.written);
                if (e.lift == 120.0) far_min_level = x.max_patch_level;
            }
            check.expect(margins_ok && counts_ok, "LOD viewpoints: the triangle count (written = generated) equals the CPU count for each, with no level near a rounding boundary", list);
            check.expect(totals[0] > totals[1] && totals[1] > totals[2], "moving the LOD viewpoint away from the terrain strictly reduces the number of triangles generated", list);
            check.expect(totals[2] == static_cast<std::size_t>(kPatchCount) * 2u && far_min_level == 1,
                         "from far away every patch collapses to level 1: a single triangle pair (256 patches, 512 triangles)", std::format("{} triangles, highest edge segment count {}", totals[2], far_min_level));
        }

        // The ceiling: a LOD viewpoint one unit above the terrain at the origin is about two units from the edges of the four patches
        // around it; clamp(80 / 2.2, 1, 16) = 16 for those edges.
        {
            FrameParams f = fp;
            f.lod_eye = {0.0, terrain_.height(0.0, 0.0) + 1.0, 0.0};
            const Expectation x = expected(f.lod_eye, f.lod_scale);
            const Capture c = capture(f, x.triangles * 3);
            float highest = 0.0f;
            for (const Record& r : c.records)
                for (const float o : r.outer) highest = std::max(highest, o);
            check.expect(x.min_margin > 1e-3 && c.written == x.triangles && c.generated == x.triangles && highest == static_cast<float>(max_level_) &&
                             x.max_patch_level == odd_round(max_level_, max_level_gl_),
                         std::format("a LOD viewpoint just above the terrain: the nearest edges are clamped to the ceiling (gl_TessLevelOuter = {}, {} segments), {} triangles as the CPU counts",
                                     max_level_, odd_round(max_level_, max_level_gl_), x.triangles),
                         std::format("highest gl_TessLevelOuter {}, {} triangles written, margin {:.3g}", highest, c.written, x.min_margin));
        }
    }

    // ---- pixels ------------------------------------------------------------------------------------------------------
    namespace
    {
        struct Tri
        {
            V3 a, b, c;
        };
        struct Hit
        {
            bool hit = false;
            double t = 0.0;
            double margin = 0.0;
            V3 world{};
        };

        class Caster
        {
        public:
            Caster(const FrameParams& fp, std::vector<Tri> tris) : fp_(fp), tris_(std::move(tris))
            {
                bins_.assign(static_cast<std::size_t>(kBins * kBins), {});
                for (std::size_t i = 0; i < tris_.size(); ++i)
                {
                    double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
                    bool unbounded = false;
                    for (const V3* v : {&tris_[i].a, &tris_[i].b, &tris_[i].c})
                    {
                        const auto clip = project(*v);
                        if (clip[3] < 0.05)
                        {
                            unbounded = true;
                            break;
                        }
                        const double sx = (clip[0] / clip[3] * 0.5 + 0.5) * fp.width, sy = (clip[1] / clip[3] * 0.5 + 0.5) * fp.height;
                        x0 = std::min(x0, sx);
                        x1 = std::max(x1, sx);
                        y0 = std::min(y0, sy);
                        y1 = std::max(y1, sy);
                    }
                    const int bw = fp.width / kBins, bh = fp.height / kBins;
                    const int bx0 = unbounded ? 0 : std::clamp(static_cast<int>(std::floor(x0 - 1.0)) / bw, 0, kBins - 1);
                    const int bx1 = unbounded ? kBins - 1 : std::clamp(static_cast<int>(std::floor(x1 + 1.0)) / bw, 0, kBins - 1);
                    const int by0 = unbounded ? 0 : std::clamp(static_cast<int>(std::floor(y0 - 1.0)) / bh, 0, kBins - 1);
                    const int by1 = unbounded ? kBins - 1 : std::clamp(static_cast<int>(std::floor(y1 + 1.0)) / bh, 0, kBins - 1);
                    if (!unbounded && (x1 < -1.0 || y1 < -1.0 || x0 > fp.width + 1.0 || y0 > fp.height + 1.0)) continue;
                    for (int by = by0; by <= by1; ++by)
                        for (int bx = bx0; bx <= bx1; ++bx) bins_[static_cast<std::size_t>(by * kBins + bx)].push_back(i);
                }
                eye_ = fp.pose.eye;
                forward_ = normalize(fp.pose.target - fp.pose.eye);
                right_ = normalize(cross(forward_, {0.0, 1.0, 0.0}));
                up_ = cross(right_, forward_);
                tan_half_ = std::tan(static_cast<double>(kFovDegrees) * 3.14159265358979323846 / 360.0);
                aspect_ = static_cast<double>(fp.width) / static_cast<double>(fp.height);
            }

            // The nearest triangle hit by the ray through the pixel centre (x + dx + 0.5, y + dy + 0.5).
            Hit cast(int x, int y, double dx = 0.0, double dy = 0.0) const
            {
                const double nx = (static_cast<double>(x) + dx + 0.5) / fp_.width * 2.0 - 1.0;
                const double ny = (static_cast<double>(y) + dy + 0.5) / fp_.height * 2.0 - 1.0;
                const V3 d = normalize(forward_ + right_ * (nx * tan_half_ * aspect_) + up_ * (ny * tan_half_));
                const int bw = fp_.width / kBins, bh = fp_.height / kBins;
                const int bx = std::clamp(x / bw, 0, kBins - 1), by = std::clamp(y / bh, 0, kBins - 1);
                Hit best;
                for (const std::size_t i : bins_[static_cast<std::size_t>(by * kBins + bx)])
                {
                    const Tri& t = tris_[i];
                    const V3 e1 = t.b - t.a, e2 = t.c - t.a;
                    const V3 p = cross(d, e2);
                    const double det = dot(e1, p);
                    if (std::abs(det) < 1e-14) continue;
                    const double inv = 1.0 / det;
                    const V3 s = eye_ - t.a;
                    const double u = dot(s, p) * inv;
                    if (u < 0.0 || u > 1.0) continue;
                    const V3 q = cross(s, e1);
                    const double v = dot(d, q) * inv;
                    if (v < 0.0 || u + v > 1.0) continue;
                    const double tt = dot(e2, q) * inv;
                    if (tt <= 0.0 || (best.hit && tt >= best.t)) continue;
                    best = {true, tt, std::min({1.0 - u - v, u, v}), eye_ + d * tt};
                }
                return best;
            }

        private:
            static constexpr int kBins = 8;
            std::array<double, 4> project(const V3& w) const
            {
                std::array<double, 4> out{};
                for (int r = 0; r < 4; ++r)
                    out[static_cast<std::size_t>(r)] = static_cast<double>(fp_.viewproj.at(r, 0)) * w[0] + static_cast<double>(fp_.viewproj.at(r, 1)) * w[1] +
                                                       static_cast<double>(fp_.viewproj.at(r, 2)) * w[2] + static_cast<double>(fp_.viewproj.at(r, 3));
                return out;
            }
            FrameParams fp_;
            std::vector<Tri> tris_;
            std::vector<std::vector<std::size_t>> bins_;
            V3 eye_{}, forward_{}, right_{}, up_{};
            double tan_half_ = 0.0, aspect_ = 1.0;
        };
    }

    void TessellationTerrain::check_pixels(demo::Env& env, demo::Check& check)
    {
        FrameParams fp = shown_;
        fp.tint = 0.0f;
        fp.width = env.width;
        fp.height = env.height;
        draw_scene(fp, 0, 0);   // no overlay: the pixels are the lit terrain and the sky
        mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, fp.width, fp.height);

        const Expectation ex = expected(fp.lod_eye, fp.lod_scale);
        const Capture cap = capture(fp, ex.triangles * 3);
        if (cap.records.size() != ex.triangles * 3)
        {
            check.expect(false, "pixel validation needs the captured triangles", std::format("{} records", cap.records.size()));
            return;
        }
        std::vector<Tri> tris;
        for (std::size_t i = 0; i + 2 < cap.records.size(); i += 3)
        {
            auto w = [&](std::size_t k) { return V3{cap.records[i + k].pos[0], cap.records[i + k].pos[1], cap.records[i + k].pos[2]}; };
            tris.push_back({w(0), w(1), w(2)});
        }
        const Caster caster(fp, std::move(tris));

        int terrain_probes = 0, sky_probes = 0, skipped = 0, bad_terrain = 0, bad_sky = 0, mutant_bad = 0;
        int worst = 0;
        const V3 mirrored_sun = {-kSun[0], kSun[1], kSun[2]};
        const int step = std::max(std::min(fp.width, fp.height) / 48, 1);
        for (int y = step / 2; y < fp.height; y += step)
            for (int x = step / 2; x < fp.width; x += step)
            {
                const std::uint8_t* got = &px[(static_cast<std::size_t>(y) * static_cast<std::size_t>(fp.width) + static_cast<std::size_t>(x)) * 4u];
                const Hit c = caster.cast(x, y);
                const Hit n[4] = {caster.cast(x, y, -1.0, 0.0), caster.cast(x, y, 1.0, 0.0), caster.cast(x, y, 0.0, -1.0), caster.cast(x, y, 0.0, 1.0)};
                auto compare = [&](const V3& want) {
                    int err = 0;
                    for (std::size_t k = 0; k < 3; ++k) err = std::max(err, std::abs(static_cast<int>(got[k]) - static_cast<int>(to8(want[k]))));
                    return err;
                };
                if (!c.hit && !n[0].hit && !n[1].hit && !n[2].hit && !n[3].hit)
                {
                    ++sky_probes;
                    const int err = compare(sky_color((static_cast<double>(y) + 0.5) / fp.height));
                    worst = std::max(worst, err);
                    if (err > 2) ++bad_sky;
                    continue;
                }
                bool smooth = c.hit && c.margin >= 0.05;
                for (const Hit& h : n) smooth = smooth && h.hit && std::abs(h.t - c.t) <= 0.1 * c.t;
                if (!smooth)
                {
                    ++skipped;   // silhouettes, triangle edges, occlusion boundaries
                    continue;
                }
                ++terrain_probes;
                const int err = compare(terrain_.shade(c.world, fp.cam_eye, kSun));
                worst = std::max(worst, err);
                if (err > 2) ++bad_terrain;
                if (compare(terrain_.shade(c.world, fp.cam_eye, mirrored_sun)) > 2) ++mutant_bad;
            }
        std::cout << std::format("note: pixel probes: {} terrain, {} sky, {} skipped (silhouettes and triangle edges); largest channel error {} / 255\n", terrain_probes, sky_probes, skipped, worst);
        check.expect(terrain_probes >= 200 && sky_probes >= 40, "the probe set is not vacuous: at least 200 terrain and 40 sky probes", std::format("{} terrain, {} sky", terrain_probes, sky_probes));
        // Tolerance 2 / 255: the GPU rounds a float colour to 8 bits (+-0.5) and the CPU model is evaluated in double instead of float.
        check.expect(bad_terrain == 0, std::format("{} terrain pixels away from silhouettes and triangle edges match the CPU lighting model (material by height and slope, sun, fog) within 2/255", terrain_probes),
                     std::format("{} probes differ", bad_terrain));
        check.expect(bad_sky == 0, std::format("{} sky pixels match the CPU sky gradient within 2/255", sky_probes), std::format("{} probes differ", bad_sky));
        check.expect(mutant_bad * 3 >= terrain_probes, "negative control: the same probes against a model with the sun mirrored disagree on at least a third of the pixels",
                     std::format("{} of {} probes differ", mutant_bad, terrain_probes));
    }

    // ---- primitive bounding box ----------------------------------------------------------------------------------------
    void TessellationTerrain::check_bounding_box(demo::Check& check)
    {
        auto box = [] {
            std::array<float, 8> b{};
            mgl::glGetFloatv(mgl::GetParameter::PrimitiveBoundingBox, b.data());
            return b;
        };
        const std::array<float, 8> initial = {-1, -1, -1, 1, 1, 1, 1, 1};
        check.expect(box() == initial, "PRIMITIVE_BOUNDING_BOX has its initial value (-1,-1,-1,1, 1,1,1,1) (ES 3.2 Table 21.6, section 13.2)");
        mgl::glPrimitiveBoundingBox(-2.0f, -3.0f, -4.0f, 0.5f, 5.0f, 6.0f, 7.0f, 8.0f);
        const std::array<float, 8> set = {-2, -3, -4, 0.5f, 5, 6, 7, 8};
        check.expect(box() == set, "glPrimitiveBoundingBox(-2,-3,-4,0.5, 5,6,7,8) is read back by glGetFloatv(PRIMITIVE_BOUNDING_BOX)");
        mgl::glPrimitiveBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

        FrameParams fp = shown_;
        fp.tint = 0.0f;
        auto image = [&](const FrameParams& f, int mode) {
            draw_scene(f, 0, mode);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return demo::read_rgba8(0, 0, f.width, f.height);
        };
        auto differing = [](const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
            int n = 0;
            for (std::size_t i = 0; i < a.size(); i += 4)
                if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2] || a[i + 3] != b[i + 3]) ++n;
            return n;
        };
        const auto base = image(fp, 0);
        const auto loose = image(fp, 1);
        const auto tight = image(fp, 2);
        check.expect(box() == initial, "draw_scene restored the initial box state after drawing with a custom box");
        check.expect(differing(base, loose) == 0,
                     "a correct (loose) box from glPrimitiveBoundingBox leaves the image bit-identical to the default box",
                     std::format("{} pixels differ", differing(base, loose)));
        check.expect(differing(base, tight) == 0,
                     "a tight per-patch box written to gl_BoundingBox by the TCS (state box absurdly small, overridden by gl_BoundingBox) leaves the image bit-identical",
                     std::format("{} pixels differ", differing(base, tight)));
        // Negative control: the comparison can fail. A camera one second further along the fly-over is a different picture.
        const FrameParams other = make_params(fp.width, fp.height, fly_pose(anim_time_ + 1.0), anim_time_ + 1.0);
        const int control = differing(base, image(other, 0));
        check.expect(control > 1000, "negative control: the image of a camera one second later differs from the base image in many pixels", std::format("{} pixels differ", control));
        // Not asserted: with a box that is too small the spec allows (but does not require) fragments outside it to be dropped.
        const int too_small = differing(base, image(fp, 3));
        std::cout << std::format("note: with a deliberately too small box and no gl_BoundingBox the image differs in {} pixels (section 13.2 leaves this open: \"may or may not generate fragments\")\n", too_small);
        draw_scene(fp, 0, 0);
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-tessellation-terrain",
        .title = "Tessellated terrain with dynamic LOD",
        .min_es = {3, 2},
        .summary = "A fly-over of a height-map terrain: the tessellation control shader picks every patch edge's level from the camera distance, "
                   "the evaluation shader displaces the vertices, glPrimitiveBoundingBox / gl_BoundingBox do not change the image.",
        .controls = "  w      wireframe: left half / whole window / off\n  l      tint by tessellation level\n"
                    "  f      freeze the LOD viewpoint while the camera keeps flying\n  b      primitive bounding box mode (default / loose / per-patch gl_BoundingBox): no visible change\n"
                    "  +/-    LOD distance scale\n  space  pause\n  r      reset",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<TessellationTerrain>(argc, argv, info);
}
