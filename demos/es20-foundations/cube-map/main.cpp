// es20-cube-map
//
// Cube map textures (OpenGL ES 3.0 section 3.8.9, Table 3.21): six 2x2 faces uploaded with
// glTexImage2D(TEXTURE_CUBE_MAP_POSITIVE_X ... NEGATIVE_Z), sampled through a samplerCube with
// NEAREST filtering. All 24 texels have different colours (each face has its own hue, the four
// texels of a face are four brightnesses of it), so any mix-up of faces, mirror image, rotation or
// transposition changes the colour of some probe.
//
// Image (256 x 256 default framebuffer):
//   * top 192 rows : a skybox - the camera sits at the centre of the cube and looks at the corner
//                    where +X, +Y and -Z meet;
//   * row of tiles : six 32 x 32 "face tiles" (+X -X +Y -Y +Z -Z). A tile samples directions that
//                    lie on the plane of one face (Table 3.21 solved for r), so it must show that
//                    face's 2x2 image upright and unmirrored;
//   * bottom tiles : four [incomplete | complete] pairs of cube maps: five faces only, a face of the
//                    wrong size, a face of the wrong format, and a complete cube map whose default
//                    MIN_FILTER needs mipmaps that were never created. Incomplete cube maps sample
//                    (0, 0, 0, 1).
//
// Validation compares every probe with a CPU implementation of Table 3.21 (major axis, sc, tc, ma,
// s = (sc/|ma| + 1)/2, t = (tc/|ma| + 1)/2, texel floor(2s), floor(2t)). Probes within 0.05 texel of a
// texel boundary or within 2 % of a face edge are skipped, because the spec leaves a tie on the
// major axis to the implementation and nearest-texel selection on a boundary is not robust.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kFaces = 6;
    constexpr int kSize = 2;  // every face is 2 x 2 texels
    constexpr double kTexelMargin = 0.05;  // texel units; probes closer to a texel boundary are skipped
    constexpr double kEdgeMargin = 0.98;   // probes with max(|sc|,|tc|) / |ma| above this are skipped (face edge / tie)

    constexpr std::array<mgl::TextureTarget, kFaces> kTargets = {
        mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
        mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
        mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
    constexpr std::array<const char*, kFaces> kFaceName = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

    constexpr std::array<Rgba8, kFaces> kBase = {{
        {220, 50, 50, 255},   // +X red
        {50, 200, 210, 255},  // -X cyan
        {60, 200, 70, 255},   // +Y green
        {200, 60, 200, 255},  // -Y magenta
        {60, 90, 230, 255},   // +Z blue
        {230, 210, 50, 255},  // -Z yellow
    }};
    constexpr std::array<double, 4> kShade = {1.0, 0.72, 0.5, 0.3};  // texel (i, j) = shade[2j + i]

    /// The colour of texel (i, j) of face `face`; j = 0 is the first row of the uploaded data (t = 0).
    Rgba8 texel_color(int face, int i, int j)
    {
        const Rgba8 b = kBase[static_cast<std::size_t>(face)];
        const double k = kShade[static_cast<std::size_t>(2 * j + i)];
        return {static_cast<std::uint8_t>(std::lround(b[0] * k)), static_cast<std::uint8_t>(std::lround(b[1] * k)),
                static_cast<std::uint8_t>(std::lround(b[2] * k)), 255};
    }

    constexpr Rgba8 kBlack = {0, 0, 0, 255};  // an incomplete texture samples (0, 0, 0, 1)

    // ---------------------------------------------------------------------------------------------
    // CPU model of ES 3.0 section 3.8.9 / Table 3.21
    // ---------------------------------------------------------------------------------------------
    struct Vec3
    {
        double x = 0, y = 0, z = 0;
    };
    Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    Vec3 operator*(Vec3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
    Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    Vec3 normalize(Vec3 a)
    {
        const double l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        return {a.x / l, a.y / l, a.z / l};
    }
    /// The value a GPU sees: components are 32-bit floats.
    Vec3 as_float(Vec3 a)
    {
        return {static_cast<double>(static_cast<float>(a.x)), static_cast<double>(static_cast<float>(a.y)),
                static_cast<double>(static_cast<float>(a.z))};
    }

    struct Lookup
    {
        int face = 0;
        double s = 0, t = 0;
        int i = 0, j = 0;
        bool ambiguous = false;
    };

    /// Table 3.21: the major axis selects the face; sc, tc, ma give (s, t); the texel is floor(s * 2), floor(t * 2).
    Lookup select_face(Vec3 r)
    {
        const double ax = std::abs(r.x), ay = std::abs(r.y), az = std::abs(r.z);
        double sc = 0, tc = 0, ma = 0;
        Lookup L;
        if (ax >= ay && ax >= az)
        {
            ma = ax;
            if (r.x >= 0) { L.face = 0; sc = -r.z; tc = -r.y; }  // +rx: sc = -rz, tc = -ry
            else          { L.face = 1; sc = r.z;  tc = -r.y; }  // -rx: sc = +rz, tc = -ry
        }
        else if (ay >= az)
        {
            ma = ay;
            if (r.y >= 0) { L.face = 2; sc = r.x; tc = r.z; }    // +ry: sc = +rx, tc = +rz
            else          { L.face = 3; sc = r.x; tc = -r.z; }   // -ry: sc = +rx, tc = -rz
        }
        else
        {
            ma = az;
            if (r.z >= 0) { L.face = 4; sc = r.x;  tc = -r.y; }  // +rz: sc = +rx, tc = -ry
            else          { L.face = 5; sc = -r.x; tc = -r.y; }  // -rz: sc = -rx, tc = -ry
        }
        L.s = 0.5 * (sc / ma + 1.0);
        L.t = 0.5 * (tc / ma + 1.0);
        L.i = std::clamp(static_cast<int>(std::floor(L.s * kSize)), 0, kSize - 1);
        L.j = std::clamp(static_cast<int>(std::floor(L.t * kSize)), 0, kSize - 1);
        const double edge = std::max(std::abs(sc), std::abs(tc)) / ma;
        L.ambiguous = edge > kEdgeMargin || std::abs(L.s * kSize - 1.0) < kTexelMargin || std::abs(L.t * kSize - 1.0) < kTexelMargin;
        return L;
    }

    /// Table 3.21 solved for r (|ma| = 1): the direction that lands on face `face` at face coordinates (s, t).
    /// Mirrors face_direction() in cube.frag.
    Vec3 face_direction(int face, double s, double t)
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

    // --- three deliberately wrong models, used as negative controls ----------------------------------
    Rgba8 wrong_flip_t(Vec3 r) { const Lookup L = select_face(r); return texel_color(L.face, L.i, kSize - 1 - L.j); }
    Rgba8 wrong_flip_s(Vec3 r) { const Lookup L = select_face(r); return texel_color(L.face, kSize - 1 - L.i, L.j); }
    Rgba8 wrong_transpose(Vec3 r) { const Lookup L = select_face(r); return texel_color(L.face, L.j, L.i); }

    // ---------------------------------------------------------------------------------------------
    // Layout
    // ---------------------------------------------------------------------------------------------
    struct Layout
    {
        int width = 0, height = 0, tile = 0, sky_h = 0;
    };

    /// Top-down view of an RGBA8 read-back (glReadPixels returns the bottom row first).
    struct Image
    {
        int w = 0, h = 0;
        std::vector<std::uint8_t> rgba;
        [[nodiscard]] Rgba8 at(int x, int y_top) const
        {
            const std::size_t i = (static_cast<std::size_t>(h - 1 - y_top) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
            return {rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]};
        }
    };

    // ---------------------------------------------------------------------------------------------
    // Completeness scenarios (bottom row)
    // ---------------------------------------------------------------------------------------------
    enum Scenario
    {
        kFiveFaces,      // -Z never specified; fix: specify it
        kWrongSize,      // -Y is 4x4 while the others are 2x2; fix: re-specify it as 2x2
        kWrongFormat,    // +Z is RGB8 while the others are RGBA8; fix: re-specify it as RGBA8
        kNeedsMipmaps,   // all six faces fine, but the default MIN_FILTER (NEAREST_MIPMAP_LINEAR) wants a mip chain; fix: glGenerateMipmap
        kScenarioCount
    };

    struct ScenarioInfo
    {
        const char* what;
        const char* fix;
    };
    constexpr std::array<ScenarioInfo, kScenarioCount> kScenarioInfo = {{
        {"-Z face never specified (five faces)", "glTexImage2D(NEGATIVE_Z, 2x2)"},
        {"-Y face is 4x4, the others 2x2 (not square-identical)", "glTexImage2D(NEGATIVE_Y, 2x2)"},
        {"+Z face is RGB8, the others RGBA8", "glTexImage2D(POSITIVE_Z, RGBA8)"},
        {"default MIN_FILTER NEAREST_MIPMAP_LINEAR, no mipmaps", "glGenerateMipmap(TEXTURE_CUBE_MAP)"},
    }};

    class CubeMap final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/cube.vert", "shaders/cube.frag");
            loc_mode_ = locate("u_mode");
            loc_face_ = locate("u_face");
            loc_dir_ = locate("u_dir");
            loc_forward_ = locate("u_forward");
            loc_right_ = locate("u_right");
            loc_up_ = locate("u_up");
            mgl::glUseProgram(program_);
            mgl::glUniform1i(locate("u_cube"), 0);  // samplerCube reads texture unit 0

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            // The cube map: one TEXTURE_CUBE_MAP object, six 2D images specified through the face targets.
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kMainCube]);
            binding_after_bind_ = demo::get_integer(mgl::GetParameter::TextureBindingCubeMap);
            for (int f = 0; f < kFaces; ++f) upload_face(f);
            set_filters(true);
            max_cube_size_ = demo::get_integer(mgl::GetParameter::MaxCubeMapTextureSize);

            for (int s = 0; s < kScenarioCount; ++s)
            {
                build_scenario(static_cast<Scenario>(s), broken(s), true);
                build_scenario(static_cast<Scenario>(s), twin(s), false);
            }
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});

            // Skybox camera: yaw 40 deg, pitch 30 deg, vertical field of view 70 deg. forward is close to the cube
            // corner (+1, +1, -1), so +X, +Y and -Z are all visible.
            constexpr double pi = 3.14159265358979323846;
            const double yaw = 40.0 * pi / 180.0, pitch = 30.0 * pi / 180.0, fov = 70.0 * pi / 180.0;
            forward_ = normalize({std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)});
            const Vec3 right = normalize(cross(forward_, {0, 1, 0}));
            const Vec3 up = cross(right, forward_);
            tan_half_fov_ = std::tan(fov / 2.0);
            right_ = right * tan_half_fov_;  // multiplied by the aspect ratio in render()
            up_ = up * tan_half_fov_;
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            const Layout lay = layout_of(env);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(program_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            draw_sky(lay, tex_[kMainCube]);
            for (int f = 0; f < kFaces; ++f) draw_face_tile(tex_[kMainCube], f, lay.tile * f, lay.tile, lay.tile);
            for (int s = 0; s < kScenarioCount; ++s)
            {
                draw_face_tile(broken(s), 0, lay.tile * 2 * s, 0, lay.tile);
                draw_face_tile(twin(s), 0, lay.tile * (2 * s + 1), 0, lay.tile);
            }
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            const Layout lay = layout_of(env);
            const Image fb = read_fb(lay);
            check_skybox(check, lay, fb);
            check_face_tiles(check, lay, fb);
            check_directions(check, env, lay);
            check_completeness(check, lay, fb);
            check_state_and_errors(check, lay);
            check_mutation(check, lay);

            render(env, 0);  // the experiments drew into the framebuffer: restore the picture
            check.expect_no_gl_error("no GL error after the experiments");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(program_);
        }

    private:
        enum : std::size_t { kMainCube, kScenarioBase, kTexCount = kScenarioBase + 2 * kScenarioCount };

        [[nodiscard]] mgl::TextureId broken(int s) const { return tex_[kScenarioBase + 2 * static_cast<std::size_t>(s)]; }
        [[nodiscard]] mgl::TextureId twin(int s) const { return tex_[kScenarioBase + 2 * static_cast<std::size_t>(s) + 1]; }

        mgl::UniformLocation locate(const char* name) const
        {
            const mgl::UniformLocation l = mgl::glGetUniformLocation(program_, name);
            if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
            return l;
        }

        static Layout layout_of(const demo::Env& env)
        {
            const int tile = env.width / 8;
            return {env.width, env.height, tile, env.height - 2 * tile};
        }

        // ------------------------------------------------------------------------------------------
        // Uploads
        // ------------------------------------------------------------------------------------------
        /// The four texels of a face in memory order: (0,0), (1,0), (0,1), (1,1) - row j = 0 first.
        static std::array<Rgba8, 4> face_texels(int face)
        {
            return {texel_color(face, 0, 0), texel_color(face, 1, 0), texel_color(face, 0, 1), texel_color(face, 1, 1)};
        }

        /// glTexImage2D of face `face` at its own face target. `size` != 2 makes a solid image of the face's base colour.
        static void upload_face(int face, int size = kSize)
        {
            if (size == kSize)
            {
                const auto texels = face_texels(face);
                mgl::glTexImage2D(kTargets[static_cast<std::size_t>(face)], 0, mgl::InternalFormat::Rgba8, kSize, kSize, 0,
                                  mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, texels.data());
            }
            else
            {
                const std::vector<Rgba8> solid(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), kBase[static_cast<std::size_t>(face)]);
                mgl::glTexImage2D(kTargets[static_cast<std::size_t>(face)], 0, mgl::InternalFormat::Rgba8, size, size, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, solid.data());
            }
        }

        /// A face with a different internal format: RGB8, rows of 2 texels are 6 bytes, so UNPACK_ALIGNMENT is 1 here.
        static void upload_face_rgb(int face)
        {
            std::vector<std::uint8_t> bytes;
            for (const Rgba8& t : face_texels(face))
                for (std::size_t c = 0; c < 3; ++c) bytes.push_back(t[c]);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 1);
            mgl::glTexImage2D(kTargets[static_cast<std::size_t>(face)], 0, mgl::InternalFormat::Rgb8, kSize, kSize, 0, mgl::PixelFormat::Rgb,
                              mgl::PixelType::UnsignedByte, bytes.data());
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
        }

        /// NEAREST magnification always; NEAREST minification when `nearest_min` (the default NEAREST_MIPMAP_LINEAR otherwise).
        static void set_filters(bool nearest_min)
        {
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Nearest);
            if (nearest_min) mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::Nearest);
        }

        void build_scenario(Scenario s, mgl::TextureId tex, bool is_broken)
        {
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex);
            set_filters(s != kNeedsMipmaps);
            for (int f = 0; f < kFaces; ++f)
            {
                if (s == kFiveFaces && is_broken && f == 5) continue;
                if (s == kWrongSize && is_broken && f == 3) upload_face(f, 4);
                else if (s == kWrongFormat && is_broken && f == 4) upload_face_rgb(f);
                else upload_face(f);
            }
            if (s == kNeedsMipmaps && !is_broken) mgl::glGenerateMipmap(mgl::TextureTarget::TextureCubeMap);
        }

        /// The single call that turns the broken cube map into its twin.
        void repair_scenario(Scenario s, mgl::TextureId tex)
        {
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex);
            switch (s)
            {
            case kFiveFaces: upload_face(5); break;
            case kWrongSize: upload_face(3); break;
            case kWrongFormat: upload_face(4); break;
            case kNeedsMipmaps: mgl::glGenerateMipmap(mgl::TextureTarget::TextureCubeMap); break;
            case kScenarioCount: break;
            }
        }

        // ------------------------------------------------------------------------------------------
        // Drawing
        // ------------------------------------------------------------------------------------------
        void draw_sky(const Layout& lay, mgl::TextureId cube) const
        {
            mgl::glViewport(0, lay.height - lay.sky_h, lay.width, lay.sky_h);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube);
            const double aspect = static_cast<double>(lay.width) / lay.sky_h;
            mgl::glUniform1i(loc_mode_, 1);
            mgl::glUniform3f(loc_forward_, static_cast<float>(forward_.x), static_cast<float>(forward_.y), static_cast<float>(forward_.z));
            mgl::glUniform3f(loc_right_, static_cast<float>(right_.x * aspect), static_cast<float>(right_.y * aspect),
                             static_cast<float>(right_.z * aspect));
            mgl::glUniform3f(loc_up_, static_cast<float>(up_.x), static_cast<float>(up_.y), static_cast<float>(up_.z));
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// A square tile at framebuffer position (x, y) (y from the bottom) that samples directions on the plane of `face`.
        void draw_face_tile(mgl::TextureId cube, int face, int x, int y, int size) const
        {
            mgl::glViewport(x, y, size, size);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube);
            mgl::glUniform1i(loc_mode_, 0);
            mgl::glUniform1i(loc_face_, face);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// Renders one pixel per direction (mode 2) into the top rows of the framebuffer and returns the colours.
        std::vector<Rgba8> sample_directions(const std::vector<Vec3>& dirs, const Layout& lay, mgl::TextureId cube) const
        {
            mgl::glUseProgram(program_);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, cube);
            mgl::glUniform1i(loc_mode_, 2);
            for (std::size_t n = 0; n < dirs.size(); ++n)
            {
                const int x = static_cast<int>(n % static_cast<std::size_t>(lay.width));
                const int y_top = static_cast<int>(n / static_cast<std::size_t>(lay.width));
                mgl::glViewport(x, lay.height - 1 - y_top, 1, 1);
                mgl::glUniform3f(loc_dir_, static_cast<float>(dirs[n].x), static_cast<float>(dirs[n].y), static_cast<float>(dirs[n].z));
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
            const Image img = read_fb(lay);
            std::vector<Rgba8> out;
            out.reserve(dirs.size());
            for (std::size_t n = 0; n < dirs.size(); ++n)
                out.push_back(img.at(static_cast<int>(n % static_cast<std::size_t>(lay.width)), static_cast<int>(n / static_cast<std::size_t>(lay.width))));
            return out;
        }

        static Image read_fb(const Layout& lay)
        {
            Image img;
            img.w = lay.width;
            img.h = lay.height;
            img.rgba = demo::read_rgba8(0, 0, lay.width, lay.height);
            return img;
        }

        // ------------------------------------------------------------------------------------------
        // Skybox
        // ------------------------------------------------------------------------------------------
        void check_skybox(demo::Check& check, const Layout& lay, const Image& fb)
        {
            const double aspect = static_cast<double>(lay.width) / lay.sky_h;
            std::array<long, kFaces> seen{};
            long probes = 0, skipped = 0, bad = 0;
            std::string first_bad;
            for (int y = 0; y < lay.sky_h; ++y)
                for (int x = 0; x < lay.width; ++x)
                {
                    // Window pixel centre -> NDC (the viewport is the sky rectangle) -> the same linear function the vertex shader evaluates.
                    const double ndc_x = 2.0 * (x + 0.5) / lay.width - 1.0;
                    const double ndc_y = 2.0 * ((lay.sky_h - 1 - y) + 0.5) / lay.sky_h - 1.0;
                    const Vec3 r = as_float(forward_ + right_ * (ndc_x * aspect) + up_ * ndc_y);
                    const Lookup L = select_face(r);
                    if (L.ambiguous)
                    {
                        ++skipped;
                        continue;
                    }
                    ++probes;
                    ++seen[static_cast<std::size_t>(L.face)];
                    const Rgba8 want = texel_color(L.face, L.i, L.j), got = fb.at(x, y);
                    if (got != want && bad++ == 0)
                        first_bad = std::format("pixel ({},{}): got {} expected {} (face {}, texel {},{})", x, y, demo::format_rgba(got),
                                                demo::format_rgba(want), kFaceName[static_cast<std::size_t>(L.face)], L.i, L.j);
                }
            int faces_seen = 0;
            for (const long n : seen)
                if (n > 100) ++faces_seen;
            check.expect(bad == 0 && probes > lay.width * lay.sky_h * 6 / 10,
                         "skybox: every pixel equals the CPU Table 3.21 lookup of its view ray",
                         bad == 0 ? std::format("{} px compared, {} skipped near edges; +X {} -X {} +Y {} -Y {} +Z {} -Z {}", probes, skipped, seen[0],
                                                seen[1], seen[2], seen[3], seen[4], seen[5])
                                  : first_bad);
            check.expect(faces_seen >= 3, "skybox: at least three different faces are visible (the test is not confined to one face)",
                         std::format("{} faces with more than 100 probes", faces_seen));
        }

        // ------------------------------------------------------------------------------------------
        // Face tiles: each shows its own face upright
        // ------------------------------------------------------------------------------------------
        void check_face_tiles(demo::Check& check, const Layout& lay, const Image& fb)
        {
            for (int f = 0; f < kFaces; ++f)
            {
                long probes = 0, bad = 0;
                std::string first_bad;
                for (int y = 0; y < lay.tile; ++y)
                    for (int x = 0; x < lay.tile; ++x)
                    {
                        const double s = (x + 0.5) / lay.tile, t = (y + 0.5) / lay.tile;  // t grows downwards, like the rows of the upload
                        if (std::abs(s * kSize - 1.0) < kTexelMargin || std::abs(t * kSize - 1.0) < kTexelMargin) continue;
                        ++probes;
                        const int i = static_cast<int>(std::floor(s * kSize)), j = static_cast<int>(std::floor(t * kSize));
                        const Rgba8 want = texel_color(f, i, j), got = fb.at(lay.tile * f + x, lay.sky_h + y);
                        if (got != want && bad++ == 0)
                            first_bad = std::format("tile pixel ({},{}): got {} expected texel ({},{}) = {}", x, y, demo::format_rgba(got), i, j,
                                                    demo::format_rgba(want));
                    }
                check.expect(bad == 0 && probes > 0,
                             std::format("face tile {}: directions on the plane of the {} face return that face's 2x2 image upright and unmirrored", f,
                                         kFaceName[static_cast<std::size_t>(f)]),
                             bad == 0 ? std::format("{} px compared", probes) : first_bad);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Single directions: hand-derived examples, the 24 texel centres, random directions
        // ------------------------------------------------------------------------------------------
        void check_directions(demo::Check& check, demo::Env& env, const Layout& lay)
        {
            // (a) Hand-derived examples of Table 3.21 (worked out on paper, independent of select_face()).
            struct Example
            {
                Vec3 r;
                int face, i, j;
                const char* why;
            };
            const Example examples[] = {
                {{1.0, 0.5, 0.5}, 0, 0, 0, "+X: sc = -rz = -0.5, tc = -ry = -0.5, ma = 1 -> s = t = 0.25"},
                {{-1.0, 0.5, 0.5}, 1, 1, 0, "-X: sc = +rz = 0.5, tc = -ry = -0.5 -> s = 0.75, t = 0.25"},
                {{0.5, 1.0, 0.5}, 2, 1, 1, "+Y: sc = rx = 0.5, tc = rz = 0.5 -> s = t = 0.75"},
                {{0.5, -1.0, 0.5}, 3, 1, 0, "-Y: sc = rx = 0.5, tc = -rz = -0.5 -> s = 0.75, t = 0.25"},
                {{0.5, 0.5, 1.0}, 4, 1, 0, "+Z: sc = rx = 0.5, tc = -ry = -0.5 -> s = 0.75, t = 0.25"},
                {{0.5, 0.5, -1.0}, 5, 0, 0, "-Z: sc = -rx = -0.5, tc = -ry = -0.5 -> s = t = 0.25"},
                {{-3.0, -6.0, 2.0}, 3, 0, 0, "-Y (|ry| = 6 is the major axis): sc = rx = -3, tc = -rz = -2, |ma| = 6 -> s = 0.25, t = 0.333"},
                {{2.0, -7.0, -9.0}, 5, 0, 1, "-Z (|rz| = 9): sc = -rx = -2, tc = -ry = 7, |ma| = 9 -> s = 0.389 (i = 0), t = 0.889 (j = 1)"},
            };
            std::vector<Vec3> dirs;
            for (const Example& e : examples) dirs.push_back(e.r);
            // (b) The 24 texel centres: face f, (s, t) = ((i + 0.5)/2, (j + 0.5)/2).
            const std::size_t centre_begin = dirs.size();
            for (int f = 0; f < kFaces; ++f)
                for (int j = 0; j < kSize; ++j)
                    for (int i = 0; i < kSize; ++i) dirs.push_back(face_direction(f, (i + 0.5) / kSize, (j + 0.5) / kSize));
            // (c) Scaled copies of one texel centre per face (a different texel on each face): the direction need not be normalised.
            const std::size_t scaled_begin = dirs.size();
            std::array<std::size_t, kFaces> scaled_source{};
            for (std::size_t f = 0; f < kFaces; ++f)
            {
                scaled_source[f] = centre_begin + 4 * f + f % 4;
                dirs.push_back(dirs[scaled_source[f]] * 1000.0);
                dirs.push_back(dirs[scaled_source[f]] * 0.001);
            }
            // (d) Random directions: components uniform in [-1, 1], a random length in [0.1, 1], seeded by --seed.
            const std::size_t random_begin = dirs.size();
            constexpr int kRandom = 1500;
            auto uniform = [&]() { return static_cast<double>(env.rng() >> 8) / 16777216.0; };
            for (int n = 0; n < kRandom; ++n)
            {
                const Vec3 v{2 * uniform() - 1, 2 * uniform() - 1, 2 * uniform() - 1};
                dirs.push_back(v * (0.1 + 0.9 * uniform()));
            }
            for (Vec3& d : dirs) d = as_float(d);  // the GPU gets 32-bit floats

            const std::vector<Rgba8> got = sample_directions(dirs, lay, tex_[kMainCube]);

            // (a)
            {
                int bad = 0;
                std::string first_bad;
                for (std::size_t n = 0; n < std::size(examples); ++n)
                {
                    const Example& e = examples[n];
                    const Rgba8 want = texel_color(e.face, e.i, e.j);
                    const Lookup L = select_face(dirs[n]);
                    const bool model_ok = L.face == e.face && L.i == e.i && L.j == e.j;
                    if ((!model_ok || got[n] != want) && bad++ == 0)
                        first_bad = std::format("r = ({}, {}, {}): got {}, hand-derived {} ({}); CPU model says face {} texel {},{}", e.r.x, e.r.y, e.r.z,
                                                demo::format_rgba(got[n]), demo::format_rgba(want), e.why, kFaceName[static_cast<std::size_t>(L.face)],
                                                L.i, L.j);
                }
                check.expect(bad == 0, "eight hand-derived directions (one per face + two oblique ones) give the texel computed on paper, in the GL and in the CPU model",
                             first_bad);
            }
            // (b)
            {
                int bad = 0;
                std::string first_bad;
                std::size_t n = centre_begin;
                for (int f = 0; f < kFaces; ++f)
                    for (int j = 0; j < kSize; ++j)
                        for (int i = 0; i < kSize; ++i, ++n)
                        {
                            const Rgba8 want = texel_color(f, i, j);
                            const Lookup L = select_face(dirs[n]);
                            const bool model_ok = L.face == f && L.i == i && L.j == j && !L.ambiguous;
                            if ((!model_ok || got[n] != want) && bad++ == 0)
                                first_bad = std::format("face {} texel ({},{}): got {} expected {}; CPU model says face {} texel {},{}",
                                                        kFaceName[static_cast<std::size_t>(f)], i, j, demo::format_rgba(got[n]),
                                                        demo::format_rgba(want), kFaceName[static_cast<std::size_t>(L.face)], L.i, L.j);
                        }
                check.expect(bad == 0, "all 24 texel centres (6 faces x 4 texels) are reached by the directions of Table 3.21 and return their own texel", first_bad);
            }
            // (c)
            {
                // The copies of centre scaled_source[f] are dirs[scaled_begin + 2 f] (x1000) and dirs[scaled_begin + 2 f + 1] (x0.001).
                int bad = 0;
                for (std::size_t f = 0; f < kFaces; ++f)
                {
                    const Rgba8 want = got[scaled_source[f]];
                    if (got[scaled_begin + 2 * f] != want || got[scaled_begin + 2 * f + 1] != want) ++bad;
                }
                check.expect(bad == 0, "scaling a direction by 1000 or 0.001 does not change the texel (the lookup uses the ratios sc/|ma| and tc/|ma|)",
                             std::format("{} of {} faces differ", bad, kFaces));
            }
            // (d)
            std::array<long, kFaces * 4> bucket{};
            long probes = 0, skipped = 0, bad = 0;
            std::string first_bad;
            std::array<long, 3> wrong_matches{};  // how many probes each wrong model would have "passed"
            for (std::size_t n = random_begin; n < dirs.size(); ++n)
            {
                const Lookup L = select_face(dirs[n]);
                if (L.ambiguous)
                {
                    ++skipped;
                    continue;
                }
                ++probes;
                ++bucket[static_cast<std::size_t>(L.face * 4 + L.j * 2 + L.i)];
                const Rgba8 want = texel_color(L.face, L.i, L.j);
                if (got[n] != want && bad++ == 0)
                    first_bad = std::format("r = ({:.4f}, {:.4f}, {:.4f}): got {}, Table 3.21 gives face {} texel ({},{}) = {}", dirs[n].x, dirs[n].y,
                                            dirs[n].z, demo::format_rgba(got[n]), kFaceName[static_cast<std::size_t>(L.face)], L.i, L.j,
                                            demo::format_rgba(want));
                if (got[n] == wrong_flip_t(dirs[n])) ++wrong_matches[0];
                if (got[n] == wrong_flip_s(dirs[n])) ++wrong_matches[1];
                if (got[n] == wrong_transpose(dirs[n])) ++wrong_matches[2];
            }
            const long min_bucket = *std::min_element(bucket.begin(), bucket.end());
            check.expect(bad == 0 && probes > kRandom / 2,
                         "random directions (any length): the colour equals the CPU Table 3.21 texel at every probe",
                         bad == 0 ? std::format("{} probes, {} skipped near edges/boundaries", probes, skipped) : first_bad);
            check.expect(min_bucket >= 10, "the random probes reach every one of the 24 (face, quadrant) texels at least 10 times",
                         std::format("fewest probes on a texel: {}", min_bucket));

            // Negative controls: a model with t flipped, s flipped or s/t exchanged must disagree with the GL at many probes.
            // Flips change the texel at every probe; the transposition only on the off-diagonal texels (half of them).
            check.expect(wrong_matches[0] * 20 < probes && wrong_matches[1] * 20 < probes && wrong_matches[2] * 10 < probes * 6,
                         "negative control: models with t flipped, s flipped or s/t transposed do NOT reproduce the GL colours",
                         std::format("of {} probes they match {}, {}, {} (flip t, flip s, transpose)", probes, wrong_matches[0], wrong_matches[1],
                                     wrong_matches[2]));
        }

        // ------------------------------------------------------------------------------------------
        // Completeness (bottom tiles)
        // ------------------------------------------------------------------------------------------
        /// True when every pixel of the tile at x (framebuffer, bottom tile row when `bottom`) equals `want`.
        long mismatches(const Image& img, int x0, int y_top0, int tile, int face, bool expect_black) const
        {
            long bad = 0;
            for (int y = 0; y < tile; ++y)
                for (int x = 0; x < tile; ++x)
                {
                    const double s = (x + 0.5) / tile, t = (y + 0.5) / tile;
                    if (std::abs(s * kSize - 1.0) < kTexelMargin || std::abs(t * kSize - 1.0) < kTexelMargin) continue;
                    const Rgba8 want = expect_black ? kBlack : texel_color(face, static_cast<int>(std::floor(s * kSize)), static_cast<int>(std::floor(t * kSize)));
                    if (img.at(x0 + x, y_top0 + y) != want) ++bad;
                }
            return bad;
        }

        void check_completeness(demo::Check& check, const Layout& lay, const Image& fb)
        {
            for (int s = 0; s < kScenarioCount; ++s)
            {
                const ScenarioInfo& info = kScenarioInfo[static_cast<std::size_t>(s)];
                const int y_top = lay.sky_h + lay.tile;
                const long bad_broken = mismatches(fb, lay.tile * 2 * s, y_top, lay.tile, 0, true);
                const long bad_twin = mismatches(fb, lay.tile * (2 * s + 1), y_top, lay.tile, 0, false);
                check.expect(bad_broken == 0 && bad_twin == 0,
                             std::format("completeness {}: '{}' samples (0,0,0,1); the complete twin shows the +X face", s, info.what),
                             std::format("{} wrong pixels in the broken tile, {} in the twin", bad_broken, bad_twin));

                if (s != kNeedsMipmaps)
                {
                    // glGenerateMipmap needs a cube-complete texture (ES 3.0 section 3.8.10.5).
                    mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, broken(s));
                    mgl::glGenerateMipmap(mgl::TextureTarget::TextureCubeMap);
                    check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                          std::format("completeness {}: glGenerateMipmap(TEXTURE_CUBE_MAP) on a cube map that is not cube complete -> INVALID_OPERATION", s));
                }

                repair_scenario(static_cast<Scenario>(s), broken(s));
                mgl::glUseProgram(program_);
                draw_face_tile(broken(s), 0, 0, lay.height - lay.tile, lay.tile);
                const Image after = read_fb(lay);
                const long bad_after = mismatches(after, 0, 0, lay.tile, 0, false);
                check.expect(bad_after == 0, std::format("completeness {}: {} makes the same cube map complete: the +X face appears", s, info.fix),
                             std::format("{} wrong pixels", bad_after));
            }

            // Re-create the broken cube maps so that the final picture shows them as designed.
            std::array<mgl::TextureId, kScenarioCount> old{};
            for (int s = 0; s < kScenarioCount; ++s) old[static_cast<std::size_t>(s)] = broken(s);
            mgl::glDeleteTextures(kScenarioCount, old.data());
            std::array<mgl::TextureId, kScenarioCount> fresh{};
            mgl::glGenTextures(kScenarioCount, fresh.data());
            for (int s = 0; s < kScenarioCount; ++s)
            {
                tex_[kScenarioBase + 2 * static_cast<std::size_t>(s)] = fresh[static_cast<std::size_t>(s)];
                build_scenario(static_cast<Scenario>(s), fresh[static_cast<std::size_t>(s)], true);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Queries and defined errors
        // ------------------------------------------------------------------------------------------
        void check_state_and_errors(demo::Check& check, const Layout& /*lay*/)
        {
            check.expect_eq(binding_after_bind_, static_cast<int>(tex_[kMainCube].value), "glGetIntegerv(TEXTURE_BINDING_CUBE_MAP) is the bound cube map");
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
            check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBindingCubeMap), 0, "TEXTURE_BINDING_CUBE_MAP is 0 after unbinding");
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kMainCube]);
            check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBinding2D), 0, "the cube map binding is independent of TEXTURE_BINDING_2D");
            check.expect(max_cube_size_ >= 2048, "MAX_CUBE_MAP_TEXTURE_SIZE >= 2048 (the ES 3.0 minimum, Table 6.28)", std::format("{}", max_cube_size_));

            // Defined errors (ES 3.0 section 3.8.3 and 3.8.1).
            const std::array<Rgba8, 8> data{};
            mgl::glTexImage2D(mgl::TextureTarget::TextureCubeMapPositiveX, 0, mgl::InternalFormat::Rgba8, 2, 4, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, data.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "a non-square cube map face (2x4) -> INVALID_VALUE (width and height must be equal)");
            mgl::glTexImage2D(mgl::TextureTarget::TextureCubeMapPositiveX, 0, mgl::InternalFormat::Rgba8, max_cube_size_ + 1, max_cube_size_ + 1, 0,
                              mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "a face larger than MAX_CUBE_MAP_TEXTURE_SIZE -> INVALID_VALUE");
            mgl::glTexImage2D(mgl::TextureTarget::TextureCubeMap, 0, mgl::InternalFormat::Rgba8, 2, 2, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, data.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glTexImage2D(TEXTURE_CUBE_MAP, ...): the target must be a face (TEXTURE_CUBE_MAP_POSITIVE_X ...) -> INVALID_ENUM");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kMainCube]);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "binding a cube map texture to TEXTURE_2D -> INVALID_OPERATION");
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
        }

        // ------------------------------------------------------------------------------------------
        // Mutation: one texel changed through glTexSubImage2D must be seen by the face-tile comparison
        // ------------------------------------------------------------------------------------------
        void check_mutation(demo::Check& check, const Layout& lay)
        {
            mgl::glUseProgram(program_);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kMainCube]);
            const Rgba8 wrong = {1, 2, 3, 255};
            mgl::glTexSubImage2D(mgl::TextureTarget::TextureCubeMapNegativeY, 0, 1, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                 wrong.data());  // texel (1, 0) of the -Y face
            draw_face_tile(tex_[kMainCube], 3, 0, lay.height - lay.tile, lay.tile);
            const Image img = read_fb(lay);
            long changed = 0, unchanged_ok = 0, unchanged = 0;
            for (int y = 0; y < lay.tile; ++y)
                for (int x = 0; x < lay.tile; ++x)
                {
                    const double s = (x + 0.5) / lay.tile, t = (y + 0.5) / lay.tile;
                    if (std::abs(s * kSize - 1.0) < kTexelMargin || std::abs(t * kSize - 1.0) < kTexelMargin) continue;
                    const int i = static_cast<int>(std::floor(s * kSize)), j = static_cast<int>(std::floor(t * kSize));
                    const Rgba8 got = img.at(x, y);
                    if (i == 1 && j == 0)
                    {
                        if (got == wrong && got != texel_color(3, 1, 0)) ++changed;
                    }
                    else
                    {
                        ++unchanged;
                        if (got == texel_color(3, i, j)) ++unchanged_ok;
                    }
                }
            check.expect(changed > 0 && unchanged_ok == unchanged,
                         "negative control: after glTexSubImage2D changes texel (1,0) of the -Y face, exactly that quadrant of the -Y tile changes colour and the comparison would fail",
                         std::format("{} pixels changed, {} of {} other pixels unchanged", changed, unchanged_ok, unchanged));
            const Rgba8 original = texel_color(3, 1, 0);
            mgl::glTexSubImage2D(mgl::TextureTarget::TextureCubeMapNegativeY, 0, 1, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                 original.data());
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
        }

        // ------------------------------------------------------------------------------------------
        mgl::ProgramId program_{};
        mgl::UniformLocation loc_mode_{}, loc_face_{}, loc_dir_{}, loc_forward_{}, loc_right_{}, loc_up_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        Vec3 forward_, right_, up_;
        double tan_half_fov_ = 0;
        int binding_after_bind_ = -1;
        int max_cube_size_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-cube-map",
        .title = "Cube map textures: face selection (Table 3.21) and completeness",
        .min_es = {3, 0},
        .summary = "Six 2x2 cube faces sampled with a samplerCube: a skybox, per-face tiles, hundreds of probe directions checked "
                   "against a CPU model of Table 3.21, and incomplete cube maps.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<CubeMap>(argc, argv, info);
}
