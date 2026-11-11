// es30-skybox-and-fog
//
// An outdoor scene built around one cube map. The cube map is painted on the CPU (sky gradient, sun
// disc with a glow, two layers of distant mountains, a hazy ground below the horizon) and is used
// three ways:
//
//   * as a SKYBOX, drawn last with the depth trick (gl_Position = clip.xyww, glDepthFunc(GL_LEQUAL)),
//   * as a MIRROR environment: a chrome sphere and a gold box show texture(sky, reflect(I, N)),
//   * as a REFRACTED environment: a glass sphere mixes texture(sky, refract(I, N, 1/1.5)) with the
//     reflection by the Fresnel-Schlick factor.
//
// A checkered ground plane recedes to the horizon and dissolves in exponential-squared fog whose colour
// is the sky colour at the horizon. ES has no fixed-function fog: the fragment shader computes it.
//
// Validation (--validate) renders a validation configuration with NEAREST filtering and compares it with
// a CPU ray caster that implements the same cube-map face selection (ES 3.0 section 3.8.9, Table 3.21),
// the GLSL reflect/refract formulas (GLSL ES 3.00 section 8.5), the Fresnel-Schlick blend and the fog.
// It also probes the six face directions with a face-coded cube map, tests seamless filtering exactly on
// cube edges (ES 3.0 section 3.8.9.1), checks the depth trick and the fog blend at a known distance, and
// runs negative controls (swapped cube faces, inverted eta, depth trick off, fog off). See README.md.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <initializer_list>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mgl = metagl;
namespace m = demo::math;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Scene constants, shared by the renderer and the CPU reference.
    // ---------------------------------------------------------------------------------------------
    constexpr int kSkySize = 512;    // texels per cube face edge of the painted sky
    constexpr int kSkyLevels = 10;   // 512 .. 1: full mipmap chain for the windowed (LINEAR) configuration
    constexpr int kFaceGrid = 5;     // face-coded validation cube map: 5 x 5 texels per face
    constexpr int kSeamSize = 2;     // seam test cube map: 2 x 2 texels per face, one colour per face

    constexpr m::Vec3 kFogColor{0.74f, 0.80f, 0.87f};  // = the sky colour at the horizon
    constexpr m::Vec3 kZenithColor{0.17f, 0.36f, 0.72f};
    constexpr float kSunAzimuthDeg = -17.0f;   // azimuth measured from -z towards +x
    constexpr float kSunElevationDeg = 12.0f;
    constexpr double kSunDiscDeg = 1.8;

    constexpr float kGroundHalf = 400.0f;
    constexpr int kGroundCells = 40;
    constexpr float kTile = 2.0f;
    constexpr m::Vec3 kAlbedoA{0.52f, 0.55f, 0.43f};
    constexpr m::Vec3 kAlbedoB{0.36f, 0.41f, 0.30f};
    constexpr m::Vec3 kAmbient{0.62f, 0.66f, 0.74f};
    constexpr m::Vec3 kSunColor{1.0f, 0.92f, 0.78f};

    constexpr float kSphereRadius = 1.0f;
    constexpr int kSlices = 128, kStacks = 64;
    constexpr m::Vec3 kChromePos{-2.4f, 1.0f, 0.0f};
    constexpr m::Vec3 kGlassPos{2.4f, 1.0f, 0.0f};
    constexpr m::Vec3 kBoxPos{0.0f, 0.8f, -3.2f};
    constexpr float kBoxSize = 1.6f;
    constexpr float kBoxAngleDeg = 40.0f;
    constexpr m::Vec3 kChromeTint{0.90f, 0.91f, 0.93f};
    constexpr m::Vec3 kGlassTint{0.90f, 0.96f, 1.0f};
    constexpr m::Vec3 kGoldTint{1.0f, 0.80f, 0.42f};
    constexpr float kEta = 1.0f / 1.5f;  // air (n = 1) -> glass (n = 1.5)
    constexpr float kF0 = 0.04f;         // ((1 - 1.5) / (1 + 1.5))^2

    constexpr float kFovyDeg = 50.0f;  // vertical field of view for aspect >= 1 (horizontal for aspect < 1)
    constexpr float kNear = 0.25f, kFar = 2000.0f;  // kNear is reduced for extreme aspect ratios (finish_camera)
    constexpr m::Vec3 kTarget{0.0f, 1.0f, -1.0f};
    constexpr float kDefaultElevation = 10.0f;
    constexpr float kDefaultDistance = 10.0f;
    constexpr float kDefaultFog = 0.035f;
    constexpr float kOrbitRate = 0.08f;  // radians per second of automatic orbit in windowed mode

    [[nodiscard]] std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    // ---------------------------------------------------------------------------------------------
    // Double-precision vector type of the CPU side (cube painting and the reference).
    // ---------------------------------------------------------------------------------------------
    struct D3
    {
        double x = 0.0, y = 0.0, z = 0.0;
    };
    D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    D3 operator*(D3 a, D3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
    double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    double length(D3 a) { return std::sqrt(dot(a, a)); }
    D3 normalize(D3 a) { return a * (1.0 / length(a)); }
    D3 mix(D3 a, D3 b, double t) { return a * (1.0 - t) + b * t; }
    double dbl(float v) { return static_cast<double>(v); }
    D3 d3(const m::Vec3& v) { return {dbl(v.x), dbl(v.y), dbl(v.z)}; }
    double rad(double degrees) { return degrees * std::numbers::pi / 180.0; }

    D3 sun_direction()
    {
        const double az = rad(kSunAzimuthDeg), el = rad(kSunElevationDeg);
        return {std::sin(az) * std::cos(el), std::sin(el), -std::cos(az) * std::cos(el)};
    }

    // ---------------------------------------------------------------------------------------------
    // Cube map face selection, ES 3.0 section 3.8.9 and Table 3.21:
    //
    //   major axis   target              sc    tc    ma
    //   +rx          POSITIVE_X          -rz   -ry   rx
    //   -rx          NEGATIVE_X           rz   -ry   rx
    //   +ry          POSITIVE_Y           rx    rz   ry
    //   -ry          NEGATIVE_Y           rx   -rz   ry
    //   +rz          POSITIVE_Z           rx   -ry   rz
    //   -rz          NEGATIVE_Z          -rx   -ry   rz
    //
    //   s = (sc / |ma| + 1) / 2,  t = (tc / |ma| + 1) / 2
    //
    // Faces are indexed in target order (0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z). Ties between
    // equal magnitudes are implementation-defined (ES 3.0 section 3.8.9); the reference handles them by
    // sampling a neighbourhood of directions (see Envelope below).
    // ---------------------------------------------------------------------------------------------
    struct FaceSt
    {
        int face = 0;
        double s = 0.0, t = 0.0;
    };

    FaceSt select_face(D3 r)
    {
        const double ax = std::abs(r.x), ay = std::abs(r.y), az = std::abs(r.z);
        double sc = 0.0, tc = 0.0, ma = 0.0;
        int face = 0;
        if (ax >= ay && ax >= az)
        {
            ma = ax;
            if (r.x >= 0.0) { face = 0; sc = -r.z; tc = -r.y; }
            else { face = 1; sc = r.z; tc = -r.y; }
        }
        else if (ay >= az)
        {
            ma = ay;
            if (r.y >= 0.0) { face = 2; sc = r.x; tc = r.z; }
            else { face = 3; sc = r.x; tc = -r.z; }
        }
        else
        {
            ma = az;
            if (r.z >= 0.0) { face = 4; sc = r.x; tc = -r.y; }
            else { face = 5; sc = -r.x; tc = -r.y; }
        }
        return {face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0)};
    }

    // The inverse of Table 3.21 for |ma| = 1: the direction that hits face `face` at (sc, tc) in [-1, 1].
    D3 face_direction(int face, double sc, double tc)
    {
        switch (face)
        {
        case 0: return {1.0, -tc, -sc};
        case 1: return {-1.0, -tc, sc};
        case 2: return {sc, 1.0, tc};
        case 3: return {sc, -1.0, -tc};
        case 4: return {sc, -tc, 1.0};
        default: return {-sc, -tc, -1.0};
        }
    }

    constexpr std::array<mgl::TextureTarget, 6> kFaceTargets = {
        mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX,
        mgl::TextureTarget::TextureCubeMapPositiveY, mgl::TextureTarget::TextureCubeMapNegativeY,
        mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};
    constexpr std::array<std::string_view, 6> kFaceNames = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

    // The six images of a cube map as uploaded: RGBA8, row j holds t = (j + 0.5) / size (the first row of
    // a glTexSubImage2D upload is t = 0).
    struct CubeImage
    {
        int size = 0;
        std::array<std::vector<std::uint8_t>, 6> texels;

        [[nodiscard]] D3 texel(int face, int i, int j) const
        {
            const std::size_t k = (static_cast<std::size_t>(j) * static_cast<std::size_t>(size) + static_cast<std::size_t>(i)) * 4;
            const auto& f = texels[static_cast<std::size_t>(face)];
            return {f[k] / 255.0, f[k + 1] / 255.0, f[k + 2] / 255.0};
        }

        // NEAREST lookup in the base level. For cube maps the wrap mode is ignored and NEAREST always
        // clamps to the edge (ES 3.0 section 3.8.9.1). flip_s / flip_t mirror the face images: wrong
        // orientation models used only to show that the probes reject them.
        [[nodiscard]] D3 nearest(D3 r, bool flip_s = false, bool flip_t = false) const
        {
            const FaceSt f = select_face(r);
            int i = std::clamp(static_cast<int>(std::floor(f.s * size)), 0, size - 1);
            int j = std::clamp(static_cast<int>(std::floor(f.t * size)), 0, size - 1);
            if (flip_s) i = size - 1 - i;
            if (flip_t) j = size - 1 - j;
            return texel(f.face, i, j);
        }
    };

    // ---------------------------------------------------------------------------------------------
    // The painted sky (linear colour 0..1 for a unit direction d).
    // ---------------------------------------------------------------------------------------------
    double mountain_far(double az) { return rad(4.2 + 1.8 * std::sin(2.0 * az + 0.7) + 1.2 * std::sin(5.0 * az + 2.1) + 0.6 * std::sin(11.0 * az + 0.3)); }
    double mountain_near(double az)
    {
        return rad(std::max(0.3, 2.0 + 1.5 * std::sin(3.0 * az + 1.9) + 0.7 * std::sin(7.0 * az + 0.4) + 0.35 * std::sin(17.0 * az + 1.3)));
    }

    // `aa` is the angular size of a texel: every hard edge (horizon, mountain ridges, sun disc) is painted
    // with its fractional coverage over one texel, so bilinear magnification shows a smooth line instead of
    // texel stair-steps.
    D3 sky_radiance(D3 d, double aa)
    {
        auto cover = [aa](double signed_distance) { return std::clamp(signed_distance / aa + 0.5, 0.0, 1.0); };
        const D3 horizon = d3(kFogColor);
        const double el = std::asin(std::clamp(d.y, -1.0, 1.0));
        // Hazy ground below the horizon (only seen in reflections and refractions).
        const D3 below = mix(horizon, D3{0.33, 0.37, 0.29}, 1.0 - std::exp(std::min(el, 0.0) / 0.12));
        const double up = std::max(el, 0.0);
        const double az = std::atan2(d.x, -d.z);
        D3 c = mix(horizon, d3(kZenithColor), std::pow(up / (0.5 * std::numbers::pi), 0.5));
        const double sun_angle = std::acos(std::clamp(dot(d, sun_direction()), -1.0, 1.0));
        c = c + D3{1.0, 0.78, 0.50} * (0.40 * std::exp(-sun_angle / 0.11));  // glow around the sun
        const double hn = mountain_near(az), hf = mountain_far(az);
        c = mix(c, mix(horizon, D3{0.42, 0.50, 0.64}, 0.62), cover(hf - up));                 // farther range: pale blue
        c = mix(c, mix(horizon, D3{0.22, 0.30, 0.34}, 0.50 + 0.40 * up / hn), cover(hn - up));  // nearer range: darker, hazier at its foot
        c = mix(c, D3{1.0, 0.97, 0.88}, cover(rad(kSunDiscDeg) - sun_angle));                   // sun disc
        c = mix(below, c, cover(el));
        return {std::clamp(c.x, 0.0, 1.0), std::clamp(c.y, 0.0, 1.0), std::clamp(c.z, 0.0, 1.0)};
    }

    template <typename Fn>
    CubeImage paint_cube(int size, Fn&& texel_color)
    {
        CubeImage img;
        img.size = size;
        for (int f = 0; f < 6; ++f)
        {
            auto& v = img.texels[static_cast<std::size_t>(f)];
            v.reserve(static_cast<std::size_t>(size * size * 4));
            for (int j = 0; j < size; ++j)
                for (int i = 0; i < size; ++i)
                {
                    const std::array<std::uint8_t, 3> c = texel_color(f, i, j);
                    v.insert(v.end(), {c[0], c[1], c[2], 255});
                }
        }
        return img;
    }

    CubeImage make_sky_image()
    {
        return paint_cube(kSkySize, [](int f, int i, int j) {
            const double sc = 2.0 * (i + 0.5) / kSkySize - 1.0, tc = 2.0 * (j + 0.5) / kSkySize - 1.0;
            const D3 c = sky_radiance(normalize(face_direction(f, sc, tc)), 1.5 / kSkySize);
            return std::array<std::uint8_t, 3>{to8(c.x), to8(c.y), to8(c.z)};
        });
    }

    // Distinct face colours: +X red, -X cyan, +Y green, -Y magenta, +Z blue, -Z yellow.
    constexpr std::array<std::array<int, 3>, 6> kFaceBase = {{{220, 60, 60}, {60, 200, 200}, {70, 200, 70}, {200, 70, 200}, {70, 90, 220}, {220, 200, 60}}};

    // Face-coded texel: the face hue, scaled by the column i and lifted by the row j, so every texel of a
    // face differs from every other one (a mirrored face is detected, not only a swapped one).
    std::array<std::uint8_t, 3> face_coded_texel(int f, int i, int j)
    {
        std::array<std::uint8_t, 3> c{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double v = kFaceBase[static_cast<std::size_t>(f)][k] * (0.55 + 0.1 * i) + 12.0 * j;
            c[k] = static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
        }
        return c;
    }

    CubeImage make_face_image() { return paint_cube(kFaceGrid, face_coded_texel); }

    CubeImage make_seam_image()
    {
        return paint_cube(kSeamSize, [](int f, int, int) {
            const auto& b = kFaceBase[static_cast<std::size_t>(f)];
            return std::array<std::uint8_t, 3>{static_cast<std::uint8_t>(b[0]), static_cast<std::uint8_t>(b[1]), static_cast<std::uint8_t>(b[2])};
        });
    }

    // ---------------------------------------------------------------------------------------------
    // Camera and per-frame parameters
    // ---------------------------------------------------------------------------------------------
    struct Params
    {
        int ticks = 0;  // animation time in 1/60 s steps (deterministic: frame / 60)
        float yaw = 0.0f;
        float elevation = kDefaultElevation;
        float distance = kDefaultDistance;
        float fog_density = kDefaultFog;
        bool fog_on = true;
        bool face_coded = false;   // key c: show the face-coded cube map instead of the sky
        bool nearest = false;      // key n: NEAREST instead of trilinear sky filtering (the validation configuration)
        bool depth_trick = true;   // key k: false draws the sky cube at its true depth (it then hides everything)
        bool inverse_eta = false;  // key i: glass with eta = 1.5 instead of 1 / 1.5
        bool paused = false;
    };

    struct Frame
    {
        float aspect = 1.0f, fovy = 0.0f;
        m::Vec3 eye{}, center{}, up_hint{0.0f, 1.0f, 0.0f};
        m::Mat4 view{}, proj{};
        float fog_density = 0.0f;
        float eta = kEta;
        bool face_coded = false, nearest = false, depth_trick = true;
    };

    // The picture must fill any surface: for portrait surfaces the HORIZONTAL field of view is kept at
    // kFovyDeg (the vertical one grows), so the objects never leave the sides.
    float fovy_for(float aspect)
    {
        const float base = m::radians(kFovyDeg);
        return aspect >= 1.0f ? base : 2.0f * std::atan(std::tan(0.5f * base) / aspect);
    }

    void finish_camera(Frame& f)
    {
        f.fovy = fovy_for(f.aspect);
        f.view = m::look_at(f.eye, f.center, f.up_hint);
        // The sky cube (half-size 0.5 around the eye) must lie entirely beyond the near plane, otherwise the
        // depth-trick-off negative control would clip it near the corners of a wide field of view and show the
        // scene there. The cube is at least 0.5 / |d| away along the view direction d = (tan_x, tan_y, 1) of a
        // frustum corner, so the near plane is pulled in (only) for extreme aspect ratios.
        const float ty = std::tan(0.5f * f.fovy), tx = ty * f.aspect;
        const float near_plane = std::min(kNear, 0.9f * 0.5f / std::sqrt(1.0f + tx * tx + ty * ty));
        f.proj = m::perspective(f.fovy, f.aspect, near_plane, kFar);
    }

    Frame make_frame(const Params& p, float aspect)
    {
        Frame f;
        f.aspect = aspect;
        const float t = static_cast<float>(p.ticks) / 60.0f;
        const float yaw = p.yaw + kOrbitRate * t;
        const float el = m::radians(p.elevation);
        f.center = kTarget;
        f.eye = f.center + m::Vec3{std::sin(yaw) * std::cos(el), std::sin(el), std::cos(yaw) * std::cos(el)} * p.distance;
        finish_camera(f);
        f.fog_density = p.fog_on ? p.fog_density : 0.0f;
        f.eta = p.inverse_eta ? 1.0f / kEta : kEta;
        f.face_coded = p.face_coded;
        f.nearest = p.nearest;
        f.depth_trick = p.depth_trick;
        return f;
    }

    // The validation close-up of the glass sphere: the default camera direction, but aimed at the glass sphere from
    // 3.7 units, so that the sphere's radius is about 60 % of half the shorter side of the surface at ANY aspect ratio
    // (in the full scene it is only ~11 % of the shorter side; on a 200-pixel side the thin ring where the Fresnel
    // term exceeds 0.1 is then about one pixel wide and holds just a couple of stable pixels).
    Frame glass_closeup_frame(const Params& p, float aspect)
    {
        Frame f = make_frame(p, aspect);
        const m::Vec3 to_eye = m::normalize(f.eye - kGlassPos);
        f.center = kGlassPos;
        f.eye = kGlassPos + to_eye * 3.7f;
        finish_camera(f);
        return f;
    }

    // A camera at the origin looking straight along one cube axis (sky only, face-coded cube map).
    Frame face_frame(int face, float aspect)
    {
        static constexpr std::array<m::Vec3, 6> kAxis = {{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
        Frame f;
        f.aspect = aspect;
        f.eye = {0.0f, 0.0f, 0.0f};
        f.center = kAxis[static_cast<std::size_t>(face)];
        f.up_hint = face == 2 ? m::Vec3{0.0f, 0.0f, -1.0f} : face == 3 ? m::Vec3{0.0f, 0.0f, 1.0f} : m::Vec3{0.0f, 1.0f, 0.0f};
        finish_camera(f);
        f.face_coded = true;
        f.nearest = true;
        return f;
    }

    // ---------------------------------------------------------------------------------------------
    // Objects and meshes
    // ---------------------------------------------------------------------------------------------
    enum class Obj
    {
        Sky,
        Ground,
        Chrome,
        Glass,
        Box,
        Count
    };

    const char* obj_name(Obj o)
    {
        switch (o)
        {
        case Obj::Sky: return "sky";
        case Obj::Ground: return "ground";
        case Obj::Chrome: return "chrome sphere (reflect)";
        case Obj::Glass: return "glass sphere (refract + Fresnel)";
        case Obj::Box: return "gold box (reflect)";
        case Obj::Count: break;
        }
        return "?";
    }

    enum class MeshId
    {
        Ground,
        Sphere,
        Cube,
        Count
    };

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    struct MeshData
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;
    };

    [[nodiscard]] float fl(int v) { return static_cast<float>(v); }
    [[nodiscard]] std::uint16_t u16(int v) { return static_cast<std::uint16_t>(v); }

    // UV sphere of radius 1: the normal of a vertex equals its position.
    MeshData make_sphere(int slices, int stacks)
    {
        const float pi = std::numbers::pi_v<float>;
        MeshData md;
        for (int i = 0; i <= stacks; ++i)
        {
            const float phi = pi * fl(i) / fl(stacks);
            for (int j = 0; j <= slices; ++j)
            {
                const float theta = 2.0f * pi * fl(j) / fl(slices);
                const float x = std::sin(phi) * std::cos(theta), y = std::cos(phi), z = std::sin(phi) * std::sin(theta);
                md.vertices.push_back({x, y, z, x, y, z});
            }
        }
        for (int i = 0; i < stacks; ++i)
            for (int j = 0; j < slices; ++j)
            {
                const int a = i * (slices + 1) + j, b = a + slices + 1;
                for (const int idx : {a, a + 1, b, a + 1, b + 1, b}) md.indices.push_back(u16(idx));
            }
        return md;
    }

    // Unit cube [-0.5, 0.5]^3 with flat normals (4 vertices per face). Also the skybox geometry: the sky
    // shader uses only the positions, as directions.
    MeshData make_cube()
    {
        MeshData md;
        struct Face
        {
            m::Vec3 n, u, v;
        };
        const Face faces[6] = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},  {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}}, {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
                               {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}, {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},  {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}};
        for (int f = 0; f < 6; ++f)
        {
            const Face& fc = faces[f];
            const float su[4] = {-1, 1, 1, -1}, sv[4] = {-1, -1, 1, 1};
            for (int k = 0; k < 4; ++k)
            {
                const m::Vec3 p = fc.n * 0.5f + fc.u * (0.5f * su[k]) + fc.v * (0.5f * sv[k]);
                md.vertices.push_back({p.x, p.y, p.z, fc.n.x, fc.n.y, fc.n.z});
            }
            const int b = f * 4;
            for (const int idx : {0, 1, 2, 0, 2, 3}) md.indices.push_back(u16(b + idx));
        }
        return md;
    }

    // Flat grid in the xz plane (y = 0), normal +y.
    MeshData make_grid(float half, int cells)
    {
        MeshData md;
        for (int j = 0; j <= cells; ++j)
            for (int i = 0; i <= cells; ++i)
                md.vertices.push_back({-half + 2.0f * half * fl(i) / fl(cells), 0.0f, -half + 2.0f * half * fl(j) / fl(cells), 0.0f, 1.0f, 0.0f});
        for (int j = 0; j < cells; ++j)
            for (int i = 0; i < cells; ++i)
            {
                const int a = j * (cells + 1) + i, b = a + cells + 1, c = a + 1, d = b + 1;
                for (const int idx : {a, b, c, c, b, d}) md.indices.push_back(u16(idx));
            }
        return md;
    }

    m::Mat4 sphere_model(const m::Vec3& c) { return m::translate(c) * m::scale({kSphereRadius, kSphereRadius, kSphereRadius}); }
    m::Mat4 box_model()
    {
        return m::translate(kBoxPos) * m::rotate(m::radians(kBoxAngleDeg), {0.0f, 1.0f, 0.0f}) * m::scale({kBoxSize, kBoxSize, kBoxSize});
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference (double precision), used only by validate(). Its own camera model (no Mat4), exact
    // ray intersections with the analytic sphere / box / plane, the GLSL ES 3.00 section 8.5 formulas
    // written out, and the cube map sampled through select_face() on the CPU copy of the uploaded texels.
    // ---------------------------------------------------------------------------------------------
    struct Ray
    {
        D3 o, d;
    };

    struct Hit
    {
        Obj obj = Obj::Sky;
        double t = 1e30;  // = eye distance (unit ray direction from the eye)
        D3 pos{}, normal{}, dir{};
        long tag = 0;  // ground tile or box face: the colour is discontinuous where it changes
        double n_dot_v = 1.0;
    };

    // What the CPU evaluates. Defaults follow the frame; the switches build the wrong models of the
    // negative controls.
    struct Model
    {
        const CubeImage* cube = nullptr;
        double eta = kEta;
        double fog_density = 0.0;
        bool reflect = true;   // false: mirrors sample along I (as if the surface were transparent)
        bool fresnel = true;   // false: glass shows the refraction only
        bool flip_s = false, flip_t = false;
    };

    // GLSL ES 3.00 section 8.5: reflect(I, N) = I - 2 dot(N, I) N
    D3 glsl_reflect(D3 i, D3 n) { return i - n * (2.0 * dot(n, i)); }

    // refract(I, N, eta): k = 1 - eta^2 (1 - dot(N, I)^2); k < 0 -> 0, else eta I - (eta dot(N, I) + sqrt(k)) N
    D3 glsl_refract(D3 i, D3 n, double eta)
    {
        const double ni = dot(n, i);
        const double k = 1.0 - eta * eta * (1.0 - ni * ni);
        if (k < 0.0) return {};
        return i * eta - n * (eta * ni + std::sqrt(k));
    }

    // Per-channel 8-bit interval of acceptable colours: the min / max over a small neighbourhood of the
    // exact direction or normal. A probe on a texel boundary (or on a cube edge, where ES 3.0 lets the
    // implementation break ties) therefore accepts either neighbouring texel instead of being skipped.
    struct Envelope
    {
        std::array<int, 3> lo{255, 255, 255}, hi{0, 0, 0};

        void add(D3 c)
        {
            const std::array<int, 3> v = {to8(c.x), to8(c.y), to8(c.z)};
            for (std::size_t k = 0; k < 3; ++k)
            {
                lo[k] = std::min(lo[k], v[k]);
                hi[k] = std::max(hi[k], v[k]);
            }
        }
        // How far a pixel lies outside the interval (0 = inside).
        [[nodiscard]] int distance(demo::Rgba8 px) const
        {
            int d = 0;
            for (std::size_t k = 0; k < 3; ++k)
            {
                const int v = px[k];
                d = std::max({d, lo[k] - v, v - hi[k]});
            }
            return d;
        }
        [[nodiscard]] int width() const { return std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}); }
        // Smallest per-channel gap between two intervals (0 = they overlap in every channel).
        [[nodiscard]] int separation(const Envelope& o) const
        {
            int s = 0;
            for (std::size_t k = 0; k < 3; ++k) s = std::max({s, o.lo[k] - hi[k], lo[k] - o.hi[k]});
            return s;
        }
        [[nodiscard]] std::string text() const
        {
            if (width() == 0) return std::format("({},{},{})", lo[0], lo[1], lo[2]);
            return std::format("({}..{},{}..{},{}..{})", lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        }
    };

    // Angular half-widths of the neighbourhoods. kSkyDelta covers float rounding of the interpolated
    // direction and of the face coordinates only. kSphereDelta covers the polygon approximation: the GPU
    // normal is normalize(P) of the point P where the ray meets a flat triangle of the 128 x 64 sphere,
    // at most the triangle's sagitta (6e-4 r) inside the analytic sphere; for n.v >= 0.3 (the sweep's
    // limit) the angle between P and the analytic hit seen from the centre is <= 6e-4 * tan(72.5 deg)
    // = 1.9e-3 rad < 3e-3.
    constexpr double kSkyDelta = 5e-4;
    constexpr double kSphereDelta = 3e-3;
    constexpr double kBoxDelta = 5e-4;
    constexpr double kMinNDotV = 0.3;

    // Calls fn(v') for a 5 x 5 grid of unit vectors around the unit vector v (half-width delta radians).
    template <typename Fn>
    void around(D3 v, double delta, Fn&& fn)
    {
        const D3 helper = std::abs(v.y) < 0.9 ? D3{0.0, 1.0, 0.0} : D3{1.0, 0.0, 0.0};
        const D3 t1 = normalize(cross(v, helper)), t2 = cross(v, t1);
        for (int a = -2; a <= 2; ++a)
            for (int b = -2; b <= 2; ++b) fn(normalize(v + t1 * (0.5 * a * delta) + t2 * (0.5 * b * delta)));
    }

    class Reference
    {
    public:
        Reference(const Frame& f, int width, int height) : w_(width), h_(height)
        {
            eye_ = d3(f.eye);
            fwd_ = normalize(d3(f.center) - eye_);
            right_ = normalize(cross(fwd_, d3(f.up_hint)));
            up_ = cross(right_, fwd_);
            tan_half_ = std::tan(0.5 * dbl(f.fovy));
            aspect_ = static_cast<double>(width) / static_cast<double>(height);
        }

        bool sky_everywhere = false;  // depth-trick negative control: the sky cube covers every object


        // Ray through the centre of window pixel (px, py), y counted from the bottom like glReadPixels.
        [[nodiscard]] Ray ray(int px, int py) const
        {
            const double nx = (px + 0.5) / w_ * 2.0 - 1.0;
            const double ny = (py + 0.5) / h_ * 2.0 - 1.0;
            return {eye_, normalize(fwd_ + right_ * (nx * tan_half_ * aspect_) + up_ * (ny * tan_half_))};
        }

        // Projection of a world point to pixel-centre coordinates.
        [[nodiscard]] bool project(D3 p, double& px, double& py) const
        {
            const D3 v = p - eye_;
            const double z = dot(v, fwd_);
            if (z <= 0.0) return false;
            px = (dot(v, right_) / (z * tan_half_ * aspect_) + 1.0) * 0.5 * w_ - 0.5;
            py = (dot(v, up_) / (z * tan_half_) + 1.0) * 0.5 * h_ - 0.5;
            return true;
        }

        [[nodiscard]] Hit trace(const Ray& r) const
        {
            Hit best;
            best.dir = r.d;
            if (sky_everywhere) return best;
            auto take = [&](Hit h) {
                h.dir = r.d;
                h.n_dot_v = -dot(h.normal, r.d);
                if (h.t < best.t) best = h;
            };
            if (r.d.y < -1e-12)  // ground y = 0
            {
                const double t = -r.o.y / r.d.y;
                const D3 p = r.o + r.d * t;
                if (std::abs(p.x) <= dbl(kGroundHalf) && std::abs(p.z) <= dbl(kGroundHalf))
                {
                    Hit h;
                    h.obj = Obj::Ground;
                    h.t = t;
                    h.pos = p;
                    h.normal = {0.0, 1.0, 0.0};
                    h.tag = static_cast<long>(std::floor(p.x / dbl(kTile))) * 100003L + static_cast<long>(std::floor(p.z / dbl(kTile)));
                    take(h);
                }
            }
            for (const Obj o : {Obj::Chrome, Obj::Glass})
            {
                const D3 c = d3(o == Obj::Chrome ? kChromePos : kGlassPos);
                const double rr = kSphereRadius;
                const D3 oc = r.o - c;
                const double b = dot(oc, r.d), disc = b * b - (dot(oc, oc) - rr * rr);
                if (disc < 0.0) continue;
                const double t = -b - std::sqrt(disc);
                if (t <= 1e-9) continue;
                Hit h;
                h.obj = o;
                h.t = t;
                h.pos = r.o + r.d * t;
                h.normal = (h.pos - c) * (1.0 / rr);
                take(h);
            }
            box(r, take);
            return best;
        }

        // A probe is stable when the hits at +-2 pixels belong to the same object (and the same ground tile /
        // box face): it is then away from silhouettes and colour edges, where the rasterization rule decides.
        [[nodiscard]] bool stable(int px, int py, const Hit& h) const
        {
            if ((h.obj == Obj::Chrome || h.obj == Obj::Glass) && h.n_dot_v < kMinNDotV) return false;
            for (int dy = -2; dy <= 2; dy += 2)
                for (int dx = -2; dx <= 2; dx += 2)
                {
                    if (dx == 0 && dy == 0) continue;
                    const Hit n = trace(ray(px + dx, py + dy));
                    if (n.obj != h.obj) return false;
                    if ((h.obj == Obj::Ground || h.obj == Obj::Box) && n.tag != h.tag) return false;
                }
            return true;
        }

        [[nodiscard]] static D3 ground_color(D3 p)
        {
            const long cx = static_cast<long>(std::floor(p.x / dbl(kTile))), cz = static_cast<long>(std::floor(p.z / dbl(kTile)));
            const D3 albedo = d3(((cx + cz) & 1) != 0 ? kAlbedoB : kAlbedoA);
            return albedo * (d3(kAmbient) + d3(kSunColor) * std::max(sun_direction().y, 0.0));
        }

        [[nodiscard]] static D3 surface_color(const Model& md, Obj o, D3 i, D3 n)
        {
            auto env = [&](D3 dir) { return md.cube->nearest(dir, md.flip_s, md.flip_t); };
            const D3 r = md.reflect ? glsl_reflect(i, n) : i;
            if (o == Obj::Chrome) return d3(kChromeTint) * env(r);
            if (o == Obj::Box) return d3(kGoldTint) * env(r);
            D3 t = glsl_refract(i, n, md.eta);
            if (dot(t, t) == 0.0) t = r;  // total internal reflection, as in scene.frag
            const double mm = 1.0 - std::clamp(dot(i * -1.0, n), 0.0, 1.0);
            const double fresnel = md.fresnel ? dbl(kF0) + (1.0 - dbl(kF0)) * std::pow(mm, 5.0) : 0.0;
            return mix(d3(kGlassTint) * env(t), env(r), fresnel);
        }

        [[nodiscard]] static D3 fogged(const Model& md, D3 c, double distance)
        {
            const double fd = md.fog_density * distance;
            return mix(d3(kFogColor), c, std::exp(-fd * fd));
        }

        [[nodiscard]] Envelope expected(const Hit& h, const Model& md) const
        {
            Envelope e;
            switch (h.obj)
            {
            case Obj::Sky: around(h.dir, kSkyDelta, [&](D3 d) { e.add(md.cube->nearest(d, md.flip_s, md.flip_t)); }); break;
            case Obj::Ground: e.add(fogged(md, ground_color(h.pos), h.t)); break;
            default:
                around(h.normal, h.obj == Obj::Box ? kBoxDelta : kSphereDelta,
                       [&](D3 n) { e.add(fogged(md, surface_color(md, h.obj, h.dir, n), h.t)); });
                break;
            }
            return e;
        }

    private:
        template <typename Take>
        void box(const Ray& r, Take& take) const
        {
            const double a = rad(kBoxAngleDeg), c = std::cos(a), s = std::sin(a);
            const D3 centre = d3(kBoxPos);
            // world -> object: q = R^-1 (p - centre) / size, with R the rotation about y (x' = c x + s z, z' = -s x + c z)
            auto to_obj = [&](D3 v) { return D3{c * v.x - s * v.z, v.y, s * v.x + c * v.z} * (1.0 / dbl(kBoxSize)); };
            const D3 o = to_obj(r.o - centre), d = to_obj(r.d);
            const double oa[3] = {o.x, o.y, o.z}, da[3] = {d.x, d.y, d.z};
            double t_in = -1e30, t_out = 1e30;
            int axis = 0;
            for (int k = 0; k < 3; ++k)
            {
                if (std::abs(da[k]) < 1e-14)
                {
                    if (std::abs(oa[k]) > 0.5) return;
                    continue;
                }
                const double t1 = (-0.5 - oa[k]) / da[k], t2 = (0.5 - oa[k]) / da[k];
                const double lo = std::min(t1, t2), hi = std::max(t1, t2);
                if (lo > t_in)
                {
                    t_in = lo;
                    axis = k;
                }
                t_out = std::min(t_out, hi);
            }
            if (t_in > t_out || t_in <= 1e-9) return;
            const double sign = da[axis] < 0.0 ? 1.0 : -1.0;  // the ray enters through the face it moves against
            D3 n_obj{};
            (axis == 0 ? n_obj.x : axis == 1 ? n_obj.y : n_obj.z) = sign;
            Hit h;
            h.obj = Obj::Box;
            h.t = t_in;
            h.pos = r.o + r.d * t_in;
            h.normal = {c * n_obj.x + s * n_obj.z, n_obj.y, -s * n_obj.x + c * n_obj.z};
            h.tag = axis * 2 + (sign > 0.0 ? 1 : 0);
            take(h);
        }

        int w_, h_;
        D3 eye_{}, fwd_{}, right_{}, up_{};
        double tan_half_ = 1.0, aspect_ = 1.0;
    };

    // ---------------------------------------------------------------------------------------------
    // GL objects
    // ---------------------------------------------------------------------------------------------
    struct GpuMesh
    {
        mgl::VertexArrayId vao{};
        mgl::BufferId vbo{};
        mgl::BufferId ebo{};
        int index_count = 0;
    };

    struct SceneUniforms
    {
        mgl::UniformLocation model, view, proj, sky, eye, material, tint, eta, f0;
        mgl::UniformLocation albedo_a, albedo_b, tile, ambient, sun_dir, sun_color, fog_color, fog_density;
    };

    struct SkyUniforms
    {
        mgl::UniformLocation view, proj, sky, depth_trick;
    };

    void uniform3(mgl::UniformLocation l, const m::Vec3& v) { mgl::glUniform3f(l, v.x, v.y, v.z); }

    constexpr std::array<float, 4> kClearColor = {0.0f, 0.0f, 0.0f, 1.0f};

    struct Img
    {
        int w = 0, h = 0;
        std::vector<std::uint8_t> px;  // RGBA8, rows bottom-up like glReadPixels
        [[nodiscard]] demo::Rgba8 at(int x, int y) const
        {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
            return {px[i], px[i + 1], px[i + 2], px[i + 3]};
        }
    };

    class SkyboxAndFog final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 128 || env.height < 128)
                throw demo::Skip(std::format("surface {}x{} is too small for the probe layout (needs at least 128x128)", env.width, env.height));

            scene_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            sky_program_ = demo::build_program(env, "shaders/sky.vert", "shaders/sky.frag");
            probe_ = demo::build_program(env, "shaders/probe.vert", "shaders/probe.frag");
            auto loc = [](mgl::ProgramId p, const char* name) { return mgl::glGetUniformLocation(p, name); };
            su_ = {loc(scene_, "u_model"),    loc(scene_, "u_view"),     loc(scene_, "u_proj"),        loc(scene_, "u_sky"),
                   loc(scene_, "u_eye"),      loc(scene_, "u_material"), loc(scene_, "u_tint"),        loc(scene_, "u_eta"),
                   loc(scene_, "u_f0"),       loc(scene_, "u_albedo_a"), loc(scene_, "u_albedo_b"),    loc(scene_, "u_tile"),
                   loc(scene_, "u_ambient"),  loc(scene_, "u_sun_dir"),  loc(scene_, "u_sun_color"),   loc(scene_, "u_fog_color"),
                   loc(scene_, "u_fog_density")};
            ku_ = {loc(sky_program_, "u_view"), loc(sky_program_, "u_proj"), loc(sky_program_, "u_sky"), loc(sky_program_, "u_depth_trick")};
            probe_cube_ = loc(probe_, "u_cube");
            probe_dir_ = loc(probe_, "u_dir");

            // Constant uniforms. Every samplerCube reads texture unit 0.
            mgl::glUseProgram(scene_);
            mgl::glUniform1i(su_.sky, 0);
            mgl::glUniform1f(su_.f0, kF0);
            uniform3(su_.albedo_a, kAlbedoA);
            uniform3(su_.albedo_b, kAlbedoB);
            mgl::glUniform1f(su_.tile, kTile);
            uniform3(su_.ambient, kAmbient);
            const D3 sun = sun_direction();
            mgl::glUniform3f(su_.sun_dir, static_cast<float>(sun.x), static_cast<float>(sun.y), static_cast<float>(sun.z));
            uniform3(su_.sun_color, kSunColor);
            uniform3(su_.fog_color, kFogColor);
            mgl::glUseProgram(sky_program_);
            mgl::glUniform1i(ku_.sky, 0);
            mgl::glUseProgram(probe_);
            mgl::glUniform1i(probe_cube_, 0);
            mgl::glUseProgram(mgl::ProgramId{0});

            // The three cube maps, painted on the CPU. The CPU copies stay for the reference.
            sky_image_ = make_sky_image();
            face_image_ = make_face_image();
            seam_image_ = make_seam_image();
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            sky_ = upload_cube(sky_image_, kSkyLevels, {0, 1, 2, 3, 4, 5});
            mgl::glGenerateMipmap(mgl::TextureTarget::TextureCubeMap);  // trilinear sky and reflections in the windowed configuration
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::LinearMipmapLinear);
            faces_ = upload_cube(face_image_, 1, {0, 1, 2, 3, 4, 5});
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::Nearest);
            seam_ = upload_cube(seam_image_, 1, {0, 1, 2, 3, 4, 5});
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::Linear);
            // REPEAT (also the default) on purpose: cube maps ignore the wrap modes (ES 3.0 section 3.8.9.1). If they
            // did not, a lookup on the right edge of a face would wrap to its own left edge, i.e. the same colour.
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::Repeat);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::Repeat);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});

            mesh_data_[static_cast<std::size_t>(MeshId::Ground)] = make_grid(kGroundHalf, kGroundCells);
            mesh_data_[static_cast<std::size_t>(MeshId::Sphere)] = make_sphere(kSlices, kStacks);
            mesh_data_[static_cast<std::size_t>(MeshId::Cube)] = make_cube();
            for (std::size_t i = 0; i < meshes_.size(); ++i) upload(meshes_[i], mesh_data_[i]);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            draw(make_frame(params_, aspect(env)), env.width, env.height);
            if (!params_.paused) ++params_.ticks;
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
            case ' ': params_.paused = !params_.paused; break;
            case 'a': params_.yaw -= 0.08f; break;
            case 'd': params_.yaw += 0.08f; break;
            case 'w': params_.distance = std::max(5.0f, params_.distance - 0.5f); break;
            case 's': params_.distance = std::min(30.0f, params_.distance + 0.5f); break;
            case 'q': params_.elevation = std::min(60.0f, params_.elevation + 3.0f); break;
            case 'e': params_.elevation = std::max(1.0f, params_.elevation - 3.0f); break;
            case 'f': params_.fog_density = std::min(0.5f, params_.fog_density * 1.25f); break;
            case 'g': params_.fog_density = std::max(0.002f, params_.fog_density / 1.25f); break;
            case 'o': params_.fog_on = !params_.fog_on; break;
            case 'c': params_.face_coded = !params_.face_coded; break;
            case 'n': params_.nearest = !params_.nearest; break;
            case 'k': params_.depth_trick = !params_.depth_trick; break;
            case 'i': params_.inverse_eta = !params_.inverse_eta; break;
            case 'r': params_ = Params{}; break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
            for (GpuMesh& mesh : meshes_)
            {
                mgl::glDeleteVertexArrays(1, &mesh.vao);
                mgl::glDeleteBuffers(1, &mesh.vbo);
                mgl::glDeleteBuffers(1, &mesh.ebo);
            }
            const std::array<mgl::TextureId, 3> textures = {sky_, faces_, seam_};
            mgl::glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
            mgl::glDeleteProgram(scene_);
            mgl::glDeleteProgram(sky_program_);
            mgl::glDeleteProgram(probe_);
        }

    private:
        static float aspect(const demo::Env& env) { return static_cast<float>(env.width) / static_cast<float>(env.height); }

        // Immutable RGBA8 cube map; face f receives the image of face order[f] (the identity except in the
        // swapped-face negative control). Leaves the texture bound to TEXTURE_CUBE_MAP on the active unit.
        static mgl::TextureId upload_cube(const CubeImage& img, int levels, const std::array<int, 6>& order)
        {
            mgl::TextureId tex{};
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, levels, mgl::SizedInternalFormat::Rgba8, img.size, img.size);
            for (std::size_t f = 0; f < 6; ++f)
                mgl::glTexSubImage2D(kFaceTargets[f], 0, 0, 0, img.size, img.size, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                     img.texels[static_cast<std::size_t>(order[f])].data());
            return tex;
        }

        static void upload(GpuMesh& mesh, const MeshData& md)
        {
            mgl::glGenVertexArrays(1, &mesh.vao);
            mgl::glBindVertexArray(mesh.vao);
            mgl::glGenBuffers(1, &mesh.vbo);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mesh.vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(md.vertices), mgl::BufferUsage::StaticDraw);
            const mgl::AttribLocation a_position{0u}, a_normal{1u};  // layout(location = 0 / 1)
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, px)));
            mgl::glEnableVertexAttribArray(a_normal);
            mgl::glVertexAttribPointer(a_normal, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, nx)));
            mgl::glGenBuffers(1, &mesh.ebo);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mesh.ebo);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(md.indices), mgl::BufferUsage::StaticDraw);
            mesh.index_count = static_cast<int>(md.indices.size());
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void draw_mesh(MeshId id) const
        {
            const GpuMesh& mesh = meshes_[static_cast<std::size_t>(id)];
            mgl::glBindVertexArray(mesh.vao);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, mesh.index_count, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
        }

        void draw_object(MeshId mesh, const m::Mat4& model, int material, const m::Vec3& tint) const
        {
            mgl::glUniformMatrix4fv(su_.model, 1, GL_FALSE, model.data());
            mgl::glUniform1i(su_.material, material);
            uniform3(su_.tint, tint);
            draw_mesh(mesh);
        }

        // One frame: ground and objects with depth test LESS, then the sky LAST with LEQUAL at depth 1.0.
        // `env_override` replaces the cube map (swapped-face negative control); `sky_only` skips the scene.
        void draw(const Frame& f, int width, int height, bool sky_only = false, mgl::TextureId env_override = mgl::TextureId{0}) const
        {
            mgl::glViewport(0, 0, width, height);
            mgl::glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], kClearColor[3]);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);

            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, sky_);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, f.nearest ? mgl::TextureMagFilter::Nearest : mgl::TextureMagFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, f.nearest ? mgl::TextureMinFilter::Nearest : mgl::TextureMinFilter::LinearMipmapLinear);
            if (env_override.value != 0) mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, env_override);
            else if (f.face_coded) mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, faces_);

            if (!sky_only)
            {
                mgl::glDepthFunc(mgl::CompareFunc::Less);
                mgl::glUseProgram(scene_);
                mgl::glUniformMatrix4fv(su_.view, 1, GL_FALSE, f.view.data());
                mgl::glUniformMatrix4fv(su_.proj, 1, GL_FALSE, f.proj.data());
                uniform3(su_.eye, f.eye);
                mgl::glUniform1f(su_.eta, f.eta);
                mgl::glUniform1f(su_.fog_density, f.fog_density);
                draw_object(MeshId::Ground, m::Mat4::identity(), 0, {1.0f, 1.0f, 1.0f});
                draw_object(MeshId::Sphere, sphere_model(kChromePos), 1, kChromeTint);
                draw_object(MeshId::Sphere, sphere_model(kGlassPos), 2, kGlassTint);
                draw_object(MeshId::Cube, box_model(), 1, kGoldTint);
            }

            // The sky is drawn last: its fragments sit at depth 1.0 and LEQUAL lets them through only where
            // the depth buffer still holds the cleared 1.0, so the expensive sky shader runs only on visible pixels.
            mgl::glDepthFunc(mgl::CompareFunc::Lequal);
            mgl::glUseProgram(sky_program_);
            mgl::glUniformMatrix4fv(ku_.view, 1, GL_FALSE, f.view.data());
            mgl::glUniformMatrix4fv(ku_.proj, 1, GL_FALSE, f.proj.data());
            mgl::glUniform1i(ku_.depth_trick, f.depth_trick ? 1 : 0);
            draw_mesh(MeshId::Cube);

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDepthFunc(mgl::CompareFunc::Less);
        }

        Model model_of(const Frame& f) const
        {
            Model md;
            md.cube = f.face_coded ? &face_image_ : &sky_image_;
            md.eta = dbl(f.eta);
            md.fog_density = dbl(f.fog_density);
            return md;
        }

        mgl::ProgramId scene_{}, sky_program_{}, probe_{};
        SceneUniforms su_{};
        SkyUniforms ku_{};
        mgl::UniformLocation probe_cube_{}, probe_dir_{};
        mgl::TextureId sky_{}, faces_{}, seam_{};
        CubeImage sky_image_, face_image_, seam_image_;
        std::array<GpuMesh, static_cast<std::size_t>(MeshId::Count)> meshes_{};
        std::array<MeshData, static_cast<std::size_t>(MeshId::Count)> mesh_data_{};
        Params params_{};
    };

    // ---------------------------------------------------------------------------------------------
    // Validation helpers
    // ---------------------------------------------------------------------------------------------
    // Tolerance, 8-bit channels. Under NEAREST every texture lookup returns an exact uploaded RGBA8 texel
    // (the envelope already contains the texels a boundary probe may legitimately select). What remains
    // is highp arithmetic (GLSL ES 3.00 section 4.5.1: a few ULP for mix, exp, normalize, far below one
    // 8-bit step for these formulas) and the float -> unorm conversion of the output, which may return
    // either of the two closest values (ES 3.0 section 2.1.6.2): +-1, plus 1 for the products of two
    // rounded quantities (tint * texel, fog blends). kTol = 2.
    constexpr int kTol = 2;
    constexpr int kDiscriminate = 12;  // a wrong model must be off by at least this much for a pixel to count as discriminating

    std::string rgba_text(demo::Rgba8 c) { return demo::format_rgba(c); }

    int sweep_step(int w, int h) { return std::max(3, std::min(w, h) / 64); }

    struct SweepStat
    {
        int n = 0, bad = 0, max_d = 0, wx = -1, wy = -1, widest = 0, tight = 0;  // tight: expected interval at most 2 steps wide
        demo::Rgba8 got{};
        Envelope want{};
    };

    // Every `step`-th stable pixel, grouped by the object the CPU ray hits, compared with the model.
    std::array<SweepStat, static_cast<std::size_t>(Obj::Count)> sweep(const Reference& ref, const Img& img, const Model& md, int step)
    {
        std::array<SweepStat, static_cast<std::size_t>(Obj::Count)> st{};
        for (int y = 3; y < img.h - 3; y += step)
            for (int x = 3; x < img.w - 3; x += step)
            {
                const Hit h = ref.trace(ref.ray(x, y));
                if (!ref.stable(x, y, h)) continue;
                SweepStat& s = st[static_cast<std::size_t>(h.obj)];
                const Envelope want = ref.expected(h, md);
                const demo::Rgba8 got = img.at(x, y);
                const int d = want.distance(got);
                ++s.n;
                s.widest = std::max(s.widest, want.width());
                if (want.width() <= 2) ++s.tight;
                if (d > kTol) ++s.bad;
                if (d > s.max_d || s.wx < 0)
                {
                    s.max_d = std::max(s.max_d, d);
                    s.wx = x;
                    s.wy = y;
                    s.got = got;
                    s.want = want;
                }
            }
        return st;
    }

    // Pixels of one object where a wrong model is clearly separated from the right one, and how many of
    // them the image rejects (outside the wrong model's interval by more than kTol).
    struct Discrimination
    {
        int candidates = 0, discriminating = 0, rejected = 0, accepted_right = 0;
    };

    Discrimination discriminate(const Reference& ref, const Img& img, Obj obj, const Model& right, const Model& wrong, int step)
    {
        Discrimination r;
        for (int y = 3; y < img.h - 3; y += step)
            for (int x = 3; x < img.w - 3; x += step)
            {
                const Hit h = ref.trace(ref.ray(x, y));
                if (h.obj != obj || !ref.stable(x, y, h)) continue;
                ++r.candidates;
                const Envelope e_right = ref.expected(h, right), e_wrong = ref.expected(h, wrong);
                if (e_right.separation(e_wrong) < kDiscriminate) continue;
                ++r.discriminating;
                const demo::Rgba8 got = img.at(x, y);
                if (e_wrong.distance(got) > kTol) ++r.rejected;
                if (e_right.distance(got) <= kTol) ++r.accepted_right;
            }
        return r;
    }

    struct Probe
    {
        bool ok = false;
        int x = 0, y = 0;
        Hit hit;
    };

    // The pixel a world point projects to, valid when the CPU ray through that pixel centre hits `expect`
    // and the pixel is stable.
    Probe make_probe(const Reference& ref, int w, int h, D3 world, Obj expect)
    {
        Probe p;
        double fx = 0.0, fy = 0.0;
        if (!ref.project(world, fx, fy)) return p;
        p.x = static_cast<int>(std::lround(fx));
        p.y = static_cast<int>(std::lround(fy));
        if (p.x < 3 || p.y < 3 || p.x > w - 4 || p.y > h - 4) return p;
        p.hit = ref.trace(ref.ray(p.x, p.y));
        p.ok = p.hit.obj == expect && ref.stable(p.x, p.y, p.hit);
        return p;
    }

    void SkyboxAndFog::validate(demo::Env& env, demo::Check& check)
    {
        const int W = env.width, H = env.height;
        const float asp = aspect(env);
        const int step = sweep_step(W, H);
        auto snapshot = [&](const Frame& f, bool sky_only = false, mgl::TextureId tex = mgl::TextureId{0}) {
            draw(f, W, H, sky_only, tex);
            return Img{W, H, demo::read_rgba8(0, 0, W, H)};
        };
        Params base{};
        base.nearest = true;  // the validation configuration: NEAREST sky lookups, t = 0
        const Frame f_base = make_frame(base, asp);

        // ---- A. Texture state -----------------------------------------------------------------------------------------
        {
            draw(f_base, W, H);
            GLint bound = 0, min_filter = 0, mag_filter = 0, max_size = 0;
            mgl::glGetIntegerv(mgl::GetParameter::TextureBindingCubeMap, &bound);
            mgl::glGetTexParameteriv(mgl::TextureTarget::TextureCubeMap, mgl::TextureParameterQuery::MinFilter, &min_filter);
            mgl::glGetTexParameteriv(mgl::TextureTarget::TextureCubeMap, mgl::TextureParameterQuery::MagFilter, &mag_filter);
            mgl::glGetIntegerv(mgl::GetParameter::MaxCubeMapTextureSize, &max_size);
            check.expect(static_cast<GLuint>(bound) == sky_.value, "TEXTURE_BINDING_CUBE_MAP names the sky cube map while the scene is drawn",
                         std::format("bound {} sky {}", bound, sky_.value));
            check.expect(min_filter == GL_NEAREST && mag_filter == GL_NEAREST,
                         "validation configuration: the sky cube map's MIN_FILTER and MAG_FILTER read back NEAREST",
                         std::format("min 0x{:x} mag 0x{:x}", min_filter, mag_filter));
            check.expect(max_size >= kSkySize, "MAX_CUBE_MAP_TEXTURE_SIZE admits the 512 x 512 sky faces (ES 3.0 Table 6.28 requires >= 2048)",
                         std::format("MAX_CUBE_MAP_TEXTURE_SIZE = {}", max_size));
        }

        // ---- B. Face orientation: the face-coded cube map seen along each axis ---------------------------------------------
        {
            constexpr int kGrid = 9;
            int flip_s_rejected = 0, flip_t_rejected = 0, flip_s_disc = 0, flip_t_disc = 0;
            for (int face = 0; face < 6; ++face)
            {
                const Frame f = face_frame(face, asp);
                Reference ref(f, W, H);
                ref.sky_everywhere = true;  // sky only: no ground, no objects
                const Img img = snapshot(f, true);
                Model md = model_of(f);
                Model fs = md, ft = md;
                fs.flip_s = true;
                ft.flip_t = true;
                int n = 0, bad = 0, exact = 0, worst = 0;
                std::string worst_text;
                for (int gy = 0; gy < kGrid; ++gy)
                    for (int gx = 0; gx < kGrid; ++gx)
                    {
                        const int x = (2 * gx + 1) * W / (2 * kGrid), y = (2 * gy + 1) * H / (2 * kGrid);
                        const Hit h = ref.trace(ref.ray(x, y));
                        const Envelope e = ref.expected(h, md);
                        const demo::Rgba8 got = img.at(x, y);
                        const int d = e.distance(got);
                        ++n;
                        if (e.width() == 0) ++exact;
                        if (d > kTol) ++bad;
                        if (d >= worst)
                        {
                            worst = d;
                            worst_text = std::format("pixel({},{}) = {} expected {}", x, y, rgba_text(got), e.text());
                        }
                        for (const auto& [wrong, disc, rej] : {std::tuple{&fs, &flip_s_disc, &flip_s_rejected}, std::tuple{&ft, &flip_t_disc, &flip_t_rejected}})
                        {
                            const Envelope ew = ref.expected(h, *wrong);
                            if (ew.separation(e) < kDiscriminate) continue;
                            ++*disc;
                            if (ew.distance(got) > kTol) ++*rej;
                        }
                    }
                check.expect(bad == 0 && exact >= n / 2,
                             std::format("looking along {}: {} sky probes match the CPU model of Table 3.21 (face-coded cube map, NEAREST)", kFaceNames[static_cast<std::size_t>(face)], n),
                             std::format("{} mismatches, {} probes away from texel/face boundaries, worst: {}", bad, exact, worst_text));
                // Model-free statement: the screen centre looks exactly along the axis, i.e. the centre texel (2, 2) of the
                // image that was uploaded to this face's target.
                const D3 c = face_image_.texel(face, kFaceGrid / 2, kFaceGrid / 2);
                const demo::Rgba8 want{to8(c.x), to8(c.y), to8(c.z), 255};
                check.expect_pixel(W / 2, H / 2, want, 1,
                                   std::format("looking along {}: the screen centre shows the centre texel of the image uploaded to TEXTURE_CUBE_MAP_{}",
                                               kFaceNames[static_cast<std::size_t>(face)],
                                               std::array{"POSITIVE_X", "NEGATIVE_X", "POSITIVE_Y", "NEGATIVE_Y", "POSITIVE_Z", "NEGATIVE_Z"}[static_cast<std::size_t>(face)]));
            }
            check.expect(flip_s_disc >= 60 && flip_s_rejected == flip_s_disc,
                         "face orientation: a model with mirrored s (sc sign flipped in Table 3.21) is rejected by every discriminating probe",
                         std::format("{} of {} discriminating probes reject it", flip_s_rejected, flip_s_disc));
            check.expect(flip_t_disc >= 60 && flip_t_rejected == flip_t_disc,
                         "face orientation: a model with mirrored t (tc sign flipped) is rejected by every discriminating probe",
                         std::format("{} of {} discriminating probes reject it", flip_t_rejected, flip_t_disc));

            // Negative control: the +X and -X images uploaded to each other's targets.
            CubeImage swapped = face_image_;
            std::swap(swapped.texels[0], swapped.texels[1]);
            const mgl::TextureId swap_tex = upload_cube(face_image_, 1, {1, 0, 2, 3, 4, 5});
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::Nearest);
            const Frame f = face_frame(0, asp);
            Reference ref(f, W, H);
            ref.sky_everywhere = true;
            const Img img = snapshot(f, true, swap_tex);
            Model right = model_of(f), wrong = right;
            wrong.cube = &swapped;
            // Probes that land on the +X face distinguish the two models; on a very wide surface the outer probes see
            // the +-Y / +-Z faces, which the swap does not change.
            int n = 0, disc = 0, rejected = 0, matches_swapped = 0;
            for (int gy = 0; gy < kGrid; ++gy)
                for (int gx = 0; gx < kGrid; ++gx)
                {
                    const int x = (2 * gx + 1) * W / (2 * kGrid), y = (2 * gy + 1) * H / (2 * kGrid);
                    const Hit h = ref.trace(ref.ray(x, y));
                    const demo::Rgba8 got = img.at(x, y);
                    const Envelope e_right = ref.expected(h, right), e_wrong = ref.expected(h, wrong);
                    ++n;
                    if (e_right.separation(e_wrong) >= kDiscriminate)
                    {
                        ++disc;
                        if (e_right.distance(got) > kTol) ++rejected;
                    }
                    if (e_wrong.distance(got) <= kTol) ++matches_swapped;
                }
            check.expect(disc >= n / 3 && rejected == disc,
                         "negative control: with the +X and -X images swapped, every probe on the +X face fails the correct model",
                         std::format("{} of {} discriminating probes (of {}) rejected", rejected, disc, n));
            check.expect(matches_swapped == n, "negative control: ... and every probe matches the model of the swapped upload",
                         std::format("{} of {} probes match", matches_swapped, n));
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &swap_tex);
        }

        // ---- C. Seamless filtering exactly on cube edges (ES 3.0 section 3.8.9.1) ---------------------------------------------
        {
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glViewport(0, 0, 4, 4);
            mgl::glUseProgram(probe_);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, seam_);
            auto sample = [&](D3 dir) {
                mgl::glUniform3f(probe_dir_, static_cast<float>(dir.x), static_cast<float>(dir.y), static_cast<float>(dir.z));
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                return demo::read_pixel(1, 1);
            };
            auto base_color = [](int face) {
                const auto& b = kFaceBase[static_cast<std::size_t>(face)];
                return std::array<int, 3>{b[0], b[1], b[2]};
            };
            struct Edge
            {
                D3 dir;
                int a, b;
            };
            // The direction lies exactly on the edge between faces a and b: whichever face the tie rule selects, s
            // (or t) is exactly 0 or 1, the LINEAR footprint straddles the edge with weights 1/2 and 1/2, and the
            // texels beyond the edge come from the neighbouring face. Both faces are uniform, so the result is
            // (a + b) / 2. A non-seamless (ES 2.0) implementation would clamp or wrap within one face: a or b.
            for (const Edge& e : {Edge{{1.0, 0.0, 1.0}, 0, 4}, Edge{{0.0, 1.0, -1.0}, 2, 5}, Edge{{-1.0, -1.0, 0.0}, 1, 3}})
            {
                const demo::Rgba8 got = sample(e.dir);
                const auto ca = base_color(e.a), cb = base_color(e.b);
                int d_avg = 0, d_a = 0, d_b = 0;
                for (std::size_t k = 0; k < 3; ++k)
                {
                    d_avg = std::max(d_avg, std::abs(2 * got[k] - (ca[k] + cb[k])));  // in half steps
                    d_a = std::max(d_a, std::abs(got[k] - ca[k]));
                    d_b = std::max(d_b, std::abs(got[k] - cb[k]));
                }
                check.expect(d_avg <= 2 * kTol && d_a > 40 && d_b > 40,
                             std::format("seamless LINEAR lookup on the {}/{} edge returns the average of both faces, not one face clamped or wrapped",
                                         kFaceNames[static_cast<std::size_t>(e.a)], kFaceNames[static_cast<std::size_t>(e.b)]),
                             std::format("got {} average ({:.1f},{:.1f},{:.1f}); distance to face {} {} and to face {} {}", rgba_text(got),
                                         0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1]), 0.5 * (ca[2] + cb[2]), kFaceNames[static_cast<std::size_t>(e.a)], d_a,
                                         kFaceNames[static_cast<std::size_t>(e.b)], d_b));
            }
            // Control: well inside a face the same LINEAR lookup returns the plain face colour.
            for (const auto& [dir, face] : {std::pair{D3{1.0, 0.2, 0.5}, 0}, std::pair{D3{0.5, -0.2, 1.0}, 4}})
            {
                const auto c = base_color(face);
                (void)sample(dir);
                check.expect_pixel(1, 1, demo::Rgba8{static_cast<std::uint8_t>(c[0]), static_cast<std::uint8_t>(c[1]), static_cast<std::uint8_t>(c[2]), 255}, 1,
                                   std::format("control: the same LINEAR lookup inside face {} returns its plain colour", kFaceNames[static_cast<std::size_t>(face)]));
            }
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glEnable(mgl::Capability::DepthTest);
        }

        // ---- D. The scene in the validation configuration: every object against the CPU ray caster -------------------------
        auto run_sweep = [&](const std::string& label, const Reference& ref, const Img& img, const Model& md, std::initializer_list<Obj> objs) {
            const auto st = sweep(ref, img, md, step);
            for (const Obj o : objs)
            {
                const SweepStat& s = st[static_cast<std::size_t>(o)];
                std::string detail = std::format("{} pixels compared ({} with an expected interval at most 2 steps wide), largest distance outside the expected "
                                                 "interval {} (tol {}), widest interval {}",
                                                 s.n, s.tight, s.max_d, kTol, s.widest);
                if (s.max_d > kTol) detail += std::format("; worst pixel({},{}) = {} expected {}", s.wx, s.wy, rgba_text(s.got), s.want.text());
                // Most probes must have a tight expectation: boundary probes accept either neighbour, but they must not
                // be what makes the check pass.
                check.expect(s.n >= 8 && s.bad == 0 && 2 * s.tight >= s.n, std::format("{}: {} matches the CPU reference", label, obj_name(o)), detail);
            }
            return st;
        };
        const std::initializer_list<Obj> all_objects = {Obj::Sky, Obj::Ground, Obj::Chrome, Obj::Glass, Obj::Box};
        const Reference ref_base(f_base, W, H);
        const Model md_base = model_of(f_base);
        const Img img_base = snapshot(f_base);
        run_sweep("scene", ref_base, img_base, md_base, all_objects);
        {
            // Counterfactual models: each wrong formula must be clearly rejected by the rendered image.
            Model no_reflect = md_base, inverse = md_base;
            no_reflect.reflect = false;
            inverse.eta = 1.0 / md_base.eta;
            const Discrimination dr = discriminate(ref_base, img_base, Obj::Chrome, md_base, no_reflect, step);
            check.expect(dr.discriminating >= 8 && dr.rejected == dr.discriminating,
                         "chrome sphere: sampling the sky along I instead of reflect(I, N) is rejected",
                         std::format("{} of {} discriminating pixels (of {}) reject it", dr.rejected, dr.discriminating, dr.candidates));
            const Discrimination di = discriminate(ref_base, img_base, Obj::Glass, md_base, inverse, step);
            check.expect(di.discriminating >= 8 && di.rejected == di.discriminating,
                         "glass sphere: refract() with the inverse ratio eta = 1.5 is rejected",
                         std::format("{} of {} discriminating pixels (of {}) reject it", di.rejected, di.discriminating, di.candidates));
        }
        {
            // The same scene in the face-coded environment: reflected and refracted rays now land on faces of very
            // different hue, so every formula (and the Fresnel blend of two such colours) is far more discriminating
            // than in the soft sky.
            Params p = base;
            p.face_coded = true;
            const Frame f = make_frame(p, asp);
            const Reference ref(f, W, H);
            const Model md = model_of(f);
            const Img img = snapshot(f);
            run_sweep("face-coded environment", ref, img, md, all_objects);
            Model no_reflect = md;
            no_reflect.reflect = false;
            // F only exceeds ~0.1 in the thin ring n.v in [0.3, 0.45] (closer to the rim the polygon error is too large). In the
            // full scene that ring is a pixel or two wide, so the Fresnel term is judged on a close-up of the glass sphere
            // (same environment, same model, sphere radius ~30 % of the shorter side) whose every pixel is scanned. The
            // close-up must itself match the right model before the wrong one can be rejected on it.
            {
                const Frame fc = glass_closeup_frame(p, asp);
                const Reference ref_c(fc, W, H);
                const Model md_c = model_of(fc);
                const Img img_c = snapshot(fc);
                run_sweep("face-coded environment, glass close-up", ref_c, img_c, md_c, {Obj::Glass});
                Model no_fresnel_c = md_c;
                no_fresnel_c.fresnel = false;
                const Discrimination df = discriminate(ref_c, img_c, Obj::Glass, md_c, no_fresnel_c, 1);
                check.expect(df.discriminating >= 8 && df.rejected == df.discriminating && df.accepted_right == df.discriminating,
                             "face-coded environment, glass close-up: glass without the Fresnel-Schlick reflection term (refraction only) is rejected",
                             std::format("{} of {} discriminating pixels (of {}) reject it, {} match the right model", df.rejected, df.discriminating,
                                         df.candidates, df.accepted_right));
            }
            const Discrimination db = discriminate(ref, img, Obj::Box, md, no_reflect, step);
            check.expect(db.discriminating >= 8 && db.rejected == db.discriminating,
                         "face-coded environment: the gold box sampled along I instead of reflect(I, N) is rejected",
                         std::format("{} of {} discriminating pixels (of {}) reject it", db.rejected, db.discriminating, db.candidates));
        }

        // ---- E. Depth trick ------------------------------------------------------------------------------------------------
        {
            // Explicit probes: the sky is drawn LAST, yet every object stays visible in front of it.
            for (const auto& [obj, centre] : {std::pair{Obj::Chrome, d3(kChromePos)}, std::pair{Obj::Glass, d3(kGlassPos)}, std::pair{Obj::Box, d3(kBoxPos)}})
            {
                const Probe p = make_probe(ref_base, W, H, centre, obj);
                if (!p.ok)
                {
                    check.expect(false, std::format("depth trick: probe on the {}", obj_name(obj)), "probe setup failed");
                    continue;
                }
                Reference sky_ref(f_base, W, H);
                sky_ref.sky_everywhere = true;
                const Envelope want = ref_base.expected(p.hit, md_base);
                const Envelope sky = sky_ref.expected(sky_ref.trace(sky_ref.ray(p.x, p.y)), md_base);
                const demo::Rgba8 got = img_base.at(p.x, p.y);
                check.expect(want.distance(got) <= kTol && sky.distance(got) > kTol,
                             std::format("depth trick: the sky drawn last at depth 1.0 with LEQUAL leaves the {} in front of it", obj_name(obj)),
                             std::format("pixel({},{}) = {} object {} sky along that ray {}", p.x, p.y, rgba_text(got), want.text(), sky.text()));
            }
            // Negative control: without .xyww the sky cube (half-size 0.5 around the eye) is drawn at its true depth, passes
            // LEQUAL against everything farther away and covers the whole scene.
            Params p = base;
            p.depth_trick = false;
            const Frame f = make_frame(p, asp);
            Reference sky_ref(f, W, H);
            sky_ref.sky_everywhere = true;
            const Img img = snapshot(f);
            const auto st = sweep(sky_ref, img, model_of(f), step);
            const SweepStat& s = st[static_cast<std::size_t>(Obj::Sky)];
            check.expect(s.n >= (W / step) * (H / step) / 2 && s.bad == 0,
                         "negative control: with gl_Position = clip instead of clip.xyww the sky cube hides the ground and every object",
                         std::format("{} pixels compared with the sky along their ray, {} differ", s.n, s.bad));
            const Probe pc = make_probe(ref_base, W, H, d3(kChromePos), Obj::Chrome);
            if (pc.ok)
            {
                const Envelope obj_env = ref_base.expected(pc.hit, md_base);
                check.expect(obj_env.distance(img.at(pc.x, pc.y)) > kTol,
                             "negative control: ... the chrome sphere probe then fails the scene model (the check would catch a broken trick)",
                             std::format("pixel {} object model {}", rgba_text(img.at(pc.x, pc.y)), obj_env.text()));
            }
        }

        // ---- F. Fog --------------------------------------------------------------------------------------------------------
        {
            const D3 tile_centre{0.5 * dbl(kTile), 0.0, 1.5 * dbl(kTile)};  // centre of a ground tile in front of the objects
            const Probe p = make_probe(ref_base, W, H, tile_centre, Obj::Ground);
            if (!p.ok)
                check.expect(false, "fog probe on the ground", "probe setup failed");
            else
            {
                const double d = p.hit.t;
                const D3 lit = Reference::ground_color(p.hit.pos);
                // Density chosen so that exp(-(density d)^2) = 1/2 exactly at this pixel's distance.
                Params half = base;
                half.fog_density = static_cast<float>(std::sqrt(std::log(2.0)) / d);
                const Frame fh = make_frame(half, asp);
                const Img img_half = snapshot(fh);
                const D3 want_half = mix(d3(kFogColor), lit, 0.5);
                const demo::Rgba8 got = img_half.at(p.x, p.y);
                const demo::Rgba8 want{to8(want_half.x), to8(want_half.y), to8(want_half.z), 255};
                int dmax = 0;
                double measured = 0.0;
                for (std::size_t k = 0; k < 3; ++k) dmax = std::max(dmax, std::abs(got[k] - want[k]));
                {
                    const double fog_r = dbl(kFogColor.x) * 255.0, lit_r = lit.x * 255.0;
                    measured = (got[0] - fog_r) / (lit_r - fog_r);
                }
                check.expect(dmax <= kTol,
                             std::format("fog at a known distance: density sqrt(ln 2) / {:.3f} gives the 50/50 blend of the lit ground and the fog colour", d),
                             std::format("pixel({},{}) = {} expected {} (measured surface weight {:.3f})", p.x, p.y, rgba_text(got), rgba_text(want), measured));
                Params off = half;
                off.fog_on = false;
                const Img img_off = snapshot(make_frame(off, asp));
                const demo::Rgba8 got_off = img_off.at(p.x, p.y);
                const demo::Rgba8 want_off{to8(lit.x), to8(lit.y), to8(lit.z), 255};
                int d_off = 0, d_off_vs_half = 0;
                for (std::size_t k = 0; k < 3; ++k)
                {
                    d_off = std::max(d_off, std::abs(got_off[k] - want_off[k]));
                    d_off_vs_half = std::max(d_off_vs_half, std::abs(got_off[k] - want[k]));
                }
                check.expect(d_off <= kTol, "fog density 0: the same pixel shows the unfogged lit ground",
                             std::format("pixel {} expected {}", rgba_text(got_off), rgba_text(want_off)));
                check.expect(d_off_vs_half >= kDiscriminate, "negative control: with the fog off the 50/50 fog check fails (the fog is really applied)",
                             std::format("{} steps from the fogged expectation", d_off_vs_half));
                // Mid-distance and far ground under the default density: the whole scene sweep with a denser fog.
                Params thick = base;
                thick.fog_density = 0.08f;
                const Frame ft = make_frame(thick, asp);
                run_sweep("dense fog (0.08)", Reference(ft, W, H), snapshot(ft), model_of(ft), {Obj::Ground, Obj::Chrome, Obj::Glass, Obj::Box});
            }
        }

        // ---- G. Negative control: the glass rendered with the inverse ratio ----------------------------------------------------
        {
            Params p = base;
            p.inverse_eta = true;
            const Frame f = make_frame(p, asp);
            const Reference ref(f, W, H);
            const Img img = snapshot(f);
            const Model md_inv = model_of(f);
            run_sweep("eta = 1.5 (inverse ratio, total internal reflection near the rim)", ref, img, md_inv, {Obj::Glass});
            const Discrimination d = discriminate(ref, img, Obj::Glass, md_inv, md_base, step);
            check.expect(d.discriminating >= 8 && d.rejected == d.discriminating,
                         "negative control: on the eta = 1.5 render the correct-eta model fails at every discriminating glass pixel",
                         std::format("{} of {} discriminating pixels fail the eta = 1/1.5 model", d.rejected, d.discriminating));
        }

        // ---- H. Keyboard: camera orbit and fog density -----------------------------------------------------------------------
        {
            auto press = [&](Params start, std::string_view keys) {
                const Params saved = params_;
                params_ = start;
                for (const char k : keys) on_key(env, k);
                const Params out = params_;
                params_ = saved;
                return out;
            };
            const Params moved = press(base, "dddddddddd" "qqq" "ff");
            check.expect(std::abs(moved.yaw - 0.8f) < 1e-5f && std::abs(moved.elevation - (kDefaultElevation + 9.0f)) < 1e-5f &&
                             std::abs(moved.fog_density - kDefaultFog * 1.5625f) < 1e-6f,
                         "keys d, q, f orbit by 0.08 rad, raise the camera by 3 degrees and multiply the fog density by 1.25 per press",
                         std::format("yaw {} elevation {} density {}", moved.yaw, moved.elevation, moved.fog_density));
            const Params toggled = press(base, "ocki ");
            check.expect(!toggled.fog_on && toggled.face_coded && !toggled.depth_trick && toggled.inverse_eta && toggled.paused,
                         "keys o, c, k, i and SPACE toggle the fog, the face-coded cube map, the depth trick, the inverse eta and the pause");
            check.expect(std::abs(press(moved, "gg").fog_density - kDefaultFog) < 1e-6f && press(moved, "r").yaw == 0.0f,
                         "key g divides the fog density by 1.25, key r resets");
            const Frame fm = make_frame(moved, asp);
            run_sweep("orbited camera, denser fog", Reference(fm, W, H), snapshot(fm), model_of(fm), all_objects);
        }

        // The capture shows whatever is on screen when validate() returns: the windowed (trilinear) configuration.
        draw(make_frame(Params{}, asp), W, H);
        check.expect_no_gl_error("no GL error during validation");
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-skybox-and-fog",
        .title = "Skybox, environment mapping and fog",
        .min_es = {3, 0},
        .summary = "A procedurally painted sky cube map drawn as a skybox with the .xyww depth trick, reflected by a chrome "
                   "sphere and a gold box, refracted with Fresnel-Schlick by a glass sphere, over a ground plane that fades "
                   "into exponential-squared fog; pixels are checked against a CPU ray caster with Table 3.21 face selection.",
        .controls = "a/d orbit camera | w/s dolly | q/e raise/lower camera | f/g denser/thinner fog | o fog on/off | "
                    "c face-coded cube map | n NEAREST/trilinear sky | k depth trick on/off | i inverse eta | SPACE pause | r reset | ESC quit",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<SkyboxAndFog>(argc, argv, info);
}
