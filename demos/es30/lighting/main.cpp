// es30-lighting
//
// A small 3D scene lit entirely in shaders: constant ambient light, a directional light, a point
// light with distance attenuation and a spot light with a smooth inner/outer cone falloff, shaded per
// fragment with the Phong or the Blinn-Phong specular model. The scene is a checkered ground plane and
// three pairs of objects. In every pair the left and the right object differ in exactly one respect:
//
//   front row   two identical spheres: Phong (left) vs Blinn-Phong (right) specular highlight
//   middle row  two identical "crystals" (a cube rotated, then stretched along the world x axis, i.e.
//               a NON-UNIFORM scale after the rotation): the left one uses the correct normal matrix
//               transpose(inverse(mat3(M))), the right one the naive mat3(M) and is shaded wrongly
//   back row    two identical coarse spheres: Gouraud (lighting per vertex, left) vs per fragment (right)
//
// A small emissive sphere marks the point light, an emissive cone the spot light.
//
// Validation (--validate): the image is compared with a CPU evaluation of the same equations. Pixels
// are chosen where the surface point and normal are known analytically: ray-plane intersection for the
// ground, ray-sphere for the spheres, ray-box in object space for the crystals and an exact ray-triangle
// test against the coarse mesh for the Gouraud and per-fragment spheres. See README.md.
//
// Spec: ES 3.0 section 2.12.6.1 (UniformMatrix), 3.6.1 (attribute interpolation), 4.1.5 (depth test),
// 2.1.6.2 (float to unorm); GLSL ES 3.00 sections 4.5.1 (highp), 8.3, 8.5 (built-in functions).

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
namespace m = demo::math;

namespace
{
    constexpr float kPi = 3.14159265358979323846f;

    // ---------------------------------------------------------------------------------------------
    // Scene constants. The GPU and the CPU reference read the same numbers, so a typo here cannot make
    // them disagree; the *formulas* are written twice (lighting.frag and reference_shade()).
    // ---------------------------------------------------------------------------------------------
    constexpr int kHiSlices = 96, kHiStacks = 48;  // smooth sphere: 9216 triangles
    constexpr int kLoSlices = 12, kLoStacks = 8;   // coarse sphere for the Gouraud comparison
    constexpr int kGridCells = 24;                 // ground: 24 x 24 cells of 2 x 2 units
    constexpr float kGroundHalf = 24.0f;
    constexpr float kSphereRadius = 0.85f;

    constexpr m::Vec3 kPhongPos{-2.3f, kSphereRadius, 2.7f};
    constexpr m::Vec3 kBlinnPos{2.3f, kSphereRadius, 2.7f};
    constexpr m::Vec3 kCrystalLeftXz{-2.3f, 0.0f, -0.2f};
    constexpr m::Vec3 kCrystalRightXz{2.3f, 0.0f, -0.2f};
    constexpr m::Vec3 kGouraudPos{-2.3f, kSphereRadius, -3.1f};
    constexpr m::Vec3 kFragmentPos{2.3f, kSphereRadius, -3.1f};
    constexpr m::Vec3 kCrystalScale{1.8f, 1.2f, 0.85f};  // applied AFTER the rotation: model = T * S * R
    constexpr float kCrystalAngle0 = 35.0f;              // degrees at t = 0
    constexpr float kPointMarkerRadius = 0.15f;
    constexpr m::Vec3 kSpotMarkerScale{0.30f, 0.50f, 0.30f};  // cone: radius 1, height 1 in object space

    constexpr float kFovyDegrees = 40.0f;
    constexpr float kNear = 0.5f, kFar = 80.0f;
    constexpr m::Vec3 kCameraTarget{0.0f, 0.4f, 0.0f};
    constexpr float kDefaultElevation = 31.0f;  // degrees above the horizon
    constexpr float kDefaultDistance = 11.0f;

    constexpr std::array<float, 4> kClearColor = {0.03f, 0.035f, 0.06f, 1.0f};
    constexpr float kGamma = 2.2f;  // lighting.frag encodes with pow(c, 1 / 2.2)

    inline std::uint8_t to8(double v)
    {
        return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
    }

    struct Material
    {
        m::Vec3 albedo{};
        m::Vec3 albedo2{};
        float checker = 0.0f;
        m::Vec3 specular{};
        float shininess = 1.0f;
        m::Vec3 emissive{};
        int model = 0;  // 0 = Phong, 1 = Blinn-Phong
    };

    constexpr Material kGroundMat{{0.46f, 0.48f, 0.54f}, {0.20f, 0.21f, 0.26f}, 1.5f, {0.12f, 0.12f, 0.12f}, 12.0f, {}, 1};
    constexpr Material kSphereMat{{0.80f, 0.22f, 0.16f}, {}, 0.0f, {0.90f, 0.90f, 0.90f}, 40.0f, {}, 1};  // model set per object
    constexpr Material kCrystalMat{{0.20f, 0.70f, 0.62f}, {}, 0.0f, {0.70f, 0.70f, 0.70f}, 60.0f, {}, 1};
    constexpr Material kCoarseMat{{0.95f, 0.75f, 0.25f}, {}, 0.0f, {0.80f, 0.80f, 0.80f}, 30.0f, {}, 1};
    constexpr Material kPointLampMat{{}, {}, 0.0f, {}, 1.0f, {1.0f, 0.72f, 0.38f}, 0};
    constexpr Material kSpotLampMat{{}, {}, 0.0f, {}, 1.0f, {0.75f, 0.88f, 1.0f}, 0};

    struct Lights
    {
        m::Vec3 ambient{};
        m::Vec3 dir_direction{}, dir_color{};
        m::Vec3 point_position{}, point_color{}, point_atten{};
        m::Vec3 spot_position{}, spot_direction{}, spot_color{}, spot_atten{};
        float spot_cos_inner = 1.0f, spot_cos_outer = 0.0f;
    };

    constexpr m::Vec3 kAmbient{0.10f, 0.11f, 0.15f};
    constexpr m::Vec3 kDirColor{0.42f, 0.44f, 0.52f};
    constexpr m::Vec3 kPointColor{2.6f, 1.5f, 0.65f};
    constexpr m::Vec3 kPointAtten{1.0f, 0.10f, 0.07f};
    constexpr m::Vec3 kSpotColor{1.3f, 1.55f, 1.8f};
    constexpr m::Vec3 kSpotAtten{1.0f, 0.04f, 0.012f};
    constexpr float kSpotInnerDeg = 11.0f, kSpotOuterDeg = 22.0f;

    // Everything the keyboard and the clock control.
    struct Params
    {
        int ticks = 0;  // animation time in 1/60 s steps (deterministic: frame / 60)
        float yaw = 0.0f;
        float elevation = kDefaultElevation;
        float distance = kDefaultDistance;
        bool dir_on = true, point_on = true, spot_on = true;
        bool blinn = true;         // specular model of the objects that follow the global switch
        bool naive_fixed = false;  // the right crystal also uses the correct normal matrix (key n)
        bool gouraud = true;       // the left coarse sphere is Gouraud-shaded (key g)
        bool paused = false;
    };

    struct Frame
    {
        float aspect = 1.0f;
        m::Vec3 eye{}, center{};
        m::Mat4 view{}, proj{};
        Lights lights{};
        float crystal_angle = 0.0f;  // radians
        Params params{};
    };

    [[nodiscard]] constexpr m::Vec3 zero3() { return {0.0f, 0.0f, 0.0f}; }

    Frame make_frame(const Params& p, float aspect)
    {
        Frame f;
        f.params = p;
        f.aspect = aspect;
        const float t = static_cast<float>(p.ticks) / 60.0f;

        // Camera: orbits the scene centre. A gentle sway is added by the clock; the keys add yaw,
        // elevation and distance.
        const float yaw = p.yaw + m::radians(9.0f) * std::sin(0.5f * t);
        const float el = m::radians(p.elevation);
        f.center = kCameraTarget;
        f.eye = f.center + m::Vec3{std::sin(yaw) * std::cos(el), std::sin(el), std::cos(yaw) * std::cos(el)} * p.distance;
        f.view = m::look_at(f.eye, f.center, {0.0f, 1.0f, 0.0f});
        f.proj = m::perspective(m::radians(kFovyDegrees), aspect, kNear, kFar);

        Lights& l = f.lights;
        l.ambient = kAmbient;
        l.dir_direction = m::normalize({-0.5f, -0.8f, -0.6f});
        l.dir_color = p.dir_on ? kDirColor : zero3();
        l.point_position = {0.7f + 0.5f * std::sin(0.6f * t + 0.6f), 1.3f + 0.15f * std::sin(1.1f * t), 2.0f + 2.0f * std::sin(0.5f * t)};
        l.point_color = p.point_on ? kPointColor : zero3();
        l.point_atten = kPointAtten;
        l.spot_position = {-0.8f, 3.4f, -1.8f};
        const m::Vec3 target{-1.5f + 1.0f * std::cos(0.7f * t), 0.0f, 1.2f + 1.0f * std::sin(0.7f * t)};
        l.spot_direction = m::normalize(target - l.spot_position);
        l.spot_color = p.spot_on ? kSpotColor : zero3();
        l.spot_atten = kSpotAtten;
        l.spot_cos_inner = std::cos(m::radians(kSpotInnerDeg));
        l.spot_cos_outer = std::cos(m::radians(kSpotOuterDeg));

        f.crystal_angle = m::radians(kCrystalAngle0) + m::radians(25.0f) * t;
        return f;
    }

    // ---------------------------------------------------------------------------------------------
    // Objects
    // ---------------------------------------------------------------------------------------------
    enum class Obj
    {
        Sky,
        Ground,
        SpherePhong,
        SphereBlinn,
        CrystalCorrect,
        CrystalNaive,
        CoarseGouraud,
        CoarseFragment,
        PointLamp,
        SpotLamp
    };

    enum class MeshId
    {
        Ground,
        Sphere,
        Coarse,
        Cube,
        Cone,
        Count
    };

    struct Item
    {
        Obj id;
        MeshId mesh;
        m::Mat4 model;
        Material mat;
        bool gouraud = false;
        bool naive_normals = false;
    };

    [[nodiscard]] m::Mat4 crystal_model(const m::Vec3& xz, float angle)
    {
        return m::translate({xz.x, 0.5f * kCrystalScale.y, xz.z}) * m::scale(kCrystalScale) * m::rotate(angle, {0.0f, 1.0f, 0.0f});
    }

    [[nodiscard]] m::Mat4 sphere_model(const m::Vec3& c, float r) { return m::translate(c) * m::scale({r, r, r}); }

    // The cone's axis (apex -> base) must point along the spot direction: the apex (object +y) sits
    // opposite to it, so rotate +y onto -direction.
    [[nodiscard]] m::Mat4 spot_lamp_model(const Lights& l)
    {
        const m::Vec3 up{0.0f, 1.0f, 0.0f};
        const m::Vec3 want = l.spot_direction * -1.0f;
        const m::Vec3 axis = m::cross(up, want);
        const float c = std::clamp(m::dot(up, want), -1.0f, 1.0f);
        m::Mat4 rot = m::Mat4::identity();
        if (m::dot(axis, axis) > 1e-12f) rot = m::rotate(std::acos(c), axis);
        return m::translate(l.spot_position) * rot * m::scale(kSpotMarkerScale);
    }

    std::vector<Item> make_items(const Frame& f)
    {
        const Params& p = f.params;
        const int global_model = p.blinn ? 1 : 0;
        auto with_model = [](Material mat, int model) {
            mat.model = model;
            return mat;
        };
        std::vector<Item> items;
        items.push_back({Obj::Ground, MeshId::Ground, m::Mat4::identity(), with_model(kGroundMat, global_model)});
        items.push_back({Obj::SpherePhong, MeshId::Sphere, sphere_model(kPhongPos, kSphereRadius), with_model(kSphereMat, 0)});
        items.push_back({Obj::SphereBlinn, MeshId::Sphere, sphere_model(kBlinnPos, kSphereRadius), with_model(kSphereMat, 1)});
        items.push_back({Obj::CrystalCorrect, MeshId::Cube, crystal_model(kCrystalLeftXz, f.crystal_angle), with_model(kCrystalMat, global_model)});
        items.push_back({Obj::CrystalNaive, MeshId::Cube, crystal_model(kCrystalRightXz, f.crystal_angle), with_model(kCrystalMat, global_model),
                         false, !p.naive_fixed});
        items.push_back({Obj::CoarseGouraud, MeshId::Coarse, sphere_model(kGouraudPos, kSphereRadius), with_model(kCoarseMat, global_model), p.gouraud});
        items.push_back({Obj::CoarseFragment, MeshId::Coarse, sphere_model(kFragmentPos, kSphereRadius), with_model(kCoarseMat, global_model)});
        items.push_back({Obj::PointLamp, MeshId::Sphere, sphere_model(f.lights.point_position, kPointMarkerRadius), kPointLampMat});
        items.push_back({Obj::SpotLamp, MeshId::Cone, spot_lamp_model(f.lights), kSpotLampMat});
        return items;
    }

    // ---------------------------------------------------------------------------------------------
    // Procedural meshes. Vertex = position + normal (object space); indices are uint16.
    // ---------------------------------------------------------------------------------------------
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
        MeshData md;
        for (int i = 0; i <= stacks; ++i)
        {
            const float phi = kPi * fl(i) / fl(stacks);
            for (int j = 0; j <= slices; ++j)
            {
                const float theta = 2.0f * kPi * fl(j) / fl(slices);
                const float x = std::sin(phi) * std::cos(theta), y = std::cos(phi), z = std::sin(phi) * std::sin(theta);
                md.vertices.push_back({x, y, z, x, y, z});
            }
        }
        for (int i = 0; i < stacks; ++i)
            for (int j = 0; j < slices; ++j)
            {
                const int a = i * (slices + 1) + j, b = a + slices + 1;
                for (const int idx : {a, a + 1, b, a + 1, b + 1, b}) md.indices.push_back(u16(idx));  // counter-clockwise seen from outside
            }
        return md;
    }

    // Unit cube [-0.5, 0.5]^3 with flat normals (4 vertices per face).
    MeshData make_cube()
    {
        MeshData md;
        struct Face
        {
            m::Vec3 n, u, v;  // u x v = n, so (-u-v, +u-v, +u+v, -u+v) is counter-clockwise seen from outside
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

    // Flat grid in the xz plane (y = 0), normal +y, counter-clockwise seen from above.
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

    // Cone: apex (0,1,0), base circle of radius 1 at y = 0, closed by a cap.
    MeshData make_cone(int slices)
    {
        MeshData md;
        const float inv_sqrt2 = 0.70710678f;
        for (int k = 0; k < slices; ++k)  // side: apex vertex + ring vertex per slice
        {
            const float a0 = 2.0f * kPi * fl(k) / fl(slices), a1 = 2.0f * kPi * fl(k + 1) / fl(slices);
            const float am = 0.5f * (a0 + a1);
            const int b = static_cast<int>(md.vertices.size());
            md.vertices.push_back({std::cos(a0), 0.0f, std::sin(a0), std::cos(a0) * inv_sqrt2, inv_sqrt2, std::sin(a0) * inv_sqrt2});
            md.vertices.push_back({0.0f, 1.0f, 0.0f, std::cos(am) * inv_sqrt2, inv_sqrt2, std::sin(am) * inv_sqrt2});
            md.vertices.push_back({std::cos(a1), 0.0f, std::sin(a1), std::cos(a1) * inv_sqrt2, inv_sqrt2, std::sin(a1) * inv_sqrt2});
            for (const int idx : {0, 1, 2}) md.indices.push_back(u16(b + idx));
        }
        const int center = static_cast<int>(md.vertices.size());
        md.vertices.push_back({0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f});
        for (int k = 0; k <= slices; ++k)
        {
            const float a = 2.0f * kPi * fl(k) / fl(slices);
            md.vertices.push_back({std::cos(a), 0.0f, std::sin(a), 0.0f, -1.0f, 0.0f});
        }
        for (int k = 0; k < slices; ++k)
            for (const int idx : {center, center + 1 + k, center + 2 + k}) md.indices.push_back(u16(idx));
        return md;
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference (double precision), used only by validate(). It is written independently of the
    // GL side: its own vector type, its own rotation / inverse formulas, its own camera model (no Mat4
    // is used for projection or ray generation), the lighting equation re-implemented from the README,
    // and exact intersection tests. It reads the same scene constants as the renderer.
    // ---------------------------------------------------------------------------------------------
    struct D3
    {
        double x = 0.0, y = 0.0, z = 0.0;
    };
    D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    D3 operator*(D3 a, D3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }  // component-wise
    double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    double length(D3 a) { return std::sqrt(dot(a, a)); }
    D3 normalize(D3 a) { return a * (1.0 / length(a)); }
    D3 d3(const m::Vec3& v) { return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)}; }
    double dbl(float v) { return static_cast<double>(v); }

    double smoothstep(double e0, double e1, double x)
    {
        const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }

    // Albedo with the checker pattern (parity of floor(x / s) + floor(z / s)).
    D3 ref_albedo(const Material& mat, D3 p)
    {
        if (mat.checker <= 0.0f) return d3(mat.albedo);
        const long cx = static_cast<long>(std::floor(p.x / dbl(mat.checker)));
        const long cz = static_cast<long>(std::floor(p.z / dbl(mat.checker)));
        return d3(((cx + cz) & 1) != 0 ? mat.albedo2 : mat.albedo);
    }

    // One light arriving from the unit direction l (surface -> light): diffuse + specular.
    D3 ref_reflect_light(const Material& mat, D3 n, D3 v, D3 l, D3 radiance, D3 albedo)
    {
        const double n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) return {};
        double s = 0.0;
        if (mat.model == 0)  // Phong: R = reflect(-L, N) = I - 2 (N.I) N with I = -L
        {
            const D3 incident = l * -1.0;
            const D3 r = incident - n * (2.0 * dot(n, incident));
            s = std::max(dot(r, v), 0.0);
        }
        else  // Blinn-Phong: H = normalize(L + V)
        {
            s = std::max(dot(n, normalize(l + v)), 0.0);
        }
        return radiance * (albedo * n_dot_l + d3(mat.specular) * std::pow(s, dbl(mat.shininess)));
    }

    double ref_spot_cone(const Lights& l, D3 p)
    {
        const D3 to_light = normalize(d3(l.spot_position) - p);
        return smoothstep(dbl(l.spot_cos_outer), dbl(l.spot_cos_inner), dot(to_light * -1.0, d3(l.spot_direction)));
    }

    // Lighting equation, linear colour (before gamma). `hard_cone` replaces the smooth falloff by a
    // step at the outer angle (only used by a negative control).
    D3 ref_shade(const Lights& l, D3 eye, const Material& mat, D3 p, D3 n, bool hard_cone = false)
    {
        const D3 v = normalize(eye - p);
        const D3 albedo = ref_albedo(mat, p);
        D3 c = d3(mat.emissive) + d3(l.ambient) * albedo;
        c = c + ref_reflect_light(mat, n, v, normalize(d3(l.dir_direction) * -1.0), d3(l.dir_color), albedo);
        {
            const D3 to_light = d3(l.point_position) - p;
            const double d = length(to_light);
            const double atten = 1.0 / (dbl(l.point_atten.x) + dbl(l.point_atten.y) * d + dbl(l.point_atten.z) * d * d);
            c = c + ref_reflect_light(mat, n, v, to_light * (1.0 / d), d3(l.point_color) * atten, albedo);
        }
        {
            const D3 to_light = d3(l.spot_position) - p;
            const double d = length(to_light);
            const double atten = 1.0 / (dbl(l.spot_atten.x) + dbl(l.spot_atten.y) * d + dbl(l.spot_atten.z) * d * d);
            double cone = ref_spot_cone(l, p);
            if (hard_cone) cone = cone > 0.0 ? 1.0 : 0.0;
            c = c + ref_reflect_light(mat, n, v, to_light * (1.0 / d), d3(l.spot_color) * (atten * cone), albedo);
        }
        return c;
    }

    demo::Rgba8 encode_color(D3 linear)  // clamp, gamma 1/2.2, round to 8 bits (ES 3.0 section 2.1.6.2)
    {
        const double e = 1.0 / static_cast<double>(kGamma);
        auto enc = [&](double v) { return to8(std::pow(std::clamp(v, 0.0, 1.0), e)); };
        return {enc(linear.x), enc(linear.y), enc(linear.z), 255};
    }

    // Face f of the unit cube: axis f / 2, sign + for odd f. Object-space normal, then the world normal
    // for model = T * S * R: correct = normalize(S^-1 R n)  (the inverse transpose of S R), naive = S R n.
    D3 crystal_face_normal(double angle, int face, bool naive)
    {
        const double c = std::cos(angle), s = std::sin(angle);
        const double sign = (face % 2) != 0 ? 1.0 : -1.0;
        D3 n{};
        (face / 2 == 0 ? n.x : face / 2 == 1 ? n.y : n.z) = sign;
        const D3 rn{c * n.x + s * n.z, n.y, -s * n.x + c * n.z};  // R n, R = rotation about y
        const D3 sc = d3(kCrystalScale);
        return normalize(naive ? D3{rn.x * sc.x, rn.y * sc.y, rn.z * sc.z} : D3{rn.x / sc.x, rn.y / sc.y, rn.z / sc.z});
    }

    D3 crystal_centre(const m::Vec3& xz) { return {dbl(xz.x), 0.5 * dbl(kCrystalScale.y), dbl(xz.z)}; }

    // World point of an object-space point q of the crystal: centre + S R q.
    D3 crystal_point(const m::Vec3& xz, double angle, D3 q)
    {
        const double c = std::cos(angle), s = std::sin(angle);
        const D3 rq{c * q.x + s * q.z, q.y, -s * q.x + c * q.z};
        const D3 sc = d3(kCrystalScale);
        return crystal_centre(xz) + D3{rq.x * sc.x, rq.y * sc.y, rq.z * sc.z};
    }

    struct Ray
    {
        D3 o, d;
    };

    struct Hit
    {
        Obj obj = Obj::Sky;
        double t = 1e30;
        D3 pos{}, normal{};
        long tag = 0;            // checker tile or crystal face: colour is discontinuous where it changes
        double n_dot_v = 1.0;    // cosine of the angle between normal and view direction
        D3 color{};              // linear colour as rendered
        D3 color_vertex{};       // coarse spheres only: colour interpolated from per-vertex lighting (Gouraud)
        D3 color_fragment{};     // coarse spheres only: colour from the interpolated position / normal
    };

    class Reference
    {
    public:
        Reference(const Frame& f, int width, int height)
            : f_(f), w_(width), h_(height), coarse_(make_sphere(kLoSlices, kLoStacks))
        {
            for (const Item& it : make_items(f)) mats_[static_cast<std::size_t>(it.id)] = it.mat;
            eye_ = d3(f.eye);
            fwd_ = normalize(d3(f.center) - eye_);
            right_ = normalize(cross(fwd_, D3{0.0, 1.0, 0.0}));
            up_ = cross(right_, fwd_);
            tan_half_ = std::tan(0.5 * dbl(m::radians(kFovyDegrees)));
            aspect_ = static_cast<double>(width) / static_cast<double>(height);
        }

        [[nodiscard]] const Frame& frame() const { return f_; }
        [[nodiscard]] const Material& material(Obj o) const { return mats_[static_cast<std::size_t>(o)]; }
        [[nodiscard]] D3 eye() const { return eye_; }

        // Ray through the centre of window pixel (px, py), y counted from the bottom like glReadPixels.
        [[nodiscard]] Ray ray(int px, int py) const
        {
            const double nx = (static_cast<double>(px) + 0.5) / static_cast<double>(w_) * 2.0 - 1.0;
            const double ny = (static_cast<double>(py) + 0.5) / static_cast<double>(h_) * 2.0 - 1.0;
            return {eye_, normalize(fwd_ + right_ * (nx * tan_half_ * aspect_) + up_ * (ny * tan_half_))};
        }

        // Perspective projection of a world point to pixel-centre coordinates (pixel i covers i - 0.5 .. i + 0.5).
        [[nodiscard]] bool project(D3 p, double& px, double& py) const
        {
            const D3 v = p - eye_;
            const double z = dot(v, fwd_);
            if (z <= 0.0) return false;
            px = (dot(v, right_) / (z * tan_half_ * aspect_) + 1.0) * 0.5 * static_cast<double>(w_) - 0.5;
            py = (dot(v, up_) / (z * tan_half_) + 1.0) * 0.5 * static_cast<double>(h_) - 0.5;
            return true;
        }

        [[nodiscard]] Hit trace(const Ray& r, bool with_color = true) const
        {
            Hit best;
            auto finish = [&](Hit& h) {
                h.n_dot_v = dot(h.normal, normalize(eye_ - h.pos));
                if (h.t < best.t) best = h;
            };

            if (r.d.y < -1e-12)  // ground plane y = 0
            {
                const double t = -r.o.y / r.d.y;
                const D3 p = r.o + r.d * t;
                if (t > 0.0 && std::abs(p.x) <= dbl(kGroundHalf) && std::abs(p.z) <= dbl(kGroundHalf))
                {
                    Hit h;
                    h.obj = Obj::Ground;
                    h.t = t;
                    h.pos = p;
                    h.normal = {0.0, 1.0, 0.0};
                    const double ck = dbl(kGroundMat.checker);
                    h.tag = static_cast<long>(std::floor(p.x / ck)) * 100003L + static_cast<long>(std::floor(p.z / ck));
                    if (with_color) h.color = shade_point(Obj::Ground, h.pos, h.normal);
                    finish(h);
                }
            }

            auto sphere = [&](Obj id, D3 c, double rad, bool lit) {
                const D3 oc = r.o - c;
                const double b = dot(oc, r.d), cc = dot(oc, oc) - rad * rad, disc = b * b - cc;
                if (disc < 0.0) return;
                const double t = -b - std::sqrt(disc);
                if (t <= 1e-9) return;
                Hit h;
                h.obj = id;
                h.t = t;
                h.pos = r.o + r.d * t;
                h.normal = (h.pos - c) * (1.0 / rad);
                if (with_color && lit) h.color = shade_point(id, h.pos, h.normal);
                finish(h);
            };
            sphere(Obj::SpherePhong, d3(kPhongPos), dbl(kSphereRadius), true);
            sphere(Obj::SphereBlinn, d3(kBlinnPos), dbl(kSphereRadius), true);
            sphere(Obj::PointLamp, d3(f_.lights.point_position), dbl(kPointMarkerRadius), true);
            // The spot lamp (a cone) is only represented by a generous bounding sphere: pixels there are never compared.
            sphere(Obj::SpotLamp, d3(f_.lights.spot_position) - d3(f_.lights.spot_direction) * 0.25, 0.5, false);

            crystal(Obj::CrystalCorrect, kCrystalLeftXz, r, false, with_color, finish);
            crystal(Obj::CrystalNaive, kCrystalRightXz, r, !f_.params.naive_fixed, with_color, finish);
            coarse(Obj::CoarseGouraud, kGouraudPos, r, with_color, finish);
            coarse(Obj::CoarseFragment, kFragmentPos, r, with_color, finish);
            return best;
        }

        // Colour the pixel must have (clear colour for rays that hit nothing).
        [[nodiscard]] demo::Rgba8 expected(const Hit& h) const
        {
            if (h.obj == Obj::Sky) return {to8(dbl(kClearColor[0])), to8(dbl(kClearColor[1])), to8(dbl(kClearColor[2])), 255};
            return encode_color(h.color);
        }

        [[nodiscard]] D3 shade_point(Obj id, D3 p, D3 n, bool hard_cone = false) const
        {
            return ref_shade(f_.lights, eye_, material(id), p, n, hard_cone);
        }
        [[nodiscard]] D3 shade_point_as(const Material& mat, D3 p, D3 n) const { return ref_shade(f_.lights, eye_, mat, p, n); }

        // A pixel is "stable" when its 8 neighbours at +-2 px hit the same object (and the same checker tile or
        // crystal face): the probe is then away from silhouettes and from colour discontinuities, where the exact
        // rasterization rule (which side of an edge a pixel centre falls on) would matter.
        [[nodiscard]] bool stable(int px, int py, const Hit& h) const
        {
            for (int dy = -2; dy <= 2; dy += 2)
                for (int dx = -2; dx <= 2; dx += 2)
                {
                    if (dx == 0 && dy == 0) continue;
                    const Hit n = trace(ray(px + dx, py + dy), false);
                    if (n.obj != h.obj) return false;
                    if ((h.obj == Obj::Ground || h.obj == Obj::CrystalCorrect || h.obj == Obj::CrystalNaive) && n.tag != h.tag) return false;
                }
            return true;
        }

    private:
        template <typename Finish>
        void crystal(Obj id, const m::Vec3& xz, const Ray& r, bool naive, bool with_color, Finish& finish) const
        {
            const double c = std::cos(dbl(f_.crystal_angle)), s = std::sin(dbl(f_.crystal_angle));
            const D3 sc = d3(kCrystalScale);
            // world -> object space: q = R^T S^-1 (p - centre); the ray parameter t is unchanged by the affine map
            auto to_model = [&](D3 v) {
                const D3 u{v.x / sc.x, v.y / sc.y, v.z / sc.z};
                return D3{c * u.x - s * u.z, u.y, s * u.x + c * u.z};
            };
            const D3 o = to_model(r.o - crystal_centre(xz));
            const D3 d = to_model(r.d);
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
            const bool positive = da[axis] < 0.0;  // the ray enters through the face it moves against
            Hit h;
            h.obj = id;
            h.t = t_in;
            h.pos = r.o + r.d * t_in;
            h.tag = axis * 2 + (positive ? 1 : 0);
            h.normal = crystal_face_normal(dbl(f_.crystal_angle), static_cast<int>(h.tag), naive);
            if (with_color) h.color = shade_point(id, h.pos, h.normal);
            finish(h);
        }

        // Exact ray / triangle test (Moller-Trumbore, front faces only) against the coarse sphere mesh.
        template <typename Finish>
        void coarse(Obj id, const m::Vec3& centre, const Ray& r, bool with_color, Finish& finish) const
        {
            const D3 c = d3(centre);
            const double rad = dbl(kSphereRadius);
            {
                // The mesh is inscribed in the sphere: a ray that misses the sphere cannot hit any of the 192 triangles.
                const D3 oc = r.o - c;
                const double bb = dot(oc, r.d);
                if (bb * bb - (dot(oc, oc) - rad * rad * 1.0001) < 0.0) return;
            }
            auto vertex = [&](std::uint16_t i) { return c + d3({coarse_.vertices[i].px, coarse_.vertices[i].py, coarse_.vertices[i].pz}) * rad; };
            auto vnormal = [&](std::uint16_t i) { return d3({coarse_.vertices[i].nx, coarse_.vertices[i].ny, coarse_.vertices[i].nz}); };
            Hit best;
            std::array<std::uint16_t, 3> best_idx{};
            std::array<double, 3> best_w{};
            for (std::size_t k = 0; k + 2 < coarse_.indices.size(); k += 3)
            {
                const std::array<std::uint16_t, 3> idx = {coarse_.indices[k], coarse_.indices[k + 1], coarse_.indices[k + 2]};
                const D3 p0 = vertex(idx[0]), e1 = vertex(idx[1]) - p0, e2 = vertex(idx[2]) - p0;
                const D3 pv = cross(r.d, e2);
                const double det = dot(e1, pv);
                if (det <= 1e-9) continue;  // back-facing or degenerate
                const D3 tv = r.o - p0;
                const double u = dot(tv, pv) / det;
                if (u < 0.0 || u > 1.0) continue;
                const D3 qv = cross(tv, e1);
                const double v = dot(r.d, qv) / det;
                if (v < 0.0 || u + v > 1.0) continue;
                const double t = dot(e2, qv) / det;
                if (t <= 1e-9 || t >= best.t) continue;
                best.t = t;
                best_idx = idx;
                best_w = {1.0 - u - v, u, v};
            }
            if (best.t >= 1e29) return;
            Hit h;
            h.obj = id;
            h.t = best.t;
            D3 pos{}, nrm{}, vcol{};
            for (std::size_t k = 0; k < 3; ++k)
            {
                pos = pos + vertex(best_idx[k]) * best_w[k];
                nrm = nrm + vnormal(best_idx[k]) * best_w[k];
                // Gouraud: the vertex shader lights the vertex with its own (normalized) normal
                if (with_color) vcol = vcol + ref_shade(f_.lights, eye_, material(id), vertex(best_idx[k]), normalize(vnormal(best_idx[k]))) * best_w[k];
            }
            h.pos = pos;
            h.normal = normalize(nrm);  // the fragment shader renormalizes the interpolated normal
            if (with_color)
            {
                h.color_vertex = vcol;
                h.color_fragment = shade_point(id, h.pos, h.normal);
                h.color = (id == Obj::CoarseGouraud && f_.params.gouraud) ? h.color_vertex : h.color_fragment;
            }
            finish(h);
        }

        Frame f_;
        int w_, h_;
        MeshData coarse_;
        std::array<Material, 10> mats_{};
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

    struct Uniforms
    {
        mgl::UniformLocation model, view, proj, normal_matrix, eye, ambient;
        mgl::UniformLocation dir_direction, dir_color;
        mgl::UniformLocation point_position, point_color, point_atten;
        mgl::UniformLocation spot_position, spot_direction, spot_color, spot_atten, spot_cos_inner, spot_cos_outer;
        mgl::UniformLocation mat_albedo, mat_albedo2, mat_checker, mat_specular, mat_shininess, mat_emissive, mat_model;
    };

    struct Program
    {
        mgl::ProgramId id{};
        Uniforms u{};
    };

    // Locations of members of struct uniforms are queried by their full name ("u_material.albedo").
    // A uniform the compiler removes (e.g. unused) is inactive and gives location -1; glUniform* silently
    // ignores -1 (ES 3.0 section 2.12.6.1), so one upload function serves both programs even though the
    // Gouraud program does not use the checker members.
    Uniforms query_uniforms(mgl::ProgramId p)
    {
        auto loc = [&](const char* name) { return mgl::glGetUniformLocation(p, name); };
        Uniforms u;
        u.model = loc("u_model");
        u.view = loc("u_view");
        u.proj = loc("u_proj");
        u.normal_matrix = loc("u_normal_matrix");
        u.eye = loc("u_eye");
        u.ambient = loc("u_ambient");
        u.dir_direction = loc("u_dir.direction");
        u.dir_color = loc("u_dir.color");
        u.point_position = loc("u_point.position");
        u.point_color = loc("u_point.color");
        u.point_atten = loc("u_point.attenuation");
        u.spot_position = loc("u_spot.position");
        u.spot_direction = loc("u_spot.direction");
        u.spot_color = loc("u_spot.color");
        u.spot_atten = loc("u_spot.attenuation");
        u.spot_cos_inner = loc("u_spot.cos_inner");
        u.spot_cos_outer = loc("u_spot.cos_outer");
        u.mat_albedo = loc("u_material.albedo");
        u.mat_albedo2 = loc("u_material.albedo2");
        u.mat_checker = loc("u_material.checker");
        u.mat_specular = loc("u_material.specular");
        u.mat_shininess = loc("u_material.shininess");
        u.mat_emissive = loc("u_material.emissive");
        u.mat_model = loc("u_material.model");
        return u;
    }

    void uniform3(mgl::UniformLocation l, const m::Vec3& v) { mgl::glUniform3f(l, v.x, v.y, v.z); }

    [[nodiscard]] std::array<float, 9> naive_normal_matrix(const m::Mat4& model)
    {
        std::array<float, 9> n{};  // upper-left 3x3 of the model matrix, column-major
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) n[static_cast<std::size_t>(c * 3 + r)] = model.at(r, c);
        return n;
    }

    // Uniforms that are the same for every object of a frame.
    void upload_frame_uniforms(const Program& p, const Frame& f)
    {
        mgl::glUseProgram(p.id);
        mgl::glUniformMatrix4fv(p.u.view, 1, GL_FALSE, f.view.data());
        mgl::glUniformMatrix4fv(p.u.proj, 1, GL_FALSE, f.proj.data());
        uniform3(p.u.eye, f.eye);
        const Lights& l = f.lights;
        uniform3(p.u.ambient, l.ambient);
        uniform3(p.u.dir_direction, l.dir_direction);
        uniform3(p.u.dir_color, l.dir_color);
        uniform3(p.u.point_position, l.point_position);
        uniform3(p.u.point_color, l.point_color);
        uniform3(p.u.point_atten, l.point_atten);
        uniform3(p.u.spot_position, l.spot_position);
        uniform3(p.u.spot_direction, l.spot_direction);
        uniform3(p.u.spot_color, l.spot_color);
        uniform3(p.u.spot_atten, l.spot_atten);
        mgl::glUniform1f(p.u.spot_cos_inner, l.spot_cos_inner);
        mgl::glUniform1f(p.u.spot_cos_outer, l.spot_cos_outer);
    }

    // Per-object uniforms: model matrix, its normal matrix and the material.
    void upload_object_uniforms(const Program& p, const Item& it)
    {
        mgl::glUniformMatrix4fv(p.u.model, 1, GL_FALSE, it.model.data());
        const std::array<float, 9> nm = it.naive_normals ? naive_normal_matrix(it.model) : m::normal_matrix(it.model);
        mgl::glUniformMatrix3fv(p.u.normal_matrix, 1, GL_FALSE, nm.data());
        uniform3(p.u.mat_albedo, it.mat.albedo);
        uniform3(p.u.mat_albedo2, it.mat.albedo2);
        mgl::glUniform1f(p.u.mat_checker, it.mat.checker);
        uniform3(p.u.mat_specular, it.mat.specular);
        mgl::glUniform1f(p.u.mat_shininess, it.mat.shininess);
        uniform3(p.u.mat_emissive, it.mat.emissive);
        mgl::glUniform1i(p.u.mat_model, it.mat.model);
    }

    // Fixed-function state of a frame. The defaults are what the demo uses; validate() changes them to
    // prove that depth testing and face culling really decide what is visible.
    struct DrawState
    {
        bool depth_test = true;
        bool reverse_order = false;  // draw the items in the opposite order (the ground last)
        mgl::CullFace cull = mgl::CullFace::Back;
        mgl::FrontFace front = mgl::FrontFace::CCW;
    };

    class Lighting final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            phong_.id = demo::build_program(env, "shaders/lighting.vert", "shaders/lighting.frag");
            gouraud_.id = demo::build_program(env, "shaders/gouraud.vert", "shaders/gouraud.frag");
            phong_.u = query_uniforms(phong_.id);
            gouraud_.u = query_uniforms(gouraud_.id);

            mesh_data_[static_cast<int>(MeshId::Ground)] = make_grid(kGroundHalf, kGridCells);
            mesh_data_[static_cast<int>(MeshId::Sphere)] = make_sphere(kHiSlices, kHiStacks);
            mesh_data_[static_cast<int>(MeshId::Coarse)] = make_sphere(kLoSlices, kLoStacks);
            mesh_data_[static_cast<int>(MeshId::Cube)] = make_cube();
            mesh_data_[static_cast<int>(MeshId::Cone)] = make_cone(24);
            for (int i = 0; i < static_cast<int>(MeshId::Count); ++i) upload(meshes_[static_cast<std::size_t>(i)], mesh_data_[static_cast<std::size_t>(i)]);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            draw(make_frame(params_, aspect(env)), env.width, env.height);  // frame k renders at t = k / 60 s ...
            if (!params_.paused) ++params_.ticks;                           // ... unless the animation is paused
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
            case ' ': params_.paused = !params_.paused; break;
            case '1': params_.dir_on = !params_.dir_on; break;
            case '2': params_.point_on = !params_.point_on; break;
            case '3': params_.spot_on = !params_.spot_on; break;
            case 'b': params_.blinn = !params_.blinn; break;
            case 'n': params_.naive_fixed = !params_.naive_fixed; break;
            case 'g': params_.gouraud = !params_.gouraud; break;
            case 'a': params_.yaw -= 0.08f; break;
            case 'd': params_.yaw += 0.08f; break;
            case 'w': params_.distance = std::max(6.0f, params_.distance - 0.5f); break;
            case 's': params_.distance = std::min(30.0f, params_.distance + 0.5f); break;
            case 'q': params_.elevation = std::min(80.0f, params_.elevation + 3.0f); break;
            case 'e': params_.elevation = std::max(8.0f, params_.elevation - 3.0f); break;
            case 'r': params_ = Params{}; break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GpuMesh& mesh : meshes_)
            {
                mgl::glDeleteVertexArrays(1, &mesh.vao);
                mgl::glDeleteBuffers(1, &mesh.vbo);
                mgl::glDeleteBuffers(1, &mesh.ebo);
            }
            mgl::glDeleteProgram(phong_.id);
            mgl::glDeleteProgram(gouraud_.id);
        }

    private:
        static float aspect(const demo::Env& env) { return static_cast<float>(env.width) / static_cast<float>(env.height); }

        void upload(GpuMesh& mesh, const MeshData& md)
        {
            mgl::glGenVertexArrays(1, &mesh.vao);
            mgl::glBindVertexArray(mesh.vao);

            mgl::glGenBuffers(1, &mesh.vbo);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mesh.vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(md.vertices), mgl::BufferUsage::StaticDraw);
            const mgl::AttribLocation a_position{0u}, a_normal{1u};  // layout(location = 0 / 1) in both vertex shaders
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, px)));
            mgl::glEnableVertexAttribArray(a_normal);
            mgl::glVertexAttribPointer(a_normal, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, nx)));

            // The ELEMENT_ARRAY_BUFFER binding IS vertex-array state: bind it while the VAO is bound.
            mgl::glGenBuffers(1, &mesh.ebo);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mesh.ebo);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(md.indices), mgl::BufferUsage::StaticDraw);
            mesh.index_count = static_cast<int>(md.indices.size());

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
        }

        void draw(const Frame& f, int width, int height, const DrawState& state = {})
        {
            mgl::glViewport(0, 0, width, height);
            mgl::glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], kClearColor[3]);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            if (state.depth_test) mgl::glEnable(mgl::Capability::DepthTest);
            else mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glEnable(mgl::Capability::CullFace);  // back faces are never visible: every mesh is closed or a one-sided plane
            mgl::glCullFace(state.cull);
            mgl::glFrontFace(state.front);

            upload_frame_uniforms(phong_, f);
            upload_frame_uniforms(gouraud_, f);
            std::vector<Item> items = make_items(f);
            if (state.reverse_order) std::reverse(items.begin(), items.end());
            for (const Item& it : items)
            {
                const Program& prog = it.gouraud ? gouraud_ : phong_;
                mgl::glUseProgram(prog.id);
                upload_object_uniforms(prog, it);
                const GpuMesh& mesh = meshes_[static_cast<std::size_t>(it.mesh)];
                mgl::glBindVertexArray(mesh.vao);
                mgl::glDrawElements(mgl::PrimitiveType::Triangles, mesh.index_count, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        Program phong_{}, gouraud_{};
        std::array<GpuMesh, static_cast<std::size_t>(MeshId::Count)> meshes_{};
        std::array<MeshData, static_cast<std::size_t>(MeshId::Count)> mesh_data_{};
        Params params_{};
    };

    // ---------------------------------------------------------------------------------------------
    // Validation helpers
    // ---------------------------------------------------------------------------------------------
    // Tolerances (8-bit channels). The float -> unorm conversion returns one of the two closest values
    // (ES 3.0 section 2.1.6.2), i.e. +-1. GLSL ES 3.00 section 4.5.1 allows a few ULP for the arithmetic
    // and pow(), which is far below one 8-bit step for these formulas; one more step is allowed for the
    // interpolation of world position and normal (kExactTol = 2). The smooth spheres are a 96 x 48 polygon
    // approximation of the analytic sphere the reference intersects: the interpolated normal at the point
    // where the ray meets the flat triangle differs from the analytic one by < 0.003 rad, and the steepest
    // highlight flanks of these materials change by 300-600 levels per radian, i.e. 1-2 more steps
    // (kSphereTol = 3). The largest difference measured on gpu, llvmpipe and zink is 1; the checks print it.
    constexpr int kExactTol = 2;
    constexpr int kSphereTol = 3;
    constexpr int kDiscriminate = 12;  // a wrong model must be off by at least this much for a probe to count as discriminating

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

    int diff_rgba(demo::Rgba8 a, demo::Rgba8 b)
    {
        int d = 0;
        for (std::size_t i = 0; i < 4; ++i) d = std::max(d, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        return d;
    }

    const char* obj_name(Obj o)
    {
        switch (o)
        {
        case Obj::Sky: return "sky";
        case Obj::Ground: return "ground";
        case Obj::SpherePhong: return "Phong sphere";
        case Obj::SphereBlinn: return "Blinn-Phong sphere";
        case Obj::CrystalCorrect: return "crystal (correct normal matrix)";
        case Obj::CrystalNaive: return "crystal (naive normal matrix)";
        case Obj::CoarseGouraud: return "coarse sphere (per vertex)";
        case Obj::CoarseFragment: return "coarse sphere (per fragment)";
        case Obj::PointLamp: return "point lamp";
        case Obj::SpotLamp: return "spot lamp";
        }
        return "?";
    }

    int tolerance_for(Obj o) { return (o == Obj::SpherePhong || o == Obj::SphereBlinn) ? kSphereTol : kExactTol; }

    struct SweepStat
    {
        int n = 0, bad = 0, max_diff = 0, wx = -1, wy = -1;
        demo::Rgba8 got{}, want{};
    };

    // Compares every `step`-th pixel whose CPU ray hits a surface the reference models, away from
    // silhouettes, tile edges and crystal edges (Reference::stable), with the CPU colour.
    std::array<SweepStat, 10> sweep(const Reference& ref, const Img& img, int step)
    {
        std::array<SweepStat, 10> st{};
        for (int y = 4; y < img.h - 4; y += step)
            for (int x = 4; x < img.w - 4; x += step)
            {
                const Hit h = ref.trace(ref.ray(x, y));
                if (h.obj == Obj::SpotLamp) continue;
                if ((h.obj == Obj::SpherePhong || h.obj == Obj::SphereBlinn) && h.n_dot_v < 0.3) continue;  // grazing: chord error grows
                if (!ref.stable(x, y, h)) continue;
                SweepStat& s = st[static_cast<std::size_t>(h.obj)];
                const demo::Rgba8 want = ref.expected(h), got = img.at(x, y);
                const int d = diff_rgba(got, want);
                ++s.n;
                if (d > tolerance_for(h.obj)) ++s.bad;
                if (d > s.max_diff)
                {
                    s.max_diff = d;
                    s.wx = x;
                    s.wy = y;
                    s.got = got;
                    s.want = want;
                }
            }
        return st;
    }

    struct Probe
    {
        bool ok = false;
        int x = 0, y = 0;
        Hit hit;
    };

    // The pixel a world point projects to, valid when the CPU ray through that pixel centre hits `expect`
    // first and the pixel is stable. The hit point differs from `world` by at most half a pixel; the
    // expected colour is always computed at the exact hit.
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

    std::string rgba_text(demo::Rgba8 c) { return demo::format_rgba(c); }

    // Records one check comparing the pixel of a probe with an expected colour; returns the difference.
    int expect_probe(demo::Check& check, const Img& img, const Probe& p, demo::Rgba8 want, int tol, const std::string& what,
                     const std::string& note = {})
    {
        if (!p.ok)
        {
            check.expect(false, what, "probe setup failed: the CPU ray through the projected pixel does not hit the expected, stable surface");
            return -1;
        }
        const demo::Rgba8 got = img.at(p.x, p.y);
        const int d = diff_rgba(got, want);
        check.expect(d <= tol, what,
                     std::format("pixel({},{}) = {} expected {} max channel diff {} tol {}{}{}", p.x, p.y, rgba_text(got), rgba_text(want), d, tol,
                                 note.empty() ? "" : "; ", note));
        return d;
    }

    // Ground point at distance rho from `origin` (on y = 0) along the xz direction at `degrees`.
    D3 ground_point(D3 origin, double degrees, double rho)
    {
        const double a = degrees * std::numbers::pi / 180.0;
        return {origin.x + rho * std::cos(a), 0.0, origin.z + rho * std::sin(a)};
    }

    // Angle (degrees) between the spot axis and the ray light -> p.
    double spot_angle_deg(const Lights& l, D3 p)
    {
        const double c = dot(normalize(p - d3(l.spot_position)), d3(l.spot_direction));
        return std::acos(std::clamp(c, -1.0, 1.0)) * 180.0 / std::numbers::pi;
    }

    void Lighting::validate(demo::Env& env, demo::Check& check)
    {
        const int W = env.width, H = env.height;
        const float asp = aspect(env);
        const Params base{};  // t = 0, every light on, Blinn-Phong, naive right crystal, Gouraud left coarse sphere
        auto frame_of = [&](const Params& p) { return make_frame(p, asp); };
        auto snapshot = [&](const Frame& f, const DrawState& state = {}) {
            draw(f, W, H, state);
            return Img{W, H, demo::read_rgba8(0, 0, W, H)};
        };
        auto lights = [&](bool dir, bool point, bool spot) {
            Params p = base;
            p.dir_on = dir;
            p.point_on = point;
            p.spot_on = spot;
            return p;
        };
        const Frame f_all = frame_of(base);

        // ---- A. mesh winding, normal matrix, uniform upload --------------------------------------------------------------
        {
            std::size_t tris = 0, bad = 0;
            for (const MeshData& md : mesh_data_)
                for (std::size_t k = 0; k + 2 < md.indices.size(); k += 3)
                {
                    const Vertex &a = md.vertices[md.indices[k]], &b = md.vertices[md.indices[k + 1]], &c = md.vertices[md.indices[k + 2]];
                    const D3 e1{dbl(b.px - a.px), dbl(b.py - a.py), dbl(b.pz - a.pz)};
                    const D3 e2{dbl(c.px - a.px), dbl(c.py - a.py), dbl(c.pz - a.pz)};
                    const D3 gn = cross(e1, e2);
                    if (length(gn) < 1e-7) continue;  // degenerate triangles at the sphere poles
                    ++tris;
                    if (dot(gn, D3{dbl(a.nx + b.nx + c.nx), dbl(a.ny + b.ny + c.ny), dbl(a.nz + b.nz + c.nz)}) <= 0.0) ++bad;
                }
            check.expect(tris > 1000 && bad == 0, "all mesh triangles are counter-clockwise seen from outside (geometric normal agrees with the vertex normals), so glCullFace(BACK) keeps the visible side",
                         std::format("{} triangles checked, {} wrong", tris, bad));
        }
        {
            const double angle = dbl(f_all.crystal_angle);
            const m::Mat4 model = crystal_model(kCrystalLeftXz, f_all.crystal_angle);
            const std::array<float, 9> nm = m::normal_matrix(model);
            const std::array<float, 9> naive = naive_normal_matrix(model);
            const double c = std::cos(angle), s = std::sin(angle);
            const D3 sc = d3(kCrystalScale);
            // model = T * S * R  =>  normal matrix = (S R)^-T = S^-1 R, written out by hand (row-major here)
            const double expect_rows[3][3] = {{c / sc.x, 0.0, s / sc.x}, {0.0, 1.0 / sc.y, 0.0}, {-s / sc.z, 0.0, c / sc.z}};
            double err = 0.0, vs_naive = 0.0;
            for (int col = 0; col < 3; ++col)
                for (int row = 0; row < 3; ++row)
                {
                    const std::size_t i = static_cast<std::size_t>(col * 3 + row);
                    err = std::max(err, std::abs(dbl(nm[i]) - expect_rows[row][col]));
                    vs_naive = std::max(vs_naive, std::abs(dbl(naive[i]) - expect_rows[row][col]));
                }
            check.expect_near(err, 0.0, 1e-5, "demo::math::normal_matrix(T S R) equals S^-1 R (largest element error)");
            check.expect(vs_naive > 0.2, "the naive upper-left 3x3 of the model matrix is NOT the normal matrix for this non-uniform scale",
                         std::format("largest element difference {:.3f}", vs_naive));

            // glUniformMatrix{3,4}fv with transpose = GL_FALSE stores column-major data; glGetUniformfv returns matrices
            // in column-major order (ES 3.0 section 6.1.12), so the round trip is exact.
            const std::vector<Item> items = make_items(f_all);
            const auto it = std::find_if(items.begin(), items.end(), [](const Item& i) { return i.id == Obj::CrystalCorrect; });
            for (const Program* prog : {&phong_, &gouraud_})
            {
                mgl::glUseProgram(prog->id);
                upload_frame_uniforms(*prog, f_all);
                upload_object_uniforms(*prog, *it);
                std::array<float, 16> got_model{}, got_view{}, got_proj{};
                std::array<float, 9> got_nm{};
                mgl::glGetUniformfv(prog->id, prog->u.model, got_model.data());
                mgl::glGetUniformfv(prog->id, prog->u.view, got_view.data());
                mgl::glGetUniformfv(prog->id, prog->u.proj, got_proj.data());
                mgl::glGetUniformfv(prog->id, prog->u.normal_matrix, got_nm.data());
                const char* name = prog == &phong_ ? "lighting program" : "Gouraud program";
                check.expect(got_model == it->model.m && got_view == f_all.view.m && got_proj == f_all.proj.m,
                             std::format("{}: mat4 uniforms u_model, u_view, u_proj read back column-major, equal to the CPU matrices", name));
                check.expect(got_nm == nm, std::format("{}: mat3 uniform u_normal_matrix reads back equal to demo::math::normal_matrix(model)", name));
            }

            // Defined errors (ES 3.0 section 2.12.6.1): the Uniform* command must match the size of the uniform, count > 1
            // needs an array, and a rejected command changes nothing. Location -1 (an inactive uniform) is silently ignored.
            mgl::glUseProgram(phong_.id);
            std::array<float, 9> before{}, after{};
            std::array<float, 18> junk{};
            mgl::glGetUniformfv(phong_.id, phong_.u.normal_matrix, before.data());
            (void)demo::drain_gl_errors();
            mgl::glUniformMatrix4fv(phong_.u.normal_matrix, 1, GL_FALSE, junk.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniformMatrix4fv on the mat3 u_normal_matrix (a mat4 normal matrix by mistake): INVALID_OPERATION");
            mgl::glUniformMatrix3fv(phong_.u.normal_matrix, 2, GL_FALSE, junk.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniformMatrix3fv with count 2 on the non-array mat3 u_normal_matrix: INVALID_OPERATION");
            mgl::glGetUniformfv(phong_.id, phong_.u.normal_matrix, after.data());
            check.expect(before == after, "the rejected glUniformMatrix* calls left u_normal_matrix unchanged");
            mgl::glUseProgram(gouraud_.id);
            mgl::glUniform1f(mgl::UniformLocation{-1}, 1.0f);
            mgl::glUniform3f(mgl::UniformLocation{-1}, 1.0f, 2.0f, 3.0f);
            check.expect_no_gl_error("glUniform*(location -1, ...) is silently ignored: no error (the location of an inactive uniform)");
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ---- B. Sweeps and probes under isolated lights ---------------------------------------------------------------------
        auto run_sweep = [&](const std::string& label, const Reference& ref, const Img& img) {
            const auto st = sweep(ref, img, 5);
            for (const Obj o : {Obj::Ground, Obj::SpherePhong, Obj::SphereBlinn, Obj::CrystalCorrect, Obj::CrystalNaive, Obj::CoarseGouraud,
                                Obj::CoarseFragment})
            {
                const SweepStat& s = st[static_cast<std::size_t>(o)];
                const bool seen = s.n >= 8;
                std::string detail = std::format("{} pixels compared, max channel diff {} (tol {})", s.n, s.max_diff, tolerance_for(o));
                if (s.max_diff > tolerance_for(o))
                    detail += std::format("; worst pixel({},{}) = {} expected {}", s.wx, s.wy, rgba_text(s.got), rgba_text(s.want));
                check.expect(seen && s.bad == 0, std::format("{}: {} matches the CPU reference", label, obj_name(o)), detail);
            }
        };

        // B1. Ambient light only: every surface shows ka * albedo (ground tiles of both colours, the sphere).
        {
            const Frame f = frame_of(lights(false, false, false));
            const Reference ref(f, W, H);
            const Img img = snapshot(f);
            const D3 amb = d3(kAmbient);
            const Probe tile_a = make_probe(ref, W, H, D3{0.75, 0.0, 0.75}, Obj::Ground);   // tile (0, 0): albedo A
            const Probe tile_b = make_probe(ref, W, H, D3{-0.75, 0.0, 0.75}, Obj::Ground);  // tile (-1, 0): albedo B
            expect_probe(check, img, tile_a, encode_color(amb * d3(kGroundMat.albedo)), 1,
                         "ambient only: checker tile A shows ambient * albedo A (gamma-encoded)");
            expect_probe(check, img, tile_b, encode_color(amb * d3(kGroundMat.albedo2)), 1,
                         "ambient only: the neighbouring tile shows ambient * albedo B");
            const Probe sph = make_probe(ref, W, H, d3(kPhongPos), Obj::SpherePhong);
            expect_probe(check, img, sph, encode_color(amb * d3(kSphereMat.albedo)), 1, "ambient only: a sphere is flat ka * albedo, no shading without a light");
            if (tile_a.ok && tile_b.ok)
                check.expect(diff_rgba(img.at(tile_a.x, tile_a.y), img.at(tile_b.x, tile_b.y)) > 10, "ambient only: the two checker tiles differ (the pattern is really drawn)");
            run_sweep("ambient only", ref, img);
        }

        // B2. Directional light only: Phong vs Blinn-Phong highlights, correct vs naive normal matrix, Gouraud vs per fragment.
        D3 gouraud_point{};  // surface point of the left coarse sphere where per-vertex and per-fragment lighting differ most
        {
            const Frame f = frame_of(lights(true, false, false));
            const Reference ref(f, W, H);
            const Img img = snapshot(f);
            run_sweep("directional only", ref, img);

            const D3 to_light = normalize(d3(f.lights.dir_direction) * -1.0);
            for (const double alpha : {0.0, 0.14, 0.28})
            {
                int lum[2] = {0, 0};
                for (const Obj o : {Obj::SpherePhong, Obj::SphereBlinn})
                {
                    const D3 centre = d3(o == Obj::SpherePhong ? kPhongPos : kBlinnPos);
                    const D3 half = normalize(to_light + normalize(ref.eye() - centre));
                    const D3 tangent = normalize(cross(half, D3{0.0, 1.0, 0.0}));
                    const D3 n = normalize(half * std::cos(alpha) + tangent * std::sin(alpha));  // normal alpha radians away from H
                    const Probe p = make_probe(ref, W, H, centre + n * dbl(kSphereRadius), o);
                    if (!p.ok)
                    {
                        check.expect(false, std::format("highlight probe on the {}", obj_name(o)), "probe setup failed");
                        continue;
                    }
                    Material other = ref.material(o);
                    other.model = 1 - other.model;
                    const demo::Rgba8 own = ref.expected(p.hit);
                    const demo::Rgba8 wrong = encode_color(ref.shade_point_as(other, p.hit.pos, p.hit.normal));
                    const bool discriminating = diff_rgba(own, wrong) >= kDiscriminate;
                    const int d_wrong = diff_rgba(img.at(p.x, p.y), wrong);
                    expect_probe(check, img, p, own, kSphereTol,
                                 std::format("directional: {} at {:.2f} rad from the half vector matches its own model", obj_name(o), alpha),
                                 std::format("the {} model would give {} (differs by {})", o == Obj::SpherePhong ? "Blinn-Phong" : "Phong", rgba_text(wrong),
                                             diff_rgba(own, wrong)));
                    if (alpha > 0.1)
                    {
                        check.expect(discriminating && d_wrong >= kDiscriminate,
                                     std::format("directional: {} at {:.2f} rad is NOT what the other specular model predicts", obj_name(o), alpha),
                                     std::format("pixel is {} from the other model's value (needs >= {})", d_wrong, kDiscriminate));
                    }
                    const demo::Rgba8 got = img.at(p.x, p.y);
                    lum[o == Obj::SpherePhong ? 0 : 1] = got[0] + got[1] + got[2];
                }
                if (alpha > 0.1)
                    check.expect(lum[1] - lum[0] >= 3 * kDiscriminate, std::format("directional: at {:.2f} rad off the highlight centre the Blinn-Phong sphere is brighter than the Phong sphere", alpha),
                                 std::format("channel sum Blinn-Phong {} Phong {}", lum[1], lum[0]));
            }

            // Crystals: model = T S R with a non-uniform S. Left crystal: correct normal matrix. Right crystal: naive mat3(model).
            const double angle = dbl(f.crystal_angle);
            for (const Obj o : {Obj::CrystalCorrect, Obj::CrystalNaive})
            {
                const m::Vec3& xz = o == Obj::CrystalCorrect ? kCrystalLeftXz : kCrystalRightXz;
                int discriminating = 0;
                for (int face = 0; face < 6; ++face)
                {
                    D3 q{};
                    (face / 2 == 0 ? q.x : face / 2 == 1 ? q.y : q.z) = (face % 2) != 0 ? 0.5 : -0.5;
                    const Probe p = make_probe(ref, W, H, crystal_point(xz, angle, q), o);
                    if (!p.ok || p.hit.tag != face) continue;  // face hidden from the camera
                    const demo::Rgba8 right_model = encode_color(ref.shade_point(o, p.hit.pos, crystal_face_normal(angle, face, false)));
                    const demo::Rgba8 naive_model = encode_color(ref.shade_point(o, p.hit.pos, crystal_face_normal(angle, face, true)));
                    const bool naive = o == Obj::CrystalNaive;
                    const int model_gap = diff_rgba(right_model, naive_model);
                    const std::string what = std::format("directional: {} face {} matches the {} normal model", obj_name(o), face, naive ? "naive" : "correct");
                    expect_probe(check, img, p, naive ? naive_model : right_model, kExactTol, what,
                                 std::format("the {} model would give {} (differs by {})", naive ? "correct" : "naive", rgba_text(naive ? right_model : naive_model), model_gap));
                    if (model_gap >= kDiscriminate)
                    {
                        ++discriminating;
                        const int d_other = diff_rgba(img.at(p.x, p.y), naive ? right_model : naive_model);
                        check.expect(d_other >= kDiscriminate,
                                     std::format("directional: {} face {} is NOT what the {} normal model predicts", obj_name(o), face, naive ? "correct" : "naive"),
                                     std::format("pixel is {} from the other model's value (needs >= {})", d_other, kDiscriminate));
                    }
                }
                check.expect(discriminating >= 1, std::format("directional: {} has a visible face where the correct and the naive normal matrix differ", obj_name(o)),
                             std::format("{} such faces", discriminating));
            }

            // Coarse spheres: find the point of the left (per-vertex) sphere where per-vertex and per-fragment lighting differ most.
            {
                const D3 centre = d3(kGouraudPos);
                const D3 half = normalize(to_light + normalize(ref.eye() - centre));
                const D3 t1 = normalize(cross(half, D3{0.0, 1.0, 0.0})), t2 = cross(half, t1);
                Probe best;
                int best_gap = -1;
                D3 best_dir{};
                for (int ia = -5; ia <= 5; ++ia)
                    for (int ib = -5; ib <= 5; ++ib)
                    {
                        const D3 n = normalize(half + t1 * (0.1 * ia) + t2 * (0.1 * ib));
                        const Probe p = make_probe(ref, W, H, centre + n * dbl(kSphereRadius), Obj::CoarseGouraud);
                        if (!p.ok) continue;
                        const int gap = diff_rgba(encode_color(p.hit.color_vertex), encode_color(p.hit.color_fragment));
                        if (gap > best_gap)
                        {
                            best_gap = gap;
                            best = p;
                            best_dir = n;
                        }
                    }
                if (best.ok)
                {
                    gouraud_point = centre + best_dir * dbl(kSphereRadius);
                    expect_probe(check, img, best, encode_color(best.hit.color_vertex), kExactTol,
                                 "directional: the per-vertex lit coarse sphere equals the interpolated vertex colours (exact ray / triangle barycentrics)",
                                 std::format("per-fragment lighting would give {}", rgba_text(encode_color(best.hit.color_fragment))));
                    check.expect(best_gap >= kDiscriminate, "directional: at that point per-vertex and per-fragment lighting of the same mesh differ clearly",
                                 std::format("{} channel steps apart", best_gap));
                    const Probe other = make_probe(ref, W, H, d3(kFragmentPos) + best_dir * dbl(kSphereRadius), Obj::CoarseFragment);
                    expect_probe(check, img, other, other.ok ? encode_color(other.hit.color_fragment) : demo::Rgba8{}, kExactTol,
                                 "directional: the per-fragment coarse sphere equals the lighting of the interpolated position and normal at the analogous point",
                                 other.ok ? std::format("per-vertex lighting would give {}", rgba_text(encode_color(other.hit.color_vertex))) : std::string{});
                }
                else
                    check.expect(false, "directional: found a probe on the per-vertex coarse sphere", "no stable probe");
            }
        }

        // B3. Point light only: attenuation at three distances, a sphere facing the lamp.
        std::array<Probe, 3> point_probes{};
        std::array<demo::Rgba8, 3> point_wants{};
        Frame f_point = frame_of(lights(false, true, false));
        {
            const Reference ref(f_point, W, H);
            const Img img = snapshot(f_point);
            run_sweep("point light only", ref, img);
            const D3 lamp = d3(f_point.lights.point_position);
            const D3 foot{lamp.x, 0.0, lamp.z};
            const double rho[3] = {0.0, 1.4, 2.8};
            bool found = false;
            double used_deg = 0.0;
            for (const double deg : {90.0, 135.0, 45.0, 180.0, 0.0, 225.0, 315.0, 270.0, 112.5, 67.5, 157.5, 22.5})
            {
                std::array<Probe, 3> cand{};
                bool all = true;
                for (std::size_t k = 0; k < 3; ++k)
                {
                    cand[k] = make_probe(ref, W, H, ground_point(foot, deg, rho[k]), Obj::Ground);
                    all = all && cand[k].ok;
                }
                if (all)
                {
                    point_probes = cand;
                    used_deg = deg;
                    found = true;
                    break;
                }
            }
            check.expect(found, "point light: found three unoccluded ground probes at increasing distance from the lamp's foot point",
                         found ? std::format("along the direction {} degrees in the xz plane", used_deg) : "none of the candidate directions is clear");
            if (found)
            {
                const double kc = dbl(kPointAtten.x), kl = dbl(kPointAtten.y), kq = dbl(kPointAtten.z);
                double previous = 1.0;  // attenuation at distance 0
                for (std::size_t k = 0; k < 3; ++k)
                {
                    const D3 pos = point_probes[k].hit.pos;
                    const double d = length(lamp - pos);
                    const double atten = 1.0 / (kc + kl * d + kq * d * d);
                    point_wants[k] = ref.expected(point_probes[k].hit);
                    expect_probe(check, img, point_probes[k], point_wants[k], kExactTol,
                                 std::format("point light: ground {:.1f} units from the lamp, attenuation {:.3f}", d, atten),
                                 std::format("1 / ({:.2f} + {:.2f} d + {:.2f} d^2) with d = {:.3f}", kc, kl, kq, d));
                    check.expect(atten < previous, std::format("point light: attenuation decreases with distance ({:.3f} < {:.3f})", atten, previous));
                    previous = atten;
                }
            }
            // The point of each sphere whose normal bisects the directions to the lamp and to the camera: the centre of the
            // specular highlight of the point light (the light direction L differs per point, unlike the directional light).
            for (const Obj o : {Obj::SpherePhong, Obj::SphereBlinn})
            {
                const D3 centre = d3(o == Obj::SpherePhong ? kPhongPos : kBlinnPos);
                const D3 n = normalize(normalize(lamp - centre) + normalize(ref.eye() - centre));
                const Probe p = make_probe(ref, W, H, centre + n * dbl(kSphereRadius), o);
                expect_probe(check, img, p, ref.expected(p.hit), kSphereTol, std::format("point light: highlight centre on the {} (normal bisects lamp and camera)", obj_name(o)));
            }
        }

        // B4. Spot light only: inside the inner cone, in the smooth falloff, outside the outer cone.
        std::array<Probe, 7> spot_probes{};
        std::array<double, 7> spot_cone{};
        const Frame f_spot = frame_of(lights(false, false, true));
        {
            const Reference ref(f_spot, W, H);
            const Img img = snapshot(f_spot);
            run_sweep("spot light only", ref, img);
            const Lights& l = f_spot.lights;
            const D3 spos = d3(l.spot_position), sdir = d3(l.spot_direction);
            const D3 axis_hit = spos + sdir * (-spos.y / sdir.y);  // where the cone axis meets the ground
            const double inner = kSpotInnerDeg, outer = kSpotOuterDeg;
            const double thetas[7] = {0.0, 0.5 * inner, 0.95 * inner, 0.5 * (inner + outer), outer - 0.2 * (outer - inner), outer + 3.0, outer + 10.0};
            const char* names[7] = {"on the axis", "half way to the inner edge", "just inside the inner cone", "in the middle of the falloff",
                                    "near the outer edge", "just outside the outer cone", "far outside the cone"};
            bool found = false;
            for (const double deg : {90.0, 135.0, 45.0, 180.0, 0.0, 225.0, 315.0, 270.0, 112.5, 67.5, 157.5, 22.5})
            {
                std::array<Probe, 7> cand{};
                bool all = true;
                for (std::size_t k = 0; k < 7 && all; ++k)
                {
                    double lo = 0.0, hi = 30.0;  // bisection on the distance from the axis point: the angle grows monotonically
                    for (int it = 0; it < 60; ++it)
                    {
                        const double mid = 0.5 * (lo + hi);
                        (spot_angle_deg(l, ground_point(axis_hit, deg, mid)) < thetas[k] ? lo : hi) = mid;
                    }
                    cand[k] = make_probe(ref, W, H, ground_point(axis_hit, deg, 0.5 * (lo + hi)), Obj::Ground);
                    all = cand[k].ok;
                }
                if (all)
                {
                    spot_probes = cand;
                    found = true;
                    break;
                }
            }
            check.expect(found, "spot light: found seven unoccluded ground probes from the cone axis to far outside the cone");
            if (found)
            {
                const D3 amb_only = d3(kAmbient);
                for (std::size_t k = 0; k < 7; ++k)
                {
                    const Hit& h = spot_probes[k].hit;
                    spot_cone[k] = ref_spot_cone(l, h.pos);
                    const double theta = spot_angle_deg(l, h.pos);
                    expect_probe(check, img, spot_probes[k], ref.expected(h), kExactTol,
                                 std::format("spot light: ground {} (angle {:.1f} deg, cone factor {:.3f})", names[k], theta, spot_cone[k]),
                                 std::format("inner {} deg, outer {} deg", inner, outer));
                }
                check.expect(spot_cone[0] == 1.0 && spot_cone[1] == 1.0 && spot_cone[2] == 1.0, "spot light: the three inner probes have cone factor exactly 1");
                check.expect(spot_cone[3] > 0.25 && spot_cone[3] < 0.75 && spot_cone[4] > 0.02 && spot_cone[4] < 0.4,
                             "spot light: the two falloff probes have a strictly partial cone factor", std::format("{:.3f} and {:.3f}", spot_cone[3], spot_cone[4]));
                check.expect(spot_cone[5] == 0.0 && spot_cone[6] == 0.0, "spot light: the two outer probes have cone factor exactly 0");
                // Outside the cone the spot adds exactly nothing: the pixel is the ambient-only colour of its tile.
                for (const std::size_t k : {std::size_t{5}, std::size_t{6}})
                {
                    const D3 alb = ref_albedo(kGroundMat, spot_probes[k].hit.pos);
                    expect_probe(check, img, spot_probes[k], encode_color(amb_only * alb), 1, "spot light: outside the outer cone the pixel is exactly ambient * albedo",
                                 "spot contributes nothing, not even a faint tail");
                }
                // Smooth falloff vs a hard cone: the hard-cone model must be wrong in the falloff region.
                const Hit& mid = spot_probes[3].hit;
                const demo::Rgba8 hard = encode_color(ref.shade_point(Obj::Ground, mid.pos, mid.normal, true));
                const int d_hard = diff_rgba(img.at(spot_probes[3].x, spot_probes[3].y), hard);
                check.expect(d_hard >= kDiscriminate, "spot light: a hard-edged cone (no smoothstep) does NOT predict the falloff pixel",
                             std::format("pixel is {} steps away from the hard-cone value {}", d_hard, rgba_text(hard)));
            }
        }

        // B5. All lights (the canonical frame): sweep, projected positions of the two lamps.
        Img img_canonical;
        {
            const Reference ref(f_all, W, H);
            const Img img = snapshot(f_all);
            img_canonical = img;
            run_sweep("all lights", ref, img);
            const demo::Rgba8 lamp_want = encode_color(d3(kPointLampMat.emissive));
            const Probe lamp = make_probe(ref, W, H, d3(f_all.lights.point_position), Obj::PointLamp);
            expect_probe(check, img, lamp, lamp_want, 1, "model/view/projection: the CPU-projected centre of the point lamp shows its emissive colour");
            const Probe spot_lamp = make_probe(ref, W, H, d3(f_all.lights.spot_position) - d3(f_all.lights.spot_direction) * 0.25, Obj::SpotLamp);
            expect_probe(check, img, spot_lamp, encode_color(d3(kSpotLampMat.emissive)), 1,
                         "model/view/projection: the CPU-projected axis midpoint of the spot lamp cone shows its emissive colour (rotated model matrix)");
            for (const Obj o : {Obj::SpherePhong, Obj::CrystalNaive, Obj::CoarseGouraud})
            {
                const D3 c = o == Obj::SpherePhong ? d3(kPhongPos) : o == Obj::CrystalNaive ? crystal_centre(kCrystalRightXz) : d3(kGouraudPos);
                const Probe p = make_probe(ref, W, H, c, o);
                expect_probe(check, img, p, ref.expected(p.hit), tolerance_for(o),
                             std::format("model/view/projection: the CPU-projected centre of the {} shows that object", obj_name(o)));
            }
        }

        // B6. Negative controls (the checks above can fail).
        {
            // (a) GPU mutation: remove the distance attenuation from the point light; the attenuated CPU values must now be wrong.
            Frame f = f_point;
            f.lights.point_atten = {1.0f, 0.0f, 0.0f};
            const Img img = snapshot(f);
            const Reference mutated_ref(f, W, H);
            if (point_probes[2].ok)
            {
                const int d = diff_rgba(img.at(point_probes[2].x, point_probes[2].y), point_wants[2]);
                check.expect(d >= kDiscriminate, "negative control: with the attenuation constants (1, 0, 0) the far ground probe no longer matches the attenuated CPU value",
                             std::format("pixel differs by {} (needs >= {})", d, kDiscriminate));
                const Probe again = make_probe(mutated_ref, W, H, point_probes[2].hit.pos, Obj::Ground);
                expect_probe(check, img, again, mutated_ref.expected(again.hit), kExactTol, "negative control: ... but it matches the CPU model that uses the same mutated constants");
            }
            // (b) GPU mutation: the right crystal gets the correct normal matrix (key n). Its pixel must move to the correct-model value.
            Params fixed = lights(true, false, false);
            fixed.naive_fixed = true;
            const Frame ff = frame_of(fixed);
            const Reference ref_fixed(ff, W, H);
            const Img img_fixed = snapshot(ff);
            const double angle = dbl(ff.crystal_angle);
            int moved = 0, matched = 0, total = 0;
            for (int face = 0; face < 6; ++face)
            {
                D3 q{};
                (face / 2 == 0 ? q.x : face / 2 == 1 ? q.y : q.z) = (face % 2) != 0 ? 0.5 : -0.5;
                const Probe p = make_probe(ref_fixed, W, H, crystal_point(kCrystalRightXz, angle, q), Obj::CrystalNaive);
                if (!p.ok || p.hit.tag != face) continue;
                const demo::Rgba8 correct = encode_color(ref_fixed.shade_point(Obj::CrystalNaive, p.hit.pos, crystal_face_normal(angle, face, false)));
                const demo::Rgba8 naive = encode_color(ref_fixed.shade_point(Obj::CrystalNaive, p.hit.pos, crystal_face_normal(angle, face, true)));
                const demo::Rgba8 got = img_fixed.at(p.x, p.y);
                ++total;
                if (diff_rgba(got, correct) <= kExactTol) ++matched;
                if (diff_rgba(correct, naive) >= kDiscriminate && diff_rgba(got, naive) >= kDiscriminate) ++moved;
            }
            check.expect(total >= 2 && matched == total && moved >= 1, "key n: with the correct normal matrix the right crystal's visible faces equal the correct-normal model and left the naive values",
                         std::format("{} visible faces, {} match the correct model, {} discriminating faces moved away from the naive model", total, matched, moved));
            const auto st = sweep(ref_fixed, img_fixed, 5);
            check.expect(st[static_cast<std::size_t>(Obj::CrystalNaive)].n >= 8 && st[static_cast<std::size_t>(Obj::CrystalNaive)].bad == 0,
                         "key n: the whole right crystal then matches the CPU reference with correct normals",
                         std::format("{} pixels, max diff {}", st[static_cast<std::size_t>(Obj::CrystalNaive)].n, st[static_cast<std::size_t>(Obj::CrystalNaive)].max_diff));
        }

        // B6b. Other states reachable from the keyboard or the clock still match the CPU reference.
        {
            Params phong_everywhere = base;
            phong_everywhere.blinn = false;  // key b: ground, crystals and coarse spheres switch to the Phong specular model
            {
                const Frame f = frame_of(phong_everywhere);
                const Reference ref(f, W, H);
                const Img img = snapshot(f);
                run_sweep("key b (Phong specular on every surface)", ref, img);
                int changed = 0;
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        if (diff_rgba(img.at(x, y), img_canonical.at(x, y)) >= kDiscriminate) ++changed;
                check.expect(changed >= 10, "key b: switching the global specular model visibly changes the image (the sweep above is discriminating)",
                             std::format("{} pixels differ by at least {} steps", changed, kDiscriminate));
            }
            Params later = base;
            later.ticks = 137;  // t = 2.28 s: lamps and cone have moved, the crystals have rotated, the camera sways
            {
                const Frame f = frame_of(later);
                const Reference ref(f, W, H);
                run_sweep("animation at t = 2.28 s", ref, snapshot(f));
            }
            Params per_fragment = lights(true, false, false);
            per_fragment.gouraud = false;  // key g: the left coarse sphere is lit per fragment like the right one
            {
                const Frame f = frame_of(per_fragment);
                const Reference ref(f, W, H);
                const Img img = snapshot(f);
                run_sweep("key g (per-fragment on both coarse spheres)", ref, img);
                const Probe p = make_probe(ref, W, H, gouraud_point, Obj::CoarseGouraud);
                if (p.ok)
                {
                    const int d_vertex = diff_rgba(img.at(p.x, p.y), encode_color(p.hit.color_vertex));
                    expect_probe(check, img, p, encode_color(p.hit.color_fragment), kExactTol, "key g: the former per-vertex sphere now shows per-fragment lighting at the probe point",
                                 std::format("the per-vertex value would be {} away", d_vertex));
                    check.expect(d_vertex >= kDiscriminate, "key g: ... which the per-vertex model would not predict", std::format("{} channel steps apart", d_vertex));
                }
                else
                    check.expect(false, "key g: probe on the former per-vertex sphere", "probe setup failed");
            }
        }

        // B6c. Depth test and face culling decide visibility (ES 3.0 sections 4.1.5 and 3.6.1).
        {
            const Reference ref(f_all, W, H);
            // Items drawn in the opposite order: the ground, drawn last, must not overwrite the objects in front of it.
            DrawState reversed;
            reversed.reverse_order = true;
            run_sweep("depth test with reversed draw order (ground drawn last)", ref, snapshot(f_all, reversed));
            reversed.depth_test = false;
            const Img no_depth = snapshot(f_all, reversed);
            const Probe sphere = make_probe(ref, W, H, d3(kPhongPos), Obj::SpherePhong);
            if (sphere.ok)
            {
                // Without the depth test the ground (drawn last) covers everything: the sphere pixel shows the ground colour there.
                const Ray r = ref.ray(sphere.x, sphere.y);
                const D3 p = r.o + r.d * (-r.o.y / r.d.y);  // where this pixel's ray meets the ground plane behind the sphere
                const demo::Rgba8 ground_colour = encode_color(ref.shade_point(Obj::Ground, p, D3{0.0, 1.0, 0.0}));
                const int d = diff_rgba(no_depth.at(sphere.x, sphere.y), ground_colour);
                check.expect(d <= kExactTol, "negative control: with glDisable(DEPTH_TEST) and the ground drawn last, the pixel at the sphere centre shows the ground behind it",
                             std::format("pixel {} expected ground {} (diff {})", rgba_text(no_depth.at(sphere.x, sphere.y)), rgba_text(ground_colour), d));
            }
            else
                check.expect(false, "negative control: depth test disabled", "probe setup failed");

            // The ground is a one-sided plane that faces +y: front-facing for the camera above it.
            const Probe tile = make_probe(ref, W, H, D3{0.75, 0.0, 0.75}, Obj::Ground);
            const demo::Rgba8 clear = {to8(dbl(kClearColor[0])), to8(dbl(kClearColor[1])), to8(dbl(kClearColor[2])), 255};
            struct Case
            {
                mgl::CullFace cull;
                mgl::FrontFace front;
                bool visible;
                const char* what;
            };
            const Case cases[] = {{mgl::CullFace::Back, mgl::FrontFace::CCW, true, "glCullFace(BACK), glFrontFace(CCW): the upward ground is front-facing and drawn"},
                                  {mgl::CullFace::Front, mgl::FrontFace::CCW, false, "glCullFace(FRONT), glFrontFace(CCW): the front-facing ground is culled (clear colour shows)"},
                                  {mgl::CullFace::Back, mgl::FrontFace::CW, false, "glCullFace(BACK), glFrontFace(CW): the winding is now back-facing, the ground is culled"},
                                  {mgl::CullFace::Front, mgl::FrontFace::CW, true, "glCullFace(FRONT), glFrontFace(CW): the ground counts as back-facing and is drawn"}};
            for (const Case& c : cases)
            {
                DrawState st;
                st.cull = c.cull;
                st.front = c.front;
                const Img img = snapshot(f_all, st);
                expect_probe(check, img, tile, c.visible ? ref.expected(tile.hit) : clear, c.visible ? kExactTol : 1, c.what);
            }
        }

        // B7. Keyboard camera: the key handler changes the camera, and the projected lamp moves to the CPU-predicted pixel.
        {
            auto press = [&](Params start, std::string_view keys) {
                const Params saved = params_;
                params_ = start;
                for (const char k : keys) on_key(env, k);
                const Params out = params_;
                params_ = saved;
                return out;
            };
            const Params moved = press(base, "dddddd" "ww" "qq");
            check.expect(std::abs(moved.yaw - 0.48f) < 1e-5f && std::abs(moved.distance - 10.0f) < 1e-5f && std::abs(moved.elevation - 37.0f) < 1e-5f,
                         "keys d, w, q orbit by 0.08 rad per press, dolly by 0.5 and raise by 3 degrees",
                         std::format("yaw {} distance {} elevation {}", moved.yaw, moved.distance, moved.elevation));
            const Params toggled = press(base, "123bng ");
            check.expect(!toggled.dir_on && !toggled.point_on && !toggled.spot_on && !toggled.blinn && toggled.naive_fixed && !toggled.gouraud && toggled.paused,
                         "keys 1 2 3 b n g and SPACE toggle the directional/point/spot light, Phong/Blinn-Phong, the normal matrix, Gouraud shading and the pause");
            check.expect(press(moved, "r").yaw == 0.0f && press(moved, "r").distance == kDefaultDistance, "key r resets the camera");

            const Frame fm = frame_of(moved);
            const Reference ref(fm, W, H);
            const Img img = snapshot(fm);
            const Reference ref_base(f_all, W, H);
            run_sweep("moved camera", ref, img);
            const Probe lamp = make_probe(ref, W, H, d3(fm.lights.point_position), Obj::PointLamp);
            expect_probe(check, img, lamp, encode_color(d3(kPointLampMat.emissive)), 1, "camera moved by the keys: the point lamp is at the newly CPU-projected pixel");
            double ax = 0, ay = 0, nx = 0, ny = 0;
            const bool both = ref_base.project(d3(f_all.lights.point_position), ax, ay) && ref.project(d3(fm.lights.point_position), nx, ny);
            check.expect(both && std::hypot(nx - ax, ny - ay) > 8.0, "camera moved by the keys: the lamp's projected pixel moved by more than 8 pixels",
                         std::format("({:.1f}, {:.1f}) -> ({:.1f}, {:.1f})", ax, ay, nx, ny));
        }

        // The capture shows whatever is on screen when validate() returns: leave the canonical frame there.
        draw(f_all, W, H);
        check.expect_no_gl_error("no GL error during validation");
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-lighting",
        .title = "Lighting: directional, point, spot; Phong vs Blinn-Phong",
        .min_es = {3, 0},
        .summary = "A lit 3D scene: ambient + directional + attenuated point + smooth-cone spot light, Phong vs "
                   "Blinn-Phong, a correct vs a naive normal matrix on a non-uniformly scaled object and Gouraud vs "
                   "per-fragment shading; pixels are checked against a CPU evaluation of the same equations.",
        .controls = "SPACE pause animation | 1/2/3 toggle directional/point/spot light | b Phong<->Blinn-Phong for the "
                    "ground, crystals and coarse spheres | n use the correct normal matrix on the right crystal | g per-fragment "
                    "shading on the left coarse sphere | a/d orbit camera | w/s dolly | q/e raise/lower camera | r reset | ESC quit",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Lighting>(argc, argv, info);
}
