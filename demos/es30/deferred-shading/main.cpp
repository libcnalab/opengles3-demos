// es30-deferred-shading
//
// Deferred shading with 48 moving coloured point lights (OpenGL ES 3.0, GLSL ES 3.00).
//
//   1. Geometry pass: the whole scene (a tiled floor, boxes, spheres, pillars, a torus and one glowing marker sphere
//      per light, all generated in this file) is rasterized ONCE into a G-buffer, a framebuffer object with three
//      colour attachments written by one fragment shader (multiple render targets, glDrawBuffers) and a depth TEXTURE:
//        COLOR_ATTACHMENT0  RGBA8     albedo.rgb + specular strength
//        COLOR_ATTACHMENT1  RGB10_A2  world-space normal, 10 bits per component
//        COLOR_ATTACHMENT2  RG8       shininess + emissive scale
//        DEPTH_ATTACHMENT   DEPTH_COMPONENT24 texture
//      Only RGBA8, RGB10_A2 and RG8 (and the other formats marked colour-renderable in ES 3.0 Table 3.13) can be
//      rendered to in core ES 3.0; RGBA16F is texture-only there, which is why the normal is stored as 10-bit fixed point.
//   2. Lighting pass: the G-buffer planes are sampled with texelFetch; the world position of each pixel is rebuilt from
//      the depth texture and the lights are accumulated into an RGB10_A2 light buffer either
//        * by one full-screen triangle that loops over all lights, or
//        * by one low-resolution sphere per light (light volume, glDrawElementsInstanced, front faces culled) whose
//          fragments add that light's contribution with glBlendFunc(ONE, ONE); pixels outside every volume cost nothing.
//   3. Composite: tone mapping and gamma into the default framebuffer, or one G-buffer plane as a debug view.
//
// A forward renderer with the same scene, lights and equations is built into the demo (key F, and the validation):
// it is the reference the deferred image is compared with.
//
// Validation (headless frame 29, t = 29/60 s): the three G-buffer planes and the depth are read back and compared with
// ray-plane / ray-box / ray-sphere / ray-cylinder hits computed on the CPU; the world position rebuilt from depth is
// compared with the analytic hit; final pixels are compared with a CPU evaluation of the lighting equations; the
// whole deferred image is compared with the forward image (both light-pass variants); and controls prove the checks can
// fail (one light disabled, additive blending disabled, normal plane overwritten).

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
namespace math = demo::math;

namespace
{
    // =============================================================================================================
    // Constants shared by the GPU programs and the CPU model
    // =============================================================================================================
    constexpr int kMaxLights = 64;                                   // size of the uniform block arrays (shaders)
    constexpr std::array<int, 5> kLightCounts = {8, 16, 32, 48, 64};  // selectable with +/-
    constexpr std::size_t kDefaultLightCount = 3;                     // index into kLightCounts: 48 lights
    constexpr double kPi = std::numbers::pi;
    constexpr double kNear = 0.5;
    constexpr double kFar = 80.0;
    constexpr double kFovY = 55.0 * kPi / 180.0;
    constexpr double kExposure = 1.15;       // composite.frag: display = (1 - exp(-radiance * exposure)) ^ (1 / 2.2)
    constexpr double kLightIntensity = 1.1;
    constexpr double kMarkerRadius = 0.25;
    constexpr double kMarkerEmissive = 1.4;  // emissive scale of the light markers (stored as 1.4 / 3 in RG8)
    constexpr double kFloorHalf = 12.0;
    constexpr double kTorusMinor = 0.2;      // tube radius of the unit torus mesh
    constexpr int kFramesPerSecond = 60;

    // Tolerances of the validation (each is justified in README.md).
    constexpr double kUnormTol = 1.0;                 // 8-bit conversion of a shader output
    constexpr double kNormalTolPlanar = 0.01;         // 10-bit storage + 8-bit readback of n * 0.5 + 0.5
    constexpr double kNormalTolCurved = 0.03;         // ... plus tessellation of spheres and cylinders
    constexpr double kDepthTol = 0.002;               // world units of eye distance
    constexpr double kDepthTolCurvedPerRadius = 0.006;  // facet sag of the 48-segment meshes
    constexpr double kPositionTolPlanar = 2.0;        // 1/255 of a world unit
    constexpr double kPositionTolCurved = 3.0;

    // =============================================================================================================
    // Small double precision vector (the CPU model works in double; GPU values are converted at the edge)
    // =============================================================================================================
    struct V3
    {
        double x = 0.0, y = 0.0, z = 0.0;
    };
    constexpr V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    constexpr V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    constexpr V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    constexpr V3 hadamard(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
    constexpr double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    constexpr V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    inline double length(V3 v) { return std::sqrt(dot(v, v)); }
    inline V3 normalize(V3 v)
    {
        const double len = length(v);
        return len > 0.0 ? v * (1.0 / len) : v;
    }
    inline V3 rot_y(V3 v, double angle)  // right-handed rotation about +y, same as math::rotate(angle, {0,1,0})
    {
        const double c = std::cos(angle), s = std::sin(angle);
        return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
    }
    inline math::Vec3 to_float(V3 v)
    {
        return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
    }
    inline double frac(double v) { return v - std::floor(v); }

    V3 hsv_to_rgb(double h, double s, double v)
    {
        const double k = frac(h) * 6.0;
        const int sector = static_cast<int>(std::floor(k));
        const double f = k - sector;
        const double p = v * (1.0 - s), q = v * (1.0 - s * f), t = v * (1.0 - s * (1.0 - f));
        switch (sector % 6)
        {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
        }
    }

    /// Tiny deterministic generator so the scene never depends on the standard library's distributions.
    struct Lcg
    {
        std::uint32_t state;
        double next()
        {
            state = state * 1664525u + 1013904223u;
            return static_cast<double>(state >> 8) / 16777216.0;
        }
    };

    // =============================================================================================================
    // Lights (CPU model; uploaded to the uniform block every frame)
    // =============================================================================================================
    struct Light
    {
        V3 pos;
        double radius = 1.0;
        V3 color;  // colour * intensity
    };

    /// Light i at time t: orbits spread with golden-ratio sequences, bobbing in height, one colour per light.
    Light light_at(int i, double t)
    {
        const double di = static_cast<double>(i);
        const double hue = frac(di * 0.6180339887498949);
        const double rho = 2.0 + 8.0 * frac(di * 0.7548776662466927 + 0.25);
        const double speed = (0.18 + 0.22 * frac(di * 0.5698402909980532)) * ((i & 1) != 0 ? 1.0 : -1.0);
        const double angle = di * 2.399963229728653 + speed * t;
        const double height = 1.0 + 1.6 * frac(di * 0.3247179572447460 + 0.5) + 0.35 * std::sin(1.7 * t + di);
        Light l;
        l.pos = {rho * std::cos(angle), height, rho * std::sin(angle)};
        l.radius = 4.0 + 1.5 * frac(di * 0.4142135623730951);
        l.color = hsv_to_rgb(hue, 0.90, 1.0) * kLightIntensity;
        return l;
    }

    std::vector<Light> lights_at(double t, int count)
    {
        std::vector<Light> lights;
        lights.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) lights.push_back(light_at(i, t));
        return lights;
    }

    // std140 layout of the `Lights` uniform block (ES 3.0 2.12.6.4): two vec4[64] arrays, array stride 16 bytes.
    struct LightBlock
    {
        std::array<std::array<float, 4>, kMaxLights> pos_radius{};
        std::array<std::array<float, 4>, kMaxLights> color{};
    };
    static_assert(sizeof(LightBlock) == 2u * kMaxLights * 16u);

    // =============================================================================================================
    // Scene
    // =============================================================================================================
    struct Material
    {
        V3 albedo;
        V3 albedo2;  // second checker colour (also the colour of the floor slab's sides)
        double specular = 0.0;
        double shininess = 1.0;
        double emissive = 0.0;
        bool checker = false;
    };

    enum class Shape { Box, Sphere, Cylinder, Torus };

    struct Object
    {
        Shape shape = Shape::Box;
        V3 center;      // box / sphere / torus: centre; cylinder: centre of the bottom cap
        V3 size;        // box: half extents; sphere: radius (x); cylinder: (radius, height, radius); torus: scale (x)
        double yaw = 0.0;
        Material material;
        bool marker = false;
    };

    std::vector<Object> build_scene()
    {
        std::vector<Object> scene;
        Lcg rng{7u};

        auto add = [&](Shape shape, V3 center, V3 size, double yaw, Material m) {
            Object o;
            o.shape = shape;
            o.center = center;
            o.size = size;
            o.yaw = yaw;
            o.material = m;
            scene.push_back(o);
        };

        // Floor: a slab whose top face is the plane y = 0 (the sides use the second colour).
        add(Shape::Box, {0.0, -0.5, 0.0}, {kFloorHalf, 0.5, kFloorHalf}, 0.0,
            {{0.42, 0.44, 0.50}, {0.12, 0.12, 0.15}, 0.30, 40.0, 0.0, true});

        // Pillars on a ring.
        for (int k = 0; k < 10; ++k)
        {
            const double a = 2.0 * kPi * k / 10.0 + 0.15;
            add(Shape::Cylinder, {10.4 * std::cos(a), 0.0, 10.4 * std::sin(a)}, {0.42, 3.8, 0.42}, 0.0,
                {{0.66, 0.60, 0.52}, {}, 0.25, 24.0, 0.0, false});
        }

        // Boxes on an outer ring, random size and yaw.
        const std::array<V3, 6> pastel = {{{0.80, 0.35, 0.30}, {0.30, 0.60, 0.80}, {0.85, 0.75, 0.30},
                                           {0.40, 0.75, 0.45}, {0.70, 0.45, 0.80}, {0.85, 0.55, 0.25}}};
        for (int k = 0; k < 8; ++k)
        {
            const double a = 2.0 * kPi * k / 8.0 + 0.3;
            const double r = 7.2 + 0.6 * (rng.next() * 2.0 - 1.0);
            const V3 half = {0.7 + 0.35 * rng.next(), 0.6 + 1.1 * rng.next(), 0.7 + 0.35 * rng.next()};
            add(Shape::Box, {r * std::cos(a), half.y, r * std::sin(a)}, half, rng.next() * kPi * 0.5,
                {pastel[static_cast<std::size_t>(k) % pastel.size()], {}, 0.45, 32.0, 0.0, false});
        }

        // Small cubes on a middle ring.
        for (int k = 0; k < 6; ++k)
        {
            const double a = 2.0 * kPi * k / 6.0 + 0.9;
            add(Shape::Box, {5.6 * std::cos(a), 0.35, 5.6 * std::sin(a)}, {0.35, 0.35, 0.35}, rng.next() * kPi * 0.5,
                {pastel[static_cast<std::size_t>(k + 2) % pastel.size()], {}, 0.45, 32.0, 0.0, false});
        }

        // Glossy spheres on an inner ring.
        const std::array<V3, 6> glossy = {{{0.90, 0.20, 0.25}, {0.20, 0.55, 0.95}, {0.95, 0.85, 0.25},
                                           {0.25, 0.85, 0.50}, {0.85, 0.35, 0.85}, {0.95, 0.95, 0.95}}};
        for (int k = 0; k < 6; ++k)
        {
            const double a = 2.0 * kPi * k / 6.0 + 0.2;
            const double radius = 0.55 + 0.3 * rng.next();
            add(Shape::Sphere, {4.4 * std::cos(a), radius, 4.4 * std::sin(a)}, {radius, radius, radius}, 0.0,
                {glossy[static_cast<std::size_t>(k)], {}, 0.90, 96.0, 0.0, false});
        }

        // Centre piece: a silver sphere inside a golden torus.
        add(Shape::Sphere, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, 0.0, {{0.85, 0.85, 0.90}, {}, 1.00, 128.0, 0.0, false});
        add(Shape::Torus, {0.0, 1.9 * kTorusMinor, 0.0}, {1.9, 1.9, 1.9}, 0.0, {{0.95, 0.72, 0.30}, {}, 1.00, 128.0, 0.0, false});
        return scene;
    }

    /// The scene plus one emissive marker sphere per light (these move with the lights).
    std::vector<Object> frame_objects(const std::vector<Object>& scene, const std::vector<Light>& lights)
    {
        std::vector<Object> objects = scene;
        for (const Light& l : lights)
        {
            Object o;
            o.shape = Shape::Sphere;
            o.center = l.pos;
            o.size = {kMarkerRadius, kMarkerRadius, kMarkerRadius};
            const double peak = std::max({l.color.x, l.color.y, l.color.z});
            o.material = {l.color * (1.0 / peak), {}, 0.0, 1.0, kMarkerEmissive, false};
            o.marker = true;
            objects.push_back(o);
        }
        return objects;
    }

    math::Mat4 model_matrix(const Object& o)
    {
        return math::translate(to_float(o.center)) * math::rotate(static_cast<float>(o.yaw), {0.0f, 1.0f, 0.0f}) *
               math::scale(to_float(o.size));
    }

    // =============================================================================================================
    // Camera
    // =============================================================================================================
    struct Camera
    {
        V3 eye, forward, right, up;
        double tan_half = 0.0;  // tan(fovy / 2)
        double aspect = 1.0;
        math::Mat4 view_proj;
    };

    Camera make_camera(double yaw, double distance, double aspect)
    {
        constexpr double pitch = 0.56;
        const V3 target = {0.0, 1.0, 0.0};
        Camera c;
        c.eye = target + V3{distance * std::cos(pitch) * std::sin(yaw), distance * std::sin(pitch),
                            distance * std::cos(pitch) * std::cos(yaw)};
        c.forward = normalize(target - c.eye);
        c.right = normalize(cross(c.forward, {0.0, 1.0, 0.0}));
        c.up = cross(c.right, c.forward);
        c.tan_half = std::tan(kFovY * 0.5);
        c.aspect = aspect;
        c.view_proj = math::perspective(static_cast<float>(kFovY), static_cast<float>(aspect), static_cast<float>(kNear),
                                        static_cast<float>(kFar)) *
                      math::look_at(to_float(c.eye), to_float(target), {0.0f, 1.0f, 0.0f});
        return c;
    }

    /// Unit direction of the view ray through the centre of pixel (px, py) of a w x h image (GL: row 0 at the bottom).
    V3 pixel_direction(const Camera& c, int px, int py, int w, int h)
    {
        const double nx = (px + 0.5) / w * 2.0 - 1.0;
        const double ny = (py + 0.5) / h * 2.0 - 1.0;
        return normalize(c.forward + c.right * (c.tan_half * c.aspect * nx) + c.up * (c.tan_half * ny));
    }

    /// Eye-space distance along the view axis for a window-space depth value (the inverse of the projection).
    double linear_depth(double window_depth)
    {
        return 2.0 * kFar * kNear / ((kFar + kNear) - (window_depth * 2.0 - 1.0) * (kFar - kNear));
    }

    // =============================================================================================================
    // Meshes
    // =============================================================================================================
    struct Vertex
    {
        float px, py, pz, nx, ny, nz;
    };

    struct MeshRange
    {
        std::uint32_t first_index = 0;
        std::uint32_t index_count = 0;
    };

    struct MeshData
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;

        std::uint16_t add_vertex(V3 p, V3 n)
        {
            vertices.push_back({static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z),
                                static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z)});
            return static_cast<std::uint16_t>(vertices.size() - 1);
        }
        void add_triangle(std::uint16_t a, std::uint16_t b, std::uint16_t c)
        {
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(c);
        }
    };

    /// UV sphere of radius 1: ring i from the +y pole, segment j around y. Triangles are counter-clockwise from outside.
    MeshRange add_sphere(MeshData& m, int segments, int rings)
    {
        MeshRange range{static_cast<std::uint32_t>(m.indices.size()), 0};
        std::vector<std::uint16_t> id;
        for (int i = 0; i <= rings; ++i)
        {
            const double theta = kPi * i / rings;
            for (int j = 0; j <= segments; ++j)
            {
                const double phi = 2.0 * kPi * j / segments;
                const V3 p = {std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
                id.push_back(m.add_vertex(p, p));
            }
        }
        const std::size_t stride = static_cast<std::size_t>(segments) + 1u;
        for (std::size_t i = 0; i < static_cast<std::size_t>(rings); ++i)
            for (std::size_t j = 0; j < static_cast<std::size_t>(segments); ++j)
            {
                const std::uint16_t a = id[i * stride + j], b = id[i * stride + j + 1];
                const std::uint16_t c = id[(i + 1) * stride + j], d = id[(i + 1) * stride + j + 1];
                m.add_triangle(a, b, c);
                m.add_triangle(b, d, c);
            }
        range.index_count = static_cast<std::uint32_t>(m.indices.size()) - range.first_index;
        return range;
    }

    /// Cube [-1, 1]^3 with flat normals: 6 faces of 4 vertices. Face = axis * 2 + (normal positive ? 1 : 0).
    MeshRange add_box(MeshData& m)
    {
        MeshRange range{static_cast<std::uint32_t>(m.indices.size()), 0};
        for (int face = 0; face < 6; ++face)
        {
            const int axis = face / 2;
            const double sign = (face % 2) != 0 ? 1.0 : -1.0;
            V3 n{};
            (axis == 0 ? n.x : axis == 1 ? n.y : n.z) = sign;
            const V3 u = axis == 1 ? V3{1.0, 0.0, 0.0} : V3{0.0, 1.0, 0.0};
            const V3 v = cross(n, u);  // u x v = n, so (-u-v, +u-v, +u+v, -u+v) is counter-clockwise seen from outside
            const std::uint16_t i0 = m.add_vertex(n - u - v, n);
            const std::uint16_t i1 = m.add_vertex(n + u - v, n);
            const std::uint16_t i2 = m.add_vertex(n + u + v, n);
            const std::uint16_t i3 = m.add_vertex(n - u + v, n);
            m.add_triangle(i0, i1, i2);
            m.add_triangle(i0, i2, i3);
        }
        range.index_count = static_cast<std::uint32_t>(m.indices.size()) - range.first_index;
        return range;
    }

    /// Cylinder of radius 1 around the y axis, y in [0, 1], smooth side normals, flat caps.
    MeshRange add_cylinder(MeshData& m, int segments)
    {
        MeshRange range{static_cast<std::uint32_t>(m.indices.size()), 0};
        std::vector<std::uint16_t> bottom, top;
        for (int j = 0; j <= segments; ++j)
        {
            const double phi = 2.0 * kPi * j / segments;
            const V3 n = {std::cos(phi), 0.0, std::sin(phi)};
            bottom.push_back(m.add_vertex(n, n));
            top.push_back(m.add_vertex(n + V3{0.0, 1.0, 0.0}, n));
        }
        for (std::size_t j = 0; j < static_cast<std::size_t>(segments); ++j)
        {
            m.add_triangle(bottom[j], top[j], bottom[j + 1]);
            m.add_triangle(bottom[j + 1], top[j], top[j + 1]);
        }
        for (const double y : {0.0, 1.0})
        {
            const V3 n = {0.0, y > 0.5 ? 1.0 : -1.0, 0.0};
            const std::uint16_t centre = m.add_vertex({0.0, y, 0.0}, n);
            std::vector<std::uint16_t> ring;
            for (int j = 0; j <= segments; ++j)
            {
                const double phi = 2.0 * kPi * j / segments;
                ring.push_back(m.add_vertex({std::cos(phi), y, std::sin(phi)}, n));
            }
            for (std::size_t j = 0; j < static_cast<std::size_t>(segments); ++j)
            {
                if (y > 0.5) m.add_triangle(centre, ring[j + 1], ring[j]);
                else m.add_triangle(centre, ring[j], ring[j + 1]);
            }
        }
        range.index_count = static_cast<std::uint32_t>(m.indices.size()) - range.first_index;
        return range;
    }

    /// Torus around the y axis: major radius 1, tube radius kTorusMinor.
    MeshRange add_torus(MeshData& m, int major, int minor)
    {
        MeshRange range{static_cast<std::uint32_t>(m.indices.size()), 0};
        std::vector<std::uint16_t> id;
        for (int i = 0; i <= major; ++i)
        {
            const double phi = 2.0 * kPi * i / major;
            for (int j = 0; j <= minor; ++j)
            {
                const double theta = 2.0 * kPi * j / minor;
                const V3 n = {std::cos(theta) * std::cos(phi), std::sin(theta), std::cos(theta) * std::sin(phi)};
                const double ring = 1.0 + kTorusMinor * std::cos(theta);
                id.push_back(m.add_vertex({ring * std::cos(phi), kTorusMinor * std::sin(theta), ring * std::sin(phi)}, n));
            }
        }
        const std::size_t stride = static_cast<std::size_t>(minor) + 1u;
        for (std::size_t i = 0; i < static_cast<std::size_t>(major); ++i)
            for (std::size_t j = 0; j < static_cast<std::size_t>(minor); ++j)
            {
                const std::uint16_t a = id[i * stride + j], p_theta = id[i * stride + j + 1];
                const std::uint16_t p_phi = id[(i + 1) * stride + j], d = id[(i + 1) * stride + j + 1];
                m.add_triangle(a, p_theta, p_phi);
                m.add_triangle(p_phi, p_theta, d);
            }
        range.index_count = static_cast<std::uint32_t>(m.indices.size()) - range.first_index;
        return range;
    }

    // =============================================================================================================
    // Ray casting against the analytic scene (the geometry the meshes approximate or reproduce exactly)
    // =============================================================================================================
    struct Hit
    {
        bool hit = false;
        double t = std::numeric_limits<double>::infinity();
        V3 pos, normal;
        int object = -1;
        int face = 0;
    };

    bool ray_sphere(V3 o, V3 d, V3 c, double r, double& t)
    {
        const V3 oc = o - c;
        const double b = dot(oc, d), cc = dot(oc, oc) - r * r;
        const double disc = b * b - cc;
        if (disc < 0.0) return false;
        const double s = std::sqrt(disc);
        if (-b - s > 1e-9) { t = -b - s; return true; }
        if (-b + s > 1e-9) { t = -b + s; return true; }
        return false;
    }

    bool ray_box(V3 o, V3 d, const Object& obj, double& t, V3& normal, int& face)
    {
        const V3 lo = rot_y(o - obj.center, -obj.yaw), ld = rot_y(d, -obj.yaw);
        const double origin[3] = {lo.x, lo.y, lo.z}, dir[3] = {ld.x, ld.y, ld.z};
        const double half[3] = {obj.size.x, obj.size.y, obj.size.z};
        double t_enter = -std::numeric_limits<double>::infinity(), t_exit = std::numeric_limits<double>::infinity();
        int enter_axis = 0;
        bool enter_positive = false;
        for (int a = 0; a < 3; ++a)
        {
            if (std::abs(dir[a]) < 1e-12)
            {
                if (std::abs(origin[a]) > half[a]) return false;
                continue;
            }
            const double t1 = (-half[a] - origin[a]) / dir[a], t2 = (half[a] - origin[a]) / dir[a];
            const double tn = std::min(t1, t2), tf = std::max(t1, t2);
            if (tn > t_enter)
            {
                t_enter = tn;
                enter_axis = a;
                enter_positive = dir[a] < 0.0;  // entering through the face the ray travels towards
            }
            t_exit = std::min(t_exit, tf);
        }
        if (t_enter > t_exit || t_enter < 1e-9) return false;
        t = t_enter;
        V3 n{};
        (enter_axis == 0 ? n.x : enter_axis == 1 ? n.y : n.z) = enter_positive ? 1.0 : -1.0;
        normal = rot_y(n, obj.yaw);
        face = enter_axis * 2 + (enter_positive ? 1 : 0);
        return true;
    }

    bool ray_cylinder(V3 o, V3 d, const Object& obj, double& t, V3& normal, int& face)
    {
        const double radius = obj.size.x, height = obj.size.y;
        const V3 lo = o - obj.center;
        double best = std::numeric_limits<double>::infinity();
        // Side: x^2 + z^2 = r^2 for y in [0, height].
        const double a = d.x * d.x + d.z * d.z;
        if (a > 1e-12)
        {
            const double b = lo.x * d.x + lo.z * d.z, c = lo.x * lo.x + lo.z * lo.z - radius * radius;
            const double disc = b * b - a * c;
            if (disc >= 0.0)
            {
                const double ts = (-b - std::sqrt(disc)) / a;
                const double y = lo.y + ts * d.y;
                if (ts > 1e-9 && y >= 0.0 && y <= height && ts < best)
                {
                    best = ts;
                    normal = normalize({lo.x + ts * d.x, 0.0, lo.z + ts * d.z});
                    face = 0;
                }
            }
        }
        // Caps: planes y = height (face 1) and y = 0 (face 2).
        if (std::abs(d.y) > 1e-12)
            for (int cap = 1; cap <= 2; ++cap)
            {
                const double plane = cap == 1 ? height : 0.0;
                const double tc = (plane - lo.y) / d.y;
                const double x = lo.x + tc * d.x, z = lo.z + tc * d.z;
                if (tc > 1e-9 && x * x + z * z <= radius * radius && tc < best)
                {
                    best = tc;
                    normal = {0.0, cap == 1 ? 1.0 : -1.0, 0.0};
                    face = cap;
                }
            }
        if (best == std::numeric_limits<double>::infinity()) return false;
        t = best;
        return true;
    }

    /// Torus around the y axis (major radius = scale, tube radius = scale * kTorusMinor), found by sphere tracing its exact
    /// distance function: d(p) = |(|p.xz| - R, p.y)| - r. Every step is at most the distance to the surface, so it cannot overshoot.
    bool ray_torus(V3 o, V3 d, const Object& obj, double& t, V3& normal)
    {
        const double major = obj.size.x, minor = obj.size.x * kTorusMinor;
        double t_bound = 0.0;
        if (!ray_sphere(o, d, obj.center, major + minor, t_bound)) return false;
        auto distance = [&](V3 p) {
            const V3 q = p - obj.center;
            const double ring = std::sqrt(q.x * q.x + q.z * q.z) - major;
            return std::sqrt(ring * ring + q.y * q.y) - minor;
        };
        double ray_t = std::max(t_bound, 0.0);
        const double t_end = ray_t + 2.0 * (major + minor);
        for (int i = 0; i < 400 && ray_t < t_end; ++i)
        {
            const double dist = distance(o + d * ray_t);
            if (dist < 1e-9)
            {
                const V3 q = o + d * ray_t - obj.center;
                const double rho = std::sqrt(q.x * q.x + q.z * q.z), ring = rho - major;
                const double len = std::sqrt(ring * ring + q.y * q.y);
                normal = normalize({ring / len * q.x / rho, q.y / len, ring / len * q.z / rho});
                t = ray_t;
                return true;
            }
            ray_t += dist;
        }
        return false;
    }

    Hit cast_ray(const std::vector<Object>& objects, V3 origin, V3 dir)
    {
        Hit best;
        for (std::size_t i = 0; i < objects.size(); ++i)
        {
            const Object& o = objects[i];
            double t = 0.0;
            V3 n{};
            int face = 0;
            bool ok = false;
            switch (o.shape)
            {
            case Shape::Box: ok = ray_box(origin, dir, o, t, n, face); break;
            case Shape::Sphere:
                ok = ray_sphere(origin, dir, o.center, o.size.x, t);
                if (ok) n = normalize(origin + dir * t - o.center);
                break;
            case Shape::Cylinder: ok = ray_cylinder(origin, dir, o, t, n, face); break;
            case Shape::Torus: ok = ray_torus(origin, dir, o, t, n); break;
            }
            if (ok && t < best.t)
            {
                best.t = t;
                best.normal = n;
                best.object = static_cast<int>(i);
                best.face = face;
            }
        }
        best.hit = best.object >= 0;
        best.pos = origin + dir * best.t;
        return best;
    }

    // =============================================================================================================
    // CPU lighting: the same equations as light_fullscreen.frag / volume.frag / forward.frag, in double precision
    // =============================================================================================================
    V3 albedo_at(const Material& m, V3 p, V3 n)
    {
        if (!m.checker) return m.albedo;
        const long cell = static_cast<long>(std::floor(p.x * 0.5)) + static_cast<long>(std::floor(p.z * 0.5));
        return (n.y > 0.5 && (cell & 1L) == 0) ? m.albedo : m.albedo2;
    }

    V3 ambient_term(V3 n)
    {
        const double t = n.y * 0.5 + 0.5;
        const V3 lo = {0.020, 0.020, 0.028}, hi = {0.060, 0.070, 0.100};
        return lo * (1.0 - t) + hi * t;
    }

    struct Terms
    {
        V3 diffuse, specular;
    };

    Terms light_terms(const Light& l, V3 p, V3 n, V3 v, V3 albedo, double spec, double shininess)
    {
        const V3 to_light = l.pos - p;
        const double d2 = dot(to_light, to_light);
        const double x = 1.0 - d2 / (l.radius * l.radius);
        if (x <= 0.0) return {};
        const double att = x * x;
        const V3 ld = to_light * (1.0 / std::sqrt(std::max(d2, 1e-12)));
        const double ndl = dot(n, ld);
        if (ndl <= 0.0) return {};
        const double ndh = std::max(dot(n, normalize(ld + v)), 0.0);
        const double s = spec * std::pow(ndh, shininess);
        return {hadamard(l.color, albedo) * (att * ndl), l.color * (att * s)};
    }

    struct Radiance
    {
        V3 base;      // ambient + emissive
        V3 diffuse;   // sum over lights
        V3 specular;  // sum over lights
        int lit_by = 0;  // number of lights with a non-zero contribution
        V3 total() const { return base + diffuse + specular; }
    };

    Radiance shade(const std::vector<Light>& lights, V3 p, V3 n, V3 eye, V3 albedo, double spec, double shininess, double emissive)
    {
        Radiance r;
        r.base = hadamard(albedo, ambient_term(n)) + albedo * emissive;
        const V3 v = normalize(eye - p);
        for (const Light& light : lights)
        {
            const Terms t = light_terms(light, p, n, v, albedo, spec, shininess);
            r.diffuse = r.diffuse + t.diffuse;
            r.specular = r.specular + t.specular;
            if (t.diffuse.x + t.diffuse.y + t.diffuse.z + t.specular.x + t.specular.y + t.specular.z > 0.0) ++r.lit_by;
        }
        return r;
    }

    /// composite.frag, mode 0: the light buffer holds radiance / 2 clamped to [0, 1]; then 1 - exp(-L * exposure) and gamma.
    V3 display_color(V3 radiance)
    {
        auto one = [](double l) {
            const double stored = std::min(std::max(l, 0.0), 2.0);
            return std::pow(1.0 - std::exp(-stored * kExposure), 1.0 / 2.2);
        };
        return {one(radiance.x), one(radiance.y), one(radiance.z)};
    }

    V3 sky_color(int py, int h)
    {
        const double t = (py + 0.5) / h;
        const V3 lo = {0.014, 0.018, 0.036}, hi = {0.050, 0.070, 0.140};
        return lo * (1.0 - t) + hi * t;
    }

    int to_byte(double v) { return static_cast<int>(std::lround(std::min(std::max(v, 0.0), 1.0) * 255.0)); }

    // =============================================================================================================
    // Validation model: what the CPU says every pixel must contain
    // =============================================================================================================
    struct Model
    {
        int w = 0, h = 0;
        Camera cam;
        std::vector<Light> lights;
        std::vector<Object> objects;
        std::vector<Hit> hits;  // one view ray per pixel, row 0 at the bottom (GL order)

        std::size_t index(int x, int y) const { return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x); }
        const Hit& at(int x, int y) const { return hits[index(x, y)]; }
    };

    Model build_model(const Camera& cam, std::vector<Light> lights, std::vector<Object> objects, int w, int h)
    {
        Model m;
        m.w = w;
        m.h = h;
        m.cam = cam;
        m.lights = std::move(lights);
        m.objects = std::move(objects);
        m.hits.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) m.hits[m.index(x, y)] = cast_ray(m.objects, cam.eye, pixel_direction(cam, x, y, w, h));
        return m;
    }

    long checker_cell(V3 p) { return static_cast<long>(std::floor(p.x * 0.5)) + static_cast<long>(std::floor(p.z * 0.5)); }

    /// A pixel is "stable" when it is far from silhouettes, creases and checker edges: the pixels `r` away in every
    /// direction hit the same object, the same face and the same checker cell, and the surface is not seen edge-on.
    bool is_stable(const Model& m, int x, int y, int r)
    {
        if (x < r || y < r || x >= m.w - r || y >= m.h - r) return false;
        const Hit& c = m.at(x, y);
        if (!c.hit) return false;
        if (dot(c.normal, normalize(m.cam.eye - c.pos)) < 0.35) return false;
        for (int dy = -r; dy <= r; dy += r)
            for (int dx = -r; dx <= r; dx += r)
            {
                if (dx == 0 && dy == 0) continue;
                const Hit& n = m.at(x + dx, y + dy);
                if (!n.hit || n.object != c.object || n.face != c.face) return false;
                if (m.objects[static_cast<std::size_t>(c.object)].material.checker &&
                    (checker_cell(n.pos) & 1L) != (checker_cell(c.pos) & 1L))
                    return false;
            }
        return true;
    }

    /// Everything the CPU knows about one stable pixel.
    struct Candidate
    {
        int x = 0, y = 0;
        Hit hit;
        const Object* object = nullptr;
        V3 albedo;
        double specular = 0.0, shininess = 1.0, emissive = 0.0;
        Radiance radiance;
        V3 display;
        bool curved = false;
        double radius = 0.0;

        double luminance(V3 v) const { return (v.x + v.y + v.z) / 3.0; }
        double specular_share() const { return luminance(radiance.specular) / std::max(luminance(radiance.total()), 1e-9); }
        double brightest() const { return std::max({display.x, display.y, display.z}); }
    };

    /// Stable pixels on a grid of `stride` pixels (the checks do not need every pixel); markers or scene objects only.
    std::vector<Candidate> build_candidates(const Model& m, int r, bool markers, int stride)
    {
        std::vector<Candidate> out;
        for (int y = 0; y < m.h; y += stride)
            for (int x = 0; x < m.w; x += stride)
            {
                if (!is_stable(m, x, y, r)) continue;
                if (m.objects[static_cast<std::size_t>(m.at(x, y).object)].marker != markers) continue;
                Candidate c;
                c.x = x;
                c.y = y;
                c.hit = m.at(x, y);
                c.object = &m.objects[static_cast<std::size_t>(c.hit.object)];
                const Material& mat = c.object->material;
                c.albedo = albedo_at(mat, c.hit.pos, c.hit.normal);
                c.specular = mat.specular;
                c.shininess = mat.shininess;
                c.emissive = mat.emissive;
                c.radiance = shade(m.lights, c.hit.pos, c.hit.normal, m.cam.eye, c.albedo, c.specular, c.shininess, c.emissive);
                c.display = display_color(c.radiance.total());
                c.curved = c.object->shape == Shape::Sphere || (c.object->shape == Shape::Cylinder && c.hit.face == 0);
                c.radius = c.object->size.x;
                out.push_back(c);
            }
        return out;
    }

    struct DiffStats
    {
        int max_diff = 0;          // largest absolute difference of any RGB channel (8-bit steps)
        double max_ratio = 0.0;    // largest difference / budget of any channel; <= 1 means "within the budget everywhere"
        std::size_t over = 0;      // pixels with a channel beyond its budget
        std::size_t total = 0;
        double mean = 0.0;         // mean absolute difference over all RGB channels
        int at_x = -1, at_y = -1;  // where max_ratio occurs
    };

    using BudgetMap = std::vector<std::array<float, 3>>;

    DiffStats compare_images(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b, int w, int h, const BudgetMap& budget)
    {
        DiffStats s;
        double sum = 0.0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const std::size_t pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
                const std::size_t i = pixel * 4u;
                bool beyond = false;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    const int d = std::abs(static_cast<int>(a[i + c]) - static_cast<int>(b[i + c]));
                    sum += d;
                    s.max_diff = std::max(s.max_diff, d);
                    const double ratio = d / static_cast<double>(budget[pixel][c]);
                    if (ratio > 1.0) beyond = true;
                    if (ratio > s.max_ratio)
                    {
                        s.max_ratio = ratio;
                        s.at_x = x;
                        s.at_y = y;
                    }
                }
                ++s.total;
                if (beyond) ++s.over;
            }
        s.mean = sum / (3.0 * static_cast<double>(std::max<std::size_t>(s.total, 1)));
        return s;
    }

    /// Per-pixel, per-channel tolerances (in 8-bit display steps) for comparing renderings of the same frame.
    ///
    /// Every rendering ends in the same composite pass, so renderings differ only by quantization before it. ES 3.0 2.1.6.2 defines
    /// the float -> normalized fixed-point conversion as "one of the two values closest to f (rounding to nearest preferred)": an
    /// error of up to ONE step of the format. The sources of error are:
    ///  * the RGB10_A2 light buffer: one step is 2/1023 of radiance (it stores radiance / 2). The forward and the full-screen pass
    ///    convert once; the volume path converts the running sum after the base pass and after each of the k lights that touch the
    ///    pixel (blending is computed in floating point and converted back when stored, ES 3.0 4.1.7): k + 1 conversions;
    ///  * the G-buffer (deferred renderings only): each stored input may be off by one step of its format (8-bit albedo, specular,
    ///    shininess and emissive, 10-bit normal components). The CPU re-evaluates the lighting with each input shifted by one step
    ///    (all eight sign combinations for the normal) and adds the resulting radiance changes;
    ///  * the 8-bit conversion of the displayed colour: two renderings may round the same value to different neighbours (1 step);
    ///    a rendering and the CPU model (which rounds to nearest) differ by at most 1.5 steps.
    /// A radiance error becomes a display error through f(L) = (1 - exp(-L * exposure))^(1 / 2.2), whose slope is largest in the
    /// dark, so each budget follows its pixel's own radiance. Budgets are the worst case; the measured ratios are printed.
    struct Budget
    {
        int w = 0, h = 0;
        BudgetMap full;     // full-screen deferred vs forward: 2 light-buffer conversions
        BudgetMap volumes;  // volume deferred vs forward or vs full-screen: k + 2 conversions
        BudgetMap model;    // volume deferred vs the CPU model: k + 1 conversions
    };

    Budget compute_budget(const Model& m)
    {
        constexpr double kStep = 2.0 / 1023.0;  // radiance per RGB10_A2 step of the light buffer
        auto display = [](double l) { return std::pow(1.0 - std::exp(-kExposure * std::min(std::max(l, 0.0), 2.0)), 1.0 / 2.2); };
        Budget b;
        b.w = m.w;
        b.h = m.h;
        const std::size_t count = m.hits.size();
        BudgetMap full(count, {1.0f, 1.0f, 1.0f}), vol(count, {1.0f, 1.0f, 1.0f}), mod(count, {1.5f, 1.5f, 1.5f});
        std::vector<Light> near_lights;
        for (int y = 0; y < m.h; ++y)
            for (int x = 0; x < m.w; ++x)
            {
                const Hit& hit = m.at(x, y);
                if (!hit.hit) continue;
                const Material& mat = m.objects[static_cast<std::size_t>(hit.object)].material;

                // Lights that can reach this point (the others contribute exactly 0 to every variation below).
                near_lights.clear();
                for (const Light& l : m.lights)
                {
                    const V3 d = l.pos - hit.pos;
                    if (dot(d, d) < l.radius * l.radius) near_lights.push_back(l);
                }
                struct Surface
                {
                    V3 albedo, normal;
                    double specular, shininess, emissive;
                };
                const Surface base = {albedo_at(mat, hit.pos, hit.normal), hit.normal, mat.specular, mat.shininess, mat.emissive};
                auto evaluate = [&](const Surface& s) {
                    return shade(near_lights, hit.pos, s.normal, m.cam.eye, s.albedo, s.specular, s.shininess, s.emissive);
                };
                const Radiance r0 = evaluate(base);
                const V3 l0 = r0.total();

                // Worst-case radiance change from one step of every G-buffer input.
                double d[3] = {0.0, 0.0, 0.0};
                auto add_change = [&](const Surface& s) {
                    const V3 l = evaluate(s).total();
                    d[0] += std::abs(l.x - l0.x);
                    d[1] += std::abs(l.y - l0.y);
                    d[2] += std::abs(l.z - l0.z);
                };
                Surface s = base;
                s.albedo = base.albedo + V3{1.0, 1.0, 1.0} * (1.0 / 255.0);
                add_change(s);
                s = base;
                s.specular += 1.0 / 255.0;
                add_change(s);
                s = base;
                s.shininess += 1.0;
                add_change(s);
                s = base;
                s.emissive += 3.0 / 255.0;
                add_change(s);
                double worst_normal[3] = {0.0, 0.0, 0.0};
                for (int sign = 0; sign < 8; ++sign)
                {
                    s = base;
                    s.normal = normalize(base.normal + V3{(sign & 1) != 0 ? 1.0 : -1.0, (sign & 2) != 0 ? 1.0 : -1.0, (sign & 4) != 0 ? 1.0 : -1.0} * (2.0 / 1023.0));
                    const V3 l = evaluate(s).total();
                    worst_normal[0] = std::max(worst_normal[0], std::abs(l.x - l0.x));
                    worst_normal[1] = std::max(worst_normal[1], std::abs(l.y - l0.y));
                    worst_normal[2] = std::max(worst_normal[2], std::abs(l.z - l0.z));
                }
                for (int c = 0; c < 3; ++c) d[c] += worst_normal[c];

                const double base_l[3] = {l0.x, l0.y, l0.z};
                for (std::size_t c = 0; c < 3; ++c)
                {
                    auto budget = [&](double conversions, double rounding) {
                        const double e = d[c] + conversions * kStep;
                        const double up = display(base_l[c] + e) - display(base_l[c]);
                        const double down = display(base_l[c]) - display(base_l[c] - e);
                        return static_cast<float>(rounding + 255.0 * std::max(up, down));
                    };
                    full[m.index(x, y)][c] = budget(2.0, 1.0);
                    vol[m.index(x, y)][c] = budget(r0.lit_by + 2, 1.0);
                    mod[m.index(x, y)][c] = budget(r0.lit_by + 1, 1.5);
                }
            }
        // Silhouette pixels may be covered by a neighbouring object in one rendering: take the largest budget of the 3 x 3 block.
        auto dilate = [&](const BudgetMap& in) {
            BudgetMap out = in;
            for (int y = 0; y < m.h; ++y)
                for (int x = 0; x < m.w; ++x)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            const int nx = x + dx, ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= m.w || ny >= m.h) continue;
                            for (std::size_t c = 0; c < 3; ++c) out[m.index(x, y)][c] = std::max(out[m.index(x, y)][c], in[m.index(nx, ny)][c]);
                        }
            return out;
        };
        b.full = dilate(full);
        b.volumes = dilate(vol);
        b.model = dilate(mod);
        return b;
    }

    using Pixels = std::vector<std::uint8_t>;

    std::array<int, 4> texel(const Pixels& img, int w, int x, int y)
    {
        const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4u;
        return {img[i], img[i + 1], img[i + 2], img[i + 3]};
    }

    // =============================================================================================================
    // GL helpers and GL-side state
    // =============================================================================================================
    struct GeometryUniforms
    {
        mgl::UniformLocation view_proj, model, normal_matrix, albedo, albedo2, specular, shininess, emissive, checker, eye,
            light_count;
    };

    struct CameraUniforms
    {
        mgl::UniformLocation eye, ray_f, ray_r, ray_u, clip, size;
    };

    GeometryUniforms locate_geometry(mgl::ProgramId p)
    {
        return {mgl::glGetUniformLocation(p, "u_view_proj"), mgl::glGetUniformLocation(p, "u_model"),
                mgl::glGetUniformLocation(p, "u_normal_matrix"), mgl::glGetUniformLocation(p, "u_albedo"),
                mgl::glGetUniformLocation(p, "u_albedo2"), mgl::glGetUniformLocation(p, "u_specular"),
                mgl::glGetUniformLocation(p, "u_shininess"), mgl::glGetUniformLocation(p, "u_emissive"),
                mgl::glGetUniformLocation(p, "u_checker"), mgl::glGetUniformLocation(p, "u_eye"),
                mgl::glGetUniformLocation(p, "u_light_count")};
    }

    CameraUniforms locate_camera(mgl::ProgramId p)
    {
        return {mgl::glGetUniformLocation(p, "u_eye"),   mgl::glGetUniformLocation(p, "u_ray_f"),
                mgl::glGetUniformLocation(p, "u_ray_r"), mgl::glGetUniformLocation(p, "u_ray_u"),
                mgl::glGetUniformLocation(p, "u_clip"),  mgl::glGetUniformLocation(p, "u_size")};
    }

    void set_camera(const CameraUniforms& u, const Camera& c, int w, int h)
    {
        mgl::glUniform3f(u.eye, static_cast<float>(c.eye.x), static_cast<float>(c.eye.y), static_cast<float>(c.eye.z));
        mgl::glUniform3f(u.ray_f, static_cast<float>(c.forward.x), static_cast<float>(c.forward.y), static_cast<float>(c.forward.z));
        const V3 r = c.right * (c.tan_half * c.aspect), up = c.up * c.tan_half;
        mgl::glUniform3f(u.ray_r, static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.z));
        mgl::glUniform3f(u.ray_u, static_cast<float>(up.x), static_cast<float>(up.y), static_cast<float>(up.z));
        mgl::glUniform2f(u.clip, static_cast<float>(kNear), static_cast<float>(kFar));
        mgl::glUniform2f(u.size, static_cast<float>(w), static_cast<float>(h));
    }

    /// The render targets: G-buffer (3 colour textures + depth texture), light buffer, forward-reference buffer.
    struct Targets
    {
        int w = 0, h = 0;
        mgl::TextureId albedo{}, normal{}, material{}, depth{}, light{}, ref_light{};
        mgl::RenderbufferId ref_depth{};
        mgl::FramebufferId gbuffer_fbo{}, light_fbo{}, ref_fbo{};
    };

    mgl::TextureId make_texture(mgl::SizedInternalFormat format, int w, int h)
    {
        mgl::TextureId tex{};
        mgl::glGenTextures(1, &tex);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, w, h);
        // NEAREST: the default MIN_FILTER wants mipmaps, and a depth texture whose compare mode is NONE is only
        // complete with NEAREST (ES 3.0 3.8.13). The shaders use texelFetch, which ignores filtering anyway.
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
        return tex;
    }

    enum class Variant
    {
        Volumes,        // deferred, additive light volumes
        Fullscreen,     // deferred, one full-screen lighting pass
        Forward,        // forward-rendered reference
    };

    struct FrameSpec
    {
        double t = 0.0;               // camera and (unless light_time >= 0) light animation time
        double light_time = -1.0;     // >= 0: lights are evaluated at this time instead of t
        int view = 0;                 // composite.frag mode
        Variant variant = Variant::Volumes;
        int light_count = 48;
        int disabled_light = -1;      // uniform block colour zeroed (negative control)
        bool blend = true;            // volumes: additive blending (negative control disables it)
        bool cull_volumes = true;     // volumes: cull FRONT faces (negative control: both faces are rasterized)
        bool overwrite_normals = false;  // negative control: normal plane replaced by "up" after the geometry pass
    };

    class DeferredShading final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            const int max_draw_buffers = demo::get_integer(mgl::GetParameter::MaxDrawBuffers);
            if (max_draw_buffers < 3) throw demo::Skip("needs MAX_DRAW_BUFFERS >= 3");

            gbuffer_ = demo::build_program(env, "shaders/gbuffer.vert", "shaders/gbuffer.frag");
            forward_ = demo::build_program(env, "shaders/gbuffer.vert", "shaders/forward.frag");
            light_full_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/light_fullscreen.frag");
            light_volume_ = demo::build_program(env, "shaders/volume.vert", "shaders/volume.frag");
            composite_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/composite.frag");

            gbuffer_u_ = locate_geometry(gbuffer_);
            forward_u_ = locate_geometry(forward_);
            light_full_cam_ = locate_camera(light_full_);
            light_volume_cam_ = locate_camera(light_volume_);
            composite_cam_ = locate_camera(composite_);
            light_full_count_ = mgl::glGetUniformLocation(light_full_, "u_light_count");
            volume_view_proj_ = mgl::glGetUniformLocation(light_volume_, "u_view_proj");
            volume_scale_u_ = mgl::glGetUniformLocation(light_volume_, "u_volume_scale");
            composite_mode_ = mgl::glGetUniformLocation(composite_, "u_mode");
            composite_exposure_ = mgl::glGetUniformLocation(composite_, "u_exposure");

            // Sampler uniforms hold texture UNIT indices: the G-buffer planes live on units 0..3, the light buffer on unit 4.
            const std::array<mgl::ProgramId, 3> samplers_programs = {light_full_, light_volume_, composite_};
            for (const mgl::ProgramId p : samplers_programs)
            {
                mgl::glUseProgram(p);
                mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_albedo"), 0);
                mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_normal"), 1);
                mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_material"), 2);
                mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_depth"), 3);
            }
            mgl::glUseProgram(composite_);
            mgl::glUniform1i(mgl::glGetUniformLocation(composite_, "u_light"), 4);
            mgl::glUseProgram(mgl::ProgramId{0});

            // Uniform block `Lights`: one buffer, binding point 0, shared by four programs (ES 3.0 2.12.6.3).
            for (const mgl::ProgramId p : {forward_, light_full_, light_volume_})
            {
                const GLuint block = mgl::glGetUniformBlockIndex(p, "Lights");
                mgl::glUniformBlockBinding(p, block, kLightsBindingPoint);
            }
            mgl::glGenBuffers(1, &light_ubo_);
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, light_ubo_);
            mgl::glBufferData(mgl::BufferTarget::Uniform, static_cast<GLsizeiptr>(sizeof(LightBlock)), nullptr,
                              mgl::BufferUsage::DynamicDraw);
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, kLightsBindingPoint, light_ubo_);
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, mgl::BufferId{0});

            // Meshes: all shapes share one vertex buffer, one index buffer and one VAO.
            sphere_ = add_sphere(mesh_, 48, 24);
            box_ = add_box(mesh_);
            cylinder_ = add_cylinder(mesh_, 48);
            torus_ = add_torus(mesh_, 64, 32);
            volume_ = add_sphere(mesh_, 16, 8);
            volume_scale_ = circumscribing_scale(volume_);

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(mesh_.vertices), mgl::BufferUsage::StaticDraw);
            mgl::glGenBuffers(1, &ibo_);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ibo_);  // recorded in the VAO
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(mesh_.indices),
                              mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, px)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, nx)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // A VAO without arrays for the full-screen triangle (all vertex data comes from gl_VertexID).
            mgl::glGenVertexArrays(1, &empty_vao_);

            scene_ = build_scene();
            create_targets(env.width, env.height);
        }

        void render(demo::Env& env, int frame) override
        {
            last_frame_ = frame;
            if (!paused_) anim_time_ = (static_cast<double>(frame) - time_origin_) / kFramesPerSecond;
            draw_frame(env, current_spec());
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            if (key >= '1' && key <= '7')
            {
                view_ = key - '1';
                forward_view_ = false;
            }
            else if (key == 'v') volumes_ = !volumes_;
            else if (key == 'f') forward_view_ = !forward_view_;
            else if (key == ' ')
            {
                paused_ = !paused_;
                if (!paused_) time_origin_ = static_cast<double>(last_frame_ + 1) - anim_time_ * kFramesPerSecond;
            }
            else if (key == 'a') yaw_offset_ -= 0.1;
            else if (key == 'd') yaw_offset_ += 0.1;
            else if (key == 'w') distance_ = std::max(8.0, distance_ - 1.0);
            else if (key == 's') distance_ = std::min(40.0, distance_ + 1.0);
            else if ((key == '+' || key == '=') && light_count_index_ + 1 < kLightCounts.size()) ++light_count_index_;
            else if (key == '-' && light_count_index_ > 0) --light_count_index_;
        }

        void validate(demo::Env& env, demo::Check& check) override;
        void check_configuration(demo::Check& check) const;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            destroy_targets();
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteVertexArrays(1, &empty_vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteBuffers(1, &ibo_);
            mgl::glDeleteBuffers(1, &light_ubo_);
            mgl::glDeleteProgram(gbuffer_);
            mgl::glDeleteProgram(forward_);
            mgl::glDeleteProgram(light_full_);
            mgl::glDeleteProgram(light_volume_);
            mgl::glDeleteProgram(composite_);
        }

    private:
        static constexpr GLuint kLightsBindingPoint = 0;

        // -------------------------------------------------------------------------------------------------------
        // Targets
        // -------------------------------------------------------------------------------------------------------
        void create_targets(int w, int h)
        {
            Targets& t = targets_;
            t.w = w;
            t.h = h;
            // Colour-renderable in core ES 3.0 (Table 3.13): RGBA8, RGB10_A2, RG8. Depth: DEPTH_COMPONENT24 (Table 3.14).
            t.albedo = make_texture(mgl::SizedInternalFormat::Rgba8, w, h);
            t.normal = make_texture(mgl::SizedInternalFormat::Rgb10A2, w, h);
            t.material = make_texture(mgl::SizedInternalFormat::Rg8, w, h);
            t.depth = make_texture(mgl::SizedInternalFormat::DepthComponent24, w, h);
            t.light = make_texture(mgl::SizedInternalFormat::Rgb10A2, w, h);
            t.ref_light = make_texture(mgl::SizedInternalFormat::Rgb10A2, w, h);
            mgl::glGenRenderbuffers(1, &t.ref_depth);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, t.ref_depth);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent24, w, h);

            constexpr auto fb = mgl::FramebufferTarget::Framebuffer;
            constexpr auto tex2d = mgl::TextureTarget::Texture2D;
            mgl::glGenFramebuffers(1, &t.gbuffer_fbo);
            mgl::glBindFramebuffer(fb, t.gbuffer_fbo);
            mgl::glFramebufferTexture2D(fb, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex2d, t.albedo, 0);
            mgl::glFramebufferTexture2D(fb, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1), tex2d, t.normal, 0);
            mgl::glFramebufferTexture2D(fb, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color2), tex2d, t.material, 0);
            mgl::glFramebufferTexture2D(fb, mgl::FramebufferAttachment::Depth, tex2d, t.depth, 0);
            // Fragment output i -> COLOR_ATTACHMENTi; the fourth slot stays NONE (ES 3.0 4.2.1).
            const std::array<mgl::DrawBuffer, 4> draw_buffers = {
                mgl::to_draw_buffer(mgl::ColorAttachment::Color0), mgl::to_draw_buffer(mgl::ColorAttachment::Color1),
                mgl::to_draw_buffer(mgl::ColorAttachment::Color2), mgl::DrawBuffer::None};
            mgl::glDrawBuffers(draw_buffers);
            require_complete("G-buffer framebuffer");

            mgl::glGenFramebuffers(1, &t.light_fbo);
            mgl::glBindFramebuffer(fb, t.light_fbo);
            mgl::glFramebufferTexture2D(fb, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex2d, t.light, 0);
            require_complete("light-buffer framebuffer");

            mgl::glGenFramebuffers(1, &t.ref_fbo);
            mgl::glBindFramebuffer(fb, t.ref_fbo);
            mgl::glFramebufferTexture2D(fb, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), tex2d, t.ref_light, 0);
            mgl::glFramebufferRenderbuffer(fb, mgl::FramebufferAttachment::Depth, mgl::RenderbufferTarget::Renderbuffer, t.ref_depth);
            require_complete("forward-reference framebuffer");

            mgl::glBindFramebuffer(fb, mgl::FramebufferId{0});
        }

        static void require_complete(const char* what)
        {
            const mgl::FramebufferStatus s = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (s != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{} is not complete: {}", what, demo::framebuffer_status_name(s)));
        }

        void destroy_targets()
        {
            Targets& t = targets_;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &t.gbuffer_fbo);
            mgl::glDeleteFramebuffers(1, &t.light_fbo);
            mgl::glDeleteFramebuffers(1, &t.ref_fbo);
            mgl::glDeleteRenderbuffers(1, &t.ref_depth);
            for (mgl::TextureId* tex : {&t.albedo, &t.normal, &t.material, &t.depth, &t.light, &t.ref_light})
                mgl::glDeleteTextures(1, tex);
            t = Targets{};
        }

        void ensure_targets(int w, int h)
        {
            if (targets_.w == w && targets_.h == h) return;
            destroy_targets();
            create_targets(w, h);
        }

        // -------------------------------------------------------------------------------------------------------
        // Passes
        // -------------------------------------------------------------------------------------------------------
        double aspect() const { return static_cast<double>(targets_.w) / static_cast<double>(targets_.h); }

        Camera camera_at(double t) const
        {
            return make_camera(0.55 + yaw_offset_ + 0.08 * t, distance_, aspect());
        }

        MeshRange mesh_of(Shape s) const
        {
            switch (s)
            {
            case Shape::Box: return box_;
            case Shape::Sphere: return sphere_;
            case Shape::Cylinder: return cylinder_;
            default: return torus_;
            }
        }

        void upload_lights(const std::vector<Light>& lights, int disabled_light) const
        {
            LightBlock block{};
            for (std::size_t i = 0; i < lights.size() && i < static_cast<std::size_t>(kMaxLights); ++i)
            {
                const Light& l = lights[i];
                block.pos_radius[i] = {static_cast<float>(l.pos.x), static_cast<float>(l.pos.y), static_cast<float>(l.pos.z),
                                       static_cast<float>(l.radius)};
                const double on = static_cast<int>(i) == disabled_light ? 0.0 : 1.0;
                block.color[i] = {static_cast<float>(l.color.x * on), static_cast<float>(l.color.y * on),
                                  static_cast<float>(l.color.z * on), 0.0f};
            }
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, light_ubo_);
            mgl::glBufferSubData(mgl::BufferTarget::Uniform, 0, static_cast<GLsizeiptr>(sizeof(block)), &block);
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, kLightsBindingPoint, light_ubo_);
        }

        void draw_object(const GeometryUniforms& u, const Object& o) const
        {
            const math::Mat4 model = model_matrix(o);
            mgl::glUniformMatrix4fv(u.model, 1, GL_FALSE, model.data());
            const std::array<float, 9> normal_matrix = math::normal_matrix(model);
            mgl::glUniformMatrix3fv(u.normal_matrix, 1, GL_FALSE, normal_matrix.data());
            const Material& m = o.material;
            mgl::glUniform3f(u.albedo, static_cast<float>(m.albedo.x), static_cast<float>(m.albedo.y), static_cast<float>(m.albedo.z));
            mgl::glUniform3f(u.albedo2, static_cast<float>(m.albedo2.x), static_cast<float>(m.albedo2.y), static_cast<float>(m.albedo2.z));
            mgl::glUniform1f(u.specular, static_cast<float>(m.specular));
            mgl::glUniform1f(u.shininess, static_cast<float>(m.shininess));
            mgl::glUniform1f(u.emissive, static_cast<float>(m.emissive));
            mgl::glUniform1i(u.checker, m.checker ? 1 : 0);
            const MeshRange r = mesh_of(o.shape);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, static_cast<GLsizei>(r.index_count), mgl::IndexType::UnsignedShort,
                                demo::buffer_offset(static_cast<std::uintptr_t>(r.first_index) * sizeof(std::uint16_t)));
        }

        void bind_gbuffer_textures() const
        {
            const std::array<std::pair<mgl::TextureUnit, mgl::TextureId>, 4> planes = {{
                {mgl::TextureUnit::Texture0, targets_.albedo},
                {mgl::TextureUnit::Texture1, targets_.normal},
                {mgl::TextureUnit::Texture2, targets_.material},
                {mgl::TextureUnit::Texture3, targets_.depth}}};
            for (const auto& [unit, tex] : planes)
            {
                mgl::glActiveTexture(unit);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            }
        }

        /// Pass 1: scene -> G-buffer (three colour attachments + depth texture).
        void geometry_pass(const Camera& cam, const std::vector<Object>& objects) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, targets_.gbuffer_fbo);
            mgl::glViewport(0, 0, targets_.w, targets_.h);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_TRUE);
            mgl::glEnable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);
            mgl::glFrontFace(mgl::FrontFace::CCW);
            // glClear clears every draw buffer; zero marks "no surface" in all three planes, depth 1 is the far plane.
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            mgl::glUseProgram(gbuffer_);
            mgl::glUniformMatrix4fv(gbuffer_u_.view_proj, 1, GL_FALSE, cam.view_proj.data());
            mgl::glBindVertexArray(vao_);
            for (const Object& o : objects) draw_object(gbuffer_u_, o);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
        }

        /// Pass 2a: one full-screen triangle. count = 0 writes only the ambient + emissive base.
        void light_pass_fullscreen(const Camera& cam, int count) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, targets_.light_fbo);
            mgl::glViewport(0, 0, targets_.w, targets_.h);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(light_full_);
            bind_gbuffer_textures();
            set_camera(light_full_cam_, cam, targets_.w, targets_.h);
            mgl::glUniform1i(light_full_count_, count);
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        /// Pass 2b: base pass, then one sphere per light with additive blending.
        void light_pass_volumes(const Camera& cam, int count, bool blend, bool cull) const
        {
            light_pass_fullscreen(cam, 0);

            if (blend)
            {
                mgl::glEnable(mgl::Capability::Blend);
                mgl::glBlendEquation(mgl::BlendEquation::FuncAdd);
                mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::One);  // dst = src + dst, clamped to [0, 1]
            }
            // Cull FRONT faces: the back faces of the sphere cover every pixel the volume can touch, even when the
            // camera is inside it. No depth test: the shader rejects pixels outside the light's radius by itself.
            if (cull)
            {
                mgl::glEnable(mgl::Capability::CullFace);
                mgl::glCullFace(mgl::CullFace::Front);
            }
            mgl::glUseProgram(light_volume_);
            bind_gbuffer_textures();
            mgl::glUniformMatrix4fv(volume_view_proj_, 1, GL_FALSE, cam.view_proj.data());
            mgl::glUniform1f(volume_scale_u_, static_cast<float>(volume_scale_));
            set_camera(light_volume_cam_, cam, targets_.w, targets_.h);
            mgl::glBindVertexArray(vao_);
            mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, static_cast<GLsizei>(volume_.index_count),
                                         mgl::IndexType::UnsignedShort,
                                         demo::buffer_offset(static_cast<std::uintptr_t>(volume_.first_index) * sizeof(std::uint16_t)),
                                         count);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);
        }

        /// The forward reference: scene + lights evaluated per fragment into the reference light buffer.
        void forward_pass(const Camera& cam, const std::vector<Object>& objects, int count) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, targets_.ref_fbo);
            mgl::glViewport(0, 0, targets_.w, targets_.h);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_TRUE);
            mgl::glEnable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glUseProgram(forward_);
            mgl::glUniformMatrix4fv(forward_u_.view_proj, 1, GL_FALSE, cam.view_proj.data());
            mgl::glUniform3f(forward_u_.eye, static_cast<float>(cam.eye.x), static_cast<float>(cam.eye.y), static_cast<float>(cam.eye.z));
            mgl::glUniform1i(forward_u_.light_count, count);
            mgl::glBindVertexArray(vao_);
            for (const Object& o : objects) draw_object(forward_u_, o);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
        }

        /// Pass 3: the default framebuffer shows the light buffer (mode 0) or a G-buffer plane.
        void composite(const Camera& cam, int mode, mgl::TextureId light_tex) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, targets_.w, targets_.h);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glUseProgram(composite_);
            bind_gbuffer_textures();
            mgl::glActiveTexture(mgl::TextureUnit::Texture4);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, light_tex);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            set_camera(composite_cam_, cam, targets_.w, targets_.h);
            mgl::glUniform1i(composite_mode_, mode);
            mgl::glUniform1f(composite_exposure_, static_cast<float>(kExposure));
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        /// One whole frame into the default framebuffer.
        void draw_frame(const demo::Env& env, const FrameSpec& spec)
        {
            if (env.width <= 0 || env.height <= 0) return;  // minimized window
            ensure_targets(env.width, env.height);
            mgl::glDisable(mgl::Capability::Dither);  // dithering would perturb stored values by one step
            const Camera cam = camera_at(spec.t);
            const std::vector<Light> lights = lights_at(spec.light_time >= 0.0 ? spec.light_time : spec.t, spec.light_count);
            const std::vector<Object> objects = frame_objects(scene_, lights);
            upload_lights(lights, spec.disabled_light);

            if (spec.variant == Variant::Forward)
            {
                forward_pass(cam, objects, spec.light_count);
                composite(cam, 0, targets_.ref_light);
                return;
            }
            geometry_pass(cam, objects);
            if (spec.overwrite_normals)
            {
                // Negative control: replace the normal plane (draw buffer 1) by "straight up" (encoded 0.5, 1, 0.5).
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, targets_.gbuffer_fbo);
                const std::array<float, 4> up = {0.5f, 1.0f, 0.5f, 1.0f};
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 1, up.data());
            }
            if (spec.variant == Variant::Volumes) light_pass_volumes(cam, spec.light_count, spec.blend, spec.cull_volumes);
            else light_pass_fullscreen(cam, spec.light_count);
            composite(cam, spec.view, targets_.light);
        }

        FrameSpec current_spec() const
        {
            FrameSpec spec;
            spec.t = anim_time_;
            spec.view = view_;
            spec.variant = forward_view_ ? Variant::Forward : (volumes_ ? Variant::Volumes : Variant::Fullscreen);
            spec.light_count = kLightCounts[light_count_index_];
            return spec;
        }

        /// Draws a frame and reads the default framebuffer (RGBA8, GL row order).
        Pixels render_image(const demo::Env& env, const FrameSpec& spec)
        {
            draw_frame(env, spec);
            return read_default();
        }

        Pixels read_default() const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return demo::read_rgba8(0, 0, targets_.w, targets_.h);
        }

        /// One G-buffer colour plane, read with glReadBuffer + glReadPixels(RGBA, UNSIGNED_BYTE).
        Pixels read_plane(mgl::ColorAttachment attachment) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, targets_.gbuffer_fbo);
            mgl::glReadBuffer(mgl::to_read_buffer(attachment));
            Pixels pixels = demo::read_rgba8(0, 0, targets_.w, targets_.h);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return pixels;
        }

        /// Re-runs only the composite pass (the G-buffer and light buffer of the last frame are still there).
        Pixels composite_image(const Camera& cam, int mode)
        {
            composite(cam, mode, targets_.light);
            return read_default();
        }

        /// Scale that makes the low-resolution volume sphere contain the unit sphere: its faces are planes at
        /// distance >= `inradius` from the centre, so dividing by the smallest one circumscribes the sphere.
        double circumscribing_scale(const MeshRange& r) const
        {
            double inradius = std::numeric_limits<double>::infinity();
            for (std::uint32_t i = r.first_index; i < r.first_index + r.index_count; i += 3)
            {
                const Vertex& a = mesh_.vertices[mesh_.indices[i]];
                const Vertex& b = mesh_.vertices[mesh_.indices[i + 1]];
                const Vertex& c = mesh_.vertices[mesh_.indices[i + 2]];
                const V3 pa = {a.px, a.py, a.pz}, pb = {b.px, b.py, b.pz}, pc = {c.px, c.py, c.pz};
                const V3 n = cross(pb - pa, pc - pa);
                if (length(n) < 1e-9) continue;  // degenerate triangle at a pole
                inradius = std::min(inradius, std::abs(dot(normalize(n), pa)));
            }
            return 1.0 / inradius;
        }

        // -------------------------------------------------------------------------------------------------------
        // State
        // -------------------------------------------------------------------------------------------------------
        mgl::ProgramId gbuffer_{}, forward_{}, light_full_{}, light_volume_{}, composite_{};
        GeometryUniforms gbuffer_u_{}, forward_u_{};
        CameraUniforms light_full_cam_{}, light_volume_cam_{}, composite_cam_{};
        mgl::UniformLocation light_full_count_{}, volume_view_proj_{}, volume_scale_u_{}, composite_mode_{}, composite_exposure_{};
        mgl::BufferId vbo_{}, ibo_{}, light_ubo_{};
        mgl::VertexArrayId vao_{}, empty_vao_{};
        Targets targets_;

        MeshData mesh_;
        MeshRange sphere_, box_, cylinder_, torus_, volume_;
        double volume_scale_ = 1.0;
        std::vector<Object> scene_;

        // Interactive state (defaults are what the headless run renders and validates).
        int view_ = 0;
        bool volumes_ = true;
        bool forward_view_ = false;
        bool paused_ = false;
        double yaw_offset_ = 0.0;
        double distance_ = 22.0;
        std::size_t light_count_index_ = kDefaultLightCount;
        int last_frame_ = 0;
        double anim_time_ = 0.0;
        double time_origin_ = 0.0;  // frame number at which the animation clock was 0

    };

    // =============================================================================================================
    // Validation
    // =============================================================================================================

    /// Mesh sanity on the CPU: every triangle is counter-clockwise seen from outside (its geometric normal agrees with
    /// the vertex normals; otherwise back-face culling would remove the surface), and the light-volume polyhedron is
    /// convex with all face planes at distance >= 1 after scaling, so it contains the sphere of the light's radius.
    void check_meshes(demo::Check& check, const MeshData& mesh, const MeshRange& volume, double volume_scale)
    {
        std::size_t triangles = 0, wrong = 0;
        for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
        {
            const Vertex& a = mesh.vertices[mesh.indices[i]];
            const Vertex& b = mesh.vertices[mesh.indices[i + 1]];
            const Vertex& c = mesh.vertices[mesh.indices[i + 2]];
            const V3 pa = {a.px, a.py, a.pz}, pb = {b.px, b.py, b.pz}, pc = {c.px, c.py, c.pz};
            const V3 n = cross(pb - pa, pc - pa);
            if (length(n) < 1e-9) continue;  // degenerate triangle at a pole
            const V3 na = {a.nx, a.ny, a.nz}, nb = {b.nx, b.ny, b.nz}, nc = {c.nx, c.ny, c.nz};
            ++triangles;
            if (dot(n, na + nb + nc) <= 0.0) ++wrong;
        }
        check.expect(triangles > 0 && wrong == 0, "all mesh triangles are counter-clockwise seen from outside (back-face culling keeps every surface)",
                     std::format("{} non-degenerate triangles, {} wrongly wound", triangles, wrong));

        double min_distance = std::numeric_limits<double>::infinity();
        std::size_t outside_vertices = 0;
        for (std::uint32_t i = volume.first_index; i < volume.first_index + volume.index_count; i += 3)
        {
            const Vertex& a = mesh.vertices[mesh.indices[i]];
            const Vertex& b = mesh.vertices[mesh.indices[i + 1]];
            const Vertex& c = mesh.vertices[mesh.indices[i + 2]];
            const V3 pa = {a.px, a.py, a.pz}, pb = {b.px, b.py, b.pz}, pc = {c.px, c.py, c.pz};
            const V3 n = cross(pb - pa, pc - pa);
            if (length(n) < 1e-9) continue;
            const V3 unit = normalize(n);
            const double distance = dot(unit, pa);
            min_distance = std::min(min_distance, distance);
            for (std::uint32_t j = volume.first_index; j < volume.first_index + volume.index_count; ++j)
            {
                const Vertex& v = mesh.vertices[mesh.indices[j]];
                if (dot(unit, V3{v.px, v.py, v.pz}) > distance + 1e-6) ++outside_vertices;
            }
        }
        check.expect(outside_vertices == 0 && min_distance * volume_scale >= 1.0 - 1e-9,
                     "light-volume sphere is convex and, scaled, circumscribes the light's radius (inradius * scale >= 1)",
                     std::format("inradius {:.5f}, scale {:.5f}, product {:.6f}, vertices beyond a face plane: {}", min_distance, volume_scale,
                                 min_distance * volume_scale, outside_vertices));
    }

    void DeferredShading::check_configuration(demo::Check& check) const
    {
        constexpr auto fb = mgl::FramebufferTarget::Framebuffer;
        mgl::glBindFramebuffer(fb, targets_.gbuffer_fbo);
        check.expect(mgl::glCheckFramebufferStatus(fb) == mgl::FramebufferStatus::Complete,
                     "G-buffer framebuffer is complete: RGBA8 + RGB10_A2 + RG8 colour textures and a DEPTH_COMPONENT24 texture");

        auto query = [&](mgl::FramebufferAttachment a, mgl::FramebufferAttachmentParameter p) {
            GLint v = -1;
            mgl::glGetFramebufferAttachmentParameteriv(fb, a, p, &v);
            return v;
        };
        using Param = mgl::FramebufferAttachmentParameter;
        const std::array<mgl::FramebufferAttachment, 3> colour = {mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                                                  mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1),
                                                                  mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color2)};
        bool normalized = true;
        for (const auto a : colour) normalized = normalized && query(a, Param::ComponentType) == static_cast<GLint>(GL_UNSIGNED_NORMALIZED);
        check.expect(normalized, "all three colour attachments are unsigned-normalized fixed point (COMPONENT_TYPE)");

        auto sizes = [&](mgl::FramebufferAttachment a) {
            return std::array<GLint, 4>{query(a, Param::RedSize), query(a, Param::GreenSize), query(a, Param::BlueSize), query(a, Param::AlphaSize)};
        };
        const auto s0 = sizes(colour[0]), s1 = sizes(colour[1]), s2 = sizes(colour[2]);
        check.expect(s0[0] >= 8 && s0[1] >= 8 && s0[2] >= 8 && s0[3] >= 8, "attachment 0 (albedo) has >= 8 bits in R, G, B and A",
                     std::format("R{} G{} B{} A{}", s0[0], s0[1], s0[2], s0[3]));
        check.expect(s1[0] >= 10 && s1[1] >= 10 && s1[2] >= 10 && s1[3] >= 2,
                     "attachment 1 (normal, RGB10_A2) has >= 10 bits in R, G, B and >= 2 in A",
                     std::format("R{} G{} B{} A{}", s1[0], s1[1], s1[2], s1[3]));
        check.expect(s2[0] >= 8 && s2[1] >= 8 && s2[2] == 0, "attachment 2 (material, RG8) has >= 8 bits in R and G and no blue",
                     std::format("R{} G{} B{} A{}", s2[0], s2[1], s2[2], s2[3]));
        const GLint depth_type = query(mgl::FramebufferAttachment::Depth, Param::ObjectType);
        const GLint depth_size = query(mgl::FramebufferAttachment::Depth, Param::DepthSize);
        check.expect(depth_type == static_cast<GLint>(GL_TEXTURE) && depth_size >= 24,
                     "the depth attachment is a texture with >= 24 depth bits (it is sampled by the lighting pass)",
                     std::format("object type {:#x}, depth size {}", depth_type, depth_size));

        const int max_draw = demo::get_integer(mgl::GetParameter::MaxDrawBuffers);
        const std::array<mgl::GetParameter, 4> draw_pnames = {mgl::GetParameter::DrawBuffer0, mgl::GetParameter::DrawBuffer1,
                                                              mgl::GetParameter::DrawBuffer2, mgl::GetParameter::DrawBuffer3};
        std::array<int, 4> draw{};
        for (std::size_t i = 0; i < 4; ++i) draw[i] = demo::get_integer(draw_pnames[i]);
        check.expect(max_draw >= 4 && draw[0] == static_cast<int>(GL_COLOR_ATTACHMENT0) && draw[1] == static_cast<int>(GL_COLOR_ATTACHMENT1) &&
                         draw[2] == static_cast<int>(GL_COLOR_ATTACHMENT2) && draw[3] == static_cast<int>(GL_NONE),
                     "glDrawBuffers state of the G-buffer: DRAW_BUFFER0..3 = COLOR_ATTACHMENT0, 1, 2, NONE (MAX_DRAW_BUFFERS >= 4)",
                     std::format("MAX_DRAW_BUFFERS {}, DRAW_BUFFER0..3 = {:#x} {:#x} {:#x} {:#x}", max_draw, draw[0], draw[1], draw[2], draw[3]));
        mgl::glBindFramebuffer(fb, mgl::FramebufferId{0});

        check.expect(mgl::glGetFragDataLocation(gbuffer_, "o_albedo") == 0 && mgl::glGetFragDataLocation(gbuffer_, "o_normal") == 1 &&
                         mgl::glGetFragDataLocation(gbuffer_, "o_material") == 2,
                     "fragment outputs are bound by layout(location): o_albedo = 0, o_normal = 1, o_material = 2 (glGetFragDataLocation)");

        // std140: the shaders assume two tightly packed vec4[64] arrays; the GL must agree.
        const GLuint block = mgl::glGetUniformBlockIndex(light_full_, "Lights");
        GLint block_size = 0;
        mgl::glGetActiveUniformBlockiv(light_full_, block, mgl::UniformBlockParameter::DataSize, &block_size);
        const std::array<const GLchar*, 2> names = {"u_pos_radius[0]", "u_color[0]"};
        std::array<GLuint, 2> indices{};
        mgl::glGetUniformIndices(light_full_, 2, names.data(), indices.data());
        std::array<GLint, 2> offsets{}, strides{};
        mgl::glGetActiveUniformsiv(light_full_, 2, indices.data(), mgl::UniformParameter::Offset, offsets.data());
        mgl::glGetActiveUniformsiv(light_full_, 2, indices.data(), mgl::UniformParameter::ArrayStride, strides.data());
        check.expect(block_size == static_cast<GLint>(sizeof(LightBlock)) && offsets[0] == 0 && offsets[1] == 16 * kMaxLights &&
                         strides[0] == 16 && strides[1] == 16,
                     "std140 layout of the Lights block matches the C++ struct: size 2048, u_pos_radius at 0, u_color at 1024, array stride 16",
                     std::format("size {}, offsets {} {}, strides {} {}", block_size, offsets[0], offsets[1], strides[0], strides[1]));
        const int max_block = demo::get_integer(mgl::GetParameter::MaxUniformBlockSize);
        check.expect(max_block >= static_cast<int>(sizeof(LightBlock)), "MAX_UNIFORM_BLOCK_SIZE accommodates the light block",
                     std::format("{} >= {}", max_block, sizeof(LightBlock)));
    }

    void DeferredShading::validate(demo::Env& env, demo::Check& check)
    {
        const int w = targets_.w, h = targets_.h;
        const double t = anim_time_;
        const int count = kLightCounts[light_count_index_];
        check.expect(env.width == w && env.height == h && w > 0 && h > 0, "render targets match the framebuffer size",
                     std::format("{}x{} pixels, {} lights, t = {:.4f} s", w, h, count, t));

        check_configuration(check);
        check_meshes(check, mesh_, volume_, volume_scale_);

        // --- Render the frame three ways and read everything back ----------------------------------------------
        FrameSpec spec;
        spec.t = t;
        spec.light_count = count;
        const Camera cam = camera_at(t);
        const std::vector<Light> lights = lights_at(t, count);
        const Model model = build_model(cam, lights, frame_objects(scene_, lights), w, h);
        const Budget budget = compute_budget(model);

        spec.variant = Variant::Volumes;
        const Pixels img_volumes = render_image(env, spec);
        const Pixels plane_albedo = read_plane(mgl::ColorAttachment::Color0);
        const Pixels plane_normal = read_plane(mgl::ColorAttachment::Color1);
        const Pixels plane_material = read_plane(mgl::ColorAttachment::Color2);
        const Pixels img_depth = composite_image(cam, 7);
        const Pixels img_position = composite_image(cam, 5);
        spec.variant = Variant::Fullscreen;
        const Pixels img_full = render_image(env, spec);
        spec.variant = Variant::Forward;
        const Pixels img_forward = render_image(env, spec);

        // --- Probes: stable pixels chosen on the CPU ----------------------------------------------------------------
        const std::vector<Candidate> candidates = build_candidates(model, 3, false, 2);
        const std::vector<Candidate> marker_candidates = build_candidates(model, 1, true, 1);
        const Object* floor_object = &model.objects.front();

        auto frac_ok = [](const Candidate& c) {
            for (const double v : {c.hit.pos.x, c.hit.pos.y, c.hit.pos.z})
            {
                const double f = frac(v + 0.25);
                if (f < 0.06 || f > 0.94) return false;
            }
            return true;
        };
        auto lit = [](const Candidate& c) { return c.brightest() >= 0.15 && c.brightest() <= 0.90; };

        struct ProbeSpec
        {
            const char* name;
            double tx, ty;  // preferred position (fractions of the image, y up)
            std::function<bool(const Candidate&)> accept;
        };
        const std::vector<ProbeSpec> probe_specs = {
            {"floor, light tile", 0.30, 0.20,
             [&](const Candidate& c) {
                 return c.object == floor_object && c.hit.face == 3 && (checker_cell(c.hit.pos) & 1L) == 0 && lit(c) && frac_ok(c) &&
                        c.specular_share() >= 0.08;
             }},
            {"floor, dark tile", 0.70, 0.30,
             [&](const Candidate& c) {
                 return c.object == floor_object && c.hit.face == 3 && (checker_cell(c.hit.pos) & 1L) == 1 && lit(c) && frac_ok(c);
             }},
            {"box, top face", 0.60, 0.55,
             [&](const Candidate& c) {
                 return c.object->shape == Shape::Box && c.object != floor_object && c.hit.face == 3 && lit(c) && frac_ok(c);
             }},
            {"box, side face", 0.40, 0.45,
             [&](const Candidate& c) {
                 return c.object->shape == Shape::Box && c.object != floor_object && c.hit.face != 3 && c.hit.face != 2 && lit(c) &&
                        frac_ok(c);
             }},
            {"sphere (diffuse dominated)", 0.50, 0.50,
             [&](const Candidate& c) {
                 return c.object->shape == Shape::Sphere && c.specular_share() < 0.04 && lit(c) && frac_ok(c);
             }},
            {"pillar side", 0.20, 0.70,
             [&](const Candidate& c) { return c.object->shape == Shape::Cylinder && lit(c) && frac_ok(c); }},
        };

        struct Probe
        {
            std::string name;
            const Candidate* c = nullptr;
        };
        std::vector<Probe> probes;
        auto nearest = [&](const std::vector<Candidate>& list, double tx, double ty, const std::function<bool(const Candidate&)>& accept) {
            const Candidate* best = nullptr;
            double best_d = std::numeric_limits<double>::infinity();
            for (const Candidate& c : list)
            {
                if (!accept(c)) continue;
                const double dx = c.x - tx * w, dy = c.y - ty * h;
                if (dx * dx + dy * dy < best_d)
                {
                    best_d = dx * dx + dy * dy;
                    best = &c;
                }
            }
            return best;
        };
        for (const ProbeSpec& ps : probe_specs)
        {
            const Candidate* c = nearest(candidates, ps.tx, ps.ty, ps.accept);
            check.expect(c != nullptr, std::format("probe pixel found: {}", ps.name),
                         c ? std::format("pixel ({}, {}), object {}, face {}", c->x, c->y, c->hit.object, c->hit.face) : std::string("none"));
            if (c) probes.push_back({ps.name, c});
        }
        {
            // The marker sphere with the most pixels: the candidate closest to the middle of its disc.
            const Candidate* c = nearest(marker_candidates, 0.5, 0.5, [](const Candidate&) { return true; });
            check.expect(c != nullptr, "probe pixel found: emissive light marker",
                         c ? std::format("pixel ({}, {}), marker {}", c->x, c->y, c->hit.object) : std::string("none"));
            if (c) probes.push_back({"light marker (emissive)", c});
        }

        // --- G-buffer planes, depth, rebuilt position and final colour at every probe ---------------------------------
        for (const Probe& p : probes)
        {
            const Candidate& c = *p.c;
            const std::string where = std::format("{} at ({}, {})", p.name, c.x, c.y);
            const double tol_n = c.curved ? kNormalTolCurved : kNormalTolPlanar;

            // RGBA8 albedo + specular.
            const auto a = texel(plane_albedo, w, c.x, c.y);
            const std::array<double, 4> a_expected = {c.albedo.x * 255.0, c.albedo.y * 255.0, c.albedo.z * 255.0, c.specular * 255.0};
            double a_dev = 0.0;
            for (std::size_t i = 0; i < 4; ++i) a_dev = std::max(a_dev, std::abs(a[i] - a_expected[i]));
            check.expect(a_dev <= kUnormTol, std::format("G-buffer albedo + specular, {}", where),
                         std::format("got ({},{},{},{}), expected ({:.1f},{:.1f},{:.1f},{:.1f}), max deviation {:.2f} (tol {})", a[0], a[1], a[2],
                                     a[3], a_expected[0], a_expected[1], a_expected[2], a_expected[3], a_dev, kUnormTol));

            // RGB10_A2 normal, read through RGBA/UNSIGNED_BYTE (8-bit conversion of the 10-bit value).
            const auto n = texel(plane_normal, w, c.x, c.y);
            const std::array<double, 3> n_gpu = {n[0] / 255.0 * 2.0 - 1.0, n[1] / 255.0 * 2.0 - 1.0, n[2] / 255.0 * 2.0 - 1.0};
            const double n_dev = std::max({std::abs(n_gpu[0] - c.hit.normal.x), std::abs(n_gpu[1] - c.hit.normal.y), std::abs(n_gpu[2] - c.hit.normal.z)});
            check.expect(n_dev <= tol_n, std::format("G-buffer normal, {}", where),
                         std::format("got ({:.4f},{:.4f},{:.4f}), analytic ({:.4f},{:.4f},{:.4f}), max deviation {:.4f} (tol {})", n_gpu[0], n_gpu[1],
                                     n_gpu[2], c.hit.normal.x, c.hit.normal.y, c.hit.normal.z, n_dev, tol_n));

            // RG8 material: shininess / 255 and emissive / 3.
            const auto m = texel(plane_material, w, c.x, c.y);
            const double m_dev = std::max(std::abs(m[0] - c.shininess), std::abs(m[1] - c.emissive / 3.0 * 255.0));
            check.expect(m_dev <= kUnormTol, std::format("G-buffer shininess + emissive, {}", where),
                         std::format("got ({},{}), expected ({:.1f},{:.1f}), max deviation {:.2f} (tol {})", m[0], m[1], c.shininess,
                                     c.emissive / 3.0 * 255.0, m_dev, kUnormTol));

            // Depth texture, packed into RGB8 by composite mode 7: eye distance along the view axis.
            const auto d = texel(img_depth, w, c.x, c.y);
            const double window_depth = static_cast<double>((d[0] << 16) | (d[1] << 8) | d[2]) / 16777215.0;
            const double z_gpu = linear_depth(window_depth);
            const double z_expected = dot(c.hit.pos - cam.eye, cam.forward);
            const double tol_z = c.curved ? kDepthTolCurvedPerRadius * c.radius + kDepthTol : kDepthTol;
            check.expect(std::abs(z_gpu - z_expected) <= tol_z, std::format("depth texture, {}", where),
                         std::format("window depth {:.7f} -> eye distance {:.4f}, analytic {:.4f}, deviation {:.5f} (tol {:.4f})", window_depth, z_gpu,
                                     z_expected, std::abs(z_gpu - z_expected), tol_z));

            // World position rebuilt from depth (composite mode 5 shows fract(position + 0.25)).
            const auto q = texel(img_position, w, c.x, c.y);
            const std::array<double, 3> q_expected = {frac(c.hit.pos.x + 0.25) * 255.0, frac(c.hit.pos.y + 0.25) * 255.0, frac(c.hit.pos.z + 0.25) * 255.0};
            const double q_dev = std::max({std::abs(q[0] - q_expected[0]), std::abs(q[1] - q_expected[1]), std::abs(q[2] - q_expected[2])});
            const double tol_q = c.curved ? kPositionTolCurved : kPositionTolPlanar;
            check.expect(q_dev <= tol_q, std::format("position rebuilt from depth, {}", where),
                         std::format("got ({},{},{}), analytic fract(p + 0.25) ({:.1f},{:.1f},{:.1f}), max deviation {:.2f} (tol {})", q[0], q[1], q[2],
                                     q_expected[0], q_expected[1], q_expected[2], q_dev, tol_q));

            // Final lit colour of the deferred (light volume) image against the analytic lighting model.
            const auto f = texel(img_volumes, w, c.x, c.y);
            const std::array<int, 3> f_expected = {to_byte(c.display.x), to_byte(c.display.y), to_byte(c.display.z)};
            const auto& f_budget = budget.model[model.index(c.x, c.y)];
            double f_ratio = 0.0;
            int f_dev = 0;
            for (std::size_t i = 0; i < 3; ++i)
            {
                f_dev = std::max(f_dev, std::abs(f[i] - f_expected[i]));
                f_ratio = std::max(f_ratio, std::abs(f[i] - f_expected[i]) / static_cast<double>(f_budget[i]));
            }
            check.expect(f_ratio <= 1.0, std::format("final colour (light volumes) vs analytic lighting, {}", where),
                         std::format("got ({},{},{}), expected ({},{},{}), specular share {:.2f}, max deviation {} = {:.2f} of the per-pixel budget ({:.1f},{:.1f},{:.1f})",
                                     f[0], f[1], f[2], f_expected[0], f_expected[1], f_expected[2], c.specular_share(), f_dev, f_ratio, f_budget[0],
                                     f_budget[1], f_budget[2]));
        }

        // --- Background pixel -----------------------------------------------------------------------------------------
        {
            int sx = -1;
            const int sy = h - 3;
            for (int x = w / 2; x < w - 3 && sx < 0; ++x)
            {
                bool all_sky = true;
                for (int dx = -2; dx <= 2 && all_sky; ++dx)
                {
                    const Hit& hit = model.at(x + dx, sy);
                    all_sky = !hit.hit;
                }
                if (all_sky) sx = x;
            }
            check.expect(sx >= 0, "probe pixel found: background (no ray hits anything)");
            if (sx >= 0)
            {
                const auto a = texel(plane_albedo, w, sx, sy);
                const auto d = texel(img_depth, w, sx, sy);
                const auto f = texel(img_volumes, w, sx, sy);
                const V3 sky = sky_color(sy, h);
                const std::array<int, 3> expected = {to_byte(sky.x), to_byte(sky.y), to_byte(sky.z)};
                check.expect(a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && d[0] == 255 && d[1] == 255 && d[2] == 255,
                             "background pixel: G-buffer cleared to 0 and depth = 1.0 (24-bit 0xFFFFFF)",
                             std::format("albedo plane ({},{},{},{}), packed depth ({},{},{})", a[0], a[1], a[2], a[3], d[0], d[1], d[2]));
                int dev = 0;
                for (std::size_t i = 0; i < 3; ++i) dev = std::max(dev, std::abs(f[i] - expected[i]));
                check.expect(dev <= 1, "background pixel shows the sky gradient (no lighting applied)",
                             std::format("got ({},{},{}), expected ({},{},{}), deviation {} (tol 1)", f[0], f[1], f[2], expected[0], expected[1],
                                         expected[2], dev));
            }
        }

        // --- Whole-image comparisons ---------------------------------------------------------------------------------
        const DiffStats d_full_fwd = compare_images(img_full, img_forward, w, h, budget.full);
        const DiffStats d_vol_fwd = compare_images(img_volumes, img_forward, w, h, budget.volumes);
        const DiffStats d_vol_full = compare_images(img_volumes, img_full, w, h, budget.volumes);
        // Pixels whose view ray hits the scene (CPU model). The negative controls below are judged against these, not the whole
        // surface: on a wide surface (1000 x 200) the scene fills only the middle third and the empty background, which no
        // lighting change can alter, would otherwise dilute the share of differing pixels.
        const std::size_t scene_pixels = static_cast<std::size_t>(std::count_if(model.hits.begin(), model.hits.end(), [](const Hit& hh) { return hh.hit; }));
        auto describe = [scene_pixels](const DiffStats& s) {
            return std::format("max difference {} steps, worst = {:.2f} of its per-pixel budget at ({},{}), mean {:.3f}, {} of {} pixels beyond budget ({} scene pixels)",
                               s.max_diff, s.max_ratio, s.at_x, s.at_y, s.mean, s.over, s.total, scene_pixels);
        };
        check.expect(d_full_fwd.over == 0,
                     std::format("deferred (full-screen lighting pass) matches the forward-rendered reference on all {} pixels", w * h),
                     describe(d_full_fwd));
        check.expect(d_vol_fwd.over == 0, "deferred (additive light volumes) matches the forward-rendered reference on every pixel", describe(d_vol_fwd));
        check.expect(d_vol_full.over == 0, "light volumes (ONE, ONE blending) and the full-screen pass produce the same image", describe(d_vol_full));

        // --- Negative controls: the same comparisons must be able to fail ------------------------------------------------
        {
            // (a) One light removed from the uniform block. Pick the (pixel, light) pair where removing it changes the image most.
            const Candidate* best_c = nullptr;
            std::size_t best_light = 0;
            double best_delta = 0.0;
            for (const Candidate& c : candidates)
            {
                if (c.object->marker) continue;
                const V3 view = normalize(cam.eye - c.hit.pos);
                for (std::size_t k = 0; k < lights.size(); ++k)
                {
                    const Terms tk = light_terms(lights[k], c.hit.pos, c.hit.normal, view, c.albedo, c.specular, c.shininess);
                    const V3 without = display_color(c.radiance.total() - tk.diffuse - tk.specular);
                    const double delta = std::max({std::abs(c.display.x - without.x), std::abs(c.display.y - without.y), std::abs(c.display.z - without.z)});
                    if (delta > best_delta)
                    {
                        best_delta = delta;
                        best_c = &c;
                        best_light = k;
                    }
                }
            }
            check.expect(best_c != nullptr && best_delta * 255.0 >= 32.0, "control (a): a pixel dominated by one light exists",
                         std::format("pixel ({}, {}), light {}, removing it changes the display colour by up to {:.0f}/255", best_c ? best_c->x : -1,
                                     best_c ? best_c->y : -1, best_light, best_delta * 255.0));
            if (best_c != nullptr)
            {
                FrameSpec off = spec;
                off.variant = Variant::Volumes;
                off.disabled_light = static_cast<int>(best_light);
                const Pixels img_off = render_image(env, off);
                const Candidate& c = *best_c;
                const V3 view = normalize(cam.eye - c.hit.pos);
                const Terms tk = light_terms(lights[best_light], c.hit.pos, c.hit.normal, view, c.albedo, c.specular, c.shininess);
                const V3 without = display_color(c.radiance.total() - tk.diffuse - tk.specular);
                const auto got = texel(img_off, w, c.x, c.y);
                const auto reference = texel(img_forward, w, c.x, c.y);
                const std::array<int, 3> without_bytes = {to_byte(without.x), to_byte(without.y), to_byte(without.z)};
                const auto& pixel_budget = budget.model[model.index(c.x, c.y)];
                const double allowed = *std::max_element(pixel_budget.begin(), pixel_budget.end());
                int dev_model = 0, dev_forward = 0;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    dev_model = std::max(dev_model, std::abs(got[i] - without_bytes[i]));
                    dev_forward = std::max(dev_forward, std::abs(got[i] - reference[i]));
                }
                check.expect(dev_model <= allowed, "control (a): with that light disabled, the deferred pixel equals the analytic model without the light",
                             std::format("got ({},{},{}), model ({},{},{}), deviation {} (budget {:.1f})", got[0], got[1], got[2], without_bytes[0],
                                         without_bytes[1], without_bytes[2], dev_model, allowed));
                check.expect(dev_forward > 2.0 * allowed, "control (a): ... and it then MISMATCHES the forward reference (the comparison can fail)",
                             std::format("deviation from the forward image {} (> 2 x budget {:.1f})", dev_forward, allowed));
            }

            // (b) Additive blending disabled: every light volume overwrites the previous result.
            FrameSpec no_blend = spec;
            no_blend.variant = Variant::Volumes;
            no_blend.blend = false;
            const Pixels img_no_blend = render_image(env, no_blend);
            const DiffStats d_nb = compare_images(img_no_blend, img_full, w, h, budget.volumes);
            check.expect(scene_pixels > 0 && d_nb.over * 20 >= scene_pixels, "control (b): without glEnable(BLEND) + glBlendFunc(ONE, ONE) the volume image differs from the full-screen image on >= 5% of the scene pixels",
                         describe(d_nb));

            // (c) Normal plane replaced by "up": the lighting pass visibly depends on the G-buffer contents.
            FrameSpec flat = spec;
            flat.variant = Variant::Volumes;
            flat.overwrite_normals = true;
            const Pixels img_flat = render_image(env, flat);
            const DiffStats d_flat = compare_images(img_flat, img_forward, w, h, budget.volumes);
            check.expect(scene_pixels > 0 && d_flat.over * 20 >= scene_pixels, "control (c): with the normal plane overwritten by (0, 1, 0) the image differs from the forward reference on >= 5% of the scene pixels",
                         describe(d_flat));
            // glClearBufferfv(COLOR, 1, ...) addressed draw buffer 1 only: the normal plane is flat "up", the albedo plane is untouched.
            const Pixels flat_normal = read_plane(mgl::ColorAttachment::Color1);
            const Pixels flat_albedo = read_plane(mgl::ColorAttachment::Color0);
            bool cleared = true, untouched = true;
            for (const Probe& p : probes)
            {
                const auto nrm = texel(flat_normal, w, p.c->x, p.c->y);
                cleared = cleared && std::abs(nrm[0] - 128) <= 1 && nrm[1] == 255 && std::abs(nrm[2] - 128) <= 1;
                untouched = untouched && texel(flat_albedo, w, p.c->x, p.c->y) == texel(plane_albedo, w, p.c->x, p.c->y);
            }
            check.expect(cleared && untouched && !probes.empty(),
                         "glClearBufferfv(COLOR, drawbuffer 1) set the normal plane to (0.5, 1, 0.5) = n(0, 1, 0) at every probe and left the albedo plane alone");

            // (d) Light volumes rasterized with both faces: every light is added twice.
            FrameSpec both_faces = spec;
            both_faces.variant = Variant::Volumes;
            both_faces.cull_volumes = false;
            const Pixels img_both = render_image(env, both_faces);
            const DiffStats d_both = compare_images(img_both, img_full, w, h, budget.volumes);
            check.expect(scene_pixels > 0 && d_both.over * 20 >= scene_pixels, "control (d): without glCullFace(FRONT) each light is added twice (front and back faces) and the image differs on >= 5% of the scene pixels",
                         describe(d_both));
        }

        // --- Animation: the lights move, and the deferred image follows the forward image at another time -------------
        {
            FrameSpec later = spec;
            later.light_time = t + 1.0;
            later.variant = Variant::Volumes;
            const Pixels deferred_later = render_image(env, later);
            later.variant = Variant::Forward;
            const Pixels forward_later = render_image(env, later);
            const std::vector<Light> lights_later = lights_at(t + 1.0, count);
            const Budget budget_later = compute_budget(build_model(cam, lights_later, frame_objects(scene_, lights_later), w, h));
            const DiffStats d_later = compare_images(deferred_later, forward_later, w, h, budget_later.volumes);
            const BudgetMap eight(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), {8.0f, 8.0f, 8.0f});
            const DiffStats d_moved = compare_images(deferred_later, img_volumes, w, h, eight);
            const Light l0 = light_at(0, t), l1 = light_at(0, t + 1.0);
            check.expect(length(l1.pos - l0.pos) > 0.05, "light 0 moves between t and t + 1 s (CPU model)",
                         std::format("moved {:.3f} world units", length(l1.pos - l0.pos)));
            check.expect(d_later.over == 0, "one second later the deferred image still matches the forward reference", describe(d_later));
            check.expect(scene_pixels > 0 && d_moved.over * 10 >= scene_pixels, "the image changes when only the lights move (same camera): >= 10% of the scene pixels differ by more than 8",
                         describe(d_moved));
        }

        // --- Leave the default framebuffer showing the frame the runtime captures ----------------------------------------
        draw_frame(env, current_spec());
        check.expect_no_gl_error("no stray GL error left by the validation");
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-deferred-shading",
        .title = "Deferred shading: G-buffer (MRT) + 48 moving point lights",
        .min_es = {3, 0},
        .summary = "A procedurally generated 3D scene is rendered once into a G-buffer (RGBA8 albedo, RGB10_A2 normal, RG8 material, "
                   "depth texture) with glDrawBuffers; a lighting pass rebuilds positions from depth and accumulates 48 coloured "
                   "moving lights, either in one full-screen pass or with additive light volumes. Checked against ray-cast "
                   "analytic values, a CPU lighting model and a forward-rendered reference.",
        .controls = "1 final | 2 albedo | 3 normals | 4 material | 5 depth | 6 world position from depth | 7 G-buffer mosaic; "
                    "V light volumes <-> full-screen pass; F forward-rendered reference; SPACE pause; "
                    "A/D orbit, W/S zoom; +/- number of lights (8..64)",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 30,
    };
    return demo::run<DeferredShading>(argc, argv, info);
}
