// es30-shadow-mapping
//
// A real 3D scene (checkered ground, a platform, boxes, pillars with a lintel, three spheres) lit by an animated
// directional light that casts shadows with the classic two-pass shadow map:
//
//   pass 1  render the scene from the light's point of view into a DEPTH_COMPONENT24 texture that is the
//           ONLY attachment of a framebuffer object (glDrawBuffers(NONE), glReadBuffer(NONE));
//   pass 2  render the scene from the camera; the fragment shader transforms each fragment into the
//           light's clip space and asks the depth texture "is something nearer to the light?" with a
//           sampler2DShadow (TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE, COMPARE_FUNC = LEQUAL).
//
// Hardware percentage-closer filtering comes from LINEAR filtering of that comparison; an optional 3x3
// grid of taps in the shader widens the penumbra. Shadow acne (self-shadowing of lit surfaces) is
// removed with glPolygonOffset in pass 1 plus a tiny receiver bias. A small inset shows the shadow
// map itself, sampled as an ordinary depth texture through a sampler object whose comparison mode
// is NONE.
//
// Everything is validated against an independent CPU model of the scene (analytic ray/box, ray/sphere
// and ray/plane intersections, in double precision): the CPU traces the camera ray of selected pixels,
// asks whether the hit point is occluded along the light direction, evaluates the lighting equation, and
// the GPU pixel must match - lit pixels with the full lighting, shadowed pixels with ambient only.
// See README.md for the details of every check.
//
// Keys (windowed): P cycles the PCF mode, B toggles the depth bias, Space pauses the animation,
// A/D orbit the camera, W/S move it closer / farther.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    // =============================================================================================
    // Double-precision vector helper for the CPU model (independent of the float math in common/).
    // =============================================================================================
    struct V3
    {
        double x = 0, y = 0, z = 0;
    };
    constexpr V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    constexpr V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    constexpr V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    constexpr V3 mul(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
    constexpr double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    constexpr V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    inline double length(V3 a) { return std::sqrt(dot(a, a)); }
    inline V3 normalize(V3 a) { return a * (1.0 / length(a)); }
    constexpr V3 lerp(V3 a, V3 b, double t) { return a * (1.0 - t) + b * t; }
    constexpr double radians(double deg) { return deg * std::numbers::pi / 180.0; }

    /// Rotation about +Y by `a` radians, the same convention as demo::math::rotate(a, {0,1,0}).
    inline V3 rot_y(V3 v, double a)
    {
        const double c = std::cos(a), s = std::sin(a);
        return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
    }

    // =============================================================================================
    // Scene constants. The CPU model and the shaders use exactly the same numbers.
    // =============================================================================================
    constexpr int kShadowSize = 1024;           // shadow map is kShadowSize x kShadowSize
    constexpr double kLightDistance = 25.0;     // the light's "eye" sits this far from the origin along L
    constexpr double kLightHalf = 11.0;         // orthographic half extent: the map covers 22 x 22 world units
    constexpr double kLightNear = 10.0;
    constexpr double kLightFar = 40.0;
    constexpr double kElevation = radians(34.0);  // light elevation above the horizon (constant)
    constexpr double kFovY = 44.0;              // camera field of view in degrees
    constexpr double kCamNear = 0.5, kCamFar = 600.0;

    constexpr float kOffsetFactor = 3.0f;       // glPolygonOffset(factor, units) of the shadow pass ("bias on")
    constexpr float kOffsetUnits = 4.0f;
    constexpr float kReceiverBias = 2.0e-4f;    // extra constant bias in the shader, window-z units ("bias on")

    constexpr double kGroundHalf = 800.0;       // the ground is a grid from -800 to +800 in x and z (beyond the far plane)
    constexpr int kGroundCells = 64;

    constexpr V3 kLightColor{0.95, 0.86, 0.68};
    constexpr V3 kSkyAmbient{0.22, 0.28, 0.40};
    constexpr V3 kGroundAmbient{0.14, 0.13, 0.12};
    constexpr V3 kFogColor{0.60, 0.72, 0.88};
    constexpr double kFogDensity = 0.010;

    struct Material
    {
        V3 albedo;
        V3 albedo2;        // second checker color (ground only)
        bool checker;
        double specular;   // k_s
        double shininess;  // exponent
    };

    enum class Kind { Box, Sphere };

    struct Shape
    {
        Kind kind;
        V3 center;
        V3 half;           // box half extents
        double radius;     // sphere radius
        double yaw;        // box rotation about +Y (radians)
        Material mat;
    };

    constexpr Material kGroundMaterial{{0.80, 0.72, 0.54}, {0.42, 0.46, 0.47}, true, 0.10, 24.0};

    /// The shapes at animation time t (seconds). Index order is part of the validation (ids).
    std::vector<Shape> make_scene(double t)
    {
        auto box = [](V3 c, V3 half, double yaw_deg, V3 albedo) {
            return Shape{Kind::Box, c, half, 0.0, radians(yaw_deg), {albedo, albedo, false, 0.30, 32.0}};
        };
        auto sphere = [](V3 c, double r, V3 albedo) {
            return Shape{Kind::Sphere, c, {}, r, 0.0, {albedo, albedo, false, 0.55, 64.0}};
        };
        const V3 orange{0.90, 0.48, 0.16}, purple{0.55, 0.30, 0.75}, teal{0.15, 0.58, 0.62}, cream{0.82, 0.78, 0.68};
        const V3 red{0.82, 0.14, 0.14}, blue{0.20, 0.36, 0.86}, green{0.20, 0.70, 0.30};

        const V3 stone{0.70, 0.62, 0.50};

        std::vector<Shape> s;
        s.push_back(box({1.0, 0.15, -0.3}, {2.6, 0.15, 2.0}, 0.0, stone));                           // 0 low platform (a raised receiver)
        s.push_back(box({-3.3, 0.8, -0.6}, {0.9, 0.8, 0.9}, 25.0, orange));                          // 1 big box
        s.push_back(box({-3.3, 1.95, -0.6}, {0.45, 0.35, 0.45}, 20.0 + 40.0 * t, purple));           // 2 spinning box on it
        s.push_back(box({2.4, 0.8, -1.0}, {0.5, 0.5, 0.5}, -35.0, teal));                            // 3 cube on the platform
        for (int i = 0; i < 4; ++i)                                                                  // 4..7 pillars
            s.push_back(box({-2.55 + 1.7 * i, 1.5, -3.8}, {0.28, 1.5, 0.28}, 0.0, cream));
        s.push_back(box({0.0, 3.15, -3.8}, {3.1, 0.15, 0.42}, 0.0, cream));                          // 8 lintel
        s.push_back(sphere({0.3, 1.2, 0.0}, 0.9, red));                                              // 9 big sphere on the platform
        s.push_back(sphere({3.0, 2.3 + 0.35 * std::sin(1.3 * t), 0.9}, 0.5, blue));                  // 10 floating sphere
        s.push_back(sphere({-0.9, 0.65, -1.1}, 0.35, green));                                        // 11 small sphere on the platform
        return s;
    }

    // =============================================================================================
    // CPU model: camera, light frame, ray casting, lighting.
    // =============================================================================================
    struct Camera
    {
        V3 eye, target, fwd, right, up;
        double tan_half_fovy = 0, aspect = 1;
    };

    Camera make_camera(double t, double yaw_offset, double zoom, int w, int h)
    {
        Camera c;
        const double az = 0.55 + 0.10 * t + yaw_offset;
        const double radius = 11.0 * zoom;
        c.eye = {radius * std::sin(az), 5.2, radius * std::cos(az)};
        c.target = {0.6, 1.9, -0.4};
        c.fwd = normalize(c.target - c.eye);
        c.right = normalize(cross(c.fwd, V3{0, 1, 0}));
        c.up = cross(c.right, c.fwd);
        c.tan_half_fovy = std::tan(radians(kFovY) * 0.5);
        c.aspect = static_cast<double>(w) / static_cast<double>(h);
        return c;
    }

    /// World-space direction of the ray through the center of window pixel (px, py) (origin lower left).
    V3 pixel_dir(const Camera& c, int w, int h, int px, int py)
    {
        const double nx = (px + 0.5) / w * 2.0 - 1.0;
        const double ny = (py + 0.5) / h * 2.0 - 1.0;
        return normalize(c.fwd + c.right * (nx * c.tan_half_fovy * c.aspect) + c.up * (ny * c.tan_half_fovy));
    }

    /// The light's orthographic view: `eye` looks along `fwd` (= -L) at the origin.
    struct LightFrame
    {
        V3 L;      // unit vector towards the light
        V3 eye, fwd, right, up;
    };

    LightFrame make_light(double azimuth)
    {
        LightFrame f;
        f.L = {std::cos(kElevation) * std::cos(azimuth), std::sin(kElevation), std::cos(kElevation) * std::sin(azimuth)};
        f.eye = f.L * kLightDistance;
        f.fwd = f.L * -1.0;
        f.right = normalize(cross(f.fwd, V3{0, 1, 0}));
        f.up = cross(f.right, f.fwd);
        return f;
    }

    /// (s, t, depth) of a world point in the shadow map: s,t in [0,1] over the map, depth = window z in [0,1].
    V3 to_light_uv(const LightFrame& f, V3 p)
    {
        const V3 rel = p - f.eye;
        return {dot(rel, f.right) / kLightHalf * 0.5 + 0.5, dot(rel, f.up) / kLightHalf * 0.5 + 0.5,
                (dot(rel, f.fwd) - kLightNear) / (kLightFar - kLightNear)};
    }

    struct Model
    {
        double t = 0;
        int w = 0, h = 0;
        std::vector<Shape> shapes;
        Camera cam;
        double light_azimuth = 0;
        LightFrame light;
    };

    Model make_model(double t, double yaw_offset, double zoom, int w, int h)
    {
        Model m;
        m.t = t;
        m.w = w;
        m.h = h;
        m.shapes = make_scene(t);
        m.cam = make_camera(t, yaw_offset, zoom, w, h);
        m.light_azimuth = 3.55 + 0.35 * t;
        m.light = make_light(m.light_azimuth);
        return m;
    }

    struct Hit
    {
        bool ok = false;
        double t = std::numeric_limits<double>::infinity();
        int id = -2;  // -1 ground, >= 0 shape index
        V3 n;
    };

    /// Ray vs shape. Returns true when the ray touches the shape at a parameter >= 0 (a ray that starts inside
    /// reports t_near < 0). `n` is the outward normal at the entry point.
    bool intersect_shape(const Shape& s, V3 o, V3 d, double& t_near, V3& n)
    {
        if (s.kind == Kind::Sphere)
        {
            const V3 oc = o - s.center;
            const double b = dot(oc, d), c = dot(oc, oc) - s.radius * s.radius;
            const double disc = b * b - c;
            if (disc < 0.0) return false;
            const double sq = std::sqrt(disc);
            const double t0 = -b - sq, t1 = -b + sq;
            if (t1 < 0.0) return false;
            t_near = t0;
            n = normalize(oc + d * t0);
            return true;
        }
        // Oriented box: slab test in the box's local frame.
        const V3 lo = rot_y(o - s.center, -s.yaw), ld = rot_y(d, -s.yaw);
        const double oo[3] = {lo.x, lo.y, lo.z}, dd[3] = {ld.x, ld.y, ld.z}, hh[3] = {s.half.x, s.half.y, s.half.z};
        double t0 = -std::numeric_limits<double>::infinity(), t1 = std::numeric_limits<double>::infinity();
        int axis = 0;
        double sign = 1.0;
        for (int a = 0; a < 3; ++a)
        {
            if (std::abs(dd[a]) < 1e-14)
            {
                if (std::abs(oo[a]) > hh[a]) return false;
                continue;
            }
            double ta = (-hh[a] - oo[a]) / dd[a], tb = (hh[a] - oo[a]) / dd[a];
            double sg = -1.0;  // entering through the -axis face
            if (ta > tb)
            {
                std::swap(ta, tb);
                sg = 1.0;
            }
            if (ta > t0)
            {
                t0 = ta;
                axis = a;
                sign = sg;
            }
            t1 = std::min(t1, tb);
            if (t0 > t1) return false;
        }
        if (t1 < 0.0) return false;
        t_near = t0;
        V3 ln{0, 0, 0};
        (axis == 0 ? ln.x : axis == 1 ? ln.y : ln.z) = sign;
        n = rot_y(ln, s.yaw);
        return true;
    }

    /// Nearest hit of a ray with the ground plane y = 0 and all shapes.
    Hit trace(const std::vector<Shape>& shapes, V3 o, V3 d, bool with_ground = true)
    {
        Hit best;
        if (with_ground && d.y < -1e-12 && o.y > 0.0)
        {
            best.ok = true;
            best.t = -o.y / d.y;
            best.id = -1;
            best.n = {0, 1, 0};
        }
        for (std::size_t i = 0; i < shapes.size(); ++i)
        {
            double t = 0;
            V3 n;
            if (intersect_shape(shapes[i], o, d, t, n) && t >= 0.0 && t < best.t)
            {
                best.ok = true;
                best.t = t;
                best.id = static_cast<int>(i);
                best.n = n;
            }
        }
        return best;
    }

    /// True when something lies between the surface point `p` (normal `n`) and the light.
    bool blocked(const std::vector<Shape>& shapes, V3 L, V3 p, V3 n)
    {
        const V3 o = p + n * 1e-4;  // lifted off the surface so a surface does not shadow itself analytically
        for (const Shape& s : shapes)
        {
            double t = 0;
            V3 dummy;
            if (intersect_shape(s, o, L, t, dummy)) return true;  // includes a ray that starts inside
        }
        return false;
    }

    int checker_parity(V3 p)
    {
        const long long i = static_cast<long long>(std::floor(p.x)) + static_cast<long long>(std::floor(p.z));
        return static_cast<int>(((i % 2) + 2) % 2);
    }

    demo::Rgba8 to_rgba8(V3 c)
    {
        auto q = [](double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); };
        return {q(c.x), q(c.y), q(c.z), 255};
    }

    V3 gamma(V3 c)
    {
        const double e = 1.0 / 2.2;
        return {std::pow(c.x, e), std::pow(c.y, e), std::pow(c.z, e)};
    }

    /// The lighting equation of scene.frag. `shadow` is 1 (lit) or 0 (shadowed); ambient is never shadowed.
    V3 shade(const Material& m, V3 albedo, V3 p, V3 n, V3 eye, V3 L, double shadow)
    {
        const V3 v = normalize(eye - p);
        const V3 h = normalize(L + v);
        const double diffuse = std::max(dot(n, L), 0.0);
        const double spec = diffuse > 0.0 ? std::pow(std::max(dot(n, h), 0.0), m.shininess) : 0.0;
        const V3 ambient = lerp(kGroundAmbient, kSkyAmbient, 0.5 * n.y + 0.5);
        V3 c = mul(albedo, ambient + kLightColor * (diffuse * shadow)) + kLightColor * (m.specular * spec * shadow);
        const double fog = std::exp(-kFogDensity * length(p - eye));
        c = lerp(kFogColor, c, fog);
        return gamma(c);
    }

    // =============================================================================================
    // Meshes
    // =============================================================================================
    struct Vertex
    {
        float px, py, pz, nx, ny, nz;
    };

    struct MeshData
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;
    };

    /// Flips every triangle whose winding disagrees with its vertex normals (front faces are CCW).
    void orient_triangles(MeshData& m)
    {
        for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3)
        {
            const Vertex &a = m.vertices[m.indices[i]], &b = m.vertices[m.indices[i + 1]], &c = m.vertices[m.indices[i + 2]];
            const V3 e1{b.px - a.px, b.py - a.py, b.pz - a.pz}, e2{c.px - a.px, c.py - a.py, c.pz - a.pz};
            const V3 n{a.nx, a.ny, a.nz};
            if (dot(cross(e1, e2), n) < 0.0) std::swap(m.indices[i + 1], m.indices[i + 2]);
        }
    }

    /// Unit cube [-0.5, 0.5]^3 with flat normals (24 vertices).
    MeshData make_cube()
    {
        MeshData m;
        const V3 normals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        const V3 tangents[6] = {{0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        for (int f = 0; f < 6; ++f)
        {
            const V3 n = normals[f], u = tangents[f], v = cross(n, u);
            const double corners[4][2] = {{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};
            const auto base = static_cast<std::uint16_t>(m.vertices.size());
            for (const auto& c : corners)
            {
                const V3 p = n * 0.5 + u * c[0] + v * c[1];
                m.vertices.push_back({static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z),
                                      static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z)});
            }
            for (const std::uint16_t k : {0, 1, 2, 0, 2, 3}) m.indices.push_back(static_cast<std::uint16_t>(base + k));
        }
        orient_triangles(m);
        return m;
    }

    /// Unit sphere, `rows` latitude bands x `cols` longitude segments, smooth normals.
    MeshData make_sphere(int rows, int cols)
    {
        MeshData m;
        for (int i = 0; i <= rows; ++i)
        {
            const double theta = std::numbers::pi * i / rows;
            for (int j = 0; j <= cols; ++j)
            {
                const double phi = 2.0 * std::numbers::pi * j / cols;
                const V3 p{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
                m.vertices.push_back({static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z),
                                      static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)});
            }
        }
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                const auto a = static_cast<std::uint16_t>(i * (cols + 1) + j), b = static_cast<std::uint16_t>(a + cols + 1);
                for (const std::uint16_t k : {a, b, static_cast<std::uint16_t>(a + 1), static_cast<std::uint16_t>(a + 1), b,
                                              static_cast<std::uint16_t>(b + 1)})
                    m.indices.push_back(k);
            }
        orient_triangles(m);
        return m;
    }

    /// Flat ground y = 0, (cells x cells) quads from -half to +half, normal +Y. Finer than two triangles so the
    /// interpolated light-space coordinates stay precise over the very large quad.
    MeshData make_ground(int cells, double half)
    {
        MeshData m;
        for (int i = 0; i <= cells; ++i)
            for (int j = 0; j <= cells; ++j)
                m.vertices.push_back({static_cast<float>(-half + 2.0 * half * j / cells), 0.0f,
                                      static_cast<float>(-half + 2.0 * half * i / cells), 0.0f, 1.0f, 0.0f});
        for (int i = 0; i < cells; ++i)
            for (int j = 0; j < cells; ++j)
            {
                const auto a = static_cast<std::uint16_t>(i * (cells + 1) + j), b = static_cast<std::uint16_t>(a + cells + 1);
                for (const std::uint16_t k : {a, b, static_cast<std::uint16_t>(a + 1), static_cast<std::uint16_t>(a + 1), b,
                                              static_cast<std::uint16_t>(b + 1)})
                    m.indices.push_back(k);
            }
        orient_triangles(m);
        return m;
    }

    struct Mesh
    {
        mgl::VertexArrayId vao{};
        mgl::BufferId vbo{};
        mgl::BufferId ibo{};
        GLsizei count = 0;
    };

    // =============================================================================================
    // Validation helpers
    // =============================================================================================
    struct InsetRect
    {
        int x, y, size;
    };

    InsetRect inset_rect(int w, int h)
    {
        const int size = std::min(w, h) * 27 / 100;
        return {6, h - 6 - size, size};
    }

    constexpr double kInsetLo = 0.05, kInsetHi = 0.95;  // shadow-map depths shown as black / white by the inset's ramp
    constexpr double kInsetBands = 24.0;                // sawtooth periods over the depth range [0, 1]
    struct InsetView { double x, y; };
    constexpr InsetView kInsetView{0.2, 0.6};           // the inset shows s,t in [0.2, 0.8]: the middle of the shadow map

    /// What the validation knows about one pixel whose camera ray hits a flat or curved surface.
    struct Probe
    {
        int x = 0, y = 0;
        int id = -2;           // -1 ground, >= 0 shape index
        V3 p, n;               // hit point and normal
        double ndl = 0;        // dot(n, L)
        V3 albedo;
        Material mat{};
        bool occluded = false; // something between the point and the light (true model)
        bool stable = false;   // the occlusion is the same on a ring of radius kRing around the point
        bool in_map = false;   // inside the shadow map's extent
        demo::Rgba8 lit{}, shadowed{};
    };

    constexpr double kRing = 0.25;      // world units: shadow-edge exclusion margin (see README)
    constexpr int kMapMargin = 4;       // texels: no depth discontinuity of the shadow map within this distance of a compared pixel
    constexpr double kMinNdl = 0.25;    // below this the lit/shadowed difference is too small to discriminate
    constexpr double kMaxProbeDistance = 24.0;  // world units from the camera: beyond 25 the ground checker fades out (scene.frag)
    constexpr int kTolerance = 3;       // 8-bit levels, see README "Tolerances"

    struct ProbeSet
    {
        std::vector<Probe> solid;  // far from every shadow edge (the strict comparison set)
        std::vector<Probe> edge;   // ground pixels near a shadow edge (the penumbra experiment)
        int ground_lit = 0, ground_shadow = 0, object_lit = 0, object_shadow = 0, back = 0;
    };

    struct Tally
    {
        int checked = 0, mismatches = 0, worst = 0;
        std::string first;
    };

    int max_diff(const std::uint8_t* px, const demo::Rgba8& e)
    {
        int d = 0;
        for (int c = 0; c < 3; ++c) d = std::max(d, std::abs(static_cast<int>(px[c]) - static_cast<int>(e[static_cast<std::size_t>(c)])));
        return d;
    }

    // ---- the shadow map as the CPU model sees it ------------------------------------------------------------
    double depth_from_t(double t) { return (t - kLightNear) / (kLightFar - kLightNear); }

    /// Origin (on the light's image plane) of the ray through the center of shadow-map texel (tx, ty).
    V3 light_texel_origin(const LightFrame& f, int tx, int ty)
    {
        const double x = ((tx + 0.5) / kShadowSize * 2.0 - 1.0) * kLightHalf;
        const double y = ((ty + 0.5) / kShadowSize * 2.0 - 1.0) * kLightHalf;
        return f.eye + f.right * x + f.up * y;
    }

    /// First surface the light sees through texel (tx, ty): what the shadow map stores there.
    Hit light_hit(const Model& m, int tx, int ty)
    {
        return trace(m.shapes, light_texel_origin(m.light, tx, ty), m.light.fwd, true);
    }

    /// Depth of the bare ground plane at the center of texel (tx, ty) (what the map holds where nothing stands).
    double ground_depth(const Model& m, int tx, int ty)
    {
        const V3 o = light_texel_origin(m.light, tx, ty);
        return depth_from_t(-o.y / m.light.fwd.y);
    }

    /// True when texels within `radius` of (tx, ty) all see the same surface id (the texel is far from a depth edge).
    bool light_neighbourhood_uniform(const Model& m, int tx, int ty, int id, int radius)
    {
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx)
            {
                const Hit q = light_hit(m, tx + dx, ty + dy);
                if (!q.ok || q.id != id) return false;
            }
        return true;
    }

    struct TexelTarget
    {
        int tx, ty, id;
        double expected;  // analytic window-space depth the map must hold
    };

    /// Texels whose depth the CPU model can predict: ground in the open, the top of every box, the center of
    /// every sphere. Ground texels under a shadow edge and texels near a depth discontinuity are skipped.
    std::vector<TexelTarget> pick_targets(const Model& m)
    {
        std::vector<V3> points = {{5, 0, -6}, {-6, 0, -5}, {6, 0, 3}, {-5, 0, 5}};
        for (const Shape& s : m.shapes)
            if (s.kind == Kind::Sphere) points.push_back(s.center);
        for (const Shape& s : m.shapes)
            if (s.kind == Kind::Box) points.push_back(s.center + V3{0, s.half.y, 0});
        points.push_back({1, 0, 7});
        points.push_back({-7, 0, 0});

        std::vector<TexelTarget> out;
        for (const V3 p : points)
        {
            const V3 uv = to_light_uv(m.light, p);
            const int tx = static_cast<int>(std::floor(uv.x * kShadowSize)), ty = static_cast<int>(std::floor(uv.y * kShadowSize));
            if (tx < 4 || ty < 4 || tx > kShadowSize - 5 || ty > kShadowSize - 5) continue;
            const Hit h = light_hit(m, tx, ty);
            if (!h.ok || !light_neighbourhood_uniform(m, tx, ty, h.id, 2)) continue;
            out.push_back({tx, ty, h.id, depth_from_t(h.t)});
            if (out.size() == 16) break;
        }
        return out;
    }

    // ---- probes: pixels whose lit / shadowed color the CPU can predict --------------------------------------
    /// The shadow-edge exclusion test: the occlusion at `p` must equal the occlusion on a ring of radius `radius` (and half
    /// of it) around `p` in the tangent plane.
    bool occlusion_stable(const std::vector<Shape>& shapes, V3 L, V3 p, V3 n, bool occluded, double radius)
    {
        const V3 t1 = normalize(cross(n, std::abs(n.x) < 0.9 ? V3{1, 0, 0} : V3{0, 1, 0}));
        const V3 t2 = cross(n, t1);
        for (const double r : {radius, radius * 0.5})
            for (int k = 0; k < 8; ++k)
            {
                const double a = 2.0 * std::numbers::pi * k / 8.0;
                const V3 q = p + t1 * (r * std::cos(a)) + t2 * (r * std::sin(a));
                if (blocked(shapes, L, q, n) != occluded) return false;
            }
        return true;
    }

    ProbeSet build_probes(const Model& m)
    {
        ProbeSet ps;
        const InsetRect inset = inset_rect(m.w, m.h);
        for (int y = 3; y < m.h - 3; y += 3)
            for (int x = 3; x < m.w - 3; x += 3)
            {
                if (x <= inset.x + inset.size + 3 && y >= inset.y - 3) continue;  // under the inset
                const V3 dir = pixel_dir(m.cam, m.w, m.h, x, y);
                const Hit h = trace(m.shapes, m.cam.eye, dir);
                if (!h.ok || h.t > kMaxProbeDistance) continue;  // far away the fog hides the difference between lit and shadowed
                const V3 p = m.cam.eye + dir * h.t;

                // The pixel must be well inside one surface: the 3x3 neighbourhood at +-2 pixels hits the same primitive
                // (and, for the ground, the same checker cell; for a box, the same face).
                bool uniform = true;
                for (int dy = -2; dy <= 2 && uniform; dy += 2)
                    for (int dx = -2; dx <= 2 && uniform; dx += 2)
                    {
                        const V3 qd = pixel_dir(m.cam, m.w, m.h, x + dx, y + dy);
                        const Hit q = trace(m.shapes, m.cam.eye, qd);
                        if (!q.ok || q.t > kCamFar || q.id != h.id) uniform = false;
                        else if (h.id == -1 && checker_parity(m.cam.eye + qd * q.t) != checker_parity(p)) uniform = false;
                        else if (h.id >= 0 && m.shapes[static_cast<std::size_t>(h.id)].kind == Kind::Box && dot(q.n, h.n) < 0.999) uniform = false;
                    }
                if (!uniform) continue;

                Probe pr;
                pr.x = x;
                pr.y = y;
                pr.id = h.id;
                pr.p = p;
                pr.n = h.n;
                pr.mat = h.id == -1 ? kGroundMaterial : m.shapes[static_cast<std::size_t>(h.id)].mat;
                pr.albedo = (pr.mat.checker && checker_parity(p) == 1) ? pr.mat.albedo2 : pr.mat.albedo;
                pr.ndl = dot(h.n, m.light.L);
                pr.lit = to_rgba8(shade(pr.mat, pr.albedo, p, h.n, m.cam.eye, m.light.L, 1.0));
                pr.shadowed = to_rgba8(shade(pr.mat, pr.albedo, p, h.n, m.cam.eye, m.light.L, 0.0));
                const V3 uv = to_light_uv(m.light, p);
                pr.in_map = uv.x > 0.03 && uv.x < 0.97 && uv.y > 0.03 && uv.y < 0.97 && uv.z > 0.02 && uv.z < 0.98;  // inside the light's frustum

                if (pr.ndl <= -0.02)  // faces away from the light: ambient only, whatever the shadow map says
                {
                    pr.stable = true;
                    ps.solid.push_back(pr);
                    ++ps.back;
                    continue;
                }
                if (pr.ndl < kMinNdl) continue;  // too oblique: lit and shadowed colors are too close to tell apart
                pr.occluded = blocked(m.shapes, m.light.L, p, h.n);
                pr.stable = occlusion_stable(m.shapes, m.light.L, p, h.n, pr.occluded, kRing);
                if (pr.stable)
                {
                    // ... and the shadow map itself shows ONE surface around the point's texel over the whole PCF footprint (the
                    // 3x3 taps of 2x2 texels reach 3 texels): no depth discontinuity of the map, e.g. where two shadows meet.
                    const int tx = static_cast<int>(std::floor(uv.x * kShadowSize)), ty = static_cast<int>(std::floor(uv.y * kShadowSize));
                    const Hit centre = light_hit(m, tx, ty);
                    pr.stable = !pr.in_map || (centre.ok && light_neighbourhood_uniform(m, tx, ty, centre.id, kMapMargin));
                }
                if (pr.stable)
                {
                    ps.solid.push_back(pr);
                    if (pr.id == -1) (pr.occluded ? ps.ground_shadow : ps.ground_lit)++;
                    else (pr.occluded ? ps.object_shadow : ps.object_lit)++;
                }
                else if (pr.id == -1)
                    ps.edge.push_back(pr);
            }
        return ps;
    }

    /// Alternative shadow geometry for the negative controls: other occluders and/or another light direction.
    struct ShadowModel
    {
        const std::vector<Shape>* shapes;
        V3 L;
    };

    /// What the comparison against the CPU model should assume about every probe's shadow state.
    enum class Expect { Model, AllLit, AllShadowed, Inverted };

    Tally evaluate(const std::vector<Probe>& probes, const std::vector<std::uint8_t>& img, int w, const ShadowModel* alt = nullptr,
                   Expect expect = Expect::Model)
    {
        Tally t;
        for (const Probe& p : probes)
        {
            bool occluded = alt != nullptr ? blocked(*alt->shapes, alt->L, p.p, p.n) : p.occluded;
            if (expect == Expect::AllLit) occluded = false;
            else if (expect == Expect::AllShadowed) occluded = true;
            else if (expect == Expect::Inverted) occluded = !occluded;
            const demo::Rgba8& expected = (p.ndl <= 0.0 || !occluded) ? p.lit : p.shadowed;
            const std::uint8_t* px = &img[(static_cast<std::size_t>(p.y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(p.x)) * 4];
            const int d = max_diff(px, expected);
            ++t.checked;
            t.worst = std::max(t.worst, d);
            if (d > kTolerance)
            {
                ++t.mismatches;
                if (t.first.empty())
                    t.first = std::format("first at ({},{}) id {} at world ({:.2f},{:.2f},{:.2f}) ndl {:.2f} occluded {}: got ({},{},{}) expected {}", p.x, p.y, p.id, p.p.x, p.p.y,
                                          p.p.z, p.ndl, occluded, px[0], px[1], px[2], demo::format_rgba(expected));
            }
        }
        return t;
    }

    struct Penumbra
    {
        int pixels = 0, lit = 0, shadow = 0, between = 0, other = 0;
    };

    /// Classifies ground pixels near a shadow edge as fully lit, fully shadowed, strictly in between, or other.
    Penumbra classify_edge(const std::vector<Probe>& edge, const std::vector<std::uint8_t>& img, int w)
    {
        Penumbra r;
        for (const Probe& p : edge)
        {
            if (max_diff(p.lit.data(), p.shadowed) < 4 * kTolerance) continue;  // colors too close to classify
            const std::uint8_t* px = &img[(static_cast<std::size_t>(p.y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(p.x)) * 4];
            ++r.pixels;
            if (max_diff(px, p.lit) <= kTolerance) ++r.lit;
            else if (max_diff(px, p.shadowed) <= kTolerance) ++r.shadow;
            else
            {
                bool between = true;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    const int lo = std::min<int>(p.lit[c], p.shadowed[c]) - kTolerance, hi = std::max<int>(p.lit[c], p.shadowed[c]) + kTolerance;
                    if (px[c] < lo || px[c] > hi) between = false;
                }
                (between ? r.between : r.other)++;
            }
        }
        return r;
    }

    // =============================================================================================
    // The demo
    // =============================================================================================
    struct RenderParams
    {
        bool bias = true;
        int pcf = 2;  // 0: NEAREST, 1 tap; 1: LINEAR (hardware PCF), 1 tap; 2: LINEAR + 3x3 taps
    };

    constexpr std::array<const char*, 3> kPcfNames = {"off (NEAREST, 1 tap)", "hardware (LINEAR, 1 tap)", "hardware + 3x3 taps"};

    struct ShadowProgram
    {
        mgl::ProgramId id{};
        mgl::UniformLocation light_vp{}, model{};
    };

    struct SceneProgram
    {
        mgl::ProgramId id{};
        mgl::UniformLocation model{}, normal_matrix{}, view_proj{}, light_vp{}, shadow{}, texel{}, bias{}, pcf_radius{}, eye{},
            light_dir{}, light_color{}, sky_ambient{}, ground_ambient{}, fog_color{}, fog_density{}, albedo{}, albedo2{},
            checker{}, specular{}, shininess{};
    };

    mgl::UniformLocation uloc(mgl::ProgramId program, const char* name)
    {
        return mgl::glGetUniformLocation(program, name);
    }

    class ShadowMapping final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // ---- programs -------------------------------------------------------------------------
            shadow_.id = demo::build_program(env, "shaders/shadow.vert", "shaders/shadow.frag");
            shadow_.light_vp = uloc(shadow_.id, "u_light_vp");
            shadow_.model = uloc(shadow_.id, "u_model");

            scene_.id = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            SceneProgram& s = scene_;
            s.model = uloc(s.id, "u_model");
            s.normal_matrix = uloc(s.id, "u_normal_matrix");
            s.view_proj = uloc(s.id, "u_view_proj");
            s.light_vp = uloc(s.id, "u_light_vp");
            s.shadow = uloc(s.id, "u_shadow");
            s.texel = uloc(s.id, "u_texel");
            s.bias = uloc(s.id, "u_bias");
            s.pcf_radius = uloc(s.id, "u_pcf_radius");
            s.eye = uloc(s.id, "u_eye");
            s.light_dir = uloc(s.id, "u_light_dir");
            s.light_color = uloc(s.id, "u_light_color");
            s.sky_ambient = uloc(s.id, "u_sky_ambient");
            s.ground_ambient = uloc(s.id, "u_ground_ambient");
            s.fog_color = uloc(s.id, "u_fog_color");
            s.fog_density = uloc(s.id, "u_fog_density");
            s.albedo = uloc(s.id, "u_albedo");
            s.albedo2 = uloc(s.id, "u_albedo2");
            s.checker = uloc(s.id, "u_checker");
            s.specular = uloc(s.id, "u_specular");
            s.shininess = uloc(s.id, "u_shininess");

            inset_program_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/inset.frag");
            inset_depth_ = uloc(inset_program_, "u_depth");
            inset_range_ = uloc(inset_program_, "u_range");
            inset_bands_ = uloc(inset_program_, "u_bands");
            inset_view_ = uloc(inset_program_, "u_view");
            dump_program_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/depth_dump.frag");
            dump_depth_ = uloc(dump_program_, "u_depth");
            dump_texels_ = uloc(dump_program_, "u_texels");

            // ---- meshes -----------------------------------------------------------------------------
            cube_ = create_mesh(make_cube());
            sphere_ = create_mesh(make_sphere(48, 96));
            ground_ = create_mesh(make_ground(kGroundCells, kGroundHalf));
            mgl::glGenVertexArrays(1, &empty_vao_);  // attribute-less draws (gl_VertexID) still want a VAO bound

            // ---- the shadow map: a depth texture that is the only attachment of an FBO -------------------
            // Immutable storage with a sized depth format (ES 3.0 3.8.4, table 3.14). DEPTH_COMPONENT24 is
            // texture-renderable in every ES 3.0 implementation.
            mgl::glGenTextures(1, &shadow_tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::DepthComponent24, kShadowSize, kShadowSize);
            // The initial MIN filter NEAREST_MIPMAP_LINEAR would make the single-level texture incomplete (3.8.13).
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            // Depth comparison (3.8.15): the shader's reference value is compared with the stored depth.
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::CompareFunc::Lequal);

            mgl::glGenFramebuffers(1, &shadow_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, shadow_fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                        mgl::TextureTarget::Texture2D, shadow_tex_, 0);
            // No color attachment: nothing is drawn to, and nothing may be read from, a color buffer.
            // The initial DRAW_BUFFER0 is COLOR_ATTACHMENT0 and the initial READ_BUFFER too; both point at an
            // attachment that does not exist here, so state it explicitly (ES 3.0 4.2.1, 4.3.1).
            const mgl::DrawBuffer none = mgl::DrawBuffer::None;
            mgl::glDrawBuffers(1, &none);
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("the depth-only shadow framebuffer is incomplete");

            // A sampler object with the comparison turned OFF and NEAREST filtering, bound to unit 1. Its state
            // supersedes the texture's (3.8.2), so the same depth texture can be read as raw depth on unit 1
            // while unit 0 (no sampler bound) still compares. NEAREST is required: a depth texture without
            // comparison and with a LINEAR filter is incomplete (3.8.13).
            mgl::glGenSamplers(1, &raw_sampler_);
            mgl::glSamplerParameteri(raw_sampler_, mgl::SamplerParameter::MinFilter, GL_NEAREST);
            mgl::glSamplerParameteri(raw_sampler_, mgl::SamplerParameter::MagFilter, GL_NEAREST);
            mgl::glSamplerParameteri(raw_sampler_, mgl::SamplerParameter::WrapS, GL_CLAMP_TO_EDGE);
            mgl::glSamplerParameteri(raw_sampler_, mgl::SamplerParameter::WrapT, GL_CLAMP_TO_EDGE);
            mgl::glSamplerParameteri(raw_sampler_, mgl::SamplerParameter::CompareMode, GL_NONE);

            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
            mgl::glBindSampler(1, raw_sampler_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);

            // ---- the read-back target: 16 x 1 RGBA8 ---------------------------------------------------------
            mgl::glGenTextures(1, &dump_tex_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, dump_tex_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, kDumpTexels, 1);
            mgl::glGenFramebuffers(1, &dump_fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, dump_fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2D, dump_tex_, 0);
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("the depth read-back framebuffer is incomplete");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void render(demo::Env& env, int frame) override
        {
            if (paused_) ++paused_frames_;
            const double t = static_cast<double>(frame - paused_frames_) / 60.0;
            model_ = make_model(t, cam_yaw_, zoom_, env.width, env.height);
            draw_frame(env, model_, params_);
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
                case 'p': params_.pcf = (params_.pcf + 1) % 3; break;
                case 'b': params_.bias = !params_.bias; break;
                case ' ': paused_ = !paused_; break;
                case 'a': cam_yaw_ -= 0.08; break;
                case 'd': cam_yaw_ += 0.08; break;
                case 'w': zoom_ = std::max(0.5, zoom_ - 0.05); break;
                case 's': zoom_ = std::min(2.0, zoom_ + 0.05); break;
                default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;  // defined below
        void shutdown(demo::Env& env) override;                        // defined below

    private:
        static constexpr int kDumpTexels = 16;

        // ---- mesh upload ---------------------------------------------------------------------------------
        static Mesh create_mesh(const MeshData& data)
        {
            Mesh m;
            m.count = static_cast<GLsizei>(data.indices.size());
            mgl::glGenVertexArrays(1, &m.vao);
            mgl::glBindVertexArray(m.vao);
            mgl::glGenBuffers(1, &m.vbo);
            mgl::glBindBuffer(mgl::BufferTarget::Array, m.vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(data.vertices), mgl::BufferUsage::StaticDraw);
            mgl::glGenBuffers(1, &m.ibo);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, m.ibo);  // the element buffer binding IS vertex-array state
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(data.indices), mgl::BufferUsage::StaticDraw);
            const mgl::AttribLocation a_position{0u}, a_normal{1u};
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, px)));
            mgl::glEnableVertexAttribArray(a_normal);
            mgl::glVertexAttribPointer(a_normal, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, nx)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            return m;
        }

        static void destroy_mesh(Mesh& m)
        {
            mgl::glDeleteVertexArrays(1, &m.vao);
            mgl::glDeleteBuffers(1, &m.vbo);
            mgl::glDeleteBuffers(1, &m.ibo);
        }

        static void draw_mesh(const Mesh& m)
        {
            mgl::glBindVertexArray(m.vao);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, m.count, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
        }

        // ---- matrices (float, for the GPU) ----------------------------------------------------------------
        static dm::Vec3 to_vec3(V3 v) { return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }

        static dm::Mat4 shape_matrix(const Shape& s)
        {
            if (s.kind == Kind::Sphere)
                return dm::translate(to_vec3(s.center)) * dm::scale({static_cast<float>(s.radius), static_cast<float>(s.radius), static_cast<float>(s.radius)});
            return dm::translate(to_vec3(s.center)) * dm::rotate(static_cast<float>(s.yaw), {0, 1, 0}) *
                   dm::scale({static_cast<float>(2 * s.half.x), static_cast<float>(2 * s.half.y), static_cast<float>(2 * s.half.z)});
        }

        static dm::Mat4 light_matrix(const LightFrame& f)
        {
            const float h = static_cast<float>(kLightHalf);
            return dm::ortho(-h, h, -h, h, static_cast<float>(kLightNear), static_cast<float>(kLightFar)) *
                   dm::look_at(to_vec3(f.eye), {0, 0, 0}, {0, 1, 0});
        }

        static dm::Mat4 view_proj_matrix(const Camera& c)
        {
            return dm::perspective(static_cast<float>(radians(kFovY)), static_cast<float>(c.aspect), static_cast<float>(kCamNear),
                                   static_cast<float>(kCamFar)) *
                   dm::look_at(to_vec3(c.eye), to_vec3(c.target), {0, 1, 0});
        }

        // ---- pass 1: the shadow map ---------------------------------------------------------------------------
        void render_shadow_pass(const Model& m, bool bias)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, shadow_fbo_);
            mgl::glViewport(0, 0, kShadowSize, kShadowSize);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_TRUE);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Depth);

            if (bias)
            {
                // Depth offset (ES 3.0 3.6.2): o = m * factor + r * units, with m the polygon's depth slope. Every
                // stored depth moves AWAY from the light by o, so the surface no longer shadows itself.
                mgl::glEnable(mgl::Capability::PolygonOffsetFill);
                mgl::glPolygonOffset(kOffsetFactor, kOffsetUnits);
            }

            mgl::glUseProgram(shadow_.id);
            const dm::Mat4 light_vp = light_matrix(m.light);
            mgl::glUniformMatrix4fv(shadow_.light_vp, 1, GL_FALSE, light_vp.data());
            const dm::Mat4 identity = dm::Mat4::identity();
            mgl::glUniformMatrix4fv(shadow_.model, 1, GL_FALSE, identity.data());
            draw_mesh(ground_);
            for (const Shape& s : m.shapes)
            {
                const dm::Mat4 model = shape_matrix(s);
                mgl::glUniformMatrix4fv(shadow_.model, 1, GL_FALSE, model.data());
                draw_mesh(s.kind == Kind::Box ? cube_ : sphere_);
            }
            mgl::glDisable(mgl::Capability::PolygonOffsetFill);
        }

        // ---- pass 2: the camera view + the inset --------------------------------------------------------------
        void set_material(const Material& mat) const
        {
            mgl::glUniform3f(scene_.albedo, static_cast<float>(mat.albedo.x), static_cast<float>(mat.albedo.y), static_cast<float>(mat.albedo.z));
            mgl::glUniform3f(scene_.albedo2, static_cast<float>(mat.albedo2.x), static_cast<float>(mat.albedo2.y), static_cast<float>(mat.albedo2.z));
            mgl::glUniform1f(scene_.checker, mat.checker ? 1.0f : 0.0f);
            mgl::glUniform1f(scene_.specular, static_cast<float>(mat.specular));
            mgl::glUniform1f(scene_.shininess, static_cast<float>(mat.shininess));
        }

        void draw_lit(const Mesh& mesh, const dm::Mat4& model, const Material& mat) const
        {
            mgl::glUniformMatrix4fv(scene_.model, 1, GL_FALSE, model.data());
            const std::array<float, 9> nm = dm::normal_matrix(model);
            mgl::glUniformMatrix3fv(scene_.normal_matrix, 1, GL_FALSE, nm.data());
            set_material(mat);
            draw_mesh(mesh);
        }

        void render_main_pass(const demo::Env& env, const Model& m, const RenderParams& p)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            const V3 sky = gamma(kFogColor);  // the horizon fades into the fog color, so the background is the fog color
            mgl::glClearColor(static_cast<float>(sky.x), static_cast<float>(sky.y), static_cast<float>(sky.z), 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);

            // The PCF mode is texture state of the shadow map: NEAREST gives one hard comparison, LINEAR lets the
            // hardware blend the comparison results of the four nearest texels (3.8.15.1).
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
            const bool linear = p.pcf != 0;
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, linear ? mgl::TextureMinFilter::Linear : mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, linear ? mgl::TextureMagFilter::Linear : mgl::TextureMagFilter::Nearest);

            mgl::glUseProgram(scene_.id);
            const dm::Mat4 vp = view_proj_matrix(m.cam);
            const dm::Mat4 lvp = light_matrix(m.light);
            mgl::glUniformMatrix4fv(scene_.view_proj, 1, GL_FALSE, vp.data());
            mgl::glUniformMatrix4fv(scene_.light_vp, 1, GL_FALSE, lvp.data());
            mgl::glUniform1i(scene_.shadow, 0);
            mgl::glUniform1f(scene_.texel, 1.0f / static_cast<float>(kShadowSize));
            mgl::glUniform1f(scene_.bias, p.bias ? kReceiverBias : 0.0f);
            mgl::glUniform1i(scene_.pcf_radius, p.pcf == 2 ? 1 : 0);
            mgl::glUniform3f(scene_.eye, static_cast<float>(m.cam.eye.x), static_cast<float>(m.cam.eye.y), static_cast<float>(m.cam.eye.z));
            mgl::glUniform3f(scene_.light_dir, static_cast<float>(m.light.L.x), static_cast<float>(m.light.L.y), static_cast<float>(m.light.L.z));
            mgl::glUniform3f(scene_.light_color, static_cast<float>(kLightColor.x), static_cast<float>(kLightColor.y), static_cast<float>(kLightColor.z));
            mgl::glUniform3f(scene_.sky_ambient, static_cast<float>(kSkyAmbient.x), static_cast<float>(kSkyAmbient.y), static_cast<float>(kSkyAmbient.z));
            mgl::glUniform3f(scene_.ground_ambient, static_cast<float>(kGroundAmbient.x), static_cast<float>(kGroundAmbient.y), static_cast<float>(kGroundAmbient.z));
            mgl::glUniform3f(scene_.fog_color, static_cast<float>(kFogColor.x), static_cast<float>(kFogColor.y), static_cast<float>(kFogColor.z));
            mgl::glUniform1f(scene_.fog_density, static_cast<float>(kFogDensity));

            draw_lit(ground_, dm::Mat4::identity(), kGroundMaterial);
            for (const Shape& s : m.shapes) draw_lit(s.kind == Kind::Box ? cube_ : sphere_, shape_matrix(s), s.mat);

            // The inset: the shadow map as a gray ramp, on texture unit 1 (comparison NONE through the sampler object).
            const InsetRect r = inset_rect(env.width, env.height);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glViewport(r.x, r.y, r.size, r.size);
            mgl::glUseProgram(inset_program_);
            mgl::glUniform1i(inset_depth_, 1);
            mgl::glUniform2f(inset_range_, static_cast<float>(kInsetLo), static_cast<float>(kInsetHi));
            mgl::glUniform1f(inset_bands_, static_cast<float>(kInsetBands));
            mgl::glUniform2f(inset_view_, static_cast<float>(kInsetView.x), static_cast<float>(kInsetView.y));
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void draw_frame(const demo::Env& env, const Model& m, const RenderParams& p)
        {
            render_shadow_pass(m, p.bias);
            render_main_pass(env, m, p);
        }

        // ---- validation (implemented after the class) ---------------------------------------------------------
        std::vector<std::uint8_t> render_and_read(const demo::Env& env, const RenderParams& p)
        {
            draw_frame(env, model_, p);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return demo::read_rgba8(0, 0, env.width, env.height);
        }

        std::vector<double> read_depth_texels(const std::vector<std::array<int, 2>>& texels);
        void check_pipeline_state(demo::Check& check);
        void check_shadow_map_depths(demo::Check& check, const Model& m);
        void check_inset(const demo::Env& env, demo::Check& check, const Model& m);
        void check_images(const demo::Env& env, demo::Check& check, const Model& m);

        // state
        ShadowProgram shadow_{};
        SceneProgram scene_{};
        mgl::ProgramId inset_program_{}, dump_program_{};
        mgl::UniformLocation inset_depth_{}, inset_range_{}, inset_bands_{}, inset_view_{}, dump_depth_{}, dump_texels_{};
        Mesh cube_{}, sphere_{}, ground_{};
        mgl::VertexArrayId empty_vao_{};
        mgl::TextureId shadow_tex_{}, dump_tex_{};
        mgl::FramebufferId shadow_fbo_{}, dump_fbo_{};
        mgl::SamplerId raw_sampler_{};

        RenderParams params_{};
        Model model_{};
        bool paused_ = false;
        int paused_frames_ = 0;
        double cam_yaw_ = 0.0, zoom_ = 1.0;
    };

    // ---------------------------------------------------------------------------------------------------
    void ShadowMapping::shutdown(demo::Env& /*env*/)
    {
        mgl::glBindSampler(1, mgl::SamplerId{0});
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glUseProgram(mgl::ProgramId{0});
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glDeleteFramebuffers(1, &shadow_fbo_);
        mgl::glDeleteFramebuffers(1, &dump_fbo_);
        mgl::glDeleteTextures(1, &shadow_tex_);
        mgl::glDeleteTextures(1, &dump_tex_);
        mgl::glDeleteSamplers(1, &raw_sampler_);
        destroy_mesh(cube_);
        destroy_mesh(sphere_);
        destroy_mesh(ground_);
        mgl::glDeleteVertexArrays(1, &empty_vao_);
        mgl::glDeleteProgram(shadow_.id);
        mgl::glDeleteProgram(scene_.id);
        mgl::glDeleteProgram(inset_program_);
        mgl::glDeleteProgram(dump_program_);
    }

    std::vector<double> ShadowMapping::read_depth_texels(const std::vector<std::array<int, 2>>& texels)
    {
        std::array<GLint, 2 * kDumpTexels> list{};
        for (int i = 0; i < kDumpTexels; ++i)
        {
            const auto& t = texels[std::min(static_cast<std::size_t>(i), texels.size() - 1)];
            list[static_cast<std::size_t>(2 * i)] = t[0];
            list[static_cast<std::size_t>(2 * i + 1)] = t[1];
        }
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, dump_fbo_);
        mgl::glViewport(0, 0, kDumpTexels, 1);
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glDisable(mgl::Capability::CullFace);
        mgl::glUseProgram(dump_program_);
        mgl::glUniform1i(dump_depth_, 1);
        mgl::glUniform2iv(dump_texels_, kDumpTexels, list.data());
        mgl::glBindVertexArray(empty_vao_);
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, kDumpTexels, 1);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

        std::vector<double> depth;
        for (std::size_t i = 0; i < texels.size() && i < static_cast<std::size_t>(kDumpTexels); ++i)
        {
            const std::uint32_t v = (static_cast<std::uint32_t>(px[4 * i]) << 16) | (static_cast<std::uint32_t>(px[4 * i + 1]) << 8) |
                                    static_cast<std::uint32_t>(px[4 * i + 2]);
            depth.push_back(static_cast<double>(v) / 16777215.0);
        }
        return depth;
    }

    // =================================================================================================
    // Validation
    // =================================================================================================

    /// State of the depth-only pipeline: framebuffer completeness, attachments, draw/read buffers, the
    /// texture's and the sampler object's comparison state, and the defined errors of a color-less framebuffer.
    void ShadowMapping::check_pipeline_state(demo::Check& check)
    {
        GLint v = 0;
        (void)demo::drain_gl_errors();
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, shadow_fbo_);
        check.expect(mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete,
                     "shadow framebuffer with only a depth texture attached is complete (ES 3.0 4.4.4.2)");

        mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                                   mgl::FramebufferAttachmentParameter::ObjectType, &v);
        check.expect_eq(v, static_cast<GLint>(GL_TEXTURE), "DEPTH_ATTACHMENT object type is TEXTURE");
        mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                                   mgl::FramebufferAttachmentParameter::ObjectName, &v);
        check.expect_eq(v, static_cast<GLint>(shadow_tex_.value), "DEPTH_ATTACHMENT object name is the shadow texture");
        mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                                   mgl::FramebufferAttachmentParameter::DepthSize, &v);
        check.expect(v >= 24, "the depth attachment has at least 24 depth bits (DEPTH_COMPONENT24)", std::format("DEPTH_SIZE = {}", v));
        mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                                   mgl::FramebufferAttachmentParameter::ObjectType, &v);
        check.expect_eq(v, static_cast<GLint>(GL_NONE), "COLOR_ATTACHMENT0 of the shadow framebuffer has no image");
        check.expect_eq(demo::get_integer(mgl::GetParameter::DrawBuffer0), static_cast<int>(GL_NONE), "DRAW_BUFFER0 of the shadow framebuffer is NONE");
        check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), static_cast<int>(GL_NONE), "READ_BUFFER of the shadow framebuffer is NONE");
        check.expect_no_gl_error("no GL error while querying the framebuffer");

        // Defined errors (ES 3.0 4.2.1, 4.3.1, 4.3.2.1) - the state must also survive them.
        std::array<std::uint8_t, 4> px{};
        mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glReadPixels from the color-less framebuffer: INVALID_OPERATION (READ_BUFFER is NONE, 4.3.2.1)");
        const mgl::DrawBuffer color1 = mgl::to_draw_buffer(mgl::ColorAttachment::Color1);
        mgl::glDrawBuffers(1, &color1);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDrawBuffers(COLOR_ATTACHMENT1) at index 0 of a framebuffer object: INVALID_OPERATION (4.2.1)");
        mgl::glReadBuffer(mgl::ReadBuffer::Back);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glReadBuffer(BACK) with a framebuffer object bound: INVALID_OPERATION (4.3.1)");
        check.expect(demo::get_integer(mgl::GetParameter::DrawBuffer0) == static_cast<int>(GL_NONE) &&
                         demo::get_integer(mgl::GetParameter::ReadBuffer) == static_cast<int>(GL_NONE),
                     "a failed glDrawBuffers / glReadBuffer leaves DRAW_BUFFER0 and READ_BUFFER at NONE");
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

        // Comparison state: on the texture (unit 0) and, overriding it, on the sampler object (unit 1).
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
        mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::CompareMode, &v);
        check.expect_eq(v, static_cast<GLint>(GL_COMPARE_REF_TO_TEXTURE), "shadow texture: TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE");
        mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::CompareFunc, &v);
        check.expect_eq(v, static_cast<GLint>(GL_LEQUAL), "shadow texture: TEXTURE_COMPARE_FUNC = LEQUAL");
        mgl::glGetSamplerParameteriv(raw_sampler_, mgl::SamplerParameter::CompareMode, &v);
        check.expect_eq(v, static_cast<GLint>(GL_NONE), "raw-depth sampler object: TEXTURE_COMPARE_MODE = NONE");
        mgl::glActiveTexture(mgl::TextureUnit::Texture1);
        check.expect_eq(demo::get_integer(mgl::GetParameter::SamplerBinding), static_cast<int>(raw_sampler_.value), "unit 1 has the raw-depth sampler bound (SAMPLER_BINDING)");
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        check.expect_eq(demo::get_integer(mgl::GetParameter::SamplerBinding), 0, "unit 0 has no sampler bound, so the texture's own comparison state is used");
        check.expect_no_gl_error("no GL error from the state queries");
    }

    /// The shadow map's contents: depth texels read back through the dump pass against analytic depths, and the
    /// measured effect of glPolygonOffset.
    void ShadowMapping::check_shadow_map_depths(demo::Check& check, const Model& m)
    {
        const std::vector<TexelTarget> targets = pick_targets(m);
        std::vector<std::array<int, 2>> texels;
        int n_ground = 0, n_box = 0, n_sphere = 0;
        for (const TexelTarget& t : targets)
        {
            texels.push_back({t.tx, t.ty});
            if (t.id == -1) ++n_ground;
            else if (m.shapes[static_cast<std::size_t>(t.id)].kind == Kind::Box) ++n_box;
            else ++n_sphere;
        }
        check.expect(n_ground >= 3 && n_box >= 3 && n_sphere >= 2, "depth read-back covers open ground, box tops and sphere surfaces",
                     std::format("{} texels: {} ground, {} box, {} sphere", targets.size(), n_ground, n_box, n_sphere));
        if (targets.empty()) return;

        // Without bias the map holds exactly the analytic depths (tolerances: see README).
        constexpr double kPlanarTolerance = 5e-6;   // sanity bound; D24 spacing is 6e-8, the rest is rasterizer interpolation
        constexpr double kSphereTolerance = 1e-4;   // inscribed facets: sag rho^2/(2r) <= 9.6e-4 units (rho = half a cell's hypotenuse, r = 0.9) = 3.2e-5 depth, plus the ray's slant
        render_shadow_pass(m, false);
        const std::vector<double> off = read_depth_texels(texels);
        double worst_planar = 0, worst_sphere = 0;
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            const bool sphere = targets[i].id >= 0 && m.shapes[static_cast<std::size_t>(targets[i].id)].kind == Kind::Sphere;
            double& worst = sphere ? worst_sphere : worst_planar;
            worst = std::max(worst, std::abs(off[i] - targets[i].expected));
        }
        check.expect(worst_planar <= kPlanarTolerance, "shadow map depth at ground and box-top texels equals the analytic window-space depth",
                     std::format("max error {:.3e} (tolerance {:.0e}) over {} texels", worst_planar, kPlanarTolerance, n_ground + n_box));
        check.expect(worst_sphere <= kSphereTolerance, "shadow map depth at sphere texels equals the analytic depth of the sphere",
                     std::format("max error {:.3e} (tolerance {:.0e}) over {} texels", worst_sphere, kSphereTolerance, n_sphere));

        // Occluders must be nearer to the light than the ground behind them: the box-top depths are below the depth of the
        // ground plane at the same texel (the depth test of pass 1 kept the nearer fragment).
        bool nearer = true;
        for (std::size_t i = 0; i < targets.size(); ++i)
            if (targets[i].id >= 0 && off[i] >= ground_depth(m, targets[i].tx, targets[i].ty)) nearer = false;
        check.expect(nearer, "every texel covered by a shape holds a smaller depth than the ground plane at that texel (depth test, pass 1)");

        // glPolygonOffset: the same texels with the bias on. o = m * factor + r * units (3.6.2, eq. 3.11) with m the depth
        // slope of the polygon in window units per texel; the spec allows m = sqrt(gx^2 + gy^2) or max(|gx|, |gy|).
        render_shadow_pass(m, true);
        const std::vector<double> on = read_depth_texels(texels);
        bool within = true, never_toward = true;
        double lo_ratio = 1e9, hi_ratio = 0;
        const double factor_d = kOffsetFactor, units_d = kOffsetUnits;
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            const double offset = on[i] - off[i];
            if (offset < -kPlanarTolerance) never_toward = false;
            if (targets[i].id != -1) continue;
            const int tx = targets[i].tx, ty = targets[i].ty;
            const double gx = (ground_depth(m, tx + 1, ty) - ground_depth(m, tx - 1, ty)) / 2.0;
            const double gy = (ground_depth(m, tx, ty + 1) - ground_depth(m, tx, ty - 1)) / 2.0;
            const double m_max = std::max(std::abs(gx), std::abs(gy)), m_euclid = std::hypot(gx, gy);
            const double lo = 0.9 * factor_d * m_max, hi = 1.1 * factor_d * m_euclid + 4.0 * units_d * std::ldexp(1.0, -24);
            if (offset < lo || offset > hi) within = false;
            const double ratio = offset / (factor_d * m_euclid);
            lo_ratio = std::min(lo_ratio, ratio);
            hi_ratio = std::max(hi_ratio, ratio);
        }
        check.expect(within, "glPolygonOffset pushes the stored ground depth away from the light by factor * m (+ units * r)",
                     std::format("measured offset / (factor * m_euclid) = {:.3f} .. {:.3f}; allowed {:.2f} .. 1.1 (m may be max(|gx|,|gy|))",
                                 lo_ratio, hi_ratio, 0.9 / std::sqrt(2.0)));
        check.expect(never_toward, "a positive polygon offset never moves a stored depth toward the light");
        check.expect(!mgl::glIsEnabled(mgl::Capability::PolygonOffsetFill), "POLYGON_OFFSET_FILL is disabled again after pass 1");
        GLfloat factor = -1, units = -1;
        mgl::glGetFloatv(mgl::GetParameter::PolygonOffsetFactor, &factor);
        mgl::glGetFloatv(mgl::GetParameter::PolygonOffsetUnits, &units);
        check.expect(factor == kOffsetFactor && units == kOffsetUnits, "POLYGON_OFFSET_FACTOR / UNITS read back what glPolygonOffset set",
                     std::format("factor {} units {}", factor, units));
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
    }

    /// The inset: depth sampled as a plain sampler2D through the NONE-mode sampler object, drawn as gray values, against the CPU
    /// model's analytic depth passed through the same display function. Rendered without bias so the stored depths are exact.
    void ShadowMapping::check_inset(const demo::Env& env, demo::Check& check, const Model& m)
    {
        const std::vector<std::uint8_t> img = render_and_read(env, RenderParams{false, 0});
        const InsetRect r = inset_rect(env.width, env.height);
        int checked = 0, on_objects = 0, bad = 0, worst = 0;
        std::string first;
        constexpr int kGrid = 20;
        for (int j = 0; j < kGrid; ++j)
            for (int i = 0; i < kGrid; ++i)
            {
                const int ix = static_cast<int>(r.size * (0.08 + 0.84 * (i + 0.5) / kGrid)), iy = static_cast<int>(r.size * (0.08 + 0.84 * (j + 0.5) / kGrid));
                const double u = (ix + 0.5) / r.size, v = (iy + 0.5) / r.size;
                const double fx = (kInsetView.x + u * kInsetView.y) * kShadowSize, fy = (kInsetView.x + v * kInsetView.y) * kShadowSize;
                const int tx = static_cast<int>(std::floor(fx)), ty = static_cast<int>(std::floor(fy));
                if (fx - tx < 0.1 || fx - tx > 0.9 || fy - ty < 0.1 || fy - ty > 0.9) continue;  // nearest-texel ties
                const Hit h = light_hit(m, tx, ty);
                if (!h.ok || !light_neighbourhood_uniform(m, tx, ty, h.id, 2)) continue;     // depth edges
                const double d = depth_from_t(h.t);
                const double band = d * kInsetBands - std::floor(d * kInsetBands);
                if (band < 0.1 || band > 0.9) continue;                                      // sawtooth discontinuity
                const double ramp = std::clamp((d - kInsetLo) / (kInsetHi - kInsetLo), 0.0, 1.0);
                const int expected = static_cast<int>(std::lround(255.0 * (0.25 * ramp + 0.75 * band)));
                const std::uint8_t* px = &img[(static_cast<std::size_t>(r.y + iy) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(r.x + ix)) * 4];
                int d8 = 0;
                for (int c = 0; c < 3; ++c) d8 = std::max(d8, std::abs(static_cast<int>(px[c]) - expected));
                ++checked;
                if (h.id >= 0) ++on_objects;
                worst = std::max(worst, d8);
                if (d8 > 2)
                {
                    ++bad;
                    if (first.empty()) first = std::format("first at inset pixel ({},{}): got {} expected {}", ix, iy, px[0], expected);
                }
            }
        check.expect(checked >= 40 && on_objects >= 3, "inset samples cover enough ground and object texels",
                     std::format("{} pixels checked, {} over shapes", checked, on_objects));
        check.expect(bad == 0, "the inset shows the stored depth: raw depth through the NONE-mode sampler object matches the CPU model (+-2 levels)",
                     std::format("{} of {} pixels off, worst {} levels{}{}", bad, checked, worst, first.empty() ? "" : "; ", first));
    }

    /// The camera image: lit and shadowed pixels against the CPU model for every PCF mode, penumbra behavior, shadow acne
    /// without bias, and the negative controls.
    void ShadowMapping::check_images(const demo::Env& env, demo::Check& check, const Model& m)
    {
        const ProbeSet probes = build_probes(m);
        check.expect(probes.ground_lit >= 100 && probes.ground_shadow >= 15 && probes.object_lit >= 20 && probes.object_shadow >= 20 && probes.back >= 50,
                     "probe set covers lit and shadowed ground, lit and shadowed object surfaces, and faces turned away from the light",
                     std::format("ground lit {} / shadowed {}, object lit {} / shadowed {}, facing away {}; {} penumbra-zone ground pixels held out",
                                 probes.ground_lit, probes.ground_shadow, probes.object_lit, probes.object_shadow, probes.back, probes.edge.size()));
        int min_sep = 255;
        for (const Probe& p : probes.solid)
            if (p.ndl >= kMinNdl) min_sep = std::min(min_sep, max_diff(p.lit.data(), p.shadowed));
        check.expect(min_sep >= 3 * kTolerance, "lit and shadowed colors differ by far more than the tolerance at every compared pixel",
                     std::format("smallest lit/shadowed difference {} levels, tolerance {}", min_sep, kTolerance));

        // Every PCF mode, bias on: lit pixels fully lit (no acne), shadowed pixels ambient only.
        std::array<Penumbra, 3> pen{};
        std::vector<std::uint8_t> default_image;
        for (int mode = 0; mode < 3; ++mode)
        {
            const std::vector<std::uint8_t> img = render_and_read(env, RenderParams{true, mode});
            if (mode == 2) default_image = img;
            const Tally t = evaluate(probes.solid, img, env.width);
            pen[static_cast<std::size_t>(mode)] = classify_edge(probes.edge, img, env.width);
            check.expect(t.mismatches == 0, std::format("PCF {}: {} pixels far from shadow edges match the CPU lighting model", kPcfNames[static_cast<std::size_t>(mode)], t.checked),
                         std::format("{} mismatches, worst {} levels (tolerance {}){}{}", t.mismatches, t.worst, kTolerance, t.first.empty() ? "" : "; ", t.first));
        }


        // The background: pixels whose camera ray (and the rays of their neighbours) leave the scene show the clear color.
        {
            const demo::Rgba8 expected = to_rgba8(gamma(kFogColor));
            int n = 0, bad = 0;
            const InsetRect inset = inset_rect(m.w, m.h);
            for (int y = 3; y < m.h - 3; y += 3)
                for (int x = 3; x < m.w - 3; x += 3)
                {
                    if (x <= inset.x + inset.size + 3 && y >= inset.y - 3) continue;  // under the inset
                    bool all_sky = true;
                    for (int dy = -2; dy <= 2 && all_sky; dy += 2)
                        for (int dx = -2; dx <= 2 && all_sky; dx += 2)
                            all_sky = !trace(m.shapes, m.cam.eye, pixel_dir(m.cam, m.w, m.h, x + dx, y + dy)).ok;
                    if (!all_sky) continue;
                    const std::uint8_t* px = &default_image[(static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4];
                    ++n;
                    if (max_diff(px, expected) > 1) ++bad;
                }
            check.expect(n >= 20 && bad == 0, "pixels above the horizon show the clear color (glClearColor + glClear)",
                         std::format("{} sky pixels, {} off by more than 1 level, expected {}", n, bad, demo::format_rgba(expected)));
        }

        // Penumbra: a single NEAREST comparison can only return 0 or 1, so every pixel is fully lit or fully shadowed
        // (3.8.15.1). With LINEAR filtering the result "may" mix several comparisons - implementation-dependent weights - and the
        // 3x3 shader taps average nine such results, so intermediate values must appear along the edge.
        check.expect(pen[0].pixels >= 40 && pen[0].between == 0 && pen[0].other == 0, "PCF off: ground pixels in the shadow-edge zone are only fully lit or fully shadowed",
                     std::format("{} pixels: {} lit, {} shadowed, {} in between, {} other", pen[0].pixels, pen[0].lit, pen[0].shadow, pen[0].between, pen[0].other));
        check.expect(pen[2].between >= 5 && pen[2].other == 0, "PCF hardware + 3x3 taps: a soft penumbra of strictly intermediate values appears along shadow edges",
                     std::format("{} pixels: {} lit, {} shadowed, {} in between, {} other", pen[2].pixels, pen[2].lit, pen[2].shadow, pen[2].between, pen[2].other));
        check.expect(pen[1].other == 0, "PCF hardware (LINEAR): every edge-zone pixel is lit, shadowed or a mix of the two (weights are implementation-dependent)",
                     std::format("{} pixels: {} lit, {} shadowed, {} in between, {} other", pen[1].pixels, pen[1].lit, pen[1].shadow, pen[1].between, pen[1].other));

        // TEXTURE_COMPARE_FUNC (Table 3.23): the shader is unchanged, only the texture parameter moves. ALWAYS returns 1 for every lookup, so
        // nothing is shadowed; NEVER returns 0, so every surface facing the light is shadowed; GREATER is the opposite test of LEQUAL
        // (Dref > Dt), so shadowed and lit swap places (the bias keeps every lit point well clear of its stored depth, 3.8.15.1).
        {
            struct Case
            {
                mgl::CompareFunc func;
                const char* name;
                Expect expect;
            };
            const Case cases[] = {{mgl::CompareFunc::Always, "ALWAYS: no pixel is shadowed", Expect::AllLit},
                                  {mgl::CompareFunc::Never, "NEVER: every pixel facing the light is shadowed", Expect::AllShadowed},
                                  {mgl::CompareFunc::Greater, "GREATER: shadowed and lit swap places", Expect::Inverted}};
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
            std::vector<Probe> in_map;  // the shader treats everything outside the light's frustum as lit, whatever the function
            for (const Probe& p : probes.solid)
                if (p.in_map) in_map.push_back(p);
            for (const Case& c : cases)
            {
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, c.func);
                const std::vector<std::uint8_t> img = render_and_read(env, RenderParams{true, 0});
                const Tally t = evaluate(in_map, img, env.width, nullptr, c.expect);
                check.expect(t.mismatches == 0, std::format("TEXTURE_COMPARE_FUNC {} ({} pixels)", c.name, t.checked),
                             std::format("{} mismatches, worst {} levels{}{}", t.mismatches, t.worst, t.first.empty() ? "" : "; ", t.first));
            }
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::CompareFunc::Lequal);
            GLint func = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::CompareFunc, &func);
            check.expect_eq(func, static_cast<GLint>(GL_LEQUAL), "TEXTURE_COMPARE_FUNC restored to LEQUAL");
        }

        // Shadow acne: without any bias a lit surface shadows itself. On the ground plane the CPU can predict it texel by texel:
        // pass 1 stores the plane's depth at the texel CENTER, pass 2 compares the depth at the fragment's own position, which is
        // larger on one side of the center (the plane is tilted against the light), so about half of the ground fails LEQUAL.
        {
            const std::vector<std::uint8_t> img = render_and_read(env, RenderParams{false, 0});
            int n = 0, agree = 0, darkened = 0, other = 0;
            for (const Probe& p : probes.solid)
            {
                if (p.id != -1 || p.occluded || !p.in_map) continue;
                const V3 uv = to_light_uv(m.light, p.p);
                const double fx = uv.x * kShadowSize, fy = uv.y * kShadowSize;
                const int tx = static_cast<int>(std::floor(fx)), ty = static_cast<int>(std::floor(fy));
                if (fx - tx < 0.03 || fx - tx > 0.97 || fy - ty < 0.03 || fy - ty > 0.97) continue;  // which texel NEAREST picks is a tie
                const double delta = uv.z - ground_depth(m, tx, ty);                                  // reference minus stored depth
                if (std::abs(delta) < 1e-5) continue;                                                   // within depth/interpolation noise
                const std::uint8_t* px = &img[(static_cast<std::size_t>(p.y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(p.x)) * 4];
                const bool dark = max_diff(px, p.shadowed) <= kTolerance, bright = max_diff(px, p.lit) <= kTolerance;
                ++n;
                if (dark) ++darkened;
                if (!dark && !bright) ++other;
                if ((delta > 0.0) == dark && (dark || bright)) ++agree;
            }
            check.expect(n >= 100 && agree == n, "without bias the CPU predicts, pixel by pixel, which lit ground pixels turn dark (shadow acne)",
                         std::format("{} pixels agree of {} (reference > stored depth means shadowed), {} neither lit nor shadowed", agree, n, other));
            check.expect(n > 0 && darkened * 4 >= n, "negative control: with the bias off at least a quarter of the lit ground pixels are wrongly shadowed (the lit check can fail)",
                         std::format("{} of {} lit ground pixels darkened ({:.1f} %), the analytic expectation is about 50 %", darkened, n,
                                     n > 0 ? 100.0 * darkened / n : 0.0));
        }

        // Negative controls for the shadow-model comparison itself: the image above (bias on, PCF 3x3) must NOT agree with a CPU model
        // whose light direction is rotated, or whose occluders are shifted.
        {
            const LightFrame rotated = make_light(m.light_azimuth + 0.25);
            const ShadowModel rotated_light{&m.shapes, rotated.L};
            std::vector<Shape> moved = m.shapes;
            for (Shape& s : moved) s.center = s.center + V3{0.6, 0.0, 0.4};
            const ShadowModel moved_occluders{&moved, m.light.L};
            const Tally a = evaluate(probes.solid, default_image, env.width, &rotated_light);
            const Tally b = evaluate(probes.solid, default_image, env.width, &moved_occluders);
            const int floor_mismatches = std::max(20, static_cast<int>(probes.solid.size() / 100));  // at least 1 % of the compared pixels, and 20
            check.expect(a.mismatches >= floor_mismatches, "negative control: a CPU model with the light direction rotated by 0.25 rad disagrees with the image",
                         std::format("{} of {} pixels mismatch (required: at least {})", a.mismatches, a.checked, floor_mismatches));
            check.expect(b.mismatches >= floor_mismatches, "negative control: a CPU model with every occluder shifted by (0.6, 0, 0.4) disagrees with the image",
                         std::format("{} of {} pixels mismatch (required: at least {})", b.mismatches, b.checked, floor_mismatches));
        }
    }

    void ShadowMapping::validate(demo::Env& env, demo::Check& check)
    {
        const Model m = model_;  // the scene of the last rendered frame
        check_pipeline_state(check);
        check_shadow_map_depths(check, m);
        check_inset(env, check, m);
        check_images(env, check, m);
        check.expect_no_gl_error("no GL error from the validation passes");
        draw_frame(env, m, params_);  // leave the default framebuffer showing the demo's own frame (it is captured next)
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-shadow-mapping",
        .title = "Shadow Mapping",
        .min_es = {3, 0},
        .summary = "A lit 3D scene with directional-light shadows: a depth-only FBO pass into a DEPTH_COMPONENT24 texture, "
                   "then a sampler2DShadow lookup with hardware PCF, polygon-offset bias and a shadow-map inset; "
                   "validated against an analytic CPU ray-cast model of the scene.",
        .controls = "P cycle PCF mode (off / hardware / hardware + 3x3), B bias on/off, Space pause, A/D orbit camera, W/S zoom",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 31,
    };
    return demo::run<ShadowMapping>(argc, argv, info);
}
