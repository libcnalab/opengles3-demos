// es30-post-processing
//
// A lit, animated 3D scene (orbiting camera, checkerboard floor, a torus, a sphere, two cubes, seven small
// spheres, two orbiting point lights with glowing bulbs, fog) is rendered into an off-screen framebuffer
// object - a color texture plus a depth attachment - and then post-processed with full-screen passes that read
// the result with texelFetch:
//
//   gamma        linear -> sRGB in the shader, and the same with a SRGB8_ALPHA8 scene target, where the
//                hardware encodes on write and decodes on every lookup
//   blur         separable Gaussian: a horizontal and a vertical pass, ping-pong between two targets
//   sobel        edge detection on the brightness of the scene
//   bloom        bright-pass -> separable blur -> add to the scene
//   dof          depth of field: the scene mixed with its blur by a circle of confusion from the depth texture
//
// The window shows an overview of all seven looks as vertical stripes (the default), or one of them
// full screen; see Info::controls.
//
// Validation (--validate, 256 x 256, after 45 frames): every offscreen image is read back with glReadPixels and the
// passes are recomputed on the CPU from exactly the bytes the GPU read (so 8-bit quantization between passes
// cannot accumulate): Gaussian blur (both directions, twice), bright-pass, Sobel, the depth texture (through a
// pass that writes it as 24-bit color), and all eight presentations including the image that is on screen.
// Scene pixels are checked analytically: the sky gradient, floor pixels (checker, Blinn-Phong with a
// directional and two point lights, fog), the lamp bulbs at their CPU-projected centers, and the depth of the
// floor. Negative controls show that the checks can fail: a blur with the wrong radius or a box kernel, a
// one-axis Sobel, a plain gamma-2.2 curve and an sRGB lookup that skips the decode all disagree with the GPU.

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
#include <iostream>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
namespace math = demo::math;

namespace
{
    constexpr int gl(GLenum e) { return static_cast<int>(e); }

    // ---------------------------------------------------------------------------------------------
    // Scene and effect parameters. The GPU gets these values as uniforms and the CPU reference uses the very
    // same numbers (as doubles), so the two cannot drift apart.
    // ---------------------------------------------------------------------------------------------
    constexpr float kFovY = math::radians(45.0f);  // vertical field of view of a landscape or square surface
    constexpr float kZNear = 0.5f;
    constexpr float kZFar = 80.0f;
    constexpr math::Vec3 kTarget{0.0f, 1.1f, 0.0f};  // the camera looks here

    // Linear-light colors.
    constexpr math::Vec3 kHorizon{0.080f, 0.058f, 0.105f};
    constexpr math::Vec3 kZenith{0.0012f, 0.0025f, 0.0100f};
    constexpr math::Vec3 kAmbient{0.035f, 0.040f, 0.065f};
    constexpr math::Vec3 kDirColor{0.18f, 0.21f, 0.34f};  // dim blue "moon"
    constexpr float kFogK = 0.045f;                        // fog = 1 - exp(-(k * distance)^2)
    constexpr math::Vec3 kFloorA{0.45f, 0.45f, 0.50f};
    constexpr math::Vec3 kFloorB{0.07f, 0.07f, 0.10f};
    constexpr float kFloorSpecular = 0.35f;
    constexpr float kFloorShininess = 40.0f;
    constexpr float kFloorHalf = 40.0f;  // the floor is a square of 80 x 80 units

    constexpr int kEffects = 7;                // raw, gamma, gamma-hw, blur, sobel, bloom, dof
    constexpr int kBlurRadiusDefault = 10;     // taps -10 .. +10
    constexpr float sigma_for(int radius) { return static_cast<float>(radius) / 2.5f; }
    constexpr float kBrightThreshold = 0.70f;  // bloom bright-pass
    constexpr float kBloomStrength = 1.4f;
    constexpr float kSobelLow = 0.15f;         // edge map = smoothstep(low, high, |gradient|)
    constexpr float kSobelHigh = 0.60f;
    constexpr float kFocusSharp = 0.75f;        // half width of the in-focus zone, in world units
    constexpr float kFocusRange = 3.0f;        // distance beyond the zone at which the blur is complete

    // Presentation modes of present.frag. 0..6 are the effects, 7 the overview of all of them.
    constexpr int kModeOverview = 7;
    constexpr std::array<const char*, 8> kModeNames = {"raw (linear, no gamma)", "gamma (shader sRGB encode)",
                                                       "gamma (SRGB8_ALPHA8 scene target)", "separable Gaussian blur",
                                                       "Sobel edges", "bloom", "depth of field", "overview (all stripes)"};

    // Tolerances, in 8-bit steps (1/255) of |stored byte - 255 * exact value|.
    // The exact value is computed on the CPU in double precision from the same inputs. A float result is converted to
    // 8 bits as "one of the two integers closest to f * 255" (ES 3.0 equation 2.3, section 2.1.6.2), so the rounding
    // alone can be 0.5 steps off, and real hardware measures up to ~0.6; the float arithmetic of the shader (texel
    // sums, exp, pow, sqrt, 1 part in 10^5 per operation, ES 3.0 section 2.1.1) adds a few hundredths. The sRGB
    // conversions (equations 3.26 and 4.1) are lookup- or polynomial-based on the tested drivers and stay below 0.5
    // steps; es30-srgb measured at most 0.85 steps for decode + 8-bit conversion. One step is therefore the bound
    // for a single pass. It is far tighter than every plausible mistake: the negative controls below are off by
    // many steps.
    constexpr double kPassTolerance = 1.0;
    // Two chained passes with an 8-bit intermediate: the second pass averages values that are each up to one step
    // off (at most one step after averaging) and adds its own conversion error.
    constexpr double kChainTolerance = 2.0;

    // ---------------------------------------------------------------------------------------------
    // Procedural meshes: interleaved position (vec3) + normal (vec3), 16-bit indices, counter-clockwise outside.
    // ---------------------------------------------------------------------------------------------
    struct MeshData
    {
        std::vector<float> vertices;
        std::vector<std::uint16_t> indices;
    };

    void push_vertex(MeshData& m, math::Vec3 p, math::Vec3 n)
    {
        for (const float v : {p.x, p.y, p.z, n.x, n.y, n.z}) m.vertices.push_back(v);
    }

    // Unit sphere: theta from the +y pole (0) to the -y pole (pi), phi around the y axis.
    MeshData make_sphere(int stacks, int slices)
    {
        MeshData m;
        for (int i = 0; i <= stacks; ++i)
        {
            const float theta = std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(stacks);
            for (int j = 0; j <= slices; ++j)
            {
                const float phi = 2.0f * std::numbers::pi_v<float> * static_cast<float>(j) / static_cast<float>(slices);
                const math::Vec3 p{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
                push_vertex(m, p, p);
            }
        }
        for (int i = 0; i < stacks; ++i)
            for (int j = 0; j < slices; ++j)
            {
                const auto a = static_cast<std::uint16_t>(i * (slices + 1) + j);
                const auto b = static_cast<std::uint16_t>(a + slices + 1);
                for (const std::uint16_t idx : {a, static_cast<std::uint16_t>(a + 1), b, static_cast<std::uint16_t>(a + 1),
                                                static_cast<std::uint16_t>(b + 1), b})
                    m.indices.push_back(idx);
            }
        return m;
    }

    // Torus around the y axis: ring angle u, tube angle v.
    MeshData make_torus(float major, float minor, int rings, int sides)
    {
        MeshData m;
        for (int i = 0; i <= rings; ++i)
        {
            const float u = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(rings);
            for (int j = 0; j <= sides; ++j)
            {
                const float v = 2.0f * std::numbers::pi_v<float> * static_cast<float>(j) / static_cast<float>(sides);
                const math::Vec3 n{std::cos(v) * std::cos(u), std::sin(v), std::cos(v) * std::sin(u)};
                const math::Vec3 p{(major + minor * std::cos(v)) * std::cos(u), minor * std::sin(v),
                                   (major + minor * std::cos(v)) * std::sin(u)};
                push_vertex(m, p, n);
            }
        }
        for (int i = 0; i < rings; ++i)
            for (int j = 0; j < sides; ++j)
            {
                const auto a = static_cast<std::uint16_t>(i * (sides + 1) + j);
                const auto b = static_cast<std::uint16_t>(a + sides + 1);
                for (const std::uint16_t idx : {a, static_cast<std::uint16_t>(a + 1), b, static_cast<std::uint16_t>(a + 1),
                                                static_cast<std::uint16_t>(b + 1), b})
                    m.indices.push_back(idx);
            }
        return m;
    }

    // Cube of side 1 with flat normals: four vertices per face (u x v = outward normal, so u -> v is counter-clockwise).
    MeshData make_cube()
    {
        struct Face
        {
            math::Vec3 n, u, v;
        };
        const Face faces[] = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},   {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
                              {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},   {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
                              {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},   {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}};
        MeshData m;
        for (const Face& f : faces)
        {
            const auto base = static_cast<std::uint16_t>(m.vertices.size() / 6);
            const math::Vec3 c = f.n * 0.5f;
            push_vertex(m, c - f.u * 0.5f - f.v * 0.5f, f.n);
            push_vertex(m, c + f.u * 0.5f - f.v * 0.5f, f.n);
            push_vertex(m, c + f.u * 0.5f + f.v * 0.5f, f.n);
            push_vertex(m, c - f.u * 0.5f + f.v * 0.5f, f.n);
            for (const int k : {0, 1, 2, 0, 2, 3}) m.indices.push_back(static_cast<std::uint16_t>(base + k));
        }
        return m;
    }

    // One big quad in the y = 0 plane, normal +y, counter-clockwise seen from above.
    MeshData make_floor(float half)
    {
        MeshData m;
        const math::Vec3 up{0, 1, 0};
        push_vertex(m, {-half, 0, half}, up);
        push_vertex(m, {half, 0, half}, up);
        push_vertex(m, {half, 0, -half}, up);
        push_vertex(m, {-half, 0, -half}, up);
        m.indices = {0, 1, 2, 0, 2, 3};
        return m;
    }

    struct Mesh
    {
        mgl::VertexArrayId vao{};
        mgl::BufferId vbo{};
        mgl::BufferId ibo{};
        GLsizei index_count = 0;
    };

    // ---------------------------------------------------------------------------------------------
    // The scene at a given time and camera.
    // ---------------------------------------------------------------------------------------------
    enum class Shape
    {
        Sphere,
        Torus,
        Cube
    };

    struct Item
    {
        Shape shape = Shape::Sphere;
        int kind = 0;  // scene.frag u_kind: 0 lit, 2 emissive lamp
        math::Mat4 model = math::Mat4::identity();
        math::Vec3 albedo{};
        math::Vec3 emissive{};
        float specular = 0.0f;
        float shininess = 1.0f;
        math::Vec3 center{};  // bounding sphere, for the CPU reference's visibility tests
        float radius = 0.0f;
    };

    struct Lamp
    {
        math::Vec3 position{};
        math::Vec3 emissive{};  // color of the bulb
        math::Vec3 light{};     // color * intensity of the light it emits
    };

    struct CameraControl
    {
        float yaw = 0.0f;       // added to the automatic orbit
        float radius = 10.0f;   // distance from the y axis
        float height = 2.6f;
        bool orbit = true;      // 0.12 rad/s
    };

    struct Camera
    {
        math::Vec3 eye{};
        math::Vec3 target = kTarget;
    };

    struct Frame
    {
        double time = 0.0;
        Camera camera;
        float aspect = 1.0f;
        float fovy = kFovY;
        math::Mat4 view_proj = math::Mat4::identity();
        std::array<Lamp, 2> lamps{};
        std::vector<Item> items;
        math::Vec3 dir_to_light{};
    };

    math::Vec3 dir_to_light()
    {
        return math::normalize(math::Vec3{-0.45f, 0.80f, 0.40f});
    }

    Camera make_camera(const CameraControl& c, double time)
    {
        const double yaw = static_cast<double>(c.yaw) + (c.orbit ? 0.12 * time : 0.0);
        Camera cam;
        cam.eye = {static_cast<float>(static_cast<double>(c.radius) * std::sin(yaw)), c.height,
                   static_cast<float>(static_cast<double>(c.radius) * std::cos(yaw))};
        return cam;
    }

    std::array<Lamp, 2> make_lamps(double t)
    {
        auto f = [](double v) { return static_cast<float>(v); };
        std::array<Lamp, 2> lamps;
        lamps[0].position = {f(3.4 * std::cos(0.9 * t + 0.4)), f(1.9 + 0.35 * std::sin(1.7 * t)), f(3.4 * std::sin(0.9 * t + 0.4))};
        lamps[0].emissive = {1.00f, 0.62f, 0.25f};
        lamps[0].light = {3.4f, 2.0f, 0.85f};
        lamps[1].position = {f(4.2 * std::cos(-0.6 * t + 2.4)), f(1.5 + 0.3 * std::sin(1.3 * t + 1.0)), f(4.2 * std::sin(-0.6 * t + 2.4))};
        lamps[1].emissive = {0.30f, 0.55f, 1.00f};
        lamps[1].light = {0.85f, 1.9f, 3.4f};
        return lamps;
    }

    std::vector<Item> make_items(double t, const std::array<Lamp, 2>& lamps)
    {
        const float ft = static_cast<float>(t);
        std::vector<Item> items;
        auto lit = [&](Shape shape, math::Mat4 model, math::Vec3 center, float radius, math::Vec3 albedo, float specular,
                       float shininess) {
            Item it;
            it.shape = shape;
            it.model = model;
            it.center = center;
            it.radius = radius;
            it.albedo = albedo;
            it.specular = specular;
            it.shininess = shininess;
            items.push_back(it);
        };
        // Centerpiece: a gold torus that tumbles.
        lit(Shape::Torus,
            math::translate({0.0f, 1.55f, 0.0f}) * math::rotate(0.7f * ft + 0.4f, {0, 1, 0}) * math::rotate(1.0f * ft + 0.6f, {1, 0, 0}),
            {0.0f, 1.55f, 0.0f}, 1.4f, {0.90f, 0.55f, 0.12f}, 0.8f, 80.0f);
        // A polished blue sphere and a spinning teal cube beside it.
        lit(Shape::Sphere, math::translate({-2.6f, 0.85f, 1.0f}) * math::scale({0.85f, 0.85f, 0.85f}), {-2.6f, 0.85f, 1.0f}, 0.85f,
            {0.12f, 0.30f, 0.85f}, 1.0f, 120.0f);
        lit(Shape::Cube, math::translate({2.5f, 0.7f, 0.5f}) * math::rotate(0.5f * ft + 0.3f, {0, 1, 0}) * math::scale({1.4f, 1.4f, 1.4f}),
            {2.5f, 0.7f, 0.5f}, 1.22f, {0.10f, 0.70f, 0.50f}, 0.4f, 30.0f);
        // A small magenta cube close to the camera (out of focus in the depth-of-field view).
        lit(Shape::Cube, math::translate({-1.5f, 0.4f, 4.0f}) * math::rotate(-0.4f * ft + 0.5f, {0, 1, 0}) * math::scale({0.8f, 0.8f, 0.8f}),
            {-1.5f, 0.4f, 4.0f}, 0.7f, {0.70f, 0.20f, 0.75f}, 0.5f, 40.0f);
        // Seven small spheres on a spiral, alternating red and yellow: depth cues for depth of field.
        for (int k = 0; k < 7; ++k)
        {
            const double angle = 0.5 + 0.9 * k;
            const double r = 3.6 + 0.55 * k;
            const math::Vec3 c{static_cast<float>(r * std::cos(angle)), 0.55f + 0.1f * std::sin(1.5f * ft + static_cast<float>(k)),
                               static_cast<float>(r * std::sin(angle))};
            lit(Shape::Sphere, math::translate(c) * math::scale({0.55f, 0.55f, 0.55f}), c, 0.55f,
                (k % 2 == 0) ? math::Vec3{0.80f, 0.15f, 0.20f} : math::Vec3{0.85f, 0.80f, 0.20f}, 0.6f, 60.0f);
        }
        // The bulbs of the two point lights.
        for (const Lamp& lamp : lamps)
        {
            Item it;
            it.shape = Shape::Sphere;
            it.kind = 2;
            it.model = math::translate(lamp.position) * math::scale({0.17f, 0.17f, 0.17f});
            it.emissive = lamp.emissive;
            it.center = lamp.position;
            it.radius = 0.17f;
            items.push_back(it);
        }
        return items;
    }

    Frame make_frame(double time, const CameraControl& control, int width, int height)
    {
        Frame f;
        f.time = time;
        f.camera = make_camera(control, time);
        f.aspect = static_cast<float>(width) / static_cast<float>(height);
        // A portrait surface keeps the horizontal field of view of the square one (tan(fovx / 2) = tan(22.5 degrees)) and
        // widens the vertical one instead, so the scene, both lamps included, stays in view at every window size.
        if (f.aspect < 1.0f) f.fovy = 2.0f * std::atan(std::tan(kFovY * 0.5f) / f.aspect);
        f.view_proj = math::perspective(f.fovy, f.aspect, kZNear, kZFar) * math::look_at(f.camera.eye, f.camera.target, {0, 1, 0});
        f.lamps = make_lamps(time);
        f.items = make_items(time, f.lamps);
        f.dir_to_light = dir_to_light();
        return f;
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference, double precision.
    // ---------------------------------------------------------------------------------------------
    struct D3
    {
        double x = 0, y = 0, z = 0;
    };
    D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    double length(D3 a) { return std::sqrt(dot(a, a)); }
    D3 normalize(D3 a) { return a * (1.0 / length(a)); }
    D3 to_d(math::Vec3 v) { return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)}; }

    using Rgb = std::array<double, 3>;
    Rgb to_rgb(math::Vec3 v) { return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)}; }
    double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }
    double mix(double a, double b, double t) { return a * (1.0 - t) + b * t; }
    double fract(double v) { return v - std::floor(v); }

    double smoothstep(double e0, double e1, double x)
    {
        const double t = clamp01((x - e0) / (e1 - e0));
        return t * t * (3.0 - 2.0 * t);
    }

    // sRGB transfer functions (ES 3.0 equation 3.26 decodes, equation 4.1 encodes).
    double srgb_encode(double c)
    {
        c = clamp01(c);
        return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    }
    double srgb_decode(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }
    // d(encode)/dx, the amplification of an error of the linear value in the encoded domain.
    double srgb_encode_slope(double x) { return x <= 0.0031308 ? 12.92 : 1.055 / 2.4 * std::pow(x, 1.0 / 2.4 - 1.0); }

    struct Basis
    {
        D3 eye, forward, right, up;  // right and up are scaled by tan(fovy / 2) (and the aspect ratio for right)
    };

    Basis make_basis(const Frame& f)
    {
        Basis b;
        b.eye = to_d(f.camera.eye);
        b.forward = normalize(to_d(f.camera.target) - b.eye);
        const D3 side = normalize(cross(b.forward, {0, 1, 0}));
        const D3 up = cross(side, b.forward);
        const double t = std::tan(static_cast<double>(f.fovy) * 0.5);
        b.right = side * (t * static_cast<double>(f.aspect));
        b.up = up * t;
        return b;
    }

    // View ray (not normalized) through the center of pixel (x, y) of a w x h target; y counts from the bottom, like GL.
    D3 ray_dir(const Basis& b, double x, double y, int w, int h)
    {
        const double nx = 2.0 * (x + 0.5) / w - 1.0;
        const double ny = 2.0 * (y + 0.5) / h - 1.0;
        return b.forward + b.right * nx + b.up * ny;
    }

    Rgb sky_cpu(D3 dir)
    {
        const double s = smoothstep(0.0, 0.25, normalize(dir).y);
        const Rgb horizon = to_rgb(kHorizon), zenith = to_rgb(kZenith);
        return {mix(horizon[0], zenith[0], s), mix(horizon[1], zenith[1], s), mix(horizon[2], zenith[2], s)};
    }

    // The filtered checkerboard of scene.frag evaluated in the middle of a square (any small filter width gives 0 or 1).
    double checker_cpu(double px, double pz)
    {
        const double w = 1.0e-3;
        auto tri = [](double v) { return std::abs(fract(v) - 0.5); };
        auto axis = [&](double p) { return 2.0 * (tri((p - 0.5 * w) * 0.5) - tri((p + 0.5 * w) * 0.5)) / w; };
        return 0.5 - 0.5 * axis(px) * axis(pz);
    }

    D3 light_term(D3 L, D3 N, D3 V, D3 albedo, double specular, double shininess)
    {
        const double n_dot_l = std::max(dot(N, L), 0.0);
        const D3 H = normalize(L + V);
        const double spec = n_dot_l > 0.0 ? std::pow(std::max(dot(N, H), 0.0), shininess) : 0.0;
        return albedo * n_dot_l + D3{specular * spec, specular * spec, specular * spec};
    }

    // Color of the floor at world point P, as scene.frag computes it (linear, before clamping to the target).
    Rgb floor_cpu(D3 P, const Frame& f)
    {
        const D3 eye = to_d(f.camera.eye);
        const D3 N{0, 1, 0};
        const double dist = length(eye - P);
        const D3 V = (eye - P) * (1.0 / dist);
        const double chk = checker_cpu(P.x, P.z);
        const D3 a = to_d(kFloorA), b = to_d(kFloorB);
        const D3 albedo = a * (1.0 - chk) + b * chk;
        const double ks = static_cast<double>(kFloorSpecular), shin = static_cast<double>(kFloorShininess);
        D3 color = D3{albedo.x * static_cast<double>(kAmbient.x), albedo.y * static_cast<double>(kAmbient.y),
                      albedo.z * static_cast<double>(kAmbient.z)};
        const D3 dc = to_d(kDirColor);
        const D3 lt = light_term(to_d(f.dir_to_light), N, V, albedo, ks, shin);
        color = color + D3{dc.x * lt.x, dc.y * lt.y, dc.z * lt.z};
        for (const Lamp& lamp : f.lamps)
        {
            const D3 to_light = to_d(lamp.position) - P;
            const double d = length(to_light);
            const double attenuation = 1.0 / (1.0 + 0.09 * d + 0.032 * d * d);
            const D3 term = light_term(to_light * (1.0 / d), N, V, albedo, ks, shin);
            const D3 lc = to_d(lamp.light);
            color = color + D3{lc.x * attenuation * term.x, lc.y * attenuation * term.y, lc.z * attenuation * term.z};
        }
        const double kd = static_cast<double>(kFogK) * dist;
        const double fog = 1.0 - std::exp(-kd * kd);
        const Rgb fc = to_rgb(kHorizon);
        return {mix(color.x, fc[0], fog), mix(color.y, fc[1], fog), mix(color.z, fc[2], fog)};
    }

    struct Sphere
    {
        D3 center;
        double radius;
    };

    // Smallest t >= 0 with |o + t d - c| = r, if any.
    bool ray_sphere(D3 o, D3 d, const Sphere& s, double& t)
    {
        const D3 oc = o - s.center;
        const double a = dot(d, d), b = dot(oc, d), c = dot(oc, oc) - s.radius * s.radius;
        const double disc = b * b - a * c;
        if (disc < 0.0) return false;
        t = (-b - std::sqrt(disc)) / a;
        return t >= 0.0;
    }

    // ---------------------------------------------------------------------------------------------
    // Images read back from the GPU and their CPU models.
    // ---------------------------------------------------------------------------------------------
    struct Bytes
    {
        int w = 0, h = 0;
        std::vector<std::uint8_t> rgba;  // GL order: the first row is the bottom row
        [[nodiscard]] std::uint8_t at(int x, int y, int c) const
        {
            return rgba[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4u + static_cast<std::size_t>(c)];
        }
    };

    struct Image
    {
        int w = 0, h = 0;
        std::vector<Rgb> px;
        Image() = default;
        Image(int width, int height) : w(width), h(height), px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {}
        [[nodiscard]] Rgb& at(int x, int y) { return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
        [[nodiscard]] const Rgb& at(int x, int y) const { return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
        [[nodiscard]] const Rgb& clamped(int x, int y) const { return at(std::clamp(x, 0, w - 1), std::clamp(y, 0, h - 1)); }
    };

    // Equation 2.1: unsigned normalized fixed-point -> float.
    Image to_image(const Bytes& b, bool srgb_decode_rgb = false)
    {
        Image img(b.w, b.h);
        for (int y = 0; y < b.h; ++y)
            for (int x = 0; x < b.w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    const double v = static_cast<double>(b.at(x, y, c)) / 255.0;
                    img.at(x, y)[static_cast<std::size_t>(c)] = srgb_decode_rgb ? srgb_decode(v) : v;
                }
        return img;
    }

    struct Cmp
    {
        std::size_t samples = 0;
        std::size_t bad = 0;
        double max_dev = 0.0;
        std::string first;

        // `got` is a stored byte, `want` the exact value in [0, 1]; where() is only called for the first failure.
        template <typename Where>
        void add(double got, double want, double tolerance, Where&& where)
        {
            ++samples;
            const double dev = std::abs(got - 255.0 * clamp01(want));
            max_dev = std::max(max_dev, dev);
            if (dev > tolerance && bad++ == 0)
                first = std::format("{}: got {} expected {:.2f} (tolerance {:.2f})", where(), got, 255.0 * clamp01(want), tolerance);
        }
    };

    Cmp compare_images(const Bytes& got, const Image& want, double tolerance)
    {
        Cmp cmp;
        for (int y = 0; y < got.h; ++y)
            for (int x = 0; x < got.w; ++x)
            {
                for (int c = 0; c < 3; ++c)
                    cmp.add(static_cast<double>(got.at(x, y, c)), want.at(x, y)[static_cast<std::size_t>(c)], tolerance,
                            [&] { return std::format("pixel ({},{}) channel {}", x, y, c); });
                cmp.add(static_cast<double>(got.at(x, y, 3)), 1.0, 0.0, [&] { return std::format("pixel ({},{}) alpha", x, y); });
            }
        return cmp;
    }

    Image quantize(const Image& img)
    {
        Image out = img;
        for (Rgb& p : out.px)
            for (double& v : p) v = std::round(clamp01(v) * 255.0) / 255.0;
        return out;
    }

    // Separable Gaussian, one direction: weights exp(-i^2 / (2 sigma^2)), i = -radius..radius, normalized by their sum,
    // coordinates clamped to the image (the same as CLAMP_TO_EDGE). box = equal weights (negative control).
    Image blur_cpu(const Image& src, int dx, int dy, int radius, double sigma, bool box = false)
    {
        Image out(src.w, src.h);
        std::vector<double> weights;
        double sum = 0.0;
        for (int i = -radius; i <= radius; ++i)
        {
            weights.push_back(box ? 1.0 : std::exp(-0.5 * static_cast<double>(i * i) / (sigma * sigma)));
            sum += weights.back();
        }
        for (int y = 0; y < src.h; ++y)
            for (int x = 0; x < src.w; ++x)
            {
                Rgb acc{};
                for (int i = -radius; i <= radius; ++i)
                {
                    const Rgb& s = src.clamped(x + i * dx, y + i * dy);
                    const double w = weights[static_cast<std::size_t>(i + radius)];
                    for (std::size_t c = 0; c < 3; ++c) acc[c] += w * s[c];
                }
                for (std::size_t c = 0; c < 3; ++c) out.at(x, y)[c] = acc[c] / sum;
            }
        return out;
    }

    Image bright_cpu(const Image& src, double threshold)
    {
        Image out(src.w, src.h);
        for (std::size_t i = 0; i < src.px.size(); ++i)
            for (std::size_t c = 0; c < 3; ++c) out.px[i][c] = std::max(src.px[i][c] - threshold, 0.0) / (1.0 - threshold);
        return out;
    }

    double brightness(const Rgb& c) { return std::sqrt(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]); }

    // Sobel; horizontal_only drops Gy (negative control).
    Image sobel_cpu(const Image& src, bool horizontal_only = false)
    {
        Image out(src.w, src.h);
        for (int y = 0; y < src.h; ++y)
            for (int x = 0; x < src.w; ++x)
            {
                auto b = [&](int dx, int dy) { return brightness(src.clamped(x + dx, y + dy)); };
                const double gx = (b(1, 1) + 2.0 * b(1, 0) + b(1, -1)) - (b(-1, 1) + 2.0 * b(-1, 0) + b(-1, -1));
                const double gy = horizontal_only ? 0.0 : (b(-1, 1) + 2.0 * b(0, 1) + b(1, 1)) - (b(-1, -1) + 2.0 * b(0, -1) + b(1, -1));
                const double e = smoothstep(static_cast<double>(kSobelLow), static_cast<double>(kSobelHigh), std::sqrt(gx * gx + gy * gy));
                out.at(x, y) = {e, e, e};
            }
        return out;
    }

    // Everything the present pass reads, as the GPU stored it.
    struct Maps
    {
        Image scene;         // RGBA8 scene
        Image scene_srgb;    // SRGB8_ALPHA8 scene decoded to linear (what a lookup returns)
        Image blur, edge, bloom;
        std::vector<double> depth;  // window-space depth of the RGBA8 scene
        double focus = 10.0;
        bool decode_sampled_srgb = true;  // false: negative control, a lookup that skipped the decode
    };

    double linear_depth(double d)
    {
        const double n = static_cast<double>(kZNear), f = static_cast<double>(kZFar);
        const double z_ndc = 2.0 * d - 1.0;
        return 2.0 * n * f / (f + n - z_ndc * (f - n));
    }

    Rgb encode(Rgb c) { return {srgb_encode(c[0]), srgb_encode(c[1]), srgb_encode(c[2])}; }

    // present.frag mode 0..6 at pixel (x, y), as the CPU computes it (a display value, before the 8-bit conversion).
    Rgb effect_cpu(int mode, int x, int y, const Maps& m, bool plain_gamma_22 = false)
    {
        auto enc = [&](Rgb c) {
            if (!plain_gamma_22) return encode(c);
            return Rgb{std::pow(clamp01(c[0]), 1.0 / 2.2), std::pow(clamp01(c[1]), 1.0 / 2.2), std::pow(clamp01(c[2]), 1.0 / 2.2)};
        };
        const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(m.scene.w) + static_cast<std::size_t>(x);
        switch (mode)
        {
        case 0: return m.scene.px[i];
        case 1: return enc(m.scene.px[i]);
        case 2: return enc(m.scene_srgb.px[i]);
        case 3: return enc(m.blur.px[i]);
        case 4: return m.edge.px[i];
        case 5:
        {
            Rgb c = m.scene.px[i];
            for (std::size_t k = 0; k < 3; ++k) c[k] += static_cast<double>(kBloomStrength) * m.bloom.px[i][k];
            return enc(c);
        }
        default:
        {
            const double z = linear_depth(m.depth[i]);
            const double coc = clamp01((std::abs(z - m.focus) - static_cast<double>(kFocusSharp)) / static_cast<double>(kFocusRange));
            Rgb c;
            for (std::size_t k = 0; k < 3; ++k) c[k] = mix(m.scene.px[i][k], m.blur.px[i][k], coc);
            return enc(c);
        }
        }
    }

    Rgb present_cpu(int mode, int x, int y, const Maps& m, bool plain_gamma_22 = false)
    {
        if (mode != kModeOverview) return effect_cpu(mode, x, y, m, plain_gamma_22);
        const int width = m.scene.w;
        const int stripe = (x * kEffects) / width;
        const bool divider = x > 0 && ((x - 1) * kEffects) / width != stripe;
        if (divider) return {0.85, 0.85, 0.85};
        return effect_cpu(stripe, x, y, m, plain_gamma_22);
    }

    Image present_image(int mode, const Maps& m, bool plain_gamma_22 = false)
    {
        Image img(m.scene.w, m.scene.h);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) img.at(x, y) = present_cpu(mode, x, y, m, plain_gamma_22);
        return img;
    }

    void report(demo::Check& check, const std::string& what, const Cmp& cmp, double tolerance)
    {
        check.expect(cmp.bad == 0, std::format("{} [{} samples]", what, cmp.samples),
                     std::format("max deviation {:.3f} of 255 ({}){}", cmp.max_dev,
                                 tolerance > 0.0 ? std::format("tolerance {:.2f}", tolerance) : std::string("per-pixel tolerance"),
                                 cmp.bad == 0 ? std::string{} : "; first: " + cmp.first));
    }

    // A negative control passes when the wrong model disagrees with the GPU in at least min_bad samples.
    void report_control(demo::Check& check, const std::string& what, const Cmp& cmp, std::size_t min_bad)
    {
        check.expect(cmp.bad >= min_bad, std::format("negative control: {}", what),
                     std::format("{} of {} samples differ by more than the tolerance (at least {} required); max deviation {:.1f} of 255",
                                 cmp.bad, cmp.samples, min_bad, cmp.max_dev));
    }

    // ---------------------------------------------------------------------------------------------
    // GL objects.
    // ---------------------------------------------------------------------------------------------
    struct Target
    {
        mgl::TextureId tex{};
        mgl::FramebufferId fbo{};
    };

    constexpr std::array<mgl::TextureUnit, 6> kUnits = {mgl::TextureUnit::Texture0, mgl::TextureUnit::Texture1, mgl::TextureUnit::Texture2,
                                                        mgl::TextureUnit::Texture3, mgl::TextureUnit::Texture4, mgl::TextureUnit::Texture5};

    mgl::UniformLocation loc(mgl::ProgramId program, const char* name) { return mgl::glGetUniformLocation(program, name); }

    constexpr int kMinSurface = 128;  // pixels in each direction (see init())

    class PostProcessing final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // The object outlines in the Sobel map are a few pixels wide at any size, so their share of the image grows as
            // the surface shrinks: below about 100 x 100 it exceeds the third that "the edge map has strong edges on a small
            // part of the image" allows (32.6% at 100 x 100, 28.1% at 128 x 128, 16% at 256 x 256). 128 keeps a margin.
            if (env.width < kMinSurface || env.height < kMinSurface)
                throw demo::Skip(std::format("needs a default framebuffer of at least {0}x{0}", kMinSurface));
            if (demo::get_integer(mgl::GetParameter::RedBits) < 8 || demo::get_integer(mgl::GetParameter::GreenBits) < 8 ||
                demo::get_integer(mgl::GetParameter::BlueBits) < 8)
                throw demo::Skip("needs 8 bits per color channel in the default framebuffer");

            sky_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/sky.frag");
            scene_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            bright_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/bright.frag");
            blur_prog_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/blur.frag");
            sobel_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/sobel.frag");
            dump_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/depth_dump.frag");
            present_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/present.frag");

            // Meshes.
            upload_mesh(sphere_, make_sphere(32, 48));
            upload_mesh(torus_, make_torus(1.0f, 0.38f, 64, 32));
            upload_mesh(cube_, make_cube());
            upload_mesh(floor_, make_floor(kFloorHalf));
            // Full-screen passes draw 3 vertices that come from gl_VertexID; a VAO with no enabled attribute serves them.
            mgl::glGenVertexArrays(1, &empty_vao_);

            // The encoding of the default framebuffer is implementation-defined (ES 3.0 section 6.1.13). When it
            // is sRGB the hardware would encode the presented image a second time; present.frag undoes it first.
            GLint encoding = GL_LINEAR;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::DefaultFramebufferQueryAttachment::Back,
                                                       mgl::FramebufferAttachmentParameter::ColorEncoding, &encoding);
            fb_srgb_ = encoding == GL_SRGB;
            if (fb_srgb_) std::cout << "   note: the default framebuffer is sRGB-encoded; present.frag decodes before writing\n";

            create_targets(env.width, env.height);
        }

        void render(demo::Env& env, int frame) override
        {
            if (paused_) ++paused_frames_;
            time_ = static_cast<double>(frame - paused_frames_) / 60.0;
            if (env.width != width_ || env.height != height_)
            {
                destroy_targets();
                create_targets(env.width, env.height);
            }
            draw_frame();
        }

        void on_key(demo::Env& env, char key) override
        {
            auto select = [&](int mode) {
                mode_ = mode;
                if (env.interactive) std::cout << "effect: " << kModeNames[static_cast<std::size_t>(mode_)] << "\n";
            };
            switch (key)
            {
            case '0': select(kModeOverview); break;
            case '1': case '2': case '3': case '4': case '5': case '6': case '7': select(key - '1'); break;
            case 'n': select(mode_ == kModeOverview ? 0 : (mode_ + 1 == kEffects ? kModeOverview : mode_ + 1)); break;
            case 'p': select(mode_ == kModeOverview ? kEffects - 1 : (mode_ == 0 ? kModeOverview : mode_ - 1)); break;
            case 'a': control_.yaw -= 0.15f; break;
            case 'd': control_.yaw += 0.15f; break;
            case 'w': control_.radius = std::max(control_.radius - 0.5f, 5.0f); break;
            case 's': control_.radius = std::min(control_.radius + 0.5f, 18.0f); break;
            case 'q': control_.height = std::min(control_.height + 0.3f, 8.0f); break;
            case 'e': control_.height = std::max(control_.height - 0.3f, 0.8f); break;
            case 'c': control_.orbit = !control_.orbit; break;
            case 'r': control_ = CameraControl{}; break;
            case ' ': paused_ = !paused_; break;
            case 'b':
                blur_radius_ = blur_radius_ == 2 ? 5 : blur_radius_ == 5 ? 10 : blur_radius_ == 10 ? 16 : 2;
                if (env.interactive) std::cout << "blur radius: " << blur_radius_ << " (sigma " << sigma_for(blur_radius_) << ")\n";
                break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            destroy_targets();
            for (Mesh* m : {&sphere_, &torus_, &cube_, &floor_})
            {
                mgl::glDeleteVertexArrays(1, &m->vao);
                mgl::glDeleteBuffers(1, &m->vbo);
                mgl::glDeleteBuffers(1, &m->ibo);
            }
            mgl::glDeleteVertexArrays(1, &empty_vao_);
            for (const mgl::ProgramId p : {sky_, scene_, bright_, blur_prog_, sobel_, dump_, present_}) mgl::glDeleteProgram(p);
        }

    private:
        // ---- creation -------------------------------------------------------------------------------
        static void upload_mesh(Mesh& mesh, const MeshData& data)
        {
            mgl::glGenVertexArrays(1, &mesh.vao);
            mgl::glBindVertexArray(mesh.vao);
            mgl::glGenBuffers(1, &mesh.vbo);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mesh.vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(data.vertices), mgl::BufferUsage::StaticDraw);
            // The element array buffer binding IS vertex array state: it is recorded in the VAO bound right now.
            mgl::glGenBuffers(1, &mesh.ibo);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mesh.ibo);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(data.indices), mgl::BufferUsage::StaticDraw);
            const GLsizei stride = 6 * static_cast<GLsizei>(sizeof(float));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, stride,
                                       demo::buffer_offset(3 * sizeof(float)));
            mesh.index_count = static_cast<GLsizei>(data.indices.size());
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        // A color texture with NEAREST filtering. texelFetch ignores filter state, but a texture whose minification
        // filter needs mipmaps (the default) is incomplete without them (ES 3.0 section 3.8.13).
        static mgl::TextureId make_texture(int w, int h, mgl::InternalFormat internal, mgl::PixelFormat format, mgl::PixelType type)
        {
            mgl::TextureId tex{};
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, internal, w, h, 0, format, type, nullptr);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            return tex;
        }

        static void require_complete(const char* what)
        {
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer '{}' is {}", what, demo::framebuffer_status_name(status)));
        }

        static Target make_color_target(int w, int h, mgl::InternalFormat internal, const char* what)
        {
            Target t;
            t.tex = make_texture(w, h, internal, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte);
            mgl::glGenFramebuffers(1, &t.fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, t.fbo);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2D, t.tex, 0);
            require_complete(what);
            return t;
        }

        void create_targets(int w, int h)
        {
            width_ = w;
            height_ = h;
            // The scene: an RGBA8 color texture and a DEPTH_COMPONENT24 depth TEXTURE (so depth of field can sample it).
            scene_a_ = make_color_target(w, h, mgl::InternalFormat::Rgba8, "scene RGBA8");
            depth_tex_ = make_texture(w, h, mgl::InternalFormat::DepthComponent24, mgl::PixelFormat::DepthComponent, mgl::PixelType::UnsignedInt);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, scene_a_.fbo);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, mgl::TextureTarget::Texture2D,
                                        depth_tex_, 0);
            require_complete("scene RGBA8 + depth texture");
            // The same scene into an SRGB8_ALPHA8 texture (hardware encode on write) with a depth RENDERBUFFER.
            scene_b_ = make_color_target(w, h, mgl::InternalFormat::Srgb8Alpha8, "scene SRGB8_ALPHA8");
            mgl::glGenRenderbuffers(1, &depth_rb_);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, depth_rb_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent24, w, h);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, scene_b_.fbo);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                           mgl::RenderbufferTarget::Renderbuffer, depth_rb_);
            require_complete("scene SRGB8_ALPHA8 + depth renderbuffer");
            // Post-processing targets (RGBA8, no depth).
            bright_t_ = make_color_target(w, h, mgl::InternalFormat::Rgba8, "bright-pass");
            tmp_ = make_color_target(w, h, mgl::InternalFormat::Rgba8, "ping");
            blur_t_ = make_color_target(w, h, mgl::InternalFormat::Rgba8, "blur");
            bloom_t_ = make_color_target(w, h, mgl::InternalFormat::Rgba8, "bloom blur");
            edge_t_ = make_color_target(w, h, mgl::InternalFormat::Rgba8, "edges");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
        }

        void destroy_targets()
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            for (Target* t : {&scene_a_, &scene_b_, &bright_t_, &tmp_, &blur_t_, &bloom_t_, &edge_t_})
            {
                mgl::glDeleteFramebuffers(1, &t->fbo);
                mgl::glDeleteTextures(1, &t->tex);
                *t = Target{};
            }
            mgl::glDeleteTextures(1, &depth_tex_);
            mgl::glDeleteRenderbuffers(1, &depth_rb_);
            depth_tex_ = mgl::TextureId{};
            depth_rb_ = mgl::RenderbufferId{};
        }

        // ---- the 3D scene ----------------------------------------------------------------------------
        void draw_mesh(const Mesh& mesh) const
        {
            mgl::glBindVertexArray(mesh.vao);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, mesh.index_count, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
        }

        void draw_scene(mgl::FramebufferId fbo) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glViewport(0, 0, width_, height_);
            mgl::glDisable(mgl::Capability::Dither);  // dithering (ES 3.0 section 4.1.9) may change a stored byte by one step
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDepthMask(GL_TRUE);                // glClear honors the depth mask
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            // Sky: a full-screen triangle without depth test or culling.
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
            const Basis b = make_basis(frame_);
            mgl::glUseProgram(sky_);
            mgl::glUniform2f(loc(sky_, "u_resolution"), static_cast<float>(width_), static_cast<float>(height_));
            auto vec3 = [](mgl::UniformLocation l, D3 v) {
                mgl::glUniform3f(l, static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
            };
            vec3(loc(sky_, "u_cam_forward"), b.forward);
            vec3(loc(sky_, "u_cam_right"), b.right);
            vec3(loc(sky_, "u_cam_up"), b.up);
            mgl::glUniform3f(loc(sky_, "u_horizon"), kHorizon.x, kHorizon.y, kHorizon.z);
            mgl::glUniform3f(loc(sky_, "u_zenith"), kZenith.x, kZenith.y, kZenith.z);
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            // Lit geometry with depth test and back-face culling.
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glEnable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);
            mgl::glFrontFace(mgl::FrontFace::CCW);
            mgl::glUseProgram(scene_);
            mgl::glUniformMatrix4fv(loc(scene_, "u_view_proj"), 1, GL_FALSE, frame_.view_proj.data());
            mgl::glUniform3f(loc(scene_, "u_eye"), frame_.camera.eye.x, frame_.camera.eye.y, frame_.camera.eye.z);
            mgl::glUniform3f(loc(scene_, "u_ambient"), kAmbient.x, kAmbient.y, kAmbient.z);
            mgl::glUniform3f(loc(scene_, "u_dir_to_light"), frame_.dir_to_light.x, frame_.dir_to_light.y, frame_.dir_to_light.z);
            mgl::glUniform3f(loc(scene_, "u_dir_color"), kDirColor.x, kDirColor.y, kDirColor.z);
            std::array<float, 6> pos{}, col{};
            for (std::size_t i = 0; i < 2; ++i)
            {
                const Lamp& l = frame_.lamps[i];
                pos[3 * i] = l.position.x, pos[3 * i + 1] = l.position.y, pos[3 * i + 2] = l.position.z;
                col[3 * i] = l.light.x, col[3 * i + 1] = l.light.y, col[3 * i + 2] = l.light.z;
            }
            mgl::glUniform3fv(loc(scene_, "u_point_pos"), 2, pos.data());
            mgl::glUniform3fv(loc(scene_, "u_point_color"), 2, col.data());
            mgl::glUniform3f(loc(scene_, "u_fog_color"), kHorizon.x, kHorizon.y, kHorizon.z);
            mgl::glUniform1f(loc(scene_, "u_fog_k"), kFogK);

            auto set_model = [&](const math::Mat4& model) {
                mgl::glUniformMatrix4fv(loc(scene_, "u_model"), 1, GL_FALSE, model.data());
                const std::array<float, 9> nm = math::normal_matrix(model);
                mgl::glUniformMatrix3fv(loc(scene_, "u_normal_matrix"), 1, GL_FALSE, nm.data());
            };
            // The floor (kind 1).
            set_model(math::Mat4::identity());
            mgl::glUniform1i(loc(scene_, "u_kind"), 1);
            mgl::glUniform3f(loc(scene_, "u_albedo_a"), kFloorA.x, kFloorA.y, kFloorA.z);
            mgl::glUniform3f(loc(scene_, "u_albedo_b"), kFloorB.x, kFloorB.y, kFloorB.z);
            mgl::glUniform1f(loc(scene_, "u_specular"), kFloorSpecular);
            mgl::glUniform1f(loc(scene_, "u_shininess"), kFloorShininess);
            draw_mesh(floor_);
            // Everything else.
            for (const Item& it : frame_.items)
            {
                set_model(it.model);
                mgl::glUniform1i(loc(scene_, "u_kind"), it.kind);
                mgl::glUniform3f(loc(scene_, "u_albedo"), it.albedo.x, it.albedo.y, it.albedo.z);
                mgl::glUniform3f(loc(scene_, "u_emissive"), it.emissive.x, it.emissive.y, it.emissive.z);
                mgl::glUniform1f(loc(scene_, "u_specular"), it.specular);
                mgl::glUniform1f(loc(scene_, "u_shininess"), it.shininess);
                draw_mesh(it.shape == Shape::Sphere ? sphere_ : it.shape == Shape::Torus ? torus_ : cube_);
            }
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        // ---- post-processing passes -------------------------------------------------------------------
        void begin_pass(mgl::FramebufferId fbo) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glViewport(0, 0, width_, height_);
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
        }

        void draw_fullscreen() const
        {
            mgl::glBindVertexArray(empty_vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        static void bind_unit(std::size_t unit, mgl::TextureId tex)
        {
            mgl::glActiveTexture(kUnits[unit]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
        }

        void pass_bright(mgl::TextureId src, const Target& dst) const
        {
            begin_pass(dst.fbo);
            mgl::glUseProgram(bright_);
            bind_unit(0, src);
            mgl::glUniform1i(loc(bright_, "u_src"), 0);
            mgl::glUniform1f(loc(bright_, "u_threshold"), kBrightThreshold);
            draw_fullscreen();
        }

        void pass_blur(mgl::TextureId src, const Target& dst, int dx, int dy, int radius) const
        {
            begin_pass(dst.fbo);
            mgl::glUseProgram(blur_prog_);
            bind_unit(0, src);
            mgl::glUniform1i(loc(blur_prog_, "u_src"), 0);
            mgl::glUniform2i(loc(blur_prog_, "u_dir"), dx, dy);
            mgl::glUniform1i(loc(blur_prog_, "u_radius"), radius);
            mgl::glUniform1f(loc(blur_prog_, "u_sigma"), sigma_for(radius));
            draw_fullscreen();
        }

        // Ping-pong: the horizontal pass writes `tmp_`, the vertical pass reads it and writes `dst`.
        void blur_chain(mgl::TextureId src, const Target& dst, int radius) const
        {
            pass_blur(src, tmp_, 1, 0, radius);
            pass_blur(tmp_.tex, dst, 0, 1, radius);
        }

        void pass_sobel(mgl::TextureId src, const Target& dst) const
        {
            begin_pass(dst.fbo);
            mgl::glUseProgram(sobel_);
            bind_unit(0, src);
            mgl::glUniform1i(loc(sobel_, "u_src"), 0);
            mgl::glUniform1f(loc(sobel_, "u_low"), kSobelLow);
            mgl::glUniform1f(loc(sobel_, "u_high"), kSobelHigh);
            draw_fullscreen();
        }

        void pass_depth_dump(const Target& dst) const
        {
            begin_pass(dst.fbo);
            mgl::glUseProgram(dump_);
            bind_unit(0, depth_tex_);
            mgl::glUniform1i(loc(dump_, "u_depth"), 0);
            draw_fullscreen();
        }

        void pass_present(int mode) const
        {
            begin_pass(mgl::FramebufferId{0});
            mgl::glUseProgram(present_);
            const mgl::TextureId textures[] = {scene_a_.tex, scene_b_.tex, blur_t_.tex, edge_t_.tex, bloom_t_.tex, depth_tex_};
            const char* names[] = {"u_scene", "u_scene_srgb", "u_blur", "u_edge", "u_bloom", "u_depth"};
            for (std::size_t i = 0; i < 6; ++i)
            {
                bind_unit(i, textures[i]);
                mgl::glUniform1i(loc(present_, names[i]), static_cast<GLint>(i));
            }
            mgl::glUniform1i(loc(present_, "u_mode"), mode);
            mgl::glUniform1i(loc(present_, "u_fb_srgb"), fb_srgb_ ? 1 : 0);
            mgl::glUniform1f(loc(present_, "u_bloom_strength"), kBloomStrength);
            mgl::glUniform1f(loc(present_, "u_z_near"), kZNear);
            mgl::glUniform1f(loc(present_, "u_z_far"), kZFar);
            mgl::glUniform1f(loc(present_, "u_focus"), focus_distance());
            mgl::glUniform1f(loc(present_, "u_sharp"), kFocusSharp);
            mgl::glUniform1f(loc(present_, "u_range"), kFocusRange);
            draw_fullscreen();
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        }

        [[nodiscard]] float focus_distance() const
        {
            // Distance along the view axis of the point the camera looks at: the in-focus plane.
            const D3 f = normalize(to_d(frame_.camera.target) - to_d(frame_.camera.eye));
            return static_cast<float>(dot(to_d(frame_.camera.target) - to_d(frame_.camera.eye), f));
        }

        // One complete frame: scene -> offscreen targets -> the passes the selected look needs -> the window.
        void draw_frame()
        {
            frame_ = make_frame(time_, control_, width_, height_);
            const bool all = mode_ == kModeOverview;
            draw_scene(scene_a_.fbo);
            if (all || mode_ == 2) draw_scene(scene_b_.fbo);
            if (all || mode_ == 3 || mode_ == 6) blur_chain(scene_a_.tex, blur_t_, blur_radius_);
            if (all || mode_ == 4) pass_sobel(scene_a_.tex, edge_t_);
            if (all || mode_ == 5)
            {
                pass_bright(scene_a_.tex, bright_t_);
                blur_chain(bright_t_.tex, bloom_t_, kBlurRadiusDefault);
            }
            pass_present(mode_);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        // ---- validation helpers -----------------------------------------------------------------------
        [[nodiscard]] Bytes read_fbo(mgl::FramebufferId fbo) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo);
            Bytes b{width_, height_, demo::read_rgba8(0, 0, width_, height_)};
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return b;
        }

        // Probe pixels chosen by the CPU reference from the camera and the scene: sky, floor (away from every square
        // edge and from every object), and the CPU-projected centers of the lamp bulbs.
        struct Probes
        {
            std::vector<std::array<int, 2>> sky, floor;
            std::vector<std::array<int, 3>> lamp;  // x, y, lamp index
        };
        [[nodiscard]] Probes find_probes() const;

        // Which pixels the CPU reference can predict: unoccluded sky (view ray elevation above 0.02), unoccluded floor
        // (within 30 units of the origin and 10 units of the eye, away from every checker square edge), or neither.
        enum class PixelKind { None, Sky, Floor };
        struct ModelView
        {
            Basis basis;
            std::vector<Sphere> spheres;  // bounding spheres of the objects, 0.1 larger than the objects
        };
        [[nodiscard]] ModelView model_view() const;
        [[nodiscard]] PixelKind classify_pixel(const ModelView& mv, int x, int y) const;
        /// The CPU value of pixel (x, y) (linear) if classify_pixel() knows it.
        [[nodiscard]] std::optional<Rgb> model_value(const ModelView& mv, int x, int y) const;

        void check_scene(demo::Check& check, const Bytes& a, const Bytes& s, const Probes& probes);
        void check_srgb_target(demo::Check& check, const Bytes& a, const Bytes& s);

        // ---- state ------------------------------------------------------------------------------------
        mgl::ProgramId sky_{}, scene_{}, bright_{}, blur_prog_{}, sobel_{}, dump_{}, present_{};
        Mesh sphere_, torus_, cube_, floor_;
        mgl::VertexArrayId empty_vao_{};
        Target scene_a_, scene_b_, bright_t_, tmp_, blur_t_, bloom_t_, edge_t_;
        mgl::TextureId depth_tex_{};
        mgl::RenderbufferId depth_rb_{};
        int width_ = 0, height_ = 0;
        bool fb_srgb_ = false;

        Frame frame_;
        CameraControl control_;
        double time_ = 0.0;
        int paused_frames_ = 0;
        bool paused_ = false;
        int mode_ = kModeOverview;
        int blur_radius_ = kBlurRadiusDefault;
    };

    PostProcessing::ModelView PostProcessing::model_view() const
    {
        ModelView mv;
        mv.basis = make_basis(frame_);
        for (const Item& it : frame_.items) mv.spheres.push_back({to_d(it.center), static_cast<double>(it.radius) + 0.1});
        return mv;
    }

    PostProcessing::PixelKind PostProcessing::classify_pixel(const ModelView& mv, int x, int y) const
    {
        const Basis& basis = mv.basis;
        const std::vector<Sphere>& spheres = mv.spheres;
        const int w = width_, h = height_;

        // Is the first thing the view ray hits (before distance t_limit) one of the bounding spheres?
        auto occluded = [&](D3 d, double t_limit, int skip) {
            for (std::size_t i = 0; i < spheres.size(); ++i)
            {
                double t = 0.0;
                if (static_cast<int>(i) != skip && ray_sphere(basis.eye, d, spheres[i], t) && t < t_limit) return true;
            }
            return false;
        };
        auto floor_point = [&](int fx, int fy, D3& fp) {
            const D3 fd = ray_dir(basis, fx, fy, w, h);
            if (fd.y >= -1.0e-3) return false;
            fp = basis.eye + fd * (-basis.eye.y / fd.y);
            return true;
        };

        const D3 d = ray_dir(basis, x, y, w, h);
        if (d.y > 0.02) return occluded(d, 1.0e30, -1) ? PixelKind::None : PixelKind::Sky;
        D3 p, px, py;
        if (!floor_point(x, y, p) || !floor_point(x + 1, y, px) || !floor_point(x, y + 1, py)) return PixelKind::None;
        if (std::abs(p.x) > 30.0 || std::abs(p.z) > 30.0 || length(p - basis.eye) > 10.0) return PixelKind::None;
        const double t = -basis.eye.y / d.y;
        if (occluded(d, t + 0.01, -1)) return PixelKind::None;
        // Away from every square edge by more than the footprint of the pixel (the checker filter width).
        const double margin = std::max({0.2, std::abs(px.x - p.x) + std::abs(py.x - p.x) + 0.05,
                                        std::abs(px.z - p.z) + std::abs(py.z - p.z) + 0.05});
        const double fx = fract(p.x), fz = fract(p.z);
        if (std::min(fx, 1.0 - fx) < margin || std::min(fz, 1.0 - fz) < margin) return PixelKind::None;
        return PixelKind::Floor;
    }

    std::optional<Rgb> PostProcessing::model_value(const ModelView& mv, int x, int y) const
    {
        const D3 d = ray_dir(mv.basis, x, y, width_, height_);
        switch (classify_pixel(mv, x, y))
        {
        case PixelKind::Sky: return sky_cpu(d);
        case PixelKind::Floor: return floor_cpu(mv.basis.eye + d * (-mv.basis.eye.y / d.y), frame_);
        case PixelKind::None: break;
        }
        return std::nullopt;
    }

    PostProcessing::Probes PostProcessing::find_probes() const
    {
        const ModelView mv = model_view();
        const Basis& basis = mv.basis;
        const std::vector<Sphere>& spheres = mv.spheres;
        const int w = width_, h = height_;
        // Is the first thing the view ray hits (before distance t_limit) one of the bounding spheres?
        auto occluded = [&](D3 d, double t_limit, int skip) {
            for (std::size_t i = 0; i < spheres.size(); ++i)
            {
                double t = 0.0;
                if (static_cast<int>(i) != skip && ray_sphere(basis.eye, d, spheres[i], t) && t < t_limit) return true;
            }
            return false;
        };

        Probes probes;
        std::vector<std::array<int, 2>> floor_candidates, sky_candidates;
        for (int y = 3; y < h - 2; y += 5)
            for (int x = 4; x < w - 3; x += 5)
            {
                const PixelKind kind = classify_pixel(mv, x, y);
                if (kind == PixelKind::Sky) sky_candidates.push_back({x, y});
                else if (kind == PixelKind::Floor) floor_candidates.push_back({x, y});
            }
        auto spread = [](const std::vector<std::array<int, 2>>& all, std::size_t wanted) {
            std::vector<std::array<int, 2>> out;
            for (std::size_t i = 0; i < wanted && !all.empty(); ++i) out.push_back(all[i * all.size() / wanted]);
            return out;
        };
        probes.floor = spread(floor_candidates, 24);
        probes.sky = spread(sky_candidates, 16);

        // Lamp bulbs: the pixel nearest to the CPU-projected center of each lamp (model-view-projection, perspective divide,
        // viewport transform) whose view ray hits the core of the bulb (70% of its radius, well inside the tessellated
        // silhouette) before any other bounding sphere. Searching the neighbourhood instead of taking the center pixel only
        // keeps the probe when, at some window size, the center pixel happens to be hidden or to fall just off the surface.
        const std::size_t first_lamp = frame_.items.size() - 2;
        constexpr int kLampSearch = 4;  // pixels around the projected center
        for (std::size_t i = 0; i < 2; ++i)
        {
            const Lamp& lamp = frame_.lamps[i];
            const math::Vec4 clip = frame_.view_proj * math::Vec4{lamp.position.x, lamp.position.y, lamp.position.z, 1.0f};
            if (clip.w <= 0.0f) continue;
            const double px = (static_cast<double>(clip.x / clip.w) * 0.5 + 0.5) * w;
            const double py = (static_cast<double>(clip.y / clip.w) * 0.5 + 0.5) * h;
            const int cx = static_cast<int>(std::floor(px)), cy = static_cast<int>(std::floor(py));
            const Sphere core{to_d(lamp.position), 0.7 * static_cast<double>(frame_.items[first_lamp + i].radius)};
            bool found = false;
            for (int r = 0; r <= kLampSearch && !found; ++r)  // rings of growing Chebyshev distance
                for (int y = cy - r; y <= cy + r && !found; ++y)
                    for (int x = cx - r; x <= cx + r && !found; ++x)
                    {
                        if (std::max(std::abs(x - cx), std::abs(y - cy)) != r || x < 0 || y < 0 || x >= w || y >= h) continue;
                        const D3 d = ray_dir(basis, x, y, w, h);
                        double t_core = 0.0, t_lamp = 0.0;
                        if (!ray_sphere(basis.eye, d, core, t_core) || !ray_sphere(basis.eye, d, spheres[first_lamp + i], t_lamp)) continue;
                        if (occluded(d, t_lamp + 1.0e-6, static_cast<int>(first_lamp + i))) continue;
                        probes.lamp.push_back({x, y, static_cast<int>(i)});
                        found = true;
                    }
        }
        return probes;
    }

    // The sky gradient, the floor, the lamp bulbs: pixels whose exact value follows from the camera, the lights and the
    // formulas of the shaders, for the RGBA8 scene (stored linear) and the SRGB8_ALPHA8 scene (stored encoded).
    void PostProcessing::check_scene(demo::Check& check, const Bytes& a, const Bytes& s, const Probes& probes)
    {
        const Basis basis = make_basis(frame_);
        Cmp sky_a, sky_s, floor_a, floor_s, lamp_a, lamp_s;
        auto probe_value = [&](int x, int y, const Rgb& linear, Cmp& ca, Cmp& cs, const char* kind) {
            for (std::size_t c = 0; c < 3; ++c)
            {
                auto where = [&] { return std::format("{} probe ({},{}) channel {}", kind, x, y, c); };
                ca.add(static_cast<double>(a.at(x, y, static_cast<int>(c))), linear[c], kPassTolerance, where);
                cs.add(static_cast<double>(s.at(x, y, static_cast<int>(c))), srgb_encode(linear[c]), kPassTolerance, where);
            }
        };
        for (const auto& p : probes.sky) probe_value(p[0], p[1], sky_cpu(ray_dir(basis, p[0], p[1], width_, height_)), sky_a, sky_s, "sky");
        for (const auto& p : probes.floor)
        {
            const D3 d = ray_dir(basis, p[0], p[1], width_, height_);
            const D3 point = basis.eye + d * (-basis.eye.y / d.y);
            probe_value(p[0], p[1], floor_cpu(point, frame_), floor_a, floor_s, "floor");
        }
        for (const auto& p : probes.lamp) probe_value(p[0], p[1], to_rgb(frame_.lamps[static_cast<std::size_t>(p[2])].emissive), lamp_a, lamp_s, "lamp");

        check.expect(probes.sky.size() >= 8 && probes.floor.size() >= 12 && probes.lamp.size() >= 1,
                     "probe selection found unoccluded sky, floor and lamp pixels",
                     std::format("{} sky, {} floor (away from square edges and every object), {} lamp pixels", probes.sky.size(),
                                 probes.floor.size(), probes.lamp.size()));
        report(check, "scene RGBA8: sky gradient equals smoothstep(0, 0.25, ray elevation) mix of horizon and zenith", sky_a, kPassTolerance);
        report(check, "scene RGBA8: floor pixels equal the CPU Blinn-Phong (directional + 2 point lights), filtered checker and fog", floor_a, kPassTolerance);
        report(check, "scene RGBA8: lamp bulbs at their CPU-projected centers show the emissive color", lamp_a, kPassTolerance);
        report(check, "scene SRGB8_ALPHA8: stored sky bytes are the sRGB encoding (equation 4.1) of the exact linear gradient", sky_s, kPassTolerance);
        report(check, "scene SRGB8_ALPHA8: stored floor bytes are the sRGB encoding of the exact CPU lighting", floor_s, kPassTolerance);
        report(check, "scene SRGB8_ALPHA8: stored lamp bytes are the sRGB encoding of the emissive color", lamp_s, kPassTolerance);
    }

    // FBO sRGB: encode on write, color encoding queries and the relation between the two scene images.
    void PostProcessing::check_srgb_target(demo::Check& check, const Bytes& a, const Bytes& s)
    {
        auto encoding = [](mgl::FramebufferId fbo) {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            GLint v = 0;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                                       mgl::FramebufferAttachmentParameter::ColorEncoding, &v);
            return v;
        };
        check.expect_eq(encoding(scene_b_.fbo), gl(GL_SRGB), "FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING of the SRGB8_ALPHA8 scene texture is GL_SRGB");
        check.expect_eq(encoding(scene_a_.fbo), gl(GL_LINEAR), "FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING of the RGBA8 scene texture is GL_LINEAR");
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

        // The same shader output went into both targets. a is round(255 v) of the exact linear value v; s is round(255 enc(v)).
        // So |s - 255 enc(a/255)| <= 255 enc'(xi) * (error of a / 255) + error of s, with 0.6 steps for each conversion (see
        // kPassTolerance) and xi the lower end of the interval (enc' falls with x): a per-pixel tolerance that is 7.7 + 0.6 steps
        // in the darkest areas and about 1.2 steps in bright ones. This also compares the depth test of the two targets (a
        // missing depth buffer would draw the geometry in submission order).
        Cmp cmp;
        std::size_t dark_pixels = 0;
        for (int y = 0; y < a.h; ++y)
            for (int x = 0; x < a.w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    const double l = static_cast<double>(a.at(x, y, c)) / 255.0;
                    const double slope = srgb_encode_slope(std::max(l - 0.6 / 255.0, 0.0));
                    cmp.add(static_cast<double>(s.at(x, y, c)), srgb_encode(l), 0.6 * slope + 0.6 + 0.02, [&] { return std::format("pixel ({},{}) channel {}", x, y, c); });
                    if (a.at(x, y, c) < 8) ++dark_pixels;
                }
        report(check, "SRGB8_ALPHA8 scene = encode(RGBA8 scene) at every pixel within the quantization of the 8-bit linear image", cmp, 0.0);

        // The comparison above matters most for dark values (< 8/255 linear), where the RGBA8 target is coarse and the
        // tolerance is widest. How many of them the image must contain follows from the CPU reference: every sky and floor
        // pixel the model can predict (classify_pixel) whose exact value v has 255 v + kPassTolerance < 7.5 must be stored as
        // a byte below 8 (check_scene holds the bytes to 255 v +- kPassTolerance). Those predicted channels are a subset of
        // the image, so the image has at least that many dark values. The objects and the excluded pixels can only add more.
        const ModelView mv = model_view();
        std::size_t predicted_dark = 0, predicted_not_dark = 0, model_channels = 0;
        std::string first_not_dark;
        for (int y = 0; y < a.h; ++y)
            for (int x = 0; x < a.w; ++x)
            {
                const std::optional<Rgb> v = model_value(mv, x, y);
                if (!v) continue;
                for (int c = 0; c < 3; ++c)
                {
                    ++model_channels;
                    if (255.0 * clamp01((*v)[static_cast<std::size_t>(c)]) + kPassTolerance >= 7.5) continue;
                    ++predicted_dark;
                    if (a.at(x, y, c) >= 8 && predicted_not_dark++ == 0)
                        first_not_dark = std::format("pixel ({},{}) channel {}: got {} expected {:.2f}", x, y, c, a.at(x, y, c),
                                                     255.0 * clamp01((*v)[static_cast<std::size_t>(c)]));
                }
            }
        check.expect(predicted_dark > 0 && predicted_not_dark == 0 && dark_pixels >= predicted_dark,
                     "the scene has the dark channel values (< 8/255 linear, where the RGBA8 target is coarse) that the CPU model predicts: "
                     "every predicted one is dark, so the comparison above covers them",
                     predicted_not_dark == 0
                         ? std::format("{} dark channel values in the image, at least the {} the model predicts among the {} channel values it knows",
                                       dark_pixels, predicted_dark, model_channels)
                         : std::format("{} of {} predicted dark channel values are not dark; first: {}", predicted_not_dark, predicted_dark, first_not_dark));
    }

    // Validation (see the header comment of this file).
    void PostProcessing::validate(demo::Env& /*env*/, demo::Check& check)
    {
        const int w = width_, h = height_;
        const auto count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);

        // 0. The image on screen right now, before this function draws anything.
        const Bytes shown = read_fbo(mgl::FramebufferId{0});

        // 1. The scene again, in both targets, from the state of the last frame.
        draw_scene(scene_a_.fbo);
        draw_scene(scene_b_.fbo);
        const Bytes a = read_fbo(scene_a_.fbo);
        const Bytes s = read_fbo(scene_b_.fbo);
        const Probes probes = find_probes();
        check_scene(check, a, s, probes);
        check_srgb_target(check, a, s);
        Maps maps;
        maps.scene = to_image(a);
        maps.scene_srgb = to_image(s, /*srgb_decode_rgb=*/true);
        maps.focus = static_cast<double>(focus_distance());

        // 2. Separable Gaussian blur of the scene: horizontal pass into the ping target, vertical pass into the result.
        const double sigma = static_cast<double>(sigma_for(kBlurRadiusDefault));
        pass_blur(scene_a_.tex, tmp_, 1, 0, kBlurRadiusDefault);
        const Bytes h_gpu = read_fbo(tmp_.fbo);
        report(check, std::format("blur, horizontal pass (radius {}, sigma {:.1f}) equals the CPU Gaussian of the scene", kBlurRadiusDefault, sigma),
               compare_images(h_gpu, blur_cpu(maps.scene, 1, 0, kBlurRadiusDefault, sigma), kPassTolerance), kPassTolerance);
        pass_blur(tmp_.tex, blur_t_, 0, 1, kBlurRadiusDefault);
        const Bytes v_gpu = read_fbo(blur_t_.fbo);
        const Image h_image = to_image(h_gpu);
        report(check, "blur, vertical pass equals the CPU Gaussian of the horizontal pass's stored bytes", compare_images(v_gpu, blur_cpu(h_image, 0, 1, kBlurRadiusDefault, sigma), kPassTolerance),
               kPassTolerance);
        report(check, "blur, whole chain (CPU: horizontal, 8-bit round, vertical) equals the GPU result",
               compare_images(v_gpu, blur_cpu(quantize(blur_cpu(maps.scene, 1, 0, kBlurRadiusDefault, sigma)), 0, 1, kBlurRadiusDefault, sigma), kChainTolerance),
               kChainTolerance);
        maps.blur = to_image(v_gpu);

        // Negative controls for the blur check: the same GPU image against two wrong CPU models.
        const Cmp wrong_radius = compare_images(h_gpu, blur_cpu(maps.scene, 1, 0, kBlurRadiusDefault - 3, sigma), kPassTolerance);
        report_control(check, std::format("a horizontal blur with radius {} instead of {} disagrees with the GPU", kBlurRadiusDefault - 3, kBlurRadiusDefault),
                       wrong_radius, count / 50);
        const Cmp box = compare_images(h_gpu, blur_cpu(maps.scene, 1, 0, kBlurRadiusDefault, sigma, /*box=*/true), kPassTolerance);
        report_control(check, "a box kernel (equal weights) instead of the Gaussian disagrees with the GPU", box, count / 50);
        const Cmp unblurred = compare_images(h_gpu, maps.scene, kPassTolerance);
        report_control(check, "the unblurred scene disagrees with the blurred result (the pass changed the image)", unblurred, count / 20);

        // 3. Bloom: bright-pass, then the same separable blur.
        pass_bright(scene_a_.tex, bright_t_);
        const Bytes bright_gpu = read_fbo(bright_t_.fbo);
        report(check, "bloom bright-pass equals max(c - 0.7, 0) / 0.3 per channel of the scene",
               compare_images(bright_gpu, bright_cpu(maps.scene, static_cast<double>(kBrightThreshold)), kPassTolerance), kPassTolerance);
        const Image bright_image = to_image(bright_gpu);
        pass_blur(bright_t_.tex, tmp_, 1, 0, kBlurRadiusDefault);
        const Bytes bh_gpu = read_fbo(tmp_.fbo);
        report(check, "bloom blur, horizontal pass equals the CPU Gaussian of the bright-pass",
               compare_images(bh_gpu, blur_cpu(bright_image, 1, 0, kBlurRadiusDefault, sigma), kPassTolerance), kPassTolerance);
        pass_blur(tmp_.tex, bloom_t_, 0, 1, kBlurRadiusDefault);
        const Bytes bv_gpu = read_fbo(bloom_t_.fbo);
        report(check, "bloom blur, vertical pass equals the CPU Gaussian of the horizontal pass",
               compare_images(bv_gpu, blur_cpu(to_image(bh_gpu), 0, 1, kBlurRadiusDefault, sigma), kPassTolerance), kPassTolerance);
        maps.bloom = to_image(bv_gpu);
        std::size_t lit = 0;
        for (const Rgb& p : bright_image.px) lit += (p[0] + p[1] + p[2] > 0.0) ? 1 : 0;
        check.expect(lit > count / 200 && lit < count / 2, "the bright-pass keeps a small, non-empty part of the image (lamps and highlights)",
                     std::format("{} of {} pixels", lit, count));

        // 4. Sobel.
        pass_sobel(scene_a_.tex, edge_t_);
        const Bytes edge_gpu = read_fbo(edge_t_.fbo);
        maps.edge = to_image(edge_gpu);
        report(check, "Sobel edge map equals the CPU 3x3 Sobel (sqrt-luma, smoothstep(0.15, 0.6, |G|)) of the scene",
               compare_images(edge_gpu, sobel_cpu(maps.scene), kPassTolerance), kPassTolerance);
        report_control(check, "a Sobel with the horizontal kernel only disagrees with the GPU",
                       compare_images(edge_gpu, sobel_cpu(maps.scene, /*horizontal_only=*/true), kPassTolerance), count / 100);
        std::size_t edge_pixels = 0;
        for (const Rgb& p : maps.edge.px) edge_pixels += p[0] > 0.25 ? 1 : 0;
        check.expect(edge_pixels > count / 200 && edge_pixels < count / 3, "the edge map has strong edges on a small part of the image",
                     std::format("{} of {} pixels above 25%", edge_pixels, count));

        // 5. The depth texture, read through the depth-dump pass, against the analytic depth of the floor.
        pass_depth_dump(tmp_);
        const Bytes dump = read_fbo(tmp_.fbo);
        maps.depth.resize(count);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const double k = static_cast<double>(dump.at(x, y, 0)) * 65536.0 + static_cast<double>(dump.at(x, y, 1)) * 256.0 + static_cast<double>(dump.at(x, y, 2));
                maps.depth[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = k / 16777215.0;
            }
        {
            const Basis basis = make_basis(frame_);
            const double n = static_cast<double>(kZNear), f = static_cast<double>(kZFar);
            double worst = 0.0;
            for (const auto& p : probes.floor)
            {
                const D3 d = ray_dir(basis, p[0], p[1], w, h);
                const D3 point = basis.eye + d * (-basis.eye.y / d.y);
                const double z_eye = dot(point - basis.eye, basis.forward);  // distance along the view axis
                const double want = 0.5 * ((f + n) / (f - n) - 2.0 * f * n / ((f - n) * z_eye)) + 0.5;
                worst = std::max(worst, std::abs(maps.depth[static_cast<std::size_t>(p[1]) * static_cast<std::size_t>(w) + static_cast<std::size_t>(p[0])] - want));
            }
            // A 24-bit depth step is 6e-8. The rasterizer interpolates window-space z linearly in screen space from
            // clip-space vertices computed in float (ES 3.0 section 3.6.1). ES 3.0 states no depth accuracy, so 2e-5 is
            // a sanity bound, not a precision claim: with near 0.5 and far 80 it corresponds to 0.004 world units at 10
            // units from the camera. The measured value is printed.
            check.expect(worst <= 2.0e-5, "depth texture (via the depth-dump pass) matches the analytic window-space depth of the floor probes",
                         std::format("max |depth - analytic| = {:.2e} over {} probes (bound 2e-5)", worst, probes.floor.size()));
            std::size_t clear_depth = 0;
            for (const auto& p : probes.sky) clear_depth += maps.depth[static_cast<std::size_t>(p[1]) * static_cast<std::size_t>(w) + static_cast<std::size_t>(p[0])] == 1.0 ? 1 : 0;
            check.expect(clear_depth == probes.sky.size(), "sky pixels keep the cleared depth 1.0", std::format("{} of {}", clear_depth, probes.sky.size()));
        }

        // 6. The presentation of every look, from the images read above.
        const double present_tolerance = fb_srgb_ ? 2.0 : kPassTolerance;
        for (int mode = 0; mode <= kModeOverview; ++mode)
        {
            pass_present(mode);
            const Bytes got = read_fbo(mgl::FramebufferId{0});
            report(check, std::format("present: {} equals the CPU model", kModeNames[static_cast<std::size_t>(mode)]),
                   compare_images(got, present_image(mode, maps), present_tolerance), present_tolerance);
            if (mode == 1)
            {
                // sRGB curve vs plain gamma 2.2: differs by up to ~8 steps in the dark sky.
                report_control(check, "a plain gamma-2.2 power curve instead of the sRGB curve disagrees with the GPU's gamma pass",
                               compare_images(got, present_image(1, maps, /*plain_gamma_22=*/true), present_tolerance), count / 20);
            }
            if (mode == 2)
            {
                Maps no_decode = maps;
                no_decode.scene_srgb = to_image(s, /*srgb_decode_rgb=*/false);  // what a lookup would return without the decode
                report_control(check, "an SRGB8_ALPHA8 lookup that skipped the decode (stored bytes used as linear) disagrees with the GPU",
                               compare_images(got, present_image(2, no_decode), present_tolerance), count / 20);
            }
        }

        // 7. The image that was on screen when validation started is the overview of this very state.
        report(check, "the frame on screen equals the CPU model of the selected look", compare_images(shown, present_image(mode_, maps), present_tolerance),
               present_tolerance);

        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        check.expect_no_gl_error("no GL error while validating");

        // Leave the window showing the frame (the runtime captures the default framebuffer after validate()).
        draw_frame();
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-post-processing",
        .title = "Post-processing: gamma, blur, Sobel, bloom, depth of field",
        .min_es = {3, 0},
        .summary = "An animated, lit 3D scene is rendered into an offscreen framebuffer (color texture + depth) and post-processed "
                   "with full-screen passes: sRGB gamma (shader and SRGB8_ALPHA8 target), separable Gaussian blur with ping-pong "
                   "targets, Sobel edges, bloom and depth of field. Every pass is checked against a CPU model of the same pixels.",
        .controls = "  0 overview (all looks as stripes)   1 raw  2 gamma  3 gamma with SRGB8_ALPHA8 scene  4 blur  5 sobel  6 bloom  7 depth of field\n"
                    "  n / p next / previous look   b blur radius (2, 5, 10, 16)\n"
                    "  a / d orbit camera   w / s dolly   q / e height   c toggle automatic orbit   r reset camera   space pause   esc quit",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 45,
        .depth_bits = 0,
        .stencil_bits = 0,
    };
    return demo::run<PostProcessing>(argc, argv, info);
}
