// es30-interactive-3d-scene
//
// The capstone of the learning path: one small outdoor scene that combines the techniques of the earlier
// demos, explored with an FPS camera.
//
//   es20-transformations-camera  model / view / projection chain, here driven by mouse-look + WASD
//   es30-lighting                Blinn-Phong (and Phong) directional sun light, hemisphere ambient
//   es20-textures-2d / mipmaps   procedural CPU textures with complete, CPU-built mip chains
//   es30-normal-mapping          tangent-space normal map on the stone wall and the plinth
//   es30-shadow-mapping          depth-only FBO pass, sampler2DShadow, hardware + 3x3 PCF, polygon offset
//   es30-skybox-and-fog          cube-map sky drawn last at depth 1.0, exponential fog to the horizon colour
//   es30-instancing              pillars and trees drawn with glDrawArraysInstanced + glVertexAttribDivisor
//
// Frame:  pass 1 shadow map (sun's view, depth only)  ->  pass 2 scene (camera)  ->  pass 3 sky  ->  inset.
//
// Controls (windowed): left mouse button + drag looks around, W/A/S/D move, Q/E down/up, wheel zooms
// (field of view), 1 shadows, 2 normal mapping, 3 fog, 4 wireframe overlay, 5 shadow-map inset,
// 6 Blinn-Phong <-> Phong, R resets the camera.
//
// Validation (--validate) renders fixed camera poses through the same code path that input drives and
// compares stable pixels with an independent CPU ray caster of the scene (double precision): sky colour per
// direction, lit and shadowed ground, Blinn-Phong on a glossy face, normal-mapped bevels, every pillar
// instance at its projected position, fog at known distances. Every technique toggle is shown to change
// the image exactly where the CPU predicts, and a scripted input sequence (mouse, wheel, held keys at a
// fixed frame time) must end at the CPU-predicted camera pose. See README.md.

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
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    // =============================================================================================
    // 1. Double-precision vectors for the CPU side (independent of the float math the GPU path uses)
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
    constexpr double comp(V3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

    /// Rotation about +Y by `a` radians: the convention of demo::math::rotate(a, {0, 1, 0}).
    inline V3 rot_y(V3 v, double a)
    {
        const double c = std::cos(a), s = std::sin(a);
        return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
    }

    dm::Vec3 to_vec3(V3 v) { return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    V3 gamma(V3 c)
    {
        const double e = 1.0 / 2.2;
        return {std::pow(std::clamp(c.x, 0.0, 1.0), e), std::pow(std::clamp(c.y, 0.0, 1.0), e), std::pow(std::clamp(c.z, 0.0, 1.0), e)};
    }

    // =============================================================================================
    // 2. The scene. The GPU path and the CPU reference read the same constants.
    // =============================================================================================
    constexpr double kSunAzimuth = radians(25.0);    // in the x/z plane, from +X towards +Z
    constexpr double kSunElevation = radians(35.0);
    V3 sun_dir()  // unit vector TOWARDS the sun
    {
        return {std::cos(kSunElevation) * std::cos(kSunAzimuth), std::sin(kSunElevation), std::cos(kSunElevation) * std::sin(kSunAzimuth)};
    }
    constexpr V3 kSunColor{1.25, 1.15, 1.00};
    constexpr V3 kSkyAmbient{0.30, 0.37, 0.50};
    constexpr V3 kGroundAmbient{0.18, 0.16, 0.12};
    constexpr V3 kHorizon{0.60, 0.70, 0.82};  // linear sky colour at the horizon == fog colour
    constexpr V3 kZenith{0.10, 0.22, 0.55};
    constexpr double kFogDensity = 0.016;     // f = exp(-density * distance)
    constexpr V3 kWireColor{1.0, 0.25, 0.85}; // display colour of the wireframe overlay

    // Shadow map: an orthographic view along -L that covers every object of the scene.
    constexpr int kShadowSize = 2048;
    constexpr double kLightDistance = 50.0, kLightHalf = 24.0, kLightNear = 15.0, kLightFar = 85.0;
    // glPolygonOffset of pass 1. The corner tap of the 3x3 PCF kernel reads (bilinear) texels up to 2 texels per axis away
    // from the fragment, a depth difference of up to 2 (|dz/dx| + |dz/dy|) <= 2 sqrt(2) |grad z|; eq. 3.10 lets m be as
    // small as |grad z| / sqrt(2). So factor >= 4 keeps every tap of a lit plane lit; 4.5 leaves a margin.
    constexpr float kOffsetFactor = 4.5f, kOffsetUnits = 4.0f;
    constexpr float kReceiverBias = 2.0e-4f;                     // window-z bias of the comparison in pass 2

    constexpr double kNear = 0.1, kFar = 1000.0;
    constexpr double kGroundHalf = 400.0;  // the ground ends where the fog has made it the horizon colour
    constexpr int kGroundCells = 100;      // 8 x 8 unit cells
    constexpr double kGroundTile = 4.0;    // world units per repetition of the grass texture
    constexpr int kTexSize = 128;
    constexpr int kTexLevels = 8;          // 128 .. 1
    constexpr int kSkySize = 256;
    constexpr double kInsetLo = 0.05, kInsetHi = 0.95;  // depths shown black / white by the shadow-map inset

    enum class Tex
    {
        None = -1,
        Grass = 0,
        Wood = 1,
        Stone = 2,
        Ball = 3,
    };
    constexpr int kTexCount = 4;

    struct Material
    {
        V3 albedo;          // multiplies the texture (white for textured materials)
        Tex tex;
        bool normal_map;    // the stone normal map on unit 1
        double specular;
        double shininess;
    };
    constexpr Material kGroundMat{{1, 1, 1}, Tex::Grass, false, 0.06, 12.0};
    constexpr Material kStoneMat{{1, 1, 1}, Tex::Stone, true, 0.25, 24.0};
    constexpr Material kMonolithMat{{0.07, 0.11, 0.20}, Tex::None, false, 0.60, 64.0};
    constexpr Material kWoodMat{{1, 1, 1}, Tex::Wood, false, 0.15, 16.0};
    constexpr Material kBallMat{{1, 1, 1}, Tex::Ball, false, 0.50, 48.0};
    constexpr Material kPillarMat{{0.80, 0.76, 0.66}, Tex::None, false, 0.15, 16.0};
    constexpr Material kTrunkMat{{0.30, 0.20, 0.12}, Tex::None, false, 0.05, 8.0};
    constexpr Material kCanopyMat{{1, 1, 1}, Tex::None, false, 0.05, 8.0};  // colour comes from the instance tint

    /// A box: rotated about +Y by yaw, texture coordinates = (position along the face's tangent axes) * uv_scale + uv_offset.
    struct BoxDef
    {
        const char* name;
        V3 center;
        V3 half;
        double yaw_deg;
        Material mat;
        double uv_scale;
        double uv_offset;
    };
    constexpr BoxDef kWall{"stone wall", {-4.5, 1.0, -4.0}, {3.0, 1.0, 0.3}, 12.0, kStoneMat, 0.25, 0.25};
    constexpr BoxDef kPlinth{"plinth", {2.6, 0.15, -1.5}, {1.0, 0.15, 1.0}, -25.0, kStoneMat, 0.25, 0.25};
    constexpr BoxDef kMonolith{"monolith", {2.6, 1.9, -1.5}, {0.55, 1.6, 0.55}, -25.0, kMonolithMat, 0.25, 0.0};
    constexpr BoxDef kCrate{"crate", {-1.5, 0.6, 1.8}, {0.6, 0.6, 0.6}, 30.0, kWoodMat, 1.0 / 1.2, 0.5};
    constexpr V3 kBallCenter{1.0, 0.7, 3.2};
    constexpr double kBallRadius = 0.7;

    // Instanced prototypes. Instance i is placed at offset_i + scale_i * (prototype position).
    constexpr V3 kPillarCenter{0, 1.6, 0}, kPillarHalf{0.3, 1.6, 0.3};
    constexpr V3 kTrunkCenter{0, 0.8, 0}, kTrunkHalf{0.14, 0.8, 0.14};
    constexpr V3 kCanopyCenter{0, 2.15, 0};
    constexpr double kCanopyRadius = 0.95;

    struct Instance
    {
        V3 offset;
        double scale;
        V3 tint;
    };
    constexpr int kPillarCount = 6;
    constexpr std::array<Instance, kPillarCount> kPillars = {{
        {{7.5, 0, -9.0}, 1.00, {1.00, 1.00, 1.00}},
        {{7.5, 0, -7.0}, 1.08, {0.96, 0.98, 1.04}},
        {{7.5, 0, -5.0}, 0.94, {1.04, 0.99, 0.94}},
        {{7.5, 0, -3.0}, 1.12, {1.00, 0.95, 0.92}},
        {{7.5, 0, -1.0}, 0.97, {0.95, 1.00, 1.00}},
        {{7.5, 0, 1.0}, 1.05, {1.03, 1.02, 0.97}},
    }};
    constexpr int kTreeCount = 10;
    constexpr std::array<Instance, kTreeCount> kTrees = {{
        {{-9.5, 0, -9.0}, 1.10, {0.16, 0.40, 0.12}},
        {{-13.0, 0, -2.5}, 0.90, {0.20, 0.42, 0.10}},
        {{-10.5, 0, 5.0}, 1.20, {0.13, 0.36, 0.15}},
        {{-15.0, 0, -12.5}, 1.30, {0.18, 0.38, 0.09}},
        {{-5.5, 0, -13.5}, 1.00, {0.22, 0.44, 0.13}},
        {{2.5, 0, -14.5}, 1.25, {0.15, 0.39, 0.11}},
        {{10.5, 0, -12.5}, 0.95, {0.19, 0.41, 0.14}},
        {{15.5, 0, 4.5}, 1.10, {0.14, 0.37, 0.10}},
        {{-16.5, 0, 9.0}, 1.00, {0.21, 0.40, 0.12}},
        {{6.0, 0, 9.5}, 0.90, {0.17, 0.43, 0.12}},
    }};

    // Shape ids of the CPU model (and the draw order of the GPU objects after the ground).
    constexpr int kIdSky = -2, kIdGround = -1;
    constexpr int kIdWall = 0, kIdPlinth = 1, kIdMonolith = 2, kIdCrate = 3, kIdBall = 4, kIdPillar0 = 5;
    constexpr int kIdTrunk0 = kIdPillar0 + kPillarCount, kIdCanopy0 = kIdTrunk0 + kTreeCount;

    // =============================================================================================
    // 3. Camera: an FPS pose (position, yaw, pitch, field of view) and what the controls do to it
    // =============================================================================================
    struct Pose
    {
        V3 pos;
        double yaw = 0;    // radians; 0 looks along -Z, positive turns right (towards +X)
        double pitch = 0;  // radians; positive looks up
        double fov = 60;   // degrees, across the SHORTER side of the window
    };
    constexpr double kMoveSpeed = 4.0;           // world units per second
    constexpr double kMouseSensitivity = 0.0025; // radians per pixel of mouse motion
    constexpr double kWheelStep = 2.5;           // degrees of field of view per wheel step
    constexpr double kMinFov = 25.0, kMaxFov = 90.0;
    constexpr double kMaxPitch = radians(85.0);
    constexpr double kMinEyeHeight = 0.3;

    V3 forward_of(const Pose& p) { return {std::cos(p.pitch) * std::sin(p.yaw), std::sin(p.pitch), -std::cos(p.pitch) * std::cos(p.yaw)}; }

    Pose look_from(V3 pos, V3 target, double fov)
    {
        const V3 d = normalize(target - pos);
        return {pos, std::atan2(d.x, -d.z), std::asin(d.y), fov};
    }

    const Pose kStartPose = look_from({1.8, 2.5, 9.5}, {0.3, 1.0, -3.0}, 60.0);

    /// Camera frame + projection for a window size. The CPU ray caster and the GPU matrices both use it.
    struct View
    {
        V3 eye, fwd, right, up;
        double tan_x = 0, tan_y = 0;  // tangents of the half fields of view
        int w = 0, h = 0;
    };

    View make_view(const Pose& p, int w, int h)
    {
        View v;
        v.eye = p.pos;
        v.fwd = forward_of(p);
        v.right = normalize(cross(v.fwd, V3{0, 1, 0}));
        v.up = cross(v.right, v.fwd);
        const double t = std::tan(radians(p.fov) * 0.5), aspect = static_cast<double>(w) / static_cast<double>(h);
        // The field of view spans the shorter side, so a portrait window does not squeeze the scene.
        if (aspect >= 1.0)
        {
            v.tan_y = t;
            v.tan_x = t * aspect;
        }
        else
        {
            v.tan_x = t;
            v.tan_y = t / aspect;
        }
        v.w = w;
        v.h = h;
        return v;
    }

    /// World direction of the ray through window point (wx, wy); the centre of pixel (x, y) is (x + 0.5, y + 0.5).
    V3 ray_dir(const View& v, double wx, double wy)
    {
        const double nx = wx / v.w * 2.0 - 1.0, ny = wy / v.h * 2.0 - 1.0;
        return normalize(v.fwd + v.right * (nx * v.tan_x) + v.up * (ny * v.tan_y));
    }

    /// Window coordinates of a world point (false when behind the near plane).
    bool project(const View& v, V3 p, double& wx, double& wy)
    {
        const V3 rel = p - v.eye;
        const double z = dot(rel, v.fwd);
        if (z < kNear) return false;
        wx = (dot(rel, v.right) / (z * v.tan_x) * 0.5 + 0.5) * v.w;
        wy = (dot(rel, v.up) / (z * v.tan_y) * 0.5 + 0.5) * v.h;
        return true;
    }

    dm::Mat4 view_matrix(const View& v) { return dm::look_at(to_vec3(v.eye), to_vec3(v.eye + v.fwd), {0, 1, 0}); }
    dm::Mat4 proj_matrix(const View& v)
    {
        return dm::perspective(static_cast<float>(2.0 * std::atan(v.tan_y)), static_cast<float>(v.tan_x / v.tan_y), static_cast<float>(kNear),
                               static_cast<float>(kFar));
    }

    // =============================================================================================
    // 4. Procedural textures, painted on the CPU. The CPU copy of every mip level stays for the reference.
    // =============================================================================================
    double hash01(int x, int y, int seed)
    {
        std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393u + static_cast<std::uint32_t>(y) * 668265263u +
                          static_cast<std::uint32_t>(seed) * 2246822519u;
        h = (h ^ (h >> 13)) * 1274126177u;
        h ^= h >> 16;
        return static_cast<double>(h & 0xFFFFFFu) / static_cast<double>(0x1000000);
    }

    /// Smooth value noise with a lattice that repeats every `period` cells, so the textures tile.
    double value_noise(double x, double y, int period, int seed)
    {
        const int xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y));
        const double fx = x - xi, fy = y - yi;
        const double sx = fx * fx * (3.0 - 2.0 * fx), sy = fy * fy * (3.0 - 2.0 * fy);
        auto lattice = [&](int i, int j) { return hash01(((i % period) + period) % period, ((j % period) + period) % period, seed); };
        const double a = lattice(xi, yi), b = lattice(xi + 1, yi), c = lattice(xi, yi + 1), d = lattice(xi + 1, yi + 1);
        return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy;
    }

    /// RGBA8 texture with its whole mip chain. Row j of a level holds t = (j + 0.5) / size (the first row uploaded is t = 0).
    struct TexImage
    {
        int size = 0;
        std::vector<std::vector<std::uint8_t>> levels;

        [[nodiscard]] int level_size(int k) const { return std::max(1, size >> k); }
        /// Texel (i, j) of level k with REPEAT wrapping.
        [[nodiscard]] const std::uint8_t* texel(int k, int i, int j) const
        {
            const int s = level_size(k);
            i = ((i % s) + s) % s;
            j = ((j % s) + s) % s;
            return &levels[static_cast<std::size_t>(k)][(static_cast<std::size_t>(j) * static_cast<std::size_t>(s) + static_cast<std::size_t>(i)) * 4];
        }
    };

    template <typename Fn>
    std::vector<std::uint8_t> paint(int size, Fn&& fn)
    {
        std::vector<std::uint8_t> px;
        px.reserve(static_cast<std::size_t>(size * size * 4));
        for (int j = 0; j < size; ++j)
            for (int i = 0; i < size; ++i)
            {
                const V3 c = fn(i, j);
                px.insert(px.end(), {to8(c.x), to8(c.y), to8(c.z), 255});
            }
        return px;
    }

    /// Box-filtered mip chain built on the CPU (glGenerateMipmap would leave the filter to the implementation, ES 3.0
    /// 3.8.10). A normal map averages the decoded vectors and renormalizes them.
    TexImage build_mips(std::vector<std::uint8_t> base, bool normal_map)
    {
        TexImage t;
        t.size = kTexSize;
        t.levels.push_back(std::move(base));
        for (int s = kTexSize / 2; s >= 1; s /= 2)
        {
            const std::size_t prev = t.levels.size() - 1;
            std::vector<std::uint8_t> next(static_cast<std::size_t>(s * s * 4));
            for (int j = 0; j < s; ++j)
                for (int i = 0; i < s; ++i)
                {
                    std::uint8_t* out = &next[(static_cast<std::size_t>(j) * static_cast<std::size_t>(s) + static_cast<std::size_t>(i)) * 4];
                    const std::uint8_t* q[4] = {t.texel(static_cast<int>(prev), 2 * i, 2 * j), t.texel(static_cast<int>(prev), 2 * i + 1, 2 * j),
                                                t.texel(static_cast<int>(prev), 2 * i, 2 * j + 1), t.texel(static_cast<int>(prev), 2 * i + 1, 2 * j + 1)};
                    if (normal_map)
                    {
                        V3 n{};
                        for (const std::uint8_t* p : q) n = n + V3{p[0] / 127.5 - 1.0, p[1] / 127.5 - 1.0, p[2] / 127.5 - 1.0};
                        n = normalize(n);
                        out[0] = to8(n.x * 0.5 + 0.5);
                        out[1] = to8(n.y * 0.5 + 0.5);
                        out[2] = to8(n.z * 0.5 + 0.5);
                        out[3] = 255;
                    }
                    else
                        for (std::size_t c = 0; c < 4; ++c)
                            out[c] = static_cast<std::uint8_t>((q[0][c] + q[1][c] + q[2][c] + q[3][c] + 2) / 4);
                }
            t.levels.push_back(std::move(next));
        }
        return t;
    }

    /// Meadow: two octaves of tileable value noise, a little per-texel grain and a yellowish hue drift.
    V3 grass_texel(int i, int j)
    {
        const double n1 = value_noise(i / 16.0, j / 16.0, 8, 11), n2 = value_noise(i / 4.0, j / 4.0, 32, 12), r = hash01(i, j, 13);
        const double f = 0.80 + 0.24 * n1 + 0.10 * n2 + 0.06 * r;
        return lerp({0.19, 0.34, 0.09}, {0.26, 0.35, 0.10}, n1) * f;
    }

    /// Crate: a dark frame around four horizontal planks separated by gaps, with wavy grain.
    V3 wood_texel(int i, int j)
    {
        const double grain = 0.88 + 0.12 * (0.5 + 0.5 * std::sin(j * 1.3 + 5.0 * value_noise(i / 32.0, j / 4.0, 4, 21)));
        const bool frame = i < 12 || i >= 116 || j < 12 || j >= 116;
        if (frame)
        {
            const bool seam = i == 12 - 1 || i == 116 || j == 12 - 1 || j == 116 || i == 0 || j == 0 || i == 127 || j == 127;
            return V3{0.33, 0.20, 0.09} * (grain * (seam ? 0.55 : 1.0));
        }
        const int k = (j - 12) / 26, local = (j - 12) % 26;
        if (local < 2) return {0.09, 0.05, 0.02};
        return V3{0.52, 0.33, 0.16} * ((0.88 + 0.12 * hash01(k, 0, 22)) * grain);
    }

    /// Stone: 2x2 square panels per tile (64 texels each), separated by darker 4-texel grooves.
    V3 stone_texel(int i, int j)
    {
        const int lx = i % 64, ly = j % 64;
        const bool groove = lx < 2 || lx >= 62 || ly < 2 || ly >= 62;
        const double n = value_noise(i / 8.0, j / 8.0, 16, 31), r = hash01(i, j, 32);
        const V3 c = V3{0.56, 0.53, 0.49} * (0.90 + 0.12 * n + 0.05 * r);
        return groove ? c * 0.62 : c;
    }

    /// Stone normal map (tangent space, +Y = +v): raised panels with 8-texel bevels; panel tops and groove bottoms are flat.
    V3 stone_normal(int i, int j)
    {
        const int lx = i % 64, ly = j % 64;
        const int e_left = lx - 2, e_right = 61 - lx, e_bottom = ly - 2, e_top = 61 - ly;
        const int e = std::min({e_left, e_right, e_bottom, e_top});
        V3 n{0, 0, 1};
        if (e >= 0 && e < 8)
        {
            const double s = std::sin(radians(40.0)), c = std::cos(radians(40.0));
            if (e == e_left) n = {-s, 0, c};
            else if (e == e_right) n = {s, 0, c};
            else if (e == e_bottom) n = {0, -s, c};
            else n = {0, s, c};
        }
        return n * 0.5 + V3{0.5, 0.5, 0.5};
    }

    /// Beach ball: six coloured segments around u, white caps at the poles.
    V3 ball_texel(int i, int j)
    {
        const double u = (i + 0.5) / kTexSize, v = (j + 0.5) / kTexSize;
        if (v < 0.12 || v > 0.88) return {0.85, 0.85, 0.82};
        constexpr std::array<V3, 6> seg = {{{0.75, 0.08, 0.06}, {0.85, 0.85, 0.82}, {0.06, 0.20, 0.70}, {0.85, 0.65, 0.05}, {0.85, 0.85, 0.82}, {0.08, 0.55, 0.15}}};
        return seg[static_cast<std::size_t>(std::min(5, static_cast<int>(u * 6.0)))];
    }

    // ---- the sky cube map --------------------------------------------------------------------------------------
    // Face selection, ES 3.0 3.8.9 table 3.21 (faces in target order +X, -X, +Y, -Y, +Z, -Z), from es30-skybox-and-fog.
    struct FaceSt
    {
        int face = 0;
        double s = 0, t = 0;
    };

    FaceSt select_face(V3 r)
    {
        const double ax = std::abs(r.x), ay = std::abs(r.y), az = std::abs(r.z);
        double sc = 0, tc = 0, ma = 0;
        int face = 0;
        if (ax >= ay && ax >= az)
        {
            ma = ax;
            if (r.x >= 0) { face = 0; sc = -r.z; tc = -r.y; }
            else { face = 1; sc = r.z; tc = -r.y; }
        }
        else if (ay >= az)
        {
            ma = ay;
            if (r.y >= 0) { face = 2; sc = r.x; tc = r.z; }
            else { face = 3; sc = r.x; tc = -r.z; }
        }
        else
        {
            ma = az;
            if (r.z >= 0) { face = 4; sc = r.x; tc = -r.y; }
            else { face = 5; sc = -r.x; tc = -r.y; }
        }
        return {face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0)};
    }

    V3 face_direction(int face, double sc, double tc)
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

    /// Linear sky radiance of a unit direction: horizon -> zenith gradient, a glow and a disc around the sun, a haze
    /// below the horizon. `aa` is the angular size of a texel: the disc and the horizon get fractional coverage.
    V3 sky_radiance(V3 d, double aa)
    {
        auto cover = [aa](double signed_distance) { return std::clamp(signed_distance / aa + 0.5, 0.0, 1.0); };
        const double el = std::asin(std::clamp(d.y, -1.0, 1.0));
        V3 c = lerp(kHorizon, kZenith, std::pow(std::max(el, 0.0) / (0.5 * std::numbers::pi), 0.55));
        const double sun_angle = std::acos(std::clamp(dot(d, sun_dir()), -1.0, 1.0));
        c = c + V3{1.0, 0.80, 0.55} * (0.35 * std::exp(-sun_angle / 0.10));
        c = lerp(c, V3{1.0, 0.98, 0.92}, cover(radians(1.8) - sun_angle));
        const V3 below = lerp(kHorizon, V3{0.33, 0.36, 0.30}, 1.0 - std::exp(std::min(el, 0.0) / 0.08));
        c = lerp(below, c, cover(el));
        return {std::clamp(c.x, 0.0, 1.0), std::clamp(c.y, 0.0, 1.0), std::clamp(c.z, 0.0, 1.0)};
    }

    struct CubeImage
    {
        int size = 0;
        std::array<std::vector<std::uint8_t>, 6> faces;
        [[nodiscard]] const std::uint8_t* texel(int f, int i, int j) const
        {
            return &faces[static_cast<std::size_t>(f)][(static_cast<std::size_t>(j) * static_cast<std::size_t>(size) + static_cast<std::size_t>(i)) * 4];
        }
    };

    CubeImage make_sky()
    {
        CubeImage img;
        img.size = kSkySize;
        for (int f = 0; f < 6; ++f)
            img.faces[static_cast<std::size_t>(f)] = paint(kSkySize, [f](int i, int j) {
                const double sc = 2.0 * (i + 0.5) / kSkySize - 1.0, tc = 2.0 * (j + 0.5) / kSkySize - 1.0;
                return gamma(sky_radiance(normalize(face_direction(f, sc, tc)), 2.0 / kSkySize));  // texels hold display colours
            });
        return img;
    }

    // =============================================================================================
    // 5. Meshes. Every mesh is NON-indexed: each triangle owns its three vertices so that it can carry the
    //    barycentric attribute (1,0,0), (0,1,0), (0,0,1) of the wireframe overlay.
    // =============================================================================================
    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
        float u, v;
        float tx, ty, tz, tw;
        float bx, by, bz;
    };

    struct VSpec  // a vertex before the barycentric coordinate is assigned
    {
        V3 p, n;
        double u, v;
        V3 t;
        double tw;
    };

    using MeshData = std::vector<Vertex>;

    /// Appends one triangle, front face counter-clockwise (ES 3.0 3.6.1) as seen from the side its normals point to.
    void add_triangle(MeshData& m, VSpec a, VSpec b, VSpec c)
    {
        if (dot(cross(b.p - a.p, c.p - a.p), a.n + b.n + c.n) < 0.0) std::swap(b, c);
        const VSpec* vs[3] = {&a, &b, &c};
        for (int k = 0; k < 3; ++k)
        {
            const VSpec& s = *vs[k];
            m.push_back({static_cast<float>(s.p.x), static_cast<float>(s.p.y), static_cast<float>(s.p.z), static_cast<float>(s.n.x),
                         static_cast<float>(s.n.y), static_cast<float>(s.n.z), static_cast<float>(s.u), static_cast<float>(s.v),
                         static_cast<float>(s.t.x), static_cast<float>(s.t.y), static_cast<float>(s.t.z), static_cast<float>(s.tw),
                         k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f, k == 2 ? 1.0f : 0.0f});
        }
    }

    /// Tangent frame of a box face with outward local normal n: T = dP/du, B = cross(n, T) = dP/dv. Side faces have B = +Y,
    /// so textures stand upright.
    struct FaceFrame
    {
        V3 n, t, b;
    };
    FaceFrame face_frame(V3 n)
    {
        V3 t{1, 0, 0};
        if (n.x > 0.5) t = {0, 0, -1};
        else if (n.x < -0.5) t = {0, 0, 1};
        else if (n.z < -0.5) t = {-1, 0, 0};
        return {n, t, cross(n, t)};
    }

    constexpr std::array<V3, 6> kFaceNormals = {{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};

    /// Box centred at the origin. Each face is two triangles (corner 0-1-2, 0-2-3): the diagonal from (-t,-b) to (+t,+b)
    /// runs through the face centre (the wireframe validation probes that point).
    MeshData make_box(V3 half, double uv_scale, double uv_offset)
    {
        MeshData m;
        for (const V3 n : kFaceNormals)
        {
            const FaceFrame f = face_frame(n);
            const double corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
            VSpec v[4];
            for (int k = 0; k < 4; ++k)
            {
                const V3 p = mul(n, half) + mul(f.t, half) * corners[k][0] + mul(f.b, half) * corners[k][1];
                v[k] = {p, n, dot(p, f.t) * uv_scale + uv_offset, dot(p, f.b) * uv_scale + uv_offset, f.t, 1.0};
            }
            add_triangle(m, v[0], v[1], v[2]);
            add_triangle(m, v[0], v[2], v[3]);
        }
        return m;
    }

    /// Unit sphere: u = phi / 2pi with phi measured from +X towards -Z, v = 1 at the north pole. T = dP/du, handedness +1.
    MeshData make_sphere(int rows, int cols)
    {
        auto vert = [&](int i, int j) {
            const double theta = std::numbers::pi * i / rows, phi = 2.0 * std::numbers::pi * j / cols;
            const V3 p{std::sin(theta) * std::cos(phi), std::cos(theta), -std::sin(theta) * std::sin(phi)};
            return VSpec{p, p, static_cast<double>(j) / cols, 1.0 - static_cast<double>(i) / rows, {-std::sin(phi), 0.0, -std::cos(phi)}, 1.0};
        };
        MeshData m;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                const VSpec a = vert(i, j), b = vert(i + 1, j), c = vert(i + 1, j + 1), d = vert(i, j + 1);
                if (i != 0) add_triangle(m, a, b, d);         // the top row's a and d are both the pole
                if (i != rows - 1) add_triangle(m, b, c, d);  // the bottom row's b and c are both the pole
            }
        return m;
    }

    /// Ground y = 0: kGroundCells^2 cells, each split along the diagonal (x0, z0) - (x1, z1).
    MeshData make_ground()
    {
        MeshData m;
        const double cell = 2.0 * kGroundHalf / kGroundCells;
        auto vert = [](double x, double z) { return VSpec{{x, 0, z}, {0, 1, 0}, x / kGroundTile, z / kGroundTile, {1, 0, 0}, -1.0}; };
        for (int i = 0; i < kGroundCells; ++i)
            for (int j = 0; j < kGroundCells; ++j)
            {
                const double x0 = -kGroundHalf + cell * j, z0 = -kGroundHalf + cell * i, x1 = x0 + cell, z1 = z0 + cell;
                add_triangle(m, vert(x0, z0), vert(x1, z0), vert(x1, z1));
                add_triangle(m, vert(x0, z0), vert(x1, z1), vert(x0, z1));
            }
        return m;
    }

    // =============================================================================================
    // 6. CPU reference: ray casting, the sun's shadow, the lighting equation, texture footprints
    // =============================================================================================
    enum class Kind { Box, Sphere };

    struct Shape
    {
        Kind kind;
        V3 center;
        V3 half;
        double radius;
        double yaw;
        Material mat;
        V3 tint;
        double uv_scale;
        double uv_offset;
        bool present;  // false: the instance was not drawn (instance-count negative control)
    };

    std::vector<Shape> make_shapes(int pillar_count)
    {
        std::vector<Shape> s;
        for (const BoxDef* b : {&kWall, &kPlinth, &kMonolith, &kCrate})
            s.push_back({Kind::Box, b->center, b->half, 0.0, radians(b->yaw_deg), b->mat, {1, 1, 1}, b->uv_scale, b->uv_offset, true});
        s.push_back({Kind::Sphere, kBallCenter, {}, kBallRadius, 0.0, kBallMat, {1, 1, 1}, 0.0, 0.0, true});
        for (int i = 0; i < kPillarCount; ++i)
        {
            const Instance& p = kPillars[static_cast<std::size_t>(i)];
            s.push_back({Kind::Box, p.offset + kPillarCenter * p.scale, kPillarHalf * p.scale, 0.0, 0.0, kPillarMat, p.tint, 0.0, 0.0, i < pillar_count});
        }
        for (const Instance& t : kTrees)  // the trunks do not read the tint (attribute 6 is disabled in their VAO)
            s.push_back({Kind::Box, t.offset + kTrunkCenter * t.scale, kTrunkHalf * t.scale, 0.0, 0.0, kTrunkMat, {1, 1, 1}, 0.0, 0.0, true});
        for (const Instance& t : kTrees)
            s.push_back({Kind::Sphere, t.offset + kCanopyCenter * t.scale, {}, kCanopyRadius * t.scale, 0.0, kCanopyMat, t.tint, 0.0, 0.0, true});
        return s;
    }

    struct Hit
    {
        bool ok = false;
        double t = std::numeric_limits<double>::infinity();
        int id = kIdSky;
        V3 n;   // world normal
        V3 ln;  // local (unrotated) face normal of a box
    };

    /// Ray vs shape (es30-shadow-mapping). True when the ray touches the shape at a parameter >= 0; a ray that starts
    /// inside reports t_near < 0.
    bool intersect_shape(const Shape& s, V3 o, V3 d, double& t_near, V3& n, V3& ln)
    {
        if (s.kind == Kind::Sphere)
        {
            const V3 oc = o - s.center;
            const double b = dot(oc, d), c = dot(oc, oc) - s.radius * s.radius;
            const double disc = b * b - c;
            if (disc < 0.0) return false;
            const double sq = std::sqrt(disc), t0 = -b - sq, t1 = -b + sq;
            if (t1 < 0.0) return false;
            t_near = t0;
            n = normalize(oc + d * t0);
            ln = n;
            return true;
        }
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
            double sg = -1.0;
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
        ln = {axis == 0 ? sign : 0.0, axis == 1 ? sign : 0.0, axis == 2 ? sign : 0.0};
        n = rot_y(ln, s.yaw);
        return true;
    }

    Hit trace(const std::vector<Shape>& shapes, V3 o, V3 d)
    {
        Hit best;
        if (d.y < -1e-12 && o.y > 0.0)
        {
            const double t = -o.y / d.y;
            const V3 p = o + d * t;
            if (std::abs(p.x) <= kGroundHalf && std::abs(p.z) <= kGroundHalf)
            {
                best = {true, t, kIdGround, {0, 1, 0}, {0, 1, 0}};
            }
        }
        for (std::size_t i = 0; i < shapes.size(); ++i)
        {
            if (!shapes[i].present) continue;
            double t = 0;
            V3 n, ln;
            if (intersect_shape(shapes[i], o, d, t, n, ln) && t >= 0.0 && t < best.t) best = {true, t, static_cast<int>(i), n, ln};
        }
        return best;
    }

    /// True when a shape lies between surface point p (normal n) and the sun.
    bool blocked(const std::vector<Shape>& shapes, V3 L, V3 p, V3 n)
    {
        const V3 o = p + n * 1e-4;
        for (const Shape& s : shapes)
        {
            if (!s.present) continue;
            double t = 0;
            V3 a, b;
            if (intersect_shape(s, o, L, t, a, b)) return true;
        }
        return false;
    }

    /// Shadow-edge exclusion (es30-shadow-mapping): the occlusion must be the same on rings of radius r and r/2 around p.
    bool occlusion_stable(const std::vector<Shape>& shapes, V3 L, V3 p, V3 n, bool occluded, double radius)
    {
        const V3 t1 = normalize(cross(n, std::abs(n.x) < 0.9 ? V3{1, 0, 0} : V3{0, 1, 0}));
        const V3 t2 = cross(n, t1);
        for (const double r : {radius, radius * 0.5})
            for (int k = 0; k < 8; ++k)
            {
                const double a = 2.0 * std::numbers::pi * k / 8.0;
                if (blocked(shapes, L, p + t1 * (r * std::cos(a)) + t2 * (r * std::sin(a)), n) != occluded) return false;
            }
        return true;
    }

    // ---- the sun's orthographic frame ----------------------------------------------------------------------------
    struct LightFrame
    {
        V3 L, eye, fwd, right, up;
    };

    LightFrame make_light()
    {
        LightFrame f;
        f.L = sun_dir();
        f.eye = f.L * kLightDistance;
        f.fwd = f.L * -1.0;
        f.right = normalize(cross(f.fwd, V3{0, 1, 0}));
        f.up = cross(f.right, f.fwd);
        return f;
    }

    /// (s, t, window depth) of a world point in the shadow map.
    V3 to_light_uv(const LightFrame& f, V3 p)
    {
        const V3 rel = p - f.eye;
        return {dot(rel, f.right) / kLightHalf * 0.5 + 0.5, dot(rel, f.up) / kLightHalf * 0.5 + 0.5,
                (dot(rel, f.fwd) - kLightNear) / (kLightFar - kLightNear)};
    }

    dm::Mat4 light_matrix(const LightFrame& f)
    {
        const float h = static_cast<float>(kLightHalf);
        return dm::ortho(-h, h, -h, h, static_cast<float>(kLightNear), static_cast<float>(kLightFar)) * dm::look_at(to_vec3(f.eye), {0, 0, 0}, {0, 1, 0});
    }

    /// First surface the sun sees through shadow-map texel (tx, ty), i.e. what the map stores there.
    Hit light_hit(const std::vector<Shape>& shapes, const LightFrame& f, int tx, int ty)
    {
        const double x = ((tx + 0.5) / kShadowSize * 2.0 - 1.0) * kLightHalf, y = ((ty + 0.5) / kShadowSize * 2.0 - 1.0) * kLightHalf;
        return trace(shapes, f.eye + f.right * x + f.up * y, f.fwd);
    }

    // ---- what the reference knows about the scene ----------------------------------------------------------------
    struct Reference
    {
        std::vector<Shape> shapes;
        const std::array<TexImage, kTexCount>* albedo = nullptr;
        const TexImage* normal = nullptr;
        const CubeImage* sky = nullptr;
        LightFrame light;
    };

    /// Texture coordinates of a surface point, exactly as the GPU path produces them.
    std::array<double, 2> surface_uv(const Reference& ref, int id, V3 p, V3 ln)
    {
        if (id == kIdGround) return {p.x / kGroundTile, p.z / kGroundTile};  // scene.frag, u_uv_mode 1
        const Shape& s = ref.shapes[static_cast<std::size_t>(id)];
        if (s.kind == Kind::Box)
        {
            // Perspective-correct interpolation reproduces any affine function of the object position exactly (ES 3.0 3.6.1).
            const V3 lp = rot_y(p - s.center, -s.yaw);
            const FaceFrame f = face_frame(ln);
            return {dot(lp, f.t) * s.uv_scale + s.uv_offset, dot(lp, f.b) * s.uv_scale + s.uv_offset};
        }
        const V3 d = normalize(p - s.center);
        double phi = std::atan2(-d.z, d.x);
        if (phi < 0.0) phi += 2.0 * std::numbers::pi;
        return {phi / (2.0 * std::numbers::pi), 1.0 - std::acos(std::clamp(d.y, -1.0, 1.0)) / std::numbers::pi};
    }

    /// Mip levels a trilinear lookup can touch when the scale factor rho is at most F (texels per pixel).
    /// ES 3.0 3.8.10: max(mu, mv) <= f <= mu + mv, so F = |du/dx| + |du/dy| + |dv/dx| + |dv/dy| bounds every allowed
    /// approximation; lambda <= log2(F), and LINEAR_MIPMAP_LINEAR reads levels floor(lambda) and floor(lambda) + 1.
    int max_level(double F) { return F < 1.0 ? 0 : std::min(kTexLevels - 1, static_cast<int>(std::floor(std::log2(F))) + 1); }

    /// Per-channel range of every texel that the bilinear footprint at (u, v) can reach on levels 0..max_level(F). The
    /// filtered value is a convex combination of such texels, so it lies in this range whatever the weights.
    void texel_range(const TexImage& t, double u, double v, double F, V3& lo, V3& hi)
    {
        double l[3] = {1e9, 1e9, 1e9}, h[3] = {-1e9, -1e9, -1e9};
        for (int k = 0; k <= max_level(F); ++k)
        {
            const int s = t.level_size(k);
            const int i0 = static_cast<int>(std::floor(u * s - 0.5)), j0 = static_cast<int>(std::floor(v * s - 0.5));
            for (int j = j0 - 1; j <= j0 + 2; ++j)
                for (int i = i0 - 1; i <= i0 + 2; ++i)
                {
                    const std::uint8_t* p = t.texel(k, i, j);
                    for (int c = 0; c < 3; ++c)
                    {
                        l[c] = std::min(l[c], p[c] / 255.0);
                        h[c] = std::max(h[c], p[c] / 255.0);
                    }
                }
        }
        lo = {l[0], l[1], l[2]};
        hi = {h[0], h[1], h[2]};
    }

    // ---- probes: pixels whose colour the CPU can predict ---------------------------------------------------------
    struct Params  // the technique toggles that change shading (wireframe and inset are validated separately)
    {
        bool shadows = true, normal_mapping = true, fog = true, blinn = true;
    };

    struct Probe
    {
        int x = 0, y = 0;
        int id = kIdSky;
        V3 p, n_geo, n_map;   // hit point, geometric normal, normal-mapped normal (== n_geo without a normal map)
        bool has_map = false;
        Material mat{};
        V3 alb_lo, alb_hi;    // linear albedo range: texture footprint * material colour * instance tint
        double ndl = 0;       // geometric n . L
        bool occluded = false;
        V3 sky_lo, sky_hi;    // display range of a sky probe
    };

    constexpr double kMinNdl = 0.2;   // lit faces below this are too oblique to separate lit from shadowed reliably
    constexpr double kRing = 0.3;     // world units: shadow-edge exclusion radius (>= 12 shadow-map texels)
    constexpr double kMaxWidth = 0.05;// probes whose predicted display range is wider (texture edges) are not used
    constexpr int kTol = 3;           // 8-bit levels outside the predicted range, see README "Tolerances"

    /// The lighting equation of scene.frag, for one albedo. Returns the display colour.
    V3 shade(const Probe& pr, V3 albedo, V3 eye, const Params& prm)
    {
        const V3 L = sun_dir();
        const V3 n = (pr.has_map && prm.normal_mapping) ? pr.n_map : pr.n_geo;
        const V3 V = normalize(eye - pr.p);
        const double sf = prm.shadows && pr.occluded ? 0.0 : 1.0;
        const double lit = pr.ndl > 0.0 ? sf : 0.0;
        const double diffuse = std::max(dot(n, L), 0.0);
        const double s = prm.blinn ? std::max(dot(n, normalize(L + V)), 0.0) : std::max(dot(n * (2.0 * dot(n, L)) - L, V), 0.0);
        const double spec = diffuse > 0.0 ? std::pow(s, pr.mat.shininess) : 0.0;
        const V3 ambient = lerp(kGroundAmbient, kSkyAmbient, 0.5 * n.y + 0.5);
        V3 c = mul(albedo, ambient + kSunColor * (diffuse * lit)) + kSunColor * (pr.mat.specular * spec * lit);
        if (prm.fog) c = lerp(kHorizon, c, std::exp(-kFogDensity * length(pr.p - eye)));
        return gamma(c);
    }

    struct Range
    {
        V3 lo, hi;  // display colour, 0..1
    };

    /// Predicted display range of a probe. The shading is monotonic in the albedo (non-negative coefficients, then a
    /// monotonic fog mix and gamma), so the albedo range maps to the colour range.
    Range expect(const Probe& pr, V3 eye, const Params& prm)
    {
        if (pr.id == kIdSky) return {pr.sky_lo, pr.sky_hi};
        const V3 a = shade(pr, pr.alb_lo, eye, prm), b = shade(pr, pr.alb_hi, eye, prm);
        return {{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}, {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}};
    }

    double width(const Range& r) { return std::max({r.hi.x - r.lo.x, r.hi.y - r.lo.y, r.hi.z - r.lo.z}); }

    /// 8-bit levels by which a pixel lies outside a predicted range (0 inside).
    double outside(const std::uint8_t* px, const Range& r)
    {
        double d = 0;
        for (int c = 0; c < 3; ++c)
        {
            const double v = px[c];
            d = std::max({d, comp(r.lo, c) * 255.0 - v, v - comp(r.hi, c) * 255.0});
        }
        return d;
    }

    /// Builds the probe for pixel (x, y), or nothing when the pixel is not stable enough to predict.
    std::optional<Probe> make_probe(const Reference& ref, const View& v, int x, int y)
    {
        const V3 dir = ray_dir(v, x + 0.5, y + 0.5);
        const Hit h = trace(ref.shapes, v.eye, dir);
        // The pixel must lie well inside one surface: the hits at +-2 pixels see the same primitive and box face.
        for (int dy = -2; dy <= 2; dy += 2)
            for (int dx = -2; dx <= 2; dx += 2)
            {
                if (dx == 0 && dy == 0) continue;
                const Hit q = trace(ref.shapes, v.eye, ray_dir(v, x + dx + 0.5, y + dy + 0.5));
                if (q.id != h.id || (h.ok && dot(q.ln, h.ln) < 0.999 && h.id >= 0 && ref.shapes[static_cast<std::size_t>(h.id)].kind == Kind::Box))
                    return std::nullopt;
            }

        Probe pr;
        pr.x = x;
        pr.y = y;
        pr.id = h.id;
        if (!h.ok)
        {
            // Sky: LINEAR cube-map lookup = convex combination of the 2x2 texels around (s, t) on one face (seamless
            // filtering only mixes faces within one texel of an edge, which is excluded here).
            const FaceSt f = select_face(dir);
            const double fs = f.s * kSkySize - 0.5, ft = f.t * kSkySize - 0.5;
            if (fs < 2.0 || ft < 2.0 || fs > kSkySize - 3.0 || ft > kSkySize - 3.0) return std::nullopt;
            double l[3] = {1, 1, 1}, hh[3] = {0, 0, 0};
            for (int j = static_cast<int>(std::floor(ft)) - 1; j <= static_cast<int>(std::floor(ft)) + 2; ++j)
                for (int i = static_cast<int>(std::floor(fs)) - 1; i <= static_cast<int>(std::floor(fs)) + 2; ++i)
                {
                    const std::uint8_t* p = ref.sky->texel(f.face, i, j);
                    for (int c = 0; c < 3; ++c)
                    {
                        l[c] = std::min(l[c], p[c] / 255.0);
                        hh[c] = std::max(hh[c], p[c] / 255.0);
                    }
                }
            pr.sky_lo = {l[0], l[1], l[2]};
            pr.sky_hi = {hh[0], hh[1], hh[2]};
            if (width({pr.sky_lo, pr.sky_hi}) > kMaxWidth) return std::nullopt;
            return pr;
        }

        pr.p = v.eye + dir * h.t;
        pr.n_geo = h.n;
        pr.n_map = h.n;
        pr.mat = h.id == kIdGround ? kGroundMat : ref.shapes[static_cast<std::size_t>(h.id)].mat;
        const V3 tint = h.id == kIdGround ? V3{1, 1, 1} : ref.shapes[static_cast<std::size_t>(h.id)].tint;
        pr.alb_lo = pr.alb_hi = mul(pr.mat.albedo, tint);

        if (pr.mat.tex != Tex::None)
        {
            // Footprint of one pixel in texels, from the texture coordinates of the neighbouring pixels (forward and
            // backward differences, the larger one: implementations may difference over a 2x2 quad).
            const std::array<double, 2> uv = surface_uv(ref, h.id, pr.p, h.ln);
            double d[2][2] = {};  // [u|v][x|y]
            for (int axis = 0; axis < 2; ++axis)
                for (const int sgn : {-1, 1})
                {
                    const int qx = x + (axis == 0 ? sgn : 0), qy = y + (axis == 1 ? sgn : 0);
                    const V3 qd = ray_dir(v, qx + 0.5, qy + 0.5);
                    const Hit q = trace(ref.shapes, v.eye, qd);
                    if (q.id != h.id) return std::nullopt;
                    const std::array<double, 2> quv = surface_uv(ref, h.id, v.eye + qd * q.t, q.ln);
                    for (int c = 0; c < 2; ++c)
                    {
                        double diff = std::abs(quv[static_cast<std::size_t>(c)] - uv[static_cast<std::size_t>(c)]);
                        if (diff > 0.5) diff = 1.0 - diff;  // across the sphere's u seam
                        d[c][axis] = std::max(d[c][axis], diff);
                    }
                }
            const double F = 1.1 * kTexSize * (d[0][0] + d[0][1] + d[1][0] + d[1][1]);
            V3 lo, hi;
            texel_range((*ref.albedo)[static_cast<std::size_t>(pr.mat.tex)], uv[0], uv[1], F, lo, hi);
            pr.alb_lo = mul(pr.alb_lo, lo);
            pr.alb_hi = mul(pr.alb_hi, hi);
            if (pr.mat.normal_map)
            {
                // The normal-map footprint must hold ONE texel value: then the shading normal is known exactly.
                V3 nlo, nhi;
                texel_range(*ref.normal, uv[0], uv[1], F, nlo, nhi);
                if (width({nlo, nhi}) > 0.0) return std::nullopt;
                const V3 n_ts = nlo * 2.0 - V3{1, 1, 1};
                const Shape& s = ref.shapes[static_cast<std::size_t>(h.id)];
                const FaceFrame fr = face_frame(h.ln);
                pr.n_map = normalize(rot_y(fr.t, s.yaw) * n_ts.x + rot_y(fr.b, s.yaw) * n_ts.y + h.n * n_ts.z);
                pr.has_map = true;
            }
        }

        pr.ndl = dot(h.n, ref.light.L);
        if (pr.ndl > -0.02 && pr.ndl < kMinNdl) return std::nullopt;  // grazing sun: lit vs shadowed is ambiguous
        if (h.id >= 0 && ref.shapes[static_cast<std::size_t>(h.id)].kind == Kind::Box)
        {
            // Near a box edge the PCF taps reach across the crease onto the neighbouring face, whose polygon offset was
            // computed from ITS slope: keep 0.12 world units (about 5 shadow-map texels) from every face border.
            const Shape& s = ref.shapes[static_cast<std::size_t>(h.id)];
            const V3 lp = rot_y(pr.p - s.center, -s.yaw);
            const FaceFrame fr = face_frame(h.ln);
            const V3 at{std::abs(fr.t.x), std::abs(fr.t.y), std::abs(fr.t.z)}, ab{std::abs(fr.b.x), std::abs(fr.b.y), std::abs(fr.b.z)};
            if (std::abs(dot(lp, fr.t)) > dot(at, s.half) - 0.12 || std::abs(dot(lp, fr.b)) > dot(ab, s.half) - 0.12) return std::nullopt;
        }
        if (pr.ndl >= kMinNdl)
        {
            const V3 uvl = to_light_uv(ref.light, pr.p);
            const bool inside = uvl.x > 0.0 && uvl.x < 1.0 && uvl.y > 0.0 && uvl.y < 1.0 && uvl.z < 1.0;
            const bool border = std::min({std::abs(uvl.x), std::abs(1.0 - uvl.x), std::abs(uvl.y), std::abs(1.0 - uvl.y), std::abs(1.0 - uvl.z)}) < 0.01;
            if (border) return std::nullopt;
            if (inside)
            {
                pr.occluded = blocked(ref.shapes, ref.light.L, pr.p, h.n);
                if (!occlusion_stable(ref.shapes, ref.light.L, pr.p, h.n, pr.occluded, kRing)) return std::nullopt;
            }
        }
        if (width(expect(pr, v.eye, Params{})) > kMaxWidth) return std::nullopt;
        return pr;
    }

    std::vector<Probe> build_probes(const Reference& ref, const View& v)
    {
        // About 4500 sample pixels at any window size.
        const int step = std::max(2, static_cast<int>(std::lround(std::sqrt(static_cast<double>(v.w) * v.h / 4500.0))));
        std::vector<Probe> out;
        for (int y = 3; y < v.h - 3; y += step)
            for (int x = 3; x < v.w - 3; x += step)
                if (std::optional<Probe> p = make_probe(ref, v, x, y)) out.push_back(*p);
        return out;
    }

    // ---- read-back images and comparisons -------------------------------------------------------------------------
    struct Image
    {
        int w = 0, h = 0;
        std::vector<std::uint8_t> px;
        [[nodiscard]] const std::uint8_t* at(int x, int y) const
        {
            return &px[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4];
        }
    };

    struct Tally
    {
        int checked = 0, bad = 0;
        double worst = 0;
        std::string first;
    };

    using ProbeFilter = std::function<bool(const Probe&)>;

    Tally evaluate(const std::vector<Probe>& probes, const Image& img, V3 eye, const Params& prm, const ProbeFilter& filter = {})
    {
        Tally t;
        for (const Probe& pr : probes)
        {
            if (filter && !filter(pr)) continue;
            const Range r = expect(pr, eye, prm);
            const std::uint8_t* px = img.at(pr.x, pr.y);
            const double d = outside(px, r);
            ++t.checked;
            t.worst = std::max(t.worst, d);
            if (d > kTol)
            {
                ++t.bad;
                if (t.first.empty())
                    t.first = std::format("first at ({},{}) id {} world ({:.2f},{:.2f},{:.2f}) n.L {:.2f} {}: got ({},{},{}) expected "
                                          "({:.0f}..{:.0f}, {:.0f}..{:.0f}, {:.0f}..{:.0f})",
                                          pr.x, pr.y, pr.id, pr.p.x, pr.p.y, pr.p.z, pr.ndl, pr.occluded ? "shadowed" : "lit", px[0], px[1], px[2],
                                          r.lo.x * 255, r.hi.x * 255, r.lo.y * 255, r.hi.y * 255, r.lo.z * 255, r.hi.z * 255);
            }
        }
        return t;
    }

    /// True when the predicted ranges under `a` and `b` are separated by more than 2 * kTol + 2 levels in some channel,
    /// i.e. no pixel can satisfy both predictions.
    bool separated(const Range& ra, const Range& rb)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double gap = std::max(comp(rb.lo, c) - comp(ra.hi, c), comp(ra.lo, c) - comp(rb.hi, c)) * 255.0;
            if (gap > 2 * kTol + 2) return true;
        }
        return false;
    }

    bool predicted_change(const Probe& pr, V3 eye, const Params& a, const Params& b) { return separated(expect(pr, eye, a), expect(pr, eye, b)); }

    bool is_instance(int id) { return id >= kIdPillar0 && id < kIdCanopy0 + kTreeCount; }

    // =============================================================================================
    // 7. The demo
    // =============================================================================================
    struct Toggles
    {
        bool shadows = true, normal_mapping = true, fog = true, wire = false, inset = false, blinn = true;
        [[nodiscard]] Params params() const { return {shadows, normal_mapping, fog, blinn}; }
    };

    struct InstanceData  // one element of an instance buffer (attributes 5 and 6)
    {
        float ox, oy, oz, scale;
        float r, g, b;
    };

    struct Mesh
    {
        mgl::VertexArrayId vao{};
        GLsizei count = 0;
    };

    struct GpuObject
    {
        const Mesh* mesh;
        dm::Mat4 model;
        Material mat;
        int instances;  // 0: glDrawArrays, otherwise glDrawArraysInstanced
        bool ground;
        bool pillars;   // instance count can be overridden (negative control)
    };

    struct SceneUniforms
    {
        mgl::UniformLocation model, normal_matrix, view_proj, light_vp, albedo_map, normal_map, shadow_map, albedo, textured, uv_mode, ground_tile,
            has_normal_map, specular, shininess, shadows, normal_mapping, blinn, wire, fog_density, eye, sun_dir, sun_color, sky_ambient,
            ground_ambient, fog_color, wire_color, shadow_texel, shadow_bias;
    };

    const mgl::AttribLocation kLocPosition{0u}, kLocNormal{1u}, kLocUv{2u}, kLocTangent{3u}, kLocBary{4u}, kLocInstance{5u}, kLocTint{6u};

    struct InsetRect
    {
        int x, y, size, border;
    };
    InsetRect inset_rect(int w, int h)
    {
        const int m = std::min(w, h);
        const int size = std::max(32, m * 30 / 100), margin = std::max(4, m / 50);
        return {margin, h - margin - size, size, 2};
    }

    void uniform3(mgl::UniformLocation loc, V3 v) { mgl::glUniform3f(loc, static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)); }

    class InteractiveScene final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override;
        void render(demo::Env& env, int frame) override;
        void validate(demo::Env& env, demo::Check& check) override;
        void shutdown(demo::Env& env) override;

        void on_key(demo::Env& env, char key) override
        {
            const char* what = nullptr;
            bool on = false;
            switch (key)
            {
            case '1': on = toggles_.shadows = !toggles_.shadows; what = "shadows"; break;
            case '2': on = toggles_.normal_mapping = !toggles_.normal_mapping; what = "normal mapping"; break;
            case '3': on = toggles_.fog = !toggles_.fog; what = "fog"; break;
            case '4': on = toggles_.wire = !toggles_.wire; what = "wireframe overlay"; break;
            case '5': on = toggles_.inset = !toggles_.inset; what = "shadow-map inset"; break;
            case '6': on = toggles_.blinn = !toggles_.blinn; what = "Blinn-Phong (off = Phong)"; break;
            case 'r': pose_ = kStartPose; what = "camera reset"; on = true; break;
            default: break;
            }
            if (what != nullptr && env.interactive) std::printf("%s: %s\n", what, on ? "on" : "off");
        }

        /// Mouse-look while the left button is held; the wheel changes the field of view.
        void on_mouse(demo::Env& /*env*/, const Mouse& m) override
        {
            if (m.left)
            {
                pose_.yaw += static_cast<double>(m.dx) * kMouseSensitivity;
                pose_.pitch = std::clamp(pose_.pitch - static_cast<double>(m.dy) * kMouseSensitivity, -kMaxPitch, kMaxPitch);
            }
            if (m.wheel != 0.0f) pose_.fov = std::clamp(pose_.fov - static_cast<double>(m.wheel) * kWheelStep, kMinFov, kMaxFov);
        }

    private:
        /// Held keys move the camera: W/S along the horizontal view direction, A/D sideways, E/Q up/down. The speed is
        /// the same in every direction (the summed direction is normalized); env.frame_seconds makes it frame-rate
        /// independent and, being a fixed 1/60 when headless, deterministic.
        void update_camera(const demo::Env& env)
        {
            const V3 fwd{std::sin(pose_.yaw), 0.0, -std::cos(pose_.yaw)}, right{std::cos(pose_.yaw), 0.0, std::sin(pose_.yaw)};
            V3 dir{};
            if (env.key_held('w')) dir = dir + fwd;
            if (env.key_held('s')) dir = dir - fwd;
            if (env.key_held('d')) dir = dir + right;
            if (env.key_held('a')) dir = dir - right;
            if (env.key_held('e')) dir = dir + V3{0, 1, 0};
            if (env.key_held('q')) dir = dir - V3{0, 1, 0};
            const double len = length(dir);
            if (len > 1e-9) pose_.pos = pose_.pos + dir * (kMoveSpeed * env.frame_seconds / len);
            pose_.pos.y = std::max(pose_.pos.y, kMinEyeHeight);
        }

        void draw_frame(int w, int h);
        void shadow_pass();
        void main_pass(const View& v);
        void sky_pass(const View& v);
        void inset_pass(int w, int h);
        void draw_object(const GpuObject& o, bool main);

        Image render_image(const demo::Env& env)
        {
            draw_frame(env.width, env.height);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return {env.width, env.height, demo::read_rgba8(0, 0, env.width, env.height)};
        }

        Mesh create_mesh(const MeshData& data, mgl::BufferId instances, bool with_tint);

        // validation parts (section 8)
        void check_state(demo::Check& check);
        void check_bulk(demo::Check& check, const char* name, const Image& img, const std::vector<Probe>& probes, const View& v, const Params& prm,
                        const ProbeFilter& filter = {});
        void check_toggle(demo::Env& env, demo::Check& check, const Reference& ref, const char* name, char key, const Params& toggled,
                          const std::vector<Probe>& probes, const View& v, int min_changed);
        void check_instances(demo::Env& env, demo::Check& check, const Reference& ref);
        void check_fog(demo::Env& env, demo::Check& check, const Reference& ref);
        void check_wire(demo::Env& env, demo::Check& check, const Reference& ref);
        void check_inset(demo::Env& env, demo::Check& check, const Reference& ref);
        void check_input(demo::Env& env, demo::Check& check);

        // CPU data (also the reference)
        std::array<TexImage, kTexCount> albedo_images_;
        TexImage normal_image_;
        CubeImage sky_image_;
        LightFrame light_;

        // GL objects
        mgl::ProgramId scene_program_{}, shadow_program_{}, sky_program_{}, inset_program_{};
        SceneUniforms su_{};
        mgl::UniformLocation shadow_model_{}, shadow_light_vp_{}, sky_view_{}, sky_proj_{}, sky_sampler_{}, inset_depth_{}, inset_range_{};
        std::vector<mgl::BufferId> buffers_;
        std::vector<Mesh> meshes_;  // owns the VAOs
        Mesh ground_{}, wall_{}, plinth_{}, monolith_{}, crate_{}, ball_{}, pillars_{}, trunks_{}, canopies_{}, sky_cube_{};
        mgl::VertexArrayId empty_vao_{};
        std::vector<GpuObject> objects_;
        std::array<mgl::TextureId, kTexCount> albedo_tex_{};
        mgl::TextureId normal_tex_{}, sky_tex_{}, shadow_tex_{};
        mgl::FramebufferId shadow_fbo_{};
        mgl::SamplerId raw_sampler_{};

        // interactive state
        Pose pose_ = kStartPose;
        Toggles toggles_{};
        int pillar_count_ = kPillarCount;  // validation's instance-count negative control lowers it
    };

    // ---------------------------------------------------------------------------------------------------
    Mesh InteractiveScene::create_mesh(const MeshData& data, mgl::BufferId instances, bool with_tint)
    {
        Mesh m;
        m.count = static_cast<GLsizei>(data.size());
        mgl::glGenVertexArrays(1, &m.vao);
        mgl::glBindVertexArray(m.vao);
        mgl::BufferId vbo{};
        mgl::glGenBuffers(1, &vbo);
        buffers_.push_back(vbo);
        mgl::glBindBuffer(mgl::BufferTarget::Array, vbo);
        mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(data), mgl::BufferUsage::StaticDraw);
        const GLsizei stride = sizeof(Vertex);
        mgl::glEnableVertexAttribArray(kLocPosition);
        mgl::glVertexAttribPointer(kLocPosition, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(offsetof(Vertex, px)));
        mgl::glEnableVertexAttribArray(kLocNormal);
        mgl::glVertexAttribPointer(kLocNormal, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(offsetof(Vertex, nx)));
        mgl::glEnableVertexAttribArray(kLocUv);
        mgl::glVertexAttribPointer(kLocUv, 2, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(offsetof(Vertex, u)));
        mgl::glEnableVertexAttribArray(kLocTangent);
        mgl::glVertexAttribPointer(kLocTangent, 4, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(offsetof(Vertex, tx)));
        mgl::glEnableVertexAttribArray(kLocBary);
        mgl::glVertexAttribPointer(kLocBary, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(offsetof(Vertex, bx)));
        if (instances.value != 0)
        {
            // Per-instance attributes: the divisor (VAO state, ES 3.0 table 6.3) makes the GL advance them once per instance.
            mgl::glBindBuffer(mgl::BufferTarget::Array, instances);
            mgl::glEnableVertexAttribArray(kLocInstance);
            mgl::glVertexAttribPointer(kLocInstance, 4, mgl::DataType::Float, GL_FALSE, sizeof(InstanceData), demo::buffer_offset(offsetof(InstanceData, ox)));
            mgl::glVertexAttribDivisor(kLocInstance, 1);
            if (with_tint)
            {
                mgl::glEnableVertexAttribArray(kLocTint);
                mgl::glVertexAttribPointer(kLocTint, 3, mgl::DataType::Float, GL_FALSE, sizeof(InstanceData), demo::buffer_offset(offsetof(InstanceData, r)));
                mgl::glVertexAttribDivisor(kLocTint, 1);
            }
        }
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        meshes_.push_back(m);
        return m;
    }

    void InteractiveScene::init(demo::Env& env)
    {
        if (std::min(env.width, env.height) < 96) throw demo::Skip("the scene needs a surface of at least 96x96 pixels");
        light_ = make_light();

        // ---- programs ---------------------------------------------------------------------------------------
        scene_program_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
        auto loc = [this](const char* n) { return mgl::glGetUniformLocation(scene_program_, n); };
        su_ = {loc("u_model"), loc("u_normal_matrix"), loc("u_view_proj"), loc("u_light_vp"), loc("u_albedo_map"), loc("u_normal_map"),
               loc("u_shadow_map"), loc("u_albedo"), loc("u_textured"), loc("u_uv_mode"), loc("u_ground_tile"), loc("u_has_normal_map"),
               loc("u_specular"), loc("u_shininess"), loc("u_shadows"), loc("u_normal_mapping"), loc("u_blinn"), loc("u_wire"),
               loc("u_fog_density"), loc("u_eye"), loc("u_sun_dir"), loc("u_sun_color"), loc("u_sky_ambient"), loc("u_ground_ambient"),
               loc("u_fog_color"), loc("u_wire_color"), loc("u_shadow_texel"), loc("u_shadow_bias")};
        shadow_program_ = demo::build_program(env, "shaders/shadow.vert", "shaders/shadow.frag");
        shadow_model_ = mgl::glGetUniformLocation(shadow_program_, "u_model");
        shadow_light_vp_ = mgl::glGetUniformLocation(shadow_program_, "u_light_vp");
        sky_program_ = demo::build_program(env, "shaders/sky.vert", "shaders/sky.frag");
        sky_view_ = mgl::glGetUniformLocation(sky_program_, "u_view");
        sky_proj_ = mgl::glGetUniformLocation(sky_program_, "u_proj");
        sky_sampler_ = mgl::glGetUniformLocation(sky_program_, "u_sky");
        inset_program_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/inset.frag");
        inset_depth_ = mgl::glGetUniformLocation(inset_program_, "u_depth");
        inset_range_ = mgl::glGetUniformLocation(inset_program_, "u_range");

        // Texture units are fixed for the whole run: 0 albedo, 1 normal map, 2 shadow map (comparing),
        // 3 shadow map (raw, through a sampler object), 4 sky cube map.
        mgl::glUseProgram(scene_program_);
        mgl::glUniform1i(su_.albedo_map, 0);
        mgl::glUniform1i(su_.normal_map, 1);
        mgl::glUniform1i(su_.shadow_map, 2);
        mgl::glUniform1f(su_.ground_tile, static_cast<float>(kGroundTile));
        mgl::glUniform1f(su_.shadow_texel, 1.0f / static_cast<float>(kShadowSize));
        mgl::glUniform1f(su_.shadow_bias, kReceiverBias);
        uniform3(su_.sun_dir, light_.L);
        uniform3(su_.sun_color, kSunColor);
        uniform3(su_.sky_ambient, kSkyAmbient);
        uniform3(su_.ground_ambient, kGroundAmbient);
        uniform3(su_.fog_color, kHorizon);
        uniform3(su_.wire_color, kWireColor);
        const dm::Mat4 lvp = light_matrix(light_);
        mgl::glUniformMatrix4fv(su_.light_vp, 1, GL_FALSE, lvp.data());
        mgl::glUseProgram(shadow_program_);
        mgl::glUniformMatrix4fv(shadow_light_vp_, 1, GL_FALSE, lvp.data());
        mgl::glUseProgram(sky_program_);
        mgl::glUniform1i(sky_sampler_, 4);
        mgl::glUseProgram(inset_program_);
        mgl::glUniform1i(inset_depth_, 3);
        mgl::glUniform2f(inset_range_, static_cast<float>(kInsetLo), static_cast<float>(kInsetHi));

        // ---- procedural textures with CPU-built mip chains -------------------------------------------------------
        albedo_images_[static_cast<std::size_t>(Tex::Grass)] = build_mips(paint(kTexSize, grass_texel), false);
        albedo_images_[static_cast<std::size_t>(Tex::Wood)] = build_mips(paint(kTexSize, wood_texel), false);
        albedo_images_[static_cast<std::size_t>(Tex::Stone)] = build_mips(paint(kTexSize, stone_texel), false);
        albedo_images_[static_cast<std::size_t>(Tex::Ball)] = build_mips(paint(kTexSize, ball_texel), false);
        normal_image_ = build_mips(paint(kTexSize, stone_normal), true);
        auto upload = [](const TexImage& img) {
            mgl::TextureId tex{};
            mgl::glGenTextures(1, &tex);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, kTexLevels, mgl::SizedInternalFormat::Rgba8, kTexSize, kTexSize);
            for (int k = 0; k < kTexLevels; ++k)
                mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, k, 0, 0, img.level_size(k), img.level_size(k), mgl::PixelFormat::Rgba,
                                     mgl::PixelType::UnsignedByte, img.levels[static_cast<std::size_t>(k)].data());
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::LinearMipmapLinear);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Linear);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::Repeat);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::Repeat);
            return tex;
        };
        for (int t = 0; t < kTexCount; ++t) albedo_tex_[static_cast<std::size_t>(t)] = upload(albedo_images_[static_cast<std::size_t>(t)]);
        normal_tex_ = upload(normal_image_);

        sky_image_ = make_sky();
        mgl::glGenTextures(1, &sky_tex_);
        mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, sky_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, kSkySize, kSkySize);
        for (std::size_t f = 0; f < 6; ++f)
            mgl::glTexSubImage2D(kFaceTargets[f], 0, 0, 0, kSkySize, kSkySize, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, sky_image_.faces[f].data());
        mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMinFilter::Linear);
        mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureMagFilter::Linear);
        mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
        mgl::glTexParameter(mgl::TextureTarget::TextureCubeMap, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);

        // ---- the shadow map: DEPTH_COMPONENT24, the only attachment of its framebuffer --------------------------------
        mgl::glGenTextures(1, &shadow_tex_);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
        mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::DepthComponent24, kShadowSize, kShadowSize);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Linear);  // LINEAR comparison = hardware PCF
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Linear);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureCompareMode::CompareRefToTexture);
        mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::CompareFunc::Lequal);
        mgl::glGenFramebuffers(1, &shadow_fbo_);
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, shadow_fbo_);
        mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, mgl::TextureTarget::Texture2D, shadow_tex_, 0);
        const mgl::DrawBuffer none = mgl::DrawBuffer::None;
        mgl::glDrawBuffers(1, &none);
        mgl::glReadBuffer(mgl::ReadBuffer::None);
        if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
            throw demo::Failure("the depth-only shadow framebuffer is incomplete");
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

        // The inset reads the same depth texture as plain depth: a sampler object with comparison NONE (and NEAREST,
        // which a non-comparing depth texture needs to be complete, ES 3.0 3.8.13) supersedes the texture state on unit 3.
        mgl::glGenSamplers(1, &raw_sampler_);
        mgl::glSamplerParameter(raw_sampler_, mgl::TextureMinFilter::Nearest);
        mgl::glSamplerParameter(raw_sampler_, mgl::TextureMagFilter::Nearest);
        mgl::glSamplerParameter(raw_sampler_, mgl::SamplerWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
        mgl::glSamplerParameter(raw_sampler_, mgl::SamplerWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
        mgl::glSamplerParameter(raw_sampler_, mgl::TextureCompareMode::None);

        mgl::glActiveTexture(mgl::TextureUnit::Texture1);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, normal_tex_);
        mgl::glActiveTexture(mgl::TextureUnit::Texture2);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
        mgl::glActiveTexture(mgl::TextureUnit::Texture3);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, shadow_tex_);
        mgl::glBindSampler(3, raw_sampler_);
        mgl::glActiveTexture(mgl::TextureUnit::Texture4);
        mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, sky_tex_);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, albedo_tex_[0]);

        // ---- instance buffers ------------------------------------------------------------------------------------
        auto instance_buffer = [this](std::span<const Instance> list) {
            std::vector<InstanceData> data;
            for (const Instance& i : list)
                data.push_back({static_cast<float>(i.offset.x), static_cast<float>(i.offset.y), static_cast<float>(i.offset.z), static_cast<float>(i.scale),
                                static_cast<float>(i.tint.x), static_cast<float>(i.tint.y), static_cast<float>(i.tint.z)});
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            buffers_.push_back(b);
            mgl::glBindBuffer(mgl::BufferTarget::Array, b);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const InstanceData>(data), mgl::BufferUsage::StaticDraw);
            return b;
        };
        const mgl::BufferId pillar_buf = instance_buffer(kPillars), tree_buf = instance_buffer(kTrees);

        // ---- meshes and their VAOs -----------------------------------------------------------------------------
        const mgl::BufferId none_buf{0};
        ground_ = create_mesh(make_ground(), none_buf, false);
        wall_ = create_mesh(make_box(kWall.half, kWall.uv_scale, kWall.uv_offset), none_buf, false);
        plinth_ = create_mesh(make_box(kPlinth.half, kPlinth.uv_scale, kPlinth.uv_offset), none_buf, false);
        monolith_ = create_mesh(make_box(kMonolith.half, kMonolith.uv_scale, kMonolith.uv_offset), none_buf, false);
        crate_ = create_mesh(make_box(kCrate.half, kCrate.uv_scale, kCrate.uv_offset), none_buf, false);
        const MeshData sphere = make_sphere(40, 80);
        ball_ = create_mesh(sphere, none_buf, false);
        pillars_ = create_mesh(make_box(kPillarHalf, 0.25, 0.0), pillar_buf, true);
        trunks_ = create_mesh(make_box(kTrunkHalf, 0.25, 0.0), tree_buf, false);  // attribute 6 stays disabled: the current value (white)
        canopies_ = create_mesh(sphere, tree_buf, true);
        sky_cube_ = create_mesh(make_box({1, 1, 1}, 1.0, 0.0), none_buf, false);
        mgl::glGenVertexArrays(1, &empty_vao_);

        // Current generic attribute values (ES 3.0 2.8): what a vertex shader reads from a DISABLED array. Drawing with
        // the instanced arrays enabled does not change them (2.9.3).
        mgl::glVertexAttrib4f(kLocInstance, 0.0f, 0.0f, 0.0f, 1.0f);
        mgl::glVertexAttrib3f(kLocTint, 1.0f, 1.0f, 1.0f);

        auto placed = [](const BoxDef& b) {
            return dm::translate(to_vec3(b.center)) * dm::rotate(static_cast<float>(radians(b.yaw_deg)), {0, 1, 0});
        };
        const float r = static_cast<float>(kBallRadius);
        objects_ = {
            {&ground_, dm::Mat4::identity(), kGroundMat, 0, true, false},
            {&wall_, placed(kWall), kWall.mat, 0, false, false},
            {&plinth_, placed(kPlinth), kPlinth.mat, 0, false, false},
            {&monolith_, placed(kMonolith), kMonolith.mat, 0, false, false},
            {&crate_, placed(kCrate), kCrate.mat, 0, false, false},
            {&ball_, dm::translate(to_vec3(kBallCenter)) * dm::scale({r, r, r}), kBallMat, 0, false, false},
            {&pillars_, dm::translate(to_vec3(kPillarCenter)), kPillarMat, kPillarCount, false, true},
            {&trunks_, dm::translate(to_vec3(kTrunkCenter)), kTrunkMat, kTreeCount, false, false},
            {&canopies_, dm::translate(to_vec3(kCanopyCenter)) * dm::scale({static_cast<float>(kCanopyRadius), static_cast<float>(kCanopyRadius),
                                                                             static_cast<float>(kCanopyRadius)}),
             kCanopyMat, kTreeCount, false, false},
        };
    }

    void InteractiveScene::render(demo::Env& env, int /*frame*/)
    {
        update_camera(env);
        draw_frame(env.width, env.height);
    }

    void InteractiveScene::draw_object(const GpuObject& o, bool main)
    {
        if (main)
        {
            mgl::glUniformMatrix4fv(su_.model, 1, GL_FALSE, o.model.data());
            const std::array<float, 9> nm = dm::normal_matrix(o.model);
            mgl::glUniformMatrix3fv(su_.normal_matrix, 1, GL_FALSE, nm.data());
            uniform3(su_.albedo, o.mat.albedo);
            mgl::glUniform1i(su_.textured, o.mat.tex != Tex::None ? 1 : 0);
            if (o.mat.tex != Tex::None) mgl::glBindTexture(mgl::TextureTarget::Texture2D, albedo_tex_[static_cast<std::size_t>(o.mat.tex)]);  // unit 0
            mgl::glUniform1i(su_.uv_mode, o.ground ? 1 : 0);
            mgl::glUniform1i(su_.has_normal_map, o.mat.normal_map ? 1 : 0);
            mgl::glUniform1f(su_.specular, static_cast<float>(o.mat.specular));
            mgl::glUniform1f(su_.shininess, static_cast<float>(o.mat.shininess));
        }
        else
            mgl::glUniformMatrix4fv(shadow_model_, 1, GL_FALSE, o.model.data());
        mgl::glBindVertexArray(o.mesh->vao);
        if (o.instances > 0)
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::Triangles, 0, o.mesh->count, o.pillars ? pillar_count_ : o.instances);
        else
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, o.mesh->count);
    }

    // Pass 1: the scene from the sun into the depth texture.
    void InteractiveScene::shadow_pass()
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, shadow_fbo_);
        mgl::glViewport(0, 0, kShadowSize, kShadowSize);
        mgl::glEnable(mgl::Capability::DepthTest);
        mgl::glDepthFunc(mgl::CompareFunc::Less);
        mgl::glDepthMask(GL_TRUE);
        mgl::glDisable(mgl::Capability::CullFace);
        mgl::glClearDepthf(1.0f);
        mgl::glClear(mgl::ClearBufferBit::Depth);
        // Depth offset (ES 3.0 3.6.2): o = m * factor + r * units pushes every stored depth away from the sun, so lit
        // surfaces do not shadow themselves ("acne").
        mgl::glEnable(mgl::Capability::PolygonOffsetFill);
        mgl::glPolygonOffset(kOffsetFactor, kOffsetUnits);
        mgl::glUseProgram(shadow_program_);
        for (const GpuObject& o : objects_) draw_object(o, false);
        mgl::glDisable(mgl::Capability::PolygonOffsetFill);
    }

    // Pass 2: the scene from the camera.
    void InteractiveScene::main_pass(const View& v)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glViewport(0, 0, v.w, v.h);
        const V3 clear = gamma(kHorizon);
        mgl::glClearColor(static_cast<float>(clear.x), static_cast<float>(clear.y), static_cast<float>(clear.z), 1.0f);
        mgl::glClearDepthf(1.0f);
        mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
        mgl::glEnable(mgl::Capability::DepthTest);
        mgl::glDepthFunc(mgl::CompareFunc::Less);
        mgl::glEnable(mgl::Capability::CullFace);
        mgl::glCullFace(mgl::CullFace::Back);

        mgl::glUseProgram(scene_program_);
        const dm::Mat4 vp = proj_matrix(v) * view_matrix(v);
        mgl::glUniformMatrix4fv(su_.view_proj, 1, GL_FALSE, vp.data());
        uniform3(su_.eye, v.eye);
        mgl::glUniform1i(su_.shadows, toggles_.shadows ? 1 : 0);
        mgl::glUniform1i(su_.normal_mapping, toggles_.normal_mapping ? 1 : 0);
        mgl::glUniform1i(su_.blinn, toggles_.blinn ? 1 : 0);
        mgl::glUniform1i(su_.wire, toggles_.wire ? 1 : 0);
        mgl::glUniform1f(su_.fog_density, toggles_.fog ? static_cast<float>(kFogDensity) : 0.0f);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        for (const GpuObject& o : objects_) draw_object(o, true);
    }

    // Pass 3: the sky, behind everything (depth 1.0, LEQUAL), without writing depth.
    void InteractiveScene::sky_pass(const View& v)
    {
        mgl::glDepthFunc(mgl::CompareFunc::Lequal);
        mgl::glDepthMask(GL_FALSE);
        mgl::glDisable(mgl::Capability::CullFace);  // the camera is inside the cube
        mgl::glUseProgram(sky_program_);
        const dm::Mat4 view = view_matrix(v), proj = proj_matrix(v);
        mgl::glUniformMatrix4fv(sky_view_, 1, GL_FALSE, view.data());
        mgl::glUniformMatrix4fv(sky_proj_, 1, GL_FALSE, proj.data());
        mgl::glBindVertexArray(sky_cube_.vao);
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, sky_cube_.count);
        mgl::glDepthMask(GL_TRUE);
        mgl::glDepthFunc(mgl::CompareFunc::Less);
    }

    // Key 5: the shadow map in a corner, with a white frame cleared through the scissor rectangle.
    void InteractiveScene::inset_pass(int w, int h)
    {
        const InsetRect r = inset_rect(w, h);
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glEnable(mgl::Capability::ScissorTest);
        mgl::glScissor(r.x - r.border, r.y - r.border, r.size + 2 * r.border, r.size + 2 * r.border);
        mgl::glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        mgl::glClear(mgl::ClearBufferBit::Color);
        mgl::glDisable(mgl::Capability::ScissorTest);
        mgl::glViewport(r.x, r.y, r.size, r.size);
        mgl::glUseProgram(inset_program_);
        mgl::glBindVertexArray(empty_vao_);
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        mgl::glViewport(0, 0, w, h);
        mgl::glEnable(mgl::Capability::DepthTest);
    }

    void InteractiveScene::draw_frame(int w, int h)
    {
        const View v = make_view(pose_, w, h);
        shadow_pass();
        main_pass(v);
        sky_pass(v);
        if (toggles_.inset) inset_pass(w, h);
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
    }

    void InteractiveScene::shutdown(demo::Env& /*env*/)
    {
        mgl::glBindSampler(3, mgl::SamplerId{0});
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        mgl::glUseProgram(mgl::ProgramId{0});
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        for (Mesh& m : meshes_) mgl::glDeleteVertexArrays(1, &m.vao);
        meshes_.clear();
        mgl::glDeleteVertexArrays(1, &empty_vao_);
        for (mgl::BufferId& b : buffers_) mgl::glDeleteBuffers(1, &b);
        buffers_.clear();
        for (mgl::TextureId& t : albedo_tex_) mgl::glDeleteTextures(1, &t);
        mgl::glDeleteTextures(1, &normal_tex_);
        mgl::glDeleteTextures(1, &sky_tex_);
        mgl::glDeleteTextures(1, &shadow_tex_);
        mgl::glDeleteFramebuffers(1, &shadow_fbo_);
        mgl::glDeleteSamplers(1, &raw_sampler_);
        mgl::glDeleteProgram(scene_program_);
        mgl::glDeleteProgram(shadow_program_);
        mgl::glDeleteProgram(sky_program_);
        mgl::glDeleteProgram(inset_program_);
        scene_program_ = shadow_program_ = sky_program_ = inset_program_ = mgl::ProgramId{0};
    }

    // =============================================================================================
    // 8. Validation
    // =============================================================================================
    /// Camera poses of the validation, set exactly like input sets them (pose_).
    Pose specular_pose()
    {
        // Look back along the mirror direction of the sun on the monolith's +X face: the highlight is at the target.
        const double yaw = radians(kMonolith.yaw_deg);
        const V3 n = rot_y({1, 0, 0}, yaw), L = sun_dir();
        const V3 target = kMonolith.center + rot_y({kMonolith.half.x, 0, 0}, yaw) + V3{0, 1.1, 0};
        const V3 R = n * (2.0 * dot(n, L)) - L;
        return look_from(target + R * 4.0, target, 60.0);
    }
    Pose wall_pose()
    {
        const double yaw = radians(kWall.yaw_deg);
        const V3 target = kWall.center + rot_y({0.6, 0.1, kWall.half.z}, yaw);
        return look_from(target + rot_y({0, 0, 1}, yaw) * 2.5 + V3{0.4, 0.35, 0}, target, 60.0);
    }
    const Pose kShadowPose = look_from({4.0, 6.0, 4.5}, {0.4, 0.0, -3.0}, 60.0);
    const Pose kColonnadePose = look_from({14.5, 2.2, -4.0}, {7.5, 1.6, -4.0}, 80.0);
    const Pose kSkyPose{{0.0, 1.8, 6.0}, std::atan2(sun_dir().x, -sun_dir().z), radians(22.0), 65.0};  // facing the sun
    const Pose kFogPose{{0.0, 1.6, 22.0}, radians(180.0), radians(-6.0), 60.0};

    struct Counts
    {
        int sky = 0, ground_lit = 0, ground_shadow = 0, object = 0, instance = 0, back = 0;
    };
    Counts count(const std::vector<Probe>& probes)
    {
        Counts c;
        for (const Probe& p : probes)
        {
            if (p.id == kIdSky) ++c.sky;
            else if (p.id == kIdGround) (p.occluded ? c.ground_shadow : c.ground_lit)++;
            else (is_instance(p.id) ? c.instance : c.object)++;
            if (p.id != kIdSky && p.ndl < 0.0) ++c.back;
        }
        return c;
    }

    void InteractiveScene::check_bulk(demo::Check& check, const char* name, const Image& img, const std::vector<Probe>& probes, const View& v,
                                      const Params& prm, const ProbeFilter& filter)
    {
        const Tally t = evaluate(probes, img, v.eye, prm, filter);
        check.expect(t.checked > 0 && t.bad == 0, std::format("{}: {} stable pixels match the CPU ray-cast reference", name, t.checked),
                     std::format("{} mismatches, worst {:.1f} levels outside the predicted range; {}", t.bad, t.worst, t.first));
    }

    /// Flips one technique with its number key, checks the whole image against the CPU model WITH the flip, and shows that
    /// the pixels where the model predicts a change really changed (the unflipped prediction fails there).
    void InteractiveScene::check_toggle(demo::Env& env, demo::Check& check, const Reference& /*ref*/, const char* name, char key, const Params& toggled,
                                        const std::vector<Probe>& probes, const View& v, int min_changed)
    {
        const Params base = toggles_.params();
        on_key(env, key);
        const Image img = render_image(env);
        on_key(env, key);
        check_bulk(check, std::format("key {} ({})", key, name).c_str(), img, probes, v, toggled);
        int predicted = 0, observed = 0;
        for (const Probe& p : probes)
        {
            if (!predicted_change(p, v.eye, base, toggled)) continue;
            ++predicted;
            if (outside(img.at(p.x, p.y), expect(p, v.eye, base)) > kTol) ++observed;
        }
        check.expect(predicted >= min_changed && observed == predicted,
                     std::format("key {} ({}): the image changes at the {} pixels where the CPU predicts a change", key, name, predicted),
                     std::format("predicted {} (need >= {}), changed {}", predicted, min_changed, observed));
    }

    void InteractiveScene::check_state(demo::Check& check)
    {
        GLint divisor = -1;
        mgl::glBindVertexArray(pillars_.vao);
        mgl::glGetVertexAttribiv(kLocInstance, mgl::VertexAttribParameter::ArrayDivisor, &divisor);
        check.expect_eq(divisor, 1, "pillar VAO: VERTEX_ATTRIB_ARRAY_DIVISOR of a_instance == 1");
        GLint tint_enabled = -1;
        mgl::glBindVertexArray(trunks_.vao);
        mgl::glGetVertexAttribiv(kLocTint, mgl::VertexAttribParameter::ArrayEnabled, &tint_enabled);
        check.expect_eq(tint_enabled, 0, "trunk VAO: a_tint array disabled (the current value is used)");
        mgl::glBindVertexArray(wall_.vao);
        mgl::glGetVertexAttribiv(kLocInstance, mgl::VertexAttribParameter::ArrayDivisor, &divisor);
        check.expect_eq(divisor, 0, "wall VAO: VERTEX_ATTRIB_ARRAY_DIVISOR of a_instance == 0 (initial value)");
        std::array<GLfloat, 4> cur{};
        mgl::glGetVertexAttribfv(kLocInstance, mgl::VertexAttribParameter::CurrentVertexAttrib, cur.data());
        check.expect(cur == std::array<GLfloat, 4>{0.0f, 0.0f, 0.0f, 1.0f}, "CURRENT_VERTEX_ATTRIB 5 is still (0,0,0,1) after the instanced draws (ES 3.0 2.9.3)",
                     std::format("({}, {}, {}, {})", static_cast<double>(cur[0]), static_cast<double>(cur[1]), static_cast<double>(cur[2]),
                                 static_cast<double>(cur[3])));
        mgl::glGetVertexAttribfv(kLocTint, mgl::VertexAttribParameter::CurrentVertexAttrib, cur.data());
        check.expect(cur == std::array<GLfloat, 4>{1.0f, 1.0f, 1.0f, 1.0f}, "CURRENT_VERTEX_ATTRIB 6 is still (1,1,1,1) after the instanced draws",
                     std::format("({}, {}, {}, {})", static_cast<double>(cur[0]), static_cast<double>(cur[1]), static_cast<double>(cur[2]),
                                 static_cast<double>(cur[3])));
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        GLint mode = 0, sampler = 0;
        mgl::glActiveTexture(mgl::TextureUnit::Texture2);
        mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::CompareMode, &mode);
        check.expect_eq(mode, static_cast<GLint>(GL_COMPARE_REF_TO_TEXTURE), "shadow map TEXTURE_COMPARE_MODE == COMPARE_REF_TO_TEXTURE");
        mgl::glActiveTexture(mgl::TextureUnit::Texture3);
        mgl::glGetIntegerv(mgl::GetParameter::SamplerBinding, &sampler);
        check.expect_eq(static_cast<GLuint>(sampler), raw_sampler_.value, "unit 3 has the raw-depth sampler object bound");
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
    }

    /// Every pillar instance appears at its projected position with its own tint; with one instance fewer the last one
    /// is gone and the background predicted by the CPU shows through.
    void InteractiveScene::check_instances(demo::Env& env, demo::Check& check, const Reference& ref)
    {
        pose_ = kColonnadePose;
        const View v = make_view(pose_, env.width, env.height);
        const Image img = render_image(env);
        Reference five = ref;
        five.shapes = make_shapes(kPillarCount - 1);
        int found = 0;
        std::optional<Probe> last_with, last_without;
        for (int i = 0; i < kPillarCount; ++i)
        {
            const Instance& inst = kPillars[static_cast<std::size_t>(i)];
            const int id = kIdPillar0 + i;
            const V3 q = inst.offset + (kPillarCenter + V3{kPillarHalf.x, 0, 0}) * inst.scale;  // centre of the +X face (towards the camera)
            double wx = 0, wy = 0;
            std::optional<Probe> probe;
            if (project(v, q, wx, wy))
            {
                // The nearest stable pixel around the projection (+-4 px) that the CPU sees on this instance.
                for (int r = 0; r <= 4 && !probe; ++r)
                    for (int dy = -r; dy <= r && !probe; ++dy)
                        for (int dx = -r; dx <= r && !probe; ++dx)
                        {
                            const int x = static_cast<int>(std::floor(wx)) + dx, y = static_cast<int>(std::floor(wy)) + dy;
                            if (x < 3 || y < 3 || x >= env.width - 3 || y >= env.height - 3) continue;
                            std::optional<Probe> p = make_probe(ref, v, x, y);
                            if (!p || p->id != id) continue;
                            if (i == kPillarCount - 1)
                            {
                                // For the negative control the background behind this pixel must be predictable too, and
                                // clearly different from the pillar.
                                std::optional<Probe> behind = make_probe(five, v, x, y);
                                if (!behind || !separated(expect(*p, v.eye, Params{}), expect(*behind, v.eye, Params{}))) continue;
                                last_without = behind;
                            }
                            probe = p;
                        }
            }
            const bool ok = probe && outside(img.at(probe->x, probe->y), expect(*probe, v.eye, Params{})) <= kTol;
            if (ok) ++found;
            check.expect(ok, std::format("pillar instance {} (offset {:.1f},{:.1f}, scale {:.2f}) is drawn at its projected position", i, inst.offset.x,
                                         inst.offset.z, inst.scale),
                         probe ? std::format("pixel ({},{}) = {} predicted {:.0f}..{:.0f} (red)", probe->x, probe->y,
                                             demo::format_rgba({img.at(probe->x, probe->y)[0], img.at(probe->x, probe->y)[1], img.at(probe->x, probe->y)[2], 255}),
                                             expect(*probe, v.eye, Params{}).lo.x * 255, expect(*probe, v.eye, Params{}).hi.x * 255)
                               : std::string("no stable pixel of this instance near its projection"));
            if (i == kPillarCount - 1) last_with = probe;
        }
        check.expect_eq(found, kPillarCount, "number of pillar instances found at their CPU-predicted positions");

        // Negative control: glDrawArraysInstanced with instanceCount - 1 drops the LAST instance (instances 0..n-2 are drawn).
        pillar_count_ = kPillarCount - 1;
        const Image fewer = render_image(env);
        pillar_count_ = kPillarCount;
        if (last_with && last_without)
        {
            const std::uint8_t* px = fewer.at(last_with->x, last_with->y);
            const double d_pillar = outside(px, expect(*last_with, v.eye, Params{}));
            const double d_behind = outside(px, expect(*last_without, v.eye, Params{}));
            check.expect(d_behind <= kTol && d_pillar > kTol,
                         "instanceCount 5: the last pillar is gone, its pixel shows the CPU-predicted background",
                         std::format("pixel ({},{}): {:.1f} levels from the background prediction (id {}), {:.1f} from the pillar", last_with->x, last_with->y,
                                     d_behind, last_without->id, d_pillar));
        }
        else
            check.expect(false, "instanceCount 5 negative control", "no pixel of the last pillar with a stable background behind it");
        int kept = 0;
        const std::vector<Probe> probes = build_probes(five, v);
        const Tally t = evaluate(probes, fewer, v.eye, Params{});
        for (const Probe& p : probes)
            if (p.id >= kIdPillar0 && p.id < kIdPillar0 + kPillarCount - 1) ++kept;
        check.expect(t.bad == 0 && kept > 0, std::format("instanceCount 5: the image matches the CPU model without the last pillar ({} pixels, {} on pillars 0-4)",
                                                         t.checked, kept),
                     std::format("{} mismatches; {}", t.bad, t.first));
    }

    void InteractiveScene::validate(demo::Env& env, demo::Check& check)
    {
        const Reference ref{make_shapes(kPillarCount), &albedo_images_, &normal_image_, &sky_image_, light_};
        const Pose saved_pose = pose_;
        const Toggles saved_toggles = toggles_;
        toggles_ = Toggles{};
        const Params base{};

        check_state(check);

        // ---- A. the start pose: every category, plus the shadow-map inset (key 5) -----------------------------------
        {
            pose_ = kStartPose;
            const View v = make_view(pose_, env.width, env.height);
            const std::vector<Probe> probes = build_probes(ref, v);
            const Image img = render_image(env);
            check_bulk(check, "start pose", img, probes, v, base);
            const Counts c = count(probes);
            check.expect(c.sky >= 20 && c.ground_lit >= 20 && c.object >= 20 && c.instance >= 5,
                         "start pose: sky, lit ground, objects and instances are all represented",
                         std::format("sky {} ground lit {} shadow {} objects {} instances {}", c.sky, c.ground_lit, c.ground_shadow, c.object, c.instance));
        }
        check_inset(env, check, ref);

        // ---- B. shadows: lit and shadowed ground around the monolith; key 1 ------------------------------------------
        {
            pose_ = kShadowPose;
            const View v = make_view(pose_, env.width, env.height);
            const std::vector<Probe> probes = build_probes(ref, v);
            const Image img = render_image(env);
            check_bulk(check, "shadow pose", img, probes, v, base);
            const Counts c = count(probes);
            check.expect(c.ground_lit >= 30 && c.ground_shadow >= 30, "shadow pose: lit AND shadowed ground at CPU-predicted points",
                         std::format("lit {} shadowed {}", c.ground_lit, c.ground_shadow));
            Params p = base;
            p.shadows = false;
            check_toggle(env, check, ref, "shadows off", '1', p, probes, v, 30);
            check_wire(env, check, ref);
        }

        // ---- C. Blinn-Phong on the monolith; key 6 Phong -----------------------------------------------------------------
        {
            pose_ = specular_pose();
            const View v = make_view(pose_, env.width, env.height);
            const std::vector<Probe> probes = build_probes(ref, v);
            const Image img = render_image(env);
            check_bulk(check, "specular pose", img, probes, v, base);
            int glossy = 0;
            for (const Probe& p : probes)
                if (p.id == kIdMonolith)
                {
                    Probe matte = p;
                    matte.mat.specular = 0.0;
                    if ((expect(p, v.eye, base).lo.x - expect(matte, v.eye, base).hi.x) * 255.0 > 20.0) ++glossy;
                }
            check.expect(glossy >= 20, "specular pose: monolith pixels whose Blinn-Phong highlight adds > 20 levels match", std::format("{} pixels", glossy));
            Params p = base;
            p.blinn = false;
            check_toggle(env, check, ref, "Phong", '6', p, probes, v, 20);
        }

        // ---- D. normal mapping on the stone wall; key 2 -------------------------------------------------------------------
        {
            pose_ = wall_pose();
            const View v = make_view(pose_, env.width, env.height);
            const std::vector<Probe> probes = build_probes(ref, v);
            const Image img = render_image(env);
            check_bulk(check, "wall pose", img, probes, v, base);
            int bevel = 0, flat = 0;
            for (const Probe& p : probes)
                if (p.has_map) (dot(p.n_map, p.n_geo) < 0.999 ? bevel : flat)++;
            check.expect(bevel >= 20 && flat >= 20, "wall pose: normal-mapped bevels (tilted normal) and flat panel tops both match",
                         std::format("bevel {} flat {}", bevel, flat));
            Params p = base;
            p.normal_mapping = false;
            check_toggle(env, check, ref, "normal mapping off", '2', p, probes, v, 20);
        }

        // ---- E. instancing ------------------------------------------------------------------------------------------------
        check_instances(env, check, ref);

        // ---- F. the sky per direction ---------------------------------------------------------------------------------------
        {
            pose_ = kSkyPose;
            const View v = make_view(pose_, env.width, env.height);
            const std::vector<Probe> probes = build_probes(ref, v);
            const Image img = render_image(env);
            check_bulk(check, "sky pose", img, probes, v, base);
            const Counts c = count(probes);
            check.expect(c.sky >= 200, "sky pose: sky pixels (cube-map colour per view direction) match", std::format("{} sky pixels", c.sky));
        }

        // ---- G. fog; key 3 ----------------------------------------------------------------------------------------------------
        check_fog(env, check, ref);

        // ---- H. the controls ----------------------------------------------------------------------------------------------------
        check_input(env, check);

        // Leave the start view on screen for the capture.
        pose_ = saved_pose;
        toggles_ = saved_toggles;
        draw_frame(env.width, env.height);
        check.expect_no_gl_error("no GL error during validation");
    }

    void InteractiveScene::check_fog(demo::Env& env, demo::Check& check, const Reference& ref)
    {
        pose_ = kFogPose;
        const View v = make_view(pose_, env.width, env.height);
        const std::vector<Probe> probes = build_probes(ref, v);
        const Image img = render_image(env);
        const Params base{};
        check_bulk(check, "fog pose", img, probes, v, base);
        Params nofog = base;
        nofog.fog = false;
        on_key(env, '3');
        const Image clear = render_image(env);
        on_key(env, '3');
        // Ground pixels at known distances in the middle column: fog factor f = exp(-density * d).
        for (const double target : {6.0, 15.0, 35.0, 80.0})
        {
            std::optional<Probe> best;
            double best_err = 1e9;
            for (int y = 3; y < env.height - 3; ++y)
                for (const int x : {env.width / 2, env.width / 2 - 3, env.width / 2 + 3})
                {
                    const V3 d = ray_dir(v, x + 0.5, y + 0.5);
                    if (d.y >= 0.0) continue;
                    const double dist = v.eye.y / -d.y;
                    if (std::abs(dist - target) >= best_err) continue;
                    if (std::optional<Probe> p = make_probe(ref, v, x, y); p && p->id == kIdGround)
                    {
                        best = p;
                        best_err = std::abs(dist - target);
                    }
                }
            if (!best)
            {
                check.expect(false, std::format("fog at {:.0f} units", target), "no stable ground pixel");
                continue;
            }
            const double dist = length(best->p - v.eye), f = std::exp(-kFogDensity * dist);
            const std::uint8_t* a = img.at(best->x, best->y);
            const std::uint8_t* b = clear.at(best->x, best->y);
            const Range ra = expect(*best, v.eye, base), rb = expect(*best, v.eye, nofog);
            check.expect(outside(a, ra) <= kTol && outside(b, rb) <= kTol,
                         std::format("fog blend at {:.1f} units (f = {:.3f}): fogged and fog-free pixels match the CPU", dist, f),
                         std::format("pixel ({},{}) fog {} predicted G {:.0f}..{:.0f}; no fog {} predicted G {:.0f}..{:.0f}", best->x, best->y,
                                     demo::format_rgba({a[0], a[1], a[2], 255}), ra.lo.y * 255, ra.hi.y * 255, demo::format_rgba({b[0], b[1], b[2], 255}),
                                     rb.lo.y * 255, rb.hi.y * 255));
        }
        check_toggle(env, check, ref, "fog off", '3', nofog, probes, v, 30);
    }

    /// Key 4: barycentric wireframe. Pixels on a triangle edge become the wire colour, pixels far from every edge do not
    /// change at all.
    void InteractiveScene::check_wire(demo::Env& env, demo::Check& check, const Reference& ref)
    {
        const View v = make_view(pose_, env.width, env.height);
        const Image plain = render_image(env);
        on_key(env, '4');
        const Image wire = render_image(env);
        on_key(env, '4');
        const demo::Rgba8 wc{to8(kWireColor.x), to8(kWireColor.y), to8(kWireColor.z), 255};
        auto is_wire = [&](const std::uint8_t* px) {
            for (std::size_t c = 0; c < 3; ++c)
                if (std::abs(static_cast<int>(px[c]) - static_cast<int>(wc[c])) > kTol) return false;
            return true;
        };
        auto inside = [&](int x, int y) { return x >= 2 && y >= 2 && x < env.width - 2 && y < env.height - 2; };

        // Edge points: the centre of a box face lies on the diagonal shared by its two triangles; ground-cell edges.
        struct EdgePoint
        {
            V3 p;                                   // a point on a triangle edge (world)
            std::vector<std::array<V3, 2>> others;  // the other edges of the two triangles that share it
            int id;
            V3 ln;
        };
        std::vector<EdgePoint> pts;
        for (int id = 0; id < static_cast<int>(ref.shapes.size()); ++id)
        {
            const Shape& s = ref.shapes[static_cast<std::size_t>(id)];
            if (s.kind != Kind::Box) continue;
            for (const V3 ln : kFaceNormals)
            {
                const FaceFrame f = face_frame(ln);
                auto corner = [&](double a, double b) { return s.center + rot_y(mul(ln, s.half) + mul(f.t, s.half) * a + mul(f.b, s.half) * b, s.yaw); };
                const V3 c0 = corner(-1, -1), c1 = corner(1, -1), c2 = corner(1, 1), c3 = corner(-1, 1);
                pts.push_back({s.center + rot_y(mul(ln, s.half), s.yaw), {{c0, c1}, {c1, c2}, {c2, c3}, {c3, c0}}, id, ln});
            }
        }
        const double cell = 2.0 * kGroundHalf / kGroundCells;
        for (int k = -4; k <= 4; ++k)
            for (int m = -4; m <= 4; ++m)
            {
                const double a = k * cell, b = m * cell;
                // Edge x = a between z = b and b + cell; its cells are split along (x0, z0) - (x1, z1).
                pts.push_back({{a, 0, b + cell / 2},
                               {{V3{a - cell, 0, b}, V3{a, 0, b + cell}}, {V3{a, 0, b}, V3{a + cell, 0, b + cell}}, {V3{a - cell, 0, b}, V3{a + cell, 0, b}},
                                {V3{a - cell, 0, b + cell}, V3{a + cell, 0, b + cell}}},
                               kIdGround,
                               {0, 1, 0}});
                // Edge z = a between x = b and b + cell.
                pts.push_back({{b + cell / 2, 0, a},
                               {{V3{b, 0, a - cell}, V3{b + cell, 0, a}}, {V3{b, 0, a}, V3{b + cell, 0, a + cell}}, {V3{b, 0, a - cell}, V3{b, 0, a + cell}},
                                {V3{b + cell, 0, a - cell}, V3{b + cell, 0, a + cell}}},
                               kIdGround,
                               {0, 1, 0}});
            }
        int edges = 0, edges_plain_not_wire = 0, edge_fail = 0;
        std::string first;
        for (const EdgePoint& e : pts)
        {
            double wx = 0, wy = 0;
            if (!project(v, e.p, wx, wy)) continue;
            const int x = static_cast<int>(std::floor(wx)), y = static_cast<int>(std::floor(wy));
            if (!inside(x, y)) continue;
            // Every other edge of the two triangles must be >= 12 px away. Both triangles are right triangles (box face or
            // ground cell split along its diagonal) whose legs are then >= 24 px, so every altitude is >= 24 / sqrt(2) = 17 px
            // and fwidth(b_i) <= sqrt(2) / 17 = 0.083 stays below the 0.12 where scene.frag starts fading dense lines out.
            // At the probed pixel (<= 0.71 px from the edge) b_k / fwidth(b_k) <= 0.71 < 0.75: the wire weight is exactly 1.
            bool clear = true;
            for (const auto& o : e.others)
            {
                double ax = 0, ay = 0, bx = 0, by = 0;
                if (!project(v, o[0], ax, ay) || !project(v, o[1], bx, by))
                {
                    clear = false;
                    break;
                }
                const double ex = bx - ax, ey = by - ay;
                if (std::abs((x + 0.5 - ax) * ey - (y + 0.5 - ay) * ex) / std::hypot(ex, ey) < 12.0) clear = false;
            }
            if (!clear) continue;
            const Hit h = trace(ref.shapes, v.eye, ray_dir(v, x + 0.5, y + 0.5));
            if (h.id != e.id || dot(h.ln, e.ln) < 0.999) continue;  // the edge must be visible
            if (e.id >= 0 && std::abs(dot(h.n, normalize(v.eye - e.p))) < 0.25) continue;  // not edge-on
            ++edges;
            if (!is_wire(plain.at(x, y))) ++edges_plain_not_wire;
            if (!is_wire(wire.at(x, y)))
            {
                ++edge_fail;
                if (first.empty())
                    first = std::format("pixel ({},{}) id {} = {}", x, y, e.id, demo::format_rgba({wire.at(x, y)[0], wire.at(x, y)[1], wire.at(x, y)[2], 255}));
            }
        }
        check.expect(edges >= 8 && edge_fail == 0, std::format("key 4: {} pixels on triangle edges show the wire colour", edges),
                     std::format("{} edge pixels, {} not wire-coloured; {}", edges, edge_fail, first));
        check.expect(edges_plain_not_wire == edges, "key 4 off: the same edge pixels are not wire-coloured",
                     std::format("{} of {}", edges_plain_not_wire, edges));

        // Interior pixels: at least 4 px from every edge of their ground triangle, so the overlay weight is exactly 0.
        int interior = 0, interior_fail = 0;
        for (int k = -4; k < 4; ++k)
            for (int m = -4; m < 4; ++m)
            {
                const V3 c00{k * cell, 0, m * cell}, c10{(k + 1) * cell, 0, m * cell}, c11{(k + 1) * cell, 0, (m + 1) * cell};
                const V3 centroid = (c00 + c10 + c11) * (1.0 / 3.0);
                double p[4][2];
                if (!project(v, c00, p[0][0], p[0][1]) || !project(v, c10, p[1][0], p[1][1]) || !project(v, c11, p[2][0], p[2][1]) ||
                    !project(v, centroid, p[3][0], p[3][1]))
                    continue;
                const int x = static_cast<int>(std::floor(p[3][0])), y = static_cast<int>(std::floor(p[3][1]));
                if (!inside(x, y)) continue;
                double dmin = 1e9;
                for (int i = 0; i < 3; ++i)
                {
                    const double* a = p[i];
                    const double* b = p[(i + 1) % 3];
                    const double ex = b[0] - a[0], ey = b[1] - a[1];
                    dmin = std::min(dmin, std::abs((x + 0.5 - a[0]) * ey - (y + 0.5 - a[1]) * ex) / std::hypot(ex, ey));
                }
                if (dmin < 4.0 || trace(ref.shapes, v.eye, ray_dir(v, x + 0.5, y + 0.5)).id != kIdGround) continue;
                ++interior;
                for (std::size_t c = 0; c < 3; ++c)
                    if (std::abs(static_cast<int>(wire.at(x, y)[c]) - static_cast<int>(plain.at(x, y)[c])) > 1)
                    {
                        ++interior_fail;
                        break;
                    }
            }
        check.expect(interior >= 4 && interior_fail == 0, std::format("key 4: {} triangle-interior pixels are unchanged by the overlay", interior),
                     std::format("{} interior pixels, {} changed", interior, interior_fail));
    }

    /// Key 5: the inset shows the shadow map; its grey level is the stored depth of the CPU-predicted surface.
    void InteractiveScene::check_inset(demo::Env& env, demo::Check& check, const Reference& ref)
    {
        pose_ = kStartPose;
        const View v = make_view(pose_, env.width, env.height);
        const Image plain = render_image(env);
        on_key(env, '5');
        const Image img = render_image(env);
        on_key(env, '5');
        const InsetRect r = inset_rect(env.width, env.height);
        int checked = 0, bad = 0, differs = 0;
        double worst = 0;
        std::string first;
        for (int gy = 0; gy < 12; ++gy)
            for (int gx = 0; gx < 12; ++gx)
            {
                const int ix = r.x + 2 + (r.size - 4) * gx / 11, iy = r.y + 2 + (r.size - 4) * gy / 11;
                const double u = (ix - r.x + 0.5) / r.size, w = (iy - r.y + 0.5) / r.size;
                const int tx = static_cast<int>(std::floor(u * kShadowSize)), ty = static_cast<int>(std::floor(w * kShadowSize));
                // NEAREST may pick a neighbouring texel when the coordinate is within rounding of a texel edge: take the 3x3 texels.
                double dmin = 1e9, dmax = -1e9, slope = 0;
                int id = -3;
                bool usable = tx >= 1 && ty >= 1 && tx < kShadowSize - 2 && ty < kShadowSize - 2;
                for (int dy = -1; dy <= 1 && usable; ++dy)
                    for (int dx = -1; dx <= 1 && usable; ++dx)
                    {
                        const Hit h = light_hit(ref.shapes, ref.light, tx + dx, ty + dy);
                        if (!h.ok || (id != -3 && h.id != id) || (h.id >= 0 && ref.shapes[static_cast<std::size_t>(h.id)].kind == Kind::Sphere)) usable = false;
                        id = h.id;
                        const double d = (h.t - kLightNear) / (kLightFar - kLightNear);
                        dmin = std::min(dmin, d);
                        dmax = std::max(dmax, d);
                    }
                if (!usable) continue;
                // The planar surface's depth change per texel (the 3x3 range spans two texels per axis) bounds the polygon
                // offset's slope term m (ES 3.0 eq. 3.9); the stored depth lies in [dmin, dmax + factor * 2 * m + units * r].
                slope = dmax - dmin;
                const double hi_depth = dmax + static_cast<double>(kOffsetFactor) * slope + static_cast<double>(kOffsetUnits) * 1e-6;
                const double glo = std::clamp((dmin - kInsetLo) / (kInsetHi - kInsetLo), 0.0, 1.0) * 255.0;
                const double ghi = std::clamp((hi_depth - kInsetLo) / (kInsetHi - kInsetLo), 0.0, 1.0) * 255.0;
                const std::uint8_t* px = img.at(ix, iy);
                double d = 0;
                for (int c = 0; c < 3; ++c) d = std::max({d, glo - px[c], px[c] - ghi});
                ++checked;
                worst = std::max(worst, d);
                if (d > kTol)
                {
                    ++bad;
                    if (first.empty()) first = std::format("inset pixel ({},{}) texel ({},{}) = {} predicted {:.1f}..{:.1f}", ix, iy, tx, ty, px[0], glo, ghi);
                }
                const std::uint8_t* q = plain.at(ix, iy);
                if (std::abs(static_cast<int>(q[0]) - static_cast<int>(px[0])) > kTol || q[0] != q[1] || q[1] != q[2]) ++differs;
            }
        check.expect(checked >= 60 && bad == 0, std::format("key 5: {} inset pixels show the CPU-predicted shadow-map depth", checked),
                     std::format("{} mismatches, worst {:.1f} levels; {}", bad, worst, first));
        check.expect(differs * 10 >= checked * 9, "key 5 off: those pixels show the scene instead", std::format("{} of {} differ", differs, checked));

        // Outside the inset and its frame nothing changes.
        const std::vector<Probe> probes = build_probes(ref, v);
        const int bx0 = r.x - r.border - 3, bx1 = r.x + r.size + r.border + 3, by0 = r.y - r.border - 3;
        check_bulk(check, "key 5 (inset on), outside the inset", img, probes, v, Params{},
                   [&](const Probe& p) { return !(p.x >= bx0 && p.x <= bx1 && p.y >= by0); });
    }

    /// The controls are deterministic: a scripted sequence of mouse and keyboard input at a fixed frame time must end at
    /// the pose computed in closed form, and must render exactly the image of that pose.
    void InteractiveScene::check_input(demo::Env& env, demo::Check& check)
    {
        const double saved_dt = env.frame_seconds;
        const auto saved_keys = env.keys_held;
        const double dt = 1.0 / 60.0;
        env.frame_seconds = dt;
        env.keys_held = {};
        pose_ = kStartPose;
        const Image start = render_image(env);

        auto hold = [&](std::string_view keys, int frames) {
            for (const char k : keys) env.keys_held[static_cast<unsigned char>(k)] = true;
            for (int i = 0; i < frames; ++i) update_camera(env);  // exactly what render() does before drawing
            for (const char k : keys) env.keys_held[static_cast<unsigned char>(k)] = false;
        };
        on_mouse(env, {120.0f, -40.0f, 0.0f, true, false, false});   // look right and up
        on_mouse(env, {300.0f, 50.0f, 0.0f, false, false, false});   // button up: no look
        on_mouse(env, {0.0f, 0.0f, 3.0f, false, false, false});      // wheel: zoom in
        hold("w", 30);
        hold("de", 12);
        hold("sa", 9);
        hold("q", 120);                                              // into the ground: clamped at the minimum eye height
        on_mouse(env, {0.0f, -2000.0f, 0.0f, true, false, false});   // pitch clamps at +85 degrees
        on_mouse(env, {0.0f, 400.0f, 0.0f, true, false, false});
        on_mouse(env, {0.0f, 0.0f, -20.0f, false, false, false});    // fov clamps at 90 degrees
        on_mouse(env, {0.0f, 0.0f, 4.0f, false, false, false});

        // Closed form, independent of the per-frame accumulation.
        const Pose s0 = kStartPose;
        const double yaw = s0.yaw + 120.0 * kMouseSensitivity;
        const double pitch = radians(85.0) - 400.0 * kMouseSensitivity;
        const double fov = 90.0 - 4.0 * kWheelStep;
        const V3 f{std::sin(yaw), 0, -std::cos(yaw)}, rt{std::cos(yaw), 0, std::sin(yaw)};
        V3 p = s0.pos + f * (kMoveSpeed * 30 * dt);
        p = p + (rt + V3{0, 1, 0}) * (kMoveSpeed * 12 * dt / std::sqrt(2.0));
        p = p - (f + rt) * (kMoveSpeed * 9 * dt / std::sqrt(2.0));
        p.y = kMinEyeHeight;
        const Pose predicted{p, yaw, pitch, fov};

        const double pos_err = length(pose_.pos - predicted.pos);
        check.expect(pos_err < 1e-9 && std::abs(pose_.yaw - yaw) < 1e-12 && std::abs(pose_.pitch - pitch) < 1e-12 && std::abs(pose_.fov - fov) < 1e-12,
                     "scripted input (mouse look, button release, wheel, W, D+E, S+A, Q, pitch and fov clamps) ends at the CPU-predicted pose",
                     std::format("pos ({:.6f},{:.6f},{:.6f}) predicted ({:.6f},{:.6f},{:.6f}); yaw {:.6f}/{:.6f} pitch {:.6f}/{:.6f} fov {:.3f}/{:.3f}", pose_.pos.x,
                                 pose_.pos.y, pose_.pos.z, p.x, p.y, p.z, pose_.yaw, yaw, pose_.pitch, pitch, pose_.fov, fov));

        const Image driven = render_image(env);
        pose_ = predicted;
        const Image direct = render_image(env);
        int diff = 0, moved = 0;
        for (std::size_t i = 0; i < driven.px.size(); i += 4)
        {
            for (std::size_t c = 0; c < 3; ++c)
                if (std::abs(static_cast<int>(driven.px[i + c]) - static_cast<int>(direct.px[i + c])) > 1)
                {
                    ++diff;
                    break;
                }
            for (std::size_t c = 0; c < 3; ++c)
                if (driven.px[i + c] != start.px[i + c])
                {
                    ++moved;
                    break;
                }
        }
        const int total = env.width * env.height;
        check.expect(diff == 0, "the input-driven frame equals the frame of the predicted pose set directly", std::format("{} pixels differ", diff));
        check.expect(moved * 2 > total, "negative control: the input changed the view", std::format("{} of {} pixels changed", moved, total));

        on_key(env, 'r');
        check.expect(pose_.pos.x == kStartPose.pos.x && pose_.pos.y == kStartPose.pos.y && pose_.pos.z == kStartPose.pos.z && pose_.yaw == kStartPose.yaw &&
                         pose_.fov == kStartPose.fov,
                     "key R resets the camera to the start pose");
        env.frame_seconds = saved_dt;
        env.keys_held = saved_keys;
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-interactive-3d-scene",
        .title = "Interactive 3D Scene (capstone)",
        .min_es = {3, 0},
        .summary = "A small outdoor scene explored with an FPS camera: procedural textures, normal mapping, Blinn-Phong sun with a PCF "
                   "shadow map, instanced pillars and trees, a cube-map sky and fog. Fixed poses and a scripted input sequence are "
                   "checked against a CPU ray caster; each technique toggle changes the image where predicted.",
        .controls = "left mouse drag look | W/A/S/D move | Q/E down/up | wheel field of view | 1 shadows | 2 normal mapping | 3 fog | "
                    "4 wireframe overlay | 5 shadow-map inset | 6 Blinn-Phong/Phong | R reset camera | ESC quit",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<InteractiveScene>(argc, argv, info);
}
