// es30-normal-mapping
//
// A lit 3D scene with tangent-space normal mapping and a specular map: a brick wall, a rotating
// brick crate, a brick sphere and an orbiting point light. All three material textures are
// generated on the CPU from a height function (no asset files):
//
//   unit 0  albedo    RGBA8   brick / mortar colours
//   unit 1  normal    RGBA8   tangent-space normals derived from the brick height field
//   unit 2  specular  R8      bricks shine, mortar does not
//
// Every texture carries a complete mip chain (128x128 down to 1x1) that is built on the CPU and
// uploaded level by level, so the CPU reference knows the content of every level. Tangents are
// generated on the CPU from positions and UVs (per-triangle dP/du, Gram-Schmidt against the vertex
// normal, handedness sign from the UV orientation). The wall is built from two panels, the right one
// with a MIRRORED u axis, so its tangent frame is left-handed (w = -1): ignoring the handedness
// sign gives visibly wrong lighting there, and the validation proves it.
//
// Validation (--validate), all against a CPU evaluation of the same equations in double precision:
//   * wall, NEAREST filtering (texel selection known exactly): lit colour at ~2500 pixels whose
//     position, UV, tangent frame and normal-map texel are computed analytically (ray / plane
//     intersection), with and without the normal map, with and without the specular map, and with
//     the handedness sign ignored. Negative controls: a CPU model with the wrong handedness and a
//     model with an inverted green channel (DirectX-style normal map) must NOT match;
//   * the rotating crate (non-axis-aligned faces: model matrix and normal matrix);
//   * the trilinear path: a face-on wall at exactly 2 texels per pixel (level of detail 1);
//   * texture unit assignment, sampler uniforms, mip chain sizes, CPU tangent generation.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    // ============================================================================================
    // Double precision vector for the CPU reference
    // ============================================================================================
    struct D3
    {
        double x = 0, y = 0, z = 0;
    };
    constexpr D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    constexpr D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    constexpr D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
    constexpr double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    constexpr D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    inline double length(D3 a) { return std::sqrt(dot(a, a)); }
    inline D3 normalize(D3 a)
    {
        const double l = length(a);
        return l > 0.0 ? a * (1.0 / l) : a;
    }
    inline dm::Vec3 to_vec3(D3 a) { return {static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)}; }

    // Upper 3x3 of a model matrix applied to a direction / full matrix applied to a point.
    inline D3 transform_direction(const dm::Mat4& m, D3 v)
    {
        auto e = [&](int r, int c) { return static_cast<double>(m.at(r, c)); };
        return {e(0, 0) * v.x + e(0, 1) * v.y + e(0, 2) * v.z, e(1, 0) * v.x + e(1, 1) * v.y + e(1, 2) * v.z,
                e(2, 0) * v.x + e(2, 1) * v.y + e(2, 2) * v.z};
    }
    inline D3 transform_point(const dm::Mat4& m, D3 p)
    {
        return transform_direction(m, p) + D3{static_cast<double>(m.at(0, 3)), static_cast<double>(m.at(1, 3)), static_cast<double>(m.at(2, 3))};
    }

    double smoothstep(double e0, double e1, double x)
    {
        const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
    int wrap_repeat(int i, int n) { return ((i % n) + n) % n; }

    // ============================================================================================
    // Procedural textures (CPU): bricks from a height function
    // ============================================================================================
    constexpr int kTexSize = 128;  // level 0 of every map (2 bricks wide, 4 courses high per tile)
    constexpr int kLevels = 8;     // 128, 64, 32, 16, 8, 4, 2, 1
    constexpr int level_size(int level) { return std::max(1, kTexSize >> level); }

    struct CpuTexture
    {
        int channels = 4;
        std::array<std::vector<std::uint8_t>, kLevels> levels;

        [[nodiscard]] int texel(int level, int i, int j, int c) const
        {
            const int n = level_size(level);
            return levels[static_cast<std::size_t>(level)][static_cast<std::size_t>((j * n + i) * channels + c)];
        }
    };

    std::uint32_t hash_u32(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }
    double lattice(int x, int y, int period, std::uint32_t seed)
    {
        const auto ux = static_cast<std::uint32_t>(wrap_repeat(x, period));
        const auto uy = static_cast<std::uint32_t>(wrap_repeat(y, period));
        return static_cast<double>(hash_u32(hash_u32(ux * 0x9E3779B1U + seed) ^ (uy * 0x85EBCA6BU))) / 4294967296.0;
    }
    // Tileable value noise: the lattice repeats every `period` cells.
    double value_noise(double x, double y, int period, std::uint32_t seed)
    {
        const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
        const double fx = x - x0, fy = y - y0;
        const double sx = fx * fx * (3.0 - 2.0 * fx), sy = fy * fy * (3.0 - 2.0 * fy);
        const double a = lattice(x0, y0, period, seed), b = lattice(x0 + 1, y0, period, seed);
        const double c = lattice(x0, y0 + 1, period, seed), d = lattice(x0 + 1, y0 + 1, period, seed);
        return (a + (b - a) * sx) * (1.0 - sy) + (c + (d - c) * sx) * sy;
    }

    struct BrickTexel
    {
        double height = 0;       // 0 = mortar floor, ~1 = brick face
        double edge = 0;         // distance to the brick border in texels
        bool mortar = false;
        std::uint32_t id = 0;    // brick identity (colour variation)
    };

    BrickTexel brick_texel(int i, int j)
    {
        // 4 courses of 32 texels; odd courses are shifted by half a brick (32 texels); bricks are 64 wide.
        const int course = j / 32;
        const int shifted = (i + ((course & 1) != 0 ? 32 : 0)) % kTexSize;
        const int column = shifted / 64;
        const int bx = shifted % 64, by = j % 32;
        const double dx = std::min(bx, 63 - bx) + 0.5, dy = std::min(by, 31 - by) + 0.5;
        BrickTexel t;
        t.edge = std::min(dx, dy);
        t.mortar = t.edge < 2.0;  // 4 texels of mortar between neighbouring bricks
        t.id = hash_u32(static_cast<std::uint32_t>(course * 2 + column) + 77U);
        const double grain = 0.10 * (value_noise(i / 4.0, j / 4.0, 32, 11) - 0.5) + 0.05 * (value_noise(i / 16.0, j / 16.0, 8, 12) - 0.5);
        const double brick_offset = 0.06 * (static_cast<double>(t.id & 255U) / 255.0 - 0.5);
        t.height = (t.mortar ? 0.0 : smoothstep(2.0, 5.0, t.edge) * (1.0 + brick_offset)) + grain;
        return t;
    }

    std::uint8_t to_byte(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    void build_mip_chain(CpuTexture& t)
    {
        for (int level = 1; level < kLevels; ++level)
        {
            const int n = level_size(level);
            auto& dst = t.levels[static_cast<std::size_t>(level)];
            dst.assign(static_cast<std::size_t>(n * n * t.channels), 0);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    for (int c = 0; c < t.channels; ++c)
                    {
                        // 2x2 box filter of the previous level, rounded to nearest.
                        const int sum = t.texel(level - 1, 2 * i, 2 * j, c) + t.texel(level - 1, 2 * i + 1, 2 * j, c) +
                                        t.texel(level - 1, 2 * i, 2 * j + 1, c) + t.texel(level - 1, 2 * i + 1, 2 * j + 1, c);
                        dst[static_cast<std::size_t>((j * n + i) * t.channels + c)] = static_cast<std::uint8_t>((sum + 2) / 4);
                    }
        }
    }

    struct TextureSet
    {
        CpuTexture albedo, normal, specular;
    };

    TextureSet make_textures()
    {
        TextureSet set;
        set.albedo.channels = 4;
        set.normal.channels = 4;
        set.specular.channels = 1;
        auto& a = set.albedo.levels[0];
        auto& n = set.normal.levels[0];
        auto& s = set.specular.levels[0];
        a.resize(static_cast<std::size_t>(kTexSize * kTexSize * 4));
        n.resize(static_cast<std::size_t>(kTexSize * kTexSize * 4));
        s.resize(static_cast<std::size_t>(kTexSize * kTexSize));

        std::vector<double> height(static_cast<std::size_t>(kTexSize * kTexSize));
        std::vector<BrickTexel> bricks(height.size());
        for (int j = 0; j < kTexSize; ++j)
            for (int i = 0; i < kTexSize; ++i)
            {
                const auto idx = static_cast<std::size_t>(j * kTexSize + i);
                bricks[idx] = brick_texel(i, j);
                height[idx] = bricks[idx].height;
            }
        auto h = [&](int i, int j) { return height[static_cast<std::size_t>(wrap_repeat(j, kTexSize) * kTexSize + wrap_repeat(i, kTexSize))]; };

        constexpr double kBumpStrength = 2.0;
        for (int j = 0; j < kTexSize; ++j)
            for (int i = 0; i < kTexSize; ++i)
            {
                const auto idx = static_cast<std::size_t>(j * kTexSize + i);
                const BrickTexel& b = bricks[idx];

                // --- normal map: n = normalize(-S dh/du, -S dh/dv, 1), central differences (tileable) ---
                const double dh_du = (h(i + 1, j) - h(i - 1, j)) * 0.5;
                const double dh_dv = (h(i, j + 1) - h(i, j - 1)) * 0.5;
                const D3 nrm = normalize({-kBumpStrength * dh_du, -kBumpStrength * dh_dv, 1.0});
                n[idx * 4 + 0] = to_byte(nrm.x * 0.5 + 0.5);
                n[idx * 4 + 1] = to_byte(nrm.y * 0.5 + 0.5);
                n[idx * 4 + 2] = to_byte(nrm.z * 0.5 + 0.5);
                n[idx * 4 + 3] = 255;

                // --- albedo ---
                const double r1 = static_cast<double>(b.id & 255U) / 255.0;
                const double r2 = static_cast<double>((b.id >> 8) & 255U) / 255.0;
                const double r3 = static_cast<double>((b.id >> 16) & 255U) / 255.0;
                const double grain_a = value_noise(i / 4.0, j / 4.0, 32, 21);
                const double grain_b = value_noise(i / 2.0, j / 2.0, 64, 22);
                double cr, cg, cb;
                if (b.mortar)
                {
                    const double k = 0.70 + 0.35 * grain_b;
                    cr = 0.66 * k;
                    cg = 0.63 * k;
                    cb = 0.58 * k;
                }
                else
                {
                    const double tone = 0.72 + 0.50 * r1;
                    const double shade = (0.70 + 0.30 * smoothstep(2.0, 6.0, b.edge)) * (0.82 + 0.36 * grain_a);
                    cr = 0.62 * tone * shade;
                    cg = (0.25 + 0.10 * r2) * tone * shade;
                    cb = (0.17 + 0.06 * r2) * tone * shade;
                }
                a[idx * 4 + 0] = to_byte(cr);
                a[idx * 4 + 1] = to_byte(cg);
                a[idx * 4 + 2] = to_byte(cb);
                a[idx * 4 + 3] = 255;

                // --- specular map: mortar is matte (~0), bricks are semi-glossy, some bricks are glazed ---
                double spec;
                if (b.mortar) spec = 0.03 * grain_b;
                else if (r3 < 0.25) spec = 0.80 + 0.20 * grain_a;
                else spec = 0.22 + 0.25 * grain_a;
                s[idx] = to_byte(spec);
            }
        build_mip_chain(set.albedo);
        build_mip_chain(set.normal);
        build_mip_chain(set.specular);
        return set;
    }

    // Bilinear lookup in one level with REPEAT wrapping (ES 3.0 section 3.8.10.2). Returns 0..255 values.
    std::array<double, 3> bilinear(const CpuTexture& t, int level, double u, double v)
    {
        const int n = level_size(level);
        const double us = u * n - 0.5, vs = v * n - 0.5;
        const int i0 = static_cast<int>(std::floor(us)), j0 = static_cast<int>(std::floor(vs));
        const double fa = us - i0, fb = vs - j0;
        std::array<double, 3> r{};
        for (int c = 0; c < std::min(3, t.channels); ++c)
        {
            auto at = [&](int i, int j) { return static_cast<double>(t.texel(level, wrap_repeat(i, n), wrap_repeat(j, n), c)); };
            r[static_cast<std::size_t>(c)] = (1 - fa) * (1 - fb) * at(i0, j0) + fa * (1 - fb) * at(i0 + 1, j0) +
                                             (1 - fa) * fb * at(i0, j0 + 1) + fa * fb * at(i0 + 1, j0 + 1);
        }
        return r;
    }

    // ============================================================================================
    // Geometry: planar faces, meshes, CPU tangent generation
    // ============================================================================================
    // A planar parallelogram P(s, t) = origin + s * eu + t * ev, s and t in [0, 1], with texture
    // coordinates (s * tile_u, t * tile_v). `normal` is the outward normal. When cross(eu, ev) points
    // opposite to `normal` the texture is MIRRORED on the surface (left-handed UV basis).
    struct Face
    {
        D3 origin, eu, ev, normal;
        double tile_u = 1, tile_v = 1;
    };

    Face transformed(const Face& f, const dm::Mat4& m)
    {
        return {transform_point(m, f.origin), transform_direction(m, f.eu), transform_direction(m, f.ev), transform_direction(m, f.normal),
                f.tile_u, f.tile_v};
    }

    std::vector<Face> wall_faces()
    {
        // Two panels meeting at x = 0. The right panel runs its u axis backwards: a mirrored texture.
        return {
            {{-4, -3, 0}, {4, 0, 0}, {0, 6, 0}, {0, 0, 1}, 4, 6},
            {{4, -3, 0}, {-4, 0, 0}, {0, 6, 0}, {0, 0, 1}, 4, 6},
        };
    }

    std::vector<Face> cube_faces()
    {
        const double h = 0.5;
        // Each face: right-handed (cross(eu, ev) = normal), one texture tile across the face.
        return {
            {{-h, -h, h}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 1, 1},     // +Z
            {{h, -h, -h}, {-1, 0, 0}, {0, 1, 0}, {0, 0, -1}, 1, 1},   // -Z
            {{h, -h, h}, {0, 0, -1}, {0, 1, 0}, {1, 0, 0}, 1, 1},     // +X
            {{-h, -h, -h}, {0, 0, 1}, {0, 1, 0}, {-1, 0, 0}, 1, 1},   // -X
            {{-h, h, h}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, 1, 1},     // +Y
            {{-h, -h, -h}, {1, 0, 0}, {0, 0, 1}, {0, -1, 0}, 1, 1},   // -Y
        };
    }

    // Interleaved vertex: position, normal, uv, tangent (xyz) + handedness (w).
    struct Vertex
    {
        float position[3];
        float normal[3];
        float uv[2];
        float tangent[4];
    };

    struct Mesh
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;
    };

    void add_face(Mesh& mesh, const Face& f)
    {
        const auto base = static_cast<std::uint16_t>(mesh.vertices.size());
        constexpr double corners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (const auto& c : corners)
        {
            const D3 p = f.origin + f.eu * c[0] + f.ev * c[1];
            Vertex v{};
            v.position[0] = static_cast<float>(p.x);
            v.position[1] = static_cast<float>(p.y);
            v.position[2] = static_cast<float>(p.z);
            v.normal[0] = static_cast<float>(f.normal.x);
            v.normal[1] = static_cast<float>(f.normal.y);
            v.normal[2] = static_cast<float>(f.normal.z);
            v.uv[0] = static_cast<float>(c[0] * f.tile_u);
            v.uv[1] = static_cast<float>(c[1] * f.tile_v);
            mesh.vertices.push_back(v);
        }
        // Counter-clockwise as seen from the normal side (the GL front face), also for mirrored faces.
        const bool ccw = dot(cross(f.eu, f.ev), f.normal) > 0.0;
        const std::uint16_t order[6] = {0, ccw ? std::uint16_t{1} : std::uint16_t{2}, ccw ? std::uint16_t{2} : std::uint16_t{1},
                                        0, ccw ? std::uint16_t{2} : std::uint16_t{3}, ccw ? std::uint16_t{3} : std::uint16_t{2}};
        for (const std::uint16_t o : order) mesh.indices.push_back(static_cast<std::uint16_t>(base + o));
    }

    // Unit sphere with u along longitude and v along latitude; (dP/du, dP/dv, N) is right-handed.
    Mesh make_sphere(double radius, int slices, int stacks, double tile_u, double tile_v)
    {
        Mesh m;
        for (int st = 0; st <= stacks; ++st)
        {
            const double lat = -std::numbers::pi / 2.0 + std::numbers::pi * st / stacks;
            for (int sl = 0; sl <= slices; ++sl)
            {
                const double lon = 2.0 * std::numbers::pi * sl / slices;
                const D3 n{std::cos(lat) * std::cos(lon), std::sin(lat), -std::cos(lat) * std::sin(lon)};
                Vertex v{};
                v.position[0] = static_cast<float>(n.x * radius);
                v.position[1] = static_cast<float>(n.y * radius);
                v.position[2] = static_cast<float>(n.z * radius);
                v.normal[0] = static_cast<float>(n.x);
                v.normal[1] = static_cast<float>(n.y);
                v.normal[2] = static_cast<float>(n.z);
                v.uv[0] = static_cast<float>(static_cast<double>(sl) / slices * tile_u);
                v.uv[1] = static_cast<float>(static_cast<double>(st) / stacks * tile_v);
                m.vertices.push_back(v);
            }
        }
        for (int st = 0; st < stacks; ++st)
            for (int sl = 0; sl < slices; ++sl)
            {
                const auto a = static_cast<std::uint16_t>(st * (slices + 1) + sl);
                const auto b = static_cast<std::uint16_t>(a + slices + 1);
                for (const std::uint16_t i : {a, static_cast<std::uint16_t>(a + 1), static_cast<std::uint16_t>(b + 1), a,
                                              static_cast<std::uint16_t>(b + 1), b})
                    m.indices.push_back(i);
            }
        return m;
    }

    // Per-vertex tangent frame from positions and UVs (E. Lengyel, "Computing Tangent Space Basis Vectors
    // for an Arbitrary Mesh"). For each triangle, with edges e1, e2 and UV differences (du, dv):
    //     dP/du = (e1 dv2 - e2 dv1) / det,   dP/dv = (e2 du1 - e1 du2) / det,   det = du1 dv2 - du2 dv1.
    // The per-vertex sums are orthogonalised against the vertex normal (Gram-Schmidt) and the
    // handedness is w = sign(dot(cross(N, T), dP/dv)): +1 when B = cross(N, T) already points along +v.
    void compute_tangents(Mesh& mesh)
    {
        const std::size_t n = mesh.vertices.size();
        std::vector<D3> tan_u(n), tan_v(n);
        auto pos = [&](std::size_t i) { return D3{double{mesh.vertices[i].position[0]}, double{mesh.vertices[i].position[1]}, double{mesh.vertices[i].position[2]}}; };
        auto uv = [&](std::size_t i) { return std::array<double, 2>{double{mesh.vertices[i].uv[0]}, double{mesh.vertices[i].uv[1]}}; };
        for (std::size_t k = 0; k + 2 < mesh.indices.size(); k += 3)
        {
            const std::size_t i0 = mesh.indices[k], i1 = mesh.indices[k + 1], i2 = mesh.indices[k + 2];
            const D3 e1 = pos(i1) - pos(i0), e2 = pos(i2) - pos(i0);
            const double du1 = uv(i1)[0] - uv(i0)[0], dv1 = uv(i1)[1] - uv(i0)[1];
            const double du2 = uv(i2)[0] - uv(i0)[0], dv2 = uv(i2)[1] - uv(i0)[1];
            const double det = du1 * dv2 - du2 * dv1;
            if (std::abs(det) < 1e-12) continue;  // degenerate UV triangle (sphere poles)
            const D3 sdir = (e1 * dv2 - e2 * dv1) * (1.0 / det);
            const D3 tdir = (e2 * du1 - e1 * du2) * (1.0 / det);
            for (const std::size_t i : {i0, i1, i2})
            {
                tan_u[i] = tan_u[i] + sdir;
                tan_v[i] = tan_v[i] + tdir;
            }
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const D3 nrm{double{mesh.vertices[i].normal[0]}, double{mesh.vertices[i].normal[1]}, double{mesh.vertices[i].normal[2]}};
            D3 t = tan_u[i] - nrm * dot(nrm, tan_u[i]);
            if (length(t) < 1e-12)  // no UV information: any vector perpendicular to the normal
                t = std::abs(nrm.x) < 0.9 ? cross(nrm, D3{1, 0, 0}) : cross(nrm, D3{0, 1, 0});
            t = normalize(t);
            const double w = dot(cross(nrm, t), tan_v[i]) < 0.0 ? -1.0 : 1.0;
            mesh.vertices[i].tangent[0] = static_cast<float>(t.x);
            mesh.vertices[i].tangent[1] = static_cast<float>(t.y);
            mesh.vertices[i].tangent[2] = static_cast<float>(t.z);
            mesh.vertices[i].tangent[3] = static_cast<float>(w);
        }
    }

    // ============================================================================================
    // Scene: camera, light, animation (all functions of the animation time)
    // ============================================================================================
    struct Camera
    {
        D3 eye, target, up{0, 1, 0};
        double fovy = 0.7;  // radians
    };

    dm::Mat4 view_proj_matrix(const Camera& c, double aspect)
    {
        const dm::Mat4 view = dm::look_at(to_vec3(c.eye), to_vec3(c.target), to_vec3(c.up));
        const dm::Mat4 proj = dm::perspective(static_cast<float>(c.fovy), static_cast<float>(aspect), 0.1f, 60.0f);
        return proj * view;
    }

    // World-space direction of the ray through the CENTRE of window pixel (px, py) (bottom-up rows).
    D3 pixel_ray(const Camera& c, int px, int py, int width, int height)
    {
        const D3 f = normalize(c.target - c.eye);
        const D3 s = normalize(cross(f, c.up));
        const D3 u = cross(s, f);
        const double ty = std::tan(c.fovy * 0.5), tx = ty * static_cast<double>(width) / static_cast<double>(height);
        const double nx = (px + 0.5) / width * 2.0 - 1.0, ny = (py + 0.5) / height * 2.0 - 1.0;
        return normalize(f + s * (nx * tx) + u * (ny * ty));
    }

    constexpr double kCameraDistance = 6.3;
    constexpr double kCameraElevation = 0.14;
    constexpr double kCameraYaw0 = 0.20;

    // The orbit camera's 38 degree field of view spans the SHORTER side of the surface: vertical on a landscape
    // surface, horizontal on a portrait one, so that the wall, the crate and the sphere (side by side along x) stay
    // in view at any aspect ratio. Camera::fovy is always the vertical angle that the projection and the CPU rays use.
    Camera orbit_camera(double yaw, double aspect)
    {
        Camera c;
        c.target = {0.0, -0.15, 0.4};
        c.eye = c.target + D3{std::sin(yaw) * std::cos(kCameraElevation), std::sin(kCameraElevation), std::cos(yaw) * std::cos(kCameraElevation)} *
                               kCameraDistance;
        c.fovy = 2.0 * std::atan(std::tan(0.5 * 38.0 * std::numbers::pi / 180.0) / std::min(aspect, 1.0));
        return c;
    }

    D3 light_position(double t)
    {
        const double a = 0.9 + 0.8 * t;
        return {2.3 * std::cos(a), 0.30 + 1.05 * std::sin(a), 2.20 + 0.30 * std::sin(1.7 * a)};
    }
    const D3 kCubeCenter{-1.1, -0.45, 0.75};
    const D3 kSphereCenter{1.15, -0.35, 0.85};
    constexpr double kSphereRadius = 0.62;

    dm::Mat4 cube_model(double t)
    {
        return dm::translate(to_vec3(kCubeCenter)) * dm::rotate(static_cast<float>(0.55 + 0.7 * t), {0.35f, 1.0f, 0.15f});
    }
    dm::Mat4 sphere_model(double t)
    {
        return dm::translate(to_vec3(kSphereCenter)) * dm::rotate(static_cast<float>(0.5 + 0.35 * t), {0.0f, 1.0f, 0.0f});
    }

    struct LightRig  // the uniforms of the lighting model, shared by the GPU draw and the CPU reference
    {
        D3 position;
        D3 color{1.0, 0.93, 0.82};
        double ambient = 0.22;
        double shininess = 48.0;
    };

    enum class Filter { Trilinear, Bilinear, Nearest };

    // Everything draw_scene() needs. The windowed frame and every validation pass fill one of these.
    struct Pass
    {
        Camera camera;
        LightRig light;
        int view = 0;  // 0 lit, 1 albedo, 2 normal map, 3 specular map
        bool normal_map = true, spec_map = true, handedness = true;
        Filter filter = Filter::Trilinear;
        bool wall = true, cube = true, sphere = true, marker = true;
        dm::Mat4 cube_model = dm::Mat4::identity();
        dm::Mat4 sphere_model = dm::Mat4::identity();
    };

    // ============================================================================================
    // CPU reference of the lit shader at one point of a face (NEAREST filtering, mip level 0)
    // ============================================================================================
    struct ShadeMode
    {
        bool normal_map = true, spec_map = true, handedness = true;
        bool flip_green = false;  // negative control: DirectX-style normal map (+Y = -v)
    };

    struct TexelCoord
    {
        int i, j;
        double frac_u, frac_v;  // position inside the texel, 0..1
    };

    TexelCoord texel_at(const Face& f, double s, double t)
    {
        const double u = s * f.tile_u * kTexSize, v = t * f.tile_v * kTexSize;  // texel coordinates (ES 3.0 eq. 3.19)
        const double fu = std::floor(u), fv = std::floor(v);
        return {wrap_repeat(static_cast<int>(fu), kTexSize), wrap_repeat(static_cast<int>(fv), kTexSize), u - fu, v - fv};
    }

    std::array<double, 3> shade_reference(const TextureSet& tex, const Face& f, double s, double t, const ShadeMode& mode,
                                          const LightRig& light, D3 eye, int view)
    {
        const TexelCoord tc = texel_at(f, s, t);
        const D3 albedo{tex.albedo.texel(0, tc.i, tc.j, 0) / 255.0, tex.albedo.texel(0, tc.i, tc.j, 1) / 255.0, tex.albedo.texel(0, tc.i, tc.j, 2) / 255.0};
        D3 n_ts{tex.normal.texel(0, tc.i, tc.j, 0) / 255.0 * 2.0 - 1.0, tex.normal.texel(0, tc.i, tc.j, 1) / 255.0 * 2.0 - 1.0,
                tex.normal.texel(0, tc.i, tc.j, 2) / 255.0 * 2.0 - 1.0};
        const double spec_texel = tex.specular.texel(0, tc.i, tc.j, 0) / 255.0;
        if (view == 1) return {albedo.x * 255.0, albedo.y * 255.0, albedo.z * 255.0};
        if (!mode.normal_map) n_ts = {0, 0, 1};
        if (mode.flip_green) n_ts.y = -n_ts.y;
        const double spec_mask = mode.spec_map ? spec_texel : 1.0;

        const D3 tangent = normalize(f.eu);  // dP/du direction (tile_u > 0)
        const D3 n_geo = normalize(f.normal);
        const double w = dot(cross(n_geo, tangent), normalize(f.ev)) < 0.0 ? -1.0 : 1.0;  // handedness of the UV basis
        const D3 bitangent = cross(n_geo, tangent) * (mode.handedness ? w : 1.0);
        const D3 n = normalize(tangent * n_ts.x + bitangent * n_ts.y + n_geo * n_ts.z);

        const D3 p = f.origin + f.eu * s + f.ev * t;
        const D3 to_light = light.position - p;
        const double dist = length(to_light);
        const D3 l = to_light * (1.0 / dist);
        const D3 v = normalize(eye - p);
        const D3 h = normalize(l + v);
        const double attenuation = 1.0 / (1.0 + 0.12 * dist + 0.04 * dist * dist);
        const double facing = smoothstep(0.0, 0.2, dot(n_geo, l));
        const double n_dot_l = std::max(dot(n, l), 0.0);
        double spec = n_dot_l > 0.0 ? std::pow(std::max(dot(n, h), 0.0), light.shininess) : 0.0;
        const double diffuse = n_dot_l * facing;
        spec *= facing;
        auto channel = [&](double a, double lc) { return std::clamp(a * (light.ambient + lc * diffuse * attenuation) + lc * spec * spec_mask * attenuation, 0.0, 1.0) * 255.0; };
        return {channel(albedo.x, light.color.x), channel(albedo.y, light.color.y), channel(albedo.z, light.color.z)};
    }

    // ============================================================================================
    // Probes: window pixels whose surface point is known analytically
    // ============================================================================================
    struct Probe
    {
        int px, py;
        int face;
        double s, t;
    };

    // Ray (eye, dir) against a face: distance along the ray and (s, t). Back faces are skipped, as the GL culls them.
    std::optional<std::array<double, 3>> intersect(const Face& f, D3 eye, D3 dir)
    {
        const double denom = dot(dir, f.normal);
        if (denom >= 0.0) return std::nullopt;
        const double dist = dot(f.origin - eye, f.normal) / denom;
        if (dist <= 0.0) return std::nullopt;
        const D3 rel = eye + dir * dist - f.origin;
        return std::array<double, 3>{dist, dot(rel, f.eu) / dot(f.eu, f.eu), dot(rel, f.ev) / dot(f.ev, f.ev)};
    }

    // Visible face point of every `stride`-th pixel: nearest front-facing face, at least `margin` world units
    // from its borders (primitive edges are not probed). With `texel_centred`, only points at least 0.2 texel
    // from a texel border are kept (NEAREST selection is then insensitive to sub-texel position errors).
    std::vector<Probe> gather_probes(const std::vector<Face>& faces, const Camera& cam, int width, int height, int stride,
                                     double margin, bool texel_centred)
    {
        std::vector<Probe> out;
        for (int py = stride / 2; py < height; py += stride)
            for (int px = stride / 2; px < width; px += stride)
            {
                const D3 dir = pixel_ray(cam, px, py, width, height);
                double best = 1e300;
                int best_face = -1;
                double bs = 0, bt = 0;
                for (std::size_t k = 0; k < faces.size(); ++k)
                {
                    const auto hit = intersect(faces[k], cam.eye, dir);
                    if (!hit || (*hit)[1] < 0.0 || (*hit)[1] > 1.0 || (*hit)[2] < 0.0 || (*hit)[2] > 1.0 || (*hit)[0] >= best) continue;
                    best = (*hit)[0];
                    best_face = static_cast<int>(k);
                    bs = (*hit)[1];
                    bt = (*hit)[2];
                }
                if (best_face < 0) continue;
                const Face& f = faces[static_cast<std::size_t>(best_face)];
                const double ms = margin / length(f.eu), mt = margin / length(f.ev);
                if (bs < ms || bs > 1.0 - ms || bt < mt || bt > 1.0 - mt) continue;
                if (texel_centred)
                {
                    const TexelCoord tc = texel_at(f, bs, bt);
                    if (tc.frac_u < 0.2 || tc.frac_u > 0.8 || tc.frac_v < 0.2 || tc.frac_v > 0.8) continue;
                }
                out.push_back({px, py, best_face, bs, bt});
            }
        return out;
    }

    using Rgb = std::array<double, 3>;

    struct Comparison
    {
        std::size_t count = 0, bad = 0;
        double max_error = 0;
        std::string first_bad;
    };

    std::array<int, 3> pixel_at(const std::vector<std::uint8_t>& img, int width, int px, int py)
    {
        const auto o = static_cast<std::size_t>((py * width + px) * 4);
        return {img[o], img[o + 1], img[o + 2]};
    }

    template <typename Ref>
    Comparison compare_probes(const std::vector<std::uint8_t>& img, int width, const std::vector<Probe>& probes, double tolerance, Ref&& ref)
    {
        Comparison r;
        for (const Probe& p : probes)
        {
            const Rgb e = ref(p);
            const auto g = pixel_at(img, width, p.px, p.py);
            double err = 0;
            for (std::size_t c = 0; c < 3; ++c) err = std::max(err, std::abs(g[c] - e[c]));
            ++r.count;
            r.max_error = std::max(r.max_error, err);
            if (err > tolerance && r.bad++ == 0)
                r.first_bad = std::format("first mismatch at pixel ({},{}): got ({},{},{}) expected ({:.1f},{:.1f},{:.1f})", p.px, p.py, g[0], g[1],
                                          g[2], e[0], e[1], e[2]);
        }
        return r;
    }

    // Number of probes at which the image differs from a model by more than `tolerance`.
    template <typename Ref>
    std::size_t count_mismatches(const std::vector<std::uint8_t>& img, int width, const std::vector<Probe>& probes, double tolerance, Ref&& ref)
    {
        return compare_probes(img, width, probes, tolerance, ref).bad;
    }

    // ============================================================================================
    // The demo
    // ============================================================================================
    constexpr std::array<mgl::TextureUnit, 3> kUnits = {mgl::TextureUnit::Texture0, mgl::TextureUnit::Texture1, mgl::TextureUnit::Texture2};
    // Lit colours are stored as 8-bit unorm. ES 3.0 section 4.1.9: the stored value is one of the two representable values
    // next to the incoming colour (floor or ceil), so the error of the stored value is < 1/255 whatever the rounding or
    // dithering. The shader's own float error (GLSL ES 3.00 section 4.5.1: +,-,* correctly rounded, 2 ULP inversesqrt, 3+2|x| ULP exp2,
    // < 2^-21 absolute log2; the largest term is pow(n.h, 48) with about 2e-5 relative error) and the CPU model's rounding
    // add well under 0.1/255. 1.25 therefore leaves a quarter LSB of headroom; the measured maximum is printed with every check.
    constexpr double kTolerance = 1.25;
    constexpr int kSlack = 3;  // ceil(2 * kTolerance): the difference of two renders is known to within two tolerances

    struct GpuMesh
    {
        mgl::VertexArrayId vao{};
        mgl::BufferId vbo{};
        mgl::BufferId ibo{};
        GLsizei index_count = 0;
    };

    struct LitUniforms
    {
        mgl::UniformLocation model, normal_matrix, view_proj, use_handedness;
        mgl::UniformLocation albedo, normal, specular;
        mgl::UniformLocation light_pos, light_color, eye, ambient, shininess, use_normal_map, use_spec_map, view;
    };

    class NormalMapping final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            lit_ = demo::build_program(env, "shaders/lit.vert", "shaders/lit.frag");
            marker_ = demo::build_program(env, "shaders/marker.vert", "shaders/marker.frag");
            u_.model = mgl::glGetUniformLocation(lit_, "u_model");
            u_.normal_matrix = mgl::glGetUniformLocation(lit_, "u_normal_matrix");
            u_.view_proj = mgl::glGetUniformLocation(lit_, "u_view_proj");
            u_.use_handedness = mgl::glGetUniformLocation(lit_, "u_use_handedness");
            u_.albedo = mgl::glGetUniformLocation(lit_, "u_albedo");
            u_.normal = mgl::glGetUniformLocation(lit_, "u_normal");
            u_.specular = mgl::glGetUniformLocation(lit_, "u_specular");
            u_.light_pos = mgl::glGetUniformLocation(lit_, "u_light_pos");
            u_.light_color = mgl::glGetUniformLocation(lit_, "u_light_color");
            u_.eye = mgl::glGetUniformLocation(lit_, "u_eye");
            u_.ambient = mgl::glGetUniformLocation(lit_, "u_ambient");
            u_.shininess = mgl::glGetUniformLocation(lit_, "u_shininess");
            u_.use_normal_map = mgl::glGetUniformLocation(lit_, "u_use_normal_map");
            u_.use_spec_map = mgl::glGetUniformLocation(lit_, "u_use_spec_map");
            u_.view = mgl::glGetUniformLocation(lit_, "u_view");
            marker_mvp_ = mgl::glGetUniformLocation(marker_, "u_mvp");
            marker_color_ = mgl::glGetUniformLocation(marker_, "u_color");

            // --- textures: one object per map, a full mip chain uploaded level by level -------------
            textures_cpu_ = make_textures();
            mgl::glGenTextures(3, tex_.data());
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 1);  // R8 rows of 1, 2, 4 bytes are not 4-byte padded
            upload_chain(0, textures_cpu_.albedo, mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba);
            upload_chain(1, textures_cpu_.normal, mgl::InternalFormat::Rgba8, mgl::PixelFormat::Rgba);
            upload_chain(2, textures_cpu_.specular, mgl::InternalFormat::R8, mgl::PixelFormat::Red);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
            // Texture unit k holds map k for the whole run; the sampler uniforms say which unit each sampler reads.
            mgl::glUseProgram(lit_);
            mgl::glUniform1i(u_.albedo, 0);
            mgl::glUniform1i(u_.normal, 1);
            mgl::glUniform1i(u_.specular, 2);
            apply_filter(Filter::Trilinear);

            // --- meshes with CPU-generated tangents ------------------------------------------------
            for (const Face& f : wall_faces()) add_face(wall_cpu_, f);
            for (const Face& f : cube_faces()) add_face(cube_cpu_, f);
            sphere_cpu_ = make_sphere(kSphereRadius, 48, 24, 3.0, 1.5);
            compute_tangents(wall_cpu_);
            compute_tangents(cube_cpu_);
            compute_tangents(sphere_cpu_);
            wall_ = upload_mesh(wall_cpu_);
            cube_ = upload_mesh(cube_cpu_);
            sphere_ = upload_mesh(sphere_cpu_);
            marker_mesh_ = make_sphere(1.0, 16, 8, 1.0, 1.0);  // positions only are used by the marker program
            marker_gpu_ = upload_mesh(marker_mesh_);
        }

        void render(demo::Env& env, int frame) override
        {
            if (frame != last_frame_)
            {
                if (!paused_ && last_frame_ >= 0) ++ticks_;
                last_frame_ = frame;
            }
            draw_frame(env);
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
            case 'n': normal_map_ = !normal_map_; break;
            case 's': spec_map_ = !spec_map_; break;
            case 'h': handedness_ = !handedness_; break;
            case 'f': filter_ = static_cast<Filter>((static_cast<int>(filter_) + 1) % 3); break;
            case 'v': view_ = (view_ + 1) % 4; break;
            case 'p': paused_ = !paused_; break;
            case 'a': yaw_ -= 0.1; break;
            case 'd': yaw_ += 0.1; break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            const int w = env.width, h = env.height;
            const double time = static_cast<double>(ticks_) / 60.0;
            const Camera cam = orbit_camera(yaw_, static_cast<double>(w) / static_cast<double>(h));

            check_state(check);
            check_tangents(check);

            // A light placed so that the mirror reflection of the eye lies on the wall at (0, 0.3, 0): the Blinn
            // highlight is then centred on the seam between the two panels and spans both of them.
            const double light_z = 1.9;
            const D3 mirror_point{0.0, 0.3, 0.0};
            LightRig val_light;
            val_light.position = {(mirror_point.x * (cam.eye.z + light_z) - cam.eye.x * light_z) / cam.eye.z,
                                  (mirror_point.y * (cam.eye.z + light_z) - cam.eye.y * light_z) / cam.eye.z, light_z};

            check_wall(check, w, h, cam, val_light, time);
            check_cube(check, w, h, cam, val_light, time);
            check_trilinear(check, w, h);
            check.expect_no_gl_error("no GL error after the validation passes");

            draw_frame(env);  // leave the normal scene in the framebuffer: the runtime captures it after validate()
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GpuMesh* m : {&wall_, &cube_, &sphere_, &marker_gpu_})
            {
                mgl::glDeleteVertexArrays(1, &m->vao);
                mgl::glDeleteBuffers(1, &m->vbo);
                mgl::glDeleteBuffers(1, &m->ibo);
            }
            mgl::glDeleteTextures(3, tex_.data());
            mgl::glDeleteProgram(lit_);
            mgl::glDeleteProgram(marker_);
        }

    private:
        // ---- GL helpers -------------------------------------------------------------------------
        void upload_chain(std::size_t unit, const CpuTexture& cpu, mgl::InternalFormat internal, mgl::PixelFormat format)
        {
            mgl::glActiveTexture(kUnits[unit]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[unit]);
            for (int level = 0; level < kLevels; ++level)
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, internal, level_size(level), level_size(level), 0, format,
                                  mgl::PixelType::UnsignedByte, cpu.levels[static_cast<std::size_t>(level)].data());
        }

        void apply_filter(Filter filter)
        {
            const mgl::TextureMinFilter min = filter == Filter::Trilinear ? mgl::TextureMinFilter::LinearMipmapLinear
                                              : filter == Filter::Bilinear ? mgl::TextureMinFilter::Linear
                                                                           : mgl::TextureMinFilter::Nearest;
            const mgl::TextureMagFilter mag = filter == Filter::Nearest ? mgl::TextureMagFilter::Nearest : mgl::TextureMagFilter::Linear;
            for (std::size_t unit = 0; unit < 3; ++unit)
            {
                mgl::glActiveTexture(kUnits[unit]);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[unit]);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, min);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mag);
            }
        }

        static GpuMesh upload_mesh(const Mesh& mesh)
        {
            GpuMesh g;
            g.index_count = static_cast<GLsizei>(mesh.indices.size());
            mgl::glGenVertexArrays(1, &g.vao);
            mgl::glGenBuffers(1, &g.vbo);
            mgl::glGenBuffers(1, &g.ibo);
            mgl::glBindVertexArray(g.vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, g.vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(mesh.vertices), mgl::BufferUsage::StaticDraw);
            // location 0 position, 1 normal, 2 uv, 3 tangent (xyz) + handedness (w), interleaved
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, position)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, normal)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, uv)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{3u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{3u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, tangent)));
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, g.ibo);  // recorded in the VAO
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(mesh.indices), mgl::BufferUsage::StaticDraw);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            return g;
        }

        void draw_mesh(const GpuMesh& m) const
        {
            mgl::glBindVertexArray(m.vao);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, m.index_count, mgl::IndexType::UnsignedShort, nullptr);
        }

        void set_model(const dm::Mat4& model) const
        {
            mgl::glUniformMatrix4fv(u_.model, 1, GL_FALSE, model.data());
            const std::array<float, 9> normal_matrix = dm::normal_matrix(model);
            mgl::glUniformMatrix3fv(u_.normal_matrix, 1, GL_FALSE, normal_matrix.data());
        }

        void draw_scene(const Pass& pass, int width, int height)
        {
            const double aspect = static_cast<double>(width) / static_cast<double>(height);
            const dm::Mat4 view_proj = view_proj_matrix(pass.camera, aspect);

            mgl::glViewport(0, 0, width, height);
            mgl::glClearColor(0.035f, 0.04f, 0.055f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::CullFace);
            mgl::glCullFace(mgl::CullFace::Back);

            apply_filter(pass.filter);

            mgl::glUseProgram(lit_);
            mgl::glUniformMatrix4fv(u_.view_proj, 1, GL_FALSE, view_proj.data());
            mgl::glUniform3f(u_.light_pos, static_cast<float>(pass.light.position.x), static_cast<float>(pass.light.position.y),
                             static_cast<float>(pass.light.position.z));
            mgl::glUniform3f(u_.light_color, static_cast<float>(pass.light.color.x), static_cast<float>(pass.light.color.y),
                             static_cast<float>(pass.light.color.z));
            mgl::glUniform3f(u_.eye, static_cast<float>(pass.camera.eye.x), static_cast<float>(pass.camera.eye.y), static_cast<float>(pass.camera.eye.z));
            mgl::glUniform1f(u_.ambient, static_cast<float>(pass.light.ambient));
            mgl::glUniform1f(u_.shininess, static_cast<float>(pass.light.shininess));
            mgl::glUniform1i(u_.use_normal_map, pass.normal_map ? 1 : 0);
            mgl::glUniform1i(u_.use_spec_map, pass.spec_map ? 1 : 0);
            mgl::glUniform1i(u_.use_handedness, pass.handedness ? 1 : 0);
            mgl::glUniform1i(u_.view, pass.view);

            if (pass.wall)
            {
                set_model(dm::Mat4::identity());
                draw_mesh(wall_);
            }
            if (pass.cube)
            {
                set_model(pass.cube_model);
                draw_mesh(cube_);
            }
            if (pass.sphere)
            {
                set_model(pass.sphere_model);
                draw_mesh(sphere_);
            }
            if (pass.marker)
            {
                mgl::glUseProgram(marker_);
                const dm::Mat4 mvp = view_proj * dm::translate(to_vec3(pass.light.position)) * dm::scale({0.07f, 0.07f, 0.07f});
                mgl::glUniformMatrix4fv(marker_mvp_, 1, GL_FALSE, mvp.data());
                mgl::glUniform3f(marker_color_, 1.0f, 0.96f, 0.80f);
                draw_mesh(marker_gpu_);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void draw_frame(const demo::Env& env)
        {
            const double time = static_cast<double>(ticks_) / 60.0;
            Pass pass;
            pass.camera = orbit_camera(yaw_, static_cast<double>(env.width) / static_cast<double>(env.height));
            pass.light.position = light_position(time);
            pass.view = view_;
            pass.normal_map = normal_map_;
            pass.spec_map = spec_map_;
            pass.handedness = handedness_;
            pass.filter = filter_;
            pass.cube_model = cube_model(time);
            pass.sphere_model = sphere_model(time);
            draw_scene(pass, env.width, env.height);
        }

        // ---- validation -------------------------------------------------------------------------
        void check_state(demo::Check& check)
        {
            // Texture units: unit k has texture k bound, and the sampler uniforms select the units.
            bool bound = true;
            std::string seen;
            for (std::size_t unit = 0; unit < 3; ++unit)
            {
                mgl::glActiveTexture(kUnits[unit]);
                const int b = demo::get_integer(mgl::GetParameter::TextureBinding2D);
                seen += std::format("unit{}={} ", unit, b);
                bound = bound && b == static_cast<int>(tex_[unit].value);
            }
            check.expect(bound, "TEXTURE_BINDING_2D of units 0, 1, 2 are the albedo, normal and specular textures", seen);
            GLint su[3] = {-1, -1, -1};
            mgl::glGetUniformiv(lit_, u_.albedo, &su[0]);
            mgl::glGetUniformiv(lit_, u_.normal, &su[1]);
            mgl::glGetUniformiv(lit_, u_.specular, &su[2]);
            check.expect(su[0] == 0 && su[1] == 1 && su[2] == 2, "sampler uniforms u_albedo/u_normal/u_specular hold the units 0/1/2",
                         std::format("read {},{},{}", su[0], su[1], su[2]));

            // The fragment shader reads three units during one draw (ES 3.0 section 3.9.2: limited by MAX_TEXTURE_IMAGE_UNITS;
            // the vertex + fragment total by MAX_COMBINED_TEXTURE_IMAGE_UNITS; Tables 6.32 and 6.33 give the minima 16 and 32).
            const int fragment_units = demo::get_integer(mgl::GetParameter::MaxTextureImageUnits);
            const int combined_units = demo::get_integer(mgl::GetParameter::MaxCombinedTextureImageUnits);
            check.expect(fragment_units >= 16 && combined_units >= 32 && fragment_units >= 3,
                         "MAX_TEXTURE_IMAGE_UNITS >= 16 and MAX_COMBINED_TEXTURE_IMAGE_UNITS >= 32 (ES 3.0 minima), enough for the 3 maps",
                         std::format("MAX_TEXTURE_IMAGE_UNITS={} MAX_COMBINED_TEXTURE_IMAGE_UNITS={}", fragment_units, combined_units));

            check_levels(check);
        }

        // Every level of every map, read back through a framebuffer object (ES 3.0 allows attaching any level of a
        // 2D texture, section 4.4.2.4) and compared byte for byte with the CPU chain. This proves the sizes
        // (a level of the wrong size would not match, and an inconsistent chain makes the texture incomplete,
        // which the trilinear pass below would show as black), the formats and the data of all 24 uploads.
        void check_levels(demo::Check& check)
        {
            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            const CpuTexture* cpu[3] = {&textures_cpu_.albedo, &textures_cpu_.normal, &textures_cpu_.specular};
            const char* names[3] = {"albedo (RGBA8)", "normal (RGBA8)", "specular (R8)"};
            for (std::size_t k = 0; k < 3; ++k)
            {
                std::size_t wrong_texels = 0;
                int incomplete = 0;
                std::string first_bad;
                for (int level = 0; level < kLevels; ++level)
                {
                    mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                                mgl::TextureTarget::Texture2D, tex_[k], level);
                    if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                    {
                        ++incomplete;
                        continue;
                    }
                    const int n = level_size(level);
                    const auto pixels = demo::read_rgba8(0, 0, n, n);
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i)
                        {
                            const auto o = static_cast<std::size_t>((j * n + i) * 4);
                            const int channels = cpu[k]->channels;
                            bool ok = true;
                            for (int c = 0; c < channels; ++c) ok = ok && pixels[o + static_cast<std::size_t>(c)] == cpu[k]->texel(level, i, j, c);
                            if (channels == 1) ok = ok && pixels[o + 1] == 0 && pixels[o + 2] == 0 && pixels[o + 3] == 255;  // R8 reads as (r, 0, 0, 1)
                            if (!ok && wrong_texels++ == 0) first_bad = std::format("first mismatch: level {} texel ({},{})", level, i, j);
                        }
                }
                check.expect(incomplete == 0 && wrong_texels == 0, std::format("{} texture: all {} mip levels (128 down to 1x1) read back identical to the CPU chain", names[k], kLevels),
                             first_bad);
            }
            // The sized internal formats show up as component sizes of the attachment (ES 3.0 section 6.1.13).
            auto sizes = [&](std::size_t k) {
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2D, tex_[k], 0);
                std::array<GLint, 4> r{};
                const mgl::FramebufferAttachmentParameter q[4] = {mgl::FramebufferAttachmentParameter::RedSize, mgl::FramebufferAttachmentParameter::GreenSize,
                                                                  mgl::FramebufferAttachmentParameter::BlueSize, mgl::FramebufferAttachmentParameter::AlphaSize};
                for (std::size_t c = 0; c < 4; ++c)
                    mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0), q[c], &r[c]);
                return r;
            };
            const auto s_normal = sizes(1), s_spec = sizes(2);
            check.expect(s_normal == std::array<GLint, 4>{8, 8, 8, 8} && s_spec[0] == 8 && s_spec[1] == 0 && s_spec[2] == 0 && s_spec[3] == 0,
                         "the normal map is RGBA8 (8/8/8/8 bits) and the specular map R8 (8/0/0/0 bits)",
                         std::format("normal {},{},{},{} specular {},{},{},{}", s_normal[0], s_normal[1], s_normal[2], s_normal[3], s_spec[0], s_spec[1], s_spec[2], s_spec[3]));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
        }

        void check_tangents(demo::Check& check) const
        {
            // The generated tangents of the wall and the crate against the analytic frames: T = normalize(eu),
            // w = +1 for right-handed UV basis, -1 for the mirrored right wall panel.
            auto verify = [&](const Mesh& mesh, const std::vector<Face>& faces, const char* name, std::vector<double> expected_w) {
                double worst_angle = 0;
                bool signs = true;
                for (std::size_t f = 0; f < faces.size(); ++f)
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        const Vertex& v = mesh.vertices[f * 4 + k];
                        const D3 t{double{v.tangent[0]}, double{v.tangent[1]}, double{v.tangent[2]}};
                        const double angle = std::acos(std::clamp(dot(t, normalize(faces[f].eu)), -1.0, 1.0));
                        worst_angle = std::max(worst_angle, angle);
                        signs = signs && static_cast<double>(v.tangent[3]) == expected_w[f];
                    }
                check.expect(worst_angle < 1e-6 && signs, std::format("CPU tangents of the {} equal dP/du and carry the expected handedness", name),
                             std::format("max angle error {:.2e} rad", worst_angle));
            };
            verify(wall_cpu_, wall_faces(), "wall (panel 2 has a mirrored u axis: w = -1)", {1.0, -1.0});
            verify(cube_cpu_, cube_faces(), "crate faces (all w = +1)", {1, 1, 1, 1, 1, 1});

            double worst_ortho = 0, worst_len = 0;
            bool w_ok = true;
            for (std::size_t i = 0; i < sphere_cpu_.vertices.size(); ++i)
            {
                const Vertex& v = sphere_cpu_.vertices[i];
                const D3 t{double{v.tangent[0]}, double{v.tangent[1]}, double{v.tangent[2]}};
                const D3 n{double{v.normal[0]}, double{v.normal[1]}, double{v.normal[2]}};
                worst_ortho = std::max(worst_ortho, std::abs(dot(t, n)));
                worst_len = std::max(worst_len, std::abs(length(t) - 1.0));
                const std::size_t stack = i / 49;
                if (stack != 0 && stack != 24) w_ok = w_ok && v.tangent[3] == 1.0f;
            }
            check.expect(worst_ortho < 1e-6 && worst_len < 1e-6 && w_ok,
                         "sphere tangents are unit length, perpendicular to the normal, and w = +1 away from the poles",
                         std::format("max |T.N| {:.1e}, max ||T|-1| {:.1e}", worst_ortho, worst_len));
        }

        std::vector<std::uint8_t> render_pass(const Pass& pass, int w, int h)
        {
            draw_scene(pass, w, h);
            return demo::read_rgba8(0, 0, w, h);
        }

        void check_wall(demo::Check& check, int w, int h, const Camera& cam, const LightRig& val_light, double time)
        {
            const std::vector<Face> faces = wall_faces();
            const std::vector<Probe> probes = gather_probes(faces, cam, w, h, 3, 0.12, true);
            check.expect(probes.size() > 200, std::format("{} wall probes (pixels at texel-centred points, away from the panel borders and the seam)", probes.size()));
            std::size_t on_b = 0;
            for (const Probe& p : probes) on_b += p.face == 1 ? 1u : 0u;
            check.expect(on_b > 50 && probes.size() - on_b > 50, std::format("probes on both panels: {} on the normal panel, {} on the mirrored panel", probes.size() - on_b, on_b));

            Pass base;
            base.camera = cam;
            base.light = val_light;
            base.filter = Filter::Nearest;
            base.cube = base.sphere = base.marker = false;
            const Pass p_full = base;
            Pass p_flat = base;
            p_flat.normal_map = false;
            Pass p_nospec = base;
            p_nospec.spec_map = false;
            Pass p_nohand = base;
            p_nohand.handedness = false;
            const auto img_full = render_pass(p_full, w, h);
            const auto img_flat = render_pass(p_flat, w, h);
            const auto img_nospec = render_pass(p_nospec, w, h);
            const auto img_nohand = render_pass(p_nohand, w, h);

            auto model = [&](ShadeMode mode, const LightRig& light, D3 eye) {
                return [&, mode, eye](const Probe& p) {
                    return shade_reference(textures_cpu_, faces[static_cast<std::size_t>(p.face)], p.s, p.t, mode, light, eye, 0);
                };
            };
            const ShadeMode m_full{}, m_flat{false, true, true, false}, m_nospec{true, false, true, false}, m_nohand{true, true, false, false};
            const ShadeMode m_green{true, true, true, true};

            auto report = [&](const char* what, const Comparison& c) {
                check.expect(c.bad == 0 && c.count > 0, std::format("{} ({} probes, max error {:.2f}/255, tolerance {})", what, c.count, c.max_error, kTolerance), c.first_bad);
            };
            report("wall, normal map + specular map + handedness: GPU lit colours match the CPU model",
                   compare_probes(img_full, w, probes, kTolerance, model(m_full, val_light, cam.eye)));
            report("wall, normal map OFF (geometric normal): GPU matches the flat-normal CPU model",
                   compare_probes(img_flat, w, probes, kTolerance, model(m_flat, val_light, cam.eye)));
            report("wall, specular map OFF (constant strength): GPU matches the CPU model",
                   compare_probes(img_nospec, w, probes, kTolerance, model(m_nospec, val_light, cam.eye)));
            report("wall, tangent handedness IGNORED (B = cross(N, T)): GPU matches the CPU model of the same mistake",
                   compare_probes(img_nohand, w, probes, kTolerance, model(m_nohand, val_light, cam.eye)));

            // The animated light of the frame being validated (same equations, different light position).
            {
                Pass p_anim = base;
                p_anim.light.position = light_position(time);
                const auto img = render_pass(p_anim, w, h);
                report("wall with the ANIMATED light position of the validated frame: GPU matches the CPU model",
                       compare_probes(img, w, probes, kTolerance, model(m_full, p_anim.light, cam.eye)));
            }

            // --- the normal map changes the shading at the pixels where its texel is not (0, 0, 1) ---------
            std::size_t sensitive = 0, differ = 0;
            for (const Probe& p : probes)
            {
                const Rgb a = model(m_full, val_light, cam.eye)(p), b = model(m_flat, val_light, cam.eye)(p);
                double predicted = 0;
                for (std::size_t c = 0; c < 3; ++c) predicted = std::max(predicted, std::abs(a[c] - b[c]));
                if (predicted < 12.0) continue;
                ++sensitive;
                const auto g1 = pixel_at(img_full, w, p.px, p.py), g2 = pixel_at(img_flat, w, p.px, p.py);
                int measured = 0;
                for (std::size_t c = 0; c < 3; ++c) measured = std::max(measured, std::abs(g1[c] - g2[c]));
                if (measured >= 12 - kSlack) ++differ;
            }
            check.expect(sensitive > 50 && differ == sensitive,
                         std::format("the normal map changes the shading: at {} probes the CPU predicts a difference >= 12/255 from the flat-normal mode, "
                                     "the GPU shows it at {}",
                                     sensitive, differ));

            // --- the specular map masks highlights --------------------------------------------------
            std::size_t mortar_probes = 0, masked = 0;
            for (const Probe& p : probes)
            {
                const TexelCoord tc = texel_at(faces[static_cast<std::size_t>(p.face)], p.s, p.t);
                const int spec_texel = textures_cpu_.specular.texel(0, tc.i, tc.j, 0);
                if (spec_texel > 12) continue;  // matte texels only (mortar)
                const Rgb with_map = model(m_full, val_light, cam.eye)(p), without = model(m_nospec, val_light, cam.eye)(p);
                double unmasked_highlight = 0;
                for (std::size_t c = 0; c < 3; ++c) unmasked_highlight = std::max(unmasked_highlight, without[c] - with_map[c]);
                if (unmasked_highlight < 20.0) continue;  // only where the unmasked render has a clear highlight
                ++mortar_probes;
                const auto g_map = pixel_at(img_full, w, p.px, p.py), g_nomap = pixel_at(img_nospec, w, p.px, p.py);
                int gpu_drop = 0;
                for (std::size_t c = 0; c < 3; ++c) gpu_drop = std::max(gpu_drop, g_nomap[c] - g_map[c]);
                if (gpu_drop >= 20 - kSlack) ++masked;
            }
            check.expect(mortar_probes >= 5 && masked == mortar_probes,
                         std::format("the specular map masks highlights: at {} matte-mortar probes the unmasked render has a highlight of >= 20/255, the "
                                     "specular map removes it on the GPU at {}",
                                     mortar_probes, masked));

            // --- handedness: only the mirrored panel depends on it ---------------------------------------
            std::size_t sens_b = 0, differ_b = 0, same_a = 0, count_a = 0;
            for (const Probe& p : probes)
            {
                const auto g1 = pixel_at(img_full, w, p.px, p.py), g2 = pixel_at(img_nohand, w, p.px, p.py);
                int measured = 0;
                for (std::size_t c = 0; c < 3; ++c) measured = std::max(measured, std::abs(g1[c] - g2[c]));
                if (p.face == 0)
                {
                    ++count_a;
                    if (measured <= kSlack) ++same_a;
                }
                else
                {
                    const Rgb a = model(m_full, val_light, cam.eye)(p), b = model(m_nohand, val_light, cam.eye)(p);
                    double predicted = 0;
                    for (std::size_t c = 0; c < 3; ++c) predicted = std::max(predicted, std::abs(a[c] - b[c]));
                    if (predicted < 12.0) continue;
                    ++sens_b;
                    if (measured >= 12 - kSlack) ++differ_b;
                }
            }
            check.expect(count_a > 0 && same_a == count_a, std::format("handedness has no effect on the normal panel (w = +1): {} of {} probes unchanged", same_a, count_a));
            check.expect(sens_b >= 20 && differ_b == sens_b,
                         std::format("handedness matters on the mirrored panel (w = -1): {} probes with a predicted difference >= 12/255, the GPU shows it at {}", sens_b,
                                     differ_b));

            // --- negative controls: wrong CPU models must NOT match the correct GPU render --------------------
            const std::size_t wrong_hand = count_mismatches(img_full, w, probes, kTolerance, model(m_nohand, val_light, cam.eye));
            std::size_t wrong_hand_a = 0;
            {
                std::vector<Probe> a_only;
                for (const Probe& p : probes)
                    if (p.face == 0) a_only.push_back(p);
                wrong_hand_a = count_mismatches(img_full, w, a_only, kTolerance, model(m_nohand, val_light, cam.eye));
            }
            check.expect(wrong_hand >= 20 && wrong_hand_a == 0,
                         std::format("negative control: a CPU model that ignores the handedness sign mismatches the GPU at {} probes (all on the mirrored panel; {} on the other)",
                                     wrong_hand, wrong_hand_a));
            const std::size_t wrong_green = count_mismatches(img_full, w, probes, kTolerance, model(m_green, val_light, cam.eye));
            check.expect(wrong_green >= 50, std::format("negative control: a CPU model with the green channel of the normal map inverted (DirectX convention) mismatches at {} of {} probes",
                                                          wrong_green, probes.size()));
            // A second, independent mutation: the highlight without the specular map must differ from the masked one.
            const std::size_t wrong_spec = count_mismatches(img_full, w, probes, kTolerance, model(m_nospec, val_light, cam.eye));
            check.expect(wrong_spec >= 10, std::format("negative control: the unmasked-specular CPU model mismatches the masked GPU render at {} probes", wrong_spec));
        }

        void check_cube(demo::Check& check, int w, int h, const Camera& cam, const LightRig& val_light, double time)
        {
            const dm::Mat4 model = cube_model(time);
            std::vector<Face> faces;
            for (const Face& f : cube_faces()) faces.push_back(transformed(f, model));
            const std::vector<Probe> probes = gather_probes(faces, cam, w, h, 2, 0.05, true);
            check.expect(probes.size() > 50, std::format("{} probes on the visible faces of the rotated crate", probes.size()));

            Pass pass;
            pass.camera = cam;
            pass.light = val_light;
            pass.filter = Filter::Nearest;
            pass.wall = pass.sphere = pass.marker = false;
            pass.cube_model = model;
            const auto img = render_pass(pass, w, h);
            const auto ref = [&](const Probe& p) {
                return shade_reference(textures_cpu_, faces[static_cast<std::size_t>(p.face)], p.s, p.t, ShadeMode{}, val_light, cam.eye, 0);
            };
            const Comparison c = compare_probes(img, w, probes, kTolerance, ref);
            check.expect(c.bad == 0 && c.count > 0,
                         std::format("rotating crate (model matrix {:.2f} rad, normal matrix, tangent frame rotated with the model): {} probes match the CPU model, max error {:.2f}/255, tolerance {}",
                                     0.55 + 0.7 * time, c.count, c.max_error, kTolerance),
                         c.first_bad);

            // Negative control: the same crate pass compared with a model that forgets the model rotation of the tangent frame.
            std::vector<Face> unrotated;
            for (const Face& f : cube_faces()) unrotated.push_back(f);
            const auto wrong = [&](const Probe& p) {
                // keep the world position of the probe but use the frame of the un-rotated face
                Face f = unrotated[static_cast<std::size_t>(p.face)];
                f.origin = faces[static_cast<std::size_t>(p.face)].origin;
                f.normal = faces[static_cast<std::size_t>(p.face)].normal;  // the geometric normal is still right
                return shade_reference(textures_cpu_, f, p.s, p.t, ShadeMode{}, val_light, cam.eye, 0);
            };
            const std::size_t mismatches = count_mismatches(img, w, probes, kTolerance, wrong);
            check.expect(mismatches >= probes.size() / 4,
                         std::format("negative control: a CPU model whose tangent frame is NOT rotated with the crate mismatches at {} of {} probes", mismatches, probes.size()));
        }

        // Face-on wall at exactly 64 pixels per world unit: one pixel spans 2 level-0 texels, so lambda = 1.
        // The albedo-only view makes the output equal to the filtered texel, which is linear in the two
        // levels blended (ES 3.0 section 3.8.10.4: trilinear = (1 - f) * level d1 + f * level d2).
        void check_trilinear(demo::Check& check, int w, int h)
        {
            Camera cam;
            const double distance = 4.0, units_per_pixel_inv = 64.0;
            // Shift by a fraction of a pixel so the sample points are not on texel centres of level 1: then bilinear
            // lookups in level 0 and level 1 differ, and a wrong level cannot hide behind the filter.
            const double cx = 1.0 + 0.37 / 64.0, cy = 0.21 / 64.0;
            cam.eye = {cx, cy, distance};
            cam.target = {cx, cy, 0.0};
            cam.fovy = 2.0 * std::atan((h / 2.0) / (units_per_pixel_inv * distance));
            const std::vector<Face> faces = wall_faces();
            const std::vector<Probe> probes = gather_probes(faces, cam, w, h, 4, 0.1, false);

            Pass pass;
            pass.camera = cam;
            pass.view = 1;  // albedo only
            pass.filter = Filter::Trilinear;
            pass.cube = pass.sphere = pass.marker = false;
            const auto img = render_pass(pass, w, h);
            GLint min_filter = 0;
            mgl::glActiveTexture(kUnits[0]);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MinFilter, &min_filter);
            check.expect(min_filter == static_cast<GLint>(GL_LINEAR_MIPMAP_LINEAR), "TEXTURE_MIN_FILTER of the albedo texture is LINEAR_MIPMAP_LINEAR (trilinear)",
                         std::format("read {:#x}", min_filter));

            // CPU: bilinear samples of levels 0..3 at the probe UVs.
            const std::size_t n = probes.size();
            std::array<std::vector<Rgb>, 4> level;
            std::vector<Rgb> got(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                const Probe& p = probes[k];
                const Face& f = faces[static_cast<std::size_t>(p.face)];
                const double u = p.s * f.tile_u, v = p.t * f.tile_v;
                for (int l = 0; l < 4; ++l) level[static_cast<std::size_t>(l)].push_back(bilinear(textures_cpu_.albedo, l, u, v));
                const auto g = pixel_at(img, w, p.px, p.py);
                got[k] = {static_cast<double>(g[0]), static_cast<double>(g[1]), static_cast<double>(g[2])};
            }
            // Least-squares blend factor f in [0, 1] between two model levels, and the largest residual.
            auto fit = [&](const std::vector<Rgb>& a, const std::vector<Rgb>& b) {
                double num = 0, den = 0;
                for (std::size_t k = 0; k < n; ++k)
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        num += (got[k][c] - a[k][c]) * (b[k][c] - a[k][c]);
                        den += (b[k][c] - a[k][c]) * (b[k][c] - a[k][c]);
                    }
                const double f = den > 0 ? std::clamp(num / den, 0.0, 1.0) : 0.0;
                double worst = 0;
                for (std::size_t k = 0; k < n; ++k)
                    for (std::size_t c = 0; c < 3; ++c) worst = std::max(worst, std::abs(got[k][c] - (a[k][c] + f * (b[k][c] - a[k][c]))));
                return std::array<double, 2>{f, worst};
            };
            auto mean_diff = [&](const std::vector<Rgb>& a, const std::vector<Rgb>& b) {
                double s = 0;
                for (std::size_t k = 0; k < n; ++k)
                    for (std::size_t c = 0; c < 3; ++c) s += std::abs(a[k][c] - b[k][c]);
                return s / (3.0 * static_cast<double>(n));
            };
            // ES 3.0 section 3.8.10 gives no precision for filter weights or the level-of-detail blend factor, so this is only a sanity
            // bound: 8-bit storage (< 1) + weights in fixed point (measured < 1 on all drivers) + headroom. The measured value is printed.
            constexpr double kTrilinearTolerance = 3.0;
            const auto lod1 = fit(level[1], level[2]);
            check.expect(n > 500 && lod1[1] <= kTrilinearTolerance,
                         std::format("trilinear sampling at 2 texels/pixel: {} pixels equal a blend of mip levels 1 and 2 (fitted lambda = 1 + {:.3f}; ES 3.0 eq. 3.20 allows [1, 2] here), max residual {:.2f}/255, tolerance {}",
                                     n, lod1[0], lod1[1], kTrilinearTolerance));
            // Negative controls: single-level models of the wrong levels must not fit.
            for (const int wrong : {0, 2, 3})
            {
                const auto r = fit(level[static_cast<std::size_t>(wrong)], level[static_cast<std::size_t>(wrong)]);
                check.expect(r[1] > 2.0 * kTrilinearTolerance,
                             std::format("negative control: a model sampling only mip level {} mismatches (max residual {:.1f}/255; mean |level{} - level1| = {:.1f})", wrong, r[1], wrong,
                                         mean_diff(level[static_cast<std::size_t>(wrong)], level[1])));
            }
        }

        // ---- members ----------------------------------------------------------------------------
        mgl::ProgramId lit_{}, marker_{};
        LitUniforms u_{};
        mgl::UniformLocation marker_mvp_{}, marker_color_{};
        std::array<mgl::TextureId, 3> tex_{};
        GpuMesh wall_{}, cube_{}, sphere_{}, marker_gpu_{};
        Mesh wall_cpu_, cube_cpu_, sphere_cpu_, marker_mesh_;
        TextureSet textures_cpu_;

        int ticks_ = 0;
        int last_frame_ = -1;
        bool paused_ = false;
        bool normal_map_ = true, spec_map_ = true, handedness_ = true;
        Filter filter_ = Filter::Trilinear;
        int view_ = 0;
        double yaw_ = kCameraYaw0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-normal-mapping",
        .title = "Normal mapping: tangent space, specular map, mip chains",
        .min_es = {3, 0},
        .summary = "A brick wall, a rotating crate and a sphere lit per fragment by an orbiting point light, with procedurally generated "
                   "albedo, tangent-space normal and specular maps (full mip chains, trilinear filtering) and CPU-computed tangents with handedness.",
        .controls = "N normal map on/off | S specular map on/off | H tangent handedness on/off (off breaks the mirrored right half of the wall) | "
                    "F filter: trilinear / bilinear / nearest | V view: lit / albedo / normal map / specular map | P pause animation | A, D orbit the camera",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
    };
    return demo::run<NormalMapping>(argc, argv, info);
}
