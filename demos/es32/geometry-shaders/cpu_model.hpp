// CPU model of the four geometry shaders of es32-geometry-shaders (no OpenGL calls in this file).
//
// The model has three layers:
//   1. meshes and particle data (generated deterministically, quantised to dyadic fractions so that the shader
//      arithmetic on them is exact in IEEE single precision);
//   2. "emulators" that re-implement what each geometry shader emits: a list of triangle strips of EVert;
//   3. a tiny software rasterizer (perspective-correct, depth-tested) with a strict and a loose coverage mode, so
//      that pixels within a fraction of a pixel of a primitive edge can be told apart from pixels whose value is
//      unambiguous.
#pragma once

#include <demo/math.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace gsmodel
{
    using demo::math::Mat4;
    using demo::math::Vec3;
    using demo::math::Vec4;

    struct Vec2
    {
        float x = 0, y = 0;
    };
    inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
    inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }

    struct Rgb
    {
        float r = 0, g = 0, b = 0;
    };
    inline Rgb operator*(Rgb c, float s) { return {c.r * s, c.g * s, c.b * s}; }
    inline Rgb operator+(Rgb a, Rgb b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }

    // The palette of the shaders (kPalette).
    inline constexpr std::array<Rgb, 6> kPalette = {{{1.00f, 0.35f, 0.30f}, {1.00f, 0.80f, 0.25f}, {0.45f, 0.90f, 0.40f},
                                                     {0.30f, 0.80f, 0.95f}, {0.50f, 0.45f, 1.00f}, {0.95f, 0.40f, 0.85f}}};

    using demo::math::cross;
    using demo::math::dot;
    using demo::math::normalize;
    inline float quantize(float v, float steps) { return std::round(v * steps) / steps; }

    // ---------------------------------------------------------------------------------------------------------
    // Emitted vertices and strips
    // ---------------------------------------------------------------------------------------------------------
    struct EVert
    {
        Vec4 clip{};       // gl_Position
        Vec3 world{};      // GS_OUT.world / GS_OUT.eye / GS_OUT.edge
        Vec3 normal{};     // GS_OUT.normal
        Vec2 uv{};         // GS_OUT.uv
        float side = 0;    // GS_OUT.side
        int prim = 0;      // GS_OUT.prim / gl_PrimitiveID / GS_OUT.face
        int inst = 0;      // GS_OUT.inst
        float aux = 0;     // model-only: half size of a sprite in pixels (for the discard margin)
    };
    using Strip = std::vector<EVert>;  // one output primitive of a geometry shader (triangle_strip)

    struct Tri
    {
        std::array<EVert, 3> v;
    };

    // Triangle strip decomposition, ES 3.2 section 10.1.5: triangle i is (i, i+1, i+2) for even i and
    // (i+1, i, i+2) for odd i. Transform feedback records the triangles in this order (section 12.2.2).
    inline std::vector<std::array<std::size_t, 3>> strip_order(std::size_t vertices)
    {
        std::vector<std::array<std::size_t, 3>> out;
        for (std::size_t i = 0; i + 2 < vertices; ++i)
        {
            if (i % 2 == 0) out.push_back({i, i + 1, i + 2});
            else out.push_back({i + 1, i, i + 2});
        }
        return out;
    }

    inline std::vector<Tri> triangles(const std::vector<Strip>& strips)
    {
        std::vector<Tri> out;
        for (const Strip& s : strips)
            for (const auto& t : strip_order(s.size())) out.push_back({{s[t[0]], s[t[1]], s[t[2]]}});
        return out;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Meshes
    // ---------------------------------------------------------------------------------------------------------
    struct Mesh
    {
        std::vector<Vec3> position;
        std::vector<Vec3> normal;
        std::vector<std::uint16_t> index;  // triangle list, counter-clockwise seen from outside
    };

    // Icosphere (radius 1, one subdivision = 80 faces) with all coordinates rounded to multiples of 1/32.
    inline Mesh icosphere()
    {
        const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<Vec3> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                               {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
        for (Vec3& p : v) p = normalize(p);
        std::vector<std::array<int, 3>> faces = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
                                                 {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                                 {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
                                                 {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
        std::map<std::pair<int, int>, int> mid;
        auto midpoint = [&](int a, int b) {
            const std::pair<int, int> key{std::min(a, b), std::max(a, b)};
            const auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            v.push_back(normalize((v[static_cast<std::size_t>(a)] + v[static_cast<std::size_t>(b)]) * 0.5f));
            return mid[key] = static_cast<int>(v.size()) - 1;
        };
        std::vector<std::array<int, 3>> fine;
        for (const auto& f : faces)
        {
            const int a = midpoint(f[0], f[1]), b = midpoint(f[1], f[2]), c = midpoint(f[2], f[0]);
            fine.push_back({f[0], a, c});
            fine.push_back({f[1], b, a});
            fine.push_back({f[2], c, b});
            fine.push_back({a, b, c});
        }
        Mesh m;
        for (Vec3 p : v)
        {
            p = {quantize(p.x, 32), quantize(p.y, 32), quantize(p.z, 32)};
            m.position.push_back(p);
            m.normal.push_back(normalize(p));
        }
        for (auto f : fine)
        {
            const Vec3 a = m.position[static_cast<std::size_t>(f[0])], b = m.position[static_cast<std::size_t>(f[1])],
                       c = m.position[static_cast<std::size_t>(f[2])];
            if (dot(cross(b - a, c - a), a + b + c) < 0.0f) std::swap(f[1], f[2]);  // make it counter-clockwise
            for (int i : f) m.index.push_back(static_cast<std::uint16_t>(i));
        }
        return m;
    }

    // Torus around the Y axis (major radius R, minor radius r), `major` x `minor` quads, analytic normals.
    inline Mesh torus(int major, int minor, float R, float r)
    {
        Mesh m;
        for (int i = 0; i < major; ++i)
            for (int j = 0; j < minor; ++j)
            {
                const float u = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(major);
                const float v = 2.0f * std::numbers::pi_v<float> * static_cast<float>(j) / static_cast<float>(minor);
                const Vec3 n = {std::cos(v) * std::cos(u), std::sin(v), std::cos(v) * std::sin(u)};
                m.normal.push_back(n);
                m.position.push_back({(R + r * std::cos(v)) * std::cos(u), r * std::sin(v), (R + r * std::cos(v)) * std::sin(u)});
            }
        auto at = [&](int i, int j) { return static_cast<std::uint16_t>((i % major) * minor + (j % minor)); };
        for (int i = 0; i < major; ++i)
            for (int j = 0; j < minor; ++j)
            {
                const std::uint16_t q[4] = {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)};
                const std::uint16_t tri[2][3] = {{q[0], q[1], q[2]}, {q[0], q[2], q[3]}};
                for (const auto& t : tri)
                {
                    std::uint16_t a = t[0], b = t[1], c = t[2];
                    const Vec3 pa = m.position[a], pb = m.position[b], pc = m.position[c];
                    if (dot(cross(pb - pa, pc - pa), m.normal[a] + m.normal[b] + m.normal[c]) < 0.0f) std::swap(b, c);
                    m.index.push_back(a);
                    m.index.push_back(b);
                    m.index.push_back(c);
                }
            }
        return m;
    }

    // Index list for TRIANGLES_ADJACENCY: per triangle (v0, a01, v1, a12, v2, a20), where a_ij is the vertex of the
    // neighbouring triangle that lies opposite to the edge i-j (ES 3.2 section 10.1.10). Requires a closed
    // 2-manifold with consistent winding.
    inline std::vector<std::uint16_t> adjacency_indices(const Mesh& m)
    {
        std::map<std::pair<int, int>, int> opposite;  // directed edge (a,b) -> third vertex of the triangle that owns it
        for (std::size_t t = 0; t + 2 < m.index.size(); t += 3)
            for (int e = 0; e < 3; ++e)
                opposite[{m.index[t + static_cast<std::size_t>(e)], m.index[t + static_cast<std::size_t>((e + 1) % 3)]}] =
                    m.index[t + static_cast<std::size_t>((e + 2) % 3)];
        std::vector<std::uint16_t> out;
        for (std::size_t t = 0; t + 2 < m.index.size(); t += 3)
            for (int e = 0; e < 3; ++e)
            {
                const int a = m.index[t + static_cast<std::size_t>(e)], b = m.index[t + static_cast<std::size_t>((e + 1) % 3)];
                out.push_back(static_cast<std::uint16_t>(a));
                out.push_back(static_cast<std::uint16_t>(opposite.at({b, a})));  // the neighbour owns the edge reversed
            }
        return out;
    }

    // A cube (side 0.75) as 12 triangles (two per face), counter-clockwise from outside.
    inline Mesh cube()
    {
        Mesh m;
        const float h = 0.375f;
        const Vec3 corner[8] = {{-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}};
        const int quad[6][4] = {{4, 5, 6, 7}, {1, 0, 3, 2}, {5, 1, 2, 6}, {0, 4, 7, 3}, {7, 6, 2, 3}, {0, 1, 5, 4}};
        for (const auto& q : quad)
        {
            const std::uint16_t base = static_cast<std::uint16_t>(m.position.size());
            for (int i = 0; i < 4; ++i)
            {
                m.position.push_back(corner[q[i]]);
                m.normal.push_back({});
            }
            for (std::uint16_t i : {0, 1, 2, 0, 2, 3}) m.index.push_back(static_cast<std::uint16_t>(base + i));
        }
        return m;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Particles of cell A
    // ---------------------------------------------------------------------------------------------------------
    struct Sprite
    {
        Vec3 position;  // world space, multiple of 1/16
        float size;     // half extent, multiple of 1/16
    };

    inline std::vector<Sprite> galaxy()
    {
        std::vector<Sprite> s;
        const int arms = 3, count = 150;
        for (int i = 0; i < count; ++i)
        {
            const float t = static_cast<float>(i / arms) / static_cast<float>(count / arms);  // position along the arm
            const float arm = static_cast<float>(i % arms);
            const float angle = arm * 2.0943951f + 4.5f * t;
            const float radius = 0.45f + 2.35f * t;
            const float y = 0.22f * std::sin(11.0f * t + arm * 2.0f);
            s.push_back({{quantize(radius * std::cos(angle), 16), quantize(y, 16), quantize(radius * std::sin(angle), 16)},
                         static_cast<float>(2 + (i * 7) % 4) / 16.0f});
        }
        s.push_back({{0, 0, 0}, 9.0f / 16.0f});  // the central bulge (index 150 -> palette 0)
        return s;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Geometry shader emulators
    // ---------------------------------------------------------------------------------------------------------
    inline Vec4 mul(const Mat4& m, Vec3 p) { return m * Vec4{p.x, p.y, p.z, 1.0f}; }

    // billboard.geom. `view` and `proj` as in the shader; `px_scale` = pixels per NDC unit / 2 (cell size / 2).
    inline std::vector<Strip> emulate_billboards(const std::vector<Sprite>& sprites, const Mat4& view, const Mat4& proj,
                                                 float half_cell_px)
    {
        std::vector<Strip> out;
        for (std::size_t i = 0; i < sprites.size(); ++i)
        {
            const Vec4 e = mul(view, sprites[i].position);
            const Vec3 center = {e.x, e.y, e.z};
            Strip s;
            for (int c = 0; c < 4; ++c)
            {
                const Vec2 corner = {static_cast<float>(c & 1) * 2.0f - 1.0f, static_cast<float>(c >> 1) * 2.0f - 1.0f};
                const Vec3 p = center + Vec3{corner.x * sprites[i].size, corner.y * sprites[i].size, 0.0f};
                EVert v;
                v.world = p;
                v.uv = corner;
                v.prim = static_cast<int>(i);
                v.clip = mul(proj, p);
                v.aux = sprites[i].size * proj.at(0, 0) * half_cell_px / v.clip.w;  // half size in pixels
                s.push_back(v);
            }
            out.push_back(std::move(s));
        }
        return out;
    }

    // crystal.geom
    inline std::vector<Strip> emulate_crystals(const Mesh& mesh, const Mat4& viewproj, float lift)
    {
        std::vector<Strip> out;
        for (std::size_t t = 0; t + 2 < mesh.index.size(); t += 3)
        {
            const int id = static_cast<int>(t / 3);
            const Vec3 p[3] = {mesh.position[mesh.index[t]], mesh.position[mesh.index[t + 1]], mesh.position[mesh.index[t + 2]]};
            const Vec3 n = cross(p[1] - p[0], p[2] - p[0]);
            const Vec3 m = n * (lift * static_cast<float>(1 + id % 3));
            auto vert = [&](Vec3 q, Vec3 nn) {
                EVert v;
                v.world = q;
                v.normal = nn;
                v.prim = id;
                v.clip = mul(viewproj, q);
                return v;
            };
            out.push_back({vert(p[0] + m, n), vert(p[1] + m, n), vert(p[2] + m, n)});
            for (int k = 0; k < 3; ++k)
            {
                const Vec3 a = p[k], b = p[(k + 1) % 3];
                const Vec3 side = cross(b - a, n);
                out.push_back({vert(a, side), vert(a + m, side), vert(b, side), vert(b + m, side)});
            }
        }
        return out;
    }

    struct SilhouetteResult
    {
        std::vector<Strip> strips;
        float min_facing = 1e30f;  // smallest |normalised dot(face normal, direction to the eye)| that was tested
    };

    // silhouette.geom. `adj` is the TRIANGLES_ADJACENCY index list of `mesh`.
    inline SilhouetteResult emulate_silhouette(const Mesh& mesh, const std::vector<std::uint16_t>& adj, const Mat4& viewproj,
                                               Vec3 eye, Vec2 halfwidth, float bias)
    {
        SilhouetteResult r;
        auto facing = [&](Vec3 a, Vec3 b, Vec3 c) {
            const Vec3 n = cross(b - a, c - a);
            const Vec3 to_eye = eye - a;
            const float len = std::sqrt(dot(n, n) * dot(to_eye, to_eye));
            r.min_facing = std::min(r.min_facing, std::fabs(dot(n, to_eye)) / len);
            return dot(n, to_eye) > 0.0f;
        };
        for (std::size_t t = 0; t + 5 < adj.size(); t += 6)
        {
            Vec3 w[6];
            for (std::size_t i = 0; i < 6; ++i) w[i] = mesh.position[adj[t + i]];
            if (!facing(w[0], w[2], w[4])) continue;
            for (int e = 0; e < 3; ++e)
            {
                const Vec3 a = w[2 * e], b = w[(2 * e + 2) % 6], c = w[2 * e + 1];
                if (facing(b, a, c)) continue;
                const Vec4 ca = mul(viewproj, a), cb = mul(viewproj, b);
                Vec2 d = {cb.x / cb.w - ca.x / ca.w, cb.y / cb.w - ca.y / ca.w};
                const float len = std::sqrt(d.x * d.x + d.y * d.y);
                d = {d.x / len, d.y / len};
                const Vec2 off = {-d.y * halfwidth.x, d.x * halfwidth.y};
                auto corner = [&](Vec3 p, Vec4 clip, float sign) {
                    EVert v;
                    v.world = p;
                    v.side = sign;
                    v.clip = {clip.x + sign * off.x * clip.w, clip.y + sign * off.y * clip.w, clip.z - bias * clip.w, clip.w};
                    return v;
                };
                r.strips.push_back({corner(a, ca, -1.0f), corner(a, ca, +1.0f), corner(b, cb, -1.0f), corner(b, cb, +1.0f)});
            }
        }
        return r;
    }

    // ring.geom: copy k is rotated about Y by (cos, sin) = spin[k] and moved to place[k].
    inline std::vector<Strip> emulate_ring(const Mesh& mesh, const Mat4& viewproj, const std::array<Vec3, 6>& place,
                                           const std::array<Vec2, 6>& spin)
    {
        std::vector<Strip> out;
        for (std::size_t t = 0; t + 2 < mesh.index.size(); t += 3)
            for (int k = 0; k < 6; ++k)
            {
                Vec3 q[3];
                for (int i = 0; i < 3; ++i)
                {
                    const Vec3 p = mesh.position[mesh.index[t + static_cast<std::size_t>(i)]];
                    const Vec2 cs = spin[static_cast<std::size_t>(k)];
                    q[i] = place[static_cast<std::size_t>(k)] + Vec3{cs.x * p.x + cs.y * p.z, p.y, -cs.y * p.x + cs.x * p.z};
                }
                const Vec3 n = cross(q[1] - q[0], q[2] - q[0]);
                Strip s;
                for (int i = 0; i < 3; ++i)
                {
                    EVert v;
                    v.world = q[i];
                    v.normal = n;
                    v.inst = k;
                    v.prim = static_cast<int>(t / 3) / 2;
                    v.clip = mul(viewproj, q[i]);
                    s.push_back(v);
                }
                out.push_back(std::move(s));
            }
        return out;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Software rasterizer
    // ---------------------------------------------------------------------------------------------------------
    struct Frag
    {
        const Tri* tri = nullptr;
        std::array<float, 3> w{};  // perspective-correct barycentric weights
        float margin = 0.0f;       // > 0: loose pass, < 0: strict pass (pixels)

        template <typename F>
        [[nodiscard]] auto lerp(F EVert::*member) const
        {
            auto r = tri->v[0].*member * w[0];
            r = r + tri->v[1].*member * w[1];
            r = r + tri->v[2].*member * w[2];
            return r;
        }
    };

    struct Pixel
    {
        bool covered = false;     // a primitive certainly covers the pixel
        bool ambiguous = false;   // the true result may be a different colour (edge, depth tie, sub-pixel sensitivity)
        Rgb color{};              // expected colour when !ambiguous (the background when nothing covers the pixel)
        float sensitivity = 0.0f; // largest colour change when the sample point moves by `jitter` pixels
    };

    inline float color_distance(Rgb a, Rgb b)
    {
        return std::max({std::fabs(a.r - b.r), std::fabs(a.g - b.g), std::fabs(a.b - b.b)});
    }

    class Raster
    {
    public:
        Raster(int width, int height) : w_(width), h_(height) {}

        // Returns false when the triangle has a vertex at or behind the eye plane (the model does not clip).
        bool add(const Tri& t)
        {
            S s;
            s.tri = t;
            for (int i = 0; i < 3; ++i)
            {
                const Vec4 c = t.v[static_cast<std::size_t>(i)].clip;
                if (c.w <= 1e-4f) return false;
                s.x[i] = (c.x / c.w * 0.5f + 0.5f) * static_cast<float>(w_);
                s.y[i] = (c.y / c.w * 0.5f + 0.5f) * static_cast<float>(h_);
                s.z[i] = c.z / c.w;
                s.iw[i] = 1.0f / c.w;
            }
            s.area = (s.x[1] - s.x[0]) * (s.y[2] - s.y[0]) - (s.y[1] - s.y[0]) * (s.x[2] - s.x[0]);
            s.minx = std::min({s.x[0], s.x[1], s.x[2]});
            s.maxx = std::max({s.x[0], s.x[1], s.x[2]});
            s.miny = std::min({s.y[0], s.y[1], s.y[2]});
            s.maxy = std::max({s.y[0], s.y[1], s.y[2]});
            if (std::fabs(s.area) > 1e-6f) tris_.push_back(s);
            return true;
        }

        // Expected colour of the pixel (px, py) (window coordinates, origin bottom left).
        //
        // A triangle whose edges are all at least `margin` pixels away from the pixel centre (towards the inside)
        // certainly covers it; one that is within `margin` pixels outside or inside an edge MAY cover it, depending on
        // how the implementation rounds vertex positions (SUBPIXEL_BITS >= 4, ES 3.2 Table 21.40) and rasterizes
        // edges. The result is flagged ambiguous when such a "may cover" triangle, being nearer than the winner of the
        // certain ones, would change the colour by more than `agree`; it is also ambiguous when two certain triangles
        // tie in depth, and when moving the sample point by `jitter` pixels changes the winner's colour by more than
        // `max_sensitivity` (e.g. the rim of a sphere impostor, where the shading has an unbounded slope).
        // shade(Frag) -> optional<Rgb>; Frag::margin is < 0 for the certain test and > 0 for the may-cover test, so the
        // shader can move its own discontinuities (discard) the same way.
        template <typename Shade>
        [[nodiscard]] Pixel resolve(int px, int py, float margin, float agree, float jitter, float max_sensitivity, Rgb background,
                                    Shade&& shade) const
        {
            const float cx = static_cast<float>(px) + 0.5f, cy = static_cast<float>(py) + 0.5f;
            struct Cand
            {
                float z;
                Rgb color;
                const S* s;
            };
            Cand best{2.0f, {}, nullptr};
            Cand second{2.0f, {}, nullptr};
            std::vector<Cand> maybe;
            for (const S& s : tris_)
            {
                if (cx < s.minx - margin - 1.0f || cx > s.maxx + margin + 1.0f || cy < s.miny - margin - 1.0f || cy > s.maxy + margin + 1.0f) continue;
                const Eval ev = evaluate(s, cx, cy);
                if (ev.dist < -margin) continue;
                if (ev.dist >= margin)
                {
                    if (const std::optional<Rgb> c = shade(Frag{&s.tri, ev.pw, -margin}))
                    {
                        const Cand cand{ev.z, *c, &s};
                        if (cand.z < best.z) second = std::exchange(best, cand);
                        else if (cand.z < second.z) second = cand;
                        continue;
                    }
                }
                if (const std::optional<Rgb> c = shade(Frag{&s.tri, ev.pw, +margin})) maybe.push_back({ev.z, *c, &s});
            }
            Pixel out;
            out.covered = best.s != nullptr;
            out.color = out.covered ? best.color : background;
            if (out.covered && second.s != nullptr && std::fabs(second.z - best.z) < 2e-6f && color_distance(second.color, best.color) > agree) out.ambiguous = true;
            for (const Cand& m : maybe)
                if (!out.covered || m.z < best.z + 1e-6f)
                    if (color_distance(m.color, out.color) > agree) out.ambiguous = true;
            if (out.covered && !out.ambiguous)
            {
                for (const float sx : {-jitter, jitter})
                    for (const float sy : {-jitter, jitter})
                    {
                        const Eval ev = evaluate(*best.s, cx + sx, cy + sy);
                        const std::optional<Rgb> c = shade(Frag{&best.s->tri, ev.pw, -margin});
                        out.sensitivity = std::max(out.sensitivity, c ? color_distance(*c, best.color) : 1.0f);
                    }
                if (out.sensitivity > max_sensitivity) out.ambiguous = true;
            }
            return out;
        }

    private:
        struct S
        {
            Tri tri;
            float x[3]{}, y[3]{}, z[3]{}, iw[3]{};
            float area = 0, minx = 0, maxx = 0, miny = 0, maxy = 0;
        };
        struct Eval
        {
            float dist = 0;  // signed distance in pixels to the nearest edge (positive inside)
            float z = 0;     // NDC depth
            std::array<float, 3> pw{};
        };
        static Eval evaluate(const S& s, float cx, float cy)
        {
            const float sign = s.area > 0 ? 1.0f : -1.0f;
            float e[3];
            Eval ev;
            ev.dist = 1e30f;
            for (int i = 0; i < 3; ++i)
            {
                const int j = (i + 1) % 3, k = (i + 2) % 3;  // edge j -> k is opposite to vertex i
                const float ex = s.x[k] - s.x[j], ey = s.y[k] - s.y[j];
                e[i] = (ex * (cy - s.y[j]) - ey * (cx - s.x[j])) * sign;
                ev.dist = std::min(ev.dist, e[i] / std::sqrt(ex * ex + ey * ey));
            }
            const float area = std::fabs(s.area);
            ev.z = (e[0] * s.z[0] + e[1] * s.z[1] + e[2] * s.z[2]) / area;
            ev.pw = {e[0] / area * s.iw[0], e[1] / area * s.iw[1], e[2] / area * s.iw[2]};
            const float sum = ev.pw[0] + ev.pw[1] + ev.pw[2];
            for (float& f : ev.pw) f /= sum;
            return ev;
        }
        int w_, h_;
        std::vector<S> tris_;
    };

    inline std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

    // Blinn-Phong used by cells B and C: albedo * (0.18 + 0.82 * diffuse) + 0.30 * specular(24).
    inline Rgb blinn_phong(Rgb albedo, Vec3 n, Vec3 world, Vec3 eye, Vec3 light)
    {
        n = normalize(n);
        const Vec3 v = normalize(eye - world);
        const Vec3 h = normalize(light + v);
        const float diff = std::max(dot(n, light), 0.0f);
        const float spec = std::pow(std::max(dot(n, h), 0.0f), 24.0f);
        return albedo * (0.18f + 0.82f * diff) + Rgb{0.30f * spec, 0.30f * spec, 0.30f * spec};
    }
}
