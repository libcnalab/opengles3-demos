// CPU model of es32-layered-rendering (no OpenGL calls in this file).
//
//   - the cube map face table (ES 3.2 Table 8.19 / 8.24): which direction belongs to which face and texel, and the
//     view matrices that render exactly that orientation;
//   - the environment room (walls, decals, boxes) as a list of axis-aligned quads, used both to build the vertex
//     buffer and to ray-cast "what colour does the cube map hold in direction d" without any GL;
//   - the exact images of the 2D-array layers and 3D-texture slices (pixel-aligned rectangles).
#pragma once

#include <demo/math.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lrmodel
{
    using demo::math::Mat4;
    using demo::math::Vec3;
    using Rgba8 = std::array<std::uint8_t, 4>;
    using Rgb8 = std::array<std::uint8_t, 3>;

    // ---------------------------------------------------------------------------------------------------------
    // Cube map face table
    // ---------------------------------------------------------------------------------------------------------
    inline constexpr std::array<const char*, 6> kFaceNames = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

    // Direction through the point (s, t) in [0,1]^2 of face `face`: sc = 2s-1, tc = 2t-1 and the major axis rows of
    // Table 8.19 solved for the direction (ma = +-1).
    inline Vec3 face_direction(int face, float s, float t)
    {
        const float sc = 2.0f * s - 1.0f, tc = 2.0f * t - 1.0f;
        switch (face)
        {
        case 0: return {1.0f, -tc, -sc};
        case 1: return {-1.0f, -tc, sc};
        case 2: return {sc, 1.0f, tc};
        case 3: return {sc, -1.0f, -tc};
        case 4: return {sc, -tc, 1.0f};
        default: return {-sc, -tc, -1.0f};
        }
    }

    // Camera of face `face` such that window x grows with s and window y grows with t (texel (i, j) of the face is
    // window pixel (i, j)): forward = the face axis, up = the direction of increasing t.
    struct FaceCamera
    {
        Vec3 forward, up;
    };
    inline FaceCamera face_camera(int face)
    {
        switch (face)
        {
        case 0: return {{1, 0, 0}, {0, -1, 0}};
        case 1: return {{-1, 0, 0}, {0, -1, 0}};
        case 2: return {{0, 1, 0}, {0, 0, 1}};
        case 3: return {{0, -1, 0}, {0, 0, -1}};
        case 4: return {{0, 0, 1}, {0, -1, 0}};
        default: return {{0, 0, -1}, {0, -1, 0}};
        }
    }

    inline Mat4 face_viewproj(int face, float near_plane, float far_plane)
    {
        const FaceCamera c = face_camera(face);
        return demo::math::perspective(demo::math::radians(90.0f), 1.0f, near_plane, far_plane) * demo::math::look_at({0, 0, 0}, c.forward, c.up);
    }

    // ---------------------------------------------------------------------------------------------------------
    // The environment: axis-aligned quads
    // ---------------------------------------------------------------------------------------------------------
    struct Quad
    {
        int axis;              // the quad lies in the plane p[axis] = coord
        float coord;
        float u0, u1;          // extent along axis (axis + 1) % 3
        float v0, v1;          // extent along axis (axis + 2) % 3
        Rgb8 color;
    };

    inline constexpr float kRoom = 4.0f;  // half edge of the room
    inline constexpr float kDecalOffset = 0.0625f;

    inline Rgb8 scaled(Rgb8 c, float f)
    {
        return {static_cast<std::uint8_t>(std::lround(static_cast<float>(c[0]) * f)), static_cast<std::uint8_t>(std::lround(static_cast<float>(c[1]) * f)),
                static_cast<std::uint8_t>(std::lround(static_cast<float>(c[2]) * f))};
    }

    inline void add_box(std::vector<Quad>& q, Vec3 center, Vec3 half, Rgb8 base)
    {
        const float c[3] = {center.x, center.y, center.z};
        const float h[3] = {half.x, half.y, half.z};
        // Per-face shading: +-X 0.8, +-Z 0.62, +Y 1.0, -Y 0.4.
        const float shade[3][2] = {{0.8f, 0.8f}, {1.0f, 0.4f}, {0.62f, 0.62f}};
        for (int a = 0; a < 3; ++a)
            for (int side = 0; side < 2; ++side)
            {
                const int u = (a + 1) % 3, v = (a + 2) % 3;
                q.push_back({a, c[a] + (side == 0 ? h[a] : -h[a]), c[u] - h[u], c[u] + h[u], c[v] - h[v], c[v] + h[v], scaled(base, shade[a][side])});
            }
    }

    inline std::vector<Quad> environment()
    {
        std::vector<Quad> q;
        const Rgb8 wall[3][2] = {{{200, 90, 60}, {60, 170, 200}}, {{200, 200, 215}, {70, 70, 80}}, {{80, 170, 90}, {150, 80, 170}}};  // +X -X, +Y -Y, +Z -Z
        for (int a = 0; a < 3; ++a)
            for (int side = 0; side < 2; ++side)
                q.push_back({a, side == 0 ? kRoom : -kRoom, -kRoom, kRoom, -kRoom, kRoom, wall[a][side]});
        // Decals, one per wall, offset a little towards the centre. Axis (a+1)%3 / (a+2)%3 extents:
        //   +X wall (u = y, v = z): window    -X wall: door    +Y (u = z, v = x): lamp    -Y: carpet    +Z (u = x, v = y): stripe    -Z: poster
        q.push_back({0, kRoom - kDecalOffset, 0.5f, 2.5f, -1.5f, 0.5f, {255, 230, 90}});
        q.push_back({0, -kRoom + kDecalOffset, -4.0f, 0.75f, -2.0f, 0.0f, {30, 50, 120}});
        q.push_back({1, kRoom - kDecalOffset, -1.0f, 1.0f, -1.0f, 1.0f, {255, 255, 255}});
        q.push_back({1, -kRoom + kDecalOffset, -2.5f, 2.5f, -2.5f, 2.5f, {180, 40, 50}});
        q.push_back({2, kRoom - kDecalOffset, -4.0f, 4.0f, -0.5f, 0.0f, {255, 140, 0}});
        q.push_back({2, -kRoom + kDecalOffset, 1.0f, 3.0f, 0.0f, 2.5f, {20, 200, 160}});
        // Three boxes standing on the floor (y = -4).
        add_box(q, {1.6f, -3.0f, 0.8f}, {1.0f, 1.0f, 1.0f}, {230, 60, 60});
        add_box(q, {-1.8f, -3.2f, 1.8f}, {0.8f, 0.8f, 0.8f}, {60, 90, 230});
        add_box(q, {-0.8f, -2.5f, -2.2f}, {0.6f, 1.5f, 0.6f}, {240, 200, 50});
        return q;
    }

    // Triangle list (position, colour) for the vertex buffer: two triangles per quad.
    inline std::vector<float> environment_vertices(const std::vector<Quad>& quads)
    {
        std::vector<float> out;
        for (const Quad& q : quads)
        {
            const int u = (q.axis + 1) % 3, v = (q.axis + 2) % 3;
            auto corner = [&](float cu, float cv) {
                float p[3];
                p[q.axis] = q.coord;
                p[u] = cu;
                p[v] = cv;
                return std::array<float, 3>{p[0], p[1], p[2]};
            };
            const auto c00 = corner(q.u0, q.v0), c10 = corner(q.u1, q.v0), c11 = corner(q.u1, q.v1), c01 = corner(q.u0, q.v1);
            for (const auto& c : {c00, c10, c11, c00, c11, c01})
            {
                out.insert(out.end(), {c[0], c[1], c[2]});
                out.insert(out.end(), {static_cast<float>(q.color[0]) / 255.0f, static_cast<float>(q.color[1]) / 255.0f, static_cast<float>(q.color[2]) / 255.0f});
            }
        }
        return out;
    }

    // The colour the cube map holds in direction `d` (ray from the origin, nearest quad wins).
    inline Rgb8 environment_color(const std::vector<Quad>& quads, Vec3 d)
    {
        const float dir[3] = {d.x, d.y, d.z};
        float best = 1e30f;
        Rgb8 color = {0, 0, 0};
        for (const Quad& q : quads)
        {
            const float da = dir[q.axis];
            if (std::fabs(da) < 1e-12f) continue;
            const float t = q.coord / da;
            if (t <= 0.0f || t >= best) continue;
            const float u = dir[(q.axis + 1) % 3] * t, v = dir[(q.axis + 2) % 3] * t;
            if (u < q.u0 || u > q.u1 || v < q.v0 || v > q.v1) continue;
            best = t;
            color = q.color;
        }
        return color;
    }

    // Is the colour the same for all nine directions at (centre +- 0.75 texel) of texel (i, j) of `face`?
    // Used to drop texels near the edge of a polygon, where rasterization decides.
    inline bool texel_is_uniform(const std::vector<Quad>& quads, int face, int i, int j, int size, Rgb8* color)
    {
        const Rgb8 centre = environment_color(quads, face_direction(face, (static_cast<float>(i) + 0.5f) / static_cast<float>(size), (static_cast<float>(j) + 0.5f) / static_cast<float>(size)));
        for (const float du : {-0.75f, 0.0f, 0.75f})
            for (const float dv : {-0.75f, 0.0f, 0.75f})
            {
                const Rgb8 c = environment_color(quads, face_direction(face, (static_cast<float>(i) + 0.5f + du) / static_cast<float>(size),
                                                                       (static_cast<float>(j) + 0.5f + dv) / static_cast<float>(size)));
                if (c != centre) return false;
            }
        *color = centre;
        return true;
    }

    // Is the environment colour the same in 8 directions at angle `radius` (radians) around `d`?
    inline bool direction_is_uniform(const std::vector<Quad>& quads, Vec3 d, float radius, Rgb8* color)
    {
        using demo::math::cross;
        using demo::math::normalize;
        d = normalize(d);
        const Vec3 helper = std::fabs(d.y) < 0.9f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        const Vec3 a = normalize(cross(d, helper));
        const Vec3 b = cross(d, a);
        const Rgb8 centre = environment_color(quads, d);
        for (int k = 0; k < 8; ++k)
        {
            const float angle = static_cast<float>(k) * 0.78539816f;
            const Vec3 p = d + a * (std::tan(radius) * std::cos(angle)) + b * (std::tan(radius) * std::sin(angle));
            if (environment_color(quads, p) != centre) return false;
        }
        *color = centre;
        return true;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Layer images (64 x 64 RGBA8, row 0 = bottom)
    // ---------------------------------------------------------------------------------------------------------
    inline constexpr int kLayerSize = 64;
    using Image = std::vector<std::uint8_t>;

    inline Image solid_image(Rgba8 c)
    {
        Image img(static_cast<std::size_t>(kLayerSize * kLayerSize * 4));
        for (std::size_t i = 0; i < img.size(); i += 4)
            for (std::size_t k = 0; k < 4; ++k) img[i + k] = c[k];
        return img;
    }

    inline void fill_rect(Image& img, int x0, int y0, int x1, int y1, Rgba8 c)
    {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x)
                for (std::size_t k = 0; k < 4; ++k) img[(static_cast<std::size_t>(y) * kLayerSize + static_cast<std::size_t>(x)) * 4 + k] = c[k];
    }

    // array_layers.geom: layer n.
    inline Image array_layer(int n)
    {
        static constexpr Rgba8 back[4] = {{51, 0, 0, 255}, {0, 51, 0, 255}, {0, 0, 102, 255}, {51, 0, 51, 255}};
        static constexpr Rgba8 body[4] = {{255, 102, 102, 255}, {102, 255, 102, 255}, {102, 102, 255, 255}, {255, 204, 102, 255}};
        Image img = solid_image(back[n]);
        fill_rect(img, 8 + 4 * n, 8 + 4 * n, 56 - 4 * n, 56 - 4 * n, body[n]);
        for (int i = 0; i <= n; ++i) fill_rect(img, 4 + 8 * i, 2, 8 + 8 * i, 6, {255, 255, 255, 255});
        return img;
    }

    // volume_slices.geom: slice k.
    inline Image volume_slice(int k)
    {
        Image img = solid_image({0, static_cast<std::uint8_t>(51 + 51 * k), static_cast<std::uint8_t>(204 - 51 * k), 255});
        fill_rect(img, 6 + 14 * k, 6 + 14 * k, 18 + 14 * k, 18 + 14 * k, {255, 255, 102, 255});
        fill_rect(img, 0, 0, 16 * (k + 1), 4, {255, 153, 0, 255});
        return img;
    }
}
