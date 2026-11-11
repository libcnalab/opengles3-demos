// Minimal column-major linear algebra for demos. Matrices match GLSL's mat4 memory layout,
// so `glUniformMatrix4fv(loc, 1, GL_FALSE, m.data())` uploads them unchanged.
// (OpenGL ES requires transpose == GL_FALSE for glUniformMatrix*.)
#pragma once

#include <array>
#include <cmath>
#include <numbers>

namespace demo::math
{
    struct Vec3
    {
        float x = 0, y = 0, z = 0;
        friend constexpr Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
        friend constexpr Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
        friend constexpr Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
    };

    struct Vec4
    {
        float x = 0, y = 0, z = 0, w = 0;
    };

    [[nodiscard]] constexpr float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    [[nodiscard]] constexpr Vec3 cross(Vec3 a, Vec3 b)
    {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }
    [[nodiscard]] inline Vec3 normalize(Vec3 v)
    {
        const float len = std::sqrt(dot(v, v));
        return len > 0.0f ? v * (1.0f / len) : v;
    }
    [[nodiscard]] constexpr float radians(float degrees) { return degrees * std::numbers::pi_v<float> / 180.0f; }

    /// Column-major 4x4: element (row r, column c) is m[c * 4 + r].
    struct Mat4
    {
        std::array<float, 16> m{};

        [[nodiscard]] static constexpr Mat4 identity()
        {
            Mat4 r;
            r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
            return r;
        }
        [[nodiscard]] constexpr float& at(int row, int col) { return m[static_cast<std::size_t>(col * 4 + row)]; }
        [[nodiscard]] constexpr float at(int row, int col) const { return m[static_cast<std::size_t>(col * 4 + row)]; }
        [[nodiscard]] const float* data() const { return m.data(); }

        friend constexpr Mat4 operator*(const Mat4& a, const Mat4& b)
        {
            Mat4 r;
            for (int c = 0; c < 4; ++c)
                for (int row = 0; row < 4; ++row)
                {
                    float s = 0.0f;
                    for (int k = 0; k < 4; ++k) s += a.at(row, k) * b.at(k, c);
                    r.at(row, c) = s;
                }
            return r;
        }
        friend constexpr Vec4 operator*(const Mat4& a, Vec4 v)
        {
            const float in[4] = {v.x, v.y, v.z, v.w};
            float out[4] = {};
            for (int row = 0; row < 4; ++row)
                for (int k = 0; k < 4; ++k) out[row] += a.at(row, k) * in[k];
            return {out[0], out[1], out[2], out[3]};
        }
    };

    [[nodiscard]] constexpr Mat4 translate(Vec3 t)
    {
        Mat4 r = Mat4::identity();
        r.at(0, 3) = t.x;
        r.at(1, 3) = t.y;
        r.at(2, 3) = t.z;
        return r;
    }

    [[nodiscard]] constexpr Mat4 scale(Vec3 s)
    {
        Mat4 r = Mat4::identity();
        r.at(0, 0) = s.x;
        r.at(1, 1) = s.y;
        r.at(2, 2) = s.z;
        return r;
    }

    /// Rotation by `angle` radians around a (normalized) axis - right-handed.
    [[nodiscard]] inline Mat4 rotate(float angle, Vec3 axis)
    {
        const Vec3 a = normalize(axis);
        const float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
        Mat4 r = Mat4::identity();
        r.at(0, 0) = c + a.x * a.x * t;       r.at(0, 1) = a.x * a.y * t - a.z * s; r.at(0, 2) = a.x * a.z * t + a.y * s;
        r.at(1, 0) = a.y * a.x * t + a.z * s; r.at(1, 1) = c + a.y * a.y * t;       r.at(1, 2) = a.y * a.z * t - a.x * s;
        r.at(2, 0) = a.z * a.x * t - a.y * s; r.at(2, 1) = a.z * a.y * t + a.x * s; r.at(2, 2) = c + a.z * a.z * t;
        return r;
    }

    /// OpenGL-convention perspective projection: right-handed eye space, NDC z in [-1, 1].
    [[nodiscard]] inline Mat4 perspective(float fovy_radians, float aspect, float z_near, float z_far)
    {
        const float f = 1.0f / std::tan(fovy_radians * 0.5f);
        Mat4 r;
        r.at(0, 0) = f / aspect;
        r.at(1, 1) = f;
        r.at(2, 2) = (z_far + z_near) / (z_near - z_far);
        r.at(2, 3) = (2.0f * z_far * z_near) / (z_near - z_far);
        r.at(3, 2) = -1.0f;
        return r;
    }

    [[nodiscard]] constexpr Mat4 ortho(float l, float r_, float b, float t, float n, float f)
    {
        Mat4 r = Mat4::identity();
        r.at(0, 0) = 2.0f / (r_ - l);
        r.at(1, 1) = 2.0f / (t - b);
        r.at(2, 2) = -2.0f / (f - n);
        r.at(0, 3) = -(r_ + l) / (r_ - l);
        r.at(1, 3) = -(t + b) / (t - b);
        r.at(2, 3) = -(f + n) / (f - n);
        return r;
    }

    [[nodiscard]] inline Mat4 look_at(Vec3 eye, Vec3 center, Vec3 up)
    {
        const Vec3 f = normalize(center - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 r = Mat4::identity();
        r.at(0, 0) = s.x;  r.at(0, 1) = s.y;  r.at(0, 2) = s.z;  r.at(0, 3) = -dot(s, eye);
        r.at(1, 0) = u.x;  r.at(1, 1) = u.y;  r.at(1, 2) = u.z;  r.at(1, 3) = -dot(u, eye);
        r.at(2, 0) = -f.x; r.at(2, 1) = -f.y; r.at(2, 2) = -f.z; r.at(2, 3) = dot(f, eye);
        return r;
    }

    /// Upper-left 3x3 inverse-transpose (normal matrix), column-major 9 floats.
    [[nodiscard]] inline std::array<float, 9> normal_matrix(const Mat4& m)
    {
        const float a = m.at(0, 0), b = m.at(0, 1), c = m.at(0, 2);
        const float d = m.at(1, 0), e = m.at(1, 1), f = m.at(1, 2);
        const float g = m.at(2, 0), h = m.at(2, 1), i = m.at(2, 2);
        const float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        const float inv = det != 0.0f ? 1.0f / det : 0.0f;
        // inverse (row-major cofactors / det), then transpose; stored column-major.
        // (inverse)^T element (r, c) = cofactor(r, c) / det.
        std::array<float, 9> n{};
        auto set = [&](int row, int col, float v) { n[static_cast<std::size_t>(col * 3 + row)] = v * inv; };
        set(0, 0, e * i - f * h);  set(0, 1, -(d * i - f * g)); set(0, 2, d * h - e * g);
        set(1, 0, -(b * i - c * h)); set(1, 1, a * i - c * g);  set(1, 2, -(a * h - b * g));
        set(2, 0, b * f - c * e);  set(2, 1, -(a * f - c * d)); set(2, 2, a * e - b * d);
        return n;
    }
}
