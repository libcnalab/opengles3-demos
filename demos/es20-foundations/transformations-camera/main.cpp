// es20-transformations-camera
//
// The entry point of the 3D path: a small depth-tested scene (checkerboard ground, a tumbling cube
// with one color per face, a moon cube that orbits it, four identical posts, a pyramid and an axis
// gizmo) drawn through the full coordinate chain
//
//     object --M--> world --V--> eye --P--> clip --/w--> NDC --viewport--> window
//
// with model / view / projection matrices built on the CPU (demo/math.hpp) and uploaded with
// glUniformMatrix4fv. The window shows the SAME scene and the SAME camera twice: once through a
// perspective projection and once through an orthographic one (stacked, or side by side in a wide
// window). The near plane is deliberately placed inside the scene, so the foreground is visibly cut
// off by the clip volume.
//
// validate() contains an independent CPU rasterizer of the scene (clip against near/far, divide,
// viewport, edge functions, z-buffer) and compares it with what the GPU drew: every pixel that is not
// within 0.6 px of a triangle edge, in both panes, plus named probes (face centres, points near cube
// corners, ground tile centres, hidden faces), the pixel heights of identical posts at different
// distances (perspective shrinks them, orthographic does not), the window position of the near-plane
// cut, a depth-test on/off comparison, re-renders with a different near and far plane, matrix upload
// with transpose = GL_TRUE, and negative controls (wrong rotation angle, wrong projection, and a
// transposed matrix uploaded with transpose = GL_FALSE).
//
// Interactive (--window): W/S dolly, A/D orbit, Q/E elevation, P projection mode, N/M near plane,
// F/G far plane, R reset.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;
namespace dm = demo::math;

namespace
{
    using dm::Mat4;
    using dm::Vec3;
    using dm::Vec4;

    // ---------------------------------------------------------------------------------------------
    // Constants shared by the GPU path and the CPU reference
    // ---------------------------------------------------------------------------------------------
    constexpr float kFovY = dm::radians(40.0f);
    constexpr float kDefaultNear = 7.6f;   // inside the scene on purpose: the foreground is clipped away
    constexpr float kDefaultFar = 60.0f;
    constexpr float kAmbient = 0.38f;
    constexpr float kCubeHalf = 0.8f;
    constexpr float kCubeY = 1.9f;
    constexpr float kPostHalfWidth = 0.25f;
    constexpr float kPostHeight = 2.4f;
    constexpr int kGroundTiles = 15;       // odd: the centre tile is centred on the world origin
    // Tolerances of the comparison with the CPU reference, with their justification:
    //  - kEdgeMargin: ES 3.0 3.6.1 ("point sampling") produces a fragment where the pixel centre is inside the
    //    polygon, but the vertex positions are snapped to a subpixel grid (SUBPIXEL_BITS >= 4) and computed in
    //    fp32. Reference pixels whose centre is closer than 0.6 px to a triangle edge (>> 1/16 px) are "unsafe" and
    //    not compared; everything else must agree.
    //  - kDepthEps: two surfaces whose window-space depths differ by less than this are not ordered reliably by a
    //    24-bit depth buffer fed with fp32 depths; such pixels are "unsafe" too.
    //  - kTol: ES 3.0 2.1.6.2 converts a float color to "one of the two unsigned binary integers closest to f * 255"
    //    (+-1), and the shader evaluates the shading in fp32 where the reference uses fp64 (+-1 more).
    constexpr double kEdgeMargin = 0.6;
    constexpr double kDepthEps = 1e-4;
    constexpr int kTol = 2;

    constexpr Vec3 kPerspClear{0.08f, 0.11f, 0.20f};
    constexpr Vec3 kOrthoClear{0.13f, 0.09f, 0.19f};
    constexpr Vec3 kSeparator{0.78f, 0.80f, 0.85f};
    constexpr Vec3 kGizmoClear{0.03f, 0.04f, 0.07f};

    [[nodiscard]] constexpr double dbl(float v) { return static_cast<double>(v); }

    [[nodiscard]] Vec3 light_dir() { return dm::normalize({0.45f, 0.85f, 0.30f}); }

    [[nodiscard]] Mat4 transposed(const Mat4& a)
    {
        Mat4 r;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col) r.at(row, col) = a.at(col, row);
        return r;
    }

    [[nodiscard]] std::uint8_t to8(double v)
    {
        return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
    }

    using Rgb8 = std::array<std::uint8_t, 3>;

    struct Rect
    {
        int x = 0, y = 0, w = 0, h = 0;  // window coordinates, origin bottom-left (GL convention)
        [[nodiscard]] bool contains(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; }
    };

    // ---------------------------------------------------------------------------------------------
    // Meshes: plain triangle lists, position + normal + color (36 bytes per vertex)
    // ---------------------------------------------------------------------------------------------
    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
        float r, g, b;
    };
    static_assert(sizeof(Vertex) == 36);

    struct MeshRange
    {
        int first = 0;  // first vertex
        int count = 0;  // number of vertices (a multiple of 3)
    };

    struct FaceAxes
    {
        Vec3 n, u, v;  // outward normal and two tangents with u x v = n (counter-clockwise seen from outside)
    };
    constexpr std::array<FaceAxes, 6> kFaces = {{
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},    // +X
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},   // -X
        {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},    // +Y
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   // -Y
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},    // +Z
        {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},   // -Z
    }};
    constexpr std::array<Vec3, 6> kCubeColors = {{
        {0.90f, 0.25f, 0.25f},  // +X red
        {0.25f, 0.75f, 0.85f},  // -X cyan
        {0.35f, 0.80f, 0.35f},  // +Y green
        {0.85f, 0.35f, 0.80f},  // -Y magenta
        {0.30f, 0.45f, 0.95f},  // +Z blue
        {0.95f, 0.80f, 0.25f},  // -Z yellow
    }};
    constexpr Vec3 kPostColor{0.95f, 0.50f, 0.10f};

    [[nodiscard]] constexpr Vec3 neg(Vec3 v) { return {-v.x, -v.y, -v.z}; }
    [[nodiscard]] constexpr float abs1(float v) { return v < 0.0f ? -v : v; }
    [[nodiscard]] constexpr Vec3 abs3(Vec3 v) { return {abs1(v.x), abs1(v.y), abs1(v.z)}; }

    class Mesh
    {
    public:
        std::vector<Vertex> vertices;

        /// An axis-aligned box. `with_bottom = false` leaves out the -Y face (boxes standing on the ground).
        MeshRange add_box(Vec3 centre, Vec3 half, const std::array<Vec3, 6>& colors, bool with_bottom = true)
        {
            const int first = static_cast<int>(vertices.size());
            for (std::size_t f = 0; f < 6; ++f)
            {
                if (f == 3 && !with_bottom) continue;
                const FaceAxes& a = kFaces[f];
                const float hn = dm::dot(abs3(a.n), half), hu = dm::dot(abs3(a.u), half), hv = dm::dot(abs3(a.v), half);
                const Vec3 c = centre + a.n * hn;
                const Vec3 p0 = c + a.u * (-hu) + a.v * (-hv), p1 = c + a.u * hu + a.v * (-hv);
                const Vec3 p2 = c + a.u * hu + a.v * hv, p3 = c + a.u * (-hu) + a.v * hv;
                add_triangle(p0, p1, p2, a.n, colors[f]);
                add_triangle(p0, p2, p3, a.n, colors[f]);
            }
            return {first, static_cast<int>(vertices.size()) - first};
        }

        MeshRange add_box(Vec3 centre, Vec3 half, Vec3 color, bool with_bottom = true)
        {
            return add_box(centre, half, std::array<Vec3, 6>{color, color, color, color, color, color}, with_bottom);
        }

        /// Square pyramid standing on y = 0 (no base face), one color per side.
        MeshRange add_pyramid(float half_base, float height, const std::array<Vec3, 4>& colors)
        {
            const int first = static_cast<int>(vertices.size());
            const Vec3 apex{0, height, 0};
            const std::array<Vec3, 4> base = {{{-half_base, 0, -half_base}, {half_base, 0, -half_base},
                                               {half_base, 0, half_base}, {-half_base, 0, half_base}}};
            const Vec3 inside{0, height * 0.25f, 0};
            for (std::size_t i = 0; i < 4; ++i)
            {
                Vec3 a = base[i], b = base[(i + 1) % 4];
                Vec3 n = dm::normalize(dm::cross(b - a, apex - a));
                if (dm::dot(n, a - inside) < 0.0f)
                {
                    std::swap(a, b);
                    n = neg(n);
                }
                add_triangle(a, b, apex, n, colors[i]);
            }
            return {first, static_cast<int>(vertices.size()) - first};
        }

        /// A checkerboard of `tiles` x `tiles` unit squares on y = 0. The row through the origin along X is tinted
        /// red and the column along Z blue: they are the world axes.
        MeshRange add_ground(int tiles)
        {
            const int first = static_cast<int>(vertices.size());
            const int half = tiles / 2;
            for (int j = -half; j <= half; ++j)
                for (int i = -half; i <= half; ++i)
                {
                    const float x0 = static_cast<float>(i) - 0.5f, x1 = x0 + 1.0f;
                    const float z0 = static_cast<float>(j) - 0.5f, z1 = z0 + 1.0f;
                    const Vec3 col = ground_color(i, j);
                    add_triangle({x0, 0, z0}, {x0, 0, z1}, {x1, 0, z1}, {0, 1, 0}, col);
                    add_triangle({x0, 0, z0}, {x1, 0, z1}, {x1, 0, z0}, {0, 1, 0}, col);
                }
            return {first, static_cast<int>(vertices.size()) - first};
        }

        /// A convex polygon in the z = z0 plane, fan-triangulated (used by the pane icons).
        void add_polygon_xy(std::initializer_list<std::pair<float, float>> pts, float z0, Vec3 col)
        {
            const std::vector<std::pair<float, float>> p(pts);
            for (std::size_t k = 1; k + 1 < p.size(); ++k)
                add_triangle({p[0].first, p[0].second, z0}, {p[k].first, p[k].second, z0}, {p[k + 1].first, p[k + 1].second, z0},
                             {0, 0, 1}, col);
        }

        /// Side view of a view volume: the near plane on the left, the far plane on the right; a wedge for a
        /// perspective projection, a box for an orthographic one.
        MeshRange add_volume_icon(bool ortho)
        {
            const int first = static_cast<int>(vertices.size());
            const Vec3 fill{0.30f, 0.38f, 0.62f}, bar{1.0f, 0.85f, 0.30f};
            if (ortho)
                add_polygon_xy({{-0.8f, -0.5f}, {0.8f, -0.5f}, {0.8f, 0.5f}, {-0.8f, 0.5f}}, -0.1f, fill);
            else
                add_polygon_xy({{-0.8f, -0.15f}, {0.8f, -0.75f}, {0.8f, 0.75f}, {-0.8f, 0.15f}}, -0.1f, fill);
            const float near_h = ortho ? 0.62f : 0.28f, far_h = ortho ? 0.62f : 0.88f;
            add_polygon_xy({{-0.88f, -near_h}, {-0.74f, -near_h}, {-0.74f, near_h}, {-0.88f, near_h}}, 0.1f, bar);
            add_polygon_xy({{0.74f, -far_h}, {0.88f, -far_h}, {0.88f, far_h}, {0.74f, far_h}}, 0.1f, bar);
            return {first, static_cast<int>(vertices.size()) - first};
        }

        [[nodiscard]] static Vec3 ground_color(int i, int j)
        {
            if (i == 0 && j == 0) return {0.80f, 0.80f, 0.86f};
            if (j == 0) return ((i + 64) % 2 == 0) ? Vec3{0.62f, 0.30f, 0.28f} : Vec3{0.46f, 0.22f, 0.21f};
            if (i == 0) return ((j + 64) % 2 == 0) ? Vec3{0.28f, 0.38f, 0.64f} : Vec3{0.21f, 0.29f, 0.48f};
            return ((i + j + 64) % 2 == 0) ? Vec3{0.34f, 0.40f, 0.36f} : Vec3{0.22f, 0.27f, 0.25f};
        }

    private:
        void add_triangle(Vec3 a, Vec3 b, Vec3 c, Vec3 n, Vec3 col)
        {
            for (const Vec3 p : {a, b, c}) vertices.push_back({p.x, p.y, p.z, n.x, n.y, n.z, col.x, col.y, col.z});
        }
    };

    // ---------------------------------------------------------------------------------------------
    // Camera, panes, scene description
    // ---------------------------------------------------------------------------------------------
    struct Camera
    {
        float azimuth = dm::radians(32.0f);
        float elevation = dm::radians(17.0f);
        float radius = 9.5f;
        Vec3 target{0.0f, 1.5f, 0.0f};

        [[nodiscard]] Vec3 eye() const
        {
            const float ce = std::cos(elevation);
            return target + Vec3{std::sin(azimuth) * ce, std::sin(elevation), std::cos(azimuth) * ce} * radius;
        }
        [[nodiscard]] Mat4 view() const { return dm::look_at(eye(), target, {0, 1, 0}); }
    };

    enum class Mode { Split, Perspective, Orthographic };

    struct Pane
    {
        Rect rect;
        bool ortho = false;
        Vec3 clear;
        std::string_view name;
    };

    [[nodiscard]] std::vector<Pane> make_panes(int w, int h, Mode mode)
    {
        constexpr int g = 1;  // gutter: the separator color of the window shows through
        if (mode == Mode::Perspective) return {{{g, g, w - 2 * g, h - 2 * g}, false, kPerspClear, "perspective"}};
        if (mode == Mode::Orthographic) return {{{g, g, w - 2 * g, h - 2 * g}, true, kOrthoClear, "orthographic"}};
        if (w >= h * 2)  // very wide window: side by side (otherwise the panes would be extremely flat)
        {
            const int half = w / 2;
            return {{{g, g, half - 2 * g, h - 2 * g}, false, kPerspClear, "perspective"},
                    {{half + g, g, w - half - 2 * g, h - 2 * g}, true, kOrthoClear, "orthographic"}};
        }
        const int half = h / 2;  // otherwise stacked: perspective on top
        return {{{g, half + g, w - 2 * g, h - half - 2 * g}, false, kPerspClear, "perspective"},
                {{g, g, w - 2 * g, half - 2 * g}, true, kOrthoClear, "orthographic"}};
    }

    /// The axis gizmo occupies a square in the bottom-right corner of its pane.
    [[nodiscard]] Rect gizmo_rect(const Rect& r)
    {
        const int s = std::clamp(std::min(r.w, r.h) * 26 / 100, 20, 120);
        return {r.x + r.w - s - 5, r.y + 5, s, s};
    }

    /// A small badge in the top-left corner shows the side view of the view volume: a wedge (perspective) or a box.
    [[nodiscard]] Rect icon_rect(const Rect& r)
    {
        const int s = std::clamp(std::min(r.w, r.h) / 5, 12, 44);
        return {r.x + 5, r.y + r.h - s - 5, 2 * s, s};
    }

    /// Vertical field of view of a pane: kFovY, widened for panes that are not much wider than tall so that the
    /// horizontal field of view never drops below that of a 1.2:1 pane.
    [[nodiscard]] float effective_fovy(float aspect)
    {
        const float t = std::tan(kFovY * 0.5f);
        return 2.0f * std::atan(std::max(t, 1.2f * t / aspect));
    }

    [[nodiscard]] Mat4 pane_projection(const Pane& pane, const Camera& cam, float n, float f)
    {
        const float aspect = static_cast<float>(pane.rect.w) / static_cast<float>(pane.rect.h);
        const float fovy = effective_fovy(aspect);
        if (!pane.ortho) return dm::perspective(fovy, aspect, n, f);
        // The orthographic box is as tall as the perspective frustum is at the orbit target: both
        // projections show the target at the same size, everything else differs.
        const float half_h = cam.radius * std::tan(fovy * 0.5f);
        return dm::ortho(-half_h * aspect, half_h * aspect, -half_h, half_h, n, f);
    }

    /// Rotation part of the view only: the gizmo shows the orientation of the world axes in the eye frame.
    [[nodiscard]] Mat4 gizmo_view(Mat4 view)
    {
        view.at(0, 3) = view.at(1, 3) = view.at(2, 3) = 0.0f;
        return view;
    }
    [[nodiscard]] Mat4 gizmo_projection() { return dm::ortho(-1.25f, 1.25f, -1.25f, 1.25f, -3.0f, 3.0f); }
    [[nodiscard]] Mat4 icon_projection() { return dm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f); }

    enum class Upload
    {
        ColumnMajor,        // the normal way: column-major data, transpose = GL_FALSE
        RowMajorTranspose,  // row-major data, transpose = GL_TRUE (legal in ES 3.0)
        WrongTranspose,     // row-major data, transpose = GL_FALSE: a bug, used as a negative control
    };

    enum class DepthMode
    {
        Off,      // glDisable(GL_DEPTH_TEST)
        Less,     // GL_LESS (the initial function), depth buffer cleared to 1.0: the nearest surface wins
        Greater,  // GL_GREATER, depth buffer cleared to 0.0: the FARTHEST surface wins
    };

    struct Params
    {
        Camera camera;
        Mode mode = Mode::Split;
        float near_plane = kDefaultNear;
        float far_plane = kDefaultFar;
        DepthMode depth = DepthMode::Less;
        Upload upload = Upload::ColumnMajor;
    };

    struct Item
    {
        std::string_view name;
        MeshRange mesh;
        Mat4 model;
        bool lit = true;
    };

    enum ItemIndex : std::size_t
    {
        kGround = 0,
        kPost0 = 1,  // kPost0 .. kPost0 + 3
        kPyramid = 5,
        kCube = 6,
        kMoon = 7,
    };

    struct Ranges
    {
        MeshRange ground, cube, post, pyramid, icon_persp, icon_ortho;
        std::array<MeshRange, 3> axis;
    };

    /// A point on the ground (y = 0) `h` units ahead of the default camera (horizontally) and `lateral` to its right.
    [[nodiscard]] Vec3 ground_point(const Camera& cam, float h, float lateral)
    {
        const Vec3 e = cam.eye();
        const Vec3 fwd{-std::sin(cam.azimuth), 0, -std::cos(cam.azimuth)};
        const Vec3 right{std::cos(cam.azimuth), 0, -std::sin(cam.azimuth)};
        return Vec3{e.x, 0, e.z} + fwd * h + right * lateral;
    }

    struct Layout
    {
        std::array<Vec3, 4> posts{};
        Vec3 pyramid;
    };

    /// World placement of the posts and the pyramid, derived from the default camera. Post 0 is placed so that
    /// the default near plane cuts it at mid-height; posts 1..3 are farther away along a receding line.
    [[nodiscard]] Layout make_layout()
    {
        const Camera cam;
        const float sin_e = std::sin(cam.elevation), cos_e = std::cos(cam.elevation);
        // Eye-space depth of a point at horizontal distance h and height y: d = h cos(el) + (eye_y - y) sin(el).
        const float h0 = (kDefaultNear - (cam.eye().y - kPostHeight * 0.5f) * sin_e) / cos_e;
        Layout l;
        // Receding line of identical posts; the lateral offsets keep their pictures from overlapping.
        constexpr std::array<float, 4> lateral = {-3.9f, -3.3f, -2.8f, -2.4f};
        for (std::size_t k = 0; k < 4; ++k) l.posts[k] = ground_point(cam, h0 + 2.4f * static_cast<float>(k), lateral[k]);
        l.pyramid = ground_point(cam, 10.6f, 3.7f);
        return l;
    }

    [[nodiscard]] Mat4 cube_pose(double t, float extra_angle)
    {
        const float spin = static_cast<float>(0.8 * t) + extra_angle;
        return dm::translate({0, kCubeY, 0}) * dm::rotate(spin, {0, 1, 0}) * dm::rotate(0.55f, {1, 0, 0});
    }

    [[nodiscard]] std::vector<Item> make_items(const Ranges& r, const Layout& layout, double t, float extra_angle = 0.0f)
    {
        std::vector<Item> items;
        items.push_back({"ground", r.ground, Mat4::identity(), true});
        for (std::size_t k = 0; k < 4; ++k)
            items.push_back({"post", r.post, dm::translate(layout.posts[k]), true});
        items.push_back({"pyramid", r.pyramid, dm::translate(layout.pyramid), true});
        const Mat4 cube = cube_pose(t, extra_angle);
        items.push_back({"cube", r.cube, cube, true});
        // A model matrix is a product: the moon spins, moves out along x, orbits the cube's centre (the
        // translation part of the cube pose) and is 0.28 times the size of the cube mesh.
        const Mat4 moon = dm::translate({0, kCubeY, 0}) * dm::rotate(static_cast<float>(1.7 * t), {0, 1, 0}) *
                          dm::translate({2.3f, 0.3f, 0}) * dm::rotate(static_cast<float>(2.5 * t), {1, 1, 0}) *
                          dm::scale({0.28f, 0.28f, 0.28f});
        items.push_back({"moon", r.cube, moon, true});
        return items;
    }

    [[nodiscard]] std::vector<Item> make_gizmo_items(const Ranges& r)
    {
        return {{"axis-x", r.axis[0], Mat4::identity(), false},
                {"axis-y", r.axis[1], Mat4::identity(), false},
                {"axis-z", r.axis[2], Mat4::identity(), false}};
    }

    [[nodiscard]] std::vector<Item> make_icon_items(const Ranges& r, bool ortho)
    {
        return {{"icon", ortho ? r.icon_ortho : r.icon_persp, Mat4::identity(), false}};
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference: a tiny software rasterizer that mimics the fixed-function stages
    // (ES 3.0 2.13 coordinate transformations, 2.18 clipping, 3.6.1 polygon rasterization, 4.1.5 depth test)
    // ---------------------------------------------------------------------------------------------
    struct RefPixel
    {
        double depth = 2.0;  // window z in [0, 1]
        Rgb8 rgb{};
        int item = -1;       // -1 = background
        int tri = -1;        // triangle index within the item's mesh range
        bool unsafe = false; // an edge, or a depth tie, makes the GPU result implementation-dependent here
    };

    struct RefImage
    {
        Rect rect;
        std::vector<RefPixel> px;
        [[nodiscard]] RefPixel& at(int x, int y) { return px[static_cast<std::size_t>((y - rect.y) * rect.w + (x - rect.x))]; }
        [[nodiscard]] const RefPixel& at(int x, int y) const
        {
            return px[static_cast<std::size_t>((y - rect.y) * rect.w + (x - rect.x))];
        }
    };

    struct ClipV
    {
        double x, y, z, w;
    };

    [[nodiscard]] ClipV lerp_clip(const ClipV& a, const ClipV& b, double t)
    {
        return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
    }

    /// Sutherland-Hodgman against the near (z >= -w) and far (z <= w) planes of the clip volume. The x/y planes
    /// need no clipping: pixels are tested individually against the pane.
    template <typename Dist>
    [[nodiscard]] std::vector<ClipV> clip_against(const std::vector<ClipV>& in, Dist dist)
    {
        std::vector<ClipV> out;
        const std::size_t n = in.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const ClipV& cur = in[i];
            const ClipV& prev = in[(i + n - 1) % n];
            const double dc = dist(cur), dp = dist(prev);
            if (dc >= 0.0)
            {
                if (dp < 0.0) out.push_back(lerp_clip(prev, cur, dp / (dp - dc)));
                out.push_back(cur);
            }
            else if (dp >= 0.0)
            {
                out.push_back(lerp_clip(prev, cur, dp / (dp - dc)));
            }
        }
        return out;
    }

    [[nodiscard]] std::vector<ClipV> clip_near_far(std::vector<ClipV> poly)
    {
        poly = clip_against(poly, [](const ClipV& v) { return v.z + v.w; });
        if (poly.size() >= 3) poly = clip_against(poly, [](const ClipV& v) { return v.w - v.z; });
        return poly;
    }

    struct WinV
    {
        double x, y, z;
    };

    struct Band
    {
        std::size_t index;
        double z;
        Rgb8 rgb;
    };

    void raster_triangle(RefImage& img, const std::array<WinV, 3>& v, Rgb8 rgb, int item, int tri, DepthMode depth,
                         std::vector<Band>& bands)
    {
        const double area2 = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
        if (std::abs(area2) < 1e-9) return;
        auto len = [&](int a, int b) { return std::hypot(v[static_cast<std::size_t>(b)].x - v[static_cast<std::size_t>(a)].x,
                                                         v[static_cast<std::size_t>(b)].y - v[static_cast<std::size_t>(a)].y); };
        const double l0 = len(1, 2), l1 = len(2, 0), l2 = len(0, 1);
        const double min_x = std::min({v[0].x, v[1].x, v[2].x}), max_x = std::max({v[0].x, v[1].x, v[2].x});
        const double min_y = std::min({v[0].y, v[1].y, v[2].y}), max_y = std::max({v[0].y, v[1].y, v[2].y});
        const int x0 = std::max(img.rect.x, static_cast<int>(std::floor(min_x - kEdgeMargin)));
        const int x1 = std::min(img.rect.x + img.rect.w - 1, static_cast<int>(std::ceil(max_x + kEdgeMargin)));
        const int y0 = std::max(img.rect.y, static_cast<int>(std::floor(min_y - kEdgeMargin)));
        const int y1 = std::min(img.rect.y + img.rect.h - 1, static_cast<int>(std::ceil(max_y + kEdgeMargin)));
        const double abs_area = std::abs(area2);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                const double px = x + 0.5, py = y + 0.5;
                auto edge = [&](std::size_t a, std::size_t b) {
                    return (v[b].x - v[a].x) * (py - v[a].y) - (v[b].y - v[a].y) * (px - v[a].x);
                };
                const double w0 = edge(1, 2) / area2, w1 = edge(2, 0) / area2, w2 = edge(0, 1) / area2;
                // distance (in pixels) from the pixel centre to each edge line, positive inside
                const double m = std::min({w0 * abs_area / l0, w1 * abs_area / l1, w2 * abs_area / l2});
                if (m < -kEdgeMargin) continue;
                const double z = w0 * v[0].z + w1 * v[1].z + w2 * v[2].z;
                RefPixel& p = img.at(x, y);
                if (m < kEdgeMargin)
                {
                    bands.push_back({static_cast<std::size_t>((y - img.rect.y) * img.rect.w + (x - img.rect.x)), z, rgb});
                    continue;
                }
                if (depth != DepthMode::Off)
                {
                    if (p.item != -1 && std::abs(z - p.depth) < kDepthEps && p.rgb != rgb) p.unsafe = true;
                    // ES 3.0 4.1.5: the fragment passes if its depth is less / greater than the stored one
                    if (depth == DepthMode::Less ? z >= p.depth : z <= p.depth) continue;
                }
                p.depth = z;
                p.rgb = rgb;
                p.item = item;
                p.tri = tri;
            }
    }

    /// The color the vertex shader produces for a flat-colored triangle (see scene.vert), computed in double.
    [[nodiscard]] Rgb8 shaded_rgb(const Mat4& model, const Vertex& v, bool lit)
    {
        double s = 1.0;
        if (lit)
        {
            const double nx = dbl(model.at(0, 0)) * dbl(v.nx) + dbl(model.at(0, 1)) * dbl(v.ny) + dbl(model.at(0, 2)) * dbl(v.nz);
            const double ny = dbl(model.at(1, 0)) * dbl(v.nx) + dbl(model.at(1, 1)) * dbl(v.ny) + dbl(model.at(1, 2)) * dbl(v.nz);
            const double nz = dbl(model.at(2, 0)) * dbl(v.nx) + dbl(model.at(2, 1)) * dbl(v.ny) + dbl(model.at(2, 2)) * dbl(v.nz);
            const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
            const Vec3 l = light_dir();
            const double diffuse = std::max((nx * dbl(l.x) + ny * dbl(l.y) + nz * dbl(l.z)) / len, 0.0);
            s = dbl(kAmbient) + (1.0 - dbl(kAmbient)) * diffuse;
        }
        return {to8(dbl(v.r) * s), to8(dbl(v.g) * s), to8(dbl(v.b) * s)};
    }

    [[nodiscard]] RefImage reference_render(const std::vector<Vertex>& verts, std::span<const Item> items, const Mat4& view,
                                            const Mat4& proj, const Rect& rect, Vec3 clear, DepthMode depth)
    {
        RefImage img;
        img.rect = rect;
        RefPixel background;
        background.depth = depth == DepthMode::Greater ? 0.0 : 2.0;  // the cleared depth buffer (2.0: "farther than 1")
        background.rgb = {to8(dbl(clear.x)), to8(dbl(clear.y)), to8(dbl(clear.z))};
        img.px.assign(static_cast<std::size_t>(rect.w * rect.h), background);
        std::vector<Band> bands;

        for (std::size_t ii = 0; ii < items.size(); ++ii)
        {
            const Item& it = items[ii];
            for (int t = 0; t < it.mesh.count; t += 3)
            {
                std::vector<ClipV> poly;
                for (int k = 0; k < 3; ++k)
                {
                    const Vertex& v = verts[static_cast<std::size_t>(it.mesh.first + t + k)];
                    // The same three steps as scene.vert, on the same float matrices.
                    const Vec4 world = it.model * Vec4{v.px, v.py, v.pz, 1.0f};
                    const Vec4 eye = view * world;
                    const Vec4 clip = proj * eye;
                    poly.push_back({dbl(clip.x), dbl(clip.y), dbl(clip.z), dbl(clip.w)});
                }
                const Rgb8 rgb = shaded_rgb(it.model, verts[static_cast<std::size_t>(it.mesh.first + t)], it.lit);
                poly = clip_near_far(std::move(poly));
                if (poly.size() < 3) continue;
                std::vector<WinV> win;
                for (const ClipV& c : poly)
                {
                    // perspective divide, then the viewport transformation (ES 3.0 2.13.1) with depth range [0, 1]
                    const double xd = c.x / c.w, yd = c.y / c.w, zd = c.z / c.w;
                    win.push_back({rect.x + (xd + 1.0) * 0.5 * rect.w, rect.y + (yd + 1.0) * 0.5 * rect.h, (zd + 1.0) * 0.5});
                }
                for (std::size_t k = 1; k + 1 < win.size(); ++k)
                    raster_triangle(img, {win[0], win[k], win[k + 1]}, rgb, static_cast<int>(ii), t / 3, depth, bands);
            }
        }
        for (const Band& b : bands)
        {
            RefPixel& p = img.px[b.index];
            const bool could_show = depth == DepthMode::Off       ? true
                                    : depth == DepthMode::Less    ? b.z <= p.depth + kDepthEps
                                                                  : b.z >= p.depth - kDepthEps;
            if (could_show && b.rgb != p.rgb) p.unsafe = true;
        }
        return img;
    }

    void overlay(RefImage& base, const RefImage& top)
    {
        for (int y = top.rect.y; y < top.rect.y + top.rect.h; ++y)
            for (int x = top.rect.x; x < top.rect.x + top.rect.w; ++x)
                if (base.rect.contains(x, y)) base.at(x, y) = top.at(x, y);
    }

    struct PaneRef
    {
        Pane pane;
        Mat4 view, proj;
        Rect gizmo, icon;
        RefImage img;
    };

    [[nodiscard]] PaneRef build_pane_ref(const std::vector<Vertex>& verts, std::span<const Item> items,
                                         std::span<const Item> gizmo_items, std::span<const Item> icon_items, const Pane& pane,
                                         const Camera& cam, float n, float f, DepthMode depth)
    {
        PaneRef pr;
        pr.pane = pane;
        pr.view = cam.view();
        pr.proj = pane_projection(pane, cam, n, f);
        pr.gizmo = gizmo_rect(pane.rect);
        pr.icon = icon_rect(pane.rect);
        pr.img = reference_render(verts, items, pr.view, pr.proj, pane.rect, pane.clear, depth);
        overlay(pr.img, reference_render(verts, gizmo_items, gizmo_view(pr.view), gizmo_projection(), pr.gizmo, kGizmoClear, DepthMode::Less));
        overlay(pr.img, reference_render(verts, icon_items, Mat4::identity(), icon_projection(), pr.icon, kGizmoClear, DepthMode::Less));
        return pr;
    }

    // ---------------------------------------------------------------------------------------------
    // Reading and comparing the GPU image
    // ---------------------------------------------------------------------------------------------
    struct Image
    {
        int w = 0, h = 0;
        std::vector<std::uint8_t> data;  // RGBA8, rows bottom-up like glReadPixels
        [[nodiscard]] static Image grab(int w, int h)
        {
            Image i;
            i.w = w;
            i.h = h;
            i.data = demo::read_rgba8(0, 0, w, h);
            return i;
        }
        [[nodiscard]] demo::Rgba8 at(int x, int y) const
        {
            const std::size_t o = static_cast<std::size_t>((y * w + x) * 4);
            return {data[o], data[o + 1], data[o + 2], data[o + 3]};
        }
    };

    [[nodiscard]] int rgb_distance(const demo::Rgba8& got, const Rgb8& want)
    {
        int e = 0;
        for (std::size_t c = 0; c < 3; ++c) e = std::max(e, std::abs(int{got[c]} - int{want[c]}));
        return e;
    }

    struct Compare
    {
        int total = 0, compared = 0, unsafe = 0, bad = 0, max_err = 0;
        std::string first_bad;
    };

    [[nodiscard]] Compare compare_pane(const RefImage& ref, const Image& gpu, int tol)
    {
        Compare c;
        c.total = ref.rect.w * ref.rect.h;
        for (int y = ref.rect.y; y < ref.rect.y + ref.rect.h; ++y)
            for (int x = ref.rect.x; x < ref.rect.x + ref.rect.w; ++x)
            {
                const RefPixel& p = ref.at(x, y);
                if (p.unsafe)
                {
                    ++c.unsafe;
                    continue;
                }
                ++c.compared;
                const demo::Rgba8 g = gpu.at(x, y);
                const int err = g[3] == 255 ? rgb_distance(g, p.rgb) : 255;
                c.max_err = std::max(c.max_err, err);
                if (err > tol)
                {
                    ++c.bad;
                    if (c.first_bad.empty())
                        c.first_bad = std::format("first at ({}, {}): gpu {} cpu ({}, {}, {})", x, y, demo::format_rgba(g),
                                                  p.rgb[0], p.rgb[1], p.rgb[2]);
                }
            }
        return c;
    }

    /// Number of safe reference pixels at which two reference images differ by more than `tol`.
    [[nodiscard]] int count_differences(const RefImage& a, const RefImage& b, int tol)
    {
        int n = 0;
        for (int y = a.rect.y; y < a.rect.y + a.rect.h; ++y)
            for (int x = a.rect.x; x < a.rect.x + a.rect.w; ++x)
            {
                const RefPixel& pa = a.at(x, y);
                const RefPixel& pb = b.at(x, y);
                if (pa.unsafe || pb.unsafe) continue;
                for (std::size_t c = 0; c < 3; ++c)
                    if (std::abs(int{pa.rgb[c]} - int{pb.rgb[c]}) > tol)
                    {
                        ++n;
                        break;
                    }
            }
        return n;
    }

    [[nodiscard]] int count_differences(const RefImage& ref, const Image& gpu, int tol)
    {
        int n = 0;
        for (int y = ref.rect.y; y < ref.rect.y + ref.rect.h; ++y)
            for (int x = ref.rect.x; x < ref.rect.x + ref.rect.w; ++x)
            {
                const RefPixel& p = ref.at(x, y);
                if (p.unsafe) continue;
                if (rgb_distance(gpu.at(x, y), p.rgb) > tol) ++n;
            }
        return n;
    }

    // ---------------------------------------------------------------------------------------------
    // Projecting known points with the same CPU matrices
    // ---------------------------------------------------------------------------------------------
    struct Projected
    {
        double x = 0, y = 0;  // window coordinates (continuous; pixel centres are at +0.5)
        double depth = 0;     // window z in [0, 1]
        double eye_depth = 0; // distance in front of the eye plane (-z_eye)
        double w = 0;
        [[nodiscard]] int px() const { return static_cast<int>(std::floor(x)); }
        [[nodiscard]] int py() const { return static_cast<int>(std::floor(y)); }
    };

    [[nodiscard]] Projected project_world(const PaneRef& pr, Vec3 world)
    {
        const Vec4 eye = pr.view * Vec4{world.x, world.y, world.z, 1.0f};
        const Vec4 clip = pr.proj * eye;
        const double w = dbl(clip.w);
        Projected p;
        p.x = pr.pane.rect.x + (dbl(clip.x) / w + 1.0) * 0.5 * pr.pane.rect.w;
        p.y = pr.pane.rect.y + (dbl(clip.y) / w + 1.0) * 0.5 * pr.pane.rect.h;
        p.depth = (dbl(clip.z) / w + 1.0) * 0.5;
        p.eye_depth = -dbl(eye.z);
        p.w = w;
        return p;
    }

    [[nodiscard]] Vec3 transform_point(const Mat4& m, Vec3 p)
    {
        const Vec4 r = m * Vec4{p.x, p.y, p.z, 1.0f};
        return {r.x, r.y, r.z};
    }

    [[nodiscard]] Vec3 transform_direction(const Mat4& m, Vec3 v)
    {
        const Vec4 r = m * Vec4{v.x, v.y, v.z, 0.0f};
        return dm::normalize({r.x, r.y, r.z});
    }

    /// Object-space point on cube face `face`: su, sv in [-1, 1] along the face tangents (0, 0 = the face centre).
    [[nodiscard]] Vec3 cube_face_point(std::size_t face, float su, float sv)
    {
        return kFaces[face].n * kCubeHalf + kFaces[face].u * (su * kCubeHalf) + kFaces[face].v * (sv * kCubeHalf);
    }
    // The two triangles of a face share the diagonal u == v; probes stay away from it (and from the face edges),
    // because pixels on a shared edge are decided by the rasterization tie rules. Several candidates are
    // listed because the picture of a probe point can land on the edge of ANOTHER face; the first candidate
    // whose pixel is safe in the CPU reference is used.
    constexpr std::array<std::pair<float, float>, 8> kProbeCandidates = {{{0.35f, -0.35f}, {-0.35f, 0.35f}, {0.55f, -0.25f}, {-0.25f, 0.55f},
                                                                          {0.2f, -0.55f}, {-0.55f, 0.2f}, {0.6f, -0.6f}, {-0.6f, 0.6f}}};

    [[nodiscard]] Rgb8 cube_face_rgb(const Mat4& model, std::size_t face)
    {
        const Vec3 n = kFaces[face].n, c = kCubeColors[face];
        return shaded_rgb(model, {0, 0, 0, n.x, n.y, n.z, c.x, c.y, c.z}, true);
    }

    [[nodiscard]] bool face_is_facing(const PaneRef& pr, const Camera& cam, const Mat4& model, std::size_t face)
    {
        const Vec3 n = transform_direction(model, kFaces[face].n);
        const Vec3 centre = transform_point(model, kFaces[face].n * kCubeHalf);
        const Vec3 to_camera = pr.pane.ortho ? cam.eye() - cam.target : cam.eye() - centre;
        return dm::dot(n, to_camera) > 0.0f;
    }

    [[nodiscard]] std::string fmt_rgb(const Rgb8& c) { return std::format("({}, {}, {})", c[0], c[1], c[2]); }

    /// First candidate point on cube face `face` whose pixel is a safe cube pixel in `a` (and in `b`, if given).
    [[nodiscard]] std::optional<Projected> find_face_probe(const PaneRef& a, const PaneRef* b, const Mat4& model, std::size_t face)
    {
        for (const auto& [su, sv] : kProbeCandidates)
        {
            const Projected p = project_world(a, transform_point(model, cube_face_point(face, su, sv)));
            if (!a.pane.rect.contains(p.px(), p.py())) continue;
            const RefPixel& ra = a.img.at(p.px(), p.py());
            if (ra.unsafe || ra.item != static_cast<int>(kCube)) continue;
            if (b != nullptr)
            {
                const RefPixel& rb = b->img.at(p.px(), p.py());
                if (rb.unsafe || rb.item != static_cast<int>(kCube)) continue;
            }
            return p;
        }
        return std::nullopt;
    }

    // ---------------------------------------------------------------------------------------------
    // The demo
    // ---------------------------------------------------------------------------------------------
    class TransformationsCamera final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            u_model_ = loc("u_model");
            u_view_ = loc("u_view");
            u_projection_ = loc("u_projection");
            u_light_dir_ = loc("u_light_dir");
            u_ambient_ = loc("u_ambient");
            u_lit_ = loc("u_lit");

            // All meshes live in one buffer; each draw selects its range with glDrawArrays(first, count).
            const std::array<Vec3, 6> post_colors = {{kPostColor, kPostColor, kPostColor, kPostColor, kPostColor, kPostColor}};
            ranges_.ground = mesh_.add_ground(kGroundTiles);
            ranges_.cube = mesh_.add_box({0, 0, 0}, {kCubeHalf, kCubeHalf, kCubeHalf}, kCubeColors);
            ranges_.post = mesh_.add_box({0, kPostHeight * 0.5f, 0}, {kPostHalfWidth, kPostHeight * 0.5f, kPostHalfWidth},
                                         post_colors, false);
            ranges_.pyramid = mesh_.add_pyramid(0.9f, 1.8f,
                                                {{{0.95f, 0.35f, 0.20f}, {0.95f, 0.65f, 0.15f}, {0.85f, 0.25f, 0.45f}, {0.98f, 0.80f, 0.35f}}});
            ranges_.icon_persp = mesh_.add_volume_icon(false);
            ranges_.icon_ortho = mesh_.add_volume_icon(true);
            ranges_.axis[0] = mesh_.add_box({0.45f, 0, 0}, {0.45f, 0.055f, 0.055f}, Vec3{0.95f, 0.20f, 0.20f});
            ranges_.axis[1] = mesh_.add_box({0, 0.45f, 0}, {0.055f, 0.45f, 0.055f}, Vec3{0.30f, 0.90f, 0.30f});
            ranges_.axis[2] = mesh_.add_box({0, 0, 0.45f}, {0.055f, 0.055f, 0.45f}, Vec3{0.30f, 0.50f, 1.00f});
            layout_ = make_layout();

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(mesh_.vertices), mgl::BufferUsage::StaticDraw);
            const auto stride = static_cast<GLsizei>(sizeof(Vertex));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(12));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 3, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(24));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            start_ = std::chrono::steady_clock::now();
        }

        void render(demo::Env& env, int frame) override
        {
            // Headless and --frames runs use frame / 60 s so that every run draws the same picture;
            // an interactive window follows the wall clock.
            if (env.interactive)
                time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
            else
                time_ = static_cast<double>(frame) / 60.0;
            draw_frame(env, params_, make_items(ranges_, layout_, time_));
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            Camera& c = params_.camera;
            switch (key)
            {
            case 'w': c.radius = std::max(1.0f, c.radius - 0.5f); break;
            case 's': c.radius = std::min(40.0f, c.radius + 0.5f); break;
            case 'a': c.azimuth -= dm::radians(8.0f); break;
            case 'd': c.azimuth += dm::radians(8.0f); break;
            case 'q': c.elevation = std::max(dm::radians(-10.0f), c.elevation - dm::radians(4.0f)); break;
            case 'e': c.elevation = std::min(dm::radians(85.0f), c.elevation + dm::radians(4.0f)); break;
            case 'p':
                params_.mode = params_.mode == Mode::Split         ? Mode::Perspective
                               : params_.mode == Mode::Perspective ? Mode::Orthographic
                                                                   : Mode::Split;
                break;
            case 'n': params_.near_plane = std::max(0.1f, params_.near_plane - 0.5f); break;
            case 'm': params_.near_plane = std::min(params_.far_plane - 1.0f, params_.near_plane + 0.5f); break;
            case 'f': params_.far_plane = std::max(params_.near_plane + 1.0f, params_.far_plane - 2.0f); break;
            case 'g': params_.far_plane = std::min(200.0f, params_.far_plane + 2.0f); break;
            case 'r': params_ = Params{}; break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteProgram(program_);
        }

    private:
        [[nodiscard]] mgl::UniformLocation loc(const char* name) const
        {
            const mgl::UniformLocation l = mgl::glGetUniformLocation(program_, name);
            if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
            return l;
        }

        /// glUniformMatrix4fv in the three flavours discussed in the README.
        static void upload_matrix(mgl::UniformLocation location, const Mat4& m, Upload mode)
        {
            switch (mode)
            {
            case Upload::ColumnMajor:
                // Mat4 stores columns one after another, exactly the layout of a GLSL mat4.
                mgl::glUniformMatrix4fv(location, 1, GL_FALSE, m.data());
                break;
            case Upload::RowMajorTranspose:
            {
                const Mat4 rows = transposed(m);  // now m[r * 4 + c]: row-major data
                mgl::glUniformMatrix4fv(location, 1, GL_TRUE, rows.data());
                break;
            }
            case Upload::WrongTranspose:
            {
                const Mat4 rows = transposed(m);
                mgl::glUniformMatrix4fv(location, 1, GL_FALSE, rows.data());  // row-major data declared column-major
                break;
            }
            }
        }

        void draw_items(std::span<const Item> items, Upload mode)
        {
            for (const Item& it : items)
            {
                upload_matrix(u_model_, it.model, mode);        // object -> world
                mgl::glUniform1f(u_lit_, it.lit ? 1.0f : 0.0f);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, it.mesh.first, it.mesh.count);
            }
        }

        void draw_inset(const Rect& r, const Mat4& view, const Mat4& proj, std::span<const Item> items, Upload mode)
        {
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glClearColor(kGizmoClear.x, kGizmoClear.y, kGizmoClear.z, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            upload_matrix(u_view_, view, mode);
            upload_matrix(u_projection_, proj, mode);
            draw_items(items, mode);
        }

        void draw_frame(const demo::Env& env, const Params& p, const std::vector<Item>& items)
        {
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            const Vec3 l = light_dir();
            mgl::glUniform3f(u_light_dir_, l.x, l.y, l.z);
            mgl::glUniform1f(u_ambient_, kAmbient);

            // The window itself is cleared to the separator color; each pane then clears only its own
            // rectangle (scissor), which leaves a one-pixel gutter between the panes.
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kSeparator.x, kSeparator.y, kSeparator.z, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            const std::vector<Item> gizmo = make_gizmo_items(ranges_);
            for (const Pane& pane : make_panes(env.width, env.height, p.mode))
            {
                const Rect& r = pane.rect;
                mgl::glEnable(mgl::Capability::ScissorTest);
                mgl::glScissor(r.x, r.y, r.w, r.h);
                mgl::glViewport(r.x, r.y, r.w, r.h);  // NDC [-1, 1]^2 -> this rectangle (ES 3.0 2.13.1)
                mgl::glClearColor(pane.clear.x, pane.clear.y, pane.clear.z, 1.0f);
                // GL_GREATER only makes sense with the depth buffer cleared to 0 ("nothing drawn yet")
                mgl::glClearDepthf(p.depth == DepthMode::Greater ? 0.0f : 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
                if (p.depth == DepthMode::Off)
                {
                    mgl::glDisable(mgl::Capability::DepthTest);
                }
                else
                {
                    mgl::glEnable(mgl::Capability::DepthTest);
                    mgl::glDepthFunc(p.depth == DepthMode::Greater ? mgl::CompareFunc::Greater : mgl::CompareFunc::Less);
                }

                const Mat4 view = p.camera.view();                                     // world -> eye
                const Mat4 proj = pane_projection(pane, p.camera, p.near_plane, p.far_plane);  // eye -> clip
                upload_matrix(u_view_, view, p.upload);
                upload_matrix(u_projection_, proj, p.upload);
                draw_items(items, p.upload);

                // Two small insets, each with its own viewport, view and projection:
                // the axis gizmo (only the rotation of the view, orthographic) and the volume icon.
                draw_inset(gizmo_rect(r), gizmo_view(view), gizmo_projection(), gizmo, p.upload);
                draw_inset(icon_rect(r), Mat4::identity(), icon_projection(), make_icon_items(ranges_, pane.ortho), p.upload);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        [[nodiscard]] std::vector<PaneRef> build_refs(const std::vector<Item>& items, const Params& p, int w, int h,
                                                      std::vector<Pane> panes = {}) const
        {
            if (panes.empty()) panes = make_panes(w, h, p.mode);
            const std::vector<Item> gizmo = make_gizmo_items(ranges_);
            std::vector<PaneRef> refs;
            for (const Pane& pane : panes)
                refs.push_back(build_pane_ref(mesh_.vertices, items, gizmo, make_icon_items(ranges_, pane.ortho), pane, p.camera,
                                              p.near_plane, p.far_plane, p.depth));
            return refs;
        }

        void print_chain(const Params& p, const Item& cube, int w, int h) const;

        mgl::ProgramId program_{};
        mgl::VertexArrayId vao_{};
        mgl::BufferId vbo_{};
        mgl::UniformLocation u_model_{}, u_view_{}, u_projection_{}, u_light_dir_{}, u_ambient_{}, u_lit_{};
        Mesh mesh_;
        Ranges ranges_;
        Layout layout_;
        Params params_;
        double time_ = 0.0;
        std::chrono::steady_clock::time_point start_;
    };

    // ---------------------------------------------------------------------------------------------
    // Validation helpers that need no GL access
    // ---------------------------------------------------------------------------------------------

    /// Where does the near plane cut a surface, in window coordinates? The world line O + s*D (depth increasing with
    /// s) crosses the plane "eye depth == near" at the cut point C. A second line through C inside both the surface
    /// and the near plane gives the direction of the cut on the screen; intersecting that direction with the pixel
    /// column through C predicts, to a fraction of a pixel, the row at which `inside` must stop holding when the
    /// column is scanned. (Scanning along a line that crosses the cut obliquely would add up to half a pixel times
    /// the slope of the cut; scanning a pixel column and predicting the cut at that column's centre does not.)
    struct CrossingResult
    {
        enum class State { Ok, Failed, NotVisible } state = State::Failed;
        std::string detail;
    };

    template <typename Inside>
    [[nodiscard]] CrossingResult probe_crossing(const PaneRef& pr, const Image& gpu, float near_plane, Vec3 origin, Vec3 dir,
                                                Vec3 side, Inside inside)
    {
        CrossingResult res;
        auto depth_of = [&](Vec3 pnt) { return project_world(pr, pnt).eye_depth; };
        const double d0 = depth_of(origin);
        const double slope = depth_of(origin + dir) - d0;
        if (slope < 1e-6)
        {
            res.detail = "the line does not move away from the eye";
            return res;
        }
        const Vec3 cut_point = origin + dir * static_cast<float>((dbl(near_plane) - d0) / slope);
        // a direction inside the surface and inside the near plane (zero depth gradient): `side` minus its depth change
        const double side_depth = depth_of(origin + side) - d0;
        const Vec3 along = side - dir * static_cast<float>(side_depth / slope);
        const Projected c = project_world(pr, cut_point);
        const Projected c2 = project_world(pr, cut_point + along * 0.25f);
        const Projected in = project_world(pr, cut_point + dir * 0.5f);  // deeper than the near plane: the visible side
        const double bx = c2.x - c.x, by = c2.y - c.y;
        if (std::abs(bx) < 0.05)
        {
            res.detail = "the cut is vertical on the screen";
            return res;
        }
        const int column = c.px();
        const double y_pred = c.y + (column + 0.5 - c.x) * by / bx;  // the cut's height at the centre of the column
        const bool inside_is_above = in.y > c.y;
        const int dy = inside_is_above ? -1 : 1;                      // scan direction: from the visible side to the cut
        const int start = static_cast<int>(std::floor(y_pred)) - dy * 8;
        const int last = static_cast<int>(std::floor(y_pred)) + dy * 6;
        const Rect& pane = pr.pane.rect;
        if (!pane.contains(column, start) || !pane.contains(column, last))
        {
            res.state = CrossingResult::State::NotVisible;
            res.detail = std::format("the cut (x {:.1f}, y {:.1f}) is too close to or outside the pane edge", c.x, y_pred);
            return res;
        }
        int first_outside = -1;
        bool clean = inside(gpu.at(column, start));
        for (int y = start; y != last + dy; y += dy)
        {
            const bool in_here = inside(gpu.at(column, y));
            if (first_outside < 0 && !in_here) first_outside = y;
            if (first_outside >= 0 && in_here) clean = false;
        }
        const double edge = first_outside < 0 ? -1e9 : (inside_is_above ? first_outside + 1.0 : static_cast<double>(first_outside));
        res.detail = std::format("column {}: cut predicted at y = {:.2f}, measured between rows {} and {} (y = {:.1f}), {}", column, y_pred,
                                 first_outside - dy, first_outside, edge, clean ? "single transition" : "no clean inside/outside transition");
        res.state = clean && first_outside >= 0 && std::abs(edge - y_pred) <= 1.0 ? CrossingResult::State::Ok : CrossingResult::State::Failed;
        return res;
    }

    /// First and last row of the vertical run of pixels around (x, y) in a pane for which `pred(x, y)` holds.
    template <typename Pred>
    [[nodiscard]] std::pair<int, int> vertical_run(const Rect& pane, int x, int y, Pred pred)
    {
        int lo = y, hi = y;
        while (lo - 1 >= pane.y && pred(x, lo - 1)) --lo;
        while (hi + 1 < pane.y + pane.h && pred(x, hi + 1)) ++hi;
        return {lo, hi};
    }

    /// Rows whose pixel centres lie inside the convex hull of the projected points `pts`, in the pixel column `column`.
    /// The hull's intersection with the line x = column + 0.5 is spanned by the crossings of the segments between
    /// all pairs of points, so no hull construction is needed.
    [[nodiscard]] std::optional<std::pair<int, int>> hull_column_run(std::span<const Projected> pts, int column)
    {
        const double cx = column + 0.5;
        double y_min = 1e30, y_max = -1e30;
        for (std::size_t i = 0; i < pts.size(); ++i)
            for (std::size_t j = i + 1; j < pts.size(); ++j)
            {
                const Projected &a = pts[i], &b = pts[j];
                if (a.x == b.x || (a.x - cx) * (b.x - cx) > 0.0) continue;
                const double y = a.y + (cx - a.x) / (b.x - a.x) * (b.y - a.y);
                y_min = std::min(y_min, y);
                y_max = std::max(y_max, y);
            }
        if (y_min > y_max) return std::nullopt;
        return std::pair<int, int>{static_cast<int>(std::ceil(y_min - 0.5)), static_cast<int>(std::floor(y_max - 0.5))};
    }

    void TransformationsCamera::print_chain(const Params& p, const Item& cube, int w, int h) const
    {
        const std::vector<Pane> panes = make_panes(w, h, p.mode);
        auto print_matrix = [](std::string_view name, const Mat4& m) {
            for (int row = 0; row < 4; ++row)
                std::cout << std::format("    {:<5} | {:+9.4f} {:+9.4f} {:+9.4f} {:+9.4f} |\n", row == 0 ? name : std::string_view{}, m.at(row, 0),
                                         m.at(row, 1), m.at(row, 2), m.at(row, 3));
        };
        std::cout << "Matrices as the CPU sees them (row r, column c = element [4c + r] of the uploaded array):\n";
        print_matrix("M", cube.model);
        print_matrix("V", p.camera.view());
        for (const Pane& pane : panes) print_matrix(pane.ortho ? "P ort" : "P per", pane_projection(pane, p.camera, p.near_plane, p.far_plane));
        std::cout << "Coordinate chain of the cube corner (+h,+h,+h), h = 0.8, in each pane:\n";
        for (const Pane& pane : panes)
        {
            const Mat4 view = p.camera.view();
            const Mat4 proj = pane_projection(pane, p.camera, p.near_plane, p.far_plane);
            const Vec4 obj{kCubeHalf, kCubeHalf, kCubeHalf, 1.0f};
            const Vec4 world = cube.model * obj;
            const Vec4 eye = view * world;
            const Vec4 clip = proj * eye;
            const double ndc_x = dbl(clip.x / clip.w), ndc_y = dbl(clip.y / clip.w), ndc_z = dbl(clip.z / clip.w);
            std::cout << std::format("  [{}] viewport ({}, {}, {}, {})\n", pane.name, pane.rect.x, pane.rect.y, pane.rect.w, pane.rect.h);
            std::cout << std::format("    object  ({:+8.3f} {:+8.3f} {:+8.3f} {:+8.3f})\n", obj.x, obj.y, obj.z, obj.w);
            std::cout << std::format("    world   ({:+8.3f} {:+8.3f} {:+8.3f} {:+8.3f})   = M * object\n", world.x, world.y, world.z, world.w);
            std::cout << std::format("    eye     ({:+8.3f} {:+8.3f} {:+8.3f} {:+8.3f})   = V * world\n", eye.x, eye.y, eye.z, eye.w);
            std::cout << std::format("    clip    ({:+8.3f} {:+8.3f} {:+8.3f} {:+8.3f})   = P * eye\n", clip.x, clip.y, clip.z, clip.w);
            std::cout << std::format("    ndc     ({:+8.3f} {:+8.3f} {:+8.3f})            = clip.xyz / clip.w\n", ndc_x, ndc_y, ndc_z);
            std::cout << std::format("    window  ({:8.2f} {:8.2f})  depth {:.4f}  = viewport({}, {}, {}, {}), depth range [0, 1]\n",
                                     pane.rect.x + (ndc_x + 1.0) * 0.5 * pane.rect.w, pane.rect.y + (ndc_y + 1.0) * 0.5 * pane.rect.h,
                                     (ndc_z + 1.0) * 0.5, pane.rect.x, pane.rect.y, pane.rect.w, pane.rect.h);
        }
    }

    void TransformationsCamera::validate(demo::Env& env, demo::Check& check)
    {
        const int W = env.width, H = env.height;
        const Params base = params_;
        const std::vector<Item> items = make_items(ranges_, layout_, time_);
        const Mat4 cube_model = items[kCube].model;
        const Camera& cam = base.camera;

        // ---- 1. The CPU model itself: known answers of the matrices in demo/math.hpp ----------------------
        {
            const double n = dbl(base.near_plane), f = dbl(base.far_plane);
            const Mat4 pp = dm::perspective(kFovY, 2.0f, base.near_plane, base.far_plane);
            auto ndc = [](const Mat4& m, Vec4 v) {
                const Vec4 c = m * v;
                return std::array<double, 4>{dbl(c.x / c.w), dbl(c.y / c.w), dbl(c.z / c.w), dbl(c.w)};
            };
            const auto near_pt = ndc(pp, {0, 0, -base.near_plane, 1});
            const auto far_pt = ndc(pp, {0, 0, -base.far_plane, 1});
            check.expect(std::abs(near_pt[2] + 1.0) < 1e-4 && std::abs(far_pt[2] - 1.0) < 1e-4 &&
                             std::abs(near_pt[3] - n) < 1e-4 && std::abs(far_pt[3] - f) < 1e-3,
                         "perspective(): the near plane maps to NDC z = -1 and the far plane to +1, w = distance",
                         std::format("near z={:.6f} w={:.4f}, far z={:.6f} w={:.3f}", near_pt[2], near_pt[3], far_pt[2], far_pt[3]));
            const float th = std::tan(kFovY * 0.5f);
            const auto corner = ndc(pp, {2.0f * th * base.near_plane, th * base.near_plane, -base.near_plane, 1});
            check.expect(std::abs(corner[0] - 1.0) < 1e-4 && std::abs(corner[1] - 1.0) < 1e-4 && std::abs(corner[2] + 1.0) < 1e-4,
                         "perspective(): the corner of the near rectangle (aspect * n tan(fovy/2), n tan(fovy/2)) maps to NDC (1, 1, -1)",
                         std::format("({:.6f}, {:.6f}, {:.6f})", corner[0], corner[1], corner[2]));
            const Mat4 po = dm::ortho(-4, 4, -2, 2, base.near_plane, base.far_plane);
            const auto o1 = ndc(po, {4, 2, -base.near_plane, 1});
            const auto o2 = ndc(po, {-4, -2, -base.far_plane, 1});
            check.expect(std::abs(o1[0] - 1) < 1e-5 && std::abs(o1[1] - 1) < 1e-5 && std::abs(o1[2] + 1) < 1e-5 &&
                             std::abs(o2[0] + 1) < 1e-5 && std::abs(o2[1] + 1) < 1e-5 && std::abs(o2[2] - 1) < 1e-5 &&
                             o1[3] == 1.0 && o2[3] == 1.0,
                         "ortho(): the box corners map to the NDC cube corners and w stays 1 (no perspective division)",
                         std::format("({:.5f}, {:.5f}, {:.5f}) and ({:.5f}, {:.5f}, {:.5f})", o1[0], o1[1], o1[2], o2[0], o2[1], o2[2]));
            const Mat4 v = cam.view();
            const Vec4 e0 = v * Vec4{cam.eye().x, cam.eye().y, cam.eye().z, 1.0f};
            const Vec4 e1 = v * Vec4{cam.target.x, cam.target.y, cam.target.z, 1.0f};
            check.expect(std::abs(e0.x) + std::abs(e0.y) + std::abs(e0.z) < 1e-4f &&
                             std::abs(e1.x) + std::abs(e1.y) < 1e-4f && std::abs(e1.z + cam.radius) < 1e-4f,
                         "look_at(): the eye maps to the eye-space origin and the target to (0, 0, -radius)",
                         std::format("eye -> ({:.5f}, {:.5f}, {:.5f}), target -> ({:.5f}, {:.5f}, {:.5f})", e0.x, e0.y, e0.z, e1.x, e1.y, e1.z));
        }

        print_chain(base, items[kCube], W, H);

        // ---- 2. GPU image of the final frame against the CPU rasterizer -------------------------------------
        const Image gpu = Image::grab(W, H);
        const std::vector<PaneRef> refs = build_refs(items, base, W, H);
        if (refs.size() != 2)
        {
            check.expect(false, "validation expects the split layout (default Params)");
            return;
        }
        const PaneRef& persp = refs[0];
        const PaneRef& ortho = refs[1];

        auto record_pane = [&](const PaneRef& pr, const Image& image, std::string_view label) {
            const Compare c = compare_pane(pr.img, image, kTol);
            check.expect(c.bad == 0,
                         std::format("{} pane [{}]: {} pixels agree with the CPU pipeline (max error {} of 255)", pr.pane.name, label,
                                     c.compared, c.max_err),
                         c.bad == 0 ? std::string{} : std::format("{} mismatches; {}", c.bad, c.first_bad));
            check.expect(c.compared * 10 >= c.total * 6,
                         std::format("{} pane [{}]: at least 60% of the {} pixels were comparable", pr.pane.name, label, c.total),
                         std::format("compared {}, excluded {} within {} px of a triangle edge or depth tie", c.compared, c.unsafe,
                                     kEdgeMargin));
            return c;
        };
        record_pane(persp, gpu, "final frame");
        record_pane(ortho, gpu, "final frame");

        // ---- 3. Named probes -----------------------------------------------------------------------------------
        for (const PaneRef* pr : {&persp, &ortho})
        {
            std::string bad_visible, bad_hidden, bad_corner;
            int visible = 0, hidden = 0, corner_probes = 0;
            for (std::size_t f = 0; f < 6; ++f)
            {
                const std::optional<Projected> found = find_face_probe(*pr, nullptr, cube_model, f);
                if (!found)
                {
                    (face_is_facing(*pr, cam, cube_model, f) ? bad_visible : bad_hidden) += std::format("face {}: no safe probe point; ", f);
                    continue;
                }
                const Projected c = *found;
                const RefPixel& rp = pr->img.at(c.px(), c.py());
                const demo::Rgba8 got = gpu.at(c.px(), c.py());
                const Rgb8 own = cube_face_rgb(cube_model, f);
                if (face_is_facing(*pr, cam, cube_model, f))
                {
                    ++visible;
                    const bool identity = rp.item == static_cast<int>(kCube) && rp.tri / 2 == static_cast<int>(f) && !rp.unsafe;
                    if (!identity || rgb_distance(got, own) > kTol)
                        bad_visible += std::format("face {} at ({}, {}): gpu {} expected {}{}; ", f, c.px(), c.py(), demo::format_rgba(got),
                                                   fmt_rgb(own), identity ? "" : " [CPU identity mismatch]");
                    // four points close to the four corners of the face (tangent coordinates in units of the half edge)
                    for (const auto& [su, sv] : std::array<std::pair<float, float>, 4>{{{0.85f, 0.55f}, {0.7f, -0.7f}, {-0.7f, 0.7f}, {-0.85f, -0.55f}}})
                        {
                            const Vec3 q = transform_point(cube_model, cube_face_point(f, su, sv));
                            const Projected qc = project_world(*pr, q);
                            const RefPixel& qr = pr->img.at(qc.px(), qc.py());
                            const demo::Rgba8 qg = gpu.at(qc.px(), qc.py());
                            ++corner_probes;
                            if (rgb_distance(qg, own) > kTol || qr.item != static_cast<int>(kCube) || qr.tri / 2 != static_cast<int>(f))
                                bad_corner += std::format("face {} point ({:+.1f},{:+.1f}) at ({}, {}): gpu {} expected {}; ", f, su, sv, qc.px(),
                                                          qc.py(), demo::format_rgba(qg), fmt_rgb(own));
                        }
                }
                else
                {
                    ++hidden;
                    // The centre of a back face lies inside the cube's silhouette: a nearer face must cover it.
                    const bool covered_by_other = rp.item == static_cast<int>(kCube) && rp.tri / 2 != static_cast<int>(f) && !rp.unsafe;
                    if (!covered_by_other || rgb_distance(got, own) <= 12 || rgb_distance(got, rp.rgb) > kTol)
                        bad_hidden += std::format("face {} at ({}, {}): gpu {} own colour {} covering face {}; ", f, c.px(), c.py(),
                                                  demo::format_rgba(got), fmt_rgb(own), rp.tri / 2);
                }
            }
            check.expect(visible == 3 && bad_visible.empty(),
                         std::format("{} pane: the {} front-facing cube faces show their predicted shaded color at a projected point near their centres",
                                     pr->pane.name, visible),
                         bad_visible);
            check.expect(corner_probes == 12 && bad_corner.empty(),
                         std::format("{} pane: {} points near the corners of the visible faces (M, V, P, viewport all in play) show the face color",
                                     pr->pane.name, corner_probes),
                         bad_corner);
            check.expect(hidden == 3 && bad_hidden.empty(),
                         std::format("{} pane: the {} back faces are hidden - their projected points show the nearer face, not their own color",
                                     pr->pane.name, hidden),
                         bad_hidden);

            // Ground tile centres: the checkerboard colors must appear at the projected tile centres.
            std::string bad_ground;
            int ground_probes = 0, parity[2] = {0, 0};
            for (int j = -6; j <= 6; j += 3)
                for (int i = -6; i <= 6; i += 3)
                {
                    // a point 0.25 off the tile centre, away from the tile's diagonal
                    const Projected c = project_world(*pr, {static_cast<float>(i) + 0.25f, 0.0f, static_cast<float>(j) - 0.25f});
                    if (!pr->pane.rect.contains(c.px(), c.py())) continue;
                    const RefPixel& rp = pr->img.at(c.px(), c.py());
                    if (rp.item != static_cast<int>(kGround) || rp.unsafe) continue;  // hidden behind an object or near an edge
                    const Vec3 col = Mesh::ground_color(i, j);
                    const Rgb8 expected = shaded_rgb(Mat4::identity(), {0, 0, 0, 0, 1, 0, col.x, col.y, col.z}, true);
                    ++ground_probes;
                    ++parity[(i + j + 64) % 2];
                    const demo::Rgba8 got = gpu.at(c.px(), c.py());
                    if (rgb_distance(got, expected) > kTol)
                        bad_ground += std::format("tile ({},{}) at ({}, {}): gpu {} expected {}; ", i, j, c.px(), c.py(), demo::format_rgba(got), fmt_rgb(expected));
                }
            check.expect(ground_probes >= 3 && parity[0] > 0 && parity[1] > 0 && bad_ground.empty(),
                         std::format("{} pane: {} ground tile centres (both checker colors) show the checkerboard color", pr->pane.name, ground_probes),
                         bad_ground);
        }

        // Axis gizmo: the midpoints of the three axis bars, projected with the rotation-only view.
        {
            std::string bad;
            const Mat4 gv = gizmo_view(persp.view);
            const Rect g = persp.gizmo;
            const std::array<Rgb8, 3> axis_rgb = {{{to8(0.95), to8(0.20), to8(0.20)}, {to8(0.30), to8(0.90), to8(0.30)}, {to8(0.30), to8(0.50), to8(1.00)}}};
            for (std::size_t a = 0; a < 3; ++a)
            {
                Vec3 mid{0, 0, 0};
                (a == 0 ? mid.x : a == 1 ? mid.y : mid.z) = 0.55f;
                const Vec4 clip = gizmo_projection() * (gv * Vec4{mid.x, mid.y, mid.z, 1.0f});
                const int px = static_cast<int>(std::floor(g.x + (dbl(clip.x) + 1.0) * 0.5 * g.w));
                const int py = static_cast<int>(std::floor(g.y + (dbl(clip.y) + 1.0) * 0.5 * g.h));
                const demo::Rgba8 got = gpu.at(px, py);
                if (rgb_distance(got, axis_rgb[a]) > kTol)
                    bad += std::format("axis {} at ({}, {}): gpu {} expected {}; ", "XYZ"[a], px, py, demo::format_rgba(got), fmt_rgb(axis_rgb[a]));
            }
            check.expect(bad.empty(), "axis gizmo: the X, Y, Z bars appear at the positions the rotation-only view predicts, unshaded red, green, blue", bad);
        }

        // Pane icons: a wedge in the perspective pane, a box in the orthographic pane (same two probe points in both).
        {
            std::string bad;
            const Rgb8 fill = {to8(0.30), to8(0.38), to8(0.62)};
            for (const PaneRef* pr : {&persp, &ortho})
            {
                const Rect& ic = pr->icon;
                auto probe = [&](float x, float y) {
                    return gpu.at(ic.x + static_cast<int>((x + 1.0f) * 0.5f * static_cast<float>(ic.w)),
                                  ic.y + static_cast<int>((y + 1.0f) * 0.5f * static_cast<float>(ic.h)));
                };
                const bool wedge_only_filled = rgb_distance(probe(0.7f, 0.62f), fill) <= kTol;  // inside the wedge, outside the box
                const bool box_only_filled = rgb_distance(probe(-0.5f, 0.3f), fill) <= kTol;    // inside the box, outside the wedge
                if (wedge_only_filled != !pr->pane.ortho || box_only_filled != pr->pane.ortho)
                    bad += std::format("{} icon: wedge-only probe {}, box-only probe {}; ", pr->pane.name, wedge_only_filled, box_only_filled);
            }
            check.expect(bad.empty(), "pane icons: the side view of the view volume is a wedge (perspective) or a box (orthographic)", bad);
        }

        // Gutter: the scissored clears must stay inside the pane rectangles (window clear color around them).
        {
            const Rgb8 sep = {to8(dbl(kSeparator.x)), to8(dbl(kSeparator.y)), to8(dbl(kSeparator.z))};
            std::string bad;
            int probes = 0;
            auto probe_gutter = [&](int x, int y) {
                if (x < 0 || y < 0 || x >= W || y >= H) return;
                for (const PaneRef& pr : refs)
                    if (pr.pane.rect.contains(x, y)) return;  // inside another pane: not a gutter pixel
                ++probes;
                if (rgb_distance(gpu.at(x, y), sep) > 1) bad += std::format("({}, {}): {}; ", x, y, demo::format_rgba(gpu.at(x, y)));
            };
            for (const PaneRef& pr : refs)
            {
                const Rect& r = pr.pane.rect;
                for (const int x : {r.x, r.x + r.w / 2, r.x + r.w - 1})
                    for (const int y : {r.y - 1, r.y + r.h}) probe_gutter(x, y);
                for (const int y : {r.y, r.y + r.h / 2, r.y + r.h - 1})
                    for (const int x : {r.x - 1, r.x + r.w}) probe_gutter(x, y);
            }
            check.expect(probes >= 12 && bad.empty(),
                         std::format("scissored clears stay inside their viewports: {} gutter pixels around the panes keep the separator color", probes),
                         bad);
        }

        // ---- 4. Perspective foreshortening: three identical posts at three distances ----------------------------
        {
            std::array<int, 3> run_len_persp{}, run_len_ortho{};
            for (const PaneRef* pr : {&persp, &ortho})
            {
                std::string detail;
                bool ok = true;
                std::array<int, 3> cpu_len{}, gpu_len{};  // predicted and measured pixel heights
                for (std::size_t k = 1; k <= 3; ++k)
                {
                    const Item& post = items[kPost0 + k];
                    // all colors the post can show: its five faces, shaded
                    std::vector<Rgb8> post_colors;
                    for (std::size_t f = 0; f < 6; ++f)
                    {
                        if (f == 3) continue;
                        const Vec3 n = kFaces[f].n;
                        post_colors.push_back(shaded_rgb(post.model, {0, 0, 0, n.x, n.y, n.z, kPostColor.x, kPostColor.y, kPostColor.z}, true));
                    }
                    auto is_post_gpu = [&](int x, int y) {
                        const demo::Rgba8 g = gpu.at(x, y);
                        return std::any_of(post_colors.begin(), post_colors.end(), [&](const Rgb8& c) { return rgb_distance(g, c) <= 3; }) ? 1 : 0;
                    };
                    // Prediction: a convex prism in front of the eye covers exactly the pixels whose centres lie inside the
                    // convex hull of its eight projected corners. Intersect the hull with the pixel column through the
                    // projected axis of the post.
                    std::vector<Projected> corners;
                    for (const float dx : {-kPostHalfWidth, kPostHalfWidth})
                        for (const float dz : {-kPostHalfWidth, kPostHalfWidth})
                            for (const float dy : {0.0f, kPostHeight})
                                corners.push_back(project_world(*pr, layout_.posts[k] + Vec3{dx, dy, dz}));
                    const Projected axis = project_world(*pr, layout_.posts[k] + Vec3{0, kPostHeight * 0.5f, 0});
                    const auto predicted = hull_column_run(corners, axis.px());
                    if (!predicted)
                    {
                        ok = false;
                        detail += std::format("post {}: column {} misses the hull; ", k, axis.px());
                        continue;
                    }
                    const auto gpu_run = vertical_run(pr->pane.rect, axis.px(), (predicted->first + predicted->second) / 2, is_post_gpu);
                    const std::size_t i = k - 1;
                    cpu_len[i] = predicted->second - predicted->first + 1;
                    gpu_len[i] = gpu_run.second - gpu_run.first + 1;
                    const bool end_ok = std::abs(predicted->first - gpu_run.first) <= 1 && std::abs(predicted->second - gpu_run.second) <= 1;
                    ok = ok && end_ok && cpu_len[i] >= 8;
                    detail += std::format("post {} (column {}): predicted rows {}..{} ({} px), GPU rows {}..{} ({} px); ", k, axis.px(), predicted->first,
                                          predicted->second, cpu_len[i], gpu_run.first, gpu_run.second, gpu_len[i]);
                }
                check.expect(ok,
                             std::format("{} pane: the predicted pixel height of three identical posts matches the GPU within 1 px at each end", pr->pane.name),
                             detail);
                (pr->pane.ortho ? run_len_ortho : run_len_persp) = gpu_len;
            }
            check.expect(run_len_persp[0] > run_len_persp[1] && run_len_persp[1] > run_len_persp[2] && run_len_persp[0] >= run_len_persp[2] + 8,
                         "perspective: identical posts get smaller with distance (near > middle > far, GPU pixel heights)",
                         std::format("{} > {} > {}", run_len_persp[0], run_len_persp[1], run_len_persp[2]));
            const int spread = std::max({run_len_ortho[0], run_len_ortho[1], run_len_ortho[2]}) - std::min({run_len_ortho[0], run_len_ortho[1], run_len_ortho[2]});
            check.expect(spread <= 2,
                         "orthographic: the same three posts have the same pixel height at every distance (spread <= 2 px)",
                         std::format("{}, {}, {}", run_len_ortho[0], run_len_ortho[1], run_len_ortho[2]));
        }

        // ---- 5. Near-plane clipping: where does the cut appear? -------------------------------------------------
        int crossings_checked = 0;
        for (const PaneRef* pr : {&persp, &ortho})
        {
            // (a) post 0: its front face's vertical centre line crosses the near plane at mid-height.
            const Item& post0 = items[kPost0];
            std::size_t front_face = 0;
            float best = -1e9f;
            const Vec3 post_centre = layout_.posts[0] + Vec3{0, kPostHeight * 0.5f, 0};
            for (const std::size_t f : {std::size_t{0}, std::size_t{1}, std::size_t{4}, std::size_t{5}})
            {
                const float s = dm::dot(kFaces[f].n, dm::normalize(cam.eye() - post_centre));
                if (s > best)
                {
                    best = s;
                    front_face = f;
                }
            }
            const Vec3 face_n = kFaces[front_face].n;  // post models are pure translations: world normal == object normal
            const Vec3 top = layout_.posts[0] + face_n * kPostHalfWidth + Vec3{0, kPostHeight, 0};
            const Rgb8 face_rgb = shaded_rgb(post0.model, {0, 0, 0, face_n.x, face_n.y, face_n.z, kPostColor.x, kPostColor.y, kPostColor.z}, true);
            const CrossingResult post_cut = probe_crossing(*pr, gpu, base.near_plane, top, Vec3{0, -1, 0}, dm::cross({0, 1, 0}, face_n),
                                                           [&](const demo::Rgba8& p) { return rgb_distance(p, face_rgb) <= 3; });

            // (b) ground: along the ground line below the orbit target, everything nearer than the near plane is gone.
            const Vec3 fwd{-std::sin(cam.azimuth), 0, -std::cos(cam.azimuth)};
            const Vec3 right{std::cos(cam.azimuth), 0, -std::sin(cam.azimuth)};
            const Vec3 g0 = Vec3{cam.eye().x, 0, cam.eye().z} + fwd * 2.0f;
            const Rgb8 bg = {to8(dbl(pr->pane.clear.x)), to8(dbl(pr->pane.clear.y)), to8(dbl(pr->pane.clear.z))};
            const CrossingResult ground_cut = probe_crossing(*pr, gpu, base.near_plane, g0, fwd, right,
                                                             [&](const demo::Rgba8& p) { return rgb_distance(p, bg) > kTol; });
            for (const auto& [what, result] : {std::pair<std::string_view, const CrossingResult&>{"cuts post 0", post_cut},
                                               std::pair<std::string_view, const CrossingResult&>{"ends the ground", ground_cut}})
            {
                if (result.state == CrossingResult::State::NotVisible)
                {
                    std::cout << std::format("   note: {} pane: the near plane {} outside this window layout ({})\n", pr->pane.name, what, result.detail);
                    continue;
                }
                ++crossings_checked;
                check.expect(result.state == CrossingResult::State::Ok,
                             std::format("{} pane: the near plane {} at the window row the matrices predict (within 1 px)", pr->pane.name, what),
                             result.detail);
            }
        }
        check.expect(crossings_checked >= 2, "the near-plane cut is in view in at least two probes of this layout",
                     std::format("{} probes", crossings_checked));

        // ---- 6. Re-renders with changed state (each is compared with a CPU reference built for that state) ------
        auto rerender = [&](const Params& p) {
            draw_frame(env, p, items);
            return Image::grab(W, H);
        };

        // 6a. Depth test disabled: the last-drawn face wins, so hidden faces show through.
        {
            check.expect(mgl::glIsEnabled(mgl::Capability::DepthTest), "depth test is enabled while the scene is drawn (glIsEnabled)");
            Params p = base;
            p.depth = DepthMode::Off;
            const Image off = rerender(p);
            const std::vector<PaneRef> off_refs = build_refs(items, p, W, H);
            const Compare c = compare_pane(off_refs[0].img, off, kTol);
            check.expect(c.bad == 0 && c.compared > 2000,
                         std::format("depth test off: {} pixels of the perspective pane agree with the painter's-order CPU model", c.compared),
                         c.first_bad);
            int flips = 0, matches = 0;
            std::string detail;
            for (std::size_t f = 0; f < 6; ++f)
            {
                if (face_is_facing(persp, cam, cube_model, f)) continue;
                const std::optional<Projected> found = find_face_probe(persp, &off_refs[0], cube_model, f);
                if (!found) continue;
                const Projected c2 = *found;
                const RefPixel& on_px = persp.img.at(c2.px(), c2.py());
                const RefPixel& off_px = off_refs[0].img.at(c2.px(), c2.py());
                const bool differs = rgb_distance(gpu.at(c2.px(), c2.py()), off_px.rgb) > 12;
                flips += differs ? 1 : 0;
                const bool gpu_ok = rgb_distance(off.at(c2.px(), c2.py()), off_px.rgb) <= kTol;
                matches += gpu_ok ? 1 : 0;
                detail += std::format("face {}: depth on {}, off {} (gpu {}); ", f, fmt_rgb(on_px.rgb), fmt_rgb(off_px.rgb), demo::format_rgba(off.at(c2.px(), c2.py())));
            }
            check.expect(flips >= 1 && matches >= 1,
                         "depth test off: at least one hidden face's centre now shows a different color than with the test on, as the CPU predicts",
                         detail);
            mgl::glDisable(mgl::Capability::DepthTest);
            const bool disabled = !mgl::glIsEnabled(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::DepthTest);
            check.expect(disabled && mgl::glIsEnabled(mgl::Capability::DepthTest), "glEnable/glDisable(GL_DEPTH_TEST) toggle the state seen by glIsEnabled");
        }

        // 6a'. glDepthFunc(GL_GREATER) with the depth buffer cleared to 0: the FARTHEST surface wins.
        {
            Params p = base;
            p.depth = DepthMode::Greater;
            const Image far_wins = rerender(p);
            const std::vector<PaneRef> gt_refs = build_refs(items, p, W, H);
            for (std::size_t i = 0; i < 2; ++i)
            {
                const Compare c = compare_pane(gt_refs[i].img, far_wins, kTol);
                check.expect(c.bad == 0 && c.compared > 2000,
                             std::format("{} pane, glDepthFunc(GL_GREATER) + glClearDepthf(0): {} pixels agree with the farthest-surface-wins CPU model",
                                         gt_refs[i].pane.name, c.compared),
                             c.first_bad);
            }
            // A back face of the cube is the farthest cube surface along its own ray, so where nothing farther (the
            // ground) lies behind it, its own color appears at the probe - the color that is hidden with GL_LESS.
            int shown = 0;
            std::string detail;
            for (std::size_t f = 0; f < 6; ++f)
            {
                if (face_is_facing(persp, cam, cube_model, f)) continue;
                for (const auto& [su, sv] : kProbeCandidates)
                {
                    const Projected q = project_world(gt_refs[0], transform_point(cube_model, cube_face_point(f, su, sv)));
                    const RefPixel& rp = gt_refs[0].img.at(q.px(), q.py());
                    if (rp.unsafe || rp.item != static_cast<int>(kCube) || rp.tri / 2 != static_cast<int>(f)) continue;
                    const bool ok = rgb_distance(far_wins.at(q.px(), q.py()), cube_face_rgb(cube_model, f)) <= kTol &&
                                    rgb_distance(gpu.at(q.px(), q.py()), cube_face_rgb(cube_model, f)) > 12;
                    shown += ok ? 1 : 0;
                    detail += std::format("face {} at ({}, {}): GL_LESS {}, GL_GREATER {}; ", f, q.px(), q.py(), demo::format_rgba(gpu.at(q.px(), q.py())),
                                          demo::format_rgba(far_wins.at(q.px(), q.py())));
                    break;
                }
            }
            check.expect(shown >= 1,
                         "GL_GREATER: at least one back face of the cube shows its own color where GL_LESS hides it, as the CPU predicts", detail);
        }

        // 6b. Near plane moved to 0.5 and far plane moved into the scene.
        {
            Params near_p = base;
            near_p.near_plane = 0.5f;
            const Image near_img = rerender(near_p);
            const std::vector<PaneRef> near_refs = build_refs(items, near_p, W, H);
            for (std::size_t i = 0; i < 2; ++i)
            {
                const Compare c = compare_pane(near_refs[i].img, near_img, kTol);
                const int changed = count_differences(refs[i].img, near_refs[i].img, kTol);
                check.expect(c.bad == 0 && c.compared * 10 >= c.total * 5,
                             std::format("{} pane, near plane 0.5: {} pixels agree with the CPU clipper (triangles reach behind the eye plane)",
                                         refs[i].pane.name, c.compared),
                             c.first_bad);
                check.expect(changed >= 500,
                             std::format("{} pane: moving the near plane from {:.1f} to 0.5 changes {} pixels of the CPU prediction (the foreground returns)",
                                         refs[i].pane.name, dbl(base.near_plane), changed));
            }
            Params far_p = base;
            far_p.far_plane = cam.radius + 0.5f;  // through the middle of the cube
            const Image far_img = rerender(far_p);
            const std::vector<PaneRef> far_refs = build_refs(items, far_p, W, H);
            for (std::size_t i = 0; i < 2; ++i)
            {
                const Compare c = compare_pane(far_refs[i].img, far_img, kTol);
                const int changed = count_differences(refs[i].img, far_refs[i].img, kTol);
                const int gpu_changed = count_differences(refs[i].img, far_img, kTol);
                check.expect(c.bad == 0 && c.compared * 10 >= c.total * 5,
                             std::format("{} pane, far plane {:.1f}: {} pixels agree with the CPU clipper", refs[i].pane.name, dbl(far_p.far_plane), c.compared),
                             c.first_bad);
                check.expect(changed >= 500 && gpu_changed >= changed / 2,
                             std::format("{} pane: the far plane removes everything beyond it ({} pixels differ from the default view in the prediction, {} on the GPU)",
                                         refs[i].pane.name, changed, gpu_changed));
            }
        }

        // 6c. Matrix upload: readback, transpose = GL_TRUE, and a transposed matrix uploaded as column-major.
        {
            mgl::glUseProgram(program_);
            const Mat4 probe_matrix = persp.proj * persp.view * cube_model;  // an arbitrary non-symmetric matrix
            auto readback_equals = [&](const Mat4& expected) {
                std::array<GLfloat, 16> got{};
                mgl::glGetUniformfv(program_, u_view_, got.data());
                return got == expected.m;
            };
            upload_matrix(u_view_, probe_matrix, Upload::ColumnMajor);
            check.expect(readback_equals(probe_matrix), "glUniformMatrix4fv(transpose = GL_FALSE): glGetUniformfv returns the 16 floats unchanged (column-major)");
            upload_matrix(u_view_, probe_matrix, Upload::RowMajorTranspose);
            check.expect(readback_equals(probe_matrix), "glUniformMatrix4fv(transpose = GL_TRUE) with row-major data reads back as the same column-major matrix");
            upload_matrix(u_view_, probe_matrix, Upload::WrongTranspose);
            check.expect(readback_equals(transposed(probe_matrix)) && !readback_equals(probe_matrix),
                         "negative control: row-major data uploaded with GL_FALSE is stored transposed (a different matrix)");

            Params p = base;
            p.upload = Upload::RowMajorTranspose;
            const Image t_img = rerender(p);
            for (const PaneRef* pr : {&persp, &ortho})
            {
                const Compare c = compare_pane(pr->img, t_img, kTol);
                check.expect(c.bad == 0, std::format("{} pane drawn with transpose = GL_TRUE: {} pixels still agree with the CPU pipeline", pr->pane.name, c.compared),
                             c.first_bad);
            }
            p.upload = Upload::WrongTranspose;
            const Image w_img = rerender(p);
            const int wrong = count_differences(persp.img, w_img, kTol) + count_differences(ortho.img, w_img, kTol);
            check.expect(wrong >= 5000,
                         "negative control: with the view/projection/model transposed (GL_FALSE on row-major data) the picture disagrees with the CPU pipeline",
                         std::format("{} of the compared pixels differ", wrong));
        }

        // ---- 7. Negative controls on the CPU model -------------------------------------------------------------------
        {
            // wrong rotation angle (+0.4 rad): predictions at the cube pixels no longer match the GPU
            const std::vector<Item> wrong_items = make_items(ranges_, layout_, time_, 0.4f);
            const std::vector<PaneRef> wrong_refs = build_refs(wrong_items, base, W, H);
            const int wrong_px = count_differences(wrong_refs[0].img, gpu, kTol);
            check.expect(wrong_px >= 150,
                         "negative control: a CPU model with the cube rotated 0.4 rad too far disagrees with the GPU image",
                         std::format("{} pixels of the perspective pane differ, the correct model has 0", wrong_px));
            int probe_miss = 0;
            for (std::size_t f = 0; f < 6; ++f)
            {
                if (!face_is_facing(wrong_refs[0], cam, wrong_items[kCube].model, f)) continue;
                const Projected c = project_world(wrong_refs[0], transform_point(wrong_items[kCube].model, cube_face_point(f, 0.35f, -0.35f)));
                if (rgb_distance(gpu.at(c.px(), c.py()), cube_face_rgb(wrong_items[kCube].model, f)) > kTol) ++probe_miss;
            }
            check.expect(probe_miss >= 1,
                         "negative control: face-centre probes computed with the wrong rotation fail against the GPU image",
                         std::format("{} of 3 probes miss", probe_miss));

            // wrong projection: predicting the perspective pane with the orthographic matrix
            Pane fake = persp.pane;
            fake.ortho = true;
            const std::vector<PaneRef> fake_refs = build_refs(items, base, W, H, {fake});
            const int fake_px = count_differences(fake_refs[0].img, gpu, kTol);
            check.expect(fake_px >= 1000,
                         "negative control: predicting the perspective pane with the orthographic matrix disagrees with the GPU image",
                         std::format("{} pixels differ", fake_px));
        }

        // ---- 8. Restore the final frame: the runtime captures the framebuffer after validate() ------------------
        draw_frame(env, base, items);
        check.expect_no_gl_error("no GL error during validation");
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-transformations-camera",
        .title = "Transformations and camera",
        .min_es = {3, 0},
        .summary = "A depth-tested 3D scene through model/view/projection matrices, shown with a perspective (top/left) and an "
                   "orthographic (bottom/right) projection; the near plane cuts the foreground. Validated against a CPU rasterizer.",
        .controls = "W/S dolly in/out, A/D orbit, Q/E elevation, P projection (split/perspective/orthographic), "
                    "N/M near plane, F/G far plane, R reset, Esc quit",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 13,
    };
    return demo::run<TransformationsCamera>(argc, argv, info);
}
