// es30-occlusion-queries
//
// Occlusion queries (ES 3.0 §4.1.6) and the query object API (§2.14 Asynchronous Queries, §6.1.7) in a
// small city block. Two wide front buildings (the occluders) stand in front of six towers. Every frame:
//
//   1. draw the ground and the occluders normally (they fill the depth buffer);
//   2. with glColorMask(FALSE...) and glDepthMask(FALSE), draw each tower's bounding box inside its own
//      glBeginQuery(ANY_SAMPLES_PASSED) / glEndQuery pair - nothing is written, only depth-tested;
//   3. read each query result and draw only the towers whose box had at least one sample pass the depth
//      test. Towers that were culled are shown as a translucent red "x-ray" box with a red outline,
//      drawn on top without depth testing, so the viewer sees what the queries removed.
//
// validate() compares the query results with a CPU visibility model (projected rectangles plus depth
// order) in configurations designed to be unambiguous, checks ANY_SAMPLES_PASSED_CONSERVATIVE against what
// the specification requires of it, the availability and blocking result paths, CURRENT_QUERY, glIsQuery
// object creation on the first glBeginQuery, the INVALID_OPERATION cases of §2.14 / §6.1.7, deleting an
// active query, and negative controls (occluders pushed behind the towers, removed, depth cleared to 0).

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
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using demo::math::Mat4;
using demo::math::Vec3;
using demo::math::Vec4;

namespace
{
    /// True when GL_RENDERER or GL_VERSION contains `needle` (selects the driver of a documented deviation).
    bool renderer_has(std::string_view needle)
    {
        auto text = [](mgl::StringName n) {
            const GLubyte* p = mgl::glGetString(n);
            return std::string(p != nullptr ? reinterpret_cast<const char*>(p) : "");
        };
        return (text(mgl::StringName::Renderer) + " | " + text(mgl::StringName::Version)).find(needle) != std::string::npos;
    }

    // ---------------------------------------------------------------------------------------------
    // Scene. World units; y is up, the camera looks straight down -z from kEye. Because the view
    // direction is exactly -z, every face that is perpendicular to z has ONE window depth, which keeps the
    // CPU visibility model simple: a box face "z = const" projects to an axis-aligned rectangle of constant
    // depth.

    struct Box
    {
        Vec3 lo, hi;
    };

    struct Tower
    {
        const char* name;
        Box box;
        std::array<float, 3> color;
    };

    constexpr Vec3 kEye{0.0f, 3.0f, 16.0f};
    constexpr float kFovDegrees = 60.0f;  // along the SHORTER side of the surface
    constexpr float kNear = 1.0f, kFar = 1000.0f;

    // Large enough that its far edges lie within a pixel or two of the horizon, even on a 6:1 surface.
    constexpr Box kGround{{-500.0f, -2.0f, -500.0f}, {500.0f, 0.0f, 15.0f}};

    // The occluders reach 1 unit below the ground. That part is hidden by the ground, but it means every
    // pixel of an occluder's projected front face holds either the occluder face or ground NEARER than it.
    constexpr std::array<Box, 2> kOccluders = {{
        {{-8.0f, -1.0f, 0.0f}, {-2.5f, 6.5f, 2.0f}},  // left block
        {{2.0f, -1.0f, -1.0f}, {7.5f, 5.0f, 1.5f}},   // right block
    }};

    constexpr int kTowerCount = 6;
    constexpr std::array<Tower, kTowerCount> kTowers = {{
        {"T0 behind the left block", {{-6.5f, 0.0f, -6.0f}, {-5.0f, 4.0f, -4.0f}}, {0.20f, 0.55f, 0.85f}},
        {"T1 tall, above the right block", {{3.5f, 0.0f, -8.0f}, {5.5f, 10.0f, -6.0f}}, {0.90f, 0.75f, 0.20f}},
        {"T2 in the street between the blocks", {{-1.0f, 0.0f, -10.0f}, {1.0f, 3.5f, -8.0f}}, {0.30f, 0.75f, 0.40f}},
        {"T3 behind the right block", {{4.0f, 0.0f, -5.0f}, {6.0f, 3.0f, -3.0f}}, {0.75f, 0.35f, 0.80f}},
        {"T4 peeking past the left block", {{-4.5f, 0.0f, -4.0f}, {-1.5f, 4.5f, -2.0f}}, {0.95f, 0.55f, 0.25f}},
        {"T5 close behind the left block", {{-7.5f, 0.0f, -2.0f}, {-6.0f, 6.0f, -0.5f}}, {0.25f, 0.75f, 0.75f}},
    }};

    constexpr std::array<float, 4> kSky = {0.58f, 0.72f, 0.88f, 1.0f};
    constexpr std::array<float, 3> kGroundColor = {0.30f, 0.36f, 0.30f};
    constexpr std::array<float, 3> kOccluderColor = {0.70f, 0.66f, 0.60f};
    constexpr std::array<float, 4> kCulledFill = {0.95f, 0.10f, 0.10f, 0.45f};
    constexpr std::array<float, 4> kCulledEdge = {0.95f, 0.10f, 0.10f, 1.0f};

    // Model margins: a pixel center at least kPixelMargin pixels inside / outside every rectangle edge is
    // unaffected by the rasterization rules of edges (ES 3.0 §3.6.1 only fixes which side of an edge a
    // sample on the edge belongs to), and kDepthMargin world units of depth separation are thousands of
    // depth-buffer steps at the distances used, so no depth comparison is a near tie.
    constexpr float kPixelMargin = 2.0f;
    constexpr float kDepthMargin = 1.0f;
    // Pixel probes stay this far from every outline segment. A 1-pixel line produces fragments whose centers lie
    // within 0.5 of the segment by the diamond-exit rule, and ES 3.0 §3.5.1 lets an implementation deviate by one
    // more pixel in x and y: at most 0.5 + sqrt(2) < 2.
    constexpr float kEdgeClearance = 2.0f;
    constexpr int kMinSurface = 200;

    Vec3 light_direction() { return demo::math::normalize(Vec3{-0.35f, 0.8f, 0.55f}); }

    Mat4 projection(int w, int h)
    {
        // kFovDegrees spans the shorter side, so the whole block stays on screen at any aspect ratio.
        const float f = 1.0f / std::tan(demo::math::radians(kFovDegrees) * 0.5f);
        const float aspect = static_cast<float>(w) / static_cast<float>(h);
        Mat4 p = demo::math::perspective(demo::math::radians(kFovDegrees), 1.0f, kNear, kFar);
        p.at(0, 0) = aspect >= 1.0f ? f / aspect : f;
        p.at(1, 1) = aspect >= 1.0f ? f : f * aspect;
        return p;
    }

    Mat4 view() { return demo::math::look_at(kEye, kEye + Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 1.0f, 0.0f}); }

    Mat4 box_model(const Box& b) { return demo::math::translate(b.lo) * demo::math::scale(b.hi - b.lo); }

    // ---------------------------------------------------------------------------------------------
    // CPU visibility model (window coordinates, origin bottom left, as glReadPixels).

    struct Rect
    {
        float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        [[nodiscard]] Rect grown(float m) const { return {x0 - m, y0 - m, x1 + m, y1 + m}; }
        [[nodiscard]] bool contains(float x, float y) const { return x >= x0 && x <= x1 && y >= y0 && y <= y1; }
        [[nodiscard]] bool contains(const Rect& r) const { return r.x0 >= x0 && r.x1 <= x1 && r.y0 >= y0 && r.y1 <= y1; }
    };

    struct Point
    {
        float x, y;
    };

    Point to_window(const Mat4& pv, Vec3 p, int w, int h)
    {
        const Vec4 c = pv * Vec4{p.x, p.y, p.z, 1.0f};
        return {(c.x / c.w + 1.0f) * 0.5f * static_cast<float>(w), (c.y / c.w + 1.0f) * 0.5f * static_cast<float>(h)};
    }

    std::array<Vec3, 8> corners(const Box& b)
    {
        std::array<Vec3, 8> c{};
        for (int i = 0; i < 8; ++i)
            c[static_cast<std::size_t>(i)] = {(i & 1) ? b.hi.x : b.lo.x, (i & 2) ? b.hi.y : b.lo.y, (i & 4) ? b.hi.z : b.lo.z};
        return c;
    }

    // Bounding rectangle of the whole projected box (every corner is in front of the camera).
    Rect full_rect(const Mat4& pv, const Box& b, int w, int h)
    {
        Rect r{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(),
               std::numeric_limits<float>::lowest()};
        for (const Vec3& c : corners(b))
        {
            const Point p = to_window(pv, c, w, h);
            r = {std::min(r.x0, p.x), std::min(r.y0, p.y), std::max(r.x1, p.x), std::max(r.y1, p.y)};
        }
        return r;
    }

    // The front face (z = hi.z): the nearest part of the box, one constant depth, an exact rectangle.
    Rect front_rect(const Mat4& pv, const Box& b, int w, int h)
    {
        const Point a = to_window(pv, {b.lo.x, b.lo.y, b.hi.z}, w, h);
        const Point c = to_window(pv, {b.hi.x, b.hi.y, b.hi.z}, w, h);
        return {a.x, a.y, c.x, c.y};
    }

    enum class Vis
    {
        Hidden,
        Visible,
        Ambiguous
    };

    const char* vis_name(Vis v) { return v == Vis::Hidden ? "hidden" : v == Vis::Visible ? "visible" : "AMBIGUOUS"; }

    struct Model
    {
        Vis vis = Vis::Ambiguous;
        int occluder = -1;                // hidden: the occluder whose front face covers the whole box
        std::vector<Point> clear_pixels;  // visible: pixel centers where the front face is certainly unoccluded
    };

    // Hidden: some occluder's front face lies at least kDepthMargin in front of the tower's nearest point and
    // its rectangle contains the tower's whole projection with kPixelMargin to spare. Every pixel there holds
    // the occluder face or nearer ground, so no sample of the tower's box can pass DepthFunc(LESS).
    // Visible: some pixel center lies kPixelMargin inside the tower's front face, inside the viewport, and
    // kPixelMargin outside the projection of every occluder that is not entirely behind that face. The depth
    // buffer there holds the clear value or ground behind the face, so that sample passes.
    // Anything else is ambiguous; validate() requires that no designed configuration is.
    Model classify(const Tower& t, std::span<const Box> occluders, int w, int h)
    {
        const Mat4 pv = projection(w, h) * view();
        Model m;
        const Rect full = full_rect(pv, t.box, w, h).grown(kPixelMargin);
        for (std::size_t i = 0; i < occluders.size(); ++i)
        {
            const Box& o = occluders[i];
            if (o.hi.z >= t.box.hi.z + kDepthMargin && front_rect(pv, o, w, h).grown(-kPixelMargin).contains(full))
            {
                m.vis = Vis::Hidden;
                m.occluder = static_cast<int>(i);
                return m;
            }
        }
        const Rect face = front_rect(pv, t.box, w, h).grown(-kPixelMargin);
        std::vector<Rect> blockers;
        for (const Box& o : occluders)
            if (o.hi.z > t.box.hi.z - kDepthMargin) blockers.push_back(full_rect(pv, o, w, h).grown(kPixelMargin));
        const int x_begin = std::max(0, static_cast<int>(std::floor(face.x0)));
        const int x_end = std::min(w, static_cast<int>(std::ceil(face.x1)));
        const int y_begin = std::max(0, static_cast<int>(std::floor(face.y0)));
        const int y_end = std::min(h, static_cast<int>(std::ceil(face.y1)));
        for (int y = y_begin; y < y_end; ++y)
            for (int x = x_begin; x < x_end; ++x)
            {
                const float cx = static_cast<float>(x) + 0.5f, cy = static_cast<float>(y) + 0.5f;
                if (!face.contains(cx, cy)) continue;
                if (std::ranges::any_of(blockers, [&](const Rect& r) { return r.contains(cx, cy); })) continue;
                m.clear_pixels.push_back({cx, cy});
            }
        m.vis = m.clear_pixels.empty() ? Vis::Ambiguous : Vis::Visible;
        return m;
    }

    // Distance from a point to a segment (outline probes).
    float segment_distance(Point p, Point a, Point b)
    {
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float len2 = dx * dx + dy * dy;
        float t = len2 > 0.0f ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2 : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        const float ex = a.x + t * dx - p.x, ey = a.y + t * dy - p.y;
        return std::sqrt(ex * ex + ey * ey);
    }

    // The 12 edges of the unit cube as corner-index pairs (corner bit 0 = x, bit 1 = y, bit 2 = z).
    constexpr std::array<std::array<int, 2>, 12> kEdges = {{
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
    }};

    // Flat shading of a face with normal +z (the only faces probed), exactly as scene.frag.
    std::array<float, 3> shade_front(const std::array<float, 3>& c)
    {
        const float shade = 0.35f + 0.65f * std::max(light_direction().z, 0.0f);
        return {c[0] * shade, c[1] * shade, c[2] * shade};
    }

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

    struct Vertex
    {
        float px, py, pz, nx, ny, nz;
    };

    // 36 vertices: the unit cube as 12 counter-clockwise (outward-facing) triangles, then 24 vertices: its
    // 12 edges as GL_LINES.
    std::vector<Vertex> make_cube()
    {
        struct Face
        {
            Vec3 n, u, v;  // u x v = n, so (-u-v, u-v, u+v, -u+v) is counter-clockwise seen from outside
        };
        const std::array<Face, 6> faces = {{
            {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},
            {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
            {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},
            {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
            {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
            {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
        }};
        std::vector<Vertex> out;
        for (const Face& f : faces)
        {
            const Vec3 c = Vec3{0.5f, 0.5f, 0.5f} + f.n * 0.5f;
            const std::array<Vec3, 4> q = {c - f.u * 0.5f - f.v * 0.5f, c + f.u * 0.5f - f.v * 0.5f, c + f.u * 0.5f + f.v * 0.5f,
                                           c - f.u * 0.5f + f.v * 0.5f};
            for (int i : {0, 1, 2, 0, 2, 3})
            {
                const Vec3& p = q[static_cast<std::size_t>(i)];
                out.push_back({p.x, p.y, p.z, f.n.x, f.n.y, f.n.z});
            }
        }
        const Box unit{{0, 0, 0}, {1, 1, 1}};
        const auto uc = corners(unit);
        for (const auto& e : kEdges)
            for (int i : e)
            {
                const Vec3& p = uc[static_cast<std::size_t>(i)];
                out.push_back({p.x, p.y, p.z, 0.0f, 0.0f, 1.0f});
            }
        return out;
    }

    constexpr GLint kCubeFirst = 0, kCubeCount = 36, kEdgeFirst = 36, kEdgeCount = 24;

    using Clock = std::chrono::steady_clock;
    constexpr auto kPollBudget = std::chrono::seconds(10);

    class OcclusionQueries final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (std::min(env.width, env.height) < kMinSurface)
                throw demo::Skip(std::format("surface {}x{} is too small: the visibility model needs at least {} pixels on the shorter side",
                                             env.width, env.height, kMinSurface));

            program_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            u_mvp_ = mgl::glGetUniformLocation(program_, "u_mvp");
            u_color_ = mgl::glGetUniformLocation(program_, "u_color");
            u_light_ = mgl::glGetUniformLocation(program_, "u_light");
            u_lit_ = mgl::glGetUniformLocation(program_, "u_lit");

            const std::vector<Vertex> cube = make_cube();
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(cube), mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, px)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, nx)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // One query object per tower and target. A name from glGenQueries is only reserved; the object
            // is created by its first glBeginQuery, which also fixes its type (target) for good.
            mgl::glGenQueries(std::span<mgl::QueryId>(any_));
            mgl::glGenQueries(std::span<mgl::QueryId>(conservative_));
        }

        void render(demo::Env& env, int frame) override
        {
            float shift = 0.0f;
            if (env.interactive && animate_) shift = 3.2f * std::sin(static_cast<float>(frame) / 60.0f * 0.7f);
            std::vector<Box> occluders;
            if (occluders_on_)
                for (Box o : kOccluders)
                {
                    o.lo.x += shift;
                    o.hi.x += shift;
                    occluders.push_back(o);
                }
            draw_frame(env, occluders);
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            if (key == ' ') animate_ = !animate_;
            if (key == 'o') occluders_on_ = !occluders_on_;
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_object_api(check);

            // --- 1. The scene, twice: results read with the blocking QUERY_RESULT alone, then results read only after
            //        QUERY_RESULT_AVAILABLE was polled to TRUE. Both against the CPU model.
            const std::span<const Box> scene(kOccluders);
            const Results blocking_r = run_queries(env, scene, 1.0f, false, check, "scene, blocking");
            const auto scene_m = compare(env, scene, blocking_r, false, check, "scene, blocking");
            int hidden = 0, visible = 0;
            for (const Model& m : scene_m) (m.vis == Vis::Hidden ? hidden : visible) += m.vis != Vis::Ambiguous ? 1 : 0;
            check.expect(hidden >= 2 && visible >= 2, "scene: the design has at least two hidden and two visible towers",
                         std::format("hidden={} visible={}", hidden, visible));
            const Results polled_r = run_queries(env, scene, 1.0f, true, check, "scene, polled");
            compare(env, scene, polled_r, false, check, "scene, polled");

            // --- 2. Negative control: the same occluders pushed 30 units back, behind every tower. Their
            //        rectangles still overlap the towers on screen, but they are now farther away, so every
            //        tower that was hidden must flip to visible. Results read with the blocking QUERY_RESULT.
            std::vector<Box> pushed(kOccluders.begin(), kOccluders.end());
            for (Box& o : pushed)
            {
                o.lo.z -= 30.0f;
                o.hi.z -= 30.0f;
            }
            const Results pushed_r = run_queries(env, pushed, 1.0f, false, check, "occluders behind");
            const auto pushed_m = compare(env, pushed, pushed_r, false, check, "occluders behind");
            for (std::size_t i = 0; i < kTowers.size(); ++i)
                if (scene_m[i].vis == Vis::Hidden)
                    check.expect(blocking_r.any[i] == 0u && pushed_r.any[i] != 0u && pushed_m[i].vis == Vis::Visible,
                                 std::format("negative control: {} flips from culled to visible when the occluders move behind it", kTowers[i].name),
                                 std::format("scene={} pushed={}", blocking_r.any[i], pushed_r.any[i]));

            // --- 3. No occluders at all (only the ground): every tower is visible.
            const Results bare_r = run_queries(env, {}, 1.0f, false, check, "no occluders");
            const auto bare_m = compare(env, {}, bare_r, false, check, "no occluders");
            for (std::size_t i = 0; i < kTowers.size(); ++i)
                check.expect(bare_m[i].vis == Vis::Visible, std::format("no occluders: the model sees {}", kTowers[i].name));

            // --- 4. Negative control for the visible towers: depth cleared to 0.0 and nothing drawn before the
            //        queries. With DepthFunc(LESS) no fragment can pass, so every ANY_SAMPLES_PASSED is FALSE -
            //        including the towers that are visible in the scene.
            const Results zero_r = run_queries(env, {}, 0.0f, false, check, "depth cleared to 0");
            compare(env, {}, zero_r, true, check, "depth cleared to 0");

            // --- 5. The rendered frame: draw only what passed, mark what was culled. Re-rendered last, so this
            //        is also the captured image.
            check_final_image(env, check, scene_m);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glDeleteQueries(std::span<const mgl::QueryId>(any_));
            mgl::glDeleteQueries(std::span<const mgl::QueryId>(conservative_));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteProgram(program_);
        }

    private:
        struct Results
        {
            std::array<GLuint, kTowerCount> any{};
            std::array<GLuint, kTowerCount> conservative{};
            bool polled = false;  // QUERY_RESULT was read after QUERY_RESULT_AVAILABLE reported TRUE
        };

        void set_color(const std::array<float, 3>& c, float alpha, float lit)
        {
            mgl::glUniform4f(u_color_, c[0], c[1], c[2], alpha);
            mgl::glUniform1f(u_lit_, lit);
        }

        void draw_box(const Mat4& pv, const Box& b, GLint first, GLint count, mgl::PrimitiveType mode)
        {
            const Mat4 mvp = pv * box_model(b);
            mgl::glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
            mgl::glDrawArrays(mode, first, count);
        }

        void draw_solid(const Mat4& pv, const Box& b) { draw_box(pv, b, kCubeFirst, kCubeCount, mgl::PrimitiveType::Triangles); }

        // Clears, then draws the ground and the occluders with depth writes: the depth buffer the queries test.
        Mat4 draw_occluders(const demo::Env& env, std::span<const Box> occluders, float clear_depth)
        {
            const Mat4 pv = projection(env.width, env.height) * view();
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClearColor(kSky[0], kSky[1], kSky[2], kSky[3]);
            mgl::glClearDepthf(clear_depth);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glClearDepthf(1.0f);

            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            const Vec3 l = light_direction();
            mgl::glUniform3f(u_light_, l.x, l.y, l.z);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glFrontFace(mgl::FrontFace::CCW);
            mgl::glCullFace(mgl::CullFace::Back);
            mgl::glEnable(mgl::Capability::CullFace);
            if (clear_depth > 0.0f)
            {
                set_color(kGroundColor, 1.0f, 1.0f);
                draw_solid(pv, kGround);
                set_color(kOccluderColor, 1.0f, 1.0f);
                for (const Box& o : occluders) draw_solid(pv, o);
            }
            return pv;
        }

        // Bounding-box pass: depth-tested only. Color and depth writes are off, so the boxes leave no trace and
        // do not occlude each other; back faces are not culled, so a box still produces fragments if the camera
        // were inside it.
        void begin_box_pass()
        {
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            mgl::glDepthMask(GL_FALSE);
            mgl::glDisable(mgl::Capability::CullFace);
        }

        void end_box_pass()
        {
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glDepthMask(GL_TRUE);
            mgl::glEnable(mgl::Capability::CullFace);
        }

        // The frame drawn by render(): returns which towers passed their query.
        std::array<bool, kTowerCount> draw_frame(const demo::Env& env, std::span<const Box> occluders)
        {
            const Mat4 pv = draw_occluders(env, occluders, 1.0f);

            begin_box_pass();
            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                mgl::glBeginQuery(mgl::QueryTarget::AnySamplesPassed, any_[i]);
                draw_solid(pv, kTowers[i].box);
                mgl::glEndQuery(mgl::QueryTarget::AnySamplesPassed);
            }
            end_box_pass();

            // QUERY_RESULT waits for the result. A real renderer would read last frame's results (or poll
            // QUERY_RESULT_AVAILABLE) instead of stalling here; see README.md.
            std::array<bool, kTowerCount> passed{};
            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                GLuint r = 0;
                mgl::glGetQueryObjectuiv(any_[i], mgl::QueryObjectParameter::Result, &r);
                passed[i] = r != 0;
            }

            for (std::size_t i = 0; i < kTowers.size(); ++i)
                if (passed[i])
                {
                    set_color(kTowers[i].color, 1.0f, 1.0f);
                    draw_solid(pv, kTowers[i].box);
                }

            // Culled towers: translucent fill (front faces only, so every interior pixel is blended exactly once)
            // and an opaque outline, both on top of everything. Destination alpha is kept at 1.
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::Blend);
            mgl::glBlendFuncSeparate(mgl::BlendFactor::SrcAlpha, mgl::BlendFactor::OneMinusSrcAlpha, mgl::BlendFactor::Zero, mgl::BlendFactor::One);
            set_color({kCulledFill[0], kCulledFill[1], kCulledFill[2]}, kCulledFill[3], 0.0f);
            for (std::size_t i = 0; i < kTowers.size(); ++i)
                if (!passed[i]) draw_solid(pv, kTowers[i].box);
            mgl::glDisable(mgl::Capability::Blend);
            set_color({kCulledEdge[0], kCulledEdge[1], kCulledEdge[2]}, kCulledEdge[3], 0.0f);
            for (std::size_t i = 0; i < kTowers.size(); ++i)
                if (!passed[i]) draw_box(pv, kTowers[i].box, kEdgeFirst, kEdgeCount, mgl::PrimitiveType::Lines);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::CullFace);
            return passed;
        }

        // Issues one ANY_SAMPLES_PASSED and one ANY_SAMPLES_PASSED_CONSERVATIVE query per tower against the given
        // occluders, then collects the results either by polling QUERY_RESULT_AVAILABLE first or by calling the
        // blocking QUERY_RESULT directly.
        Results run_queries(const demo::Env& env, std::span<const Box> occluders, float clear_depth, bool poll, demo::Check& check,
                            const std::string& label)
        {
            const Mat4 pv = draw_occluders(env, occluders, clear_depth);
            begin_box_pass();
            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                mgl::glBeginQuery(mgl::QueryTarget::AnySamplesPassed, any_[i]);
                draw_solid(pv, kTowers[i].box);
                mgl::glEndQuery(mgl::QueryTarget::AnySamplesPassed);
            }
            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                mgl::glBeginQuery(mgl::QueryTarget::AnySamplesPassedConservative, conservative_[i]);
                draw_solid(pv, kTowers[i].box);
                mgl::glEndQuery(mgl::QueryTarget::AnySamplesPassedConservative);
            }
            end_box_pass();
            check.expect_no_gl_error(std::format("{}: 2x{} Begin/EndQuery pairs raise no error", label, kTowerCount));

            if (poll)
            {
                // §6.1.7: repeatedly querying QUERY_RESULT_AVAILABLE "is guaranteed to return true eventually"
                // (no glFlush needed), and once a query is available, every earlier query of the same type is too.
                for (const auto* names : {&any_, &conservative_})
                {
                    const bool is_any = names == &any_;
                    const mgl::QueryId last = names->back();
                    GLuint available = GL_FALSE;
                    int polls = 0;
                    const auto start = Clock::now();
                    while (Clock::now() - start < kPollBudget)
                    {
                        mgl::glGetQueryObjectuiv(last, mgl::QueryObjectParameter::ResultAvailable, &available);
                        ++polls;
                        if (available == GL_TRUE) break;
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                    const char* target = is_any ? "ANY_SAMPLES_PASSED" : "ANY_SAMPLES_PASSED_CONSERVATIVE";
                    std::cout << std::format("  [{}] {}: last query available after {} poll(s)\n", label, target, polls);
                    check.expect(available == GL_TRUE, std::format("{}: {} QUERY_RESULT_AVAILABLE becomes TRUE by polling alone", label, target),
                                 std::format("still {} after {} polls", available, polls));
                    int earlier_available = 0;
                    for (std::size_t i = 0; i + 1 < names->size(); ++i)
                    {
                        GLuint a = GL_FALSE;
                        mgl::glGetQueryObjectuiv((*names)[i], mgl::QueryObjectParameter::ResultAvailable, &a);
                        earlier_available += a == GL_TRUE ? 1 : 0;
                    }
                    check.expect_or_known_deviation(earlier_available == kTowerCount - 1,
                                                    std::format("{}: every {} query issued before an available one is available", label, target),
                                                    std::format("{} of {} earlier queries available", earlier_available, kTowerCount - 1), "zink",
                                                    "zink reports QUERY_RESULT_AVAILABLE = FALSE for queries issued before one that already reports TRUE "
                                                    "(ES 3.0 §6.1.7 requires all earlier queries of the same type to report TRUE)");
                    // Whatever the order, each of them must become available by polling alone.
                    int still_pending = 0;
                    for (std::size_t i = 0; i + 1 < names->size(); ++i)
                    {
                        GLuint a = GL_FALSE;
                        const auto t0 = Clock::now();
                        do
                        {
                            mgl::glGetQueryObjectuiv((*names)[i], mgl::QueryObjectParameter::ResultAvailable, &a);
                            if (a != GL_TRUE) std::this_thread::sleep_for(std::chrono::microseconds(200));
                        } while (a != GL_TRUE && Clock::now() - t0 < kPollBudget);
                        still_pending += a == GL_TRUE ? 0 : 1;
                    }
                    check.expect_eq(still_pending, 0, std::format("{}: every earlier {} query becomes available by polling alone", label, target));
                }
            }

            Results r;
            r.polled = poll;
            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                r.any[i] = 0xDEADu;
                r.conservative[i] = 0xDEADu;
                mgl::glGetQueryObjectuiv(any_[i], mgl::QueryObjectParameter::Result, &r.any[i]);
                mgl::glGetQueryObjectuiv(conservative_[i], mgl::QueryObjectParameter::Result, &r.conservative[i]);
            }
            check.expect_no_gl_error(std::format("{}: QUERY_RESULT of every query raises no error", label));
            return r;
        }

        std::array<Model, kTowerCount> compare(const demo::Env& env, std::span<const Box> occluders, const Results& r, bool all_fail,
                                               demo::Check& check, const std::string& label)
        {
            std::array<Model, kTowerCount> models;
            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                const Tower& t = kTowers[i];
                Model& m = models[i];
                if (!all_fail)
                {
                    m = classify(t, occluders, env.width, env.height);
                    check.expect(m.vis != Vis::Ambiguous, std::format("{}: {} is unambiguously hidden or visible in the model", label, t.name));
                }
                const bool want = !all_fail && m.vis == Vis::Visible;
                std::cout << std::format("  [{}] {:<38} model {:<9} ANY_SAMPLES_PASSED={} CONSERVATIVE={}\n", label, t.name,
                                         all_fail ? "no-pass" : vis_name(m.vis), r.any[i], r.conservative[i]);
                // Results read after polling are compared strictly too, except on zink, whose polled results are wrong
                // (KNOWN_LIMITATIONS.md); its blocking results are compared strictly like every other driver's.
                auto record = [&](bool ok, const std::string& what, const std::string& detail) {
                    if (r.polled)
                        check.expect_or_known_deviation(ok, what, detail, "zink",
                                                        "zink: once QUERY_RESULT_AVAILABLE has been polled to TRUE, QUERY_RESULT of an occlusion query is a "
                                                        "partial sample count (e.g. 30 or 196 for a box covering thousands of pixels, 0 for a visible box) "
                                                        "instead of the samples-boolean (ES 3.0 §4.1.6, §6.1.7); blocking reads are correct");
                    else
                        check.expect(ok, what, detail);
                };
                if (all_fail || m.vis != Vis::Ambiguous)
                    record((r.any[i] != 0u) == want, std::format("{}: ANY_SAMPLES_PASSED of {} is {} as in the model", label, t.name, want ? "TRUE" : "FALSE"),
                           std::format("result {}", r.any[i]));
                // §4.1.6: "the samples-boolean state of FALSE or TRUE is written to the corresponding query object as the
                // query result value" - for both targets. CONSERVATIVE may additionally set it to TRUE in
                // implementation-dependent cases, so it is never FALSE when a sample passed.
                record(r.any[i] <= 1u, std::format("{}: ANY_SAMPLES_PASSED result of {} is exactly FALSE (0) or TRUE (1)", label, t.name),
                       std::format("{}", r.any[i]));
                record(r.conservative[i] <= 1u, std::format("{}: ANY_SAMPLES_PASSED_CONSERVATIVE result of {} is exactly FALSE (0) or TRUE (1)", label, t.name),
                       std::format("{}", r.conservative[i]));
                if (r.any[i] != 0u || want)
                    record(r.conservative[i] != 0u,
                           std::format("{}: ANY_SAMPLES_PASSED_CONSERVATIVE of {} is TRUE because samples passed", label, t.name),
                           std::format("ANY={} CONSERVATIVE={}", r.any[i], r.conservative[i]));
            }
            return models;
        }

        // glIsQuery, CURRENT_QUERY and every INVALID_OPERATION case of ES 3.0 §2.14 and §6.1.7, on fresh names.
        void check_object_api(demo::Check& check)
        {
            const mgl::QueryTarget any = mgl::QueryTarget::AnySamplesPassed;
            const mgl::QueryTarget cons = mgl::QueryTarget::AnySamplesPassedConservative;
            const mgl::QueryTarget tf = mgl::QueryTarget::TransformFeedbackPrimitivesWritten;
            auto current = [](mgl::QueryTarget target) {
                GLint v = -1;
                mgl::glGetQueryiv(target, mgl::QueryParameter::CurrentQuery, &v);
                return v;
            };
            auto id = [](mgl::QueryId q) { return static_cast<GLint>(q.value); };

            std::array<mgl::QueryId, 5> q{};
            mgl::glGenQueries(std::span<mgl::QueryId>(q));
            check.expect_no_gl_error("glGenQueries(5)");
            bool distinct = true;
            for (std::size_t i = 0; i < q.size(); ++i)
                for (std::size_t j = i + 1; j < q.size(); ++j) distinct = distinct && q[i].value != q[j].value;
            check.expect(distinct && std::ranges::none_of(q, [](mgl::QueryId x) { return x.value == 0; }), "glGenQueries returns 5 distinct non-zero names");
            const mgl::QueryId qa = q[0], qb = q[1], qc = q[2], qd = q[3], qe = q[4];

            // §2.14: GenQueries marks names used, "but no object is associated with them until the first time they
            // are used by BeginQuery"; §6.1.7: IsQuery is TRUE only for the name of a query object.
            check.expect(!mgl::glIsQuery(qa), "glIsQuery is FALSE for a generated name before its first glBeginQuery");
            check.expect(!mgl::glIsQuery(mgl::QueryId{0}), "glIsQuery(0) is FALSE");
            GLuint value = 0;
            mgl::glGetQueryObjectuiv(qa, mgl::QueryObjectParameter::ResultAvailable, &value);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetQueryObjectuiv on a generated but never begun name -> INVALID_OPERATION");

            check.expect_eq(current(any), 0, "CURRENT_QUERY(ANY_SAMPLES_PASSED) is 0 outside Begin/End");
            check.expect_eq(current(cons), 0, "CURRENT_QUERY(ANY_SAMPLES_PASSED_CONSERVATIVE) is 0 outside Begin/End");
            check.expect_eq(current(tf), 0, "CURRENT_QUERY(TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) is 0 outside Begin/End");

            // --- An active ANY_SAMPLES_PASSED query.
            mgl::glBeginQuery(any, qa);
            check.expect_no_gl_error("glBeginQuery(ANY_SAMPLES_PASSED, qa)");
            check.expect(mgl::glIsQuery(qa), "glIsQuery is TRUE once glBeginQuery created the object");
            check.expect_eq(current(any), id(qa), "CURRENT_QUERY(ANY_SAMPLES_PASSED) is qa inside Begin/End");
            // The two occlusion targets share one active-query state (§2.14: "the required state for occlusion
            // queries is shared"), so whether CURRENT_QUERY(ANY_SAMPLES_PASSED_CONSERVATIVE) reports qa now is not
            // pinned down by the text. Printed, not asserted.
            std::cout << std::format("  [api] while ANY_SAMPLES_PASSED query {} is active: CURRENT_QUERY(ANY_SAMPLES_PASSED_CONSERVATIVE) = {} (not asserted)\n",
                                     qa.value, current(cons));
            check.expect_eq(current(tf), 0, "CURRENT_QUERY(TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) stays 0 while an occlusion query is active");

            mgl::glBeginQuery(any, qb);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery on a target whose query is active -> INVALID_OPERATION");
            check.expect(!mgl::glIsQuery(qb), "the failed glBeginQuery did not create an object for qb");
            mgl::glBeginQuery(cons, qb);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glBeginQuery(ANY_SAMPLES_PASSED_CONSERVATIVE) while ANY_SAMPLES_PASSED is active -> INVALID_OPERATION (either target counts)");
            mgl::glBeginQuery(tf, qa);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery with a name that is the active query of another target -> INVALID_OPERATION");
            check.expect_eq(current(any), id(qa), "the failed glBeginQuery calls left CURRENT_QUERY(ANY_SAMPLES_PASSED) = qa");
            check.expect_eq(current(tf), 0, "the failed glBeginQuery calls left CURRENT_QUERY(TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) = 0");

            value = 0xDEADu;
            mgl::glGetQueryObjectuiv(qa, mgl::QueryObjectParameter::Result, &value);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetQueryObjectuiv(QUERY_RESULT) on the active query -> INVALID_OPERATION");
            mgl::glGetQueryObjectuiv(qa, mgl::QueryObjectParameter::ResultAvailable, &value);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetQueryObjectuiv(QUERY_RESULT_AVAILABLE) on the active query -> INVALID_OPERATION");

            // (glEndQuery(ANY_SAMPLES_PASSED_CONSERVATIVE) is not called here: with the shared occlusion state, whether it
            // ends qa or raises INVALID_OPERATION is not pinned down either.)
            mgl::glEndQuery(any);
            check.expect_no_gl_error("glEndQuery(ANY_SAMPLES_PASSED)");
            check.expect_eq(current(any), 0, "CURRENT_QUERY(ANY_SAMPLES_PASSED) is 0 again after glEndQuery");
            check.expect(mgl::glIsQuery(qa), "glIsQuery stays TRUE after glEndQuery");
            mgl::glGetQueryObjectuiv(qa, mgl::QueryObjectParameter::Result, &value);
            check.expect_no_gl_error("glGetQueryObjectuiv(QUERY_RESULT) after glEndQuery");
            check.expect_eq(value, 0u, "an occlusion query with no draw between Begin and End is FALSE");

            // --- Errors with nothing active.
            mgl::glEndQuery(any);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glEndQuery without an active query -> INVALID_OPERATION");
            mgl::glBeginQuery(any, mgl::QueryId{0});
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery with id 0 -> INVALID_OPERATION");
            check.expect_eq(current(any), 0, "the failed glBeginQuery(id 0) activated nothing");
            // A query object's type is fixed by its first glBeginQuery. ES 3.0 §2.14 knows two query TYPES, primitive
            // and occlusion; ANY_SAMPLES_PASSED and ANY_SAMPLES_PASSED_CONSERVATIVE are both occlusion. ES 3.1/3.2 §4.2
            // spell it out ("occlusion query objects specified by either of the two targets ... may be reused for
            // either target", a clarification per ES 3.2 Appendix E.2, Bug 13342). So an occlusion object moves between
            // the two occlusion targets freely, but an occlusion object on the primitives target is a type mismatch.
            mgl::glBeginQuery(tf, qa);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glBeginQuery(TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) on an occlusion query object -> INVALID_OPERATION (type mismatch)");
            check.expect_eq(current(tf), 0, "the type-mismatch glBeginQuery activated nothing");
            mgl::glBeginQuery(cons, qa);
            const mgl::ErrorCode reuse_error = demo::drain_gl_errors();
            const GLint reuse_current = current(cons);
            if (reuse_current != 0) mgl::glEndQuery(cons);
            // Re-verified because Mesa AND ANGLE both raise INVALID_OPERATION here: the reading stands. ES 3.0 §2.14 "There
            // are two query types supported by the GL" (primitive and occlusion) and the error is for "an existing query
            // object whose type does not match target"; ES 3.1 §4.2 (update of January 29, 2015) and ES 3.2 §4.2 add
            // "occlusion query objects specified by either of the two targets ANY_SAMPLES_PASSED or ANY_SAMPLES_PASSED_-
            // CONSERVATIVE may be reused for either target in future queries", and the change log states the intent:
            // "make clear that occlusion query objects may be specified and reused with any mix of the two valid
            // occlusion query targets (Bug 13342)". Both implementations key the type on the target (Mesa
            // q->Target != target, ANGLE getType() != target), i.e. the wording before that clarification.
            const char* const reuse_driver = renderer_has("ANGLE") ? "ANGLE" : "Mesa";
            check.expect_or_known_deviation(reuse_error == mgl::ErrorCode::NoError && reuse_current == id(qa),
                                            "an ANY_SAMPLES_PASSED query object can be reused with ANY_SAMPLES_PASSED_CONSERVATIVE",
                                            std::format("glBeginQuery -> {}, CURRENT_QUERY = {}", demo::error_name(reuse_error), reuse_current), reuse_driver,
                                            std::format("{} treats each occlusion target as its own query type and raises INVALID_OPERATION (ES 3.2 §4.2 allows "
                                                        "reusing an occlusion query object with either occlusion target, Bug 13342)", reuse_driver));
            check.expect_no_gl_error("the reuse attempt left no error behind");
            check.expect_eq(current(cons), 0, "no query is active after the reuse attempt");
            const mgl::QueryId never_generated{std::max({qa.value, qb.value, qc.value, qd.value, qe.value}) + 1000u};
            mgl::glBeginQuery(any, never_generated);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery with a name never returned by glGenQueries -> INVALID_OPERATION");
            check.expect(!mgl::glIsQuery(never_generated), "glIsQuery is FALSE for a name never returned by glGenQueries");

            // --- Different query types may be active at the same time; a name active on one cannot start another.
            mgl::glBeginQuery(tf, qb);
            check.expect_no_gl_error("glBeginQuery(TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qb)");
            mgl::glBeginQuery(any, qb);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery(ANY_SAMPLES_PASSED) with the name of the active primitives query -> INVALID_OPERATION");
            mgl::glBeginQuery(any, qc);
            check.expect_no_gl_error("an occlusion query can be active together with a primitives query");
            check.expect(current(any) == id(qc) && current(tf) == id(qb), "CURRENT_QUERY reports qc and qb for the two active targets",
                         std::format("any={} tf={}", current(any), current(tf)));
            mgl::glEndQuery(any);
            mgl::glEndQuery(tf);
            check.expect_no_gl_error("both queries end without error");
            mgl::glGetQueryObjectuiv(qb, mgl::QueryObjectParameter::Result, &value);
            check.expect_eq(value, 0u, "the primitives query with no transform feedback active counted 0 primitives");
            mgl::glBeginQuery(any, qb);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery(ANY_SAMPLES_PASSED) on a primitives query object -> INVALID_OPERATION (type mismatch)");
            check.expect_eq(current(any), 0, "that glBeginQuery activated nothing");

            // --- Deleting an active query (§2.14 and Appendix D.1.3). The name "immediately becomes unused, but the
            //     underlying object is not deleted until it is no longer active": it is neither an error nor
            //     an implicit EndQuery, so the query stays active and glEndQuery still ends it.
            mgl::glBeginQuery(any, qd);
            mgl::glDeleteQueries(1, &qd);
            check.expect_no_gl_error("glDeleteQueries on an active query raises no error");
            check.expect(!mgl::glIsQuery(qd), "the deleted active query's name is unused at once: glIsQuery FALSE");
            const GLint after_delete = current(any);
            mgl::glEndQuery(any);
            const mgl::ErrorCode end_error = demo::drain_gl_errors();
            std::cout << std::format("  [api] after deleting the active query: CURRENT_QUERY = {} (name was {}), glEndQuery -> {}\n",
                                     after_delete, qd.value, demo::error_name(end_error));
            check.expect_or_known_deviation(after_delete == id(qd) && end_error == mgl::ErrorCode::NoError,
                                            "deleting an active query does not end it: CURRENT_QUERY keeps the name and glEndQuery succeeds",
                                            std::format("CURRENT_QUERY={} glEndQuery->{}", after_delete, demo::error_name(end_error)), "Mesa",
                                            "Mesa's glDeleteQueries ends an active query implicitly (ES 3.0 §2.14 keeps it active until glEndQuery)");
            check.expect_eq(current(any), 0, "no occlusion query is active after the deleted query ended");
            mgl::glBeginQuery(any, qd);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBeginQuery with a deleted name -> INVALID_OPERATION");
            mgl::glGetQueryObjectuiv(qd, mgl::QueryObjectParameter::ResultAvailable, &value);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetQueryObjectuiv on a deleted name -> INVALID_OPERATION");

            // §2.14: unused names and zero in glDeleteQueries are silently ignored.
            const std::array<mgl::QueryId, 3> ignored = {mgl::QueryId{0}, never_generated, qd};
            mgl::glDeleteQueries(std::span<const mgl::QueryId>(ignored));
            check.expect_no_gl_error("glDeleteQueries ignores 0, never-generated and already deleted names");

            const std::array<mgl::QueryId, 4> rest = {qa, qb, qc, qe};
            mgl::glDeleteQueries(std::span<const mgl::QueryId>(rest));
            check.expect(!mgl::glIsQuery(qa) && !mgl::glIsQuery(qb) && !mgl::glIsQuery(qc), "glIsQuery is FALSE after glDeleteQueries");
            check.expect_no_gl_error("query object API section leaves no error");
        }

        void check_final_image(const demo::Env& env, demo::Check& check, const std::array<Model, kTowerCount>& model)
        {
            const std::array<bool, kTowerCount> passed = draw_frame(env, kOccluders);
            check.expect_no_gl_error("the frame (queries, conditional draws, culled overlay) raises no error");
            for (std::size_t i = 0; i < kTowers.size(); ++i)
                check.expect_eq(passed[i], model[i].vis == Vis::Visible, std::format("frame: {} drawn exactly when the model says visible", kTowers[i].name));

            const int w = env.width, h = env.height;
            const Mat4 pv = projection(w, h) * view();
            // Every outline segment of a culled tower, and the full rectangles of all culled towers (their overlay
            // is drawn on top of everything).
            std::vector<std::array<Point, 2>> segments;
            std::vector<Rect> overlay;
            for (std::size_t i = 0; i < kTowers.size(); ++i)
                if (model[i].vis == Vis::Hidden)
                {
                    const auto c = corners(kTowers[i].box);
                    for (const auto& e : kEdges)
                        segments.push_back({to_window(pv, c[static_cast<std::size_t>(e[0])], w, h), to_window(pv, c[static_cast<std::size_t>(e[1])], w, h)});
                    overlay.push_back(full_rect(pv, kTowers[i].box, w, h));
                }
            auto near_segment = [&](Point p) {
                return std::ranges::any_of(segments, [&](const auto& s) { return segment_distance(p, s[0], s[1]) < kEdgeClearance; });
            };
            auto pick = [](const std::vector<Point>& c) {
                float sx = 0, sy = 0;
                for (const Point& p : c) sx += p.x, sy += p.y;
                const Point mean{sx / static_cast<float>(c.size()), sy / static_cast<float>(c.size())};
                return *std::ranges::min_element(c, {}, [&](const Point& p) { return std::hypot(p.x - mean.x, p.y - mean.y); });
            };

            for (std::size_t i = 0; i < kTowers.size(); ++i)
            {
                const Tower& t = kTowers[i];
                std::vector<Point> candidates;
                Rgba8 want{};
                if (model[i].vis == Vis::Visible)
                {
                    // An unoccluded pixel of the front face, away from every culled overlay and from towers that could
                    // be in front of this one.
                    std::vector<Rect> exclude;
                    for (const Rect& r : overlay) exclude.push_back(r.grown(kPixelMargin));
                    for (std::size_t j = 0; j < kTowers.size(); ++j)
                        if (j != i && kTowers[j].box.hi.z > t.box.hi.z - kDepthMargin) exclude.push_back(full_rect(pv, kTowers[j].box, w, h).grown(kPixelMargin));
                    for (const Point& p : model[i].clear_pixels)
                        if (std::ranges::none_of(exclude, [&](const Rect& r) { return r.contains(p.x, p.y); })) candidates.push_back(p);
                    const auto c = shade_front(t.color);
                    want = {to8(c[0]), to8(c[1]), to8(c[2]), 255};
                }
                else if (model[i].vis == Vis::Hidden)
                {
                    // A pixel of the culled overlay over the occluder's front face, away from the outline and from
                    // the other culled towers. Expected: the 8-bit occluder color blended once with the fill
                    // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA; alpha uses ZERO, ONE and stays 1).
                    const Rect face = front_rect(pv, t.box, w, h).grown(-kPixelMargin);
                    std::vector<Rect> exclude;
                    for (std::size_t j = 0; j < kTowers.size(); ++j)
                        if (j != i) exclude.push_back(full_rect(pv, kTowers[j].box, w, h).grown(kPixelMargin));
                    for (int y = std::max(0, static_cast<int>(face.y0)); y < std::min(h, static_cast<int>(face.y1) + 1); ++y)
                        for (int x = std::max(0, static_cast<int>(face.x0)); x < std::min(w, static_cast<int>(face.x1) + 1); ++x)
                        {
                            const Point p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                            if (face.contains(p.x, p.y) && !near_segment(p) && std::ranges::none_of(exclude, [&](const Rect& r) { return r.contains(p.x, p.y); }))
                                candidates.push_back(p);
                        }
                    const auto oc = shade_front(kOccluderColor);
                    const float a = kCulledFill[3];
                    want = {to8(static_cast<float>(to8(oc[0])) / 255.0f * (1.0f - a) + kCulledFill[0] * a),
                            to8(static_cast<float>(to8(oc[1])) / 255.0f * (1.0f - a) + kCulledFill[1] * a),
                            to8(static_cast<float>(to8(oc[2])) / 255.0f * (1.0f - a) + kCulledFill[2] * a), 255};
                }
                else
                    continue;
                if (!check.expect(!candidates.empty(), std::format("frame: {} has a probe pixel", t.name))) continue;
                const Point p = pick(candidates);
                // Tolerance: +-1 for the conversion of the shaded float color to 8 bits (ES 3.0 §2.1.6.2); the blended
                // probe adds a second conversion (the stored destination), hence +-2.
                const int tol = model[i].vis == Vis::Visible ? 1 : 2;
                check.expect_pixel(static_cast<int>(p.x), static_cast<int>(p.y), want, tol,
                                   std::format("frame: {} pixel ({}, {}) shows {}", t.name, static_cast<int>(p.x), static_cast<int>(p.y),
                                               model[i].vis == Vis::Visible ? "the lit tower" : "the culled overlay over the occluder"));
            }
        }

        mgl::ProgramId program_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};
        mgl::UniformLocation u_mvp_{}, u_color_{}, u_light_{}, u_lit_{};
        std::array<mgl::QueryId, kTowerCount> any_{};
        std::array<mgl::QueryId, kTowerCount> conservative_{};
        bool animate_ = true;
        bool occluders_on_ = true;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-occlusion-queries",
        .title = "Occlusion queries: culling hidden towers behind a city block",
        .min_es = {3, 0},
        .summary = "Bounding boxes of six towers are depth-tested against two front buildings inside ANY_SAMPLES_PASSED "
                   "queries; only towers whose query passed are drawn, culled ones are shown as red x-ray boxes. The "
                   "results are compared with a CPU visibility model, together with CONSERVATIVE queries, result "
                   "availability, CURRENT_QUERY, glIsQuery and the INVALID_OPERATION cases.",
        .controls = "space: pause / resume the sliding occluders   o: toggle the occluders",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<OcclusionQueries>(argc, argv, info);
}
