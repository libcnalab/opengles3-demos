// es20-culling-front-face
//
// Which side of a triangle is the "front"? OpenGL ES decides from the triangle's signed area in
// WINDOW coordinates (ES 3.0 section 3.6.1, equation 3.6) and the glFrontFace setting; glCullFace
// and GL_CULL_FACE then decide which polygons are rasterized at all; the fragment shader sees the
// result in gl_FrontFacing. This demo draws one small test scene in every combination and checks
// each cell's pixels against a CPU model of exactly those rules.
//
// Image (256 x 256 default framebuffer): a grid of 8 columns x 7 rows of cells; every cell has its
// own glViewport. A front-facing fragment is green, a back-facing fragment orange.
//
//   columns (state)                              rows (primitive)
//     0  culling off, FrontFace CCW                 0  triangle, vertices listed counter-clockwise
//     1  culling off, FrontFace CW                  1  triangle, vertices listed clockwise
//     2  CCW, CullFace BACK                         2  CCW triangle mirrored in x by the vertex shader
//     3  CCW, CullFace FRONT                        3  CW  triangle mirrored in x by the vertex shader
//     4  CCW, CullFace FRONT_AND_BACK               4  CCW triangle rotated by 180 degrees (winding kept)
//     5  CW,  CullFace BACK                         5  GL_POINTS  (never culled, always "front")
//     6  CW,  CullFace FRONT                        6  GL_LINES   (never culled, always "front")
//     7  CW,  CullFace FRONT_AND_BACK
//
// Columns 0 and 1 set CullFace(FRONT_AND_BACK) but leave GL_CULL_FACE disabled: the mode has no
// effect until culling is enabled.
//
// Validation: for every cell the CPU computes the window-space signed area of the transformed
// triangle, applies the front/back and culling rules of section 3.6.1, and predicts "no fragment",
// "front color" or "back color"; the pixels must agree. The state is also read back with glGet*,
// and a negative viewport size is shown to be an error (so mirroring must be done in the shader).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    using Rgba8 = demo::Rgba8;

    constexpr Rgba8 kFront = {46, 184, 92, 255};   // gl_FrontFacing == true
    constexpr Rgba8 kBack = {235, 92, 46, 255};    // gl_FrontFacing == false
    constexpr Rgba8 kBackdropA = {22, 25, 33, 255};  // cell background, checker parity 0
    constexpr Rgba8 kBackdropB = {42, 47, 62, 255};  // cell background, checker parity 1

    constexpr int kColumns = 8;
    constexpr int kRows = 7;

    struct Vec2
    {
        float x = 0, y = 0;
    };

    // The test triangle in normalized device coordinates, vertices listed counter-clockwise.
    // It is deliberately asymmetric so that a mirror image is recognizable in the picture.
    constexpr std::array<Vec2, 3> kTriangleCcw = {{{-0.7f, -0.6f}, {0.6f, -0.6f}, {-0.2f, 0.7f}}};

    struct State  // one column of the grid
    {
        const char* label;
        bool cull_enabled;
        mgl::FrontFace front_face;
        mgl::CullFace mode;
    };

    constexpr std::array<State, kColumns> kStates = {{
        {"culling off, FrontFace CCW (CullFace ignored)", false, mgl::FrontFace::CCW, mgl::CullFace::FrontAndBack},
        {"culling off, FrontFace CW (CullFace ignored)", false, mgl::FrontFace::CW, mgl::CullFace::FrontAndBack},
        {"FrontFace CCW, CullFace BACK", true, mgl::FrontFace::CCW, mgl::CullFace::Back},
        {"FrontFace CCW, CullFace FRONT", true, mgl::FrontFace::CCW, mgl::CullFace::Front},
        {"FrontFace CCW, CullFace FRONT_AND_BACK", true, mgl::FrontFace::CCW, mgl::CullFace::FrontAndBack},
        {"FrontFace CW, CullFace BACK", true, mgl::FrontFace::CW, mgl::CullFace::Back},
        {"FrontFace CW, CullFace FRONT", true, mgl::FrontFace::CW, mgl::CullFace::Front},
        {"FrontFace CW, CullFace FRONT_AND_BACK", true, mgl::FrontFace::CW, mgl::CullFace::FrontAndBack},
    }};

    enum class Kind
    {
        Triangle,
        Point,
        Line
    };

    struct Row  // one row of the grid
    {
        const char* label;
        Kind kind;
        bool reversed;  // triangle vertices submitted in clockwise order (A, C, B)
        Vec2 flip;      // u_flip of the vertex shader
    };

    constexpr std::array<Row, kRows> kRowDefs = {{
        {"CCW triangle", Kind::Triangle, false, {1.0f, 1.0f}},
        {"CW triangle", Kind::Triangle, true, {1.0f, 1.0f}},
        {"CCW triangle mirrored in x", Kind::Triangle, false, {-1.0f, 1.0f}},
        {"CW triangle mirrored in x", Kind::Triangle, true, {-1.0f, 1.0f}},
        {"CCW triangle rotated 180 degrees", Kind::Triangle, false, {-1.0f, -1.0f}},
        {"point", Kind::Point, false, {1.0f, 1.0f}},
        {"line", Kind::Line, false, {1.0f, 1.0f}},
    }};

    // Vertex buffer layout (vec2 each): [0..2] CCW triangle, [3..5] same triangle in CW order,
    // [6..7] horizontal line, [8] point.
    constexpr GLint kCcwFirst = 0, kCwFirst = 3, kLineFirst = 6, kPointFirst = 8;

    struct Rect
    {
        int x = 0, y = 0, w = 0, h = 0;
    };

    struct Pixel
    {
        int x = 0, y = 0;
    };

    /// Everything the CPU predicts for one cell.
    struct Plan
    {
        Rect rect;
        std::size_t column = 0, row = 0;
        Rgba8 backdrop{};
        bool drawn = false;
        bool front = false;
        Rgba8 expected{};              // color expected at the probes
        std::vector<Pixel> probes;     // pixels that must show `expected`
        float signed_area = 0.0f;      // window-space area (triangles only)
    };

    // ---------------------------------------------------------------------------------------------
    // CPU reference: ES 3.0 section 3.6.1
    // ---------------------------------------------------------------------------------------------
    Vec2 to_window(Vec2 ndc, Vec2 flip, const Rect& viewport)
    {
        // Viewport transformation (section 2.13.1): x_w = (x_nd + 1) * w / 2 + x, same for y.
        return {(ndc.x * flip.x + 1.0f) * 0.5f * static_cast<float>(viewport.w) + static_cast<float>(viewport.x),
                (ndc.y * flip.y + 1.0f) * 0.5f * static_cast<float>(viewport.h) + static_cast<float>(viewport.y)};
    }

    /// Equation 3.6: a = 1/2 * sum over i of (x_i * y_{i+1} - x_{i+1} * y_i), window coordinates.
    float signed_area(const std::array<Vec2, 3>& v)
    {
        float a = 0.0f;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const Vec2& p = v[i];
            const Vec2& q = v[(i + 1) % 3];
            a += p.x * q.y - q.x * p.y;
        }
        return 0.5f * a;
    }

    /// "If the sign of a (including the possible reversal ... by FrontFace) is positive, the polygon is
    /// front-facing; otherwise, it is back-facing." CCW uses a as computed, CW reverses its sign.
    bool is_front_facing(float area, mgl::FrontFace dir)
    {
        return (dir == mgl::FrontFace::CCW ? area : -area) > 0.0f;
    }

    /// "Front-facing polygons are rasterized if either culling is disabled or the CullFace mode is BACK
    /// while back-facing polygons are rasterized only if either culling is disabled or the CullFace mode
    /// is FRONT." FRONT_AND_BACK therefore removes every polygon.
    bool polygon_is_rasterized(bool front, bool cull_enabled, mgl::CullFace mode)
    {
        if (!cull_enabled) return true;
        return front ? mode == mgl::CullFace::Back : mode == mgl::CullFace::Front;
    }

    float distance_to_line(Vec2 p, Vec2 a, Vec2 b)
    {
        const float len = std::hypot(b.x - a.x, b.y - a.y);
        return std::abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / len;
    }

    GLint as_int(mgl::CullFace m) { return static_cast<GLint>(static_cast<GLenum>(m)); }
    GLint as_int(mgl::FrontFace f) { return static_cast<GLint>(static_cast<GLenum>(f)); }

    class CullingFrontFace final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 192 || env.height < 192) throw demo::Skip("surface smaller than 192 x 192");
            width_ = env.width;
            height_ = env.height;

            // Initial state (ES 3.0 section 3.6.1: culling disabled, CullFace BACK, FrontFace CCW).
            initial_.cull_enabled = mgl::glIsEnabled(mgl::Capability::CullFace);
            initial_.mode = demo::get_integer(mgl::GetParameter::CullFaceMode);
            initial_.front_face = demo::get_integer(mgl::GetParameter::FrontFace);

            // Points are only guaranteed to be one pixel wide (section 3.4); use a visible size if the
            // implementation supports it. The point is placed at a pixel center, so any size works.
            std::array<GLfloat, 2> point_range{};
            mgl::glGetFloatv(mgl::GetParameter::AliasedPointSizeRange, point_range.data());
            point_range_ = point_range;
            point_size_ = std::clamp(8.0f, point_range[0], point_range[1]);

            program_ = demo::build_program(env, "shaders/triangle.vert", "shaders/facing.frag");
            u_flip_ = mgl::glGetUniformLocation(program_, "u_flip");
            u_point_size_ = mgl::glGetUniformLocation(program_, "u_point_size");
            u_front_ = mgl::glGetUniformLocation(program_, "u_front_color");
            u_back_ = mgl::glGetUniformLocation(program_, "u_back_color");
            if (u_flip_.value < 0 || u_point_size_.value < 0 || u_front_.value < 0 || u_back_.value < 0)
                throw demo::Failure("triangle.vert / facing.frag: a uniform is not active");

            cell_w_ = width_ / kColumns;
            cell_h_ = height_ / kRows;
            create_geometry();
            make_plans();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            mgl::glUniform4f(u_front_, to_f(kFront[0]), to_f(kFront[1]), to_f(kFront[2]), to_f(kFront[3]));
            mgl::glUniform4f(u_back_, to_f(kBack[0]), to_f(kBack[1]), to_f(kBack[2]), to_f(kBack[3]));
            mgl::glUniform1f(u_point_size_, point_size_);

            // Clear the whole surface once (this also covers the few rows below the grid that the
            // integer cell size leaves over); each cell then repaints its own background.
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glClearColor(to_f(kBackdropA[0]), to_f(kBackdropA[1]), to_f(kBackdropA[2]), 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            for (const Plan& plan : plans_)
            {
                const State& state = kStates[plan.column];
                const Row& row = kRowDefs[plan.row];

                // Cell background: a scissored clear keeps the clear inside the cell.
                mgl::glEnable(mgl::Capability::ScissorTest);
                mgl::glScissor(plan.rect.x, plan.rect.y, plan.rect.w, plan.rect.h);
                mgl::glClearColor(to_f(plan.backdrop[0]), to_f(plan.backdrop[1]), to_f(plan.backdrop[2]), 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
                mgl::glDisable(mgl::Capability::ScissorTest);

                // The viewport maps NDC [-1, 1] to the cell. Nothing is clipped to it: a primitive
                // inside NDC [-1, 1] simply lands inside the cell.
                mgl::glViewport(plan.rect.x, plan.rect.y, plan.rect.w, plan.rect.h);

                // Face-culling state of this column. glCullFace and glFrontFace are plain state; only
                // glEnable(CULL_FACE) makes the mode matter.
                if (state.cull_enabled)
                    mgl::glEnable(mgl::Capability::CullFace);
                else
                    mgl::glDisable(mgl::Capability::CullFace);
                mgl::glCullFace(state.mode);
                mgl::glFrontFace(state.front_face);

                mgl::glUniform2f(u_flip_, row.flip.x, row.flip.y);
                switch (row.kind)
                {
                case Kind::Triangle:
                    mgl::glDrawArrays(mgl::PrimitiveType::Triangles, row.reversed ? kCwFirst : kCcwFirst, 3);
                    break;
                case Kind::Line:
                    mgl::glDrawArrays(mgl::PrimitiveType::Lines, kLineFirst, 2);
                    break;
                case Kind::Point:
                    mgl::glDrawArrays(mgl::PrimitiveType::Points, kPointFirst, 1);
                    break;
                }
            }

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_model(check);
            check_cells(check);
            check_state(check);
            check_negative_viewport(env, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteProgram(program_);
        }

    private:
        static float to_f(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

        // -----------------------------------------------------------------------------------------
        // Geometry and CPU plan
        // -----------------------------------------------------------------------------------------
        void create_geometry()
        {
            // Points and lines are placed on pixel centers of the cell so that any rasterization
            // detail (point size 1, line width 1) covers exactly the probed pixels:
            //   NDC = 2 * (pixel + 0.5) / size - 1.
            const float px = 2.0f * (static_cast<float>(cell_w_ / 2) + 0.5f) / static_cast<float>(cell_w_) - 1.0f;
            const float py = 2.0f * (static_cast<float>(cell_h_ / 2) + 0.5f) / static_cast<float>(cell_h_) - 1.0f;
            const Vec2 a = kTriangleCcw[0], b = kTriangleCcw[1], c = kTriangleCcw[2];
            const std::array<Vec2, 9> vertices = {{a, b, c,              // counter-clockwise
                                                   a, c, b,              // clockwise: two vertices swapped
                                                   {-0.6f, py}, {0.6f, py},  // line
                                                   {px, py}}};            // point

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vec2>(vertices), mgl::BufferUsage::StaticDraw);
            const mgl::AttribLocation a_position{0u};  // layout(location = 0) in the vertex shader
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vec2), demo::buffer_offset(0));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void make_plans()
        {
            for (std::size_t row = 0; row < kRows; ++row)
                for (std::size_t column = 0; column < kColumns; ++column)
                {
                    Plan plan;
                    plan.column = column;
                    plan.row = row;
                    // Window coordinates have their origin bottom-left; row 0 is drawn at the top.
                    plan.rect = {static_cast<int>(column) * cell_w_, height_ - (static_cast<int>(row) + 1) * cell_h_, cell_w_, cell_h_};
                    plan.backdrop = (row + column) % 2 == 0 ? kBackdropA : kBackdropB;

                    const State& state = kStates[column];
                    const Row& def = kRowDefs[row];
                    if (def.kind == Kind::Triangle)
                    {
                        std::array<Vec2, 3> ndc = kTriangleCcw;
                        if (def.reversed) std::swap(ndc[1], ndc[2]);
                        std::array<Vec2, 3> win{};
                        for (std::size_t i = 0; i < 3; ++i) win[i] = to_window(ndc[i], def.flip, plan.rect);
                        plan.signed_area = signed_area(win);
                        plan.front = is_front_facing(plan.signed_area, state.front_face);
                        plan.drawn = polygon_is_rasterized(plan.front, state.cull_enabled, state.mode);

                        // Probes: the centroid and three points close to the vertices, all well inside.
                        const std::array<std::array<float, 3>, 4> bary = {{{1.0f / 3, 1.0f / 3, 1.0f / 3},
                                                                           {0.6f, 0.2f, 0.2f},
                                                                           {0.2f, 0.6f, 0.2f},
                                                                           {0.2f, 0.2f, 0.6f}}};
                        for (const auto& w : bary)
                        {
                            const Vec2 p = {w[0] * win[0].x + w[1] * win[1].x + w[2] * win[2].x,
                                            w[0] * win[0].y + w[1] * win[1].y + w[2] * win[2].y};
                            const Pixel pixel = {static_cast<int>(std::floor(p.x)), static_cast<int>(std::floor(p.y))};
                            const Vec2 center = {static_cast<float>(pixel.x) + 0.5f, static_cast<float>(pixel.y) + 0.5f};
                            const float margin = std::min({distance_to_line(center, win[0], win[1]),
                                                           distance_to_line(center, win[1], win[2]),
                                                           distance_to_line(center, win[2], win[0])});
                            if (margin < 2.0f) throw demo::Failure("triangle probe closer than 2 px to an edge");
                            plan.probes.push_back(pixel);
                        }
                    }
                    else
                    {
                        // Points and lines are not polygons: culling never applies, and "all other
                        // fragments are considered front-facing" (section 3.9.2.2).
                        plan.front = true;
                        plan.drawn = true;
                        const int cy = plan.rect.y + cell_h_ / 2;
                        if (def.kind == Kind::Point)
                        {
                            plan.probes.push_back({plan.rect.x + cell_w_ / 2, cy});
                        }
                        else
                        {
                            // Line from NDC x -0.6 to 0.6 (about 25 % .. 80 % of the cell width), width 1.
                            for (const int dx : {-cell_w_ / 4, 0, cell_w_ / 4})
                                plan.probes.push_back({plan.rect.x + cell_w_ / 2 + dx, cy});
                        }
                    }
                    plan.expected = plan.drawn ? (plan.front ? kFront : kBack) : plan.backdrop;
                    plans_.push_back(std::move(plan));
                }
        }

        // -----------------------------------------------------------------------------------------
        // Validation
        // -----------------------------------------------------------------------------------------
        static std::string describe(const Plan& plan)
        {
            const Row& row = kRowDefs[plan.row];
            const char* outcome = !plan.drawn ? "culled, backdrop remains" : plan.front ? "front-facing, green" : "back-facing, orange";
            if (row.kind == Kind::Triangle)
                return std::format("{} | {}: area {:+.1f} -> {}", row.label, kStates[plan.column].label, plan.signed_area, outcome);
            return std::format("{} | {}: {}", row.label, kStates[plan.column].label, outcome);
        }

        void check_model(demo::Check& check) const
        {
            // The facing rules rest on the window-space signs of the four triangle variants; if the
            // test data were wrong every expectation below would be wrong, so state them explicitly.
            auto area_of = [&](std::size_t row) {
                for (const Plan& p : plans_)
                    if (p.row == row && p.column == 0) return p.signed_area;
                return 0.0f;
            };
            const float ccw = area_of(0), cw = area_of(1), mirrored = area_of(2), mirrored_cw = area_of(3), rotated = area_of(4);
            check.expect(ccw > 0.0f && cw < 0.0f && mirrored < 0.0f && mirrored_cw > 0.0f && rotated > 0.0f,
                         "model: window-space signed area is + for CCW, - for CW, flipped by an x-mirror, unchanged by a 180-degree rotation",
                         std::format("areas: ccw {:+.1f}, cw {:+.1f}, mirrored ccw {:+.1f}, mirrored cw {:+.1f}, rotated ccw {:+.1f}",
                                     ccw, cw, mirrored, mirrored_cw, rotated));
        }

        void check_cells(demo::Check& check)
        {
            int drawn_front = 0, drawn_back = 0, culled = 0;
            for (const Plan& plan : plans_)
            {
                // The expected value is the front color, the back color, or the untouched backdrop.
                bool ok = true;
                std::string detail;
                for (const Pixel& p : plan.probes)
                {
                    const Rgba8 got = demo::read_pixel(p.x, p.y);
                    for (std::size_t ch = 0; ch < 4; ++ch)
                        if (std::abs(int{got[ch]} - int{plan.expected[ch]}) > 1 && ok)
                        {
                            ok = false;
                            detail = std::format("pixel({},{}) = {} expected {}", p.x, p.y, demo::format_rgba(got),
                                                 demo::format_rgba(plan.expected));
                        }
                }
                // A corner of the cell lies outside every primitive: it must still be the backdrop.
                const Rgba8 corner = demo::read_pixel(plan.rect.x, plan.rect.y);
                if (corner != plan.backdrop && ok)
                {
                    ok = false;
                    detail = std::format("cell corner = {} expected backdrop {}", demo::format_rgba(corner),
                                         demo::format_rgba(plan.backdrop));
                }
                check.expect(ok, describe(plan), detail);
                if (!plan.drawn) ++culled;
                else if (plan.front) ++drawn_front;
                else ++drawn_back;
            }
            // The grid must exercise all three outcomes, otherwise the cell checks would not
            // discriminate between rules (for example a driver that ignores glFrontFace).
            check.expect(drawn_front > 0 && drawn_back > 0 && culled > 0,
                         "the grid shows front-facing, back-facing and culled cells",
                         std::format("{} front, {} back, {} culled cells", drawn_front, drawn_back, culled));
        }

        void check_state(demo::Check& check)
        {
            check.expect(!initial_.cull_enabled, "initially GL_CULL_FACE is disabled (3.6.1)");
            check.expect_eq(initial_.mode, as_int(mgl::CullFace::Back), "initial GL_CULL_FACE_MODE = BACK");
            check.expect_eq(initial_.front_face, as_int(mgl::FrontFace::CCW), "initial GL_FRONT_FACE = CCW");
            check.expect(point_range_[1] >= 1.0f, "ALIASED_POINT_SIZE_RANGE maximum is at least 1 (3.4)",
                         std::format("range = [{}, {}], used point size {}", point_range_[0], point_range_[1], point_size_));

            // render() ended with culling disabled; toggle the flag and query it.
            check.expect(!mgl::glIsEnabled(mgl::Capability::CullFace), "glIsEnabled(CULL_FACE) is FALSE after glDisable");
            mgl::glEnable(mgl::Capability::CullFace);
            check.expect(mgl::glIsEnabled(mgl::Capability::CullFace), "glIsEnabled(CULL_FACE) is TRUE after glEnable");
            mgl::glDisable(mgl::Capability::CullFace);
            check.expect(!mgl::glIsEnabled(mgl::Capability::CullFace), "glIsEnabled(CULL_FACE) is FALSE after glDisable again");

            for (const mgl::CullFace mode : {mgl::CullFace::Front, mgl::CullFace::FrontAndBack, mgl::CullFace::Back})
            {
                mgl::glCullFace(mode);
                check.expect_eq(demo::get_integer(mgl::GetParameter::CullFaceMode), as_int(mode),
                                std::format("glCullFace({}) -> GL_CULL_FACE_MODE",
                                            mode == mgl::CullFace::Front ? "FRONT" : mode == mgl::CullFace::Back ? "BACK" : "FRONT_AND_BACK"));
            }
            for (const mgl::FrontFace dir : {mgl::FrontFace::CW, mgl::FrontFace::CCW})
            {
                mgl::glFrontFace(dir);
                check.expect_eq(demo::get_integer(mgl::GetParameter::FrontFace), as_int(dir),
                                std::format("glFrontFace({}) -> GL_FRONT_FACE", dir == mgl::FrontFace::CW ? "CW" : "CCW"));
            }
            check.expect_no_gl_error("valid culling commands raised no GL error");
        }

        void check_negative_viewport(demo::Env& env, demo::Check& check)
        {
            // "INVALID_VALUE is generated if either w or h is negative" (section 2.13.1): a negative
            // viewport size cannot be used to mirror the image, and the viewport stays unchanged.
            std::array<GLint, 4> before{};
            mgl::glGetIntegerv(mgl::GetParameter::Viewport, before.data());
            mgl::glViewport(0, 0, -env.width, env.height);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glViewport with a negative width -> INVALID_VALUE");
            mgl::glViewport(0, 0, env.width, -env.height);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glViewport with a negative height -> INVALID_VALUE");
            std::array<GLint, 4> after{};
            mgl::glGetIntegerv(mgl::GetParameter::Viewport, after.data());
            check.expect(before == after, "a rejected glViewport leaves the viewport unchanged",
                         std::format("before ({}, {}, {}, {}), after ({}, {}, {}, {})", before[0], before[1], before[2],
                                     before[3], after[0], after[1], after[2], after[3]));
        }

        struct InitialState
        {
            bool cull_enabled = true;
            GLint mode = -1, front_face = -1;
        };

        int width_ = 0, height_ = 0, cell_w_ = 0, cell_h_ = 0;
        float point_size_ = 1.0f;
        std::array<GLfloat, 2> point_range_{};
        InitialState initial_;
        std::vector<Plan> plans_;
        mgl::ProgramId program_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};
        mgl::UniformLocation u_flip_{}, u_point_size_{}, u_front_{}, u_back_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-culling-front-face",
        .title = "Face culling and front-face winding",
        .min_es = {3, 0},
        .summary = "Front/back determination from window-space winding, glFrontFace, glCullFace (incl. FRONT_AND_BACK "
                   "and points/lines), gl_FrontFacing, and winding flips by mirroring - every combination checked "
                   "against a CPU model of ES 3.0 3.6.1.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<CullingFrontFace>(argc, argv, info);
}
