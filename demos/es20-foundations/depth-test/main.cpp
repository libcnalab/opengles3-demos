// es20-depth-test
//
// The depth buffer test and everything that feeds or masks it: all eight depth comparison
// functions, the depth write mask (and its effect on glClear), glClearDepthf, glDepthRangef,
// glPolygonOffset, and a fragment shader that overrides the depth with gl_FragDepth.
//
// The window is a 4x4 grid of independent experiments ("cells"), read left to right, top to
// bottom:
//
//     0 NEVER      1 LESS       2 EQUAL      3 LEQUAL        glDepthFunc truth table
//     4 GREATER    5 NOTEQUAL   6 GEQUAL     7 ALWAYS
//     8 test off   9 depth mask 10 masked    11 clear        enable bit / glDepthMask /
//                                clear          values      glClear / glClearDepthf
//     12 depth     13 gl_Frag-  14 offset     15 offset      glDepthRangef / gl_FragDepth /
//        range        Depth        (flat)       (slope)     glPolygonOffset
//
// ES cannot read the depth buffer back (glReadPixels does not accept DEPTH_COMPONENT), so every
// experiment is built so that the depth values *decide which color survives*. The expected color
// of each probed pixel is derived on the CPU from the rules in the specification and compared
// with what the GPU produced. Depth values are chosen far apart (0.25 / 0.5 / 0.75) wherever
// the specification does not promise bit exactness; the EQUAL tests compare a depth with itself
// (the NDC depth is exactly 0.0 at every vertex of both quads, and a plane with zero gradient and
// zero offset yields the same window depth however the rasterizer evaluates it).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    using Color = std::array<float, 4>;

    // All components are multiples of 0.2, so they convert to exact 8-bit values (51, 102, ...).
    constexpr Color kBackground = {0.2f, 0.2f, 0.2f, 1.0f};  // cell background (glClear)
    constexpr Color kBase = {0.4f, 0.4f, 0.8f, 1.0f};        // blue   : the quad that is drawn first
    constexpr Color kNear = {0.2f, 0.8f, 0.4f, 1.0f};        // green  : stripe nearer than the base
    constexpr Color kEqual = {1.0f, 0.8f, 0.2f, 1.0f};       // yellow : stripe at the base's depth
    constexpr Color kFar = {0.8f, 0.2f, 0.8f, 1.0f};         // magenta: stripe farther than the base
    constexpr Color kOrange = {1.0f, 0.6f, 0.0f, 1.0f};      // orange : the "probe" quad of two-phase cells

    demo::Rgba8 to_rgba8(const Color& c)
    {
        auto q = [](float v) { return static_cast<std::uint8_t>(std::lround(v * 255.0f)); };
        return {q(c[0]), q(c[1]), q(c[2]), q(c[3])};
    }

    // ---- CPU model of the depth pipeline --------------------------------------------------------

    /// ES 3.0 §2.13.1: window-space depth of a vertex with NDC depth z_d under DepthRangef(n, f).
    constexpr double window_z(double z_ndc, double n = 0.0, double f = 1.0)
    {
        return (f - n) / 2.0 * z_ndc + (n + f) / 2.0;
    }

    /// ES 3.0 §4.1.5: the test passes if the *incoming* depth <func> the *stored* depth.
    constexpr bool depth_test_passes(mgl::CompareFunc func, double incoming, double stored)
    {
        switch (func)
        {
            case mgl::CompareFunc::Never: return false;
            case mgl::CompareFunc::Less: return incoming < stored;
            case mgl::CompareFunc::Equal: return incoming == stored;
            case mgl::CompareFunc::Lequal: return incoming <= stored;
            case mgl::CompareFunc::Greater: return incoming > stored;
            case mgl::CompareFunc::Notequal: return incoming != stored;
            case mgl::CompareFunc::Gequal: return incoming >= stored;
            case mgl::CompareFunc::Always: return true;
        }
        return false;
    }

    struct FuncName
    {
        mgl::CompareFunc func;
        const char* name;
    };
    constexpr std::array<FuncName, 8> kFuncs = {{
        {mgl::CompareFunc::Never, "NEVER"},
        {mgl::CompareFunc::Less, "LESS"},
        {mgl::CompareFunc::Equal, "EQUAL"},
        {mgl::CompareFunc::Lequal, "LEQUAL"},
        {mgl::CompareFunc::Greater, "GREATER"},
        {mgl::CompareFunc::Notequal, "NOTEQUAL"},
        {mgl::CompareFunc::Gequal, "GEQUAL"},
        {mgl::CompareFunc::Always, "ALWAYS"},
    }};

    // NDC depth of the quad that is drawn first in most cells. 0.0 maps to window depth 0.5 under
    // the default depth range; both operands of an EQUAL test are computed from exactly this value.
    constexpr double kStoredNdc = 0.0;
    constexpr double kStored = window_z(kStoredNdc);  // 0.5

    /// One grid cell in window coordinates (origin lower left, like glViewport).
    struct Cell
    {
        int x, y, w, h;
    };

    struct Program
    {
        mgl::ProgramId id{};
        mgl::UniformLocation rect{}, depth{}, color{}, depth_line{};
    };

    /// A pixel that validate() compares with the color the CPU model predicts.
    struct Probe
    {
        int x, y;
        demo::Rgba8 expected;
        std::string what;
    };

    /// State read in init(), before the demo touches anything: must equal the specified defaults.
    struct InitialState
    {
        bool depth_test = true;
        bool offset_fill = true;
        int depth_func = 0;
        GLboolean depth_mask = GL_FALSE;
        std::array<GLfloat, 2> depth_range{-1.0f, -1.0f};
        GLfloat clear_value = -1.0f;
        GLfloat offset_factor = -1.0f;
        GLfloat offset_units = -1.0f;
        int depth_bits = 0;
    };

    class DepthTest final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            solid_ = make_program(env, "shaders/quad.vert", "shaders/solid.frag", false);
            frag_depth_ = make_program(env, "shaders/quad.vert", "shaders/fragdepth.frag", true);

            // The context starts in the default state; remember it for validate().
            initial_.depth_test = mgl::glIsEnabled(mgl::Capability::DepthTest);
            initial_.offset_fill = mgl::glIsEnabled(mgl::Capability::PolygonOffsetFill);
            initial_.depth_func = demo::get_integer(mgl::GetParameter::DepthFunc);
            mgl::glGetBooleanv(mgl::GetParameter::DepthWritemask, &initial_.depth_mask);
            mgl::glGetFloatv(mgl::GetParameter::DepthRange, initial_.depth_range.data());
            mgl::glGetFloatv(mgl::GetParameter::DepthClearValue, &initial_.clear_value);
            mgl::glGetFloatv(mgl::GetParameter::PolygonOffsetFactor, &initial_.offset_factor);
            mgl::glGetFloatv(mgl::GetParameter::PolygonOffsetUnits, &initial_.offset_units);
            initial_.depth_bits = demo::get_integer(mgl::GetParameter::DepthBits);
            depth_bits_ = initial_.depth_bits;
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            probes_.clear();

            // Start from a known state and a black frame; the grid lines are this black showing
            // between the cells.
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            for (std::size_t i = 0; i < kFuncs.size(); ++i)
                cell_depth_func(cell(env, static_cast<int>(i)), kFuncs[i]);
            cell_test_disabled(cell(env, 8));
            cell_depth_mask(cell(env, 9));
            cell_masked_clear(cell(env, 10));
            cell_clear_values(cell(env, 11));
            cell_depth_range(cell(env, 12));
            cell_frag_depth(cell(env, 13));
            cell_offset_flat(cell(env, 14));
            cell_offset_slope(cell(env, 15));

            // Leave the context in its default state; nothing in the next frame may depend on it.
            restore_defaults();
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            for (const Probe& p : probes_)
                check.expect_pixel(p.x, p.y, p.expected, 1, p.what);
            check_frag_depth_vs_range(check);
            check_initial_state(check);
            check_state_round_trips(check);
            restore_defaults();
            check.expect_no_gl_error("no GL error raised by the depth experiments");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            restore_defaults();
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(solid_.id);
            mgl::glDeleteProgram(frag_depth_.id);
        }

    private:
        // ---- program / drawing helpers ---------------------------------------------------------

        static Program make_program(const demo::Env& env, const char* vs, const char* fs, bool with_depth_line)
        {
            Program p;
            p.id = demo::build_program(env, vs, fs);
            p.rect = mgl::glGetUniformLocation(p.id, "u_rect");
            p.depth = mgl::glGetUniformLocation(p.id, "u_depth");
            p.color = mgl::glGetUniformLocation(p.id, "u_color");
            if (with_depth_line) p.depth_line = mgl::glGetUniformLocation(p.id, "u_depth_line");
            if (p.rect.value < 0 || p.depth.value < 0 || p.color.value < 0 || (with_depth_line && p.depth_line.value < 0))
                throw demo::Failure(std::string("missing uniform in ") + fs);
            return p;
        }

        void use(const Program& p)
        {
            mgl::glUseProgram(p.id);
            current_ = &p;
        }

        /// Draws the rectangle [x0,x1] x [y0,y1] (NDC of the current viewport) as a plane whose NDC
        /// depth is z + slope * x_ndc.
        void draw_quad(double x0, double y0, double x1, double y1, double z, double slope, const Color& c)
        {
            mgl::glUniform4f(current_->rect, static_cast<float>(x0), static_cast<float>(y0), static_cast<float>(x1),
                             static_cast<float>(y1));
            mgl::glUniform2f(current_->depth, static_cast<float>(z), static_cast<float>(slope));
            mgl::glUniform4f(current_->color, c[0], c[1], c[2], c[3]);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        static Cell cell(const demo::Env& env, int index)
        {
            const int cw = env.width / 4, ch = env.height / 4;
            const int col = index % 4, row = index / 4;  // row 0 is the top row of the image
            return {col * cw, (3 - row) * ch, cw, ch};
        }

        /// Prepares one cell: viewport = the cell, cleared color and depth, all state defaulted.
        /// The cell is cleared one pixel short of its border so a black grid line stays visible.
        void begin_cell(const Cell& c)
        {
            restore_defaults();
            use(solid_);
            mgl::glViewport(c.x, c.y, c.w, c.h);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(c.x + 1, c.y + 1, c.w - 2, c.h - 2);
            // glClear obeys the depth write mask (ES 3.0 §4.2.3 -> §4.2.2): it must be TRUE here.
            mgl::glDepthMask(GL_TRUE);
            mgl::glClearColor(kBackground[0], kBackground[1], kBackground[2], kBackground[3]);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        static void restore_defaults()
        {
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::PolygonOffsetFill);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_TRUE);
            mgl::glDepthRangef(0.0f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glPolygonOffset(0.0f, 0.0f);
        }

        /// Window pixel that covers NDC position (nx, ny) of this cell.
        static std::pair<int, int> pixel_of(const Cell& c, double nx, double ny)
        {
            return {c.x + static_cast<int>(std::floor((nx + 1.0) * 0.5 * c.w)),
                    c.y + static_cast<int>(std::floor((ny + 1.0) * 0.5 * c.h))};
        }

        /// Records "the pixel at NDC (nx, ny) of this cell must show `color`".
        void expect_color(const Cell& c, double nx, double ny, const Color& color, std::string what)
        {
            const auto [px, py] = pixel_of(c, nx, ny);
            probes_.push_back({px, py, to_rgba8(color), std::move(what)});
        }

        static const Color& winner(bool passes, const Color& drawn) { return passes ? drawn : kBase; }

        // ---- cells 0-7: glDepthFunc truth table ------------------------------------------------

        // The blue base quad is drawn with ALWAYS and stores window depth 0.5. Three stripes are
        // then drawn with the function under test, one nearer (0.25), one at the same depth
        // (0.5) and one farther (0.75). A stripe replaces the blue only if its test passes.
        void cell_depth_func(const Cell& c, const FuncName& fn)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);

            mgl::glDepthFunc(fn.func);
            struct Stripe
            {
                double x0, x1, z_ndc;
                const Color* color;
                const char* relation;
            };
            const Stripe stripes[] = {
                {-0.9, -0.3, -0.5, &kNear, "nearer (0.25 vs 0.5)"},
                {-0.3, 0.3, kStoredNdc, &kEqual, "equal (0.5 vs 0.5)"},
                {0.3, 0.9, 0.5, &kFar, "farther (0.75 vs 0.5)"},
            };
            for (const Stripe& s : stripes) draw_quad(s.x0, -0.5, s.x1, 0.5, s.z_ndc, 0.0, *s.color);

            for (const Stripe& s : stripes)
            {
                const bool passes = depth_test_passes(fn.func, window_z(s.z_ndc), kStored);
                expect_color(c, (s.x0 + s.x1) / 2, 0.0, winner(passes, *s.color),
                             std::format("DepthFunc({}): fragment {} {}", fn.name, s.relation,
                                         passes ? "passes and replaces the base" : "fails and the base stays"));
            }
        }

        // ---- cell 8: disabling the test bypasses the depth write as well -----------------------

        void cell_test_disabled(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);  // stores 0.5

            // §4.1.5: with the test disabled "the depth comparison and subsequent possible updates
            // to the depth buffer value are bypassed" - a farther fragment still lands, and the
            // stored depth stays 0.5 even though the nearer stripe (0.25) would have changed it.
            mgl::glDisable(mgl::Capability::DepthTest);
            draw_quad(-0.9, -0.5, -0.3, 0.5, 0.5, 0.0, kFar);   // 0.75: would fail LESS
            draw_quad(0.3, -0.5, 0.9, 0.5, -0.5, 0.0, kNear);   // 0.25: would pass LESS and be stored

            // Phase 2: enable LESS and draw a fragment at 0.4 over the upper half of the right
            // stripe. It passes only if the nearer stripe did NOT update the depth buffer.
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            draw_quad(0.3, 0.0, 0.9, 0.5, -0.2, 0.0, kOrange);

            const double stored_after_disabled_draws = kStored;  // no updates while disabled
            const bool probe_passes = depth_test_passes(mgl::CompareFunc::Less, window_z(-0.2), stored_after_disabled_draws);
            expect_color(c, -0.6, 0.0, kFar, "depth test disabled: a fragment farther than the stored depth is still drawn");
            expect_color(c, 0.6, -0.25, kNear, "depth test disabled: the nearer stripe is drawn (lower half, untouched by phase 2)");
            expect_color(c, 0.6, 0.25, winner(probe_passes, kOrange),
                         "depth test disabled: nothing was written, so the later LESS fragment (0.4 < 0.5) passes");
        }

        // ---- cell 9: glDepthMask ----------------------------------------------------------------

        void cell_depth_mask(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);  // stores 0.5
            mgl::glDepthFunc(mgl::CompareFunc::Less);

            // Both stripes are nearer (0.25) and pass. Only the right one may update the depth buffer.
            mgl::glDepthMask(GL_FALSE);
            draw_quad(-0.9, -0.6, -0.1, 0.6, -0.5, 0.0, kEqual);  // color written, depth NOT written
            mgl::glDepthMask(GL_TRUE);
            draw_quad(0.1, -0.6, 0.9, 0.6, -0.5, 0.0, kNear);     // color and depth written (0.25)

            // A fragment at 0.4 over the upper half of both stripes: behind a stored 0.25, in front
            // of a stored 0.5.
            draw_quad(-0.9, 0.0, 0.9, 0.6, -0.2, 0.0, kOrange);

            const double stored_left = kStored;               // the masked draw left 0.5
            const double stored_right = window_z(-0.5);       // the unmasked draw stored 0.25
            const bool left_wins = depth_test_passes(mgl::CompareFunc::Less, window_z(-0.2), stored_left);
            const bool right_wins = depth_test_passes(mgl::CompareFunc::Less, window_z(-0.2), stored_right);
            expect_color(c, -0.5, -0.3, kEqual, "DepthMask(FALSE): the stripe's color is still written (lower left)");
            expect_color(c, 0.5, -0.3, kNear, "DepthMask(TRUE): the stripe's color is written (lower right)");
            expect_color(c, -0.5, 0.3, left_wins ? kOrange : kEqual,
                         "DepthMask(FALSE): depth was not stored, so the later 0.4 fragment still passes");
            expect_color(c, 0.5, 0.3, right_wins ? kOrange : kNear,
                         "DepthMask(TRUE): depth 0.25 was stored, so the later 0.4 fragment fails");
        }

        // ---- cell 10: glClear obeys the depth write mask ---------------------------------------

        void cell_masked_clear(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);  // stores 0.5 everywhere

            // Clear the depth buffer to 1.0 in the left half with the write mask off and in the
            // right half with it on. §4.2.3: the masking operations of §4.2.2 are applied to clears.
            const int half = c.w / 2;
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glDepthMask(GL_FALSE);
            mgl::glScissor(c.x, c.y, half, c.h);
            mgl::glClear(mgl::ClearBufferBit::Depth);
            mgl::glDepthMask(GL_TRUE);
            mgl::glScissor(c.x + half, c.y, c.w - half, c.h);
            mgl::glClear(mgl::ClearBufferBit::Depth);
            mgl::glDisable(mgl::Capability::ScissorTest);

            mgl::glDepthFunc(mgl::CompareFunc::Less);
            draw_quad(-0.9, -0.5, 0.9, 0.5, 0.5, 0.0, kNear);  // 0.75

            const double stored_left = kStored;  // the masked clear did nothing
            const double stored_right = 1.0;     // the clear value
            expect_color(c, -0.45, 0.0, winner(depth_test_passes(mgl::CompareFunc::Less, window_z(0.5), stored_left), kNear),
                         "Clear(DEPTH) with DepthMask(FALSE) leaves the stored 0.5: a 0.75 fragment fails");
            expect_color(c, 0.45, 0.0, winner(depth_test_passes(mgl::CompareFunc::Less, window_z(0.5), stored_right), kNear),
                         "Clear(DEPTH) with DepthMask(TRUE) resets to 1.0: a 0.75 fragment passes");
        }

        // ---- cell 11: glClearDepthf and clamping -----------------------------------------------

        void cell_clear_values(const Cell& c)
        {
            begin_cell(c);
            // Four bands cleared to different depths. The first and last arguments are outside
            // [0,1]; §4.2.3: "d is clamped to the range [0,1]".
            constexpr std::array<double, 4> requested = {-3.0, 0.4, 0.6, 5.0};
            const int quarter = c.w / 4;
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (int i = 0; i < 4; ++i)
            {
                mgl::glScissor(c.x + i * quarter, c.y, i == 3 ? c.w - 3 * quarter : quarter, c.h);
                mgl::glClearDepthf(static_cast<float>(requested[static_cast<std::size_t>(i)]));
                mgl::glClear(mgl::ClearBufferBit::Depth);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);

            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            draw_quad(-0.9, -0.7, 0.9, 0.7, kStoredNdc, 0.0, kNear);  // 0.5 against four clear values

            for (int i = 0; i < 4; ++i)
            {
                const double clear = std::clamp(requested[static_cast<std::size_t>(i)], 0.0, 1.0);
                const bool passes = depth_test_passes(mgl::CompareFunc::Less, kStored, clear);
                expect_color(c, -0.75 + 0.5 * i, 0.0, passes ? kNear : kBackground,
                             std::format("ClearDepthf({}) (clamped to {}): a 0.5 fragment {} LESS", requested[static_cast<std::size_t>(i)],
                                         clear, passes ? "passes" : "fails"));
            }
        }

        // ---- cell 12: glDepthRangef -------------------------------------------------------------

        void cell_depth_range(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);  // default range: stores 0.5
            mgl::glDepthFunc(mgl::CompareFunc::Less);

            struct Range
            {
                double n, f, z_ndc;
            };
            // The same vertex depth lands on different window depths when the range changes
            // (§2.13.1); a reversed range (n > f) is legal and flips the direction.
            constexpr std::array<Range, 4> ranges = {{
                {0.0, 1.0, -0.5},  // default range: 0.25
                {0.5, 1.0, -0.5},  // squeezed into the far half: 0.625
                {0.0, 0.5, 0.5},   // squeezed into the near half: 0.375
                {1.0, 0.0, -0.5},  // reversed: 0.75
            }};
            for (std::size_t i = 0; i < ranges.size(); ++i)
            {
                const double x0 = -0.9 + 0.45 * static_cast<double>(i);
                mgl::glDepthRangef(static_cast<float>(ranges[i].n), static_cast<float>(ranges[i].f));
                draw_quad(x0, -0.5, x0 + 0.45, 0.5, ranges[i].z_ndc, 0.0, kNear);
            }
            mgl::glDepthRangef(0.0f, 1.0f);

            for (std::size_t i = 0; i < ranges.size(); ++i)
            {
                const double x0 = -0.9 + 0.45 * static_cast<double>(i);
                const double wz = window_z(ranges[i].z_ndc, ranges[i].n, ranges[i].f);
                const bool passes = depth_test_passes(mgl::CompareFunc::Less, wz, kStored);
                expect_color(c, x0 + 0.225, 0.0, winner(passes, kNear),
                             std::format("DepthRangef({}, {}) with NDC z {}: window z = {} {} the stored 0.5 under LESS",
                                         ranges[i].n, ranges[i].f, ranges[i].z_ndc, wz, passes ? "passes" : "fails against"));
            }
        }

        // ---- cell 13: gl_FragDepth -------------------------------------------------------------

        void cell_frag_depth(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Always);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);  // stores 0.5
            mgl::glDepthFunc(mgl::CompareFunc::Less);

            // Pixel x of NDC x within this cell, for the ramp stripe (gl_FragCoord is in window pixels).
            auto pixel_x = [&](double nx) { return static_cast<double>(c.x) + (nx + 1.0) * 0.5 * c.w; };
            const double ramp_x0 = pixel_x(0.45), ramp_x1 = pixel_x(0.9);

            use(frag_depth_);
            // Stripe 1: the vertices say 0.75 (would fail), the shader says 0.25 (passes).
            mgl::glUniform2f(frag_depth_.depth_line, 0.25f, 0.0f);
            draw_quad(-0.9, -0.5, -0.45, 0.5, 0.5, 0.0, kNear);
            // Stripe 2: the vertices say 0.25 (would pass), the shader says 0.75 (fails).
            mgl::glUniform2f(frag_depth_.depth_line, 0.75f, 0.0f);
            draw_quad(-0.45, -0.5, 0.0, 0.5, -0.5, 0.0, kNear);
            // Stripe 3: gl_FragDepth = 0.25 under glDepthRangef(0.5, 1) and an EQUAL test against the
            // stored 0.5. ES 3.0 §3.9.2.3: the shader's depth is clamped to [0,1] and converted to fixed
            // point, but "the depth range computation is not applied here". The fragment's depth is
            // therefore 0.25, EQUAL fails, and the base stays. (A depth range *mapping* would give 0.625
            // and fail too; clamping the value into [n, f] = [0.5, 1] would give 0.5 and pass - see
            // check_frag_depth_vs_range().)
            mgl::glDepthRangef(0.5f, 1.0f);
            mgl::glDepthFunc(mgl::CompareFunc::Equal);
            mgl::glUniform2f(frag_depth_.depth_line, 0.25f, 0.0f);
            draw_quad(0.0, -0.5, 0.45, 0.5, -0.5, 0.0, kNear);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthRangef(0.0f, 1.0f);
            const auto [range_px, range_py] = pixel_of(c, 0.225, 0.0);
            range_probe_ = {range_px, range_py, to_rgba8(kBase), {}};
            // Stripe 4: depth ramps from 0 to 1 across the stripe, one value per fragment.
            mgl::glUniform2f(frag_depth_.depth_line, static_cast<float>(-ramp_x0 / (ramp_x1 - ramp_x0)),
                             static_cast<float>(1.0 / (ramp_x1 - ramp_x0)));
            draw_quad(0.45, -0.5, 0.9, 0.5, kStoredNdc, 0.0, kNear);

            // Phase 2 (fixed-function depth 0.4): which depth did stripes 1 and 2 leave behind?
            use(solid_);
            draw_quad(-0.9, 0.0, 0.0, 0.5, -0.2, 0.0, kOrange);

            const double probe_z = window_z(-0.2);
            expect_color(c, -0.675, -0.25, kNear, "gl_FragDepth = 0.25 overrides interpolated 0.75: the fragment passes LESS (0.5)");
            // Stripe 1 is on top of the base; if the interpolated 0.75 had been stored instead, the
            // orange fragment (0.4) would have passed and replaced it.
            expect_color(c, -0.675, 0.25, depth_test_passes(mgl::CompareFunc::Less, probe_z, 0.25) ? kOrange : kNear,
                         "the value written to gl_FragDepth (0.25) is what is stored: a later 0.4 fragment fails");
            expect_color(c, -0.225, -0.25, kBase, "gl_FragDepth = 0.75 overrides interpolated 0.25: the fragment fails LESS (0.5)");
            expect_color(c, -0.225, 0.25, winner(depth_test_passes(mgl::CompareFunc::Less, probe_z, kStored), kOrange),
                         "the failed fragment stored nothing: a later 0.4 fragment passes against 0.5");
            for (const double fraction : {0.25, 0.75})
            {
                // Probe the pixel at this fraction of the stripe and evaluate the ramp at its center.
                const int px = static_cast<int>(std::floor(ramp_x0 + fraction * (ramp_x1 - ramp_x0)));
                const double nx = (static_cast<double>(px) + 0.5 - static_cast<double>(c.x)) / c.w * 2.0 - 1.0;
                const double frag_depth = (static_cast<double>(px) + 0.5 - ramp_x0) / (ramp_x1 - ramp_x0);  // gl_FragCoord.x = px + 0.5
                const bool passes = depth_test_passes(mgl::CompareFunc::Less, frag_depth, kStored);
                expect_color(c, nx, 0.0, winner(passes, kNear),
                             std::format("gl_FragDepth ramp: depth {:.3f} {} the stored 0.5 under LESS", frag_depth,
                                         passes ? "passes" : "fails against"));
            }
        }

        // ---- cells 14-15: glPolygonOffset ------------------------------------------------------

        struct Decal
        {
            double x0, x1;
            mgl::CompareFunc func;
            bool fill_enabled;  // POLYGON_OFFSET_FILL
            double factor, units;
            const Color* color;
            const char* label;
        };

        // A decal drawn exactly on top of the base quad (same plane, same vertices -> identical
        // depth values) loses a LESS test: equal is not less. A negative polygon offset moves the
        // decal toward the viewer by o = m*factor + r*units (§3.6.2, eq. 3.11) so it wins; a
        // positive one moves it away. glPolygonOffset alone changes nothing: POLYGON_OFFSET_FILL
        // must be enabled.
        void cell_offset_flat(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            draw_quad(-0.9, -0.9, 0.9, 0.9, kStoredNdc, 0.0, kBase);  // passes against the cleared 1.0, stores 0.5

            const std::array<Decal, 4> decals = {{
                {-0.9, -0.45, mgl::CompareFunc::Less, false, -1.0, -4.0, &kFar, "LESS, offset set but FILL disabled"},
                {-0.45, 0.0, mgl::CompareFunc::Less, true, 0.0, -4.0, &kNear, "LESS, FILL enabled, units -4"},
                {0.0, 0.45, mgl::CompareFunc::Lequal, false, 0.0, 0.0, &kEqual, "LEQUAL, no offset"},
                {0.45, 0.9, mgl::CompareFunc::Lequal, true, 0.0, 4.0, &kOrange, "LEQUAL, FILL enabled, units +4"},
            }};
            for (const Decal& d : decals)
            {
                mgl::glDepthFunc(d.func);
                mgl::glPolygonOffset(static_cast<float>(d.factor), static_cast<float>(d.units));
                if (d.fill_enabled) mgl::glEnable(mgl::Capability::PolygonOffsetFill);
                else mgl::glDisable(mgl::Capability::PolygonOffsetFill);
                draw_quad(d.x0, -0.5, d.x1, 0.5, kStoredNdc, 0.0, *d.color);
            }
            mgl::glDisable(mgl::Capability::PolygonOffsetFill);

            // A flat quad has depth slope m = 0, so o = r * units where r is the smallest
            // resolvable depth difference: 2^-bits for a fixed-point buffer.
            const double r = std::ldexp(1.0, -depth_bits_);
            for (const Decal& d : decals)
            {
                const double offset = d.fill_enabled ? 0.0 * d.factor + r * d.units : 0.0;
                const bool passes = depth_test_passes(d.func, kStored + offset, kStored);
                expect_color(c, (d.x0 + d.x1) / 2, 0.0, winner(passes, *d.color),
                             std::format("coplanar decal, {}: {}", d.label, passes ? "decal wins" : "base stays"));
            }
        }

        void cell_offset_slope(const Cell& c)
        {
            begin_cell(c);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);

            // The base is a plane tilted along x: z_ndc = b * x_ndc. Its window-space depth slope is
            // m = dz_w/dx_w = b / viewport_width (z_w = z_d/2 + 1/2 and x_d = 2 x_w / width), and
            // dz_w/dy_w = 0, so m is exact under both approximations of §3.6.2 (eq. 3.9 and 3.10).
            const double b = 0.5;
            const double m = b / static_cast<double>(c.w);
            // Decals lie in the same plane pushed back by 2m in window depth (4m in NDC): they
            // fail LESS unless the offset gives back more than 2m.
            const double delta_w = 2.0 * m;
            const double shift_ndc = 2.0 * delta_w;
            draw_quad(-0.9, -0.9, 0.9, 0.9, 0.0, b, kBase);

            const std::array<Decal, 4> decals = {{
                {-0.9, -0.45, mgl::CompareFunc::Less, true, 0.0, 0.0, &kFar, "factor 0, units 0 (still 2m behind)"},
                {-0.45, 0.0, mgl::CompareFunc::Less, true, -0.25, 0.0, &kEqual, "factor -0.25 (offset -0.25m, still behind)"},
                {0.0, 0.45, mgl::CompareFunc::Less, true, -8.0, 0.0, &kNear, "factor -8 (offset -8m, now in front)"},
                {0.45, 0.9, mgl::CompareFunc::Less, true, 0.0, -4.0, &kOrange, "units -4 only (a few LSBs, far less than 2m)"},
            }};
            for (const Decal& d : decals)
            {
                mgl::glPolygonOffset(static_cast<float>(d.factor), static_cast<float>(d.units));
                mgl::glEnable(mgl::Capability::PolygonOffsetFill);
                draw_quad(d.x0, -0.5, d.x1, 0.5, shift_ndc, b, *d.color);
            }
            mgl::glDisable(mgl::Capability::PolygonOffsetFill);

            const double r = std::ldexp(1.0, -depth_bits_);
            for (const Decal& d : decals)
            {
                const double mid = (d.x0 + d.x1) / 2;
                const double base_z = window_z(b * mid);                    // stored by the base quad
                const double offset = m * d.factor + r * d.units;           // eq. 3.11
                const double decal_z = base_z + delta_w + offset;
                const bool passes = depth_test_passes(d.func, decal_z, base_z);
                expect_color(c, mid, 0.0, winner(passes, *d.color),
                             std::format("tilted decal 2m behind the base, {}: {}", d.label, passes ? "decal wins" : "base stays"));
            }
        }

        // ---- gl_FragDepth versus the depth range: a spec rule that not every driver follows -----------

        /// ES 3.0 §3.9.2.3 (ES 3.2 §14.2.3): the shader's depth is clamped to [0, 1] and "the depth range
        /// computation is not applied". With DepthRangef(0.5, 1), a written 0.25 therefore stays 0.25 and the
        /// EQUAL test against the stored 0.5 must fail (the base color stays). llvmpipe and zink conform;
        /// radeonsi (Mesa 25.0.7) clamps the value into [n, f] = [0.5, 1], so EQUAL passes. That is recorded
        /// as a renderer-specific XFAIL (docs/KNOWN_LIMITATIONS.md); every other driver gets the strict check.
        void check_frag_depth_vs_range(demo::Check& check) const
        {
            const demo::Rgba8 got = demo::read_pixel(range_probe_.x, range_probe_.y);
            const demo::Rgba8 want = to_rgba8(kBase);
            bool ok = true;
            for (std::size_t i = 0; i < 4; ++i) ok = ok && std::abs(int{got[i]} - int{want[i]}) <= 1;
            check.expect_or_known_deviation(
                ok, "gl_FragDepth is not remapped by DepthRangef(0.5, 1): 0.25 stays 0.25, EQUAL vs 0.5 fails (§3.9.2.3)",
                std::format("pixel({},{}) = {} expected {}", range_probe_.x, range_probe_.y, demo::format_rgba(got), demo::format_rgba(want)),
                "radeonsi", "Mesa 25.0.7 radeonsi clamps gl_FragDepth into [near, far]");
        }

        // ---- state queries ---------------------------------------------------------------------

        void check_initial_state(demo::Check& check) const
        {
            // ES 3.0 §4.1.5, §4.2.2, §4.2.3, §2.13.1, §3.6.2 (initial values) and Table 6.x queries.
            check.expect(!initial_.depth_test, "initial state: DEPTH_TEST is disabled (§4.1.5)");
            check.expect(!initial_.offset_fill, "initial state: POLYGON_OFFSET_FILL is disabled (§3.6.2)");
            check.expect_eq(initial_.depth_func, static_cast<int>(mgl::CompareFunc::Less), "initial state: DEPTH_FUNC is LESS (§4.1.5)");
            check.expect(initial_.depth_mask == GL_TRUE, "initial state: DEPTH_WRITEMASK is TRUE (§4.2.2)");
            check.expect_near(static_cast<double>(initial_.depth_range[0]), 0.0, 0.0, "initial state: DEPTH_RANGE near = 0 (§2.13.1)");
            check.expect_near(static_cast<double>(initial_.depth_range[1]), 1.0, 0.0, "initial state: DEPTH_RANGE far = 1 (§2.13.1)");
            check.expect_near(static_cast<double>(initial_.clear_value), 1.0, 0.0, "initial state: DEPTH_CLEAR_VALUE is 1.0 (§4.2.3)");
            check.expect_near(static_cast<double>(initial_.offset_factor), 0.0, 0.0, "initial state: POLYGON_OFFSET_FACTOR is 0 (§3.6.2)");
            check.expect_near(static_cast<double>(initial_.offset_units), 0.0, 0.0, "initial state: POLYGON_OFFSET_UNITS is 0 (§3.6.2)");
            check.expect(initial_.depth_bits >= 16, "the default framebuffer has a depth buffer (DEPTH_BITS >= 16)",
                         std::format("DEPTH_BITS = {}", initial_.depth_bits));
        }

        static void check_state_round_trips(demo::Check& check)
        {
            for (const FuncName& fn : kFuncs)
            {
                mgl::glDepthFunc(fn.func);
                check.expect_eq(demo::get_integer(mgl::GetParameter::DepthFunc), static_cast<int>(fn.func),
                                std::format("glDepthFunc({}) is returned by GL_DEPTH_FUNC", fn.name));
            }

            mgl::glEnable(mgl::Capability::DepthTest);
            check.expect(mgl::glIsEnabled(mgl::Capability::DepthTest), "glEnable(DEPTH_TEST): glIsEnabled is TRUE");
            mgl::glDisable(mgl::Capability::DepthTest);
            check.expect(!mgl::glIsEnabled(mgl::Capability::DepthTest), "glDisable(DEPTH_TEST): glIsEnabled is FALSE");
            mgl::glEnable(mgl::Capability::PolygonOffsetFill);
            check.expect(mgl::glIsEnabled(mgl::Capability::PolygonOffsetFill), "glEnable(POLYGON_OFFSET_FILL): glIsEnabled is TRUE");
            mgl::glDisable(mgl::Capability::PolygonOffsetFill);
            check.expect(!mgl::glIsEnabled(mgl::Capability::PolygonOffsetFill), "glDisable(POLYGON_OFFSET_FILL): glIsEnabled is FALSE");

            GLboolean mask = GL_TRUE;
            mgl::glDepthMask(GL_FALSE);
            mgl::glGetBooleanv(mgl::GetParameter::DepthWritemask, &mask);
            check.expect(mask == GL_FALSE, "glDepthMask(FALSE): DEPTH_WRITEMASK is FALSE");
            mgl::glDepthMask(GL_TRUE);
            mgl::glGetBooleanv(mgl::GetParameter::DepthWritemask, &mask);
            check.expect(mask == GL_TRUE, "glDepthMask(TRUE): DEPTH_WRITEMASK is TRUE");

            // §2.13.1: n and f are clamped to [0,1] when specified; n > f is allowed and kept.
            struct RangeCase
            {
                float n, f, expect_n, expect_f;
                const char* label;
            };
            const RangeCase range_cases[] = {
                {0.25f, 0.75f, 0.25f, 0.75f, "(0.25, 0.75)"},
                {-1.0f, 2.0f, 0.0f, 1.0f, "(-1, 2) is clamped to (0, 1)"},
                {1.0f, 0.0f, 1.0f, 0.0f, "(1, 0) reversed range is kept"},
            };
            for (const RangeCase& rc : range_cases)
            {
                std::array<GLfloat, 2> range{};
                mgl::glDepthRangef(rc.n, rc.f);
                mgl::glGetFloatv(mgl::GetParameter::DepthRange, range.data());
                check.expect_near(static_cast<double>(range[0]), static_cast<double>(rc.expect_n), 1e-6,
                                  std::format("glDepthRangef{}: DEPTH_RANGE near", rc.label));
                check.expect_near(static_cast<double>(range[1]), static_cast<double>(rc.expect_f), 1e-6,
                                  std::format("glDepthRangef{}: DEPTH_RANGE far", rc.label));
            }

            // §4.2.3: the clear value is clamped to [0,1].
            const std::pair<float, float> clears[] = {{0.3f, 0.3f}, {5.0f, 1.0f}, {-3.0f, 0.0f}};
            for (const auto& [set, expected] : clears)
            {
                GLfloat got = -1.0f;
                mgl::glClearDepthf(set);
                mgl::glGetFloatv(mgl::GetParameter::DepthClearValue, &got);
                check.expect_near(static_cast<double>(got), static_cast<double>(expected), 1e-6,
                                  std::format("glClearDepthf({}): DEPTH_CLEAR_VALUE", set));
            }

            // §3.6.2: factor and units are stored unclamped, either sign.
            GLfloat factor = 0.0f, units = 0.0f;
            mgl::glPolygonOffset(2.5f, -3.5f);
            mgl::glGetFloatv(mgl::GetParameter::PolygonOffsetFactor, &factor);
            mgl::glGetFloatv(mgl::GetParameter::PolygonOffsetUnits, &units);
            check.expect_near(static_cast<double>(factor), 2.5, 1e-6, "glPolygonOffset(2.5, -3.5): POLYGON_OFFSET_FACTOR");
            check.expect_near(static_cast<double>(units), -3.5, 1e-6, "glPolygonOffset(2.5, -3.5): POLYGON_OFFSET_UNITS");
        }

        std::vector<Probe> probes_;
        Probe range_probe_{0, 0, {}, {}};
        InitialState initial_;
        int depth_bits_ = 24;
        Program solid_, frag_depth_;
        const Program* current_ = nullptr;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-depth-test",
        .title = "Depth buffer test",
        .min_es = {3, 0},
        .summary = "All eight depth functions, the depth write mask, clear value, depth range, polygon offset "
                   "and gl_FragDepth; the winning color of 16 overlapping-quad experiments is checked on the CPU.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<DepthTest>(argc, argv, info);
}
