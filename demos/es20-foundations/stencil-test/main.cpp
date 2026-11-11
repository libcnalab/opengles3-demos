// es20-stencil-test
//
// The stencil buffer: the comparison functions and masks of the stencil test, all eight stencil
// operations (including the clamping and wrapping behaviour at 0 and 255), the three outcomes
// sfail / dpfail / dppass, the write mask, separate front/back state, and two classic uses
// (a stencil-masked "portal" with an outline, and an even-odd polygon fill).
//
// OpenGL ES cannot read the stencil buffer back (glReadPixels does not accept STENCIL_INDEX), so
// the demo *measures* the stencil value indirectly with the stencil test itself. The readout is
// 255 full-window draws: draw number v is made with glStencilFunc(GL_EQUAL, v, 0xFF), glStencilOp
// all KEEP, and writes the color (v / 255, 0.4, 0.4). A pixel whose stencil value is v passes
// exactly one of those tests and so ends up with red == v. Reading the red channel therefore
// reads the stencil value, exactly, for all 256 possible values. The color mask is off while the
// experiments run, so nothing but the readout writes color.
//
// The top of the window is a grid of experiments (a heat map: red = the stencil value that the
// experiment left behind). Each cell is cleared to a start value, receives one or more draws with
// the state under test, and is then probed. The expected value of every cell is computed by a
// small CPU model of the rules in ES 3.0 §4.1.4, §4.1.5, §4.2.2 and §4.2.3:
//
//     rows 1-5, left  : A  the eight stencil operations (columns) from five start values (rows)
//     row 6,    left  : C  glStencilMask - only the masked bits change, also for glClear
//     row 7,    left  : D  sfail / dpfail / dppass with the depth test, depth mask, disabled test
//     rows 1-7, right : B  the eight stencil functions (columns) with reference / mask / clamping
//     row 8 (below)   : E  separate front/back state: winding, glFrontFace, points, masks
//
// The bottom half shows two visual uses, each probed against a CPU reference image.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    using Color = std::array<float, 4>;

    // Components are multiples of 0.2 (exact 8-bit values), except the readout's 0.4 which is 102.
    constexpr Color kCellBase = {0.0f, 0.4f, 0.4f, 1.0f};  // equals the readout color for value 0
    constexpr Color kPanelBg = {0.2f, 0.2f, 0.2f, 1.0f};
    constexpr Color kCoolA = {0.2f, 0.4f, 0.8f, 1.0f}, kCoolB = {0.4f, 0.6f, 1.0f, 1.0f};  // outside the window
    constexpr Color kWarmA = {1.0f, 0.6f, 0.2f, 1.0f}, kWarmB = {1.0f, 0.8f, 0.4f, 1.0f};  // inside the window
    constexpr Color kGoldA = {1.0f, 0.8f, 0.2f, 1.0f}, kGoldB = {1.0f, 0.6f, 0.0f, 1.0f};  // star fill
    constexpr Color kWhite = {1.0f, 1.0f, 1.0f, 1.0f};

    demo::Rgba8 to_rgba8(const Color& c)
    {
        auto q = [](float v) { return static_cast<std::uint8_t>(std::lround(v * 255.0f)); };
        return {q(c[0]), q(c[1]), q(c[2]), q(c[3])};
    }

    /// Color of the diagonal stripe pattern of stripes.frag at window pixel (px, py). The shader
    /// evaluates floor((x + y) / period) at the pixel center, i.e. (px + py + 1) / period.
    const Color& stripe_color(int px, int py, const Color& a, const Color& b, int period)
    {
        return ((px + py + 1) / period) % 2 == 0 ? a : b;
    }

    // ---- CPU model of the stencil (and depth) pipeline -------------------------------------------

    constexpr unsigned kAllBits = 0xFFFFFFFFu;  // the specified initial masks: all ones
    constexpr unsigned kStencilMax = 255;       // 8 stencil bits

    /// ES 3.0 §4.1.4: the stencil test passes if (ref & mask) <func> (stored & mask). Note the
    /// operand order: the *reference* is on the left, the opposite of the depth test.
    constexpr bool stencil_compare(mgl::CompareFunc func, unsigned masked_ref, unsigned masked_stored)
    {
        switch (func)
        {
            case mgl::CompareFunc::Never: return false;
            case mgl::CompareFunc::Less: return masked_ref < masked_stored;
            case mgl::CompareFunc::Equal: return masked_ref == masked_stored;
            case mgl::CompareFunc::Lequal: return masked_ref <= masked_stored;
            case mgl::CompareFunc::Greater: return masked_ref > masked_stored;
            case mgl::CompareFunc::Notequal: return masked_ref != masked_stored;
            case mgl::CompareFunc::Gequal: return masked_ref >= masked_stored;
            case mgl::CompareFunc::Always: return true;
        }
        return false;
    }

    /// ES 3.0 §4.1.4: the eight stencil operations on an 8-bit unsigned value.
    constexpr unsigned stencil_apply(mgl::StencilOp op, unsigned stored, unsigned ref)
    {
        switch (op)
        {
            case mgl::StencilOp::Keep: return stored;
            case mgl::StencilOp::Zero: return 0;
            case mgl::StencilOp::Replace: return ref;
            case mgl::StencilOp::Incr: return stored < kStencilMax ? stored + 1 : kStencilMax;  // saturates
            case mgl::StencilOp::Decr: return stored > 0 ? stored - 1 : 0;                      // saturates
            case mgl::StencilOp::Invert: return ~stored & kStencilMax;
            case mgl::StencilOp::IncrWrap: return (stored + 1) & kStencilMax;                   // 255 -> 0
            case mgl::StencilOp::DecrWrap: return (stored - 1) & kStencilMax;                   // 0 -> 255
        }
        return stored;
    }

    /// "Stencil comparison operations and queries of ref clamp its value to [0, 2^s - 1]."
    constexpr unsigned clamp_ref(int ref) { return static_cast<unsigned>(std::clamp(ref, 0, static_cast<int>(kStencilMax))); }

    /// One face's stencil state (glStencilFunc / glStencilOp / glStencilMask).
    struct StencilSide
    {
        mgl::CompareFunc func = mgl::CompareFunc::Always;
        int ref = 0;
        unsigned mask = 0xFF;  // value mask for the comparison
        mgl::StencilOp sfail = mgl::StencilOp::Keep, dpfail = mgl::StencilOp::Keep, dppass = mgl::StencilOp::Keep;
        unsigned write_mask = 0xFF;
        constexpr bool operator==(const StencilSide&) const = default;
    };

    enum class Prim { TriangleCcw, TriangleCw, Point };
    enum class Calls { Auto, FrontAndBack };  // how the state is submitted (see submit())

    /// One step of an experiment: either a draw or a glClear(STENCIL), with the state it needs.
    struct Step
    {
        bool is_clear = false;
        int clear_value = 0;
        bool stencil_test = true;
        StencilSide front{}, back{};
        Calls calls = Calls::Auto;
        bool depth_test = false;  // GL_LESS when enabled
        bool depth_write = true;
        double z_ndc = 0.0;
        Prim prim = Prim::TriangleCcw;
    };

    Step draw_with(const StencilSide& both)
    {
        Step s;
        s.front = s.back = both;
        return s;
    }

    /// One grid cell: cleared to `start`, then the steps run, then the stencil value is read out.
    struct Experiment
    {
        std::string label;
        unsigned start = 0;
        bool front_face_cw = false;
        std::vector<Step> steps;
    };

    constexpr double kDepthStart = 0.5;  // depth buffer value in every experiment cell

    /// ES 3.0 §2.13.1 with the default depth range.
    constexpr double window_z(double z_ndc) { return z_ndc / 2.0 + 0.5; }

    /// Applies the rules of §4.1.4 / §4.1.5 / §4.2.2 / §4.2.3 to a single pixel and returns the
    /// stencil value that remains. This is the CPU reference for every grid cell.
    unsigned model_run(const Experiment& e)
    {
        unsigned stored = e.start;
        double depth = kDepthStart;
        for (const Step& s : e.steps)
        {
            if (s.is_clear)
            {
                // The clear value is masked to s bits and the *front* write mask applies (§4.2.2, §4.2.3).
                const unsigned v = static_cast<unsigned>(s.clear_value) & kStencilMax;
                const unsigned wm = s.front.write_mask & kStencilMax;
                stored = (stored & ~wm & kStencilMax) | (v & wm);
                continue;
            }
            // Front-facing polygons and all non-polygon primitives use the front state.
            const bool ccw = s.prim == Prim::TriangleCcw;
            const bool front_facing = s.prim == Prim::Point || (ccw != e.front_face_cw);
            const StencilSide& side = front_facing ? s.front : s.back;
            const unsigned ref = clamp_ref(side.ref);

            auto update = [&](mgl::StencilOp op) {
                const unsigned updated = stencil_apply(op, stored, ref);
                stored = (stored & ~side.write_mask & kStencilMax) | (updated & side.write_mask & kStencilMax);
            };

            if (s.stencil_test)
            {
                const unsigned mask = side.mask & kStencilMax;
                if (!stencil_compare(side.func, ref & mask, stored & mask))
                {
                    update(side.sfail);  // the fragment is discarded: no depth update
                    continue;
                }
            }
            const double z = window_z(s.z_ndc);
            const bool depth_passes = !s.depth_test || z < depth;
            if (s.stencil_test) update(depth_passes ? side.dppass : side.dpfail);
            if (s.depth_test && depth_passes && s.depth_write) depth = z;  // §4.1.5: only an enabled test writes depth
        }
        return stored;
    }

    struct NamedFunc
    {
        mgl::CompareFunc func;
        const char* name;
    };
    constexpr std::array<NamedFunc, 8> kFuncs = {{
        {mgl::CompareFunc::Never, "NEVER"},
        {mgl::CompareFunc::Less, "LESS"},
        {mgl::CompareFunc::Equal, "EQUAL"},
        {mgl::CompareFunc::Lequal, "LEQUAL"},
        {mgl::CompareFunc::Greater, "GREATER"},
        {mgl::CompareFunc::Notequal, "NOTEQUAL"},
        {mgl::CompareFunc::Gequal, "GEQUAL"},
        {mgl::CompareFunc::Always, "ALWAYS"},
    }};

    struct NamedOp
    {
        mgl::StencilOp op;
        const char* name;
    };
    constexpr std::array<NamedOp, 8> kOps = {{
        {mgl::StencilOp::Keep, "KEEP"},
        {mgl::StencilOp::Zero, "ZERO"},
        {mgl::StencilOp::Replace, "REPLACE"},
        {mgl::StencilOp::Incr, "INCR"},
        {mgl::StencilOp::Decr, "DECR"},
        {mgl::StencilOp::Invert, "INVERT"},
        {mgl::StencilOp::IncrWrap, "INCR_WRAP"},
        {mgl::StencilOp::DecrWrap, "DECR_WRAP"},
    }};

    // ---- layout ------------------------------------------------------------------------------------

    struct Rect
    {
        int x, y, w, h;  // window pixels, origin lower left
    };

    struct Program
    {
        mgl::ProgramId id{};
        mgl::UniformLocation color{}, flip{}, depth{}, center{}, point_size{}, circle{}, star{}, color_a{}, color_b{}, period{};
    };

    /// A stencil value that validate() expects at a cell's probe pixel.
    struct CellProbe
    {
        int x, y;
        unsigned expected;
        std::string what;
    };

    /// A pixel color that validate() expects in one of the visual panels.
    struct PixelProbe
    {
        int x, y;
        demo::Rgba8 expected;
        std::string what;
    };

    /// Stencil state of one face as returned by glGetIntegerv.
    struct QueriedSide
    {
        int func = 0, ref = 0, value_mask = 0, fail = 0, dpfail = 0, dppass = 0, write_mask = 0;
    };

    struct InitialState
    {
        bool stencil_test = true;
        QueriedSide front, back;
        int clear_value = -1;
        int stencil_bits = 0;
    };

    constexpr int kSegments = 48;  // keep in sync with circle.vert
    constexpr int kDesign = 256;   // the layout is designed on a 256x256 grid and stretched to the surface
    constexpr double kWindowRadius = 0.55, kOutlineRadius = 0.67;
    constexpr double kStarRadius = 0.8;

    class StencilTest final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // The 256x256 design is stretched to the surface. Below a quarter of it in either direction an
            // experiment cell (16x14 design px) is under 4x3.5 px - one border pixel on each side leaves almost
            // nothing to probe - and the visual panels (120 design px) are too coarse for the probes' margins
            // around the circle and star edges (0.06 and 0.044 NDC).
            if (env.width < kDesign / 4 || env.height < kDesign / 4)
                throw demo::Skip(std::format("the stretched 256x256 layout needs at least a {0}x{0} surface (got {1}x{2})",
                                             kDesign / 4, env.width, env.height));
            fill_ = make_program(env, "shaders/fullscreen.vert", "shaders/solid.frag");
            stripes_ = make_program(env, "shaders/fullscreen.vert", "shaders/stripes.frag");
            point_ = make_program(env, "shaders/point.vert", "shaders/solid.frag");
            circle_ = make_program(env, "shaders/circle.vert", "shaders/solid.frag");
            star_ = make_program(env, "shaders/star.vert", "shaders/solid.frag");

            std::array<GLfloat, 2> point_range{1.0f, 1.0f};
            mgl::glGetFloatv(mgl::GetParameter::AliasedPointSizeRange, point_range.data());
            point_size_ = std::clamp(6.0f, point_range[0], point_range[1]);

            // The context starts in the default state; remember it for validate().
            initial_.stencil_test = mgl::glIsEnabled(mgl::Capability::StencilTest);
            initial_.front = read_side(false);
            initial_.back = read_side(true);
            initial_.clear_value = demo::get_integer(mgl::GetParameter::StencilClearValue);
            initial_.stencil_bits = demo::get_integer(mgl::GetParameter::StencilBits);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            cell_probes_.clear();
            pixel_probes_.clear();
            width_ = env.width;
            height_ = env.height;

            reset_state();
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);

            panel_operations();
            panel_functions();
            panel_write_masks();
            panel_depth_interplay();
            panel_faces();
            readout(env);

            portal_panel(top_rect(4, 124, 120, 120));
            star_panel(top_rect(132, 124, 120, 120));

            reset_state();
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            for (const CellProbe& p : cell_probes_)
            {
                // The readout color is (v/255, 0.4, 0.4); a stencil value is exact, so no tolerance.
                const auto v = static_cast<std::uint8_t>(p.expected);
                check.expect_pixel(p.x, p.y, {v, 102, 102, 255}, 0, p.what);
            }
            for (const PixelProbe& p : pixel_probes_) check.expect_pixel(p.x, p.y, p.expected, 1, p.what);
            check_initial_state(check);
            check_state_round_trips(check);
            reset_state();
            check.expect_no_gl_error("no GL error raised by the stencil experiments");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            reset_state();
            mgl::glUseProgram(mgl::ProgramId{0});
            for (const Program* p : {&fill_, &stripes_, &point_, &circle_, &star_}) mgl::glDeleteProgram(p->id);
        }

    private:
        // ---- programs and small GL helpers ---------------------------------------------------------

        static Program make_program(const demo::Env& env, const char* vs, const char* fs)
        {
            Program p;
            p.id = demo::build_program(env, vs, fs);
            p.color = mgl::glGetUniformLocation(p.id, "u_color");
            p.flip = mgl::glGetUniformLocation(p.id, "u_flip");
            p.depth = mgl::glGetUniformLocation(p.id, "u_depth");
            p.center = mgl::glGetUniformLocation(p.id, "u_center");
            p.point_size = mgl::glGetUniformLocation(p.id, "u_point_size");
            p.circle = mgl::glGetUniformLocation(p.id, "u_circle");
            p.star = mgl::glGetUniformLocation(p.id, "u_star");
            p.color_a = mgl::glGetUniformLocation(p.id, "u_color_a");
            p.color_b = mgl::glGetUniformLocation(p.id, "u_color_b");
            p.period = mgl::glGetUniformLocation(p.id, "u_period");
            return p;
        }

        void use(const Program& p)
        {
            mgl::glUseProgram(p.id);
            current_ = &p;
        }

        void set_color(const Color& c) { mgl::glUniform4f(current_->color, c[0], c[1], c[2], c[3]); }

        /// glDrawArrays of the one-triangle viewport cover (works with fill_ and stripes_).
        void draw_cover(bool clockwise, double z_ndc)
        {
            mgl::glUniform1i(current_->flip, clockwise ? 1 : 0);
            mgl::glUniform1f(current_->depth, static_cast<float>(z_ndc));
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        static QueriedSide read_side(bool back)
        {
            using P = mgl::GetParameter;
            QueriedSide s;
            s.func = demo::get_integer(back ? P::StencilBackFunc : P::StencilFunc);
            s.ref = demo::get_integer(back ? P::StencilBackRef : P::StencilRef);
            s.value_mask = demo::get_integer(back ? P::StencilBackValueMask : P::StencilValueMask);
            s.fail = demo::get_integer(back ? P::StencilBackFail : P::StencilFail);
            s.dpfail = demo::get_integer(back ? P::StencilBackPassDepthFail : P::StencilPassDepthFail);
            s.dppass = demo::get_integer(back ? P::StencilBackPassDepthPass : P::StencilPassDepthPass);
            s.write_mask = demo::get_integer(back ? P::StencilBackWritemask : P::StencilWritemask);
            return s;
        }

        /// Everything an experiment may have changed, back to the specified defaults.
        static void reset_state()
        {
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::StencilTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glDepthMask(GL_TRUE);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glFrontFace(mgl::FrontFace::CCW);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, kAllBits);  // sets front and back
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            mgl::glStencilMask(kAllBits);
            mgl::glClearStencil(0);
            mgl::glClearDepthf(1.0f);
        }

        /// A rect given in "design pixels" of a 256x256 design, measured from the top-left corner of the
        /// window. The design is stretched to the surface, independently in x and y: every design edge maps
        /// to round(edge * size / 256), so neighbouring rects share their edges and the layout partitions
        /// the surface exactly at any size and aspect ratio.
        Rect top_rect(int x, int y_top, int w, int h) const
        {
            auto sx = [this](int v) { return static_cast<int>(std::lround(static_cast<double>(v) * width_ / kDesign)); };
            auto sy = [this](int v) { return static_cast<int>(std::lround(static_cast<double>(v) * height_ / kDesign)); };
            return {sx(x), height_ - sy(y_top + h), sx(x + w) - sx(x), sy(y_top + h) - sy(y_top)};
        }

        /// Pixel of window position (nx, ny) in the NDC of `r`.
        static std::pair<int, int> pixel_at(const Rect& r, double nx, double ny)
        {
            return {r.x + static_cast<int>(std::floor((nx + 1.0) * 0.5 * r.w)),
                    r.y + static_cast<int>(std::floor((ny + 1.0) * 0.5 * r.h))};
        }

        /// NDC position (within `r`) of the center of window pixel (px, py): where the rasterizer samples.
        static std::pair<double, double> pixel_center_ndc(const Rect& r, int px, int py)
        {
            return {(static_cast<double>(px - r.x) + 0.5) / r.w * 2.0 - 1.0, (static_cast<double>(py - r.y) + 0.5) / r.h * 2.0 - 1.0};
        }

        // ---- running one experiment cell -----------------------------------------------------------

        /// Cell (col, row) of the grid whose first row starts `band_y` design pixels from the top.
        Rect lab_cell(int col, int row, int band_y) const { return top_rect(col * 16, band_y + row * 14, 16, 14); }

        void run(const Rect& cell, const Experiment& e)
        {
            // The experiment draws into the cell minus a one-pixel border (left black).
            const Rect view{cell.x + 1, cell.y + 1, cell.w - 2, cell.h - 2};
            probe_x_ = cell.x + cell.w / 2;
            probe_y_ = cell.y + cell.h / 2;
            view_ = view;

            reset_state();
            mgl::glViewport(view.x, view.y, view.w, view.h);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(view.x, view.y, view.w, view.h);
            // The cell starts at the experiment's stencil value and a depth of 0.5. Both clears obey the
            // write masks, which reset_state() has just set to "all ones".
            mgl::glClearColor(kCellBase[0], kCellBase[1], kCellBase[2], kCellBase[3]);
            mgl::glClearStencil(static_cast<GLint>(e.start));
            mgl::glClearDepthf(static_cast<float>(kDepthStart));
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);

            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);  // experiments only touch stencil/depth
            mgl::glFrontFace(e.front_face_cw ? mgl::FrontFace::CW : mgl::FrontFace::CCW);
            for (const Step& s : e.steps) submit(s);

            cell_probes_.push_back({probe_x_, probe_y_, model_run(e), e.label});
        }

        /// Issues the GL state and the draw (or clear) of one step.
        void submit(const Step& s)
        {
            // Depth test: GL_LESS (the initial function) whenever it is enabled.
            if (s.depth_test) mgl::glEnable(mgl::Capability::DepthTest);
            else mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDepthMask(s.depth_write ? GL_TRUE : GL_FALSE);

            // Stencil test and its three pieces of state. glStencilFunc/Op/Mask set the front AND back
            // state; the *Separate variants set one face (or both with GL_FRONT_AND_BACK).
            if (s.stencil_test) mgl::glEnable(mgl::Capability::StencilTest);
            else mgl::glDisable(mgl::Capability::StencilTest);

            const StencilSide& f = s.front;
            const StencilSide& b = s.back;
            if (s.calls == Calls::FrontAndBack)
            {
                mgl::glStencilFuncSeparate(mgl::CullFace::FrontAndBack, f.func, f.ref, f.mask);
                mgl::glStencilOpSeparate(mgl::CullFace::FrontAndBack, f.sfail, f.dpfail, f.dppass);
                mgl::glStencilMaskSeparate(mgl::CullFace::FrontAndBack, f.write_mask);
            }
            else if (f == b)
            {
                mgl::glStencilFunc(f.func, f.ref, f.mask);
                mgl::glStencilOp(f.sfail, f.dpfail, f.dppass);
                mgl::glStencilMask(f.write_mask);
            }
            else
            {
                mgl::glStencilFuncSeparate(mgl::CullFace::Front, f.func, f.ref, f.mask);
                mgl::glStencilOpSeparate(mgl::CullFace::Front, f.sfail, f.dpfail, f.dppass);
                mgl::glStencilMaskSeparate(mgl::CullFace::Front, f.write_mask);
                mgl::glStencilFuncSeparate(mgl::CullFace::Back, b.func, b.ref, b.mask);
                mgl::glStencilOpSeparate(mgl::CullFace::Back, b.sfail, b.dpfail, b.dppass);
                mgl::glStencilMaskSeparate(mgl::CullFace::Back, b.write_mask);
            }

            if (s.is_clear)
            {
                mgl::glClearStencil(s.clear_value);
                mgl::glClear(mgl::ClearBufferBit::Stencil);  // masked by the front write mask, scissored to the cell
                return;
            }

            if (s.prim == Prim::Point)
            {
                // Put the point at the center of the probe pixel: pixel-center NDC within the cell's viewport.
                const double nx = (static_cast<double>(probe_x_ - view_.x) + 0.5) / view_.w * 2.0 - 1.0;
                const double ny = (static_cast<double>(probe_y_ - view_.y) + 0.5) / view_.h * 2.0 - 1.0;
                use(point_);
                mgl::glUniform2f(point_.center, static_cast<float>(nx), static_cast<float>(ny));
                mgl::glUniform1f(point_.point_size, point_size_);
                mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 1);
            }
            else
            {
                use(fill_);
                set_color(kWhite);
                draw_cover(s.prim == Prim::TriangleCw, s.z_ndc);
            }
        }

        // ---- panel A: the eight stencil operations ---------------------------------------------------

        void panel_operations()
        {
            constexpr std::array<unsigned, 5> starts = {0, 1, 0x55, 254, 255};
            constexpr int kRef = 0x2A;  // REPLACE writes this; it differs from every start value
            for (std::size_t col = 0; col < kOps.size(); ++col)
                for (std::size_t row = 0; row < starts.size(); ++row)
                {
                    StencilSide side;  // func ALWAYS, depth test off: every fragment takes the dppass path
                    side.ref = kRef;
                    side.sfail = side.dpfail = side.dppass = kOps[col].op;
                    Experiment e;
                    e.label = std::format("StencilOp {} applied to stencil value {}", kOps[col].name, starts[row]);
                    e.start = starts[row];
                    e.steps = {draw_with(side)};
                    run(lab_cell(static_cast<int>(col), static_cast<int>(row), 0), e);
                }
        }

        // ---- panel B: the eight stencil functions ----------------------------------------------------

        void panel_functions()
        {
            struct Case
            {
                unsigned stored;
                int ref;
                unsigned mask;
                const char* what;
            };
            // Stored values are never 0: a failed test zeroes the pixel (sfail = ZERO) and a passed
            // test keeps it (dppass = KEEP), so "0" unambiguously means "failed".
            constexpr std::array<Case, 7> cases = {{
                {100, 50, 0xFF, "ref < stored"},
                {100, 100, 0xFF, "ref == stored"},
                {100, 150, 0xFF, "ref > stored"},
                {0x65, 0x35, 0x0F, "masked low nibbles equal (5 == 5) although ref < stored"},
                {0x1A, 0xF3, 0x0F, "masked low nibbles 3 < 10 although ref > stored"},
                {255, 263, 0xFF, "ref 263 clamps to 255 (not masked to 7)"},
                {1, -1, 0xFF, "ref -1 clamps to 0"},
            }};
            for (std::size_t col = 0; col < kFuncs.size(); ++col)
                for (std::size_t row = 0; row < cases.size(); ++row)
                {
                    const Case& c = cases[row];
                    StencilSide side;
                    side.func = kFuncs[col].func;
                    side.ref = c.ref;
                    side.mask = c.mask;
                    side.sfail = mgl::StencilOp::Zero;
                    Experiment e;
                    e.label = std::format("StencilFunc({}, ref {}, mask {:#x}) on stored {}: {}", kFuncs[col].name, c.ref, c.mask,
                                          c.stored, c.what);
                    e.start = c.stored;
                    e.steps = {draw_with(side)};
                    run(lab_cell(static_cast<int>(col) + 8, static_cast<int>(row), 0), e);
                }
        }

        // ---- panel C: glStencilMask -------------------------------------------------------------------

        void panel_write_masks()
        {
            struct Case
            {
                unsigned start;
                unsigned mask;
                mgl::StencilOp op;
                const char* op_name;
                int ref;
            };
            constexpr std::array<Case, 6> cases = {{
                {0xAA, 0x0F, mgl::StencilOp::Replace, "REPLACE 0x35", 0x35},
                {0xAA, 0x3C, mgl::StencilOp::Invert, "INVERT", 0},
                {0xFF, 0xF0, mgl::StencilOp::Zero, "ZERO", 0},
                {0x0F, 0xF0, mgl::StencilOp::Incr, "INCR (carries out of the masked nibble)", 0},
                {0x10, 0x0F, mgl::StencilOp::Decr, "DECR (borrows from the masked nibble)", 0},
                {0x33, 0x00, mgl::StencilOp::Replace, "REPLACE 0x77", 0x77},
            }};
            int col = 0;
            for (const Case& c : cases)
            {
                StencilSide side;
                side.ref = c.ref;
                side.sfail = side.dpfail = side.dppass = c.op;
                side.write_mask = c.mask;
                Experiment e;
                e.label = std::format("StencilMask({:#04x}) with {} on {:#04x}", c.mask, c.op_name, c.start);
                e.start = c.start;
                e.steps = {draw_with(side)};
                run(lab_cell(col++, 5, 0), e);
            }

            // glClear(STENCIL) is subject to the write mask, and uses the *front* mask (§4.2.2).
            {
                // The stencil test (func NEVER here) does not apply to clears: only the write mask does.
                StencilSide never;
                never.func = mgl::CompareFunc::Never;
                Step clear = draw_with(never);
                clear.is_clear = true;
                clear.clear_value = 0xFF;
                clear.front.write_mask = clear.back.write_mask = 0x81;
                Experiment e;
                e.label = "glClearStencil(0xFF) + glClear with StencilMask(0x81) on 0x00";
                e.start = 0x00;
                e.steps = {clear};
                run(lab_cell(col++, 5, 0), e);
            }
            {
                StencilSide never;
                never.func = mgl::CompareFunc::Never;
                Step clear = draw_with(never);
                clear.is_clear = true;
                clear.clear_value = 0xFF;
                clear.front.write_mask = 0x0F;  // StencilMaskSeparate(FRONT, 0x0F) / (BACK, 0xF0)
                clear.back.write_mask = 0xF0;
                Experiment e;
                e.label = "glClear with front mask 0x0F, back mask 0xF0: the clear uses the front mask";
                e.start = 0x00;
                e.steps = {clear};
                run(lab_cell(col++, 5, 0), e);
            }
        }

        // ---- panel D: sfail / dpfail / dppass ---------------------------------------------------------

        void panel_depth_interplay()
        {
            // Every draw uses distinguishable ops: sfail = INCR, dpfail = ZERO, dppass = REPLACE 0x7E.
            // The cell starts at stencil 0x10 and depth 0.5; the depth test is LESS, so a fragment at
            // NDC z = -0.5 (window 0.25) passes it and one at z = +0.5 (0.75) fails it.
            auto stage_side = [](mgl::CompareFunc func, mgl::StencilOp sfail, mgl::StencilOp dpfail, mgl::StencilOp dppass) {
                StencilSide s;
                s.func = func;
                s.ref = 0x7E;
                s.sfail = sfail;
                s.dpfail = dpfail;
                s.dppass = dppass;
                return s;
            };
            using Op = mgl::StencilOp;
            const StencilSide never = stage_side(mgl::CompareFunc::Never, Op::Incr, Op::Zero, Op::Replace);
            const StencilSide always = stage_side(mgl::CompareFunc::Always, Op::Incr, Op::Zero, Op::Replace);
            const StencilSide later = stage_side(mgl::CompareFunc::Always, Op::Keep, Op::Zero, Op::Replace);

            auto depth_draw = [](const StencilSide& side, double z, bool depth_test = true, bool depth_write = true) {
                Step s = draw_with(side);
                s.depth_test = depth_test;
                s.depth_write = depth_write;
                s.z_ndc = z;
                return s;
            };

            struct Case
            {
                const char* label;
                std::vector<Step> steps;
            };
            std::vector<Case> cases;
            cases.push_back({"stencil test fails (NEVER) while the depth test would pass: sfail = INCR",
                             {depth_draw(never, -0.5)}});
            cases.push_back({"stencil passes, depth fails (0.75 vs 0.5): dpfail = ZERO", {depth_draw(always, 0.5)}});
            cases.push_back({"stencil passes, depth passes (0.25 vs 0.5): dppass = REPLACE", {depth_draw(always, -0.5)}});
            cases.push_back({"depth test disabled: treated as passed (dppass = REPLACE), although 0.75 > 0.5",
                             {depth_draw(always, 0.5, false)}});
            // A fragment discarded by the stencil test must not update the depth buffer: the second draw
            // (0.4 vs 0.5) then still passes and takes dppass. If the first had stored 0.25, it would fail.
            cases.push_back({"a stencil-failed fragment writes no depth: later 0.4 fragment passes",
                             {depth_draw(never, -0.5), depth_draw(later, -0.2)}});
            // Conversely a fragment that passes both tests stores its depth (0.25): the next (0.4) fails.
            cases.push_back({"a passing fragment stores its depth: later 0.4 fragment fails (dpfail)",
                             {depth_draw(always, -0.5), depth_draw(later, -0.2)}});
            // DepthMask(FALSE): the depth test result is unchanged and the stencil still updates, but the
            // depth value is not stored, so the later fragment (0.4 vs 0.5) passes and INCRs.
            const StencilSide later_incr = stage_side(mgl::CompareFunc::Always, Op::Keep, Op::Zero, Op::Incr);
            cases.push_back({"DepthMask(FALSE): stencil updated by dppass, depth not stored (later 0.4 passes)",
                             {depth_draw(always, -0.5, true, false), depth_draw(later_incr, -0.2)}});
            Step disabled = depth_draw(always, -0.5);
            disabled.stencil_test = false;
            cases.push_back({"stencil test disabled: no stencil modification at all", {disabled}});

            int col = 0;
            for (Case& c : cases)
            {
                Experiment e;
                e.label = c.label;
                e.start = 0x10;
                e.steps = std::move(c.steps);
                run(lab_cell(col++, 6, 0), e);
            }
        }

        // ---- panel E: separate front/back state -------------------------------------------------------

        void panel_faces()
        {
            using Op = mgl::StencilOp;
            auto side = [](mgl::CompareFunc func, int ref, Op sfail, Op dpfail, Op dppass, unsigned write_mask = 0xFF) {
                StencilSide s;
                s.func = func;
                s.ref = ref;
                s.sfail = sfail;
                s.dpfail = dpfail;
                s.dppass = dppass;
                s.write_mask = write_mask;
                return s;
            };
            auto two_sided = [](const StencilSide& front, const StencilSide& back, Prim prim) {
                Step s;
                s.front = front;
                s.back = back;
                s.prim = prim;
                return s;
            };

            // front INCR / back DECR on dppass: the classic two-sided counting state (shadow volumes).
            const StencilSide f_incr = side(mgl::CompareFunc::Always, 0x11, Op::Keep, Op::Keep, Op::Incr);
            const StencilSide b_decr = side(mgl::CompareFunc::Always, 0x22, Op::Keep, Op::Keep, Op::Decr);
            const StencilSide f_incr_w = side(mgl::CompareFunc::Always, 0, Op::Keep, Op::Keep, Op::IncrWrap);
            const StencilSide b_decr_w = side(mgl::CompareFunc::Always, 0, Op::Keep, Op::Keep, Op::DecrWrap);
            // front EQUAL 100 / back EQUAL 7: with a stored 100 only the front passes.
            const StencilSide f_eq = side(mgl::CompareFunc::Equal, 100, Op::Zero, Op::Keep, Op::Invert);
            const StencilSide b_eq = side(mgl::CompareFunc::Equal, 7, Op::Zero, Op::Keep, Op::Invert);
            // REPLACE 0xFF through different write masks.
            const StencilSide f_mask = side(mgl::CompareFunc::Always, 0xFF, Op::Keep, Op::Keep, Op::Replace, 0x0F);
            const StencilSide b_mask = side(mgl::CompareFunc::Always, 0xFF, Op::Keep, Op::Keep, Op::Replace, 0xF0);

            struct Case
            {
                const char* label;
                unsigned start;
                bool front_face_cw;
                std::vector<Step> steps;
            };
            std::vector<Case> cases;
            cases.push_back({"front INCR / back DECR: a counter-clockwise triangle is front-facing: 100 -> 101", 100, false,
                             {two_sided(f_incr, b_decr, Prim::TriangleCcw)}});
            cases.push_back({"front INCR / back DECR: a clockwise triangle is back-facing: 100 -> 99", 100, false,
                             {two_sided(f_incr, b_decr, Prim::TriangleCw)}});
            cases.push_back({"glFrontFace(CW): the counter-clockwise triangle is now back-facing: 100 -> 99", 100, true,
                             {two_sided(f_incr, b_decr, Prim::TriangleCcw)}});
            cases.push_back({"glFrontFace(CW): the clockwise triangle is now front-facing: 100 -> 101", 100, true,
                             {two_sided(f_incr, b_decr, Prim::TriangleCw)}});
            cases.push_back({"a point uses the front state even with glFrontFace(CW): 100 -> 101", 100, true,
                             {two_sided(f_incr, b_decr, Prim::Point)}});
            cases.push_back({"front INCR_WRAP then back DECR_WRAP from 255: wraps to 0 and back to 255", 255, false,
                             {two_sided(f_incr_w, b_decr_w, Prim::TriangleCcw), two_sided(f_incr_w, b_decr_w, Prim::TriangleCw)}});
            cases.push_back({"front EQUAL 100 passes (INVERT): 100 -> 155", 100, false, {two_sided(f_eq, b_eq, Prim::TriangleCcw)}});
            cases.push_back({"back EQUAL 7 fails (sfail ZERO): 100 -> 0", 100, false, {two_sided(f_eq, b_eq, Prim::TriangleCw)}});
            cases.push_back({"front write mask 0x0F: REPLACE 0xFF on 0x00 -> 0x0F", 0x00, false,
                             {two_sided(f_mask, b_mask, Prim::TriangleCcw)}});
            cases.push_back({"back write mask 0xF0: REPLACE 0xFF on 0x00 -> 0xF0", 0x00, false,
                             {two_sided(f_mask, b_mask, Prim::TriangleCw)}});
            {
                // *Separate(GL_FRONT_AND_BACK) sets both faces: a clockwise triangle sees the same state.
                Step s = two_sided(side(mgl::CompareFunc::Equal, 100, Op::Zero, Op::Keep, Op::Incr), StencilSide{}, Prim::TriangleCw);
                s.back = s.front;
                s.calls = Calls::FrontAndBack;
                cases.push_back({"StencilFuncSeparate/OpSeparate(FRONT_AND_BACK) reach the back face: 100 -> 101", 100, false, {s}});
            }
            // Plain glStencilFunc/glStencilOp set front AND back: a clockwise (back-facing) triangle sees INVERT.
            cases.push_back({"glStencilOp (non-separate) also sets the back face: INVERT on a back-facing triangle: 100 -> 155", 100, false,
                             {two_sided(side(mgl::CompareFunc::Always, 0, Op::Keep, Op::Keep, Op::Invert),
                                        side(mgl::CompareFunc::Always, 0, Op::Keep, Op::Keep, Op::Invert), Prim::TriangleCw)}});
            cases.push_back({"glStencilMask (non-separate) also masks the back face: REPLACE 0xFF through 0x0F on 0xA0 -> 0xAF", 0xA0, false,
                             {two_sided(side(mgl::CompareFunc::Always, 0xFF, Op::Keep, Op::Keep, Op::Replace, 0x0F),
                                        side(mgl::CompareFunc::Always, 0xFF, Op::Keep, Op::Keep, Op::Replace, 0x0F), Prim::TriangleCw)}});

            int col = 0;
            for (Case& c : cases)
            {
                Experiment e;
                e.label = c.label;
                e.start = c.start;
                e.front_face_cw = c.front_face_cw;
                e.steps = std::move(c.steps);
                run(lab_cell(col++, 0, 102), e);
            }
        }

        // ---- the readout ----------------------------------------------------------------------------

        /// Turns every stencil value of the lab area into a color, using nothing but the stencil test.
        /// Value 0 needs no draw: such pixels keep their cell color, which equals the color for 0.
        void readout(const demo::Env& env)
        {
            reset_state();
            const Rect lab = top_rect(0, 0, 256, 116);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(lab.x, lab.y, env.width, lab.h);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);  // never modify
            use(fill_);
            for (int v = 1; v <= static_cast<int>(kStencilMax); ++v)
            {
                mgl::glStencilFunc(mgl::CompareFunc::Equal, v, 0xFF);  // passes only where stencil == v
                mgl::glUniform4f(fill_.color, static_cast<float>(v) / 255.0f, 0.4f, 0.4f, 1.0f);
                draw_cover(false, 0.0);
            }
            reset_state();
        }

        // ---- visual panel 1: stencil-masked window with an outline -----------------------------------

        /// Clears one visual panel and makes it the viewport (scissor stays enabled).
        void begin_panel(const Rect& r)
        {
            reset_state();
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glClearColor(kPanelBg[0], kPanelBg[1], kPanelBg[2], kPanelBg[3]);
            mgl::glClearStencil(0);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Stencil);
        }

        void use_stripes(const Color& a, const Color& b, float period)
        {
            use(stripes_);
            mgl::glUniform4f(stripes_.color_a, a[0], a[1], a[2], a[3]);
            mgl::glUniform4f(stripes_.color_b, b[0], b[1], b[2], b[3]);
            mgl::glUniform1f(stripes_.period, period);
        }

        void draw_circle(double radius)
        {
            use(circle_);
            mgl::glUniform3f(circle_.circle, 0.0f, 0.0f, static_cast<float>(radius));
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleFan, 0, kSegments + 2);
        }

        void portal_panel(const Rect& r)
        {
            begin_panel(r);

            // 1. The world outside the window, drawn everywhere (stencil test still disabled).
            use_stripes(kCoolA, kCoolB, 16.0f);
            draw_cover(false, 0.0);

            // 2. The window: a disc drawn into the stencil buffer only. Color writes are masked off,
            //    ALWAYS passes, and REPLACE writes the reference value 1 where the disc covers.
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 1, 0xFF);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Replace);
            use(circle_);
            set_color(kWhite);
            draw_circle(kWindowRadius);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            // 3. The world seen through the window: drawn over the whole panel, but only where stencil == 1.
            mgl::glStencilFunc(mgl::CompareFunc::Equal, 1, 0xFF);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            use_stripes(kWarmA, kWarmB, 8.0f);
            draw_cover(false, 0.0);

            // 4. The outline: a larger white disc drawn only where stencil != 1, i.e. a ring around the window.
            mgl::glStencilFunc(mgl::CompareFunc::Notequal, 1, 0xFF);
            use(circle_);
            set_color(kWhite);
            draw_circle(kOutlineRadius);
            mgl::glDisable(mgl::Capability::StencilTest);

            // CPU reference: distance from the panel center decides which of the three layers shows.
            struct Spot
            {
                double nx, ny;
            };
            const Spot spots[] = {{0.0, 0.0}, {0.3, 0.2}, {-0.25, -0.3}, {0.61, 0.0}, {0.0, -0.61},
                                  {-0.43, 0.43}, {0.8, 0.0}, {-0.7, -0.7}, {0.2, 0.85}};
            for (const Spot& s : spots)
            {
                const auto [px, py] = pixel_at(r, s.nx, s.ny);
                const auto [cx, cy] = pixel_center_ndc(r, px, py);  // the rasterizer samples pixel centers
                const double dist = std::hypot(cx, cy);
                demo::Rgba8 expected;
                const char* where;
                if (dist < kWindowRadius)
                {
                    expected = to_rgba8(stripe_color(px, py, kWarmA, kWarmB, 8));
                    where = "inside the window: the stencil-masked inner world";
                }
                else if (dist < kOutlineRadius)
                {
                    expected = to_rgba8(kWhite);
                    where = "on the outline ring (stencil != 1 around the window)";
                }
                else
                {
                    expected = to_rgba8(stripe_color(px, py, kCoolA, kCoolB, 16));
                    where = "outside the outline: the outer world";
                }
                pixel_probes_.push_back({px, py, expected,
                                         std::format("portal: radius {:.2f} - {}", dist, where)});
            }
        }

        // ---- visual panel 2: even-odd fill with INVERT -----------------------------------------------

        void star_panel(const Rect& r)
        {
            begin_panel(r);

            // Draw a pentagram as a triangle fan into the stencil buffer with INVERT: each fan triangle
            // flips the bit, so a pixel ends up 1 where an odd number of triangles cover it (the arms)
            // and back at 0 where an even number do (the inner pentagon).
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFF);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Invert);
            use(star_);
            mgl::glUniform3f(star_.star, 0.0f, 0.0f, static_cast<float>(kStarRadius));
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleFan, 0, 7);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            // Fill wherever the stencil bit is set. The mask 1 compares only bit 0.
            mgl::glStencilFunc(mgl::CompareFunc::Equal, 1, 1);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
            use_stripes(kGoldA, kGoldB, 8.0f);
            draw_cover(false, 0.0);
            mgl::glDisable(mgl::Capability::StencilTest);

            // CPU reference: count the fan triangles that contain each probe point.
            using Pt = std::array<double, 2>;
            std::array<Pt, 7> fan{};
            fan[0] = {0.0, 0.0};
            const int order[6] = {0, 2, 4, 1, 3, 0};
            for (int i = 0; i < 6; ++i)
            {
                const double a = std::numbers::pi / 2.0 + 2.0 * std::numbers::pi / 5.0 * order[i];
                fan[static_cast<std::size_t>(i) + 1] = {kStarRadius * std::cos(a), kStarRadius * std::sin(a)};
            }
            auto covered_by = [&](double x, double y) {
                int count = 0;
                for (std::size_t i = 1; i + 1 < fan.size(); ++i)
                {
                    auto edge = [&](const Pt& p, const Pt& q) { return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]); };
                    const double e0 = edge(fan[0], fan[i]), e1 = edge(fan[i], fan[i + 1]), e2 = edge(fan[i + 1], fan[0]);
                    if (e0 > 0 && e1 > 0 && e2 > 0) ++count;  // counter-clockwise triangles
                }
                return count;
            };

            // Probes are kept off the fan's internal edges (the rays from the center to the five tips),
            // where the edge rule alone decides which neighbouring triangle owns a pixel: arm probes sit
            // 5 degrees beside the axis of an arm, and the CPU model is evaluated at the pixel center.
            struct Spot
            {
                double radius, degrees;
                const char* what;
            };
            const Spot spots[] = {
                {0.06, 54.0, "center of the inner pentagon"}, {0.15, 126.0, "inner pentagon"},
                {0.5, 95.0, "arm 1"},                         {0.5, 167.0, "arm 2"},
                {0.5, 239.0, "arm 3"},                        {0.5, 311.0, "arm 4"},
                {0.5, 23.0, "arm 5"},                         {0.5, 126.0, "notch between arms"},
                {0.5, 198.0, "notch between arms"},           {0.5, 270.0, "notch between arms"},
                {0.9, 95.0, "outside the star"},
            };
            for (const Spot& s : spots)
            {
                const double nx = s.radius * std::cos(s.degrees * std::numbers::pi / 180.0);
                const double ny = s.radius * std::sin(s.degrees * std::numbers::pi / 180.0);
                const auto [px, py] = pixel_at(r, nx, ny);
                const auto [cx, cy] = pixel_center_ndc(r, px, py);
                const int count = covered_by(cx, cy);
                const bool filled = count % 2 == 1;
                pixel_probes_.push_back({px, py, to_rgba8(filled ? stripe_color(px, py, kGoldA, kGoldB, 8) : kPanelBg),
                                         std::format("star: {} is covered by {} fan triangle(s): {}", s.what, count,
                                                     filled ? "odd, filled" : "even, empty")});
            }
        }

        // ---- state queries ---------------------------------------------------------------------------

        void check_initial_state(demo::Check& check) const
        {
            // ES 3.0 §4.1.4 and §4.2.3 (initial values). The initial masks are "2^s - 1 where s is
            // at least the number of bits of the deepest stencil buffer": compare the low 8 bits only.
            check.expect(!initial_.stencil_test, "initial state: STENCIL_TEST is disabled (§4.1.4)");
            check.expect(initial_.stencil_bits >= 8, "the default framebuffer has an 8-bit stencil buffer (STENCIL_BITS >= 8)",
                         std::format("STENCIL_BITS = {}", initial_.stencil_bits));
            check.expect_eq(initial_.clear_value, 0, "initial state: STENCIL_CLEAR_VALUE is 0 (§4.2.3)");
            const std::pair<const char*, const QueriedSide*> sides[] = {{"front", &initial_.front}, {"back", &initial_.back}};
            for (const auto& [name, s] : sides)
            {
                check.expect_eq(s->func, static_cast<int>(mgl::CompareFunc::Always), std::format("initial state: {} STENCIL_FUNC is ALWAYS", name));
                check.expect_eq(s->ref, 0, std::format("initial state: {} STENCIL_REF is 0", name));
                check.expect_eq(s->value_mask & 0xFF, 0xFF, std::format("initial state: {} STENCIL_VALUE_MASK is all ones", name));
                check.expect_eq(s->write_mask & 0xFF, 0xFF, std::format("initial state: {} STENCIL_WRITEMASK is all ones", name));
                const int keep = static_cast<int>(mgl::StencilOp::Keep);
                check.expect(s->fail == keep && s->dpfail == keep && s->dppass == keep,
                             std::format("initial state: {} stencil operations are all KEEP", name));
            }
        }

        static void check_state_round_trips(demo::Check& check)
        {
            using P = mgl::GetParameter;
            check.expect(!mgl::glIsEnabled(mgl::Capability::StencilTest), "STENCIL_TEST is disabled after the experiments");
            mgl::glEnable(mgl::Capability::StencilTest);
            check.expect(mgl::glIsEnabled(mgl::Capability::StencilTest), "glEnable(STENCIL_TEST): glIsEnabled is TRUE");
            mgl::glDisable(mgl::Capability::StencilTest);
            check.expect(!mgl::glIsEnabled(mgl::Capability::StencilTest), "glDisable(STENCIL_TEST): glIsEnabled is FALSE");

            // glStencilFunc sets both faces: every function, with a reference and a value mask.
            for (const NamedFunc& fn : kFuncs)
            {
                mgl::glStencilFunc(fn.func, 0x2B, 0x7E);
                const QueriedSide f = read_side(false), b = read_side(true);
                check.expect(f.func == static_cast<int>(fn.func) && b.func == static_cast<int>(fn.func),
                             std::format("glStencilFunc({}): STENCIL_FUNC and STENCIL_BACK_FUNC", fn.name));
                check.expect(f.ref == 0x2B && b.ref == 0x2B && f.value_mask == 0x7E && b.value_mask == 0x7E,
                             std::format("glStencilFunc({}): ref 0x2B and value mask 0x7E on both faces", fn.name));
            }

            // The reference is clamped to [0, 2^s - 1] when queried (§4.1.4).
            mgl::glStencilFunc(mgl::CompareFunc::Always, 300, 0xFF);
            check.expect_eq(demo::get_integer(P::StencilRef), 255, "glStencilFunc(ref 300): STENCIL_REF clamps to 255");
            mgl::glStencilFunc(mgl::CompareFunc::Always, -5, 0xFF);
            check.expect_eq(demo::get_integer(P::StencilBackRef), 0, "glStencilFunc(ref -5): STENCIL_BACK_REF clamps to 0");

            // glStencilFuncSeparate changes one face only.
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFF);
            mgl::glStencilFuncSeparate(mgl::CullFace::Front, mgl::CompareFunc::Greater, 0x11, 0xF0);
            QueriedSide f = read_side(false), b = read_side(true);
            check.expect(f.func == static_cast<int>(mgl::CompareFunc::Greater) && f.ref == 0x11 && f.value_mask == 0xF0,
                         "glStencilFuncSeparate(FRONT): front func/ref/mask set");
            check.expect(b.func == static_cast<int>(mgl::CompareFunc::Always) && b.ref == 0 && (b.value_mask & 0xFF) == 0xFF,
                         "glStencilFuncSeparate(FRONT): back face unchanged");
            mgl::glStencilFuncSeparate(mgl::CullFace::Back, mgl::CompareFunc::Notequal, 0x22, 0x0F);
            f = read_side(false);
            b = read_side(true);
            check.expect(b.func == static_cast<int>(mgl::CompareFunc::Notequal) && b.ref == 0x22 && b.value_mask == 0x0F,
                         "glStencilFuncSeparate(BACK): back func/ref/mask set");
            check.expect(f.func == static_cast<int>(mgl::CompareFunc::Greater) && f.ref == 0x11, "glStencilFuncSeparate(BACK): front face unchanged");

            // glStencilOp: each of the eight operations in each of the three slots, both faces.
            for (const NamedOp& a : kOps)
            {
                mgl::glStencilOp(a.op, a.op, a.op);
                const QueriedSide qf = read_side(false), qb = read_side(true);
                const int v = static_cast<int>(a.op);
                check.expect(qf.fail == v && qf.dpfail == v && qf.dppass == v && qb.fail == v && qb.dpfail == v && qb.dppass == v,
                             std::format("glStencilOp({0}, {0}, {0}): all slots and both faces return {0}", a.name));
            }
            mgl::glStencilOp(mgl::StencilOp::Incr, mgl::StencilOp::Decr, mgl::StencilOp::Invert);
            f = read_side(false);
            check.expect(f.fail == static_cast<int>(mgl::StencilOp::Incr) && f.dpfail == static_cast<int>(mgl::StencilOp::Decr) &&
                             f.dppass == static_cast<int>(mgl::StencilOp::Invert),
                         "glStencilOp(INCR, DECR, INVERT): sfail, dpfail, dppass land in the right slots");
            mgl::glStencilOpSeparate(mgl::CullFace::Front, mgl::StencilOp::Zero, mgl::StencilOp::Replace, mgl::StencilOp::IncrWrap);
            mgl::glStencilOpSeparate(mgl::CullFace::Back, mgl::StencilOp::DecrWrap, mgl::StencilOp::Keep, mgl::StencilOp::Zero);
            f = read_side(false);
            b = read_side(true);
            check.expect(f.fail == static_cast<int>(mgl::StencilOp::Zero) && f.dpfail == static_cast<int>(mgl::StencilOp::Replace) &&
                             f.dppass == static_cast<int>(mgl::StencilOp::IncrWrap),
                         "glStencilOpSeparate(FRONT): front operations set");
            check.expect(b.fail == static_cast<int>(mgl::StencilOp::DecrWrap) && b.dpfail == static_cast<int>(mgl::StencilOp::Keep) &&
                             b.dppass == static_cast<int>(mgl::StencilOp::Zero),
                         "glStencilOpSeparate(BACK): back operations set independently");

            // Write masks.
            mgl::glStencilMask(0x5A);
            check.expect(demo::get_integer(P::StencilWritemask) == 0x5A && demo::get_integer(P::StencilBackWritemask) == 0x5A,
                         "glStencilMask(0x5A): both faces");
            mgl::glStencilMaskSeparate(mgl::CullFace::Front, 0x0F);
            mgl::glStencilMaskSeparate(mgl::CullFace::Back, 0xF0);
            check.expect(demo::get_integer(P::StencilWritemask) == 0x0F && demo::get_integer(P::StencilBackWritemask) == 0xF0,
                         "glStencilMaskSeparate(FRONT 0x0F, BACK 0xF0): independent masks");
            mgl::glStencilMaskSeparate(mgl::CullFace::FrontAndBack, 0x3C);
            check.expect(demo::get_integer(P::StencilWritemask) == 0x3C && demo::get_integer(P::StencilBackWritemask) == 0x3C,
                         "glStencilMaskSeparate(FRONT_AND_BACK, 0x3C): both faces");

            mgl::glClearStencil(0x5A);
            check.expect_eq(demo::get_integer(P::StencilClearValue), 0x5A, "glClearStencil(0x5A): STENCIL_CLEAR_VALUE");
        }

        // ---- members ------------------------------------------------------------------------------------

        std::vector<CellProbe> cell_probes_;
        std::vector<PixelProbe> pixel_probes_;
        InitialState initial_;
        Program fill_, stripes_, point_, circle_, star_;
        const Program* current_ = nullptr;
        float point_size_ = 1.0f;
        int width_ = kDesign, height_ = kDesign;
        Rect view_{0, 0, 1, 1};
        int probe_x_ = 0, probe_y_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-stencil-test",
        .title = "Stencil buffer test",
        .min_es = {3, 0},
        .summary = "All stencil functions and operations (clamp and wrap), write masks, separate front/back state and "
                   "sfail/dpfail/dppass; the stencil value is measured with 255 EQUAL-test draws and checked against a CPU model.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<StencilTest>(argc, argv, info);
}
