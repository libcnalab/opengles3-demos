// es20-clear-viewport-scissor
//
// How a frame is prepared and where its pixels may land: glClear and its clear values, the
// per-buffer write masks, the viewport transform, DepthRange and the scissor test.
//
// The image is a gallery of 16 panels of 64x64 pixels (4x4, panel A top-left, panel P bottom-right).
// Every panel is one experiment confined to its own rectangle by the scissor test. OpenGL ES cannot
// read the depth or stencil buffer back, so those buffers are observed *indirectly*: after the
// clears under test, a probe quad is drawn with the depth or the stencil test enabled and the color
// buffer shows which pixels passed. The panel therefore displays the content of the hidden buffer.
//
//   A  clear color            B  scissored clear         C  color mask vs clear     D  color mask vs draw
//   E  depth clear "ruler"    F  depth clear: scissor    G  glDepthRangef           H  viewport vs clear
//                                + depth mask
//   I  stencil "bit ruler"    J  stencil mask vs clear   K  one clear, 3 buffers    O  scissored draw
//   L  viewport mapping       M  four viewports          N  negative viewport origin P  clipping = viewport
//
// Validation: each panel's expected pixels are painted on the CPU from the rules of ES 3.0
// (sections 2.13.1, 4.1.2, 4.2.2, 4.2.3) and compared with a glReadPixels of the real framebuffer;
// the state queries and the INVALID_VALUE cases are checked separately.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Gallery layout and palette
    // ---------------------------------------------------------------------------------------------
    constexpr int kPanel = 64;  // panel edge in pixels
    constexpr int kColumns = 4;
    constexpr int kRows = 4;

    struct Rect
    {
        int x, y, w, h;  // lower-left corner and size, in window (or panel-local) pixels
    };
    struct Color
    {
        float r, g, b, a;
    };
    struct Vec2
    {
        float x, y;
    };

    enum class Panel : int
    {
        ClearColor, ScissoredClear, ColorMaskClear, ColorMaskDraw,
        DepthRuler, DepthClear, DepthRange, ViewportVsClear,
        StencilRuler, StencilClear, CombinedClear, ScissoredDraw,
        ViewportMapping, FourViewports, NegativeViewport, ClippedTriangle,
    };

    // Row-major from the top: panel 0 is the top-left cell, window y grows upwards.
    constexpr Rect panel_rect(Panel p)
    {
        const int i = static_cast<int>(p);
        return {(i % kColumns) * kPanel, (kRows - 1 - i / kColumns) * kPanel, kPanel, kPanel};
    }

    constexpr Rect shifted(Rect local, Rect origin)
    {
        return {origin.x + local.x, origin.y + local.y, local.w, local.h};
    }

    constexpr Color kSurface{0.02f, 0.02f, 0.03f, 1.0f};  // frame clear (visible only outside the gallery)
    constexpr Color kDark{0.05f, 0.06f, 0.09f, 1.0f};     // panel background

    // Panel A: the clear itself.
    constexpr Color kClearA{0.25f, 0.5f, 0.75f, 1.0f};
    // Panel B: a scissor box inside the panel (panel-local pixels).
    constexpr Color kNavy{0.12f, 0.14f, 0.35f, 1.0f};
    constexpr Color kGold{0.95f, 0.75f, 0.15f, 1.0f};
    constexpr Rect kBoxB{12, 8, 40, 30};
    // Panels C, D: base color with partial alpha so that every channel is distinguishable.
    constexpr Color kBase{0.2f, 0.4f, 0.6f, 0.8f};
    constexpr Color kQuadD{1.0f, 0.8f, 0.2f, 1.0f};
    // Panel E: clear depth per 16-pixel band; 16 columns probe z_w = (i + 0.5) / 16.
    constexpr std::array<float, 4> kRulerDepth = {0.125f, 0.375f, 0.625f, 0.875f};
    // Panel F: two scissor boxes cleared to depth 0.0, the second one with the depth mask off.
    constexpr Rect kBoxF1{8, 8, 24, 24};
    constexpr Rect kBoxF2{36, 36, 20, 20};
    constexpr Color kLitF{0.55f, 0.85f, 0.35f, 1.0f};
    // Panel G: depth range experiment; cells are 16 px wide and 32 px tall.
    constexpr float kClearDepthG = 0.6f;
    constexpr float kProbeZ = 0.3f;  // NDC z of the top row; the bottom row uses -kProbeZ
    constexpr std::array<std::array<float, 2>, 4> kRanges = {{{0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.5f}, {0.5f, 1.0f}}};
    constexpr Color kLitG{0.35f, 0.75f, 0.95f, 1.0f};
    // Panel H: a viewport smaller than the scissor box does not limit glClear.
    constexpr Color kSlate{0.20f, 0.28f, 0.30f, 1.0f};
    constexpr Color kOrange{0.95f, 0.50f, 0.10f, 1.0f};
    constexpr Rect kViewportH{10, 14, 30, 26};
    // Panels I, J: stencil values of the four bands (16 px each).
    constexpr std::array<int, 4> kStencilBands = {0x35, 0xC9, 0x1AB, 0x00};  // 0x1AB is masked to the bit planes
    constexpr int kStencilInit = 0xA5;
    // Panel K: one glClear for all three buffers.
    constexpr Color kPurple{0.35f, 0.20f, 0.45f, 1.0f};
    constexpr Color kYellow{1.0f, 0.9f, 0.2f, 1.0f};
    constexpr float kClearDepthK = 0.4f;
    constexpr int kClearStencilK = 0x42;
    // Panel O: scissor box for the draw.
    constexpr Color kSteel{0.25f, 0.40f, 0.60f, 1.0f};
    constexpr Color kAmber{0.95f, 0.70f, 0.20f, 1.0f};
    constexpr Rect kBoxO{10, 14, 36, 22};
    // Panel L: viewport inside the panel; the inner quad covers NDC [-0.5, 0.5]^2.
    constexpr Color kTeal{0.10f, 0.60f, 0.55f, 1.0f};
    constexpr Color kPink{0.95f, 0.40f, 0.60f, 1.0f};
    constexpr Rect kViewportL{8, 12, 40, 24};
    // Panel M: four 32x32 viewports, the same NDC triangle in each.
    constexpr std::array<Color, 4> kQuadrant = {{{0.90f, 0.30f, 0.30f, 1.0f},
                                                 {0.30f, 0.85f, 0.40f, 1.0f},
                                                 {0.35f, 0.50f, 0.95f, 1.0f},
                                                 {0.95f, 0.85f, 0.30f, 1.0f}}};
    constexpr std::array<Vec2, 3> kTriangleM = {{{-0.75f, -0.75f}, {0.75f, -0.75f}, {0.0f, 0.75f}}};
    // Panel N: viewport whose lower-left corner is outside the panel (and the window).
    constexpr Rect kViewportN{-32, -32, 96, 96};
    constexpr Color kMint{0.45f, 0.90f, 0.70f, 1.0f};
    constexpr Color kCoral{0.95f, 0.45f, 0.40f, 1.0f};
    // Panel P: viewport for the oversized triangle.
    constexpr Color kViolet{0.65f, 0.45f, 0.95f, 1.0f};
    constexpr Rect kViewportP{12, 8, 44, 40};

    Color ruler_color(int i)
    {
        const float f = static_cast<float>(i);
        return {0.25f + 0.045f * f, 0.90f - 0.04f * f, 0.55f, 1.0f};
    }

    constexpr GLboolean gl_bool(bool b) { return b ? GL_TRUE : GL_FALSE; }

    // ---------------------------------------------------------------------------------------------
    // CPU side: a snapshot of the framebuffer and a painted per-panel expectation
    // ---------------------------------------------------------------------------------------------
    class Image
    {
    public:
        Image(int w, int h) : width_(w), pixels_(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u)
        {
            // RGBA / UNSIGNED_BYTE is the combination every ES implementation must accept (ES 3.0 §4.3.2).
            mgl::glReadPixels(0, 0, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pixels_.data());
        }
        [[nodiscard]] demo::Rgba8 at(int x, int y) const
        {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x)) * 4u;
            return {pixels_[i], pixels_[i + 1], pixels_[i + 2], pixels_[i + 3]};
        }

    private:
        int width_;
        std::vector<std::uint8_t> pixels_;
    };

    /// Calls f(x, y) for every panel-local pixel of `r` that lies inside the panel.
    template <typename F>
    void for_each_pixel(Rect r, F&& f)
    {
        for (int y = std::max(r.y, 0); y < std::min(r.y + r.h, kPanel); ++y)
            for (int x = std::max(r.x, 0); x < std::min(r.x + r.w, kPanel); ++x) f(x, y);
    }

    /// Expected content of one panel; pixels with care == 0 are not compared.
    struct Model
    {
        std::vector<demo::Rgba8> color = std::vector<demo::Rgba8>(kPanel * kPanel);
        std::vector<char> care = std::vector<char>(kPanel * kPanel, 1);

        [[nodiscard]] static std::size_t index(int x, int y)
        {
            return static_cast<std::size_t>(y) * kPanel + static_cast<std::size_t>(x);
        }
        void fill(Rect r, demo::Rgba8 c)
        {
            for_each_pixel(r, [&](int x, int y) { color[index(x, y)] = c; });
        }
    };

    constexpr Rect kWholePanel{0, 0, kPanel, kPanel};

    /// Window state, as returned by the glGet* queries used in this demo.
    struct State
    {
        std::array<GLint, 4> viewport{};
        std::array<GLint, 4> scissor{};
        std::array<GLfloat, 4> clear_color{};
        GLfloat clear_depth = -1.0f;
        GLint clear_stencil = -1;
        std::array<GLboolean, 4> color_mask{};
        GLboolean depth_mask = GL_FALSE;
        GLint stencil_mask = 0;
        GLint stencil_back_mask = 0;
        std::array<GLfloat, 2> depth_range{};
        std::array<GLint, 2> max_viewport{};
        bool scissor_enabled = false;
        GLint depth_bits = 0, stencil_bits = 0, alpha_bits = 0;
    };

    State query_state()
    {
        State s;
        mgl::glGetIntegerv(mgl::GetParameter::Viewport, s.viewport.data());
        mgl::glGetIntegerv(mgl::GetParameter::ScissorBox, s.scissor.data());
        mgl::glGetFloatv(mgl::GetParameter::ColorClearValue, s.clear_color.data());
        mgl::glGetFloatv(mgl::GetParameter::DepthClearValue, &s.clear_depth);
        mgl::glGetIntegerv(mgl::GetParameter::StencilClearValue, &s.clear_stencil);
        mgl::glGetBooleanv(mgl::GetParameter::ColorWritemask, s.color_mask.data());
        mgl::glGetBooleanv(mgl::GetParameter::DepthWritemask, &s.depth_mask);
        mgl::glGetIntegerv(mgl::GetParameter::StencilWritemask, &s.stencil_mask);
        mgl::glGetIntegerv(mgl::GetParameter::StencilBackWritemask, &s.stencil_back_mask);
        mgl::glGetFloatv(mgl::GetParameter::DepthRange, s.depth_range.data());
        mgl::glGetIntegerv(mgl::GetParameter::MaxViewportDims, s.max_viewport.data());
        s.scissor_enabled = mgl::glIsEnabled(mgl::Capability::ScissorTest);
        mgl::glGetIntegerv(mgl::GetParameter::DepthBits, &s.depth_bits);
        mgl::glGetIntegerv(mgl::GetParameter::StencilBits, &s.stencil_bits);
        mgl::glGetIntegerv(mgl::GetParameter::AlphaBits, &s.alpha_bits);
        return s;
    }

    /// Signed distance (pixels) from a point to the triangle's nearest edge line: positive inside.
    /// Used to keep probe pixels away from rasterization-dependent edges.
    float triangle_margin(const std::array<Vec2, 3>& t, float px, float py)
    {
        float margin = 1e9f;
        for (int i = 0; i < 3; ++i)
        {
            const Vec2 a = t[static_cast<std::size_t>(i)], b = t[static_cast<std::size_t>((i + 1) % 3)];
            const float len = std::hypot(b.x - a.x, b.y - a.y);
            margin = std::min(margin, ((b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)) / len);
        }
        return margin;
    }

    class ClearViewportScissor final : public demo::Demo
    {
    public:
        // =========================================================================================
        // init / render
        // =========================================================================================
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/rect.vert", "shaders/solid.frag");
            u_vertex_ = mgl::glGetUniformLocation(program_, "u_vertex");
            u_z_ = mgl::glGetUniformLocation(program_, "u_z");
            u_color_ = mgl::glGetUniformLocation(program_, "u_color");
            if (u_vertex_.value < 0 || u_z_.value < 0 || u_color_.value < 0)
                throw demo::Failure("rect.vert / solid.frag: an expected uniform is not active");
            if (env.width < kPanel * kColumns || env.height < kPanel * kRows)
                throw demo::Failure(std::format("the gallery needs at least a {}x{} surface (got {}x{})",
                                                kPanel * kColumns, kPanel * kRows, env.width, env.height));

            // The very first GL state of the context: nothing has been set yet, so these are the
            // *initial* values the specification lists (checked in validate()).
            initial_ = query_state();
            if (initial_.depth_bits < 16 || initial_.stencil_bits < 8)
                throw demo::Skip(std::format("needs a default framebuffer with >= 16 depth and >= 8 stencil bits (got {} / {})",
                                             initial_.depth_bits, initial_.stencil_bits));
            has_alpha_ = initial_.alpha_bits > 0;
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            // Start every frame from the same state: the whole window (not only the gallery) is
            // cleared in one call to all three buffers.
            reset_state();
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kSurface.r, kSurface.g, kSurface.b, kSurface.a);
            mgl::glClearDepthf(1.0f);
            mgl::glClearStencil(0);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);

            mgl::glUseProgram(program_);
            draw_clear_color();
            draw_scissored_clear();
            draw_color_mask_clear();
            draw_color_mask_draw();
            draw_depth_ruler();
            draw_depth_clear();
            draw_depth_range();
            draw_viewport_vs_clear();
            draw_stencil_ruler();
            draw_stencil_clear();
            draw_combined_clear();
            draw_scissored_draw();
            draw_viewport_mapping();
            draw_four_viewports();
            draw_negative_viewport();
            draw_clipped_triangle();

            // Leave the state as the context started: full-window viewport and scissor box.
            reset_state();
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glScissor(0, 0, env.width, env.height);
        }

        // =========================================================================================
        // validate
        // =========================================================================================
        void validate(demo::Env& env, demo::Check& check) override
        {
            const Image image(env.width, env.height);

            check_clear_color(check, image);
            check_scissored_clear(check, image);
            check_color_mask(check, image);
            check_depth(check, image);
            check_stencil(check, image);
            check_combined_clear(check, image);
            check_scissored_draw(check, image);
            check_viewports(check, image);

            check_initial_state(env, check);
            check_state_round_trips(check);
            check_limits_and_errors(env, check);
            check_corner_cases(check);

            // The experiments above scribbled over the window; restore the canonical image.
            render(env, 0);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program_);
        }

    private:
        // =========================================================================================
        // GL helpers (every call is a plain metagl::gl* call)
        // =========================================================================================

        /// The initial state of the per-fragment operations and masks of ES 3.0 (Table 6.x defaults).
        static void reset_state()
        {
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::StencilTest);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            mgl::glDepthMask(GL_TRUE);
            mgl::glStencilMask(0xFFFFFFFFu);
            mgl::glDepthRangef(0.0f, 1.0f);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFFFFFFFu);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);
        }

        /// Confines viewport and scissor box to one panel and returns its window rectangle.
        static Rect begin_panel(Panel p)
        {
            const Rect r = panel_rect(p);
            reset_state();
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glScissor(r.x, r.y, r.w, r.h);
            mgl::glEnable(mgl::Capability::ScissorTest);
            return r;
        }

        static void set_scissor(Rect r) { mgl::glScissor(r.x, r.y, r.w, r.h); }

        static void clear_color(Color c)
        {
            mgl::glClearColor(c.r, c.g, c.b, c.a);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        /// Quad in normalized device coordinates, constant NDC depth `z`.
        void quad(float x0, float y0, float x1, float y1, float z, Color c) const
        {
            const std::array<float, 8> corners = {x0, y0, x1, y0, x0, y1, x1, y1};
            mgl::glUniform2fv(u_vertex_, 4, corners.data());
            mgl::glUniform1f(u_z_, z);
            mgl::glUniform4f(u_color_, c.r, c.g, c.b, c.a);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// Quad covering a panel-local pixel rectangle; the viewport must be the panel.
        void quad_px(Rect local, float z, Color c) const
        {
            auto ndc = [](int px) { return static_cast<float>(px) / 32.0f - 1.0f; };
            quad(ndc(local.x), ndc(local.y), ndc(local.x + local.w), ndc(local.y + local.h), z, c);
        }

        void triangle(const std::array<Vec2, 3>& t, float z, Color c) const
        {
            const std::array<float, 8> corners = {t[0].x, t[0].y, t[1].x, t[1].y, t[2].x, t[2].y, 0.0f, 0.0f};
            mgl::glUniform2fv(u_vertex_, 4, corners.data());
            mgl::glUniform1f(u_z_, z);
            mgl::glUniform4f(u_color_, c.r, c.g, c.b, c.a);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        /// Draws 8 columns, one per stencil bit (MSB on the left), lit where that bit is set.
        /// glStencilFunc(EQUAL, bit, bit) compares (ref & mask) with (stencil & mask), so it tests one bit.
        void observe_stencil_bits(Rect r) const
        {
            set_scissor(r);
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilOp(mgl::StencilOp::Keep, mgl::StencilOp::Keep, mgl::StencilOp::Keep);  // observe only
            for (int c = 0; c < 8; ++c)
            {
                const int bit = 7 - c;
                mgl::glStencilFunc(mgl::CompareFunc::Equal, 1 << bit, 1u << bit);
                quad(-1.0f + 0.25f * static_cast<float>(c), -1.0f, -1.0f + 0.25f * static_cast<float>(c + 1), 1.0f, 0.0f,
                     ruler_color(c));
            }
        }

        // =========================================================================================
        // The 16 panels: drawing
        // =========================================================================================

        // A - glClearColor + glClear(COLOR_BUFFER_BIT): the simplest clear.
        void draw_clear_color() const
        {
            begin_panel(Panel::ClearColor);
            clear_color(kClearA);
        }

        // B - the clear honours the scissor box (ES 3.0 §4.2.3: "the scissor test ... [is] applied").
        void draw_scissored_clear() const
        {
            const Rect r = begin_panel(Panel::ScissoredClear);
            clear_color(kNavy);
            set_scissor(shifted(kBoxB, r));
            clear_color(kGold);
        }

        // C - glColorMask limits which channels a clear writes (ES 3.0 §4.2.2): one channel per strip.
        void draw_color_mask_clear() const
        {
            const Rect r = begin_panel(Panel::ColorMaskClear);
            clear_color(kBase);
            mgl::glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            for (int k = 0; k < 4; ++k)
            {
                set_scissor({r.x + 16 * k, r.y, 16, kPanel});
                mgl::glColorMask(gl_bool(k == 0), gl_bool(k == 1), gl_bool(k == 2), gl_bool(k == 3));
                mgl::glClear(mgl::ClearBufferBit::Color);
            }
        }

        // D - the same mask applies to fragments written by a draw.
        void draw_color_mask_draw() const
        {
            begin_panel(Panel::ColorMaskDraw);
            clear_color(kBase);
            mgl::glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_TRUE);
            quad(-1.0f, -1.0f, 0.0f, 1.0f, 0.0f, kQuadD);  // left half writes R, B, A
            mgl::glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
            quad(0.0f, -1.0f, 1.0f, 1.0f, 0.0f, kQuadD);  // right half writes G only
        }

        // E - depth "ruler": four bands cleared to different depths; 16 columns at z_w = (i + .5) / 16
        //     pass GL_LESS exactly where z_w < the band's clear depth.
        void draw_depth_ruler() const
        {
            const Rect r = begin_panel(Panel::DepthRuler);
            clear_color(kDark);
            for (int b = 0; b < 4; ++b)
            {
                set_scissor({r.x, r.y + 16 * b, kPanel, 16});
                mgl::glClearDepthf(kRulerDepth[static_cast<std::size_t>(b)]);
                mgl::glClear(mgl::ClearBufferBit::Depth);
            }
            set_scissor(r);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_FALSE);  // observe only: the probes must not change the depth buffer
            for (int i = 0; i < 16; ++i)
            {
                const float x0 = -1.0f + 0.125f * static_cast<float>(i);
                quad(x0, -1.0f, x0 + 0.125f, 1.0f, ruler_ndc_z(i), ruler_color(i));
            }
        }

        // F - depth clear honours the scissor box and the depth mask: the box cleared with the mask off keeps 1.0.
        void draw_depth_clear() const
        {
            const Rect r = begin_panel(Panel::DepthClear);
            clear_color(kDark);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Depth);  // depth 1.0 over the whole panel
            mgl::glClearDepthf(0.0f);
            set_scissor(shifted(kBoxF1, r));
            mgl::glClear(mgl::ClearBufferBit::Depth);  // depth 0.0 inside box 1
            mgl::glDepthMask(GL_FALSE);
            set_scissor(shifted(kBoxF2, r));
            mgl::glClear(mgl::ClearBufferBit::Depth);  // masked: box 2 keeps depth 1.0
            // Observe with a quad at z_w = 0.5: it passes GL_LESS only where the depth buffer holds 1.0.
            set_scissor(r);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            quad(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, kLitF);
        }

        // G - glDepthRangef(n, f): z_w = (f-n)/2 * z_d + (n+f)/2 (ES 3.0 §2.13.1). The depth buffer is
        //     cleared to 0.6 (the clear value is a window-space depth, unaffected by the range); 4 ranges x 2 depths.
        void draw_depth_range() const
        {
            begin_panel(Panel::DepthRange);
            clear_color(kDark);
            mgl::glClearDepthf(kClearDepthG);
            mgl::glClear(mgl::ClearBufferBit::Depth);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_FALSE);
            for (int c = 0; c < 4; ++c)
            {
                mgl::glDepthRangef(kRanges[static_cast<std::size_t>(c)][0], kRanges[static_cast<std::size_t>(c)][1]);
                quad_px({16 * c, 32, 16, 32}, kProbeZ, kLitG);   // top row    : z_d = +0.3
                quad_px({16 * c, 0, 16, 32}, -kProbeZ, kLitG);   // bottom row : z_d = -0.3
            }
        }

        // H - the viewport does not limit glClear; it only decides where NDC [-1, 1] lands.
        void draw_viewport_vs_clear() const
        {
            const Rect r = begin_panel(Panel::ViewportVsClear);
            const Rect vp = shifted(kViewportH, r);
            mgl::glViewport(vp.x, vp.y, vp.w, vp.h);
            clear_color(kSlate);                           // clears the whole scissor box (= the panel)
            quad(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, kOrange);  // fills exactly the viewport
        }

        // I - stencil "bit ruler": four bands cleared with different values, read as 8 bit columns.
        void draw_stencil_ruler() const
        {
            const Rect r = begin_panel(Panel::StencilRuler);
            clear_color(kDark);
            for (int b = 0; b < 4; ++b)
            {
                set_scissor({r.x, r.y + 16 * b, kPanel, 16});
                mgl::glClearStencil(kStencilBands[static_cast<std::size_t>(b)]);
                mgl::glClear(mgl::ClearBufferBit::Stencil);
            }
            observe_stencil_bits(r);
        }

        // J - glStencilMask limits the bits a clear writes; the clear always uses the *front* mask.
        void draw_stencil_clear() const
        {
            const Rect r = begin_panel(Panel::StencilClear);
            clear_color(kDark);
            mgl::glClearStencil(kStencilInit);
            mgl::glClear(mgl::ClearBufferBit::Stencil);  // 0xA5 everywhere

            set_scissor({r.x, r.y + 0, kPanel, 16});
            mgl::glStencilMask(0x0F);
            mgl::glClearStencil(0xFF);
            mgl::glClear(mgl::ClearBufferBit::Stencil);  // low nibble only -> 0xAF

            set_scissor({r.x, r.y + 16, kPanel, 16});
            mgl::glStencilMask(0xF0);
            mgl::glClearStencil(0x00);
            mgl::glClear(mgl::ClearBufferBit::Stencil);  // high nibble only -> 0x05

            set_scissor({r.x, r.y + 32, kPanel, 16});
            mgl::glStencilMask(0x00);
            mgl::glClearStencil(0xFF);
            mgl::glClear(mgl::ClearBufferBit::Stencil);  // nothing writable -> 0xA5

            set_scissor({r.x, r.y + 48, kPanel, 16});
            mgl::glStencilMaskSeparate(mgl::CullFace::Front, 0xFF);
            mgl::glStencilMaskSeparate(mgl::CullFace::Back, 0x00);
            mgl::glClearStencil(0x3C);
            mgl::glClear(mgl::ClearBufferBit::Stencil);  // front mask rules the clear -> 0x3C

            mgl::glStencilMask(0xFFFFFFFFu);
            observe_stencil_bits(r);
        }

        // K - one glClear call for color + depth + stencil; each buffer is then observed in its own strip.
        void draw_combined_clear() const
        {
            begin_panel(Panel::CombinedClear);
            mgl::glClearColor(kPurple.r, kPurple.g, kPurple.b, kPurple.a);
            mgl::glClearDepthf(kClearDepthK);
            mgl::glClearStencil(kClearStencilK);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth | mgl::ClearBufferBit::Stencil);

            // Strip 1 (x 20..43): depth probes. z_w = 0.2 < 0.4 passes (top), z_w = 0.6 fails (bottom).
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_FALSE);
            quad_px({20, 32, 24, 32}, -0.6f, kYellow);
            quad_px({20, 0, 24, 32}, 0.2f, kYellow);
            mgl::glDisable(mgl::Capability::DepthTest);

            // Strip 2 (x 44..63): stencil probes. EQUAL 0x42 passes (top), EQUAL 0x41 fails (bottom).
            mgl::glEnable(mgl::Capability::StencilTest);
            mgl::glStencilFunc(mgl::CompareFunc::Equal, kClearStencilK, 0xFF);
            quad_px({44, 32, 20, 32}, 0.0f, kYellow);
            mgl::glStencilFunc(mgl::CompareFunc::Equal, kClearStencilK - 1, 0xFF);
            quad_px({44, 0, 20, 32}, 0.0f, kYellow);
        }

        // O - the scissor box limits draws only while GL_SCISSOR_TEST is enabled.
        void draw_scissored_draw() const
        {
            const Rect r = begin_panel(Panel::ScissoredDraw);
            clear_color(kDark);
            set_scissor({r.x + 30, r.y + 4, 8, 8});          // a box is set ...
            mgl::glDisable(mgl::Capability::ScissorTest);     // ... but the test is off: it is ignored
            quad(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, kSteel);     // covers the whole viewport (= panel)
            set_scissor(shifted(kBoxO, r));
            mgl::glEnable(mgl::Capability::ScissorTest);
            quad(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, kAmber);     // same quad, now cut to the box
        }

        // L - viewport mapping x_w = (w/2) x_d + (x + w/2): full NDC square and the NDC square [-0.5, 0.5].
        void draw_viewport_mapping() const
        {
            const Rect r = begin_panel(Panel::ViewportMapping);
            clear_color(kDark);
            const Rect vp = shifted(kViewportL, r);
            mgl::glViewport(vp.x, vp.y, vp.w, vp.h);
            quad(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, kTeal);
            quad(-0.5f, -0.5f, 0.5f, 0.5f, 0.0f, kPink);
        }

        // M - the same NDC triangle drawn through four different viewports.
        void draw_four_viewports() const
        {
            const Rect r = begin_panel(Panel::FourViewports);
            clear_color(kDark);
            for (int q = 0; q < 4; ++q)
            {
                mgl::glViewport(r.x + 32 * (q % 2), r.y + 32 * (q / 2), 32, 32);
                triangle(kTriangleM, 0.0f, kQuadrant[static_cast<std::size_t>(q)]);
            }
        }

        // N - a viewport may start left of / below the window; fragments outside the window are simply not produced.
        void draw_negative_viewport() const
        {
            const Rect r = begin_panel(Panel::NegativeViewport);
            clear_color(kDark);
            const Rect vp = shifted(kViewportN, r);
            mgl::glViewport(vp.x, vp.y, vp.w, vp.h);
            quad(-0.25f, -0.25f, 0.25f, 0.25f, 0.0f, kMint);
            quad(0.5f, 0.5f, 1.0f, 1.0f, 0.0f, kCoral);
        }

        // P - an oversized triangle is clipped to the clip volume, which the viewport transform maps
        //     onto the viewport rectangle: it fills exactly the viewport, no more.
        void draw_clipped_triangle() const
        {
            const Rect r = begin_panel(Panel::ClippedTriangle);
            clear_color(kDark);
            const Rect vp = shifted(kViewportP, r);
            mgl::glViewport(vp.x, vp.y, vp.w, vp.h);
            triangle({{{-3.0f, -3.0f}, {3.0f, -3.0f}, {0.0f, 5.0f}}}, 0.0f, kViolet);
        }

        static float ruler_ndc_z(int i) { return (static_cast<float>(i) + 0.5f) / 8.0f - 1.0f; }

        // =========================================================================================
        // The 16 panels: expectations and comparison
        // =========================================================================================

        /// Expected 8-bit pixel for a float color written to a normalized RGBA8 buffer (ES 3.0 §2.1.2, eq. 2.3).
        [[nodiscard]] demo::Rgba8 rgba8(Color c) const
        {
            auto q = [](float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
            return {q(c.r), q(c.g), q(c.b), has_alpha_ ? q(c.a) : std::uint8_t{255}};
        }

        /// New pixel after writing `src` through a color mask.
        [[nodiscard]] demo::Rgba8 masked(demo::Rgba8 dst, demo::Rgba8 src, std::array<bool, 4> mask) const
        {
            demo::Rgba8 out = dst;
            for (std::size_t i = 0; i < 4; ++i)
                if (mask[i]) out[i] = src[i];
            if (!has_alpha_) out[3] = 255;
            return out;
        }

        /// Compares the cared-about pixels of `region` (panel-local) with the model; one check.
        void compare(demo::Check& check, const Image& image, Panel p, const Model& m, Rect region, std::string_view what,
                     int tolerance = 1) const
        {
            const Rect o = panel_rect(p);
            int compared = 0, bad = 0;
            std::string first_bad;
            for_each_pixel(region, [&](int x, int y) {
                if (!m.care[Model::index(x, y)]) return;
                ++compared;
                const demo::Rgba8 got = image.at(o.x + x, o.y + y), want = m.color[Model::index(x, y)];
                bool ok = true;
                for (std::size_t i = 0; i < 4; ++i)
                    if (std::abs(int{got[i]} - int{want[i]}) > tolerance) ok = false;
                if (!ok && bad++ == 0)
                    first_bad = std::format("first at panel ({},{}): got {} expected {}", x, y, demo::format_rgba(got),
                                            demo::format_rgba(want));
            });
            check.expect(compared > 0 && bad == 0, what,
                         bad == 0 ? std::format("{} pixels compared", compared)
                                  : std::format("{} of {} pixels differ; {}", bad, compared, first_bad));
        }

        /// True when every listed panel-local pixel equals `want` (within `tolerance`).
        bool all_equal(const Image& image, Panel p, std::initializer_list<std::array<int, 2>> pixels, demo::Rgba8 want,
                       int tolerance = 1) const
        {
            const Rect o = panel_rect(p);
            for (const auto& px : pixels)
            {
                const demo::Rgba8 got = image.at(o.x + px[0], o.y + px[1]);
                for (std::size_t i = 0; i < 4; ++i)
                    if (std::abs(int{got[i]} - int{want[i]}) > tolerance) return false;
            }
            return true;
        }

        Model background(Color c) const
        {
            Model m;
            m.fill(kWholePanel, rgba8(c));
            return m;
        }

        // ---- A ---------------------------------------------------------------------------------
        void check_clear_color(demo::Check& check, const Image& image) const
        {
            compare(check, image, Panel::ClearColor, background(kClearA), kWholePanel,
                    "A: glClearColor(0.25,0.5,0.75,1) + glClear(COLOR) fills the whole scissor box (round(c*255) +-1)");
        }

        // ---- B ---------------------------------------------------------------------------------
        void check_scissored_clear(demo::Check& check, const Image& image) const
        {
            Model m = background(kNavy);
            m.fill(kBoxB, rgba8(kGold));
            compare(check, image, Panel::ScissoredClear, m, kWholePanel,
                    "B: a clear under glScissor(12,8,40,30) changes exactly the 40x30 box and nothing around it");
            // ES 3.0 §4.1.2: passes iff left <= x < left + width and bottom <= y < bottom + height.
            check.expect(all_equal(image, Panel::ScissoredClear, {{12, 8}, {51, 8}, {12, 37}, {51, 37}}, rgba8(kGold)),
                         "B: the four corner pixels of the box (left, left+w-1, bottom, bottom+h-1) are inside");
            check.expect(all_equal(image, Panel::ScissoredClear,
                                   {{11, 8}, {52, 8}, {12, 7}, {12, 38}, {51, 7}, {51, 38}, {11, 37}, {52, 37}}, rgba8(kNavy)),
                         "B: the pixels at left-1, left+w, bottom-1 and bottom+h are outside (half-open interval)");
        }

        // ---- C, D ------------------------------------------------------------------------------
        void check_color_mask(demo::Check& check, const Image& image) const
        {
            const demo::Rgba8 base = rgba8(kBase), white = rgba8({1, 1, 1, 1});
            const char* names[4] = {"R", "G", "B", "A"};
            for (int k = 0; k < 4; ++k)
            {
                Model m = background(kBase);
                std::array<bool, 4> mask{k == 0, k == 1, k == 2, k == 3};
                m.fill({16 * k, 0, 16, kPanel}, masked(base, white, mask));
                compare(check, image, Panel::ColorMaskClear, m, {16 * k, 0, 16, kPanel},
                        std::format("C: glColorMask enabling only {} -> a white clear writes only that channel of the strip", names[k]));
            }

            const demo::Rgba8 quad_color = rgba8(kQuadD);
            Model d = background(kBase);
            d.fill({0, 0, 32, kPanel}, masked(base, quad_color, {true, false, true, true}));
            d.fill({32, 0, 32, kPanel}, masked(base, quad_color, {false, true, false, false}));
            compare(check, image, Panel::ColorMaskDraw, d, {0, 0, 32, kPanel},
                    "D: a draw with glColorMask(R,B,A) keeps the destination green channel");
            compare(check, image, Panel::ColorMaskDraw, d, {32, 0, 32, kPanel},
                    "D: a draw with glColorMask(G only) writes only the green channel");
        }

        // ---- E, F, G ---------------------------------------------------------------------------
        void check_depth(demo::Check& check, const Image& image) const
        {
            for (int b = 0; b < 4; ++b)
            {
                const float depth = kRulerDepth[static_cast<std::size_t>(b)];
                Model m = background(kDark);
                int lit = 0;
                for (int i = 0; i < 16; ++i)
                {
                    const float zw = (static_cast<float>(i) + 0.5f) / 16.0f;
                    if (zw < depth)
                    {
                        m.fill({4 * i, 16 * b, 4, 16}, rgba8(ruler_color(i)));
                        ++lit;
                    }
                }
                compare(check, image, Panel::DepthRuler, m, {0, 16 * b, kPanel, 16},
                        std::format("E: glClearDepthf({}) + glClear(DEPTH): exactly the {} probe columns with z_w < {} pass GL_LESS",
                                    depth, lit, depth));
            }

            Model f = background(kDark);
            f.fill(kWholePanel, rgba8(kLitF));  // depth 1.0 -> the z_w = 0.5 probe passes
            f.fill(kBoxF1, rgba8(kDark));       // cleared to 0.0 -> fails
            compare(check, image, Panel::DepthClear, f, kWholePanel,
                    "F: depth cleared to 0.0 inside the scissor box only; the same clear with glDepthMask(FALSE) changed nothing");
            check.expect(all_equal(image, Panel::DepthClear, {{8, 8}, {31, 31}, {8, 31}, {31, 8}}, rgba8(kDark)) &&
                             all_equal(image, Panel::DepthClear, {{7, 8}, {32, 8}, {8, 32}, {8, 7}}, rgba8(kLitF)),
                         "F: scissored depth clear is exact at the box edges");
            check.expect(all_equal(image, Panel::DepthClear, {{36, 36}, {55, 55}, {45, 45}}, rgba8(kLitF)),
                         "F: depth clear with glDepthMask(FALSE) leaves the old depth 1.0 (probe passes inside box 2)");

            // G: z_w = (f-n)/2 * z_d + (n+f)/2, compared with the clear depth 0.6.
            Model g = background(kDark);
            for (int c = 0; c < 4; ++c)
            {
                const double n = static_cast<double>(kRanges[static_cast<std::size_t>(c)][0]);
                const double fr = static_cast<double>(kRanges[static_cast<std::size_t>(c)][1]);
                for (int row = 0; row < 2; ++row)
                {
                    const double zd = static_cast<double>(row == 0 ? kProbeZ : -kProbeZ);
                    const double zw = (fr - n) / 2.0 * zd + (n + fr) / 2.0;
                    if (zw < static_cast<double>(kClearDepthG))
                        g.fill({16 * c, row == 0 ? 32 : 0, 16, 32}, rgba8(kLitG));
                }
            }
            const char* range_names[4] = {"[0,1]", "[1,0] (reversed)", "[0,0.5]", "[0.5,1]"};
            for (int c = 0; c < 4; ++c)
                compare(check, image, Panel::DepthRange, g, {16 * c, 0, 16, kPanel},
                        std::format("G: glDepthRangef{} maps z_d = +-0.3 to the z_w that decides GL_LESS against 0.6", range_names[c]));
        }

        // ---- H ---------------------------------------------------------------------------------
        void check_viewports(demo::Check& check, const Image& image) const
        {
            Model h = background(kSlate);
            h.fill(kViewportH, rgba8(kOrange));
            compare(check, image, Panel::ViewportVsClear, h, kWholePanel,
                    "H: glClear ignores the (small) viewport; the full-NDC quad fills exactly the viewport");

            // L: x_w = ox + (w/2) x_d with ox = x + w/2.
            Model l = background(kDark);
            l.fill(kViewportL, rgba8(kTeal));
            l.fill({18, 18, 20, 12}, rgba8(kPink));  // x: 28 +- 20*0.5 = [18,38), y: 24 +- 12*0.5 = [18,30)
            compare(check, image, Panel::ViewportMapping, l, kWholePanel,
                    "L: viewport (8,12,40,24) maps NDC [-1,1]^2 to x 8..47, y 12..35 and NDC [-.5,.5]^2 to x 18..37, y 18..29");

            // M: the triangle in every 32x32 viewport. Pixels within 1 px of an edge are not compared.
            Model m = background(kDark);
            int skipped = 0;
            for (int q = 0; q < 4; ++q)
            {
                const float ox = static_cast<float>(32 * (q % 2)) + 16.0f, oy = static_cast<float>(32 * (q / 2)) + 16.0f;
                std::array<Vec2, 3> t{};
                for (std::size_t i = 0; i < 3; ++i) t[i] = {ox + 16.0f * kTriangleM[i].x, oy + 16.0f * kTriangleM[i].y};
                for_each_pixel({32 * (q % 2), 32 * (q / 2), 32, 32}, [&](int x, int y) {
                    const float margin = triangle_margin(t, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                    if (margin >= 1.0f) m.color[Model::index(x, y)] = rgba8(kQuadrant[static_cast<std::size_t>(q)]);
                    else if (margin > -1.0f) { m.care[Model::index(x, y)] = 0; ++skipped; }
                });
            }
            compare(check, image, Panel::FourViewports, m, kWholePanel,
                    std::format("M: one NDC triangle through four viewports lands in each 32x32 quadrant ({} edge pixels not compared)", skipped));

            // N: viewport (-32,-32,96,96): center at (16,16), half size 48.
            Model n = background(kDark);
            n.fill({4, 4, 24, 24}, rgba8(kMint));    // NDC +-0.25 -> 16 +- 12
            n.fill({40, 40, 24, 24}, rgba8(kCoral)); // NDC 0.5..1 -> 16 + 24 .. 16 + 48
            compare(check, image, Panel::NegativeViewport, n, kWholePanel,
                    "N: viewport (-32,-32,96,96) is legal; NDC (0,0) lands at pixel (16,16) of the panel");

            // P: clipping to the clip volume = the viewport rectangle.
            Model p = background(kDark);
            p.fill(kViewportP, rgba8(kViolet));
            compare(check, image, Panel::ClippedTriangle, p, kWholePanel,
                    "P: a triangle far larger than NDC [-1,1]^2 is clipped to exactly the viewport rectangle");
        }

        // ---- I, J ------------------------------------------------------------------------------
        void check_stencil(demo::Check& check, const Image& image) const
        {
            const int mask = (1 << std::min(initial_.stencil_bits, 8)) - 1;  // the bit planes visible in 8 columns
            auto ruler = [&](int b, int value) {
                Model m = background(kDark);
                for (int c = 0; c < 8; ++c)
                    if (((value & mask) >> (7 - c)) & 1) m.fill({8 * c, 16 * b, 8, 16}, rgba8(ruler_color(c)));
                return m;
            };
            for (int b = 0; b < 4; ++b)
            {
                const int v = kStencilBands[static_cast<std::size_t>(b)];
                compare(check, image, Panel::StencilRuler, ruler(b, v), {0, 16 * b, kPanel, 16},
                        std::format("I: glClearStencil(0x{:X}) + glClear(STENCIL) -> bit columns show 0x{:02X}{}", v, v & mask,
                                    v > 0xFF ? " (the value is masked to the stencil bit planes)" : ""));
            }

            const std::array<int, 4> expected = {0xAF, 0x05, 0xA5, 0x3C};
            const char* what[4] = {"glStencilMask(0x0F) + clear 0xFF over 0xA5 -> 0xAF (only the low nibble is written)",
                                   "glStencilMask(0xF0) + clear 0x00 over 0xA5 -> 0x05 (only the high nibble is written)",
                                   "glStencilMask(0x00) + clear 0xFF over 0xA5 -> 0xA5 (nothing is written)",
                                   "front mask 0xFF / back mask 0x00 + clear 0x3C -> 0x3C (a clear uses the front mask)"};
            for (int b = 0; b < 4; ++b)
                compare(check, image, Panel::StencilClear, ruler(b, expected[static_cast<std::size_t>(b)]), {0, 16 * b, kPanel, 16},
                        std::format("J: {}", what[b]));
        }

        // ---- K ---------------------------------------------------------------------------------
        void check_combined_clear(demo::Check& check, const Image& image) const
        {
            Model m = background(kPurple);
            m.fill({20, 32, 24, 32}, rgba8(kYellow));  // depth probe 0.2 < 0.4 passed
            m.fill({44, 32, 20, 32}, rgba8(kYellow));  // stencil EQUAL 0x42 passed
            compare(check, image, Panel::CombinedClear, m, kWholePanel,
                    "K: one glClear(COLOR|DEPTH|STENCIL): color shows the clear color, depth 0.4 passes z_w=0.2 and rejects 0.6, "
                    "stencil 0x42 passes EQUAL 0x42 and rejects 0x41");
        }

        // ---- O ---------------------------------------------------------------------------------
        void check_scissored_draw(demo::Check& check, const Image& image) const
        {
            Model m = background(kSteel);
            m.fill(kBoxO, rgba8(kAmber));
            compare(check, image, Panel::ScissoredDraw, m, kWholePanel,
                    "O: with GL_SCISSOR_TEST disabled the (tiny) scissor box is ignored; enabled, a full-panel quad is cut to the box");
        }

        // =========================================================================================
        // State queries, limits, errors
        // =========================================================================================
        void check_initial_state(const demo::Env& env, demo::Check& check) const
        {
            const State& s = initial_;
            check.expect(s.viewport == std::array<GLint, 4>{0, 0, env.width, env.height},
                         "initial GL_VIEWPORT = (0, 0, surface width, surface height) (ES 3.0 §2.13.1)",
                         std::format("viewport = ({}, {}, {}, {}), surface {}x{}", s.viewport[0], s.viewport[1], s.viewport[2],
                                     s.viewport[3], env.width, env.height));
            check.expect(s.scissor == std::array<GLint, 4>{0, 0, env.width, env.height},
                         "initial GL_SCISSOR_BOX = (0, 0, surface width, surface height) (ES 3.0 §4.1.2)",
                         std::format("box = ({}, {}, {}, {})", s.scissor[0], s.scissor[1], s.scissor[2], s.scissor[3]));
            check.expect(!s.scissor_enabled, "initially GL_SCISSOR_TEST is disabled (ES 3.0 §4.1.2)");
            check.expect(s.clear_color == std::array<GLfloat, 4>{0.0f, 0.0f, 0.0f, 0.0f},
                         "initial GL_COLOR_CLEAR_VALUE = (0,0,0,0) (ES 3.0 §4.2.3)");
            check.expect_near(s.clear_depth, 1.0, 0.0, "initial GL_DEPTH_CLEAR_VALUE = 1.0 (ES 3.0 §4.2.3)");
            check.expect_eq(s.clear_stencil, 0, "initial GL_STENCIL_CLEAR_VALUE = 0 (ES 3.0 §4.2.3)");
            check.expect(std::all_of(s.color_mask.begin(), s.color_mask.end(), [](GLboolean b) { return b == GL_TRUE; }),
                         "initial GL_COLOR_WRITEMASK = (TRUE,TRUE,TRUE,TRUE) (ES 3.0 §4.2.2)");
            check.expect(s.depth_mask == GL_TRUE, "initial GL_DEPTH_WRITEMASK = TRUE (ES 3.0 §4.2.2)");
            const int plane_mask = (1 << s.stencil_bits) - 1;
            check.expect((s.stencil_mask & plane_mask) == plane_mask && (s.stencil_back_mask & plane_mask) == plane_mask,
                         "initial front and back GL_STENCIL_WRITEMASK have all stencil bit planes set (\"1's\", ES 3.0 §4.2.2)",
                         std::format("front=0x{:X} back=0x{:X} ({} stencil bits)", static_cast<unsigned>(s.stencil_mask),
                                     static_cast<unsigned>(s.stencil_back_mask), s.stencil_bits));
            check.expect(s.depth_range[0] == 0.0f && s.depth_range[1] == 1.0f, "initial GL_DEPTH_RANGE = (0, 1) (ES 3.0 §2.13.1)");
        }

        void check_state_round_trips(demo::Check& check) const
        {
            // Viewport and scissor box: set, read back, and the scissor enable flag.
            mgl::glViewport(3, 5, 7, 9);
            mgl::glScissor(1, 2, 3, 4);
            mgl::glEnable(mgl::Capability::ScissorTest);
            State s = query_state();
            check.expect(s.viewport == std::array<GLint, 4>{3, 5, 7, 9}, "glViewport(3,5,7,9) is returned by GL_VIEWPORT");
            check.expect(s.scissor == std::array<GLint, 4>{1, 2, 3, 4}, "glScissor(1,2,3,4) is returned by GL_SCISSOR_BOX");
            check.expect(s.scissor_enabled, "glEnable(GL_SCISSOR_TEST): glIsEnabled = TRUE");
            mgl::glDisable(mgl::Capability::ScissorTest);
            check.expect(!mgl::glIsEnabled(mgl::Capability::ScissorTest), "glDisable(GL_SCISSOR_TEST): glIsEnabled = FALSE");

            // Clear values.
            mgl::glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
            mgl::glClearDepthf(0.375f);
            mgl::glClearStencil(0x12);
            s = query_state();
            check.expect(s.clear_color == std::array<GLfloat, 4>{0.25f, 0.5f, 0.75f, 1.0f},
                         "glClearColor(.25,.5,.75,1) is returned by GL_COLOR_CLEAR_VALUE");
            check.expect_near(s.clear_depth, 0.375, 1e-6, "glClearDepthf(0.375) is returned by GL_DEPTH_CLEAR_VALUE");
            check.expect_eq(s.clear_stencil, 0x12, "glClearStencil(0x12) is returned by GL_STENCIL_CLEAR_VALUE");
            // ES 3.0 §4.2.3: "d is clamped to the range [0, 1]" when specified.
            mgl::glClearDepthf(2.0f);
            check.expect_near(query_state().clear_depth, 1.0, 0.0, "glClearDepthf(2.0) is clamped to 1.0 (ES 3.0 §4.2.3)");
            mgl::glClearDepthf(-3.0f);
            check.expect_near(query_state().clear_depth, 0.0, 0.0, "glClearDepthf(-3.0) is clamped to 0.0 (ES 3.0 §4.2.3)");

            // Write masks.
            mgl::glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
            mgl::glDepthMask(GL_FALSE);
            mgl::glStencilMask(0x0F);
            s = query_state();
            check.expect(s.color_mask == std::array<GLboolean, 4>{GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE},
                         "glColorMask(T,F,T,F) is returned by GL_COLOR_WRITEMASK");
            check.expect(s.depth_mask == GL_FALSE, "glDepthMask(FALSE) is returned by GL_DEPTH_WRITEMASK");
            check.expect((s.stencil_mask & 0xFF) == 0x0F && (s.stencil_back_mask & 0xFF) == 0x0F,
                         "glStencilMask(0x0F) sets both the front and the back write mask (ES 3.0 §4.2.2)");
            mgl::glStencilMaskSeparate(mgl::CullFace::Back, 0xC3);
            s = query_state();
            check.expect((s.stencil_mask & 0xFF) == 0x0F && (s.stencil_back_mask & 0xFF) == 0xC3,
                         "glStencilMaskSeparate(BACK, 0xC3) changes only the back write mask");

            // Depth range.
            mgl::glDepthRangef(0.25f, 0.75f);
            s = query_state();
            check.expect(s.depth_range[0] == 0.25f && s.depth_range[1] == 0.75f, "glDepthRangef(0.25,0.75) is returned by GL_DEPTH_RANGE");
            mgl::glDepthRangef(-0.5f, 1.5f);
            s = query_state();
            check.expect(s.depth_range[0] == 0.0f && s.depth_range[1] == 1.0f,
                         "glDepthRangef(-0.5,1.5): n and f are clamped to [0,1] (ES 3.0 §2.13.1)");
            reset_state();
        }

        void check_limits_and_errors(const demo::Env& env, demo::Check& check) const
        {
            check.expect_no_gl_error("no GL error so far");
            const State& s = initial_;
            check.expect(s.max_viewport[0] >= env.width && s.max_viewport[1] >= env.height,
                         "GL_MAX_VIEWPORT_DIMS covers the surface (ES 3.0 §2.13.1)",
                         std::format("max {}x{}, surface {}x{}", s.max_viewport[0], s.max_viewport[1], env.width, env.height));

            // "Viewport width and height are clamped to implementation-dependent maximums when specified."
            if (s.max_viewport[0] < (1 << 28) && s.max_viewport[1] < (1 << 28))
            {
                mgl::glViewport(0, 0, s.max_viewport[0] + 1000, s.max_viewport[1] + 1000);
                const State t = query_state();
                check.expect(t.viewport[2] == s.max_viewport[0] && t.viewport[3] == s.max_viewport[1],
                             "glViewport wider/taller than GL_MAX_VIEWPORT_DIMS is clamped to it (ES 3.0 §2.13.1)",
                             std::format("stored {}x{}, max {}x{}", t.viewport[2], t.viewport[3], s.max_viewport[0], s.max_viewport[1]));
                check.expect_no_gl_error("clamping an oversized viewport raises no GL error");
            }

            // Negative sizes: INVALID_VALUE, and the state is left untouched.
            mgl::glViewport(4, 6, 20, 30);
            mgl::glScissor(5, 7, 21, 31);
            mgl::glViewport(0, 0, -1, 10);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glViewport with a negative width -> INVALID_VALUE");
            mgl::glViewport(0, 0, 10, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glViewport with a negative height -> INVALID_VALUE");
            mgl::glScissor(0, 0, -1, 10);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glScissor with a negative width -> INVALID_VALUE");
            mgl::glScissor(0, 0, 10, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glScissor with a negative height -> INVALID_VALUE");
            const State t = query_state();
            check.expect(t.viewport == std::array<GLint, 4>{4, 6, 20, 30} && t.scissor == std::array<GLint, 4>{5, 7, 21, 31},
                         "a rejected glViewport / glScissor leaves GL_VIEWPORT and GL_SCISSOR_BOX unchanged");
        }

        // ---- Corner cases drawn into a scratch rectangle (the canonical image is restored afterwards) ----
        void check_corner_cases(demo::Check& check) const
        {
            const Rect r = panel_rect(Panel::ClearColor);
            const Color marker{0.30f, 0.60f, 0.20f, 1.0f}, other{0.90f, 0.10f, 0.70f, 1.0f};
            const int cx = r.x + 32, cy = r.y + 32;

            // glClear(0): "If buf is zero, no buffers are cleared" - and it is not an error.
            begin_panel(Panel::ClearColor);
            clear_color(marker);
            mgl::glClearColor(other.r, other.g, other.b, other.a);
            mgl::glClear(mgl::ClearBufferBit{});
            check.expect_no_gl_error("glClear with an empty mask raises no error (ES 3.0 §4.2.3)");
            check.expect_pixel(cx, cy, rgba8(marker), 1, "glClear with an empty mask clears nothing");

            // Scissor test enabled with an empty box: no fragment passes, neither for clears nor for draws.
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x + 8, r.y + 8, 0, 0);
            mgl::glClear(mgl::ClearBufferBit::Color);
            quad(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, other);
            check.expect_pixel(cx, cy, rgba8(marker), 1, "a 0x0 scissor box rejects every fragment of a clear and of a draw");
            mgl::glScissor(r.x, r.y, 0, kPanel);
            mgl::glClear(mgl::ClearBufferBit::Color);
            check.expect_pixel(r.x, cy, rgba8(marker), 1, "a scissor box of width 0 and full height also rejects everything");

            // The scissor box is ignored while the test is disabled.
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x + 8, r.y + 8, 1, 1);
            mgl::glViewport(r.x, r.y, kPanel, kPanel);
            mgl::glClear(mgl::ClearBufferBit::Color);
            check.expect_pixel(cx, cy, rgba8(other), 1, "with GL_SCISSOR_TEST disabled glClear clears the whole window (box ignored)");

            // A clear with every color channel masked changes nothing; masks do not depend on the scissor.
            mgl::glClearColor(marker.r, marker.g, marker.b, marker.a);
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            mgl::glClear(mgl::ClearBufferBit::Color);
            check.expect_pixel(cx, cy, rgba8(other), 1, "a clear with glColorMask(F,F,F,F) leaves the color buffer untouched");
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            // Clear color is clamped to [0,1] when it is applied to a normalized buffer.
            mgl::glClearColor(2.0f, -1.0f, 0.5f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            check.expect_pixel(cx, cy, rgba8({1.0f, 0.0f, 0.5f, 1.0f}), 1, "glClearColor(2,-1,.5,1) writes (255,0,128,255): components are clamped (ES 3.0 §4.2.3)");
            check.expect_no_gl_error("corner cases raised no GL error");
        }

        // ---------------------------------------------------------------------------------------
        mgl::ProgramId program_{};
        mgl::UniformLocation u_vertex_{}, u_z_{}, u_color_{};
        State initial_{};
        bool has_alpha_ = true;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-clear-viewport-scissor",
        .title = "Clear, viewport and scissor",
        .min_es = {3, 0},
        .summary = "Clear values and masks, the scissor test, the viewport transform and glDepthRangef, shown in a 4x4 "
                   "gallery; depth and stencil contents are observed through depth- and stencil-tested probe draws.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ClearViewportScissor>(argc, argv, info);
}
