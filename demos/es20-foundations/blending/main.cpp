// es20-blending
//
// Fragment blending (OpenGL ES 3.0 section 4.1.7): every blend factor of Table 4.2, the five blend
// equations of Table 4.1 (FUNC_ADD, FUNC_SUBTRACT, FUNC_REVERSE_SUBTRACT and the ES 3.0 MIN / MAX),
// separate RGB / alpha factors and equations, the constant blend color, and the classic use:
// alpha compositing of translucent shapes in painter's order.
//
// Image (256 x 256 default framebuffer, RGBA8):
//   * Top: a 16 x 4 contact sheet. Every cell is a *destination* color (the 1-pixel-inset
//     frame) with a *source* square drawn on top of it using one blend configuration.
//       row 0  source factors      glBlendFunc(X, ZERO)   result = src * X       (15 + 1 cells)
//       row 1  destination factors glBlendFunc(ZERO, X)   result = dst * X       (14 cells) + 2 controls
//       row 2  equations, separate equations, constant-color clamping, additive, premultiplied
//       row 3  glBlendFuncSeparate (4 cells) and a glBlendColor ramp (12 cells)
//   * Bottom: three translucent discs (red, green, blue) over a checkerboard. The left cluster
//     is drawn red-green-blue, the right one blue-green-red: the "over" operator is not
//     commutative, so the overlaps differ.
//
// Validation: every cell is compared with a CPU implementation of Tables 4.1 and 4.2 that works on
// the 8-bit quantized source and destination values the framebuffer actually holds; the composite
// scene is compared with the same model applied once per draw (the framebuffer re-quantizes to
// 8 bits after every draw). The blend state is also read back with glGet*.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    using Rgba8 = demo::Rgba8;
    using Rgbaf = std::array<float, 4>;

    // ---------------------------------------------------------------------------------------------
    // Colors. Every color is a multiple of 1/255 so that it is exactly representable in the 8-bit
    // framebuffer: source (fragment shader output) and destination (framebuffer) then enter the
    // blend equation without any quantization error on the input side.
    // ---------------------------------------------------------------------------------------------
    constexpr Rgba8 kSrc = {204, 102, 51, 153};            // (0.80, 0.40, 0.20, 0.60)
    constexpr Rgba8 kDst = {51, 128, 230, 179};            // (0.20, 0.50, 0.90, 0.70)
    constexpr Rgba8 kDstLowAlpha = {51, 128, 230, 77};     // destination alpha 0.30 (SRC_ALPHA_SATURATE other branch)
    constexpr Rgba8 kDstOpaque = {51, 128, 230, 255};
    constexpr Rgba8 kEqSrc = {204, 77, 153, 77};           // (0.8, 0.3, 0.6, 0.3): s - d and d - s both non-trivial
    constexpr Rgba8 kEqDst = {77, 179, 51, 230};           // (0.3, 0.7, 0.2, 0.9)
    constexpr Rgba8 kPremultSrc = {122, 61, 31, 153};      // kSrc.rgb * kSrc.a: premultiplied alpha
    constexpr Rgbaf kConstant = {64.0f / 255.0f, 191.0f / 255.0f, 128.0f / 255.0f, 89.0f / 255.0f};
    constexpr Rgbaf kHalf = {128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f};
    constexpr Rgbaf kOutOfRange = {1.5f, -0.5f, 0.5f, 2.0f};  // glBlendColor values outside [0, 1]
    constexpr std::array<float, 4> kClear = {0.10f, 0.11f, 0.14f, 1.0f};

    constexpr Rgba8 kLight = {205, 205, 205, 255};
    constexpr Rgba8 kDark = {120, 120, 120, 255};
    constexpr Rgba8 kDiscRed = {230, 40, 40, 153};
    constexpr Rgba8 kDiscGreen = {40, 200, 60, 153};
    constexpr Rgba8 kDiscBlue = {50, 70, 230, 153};

    /// Per-component tolerance of one blend operation, in 8-bit steps.
    ///   * The result is computed in floating point and then converted to 8 bits (eq. 2.3, round to
    ///     nearest). A GPU that blends in float agrees with the CPU model to +-1 at most (float
    ///     rounding right at a .5 boundary).
    ///   * An implementation that blends in 8-bit fixed point (llvmpipe does) approximates each
    ///     factor product with a rounded division by 255, which costs up to one step per product;
    ///     a sum of two products therefore may differ by two steps.
    /// The spec only demands "a precision no lower than the destination components" (4.1.7.1).
    constexpr int kBlendTolerance = 2;

    constexpr std::array<mgl::BlendFactor, 15> kSourceFactors = {
        mgl::BlendFactor::Zero,          mgl::BlendFactor::One,
        mgl::BlendFactor::SrcColor,      mgl::BlendFactor::OneMinusSrcColor,
        mgl::BlendFactor::DstColor,      mgl::BlendFactor::OneMinusDstColor,
        mgl::BlendFactor::SrcAlpha,      mgl::BlendFactor::OneMinusSrcAlpha,
        mgl::BlendFactor::DstAlpha,      mgl::BlendFactor::OneMinusDstAlpha,
        mgl::BlendFactor::ConstantColor, mgl::BlendFactor::OneMinusConstantColor,
        mgl::BlendFactor::ConstantAlpha, mgl::BlendFactor::OneMinusConstantAlpha,
        mgl::BlendFactor::SrcAlphaSaturate,
    };
    // Same list without SRC_ALPHA_SATURATE, which Table 4.2 lists but which is only meaningful
    // (and in the ES 2.0 lineage only legal) as a source factor.
    constexpr std::array<mgl::BlendFactor, 14> kDestinationFactors = {
        mgl::BlendFactor::Zero,          mgl::BlendFactor::One,
        mgl::BlendFactor::SrcColor,      mgl::BlendFactor::OneMinusSrcColor,
        mgl::BlendFactor::DstColor,      mgl::BlendFactor::OneMinusDstColor,
        mgl::BlendFactor::SrcAlpha,      mgl::BlendFactor::OneMinusSrcAlpha,
        mgl::BlendFactor::DstAlpha,      mgl::BlendFactor::OneMinusDstAlpha,
        mgl::BlendFactor::ConstantColor, mgl::BlendFactor::OneMinusConstantColor,
        mgl::BlendFactor::ConstantAlpha, mgl::BlendFactor::OneMinusConstantAlpha,
    };

    std::string_view name(mgl::BlendFactor f)
    {
        switch (f)
        {
        case mgl::BlendFactor::Zero: return "ZERO";
        case mgl::BlendFactor::One: return "ONE";
        case mgl::BlendFactor::SrcColor: return "SRC_COLOR";
        case mgl::BlendFactor::OneMinusSrcColor: return "ONE_MINUS_SRC_COLOR";
        case mgl::BlendFactor::SrcAlpha: return "SRC_ALPHA";
        case mgl::BlendFactor::OneMinusSrcAlpha: return "ONE_MINUS_SRC_ALPHA";
        case mgl::BlendFactor::DstAlpha: return "DST_ALPHA";
        case mgl::BlendFactor::OneMinusDstAlpha: return "ONE_MINUS_DST_ALPHA";
        case mgl::BlendFactor::DstColor: return "DST_COLOR";
        case mgl::BlendFactor::OneMinusDstColor: return "ONE_MINUS_DST_COLOR";
        case mgl::BlendFactor::SrcAlphaSaturate: return "SRC_ALPHA_SATURATE";
        case mgl::BlendFactor::ConstantColor: return "CONSTANT_COLOR";
        case mgl::BlendFactor::OneMinusConstantColor: return "ONE_MINUS_CONSTANT_COLOR";
        case mgl::BlendFactor::ConstantAlpha: return "CONSTANT_ALPHA";
        case mgl::BlendFactor::OneMinusConstantAlpha: return "ONE_MINUS_CONSTANT_ALPHA";
        }
        return "?";
    }

    /// The numeric value a glGet* query returns for an enumerator.
    GLint as_int(mgl::BlendFactor f) { return static_cast<GLint>(static_cast<GLenum>(f)); }
    GLint as_int(mgl::BlendEquation e) { return static_cast<GLint>(static_cast<GLenum>(e)); }

    // ---------------------------------------------------------------------------------------------
    // CPU reference: ES 3.0 section 4.1.7, Tables 4.1 and 4.2.
    // ---------------------------------------------------------------------------------------------
    struct BlendConfig
    {
        mgl::BlendFactor src_rgb = mgl::BlendFactor::One;
        mgl::BlendFactor dst_rgb = mgl::BlendFactor::Zero;
        mgl::BlendFactor src_alpha = mgl::BlendFactor::One;
        mgl::BlendFactor dst_alpha = mgl::BlendFactor::Zero;
        mgl::BlendEquation eq_rgb = mgl::BlendEquation::FuncAdd;
        mgl::BlendEquation eq_alpha = mgl::BlendEquation::FuncAdd;
        Rgbaf constant = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    float clamp01(float v) { return std::clamp(v, 0.0f, 1.0f); }

    Rgbaf to_float(Rgba8 c)
    {
        return {static_cast<float>(c[0]) / 255.0f, static_cast<float>(c[1]) / 255.0f,
                static_cast<float>(c[2]) / 255.0f, static_cast<float>(c[3]) / 255.0f};
    }

    std::uint8_t to_byte(float v)
    {
        // Equation 2.3: round to nearest after clamping to [0, 1] (section 4.1.8).
        return static_cast<std::uint8_t>(std::lround(clamp01(v) * 255.0f));
    }

    /// Table 4.2: the factor as (R, G, B, A) weights. The RGB equation uses the first three entries,
    /// the alpha equation the fourth ("RGB Blend Factors" / "Alpha Blend Factor" columns).
    Rgbaf weights(mgl::BlendFactor f, const Rgbaf& s, const Rgbaf& d, const Rgbaf& c)
    {
        const float saturate = std::min(s[3], 1.0f - d[3]);  // footnote 1: f = min(As, 1 - Ad)
        switch (f)
        {
        case mgl::BlendFactor::Zero: return {0.0f, 0.0f, 0.0f, 0.0f};
        case mgl::BlendFactor::One: return {1.0f, 1.0f, 1.0f, 1.0f};
        case mgl::BlendFactor::SrcColor: return s;
        case mgl::BlendFactor::OneMinusSrcColor: return {1.0f - s[0], 1.0f - s[1], 1.0f - s[2], 1.0f - s[3]};
        case mgl::BlendFactor::DstColor: return d;
        case mgl::BlendFactor::OneMinusDstColor: return {1.0f - d[0], 1.0f - d[1], 1.0f - d[2], 1.0f - d[3]};
        case mgl::BlendFactor::SrcAlpha: return {s[3], s[3], s[3], s[3]};
        case mgl::BlendFactor::OneMinusSrcAlpha: return {1.0f - s[3], 1.0f - s[3], 1.0f - s[3], 1.0f - s[3]};
        case mgl::BlendFactor::DstAlpha: return {d[3], d[3], d[3], d[3]};
        case mgl::BlendFactor::OneMinusDstAlpha: return {1.0f - d[3], 1.0f - d[3], 1.0f - d[3], 1.0f - d[3]};
        case mgl::BlendFactor::ConstantColor: return c;
        case mgl::BlendFactor::OneMinusConstantColor: return {1.0f - c[0], 1.0f - c[1], 1.0f - c[2], 1.0f - c[3]};
        case mgl::BlendFactor::ConstantAlpha: return {c[3], c[3], c[3], c[3]};
        case mgl::BlendFactor::OneMinusConstantAlpha: return {1.0f - c[3], 1.0f - c[3], 1.0f - c[3], 1.0f - c[3]};
        case mgl::BlendFactor::SrcAlphaSaturate: return {saturate, saturate, saturate, 1.0f};
        }
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    /// Table 4.1, one component. MIN and MAX ignore the factors altogether.
    float combine(mgl::BlendEquation e, float s, float factor_s, float d, float factor_d)
    {
        switch (e)
        {
        case mgl::BlendEquation::FuncAdd: return s * factor_s + d * factor_d;
        case mgl::BlendEquation::FuncSubtract: return s * factor_s - d * factor_d;
        case mgl::BlendEquation::FuncReverseSubtract: return d * factor_d - s * factor_s;
        case mgl::BlendEquation::Min: return std::min(s, d);
        case mgl::BlendEquation::Max: return std::max(s, d);
        default: return 0.0f;
        }
    }

    /// One blend operation on 8-bit source and destination values; returns the 8-bit value that the
    /// framebuffer holds afterwards. `dst_has_alpha == false`: Ad is taken to be 1 (section 4.1.7.4).
    Rgba8 reference_blend(const BlendConfig& cfg, Rgba8 src8, Rgba8 dst8, bool dst_has_alpha)
    {
        // "The components of the source and destination values and blend factors are clamped to
        // [0, 1] prior to evaluating the blend equation." The constant color is clamped too
        // (section 4.1.7.3) because the destination is unsigned normalized.
        Rgbaf s = to_float(src8), d = to_float(dst8), c = cfg.constant;
        if (!dst_has_alpha) d[3] = 1.0f;
        for (auto* v : {&s, &d, &c})
            for (float& x : *v) x = clamp01(x);

        const Rgbaf src_rgb = weights(cfg.src_rgb, s, d, c), dst_rgb = weights(cfg.dst_rgb, s, d, c);
        const Rgbaf src_a = weights(cfg.src_alpha, s, d, c), dst_a = weights(cfg.dst_alpha, s, d, c);

        Rgba8 out{};
        for (std::size_t i = 0; i < 3; ++i)
            out[i] = to_byte(combine(cfg.eq_rgb, s[i], clamp01(src_rgb[i]), d[i], clamp01(dst_rgb[i])));
        out[3] = to_byte(combine(cfg.eq_alpha, s[3], clamp01(src_a[3]), d[3], clamp01(dst_a[3])));
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // Geometry
    // ---------------------------------------------------------------------------------------------
    struct Rect  // window coordinates, origin bottom-left, in pixels
    {
        int x = 0, y = 0, w = 0, h = 0;
    };

    constexpr Rect inset(Rect r, int d) { return {r.x + d, r.y + d, r.w - 2 * d, r.h - 2 * d}; }

    constexpr int kColumns = 16;
    constexpr int kRows = 4;
    constexpr int kChecker = 12;  // checkerboard square size in pixels

    struct Layout
    {
        int width = 0, height = 0, margin = 8, cell_w = 0, cell_h = 0;
        Rect composite;

        static Layout make(int width, int height)
        {
            Layout l;
            l.width = width;
            l.height = height;
            l.cell_w = (width - 2 * l.margin) / kColumns;
            l.cell_h = std::max(10, height * 20 / 256);
            const int grid_bottom = height - l.margin - kRows * l.cell_h;
            l.composite = {l.margin, l.margin, width - 2 * l.margin, grid_bottom - 8 - l.margin};
            return l;
        }

        [[nodiscard]] Rect cell(int column, int row) const
        {
            const int left = (width - kColumns * cell_w) / 2;
            return {left + column * cell_w, height - margin - (row + 1) * cell_h, cell_w, cell_h};
        }
    };

    struct Cell
    {
        std::string label;
        Rgba8 src{};
        Rgba8 dst{};
        BlendConfig cfg;
        bool blend = true;            // false: control cell with blending disabled
        bool func_separate = false;   // glBlendFuncSeparate instead of glBlendFunc
        bool eq_separate = false;     // glBlendEquationSeparate instead of glBlendEquation
        Rgba8 expected{};             // filled in by finish()
    };

    struct Disc
    {
        float cx = 0, cy = 0, r = 0;
        Rgba8 color{};
    };

    struct Probe  // one composite-scene sample point with its CPU-reference color
    {
        int x = 0, y = 0;
        Rgba8 expected{};
        int tolerance = 0;
        int layers = 0;
        std::string label;
    };

    constexpr float kDiscRadius = 36.0f;

    class Blending final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // The whole demo is written for an 8-bit-per-channel default framebuffer.
            const int red = demo::get_integer(mgl::GetParameter::RedBits);
            const int green = demo::get_integer(mgl::GetParameter::GreenBits);
            const int blue = demo::get_integer(mgl::GetParameter::BlueBits);
            alpha_bits_ = demo::get_integer(mgl::GetParameter::AlphaBits);
            if (red != 8 || green != 8 || blue != 8 || (alpha_bits_ != 0 && alpha_bits_ != 8))
                throw demo::Skip(std::format("needs an 8-bit RGB(A) default framebuffer, have {}{}{}{}", red, green,
                                             blue, alpha_bits_));
            if (env.width < 192 || env.height < 192) throw demo::Skip("surface smaller than 192 x 192");
            dst_has_alpha_ = alpha_bits_ == 8;

            // Initial state (ES 3.0 section 4.1.7.4 and 4.1.9) - read before anything changes it.
            initial_.blend_enabled = mgl::glIsEnabled(mgl::Capability::Blend);
            initial_.dither_enabled = mgl::glIsEnabled(mgl::Capability::Dither);
            initial_.src_rgb = demo::get_integer(mgl::GetParameter::BlendSrcRgb);
            initial_.dst_rgb = demo::get_integer(mgl::GetParameter::BlendDstRgb);
            initial_.src_alpha = demo::get_integer(mgl::GetParameter::BlendSrcAlpha);
            initial_.dst_alpha = demo::get_integer(mgl::GetParameter::BlendDstAlpha);
            initial_.eq_rgb = demo::get_integer(mgl::GetParameter::BlendEquationRgb);
            initial_.eq_alpha = demo::get_integer(mgl::GetParameter::BlendEquationAlpha);
            mgl::glGetFloatv(mgl::GetParameter::BlendColor, initial_.color.data());

            program_ = demo::build_program(env, "shaders/rect.vert", "shaders/color.frag");
            u_rect_ = mgl::glGetUniformLocation(program_, "u_rect");
            u_color_ = mgl::glGetUniformLocation(program_, "u_color");
            u_disc_ = mgl::glGetUniformLocation(program_, "u_disc");
            if (u_rect_.value < 0 || u_color_.value < 0 || u_disc_.value < 0)
                throw demo::Failure("rect.vert / color.frag: a uniform is not active");

            layout_ = Layout::make(env.width, env.height);
            build_cells();
            build_composite();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);

            // Dithering (initially enabled, section 4.1.9) may add +-1 to a color on the way into the
            // framebuffer. Switching it off makes the blend result reproducible; it is not part of
            // blending itself.
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(program_);
            for (std::size_t i = 0; i < cells_.size(); ++i)
                draw_cell(cells_[i], layout_.cell(static_cast<int>(i) % kColumns, static_cast<int>(i) / kColumns));
            draw_composite();
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_framebuffer(check);
            check_initial_state(check);
            check_cells(check);
            check_composite(check);
            check_state_queries(check);
            std::cout << std::format("     largest deviation from the CPU reference over all pixel checks: {}/255 "
                                     "(allowed per blend operation: {})\n",
                                     max_deviation_, kBlendTolerance);
            (void)env;
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program_);
        }

    private:
        // -----------------------------------------------------------------------------------------
        // Test sheet definition
        // -----------------------------------------------------------------------------------------
        void add(Cell cell)
        {
            if (!cell.func_separate)
            {
                // glBlendFunc(src, dst) sets the alpha factors to the same values as the RGB ones.
                cell.cfg.src_alpha = cell.cfg.src_rgb;
                cell.cfg.dst_alpha = cell.cfg.dst_rgb;
            }
            if (!cell.eq_separate) cell.cfg.eq_alpha = cell.cfg.eq_rgb;
            cell.expected = cell.blend ? reference_blend(cell.cfg, cell.src, cell.dst, dst_has_alpha_) : cell.src;
            cells_.push_back(std::move(cell));
        }

        void add_factor_cell(mgl::BlendFactor src_factor, mgl::BlendFactor dst_factor, Rgba8 dst, std::string label)
        {
            Cell c;
            c.label = std::move(label);
            c.src = kSrc;
            c.dst = dst;
            c.cfg.src_rgb = src_factor;
            c.cfg.dst_rgb = dst_factor;
            c.cfg.constant = kConstant;
            add(std::move(c));
        }

        void add_equation_cell(mgl::BlendEquation eq_rgb, mgl::BlendEquation eq_alpha, bool separate,
                               mgl::BlendFactor factor, Rgbaf constant, std::string label)
        {
            Cell c;
            c.label = std::move(label);
            c.src = kEqSrc;
            c.dst = kEqDst;
            c.cfg.src_rgb = c.cfg.dst_rgb = factor;
            c.cfg.eq_rgb = eq_rgb;
            c.cfg.eq_alpha = eq_alpha;
            c.cfg.constant = constant;
            c.eq_separate = separate;
            add(std::move(c));
        }

        void build_cells()
        {
            using F = mgl::BlendFactor;
            using E = mgl::BlendEquation;

            // Row 0: every source factor, destination factor ZERO -> result = src * factor.
            for (const F f : kSourceFactors)
                add_factor_cell(f, F::Zero, kDst, std::format("source factor {}", name(f)));
            // SRC_ALPHA_SATURATE is min(As, 1 - Ad). With Ad = 0.7 the minimum is 1 - Ad = 0.3, with
            // Ad = 0.3 it is As = 0.6: one more cell exercises the other branch of the min().
            add_factor_cell(F::SrcAlphaSaturate, F::Zero, kDstLowAlpha, "source factor SRC_ALPHA_SATURATE (Ad = 0.3)");

            // Row 1: every destination factor, source factor ZERO -> result = dst * factor.
            for (const F f : kDestinationFactors)
                add_factor_cell(F::Zero, f, kDst, std::format("destination factor {}", name(f)));
            {
                Cell c;  // control: blending disabled, the fragment simply replaces the destination
                c.label = "blending disabled (source replaces destination)";
                c.src = kSrc;
                c.dst = kDst;
                c.blend = false;
                add(std::move(c));
            }
            add_factor_cell(F::SrcAlpha, F::OneMinusSrcAlpha, kDst, "classic over: SRC_ALPHA, ONE_MINUS_SRC_ALPHA");

            // Row 2: equations with half-weights (CONSTANT_COLOR = 0.5) so that the sum, the two
            // differences and their clamping at 0 are all visible and non-trivial.
            add_equation_cell(E::FuncAdd, E::FuncAdd, false, F::ConstantColor, kHalf, "FUNC_ADD, factors 0.5");
            add_equation_cell(E::FuncSubtract, E::FuncSubtract, false, F::ConstantColor, kHalf,
                              "FUNC_SUBTRACT, factors 0.5 (negative results clamp to 0)");
            add_equation_cell(E::FuncReverseSubtract, E::FuncReverseSubtract, false, F::ConstantColor, kHalf,
                              "FUNC_REVERSE_SUBTRACT, factors 0.5");
            add_equation_cell(E::FuncAdd, E::FuncAdd, false, F::One, kHalf, "FUNC_ADD, factors ONE (sum clamps to 1)");
            // MIN / MAX (ES 3.0): "the blend factors are ignored" - ZERO factors would give black
            // with any other equation, ONE factors would give the (clamped) sum.
            add_equation_cell(E::Min, E::Min, false, F::Zero, kHalf, "MIN, factors ZERO (ignored)");
            add_equation_cell(E::Max, E::Max, false, F::Zero, kHalf, "MAX, factors ZERO (ignored)");
            add_equation_cell(E::Min, E::Min, false, F::One, kHalf, "MIN, factors ONE (ignored)");
            add_equation_cell(E::Max, E::Max, false, F::One, kHalf, "MAX, factors ONE (ignored)");
            // glBlendEquationSeparate: different equations for RGB and alpha.
            add_equation_cell(E::FuncAdd, E::Max, true, F::ConstantColor, kHalf, "separate equations: RGB ADD, alpha MAX");
            add_equation_cell(E::FuncSubtract, E::FuncReverseSubtract, true, F::ConstantColor, kHalf,
                              "separate equations: RGB SUBTRACT, alpha REVERSE_SUBTRACT");
            add_equation_cell(E::Min, E::FuncAdd, true, F::ConstantColor, kHalf, "separate equations: RGB MIN, alpha ADD");
            add_equation_cell(E::Max, E::Min, true, F::ConstantColor, kHalf, "separate equations: RGB MAX, alpha MIN");
            // glBlendColor values outside [0, 1] are clamped when the factors are computed (4.1.7.3).
            // kOutOfRange = (1.5, -0.5, 0.5, 2.0) -> factors (1, 0, 0.5, 1).
            {
                Cell c;
                c.label = "constant color (1.5, -0.5, 0.5, 2.0) clamped: CONSTANT_COLOR, ZERO";
                c.src = kSrc;
                c.dst = kDst;
                c.cfg.src_rgb = F::ConstantColor;
                c.cfg.constant = kOutOfRange;
                add(std::move(c));
                Cell d;
                d.label = "constant color (1.5, -0.5, 0.5, 2.0) clamped: CONSTANT_COLOR, ONE_MINUS_CONSTANT_COLOR";
                d.src = kSrc;
                d.dst = kDst;
                d.cfg.src_rgb = F::ConstantColor;
                d.cfg.dst_rgb = F::OneMinusConstantColor;
                d.cfg.constant = kOutOfRange;
                add(std::move(d));
            }
            add_factor_cell(F::SrcAlpha, F::One, kDst, "additive: SRC_ALPHA, ONE");
            {
                Cell c;  // premultiplied alpha: the shader already multiplied rgb by alpha
                c.label = "premultiplied alpha: ONE, ONE_MINUS_SRC_ALPHA";
                c.src = kPremultSrc;
                c.dst = kDst;
                c.cfg.src_rgb = F::One;
                c.cfg.dst_rgb = F::OneMinusSrcAlpha;
                add(std::move(c));
            }

            // Row 3: glBlendFuncSeparate - four independent factors.
            auto separate = [&](F srgb, F drgb, F salpha, F dalpha, Rgba8 dst, std::string label) {
                Cell c;
                c.label = std::move(label);
                c.src = kSrc;
                c.dst = dst;
                c.cfg = {srgb, drgb, salpha, dalpha, E::FuncAdd, E::FuncAdd, kConstant};
                c.func_separate = true;
                add(std::move(c));
            };
            separate(F::SrcAlpha, F::OneMinusSrcAlpha, F::One, F::Zero, kDst,
                     "separate factors: RGB over, alpha replaced by source alpha");
            separate(F::Zero, F::SrcColor, F::DstAlpha, F::DstAlpha, kDst,
                     "separate factors: RGB (ZERO, SRC_COLOR), alpha (DST_ALPHA, DST_ALPHA)");
            separate(F::ConstantColor, F::OneMinusDstColor, F::OneMinusConstantAlpha, F::ConstantAlpha, kDst,
                     "separate factors: RGB (CONSTANT_COLOR, ONE_MINUS_DST_COLOR), alpha (ONE_MINUS_CONSTANT_ALPHA, CONSTANT_ALPHA)");
            separate(F::SrcAlpha, F::OneMinusSrcAlpha, F::One, F::OneMinusSrcAlpha, kDstOpaque,
                     "separate factors: RGB over, alpha keeps an opaque destination opaque");
            // glBlendColor ramp: CONSTANT_ALPHA / ONE_MINUS_CONSTANT_ALPHA is a cross-fade whose
            // weight is the blend color's alpha, k / 11.
            for (int k = 0; k < 12; ++k)
            {
                Cell c;
                c.label = std::format("glBlendColor alpha = {}/11: CONSTANT_ALPHA, ONE_MINUS_CONSTANT_ALPHA", k);
                c.src = kSrc;
                c.dst = kDst;
                c.cfg.src_rgb = F::ConstantAlpha;
                c.cfg.dst_rgb = F::OneMinusConstantAlpha;
                c.cfg.constant = {0.0f, 0.0f, 0.0f, static_cast<float>(k) / 11.0f};
                add(std::move(c));
            }
        }

        // -----------------------------------------------------------------------------------------
        // Composite scene: three translucent discs over a checkerboard, painter's order.
        // -----------------------------------------------------------------------------------------
        static BlendConfig over_config()
        {
            // RGB: Cs * As + Cd * (1 - As).  Alpha: As + Ad * (1 - As) - an opaque backdrop stays opaque.
            return {mgl::BlendFactor::SrcAlpha, mgl::BlendFactor::OneMinusSrcAlpha,
                    mgl::BlendFactor::One,      mgl::BlendFactor::OneMinusSrcAlpha,
                    mgl::BlendEquation::FuncAdd, mgl::BlendEquation::FuncAdd, {0.0f, 0.0f, 0.0f, 0.0f}};
        }

        Rgba8 backdrop_at(int x, int y) const
        {
            const int i = (x - layout_.composite.x) / kChecker, j = (y - layout_.composite.y) / kChecker;
            return ((i + j) % 2 == 0) ? kLight : kDark;
        }

        void build_composite()
        {
            const Rect r = layout_.composite;
            const float cy = static_cast<float>(r.y) + static_cast<float>(r.h) / 2.0f;
            const float left_x = static_cast<float>(r.x) + static_cast<float>(r.w) / 4.0f;
            const float right_x = static_cast<float>(r.x) + 3.0f * static_cast<float>(r.w) / 4.0f;
            // Venn arrangement: red and green on top, blue below; every pair and the triple overlap.
            auto cluster = [&](float cx, std::array<int, 3> order, const char* side) {
                std::array<Disc, 3> discs = {{{cx - 20.0f, cy + 12.0f, kDiscRadius, kDiscRed},
                                              {cx + 20.0f, cy + 12.0f, kDiscRadius, kDiscGreen},
                                              {cx, cy - 22.0f, kDiscRadius, kDiscBlue}}};
                clusters_.push_back({discs, order});
                probe_cluster(clusters_.back(), side);
            };
            cluster(left_x, {0, 1, 2}, "left cluster, drawn red-green-blue");
            cluster(right_x, {2, 1, 0}, "right cluster, drawn blue-green-red");
        }

        struct Cluster
        {
            std::array<Disc, 3> discs;
            std::array<int, 3> order;  // draw order (indices into discs)
        };

        /// For each of the 8 membership patterns (which discs cover the pixel) pick the pixel whose
        /// center is farthest from every disc outline and every checkerboard edge, and compute the
        /// expected color by applying the blend model once per covering disc in draw order.
        void probe_cluster(const Cluster& cl, const char* side)
        {
            static constexpr std::array<const char*, 3> kLetter = {"red", "green", "blue"};
            const Rect r = layout_.composite;
            const float cx = cl.discs[2].cx, cy = (cl.discs[0].cy + cl.discs[2].cy) / 2.0f;
            struct Best
            {
                float margin = -1.0f;
                int x = 0, y = 0;
            };
            std::array<Best, 8> best{};
            for (int y = static_cast<int>(cy - 1.6f * kDiscRadius); y < static_cast<int>(cy + 1.6f * kDiscRadius); ++y)
                for (int x = static_cast<int>(cx - 1.6f * kDiscRadius); x < static_cast<int>(cx + 1.6f * kDiscRadius); ++x)
                {
                    if (x < r.x || y < r.y || x >= r.x + r.w || y >= r.y + r.h) continue;
                    const float px = static_cast<float>(x) + 0.5f, py = static_cast<float>(y) + 0.5f;
                    unsigned pattern = 0;
                    float margin = 1e9f;
                    for (std::size_t k = 0; k < 3; ++k)
                    {
                        const float dist = std::hypot(px - cl.discs[k].cx, py - cl.discs[k].cy);
                        if (dist < cl.discs[k].r) pattern |= 1u << k;
                        margin = std::min(margin, std::abs(dist - cl.discs[k].r));
                    }
                    const float fx = std::fmod(px - static_cast<float>(r.x), static_cast<float>(kChecker));
                    const float fy = std::fmod(py - static_cast<float>(r.y), static_cast<float>(kChecker));
                    margin = std::min({margin, fx, static_cast<float>(kChecker) - fx, fy, static_cast<float>(kChecker) - fy});
                    if (margin > best[pattern].margin) best[pattern] = {margin, x, y};
                }

            for (unsigned pattern = 0; pattern < 8; ++pattern)
            {
                if (best[pattern].margin < 3.0f)
                    throw demo::Failure(std::format("composite layout: no probe pixel 3 px away from all edges for pattern {}", pattern));
                Probe p;
                p.x = best[pattern].x;
                p.y = best[pattern].y;
                Rgba8 color = backdrop_at(p.x, p.y);
                for (const int k : cl.order)
                    if ((pattern & (1u << static_cast<unsigned>(k))) != 0)
                        color = reference_blend(over_config(), cl.discs[static_cast<std::size_t>(k)].color, color, dst_has_alpha_);
                p.expected = color;
                p.layers = std::popcount(pattern);
                // Tolerance: the error of one operation is at most kBlendTolerance; an error in the
                // destination reaches the next result scaled by (1 - As) = 0.4, so n stacked blends
                // are bounded by 2 * (1 + 0.4 + 0.16) = 3.1 -> 2, 3, 4 for 1, 2, 3 layers.
                // A pixel without any disc is a blend-disabled draw: 8-bit conversion only (+-1).
                p.tolerance = p.layers == 0 ? 1 : kBlendTolerance + (p.layers - 1);
                std::string who;
                for (unsigned k = 0; k < 3; ++k)
                    if ((pattern & (1u << k)) != 0) who += std::format("{}{}", who.empty() ? "" : "+", kLetter[k]);
                p.label = std::format("{}: {}", side, who.empty() ? "backdrop only" : who);
                probes_.push_back(std::move(p));
            }
        }

        // -----------------------------------------------------------------------------------------
        // Drawing
        // -----------------------------------------------------------------------------------------
        /// Draws one rectangle (window pixel coordinates) in a flat color, optionally as inscribed disc.
        void draw_rect(float x0, float y0, float x1, float y1, Rgba8 color, bool disc)
        {
            const float w = static_cast<float>(layout_.width), h = static_cast<float>(layout_.height);
            mgl::glUniform4f(u_rect_, 2.0f * x0 / w - 1.0f, 2.0f * y0 / h - 1.0f, 2.0f * x1 / w - 1.0f, 2.0f * y1 / h - 1.0f);
            const Rgbaf c = to_float(color);
            mgl::glUniform4f(u_color_, c[0], c[1], c[2], c[3]);
            mgl::glUniform1i(u_disc_, disc ? 1 : 0);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);  // corners come from gl_VertexID
        }

        void draw_rect(Rect r, Rgba8 color)
        {
            draw_rect(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.x + r.w),
                      static_cast<float>(r.y + r.h), color, false);
        }

        void draw_cell(const Cell& c, Rect cell)
        {
            // 1. The destination: written with blending disabled, so the framebuffer holds exactly c.dst.
            mgl::glDisable(mgl::Capability::Blend);
            draw_rect(inset(cell, 1), c.dst);

            // 2. The source square over it, with this cell's blend configuration.
            if (c.blend)
            {
                if (c.func_separate)
                    mgl::glBlendFuncSeparate(c.cfg.src_rgb, c.cfg.dst_rgb, c.cfg.src_alpha, c.cfg.dst_alpha);
                else
                    mgl::glBlendFunc(c.cfg.src_rgb, c.cfg.dst_rgb);
                if (c.eq_separate)
                    mgl::glBlendEquationSeparate(c.cfg.eq_rgb, c.cfg.eq_alpha);
                else
                    mgl::glBlendEquation(c.cfg.eq_rgb);
                mgl::glBlendColor(c.cfg.constant[0], c.cfg.constant[1], c.cfg.constant[2], c.cfg.constant[3]);
                mgl::glEnable(mgl::Capability::Blend);
            }
            draw_rect(inset(cell, 3), c.src);
            mgl::glDisable(mgl::Capability::Blend);
        }

        void draw_composite()
        {
            const Rect r = layout_.composite;
            // Backdrop: light fill, dark squares on top (blending disabled: plain replacement).
            mgl::glDisable(mgl::Capability::Blend);
            draw_rect(r, kLight);
            for (int j = 0; j * kChecker < r.h; ++j)
                for (int i = 0; i * kChecker < r.w; ++i)
                {
                    if ((i + j) % 2 == 0) continue;
                    Rect sq = {r.x + i * kChecker, r.y + j * kChecker, kChecker, kChecker};
                    sq.w = std::min(sq.w, r.x + r.w - sq.x);
                    sq.h = std::min(sq.h, r.y + r.h - sq.y);
                    draw_rect(sq, kDark);
                }

            // Discs in painter's order with the "over" operator.
            const BlendConfig over = over_config();
            mgl::glBlendFuncSeparate(over.src_rgb, over.dst_rgb, over.src_alpha, over.dst_alpha);
            mgl::glBlendEquation(mgl::BlendEquation::FuncAdd);
            mgl::glEnable(mgl::Capability::Blend);
            for (const Cluster& cl : clusters_)
                for (const int k : cl.order)
                {
                    const Disc& d = cl.discs[static_cast<std::size_t>(k)];
                    draw_rect(d.cx - d.r, d.cy - d.r, d.cx + d.r, d.cy + d.r, d.color, true);
                }
            // Blending stays enabled here on purpose: validate() reads the enable flag back.
        }

        // -----------------------------------------------------------------------------------------
        // Validation
        // -----------------------------------------------------------------------------------------
        /// Reads a pixel block and compares every pixel with `expected`. Alpha is skipped for
        /// framebuffers without an alpha channel, where the stored alpha is meaningless.
        bool expect_region(demo::Check& check, Rect r, Rgba8 expected, int tolerance, std::string_view what)
        {
            const std::vector<std::uint8_t> px = demo::read_rgba8(r.x, r.y, r.w, r.h);
            const std::size_t channels = dst_has_alpha_ ? 4 : 3;
            int worst = 0;
            Rgba8 first_bad{};
            bool have_bad = false;
            for (std::size_t p = 0; p + 3 < px.size(); p += 4)
                for (std::size_t ch = 0; ch < channels; ++ch)
                {
                    const int dev = std::abs(int{px[p + ch]} - int{expected[ch]});
                    worst = std::max(worst, dev);
                    if (dev > tolerance && !have_bad)
                    {
                        have_bad = true;
                        first_bad = {px[p], px[p + 1], px[p + 2], px[p + 3]};
                    }
                }
            max_deviation_ = std::max(max_deviation_, worst);
            const std::string detail =
                have_bad ? std::format("{}x{} px region: expected {} but found {} (max deviation {}, tolerance {})", r.w,
                                       r.h, demo::format_rgba(expected), demo::format_rgba(first_bad), worst, tolerance)
                         : std::format("{}x{} px region = {} (max deviation {})", r.w, r.h, demo::format_rgba(expected), worst);
            return check.expect(!have_bad, what, detail);
        }

        void check_framebuffer(demo::Check& check)
        {
            check.expect(demo::get_integer(mgl::GetParameter::RedBits) == 8 &&
                             demo::get_integer(mgl::GetParameter::GreenBits) == 8 &&
                             demo::get_integer(mgl::GetParameter::BlueBits) == 8,
                         "default framebuffer has 8 bits per color channel (blending needs a fixed-point buffer, 4.1.7)");
            std::cout << std::format("     ALPHA_BITS = {}: {}\n", alpha_bits_,
                                     dst_has_alpha_ ? "destination alpha is stored" : "no destination alpha, Ad is taken as 1");
        }

        void check_initial_state(demo::Check& check)
        {
            check.expect(!initial_.blend_enabled, "initially BLEND is disabled (4.1.7.4)");
            check.expect(initial_.dither_enabled, "initially DITHER is enabled (4.1.9)");
            check.expect_eq(initial_.src_rgb, as_int(mgl::BlendFactor::One), "initial BLEND_SRC_RGB = ONE");
            check.expect_eq(initial_.src_alpha, as_int(mgl::BlendFactor::One), "initial BLEND_SRC_ALPHA = ONE");
            check.expect_eq(initial_.dst_rgb, as_int(mgl::BlendFactor::Zero), "initial BLEND_DST_RGB = ZERO");
            check.expect_eq(initial_.dst_alpha, as_int(mgl::BlendFactor::Zero), "initial BLEND_DST_ALPHA = ZERO");
            check.expect_eq(initial_.eq_rgb, as_int(mgl::BlendEquation::FuncAdd), "initial BLEND_EQUATION_RGB = FUNC_ADD");
            check.expect_eq(initial_.eq_alpha, as_int(mgl::BlendEquation::FuncAdd), "initial BLEND_EQUATION_ALPHA = FUNC_ADD");
            check.expect(initial_.color == Rgbaf{0.0f, 0.0f, 0.0f, 0.0f}, "initial BLEND_COLOR = (0, 0, 0, 0)",
                         std::format("({}, {}, {}, {})", initial_.color[0], initial_.color[1], initial_.color[2], initial_.color[3]));
        }

        void check_cells(demo::Check& check)
        {
            // The destination frame of every cell was written with blending disabled: exact.
            bool frames_ok = true;
            for (std::size_t i = 0; i < cells_.size(); ++i)
            {
                const Rect cell = layout_.cell(static_cast<int>(i) % kColumns, static_cast<int>(i) / kColumns);
                const Rect frame = {cell.x + 1, cell.y + 1, 1, cell.h - 2};  // left edge column of the destination quad
                const std::vector<std::uint8_t> px = demo::read_rgba8(frame.x, frame.y, frame.w, frame.h);
                for (std::size_t p = 0; p + 3 < px.size(); p += 4)
                    for (std::size_t ch = 0; ch < (dst_has_alpha_ ? 4u : 3u); ++ch)
                        if (std::abs(int{px[p + ch]} - int{cells_[i].dst[ch]}) > 1) frames_ok = false;
            }
            check.expect(frames_ok, std::format("all {} destination frames (blending disabled) hold their color exactly (+-1)", cells_.size()));

            // Everything the draws did not cover keeps the clear color: the surface corners, the margins,
            // the gap between sheet and composite, and the 1-pixel gutter around every cell.
            {
                const Rgba8 clear = {to_byte(kClear[0]), to_byte(kClear[1]), to_byte(kClear[2]), to_byte(kClear[3])};
                std::vector<std::pair<int, int>> samples = {{0, 0}, {layout_.width - 1, 0}, {0, layout_.height - 1},
                                                            {layout_.width - 1, layout_.height - 1},
                                                            {layout_.width / 2, layout_.height - 3},
                                                            {layout_.width / 2, layout_.composite.y + layout_.composite.h + 3}};
                for (std::size_t i = 0; i < cells_.size(); ++i)
                {
                    const Rect cell = layout_.cell(static_cast<int>(i) % kColumns, static_cast<int>(i) / kColumns);
                    samples.emplace_back(cell.x, cell.y);  // lower-left pixel of the cell = gutter
                }
                bool all_clear = true;
                for (const auto& [x, y] : samples)
                {
                    const Rgba8 got = demo::read_pixel(x, y);
                    for (std::size_t ch = 0; ch < (dst_has_alpha_ ? 4u : 3u); ++ch)
                        if (std::abs(int{got[ch]} - int{clear[ch]}) > 1) all_clear = false;
                }
                check.expect(all_clear, std::format("{} pixels outside every drawn rectangle keep the clear color", samples.size()));
            }

            const int inner_margin = 1;  // stay one pixel inside the source quad, away from its edges
            std::set<Rgba8> seen;        // colors actually found in the framebuffer, one per cell
            for (std::size_t i = 0; i < cells_.size(); ++i)
            {
                const Cell& c = cells_[i];
                const Rect cell = layout_.cell(static_cast<int>(i) % kColumns, static_cast<int>(i) / kColumns);
                const Rect probe = inset(cell, 3 + inner_margin);
                const std::string what = std::format("cell {:2} {}: result {}", i, c.label, demo::format_rgba(c.expected));
                // Controls without blending are only subject to the 8-bit conversion (+-1).
                expect_region(check, probe, c.expected, c.blend ? kBlendTolerance : 1, what);
                seen.insert(demo::read_pixel(probe.x + probe.w / 2, probe.y + probe.h / 2));
            }

            // Guard against a degenerate sheet in which most cells would look alike (for example if
            // the blend state were ignored): the 64 cells must show many different colors. A few
            // coincide by construction (ONE/ZERO reproduce the source or destination, ZERO/ZERO and
            // the 0-weight end of the ramp give the destination or black).
            check.expect(seen.size() >= 48, "the contact sheet is informative: at least 48 different colors across its 64 cells",
                         std::format("{} distinct colors read back", seen.size()));
        }

        void check_composite(demo::Check& check)
        {
            for (const Probe& p : probes_)
                expect_region(check, {p.x, p.y, 1, 1}, p.expected, p.tolerance,
                              std::format("composite {} ({} layers): {}", p.label, p.layers, demo::format_rgba(p.expected)));

            // Painter's order matters: the triple overlap of the left (R,G,B) and right (B,G,R)
            // clusters has a different color. The CPU model predicts it; so must the pixels.
            const Probe* left = nullptr;
            const Probe* right = nullptr;
            for (const Probe& p : probes_)
                if (p.layers == 3) (left == nullptr ? left : right) = &p;
            if (left != nullptr && right != nullptr)
            {
                int model_gap = 0;
                for (std::size_t ch = 0; ch < 3; ++ch)
                    model_gap = std::max(model_gap, std::abs(int{left->expected[ch]} - int{right->expected[ch]}));
                const demo::Rgba8 a = demo::read_pixel(left->x, left->y), b = demo::read_pixel(right->x, right->y);
                int pixel_gap = 0;
                for (std::size_t ch = 0; ch < 3; ++ch) pixel_gap = std::max(pixel_gap, std::abs(int{a[ch]} - int{b[ch]}));
                check.expect(model_gap >= 20 && pixel_gap >= 20 && std::abs(pixel_gap - model_gap) <= 2 * kBlendTolerance + 2,
                             "painter's order: the triple overlap of RGB-order and BGR-order clusters differs by the predicted amount",
                             std::format("model gap {}, pixel gap {}", model_gap, pixel_gap));
            }
            else
            {
                check.expect(false, "painter's order probes exist for both clusters");
            }
        }

        void check_state_queries(demo::Check& check)
        {
            using F = mgl::BlendFactor;
            using E = mgl::BlendEquation;

            // render() left blending enabled (the composite scene); toggle and query.
            check.expect(mgl::glIsEnabled(mgl::Capability::Blend), "glIsEnabled(BLEND) is TRUE after glEnable(BLEND)");
            mgl::glDisable(mgl::Capability::Blend);
            check.expect(!mgl::glIsEnabled(mgl::Capability::Blend), "glIsEnabled(BLEND) is FALSE after glDisable(BLEND)");
            mgl::glEnable(mgl::Capability::Blend);
            check.expect(mgl::glIsEnabled(mgl::Capability::Blend), "glIsEnabled(BLEND) is TRUE again after re-enabling");
            check.expect(!mgl::glIsEnabled(mgl::Capability::Dither), "render() disabled DITHER and glIsEnabled(DITHER) reports it");

            // glBlendFunc sets RGB and alpha to the same pair.
            mgl::glBlendFunc(F::SrcAlpha, F::OneMinusSrcAlpha);
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendSrcRgb), as_int(F::SrcAlpha), "glBlendFunc: BLEND_SRC_RGB = SRC_ALPHA");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendSrcAlpha), as_int(F::SrcAlpha), "glBlendFunc: BLEND_SRC_ALPHA = SRC_ALPHA");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendDstRgb), as_int(F::OneMinusSrcAlpha), "glBlendFunc: BLEND_DST_RGB = ONE_MINUS_SRC_ALPHA");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendDstAlpha), as_int(F::OneMinusSrcAlpha), "glBlendFunc: BLEND_DST_ALPHA = ONE_MINUS_SRC_ALPHA");

            // glBlendFuncSeparate sets four independent values.
            mgl::glBlendFuncSeparate(F::ConstantColor, F::DstColor, F::SrcAlphaSaturate, F::OneMinusConstantAlpha);
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendSrcRgb), as_int(F::ConstantColor), "glBlendFuncSeparate: BLEND_SRC_RGB = CONSTANT_COLOR");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendDstRgb), as_int(F::DstColor), "glBlendFuncSeparate: BLEND_DST_RGB = DST_COLOR");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendSrcAlpha), as_int(F::SrcAlphaSaturate), "glBlendFuncSeparate: BLEND_SRC_ALPHA = SRC_ALPHA_SATURATE");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendDstAlpha), as_int(F::OneMinusConstantAlpha), "glBlendFuncSeparate: BLEND_DST_ALPHA = ONE_MINUS_CONSTANT_ALPHA");

            // glBlendEquation sets both equations, glBlendEquationSeparate each one.
            mgl::glBlendEquation(E::FuncSubtract);
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendEquationRgb), as_int(E::FuncSubtract), "glBlendEquation: BLEND_EQUATION_RGB = FUNC_SUBTRACT");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendEquationAlpha), as_int(E::FuncSubtract), "glBlendEquation: BLEND_EQUATION_ALPHA = FUNC_SUBTRACT");
            mgl::glBlendEquationSeparate(E::FuncReverseSubtract, E::Max);
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendEquationRgb), as_int(E::FuncReverseSubtract), "glBlendEquationSeparate: BLEND_EQUATION_RGB = FUNC_REVERSE_SUBTRACT");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendEquationAlpha), as_int(E::Max), "glBlendEquationSeparate: BLEND_EQUATION_ALPHA = MAX");
            mgl::glBlendEquationSeparate(E::Min, E::FuncAdd);
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendEquationRgb), as_int(E::Min), "glBlendEquationSeparate: BLEND_EQUATION_RGB = MIN");
            check.expect_eq(demo::get_integer(mgl::GetParameter::BlendEquationAlpha), as_int(E::FuncAdd), "glBlendEquationSeparate: BLEND_EQUATION_ALPHA = FUNC_ADD");

            // glBlendColor: an in-range color is stored and read back unchanged.
            mgl::glBlendColor(0.25f, 0.5f, 0.75f, 0.125f);
            Rgbaf color{};
            mgl::glGetFloatv(mgl::GetParameter::BlendColor, color.data());
            check.expect_near(static_cast<double>(color[0]), 0.25, 1e-6, "glBlendColor: BLEND_COLOR red = 0.25");
            check.expect_near(static_cast<double>(color[1]), 0.5, 1e-6, "glBlendColor: BLEND_COLOR green = 0.5");
            check.expect_near(static_cast<double>(color[2]), 0.75, 1e-6, "glBlendColor: BLEND_COLOR blue = 0.75");
            check.expect_near(static_cast<double>(color[3]), 0.125, 1e-6, "glBlendColor: BLEND_COLOR alpha = 0.125");

            check.expect_no_gl_error("valid blend commands raised no GL error");
        }

        struct InitialState
        {
            bool blend_enabled = true;
            bool dither_enabled = false;
            GLint src_rgb = -1, dst_rgb = -1, src_alpha = -1, dst_alpha = -1, eq_rgb = -1, eq_alpha = -1;
            Rgbaf color{-1.0f, -1.0f, -1.0f, -1.0f};
        };

        int alpha_bits_ = 0;
        bool dst_has_alpha_ = true;
        int max_deviation_ = 0;
        InitialState initial_;
        Layout layout_;
        std::vector<Cell> cells_;
        std::vector<Cluster> clusters_;
        std::vector<Probe> probes_;
        mgl::ProgramId program_{};
        mgl::UniformLocation u_rect_{}, u_color_{}, u_disc_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-blending",
        .title = "Blending",
        .min_es = {3, 0},
        .summary = "All blend factors and equations (incl. MIN/MAX), separate RGB/alpha state, the constant blend "
                   "color and painter's-order alpha compositing, checked against a CPU model of ES 3.0 4.1.7.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Blending>(argc, argv, info);
}
