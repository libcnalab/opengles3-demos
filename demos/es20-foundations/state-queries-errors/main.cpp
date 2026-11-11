// es20-state-queries-errors
//
// The GL state machine seen through its query commands, and the GL error model.
//
//   * Initial values of a broad slice of the ES 3.0 state tables (chapter 6.2), each read back with
//     all four simple queries (glGetBooleanv / glGetIntegerv / glGetInteger64v / glGetFloatv) and,
//     for capabilities, glIsEnabled.
//   * The type conversions of ES 3.0 §6.1.2: boolean <-> integer <-> float, rounding of ordinary
//     floating-point state, the *linear* mapping of colors / depth range / depth clear value to
//     [-2^31+1, 2^31-1] (Table 4.5 INT entry, Equation 2.4), and saturation of 64-bit values queried
//     as 32-bit integers.
//   * Indexed state (glGetIntegeri_v / glGetInteger64i_v) for uniform-buffer and transform-feedback
//     binding points, and glGetStringi(EXTENSIONS).
//   * Hints (§5.3) and the DITHER enable (§4.1.9), including the pixel-level dithering rules.
//   * GL errors (§2.5): sticky flags, "the offending command is ignored", unchanged output pointers,
//     zero return values, and INVALID_FRAMEBUFFER_OPERATION for incomplete framebuffers (§4.4.4.4).
//
// Every experiment runs in init(), so the picture can show its outcome: a "dashboard" with one cell
// per state variable / error case, green when all of that cell's GL query checks passed, red when one
// failed, amber for a documented driver deviation. The cell frame colour identifies the group. The
// checks themselves are the GL queries; validate() records them and then verifies the picture.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    using P = mgl::GetParameter;
    using EC = mgl::ErrorCode;

    // ES 3.0 §6.1.2: colors, DepthRangef values and the depth clear value queried with GetIntegerv /
    // GetInteger64v use the INT entry of Table 4.5, i.e. Equation 2.4 with b = 32:
    //     c = convert_float_int(f * (2^31 - 1), 32)
    // where convert_float_int returns "one of the two ... integer values ... closest" to the product.
    constexpr double kIntScale = 2147483647.0;

    constexpr GLint kSentinel = 0x5EED1234;
    constexpr GLint64 kSentinel64 = 0x5EED12345678LL;

    // ---------------------------------------------------------------------------------------------
    // Check bookkeeping. Records are collected in init() (so render() can show them) and replayed into
    // demo::Check by validate().
    // ---------------------------------------------------------------------------------------------
    struct Rec
    {
        bool ok = false;
        std::string what;
        std::string detail;
        std::string deviation_renderer;  // non-empty: documented driver deviation
        std::string deviation_reason;
    };

    enum class Status
    {
        Pass,
        Fail,
        KnownDeviation
    };

    struct Group
    {
        const char* name;
        demo::Rgba8 tint;  // cell frame colour
    };

    // Frame tints. All colour components are k/255, so the clear converts them exactly (up to the
    // "one of the two closest values" freedom of Equation 2.3).
    constexpr std::array<Group, 8> kGroups = {{
        {"initial state (ES 3.0 Tables 6.2-6.36)", {70, 110, 200, 255}},
        {"implementation-dependent minimums (Tables 6.28-6.35)", {140, 90, 190, 255}},
        {"data conversions (6.1.2)", {40, 170, 190, 255}},
        {"indexed state (6.1.1, 6.1.9, 6.1.6)", {200, 120, 60, 255}},
        {"hints (5.3)", {190, 190, 190, 255}},
        {"dither (4.1.9)", {120, 160, 60, 255}},
        {"error semantics (2.5)", {200, 80, 150, 255}},
        {"incomplete framebuffers (4.4.4.4)", {110, 110, 60, 255}},
    }};
    enum GroupId
    {
        kInitial,
        kMinimums,
        kConversions,
        kIndexed,
        kHints,
        kDither,
        kErrors,
        kFramebuffer
    };

    constexpr demo::Rgba8 kBackground = {24, 24, 32, 255};
    constexpr demo::Rgba8 kPassColor = {40, 180, 80, 255};
    constexpr demo::Rgba8 kFailColor = {230, 40, 40, 255};
    constexpr demo::Rgba8 kDeviationColor = {240, 170, 20, 255};

    struct Item
    {
        int group = 0;
        std::string label;
        std::vector<Rec> recs;
    };

    class Recorder
    {
    public:
        void set_identity(std::string renderer, std::string version)
        {
            renderer_ = std::move(renderer);
            version_ = std::move(version);
        }

        /// Starts a new dashboard cell. A GL error left over from earlier code would be attributed to the
        /// wrong case, so a pending error is recorded as a failure of this item.
        void begin(int group, std::string label)
        {
            items_.push_back(Item{group, std::move(label), {}});
            const EC pending = demo::drain_gl_errors();
            if (pending != EC::NoError)
                expect(false, "no GL error pending before this case", std::string(demo::error_name(pending)));
        }

        bool expect(bool ok, std::string what, std::string detail = {})
        {
            items_.back().recs.push_back(Rec{ok, std::move(what), ok ? std::string{} : std::move(detail), {}, {}});
            return ok;
        }

        /// A strict check whose failure on the named driver is a documented deviation (docs/KNOWN_LIMITATIONS.md).
        bool expect_or_deviation(bool ok, std::string what, std::string detail, std::string renderer, std::string reason)
        {
            items_.back().recs.push_back(Rec{ok, std::move(what), ok ? std::string{} : std::move(detail),
                                             std::move(renderer), std::move(reason)});
            return ok;
        }

        /// The first queued error must be `expected`; the queue is drained afterwards.
        bool error(EC expected, std::string what)
        {
            const EC first = mgl::glGetError();
            for (int i = 0; i < 16 && mgl::glGetError() != EC::NoError; ++i)
            {
            }
            return expect(first == expected, std::move(what),
                          std::format("got {} expected {}", demo::error_name(first), demo::error_name(expected)));
        }

        bool no_error(std::string what) { return error(EC::NoError, std::move(what)); }

        [[nodiscard]] bool matches_deviation(const Rec& r) const
        {
            return !r.deviation_renderer.empty() && (renderer_.find(r.deviation_renderer) != std::string::npos ||
                                                     version_.find(r.deviation_renderer) != std::string::npos);
        }

        [[nodiscard]] Status status(const Item& item) const
        {
            Status s = Status::Pass;
            for (const Rec& r : item.recs)
            {
                if (r.ok) continue;
                if (matches_deviation(r))
                    s = Status::KnownDeviation;
                else
                    return Status::Fail;
            }
            return s;
        }

        std::vector<Item>& items() { return items_; }
        [[nodiscard]] const std::vector<Item>& items() const { return items_; }

    private:
        std::vector<Item> items_;
        std::string renderer_, version_;
    };

    // ---------------------------------------------------------------------------------------------
    // Query helpers
    // ---------------------------------------------------------------------------------------------
    struct Values
    {
        std::array<GLboolean, 16> b{};
        std::array<GLint, 16> i{};
        std::array<GLint64, 16> i64{};
        std::array<GLfloat, 16> f{};
    };

    /// Reads one state variable with all four simple query commands (ES 3.0 §6.1.1).
    Values get_all(P pname)
    {
        Values v;
        v.b.fill(0x5A);
        v.i.fill(kSentinel);
        v.i64.fill(kSentinel64);
        v.f.fill(-12345.0f);
        mgl::glGetBooleanv(pname, v.b.data());
        mgl::glGetIntegerv(pname, v.i.data());
        mgl::glGetInteger64v(pname, v.i64.data());
        mgl::glGetFloatv(pname, v.f.data());
        return v;
    }

    template <typename T>
    std::string join(const T& arr, int n)
    {
        std::string s;
        for (int k = 0; k < n; ++k)
        {
            if (k) s += ",";
            if constexpr (std::is_same_v<typename T::value_type, GLboolean>)
                s += std::to_string(static_cast<int>(arr[static_cast<std::size_t>(k)]));
            else
                s += std::format("{}", arr[static_cast<std::size_t>(k)]);
        }
        return s;
    }

    std::string join_d(const std::vector<double>& v)
    {
        std::string s;
        for (std::size_t k = 0; k < v.size(); ++k) s += (k ? "," : "") + std::format("{}", v[k]);
        return s;
    }

    enum class Kind
    {
        Integer,     // integer or enumerated state (type Z, Z+, Zk)
        Boolean,     // boolean state (type B)
        Float,       // ordinary floating-point state (R, R+): rounded when read as an integer
        Normalized,  // RGBA color component, DepthRangef value or depth clear value: linear mapping
    };

    bool in_int_mapping(double f, GLint64 got)
    {
        // Equation 2.4 (b = 32) clamps f to [-1, 1] and returns one of the two integers closest to f * (2^31-1).
        const double r = std::clamp(f, -1.0, 1.0) * kIntScale;
        return got == static_cast<GLint64>(std::floor(r)) || got == static_cast<GLint64>(std::ceil(r));
    }

    /// Verifies `pname` against `expected` through every query command, applying the ES 3.0 §6.1.2
    /// conversion that belongs to `kind`. Each command is a separate check.
    void verify_state(Recorder& rec, P pname, std::string_view name, Kind kind, const std::vector<double>& expected,
                      std::optional<mgl::Capability> cap = std::nullopt)
    {
        const int n = static_cast<int>(expected.size());
        const Values v = get_all(pname);
        rec.no_error(std::format("{}: accepted by GetBooleanv/Integerv/Integer64v/Floatv", name));

        bool ok_i = true, ok_i64 = true, ok_f = true, ok_b = true;
        for (int k = 0; k < n; ++k)
        {
            const auto u = static_cast<std::size_t>(k);
            const double e = expected[u];
            switch (kind)
            {
            case Kind::Integer:
            case Kind::Boolean:
                ok_i = ok_i && v.i[u] == static_cast<GLint>(e);
                ok_i64 = ok_i64 && v.i64[u] == static_cast<GLint64>(e);
                ok_f = ok_f && v.f[u] == static_cast<GLfloat>(e);  // integer coerced to float, booleans 1.0/0.0
                break;
            case Kind::Float:
                ok_i = ok_i && v.i[u] == static_cast<GLint>(std::lround(e));  // rounded to the nearest integer
                ok_i64 = ok_i64 && v.i64[u] == static_cast<GLint64>(std::llround(e));
                ok_f = ok_f && v.f[u] == static_cast<GLfloat>(e);
                break;
            case Kind::Normalized:
                ok_i = ok_i && in_int_mapping(e, v.i[u]);
                ok_i64 = ok_i64 && in_int_mapping(e, v.i64[u]);
                ok_f = ok_f && v.f[u] == static_cast<GLfloat>(e);
                break;
            }
            ok_b = ok_b && v.b[u] == (e != 0.0 ? GL_TRUE : GL_FALSE);  // FALSE iff zero
        }
        const std::string exp = join_d(expected);
        const char* rule = kind == Kind::Normalized ? " (linear map to 2^31-1)" : kind == Kind::Float ? " (rounded)" : "";
        rec.expect(ok_i, std::format("{}: GetIntegerv = {}{}", name, exp, rule), std::format("got {}", join(v.i, n)));
        rec.expect(ok_i64, std::format("{}: GetInteger64v = {}{}", name, exp, rule), std::format("got {}", join(v.i64, n)));
        rec.expect(ok_f, std::format("{}: GetFloatv = {}", name, exp), std::format("got {}", join(v.f, n)));
        rec.expect(ok_b, std::format("{}: GetBooleanv = (value != 0)", name), std::format("got {}", join(v.b, n)));
        if (cap)
        {
            const bool enabled = mgl::glIsEnabled(*cap);
            rec.expect(enabled == (expected[0] != 0.0), std::format("{}: glIsEnabled = {}", name, expected[0] != 0.0),
                       std::format("got {}", enabled));
        }
    }

    GLint get_i(P p)
    {
        GLint v = kSentinel;
        mgl::glGetIntegerv(p, &v);
        return v;
    }

    std::array<GLint, 4> get_i4(P p)
    {
        std::array<GLint, 4> v{kSentinel, kSentinel, kSentinel, kSentinel};
        mgl::glGetIntegerv(p, v.data());
        return v;
    }

    GLfloat get_f(P p)
    {
        GLfloat v = -12345.0f;
        mgl::glGetFloatv(p, &v);
        return v;
    }

    GLint64 get_i64(P p)
    {
        GLint64 v = kSentinel64;
        mgl::glGetInteger64v(p, &v);
        return v;
    }

    GLint get_ii(P p, GLuint index)
    {
        GLint v = kSentinel;
        mgl::glGetIntegeri_v(p, index, &v);
        return v;
    }

    GLint64 get_i64i(P p, GLuint index)
    {
        GLint64 v = kSentinel64;
        mgl::glGetInteger64i_v(p, index, &v);
        return v;
    }

    GLint buffer_param(mgl::BufferTarget t, mgl::BufferParameter p)
    {
        GLint v = kSentinel;
        mgl::glGetBufferParameteriv(t, p, &v);
        return v;
    }

    float unorm(int k) { return static_cast<float>(k) / 255.0f; }

    struct Rect
    {
        int x = 0, y = 0, w = 0, h = 0;
    };

    struct Cell
    {
        Rect outer, inner;
    };

    // ---------------------------------------------------------------------------------------------
    class StateQueriesErrors final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            const auto str = [](mgl::StringName n) {
                const GLubyte* s = mgl::glGetString(n);
                return s ? std::string(reinterpret_cast<const char*>(s)) : std::string{};
            };
            rec_.set_identity(str(mgl::StringName::Renderer), str(mgl::StringName::Version));
            (void)demo::drain_gl_errors();

            // The initial-state tables must be read before this program changes anything.
            initial_state(env);
            implementation_minimums(env);
            conversions();
            indexed_state();
            hints();
            dither();
            error_semantics();
            incomplete_framebuffers();

            // Layout sanity: every cell needs an interior to show its status.
            const auto n = static_cast<int>(rec_.items().size());
            const Cell c = cell(0, n, env.width, env.height);
            if (c.inner.w < 2 || c.inner.h < 2)
                throw demo::Skip(std::format("surface {}x{} too small for {} dashboard cells", env.width, env.height, n));

            int failed = 0, deviations = 0;
            for (const Item& it : rec_.items())
            {
                const Status s = rec_.status(it);
                failed += s == Status::Fail;
                deviations += s == Status::KnownDeviation;
            }
            std::cout << std::format("     dashboard: {} cells, {} failed, {} known deviations\n", n, failed, deviations);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::ScissorTest);
            clear_to(kBackground);
            mgl::glEnable(mgl::Capability::ScissorTest);
            const auto n = static_cast<int>(rec_.items().size());
            for (int i = 0; i < n; ++i)
            {
                const Cell c = cell(i, n, env.width, env.height);
                mgl::glScissor(c.outer.x, c.outer.y, c.outer.w, c.outer.h);
                clear_to(kGroups[static_cast<std::size_t>(rec_.items()[static_cast<std::size_t>(i)].group)].tint);
                mgl::glScissor(c.inner.x, c.inner.y, c.inner.w, c.inner.h);
                clear_to(cell_color(i));
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            // 1. Replay the GL query checks made in init().
            for (const Item& it : rec_.items())
                for (const Rec& r : it.recs)
                {
                    const std::string what = std::format("[{}] {}", it.label, r.what);
                    if (r.deviation_renderer.empty())
                        check.expect(r.ok, what, r.detail);
                    else
                        check.expect_or_known_deviation(r.ok, what, r.detail, r.deviation_renderer, r.deviation_reason);
                }

            // 2. The dashboard shows every cell in its status colour inside its group frame (±1: Equation 2.3
            //    may return either neighbour of k/255 * 255, and dithering selects one of the two neighbours).
            verify_dashboard(env, check, "dashboard");

            // 3. Negative control: the picture comparison must be able to fail. Force one cell to "failed",
            //    redraw, and require exactly that cell to turn red while its neighbour keeps its colour.
            const auto n = static_cast<int>(rec_.items().size());
            const int victim = first_item_of(kErrors);
            forced_fail_ = victim;
            render(env, 0);
            const Cell cv = cell(victim, n, env.width, env.height);
            check.expect_pixel(cv.inner.x + cv.inner.w / 2, cv.inner.y + cv.inner.h / 2, kFailColor, 1,
                               "negative control: a cell forced to 'failed' turns red");
            const Cell cn = cell(victim + 1, n, env.width, env.height);
            check.expect_pixel(cn.inner.x + cn.inner.w / 2, cn.inner.y + cn.inner.h / 2, cell_color(victim + 1), 1,
                               "negative control: its neighbour keeps its own status colour");
            forced_fail_ = -1;
            render(env, 0);
            check.expect_pixel(cv.inner.x + cv.inner.w / 2, cv.inner.y + cv.inner.h / 2, cell_color(victim), 1,
                               "negative control: the cell shows its real status again after the redraw");
        }

        void shutdown(demo::Env& /*env*/) override {}

    private:
        // =========================================================================================
        // Initial state (ES 3.0 §6.2, Tables 6.2-6.36)
        // =========================================================================================
        void initial_state(const demo::Env& env)
        {
            const double w = env.width, h = env.height;
            struct Cap
            {
                mgl::Capability cap;
                P pname;
                const char* name;
                bool initial;
            };
            const Cap caps[] = {
                {mgl::Capability::PrimitiveRestartFixedIndex, P::PrimitiveRestartFixedIndex, "PRIMITIVE_RESTART_FIXED_INDEX", false},
                {mgl::Capability::RasterizerDiscard, P::RasterizerDiscard, "RASTERIZER_DISCARD", false},
                {mgl::Capability::CullFace, P::CullFace, "CULL_FACE", false},
                {mgl::Capability::PolygonOffsetFill, P::PolygonOffsetFill, "POLYGON_OFFSET_FILL", false},
                {mgl::Capability::SampleAlphaToCoverage, P::SampleAlphaToCoverage, "SAMPLE_ALPHA_TO_COVERAGE", false},
                {mgl::Capability::SampleCoverage, P::SampleCoverage, "SAMPLE_COVERAGE", false},
                {mgl::Capability::ScissorTest, P::ScissorTest, "SCISSOR_TEST", false},
                {mgl::Capability::StencilTest, P::StencilTest, "STENCIL_TEST", false},
                {mgl::Capability::DepthTest, P::DepthTest, "DEPTH_TEST", false},
                {mgl::Capability::Blend, P::Blend, "BLEND", false},
                {mgl::Capability::Dither, P::Dither, "DITHER", true},
            };
            for (const Cap& c : caps)
            {
                rec_.begin(kInitial, c.name);
                verify_state(rec_, c.pname, c.name, Kind::Boolean, {c.initial ? 1.0 : 0.0}, c.cap);
            }

            struct Entry
            {
                P pname;
                const char* name;
                Kind kind;
                std::vector<double> value;
            };
            const std::vector<Entry> entries = {
                // Table 6.2 / 6.3: vertex arrays
                {P::ElementArrayBufferBinding, "ELEMENT_ARRAY_BUFFER_BINDING", Kind::Integer, {0}},
                {P::ArrayBufferBinding, "ARRAY_BUFFER_BINDING", Kind::Integer, {0}},
                {P::VertexArrayBinding, "VERTEX_ARRAY_BINDING", Kind::Integer, {0}},
                // Table 6.5: transformation state
                {P::Viewport, "VIEWPORT", Kind::Integer, {0, 0, w, h}},
                {P::DepthRange, "DEPTH_RANGE", Kind::Normalized, {0.0, 1.0}},
                {P::TransformFeedbackBinding, "TRANSFORM_FEEDBACK_BINDING", Kind::Integer, {0}},
                // Table 6.6: rasterization
                {P::LineWidth, "LINE_WIDTH", Kind::Float, {1.0}},
                {P::CullFaceMode, "CULL_FACE_MODE", Kind::Integer, {GL_BACK}},
                {P::FrontFace, "FRONT_FACE", Kind::Integer, {GL_CCW}},
                {P::PolygonOffsetFactor, "POLYGON_OFFSET_FACTOR", Kind::Float, {0.0}},
                {P::PolygonOffsetUnits, "POLYGON_OFFSET_UNITS", Kind::Float, {0.0}},
                // Table 6.7: multisampling
                {P::SampleCoverageValue, "SAMPLE_COVERAGE_VALUE", Kind::Float, {1.0}},
                {P::SampleCoverageInvert, "SAMPLE_COVERAGE_INVERT", Kind::Boolean, {0}},
                // Table 6.8: texture units
                {P::ActiveTexture, "ACTIVE_TEXTURE", Kind::Integer, {GL_TEXTURE0}},
                {P::TextureBinding2D, "TEXTURE_BINDING_2D", Kind::Integer, {0}},
                {P::TextureBinding3D, "TEXTURE_BINDING_3D", Kind::Integer, {0}},
                {P::TextureBinding2DArray, "TEXTURE_BINDING_2D_ARRAY", Kind::Integer, {0}},
                {P::TextureBindingCubeMap, "TEXTURE_BINDING_CUBE_MAP", Kind::Integer, {0}},
                {P::SamplerBinding, "SAMPLER_BINDING", Kind::Integer, {0}},
                // Table 6.11: pixel operations
                {P::ScissorBox, "SCISSOR_BOX", Kind::Integer, {0, 0, w, h}},
                {P::StencilFunc, "STENCIL_FUNC", Kind::Integer, {GL_ALWAYS}},
                {P::StencilRef, "STENCIL_REF", Kind::Integer, {0}},
                {P::StencilFail, "STENCIL_FAIL", Kind::Integer, {GL_KEEP}},
                {P::StencilPassDepthFail, "STENCIL_PASS_DEPTH_FAIL", Kind::Integer, {GL_KEEP}},
                {P::StencilPassDepthPass, "STENCIL_PASS_DEPTH_PASS", Kind::Integer, {GL_KEEP}},
                {P::StencilBackFunc, "STENCIL_BACK_FUNC", Kind::Integer, {GL_ALWAYS}},
                {P::StencilBackRef, "STENCIL_BACK_REF", Kind::Integer, {0}},
                {P::StencilBackFail, "STENCIL_BACK_FAIL", Kind::Integer, {GL_KEEP}},
                {P::StencilBackPassDepthFail, "STENCIL_BACK_PASS_DEPTH_FAIL", Kind::Integer, {GL_KEEP}},
                {P::StencilBackPassDepthPass, "STENCIL_BACK_PASS_DEPTH_PASS", Kind::Integer, {GL_KEEP}},
                {P::DepthFunc, "DEPTH_FUNC", Kind::Integer, {GL_LESS}},
                {P::BlendSrcRgb, "BLEND_SRC_RGB", Kind::Integer, {GL_ONE}},
                {P::BlendSrcAlpha, "BLEND_SRC_ALPHA", Kind::Integer, {GL_ONE}},
                {P::BlendDstRgb, "BLEND_DST_RGB", Kind::Integer, {GL_ZERO}},
                {P::BlendDstAlpha, "BLEND_DST_ALPHA", Kind::Integer, {GL_ZERO}},
                {P::BlendEquationRgb, "BLEND_EQUATION_RGB", Kind::Integer, {GL_FUNC_ADD}},
                {P::BlendEquationAlpha, "BLEND_EQUATION_ALPHA", Kind::Integer, {GL_FUNC_ADD}},
                {P::BlendColor, "BLEND_COLOR", Kind::Normalized, {0, 0, 0, 0}},
                // Table 6.12: framebuffer control
                {P::ColorWritemask, "COLOR_WRITEMASK", Kind::Boolean, {1, 1, 1, 1}},
                {P::DepthWritemask, "DEPTH_WRITEMASK", Kind::Boolean, {1}},
                {P::ColorClearValue, "COLOR_CLEAR_VALUE", Kind::Normalized, {0, 0, 0, 0}},
                {P::DepthClearValue, "DEPTH_CLEAR_VALUE", Kind::Normalized, {1.0}},
                {P::StencilClearValue, "STENCIL_CLEAR_VALUE", Kind::Integer, {0}},
                {P::DrawFramebufferBinding, "DRAW_FRAMEBUFFER_BINDING", Kind::Integer, {0}},
                {P::ReadFramebufferBinding, "READ_FRAMEBUFFER_BINDING", Kind::Integer, {0}},
                {P::RenderbufferBinding, "RENDERBUFFER_BINDING", Kind::Integer, {0}},
                // Table 6.13: default framebuffer - draw buffer 0 / read buffer BACK, others NONE (4.2.1, 4.3.1)
                {P::DrawBuffer0, "DRAW_BUFFER0", Kind::Integer, {GL_BACK}},
                {P::DrawBuffer1, "DRAW_BUFFER1", Kind::Integer, {GL_NONE}},
                {P::ReadBuffer, "READ_BUFFER", Kind::Integer, {GL_BACK}},
                // Table 6.16: pixels
                {P::UnpackImageHeight, "UNPACK_IMAGE_HEIGHT", Kind::Integer, {0}},
                {P::UnpackSkipImages, "UNPACK_SKIP_IMAGES", Kind::Integer, {0}},
                {P::UnpackRowLength, "UNPACK_ROW_LENGTH", Kind::Integer, {0}},
                {P::UnpackSkipRows, "UNPACK_SKIP_ROWS", Kind::Integer, {0}},
                {P::UnpackSkipPixels, "UNPACK_SKIP_PIXELS", Kind::Integer, {0}},
                {P::UnpackAlignment, "UNPACK_ALIGNMENT", Kind::Integer, {4}},
                {P::PackRowLength, "PACK_ROW_LENGTH", Kind::Integer, {0}},
                {P::PackSkipRows, "PACK_SKIP_ROWS", Kind::Integer, {0}},
                {P::PackSkipPixels, "PACK_SKIP_PIXELS", Kind::Integer, {0}},
                {P::PackAlignment, "PACK_ALIGNMENT", Kind::Integer, {4}},
                {P::PixelPackBufferBinding, "PIXEL_PACK_BUFFER_BINDING", Kind::Integer, {0}},
                {P::PixelUnpackBufferBinding, "PIXEL_UNPACK_BUFFER_BINDING", Kind::Integer, {0}},
                // Table 6.18: programs
                {P::CurrentProgram, "CURRENT_PROGRAM", Kind::Integer, {0}},
                // Tables 6.24 / 6.25: transform feedback and uniform buffer generic bindings
                {P::TransformFeedbackBufferBinding, "TRANSFORM_FEEDBACK_BUFFER_BINDING", Kind::Integer, {0}},
                {P::TransformFeedbackActive, "TRANSFORM_FEEDBACK_ACTIVE", Kind::Boolean, {0}},
                {P::TransformFeedbackPaused, "TRANSFORM_FEEDBACK_PAUSED", Kind::Boolean, {0}},
                {P::UniformBufferBinding, "UNIFORM_BUFFER_BINDING", Kind::Integer, {0}},
                // Table 6.36: miscellaneous
                {P::CopyReadBufferBinding, "COPY_READ_BUFFER_BINDING", Kind::Integer, {0}},
                {P::CopyWriteBufferBinding, "COPY_WRITE_BUFFER_BINDING", Kind::Integer, {0}},
            };
            for (const Entry& e : entries)
            {
                rec_.begin(kInitial, e.name);
                verify_state(rec_, e.pname, e.name, e.kind, e.value);
            }

            // Stencil masks: the front and back value masks are "2^s - 1, where s is greater than or equal to the
            // number of bits in the deepest stencil buffer" (4.1.4); the write masks are "1's" (Table 6.12). The
            // exact s is implementation-dependent (and 2^32-1 is not representable as a GLint), so the check is
            // that the queried 32-bit pattern is of the form 2^s - 1 and covers every stencil bit.
            const GLint stencil_bits = get_i(P::StencilBits);
            const std::uint32_t needed = stencil_bits >= 32 ? 0xFFFFFFFFu : (1u << stencil_bits) - 1u;
            const std::pair<P, const char*> masks[] = {
                {P::StencilValueMask, "STENCIL_VALUE_MASK"},
                {P::StencilBackValueMask, "STENCIL_BACK_VALUE_MASK"},
                {P::StencilWritemask, "STENCIL_WRITEMASK"},
                {P::StencilBackWritemask, "STENCIL_BACK_WRITEMASK"},
            };
            for (const auto& [pname, name] : masks)
            {
                rec_.begin(kInitial, name);
                const Values v = get_all(pname);
                rec_.no_error(std::format("{}: accepted by all four Get commands", name));
                const auto m = static_cast<std::uint32_t>(v.i[0]);
                rec_.expect((m & (m + 1u)) == 0u && (m & needed) == needed,
                            std::format("{}: GetIntegerv is 2^s-1 covering all {} stencil bits", name, stencil_bits),
                            std::format("got 0x{:08X}", m));
                rec_.expect(v.b[0] == GL_TRUE, std::format("{}: GetBooleanv = TRUE (non-zero)", name));
                std::cout << std::format("     {}: Integerv=0x{:08X} Integer64v={} Floatv={}\n", name, m, v.i64[0], v.f[0]);
            }
        }

        // =========================================================================================
        // Implementation-dependent values: ES 3.0 Tables 6.28-6.35 give minimums (maximum for the UBO
        // offset alignment). Each is also read with the other query types to verify the conversions.
        // =========================================================================================
        void implementation_minimums(const demo::Env& env)
        {
            struct Min
            {
                P pname;
                const char* name;
                GLint64 minimum;
            };
            const Min mins[] = {
                {P::SubpixelBits, "SUBPIXEL_BITS", 4},
                {P::MaxTextureSize, "MAX_TEXTURE_SIZE", 2048},
                {P::Max3DTextureSize, "MAX_3D_TEXTURE_SIZE", 256},
                {P::MaxArrayTextureLayers, "MAX_ARRAY_TEXTURE_LAYERS", 256},
                {P::MaxCubeMapTextureSize, "MAX_CUBE_MAP_TEXTURE_SIZE", 2048},
                {P::MaxRenderbufferSize, "MAX_RENDERBUFFER_SIZE", 2048},
                {P::MaxDrawBuffers, "MAX_DRAW_BUFFERS", 4},
                {P::MaxColorAttachments, "MAX_COLOR_ATTACHMENTS", 4},
                {P::MaxVertexAttribs, "MAX_VERTEX_ATTRIBS", 16},
                {P::MaxUniformBufferBindings, "MAX_UNIFORM_BUFFER_BINDINGS", 24},
                {P::MaxTransformFeedbackSeparateAttribs, "MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS", 4},
                {P::MajorVersion, "MAJOR_VERSION", 3},
            };
            for (const Min& m : mins)
            {
                rec_.begin(kMinimums, m.name);
                const Values v = get_all(m.pname);
                rec_.no_error(std::format("{}: accepted by all four Get commands", m.name));
                rec_.expect(v.i[0] >= m.minimum, std::format("{}: GetIntegerv >= {}", m.name, m.minimum),
                            std::format("got {}", v.i[0]));
                rec_.expect(v.i64[0] == v.i[0] && v.f[0] == static_cast<GLfloat>(v.i[0]) && v.b[0] == GL_TRUE,
                            std::format("{}: Integer64v / Floatv / Booleanv agree with Integerv", m.name),
                            std::format("i={} i64={} f={} b={}", v.i[0], v.i64[0], v.f[0], v.b[0]));
            }

            rec_.begin(kMinimums, "UNIFORM_BUFFER_OFFSET_ALIGNMENT");
            {
                const GLint a = get_i(P::UniformBufferOffsetAlignment);
                rec_.expect(a >= 1 && a <= 256, "UNIFORM_BUFFER_OFFSET_ALIGNMENT is at most 256 (Table 6.33: a maximum)",
                            std::format("got {}", a));
                std::cout << std::format("     UNIFORM_BUFFER_OFFSET_ALIGNMENT = {}\n", a);
            }

            rec_.begin(kMinimums, "MAX_VIEWPORT_DIMS");
            {
                const auto d = get_i4(P::MaxViewportDims);
                // 2.13.1: at least the visible dimensions of the display / window being rendered to.
                rec_.expect(d[0] >= env.width && d[1] >= env.height,
                            "MAX_VIEWPORT_DIMS >= the surface size (2.13.1)", std::format("got {}x{}", d[0], d[1]));
            }

            rec_.begin(kMinimums, "ALIASED_LINE_WIDTH/POINT_SIZE_RANGE");
            for (const auto& [pname, name] : {std::pair{P::AliasedLineWidthRange, "ALIASED_LINE_WIDTH_RANGE"},
                                              std::pair{P::AliasedPointSizeRange, "ALIASED_POINT_SIZE_RANGE"}})
            {
                std::array<GLfloat, 2> r{-1.0f, -1.0f};
                mgl::glGetFloatv(pname, r.data());
                rec_.expect(r[0] <= 1.0f && r[1] >= 1.0f, std::format("{} contains 1.0 (Table 6.28 minimum 1,1)", name),
                            std::format("got ({}, {})", r[0], r[1]));
                std::cout << std::format("     {} = ({}, {})\n", name, r[0], r[1]);
            }

            rec_.begin(kMinimums, "MAX_TEXTURE_LOD_BIAS");
            {
                const Values v = get_all(P::MaxTextureLodBias);
                rec_.expect(v.f[0] >= 2.0f, "MAX_TEXTURE_LOD_BIAS: GetFloatv >= 2.0", std::format("got {}", v.f[0]));
                const double frac = v.f[0] - std::floor(v.f[0]);
                if (frac != 0.5)  // a tie has no single "nearest integer"
                    rec_.expect(v.i[0] == std::lround(v.f[0]), "MAX_TEXTURE_LOD_BIAS: GetIntegerv = float rounded to nearest",
                                std::format("float={} int={}", v.f[0], v.i[0]));
                std::cout << std::format("     MAX_TEXTURE_LOD_BIAS = {} (as integer {})\n", v.f[0], v.i[0]);
            }
        }

        // =========================================================================================
        // ES 3.0 §6.1.2 data conversions with values chosen to tell the rules apart
        // =========================================================================================
        void conversions()
        {
            // Booleans read as integers / floats: TRUE -> 1 / 1.0, FALSE -> 0 / 0.0.
            rec_.begin(kConversions, "boolean -> int/float (COLOR_WRITEMASK)");
            mgl::glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
            verify_state(rec_, P::ColorWritemask, "COLOR_WRITEMASK after ColorMask(T,F,T,F)", Kind::Boolean, {1, 0, 1, 0});
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            rec_.begin(kConversions, "boolean -> int/float (DEPTH_WRITEMASK)");
            mgl::glDepthMask(GL_FALSE);
            verify_state(rec_, P::DepthWritemask, "DEPTH_WRITEMASK after DepthMask(FALSE)", Kind::Boolean, {0});
            mgl::glDepthMask(GL_TRUE);

            // Ordinary floating-point state: GetIntegerv rounds to nearest, GetBooleanv is FALSE iff zero.
            // 0.25 rounds to 0 but is still TRUE as a boolean; -2.75 rounds to -3.
            rec_.begin(kConversions, "float -> int rounding (POLYGON_OFFSET)");
            mgl::glPolygonOffset(0.25f, -2.75f);
            verify_state(rec_, P::PolygonOffsetFactor, "POLYGON_OFFSET_FACTOR=0.25", Kind::Float, {0.25});
            verify_state(rec_, P::PolygonOffsetUnits, "POLYGON_OFFSET_UNITS=-2.75", Kind::Float, {-2.75});
            const GLint factor_as_int = get_i(P::PolygonOffsetFactor);
            mgl::glPolygonOffset(0.0f, 0.0f);

            // SAMPLE_COVERAGE_VALUE is R+ state that is NOT in the §6.1.2 list of linearly mapped values.
            rec_.begin(kConversions, "float -> int rounding (SAMPLE_COVERAGE_VALUE)");
            mgl::glSampleCoverage(0.75f, GL_TRUE);
            verify_state(rec_, P::SampleCoverageValue, "SAMPLE_COVERAGE_VALUE=0.75", Kind::Float, {0.75});
            verify_state(rec_, P::SampleCoverageInvert, "SAMPLE_COVERAGE_INVERT=TRUE", Kind::Boolean, {1});
            mgl::glSampleCoverage(1.0f, GL_FALSE);

            // Colours are mapped linearly: f -> one of the two integers nearest f * (2^31 - 1).
            rec_.begin(kConversions, "color -> int linear (COLOR_CLEAR_VALUE)");
            mgl::glClearColor(1.0f, 0.5f, 0.25f, 0.0f);
            verify_state(rec_, P::ColorClearValue, "COLOR_CLEAR_VALUE=(1,0.5,0.25,0)", Kind::Normalized, {1.0, 0.5, 0.25, 0.0});
            const GLint clear_as_int = get_i4(P::ColorClearValue)[2];
            // "The specified components are stored as floating-point values" (4.2.3): the clear colour is clamped
            // only when a fixed-point buffer is cleared, so a negative value reads back unchanged. -0.5 is
            // inside [-1, 1], so its integer form is defined too (signed Equation 2.4).
            mgl::glClearColor(-0.5f, 0.75f, 0.0f, 1.0f);
            verify_state(rec_, P::ColorClearValue, "COLOR_CLEAR_VALUE=(-0.5,0.75,0,1) stored unclamped", Kind::Normalized,
                         {-0.5, 0.75, 0.0, 1.0});
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

            rec_.begin(kConversions, "color -> int linear (BLEND_COLOR)");
            mgl::glBlendColor(0.5f, 1.0f, 0.0f, 0.25f);
            verify_state(rec_, P::BlendColor, "BLEND_COLOR=(0.5,1,0,0.25)", Kind::Normalized, {0.5, 1.0, 0.0, 0.25});
            mgl::glBlendColor(0.0f, 0.0f, 0.0f, 0.0f);

            // DepthRangef values are mapped linearly and clamped to [0,1] when specified (2.13.1).
            rec_.begin(kConversions, "depth range -> int linear, clamped");
            mgl::glDepthRangef(0.25f, 0.75f);
            verify_state(rec_, P::DepthRange, "DEPTH_RANGE=(0.25,0.75)", Kind::Normalized, {0.25, 0.75});
            mgl::glDepthRangef(1.5f, -2.0f);
            verify_state(rec_, P::DepthRange, "DepthRangef(1.5,-2) clamped to (1,0)", Kind::Normalized, {1.0, 0.0});
            mgl::glDepthRangef(0.0f, 1.0f);

            // The depth clear value is mapped linearly and clamped to [0,1] (4.2.3).
            rec_.begin(kConversions, "depth clear -> int linear, clamped");
            mgl::glClearDepthf(0.5f);
            verify_state(rec_, P::DepthClearValue, "DEPTH_CLEAR_VALUE=0.5", Kind::Normalized, {0.5});
            mgl::glClearDepthf(3.0f);
            verify_state(rec_, P::DepthClearValue, "ClearDepthf(3) clamped to 1", Kind::Normalized, {1.0});
            mgl::glClearDepthf(1.0f);

            // Integers read as floats / booleans; zero is the only FALSE.
            rec_.begin(kConversions, "int -> float/bool (UNPACK_ALIGNMENT, STENCIL masks)");
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 8);
            verify_state(rec_, P::UnpackAlignment, "UNPACK_ALIGNMENT=8", Kind::Integer, {8});
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
            mgl::glStencilFunc(mgl::CompareFunc::Always, 5, 0u);
            verify_state(rec_, P::StencilValueMask, "STENCIL_VALUE_MASK=0", Kind::Integer, {0});
            verify_state(rec_, P::StencilRef, "STENCIL_REF=5", Kind::Integer, {5});
            mgl::glStencilFunc(mgl::CompareFunc::Always, 0, 0xFFFFFFFFu);

            // The two rules really differ: the same 0.25 is 0 as rounded state and ~2^29 as a colour.
            rec_.begin(kConversions, "rounding vs linear mapping differ");
            rec_.expect(factor_as_int == 0 && clear_as_int > (1 << 28),
                        "0.25 reads as 0 from POLYGON_OFFSET_FACTOR but as ~2^29 from COLOR_CLEAR_VALUE",
                        std::format("factor={} clear={}", factor_as_int, clear_as_int));

            // "If a value is so large in magnitude that it cannot be represented with the requested type, then
            // the nearest value representable using the requested type is returned."
            rec_.begin(kConversions, "64-bit -> 32-bit saturation");
            for (const auto& [pname, name] : {std::pair{P::MaxElementIndex, "MAX_ELEMENT_INDEX"},
                                              std::pair{P::MaxServerWaitTimeout, "MAX_SERVER_WAIT_TIMEOUT"}})
            {
                const GLint64 v64 = get_i64(pname);
                const GLint v32 = get_i(pname);
                const GLint64 sat = std::clamp<GLint64>(v64, std::numeric_limits<GLint>::min(), std::numeric_limits<GLint>::max());
                rec_.expect(v32 == sat, std::format("{}: GetIntegerv = GetInteger64v saturated to 32 bits", name),
                            std::format("i64={} i32={}", v64, v32));
                std::cout << std::format("     {} = {} (GetIntegerv {})\n", name, v64, v32);
            }
            rec_.expect(get_i64(P::MaxElementIndex) >= (1 << 24) - 1, "MAX_ELEMENT_INDEX >= 2^24-1 (Table 6.28)");
        }

        // =========================================================================================
        // Indexed state: ES 3.0 §6.1.1 (GetIntegeri_v / GetInteger64i_v), §6.1.9, §2.10.1.1
        // =========================================================================================
        void indexed_state()
        {
            const auto max_ubo = static_cast<GLuint>(get_i(P::MaxUniformBufferBindings));
            const auto max_tf = static_cast<GLuint>(get_i(P::MaxTransformFeedbackSeparateAttribs));
            const auto align = static_cast<GLintptr>(get_i(P::UniformBufferOffsetAlignment));

            const std::pair<mgl::BufferTarget, const char*> targets[] = {{mgl::BufferTarget::Uniform, "UNIFORM_BUFFER"},
                                                                         {mgl::BufferTarget::TransformFeedback, "TRANSFORM_FEEDBACK_BUFFER"}};
            for (const auto& [target, tname] : targets)
            {
                const bool ubo = target == mgl::BufferTarget::Uniform;
                const P binding = ubo ? P::UniformBufferBinding : P::TransformFeedbackBufferBinding;
                const P start = ubo ? P::UniformBufferStart : P::TransformFeedbackBufferStart;
                const P size = ubo ? P::UniformBufferSize : P::TransformFeedbackBufferSize;
                const GLuint count = ubo ? max_ubo : max_tf;

                rec_.begin(kIndexed, std::format("{} initial indexed state", tname));
                bool all_zero = true;
                for (GLuint i = 0; i < count; ++i)
                    all_zero = all_zero && get_ii(binding, i) == 0 && get_i64i(start, i) == 0 && get_i64i(size, i) == 0;
                rec_.expect(all_zero, std::format("{}_BINDING/START/SIZE are 0 for all {} indices (Tables 6.24/6.25)", tname, count));
                rec_.no_error(std::format("{}: indices 0..{} accepted", tname, count - 1));

                // Out-of-range index: INVALID_VALUE, and the output is untouched (2.5).
                rec_.begin(kIndexed, std::format("{} index out of range", tname));
                {
                    GLint v = kSentinel;
                    mgl::glGetIntegeri_v(binding, count, &v);
                    rec_.error(EC::InvalidValue, std::format("GetIntegeri_v({}_BINDING, {}) -> INVALID_VALUE", tname, count));
                    rec_.expect(v == kSentinel, "... and the output value is not modified", std::format("got {}", v));
                    GLint64 v64 = kSentinel64;
                    mgl::glGetInteger64i_v(start, count, &v64);
                    rec_.error(EC::InvalidValue, std::format("GetInteger64i_v({}_START, {}) -> INVALID_VALUE", tname, count));
                    rec_.expect(v64 == kSentinel64, "... and the output value is not modified", std::format("got {}", v64));
                    v64 = kSentinel64;
                    mgl::glGetInteger64i_v(size, count + 7, &v64);
                    rec_.error(EC::InvalidValue, std::format("GetInteger64i_v({}_SIZE, {}) -> INVALID_VALUE", tname, count + 7));
                    rec_.expect(v64 == kSentinel64, "... and the output value is not modified", std::format("got {}", v64));
                }

                // BindBufferRange / BindBufferBase set the indexed and the generic binding.
                rec_.begin(kIndexed, std::format("{} BindBufferRange/Base", tname));
                mgl::BufferId buf{};
                mgl::glGenBuffers(1, &buf);
                mgl::glBindBuffer(target, buf);
                const GLsizeiptr bytes = 2 * std::max<GLsizeiptr>(align, 16) + 256;
                mgl::glBufferData(target, bytes, nullptr, mgl::BufferUsage::DynamicDraw);
                mgl::glBindBuffer(target, mgl::BufferId{0});
                // UBO offsets must be multiples of UNIFORM_BUFFER_OFFSET_ALIGNMENT; TF offsets/sizes multiples of 4.
                const GLintptr offset = ubo ? align : 16;
                const GLsizeiptr range = ubo ? 64 : 32;
                const GLuint ri = 1, bi = count - 1;
                mgl::glBindBufferRange(target, ri, buf, offset, range);
                mgl::glBindBufferBase(target, bi, buf);
                rec_.no_error(std::format("BindBufferRange({}, {}, buf, {}, {}) and BindBufferBase({}, {}) succeed", tname, ri,
                                          offset, range, tname, bi));
                const auto b = static_cast<GLint>(buf.value);
                rec_.expect(get_ii(binding, ri) == b, std::format("GetIntegeri_v({}_BINDING, {}) = buffer", tname, ri),
                            std::format("got {}", get_ii(binding, ri)));
                rec_.expect(get_i64i(start, ri) == offset, std::format("GetInteger64i_v({}_START, {}) = {}", tname, ri, offset),
                            std::format("got {}", get_i64i(start, ri)));
                rec_.expect(get_i64i(size, ri) == range, std::format("GetInteger64i_v({}_SIZE, {}) = {}", tname, ri, range),
                            std::format("got {}", get_i64i(size, ri)));
                // A table entry listed for GetInteger64i_v may also be read with GetIntegeri_v and vice versa (6.2).
                rec_.expect(get_ii(start, ri) == offset && get_i64i(binding, ri) == b,
                            "the 32-bit and 64-bit indexed queries are interchangeable",
                            std::format("Integeri_v(START)={} Integer64i_v(BINDING)={}", get_ii(start, ri), get_i64i(binding, ri)));
                rec_.expect(get_ii(binding, bi) == b && get_i64i(start, bi) == 0 && get_i64i(size, bi) == 0,
                            std::format("BindBufferBase at index {}: binding = buffer, START = SIZE = 0 (6.1.9 footnote)", bi),
                            std::format("binding={} start={} size={}", get_ii(binding, bi), get_i64i(start, bi), get_i64i(size, bi)));
                rec_.expect(get_ii(binding, 0) == 0, std::format("{}_BINDING index 0 is untouched", tname));
                rec_.expect(get_i(binding) == b, std::format("the generic {}_BINDING is the buffer as well (2.10.1.1)", tname),
                            std::format("got {}", get_i(binding)));
                mgl::glBindBufferBase(target, ri, mgl::BufferId{0});
                mgl::glBindBufferBase(target, bi, mgl::BufferId{0});
                mgl::glBindBuffer(target, mgl::BufferId{0});
                rec_.expect(get_ii(binding, ri) == 0 && get_i64i(start, ri) == 0 && get_i64i(size, ri) == 0,
                            "after binding 0: BINDING/START/SIZE read 0 (\"no buffer bound\")");
                mgl::glDeleteBuffers(1, &buf);
                rec_.no_error("unbinding and deleting the buffer");
            }

            // glGetStringi(EXTENSIONS): every name in [0, NUM_EXTENSIONS) and only those of GetString(EXTENSIONS).
            rec_.begin(kIndexed, "GetStringi(EXTENSIONS)");
            {
                const GLint n = get_i(P::NumExtensions);
                const GLubyte* all_raw = mgl::glGetString(mgl::StringName::Extensions);
                const std::string all = all_raw ? reinterpret_cast<const char*>(all_raw) : "";
                std::istringstream tokens(all);
                std::vector<std::string> listed;
                for (std::string t; tokens >> t;) listed.push_back(t);
                bool all_found = true;
                std::string missing;
                for (GLint i = 0; i < n; ++i)
                {
                    const GLubyte* s = mgl::glGetStringi(mgl::StringName::Extensions, static_cast<GLuint>(i));
                    const std::string name = s ? reinterpret_cast<const char*>(s) : "";
                    if (name.empty() || std::find(listed.begin(), listed.end(), name) == listed.end())
                    {
                        all_found = false;
                        missing = name;
                    }
                }
                rec_.no_error("GetStringi(EXTENSIONS, 0..NUM_EXTENSIONS-1) accepted");
                rec_.expect(all_found && static_cast<GLint>(listed.size()) == n,
                            "the indexed names are exactly the names in GetString(EXTENSIONS) (6.1.6)",
                            std::format("NUM_EXTENSIONS={} string tokens={} unmatched='{}'", n, listed.size(), missing));
                std::cout << std::format("     NUM_EXTENSIONS = {}\n", n);
                const GLubyte* past = mgl::glGetStringi(mgl::StringName::Extensions, static_cast<GLuint>(n));
                rec_.error(EC::InvalidValue, "GetStringi(EXTENSIONS, NUM_EXTENSIONS) -> INVALID_VALUE");
                rec_.expect(past == nullptr, "... and returns NULL (\"if the generating command returns a value, it returns zero\")");
            }
        }

        // =========================================================================================
        // Hints (ES 3.0 §5.3): initial DONT_CARE, every mode round-trips; interpretation is free.
        // =========================================================================================
        void hints()
        {
            const std::pair<mgl::HintTarget, P> targets[] = {
                {mgl::HintTarget::GenerateMipmap, P::GenerateMipmapHint},
                {mgl::HintTarget::FragmentShaderDerivative, P::FragmentShaderDerivativeHint},
            };
            const std::pair<mgl::HintMode, GLint> modes[] = {
                {mgl::HintMode::Fastest, GL_FASTEST}, {mgl::HintMode::Nicest, GL_NICEST}, {mgl::HintMode::DontCare, GL_DONT_CARE}};
            for (const auto& [target, pname] : targets)
            {
                const char* name = pname == P::GenerateMipmapHint ? "GENERATE_MIPMAP_HINT" : "FRAGMENT_SHADER_DERIVATIVE_HINT";
                rec_.begin(kHints, std::format("{} round trip", name));
                for (const auto& [mode, value] : modes)
                {
                    mgl::glHint(target, mode);
                    rec_.no_error(std::format("glHint({}, {}) accepted", name, mgl::to_string(mode)));
                    const GLint got = get_i(pname);
                    rec_.expect(got == value, std::format("{} reads back {}", name, mgl::to_string(mode)),
                                std::format("got 0x{:X}", got));
                    rec_.expect(get_f(pname) == static_cast<GLfloat>(value), std::format("{} via GetFloatv = enum value", name));
                }
            }
            // INVALID_ENUM for glHint needs a target or mode outside Table 5.2; meta-gl's HintTarget/HintMode
            // only contain legal values, so that case is unreachable without a raw cast (META_GL_GAPS O1).
            std::cout << "     glHint INVALID_ENUM: not reachable through meta-gl's typed HintTarget/HintMode (O1)\n";
        }

        // =========================================================================================
        // DITHER (ES 3.0 §4.1.9): state, plus the rule that dithering may only choose between the two
        // representable neighbours of a colour, and that without dithering the choice must not depend on
        // the pixel position.
        // =========================================================================================
        void dither()
        {
            rec_.begin(kDither, "DITHER toggling");
            mgl::glDisable(mgl::Capability::Dither);
            verify_state(rec_, P::Dither, "DITHER after glDisable", Kind::Boolean, {0}, mgl::Capability::Dither);
            mgl::glEnable(mgl::Capability::Dither);
            verify_state(rec_, P::Dither, "DITHER after glEnable", Kind::Boolean, {1}, mgl::Capability::Dither);

            constexpr int kSize = 32;
            mgl::RenderbufferId rb{};
            mgl::FramebufferId fb{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, kSize, kSize);
            mgl::glGenFramebuffers(1, &fb);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fb);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer,
                                           mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rb);
            const bool complete = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete;

            // 0.5, 0.3, 0.7 have no exact 8-bit representation: each lies between two representable values.
            const std::array<float, 4> c = {0.5f, 0.3f, 0.7f, 0.5f};
            std::array<int, 4> lo{}, hi{};
            for (std::size_t k = 0; k < 4; ++k)
            {
                lo[k] = static_cast<int>(std::floor(c[k] * 255.0f));
                hi[k] = static_cast<int>(std::ceil(c[k] * 255.0f));
            }
            mgl::glViewport(0, 0, kSize, kSize);
            mgl::glClearColor(c[0], c[1], c[2], c[3]);
            for (const bool on : {false, true})
            {
                rec_.begin(kDither, on ? "clear with DITHER enabled" : "clear with DITHER disabled");
                if (!complete)
                {
                    rec_.expect(false, "RGBA8 renderbuffer framebuffer is complete");
                    continue;
                }
                if (on)
                    mgl::glEnable(mgl::Capability::Dither);
                else
                    mgl::glDisable(mgl::Capability::Dither);
                mgl::glClear(mgl::ClearBufferBit::Color);
                const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, kSize, kSize);
                bool neighbours = true, uniform = true;
                int differing = 0;
                for (std::size_t p = 0; p < px.size(); p += 4)
                {
                    bool same = true;
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        const int v = px[p + k];
                        neighbours = neighbours && (v == lo[k] || v == hi[k]);
                        same = same && px[p + k] == px[k];
                    }
                    uniform = uniform && same;
                    differing += !same;
                }
                rec_.expect(neighbours,
                            std::format("DITHER {}: every pixel is one of the two representable neighbours of (0.5,0.3,0.7,0.5)",
                                        on ? "on" : "off"),
                            std::format("first pixel ({},{},{},{})", px[0], px[1], px[2], px[3]));
                if (!on)
                    rec_.expect(uniform, "DITHER off: the choice does not depend on the pixel position (uniform image)",
                                std::format("{} pixels differ from pixel (0,0)", differing));
                std::cout << std::format("     DITHER {}: pixel(0,0)=({},{},{},{}), {} of {} pixels differ from it\n",
                                         on ? "on " : "off", px[0], px[1], px[2], px[3], differing, kSize * kSize);
                rec_.no_error("clear and readback");
            }
            mgl::glEnable(mgl::Capability::Dither);  // back to the initial state
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glDeleteFramebuffers(1, &fb);
            mgl::glDeleteRenderbuffers(1, &rb);
        }

        // =========================================================================================
        // Error semantics (ES 3.0 §2.5)
        // =========================================================================================
        void error_semantics()
        {
            rec_.begin(kErrors, "GetError with no error");
            rec_.expect(mgl::glGetError() == EC::NoError && mgl::glGetError() == EC::NoError,
                        "glGetError returns NO_ERROR when no error occurred (and keeps doing so)");

            // Sticky: the flag stays set across later, valid commands until glGetError reads it; those
            // later commands still execute.
            rec_.begin(kErrors, "error flag is sticky");
            {
                mgl::glLineWidth(-1.0f);  // INVALID_VALUE (3.5)
                mgl::glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
                std::array<GLint, 4> vp{};
                mgl::glGetIntegerv(P::Viewport, vp.data());
                (void)mgl::glIsEnabled(mgl::Capability::Blend);
                std::array<GLfloat, 4> cc{};
                mgl::glGetFloatv(P::ColorClearValue, cc.data());
                const EC first = mgl::glGetError();
                rec_.expect(first == EC::InvalidValue,
                            "INVALID_VALUE from LineWidth(-1) survives later valid commands until glGetError",
                            std::format("got {}", demo::error_name(first)));
                rec_.expect(cc[0] == 0.25f && cc[1] == 0.5f && cc[2] == 0.75f && cc[3] == 1.0f,
                            "commands after the error still execute (ClearColor took effect)");
                rec_.expect(mgl::glGetError() == EC::NoError, "reading the error cleared the flag");
                mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            }

            // The first error is recorded; later errors "do not affect this recorded code". An implementation
            // with several flag-code pairs may report the later codes too, in unspecified order.
            rec_.begin(kErrors, "first error is kept");
            {
                mgl::TextureId tex{};
                mgl::glGenTextures(1, &tex);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
                (void)demo::drain_gl_errors();

                mgl::glLineWidth(0.0f);                                         // INVALID_VALUE
                mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex);     // INVALID_OPERATION
                GLint dummy = 0;
                mgl::glGetIntegeri_v(P::DepthFunc, 0, &dummy);                   // INVALID_ENUM
                std::vector<EC> seq;
                for (int i = 0; i < 8; ++i)
                {
                    const EC e = mgl::glGetError();
                    if (e == EC::NoError) break;
                    seq.push_back(e);
                }
                std::string s;
                for (const EC e : seq) s += std::string(demo::error_name(e)) + " ";
                const bool has_value = std::find(seq.begin(), seq.end(), EC::InvalidValue) != seq.end();
                const bool only_known = std::all_of(seq.begin(), seq.end(), [](EC e) {
                    return e == EC::InvalidValue || e == EC::InvalidOperation || e == EC::InvalidEnum;
                });
                rec_.expect(has_value && only_known && seq.size() <= 3,
                            "after VALUE, OPERATION, ENUM errors the first (INVALID_VALUE) is reported, nothing foreign",
                            std::format("sequence: {}", s));
                rec_.expect(mgl::glGetError() == EC::NoError, "all flags are reset once NO_ERROR is returned");
                std::cout << std::format("     error sequence after three errors: {}\n", s);
                mgl::glDeleteTextures(1, &tex);
            }

            // "The command generating the error is ignored so that it has no effect on GL state."
            rec_.begin(kErrors, "Viewport(w<0) ignored");
            {
                mgl::glViewport(3, 4, 50, 60);
                mgl::glViewport(1, 2, -3, 4);
                rec_.error(EC::InvalidValue, "glViewport with negative width -> INVALID_VALUE (2.13.1)");
                const auto vp = get_i4(P::Viewport);
                rec_.expect(vp == std::array<GLint, 4>{3, 4, 50, 60}, "VIEWPORT is unchanged",
                            std::format("got {},{},{},{}", vp[0], vp[1], vp[2], vp[3]));
            }
            rec_.begin(kErrors, "Scissor(h<0) ignored");
            {
                mgl::glScissor(5, 6, 7, 8);
                mgl::glScissor(1, 2, 3, -4);
                rec_.error(EC::InvalidValue, "glScissor with negative height -> INVALID_VALUE (4.1.2)");
                const auto sb = get_i4(P::ScissorBox);
                rec_.expect(sb == std::array<GLint, 4>{5, 6, 7, 8}, "SCISSOR_BOX is unchanged",
                            std::format("got {},{},{},{}", sb[0], sb[1], sb[2], sb[3]));
            }
            rec_.begin(kErrors, "LineWidth(<=0) ignored");
            {
                mgl::glLineWidth(0.0f);
                rec_.error(EC::InvalidValue, "glLineWidth(0) -> INVALID_VALUE (3.5)");
                mgl::glLineWidth(-2.0f);
                rec_.error(EC::InvalidValue, "glLineWidth(-2) -> INVALID_VALUE");
                rec_.expect(get_f(P::LineWidth) == 1.0f, "LINE_WIDTH is still 1.0");
            }
            rec_.begin(kErrors, "PixelStorei out of range ignored");
            {
                mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 3);
                rec_.error(EC::InvalidValue, "UNPACK_ALIGNMENT=3 (not 1,2,4,8) -> INVALID_VALUE (3.7.1)");
                rec_.expect(get_i(P::UnpackAlignment) == 4, "UNPACK_ALIGNMENT is still 4");
                mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, -1);
                rec_.error(EC::InvalidValue, "PACK_SKIP_ROWS=-1 -> INVALID_VALUE (4.3.2)");
                rec_.expect(get_i(P::PackSkipRows) == 0, "PACK_SKIP_ROWS is still 0");
            }
            rec_.begin(kErrors, "BindTexture wrong target ignored");
            {
                mgl::TextureId tex{};
                mgl::glGenTextures(1, &tex);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
                mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex);
                rec_.error(EC::InvalidOperation, "binding a 2D texture to TEXTURE_CUBE_MAP -> INVALID_OPERATION (3.8.1)");
                rec_.expect(get_i(P::TextureBindingCubeMap) == 0, "TEXTURE_BINDING_CUBE_MAP is unchanged (0)");
                rec_.expect(get_i(P::TextureBinding2D) == static_cast<GLint>(tex.value), "TEXTURE_BINDING_2D is unchanged");
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
                mgl::glDeleteTextures(1, &tex);
            }
            rec_.begin(kErrors, "BindVertexArray(deleted) ignored");
            {
                std::array<mgl::VertexArrayId, 2> vao{};
                mgl::glGenVertexArrays(2, vao.data());
                mgl::glDeleteVertexArrays(1, &vao[1]);
                mgl::glBindVertexArray(vao[0]);
                mgl::glBindVertexArray(vao[1]);
                rec_.error(EC::InvalidOperation, "glBindVertexArray(a deleted name) -> INVALID_OPERATION (2.11)");
                rec_.expect(get_i(P::VertexArrayBinding) == static_cast<GLint>(vao[0].value),
                            "VERTEX_ARRAY_BINDING still names the previously bound VAO");
                mgl::glBindVertexArray(mgl::VertexArrayId{0});
                mgl::glDeleteVertexArrays(1, &vao[0]);
            }

            mgl::BufferId buf{};
            mgl::glGenBuffers(1, &buf);
            rec_.begin(kErrors, "BindBufferBase(ARRAY_BUFFER) ignored");
            {
                mgl::glBindBufferBase(mgl::BufferTarget::Array, 0, buf);
                rec_.error(EC::InvalidEnum, "glBindBufferBase(ARRAY_BUFFER, ...) -> INVALID_ENUM (2.10.1.1: only TF/UNIFORM)");
                rec_.expect(get_i(P::ArrayBufferBinding) == 0, "ARRAY_BUFFER_BINDING is unchanged (0)");
            }
            rec_.begin(kErrors, "GetIntegeri_v(non-indexed) ignored");
            {
                GLint v = kSentinel;
                mgl::glGetIntegeri_v(P::DepthFunc, 0, &v);
                rec_.error(EC::InvalidEnum, "glGetIntegeri_v(DEPTH_FUNC, 0) -> INVALID_ENUM (not indexed state)");
                rec_.expect(v == kSentinel, "... and the output value is not modified", std::format("got {}", v));
                GLint64 v64 = kSentinel64;
                mgl::glGetInteger64i_v(P::LineWidth, 0, &v64);
                rec_.error(EC::InvalidEnum, "glGetInteger64i_v(LINE_WIDTH, 0) -> INVALID_ENUM");
                rec_.expect(v64 == kSentinel64, "... and the output value is not modified", std::format("got {}", v64));
            }
            rec_.begin(kErrors, "GetStringi(RENDERER) returns NULL");
            {
                const GLubyte* s = mgl::glGetStringi(mgl::StringName::Renderer, 0);
                rec_.error(EC::InvalidEnum, "glGetStringi(RENDERER, 0) -> INVALID_ENUM (name may only be EXTENSIONS, 6.1.6)");
                rec_.expect(s == nullptr, "... and returns NULL (zero)");
            }
            rec_.begin(kErrors, "MapBufferRange past the end");
            {
                mgl::glBindBuffer(mgl::BufferTarget::Array, buf);
                mgl::glBufferData(mgl::BufferTarget::Array, 64, nullptr, mgl::BufferUsage::StaticDraw);
                void* p = mgl::glMapBufferRange(mgl::BufferTarget::Array, 0, 65, mgl::MapBufferAccessMask::Read);
                rec_.error(EC::InvalidValue, "glMapBufferRange(offset+length > BUFFER_SIZE) -> INVALID_VALUE (2.10.3)");
                rec_.expect(p == nullptr, "... and returns NULL");
                rec_.expect(buffer_param(mgl::BufferTarget::Array, mgl::BufferParameter::Mapped) == GL_FALSE,
                            "BUFFER_MAPPED is still FALSE");
            }
            rec_.begin(kErrors, "UnmapBuffer while unmapped");
            {
                const GLboolean r = mgl::glUnmapBuffer(mgl::BufferTarget::Array);
                rec_.error(EC::InvalidOperation, "glUnmapBuffer on an unmapped buffer -> INVALID_OPERATION (2.10.3)");
                // "If the buffer data store is already in the unmapped state, UnmapBuffer returns FALSE" (2.10.3).
                // Mesa's glthread (on by default for radeonsi) runs UnmapBuffer asynchronously and returns TRUE;
                // the error itself is still raised. Same deviation as es30-buffer-mapping (KNOWN_LIMITATIONS.md).
                rec_.expect_or_deviation(r == GL_FALSE, "... and returns FALSE", std::format("returned {}", static_cast<int>(r)),
                                         "radeonsi",
                                         "Mesa glthread returns TRUE from glUnmapBuffer without waiting for the result "
                                         "(passes with mesa_glthread=false)");
                mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            }
            rec_.begin(kErrors, "BindBufferRange invalid ranges ignored");
            {
                const auto max_ubo = static_cast<GLuint>(get_i(P::MaxUniformBufferBindings));
                const GLint align = get_i(P::UniformBufferOffsetAlignment);
                mgl::glBindBuffer(mgl::BufferTarget::Uniform, buf);
                mgl::glBufferData(mgl::BufferTarget::Uniform, 1024, nullptr, mgl::BufferUsage::DynamicDraw);
                mgl::glBindBuffer(mgl::BufferTarget::Uniform, mgl::BufferId{0});

                mgl::glBindBufferRange(mgl::BufferTarget::Uniform, max_ubo, buf, 0, 16);
                rec_.error(EC::InvalidValue, "BindBufferRange(UNIFORM, MAX_UNIFORM_BUFFER_BINDINGS, ...) -> INVALID_VALUE");
                rec_.expect(get_i(P::UniformBufferBinding) == 0, "... the generic UNIFORM_BUFFER_BINDING is not changed either");
                mgl::glBindBufferRange(mgl::BufferTarget::Uniform, 0, buf, 0, 0);
                rec_.error(EC::InvalidValue, "BindBufferRange(UNIFORM, 0, buf, 0, size 0) -> INVALID_VALUE");
                rec_.expect(get_ii(P::UniformBufferBinding, 0) == 0 && get_i(P::UniformBufferBinding) == 0,
                            "... indexed binding 0 and the generic binding stay 0");
                if (align > 1)
                {
                    mgl::glBindBufferRange(mgl::BufferTarget::Uniform, 0, buf, 1, 16);
                    rec_.error(EC::InvalidValue, std::format("BindBufferRange offset 1, not a multiple of {} -> INVALID_VALUE (2.12.6.5)", align));
                    rec_.expect(get_ii(P::UniformBufferBinding, 0) == 0, "... indexed binding 0 stays 0");
                }
                mgl::glBindBufferRange(mgl::BufferTarget::TransformFeedback, 0, buf, 2, 16);
                rec_.error(EC::InvalidValue, "BindBufferRange(TRANSFORM_FEEDBACK, offset 2) -> INVALID_VALUE (2.15.2: multiple of 4)");
                mgl::glBindBufferRange(mgl::BufferTarget::TransformFeedback, 0, buf, 4, 6);
                rec_.error(EC::InvalidValue, "BindBufferRange(TRANSFORM_FEEDBACK, size 6) -> INVALID_VALUE");
                rec_.expect(get_ii(P::TransformFeedbackBufferBinding, 0) == 0 && get_i(P::TransformFeedbackBufferBinding) == 0,
                            "... TRANSFORM_FEEDBACK_BUFFER bindings stay 0");
            }
            mgl::glDeleteBuffers(1, &buf);
            // OUT_OF_MEMORY ("Unknown" whether the command is ignored) cannot be provoked deterministically and is
            // deliberately not tested.
        }

        // =========================================================================================
        // INVALID_FRAMEBUFFER_OPERATION (ES 3.0 §4.4.4.4)
        // =========================================================================================
        void incomplete_framebuffers()
        {
            using FS = mgl::FramebufferStatus;
            rec_.begin(kFramebuffer, "FBO without attachments");
            mgl::FramebufferId empty{};
            mgl::glGenFramebuffers(1, &empty);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, empty);
            const FS s0 = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            rec_.expect(s0 == FS::IncompleteMissingAttachment, "status FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT",
                        std::string(demo::framebuffer_status_name(s0)));
            rec_.no_error("glCheckFramebufferStatus itself raises no error");
            mgl::glClear(mgl::ClearBufferBit::Color);
            rec_.error(EC::InvalidFramebufferOperation, "glClear -> INVALID_FRAMEBUFFER_OPERATION");
            // "...even if RASTERIZER_DISCARD is enabled" - Clear is a rendering command (2.6).
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glClear(mgl::ClearBufferBit::Color);
            rec_.error(EC::InvalidFramebufferOperation, "glClear with RASTERIZER_DISCARD enabled -> still INVALID_FRAMEBUFFER_OPERATION");
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &empty);

            // A complete FBO made incomplete by a colour-format image on the depth attachment: the failing
            // clear and readback must have no effect on the colour image or on the client memory.
            rec_.begin(kFramebuffer, "incomplete attachment: clear and read ignored");
            constexpr int kSize = 8;
            std::array<mgl::RenderbufferId, 2> rb{};
            mgl::glGenRenderbuffers(2, rb.data());
            for (const auto r : rb)
            {
                mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, r);
                mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, kSize, kSize);
            }
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::FramebufferId fb{};
            mgl::glGenFramebuffers(1, &fb);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fb);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rb[0]);
            const FS s1 = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            rec_.expect(s1 == FS::Complete, "RGBA8 colour-only FBO is complete", std::string(demo::framebuffer_status_name(s1)));
            mgl::glViewport(0, 0, kSize, kSize);
            mgl::glClearColor(unorm(0), unorm(0), unorm(255), unorm(255));
            mgl::glClear(mgl::ClearBufferBit::Color);

            // 4.4.4.1: an image on DEPTH_ATTACHMENT must have a depth-renderable format.
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                           mgl::RenderbufferTarget::Renderbuffer, rb[1]);
            const FS s2 = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            rec_.expect(s2 == FS::IncompleteAttachment, "RGBA8 image on DEPTH_ATTACHMENT -> FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
                        std::string(demo::framebuffer_status_name(s2)));
            rec_.no_error("attaching the wrong image is not itself an error");
            mgl::glClearColor(unorm(255), unorm(0), unorm(0), unorm(255));
            mgl::glClear(mgl::ClearBufferBit::Color);
            rec_.error(EC::InvalidFramebufferOperation, "glClear on the incomplete FBO -> INVALID_FRAMEBUFFER_OPERATION");
            std::array<std::uint8_t, 4> client{11, 22, 33, 44};
            mgl::glReadPixels(0, 0, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, client.data());
            rec_.error(EC::InvalidFramebufferOperation, "glReadPixels from the incomplete FBO -> INVALID_FRAMEBUFFER_OPERATION");
            rec_.expect(client == std::array<std::uint8_t, 4>{11, 22, 33, 44}, "... and the client memory is not written");

            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth,
                                           mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            const FS s3 = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            rec_.expect(s3 == FS::Complete, "complete again after detaching the depth image", std::string(demo::framebuffer_status_name(s3)));
            const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, kSize, kSize);
            bool blue = true;
            for (std::size_t p = 0; p < px.size(); p += 4)
                blue = blue && px[p] == 0 && px[p + 1] == 0 && px[p + 2] == 255 && px[p + 3] == 255;
            rec_.expect(blue, "the colour image still holds the earlier blue clear: the failed clear had no effect",
                        std::format("pixel 0 = ({},{},{},{})", px[0], px[1], px[2], px[3]));
            rec_.no_error("readback of the complete FBO");

            mgl::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fb);
            mgl::glDeleteRenderbuffers(2, rb.data());
        }

        // =========================================================================================
        // Dashboard
        // =========================================================================================
        static Cell cell(int i, int n, int width, int height)
        {
            // Columns chosen so that cells are roughly square for the surface aspect ratio.
            const int cols = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n) * width / std::max(height, 1)))));
            const int rows = (n + cols - 1) / cols;
            const int c = i % cols, r = i / cols;  // row 0 at the top
            Cell cell;
            cell.outer.x = c * width / cols;
            cell.outer.w = (c + 1) * width / cols - cell.outer.x;
            const int y_top = r * height / rows, y_bottom = (r + 1) * height / rows;
            cell.outer.y = height - y_bottom;
            cell.outer.h = y_bottom - y_top;
            const int t = std::max(1, std::min(cell.outer.w, cell.outer.h) / 8);  // frame thickness
            cell.inner = {cell.outer.x + t, cell.outer.y + t, cell.outer.w - 2 * t, cell.outer.h - 2 * t};
            return cell;
        }

        demo::Rgba8 cell_color(int i) const
        {
            if (i == forced_fail_) return kFailColor;
            switch (rec_.status(rec_.items()[static_cast<std::size_t>(i)]))
            {
            case Status::Pass: return kPassColor;
            case Status::Fail: return kFailColor;
            case Status::KnownDeviation: return kDeviationColor;
            }
            return kFailColor;
        }

        static void clear_to(demo::Rgba8 c)
        {
            mgl::glClearColor(unorm(c[0]), unorm(c[1]), unorm(c[2]), unorm(c[3]));
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        int first_item_of(int group) const
        {
            for (std::size_t i = 0; i < rec_.items().size(); ++i)
                if (rec_.items()[i].group == group) return static_cast<int>(i);
            return 0;
        }

        void verify_dashboard(const demo::Env& env, demo::Check& check, std::string_view what) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, env.width, env.height);
            const auto at = [&](int x, int y) {
                const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4;
                return demo::Rgba8{px[o], px[o + 1], px[o + 2], px[o + 3]};
            };
            const auto near = [](demo::Rgba8 a, demo::Rgba8 b) {
                for (std::size_t k = 0; k < 4; ++k)
                    if (std::abs(int(a[k]) - int(b[k])) > 1) return false;
                return true;
            };
            const auto n = static_cast<int>(rec_.items().size());
            for (int g = 0; g < static_cast<int>(kGroups.size()); ++g)
            {
                int cells = 0, bad = 0;
                std::string first_bad;
                for (int i = 0; i < n; ++i)
                {
                    if (rec_.items()[static_cast<std::size_t>(i)].group != g) continue;
                    ++cells;
                    const Cell c = cell(i, n, env.width, env.height);
                    const demo::Rgba8 centre = at(c.inner.x + c.inner.w / 2, c.inner.y + c.inner.h / 2);
                    const demo::Rgba8 inner_corner = at(c.inner.x, c.inner.y + c.inner.h - 1);
                    const demo::Rgba8 frame = at(c.outer.x, c.outer.y);
                    const bool ok = near(centre, cell_color(i)) && near(inner_corner, cell_color(i)) &&
                                    near(frame, kGroups[static_cast<std::size_t>(g)].tint);
                    if (!ok && bad++ == 0)
                        first_bad = std::format("cell {} '{}': centre {} frame {}", i, rec_.items()[static_cast<std::size_t>(i)].label,
                                                demo::format_rgba(centre), demo::format_rgba(frame));
                }
                if (cells == 0) continue;
                check.expect(bad == 0,
                             std::format("{}: all {} '{}' cells show their status colour inside the group frame (+-1)", what,
                                         cells, kGroups[static_cast<std::size_t>(g)].name),
                             first_bad);
            }
            // The unused cells after the last item keep the background colour.
            const Cell last = cell(n - 1, n, env.width, env.height);
            if (last.outer.x + last.outer.w < env.width)
            {
                const int x = (last.outer.x + last.outer.w + env.width) / 2;
                check.expect(near(at(x, last.outer.y + last.outer.h / 2), kBackground),
                             std::format("{}: area after the last cell is background", what));
            }
        }

        Recorder rec_;
        int forced_fail_ = -1;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-state-queries-errors",
        .title = "State queries, conversions and GL errors",
        .min_es = {3, 0},
        .summary = "Initial values of the ES 3.0 state tables through all four Get commands, the 6.1.2 conversion rules, "
                   "indexed queries, hints, dithering and the 2.5 error model, shown as a pass/fail dashboard.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<StateQueriesErrors>(argc, argv, info);
}
