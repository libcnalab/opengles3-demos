// es31-framebuffer-no-attachments
//
// A framebuffer object that has NO attachment at all (OpenGL ES 3.1, ES 3.2 sections 9.2.1 and 9.4.2): its
// size and sample count come from the parameters set with glFramebufferParameteri, rasterization still
// happens, and the fragment shader's side effects (atomics on a shader storage buffer, image stores) are the
// only way to observe it. The demo draws into such a framebuffer and compares the fragments that arrived
// with a CPU model, exactly:
//   * how many fragments, and which pixel each one belongs to (viewport, scissor, the 0..width x 0..height
//     rectangle of the framebuffer, triangles with vertices on the 1/16 pixel grid, a 33 x 17 and a
//     MAX_FRAMEBUFFER_WIDTH x HEIGHT framebuffer, a 4-sample framebuffer),
//   * that attachments override the default size and that detaching them restores it,
//   * glFramebufferParameteri / glGetFramebufferParameteriv: initial values, completeness rules, every error,
//   * layout(early_fragment_tests): in a depth-tested framebuffer the invocations that reach the shader are
//     exactly the visible fragments (and with the late tests the visible fragments are among them),
//   * gl_HelperInvocation: the 2 x 2 quads that are only partly covered need helper invocations for the
//     derivatives, yet no store or atomic of a helper changes memory: the counters are exact.
//
// The window (256 x 256) is a contact sheet of eight of the hit maps, measured on the GPU: white = one fragment,
// dark = none, yellow/orange = several, red = a pixel that disagrees with the CPU model (none, when it passes).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

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

    constexpr int kW = 64;  // the default size used by most runs
    constexpr int kH = 32;
    constexpr int kGuard = 32;  // extra columns and rows of the hit map: a fragment outside the framebuffer would land there
    constexpr std::uint32_t kUnset = 0xFFFFFFFFu;

    constexpr std::size_t kCountHeader = 5;   // words before the hit map in count.frag's buffer: total, min_x, min_y, max_x, max_y
    constexpr std::size_t kHelperHeader = 4;  // helper.frag's buffer: guarded, unguarded, helper_seen, deriv_bad

    struct Box
    {
        int x, y, w, h;
    };

    struct WinRect
    {
        int x0, y0, x1, y1;  // pixels [x0, x1) x [y0, y1)
    };

    struct Ndc
    {
        float x0, y0, x1, y1;
    };

    /// NDC rectangle that covers exactly the window rectangle `r` when `vp` is the viewport.
    Ndc ndc_for(const Box& vp, const WinRect& r)
    {
        const auto cx = [&](int x) { return static_cast<float>((x - vp.x) * 2.0 / vp.w - 1.0); };
        const auto cy = [&](int y) { return static_cast<float>((y - vp.y) * 2.0 / vp.h - 1.0); };
        return {cx(r.x0), cy(r.y0), cx(r.x1), cy(r.y1)};
    }

    /// The window rectangle an NDC rectangle covers under `vp`. Every case of this demo is pixel aligned.
    WinRect window_rect(const Box& vp, const Ndc& q)
    {
        const auto wx = [&](float v) { return vp.x + (static_cast<double>(v) + 1.0) / 2.0 * vp.w; };
        const auto wy = [&](float v) { return vp.y + (static_cast<double>(v) + 1.0) / 2.0 * vp.h; };
        const std::array<double, 4> v = {wx(q.x0), wy(q.y0), wx(q.x1), wy(q.y1)};
        for (const double d : v)
            if (std::abs(d - std::round(d)) > 1e-6) throw demo::Failure("test design: a rectangle edge is not on a pixel boundary");
        return {static_cast<int>(std::lround(v[0])), static_cast<int>(std::lround(v[1])), static_cast<int>(std::lround(v[2])), static_cast<int>(std::lround(v[3]))};
    }

    // ---------------------------------------------------------------------------------------------
    // CPU models. A hit map is row-major (y * width + x), one count per pixel.
    // ---------------------------------------------------------------------------------------------
    using HitMap = std::vector<std::uint32_t>;

    bool in_rect(const WinRect& r, int x, int y) { return x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1; }

    WinRect box_rect(const Box& b) { return {b.x, b.y, b.x + b.w, b.y + b.h}; }

    /// Fragments of an axis-aligned rectangle (ES 3.2 sections 13.5-13.7 rasterization, 13.8.2 scissor, 9.2 "no attachments":
    /// rendering is limited to (0,0)-(width,height)). Pixel centres are strictly inside or outside, so the rule is exact.
    HitMap model_quad(int fw, int fh, const WinRect& rect, const std::optional<Box>& scissor)
    {
        HitMap hits(static_cast<std::size_t>(fw * fh), 0);
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x)
                if (in_rect(rect, x, y) && (!scissor || in_rect(box_rect(*scissor), x, y))) hits[static_cast<std::size_t>(y * fw + x)] = 1;
        return hits;
    }

    // Triangles with vertices on the 1/16 pixel grid, in sixteenths of a pixel. Exactly representable in NDC for a 64 x 32 viewport
    // and exactly on the subpixel grid of any implementation with at least 4 subpixel bits (ES 3.2 table 21.40: SUBPIXEL_BITS >= 4).
    struct Tri
    {
        std::array<std::array<int, 2>, 3> v;
    };

    constexpr std::array<Tri, 2> kTriangles = {{
        {{{{84, 56}, {940, 146}, {358, 465}}}},       // (5.25, 3.5) (58.75, 9.125) (22.375, 29.0625)
        {{{{33, 488}, {991, 231}, {991, 271}}}},      // (2.0625, 30.5) (61.9375, 14.4375) (61.9375, 16.9375): a thin sliver
    }};

    struct TriModel
    {
        HitMap hits;
        int on_edge = 0;  // pixel centres exactly on an edge (excluded from the map; the triangles are chosen so that there are none)
    };

    TriModel model_triangles(const std::vector<Tri>& tris, int fw, int fh, const std::optional<Box>& scissor)
    {
        TriModel m;
        m.hits.assign(static_cast<std::size_t>(fw * fh), 0);
        for (const Tri& t : tris)
        {
            std::array<std::array<int, 2>, 3> v = t.v;
            const long long area = static_cast<long long>(v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - static_cast<long long>(v[1][1] - v[0][1]) * (v[2][0] - v[0][0]);
            if (area < 0) std::swap(v[1], v[2]);
            for (int y = 0; y < fh; ++y)
                for (int x = 0; x < fw; ++x)
                {
                    if (scissor && !in_rect(box_rect(*scissor), x, y)) continue;
                    const long long px = 16LL * x + 8, py = 16LL * y + 8;  // the pixel centre in sixteenths
                    bool all_positive = true, any_negative = false, any_zero = false;
                    for (std::size_t i = 0; i < 3; ++i)
                    {
                        const auto& p = v[i];
                        const auto& q = v[(i + 1) % 3];
                        const long long e = static_cast<long long>(q[0] - p[0]) * (py - p[1]) - static_cast<long long>(q[1] - p[1]) * (px - p[0]);
                        all_positive = all_positive && e > 0;
                        any_negative = any_negative || e < 0;
                        any_zero = any_zero || e == 0;
                    }
                    if (all_positive) ++m.hits[static_cast<std::size_t>(y * fw + x)];
                    else if (any_zero && !any_negative) ++m.on_edge;
                }
        }
        return m;
    }

    /// Uncovered pixels that sit in a 2 x 2 quad (aligned to even coordinates) with at least one covered pixel: the pixels for which
    /// an implementation that shades quads must create helper invocations.
    int helper_candidates(const HitMap& hits, int fw, int fh)
    {
        int n = 0;
        for (int qy = 0; qy < fh; qy += 2)
            for (int qx = 0; qx < fw; qx += 2)
            {
                int covered = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        covered += (qy + dy < fh && qx + dx < fw && hits[static_cast<std::size_t>((qy + dy) * fw + qx + dx)] != 0) ? 1 : 0;
                if (covered > 0) n += 4 - covered;
            }
        return n;
    }

    std::uint64_t sum(const HitMap& m)
    {
        std::uint64_t s = 0;
        for (const std::uint32_t v : m) s += v;
        return s;
    }

    // ---------------------------------------------------------------------------------------------
    // What the GPU reported for one draw.
    // ---------------------------------------------------------------------------------------------
    struct Observed
    {
        std::uint32_t total = 0, min_x = 0, min_y = 0, max_x = 0, max_y = 0;
        std::uint32_t guarded = 0, unguarded = 0, helper_seen = 0, deriv_bad = 0;  // the helper shader's counters
        std::vector<std::uint32_t> cells;  // stride * rows
        int stride = 0;
    };

    struct Case
    {
        std::string name;
        int fw = 0, fh = 0;
        HitMap expected;  // fw * fh
        Observed observed;
    };

    struct MapCompare
    {
        int mismatches = 0;
        std::uint64_t guard_hits = 0;
        std::string first;
    };

    /// Exact comparison of the hit map inside the framebuffer; every hit outside it (the guard region) is counted.
    MapCompare compare_map(const Case& c)
    {
        MapCompare r;
        const Observed& o = c.observed;
        const int rows = static_cast<int>(o.cells.size()) / o.stride;
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < o.stride; ++x)
            {
                const std::uint32_t got = o.cells[static_cast<std::size_t>(y * o.stride + x)];
                if (x >= c.fw || y >= c.fh)
                {
                    r.guard_hits += got;
                    continue;
                }
                const std::uint32_t want = c.expected[static_cast<std::size_t>(y * c.fw + x)];
                if (got != want && r.mismatches++ == 0) r.first = std::format("first at ({},{}): {} hits, expected {}", x, y, got, want);
            }
        return r;
    }

    struct Bounds
    {
        std::uint32_t min_x = kUnset, min_y = kUnset, max_x = 0, max_y = 0;
    };

    Bounds model_bounds(const Case& c)
    {
        Bounds b;
        for (int y = 0; y < c.fh; ++y)
            for (int x = 0; x < c.fw; ++x)
                if (c.expected[static_cast<std::size_t>(y * c.fw + x)] != 0)
                {
                    b.min_x = std::min(b.min_x, static_cast<std::uint32_t>(x));
                    b.min_y = std::min(b.min_y, static_cast<std::uint32_t>(y));
                    b.max_x = std::max(b.max_x, static_cast<std::uint32_t>(x));
                    b.max_y = std::max(b.max_y, static_cast<std::uint32_t>(y));
                }
        return b;
    }

    // ---------------------------------------------------------------------------------------------
    // The contact sheet: tile colors from a hit map.
    // ---------------------------------------------------------------------------------------------
    using Rgba = std::array<std::uint8_t, 4>;

    Rgba hit_color(std::uint32_t hits, bool wrong)
    {
        if (wrong) return {255, 0, 0, 255};
        switch (hits)
        {
        case 0: return {22, 22, 34, 255};
        case 1: return {240, 240, 240, 255};
        case 2: return {255, 205, 0, 255};
        default: return {255, 120, 0, 255};
        }
    }

    // ---------------------------------------------------------------------------------------------
    class NoAttachments final : public demo::Demo
    {
    public:
        // The programs and the uniforms each one uses.
        struct Prog
        {
            mgl::ProgramId id{};
            mgl::UniformLocation rect{}, pos{}, depth{}, stride{};
        };

        void init(demo::Env& env) override
        {
            prog_rect_ = make_prog(demo::build_program(env, "shaders/rect.vert", "shaders/count.frag"));
            prog_rect_early_ = make_prog(demo::build_program(env, "shaders/rect.vert", "shaders/count_early.frag"));
            prog_tri_ = make_prog(demo::build_program(env, "shaders/triangles.vert", "shaders/count.frag"));
            prog_tri_helper_ = make_prog(demo::build_program(env, "shaders/triangles.vert", "shaders/helper.frag"));
            prog_rect_helper_ = make_prog(demo::build_program(env, "shaders/rect.vert", "shaders/helper.frag"));
            prog_tri_image_ = make_prog(demo::build_program(env, "shaders/triangles.vert", "shaders/count_image.frag"));

            bad_source_ = env.load_text("shaders/bad_early_depth.frag");
            rect_vert_source_ = env.load_text("shaders/rect.vert");
            max_fb_width_ = demo::get_integer(mgl::GetParameter::MaxFramebufferWidth);
            max_fb_height_ = demo::get_integer(mgl::GetParameter::MaxFramebufferHeight);
            max_fb_samples_ = demo::get_integer(mgl::GetParameter::MaxFramebufferSamples);
            std::array<GLint, 2> dims{};
            mgl::glGetIntegerv(mgl::GetParameter::MaxViewportDims, dims.data());
            max_viewport_ = {dims[0], dims[1]};

            mgl::glGenBuffers(1, &ssbo_);
            mgl::glGenFramebuffers(static_cast<GLsizei>(fbos_.size()), fbos_.data());
            mgl::glGenRenderbuffers(static_cast<GLsizei>(rbos_.size()), rbos_.data());
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDisable(mgl::Capability::Dither);

            run_experiments();
            build_tiles();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glClearColor(0.08f, 0.08f, 0.12f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            const int tile_w = env.width / 2, tile_h = env.height / 4;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo_read_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            for (std::size_t i = 0; i < tile_tex_.size(); ++i)
            {
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2D, tile_tex_[i], 0);
                const int col = static_cast<int>(i % 2), row = static_cast<int>(i / 2);
                mgl::glBlitFramebuffer(0, 0, kW, kH, col * tile_w + 2, row * tile_h + 2, (col + 1) * tile_w - 2, (row + 1) * tile_h - 2,
                                       mgl::ClearBufferBit::Color, mgl::BlitFilter::Nearest);
            }
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            check_limits(check);
            check_parameters_and_completeness(check);
            check_parameter_errors(check);
            check_rasterization(check);
            check_sizes(check);
            check_multisample(check);
            check_images(check);
            check_early_fragment_tests(check);
            check_helper_invocations(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            for (const Prog* p : {&prog_rect_, &prog_rect_early_, &prog_tri_, &prog_tri_helper_, &prog_rect_helper_, &prog_tri_image_}) mgl::glDeleteProgram(p->id);
            mgl::glDeleteBuffers(1, &ssbo_);
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(fbos_.size()), fbos_.data());
            mgl::glDeleteRenderbuffers(static_cast<GLsizei>(rbos_.size()), rbos_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tile_tex_.size()), tile_tex_.data());
            mgl::glDeleteFramebuffers(1, &fbo_read_);
            if (!extra_fbos_.empty()) mgl::glDeleteFramebuffers(static_cast<GLsizei>(extra_fbos_.size()), extra_fbos_.data());
        }

    private:
        // ------------------------------------------------------------------------------------------
        // Setup helpers.
        // ------------------------------------------------------------------------------------------
        static Prog make_prog(mgl::ProgramId id)
        {
            Prog p;
            p.id = id;
            p.rect = mgl::glGetUniformLocation(id, "u_rect");
            p.pos = mgl::glGetUniformLocation(id, "u_pos");
            p.depth = mgl::glGetUniformLocation(id, "u_depth");
            p.stride = mgl::glGetUniformLocation(id, "u_stride");
            return p;
        }

        static void bind_fbo(mgl::FramebufferId f) { mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, f); }

        static void set_param(mgl::FramebufferDefaultParameter p, int v) { mgl::glFramebufferParameteri(mgl::FramebufferTarget::Framebuffer, p, v); }

        static int get_param(mgl::FramebufferDefaultParameter p)
        {
            GLint v = -12345;
            mgl::glGetFramebufferParameteriv(mgl::FramebufferTarget::Framebuffer, p, &v);
            return static_cast<int>(v);
        }

        /// Gives the framebuffer bound to GL_FRAMEBUFFER a default size, samples and sample-location mode.
        static void configure_default(int w, int h, int samples = 0, bool fixed = false)
        {
            set_param(mgl::FramebufferDefaultParameter::Width, w);
            set_param(mgl::FramebufferDefaultParameter::Height, h);
            set_param(mgl::FramebufferDefaultParameter::Samples, samples);
            set_param(mgl::FramebufferDefaultParameter::FixedSampleLocations, fixed ? 1 : 0);
        }

        static void require_complete(std::string_view what)
        {
            const mgl::FramebufferStatus s = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (s != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{}: framebuffer is not complete ({})", what, demo::framebuffer_status_name(s)));
        }

        // ------------------------------------------------------------------------------------------
        // One draw, observed through the shader storage buffer.
        // ------------------------------------------------------------------------------------------
        struct Draw
        {
            Box viewport{0, 0, kW, kH};
            std::optional<Box> scissor;
            float depth = 0.5f;
            bool record_cells = true;
        };

        void begin_draw(const Draw& d)
        {
            mgl::glViewport(d.viewport.x, d.viewport.y, d.viewport.w, d.viewport.h);
            if (d.scissor)
            {
                mgl::glEnable(mgl::Capability::ScissorTest);
                mgl::glScissor(d.scissor->x, d.scissor->y, d.scissor->w, d.scissor->h);
            }
        }

        static void end_draw(const Draw& d)
        {
            if (d.scissor) mgl::glDisable(mgl::Capability::ScissorTest);
        }

        /// Zeroes (and sets the minima of) the storage buffer for `header_words` header words and `cell_words` hit cells.
        void reset_buffer(std::size_t header_words, std::size_t cell_words)
        {
            std::vector<std::uint32_t> init(header_words + cell_words, 0u);
            if (header_words == kCountHeader)
            {
                init[1] = kUnset;  // min_x
                init[2] = kUnset;  // min_y
            }
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(init.size() * sizeof(std::uint32_t)), init.data(), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, ssbo_);
        }

        std::vector<std::uint32_t> read_buffer(std::size_t words)
        {
            // The mapped read is an access by the commands of ES 3.2 section 6: BUFFER_UPDATE_BARRIER_BIT (section 7.11.2).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            std::vector<std::uint32_t> out(words);
            const void* p = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, static_cast<GLsizeiptr>(words * sizeof(std::uint32_t)), mgl::MapBufferAccessMask::Read);
            if (p == nullptr) throw demo::Failure("glMapBufferRange on the hit buffer failed");
            std::memcpy(out.data(), p, words * sizeof(std::uint32_t));
            (void)mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage);
            return out;
        }

        void use(const Prog& p, const Draw& d, int stride)
        {
            mgl::glUseProgram(p.id);
            mgl::glUniform1f(p.depth, d.depth);
            mgl::glUniform1i(p.stride, d.record_cells ? stride : 0);
        }

        Observed fetch_count(int fw, int fh, bool record_cells)
        {
            const int stride = fw + kGuard, rows = fh + kGuard;
            const std::size_t cells = record_cells ? static_cast<std::size_t>(stride * rows) : 0;
            const std::vector<std::uint32_t> raw = read_buffer(kCountHeader + cells);
            Observed o;
            o.total = raw[0];
            o.min_x = raw[1];
            o.min_y = raw[2];
            o.max_x = raw[3];
            o.max_y = raw[4];
            o.cells.assign(raw.begin() + kCountHeader, raw.end());
            o.stride = stride;
            return o;
        }

        Observed fetch_helper(int fw, int fh)
        {
            const int stride = fw + kGuard, rows = fh + kGuard;
            const std::vector<std::uint32_t> raw = read_buffer(4 + static_cast<std::size_t>(stride * rows));
            Observed o;
            o.guarded = raw[0];
            o.unguarded = raw[1];
            o.helper_seen = raw[2];
            o.deriv_bad = raw[3];
            o.cells.assign(raw.begin() + 4, raw.end());
            o.stride = stride;
            return o;
        }

        /// Draws one NDC rectangle with a counting program into the framebuffer bound to GL_FRAMEBUFFER.
        Case run_quad(std::string name, const Prog& p, int fw, int fh, const Draw& d, const Ndc& ndc)
        {
            const int stride = fw + kGuard;
            reset_buffer(kCountHeader, d.record_cells ? static_cast<std::size_t>(stride * (fh + kGuard)) : 0);
            begin_draw(d);
            use(p, d, stride);
            mgl::glUniform4f(p.rect, ndc.x0, ndc.y0, ndc.x1, ndc.y1);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            end_draw(d);
            Case c;
            c.name = std::move(name);
            c.fw = fw;
            c.fh = fh;
            c.expected = model_quad(fw, fh, window_rect(d.viewport, ndc), d.scissor);
            c.observed = fetch_count(fw, fh, d.record_cells);
            return c;
        }

        static std::array<float, 12> triangle_ndc()
        {
            std::array<float, 12> pos{};
            for (std::size_t t = 0; t < 2; ++t)
                for (std::size_t i = 0; i < 3; ++i)
                {
                    pos[(3 * t + i) * 2] = static_cast<float>(kTriangles[t].v[i][0] / 512.0 - 1.0);      // X / 16 / 32 - 1
                    pos[(3 * t + i) * 2 + 1] = static_cast<float>(kTriangles[t].v[i][1] / 256.0 - 1.0);  // Y / 16 / 16 - 1
                }
            return pos;
        }

        Case run_triangles(std::string name, const Prog& p, int fw, int fh, const Draw& d)
        {
            const int stride = fw + kGuard;
            reset_buffer(p.id.value == prog_tri_helper_.id.value ? kHelperHeader : kCountHeader, static_cast<std::size_t>(stride * (fh + kGuard)));
            begin_draw(d);
            use(p, d, stride);
            const std::array<float, 12> pos = triangle_ndc();
            mgl::glUniform2fv(p.pos, 6, pos.data());
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 6);
            end_draw(d);
            Case c;
            c.name = std::move(name);
            c.fw = fw;
            c.fh = fh;
            const TriModel m = model_triangles({kTriangles.begin(), kTriangles.end()}, fw, fh, d.scissor);
            c.expected = m.hits;
            tri_on_edge_ += m.on_edge;
            c.observed = p.id.value == prog_tri_helper_.id.value ? fetch_helper(fw, fh) : fetch_count(fw, fh, true);
            return c;
        }

        /// One triangle (plus a degenerate second one) that covers the whole framebuffer: its edges are far away, so after the scissor test every
        /// pixel inside the scissor box is covered by a single primitive and every one of its samples is covered.
        Case run_cover_triangle(std::string name, const Prog& p, int fw, int fh, const Draw& d)
        {
            const int stride = fw + kGuard;
            const bool helper = p.id.value == prog_tri_helper_.id.value;
            reset_buffer(helper ? kHelperHeader : kCountHeader, static_cast<std::size_t>(stride * (fh + kGuard)));
            begin_draw(d);
            use(p, d, stride);
            const std::array<float, 12> pos = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            mgl::glUniform2fv(p.pos, 6, pos.data());
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 6);
            end_draw(d);
            Case c;
            c.name = std::move(name);
            c.fw = fw;
            c.fh = fh;
            c.expected = model_quad(fw, fh, {0, 0, fw, fh}, d.scissor);
            c.observed = helper ? fetch_helper(fw, fh) : fetch_count(fw, fh, true);
            return c;
        }

        // ------------------------------------------------------------------------------------------
        // The experiments (run once, in init(), so that the contact sheet can show them).
        // ------------------------------------------------------------------------------------------
        void run_experiments()
        {
            // The framebuffer without attachments: 64 x 32, one sample.
            bind_fbo(fbos_[0]);
            configure_default(kW, kH);
            require_complete("framebuffer with no attachments (64 x 32)");

            const Box full{0, 0, kW, kH};
            const Ndc whole = ndc_for(full, {0, 0, kW, kH});

            full_ = run_quad("full viewport quad", prog_rect_, kW, kH, {}, whole);
            Draw d;
            sub_ = run_quad("rectangle [5,41) x [3,19)", prog_rect_, kW, kH, d, ndc_for(full, {5, 3, 41, 19}));
            d.scissor = Box{10, 4, 20, 12};
            scissored_ = run_quad("same rectangle, scissor (10,4) 20 x 12", prog_rect_, kW, kH, d, ndc_for(full, {5, 3, 41, 19}));

            Draw small;
            small.viewport = {8, 4, 32, 16};
            small_viewport_ = run_quad("viewport (8,4) 32 x 16, full-NDC quad", prog_rect_, kW, kH, small, {-1.0f, -1.0f, 1.0f, 1.0f});
            Draw big;
            big.viewport = {0, 0, 2 * kW, 2 * kH};
            big_viewport_ = run_quad("viewport 128 x 64 larger than the framebuffer", prog_rect_, kW, kH, big, {-1.0f, -1.0f, 1.0f, 1.0f});
            Draw neg;
            neg.viewport = {-32, -16, kW, kH};
            neg_viewport_ = run_quad("viewport (-32,-16) 64 x 32 half outside", prog_rect_, kW, kH, neg, {-1.0f, -1.0f, 1.0f, 1.0f});

            triangles_ = run_triangles("two triangles, SSBO", prog_tri_, kW, kH, {});
            Draw tri_scissor;
            tri_scissor.scissor = Box{20, 6, 24, 18};
            triangles_scissored_ = run_triangles("two triangles, scissor (20,6) 24 x 18", prog_tri_, kW, kH, tri_scissor);

            // A framebuffer that does have an attachment: the default size is ignored (and comes back when it is detached).
            bind_fbo(fbos_[1]);
            configure_default(kW, kH);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rbos_[0]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 16, 8);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rbos_[0]);
            require_complete("framebuffer with a 16 x 8 attachment");
            attachment_wins_ = run_quad("16 x 8 attachment, default size 64 x 32", prog_rect_, kW, kH, {}, whole);
            // The expected map is the 16 x 8 corner only.
            attachment_wins_.expected = model_quad(kW, kH, {0, 0, 16, 8}, std::nullopt);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            require_complete("framebuffer after detaching its attachment");
            detached_ = run_quad("attachment detached again", prog_rect_, kW, kH, {}, whole);

            // Early fragment tests (a depth-tested framebuffer) and helper invocations are experiments of their own.
            run_early();
            bind_fbo(fbos_[0]);
            configure_default(kW, kH);
            Draw hd;
            helper_triangles_ = run_triangles("helper invocations: two triangles", prog_tri_helper_, kW, kH, hd);
            const Ndc r = ndc_for(full, {5, 3, 41, 19});
            const int stride = kW + kGuard;
            reset_buffer(4, static_cast<std::size_t>(stride * (kH + kGuard)));
            begin_draw(hd);
            use(prog_rect_helper_, hd, stride);
            mgl::glUniform4f(prog_rect_helper_.rect, r.x0, r.y0, r.x1, r.y1);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            helper_rect_.name = "helper invocations: rectangle starting at an odd pixel";
            helper_rect_.fw = kW;
            helper_rect_.fh = kH;
            helper_rect_.expected = model_quad(kW, kH, {5, 3, 41, 19}, std::nullopt);
            helper_rect_.observed = fetch_helper(kW, kH);
            bind_fbo(mgl::FramebufferId{0});
        }

        // A: a rectangle in front (depth 0.25); B: a rectangle behind it (depth 0.75) that A partly hides; then two probe draws that
        // show what the depth buffer holds.
        static constexpr WinRect kRectA = {8, 4, 40, 28};
        static constexpr WinRect kRectB = {24, 10, 56, 31};

        struct EarlySequence
        {
            Case a, b, probe_half, probe_far;
        };

        EarlySequence run_early_sequence(bool early)
        {
            EarlySequence s;
            const Box full{0, 0, kW, kH};
            const Prog& variant = early ? prog_rect_early_ : prog_rect_;
            mgl::glViewport(0, 0, kW, kH);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Less);
            mgl::glDepthMask(GL_TRUE);
            mgl::glClearDepthf(1.0f);
            mgl::glClear(mgl::ClearBufferBit::Depth);

            Draw d;
            d.depth = 0.25f;
            s.a = run_quad(early ? "early: A in front" : "late: A in front", variant, kW, kH, d, ndc_for(full, kRectA));
            d.depth = 0.75f;
            s.b = run_quad(early ? "early: B behind A" : "late: B behind A", variant, kW, kH, d, ndc_for(full, kRectB));
            // B's expected map: all of B for the late variant's superset test; the visible part is derived from the model below.
            // The probes use the early program and write no depth.
            mgl::glDepthMask(GL_FALSE);
            d.depth = 0.5f;
            s.probe_half = run_quad("probe at depth 0.5", prog_rect_early_, kW, kH, d, ndc_for(full, {0, 0, kW, kH}));
            d.depth = 0.9f;
            s.probe_far = run_quad("probe at depth 0.9", prog_rect_early_, kW, kH, d, ndc_for(full, {0, 0, kW, kH}));
            mgl::glDepthMask(GL_TRUE);
            mgl::glDisable(mgl::Capability::DepthTest);
            return s;
        }

        void run_early()
        {
            bind_fbo(fbos_[2]);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rbos_[1]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent24, kW, kH);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, mgl::RenderbufferTarget::Renderbuffer, rbos_[1]);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            require_complete("depth-only framebuffer (no color attachment)");
            early_ = run_early_sequence(true);
            late_ = run_early_sequence(false);

            // The CPU model: A (depth 0.25) fills its rectangle; B (depth 0.75) is visible where A is not in front of it.
            // The probes (depth test LESS at 0.5 / 0.9, no depth writes) pass where the stored depth is larger.
            a_model_ = model_quad(kW, kH, kRectA, std::nullopt);
            b_all_ = model_quad(kW, kH, kRectB, std::nullopt);
            b_visible_.assign(b_all_.size(), 0);
            probe_half_.assign(b_all_.size(), 0);
            probe_far_.assign(b_all_.size(), 0);
            for (std::size_t i = 0; i < b_all_.size(); ++i)
            {
                b_visible_[i] = (b_all_[i] != 0 && a_model_[i] == 0) ? 1 : 0;
                probe_half_[i] = a_model_[i] == 0 ? 1 : 0;                      // stored depth > 0.5: everything except A
                probe_far_[i] = (a_model_[i] == 0 && b_all_[i] == 0) ? 1 : 0;   // stored depth > 0.9: neither A nor B
            }
            for (EarlySequence* s : {&early_, &late_})
            {
                s->a.expected = a_model_;
                s->b.expected = b_visible_;  // for the late variant this is the part that must run, not all that may
                s->probe_half.expected = probe_half_;
                s->probe_far.expected = probe_far_;
            }
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            bind_fbo(mgl::FramebufferId{0});
        }

        // The image variant needs textures and a framebuffer to read them; it is run from validate() (it is a check, not a picture).
        struct ImageResult
        {
            std::vector<std::uint32_t> marker, covered;
        };

        ImageResult run_images()
        {
            bind_fbo(fbos_[0]);
            configure_default(kW, kH);
            const std::vector<std::uint32_t> zeros(static_cast<std::size_t>(kW * kH), 0u);
            // Writes by shaders and the API writes below are ordered by TEXTURE_UPDATE_BARRIER_BIT (ES 3.2 7.11.2).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureUpdate);
            for (const mgl::TextureId t : {tex_[0], tex_[1]})
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
                mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, kW, kH, mgl::PixelFormat::RedInteger, mgl::PixelType::UnsignedInt, zeros.data());
            }
            mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_[0], 0, GL_FALSE, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);
            mgl::glBindImageTexture(mgl::ImageUnit{1}, tex_[1], 0, GL_FALSE, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32UI);

            mgl::glViewport(0, 0, kW, kH);
            mgl::glUseProgram(prog_tri_image_.id);
            const std::array<float, 12> pos = triangle_ndc();
            mgl::glUniform2fv(prog_tri_image_.pos, 6, pos.data());
            mgl::glUniform1f(prog_tri_image_.depth, 0.5f);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 6);
            bind_fbo(mgl::FramebufferId{0});

            // Read the images through a framebuffer (glReadPixels): FRAMEBUFFER_BARRIER_BIT makes the shader stores visible to it.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::Framebuffer | mgl::MemoryBarrierMask::ShaderImageAccess);
            ImageResult r;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo_read_);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            for (int i = 0; i < 2; ++i)
            {
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                            mgl::TextureTarget::Texture2D, tex_[static_cast<std::size_t>(i)], 0);
                std::vector<std::uint32_t> rgba(static_cast<std::size_t>(4 * kW * kH));
                // RGBA_INTEGER / UNSIGNED_INT is the pair every unsigned-integer color buffer can be read with (ES 3.2 16.1.2).
                mgl::glReadPixels(0, 0, kW, kH, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, rgba.data());
                std::vector<std::uint32_t>& out = (i == 0) ? r.marker : r.covered;
                out.resize(static_cast<std::size_t>(kW * kH));
                for (std::size_t p = 0; p < out.size(); ++p) out[p] = rgba[4 * p];
            }
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                        mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            return r;
        }

        void build_tiles()
        {
            // R32UI textures for the image variant, and the RGBA8 tiles of the contact sheet.
            for (const mgl::TextureId t : {tex_[0], tex_[1]})
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
                mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::R32UI, kW, kH);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            }
            mgl::glGenFramebuffers(1, &fbo_read_);

            const std::array<const Case*, 8> shown = {&scissored_, &small_viewport_, &neg_viewport_, &triangles_, &attachment_wins_, &early_.b, &late_.b, &helper_triangles_};
            mgl::glGenTextures(static_cast<GLsizei>(tile_tex_.size()), tile_tex_.data());
            for (std::size_t i = 0; i < shown.size(); ++i)
            {
                const Case& c = *shown[i];
                std::vector<std::uint8_t> px(static_cast<std::size_t>(4 * kW * kH));
                for (int y = 0; y < kH; ++y)
                    for (int x = 0; x < kW; ++x)
                    {
                        const std::uint32_t got = c.observed.cells[static_cast<std::size_t>(y * c.observed.stride + x)];
                        const std::uint32_t want = c.expected[static_cast<std::size_t>(y * c.fw + x)];
                        // For the late-test run the expectation is the visible part only: an occluded fragment that ran is allowed (shown in cyan).
                        const bool late = &c == &late_.b;
                        const bool wrong = late ? (want != 0 && got != want) : got != want;
                        Rgba col = hit_color(got, wrong);
                        if (late && want == 0 && got != 0) col = {0, 200, 255, 255};
                        for (std::size_t k = 0; k < 4; ++k) px[static_cast<std::size_t>(4 * (y * kW + x)) + k] = col[k];
                    }
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tile_tex_[i]);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kW, kH, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, px.data());
            }
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        // ------------------------------------------------------------------------------------------
        // Checks.
        // ------------------------------------------------------------------------------------------
        static std::string bounds_text(const Observed& o)
        {
            return std::format("({},{})-({},{})", o.min_x, o.min_y, o.max_x, o.max_y);
        }

        /// Total, exact map and bounds against the model; `fragments` is the expected number of fragments.
        void expect_case(demo::Check& check, const Case& c, std::string what)
        {
            const MapCompare m = compare_map(c);
            const std::uint64_t want_total = sum(c.expected);
            const Bounds b = model_bounds(c);
            const bool bounds_ok = c.observed.min_x == b.min_x && c.observed.min_y == b.min_y && c.observed.max_x == b.max_x && c.observed.max_y == b.max_y;
            const bool ok = c.observed.total == want_total && m.mismatches == 0 && m.guard_hits == 0 && bounds_ok;
            check.expect(ok, std::move(what),
                         ok ? std::format("{} fragments, every pixel as modelled, bounds {}", c.observed.total, bounds_text(c.observed))
                            : std::format("total {} (expected {}); {} pixels differ {}; {} hits outside the framebuffer; bounds {} (expected ({},{})-({},{}))", c.observed.total, want_total,
                                          m.mismatches, m.first, m.guard_hits, bounds_text(c.observed), b.min_x, b.min_y, b.max_x, b.max_y));
        }

        void check_limits(demo::Check& check)
        {
            // ES 3.2 table 21.40: the minimum values of the limits that bound FRAMEBUFFER_DEFAULT_*.
            check.expect(max_fb_width_ >= 2048 && max_fb_height_ >= 2048 && max_fb_samples_ >= 4,
                         "MAX_FRAMEBUFFER_WIDTH / HEIGHT >= 2048 and MAX_FRAMEBUFFER_SAMPLES >= 4 (ES 3.2 table 21.40)",
                         std::format("{} x {}, {} samples", max_fb_width_, max_fb_height_, max_fb_samples_));
            std::cout << std::format("   note: MAX_FRAMEBUFFER_WIDTH/HEIGHT/SAMPLES = {}/{}/{}, MAX_VIEWPORT_DIMS = {} x {}\n", max_fb_width_, max_fb_height_, max_fb_samples_,
                                     max_viewport_[0], max_viewport_[1]);
        }

        void check_parameters_and_completeness(demo::Check& check)
        {
            mgl::FramebufferId f{};
            mgl::glGenFramebuffers(1, &f);
            extra_fbos_.push_back(f);
            bind_fbo(f);

            // ES 3.2 table 21.15: all four are 0 / FALSE in a new framebuffer object.
            check.expect(get_param(mgl::FramebufferDefaultParameter::Width) == 0 && get_param(mgl::FramebufferDefaultParameter::Height) == 0 &&
                             get_param(mgl::FramebufferDefaultParameter::Samples) == 0 && get_param(mgl::FramebufferDefaultParameter::FixedSampleLocations) == 0,
                         "a new framebuffer object has FRAMEBUFFER_DEFAULT_WIDTH = HEIGHT = SAMPLES = 0 and FIXED_SAMPLE_LOCATIONS = FALSE (ES 3.2 table 21.15)");

            // ES 3.2 9.4.2: "at least one image attached, or the FRAMEBUFFER_DEFAULT_WIDTH and FRAMEBUFFER_DEFAULT_HEIGHT are both non-zero"
            // else FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT.
            check.expect_eq(status_name(), std::string("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"), "no attachment and default size 0 x 0: FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT (ES 3.2 9.4.2)");
            set_param(mgl::FramebufferDefaultParameter::Width, 16);
            check.expect_eq(status_name(), std::string("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"), "default width 16 but height 0: still incomplete (both must be non-zero)");
            set_param(mgl::FramebufferDefaultParameter::Width, 0);
            set_param(mgl::FramebufferDefaultParameter::Height, 16);
            check.expect_eq(status_name(), std::string("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"), "default height 16 but width 0: still incomplete");

            // Drawing into an incomplete framebuffer is INVALID_FRAMEBUFFER_OPERATION (ES 3.2 9.4.4).
            Draw d;
            (void)demo::drain_gl_errors();
            use(prog_rect_, d, kW + kGuard);
            mgl::glUniform4f(prog_rect_.rect, -1.0f, -1.0f, 1.0f, 1.0f);
            mgl::glViewport(0, 0, 16, 16);
            reset_buffer(kCountHeader, 16);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            check.expect_gl_error(mgl::ErrorCode::InvalidFramebufferOperation, "a draw into the incomplete framebuffer -> INVALID_FRAMEBUFFER_OPERATION (ES 3.2 9.4.4)");
            const std::vector<std::uint32_t> untouched = read_buffer(kCountHeader + 16);
            check.expect_eq(untouched[0], 0u, "...and no fragment ran (the counter is still 0)");

            set_param(mgl::FramebufferDefaultParameter::Width, 24);
            set_param(mgl::FramebufferDefaultParameter::Height, 12);
            check.expect_eq(status_name(), std::string("GL_FRAMEBUFFER_COMPLETE"), "default size 24 x 12 and no attachment: FRAMEBUFFER_COMPLETE (ES 3.2 9.4.2)");
            check.expect(get_param(mgl::FramebufferDefaultParameter::Width) == 24 && get_param(mgl::FramebufferDefaultParameter::Height) == 12,
                         "glGetFramebufferParameteriv returns the width 24 and height 12 that were set");

            // Samples and the sample-location flag.
            set_param(mgl::FramebufferDefaultParameter::Samples, 4);
            set_param(mgl::FramebufferDefaultParameter::FixedSampleLocations, 1);
            check.expect(get_param(mgl::FramebufferDefaultParameter::Samples) == 4 && get_param(mgl::FramebufferDefaultParameter::FixedSampleLocations) == 1,
                         "FRAMEBUFFER_DEFAULT_SAMPLES = 4 and FIXED_SAMPLE_LOCATIONS = TRUE read back as set");
            const mgl::FramebufferStatus ms = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            check.expect(ms == mgl::FramebufferStatus::Complete, "a 24 x 12 framebuffer with 4 default samples and fixed sample locations is complete",
                         std::string(demo::framebuffer_status_name(ms)));
            // (SAMPLE_BUFFERS / SAMPLES of such a framebuffer are checked in check_multisample, together with the rasterization.)
            set_param(mgl::FramebufferDefaultParameter::Samples, 0);
            set_param(mgl::FramebufferDefaultParameter::FixedSampleLocations, 0);
            check.expect(demo::get_integer(mgl::GetParameter::SampleBuffers) == 0 && demo::get_integer(mgl::GetParameter::Samples) == 0,
                         "with default samples 0 again: SAMPLE_BUFFERS = 0 and SAMPLES = 0");
            check.expect(get_param(mgl::FramebufferDefaultParameter::Samples) == 0 && get_param(mgl::FramebufferDefaultParameter::FixedSampleLocations) == 0,
                         "FRAMEBUFFER_DEFAULT_SAMPLES and FIXED_SAMPLE_LOCATIONS read back as 0 / FALSE after being reset");

            // The largest sizes and sample count the implementation advertises are accepted and complete.
            set_param(mgl::FramebufferDefaultParameter::Width, max_fb_width_);
            set_param(mgl::FramebufferDefaultParameter::Height, max_fb_height_);
            set_param(mgl::FramebufferDefaultParameter::Samples, max_fb_samples_);
            check.expect_no_gl_error("FRAMEBUFFER_DEFAULT_WIDTH/HEIGHT/SAMPLES = MAX_FRAMEBUFFER_WIDTH/HEIGHT/SAMPLES are accepted");
            const mgl::FramebufferStatus mx = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            check.expect(mx == mgl::FramebufferStatus::Complete, std::format("a {} x {} framebuffer with {} default samples and no attachment is complete", max_fb_width_, max_fb_height_, max_fb_samples_),
                         std::string(demo::framebuffer_status_name(mx)));
            bind_fbo(mgl::FramebufferId{0});
        }

        static std::string status_name()
        {
            return std::string(demo::framebuffer_status_name(mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer)));
        }

        void check_parameter_errors(demo::Check& check)
        {
            mgl::FramebufferId f{};
            mgl::glGenFramebuffers(1, &f);
            extra_fbos_.push_back(f);
            bind_fbo(f);
            configure_default(20, 10, 1);
            (void)demo::drain_gl_errors();

            const auto try_set = [&](mgl::FramebufferDefaultParameter p, long long v) {
                mgl::glFramebufferParameteri(mgl::FramebufferTarget::Framebuffer, p, static_cast<GLint>(v));
            };
            using P = mgl::FramebufferDefaultParameter;
            try_set(P::Width, static_cast<long long>(max_fb_width_) + 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_WIDTH = MAX_FRAMEBUFFER_WIDTH + 1 -> INVALID_VALUE (ES 3.2 9.2.1)");
            try_set(P::Height, static_cast<long long>(max_fb_height_) + 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_HEIGHT = MAX_FRAMEBUFFER_HEIGHT + 1 -> INVALID_VALUE");
            try_set(P::Samples, static_cast<long long>(max_fb_samples_) + 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_SAMPLES = MAX_FRAMEBUFFER_SAMPLES + 1 -> INVALID_VALUE");
            try_set(P::Width, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_WIDTH = -1 -> INVALID_VALUE");
            try_set(P::Height, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_HEIGHT = -1 -> INVALID_VALUE");
            try_set(P::Samples, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_SAMPLES = -1 -> INVALID_VALUE");
            try_set(P::Width, 0x7FFFFFFF);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_WIDTH = INT_MAX -> INVALID_VALUE");
            // ES 3.2 2.5: an erroring command has no effect.
            check.expect(get_param(P::Width) == 20 && get_param(P::Height) == 10 && get_param(P::Samples) == 1,
                         "after the rejected values the parameters still read 20 x 10, 1 sample (an erroring command has no effect)",
                         std::format("{} x {}, {} samples", get_param(P::Width), get_param(P::Height), get_param(P::Samples)));

            // The default framebuffer has no such parameters: INVALID_OPERATION for both commands (ES 3.2 9.2.1, 9.2.3).
            bind_fbo(mgl::FramebufferId{0});
            try_set(P::Width, 8);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glFramebufferParameteri with the default framebuffer bound -> INVALID_OPERATION (ES 3.2 9.2.1)");
            GLint v = -7;
            mgl::glGetFramebufferParameteriv(mgl::FramebufferTarget::Framebuffer, P::Width, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetFramebufferParameteriv with the default framebuffer bound -> INVALID_OPERATION (ES 3.2 9.2.3)");

            // The targets: READ_FRAMEBUFFER names the read framebuffer, FRAMEBUFFER is DRAW_FRAMEBUFFER.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, f);
            mgl::glFramebufferParameteri(mgl::FramebufferTarget::ReadFramebuffer, P::Width, 48);
            GLint via_read = -1;
            mgl::glGetFramebufferParameteriv(mgl::FramebufferTarget::ReadFramebuffer, P::Width, &via_read);
            check.expect(via_read == 48, "READ_FRAMEBUFFER target: the parameter is set on and read from the read framebuffer (the draw framebuffer is still the default one)",
                         std::format("read back {}", via_read));
            mgl::glGetFramebufferParameteriv(mgl::FramebufferTarget::DrawFramebuffer, P::Width, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "DRAW_FRAMEBUFFER target while the draw framebuffer is the default one -> INVALID_OPERATION");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::DrawFramebuffer, f);
            GLint via_draw = -1;
            mgl::glGetFramebufferParameteriv(mgl::FramebufferTarget::DrawFramebuffer, P::Width, &via_draw);
            GLint via_fb = -1;
            mgl::glGetFramebufferParameteriv(mgl::FramebufferTarget::Framebuffer, P::Width, &via_fb);
            check.expect(via_draw == 48 && via_fb == 48, "the same framebuffer bound to DRAW_FRAMEBUFFER: FRAMEBUFFER and DRAW_FRAMEBUFFER give the same value (FRAMEBUFFER is equivalent to DRAW_FRAMEBUFFER)");
            bind_fbo(mgl::FramebufferId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
        }

        void check_rasterization(demo::Check& check)
        {
            expect_case(check, full_, "no attachments, default 64 x 32: a viewport-covering quad produces 2048 fragments, one per pixel, nothing outside (bounds (0,0)-(63,31))");
            expect_case(check, sub_, "rectangle [5,41) x [3,19): 576 fragments in exactly those pixels");
            expect_case(check, scissored_, "the same rectangle with scissor (10,4) 20 x 12: exactly the intersection, [10,30) x [4,16) = 240 fragments (ES 3.2 13.8.2)");
            expect_case(check, small_viewport_, "viewport (8,4) 32 x 16 with a full-NDC quad: exactly [8,40) x [4,20) = 512 fragments");
            expect_case(check, big_viewport_, "viewport 128 x 64 larger than the 64 x 32 framebuffer: rendering is limited to (0,0)-(64,32), 2048 fragments, none outside (ES 3.2 9.2)");
            expect_case(check, neg_viewport_, "viewport (-32,-16) 64 x 32, half outside the framebuffer: exactly [0,32) x [0,16) = 512 fragments");

            // Triangles: exact per-pixel agreement with the 1/16-pixel-grid model.
            expect_case(check, triangles_, "two triangles on the 1/16 pixel grid (one 633-pixel, one thin 73-pixel sliver): every pixel hit exactly as the exact-arithmetic coverage model says");
            expect_case(check, triangles_scissored_, "the same triangles with scissor (20,6) 24 x 18: the model intersected with the scissor box");
            check.expect(tri_on_edge_ == 0, "test design: no pixel centre lies exactly on a triangle edge, so the coverage rule is unambiguous", std::format("{} on-edge pixels", tri_on_edge_));

            // Negative controls: the comparisons reject a wrong expectation.
            Case wrong = full_;
            wrong.expected = sub_.expected;
            const MapCompare neg1 = compare_map(wrong);
            Case wrong2 = triangles_;
            wrong2.expected = model_triangles({kTriangles[0]}, kW, kH, std::nullopt).hits;  // the first triangle only
            const MapCompare neg2 = compare_map(wrong2);
            check.expect(neg1.mismatches == 2048 - 576 && neg2.mismatches == 73,
                         "negative control: the full-quad hits judged against the sub-rectangle's model differ in 1472 pixels; the two-triangle hits judged against the first triangle alone differ in the 73 sliver pixels",
                         std::format("{} and {} mismatches", neg1.mismatches, neg2.mismatches));
        }

        void check_sizes(demo::Check& check)
        {
            // A new default size takes effect (33 x 17 is not a power of two).
            bind_fbo(fbos_[3]);
            configure_default(33, 17);
            require_complete("33 x 17 framebuffer");
            Draw d;
            d.viewport = {0, 0, 33, 17};
            const Case odd = run_quad("33 x 17", prog_rect_, 33, 17, d, {-1.0f, -1.0f, 1.0f, 1.0f});
            expect_case(check, odd, "default size 33 x 17 (not a power of two): a viewport-covering quad gives 561 fragments, bounds (0,0)-(32,16)");
            Draw d2;
            d2.viewport = {0, 0, kW, kH};
            const Case odd_big = run_quad("33 x 17, viewport 64 x 32", prog_rect_, 33, 17, d2, {-1.0f, -1.0f, 1.0f, 1.0f});
            expect_case(check, odd_big, "the same framebuffer with a 64 x 32 viewport: still only the 561 fragments inside 33 x 17");
            // Resizing the same framebuffer object: the next draw sees the new size.
            configure_default(40, 8);
            require_complete("40 x 8 framebuffer");
            Draw d3;
            d3.viewport = {0, 0, 40, 8};
            expect_case(check, run_quad("40 x 8", prog_rect_, 40, 8, d3, {-1.0f, -1.0f, 1.0f, 1.0f}), "after glFramebufferParameteri(40, 8) on the same object: 320 fragments, bounds (0,0)-(39,7)");

            // The maximum size. The viewport can be no larger than MAX_VIEWPORT_DIMS, so the far corner of the viewport / framebuffer
            // intersection is the target: a 3 x 2 scissor box there.
            const int vw = std::min(max_fb_width_, max_viewport_[0]);
            const int vh = std::min(max_fb_height_, max_viewport_[1]);
            configure_default(max_fb_width_, max_fb_height_);
            require_complete("maximum-size framebuffer");
            Draw m;
            m.viewport = {0, 0, vw, vh};
            m.scissor = Box{vw - 3, vh - 2, 3, 2};
            m.record_cells = false;
            const Case big = [&] {
                Case c = run_quad("max size", prog_rect_, 8, 8, m, {-1.0f, -1.0f, 1.0f, 1.0f});
                return c;
            }();
            const bool ok = big.observed.total == 6 && big.observed.min_x == static_cast<std::uint32_t>(vw - 3) && big.observed.max_x == static_cast<std::uint32_t>(vw - 1) &&
                            big.observed.min_y == static_cast<std::uint32_t>(vh - 2) && big.observed.max_y == static_cast<std::uint32_t>(vh - 1);
            check.expect(ok, std::format("a {} x {} framebuffer with no attachment (MAX_FRAMEBUFFER_WIDTH x HEIGHT): a 3 x 2 scissor box at the far corner of the viewport gives exactly 6 fragments at ({}..{}, {}..{})",
                                         max_fb_width_, max_fb_height_, vw - 3, vw - 1, vh - 2, vh - 1),
                         std::format("{} fragments, bounds {}", big.observed.total, bounds_text(big.observed)));
            bind_fbo(mgl::FramebufferId{0});

            // Attachments take precedence over the default size, and detaching restores it.
            expect_case(check, attachment_wins_, "default size 64 x 32 but a 16 x 8 color attachment: only the 16 x 8 attachment is rendered (128 fragments, bounds (0,0)-(15,7)) (ES 3.2 9.2.1: the size comes from the attachment)");
            expect_case(check, detached_, "the attachment detached again: the framebuffer is 64 x 32 again, 2048 fragments");
        }

        void check_multisample(demo::Check& check)
        {
            const int samples = std::min(4, max_fb_samples_);
            if (samples < 2)
            {
                std::cout << "   note: MAX_FRAMEBUFFER_SAMPLES < 2, the multisample run is skipped\n";
                return;
            }
            bind_fbo(fbos_[4]);
            configure_default(kW, kH, samples, true);
            require_complete("multisample framebuffer with no attachments");
            // What the SAMPLE_BUFFERS / SAMPLES queries return here is not defined by the text, so only the rule that does apply is
            // asserted. ES 3.2 9.2.1 says such a framebuffer "is considered to have sample buffers if and only if the value of
            // FRAMEBUFFER_DEFAULT_SAMPLES is non-zero", which governs rasterization (checked below through the invocation counts). The
            // section that defines the queried values, 9.2.3.1, derives them only from attached images: "the effective value of
            // SAMPLES is equal to the value of RENDERBUFFER_SAMPLES or TEXTURE_SAMPLES (depending on the type of the attached
            // images)", and neither ES 3.1/3.2 nor ARB_framebuffer_no_attachments adds a rule for a framebuffer without attachments.
            // (An earlier version asserted SAMPLE_BUFFERS = 1 and SAMPLES >= 4 with a Mesa XFAIL; ANGLE reports 0 / 0 as well, and
            // re-reading 9.2.3.1 showed that the expectation was an inference, not a rule.) What 9.2.3.1 does guarantee: "The
            // effective value of SAMPLE_BUFFERS is one if SAMPLES is non-zero, and zero otherwise."
            const int sample_buffers = demo::get_integer(mgl::GetParameter::SampleBuffers);
            const int actual = demo::get_integer(mgl::GetParameter::Samples);
            std::cout << std::format("   [query] no attachments, FRAMEBUFFER_DEFAULT_SAMPLES = {}: SAMPLE_BUFFERS = {}, SAMPLES = {} (values not asserted)\n", samples,
                                     sample_buffers, actual);
            check.expect(actual >= 0 && sample_buffers == (actual != 0 ? 1 : 0),
                         std::format("a framebuffer without attachments and FRAMEBUFFER_DEFAULT_SAMPLES = {}: SAMPLE_BUFFERS is 1 exactly when SAMPLES is non-zero (ES 3.2 9.2.3.1)", samples),
                         std::format("SAMPLE_BUFFERS {}, SAMPLES {}", sample_buffers, actual));
            check.expect_no_gl_error("the SAMPLE_BUFFERS / SAMPLES queries raise no error");

            // A single triangle covering the whole framebuffer, limited by the scissor box to [5,41) x [3,19): every pixel in the box has one
            // fragment with all its samples covered; no primitive edge crosses the box.
            Draw d;
            d.scissor = Box{5, 3, 36, 16};
            const Case ms = run_cover_triangle("multisample, covering triangle + scissor", prog_tri_, kW, kH, d);
            const Case ms_helper = run_cover_triangle("multisample, covering triangle + scissor, helper shader", prog_tri_helper_, kW, kH, d);
            // The same triangle and box on a multisample renderbuffer attachment: tells whether a deviation seen above belongs to the
            // attachment-less framebuffer or to multisampling in general.
            bind_fbo(fbos_[5]);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rbos_[2]);
            mgl::glRenderbufferStorageMultisample(mgl::RenderbufferTarget::Renderbuffer, samples, mgl::InternalFormat::Rgba8, kW, kH);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rbos_[2]);
            require_complete("multisample renderbuffer framebuffer");
            const Case rb_helper = run_cover_triangle("multisample renderbuffer, covering triangle + scissor, helper shader", prog_tri_helper_, kW, kH, d);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            // The single-sample control with the same triangle and box (the 64 x 32 framebuffer without samples).
            bind_fbo(fbos_[0]);
            const Case single = run_cover_triangle("single sample, covering triangle + scissor", prog_tri_, kW, kH, d);
            const Case single_helper = run_cover_triangle("single sample, covering triangle + scissor, helper shader", prog_tri_helper_, kW, kH, d);
            bind_fbo(mgl::FramebufferId{0});

            // ES 3.2 7.11.1: with a multisample buffer the number of invocations per fragment is in [1, N], N the covered samples, at most SAMPLES
            // and so at most MAX_FRAMEBUFFER_SAMPLES (the SAMPLES query itself is not usable here, see above). Only the pixels inside the box are looked at.
            const WinRect rect{5, 3, 41, 19};
            int bad = 0;
            std::uint32_t most = 0;
            std::string first;
            for (int y = rect.y0; y < rect.y1; ++y)
                for (int x = rect.x0; x < rect.x1; ++x)
                {
                    const std::uint32_t got = ms.observed.cells[static_cast<std::size_t>(y * ms.observed.stride + x)];
                    most = std::max(most, got);
                    if ((got < 1 || got > static_cast<std::uint32_t>(max_fb_samples_)) && bad++ == 0) first = std::format("({},{}): {} invocations", x, y, got);
                }
            check.expect(bad == 0,
                         std::format("{} default samples, no attachment: every pixel of the 36 x 16 scissor box receives between 1 and MAX_FRAMEBUFFER_SAMPLES = {} invocations (ES 3.2 7.11.1)", samples, max_fb_samples_),
                         bad == 0 ? std::format("{} invocations in total, at most {} per pixel", ms.observed.total, most) : std::format("{} bad, {}", bad, first));
            std::cout << std::format("   note: {} default samples: {} invocations for the 576 pixels of the box (one per pixel without per-sample shading); SAMPLES query {}\n", samples,
                                     ms.observed.total, actual);

            // The scissor box starts at the odd pixel (5,3), so pixels outside it share 2 x 2 quads with the box's border pixels. A fragment that
            // fails the scissor test is not shaded (ES 3.2 7.11.1: "the fragment shader will not be executed"), and if an implementation shades its
            // pixel anyway as a helper invocation, stores and atomics have no effect (GLSL ES 3.10 7.1.2). Either way no pixel outside the box is
            // touched, the atomic behind gl_HelperInvocation never runs, and unguarded == guarded.
            int outside = 0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x)
                    outside += (!in_rect(rect, x, y) && ms_helper.observed.cells[static_cast<std::size_t>(y * ms_helper.observed.stride + x)] != 0) ? 1 : 0;
            const Observed& mh = ms_helper.observed;
            check.expect_or_known_deviation(mh.helper_seen == 0 && mh.unguarded == mh.guarded && outside == 0,
                                            std::format("{} default samples: nothing outside the scissor box is touched: no pixel outside it is hit, the atomic behind gl_HelperInvocation never ran, unguarded == guarded == the invocations inside the box", samples),
                                            std::format("guarded {}, unguarded {}, helper_seen {}, {} pixels outside the box hit", mh.guarded, mh.unguarded, mh.helper_seen, outside),
                                            "llvmpipe",
                                            "ES 3.2 7.11.1: a fragment that fails the scissor test is not shaded (and GLSL ES 3.10 7.1.2: a helper invocation has no side effects); with "
                                            "FRAMEBUFFER_DEFAULT_SAMPLES = 4 llvmpipe (and zink on lavapipe) shades the pixels outside the box that share a 2 x 2 quad with it as ordinary "
                                            "invocations (gl_HelperInvocation false) whose atomics take effect. Single-sample framebuffers are correct");
            std::cout << std::format("   note: helper shader, multisample: guarded {}, unguarded {}, helper_seen {}, {} pixels outside the box hit\n", mh.guarded, mh.unguarded, mh.helper_seen, outside);

            // Same question for a multisample renderbuffer attachment (the deviation, if any, is not about attachment-less framebuffers).
            int rb_outside = 0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x)
                    rb_outside += (!in_rect(rect, x, y) && rb_helper.observed.cells[static_cast<std::size_t>(y * rb_helper.observed.stride + x)] != 0) ? 1 : 0;
            const Observed& rh = rb_helper.observed;
            check.expect_or_known_deviation(rh.helper_seen == 0 && rh.unguarded == rh.guarded && rb_outside == 0,
                                            std::format("control: the same triangle and scissor box on a {}-sample RGBA8 renderbuffer attachment: nothing outside the box is touched either", samples),
                                            std::format("guarded {}, unguarded {}, helper_seen {}, {} pixels outside the box hit", rh.guarded, rh.unguarded, rh.helper_seen, rb_outside),
                                            "llvmpipe",
                                            "same rule as above (ES 3.2 7.11.1 / GLSL ES 3.10 7.1.2): llvmpipe and zink on lavapipe also shade the quad neighbours outside the scissor box of a multisample "
                                            "renderbuffer framebuffer as ordinary invocations, so the deviation is about multisampling, not about attachment-less framebuffers");

            // Contrast: the single-sample framebuffer runs exactly one invocation per fragment (ES 3.2 7.11.1: "exactly one when the value of SAMPLE_BUFFERS is zero"),
            // and its helper invocations (the same quad neighbours) are silent.
            expect_case(check, single, "single-sample control: the same triangle and box in a framebuffer with SAMPLE_BUFFERS = 0 run exactly one invocation per fragment, 576 in all, in exactly the box (7.11.1)");
            const Observed& sh = single_helper.observed;
            check.expect(sh.guarded == 576 && sh.unguarded == 576 && sh.helper_seen == 0,
                         "single-sample control, helper shader: the odd-origin scissor box makes its quad neighbours helpers, and they leave no trace (guarded = unguarded = 576, helper_seen = 0)",
                         std::format("guarded {}, unguarded {}, helper_seen {}", sh.guarded, sh.unguarded, sh.helper_seen));
        }

        void check_images(demo::Check& check)
        {
            const ImageResult r = run_images();
            const TriModel m = model_triangles({kTriangles.begin(), kTriangles.end()}, kW, kH, std::nullopt);
            int bad_covered = 0, bad_marker = 0;
            std::string first_covered, first_marker;
            std::uint64_t covered_pixels = 0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x)
                {
                    const std::size_t i = static_cast<std::size_t>(y * kW + x);
                    const bool hit = m.hits[i] != 0;
                    covered_pixels += hit ? 1 : 0;
                    const std::uint32_t want_covered = hit ? 1u : 0u;
                    if (r.covered[i] != want_covered && bad_covered++ == 0) first_covered = std::format("({},{}): {} expected {}", x, y, r.covered[i], want_covered);
                    const std::uint32_t want_marker = hit ? ((static_cast<std::uint32_t>(y) << 16) | static_cast<std::uint32_t>(x)) : 0u;
                    if (r.marker[i] != want_marker && bad_marker++ == 0) first_marker = std::format("({},{}): {:#x} expected {:#x}", x, y, r.marker[i], want_marker);
                }
            check.expect(bad_covered == 0, "image variant: imageStore(1) on an r32ui image (unit 1) marks exactly the pixels the model covers (no attachment, glBindImageTexture, FRAMEBUFFER_BARRIER_BIT before the read)",
                         bad_covered == 0 ? std::format("{} covered pixels", covered_pixels) : std::format("{} pixels differ, first {}", bad_covered, first_covered));
            check.expect(bad_marker == 0, "image variant: imageStore wrote (y << 16 | x), the invocation's own window position, at each covered pixel (unit 0) and nothing elsewhere",
                         bad_marker == 0 ? std::string("exact") : std::format("{} pixels differ, first {}", bad_marker, first_marker));
            check.expect_no_gl_error("image variant: no GL error");
        }

        void check_early_fragment_tests(demo::Check& check)
        {
            const HitMap& a_model = a_model_;
            const HitMap& b_all = b_all_;
            const HitMap& b_visible = b_visible_;
            const std::uint64_t rasterized = sum(b_all), visible = sum(b_visible);
            check.expect(visible < rasterized && visible > 0 && sum(a_model) == 768,
                         "test design: B rasterizes more fragments than are visible (the occluded part is not empty), so counting them tells early and late tests apart",
                         std::format("B rasterizes {}, {} are visible", rasterized, visible));

            // Early fragment tests: exact.
            {
                Case a = early_.a;
                a.expected = a_model;
                expect_case(check, a, "layout(early_fragment_tests), depth FBO: draw A (nothing in front of it) runs the shader for all 768 fragments of its rectangle");
                Case b = early_.b;
                b.expected = b_visible;
                const MapCompare m = compare_map(b);
                check.expect(b.observed.total == visible && m.mismatches == 0 && m.guard_hits == 0,
                             std::format("layout(early_fragment_tests): the shader runs for exactly the {} visible fragments of B and for no occluded one (B rasterizes {})", visible, rasterized),
                             m.mismatches == 0 ? std::format("{} invocations, map exact", b.observed.total) : std::format("{} invocations, {} pixels differ {}", b.observed.total, m.mismatches, m.first));
                std::cout << std::format("   note: early tests: B ran {} invocations ({} visible, {} rasterized)\n", b.observed.total, visible, rasterized);
            }

            // Late tests: the visible fragments are among the invocations (ES 3.2 7.11.1: a fragment that never contributes may not run, one that does must).
            {
                const Case& b = late_.b;
                const MapCompare vis = [&] {
                    Case c = b;
                    c.expected = b_visible;
                    // only look at the visible pixels
                    MapCompare r;
                    for (int y = 0; y < kH; ++y)
                        for (int x = 0; x < kW; ++x)
                            if (b_visible[static_cast<std::size_t>(y * kW + x)] != 0 && b.observed.cells[static_cast<std::size_t>(y * b.observed.stride + x)] != 1 && r.mismatches++ == 0)
                                r.first = std::format("({},{}) ran {} times", x, y, b.observed.cells[static_cast<std::size_t>(y * b.observed.stride + x)]);
                    return r;
                }();
                int outside_b = 0, occluded_run = 0;
                for (int y = 0; y < kH; ++y)
                    for (int x = 0; x < kW; ++x)
                    {
                        const std::uint32_t got = b.observed.cells[static_cast<std::size_t>(y * b.observed.stride + x)];
                        const std::size_t i = static_cast<std::size_t>(y * kW + x);
                        if (b_all[i] == 0 && got != 0) ++outside_b;
                        if (b_all[i] != 0 && b_visible[i] == 0 && got != 0) ++occluded_run;
                    }
                check.expect(vis.mismatches == 0 && outside_b == 0 && b.observed.total >= visible && b.observed.total <= rasterized,
                             std::format("without early tests: all {} visible fragments of B run exactly once, nothing runs outside B, and the total lies between the {} visible and the {} rasterized fragments", visible, visible, rasterized),
                             std::format("{} invocations, {} visible pixels wrong {}, {} outside B", b.observed.total, vis.mismatches, vis.first, outside_b));
                std::cout << std::format("   note: late tests: B ran {} invocations, {} of them for occluded fragments (early: {})\n", b.observed.total, occluded_run, early_.b.observed.total);
            }

            // The depth buffer is the same either way: the occluded fragments of B must not have changed it (probes, early program, depth writes off).
            const std::array<std::pair<const EarlySequence*, std::string_view>, 2> sequences = {{{&early_, "early"}, {&late_, "late"}}};
            for (const auto& [seq, name] : sequences)
            {
                const Case& p1 = seq->probe_half;
                const Case& p2 = seq->probe_far;
                const MapCompare m1 = compare_map(p1), m2 = compare_map(p2);
                check.expect(m1.mismatches == 0 && m2.mismatches == 0 && m1.guard_hits == 0 && m2.guard_hits == 0 && p1.observed.total == 2048 - 768 && p2.observed.total == 2048 - 768 - visible,
                             std::format("{} variant: the depth buffer afterwards holds A's 0.25, B's 0.75 where A is not in front, and the cleared 1.0 elsewhere (probe draws at depth 0.5 / 0.9 pass in exactly the predicted {} / {} pixels)", name,
                                         2048 - 768, 2048 - 768 - visible),
                             std::format("probe 0.5: {} fragments {} mismatches; probe 0.9: {} fragments {} mismatches", p1.observed.total, m1.mismatches, p2.observed.total, m2.mismatches));
            }

            // GLSL ES 3.10 section 4.4.1 (3.20 section 4.4.1.3, same words): with early_fragment_tests "it is an error to statically write to
            // gl_FragDepth in the fragment shader". The text does not say compile-time error, and section 1.3 leaves the stage
            // open: "there are many errors which may be detected either at compile or link time, depending on the
            // implementation." So the check builds the shader together with a vertex shader and requires that compiling or
            // linking fails.
            std::string log;
            const mgl::ShaderId bad = demo::compile_shader(mgl::ShaderType::Fragment, source_of_bad_early_depth(), log);
            bool linked = false;
            if (bad.value != 0)
            {
                std::string vlog;
                const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, rect_vert_source_, vlog);
                if (vs.value == 0) throw demo::Failure("shaders/rect.vert failed to compile:\n" + vlog);
                const std::array<mgl::ShaderId, 2> both = {vs, bad};
                std::string llog;
                const mgl::ProgramId p = demo::link_program(both, llog);
                linked = p.value != 0;
                if (linked) mgl::glDeleteProgram(p);
                mgl::glDeleteShader(vs);
                mgl::glDeleteShader(bad);
            }
            std::cout << std::format("   note: early_fragment_tests + static gl_FragDepth write: compile {}, link {}\n", bad.value != 0 ? "succeeded" : "failed",
                                     bad.value == 0 ? "not attempted" : (linked ? "succeeded" : "failed"));
            const char* const driver = renderer_has("ANGLE") ? "ANGLE" : "Mesa";
            check.expect_or_known_deviation(!linked, "layout(early_fragment_tests) with a static write to gl_FragDepth fails to compile or to link (GLSL ES 3.10 sections 4.4.1, 1.3)",
                                            "the shader compiled and the program linked", driver,
                                            std::format("GLSL ES 3.10 4.4.1: 'it is an error to statically write to gl_FragDepth' with early_fragment_tests; {} "
                                                        "reports the error neither at compile nor at link time", driver));
        }

        std::string source_of_bad_early_depth() const { return bad_source_; }

        void check_helper_invocations(demo::Check& check)
        {
            const int candidates = helper_candidates(helper_triangles_.expected, kW, kH);
            const int rect_candidates = helper_candidates(helper_rect_.expected, kW, kH);
            check.expect(candidates >= 100 && rect_candidates >= 20,
                         "test design: the triangles leave >= 100 uncovered pixels in partly covered 2 x 2 quads (and the odd-aligned rectangle >= 20): a helper that stored would show up in the counts",
                         std::format("{} and {} candidate helper pixels", candidates, rect_candidates));
            for (const Case* c : {&helper_triangles_, &helper_rect_})
            {
                const Observed& o = c->observed;
                const std::uint64_t cover = sum(c->expected);
                const MapCompare m = compare_map(*c);
                check.expect(o.guarded == cover && o.unguarded == cover,
                             std::format("{}: the {} fragments are counted exactly once each, both behind if (!gl_HelperInvocation) and by an unguarded atomic (helper invocations add nothing)", c->name, cover),
                             std::format("guarded {}, unguarded {}, expected {}", o.guarded, o.unguarded, cover));
                check.expect(o.helper_seen == 0, std::format("{}: an atomic executed only when gl_HelperInvocation is true leaves its counter at 0 (stores/atomics by helpers have no effect)", c->name),
                             std::format("helper_seen = {}", o.helper_seen));
                check.expect(m.mismatches == 0 && m.guard_hits == 0, std::format("{}: the per-pixel map of the unguarded atomic equals the coverage model (no hit on an uncovered pixel)", c->name),
                             std::format("{} pixels differ {}, {} outside", m.mismatches, m.first, m.guard_hits));
                // dFdx(gl_FragCoord.x) is 1 and dFdy(gl_FragCoord.y) is 1 for any approximation of a derivative of a linear function.
                check.expect(o.deriv_bad == 0, std::format("{}: dFdx(gl_FragCoord.x) and dFdy(gl_FragCoord.y) are 1 (within 0.01) in every real invocation, also next to uncovered pixels (sanity tolerance: the spec leaves derivative precision open)", c->name),
                             std::format("{} invocations outside", o.deriv_bad));
            }
            // Negative control: if the helpers had stored, the unguarded count would have exceeded the coverage by up to `candidates`.
            check.expect(helper_triangles_.observed.unguarded + static_cast<std::uint32_t>(candidates) > sum(helper_triangles_.expected),
                         "negative control: with helper pixels counted (as a leaking implementation would) the total would exceed the coverage, so the equality above can fail");
        }

        // ------------------------------------------------------------------------------------------
        // Members.
        // ------------------------------------------------------------------------------------------
        Prog prog_rect_, prog_rect_early_, prog_tri_, prog_tri_helper_, prog_rect_helper_, prog_tri_image_;
        mgl::BufferId ssbo_{};
        std::array<mgl::FramebufferId, 6> fbos_{};
        std::array<mgl::RenderbufferId, 3> rbos_{};
        std::array<mgl::TextureId, 2> tex_{};
        std::array<mgl::TextureId, 8> tile_tex_{};
        mgl::FramebufferId fbo_read_{};
        std::vector<mgl::FramebufferId> extra_fbos_;
        HitMap a_model_, b_all_, b_visible_, probe_half_, probe_far_;
        int max_fb_width_ = 0, max_fb_height_ = 0, max_fb_samples_ = 0;
        std::array<int, 2> max_viewport_{};
        int tri_on_edge_ = 0;
        std::string bad_source_;
        std::string rect_vert_source_;

        Case full_, sub_, scissored_, small_viewport_, big_viewport_, neg_viewport_, triangles_, triangles_scissored_, attachment_wins_, detached_, helper_triangles_, helper_rect_;
        EarlySequence early_, late_;
    };
}

int main(int argc, char** argv)
{
    static const demo::Info info{
        .name = "es31-framebuffer-no-attachments",
        .title = "Framebuffer without attachments",
        .min_es = {3, 1},
        .summary = "A framebuffer object with no attachments (glFramebufferParameteri size and samples): rasterization still happens and is observed through "
                   "shader storage buffer and image writes against a CPU model; early_fragment_tests and gl_HelperInvocation side-effect counts.",
        .controls = "",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 1,
        .depth_bits = 0,
        .stencil_bits = 0,
    };
    return demo::run<NoAttachments>(argc, argv, info);
}
