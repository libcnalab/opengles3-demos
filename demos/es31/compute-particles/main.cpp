// es31-compute-particles
//
// A GPU particle fountain in OpenGL ES 3.1. 16384 particles live in one shader storage buffer (std430 array of
// {vec3 pos; float age; vec3 vel; float life;}). Every frame
//
//   1. glMemoryBarrier(SHADER_STORAGE_BARRIER_BIT)  - the previous step's stores and the previous draw's vertex
//                                                      fetches complete before this step writes (ES 3.2 7.11.2),
//   2. glDispatchCompute(128, 1, 1)                 - simulate.comp integrates one step: gravity, a vortex around the
//                                                      vertical axis, drag, an inelastic ground bounce; dead particles
//                                                      respawn from a hash of (index, step) at one of three nozzles
//                                                      that the CPU turns a little every step (a sprinkler, so the
//                                                      jets become spiral arms), and an atomic counter counts the
//                                                      respawns,
//   3. glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT) - vertex data sourced from the buffer after the barrier
//                                                      reflects the shader stores before it,
//   4. glDrawArraysInstanced                        - the SAME buffer object, bound as ARRAY_BUFFER, feeds a per-
//                                                      instance attribute pair (divisor 1); each instance is a camera-
//                                                      facing quad, blended additively over a ground grid.
//
// The CPU never touches the particle data while the demo runs.
//
// Validation (--validate), see README.md:
//   * the std430 layout of the block as reported by glGetProgramResourceiv,
//   * step-wise: the developed fountain is copied into a scratch buffer and stepped three times; after every step the
//     whole buffer is read back (glMapBufferRange) and every component is compared with a CPU evaluation of the same
//     expression tree that carries a rigorous rounding-error bound derived from GLSL ES 3.10 section 4.7.1,
//   * trajectory: 4000 particles bound with glBindBufferRange (p.length() == 4000, the guard beyond the range stays
//     untouched) run 90 steps; positions/velocities within the propagated bound, ages and lives exactly equal, the
//     atomic respawn count exactly equal; negative controls (gravity +2 %, one step fewer) must be rejected,
//   * render: a 17-particle validation scene whose uploaded positions are outside the view is drawn before and after
//     one compute step; the CPU-projected pixels are background before and the additive particle colour after.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
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
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
namespace m = demo::math;
using demo::Rgba8;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Data layout shared with simulate.comp (std430, 32 bytes per element)
    // ---------------------------------------------------------------------------------------------
    struct Particle
    {
        float px, py, pz, age;
        float vx, vy, vz, life;
    };
    static_assert(sizeof(Particle) == 32);

    constexpr std::uint32_t kLocalSize = 128;            // local_size_x of simulate.comp
    constexpr std::uint32_t kParticles = 16384;          // display system: 128 work groups
    constexpr int kWarmupSteps = 300;                    // > the longest initial life, so the fountain is developed
    constexpr float kDt = 1.0f / 60.0f;
    constexpr float kFovyDeg = 45.0f;
    constexpr std::array<float, 3> kBackground = {0.012f, 0.016f, 0.035f};

    // All parameters of one step, exactly as uploaded (float / uint), so that the CPU model uses the same values.
    struct SimParams
    {
        float dt = kDt;
        float gdt = 9.81f * kDt;
        float swirl = 0.9f;
        float pull = 0.0f;
        float soft = 0.25f;
        float damp = 1.0f - 0.25f * kDt;
        float ground = 0.0f;
        float rest = 0.35f;
        float fric = 0.6f;
        float axis_x = 0.0f, axis_z = 0.0f;
        float em_x = 0.0f, em_y = 0.04f, em_z = 0.0f;
        float jitter = 0.10f;
        float radial = 1.9f;
        float spread = 0.7f;
        float side = 0.45f;
        float spin = 2.1f;  // rad/s of the nozzle rotation (applied on the CPU, see arm_dirs)
        float speed = 5.0f;
        float speed_var = 0.6f;
        std::uint32_t life_min = 85;
        std::uint32_t life_var = 50;
    };

    std::uint32_t hash32(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }
    float unit24(std::uint32_t h) { return static_cast<float>(h >> 8) * 0x1p-24f; }

    /// Horizontal unit directions of the three nozzles at `step` (x0, z0, x1, z1, x2, z2). Computed on the CPU (trig
    /// has no defined precision in GLSL ES 3.10 section 4.7.1) and uploaded as u_arm[3]; the model uses the same floats.
    std::array<float, 6> arm_dirs(const SimParams& sp, std::uint32_t step)
    {
        std::array<float, 6> d{};
        const double base = std::fmod(static_cast<double>(step) * static_cast<double>(sp.spin) * static_cast<double>(sp.dt), 6.283185307179586);
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double a = base + 2.0943951023931953 * static_cast<double>(k);
            d[2 * k] = static_cast<float>(std::cos(a));
            d[2 * k + 1] = static_cast<float>(std::sin(a));
        }
        return d;
    }

    // ---------------------------------------------------------------------------------------------
    // CPU model with a rigorous rounding-error bound.
    //
    // GLSL ES 3.10 section 4.7.1 (highp): a + b, a - b and a * b are correctly rounded and "the rounding mode is not
    // defined but must not affect the result by more than 1 ULP"; a*b+c is one correctly rounded operation or two;
    // a / b is within 2.5 ULP for |b| in [2^-126, 2^126]; comparisons are exact; subnormals may be flushed to 0.
    // ULP(x) <= 2^-23 |x| for every normal float, so each GPU operation on GPU inputs x', y' returns
    // op(x', y') (1 + d) with |d| <= k 2^-23 (k = 1, or 2.5 for division), plus at most 2^-126 for a flushed subnormal.
    //
    // E carries the exact-arithmetic value (in double) of the expression applied to the inputs and a bound e on
    // |GPU result - value|. Propagation is first-order exact: |x'y' - xy| <= |x| e_y + |y| e_x + e_x e_y and
    // |x'/y' - x/y| <= (|y| e_x + |x| e_y) / (|y| (|y| - e_y)). A fused a*b+c is never worse than the two-rounding bound.
    // Double arithmetic itself adds ~1e-16 relative, covered by a 1e-12 relative slack in the comparison.
    // ---------------------------------------------------------------------------------------------
    constexpr double kUlp = 0x1p-23;
    constexpr double kFlush = 0x1p-126;

    struct E
    {
        double v = 0.0;
        double e = 0.0;
    };
    E exact(float f) { return {static_cast<double>(f), 0.0}; }
    E rounded(double v, double propagated, double ulps) { return {v, propagated + ulps * kUlp * (std::abs(v) + propagated) + kFlush}; }
    E operator+(E a, E b) { return rounded(a.v + b.v, a.e + b.e, 1.0); }
    E operator-(E a, E b) { return rounded(a.v - b.v, a.e + b.e, 1.0); }
    E operator-(E a) { return {-a.v, a.e}; }  // negation is exact
    E operator*(E a, E b) { return rounded(a.v * b.v, std::abs(a.v) * b.e + std::abs(b.v) * a.e + a.e * b.e, 1.0); }
    E operator/(E a, E b)
    {
        const double den = std::abs(b.v) - b.e;
        const double prop = den > 0.0 ? (std::abs(b.v) * a.e + std::abs(a.v) * b.e) / (std::abs(b.v) * den) : std::numeric_limits<double>::infinity();
        return rounded(a.v / b.v, prop, 2.5);
    }

    struct ModelParticle
    {
        E px, py, pz;
        float age = 0.0f;
        E vx, vy, vz;
        float life = 0.0f;
    };

    ModelParticle from_gpu(const Particle& q)
    {
        return {exact(q.px), exact(q.py), exact(q.pz), q.age, exact(q.vx), exact(q.vy), exact(q.vz), q.life};
    }

    enum class Branch
    {
        Spawn,
        Fly,
        Bounce
    };

    struct StepResult
    {
        ModelParticle out;        // the branch the model is sure about (or the "no bounce" branch if ambiguous)
        ModelParticle alt;        // the "bounce" branch when the bounce decision is ambiguous
        Branch branch = Branch::Fly;
        bool ambiguous = false;   // the GPU's y may lie on either side of the ground within the rounding bound
    };

    // The step of simulate.comp, operation by operation.
    StepResult model_step(const SimParams& sp, std::uint32_t index, std::uint32_t step, ModelParticle q)
    {
        StepResult r;
        q.age += 1.0f;  // integer-valued floats below 2^24: exact on both sides
        if (q.age >= q.life)
        {
            const std::uint32_t h0 = hash32(index * 0x9e3779b9u + hash32(step ^ 0x632be5abu));
            const std::uint32_t h1 = hash32(h0), h2 = hash32(h1), h3 = hash32(h2), h4 = hash32(h3), h5 = hash32(h4), h6 = hash32(h5);
            const E half = exact(0.5f);
            const std::array<float, 6> arms = arm_dirs(sp, step);
            const E dx = exact(arms[2 * (h5 % 3u)]), dz = exact(arms[2 * (h5 % 3u) + 1]);
            const E along = exact(sp.radial) + (exact(unit24(h2)) - half) * exact(sp.spread);
            const E across = (exact(unit24(h4)) - half) * exact(sp.side);
            q.px = exact(sp.em_x) + (exact(unit24(h0)) - half) * exact(sp.jitter);
            q.py = exact(sp.em_y);
            q.pz = exact(sp.em_z) + (exact(unit24(h1)) - half) * exact(sp.jitter);
            q.vx = dx * along - dz * across;
            q.vy = exact(sp.speed) + exact(unit24(h3)) * exact(sp.speed_var);
            q.vz = dz * along + dx * across;
            q.age = 0.0f;
            q.life = static_cast<float>(sp.life_min + h6 % sp.life_var);
            r.out = q;
            r.branch = Branch::Spawn;
            return r;
        }
        const E dt = exact(sp.dt), damp = exact(sp.damp);
        const E rx = q.px - exact(sp.axis_x);
        const E rz = q.pz - exact(sp.axis_z);
        const E d2 = rx * rx + rz * rz + exact(sp.soft);
        const E ax = (-exact(sp.swirl) * rz - exact(sp.pull) * rx) / d2;
        const E az = (exact(sp.swirl) * rx - exact(sp.pull) * rz) / d2;
        q.vx = (q.vx + ax * dt) * damp;
        q.vy = (q.vy - exact(sp.gdt)) * damp;
        q.vz = (q.vz + az * dt) * damp;
        q.px = q.px + q.vx * dt;
        q.py = q.py + q.vy * dt;
        q.pz = q.pz + q.vz * dt;

        const double g = static_cast<double>(sp.ground);
        ModelParticle b = q;
        b.py = exact(sp.ground) + (exact(sp.ground) - q.py) * exact(sp.rest);
        b.vy = -q.vy * exact(sp.rest);
        b.vx = q.vx * exact(sp.fric);
        b.vz = q.vz * exact(sp.fric);
        if (q.py.v + q.py.e < g)
        {
            r.out = b;
            r.branch = Branch::Bounce;
        }
        else if (q.py.v - q.py.e >= g)
        {
            r.out = q;
            r.branch = Branch::Fly;
        }
        else
        {
            // Within the rounding bound of the plane: section 4.7.1 permits either side, so either branch conforms.
            r.out = q;
            r.alt = b;
            r.ambiguous = true;
        }
        return r;
    }

    // |gpu - model| / bound of one component (<= 1 means within the bound).
    double ratio(float gpu, const E& model)
    {
        const double bound = model.e + 1e-12 * std::abs(model.v) + 1e-30;
        return std::abs(static_cast<double>(gpu) - model.v) / bound;
    }

    double particle_ratio(const Particle& g, const ModelParticle& mdl)
    {
        if (g.age != mdl.age || g.life != mdl.life) return std::numeric_limits<double>::infinity();  // integer bookkeeping is exact
        return std::max({ratio(g.px, mdl.px), ratio(g.py, mdl.py), ratio(g.pz, mdl.pz), ratio(g.vx, mdl.vx), ratio(g.vy, mdl.vy), ratio(g.vz, mdl.vz)});
    }

    // ---------------------------------------------------------------------------------------------
    // Camera
    // ---------------------------------------------------------------------------------------------
    struct Camera
    {
        m::Mat4 view, proj;
        float fovy = m::radians(kFovyDeg);
        float aspect = 1.0f;
    };

    Camera display_camera(int width, int height, float yaw, float elevation_deg, float distance_scale)
    {
        Camera c;
        c.aspect = static_cast<float>(width) / static_cast<float>(height);
        // The fountain is about 5 units wide and 2.5 high: in a portrait surface back off so that it still fits across.
        const float distance = 7.0f * distance_scale * std::max(1.0f, 0.85f / c.aspect);
        const float el = m::radians(elevation_deg);
        const m::Vec3 center{0.0f, 0.55f, 0.0f};
        const m::Vec3 eye = center + m::Vec3{std::sin(yaw) * std::cos(el), std::sin(el), std::cos(yaw) * std::cos(el)} * distance;
        c.view = m::look_at(eye, center, {0.0f, 1.0f, 0.0f});
        c.proj = m::perspective(c.fovy, c.aspect, 0.1f, 100.0f);
        return c;
    }

    // ---------------------------------------------------------------------------------------------
    // Render-check scene: 4 x 4 particles in the plane z = 0 plus a second particle on top of target 5.
    // ---------------------------------------------------------------------------------------------
    constexpr int kGridN = 4;
    constexpr std::uint32_t kRenderParticles = kGridN * kGridN + 1;
    constexpr float kRenderHalf = 0.12f;        // world half size of the validation quads
    constexpr float kStaleLift = 9.0f;          // the uploaded positions are this far above the targets (outside the view)
    constexpr Rgba8 kFlatColor8 = {60, 100, 110, 255};
    constexpr Rgba8 kClear8 = {8, 10, 16, 255};

    m::Vec3 render_target(std::uint32_t i)
    {
        const std::uint32_t k = i < kGridN * kGridN ? i : 5u;  // the extra particle overlaps target 5
        const float gx = static_cast<float>(k % kGridN), gy = static_cast<float>(k / kGridN);
        return {-1.5f + gx, 0.0f + 0.8f * gy, 0.0f};
    }

    Camera render_check_camera(int width, int height)
    {
        Camera c;
        c.aspect = static_cast<float>(width) / static_cast<float>(height);
        const float f = 1.0f / std::tan(c.fovy * 0.5f);
        // Fit the grid (x in [-1.5, 1.5], y in [0, 2.4] around 1.2) into 70 % of NDC in both directions at any aspect.
        const float d = std::max(1.5f * f / (0.7f * c.aspect), 1.2f * f / 0.7f);
        c.view = m::look_at({0.0f, 1.2f, d}, {0.0f, 1.2f, 0.0f}, {0.0f, 1.0f, 0.0f});
        c.proj = m::perspective(c.fovy, c.aspect, 0.1f, 100.0f);
        return c;
    }

    struct Projected
    {
        double x = 0, y = 0;    // window coordinates
        double half_px_x = 0;   // half extent of the quad in pixels
        double half_px_y = 0;
        bool in_front = false;
    };

    Projected project(const Camera& c, m::Vec3 p, double half_world, int width, int height)
    {
        const m::Vec4 eye = c.view * m::Vec4{p.x, p.y, p.z, 1.0f};
        const m::Vec4 clip = c.proj * eye;
        Projected r;
        r.in_front = clip.w > 0.0f;
        const double w = static_cast<double>(clip.w);
        r.x = (static_cast<double>(clip.x) / w * 0.5 + 0.5) * width;
        r.y = (static_cast<double>(clip.y) / w * 0.5 + 0.5) * height;
        r.half_px_x = half_world * static_cast<double>(c.proj.at(0, 0)) / w * 0.5 * width;
        r.half_px_y = half_world * static_cast<double>(c.proj.at(1, 1)) / w * 0.5 * height;
        return r;
    }

    int max_channel_diff(const Rgba8& a, const Rgba8& b)
    {
        int d = 0;
        for (std::size_t i = 0; i < 3; ++i) d = std::max(d, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        return d;
    }

    Rgba8 add8(Rgba8 a, Rgba8 b, int times)
    {
        Rgba8 r{};
        for (std::size_t i = 0; i < 3; ++i) r[i] = static_cast<std::uint8_t>(std::min(255, a[i] + times * b[i]));
        r[3] = 255;
        return r;
    }

    float f8(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

    // ---------------------------------------------------------------------------------------------
    // The demo
    // ---------------------------------------------------------------------------------------------
    struct SimLocations
    {
        mgl::UniformLocation dt, gdt, swirl, pull, soft, damp, ground, rest, fric, axis, emitter, jitter, arm, radial, spread, side, speed, speed_var, life_min, life_var, step;
    };

    struct DrawLocations
    {
        mgl::UniformLocation view, proj, half, min_px, px_scale, intensity, flat, flat_color;
    };

    class ComputeParticles final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 96 || env.height < 96) throw demo::Skip("the surface must be at least 96 x 96 pixels");

            // ---- programs ---------------------------------------------------------------------------------------------
            std::string log;
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, env.load_text("shaders/simulate.comp"), log);
            if (cs.value == 0) throw demo::Failure("simulate.comp failed to compile:\n" + log);
            const std::array<mgl::ShaderId, 1> shaders = {cs};
            prog_sim_ = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            if (prog_sim_.value == 0) throw demo::Failure("simulate.comp failed to link:\n" + log);
            prog_particles_ = demo::build_program(env, "shaders/particle.vert", "shaders/particle.frag");
            prog_grid_ = demo::build_program(env, "shaders/grid.vert", "shaders/grid.frag");

            auto u = [](mgl::ProgramId p, const char* name) { return mgl::glGetUniformLocation(p, name); };
            sim_ = {u(prog_sim_, "u_dt"),     u(prog_sim_, "u_gdt"),     u(prog_sim_, "u_swirl"),   u(prog_sim_, "u_pull"),     u(prog_sim_, "u_soft"),
                    u(prog_sim_, "u_damp"),   u(prog_sim_, "u_ground"),  u(prog_sim_, "u_rest"),    u(prog_sim_, "u_fric"),     u(prog_sim_, "u_axis"),
                    u(prog_sim_, "u_emitter"), u(prog_sim_, "u_jitter"), u(prog_sim_, "u_arm"), u(prog_sim_, "u_radial"), u(prog_sim_, "u_spread"), u(prog_sim_, "u_side"),  u(prog_sim_, "u_speed"),    u(prog_sim_, "u_speed_var"),
                    u(prog_sim_, "u_life_min"), u(prog_sim_, "u_life_var"), u(prog_sim_, "u_step")};
            draw_ = {u(prog_particles_, "u_view"), u(prog_particles_, "u_proj"),     u(prog_particles_, "u_half"), u(prog_particles_, "u_min_px"),
                     u(prog_particles_, "u_px_scale"), u(prog_particles_, "u_intensity"), u(prog_particles_, "u_flat"), u(prog_particles_, "u_flat_color")};
            grid_view_ = u(prog_grid_, "u_view");
            grid_proj_ = u(prog_grid_, "u_proj");
            grid_bg_ = u(prog_grid_, "u_background");

            // ---- buffers ------------------------------------------------------------------------------------------------
            // Initial display state: every particle sits at the emitter and dies after 1..240 steps, so the respawns are
            // spread over the warm-up; after kWarmupSteps every particle has been launched at least once.
            std::vector<Particle> initial(kParticles);
            for (std::uint32_t i = 0; i < kParticles; ++i)
                initial[i] = {0.0f, display_.em_y, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, static_cast<float>(1u + i % 240u)};
            buf_particles_ = make_buffer(std::span<const Particle>(initial), mgl::BufferUsage::DynamicCopy);
            const std::uint32_t zero = 0;
            buf_counter_ = make_buffer(std::span<const std::uint32_t>(&zero, 1), mgl::BufferUsage::DynamicCopy);

            const std::array<float, 8> quad = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};  // triangle strip
            buf_quad_ = make_buffer(std::span<const float>(quad), mgl::BufferUsage::StaticDraw);

            std::vector<float> grid;
            constexpr int kLines = 24;            // lines from -6 to 6 every 0.5
            for (int k = 0; k <= kLines; ++k)
            {
                const float t = -6.0f + 0.5f * static_cast<float>(k);
                const float major = (k % 4 == 0) ? 1.0f : 0.0f;
                for (const float v : {t, major, -6.0f, t, major, 6.0f, -6.0f, major, t, 6.0f, major, t}) grid.push_back(v);
            }
            grid_vertices_ = static_cast<GLsizei>(grid.size() / 3);
            buf_grid_ = make_buffer(std::span<const float>(grid), mgl::BufferUsage::StaticDraw);

            vao_display_ = make_particle_vao(buf_particles_);
            mgl::glGenVertexArrays(1, &vao_grid_);
            mgl::glBindVertexArray(vao_grid_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_grid_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, 12, demo::buffer_offset(0));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // ---- warm-up: the fountain is fully developed before the first frame --------------------------------------
            mgl::glBindBufferBase(mgl::BufferTarget::AtomicCounter, 0, buf_counter_);
            for (int s = 0; s < kWarmupSteps; ++s) simulate(buf_particles_, kParticles, display_, step_++);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            if (!paused_)
            {
                simulate(buf_particles_, kParticles, display_, step_++);
                if (env.interactive) yaw_ += 0.0035f;  // slow automatic orbit in windowed mode
            }
            draw_display(env);
        }

        void on_key(demo::Env& /*env*/, char key) override
        {
            switch (key)
            {
            case ' ': paused_ = !paused_; break;
            case 'a': yaw_ -= 0.08f; break;
            case 'd': yaw_ += 0.08f; break;
            case 'w': distance_ = std::max(0.5f, distance_ - 0.08f); break;
            case 's': distance_ = std::min(3.0f, distance_ + 0.08f); break;
            case 'q': elevation_ = std::min(80.0f, elevation_ + 3.0f); break;
            case 'e': elevation_ = std::max(2.0f, elevation_ - 3.0f); break;
            case 'r':
                yaw_ = kDefaultYaw;
                elevation_ = kDefaultElevation;
                distance_ = 1.0f;
                break;
            default: break;
            }
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::AtomicCounter, 0, mgl::BufferId{0});
            for (mgl::VertexArrayId& v : vaos_) mgl::glDeleteVertexArrays(1, &v);
            mgl::glDeleteVertexArrays(1, &vao_grid_);
            if (!buffers_.empty()) mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            mgl::glDeleteProgram(prog_sim_);
            mgl::glDeleteProgram(prog_particles_);
            mgl::glDeleteProgram(prog_grid_);
            vaos_.clear();
            buffers_.clear();
        }

    private:
        static constexpr float kDefaultYaw = 0.55f;
        static constexpr float kDefaultElevation = 26.0f;

        template <typename T>
        mgl::BufferId make_buffer(std::span<const T> data, mgl::BufferUsage usage)
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, b);
            mgl::glBufferData(mgl::BufferTarget::CopyWrite, static_cast<GLsizeiptr>(data.size_bytes()), data.data(), usage);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
            buffers_.push_back(b);
            return b;
        }

        /// VAO of an instanced billboard draw: attribute 0 = quad corner, attributes 1 and 2 = the two vec4 halves of
        /// each 32-byte std430 particle, advanced once per instance.
        mgl::VertexArrayId make_particle_vao(mgl::BufferId particles)
        {
            mgl::VertexArrayId vao{};
            mgl::glGenVertexArrays(1, &vao);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_quad_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 8, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, particles);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(Particle), demo::buffer_offset(0));
            mgl::glVertexAttribDivisor(mgl::AttribLocation{1u}, 1);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{2u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{2u}, 4, mgl::DataType::Float, GL_FALSE, sizeof(Particle), demo::buffer_offset(16));
            mgl::glVertexAttribDivisor(mgl::AttribLocation{2u}, 1);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            vaos_.push_back(vao);
            return vao;
        }

        void set_sim_uniforms(const SimParams& sp, std::uint32_t step)
        {
            mgl::glUniform1f(sim_.dt, sp.dt);
            mgl::glUniform1f(sim_.gdt, sp.gdt);
            mgl::glUniform1f(sim_.swirl, sp.swirl);
            mgl::glUniform1f(sim_.pull, sp.pull);
            mgl::glUniform1f(sim_.soft, sp.soft);
            mgl::glUniform1f(sim_.damp, sp.damp);
            mgl::glUniform1f(sim_.ground, sp.ground);
            mgl::glUniform1f(sim_.rest, sp.rest);
            mgl::glUniform1f(sim_.fric, sp.fric);
            mgl::glUniform2f(sim_.axis, sp.axis_x, sp.axis_z);
            mgl::glUniform3f(sim_.emitter, sp.em_x, sp.em_y, sp.em_z);
            mgl::glUniform1f(sim_.jitter, sp.jitter);
            const std::array<float, 6> arms = arm_dirs(sp, step);
            mgl::glUniform2fv(sim_.arm, 3, arms.data());
            mgl::glUniform1f(sim_.radial, sp.radial);
            mgl::glUniform1f(sim_.spread, sp.spread);
            mgl::glUniform1f(sim_.side, sp.side);
            mgl::glUniform1f(sim_.speed, sp.speed);
            mgl::glUniform1f(sim_.speed_var, sp.speed_var);
            mgl::glUniform1ui(sim_.life_min, sp.life_min);
            mgl::glUniform1ui(sim_.life_var, sp.life_var);
            mgl::glUniform1ui(sim_.step, step);
        }

        /// One simulation step on the whole buffer (glBindBufferBase: p.length() = buffer size / 32).
        void simulate(mgl::BufferId buffer, std::uint32_t count, const SimParams& sp, std::uint32_t step)
        {
            // Orders the previous step's stores AND the previous draw's vertex fetches before this step's stores.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buffer);
            mgl::glUseProgram(prog_sim_);
            set_sim_uniforms(sp, step);
            mgl::glDispatchCompute((count + kLocalSize - 1) / kLocalSize, 1, 1);
            // The draw sources the same buffer as vertex attributes.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::VertexAttribArray);
        }

        void set_draw_uniforms(const Camera& c, int height, float half, float min_px, float intensity, bool flat, Rgba8 flat_color)
        {
            mgl::glUseProgram(prog_particles_);
            mgl::glUniformMatrix4fv(draw_.view, 1, GL_FALSE, c.view.data());
            mgl::glUniformMatrix4fv(draw_.proj, 1, GL_FALSE, c.proj.data());
            mgl::glUniform1f(draw_.half, half);
            mgl::glUniform1f(draw_.min_px, min_px);
            mgl::glUniform1f(draw_.px_scale, 2.0f * std::tan(c.fovy * 0.5f) / static_cast<float>(height));
            mgl::glUniform1f(draw_.intensity, intensity);
            mgl::glUniform1i(draw_.flat, flat ? 1 : 0);
            mgl::glUniform3f(draw_.flat_color, f8(flat_color[0]), f8(flat_color[1]), f8(flat_color[2]));
        }

        void draw_display(const demo::Env& env)
        {
            const Camera cam = display_camera(env.width, env.height, yaw_, elevation_, distance_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kBackground[0], kBackground[1], kBackground[2], 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            // Ground grid: depth-tested and depth-writing, so it hides nothing but gives the depth cue.
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glDepthFunc(mgl::CompareFunc::Lequal);
            mgl::glDepthMask(GL_TRUE);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glUseProgram(prog_grid_);
            mgl::glUniformMatrix4fv(grid_view_, 1, GL_FALSE, cam.view.data());
            mgl::glUniformMatrix4fv(grid_proj_, 1, GL_FALSE, cam.proj.data());
            mgl::glUniform3f(grid_bg_, kBackground[0], kBackground[1], kBackground[2]);
            mgl::glBindVertexArray(vao_grid_);
            mgl::glDrawArrays(mgl::PrimitiveType::Lines, 0, grid_vertices_);

            // Particles: additive, depth-tested against the grid but not writing depth (order independent).
            mgl::glDepthMask(GL_FALSE);
            mgl::glEnable(mgl::Capability::Blend);
            mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::One);
            set_draw_uniforms(cam, env.height, 0.022f, 1.4f, 0.30f, false, {});
            mgl::glBindVertexArray(vao_display_);
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, static_cast<GLsizei>(kParticles));

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDepthMask(GL_TRUE);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::DepthTest);
        }

        /// Reads `count` particles after shader writes (BUFFER_UPDATE_BARRIER_BIT, ES 3.2 section 7.11.2).
        std::vector<Particle> read_particles(mgl::BufferId buffer, std::size_t first, std::size_t count)
        {
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            std::vector<Particle> out(count);
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, buffer);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::CopyRead, static_cast<GLintptr>(first * sizeof(Particle)),
                                                       static_cast<GLsizeiptr>(count * sizeof(Particle)), mgl::MapBufferAccessMask::Read);
            if (mapped == nullptr) throw demo::Failure("glMapBufferRange(READ) of the particle buffer returned NULL");
            std::memcpy(out.data(), mapped, count * sizeof(Particle));
            (void)mgl::glUnmapBuffer(mgl::BufferTarget::CopyRead);
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, mgl::BufferId{0});
            return out;
        }

        std::uint32_t read_counter()
        {
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            std::uint32_t v = 0;
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, buf_counter_);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::CopyRead, 0, 4, mgl::MapBufferAccessMask::Read);
            if (mapped == nullptr) throw demo::Failure("glMapBufferRange(READ) of the counter returned NULL");
            std::memcpy(&v, mapped, 4);
            (void)mgl::glUnmapBuffer(mgl::BufferTarget::CopyRead);
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, mgl::BufferId{0});
            return v;
        }

        void reset_counter()
        {
            // BUFFER_UPDATE: the BufferSubData write waits for the atomic increments issued before the barrier.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const std::uint32_t zero = 0;
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, buf_counter_);
            mgl::glBufferSubData(mgl::BufferTarget::CopyWrite, 0, std::span<const std::uint32_t>(&zero, 1));
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
        }

        void check_layout(demo::Check& check);
        void check_stepwise(demo::Check& check);
        void check_trajectory(demo::Check& check);
        void check_render(const demo::Env& env, demo::Check& check);

        SimParams display_{};
        std::uint32_t step_ = 0;
        bool paused_ = false;
        float yaw_ = kDefaultYaw, elevation_ = kDefaultElevation, distance_ = 1.0f;

        mgl::ProgramId prog_sim_{}, prog_particles_{}, prog_grid_{};
        SimLocations sim_{};
        DrawLocations draw_{};
        mgl::UniformLocation grid_view_{}, grid_proj_{}, grid_bg_{};
        mgl::BufferId buf_particles_{}, buf_counter_{}, buf_quad_{}, buf_grid_{};
        mgl::VertexArrayId vao_display_{}, vao_grid_{};
        GLsizei grid_vertices_ = 0;
        std::vector<mgl::BufferId> buffers_;
        std::vector<mgl::VertexArrayId> vaos_;
    };

    // ---------------------------------------------------------------------------------------------------------
    // 1. std430 layout as the GL reports it (ES 3.2 section 7.6.2.2 rules; program interface BUFFER_VARIABLE).
    // ---------------------------------------------------------------------------------------------------------
    void ComputeParticles::check_layout(demo::Check& check)
    {
        struct Expect
        {
            const char* name;
            GLint offset;
        };
        constexpr std::array<Expect, 4> members = {{{"p[0].pos", 0}, {"p[0].age", 12}, {"p[0].vel", 16}, {"p[0].life", 28}}};
        for (const Expect& e : members)
        {
            const GLuint index = mgl::glGetProgramResourceIndex(prog_sim_, mgl::ProgramInterface::BufferVariable, e.name);
            if (!check.expect(index != GL_INVALID_INDEX, std::format("buffer variable {} is active", e.name))) continue;
            const std::array<mgl::ProgramResourceProperty, 2> props = {mgl::ProgramResourceProperty::Offset, mgl::ProgramResourceProperty::TopLevelArrayStride};
            std::array<GLint, 2> v{-1, -1};
            mgl::glGetProgramResourceiv(prog_sim_, mgl::ProgramInterface::BufferVariable, index, static_cast<GLsizei>(props.size()), props.data(),
                                        static_cast<GLsizei>(v.size()), nullptr, v.data());
            check.expect_eq(v[0], e.offset, std::format("std430 OFFSET of {} (vec3 aligned to 16, size 12; the float packs after it)", e.name));
            check.expect_eq(v[1], GLint{32}, std::format("TOP_LEVEL_ARRAY_STRIDE of {} = 32 bytes = sizeof(Particle) used by the vertex attributes", e.name));
        }
        const GLuint block = mgl::glGetProgramResourceIndex(prog_sim_, mgl::ProgramInterface::ShaderStorageBlock, "Particles");
        if (check.expect(block != GL_INVALID_INDEX, "shader storage block Particles is active"))
        {
            const mgl::ProgramResourceProperty prop = mgl::ProgramResourceProperty::BufferDataSize;
            GLint size = -1;
            mgl::glGetProgramResourceiv(prog_sim_, mgl::ProgramInterface::ShaderStorageBlock, block, 1, &prop, 1, nullptr, &size);
            check.expect_eq(size, GLint{32}, "BUFFER_DATA_SIZE of Particles = one element of the runtime-sized array (ES 3.2 section 7.6.2.2)");
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // 2. Step-wise: one step from bit-identical inputs, every component within the derived rounding bound.
    // ---------------------------------------------------------------------------------------------------------
    void ComputeParticles::check_stepwise(demo::Check& check)
    {
        constexpr int kSteps = 3;
        const std::size_t bytes = std::size_t{kParticles} * sizeof(Particle);
        const std::vector<Particle> zeros(kParticles, Particle{});
        const mgl::BufferId scratch = make_buffer(std::span<const Particle>(zeros), mgl::BufferUsage::DynamicCopy);

        // Copy the developed fountain (BUFFER_UPDATE: CopyBufferSubData reads after the shader stores).
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glBindBuffer(mgl::BufferTarget::CopyRead, buf_particles_);
        mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, scratch);
        mgl::glCopyBufferSubData(mgl::BufferTarget::CopyRead, mgl::BufferTarget::CopyWrite, 0, 0, static_cast<GLsizeiptr>(bytes));
        mgl::glBindBuffer(mgl::BufferTarget::CopyRead, mgl::BufferId{0});
        mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});

        std::vector<Particle> before = read_particles(scratch, 0, kParticles);
        int bad = 0, spawns_total = 0, bounces_total = 0, ambiguous_total = 0;
        double worst = 0.0;
        std::string first_bad;
        for (int k = 0; k < kSteps; ++k)
        {
            const std::uint32_t step = 900000u + static_cast<std::uint32_t>(k);
            reset_counter();
            simulate(scratch, kParticles, display_, step);
            const std::vector<Particle> after = read_particles(scratch, 0, kParticles);
            const std::uint32_t gpu_spawns = read_counter();
            std::uint32_t cpu_spawns = 0;
            for (std::uint32_t i = 0; i < kParticles; ++i)
            {
                const StepResult r = model_step(display_, i, step, from_gpu(before[i]));
                double q = particle_ratio(after[i], r.out);
                if (r.ambiguous)
                {
                    q = std::min(q, particle_ratio(after[i], r.alt));
                    ++ambiguous_total;
                }
                if (r.branch == Branch::Spawn) ++cpu_spawns;
                if (r.branch == Branch::Bounce) ++bounces_total;
                worst = std::max(worst, q);
                if (q > 1.0 && bad++ == 0)
                    first_bad = std::format("step {} particle {}: gpu pos ({}, {}, {}) vel ({}, {}, {}) age {} life {}; model pos ({}, {}, {}) +- ({:.3g}, {:.3g}, {:.3g}) age {} life {}", k, i,
                                            after[i].px, after[i].py, after[i].pz, after[i].vx, after[i].vy, after[i].vz, after[i].age, after[i].life, r.out.px.v,
                                            r.out.py.v, r.out.pz.v, r.out.px.e, r.out.py.e, r.out.pz.e, r.out.age, r.out.life);
            }
            spawns_total += static_cast<int>(cpu_spawns);
            check.expect_eq(gpu_spawns, cpu_spawns, std::format("step {}: atomic counter = number of respawns predicted by the CPU (exact integer decisions)", k));
            before = after;
        }
        std::cout << std::format("[compute-particles] step-wise: {} particles x {} steps, max |gpu - model| / bound = {:.4f}, {} respawns, {} bounces, {} ambiguous bounce decisions\n",
                                 kParticles, kSteps, worst, spawns_total, bounces_total, ambiguous_total);
        check.expect(bad == 0, "step-wise: every position / velocity component within the GLSL ES 3.10 4.7.1 rounding bound of the CPU model, ages and lives exact",
                     std::format("{} particle-steps outside, max ratio {:.3g}; first: {}", bad, worst, first_bad));
        check.expect(spawns_total > 0 && bounces_total > 0, "step-wise: the compared steps exercise the respawn and the bounce branch",
                     std::format("{} respawns, {} bounces", spawns_total, bounces_total));
    }

    // ---------------------------------------------------------------------------------------------------------
    // 3. Trajectory: 90 steps from a CPU-defined state, compared with the CPU model run independently.
    // ---------------------------------------------------------------------------------------------------------
    void ComputeParticles::check_trajectory(demo::Check& check)
    {
        constexpr std::uint32_t kN = 4000;       // not a multiple of 128: the last work group straddles the range end
        constexpr std::uint32_t kAlloc = 4096;   // particles 4000..4095 are a guard that no invocation may touch
        constexpr int kSteps = 90;
        SimParams sp = display_;
        sp.ground = -1000.0f;  // no bounce in this check: the only branch (respawn) is decided by exact integers

        std::vector<Particle> init(kAlloc);
        for (std::uint32_t i = 0; i < kAlloc; ++i)
        {
            const std::uint32_t h = hash32(i + 0x51ed270bu);
            auto r = [&](std::uint32_t salt, float lo, float hi) { return lo + (hi - lo) * unit24(hash32(h ^ salt)); };
            const float life = static_cast<float>(30u + hash32(h ^ 7u) % 150u);
            init[i] = {r(1, -1.6f, 1.6f), r(2, 0.3f, 3.0f), r(3, -1.6f, 1.6f), std::floor(r(4, 0.0f, 0.999f) * life),
                       r(5, -2.0f, 2.0f), r(6, -1.5f, 4.0f), r(7, -2.0f, 2.0f), life};
        }
        Particle guard{};
        std::memset(&guard, 0xA5, sizeof guard);
        for (std::uint32_t i = kN; i < kAlloc; ++i) init[i] = guard;
        const mgl::BufferId buf = make_buffer(std::span<const Particle>(init), mgl::BufferUsage::DynamicCopy);

        reset_counter();
        for (int s = 0; s < kSteps; ++s)
        {
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);  // step s+1 reads what step s stored
            // Range binding: p.length() = 4000 * 32 / 32. 32 groups = 4096 invocations; 96 must return early.
            mgl::glBindBufferRange(mgl::BufferTarget::ShaderStorage, 0, buf, 0, static_cast<GLsizeiptr>(kN * sizeof(Particle)));
            mgl::glUseProgram(prog_sim_);
            set_sim_uniforms(sp, 5000u + static_cast<std::uint32_t>(s));
            mgl::glDispatchCompute((kN + kLocalSize - 1) / kLocalSize, 1, 1);
        }
        const std::vector<Particle> gpu = read_particles(buf, 0, kAlloc);
        const std::uint32_t gpu_spawns = read_counter();

        // CPU model, run independently from the same initial state; optional mutations for the negative controls.
        auto run_model = [&](const SimParams& p, int steps, std::uint32_t& spawns, int& ambiguous) {
            std::vector<ModelParticle> s(kN);
            for (std::uint32_t i = 0; i < kN; ++i) s[i] = from_gpu(init[i]);
            spawns = 0;
            ambiguous = 0;
            for (int k = 0; k < steps; ++k)
                for (std::uint32_t i = 0; i < kN; ++i)
                {
                    const StepResult r = model_step(p, i, 5000u + static_cast<std::uint32_t>(k), s[i]);
                    if (r.branch == Branch::Spawn) ++spawns;
                    if (r.ambiguous) ++ambiguous;
                    s[i] = r.out;
                }
            return s;
        };
        auto compare = [&](const std::vector<ModelParticle>& mdl, double& worst, double& worst_bound) {
            int bad = 0;
            worst = 0.0;
            worst_bound = 0.0;
            for (std::uint32_t i = 0; i < kN; ++i)
            {
                const double q = particle_ratio(gpu[i], mdl[i]);
                worst = std::max(worst, q);
                worst_bound = std::max({worst_bound, mdl[i].px.e, mdl[i].py.e, mdl[i].pz.e});
                if (q > 1.0) ++bad;
            }
            return bad;
        };

        std::uint32_t spawns = 0;
        int ambiguous = 0;
        const std::vector<ModelParticle> ref = run_model(sp, kSteps, spawns, ambiguous);
        double worst = 0.0, worst_bound = 0.0;
        const int bad = compare(ref, worst, worst_bound);
        double max_abs = 0.0;
        for (std::uint32_t i = 0; i < kN; ++i)
            max_abs = std::max({max_abs, std::abs(static_cast<double>(gpu[i].px) - ref[i].px.v), std::abs(static_cast<double>(gpu[i].py) - ref[i].py.v), std::abs(static_cast<double>(gpu[i].pz) - ref[i].pz.v)});
        std::cout << std::format("[compute-particles] trajectory: {} particles x {} steps, max |gpu - model| position = {:.3g}, largest position bound = {:.3g}, "
                                 "max ratio = {:.4f}, {} respawns\n",
                                 kN, kSteps, max_abs, worst_bound, worst, spawns);
        check.expect(ambiguous == 0, "trajectory: no branch decision of the model lies within its rounding bound (bounce disabled)");
        check.expect(bad == 0, "trajectory: after 90 steps every particle within the propagated rounding bound, ages and lives exact",
                     std::format("{} of {} particles outside, max ratio {:.3g}", bad, kN, worst));
        check.expect_eq(gpu_spawns, spawns, "trajectory: atomic counter = respawns over 90 steps of the CPU model (hash RNG reproduced exactly)");
        check.expect(spawns > kN / 4, "trajectory: a substantial share of the particles was respawned by the GPU RNG during the run",
                     std::format("{} respawns", spawns));

        int guard_bad = 0;
        for (std::uint32_t i = kN; i < kAlloc; ++i)
            if (std::memcmp(&gpu[i], &guard, sizeof guard) != 0) ++guard_bad;
        check.expect(guard_bad == 0, "p.length() = 4000 for glBindBufferRange(0, 4000 * 32): the 96 invocations beyond it store nothing (guard intact)",
                     std::format("{} guard particles changed", guard_bad));

        // Negative controls: the same comparison must reject a model that is subtly wrong.
        SimParams wrong_g = sp;
        wrong_g.gdt = sp.gdt * 1.02f;
        double w1 = 0, b1 = 0;
        std::uint32_t s1 = 0;
        int a1 = 0;
        const int bad_g = compare(run_model(wrong_g, kSteps, s1, a1), w1, b1);
        check.expect(bad_g > static_cast<int>(kN / 2), "negative control: a model with gravity 2 % too strong is rejected for most particles",
                     std::format("{} of {} outside the bound, max ratio {:.3g}", bad_g, kN, w1));
        double w2 = 0, b2 = 0;
        std::uint32_t s2 = 0;
        int a2 = 0;
        const int bad_s = compare(run_model(sp, kSteps - 1, s2, a2), w2, b2);
        check.expect(bad_s == static_cast<int>(kN), "negative control: a model one step short (89 steps) is rejected for every particle",
                     std::format("{} of {} outside the bound", bad_s, kN));
        std::cout << std::format("[compute-particles] negative controls: gravity +2 % -> {} / {} rejected (max ratio {:.3g}); 89 steps -> {} / {} rejected\n", bad_g, kN, w1, bad_s, kN);
    }

    // ---------------------------------------------------------------------------------------------------------
    // 4. Render: vertex fetch from the SSBO after VERTEX_ATTRIB_ARRAY_BARRIER_BIT, pixels at CPU-projected positions.
    // ---------------------------------------------------------------------------------------------------------
    void ComputeParticles::check_render(const demo::Env& env, demo::Check& check)
    {
        const int W = env.width, H = env.height;
        SimParams sp{};
        sp.gdt = 0.0f;
        sp.swirl = 0.0f;
        sp.pull = 0.0f;
        sp.damp = 1.0f;
        sp.ground = -1000.0f;
        // One step moves p by v * dt: the uploaded positions are kStaleLift above the targets, v brings them down.
        const float v_down = -kStaleLift / kDt;
        std::vector<Particle> init(kRenderParticles);
        for (std::uint32_t i = 0; i < kRenderParticles; ++i)
        {
            const m::Vec3 t = render_target(i);
            init[i] = {t.x, t.y + kStaleLift, t.z, 0.0f, 0.0f, v_down, 0.0f, 1000.0f};
        }
        const mgl::BufferId buf = make_buffer(std::span<const Particle>(init), mgl::BufferUsage::DynamicCopy);
        const mgl::VertexArrayId vao = make_particle_vao(buf);
        const Camera cam = render_check_camera(W, H);

        auto draw_scene = [&]() {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, W, H);
            mgl::glClearColor(f8(kClear8[0]), f8(kClear8[1]), f8(kClear8[2]), 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::Blend);
            mgl::glBlendFunc(mgl::BlendFactor::One, mgl::BlendFactor::One);
            set_draw_uniforms(cam, H, kRenderHalf, 0.0f, 1.0f, true, kFlatColor8);
            mgl::glBindVertexArray(vao);
            mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, static_cast<GLsizei>(kRenderParticles));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDisable(mgl::Capability::Blend);
        };

        // CPU projection of the 16 targets, and the design preconditions (quads big enough, separated, inside).
        std::array<Projected, kGridN * kGridN> proj{};
        double min_half = 1e9, min_gap = 1e9;
        bool inside = true;
        for (std::uint32_t i = 0; i < kGridN * kGridN; ++i)
        {
            proj[i] = project(cam, render_target(i), kRenderHalf, W, H);
            min_half = std::min({min_half, proj[i].half_px_x, proj[i].half_px_y});
            inside = inside && proj[i].in_front && proj[i].x - proj[i].half_px_x > 2 && proj[i].x + proj[i].half_px_x < W - 2 &&
                     proj[i].y - proj[i].half_px_y > 2 && proj[i].y + proj[i].half_px_y < H - 2;
            if (i % kGridN != 0) min_gap = std::min(min_gap, proj[i].x - proj[i - 1].x - proj[i].half_px_x - proj[i - 1].half_px_x);
            if (i >= kGridN) min_gap = std::min(min_gap, proj[i].y - proj[i - kGridN].y - proj[i].half_px_y - proj[i - kGridN].half_px_y);
        }
        check.expect(inside && min_half >= 4.0 && min_gap >= 8.0, "render check layout: 16 quads inside the surface, at least 4 px half size, at least 8 px apart",
                     std::format("half {:.1f} px, gap {:.1f} px", min_half, min_gap));

        // Samples: inside each quad (centre and +-2 px), and outside at half + 3 px on the four sides.
        auto inside_samples = [&](const Projected& p) {
            const int cx = static_cast<int>(std::floor(p.x)), cy = static_cast<int>(std::floor(p.y));
            return std::array<std::array<int, 2>, 5>{{{cx, cy}, {cx - 2, cy}, {cx + 2, cy}, {cx, cy - 2}, {cx, cy + 2}}};
        };
        auto outside_samples = [&](const Projected& p) {
            const int cx = static_cast<int>(std::floor(p.x)), cy = static_cast<int>(std::floor(p.y));
            const int dx = static_cast<int>(std::ceil(p.half_px_x + 3.0)), dy = static_cast<int>(std::ceil(p.half_px_y + 3.0));
            return std::array<std::array<int, 2>, 4>{{{cx - dx, cy}, {cx + dx, cy}, {cx, cy - dy}, {cx, cy + dy}}};
        };
        auto probe_all = [&](std::string_view what, bool after_step) {
            const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, W, H);
            auto px = [&](int x, int y) {
                const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(W) + static_cast<std::size_t>(x)) * 4u;
                return Rgba8{img[o], img[o + 1], img[o + 2], img[o + 3]};
            };
            int bad = 0, n = 0;
            std::string first;
            auto test = [&](int x, int y, Rgba8 expected) {
                if (x < 0 || y < 0 || x >= W || y >= H) return;
                ++n;
                const Rgba8 got = px(x, y);
                if (max_channel_diff(got, expected) > 1 && bad++ == 0)
                    first = std::format("({}, {}) is {} expected {}", x, y, demo::format_rgba(got), demo::format_rgba(expected));
            };
            for (std::uint32_t i = 0; i < kGridN * kGridN; ++i)
            {
                // Additive blending onto the clear colour: one particle adds the flat colour, target 5 receives two.
                const Rgba8 expected = after_step ? add8(kClear8, kFlatColor8, i == 5 ? 2 : 1) : kClear8;
                for (const auto& s : inside_samples(proj[i])) test(s[0], s[1], expected);
                for (const auto& s : outside_samples(proj[i])) test(s[0], s[1], kClear8);
            }
            return check.expect(bad == 0 && n > 100, what, std::format("{} of {} samples wrong; first: {}", bad, n, first));
        };

        // Negative control: drawn from the uploaded data (no step yet) every target pixel is background.
        draw_scene();
        probe_all("render negative control: drawn before the compute step (uploaded positions are 9 units above, outside the view) every target is background", false);

        // One compute step writes the positions; VERTEX_ATTRIB_ARRAY_BARRIER_BIT (inside simulate) orders the vertex fetch.
        simulate(buf, kRenderParticles, sp, 1u);
        draw_scene();
        probe_all("render: after the step + VERTEX_ATTRIB_ARRAY barrier the instanced quads sit at the CPU-projected targets, colour = clear + flat (x2 on the "
                  "doubly covered target 5)",
                  true);

        // The positions the draw used are the ones the compute shader stored (readback).
        const std::vector<Particle> stored = read_particles(buf, 0, kRenderParticles);
        double max_err = 0.0;
        for (std::uint32_t i = 0; i < kRenderParticles; ++i)
        {
            const m::Vec3 t = render_target(i);
            max_err = std::max({max_err, std::abs(static_cast<double>(stored[i].px - t.x)), std::abs(static_cast<double>(stored[i].py - t.y)),
                                std::abs(static_cast<double>(stored[i].pz - t.z))});
        }
        check.expect(max_err < 1e-4, "render: the stored positions are the targets (one step of v * dt from 9 units above)", std::format("max error {:.3g}", max_err));
    }

    void ComputeParticles::validate(demo::Env& env, demo::Check& check)
    {
        check_layout(check);
        check_stepwise(check);
        check_trajectory(check);
        check_render(env, check);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_particles_);
        check.expect_no_gl_error("no GL error during validation");

        // The capture shows whatever is on screen when validate() returns: the fountain (its buffer was not modified).
        draw_display(env);
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-compute-particles",
        .title = "Compute particles: an SSBO fountain simulated on the GPU",
        .min_es = {3, 1},
        .summary = "16384 particles integrated by a compute shader in a std430 SSBO (gravity, vortex, drag, ground bounce, hash-based "
                   "respawn counted by an atomic counter) and drawn from the same buffer as instanced additive billboards after "
                   "glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT); validated against a CPU model with a derived rounding bound.",
        .controls = "SPACE pause | a/d orbit | w/s dolly | q/e raise/lower camera | r reset camera | ESC quit",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 2,
    };
    return demo::run<ComputeParticles>(argc, argv, info);
}
