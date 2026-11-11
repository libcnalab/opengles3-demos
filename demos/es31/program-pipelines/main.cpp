// es31-program-pipelines
//
// Separable programs and program pipeline objects (OpenGL ES 3.1; ES 3.2 sections 7.3, 7.4, 7.6.1 and
// 11.1.3.11). Instead of linking a vertex and a fragment shader into ONE program, each stage lives in
// its own *separable* program and a *program pipeline object* says which program supplies which stage.
// The demo shows:
//   - two ways to make a separable program: glCreateShaderProgramv (one stage, one call; a compile
//     failure returns a program whose LINK_STATUS is FALSE and whose log carries the compile log) and
//     glProgramParameteri(PROGRAM_SEPARABLE) + glLinkProgram (here a program holding BOTH stages);
//   - glUseProgramStages: replacing the vertex or the fragment stage of a bound pipeline at run time,
//     taking one stage from a multi-stage program, VERTEX|FRAGMENT / ALL_SHADER_BITS;
//   - glActiveShaderProgram + glUniform* (updates the pipeline's "active" program) versus
//     glProgramUniform* (names the program explicitly, no binding needed), all 33 variants;
//   - glUseProgram overriding a bound pipeline;
//   - interface matching between programs: matched BY LOCATION (different names are fine), and the
//     mismatches (location, type, precision) that make glValidateProgramPipeline fail and the draw
//     raise INVALID_OPERATION;
//   - glGetProgramPipelineiv / glGetProgramPipelineInfoLog and the defined errors.
//
// Picture (256x256, 8 columns of 32-pixel cells, rows numbered from the bottom; see README.md):
//   rows 0-2   scene cells 0-23: each is drawn by a pipeline in a different configuration; the CPU
//              model predicts the four quadrants of every cell
//   rows 3-7   the uniform zoo: 38 cells, green = the fragment shader sees exactly the value that a
//              glProgramUniform* call stored in a program that was never made current

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int kColumns = 8;
    constexpr int kSceneRows = 3;
    constexpr int kSceneCells = kColumns * kSceneRows;
    constexpr int kZooTests = 38;

    // Uniform locations, fixed by layout(location = N) in the shaders. Every program has its own
    // uniform name space (GLSL ES 3.10 section 9), so "location 0" names a different variable in each.
    constexpr mgl::UniformLocation kLocTint{0};   // vertex programs:   layout(location = 0) uniform vec3 u_tint
    constexpr mgl::UniformLocation kLocBg{1};     // fragment programs: layout(location = 1) uniform vec3 u_bg
    constexpr mgl::UniformLocation kLocColor{0};  // fs_solid.frag:     layout(location = 0) uniform vec3 u_color

    struct Rgb
    {
        float r, g, b;
    };

    constexpr std::array<float, 4> kClear = {0.10f, 0.10f, 0.12f, 1.0f};

    // Colors are chosen so that every channel is 0.25, 0.4, 0.75 or 1.0: unambiguous after quantization.
    constexpr Rgb kTintA{1.00f, 0.25f, 0.25f};  // vs_pair_normal   (red)
    constexpr Rgb kTintB{0.25f, 0.40f, 1.00f};  // vs_pair_mirror   (blue)
    constexpr Rgb kTintC{0.25f, 1.00f, 0.25f};  // program holding both stages (green)
    constexpr Rgb kTintR{1.00f, 0.75f, 0.00f};  // vs_pair_renamed  (orange)
    constexpr Rgb kTintD{0.75f, 0.25f, 1.00f};  // the program that is deleted while in use (violet)
    constexpr Rgb kBgX{0.75f, 0.75f, 0.25f};    // fs_vertical      (olive)
    constexpr Rgb kBgY{0.25f, 0.75f, 0.75f};    // fs_horizontal    (teal)
    constexpr Rgb kBgC{0.75f, 0.25f, 0.75f};    // progC fragment   (magenta)
    constexpr Rgb kBgS{0.40f, 0.40f, 0.75f};    // fs_horizontal made from two source strings

    // What one scene cell is expected to show, as a function of the programs installed when it was drawn.
    struct CellModel
    {
        enum class Kind { Empty, Split, Solid } kind = Kind::Empty;
        bool mirror = false;      // vertex stage mirrors u
        bool horizontal = false;  // fragment stage splits on v instead of u
        Rgb tint{};               // vertex stage's u_tint (or the solid color)
        Rgb bg{};                 // fragment stage's u_bg
    };

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(v * 255.0f)); }

    demo::Rgba8 expected_color(const CellModel& m, float u, float v)
    {
        auto rgb = [](Rgb c) { return demo::Rgba8{to8(c.r), to8(c.g), to8(c.b), 255}; };
        switch (m.kind)
        {
        case CellModel::Kind::Empty: return {to8(kClear[0]), to8(kClear[1]), to8(kClear[2]), 255};
        case CellModel::Kind::Solid: return rgb(m.tint);
        case CellModel::Kind::Split: break;
        }
        const float uu = m.mirror ? 1.0f - u : u;  // vertex stage: v_uv.x = 1 - corner.x
        const bool tinted = m.horizontal ? v < 0.5f : uu < 0.5f;
        return rgb(tinted ? m.tint : m.bg);
    }

    const char* const kCellLabels[kSceneCells] = {
        "A0  pipeline P1 = vertex A + fragment X (two glUseProgramStages calls)",
        "A1  P1: only the vertex stage replaced by B (pattern jumps to the right, blue)",
        "A2  P1: only the fragment stage replaced by Y (pattern becomes horizontal)",
        "A3  P1: vertex stage back to A (red, horizontal)",
        "A4  P1: both stages from ONE program (VERTEX|FRAGMENT bits of progC), green",
        "A5  glProgramUniform3f on progC while it is installed in P1: the picture follows at once (yellow)",
        "A6  both stages replaced by single-stage programs again (vertex B + fragment Y)",
        "A7  ALL_SHADER_BITS with progC: both stages in one call (same picture as A5)",
        "B0  glProgramUniform3f on programs that are neither current nor active (orange, dark blue)",
        "B1  glActiveShaderProgram(vertex A) + glUniform3f(0, ...): updates A's u_tint (violet)",
        "B2  the same glUniform3f(0, ...) call with vertex B active: updates B's u_tint (teal)",
        "B3  glActiveShaderProgram(fragment X) + glUniform3f(1, ...): updates X's u_bg",
        "B4  glUseProgram(progU) overrides the bound pipeline: solid orange fill",
        "B5  glUseProgram(0): the bound pipeline is in effect again",
        "B6  a second pipeline P2 (A + Y) next to P3: pipelines are independent objects",
        "B7  vertex program with DIFFERENT output names but the same locations (valid, matched by location)",
        "C0  fragment program built by glCreateShaderProgramv from TWO source strings",
        "C1  location mismatch (outputs at 3,4 vs inputs at 0,1): the draw is rejected, cell stays empty",
        "C2  type mismatch (vec3 vs vec2 at location 0): the draw is rejected, cell stays empty",
        "C3  precision mismatch (highp vs mediump): the draw is rejected, cell stays empty",
        "C4  vertex program deleted while attached to the pipeline: still used (violet)",
        "C5  progC for the vertex stage only + another fragment program: rejected (progC not active for all its stages)",
        "C6  another vertex program + progC for the fragment stage only: rejected, cell stays empty",
        "C7  scratch cell for the draws of valid pipelines in validate() (cleared again)",
    };

    class ProgramPipelines final : public demo::Demo
    {
    public:
        // ---------------------------------------------------------------------------------------------
        void init(demo::Env& env) override
        {
            src_vs_normal_ = env.load_text("shaders/vs_pair_normal.vert");
            src_fs_vertical_ = env.load_text("shaders/fs_vertical.frag");
            src_fs_horizontal_ = env.load_text("shaders/fs_horizontal.frag");

            // --- separable programs, one stage each: glCreateShaderProgramv (ES 3.2 section 7.3) ---------
            vs_normal_ = create_stage(mgl::ShaderType::Vertex, src_vs_normal_, "vs_pair_normal.vert");
            vs_mirror_ = create_stage(mgl::ShaderType::Vertex, env.load_text("shaders/vs_pair_mirror.vert"), "vs_pair_mirror.vert");
            vs_renamed_ = create_stage(mgl::ShaderType::Vertex, env.load_text("shaders/vs_pair_renamed.vert"), "vs_pair_renamed.vert");
            vs_bad_location_ = create_stage(mgl::ShaderType::Vertex, env.load_text("shaders/vs_pair_location.vert"), "vs_pair_location.vert");
            vs_bad_type_ = create_stage(mgl::ShaderType::Vertex, env.load_text("shaders/vs_pair_type.vert"), "vs_pair_type.vert");
            vs_none_ = create_stage(mgl::ShaderType::Vertex, env.load_text("shaders/vs_fullscreen.vert"), "vs_fullscreen.vert");
            fs_vertical_ = create_stage(mgl::ShaderType::Fragment, src_fs_vertical_, "fs_vertical.frag");
            fs_horizontal_ = create_stage(mgl::ShaderType::Fragment, src_fs_horizontal_, "fs_horizontal.frag");
            fs_mediump_ = create_stage(mgl::ShaderType::Fragment, env.load_text("shaders/fs_mediump.frag"), "fs_mediump.frag");
            fs_zoo_ = create_stage(mgl::ShaderType::Fragment, env.load_text("shaders/fs_uniform_zoo.frag"), "fs_uniform_zoo.frag");

            // The same fragment source passed as TWO strings: count = 2 concatenates them, so the
            // split may fall anywhere (here: right after the #version line).
            {
                const std::string& s = src_fs_horizontal_;
                const std::size_t cut = s.find('\n') + 1;
                const std::string head = s.substr(0, cut);
                const std::string tail = s.substr(cut);
                const GLchar* strings[2] = {head.c_str(), tail.c_str()};
                fs_split_ = mgl::glCreateShaderProgramv(mgl::ShaderType::Fragment, 2, strings);
                require_linked(fs_split_, "fs_horizontal.frag as two strings");
            }

            // A compile failure: the call still returns a program name (nothing "went wrong" at the API level).
            src_broken_ = env.load_text("shaders/fs_broken.frag");
            prog_broken_ = create_stage_unchecked(mgl::ShaderType::Fragment, src_broken_);

            // --- a separable program with TWO stages: glProgramParameteri + glLinkProgram (section 7.3) ---
            {
                const mgl::ShaderId vs = compile(mgl::ShaderType::Vertex, src_vs_normal_, "vs_pair_normal.vert (progC)");
                const mgl::ShaderId fs = compile(mgl::ShaderType::Fragment, src_fs_vertical_, "fs_vertical.frag (progC)");
                prog_combined_ = mgl::glCreateProgram();
                mgl::glAttachShader(prog_combined_, vs);
                mgl::glAttachShader(prog_combined_, fs);
                mgl::glProgramParameteri(prog_combined_, mgl::ProgramParameter::Separable, GL_TRUE);  // BEFORE the link
                mgl::glLinkProgram(prog_combined_);
                mgl::glDetachShader(prog_combined_, vs);
                mgl::glDetachShader(prog_combined_, fs);
                mgl::glDeleteShader(vs);
                mgl::glDeleteShader(fs);
                require_linked(prog_combined_, "progC (vertex + fragment, separable)");
            }

            // --- an ordinary program (NOT separable): vs_fullscreen + fs_solid --------------------------
            src_vs_none_ = env.load_text("shaders/vs_fullscreen.vert");
            src_fs_solid_ = env.load_text("shaders/fs_solid.frag");
            prog_plain_ = demo::build_program_from_source(src_vs_none_, src_fs_solid_, "progU");

            // --- program pipeline objects: names only, they become objects when first bound/used ---------
            mgl::glGenProgramPipelines(static_cast<GLsizei>(pipelines_.size()), pipelines_.data());

            // The initial state of every program's uniforms is set in render() each frame.
            cells_.fill(CellModel{});
        }

        // ---------------------------------------------------------------------------------------------
        void render(demo::Env& env, int /*frame*/) override
        {
            const int cs = cell_px(env);
            mgl::glUseProgram(mgl::ProgramId{0});  // a current program would hide every pipeline
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glEnable(mgl::Capability::ScissorTest);
            cells_.fill(CellModel{});

            // glProgramUniform*: the program is named explicitly, so it need not be current, bound,
            // or even part of a pipeline. Every uniform used below starts each frame from a known value.
            set_tint(vs_normal_, kTintA);
            set_tint(vs_mirror_, kTintB);
            set_tint(vs_renamed_, kTintR);
            set_tint(vs_bad_location_, kTintA);
            set_tint(vs_bad_type_, kTintA);
            set_bg(fs_vertical_, kBgX);
            set_bg(fs_horizontal_, kBgY);
            set_bg(fs_split_, kBgS);
            set_bg(fs_mediump_, kBgX);
            set_tint(prog_combined_, kTintC);
            set_bg(prog_combined_, kBgC);

            // ===== Row A: replacing stages of ONE bound pipeline ==========================================
            const mgl::ProgramPipelineId p1 = pipe(kP1);
            mgl::glBindProgramPipeline(p1);
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Fragment, fs_vertical_);
            draw_cell(env, 0, split(false, false, kTintA, kBgX));

            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Vertex, vs_mirror_);       // only the vertex stage changes
            draw_cell(env, 1, split(true, false, kTintB, kBgX));

            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Fragment, fs_horizontal_);  // only the fragment stage changes
            draw_cell(env, 2, split(true, true, kTintB, kBgY));

            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Vertex, vs_normal_);
            draw_cell(env, 3, split(false, true, kTintA, kBgY));

            // One program, two stages: the bits say which executables of the program are installed.
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Vertex | mgl::ShaderStageMask::Fragment, prog_combined_);
            draw_cell(env, 4, split(false, false, kTintC, kBgC));

            // glProgramUniform* reaches a program that is installed in a pipeline: the picture follows at once.
            const Rgb tint_a5{1.00f, 1.00f, 0.25f};
            mgl::glProgramUniform3f(prog_combined_, kLocTint, tint_a5.r, tint_a5.g, tint_a5.b);
            draw_cell(env, 5, split(false, false, tint_a5, kBgC));

            // Both stages go back to single-stage programs. (Replacing only ONE of progC's two stages would make
            // the pipeline invalid - see check_validation(): progC is then active for some, not all, of the
            // stages it was linked with, ES 3.2 section 11.1.3.11.)
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Vertex, vs_mirror_);
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Fragment, fs_horizontal_);
            draw_cell(env, 6, split(true, true, kTintB, kBgY));

            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::AllShaderBits, prog_combined_);
            draw_cell(env, 7, split(false, false, tint_a5, kBgC));

            // ===== Row B: uniforms and precedence =========================================================
            const mgl::ProgramPipelineId p3 = pipe(kP3);
            mgl::glBindProgramPipeline(p3);
            mgl::glUseProgramStages(p3, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glUseProgramStages(p3, mgl::ShaderStageMask::Fragment, fs_vertical_);

            // B0: glProgramUniform* reaches programs that are installed in the pipeline but are not "active".
            const Rgb tint_b0{1.00f, 0.75f, 0.25f}, bg_b0{0.25f, 0.25f, 0.75f};
            mgl::glProgramUniform3f(vs_normal_, kLocTint, tint_b0.r, tint_b0.g, tint_b0.b);
            mgl::glProgramUniform3f(fs_vertical_, kLocBg, bg_b0.r, bg_b0.g, bg_b0.b);
            draw_cell(env, 8, split(false, false, tint_b0, bg_b0));

            // B1: glUniform* writes to the pipeline's ACTIVE program, chosen by glActiveShaderProgram.
            const Rgb tint_b1{0.75f, 0.25f, 1.00f};
            mgl::glActiveShaderProgram(p3, vs_normal_);
            mgl::glUniform3f(kLocTint, tint_b1.r, tint_b1.g, tint_b1.b);
            draw_cell(env, 9, split(false, false, tint_b1, bg_b0));

            // B2: the very same call, with another program active, updates that program instead.
            const Rgb tint_b2{0.25f, 0.75f, 0.75f};
            mgl::glUseProgramStages(p3, mgl::ShaderStageMask::Vertex, vs_mirror_);
            mgl::glActiveShaderProgram(p3, vs_mirror_);
            mgl::glUniform3f(kLocTint, tint_b2.r, tint_b2.g, tint_b2.b);
            draw_cell(env, 10, split(true, false, tint_b2, bg_b0));

            // B3: location 1 is the fragment program's u_bg.
            const Rgb bg_b3{0.75f, 0.75f, 0.10f};
            mgl::glActiveShaderProgram(p3, fs_vertical_);
            mgl::glUniform3f(kLocBg, bg_b3.r, bg_b3.g, bg_b3.b);
            draw_cell(env, 11, split(true, false, tint_b2, bg_b3));

            // B4: glUseProgram wins over the bound pipeline, for rendering AND for uniform updates.
            const Rgb solid{1.00f, 0.60f, 0.10f};
            mgl::glUseProgram(prog_plain_);
            mgl::glUniform3f(kLocColor, solid.r, solid.g, solid.b);  // updates progU, not the active program of P3
            draw_cell(env, 12, CellModel{CellModel::Kind::Solid, false, false, solid, {}});

            // B5: without a current program the bound pipeline is in effect again.
            mgl::glUseProgram(mgl::ProgramId{0});
            draw_cell(env, 13, split(true, false, tint_b2, bg_b3));

            // B6: another pipeline object; P3 keeps its own stages (vs_mirror_ / fs_vertical_).
            const mgl::ProgramPipelineId p2 = pipe(kP2);
            mgl::glBindProgramPipeline(p2);
            mgl::glUseProgramStages(p2, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glUseProgramStages(p2, mgl::ShaderStageMask::Fragment, fs_horizontal_);
            draw_cell(env, 14, split(false, true, tint_b1, kBgY));

            // B7: different names, same locations: a valid interface.
            const mgl::ProgramPipelineId p4 = pipe(kP4);
            mgl::glBindProgramPipeline(p4);
            mgl::glUseProgramStages(p4, mgl::ShaderStageMask::Vertex, vs_renamed_);
            mgl::glUseProgramStages(p4, mgl::ShaderStageMask::Fragment, fs_vertical_);
            draw_cell(env, 15, split(false, false, kTintR, bg_b3));

            // ===== Row C ===================================================================================
            const mgl::ProgramPipelineId p5 = pipe(kP5);
            mgl::glBindProgramPipeline(p5);
            mgl::glUseProgramStages(p5, mgl::ShaderStageMask::Vertex, vs_mirror_);
            mgl::glUseProgramStages(p5, mgl::ShaderStageMask::Fragment, fs_split_);
            draw_cell(env, 16, split(true, true, tint_b2, kBgS));

            // C1-C3 (cells 17-19) use invalid pipelines; the draw would raise INVALID_OPERATION, so they are
            // exercised in validate() where the error is expected and consumed. Their cells stay empty.

            // C4: DeleteProgram on a program that is the current program of a pipeline stage only flags it
            // for deletion (ES 3.2 section 7.3); the pipeline keeps drawing with it.
            {
                const mgl::ProgramId temp = create_stage(mgl::ShaderType::Vertex, src_vs_normal_, "temporary vertex program");
                set_tint(temp, kTintD);
                const mgl::ProgramPipelineId p6 = pipe(kP6);
                mgl::glBindProgramPipeline(p6);
                mgl::glUseProgramStages(p6, mgl::ShaderStageMask::Vertex, temp);
                mgl::glUseProgramStages(p6, mgl::ShaderStageMask::Fragment, fs_vertical_);
                mgl::glDeleteProgram(temp);
                draw_cell(env, 20, split(false, false, kTintD, bg_b3));
                mgl::glUseProgramStages(p6, mgl::ShaderStageMask::Vertex, vs_normal_);  // the last reference goes away
            }

            // ===== Rows 3-7: the uniform zoo, set with glProgramUniform* only ==============================
            upload_zoo(cs, kSceneRows * cs);
            draw_zoo(env);

            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
        }

        // ---------------------------------------------------------------------------------------------
        void validate(demo::Env& env, demo::Check& check) override
        {
            check_scene_pixels(env, check, "after the scene was drawn");
            check_zoo(env, check);
            check_negative_control(env, check);
            check_create_shader_program(check);
            check_pipeline_objects(check);
            check_uniform_targets(check);
            check_deleted_program(check);
            check_validation(env, check);
            check_errors(check);
            check_scene_pixels_empty(env, check);

            // Leave the canonical picture in the framebuffer (the capture is taken after validate()).
            render(env, 0);
        }

        // ---------------------------------------------------------------------------------------------
        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
            mgl::glDeleteProgramPipelines(static_cast<GLsizei>(pipelines_.size()), pipelines_.data());
            for (const mgl::ProgramId p : {vs_normal_, vs_mirror_, vs_renamed_, vs_bad_location_, vs_bad_type_, vs_none_,
                                           fs_vertical_, fs_horizontal_, fs_split_, fs_mediump_, fs_zoo_, prog_combined_,
                                           prog_plain_, prog_broken_})
                mgl::glDeleteProgram(p);
        }

    private:
        // ================================ small helpers =================================================
        enum PipeIndex { kP1, kP2, kP3, kP4, kP5, kP6, kPZ, kPipeCount };

        mgl::ProgramPipelineId pipe(PipeIndex i) const { return pipelines_[static_cast<std::size_t>(i)]; }
        static int cell_px(const demo::Env& env) { return env.width / kColumns; }

        static CellModel split(bool mirror, bool horizontal, Rgb tint, Rgb bg)
        {
            return CellModel{CellModel::Kind::Split, mirror, horizontal, tint, bg};
        }

        static GLint program_param(mgl::ProgramId p, mgl::ProgramParameter pname)
        {
            GLint v = -1;
            mgl::glGetProgramiv(p, pname, &v);
            return v;
        }

        static GLint pipeline_param(mgl::ProgramPipelineId p, mgl::ProgramPipelineParameter pname)
        {
            GLint v = -1;
            mgl::glGetProgramPipelineiv(p, pname, &v);
            return v;
        }

        static std::string pipeline_log(mgl::ProgramPipelineId p)
        {
            const GLint len = pipeline_param(p, mgl::ProgramPipelineParameter::InfoLogLength);
            std::string log(static_cast<std::size_t>(std::max(len, 1)), '\0');
            GLsizei written = 0;
            mgl::glGetProgramPipelineInfoLog(p, static_cast<GLsizei>(log.size()), &written, log.data());
            log.resize(static_cast<std::size_t>(written));
            return log;
        }

        static mgl::ShaderId compile(mgl::ShaderType type, const std::string& source, const char* label)
        {
            const mgl::ShaderId shader = mgl::glCreateShader(type);
            mgl::glShaderSource(shader, source);
            mgl::glCompileShader(shader);
            GLint ok = GL_FALSE;
            mgl::glGetShaderiv(shader, mgl::ShaderParameter::CompileStatus, &ok);
            if (ok != GL_TRUE)
            {
                const std::string log = demo::shader_info_log(shader);
                mgl::glDeleteShader(shader);
                throw demo::Failure(std::format("{} failed to compile:\n{}", label, log));
            }
            return shader;
        }

        static mgl::ProgramId create_stage_unchecked(mgl::ShaderType type, const std::string& source)
        {
            const GLchar* strings[1] = {source.c_str()};
            return mgl::glCreateShaderProgramv(type, 1, strings);
        }

        static void require_linked(mgl::ProgramId p, const char* label)
        {
            if (p.value == 0u) throw demo::Failure(std::format("{}: glCreateShaderProgramv returned 0", label));
            if (program_param(p, mgl::ProgramParameter::LinkStatus) != GL_TRUE)
                throw demo::Failure(std::format("{} failed to link:\n{}", label, demo::program_info_log(p)));
        }

        static mgl::ProgramId create_stage(mgl::ShaderType type, const std::string& source, const char* label)
        {
            const mgl::ProgramId p = create_stage_unchecked(type, source);
            require_linked(p, label);
            return p;
        }

        static void set_tint(mgl::ProgramId p, Rgb c) { mgl::glProgramUniform3f(p, kLocTint, c.r, c.g, c.b); }
        static void set_bg(mgl::ProgramId p, Rgb c) { mgl::glProgramUniform3f(p, kLocBg, c.r, c.g, c.b); }

        mgl::UniformLocation zoo_loc(const char* name) const
        {
            const mgl::UniformLocation l = mgl::glGetUniformLocation(fs_zoo_, name);
            if (l.value < 0) throw demo::Failure(std::format("zoo uniform '{}' is not active", name));
            return l;
        }

        // Draws one full-cell triangle with whatever pipeline/program is current and records the expectation.
        void draw_cell(const demo::Env& env, int index, const CellModel& model)
        {
            const int cs = cell_px(env);
            const int x = (index % kColumns) * cs, y = (index / kColumns) * cs;
            mgl::glViewport(x, y, cs, cs);
            mgl::glScissor(x, y, cs, cs);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            cells_[static_cast<std::size_t>(index)] = model;
        }

        // ================================ the uniform zoo ===============================================
        // The zoo pipeline: a vertex program without outputs + the zoo fragment program, drawn over rows 3-7.
        void draw_zoo(const demo::Env& env)
        {
            const int cs = cell_px(env);
            const mgl::ProgramPipelineId pz = pipe(kPZ);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindProgramPipeline(pz);
            mgl::glUseProgramStages(pz, mgl::ShaderStageMask::Vertex, vs_none_);
            mgl::glUseProgramStages(pz, mgl::ShaderStageMask::Fragment, fs_zoo_);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, kSceneRows * cs, kColumns * cs, (kColumns - kSceneRows) * cs);
            mgl::glScissor(0, kSceneRows * cs, kColumns * cs, (kColumns - kSceneRows) * cs);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        // Every glProgramUniform* entry point of ES 3.1: float/int/uint x 1-4 components, scalar and array
        // ("v") forms, and the nine matrix shapes. The zoo program is never current.
        void upload_zoo(int cell, int origin_y)
        {
            const mgl::ProgramId z = fs_zoo_;
            mgl::glProgramUniform1i(z, zoo_loc("u_cell_px"), cell);
            mgl::glProgramUniform2i(z, zoo_loc("u_origin"), 0, origin_y);

            mgl::glProgramUniform1f(z, zoo_loc("u_f"), 0.5f);
            mgl::glProgramUniform2f(z, zoo_loc("u_v2"), 0.25f, -1.5f);
            mgl::glProgramUniform3f(z, zoo_loc("u_v3"), 1.0f, 2.0f, -3.0f);
            mgl::glProgramUniform4f(z, zoo_loc("u_v4"), 0.125f, 0.375f, -0.625f, 8.0f);
            const float farr[] = {1.5f, -2.5f};
            const float f8[] = {1, 2, 3, 4, 5, 6, 7, 8};
            mgl::glProgramUniform1fv(z, zoo_loc("u_farr"), 2, farr);
            mgl::glProgramUniform2fv(z, zoo_loc("u_v2arr"), 2, f8);
            mgl::glProgramUniform3fv(z, zoo_loc("u_v3arr"), 2, f8);
            mgl::glProgramUniform4fv(z, zoo_loc("u_v4arr"), 2, f8);

            mgl::glProgramUniform1i(z, zoo_loc("u_i"), -7);
            mgl::glProgramUniform2i(z, zoo_loc("u_iv2"), 1, -2);
            mgl::glProgramUniform3i(z, zoo_loc("u_iv3"), 3, -4, 5);
            mgl::glProgramUniform4i(z, zoo_loc("u_iv4"), -6, 7, -8, 9);
            const GLint iarr[] = {100, -100};
            const GLint i8[] = {1, 2, 3, 4, 5, 6, 7, 8};
            mgl::glProgramUniform1iv(z, zoo_loc("u_iarr"), 2, iarr);
            mgl::glProgramUniform2iv(z, zoo_loc("u_iv2arr"), 2, i8);
            mgl::glProgramUniform3iv(z, zoo_loc("u_iv3arr"), 2, i8);
            mgl::glProgramUniform4iv(z, zoo_loc("u_iv4arr"), 2, i8);

            mgl::glProgramUniform1ui(z, zoo_loc("u_u"), 0x80000001u);
            mgl::glProgramUniform2ui(z, zoo_loc("u_uv2"), 1u, 0xFFFFFFFFu);
            mgl::glProgramUniform3ui(z, zoo_loc("u_uv3"), 10u, 20u, 30u);
            mgl::glProgramUniform4ui(z, zoo_loc("u_uv4"), 1u, 2u, 3u, 4u);
            const GLuint uarr[] = {7u, 9u};
            const GLuint u8[] = {1, 2, 3, 4, 5, 6, 7, 8};
            mgl::glProgramUniform1uiv(z, zoo_loc("u_uarr"), 2, uarr);
            mgl::glProgramUniform2uiv(z, zoo_loc("u_uv2arr"), 2, u8);
            mgl::glProgramUniform3uiv(z, zoo_loc("u_uv3arr"), 2, u8);
            mgl::glProgramUniform4uiv(z, zoo_loc("u_uv4arr"), 2, u8);

            // Matrices are column-major; OpenGL ES requires transpose == GL_FALSE.
            const float m[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            mgl::glProgramUniformMatrix2fv(z, zoo_loc("u_m2"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix3fv(z, zoo_loc("u_m3"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix4fv(z, zoo_loc("u_m4"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix2x3fv(z, zoo_loc("u_m23"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix3x2fv(z, zoo_loc("u_m32"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix2x4fv(z, zoo_loc("u_m24"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix4x2fv(z, zoo_loc("u_m42"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix3x4fv(z, zoo_loc("u_m34"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix4x3fv(z, zoo_loc("u_m43"), 1, GL_FALSE, m);
            mgl::glProgramUniformMatrix2fv(z, zoo_loc("u_marr"), 2, GL_FALSE, m);

            mgl::glProgramUniform2i(z, zoo_loc("u_bv2"), 1, 0);  // bool vectors accept any Uniform*{i,ui,f}

            mgl::glProgramUniform1f(z, zoo_loc("u_s.a"), 0.75f);
            mgl::glProgramUniform2i(z, zoo_loc("u_s.b"), 4, -4);
            mgl::glProgramUniform3f(z, zoo_loc("u_sa[0].c"), 1.0f, 1.0f, 1.0f);
            mgl::glProgramUniform3f(z, zoo_loc("u_sa[1].c"), 9.0f, 8.0f, 7.0f);

            const float part[] = {0.0f, 1.0f, 2.0f, 3.0f};
            mgl::glProgramUniform1fv(z, zoo_loc("u_part"), 4, part);
            mgl::glProgramUniform1f(z, zoo_loc("u_part[2]"), 42.0f);  // replaces only element 2
        }

        // ================================ validate(): pixels =============================================
        bool cell_matches(const demo::Env& env, int index, const CellModel& model, std::string& detail) const
        {
            const int cs = cell_px(env);
            const int x = (index % kColumns) * cs, y = (index / kColumns) * cs;
            // The four quadrant centers: away from the cell edges and from the split lines at 0.5.
            constexpr float kProbe[4][2] = {{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}};
            bool ok = true;
            for (const auto& p : kProbe)
            {
                const int px = x + static_cast<int>(p[0] * static_cast<float>(cs));
                const int py = y + static_cast<int>(p[1] * static_cast<float>(cs));
                const demo::Rgba8 got = demo::read_pixel(px, py);
                const demo::Rgba8 want = expected_color(model, p[0], p[1]);
                bool same = true;
                for (std::size_t c = 0; c < 4; ++c)
                    if (std::abs(int{got[c]} - int{want[c]}) > 1) same = false;  // 8-bit quantization +-1
                if (!same)
                {
                    ok = false;
                    detail += std::format("pixel({},{}) = {} expected {}; ", px, py, demo::format_rgba(got), demo::format_rgba(want));
                }
            }
            return ok;
        }

        void check_scene_pixels(const demo::Env& env, demo::Check& check, const char* when)
        {
            int drawn = 0;
            for (int i = 0; i < kSceneCells; ++i)
            {
                const CellModel& m = cells_[static_cast<std::size_t>(i)];
                if (m.kind == CellModel::Kind::Empty) continue;
                ++drawn;
                std::string detail;
                check.expect(cell_matches(env, i, m, detail), std::format("cell {} ({}): four quadrants match the CPU model", i, kCellLabels[i]), detail);
            }
            check.expect_eq(drawn, 18, std::format("18 of the 24 scene cells were drawn ({})", when));
        }

        // Cells whose pipeline is invalid (17-19) and the unused cells keep the clear color.
        void check_scene_pixels_empty(const demo::Env& env, demo::Check& check)
        {
            std::string bad;
            for (int i = 0; i < kSceneCells; ++i)
            {
                if (cells_[static_cast<std::size_t>(i)].kind != CellModel::Kind::Empty) continue;
                std::string detail;
                if (!cell_matches(env, i, CellModel{}, detail)) bad += std::format("cell {}: {} ", i, detail);
            }
            check.expect(bad.empty(), "the 6 cells that were never drawn (invalid pipelines C1-C3 and unused cells) still show the clear color", bad);
        }

        void check_zoo(const demo::Env& env, demo::Check& check)
        {
            const int cs = cell_px(env);
            const int y0 = kSceneRows * cs;
            int green = 0;
            std::string red;
            for (int i = 0; i < kZooTests; ++i)
            {
                const demo::Rgba8 p = demo::read_pixel((i % kColumns) * cs + cs / 2, y0 + (i / kColumns) * cs + cs / 2);
                if (p == demo::Rgba8{0, 255, 0, 255}) ++green;
                else red += std::format("{} ", i);
            }
            check.expect(green == kZooTests,
                         std::format("uniform zoo: all {} cells are green (glProgramUniform* values seen by a program that was never current)", kZooTests),
                         red.empty() ? "" : "red cells: " + red);

            // glGetUniform*v reads the stored values back, again with the program never current.
            struct Expect { const char* name; char kind; std::vector<double> v; };
            const Expect expect[] = {
                {"u_f", 'f', {0.5}}, {"u_v2", 'f', {0.25, -1.5}}, {"u_v3", 'f', {1, 2, -3}}, {"u_v4", 'f', {0.125, 0.375, -0.625, 8}},
                {"u_farr[0]", 'f', {1.5}}, {"u_farr[1]", 'f', {-2.5}}, {"u_v2arr[1]", 'f', {3, 4}}, {"u_v3arr[1]", 'f', {4, 5, 6}},
                {"u_v4arr[1]", 'f', {5, 6, 7, 8}},
                {"u_i", 'i', {-7}}, {"u_iv2", 'i', {1, -2}}, {"u_iv3", 'i', {3, -4, 5}}, {"u_iv4", 'i', {-6, 7, -8, 9}},
                {"u_iarr[1]", 'i', {-100}}, {"u_iv2arr[1]", 'i', {3, 4}}, {"u_iv3arr[1]", 'i', {4, 5, 6}}, {"u_iv4arr[1]", 'i', {5, 6, 7, 8}},
                {"u_u", 'u', {2147483649.0}}, {"u_uv2", 'u', {1, 4294967295.0}}, {"u_uv3", 'u', {10, 20, 30}}, {"u_uv4", 'u', {1, 2, 3, 4}},
                {"u_uarr[1]", 'u', {9}}, {"u_uv2arr[1]", 'u', {3, 4}}, {"u_uv3arr[1]", 'u', {4, 5, 6}}, {"u_uv4arr[1]", 'u', {5, 6, 7, 8}},
                {"u_m2", 'f', {1, 2, 3, 4}}, {"u_m3", 'f', {1, 2, 3, 4, 5, 6, 7, 8, 9}},
                {"u_m4", 'f', {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
                {"u_m23", 'f', {1, 2, 3, 4, 5, 6}}, {"u_m32", 'f', {1, 2, 3, 4, 5, 6}}, {"u_m24", 'f', {1, 2, 3, 4, 5, 6, 7, 8}},
                {"u_m42", 'f', {1, 2, 3, 4, 5, 6, 7, 8}}, {"u_m34", 'f', {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                {"u_m43", 'f', {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, {"u_marr[1]", 'f', {5, 6, 7, 8}},
                {"u_bv2", 'i', {1, 0}}, {"u_s.a", 'f', {0.75}}, {"u_s.b", 'i', {4, -4}}, {"u_sa[1].c", 'f', {9, 8, 7}},
                {"u_part[2]", 'f', {42}}, {"u_part[3]", 'f', {3}}, {"u_cell_px", 'i', {static_cast<double>(cs)}},
                {"u_origin", 'i', {0, static_cast<double>(y0)}},
            };
            int ok = 0;
            std::string bad;
            for (const auto& e : expect)
            {
                const mgl::UniformLocation l = zoo_loc(e.name);
                bool match = true;
                if (e.kind == 'f')
                {
                    std::array<GLfloat, 16> got{};
                    mgl::glGetUniformfv(fs_zoo_, l, got.data());
                    for (std::size_t i = 0; i < e.v.size(); ++i) match = match && got[i] == static_cast<float>(e.v[i]);
                }
                else if (e.kind == 'i')
                {
                    std::array<GLint, 16> got{};
                    mgl::glGetUniformiv(fs_zoo_, l, got.data());
                    for (std::size_t i = 0; i < e.v.size(); ++i) match = match && got[i] == static_cast<GLint>(e.v[i]);
                }
                else
                {
                    std::array<GLuint, 16> got{};
                    mgl::glGetUniformuiv(fs_zoo_, l, got.data());
                    for (std::size_t i = 0; i < e.v.size(); ++i) match = match && got[i] == static_cast<GLuint>(e.v[i]);
                }
                if (match) ++ok;
                else bad += std::string(e.name) + " ";
            }
            check.expect(ok == static_cast<int>(std::size(expect)),
                         std::format("glGetUniform{{f,i,ui}}v reads back all {} values stored by glProgramUniform*", std::size(expect)), bad);
            check.expect_no_gl_error("no GL error from the zoo uploads and readbacks");
        }

        // Negative control: a comparison that cannot fail proves nothing. Changing ONE uniform of a
        // program in the pipeline must change the pixels, and the old expectation must then be violated.
        void check_negative_control(const demo::Env& env, demo::Check& check)
        {
            const CellModel before = cells_[0];  // cell 0: vertex A + fragment X, red | olive
            const mgl::ProgramPipelineId p1 = pipe(kP1);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindProgramPipeline(p1);
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glUseProgramStages(p1, mgl::ShaderStageMask::Fragment, fs_vertical_);
            const Rgb changed{0.25f, 0.25f, 1.00f};
            mgl::glProgramUniform3f(vs_normal_, kLocTint, changed.r, changed.g, changed.b);
            set_bg(fs_vertical_, kBgX);  // B3 left a different value in this program
            mgl::glEnable(mgl::Capability::ScissorTest);
            draw_cell(env, 0, split(false, false, changed, kBgX));
            std::string stale, fresh;
            check.expect(!cell_matches(env, 0, before, stale),
                         "negative control: after glProgramUniform3f changed u_tint, cell 0 no longer matches the OLD model (the comparison can fail)", "");
            check.expect(cell_matches(env, 0, cells_[0], fresh), "negative control: cell 0 matches the NEW model", fresh);
            mgl::glProgramUniform3f(vs_normal_, kLocTint, kTintA.r, kTintA.g, kTintA.b);
            draw_cell(env, 0, before);
            std::string restored;
            check.expect(cell_matches(env, 0, before, restored), "negative control: restoring the uniform restores the original cell", restored);

            // The same control for the zoo: store a wrong ivec3 with glProgramUniform3i, redraw, and exactly cell 10
            // (u_iv3, expected (3, -4, 5)) must turn red while its neighbour stays green.
            const int cs = cell_px(env);
            const int y0 = kSceneRows * cs;
            mgl::glProgramUniform3i(fs_zoo_, zoo_loc("u_iv3"), 3, -4, 6);
            draw_zoo(env);
            check.expect_pixel(10 % kColumns * cs + cs / 2, y0 + 10 / kColumns * cs + cs / 2, {255, 0, 0, 255}, 0,
                               "negative control: glProgramUniform3i(u_iv3 = 3,-4,6) turns zoo cell 10 red");
            check.expect_pixel(11 % kColumns * cs + cs / 2, y0 + 11 / kColumns * cs + cs / 2, {0, 255, 0, 255}, 0,
                               "negative control: the neighbouring zoo cell 11 stays green");
            mgl::glProgramUniform3i(fs_zoo_, zoo_loc("u_iv3"), 3, -4, 5);
            draw_zoo(env);
            check.expect_pixel(10 % kColumns * cs + cs / 2, y0 + 10 / kColumns * cs + cs / 2, {0, 255, 0, 255}, 0,
                               "negative control: restoring u_iv3 turns zoo cell 10 green again");
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
        }

        // ================================ validate(): glCreateShaderProgramv =============================
        void check_create_shader_program(demo::Check& check)
        {
            check.expect_eq(program_param(vs_normal_, mgl::ProgramParameter::LinkStatus), GL_TRUE, "glCreateShaderProgramv: LINK_STATUS = TRUE for valid source");
            check.expect_eq(program_param(vs_normal_, mgl::ProgramParameter::Separable), GL_TRUE, "glCreateShaderProgramv: PROGRAM_SEPARABLE = TRUE (set by the call itself)");
            check.expect_eq(program_param(vs_normal_, mgl::ProgramParameter::AttachedShaders), 0, "glCreateShaderProgramv: the shader was detached again (ATTACHED_SHADERS = 0)");
            check.expect_eq(program_param(prog_combined_, mgl::ProgramParameter::Separable), GL_TRUE, "glProgramParameteri(PROGRAM_SEPARABLE) before glLinkProgram: PROGRAM_SEPARABLE = TRUE");
            check.expect_eq(program_param(prog_combined_, mgl::ProgramParameter::AttachedShaders), 0, "progC: both shaders detached after the link");
            check.expect_eq(program_param(prog_plain_, mgl::ProgramParameter::Separable), GL_FALSE, "an ordinary program is not separable (PROGRAM_SEPARABLE = FALSE)");

            // The compile failure.
            check.expect(prog_broken_.value != 0u && mgl::glIsProgram(prog_broken_), "a source that fails to compile still returns a program name (not 0)");
            check.expect_eq(program_param(prog_broken_, mgl::ProgramParameter::LinkStatus), GL_FALSE, "...whose LINK_STATUS is FALSE");
            const std::string program_log = demo::program_info_log(prog_broken_);
            const mgl::ShaderId probe = mgl::glCreateShader(mgl::ShaderType::Fragment);
            mgl::glShaderSource(probe, src_broken_);
            mgl::glCompileShader(probe);
            const std::string shader_log = demo::shader_info_log(probe);
            GLint compiled = GL_TRUE;
            mgl::glGetShaderiv(probe, mgl::ShaderParameter::CompileStatus, &compiled);
            mgl::glDeleteShader(probe);
            check.expect_eq(compiled, GL_FALSE, "the broken source also fails an ordinary glCompileShader (the premise)");
            check.expect(program_log.find(shader_log) != std::string::npos,
                         "the program's info log contains the shader's compile log (\"append-shader-info-log-to-program-info-log\")",
                         std::format("program log {} chars, shader log {} chars", program_log.size(), shader_log.size()));
            std::printf("   note: compile log copied into the program log: \"%s\"\n", first_line(program_log).c_str());

            mgl::glUseProgram(prog_broken_);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUseProgram on the program with LINK_STATUS FALSE -> INVALID_OPERATION");

            const GLchar* none = nullptr;
            const mgl::ProgramId negative = mgl::glCreateShaderProgramv(mgl::ShaderType::Vertex, -1, &none);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glCreateShaderProgramv with count < 0 -> INVALID_VALUE");
            check.expect_eq(negative.value, 0u, "...and it returns 0");
        }

        static std::string first_line(const std::string& s)
        {
            return s.substr(0, std::min(s.find('\n'), std::size_t{100}));
        }

        // ================================ validate(): pipeline objects ===================================
        void check_pipeline_objects(demo::Check& check)
        {
            mgl::ProgramPipelineId fresh{};
            mgl::glGenProgramPipelines(1, &fresh);
            check.expect(!mgl::glIsProgramPipeline(fresh), "glGenProgramPipelines: a generated name is not a pipeline object until it is first bound/used (IsProgramPipeline = FALSE)");
            check.expect(!mgl::glIsProgramPipeline(mgl::ProgramPipelineId{0}), "IsProgramPipeline(0) = FALSE");
            mgl::glBindProgramPipeline(fresh);
            check.expect(mgl::glIsProgramPipeline(fresh), "glBindProgramPipeline creates the object (IsProgramPipeline = TRUE)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ProgramPipelineBinding), static_cast<int>(fresh.value), "PROGRAM_PIPELINE_BINDING reports the bound pipeline");

            // Initial state (ES 3.2 section 7.4.2): every program name 0, validation status FALSE, empty log.
            using PP = mgl::ProgramPipelineParameter;
            check.expect(pipeline_param(fresh, PP::ActiveProgram) == 0 && pipeline_param(fresh, PP::VertexShader) == 0 &&
                             pipeline_param(fresh, PP::FragmentShader) == 0 && pipeline_param(fresh, PP::ComputeShader) == 0,
                         "initial state: ACTIVE_PROGRAM, VERTEX_SHADER, FRAGMENT_SHADER, COMPUTE_SHADER are all 0");
            check.expect_eq(pipeline_param(fresh, PP::ValidateStatus), GL_FALSE, "initial state: VALIDATE_STATUS = FALSE");
            check.expect_eq(pipeline_param(fresh, PP::InfoLogLength), 0, "initial state: INFO_LOG_LENGTH = 0");

            // Stage bookkeeping through glGetProgramPipelineiv.
            mgl::glUseProgramStages(fresh, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glUseProgramStages(fresh, mgl::ShaderStageMask::Fragment, fs_vertical_);
            check.expect(pipeline_param(fresh, PP::VertexShader) == static_cast<GLint>(vs_normal_.value) &&
                             pipeline_param(fresh, PP::FragmentShader) == static_cast<GLint>(fs_vertical_.value),
                         "glUseProgramStages(VERTEX / FRAGMENT): VERTEX_SHADER and FRAGMENT_SHADER report those programs");
            check.expect_eq(pipeline_param(fresh, PP::ComputeShader), 0, "COMPUTE_SHADER stays 0");
            mgl::glUseProgramStages(fresh, mgl::ShaderStageMask::Vertex | mgl::ShaderStageMask::Fragment, prog_combined_);
            check.expect(pipeline_param(fresh, PP::VertexShader) == static_cast<GLint>(prog_combined_.value) &&
                             pipeline_param(fresh, PP::FragmentShader) == static_cast<GLint>(prog_combined_.value),
                         "VERTEX|FRAGMENT with a two-stage program: both queries report progC");
            mgl::glUseProgramStages(fresh, mgl::ShaderStageMask::Fragment, fs_horizontal_);
            check.expect(pipeline_param(fresh, PP::VertexShader) == static_cast<GLint>(prog_combined_.value) &&
                             pipeline_param(fresh, PP::FragmentShader) == static_cast<GLint>(fs_horizontal_.value),
                         "replacing only FRAGMENT leaves the VERTEX stage on progC");
            mgl::glUseProgramStages(fresh, mgl::ShaderStageMask::AllShaderBits, prog_combined_);
            check.expect(pipeline_param(fresh, PP::VertexShader) == static_cast<GLint>(prog_combined_.value) &&
                             pipeline_param(fresh, PP::FragmentShader) == static_cast<GLint>(prog_combined_.value),
                         "ALL_SHADER_BITS installs progC for every stage it has");
            mgl::glUseProgramStages(fresh, mgl::ShaderStageMask::Vertex, mgl::ProgramId{0});
            check.expect(pipeline_param(fresh, PP::VertexShader) == 0 && pipeline_param(fresh, PP::FragmentShader) == static_cast<GLint>(prog_combined_.value),
                         "glUseProgramStages(VERTEX, 0) clears the vertex stage only");

            mgl::glActiveShaderProgram(fresh, vs_mirror_);
            check.expect_eq(pipeline_param(fresh, PP::ActiveProgram), static_cast<int>(vs_mirror_.value), "glActiveShaderProgram: ACTIVE_PROGRAM reports it");
            check.expect_eq(pipeline_param(fresh, PP::VertexShader), 0, "...and the active program is independent of the stage programs");
            mgl::glActiveShaderProgram(fresh, mgl::ProgramId{0});
            check.expect_eq(pipeline_param(fresh, PP::ActiveProgram), 0, "glActiveShaderProgram(0): ACTIVE_PROGRAM = 0");

            // A pipeline created by UseProgramStages alone (never bound) is an object, too (section 7.4).
            mgl::ProgramPipelineId unbound{};
            mgl::glGenProgramPipelines(1, &unbound);
            mgl::glUseProgramStages(unbound, mgl::ShaderStageMask::Vertex, vs_normal_);
            check.expect(mgl::glIsProgramPipeline(unbound), "glUseProgramStages on a generated, never-bound name creates the pipeline object");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ProgramPipelineBinding), static_cast<int>(fresh.value), "...without binding it (PROGRAM_PIPELINE_BINDING unchanged)");

            // Deleting the bound pipeline reverts the binding to 0.
            mgl::glDeleteProgramPipelines(1, &fresh);
            check.expect(!mgl::glIsProgramPipeline(fresh), "glDeleteProgramPipelines: the name is no longer a pipeline object");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ProgramPipelineBinding), 0, "deleting the bound pipeline reverts PROGRAM_PIPELINE_BINDING to 0");
            mgl::glDeleteProgramPipelines(1, &unbound);
            mgl::ProgramPipelineId zero{};
            mgl::glDeleteProgramPipelines(1, &zero);
            check.expect_no_gl_error("deleting pipeline names, including 0 and already deleted ones, raises no error");
            mgl::glDeleteProgramPipelines(1, &unbound);
            check.expect_no_gl_error("deleting an already deleted pipeline name again is silently ignored");
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
        }

        // ================================ validate(): uniform targeting ==================================
        static std::array<float, 3> read_vec3(mgl::ProgramId p, mgl::UniformLocation l)
        {
            std::array<float, 3> v{};
            mgl::glGetUniformfv(p, l, v.data());
            return v;
        }

        void check_uniform_targets(demo::Check& check)
        {
            using V3 = std::array<float, 3>;
            using PP = mgl::ProgramPipelineParameter;
            const mgl::ProgramPipelineId p = pipe(kP3);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindProgramPipeline(p);
            mgl::glUseProgramStages(p, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glUseProgramStages(p, mgl::ShaderStageMask::Fragment, fs_vertical_);
            mgl::glProgramUniform3f(vs_normal_, kLocTint, 0.1f, 0.2f, 0.3f);
            mgl::glProgramUniform3f(vs_mirror_, kLocTint, 0.4f, 0.5f, 0.6f);
            mgl::glProgramUniform3f(fs_vertical_, kLocBg, 0.7f, 0.8f, 0.9f);

            // glUniform3f(0, ...) goes to the ACTIVE program; the identical call hits different programs.
            mgl::glActiveShaderProgram(p, vs_normal_);
            check.expect_eq(pipeline_param(p, PP::ActiveProgram), static_cast<int>(vs_normal_.value), "ACTIVE_PROGRAM = vertex A");
            mgl::glUniform3f(kLocTint, 1.0f, 2.0f, 3.0f);
            check.expect(read_vec3(vs_normal_, kLocTint) == V3{1, 2, 3} && read_vec3(vs_mirror_, kLocTint) == V3{0.4f, 0.5f, 0.6f} &&
                             read_vec3(fs_vertical_, kLocBg) == V3{0.7f, 0.8f, 0.9f},
                         "active = vertex A: glUniform3f(0, 1,2,3) changed A's u_tint only (vertex B and fragment X untouched)");
            mgl::glActiveShaderProgram(p, vs_mirror_);
            mgl::glUniform3f(kLocTint, 4.0f, 5.0f, 6.0f);
            check.expect(read_vec3(vs_mirror_, kLocTint) == V3{4, 5, 6} && read_vec3(vs_normal_, kLocTint) == V3{1, 2, 3},
                         "active = vertex B: the same call glUniform3f(0, 4,5,6) changed B's u_tint, A keeps (1,2,3)");
            mgl::glActiveShaderProgram(p, fs_vertical_);
            mgl::glUniform3f(kLocBg, 7.0f, 8.0f, 9.0f);
            check.expect(read_vec3(fs_vertical_, kLocBg) == V3{7, 8, 9} && read_vec3(vs_mirror_, kLocTint) == V3{4, 5, 6},
                         "active = fragment X: glUniform3f(1, 7,8,9) changed X's u_bg");
            mgl::glUniform3f(kLocTint, 0.0f, 0.0f, 0.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "active = fragment X: location 0 does not exist in X -> INVALID_OPERATION (the active program defines the locations)");
            check.expect(read_vec3(vs_normal_, kLocTint) == V3{1, 2, 3}, "...and it changed nothing");

            // glProgramUniform* ignores the pipeline's active program.
            mgl::glProgramUniform3f(vs_normal_, kLocTint, 11.0f, 12.0f, 13.0f);
            check.expect(read_vec3(vs_normal_, kLocTint) == V3{11, 12, 13} && read_vec3(fs_vertical_, kLocBg) == V3{7, 8, 9},
                         "glProgramUniform3f(vertex A, ...) while fragment X is active changed A only");
            mgl::glProgramUniform3f(vs_normal_, kLocTint, kTintA.r, kTintA.g, kTintA.b);

            // glUseProgram overrides the pipeline for uniform updates (ES 3.2 section 7.4: "the bound
            // program pipeline object has no effect on rendering or uniform updates").
            mgl::glActiveShaderProgram(p, vs_normal_);
            mgl::glProgramUniform3f(prog_plain_, kLocColor, 0.0f, 0.0f, 0.0f);
            mgl::glUseProgram(prog_plain_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::CurrentProgram), static_cast<int>(prog_plain_.value), "CURRENT_PROGRAM = progU");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ProgramPipelineBinding), static_cast<int>(p.value), "the pipeline stays bound (PROGRAM_PIPELINE_BINDING unchanged)");
            mgl::glUniform3f(kLocColor, 0.5f, 0.25f, 0.125f);
            check.expect(read_vec3(prog_plain_, kLocColor) == V3{0.5f, 0.25f, 0.125f} && read_vec3(vs_normal_, kLocTint) == V3{kTintA.r, kTintA.g, kTintA.b},
                         "glUseProgram(progU): glUniform3f(0, ...) updates progU, not the pipeline's active program (vertex A)");
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glUniform3f(kLocTint, 0.5f, 0.5f, 0.5f);
            check.expect(read_vec3(vs_normal_, kLocTint) == V3{0.5f, 0.5f, 0.5f}, "glUseProgram(0): glUniform3f(0, ...) reaches vertex A (the active program) again");
            mgl::glProgramUniform3f(vs_normal_, kLocTint, kTintA.r, kTintA.g, kTintA.b);

            // No active program, no current program -> nothing to update.
            mgl::glActiveShaderProgram(p, mgl::ProgramId{0});
            mgl::glUniform3f(kLocTint, 1.0f, 1.0f, 1.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniform* with a bound pipeline that has no active program -> INVALID_OPERATION");
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
            mgl::glUniform3f(kLocTint, 1.0f, 1.0f, 1.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniform* with no current program and no bound pipeline -> INVALID_OPERATION");
        }

        // ================================ validate(): deletion while in use ==============================
        void check_deleted_program(demo::Check& check)
        {
            const mgl::ProgramId temp = create_stage(mgl::ShaderType::Vertex, src_vs_normal_, "temporary vertex program");
            const mgl::ProgramPipelineId p = pipe(kP6);
            mgl::glUseProgramStages(p, mgl::ShaderStageMask::Vertex, temp);
            mgl::glDeleteProgram(temp);
            check.expect(mgl::glIsProgram(temp) && program_param(temp, mgl::ProgramParameter::DeleteStatus) == GL_TRUE,
                         "glDeleteProgram on a program that is the vertex-stage program of a pipeline only flags it (IsProgram TRUE, DELETE_STATUS TRUE)");
            check.expect_eq(pipeline_param(p, mgl::ProgramPipelineParameter::VertexShader), static_cast<int>(temp.value), "...and the pipeline still reports it as its vertex program");
            mgl::glUseProgramStages(p, mgl::ShaderStageMask::Vertex, vs_normal_);
            check.expect(!mgl::glIsProgram(temp), "after the pipeline lets go of it the program is deleted (IsProgram FALSE)");
            check.expect_no_gl_error("no GL error in the deletion scenario");
        }

        // ================================ validate(): glValidateProgramPipeline ==========================
        // Builds a pipeline from two programs, validates it, optionally draws with it into cell `cell`.
        void check_one_pipeline(const demo::Env& env, demo::Check& check, const char* name, mgl::ProgramPipelineId p,
                                mgl::ProgramId vs, mgl::ProgramId fs, bool expect_valid, int cell)
        {
            using PP = mgl::ProgramPipelineParameter;
            mgl::glBindProgramPipeline(p);
            mgl::glUseProgramStages(p, mgl::ShaderStageMask::Vertex, vs);
            mgl::glUseProgramStages(p, mgl::ShaderStageMask::Fragment, fs);
            mgl::glValidateProgramPipeline(p);
            const GLint status = pipeline_param(p, PP::ValidateStatus);
            check.expect_eq(status, expect_valid ? GL_TRUE : GL_FALSE,
                            std::format("{}: glValidateProgramPipeline -> VALIDATE_STATUS = {}", name, expect_valid ? "TRUE" : "FALSE"));
            const GLint log_len = pipeline_param(p, PP::InfoLogLength);
            const std::string log = pipeline_log(p);
            check.expect(log.empty() ? log_len == 0 : log_len == static_cast<GLint>(log.size()) + 1,
                         std::format("{}: INFO_LOG_LENGTH ({}) = log length + terminator (or 0 for an empty log)", name, log_len));
            std::printf("   note: %s info log: \"%s\"\n", name, log.empty() ? "(empty)" : first_line(log).c_str());

            // "If validation failed, such errors (INVALID_OPERATION from commands that transfer vertices) are
            // generated under the current state" (ES 3.2 section 11.1.3.11); if it succeeded, they are not.
            const int cs = cell_px(env);
            const int x = (cell % kColumns) * cs, y = (cell / kColumns) * cs;
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glViewport(x, y, cs, cs);
            mgl::glScissor(x, y, cs, cs);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            if (expect_valid)
            {
                check.expect_no_gl_error(std::format("{}: a valid pipeline draws without error", name));
                mgl::glClear(mgl::ClearBufferBit::Color);  // give the scratch cell back (scissored to it)
            }
            else
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, std::format("{}: drawing with the invalid pipeline -> INVALID_OPERATION", name));
        }

        void check_validation(const demo::Env& env, demo::Check& check)
        {
            using PP = mgl::ProgramPipelineParameter;
            // Valid pipelines: an empty interface (vertex without outputs + fragment without inputs), and
            // outputs matching inputs by location, with identical names and with different names.
            check_one_pipeline(env, check, "zoo pipeline (no user-defined interface)", pipe(kPZ), vs_none_, fs_zoo_, true, 23);
            check_one_pipeline(env, check, "vertex A + fragment X (same names, same locations)", pipe(kP2), vs_normal_, fs_vertical_, true, 23);
            check_one_pipeline(env, check, "renamed outputs + fragment X (same locations, different names)", pipe(kP4), vs_renamed_, fs_vertical_, true, 23);

            // Mismatches. Each pair differs from the valid pair "vertex A + fragment X" in exactly one respect.
            const mgl::ProgramPipelineId bad_loc = pipe(kP2), bad_type = pipe(kP4), bad_prec = pipe(kP5);
            check_one_pipeline(env, check, "LOCATION mismatch (outputs at 3,4; inputs at 0,1)", bad_loc, vs_bad_location_, fs_vertical_, false, 17);
            check_one_pipeline(env, check, "TYPE mismatch (vec3 output, vec2 input, same location)", bad_type, vs_bad_type_, fs_vertical_, false, 18);
            check_one_pipeline(env, check, "PRECISION mismatch (highp outputs, mediump inputs)", bad_prec, vs_normal_, fs_mediump_, false, 19);

            // Mixing ONE stage of a two-stage program with another program: progC was linked with a vertex AND a
            // fragment shader, so it must be active for both or for neither (ES 3.2 section 11.1.3.11, bullet 2).
            check_one_pipeline(env, check, "progC for the vertex stage only (fragment stage from another program)", pipe(kP3), prog_combined_, fs_horizontal_, false, 21);
            check_one_pipeline(env, check, "progC for the fragment stage only (vertex stage from another program)", pipe(kP6), vs_mirror_, prog_combined_, false, 22);

            // Repair: replace the offending stage, validate again, the very same pipeline object becomes valid.
            mgl::glUseProgramStages(bad_loc, mgl::ShaderStageMask::Vertex, vs_normal_);
            mgl::glValidateProgramPipeline(bad_loc);
            check.expect_eq(pipeline_param(bad_loc, PP::ValidateStatus), GL_TRUE, "mutation: replacing the offending vertex program with a matching one makes the same pipeline valid again");
            mgl::glUseProgramStages(bad_prec, mgl::ShaderStageMask::Fragment, fs_vertical_);
            mgl::glValidateProgramPipeline(bad_prec);
            check.expect_eq(pipeline_param(bad_prec, PP::ValidateStatus), GL_TRUE, "mutation: replacing the mediump fragment program makes it valid again");
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        // ================================ validate(): errors =============================================
        void check_errors(demo::Check& check)
        {
            using SM = mgl::ShaderStageMask;
            using PP = mgl::ProgramPipelineParameter;
            using EC = mgl::ErrorCode;
            mgl::ProgramPipelineId p{};
            mgl::glGenProgramPipelines(1, &p);
            mgl::glBindProgramPipeline(p);
            mgl::glUseProgramStages(p, SM::Vertex, vs_normal_);

            // Programs that cannot supply a stage.
            mgl::glUseProgramStages(p, SM::Vertex, prog_plain_);
            check.expect_gl_error(EC::InvalidOperation, "glUseProgramStages with a program linked WITHOUT PROGRAM_SEPARABLE -> INVALID_OPERATION");
            check.expect_eq(pipeline_param(p, PP::VertexShader), static_cast<int>(vs_normal_.value), "...the stage is not modified");
            mgl::glUseProgramStages(p, SM::Fragment, prog_broken_);
            check.expect_gl_error(EC::InvalidOperation, "glUseProgramStages with a program that failed to link -> INVALID_OPERATION");
            check.expect_eq(pipeline_param(p, PP::FragmentShader), 0, "...the stage is not modified");
            mgl::glActiveShaderProgram(p, prog_broken_);
            check.expect_gl_error(EC::InvalidOperation, "glActiveShaderProgram with a program that failed to link -> INVALID_OPERATION");
            check.expect_eq(pipeline_param(p, PP::ActiveProgram), 0, "...the active program is not modified");
            mgl::glProgramUniform1f(prog_broken_, mgl::UniformLocation{0}, 1.0f);
            check.expect_gl_error(EC::InvalidOperation, "glProgramUniform1f on a program that has not been linked successfully -> INVALID_OPERATION");

            // "PROGRAM_SEPARABLE ... indicates whether program can be bound for individual pipeline stages using
            // UseProgramStages after it is next linked" (section 7.3): setting the flag alone changes nothing.
            {
                const mgl::ShaderId vs = compile(mgl::ShaderType::Vertex, src_vs_none_, "vs_fullscreen.vert (late)");
                const mgl::ShaderId fs = compile(mgl::ShaderType::Fragment, src_fs_solid_, "fs_solid.frag (late)");
                const mgl::ProgramId late = mgl::glCreateProgram();
                mgl::glAttachShader(late, vs);
                mgl::glAttachShader(late, fs);
                mgl::glLinkProgram(late);  // linked WITHOUT the flag
                mgl::glProgramParameteri(late, mgl::ProgramParameter::Separable, GL_TRUE);
                mgl::glUseProgramStages(p, SM::Vertex, late);
                const bool raised = demo::drain_gl_errors() == EC::InvalidOperation;
                check.expect_or_known_deviation(raised,
                    "PROGRAM_SEPARABLE set AFTER the link has no effect until the next link: glUseProgramStages -> INVALID_OPERATION (\"was linked without the PROGRAM_SEPARABLE parameter set\")",
                    raised ? "" : "no error: the flag set after the link was honoured immediately",
                    "Mesa", "Mesa tests the flag set by glProgramParameteri at UseProgramStages time instead of the value that was in effect at the last link");
                mgl::glLinkProgram(late);  // now linked WITH the flag
                mgl::glUseProgramStages(p, SM::Vertex, late);
                check.expect_no_gl_error("after glLinkProgram the program is separable and glUseProgramStages accepts it");
                mgl::glUseProgramStages(p, SM::Vertex, vs_normal_);
                mgl::glDetachShader(late, vs);
                mgl::glDetachShader(late, fs);
                mgl::glDeleteShader(vs);
                mgl::glDeleteShader(fs);
                mgl::glDeleteProgram(late);
            }

            // Names that are not what they should be.
            const mgl::ProgramId unknown{0x7ff00001u};
            mgl::glUseProgramStages(p, SM::Vertex, unknown);
            check.expect_gl_error(EC::InvalidValue, "glUseProgramStages with a name that is neither a program nor a shader -> INVALID_VALUE");
            mgl::glActiveShaderProgram(p, unknown);
            check.expect_gl_error(EC::InvalidValue, "glActiveShaderProgram with an unknown program name -> INVALID_VALUE");
            mgl::glProgramUniform1f(unknown, mgl::UniformLocation{0}, 1.0f);
            check.expect_gl_error(EC::InvalidValue, "glProgramUniform1f with an unknown program name -> INVALID_VALUE");
            const mgl::ShaderId shader = mgl::glCreateShader(mgl::ShaderType::Vertex);
            const mgl::ProgramId shader_as_program{shader.value};
            mgl::glUseProgramStages(p, SM::Vertex, shader_as_program);
            check.expect_gl_error(EC::InvalidOperation, "glUseProgramStages with the name of a shader object -> INVALID_OPERATION");
            mgl::glActiveShaderProgram(p, shader_as_program);
            check.expect_gl_error(EC::InvalidOperation, "glActiveShaderProgram with the name of a shader object -> INVALID_OPERATION");
            mgl::glProgramUniform1f(shader_as_program, mgl::UniformLocation{0}, 1.0f);
            check.expect_gl_error(EC::InvalidOperation, "glProgramUniform1f with the name of a shader object -> INVALID_OPERATION");
            mgl::glDeleteShader(shader);

            // Pipeline names that were never generated, or were deleted.
            const mgl::ProgramPipelineId never{0x7ff00002u};
            mgl::glBindProgramPipeline(never);
            check.expect_gl_error(EC::InvalidOperation, "glBindProgramPipeline with a name that was never generated -> INVALID_OPERATION");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ProgramPipelineBinding), static_cast<int>(p.value), "...the binding is unchanged");
            mgl::glUseProgramStages(never, SM::Vertex, vs_normal_);
            check.expect_gl_error(EC::InvalidOperation, "glUseProgramStages on a never generated pipeline name -> INVALID_OPERATION");
            mgl::glActiveShaderProgram(never, vs_normal_);
            check.expect_gl_error(EC::InvalidOperation, "glActiveShaderProgram on a never generated pipeline name -> INVALID_OPERATION");
            mgl::glValidateProgramPipeline(never);
            check.expect_gl_error(EC::InvalidOperation, "glValidateProgramPipeline on a never generated pipeline name -> INVALID_OPERATION");
            GLint ignored = 0;
            mgl::glGetProgramPipelineiv(never, PP::ValidateStatus, &ignored);
            check.expect_gl_error(EC::InvalidOperation, "glGetProgramPipelineiv on a never generated pipeline name -> INVALID_OPERATION");
            mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
            mgl::glDeleteProgramPipelines(1, &p);
            mgl::glBindProgramPipeline(p);
            check.expect_gl_error(EC::InvalidOperation, "glBindProgramPipeline with a deleted pipeline name -> INVALID_OPERATION");
        }

        // ---------------------------------------------------------------------------------------------
        std::string src_vs_normal_, src_fs_vertical_, src_fs_horizontal_, src_broken_, src_vs_none_, src_fs_solid_;

        mgl::ProgramId vs_normal_{}, vs_mirror_{}, vs_renamed_{}, vs_bad_location_{}, vs_bad_type_{}, vs_none_{};
        mgl::ProgramId fs_vertical_{}, fs_horizontal_{}, fs_split_{}, fs_mediump_{}, fs_zoo_{};
        mgl::ProgramId prog_combined_{}, prog_plain_{}, prog_broken_{};
        std::array<mgl::ProgramPipelineId, kPipeCount> pipelines_{};
        std::array<CellModel, kSceneCells> cells_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-program-pipelines",
        .title = "Separable programs and program pipelines",
        .min_es = {3, 1},
        .summary = "Separable programs (glCreateShaderProgramv, PROGRAM_SEPARABLE), program pipeline objects, glUseProgramStages, "
                   "glActiveShaderProgram vs glProgramUniform*, location-based interface matching and validation.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ProgramPipelines>(argc, argv, info);
}
