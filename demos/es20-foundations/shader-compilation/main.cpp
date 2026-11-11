// es20-shader-compilation
//
// The life cycle of shader and program objects, including the parts that most tutorials skip:
// multi-string sources with explicit lengths, compile and link failures (which are NOT GL errors),
// deferred deletion of attached shaders and of the current program, program validation against
// the current state, guaranteed precision ranges, and the shader-binary entry point.
//
// The image shows two programs: the left half is drawn by a program whose vertex shader was
// assembled from three source strings, the right half by a conventionally loaded one.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <array>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr std::array<float, 4> kLeftColor = {0.15f, 0.75f, 0.30f, 1.0f};   // program A
    constexpr std::array<float, 4> kRightColor = {0.20f, 0.35f, 0.85f, 1.0f};  // program B
    constexpr std::array<float, 4> kProbeColor = {0.90f, 0.60f, 0.10f, 1.0f};  // "doomed" program

    GLint shader_param(mgl::ShaderId s, mgl::ShaderParameter p)
    {
        GLint v = -1;
        mgl::glGetShaderiv(s, p, &v);
        return v;
    }

    GLint program_param(mgl::ProgramId p, mgl::ProgramParameter q)
    {
        GLint v = -1;
        mgl::glGetProgramiv(p, q, &v);
        return v;
    }

    /// Compiles without throwing; returns the shader (also on failure) so its state can be inspected.
    mgl::ShaderId compile_keep(mgl::ShaderType type, std::string_view src)
    {
        const mgl::ShaderId s = mgl::glCreateShader(type);
        mgl::glShaderSource(s, src);
        mgl::glCompileShader(s);
        return s;
    }

    mgl::ProgramId link_keep(std::initializer_list<mgl::ShaderId> shaders)
    {
        const mgl::ProgramId p = mgl::glCreateProgram();
        for (const auto s : shaders) mgl::glAttachShader(p, s);
        mgl::glLinkProgram(p);
        return p;
    }

    demo::Rgba8 to_rgba8(const std::array<float, 4>& c)
    {
        auto q = [](float v) { return static_cast<std::uint8_t>(v * 255.0f + 0.5f); };
        return {q(c[0]), q(c[1]), q(c[2]), q(c[3])};
    }

    class ShaderCompilation final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            vertex_source_ = env.load_text("shaders/fullscreen.vert");
            const std::string fragment_source = env.load_text("shaders/solid.frag");

            // --- Program A: vertex shader from three strings --------------------------------
            // glShaderSource concatenates `count` strings. A negative length means "null-terminated";
            // a non-negative length takes exactly that many characters - anything after it is ignored.
            const std::size_t first_newline = vertex_source_.find('\n') + 1;
            const std::size_t closing = vertex_source_.rfind('}');
            part0_ = vertex_source_.substr(0, first_newline);                     // "#version 300 es\n"
            part1_ = vertex_source_.substr(first_newline, closing - first_newline);
            part2_ = vertex_source_.substr(closing) + "<-- garbage beyond the explicit length";
            const GLchar* strings[3] = {part0_.c_str(), part1_.c_str(), part2_.c_str()};
            const GLint lengths[3] = {-1, -1, static_cast<GLint>(vertex_source_.size() - closing)};

            vs_a_ = mgl::glCreateShader(mgl::ShaderType::Vertex);
            mgl::glShaderSource(vs_a_, 3, strings, lengths);
            mgl::glCompileShader(vs_a_);
            if (shader_param(vs_a_, mgl::ShaderParameter::CompileStatus) != GL_TRUE)
                throw demo::Failure("multi-string vertex shader failed:\n" + demo::shader_info_log(vs_a_));

            fs_ = compile_keep(mgl::ShaderType::Fragment, fragment_source);
            if (shader_param(fs_, mgl::ShaderParameter::CompileStatus) != GL_TRUE)
                throw demo::Failure("solid.frag failed:\n" + demo::shader_info_log(fs_));

            program_a_ = link_keep({vs_a_, fs_});
            if (program_param(program_a_, mgl::ProgramParameter::LinkStatus) != GL_TRUE)
                throw demo::Failure("program A failed to link:\n" + demo::program_info_log(program_a_));
            // The shaders stay attached until validate() has inspected glGetAttachedShaders.

            // --- Program B: the usual single-string path (helper = same calls, see gl_util.cpp) --
            program_b_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/solid.frag");

            color_a_ = mgl::glGetUniformLocation(program_a_, "u_color");
            color_b_ = mgl::glGetUniformLocation(program_b_, "u_color");
            if (color_a_.value < 0 || color_b_.value < 0)
                throw demo::Failure("u_color not active");
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glEnable(mgl::Capability::ScissorTest);

            // Uniform values belong to the program object: glUniform* writes to the *current* program.
            mgl::glUseProgram(program_a_);
            mgl::glUniform4f(color_a_, kLeftColor[0], kLeftColor[1], kLeftColor[2], kLeftColor[3]);
            mgl::glScissor(0, 0, env.width / 2, env.height);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);  // default VAO 0, no attributes

            mgl::glUseProgram(program_b_);
            mgl::glUniform4f(color_b_, kRightColor[0], kRightColor[1], kRightColor[2], kRightColor[3]);
            mgl::glScissor(env.width / 2, 0, env.width - env.width / 2, env.height);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_multi_string_source(check);
            check_attached_shaders(check);
            check_compile_failures(check);
            check_link_failures(check);
            check_deferred_shader_deletion(check);
            check_deferred_program_deletion(env, check);
            check_validation(env, check);
            check_precision_formats(check);
            check_compiler_and_binaries(check);

            // The experiments above drew into the framebuffer; restore the canonical image, then
            // verify it (and therefore that both programs execute correctly).
            render(env, 0);
            check.expect_pixel(env.width / 4, env.height / 2, to_rgba8(kLeftColor), 1,
                               "left half drawn by the multi-string program A");
            check.expect_pixel(3 * env.width / 4, env.height / 2, to_rgba8(kRightColor), 1,
                               "right half drawn by program B");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program_a_);  // deleting the program detaches (and so frees) vs_a_/fs_ ...
            mgl::glDeleteProgram(program_b_);
            mgl::glDeleteShader(vs_a_);        // ... once these deletion flags are also set
            mgl::glDeleteShader(fs_);
        }

    private:
        void check_multi_string_source(demo::Check& check)
        {
            // SHADER_SOURCE_LENGTH counts the concatenated source *including* a null terminator
            // (ES 3.0 §6.1.12); glGetShaderSource returns the concatenation, so the garbage after the
            // third string's explicit length must not appear.
            const GLint len = shader_param(vs_a_, mgl::ShaderParameter::SourceLength);
            check.expect_eq(len, static_cast<GLint>(vertex_source_.size() + 1),
                            "SHADER_SOURCE_LENGTH = concatenated length + 1 (explicit length honoured)");
            std::string back(static_cast<std::size_t>(len), '\0');
            GLsizei written = 0;
            mgl::glGetShaderSource(vs_a_, len, &written, back.data());
            back.resize(static_cast<std::size_t>(written));
            check.expect(back == vertex_source_, "glGetShaderSource returns exactly the concatenated strings");
            check.expect_eq(shader_param(vs_a_, mgl::ShaderParameter::ShaderType), static_cast<GLint>(GL_VERTEX_SHADER),
                            "SHADER_TYPE = GL_VERTEX_SHADER");
        }

        void check_attached_shaders(demo::Check& check)
        {
            check.expect_eq(program_param(program_a_, mgl::ProgramParameter::AttachedShaders), 2,
                            "ATTACHED_SHADERS = 2 for program A");
            std::array<mgl::ShaderId, 4> attached{};
            GLsizei count = 0;
            mgl::glGetAttachedShaders(program_a_, static_cast<GLsizei>(attached.size()), &count, attached.data());
            const bool both = count == 2 &&
                              ((attached[0] == vs_a_ && attached[1] == fs_) || (attached[0] == fs_ && attached[1] == vs_a_));
            check.expect(both, "glGetAttachedShaders lists exactly the vertex and fragment shader");
            // A linked program no longer needs its shader objects.
            mgl::glDetachShader(program_a_, vs_a_);
            mgl::glDetachShader(program_a_, fs_);
            check.expect_eq(program_param(program_a_, mgl::ProgramParameter::AttachedShaders), 0,
                            "after glDetachShader the program has no attached shaders");
            check.expect_eq(program_param(program_a_, mgl::ProgramParameter::LinkStatus), GL_TRUE,
                            "detaching shaders does not affect the linked executable");
        }

        void check_compile_failures(demo::Check& check)
        {
            // A syntax error, and a fragment shader without a float precision (GLSL ES 3.00 §4.5.4:
            // the fragment language has no default float precision).
            const struct { const char* src; const char* what; } cases[] = {
                {"#version 300 es\nprecision mediump float;\nout vec4 o;\nvoid main() { o = vec4(1.0) }\n",
                 "missing semicolon"},
                {"#version 300 es\nout vec4 o;\nvoid main() { o = vec4(1.0); }\n",
                 "fragment shader without a default float precision"},
            };
            for (const auto& c : cases)
            {
                const mgl::ShaderId s = compile_keep(mgl::ShaderType::Fragment, c.src);
                check.expect_eq(shader_param(s, mgl::ShaderParameter::CompileStatus), GL_FALSE,
                                std::format("COMPILE_STATUS = FALSE ({})", c.what));
                // A failed compile is reported through status + log, not through glGetError.
                check.expect_no_gl_error(std::format("compile failure raised no GL error ({})", c.what));
                const std::string log = demo::shader_info_log(s);
                const GLint log_len = shader_param(s, mgl::ShaderParameter::InfoLogLength);
                // The log content is implementation-defined; its reported length must match it.
                check.expect(log_len == 0 || log_len == static_cast<GLint>(log.size() + 1),
                             "INFO_LOG_LENGTH = log length + 1 (or 0 for an empty log)");
                std::cout << "     compile log (" << c.what << "): " << log.substr(0, log.find('\n')) << "\n";
                mgl::glDeleteShader(s);
            }
        }

        void check_link_failures(demo::Check& check)
        {
            // (a) interface mismatch: vec3 output linked against vec4 input (GLSL ES 3.00 §4.3.10).
            const mgl::ShaderId vs = compile_keep(mgl::ShaderType::Vertex,
                "#version 300 es\nout vec3 v_c;\nvoid main() { v_c = vec3(1.0); gl_Position = vec4(0.0); }\n");
            const mgl::ShaderId fs = compile_keep(mgl::ShaderType::Fragment,
                "#version 300 es\nprecision mediump float;\nin vec4 v_c;\nout vec4 o;\nvoid main() { o = v_c; }\n");
            const mgl::ProgramId mismatch = link_keep({vs, fs});
            check.expect_eq(program_param(mismatch, mgl::ProgramParameter::LinkStatus), GL_FALSE,
                            "LINK_STATUS = FALSE for a vec3 -> vec4 varying type mismatch");
            check.expect_no_gl_error("link failure raised no GL error");
            std::cout << "     link log: " << demo::program_info_log(mismatch).substr(0, 120) << "\n";

            // Using a program whose last link failed is an error (ES 3.0 §2.12.3).
            mgl::glUseProgram(mismatch);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUseProgram(unlinked program) -> INVALID_OPERATION");

            // (b) ES 3.0 programs must contain both a vertex and a fragment shader.
            const mgl::ProgramId vertex_only = link_keep({vs});
            check.expect_eq(program_param(vertex_only, mgl::ProgramParameter::LinkStatus), GL_FALSE,
                            "LINK_STATUS = FALSE for a program without a fragment shader");

            mgl::glDeleteProgram(mismatch);
            mgl::glDeleteProgram(vertex_only);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
        }

        void check_deferred_shader_deletion(demo::Check& check)
        {
            const mgl::ShaderId s = compile_keep(mgl::ShaderType::Vertex, vertex_source_);
            const mgl::ProgramId p = mgl::glCreateProgram();
            mgl::glAttachShader(p, s);
            mgl::glDeleteShader(s);  // attached -> only flagged (ES 3.0 §2.12.1)
            check.expect(mgl::glIsShader(s), "a deleted but attached shader still exists (glIsShader = TRUE)");
            check.expect_eq(shader_param(s, mgl::ShaderParameter::DeleteStatus), GL_TRUE, "DELETE_STATUS = TRUE");
            mgl::glDetachShader(p, s);  // last attachment gone -> really deleted
            check.expect(!mgl::glIsShader(s), "detaching the flagged shader deletes it (glIsShader = FALSE)");
            mgl::glDeleteShader(mgl::ShaderId{0});  // "DeleteShader will silently ignore the value zero"
            check.expect_no_gl_error("glDeleteShader(0) is silently ignored");
            mgl::glDeleteProgram(p);
        }

        void check_deferred_program_deletion(demo::Env& env, demo::Check& check)
        {
            const mgl::ProgramId doomed = demo::build_program_from_source(vertex_source_, demo_fragment_source(), "doomed");
            const mgl::UniformLocation color = mgl::glGetUniformLocation(doomed, "u_color");
            mgl::glUseProgram(doomed);
            mgl::glDeleteProgram(doomed);  // in use -> only flagged
            check.expect(mgl::glIsProgram(doomed), "the current program survives glDeleteProgram (glIsProgram = TRUE)");
            check.expect_eq(program_param(doomed, mgl::ProgramParameter::DeleteStatus), GL_TRUE,
                            "DELETE_STATUS = TRUE for the current program");

            // It is still fully usable while current.
            mgl::glUniform4f(color, kProbeColor[0], kProbeColor[1], kProbeColor[2], kProbeColor[3]);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(0, 0, 4, 4);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glDisable(mgl::Capability::ScissorTest);
            check.expect_pixel(1, 1, to_rgba8(kProbeColor), 1, "a flagged-for-deletion current program still renders");

            mgl::glUseProgram(program_a_);  // no longer current -> deleted now
            check.expect(!mgl::glIsProgram(doomed), "the program is deleted once it stops being current");
            (void)env;
        }

        void check_validation(demo::Env& env, demo::Check& check)
        {
            // Two active samplers of *different types* on the same unit (both default to unit 0):
            // drawing must fail with INVALID_OPERATION and glValidateProgram must report it
            // (ES 3.0 §2.12.9.7).
            const mgl::ProgramId p = demo::build_program_from_source(vertex_source_,
                "#version 300 es\nprecision mediump float;\nuniform sampler2D u_2d;\nuniform samplerCube u_cube;\n"
                "out vec4 o;\nvoid main() { o = texture(u_2d, vec2(0.5)) + texture(u_cube, vec3(1.0, 0.0, 0.0)); }\n",
                "sampler-conflict");
            mgl::glUseProgram(p);
            mgl::glValidateProgram(p);
            check.expect_eq(program_param(p, mgl::ProgramParameter::ValidateStatus), GL_FALSE,
                            "VALIDATE_STATUS = FALSE: sampler2D and samplerCube both on unit 0");
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(0, 0, 1, 1);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "drawing with the conflicting samplers -> INVALID_OPERATION");

            mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_cube"), 1);  // move the cube sampler to unit 1
            mgl::glValidateProgram(p);
            check.expect_eq(program_param(p, mgl::ProgramParameter::ValidateStatus), GL_TRUE,
                            "VALIDATE_STATUS = TRUE once the samplers use different units");
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);  // incomplete textures sample (0,0,0,1)
            check.expect_no_gl_error("drawing after fixing the sampler units succeeds");
            mgl::glDisable(mgl::Capability::ScissorTest);

            mgl::glUseProgram(program_a_);
            mgl::glValidateProgram(program_a_);
            check.expect_eq(program_param(program_a_, mgl::ProgramParameter::ValidateStatus), GL_TRUE,
                            "program A validates against the current state");
            mgl::glDeleteProgram(p);
            (void)env;
        }

        void check_precision_formats(demo::Check& check)
        {
            // Minimum ranges/precisions: GLSL ES 3.00 §4.5.1 (encoded as log2 values, ES 3.0 §6.1.12).
            // ES 3.0 requires highp in both stages.
            struct Req { mgl::PrecisionType type; const char* name; int r0, r1, prec; };
            const Req reqs[] = {
                {mgl::PrecisionType::HighFloat, "HIGH_FLOAT", 126, 127, 23},
                {mgl::PrecisionType::MediumFloat, "MEDIUM_FLOAT", 14, 14, 10},
                {mgl::PrecisionType::LowFloat, "LOW_FLOAT", 1, 1, 8},
                {mgl::PrecisionType::HighInt, "HIGH_INT", 31, 30, 0},
                {mgl::PrecisionType::MediumInt, "MEDIUM_INT", 15, 14, 0},
                {mgl::PrecisionType::LowInt, "LOW_INT", 8, 7, 0},
            };
            for (const auto stage : {mgl::ShaderType::Vertex, mgl::ShaderType::Fragment})
                for (const auto& r : reqs)
                {
                    GLint range[2] = {-1, -1};
                    GLint precision = -1;
                    mgl::glGetShaderPrecisionFormat(stage, r.type, range, &precision);
                    check.expect(range[0] >= r.r0 && range[1] >= r.r1 && precision >= r.prec,
                                 std::format("{} {} meets the GLSL ES 3.00 minimum", stage == mgl::ShaderType::Vertex ? "VS" : "FS", r.name),
                                 std::format("range=({},{}) precision={} required >=({},{}) >={}", range[0], range[1],
                                             precision, r.r0, r.r1, r.prec));
                }
        }

        void check_compiler_and_binaries(demo::Check& check)
        {
            GLboolean has_compiler = GL_FALSE;
            mgl::glGetBooleanv(mgl::GetParameter::ShaderCompiler, &has_compiler);
            check.expect(has_compiler == GL_TRUE, "SHADER_COMPILER = TRUE (required in ES 3.0)");

            // glReleaseShaderCompiler is only a hint; compiling afterwards must still work.
            mgl::glReleaseShaderCompiler();
            const mgl::ShaderId s = compile_keep(mgl::ShaderType::Vertex, vertex_source_);
            check.expect_eq(shader_param(s, mgl::ShaderParameter::CompileStatus), GL_TRUE,
                            "compiling after glReleaseShaderCompiler succeeds");

            // OpenGL ES defines no shader binary format; the list may legitimately be empty.
            const int n_formats = demo::get_integer(mgl::GetParameter::NumShaderBinaryFormats);
            std::vector<GLint> formats(static_cast<std::size_t>(std::max(n_formats, 0)));
            if (n_formats > 0) mgl::glGetIntegerv(mgl::GetParameter::ShaderBinaryFormats, formats.data());
            std::cout << "     NUM_SHADER_BINARY_FORMATS = " << n_formats << "\n";
            check.expect(n_formats >= 0, "NUM_SHADER_BINARY_FORMATS is a valid count");
            // A format the implementation did not advertise must be rejected with INVALID_ENUM (ES 3.0 §2.12.2).
            // meta-gl's ShaderBinaryFormat is opaque: its values come from the query above, so a raw
            // value is the intended way to name one. 0 (GL_NONE) is never a binary format token.
            const std::uint8_t junk[4] = {0, 1, 2, 3};
            mgl::glShaderBinary(1, &s, static_cast<mgl::ShaderBinaryFormat>(0u), junk, 4);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glShaderBinary with an unsupported format -> INVALID_ENUM");
            mgl::glDeleteShader(s);
        }

        static std::string demo_fragment_source()
        {
            return "#version 300 es\nprecision mediump float;\nuniform vec4 u_color;\nlayout(location = 0) out vec4 o;\n"
                   "void main() { o = u_color; }\n";
        }

        std::string vertex_source_, part0_, part1_, part2_;
        mgl::ShaderId vs_a_{}, fs_{};
        mgl::ProgramId program_a_{}, program_b_{};
        mgl::UniformLocation color_a_{}, color_b_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-shader-compilation",
        .title = "Shader and program objects",
        .min_es = {3, 0},
        .summary = "Compile/link status and logs, multi-string sources, deferred deletion, program validation, "
                   "precision guarantees and the shader-binary entry point.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ShaderCompilation>(argc, argv, info);
}
