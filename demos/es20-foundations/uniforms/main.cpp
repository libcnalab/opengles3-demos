// es20-uniforms
//
// Every glUniform* entry point of OpenGL ES 2.0 and 3.0 (float/int/uint scalars and vectors, their
// array (v) forms, all nine matrix shapes), plus bool, struct, struct-array, partial-array and
// matrix-array uniforms. The fragment shader compares each uniform with the exact value uploaded
// here and paints its grid cell green or red. Validation then also reads every value back with
// glGetUniform{f,i,ui}v, enumerates the program's active uniforms, and provokes each
// INVALID_OPERATION case of ES 3.0 §2.12.6.1.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <array>
#include <format>
#include <map>
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int kTests = 38;
    constexpr int kColumns = 8;
    constexpr int kRows = (kTests + kColumns - 1) / kColumns;  // 5: the grid partitions the whole surface

    class Uniforms final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kColumns || env.height < kRows)
                throw demo::Skip(std::format("the {}x{} grid of uniform cells needs at least a {}x{} surface (got {}x{})",
                                             kColumns, kRows, kColumns, kRows, env.width, env.height));
            program_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/uniforms.frag");
            mgl::glUseProgram(program_);  // glUniform* always writes to the *current* program
            upload();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glUseProgram(program_);
            const Cell c = cell_size(env);
            mgl::glUniform2i(loc("u_cell_size"), c.w, c.h);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            // 1. What the shader sees: one cell per uniform.
            const Cell c = cell_size(env);
            int green = 0;
            std::string red;
            for (int i = 0; i < kTests; ++i)
            {
                const demo::Rgba8 p = demo::read_pixel(c.center_x(i), c.center_y(i));
                if (p == demo::Rgba8{0, 255, 0, 255}) ++green;
                else red += std::format("{} ", i);
            }
            check.expect(green == kTests, std::format("all {} uniform cells are green (shader sees the exact uploaded values)", kTests),
                         red.empty() ? "" : "red cells: " + red);

            // Negative control: the comparison must be able to fail. Change one value, redraw, and
            // require exactly that cell to turn red; then restore it.
            mgl::glUseProgram(program_);
            mgl::glUniform3i(loc("u_iv3"), 3, -4, 6);  // cell 10 expects (3, -4, 5)
            render(env, 0);
            check.expect_pixel(c.center_x(10), c.center_y(10), {255, 0, 0, 255}, 0,
                               "negative control: a wrong ivec3 component turns its cell red");
            check.expect_pixel(c.center_x(11), c.center_y(11), {0, 255, 0, 255}, 0,
                               "negative control: the neighbouring cell stays green");
            mgl::glUniform3i(loc("u_iv3"), 3, -4, 5);
            render(env, 0);

            check_readback(check);
            check_active_uniforms(check);
            check_errors(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(program_);
        }

    private:
        // Cell i covers [x*w, (x+1)*w) x [y*h, (y+1)*h) with x = i % 8, y = i / 8 (the shader's integer division
        // of gl_FragCoord). The size follows each axis of the surface, so all five rows are always visible.
        struct Cell
        {
            int w, h;
            [[nodiscard]] int center_x(int i) const { return (i % kColumns) * w + w / 2; }
            [[nodiscard]] int center_y(int i) const { return (i / kColumns) * h + h / 2; }
        };
        static Cell cell_size(const demo::Env& env) { return {env.width / kColumns, env.height / kRows}; }

        mgl::UniformLocation loc(const char* name) const
        {
            const mgl::UniformLocation l = mgl::glGetUniformLocation(program_, name);
            if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
            return l;
        }

        void upload()
        {
            // --- float scalars/vectors: glUniform{1234}f and their array forms glUniform{1234}fv ----
            mgl::glUniform1f(loc("u_f"), 0.5f);
            mgl::glUniform2f(loc("u_v2"), 0.25f, -1.5f);
            mgl::glUniform3f(loc("u_v3"), 1.0f, 2.0f, -3.0f);
            mgl::glUniform4f(loc("u_v4"), 0.125f, 0.375f, -0.625f, 8.0f);
            const float farr[] = {1.5f, -2.5f};
            const float f8[] = {1, 2, 3, 4, 5, 6, 7, 8};
            mgl::glUniform1fv(loc("u_farr"), 2, farr);   // "u_farr" == "u_farr[0]"; count = 2 elements
            mgl::glUniform2fv(loc("u_v2arr"), 2, f8);
            mgl::glUniform3fv(loc("u_v3arr"), 2, f8);
            mgl::glUniform4fv(loc("u_v4arr"), 2, f8);

            // --- signed integers: glUniform{1234}i[v] -------------------------------------------
            mgl::glUniform1i(loc("u_i"), -7);
            mgl::glUniform2i(loc("u_iv2"), 1, -2);
            mgl::glUniform3i(loc("u_iv3"), 3, -4, 5);
            mgl::glUniform4i(loc("u_iv4"), -6, 7, -8, 9);
            const GLint iarr[] = {100, -100};
            const GLint i8[] = {1, 2, 3, 4, 5, 6, 7, 8};
            mgl::glUniform1iv(loc("u_iarr"), 2, iarr);
            mgl::glUniform2iv(loc("u_iv2arr"), 2, i8);
            mgl::glUniform3iv(loc("u_iv3arr"), 2, i8);
            mgl::glUniform4iv(loc("u_iv4arr"), 2, i8);

            // --- unsigned integers (ES 3.0): glUniform{1234}ui[v] -----------------------------------
            mgl::glUniform1ui(loc("u_u"), 0x80000001u);
            mgl::glUniform2ui(loc("u_uv2"), 1u, 0xFFFFFFFFu);
            mgl::glUniform3ui(loc("u_uv3"), 10u, 20u, 30u);
            mgl::glUniform4ui(loc("u_uv4"), 1u, 2u, 3u, 4u);
            const GLuint uarr[] = {7u, 9u};
            const GLuint u8[] = {1, 2, 3, 4, 5, 6, 7, 8};
            mgl::glUniform1uiv(loc("u_uarr"), 2, uarr);
            mgl::glUniform2uiv(loc("u_uv2arr"), 2, u8);
            mgl::glUniform3uiv(loc("u_uv3arr"), 2, u8);
            mgl::glUniform4uiv(loc("u_uv4arr"), 2, u8);

            // --- matrices: column-major data with transpose = GL_FALSE (ES 3.0+ also allows GL_TRUE) --
            const float m[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            mgl::glUniformMatrix2fv(loc("u_m2"), 1, GL_FALSE, m);
            mgl::glUniformMatrix3fv(loc("u_m3"), 1, GL_FALSE, m);
            mgl::glUniformMatrix4fv(loc("u_m4"), 1, GL_FALSE, m);
            mgl::glUniformMatrix2x3fv(loc("u_m23"), 1, GL_FALSE, m);  // 2 columns of 3 rows
            mgl::glUniformMatrix3x2fv(loc("u_m32"), 1, GL_FALSE, m);  // 3 columns of 2 rows
            mgl::glUniformMatrix2x4fv(loc("u_m24"), 1, GL_FALSE, m);
            mgl::glUniformMatrix4x2fv(loc("u_m42"), 1, GL_FALSE, m);
            mgl::glUniformMatrix3x4fv(loc("u_m34"), 1, GL_FALSE, m);
            mgl::glUniformMatrix4x3fv(loc("u_m43"), 1, GL_FALSE, m);
            mgl::glUniformMatrix2fv(loc("u_marr"), 2, GL_FALSE, m);   // array of two mat2

            // --- bool: any Uniform* of the right size; 0 -> false, non-zero -> true --------------
            mgl::glUniform2i(loc("u_bv2"), 1, 0);

            // --- structs: each member has its own location, named with '.' and '[i]' ----------------
            mgl::glUniform1f(loc("u_s.a"), 0.75f);
            mgl::glUniform2i(loc("u_s.b"), 4, -4);
            mgl::glUniform3f(loc("u_sa[0].c"), 1.0f, 1.0f, 1.0f);
            mgl::glUniform3f(loc("u_sa[1].c"), 9.0f, 8.0f, 7.0f);

            // --- partial array update: element k has its own location ----------------------------
            const float part[] = {0.0f, 1.0f, 2.0f, 3.0f};
            mgl::glUniform1fv(loc("u_part"), 4, part);
            mgl::glUniform1f(loc("u_part[2]"), 42.0f);  // replaces only element 2
        }

        void check_readback(demo::Check& check)
        {
            // glGetUniform*v returns the whole value at one location; array elements are read one by one.
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
                {"u_part[2]", 'f', {42}}, {"u_part[3]", 'f', {3}},
            };
            int ok = 0;
            std::string bad;
            for (const auto& e : expect)
            {
                const mgl::UniformLocation l = loc(e.name);
                bool match = true;
                if (e.kind == 'f')
                {
                    std::array<GLfloat, 16> got{};
                    mgl::glGetUniformfv(program_, l, got.data());
                    for (std::size_t i = 0; i < e.v.size(); ++i) match = match && got[i] == static_cast<float>(e.v[i]);
                }
                else if (e.kind == 'i')
                {
                    std::array<GLint, 16> got{};
                    mgl::glGetUniformiv(program_, l, got.data());
                    for (std::size_t i = 0; i < e.v.size(); ++i) match = match && got[i] == static_cast<GLint>(e.v[i]);
                }
                else
                {
                    std::array<GLuint, 16> got{};
                    mgl::glGetUniformuiv(program_, l, got.data());
                    for (std::size_t i = 0; i < e.v.size(); ++i) match = match && got[i] == static_cast<GLuint>(e.v[i]);
                }
                if (match) ++ok;
                else bad += std::string(e.name) + " ";
            }
            check.expect(ok == static_cast<int>(std::size(expect)),
                         std::format("glGetUniform{{f,i,ui}}v reads back all {} values exactly", std::size(expect)), bad);
        }

        void check_active_uniforms(demo::Check& check)
        {
            GLint count = 0, max_len = 0;
            mgl::glGetProgramiv(program_, mgl::ProgramParameter::ActiveUniforms, &count);
            mgl::glGetProgramiv(program_, mgl::ProgramParameter::ActiveUniformMaxLength, &max_len);
            std::map<std::string, std::pair<GLint, mgl::UniformType>> active;
            std::size_t longest = 0;
            for (GLint i = 0; i < count; ++i)
            {
                std::string name(static_cast<std::size_t>(max_len), '\0');
                GLsizei len = 0;
                GLint size = 0;
                mgl::UniformType type{};
                mgl::glGetActiveUniform(program_, static_cast<GLuint>(i), max_len, &len, &size, &type, name.data());
                name.resize(static_cast<std::size_t>(len));
                longest = std::max(longest, name.size());
                active[name] = {size, type};
            }
            check.expect(max_len >= static_cast<GLint>(longest + 1), "ACTIVE_UNIFORM_MAX_LENGTH covers the longest name + terminator");

            // Arrays are reported once, as "name[0]" with size = element count (ES 3.0 §2.12.6).
            const struct { const char* name; GLint size; mgl::UniformType type; } expected[] = {
                {"u_f", 1, mgl::UniformType::Float},
                {"u_farr[0]", 2, mgl::UniformType::Float},
                {"u_v4arr[0]", 2, mgl::UniformType::FloatVec4},
                {"u_iv3", 1, mgl::UniformType::IntVec3},
                {"u_uv2arr[0]", 2, mgl::UniformType::UnsignedIntVec2},
                {"u_m23", 1, mgl::UniformType::FloatMat2x3},
                {"u_m43", 1, mgl::UniformType::FloatMat4x3},
                {"u_marr[0]", 2, mgl::UniformType::FloatMat2},
                {"u_bv2", 1, mgl::UniformType::BoolVec2},
                {"u_s.b", 1, mgl::UniformType::IntVec2},
                {"u_sa[1].c", 1, mgl::UniformType::FloatVec3},
                {"u_part[0]", 4, mgl::UniformType::Float},
            };
            for (const auto& e : expected)
            {
                const auto it = active.find(e.name);
                check.expect(it != active.end() && it->second.first == e.size && it->second.second == e.type,
                             std::format("glGetActiveUniform lists {} (size {}, {})", e.name, e.size, metagl::to_string(e.type)),
                             it == active.end() ? "missing" : std::format("size={} type={}", it->second.first, metagl::to_string(it->second.second)));
            }
            check.expect(active.find("u_farr[1]") == active.end(), "array elements beyond [0] are not listed separately");
        }

        void check_errors(demo::Check& check)
        {
            // ES 3.0 §2.12.6.1: each condition -> INVALID_OPERATION and "no uniform values are changed".
            mgl::glUniform1i(loc("u_f"), 3);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniform1i on a float uniform -> INVALID_OPERATION");
            mgl::glUniform4f(loc("u_v3"), 1, 1, 1, 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniform4f on a vec3 uniform -> INVALID_OPERATION");
            const float two[] = {9.0f, 9.0f};
            mgl::glUniform1fv(loc("u_f"), 2, two);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "count = 2 on a non-array uniform -> INVALID_OPERATION");
            mgl::glUniform1f(mgl::UniformLocation{10000}, 1.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "a location that does not exist -> INVALID_OPERATION");
            GLfloat f = 0.0f;
            mgl::glGetUniformfv(program_, loc("u_f"), &f);
            check.expect_eq(f, 0.5f, "the failed calls left u_f unchanged");

            mgl::glUniform1f(mgl::UniformLocation{-1}, 123.0f);
            check.expect_no_gl_error("location -1 is silently ignored (no error)");

            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glUniform1f(loc("u_f"), 1.0f);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUniform* with no current program -> INVALID_OPERATION");
            mgl::glUseProgram(program_);
        }

        mgl::ProgramId program_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-uniforms",
        .title = "Uniform variables",
        .min_es = {3, 0},
        .summary = "All glUniform*/glUniformMatrix* variants plus bool, struct, array and matrix-array uniforms; "
                   "each verified in the shader, by glGetUniform*v readback and by active-uniform enumeration.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Uniforms>(argc, argv, info);
}
