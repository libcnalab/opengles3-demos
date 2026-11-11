// es30-hello-triangle
//
// The smallest complete OpenGL ES 3.0 program: one vertex buffer, one vertex array object,
// one shader program, one glDrawArrays call. Every step is written out with metagl::gl* calls
// on purpose - later demos use demo::build_program() for the compile/link boilerplate shown here.
//
// Validation: the rendered image is compared against a CPU evaluation of the same
// barycentric interpolation the rasterizer performs (ES 3.0 spec §3.6.1, eq. 3.9).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <array>
#include <cmath>
#include <format>
#include <string>

namespace mgl = metagl;

namespace
{
    // Interleaved vertex layout: position (vec2) followed by color (vec3).
    struct Vertex
    {
        float x, y;
        float r, g, b;
    };

    constexpr std::array<Vertex, 3> kVertices = {{
        {-0.8f, -0.8f, 1.0f, 0.0f, 0.0f},  // bottom-left : red
        {+0.8f, -0.8f, 0.0f, 1.0f, 0.0f},  // bottom-right: green
        {+0.0f, +0.8f, 0.0f, 0.0f, 1.0f},  // top         : blue
    }};

    constexpr std::array<float, 4> kClearColor = {0.10f, 0.12f, 0.20f, 1.0f};

    mgl::ShaderId compile(mgl::ShaderType type, const std::string& source, const char* label)
    {
        const mgl::ShaderId shader = mgl::glCreateShader(type);
        mgl::glShaderSource(shader, source);  // meta-gl std::string_view overload: count = 1, explicit length
        mgl::glCompileShader(shader);

        GLint status = GL_FALSE;
        mgl::glGetShaderiv(shader, mgl::ShaderParameter::CompileStatus, &status);
        if (status != GL_TRUE)
        {
            const std::string log = demo::shader_info_log(shader);
            mgl::glDeleteShader(shader);
            throw demo::Failure(std::format("{} failed to compile:\n{}", label, log));
        }
        return shader;
    }

    class HelloTriangle final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // 1. Shaders -> program.
            const mgl::ShaderId vs = compile(mgl::ShaderType::Vertex, env.load_text("shaders/triangle.vert"), "triangle.vert");
            const mgl::ShaderId fs = compile(mgl::ShaderType::Fragment, env.load_text("shaders/triangle.frag"), "triangle.frag");

            program_ = mgl::glCreateProgram();
            mgl::glAttachShader(program_, vs);
            mgl::glAttachShader(program_, fs);
            mgl::glLinkProgram(program_);

            GLint linked = GL_FALSE;
            mgl::glGetProgramiv(program_, mgl::ProgramParameter::LinkStatus, &linked);
            // Shader objects may be deleted once linked; deletion is deferred while attached,
            // so detach first to actually release them.
            mgl::glDetachShader(program_, vs);
            mgl::glDetachShader(program_, fs);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (linked != GL_TRUE)
                throw demo::Failure("program failed to link:\n" + demo::program_info_log(program_));

            // 2. Vertex data -> buffer object (GL_ARRAY_BUFFER, GL_STATIC_DRAW: written once, drawn many times).
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(kVertices), mgl::BufferUsage::StaticDraw);

            // 3. Vertex array object: records attribute formats, enables and the buffer bindings
            //    captured by glVertexAttribPointer (the ARRAY_BUFFER binding itself is NOT VAO state).
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            const mgl::AttribLocation a_position{0u};  // layout(location = 0) in the vertex shader
            const mgl::AttribLocation a_color{1u};     // layout(location = 1)
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glEnableVertexAttribArray(a_color);
            mgl::glVertexAttribPointer(a_color, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], kClearColor[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            // State that the draw relied on is still bound (no hidden state changes).
            check.expect_eq(demo::get_integer(mgl::GetParameter::CurrentProgram), static_cast<int>(program_.value),
                            "GL_CURRENT_PROGRAM is the triangle program");
            check.expect_eq(demo::get_integer(mgl::GetParameter::VertexArrayBinding), static_cast<int>(vao_.value),
                            "GL_VERTEX_ARRAY_BINDING is the triangle VAO");
            GLint stride = 0;
            mgl::glBindVertexArray(vao_);
            mgl::glGetVertexAttribiv(mgl::AttribLocation{1u}, mgl::VertexAttribParameter::ArrayStride, &stride);
            check.expect_eq(stride, static_cast<GLint>(sizeof(Vertex)), "color attribute stride recorded in the VAO");

            // Pixels outside the triangle keep the clear color.
            const demo::Rgba8 clear = {to8(kClearColor[0]), to8(kClearColor[1]), to8(kClearColor[2]), 255};
            check.expect_pixel(2, 2, clear, 1, "bottom-left corner shows the clear color");
            check.expect_pixel(env.width - 3, env.height - 3, clear, 1, "top-right corner shows the clear color");

            // Pixels inside: smooth interpolation of the vertex colors. The reference evaluates the
            // barycentric coordinates of the pixel *center*, exactly where the rasterizer samples.
            const struct { float bx, by; const char* what; } probes[] = {
                {0.34f, 0.33f, "centroid is an even mix of red, green and blue"},
                {0.85f, 0.10f, "pixel near the red vertex is mostly red"},
                {0.10f, 0.85f, "pixel near the green vertex is mostly green"},
                {0.075f, 0.075f, "pixel near the blue vertex is mostly blue"},
            };
            for (const auto& p : probes)
            {
                // Point with barycentric weights (bx, by, 1-bx-by) in NDC -> window pixel.
                const float wz = 1.0f - p.bx - p.by;
                const float nx = p.bx * kVertices[0].x + p.by * kVertices[1].x + wz * kVertices[2].x;
                const float ny = p.bx * kVertices[0].y + p.by * kVertices[1].y + wz * kVertices[2].y;
                const int px = static_cast<int>((nx + 1.0f) * 0.5f * static_cast<float>(env.width));
                const int py = static_cast<int>((ny + 1.0f) * 0.5f * static_cast<float>(env.height));
                check.expect_pixel(px, py, expected_at(env, px, py), 3, p.what);
            }
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteProgram(program_);
        }

    private:
        static std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(v * 255.0f)); }

        // CPU reference: barycentric interpolation of the vertex colors at the pixel center.
        static demo::Rgba8 expected_at(const demo::Env& env, int px, int py)
        {
            auto to_window = [&](const Vertex& v) {
                return std::array<float, 2>{(v.x + 1.0f) * 0.5f * static_cast<float>(env.width),
                                            (v.y + 1.0f) * 0.5f * static_cast<float>(env.height)};
            };
            const auto a = to_window(kVertices[0]), b = to_window(kVertices[1]), c = to_window(kVertices[2]);
            const float x = static_cast<float>(px) + 0.5f, y = static_cast<float>(py) + 0.5f;
            auto area = [](std::array<float, 2> p, std::array<float, 2> q, float rx, float ry) {
                return (q[0] - p[0]) * (ry - p[1]) - (q[1] - p[1]) * (rx - p[0]);
            };
            const float total = area(a, b, c[0], c[1]);
            const float wa = area(b, c, x, y) / total;
            const float wb = area(c, a, x, y) / total;
            const float wc = 1.0f - wa - wb;
            return {to8(wa * kVertices[0].r + wb * kVertices[1].r + wc * kVertices[2].r),
                    to8(wa * kVertices[0].g + wb * kVertices[1].g + wc * kVertices[2].g),
                    to8(wa * kVertices[0].b + wb * kVertices[1].b + wc * kVertices[2].b), 255};
        }

        mgl::ProgramId program_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-hello-triangle",
        .title = "Hello Triangle",
        .min_es = {3, 0},
        .summary = "One VBO, one VAO, one GLSL ES 3.00 program and glDrawArrays; vertex colors are "
                   "interpolated across the triangle and checked against a CPU reference.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<HelloTriangle>(argc, argv, info);
}
