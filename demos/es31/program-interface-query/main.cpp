// es31-program-interface-query
//
// The program interface query API of OpenGL ES 3.1 (ES 3.2 section 7.3.1): glGetProgramInterfaceiv,
// glGetProgramResourceIndex / Name / iv / Location. They let an application enumerate everything a linked
// program exposes - uniforms, uniform blocks, vertex inputs, fragment outputs, shader storage blocks and
// buffer variables, atomic counter buffers, transform feedback varyings - together with the properties of
// Table 7.2 (type, array size, location, block membership, offset, strides, "referenced by which stage").
//
// Two programs cover all eight interfaces:
//   gfx   vertex + fragment shader (shaders/query.vert, query.frag): UNIFORM, UNIFORM_BLOCK, PROGRAM_INPUT,
//         PROGRAM_OUTPUT, TRANSFORM_FEEDBACK_VARYING. ES 3.1 does not require shader storage blocks or
//         atomic counters in vertex/fragment shaders, so
//   comp  a compute shader (shaders/query.comp): BUFFER_VARIABLE, SHADER_STORAGE_BLOCK, ATOMIC_COUNTER_BUFFER
//         (and UNIFORM again, for the atomic counters).
//
// Every expectation is derived from the shader source, by hand, with the layout rules of ES 3.2 sections
// 7.6.2.2 (std140) and 7.8 (std430); the comments say how. The GL is then asked, and the two are compared.
// Besides the numbers, the demo proves that they are *usable*: the gfx program is drawn with explicit uniform
// locations, sampler units and uniform-block bindings that come from layout() qualifiers only (the
// application never calls glUniform1i for a sampler or glUniformBlockBinding), and the compute program is run
// with buffers that the application laid out from the *queried* offsets and strides.
//
// Picture (256x256). Upper half: 8 cells of 64x64 pixels, one per resource group of the gfx program (see
// shaders/query.frag). Strip at y = 96..127: a patch drawn through the highest legal explicit uniform
// location (MAX_UNIFORM_LOCATIONS - 1). Lower 96 rows: the "check matrix" - one 8x8 cell per recorded check
// of validate(), grouped by interface, green = passed. It is painted after all checks ran.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    using PI = mgl::ProgramInterface;
    using PIP = mgl::ProgramInterfaceParameter;
    using PRP = mgl::ProgramResourceProperty;
    using EC = mgl::ErrorCode;

    constexpr GLint kAnyLocation = -2;  // "a valid location (>= 0), assigned by the linker"
    constexpr int kGridColumns = 4;
    constexpr int kCellPx = 64;
    constexpr std::array<float, 4> kClear = {0.10f, 0.10f, 0.12f, 1.0f};

    // ---------------------------------------------------------------------------------------------------
    // Records every check twice: in the demo's Check (console, PASS/FAIL) and in a matrix that is painted at
    // the end of validate().
    class Recorder
    {
    public:
        explicit Recorder(demo::Check& check) : check_(check) {}

        void group(std::string name) { groups_.push_back({std::move(name), {}}); }

        bool ok(bool condition, std::string_view what, std::string_view detail = {})
        {
            groups_.back().results.push_back(condition);
            return check_.expect(condition, what, detail);
        }

        template <typename A, typename B>
        bool eq(const A& actual, const B& expected, std::string_view what)
        {
            const bool same = actual == expected;
            return ok(same, what, same ? std::string{} : std::format("actual={} expected={}", actual, expected));
        }

        bool error(EC expected, std::string_view what)
        {
            const bool same = check_.expect_gl_error(expected, what);
            groups_.back().results.push_back(same);
            return same;
        }

        bool no_error(std::string_view what)
        {
            const bool same = check_.expect_no_gl_error(what);
            groups_.back().results.push_back(same);
            return same;
        }

        bool deviation(bool condition, std::string_view what, std::string_view detail, std::string_view renderer, std::string_view reason)
        {
            const bool same = check_.expect_or_known_deviation(condition, what, detail, renderer, reason);
            groups_.back().results.push_back(same);
            return same;
        }

        struct Group
        {
            std::string name;
            std::vector<bool> results;
        };
        [[nodiscard]] const std::vector<Group>& groups() const { return groups_; }
        [[nodiscard]] demo::Check& check() { return check_; }

    private:
        demo::Check& check_;
        std::vector<Group> groups_;
    };

    // ---------------------------------------------------------------------------------------------------
    // Thin wrappers over the five query commands, so that the tables below read like the spec.
    struct Query
    {
        mgl::ProgramId program;
        PI iface;

        [[nodiscard]] GLint param(PIP p) const
        {
            GLint v = -12345;
            mgl::glGetProgramInterfaceiv(program, iface, p, &v);
            return v;
        }
        [[nodiscard]] GLint count() const { return param(PIP::ActiveResources); }

        [[nodiscard]] GLuint index(std::string_view name) const
        {
            const std::string z(name);
            return mgl::glGetProgramResourceIndex(program, iface, z.c_str());
        }

        [[nodiscard]] std::string name(GLuint index) const
        {
            std::string buf(256, '\0');
            GLsizei len = 0;
            mgl::glGetProgramResourceName(program, iface, index, static_cast<GLsizei>(buf.size()), &len, buf.data());
            buf.resize(static_cast<std::size_t>(len));
            return buf;
        }

        [[nodiscard]] GLint prop(GLuint index, PRP p) const
        {
            GLint v = -12345;
            GLsizei len = 0;
            mgl::glGetProgramResourceiv(program, iface, index, std::span<const PRP>(&p, 1), 1, &len, &v);
            return v;
        }

        [[nodiscard]] std::vector<GLint> prop_array(GLuint index, PRP p, GLsizei capacity) const
        {
            std::vector<GLint> v(static_cast<std::size_t>(capacity), -12345);
            GLsizei len = 0;
            mgl::glGetProgramResourceiv(program, iface, index, std::span<const PRP>(&p, 1), capacity, &len, v.data());
            v.resize(static_cast<std::size_t>(std::max(len, 0)));
            return v;
        }

        [[nodiscard]] GLint location(std::string_view name) const
        {
            const std::string z(name);
            return mgl::glGetProgramResourceLocation(program, iface, z.c_str());
        }
    };

    GLint program_param(mgl::ProgramId p, mgl::ProgramParameter pname)
    {
        GLint v = -12345;
        mgl::glGetProgramiv(p, pname, &v);
        return v;
    }

    std::string join(const std::vector<std::string>& v)
    {
        std::string s;
        for (const auto& x : v) s += x + " ";
        return s;
    }

    GLint type_token(GLenum t) { return static_cast<GLint>(t); }

    // ---------------------------------------------------------------------------------------------------
    // What ES 3.2 section 7.3.1 says each query returns, per resource, derived from the shader source.
    struct UniformExp
    {
        const char* name;       // name string of the list entry (arrays "[0]", struct members "a.b", block members "Block.m")
        GLenum type;
        GLint array_size;
        GLint location;         // explicit location, -1 (block member / atomic counter) or kAnyLocation
        const char* block;      // UNIFORM_BLOCK entry that contains it, or nullptr
        GLint offset;           // -1 when not backed by a buffer object
        GLint array_stride;     // -1 when not backed by a buffer object, 0 when not an array of basic types
        GLint matrix_stride;    // -1 when not backed by a buffer object, 0 when not a matrix
        GLint row_major;
        bool vs, fs;            // REFERENCED_BY_VERTEX_SHADER / _FRAGMENT_SHADER
        bool refs_deviate;      // REFERENCED_BY_FRAGMENT_SHADER is subject to the documented Mesa deviation
    };

    constexpr bool kNo = false;

    // Default block: offsets and strides are -1 ("not backed by a buffer object"), IS_ROW_MAJOR is 0.
    // Block members (std140, ES 3.2 section 7.6.2.2):
    //   Params { vec4 scale; mat3 basis; row_major mat2x3 skew; float k[2]; vec3 offset; }
    //     scale   rule 2,3: vec4 base alignment 16                     -> offset 0
    //     basis   rule 5,4: 3 column vectors, each a vec3 rounded to 16 -> offset 16, MATRIX_STRIDE 16, 48 bytes -> 64
    //     skew    rule 7:   row_major with C = 2 columns, R = 3 rows is an array of 3 row vectors (vec2),
    //                       each rounded up to 16                        -> offset 64, MATRIX_STRIDE 16, 48 bytes -> 112
    //     k[2]    rule 4:   array of scalars: stride and alignment 16    -> offset 112, ARRAY_STRIDE 16, 32 bytes -> 144
    //     offset  vec3, alignment 16                                     -> offset 144 (ends at 156)
    //   Cell { vec4 color; ivec2 id; }: color 0 | id 16 (ivec2 alignment 8, 16 is a multiple) ends at 24.
    const std::vector<UniformExp>& gfx_uniforms()
    {
        static const std::vector<UniformExp> v = {
            {"u_tint", GL_FLOAT_VEC4, 1, 4, nullptr, -1, -1, -1, 0, true, true, kNo},
            {"u_weights[0]", GL_FLOAT, 3, 8, nullptr, -1, -1, -1, 0, false, true, kNo},
            {"u_light.dir", GL_FLOAT_VEC3, 1, 20, nullptr, -1, -1, -1, 0, false, true, kNo},
            {"u_light.power", GL_FLOAT, 1, 21, nullptr, -1, -1, -1, 0, false, true, kNo},
            {"u_tex", GL_SAMPLER_2D, 1, kAnyLocation, nullptr, -1, -1, -1, 0, false, true, kNo},
            {"u_texarr[0]", GL_SAMPLER_2D, 2, 30, nullptr, -1, -1, -1, 0, false, true, kNo},
            {"u_pick", GL_INT_VEC2, 1, kAnyLocation, nullptr, -1, -1, -1, 0, false, true, kNo},
            {"Params.scale", GL_FLOAT_VEC4, 1, -1, "Params", 0, 0, 0, 0, true, false, true},
            {"Params.basis", GL_FLOAT_MAT3, 1, -1, "Params", 16, 0, 16, 0, true, false, true},
            {"Params.skew", GL_FLOAT_MAT2x3, 1, -1, "Params", 64, 0, 16, 1, true, false, true},
            {"Params.k[0]", GL_FLOAT, 2, -1, "Params", 112, 16, 0, 0, true, false, true},
            {"Params.offset", GL_FLOAT_VEC3, 1, -1, "Params", 144, 0, 0, 0, true, false, true},
            {"Cell.color", GL_FLOAT_VEC4, 1, -1, "Cell[0]", 0, 0, 0, 0, false, true, kNo},
            {"Cell.id", GL_INT_VEC2, 1, -1, "Cell[0]", 16, 0, 0, 0, false, true, kNo},
        };
        return v;
    }

    struct BlockExp
    {
        const char* name;
        GLint binding;
        std::vector<const char*> members;  // names in the UNIFORM / BUFFER_VARIABLE list
        GLint min_data_size;               // the end of the last member (BUFFER_DATA_SIZE is "implementation-dependent minimum")
        bool vs, fs, cs;
        bool refs_deviate;
    };

    // The blocks of the gfx program (names as listed: arrays of blocks get one entry per element, "Cell[0]").
    const std::vector<BlockExp>& gfx_blocks()
    {
        static const std::vector<BlockExp> v = {
            {"Params", 3, {"Params.scale", "Params.basis", "Params.skew", "Params.k[0]", "Params.offset"}, 156, true, false, false, true},
            {"Cell[0]", 6, {"Cell.color", "Cell.id"}, 24, false, true, false, false},
            {"Cell[1]", 7, {"Cell.color", "Cell.id"}, 24, false, true, false, false},
        };
        return v;
    }

    struct IoExp
    {
        const char* name;
        GLenum type;
        GLint array_size;
        GLint location;
    };

    const std::vector<IoExp>& gfx_inputs()
    {
        static const std::vector<IoExp> v = {
            {"a_position", GL_FLOAT_VEC3, 1, 0},
            {"a_color", GL_FLOAT_VEC4, 1, 3},
            {"a_cell", GL_INT_VEC2, 1, kAnyLocation},
            {"gl_VertexID", GL_INT, 1, -1},  // built-in inputs are listed, with location -1 (section 7.3.1.1)
        };
        return v;
    }

    const std::vector<IoExp>& gfx_outputs()
    {
        static const std::vector<IoExp> v = {
            {"o_color", GL_FLOAT_VEC4, 1, 0},
            {"o_id[0]", GL_INT_VEC2, 2, 1},  // layout(location = 1) out ivec2 o_id[2]: one entry, two locations
        };
        return v;
    }

    const std::vector<IoExp>& gfx_tf()
    {
        static const std::vector<IoExp> v = {
            {"v_color", GL_FLOAT_VEC4, 1, 0},
            {"v_w", GL_FLOAT, 2, 0},  // captured as a whole array: no "[0]" suffix, ARRAY_SIZE 2
            {"gl_Position", GL_FLOAT_VEC4, 1, 0},
        };
        return v;
    }

    // Compute program. std430 (ES 3.2 section 7.8; rules of 7.6.2.2 without the rounding of array strides to 16):
    //   Source { vec4 head; mat3 m; float w[3]; row_major mat2x3 rm; }
    //     head 0 | m 16: 3 columns of vec3 (base alignment 16, stride 16), 48 bytes -> 64 | w 64: stride 4, 12 bytes -> 76
    //     rm: row_major C = 2, R = 3 is 3 rows of vec2 (alignment 8, stride 8) -> offset 80 (76 rounded to 8), 24 bytes -> 104
    //   Result { uint count; Item items[]; }  Item { vec2 p; float s; }: base alignment 8, size 12 -> array stride 16
    //     count 0 | items at 8: p at 8, s at 16
    //   Plain { uint plain_value; } 0
    struct BufferVarExp
    {
        const char* name;
        GLenum type;
        GLint array_size;
        const char* block;
        GLint offset, array_stride, matrix_stride, row_major, top_size, top_stride;
    };

    const std::vector<BufferVarExp>& comp_buffer_vars()
    {
        static const std::vector<BufferVarExp> v = {
            {"Source.head", GL_FLOAT_VEC4, 1, "Source", 0, 0, 0, 0, 1, 0},
            {"Source.m", GL_FLOAT_MAT3, 1, "Source", 16, 0, 16, 0, 1, 0},
            {"Source.w[0]", GL_FLOAT, 3, "Source", 64, 4, 0, 0, 1, 0},
            {"Source.rm", GL_FLOAT_MAT2x3, 1, "Source", 80, 0, 8, 1, 1, 0},
            {"Result.count", GL_UNSIGNED_INT, 1, "Result", 0, 0, 0, 0, 1, 0},
            // items[] is a runtime-sized array of structures: a "top-level array" whose size is 0 and whose stride
            // is 16; only the entries for element 0 exist, and ARRAY_SIZE (of p and s themselves) is 1.
            {"Result.items[0].p", GL_FLOAT_VEC2, 1, "Result", 8, 0, 0, 0, 0, 16},
            {"Result.items[0].s", GL_FLOAT, 1, "Result", 16, 0, 0, 0, 0, 16},
            {"plain_value", GL_UNSIGNED_INT, 1, "Plain", 0, 0, 0, 0, 1, 0},
        };
        return v;
    }

    const std::vector<BlockExp>& comp_ssbos()
    {
        static const std::vector<BlockExp> v = {
            {"Source", 1, {"Source.head", "Source.m", "Source.w[0]", "Source.rm"}, 104, false, false, true, false},
            {"Result", 5, {"Result.count", "Result.items[0].p", "Result.items[0].s"}, 20, false, false, true, false},
            {"Plain", 0, {"plain_value"}, 4, false, false, true, false},
        };
        return v;
    }

    // ---------------------------------------------------------------------------------------------------
    // CPU-side data of the picture.
    struct Vertex
    {
        float x, y, z;
        float r, g, b, a;
        std::int32_t cell, aux;
    };

    // Texel colors (bytes). Unit 2: u_tex. Units 4 and 5: u_texarr[0] and [1].
    constexpr std::array<std::uint8_t, 4> kTexUnit2 = {255, 64, 191, 255};
    constexpr std::array<std::uint8_t, 4> kTexUnit4 = {64, 191, 255, 255};
    constexpr std::array<std::uint8_t, 4> kTexUnit5 = {191, 255, 64, 255};

    // Values uploaded to the uniforms / blocks. All are multiples of 1/16 or similarly exact, away from x.5 / 255.
    constexpr std::array<float, 4> kTint = {0.75f, 1.00f, 1.00f, 0.25f};
    constexpr std::array<float, 4> kVertexColor = {1.00f, 0.75f, 0.25f, 1.00f};
    constexpr std::array<float, 3> kWeights = {0.25f, 0.75f, 0.125f};
    constexpr std::array<float, 3> kLightDir = {1.00f, 0.50f, 0.25f};
    constexpr float kLightPower = 0.75f;
    constexpr std::array<GLint, 2> kPick = {3, 1};

    // The std140 image of block Params at the offsets derived in gfx_uniforms().
    struct ParamsImage
    {
        std::array<float, 40> f{};  // 160 bytes
        void set(int byte_offset, float v) { f[static_cast<std::size_t>(byte_offset / 4)] = v; }
    };

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(v * 255.0f)); }

    // ---------------------------------------------------------------------------------------------------
    class ProgramInterfaceQuery final : public demo::Demo
    {
    public:
        // =============================================================================================
        void init(demo::Env& env) override
        {
            // ---- gfx: vertex + fragment, with transform feedback varyings chosen BEFORE the link ---------
            {
                const mgl::ShaderId vs = compile(mgl::ShaderType::Vertex, env.load_text("shaders/query.vert"), "query.vert");
                const mgl::ShaderId fs = compile(mgl::ShaderType::Fragment, env.load_text("shaders/query.frag"), "query.frag");
                gfx_ = mgl::glCreateProgram();
                mgl::glAttachShader(gfx_, vs);
                mgl::glAttachShader(gfx_, fs);
                // The TRANSFORM_FEEDBACK_VARYING interface lists these, in this order (section 7.3.1.1).
                const GLchar* varyings[] = {"v_color", "v_w", "gl_Position"};
                mgl::glTransformFeedbackVaryings(gfx_, 3, varyings, mgl::TransformFeedbackBufferMode::InterleavedAttribs);
                mgl::glLinkProgram(gfx_);
                finish_link(gfx_, vs, fs, "gfx");
            }
            // ---- comp: shader storage blocks, buffer variables, atomic counters -----------------------------
            {
                const mgl::ShaderId cs = compile(mgl::ShaderType::Compute, env.load_text("shaders/query.comp"), "query.comp");
                comp_ = mgl::glCreateProgram();
                mgl::glAttachShader(comp_, cs);
                mgl::glLinkProgram(comp_);
                finish_link(comp_, cs, mgl::ShaderId{0}, "comp");
            }

            // ---- the highest legal explicit uniform location --------------------------------------------------
            max_locations_ = demo::get_integer(mgl::GetParameter::MaxUniformLocations);
            const std::string edge_src = with_define(env.load_text("shaders/loc_edge.frag"), "LOC", max_locations_ - 1);
            {
                const mgl::ShaderId vs = compile(mgl::ShaderType::Vertex, env.load_text("shaders/fullscreen.vert"), "fullscreen.vert");
                const mgl::ShaderId fs = compile(mgl::ShaderType::Fragment, edge_src, "loc_edge.frag");
                edge_ = mgl::glCreateProgram();
                mgl::glAttachShader(edge_, vs);
                mgl::glAttachShader(edge_, fs);
                mgl::glLinkProgram(edge_);
                finish_link(edge_, vs, fs, "edge");
            }
            src_edge_ = env.load_text("shaders/loc_edge.frag");
            src_dup_ = env.load_text("shaders/loc_duplicate.frag");
            src_overlap_ = env.load_text("shaders/loc_overlap.frag");
            src_fullscreen_ = env.load_text("shaders/fullscreen.vert");

            // ---- vertex data: 8 cells x 2 triangles. Attribute locations come from the QUERY API ---------------
            std::vector<Vertex> vertices;
            constexpr float quad[6][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 1}};
            for (int cell = 0; cell < 8; ++cell)
                for (const auto& q : quad)
                    vertices.push_back({q[0], q[1], 0.0f, kVertexColor[0], kVertexColor[1], kVertexColor[2], kVertexColor[3], cell, cell * 2});
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(vertices), mgl::BufferUsage::StaticDraw);
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            const auto in_loc = [&](const char* name) {
                const GLint l = mgl::glGetProgramResourceLocation(gfx_, PI::ProgramInput, name);
                if (l < 0) throw demo::Failure(std::format("input '{}' has no location", name));
                return mgl::AttribLocation{static_cast<GLuint>(l)};
            };
            const mgl::AttribLocation a_position = in_loc("a_position"), a_color = in_loc("a_color"), a_cell = in_loc("a_cell");
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glEnableVertexAttribArray(a_color);
            mgl::glVertexAttribPointer(a_color, 4, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glEnableVertexAttribArray(a_cell);
            mgl::glVertexAttribIPointer(a_cell, 2, mgl::DataType::Int, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, cell)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // ---- textures for the units that layout(binding) selects ------------------------------------------
            mgl::glGenTextures(3, textures_.data());
            const std::array<const std::array<std::uint8_t, 4>*, 3> colors = {&kTexUnit2, &kTexUnit4, &kTexUnit5};
            const std::array<mgl::TextureUnit, 3> units = {mgl::TextureUnit::Texture2, mgl::TextureUnit::Texture4, mgl::TextureUnit::Texture5};
            for (std::size_t i = 0; i < 3; ++i)
            {
                mgl::glActiveTexture(units[i]);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, textures_[i]);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, GL_NEAREST);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, GL_NEAREST);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 1, 1, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, colors[i]->data());
            }
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            // ---- uniform buffers for the blocks, bound to the binding points the shader declared ----------------
            // Block Params: packed at the std140 offsets derived by hand (not the queried ones).
            ParamsImage params;
            for (std::size_t i = 0; i < params.f.size(); ++i) params.f[i] = 0.5f;  // sentinel: a wrong offset reads 0.5
            params.set(0, 0.0625f); params.set(4, 0.125f); params.set(8, 0.1875f); params.set(12, 1.0f);  // scale
            // basis, column-major, 3 columns at stride 16: column 1 = (0.125, 0.0625, 0.0625)
            params.set(16 + 16 + 0, 0.125f); params.set(16 + 16 + 4, 0.0625f); params.set(16 + 16 + 8, 0.0625f);
            // skew, row_major mat2x3: element (column c, row r) at 64 + r * 16 + c * 4
            params.set(64 + 1 * 16 + 0 * 4, 0.0625f);  // skew[0][1]
            params.set(64 + 0 * 16 + 1 * 4, 0.125f);   // skew[1][0]
            params.set(64 + 2 * 16 + 1 * 4, 0.25f);    // skew[1][2]
            params.set(112, 0.0625f); params.set(128, 0.125f);                   // k[0], k[1] at stride 16
            params.set(144, 0.25f); params.set(148, 0.1875f); params.set(152, 0.0625f);  // offset
            struct CellImage { float color[4]; std::int32_t id[2]; std::int32_t pad[2]; };
            const CellImage cell0 = {{0.25f, 0.75f, 0.25f, 1.0f}, {1, 2}, {0, 0}};
            const CellImage cell1 = {{0.75f, 0.25f, 0.75f, 1.0f}, {3, 4}, {0, 0}};
            mgl::glGenBuffers(3, ubos_.data());
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, ubos_[0]);
            mgl::glBufferData(mgl::BufferTarget::Uniform, std::span<const float>(params.f), mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, ubos_[1]);
            mgl::glBufferData(mgl::BufferTarget::Uniform, sizeof(CellImage), &cell0, mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, ubos_[2]);
            mgl::glBufferData(mgl::BufferTarget::Uniform, sizeof(CellImage), &cell1, mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, 3, ubos_[0]);  // layout(binding = 3) Params
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, 6, ubos_[1]);  // layout(binding = 6) Cell cells[0]
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, 7, ubos_[2]);  // ... and cells[1] at the next binding point
        }

        // =============================================================================================
        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);

            // ---- the 8 cells -------------------------------------------------------------------------------
            // Uniform values go to HARD-CODED locations 4, 8, 20, 21: they are what the layout qualifiers say.
            // Samplers and uniform blocks are never configured by the API: layout(binding) did it.
            mgl::glUseProgram(gfx_);
            mgl::glUniform4f(mgl::UniformLocation{4}, kTint[0], kTint[1], kTint[2], kTint[3]);                  // u_tint
            mgl::glUniform1fv(mgl::UniformLocation{8}, 3, kWeights.data());                                    // u_weights[0..2] at 8, 9, 10
            mgl::glUniform3f(mgl::UniformLocation{20}, kLightDir[0], kLightDir[1], kLightDir[2]);              // u_light.dir
            mgl::glUniform1f(mgl::UniformLocation{21}, kLightPower);                                           // u_light.power
            // u_pick has no layout(location): its location is whatever the linker chose, so ask for it.
            const GLint pick_location = mgl::glGetProgramResourceLocation(gfx_, PI::Uniform, "u_pick");
            mgl::glUniform2i(mgl::UniformLocation{pick_location}, kPick[0], kPick[1]);

            // The vertex shader places cell n in the upper half of NDC (y in [0, 1]), i.e. the upper half of the window.
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glBindVertexArray(vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 48);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});

            // ---- the patch through the highest explicit location ---------------------------------------------
            mgl::glUseProgram(edge_);
            mgl::glUniform1f(mgl::UniformLocation{max_locations_ - 1}, 0.75f);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 96, 32, 32);
            mgl::glScissor(0, 96, 32, 32);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // =============================================================================================
        void validate(demo::Env& env, demo::Check& check) override
        {
            Recorder rec(check);
            check_picture(rec, env);
            check_uniform_interface(rec);
            check_empty_interfaces(rec);
            check_program_io(rec);
            check_transform_feedback(rec);
            check_compute_interfaces(rec);
            check_names_and_locations(rec);
            check_bindings(rec);
            check_explicit_locations(rec, env);
            check_errors(rec);
            check_compute_run(rec);
            paint_matrix(env, rec);
        }

        // =============================================================================================
        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, 3, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, 6, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::Uniform, 7, mgl::BufferId{0});
            for (const mgl::TextureUnit u : {mgl::TextureUnit::Texture2, mgl::TextureUnit::Texture4, mgl::TextureUnit::Texture5})
            {
                mgl::glActiveTexture(u);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            }
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glDeleteTextures(3, textures_.data());
            mgl::glDeleteBuffers(3, ubos_.data());
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteProgram(gfx_);
            mgl::glDeleteProgram(comp_);
            mgl::glDeleteProgram(edge_);
        }

    private:
        // ---------------------------------------------------------------------------------------------
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

        static void finish_link(mgl::ProgramId p, mgl::ShaderId a, mgl::ShaderId b, const char* label)
        {
            const GLint ok = program_param(p, mgl::ProgramParameter::LinkStatus);
            for (const mgl::ShaderId s : {a, b})
            {
                if (s.value == 0u) continue;
                mgl::glDetachShader(p, s);
                mgl::glDeleteShader(s);
            }
            if (ok != GL_TRUE) throw demo::Failure(std::format("{} failed to link:\n{}", label, demo::program_info_log(p)));
        }

        // Inserts "#define NAME value" after the #version line (a #version directive must stay first).
        static std::string with_define(const std::string& source, const char* name, GLint value)
        {
            const std::size_t cut = source.find('\n') + 1;
            return source.substr(0, cut) + std::format("#define {} {}\n", name, value) + source.substr(cut);
        }

        // Tries to build a vertex+fragment program from `fs_source`. Returns "" when it linked, otherwise the stage
        // that rejected it ("compile" or "link").
        std::string try_build(const std::string& fs_source) const
        {
            const mgl::ShaderId vs = mgl::glCreateShader(mgl::ShaderType::Vertex);
            mgl::glShaderSource(vs, src_fullscreen_);
            mgl::glCompileShader(vs);
            const mgl::ShaderId fs = mgl::glCreateShader(mgl::ShaderType::Fragment);
            mgl::glShaderSource(fs, fs_source);
            mgl::glCompileShader(fs);
            GLint compiled = GL_FALSE;
            mgl::glGetShaderiv(fs, mgl::ShaderParameter::CompileStatus, &compiled);
            std::string rejected_by = "compile";
            const mgl::ProgramId p = mgl::glCreateProgram();
            mgl::glAttachShader(p, vs);
            mgl::glAttachShader(p, fs);
            mgl::glLinkProgram(p);
            if (compiled == GL_TRUE)
                rejected_by = program_param(p, mgl::ProgramParameter::LinkStatus) == GL_TRUE ? "" : "link";
            mgl::glDeleteProgram(p);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            return rejected_by;
        }

        // ---------------------------------------------------------------------------------------------
        // Picture: the eight cells, the explicit-location patch, and a negative control.
        struct Probe
        {
            int cell, x_quarter;     // x_quarter 1 = left half, 3 = right half (of 4)
            demo::Rgba8 expected;
            const char* what;
        };

        void check_picture(Recorder& rec, const demo::Env& env)
        {
            rec.group("picture");
            const auto bytes = [](float r, float g, float b) { return demo::Rgba8{to8(r), to8(g), to8(b), 255}; };
            // CPU model of query.frag. a_color * tint.rgb is formed in the vertex stage; the fragment stage scales it by tint.a + 0.5.
            const float f = kTint[3] + 0.5f;
            const std::array<Probe, 12> probes = {{
                {0, 1, bytes(kVertexColor[0] * kTint[0] * f, kVertexColor[1] * kTint[1] * f, kVertexColor[2] * kTint[2] * f), "cell 0: u_tint at layout(location = 4), used by BOTH stages"},
                {1, 1, bytes(kWeights[0], kWeights[1], kWeights[2]), "cell 1: u_weights[3] written with one glUniform1fv at location 8 (elements at 8, 9, 10)"},
                {2, 1, bytes(kLightDir[0] * kLightPower, kLightDir[1] * kLightPower, kLightDir[2] * kLightPower), "cell 2: struct u_light, members at locations 20 and 21"},
                {3, 1, {kTexUnit2[0], kTexUnit2[1], kTexUnit2[2], 255}, "cell 3: u_tex reads texture unit 2 from layout(binding = 2), no glUniform1i"},
                {4, 1, {kTexUnit4[0], kTexUnit4[1], kTexUnit4[2], 255}, "cell 4 left: u_texarr[0] reads unit 4 (layout(binding = 4))"},
                {4, 3, {kTexUnit5[0], kTexUnit5[1], kTexUnit5[2], 255}, "cell 4 right: u_texarr[1] reads unit 5 (next consecutive unit)"},
                {5, 1, bytes(0.5625f, 0.625f, 0.5625f), "cell 5: block Params (std140, binding 3) read at the hand-derived offsets: scale + basis[1] + skew + k + offset"},
                {5, 3, bytes(0.5625f, 0.625f, 0.5625f), "cell 5 right half: same value everywhere"},
                {6, 1, bytes(0.25f, 0.75f, 0.25f), "cell 6 left: block array element Cell[0] reads binding point 6"},
                {6, 3, bytes(0.75f, 0.25f, 0.75f), "cell 6 right: block array element Cell[1] reads the NEXT binding point, 7"},
                {7, 1, bytes(0.75f, 0.25f, 0.75f), "cell 7: u_pick set at the location returned by glGetProgramResourceLocation"},
                {7, 3, bytes(0.75f, 0.25f, 0.75f), "cell 7 right half: same value everywhere"},
            }};
            for (const Probe& p : probes)
            {
                const demo::Rgba8 got = demo::read_pixel(probe_x(p), probe_y(p));
                bool same = true;
                for (std::size_t c = 0; c < 4; ++c)
                    if (std::abs(int{got[c]} - int{p.expected[c]}) > 1) same = false;  // 8-bit quantization +-1
                rec.ok(same, p.what, std::format("pixel({},{}) = {} expected {}", probe_x(p), probe_y(p), demo::format_rgba(got), demo::format_rgba(p.expected)));
            }

            // The patch drawn through the highest legal explicit location: u_edge = 0.75 -> (191, 64, 191).
            const demo::Rgba8 patch = demo::read_pixel(16, 112);
            const demo::Rgba8 want = {to8(0.75f), to8(0.25f), to8(0.75f), 255};
            bool patch_ok = true;
            for (std::size_t c = 0; c < 4; ++c)
                if (std::abs(int{patch[c]} - int{want[c]}) > 1) patch_ok = false;
            rec.ok(patch_ok, std::format("a uniform at layout(location = MAX_UNIFORM_LOCATIONS - 1 = {}) is set with glUniform1f and seen by the shader", max_locations_ - 1),
                   std::format("pixel(16,112) = {} expected {}", demo::format_rgba(patch), demo::format_rgba(want)));

            // Negative control: move u_tint, redraw cell 0, and the old expectation must break.
            mgl::glUseProgram(gfx_);
            mgl::glUniform4f(mgl::UniformLocation{4}, 0.25f, 0.25f, 1.0f, 0.25f);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glBindVertexArray(vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 6);  // cell 0 only
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            const demo::Rgba8 changed = demo::read_pixel(probe_x(probes[0]), probe_y(probes[0]));
            const demo::Rgba8 expect_new = bytes(kVertexColor[0] * 0.25f * f, kVertexColor[1] * 0.25f * f, kVertexColor[2] * 1.0f * f);
            bool new_ok = true, old_ok = true;
            for (std::size_t c = 0; c < 4; ++c)
            {
                if (std::abs(int{changed[c]} - int{expect_new[c]}) > 1) new_ok = false;
                if (std::abs(int{changed[c]} - int{probes[0].expected[c]}) > 1) old_ok = false;
            }
            rec.ok(new_ok && !old_ok, "negative control: glUniform4f(4, ...) changes cell 0 to the new model and breaks the old expectation (the comparison can fail)",
                   std::format("pixel = {}", demo::format_rgba(changed)));
            mgl::glUniform4f(mgl::UniformLocation{4}, kTint[0], kTint[1], kTint[2], kTint[3]);
            mgl::glBindVertexArray(vao_);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 6);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            const demo::Rgba8 restored = demo::read_pixel(probe_x(probes[0]), probe_y(probes[0]));
            bool restored_ok = true;
            for (std::size_t c = 0; c < 4; ++c)
                if (std::abs(int{restored[c]} - int{probes[0].expected[c]}) > 1) restored_ok = false;
            rec.ok(restored_ok, "negative control: restoring u_tint restores cell 0");
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        static int probe_x(const Probe& p) { return (p.cell % kGridColumns) * kCellPx + p.x_quarter * (kCellPx / 4); }
        static int probe_y(const Probe& p) { return 128 + (p.cell / kGridColumns) * kCellPx + kCellPx / 2; }

        // ---------------------------------------------------------------------------------------------
        // Generic: one block entry, compared with its table row.
        void check_block_list(Recorder& rec, const Query& blocks, const Query& vars, const std::vector<BlockExp>& expected,
                              const char* interface_name)
        {
            // List-level properties.
            GLint longest = 0, most_members = 0;
            for (const BlockExp& b : expected)
            {
                longest = std::max(longest, static_cast<GLint>(std::strlen(b.name)) + 1);
                most_members = std::max(most_members, static_cast<GLint>(b.members.size()));
            }
            rec.eq(blocks.count(), static_cast<GLint>(expected.size()), std::format("{}: ACTIVE_RESOURCES = {} (one entry per block, per array element)", interface_name, expected.size()));
            rec.eq(blocks.param(PIP::MaxNameLength), longest, std::format("{}: MAX_NAME_LENGTH = longest name + terminator = {}", interface_name, longest));
            rec.eq(blocks.param(PIP::MaxNumActiveVariables), most_members, std::format("{}: MAX_NUM_ACTIVE_VARIABLES = most members of one block = {}", interface_name, most_members));

            for (const BlockExp& b : expected)
            {
                const GLuint idx = blocks.index(b.name);
                if (!rec.ok(idx != GL_INVALID_INDEX, std::format("{} '{}': glGetProgramResourceIndex finds it", interface_name, b.name))) continue;
                std::vector<std::string> bad;
                if (blocks.name(idx) != b.name) bad.push_back("name round trip");
                if (blocks.prop(idx, PRP::NameLength) != static_cast<GLint>(std::strlen(b.name)) + 1) bad.push_back("NAME_LENGTH");
                if (blocks.prop(idx, PRP::BufferBinding) != b.binding) bad.push_back(std::format("BUFFER_BINDING {} != {}", blocks.prop(idx, PRP::BufferBinding), b.binding));
                const GLint n = blocks.prop(idx, PRP::NumActiveVariables);
                if (n != static_cast<GLint>(b.members.size())) bad.push_back(std::format("NUM_ACTIVE_VARIABLES {} != {}", n, b.members.size()));
                const GLint size = blocks.prop(idx, PRP::BufferDataSize);
                if (size < b.min_data_size) bad.push_back(std::format("BUFFER_DATA_SIZE {} < {}", size, b.min_data_size));
                // ACTIVE_VARIABLES: indices into the variable list; compare the NAMES, order is not specified.
                std::set<std::string> got, want;
                for (const GLint v : blocks.prop_array(idx, PRP::ActiveVariables, 16)) got.insert(vars.name(static_cast<GLuint>(v)));
                for (const char* m : b.members) want.insert(m);
                if (got != want) bad.push_back("ACTIVE_VARIABLES names differ");
                rec.ok(bad.empty(), std::format("{} '{}': name, NAME_LENGTH, BUFFER_BINDING = {}, NUM_ACTIVE_VARIABLES = {}, BUFFER_DATA_SIZE >= {} (driver: {}), ACTIVE_VARIABLES",
                                                interface_name, b.name, b.binding, b.members.size(), b.min_data_size, size), join(bad));

                const bool vs = blocks.prop(idx, PRP::ReferencedByVertexShader) != 0;
                const bool fs = blocks.prop(idx, PRP::ReferencedByFragmentShader) != 0;
                const bool cs = blocks.prop(idx, PRP::ReferencedByComputeShader) != 0;
                const bool refs = vs == b.vs && fs == b.fs && cs == b.cs;
                const std::string what = std::format("{} '{}': REFERENCED_BY_VERTEX/FRAGMENT/COMPUTE = {}/{}/{}", interface_name, b.name, b.vs, b.fs, b.cs);
                const std::string detail = refs ? "" : std::format("driver reports {}/{}/{}", vs, fs, cs);
                if (b.refs_deviate)
                    rec.deviation(refs, what, detail, "radeonsi",
                                  "a uniform block that only the vertex shader declares is reported as referenced by the fragment shader when the fragment shader declares other uniform blocks (Cell[2]); with a fragment shader that has no blocks the answer is correct (llvmpipe and zink conform)");
                else
                    rec.ok(refs, what, detail);
            }
        }

        // ---------------------------------------------------------------------------------------------
        void check_uniform_interface(Recorder& rec)
        {
            rec.group("UNIFORM (gfx)");
            const Query uniforms{gfx_, PI::Uniform};
            const Query blocks{gfx_, PI::UniformBlock};
            const auto& table = gfx_uniforms();

            GLint longest = 0;
            for (const UniformExp& u : table) longest = std::max(longest, static_cast<GLint>(std::strlen(u.name)) + 1);
            rec.eq(uniforms.count(), static_cast<GLint>(table.size()), std::format("UNIFORM: ACTIVE_RESOURCES = {} (7 default-block entries + 7 block members)", table.size()));
            rec.eq(uniforms.count(), program_param(gfx_, mgl::ProgramParameter::ActiveUniforms), "UNIFORM: ACTIVE_RESOURCES equals the ES 3.0 query glGetProgramiv(ACTIVE_UNIFORMS)");
            rec.eq(uniforms.param(PIP::MaxNameLength), longest, std::format("UNIFORM: MAX_NAME_LENGTH = longest name + terminator = {}", longest));

            std::set<GLint> explicit_locations;
            for (const UniformExp& u : table)
                if (u.location >= 0) explicit_locations.insert(u.location);

            for (const UniformExp& u : table)
            {
                const GLuint idx = uniforms.index(u.name);
                if (!rec.ok(idx != GL_INVALID_INDEX, std::format("UNIFORM '{}': glGetProgramResourceIndex finds it", u.name))) continue;
                std::vector<std::string> bad;
                if (uniforms.name(idx) != u.name) bad.push_back("name round trip");
                const auto expect = [&](PRP p, GLint want, const char* label) {
                    const GLint got = uniforms.prop(idx, p);
                    if (got != want) bad.push_back(std::format("{} = {} (expected {})", label, got, want));
                };
                expect(PRP::NameLength, static_cast<GLint>(std::strlen(u.name)) + 1, "NAME_LENGTH");
                expect(PRP::Type, type_token(u.type), "TYPE");
                expect(PRP::ArraySize, u.array_size, "ARRAY_SIZE");
                const GLint location = uniforms.prop(idx, PRP::Location);
                if (u.location == kAnyLocation)
                {
                    if (location < 0 || location >= max_locations_ || explicit_locations.contains(location))
                        bad.push_back(std::format("LOCATION = {} (expected a linker-assigned location that is not an explicit one)", location));
                }
                else if (location != u.location)
                    bad.push_back(std::format("LOCATION = {} (expected {})", location, u.location));
                const GLint block_index = u.block != nullptr ? static_cast<GLint>(blocks.index(u.block)) : -1;
                expect(PRP::BlockIndex, block_index, "BLOCK_INDEX");
                expect(PRP::Offset, u.offset, "OFFSET");
                expect(PRP::ArrayStride, u.array_stride, "ARRAY_STRIDE");
                expect(PRP::MatrixStride, u.matrix_stride, "MATRIX_STRIDE");
                expect(PRP::IsRowMajor, u.row_major, "IS_ROW_MAJOR");
                expect(PRP::AtomicCounterBufferIndex, -1, "ATOMIC_COUNTER_BUFFER_INDEX");
                rec.ok(bad.empty(), std::format("UNIFORM '{}': TYPE, ARRAY_SIZE {}, LOCATION, BLOCK_INDEX, OFFSET, ARRAY_STRIDE, MATRIX_STRIDE, IS_ROW_MAJOR as derived from the shader", u.name, u.array_size), join(bad));

                const bool vs = uniforms.prop(idx, PRP::ReferencedByVertexShader) != 0;
                const bool fs = uniforms.prop(idx, PRP::ReferencedByFragmentShader) != 0;
                const bool cs = uniforms.prop(idx, PRP::ReferencedByComputeShader) != 0;
                const bool refs = vs == u.vs && fs == u.fs && !cs;
                const std::string what = std::format("UNIFORM '{}': REFERENCED_BY_VERTEX/FRAGMENT = {}/{}", u.name, u.vs, u.fs);
                const std::string detail = refs ? "" : std::format("driver reports {}/{} (compute {})", vs, fs, cs);
                if (u.refs_deviate)
                    rec.deviation(refs, what, detail, "radeonsi", "see UNIFORM_BLOCK 'Params': members of a block declared only in the vertex shader are reported as referenced by the fragment shader");
                else
                    rec.ok(refs, what, detail);
            }

            // The "uniform location" of a block member, an atomic counter or a built-in is -1 (section 7.3.1.1).
            std::vector<std::string> all_locations;
            for (GLint i = 0; i < uniforms.count(); ++i)
                all_locations.push_back(std::format("{}={}", uniforms.name(static_cast<GLuint>(i)), uniforms.prop(static_cast<GLuint>(i), PRP::Location)));
            std::printf("   note: UNIFORM locations: %s\n", join(all_locations).c_str());

            rec.group("UNIFORM_BLOCK (gfx)");
            check_block_list(rec, blocks, uniforms, gfx_blocks(), "UNIFORM_BLOCK");
            // The ES 3.0 block queries agree with the new interface.
            rec.eq(blocks.count(), program_param(gfx_, mgl::ProgramParameter::ActiveUniformBlocks), "UNIFORM_BLOCK: ACTIVE_RESOURCES equals glGetProgramiv(ACTIVE_UNIFORM_BLOCKS)");
            std::vector<std::string> mismatch;
            for (const BlockExp& b : gfx_blocks())
            {
                const GLuint idx = blocks.index(b.name);
                GLint binding = -1, size = -1;
                mgl::glGetActiveUniformBlockiv(gfx_, idx, mgl::UniformBlockParameter::Binding, &binding);
                mgl::glGetActiveUniformBlockiv(gfx_, idx, mgl::UniformBlockParameter::DataSize, &size);
                if (binding != blocks.prop(idx, PRP::BufferBinding)) mismatch.push_back(std::string(b.name) + " binding");
                if (size != blocks.prop(idx, PRP::BufferDataSize)) mismatch.push_back(std::string(b.name) + " size");
            }
            rec.ok(mismatch.empty(), "UNIFORM_BLOCK: BUFFER_BINDING and BUFFER_DATA_SIZE equal glGetActiveUniformBlockiv(UNIFORM_BLOCK_BINDING / _DATA_SIZE)", join(mismatch));
        }

        void check_empty_interfaces(Recorder& rec)
        {
            rec.group("empty interfaces");
            // Interfaces with no active resource report zero, with no error (section 7.3.1).
            for (const PI iface : {PI::BufferVariable, PI::ShaderStorageBlock, PI::AtomicCounterBuffer})
            {
                const Query q{gfx_, iface};
                rec.eq(q.count(), 0, std::format("gfx has no {}: ACTIVE_RESOURCES = 0", mgl::to_string(iface)));
                if (iface != PI::AtomicCounterBuffer) rec.eq(q.param(PIP::MaxNameLength), 0, std::format("gfx has no {}: MAX_NAME_LENGTH = 0", mgl::to_string(iface)));
                if (iface != PI::BufferVariable) rec.eq(q.param(PIP::MaxNumActiveVariables), 0, std::format("gfx has no {}: MAX_NUM_ACTIVE_VARIABLES = 0", mgl::to_string(iface)));
            }
            const Query comp_blocks{comp_, PI::UniformBlock};
            rec.eq(comp_blocks.count(), 0, "comp has no uniform block: ACTIVE_RESOURCES = 0");
            rec.eq(comp_blocks.param(PIP::MaxNameLength), 0, "comp has no uniform block: MAX_NAME_LENGTH = 0");
            rec.eq(comp_blocks.param(PIP::MaxNumActiveVariables), 0, "comp has no uniform block: MAX_NUM_ACTIVE_VARIABLES = 0");
            rec.eq(Query{comp_, PI::TransformFeedbackVarying}.count(), 0, "comp has no transform feedback varyings: ACTIVE_RESOURCES = 0");
            rec.no_error("no GL error from the empty-list queries");
        }

        // ---------------------------------------------------------------------------------------------
        void check_io_list(Recorder& rec, const Query& q, const std::vector<IoExp>& table, const char* iface_name, GLint stage_vs, GLint stage_fs)
        {
            GLint longest = 0;
            for (const IoExp& e : table) longest = std::max(longest, static_cast<GLint>(std::strlen(e.name)) + 1);
            rec.eq(q.count(), static_cast<GLint>(table.size()), std::format("{}: ACTIVE_RESOURCES = {}", iface_name, table.size()));
            rec.eq(q.param(PIP::MaxNameLength), longest, std::format("{}: MAX_NAME_LENGTH = longest name + terminator = {}", iface_name, longest));
            std::set<GLint> explicit_locations;
            for (const IoExp& e : table)
                if (e.location >= 0) explicit_locations.insert(e.location);
            for (const IoExp& e : table)
            {
                const GLuint idx = q.index(e.name);
                if (!rec.ok(idx != GL_INVALID_INDEX, std::format("{} '{}': glGetProgramResourceIndex finds it", iface_name, e.name))) continue;
                std::vector<std::string> bad;
                if (q.name(idx) != e.name) bad.push_back("name round trip");
                if (q.prop(idx, PRP::NameLength) != static_cast<GLint>(std::strlen(e.name)) + 1) bad.push_back("NAME_LENGTH");
                if (q.prop(idx, PRP::Type) != type_token(e.type)) bad.push_back(std::format("TYPE {} != {}", q.prop(idx, PRP::Type), type_token(e.type)));
                if (q.prop(idx, PRP::ArraySize) != e.array_size) bad.push_back(std::format("ARRAY_SIZE {} != {}", q.prop(idx, PRP::ArraySize), e.array_size));
                const GLint loc = q.prop(idx, PRP::Location);
                if (e.location == kAnyLocation)
                {
                    if (loc < 0 || explicit_locations.contains(loc)) bad.push_back(std::format("LOCATION {} (expected a linker-assigned one, not {})", loc, join_set(explicit_locations)));
                }
                else if (loc != e.location)
                    bad.push_back(std::format("LOCATION {} != {}", loc, e.location));
                if (q.prop(idx, PRP::ReferencedByVertexShader) != stage_vs) bad.push_back("REFERENCED_BY_VERTEX_SHADER");
                if (q.prop(idx, PRP::ReferencedByFragmentShader) != stage_fs) bad.push_back("REFERENCED_BY_FRAGMENT_SHADER");
                if (q.prop(idx, PRP::ReferencedByComputeShader) != 0) bad.push_back("REFERENCED_BY_COMPUTE_SHADER");
                rec.ok(bad.empty(), std::format("{} '{}': NAME_LENGTH, TYPE, ARRAY_SIZE {}, LOCATION, REFERENCED_BY_* = {}/{}", iface_name, e.name, e.array_size, stage_vs, stage_fs), join(bad));
            }
        }

        static std::string join_set(const std::set<GLint>& s)
        {
            std::string r;
            for (const GLint v : s) r += std::format("{} ", v);
            return r;
        }

        void check_program_io(Recorder& rec)
        {
            rec.group("PROGRAM_INPUT / PROGRAM_OUTPUT (gfx)");
            const Query in{gfx_, PI::ProgramInput};
            const Query out{gfx_, PI::ProgramOutput};
            check_io_list(rec, in, gfx_inputs(), "PROGRAM_INPUT", 1, 0);
            check_io_list(rec, out, gfx_outputs(), "PROGRAM_OUTPUT", 0, 1);
            // Only the FIRST stage's inputs and the LAST stage's outputs are enumerated (section 7.3.1).
            rec.eq(in.index("v_color"), GL_INVALID_INDEX, "PROGRAM_INPUT does not list the fragment shader's input v_color (only the first stage's inputs)");
            rec.eq(out.index("v_color"), GL_INVALID_INDEX, "PROGRAM_OUTPUT does not list the vertex shader's output v_color (only the last stage's outputs)");
            rec.eq(out.index("gl_Position"), GL_INVALID_INDEX, "PROGRAM_OUTPUT does not list gl_Position of the vertex stage");
            rec.eq(in.index("o_color"), GL_INVALID_INDEX, "PROGRAM_INPUT does not list a fragment output");
            rec.no_error("no GL error from the lookups of unknown names (INVALID_INDEX is not an error)");
        }

        // ---------------------------------------------------------------------------------------------
        void check_transform_feedback(Recorder& rec)
        {
            rec.group("TRANSFORM_FEEDBACK_VARYING (gfx)");
            const Query q{gfx_, PI::TransformFeedbackVarying};
            const auto& table = gfx_tf();
            rec.eq(q.count(), static_cast<GLint>(table.size()), "TRANSFORM_FEEDBACK_VARYING: ACTIVE_RESOURCES = 3 (the number given to glTransformFeedbackVaryings)");
            rec.eq(q.count(), program_param(gfx_, mgl::ProgramParameter::TransformFeedbackVaryings), "...equals glGetProgramiv(TRANSFORM_FEEDBACK_VARYINGS)");
            GLint longest = 0;
            for (const IoExp& e : table) longest = std::max(longest, static_cast<GLint>(std::strlen(e.name)) + 1);
            rec.eq(q.param(PIP::MaxNameLength), longest, std::format("TRANSFORM_FEEDBACK_VARYING: MAX_NAME_LENGTH = {}", longest));
            // "the active resource list will use the variable order specified in the most recent call to TransformFeedbackVaryings".
            for (std::size_t i = 0; i < table.size(); ++i)
            {
                const IoExp& e = table[i];
                const auto idx = static_cast<GLuint>(i);
                std::vector<std::string> bad;
                if (q.name(idx) != e.name) bad.push_back(std::format("entry {} is '{}'", i, q.name(idx)));
                if (q.index(e.name) != idx) bad.push_back("glGetProgramResourceIndex");
                if (q.prop(idx, PRP::NameLength) != static_cast<GLint>(std::strlen(e.name)) + 1) bad.push_back("NAME_LENGTH");
                if (q.prop(idx, PRP::Type) != type_token(e.type)) bad.push_back("TYPE");
                if (q.prop(idx, PRP::ArraySize) != e.array_size) bad.push_back("ARRAY_SIZE");
                rec.ok(bad.empty(), std::format("TRANSFORM_FEEDBACK_VARYING entry {} is '{}' (the order passed to glTransformFeedbackVaryings), NAME_LENGTH, TYPE, ARRAY_SIZE {}", i, e.name, e.array_size), join(bad));

                // The ES 3.0 command reports the same varying.
                std::string buf(64, '\0');
                GLsizei len = 0, size = 0;
                mgl::UniformType type{};
                mgl::glGetTransformFeedbackVarying(gfx_, idx, static_cast<GLsizei>(buf.size()), &len, &size, &type, buf.data());
                buf.resize(static_cast<std::size_t>(len));
                rec.ok(buf == e.name && size == e.array_size && static_cast<GLint>(type) == type_token(e.type),
                       std::format("glGetTransformFeedbackVarying({}) reports the same name, size and type", i), std::format("{} size {} type {:#x}", buf, size, static_cast<unsigned>(type)));
            }
            rec.eq(q.index("v_uv"), GL_INVALID_INDEX, "a varying that is not captured ('v_uv') is not in the list");
            rec.no_error("no GL error from the transform feedback lookups");
        }

        // ---------------------------------------------------------------------------------------------
        void check_compute_interfaces(Recorder& rec)
        {
            // ---- SHADER_STORAGE_BLOCK and BUFFER_VARIABLE ------------------------------------------------------
            rec.group("SHADER_STORAGE_BLOCK / BUFFER_VARIABLE (comp)");
            const Query ssbo{comp_, PI::ShaderStorageBlock};
            const Query vars{comp_, PI::BufferVariable};
            check_block_list(rec, ssbo, vars, comp_ssbos(), "SHADER_STORAGE_BLOCK");

            const auto& table = comp_buffer_vars();
            GLint longest = 0;
            for (const BufferVarExp& v : table) longest = std::max(longest, static_cast<GLint>(std::strlen(v.name)) + 1);
            rec.eq(vars.count(), static_cast<GLint>(table.size()), std::format("BUFFER_VARIABLE: ACTIVE_RESOURCES = {}", table.size()));
            rec.eq(vars.param(PIP::MaxNameLength), longest, std::format("BUFFER_VARIABLE: MAX_NAME_LENGTH = longest name + terminator = {}", longest));
            for (const BufferVarExp& v : table)
            {
                const GLuint idx = vars.index(v.name);
                if (!rec.ok(idx != GL_INVALID_INDEX, std::format("BUFFER_VARIABLE '{}': glGetProgramResourceIndex finds it", v.name))) continue;
                std::vector<std::string> bad;
                if (vars.name(idx) != v.name) bad.push_back("name round trip");
                const auto expect = [&](PRP p, GLint want, const char* label) {
                    const GLint got = vars.prop(idx, p);
                    if (got != want) bad.push_back(std::format("{} = {} (expected {})", label, got, want));
                };
                expect(PRP::NameLength, static_cast<GLint>(std::strlen(v.name)) + 1, "NAME_LENGTH");
                expect(PRP::Type, type_token(v.type), "TYPE");
                expect(PRP::ArraySize, v.array_size, "ARRAY_SIZE");
                expect(PRP::BlockIndex, static_cast<GLint>(ssbo.index(v.block)), "BLOCK_INDEX");
                expect(PRP::Offset, v.offset, "OFFSET");
                expect(PRP::ArrayStride, v.array_stride, "ARRAY_STRIDE");
                expect(PRP::MatrixStride, v.matrix_stride, "MATRIX_STRIDE");
                expect(PRP::IsRowMajor, v.row_major, "IS_ROW_MAJOR");
                expect(PRP::TopLevelArraySize, v.top_size, "TOP_LEVEL_ARRAY_SIZE");
                expect(PRP::TopLevelArrayStride, v.top_stride, "TOP_LEVEL_ARRAY_STRIDE");
                expect(PRP::ReferencedByComputeShader, 1, "REFERENCED_BY_COMPUTE_SHADER");
                expect(PRP::ReferencedByVertexShader, 0, "REFERENCED_BY_VERTEX_SHADER");
                expect(PRP::ReferencedByFragmentShader, 0, "REFERENCED_BY_FRAGMENT_SHADER");
                rec.ok(bad.empty(), std::format("BUFFER_VARIABLE '{}': TYPE, ARRAY_SIZE {}, BLOCK_INDEX, OFFSET {}, strides, IS_ROW_MAJOR, TOP_LEVEL_ARRAY_*, REFERENCED_BY_COMPUTE", v.name, v.array_size, v.offset), join(bad));
            }

            // ---- UNIFORM of the compute program: a plain uniform and three atomic counters --------------------------
            rec.group("ATOMIC_COUNTER_BUFFER / UNIFORM (comp)");
            const Query uniforms{comp_, PI::Uniform};
            const Query acb{comp_, PI::AtomicCounterBuffer};
            rec.eq(uniforms.count(), 4, "comp UNIFORM: ACTIVE_RESOURCES = 4 (u_n and the atomic counters c_a, c_b, c_c)");
            rec.eq(uniforms.param(PIP::MaxNameLength), 4, "comp UNIFORM: MAX_NAME_LENGTH = 4 (\"c_a\" + terminator, \"u_n\" + terminator)");
            rec.eq(acb.count(), 2, "ATOMIC_COUNTER_BUFFER: ACTIVE_RESOURCES = 2 (one per binding point that an active counter uses: 1 and 3)");
            rec.eq(acb.count(), program_param(comp_, mgl::ProgramParameter::ActiveAtomicCounterBuffers), "...equals glGetProgramiv(ACTIVE_ATOMIC_COUNTER_BUFFERS)");
            rec.eq(acb.param(PIP::MaxNumActiveVariables), 2, "ATOMIC_COUNTER_BUFFER: MAX_NUM_ACTIVE_VARIABLES = 2 (c_a and c_b share binding 1)");

            // Which list entry is the buffer of binding N? The order is implementation-dependent, BUFFER_BINDING tells.
            std::map<GLint, GLuint> acb_by_binding;
            for (GLint i = 0; i < acb.count(); ++i) acb_by_binding[acb.prop(static_cast<GLuint>(i), PRP::BufferBinding)] = static_cast<GLuint>(i);
            rec.ok(acb_by_binding.contains(1) && acb_by_binding.contains(3), "ATOMIC_COUNTER_BUFFER: BUFFER_BINDING values are {1, 3}");

            struct CounterExp { const char* name; GLint binding; GLint offset; };
            const CounterExp counters[] = {{"c_a", 1, 0}, {"c_b", 1, 4}, {"c_c", 3, 8}};
            {
                const GLuint n = uniforms.index("u_n");
                std::vector<std::string> bad;
                if (uniforms.prop(n, PRP::Type) != type_token(GL_UNSIGNED_INT)) bad.push_back("TYPE");
                if (uniforms.prop(n, PRP::Location) != 2) bad.push_back("LOCATION");
                if (uniforms.prop(n, PRP::BlockIndex) != -1 || uniforms.prop(n, PRP::AtomicCounterBufferIndex) != -1) bad.push_back("BLOCK_INDEX / ATOMIC_COUNTER_BUFFER_INDEX");
                if (uniforms.prop(n, PRP::Offset) != -1) bad.push_back("OFFSET");
                if (uniforms.prop(n, PRP::ReferencedByComputeShader) != 1) bad.push_back("REFERENCED_BY_COMPUTE_SHADER");
                rec.ok(bad.empty(), "comp UNIFORM 'u_n': uint at layout(location = 2), no block, no counter buffer, referenced by the compute shader", join(bad));
            }
            for (const CounterExp& c : counters)
            {
                const GLuint idx = uniforms.index(c.name);
                std::vector<std::string> bad;
                if (idx == GL_INVALID_INDEX) bad.push_back("not found");
                else
                {
                    if (uniforms.prop(idx, PRP::Type) != type_token(GL_UNSIGNED_INT_ATOMIC_COUNTER)) bad.push_back("TYPE");
                    if (uniforms.prop(idx, PRP::ArraySize) != 1) bad.push_back("ARRAY_SIZE");
                    if (uniforms.prop(idx, PRP::Location) != -1) bad.push_back("LOCATION (atomic counters have none)");
                    if (uniforms.prop(idx, PRP::BlockIndex) != -1) bad.push_back("BLOCK_INDEX");
                    if (uniforms.prop(idx, PRP::Offset) != c.offset) bad.push_back(std::format("OFFSET {} != {}", uniforms.prop(idx, PRP::Offset), c.offset));
                    if (acb_by_binding.contains(c.binding) && uniforms.prop(idx, PRP::AtomicCounterBufferIndex) != static_cast<GLint>(acb_by_binding[c.binding]))
                        bad.push_back("ATOMIC_COUNTER_BUFFER_INDEX does not point at the buffer of binding " + std::to_string(c.binding));
                    if (uniforms.prop(idx, PRP::ReferencedByComputeShader) != 1) bad.push_back("REFERENCED_BY_COMPUTE_SHADER");
                }
                rec.ok(bad.empty(), std::format("comp UNIFORM '{}': UNSIGNED_INT_ATOMIC_COUNTER, LOCATION -1, OFFSET {} (layout(binding = {}, offset = {})), ATOMIC_COUNTER_BUFFER_INDEX", c.name, c.offset, c.binding, c.offset), join(bad));
            }
            struct BufferExp { GLint binding; std::vector<const char*> members; GLint min_size; };
            const BufferExp buffers[] = {{1, {"c_a", "c_b"}, 8}, {3, {"c_c"}, 12}};
            for (const BufferExp& b : buffers)
            {
                if (!acb_by_binding.contains(b.binding)) continue;
                const GLuint idx = acb_by_binding[b.binding];
                std::vector<std::string> bad;
                if (acb.prop(idx, PRP::NumActiveVariables) != static_cast<GLint>(b.members.size())) bad.push_back("NUM_ACTIVE_VARIABLES");
                std::set<std::string> got, want;
                for (const GLint v : acb.prop_array(idx, PRP::ActiveVariables, 8)) got.insert(uniforms.name(static_cast<GLuint>(v)));
                for (const char* m : b.members) want.insert(m);
                if (got != want) bad.push_back("ACTIVE_VARIABLES: " + join(std::vector<std::string>(got.begin(), got.end())));
                const GLint size = acb.prop(idx, PRP::BufferDataSize);
                if (size < b.min_size) bad.push_back(std::format("BUFFER_DATA_SIZE {} < {}", size, b.min_size));
                if (acb.prop(idx, PRP::ReferencedByComputeShader) != 1 || acb.prop(idx, PRP::ReferencedByVertexShader) != 0 || acb.prop(idx, PRP::ReferencedByFragmentShader) != 0)
                    bad.push_back("REFERENCED_BY_*");
                rec.ok(bad.empty(), std::format("ATOMIC_COUNTER_BUFFER (binding {}): NUM_ACTIVE_VARIABLES {}, ACTIVE_VARIABLES, BUFFER_DATA_SIZE >= {} (driver: {}), REFERENCED_BY_COMPUTE", b.binding, b.members.size(), b.min_size, size), join(bad));
            }
        }

        // ---------------------------------------------------------------------------------------------
        void check_names_and_locations(Recorder& rec)
        {
            rec.group("names and locations");
            const Query u{gfx_, PI::Uniform};
            // Array names: the list entry is "name[0]"; the base name matches it, other elements do not (section 7.3.1).
            rec.eq(u.index("u_weights"), u.index("u_weights[0]"), "glGetProgramResourceIndex(\"u_weights\") = index of \"u_weights[0]\" (\"[0]\" appended)");
            rec.eq(u.index("u_weights[1]"), GL_INVALID_INDEX, "glGetProgramResourceIndex(\"u_weights[1]\") = INVALID_INDEX (an element other than the first is not a list entry)");
            rec.eq(u.index("u_light"), GL_INVALID_INDEX, "a structure name (\"u_light\") is not a resource, only its members are");
            rec.eq(u.index("no_such_uniform"), GL_INVALID_INDEX, "an unknown name returns INVALID_INDEX");
            rec.no_error("INVALID_INDEX for an unknown name is returned without a GL error");
            const Query blocks{gfx_, PI::UniformBlock};
            rec.eq(blocks.index("Cell"), blocks.index("Cell[0]"), "UNIFORM_BLOCK: \"Cell\" matches the entry \"Cell[0]\"");
            rec.ok(blocks.index("Cell[1]") != GL_INVALID_INDEX && blocks.index("Cell[1]") != blocks.index("Cell[0]"), "UNIFORM_BLOCK: an array of blocks has one entry per element: \"Cell[1]\" is its own entry");
            rec.eq(blocks.index("Cell[2]"), GL_INVALID_INDEX, "UNIFORM_BLOCK: \"Cell[2]\" does not exist");
            // Members of an array of blocks are listed once, named after the block (not "Cell[0].color"), and belong to its FIRST element.
            const Query uniforms{gfx_, PI::Uniform};
            rec.eq(uniforms.prop(uniforms.index("Cell.color"), PRP::BlockIndex), static_cast<GLint>(blocks.index("Cell[0]")),
                   "BLOCK_INDEX of a member of a block array is the index of the FIRST block of the array (\"Cell.color\" -> Cell[0])");
            const Query out{gfx_, PI::ProgramOutput};
            rec.eq(out.index("o_id"), out.index("o_id[0]"), "PROGRAM_OUTPUT: \"o_id\" matches the entry \"o_id[0]\"");
            rec.eq(out.index("o_id[1]"), GL_INVALID_INDEX, "PROGRAM_OUTPUT: \"o_id[1]\" is not a list entry");
            const Query ssbo_vars{comp_, PI::BufferVariable};
            rec.eq(ssbo_vars.index("Source.w"), ssbo_vars.index("Source.w[0]"), "BUFFER_VARIABLE: \"Source.w\" matches \"Source.w[0]\"");
            rec.eq(ssbo_vars.index("Result.items[1].p"), GL_INVALID_INDEX, "BUFFER_VARIABLE: only element 0 of a top-level array exists (\"Result.items[1].p\")");
            rec.ok(ssbo_vars.index("plain_value") != GL_INVALID_INDEX, "BUFFER_VARIABLE: members of a block without an instance name are listed without the block name (\"plain_value\")");
            rec.eq(ssbo_vars.index("Plain.plain_value"), GL_INVALID_INDEX, "...so \"Plain.plain_value\" does not exist");

            // glGetProgramResourceName: truncation and the returned length (section 7.3.1).
            {
                const GLuint idx = u.index("Params.scale");
                std::array<char, 8> small{};
                small.fill('#');
                GLsizei len = -1;
                mgl::glGetProgramResourceName(gfx_, PI::Uniform, idx, 5, &len, small.data());
                rec.ok(len == 4 && std::string_view(small.data()) == "Para" && small[5] == '#',
                       "glGetProgramResourceName with bufSize 5: the first 4 characters and a terminator are written, length = 4, nothing beyond",
                       std::format("len {} text '{}'", len, small.data()));
                small.fill('#');
                mgl::glGetProgramResourceName(gfx_, PI::Uniform, idx, 0, &len, small.data());
                rec.ok(small[0] == '#', "glGetProgramResourceName with bufSize 0 writes nothing");
                rec.no_error("no GL error from bufSize 0");
            }
            // Several properties in one call come back in order, and `count` limits how many values are written.
            {
                const GLuint idx = u.index("Params.skew");
                const PRP props[] = {PRP::Type, PRP::Offset, PRP::MatrixStride, PRP::IsRowMajor};
                std::array<GLint, 8> vals{};
                vals.fill(-777);
                GLsizei len = 0;
                mgl::glGetProgramResourceiv(gfx_, PI::Uniform, idx, std::span<const PRP>(props), 8, &len, vals.data());
                rec.ok(len == 4 && vals[0] == type_token(GL_FLOAT_MAT2x3) && vals[1] == 64 && vals[2] == 16 && vals[3] == 1 && vals[4] == -777,
                       "glGetProgramResourceiv with 4 properties: values written in the order of props, length = 4, nothing beyond",
                       std::format("len {} values {} {} {} {} {}", len, vals[0], vals[1], vals[2], vals[3], vals[4]));
                vals.fill(-777);
                mgl::glGetProgramResourceiv(gfx_, PI::Uniform, idx, std::span<const PRP>(props), 2, &len, vals.data());
                rec.ok(len == 2 && vals[0] == type_token(GL_FLOAT_MAT2x3) && vals[1] == 64 && vals[2] == -777, "with count = 2 only the first two values are written, length = 2",
                       std::format("len {} values {} {} {}", len, vals[0], vals[1], vals[2]));
                const GLuint params = blocks.index("Params");
                const PRP active = PRP::ActiveVariables;
                std::array<GLint, 8> av{};
                av.fill(-777);
                mgl::glGetProgramResourceiv(gfx_, PI::UniformBlock, params, std::span<const PRP>(&active, 1), 3, &len, av.data());
                const bool truncated = len == 3 && av[3] == -777;
                rec.deviation(truncated, "ACTIVE_VARIABLES of block Params (5 entries) with count = 3: only the first three indices are written, length = 3",
                              std::format("len {} values {} {} {} {} {}", len, av[0], av[1], av[2], av[3], av[4]), "Mesa",
                              "glGetProgramResourceiv(ACTIVE_VARIABLES) ignores `count`: it writes all 5 indices and reports length 5, although \"only the first count integer values will be written to params\"");
                rec.no_error("no GL error from the truncating queries");
            }

            // ---- glGetProgramResourceLocation -----------------------------------------------------------------
            struct Loc { PI iface; const char* name; GLint expected; const char* what; };
            const Loc locations[] = {
                {PI::Uniform, "u_tint", 4, "UNIFORM u_tint -> explicit location 4"},
                {PI::Uniform, "u_weights", 8, "UNIFORM \"u_weights\" (base name of an array) -> the location of element 0, 8"},
                {PI::Uniform, "u_weights[0]", 8, "UNIFORM u_weights[0] -> 8"},
                {PI::Uniform, "u_weights[1]", 9, "UNIFORM u_weights[1] -> 9: array elements take consecutive explicit locations"},
                {PI::Uniform, "u_weights[2]", 10, "UNIFORM u_weights[2] -> 10"},
                {PI::Uniform, "u_weights[3]", -1, "UNIFORM u_weights[3] -> -1: the index is not less than the number of active elements"},
                {PI::Uniform, "u_light.dir", 20, "UNIFORM u_light.dir -> 20: the first inner-most member of a structure takes the declared location"},
                {PI::Uniform, "u_light.power", 21, "UNIFORM u_light.power -> 21: the next member takes the next location"},
                {PI::Uniform, "u_texarr[1]", 31, "UNIFORM u_texarr[1] -> 31 (layout(location = 30) on the array)"},
                {PI::Uniform, "Params.scale", -1, "UNIFORM Params.scale -> -1: members of a uniform block have no location"},
                {PI::Uniform, "no_such_uniform", -1, "UNIFORM with an unknown name -> -1"},
                {PI::ProgramInput, "a_position", 0, "PROGRAM_INPUT a_position -> explicit location 0"},
                {PI::ProgramInput, "a_color", 3, "PROGRAM_INPUT a_color -> explicit location 3"},
                {PI::ProgramInput, "gl_VertexID", -1, "PROGRAM_INPUT gl_VertexID -> -1: built-in inputs have no location"},
                {PI::ProgramOutput, "o_color", 0, "PROGRAM_OUTPUT o_color -> 0"},
                {PI::ProgramOutput, "o_id", 1, "PROGRAM_OUTPUT \"o_id\" -> 1 (first element)"},
                {PI::ProgramOutput, "o_id[1]", 2, "PROGRAM_OUTPUT o_id[1] -> 2: array elements of an output take consecutive locations"},
            };
            for (const Loc& l : locations)
                rec.eq(Query{gfx_, l.iface}.location(l.name), l.expected, l.what);
            rec.eq(Query{comp_, PI::Uniform}.location("u_n"), 2, "comp UNIFORM u_n -> explicit location 2");
            rec.eq(Query{comp_, PI::Uniform}.location("c_b"), -1, "comp UNIFORM c_b (an atomic counter) -> -1");
            rec.no_error("no GL error from glGetProgramResourceLocation on valid interfaces");

            // Cross-checks against the ES 2.0/3.0 commands: same numbers.
            std::vector<std::string> bad;
            for (const char* n : {"u_tint", "u_weights[1]", "u_light.power", "u_texarr[1]", "u_tex", "u_pick"})
                if (mgl::glGetUniformLocation(gfx_, n).value != u.location(n)) bad.push_back(n);
            rec.ok(bad.empty(), "glGetUniformLocation returns the same locations as glGetProgramResourceLocation(UNIFORM)", join(bad));
            rec.eq(static_cast<GLint>(mgl::glGetAttribLocation(gfx_, "a_cell").value), Query{gfx_, PI::ProgramInput}.location("a_cell"), "glGetAttribLocation(a_cell) = glGetProgramResourceLocation(PROGRAM_INPUT, a_cell) (a linker-assigned location)");
            rec.eq(mgl::glGetFragDataLocation(gfx_, "o_id"), 1, "glGetFragDataLocation(o_id) = 1 = glGetProgramResourceLocation(PROGRAM_OUTPUT, o_id)");
            rec.no_error("no GL error from the ES 2.0/3.0 location queries");
        }

        // ---------------------------------------------------------------------------------------------
        void check_bindings(Recorder& rec)
        {
            rec.group("layout(binding)");
            // Sampler uniforms start at the unit named by layout(binding); the application never calls glUniform1i.
            const auto sampler = [&](const char* name) {
                GLint v = -1;
                mgl::glGetUniformiv(gfx_, mgl::glGetUniformLocation(gfx_, name), &v);
                return v;
            };
            rec.eq(sampler("u_tex"), 2, "layout(binding = 2) sampler2D u_tex: glGetUniformiv = 2");
            rec.eq(sampler("u_texarr[0]"), 4, "layout(binding = 4) sampler2D u_texarr[2]: element 0 reads back 4");
            rec.eq(sampler("u_texarr[1]"), 5, "...and element 1 the next unit, 5");
            // Uniform block bindings: BUFFER_BINDING of the interface, and the ES 3.0 glGetActiveUniformBlockiv.
            const Query blocks{gfx_, PI::UniformBlock};
            rec.eq(blocks.prop(blocks.index("Params"), PRP::BufferBinding), 3, "layout(std140, binding = 3) uniform Params: BUFFER_BINDING = 3");
            rec.eq(blocks.prop(blocks.index("Cell[0]"), PRP::BufferBinding), 6, "layout(binding = 6) uniform Cell cells[2]: element 0 BUFFER_BINDING = 6");
            rec.eq(blocks.prop(blocks.index("Cell[1]"), PRP::BufferBinding), 7, "...element 1 takes the next binding point, 7");
            // Shader storage blocks: the binding comes from the layout qualifier only; no qualifier = binding point 0.
            const Query ssbo{comp_, PI::ShaderStorageBlock};
            rec.eq(ssbo.prop(ssbo.index("Source"), PRP::BufferBinding), 1, "layout(std430, binding = 1) buffer Source: BUFFER_BINDING = 1");
            rec.eq(ssbo.prop(ssbo.index("Result"), PRP::BufferBinding), 5, "layout(std430, binding = 5) buffer Result: BUFFER_BINDING = 5");
            rec.eq(ssbo.prop(ssbo.index("Plain"), PRP::BufferBinding), 0, "a buffer block without a binding qualifier: BUFFER_BINDING = 0");
            // The glUniformBlockBinding state is shared with the new interface: changing it changes BUFFER_BINDING.
            const GLuint cell1 = blocks.index("Cell[1]");
            mgl::glUniformBlockBinding(gfx_, cell1, 9);
            rec.eq(blocks.prop(cell1, PRP::BufferBinding), 9, "glUniformBlockBinding(Cell[1], 9) is visible as BUFFER_BINDING = 9 (uniform block bindings can be changed after linking)");
            mgl::glUniformBlockBinding(gfx_, cell1, 7);
            rec.eq(blocks.prop(cell1, PRP::BufferBinding), 7, "...and back to 7");
            rec.no_error("no GL error from the binding queries");
        }

        // ---------------------------------------------------------------------------------------------
        void check_explicit_locations(Recorder& rec, const demo::Env& /*env*/)
        {
            rec.group("explicit uniform locations");
            rec.ok(max_locations_ >= 1024, std::format("MAX_UNIFORM_LOCATIONS = {} (ES 3.2 table 21.43 minimum: 1024)", max_locations_));
            const Query u{gfx_, PI::Uniform};
            bool in_range = true;
            for (GLint i = 0; i < u.count(); ++i)
            {
                const GLint loc = u.prop(static_cast<GLuint>(i), PRP::Location);
                if (loc >= max_locations_) in_range = false;
            }
            rec.ok(in_range, "every location reported for gfx (explicit and generated) is below MAX_UNIFORM_LOCATIONS");
            rec.eq(Query{edge_, PI::Uniform}.location("u_edge"), max_locations_ - 1, std::format("layout(location = MAX_UNIFORM_LOCATIONS - 1) uniform: LOCATION = {}", max_locations_ - 1));
            // Generated locations never take an explicit one (ES 3.2 section 7.6).
            std::set<GLint> explicit_locations = {4, 8, 9, 10, 20, 21, 30, 31};
            rec.ok(!explicit_locations.contains(u.location("u_tex")) && !explicit_locations.contains(u.location("u_pick")),
                   "locations generated by the linker (u_tex, u_pick) avoid every explicit location (4, 8-10, 20, 21, 30, 31)",
                   std::format("u_tex = {}, u_pick = {}", u.location("u_tex"), u.location("u_pick")));
            mgl::glUseProgram(edge_);
            mgl::glUniform1f(mgl::UniformLocation{max_locations_}, 1.0f);
            rec.error(EC::InvalidOperation, "glUniform1f at location MAX_UNIFORM_LOCATIONS (no uniform has it) -> INVALID_OPERATION");
            mgl::glUseProgram(mgl::ProgramId{0});

            // The explicit-location rules of GLSL ES 3.10 section 4.4.3, as rejected programs.
            const std::string beyond = with_define(src_edge_, "LOC", max_locations_);
            const std::string beyond_by = try_build(beyond);
            rec.ok(!beyond_by.empty(), std::format("layout(location = MAX_UNIFORM_LOCATIONS = {}) is out of range: the program is rejected (at {})", max_locations_, beyond_by.empty() ? "nothing: it linked" : beyond_by));
            const std::string dup_by = try_build(src_dup_);
            rec.ok(!dup_by.empty(), std::format("two uniforms with the same explicit location: the program is rejected (at {}), even though neither is used", dup_by.empty() ? "nothing: it linked" : dup_by));
            const std::string overlap_by = try_build(src_overlap_);
            rec.ok(!overlap_by.empty(), std::format("an array's consecutive locations (1, 2, 3) overlapping another uniform's location (3): the program is rejected (at {})", overlap_by.empty() ? "nothing: it linked" : overlap_by));
            const std::string fine_by = try_build(with_define(src_edge_, "LOC", max_locations_ - 1));
            rec.ok(fine_by.empty(), "control: the same program with layout(location = MAX_UNIFORM_LOCATIONS - 1) is accepted");
            rec.no_error("no GL error while building the rejected programs");
        }

        // ---------------------------------------------------------------------------------------------
        void check_errors(Recorder& rec)
        {
            rec.group("errors");
            const mgl::ProgramId unknown{0x7ff00001u};
            const mgl::ShaderId shader = mgl::glCreateShader(mgl::ShaderType::Vertex);
            const mgl::ProgramId shader_name{shader.value};
            const mgl::ProgramId never_linked = mgl::glCreateProgram();
            GLint v = 0;

            // ---- programs and interfaces ------------------------------------------------------------------------
            mgl::glGetProgramInterfaceiv(unknown, PI::Uniform, PIP::ActiveResources, &v);
            rec.error(EC::InvalidValue, "glGetProgramInterfaceiv with a name that is no program -> INVALID_VALUE");
            mgl::glGetProgramInterfaceiv(shader_name, PI::Uniform, PIP::ActiveResources, &v);
            rec.error(EC::InvalidOperation, "glGetProgramInterfaceiv with the name of a shader object -> INVALID_OPERATION");
            (void)mgl::glGetProgramResourceIndex(unknown, PI::Uniform, "x");
            rec.error(EC::InvalidValue, "glGetProgramResourceIndex with an unknown program -> INVALID_VALUE");
            (void)mgl::glGetProgramResourceIndex(shader_name, PI::Uniform, "x");
            rec.error(EC::InvalidOperation, "glGetProgramResourceIndex with a shader object -> INVALID_OPERATION");
            (void)mgl::glGetProgramResourceLocation(unknown, PI::Uniform, "x");
            rec.error(EC::InvalidValue, "glGetProgramResourceLocation with an unknown program -> INVALID_VALUE");

            // ---- GetProgramInterfaceiv ------------------------------------------------------------------------------
            mgl::glGetProgramInterfaceiv(comp_, PI::AtomicCounterBuffer, PIP::MaxNameLength, &v);
            rec.error(EC::InvalidOperation, "MAX_NAME_LENGTH on ATOMIC_COUNTER_BUFFER -> INVALID_OPERATION (atomic counters have no names)");
            for (const PI iface : {PI::Uniform, PI::ProgramInput, PI::ProgramOutput, PI::BufferVariable, PI::TransformFeedbackVarying})
            {
                mgl::glGetProgramInterfaceiv(gfx_, iface, PIP::MaxNumActiveVariables, &v);
                rec.error(EC::InvalidOperation, std::format("MAX_NUM_ACTIVE_VARIABLES on {} -> INVALID_OPERATION (only blocks and atomic counter buffers have active variables)", mgl::to_string(iface)));
            }

            // ---- names and indices ------------------------------------------------------------------------------------
            char name[16];
            GLsizei len = 0;
            (void)mgl::glGetProgramResourceIndex(comp_, PI::AtomicCounterBuffer, "c_a");
            rec.error(EC::InvalidEnum, "glGetProgramResourceIndex on ATOMIC_COUNTER_BUFFER -> INVALID_ENUM (no name strings)");
            mgl::glGetProgramResourceName(comp_, PI::AtomicCounterBuffer, 0, 16, &len, name);
            rec.error(EC::InvalidEnum, "glGetProgramResourceName on ATOMIC_COUNTER_BUFFER -> INVALID_ENUM");
            mgl::glGetProgramResourceName(gfx_, PI::Uniform, static_cast<GLuint>(Query{gfx_, PI::Uniform}.count()), 16, &len, name);
            rec.error(EC::InvalidValue, "glGetProgramResourceName with index == ACTIVE_RESOURCES -> INVALID_VALUE");
            mgl::glGetProgramResourceName(gfx_, PI::Uniform, 0, -1, &len, name);
            rec.error(EC::InvalidValue, "glGetProgramResourceName with a negative bufSize -> INVALID_VALUE");

            // ---- GetProgramResourceiv: properties that Table 7.2 does not allow for the interface ---------------------------
            struct Bad { PI iface; PRP prop; const char* what; };
            const Bad bad[] = {
                {PI::ProgramInput, PRP::Offset, "OFFSET on PROGRAM_INPUT"},
                {PI::ProgramInput, PRP::BlockIndex, "BLOCK_INDEX on PROGRAM_INPUT"},
                {PI::ProgramOutput, PRP::ArrayStride, "ARRAY_STRIDE on PROGRAM_OUTPUT"},
                {PI::Uniform, PRP::TopLevelArraySize, "TOP_LEVEL_ARRAY_SIZE on UNIFORM (buffer variables only)"},
                {PI::Uniform, PRP::ActiveVariables, "ACTIVE_VARIABLES on UNIFORM (blocks and counter buffers only)"},
                {PI::Uniform, PRP::BufferBinding, "BUFFER_BINDING on UNIFORM"},
                {PI::UniformBlock, PRP::Type, "TYPE on UNIFORM_BLOCK"},
                {PI::UniformBlock, PRP::Location, "LOCATION on UNIFORM_BLOCK"},
                {PI::UniformBlock, PRP::Offset, "OFFSET on UNIFORM_BLOCK"},
                {PI::TransformFeedbackVarying, PRP::ReferencedByVertexShader, "REFERENCED_BY_VERTEX_SHADER on TRANSFORM_FEEDBACK_VARYING"},
                {PI::TransformFeedbackVarying, PRP::Location, "LOCATION on TRANSFORM_FEEDBACK_VARYING"},
            };
            for (const Bad& b : bad)
            {
                GLsizei n = 0;
                GLint out = 0;
                mgl::glGetProgramResourceiv(gfx_, b.iface, 0, std::span<const PRP>(&b.prop, 1), 1, &n, &out);
                rec.error(EC::InvalidOperation, std::format("glGetProgramResourceiv: {} -> INVALID_OPERATION (not allowed by table 7.2)", b.what));
            }
            struct BadComp { PI iface; PRP prop; const char* what; };
            const BadComp bad_comp[] = {
                {PI::BufferVariable, PRP::Location, "LOCATION on BUFFER_VARIABLE"},
                {PI::BufferVariable, PRP::AtomicCounterBufferIndex, "ATOMIC_COUNTER_BUFFER_INDEX on BUFFER_VARIABLE"},
                {PI::ShaderStorageBlock, PRP::ArraySize, "ARRAY_SIZE on SHADER_STORAGE_BLOCK"},
                {PI::ShaderStorageBlock, PRP::Type, "TYPE on SHADER_STORAGE_BLOCK"},
                {PI::AtomicCounterBuffer, PRP::NameLength, "NAME_LENGTH on ATOMIC_COUNTER_BUFFER"},
                {PI::AtomicCounterBuffer, PRP::Type, "TYPE on ATOMIC_COUNTER_BUFFER"},
            };
            for (const BadComp& b : bad_comp)
            {
                GLsizei n = 0;
                GLint out = 0;
                mgl::glGetProgramResourceiv(comp_, b.iface, 0, std::span<const PRP>(&b.prop, 1), 1, &n, &out);
                rec.error(EC::InvalidOperation, std::format("glGetProgramResourceiv: {} -> INVALID_OPERATION (not allowed by table 7.2)", b.what));
            }
            {
                // One bad property among good ones still fails the whole call.
                const PRP mixed[] = {PRP::Type, PRP::Offset};
                GLsizei n = 0;
                GLint out[2] = {};
                mgl::glGetProgramResourceiv(gfx_, PI::ProgramInput, 0, std::span<const PRP>(mixed), 2, &n, out);
                rec.error(EC::InvalidOperation, "glGetProgramResourceiv: TYPE (valid) together with OFFSET (not valid for PROGRAM_INPUT) -> INVALID_OPERATION");
                mgl::glGetProgramResourceiv(gfx_, PI::Uniform, 0, 0, mixed, 1, &n, out);
                rec.error(EC::InvalidValue, "glGetProgramResourceiv with propCount = 0 -> INVALID_VALUE");
                mgl::glGetProgramResourceiv(gfx_, PI::Uniform, 0, 1, mixed, -1, &n, out);
                rec.error(EC::InvalidValue, "glGetProgramResourceiv with a negative count -> INVALID_VALUE");
                mgl::glGetProgramResourceiv(unknown, PI::Uniform, 0, 1, mixed, 1, &n, out);
                rec.error(EC::InvalidValue, "glGetProgramResourceiv with an unknown program -> INVALID_VALUE");
            }

            // ---- GetProgramResourceLocation: only UNIFORM, PROGRAM_INPUT, PROGRAM_OUTPUT ----------------------------------
            for (const PI iface : {PI::UniformBlock, PI::ShaderStorageBlock, PI::BufferVariable, PI::AtomicCounterBuffer, PI::TransformFeedbackVarying})
            {
                const GLint loc = mgl::glGetProgramResourceLocation(gfx_, iface, "Params");
                rec.error(EC::InvalidEnum, std::format("glGetProgramResourceLocation on {} -> INVALID_ENUM", mgl::to_string(iface)));
                rec.eq(loc, -1, "...and it returns -1");
            }

            // ---- programs that were never (successfully) linked ---------------------------------------------------------
            rec.eq(Query{never_linked, PI::Uniform}.count(), 0, "a program that was never linked: ACTIVE_RESOURCES = 0, no error (\"the program object is considered to have no active variables\")");
            rec.no_error("...and no error was generated");
            const GLint loc_never = mgl::glGetProgramResourceLocation(never_linked, PI::Uniform, "u");
            rec.error(EC::InvalidOperation, "glGetProgramResourceLocation on a program that has not been linked successfully -> INVALID_OPERATION");
            rec.eq(loc_never, -1, "...and it returns -1");
            mgl::glLinkProgram(never_linked);  // nothing attached: "No shader objects are attached to program" -> link fails
            rec.eq(program_param(never_linked, mgl::ProgramParameter::LinkStatus), GL_FALSE, "linking a program with no shaders fails (the premise)");
            rec.eq(Query{never_linked, PI::Uniform}.count(), 0, "a program whose link failed (nothing to record): ACTIVE_RESOURCES = 0");
            (void)mgl::glGetProgramResourceLocation(never_linked, PI::ProgramInput, "a");
            rec.error(EC::InvalidOperation, "glGetProgramResourceLocation on a program whose link failed -> INVALID_OPERATION");

            mgl::glDeleteProgram(never_linked);
            mgl::glDeleteShader(shader);
        }

        // ---------------------------------------------------------------------------------------------
        // The compute program, run with buffers that are laid out from the QUERIED offsets and strides.
        void check_compute_run(Recorder& rec)
        {
            rec.group("compute run with queried layout");
            const Query vars{comp_, PI::BufferVariable};
            const Query ssbo{comp_, PI::ShaderStorageBlock};
            const Query acb{comp_, PI::AtomicCounterBuffer};
            const Query uniforms{comp_, PI::Uniform};
            const auto offset = [&](const char* n) { return vars.prop(vars.index(n), PRP::Offset); };
            const auto size_of = [&](const char* block) { return static_cast<std::size_t>(ssbo.prop(ssbo.index(block), PRP::BufferDataSize)); };

            // ---- pack Source from the queried numbers ------------------------------------------------------------
            const GLint head = offset("Source.head"), m = offset("Source.m"), w = offset("Source.w[0]"), rm = offset("Source.rm");
            const GLint m_stride = vars.prop(vars.index("Source.m"), PRP::MatrixStride);
            const GLint w_stride = vars.prop(vars.index("Source.w[0]"), PRP::ArrayStride);
            const GLint rm_stride = vars.prop(vars.index("Source.rm"), PRP::MatrixStride);
            std::vector<std::uint8_t> source(size_of("Source"), 0);
            const auto put = [&](std::vector<std::uint8_t>& image, GLint at, float value) {
                if (at < 0 || static_cast<std::size_t>(at) + 4 > image.size()) return;
                std::memcpy(image.data() + at, &value, sizeof value);
            };
            const std::array<float, 4> head_v = {1, 2, 3, 4};
            for (int i = 0; i < 4; ++i) put(source, head + 4 * i, head_v[static_cast<std::size_t>(i)]);
            for (int c = 0; c < 3; ++c)           // mat3, column-major: element (row r, column c) at m + c * stride + r * 4
                for (int r = 0; r < 3; ++r) put(source, m + c * m_stride + r * 4, static_cast<float>(10 + 3 * c + r));
            for (int i = 0; i < 3; ++i) put(source, w + i * w_stride, static_cast<float>(20 + i));
            for (int r = 0; r < 3; ++r)           // row_major mat2x3: ROWS are `stride` apart, element (row r, column c) at rm + r * stride + c * 4
                for (int c = 0; c < 2; ++c) put(source, rm + r * rm_stride + c * 4, static_cast<float>(28 + 2 * r + c));
            // CPU model of query.comp: s = head.x + head.w + m[2][1] + m[0][0] + w[2] + w[1] + rm[1][2]
            //   m[c][r] = 10 + 3c + r -> m[2][1] = 17, m[0][0] = 10; rm[c][r] = 28 + 2r + c -> rm[1][2] = 33
            const float s = 1.0f + 4.0f + 17.0f + 10.0f + 22.0f + 21.0f + 33.0f;
            const GLuint n = 7;
            const GLuint expected_count = n + static_cast<GLuint>(s);

            std::array<mgl::BufferId, 5> bufs{};
            mgl::glGenBuffers(5, bufs.data());
            const auto fill = [&](mgl::BufferTarget target, mgl::BufferId b, std::size_t bytes, const void* data) {
                mgl::glBindBuffer(target, b);
                std::vector<std::uint8_t> zeros(bytes, 0);
                mgl::glBufferData(target, static_cast<GLsizeiptr>(bytes), data != nullptr ? data : zeros.data(), mgl::BufferUsage::DynamicCopy);
            };
            const std::size_t result_bytes = std::max<std::size_t>(size_of("Result"), 64);
            const std::size_t plain_bytes = size_of("Plain");
            const GLuint acb1 = acb_index(acb, 1), acb3 = acb_index(acb, 3);
            const std::size_t counters1_bytes = static_cast<std::size_t>(acb.prop(acb1, PRP::BufferDataSize));
            const std::size_t counters3_bytes = static_cast<std::size_t>(acb.prop(acb3, PRP::BufferDataSize));
            fill(mgl::BufferTarget::ShaderStorage, bufs[0], source.size(), source.data());
            fill(mgl::BufferTarget::ShaderStorage, bufs[1], result_bytes, nullptr);
            fill(mgl::BufferTarget::ShaderStorage, bufs[2], plain_bytes, nullptr);
            fill(mgl::BufferTarget::AtomicCounter, bufs[3], counters1_bytes, nullptr);
            fill(mgl::BufferTarget::AtomicCounter, bufs[4], counters3_bytes, nullptr);
            // The binding points come from the shader's layout qualifiers, as reported by BUFFER_BINDING.
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, static_cast<GLuint>(ssbo.prop(ssbo.index("Source"), PRP::BufferBinding)), bufs[0]);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, static_cast<GLuint>(ssbo.prop(ssbo.index("Result"), PRP::BufferBinding)), bufs[1]);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, static_cast<GLuint>(ssbo.prop(ssbo.index("Plain"), PRP::BufferBinding)), bufs[2]);
            mgl::glBindBufferBase(mgl::BufferTarget::AtomicCounter, static_cast<GLuint>(acb.prop(acb1, PRP::BufferBinding)), bufs[3]);
            mgl::glBindBufferBase(mgl::BufferTarget::AtomicCounter, static_cast<GLuint>(acb.prop(acb3, PRP::BufferBinding)), bufs[4]);

            mgl::glUseProgram(comp_);
            mgl::glUniform1ui(mgl::UniformLocation{uniforms.location("u_n")}, n);
            mgl::glDispatchCompute(5, 1, 1);  // five invocations, all writing the same values; the counters count them
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            mgl::glUseProgram(mgl::ProgramId{0});
            rec.no_error("the compute dispatch raised no error");

            const auto read_u32 = [&](mgl::BufferTarget target, mgl::BufferId b, std::size_t bytes, GLint at) {
                mgl::glBindBuffer(target, b);
                const auto* p = static_cast<const std::uint8_t*>(mgl::glMapBufferRange(target, 0, static_cast<GLsizeiptr>(bytes), mgl::MapBufferAccessMask::Read));
                std::uint32_t value = 0xDEADBEEFu;
                if (p != nullptr && at >= 0 && static_cast<std::size_t>(at) + 4 <= bytes) std::memcpy(&value, p + at, sizeof value);
                if (p != nullptr) (void)mgl::glUnmapBuffer(target);
                return value;
            };
            const auto bits = [](std::uint32_t u) { float f; std::memcpy(&f, &u, sizeof f); return f; };
            const std::uint32_t count = read_u32(mgl::BufferTarget::ShaderStorage, bufs[1], result_bytes, offset("Result.count"));
            rec.eq(count, expected_count, std::format("Result.count read at the queried OFFSET {} = u_n + head.x + head.w + m[2][1] + m[0][0] + w[2] + w[1] + rm[1][2] = {}", offset("Result.count"), expected_count));
            const float px = bits(read_u32(mgl::BufferTarget::ShaderStorage, bufs[1], result_bytes, offset("Result.items[0].p")));
            const float py = bits(read_u32(mgl::BufferTarget::ShaderStorage, bufs[1], result_bytes, offset("Result.items[0].p") + 4));
            const float ps = bits(read_u32(mgl::BufferTarget::ShaderStorage, bufs[1], result_bytes, offset("Result.items[0].s")));
            rec.ok(px == 10.0f && py == 21.0f && ps == 4.0f, "Result.items[0].p / .s at the queried offsets = (m[0][0], w[1]) = (10, 21) and head.w = 4",
                   std::format("p = ({}, {}), s = {}", px, py, ps));
            rec.eq(read_u32(mgl::BufferTarget::ShaderStorage, bufs[2], plain_bytes, offset("plain_value")), n, "plain_value (a block without instance name, binding 0 by default) = u_n = 7");
            const auto counter_offset = [&](const char* c) { return uniforms.prop(uniforms.index(c), PRP::Offset); };
            rec.eq(read_u32(mgl::BufferTarget::AtomicCounter, bufs[3], counters1_bytes, counter_offset("c_a")), 5u, "atomic counter c_a (binding 1, queried offset 0) was incremented once by each of the 5 invocations = 5");
            rec.eq(read_u32(mgl::BufferTarget::AtomicCounter, bufs[3], counters1_bytes, counter_offset("c_b")), 10u, "atomic counter c_b (binding 1, queried offset 4) twice per invocation = 10");
            rec.eq(read_u32(mgl::BufferTarget::AtomicCounter, bufs[4], counters3_bytes, counter_offset("c_c")), 15u, "atomic counter c_c (binding 3, queried offset 8) three times per invocation = 15");

            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::AtomicCounter, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 5, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::AtomicCounter, 1, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::AtomicCounter, 3, mgl::BufferId{0});
            mgl::glDeleteBuffers(5, bufs.data());
            rec.no_error("no GL error from the buffer reads");
        }

        static GLuint acb_index(const Query& acb, GLint binding)
        {
            for (GLint i = 0; i < acb.count(); ++i)
                if (acb.prop(static_cast<GLuint>(i), PRP::BufferBinding) == binding) return static_cast<GLuint>(i);
            return 0;
        }

        // ---------------------------------------------------------------------------------------------
        // The check matrix: one cell per recorded check, one band per group, green = passed.
        void paint_matrix(const demo::Env& env, const Recorder& rec)
        {
            const int region_h = 96;
            int cell = 8;
            auto rows_needed = [&](int px) {
                const int columns = env.width / px;
                int rows = 0;
                for (const auto& g : rec.groups()) rows += (static_cast<int>(g.results.size()) + columns - 1) / columns;
                return rows;
            };
            while (cell > 2 && rows_needed(cell) * cell > region_h) cell /= 2;
            const int columns = env.width / cell;
            mgl::glEnable(mgl::Capability::ScissorTest);
            int row = 0;
            int painted = 0, failed = 0;
            for (const auto& g : rec.groups())
            {
                for (std::size_t i = 0; i < g.results.size(); ++i)
                {
                    const int c = static_cast<int>(i) % columns;
                    const int r = row + static_cast<int>(i) / columns;
                    if ((r + 1) * cell > region_h) break;
                    if (g.results[i]) mgl::glClearColor(0.10f, 0.75f, 0.20f, 1.0f);
                    else mgl::glClearColor(0.95f, 0.10f, 0.10f, 1.0f);
                    mgl::glScissor(c * cell, region_h - (r + 1) * cell, cell - 1, cell - 1);
                    mgl::glClear(mgl::ClearBufferBit::Color);
                    ++painted;
                    if (!g.results[i]) ++failed;
                }
                row += (static_cast<int>(g.results.size()) + columns - 1) / columns;
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
            std::printf("   note: check matrix: %d checks in %zu groups painted with %d-pixel cells, %d red\n", painted, rec.groups().size(), cell, failed);
            for (const auto& g : rec.groups()) std::printf("   note:   group '%s': %zu checks\n", g.name.c_str(), g.results.size());
        }

        // ---------------------------------------------------------------------------------------------
        mgl::ProgramId gfx_{}, comp_{}, edge_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};
        std::array<mgl::TextureId, 3> textures_{};
        std::array<mgl::BufferId, 3> ubos_{};
        GLint max_locations_ = 0;
        std::string src_edge_, src_dup_, src_overlap_, src_fullscreen_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-program-interface-query",
        .title = "Program interface queries",
        .min_es = {3, 1},
        .summary = "glGetProgramInterfaceiv / glGetProgramResource{Index,Name,iv,Location} on all eight interfaces of ES 3.1, "
                   "compared with values derived from the shader source; explicit uniform locations, layout(binding) and "
                   "MAX_UNIFORM_LOCATIONS.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ProgramInterfaceQuery>(argc, argv, info);
}
