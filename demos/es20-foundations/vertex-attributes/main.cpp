// es20-vertex-attributes
//
// How per-vertex data reaches a vertex shader (ES 3.0 sections 2.8 - 2.10, 2.12.5, 6.1.10, 6.1.12):
//
//   A  data types and the `normalized` flag      - BYTE ... FLOAT, converted by equations 2.1 / 2.2
//   B  component counts                          - sizes 1..4, missing components are (0, 0, 0, 1)
//   C  memory layouts                            - interleaved / planar / one buffer, stride + offset
//   D  constant ("current") attribute values     - glVertexAttrib{1,2,3,4}f[v] while the array is off
//   E  client-memory arrays                      - legal on the default vertex array object only
//   F  attribute locations                       - glBindAttribLocation before link vs the linker's choice
//   G  glGetActiveAttrib enumeration, glGetVertexAttrib* / glGetVertexAttribPointerv state queries
//
// The window is an 8 x 8 grid of cells (row 0 is the top row). Every cell is one draw call into its own
// viewport, and validate() compares the center pixel of each cell with a value computed on the CPU from
// the spec's conversion formulas. The alpha channel carries the w component of the attribute, which is
// how the "missing components are 0, 0, 0, 1" rule becomes observable.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr GLuint kPositionLoc = 0;  // layout(location = 0) in attrib.vert
    constexpr GLuint kValueLoc = 1;     // layout(location = 1) in attrib.vert

    constexpr int kGrid = 8;                                          // cells per side
    constexpr float kInset = 0.875f;                                  // quads span +-kInset: a 2 px gap at 32 px pitch
    constexpr std::array<float, 4> kClear = {0.10f, 0.12f, 0.20f, 1.0f};

    mgl::AttribLocation loc(GLuint index) { return mgl::AttribLocation{index}; }

    std::uint8_t to8(double v)
    {
        return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
    }

    struct Rect
    {
        int x, y, w, h;  // window coordinates, origin bottom-left (GL convention)
        [[nodiscard]] int cx() const { return x + w / 2; }
        [[nodiscard]] int cy() const { return y + h / 2; }
    };

    // ---------------------------------------------------------------------------------------------
    // A. Data types. `raw` holds the three values exactly as they are stored in the buffer: integers
    //    for integer types, a 16.16 number for FIXED, the value itself for FLOAT. gain / bias map the
    //    value that reaches the shader into [0, 1] (see attrib.vert) so that it can be seen.
    // ---------------------------------------------------------------------------------------------
    struct FormatCase
    {
        const char* label;
        mgl::DataType type;
        bool normalized;
        std::array<double, 3> raw;
        std::array<float, 3> gain;
        std::array<float, 3> bias;
        int col, row;
    };

    constexpr float kHalf = 0.5f;
    using DT = mgl::DataType;

    constexpr std::array<FormatCase, 17> kFormats = {{
        // Unsigned bytes: normalized divides by 255 inside the GL; the "raw" twin lets the shader do it.
        {"UNSIGNED_BYTE  normalized=TRUE   c / 255", DT::UnsignedByte, true, {255, 102, 51}, {1, 1, 1}, {0, 0, 0}, 0, 0},
        {"UNSIGNED_BYTE  normalized=FALSE  255.0 reaches the shader, *1/255", DT::UnsignedByte, false, {255, 102, 51},
         {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f}, {0, 0, 0}, 1, 0},
        // Signed bytes (ES 3.0: max(c / 127, -1), so 0 is exact). Display: value * 0.5 + 0.5.
        {"BYTE           normalized=TRUE   max(c / 127, -1)", DT::Byte, true, {100, 0, -127}, {kHalf, kHalf, kHalf},
         {kHalf, kHalf, kHalf}, 0, 1},
        {"BYTE           normalized=FALSE  raw integer, *0.5/127 + 0.5", DT::Byte, false, {100, 0, -127},
         {kHalf / 127.0f, kHalf / 127.0f, kHalf / 127.0f}, {kHalf, kHalf, kHalf}, 1, 1},
        {"UNSIGNED_SHORT normalized=TRUE   c / 65535", DT::UnsignedShort, true, {65535, 16384, 49152}, {1, 1, 1},
         {0, 0, 0}, 0, 2},
        {"UNSIGNED_SHORT normalized=FALSE  raw integer, *1/65535", DT::UnsignedShort, false, {65535, 16384, 49152},
         {1.0f / 65535.0f, 1.0f / 65535.0f, 1.0f / 65535.0f}, {0, 0, 0}, 1, 2},
        {"SHORT          normalized=TRUE   max(c / 32767, -1)", DT::Short, true, {32767, 0, -16384},
         {kHalf, kHalf, kHalf}, {kHalf, kHalf, kHalf}, 0, 3},
        {"SHORT          normalized=FALSE  raw integer, *0.5/32767 + 0.5", DT::Short, false, {32767, 0, -16384},
         {kHalf / 32767.0f, kHalf / 32767.0f, kHalf / 32767.0f}, {kHalf, kHalf, kHalf}, 1, 3},
        {"UNSIGNED_INT   normalized=TRUE   c / (2^32 - 1)", DT::UnsignedInt, true,
         {4294967295.0, 1073741824.0, 3221225472.0}, {1, 1, 1}, {0, 0, 0}, 0, 4},
        {"UNSIGNED_INT   normalized=FALSE  raw integer, *1/(2^32 - 1)", DT::UnsignedInt, false,
         {4294967295.0, 1073741824.0, 3221225472.0},
         {1.0f / 4294967295.0f, 1.0f / 4294967295.0f, 1.0f / 4294967295.0f}, {0, 0, 0}, 1, 4},
        {"INT            normalized=TRUE   max(c / (2^31 - 1), -1)", DT::Int, true,
         {2147483647.0, 0.0, -1073741824.0}, {kHalf, kHalf, kHalf}, {kHalf, kHalf, kHalf}, 0, 5},
        {"INT            normalized=FALSE  raw integer, *0.5/(2^31 - 1) + 0.5", DT::Int, false,
         {2147483647.0, 0.0, -1073741824.0}, {kHalf / 2147483647.0f, kHalf / 2147483647.0f, kHalf / 2147483647.0f},
         {kHalf, kHalf, kHalf}, 1, 5},
        // FIXED is 16.16 two's complement: value = c / 65536, whatever the flag says. (3, -1, 0.5) * 0.25 + 0.25.
        {"FIXED          normalized=TRUE   flag ignored, c / 65536", DT::Fixed, true, {196608, -65536, 32768},
         {0.25f, 0.25f, 0.25f}, {0.25f, 0.25f, 0.25f}, 0, 6},
        {"FIXED          normalized=FALSE  c / 65536", DT::Fixed, false, {196608, -65536, 32768},
         {0.25f, 0.25f, 0.25f}, {0.25f, 0.25f, 0.25f}, 1, 6},
        {"FLOAT          normalized=TRUE   flag ignored", DT::Float, true, {0.9, 0.45, 0.15}, {1, 1, 1}, {0, 0, 0}, 0, 7},
        {"FLOAT          normalized=FALSE  the value itself", DT::Float, false, {0.9, 0.45, 0.15}, {1, 1, 1}, {0, 0, 0},
         1, 7},
        // The edge of the signed range. ES 3.0 maps both -128 and -127 to -1.0; the ES 2.0 formula
        // (2c + 1) / 255 gave -1.0, -0.992 and -0.984. A gain of 16 on (v + 1) magnifies that difference.
        {"BYTE           normalized=TRUE   -128, -127, -126 (gain 16 on v + 1)", DT::Byte, true, {-128, -127, -126},
         {16, 16, 16}, {16, 16, 16}, 2, 1},
    }};

    /// What the vertex shader receives for one stored component: ES 3.0 2.1.6.1 (equations 2.1, 2.2)
    /// when `normalized` applies, ES 3.0 2.9 otherwise ("converted directly to floating-point").
    double shader_value(mgl::DataType type, bool normalized, double raw)
    {
        switch (type)
        {
        case DT::UnsignedByte: return normalized ? raw / 255.0 : raw;
        case DT::Byte: return normalized ? std::max(raw / 127.0, -1.0) : raw;
        case DT::UnsignedShort: return normalized ? raw / 65535.0 : raw;
        case DT::Short: return normalized ? std::max(raw / 32767.0, -1.0) : raw;
        case DT::UnsignedInt: return normalized ? raw / 4294967295.0 : raw;
        case DT::Int: return normalized ? std::max(raw / 2147483647.0, -1.0) : raw;
        case DT::Fixed: return raw / 65536.0;                              // "ignored if type is FIXED"
        case DT::Float: return static_cast<double>(static_cast<float>(raw));  // "ignored if type is FLOAT"
        default: return raw;
        }
    }

    std::size_t type_size(mgl::DataType type)
    {
        switch (type)
        {
        case DT::Byte:
        case DT::UnsignedByte: return 1;
        case DT::Short:
        case DT::UnsignedShort: return 2;
        default: return 4;  // INT, UNSIGNED_INT, FIXED, FLOAT
        }
    }

    template <typename T>
    void store(std::vector<std::uint8_t>& bytes, std::size_t at, T value)
    {
        std::memcpy(bytes.data() + at, &value, sizeof value);
    }

    void store_component(std::vector<std::uint8_t>& bytes, std::size_t at, mgl::DataType type, double v)
    {
        switch (type)
        {
        case DT::Byte: store(bytes, at, static_cast<std::int8_t>(v)); break;
        case DT::UnsignedByte: store(bytes, at, static_cast<std::uint8_t>(v)); break;
        case DT::Short: store(bytes, at, static_cast<std::int16_t>(v)); break;
        case DT::UnsignedShort: store(bytes, at, static_cast<std::uint16_t>(v)); break;
        case DT::Int:
        case DT::Fixed: store(bytes, at, static_cast<std::int32_t>(v)); break;  // FIXED: `raw` is already c * 65536
        case DT::UnsignedInt: store(bytes, at, static_cast<std::uint32_t>(v)); break;
        case DT::Float: store(bytes, at, static_cast<float>(v)); break;
        default: break;
        }
    }

    std::size_t round_up(std::size_t v, std::size_t multiple) { return (v + multiple - 1) / multiple * multiple; }

    /// shader_value() * gain + bias, clamped and quantized like the default framebuffer would.
    demo::Rgba8 shaded(const std::array<double, 4>& value, const std::array<float, 4>& gain,
                       const std::array<float, 4>& bias)
    {
        demo::Rgba8 out{};
        for (std::size_t i = 0; i < 4; ++i)
            out[i] = to8(value[i] * static_cast<double>(gain[i]) + static_cast<double>(bias[i]));
        return out;
    }

    demo::Rgba8 expected_for(const FormatCase& c)
    {
        // Size 3: the GL supplies w = 1 (ES 3.0 2.9.1). Alpha is therefore 1 * 1 + 0 = 1.
        const std::array<double, 4> v = {shader_value(c.type, c.normalized, c.raw[0]),
                                         shader_value(c.type, c.normalized, c.raw[1]),
                                         shader_value(c.type, c.normalized, c.raw[2]), 1.0};
        return shaded(v, {c.gain[0], c.gain[1], c.gain[2], 1.0f}, {c.bias[0], c.bias[1], c.bias[2], 0.0f});
    }

    // ---------------------------------------------------------------------------------------------
    // B. Component counts (FLOAT arrays of 1..4 values per vertex).
    // ---------------------------------------------------------------------------------------------
    struct SizeCase
    {
        int size;
        std::array<float, 4> v;
        int col, row;
    };

    constexpr std::array<SizeCase, 4> kSizes = {{
        {1, {0.8f, 0, 0, 0}, 3, 0},
        {2, {0.2f, 0.6f, 0, 0}, 3, 1},
        {3, {0.3f, 0.5f, 0.9f, 0}, 3, 2},
        {4, {0.9f, 0.4f, 0.2f, 0.5f}, 3, 3},
    }};

    /// (x, y, z, w) after the GL expanded an array element or a glVertexAttrib call of `n` components.
    std::array<double, 4> expand(int n, const std::array<float, 4>& v)
    {
        std::array<double, 4> out = {0.0, 0.0, 0.0, 1.0};
        for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = static_cast<double>(v[static_cast<std::size_t>(i)]);
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // C. Layouts: four quads (two triangles each) described in four different ways.
    // ---------------------------------------------------------------------------------------------
    struct QuadDef
    {
        float x0, y0, x1, y1;
        std::array<std::uint8_t, 3> rgb;
    };

    constexpr std::array<QuadDef, 4> kQuads = {{
        {-0.9f, -0.9f, -0.2f, -0.2f, {230, 57, 70}},   // lower left : red
        {0.2f, -0.9f, 0.9f, -0.2f, {42, 157, 143}},    // lower right: teal
        {-0.9f, 0.2f, -0.2f, 0.9f, {69, 123, 157}},    // upper left : blue
        {0.2f, 0.2f, 0.9f, 0.9f, {244, 162, 97}},      // upper right: orange
    }};
    constexpr int kQuadVertices = 24;  // 4 quads x 2 triangles x 3 vertices

    struct LayoutVertex
    {
        float x, y;
        std::array<std::uint8_t, 3> rgb;
    };

    std::vector<LayoutVertex> layout_vertices()
    {
        std::vector<LayoutVertex> v;
        for (const QuadDef& q : kQuads)
        {
            v.push_back({q.x0, q.y0, q.rgb});
            v.push_back({q.x1, q.y0, q.rgb});
            v.push_back({q.x1, q.y1, q.rgb});
            v.push_back({q.x0, q.y0, q.rgb});
            v.push_back({q.x1, q.y1, q.rgb});
            v.push_back({q.x0, q.y1, q.rgb});
        }
        return v;
    }

    struct InterleavedFloat  // layout 1: x y r g b, 20 bytes per vertex
    {
        float x, y, r, g, b;
    };
    struct InterleavedPacked  // layout 4: 8 bytes per vertex, positions as normalized SHORT, colors as normalized UBYTE
    {
        std::int16_t x, y;
        std::uint8_t r, g, b, pad;
    };
    static_assert(sizeof(InterleavedFloat) == 20);
    static_assert(sizeof(InterleavedPacked) == 8);

    // What validate() expects about each layout's recorded state: {stride, offset, type, normalized}
    // for the position array (attribute 0) and the color array (attribute 1).
    struct ArrayExpect
    {
        GLint stride;
        std::size_t offset;
        mgl::DataType type;
        bool normalized;
    };
    struct LayoutExpect
    {
        const char* name;
        ArrayExpect position, color;
        bool same_buffer;
    };

    constexpr std::size_t kPlanarHeader = 16;  // layout 3: unused bytes in front of the arrays
    constexpr std::size_t kPackedHeader = 8;   // layout 4

    const std::array<LayoutExpect, 4> kLayouts = {{
        {"interleaved floats  x y r g b", {20, 0, DT::Float, false}, {20, 8, DT::Float, false}, true},
        {"planar, two buffers", {0, 0, DT::Float, false}, {0, 0, DT::Float, false}, false},
        {"planar, one buffer with a header",
         {0, kPlanarHeader, DT::Float, false},
         {0, kPlanarHeader + kQuadVertices * 2 * sizeof(float), DT::Float, false},
         true},
        {"packed interleaved, SHORT + UBYTE normalized",
         {8, kPackedHeader, DT::Short, true},
         {8, kPackedHeader + 4, DT::UnsignedByte, true},
         true},
    }};

    // ---------------------------------------------------------------------------------------------
    // D. Constant attribute values.
    // ---------------------------------------------------------------------------------------------
    struct ConstCase
    {
        int n;  // number of components given
        bool vector_form;  // glVertexAttribNfv instead of glVertexAttribNf
        std::array<float, 4> v;
        int col, row;
    };

    constexpr std::array<ConstCase, 8> kConsts = {{
        {1, false, {0.9f, 0, 0, 0}, 4, 4},
        {1, true, {0.45f, 0, 0, 0}, 5, 4},
        {2, false, {0.2f, 0.6f, 0, 0}, 6, 4},
        {2, true, {0.9f, 0.3f, 0, 0}, 7, 4},
        {3, false, {0.3f, 0.5f, 0.9f, 0}, 4, 5},
        {3, true, {0.7f, 0.1f, 0.5f, 0}, 5, 5},
        {4, false, {0.9f, 0.4f, 0.2f, 0.5f}, 6, 5},
        {4, true, {0.1f, 0.8f, 0.6f, 0.25f}, 7, 5},
    }};
    constexpr std::array<float, 3> kArrayColor = {0.2f, 0.8f, 0.3f};  // the enabled-array cell next to the constants

    // ---------------------------------------------------------------------------------------------
    // E. Client arrays and F. named attributes: colors of the cells.
    // ---------------------------------------------------------------------------------------------
    constexpr std::array<float, 3> kClientA = {0.85f, 0.35f, 0.20f};
    constexpr std::array<float, 3> kClientB = {0.20f, 0.55f, 0.85f};
    constexpr std::array<float, 3> kMixed = {0.60f, 0.80f, 0.20f};
    constexpr std::array<float, 3> kBoundColor = {0.90f, 0.50f, 0.10f};
    constexpr std::array<float, 3> kAutoColor = {0.20f, 0.70f, 0.80f};

    demo::Rgba8 rgb_expected(const std::array<float, 3>& c)
    {
        return {to8(static_cast<double>(c[0])), to8(static_cast<double>(c[1])), to8(static_cast<double>(c[2])), 255};
    }

    // ---------------------------------------------------------------------------------------------
    // Small GL helpers (programs with attribute bindings, state queries).
    // ---------------------------------------------------------------------------------------------
    struct Binding
    {
        GLuint index;
        const char* name;
    };

    struct Prog
    {
        mgl::ProgramId program{};
        mgl::ShaderId vs{}, fs{};
        bool linked = false;
    };

    /// glCreateProgram + glAttachShader x2 + glBindAttribLocation* + glLinkProgram. The bindings must be
    /// issued before the link: they have no effect on an already linked program (ES 3.0 2.12.5).
    /// The shaders stay attached so that the program can be linked again.
    Prog make_program(std::string_view vs_src, std::string_view fs_src, std::span<const Binding> bindings,
                      std::string_view label)
    {
        Prog p;
        std::string log;
        p.vs = demo::compile_shader(mgl::ShaderType::Vertex, vs_src, log);
        if (p.vs.value == 0) throw demo::Failure(std::format("{}: vertex shader:\n{}", label, log));
        p.fs = demo::compile_shader(mgl::ShaderType::Fragment, fs_src, log);
        if (p.fs.value == 0) throw demo::Failure(std::format("{}: fragment shader:\n{}", label, log));
        p.program = mgl::glCreateProgram();
        mgl::glAttachShader(p.program, p.vs);
        mgl::glAttachShader(p.program, p.fs);
        for (const Binding& b : bindings) mgl::glBindAttribLocation(p.program, loc(b.index), b.name);
        mgl::glLinkProgram(p.program);
        GLint status = GL_FALSE;
        mgl::glGetProgramiv(p.program, mgl::ProgramParameter::LinkStatus, &status);
        p.linked = status == GL_TRUE;
        return p;
    }

    void destroy(Prog& p)
    {
        mgl::glDeleteProgram(p.program);  // also drops the attachments ...
        mgl::glDeleteShader(p.vs);        // ... so these deletions take effect immediately
        mgl::glDeleteShader(p.fs);
        p = {};
    }

    GLint program_param(mgl::ProgramId p, mgl::ProgramParameter q)
    {
        GLint v = -1;
        mgl::glGetProgramiv(p, q, &v);
        return v;
    }

    GLint attrib_i(GLuint index, mgl::VertexAttribParameter pname)
    {
        GLint v = -12345;
        mgl::glGetVertexAttribiv(loc(index), pname, &v);
        return v;
    }

    void* attrib_pointer(GLuint index)
    {
        void* p = reinterpret_cast<void*>(std::uintptr_t{0xDEAD});
        mgl::glGetVertexAttribPointerv(loc(index), mgl::VertexAttribParameter::ArrayPointer, &p);
        return p;
    }

    std::array<float, 4> current_attrib(GLuint index)
    {
        std::array<float, 4> v = {-1.0f, -1.0f, -1.0f, -1.0f};
        mgl::glGetVertexAttribfv(loc(index), mgl::VertexAttribParameter::CurrentVertexAttrib, v.data());
        return v;
    }

    struct Probe
    {
        int x, y;
        demo::Rgba8 expected;
        const char* what;
    };

    /// One recorded check for several pixels; the detail names every pixel that differs.
    bool expect_probes(demo::Check& check, std::string_view what, std::span<const Probe> probes, int tolerance)
    {
        bool ok = true;
        std::string detail;
        for (const Probe& p : probes)
        {
            const demo::Rgba8 got = demo::read_pixel(p.x, p.y);
            bool match = true;
            for (std::size_t i = 0; i < 4; ++i)
                if (std::abs(int{got[i]} - int{p.expected[i]}) > tolerance) match = false;
            if (!match)
            {
                ok = false;
                detail += std::format("{}: pixel({},{}) = {} expected {}; ", p.what, p.x, p.y, demo::format_rgba(got),
                                      demo::format_rgba(p.expected));
            }
        }
        return check.expect(ok, what, detail);
    }

    class VertexAttributes final : public demo::Demo
    {
    public:
        // =========================================================================================
        void init(demo::Env& env) override
        {
            pitch_ = std::min(env.width, env.height) / kGrid;
            attrib_vs_src_ = env.load_text("shaders/attrib.vert");
            named_vs_src_ = env.load_text("shaders/named.vert");
            enumerate_vs_src_ = env.load_text("shaders/enumerate.vert");
            frag_src_ = env.load_text("shaders/attrib.frag");

            program_ = demo::build_program_from_source(attrib_vs_src_, frag_src_, "attrib");
            u_gain_ = mgl::glGetUniformLocation(program_, "u_gain");
            u_bias_ = mgl::glGetUniformLocation(program_, "u_bias");
            if (u_gain_.value < 0 || u_bias_.value < 0) throw demo::Failure("u_gain / u_bias are not active");

            create_shared_buffers();
            create_format_buffers();
            create_layout_buffers_and_vaos();
            create_named_programs();
            create_enumeration_program();
        }

        // =========================================================================================
        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);  // attribute 0 -> unit quad (VBO); attribute 1 is set up per cell

            draw_formats();
            draw_sizes();
            draw_layouts();
            draw_constants();
            draw_named_attributes();
            draw_client_arrays();

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // =========================================================================================
        void validate(demo::Env& env, demo::Check& check) override
        {
            check_limits(check);
            check_formats(check);
            check_sizes(check);
            check_layouts(check);
            check_constants(check);
            check_named_attributes(check);
            check_active_attributes(check);
            check_client_arrays(check);
            check_state_queries(check);
            check_attribute_errors(check);

            // Everything above is verified. The PNG capture is taken after validate(): make alpha opaque so that
            // viewers do not composite the cells whose attribute w was 0.5 / 0.25 over a background.
            mgl::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            const Rect w4 = cell(kSizes[3].col, kSizes[3].row);
            check.expect_pixel(w4.cx(), w4.cy(), {229, 102, 51, 255}, 1,
                               "glColorMask(alpha only) + glClear: alpha is now opaque, the RGB of the w = 0.5 cell is untouched");
        }

        // =========================================================================================
        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteVertexArrays(1, &vao_bound_);
            mgl::glDeleteVertexArrays(1, &vao_auto_);
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(layout_vao_.size()), layout_vao_.data());
            mgl::glDeleteBuffers(1, &quad_vbo_);
            mgl::glDeleteBuffers(1, &format_vbo_);
            mgl::glDeleteBuffers(1, &sizes_vbo_);
            mgl::glDeleteBuffers(1, &array_color_vbo_);
            mgl::glDeleteBuffers(1, &named_color_vbo_);
            mgl::glDeleteBuffers(static_cast<GLsizei>(layout_vbo_.size()), layout_vbo_.data());
            mgl::glDeleteProgram(program_);
            destroy(bound_);
            destroy(auto_);
            destroy(enum_);
        }

    private:
        // ---- geometry of the grid -------------------------------------------------------------
        [[nodiscard]] Rect cell(int col, int row, int span = 1) const
        {
            return {col * pitch_, (kGrid - row - span) * pitch_, span * pitch_, span * pitch_};
        }

        void draw_quad(const Rect& r)
        {
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);  // the unit quad as a strip
        }

        void set_gain_bias(const std::array<float, 4>& gain, const std::array<float, 4>& bias)
        {
            mgl::glUniform4f(u_gain_, gain[0], gain[1], gain[2], gain[3]);
            mgl::glUniform4f(u_bias_, bias[0], bias[1], bias[2], bias[3]);
        }

        void identity_gain_bias() { set_gain_bias({1, 1, 1, 1}, {0, 0, 0, 0}); }

        // ---- setup ---------------------------------------------------------------------------
        void create_shared_buffers()
        {
            // The unit quad as a TRIANGLE_STRIP; every cell is this quad drawn into a different viewport.
            const std::array<float, 8> quad = {-kInset, -kInset, kInset, -kInset, -kInset, kInset, kInset, kInset};
            mgl::glGenBuffers(1, &quad_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(quad), mgl::BufferUsage::StaticDraw);

            // Colors of the "enabled array" cell (four identical vec3 vertices).
            std::array<float, 12> colors{};
            for (std::size_t v = 0; v < 4; ++v)
                for (std::size_t c = 0; c < 3; ++c) colors[v * 3 + c] = kArrayColor[c];
            mgl::glGenBuffers(1, &array_color_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, array_color_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(colors), mgl::BufferUsage::StaticDraw);

            // The work VAO: attribute 0 comes from the quad. The buffer bound to ARRAY_BUFFER when
            // glVertexAttribPointer runs is what the VAO remembers (ES 3.0 2.10.6).
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glEnableVertexAttribArray(loc(kPositionLoc));
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void create_format_buffers()
        {
            // Section A: every case gets its own 4-vertex block (identical vertices -> a flat cell) at a
            // 16-byte aligned offset. The stride is padded to a multiple of 4 bytes and the padding is
            // poisoned: the arrays have size 3, so a stray padding byte can never become a component.
            std::vector<std::uint8_t> blob;
            for (const FormatCase& c : kFormats)
            {
                blob.resize(round_up(blob.size(), 16), 0xEE);
                const std::size_t elem = type_size(c.type);
                const std::size_t stride = round_up(3 * elem, 4);
                format_layout_.push_back({blob.size(), static_cast<GLsizei>(stride)});
                for (int vertex = 0; vertex < 4; ++vertex)
                {
                    const std::size_t base = blob.size();
                    blob.resize(base + stride, 0xEE);
                    for (std::size_t k = 0; k < 3; ++k) store_component(blob, base + k * elem, c.type, c.raw[k]);
                }
            }
            mgl::glGenBuffers(1, &format_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, format_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const std::uint8_t>(blob),
                              mgl::BufferUsage::StaticDraw);

            // Section B: tightly packed FLOAT arrays of 1..4 components (stride 0).
            std::vector<float> floats;
            for (const SizeCase& s : kSizes)
            {
                while ((floats.size() * sizeof(float)) % 16 != 0) floats.push_back(-1.0f);
                size_offset_.push_back(floats.size() * sizeof(float));
                for (int vertex = 0; vertex < 4; ++vertex)
                    for (int k = 0; k < s.size; ++k) floats.push_back(s.v[static_cast<std::size_t>(k)]);
            }
            mgl::glGenBuffers(1, &sizes_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, sizes_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(floats), mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void create_layout_buffers_and_vaos()
        {
            const std::vector<LayoutVertex> verts = layout_vertices();
            mgl::glGenBuffers(static_cast<GLsizei>(layout_vbo_.size()), layout_vbo_.data());
            mgl::glGenVertexArrays(static_cast<GLsizei>(layout_vao_.size()), layout_vao_.data());
            const auto A = mgl::BufferTarget::Array;
            const auto F = mgl::DataType::Float;

            // Layout 1: one buffer, one struct per vertex {x, y, r, g, b}; stride 20, offsets 0 and 8.
            std::vector<InterleavedFloat> v1;
            for (const LayoutVertex& v : verts)
                v1.push_back({v.x, v.y, static_cast<float>(v.rgb[0]) / 255.0f, static_cast<float>(v.rgb[1]) / 255.0f,
                              static_cast<float>(v.rgb[2]) / 255.0f});
            mgl::glBindBuffer(A, layout_vbo_[0]);
            mgl::glBufferData(A, std::span<const InterleavedFloat>(v1), mgl::BufferUsage::StaticDraw);
            mgl::glBindVertexArray(layout_vao_[0]);
            mgl::glEnableVertexAttribArray(loc(kPositionLoc));
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, F, GL_FALSE, sizeof(InterleavedFloat),
                                       demo::buffer_offset(offsetof(InterleavedFloat, x)));
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, F, GL_FALSE, sizeof(InterleavedFloat),
                                       demo::buffer_offset(offsetof(InterleavedFloat, r)));

            // Layout 2: planar, one buffer per attribute; stride 0 means "tightly packed".
            std::vector<float> positions, colors;
            for (const LayoutVertex& v : verts)
            {
                positions.push_back(v.x);
                positions.push_back(v.y);
                for (const std::uint8_t c : v.rgb) colors.push_back(static_cast<float>(c) / 255.0f);
            }
            mgl::glBindBuffer(A, layout_vbo_[1]);
            mgl::glBufferData(A, std::span<const float>(positions), mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(A, layout_vbo_[2]);
            mgl::glBufferData(A, std::span<const float>(colors), mgl::BufferUsage::StaticDraw);
            mgl::glBindVertexArray(layout_vao_[1]);
            mgl::glEnableVertexAttribArray(loc(kPositionLoc));
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            mgl::glBindBuffer(A, layout_vbo_[1]);  // each pointer call captures the buffer bound *now*
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, F, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(A, layout_vbo_[2]);
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, F, GL_FALSE, 0, demo::buffer_offset(0));

            // Layout 3: planar inside ONE buffer: [16 unused bytes][all positions][all colors].
            std::vector<std::uint8_t> blob(kPlanarHeader, 0xEE);
            const auto* pos_bytes = reinterpret_cast<const std::uint8_t*>(positions.data());
            const auto* col_bytes = reinterpret_cast<const std::uint8_t*>(colors.data());
            blob.insert(blob.end(), pos_bytes, pos_bytes + positions.size() * sizeof(float));
            blob.insert(blob.end(), col_bytes, col_bytes + colors.size() * sizeof(float));
            mgl::glBindBuffer(A, layout_vbo_[3]);
            mgl::glBufferData(A, std::span<const std::uint8_t>(blob), mgl::BufferUsage::StaticDraw);
            mgl::glBindVertexArray(layout_vao_[2]);
            mgl::glEnableVertexAttribArray(loc(kPositionLoc));
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, F, GL_FALSE, 0, demo::buffer_offset(kLayouts[2].position.offset));
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, F, GL_FALSE, 0, demo::buffer_offset(kLayouts[2].color.offset));

            // Layout 4: a compact vertex - SHORT x, y normalized to [-1, 1] and UBYTE r, g, b normalized to
            // [0, 1] (one padding byte keeps the stride a multiple of 4) - behind an 8 byte header.
            std::vector<std::uint8_t> packed(kPackedHeader, 0xEE);
            for (const LayoutVertex& v : verts)
            {
                const InterleavedPacked pv = {static_cast<std::int16_t>(std::lround(static_cast<double>(v.x) * 32767.0)),
                                              static_cast<std::int16_t>(std::lround(static_cast<double>(v.y) * 32767.0)),
                                              v.rgb[0], v.rgb[1], v.rgb[2], 0xEE};
                const auto* b = reinterpret_cast<const std::uint8_t*>(&pv);
                packed.insert(packed.end(), b, b + sizeof pv);
            }
            mgl::glBindBuffer(A, layout_vbo_[4]);
            mgl::glBufferData(A, std::span<const std::uint8_t>(packed), mgl::BufferUsage::StaticDraw);
            mgl::glBindVertexArray(layout_vao_[3]);
            mgl::glEnableVertexAttribArray(loc(kPositionLoc));
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, mgl::DataType::Short, GL_TRUE, sizeof(InterleavedPacked),
                                       demo::buffer_offset(kPackedHeader + offsetof(InterleavedPacked, x)));
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, mgl::DataType::UnsignedByte, GL_TRUE, sizeof(InterleavedPacked),
                                       demo::buffer_offset(kPackedHeader + offsetof(InterleavedPacked, r)));

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(A, mgl::BufferId{0});
        }

        void create_named_programs()
        {
            // Program 1: the application chooses the locations - before linking.
            //   a_unused is bound too (to 9) and "no_such_input" is a name the shader does not even contain;
            //   both bindings are legal and silently ignored (ES 3.0 2.12.5: "any name"; "ignored").
            const std::array<Binding, 4> bindings = {{{5, "a_position"}, {2, "a_color"}, {9, "a_unused"},
                                                      {3, "no_such_input"}}};
            bound_ = make_program(named_vs_src_, frag_src_, bindings, "named (bound)");
            if (!bound_.linked) throw demo::Failure("named (bound) failed to link:\n" + demo::program_info_log(bound_.program));

            // Program 2: the same shaders, no bindings. The linker picks the locations; the only way to
            // learn them is glGetAttribLocation after the link.
            auto_ = make_program(named_vs_src_, frag_src_, {}, "named (auto)");
            if (!auto_.linked) throw demo::Failure("named (auto) failed to link:\n" + demo::program_info_log(auto_.program));
            auto_position_ = mgl::glGetAttribLocation(auto_.program, "a_position");
            auto_color_ = mgl::glGetAttribLocation(auto_.program, "a_color");
            if (!auto_position_.valid() || !auto_color_.valid()) throw demo::Failure("a_position / a_color are not active");

            // Colors for both programs: two blocks of four identical vec3 vertices.
            std::array<float, 24> colors{};
            for (std::size_t v = 0; v < 4; ++v)
                for (std::size_t c = 0; c < 3; ++c)
                {
                    colors[v * 3 + c] = kBoundColor[c];
                    colors[12 + v * 3 + c] = kAutoColor[c];
                }
            mgl::glGenBuffers(1, &named_color_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, named_color_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(colors), mgl::BufferUsage::StaticDraw);

            // VAOs that use the locations the programs actually have.
            mgl::glGenVertexArrays(1, &vao_bound_);
            mgl::glBindVertexArray(vao_bound_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glEnableVertexAttribArray(loc(5));
            mgl::glVertexAttribPointer(loc(5), 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, named_color_vbo_);
            mgl::glEnableVertexAttribArray(loc(2));
            mgl::glVertexAttribPointer(loc(2), 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));

            mgl::glGenVertexArrays(1, &vao_auto_);
            mgl::glBindVertexArray(vao_auto_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glEnableVertexAttribArray(auto_position_);
            mgl::glVertexAttribPointer(auto_position_, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, named_color_vbo_);
            mgl::glEnableVertexAttribArray(auto_color_);
            mgl::glVertexAttribPointer(auto_color_, 3, mgl::DataType::Float, GL_FALSE, 0,
                                       demo::buffer_offset(12 * sizeof(float)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            std::cout << std::format("     linker-assigned locations: a_position = {}, a_color = {} (implementation-defined)\n",
                                     auto_position_.value, auto_color_.value);
        }

        void create_enumeration_program()
        {
            // a_basis (mat2) is bound to 4 and therefore occupies slots 4 AND 5; the linker must place the other
            // attributes around it.
            const std::array<Binding, 1> bindings = {{{4, "a_basis"}}};
            enum_ = make_program(enumerate_vs_src_, frag_src_, bindings, "enumerate");
            if (!enum_.linked) throw demo::Failure("enumerate failed to link:\n" + demo::program_info_log(enum_.program));
        }

        // ---- drawing ---------------------------------------------------------------------------
        void draw_formats()
        {
            // The ARRAY_BUFFER binding must be the data buffer while glVertexAttribPointer runs.
            mgl::glBindBuffer(mgl::BufferTarget::Array, format_vbo_);
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            for (std::size_t i = 0; i < kFormats.size(); ++i)
            {
                const FormatCase& c = kFormats[i];
                set_gain_bias({c.gain[0], c.gain[1], c.gain[2], 1.0f}, {c.bias[0], c.bias[1], c.bias[2], 0.0f});
                mgl::glVertexAttribPointer(loc(kValueLoc), 3, c.type, c.normalized ? GL_TRUE : GL_FALSE,
                                           format_layout_[i].stride, demo::buffer_offset(format_layout_[i].offset));
                draw_quad(cell(c.col, c.row));
            }
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void draw_sizes()
        {
            identity_gain_bias();
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            mgl::glBindBuffer(mgl::BufferTarget::Array, sizes_vbo_);
            for (std::size_t i = 0; i < kSizes.size(); ++i)
            {
                // size 1..4 FLOAT values per vertex, stride 0 (tightly packed)
                mgl::glVertexAttribPointer(loc(kValueLoc), kSizes[i].size, mgl::DataType::Float, GL_FALSE, 0,
                                           demo::buffer_offset(size_offset_[i]));
                draw_quad(cell(kSizes[i].col, kSizes[i].row));
            }
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void draw_layouts()
        {
            identity_gain_bias();
            static constexpr int cols[4] = {4, 6, 4, 6};
            static constexpr int rows[4] = {0, 0, 2, 2};
            for (std::size_t i = 0; i < 4; ++i)
            {
                mgl::glBindVertexArray(layout_vao_[i]);  // the whole layout is VAO state: nothing else to set
                const Rect r = cell(cols[i], rows[i], 2);
                mgl::glViewport(r.x, r.y, r.w, r.h);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, kQuadVertices);
            }
            mgl::glBindVertexArray(vao_);
        }

        void draw_constants()
        {
            identity_gain_bias();
            const mgl::AttribLocation a = loc(kValueLoc);
            mgl::glBindVertexArray(vao_);
            mgl::glDisableVertexAttribArray(a);  // the array is off: the shader sees the *current* value
            for (std::size_t i = 0; i < kConsts.size(); ++i)
            {
                const ConstCase& c = kConsts[i];
                if (c.n == 1)
                {
                    if (c.vector_form) mgl::glVertexAttrib1fv(a, c.v.data());
                    else mgl::glVertexAttrib1f(a, c.v[0]);
                }
                else if (c.n == 2)
                {
                    if (c.vector_form) mgl::glVertexAttrib2fv(a, c.v.data());
                    else mgl::glVertexAttrib2f(a, c.v[0], c.v[1]);
                }
                else if (c.n == 3)
                {
                    if (c.vector_form) mgl::glVertexAttrib3fv(a, c.v.data());
                    else mgl::glVertexAttrib3f(a, c.v[0], c.v[1], c.v[2]);
                }
                else
                {
                    if (c.vector_form) mgl::glVertexAttrib4fv(a, c.v.data());
                    else mgl::glVertexAttrib4f(a, c.v[0], c.v[1], c.v[2], c.v[3]);
                }
                observed_current_[i] = current_attrib(kValueLoc);  // validate() compares with the spec's expansion
                draw_quad(cell(c.col, c.row));
            }

            // Enabling an array and drawing from it must not disturb the current value (ES 3.0 2.9.3) ...
            mgl::glBindBuffer(mgl::BufferTarget::Array, array_color_vbo_);
            mgl::glEnableVertexAttribArray(a);
            mgl::glVertexAttribPointer(a, 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            draw_quad(cell(4, 6));
            // ... so after disabling the array again the last glVertexAttrib4fv value is back in effect.
            mgl::glDisableVertexAttribArray(a);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            draw_quad(cell(5, 6));
        }

        void draw_named_attributes()
        {
            // Both programs draw a plain quad; only the way their input locations were obtained differs.
            mgl::glUseProgram(bound_.program);
            mgl::glBindVertexArray(vao_bound_);
            draw_quad(cell(3, 7));
            mgl::glUseProgram(auto_.program);
            mgl::glBindVertexArray(vao_auto_);
            draw_quad(cell(4, 7));
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
        }

        void draw_client_arrays()
        {
            // Client-memory arrays: glVertexAttribPointer with NO buffer bound to ARRAY_BUFFER and a real CPU
            // pointer. ES 3.0 allows this only while the default vertex array object (0) is bound
            // (INVALID_OPERATION otherwise, section 2.9, footnote 5).
            identity_gain_bias();
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            const mgl::DataType F = mgl::DataType::Float;

            client_positions_ = {-kInset, -kInset, kInset, -kInset, -kInset, kInset, kInset, kInset};
            fill_client_colors(kClientA);
            mgl::glEnableVertexAttribArray(loc(kPositionLoc));
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, F, GL_FALSE, 0, client_positions_.data());
            mgl::glEnableVertexAttribArray(loc(kValueLoc));
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, F, GL_FALSE, 0, client_colors_.data());
            draw_quad(cell(3, 4));

            // The pointers are not captured at specification time: the CPU array is read when the draw call
            // executes, so changing it and drawing again gives a different color without another pointer call.
            fill_client_colors(kClientB);
            draw_quad(cell(3, 5));

            // Sources can be mixed in one draw call: positions from a buffer object, colors from client memory.
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, F, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            fill_client_colors(kMixed);
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, F, GL_FALSE, 0, client_colors_.data());
            draw_quad(cell(3, 6));

            reset_default_vao_arrays();
        }

        void fill_client_colors(const std::array<float, 3>& c)
        {
            for (std::size_t v = 0; v < 4; ++v)
                for (std::size_t k = 0; k < 3; ++k) client_colors_[v * 3 + k] = c[k];
        }

        /// Back to the initial state of the default VAO's attributes 0 and 1 (disabled, no buffer, NULL).
        void reset_default_vao_arrays()
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            for (const GLuint index : {kPositionLoc, kValueLoc})
            {
                mgl::glDisableVertexAttribArray(loc(index));
                mgl::glVertexAttribPointer(loc(index), 4, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            }
        }

        // ---- validation ------------------------------------------------------------------------
        void check_limits(demo::Check& check)
        {
            max_attribs_ = demo::get_integer(mgl::GetParameter::MaxVertexAttribs);
            check.expect(max_attribs_ >= 16, "MAX_VERTEX_ATTRIBS >= 16 (ES 3.0 Table 6.31)",
                         std::format("MAX_VERTEX_ATTRIBS = {}", max_attribs_));
            check.expect(demo::get_integer(mgl::GetParameter::AlphaBits) >= 8,
                         "default framebuffer has an 8-bit alpha channel (the w component is read back through it)");
            // The very first thing nobody wrote to: attribute MAX-1 still holds the initial (0, 0, 0, 1) (ES 3.0 2.8).
            const auto init = current_attrib(static_cast<GLuint>(max_attribs_ - 1));
            check.expect(init == std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f},
                         "initial current value of an untouched attribute is (0, 0, 0, 1)",
                         std::format("({}, {}, {}, {})", init[0], init[1], init[2], init[3]));
        }

        void check_formats(demo::Check& check)
        {
            for (const FormatCase& c : kFormats)
            {
                const Rect r = cell(c.col, c.row);
                check.expect_pixel(r.cx(), r.cy(), expected_for(c), 1, c.label);
            }
        }

        void check_sizes(demo::Check& check)
        {
            for (const SizeCase& s : kSizes)
            {
                const Rect r = cell(s.col, s.row);
                const demo::Rgba8 want = shaded(expand(s.size, s.v), {1, 1, 1, 1}, {0, 0, 0, 0});
                check.expect_pixel(r.cx(), r.cy(), want, 1,
                                   std::format("FLOAT array with {} component(s): the rest is (0, 0, 0, 1)", s.size));
            }
        }

        void check_layouts(demo::Check& check)
        {
            static constexpr int cols[4] = {4, 6, 4, 6};
            static constexpr int rows[4] = {0, 0, 2, 2};
            for (std::size_t i = 0; i < 4; ++i)
            {
                const Rect r = cell(cols[i], rows[i], 2);
                // NDC (+-0.55, +-0.55) is the center of each quad, NDC (0, 0) lies in the gap between them.
                auto at = [&](double ndc) { return static_cast<int>(std::floor((ndc + 1.0) * 0.5 * static_cast<double>(r.w))); };
                std::array<Probe, 5> probes;
                for (std::size_t q = 0; q < 4; ++q)
                {
                    const auto& rgb = kQuads[q].rgb;
                    probes[q] = {r.x + at((q & 1) != 0 ? 0.55 : -0.55), r.y + at((q & 2) != 0 ? 0.55 : -0.55),
                                 {rgb[0], rgb[1], rgb[2], 255}, "quad"};
                }
                probes[4] = {r.cx(), r.cy(), {to8(static_cast<double>(kClear[0])), to8(static_cast<double>(kClear[1])), to8(static_cast<double>(kClear[2])), 255}, "gap"};
                expect_probes(check, std::format("layout: {} - all four quads in place, gap empty", kLayouts[i].name),
                              probes, 1);

                // The recorded array state describes the layout (ES 3.0 6.1.12).
                mgl::glBindVertexArray(layout_vao_[i]);
                const LayoutExpect& e = kLayouts[i];
                bool state_ok = true;
                std::string detail;
                for (const GLuint a : {kPositionLoc, kValueLoc})
                {
                    const ArrayExpect& x = a == kPositionLoc ? e.position : e.color;
                    const bool ok = attrib_i(a, mgl::VertexAttribParameter::ArrayStride) == x.stride &&
                                    attrib_i(a, mgl::VertexAttribParameter::ArrayType) == static_cast<GLint>(x.type) &&
                                    (attrib_i(a, mgl::VertexAttribParameter::ArrayNormalized) != 0) == x.normalized &&
                                    attrib_pointer(a) == demo::buffer_offset(x.offset);
                    if (!ok) detail += std::format("attribute {} state differs; ", a);
                    state_ok = state_ok && ok;
                }
                const GLint b0 = attrib_i(kPositionLoc, mgl::VertexAttribParameter::ArrayBufferBinding);
                const GLint b1 = attrib_i(kValueLoc, mgl::VertexAttribParameter::ArrayBufferBinding);
                if ((b0 == b1) != e.same_buffer || b0 == 0 || b1 == 0) { state_ok = false; detail += "buffer bindings differ; "; }
                check.expect(state_ok, std::format("layout state: {} - stride / type / normalized / offset / buffer", e.name),
                             detail);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void check_constants(demo::Check& check)
        {
            for (std::size_t i = 0; i < kConsts.size(); ++i)
            {
                const ConstCase& c = kConsts[i];
                const auto want = expand(c.n, c.v);
                const std::string form = std::format("glVertexAttrib{}f{}", c.n, c.vector_form ? "v" : "");
                const Rect r = cell(c.col, c.row);
                check.expect_pixel(r.cx(), r.cy(), shaded(want, {1, 1, 1, 1}, {0, 0, 0, 0}), 1,
                                   std::format("{}: disabled array shows the constant, missing components (0, 0, 0, 1)", form));
                bool same = true;
                for (std::size_t k = 0; k < 4; ++k) same = same && static_cast<double>(observed_current_[i][k]) == want[k];
                check.expect(same, std::format("{}: CURRENT_VERTEX_ATTRIB holds the expanded vector exactly", form),
                             std::format("got ({}, {}, {}, {})", observed_current_[i][0], observed_current_[i][1],
                                         observed_current_[i][2], observed_current_[i][3]));
            }
            const Rect enabled = cell(4, 6), back = cell(5, 6);
            check.expect_pixel(enabled.cx(), enabled.cy(), rgb_expected(kArrayColor), 1,
                               "an enabled array overrides the constant");
            const auto last = kConsts.back();
            check.expect_pixel(back.cx(), back.cy(), shaded(expand(last.n, last.v), {1, 1, 1, 1}, {0, 0, 0, 0}), 1,
                               "disabling the array brings the constant back: drawing from an array did not change it");
            const auto now = current_attrib(kValueLoc);
            check.expect(now == last.v, "CURRENT_VERTEX_ATTRIB unchanged after drawing from an enabled array",
                         std::format("got ({}, {}, {}, {})", now[0], now[1], now[2], now[3]));
        }

        void check_named_attributes(demo::Check& check)
        {
            const Rect cb = cell(3, 7), ca = cell(4, 7);
            check.expect_pixel(cb.cx(), cb.cy(), rgb_expected(kBoundColor), 1,
                               "glBindAttribLocation(a_position 5, a_color 2): drawing with those slots works");
            check.expect_pixel(ca.cx(), ca.cy(), rgb_expected(kAutoColor), 1,
                               "linker-assigned locations (from glGetAttribLocation): drawing works");

            // -- explicit bindings take effect at link time and are reported by glGetAttribLocation
            auto where = [](const Prog& p, const char* name) { return mgl::glGetAttribLocation(p.program, name).value; };
            check.expect_eq(where(bound_, "a_position"), 5, "glGetAttribLocation(a_position) == the index bound before the link");
            check.expect_eq(where(bound_, "a_color"), 2, "glGetAttribLocation(a_color) == the index bound before the link");
            check.expect_eq(where(bound_, "a_unused"), -1,
                            "an unused attribute is not active: -1, although glBindAttribLocation(a_unused, 9) was issued");
            check.expect_eq(where(bound_, "no_such_input"), -1, "a name the shader does not contain: -1");
            check.expect_eq(program_param(bound_.program, mgl::ProgramParameter::ActiveAttributes), 2,
                            "ACTIVE_ATTRIBUTES == 2 (a_unused is optimized away)");

            // -- linker-chosen locations: distinct, inside [0, MAX_VERTEX_ATTRIBS)
            check.expect(auto_position_.value >= 0 && auto_position_.value < max_attribs_ && auto_color_.value >= 0 &&
                             auto_color_.value < max_attribs_ && auto_position_.value != auto_color_.value,
                         "linker-assigned locations are two distinct values in [0, MAX_VERTEX_ATTRIBS)");
            check.expect_eq(where(auto_, "a_unused"), -1, "unused attribute has no location in the auto program either");

            // -- glBindAttribLocation takes effect at the NEXT link only (ES 3.0 2.12.5)
            const std::array<Binding, 2> first = {{{5, "a_position"}, {2, "a_color"}}};
            Prog p = make_program(named_vs_src_, frag_src_, first, "rebind");
            check.expect_eq(where(p, "a_position"), 5, "scratch program: a_position at the first binding");
            mgl::glBindAttribLocation(p.program, loc(7), "a_position");
            check.expect_eq(where(p, "a_position"), 5, "re-binding a linked program changes nothing until glLinkProgram");
            mgl::glLinkProgram(p.program);
            check.expect_eq(where(p, "a_position"), 7, "after the next glLinkProgram the new binding applies");
            destroy(p);

            // -- a layout(location) qualifier in the shader beats glBindAttribLocation (ES 3.0 2.12.5)
            const std::array<Binding, 1> conflicting = {{{9, "a_position"}}};
            p = make_program(attrib_vs_src_, frag_src_, conflicting, "layout wins");
            check.expect(p.linked, "layout(location) + conflicting glBindAttribLocation still links");
            check.expect_eq(where(p, "a_position"), 0, "layout(location = 0) in the shader overrides glBindAttribLocation(9)");
            check.expect_eq(where(p, "a_value"), 1, "layout(location = 1) reported by glGetAttribLocation");
            destroy(p);

            // -- aliasing two attributes to one slot is a link error in GLSL ES 3.00 (ES 3.0 2.12.5)
            const std::array<Binding, 2> alias = {{{4, "a_position"}, {4, "a_color"}}};
            p = make_program(named_vs_src_, frag_src_, alias, "alias");
            check.expect(!p.linked, "two active attributes bound to the same index: LINK_STATUS == FALSE");
            check.expect_no_gl_error("a failed link is not a GL error");
            // ... and a program that last failed to link has no attribute locations (INVALID_OPERATION, result -1).
            const GLint failed_loc = where(p, "a_position");
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetAttribLocation on an unsuccessfully linked program -> INVALID_OPERATION");
            check.expect_eq(failed_loc, -1, "... and returns -1");
            destroy(p);

            // -- argument errors of glBindAttribLocation
            mgl::glBindAttribLocation(bound_.program, loc(0), "gl_Vertex");
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBindAttribLocation with a reserved gl_ name -> INVALID_OPERATION");
            mgl::glBindAttribLocation(bound_.program, loc(static_cast<GLuint>(max_attribs_)), "a_position");
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindAttribLocation(index == MAX_VERTEX_ATTRIBS) -> INVALID_VALUE");
        }

        void check_active_attributes(demo::Check& check)
        {
            struct Expected
            {
                const char* name;
                mgl::UniformType type;
                const char* type_name;
                int slots;
            };
            const std::array<Expected, 7> expected = {{
                {"a_weight", mgl::UniformType::Float, "FLOAT", 1},
                {"a_position", mgl::UniformType::FloatVec2, "FLOAT_VEC2", 1},
                {"a_tint", mgl::UniformType::FloatVec3, "FLOAT_VEC3", 1},
                {"a_value", mgl::UniformType::FloatVec4, "FLOAT_VEC4", 1},
                {"a_basis", mgl::UniformType::FloatMat2, "FLOAT_MAT2", 2},
                {"a_cell", mgl::UniformType::IntVec2, "INT_VEC2", 1},
                {"a_id", mgl::UniformType::UnsignedInt, "UNSIGNED_INT", 1},
            }};

            const mgl::ProgramId p = enum_.program;
            const GLint active = program_param(p, mgl::ProgramParameter::ActiveAttributes);
            check.expect_eq(active, static_cast<GLint>(expected.size()),
                            "ACTIVE_ATTRIBUTES counts the 7 used inputs (a_unused is not active)");
            std::size_t longest = 0;
            for (const Expected& e : expected) longest = std::max(longest, std::strlen(e.name));
            check.expect_eq(program_param(p, mgl::ProgramParameter::ActiveAttributeMaxLength), static_cast<GLint>(longest + 1),
                            "ACTIVE_ATTRIBUTE_MAX_LENGTH == longest name + terminator");

            std::array<bool, 7> seen{};
            std::vector<int> owner(static_cast<std::size_t>(max_attribs_), -1);  // which attribute holds each slot
            bool slots_ok = true;
            for (GLint i = 0; i < active; ++i)
            {
                // The ActiveAttribIndex overload: `i` is an index into the list of active attributes, which is a
                // different namespace from attribute locations (ES 3.0 2.12.5: "index has no relation to the
                // generic attribute"). See docs/META_GL_GAPS.md O5.
                std::array<char, 64> name{};
                GLsizei length = -1;
                GLint size = -1;
                mgl::UniformType type = mgl::UniformType::Sampler2D;  // not a valid attribute type: detects "unmodified"
                mgl::glGetActiveAttrib(p, mgl::ActiveAttribIndex{static_cast<GLuint>(i)}, static_cast<GLsizei>(name.size()),
                                       &length, &size, &type, name.data());
                const std::string_view got(name.data(), static_cast<std::size_t>(std::max(length, 0)));
                const auto hit = std::find_if(expected.begin(), expected.end(),
                                              [&](const Expected& e) { return got == e.name; });
                if (hit == expected.end())
                {
                    check.expect(false, std::format("active attribute index {} is a known name", i), std::format("'{}'", got));
                    continue;
                }
                const auto k = static_cast<std::size_t>(hit - expected.begin());
                seen[k] = true;
                const GLint location = mgl::glGetAttribLocation(p, hit->name).value;
                check.expect(type == hit->type && size == 1 && static_cast<std::size_t>(length) == std::strlen(hit->name) &&
                                 location >= 0 && location + hit->slots <= max_attribs_,
                             std::format("glGetActiveAttrib(index {}): {} is {} x1 at location {}", i, hit->name, hit->type_name, location),
                             std::format("type={} size={} length={}", static_cast<GLenum>(type), size, length));
                for (int s = 0; s < hit->slots; ++s)
                {
                    int& slot = owner[static_cast<std::size_t>(std::clamp(location + s, 0, max_attribs_ - 1))];
                    if (slot != -1) slots_ok = false;
                    slot = static_cast<int>(k);
                }
            }
            check.expect(std::all_of(seen.begin(), seen.end(), [](bool b) { return b; }),
                         "the enumeration visits every expected attribute exactly once (order is unspecified)");
            check.expect(slots_ok, "no two attributes share a slot; the mat2 occupies two consecutive slots");
            check.expect_eq(mgl::glGetAttribLocation(p, "a_basis").value, 4, "a_basis was bound to 4 (and so uses slots 4, 5)");
            check.expect_eq(mgl::glGetAttribLocation(p, "a_unused").value, -1, "a_unused: -1");

            // index == ACTIVE_ATTRIBUTES is out of range: INVALID_VALUE and the outputs stay untouched.
            std::array<char, 8> name{'#', '#', '#', '#', '#', '#', '#', '#'};
            GLsizei length = -7;
            GLint size = -7;
            mgl::UniformType type = mgl::UniformType::Sampler2D;
            mgl::glGetActiveAttrib(p, mgl::ActiveAttribIndex{static_cast<GLuint>(active)}, static_cast<GLsizei>(name.size()),
                                   &length, &size, &type, name.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetActiveAttrib(index == ACTIVE_ATTRIBUTES) -> INVALID_VALUE");
            check.expect(length == -7 && size == -7 && type == mgl::UniformType::Sampler2D && name[0] == '#',
                         "... and length, size, type and name are left unmodified");
        }

        void check_client_arrays(demo::Check& check)
        {
            const Rect a = cell(3, 4), b = cell(3, 5), m = cell(3, 6);
            check.expect_pixel(a.cx(), a.cy(), rgb_expected(kClientA), 1, "client-memory arrays on the default VAO draw correctly");
            check.expect_pixel(b.cx(), b.cy(), rgb_expected(kClientB), 1,
                               "client memory is read at draw time: edited CPU array, same pointers, new color");
            check.expect_pixel(m.cx(), m.cy(), rgb_expected(kMixed), 1,
                               "one draw call mixing a buffer-object array and a client-memory array");

            // State of client arrays: the pointer is the CPU address, the buffer binding is 0 (ES 3.0 6.1.12:
            // "If the zero object is bound, these values are client state").
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, quad_vbo_);
            mgl::glVertexAttribPointer(loc(kPositionLoc), 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, mgl::DataType::Float, GL_FALSE, 0, client_colors_.data());
            check.expect(attrib_pointer(kValueLoc) == client_colors_.data(),
                         "glGetVertexAttribPointerv returns the client memory address on VAO 0");
            check.expect_eq(attrib_i(kValueLoc, mgl::VertexAttribParameter::ArrayBufferBinding), 0,
                            "client array: VERTEX_ATTRIB_ARRAY_BUFFER_BINDING == 0");
            check.expect_eq(attrib_i(kPositionLoc, mgl::VertexAttribParameter::ArrayBufferBinding),
                            static_cast<GLint>(quad_vbo_.value), "buffer array next to it: binding is the quad VBO");
            reset_default_vao_arrays();
            check.expect(attrib_pointer(kValueLoc) == nullptr, "resetting with a NULL pointer restores the initial client state");
            check.expect_no_gl_error("client array setup on VAO 0 raised no GL error");

            // The same call on a NON-default VAO is an error, a NULL pointer is not (ES 3.0 2.9, footnote 5).
            mgl::VertexArrayId scratch{};
            mgl::glGenVertexArrays(1, &scratch);
            mgl::glBindVertexArray(scratch);
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, mgl::DataType::Float, GL_FALSE, 0, client_colors_.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "client pointer with a non-zero VAO and no ARRAY_BUFFER -> INVALID_OPERATION");
            // ES 3.0 2.5 also says that a command which generates an error "has no effect on GL state". Mesa 25.0
            // records the pointer before it rejects the call, so this is reported as a note, not asserted.
            if (attrib_pointer(kValueLoc) != nullptr)
                std::cout << "     note: driver deviation - the failed glVertexAttribPointer still stored its pointer "
                             "(ES 3.0 2.5: an erroring command has no effect on GL state)\n";
            mgl::glVertexAttribPointer(loc(kValueLoc), 3, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            check.expect_no_gl_error("a NULL pointer with a non-zero VAO and no ARRAY_BUFFER is allowed");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDeleteVertexArrays(1, &scratch);
        }

        void check_state_queries(demo::Check& check)
        {
            constexpr GLuint kIdx = 3;
            std::array<mgl::VertexArrayId, 2> vaos{};
            std::array<mgl::BufferId, 2> bufs{};
            mgl::glGenVertexArrays(2, vaos.data());
            mgl::glGenBuffers(2, bufs.data());
            // A generated name is not a vertex array object until it is first bound (ES 3.0 2.11, 6.1.10).
            check.expect(!mgl::glIsVertexArray(vaos[0]), "glIsVertexArray is FALSE for a generated but never bound name");
            const std::vector<std::uint8_t> zeros(256, 0);
            for (const mgl::BufferId b : bufs)
            {
                mgl::glBindBuffer(mgl::BufferTarget::Array, b);
                mgl::glBufferData(mgl::BufferTarget::Array, std::span<const std::uint8_t>(zeros), mgl::BufferUsage::StaticDraw);
            }
            using P = mgl::VertexAttribParameter;

            // -- initial state of a brand-new VAO (ES 3.0 Table 6.2 and the end of 2.9)
            mgl::glBindVertexArray(vaos[0]);
            check.expect(mgl::glIsVertexArray(vaos[0]), "glIsVertexArray is TRUE once the name has been bound");
            check.expect(attrib_i(kIdx, P::ArrayEnabled) == GL_FALSE && attrib_i(kIdx, P::ArraySize) == 4 &&
                             attrib_i(kIdx, P::ArrayStride) == 0 &&
                             attrib_i(kIdx, P::ArrayType) == static_cast<GLint>(mgl::DataType::Float) &&
                             attrib_i(kIdx, P::ArrayNormalized) == GL_FALSE && attrib_i(kIdx, P::ArrayInteger) == GL_FALSE &&
                             attrib_i(kIdx, P::ArrayDivisor) == 0 && attrib_i(kIdx, P::ArrayBufferBinding) == 0 &&
                             attrib_pointer(kIdx) == nullptr,
                         "initial array state: disabled, size 4, stride 0, FLOAT, not normalized, divisor 0, no buffer, NULL");

            // -- state written by glVertexAttribPointer / glEnableVertexAttribArray
            mgl::glBindBuffer(mgl::BufferTarget::Array, bufs[0]);
            mgl::glVertexAttribPointer(loc(kIdx), 2, mgl::DataType::UnsignedShort, GL_TRUE, 12, demo::buffer_offset(20));
            mgl::glEnableVertexAttribArray(loc(kIdx));
            check.expect_eq(attrib_i(kIdx, P::ArrayEnabled), GL_TRUE, "VERTEX_ATTRIB_ARRAY_ENABLED after glEnableVertexAttribArray");
            check.expect_eq(attrib_i(kIdx, P::ArraySize), 2, "VERTEX_ATTRIB_ARRAY_SIZE == 2");
            check.expect_eq(attrib_i(kIdx, P::ArrayStride), 12, "VERTEX_ATTRIB_ARRAY_STRIDE == 12");
            check.expect_eq(attrib_i(kIdx, P::ArrayType), static_cast<GLint>(mgl::DataType::UnsignedShort),
                            "VERTEX_ATTRIB_ARRAY_TYPE == UNSIGNED_SHORT");
            check.expect_eq(attrib_i(kIdx, P::ArrayNormalized), GL_TRUE, "VERTEX_ATTRIB_ARRAY_NORMALIZED == TRUE");
            check.expect_eq(attrib_i(kIdx, P::ArrayInteger), GL_FALSE, "VERTEX_ATTRIB_ARRAY_INTEGER == FALSE (glVertexAttribPointer)");
            check.expect_eq(attrib_i(kIdx, P::ArrayBufferBinding), static_cast<GLint>(bufs[0].value),
                            "VERTEX_ATTRIB_ARRAY_BUFFER_BINDING == the buffer bound at the pointer call");
            check.expect(attrib_pointer(kIdx) == demo::buffer_offset(20), "VERTEX_ATTRIB_ARRAY_POINTER == the byte offset 20");
            GLfloat as_float = -1.0f;
            mgl::glGetVertexAttribfv(loc(kIdx), P::ArraySize, &as_float);
            check.expect_near(static_cast<double>(as_float), 2.0, 0.0, "glGetVertexAttribfv returns the same state as a float");

            // -- the ARRAY_BUFFER binding is not VAO state; the recorded buffer does not follow it
            mgl::glBindBuffer(mgl::BufferTarget::Array, bufs[1]);
            check.expect_eq(attrib_i(kIdx, P::ArrayBufferBinding), static_cast<GLint>(bufs[0].value),
                            "re-binding ARRAY_BUFFER later does not change the recorded buffer");

            // -- per-VAO state: a second VAO starts from the initial state, the first one keeps its own
            mgl::glBindVertexArray(vaos[1]);
            check.expect(attrib_i(kIdx, P::ArrayEnabled) == GL_FALSE && attrib_i(kIdx, P::ArraySize) == 4 &&
                             attrib_i(kIdx, P::ArrayBufferBinding) == 0,
                         "a second VAO does not see the first VAO's attribute state");
            mgl::glVertexAttribPointer(loc(kIdx), 3, mgl::DataType::Byte, GL_FALSE, 0, demo::buffer_offset(8));
            check.expect(attrib_i(kIdx, P::ArraySize) == 3 && attrib_i(kIdx, P::ArrayType) == static_cast<GLint>(mgl::DataType::Byte) &&
                             attrib_i(kIdx, P::ArrayNormalized) == GL_FALSE &&
                             attrib_i(kIdx, P::ArrayBufferBinding) == static_cast<GLint>(bufs[1].value) &&
                             attrib_pointer(kIdx) == demo::buffer_offset(8),
                         "second VAO: size 3, BYTE, not normalized, its own buffer, offset 8");
            mgl::glBindVertexArray(vaos[0]);
            check.expect(attrib_i(kIdx, P::ArrayEnabled) == GL_TRUE && attrib_i(kIdx, P::ArraySize) == 2 &&
                             attrib_i(kIdx, P::ArrayType) == static_cast<GLint>(mgl::DataType::UnsignedShort),
                         "switching back to the first VAO restores its attribute state");
            mgl::glDisableVertexAttribArray(loc(kIdx));
            check.expect(attrib_i(kIdx, P::ArrayEnabled) == GL_FALSE && attrib_i(kIdx, P::ArraySize) == 2,
                         "glDisableVertexAttribArray clears only the enable flag");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect(attrib_i(kIdx, P::ArraySize) == 4 && attrib_i(kIdx, P::ArrayBufferBinding) == 0,
                         "the default VAO has state of its own (untouched at index 3)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ArrayBufferBinding), static_cast<int>(bufs[1].value),
                            "ARRAY_BUFFER_BINDING is context state (Table 6.3): it stayed bound through all VAO switches");

            // -- the current value is context state, not VAO state (ES 3.0 6.1.12; Table 6.22 "Vertex Shader State (not part of program objects)")
            constexpr GLuint kCur = 6;
            mgl::glBindVertexArray(vaos[0]);
            mgl::glVertexAttrib4f(loc(kCur), 3.0f, 5.0f, 7.0f, 9.0f);
            mgl::glBindVertexArray(vaos[1]);
            const auto f = current_attrib(kCur);
            check.expect(f == std::array<float, 4>{3.0f, 5.0f, 7.0f, 9.0f},
                         "CURRENT_VERTEX_ATTRIB (glGetVertexAttribfv) is the same under another VAO");
            std::array<GLint, 4> iv = {-1, -1, -1, -1};
            mgl::glGetVertexAttribiv(loc(kCur), P::CurrentVertexAttrib, iv.data());
            check.expect(iv == std::array<GLint, 4>{3, 5, 7, 9}, "CURRENT_VERTEX_ATTRIB (glGetVertexAttribiv) converts to integers");

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteVertexArrays(2, vaos.data());
            mgl::glDeleteBuffers(2, bufs.data());
            check.expect(!mgl::glIsVertexArray(vaos[0]) && !mgl::glIsVertexArray(vaos[1]),
                         "glIsVertexArray is FALSE after glDeleteVertexArrays");
        }

        void check_attribute_errors(demo::Check& check)
        {
            // Every command taking an attribute index rejects index >= MAX_VERTEX_ATTRIBS with INVALID_VALUE.
            const mgl::AttribLocation bad = loc(static_cast<GLuint>(max_attribs_));
            mgl::glVertexAttrib4f(bad, 1, 2, 3, 4);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttrib4f(index == MAX_VERTEX_ATTRIBS) -> INVALID_VALUE");
            mgl::glEnableVertexAttribArray(bad);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glEnableVertexAttribArray(index == MAX_VERTEX_ATTRIBS) -> INVALID_VALUE");
            mgl::glVertexAttribPointer(bad, 3, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer(index == MAX_VERTEX_ATTRIBS) -> INVALID_VALUE");
            GLint untouched = 42;
            mgl::glGetVertexAttribiv(bad, mgl::VertexAttribParameter::ArraySize, &untouched);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetVertexAttribiv(index == MAX_VERTEX_ATTRIBS) -> INVALID_VALUE");
            // ES 3.0 2.5: "If the generating command modifies values through a pointer argument, no change is made
            // to these values." Mesa 25.0 stores 0 anyway, so this is reported as a note, not asserted.
            if (untouched != 42)
                std::cout << std::format("     note: driver deviation - the failed glGetVertexAttribiv overwrote its output "
                                         "with {} (ES 3.0 2.5: outputs stay unmodified on error)\n", untouched);

            // Size and stride constraints of glVertexAttribPointer (ES 3.0 2.9, Table 2.4).
            mgl::glVertexAttribPointer(loc(0), 5, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer(size 5) -> INVALID_VALUE");
            mgl::glVertexAttribPointer(loc(0), 3, mgl::DataType::Float, GL_FALSE, -4, nullptr);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer(negative stride) -> INVALID_VALUE");
        }

        // ---- members --------------------------------------------------------------------------
        int pitch_ = 32;
        int max_attribs_ = 16;
        std::string attrib_vs_src_, named_vs_src_, enumerate_vs_src_, frag_src_;

        mgl::ProgramId program_{};
        mgl::UniformLocation u_gain_{}, u_bias_{};
        Prog bound_{}, auto_{}, enum_{};
        mgl::AttribLocation auto_position_{}, auto_color_{};

        mgl::BufferId quad_vbo_{}, format_vbo_{}, sizes_vbo_{}, array_color_vbo_{}, named_color_vbo_{};
        std::array<mgl::BufferId, 5> layout_vbo_{};
        mgl::VertexArrayId vao_{}, vao_bound_{}, vao_auto_{};
        std::array<mgl::VertexArrayId, 4> layout_vao_{};

        struct FormatBlock
        {
            std::size_t offset;
            GLsizei stride;
        };
        std::vector<FormatBlock> format_layout_;
        std::vector<std::size_t> size_offset_;

        std::array<float, 8> client_positions_{};
        std::array<float, 12> client_colors_{};
        std::array<std::array<float, 4>, 8> observed_current_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-vertex-attributes",
        .title = "Vertex attributes",
        .min_es = {3, 0},
        .summary = "Attribute data types and normalization, component counts, interleaved vs planar layouts, "
                   "constant attribute values, client-memory arrays, attribute locations and state queries.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<VertexAttributes>(argc, argv, info);
}
