// es32-base-vertex-and-robustness
//
// Two OpenGL ES 3.2 additions that both deal with "where does the GL read from":
//
// 1. Base vertex drawing (ES 3.2 section 10.5). glDrawElementsBaseVertex, glDrawRangeElementsBaseVertex and
//    glDrawElementsInstancedBaseVertex add `basevertex` to every index before the vertex is fetched. Ten small
//    meshes share ONE vertex buffer and ONE element array buffer that holds a single quad pattern
//    {0,1,2, 0,2,3} (as UNSIGNED_BYTE, UNSIGNED_SHORT and UNSIGNED_INT) plus one triangle strip. Every draw
//    picks its mesh only through basevertex. The scene is an 8 x 8 grid of 32-pixel cells:
//      row 6  meshes 1-3 (glDrawElementsBaseVertex with ubyte / ushort / uint indices), mesh 4 (Range)
//      row 5  mesh 5 drawn 4 times by glDrawElementsInstancedBaseVertex; an instanced attribute moves and
//             tints each instance (instanced attributes are NOT shifted by basevertex)
//      row 4  vertex-ID cells: the shader compares gl_VertexID with the vertex's own position in the buffer
//             (green = equal). ES 3.2 defines the vertex ID as index + basevertex.
//      row 3  primitive restart with basevertex: the restart comparison happens BEFORE basevertex is added
//      row 2  robust buffer access: two draws whose basevertex points far behind the end of the buffer,
//             confined by the scissor test, then a normal quad over the same cell
//      row 7  cell (0,7) holds mesh 0 (basevertex 0). Nothing draws it, so it must stay the clear colour:
//             if basevertex were ignored every draw would land there (built-in negative control).
//
// 2. Robustness (ES 3.2 sections 2.3.2, 7.12, 16.1.2, 20.2): the context is created with robust buffer access
//    and LOSE_CONTEXT_ON_RESET (demo::Info::robust_context -> EGL 1.5 EGL_CONTEXT_OPENGL_ROBUST_ACCESS and
//    EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY). validate() checks CONTEXT_FLAGS, the reset status and
//    strategy, and the bounded queries glReadnPixels / glGetnUniform{f,i,ui}v with canary-filled buffers.
//
// What validate() proves:
//   * every cell of the grid against a CPU model that expands the same index data with index + basevertex,
//     primitive restart before basevertex, instanced attributes fetched by instance number, and rasterizes
//     the triangles at pixel centres (probes closer than 1.5 px to any edge are skipped)
//   * a mutation of the CPU model (basevertex ignored) disagrees with the image in every drawn cell, and a GPU
//     control: glDrawElements and glDrawElementsBaseVertex(basevertex 0) do draw mesh 0 into cell (0,7)
//   * INVALID_VALUE for end < start and negative count / instancecount, and the erroneous draws have no effect
//   * glReadnPixels: exact bufSize succeeds and matches glReadPixels; bufSize one byte short, 0 and negative fail
//     without touching the buffer; nothing is written past bufSize (canary bytes)
//   * glGetnUniform{f,i,ui}v on vec4 / ivec3 / uvec2 / mat2: the same rules, plus an invalid location
//   * GetGraphicsResetStatus == NO_ERROR, RESET_NOTIFICATION_STRATEGY == LOSE_CONTEXT_ON_RESET (as requested),
//     CONTEXT_FLAG_ROBUST_ACCESS_BIT set
//   * robust access: the out-of-bounds draws generate no error, cause no reset, and later rendering is correct.
//     The attribute VALUES they read are undefined in ES 3.2 section 10.3.5 and are not asserted.

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
#include <string>
#include <vector>

namespace mgl = metagl;

namespace
{
    // ---------------------------------------------------------------------------------------------------
    // Shared vertex buffer: 260 vertices. Mesh k (k = 0..9) is the quad at vertices 4k .. 4k+3. The strip
    // quads A and B sit at 248..251 and 252..255 so that UNSIGNED_BYTE indices reach them. Everything else is
    // filler parked outside the viewport.
    // ---------------------------------------------------------------------------------------------------
    struct Vertex
    {
        float x, y;
        float r, g, b;
        std::int32_t index;  // the vertex's own position in the buffer (gl_VertexID check)
    };

    struct Instance
    {
        float dx, dy, tint;
    };

    struct Color
    {
        float r, g, b;
    };

    constexpr int kVertexCount = 260;
    constexpr int kGrid = 8;
    constexpr float kCell = 2.0f / kGrid;    // cell size in NDC (32 px at 256 x 256)
    constexpr float kInset = kCell / 8.0f;   // quads leave a 4 px border inside their cell
    constexpr std::array<float, 3> kClear = {0.08f, 0.08f, 0.12f};

    constexpr int kStripA = 248;  // strip quad A: vertices 248..251, cell (0,3)
    constexpr int kStripB = 252;  // strip quad B: vertices 252..255, cell (1,3)

    struct MeshDef
    {
        int col, row;
        Color color;
    };

    // Mesh k -> cell and colour. Colours are multiples of 0.25 so that products with the tints are exact.
    constexpr std::array<MeshDef, 10> kMeshes = {{
        {0, 7, {1.00f, 0.00f, 0.00f}},  // 0: never drawn (negative control)
        {0, 6, {0.00f, 0.50f, 1.00f}},  // 1: DrawElementsBaseVertex, ubyte
        {1, 6, {1.00f, 0.50f, 0.00f}},  // 2: DrawElementsBaseVertex, ushort
        {2, 6, {0.50f, 1.00f, 0.25f}},  // 3: DrawElementsBaseVertex, uint
        {3, 6, {1.00f, 1.00f, 0.00f}},  // 4: DrawRangeElementsBaseVertex
        {0, 5, {0.00f, 1.00f, 1.00f}},  // 5: DrawElementsInstancedBaseVertex, 4 instances
        {0, 4, {0.50f, 0.50f, 0.50f}},  // 6: vertex-ID check, DrawElementsBaseVertex
        {1, 4, {0.50f, 0.50f, 0.50f}},  // 7: vertex-ID check, DrawRangeElementsBaseVertex
        {2, 4, {0.50f, 0.50f, 0.50f}},  // 8: vertex-ID check, DrawElementsInstancedBaseVertex (2 instances)
        {0, 2, {0.75f, 0.25f, 1.00f}},  // 9: drawn over the robust-access cell
    }};
    constexpr Color kStripColorA = {0.25f, 0.50f, 1.00f};
    constexpr Color kStripColorB = {1.00f, 0.25f, 0.50f};
    constexpr int kRobustCol = 0, kRobustRow = 2;

    // Instance 0 is the identity: non-instanced draws fetch instanced attributes for instance 0.
    constexpr std::array<Instance, 4> kInstances = {{
        {0.00f, 0.0f, 1.00f},
        {0.25f, 0.0f, 0.75f},
        {0.50f, 0.0f, 0.50f},
        {0.75f, 0.0f, 0.25f},
    }};

    float cell_x(int col) { return -1.0f + static_cast<float>(col) * kCell; }
    float cell_y(int row) { return -1.0f + static_cast<float>(row) * kCell; }

    // Quad corners. Triangle-list order BL, BR, TR, TL (indices {0,1,2, 0,2,3});
    // triangle-strip order BL, BR, TL, TR.
    void put_quad(std::vector<Vertex>& v, int first, int col, int row, Color c, bool strip_order)
    {
        const float x0 = cell_x(col) + kInset, x1 = cell_x(col) + kCell - kInset;
        const float y0 = cell_y(row) + kInset, y1 = cell_y(row) + kCell - kInset;
        const std::array<std::array<float, 2>, 4> list = {{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
        const std::array<std::array<float, 2>, 4> strip = {{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}}};
        for (int i = 0; i < 4; ++i)
        {
            const auto& p = strip_order ? strip[static_cast<std::size_t>(i)] : list[static_cast<std::size_t>(i)];
            v[static_cast<std::size_t>(first + i)] = {p[0], p[1], c.r, c.g, c.b, first + i};
        }
    }

    std::vector<Vertex> build_vertices()
    {
        std::vector<Vertex> v(kVertexCount);
        for (int i = 0; i < kVertexCount; ++i)
            v[static_cast<std::size_t>(i)] = {-2.0f, -2.0f, 1.0f, 0.0f, 1.0f, i};  // filler: outside the viewport
        for (std::size_t k = 0; k < kMeshes.size(); ++k)
            put_quad(v, static_cast<int>(4 * k), kMeshes[k].col, kMeshes[k].row, kMeshes[k].color, false);
        put_quad(v, kStripA, 0, 3, kStripColorA, true);
        put_quad(v, kStripB, 1, 3, kStripColorB, true);
        return v;
    }

    // ---------------------------------------------------------------------------------------------------
    // Element array buffer (56 bytes). Offsets are multiples of the index size.
    // ---------------------------------------------------------------------------------------------------
    constexpr std::size_t kOffQuadU8 = 0;    // ubyte  {0,1,2,0,2,3}           bytes  0..5
    constexpr std::size_t kOffQuadU16 = 8;   // ushort {0,1,2,0,2,3}           bytes  8..19
    constexpr std::size_t kOffQuadU32 = 20;  // uint   {0,1,2,0,2,3}           bytes 20..43
    constexpr std::size_t kOffStripU8 = 44;  // ubyte  strip with restart      bytes 44..52
    constexpr std::size_t kIndexBytes = 56;

    // The strip, drawn with basevertex 1: 247..250 -> vertices 248..251 (quad A), 255 = restart index of
    // UNSIGNED_BYTE (compared BEFORE basevertex is added), 251..254 -> vertices 252..255 (quad B). Note that
    // 254 + 1 = 255: if the comparison happened after adding basevertex, quad B would lose its second
    // triangle; and 255 + 1 = 256 would not restart at all.
    constexpr std::array<std::uint8_t, 9> kStrip = {247, 248, 249, 250, 255, 251, 252, 253, 254};

    std::array<std::uint8_t, kIndexBytes> build_indices()
    {
        std::array<std::uint8_t, kIndexBytes> bytes{};
        constexpr std::array<std::uint32_t, 6> quad = {0, 1, 2, 0, 2, 3};
        for (std::size_t i = 0; i < quad.size(); ++i)
        {
            const auto u8 = static_cast<std::uint8_t>(quad[i]);
            const auto u16 = static_cast<std::uint16_t>(quad[i]);
            const std::uint32_t u32 = quad[i];
            std::memcpy(&bytes[kOffQuadU8 + i], &u8, 1);
            std::memcpy(&bytes[kOffQuadU16 + 2 * i], &u16, 2);
            std::memcpy(&bytes[kOffQuadU32 + 4 * i], &u32, 4);
        }
        std::memcpy(&bytes[kOffStripU8], kStrip.data(), kStrip.size());
        return bytes;
    }

    // ---------------------------------------------------------------------------------------------------
    // The draws of one frame. render() issues them (switch on `cmd`) and the CPU model expands the same
    // records, so the expectation cannot drift from what was drawn.
    // ---------------------------------------------------------------------------------------------------
    enum class Cmd
    {
        BaseVertex,           // glDrawElementsBaseVertex
        RangeBaseVertex,      // glDrawRangeElementsBaseVertex
        InstancedBaseVertex,  // glDrawElementsInstancedBaseVertex
    };

    struct Draw
    {
        const char* what;
        Cmd cmd;
        mgl::PrimitiveType mode;
        mgl::IndexType type;
        std::size_t offset;
        int count;
        int basevertex;
        int instances = 1;               // InstancedBaseVertex
        GLuint start = 0, end = 0;       // RangeBaseVertex: bounds of the indices BEFORE basevertex
        int u_mode = 0;                  // 0 colour, 1 vertex-ID check
        bool restart = false;            // PRIMITIVE_RESTART_FIXED_INDEX
        bool out_of_bounds = false;      // robust-access draw: scissored to the robust cell, not modelled
    };

    using PT = mgl::PrimitiveType;
    using IT = mgl::IndexType;

    const std::array<Draw, 12> kDraws = {{
        {"mesh 1, glDrawElementsBaseVertex, UNSIGNED_BYTE", Cmd::BaseVertex, PT::Triangles, IT::UnsignedByte, kOffQuadU8, 6, 4},
        {"mesh 2, glDrawElementsBaseVertex, UNSIGNED_SHORT", Cmd::BaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, 8},
        {"mesh 3, glDrawElementsBaseVertex, UNSIGNED_INT", Cmd::BaseVertex, PT::Triangles, IT::UnsignedInt, kOffQuadU32, 6, 12},
        {"mesh 4, glDrawRangeElementsBaseVertex [0,3]", Cmd::RangeBaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, 16, 1, 0, 3},
        {"mesh 5, glDrawElementsInstancedBaseVertex x4", Cmd::InstancedBaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, 20, 4},
        {"mesh 6, gl_VertexID via glDrawElementsBaseVertex", Cmd::BaseVertex, PT::Triangles, IT::UnsignedByte, kOffQuadU8, 6, 24, 1, 0, 0, 1},
        {"mesh 7, gl_VertexID via glDrawRangeElementsBaseVertex", Cmd::RangeBaseVertex, PT::Triangles, IT::UnsignedInt, kOffQuadU32, 6, 28, 1, 0, 3, 1},
        {"mesh 8, gl_VertexID via glDrawElementsInstancedBaseVertex x2", Cmd::InstancedBaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, 32, 2, 0, 0, 1},
        {"strip A+B, primitive restart + basevertex 1", Cmd::BaseVertex, PT::TriangleStrip, IT::UnsignedByte, kOffStripU8, 9, 1, 1, 0, 0, 0, true},
        {"robust: basevertex just past the end of the buffer", Cmd::BaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, kVertexCount, 1, 0, 0, 0, false, true},
        {"robust: basevertex 2^20 (24 MiB past the end)", Cmd::InstancedBaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, 1 << 20, 2, 0, 0, 0, false, true},
        {"mesh 9, drawn over the robust cell", Cmd::BaseVertex, PT::Triangles, IT::UnsignedShort, kOffQuadU16, 6, 36},
    }};

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

    demo::Rgba8 clear_rgba() { return {to8(kClear[0]), to8(kClear[1]), to8(kClear[2]), 255}; }

    // ---------------------------------------------------------------------------------------------------
    // CPU model: expands the draws exactly as ES 3.2 section 10.5 describes and rasterizes at pixel centres.
    // ---------------------------------------------------------------------------------------------------
    struct Tri
    {
        std::array<std::array<float, 2>, 3> p;  // window coordinates
        demo::Rgba8 color;
    };

    std::uint32_t read_index(const std::array<std::uint8_t, kIndexBytes>& bytes, IT type, std::size_t offset, int i)
    {
        const auto at = static_cast<std::size_t>(i);
        switch (type)
        {
        case IT::UnsignedByte: return bytes[offset + at];
        case IT::UnsignedShort: { std::uint16_t v; std::memcpy(&v, &bytes[offset + 2 * at], 2); return v; }
        case IT::UnsignedInt: { std::uint32_t v; std::memcpy(&v, &bytes[offset + 4 * at], 4); return v; }
        }
        return 0;
    }

    std::uint32_t restart_index(IT type)
    {
        switch (type)
        {
        case IT::UnsignedByte: return 0xFFu;
        case IT::UnsignedShort: return 0xFFFFu;
        case IT::UnsignedInt: return 0xFFFFFFFFu;
        }
        return 0;
    }

    // Deliberately wrong variants of the model, used to show that the pixel checks can fail.
    enum class Mutation
    {
        None,
        IgnoreBaseVertex,        // basevertex treated as 0
        RestartAfterBaseVertex,  // restart index compared with index + basevertex instead of the raw index
    };

    std::vector<Tri> model_triangles(int width, int height, Mutation mutation = Mutation::None)
    {
        const std::vector<Vertex> verts = build_vertices();
        const auto bytes = build_indices();
        std::vector<Tri> tris;
        for (const Draw& d : kDraws)
        {
            if (d.out_of_bounds) continue;  // undefined attribute values (section 10.3.5): confined by scissor
            const std::uint32_t base = mutation == Mutation::IgnoreBaseVertex ? 0u : static_cast<std::uint32_t>(d.basevertex);
            for (int inst = 0; inst < d.instances; ++inst)
            {
                // Instanced attribute: element floor(instance / 1) + baseinstance(0) - independent of basevertex.
                const Instance& in = kInstances[static_cast<std::size_t>(inst)];
                std::vector<std::vector<std::uint32_t>> runs(1);
                for (int i = 0; i < d.count; ++i)
                {
                    const std::uint32_t idx = read_index(bytes, d.type, d.offset, i);
                    // Compared BEFORE adding basevertex (section 10.3.4).
                    const std::uint32_t compared = mutation == Mutation::RestartAfterBaseVertex ? idx + base : idx;
                    if (d.restart && compared == restart_index(d.type))
                    {
                        runs.emplace_back();
                        continue;
                    }
                    runs.back().push_back(idx + base);  // vertex ID = index + basevertex
                }
                for (const auto& run : runs)
                {
                    auto make = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                        Tri t{};
                        const std::array<std::uint32_t, 3> ids = {a, b, c};
                        bool vertex_ids_ok = true;
                        for (std::size_t k = 0; k < 3; ++k)
                        {
                            const Vertex& v = verts.at(ids[k]);
                            t.p[k] = {(v.x + in.dx + 1.0f) * 0.5f * static_cast<float>(width),
                                      (v.y + in.dy + 1.0f) * 0.5f * static_cast<float>(height)};
                            vertex_ids_ok = vertex_ids_ok && static_cast<std::uint32_t>(v.index) == ids[k];
                        }
                        const Vertex& v0 = verts.at(a);
                        if (d.u_mode == 1)
                            t.color = vertex_ids_ok ? demo::Rgba8{0, 255, 0, 255} : demo::Rgba8{255, 0, 0, 255};
                        else
                            t.color = {to8(v0.r * in.tint), to8(v0.g * in.tint), to8(v0.b * in.tint), 255};
                        tris.push_back(t);
                    };
                    if (d.mode == PT::Triangles)
                        for (std::size_t i = 0; i + 2 < run.size(); i += 3) make(run[i], run[i + 1], run[i + 2]);
                    else
                        for (std::size_t i = 0; i + 2 < run.size(); ++i) make(run[i], run[i + 1], run[i + 2]);
                }
            }
        }
        return tris;
    }

    float edge_fn(const std::array<float, 2>& a, const std::array<float, 2>& b, float x, float y)
    {
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    }

    float segment_distance(const std::array<float, 2>& a, const std::array<float, 2>& b, float x, float y)
    {
        const float dx = b[0] - a[0], dy = b[1] - a[1];
        const float len2 = dx * dx + dy * dy;
        float t = len2 > 0.0f ? ((x - a[0]) * dx + (y - a[1]) * dy) / len2 : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        const float ex = a[0] + t * dx - x, ey = a[1] + t * dy - y;
        return std::sqrt(ex * ex + ey * ey);
    }

    // Colour the model predicts at pixel (px, py): the last triangle covering the pixel centre wins.
    demo::Rgba8 model_color(const std::vector<Tri>& tris, int px, int py)
    {
        const float x = static_cast<float>(px) + 0.5f, y = static_cast<float>(py) + 0.5f;
        demo::Rgba8 c = clear_rgba();
        for (const Tri& t : tris)
        {
            const float e0 = edge_fn(t.p[0], t.p[1], x, y), e1 = edge_fn(t.p[1], t.p[2], x, y),
                        e2 = edge_fn(t.p[2], t.p[0], x, y);
            if ((e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0)) c = t.color;
        }
        return c;
    }

    // True when the pixel centre is within 1.5 px of any triangle edge: rasterization of edge pixels is
    // implementation-dependent, so such probes are skipped.
    bool near_edge(const std::vector<Tri>& tris, int px, int py)
    {
        const float x = static_cast<float>(px) + 0.5f, y = static_cast<float>(py) + 0.5f;
        for (const Tri& t : tris)
            for (std::size_t k = 0; k < 3; ++k)
                if (segment_distance(t.p[k], t.p[(k + 1) % 3], x, y) < 1.5f) return true;
        return false;
    }

    bool rgba_near(const std::uint8_t* got, demo::Rgba8 want, int tol)
    {
        for (std::size_t c = 0; c < 4; ++c)
            if (std::abs(static_cast<int>(got[c]) - static_cast<int>(want[c])) > tol) return false;
        return true;
    }

    std::string cell_label(int col, int row)
    {
        if (col == 0 && row == 7) return "mesh 0 (never drawn: must stay clear)";
        if (row == 6 && col <= 3) return kDraws[static_cast<std::size_t>(col)].what;
        if (row == 5 && col <= 3) return std::format("mesh 5 instance {} (offset + tint from the instanced attribute)", col);
        if (row == 4 && col <= 3) return std::format("vertex-ID cell {} (green: gl_VertexID == index + basevertex)", col);
        if (row == 3 && col == 0) return "strip quad A (before the restart)";
        if (row == 3 && col == 1) return "strip quad B (after the restart; 254 + 1 = 255 must NOT restart)";
        if (row == kRobustRow && col == kRobustCol) return "robust cell: out-of-bounds draws scissored here, mesh 9 on top";
        return "empty cell keeps the clear colour";
    }

    class BaseVertexAndRobustness final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            u_mode_ = mgl::glGetUniformLocation(program_, "u_mode");
            if (u_mode_.value < 0) throw demo::Failure("u_mode is not an active uniform");

            probe_ = demo::build_program(env, "shaders/probe.vert", "shaders/probe.frag");
            u_f_ = mgl::glGetUniformLocation(probe_, "u_f");
            u_i_ = mgl::glGetUniformLocation(probe_, "u_i");
            u_u_ = mgl::glGetUniformLocation(probe_, "u_u");
            u_m_ = mgl::glGetUniformLocation(probe_, "u_m");
            if (u_f_.value < 0 || u_i_.value < 0 || u_u_.value < 0 || u_m_.value < 0)
                throw demo::Failure("a probe uniform is not active");

            const std::vector<Vertex> verts = build_vertices();
            const auto indices = build_indices();

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);

            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(verts), mgl::BufferUsage::StaticDraw);
            const mgl::AttribLocation a_position{0u}, a_color{1u}, a_index{2u}, a_instance{3u};
            mgl::glEnableVertexAttribArray(a_position);
            mgl::glVertexAttribPointer(a_position, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glEnableVertexAttribArray(a_color);
            mgl::glVertexAttribPointer(a_color, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glEnableVertexAttribArray(a_index);
            mgl::glVertexAttribIPointer(a_index, 1, mgl::DataType::Int, sizeof(Vertex),
                                        demo::buffer_offset(offsetof(Vertex, index)));

            mgl::glGenBuffers(1, &instance_vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, instance_vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Instance>(kInstances), mgl::BufferUsage::StaticDraw);
            mgl::glEnableVertexAttribArray(a_instance);
            mgl::glVertexAttribPointer(a_instance, 3, mgl::DataType::Float, GL_FALSE, sizeof(Instance),
                                       demo::buffer_offset(0));
            mgl::glVertexAttribDivisor(a_instance, 1);

            // The ELEMENT_ARRAY_BUFFER binding is VAO state: one index buffer for every draw.
            mgl::glGenBuffers(1, &ibo_);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ibo_);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint8_t>(indices),
                              mgl::BufferUsage::StaticDraw);

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // Robust buffer access is a property of the context (ES 3.2 section 20.2). The out-of-bounds draws
            // are issued only when the flag is set: without it, their behaviour is not defined at all.
            context_flags_ = demo::get_integer(mgl::GetParameter::ContextFlags);
            robust_ = (static_cast<GLbitfield>(context_flags_) & static_cast<GLbitfield>(mgl::ContextFlagMask::RobustAccess)) != 0u;
            std::cout << std::format("note: CONTEXT_FLAGS = 0x{:x} (robust access {}, debug {})\n", context_flags_,
                                     robust_ ? "yes" : "no",
                                     (static_cast<GLbitfield>(context_flags_) & static_cast<GLbitfield>(mgl::ContextFlagMask::Debug)) != 0u ? "yes" : "no");
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            oob_error_ = mgl::ErrorCode::NoError;
            for (const Draw& d : kDraws)
            {
                if (d.out_of_bounds && !robust_) continue;
                mgl::glUniform1i(u_mode_, d.u_mode);
                if (d.restart) mgl::glEnable(mgl::Capability::PrimitiveRestartFixedIndex);
                if (d.out_of_bounds)
                {
                    // Whatever the out-of-bounds vertices hold, the scissor test keeps every fragment inside
                    // the robust cell (ES 3.2 section 13.8.2), so the rest of the image stays checkable.
                    const auto box = cell_box(env, kRobustCol, kRobustRow);
                    mgl::glEnable(mgl::Capability::ScissorTest);
                    mgl::glScissor(box[0], box[1], box[2], box[3]);
                }
                const void* indices = demo::buffer_offset(d.offset);
                switch (d.cmd)
                {
                case Cmd::BaseVertex:
                    mgl::glDrawElementsBaseVertex(d.mode, d.count, d.type, indices, d.basevertex);
                    break;
                case Cmd::RangeBaseVertex:
                    mgl::glDrawRangeElementsBaseVertex(d.mode, d.start, d.end, d.count, d.type, indices, d.basevertex);
                    break;
                case Cmd::InstancedBaseVertex:
                    mgl::glDrawElementsInstancedBaseVertex(d.mode, d.count, d.type, indices, d.instances, d.basevertex);
                    break;
                }
                if (d.out_of_bounds)
                {
                    const mgl::ErrorCode e = demo::drain_gl_errors();
                    if (oob_error_ == mgl::ErrorCode::NoError) oob_error_ = e;
                    mgl::glDisable(mgl::Capability::ScissorTest);
                }
                if (d.restart) mgl::glDisable(mgl::Capability::PrimitiveRestartFixedIndex);
            }
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            render(env, 0);
            check_draw_errors(check);
            check_scene(env, check);
            check_mutation(env, check);
            check_robust_state(check);
            check_gpu_negative_control(env, check);

            render(env, 0);
            check_readn_pixels(env, check);
            check_getn_uniform(check);

            render(env, 0);  // leave the scene in the framebuffer for --capture
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteBuffers(1, &instance_vbo_);
            mgl::glDeleteBuffers(1, &ibo_);
            mgl::glDeleteProgram(program_);
            mgl::glDeleteProgram(probe_);
        }

    private:
        // Pixel box {x, y, w, h} of a grid cell.
        static std::array<int, 4> cell_box(const demo::Env& env, int col, int row)
        {
            auto px = [](float ndc, int size) {
                return static_cast<int>(std::lround((ndc + 1.0f) * 0.5f * static_cast<float>(size)));
            };
            const int x0 = px(cell_x(col), env.width), x1 = px(cell_x(col) + kCell, env.width);
            const int y0 = px(cell_y(row), env.height), y1 = px(cell_y(row) + kCell, env.height);
            return {x0, y0, x1 - x0, y1 - y0};
        }

        // Probe pixels of a cell: a 3 x 3 grid at 1/4, 1/2 and 3/4 of the cell.
        static std::vector<std::array<int, 2>> cell_probes(const demo::Env& env, int col, int row)
        {
            const auto box = cell_box(env, col, row);
            std::vector<std::array<int, 2>> probes;
            for (int j = 1; j <= 3; ++j)
                for (int i = 1; i <= 3; ++i)
                    probes.push_back({box[0] + box[2] * i / 4, box[1] + box[3] * j / 4});
            return probes;
        }

        // INVALID_VALUE cases of section 10.5 (end < start) and the implicit sizei rule of section 2.3.1
        // (negative count / instancecount). Each call would draw mesh 0 into cell (0,7) if it were executed;
        // check_scene() then proves the cell is still clear ("the command ... is ignored", section 2.3.1).
        void check_draw_errors(demo::Check& check)
        {
            check.expect_no_gl_error("the base-vertex scene renders without GL errors");
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            mgl::glUniform1i(u_mode_, 0);
            const void* quad16 = demo::buffer_offset(kOffQuadU16);
            mgl::glDrawRangeElementsBaseVertex(PT::Triangles, 3, 0, 6, IT::UnsignedShort, quad16, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawRangeElementsBaseVertex with end < start -> INVALID_VALUE");
            mgl::glDrawElementsBaseVertex(PT::Triangles, -1, IT::UnsignedShort, quad16, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawElementsBaseVertex with count -1 -> INVALID_VALUE");
            mgl::glDrawRangeElementsBaseVertex(PT::Triangles, 0, 3, -1, IT::UnsignedShort, quad16, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDrawRangeElementsBaseVertex with count -1 -> INVALID_VALUE");
            mgl::glDrawElementsInstancedBaseVertex(PT::Triangles, 6, IT::UnsignedShort, quad16, -1, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue,
                                  "glDrawElementsInstancedBaseVertex with instancecount -1 -> INVALID_VALUE");
        }

        void check_scene(const demo::Env& env, demo::Check& check)
        {
            const std::vector<Tri> tris = model_triangles(env.width, env.height);
            const std::vector<std::uint8_t> image = demo::read_rgba8(0, 0, env.width, env.height);
            for (int row = kGrid - 1; row >= 0; --row)
                for (int col = 0; col < kGrid; ++col)
                {
                    int probed = 0;
                    std::string detail;
                    bool ok = true;
                    for (const auto& p : cell_probes(env, col, row))
                    {
                        if (near_edge(tris, p[0], p[1])) continue;
                        ++probed;
                        const demo::Rgba8 want = model_color(tris, p[0], p[1]);
                        const std::uint8_t* got = &image[(static_cast<std::size_t>(p[1]) * static_cast<std::size_t>(env.width) +
                                                          static_cast<std::size_t>(p[0])) * 4];
                        // Colours are multiples of 0.25 (x tint): one 8-bit quantization step of tolerance.
                        if (!rgba_near(got, want, 1) && ok)
                        {
                            ok = false;
                            detail = std::format("pixel ({},{}) = {} expected {}", p[0], p[1],
                                                 demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
                        }
                    }
                    if (probed < 4)
                    {
                        ok = false;
                        detail = std::format("only {} probes away from edges", probed);
                    }
                    check.expect(ok, std::format("cell ({},{}) {}: {} probes match the CPU model", col, row,
                                                 cell_label(col, row), probed),
                                 detail);
                }
        }

        // Mutation argument: deliberately wrong models must disagree with the image, otherwise the scene check
        // above could not detect the error they model. Returns {drawn cells, drawn cells where the mutated model
        // disagrees with the image}; `only` restricts the count to one cell.
        static std::array<int, 2> mutation_hits(const demo::Env& env, const std::vector<std::uint8_t>& image,
                                                Mutation mutation, std::array<int, 2> only = {-1, -1})
        {
            const std::vector<Tri> good = model_triangles(env.width, env.height);
            const std::vector<Tri> mutated = model_triangles(env.width, env.height, mutation);
            int drawn_cells = 0, detected = 0;
            for (int row = 0; row < kGrid; ++row)
                for (int col = 0; col < kGrid; ++col)
                {
                    if (only[0] >= 0 && (col != only[0] || row != only[1])) continue;
                    bool drawn = false, differs = false;
                    for (const auto& p : cell_probes(env, col, row))
                    {
                        if (near_edge(good, p[0], p[1])) continue;
                        drawn = drawn || model_color(good, p[0], p[1]) != clear_rgba();
                        const std::uint8_t* got = &image[(static_cast<std::size_t>(p[1]) * static_cast<std::size_t>(env.width) +
                                                          static_cast<std::size_t>(p[0])) * 4];
                        differs = differs || !rgba_near(got, model_color(mutated, p[0], p[1]), 1);
                    }
                    if (drawn) ++drawn_cells;
                    if (drawn && differs) ++detected;
                }
            return {drawn_cells, detected};
        }

        void check_mutation(const demo::Env& env, demo::Check& check)
        {
            const std::vector<std::uint8_t> image = demo::read_rgba8(0, 0, env.width, env.height);
            const auto all = mutation_hits(env, image, Mutation::IgnoreBaseVertex);
            check.expect(all[0] == 15 && all[1] == all[0],
                         "mutation: a model that ignores basevertex disagrees with the image in every drawn cell",
                         std::format("{} of {} drawn cells (15 expected) detected", all[1], all[0]));
            const auto strip = mutation_hits(env, image, Mutation::RestartAfterBaseVertex, {1, 3});
            check.expect(strip[0] == 1 && strip[1] == 1,
                         "mutation: a model that compares the restart index after adding basevertex disagrees with cell (1,3)",
                         std::format("drawn {} detected {}", strip[0], strip[1]));
        }

        void check_robust_state(demo::Check& check)
        {
            // CONTEXT_FLAGS (ES 3.2 section 20.2). The runtime requested robust access
            // (EGL_CONTEXT_OPENGL_ROBUST_ACCESS = TRUE); EGL 1.5 section 3.7.1.5: "If the implementation does not
            // support robust buffer access, context creation will fail." A context exists, so the bit must be set.
            check.expect(robust_, "CONTEXT_FLAGS has CONTEXT_FLAG_ROBUST_ACCESS_BIT (robust context requested)",
                         std::format("CONTEXT_FLAGS = 0x{:x}", context_flags_));

            // Reset status (section 2.3.2): NO_ERROR means "not in a reset state since the last call".
            const mgl::GraphicsResetStatus s1 = mgl::glGetGraphicsResetStatus();
            check.expect(s1 == mgl::GraphicsResetStatus::NoError, "glGetGraphicsResetStatus() == NO_ERROR on a healthy context",
                         std::format("0x{:x}", static_cast<GLenum>(s1)));

            // Strategy: one of the two values of section 2.3.2 (state table: Z2), and the requested one.
            const int strategy = demo::get_integer(mgl::GetParameter::ResetNotificationStrategy);
            const bool allowed = strategy == static_cast<int>(mgl::ResetNotificationStrategy::NoResetNotification) ||
                                 strategy == static_cast<int>(mgl::ResetNotificationStrategy::LoseContextOnReset);
            check.expect(allowed, "RESET_NOTIFICATION_STRATEGY is NO_RESET_NOTIFICATION or LOSE_CONTEXT_ON_RESET",
                         std::format("0x{:x}", strategy));
            check.expect_eq(strategy, static_cast<int>(mgl::ResetNotificationStrategy::LoseContextOnReset),
                            "RESET_NOTIFICATION_STRATEGY == LOSE_CONTEXT_ON_RESET (requested at context creation)");

            // Robust buffer access (section 10.3.5): out-of-bounds vertex fetches give undefined attribute values
            // "but cannot result in application failure". Observable: the draws returned, raised no error, did not
            // reset the context, and the cell they were scissored into is correct after mesh 9 (check_scene).
            if (robust_)
            {
                check.expect(oob_error_ == mgl::ErrorCode::NoError,
                             "robust access: draws with basevertex past the end of the vertex buffer raise no GL error",
                             demo::error_name(oob_error_));
                const mgl::GraphicsResetStatus s2 = mgl::glGetGraphicsResetStatus();
                check.expect(s2 == mgl::GraphicsResetStatus::NoError,
                             "robust access: no graphics reset after the out-of-bounds draws",
                             std::format("0x{:x}", static_cast<GLenum>(s2)));
            }
            check.expect_no_gl_error("state queries raise no error");
        }

        // GPU negative control: without basevertex (glDrawElements, or basevertex 0) the same index data draws
        // mesh 0, into the cell that check_scene() requires to be clear. So that requirement is not vacuous.
        void check_gpu_negative_control(const demo::Env& env, demo::Check& check)
        {
            const auto probe0 = cell_probes(env, 0, 7)[4];
            const auto probe2 = cell_probes(env, 1, 6)[4];
            const demo::Rgba8 red = {255, 0, 0, 255};
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            mgl::glUniform1i(u_mode_, 0);

            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDrawElements(PT::Triangles, 6, IT::UnsignedShort, demo::buffer_offset(kOffQuadU16));
            check.expect_pixel(probe0[0], probe0[1], red, 1, "negative control: glDrawElements (no basevertex) draws mesh 0 into cell (0,7)");
            check.expect_pixel(probe2[0], probe2[1], clear_rgba(), 1, "negative control: ... and nothing into mesh 2's cell (1,6)");

            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDrawElementsBaseVertex(PT::Triangles, 6, IT::UnsignedShort, demo::buffer_offset(kOffQuadU16), 0);
            check.expect_pixel(probe0[0], probe0[1], red, 1, "negative control: glDrawElementsBaseVertex(basevertex 0) == glDrawElements");

            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDrawElementsBaseVertex(PT::Triangles, 6, IT::UnsignedShort, demo::buffer_offset(kOffQuadU16), 8);
            const Color c2 = kMeshes[2].color;
            check.expect_pixel(probe2[0], probe2[1], {to8(c2.r), to8(c2.g), to8(c2.b), 255}, 1,
                               "the same call with basevertex 8 draws mesh 2 into cell (1,6)");
            check.expect_pixel(probe0[0], probe0[1], clear_rgba(), 1, "... and cell (0,7) stays clear");
        }

        // glReadnPixels (ES 3.2 section 16.1.2): "An INVALID_OPERATION error is generated by ReadnPixels if the
        // buffer size required to store the requested data is greater than bufSize." On an error the command is
        // ignored and "no change is made to these values" (section 2.3.1).
        void check_readn_pixels(const demo::Env& env, demo::Check& check)
        {
            constexpr int w = 4, h = 4;
            constexpr GLsizei kRequired = w * h * 4;  // RGBA / UNSIGNED_BYTE, PACK_ALIGNMENT 4: rows have no padding
            constexpr std::uint8_t kCanary = 0xA5;
            constexpr std::size_t kTail = 16;
            const auto box = cell_box(env, kMeshes[1].col, kMeshes[1].row);
            const int x = box[0] + box[2] / 2 - w / 2, y = box[1] + box[3] / 2 - h / 2;
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);

            std::array<std::uint8_t, kRequired + kTail> buf{};
            auto untouched = [&](std::size_t from) {
                return std::all_of(buf.begin() + static_cast<std::ptrdiff_t>(from), buf.end(),
                                   [](std::uint8_t b) { return b == kCanary; });
            };

            buf.fill(kCanary);
            mgl::glReadnPixels(x, y, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, kRequired, buf.data());
            check.expect_no_gl_error("glReadnPixels with bufSize == required size (64 bytes) succeeds");
            const std::vector<std::uint8_t> ref = demo::read_rgba8(x, y, w, h);
            check.expect(std::equal(ref.begin(), ref.end(), buf.begin()), "glReadnPixels returns the same bytes as glReadPixels");
            const Color c1 = kMeshes[1].color;
            const demo::Rgba8 want = {to8(c1.r), to8(c1.g), to8(c1.b), 255};
            bool colors_ok = true;
            for (int i = 0; i < w * h; ++i) colors_ok = colors_ok && rgba_near(&buf[static_cast<std::size_t>(i) * 4], want, 1);
            check.expect(colors_ok, "glReadnPixels pixels show mesh 1's colour", demo::format_rgba(want));
            check.expect(untouched(kRequired), "glReadnPixels writes nothing past bufSize (16 canary bytes intact)");

            buf.fill(kCanary);
            mgl::glReadnPixels(x, y, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, kRequired - 1, buf.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glReadnPixels with bufSize = required - 1 -> INVALID_OPERATION");
            check.expect(untouched(0), "glReadnPixels with a too small bufSize leaves the whole buffer untouched");

            buf.fill(kCanary);
            mgl::glReadnPixels(x, y, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, 0, buf.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glReadnPixels with bufSize 0 -> INVALID_OPERATION");
            check.expect(untouched(0), "glReadnPixels with bufSize 0 leaves the buffer untouched");

            // Negative bufSize: section 16.1.2 lists INVALID_VALUE, and the required size also exceeds it
            // (INVALID_OPERATION). Section 2.3.1: "the GL implementation may choose to generate any of the
            // applicable errors", so both codes are conformant; the buffer must stay untouched either way.
            buf.fill(kCanary);
            mgl::glReadnPixels(x, y, w, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, -1, buf.data());
            // Mesa 25.0 accepts the negative bufSize as "unlimited" (no error, the 64 bytes are written), which
            // contradicts the explicit INVALID_VALUE rule; kept strict, labelled as a known deviation.
            const mgl::ErrorCode neg = demo::drain_gl_errors();
            constexpr std::string_view kMesaReason =
                "Mesa does not reject a negative bufSize in glReadnPixels (ES 3.2 section 16.1.2: INVALID_VALUE "
                "if bufSize is negative); it reports no error and writes the pixels";
            check.expect_or_known_deviation(neg == mgl::ErrorCode::InvalidValue || neg == mgl::ErrorCode::InvalidOperation,
                                            "glReadnPixels with bufSize -1 -> INVALID_VALUE (or the also applicable INVALID_OPERATION)",
                                            demo::error_name(neg), "Mesa", kMesaReason);
            check.expect_or_known_deviation(untouched(0), "glReadnPixels with bufSize -1 leaves the buffer untouched",
                                            "the pixels were written", "Mesa", kMesaReason);

            // Nothing requested, nothing required: width 0 with bufSize 0 is not an error and writes nothing.
            buf.fill(kCanary);
            mgl::glReadnPixels(x, y, 0, h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, 0, buf.data());
            check.expect_no_gl_error("glReadnPixels of a 0 x 4 rectangle with bufSize 0 succeeds");
            check.expect(untouched(0), "glReadnPixels of an empty rectangle writes nothing");
        }

        // glGetnUniform*v (ES 3.2 section 7.12): "An INVALID_OPERATION error is generated by GetnUniform* if the
        // buffer size required to store the requested data is greater than bufSize." bufSize counts bytes
        // (gl.xml declares params with len="bufSize / 4" for these 4-byte types).
        template <typename T, std::size_t N>
        void check_getn(demo::Check& check, const char* name, mgl::UniformLocation loc, const std::array<T, N>& expected,
                        T canary)
        {
            constexpr GLsizei kRequired = static_cast<GLsizei>(N * sizeof(T));
            std::array<T, N + 4> buf{};
            auto untouched = [&](std::size_t from) {
                return std::all_of(buf.begin() + static_cast<std::ptrdiff_t>(from), buf.end(), [&](T v) { return v == canary; });
            };
            auto query = [&](GLsizei size) {
                buf.fill(canary);
                mgl::glGetnUniform(probe_, loc, size, buf.data());  // -> glGetnUniformfv / iv / uiv by T
            };

            query(kRequired);
            check.expect_no_gl_error(std::format("glGetnUniform {} with bufSize == {} bytes succeeds", name, kRequired));
            check.expect(std::equal(expected.begin(), expected.end(), buf.begin()), std::format("glGetnUniform {} returns the uploaded values", name));
            check.expect(untouched(N), std::format("glGetnUniform {} writes nothing past bufSize", name));

            query(kRequired + 16);
            check.expect_no_gl_error(std::format("glGetnUniform {} with a larger bufSize succeeds", name));
            check.expect(std::equal(expected.begin(), expected.end(), buf.begin()) && untouched(N),
                         std::format("glGetnUniform {} with a larger bufSize returns exactly {} values (the uniform's type decides)", name, N));

            query(kRequired - 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, std::format("glGetnUniform {} with bufSize = {} - 1 -> INVALID_OPERATION", name, kRequired));
            check.expect(untouched(0), std::format("glGetnUniform {} with a too small bufSize leaves params untouched", name));

            query(static_cast<GLsizei>(N));  // a component count is not a byte count
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  std::format("glGetnUniform {} with bufSize = component count ({}) -> INVALID_OPERATION", name, N));
            check.expect(untouched(0), std::format("glGetnUniform {} with bufSize {} leaves params untouched", name, N));

            // Negative bufSize: the implicit sizei rule of section 2.3.1 (INVALID_VALUE) and the INVALID_OPERATION
            // rule both apply; section 2.3.1 lets the implementation choose either. params must stay untouched.
            query(-1);
            const mgl::ErrorCode neg = demo::drain_gl_errors();
            check.expect(neg == mgl::ErrorCode::InvalidValue || neg == mgl::ErrorCode::InvalidOperation,
                         std::format("glGetnUniform {} with bufSize -1 -> INVALID_VALUE or INVALID_OPERATION", name),
                         demo::error_name(neg));
            check.expect(untouched(0), std::format("glGetnUniform {} with bufSize -1 leaves params untouched", name));
        }

        void check_getn_uniform(demo::Check& check)
        {
            mgl::glUseProgram(probe_);
            mgl::glUniform4f(u_f_, 1.5f, -2.25f, 3.0f, 0.125f);
            mgl::glUniform3i(u_i_, 7, -8, 9);
            mgl::glUniform2ui(u_u_, 4000000000u, 17u);
            const std::array<GLfloat, 4> m = {1.0f, 2.0f, 3.0f, 4.0f};  // column-major: col0 = (1,2), col1 = (3,4)
            mgl::glUniformMatrix2fv(u_m_, 1, GL_FALSE, m.data());
            check.expect_no_gl_error("probe uniforms uploaded");

            check_getn<GLfloat, 4>(check, "fv(vec4)", u_f_, {1.5f, -2.25f, 3.0f, 0.125f}, -777.0f);
            check_getn<GLint, 3>(check, "iv(ivec3)", u_i_, {7, -8, 9}, 0x5A5A5A5A);
            check_getn<GLuint, 2>(check, "uiv(uvec2)", u_u_, {4000000000u, 17u}, 0xA5A5A5A5u);
            check_getn<GLfloat, 4>(check, "fv(mat2, column-major)", u_m_, m, -777.0f);

            // An invalid location is its own INVALID_OPERATION case of section 7.12, independent of bufSize.
            std::array<GLfloat, 8> buf{};
            buf.fill(-777.0f);
            mgl::glGetnUniformfv(probe_, mgl::UniformLocation{-1}, static_cast<GLsizei>(sizeof(buf)), buf.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetnUniformfv at location -1 -> INVALID_OPERATION");
            check.expect(std::all_of(buf.begin(), buf.end(), [](GLfloat v) { return v == -777.0f; }),
                         "glGetnUniformfv at location -1 leaves params untouched");
            mgl::glUseProgram(program_);
        }

        mgl::ProgramId program_{};
        mgl::ProgramId probe_{};
        mgl::UniformLocation u_mode_{};
        mgl::UniformLocation u_f_{}, u_i_{}, u_u_{}, u_m_{};
        mgl::VertexArrayId vao_{};
        mgl::BufferId vbo_{};
        mgl::BufferId instance_vbo_{};
        mgl::BufferId ibo_{};
        int context_flags_ = 0;
        bool robust_ = false;
        mgl::ErrorCode oob_error_ = mgl::ErrorCode::NoError;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-base-vertex-and-robustness",
        .title = "Base vertex drawing and robustness",
        .min_es = {3, 2},
        .summary = "Ten meshes in one vertex buffer drawn from one index buffer with glDrawElementsBaseVertex, "
                   "glDrawRangeElementsBaseVertex and glDrawElementsInstancedBaseVertex (gl_VertexID, instancing, "
                   "primitive restart), checked against a CPU model; in a robust context: CONTEXT_FLAGS, reset "
                   "status and strategy, out-of-bounds draws, and bounded glReadnPixels / glGetnUniform*v queries.",
        .source_dir = DEMO_SOURCE_DIR,
        .robust_context = true,
    };
    return demo::run<BaseVertexAndRobustness>(argc, argv, info);
}
