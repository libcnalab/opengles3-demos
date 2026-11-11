// es20-indexed-drawing
//
// Indexed drawing (ES 3.0 sections 2.9.3 and 2.10.7): glDrawElements fetches vertices through an
// index array, so a vertex shared by several triangles is stored - and processed - once.
//
// The 3 x 3 cell grid below has 16 vertices and 18 triangles (54 indices). Every tile of the 4 x 4
// window is one draw of (a variation of) that mesh:
//
//   row 0   UNSIGNED_BYTE | UNSIGNED_SHORT | UNSIGNED_INT (indices > 65535) | byte offset into the buffer
//   row 1   glBufferData -> glBufferSubData on the indices -> ... on a vertex -> glBufferData again (new size)
//   row 2   orphaning | a second VAO with its own index buffer | back to the first VAO | client-memory indices
//   row 3   the same mesh without indices | a failed draw (count < 0) | a failed glBufferSubData | first row only
//
// validate() computes, for every drawn triangle, the pixel containing its centroid and the color the
// rasterizer must produce there (barycentric interpolation of the vertex colors at that pixel's center,
// ES 3.0 3.6.1), and checks that the triangles that must NOT be drawn left the clear color untouched.
// Buffer sizes / usages, the ELEMENT_ARRAY_BUFFER binding of each VAO and the error codes of the negative
// cases are checked with state queries.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;

namespace
{
    constexpr int kTiles = 4;                                        // 4 x 4 tiles
    constexpr int kMinTile = 40;  // smallest tile edge: leaves >= 2 px of clear margin around the mesh (5 % per side)
    constexpr std::array<float, 4> kClear = {0.10f, 0.12f, 0.20f, 1.0f};

    // ---------------------------------------------------------------------------------------------
    // The mesh: 4 x 4 vertices, 3 x 3 cells, two triangles per cell.
    // ---------------------------------------------------------------------------------------------
    struct Vertex
    {
        float x, y;     // NDC
        float r, g, b;  // color
    };
    static_assert(sizeof(Vertex) == 20);

    constexpr int kVertexCount = 16;
    constexpr int kTriangleCount = 18;
    constexpr int kIndexCount = 54;
    // UNSIGNED_INT demonstration: the mesh lives at vertex indices 70000..70015. 70000 does not fit in 16 bits,
    // and 70000 mod 65536 = 4464 is where a decoy copy of the mesh sits (see create_big_buffer()).
    constexpr std::uint32_t kBigBase = 70000;
    constexpr std::uint32_t kDecoyBase = kBigBase & 0xFFFFu;

    using Tri = std::array<int, 3>;
    using Vertices = std::array<Vertex, kVertexCount>;

    Vertices make_vertices()
    {
        Vertices v{};
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
            {
                const float ti = static_cast<float>(i) / 3.0f, tj = static_cast<float>(j) / 3.0f;
                // Every vertex gets a different color so that a wrong index shows up as a wrong color.
                v[static_cast<std::size_t>(j * 4 + i)] = {-0.9f + 1.8f * ti, -0.9f + 1.8f * tj, 0.15f + 0.85f * ti,
                                                          0.15f + 0.85f * tj, static_cast<float>((i + 2 * j) % 4) / 3.0f};
            }
        return v;
    }

    /// Triangles in cell order (row-major); cell c owns triangles 2c and 2c + 1. Counter-clockwise.
    std::array<Tri, kTriangleCount> make_triangles()
    {
        std::array<Tri, kTriangleCount> t{};
        std::size_t n = 0;
        for (int cj = 0; cj < 3; ++cj)
            for (int ci = 0; ci < 3; ++ci)
            {
                const int v00 = cj * 4 + ci;
                t[n++] = {v00, v00 + 1, v00 + 5};
                t[n++] = {v00, v00 + 5, v00 + 4};
            }
        return t;
    }

    template <typename Index>
    std::array<Index, kIndexCount> flatten(const std::array<Tri, kTriangleCount>& tris, Index base = 0)
    {
        std::array<Index, kIndexCount> out{};
        std::size_t n = 0;
        for (const Tri& t : tris)
            for (const int v : t) out[n++] = static_cast<Index>(static_cast<std::uint32_t>(v) + base);
        return out;
    }

    Tri sorted(Tri t)
    {
        std::sort(t.begin(), t.end());
        return t;
    }

    // ---------------------------------------------------------------------------------------------
    // CPU reference
    // ---------------------------------------------------------------------------------------------
    struct Rect
    {
        int x, y, w, h;  // window coordinates, origin bottom-left
        [[nodiscard]] int cx() const { return x + w / 2; }
        [[nodiscard]] int cy() const { return y + h / 2; }
    };

    std::uint8_t to8(double v)
    {
        return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
    }

    struct Pt
    {
        double x, y;
    };

    Pt to_window(const Rect& tile, const Vertex& v)
    {
        return {tile.x + (static_cast<double>(v.x) + 1.0) * 0.5 * tile.w,
                tile.y + (static_cast<double>(v.y) + 1.0) * 0.5 * tile.h};
    }

    double edge(Pt a, Pt b, Pt p) { return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x); }

    struct Probe
    {
        int x, y;
        demo::Rgba8 color;
        bool interior;  // all barycentric weights > 0.05: far enough from every edge for an exact comparison
    };

    /// The pixel containing the centroid of triangle (a, b, c) and the color the rasterizer produces at
    /// that pixel's center: barycentric interpolation of the vertex colors (ES 3.0 3.6.1).
    /// Degenerate triangles (zero area) draw nothing and have no probe.
    std::optional<Probe> probe_triangle(const Rect& tile, const Vertex& a, const Vertex& b, const Vertex& c)
    {
        const Pt A = to_window(tile, a), B = to_window(tile, b), C = to_window(tile, c);
        const double area = edge(A, B, C);
        if (std::abs(area) < 1e-6) return std::nullopt;
        const int px = static_cast<int>(std::floor((A.x + B.x + C.x) / 3.0));
        const int py = static_cast<int>(std::floor((A.y + B.y + C.y) / 3.0));
        const Pt p{px + 0.5, py + 0.5};
        const double wa = edge(B, C, p) / area, wb = edge(C, A, p) / area, wc = 1.0 - wa - wb;
        auto mix = [&](float va, float vb, float vc) {
            return to8(wa * static_cast<double>(va) + wb * static_cast<double>(vb) + wc * static_cast<double>(vc));
        };
        return Probe{px, py, {mix(a.r, b.r, c.r), mix(a.g, b.g, c.g), mix(a.b, b.b, c.b), 255},
                     std::min({wa, wb, wc}) > 0.05};
    }

    bool differs(demo::Rgba8 got, demo::Rgba8 want, int tolerance)
    {
        for (std::size_t i = 0; i < 4; ++i)
            if (std::abs(int{got[i]} - int{want[i]}) > tolerance) return true;
        return false;
    }

    /// One recorded check for a whole tile: every triangle in `drawn` must show the interpolated color at its
    /// centroid pixel (tolerance 3: mediump varyings and 8-bit rounding), every canonical triangle that is not
    /// in `drawn` must show the clear color there (tolerance 1), and so must the tile's corner pixels.
    void check_tile(demo::Check& check, std::string_view what, const Rect& tile, std::span<const Vertex> verts,
                    std::span<const Tri> drawn, std::span<const Tri> canonical)
    {
        const demo::Rgba8 clear = {to8(static_cast<double>(kClear[0])), to8(static_cast<double>(kClear[1])),
                                   to8(static_cast<double>(kClear[2])), 255};
        bool ok = true;
        std::string detail;
        int drawn_probes = 0, absent_probes = 0;

        for (const Tri& t : drawn)
        {
            const auto pr = probe_triangle(tile, verts[static_cast<std::size_t>(t[0])], verts[static_cast<std::size_t>(t[1])],
                                           verts[static_cast<std::size_t>(t[2])]);
            if (!pr) continue;
            ++drawn_probes;
            if (!pr->interior)
            {
                ok = false;
                detail += "internal: probe too close to an edge; ";
            }
            const demo::Rgba8 got = demo::read_pixel(pr->x, pr->y);
            if (differs(got, pr->color, 3))
            {
                ok = false;
                detail += std::format("triangle ({},{},{}) pixel({},{}) = {} expected {}; ", t[0], t[1], t[2], pr->x, pr->y,
                                      demo::format_rgba(got), demo::format_rgba(pr->color));
            }
        }
        for (const Tri& c : canonical)
        {
            const bool present = std::any_of(drawn.begin(), drawn.end(), [&](const Tri& t) { return sorted(t) == sorted(c); });
            if (present) continue;
            const auto pr = probe_triangle(tile, verts[static_cast<std::size_t>(c[0])], verts[static_cast<std::size_t>(c[1])],
                                           verts[static_cast<std::size_t>(c[2])]);
            if (!pr) continue;
            ++absent_probes;
            const demo::Rgba8 got = demo::read_pixel(pr->x, pr->y);
            if (differs(got, clear, 1))
            {
                ok = false;
                detail += std::format("absent triangle ({},{},{}) pixel({},{}) = {} expected the clear color; ", c[0], c[1], c[2],
                                      pr->x, pr->y, demo::format_rgba(got));
            }
        }
        // The mesh spans NDC [-0.9, 0.9], so a clear margin of 0.05 * size runs around it. The corner probes sit
        // one pixel inside the margin's outer half (at most 2 px in from the tile edge), never on the mesh edge.
        const auto inset = [](int size) { return std::clamp(static_cast<int>(std::floor(0.05 * size)) - 1, 0, 2); };
        const double ix = inset(tile.w), iy = inset(tile.h);
        for (const Pt corner : {Pt{ix, iy}, Pt{static_cast<double>(tile.w) - 1 - ix, iy}, Pt{ix, static_cast<double>(tile.h) - 1 - iy},
                                Pt{static_cast<double>(tile.w) - 1 - ix, static_cast<double>(tile.h) - 1 - iy}})
        {
            const demo::Rgba8 got = demo::read_pixel(tile.x + static_cast<int>(corner.x), tile.y + static_cast<int>(corner.y));
            if (differs(got, clear, 1))
            {
                ok = false;
                detail += std::format("tile corner = {}; ", demo::format_rgba(got));
            }
        }
        const bool expected_nonempty = std::any_of(drawn.begin(), drawn.end(), [](const Tri& t) {
            return t[0] != t[1] || t[1] != t[2];
        });
        if (expected_nonempty && drawn_probes == 0)
        {
            ok = false;
            detail += "internal: no triangle was probed; ";
        }
        check.expect(ok, std::format("{} - {} triangles show the interpolated colors, {} absent ones show the clear color", what,
                                     drawn_probes, absent_probes),
                     detail);
    }

    // ---------------------------------------------------------------------------------------------
    // Small GL helpers
    // ---------------------------------------------------------------------------------------------
    struct BufferState
    {
        GLint size = -1;
        GLint usage = -1;
    };

    BufferState query_buffer(mgl::BufferTarget target)
    {
        BufferState s;
        mgl::glGetBufferParameteriv(target, mgl::BufferParameter::Size, &s.size);
        mgl::glGetBufferParameteriv(target, mgl::BufferParameter::Usage, &s.usage);
        return s;
    }

    GLint usage_value(mgl::BufferUsage u) { return static_cast<GLint>(u); }

    class IndexedDrawing final : public demo::Demo
    {
    public:
        // =========================================================================================
        void init(demo::Env& env) override
        {
            // The tiles partition the surface: each one is width/4 x height/4 pixels (not necessarily square).
            tile_w_ = env.width / kTiles;
            tile_h_ = env.height / kTiles;
            if (tile_w_ < kMinTile || tile_h_ < kMinTile)
                throw demo::Skip(std::format("the 4x4 tiles need at least {0}x{0} px each, i.e. a {1}x{1} surface (got {2}x{3})",
                                             kMinTile, kTiles * kMinTile, env.width, env.height));
            program_ = demo::build_program(env, "shaders/mesh.vert", "shaders/mesh.frag");

            mesh_ = make_vertices();
            tris_ = make_triangles();
            idx8_ = flatten<std::uint8_t>(tris_);
            idx16_ = flatten<std::uint16_t>(tris_);
            idx32_ = flatten<std::uint32_t>(tris_, kBigBase);

            // Variants of the vertex data used by the mutation tiles.
            moved_ = mesh_;
            moved_[5] = {mesh_[5].x - 0.08f, mesh_[5].y + 0.08f, 1.0f, 1.0f, 1.0f};  // one vertex: moved and white
            inverted_.fill(Vertex{});
            rotated_ = mesh_;
            for (std::size_t i = 0; i < mesh_.size(); ++i)
            {
                inverted_[i] = {mesh_[i].x, mesh_[i].y, 1.0f - mesh_[i].r, 1.0f - mesh_[i].g, 1.0f - mesh_[i].b};
                rotated_[i] = {mesh_[i].x, mesh_[i].y, mesh_[i].g, mesh_[i].b, mesh_[i].r};
            }

            // Triangles of the checkerboard index buffer: cells (ci + cj) even.
            for (int cell = 0; cell < 9; ++cell)
                if (((cell % 3) + (cell / 3)) % 2 == 0)
                {
                    checker_tris_.push_back(tris_[static_cast<std::size_t>(2 * cell)]);
                    checker_tris_.push_back(tris_[static_cast<std::size_t>(2 * cell + 1)]);
                }
            for (const Tri& t : checker_tris_)
                for (const int v : t) idx_checker_.push_back(static_cast<std::uint8_t>(v));

            create_vertex_buffers();
            create_vaos_and_index_buffers();
        }

        // =========================================================================================
        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(program_);

            const auto offset = [](std::size_t bytes) { return demo::buffer_offset(bytes); };

            // ---- row 0: the three index types, and an offset into the index buffer ------------------
            // The index buffer is ELEMENT_ARRAY_BUFFER state of the VAO; the `indices` argument is a byte offset into it.
            mgl::glBindVertexArray(vao_ubyte_);
            set_tile(0, 0);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, mgl::IndexType::UnsignedByte, offset(0));

            mgl::glBindVertexArray(vao_ushort_);
            set_tile(1, 0);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, mgl::IndexType::UnsignedShort, offset(0));

            mgl::glBindVertexArray(vao_uint_);  // vertex buffer with 70016 vertices; the mesh is at 70000..70015
            set_tile(2, 0);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, mgl::IndexType::UnsignedInt, offset(0));

            // Same VAO as tile (1, 0): only the `indices` offset differs. Skip the first 18 indices (= 6 triangles =
            // the bottom row of cells): offset = 18 * sizeof(GLushort) bytes, count = the remaining 36 indices.
            mgl::glBindVertexArray(vao_ushort_);
            set_tile(3, 0);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, 36, mgl::IndexType::UnsignedShort, offset(18 * sizeof(std::uint16_t)));

            // ---- rows 1-2: change buffer contents between draws --------------------------------------
            draw_mutation_sequence();

            // ---- row 2: VAO switching --------------------------------------------------------------
            // No glBindBuffer(ELEMENT_ARRAY_BUFFER) between these draws: binding the VAO selects the index buffer.
            mgl::glBindVertexArray(vao_checker_);
            set_tile(1, 2);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, static_cast<GLsizei>(idx_checker_.size()),
                                mgl::IndexType::UnsignedByte, offset(0));
            mgl::glBindVertexArray(vao_ubyte_);
            set_tile(2, 2);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, mgl::IndexType::UnsignedByte, offset(0));

            // ---- row 2: client-memory indices (default VAO, no ELEMENT_ARRAY_BUFFER) ------------------
            // With zero bound to ELEMENT_ARRAY_BUFFER the `indices` pointer is an ordinary CPU address (ES 3.0 2.10.7).
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_mesh_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            set_tile(3, 2);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, mgl::IndexType::UnsignedShort, idx16_.data());
            mgl::glDisableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glDisableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 4, mgl::DataType::Float, GL_FALSE, 0, nullptr);  // initial state
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 4, mgl::DataType::Float, GL_FALSE, 0, nullptr);

            // ---- row 3 --------------------------------------------------------------------------------
            // The same mesh without an index buffer: 54 expanded vertices and glDrawArrays.
            mgl::glBindVertexArray(vao_expanded_);
            set_tile(0, 3);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, kIndexCount);

            draw_negative_cases();

            // First row of cells only: offset 0, count 18 - together with tile (3, 0) this splits the mesh into two batches.
            mgl::glBindVertexArray(vao_ushort_);
            set_tile(3, 3);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, 18, mgl::IndexType::UnsignedShort, offset(0));

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // =========================================================================================
        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            check_limits(check);
            check_tiles(check);
            check_buffer_sizes(check);
            check_mutation_state(check);
            check_vao_bindings(check);
            check_buffer_objects(check);
            check_negative_cases(check);
        }

        // =========================================================================================
        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            for (mgl::VertexArrayId* v : {&vao_ubyte_, &vao_ushort_, &vao_uint_, &vao_dyn_, &vao_checker_, &vao_expanded_})
                mgl::glDeleteVertexArrays(1, v);
            for (mgl::BufferId* b : {&vbo_mesh_, &vbo_big_, &vbo_dyn_, &vbo_expanded_, &ibo_ubyte_, &ibo_ushort_, &ibo_uint_,
                                     &ibo_dyn_, &ibo_checker_})
                mgl::glDeleteBuffers(1, b);
            mgl::glDeleteProgram(program_);
        }

    private:
        [[nodiscard]] Rect tile(int col, int row) const
        {
            return {col * tile_w_, (kTiles - 1 - row) * tile_h_, tile_w_, tile_h_};
        }

        void set_tile(int col, int row)
        {
            const Rect r = tile(col, row);
            mgl::glViewport(r.x, r.y, r.w, r.h);
        }

        // ---- setup ---------------------------------------------------------------------------
        void create_vertex_buffers()
        {
            const auto A = mgl::BufferTarget::Array;

            // The mesh: 16 vertices, 320 bytes, written once (STATIC_DRAW).
            mgl::glGenBuffers(1, &vbo_mesh_);
            mgl::glBindBuffer(A, vbo_mesh_);
            mgl::glBufferData(A, std::span<const Vertex>(mesh_), mgl::BufferUsage::StaticDraw);

            // The same mesh in a buffer with 70016 vertices: the first 70000 are filler, a decoy copy of the mesh
            // (all vertices magenta) sits at the index a 16-bit truncation of 70000 would hit, and the real mesh is last.
            std::vector<Vertex> big(kBigBase + kVertexCount, Vertex{0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
            for (std::size_t v = 0; v < mesh_.size(); ++v)
            {
                big[kDecoyBase + v] = {mesh_[v].x, mesh_[v].y, 1.0f, 0.0f, 1.0f};
                big[kBigBase + v] = mesh_[v];
            }
            mgl::glGenBuffers(1, &vbo_big_);
            mgl::glBindBuffer(A, vbo_big_);
            mgl::glBufferData(A, std::span<const Vertex>(big), mgl::BufferUsage::StaticDraw);

            // 54 expanded vertices for the non-indexed comparison: three per triangle, shared corners duplicated.
            std::vector<Vertex> expanded;
            for (const Tri& t : tris_)
                for (const int v : t) expanded.push_back(mesh_[static_cast<std::size_t>(v)]);
            mgl::glGenBuffers(1, &vbo_expanded_);
            mgl::glBindBuffer(A, vbo_expanded_);
            mgl::glBufferData(A, std::span<const Vertex>(expanded), mgl::BufferUsage::StaticDraw);

            // The vertex buffer that the mutation tiles rewrite. Its contents are (re)specified in render().
            mgl::glGenBuffers(1, &vbo_dyn_);
            mgl::glBindBuffer(A, vbo_dyn_);
            mgl::glBufferData(A, std::span<const Vertex>(mesh_), mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(A, mgl::BufferId{0});
        }

        /// Attribute 0 = position, attribute 1 = color, both sourced from `vbo` (stride 20). The buffer bound to
        /// ARRAY_BUFFER at the time of the pointer calls is recorded in the VAO.
        void describe_attributes(mgl::VertexArrayId vao, mgl::BufferId vbo)
        {
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, r)));
        }

        /// Creates the index buffer while `vao` is bound: ELEMENT_ARRAY_BUFFER is part of the VAO, so this
        /// is what ties the buffer to it. (Binding the buffer with VAO 0 current would configure VAO 0 instead.)
        template <typename Index>
        void attach_index_buffer(mgl::VertexArrayId vao, mgl::BufferId& ibo, std::span<const Index> data, mgl::BufferUsage usage)
        {
            mgl::glGenBuffers(1, &ibo);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ibo);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, data, usage);
        }

        void create_vaos_and_index_buffers()
        {
            for (mgl::VertexArrayId* v : {&vao_ubyte_, &vao_ushort_, &vao_uint_, &vao_dyn_, &vao_checker_, &vao_expanded_})
                mgl::glGenVertexArrays(1, v);

            describe_attributes(vao_ubyte_, vbo_mesh_);
            attach_index_buffer(vao_ubyte_, ibo_ubyte_, std::span<const std::uint8_t>(idx8_), mgl::BufferUsage::StaticDraw);

            describe_attributes(vao_ushort_, vbo_mesh_);
            attach_index_buffer(vao_ushort_, ibo_ushort_, std::span<const std::uint16_t>(idx16_), mgl::BufferUsage::StaticDraw);

            describe_attributes(vao_uint_, vbo_big_);
            attach_index_buffer(vao_uint_, ibo_uint_, std::span<const std::uint32_t>(idx32_), mgl::BufferUsage::StaticDraw);

            describe_attributes(vao_dyn_, vbo_dyn_);
            attach_index_buffer(vao_dyn_, ibo_dyn_, std::span<const std::uint8_t>(idx8_), mgl::BufferUsage::DynamicDraw);

            describe_attributes(vao_checker_, vbo_mesh_);  // same vertices, different index buffer
            attach_index_buffer(vao_checker_, ibo_checker_, std::span<const std::uint8_t>(idx_checker_),
                                mgl::BufferUsage::StaticDraw);

            describe_attributes(vao_expanded_, vbo_expanded_);  // no index buffer at all

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        // ---- rows 1-2: the buffer mutation sequence ---------------------------------------------
        void draw_mutation_sequence()
        {
            const auto A = mgl::BufferTarget::Array;
            const auto E = mgl::BufferTarget::ElementArray;
            const mgl::IndexType u8 = mgl::IndexType::UnsignedByte;
            mgl::glBindVertexArray(vao_dyn_);  // from here on ELEMENT_ARRAY_BUFFER means ibo_dyn_

            // (a) Start every frame from the same contents: glBufferData replaces the whole data store.
            mgl::glBindBuffer(A, vbo_dyn_);
            mgl::glBufferData(A, std::span<const Vertex>(mesh_), mgl::BufferUsage::StaticDraw);
            mgl::glBufferData(E, std::span<const std::uint8_t>(idx8_), mgl::BufferUsage::DynamicDraw);
            vbo_stage_[0] = query_buffer(A);
            ibo_stage_[0] = query_buffer(E);
            set_tile(0, 1);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, u8, demo::buffer_offset(0));

            // (b) glBufferSubData on the INDICES: overwrite the two triangles of the center cell (indices 24..29)
            //     with vertex 0 three times each = degenerate triangles, which cover no pixel. Vertices untouched.
            const std::array<std::uint8_t, 6> collapsed = {0, 0, 0, 0, 0, 0};
            mgl::glBufferSubData(E, 4 * 6 * sizeof(std::uint8_t), std::span<const std::uint8_t>(collapsed));
            ibo_stage_[1] = query_buffer(E);  // BufferSubData changes neither size nor usage
            set_tile(1, 1);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, u8, demo::buffer_offset(0));
            // Put the indices back (the earlier draw already consumed the modified data).
            mgl::glBufferSubData(E, 4 * 6 * sizeof(std::uint8_t),
                                 std::span<const std::uint8_t>(idx8_.data() + 24, 6));

            // (c) glBufferSubData on one VERTEX: vertex 5 becomes white and moves. Indices untouched.
            mgl::glBufferSubData(A, 5 * sizeof(Vertex), std::span<const Vertex>(&moved_[5], 1));
            vbo_stage_[1] = query_buffer(A);
            set_tile(2, 1);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, u8, demo::buffer_offset(0));

            // (d) glBufferData again, with a DIFFERENT size and usage: the data store is replaced (32 vertices,
            //     the second half unused padding). The buffer NAME - and so the VAO's bindings - stay valid.
            std::array<Vertex, 2 * kVertexCount> doubled{};
            std::copy(inverted_.begin(), inverted_.end(), doubled.begin());
            mgl::glBufferData(A, std::span<const Vertex>(doubled), mgl::BufferUsage::DynamicDraw);
            vbo_stage_[2] = query_buffer(A);
            set_tile(3, 1);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, u8, demo::buffer_offset(0));

            // (e) "Orphaning": respecify the store with no data (contents undefined, so the GL may hand out new
            //     memory instead of waiting for pending draws), then fill the part that is used.
            mgl::glBufferData(A, static_cast<GLsizeiptr>(sizeof(doubled)), nullptr, mgl::BufferUsage::StreamDraw);
            mgl::glBufferSubData(A, 0, std::span<const Vertex>(rotated_));
            vbo_stage_[3] = query_buffer(A);
            set_tile(0, 2);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, u8, demo::buffer_offset(0));

            mgl::glBindBuffer(A, mgl::BufferId{0});
        }

        // ---- row 3: calls that must fail, with defined errors ------------------------------------
        void draw_negative_cases()
        {
            const mgl::IndexType u8 = mgl::IndexType::UnsignedByte;
            const auto E = mgl::BufferTarget::ElementArray;
            mgl::glBindVertexArray(vao_ubyte_);

            // A negative count is an INVALID_VALUE error (ES 3.0 2.5: negative sizei) and draws nothing.
            // The error codes are read right after the calls and asserted in validate().
            neg_.before_draw = mgl::glGetError();
            set_tile(1, 3);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, -1, u8, demo::buffer_offset(0));
            neg_.draw_count = mgl::glGetError();

            // BufferSubData beyond the end of the data store (offset 50 + 8 bytes > 54) -> INVALID_VALUE; a negative
            // offset likewise (ES 3.0 2.10.2). Neither modifies the buffer, so the draw below is the normal mesh.
            const std::array<std::uint8_t, 8> junk = {7, 7, 7, 7, 7, 7, 7, 7};
            neg_.before_sub = mgl::glGetError();
            mgl::glBufferSubData(E, 50, std::span<const std::uint8_t>(junk));
            neg_.sub_range = mgl::glGetError();
            mgl::glBufferSubData(E, -1, std::span<const std::uint8_t>(junk));
            neg_.sub_negative = mgl::glGetError();
            set_tile(2, 3);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, kIndexCount, u8, demo::buffer_offset(0));
        }

        // ---- validation ------------------------------------------------------------------------
        void check_limits(demo::Check& check)
        {
            GLint64 max_index = 0;
            mgl::glGetInteger64v(mgl::GetParameter::MaxElementIndex, &max_index);
            check.expect(max_index >= (1 << 24) - 1 && max_index >= static_cast<GLint64>(kBigBase) + kVertexCount - 1,
                         "MAX_ELEMENT_INDEX >= 2^24 - 1 (Table 6.28), so the indices 70000..70015 are legal",
                         std::format("MAX_ELEMENT_INDEX = {}", max_index));
        }

        void check_tiles(demo::Check& check)
        {
            const auto canonical = std::span<const Tri>(tris_);
            const auto all = canonical;
            const auto verts = std::span<const Vertex>(mesh_);

            check_tile(check, "UNSIGNED_BYTE indices", tile(0, 0), verts, all, canonical);
            check_tile(check, "UNSIGNED_SHORT indices", tile(1, 0), verts, all, canonical);
            check_tile(check, "UNSIGNED_INT indices 70000..70015 (a 16-bit truncation would show the magenta decoy)",
                       tile(2, 0), verts, all, canonical);
            check_tile(check, "UNSIGNED_SHORT, indices = byte offset 36, count 36: bottom row of cells skipped", tile(3, 0),
                       verts, canonical.subspan(6), canonical);

            check_tile(check, "dynamic buffers, initial glBufferData", tile(0, 1), verts, all, canonical);
            std::vector<Tri> hole(canonical.begin(), canonical.end());
            hole.erase(hole.begin() + 8, hole.begin() + 10);
            check_tile(check, "glBufferSubData on the index buffer: center cell collapsed to a hole", tile(1, 1), verts, hole,
                       canonical);
            check_tile(check, "glBufferSubData on one vertex: white vertex, moved", tile(2, 1), moved_, all, canonical);
            check_tile(check, "glBufferData with 2x the size: new (inverted) colors", tile(3, 1), inverted_, all, canonical);
            check_tile(check, "orphaning (glBufferData with NULL) + glBufferSubData: rotated colors", tile(0, 2), rotated_, all,
                       canonical);

            check_tile(check, "VAO with its own index buffer: only the checkerboard cells", tile(1, 2), verts, checker_tris_,
                       canonical);
            check_tile(check, "back to the first VAO, no glBindBuffer: the full mesh again", tile(2, 2), verts, all, canonical);
            check_tile(check, "client-memory indices on VAO 0 (no ELEMENT_ARRAY_BUFFER)", tile(3, 2), verts, all, canonical);

            check_tile(check, "no indices: 54 expanded vertices with glDrawArrays give the same picture", tile(0, 3), verts, all,
                       canonical);
            check_tile(check, "glDrawElements(count = -1) drew nothing", tile(1, 3), verts, std::span<const Tri>{}, canonical);
            check_tile(check, "after the failed glBufferSubData the buffer still holds the original indices", tile(2, 3), verts,
                       all, canonical);
            check_tile(check, "offset 0, count 18: only the bottom row of cells", tile(3, 3), verts, canonical.first(6), canonical);
        }

        void check_buffer_sizes(demo::Check& check)
        {
            struct Entry
            {
                const char* name;
                mgl::BufferId buffer;
                mgl::VertexArrayId owner;  // for index buffers: the VAO whose ELEMENT_ARRAY_BUFFER binding is the buffer
                GLint size;
            };
            const std::array<Entry, 7> entries = {{
                {"vertex buffer (16 vertices)", vbo_mesh_, {}, static_cast<GLint>(sizeof(mesh_))},
                {"big vertex buffer (70016 vertices)", vbo_big_, {}, static_cast<GLint>((kBigBase + kVertexCount) * sizeof(Vertex))},
                {"expanded vertex buffer (54 vertices)", vbo_expanded_, {}, static_cast<GLint>(kIndexCount * sizeof(Vertex))},
                {"UNSIGNED_BYTE index buffer", ibo_ubyte_, vao_ubyte_, static_cast<GLint>(kIndexCount * sizeof(std::uint8_t))},
                {"UNSIGNED_SHORT index buffer", ibo_ushort_, vao_ushort_, static_cast<GLint>(kIndexCount * sizeof(std::uint16_t))},
                {"UNSIGNED_INT index buffer", ibo_uint_, vao_uint_, static_cast<GLint>(kIndexCount * sizeof(std::uint32_t))},
                {"checkerboard index buffer", ibo_checker_, vao_checker_, static_cast<GLint>(idx_checker_.size())},
            }};
            GLint indexed_bytes = 0, expanded_bytes = 0;
            for (const Entry& e : entries)
            {
                BufferState s;
                if (e.owner.value != 0)
                {
                    mgl::glBindVertexArray(e.owner);  // ELEMENT_ARRAY_BUFFER now refers to the VAO's index buffer
                    s = query_buffer(mgl::BufferTarget::ElementArray);
                }
                else
                {
                    mgl::glBindBuffer(mgl::BufferTarget::Array, e.buffer);
                    s = query_buffer(mgl::BufferTarget::Array);
                }
                check.expect(s.size == e.size && s.usage == usage_value(mgl::BufferUsage::StaticDraw),
                             std::format("{}: BUFFER_SIZE = {} bytes, BUFFER_USAGE = STATIC_DRAW", e.name, e.size),
                             std::format("size={} usage={:#x}", s.size, s.usage));
                if (e.buffer.value == vbo_mesh_.value || e.buffer.value == ibo_ubyte_.value) indexed_bytes += s.size;
                if (e.buffer.value == vbo_expanded_.value) expanded_bytes = s.size;
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            check.expect(indexed_bytes < expanded_bytes,
                         "vertex reuse: 16 vertices + 54 byte indices take less memory than 54 expanded vertices",
                         std::format("indexed = {} bytes, expanded = {} bytes", indexed_bytes, expanded_bytes));
        }

        void check_mutation_state(demo::Check& check)
        {
            // State recorded inside render() after each step of the mutation sequence (see draw_mutation_sequence()).
            const GLint v = static_cast<GLint>(sizeof(Vertex));
            auto state_is = [&](const BufferState& s, GLint size, mgl::BufferUsage usage) {
                return s.size == size && s.usage == usage_value(usage);
            };
            check.expect(state_is(vbo_stage_[0], kVertexCount * v, mgl::BufferUsage::StaticDraw),
                         "vertex buffer after glBufferData(320 bytes, STATIC_DRAW): size 320, usage STATIC_DRAW",
                         std::format("size={} usage={:#x}", vbo_stage_[0].size, vbo_stage_[0].usage));
            check.expect(state_is(vbo_stage_[1], kVertexCount * v, mgl::BufferUsage::StaticDraw),
                         "glBufferSubData does not change BUFFER_SIZE or BUFFER_USAGE",
                         std::format("size={} usage={:#x}", vbo_stage_[1].size, vbo_stage_[1].usage));
            check.expect(state_is(vbo_stage_[2], 2 * kVertexCount * v, mgl::BufferUsage::DynamicDraw),
                         "glBufferData(640 bytes, DYNAMIC_DRAW) re-specifies: size 640, usage DYNAMIC_DRAW",
                         std::format("size={} usage={:#x}", vbo_stage_[2].size, vbo_stage_[2].usage));
            check.expect(state_is(vbo_stage_[3], 2 * kVertexCount * v, mgl::BufferUsage::StreamDraw),
                         "orphaning with glBufferData(640, NULL, STREAM_DRAW): size 640, usage STREAM_DRAW",
                         std::format("size={} usage={:#x}", vbo_stage_[3].size, vbo_stage_[3].usage));
            check.expect(state_is(ibo_stage_[0], kIndexCount, mgl::BufferUsage::DynamicDraw),
                         "index buffer after glBufferData(54 bytes, DYNAMIC_DRAW)",
                         std::format("size={} usage={:#x}", ibo_stage_[0].size, ibo_stage_[0].usage));
            check.expect(state_is(ibo_stage_[1], kIndexCount, mgl::BufferUsage::DynamicDraw),
                         "glBufferSubData on the index buffer keeps size 54 and usage DYNAMIC_DRAW",
                         std::format("size={} usage={:#x}", ibo_stage_[1].size, ibo_stage_[1].usage));

            // The VAO keeps referring to the same buffer *names* through all the respecifications.
            mgl::glBindVertexArray(vao_dyn_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::ElementArrayBufferBinding), static_cast<int>(ibo_dyn_.value),
                            "after the re-specification the VAO's ELEMENT_ARRAY_BUFFER binding is still the same buffer name");
            GLint bound = -1;
            mgl::glGetVertexAttribiv(mgl::AttribLocation{0u}, mgl::VertexAttribParameter::ArrayBufferBinding, &bound);
            check.expect_eq(bound, static_cast<GLint>(vbo_dyn_.value),
                            "... and the position attribute still sources from the same vertex buffer name");
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_dyn_);
            const BufferState last = query_buffer(mgl::BufferTarget::Array);
            check.expect(state_is(last, 2 * kVertexCount * v, mgl::BufferUsage::StreamDraw),
                         "queried again after the frame: the vertex buffer is still 640 bytes, STREAM_DRAW");
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void check_vao_bindings(demo::Check& check)
        {
            using Q = mgl::GetParameter;
            struct Entry
            {
                const char* name;
                mgl::VertexArrayId vao;
                mgl::BufferId expected;
            };
            const std::array<Entry, 6> entries = {{
                {"UNSIGNED_BYTE VAO", vao_ubyte_, ibo_ubyte_},
                {"UNSIGNED_SHORT VAO", vao_ushort_, ibo_ushort_},
                {"UNSIGNED_INT VAO", vao_uint_, ibo_uint_},
                {"dynamic VAO", vao_dyn_, ibo_dyn_},
                {"checkerboard VAO", vao_checker_, ibo_checker_},
                {"non-indexed VAO", vao_expanded_, {}},
            }};
            // ARRAY_BUFFER is not VAO state: bind a buffer to it and watch it survive every VAO switch.
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_mesh_);
            bool array_binding_kept = true;
            for (const Entry& e : entries)
            {
                mgl::glBindVertexArray(e.vao);
                check.expect_eq(demo::get_integer(Q::ElementArrayBufferBinding), static_cast<int>(e.expected.value),
                                std::format("{}: ELEMENT_ARRAY_BUFFER_BINDING is its own index buffer", e.name));
                array_binding_kept = array_binding_kept && demo::get_integer(Q::ArrayBufferBinding) == static_cast<int>(vbo_mesh_.value);
            }
            check.expect(array_binding_kept, "ARRAY_BUFFER_BINDING is not VAO state: it survives every glBindVertexArray");

            // The default VAO has an element array binding of its own.
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect_eq(demo::get_integer(Q::ElementArrayBufferBinding), 0, "VAO 0: ELEMENT_ARRAY_BUFFER_BINDING is 0");
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ibo_checker_);
            check.expect_eq(demo::get_integer(Q::ElementArrayBufferBinding), static_cast<int>(ibo_checker_.value),
                            "glBindBuffer(ELEMENT_ARRAY_BUFFER) while VAO 0 is bound sets VAO 0's binding");
            mgl::glBindVertexArray(vao_ubyte_);
            check.expect_eq(demo::get_integer(Q::ElementArrayBufferBinding), static_cast<int>(ibo_ubyte_.value),
                            "... and did not touch the other VAO");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect_eq(demo::get_integer(Q::ElementArrayBufferBinding), static_cast<int>(ibo_checker_.value),
                            "VAO 0 remembers its element array buffer");
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});

            // Unbinding while a VAO is bound clears that VAO's index buffer - and only that VAO's.
            mgl::VertexArrayId scratch{};
            mgl::glGenVertexArrays(1, &scratch);
            mgl::glBindVertexArray(scratch);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ibo_ubyte_);
            const int bound = demo::get_integer(Q::ElementArrayBufferBinding);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
            const int unbound = demo::get_integer(Q::ElementArrayBufferBinding);
            mgl::glBindVertexArray(vao_ubyte_);
            check.expect(bound == static_cast<int>(ibo_ubyte_.value) && unbound == 0 &&
                             demo::get_integer(Q::ElementArrayBufferBinding) == static_cast<int>(ibo_ubyte_.value),
                            "glBindBuffer(ELEMENT_ARRAY_BUFFER, 0) with a VAO bound unbinds only that VAO's index buffer");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDeleteVertexArrays(1, &scratch);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void check_buffer_objects(demo::Check& check)
        {
            const auto A = mgl::BufferTarget::Array;
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            check.expect(!mgl::glIsBuffer(b), "glIsBuffer is FALSE for a name that was generated but never bound (ES 3.0 2.10, 6.1.9)");
            mgl::glBindBuffer(A, b);
            check.expect(mgl::glIsBuffer(b), "glIsBuffer is TRUE once the name has been bound (the buffer object exists)");
            BufferState s = query_buffer(A);
            check.expect(s.size == 0 && s.usage == usage_value(mgl::BufferUsage::StaticDraw),
                         "a new buffer object starts with BUFFER_SIZE 0 and BUFFER_USAGE STATIC_DRAW (Table 2.7)",
                         std::format("size={} usage={:#x}", s.size, s.usage));

            const std::vector<std::uint8_t> bytes(512, 1);
            mgl::glBufferData(A, 100, nullptr, mgl::BufferUsage::StreamDraw);
            s = query_buffer(A);
            check.expect(s.size == 100 && s.usage == usage_value(mgl::BufferUsage::StreamDraw),
                         "glBufferData(100, NULL, STREAM_DRAW): size 100, usage STREAM_DRAW (contents undefined)");
            mgl::glBufferData(A, 400, bytes.data(), mgl::BufferUsage::DynamicDraw);
            s = query_buffer(A);
            check.expect(s.size == 400 && s.usage == usage_value(mgl::BufferUsage::DynamicDraw),
                         "glBufferData(400, data, DYNAMIC_DRAW) re-specifies the same buffer: size 400, usage DYNAMIC_DRAW");
            mgl::glBufferSubData(A, 10, 20, bytes.data());
            s = query_buffer(A);
            check.expect(s.size == 400 && s.usage == usage_value(mgl::BufferUsage::DynamicDraw),
                         "glBufferSubData leaves BUFFER_SIZE and BUFFER_USAGE alone");

            bool usages_ok = true;
            std::string detail;
            for (const mgl::BufferUsage u : {mgl::BufferUsage::StreamDraw, mgl::BufferUsage::StreamRead, mgl::BufferUsage::StreamCopy,
                                             mgl::BufferUsage::StaticDraw, mgl::BufferUsage::StaticRead, mgl::BufferUsage::StaticCopy,
                                             mgl::BufferUsage::DynamicDraw, mgl::BufferUsage::DynamicRead, mgl::BufferUsage::DynamicCopy})
            {
                mgl::glBufferData(A, 16, nullptr, u);
                s = query_buffer(A);
                if (s.usage != usage_value(u) || s.size != 16)
                {
                    usages_ok = false;
                    detail += std::format("usage {:#x} reads back {:#x}; ", usage_value(u), s.usage);
                }
            }
            check.expect(usages_ok, "all nine BUFFER_USAGE values are stored and read back exactly", detail);
            mgl::glBufferData(A, 0, nullptr, mgl::BufferUsage::StaticDraw);
            check.expect_eq(query_buffer(A).size, 0, "glBufferData with size 0 empties the buffer (BUFFER_SIZE 0)");
            check.expect_no_gl_error("buffer specification and queries raised no GL error");

            mgl::glBufferData(A, -1, nullptr, mgl::BufferUsage::StaticDraw);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBufferData with a negative size -> INVALID_VALUE");

            mgl::glDeleteBuffers(1, &b);
            check.expect(!mgl::glIsBuffer(b), "glIsBuffer is FALSE after glDeleteBuffers");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ArrayBufferBinding), 0,
                            "deleting the bound buffer resets the binding to 0 (ES 3.0 2.10.1)");
            GLint untouched = 42;
            mgl::glGetBufferParameteriv(A, mgl::BufferParameter::Size, &untouched);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "querying a target with no buffer bound (zero) -> INVALID_OPERATION (ES 3.0 2.10.1)");
            mgl::glBufferSubData(mgl::BufferTarget::ElementArray, 0, 1, bytes.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "glBufferSubData with ELEMENT_ARRAY_BUFFER bound to zero (VAO 0) -> INVALID_OPERATION");
        }

        void check_negative_cases(demo::Check& check)
        {
            check.expect_eq(static_cast<int>(neg_.before_draw), static_cast<int>(mgl::ErrorCode::NoError),
                            "no pending error before the negative draw test");
            check.expect(neg_.draw_count == mgl::ErrorCode::InvalidValue,
                         "glDrawElements with count = -1 -> INVALID_VALUE (ES 3.0 2.5: negative sizei)",
                         std::format("glGetError() = {}", demo::error_name(neg_.draw_count)));
            check.expect_eq(static_cast<int>(neg_.before_sub), static_cast<int>(mgl::ErrorCode::NoError),
                            "no pending error before the negative glBufferSubData test");
            check.expect(neg_.sub_range == mgl::ErrorCode::InvalidValue,
                         "glBufferSubData(offset 50, 8 bytes) past the end of a 54 byte buffer -> INVALID_VALUE (ES 3.0 2.10.2)",
                         std::format("glGetError() = {}", demo::error_name(neg_.sub_range)));
            check.expect(neg_.sub_negative == mgl::ErrorCode::InvalidValue,
                         "glBufferSubData with a negative offset -> INVALID_VALUE (ES 3.0 2.10.2)",
                         std::format("glGetError() = {}", demo::error_name(neg_.sub_negative)));
        }

        // ---- members --------------------------------------------------------------------------
        int tile_w_ = 64;
        int tile_h_ = 64;
        mgl::ProgramId program_{};

        Vertices mesh_{}, moved_{}, inverted_{}, rotated_{};
        std::array<Tri, kTriangleCount> tris_{};
        std::vector<Tri> checker_tris_;
        std::array<std::uint8_t, kIndexCount> idx8_{};
        std::array<std::uint16_t, kIndexCount> idx16_{};
        std::array<std::uint32_t, kIndexCount> idx32_{};
        std::vector<std::uint8_t> idx_checker_;

        mgl::BufferId vbo_mesh_{}, vbo_big_{}, vbo_dyn_{}, vbo_expanded_{};
        mgl::BufferId ibo_ubyte_{}, ibo_ushort_{}, ibo_uint_{}, ibo_dyn_{}, ibo_checker_{};
        mgl::VertexArrayId vao_ubyte_{}, vao_ushort_{}, vao_uint_{}, vao_dyn_{}, vao_checker_{}, vao_expanded_{};

        // Observed in render(), asserted in validate().
        std::array<BufferState, 4> vbo_stage_{};
        std::array<BufferState, 2> ibo_stage_{};
        struct NegativeResults
        {
            mgl::ErrorCode before_draw = mgl::ErrorCode::NoError, draw_count = mgl::ErrorCode::NoError;
            mgl::ErrorCode before_sub = mgl::ErrorCode::NoError, sub_range = mgl::ErrorCode::NoError;
            mgl::ErrorCode sub_negative = mgl::ErrorCode::NoError;
        } neg_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-indexed-drawing",
        .title = "Indexed drawing",
        .min_es = {3, 0},
        .summary = "glDrawElements with UNSIGNED_BYTE/SHORT/INT indices and buffer offsets, glBufferSubData and "
                   "glBufferData re-specification, buffer queries, and the VAO-owned ELEMENT_ARRAY_BUFFER binding.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<IndexedDrawing>(argc, argv, info);
}
