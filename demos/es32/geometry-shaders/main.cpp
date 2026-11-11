// es32-geometry-shaders
//
// Geometry shaders (OpenGL ES 3.2, section 11.3; GLSL ES 3.20 sections 4.4.1, 4.4.2, 7.1.4, 8.13). A geometry
// shader sits between the vertex and the fragment shader, sees one whole primitive (with its neighbours for the
// adjacency types) and emits any number of new primitives of one output type. The 256x256 image is a 2x2 grid of
// 3D scenes, each drawn with one draw call per program and a perspective camera:
//
//   top left      A  amplification: 151 POINTS become camera-facing quads (layout(points) in, a 4-vertex
//                    triangle_strip out); the fragment shader turns each quad into a lit sphere. The colour comes
//                    from gl_PrimitiveIDIn.
//   top right     B  an icosphere whose 80 TRIANGLES become prisms: a cap strip and three side strips separated by
//                    EndPrimitive (max_vertices = 15); height tier and colour come from gl_PrimitiveIDIn /
//                    gl_PrimitiveID.
//   bottom left   C  TRIANGLES_ADJACENCY: a torus is drawn normally, then again with an adjacency index buffer; the
//                    geometry shader finds the silhouette edges (front-facing triangle next to a back-facing one)
//                    and turns each into a thin quad.
//   bottom right  D  layout(invocations = 6): ONE triangle list (a cube) is placed six times by gl_InvocationID.
//
// All programs pass data VS -> GS -> FS through interface blocks. validate() compares the pixels of every cell with
// a CPU model that re-implements the four geometry shaders and rasterizes their output (cpu_model.hpp), captures the
// geometry shader output with transform feedback and compares it exactly with the CPU model, and then goes through
// program queries, limits, the draw-mode / input-type mismatch error, and the link rules for geometry shaders.

#include "cpu_model.hpp"

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
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;
namespace cm = gsmodel;
using demo::math::Mat4;
using demo::math::Vec3;
using demo::math::Vec4;

namespace
{
    enum Cell : int { kCellA = 0, kCellB = 1, kCellC = 2, kCellD = 3 };
    constexpr int kCells = 4;

    // Clear colours of the cells (8-bit values, so the clear is exactly representable).
    constexpr std::array<std::array<std::uint8_t, 3>, kCells> kClear8 = {{{10, 14, 38}, {30, 18, 40}, {12, 32, 34}, {40, 28, 16}}};

    constexpr float kTime0 = 1.1f;          // animation time of frame 0 (seconds)
    constexpr float kCrystalLift = 0.5f;    // u_lift of cell B (power of two: exact arithmetic)
    constexpr float kLineHalfWidthPx = 1.25f;
    constexpr float kLineBias = 0.002f;     // clip-space z bias of the silhouette quads
    constexpr float kMarginPx = 0.75f;      // pixels: edge margin of the certain / may-cover CPU rasterization
    constexpr int kPixelTolerance = 3;      // 8-bit levels
    constexpr float kAgreeLevels = static_cast<float>(kPixelTolerance) / 255.0f;
    constexpr float kJitterPx = 1.0f / 16.0f;  // SUBPIXEL_BITS >= 4 (ES 3.2 Table 21.40): positions are only specified to 1/16 pixel
    constexpr float kSensitivityLevels = 2.0f / 255.0f;

    constexpr std::array<const GLchar*, 0> kNoTf{};
    constexpr std::array<const GLchar*, 3> kTfBillboard = {"GS_OUT.eye", "GS_OUT.uv", "GS_OUT.prim"};
    constexpr std::array<const GLchar*, 3> kTfCrystal = {"GS_OUT.world", "GS_OUT.normal", "GS_OUT.prim"};
    constexpr std::array<const GLchar*, 2> kTfSilhouette = {"GS_OUT.edge", "GS_OUT.side"};
    constexpr std::array<const GLchar*, 4> kTfRing = {"GS_OUT.world", "GS_OUT.normal", "GS_OUT.inst", "GS_OUT.face"};

    mgl::UniformLocation loc(int n) { return mgl::UniformLocation{n}; }

    void set_mat4(int location, const Mat4& m) { mgl::glUniformMatrix4fv(loc(location), 1, GL_FALSE, m.data()); }
    void set_vec3(int location, Vec3 v) { mgl::glUniform3f(loc(location), v.x, v.y, v.z); }

    struct Camera
    {
        Mat4 view, proj, viewproj;
        Vec3 eye;
    };

    Camera make_camera(float azimuth, float elevation_deg, float distance, float fov_deg, float aspect)
    {
        const float el = demo::math::radians(elevation_deg);
        Camera c;
        c.eye = {distance * std::cos(el) * std::cos(azimuth), distance * std::sin(el), distance * std::cos(el) * std::sin(azimuth)};
        c.view = demo::math::look_at(c.eye, {0, 0, 0}, {0, 1, 0});
        c.proj = demo::math::perspective(demo::math::radians(fov_deg), aspect, 1.0f, 30.0f);
        c.viewproj = c.proj * c.view;
        return c;
    }

    // Everything that depends on the animation time.
    struct Scene
    {
        std::array<Camera, kCells> cam;
        Vec3 light{};  // unit vector towards the light (world space)
        std::array<Vec3, 6> place{};
        std::array<cm::Vec2, 6> spin{};
    };

    Scene make_scene(float t, float aspect)
    {
        Scene s;
        s.light = cm::normalize({0.55f, 0.75f, 0.45f});
        s.cam[kCellA] = make_camera(0.9f + 0.35f * t, 34.0f, 7.6f, 38.0f, aspect);
        s.cam[kCellB] = make_camera(2.1f + 0.5f * t, 24.0f, 5.0f, 40.0f, aspect);
        s.cam[kCellC] = make_camera(0.4f + 0.45f * t, 38.0f, 4.7f, 40.0f, aspect);
        s.cam[kCellD] = make_camera(1.0f + 0.2f * t, 30.0f, 6.3f, 38.0f, aspect);
        for (int k = 0; k < 6; ++k)
        {
            const float phi = static_cast<float>(k) * demo::math::radians(60.0f) + 0.3f * t;
            const float spin = 0.9f * t + 0.5f * static_cast<float>(k);
            s.place[static_cast<std::size_t>(k)] = {1.55f * std::cos(phi), 0.35f * std::sin(1.2f * t + 1.05f * static_cast<float>(k)), 1.55f * std::sin(phi)};
            s.spin[static_cast<std::size_t>(k)] = {std::cos(spin), std::sin(spin)};
        }
        return s;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Transform feedback layouts: words per captured vertex, in the order of the varying lists above.
    // ---------------------------------------------------------------------------------------------------------
    enum class Layout { Billboard, Crystal, Silhouette, Ring };

    std::uint32_t bits_of(float f)
    {
        std::uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        return u;
    }
    std::uint32_t bits_of(int i) { return static_cast<std::uint32_t>(i); }

    std::vector<std::uint32_t> pack(const std::vector<cm::Strip>& strips, Layout layout)
    {
        std::vector<std::uint32_t> w;
        for (const cm::Strip& s : strips)
            for (const auto& tri : cm::strip_order(s.size()))
                for (const std::size_t i : tri)
                {
                    const cm::EVert& v = s[i];
                    switch (layout)
                    {
                    case Layout::Billboard:
                        for (const float f : {v.world.x, v.world.y, v.world.z, v.uv.x, v.uv.y}) w.push_back(bits_of(f));
                        w.push_back(bits_of(v.prim));
                        break;
                    case Layout::Crystal:
                        for (const float f : {v.world.x, v.world.y, v.world.z, v.normal.x, v.normal.y, v.normal.z}) w.push_back(bits_of(f));
                        w.push_back(bits_of(v.prim));
                        break;
                    case Layout::Silhouette:
                        for (const float f : {v.world.x, v.world.y, v.world.z, v.side}) w.push_back(bits_of(f));
                        break;
                    case Layout::Ring:
                        for (const float f : {v.world.x, v.world.y, v.world.z, v.normal.x, v.normal.y, v.normal.z}) w.push_back(bits_of(f));
                        w.push_back(bits_of(v.inst));
                        w.push_back(bits_of(v.prim));
                        break;
                    }
                }
        return w;
    }

    std::size_t triangle_count(const std::vector<cm::Strip>& strips)
    {
        std::size_t n = 0;
        for (const cm::Strip& s : strips) n += s.size() >= 3 ? s.size() - 2 : 0;
        return n;
    }

    // +0.0 and -0.0 are the same value for a float, so they compare equal; everything else must match bit for bit.
    bool same_word(std::uint32_t a, std::uint32_t b) { return a == b || ((a | b) & 0x7fffffffu) == 0u; }

    struct WordDiff
    {
        std::size_t mismatches = 0;
        std::size_t first = 0;
    };
    WordDiff compare_words(const std::vector<std::uint32_t>& got, const std::vector<std::uint32_t>& expected)
    {
        WordDiff d;
        if (got.size() != expected.size())
        {
            d.mismatches = std::max(got.size(), expected.size());
            return d;
        }
        for (std::size_t i = 0; i < got.size(); ++i)
            if (!same_word(got[i], expected[i]))
            {
                if (d.mismatches == 0) d.first = i;
                ++d.mismatches;
            }
        return d;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Pixel comparison with the CPU model
    // ---------------------------------------------------------------------------------------------------------
    struct CompareStats
    {
        int compared = 0;    // pixels whose expectation is unambiguous
        int covered = 0;     // ... of which covered by some primitive
        int skipped = 0;     // pixels near a primitive edge or a depth tie (strict and loose model disagree)
        int mismatches = 0;
        int max_diff = 0;
        int first_x = -1, first_y = -1;
        std::vector<std::string> samples;  // the first few mismatching pixels, for the log
    };

    // `gpu` holds a w*h RGBA8 block (bottom-up rows) of one cell.
    template <typename Shade>
    CompareStats compare_cell(const std::vector<std::uint8_t>& gpu, int w, int h, const cm::Raster& raster,
                              std::array<std::uint8_t, 3> background, Shade&& shade)
    {
        CompareStats st;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const cm::Pixel model = raster.resolve(x, y, kMarginPx, kAgreeLevels, kJitterPx, kSensitivityLevels,
                                                       {static_cast<float>(background[0]) / 255.0f, static_cast<float>(background[1]) / 255.0f,
                                                        static_cast<float>(background[2]) / 255.0f},
                                                       shade);
                if (model.ambiguous)
                {
                    ++st.skipped;
                    continue;
                }
                std::array<std::uint8_t, 3> expected = background;
                if (model.covered)
                {
                    expected = {cm::to8(model.color.r), cm::to8(model.color.g), cm::to8(model.color.b)};
                    ++st.covered;
                }
                ++st.compared;
                const std::uint8_t* px = &gpu[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4];
                int diff = 0;
                for (std::size_t c = 0; c < 3; ++c) diff = std::max(diff, std::abs(static_cast<int>(px[c]) - static_cast<int>(expected[c])));
                if (px[3] != 255) diff = 255;
                st.max_diff = std::max(st.max_diff, diff);
                if (diff > kPixelTolerance)
                {
                    if (st.samples.size() < 6)
                        st.samples.push_back(std::format("({},{}) gpu=({},{},{}) model=({},{},{})", x, y, px[0], px[1], px[2], expected[0], expected[1], expected[2]));
                    if (st.mismatches == 0)
                    {
                        st.first_x = x;
                        st.first_y = y;
                    }
                    ++st.mismatches;
                }
            }
        return st;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Small shader sources used by the matrix of input types and by the link-rule tests
    // ---------------------------------------------------------------------------------------------------------
    constexpr std::string_view kTinyVs =
        "#version 320 es\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
    constexpr std::string_view kTinyFs =
        "#version 320 es\n"
        "precision highp float;\n"
        "layout(location = 0) uniform vec4 u_color;\n"
        "out vec4 frag_color;\n"
        "void main() { frag_color = u_color; }\n";

    struct TinyKind
    {
        const char* name;
        const char* in_layout;
        const char* out_layout;
        int max_vertices;
        const char* body;
        GLenum in_type;
        GLenum out_type;
    };
    // One geometry shader per input primitive type (ES 3.2 section 11.3.1); each emits a valid output primitive.
    constexpr std::array<TinyKind, 5> kTinyKinds = {{
        {"points", "points", "points", 1, "gl_Position = gl_in[0].gl_Position; EmitVertex();", GL_POINTS, GL_POINTS},
        {"lines", "lines", "line_strip", 2, "gl_Position = gl_in[0].gl_Position; EmitVertex(); gl_Position = gl_in[1].gl_Position; EmitVertex();",
         GL_LINES, GL_LINE_STRIP},
        {"lines_adjacency", "lines_adjacency", "points", 1, "gl_Position = gl_in[1].gl_Position; EmitVertex();", GL_LINES_ADJACENCY, GL_POINTS},
        {"triangles", "triangles", "triangle_strip", 3,
         "gl_Position = gl_in[0].gl_Position; EmitVertex(); gl_Position = gl_in[1].gl_Position; EmitVertex(); gl_Position = gl_in[2].gl_Position; EmitVertex();",
         GL_TRIANGLES, GL_TRIANGLE_STRIP},
        {"triangles_adjacency", "triangles_adjacency", "triangle_strip", 3,
         "gl_Position = gl_in[0].gl_Position; EmitVertex(); gl_Position = gl_in[2].gl_Position; EmitVertex(); gl_Position = gl_in[4].gl_Position; EmitVertex();",
         GL_TRIANGLES_ADJACENCY, GL_TRIANGLE_STRIP},
    }};

    std::string tiny_gs(const TinyKind& k)
    {
        return std::format("#version 320 es\nlayout({}) in;\nlayout({}, max_vertices = {}) out;\nvoid main() {{ {} }}\n", k.in_layout,
                           k.out_layout, k.max_vertices, k.body);
    }

    struct DrawMode
    {
        mgl::PrimitiveType mode;
        const char* name;
        int compatible_with;  // index into kTinyKinds of the only input type that accepts this mode (ES 3.2 section 11.3.4.6)
    };
    constexpr std::array<DrawMode, 11> kDrawModes = {{
        {mgl::PrimitiveType::Points, "POINTS", 0},
        {mgl::PrimitiveType::Lines, "LINES", 1},
        {mgl::PrimitiveType::LineLoop, "LINE_LOOP", 1},
        {mgl::PrimitiveType::LineStrip, "LINE_STRIP", 1},
        {mgl::PrimitiveType::Triangles, "TRIANGLES", 3},
        {mgl::PrimitiveType::TriangleStrip, "TRIANGLE_STRIP", 3},
        {mgl::PrimitiveType::TriangleFan, "TRIANGLE_FAN", 3},
        {mgl::PrimitiveType::LinesAdjacency, "LINES_ADJACENCY", 2},
        {mgl::PrimitiveType::LineStripAdjacency, "LINE_STRIP_ADJACENCY", 2},
        {mgl::PrimitiveType::TrianglesAdjacency, "TRIANGLES_ADJACENCY", 4},
        {mgl::PrimitiveType::TriangleStripAdjacency, "TRIANGLE_STRIP_ADJACENCY", 4},
    }};

    struct LinkOutcome
    {
        bool compiled = true;  // every shader compiled
        bool linked = false;
        std::string log;
    };

    class GeometryShaders final : public demo::Demo
    {
    public:
        // =====================================================================================================
        void init(demo::Env& env) override
        {
            // --- programs ----------------------------------------------------------------------------------
            // The geometry-shader programs declare their transform feedback varyings before linking; that state is
            // inert until BeginTransformFeedback, so the same programs render and (in validate()) capture.
            prog_a_ = link_stages(env, {{mgl::ShaderType::Vertex, "billboard.vert"}, {mgl::ShaderType::Geometry, "billboard.geom"},
                                        {mgl::ShaderType::Fragment, "billboard.frag"}}, kTfBillboard);
            prog_b_ = link_stages(env, {{mgl::ShaderType::Vertex, "crystal.vert"}, {mgl::ShaderType::Geometry, "crystal.geom"},
                                        {mgl::ShaderType::Fragment, "crystal.frag"}}, kTfCrystal);
            prog_c_fill_ = link_stages(env, {{mgl::ShaderType::Vertex, "mesh.vert"}, {mgl::ShaderType::Fragment, "mesh.frag"}}, kNoTf);
            prog_c_sil_ = link_stages(env, {{mgl::ShaderType::Vertex, "mesh.vert"}, {mgl::ShaderType::Geometry, "silhouette.geom"},
                                            {mgl::ShaderType::Fragment, "silhouette.frag"}}, kTfSilhouette);
            prog_d_ = link_stages(env, {{mgl::ShaderType::Vertex, "ring.vert"}, {mgl::ShaderType::Geometry, "ring.geom"},
                                        {mgl::ShaderType::Fragment, "ring.frag"}}, kTfRing);

            // --- geometry ----------------------------------------------------------------------------------
            sprites_ = cm::galaxy();
            std::vector<float> sprite_data;
            for (const cm::Sprite& s : sprites_) sprite_data.insert(sprite_data.end(), {s.position.x, s.position.y, s.position.z, s.size});
            ico_ = cm::icosphere();
            torus_ = cm::torus(24, 12, 1.0f, 0.42f);
            torus_adj_ = cm::adjacency_indices(torus_);
            cube_ = cm::cube();

            std::vector<float> torus_data;
            for (std::size_t i = 0; i < torus_.position.size(); ++i)
                torus_data.insert(torus_data.end(), {torus_.position[i].x, torus_.position[i].y, torus_.position[i].z, torus_.normal[i].x,
                                                     torus_.normal[i].y, torus_.normal[i].z});

            mgl::glGenBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            upload(mgl::BufferTarget::Array, buf_[kVboSprites], std::span<const float>(sprite_data));
            upload(mgl::BufferTarget::Array, buf_[kVboIco], std::span<const Vec3>(ico_.position));
            upload(mgl::BufferTarget::ElementArray, buf_[kEboIco], std::span<const std::uint16_t>(ico_.index));
            upload(mgl::BufferTarget::Array, buf_[kVboTorus], std::span<const float>(torus_data));
            upload(mgl::BufferTarget::ElementArray, buf_[kEboTorus], std::span<const std::uint16_t>(torus_.index));
            upload(mgl::BufferTarget::ElementArray, buf_[kEboTorusAdj], std::span<const std::uint16_t>(torus_adj_));
            upload(mgl::BufferTarget::Array, buf_[kVboCube], std::span<const Vec3>(cube_.position));
            upload(mgl::BufferTarget::ElementArray, buf_[kEboCube], std::span<const std::uint16_t>(cube_.index));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});

            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            // A: points (position + size)
            mgl::glBindVertexArray(vao_[kVaoA]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kVboSprites]);
            attrib(0, 3, 16, 0);
            attrib(1, 1, 16, 12);
            // B: icosphere, positions + element array
            mgl::glBindVertexArray(vao_[kVaoB]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kVboIco]);
            attrib(0, 3, 12, 0);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buf_[kEboIco]);
            // C: torus (position + normal) with the plain triangle list ...
            mgl::glBindVertexArray(vao_[kVaoCFill]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kVboTorus]);
            attrib(0, 3, 24, 0);
            attrib(1, 3, 24, 12);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buf_[kEboTorus]);
            // ... and with the adjacency index list (the ELEMENT_ARRAY_BUFFER binding is part of the VAO)
            mgl::glBindVertexArray(vao_[kVaoCSil]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kVboTorus]);
            attrib(0, 3, 24, 0);
            attrib(1, 3, 24, 12);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buf_[kEboTorusAdj]);
            // D: cube
            mgl::glBindVertexArray(vao_[kVaoD]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kVboCube]);
            attrib(0, 3, 12, 0);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buf_[kEboCube]);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            mgl::glGenBuffers(1, &tf_buffer_);
            mgl::glGenQueries(static_cast<GLsizei>(queries_.size()), queries_.data());
        }

        // =====================================================================================================
        void render(demo::Env& env, int frame) override
        {
            time_ = kTime0 + static_cast<float>(frame) / 60.0f;
            cell_w_ = env.width / 2;
            cell_h_ = env.height / 2;
            scene_ = make_scene(time_, static_cast<float>(cell_w_) / static_cast<float>(cell_h_));

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDisable(mgl::Capability::RasterizerDiscard);
            mgl::glDisable(mgl::Capability::CullFace);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glScissor(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            for (int c = 0; c < kCells; ++c)
            {
                const int x = cell_x(c), y = cell_y(c);
                mgl::glViewport(x, y, cell_w_, cell_h_);
                mgl::glScissor(x, y, cell_w_, cell_h_);
                const auto& bg = kClear8[static_cast<std::size_t>(c)];
                mgl::glClearColor(static_cast<float>(bg[0]) / 255.0f, static_cast<float>(bg[1]) / 255.0f, static_cast<float>(bg[2]) / 255.0f, 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);
                switch (c)  // viewport and scissor stay on this cell for its draw calls
                {
                case kCellA: draw_a(scene_.cam[kCellA].view, scene_.cam[kCellA].proj, light_eye(scene_.cam[kCellA])); break;
                case kCellB: draw_b(scene_.cam[kCellB]); break;
                case kCellC: draw_c(scene_.cam[kCellC]); break;
                default: draw_d(scene_.cam[kCellD], scene_.place, scene_.spin); break;
                }
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        // =====================================================================================================
        void validate(demo::Env& env, demo::Check& check) override
        {
            demo::drain_gl_errors();
            check_limits(check);
            check_program_queries(check);
            check_pixels(env, check);
            check_transform_feedback(check);
            check_adjacency_vertex_order(check);
            check_draw_mode_matrix(check);
            check_link_rules(env, check);
            demo::drain_gl_errors();
        }

        // =====================================================================================================
        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            for (const mgl::ProgramId p : {prog_a_, prog_b_, prog_c_fill_, prog_c_sil_, prog_d_}) mgl::glDeleteProgram(p);
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glDeleteBuffers(1, &tf_buffer_);
            mgl::glDeleteQueries(static_cast<GLsizei>(queries_.size()), queries_.data());
        }

    private:
        enum Buffers : std::size_t { kVboSprites, kVboIco, kEboIco, kVboTorus, kEboTorus, kEboTorusAdj, kVboCube, kEboCube, kBufferCount };
        enum Vaos : std::size_t { kVaoA, kVaoB, kVaoCFill, kVaoCSil, kVaoD, kVaoCount };

        int cell_x(int c) const { return (c % 2) * cell_w_; }
        int cell_y(int c) const { return (1 - c / 2) * cell_h_; }  // cells 0, 1 are the top row (GL origin is bottom left)

        // --- program construction ------------------------------------------------------------------------------
        template <std::size_t N>
        mgl::ProgramId link_stages(const demo::Env& env, std::initializer_list<std::pair<mgl::ShaderType, const char*>> stages,
                                   const std::array<const GLchar*, N>& tf_varyings)
        {
            std::vector<mgl::ShaderId> shaders;
            for (const auto& [type, file] : stages)
            {
                std::string log;
                const mgl::ShaderId s = demo::compile_shader(type, env.load_text(std::string("shaders/") + file), log);
                if (s.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", file, log));
                shaders.push_back(s);
            }
            const mgl::ProgramId program = mgl::glCreateProgram();
            for (const mgl::ShaderId s : shaders) mgl::glAttachShader(program, s);
            if constexpr (N > 0)
                mgl::glTransformFeedbackVaryings(program, static_cast<GLsizei>(N), tf_varyings.data(), mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(program);
            GLint status = GL_FALSE;
            mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &status);
            const std::string log = demo::program_info_log(program);
            for (const mgl::ShaderId s : shaders)
            {
                mgl::glDetachShader(program, s);
                mgl::glDeleteShader(s);
            }
            if (status != GL_TRUE) throw demo::Failure("program failed to link:\n" + log);
            return program;
        }

        // Compiles and links the given sources (type, text). Used by validate() for the programs that are SUPPOSED to
        // fail or to be small. Everything it creates is deleted again.
        LinkOutcome try_link(std::initializer_list<std::pair<mgl::ShaderType, std::string>> sources, mgl::ProgramId* keep = nullptr,
                             std::span<const GLchar* const> tf_varyings = {})
        {
            LinkOutcome out;
            std::vector<mgl::ShaderId> shaders;
            const mgl::ProgramId program = mgl::glCreateProgram();
            for (const auto& [type, text] : sources)
            {
                const mgl::ShaderId s = mgl::glCreateShader(type);
                mgl::glShaderSource(s, text);
                mgl::glCompileShader(s);
                GLint ok = GL_FALSE;
                mgl::glGetShaderiv(s, mgl::ShaderParameter::CompileStatus, &ok);
                if (ok != GL_TRUE)
                {
                    out.compiled = false;
                    out.log += "compile: " + demo::shader_info_log(s);
                }
                shaders.push_back(s);
                mgl::glAttachShader(program, s);
            }
            if (!tf_varyings.empty())
                mgl::glTransformFeedbackVaryings(program, static_cast<GLsizei>(tf_varyings.size()), tf_varyings.data(), mgl::TransformFeedbackBufferMode::InterleavedAttribs);
            mgl::glLinkProgram(program);
            GLint linked = GL_FALSE;
            mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &linked);
            out.linked = linked == GL_TRUE;
            out.log += "link: " + demo::program_info_log(program);
            for (const mgl::ShaderId s : shaders)
            {
                mgl::glDetachShader(program, s);
                mgl::glDeleteShader(s);
            }
            if (keep != nullptr && out.linked) *keep = program;
            else mgl::glDeleteProgram(program);
            return out;
        }

        template <typename T>
        static void upload(mgl::BufferTarget target, mgl::BufferId buffer, std::span<const T> data)
        {
            mgl::glBindBuffer(target, buffer);
            mgl::glBufferData(target, data, mgl::BufferUsage::StaticDraw);
        }
        static void attrib(unsigned location, GLint components, GLsizei stride, std::uintptr_t offset)
        {
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{location});
            mgl::glVertexAttribPointer(mgl::AttribLocation{location}, components, mgl::DataType::Float, GL_FALSE, stride, demo::buffer_offset(offset));
        }

        // --- drawing -------------------------------------------------------------------------------------------
        Vec3 light_eye(const Camera& cam) const
        {
            const Vec4 v = cam.view * Vec4{scene_.light.x, scene_.light.y, scene_.light.z, 0.0f};
            return {v.x, v.y, v.z};
        }

        // Each cell has a setup (program, uniforms, vertex array) and an issue (the draw call). Transform feedback needs the
        // split: UseProgram is an error while a capture is active (ES 3.2 section 12.2.2), the draw call is not.
        void setup_a(const Mat4& view, const Mat4& proj, Vec3 light_in_eye) const
        {
            mgl::glUseProgram(prog_a_);
            set_mat4(0, view);
            set_mat4(1, proj);
            set_vec3(2, light_in_eye);
            mgl::glBindVertexArray(vao_[kVaoA]);
        }
        void issue_a() const { mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, static_cast<GLsizei>(sprites_.size())); }
        void draw_a(const Mat4& view, const Mat4& proj, Vec3 light_in_eye) const
        {
            setup_a(view, proj, light_in_eye);
            issue_a();
        }

        void setup_b(const Camera& cam) const
        {
            mgl::glUseProgram(prog_b_);
            set_mat4(0, cam.viewproj);
            mgl::glUniform1f(loc(1), kCrystalLift);
            set_vec3(2, cam.eye);
            set_vec3(3, scene_.light);
            mgl::glBindVertexArray(vao_[kVaoB]);
        }
        void issue_b() const
        {
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, static_cast<GLsizei>(ico_.index.size()), mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
        }
        void draw_b(const Camera& cam) const
        {
            setup_b(cam);
            issue_b();
        }

        void draw_c_fill(const Camera& cam) const
        {
            mgl::glUseProgram(prog_c_fill_);
            set_mat4(0, cam.viewproj);
            set_vec3(2, cam.eye);
            set_vec3(3, scene_.light);
            mgl::glBindVertexArray(vao_[kVaoCFill]);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, static_cast<GLsizei>(torus_.index.size()), mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
        }
        void setup_c_silhouette(const Camera& cam) const
        {
            mgl::glUseProgram(prog_c_sil_);
            set_mat4(0, cam.viewproj);
            set_vec3(1, cam.eye);
            mgl::glUniform2f(loc(2), kLineHalfWidthPx * 2.0f / static_cast<float>(cell_w_), kLineHalfWidthPx * 2.0f / static_cast<float>(cell_h_));
            mgl::glUniform1f(loc(3), kLineBias);
            mgl::glBindVertexArray(vao_[kVaoCSil]);
        }
        void issue_c_silhouette() const
        {
            mgl::glDrawElements(mgl::PrimitiveType::TrianglesAdjacency, static_cast<GLsizei>(torus_adj_.size()), mgl::IndexType::UnsignedShort,
                                demo::buffer_offset(0));
        }
        void draw_c(const Camera& cam) const
        {
            draw_c_fill(cam);
            setup_c_silhouette(cam);
            issue_c_silhouette();
        }

        void setup_d(const Camera& cam, const std::array<Vec3, 6>& place, const std::array<cm::Vec2, 6>& spin) const
        {
            mgl::glUseProgram(prog_d_);
            set_mat4(0, cam.viewproj);
            std::array<float, 24> p{};
            std::array<float, 12> sp{};
            for (std::size_t k = 0; k < 6; ++k)
            {
                p[k * 4 + 0] = place[k].x;
                p[k * 4 + 1] = place[k].y;
                p[k * 4 + 2] = place[k].z;
                sp[k * 2 + 0] = spin[k].x;
                sp[k * 2 + 1] = spin[k].y;
            }
            mgl::glUniform4fv(loc(1), 6, p.data());
            mgl::glUniform2fv(loc(7), 6, sp.data());
            set_vec3(15, scene_.light);
            mgl::glBindVertexArray(vao_[kVaoD]);
        }
        void issue_d() const
        {
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, static_cast<GLsizei>(cube_.index.size()), mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
        }
        void draw_d(const Camera& cam, const std::array<Vec3, 6>& place, const std::array<cm::Vec2, 6>& spin) const
        {
            setup_d(cam, place, spin);
            issue_d();
        }

        // =====================================================================================================
        // validate(): limits
        // =====================================================================================================
        void check_limits(demo::Check& check) const
        {
            struct Limit
            {
                mgl::GetParameter pname;
                const char* name;
                int minimum;
            };
            // ES 3.2 Table 21.47 (Implementation Dependent Geometry Shader Limits)
            const Limit limits[] = {
                {mgl::GetParameter::MaxGeometryUniformComponents, "MAX_GEOMETRY_UNIFORM_COMPONENTS", 1024},
                {mgl::GetParameter::MaxGeometryUniformBlocks, "MAX_GEOMETRY_UNIFORM_BLOCKS", 12},
                {mgl::GetParameter::MaxGeometryInputComponents, "MAX_GEOMETRY_INPUT_COMPONENTS", 64},
                {mgl::GetParameter::MaxGeometryOutputComponents, "MAX_GEOMETRY_OUTPUT_COMPONENTS", 64},
                {mgl::GetParameter::MaxGeometryOutputVertices, "MAX_GEOMETRY_OUTPUT_VERTICES", 256},
                {mgl::GetParameter::MaxGeometryTotalOutputComponents, "MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS", 1024},
                {mgl::GetParameter::MaxGeometryTextureImageUnits, "MAX_GEOMETRY_TEXTURE_IMAGE_UNITS", 16},
                {mgl::GetParameter::MaxGeometryShaderInvocations, "MAX_GEOMETRY_SHADER_INVOCATIONS", 32},
                {mgl::GetParameter::MaxGeometryAtomicCounterBuffers, "MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS", 0},
                {mgl::GetParameter::MaxGeometryAtomicCounters, "MAX_GEOMETRY_ATOMIC_COUNTERS", 0},
                {mgl::GetParameter::MaxGeometryShaderStorageBlocks, "MAX_GEOMETRY_SHADER_STORAGE_BLOCKS", 0},
                {mgl::GetParameter::MaxGeometryImageUniforms, "MAX_GEOMETRY_IMAGE_UNIFORMS", 0},
            };
            std::string summary;
            for (const Limit& l : limits)
            {
                const int v = demo::get_integer(l.pname);
                check.expect(v >= l.minimum, std::format("{} >= {} (ES 3.2 Table 21.47)", l.name, l.minimum), std::format("value {}", v));
                summary += std::format(" {}={}", l.name + 4, v);
            }
            std::printf("   geometry shader limits:%s\n", summary.c_str());
            check.expect_no_gl_error("the limit queries raised no error");
            // Relations the specification states: the shaders of this demo stay within all limits.
            check.expect(15 * 10 <= demo::get_integer(mgl::GetParameter::MaxGeometryTotalOutputComponents),
                         "the crystal shader (15 vertices x 10 components) fits MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS");
            const int provoking = demo::get_integer(mgl::GetParameter::LayerProvokingVertex);
            check.expect(provoking == GL_FIRST_VERTEX_CONVENTION || provoking == GL_LAST_VERTEX_CONVENTION || provoking == GL_UNDEFINED_VERTEX,
                         "LAYER_PROVOKING_VERTEX is FIRST_VERTEX_CONVENTION, LAST_VERTEX_CONVENTION or UNDEFINED_VERTEX (section 11.3.4.5)",
                         std::format("value 0x{:X}", provoking));
        }

        // =====================================================================================================
        // validate(): program queries (ES 3.2 section 7.12)
        // =====================================================================================================
        static GLint program_value(mgl::ProgramId p, mgl::ProgramParameter pname)
        {
            GLint v = -12345;
            mgl::glGetProgramiv(p, pname, &v);
            return v;
        }

        void check_program_queries(demo::Check& check)
        {
            struct Expect
            {
                const char* name;
                mgl::ProgramId program;
                GLenum in_type, out_type;
                GLint vertices_out, invocations;
            };
            const Expect programs[] = {
                {"billboard (A)", prog_a_, GL_POINTS, GL_TRIANGLE_STRIP, 4, 1},
                {"crystal (B)", prog_b_, GL_TRIANGLES, GL_TRIANGLE_STRIP, 15, 1},
                {"silhouette (C)", prog_c_sil_, GL_TRIANGLES_ADJACENCY, GL_TRIANGLE_STRIP, 12, 1},
                {"ring (D)", prog_d_, GL_TRIANGLES, GL_TRIANGLE_STRIP, 3, 6},
            };
            for (const Expect& e : programs)
            {
                const GLint in = program_value(e.program, mgl::ProgramParameter::GeometryInputType);
                const GLint out = program_value(e.program, mgl::ProgramParameter::GeometryOutputType);
                const GLint vertices = program_value(e.program, mgl::ProgramParameter::GeometryVerticesOut);
                const GLint inv = program_value(e.program, mgl::ProgramParameter::GeometryShaderInvocations);
                check.expect(static_cast<GLenum>(in) == e.in_type, std::format("{}: GEOMETRY_INPUT_TYPE", e.name),
                             std::format("0x{:X} expected 0x{:X}", in, e.in_type));
                check.expect(static_cast<GLenum>(out) == e.out_type, std::format("{}: GEOMETRY_OUTPUT_TYPE", e.name),
                             std::format("0x{:X} expected 0x{:X}", out, e.out_type));
                check.expect_eq(vertices, e.vertices_out, std::format("{}: GEOMETRY_VERTICES_OUT equals the max_vertices of the source", e.name));
                check.expect_eq(inv, e.invocations, std::format("{}: GEOMETRY_SHADER_INVOCATIONS equals the invocations of the source (1 when omitted)", e.name));
            }
            check.expect_no_gl_error("the four programs answered all geometry queries without an error");

            // The error rule of section 7.12: queried for a program without a geometry shader -> INVALID_OPERATION.
            for (const mgl::ProgramParameter pname :
                 {mgl::ProgramParameter::GeometryVerticesOut, mgl::ProgramParameter::GeometryInputType, mgl::ProgramParameter::GeometryOutputType,
                  mgl::ProgramParameter::GeometryShaderInvocations})
            {
                GLint v = 777;
                demo::drain_gl_errors();
                mgl::glGetProgramiv(prog_c_fill_, pname, &v);
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                      std::format("glGetProgramiv(0x{:X}) on a program without a geometry shader -> INVALID_OPERATION",
                                                  static_cast<GLenum>(pname)));
            }

            // One program per input primitive type (ES 3.2 section 11.3.1): the reported types follow the layouts.
            for (const TinyKind& k : kTinyKinds)
            {
                mgl::ProgramId p{};
                const LinkOutcome o = try_link(
                    {{mgl::ShaderType::Vertex, std::string(kTinyVs)}, {mgl::ShaderType::Geometry, tiny_gs(k)}, {mgl::ShaderType::Fragment, std::string(kTinyFs)}}, &p);
                check.expect(o.linked, std::format("tiny geometry program '{}' links", k.name), o.log);
                if (!o.linked) continue;
                check.expect(static_cast<GLenum>(program_value(p, mgl::ProgramParameter::GeometryInputType)) == k.in_type,
                             std::format("layout({}) in -> GEOMETRY_INPUT_TYPE", k.in_layout));
                check.expect(static_cast<GLenum>(program_value(p, mgl::ProgramParameter::GeometryOutputType)) == k.out_type,
                             std::format("layout({}) out -> GEOMETRY_OUTPUT_TYPE", k.out_layout));
                check.expect_eq(program_value(p, mgl::ProgramParameter::GeometryVerticesOut), static_cast<GLint>(k.max_vertices),
                                std::format("'{}': GEOMETRY_VERTICES_OUT", k.name));
                mgl::glDeleteProgram(p);
            }
            check.expect_no_gl_error("tiny programs: no GL error");
        }

        // =====================================================================================================
        // validate(): pixels of the four cells against the CPU model
        // =====================================================================================================
        void check_pixels(demo::Env& env, demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            const int cw = cell_w_, ch = cell_h_;
            const float half = static_cast<float>(cw) * 0.5f;
            const Scene& s = scene_;
            std::vector<std::uint8_t> gpu;
            auto read_cell = [&](int c) { return demo::read_rgba8(cell_x(c), cell_y(c), cw, ch); };
            auto report = [&](const char* name, const CompareStats& st, int min_covered) {
                std::printf("   %s: %d pixels compared (%d covered), %d ambiguous pixels skipped, max deviation %d/255, %d mismatches\n", name, st.compared,
                            st.covered, st.skipped, st.max_diff, st.mismatches);
                for (const std::string& line : st.samples) std::printf("      mismatch %s\n", line.c_str());
                check.expect(st.mismatches == 0, std::format("{}: every unambiguous pixel equals the CPU model (+-{}/255)", name, kPixelTolerance),
                             st.mismatches == 0 ? std::string{} : std::format("{} mismatches, first at ({},{})", st.mismatches, st.first_x, st.first_y));
                check.expect(st.covered >= min_covered && st.compared > cw * ch / 2,
                             std::format("{}: the comparison is not vacuous ({} covered pixels compared, >= {} needed)", name, st.covered, min_covered));
            };
            (void)env;

            // ---- A: point sprites ------------------------------------------------------------------------
            {
                const Camera& cam = s.cam[kCellA];
                const Vec3 le = light_eye(cam);
                const auto strips = cm::emulate_billboards(sprites_, cam.view, cam.proj, half);
                cm::Raster raster(cw, ch);
                bool inside = true;
                for (const cm::Tri& t : cm::triangles(strips)) inside = raster.add(t) && inside;
                check.expect(inside, "cell A: every emitted vertex is in front of the camera (the CPU model does not clip)");
                gpu = read_cell(kCellA);
                auto shade = [&](int palette_shift) {
                    return [&, palette_shift](const cm::Frag& f) -> std::optional<cm::Rgb> {
                        const cm::Vec2 uv = f.lerp(&cm::EVert::uv);
                        const float r2 = uv.x * uv.x + uv.y * uv.y;
                        const float limit = 1.0f + f.margin / f.tri->v[0].aux;  // the discard edge moves with the margin
                        if (r2 > limit * limit) return std::nullopt;
                        const Vec3 n = {uv.x, uv.y, std::sqrt(std::max(1.0f - r2, 0.0f))};
                        const float diff = std::max(cm::dot(n, le), 0.0f);
                        return cm::kPalette[static_cast<std::size_t>((f.tri->v[0].prim + palette_shift) % 6)] * (0.25f + 0.75f * diff);
                    };
                };
                const CompareStats st = compare_cell(gpu, cw, ch, raster, kClear8[kCellA], shade(0));
                report("cell A (points -> billboards)", st, 300);
                const CompareStats wrong = compare_cell(gpu, cw, ch, raster, kClear8[kCellA], shade(1));
                check.expect(wrong.mismatches > 100, "negative control A: shifting the CPU palette by one entry breaks the comparison",
                             std::format("{} mismatches", wrong.mismatches));
            }

            // ---- B: crystals --------------------------------------------------------------------------------
            {
                const Camera& cam = s.cam[kCellB];
                auto build = [&](float lift) {
                    cm::Raster raster(cw, ch);
                    bool inside = true;
                    for (const cm::Tri& t : cm::triangles(cm::emulate_crystals(ico_, cam.viewproj, lift))) inside = raster.add(t) && inside;
                    return std::pair{std::move(raster), inside};
                };
                auto shade = [&](const cm::Frag& f) -> std::optional<cm::Rgb> {
                    return cm::blinn_phong(cm::kPalette[static_cast<std::size_t>(f.tri->v[0].prim % 6)], f.tri->v[0].normal, f.lerp(&cm::EVert::world),
                                           cam.eye, s.light);
                };
                const auto [raster, inside] = build(kCrystalLift);
                check.expect(inside, "cell B: every emitted vertex is in front of the camera");
                gpu = read_cell(kCellB);
                const CompareStats st = compare_cell(gpu, cw, ch, raster, kClear8[kCellB], shade);
                report("cell B (triangles -> prisms)", st, 2000);
                const auto [wrong_raster, wrong_inside] = build(kCrystalLift * 1.25f);
                (void)wrong_inside;
                const CompareStats wrong = compare_cell(gpu, cw, ch, wrong_raster, kClear8[kCellB], shade);
                check.expect(wrong.mismatches > 100, "negative control B: a 25% taller prism in the CPU model breaks the comparison",
                             std::format("{} mismatches", wrong.mismatches));
            }

            // ---- C: torus + silhouette ------------------------------------------------------------------------
            {
                const Camera& cam = s.cam[kCellC];
                const cm::Vec2 hw = {kLineHalfWidthPx * 2.0f / static_cast<float>(cw), kLineHalfWidthPx * 2.0f / static_cast<float>(ch)};
                auto build = [&](cm::Vec2 halfwidth, float* min_facing, std::size_t* strips_out) {
                    cm::Raster raster(cw, ch);
                    bool inside = true;
                    // fill pass: the torus as an ordinary triangle list with per-vertex normals
                    for (std::size_t t = 0; t + 2 < torus_.index.size(); t += 3)
                    {
                        cm::Tri tri;
                        for (std::size_t i = 0; i < 3; ++i)
                        {
                            cm::EVert& v = tri.v[i];
                            v.world = torus_.position[torus_.index[t + i]];
                            v.normal = torus_.normal[torus_.index[t + i]];
                            v.clip = cm::mul(cam.viewproj, v.world);
                        }
                        inside = raster.add(tri) && inside;
                    }
                    cm::SilhouetteResult sil = cm::emulate_silhouette(torus_, torus_adj_, cam.viewproj, cam.eye, halfwidth, kLineBias);
                    for (cm::Strip& strip : sil.strips)
                        for (cm::EVert& v : strip) v.inst = 1;  // marks outline vertices for the shading lambda
                    for (const cm::Tri& t : cm::triangles(sil.strips)) inside = raster.add(t) && inside;
                    if (min_facing != nullptr) *min_facing = sil.min_facing;
                    if (strips_out != nullptr) *strips_out = sil.strips.size();
                    return std::pair{std::move(raster), inside};
                };
                auto shade = [&](const cm::Frag& f) -> std::optional<cm::Rgb> {
                    if (f.tri->v[0].inst == 1) return cm::Rgb{1.0f, 0.82f, 0.20f};
                    return cm::blinn_phong({0.10f, 0.62f, 0.72f}, f.lerp(&cm::EVert::normal), f.lerp(&cm::EVert::world), cam.eye, s.light);
                };
                float min_facing = 0;
                std::size_t edges = 0;
                const auto [raster, inside] = build(hw, &min_facing, &edges);
                check.expect(inside, "cell C: every emitted vertex is in front of the camera");
                check.expect(min_facing > 1e-4f, "cell C: no face is within 1e-4 of edge-on, so the CPU and GPU facing decisions cannot differ by rounding",
                             std::format("smallest |cos| between a face normal and the view direction = {}", min_facing));
                std::printf("   cell C: %zu silhouette edges of %zu triangles\n", edges, torus_.index.size() / 3);
                check.expect(edges >= 20 && edges <= 200, "cell C: a plausible number of silhouette edges for a torus (not zero, not every edge)",
                             std::format("{} edges", edges));
                gpu = read_cell(kCellC);
                const CompareStats st = compare_cell(gpu, cw, ch, raster, kClear8[kCellC], shade);
                report("cell C (adjacency silhouette)", st, 2000);
                // Outline pixels are part of the comparison: count the outline-coloured pixels the GPU produced.
                int outline_pixels = 0;
                for (std::size_t i = 0; i + 3 < gpu.size(); i += 4)
                    if (std::abs(gpu[i] - 255) <= 2 && std::abs(gpu[i + 1] - 209) <= 2 && std::abs(gpu[i + 2] - 51) <= 2) ++outline_pixels;
                check.expect(outline_pixels > 150, "cell C: the geometry shader drew a visible outline (outline-coloured pixels)", std::format("{} pixels", outline_pixels));
                // Negative control: a CPU model with a 3-pixel line instead of 1.25 must disagree with the GPU image.
                const auto [wrong_raster, wrong_inside] = build({hw.x * 2.4f, hw.y * 2.4f}, nullptr, nullptr);
                (void)wrong_inside;
                const CompareStats wrong = compare_cell(gpu, cw, ch, wrong_raster, kClear8[kCellC], shade);
                check.expect(wrong.mismatches > 100, "negative control C: a CPU outline 2.4 times thicker breaks the comparison",
                             std::format("{} mismatches", wrong.mismatches));
            }

            // ---- D: instanced copies ------------------------------------------------------------------------------
            {
                const Camera& cam = s.cam[kCellD];
                cm::Raster raster(cw, ch);
                bool inside = true;
                for (const cm::Tri& t : cm::triangles(cm::emulate_ring(cube_, cam.viewproj, s.place, s.spin))) inside = raster.add(t) && inside;
                check.expect(inside, "cell D: every emitted vertex is in front of the camera");
                auto shade = [&](int copies_shift) {
                    return [&, copies_shift](const cm::Frag& f) -> std::optional<cm::Rgb> {
                        const cm::EVert& v = f.tri->v[0];
                        const cm::Rgb albedo = cm::kPalette[static_cast<std::size_t>((v.inst + copies_shift) % 6)] * (0.5f + 0.25f * static_cast<float>(v.prim % 3));
                        const float diff = std::max(cm::dot(cm::normalize(v.normal), s.light), 0.0f);
                        return albedo * (0.25f + 0.75f * diff);
                    };
                };
                gpu = read_cell(kCellD);
                const CompareStats st = compare_cell(gpu, cw, ch, raster, kClear8[kCellD], shade(0));
                report("cell D (invocations = 6)", st, 1500);
                // gl_InvocationID is what picks the colour and the place: swapping it in the model must break the match.
                const CompareStats wrong = compare_cell(gpu, cw, ch, raster, kClear8[kCellD], shade(1));
                check.expect(wrong.mismatches > 100, "negative control D: shifting gl_InvocationID by one in the CPU colour breaks the comparison",
                             std::format("{} mismatches", wrong.mismatches));
            }
            check.expect_no_gl_error("pixel comparison: no GL error");
        }

        // =====================================================================================================
        // validate(): transform feedback of the geometry shader output
        // =====================================================================================================
        struct Captured
        {
            std::vector<std::uint32_t> words;
            std::vector<std::uint32_t> slack;  // the words after the expected ones (must stay untouched)
            GLuint generated = 0;
            GLuint written = 0;
            mgl::ErrorCode error = mgl::ErrorCode::NoError;
        };

        // Captures what the last vertex stage emits for `draw` into the TF buffer, with rasterization discarded.
        Captured capture(std::size_t expected_words, mgl::TransformFeedbackPrimitive mode, const std::function<void()>& setup,
                         const std::function<void()>& issue)
        {
            constexpr std::size_t kSlack = 32;
            constexpr std::uint32_t kSentinel = 0xA5A5A5A5u;
            std::vector<std::uint32_t> fill(expected_words + kSlack, kSentinel);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, tf_buffer_);
            mgl::glBufferData(mgl::BufferTarget::CopyWrite, std::span<const std::uint32_t>(fill), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});

            setup();  // program, uniforms and vertex array: UseProgram is not allowed while a capture is active
            demo::drain_gl_errors();
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            mgl::glBeginQuery(mgl::QueryTarget::PrimitivesGenerated, queries_[0]);
            mgl::glBeginQuery(mgl::QueryTarget::TransformFeedbackPrimitivesWritten, queries_[1]);
            mgl::glBeginTransformFeedback(mode);
            issue();
            mgl::glEndTransformFeedback();
            mgl::glEndQuery(mgl::QueryTarget::TransformFeedbackPrimitivesWritten);
            mgl::glEndQuery(mgl::QueryTarget::PrimitivesGenerated);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);

            Captured c;
            c.error = demo::drain_gl_errors();
            mgl::glGetQueryObjectuiv(queries_[0], mgl::QueryObjectParameter::Result, &c.generated);
            mgl::glGetQueryObjectuiv(queries_[1], mgl::QueryObjectParameter::Result, &c.written);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::TransformFeedback, 0, static_cast<GLsizeiptr>(fill.size() * sizeof(std::uint32_t)),
                                                       mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                std::vector<std::uint32_t> all(fill.size());
                std::memcpy(all.data(), mapped, all.size() * sizeof(std::uint32_t));
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::TransformFeedback);
                c.words.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(expected_words));
                c.slack.assign(all.begin() + static_cast<std::ptrdiff_t>(expected_words), all.end());
            }
            mgl::glBindBuffer(mgl::BufferTarget::TransformFeedback, mgl::BufferId{0});
            mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
            return c;
        }

        void tf_case(demo::Check& check, const char* name, const std::vector<cm::Strip>& strips, Layout layout, const std::function<void()>& setup,
                     const std::function<void()>& issue)
        {
            const std::vector<std::uint32_t> expected = pack(strips, layout);
            const std::size_t triangles = triangle_count(strips);
            const Captured c = capture(expected.size(), mgl::TransformFeedbackPrimitive::Triangles, setup, issue);
            check.expect(c.error == mgl::ErrorCode::NoError, std::format("{}: capture raised no GL error", name), std::string(demo::error_name(c.error)));
            check.expect_eq(c.generated, static_cast<GLuint>(triangles),
                            std::format("{}: PRIMITIVES_GENERATED counts the {} triangles the geometry shader emitted", name, triangles));
            check.expect_eq(c.written, static_cast<GLuint>(triangles),
                            std::format("{}: TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN equals PRIMITIVES_GENERATED (nothing overflowed)", name));
            const WordDiff d = compare_words(c.words, expected);
            check.expect(d.mismatches == 0 && !c.words.empty(),
                         std::format("{}: the {} captured words (the last vertex stage's output) equal the CPU model bit for bit", name, expected.size()),
                         std::format("{} mismatches, first at word {}", d.mismatches, d.first));
            const bool slack_untouched = !c.slack.empty() && std::all_of(c.slack.begin(), c.slack.end(), [](std::uint32_t w) { return w == 0xA5A5A5A5u; });
            check.expect(slack_untouched, std::format("{}: nothing was written beyond the expected {} words", name, expected.size()));
            // Negative control: one flipped bit in a copy of the model must be detected.
            std::vector<std::uint32_t> broken = expected;
            if (!broken.empty()) broken[broken.size() / 2] ^= 1u;
            check.expect(compare_words(c.words, broken).mismatches == 1, std::format("negative control {}: a single flipped bit in the model is detected", name));
        }

        void check_transform_feedback(demo::Check& check)
        {
            const float half = static_cast<float>(cell_w_) * 0.5f;
            const Scene& s = scene_;

            // A: the exact view is a pure translation by multiples of 1/16, so eye = position + t is exact.
            {
                const Mat4 view = demo::math::translate({0.5f, -0.25f, -3.0f});
                const Mat4 proj = s.cam[kCellA].proj;
                tf_case(check, "TF cell A (points -> strips)", cm::emulate_billboards(sprites_, view, proj, half), Layout::Billboard,
                        [&] { setup_a(view, proj, {0, 0, 1}); }, [&] { issue_a(); });
            }
            // B: world-space output only: exact for the quantised icosphere and a power-of-two lift.
            tf_case(check, "TF cell B (triangles -> prisms)", cm::emulate_crystals(ico_, s.cam[kCellB].viewproj, kCrystalLift), Layout::Crystal,
                    [&] { setup_b(s.cam[kCellB]); }, [&] { issue_b(); });
            // C: the captured values are copies of vertex positions; which edges appear is decided by u_eye.
            {
                const Camera& cam = s.cam[kCellC];
                const cm::Vec2 hw = {kLineHalfWidthPx * 2.0f / static_cast<float>(cell_w_), kLineHalfWidthPx * 2.0f / static_cast<float>(cell_h_)};
                const cm::SilhouetteResult sil = cm::emulate_silhouette(torus_, torus_adj_, cam.viewproj, cam.eye, hw, kLineBias);
                tf_case(check, "TF cell C (adjacency -> outline quads)", sil.strips, Layout::Silhouette, [&] { setup_c_silhouette(cam); },
                        [&] { issue_c_silhouette(); });
            }
            // D: spins of multiples of 90 degrees make the rotation exact; the six copies come out in invocation order.
            {
                const std::array<Vec3, 6> place = {{{1.5f, 0.0f, 0.0f}, {0.75f, 0.5f, 1.25f}, {-0.5f, -0.25f, 1.75f}, {-1.5f, 0.25f, 0.0f}, {-0.75f, 0.0f, -1.25f}, {0.5f, 0.5f, -1.75f}}};
                const std::array<cm::Vec2, 6> spin = {{{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 0}, {0, 1}}};
                tf_case(check, "TF cell D (invocations = 6)", cm::emulate_ring(cube_, s.cam[kCellD].viewproj, place, spin), Layout::Ring,
                        [&] { setup_d(s.cam[kCellD], place, spin); }, [&] { issue_d(); });
            }

            // The primitive mode rule of section 12.2.2: with a geometry shader the capture mode is compared with the
            // OUTPUT type of the shader, not with the mode of the draw call.
            auto expect_mode = [&](mgl::TransformFeedbackPrimitive mode, const char* mode_name, mgl::ErrorCode expected, const char* what) {
                mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, tf_buffer_);
                const std::vector<std::uint32_t> zeros(4096, 0u);
                mgl::glBufferData(mgl::BufferTarget::CopyWrite, std::span<const std::uint32_t>(zeros), mgl::BufferUsage::DynamicCopy);
                mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
                setup_a(demo::math::translate({0.0f, 0.0f, -3.0f}), s.cam[kCellA].proj, {0, 0, 1});
                demo::drain_gl_errors();
                mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, tf_buffer_);
                mgl::glEnable(mgl::Capability::RasterizerDiscard);
                mgl::glBeginTransformFeedback(mode);
                const mgl::ErrorCode begin = demo::drain_gl_errors();
                issue_a();
                check.expect_gl_error(expected, std::format("{}: BeginTransformFeedback({}) then a GL_POINTS draw of a geometry shader that outputs triangle_strip", what, mode_name));
                mgl::glEndTransformFeedback();
                mgl::glDisable(mgl::Capability::RasterizerDiscard);
                mgl::glBindBufferBase(mgl::BufferTarget::TransformFeedback, 0, mgl::BufferId{0});
                check.expect(begin == mgl::ErrorCode::NoError, std::format("{}: BeginTransformFeedback({}) itself is legal", what, mode_name));
                demo::drain_gl_errors();
            };
            expect_mode(mgl::TransformFeedbackPrimitive::Triangles, "TRIANGLES", mgl::ErrorCode::NoError,
                        "capture mode matches the shader output (the draw mode POINTS is not what is compared, section 12.2.2)");
            expect_mode(mgl::TransformFeedbackPrimitive::Points, "POINTS", mgl::ErrorCode::InvalidOperation, "capture mode POINTS does not match triangle_strip output");
            expect_mode(mgl::TransformFeedbackPrimitive::Lines, "LINES", mgl::ErrorCode::InvalidOperation, "capture mode LINES does not match triangle_strip output");
            check.expect_no_gl_error("transform feedback tests: no GL error left");
        }

        // =====================================================================================================
        // validate(): which vertices a geometry shader sees for the adjacency primitive types (ES 3.2 sections 10.1.8-10.1.10)
        // =====================================================================================================
        void check_adjacency_vertex_order(demo::Check& check)
        {
            // The vertex shader forwards gl_VertexID; the geometry shader writes, for every vertex gl_in[k] of its input
            // primitive, one point carrying (that vertex's gl_VertexID, gl_PrimitiveIDIn). Transform feedback records them.
            const std::string vs = "#version 320 es\nlayout(location = 0) in vec2 a_pos;\nflat out int vid;\n"
                                   "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); vid = gl_VertexID; }\n";
            auto gs = [](const char* layout, int n) {
                return std::format("#version 320 es\nlayout({}) in;\nlayout(points, max_vertices = {}) out;\nflat in int vid[];\n"
                                   "flat out int o_seen;\nflat out int o_prim;\n"
                                   "void main() {{ for (int k = 0; k < {}; ++k) {{ o_seen = vid[k]; o_prim = gl_PrimitiveIDIn; "
                                   "gl_Position = gl_in[k].gl_Position; EmitVertex(); EndPrimitive(); }} }}\n",
                                   layout, n, n);
            };
            const std::string fs = "#version 320 es\nprecision highp float;\nout vec4 c;\nvoid main() { c = vec4(1.0); }\n";
            constexpr std::array<const GLchar*, 2> varyings = {"o_seen", "o_prim"};

            std::array<float, 24> positions{};
            for (std::size_t i = 0; i < 12; ++i)
            {
                positions[i * 2] = static_cast<float>(i) / 12.0f;
                positions[i * 2 + 1] = 0.5f;
            }
            mgl::BufferId vbo{};
            mgl::VertexArrayId vao{};
            mgl::glGenBuffers(1, &vbo);
            mgl::glGenVertexArrays(1, &vao);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(positions), mgl::BufferUsage::StaticDraw);
            attrib(0, 2, 8, 0);

            struct Case
            {
                const char* name;
                const char* layout;
                int n;                  // vertices per geometry shader invocation
                mgl::PrimitiveType mode;
                int vertices;           // draw count
                int primitives;         // expected number of input primitives
                int (*first)(int);      // gl_VertexID of gl_in[0] of primitive p
            };
            const Case cases[] = {
                {"LINES_ADJACENCY, 8 vertices: segments (4p .. 4p+3), the 2nd and 3rd are the line (10.1.8)", "lines_adjacency", 4, mgl::PrimitiveType::LinesAdjacency, 8, 2,
                 [](int p) { return 4 * p; }},
                {"LINE_STRIP_ADJACENCY, 7 vertices: n + 3 vertices give n = 4 segments (p .. p+3) (10.1.9)", "lines_adjacency", 4, mgl::PrimitiveType::LineStripAdjacency, 7, 4,
                 [](int p) { return p; }},
                {"LINES_ADJACENCY, 7 vertices: the trailing 3 vertices are ignored (10.1.8)", "lines_adjacency", 4, mgl::PrimitiveType::LinesAdjacency, 7, 1, [](int p) { return 4 * p; }},
                {"LINE_STRIP_ADJACENCY, 3 vertices: fewer than four vertices draw nothing (10.1.9)", "lines_adjacency", 4, mgl::PrimitiveType::LineStripAdjacency, 3, 0,
                 [](int p) { return p; }},
                {"TRIANGLES_ADJACENCY, 12 vertices: 6 vertices per triangle in order, gl_in[0], gl_in[2], gl_in[4] are the triangle (10.1.10)", "triangles_adjacency", 6,
                 mgl::PrimitiveType::TrianglesAdjacency, 12, 2, [](int p) { return 6 * p; }},
            };
            for (const Case& c : cases)
            {
                mgl::ProgramId program{};
                const LinkOutcome o = try_link({{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs(c.layout, c.n)}, {mgl::ShaderType::Fragment, fs}}, &program, varyings);
                check.expect(o.linked, std::format("adjacency order program ({}) links", c.layout), o.log);
                if (!o.linked) continue;
                std::vector<std::uint32_t> expected;
                for (int p = 0; p < c.primitives; ++p)
                    for (int k = 0; k < c.n; ++k)
                    {
                        expected.push_back(bits_of(c.first(p) + k));  // gl_VertexID of gl_in[k]
                        expected.push_back(bits_of(p));               // gl_PrimitiveIDIn counts primitives, not adjacency vertices
                    }
                const Captured cap = capture(expected.size(), mgl::TransformFeedbackPrimitive::Points, [&] { mgl::glUseProgram(program); },
                                             [&] { mgl::glDrawArrays(c.mode, 0, c.vertices); });
                const WordDiff d = compare_words(cap.words, expected);
                check.expect(cap.error == mgl::ErrorCode::NoError && cap.written == static_cast<GLuint>(c.primitives * c.n) && cap.generated == cap.written &&
                                 (expected.empty() || d.mismatches == 0),
                             c.name,
                             std::format("error {}, {} points written, {} generated, {} word mismatches, expected {} points", demo::error_name(cap.error), cap.written,
                                         cap.generated, d.mismatches, c.primitives * c.n));
                mgl::glUseProgram(mgl::ProgramId{0});
                mgl::glDeleteProgram(program);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteVertexArrays(1, &vao);
            mgl::glDeleteBuffers(1, &vbo);
            check.expect_no_gl_error("adjacency vertex order: no GL error left");
        }

        // =====================================================================================================
        // validate(): draw mode vs. geometry shader input type (ES 3.2 section 11.3.4.6)
        // =====================================================================================================
        void check_draw_mode_matrix(demo::Check& check)
        {
            const std::array<float, 12> positions = {-1, -1, 3, -1, -1, 3, -1, -1, 3, -1, -1, 3};  // 6 vertices; the first three cover the viewport
            mgl::BufferId vbo{};
            mgl::VertexArrayId vao{};
            mgl::glGenBuffers(1, &vbo);
            mgl::glGenVertexArrays(1, &vao);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(positions), mgl::BufferUsage::StaticDraw);
            attrib(0, 2, 8, 0);

            std::array<mgl::ProgramId, kTinyKinds.size()> progs{};
            for (std::size_t i = 0; i < kTinyKinds.size(); ++i)
            {
                const LinkOutcome o = try_link({{mgl::ShaderType::Vertex, std::string(kTinyVs)}, {mgl::ShaderType::Geometry, tiny_gs(kTinyKinds[i])},
                                                {mgl::ShaderType::Fragment, std::string(kTinyFs)}}, &progs[i]);
                check.expect(o.linked, std::format("matrix program '{}' links", kTinyKinds[i].name), o.log);
            }

            // 5 input types x 11 draw modes, rasterization discarded (the error is raised regardless).
            mgl::glEnable(mgl::Capability::RasterizerDiscard);
            int combos = 0, wrong = 0;
            std::string first_wrong;
            for (std::size_t i = 0; i < kTinyKinds.size(); ++i)
            {
                if (progs[i].value == 0) continue;
                mgl::glUseProgram(progs[i]);
                for (const DrawMode& m : kDrawModes)
                {
                    demo::drain_gl_errors();
                    mgl::glDrawArrays(m.mode, 0, 6);
                    const mgl::ErrorCode e = demo::drain_gl_errors();
                    const bool compatible = static_cast<std::size_t>(m.compatible_with) == i;
                    const mgl::ErrorCode want = compatible ? mgl::ErrorCode::NoError : mgl::ErrorCode::InvalidOperation;
                    ++combos;
                    if (e != want)
                    {
                        ++wrong;
                        if (first_wrong.empty())
                            first_wrong = std::format("layout({}) in with {} gave {} (expected {})", kTinyKinds[i].in_layout, m.name, demo::error_name(e), demo::error_name(want));
                    }
                }
            }
            check.expect(combos == 55 && wrong == 0,
                         "ES 3.2 section 11.3.4.6: of the 55 (input type, draw mode) pairs exactly the compatible 11 draw without an error, all others raise INVALID_OPERATION",
                         first_wrong.empty() ? std::format("{} pairs", combos) : first_wrong);
            mgl::glDisable(mgl::Capability::RasterizerDiscard);

            // "No fragments will be rendered": draw a viewport-filling triangle into a 16x16 target.
            mgl::RenderbufferId rb{};
            mgl::FramebufferId fbo{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, 16, 16);
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rb);
            check.expect(mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete, "16x16 test framebuffer is complete");
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glViewport(0, 0, 16, 16);
            if (progs[3].value != 0)
            {
                mgl::glUseProgram(progs[3]);  // layout(triangles) in
                mgl::glUniform4f(loc(0), 1.0f, 0.0f, 0.0f, 1.0f);
                auto draw_and_read = [&](mgl::PrimitiveType mode) {
                    mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    mgl::glClear(mgl::ClearBufferBit::Color);
                    demo::drain_gl_errors();
                    mgl::glDrawArrays(mode, 0, 3);
                    const mgl::ErrorCode e = demo::drain_gl_errors();
                    return std::pair{e, demo::read_pixel(8, 8)};
                };
                const auto [e_ok, px_ok] = draw_and_read(mgl::PrimitiveType::Triangles);
                check.expect(e_ok == mgl::ErrorCode::NoError && px_ok == demo::Rgba8{255, 0, 0, 255}, "matching draw (TRIANGLES into layout(triangles) in) fills the target",
                             format_px(px_ok));
                const auto [e_bad, px_bad] = draw_and_read(mgl::PrimitiveType::Points);
                check.expect(e_bad == mgl::ErrorCode::InvalidOperation && px_bad == demo::Rgba8{0, 0, 0, 255},
                             "mismatching draw (POINTS into layout(triangles) in): INVALID_OPERATION and no fragment is rendered", format_px(px_bad));
                const auto [e_adj, px_adj] = draw_and_read(mgl::PrimitiveType::TrianglesAdjacency);
                check.expect(e_adj == mgl::ErrorCode::InvalidOperation && px_adj == demo::Rgba8{0, 0, 0, 255},
                             "mismatching draw (TRIANGLES_ADJACENCY into layout(triangles) in): INVALID_OPERATION and no fragment is rendered", format_px(px_adj));
                const auto [e_strip, px_strip] = draw_and_read(mgl::PrimitiveType::TriangleStrip);
                check.expect(e_strip == mgl::ErrorCode::NoError && px_strip == demo::Rgba8{255, 0, 0, 255}, "TRIANGLE_STRIP is accepted by layout(triangles) in",
                             format_px(px_strip));
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glDeleteRenderbuffers(1, &rb);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteVertexArrays(1, &vao);
            mgl::glDeleteBuffers(1, &vbo);
            for (const mgl::ProgramId p : progs)
                if (p.value != 0) mgl::glDeleteProgram(p);
            check.expect_no_gl_error("draw-mode matrix: no GL error left");
        }

        static std::string format_px(demo::Rgba8 p) { return demo::format_rgba(p); }

        // =====================================================================================================
        // validate(): which geometry-shader programs must NOT link (ES 3.2 sections 7.3, 7.4.1, 11.3.1-11.3.4)
        // =====================================================================================================
        void check_link_rules(demo::Env& env, demo::Check& check)
        {
            (void)env;
            const int max_vertices = demo::get_integer(mgl::GetParameter::MaxGeometryOutputVertices);
            const int max_invocations = demo::get_integer(mgl::GetParameter::MaxGeometryShaderInvocations);
            const int max_out_components = demo::get_integer(mgl::GetParameter::MaxGeometryOutputComponents);
            const int max_total = demo::get_integer(mgl::GetParameter::MaxGeometryTotalOutputComponents);
            const std::string vs(kTinyVs), fs(kTinyFs);

            auto gs = [](std::string_view in_layout, std::string_view out_layout, std::string_view globals, std::string_view body) {
                std::string s = "#version 320 es\n";
                if (!in_layout.empty()) s += std::format("layout({}) in;\n", in_layout);
                if (!out_layout.empty()) s += std::format("layout({}) out;\n", out_layout);
                s += std::string(globals) + "\nvoid main() {\n" + std::string(body) + "\n}\n";
                return s;
            };
            const std::string tri_body = "gl_Position = gl_in[0].gl_Position; EmitVertex(); gl_Position = gl_in[1].gl_Position; EmitVertex(); "
                                         "gl_Position = gl_in[2].gl_Position; EmitVertex();";
            auto expect_fail = [&](const char* what, std::initializer_list<std::pair<mgl::ShaderType, std::string>> sources) {
                const LinkOutcome o = try_link(sources);
                check.expect(!o.linked, std::format("{}: the program does not link", what), "linked successfully");
                std::printf("   note: '%s' failed at %s time\n", what, o.compiled ? "link" : "compile");
                return o;
            };
            auto expect_ok = [&](const char* what, std::initializer_list<std::pair<mgl::ShaderType, std::string>> sources) {
                const LinkOutcome o = try_link(sources);
                check.expect(o.linked, std::format("{}: the program links", what), o.log);
            };

            // The valid control all variants are derived from.
            expect_ok("control: layout(triangles) in + layout(triangle_strip, max_vertices = 3) out",
                      {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs("triangles", "triangle_strip, max_vertices = 3", "", tri_body)}, {mgl::ShaderType::Fragment, fs}});
            expect_ok("max_vertices = 256 (the minimum of MAX_GEOMETRY_OUTPUT_VERTICES, Table 21.47)",
                      {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs("triangles", "triangle_strip, max_vertices = 256", "", tri_body)}, {mgl::ShaderType::Fragment, fs}});
            expect_ok(std::format("invocations = {} (MAX_GEOMETRY_SHADER_INVOCATIONS)", max_invocations).c_str(),
                      {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs(std::format("triangles, invocations = {}", max_invocations), "triangle_strip, max_vertices = 3", "", tri_body)},
                       {mgl::ShaderType::Fragment, fs}});

            // ES 3.2 section 7.3 (LinkProgram): the input type, the output type and max_vertices must be specified.
            expect_fail("geometry shader without an input primitive layout (section 11.3.1)",
                        {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs("", "triangle_strip, max_vertices = 3", "", "EmitVertex();")}, {mgl::ShaderType::Fragment, fs}});
            expect_fail("geometry shader without an output primitive type (section 11.3.2)",
                        {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs("triangles", "max_vertices = 3", "", tri_body)}, {mgl::ShaderType::Fragment, fs}});
            expect_fail("geometry shader without max_vertices (section 11.3.2)",
                        {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs("triangles", "triangle_strip", "", tri_body)}, {mgl::ShaderType::Fragment, fs}});
            expect_fail("geometry + fragment shader without a vertex shader, not separable (section 7.3)",
                        {{mgl::ShaderType::Geometry, gs("triangles", "triangle_strip, max_vertices = 3", "", tri_body)}, {mgl::ShaderType::Fragment, fs}});

            // GLSL ES 3.20 section 4.4.1/4.4.2: limits of the layout qualifiers.
            expect_fail(std::format("max_vertices = {} (MAX_GEOMETRY_OUTPUT_VERTICES + 1)", max_vertices + 1).c_str(),
                        {{mgl::ShaderType::Vertex, vs}, {mgl::ShaderType::Geometry, gs("triangles", std::format("triangle_strip, max_vertices = {}", max_vertices + 1), "", tri_body)},
                         {mgl::ShaderType::Fragment, fs}});
            {
                // "If a shader specifies an invocation count greater than the implementation-dependent maximum, or
                // less than or equal to zero, a compile-time error results."
                for (const int bad : {max_invocations + 1, 0})
                {
                    const LinkOutcome o = try_link({{mgl::ShaderType::Vertex, vs},
                                                    {mgl::ShaderType::Geometry, gs(std::format("triangles, invocations = {}", bad), "triangle_strip, max_vertices = 3", "", tri_body)},
                                                    {mgl::ShaderType::Fragment, fs}});
                    check.expect(!o.compiled && !o.linked, std::format("invocations = {} is a compile-time error (GLSL ES 3.20 section 4.4.1)", bad), o.log);
                }
            }
            // ES 3.2 section 11.3.1: a shader that accesses more input vertices than the input type provides does not link.
            expect_fail("gl_in[3] with layout(triangles) in (only 3 vertices exist)",
                        {{mgl::ShaderType::Vertex, vs},
                         {mgl::ShaderType::Geometry, gs("triangles", "triangle_strip, max_vertices = 3", "", "gl_Position = gl_in[3].gl_Position; EmitVertex();")},
                         {mgl::ShaderType::Fragment, fs}});

            // Interface blocks between the stages (ES 3.2 sections 7.4.1, 11.3.4.3; GLSL ES 3.20 section 4.3.9).
            const std::string vs_block = "#version 320 es\nlayout(location = 0) in vec2 a_pos;\nout Blk { vec3 c; } v_out;\nvoid main() { v_out.c = vec3(a_pos, 0.5); gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
            // Reading `.xyz` works for both the vec3 and the vec4 flavour of the member, so a mismatch is the only thing wrong.
            const std::string gs_body3 = "gl_Position = gl_in[0].gl_Position; g_out.c = g_in[0].c.xyz; EmitVertex(); gl_Position = gl_in[1].gl_Position; g_out.c = g_in[1].c.xyz; EmitVertex(); "
                                         "gl_Position = gl_in[2].gl_Position; g_out.c = g_in[2].c.xyz; EmitVertex();";
            const std::string gs_body2 = "gl_Position = gl_in[0].gl_Position; g_out.c = g_in[0].c.xyz; EmitVertex(); gl_Position = gl_in[1].gl_Position; g_out.c = g_in[1].c.xyz; EmitVertex();";
            const std::string fs_block = "#version 320 es\nprecision highp float;\nin Out { vec3 c; } f_in;\nout vec4 frag_color;\nvoid main() { frag_color = vec4(f_in.c, 1.0); }\n";
            auto gs_with_block = [&](std::string_view in_block, std::string_view in_instance, const std::string& body) {
                return gs("triangles", "triangle_strip, max_vertices = 3", std::format("in {} {};\nout Out {{ vec3 c; }} g_out;", in_block, in_instance), body);
            };
            expect_ok("blocks: VS out Blk {vec3 c} v_out -> GS in Blk {vec3 c} g_in[] -> FS in Out {vec3 c} f_in (instance names differ, block names match)",
                      {{mgl::ShaderType::Vertex, vs_block}, {mgl::ShaderType::Geometry, gs_with_block("Blk { vec3 c; }", "g_in[]", gs_body3)}, {mgl::ShaderType::Fragment, fs_block}});
            expect_fail("blocks: the GS declares Blk { vec4 c; } but the VS writes Blk { vec3 c; } (member type mismatch)",
                        {{mgl::ShaderType::Vertex, vs_block}, {mgl::ShaderType::Geometry, gs_with_block("Blk { vec4 c; }", "g_in[]", gs_body3)}, {mgl::ShaderType::Fragment, fs_block}});
            expect_fail("blocks: the GS reads Other { vec3 c; } that no vertex shader output matches (no matching output, section 7.4.1)",
                        {{mgl::ShaderType::Vertex, vs_block}, {mgl::ShaderType::Geometry, gs_with_block("Other { vec3 c; }", "g_in[]", gs_body3)}, {mgl::ShaderType::Fragment, fs_block}});
            expect_fail("blocks: the GS input block array has 2 elements for layout(triangles) in (needs 3, section 11.3.4.3)",
                        {{mgl::ShaderType::Vertex, vs_block}, {mgl::ShaderType::Geometry, gs_with_block("Blk { vec3 c; }", "g_in[2]", gs_body2)}, {mgl::ShaderType::Fragment, fs_block}});

            // ES 3.2 section 11.3.4.4: LinkProgram may fail when vertices x components exceed MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS (observation).
            {
                // K vec4 outputs that the fragment shader reads (so they are active) plus gl_Position. K stays within the
                // geometry output and the fragment input limits so that those are not the reason for a failure.
                const int max_fragment_in = demo::get_integer(mgl::GetParameter::MaxFragmentInputComponents);
                const int k = std::max(1, std::min(max_out_components, max_fragment_in) / 4 - 1);
                const long long per_vertex = 4LL * k + 4;
                if (static_cast<long long>(max_vertices) * per_vertex > max_total && per_vertex <= max_out_components + 4)
                {
                    std::string out_decl = std::format("out vec4 o[{}];", k);
                    std::string in_decl = std::format("in vec4 o[{}];", k);
                    std::string body = "for (int i = 0; i < 3; ++i) { gl_Position = gl_in[i].gl_Position; for (int j = 0; j < " + std::to_string(k) +
                                       "; ++j) o[j] = vec4(float(j + i)); EmitVertex(); }";
                    const std::string total_fs = "#version 320 es\nprecision highp float;\n" + in_decl +
                                                 "\nout vec4 frag_color;\nvoid main() { vec4 s = vec4(0.0); for (int j = 0; j < " + std::to_string(k) + "; ++j) s += o[j]; frag_color = s; }\n";
                    const LinkOutcome o = try_link({{mgl::ShaderType::Vertex, vs},
                                                    {mgl::ShaderType::Geometry, gs("triangles", std::format("triangle_strip, max_vertices = {}", max_vertices), out_decl, body)},
                                                    {mgl::ShaderType::Fragment, total_fs}});
                    // ES 3.2 section 11.3.4.4: "LinkProgram will fail if it determines that the total component limit would be
                    // violated." The failure is conditional on the linker determining the violation, so a successful link is
                    // conformant too: the outcome is printed, not checked. (The program is never used for drawing.)
                    std::printf("   note: max_vertices = %d x %lld output components = %lld > MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS = %d: "
                                "LinkProgram %s (section 11.3.4.4 requires a failure only \"if it determines\" the violation; either outcome conforms)\n",
                                max_vertices, per_vertex, static_cast<long long>(max_vertices) * per_vertex, max_total,
                                o.linked ? "succeeded" : "failed");
                }
                else
                    std::printf("   note: this implementation's limits make a total-output-components violation unreachable (max_vertices %d, per-vertex %lld, total %d)\n",
                                max_vertices, per_vertex, max_total);
            }
            check.expect_no_gl_error("link rules: no GL error left");
        }

        // --- state ---------------------------------------------------------------------------------------------
        mgl::ProgramId prog_a_{}, prog_b_{}, prog_c_fill_{}, prog_c_sil_{}, prog_d_{};
        std::array<mgl::BufferId, kBufferCount> buf_{};
        std::array<mgl::VertexArrayId, kVaoCount> vao_{};
        mgl::BufferId tf_buffer_{};
        std::array<mgl::QueryId, 2> queries_{};

        std::vector<cm::Sprite> sprites_;
        cm::Mesh ico_, torus_, cube_;
        std::vector<std::uint16_t> torus_adj_;

        float time_ = kTime0;
        int cell_w_ = 128, cell_h_ = 128;
        Scene scene_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-geometry-shaders",
        .title = "Geometry shaders: amplification, adjacency, invocations",
        .min_es = {3, 2},
        .summary = "Four 3D scenes drawn by geometry shaders: points to billboards, triangles to prisms (EndPrimitive), "
                   "TRIANGLES_ADJACENCY silhouette outlines and layout(invocations = 6) copies. Pixels, transform feedback "
                   "of the last vertex stage, program queries, limits and the draw/link error rules are checked.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<GeometryShaders>(argc, argv, info);
}
