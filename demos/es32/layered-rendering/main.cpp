// es32-layered-rendering
//
// Layered rendering (OpenGL ES 3.2 sections 9.2.8, 9.4, 9.8 and 11.3.4.5): glFramebufferTexture attaches a WHOLE level
// of a 2D array, 3D or cube map texture to a framebuffer (a "layered" attachment). A geometry shader then chooses the
// layer of every primitive it emits by writing gl_Layer, so one draw call renders different content into every
// layer. The 256x256 image is a 2x2 grid:
//
//   top left      a mirror ball in a room. The room (walls, decals, boxes) was rendered ONCE into all six faces of a
//                 cube map by a geometry shader with layout(invocations = 6) and gl_Layer = gl_InvocationID; the ball
//                 reflects it and the background is a skybox, both read from that cube map
//   top right     the six faces of the same cube map as a "cube net" (each tile looks up the face through a direction)
//   bottom left   a 2D array texture 64x64x4, four different images painted by four POINTS (layer = gl_PrimitiveIDIn)
//   bottom right  a 3D texture 64x64x4, four slices painted by ONE point with layout(invocations = 4)
//
// validate() reads every layer back (glFramebufferTextureLayer + glReadPixels) and compares it exactly with a CPU
// model, compares the displayed tiles and the reflection pixels with the model, checks the attachment queries and the
// completeness rules of layered framebuffers (INCOMPLETE_LAYER_TARGETS), the errors of glFramebufferTexture*, and what
// the specification defines for gl_Layer (default layer 0, clear touches all layers, ReadPixels reads layer 0, no
// effect on a non-layered framebuffer, the provoking vertex).

#include "cpu_model.hpp"

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
namespace lr = lrmodel;
using demo::math::Mat4;
using demo::math::Vec3;

namespace
{
    constexpr int kLayerSize = lr::kLayerSize;  // 64
    constexpr int kLayers = 4;                  // array layers and 3D slices
    constexpr int kCubeSize = 64;
    constexpr float kEnvNear = 0.5f, kEnvFar = 20.0f;
    constexpr float kTime0 = 0.9f;
    constexpr float kPreviewFovDegrees = 62.0f;

    mgl::UniformLocation loc(int n) { return mgl::UniformLocation{n}; }

    // Layer l of a layered cube map is the face TEXTURE_CUBE_MAP_POSITIVE_X + l (Table 8.24).
    constexpr std::array<mgl::TextureTarget, 6> kCubeFaceTargets = {
        mgl::TextureTarget::TextureCubeMapPositiveX, mgl::TextureTarget::TextureCubeMapNegativeX, mgl::TextureTarget::TextureCubeMapPositiveY,
        mgl::TextureTarget::TextureCubeMapNegativeY, mgl::TextureTarget::TextureCubeMapPositiveZ, mgl::TextureTarget::TextureCubeMapNegativeZ};

    mgl::FramebufferAttachment color0() { return mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0); }
    mgl::FramebufferAttachment color1() { return mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color1); }

    // Cell origins of the 2x2 grid in window coordinates (GL origin is the lower left corner).
    struct CellRect
    {
        int x, y;
    };

    struct Probe
    {
        int x, y;
        lr::Rgb8 expected;
    };

    class LayeredRendering final : public demo::Demo
    {
    public:
        // =====================================================================================================
        void init(demo::Env& env) override
        {
            using ST = mgl::ShaderType;
            prog_array_ = make_program({{ST::Vertex, env.load_text("shaders/points.vert")}, {ST::Geometry, env.load_text("shaders/array_layers.geom")},
                                        {ST::Fragment, env.load_text("shaders/flat_color.frag")}}, "array layers");
            prog_volume_ = make_program({{ST::Vertex, env.load_text("shaders/points.vert")}, {ST::Geometry, env.load_text("shaders/volume_slices.geom")},
                                         {ST::Fragment, env.load_text("shaders/flat_color.frag")}}, "volume slices");
            prog_env_ = make_program({{ST::Vertex, env.load_text("shaders/env.vert")}, {ST::Geometry, env.load_text("shaders/env.geom")},
                                      {ST::Fragment, env.load_text("shaders/env.frag")}}, "environment");
            prog_sky_ = make_program({{ST::Vertex, env.load_text("shaders/fullscreen.vert")}, {ST::Fragment, env.load_text("shaders/sky.frag")}}, "sky");
            prog_sphere_ = make_program({{ST::Vertex, env.load_text("shaders/sphere.vert")}, {ST::Fragment, env.load_text("shaders/sphere.frag")}}, "sphere");
            prog_tile_cube_ = make_program({{ST::Vertex, env.load_text("shaders/fullscreen.vert")}, {ST::Fragment, env.load_text("shaders/show_cube_face.frag")}}, "cube tile");
            prog_tile_array_ = make_program({{ST::Vertex, env.load_text("shaders/fullscreen.vert")}, {ST::Fragment, env.load_text("shaders/show_array.frag")}}, "array tile");
            prog_tile_volume_ = make_program({{ST::Vertex, env.load_text("shaders/fullscreen.vert")}, {ST::Fragment, env.load_text("shaders/show_volume.frag")}}, "volume tile");

            // --- textures: all immutable storage, one level, NEAREST (texelFetch needs a complete texture) -----------
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kTexArray]);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 1, mgl::SizedInternalFormat::Rgba8, kLayerSize, kLayerSize, kLayers);
            nearest(mgl::TextureTarget::Texture2DArray);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_[kTexVolume]);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture3D, 1, mgl::SizedInternalFormat::Rgba8, kLayerSize, kLayerSize, kLayers);
            nearest(mgl::TextureTarget::Texture3D);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kTexCube]);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::Rgba8, kCubeSize, kCubeSize);
            nearest(mgl::TextureTarget::TextureCubeMap);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kTexCubeDepth]);
            mgl::glTexStorage2D(mgl::TextureTarget::TextureCubeMap, 1, mgl::SizedInternalFormat::DepthComponent24, kCubeSize, kCubeSize);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kTexSmall]);
            mgl::glTexStorage3D(mgl::TextureTarget::Texture2DArray, 1, mgl::SizedInternalFormat::Rgba8, kSmall, kSmall, kLayers);
            nearest(mgl::TextureTarget::Texture2DArray);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});

            // --- framebuffers: the whole texture level is attached, so the attachments are layered -----------------
            mgl::glGenFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboArray]);
            mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), tex_[kTexArray], 0);
            require_complete("array FBO");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboVolume]);
            mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), tex_[kTexVolume], 0);
            require_complete("3D FBO");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboCube]);
            mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), tex_[kTexCube], 0);
            mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferAttachment::Depth, tex_[kTexCubeDepth], 0);
            require_complete("cube FBO (layered color + layered depth)");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            // --- environment geometry and the mirror ball -------------------------------------------------------------
            quads_ = lr::environment();
            const std::vector<float> env_vertices = lr::environment_vertices(quads_);
            env_vertex_count_ = static_cast<GLsizei>(env_vertices.size() / 6);
            mgl::glGenBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kBufEnv]);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(env_vertices), mgl::BufferUsage::StaticDraw);

            std::vector<float> sphere;
            std::vector<std::uint16_t> sphere_index;
            constexpr int kLon = 96, kLat = 48;
            for (int j = 0; j <= kLat; ++j)
                for (int i = 0; i <= kLon; ++i)
                {
                    const float phi = 3.14159265f * static_cast<float>(j) / static_cast<float>(kLat);
                    const float theta = 2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(kLon);
                    sphere.insert(sphere.end(), {std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)});
                }
            for (int j = 0; j < kLat; ++j)
                for (int i = 0; i < kLon; ++i)
                {
                    const auto at = [&](int a, int b) { return static_cast<std::uint16_t>(b * (kLon + 1) + a); };
                    for (const std::uint16_t idx : {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1)}) sphere_index.push_back(idx);
                }
            sphere_index_count_ = static_cast<GLsizei>(sphere_index.size());
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kBufSphere]);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(sphere), mgl::BufferUsage::StaticDraw);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buf_[kBufSphereIndex]);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, std::span<const std::uint16_t>(sphere_index), mgl::BufferUsage::StaticDraw);

            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glBindVertexArray(vao_[kVaoEnv]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kBufEnv]);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, 24, demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, 24, demo::buffer_offset(12));
            mgl::glBindVertexArray(vao_[kVaoSphere]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kBufSphere]);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, 12, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, buf_[kBufSphereIndex]);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);  // attribute-less draws (gl_VertexID)
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            mgl::glGenSamplers(static_cast<GLsizei>(samplers_.size()), samplers_.data());
            for (std::size_t i = 0; i < samplers_.size(); ++i)
            {
                const GLint filter = i == kSamplerNearest ? GL_NEAREST : GL_LINEAR;
                mgl::glSamplerParameteri(samplers_[i], mgl::SamplerParameter::MinFilter, filter);
                mgl::glSamplerParameteri(samplers_[i], mgl::SamplerParameter::MagFilter, filter);
                mgl::glSamplerParameteri(samplers_[i], mgl::SamplerParameter::WrapS, GL_CLAMP_TO_EDGE);
                mgl::glSamplerParameteri(samplers_[i], mgl::SamplerParameter::WrapT, GL_CLAMP_TO_EDGE);
                mgl::glSamplerParameteri(samplers_[i], mgl::SamplerParameter::WrapR, GL_CLAMP_TO_EDGE);
            }
        }

        // =====================================================================================================
        void render(demo::Env& env, int frame) override
        {
            time_ = kTime0 + static_cast<float>(frame) / 60.0f;
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::Blend);
            mgl::glDisable(mgl::Capability::CullFace);

            // ---- the three layered passes ------------------------------------------------------------------------
            render_array_layers();
            render_volume_slices();
            render_environment();

            // ---- the display: four 128x128 cells ----------------------------------------------------------------
            const int cw = env.width / 2, ch = env.height / 2;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

            // top left: the mirror ball
            cell_preview_ = {0, ch};
            cell_net_ = {cw, ch};
            cell_array_ = {0, 0};
            cell_volume_ = {cw, 0};
            cell_w_ = cw;
            cell_h_ = ch;
            draw_preview(cell_preview_, cw, ch);
            // top right: the cube net
            draw_net(cell_net_);
            // bottom left / right: array layers and volume slices, texel for texel
            draw_tiles(cell_array_, prog_array_tiles(), tex_[kTexArray], mgl::TextureTarget::Texture2DArray);
            draw_tiles(cell_volume_, prog_volume_tiles(), tex_[kTexVolume], mgl::TextureTarget::Texture3D);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glBindSampler(0, mgl::SamplerId{0});
        }

        // =====================================================================================================
        void validate(demo::Env& env, demo::Check& check) override
        {
            demo::drain_gl_errors();
            check_attachment_queries(check);
            check_completeness(check);
            check_errors(check);
            check_layer_images(check);
            check_cube_texture_layer(check);
            check_display(env, check);
            check_layer_semantics(check);
            check_default_layers(check);
            demo::drain_gl_errors();
            // The capture reads the default framebuffer after validate(): restore the state render() leaves.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // =====================================================================================================
        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            for (const mgl::ProgramId p : {prog_array_, prog_volume_, prog_env_, prog_sky_, prog_sphere_, prog_tile_cube_, prog_tile_array_, prog_tile_volume_})
                mgl::glDeleteProgram(p);
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glDeleteSamplers(static_cast<GLsizei>(samplers_.size()), samplers_.data());
        }

    private:
        static constexpr int kSmall = 8;  // size of the semantic-test texture
        enum Tex : std::size_t { kTexArray, kTexVolume, kTexCube, kTexCubeDepth, kTexSmall, kTexCount };
        enum Fbo : std::size_t { kFboArray, kFboVolume, kFboCube, kFboRead, kFboCount };
        enum Buf : std::size_t { kBufEnv, kBufSphere, kBufSphereIndex, kBufCount };
        enum Vao : std::size_t { kVaoEnv, kVaoSphere, kVaoEmpty, kVaoCount };
        enum Sampler : std::size_t { kSamplerNearest, kSamplerLinear, kSamplerCount };

        mgl::ProgramId prog_array_tiles() const { return prog_tile_array_; }
        mgl::ProgramId prog_volume_tiles() const { return prog_tile_volume_; }

        // --- helpers ---------------------------------------------------------------------------------------------
        static mgl::ProgramId make_program(std::initializer_list<std::pair<mgl::ShaderType, std::string>> sources, const char* label)
        {
            std::vector<mgl::ShaderId> shaders;
            for (const auto& [type, text] : sources)
            {
                std::string log;
                const mgl::ShaderId s = demo::compile_shader(type, text, log);
                if (s.value == 0) throw demo::Failure(std::format("{}: shader failed to compile:\n{}", label, log));
                shaders.push_back(s);
            }
            std::string log;
            const mgl::ProgramId p = demo::link_program(shaders, log);
            for (const mgl::ShaderId s : shaders) mgl::glDeleteShader(s);
            if (p.value == 0) throw demo::Failure(std::format("{}: program failed to link:\n{}", label, log));
            return p;
        }

        static void nearest(mgl::TextureTarget target)
        {
            mgl::glTexParameteri(target, mgl::TextureParameterSetter::MinFilter, GL_NEAREST);
            mgl::glTexParameteri(target, mgl::TextureParameterSetter::MagFilter, GL_NEAREST);
        }

        static void require_complete(const char* what)
        {
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{} is not framebuffer complete: {}", what, demo::framebuffer_status_name(status)));
        }

        static void clear_color(float r, float g, float b, float a)
        {
            const float v[4] = {r, g, b, a};
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, v);
        }

        // --- the layered passes ----------------------------------------------------------------------------------
        void render_array_layers() const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboArray]);
            mgl::glViewport(0, 0, kLayerSize, kLayerSize);
            mgl::glDisable(mgl::Capability::DepthTest);
            clear_color(0.0f, 0.0f, 0.0f, 0.0f);  // clears ALL layers
            mgl::glUseProgram(prog_array_);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, kLayers);  // four points: gl_PrimitiveIDIn 0..3 = the layers
        }
        void render_volume_slices() const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboVolume]);
            mgl::glViewport(0, 0, kLayerSize, kLayerSize);
            clear_color(0.0f, 0.0f, 0.0f, 0.0f);
            mgl::glUseProgram(prog_volume_);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            mgl::glDrawArrays(mgl::PrimitiveType::Points, 0, 1);  // one point, four invocations
        }
        void render_environment() const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboCube]);
            mgl::glViewport(0, 0, kCubeSize, kCubeSize);
            clear_color(0.0f, 0.0f, 0.0f, 0.0f);
            const float one = 1.0f;
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Depth, 0, &one);
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glUseProgram(prog_env_);
            std::array<float, 6 * 16> matrices{};
            for (int face = 0; face < 6; ++face)
            {
                const Mat4 m = lr::face_viewproj(face, kEnvNear, kEnvFar);
                std::copy(m.m.begin(), m.m.end(), matrices.begin() + face * 16);
            }
            mgl::glUniformMatrix4fv(loc(0), 6, GL_FALSE, matrices.data());
            mgl::glBindVertexArray(vao_[kVaoEnv]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, env_vertex_count_);
            mgl::glDisable(mgl::Capability::DepthTest);
        }

        // --- the display ------------------------------------------------------------------------------------------
        struct PreviewCamera
        {
            Vec3 eye;
            Mat4 viewproj, inverse;
        };

        static PreviewCamera preview_camera(float t, float aspect)
        {
            PreviewCamera c;
            const float az = 0.55f + 0.5f * t, el = demo::math::radians(16.0f), dist = 3.0f;
            c.eye = {dist * std::cos(el) * std::cos(az), dist * std::sin(el), dist * std::cos(el) * std::sin(az)};
            const Mat4 view = demo::math::look_at(c.eye, {0, 0, 0}, {0, 1, 0});
            const Mat4 proj = demo::math::perspective(demo::math::radians(kPreviewFovDegrees), aspect, 0.5f, 20.0f);
            c.viewproj = proj * view;
            c.inverse = invert(c.viewproj);
            return c;
        }

        // General 4x4 inverse (cofactor expansion); the matrices here are always invertible.
        static Mat4 invert(const Mat4& a)
        {
            const float* m = a.data();
            std::array<float, 16> inv{};
            inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
            inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
            inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
            inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
            inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
            inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
            inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
            inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
            inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
            inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
            inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
            inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
            inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
            inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
            inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
            inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
            const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
            Mat4 r;
            for (std::size_t i = 0; i < 16; ++i) r.m[i] = inv[i] / det;
            return r;
        }

        void bind_cube_with(std::size_t sampler) const
        {
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kTexCube]);
            mgl::glBindSampler(0, samplers_[sampler]);
        }

        void draw_preview(CellRect cell, int w, int h) const
        {
            mgl::glViewport(cell.x, cell.y, w, h);
            mgl::glScissor(cell.x, cell.y, w, h);
            const PreviewCamera cam = preview_camera(time_, static_cast<float>(w) / static_cast<float>(h));
            bind_cube_with(kSamplerLinear);
            mgl::glDisable(mgl::Capability::DepthTest);
            // skybox: the far plane seen through every pixel
            mgl::glUseProgram(prog_sky_);
            mgl::glUniformMatrix4fv(loc(0), 1, GL_FALSE, cam.inverse.data());
            mgl::glUniform3f(loc(1), cam.eye.x, cam.eye.y, cam.eye.z);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            // the mirror ball
            mgl::glEnable(mgl::Capability::DepthTest);
            mgl::glUseProgram(prog_sphere_);
            mgl::glUniformMatrix4fv(loc(0), 1, GL_FALSE, cam.viewproj.data());
            mgl::glUniform3f(loc(1), cam.eye.x, cam.eye.y, cam.eye.z);
            mgl::glBindVertexArray(vao_[kVaoSphere]);
            mgl::glDrawElements(mgl::PrimitiveType::Triangles, sphere_index_count_, mgl::IndexType::UnsignedShort, demo::buffer_offset(0));
            mgl::glDisable(mgl::Capability::DepthTest);
        }

        // Cube net: 4 x 3 tiles of 32x32 pixels, centred vertically in the cell.
        // {face, column, row} (row 0 = bottom): -X +Z +X -Z in the middle row, +Y above and -Y below +Z.
        static constexpr std::array<std::array<int, 3>, 6> kNet = {{{1, 0, 1}, {4, 1, 1}, {0, 2, 1}, {5, 3, 1}, {2, 1, 2}, {3, 1, 0}}};

        static CellRect net_tile(CellRect cell, const std::array<int, 3>& t) { return {cell.x + t[1] * 32, cell.y + 16 + t[2] * 32}; }

        void draw_net(CellRect cell) const
        {
            mgl::glViewport(cell.x, cell.y, cell_w_, cell_h_);
            mgl::glScissor(cell.x, cell.y, cell_w_, cell_h_);
            clear_color(30.0f / 255.0f, 30.0f / 255.0f, 36.0f / 255.0f, 1.0f);
            bind_cube_with(kSamplerNearest);
            mgl::glUseProgram(prog_tile_cube_);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            for (const auto& t : kNet)
            {
                const CellRect o = net_tile(cell, t);
                mgl::glViewport(o.x, o.y, 32, 32);
                mgl::glScissor(o.x, o.y, 32, 32);
                mgl::glUniform2i(loc(0), o.x, o.y);
                mgl::glUniform1i(loc(1), t[0]);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
        }

        void draw_tiles(CellRect cell, mgl::ProgramId program, mgl::TextureId texture, mgl::TextureTarget target) const
        {
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(target, texture);
            mgl::glBindSampler(0, samplers_[kSamplerNearest]);
            mgl::glUseProgram(program);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            for (int n = 0; n < kLayers; ++n)
            {
                const CellRect o = {cell.x + (n % 2) * kLayerSize, cell.y + (n / 2) * kLayerSize};
                mgl::glViewport(o.x, o.y, kLayerSize, kLayerSize);
                mgl::glScissor(o.x, o.y, kLayerSize, kLayerSize);
                mgl::glUniform2i(loc(0), o.x, o.y);
                mgl::glUniform1i(loc(1), n);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
        }

        // =====================================================================================================
        // validate(): attachment queries (ES 3.2 section 9.2.3)
        // =====================================================================================================
        static GLint attachment_value(mgl::FramebufferAttachment attachment, mgl::FramebufferAttachmentParameter pname)
        {
            GLint v = -999;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, attachment, pname, &v);
            return v;
        }

        void check_attachment_queries(demo::Check& check)
        {
            using P = mgl::FramebufferAttachmentParameter;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[kFboRead]);
            struct Case
            {
                const char* name;
                std::function<void()> attach;
                GLint layered;
                GLint layer;
            };
            mgl::TextureId plain2d{};
            mgl::glGenTextures(1, &plain2d);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, plain2d);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, 4, 4);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            const auto tex_array = tex_[kTexArray], tex_volume = tex_[kTexVolume], tex_cube = tex_[kTexCube];
            const Case cases[] = {
                {"glFramebufferTexture(2D array)", [&] { mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), tex_array, 0); }, GL_TRUE, 0},
                {"glFramebufferTexture(3D)", [&] { mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), tex_volume, 0); }, GL_TRUE, 0},
                {"glFramebufferTexture(cube map)", [&] { mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), tex_cube, 0); }, GL_TRUE, 0},
                {"glFramebufferTexture(plain 2D texture)", [&] { mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), plain2d, 0); }, GL_FALSE, 0},
                {"glFramebufferTextureLayer(2D array, layer 2)", [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, color0(), tex_array, 0, 2); }, GL_FALSE, 2},
                {"glFramebufferTextureLayer(3D, layer 3)", [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, color0(), tex_volume, 0, 3); }, GL_FALSE, 3},
                {"glFramebufferTexture2D(cube face +Z)",
                 [&] { mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, color0(), mgl::TextureTarget::TextureCubeMapPositiveZ, tex_cube, 0); }, GL_FALSE, 0},
                {"glFramebufferTexture2D(plain 2D texture)", [&] { mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, color0(), mgl::TextureTarget::Texture2D, plain2d, 0); },
                 GL_FALSE, 0},
            };
            for (const Case& c : cases)
            {
                demo::drain_gl_errors();
                c.attach();
                check.expect_no_gl_error(std::format("{}: no error", c.name));
                check.expect_eq(attachment_value(color0(), P::Layered), c.layered, std::format("{}: FRAMEBUFFER_ATTACHMENT_LAYERED (section 9.2.3)", c.name));
                check.expect_eq(attachment_value(color0(), P::ObjectType), static_cast<GLint>(GL_TEXTURE), std::format("{}: OBJECT_TYPE is TEXTURE", c.name));
                check.expect_eq(attachment_value(color0(), P::TextureLevel), 0, std::format("{}: TEXTURE_LEVEL", c.name));
                check.expect_eq(attachment_value(color0(), P::TextureLayer), c.layer,
                                std::format("{}: TEXTURE_LAYER (the layer for a non-layered 3D/array attachment, zero otherwise)", c.name));
            }
            // Detached again: the state returns to its defaults (section 9.2.8.1) and LAYERED is FALSE.
            mgl::glFramebufferTexture(mgl::FramebufferTarget::Framebuffer, color0(), mgl::TextureId{0}, 0);
            check.expect_eq(attachment_value(color0(), P::ObjectType), static_cast<GLint>(GL_NONE), "detaching with texture 0: OBJECT_TYPE is NONE");
            check.expect_no_gl_error("attachment queries: no GL error left");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteTextures(1, &plain2d);
        }

        // =====================================================================================================
        // validate(): completeness (ES 3.2 sections 9.4.1, 9.4.2)
        // =====================================================================================================
        void expect_status(demo::Check& check, mgl::FramebufferStatus expected, const std::string& what)
        {
            const mgl::FramebufferStatus got = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            check.expect(got == expected, what, std::format("got {}, expected {}", demo::framebuffer_status_name(got), demo::framebuffer_status_name(expected)));
        }

        void check_completeness(demo::Check& check)
        {
            using S = mgl::FramebufferStatus;
            const mgl::TextureId arr = tex_[kTexArray], vol = tex_[kTexVolume], cube = tex_[kTexCube];
            mgl::FramebufferId fbo{};
            mgl::RenderbufferId rb{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::DepthComponent24, kLayerSize, kLayerSize);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            const auto F = mgl::FramebufferTarget::Framebuffer;

            expect_status(check, S::IncompleteMissingAttachment, "no attachment at all: FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT (control)");

            mgl::glFramebufferTexture(F, color0(), arr, 0);
            expect_status(check, S::Complete, "layered 2D array color: COMPLETE");
            mgl::glFramebufferTexture(F, color0(), vol, 0);
            expect_status(check, S::Complete, "layered 3D color: COMPLETE");
            mgl::glFramebufferTexture(F, color0(), cube, 0);
            expect_status(check, S::Complete, "layered cube map color: COMPLETE");

            // Rule 1: "If any framebuffer attachment is layered, all populated attachments must be layered."
            mgl::glFramebufferTexture(F, color0(), arr, 0);
            mgl::glFramebufferRenderbuffer(F, mgl::FramebufferAttachment::Depth, mgl::RenderbufferTarget::Renderbuffer, rb);
            expect_status(check, S::IncompleteLayerTargets, "layered color + non-layered depth renderbuffer: FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");
            mgl::glFramebufferTextureLayer(F, color0(), arr, 0, 1);
            expect_status(check, S::Complete, "control: the same depth renderbuffer with a NON-layered color (TextureLayer) is COMPLETE");
            mgl::glFramebufferRenderbuffer(F, mgl::FramebufferAttachment::Depth, mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});

            mgl::glFramebufferTexture(F, color0(), arr, 0);
            mgl::glFramebufferTextureLayer(F, color1(), arr, 0, 0);
            expect_status(check, S::IncompleteLayerTargets, "layered color0 + non-layered color1: FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");
            mgl::glFramebufferTexture(F, color1(), arr, 0);
            expect_status(check, S::Complete, "control: two layered color attachments of the SAME target (2D array) are COMPLETE");

            // Rule 2: "all populated color attachments must be from textures of the same target".
            mgl::glFramebufferTexture(F, color1(), vol, 0);
            expect_status(check, S::IncompleteLayerTargets, "layered 2D array color0 + layered 3D color1 (different targets): FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");
            mgl::glFramebufferTexture(F, color1(), cube, 0);
            expect_status(check, S::IncompleteLayerTargets, "layered 2D array color0 + layered cube map color1: FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");
            mgl::glFramebufferTexture(F, color1(), mgl::TextureId{0}, 0);

            // Attachment completeness: a non-layered attachment must select an existing layer (section 9.4.1).
            mgl::glFramebufferTextureLayer(F, color0(), arr, 0, kLayers);
            check.expect_no_gl_error("glFramebufferTextureLayer(layer = layer count) is accepted (layer < MAX_ARRAY_TEXTURE_LAYERS)");
            expect_status(check, S::IncompleteAttachment, "a non-layered attachment of layer 4 of a 4-layer array: FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
            mgl::glFramebufferTextureLayer(F, color0(), arr, 0, kLayers - 1);
            expect_status(check, S::Complete, "control: layer 3 of the same array is COMPLETE");

            mgl::glBindFramebuffer(F, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glDeleteRenderbuffers(1, &rb);

            // The framebuffers of the demo itself: layered color + layered depth (cube map) is complete.
            mgl::glBindFramebuffer(F, fbo_[kFboCube]);
            expect_status(check, S::Complete, "the environment framebuffer (layered cube color + layered cube depth) is COMPLETE");
            mgl::glBindFramebuffer(F, mgl::FramebufferId{0});
            check.expect_no_gl_error("completeness tests: no GL error left");

            const int max_layers = demo::get_integer(mgl::GetParameter::MaxFramebufferLayers);
            check.expect(max_layers >= 256, "MAX_FRAMEBUFFER_LAYERS >= 256 (ES 3.2 Table 21.40)", std::format("value {}", max_layers));
            check.expect(max_layers >= kLayers && demo::get_integer(mgl::GetParameter::MaxArrayTextureLayers) >= 256, "MAX_ARRAY_TEXTURE_LAYERS >= 256",
                         std::format("MAX_ARRAY_TEXTURE_LAYERS {}", demo::get_integer(mgl::GetParameter::MaxArrayTextureLayers)));
        }

        // =====================================================================================================
        // validate(): errors of glFramebufferTexture / glFramebufferTextureLayer (ES 3.2 section 9.2.8)
        // =====================================================================================================
        void check_errors(demo::Check& check)
        {
            const auto F = mgl::FramebufferTarget::Framebuffer;
            const mgl::TextureId arr = tex_[kTexArray];
            mgl::TextureId plain2d{};
            mgl::glGenTextures(1, &plain2d);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, plain2d);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, 4, 4);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            demo::drain_gl_errors();
            mgl::glBindFramebuffer(F, mgl::FramebufferId{0});
            mgl::glFramebufferTexture(F, color0(), arr, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glFramebufferTexture with the default framebuffer bound -> INVALID_OPERATION");
            mgl::glFramebufferTextureLayer(F, color0(), arr, 0, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glFramebufferTextureLayer with the default framebuffer bound -> INVALID_OPERATION");

            mgl::glBindFramebuffer(F, fbo_[kFboRead]);
            mgl::glFramebufferTexture(F, color0(), mgl::TextureId{0xFFFFFFu}, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glFramebufferTexture with a name that is not a texture -> INVALID_VALUE");
            mgl::glFramebufferTexture(F, color0(), arr, 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glFramebufferTexture(level 1) of an immutable 1-level texture -> INVALID_VALUE (level not supported)");
            mgl::glFramebufferTexture(F, color0(), arr, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glFramebufferTexture(level -1) -> INVALID_VALUE");

            const int max_array = demo::get_integer(mgl::GetParameter::MaxArrayTextureLayers);
            mgl::glFramebufferTextureLayer(F, color0(), arr, 0, max_array);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer(layer = MAX_ARRAY_TEXTURE_LAYERS) -> INVALID_VALUE");
            mgl::glFramebufferTextureLayer(F, color0(), arr, 0, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer(layer -1) -> INVALID_VALUE");
            mgl::glFramebufferTextureLayer(F, color0(), plain2d, 0, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glFramebufferTextureLayer with a plain 2D texture -> INVALID_OPERATION");
            mgl::glFramebufferTexture2D(F, color0(), mgl::TextureTarget::Texture2D, arr, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glFramebufferTexture2D(TEXTURE_2D) with the name of a 2D array texture -> INVALID_OPERATION");
            // An error leaves the attachment as it was: nothing is attached to the (new) scratch framebuffer.
            check.expect_eq(attachment_value(color0(), mgl::FramebufferAttachmentParameter::ObjectType), static_cast<GLint>(GL_NONE),
                            "after the failed calls nothing is attached (an erroring command changes no state)");
            mgl::glBindFramebuffer(F, mgl::FramebufferId{0});
            mgl::glDeleteTextures(1, &plain2d);
            check.expect_no_gl_error("error tests: no GL error left");
        }

        // =====================================================================================================
        // validate(): exact comparison of every layer
        // =====================================================================================================
        // Reads one layer of a layered texture. Cube map faces are attached with glFramebufferTexture2D and the face target
        // (the ES 2.0 way); glFramebufferTextureLayer(cube map) is checked separately in check_cube_texture_layer().
        std::vector<std::uint8_t> read_layer(mgl::TextureId texture, int layer, int size, bool cube = false)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo_[kFboRead]);
            if (cube)
                mgl::glFramebufferTexture2D(mgl::FramebufferTarget::ReadFramebuffer, color0(), kCubeFaceTargets[static_cast<std::size_t>(layer)], texture, 0);
            else
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::ReadFramebuffer, color0(), texture, 0, layer);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            return demo::read_rgba8(0, 0, size, size);
        }

        static std::size_t differing_bytes(const std::vector<std::uint8_t>& got, const lr::Image& expected)
        {
            if (got.size() != expected.size()) return std::max(got.size(), expected.size());
            std::size_t n = 0;
            for (std::size_t i = 0; i < got.size(); ++i) n += got[i] != expected[i];
            return n;
        }

        void check_layer_images(demo::Check& check)
        {
            struct Layered
            {
                const char* name;
                mgl::TextureId texture;
                lr::Image (*model)(int);
            };
            const Layered images[] = {
                {"2D array (points, layer = gl_PrimitiveIDIn)", tex_[kTexArray], &lr::array_layer},
                {"3D texture (one point, layout(invocations = 4), slice = gl_InvocationID)", tex_[kTexVolume], &lr::volume_slice},
            };
            for (const Layered& l : images)
            {
                std::size_t total = 0;
                std::string detail;
                std::vector<std::uint8_t> first;
                for (int layer = 0; layer < kLayers; ++layer)
                {
                    const auto got = read_layer(l.texture, layer, kLayerSize);
                    const std::size_t d = differing_bytes(got, l.model(layer));
                    total += d;
                    if (d != 0) detail += std::format(" layer {}: {} bytes differ;", layer, d);
                    if (layer == 0) first = got;
                }
                check.expect(total == 0, std::format("{}: all {} layers equal the CPU images byte for byte ({} x {} x 4 bytes each)", l.name, kLayers, kLayerSize, kLayerSize), detail);
                check.expect(differing_bytes(first, l.model(1)) > 1000,
                             std::format("negative control ({}): layer 0 compared with the model of layer 1 differs", l.name));
            }
            // Layers differ from each other: the pass really rendered different content per layer.
            check.expect(differing_bytes(read_layer(tex_[kTexArray], 0, kLayerSize), lr::array_layer(0)) == 0 &&
                             differing_bytes(read_layer(tex_[kTexArray], 3, kLayerSize), lr::array_layer(0)) > 1000,
                         "array layer 3 is not a copy of layer 0");

            // ---- cube map: all six faces, texel by texel -------------------------------------------------------------
            std::size_t compared = 0, bad = 0, alpha_bad = 0, skipped = 0;
            std::string first_bad;
            std::size_t wrong_model_bad = 0;
            for (int face = 0; face < 6; ++face)
            {
                const auto got = read_layer(tex_[kTexCube], face, kCubeSize, true);
                for (int j = 0; j < kCubeSize; ++j)
                    for (int i = 0; i < kCubeSize; ++i)
                    {
                        lr::Rgb8 want{};
                        if (!lr::texel_is_uniform(quads_, face, i, j, kCubeSize, &want))
                        {
                            ++skipped;
                            continue;
                        }
                        ++compared;
                        const std::uint8_t* px = &got[(static_cast<std::size_t>(j) * kCubeSize + static_cast<std::size_t>(i)) * 4];
                        if (px[0] != want[0] || px[1] != want[1] || px[2] != want[2])
                        {
                            if (bad == 0) first_bad = std::format("face {} texel ({},{}) is ({},{},{}), model ({},{},{})", lr::kFaceNames[static_cast<std::size_t>(face)], i, j, px[0], px[1], px[2], want[0], want[1], want[2]);
                            ++bad;
                        }
                        if (px[3] != 40 * face + 15) ++alpha_bad;
                        // Negative control: the same texel against the model of the OPPOSITE face.
                        lr::Rgb8 other{};
                        if (lr::texel_is_uniform(quads_, face ^ 1, i, j, kCubeSize, &other) && (px[0] != other[0] || px[1] != other[1] || px[2] != other[2])) ++wrong_model_bad;
                    }
            }
            std::printf("   cube map: %zu texels compared exactly, %zu skipped next to polygon edges\n", compared, skipped);
            check.expect(bad == 0 && compared > 6 * 64 * 64 / 2, "cube map: every unambiguous texel of all six faces equals the ray-cast of the room (the face orientation of Table 8.19)",
                         bad == 0 ? std::string{} : std::format("{} texels differ, first: {}", bad, first_bad));
            check.expect(alpha_bad == 0, "cube map: the fragment shader's gl_Layer is the face written (alpha tag 40 * face + 15 in every texel)", std::format("{} texels differ", alpha_bad));
            check.expect(wrong_model_bad > 1000, "negative control (cube map): comparing a face with the model of the opposite face differs", std::format("{} texels", wrong_model_bad));
            check.expect_no_gl_error("layer readback: no GL error");
        }

        // glFramebufferTextureLayer accepts a cube map texture and translates the layer into a face (ES 3.2 section 9.2.8, Table 8.24).
        void check_cube_texture_layer(demo::Check& check)
        {
            const auto R = mgl::FramebufferTarget::ReadFramebuffer;
            mgl::glBindFramebuffer(R, fbo_[kFboRead]);
            mgl::glFramebufferTexture(R, color0(), tex_[kTexArray], 0);  // a known previous attachment
            demo::drain_gl_errors();
            mgl::glFramebufferTextureLayer(R, color0(), tex_[kTexCube], 0, 2);  // layer 2 = TEXTURE_CUBE_MAP_POSITIVE_Y
            const mgl::ErrorCode error = demo::drain_gl_errors();
            GLint name = -1;
            mgl::glGetFramebufferAttachmentParameteriv(R, color0(), mgl::FramebufferAttachmentParameter::ObjectName, &name);
            const bool attached = name == static_cast<GLint>(tex_[kTexCube].value);
            check.expect_or_known_deviation(error == mgl::ErrorCode::NoError && attached,
                                            "glFramebufferTextureLayer(cube map, layer 2) is accepted and attaches the cube map (cube map face +Y)",
                                            std::format("error {}, attached object name {} (cube map is {})", demo::error_name(error), name, tex_[kTexCube].value), "Mesa",
                                            "Mesa accepts glFramebufferTextureLayer on a cube map without an error but leaves the previous attachment in place");
            if (attached)
            {
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                const auto via_layer = demo::read_rgba8(0, 0, kCubeSize, kCubeSize);
                check.expect(via_layer == read_layer(tex_[kTexCube], 2, kCubeSize, true), "the cube face attached with TextureLayer(layer 2) holds the same image as TEXTURE_CUBE_MAP_POSITIVE_Y");
            }
            mgl::glBindFramebuffer(R, mgl::FramebufferId{0});
        }

        // =====================================================================================================
        // validate(): the displayed cells
        // =====================================================================================================
        void check_display(demo::Env& env, demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            const int cw = cell_w_, ch = cell_h_;
            (void)env;

            // ---- bottom cells: the layers, texel for texel -------------------------------------------------------------
            auto tiles_match = [&](CellRect cell, lr::Image (*model)(int)) {
                const auto got = demo::read_rgba8(cell.x, cell.y, cw, ch);
                std::size_t bad = 0;
                for (int n = 0; n < kLayers; ++n)
                {
                    const lr::Image img = model(n);
                    for (int y = 0; y < kLayerSize; ++y)
                        for (int x = 0; x < kLayerSize; ++x)
                            for (std::size_t k = 0; k < 4; ++k)
                            {
                                const int gx = (n % 2) * kLayerSize + x, gy = (n / 2) * kLayerSize + y;
                                bad += got[(static_cast<std::size_t>(gy) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(gx)) * 4 + k] !=
                                       img[(static_cast<std::size_t>(y) * kLayerSize + static_cast<std::size_t>(x)) * 4 + k];
                            }
                }
                return bad;
            };
            check.expect_eq(tiles_match(cell_array_, &lr::array_layer), std::size_t{0},
                            "bottom left: the 2x2 tiles (texelFetch from sampler2DArray) equal the four CPU layer images, all 128x128 pixels");
            check.expect_eq(tiles_match(cell_volume_, &lr::volume_slice), std::size_t{0},
                            "bottom right: the 2x2 tiles (texelFetch from sampler3D) equal the four CPU slice images, all 128x128 pixels");

            // ---- top right: the cube net -------------------------------------------------------------------------------------
            {
                const auto got = demo::read_rgba8(cell_net_.x, cell_net_.y, cw, ch);
                std::size_t compared = 0, bad = 0;
                std::string first_bad;
                for (const auto& tile : kNet)
                {
                    const CellRect o = net_tile({0, 0}, tile);
                    for (int py = 0; py < 32; ++py)
                        for (int px = 0; px < 32; ++px)
                        {
                            lr::Rgb8 want{};
                            if (!lr::texel_is_uniform(quads_, tile[0], 2 * px, 2 * (31 - py), kCubeSize, &want)) continue;
                            ++compared;
                            const std::uint8_t* p = &got[(static_cast<std::size_t>(o.y + py) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(o.x + px)) * 4];
                            if (p[0] != want[0] || p[1] != want[1] || p[2] != want[2] || p[3] != 255)
                            {
                                if (bad == 0) first_bad = std::format("face {} tile pixel ({},{}) is ({},{},{}), model ({},{},{})", lr::kFaceNames[static_cast<std::size_t>(tile[0])], px, py, p[0], p[1], p[2], want[0], want[1], want[2]);
                                ++bad;
                            }
                        }
                }
                check.expect(bad == 0 && compared > 6 * 32 * 32 / 2, std::format("top right: {} pixels of the six cube-net tiles equal the model of the texel they sample (direction -> face -> texel)", compared),
                             bad == 0 ? std::string{} : std::format("{} differ, first: {}", bad, first_bad));
                // the empty corners of the net keep the cell clear colour
                const std::uint8_t* corner = &got[(static_cast<std::size_t>(2) * static_cast<std::size_t>(cw) + 2) * 4];
                check.expect(corner[0] == 30 && corner[1] == 30 && corner[2] == 36, "top right: the unused corner of the net keeps the cell clear colour (viewport/scissor per tile)");
            }

            // ---- top left: sky and mirror ball ------------------------------------------------------------------------------
            {
                const auto got = demo::read_rgba8(cell_preview_.x, cell_preview_.y, cw, ch);
                const PreviewCamera cam = preview_camera(time_, static_cast<float>(cw) / static_cast<float>(ch));
                using demo::math::cross;
                using demo::math::dot;
                using demo::math::normalize;
                const Vec3 f = normalize(Vec3{0, 0, 0} - cam.eye);
                const Vec3 r = normalize(cross(f, {0, 1, 0}));
                const Vec3 u = cross(r, f);
                const float tan_half = std::tan(demo::math::radians(kPreviewFovDegrees) * 0.5f);
                const float px_per_unit = static_cast<float>(ch) * 0.5f / (tan_half * std::sqrt(dot(cam.eye, cam.eye)));
                std::size_t sky_probes = 0, ball_probes = 0, bad = 0, skipped = 0;
                std::array<lr::Rgb8, 64> distinct{};
                std::size_t distinct_count = 0;
                int worst = 0;
                std::string first_bad;
                for (int y = 1; y < ch; y += 2)
                    for (int x = 1; x < cw; x += 2)
                    {
                        const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(cw) * 2.0f - 1.0f;
                        const float ny = (static_cast<float>(y) + 0.5f) / static_cast<float>(ch) * 2.0f - 1.0f;
                        const Vec3 d = normalize(f + r * (nx * tan_half) + u * (ny * tan_half));
                        const float b = dot(cam.eye, d);
                        const float c = dot(cam.eye, cam.eye) - 1.0f;
                        const float disc = b * b - c;
                        const float impact = std::sqrt(std::max(dot(cam.eye, cam.eye) - b * b, 0.0f));  // distance of the ray from the ball's centre
                        if (std::fabs(impact - 1.0f) * px_per_unit < 3.0f)
                        {
                            ++skipped;  // within 3 pixels of the ball's outline
                            continue;
                        }
                        lr::Rgb8 want{};
                        std::array<float, 3> expected{};
                        bool ball = false;
                        if (disc > 0.0f && -b - std::sqrt(disc) > 0.0f)
                        {
                            ball = true;
                            const float tt = -b - std::sqrt(disc);
                            const Vec3 p = cam.eye + d * tt;
                            const Vec3 n = p;  // unit sphere
                            const Vec3 v = normalize(cam.eye - p);
                            const Vec3 refl = Vec3{-v.x, -v.y, -v.z} - n * (2.0f * dot(Vec3{-v.x, -v.y, -v.z}, n));
                            if (!lr::direction_is_uniform(quads_, refl, 0.08f, &want))
                            {
                                ++skipped;
                                continue;
                            }
                            const float rim = std::pow(1.0f - std::max(dot(n, v), 0.0f), 3.0f);
                            for (std::size_t k = 0; k < 3; ++k) expected[k] = static_cast<float>(want[k]) / 255.0f * (1.0f - 0.12f * rim) + 0.12f * rim;
                        }
                        else
                        {
                            if (!lr::direction_is_uniform(quads_, d, 0.08f, &want))
                            {
                                ++skipped;
                                continue;
                            }
                            for (std::size_t k = 0; k < 3; ++k) expected[k] = static_cast<float>(want[k]) / 255.0f;
                        }
                        const std::uint8_t* p = &got[(static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(x)) * 4];
                        int diff = 0;
                        for (std::size_t k = 0; k < 3; ++k) diff = std::max(diff, std::abs(static_cast<int>(p[k]) - static_cast<int>(std::lround(std::clamp(expected[k], 0.0f, 1.0f) * 255.0f))));
                        worst = std::max(worst, diff);
                        (ball ? ball_probes : sky_probes) += 1;
                        if (ball && std::find(distinct.begin(), distinct.begin() + static_cast<std::ptrdiff_t>(distinct_count), want) == distinct.begin() + static_cast<std::ptrdiff_t>(distinct_count) && distinct_count < distinct.size())
                            distinct[distinct_count++] = want;
                        if (diff > 2)
                        {
                            if (bad == 0) first_bad = std::format("{} pixel ({},{}) is ({},{},{}), model ({},{},{})", ball ? "ball" : "sky", x, y, p[0], p[1], p[2], std::lround(expected[0] * 255.0f), std::lround(expected[1] * 255.0f), std::lround(expected[2] * 255.0f));
                            ++bad;
                        }
                    }
                std::printf("   top left: %zu sky + %zu ball probes, %zu skipped near outlines/colour edges, largest deviation %d/255, %zu distinct reflected colours\n", sky_probes, ball_probes,
                            skipped, worst, distinct_count);
                check.expect(bad == 0, "top left: sky and mirror-ball pixels equal the CPU model (view ray -> ray-cast of the room; reflect() -> the same, plus the rim term), +-2/255 (8-bit rounding, bilinear weights on equal texels)",
                             bad == 0 ? std::string{} : std::format("{} differ, first: {}", bad, first_bad));
                check.expect(sky_probes > 500 && ball_probes > 300 && distinct_count >= 5,
                             "top left: the comparison covers both the skybox and a mirror ball that reflects at least five different environment colours",
                             std::format("{} sky, {} ball probes, {} reflected colours", sky_probes, ball_probes, distinct_count));
            }
            check.expect_no_gl_error("display comparison: no GL error");
        }

        // =====================================================================================================
        // validate(): what the specification defines for gl_Layer (ES 3.2 sections 9.8, 11.3.4.5, GLSL ES 3.20 7.1.4, 7.1.5)
        // =====================================================================================================
        static constexpr const char* kVsFullscreen =
            "#version 320 es\n"
            "void main() {\n"
            "    vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);\n"
            "    gl_Position = vec4(p, 0.0, 1.0);\n"
            "}\n";
        static constexpr const char* kFsUniform =
            "#version 320 es\nprecision highp float;\nlayout(location = 0) uniform vec4 u_color;\nout vec4 frag_color;\nvoid main() { frag_color = u_color; }\n";
        static constexpr const char* kFsLayer =
            "#version 320 es\nprecision highp float;\nout vec4 frag_color;\n"
            "void main() { frag_color = vec4(float(gl_Layer * 64 + 63) / 255.0, 0.0, 0.0, 1.0); }\n";  // red = 63, 127, 191, 255 for layers 0..3

        static std::string geometry_source(const char* layout_in, const char* body)
        {
            return std::format("#version 320 es\nlayout({}) in;\nlayout(triangle_strip, max_vertices = 3) out;\n"
                               "void main() {{ for (int i = 0; i < 3; ++i) {{ gl_Position = gl_in[i].gl_Position; {} EmitVertex(); }} EndPrimitive(); }}\n",
                               layout_in, body);
        }

        // Mean of one layer: every pixel of the 8x8 layer must have this colour.
        std::optional<lr::Rgba8> uniform_layer(int layer)
        {
            const auto px = read_layer(tex_[kTexSmall], layer, kSmall);
            lr::Rgba8 first = {px[0], px[1], px[2], px[3]};
            for (std::size_t i = 0; i < px.size(); i += 4)
                if (px[i] != first[0] || px[i + 1] != first[1] || px[i + 2] != first[2] || px[i + 3] != first[3]) return std::nullopt;
            return first;
        }

        void check_layer_semantics(demo::Check& check)
        {
            const auto F = mgl::FramebufferTarget::Framebuffer;
            mgl::FramebufferId layered{}, single{};
            mgl::glGenFramebuffers(1, &layered);
            mgl::glGenFramebuffers(1, &single);
            mgl::glBindFramebuffer(F, layered);
            mgl::glFramebufferTexture(F, color0(), tex_[kTexSmall], 0);  // all four layers of the 8x8 array
            check.expect(mgl::glCheckFramebufferStatus(F) == mgl::FramebufferStatus::Complete, "semantic tests: the layered 8x8x4 framebuffer is complete");
            mgl::glBindFramebuffer(F, single);
            mgl::glFramebufferTextureLayer(F, color0(), tex_[kTexSmall], 0, 1);  // layer 1 only: NOT layered
            check.expect(mgl::glCheckFramebufferStatus(F) == mgl::FramebufferStatus::Complete, "semantic tests: the non-layered framebuffer (layer 1) is complete");

            using ST = mgl::ShaderType;
            const mgl::ProgramId no_gs = make_program({{ST::Vertex, kVsFullscreen}, {ST::Fragment, kFsUniform}}, "no geometry shader");
            const mgl::ProgramId gs_none = make_program({{ST::Vertex, kVsFullscreen}, {ST::Geometry, geometry_source("triangles", "")}, {ST::Fragment, kFsUniform}}, "gs without gl_Layer");
            const mgl::ProgramId gs_inv = make_program({{ST::Vertex, kVsFullscreen}, {ST::Geometry, geometry_source("triangles, invocations = 4", "gl_Layer = gl_InvocationID;")},
                                                        {ST::Fragment, kFsLayer}}, "gs layer = invocation");
            const mgl::ProgramId gs_three = make_program({{ST::Vertex, kVsFullscreen}, {ST::Geometry, geometry_source("triangles", "gl_Layer = 3;")}, {ST::Fragment, kFsUniform}}, "gs layer 3");
            const mgl::ProgramId gs_two = make_program(
                {{ST::Vertex, kVsFullscreen},
                 {ST::Geometry,
                  "#version 320 es\nlayout(triangles) in;\nlayout(triangle_strip, max_vertices = 6) out;\n"
                  "void main() { for (int i = 0; i < 3; ++i) { gl_Position = gl_in[i].gl_Position; gl_Layer = 1; EmitVertex(); } EndPrimitive();\n"
                  "              for (int i = 0; i < 3; ++i) { gl_Position = gl_in[i].gl_Position; gl_Layer = 3; EmitVertex(); } EndPrimitive(); }\n"},
                 {ST::Fragment, kFsUniform}}, "gs two layers");
            const mgl::ProgramId gs_prov = make_program({{ST::Vertex, kVsFullscreen}, {ST::Geometry, geometry_source("triangles", "gl_Layer = i;")}, {ST::Fragment, kFsUniform}}, "gs layer = vertex index");

            mgl::glDisable(mgl::Capability::DepthTest);
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glBindVertexArray(vao_[kVaoEmpty]);
            mgl::glViewport(0, 0, kSmall, kSmall);
            auto clear_all = [&] {
                mgl::glBindFramebuffer(F, layered);
                clear_color(0.0f, 0.0f, 0.0f, 1.0f);
            };
            auto draw = [&](mgl::FramebufferId fbo, mgl::ProgramId program, float r, float g, float b, bool has_color = true) {
                mgl::glBindFramebuffer(F, fbo);
                mgl::glUseProgram(program);
                if (has_color) mgl::glUniform4f(loc(0), r, g, b, 1.0f);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            };
            const lr::Rgba8 black = {0, 0, 0, 255};
            auto expect_layers = [&](const char* what, std::array<std::optional<lr::Rgba8>, 4> expected) {
                std::string detail;
                bool ok = true;
                for (int l = 0; l < 4; ++l)
                {
                    const auto got = uniform_layer(l);
                    if (!got || *got != *expected[static_cast<std::size_t>(l)])
                    {
                        ok = false;
                        detail += got ? std::format(" layer {} is ({},{},{},{});", l, (*got)[0], (*got)[1], (*got)[2], (*got)[3]) : std::format(" layer {} is not uniform;", l);
                    }
                }
                check.expect(ok, what, detail);
            };

            // 1. Without a geometry shader every fragment has layer 0 (section 9.8).
            clear_all();
            draw(layered, no_gs, 1.0f, 0.0f, 0.0f);
            expect_layers("no geometry shader, layered framebuffer: the layer number is zero (only layer 0 is painted)", {lr::Rgba8{255, 0, 0, 255}, black, black, black});
            // 2. A geometry shader that does not statically assign gl_Layer: also layer 0.
            clear_all();
            draw(layered, gs_none, 0.0f, 1.0f, 0.0f);
            expect_layers("geometry shader without a gl_Layer assignment: the layer number is zero", {lr::Rgba8{0, 255, 0, 255}, black, black, black});
            // 3. gl_Layer = gl_InvocationID, the fragment shader reads the same value back.
            clear_all();
            draw(layered, gs_inv, 0.0f, 0.0f, 0.0f, false);
            expect_layers("layout(invocations = 4), gl_Layer = gl_InvocationID: layer k is painted by invocation k, and the fragment shader's gl_Layer is k (red = 64k + 63)",
                          {lr::Rgba8{63, 0, 0, 255}, lr::Rgba8{127, 0, 0, 255}, lr::Rgba8{191, 0, 0, 255}, lr::Rgba8{255, 0, 0, 255}});
            // 4. ReadPixels on a layered framebuffer reads layer 0.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, layered);
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            const demo::Rgba8 center = demo::read_pixel(4, 4);
            check.expect(center == demo::Rgba8{63, 0, 0, 255}, "glReadPixels from the layered framebuffer returns layer 0 (section 9.8: \"the image at layer zero ... is always used\")", demo::format_rgba(center));
            // 5. glClear on a layered framebuffer clears every layer.
            clear_all();
            expect_layers("glClearBufferfv on a layered framebuffer clears all four layers (section 9.8)", {black, black, black, black});
            // 6. A layer written by the geometry shader has no effect when the framebuffer is not layered.
            draw(single, gs_three, 0.0f, 0.0f, 1.0f);
            expect_layers("gl_Layer = 3 into a NON-layered framebuffer (layer 1 attached): painted layer 1, layer 3 untouched (section 9.8)", {black, lr::Rgba8{0, 0, 255, 255}, black, black});

            // 7a. "Defined results can be achieved if EndPrimitive is called between two vertices emitted with different layer
            //     numbers": two triangles from one invocation, layers 1 and 3.
            clear_all();
            draw(layered, gs_two, 1.0f, 1.0f, 0.0f);
            expect_layers("one invocation emits two triangles with gl_Layer 1 and 3 separated by EndPrimitive: exactly layers 1 and 3 are painted (section 9.8)",
                          {black, lr::Rgba8{255, 255, 0, 255}, black, lr::Rgba8{255, 255, 0, 255}});

            // 7b. The provoking vertex: one triangle strip whose vertices carry layers 0, 1, 2.
            const int provoking = demo::get_integer(mgl::GetParameter::LayerProvokingVertex);
            clear_all();
            draw(layered, gs_prov, 1.0f, 1.0f, 1.0f);
            const lr::Rgba8 white = {255, 255, 255, 255};
            if (provoking == GL_FIRST_VERTEX_CONVENTION)
                expect_layers("LAYER_PROVOKING_VERTEX = FIRST_VERTEX_CONVENTION: a triangle with vertex layers 0,1,2 lands in layer 0", {white, black, black, black});
            else if (provoking == GL_LAST_VERTEX_CONVENTION)
                expect_layers("LAYER_PROVOKING_VERTEX = LAST_VERTEX_CONVENTION: a triangle with vertex layers 0,1,2 lands in layer 2", {black, black, white, black});
            else
                std::printf("   note: LAYER_PROVOKING_VERTEX is UNDEFINED_VERTEX: the layer of a mixed-layer primitive is not defined, nothing asserted\n");
            std::printf("   note: LAYER_PROVOKING_VERTEX = %s\n", provoking == GL_FIRST_VERTEX_CONVENTION ? "FIRST_VERTEX_CONVENTION" : provoking == GL_LAST_VERTEX_CONVENTION ? "LAST_VERTEX_CONVENTION" : "UNDEFINED_VERTEX");

            // A layer outside the range (gl_Layer >= the number of layers) makes the effect on the framebuffer undefined
            // (section 9.8): deliberately not drawn or asserted anywhere in this demo.

            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(F, mgl::FramebufferId{0});
            for (const mgl::ProgramId p : {no_gs, gs_none, gs_inv, gs_three, gs_two, gs_prov}) mgl::glDeleteProgram(p);
            mgl::glDeleteFramebuffers(1, &layered);
            mgl::glDeleteFramebuffers(1, &single);
            check.expect_no_gl_error("layer semantics: no GL error left");
        }

        // =====================================================================================================
        // validate(): FRAMEBUFFER_DEFAULT_LAYERS (attachment-less framebuffer, ES 3.2 section 9.2.1)
        // =====================================================================================================
        void check_default_layers(demo::Check& check)
        {
            const auto F = mgl::FramebufferTarget::Framebuffer;
            mgl::FramebufferId fbo{};
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(F, fbo);
            GLint v = -1;
            mgl::glGetFramebufferParameteriv(F, mgl::FramebufferDefaultParameter::Layers, &v);
            check.expect_eq(v, 0, "FRAMEBUFFER_DEFAULT_LAYERS starts at 0 (not layered)");
            mgl::glFramebufferParameteri(F, mgl::FramebufferDefaultParameter::Width, 16);
            mgl::glFramebufferParameteri(F, mgl::FramebufferDefaultParameter::Height, 16);
            mgl::glFramebufferParameteri(F, mgl::FramebufferDefaultParameter::Layers, 5);
            mgl::glGetFramebufferParameteriv(F, mgl::FramebufferDefaultParameter::Layers, &v);
            check.expect_eq(v, 5, "glFramebufferParameteri(FRAMEBUFFER_DEFAULT_LAYERS, 5) reads back 5");
            expect_status(check, mgl::FramebufferStatus::Complete, "an attachment-less framebuffer with default width, height and layers is complete");
            demo::drain_gl_errors();
            mgl::glFramebufferParameteri(F, mgl::FramebufferDefaultParameter::Layers, demo::get_integer(mgl::GetParameter::MaxFramebufferLayers) + 1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_LAYERS > MAX_FRAMEBUFFER_LAYERS -> INVALID_VALUE");
            mgl::glFramebufferParameteri(F, mgl::FramebufferDefaultParameter::Layers, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "FRAMEBUFFER_DEFAULT_LAYERS -1 -> INVALID_VALUE");
            mgl::glGetFramebufferParameteriv(F, mgl::FramebufferDefaultParameter::Layers, &v);
            check.expect_eq(v, 5, "after the failed calls the value is still 5");
            mgl::glBindFramebuffer(F, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
        }

        // --- state -----------------------------------------------------------------------------------------------
        mgl::ProgramId prog_array_{}, prog_volume_{}, prog_env_{}, prog_sky_{}, prog_sphere_{}, prog_tile_cube_{}, prog_tile_array_{}, prog_tile_volume_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        std::array<mgl::FramebufferId, kFboCount> fbo_{};
        std::array<mgl::BufferId, kBufCount> buf_{};
        std::array<mgl::VertexArrayId, kVaoCount> vao_{};
        std::array<mgl::SamplerId, kSamplerCount> samplers_{};
        std::vector<lr::Quad> quads_;
        GLsizei env_vertex_count_ = 0;
        GLsizei sphere_index_count_ = 0;
        float time_ = kTime0;
        int cell_w_ = 128, cell_h_ = 128;
        CellRect cell_preview_{}, cell_net_{}, cell_array_{}, cell_volume_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-layered-rendering",
        .title = "Layered rendering: gl_Layer, cube map in one pass",
        .min_es = {3, 2},
        .summary = "glFramebufferTexture attaches whole 2D array / 3D / cube map textures; geometry shaders choose the layer with gl_Layer "
                   "(a room rendered into six cube faces in one pass and shown as a mirror ball, plus array and volume layers). Every layer is read "
                   "back and compared exactly; completeness rules, queries and the layer semantics of the specification are checked.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<LayeredRendering>(argc, argv, info);
}
