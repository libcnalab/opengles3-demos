// es31-compute-indirect-and-to-render
//
// GPU-driven work in OpenGL ES 3.1: compute shaders decide how much work follows, and the next stage consumes the
// decision straight from a buffer object, without the CPU ever reading it.
//
//   * glDispatchComputeIndirect (ES 3.2 section 17): the three group counts are read from the buffer bound to
//     DISPATCH_INDIRECT_BUFFER at a byte offset that must be a multiple of 4.
//       - a compaction pass counts the active elements of an array on the GPU and writes ceil(n / 64) as the group
//         count of the pass that processes them;
//       - a table of counts is dispatched from ten different byte offsets 0, 4, 8, ... (overlapping 12-byte
//         records, one with a zero count) and every launched group is accounted for;
//       - the group counts of an image fill are derived from imageSize() of the image by a first pass;
//       - the errors of the command: unaligned or negative offset (INVALID_VALUE), a command that would read beyond
//         the end of the buffer, no buffer bound, no program (INVALID_OPERATION), and that an erroring dispatch
//         launches nothing.
//   * Compute to render (ES 3.2 sections 10.3.9, 10.5, 7.11.2): a compute shader writes the vertices and indices of a
//     ring of triangles and the DrawElementsIndirectCommand (count from an atomic counter, firstIndex = 2,
//     baseVertex = 3) into buffers; after glMemoryBarrier(VERTEX_ATTRIB_ARRAY | ELEMENT_ARRAY | COMMAND) a single
//     glDrawElementsIndirect renders them. The same buffers are SSBOs for the compute shader, ARRAY_BUFFER,
//     ELEMENT_ARRAY_BUFFER and DRAW_INDIRECT_BUFFER for the draw.
//   * glMemoryBarrierByRegion (ES 3.2 section 7.11.2): only ATOMIC_COUNTER, FRAMEBUFFER, SHADER_IMAGE_ACCESS,
//     SHADER_STORAGE, TEXTURE_FETCH and UNIFORM are legal, because the command only orders FRAGMENT shader accesses
//     and only for a small framebuffer region (possibly one pixel). Three fragment-shader pipelines use it where each
//     pixel depends only on the same pixel of the previous pass: a chain of five passes over an r32ui image
//     (SHADER_IMAGE_ACCESS), an image store followed by a texelFetch (TEXTURE_FETCH) and a buffer store followed by a
//     buffer load (SHADER_STORAGE).
//
// Image (any window of at least 128 x 128; four cells, each half the window in both directions, GL coordinates):
//   top left      the compute-generated ring of triangles (5 of 8 sectors survive the compute culling test), drawn
//                 into the largest square that fits the cell, centered
//   top right     64 x 64 image painted by an indirectly dispatched 8 x 8 group grid (white corner = group origin),
//                 shown at the largest integer number of pixels per texel that fits the cell, from its bottom-left corner
//   bottom left   the five-pass r32ui image chain, shown as a color ramp with 8-pixel stripes
//   bottom right  upper strip: image store -> texelFetch ramp; lower strip: buffer store -> buffer load ramp
//
// Validation: all pixels of the three fragment pipelines and of the filled image against CPU models, probes of the
// star (and its culled sectors), the vertex / index / command buffers word by word, the work accounting of the indirect
// dispatches and the defined errors.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kMinWindow = 128;  // every cell at least 64 x 64: the whole 64 x 64 image fits at 1 pixel per texel

    /// The four cells, derived from the window size. Left / bottom cells are floor(size / 2) wide / high.
    struct Layout
    {
        int w = 0, h = 0;
        int cw = 0, ch = 0;  // width of the left cells, height of the bottom cells
        int rw = 0, th = 0;  // width of the right cells, height of the top cells
        int star_x = 0, star_y = 0, star_side = 0;  // square viewport of the star, centered in the top-left cell
        int img_scale = 1;                          // pixels per texel of the displayed 64 x 64 image
        static Layout of(int w, int h);
    };

    std::uint8_t c8(int v) { return static_cast<std::uint8_t>(std::clamp(v, 0, 255)); }
    Rgba8 color8(double r, double g, double b)
    {
        return {c8(static_cast<int>(std::lround(255.0 * r))), c8(static_cast<int>(std::lround(255.0 * g))), c8(static_cast<int>(std::lround(255.0 * b))), 255};
    }
    int max_channel_diff(const Rgba8& a, const Rgba8& b)
    {
        int d = 0;
        for (std::size_t i = 0; i < 4; ++i) d = std::max(d, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        return d;
    }
    std::uint32_t hash32(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    // ---------------------------------------------------------------------------------------------
    // CPU models
    // ---------------------------------------------------------------------------------------------
    constexpr int kSectors = 8;
    constexpr std::uint32_t kSeed = 2;
    constexpr double kTwoPi = 6.28318530717958647692;
    constexpr std::uint32_t kBaseVertex = 3;
    constexpr std::uint32_t kFirstIndex = 2;
    const Rgba8 kClear = color8(0.10, 0.10, 0.14);

    bool sector_visible(int s) { return ((static_cast<std::uint32_t>(s) * 7u + kSeed) % 5u) < 3u; }

    struct Triangle
    {
        std::array<std::array<double, 2>, 3> v;  // base0, tip, base1 (the order the compute shader writes them)
    };

    Triangle sector_triangle(int s)
    {
        const double a = kTwoPi * static_cast<double>(s) / 8.0;
        const double hw = 0.4 * kTwoPi / 8.0;
        return {{{{0.40 * std::cos(a - hw), 0.40 * std::sin(a - hw)}, {0.90 * std::cos(a), 0.90 * std::sin(a)}, {0.40 * std::cos(a + hw), 0.40 * std::sin(a + hw)}}}};
    }

    std::array<double, 2> centroid(const Triangle& t)
    {
        return {(t.v[0][0] + t.v[1][0] + t.v[2][0]) / 3.0, (t.v[0][1] + t.v[1][1] + t.v[2][1]) / 3.0};
    }

    Rgba8 sector_color(int s)
    {
        const auto u = static_cast<unsigned>(s);
        return color8(0.25 + 0.75 * static_cast<double>(u & 3u) / 3.0, 0.25 + 0.75 * static_cast<double>((u >> 1) & 3u) / 3.0, 0.25 + 0.75 * static_cast<double>((u >> 2) & 3u) / 3.0);
    }

    // Image fill of the indirectly dispatched pass: 64 x 64 texels, 8 x 8 groups.
    constexpr int kImg = 64;
    Rgba8 fill_color(int x, int y)
    {
        const int tx = x / 8, ty = y / 8, lx = x % 8, ly = y % 8;
        if (lx == 0 && ly == 0) return {255, 255, 255, 255};
        if (((tx + ty) & 1) == 0)
        {
            const double k = 0.4 + 0.6 * lx / 7.0;
            return color8(0.9 * k, 0.5 * k, 0.1 * k);
        }
        const double k = 0.4 + 0.6 * ly / 7.0;
        return color8(0.1 * k, 0.5 * k, 0.9 * k);
    }

    // Fragment pipelines, in window coordinates.
    std::uint32_t chain_value(int x, int y, bool all_passes)
    {
        const auto ux = static_cast<std::uint32_t>(x), uy = static_cast<std::uint32_t>(y);
        std::uint32_t v = ux * 3u + uy * 5u + 1u;
        if (all_passes)
            for (std::uint32_t k = 1; k <= 3; ++k) v += k * 16u + ((ux >> 3) & 1u) * k * 64u;
        return v;
    }
    Rgba8 chain_color(std::uint32_t v)
    {
        return color8(static_cast<double>(v % 1024u) / 1023.0, static_cast<double>((v / 4u) % 256u) / 255.0, static_cast<double>((v >> 3) & 1u) * 0.8 + 0.1);
    }
    Rgba8 tex_color(int x, int y)
    {
        const double t = static_cast<double>((static_cast<unsigned>(x) ^ static_cast<unsigned>(y)) & 127u) / 127.0;
        return color8(t, 1.0 - t, 0.5);
    }
    Rgba8 ssbo_color(int x, int y)
    {
        const std::uint32_t v = static_cast<std::uint32_t>(x) * 11u + static_cast<std::uint32_t>(y) * 17u + 3u;
        return color8(static_cast<double>(v % 256u) / 255.0, static_cast<double>(v / 256u) / 28.0, 0.5);
    }

    // The group-count table of write_records.comp and the layout of the probe buffer of record_probe.comp.
    constexpr std::array<std::uint32_t, 12> kTable = {3, 2, 2, 4, 1, 0, 1, 5, 1, 2, 1, 3};
    constexpr std::size_t kSlots = 16;
    constexpr std::size_t kProbeGroups = 0, kProbeDims = 16, kProbeVisits = 80, kProbeInvocations = 592, kProbeWords = 608;
    constexpr std::uint32_t kProbeLocalSize = 4;  // local_size_x of record_probe.comp
    constexpr std::size_t kRecWords = 16;         // the record buffer has 16 words = 64 bytes (words 12..15 hold 1)
    static_assert(kProbeInvocations + kSlots == kProbeWords);

    constexpr int kInputCount = 1000;
    constexpr std::uint32_t kCompactGroup = 64;

    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    Layout Layout::of(int w, int h)
    {
        Layout l;
        l.w = w;
        l.h = h;
        l.cw = w / 2;
        l.ch = h / 2;
        l.rw = w - l.cw;
        l.th = h - l.ch;
        l.star_side = std::min(l.cw, l.th);
        l.star_x = (l.cw - l.star_side) / 2;
        l.star_y = l.ch + (l.th - l.star_side) / 2;
        l.img_scale = std::max(1, std::min(l.rw, l.th) / kImg);
        return l;
    }

    class ComputeIndirect final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            max_frag_images_ = demo::get_integer(mgl::GetParameter::MaxFragmentImageUniforms);
            max_frag_ssbo_ = demo::get_integer(mgl::GetParameter::MaxFragmentShaderStorageBlocks);
            have_frag_images_ = max_frag_images_ > 0;
            have_frag_ssbo_ = max_frag_ssbo_ > 0;
            if (env.width < kMinWindow || env.height < kMinWindow)
                throw demo::Skip(std::format("needs a window of at least {0} x {0} pixels (each cell must hold the 64 x 64 image)", kMinWindow));
            if (env.width > demo::get_integer(mgl::GetParameter::MaxTextureSize))
                throw demo::Skip("the window is wider than MAX_TEXTURE_SIZE: the per-pixel images of the ByRegion pipelines cannot cover it");

            prog_gen_ = load_compute(env, "shaders/gen_geometry.comp");
            prog_finalize_ = load_compute(env, "shaders/finalize_command.comp");
            prog_img_setup_ = load_compute(env, "shaders/img_setup.comp");
            prog_img_fill_ = load_compute(env, "shaders/img_fill.comp");
            prog_records_ = load_compute(env, "shaders/write_records.comp");
            prog_probe_ = load_compute(env, "shaders/record_probe.comp");
            prog_compact_ = load_compute(env, "shaders/compact_and_count.comp");
            prog_process_ = load_compute(env, "shaders/process_active.comp");
            prog_star_ = build_graphics(env, "shaders/star.vert", "shaders/star.frag");
            prog_show_ = build_graphics(env, "shaders/fullscreen.vert", "shaders/show_image.frag");
            if (have_frag_images_)
            {
                prog_chain_ = build_graphics(env, "shaders/fullscreen.vert", "shaders/chain.frag");
                prog_tex_a_ = build_graphics(env, "shaders/fullscreen.vert", "shaders/tex_a.frag");
                prog_tex_b_ = build_graphics(env, "shaders/fullscreen.vert", "shaders/tex_b.frag");
            }
            if (have_frag_ssbo_)
            {
                prog_ssbo_a_ = build_graphics(env, "shaders/fullscreen.vert", "shaders/ssbo_a.frag");
                prog_ssbo_b_ = build_graphics(env, "shaders/fullscreen.vert", "shaders/ssbo_b.frag");
            }

            // ---- buffers --------------------------------------------------------------------------------------------
            // Geometry: [3 junk vertices][up to 8 triangles x 3 vertices], [2 junk indices][up to 8 x 3 indices], command + counter.
            const std::size_t vertex_bytes = (kBaseVertex + 3u * kSectors) * 16u;
            std::vector<float> junk_vertices(vertex_bytes / 4u, 0.0f);
            for (std::size_t i = 0; i < junk_vertices.size(); i += 4) junk_vertices[i] = junk_vertices[i + 1] = 7.0f;  // positions far outside the view
            geo_vbo_ = make_buffer(vertex_bytes, junk_vertices.data(), mgl::BufferUsage::DynamicCopy);
            const std::vector<std::uint32_t> zero_indices(kFirstIndex + 3u * kSectors, 0u);  // junk indices are valid (0) but would draw the wrong triangles
            geo_ibo_ = make_buffer(zero_indices.size() * 4u, zero_indices.data(), mgl::BufferUsage::DynamicCopy);
            const std::array<std::uint32_t, 8> zero_cmd{};
            geo_cmd_ = make_buffer(sizeof(zero_cmd), zero_cmd.data(), mgl::BufferUsage::DynamicCopy);

            // Compaction chain: 1000 input words, the work buffer (command + counters + list), the output (sentinel-filled).
            std::vector<std::uint32_t> input(kInputCount);
            for (int i = 0; i < kInputCount; ++i) input[static_cast<std::size_t>(i)] = hash32(static_cast<std::uint32_t>(i) + 1u);
            buf_in_ = make_buffer(input.size() * 4u, input.data(), mgl::BufferUsage::StaticDraw);
            const std::vector<std::uint32_t> work(6u + kInputCount, 0u);
            buf_work_ = make_buffer(work.size() * 4u, work.data(), mgl::BufferUsage::DynamicCopy);
            const std::vector<std::uint32_t> out(1024, kSentinel);
            buf_out_ = make_buffer(out.size() * 4u, out.data(), mgl::BufferUsage::DynamicCopy);

            // Table of counts (16 words; the tail is 1 so that every offset up to 52 is a valid, harmless command) and probe buffer.
            std::vector<std::uint32_t> rec(kRecWords, 1u);
            buf_rec_ = make_buffer(rec.size() * 4u, rec.data(), mgl::BufferUsage::DynamicCopy);
            const std::vector<std::uint32_t> probe(kProbeWords, 0u);
            buf_probe_ = make_buffer(probe.size() * 4u, probe.data(), mgl::BufferUsage::DynamicCopy);

            // Image fill: 64 x 64 RGBA8 texture and a 16-byte command buffer.
            const std::array<std::uint32_t, 4> cmd_init = {0, 0, 0, 0};
            buf_cmd_img_ = make_buffer(sizeof(cmd_init), cmd_init.data(), mgl::BufferUsage::DynamicCopy);
            tex_img_ = new_texture(mgl::SizedInternalFormat::Rgba8, kImg, kImg);

            // Fragment pipelines: per-pixel images and buffer, sized from the window.
            allocate_pixel_targets(env.width, env.height);

            // ---- vertex array of the star: attribute 0 = vec2 position, attribute 1 = uint sector (integer attribute) ----
            mgl::glGenVertexArrays(1, &vao_star_);
            mgl::glBindVertexArray(vao_star_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, geo_vbo_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
            mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 2, mgl::DataType::Float, GL_FALSE, 16, demo::buffer_offset(0));
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
            mgl::glVertexAttribIPointer(mgl::AttribLocation{1u}, 1, mgl::DataType::UnsignedInt, 16, demo::buffer_offset(8));
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, geo_ibo_);  // recorded in the VAO
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glGenVertexArrays(1, &vao_empty_);

            generate_star();
            fill_image();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            if (env.width != lay_.w || env.height != lay_.h) allocate_pixel_targets(env.width, env.height);  // interactive resize
            const Layout& l = lay_;
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glViewport(0, 0, l.w, l.h);
            mgl::glClearColor(0.10f, 0.10f, 0.14f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            // ---- top left: the indirect indexed draw ----------------------------------------------------------------
            mgl::glViewport(l.star_x, l.star_y, l.star_side, l.star_side);
            mgl::glUseProgram(prog_star_);
            mgl::glBindVertexArray(vao_star_);
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, geo_cmd_);
            mgl::glDrawElementsIndirect(mgl::PrimitiveType::Triangles, mgl::IndexType::UnsignedInt, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::DrawIndirect, mgl::BufferId{0});

            // ---- top right: the indirectly filled image --------------------------------------------------------------
            mgl::glBindVertexArray(vao_empty_);
            mgl::glViewport(l.cw, l.ch, kImg * l.img_scale, kImg * l.img_scale);  // exactly the displayed image: texelFetch stays in range
            mgl::glUseProgram(prog_show_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_img_);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_show_, "u_tex"), 0);
            mgl::glUniform2i(mgl::glGetUniformLocation(prog_show_, "u_origin"), l.cw, l.ch);
            mgl::glUniform1i(mgl::glGetUniformLocation(prog_show_, "u_scale"), l.img_scale);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            // ---- bottom left: five passes over one r32ui image, MemoryBarrierByRegion(SHADER_IMAGE_ACCESS) between them ----
            if (have_frag_images_)
            {
                mgl::glViewport(0, 0, l.cw, l.ch);
                mgl::glUseProgram(prog_chain_);
                mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_chain_, 0, GL_FALSE, 0, mgl::ImageAccess::ReadWrite, mgl::InternalFormat::R32UI);
                const mgl::UniformLocation pass = mgl::glGetUniformLocation(prog_chain_, "u_pass");
                for (int p = 0; p <= 4; ++p)
                {
                    mgl::glUniform1i(pass, p);
                    mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                    // Pass p+1 reads what pass p stored at the same pixel; after the last pass the barrier orders its loads
                    // before the stores of the next frame's pass 0.
                    mgl::glMemoryBarrierByRegion(mgl::MemoryBarrierByRegionMask::ShaderImageAccess);
                }

                // ---- bottom right, upper strip: image store -> texture fetch ------------------------------------------
                mgl::glViewport(l.cw, l.ch / 2, l.rw, l.ch - l.ch / 2);
                mgl::glUseProgram(prog_tex_a_);
                mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_tf_, 0, GL_FALSE, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::R32F);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                mgl::glMemoryBarrierByRegion(mgl::MemoryBarrierByRegionMask::TextureFetch);  // texture fetches after the barrier see the image stores
                mgl::glUseProgram(prog_tex_b_);
                mgl::glActiveTexture(mgl::TextureUnit::Texture0);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_tf_);
                mgl::glUniform1i(mgl::glGetUniformLocation(prog_tex_b_, "u_tex"), 0);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                mgl::glMemoryBarrierByRegion(mgl::MemoryBarrierByRegionMask::TextureFetch | mgl::MemoryBarrierByRegionMask::ShaderImageAccess);  // WAR for the next frame
            }

            // ---- bottom right, lower strip: buffer store -> buffer load in fragment shaders ---------------------------
            if (have_frag_ssbo_)
            {
                mgl::glViewport(l.cw, 0, l.rw, l.ch / 2);
                mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_frag_ssbo_);
                mgl::glUseProgram(prog_ssbo_a_);
                mgl::glUniform1ui(mgl::glGetUniformLocation(prog_ssbo_a_, "u_stride"), static_cast<GLuint>(l.w));
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                mgl::glMemoryBarrierByRegion(mgl::MemoryBarrierByRegionMask::ShaderStorage);
                mgl::glUseProgram(prog_ssbo_b_);
                mgl::glUniform1ui(mgl::glGetUniformLocation(prog_ssbo_b_, "u_stride"), static_cast<GLuint>(l.w));
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                mgl::glMemoryBarrierByRegion(mgl::MemoryBarrierByRegionMask::ShaderStorage);
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override;

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindImageTexture(mgl::ImageUnit{0}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            if (fbo_.value != 0) mgl::glDeleteFramebuffers(1, &fbo_);
            if (!textures_.empty()) mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
            if (!buffers_.empty()) mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            for (mgl::ProgramId p : programs_) mgl::glDeleteProgram(p);
            mgl::glDeleteVertexArrays(1, &vao_star_);
            mgl::glDeleteVertexArrays(1, &vao_empty_);
            free_pixel_targets();
            textures_.clear();
            buffers_.clear();
            programs_.clear();
        }

    private:
        static constexpr std::uint32_t kSentinel = 0xAAAAAAAAu;

        // ---- helpers ---------------------------------------------------------------------------------------------
        mgl::ProgramId load_compute(const demo::Env& env, std::string_view file)
        {
            std::string log;
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, env.load_text(file), log);
            if (cs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", file, log));
            const std::array<mgl::ShaderId, 1> shaders = {cs};
            const mgl::ProgramId program = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            if (program.value == 0) throw demo::Failure(std::format("{} failed to link:\n{}", file, log));
            programs_.push_back(program);
            return program;
        }

        mgl::ProgramId build_graphics(const demo::Env& env, std::string_view vert, std::string_view frag)
        {
            const mgl::ProgramId program = demo::build_program(env, vert, frag);
            programs_.push_back(program);
            return program;
        }

        /// A buffer with initial data. The generic COPY_WRITE target is used for the upload so no vertex array state is touched.
        mgl::BufferId make_buffer(std::size_t bytes, const void* data, mgl::BufferUsage usage)
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, b);
            mgl::glBufferData(mgl::BufferTarget::CopyWrite, static_cast<GLsizeiptr>(bytes), data, usage);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
            buffers_.push_back(b);
            return b;
        }

        mgl::TextureId new_texture(mgl::SizedInternalFormat format, int w, int h, bool track = true)
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, format, w, h);  // immutable: required by glBindImageTexture
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            if (track) textures_.push_back(t);
            return t;
        }

        /// Reads `words` 32-bit words of a buffer object after shader writes (BUFFER_UPDATE_BARRIER_BIT, ES 3.2 7.11.2).
        std::vector<std::uint32_t> read_buffer(mgl::BufferId buffer, std::size_t first_word, std::size_t words)
        {
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            std::vector<std::uint32_t> out(words, 0u);
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, buffer);
            const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::CopyRead, static_cast<GLintptr>(first_word * 4u), static_cast<GLsizeiptr>(words * 4u), mgl::MapBufferAccessMask::Read);
            if (mapped != nullptr)
            {
                std::memcpy(out.data(), mapped, words * 4u);
                (void)mgl::glUnmapBuffer(mgl::BufferTarget::CopyRead);
            }
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, mgl::BufferId{0});
            return out;
        }

        void write_buffer(mgl::BufferId buffer, const std::vector<std::uint32_t>& words)
        {
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, buffer);
            mgl::glBufferSubData(mgl::BufferTarget::CopyWrite, 0, std::span<const std::uint32_t>(words));
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
        }

        /// The images and the buffer of the ByRegion pipelines are addressed by window coordinates (gl_FragCoord), so they
        /// are sized from the window: the r32ui chain covers the bottom-left cell, the r32f image the width of the window
        /// and the bottom half (only the bottom-right upper strip is used), the buffer one word per pixel of the
        /// bottom-right lower strip, rows `w` words apart.
        void allocate_pixel_targets(int w, int h)
        {
            free_pixel_targets();
            lay_ = Layout::of(w, h);
            tex_chain_ = new_texture(mgl::SizedInternalFormat::R32UI, lay_.cw, lay_.ch, false);
            tex_tf_ = new_texture(mgl::SizedInternalFormat::R32F, lay_.w, lay_.ch, false);
            const std::vector<std::uint32_t> frag_words(static_cast<std::size_t>(lay_.w) * static_cast<std::size_t>(lay_.ch / 2), 0u);
            mgl::glGenBuffers(1, &buf_frag_ssbo_);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, buf_frag_ssbo_);
            mgl::glBufferData(mgl::BufferTarget::CopyWrite, static_cast<GLsizeiptr>(frag_words.size() * 4u), frag_words.data(), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
        }

        void free_pixel_targets()
        {
            if (tex_chain_.value != 0) mgl::glDeleteTextures(1, &tex_chain_);
            if (tex_tf_.value != 0) mgl::glDeleteTextures(1, &tex_tf_);
            if (buf_frag_ssbo_.value != 0) mgl::glDeleteBuffers(1, &buf_frag_ssbo_);
            tex_chain_ = mgl::TextureId{0};
            tex_tf_ = mgl::TextureId{0};
            buf_frag_ssbo_ = mgl::BufferId{0};
        }

        std::vector<std::uint8_t> read_texture(mgl::TextureId tex, int w, int h)
        {
            if (fbo_.value == 0) mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, tex, 0);
            std::vector<std::uint8_t> bytes;
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete)
                bytes = demo::read_rgba8(0, 0, w, h);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return bytes;
        }

        // ---- compute passes whose results are drawn ---------------------------------------------------------------
        void generate_star()
        {
            // Pass 1: every sector decides for itself and appends its triangle (SSBO bindings 0, 1, 2).
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, geo_vbo_);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, geo_ibo_);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 2, geo_cmd_);
            mgl::glUseProgram(prog_gen_);
            mgl::glUniform1ui(mgl::glGetUniformLocation(prog_gen_, "u_seed"), kSeed);
            mgl::glDispatchCompute(1, 1, 1);
            // Pass 2 reads the counter that pass 1 incremented: SHADER_STORAGE_BARRIER_BIT (accesses to shader buffer
            // variables after the barrier reflect the writes before it).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glUseProgram(prog_finalize_);
            mgl::glDispatchCompute(1, 1, 1);
            // The draw fetches vertices (ARRAY_BUFFER), indices (ELEMENT_ARRAY_BUFFER) and its command (DRAW_INDIRECT_BUFFER)
            // from buffers that shaders wrote: one barrier bit for each of the three kinds of fetch.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::VertexAttribArray | mgl::MemoryBarrierMask::ElementArray | mgl::MemoryBarrierMask::Command);
        }

        void fill_image()
        {
            // Pass 1: the group counts come from imageSize() of the image the next pass fills.
            mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_img_, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::Rgba8);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_cmd_img_);
            mgl::glUseProgram(prog_img_setup_);
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::Command);  // the indirect command is fetched after the barrier

            // Pass 2: indirect dispatch.
            mgl::glBindImageTexture(mgl::ImageUnit{0}, tex_img_, 0, GL_FALSE, 0, mgl::ImageAccess::WriteOnly, mgl::InternalFormat::Rgba8);
            mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, buf_cmd_img_);
            mgl::glUseProgram(prog_img_fill_);
            mgl::glDispatchComputeIndirect(0);
            mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, mgl::BufferId{0});
            // The texture is sampled by the display pass and read by glReadPixels.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::TextureFetch | mgl::MemoryBarrierMask::Framebuffer);
        }

        // ---- checks (defined below the class) ----------------------------------------------------------------------
        void check_star(demo::Check& check);
        void check_image_fill(demo::Check& check);
        void check_compaction_chain(demo::Check& check);
        void check_record_offsets(demo::Check& check);
        void check_dispatch_errors(demo::Check& check);
        void check_by_region(demo::Check& check);

        /// Runs record_probe.comp with glDispatchComputeIndirect(offset) from buf_rec_ into probe slot `slot`.
        void dispatch_record(GLintptr offset, std::uint32_t slot)
        {
            mgl::glUseProgram(prog_probe_);
            mgl::glUniform1ui(mgl::glGetUniformLocation(prog_probe_, "u_slot"), slot);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 2, buf_probe_);
            mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, buf_rec_);
            mgl::glDispatchComputeIndirect(offset);
        }

        int max_frag_images_ = 0, max_frag_ssbo_ = 0;
        bool have_frag_images_ = false, have_frag_ssbo_ = false;
        std::vector<mgl::ProgramId> programs_;
        std::vector<mgl::BufferId> buffers_;
        std::vector<mgl::TextureId> textures_;
        mgl::ProgramId prog_gen_{}, prog_finalize_{}, prog_img_setup_{}, prog_img_fill_{}, prog_records_{}, prog_probe_{}, prog_compact_{}, prog_process_{},
            prog_star_{}, prog_show_{}, prog_chain_{}, prog_tex_a_{}, prog_tex_b_{}, prog_ssbo_a_{}, prog_ssbo_b_{};
        mgl::BufferId geo_vbo_{}, geo_ibo_{}, geo_cmd_{}, buf_in_{}, buf_work_{}, buf_out_{}, buf_rec_{}, buf_probe_{}, buf_cmd_img_{}, buf_frag_ssbo_{};
        mgl::TextureId tex_img_{}, tex_chain_{}, tex_tf_{};
        mgl::VertexArrayId vao_star_{}, vao_empty_{};
        mgl::FramebufferId fbo_{};
        Layout lay_{};
    };

    // ---------------------------------------------------------------------------------------------------------
    // Checks
    // ---------------------------------------------------------------------------------------------------------
    struct Probe
    {
        int x, y;
        Rgba8 expected;
    };

    bool probe_set(demo::Check& check, std::string_view what, const std::vector<Probe>& probes, int tolerance)
    {
        int bad = 0;
        std::string first;
        for (const Probe& p : probes)
        {
            const Rgba8 got = demo::read_pixel(p.x, p.y);
            if (max_channel_diff(got, p.expected) > tolerance && bad++ == 0)
                first = std::format("pixel ({},{}) is {} expected {}", p.x, p.y, demo::format_rgba(got), demo::format_rgba(p.expected));
        }
        return check.expect(bad == 0, what, first);
    }

    /// Compares a window region (read from the default framebuffer, bottom-up) with a per-pixel model.
    template <typename Model>
    void compare_region(demo::Check& check, std::string_view what, int x0, int y0, int w, int h, int tolerance, Model&& model)
    {
        const std::vector<std::uint8_t> got = demo::read_rgba8(x0, y0, w, h);
        int bad = 0, max_diff = 0;
        std::string first;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const std::size_t o = static_cast<std::size_t>((y * w + x) * 4);
                const Rgba8 g = {got[o], got[o + 1], got[o + 2], got[o + 3]};
                const Rgba8 e = model(x0 + x, y0 + y);
                const int d = max_channel_diff(g, e);
                max_diff = std::max(max_diff, d);
                if (d > tolerance && bad++ == 0) first = std::format("pixel ({},{}) is {} expected {}", x0 + x, y0 + y, demo::format_rgba(g), demo::format_rgba(e));
            }
        std::cout << std::format("note: {}: max channel difference {} over {} pixels\n", what, max_diff, w * h);
        check.expect(bad == 0, std::format("{} (all {} pixels within {})", what, w * h, tolerance), first);
    }

    void ComputeIndirect::check_star(demo::Check& check)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        int visible = 0;
        for (int s = 0; s < kSectors; ++s) visible += sector_visible(s) ? 1 : 0;
        const auto tri_count = static_cast<std::uint32_t>(visible);

        // -- the three buffers the compute shader wrote --
        const std::vector<std::uint32_t> cmd = read_buffer(geo_cmd_, 0, 8);
        check.expect(cmd[0] == 3u * tri_count && cmd[1] == 1u && cmd[2] == kFirstIndex && cmd[3] == kBaseVertex && cmd[4] == 0u,
                     "the DrawElementsIndirectCommand written by the compute shader is {count 3 x triangles, instanceCount 1, firstIndex 2, baseVertex 3, 0}",
                     std::format("got {{{}, {}, {}, {}, {}}} expected {{{}, 1, 2, 3, 0}}", cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], 3u * tri_count));
        check.expect(cmd[5] == tri_count, std::format("the atomic counter in the command buffer counted {} surviving sectors", tri_count), std::format("counter {}", cmd[5]));

        const std::vector<std::uint32_t> idx = read_buffer(geo_ibo_, 0, kFirstIndex + 3u * kSectors);
        bool idx_ok = idx[0] == 0u && idx[1] == 0u;  // the junk entries were not touched
        for (std::uint32_t i = 0; i < 3u * tri_count; ++i) idx_ok = idx_ok && idx[kFirstIndex + i] == i;
        check.expect(idx_ok, "index buffer: entries [firstIndex, firstIndex + count) are 0, 1, 2, ... (relative to baseVertex), the junk entries before them are untouched");

        const std::vector<std::uint32_t> vtx = read_buffer(geo_vbo_, 0, (kBaseVertex + 3u * kSectors) * 4u);
        std::set<int> seen;
        double max_err = 0.0;
        bool sectors_ok = true;
        for (std::uint32_t slot = 0; slot < tri_count; ++slot)
        {
            const std::size_t base = (kBaseVertex + 3u * slot) * 4u;
            const std::uint32_t sector = vtx[base + 2];
            if (sector >= static_cast<std::uint32_t>(kSectors) || !sector_visible(static_cast<int>(sector)) || vtx[base + 6] != sector || vtx[base + 10] != sector)
            {
                sectors_ok = false;
                continue;
            }
            seen.insert(static_cast<int>(sector));
            const Triangle t = sector_triangle(static_cast<int>(sector));
            for (std::size_t j = 0; j < 3; ++j)
                for (std::size_t k = 0; k < 2; ++k) max_err = std::max(max_err, std::fabs(static_cast<double>(std::bit_cast<float>(vtx[base + 4 * j + k])) - t.v[j][k]));
        }
        std::cout << std::format("note: generated vertex positions differ from the CPU model by at most {:.2e} (sin/cos precision is implementation dependent)\n", max_err);
        std::set<int> expected_sectors;
        for (int s = 0; s < kSectors; ++s)
            if (sector_visible(s)) expected_sectors.insert(s);
        check.expect(sectors_ok && seen == expected_sectors, "vertex buffer: every surviving sector appears exactly once with all three vertices carrying its sector number");
        check.expect(max_err < 1e-3, std::format("vertex buffer: positions match the CPU model within 1e-3 NDC ({:.2f} px of this viewport; sin/cos accuracy is not specified)", 1e-3 * 0.5 * lay_.star_side),
                     std::format("max error {:.2e}", max_err));
        check.expect(vtx[0] == std::bit_cast<std::uint32_t>(7.0f) && vtx[4] == std::bit_cast<std::uint32_t>(7.0f), "vertex buffer: the junk vertices before baseVertex are untouched");

        // -- the rendered pixels: the square star viewport in the top-left cell --
        const Layout& l = lay_;
        auto pixel_of = [&](const std::array<double, 2>& ndc) {
            return std::array<int, 2>{l.star_x + static_cast<int>(std::floor((ndc[0] + 1.0) * 0.5 * l.star_side)),
                                      l.star_y + static_cast<int>(std::floor((ndc[1] + 1.0) * 0.5 * l.star_side))};
        };
        std::vector<Probe> shown, hidden;
        for (int s = 0; s < kSectors; ++s)
        {
            const std::array<int, 2> p = pixel_of(centroid(sector_triangle(s)));
            (sector_visible(s) ? shown : hidden).push_back({p[0], p[1], sector_visible(s) ? sector_color(s) : kClear});
        }
        probe_set(check, std::format("{} surviving sectors are drawn by glDrawElementsIndirect in their palette color at the triangle centroids", shown.size()), shown, 1);
        // Positions where culled sectors would have been, the star center, and the gap between two neighbouring sectors.
        const std::array<int, 2> hole = pixel_of({0.0, 0.0});
        hidden.push_back({hole[0], hole[1], kClear});
        const double mid = kTwoPi * 4.5 / 8.0;
        const std::array<int, 2> gap = pixel_of({0.65 * std::cos(mid), 0.65 * std::sin(mid)});
        hidden.push_back({gap[0], gap[1], kClear});
        probe_set(check, std::format("{} culled sectors / the center / a gap between two triangles show the clear color", hidden.size()), hidden, 1);

        // Negative control: the probe machinery can fail.
        const Probe& p0 = shown.front();
        check.expect(max_channel_diff(demo::read_pixel(p0.x, p0.y), kClear) > 20, "negative control: the first drawn centroid does NOT match the clear color (the probe can fail)");
    }

    void ComputeIndirect::check_image_fill(demo::Check& check)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        const std::vector<std::uint32_t> cmd = read_buffer(buf_cmd_img_, 0, 4);
        check.expect(cmd[0] == 8u && cmd[1] == 8u && cmd[2] == 1u, "img_setup.comp derived the group counts (8, 8, 1) from imageSize() of the 64 x 64 image", std::format("got ({}, {}, {})", cmd[0], cmd[1], cmd[2]));

        const std::vector<std::uint8_t> bytes = read_texture(tex_img_, kImg, kImg);
        int bad = 0, max_diff = 0;
        std::string first;
        for (int y = 0; y < kImg; ++y)
            for (int x = 0; x < kImg; ++x)
            {
                const std::size_t o = static_cast<std::size_t>((y * kImg + x) * 4);
                const Rgba8 g = bytes.size() >= o + 4 ? Rgba8{bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]} : Rgba8{0, 0, 0, 0};
                const Rgba8 e = fill_color(x, y);
                const int d = max_channel_diff(g, e);
                max_diff = std::max(max_diff, d);
                if (d > 1 && bad++ == 0) first = std::format("texel ({},{}) is {} expected {}", x, y, demo::format_rgba(g), demo::format_rgba(e));
            }
        std::cout << std::format("note: indirectly filled image: max channel difference {} over {} texels\n", max_diff, kImg * kImg);
        check.expect(bad == 0, "the image painted by glDispatchComputeIndirect(0) equals the CPU model at all 4096 texels (every texel written: the 8 x 8 groups covered it; tolerance 1 = 8-bit quantization)", first);

        // The same image as displayed in the top-right cell (lay_.img_scale pixels per texel); probe the middle of each texel.
        std::vector<Probe> probes;
        const int sc = lay_.img_scale;
        for (const std::array<int, 2>& t : std::array<std::array<int, 2>, 6>{{{0, 0}, {5, 3}, {9, 12}, {22, 40}, {63, 63}, {31, 8}}})
            probes.push_back({lay_.cw + t[0] * sc + sc / 2, lay_.ch + t[1] * sc + sc / 2, fill_color(t[0], t[1])});
        probe_set(check, "top-right cell shows that image through a sampler (6 probes: tile corners marked white, orange and blue tiles)", probes, 2);
    }

    void ComputeIndirect::check_compaction_chain(demo::Check& check)
    {
        std::vector<std::uint32_t> input(kInputCount), list;
        for (int i = 0; i < kInputCount; ++i)
        {
            input[static_cast<std::size_t>(i)] = hash32(static_cast<std::uint32_t>(i) + 1u);
            if ((input[static_cast<std::size_t>(i)] & 3u) != 0u) list.push_back(static_cast<std::uint32_t>(i));
        }
        const auto active = static_cast<std::uint32_t>(list.size());
        const std::uint32_t groups = (active + kCompactGroup - 1u) / kCompactGroup;

        // The chain: pass 1 -> barrier -> glDispatchComputeIndirect. Reset the work buffer and the output first.
        write_buffer(buf_work_, std::vector<std::uint32_t>(6u + kInputCount, 0u));
        write_buffer(buf_out_, std::vector<std::uint32_t>(1024, kSentinel));
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_in_);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 1, buf_work_);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 3, buf_out_);
        mgl::glUseProgram(prog_compact_);
        mgl::glDispatchCompute(1, 1, 1);
        // Pass 2 reads the list through an SSBO and its command through DISPATCH_INDIRECT_BUFFER: both barrier bits.
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage | mgl::MemoryBarrierMask::Command);
        mgl::glUseProgram(prog_process_);
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, buf_work_);  // the command is the first 12 bytes of the work buffer
        mgl::glDispatchComputeIndirect(0);
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, mgl::BufferId{0});

        const std::vector<std::uint32_t> work = read_buffer(buf_work_, 0, 6);
        check.expect(work[3] == active, std::format("compaction pass counted {} active elements of {}", active, kInputCount), std::format("got {}", work[3]));
        check.expect(work[0] == groups && work[1] == 1u && work[2] == 1u,
                     std::format("the GPU wrote the command ({}, 1, 1) = ceil({} / 64) for the next dispatch", groups, active), std::format("got ({}, {}, {})", work[0], work[1], work[2]));
        check.expect(work[4] == groups, std::format("glDispatchComputeIndirect launched exactly {} work groups", groups), std::format("counted {}", work[4]));
        check.expect(work[5] == groups * kCompactGroup, std::format("... and {} invocations (64 per group)", groups * kCompactGroup), std::format("counted {}", work[5]));

        const std::vector<std::uint32_t> out = read_buffer(buf_out_, 0, 1024);
        int bad = 0;
        std::string first;
        for (std::uint32_t g = 0; g < 1024; ++g)
        {
            const std::uint32_t e = g < active ? input[list[g]] * input[list[g]] + 1u : kSentinel;
            if (out[g] != e && bad++ == 0) first = std::format("out[{}] = {:#x} expected {:#x}", g, out[g], e);
        }
        check.expect(bad == 0, std::format("the {} active elements were squared in the order of the compacted list; the {} output words behind them keep their sentinel", active, 1024u - active), first);

        // Mutation: one group dispatched directly (what a stale zero/one command would do) leaves most elements unprocessed.
        write_buffer(buf_out_, std::vector<std::uint32_t>(1024, kSentinel));
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glDispatchCompute(1, 1, 1);
        const std::vector<std::uint32_t> out1 = read_buffer(buf_out_, 0, 1024);
        int bad1 = 0;
        for (std::uint32_t g = 0; g < 1024; ++g)
        {
            const std::uint32_t e = g < active ? input[list[g]] * input[list[g]] + 1u : kSentinel;
            if (out1[g] != e) ++bad1;
        }
        check.expect(active > kCompactGroup && bad1 > 0, "negative control: dispatching ONE group directly leaves elements unprocessed, so the output comparison above can fail",
                     std::format("{} words differ from the model", bad1));
    }

    void ComputeIndirect::check_record_offsets(demo::Check& check)
    {
        // write_records.comp fills words 0..11; words 12..15 are 1 (initial data). Word k..k+2 is a command at byte offset 4k.
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, buf_rec_);
        mgl::glUseProgram(prog_records_);
        mgl::glDispatchCompute(1, 1, 1);
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::Command);  // the records are fetched as dispatch commands next

        write_buffer(buf_probe_, std::vector<std::uint32_t>(kProbeWords, 0u));
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        constexpr std::uint32_t kRecordCount = 10;  // offsets 0, 4, ..., 36
        for (std::uint32_t k = 0; k < kRecordCount; ++k) dispatch_record(static_cast<GLintptr>(4u * k), k);
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, mgl::BufferId{0});
        const std::vector<std::uint32_t> probe = read_buffer(buf_probe_, 0, kProbeWords);

        for (std::uint32_t k = 0; k < kRecordCount; ++k)
        {
            const std::uint32_t gx = kTable[k], gy = kTable[k + 1], gz = kTable[k + 2];
            const std::uint32_t groups = gx * gy * gz;
            bool ok = probe[kProbeGroups + k] == groups && probe[kProbeInvocations + k] == groups * kProbeLocalSize;
            std::string detail = std::format("groups {} invocations {}", probe[kProbeGroups + k], probe[kProbeInvocations + k]);
            if (groups > 0) ok = ok && probe[kProbeDims + 4 * k] == gx && probe[kProbeDims + 4 * k + 1] == gy && probe[kProbeDims + 4 * k + 2] == gz;
            else ok = ok && probe[kProbeDims + 4 * k] == 0u;  // no group ran, so gl_NumWorkGroups was never recorded
            for (std::uint32_t g = 0; g < 32; ++g) ok = ok && probe[kProbeVisits + 32u * k + g] == (g < groups ? 1u : 0u);
            check.expect(ok,
                         std::format("glDispatchComputeIndirect(byte offset {}): words {}..{} = ({}, {}, {}) {}", 4u * k, k, k + 2, gx, gy, gz,
                                     groups == 0 ? "has a zero count: no group runs"
                                                 : std::format("launch {} groups, each group id exactly once, gl_NumWorkGroups = the record", groups)),
                         detail);
        }
    }

    void ComputeIndirect::check_dispatch_errors(demo::Check& check)
    {
        // Slot 15 counts what runs; every erroring call below must launch nothing (ES 3.2 section 2.3.1: an erroring command has no effect).
        const std::uint32_t slot = 15;
        mgl::glUseProgram(prog_probe_);
        mgl::glUniform1ui(mgl::glGetUniformLocation(prog_probe_, "u_slot"), slot);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 2, buf_probe_);
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, buf_rec_);
        constexpr GLintptr kSize = static_cast<GLintptr>(kRecWords * 4u);

        mgl::glDispatchComputeIndirect(2);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDispatchComputeIndirect(2): offset not a multiple of 4 -> INVALID_VALUE");
        mgl::glDispatchComputeIndirect(6);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDispatchComputeIndirect(6): offset not a multiple of 4 -> INVALID_VALUE");
        mgl::glDispatchComputeIndirect(-4);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glDispatchComputeIndirect(-4): negative offset -> INVALID_VALUE");
        mgl::glDispatchComputeIndirect(kSize - 8);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDispatchComputeIndirect(size - 8): the 12-byte command would extend beyond the end of the buffer -> INVALID_OPERATION");
        mgl::glDispatchComputeIndirect(kSize);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDispatchComputeIndirect(size): offset at the end of the buffer -> INVALID_OPERATION");
        mgl::glDispatchComputeIndirect(1 << 20);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDispatchComputeIndirect(1 MiB): offset far beyond the end of the 64-byte buffer -> INVALID_OPERATION");
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, mgl::BufferId{0});
        mgl::glDispatchComputeIndirect(0);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDispatchComputeIndirect with DISPATCH_INDIRECT_BUFFER = 0 -> INVALID_OPERATION (ES 3.2 section 17)");
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, buf_rec_);
        mgl::glUseProgram(mgl::ProgramId{0});
        mgl::glDispatchComputeIndirect(0);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glDispatchComputeIndirect with no current program -> INVALID_OPERATION");

        const std::vector<std::uint32_t> before = read_buffer(buf_probe_, kProbeInvocations + slot, 1);
        check.expect(before[0] == 0u, "none of the eight erroring dispatches launched an invocation (slot 15 is still 0)", std::format("invocations {}", before[0]));

        // The last valid offset: size - 12 = 52 reads words 13..15 = (1, 1, 1) -> one group of 4 invocations, no error.
        mgl::glUseProgram(prog_probe_);
        mgl::glDispatchComputeIndirect(kSize - 12);
        check.expect_no_gl_error("glDispatchComputeIndirect(size - 12): the last offset whose 12-byte command fits raises no error");
        mgl::glBindBuffer(mgl::BufferTarget::DispatchIndirect, mgl::BufferId{0});
        const std::vector<std::uint32_t> after = read_buffer(buf_probe_, 0, kProbeWords);
        check.expect(after[kProbeGroups + slot] == 1u && after[kProbeInvocations + slot] == kProbeLocalSize && after[kProbeDims + 4 * slot] == 1u,
                     "... and it launches exactly one group of 4 invocations with gl_NumWorkGroups = (1, 1, 1)",
                     std::format("groups {} invocations {}", after[kProbeGroups + slot], after[kProbeInvocations + slot]));

        // glMemoryBarrier(ALL) / glMemoryBarrierByRegion(ALL) are legal at any time.
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::AllBarrierBits);
        check.expect_no_gl_error("glMemoryBarrier(ALL_BARRIER_BITS) raises no error");
        mgl::glMemoryBarrierByRegion(mgl::MemoryBarrierByRegionMask::AllBarrierBits);
        check.expect_no_gl_error("glMemoryBarrierByRegion(ALL_BARRIER_BITS) raises no error");
    }

    void ComputeIndirect::check_by_region(demo::Check& check)
    {
        mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        const Layout& l = lay_;
        if (!have_frag_images_)
            std::cout << "note: MAX_FRAGMENT_IMAGE_UNIFORMS is 0 (allowed by ES 3.1): the image and texture-fetch ByRegion pipelines are skipped\n";
        if (!have_frag_ssbo_)
            std::cout << "note: MAX_FRAGMENT_SHADER_STORAGE_BLOCKS is 0 (allowed by ES 3.1): the buffer ByRegion pipeline is skipped\n";

        if (have_frag_images_)
        {
            compare_region(check, "bottom-left: five fragment passes over an r32ui image with glMemoryBarrierByRegion(SHADER_IMAGE_ACCESS) between them", 0, 0, l.cw, l.ch, 1,
                           [](int x, int y) { return chain_color(chain_value(x, y, true)); });
            // Negative control: without the three read-modify-write passes the picture would be different at (almost) every pixel.
            int differ = 0;
            for (int y = 0; y < l.ch; ++y)
                for (int x = 0; x < l.cw; ++x)
                    if (max_channel_diff(chain_color(chain_value(x, y, true)), chain_color(chain_value(x, y, false))) > 1) ++differ;
            check.expect(differ > l.cw * l.ch / 2, "negative control: the model of a chain that skipped passes 1..3 differs from the real model at more than half of the pixels",
                         std::format("{} of {} pixels differ", differ, l.cw * l.ch));
            compare_region(check, "bottom-right upper strip: imageStore (r32f) then texelFetch of the same pixel, glMemoryBarrierByRegion(TEXTURE_FETCH) between", l.cw, l.ch / 2, l.rw, l.ch - l.ch / 2, 1,
                           [](int x, int y) { return tex_color(x, y); });
        }
        if (have_frag_ssbo_)
            compare_region(check, "bottom-right lower strip: buffer store then buffer load of the same pixel in fragment shaders, glMemoryBarrierByRegion(SHADER_STORAGE) between", l.cw, 0, l.rw, l.ch / 2, 1,
                           [](int x, int y) { return ssbo_color(x, y); });
        check.expect_no_gl_error("no error from glMemoryBarrierByRegion with the legal bits ATOMIC_COUNTER, FRAMEBUFFER, SHADER_IMAGE_ACCESS, SHADER_STORAGE, TEXTURE_FETCH, UNIFORM");
        for (const auto bit : {mgl::MemoryBarrierByRegionMask::AtomicCounter, mgl::MemoryBarrierByRegionMask::Framebuffer, mgl::MemoryBarrierByRegionMask::ShaderImageAccess,
                               mgl::MemoryBarrierByRegionMask::ShaderStorage, mgl::MemoryBarrierByRegionMask::TextureFetch, mgl::MemoryBarrierByRegionMask::Uniform})
            mgl::glMemoryBarrierByRegion(bit);
        check.expect_no_gl_error("each of the six legal glMemoryBarrierByRegion bits is accepted on its own");
    }

    void ComputeIndirect::validate(demo::Env& /*env*/, demo::Check& check)
    {
        std::cout << std::format("note: MAX_FRAGMENT_IMAGE_UNIFORMS={} MAX_FRAGMENT_SHADER_STORAGE_BLOCKS={}\n", max_frag_images_, max_frag_ssbo_);
        // Everything that reads the default framebuffer goes first: the rest only dispatches compute shaders.
        check_star(check);
        check_image_fill(check);
        check_by_region(check);
        check_compaction_chain(check);
        check_record_offsets(check);
        check_dispatch_errors(check);
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-compute-indirect-and-to-render",
        .title = "Indirect dispatch and compute to render",
        .min_es = {3, 1},
        .summary = "GPU-driven work: glDispatchComputeIndirect with GPU-written group counts (offsets, zero counts, errors), compute-generated vertices, indices "
                   "and a DrawElementsIndirect command rendered after the matching memory barriers, and glMemoryBarrierByRegion between fragment-shader passes.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ComputeIndirect>(argc, argv, info);
}
