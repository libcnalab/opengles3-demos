// es30-texture-3d-and-arrays
//
// Three-dimensional textures and two-dimensional array textures (OpenGL ES 3.0, sections 3.8.3, 3.8.10.2
// and 4.4.2.4): glTexImage3D / glTexSubImage3D / glCopyTexSubImage3D, sampler3D and sampler2DArray with NEAREST
// and LINEAR filtering (including filtering ACROSS slices of a 3D texture, which does not happen between the
// layers of an array), the wrapping of the r coordinate, the rounding of the array layer coordinate, texelFetch,
// textureSize, UNPACK_IMAGE_HEIGHT / UNPACK_SKIP_IMAGES, and rendering into one slice or layer with
// glFramebufferTextureLayer.
//
// The window is a contact sheet of 8 x 8 panels (32 x 32 pixels each, panel 0 at the lower left, panels are
// numbered left to right and then upwards). Every panel is one draw call that shows one slice, layer or filtered
// lookup of one texture; validate() compares every panel pixel with a CPU model of the sampling equations.
//
//   panels  0- 7  3D texture, NEAREST / LINEAR, r wrap modes     8-15  2D array, layer rounding and clamping
//   panels 16-17  texelFetch (3D slice, array layer)             18-27 glTexSubImage3D / glCopyTexSubImage3D results
//   panels 28-35  UNPACK_IMAGE_HEIGHT and UNPACK_SKIP_IMAGES     36-41 textureSize at several levels
//   panels 42-45  texelFetch from level 0 and level 1            46-53 rendered through glFramebufferTextureLayer
//   panels 54-57  the largest allowed slice / layer index (MAX_3D_TEXTURE_SIZE, MAX_ARRAY_TEXTURE_LAYERS)

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using W = mgl::TextureWrapMode;

namespace
{
    constexpr int kGrid = 8;                       // 8 x 8 panels
    constexpr Rgba8 kGray{90, 90, 90, 255};        // filler for "this texel must not have been touched"
    constexpr int kTolLinear = 3;                  // see README "Tolerances"
    constexpr int kTolRendered = 1;                // 8-bit conversion of a computed color (+-1)
    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    // Panel numbers (index = 8 * row + column); build_panels() verifies them while it creates the panels.
    constexpr std::size_t kPanelFetch = 16, kPanelSubImage = 18, kPanelPixelStore = 28, kPanelSize = 36, kPanelFetchLod = 42, kPanelF3 = 46,
                          kPanelFa = 50, kPanelBig = 54, kPanelCount = 58;

    using Color = std::array<double, 4>;

    // ---------------------------------------------------------------------------------------------
    // CPU model: a texture level as an array of texels, in the order glTexImage3D consumes them.
    // ---------------------------------------------------------------------------------------------
    struct Volume
    {
        int w = 0, h = 0, d = 0;
        std::vector<Rgba8> texels;  // x fastest, then y, then z (ES 3.0 section 3.8.3: i = N mod w, j = ..., k = ...)

        [[nodiscard]] std::size_t index(int i, int j, int k) const
        {
            return (static_cast<std::size_t>(k) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)) * static_cast<std::size_t>(w) +
                   static_cast<std::size_t>(i);
        }
        [[nodiscard]] const Rgba8& at(int i, int j, int k) const { return texels[index(i, j, k)]; }
        Rgba8& at(int i, int j, int k) { return texels[index(i, j, k)]; }
        [[nodiscard]] const void* data() const { return texels.data(); }
    };

    template <typename F>
    Volume make_volume(int w, int h, int d, F&& texel)
    {
        Volume v;
        v.w = w;
        v.h = h;
        v.d = d;
        for (int k = 0; k < d; ++k)
            for (int j = 0; j < h; ++j)
                for (int i = 0; i < w; ++i) v.texels.push_back(texel(i, j, k));
        return v;
    }

    Volume filled(int w, int h, int d, Rgba8 color)
    {
        return make_volume(w, h, d, [&](int, int, int) { return color; });
    }

    std::uint8_t u8(int v) { return static_cast<std::uint8_t>(v); }

    // --- the texel data of every texture ------------------------------------------------------------
    // 4 x 4 x 4: every texel has its own color (i, j, k all visible in a channel).
    Rgba8 t3_texel(int i, int j, int k) { return {u8(40 + 60 * i), u8(40 + 60 * j), u8(40 + 60 * k), 255}; }
    // 4 x 4 x 4 array: the layers differ in blue and in how the red ramp is shifted.
    Rgba8 ta_texel(int i, int j, int l) { return {u8(40 + 60 * ((i + l) % 4)), u8(40 + 60 * j), u8(30 + 70 * l), 255}; }
    // textureSize / texelFetch probes: level 0 is 8 x 4 x 2 (array: 3 layers), level 1 is 4 x 2 x 1.
    Rgba8 s3_level0(int i, int j, int k) { return {u8(20 + 30 * i), u8(20 + 50 * j), u8(40 + 100 * k), 255}; }
    Rgba8 s3_level1(int i, int j, int k) { return {u8(230 - 50 * i), u8(40 + 120 * j), u8(200 - 30 * k), 255}; }
    Rgba8 sa_level0(int i, int j, int l) { return {u8(20 + 30 * i), u8(20 + 50 * j), u8(40 + 80 * l), 255}; }
    Rgba8 sa_level1(int i, int j, int l) { return {u8(230 - 50 * i), u8(40 + 120 * j), u8(60 + 70 * l), 255}; }

    // Client memory for the pixel-store tests: 4 "images" of 6 rows of 4 texels. Only rows 0-3 of every image
    // are meant as texels when UNPACK_IMAGE_HEIGHT = 6; the rows 4-5 are padding with distinct colors, so
    // reading them (or the wrong image) can never produce the expected pixels by accident.
    constexpr int kMemCols = 4, kMemImageRows = 6, kMemImages = 4;
    Rgba8 mem_texel(int row, int col) { return {u8(30 + 9 * row), u8(30 + 50 * col), u8(220 - 8 * row), 255}; }
    std::vector<Rgba8> make_memory()
    {
        std::vector<Rgba8> m;
        for (int row = 0; row < kMemImages * kMemImageRows; ++row)
            for (int col = 0; col < kMemCols; ++col) m.push_back(mem_texel(row, col));
        return m;
    }

    // ES 3.0 section 3.8.3: "If UNPACK_IMAGE_HEIGHT is zero, then the number of rows in each two-dimensional image is
    // height; otherwise the number of rows is UNPACK_IMAGE_HEIGHT" and "the pointer is advanced by
    // UNPACK_SKIP_IMAGES times the number of elements in one two-dimensional image". (Row length 0, alignment 4:
    // a row of 4 RGBA8 texels is 16 bytes, so there is no row padding.)
    Volume unpack_3d(const std::vector<Rgba8>& memory, int w, int h, int d, int skip_images, int image_height)
    {
        const int rows_per_image = image_height != 0 ? image_height : h;
        return make_volume(w, h, d, [&](int i, int j, int k) {
            return memory[static_cast<std::size_t>(((skip_images + k) * rows_per_image + j) * w + i)];
        });
    }

    // The pattern rendered into slices and layers (see pattern.frag): color = base + gx * x + gy * y.
    struct Pattern
    {
        std::array<double, 3> base, gx, gy;
    };
    Rgba8 pattern_texel(const Pattern& p, int x, int y, int size)
    {
        const double fx = (x + 0.5) / size, fy = (y + 0.5) / size;
        Rgba8 out{0, 0, 0, 255};
        for (std::size_t c = 0; c < 3; ++c)
            out[c] = u8(static_cast<int>(std::lround((p.base[c] + p.gx[c] * fx + p.gy[c] * fy) * 255.0)));
        return out;
    }
    constexpr int kFboSize = 16;
    constexpr Pattern kF3Slice1{{0.0, 0.0, 0.25}, {0.9, 0.0, 0.0}, {0.0, 0.9, 0.0}};
    constexpr Pattern kF3Slice3{{0.1, 0.2, 0.8}, {0.0, 0.7, 0.0}, {0.7, 0.0, 0.0}};
    constexpr Pattern kFaLayer0{{0.9, 0.1, 0.1}, {-0.8, 0.8, 0.0}, {0.0, 0.0, 0.8}};
    constexpr Pattern kFaLayer2{{0.05, 0.6, 0.1}, {0.9, -0.5, 0.0}, {0.0, 0.0, 0.85}};
    // The 8 x 8 framebuffer that glCopyTexSubImage3D reads from.
    constexpr int kSrcSize = 8;
    constexpr Pattern kSrcPattern{{0.1, 0.1, 0.2}, {0.8, 0.0, 0.1}, {0.0, 0.8, 0.5}};

    // --- CPU reference of ES 3.0 sections 3.8.10.2 (Table 3.22, equation 3.22) --------------------------
    Color to_color(Rgba8 t) { return {double(t[0]), double(t[1]), double(t[2]), double(t[3])}; }

    int wrap(int coord, int size, W mode)
    {
        auto fmod_i = [](int a, int b) { return ((a % b) + b) % b; };
        switch (mode)
        {
        case W::Repeat: return fmod_i(coord, size);
        case W::MirroredRepeat:
        {
            const int m = fmod_i(coord, 2 * size) - size;
            return (size - 1) - (m >= 0 ? m : -(1 + m));
        }
        default: return std::clamp(coord, 0, size - 1);  // CLAMP_TO_EDGE
        }
    }

    int ifloor(double x) { return static_cast<int>(std::floor(x)); }

    // sampler3D. s and t are always CLAMP_TO_EDGE in this demo, r uses `wrap_r`.
    Color sample_volume(const Volume& v, bool linear, W wrap_r, double s, double t, double r)
    {
        const double u = s * v.w, vv = t * v.h, w = r * v.d;
        auto at = [&](int i, int j, int k) {
            return to_color(v.at(wrap(i, v.w, W::ClampToEdge), wrap(j, v.h, W::ClampToEdge), wrap(k, v.d, wrap_r)));
        };
        if (!linear) return at(ifloor(u), ifloor(vv), ifloor(w));
        const int i0 = ifloor(u - 0.5), j0 = ifloor(vv - 0.5), k0 = ifloor(w - 0.5);
        const double a = (u - 0.5) - i0, b = (vv - 0.5) - j0, g = (w - 0.5) - k0;
        Color out{};
        for (int dk = 0; dk < 2; ++dk)
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di)
                {
                    const double weight = (di ? a : 1 - a) * (dj ? b : 1 - b) * (dk ? g : 1 - g);
                    const Color c = at(i0 + di, j0 + dj, k0 + dk);
                    for (std::size_t ch = 0; ch < 4; ++ch) out[ch] += weight * c[ch];
                }
        return out;
    }

    // sampler2DArray. NEAREST: l = clamp(RNE(r), 0, d-1); LINEAR: l = clamp(floor(r + 0.5), 0, d-1); the 2D filter
    // then reads texels of that ONE layer. (Section 3.8.10.2, including its footnote 3.)
    Color sample_array(const Volume& v, bool linear, double s, double t, double r)
    {
        const double rounded = linear ? std::floor(r + 0.5) : std::nearbyint(r);  // nearbyint: round half to even
        const int layer = std::clamp(static_cast<int>(rounded), 0, v.d - 1);
        const double u = s * v.w, vv = t * v.h;
        auto at = [&](int i, int j) {
            return to_color(v.at(wrap(i, v.w, W::ClampToEdge), wrap(j, v.h, W::ClampToEdge), layer));
        };
        if (!linear) return at(ifloor(u), ifloor(vv));
        const int i0 = ifloor(u - 0.5), j0 = ifloor(vv - 0.5);
        const double a = (u - 0.5) - i0, b = (vv - 0.5) - j0;
        Color out{};
        for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di)
            {
                const double weight = (di ? a : 1 - a) * (dj ? b : 1 - b);
                const Color c = at(i0 + di, j0 + dj);
                for (std::size_t ch = 0; ch < 4; ++ch) out[ch] += weight * c[ch];
            }
        return out;
    }

    bool near_edge(double texel_coord)
    {
        const double f = texel_coord - std::floor(texel_coord);
        return f < 0.1 || f > 0.9;
    }

    // ---------------------------------------------------------------------------------------------
    // Panels
    // ---------------------------------------------------------------------------------------------
    enum class Mode : int { Sample3D = 0, SampleArray = 1, Fetch3D = 2, FetchArray = 3, Size3D = 4, SizeArray = 5 };

    struct Panel
    {
        std::string name;
        Mode mode = Mode::Sample3D;
        mgl::TextureId tex{};  // bound to unit 0 (3D modes) or unit 1 (array modes)
        bool linear = false;
        W wrap_r = W::ClampToEdge;
        float r = 0.0f;
        int index = 0;  // slice / layer of the texelFetch modes
        int lod = 0;
        std::function<Color(double, double)> ref;  // expected color at texture coordinates (s, t) in (0, 1)
        int tol = 0;
        int boundary_w = 0, boundary_h = 0;  // NEAREST probes within 0.1 texel of a boundary of this grid are skipped
    };

    struct Probe
    {
        int probes = 0;
        int bad = 0;
        double max_dev = 0.0;
        std::string first;
    };

    enum : std::size_t
    {
        kT3, kTa, kT3Sub, kT3Copy, kTaCopy, kPxA, kPxB, kPxC, kPxArr, kPx2d, kSize3d, kSizeArr, kF3, kFa, kBigArr, kBig3d,
        kSrc, kClearCopy, kScratch3d, kScratchArr, kScratch2d, kTexCount
    };

    struct LayerJob
    {
        std::size_t tex;
        int layer;
        Pattern pattern;
    };
    constexpr std::array<LayerJob, 4> kLayerJobs{{{kF3, 1, kF3Slice1}, {kF3, 3, kF3Slice3}, {kFa, 0, kFaLayer0}, {kFa, 2, kFaLayer2}}};

    constexpr std::array<double, 4> kBigColorFirst{255, 0, 0, 255};
    constexpr std::array<double, 4> kBigColorLast{0, 255, 128, 255};

    class Texture3dAndArrays final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            sample_program_ = demo::build_program(env, "shaders/quad.vert", "shaders/sample.frag");
            pattern_program_ = demo::build_program(env, "shaders/quad.vert", "shaders/pattern.frag");

            // ES 3.0 Table 6.28: both limits are at least 256.
            max_3d_ = demo::get_integer(mgl::GetParameter::Max3DTextureSize);
            max_layers_ = demo::get_integer(mgl::GetParameter::MaxArrayTextureLayers);

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glGenFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            mgl::glDisable(mgl::Capability::Dither);

            build_main_textures();
            build_subimage_and_copy_textures();
            build_pixel_store_textures();
            build_size_textures();
            build_render_targets();
            build_panels();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            render_layers();

            // ---- the contact sheet ---------------------------------------------------------------
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.10f, 0.10f, 0.14f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            mgl::glUseProgram(sample_program_);
            const mgl::UniformLocation u_tex3d = mgl::glGetUniformLocation(sample_program_, "u_tex3d");
            const mgl::UniformLocation u_tex_array = mgl::glGetUniformLocation(sample_program_, "u_tex_array");
            const mgl::UniformLocation u_mode = mgl::glGetUniformLocation(sample_program_, "u_mode");
            const mgl::UniformLocation u_r = mgl::glGetUniformLocation(sample_program_, "u_r");
            const mgl::UniformLocation u_index = mgl::glGetUniformLocation(sample_program_, "u_index");
            const mgl::UniformLocation u_lod = mgl::glGetUniformLocation(sample_program_, "u_lod");
            mgl::glUniform1i(u_tex3d, 0);       // sampler uniforms hold texture UNIT indices
            mgl::glUniform1i(u_tex_array, 1);   // ... and the two sampler types need different units

            pw_ = std::min(env.width, env.height) / kGrid;
            for (std::size_t i = 0; i < panels_.size(); ++i)
            {
                const Panel& p = panels_[i];
                mgl::glViewport(static_cast<GLint>(i % kGrid) * pw_, static_cast<GLint>(i / kGrid) * pw_, pw_, pw_);
                bind_for(p);
                mgl::glUniform1i(u_mode, static_cast<int>(p.mode));
                mgl::glUniform1f(u_r, p.r);
                mgl::glUniform1i(u_index, p.index);
                mgl::glUniform1i(u_lod, p.lod);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }

            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glViewport(0, 0, env.width, env.height);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            width_ = env.width;
            frame_ = demo::read_rgba8(0, 0, env.width, env.height);

            check_limits(check);
            check_copy_source(check);
            check_panels(check);
            check_negative_controls(check);
            check_layer_readback(check);
            check_attachment_state(check);
            check_pixel_store(check);
            check_errors(check);
            check_framebuffer_completeness(check);
            check_copy_after_scissored_clears(check);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation sequence");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(static_cast<GLsizei>(fbo_.size()), fbo_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(pattern_program_);
            mgl::glDeleteProgram(sample_program_);
        }

    private:
        // ------------------------------------------------------------------------------------------
        // Texture construction (init)
        // ------------------------------------------------------------------------------------------
        void upload_3d(mgl::TextureTarget target, std::size_t slot, const Volume& v, int level = 0)
        {
            mgl::glBindTexture(target, tex_[slot]);
            mgl::glTexImage3D(target, level, mgl::InternalFormat::Rgba8, v.w, v.h, v.d, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, v.data());
        }

        void build_main_textures()
        {
            v_t3_ = make_volume(4, 4, 4, t3_texel);
            v_ta_ = make_volume(4, 4, 4, ta_texel);
            upload_3d(mgl::TextureTarget::Texture3D, kT3, v_t3_);
            upload_3d(mgl::TextureTarget::Texture2DArray, kTa, v_ta_);
        }

        void build_subimage_and_copy_textures()
        {
            // --- glTexSubImage3D: gray 4x4x4, a 2x2x2 box at (1,1,1), and the whole slice z = 3 -----------------
            v_t3sub_ = filled(4, 4, 4, kGray);
            upload_3d(mgl::TextureTarget::Texture3D, kT3Sub, v_t3sub_);
            const Volume box = make_volume(2, 2, 2, [](int i, int j, int k) { return Rgba8{u8(60 + 100 * i), u8(60 + 100 * j), u8(60 + 100 * k), 255}; });
            mgl::glTexSubImage3D(mgl::TextureTarget::Texture3D, 0, 1, 1, 1, 2, 2, 2, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, box.data());
            const Volume slice3 = make_volume(4, 4, 1, [](int i, int j, int) { return (i + j) % 2 != 0 ? Rgba8{230, 200, 40, 255} : Rgba8{40, 120, 200, 255}; });
            mgl::glTexSubImage3D(mgl::TextureTarget::Texture3D, 0, 0, 0, 3, 4, 4, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, slice3.data());
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i) v_t3sub_.at(1 + i, 1 + j, 1 + k) = box.at(i, j, k);
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 4; ++i) v_t3sub_.at(i, j, 3) = slice3.at(i, j, 0);

            // --- glCopyTexSubImage3D: the source is the color buffer of a framebuffer object --------------------
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kSrc]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kSrcSize, kSrcSize, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, nullptr);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[1]);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, tex_[kSrc], 0);
            require_complete("copy source framebuffer");
            mgl::glViewport(0, 0, kSrcSize, kSrcSize);
            mgl::glUseProgram(pattern_program_);
            draw_pattern(kSrcPattern, kSrcSize);
            // What the copy must reproduce is the framebuffer as it is: read it back (bit exact), so the copy checks
            // below do not depend on how the shader rounded its output.
            src_ = make_volume(kSrcSize, kSrcSize, 1, [](int, int, int) { return Rgba8{}; });
            {
                const auto bytes = demo::read_rgba8(0, 0, kSrcSize, kSrcSize);
                for (std::size_t i = 0; i < src_.texels.size(); ++i) src_.texels[i] = {bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]};
            }

            v_t3copy_ = filled(4, 4, 3, kGray);
            upload_3d(mgl::TextureTarget::Texture3D, kT3Copy, v_t3copy_);
            v_tacopy_ = filled(4, 4, 3, kGray);
            upload_3d(mgl::TextureTarget::Texture2DArray, kTaCopy, v_tacopy_);

            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, fbo_[1]);
            // 4 x 3 texels of the framebuffer, lower left corner (2, 4), into the xy rectangle at (0, 1) of slice 2.
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_[kT3Copy]);
            mgl::glCopyTexSubImage3D(mgl::TextureTarget::Texture3D, 0, 0, 1, 2, 2, 4, 4, 3);
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 4; ++i) v_t3copy_.at(i, 1 + j, 2) = src_.at(2 + i, 4 + j, 0);
            // 3 x 4 texels, lower left (0, 1), into layer 1 of the array at (1, 0).
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kTaCopy]);
            mgl::glCopyTexSubImage3D(mgl::TextureTarget::Texture2DArray, 0, 1, 0, 1, 0, 1, 3, 4);
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 3; ++i) v_tacopy_.at(1 + i, j, 1) = src_.at(i, 1 + j, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::ReadFramebuffer, mgl::FramebufferId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void build_pixel_store_textures()
        {
            memory_ = make_memory();
            initial_image_height_ = demo::get_integer(mgl::GetParameter::UnpackImageHeight);
            initial_skip_images_ = demo::get_integer(mgl::GetParameter::UnpackSkipImages);

            // A: IMAGE_HEIGHT 6 (each image has 6 rows in memory), SKIP_IMAGES 1: memory images 1 and 2.
            set_unpack(6, 1);
            seen_image_height_ = demo::get_integer(mgl::GetParameter::UnpackImageHeight);
            seen_skip_images_ = demo::get_integer(mgl::GetParameter::UnpackSkipImages);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_[kPxA]);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 0, mgl::InternalFormat::Rgba8, 4, 4, 2, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, memory_.data());
            v_pxa_ = unpack_3d(memory_, 4, 4, 2, 1, 6);

            // B: the same memory and SKIP_IMAGES, but IMAGE_HEIGHT 0 (rows per image = height = 4): a different result.
            set_unpack(0, 1);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_[kPxB]);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 0, mgl::InternalFormat::Rgba8, 4, 4, 2, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, memory_.data());
            v_pxb_ = unpack_3d(memory_, 4, 4, 2, 1, 0);

            // C: the same as A but through glTexSubImage3D into an existing gray texture.
            set_unpack(0, 0);
            v_pxc_ = filled(4, 4, 2, kGray);
            upload_3d(mgl::TextureTarget::Texture3D, kPxC, v_pxc_);
            set_unpack(6, 1);
            mgl::glTexSubImage3D(mgl::TextureTarget::Texture3D, 0, 0, 0, 0, 4, 4, 2, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte,
                                 memory_.data());
            v_pxc_ = unpack_3d(memory_, 4, 4, 2, 1, 6);

            // Array: TEXTURE_2D_ARRAY takes the same path (images 2 and 3).
            set_unpack(6, 2);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kPxArr]);
            mgl::glTexImage3D(mgl::TextureTarget::Texture2DArray, 0, mgl::InternalFormat::Rgba8, 4, 4, 2, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, memory_.data());
            v_pxarr_ = unpack_3d(memory_, 4, 4, 2, 2, 6);

            // glTexImage2D ignores UNPACK_SKIP_IMAGES (section 3.8.3: "equivalent to TexImage3D with a depth of 1,
            // except that UNPACK_SKIP_IMAGES is ignored"): the 2D texture holds the FIRST four rows of memory.
            set_unpack(6, 1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kPx2d]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 4, 4, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, memory_.data());
            set_unpack(0, 0);
            v_px2d_ = unpack_3d(memory_, 4, 4, 1, 0, 0);
        }

        static void set_unpack(int image_height, int skip_images)
        {
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, image_height);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, skip_images);
        }

        void build_size_textures()
        {
            // 3D: 8x4x2, 4x2x1, 2x1x1, 1x1x1. Level 0 and 1 carry data, levels 2 and 3 only exist (textureSize).
            v_s3_[0] = make_volume(8, 4, 2, s3_level0);
            v_s3_[1] = make_volume(4, 2, 1, s3_level1);
            upload_3d(mgl::TextureTarget::Texture3D, kSize3d, v_s3_[0], 0);
            upload_3d(mgl::TextureTarget::Texture3D, kSize3d, v_s3_[1], 1);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 2, mgl::InternalFormat::Rgba8, 2, 1, 1, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 3, mgl::InternalFormat::Rgba8, 1, 1, 1, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            // Array: the number of layers does not shrink with the level.
            v_sa_[0] = make_volume(8, 4, 3, sa_level0);
            v_sa_[1] = make_volume(4, 2, 3, sa_level1);
            upload_3d(mgl::TextureTarget::Texture2DArray, kSizeArr, v_sa_[0], 0);
            upload_3d(mgl::TextureTarget::Texture2DArray, kSizeArr, v_sa_[1], 1);
            mgl::glTexImage3D(mgl::TextureTarget::Texture2DArray, 2, mgl::InternalFormat::Rgba8, 2, 1, 3, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
            mgl::glTexImage3D(mgl::TextureTarget::Texture2DArray, 3, mgl::InternalFormat::Rgba8, 1, 1, 3, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, nullptr);
        }

        void build_render_targets()
        {
            // 16x16x4 targets, filled with gray so that "this slice was not rendered" is observable.
            v_f3_ = filled(kFboSize, kFboSize, 4, kGray);
            v_fa_ = filled(kFboSize, kFboSize, 4, kGray);
            upload_3d(mgl::TextureTarget::Texture3D, kF3, v_f3_);
            upload_3d(mgl::TextureTarget::Texture2DArray, kFa, v_fa_);
            for (const LayerJob& job : kLayerJobs)
            {
                Volume& v = job.tex == kF3 ? v_f3_ : v_fa_;
                for (int y = 0; y < kFboSize; ++y)
                    for (int x = 0; x < kFboSize; ++x) v.at(x, y, job.layer) = pattern_texel(job.pattern, x, y, kFboSize);
            }

            // The extreme slice / layer index: 1 x 1 x N textures with N = the implementation's limit. Allocated
            // without data; only layer 0 and layer N-1 are ever written (cleared through glFramebufferTextureLayer).
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kBigArr]);
            mgl::glTexImage3D(mgl::TextureTarget::Texture2DArray, 0, mgl::InternalFormat::Rgba8, 1, 1, max_layers_, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, nullptr);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_[kBig3d]);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 0, mgl::InternalFormat::Rgba8, 1, 1, max_3d_, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, nullptr);

            // Scratch objects for the error checks (validate() must not damage the textures that were drawn).
            upload_3d(mgl::TextureTarget::Texture3D, kScratch3d, filled(4, 4, 4, kGray));
            upload_3d(mgl::TextureTarget::Texture2DArray, kScratchArr, filled(4, 4, 4, kGray));
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kScratch2d]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 4, 4, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, nullptr);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        }

        void require_complete(const char* what) const
        {
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("{} is incomplete: {}", what, demo::framebuffer_status_name(status)));
        }

        // ------------------------------------------------------------------------------------------
        // Panels
        // ------------------------------------------------------------------------------------------
        void add_sample_3d(std::string name, std::size_t slot, const Volume& v, bool linear, W wrap_r, float r)
        {
            Panel p;
            p.name = std::move(name);
            p.mode = Mode::Sample3D;
            p.tex = tex_[slot];
            p.linear = linear;
            p.wrap_r = wrap_r;
            p.r = r;
            p.ref = [&v, linear, wrap_r, r](double s, double t) { return sample_volume(v, linear, wrap_r, s, t, r); };
            p.tol = linear ? kTolLinear : 0;
            if (!linear) p.boundary_w = v.w, p.boundary_h = v.h;
            panels_.push_back(std::move(p));
        }

        // Slice k of a 3D texture sampled with NEAREST at the center of the slice (r = (k + 0.5) / depth).
        void add_slice(std::string name, std::size_t slot, const Volume& v, int k, int tol = 0)
        {
            add_sample_3d(std::move(name), slot, v, false, W::ClampToEdge, (static_cast<float>(k) + 0.5f) / static_cast<float>(v.d));
            panels_.back().tol = tol;
        }

        void add_sample_array(std::string name, std::size_t slot, const Volume& v, bool linear, float r)
        {
            Panel p;
            p.name = std::move(name);
            p.mode = Mode::SampleArray;
            p.tex = tex_[slot];
            p.linear = linear;
            p.r = r;
            p.ref = [&v, linear, r](double s, double t) { return sample_array(v, linear, s, t, r); };
            p.tol = linear ? kTolLinear : 0;
            if (!linear) p.boundary_w = v.w, p.boundary_h = v.h;
            panels_.push_back(std::move(p));
        }

        void add_layer(std::string name, std::size_t slot, const Volume& v, int layer, int tol = 0)
        {
            add_sample_array(std::move(name), slot, v, false, static_cast<float>(layer));
            panels_.back().tol = tol;
        }

        void add_fetch(std::string name, Mode mode, std::size_t slot, const Volume& v, int index, int lod, bool linear_state)
        {
            Panel p;
            p.name = std::move(name);
            p.mode = mode;
            p.tex = tex_[slot];
            p.linear = linear_state;  // texelFetch ignores filtering: the texture is set to LINEAR on purpose
            p.index = index;
            p.lod = lod;
            p.ref = [&v, index](double s, double t) {
                return to_color(v.at(std::min(ifloor(s * v.w), v.w - 1), std::min(ifloor(t * v.h), v.h - 1), index));
            };
            panels_.push_back(std::move(p));
        }

        void add_size(std::string name, Mode mode, std::size_t slot, std::array<int, 3> size, int lod)
        {
            Panel p;
            p.name = std::move(name);
            p.mode = mode;
            p.tex = tex_[slot];
            p.lod = lod;
            const Color expected{double(size[0] * 20), double(size[1] * 20), double(size[2] * 20), 255.0};
            p.ref = [expected](double, double) { return expected; };
            panels_.push_back(std::move(p));
        }

        void add_uniform(std::string name, Mode mode, std::size_t slot, float r, Color expected)
        {
            Panel p;
            p.name = std::move(name);
            p.mode = mode;
            p.tex = tex_[slot];
            p.r = r;
            p.ref = [expected](double, double) { return expected; };
            panels_.push_back(std::move(p));
        }

        void build_panels()
        {
            // ---- 3D texture T3 (4x4x4): panels 0-7 ----------------------------------------------------
            add_slice("3D NEAREST, r = 0.375 selects slice 1", kT3, v_t3_, 1);
            add_slice("3D NEAREST, r = 0.875 selects slice 3", kT3, v_t3_, 3);
            add_sample_3d("3D LINEAR, r = 0.25 lies between slices 0 and 1 (50% each)", kT3, v_t3_, true, W::ClampToEdge, 0.25f);
            add_sample_3d("3D LINEAR, r = 0.3 (70% slice 1, 30% slice 0)", kT3, v_t3_, true, W::ClampToEdge, 0.3f);
            add_sample_3d("3D NEAREST, r = 1.375 with WRAP_R = REPEAT selects slice 1", kT3, v_t3_, false, W::Repeat, 1.375f);
            add_sample_3d("3D NEAREST, r = 1.375 with WRAP_R = MIRRORED_REPEAT selects slice 2", kT3, v_t3_, false, W::MirroredRepeat, 1.375f);
            add_sample_3d("3D NEAREST, r = 1.375 with WRAP_R = CLAMP_TO_EDGE selects slice 3", kT3, v_t3_, false, W::ClampToEdge, 1.375f);
            add_sample_3d("3D LINEAR, r = -0.05 with REPEAT blends slice 3 (70%) and slice 0 (30%)", kT3, v_t3_, true, W::Repeat, -0.05f);

            // ---- 2D array TA (4x4, 4 layers): panels 8-15 ---------------------------------------------
            add_sample_array("array NEAREST, layer coordinate 2.0 selects layer 2", kTa, v_ta_, false, 2.0f);
            add_sample_array("array LINEAR, layer coordinate 1.3 selects layer 1 (no blending)", kTa, v_ta_, true, 1.3f);
            add_sample_array("array NEAREST, layer coordinate 1.7 rounds to layer 2", kTa, v_ta_, false, 1.7f);
            add_sample_array("array LINEAR, layer coordinate 2.6 selects layer 3", kTa, v_ta_, true, 2.6f);
            add_sample_array("array NEAREST, layer coordinate -3.2 clamps to layer 0", kTa, v_ta_, false, -3.2f);
            add_sample_array("array NEAREST, layer coordinate 9.6 clamps to layer 3", kTa, v_ta_, false, 9.6f);
            add_sample_array("array LINEAR, layer coordinate 1.4 selects layer 1", kTa, v_ta_, true, 1.4f);
            add_sample_array("array LINEAR, layer coordinate 1.6 selects layer 2", kTa, v_ta_, true, 1.6f);

            // ---- texelFetch (the textures are LINEAR on purpose): panels 16-17 ---------------------------
            expect_next(kPanelFetch);
            add_fetch("texelFetch(sampler3D) slice 2 ignores LINEAR filtering", Mode::Fetch3D, kT3, v_t3_, 2, 0, true);
            add_fetch("texelFetch(sampler2DArray) layer 3 ignores LINEAR filtering", Mode::FetchArray, kTa, v_ta_, 3, 0, true);

            // ---- glTexSubImage3D and glCopyTexSubImage3D results: panels 18-27 --------------------------
            expect_next(kPanelSubImage);
            for (int k = 0; k < 4; ++k)
                add_slice(std::format("glTexSubImage3D result, slice {}", k), kT3Sub, v_t3sub_, k);
            for (int k = 0; k < 3; ++k)
                add_slice(std::format("glCopyTexSubImage3D into a 3D texture, slice {}{}", k, k == 2 ? " (copied)" : " (untouched gray)"), kT3Copy, v_t3copy_, k);
            for (int l = 0; l < 3; ++l)
                add_layer(std::format("glCopyTexSubImage3D into an array, layer {}{}", l, l == 1 ? " (copied)" : " (untouched gray)"), kTaCopy, v_tacopy_, l);

            // ---- pixel store: panels 28-35 ----------------------------------------------------------------
            expect_next(kPanelPixelStore);
            for (int k = 0; k < 2; ++k) add_slice(std::format("UNPACK_IMAGE_HEIGHT 6 + SKIP_IMAGES 1, glTexImage3D slice {}", k), kPxA, v_pxa_, k);
            for (int k = 0; k < 2; ++k) add_slice(std::format("UNPACK_IMAGE_HEIGHT 0 + SKIP_IMAGES 1, glTexImage3D slice {}", k), kPxB, v_pxb_, k);
            for (int k = 0; k < 2; ++k) add_slice(std::format("UNPACK_IMAGE_HEIGHT 6 + SKIP_IMAGES 1, glTexSubImage3D slice {}", k), kPxC, v_pxc_, k);
            for (int l = 0; l < 2; ++l) add_layer(std::format("UNPACK_IMAGE_HEIGHT 6 + SKIP_IMAGES 2, array layer {}", l), kPxArr, v_pxarr_, l);

            // ---- textureSize: panels 36-41 -----------------------------------------------------------------
            expect_next(kPanelSize);
            add_size("textureSize(sampler3D, 0) = (8, 4, 2)", Mode::Size3D, kSize3d, {8, 4, 2}, 0);
            add_size("textureSize(sampler3D, 1) = (4, 2, 1)", Mode::Size3D, kSize3d, {4, 2, 1}, 1);
            add_size("textureSize(sampler3D, 2) = (2, 1, 1)", Mode::Size3D, kSize3d, {2, 1, 1}, 2);
            add_size("textureSize(sampler2DArray, 0) = (8, 4, 3 layers)", Mode::SizeArray, kSizeArr, {8, 4, 3}, 0);
            add_size("textureSize(sampler2DArray, 1) = (4, 2, 3 layers)", Mode::SizeArray, kSizeArr, {4, 2, 3}, 1);
            add_size("textureSize(sampler2DArray, 3) = (1, 1, 3 layers)", Mode::SizeArray, kSizeArr, {1, 1, 3}, 3);

            // ---- texelFetch with a level of detail: panels 42-45 -------------------------------------------
            expect_next(kPanelFetchLod);
            add_fetch("texelFetch(sampler3D, level 0) slice 1", Mode::Fetch3D, kSize3d, v_s3_[0], 1, 0, false);
            add_fetch("texelFetch(sampler3D, level 1) slice 0", Mode::Fetch3D, kSize3d, v_s3_[1], 0, 1, false);
            add_fetch("texelFetch(sampler2DArray, level 0) layer 2", Mode::FetchArray, kSizeArr, v_sa_[0], 2, 0, false);
            add_fetch("texelFetch(sampler2DArray, level 1) layer 1", Mode::FetchArray, kSizeArr, v_sa_[1], 1, 1, false);

            // ---- rendered through glFramebufferTextureLayer: panels 46-53 ---------------------------------
            expect_next(kPanelF3);
            for (int k = 0; k < 4; ++k)
                add_slice(std::format("3D render target, slice {} ({})", k, k == 1 || k == 3 ? "rendered" : "never attached: gray"), kF3, v_f3_, k, kTolRendered);
            expect_next(kPanelFa);
            for (int l = 0; l < 4; ++l)
                add_layer(std::format("array render target, layer {} ({})", l, l == 0 || l == 2 ? "rendered" : "never attached: gray"), kFa, v_fa_, l, kTolRendered);

            // ---- the largest slice / layer index: panels 54-57 --------------------------------------------
            expect_next(kPanelBig);
            add_uniform("array layer 0 of the 1x1xMAX_ARRAY_TEXTURE_LAYERS texture", Mode::SampleArray, kBigArr, 0.0f, kBigColorFirst);
            add_uniform("array layer MAX_ARRAY_TEXTURE_LAYERS-1", Mode::SampleArray, kBigArr, static_cast<float>(max_layers_ - 1), kBigColorLast);
            add_uniform("3D slice 0 of the 1x1xMAX_3D_TEXTURE_SIZE texture", Mode::Sample3D, kBig3d, 0.5f / static_cast<float>(max_3d_), kBigColorFirst);
            add_uniform("3D slice MAX_3D_TEXTURE_SIZE-1", Mode::Sample3D, kBig3d, (static_cast<float>(max_3d_) - 0.5f) / static_cast<float>(max_3d_), kBigColorLast);
            expect_next(kPanelCount);
        }

        void expect_next(std::size_t index) const
        {
            if (panels_.size() != index) throw demo::Failure(std::format("panel layout error: expected the next panel to be {}, it is {}", index, panels_.size()));
        }

        // Texture state for a panel: the right texture on the right unit, with the panel's filter and wrap modes.
        void bind_for(const Panel& p) const
        {
            const bool is_3d = static_cast<int>(p.mode) % 2 == 0;
            const mgl::TextureTarget target = is_3d ? mgl::TextureTarget::Texture3D : mgl::TextureTarget::Texture2DArray;
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, is_3d ? p.tex : mgl::TextureId{0});
            mgl::glActiveTexture(mgl::TextureUnit::Texture1);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, is_3d ? mgl::TextureId{0} : p.tex);
            mgl::glActiveTexture(is_3d ? mgl::TextureUnit::Texture0 : mgl::TextureUnit::Texture1);

            // The default MIN_FILTER (NEAREST_MIPMAP_LINEAR) would make every texture incomplete: only level 0 is
            // sampled here, so NEAREST / LINEAR are used for both filters.
            const GLint filter = p.linear ? GL_LINEAR : GL_NEAREST;
            mgl::glTexParameteri(target, mgl::TextureParameterSetter::MinFilter, filter);
            mgl::glTexParameteri(target, mgl::TextureParameterSetter::MagFilter, filter);
            mgl::glTexParameteri(target, mgl::TextureParameterSetter::WrapS, static_cast<GLint>(W::ClampToEdge));
            mgl::glTexParameteri(target, mgl::TextureParameterSetter::WrapT, static_cast<GLint>(W::ClampToEdge));
            // TEXTURE_WRAP_R only exists for 3D textures. The array textures keep their initial WRAP_R (REPEAT) and
            // must still CLAMP the layer coordinate: a layer index is not a wrapped coordinate.
            if (is_3d) mgl::glTexParameteri(target, mgl::TextureParameterSetter::WrapR, static_cast<GLint>(p.wrap_r));
        }

        // ------------------------------------------------------------------------------------------
        // Rendering into slices and layers
        // ------------------------------------------------------------------------------------------
        void render_layers()
        {
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            mgl::glViewport(0, 0, kFboSize, kFboSize);
            mgl::glUseProgram(pattern_program_);
            for (const LayerJob& job : kLayerJobs)
            {
                // Attach ONE two-dimensional image of a three-dimensional / array texture (ES 3.0 section 4.4.2.4).
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[job.tex], 0, job.layer);
                require_complete("layer framebuffer");
                draw_pattern(job.pattern, kFboSize);
            }

            // First and last layer of the 1 x 1 x N textures: a clear of the attached image.
            mgl::glViewport(0, 0, 1, 1);
            clear_layer(tex_[kBigArr], 0, kBigColorFirst);
            clear_layer(tex_[kBigArr], max_layers_ - 1, kBigColorLast);
            clear_layer(tex_[kBig3d], 0, kBigColorFirst);
            clear_layer(tex_[kBig3d], max_3d_ - 1, kBigColorLast);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        // One full-viewport draw of pattern.frag into the current framebuffer (pattern_program_ must be in use).
        void draw_pattern(const Pattern& pattern, int size) const
        {
            const auto uniform = [&](const char* name) { return mgl::glGetUniformLocation(pattern_program_, name); };
            const auto f = [](double v) { return static_cast<float>(v); };
            mgl::glUniform3f(uniform("u_base"), f(pattern.base[0]), f(pattern.base[1]), f(pattern.base[2]));
            mgl::glUniform3f(uniform("u_gx"), f(pattern.gx[0]), f(pattern.gx[1]), f(pattern.gx[2]));
            mgl::glUniform3f(uniform("u_gy"), f(pattern.gy[0]), f(pattern.gy[1]), f(pattern.gy[2]));
            mgl::glUniform2f(uniform("u_size"), static_cast<float>(size), static_cast<float>(size));
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        void clear_layer(mgl::TextureId tex, int layer, const Color& c)
        {
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex, 0, layer);
            require_complete("big layer framebuffer");
            mgl::glClearColor(static_cast<float>(c[0] / 255.0), static_cast<float>(c[1] / 255.0), static_cast<float>(c[2] / 255.0), 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        // ------------------------------------------------------------------------------------------
        // Validation helpers
        // ------------------------------------------------------------------------------------------
        [[nodiscard]] Rgba8 frame_pixel(int x, int y) const
        {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x)) * 4;
            return {frame_[o], frame_[o + 1], frame_[o + 2], frame_[o + 3]};
        }

        // Compares every pixel of panel `idx` with `ref`. Passing |got - want| <= tolerance + 0.5: the expected value is
        // a real number, the framebuffer holds the rounded 8-bit value.
        [[nodiscard]] Probe probe(std::size_t idx, const std::function<Color(double, double)>& ref, int tol, int boundary_w, int boundary_h) const
        {
            Probe r;
            const int gx = static_cast<int>(idx) % kGrid, gy = static_cast<int>(idx) / kGrid;
            for (int py = 0; py < pw_; ++py)
                for (int px = 0; px < pw_; ++px)
                {
                    const double s = (px + 0.5) / pw_, t = (py + 0.5) / pw_;
                    if (boundary_w > 0 && near_edge(s * boundary_w)) continue;
                    if (boundary_h > 0 && near_edge(t * boundary_h)) continue;
                    const Color want = ref(s, t);
                    const Rgba8 got = frame_pixel(gx * pw_ + px, gy * pw_ + py);
                    double dev = 0.0;
                    for (std::size_t c = 0; c < 4; ++c) dev = std::max(dev, std::abs(got[c] - want[c]));
                    ++r.probes;
                    r.max_dev = std::max(r.max_dev, dev);
                    if (dev > tol + 0.5 && r.bad++ == 0)
                        r.first = std::format("first at panel pixel ({},{}): got {} want ({:.1f},{:.1f},{:.1f},{:.1f})", px, py, demo::format_rgba(got),
                                              want[0], want[1], want[2], want[3]);
                }
            return r;
        }

        // Fraction of probes where the panel does NOT match `other` within `tol`.
        [[nodiscard]] double mismatch_fraction(std::size_t idx, const std::function<Color(double, double)>& other, int tol) const
        {
            const Panel& p = panels_[idx];
            const Probe full = probe(idx, other, tol, p.boundary_w, p.boundary_h);
            return full.probes > 0 ? static_cast<double>(full.bad) / full.probes : 0.0;
        }

        void check_limits(demo::Check& check) const
        {
            check.expect(max_3d_ >= 256, "MAX_3D_TEXTURE_SIZE is at least 256 (ES 3.0 Table 6.28)", std::format("value {}", max_3d_));
            check.expect(max_layers_ >= 256, "MAX_ARRAY_TEXTURE_LAYERS is at least 256 (ES 3.0 Table 6.28)", std::format("value {}", max_layers_));
        }

        void check_panels(demo::Check& check) const
        {
            for (std::size_t i = 0; i < panels_.size(); ++i)
            {
                const Panel& p = panels_[i];
                const Probe r = probe(i, p.ref, p.tol, p.boundary_w, p.boundary_h);
                check.expect(r.probes >= 100 && r.bad == 0, std::format("panel {}: {}", i, p.name),
                             r.bad == 0 ? std::format("{} probes match the CPU model, max deviation {:.2f}/255 (tolerance {})", r.probes, r.max_dev, p.tol)
                                        : std::format("{} of {} probes differ, {}", r.bad, r.probes, r.first));
            }
        }

        // Negative controls: the same comparisons, made against the WRONG model, must fail. They prove that the checks
        // above can tell the intended behavior from its most plausible mistakes.
        void check_negative_controls(demo::Check& check) const
        {
            // Filtering across slices (panels 2, 3): a single slice is not what a trilinear lookup returns.
            for (const std::size_t idx : {std::size_t{2}, std::size_t{3}})
            {
                const float r = panels_[idx].r;
                const int slice_a = ifloor(static_cast<double>(r) * 4 - 0.5), slice_b = slice_a + 1;
                const double mismatch_a = mismatch_fraction(idx, [this, slice_a](double s, double t) { return in_slice(slice_a, s, t); }, kTolLinear);
                const double mismatch_b = mismatch_fraction(idx, [this, slice_b](double s, double t) { return in_slice(slice_b, s, t); }, kTolLinear);
                check.expect(mismatch_a > 0.9 && mismatch_b > 0.9,
                             std::format("negative control: panel {} (3D LINEAR across slices) matches neither slice {} nor slice {} alone", idx, slice_a, slice_b),
                             std::format("mismatch {:.0f}% / {:.0f}% of the probes", 100 * mismatch_a, 100 * mismatch_b));
            }

            // The three r wrap modes (panels 4, 5, 6) must give three different images.
            double min_pairwise = 1.0;
            for (const auto& pair : {std::pair<std::size_t, std::size_t>{4, 5}, {4, 6}, {5, 6}})
            {
                const Panel& other = panels_[pair.second];
                min_pairwise = std::min(min_pairwise, mismatch_fraction(pair.first, other.ref, 0));
            }
            check.expect(min_pairwise > 0.9, "negative control: REPEAT, MIRRORED_REPEAT and CLAMP_TO_EDGE give pairwise different images for r = 1.375",
                         std::format("smallest pairwise mismatch {:.0f}%", 100 * min_pairwise));
            check.expect(mismatch_fraction(7, [this](double s, double t) { return sample_volume(v_t3_, true, W::ClampToEdge, s, t, -0.05); }, kTolLinear) > 0.9,
                         "negative control: LINEAR with REPEAT at r = -0.05 is NOT the CLAMP_TO_EDGE result (slice 0 alone)");

            // Array layer selection (panels 9, 10): not a trilinear blend, and a rounding, not a floor.
            check.expect(mismatch_fraction(9, [this](double s, double t) { return sample_volume(v_ta_, true, W::ClampToEdge, s, t, 1.3 / 4); }, kTolLinear) > 0.9,
                         "negative control: array LINEAR at layer coordinate 1.3 does NOT blend neighbouring layers (a 3D texture would)");
            check.expect(mismatch_fraction(10, [this](double s, double t) { return sample_array(v_ta_, false, s, t, 1.0); }, 0) > 0.9,
                         "negative control: layer coordinate 1.7 rounds to layer 2; selecting layer 1 (floor) fails the comparison");
            check.expect(mismatch_fraction(13, [this](double s, double t) { return sample_array(v_ta_, false, s, t, 0.0); }, 0) > 0.9,
                         "negative control: layer coordinate 9.6 is clamped to the LAST layer, it does not wrap to layer 0 (WRAP_R = REPEAT is ignored for arrays)");

            // Pixel store (panels 26 and 28): IMAGE_HEIGHT 0 and 6 read different memory.
            check.expect(mismatch_fraction(kPanelPixelStore, [this](double s, double t) { return sample_volume(v_pxb_, false, W::ClampToEdge, s, t, 0.25); }, 0) > 0.5,
                         "negative control: with UNPACK_IMAGE_HEIGHT = 0 the same memory gives a different image than with 6");
            check.expect(mismatch_fraction(kPanelPixelStore, [this](double s, double t) { return sample_volume(v_pxa_, false, W::ClampToEdge, s, t, 0.75); }, 0) > 0.5,
                         "negative control: slice 0 and slice 1 of the pixel-store texture differ");

            // Render targets: rendered slices differ from the gray filler, the others do not.
            const auto gray = [](double, double) { return to_color(kGray); };
            check.expect(mismatch_fraction(kPanelF3 + 1, gray, kTolRendered) > 0.9 && mismatch_fraction(kPanelF3 + 3, gray, kTolRendered) > 0.9 &&
                             mismatch_fraction(kPanelFa + 0, gray, kTolRendered) > 0.9 && mismatch_fraction(kPanelFa + 2, gray, kTolRendered) > 0.9,
                         "negative control: the rendered slices 1 and 3 and layers 0 and 2 are not gray (they were written)");
            check.expect(mismatch_fraction(kPanelF3 + 0, gray, kTolRendered) == 0.0 && mismatch_fraction(kPanelF3 + 2, gray, kTolRendered) == 0.0 &&
                             mismatch_fraction(kPanelFa + 1, gray, kTolRendered) == 0.0 && mismatch_fraction(kPanelFa + 3, gray, kTolRendered) == 0.0,
                         "slices 0 and 2 of the 3D target and layers 1 and 3 of the array target stayed gray: rendering touched ONLY the attached image");
        }

        // What a LINEAR lookup would return if it filtered inside ONE slice only (r at the center of slice k).
        [[nodiscard]] Color in_slice(int k, double s, double t) const
        {
            return sample_volume(v_t3_, true, W::ClampToEdge, s, t, (k + 0.5) / v_t3_.d);
        }

        // glReadPixels from a framebuffer with one layer attached (the other way to see the same data).
        std::vector<Rgba8> read_layer(std::size_t slot, int layer, int size)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[slot], 0, layer);
            const auto bytes = demo::read_rgba8(0, 0, size, size);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            std::vector<Rgba8> out;
            for (std::size_t i = 0; i + 3 < bytes.size(); i += 4) out.push_back({bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]});
            return out;
        }

        static int max_deviation(const std::vector<Rgba8>& got, const Volume& expected, int layer, int size)
        {
            int dev = 0;
            for (int y = 0; y < size; ++y)
                for (int x = 0; x < size; ++x)
                    for (std::size_t c = 0; c < 4; ++c)
                        dev = std::max(dev, std::abs(int{got[static_cast<std::size_t>(y * size + x)][c]} - int{expected.at(x, y, layer)[c]}));
            return dev;
        }

        void check_layer_readback(demo::Check& check)
        {
            struct Case { std::size_t slot; const Volume* vol; int layer; const char* what; };
            const std::array<Case, 8> cases{{{kF3, &v_f3_, 0, "3D slice 0 (never attached): glReadPixels returns the gray filler"},
                                            {kF3, &v_f3_, 1, "3D slice 1: glReadPixels returns the rendered pattern"},
                                            {kF3, &v_f3_, 2, "3D slice 2 (never attached): gray"},
                                            {kF3, &v_f3_, 3, "3D slice 3: rendered pattern"},
                                            {kFa, &v_fa_, 0, "array layer 0: rendered pattern"},
                                            {kFa, &v_fa_, 1, "array layer 1 (never attached): gray"},
                                            {kFa, &v_fa_, 2, "array layer 2: rendered pattern"},
                                            {kFa, &v_fa_, 3, "array layer 3 (never attached): gray"}}};
            for (const Case& c : cases)
            {
                const auto got = read_layer(c.slot, c.layer, kFboSize);
                const int dev = max_deviation(got, *c.vol, c.layer, kFboSize);
                check.expect(dev <= kTolRendered, std::format("framebuffer with {}", c.what), std::format("max deviation {}/255 (tolerance {})", dev, kTolRendered));
            }
            // glCopyTexSubImage3D / glTexSubImage3D / glTexImage3D contents are also read back directly for exactness.
            {
                const auto got = read_layer(kT3Copy, 2, 4);
                const int dev = max_deviation(got, v_t3copy_, 2, 4);
                check.expect(dev == 0, "glReadPixels from slice 2 of the copy texture equals the CPU model of glCopyTexSubImage3D (exact)", std::format("max deviation {}", dev));
            }
            {
                const auto got = read_layer(kTaCopy, 1, 4);
                const int dev = max_deviation(got, v_tacopy_, 1, 4);
                check.expect(dev == 0, "glReadPixels from layer 1 of the array copy texture equals the CPU model of glCopyTexSubImage3D (exact)", std::format("max deviation {}", dev));
            }
        }

        static int attachment_param(mgl::FramebufferAttachmentParameter pname)
        {
            GLint v = -12345;
            mgl::glGetFramebufferAttachmentParameteriv(mgl::FramebufferTarget::Framebuffer, kColor0, pname, &v);
            return v;
        }

        void check_attachment_state(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kF3], 0, 3);
            check.expect(attachment_param(mgl::FramebufferAttachmentParameter::ObjectType) == static_cast<int>(GL_TEXTURE) &&
                             attachment_param(mgl::FramebufferAttachmentParameter::ObjectName) == static_cast<int>(tex_[kF3].value),
                         "attachment OBJECT_TYPE = TEXTURE and OBJECT_NAME = the 3D texture after glFramebufferTextureLayer");
            check.expect_eq(attachment_param(mgl::FramebufferAttachmentParameter::TextureLayer), 3, "FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER = 3 (ES 3.0 section 4.4.2.4)");
            check.expect_eq(attachment_param(mgl::FramebufferAttachmentParameter::TextureLevel), 0, "FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL = 0");
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kFa], 0, 2);
            check.expect_eq(attachment_param(mgl::FramebufferAttachmentParameter::TextureLayer), 2, "FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER = 2 for an array texture");
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            // Texture binding state and the wrap-R state of a 3D texture.
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, tex_[kScratch3d]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, tex_[kScratchArr]);
            check.expect(demo::get_integer(mgl::GetParameter::TextureBinding3D) == static_cast<int>(tex_[kScratch3d].value) &&
                             demo::get_integer(mgl::GetParameter::TextureBinding2DArray) == static_cast<int>(tex_[kScratchArr].value),
                         "TEXTURE_BINDING_3D and TEXTURE_BINDING_2D_ARRAY are separate bindings and read back what was bound");
            GLint wrap_r = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture3D, mgl::TextureParameterQuery::WrapR, &wrap_r);
            check.expect_eq(wrap_r, static_cast<GLint>(GL_REPEAT), "initial TEXTURE_WRAP_R of a 3D texture is REPEAT (ES 3.0 section 3.8.14)");
            mgl::glTexParameteri(mgl::TextureTarget::Texture3D, mgl::TextureParameterSetter::WrapR, static_cast<GLint>(GL_MIRRORED_REPEAT));
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture3D, mgl::TextureParameterQuery::WrapR, &wrap_r);
            check.expect_eq(wrap_r, static_cast<GLint>(GL_MIRRORED_REPEAT), "glTexParameteri(WRAP_R) reads back");
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2DArray, mgl::TextureId{0});
        }

        void check_pixel_store(demo::Check& check)
        {
            check.expect(initial_image_height_ == 0 && initial_skip_images_ == 0, "initial UNPACK_IMAGE_HEIGHT and UNPACK_SKIP_IMAGES are 0 (ES 3.0 Table 3.1)",
                         std::format("read {} and {}", initial_image_height_, initial_skip_images_));
            check.expect(seen_image_height_ == 6 && seen_skip_images_ == 1, "glGetIntegerv returns UNPACK_IMAGE_HEIGHT and UNPACK_SKIP_IMAGES as set by glPixelStorei (meta-gl gap G1 fix)",
                         std::format("read {} and {}", seen_image_height_, seen_skip_images_));

            // The 2D texture that was uploaded with SKIP_IMAGES = 1 and IMAGE_HEIGHT = 6.
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, tex_[kPx2d], 0);
            const auto bytes = demo::read_rgba8(0, 0, 4, 4);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            int bad = 0;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                {
                    const std::size_t o = static_cast<std::size_t>(y * 4 + x) * 4;
                    const Rgba8 got{bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]};
                    if (got != v_px2d_.at(x, y, 0)) ++bad;
                }
            check.expect(bad == 0, "glTexImage2D ignores UNPACK_SKIP_IMAGES: the 2D texture holds the first four rows of memory even with SKIP_IMAGES = 1",
                         std::format("{} of 16 texels wrong", bad));

            // Defined errors of glPixelStorei for the two parameters (valid range [0, inf), Table 3.1).
            expect_error(check, mgl::ErrorCode::InvalidValue, "glPixelStorei(UNPACK_SKIP_IMAGES, -1) -> INVALID_VALUE",
                         [] { mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, -1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glPixelStorei(UNPACK_IMAGE_HEIGHT, -1) -> INVALID_VALUE",
                         [] { mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, -1); });
            check.expect(demo::get_integer(mgl::GetParameter::UnpackSkipImages) == 0 && demo::get_integer(mgl::GetParameter::UnpackImageHeight) == 0,
                         "a rejected glPixelStorei leaves the parameter unchanged");
        }

        template <typename F>
        static void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
        {
            (void)demo::drain_gl_errors();  // start from an empty queue so that the error belongs to this call
            call();
            check.expect_gl_error(code, what);
        }

        void check_errors(demo::Check& check)
        {
            using TT = mgl::TextureTarget;
            const auto rgba = mgl::PixelFormat::Rgba;
            const auto ub = mgl::PixelType::UnsignedByte;
            mgl::glBindTexture(TT::Texture3D, tex_[kScratch3d]);
            mgl::glBindTexture(TT::Texture2DArray, tex_[kScratchArr]);
            mgl::glBindTexture(TT::Texture2D, tex_[kScratch2d]);

            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexImage3D(TEXTURE_3D) with depth = MAX_3D_TEXTURE_SIZE + 1 -> INVALID_VALUE (section 3.8.3)",
                         [&] { mgl::glTexImage3D(TT::Texture3D, 0, mgl::InternalFormat::Rgba8, 1, 1, max_3d_ + 1, 0, rgba, ub, nullptr); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexImage3D(TEXTURE_3D) with width = MAX_3D_TEXTURE_SIZE + 1 -> INVALID_VALUE",
                         [&] { mgl::glTexImage3D(TT::Texture3D, 0, mgl::InternalFormat::Rgba8, max_3d_ + 1, 1, 1, 0, rgba, ub, nullptr); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexImage3D(TEXTURE_2D_ARRAY) with depth = MAX_ARRAY_TEXTURE_LAYERS + 1 -> INVALID_VALUE",
                         [&] { mgl::glTexImage3D(TT::Texture2DArray, 0, mgl::InternalFormat::Rgba8, 1, 1, max_layers_ + 1, 0, rgba, ub, nullptr); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexImage3D with a negative depth -> INVALID_VALUE",
                         [&] { mgl::glTexImage3D(TT::Texture3D, 0, mgl::InternalFormat::Rgba8, 1, 1, -1, 0, rgba, ub, nullptr); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexImage3D with border = 1 -> INVALID_VALUE",
                         [&] { mgl::glTexImage3D(TT::Texture3D, 0, mgl::InternalFormat::Rgba8, 1, 1, 1, 1, rgba, ub, nullptr); });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "glTexImage3D(TEXTURE_2D, ...) -> INVALID_ENUM (target must be TEXTURE_3D or TEXTURE_2D_ARRAY; ES 3.2 section 8.5 errors)",
                         [&] { mgl::glTexImage3D(TT::Texture2D, 0, mgl::InternalFormat::Rgba8, 1, 1, 1, 0, rgba, ub, nullptr); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexSubImage3D(TEXTURE_3D) with zoffset + depth > texture depth -> INVALID_VALUE (section 3.8.5)",
                         [&] { mgl::glTexSubImage3D(TT::Texture3D, 0, 0, 0, 3, 1, 1, 2, rgba, ub, memory_.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexSubImage3D(TEXTURE_2D_ARRAY) with zoffset + depth > layers -> INVALID_VALUE",
                         [&] { mgl::glTexSubImage3D(TT::Texture2DArray, 0, 0, 0, 3, 1, 1, 2, rgba, ub, memory_.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glTexSubImage3D with a negative xoffset -> INVALID_VALUE",
                         [&] { mgl::glTexSubImage3D(TT::Texture3D, 0, -1, 0, 0, 1, 1, 1, rgba, ub, memory_.data()); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glCopyTexSubImage3D with zoffset = depth -> INVALID_VALUE",
                         [&] { mgl::glCopyTexSubImage3D(TT::Texture3D, 0, 0, 0, 4, 0, 0, 1, 1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glCopyTexSubImage3D(TEXTURE_2D_ARRAY) with zoffset = number of layers -> INVALID_VALUE",
                         [&] { mgl::glCopyTexSubImage3D(TT::Texture2DArray, 0, 0, 0, 4, 0, 0, 1, 1); });

            // glFramebufferTextureLayer (section 4.4.2.4).
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glFramebufferTextureLayer with a 2D texture -> INVALID_OPERATION (3D or 2D array only)",
                         [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratch2d], 0, 0); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with layer = MAX_ARRAY_TEXTURE_LAYERS for an array -> INVALID_VALUE",
                         [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratchArr], 0, max_layers_); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with layer = MAX_3D_TEXTURE_SIZE for a 3D texture -> INVALID_VALUE",
                         [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratch3d], 0, max_3d_); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with a negative layer -> INVALID_VALUE",
                         [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratchArr], 0, -1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glFramebufferTextureLayer with level 40 (beyond log2 of the maximum size) -> INVALID_VALUE",
                         [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratchArr], 40, 0); });
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glFramebufferTextureLayer with the default framebuffer bound -> INVALID_OPERATION",
                         [&] { mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratchArr], 0, 0); });
        }

        // The framebuffer glCopyTexSubImage3D reads from, as it was drawn: +-1 of the pattern formula.
        void check_copy_source(demo::Check& check) const
        {
            int dev = 0;
            for (int y = 0; y < kSrcSize; ++y)
                for (int x = 0; x < kSrcSize; ++x)
                {
                    const Rgba8 want = pattern_texel(kSrcPattern, x, y, kSrcSize);
                    for (std::size_t c = 0; c < 4; ++c) dev = std::max(dev, std::abs(int{src_.at(x, y, 0)[c]} - int{want[c]}));
                }
            check.expect(dev <= kTolRendered, "the 8x8 source framebuffer of glCopyTexSubImage3D holds the drawn pattern",
                         std::format("max deviation {}/255 (tolerance {})", dev, kTolRendered));
        }

        // glCopyTexSubImage3D from a framebuffer that was filled by scissored glClear calls (an ordinary way to paint a
        // few solid rectangles). The copy is a command like any other: it must see everything that was issued before it,
        // without a glFlush or glFinish in between (ES 3.0 section 2.1: commands are executed in the order they are issued).
        void check_copy_after_scissored_clears(demo::Check& check)
        {
            const std::array<Rgba8, 4> quadrant{{{230, 40, 40, 255}, {40, 200, 60, 255}, {50, 70, 230, 255}, {240, 220, 30, 255}}};  // lower left, lower right, upper left, upper right
            auto quadrant_color = [&](int x, int y) { return quadrant[static_cast<std::size_t>((y / 4) * 2 + x / 4)]; };

            // A fresh 8x8 source texture, so that nothing earlier in the program has touched the image.
            mgl::TextureId source{};
            mgl::glGenTextures(1, &source);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, source);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kSrcSize, kSrcSize, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, nullptr);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[1]);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, source, 0);
            mgl::glViewport(0, 0, kSrcSize, kSrcSize);
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (int q = 0; q < 4; ++q)
            {
                const Rgba8& c = quadrant[static_cast<std::size_t>(q)];
                mgl::glScissor((q % 2) * 4, (q / 2) * 4, 4, 4);
                mgl::glClearColor(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);

            // The destination is a gray 4x4x3 texture, like the main copy test: 4 x 3 texels from (2, 4) into slice 2 at (0, 1).
            const Volume gray = filled(4, 4, 3, kGray);
            upload_3d(mgl::TextureTarget::Texture3D, kClearCopy, gray);
            mgl::glCopyTexSubImage3D(mgl::TextureTarget::Texture3D, 0, 0, 1, 2, 2, 4, 4, 3);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});

            const auto got = read_layer(kClearCopy, 2, 4);
            int wrong = 0;
            Rgba8 first_got{}, first_want{};
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                {
                    const Rgba8 want = y >= 1 ? quadrant_color(2 + x, 4 + (y - 1)) : kGray;
                    const Rgba8 have = got[static_cast<std::size_t>(y * 4 + x)];
                    if (have != want && wrong++ == 0) first_got = have, first_want = want;
                }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[1]);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteTextures(1, &source);
            check.expect_or_known_deviation(wrong == 0, "glCopyTexSubImage3D copies a framebuffer that was painted by scissored glClear calls (command order)",
                                            wrong == 0 ? std::string{}
                                                       : std::format("{} of 16 texels wrong, first got {} want {}", wrong, demo::format_rgba(first_got), demo::format_rgba(first_want)),
                                            "zink", "the copy does not see the pending scissored clears of the source attachment and returns stale texels; a glFlush before the copy gives the correct result");
        }

        // ES 3.0 section 4.4.4.1: a layer that does not exist makes the attachment incomplete.
        void check_framebuffer_completeness(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_[0]);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratchArr], 0, 3);
            const mgl::FramebufferStatus last = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratchArr], 0, 4);  // the array has 4 layers: 0-3
            const mgl::FramebufferStatus beyond = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, tex_[kScratch3d], 0, 4);  // a 3D texture of depth 4
            const mgl::FramebufferStatus beyond_3d = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureId{0}, 0, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect(last == mgl::FramebufferStatus::Complete, "attaching the last layer (3 of 4) of an array is complete", std::string(demo::framebuffer_status_name(last)));
            check.expect(beyond == mgl::FramebufferStatus::IncompleteAttachment, "attaching layer 4 of a 4-layer array is FRAMEBUFFER_INCOMPLETE_ATTACHMENT (section 4.4.4.1)",
                         std::string(demo::framebuffer_status_name(beyond)));
            check.expect(beyond_3d == mgl::FramebufferStatus::IncompleteAttachment, "attaching slice 4 of a 3D texture of depth 4 is FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
                         std::string(demo::framebuffer_status_name(beyond_3d)));
        }

        // ------------------------------------------------------------------------------------------
        mgl::ProgramId sample_program_{};
        mgl::ProgramId pattern_program_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        std::array<mgl::FramebufferId, 2> fbo_{};  // [0] layer attachments, [1] copy source
        int max_3d_ = 0, max_layers_ = 0;

        Volume src_, v_t3_, v_ta_, v_t3sub_, v_t3copy_, v_tacopy_, v_pxa_, v_pxb_, v_pxc_, v_pxarr_, v_px2d_, v_f3_, v_fa_;
        std::array<Volume, 2> v_s3_, v_sa_;
        std::vector<Rgba8> memory_;
        std::vector<Panel> panels_;

        int initial_image_height_ = -1, initial_skip_images_ = -1, seen_image_height_ = -1, seen_skip_images_ = -1;
        int pw_ = 32;
        int width_ = 0;
        std::vector<std::uint8_t> frame_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-texture-3d-and-arrays",
        .title = "3D textures and 2D array textures",
        .min_es = {3, 0},
        .summary = "sampler3D and sampler2DArray: NEAREST/LINEAR filtering across slices, r wrapping, layer rounding, texelFetch, "
                   "textureSize, glTexSubImage3D/glCopyTexSubImage3D, UNPACK_IMAGE_HEIGHT/SKIP_IMAGES and rendering into one "
                   "slice or layer with glFramebufferTextureLayer.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Texture3dAndArrays>(argc, argv, info);
}
