// es30-sampler-objects
//
// Sampler objects (OpenGL ES 3.0, section 3.8.2): the sampling state of a texture (filters, wrap
// modes, LOD range, comparison mode) can be moved out of the texture into a sampler object that is
// bound to a texture UNIT. While a sampler is bound, its state supersedes the state of whatever
// texture is bound to that unit; binding sampler 0 gives the texture its own state back.
//
// Contact sheet of 16 cells (64 x 64 pixels each, row 0 is at the bottom). Every cell is a draw call;
// "A" is a 4x4 texture whose own state is NEAREST/REPEAT, "B" is the same texels with LINEAR/CLAMP_TO_EDGE
// and "M" is a 64x64 texture with 7 mipmap levels, every level one flat color.
//
//   row 3 (top):  12 M, sampler MAX_LOD 0   | 13 M, LINEAR_MIPMAP_LINEAR, LOD 1.5 | 14 M, own MIN_LOD 3 | 15 M, own MIN_LOD 3 + sampler
//   row 2:         8 A, TexParameter under a sampler | 9 A, sampler unbound        | 10 M, default LOD range | 11 M, sampler MIN_LOD 3
//   row 1:         4 B + sampler NEAREST/REPEAT | 5 B on 2 units, 2 samplers | 6 A on 2 units, 1 sampler | 7 the same sampler switched
//   row 0 (bottom): 0 A, own state | 1 A + sampler LINEAR/CLAMP | 2 A, sampler deleted | A incomplete | 3 B, own state
//   (kTexelCells and kLodCells below describe every cell; render() draws them in the order 0, 1, ... 15)
//
// Validation: every texel cell is compared with a CPU implementation of the ES 3.0 3.8.10 sampling
// equations (nearest, bilinear, three wrap modes), every LOD cell with the expected mip level color.
// The sampler state is read back with glGetSamplerParameter{i,f}v, and the object life cycle,
// SAMPLER_BINDING per unit and the defined errors are checked.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using W = mgl::TextureWrapMode;

namespace
{
    constexpr int kBase = 4;     // textures A and B are 4 x 4
    constexpr int kMip = 64;     // texture M is 64 x 64 with 7 levels
    constexpr int kLevels = 7;
    constexpr int kCell = 64;    // cell size in pixels
    constexpr int kGrid = 4;

    constexpr std::array<mgl::TextureUnit, 4> kUnits = {mgl::TextureUnit::Texture0, mgl::TextureUnit::Texture1,
                                                       mgl::TextureUnit::Texture2, mgl::TextureUnit::Texture3};

    constexpr GLint value_of(W w) { return static_cast<GLint>(static_cast<GLenum>(w)); }
    constexpr GLint value_of(mgl::TextureMinFilter f) { return static_cast<GLint>(static_cast<GLenum>(f)); }
    constexpr GLint value_of(mgl::TextureMagFilter f) { return static_cast<GLint>(static_cast<GLenum>(f)); }

    Rgba8 base_texel(int i, int j)
    {
        return {static_cast<std::uint8_t>(40 + 60 * i), static_cast<std::uint8_t>(40 + 60 * j),
                static_cast<std::uint8_t>((i + j) % 2 ? 220 : 30), 255};
    }

    // One flat color per mipmap level of texture M (level 0 is 64x64, level 6 is 1x1).
    constexpr std::array<Rgba8, kLevels> kLevelColor = {{
        {230, 40, 40, 255}, {40, 200, 60, 255}, {50, 70, 230, 255}, {230, 210, 40, 255},
        {200, 60, 200, 255}, {40, 200, 200, 255}, {240, 240, 240, 255}}};

    // ---------------------------------------------------------------------------------------------
    // CPU reference of ES 3.0 3.8.10.2: texel selection with NEAREST or LINEAR filtering and the wrap
    // modes of Table 3.22 (independent for s and t). u, v are in texel units divided by the size.
    // ---------------------------------------------------------------------------------------------
    struct Look
    {
        bool linear = false;
        W wrap_s = W::Repeat;
        W wrap_t = W::Repeat;
    };

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

    std::array<double, 4> sample_ref(const std::vector<Rgba8>& texels, const Look& look, double u, double v)
    {
        const double us = u * kBase, vs = v * kBase;
        auto at = [&](int i, int j) {
            const Rgba8 t = texels[static_cast<std::size_t>(wrap(j, kBase, look.wrap_t) * kBase + wrap(i, kBase, look.wrap_s))];
            return std::array<double, 4>{double(t[0]), double(t[1]), double(t[2]), double(t[3])};
        };
        if (!look.linear) return at(static_cast<int>(std::floor(us)), static_cast<int>(std::floor(vs)));
        const int i0 = static_cast<int>(std::floor(us - 0.5)), j0 = static_cast<int>(std::floor(vs - 0.5));
        const double a = (us - 0.5) - std::floor(us - 0.5), b = (vs - 0.5) - std::floor(vs - 0.5);
        std::array<double, 4> r{};
        const auto t00 = at(i0, j0), t10 = at(i0 + 1, j0), t01 = at(i0, j0 + 1), t11 = at(i0 + 1, j0 + 1);
        for (std::size_t c = 0; c < 4; ++c)
            r[c] = (1 - a) * (1 - b) * t00[c] + a * (1 - b) * t10[c] + (1 - a) * b * t01[c] + a * b * t11[c];
        return r;
    }

    bool near_texel_edge(double texel_coord)
    {
        const double f = texel_coord - std::floor(texel_coord);
        return f < 0.1 || f > 0.9;
    }

    // ---------------------------------------------------------------------------------------------
    // What every cell must show. The draw calls in render() produce exactly these configurations.
    // ---------------------------------------------------------------------------------------------
    struct Cell
    {
        const char* what;
        Look left;             // look of texture unit 0 (the whole cell unless split)
        Look right;            // look of texture unit 1 (split cells: right half)
        bool split = false;
        float su = 3.0f, sv = 3.0f, ou = -1.0f, ov = -1.0f;  // texture coordinate = corner * scale + offset
        bool right_incomplete = false;  // unit 1 samples an incomplete texture: (0, 0, 0, 1)
    };

    constexpr Look kNearestRepeat{false, W::Repeat, W::Repeat};
    constexpr Look kLinearClamp{true, W::ClampToEdge, W::ClampToEdge};
    constexpr Look kNearestClamp{false, W::ClampToEdge, W::ClampToEdge};
    constexpr Look kLinearRepeat{true, W::Repeat, W::Repeat};

    // Cells 0..9: textures A and B sampled over [-1, 2]^2 (three repetitions); split cells over [-0.5, 1.5]^2.
    const std::array<Cell, 10> kTexelCells = {{
        {"texture A, no sampler: its own NEAREST/REPEAT state", kNearestRepeat, {}},
        {"texture A + sampler LINEAR/CLAMP_TO_EDGE: the sampler overrides the texture", kLinearClamp, {}},
        {"split: sampler deleted while bound, so texture A's own NEAREST/REPEAT state applies again", kNearestRepeat, {}, true, 2.0f, 2.0f, -0.5f, -0.5f, true},
        {"texture B, no sampler: its own LINEAR/CLAMP_TO_EDGE state", kLinearClamp, {}},
        {"texture B + sampler NEAREST/REPEAT: the sampler overrides the texture", kNearestRepeat, {}},
        {"split: texture B on unit 0 + sampler NEAREST/REPEAT | on unit 1 + sampler LINEAR, MIRRORED_REPEAT/CLAMP_TO_EDGE",
         kNearestRepeat, {true, W::MirroredRepeat, W::ClampToEdge}, true, 2.0f, 2.0f, -0.5f, -0.5f},
        {"split: ONE sampler (LINEAR/CLAMP_TO_EDGE) bound to units 0 and 1: both halves identical", kLinearClamp, kLinearClamp, true, 2.0f, 2.0f, -0.5f, -0.5f},
        {"split: the same sampler switched to NEAREST: both units change with it", kNearestClamp, kNearestClamp, true, 2.0f, 2.0f, -0.5f, -0.5f},
        {"glTexParameter while sampler NEAREST/CLAMP is bound: the sampler still wins", kNearestClamp, {}},
        {"sampler unbound with glBindSampler(0, 0): the texture's changed state (LINEAR/REPEAT) is used", kLinearRepeat, {}},
    }};

    // Cells 10..15: texture M with a sampler (or its own state) deciding the LOD range. The quad maps
    // 128 texels onto 64 pixels in u and keeps v constant, so rho = 2 and lambda = 1 for every pixel.
    struct LodCell
    {
        const char* what;
        std::array<double, 4> expected;  // color; for the fractional case the mix of two levels
        int tolerance;
    };

    std::array<double, 4> level_color(int level)
    {
        const Rgba8& c = kLevelColor[static_cast<std::size_t>(level)];
        return {double(c[0]), double(c[1]), double(c[2]), double(c[3])};
    }

    std::array<double, 4> mix_levels(int a, int b, double t)
    {
        std::array<double, 4> r{};
        for (std::size_t i = 0; i < 4; ++i) r[i] = (1 - t) * level_color(a)[i] + t * level_color(b)[i];
        return r;
    }

    const std::array<LodCell, 6> kLodCells = {{
        {"sampler NEAREST_MIPMAP_NEAREST, default LOD range: lambda = 1 selects level 1", level_color(1), 1},
        {"sampler MIN_LOD = 3 (glSamplerParameterf): lambda is clamped up to 3, level 3", level_color(3), 1},
        {"sampler MAX_LOD = 0 (glSamplerParameteri): lambda is clamped down to 0, level 0", level_color(0), 1},
        {"sampler LINEAR_MIPMAP_LINEAR, MIN_LOD = MAX_LOD = 1.5 (glSamplerParameterfv): half level 1, half level 2", mix_levels(1, 2, 0.5), 3},
        {"texture M's own MIN_LOD = 3, no sampler: level 3", level_color(3), 1},
        {"same texture, now with a sampler (default LOD range): the sampler overrides the texture's MIN_LOD, level 1", level_color(1), 1},
    }};

    class SamplerObjects final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kGrid * kCell || env.height < kGrid * kCell)
                throw demo::Skip(std::format("this demo draws into the lower left {0}x{0} pixels and needs a window at least that large", kGrid * kCell));

            program_ = demo::build_program(env, "shaders/quad.vert", "shaders/sampled.frag");
            u_tex_a_ = mgl::glGetUniformLocation(program_, "u_tex_a");
            u_tex_b_ = mgl::glGetUniformLocation(program_, "u_tex_b");
            u_scale_ = mgl::glGetUniformLocation(program_, "u_uv_scale");
            u_offset_ = mgl::glGetUniformLocation(program_, "u_uv_offset");
            u_split_ = mgl::glGetUniformLocation(program_, "u_split");

            for (int j = 0; j < kBase; ++j)
                for (int i = 0; i < kBase; ++i) texels_.push_back(base_texel(i, j));

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());

            // Textures A and B: the same 4x4 texels, different OWN sampling state.
            const std::array<Look, 2> own = {kNearestRepeat, kLinearClamp};
            for (std::size_t t = 0; t < 2; ++t)
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[t]);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kBase, kBase, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, texels_.data());
                set_own_state(own[t]);
            }

            // Texture M: 7 levels, each level a single flat color; own filter NEAREST_MIPMAP_NEAREST.
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexM]);
            for (int level = 0; level < kLevels; ++level)
            {
                const int size = kMip >> level;
                const std::vector<Rgba8> flat(static_cast<std::size_t>(size * size), kLevelColor[static_cast<std::size_t>(level)]);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgba8, size, size, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, flat.data());
            }
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, value_of(mgl::TextureMinFilter::NearestMipmapNearest));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, value_of(mgl::TextureMagFilter::Nearest));

            // Samplers that live for the whole run. Their parameters are (re)set where they are used.
            mgl::glGenSamplers(static_cast<GLsizei>(samplers_.size()), samplers_.data());
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            mgl::glViewport(0, 0, kGrid * kCell, kGrid * kCell);
            mgl::glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(program_);
            mgl::glUniform1i(u_tex_a_, 0);  // sampler uniforms hold a texture UNIT index
            mgl::glUniform1i(u_tex_b_, 1);

            const mgl::SamplerId none{0};
            const mgl::TextureId a = tex_[kTexA], b = tex_[kTexB], m = tex_[kTexM];
            const mgl::SamplerId s_override = samplers_[0];  // a sampler that is used in cell 1 and deleted in cell 2

            // --- cell 0: no sampler on either unit. The texture's own state applies.
            set_uniform_split(false);
            bind_unit(0, a, none);
            bind_unit(1, a, none);
            draw_cell(0);

            // --- cell 1: a sampler with LINEAR + CLAMP_TO_EDGE is bound to unit 0. Texture A's own state is NEAREST/REPEAT.
            fresh_sampler(s_override, mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Linear, W::ClampToEdge, W::ClampToEdge);
            bind_unit(0, a, s_override);
            sampler_name_while_bound_ = s_override.value;
            sampler_binding_while_bound_ = demo::get_integer(mgl::GetParameter::SamplerBinding);  // active unit is 0
            draw_cell(1);

            // --- cell 2: deleting a sampler that is bound behaves as glBindSampler(unit, 0).
            mgl::glDeleteSamplers(1, &samplers_[0]);
            sampler_binding_after_delete_ = demo::get_integer(mgl::GetParameter::SamplerBinding);
            is_sampler_after_delete_ = mgl::glIsSampler(s_override);
            // The right half of cell 2 uses unit 1: texture A with a sampler that still has its INITIAL state, MIN_FILTER =
            // NEAREST_MIPMAP_LINEAR. A has a single level, so it is not mipmap complete: with this sampler it is incomplete
            // on unit 1 (3.8.13.1) although the very same texture is complete on unit 0.
            bind_unit(1, a, samplers_[8]);
            set_uniform_split(true);
            draw_cell(2);
            set_uniform_split(false);
            mgl::glGenSamplers(1, &samplers_[0]);  // so that the next frame (and shutdown) has a valid name again

            // --- cell 3: texture B alone (its own LINEAR/CLAMP_TO_EDGE state) ...
            bind_unit(0, b, none);
            bind_unit(1, b, none);
            draw_cell(3);

            // --- cell 4: ... and with a NEAREST/REPEAT sampler on unit 0.
            const mgl::SamplerId s_nr = samplers_[1];
            fresh_sampler(s_nr, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest, W::Repeat, W::Repeat);
            bind_unit(0, b, s_nr);
            draw_cell(4);

            // --- cell 5: ONE texture on TWO units with TWO different samplers in a single draw call.
            const mgl::SamplerId s_left = samplers_[2], s_right = samplers_[3];
            fresh_sampler(s_left, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest, W::Repeat, W::Repeat);
            fresh_sampler(s_right, mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Linear, W::ClampToEdge, W::ClampToEdge);
            const GLint mirrored = value_of(W::MirroredRepeat);
            mgl::glSamplerParameteriv(s_right, mgl::SamplerParameter::WrapS, &mirrored);  // the vector form of the setter
            bind_unit(0, b, s_left);
            bind_unit(1, b, s_right);
            set_uniform_split(true);
            draw_cell(5);

            // --- cell 6: one sampler bound to both units.
            const mgl::SamplerId s_shared = samplers_[4];
            fresh_sampler(s_shared, mgl::TextureMinFilter::Linear, mgl::TextureMagFilter::Linear, W::ClampToEdge, W::ClampToEdge);
            bind_unit(0, a, s_shared);
            bind_unit(1, a, s_shared);
            draw_cell(6);

            // --- cell 7: changing the sampler changes what every unit it is bound to sees (3.8.2).
            mgl::glSamplerParameteri(s_shared, mgl::SamplerParameter::MinFilter, value_of(mgl::TextureMinFilter::Nearest));
            mgl::glSamplerParameteri(s_shared, mgl::SamplerParameter::MagFilter, value_of(mgl::TextureMagFilter::Nearest));
            draw_cell(7);

            // --- cell 8: glTexParameter modifies the TEXTURE bound to the active unit, not the sampler.
            const mgl::SamplerId s_nc = samplers_[5];
            fresh_sampler(s_nc, mgl::TextureMinFilter::Nearest, mgl::TextureMagFilter::Nearest, W::ClampToEdge, W::ClampToEdge);
            bind_unit(0, a, s_nc);
            bind_unit(1, a, none);
            set_uniform_split(false);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, value_of(mgl::TextureMinFilter::Linear));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, value_of(mgl::TextureMagFilter::Linear));
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MagFilter, &texture_mag_after_texparameter_);
            mgl::glGetSamplerParameteriv(s_nc, mgl::SamplerParameter::MagFilter, &sampler_mag_after_texparameter_);
            draw_cell(8);

            // --- cell 9: unbind the sampler. The texture's own state (now LINEAR + REPEAT) is used.
            mgl::glBindSampler(0, none);
            draw_cell(9);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, value_of(mgl::TextureMinFilter::Nearest));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, value_of(mgl::TextureMagFilter::Nearest));

            // --- cells 10..13: LOD range of a sampler on the mipmapped texture M.
            const mgl::SamplerId s_lod = samplers_[6];
            bind_unit(0, m, s_lod);
            bind_unit(1, a, none);
            mgl::glSamplerParameteri(s_lod, mgl::SamplerParameter::MinFilter, value_of(mgl::TextureMinFilter::NearestMipmapNearest));
            mgl::glSamplerParameteri(s_lod, mgl::SamplerParameter::MagFilter, value_of(mgl::TextureMagFilter::Nearest));
            mgl::glSamplerParameterf(s_lod, mgl::SamplerParameter::MinLod, -1000.0f);
            mgl::glSamplerParameterf(s_lod, mgl::SamplerParameter::MaxLod, 1000.0f);
            draw_lod_cell(10);

            mgl::glSamplerParameterf(s_lod, mgl::SamplerParameter::MinLod, 3.0f);  // clamp lambda from below
            draw_lod_cell(11);

            mgl::glSamplerParameterf(s_lod, mgl::SamplerParameter::MinLod, -1000.0f);
            mgl::glSamplerParameteri(s_lod, mgl::SamplerParameter::MaxLod, 0);  // clamp lambda from above (int form of a float parameter)
            draw_lod_cell(12);

            mgl::glSamplerParameteri(s_lod, mgl::SamplerParameter::MinFilter, value_of(mgl::TextureMinFilter::LinearMipmapLinear));
            const GLfloat lod = 1.5f;
            mgl::glSamplerParameterfv(s_lod, mgl::SamplerParameter::MinLod, &lod);
            mgl::glSamplerParameterfv(s_lod, mgl::SamplerParameter::MaxLod, &lod);
            draw_lod_cell(13);

            // --- cell 14: the same effect from the texture's own MIN_LOD, no sampler ...
            bind_unit(0, m, none);
            mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinLod, 3.0f);
            draw_lod_cell(14);

            // --- cell 15: ... and a sampler with the default LOD range takes it away again.
            const mgl::SamplerId s_lod2 = samplers_[7];
            fresh_sampler(s_lod2, mgl::TextureMinFilter::NearestMipmapNearest, mgl::TextureMagFilter::Nearest, W::Repeat, W::Repeat);
            bind_unit(0, m, s_lod2);
            draw_lod_cell(15);

            // Leave no sampler bound and texture M in its initial state, so that every frame is the same.
            bind_unit(0, a, none);
            bind_unit(1, a, none);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, m);
            mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinLod, -1000.0f);
            set_uniform_split(false);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            // The cells are read first; the state tests below do not draw.
            check_texel_cells(check);
            check_lod_cells(check);
            check_override_is_visible(check);
            check_render_time_state(check);
            check_sampler_defaults(check);
            check_sampler_roundtrips(check);
            check_sampler_lifecycle(check);
            check_sampler_bindings(check);
            check_errors(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            for (int unit = 0; unit < 4; ++unit) mgl::glBindSampler(static_cast<GLuint>(unit), mgl::SamplerId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteSamplers(static_cast<GLsizei>(samplers_.size()), samplers_.data());
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(program_);
        }

    private:
        enum : std::size_t { kTexA, kTexB, kTexM, kTexCount };

        // ------------------------------------------------------------------------------------------
        // Drawing helpers
        // ------------------------------------------------------------------------------------------
        /// Texture + sampler on one unit. glBindSampler takes the unit INDEX, glBindTexture the active unit.
        static void bind_unit(int unit, mgl::TextureId texture, mgl::SamplerId sampler)
        {
            mgl::glActiveTexture(kUnits[static_cast<std::size_t>(unit)]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, texture);
            mgl::glBindSampler(static_cast<GLuint>(unit), sampler);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        }

        /// Gives a sampler object a known state with the integer setters.
        static void fresh_sampler(mgl::SamplerId s, mgl::TextureMinFilter min, mgl::TextureMagFilter mag, W wrap_s, W wrap_t)
        {
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::MinFilter, value_of(min));
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::MagFilter, value_of(mag));
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::WrapS, value_of(wrap_s));
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::WrapT, value_of(wrap_t));
        }

        /// Sets the OWN state of the texture bound to the active unit (what glTexParameter has always done).
        static void set_own_state(const Look& look)
        {
            const GLint filter = look.linear ? value_of(mgl::TextureMinFilter::Linear) : value_of(mgl::TextureMinFilter::Nearest);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, filter);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, filter);
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::WrapS, value_of(look.wrap_s));
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::WrapT, value_of(look.wrap_t));
        }

        void set_uniform_split(bool split) const { mgl::glUniform1i(u_split_, split ? 1 : 0); }

        static void set_cell_viewport(int index)
        {
            mgl::glViewport((index % kGrid) * kCell, (index / kGrid) * kCell, kCell, kCell);
        }

        void draw_cell(int index) const
        {
            const Cell& c = kTexelCells[static_cast<std::size_t>(index)];
            set_cell_viewport(index);
            mgl::glUniform2f(u_scale_, c.su, c.sv);
            mgl::glUniform2f(u_offset_, c.ou, c.ov);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// u spans 2 over the cell (128 texels over 64 pixels: lambda = 1), v is constant 0.5.
        void draw_lod_cell(int index) const
        {
            set_cell_viewport(index);
            mgl::glUniform2f(u_scale_, 2.0f, 0.0f);
            mgl::glUniform2f(u_offset_, 0.0f, 0.5f);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        // ------------------------------------------------------------------------------------------
        // Checks on the pixels
        // ------------------------------------------------------------------------------------------
        static int cell_x(int index) { return (index % kGrid) * kCell; }
        static int cell_y(int index) { return (index / kGrid) * kCell; }

        struct HalfResult
        {
            int probes = 0;
            int bad = 0;
            std::string first_bad;
        };

        /// Compares one half (or the whole) of a cell with the CPU sampler. `pixels` is the whole cell read back.
        HalfResult compare_half(const std::vector<std::uint8_t>& pixels, const Cell& cell, const Look& look, bool left_half, bool right_half,
                                bool incomplete = false) const
        {
            HalfResult r;
            const int tolerance = look.linear ? 3 : 0;  // bilinear weights are only required to be precise to the implementation's accuracy
            for (int ly = 2; ly < kCell; ly += 4)
                for (int lx = 2; lx < kCell; lx += 4)
                {
                    if (cell.split && ((lx < kCell / 2) != left_half || (lx >= kCell / 2) != right_half)) continue;
                    if (cell.split && std::abs(lx - kCell / 2) < 3) continue;  // keep away from the seam of the two halves
                    double cx = (lx + 0.5) / kCell;
                    if (cell.split) cx = cx < 0.5 ? cx * 2.0 : (cx - 0.5) * 2.0;
                    const double u = cx * double{cell.su} + double{cell.ou};
                    const double v = ((ly + 0.5) / kCell) * double{cell.sv} + double{cell.ov};
                    if (!look.linear && (near_texel_edge(u * kBase) || near_texel_edge(v * kBase))) continue;  // texel boundary: implementation-defined rounding
                    // An incomplete texture returns (0, 0, 0, 1) whatever the coordinates are (ES 3.0 3.9.2.1).
                    const std::array<double, 4> ref = incomplete ? std::array<double, 4>{0.0, 0.0, 0.0, 255.0} : sample_ref(texels_, look, u, v);
                    ++r.probes;
                    bool ok = true;
                    std::array<int, 4> got{};
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        got[k] = pixels[static_cast<std::size_t>((ly * kCell + lx) * 4) + k];
                        ok = ok && std::abs(got[k] - ref[k]) <= tolerance + 0.5;
                    }
                    if (!ok && r.bad++ == 0)
                        r.first_bad = std::format("at uv=({:.3f},{:.3f}) got ({},{},{},{}) ref ({:.1f},{:.1f},{:.1f},{:.1f})", u, v, got[0], got[1], got[2], got[3],
                                                  ref[0], ref[1], ref[2], ref[3]);
                }
            return r;
        }

        void check_texel_cells(demo::Check& check) const
        {
            for (int c = 0; c < static_cast<int>(kTexelCells.size()); ++c)
            {
                const Cell& cell = kTexelCells[static_cast<std::size_t>(c)];
                const std::vector<std::uint8_t> pixels = demo::read_rgba8(cell_x(c), cell_y(c), kCell, kCell);
                if (!cell.split)
                {
                    const HalfResult r = compare_half(pixels, cell, cell.left, true, true);
                    check.expect(r.probes > 20 && r.bad == 0, std::format("cell {}: {}: matches the CPU sampler at {} probes", c, cell.what, r.probes), r.first_bad);
                }
                else
                {
                    const HalfResult l = compare_half(pixels, cell, cell.left, true, false);
                    const HalfResult r = compare_half(pixels, cell, cell.right, false, true, cell.right_incomplete);
                    check.expect(l.probes > 10 && l.bad == 0, std::format("cell {} (left half, unit 0): {}", c, cell.what), std::format("{} probes; {}", l.probes, l.first_bad));
                    check.expect(r.probes > 10 && r.bad == 0,
                                 std::format("cell {} (right half, unit 1): {} at {} probes", c,
                                             cell.right_incomplete ? "A + a sampler in its initial NEAREST_MIPMAP_LINEAR state is incomplete (no mip chain): all probes are (0, 0, 0, 1)"
                                                                 : "matches the CPU sampler of the other sampler", r.probes),
                                 r.first_bad);
                }
            }
        }

        void check_lod_cells(demo::Check& check) const
        {
            for (std::size_t i = 0; i < kLodCells.size(); ++i)
            {
                const int index = 10 + static_cast<int>(i);
                const LodCell& cell = kLodCells[i];
                const std::vector<std::uint8_t> pixels = demo::read_rgba8(cell_x(index), cell_y(index), kCell, kCell);
                int bad = 0;
                std::string first;
                for (int ly = 4; ly < kCell; ly += 8)
                    for (int lx = 4; lx < kCell; lx += 8)
                        for (std::size_t k = 0; k < 4; ++k)
                        {
                            const int got = pixels[static_cast<std::size_t>((ly * kCell + lx) * 4) + k];
                            if (std::abs(got - cell.expected[k]) > cell.tolerance + 0.5 && bad++ == 0)
                                first = std::format("at ({},{}) channel {}: got {} expected {:.1f}", lx, ly, k, got, cell.expected[k]);
                        }
                check.expect(bad == 0, std::format("cell {}: {}", index, cell.what), first);
            }
        }

        // Negative control / mutation argument: the sampler changes the picture. If the sampler were ignored,
        // cell 1 would equal cell 0 (same texture, same coordinates); if the CPU model were wrong, cell 0
        // would not match the model of cell 1.
        void check_override_is_visible(demo::Check& check) const
        {
            const std::vector<std::uint8_t> c0 = demo::read_rgba8(cell_x(0), cell_y(0), kCell, kCell);
            const std::vector<std::uint8_t> c1 = demo::read_rgba8(cell_x(1), cell_y(1), kCell, kCell);
            int differing = 0;
            for (std::size_t i = 0; i < c0.size(); i += 4)
                if (std::abs(c0[i] - c1[i]) > 8 || std::abs(c0[i + 1] - c1[i + 1]) > 8 || std::abs(c0[i + 2] - c1[i + 2]) > 8) ++differing;
            check.expect(differing > kCell * kCell / 4, "negative control: cells 0 and 1 (same texture and coordinates, with and without sampler) differ in many pixels",
                         std::format("{} of {} pixels differ", differing, kCell * kCell));

            const HalfResult wrong = compare_half(c0, kTexelCells[1], kTexelCells[1].left, true, true);  // cell 0's pixels against cell 1's model
            check.expect(wrong.bad > 20, "negative control: the pixels of cell 0 do not match the CPU model of LINEAR/CLAMP_TO_EDGE (the comparison can fail)",
                         std::format("{} of {} probes differ", wrong.bad, wrong.probes));
        }

        // ------------------------------------------------------------------------------------------
        // Checks on state (values captured during render() and queries issued here)
        // ------------------------------------------------------------------------------------------
        void check_render_time_state(demo::Check& check) const
        {
            check.expect(sampler_binding_while_bound_ != 0 && sampler_binding_while_bound_ == static_cast<GLint>(sampler_name_while_bound_),
                         "SAMPLER_BINDING of unit 0 is the name of the bound sampler (cell 1)",
                         std::format("SAMPLER_BINDING = {}, sampler = {}", sampler_binding_while_bound_, sampler_name_while_bound_));
            check.expect_eq(sampler_binding_after_delete_, 0, "deleting the bound sampler resets SAMPLER_BINDING of its unit to 0 (cell 2)");
            check.expect(!is_sampler_after_delete_, "glIsSampler of a deleted sampler is FALSE");
            check.expect(texture_mag_after_texparameter_ == value_of(mgl::TextureMagFilter::Linear),
                         "glTexParameter with a sampler bound changed the TEXTURE's MAG_FILTER (glGetTexParameteriv = LINEAR)", std::format("{:#x}", texture_mag_after_texparameter_));
            check.expect(sampler_mag_after_texparameter_ == value_of(mgl::TextureMagFilter::Nearest),
                         "... and left the SAMPLER's MAG_FILTER alone (glGetSamplerParameteriv = NEAREST)", std::format("{:#x}", sampler_mag_after_texparameter_));
        }

        // Sampler state of a new sampler = the initial state of a texture (Table 6.10, 3.8.14).
        void check_sampler_defaults(demo::Check& check) const
        {
            mgl::SamplerId s{};
            mgl::glGenSamplers(1, &s);
            auto geti = [&](mgl::SamplerParameter p) {
                GLint v = 0;
                mgl::glGetSamplerParameteriv(s, p, &v);
                return v;
            };
            auto getf = [&](mgl::SamplerParameter p) {
                GLfloat v = 0;
                mgl::glGetSamplerParameterfv(s, p, &v);
                return v;
            };
            check.expect(geti(mgl::SamplerParameter::MinFilter) == GL_NEAREST_MIPMAP_LINEAR && geti(mgl::SamplerParameter::MagFilter) == GL_LINEAR,
                         "initial sampler filters: MIN = NEAREST_MIPMAP_LINEAR, MAG = LINEAR");
            check.expect(geti(mgl::SamplerParameter::WrapS) == GL_REPEAT && geti(mgl::SamplerParameter::WrapT) == GL_REPEAT && geti(mgl::SamplerParameter::WrapR) == GL_REPEAT,
                         "initial sampler wrap modes: S = T = R = REPEAT");
            check.expect(getf(mgl::SamplerParameter::MinLod) == -1000.0f && getf(mgl::SamplerParameter::MaxLod) == 1000.0f, "initial sampler LOD range: -1000 .. 1000");
            check.expect(geti(mgl::SamplerParameter::CompareMode) == GL_NONE && geti(mgl::SamplerParameter::CompareFunc) == GL_LEQUAL,
                         "initial sampler comparison: MODE = NONE, FUNC = LEQUAL");
            mgl::glDeleteSamplers(1, &s);
        }

        // Every setter form followed by the matching getter: i, f, iv, fv for filters, wraps, LOD range, comparison.
        void check_sampler_roundtrips(demo::Check& check) const
        {
            mgl::SamplerId s{};
            mgl::glGenSamplers(1, &s);
            auto geti = [&](mgl::SamplerParameter p) {
                GLint v = 0;
                mgl::glGetSamplerParameteriv(s, p, &v);
                return v;
            };
            auto getf = [&](mgl::SamplerParameter p) {
                GLfloat v = 0;
                mgl::glGetSamplerParameterfv(s, p, &v);
                return v;
            };
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::MinFilter, GL_LINEAR_MIPMAP_NEAREST);
            mgl::glSamplerParameterf(s, mgl::SamplerParameter::MagFilter, static_cast<GLfloat>(GL_NEAREST));
            check.expect(geti(mgl::SamplerParameter::MinFilter) == GL_LINEAR_MIPMAP_NEAREST && geti(mgl::SamplerParameter::MagFilter) == GL_NEAREST,
                         "glSamplerParameteri / glSamplerParameterf: MIN_FILTER = LINEAR_MIPMAP_NEAREST, MAG_FILTER = NEAREST read back with glGetSamplerParameteriv");
            check.expect_eq(getf(mgl::SamplerParameter::MagFilter), static_cast<GLfloat>(GL_NEAREST), "glGetSamplerParameterfv returns an enum parameter as a float");

            const GLint wrap_s = GL_MIRRORED_REPEAT;
            const GLfloat wrap_t = static_cast<GLfloat>(GL_CLAMP_TO_EDGE);
            mgl::glSamplerParameteriv(s, mgl::SamplerParameter::WrapS, &wrap_s);
            mgl::glSamplerParameterfv(s, mgl::SamplerParameter::WrapT, &wrap_t);
            mgl::glSamplerParameterf(s, mgl::SamplerParameter::WrapR, static_cast<GLfloat>(GL_CLAMP_TO_EDGE));
            check.expect(geti(mgl::SamplerParameter::WrapS) == GL_MIRRORED_REPEAT && geti(mgl::SamplerParameter::WrapT) == GL_CLAMP_TO_EDGE && geti(mgl::SamplerParameter::WrapR) == GL_CLAMP_TO_EDGE,
                         "glSamplerParameteriv / fv / f: WRAP_S = MIRRORED_REPEAT, WRAP_T and WRAP_R = CLAMP_TO_EDGE read back");

            mgl::glSamplerParameterf(s, mgl::SamplerParameter::MinLod, -3.25f);
            const GLfloat max_lod = 7.5f;
            mgl::glSamplerParameterfv(s, mgl::SamplerParameter::MaxLod, &max_lod);
            check.expect(getf(mgl::SamplerParameter::MinLod) == -3.25f && getf(mgl::SamplerParameter::MaxLod) == 7.5f, "glSamplerParameterf / fv: MIN_LOD = -3.25, MAX_LOD = 7.5 read back exactly");
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::MinLod, -2);
            check.expect_eq(getf(mgl::SamplerParameter::MinLod), -2.0f, "glSamplerParameteri on the float parameter MIN_LOD converts the integer (2.3.1)");

            const GLint compare_mode = GL_COMPARE_REF_TO_TEXTURE;
            mgl::glSamplerParameteriv(s, mgl::SamplerParameter::CompareMode, &compare_mode);
            mgl::glSamplerParameteri(s, mgl::SamplerParameter::CompareFunc, GL_GREATER);
            check.expect(geti(mgl::SamplerParameter::CompareMode) == GL_COMPARE_REF_TO_TEXTURE && geti(mgl::SamplerParameter::CompareFunc) == GL_GREATER,
                         "glSamplerParameteriv / i: COMPARE_MODE = COMPARE_REF_TO_TEXTURE, COMPARE_FUNC = GREATER read back");
            check.expect_no_gl_error("none of the setters and getters raised an error");
            mgl::glDeleteSamplers(1, &s);
        }

        void check_sampler_lifecycle(demo::Check& check) const
        {
            mgl::SamplerId s{};
            mgl::glGenSamplers(1, &s);
            // 3.8.2: a sampler acquires state when it is first used in BindSampler, SamplerParameter*,
            // GetSamplerParameter* or IsSampler; 6.1.5: IsSampler is TRUE for a name returned by GenSamplers.
            check.expect(mgl::glIsSampler(s), "glIsSampler is TRUE for a name returned by glGenSamplers (6.1.5)");
            check.expect(!mgl::glIsSampler(mgl::SamplerId{0}), "glIsSampler(0) is FALSE: zero is not a sampler");
            check.expect(!mgl::glIsSampler(mgl::SamplerId{0x7fff0001u}), "glIsSampler of a name that was never generated is FALSE");
            mgl::glDeleteSamplers(1, &s);
            check.expect(!mgl::glIsSampler(s), "glIsSampler is FALSE after glDeleteSamplers");
            mgl::glDeleteSamplers(1, &s);  // deleting it again is silently ignored (3.8.2: unused names are ignored)
            const mgl::SamplerId zero{0};
            mgl::glDeleteSamplers(1, &zero);
            check.expect_no_gl_error("deleting an unused name and the name zero is silently ignored");
        }

        // SAMPLER_BINDING is per texture unit and queried for the active unit.
        void check_sampler_bindings(demo::Check& check) const
        {
            std::array<mgl::SamplerId, 2> s{};
            mgl::glGenSamplers(2, s.data());
            mgl::glBindSampler(0, s[0]);
            mgl::glBindSampler(1, s[1]);
            mgl::glBindSampler(2, s[0]);  // one sampler may be bound to several units
            auto binding_of_unit = [](int unit) {
                mgl::glActiveTexture(kUnits[static_cast<std::size_t>(unit)]);
                return demo::get_integer(mgl::GetParameter::SamplerBinding);
            };
            const GLint b0 = binding_of_unit(0), b1 = binding_of_unit(1), b2 = binding_of_unit(2), b3 = binding_of_unit(3);
            check.expect(b0 == static_cast<GLint>(s[0].value) && b1 == static_cast<GLint>(s[1].value) && b2 == static_cast<GLint>(s[0].value) && b3 == 0,
                         "SAMPLER_BINDING per unit: unit 0 = A, unit 1 = B, unit 2 = A, unit 3 = none",
                         std::format("units 0..3 report {}, {}, {}, {}; A = {}, B = {}", b0, b1, b2, b3, s[0].value, s[1].value));
            mgl::glBindSampler(0, mgl::SamplerId{0});
            check.expect(binding_of_unit(0) == 0 && binding_of_unit(2) == static_cast<GLint>(s[0].value), "glBindSampler(0, 0) unbinds unit 0 only");
            mgl::glBindSampler(0, s[0]);
            mgl::glDeleteSamplers(1, &s[0]);  // bound to units 0 and 2
            check.expect(binding_of_unit(0) == 0 && binding_of_unit(2) == 0 && binding_of_unit(1) == static_cast<GLint>(s[1].value),
                         "deleting a sampler that is bound to two units unbinds both and leaves the other unit alone");
            mgl::glDeleteSamplers(1, &s[1]);
            check.expect(binding_of_unit(1) == 0, "deleting the sampler of unit 1 unbinds it");
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            check.expect_no_gl_error("no GL error from the binding queries");
        }

        template <typename Call>
        static void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, Call&& call)
        {
            demo::drain_gl_errors();
            call();
            check.expect_gl_error(code, what);
        }

        void check_errors(demo::Check& check) const
        {
            mgl::SamplerId s{};
            mgl::glGenSamplers(1, &s);
            mgl::SamplerId gone{};
            mgl::glGenSamplers(1, &gone);
            mgl::glDeleteSamplers(1, &gone);
            const mgl::SamplerId never{0x7fff0002u};
            GLint out = 0;
            const GLint max_units = demo::get_integer(mgl::GetParameter::MaxCombinedTextureImageUnits);

            expect_error(check, mgl::ErrorCode::InvalidOperation, "glBindSampler with a deleted sampler name -> INVALID_OPERATION (3.8.2)",
                         [&] { mgl::glBindSampler(0, gone); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glBindSampler with a name that was never generated -> INVALID_OPERATION",
                         [&] { mgl::glBindSampler(0, never); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "glBindSampler with unit = MAX_COMBINED_TEXTURE_IMAGE_UNITS -> INVALID_VALUE",
                         [&] { mgl::glBindSampler(static_cast<GLuint>(max_units), s); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glSamplerParameteri on sampler 0 -> INVALID_OPERATION (3.8.2)",
                         [&] { mgl::glSamplerParameteri(mgl::SamplerId{0}, mgl::SamplerParameter::MinFilter, GL_LINEAR); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glSamplerParameterf on a deleted sampler -> INVALID_OPERATION",
                         [&] { mgl::glSamplerParameterf(gone, mgl::SamplerParameter::MinLod, 1.0f); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glGetSamplerParameteriv on a name that was never generated -> INVALID_OPERATION (6.1.5)",
                         [&] { mgl::glGetSamplerParameteriv(never, mgl::SamplerParameter::MinFilter, &out); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glGetSamplerParameterfv on sampler 0 -> INVALID_OPERATION",
                         [&] { GLfloat f = 0; mgl::glGetSamplerParameterfv(mgl::SamplerId{0}, mgl::SamplerParameter::MinLod, &f); });

            // Illegal values for an enum parameter: Table 3.20 lists the legal ones; the error code is
            // INVALID_ENUM (ES 3.2 8.10 states it; ES 3.0 refers to TexParameter*).
            expect_error(check, mgl::ErrorCode::InvalidEnum, "MAG_FILTER = NEAREST_MIPMAP_NEAREST (not a magnification filter) -> INVALID_ENUM",
                         [&] { mgl::glSamplerParameteri(s, mgl::SamplerParameter::MagFilter, GL_NEAREST_MIPMAP_NEAREST); });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "WRAP_S = LINEAR -> INVALID_ENUM",
                         [&] { mgl::glSamplerParameteri(s, mgl::SamplerParameter::WrapS, GL_LINEAR); });
            expect_error(check, mgl::ErrorCode::InvalidEnum, "COMPARE_FUNC = REPEAT -> INVALID_ENUM",
                         [&] { mgl::glSamplerParameteri(s, mgl::SamplerParameter::CompareFunc, GL_REPEAT); });
            GLint kept = 0;
            mgl::glGetSamplerParameteriv(s, mgl::SamplerParameter::MagFilter, &kept);
            check.expect_eq(kept, static_cast<GLint>(GL_LINEAR), "a rejected glSamplerParameteri leaves the parameter unchanged (MAG_FILTER still LINEAR)");
            mgl::glDeleteSamplers(1, &s);
        }

        mgl::ProgramId program_{};
        mgl::UniformLocation u_tex_a_{}, u_tex_b_{}, u_scale_{}, u_offset_{}, u_split_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        std::array<mgl::SamplerId, 9> samplers_{};
        std::vector<Rgba8> texels_;

        GLuint sampler_name_while_bound_ = 0;
        GLint sampler_binding_while_bound_ = -1;
        GLint sampler_binding_after_delete_ = -1;
        bool is_sampler_after_delete_ = true;
        GLint texture_mag_after_texparameter_ = 0;
        GLint sampler_mag_after_texparameter_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-sampler-objects",
        .title = "Sampler objects: sampling state separate from textures",
        .min_es = {3, 0},
        .summary = "glGenSamplers/glBindSampler/glSamplerParameter*: a sampler overrides the texture's own filters, wrap modes and LOD range; "
                   "one texture on two units with two samplers; queries, life cycle and errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<SamplerObjects>(argc, argv, info);
}
