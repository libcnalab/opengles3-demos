// es20-textures-2d
//
// 2D texture objects, units, filtering, wrapping, sub-image updates, unpack alignment, the
// unsized ES 2.0 formats (LUMINANCE, LUMINANCE_ALPHA, ALPHA) with packed 16-bit types, and the
// result of sampling an incomplete texture. Sixteen cells, each sampling one configuration;
// every probe is compared with a CPU evaluation of the ES 3.0 §3.8.10 sampling equations (LINEAR only
// against the bound those equations guarantee: the min/max of the contributing texels).
//
//   row 3: RGBA4444 | RGBA5551 | two units  | incomplete
//   row 2: LUMIN.   | LUM.ALPHA| ALPHA      | RGB565
//   row 1: CLAMP    | SubImage | RGB align4 | RGB align1
//   row 0: NEAREST  | LINEAR   | REPEAT     | MIRRORED_REPEAT        (row 0 is at the bottom)

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kBase = 4;  // base texture is 4x4

    Rgba8 base_texel(int i, int j)
    {
        return {static_cast<std::uint8_t>(40 + 60 * i), static_cast<std::uint8_t>(40 + 60 * j),
                static_cast<std::uint8_t>((i + j) % 2 ? 220 : 30), 255};
    }

    // --- CPU reference of ES 3.0 §3.8.10.2 (Table 3.22 wrap, eq. 3.22 in its 2D form) ----------
    int wrap(int coord, int size, mgl::TextureWrapMode mode)
    {
        auto fmod_i = [](int a, int b) { return ((a % b) + b) % b; };
        switch (mode)
        {
        case mgl::TextureWrapMode::Repeat: return fmod_i(coord, size);
        case mgl::TextureWrapMode::MirroredRepeat:
        {
            const int m = fmod_i(coord, 2 * size) - size;
            return (size - 1) - (m >= 0 ? m : -(1 + m));
        }
        default: return std::clamp(coord, 0, size - 1);  // CLAMP_TO_EDGE
        }
    }

    std::array<double, 4> sample_ref(const std::vector<Rgba8>& texels, int w, int h, bool linear,
                                     mgl::TextureWrapMode wrap_mode, double u, double v)
    {
        const double us = u * w, vs = v * h;
        auto at = [&](int i, int j) {
            const Rgba8 t = texels[static_cast<std::size_t>(wrap(j, h, wrap_mode) * w + wrap(i, w, wrap_mode))];
            return std::array<double, 4>{double(t[0]), double(t[1]), double(t[2]), double(t[3])};
        };
        if (!linear) return at(static_cast<int>(std::floor(us)), static_cast<int>(std::floor(vs)));
        const int i0 = static_cast<int>(std::floor(us - 0.5)), j0 = static_cast<int>(std::floor(vs - 0.5));
        const double a = (us - 0.5) - std::floor(us - 0.5), b = (vs - 0.5) - std::floor(vs - 0.5);
        std::array<double, 4> r{};
        const auto t00 = at(i0, j0), t10 = at(i0 + 1, j0), t01 = at(i0, j0 + 1), t11 = at(i0 + 1, j0 + 1);
        for (int c = 0; c < 4; ++c)
            r[c] = (1 - a) * (1 - b) * t00[c] + a * (1 - b) * t10[c] + (1 - a) * b * t01[c] + a * b * t11[c];
        return r;
    }

    std::uint8_t unorm(int value, int bits) { return static_cast<std::uint8_t>(std::lround(value * 255.0 / ((1 << bits) - 1))); }

    struct Cell
    {
        mgl::TextureId tex;
        bool linear = false;
        mgl::TextureWrapMode wrap = mgl::TextureWrapMode::ClampToEdge;
        float uv_scale = 1.0f, uv_offset = 0.0f;  // passed to glUniform2f
        bool split = false;
    };

    class Textures2D final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            program_ = demo::build_program(env, "shaders/quad.vert", "shaders/textured.frag");
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());

            // --- base 4x4 RGBA8 texture -----------------------------------------------------
            std::vector<Rgba8> base;
            for (int j = 0; j < kBase; ++j)
                for (int i = 0; i < kBase; ++i) base.push_back(base_texel(i, j));
            base_ = base;
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kBaseTex]);
            // Unsized internal format RGBA (ES 2.0 style): format/type define the texel layout.
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, kBase, kBase, 0,
                              mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, base.data());

            // --- same texels, then a 2x2 glTexSubImage2D patch of white at texel (1,1) -------------
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kSubTex]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, kBase, kBase, 0,
                              mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, base.data());
            const std::array<Rgba8, 4> white = {{{255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}}};
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 1, 1, 2, 2, mgl::PixelFormat::Rgba,
                                 mgl::PixelType::UnsignedByte, white.data());

            // --- RGB 3x2 with UNPACK_ALIGNMENT 4: each 9-byte row is padded to 12 bytes ----------
            const std::uint8_t pad = 0xEE;  // padding bytes the GL must skip
            const std::uint8_t rgb_padded[24] = {255, 0, 0, 0, 255, 0, 0, 0, 255, pad, pad, pad,
                                                 255, 255, 0, 0, 255, 255, 255, 0, 255, pad, pad, pad};
            const std::uint8_t rgb_tight[18] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 255};
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);  // the initial value
            unpack_alignment_seen_[0] = demo::get_integer(mgl::GetParameter::UnpackAlignment);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kRgbAlign4]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgb, 3, 2, 0, mgl::PixelFormat::Rgb,
                              mgl::PixelType::UnsignedByte, rgb_padded);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 1);  // rows are byte-aligned: no padding
            unpack_alignment_seen_[1] = demo::get_integer(mgl::GetParameter::UnpackAlignment);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kRgbAlign1]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgb, 3, 2, 0, mgl::PixelFormat::Rgb,
                              mgl::PixelType::UnsignedByte, rgb_tight);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);

            // --- legacy unsized formats (ES 3.0 Table 3.24: how they appear to the shader) --------
            const std::uint8_t lum[2] = {200, 50};
            upload(kLum, mgl::InternalFormat::Luminance, 2, 1, mgl::PixelFormat::Luminance, mgl::PixelType::UnsignedByte, lum);
            const std::uint8_t lum_alpha[2] = {100, 180};
            upload(kLumAlpha, mgl::InternalFormat::LuminanceAlpha, 1, 1, mgl::PixelFormat::LuminanceAlpha, mgl::PixelType::UnsignedByte, lum_alpha);
            const std::uint8_t alpha[1] = {77};
            upload(kAlpha, mgl::InternalFormat::Alpha, 1, 1, mgl::PixelFormat::Alpha, mgl::PixelType::UnsignedByte, alpha);

            // --- packed 16-bit texel types: bit fields are unsigned normalized ---------------------
            const std::uint16_t p565 = static_cast<std::uint16_t>((31 << 11) | (32 << 5) | 10);
            upload(kRgb565, mgl::InternalFormat::Rgb, 1, 1, mgl::PixelFormat::Rgb, mgl::PixelType::UnsignedShort565, &p565);
            const std::uint16_t p4444 = static_cast<std::uint16_t>((15 << 12) | (8 << 8) | (3 << 4) | 12);
            upload(kRgba4444, mgl::InternalFormat::Rgba, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedShort4444, &p4444);
            const std::uint16_t p5551 = static_cast<std::uint16_t>((0 << 11) | (31 << 6) | (16 << 1) | 1);
            upload(kRgba5551, mgl::InternalFormat::Rgba, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedShort5551, &p5551);

            // --- a 1x1 magenta texture that will live on texture unit 3 ---------------------------
            const Rgba8 magenta = {255, 0, 255, 255};
            upload(kUnit3, mgl::InternalFormat::Rgba, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, &magenta);

            // --- incomplete: a 2x2 level 0 without level 1, while the default MIN_FILTER
            //     (NEAREST_MIPMAP_LINEAR) requires the full chain. (A 1x1 texture would be complete:
            //     its chain is just level 0.)
            const std::array<Rgba8, 4> magenta4 = {magenta, magenta, magenta, magenta};
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kIncomplete]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, 2, 2, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, magenta4.data());

            // Bind the magenta texture on unit 3 and leave unit 0 active for everything else.
            mgl::glActiveTexture(mgl::TextureUnit::Texture3);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kUnit3]);
            active_texture_seen_ = demo::get_integer(mgl::GetParameter::ActiveTexture);
            binding_unit3_seen_ = demo::get_integer(mgl::GetParameter::TextureBinding2D);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);

            using W = mgl::TextureWrapMode;
            cells_ = {
                Cell{tex_[kBaseTex], false, W::ClampToEdge},               // 0 NEAREST
                Cell{tex_[kBaseTex], true, W::ClampToEdge},                // 1 LINEAR
                Cell{tex_[kBaseTex], false, W::Repeat, 3.0f, -1.0f},       // 2 REPEAT over [-1, 2]
                Cell{tex_[kBaseTex], false, W::MirroredRepeat, 3.0f, -1.0f},
                Cell{tex_[kBaseTex], false, W::ClampToEdge, 3.0f, -1.0f},  // 4 CLAMP_TO_EDGE over [-1, 2]
                Cell{tex_[kSubTex]},
                Cell{tex_[kRgbAlign4]},
                Cell{tex_[kRgbAlign1]},
                Cell{tex_[kLum]},
                Cell{tex_[kLumAlpha]},
                Cell{tex_[kAlpha]},
                Cell{tex_[kRgb565]},
                Cell{tex_[kRgba4444]},
                Cell{tex_[kRgba5551]},
                Cell{tex_[kBaseTex], false, W::ClampToEdge, 1.0f, 0.0f, true},  // 14 unit 0 | unit 3
                Cell{tex_[kIncomplete]},
            };
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(program_);
            const mgl::UniformLocation scale = mgl::glGetUniformLocation(program_, "u_uv_scale");
            const mgl::UniformLocation offset = mgl::glGetUniformLocation(program_, "u_uv_offset");
            const mgl::UniformLocation tex_a = mgl::glGetUniformLocation(program_, "u_tex_a");
            const mgl::UniformLocation tex_b = mgl::glGetUniformLocation(program_, "u_tex_b");
            const mgl::UniformLocation split = mgl::glGetUniformLocation(program_, "u_split");
            mgl::glUniform1i(tex_a, 0);  // sampler uniforms hold a texture UNIT index, not a texture name

            const int cw = env.width / 4, ch = env.height / 4;
            for (std::size_t c = 0; c < cells_.size(); ++c)
            {
                const Cell& cell = cells_[c];
                mgl::glViewport(static_cast<GLint>(c % 4) * cw, static_cast<GLint>(c / 4) * ch, cw, ch);
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, cell.tex);  // on unit 0
                if (c != kIncompleteCell) set_sampling(cell.linear, cell.wrap);
                mgl::glUniform2f(scale, cell.uv_scale, cell.uv_scale);
                mgl::glUniform2f(offset, cell.uv_offset, cell.uv_offset);
                mgl::glUniform1i(tex_b, cell.split ? 3 : 0);
                mgl::glUniform1i(split, cell.split ? 1 : 0);
                mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            }
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            const int cw = env.width / 4, ch = env.height / 4;
            auto origin = [&](int c) { return std::array<int, 2>{(c % 4) * cw, (c / 4) * ch}; };

            // Cells 0-3 (NEAREST, REPEAT, MIRRORED_REPEAT, CLAMP_TO_EDGE): every probe away from a texel boundary must be
            // exactly the texel that the CPU sampler selects.
            static const char* names[] = {"NEAREST", "LINEAR", "REPEAT", "MIRRORED_REPEAT", "CLAMP_TO_EDGE"};
            for (int c = 0; c <= 4; ++c)
            {
                if (c == 1) continue;  // LINEAR: see below
                const Cell& cell = cells_[static_cast<std::size_t>(c)];
                int probes = 0, bad = 0;
                std::string first_bad;
                for (int ly = 2; ly < ch; ly += 4)
                    for (int lx = 2; lx < cw; lx += 4)
                    {
                        const double u = double(cell.uv_offset) + double(cell.uv_scale) * (lx + 0.5) / cw;
                        const double v = double(cell.uv_offset) + double(cell.uv_scale) * (ly + 0.5) / ch;
                        if (near_edge(u * kBase) || near_edge(v * kBase)) continue;  // texel boundary
                        const auto ref = sample_ref(base_, kBase, kBase, false, cell.wrap, u, v);
                        const Rgba8 got = demo::read_pixel(origin(c)[0] + lx, origin(c)[1] + ly);
                        ++probes;
                        bool ok = true;
                        for (int k = 0; k < 4; ++k) ok = ok && std::abs(got[k] - ref[k]) <= 0.5;
                        if (!ok && bad++ == 0)
                            first_bad = std::format("at uv=({:.3f},{:.3f}) got {} ref ({:.1f},{:.1f},{:.1f},{:.1f})", u, v,
                                                    demo::format_rgba(got), ref[0], ref[1], ref[2], ref[3]);
                    }
                check.expect(probes > 10 && bad == 0, std::format("cell {}: {} matches the CPU sampler exactly at {} probes", c, names[c], probes),
                             first_bad);
            }

            // Cell 1 (LINEAR). ES 3.0 section 3.8.10.2 defines the result as the bilinear sum of the 2x2 texels
            // around (u - 0.5, v - 0.5) with weights built from alpha = frac(u - 0.5), beta = frac(v - 0.5), but gives no
            // precision for the weights. What the equation does guarantee for any alpha, beta in [0, 1): the weights
            // (1-a)(1-b), a(1-b), (1-a)b, ab are non-negative and sum to one, so every channel lies between the minimum
            // and maximum of the contributing texels. That bound is the check. The texel set is taken over
            // u, v +- 0.1 texel (the same coordinate margin that the NEAREST probes keep from texel boundaries), and
            // +-1 is allowed for the conversion to 8 bits, which may return either of the two closest integers
            // (section 2.1.6.2). The deviation from the ideal equation is printed, not checked.
            {
                const Cell& cell = cells_[1];
                constexpr double kMargin = 0.1;
                int probes = 0, bad = 0, rejected_shifted = 0;
                double max_err = 0.0;
                std::string first_bad;
                auto hull = [&](double u, double v) {
                    std::array<double, 4> lo{255, 255, 255, 255}, hi{0, 0, 0, 0};
                    const double us = u * kBase - 0.5, vs = v * kBase - 0.5;
                    for (int j = static_cast<int>(std::floor(vs - kMargin)); j <= static_cast<int>(std::floor(vs + kMargin)) + 1; ++j)
                        for (int i = static_cast<int>(std::floor(us - kMargin)); i <= static_cast<int>(std::floor(us + kMargin)) + 1; ++i)
                        {
                            const Rgba8 t = base_[static_cast<std::size_t>(wrap(j, kBase, cell.wrap) * kBase + wrap(i, kBase, cell.wrap))];
                            for (std::size_t k = 0; k < 4; ++k)
                            {
                                lo[k] = std::min(lo[k], double(t[k]) - 1.0);
                                hi[k] = std::max(hi[k], double(t[k]) + 1.0);
                            }
                        }
                    return std::array<std::array<double, 4>, 2>{lo, hi};
                };
                auto inside = [](const auto& h, const auto& val) {
                    for (std::size_t k = 0; k < 4; ++k)
                        if (val[k] < h[0][k] || val[k] > h[1][k]) return false;
                    return true;
                };
                for (int ly = 2; ly < ch; ly += 8)
                    for (int lx = 2; lx < cw; lx += 8)
                    {
                        const double u = (lx + 0.5) / cw, v = (ly + 0.5) / ch;
                        const auto h = hull(u, v);
                        const auto ref = sample_ref(base_, kBase, kBase, true, cell.wrap, u, v);
                        const Rgba8 got = demo::read_pixel(origin(1)[0] + lx, origin(1)[1] + ly);
                        ++probes;
                        const std::array<double, 4> g{double(got[0]), double(got[1]), double(got[2]), double(got[3])};
                        for (std::size_t k = 0; k < 4; ++k) max_err = std::max(max_err, std::abs(g[k] - ref[k]));
                        if (!inside(h, g) && bad++ == 0)
                            first_bad = std::format("at uv=({:.3f},{:.3f}) got {} outside [{:.0f}..{:.0f}, {:.0f}..{:.0f}, {:.0f}..{:.0f}, {:.0f}..{:.0f}]", u, v,
                                                    demo::format_rgba(got), h[0][0], h[1][0], h[0][1], h[1][1], h[0][2], h[1][2], h[0][3], h[1][3]);
                        // Mutation: the value a sampler two texels off would return must fall outside the bound often.
                        if (!inside(h, sample_ref(base_, kBase, kBase, true, cell.wrap, u + 0.5, v + 0.5))) ++rejected_shifted;
                    }
                check.expect(probes > 10 && bad == 0,
                             std::format("cell 1: LINEAR stays within the min/max of the contributing 2x2 texels (+-1) at {} probes "
                                         "(eq. 3.22 weights are convex)", probes),
                             first_bad);
                check.expect(rejected_shifted * 4 > probes,
                             "negative control: the LINEAR bound rejects the value sampled two texels off at more than a quarter of the probes",
                             std::format("{} of {} rejected", rejected_shifted, probes));
                std::cout << std::format("   note: LINEAR cell: largest deviation from the ideal bilinear equation {:.1f}/255 over {} probes "
                                         "(observation; section 3.8.10.2 gives no weight precision)\n",
                                         max_err, probes);
            }

            // Cell 5: the sub-image patch replaced exactly texels (1..2, 1..2).
            {
                int bad = 0;
                for (int j = 0; j < kBase; ++j)
                    for (int i = 0; i < kBase; ++i)
                    {
                        const bool patched = i >= 1 && i <= 2 && j >= 1 && j <= 2;
                        const Rgba8 want = patched ? Rgba8{255, 255, 255, 255} : base_texel(i, j);
                        if (demo::read_pixel(origin(5)[0] + i * cw / 4 + cw / 8, origin(5)[1] + j * ch / 4 + ch / 8) != want) ++bad;
                    }
                check.expect(bad == 0, "cell 5: glTexSubImage2D replaced exactly the 2x2 patch", std::format("{} texels wrong", bad));
            }

            // Cells 6/7: identical texels from a padded (alignment 4) and a tight (alignment 1) upload.
            const Rgba8 rgb[6] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}, {0, 255, 255, 255}, {255, 0, 255, 255}};
            for (int c = 6; c <= 7; ++c)
            {
                int bad = 0;
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 3; ++i)
                        if (demo::read_pixel(origin(c)[0] + (2 * i + 1) * cw / 6, origin(c)[1] + (2 * j + 1) * ch / 4) != rgb[j * 3 + i]) ++bad;
                check.expect(bad == 0, std::format("cell {}: RGB 3x2 upload with UNPACK_ALIGNMENT {} gives the intended texels", c, c == 6 ? 4 : 1),
                             std::format("{} texels wrong", bad));
            }
            check.expect(unpack_alignment_seen_[0] == 4 && unpack_alignment_seen_[1] == 1,
                         "glGetIntegerv(GL_UNPACK_ALIGNMENT) reflects glPixelStorei (meta-gl gap G1 fix)",
                         std::format("read {} then {}", unpack_alignment_seen_[0], unpack_alignment_seen_[1]));

            // Cells 8-13: legacy unsized formats and packed types.
            auto center = [&](int c) { return std::array<int, 2>{origin(c)[0] + cw / 4, origin(c)[1] + ch / 2}; };
            check.expect_pixel(center(8)[0], center(8)[1], {200, 200, 200, 255}, 0, "cell 8: LUMINANCE L -> (L, L, L, 1)");
            check.expect_pixel(center(9)[0], center(9)[1], {100, 100, 100, 180}, 0, "cell 9: LUMINANCE_ALPHA -> (L, L, L, A)");
            check.expect_pixel(center(10)[0], center(10)[1], {0, 0, 0, 77}, 0, "cell 10: ALPHA A -> (0, 0, 0, A)");
            check.expect_pixel(center(11)[0], center(11)[1], {unorm(31, 5), unorm(32, 6), unorm(10, 5), 255}, 1,
                               "cell 11: UNSIGNED_SHORT_5_6_5 fields are unorm (c / (2^n - 1))");
            check.expect_pixel(center(12)[0], center(12)[1], {unorm(15, 4), unorm(8, 4), unorm(3, 4), unorm(12, 4)}, 1,
                               "cell 12: UNSIGNED_SHORT_4_4_4_4");
            check.expect_pixel(center(13)[0], center(13)[1], {unorm(0, 5), unorm(31, 5), unorm(16, 5), 255}, 1,
                               "cell 13: UNSIGNED_SHORT_5_5_5_1");
            const int lum_texel_1_x = origin(8)[0] + 3 * cw / 4;
            check.expect_pixel(lum_texel_1_x, center(8)[1], {50, 50, 50, 255}, 0, "cell 8: second LUMINANCE texel");

            // Cell 14: left half from unit 0, right half from unit 3.
            check.expect_pixel(origin(14)[0] + cw / 8, origin(14)[1] + ch / 8, base_texel(0, 0), 0, "cell 14: left half samples unit 0");
            check.expect_pixel(origin(14)[0] + 7 * cw / 8, origin(14)[1] + ch / 2, {255, 0, 255, 255}, 0,
                               "cell 14: right half samples the texture bound to unit 3");
            check.expect_eq(active_texture_seen_, static_cast<int>(GL_TEXTURE3), "GL_ACTIVE_TEXTURE = GL_TEXTURE3 after glActiveTexture");
            check.expect_eq(binding_unit3_seen_, static_cast<int>(tex_[kUnit3].value), "GL_TEXTURE_BINDING_2D is per unit (unit 3)");

            // Cell 15: an incomplete texture samples as (0, 0, 0, 1) (ES 3.0 §3.8.13, §3.9.2).
            check.expect_pixel(center(15)[0] + cw / 4, center(15)[1], {0, 0, 0, 255}, 0,
                               "cell 15: sampling an incomplete texture returns (0, 0, 0, 1)");

            check_parameter_queries(check);
            check_errors(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(program_);
        }

    private:
        static bool near_edge(double texel_coord)
        {
            const double f = texel_coord - std::floor(texel_coord);
            return f < 0.1 || f > 0.9;
        }

        // Texture parameters are texture-object state in ES 2.0 (sampler objects come in ES 3.0).
        // The four setter variants are used on purpose: each takes the enum value in a different form.
        static void set_sampling(bool linear, mgl::TextureWrapMode wrap_mode)
        {
            const GLenum filter = linear ? GL_LINEAR : GL_NEAREST;
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinFilter, static_cast<GLint>(filter));
            mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MagFilter, static_cast<GLfloat>(filter));
            const GLint wrap_i = static_cast<GLint>(wrap_mode);
            const GLfloat wrap_f = static_cast<GLfloat>(wrap_i);
            mgl::glTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::WrapS, &wrap_i);
            mgl::glTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::WrapT, &wrap_f);
        }

        void upload(std::size_t slot, mgl::InternalFormat internal, int w, int h, mgl::PixelFormat format, mgl::PixelType type,
                    const void* data)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[slot]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, internal, w, h, 0, format, type, data);
        }

        void check_parameter_queries(demo::Check& check)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kBaseTex]);
            set_sampling(true, mgl::TextureWrapMode::MirroredRepeat);
            GLint min = 0, mag = 0, ws = 0, wt = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MinFilter, &min);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MagFilter, &mag);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::WrapS, &ws);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::WrapT, &wt);
            check.expect(min == GL_LINEAR && mag == GL_LINEAR && ws == GL_MIRRORED_REPEAT && wt == GL_MIRRORED_REPEAT,
                         "glGetTexParameteriv returns what glTexParameter{i,f,iv,fv} set",
                         std::format("min={:#x} mag={:#x} wrap=({:#x},{:#x})", min, mag, ws, wt));
            GLfloat magf = 0.0f;
            mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MagFilter, &magf);
            check.expect_eq(magf, static_cast<GLfloat>(GL_LINEAR), "glGetTexParameterfv returns the enum as a float");

            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kIncomplete]);
            GLint default_min = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, mgl::TextureParameterQuery::MinFilter, &default_min);
            check.expect_eq(default_min, static_cast<GLint>(GL_NEAREST_MIPMAP_LINEAR), "default TEXTURE_MIN_FILTER is NEAREST_MIPMAP_LINEAR");

            mgl::TextureId scratch{};
            mgl::glGenTextures(1, &scratch);
            check.expect(!mgl::glIsTexture(scratch), "a generated but never bound name is not yet a texture (glIsTexture = FALSE)");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, scratch);
            check.expect(mgl::glIsTexture(scratch), "binding creates the texture object");
            mgl::glDeleteTextures(1, &scratch);
            check.expect(!mgl::glIsTexture(scratch), "glDeleteTextures deletes it");
            check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBinding2D), 0, "deleting the bound texture reverts the binding to 0");
        }

        void check_errors(demo::Check& check)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kBaseTex]);
            const Rgba8 px = {0, 0, 0, 0};
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, 1, 1, 1, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, &px);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexImage2D with border = 1 -> INVALID_VALUE (border must be 0)");
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba, -1, 1, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, &px);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexImage2D with a negative width -> INVALID_VALUE");
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 3, 3, 2, 2, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, &px);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexSubImage2D outside the level -> INVALID_VALUE");
            mgl::glBindTexture(mgl::TextureTarget::TextureCubeMap, tex_[kBaseTex]);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "binding a 2D texture to TEXTURE_CUBE_MAP -> INVALID_OPERATION");
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 3);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "UNPACK_ALIGNMENT 3 -> INVALID_VALUE (1, 2, 4, 8 only)");
        }

        enum : std::size_t { kBaseTex, kSubTex, kRgbAlign4, kRgbAlign1, kLum, kLumAlpha, kAlpha, kRgb565, kRgba4444, kRgba5551, kUnit3, kIncomplete, kCount };
        static constexpr std::size_t kIncompleteCell = 15;

        mgl::ProgramId program_{};
        std::array<mgl::TextureId, kCount> tex_{};
        std::vector<Rgba8> base_;
        std::vector<Cell> cells_;
        int unpack_alignment_seen_[2] = {0, 0};
        int active_texture_seen_ = 0;
        int binding_unit3_seen_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-textures-2d",
        .title = "2D textures: filtering, wrapping, formats, units",
        .min_es = {3, 0},
        .summary = "Texture objects and units, NEAREST/LINEAR, REPEAT/MIRRORED_REPEAT/CLAMP_TO_EDGE, sub-image updates, "
                   "unpack alignment, LUMINANCE/ALPHA and packed 16-bit formats, incomplete textures.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Textures2D>(argc, argv, info);
}
