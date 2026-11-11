// es20-mipmaps
//
// Mipmapping (OpenGL ES 3.0 sections 3.8.10.1 level of detail, 3.8.10.4 mipmapping, 3.8.10.5
// glGenerateMipmap, 3.8.13 texture completeness) with every level uploaded by hand so that each
// level has a recognisable content.
//
// Every cell of the 8 x 15 image is a "strip": a viewport-sized quad whose s coordinate grows
// linearly with x and whose t coordinate is constant (so all rows of a cell are identical). For such a quad the scale
// factor rho of equation 3.20 is exactly |du/dx| (the other derivatives are zero), so the level of
// detail lambda = log2(rho) of a strip is a number chosen by the demo, independent of how an
// implementation approximates rho for general quads. Cell (row, column k) is drawn at lambda = k
// (or k + 0.25, k + 0.5, k + 0.75, see below).
//
//   row  0  ladder   NEAREST_MIPMAP_NEAREST   lambda = k          level colours, exactly
//   row  1  ladder   LINEAR_MIPMAP_NEAREST    lambda = k
//   row  2  ladder   NEAREST_MIPMAP_LINEAR    lambda = k + 0.5    50/50 blend of two level colours
//   row  3  ladder   LINEAR_MIPMAP_LINEAR     lambda = k + 0.5
//   row  4  ladder   NEAREST_MIPMAP_NEAREST   lambda = k + 0.25   rounds down (equation 3.23)
//   row  5  ladder   NEAREST_MIPMAP_NEAREST   lambda = k + 0.75   rounds up
//   row  6  stripes  NEAREST_MIPMAP_NEAREST   lambda = k          texel pattern, CPU sampler
//   row  7  stripes  LINEAR_MIPMAP_NEAREST    lambda = k
//   row  8  stripes  NEAREST_MIPMAP_LINEAR    lambda = k + 0.5
//   row  9  stripes  LINEAR_MIPMAP_LINEAR     lambda = k + 0.5
//   row 10  checker  glGenerateMipmap'ed 2x2 checker, NEAREST_MIPMAP_NEAREST, lambda = k
//   row 11  completeness pairs: [incomplete | complete twin] x 4
//   row 12  completeness pairs x 3
//   row 13  TEXTURE_BASE_LEVEL / TEXTURE_MAX_LEVEL
//   row 14  TEXTURE_MIN_LOD / TEXTURE_MAX_LOD
//
// "ladder" is a 64x64 texture whose seven levels (64 ... 1) are solid, distinct colours. "stripes"
// has seven levels whose texels alternate between six colours (a different phase per level), so
// that NEAREST and LINEAR filtering inside a level give different results.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kCols = 8;
    constexpr int kRows = 15;
    constexpr int kBaseSize = 64;  // level 0 of the ladder / stripes / checker textures
    constexpr int kLevels = 7;     // 64, 32, 16, 8, 4, 2, 1
    constexpr int kUnbounded = 1000;  // initial TEXTURE_MAX_LEVEL; |initial TEXTURE_MIN/MAX_LOD|

    constexpr std::array<Rgba8, kLevels> kLadder = {{
        {230, 40, 40, 255},   // level 0: red
        {240, 150, 30, 255},  // level 1: orange
        {230, 220, 40, 255},  // level 2: yellow
        {40, 200, 60, 255},   // level 3: green
        {40, 200, 210, 255},  // level 4: cyan
        {50, 80, 230, 255},   // level 5: blue
        {200, 50, 200, 255},  // level 6: magenta
    }};
    constexpr std::array<Rgba8, 6> kPalette = {{
        {230, 40, 40, 255}, {240, 170, 30, 255}, {80, 200, 60, 255},
        {40, 190, 210, 255}, {70, 80, 230, 255}, {200, 60, 200, 255},
    }};
    constexpr Rgba8 kBlack = {0, 0, 0, 255};                 // sampling an incomplete texture: (0, 0, 0, 1)
    constexpr Rgba8 kChecker[2] = {{240, 200, 40, 255}, {40, 90, 220, 255}};
    constexpr Rgba8 kUniform = {90, 160, 220, 255};          // base colour of the glGenerateMipmap tests
    constexpr Rgba8 kBogus = {255, 0, 0, 255};               // pre-existing content glGenerateMipmap must replace
    constexpr Rgba8 kBogus2 = {0, 255, 0, 255};              // other pre-existing content (the result must not depend on it)

    // ---------------------------------------------------------------------------------------------
    // What is checked, and what is only measured.
    //   * Level selection. For a strip, du/dy = dv/dx = dv/dy = 0, so condition 2 of section 3.8.10.1
    //     (max{mu, mv, mw} <= f <= mu + mv + mw) pins the approximated scale factor to f = |du/dx|: rho,
    //     and with it lambda = log2(rho), is fixed even for an implementation that approximates rho.
    //     Every strip keeps lambda at least kLodMargin = 1/4 away from each boundary at which the selected
    //     level(s) would change (integer + 1/2 for *_MIPMAP_NEAREST, integers for *_MIPMAP_LINEAR), so
    //     the level choice is robust and checked exactly.
    //   * NEAREST inside a level returns one texel: exact, away from texel boundaries (0.1 texel margin).
    //   * LINEAR inside a level. Section 3.8.10.2 gives the bilinear equation but no precision for its
    //     weights. The weights are non-negative and sum to one for any alpha in [0, 1), so the result
    //     lies between the minimum and maximum of the contributing texels (taken over u +- 0.1 texel,
    //     the same coordinate margin as for NEAREST). That bound is checked.
    //   * Blending two levels (*_MIPMAP_LINEAR). tau = (1 - frac(lambda)) tau1 + frac(lambda) tau2. The
    //     levels are fixed (above); the weight frac(lambda) has no specified precision. The check is a
    //     sanity bound that uses the same margin as the level selection: frac(lambda) within kLodMargin
    //     of its ideal value 1/2 (a lambda error of 1/4 would already break the k + 0.25 / k + 0.75
    //     level-selection rows).
    //   * Plus one 8-bit step whenever a result is a weighted sum: the conversion to the framebuffer
    //     returns either of the two closest integers (section 2.1.6.2).
    // The deviation from the ideal equations (filter weights, blend weight) is printed as an observation.
    // ---------------------------------------------------------------------------------------------
    constexpr double kLodMargin = 0.25;
    constexpr double kTexelMargin = 0.1;

    // ---------------------------------------------------------------------------------------------
    // CPU model of ES 3.0 sections 3.8.10.1-3.8.10.4. All textures used with the model have rows
    // that are identical, so a level is described by one row of texels (its "column" of colours) and
    // the t coordinate plays no role.
    // ---------------------------------------------------------------------------------------------
    using Column = std::vector<Rgba8>;
    struct Model
    {
        std::vector<Column> levels;  // levels[k] = texels of level k (width = 2^(6-k) for the 64x64 textures)
    };

    bool mipmapped(mgl::TextureMinFilter f)
    {
        return f != mgl::TextureMinFilter::Nearest && f != mgl::TextureMinFilter::Linear;
    }
    bool linear_in_level(mgl::TextureMinFilter f)
    {
        return f == mgl::TextureMinFilter::Linear || f == mgl::TextureMinFilter::LinearMipmapNearest ||
               f == mgl::TextureMinFilter::LinearMipmapLinear;
    }
    bool blends_levels(mgl::TextureMinFilter f)
    {
        return f == mgl::TextureMinFilter::NearestMipmapLinear || f == mgl::TextureMinFilter::LinearMipmapLinear;
    }
    int floor_log2(int v)
    {
        int r = 0;
        while (v > 1) { v >>= 1; ++r; }
        return r;
    }

    struct Fetched
    {
        std::array<double, 4> value{};     // ideal value of the equations
        std::array<double, 4> lo{}, hi{};  // guaranteed range: min / max of the texels that can contribute
        bool ambiguous = false;            // NEAREST within 0.1 texel of a texel boundary between different texels
    };

    /// One texel fetch inside a level (CLAMP_TO_EDGE; Table 3.22): NEAREST takes texel floor(u),
    /// LINEAR blends texels floor(u - 0.5) and +1 with weight frac(u - 0.5) (section 3.8.10.2).
    Fetched fetch(const Column& col, bool linear, double u)
    {
        const int w = static_cast<int>(col.size());
        auto at = [&](int i) { return col[static_cast<std::size_t>(std::clamp(i, 0, w - 1))]; };
        Fetched f;
        if (!linear)
        {
            const int i = static_cast<int>(std::floor(u));
            const double frac = u - std::floor(u);
            const Rgba8 t = at(i);
            if ((frac < kTexelMargin || frac > 1.0 - kTexelMargin) && t != at(frac < 0.5 ? i - 1 : i + 1)) f.ambiguous = true;
            for (std::size_t c = 0; c < 4; ++c) f.value[c] = f.lo[c] = f.hi[c] = t[c];
        }
        else
        {
            const double x = u - 0.5;
            const int i0 = static_cast<int>(std::floor(x));
            const double a = x - std::floor(x);
            const Rgba8 t0 = at(i0), t1 = at(i0 + 1);
            for (std::size_t c = 0; c < 4; ++c)
            {
                f.value[c] = (1.0 - a) * t0[c] + a * t1[c];
                f.lo[c] = 255.0;
                f.hi[c] = 0.0;
            }
            for (int i = static_cast<int>(std::floor(x - kTexelMargin)); i <= static_cast<int>(std::floor(x + kTexelMargin)) + 1; ++i)
                for (std::size_t c = 0; c < 4; ++c)
                {
                    f.lo[c] = std::min(f.lo[c], double(at(i)[c]));
                    f.hi[c] = std::max(f.hi[c], double(at(i)[c]));
                }
        }
        return f;
    }

    /// Sampling state that the model needs.
    struct Setup
    {
        mgl::TextureMinFilter min = mgl::TextureMinFilter::NearestMipmapNearest;
        mgl::TextureMagFilter mag = mgl::TextureMagFilter::Nearest;
        int base = 0;
        int max_level = kUnbounded;
        double min_lod = -kUnbounded;
        double max_lod = kUnbounded;
        double range = 0.0;  // s range over the strip
        int cw = 32;         // strip width in pixels
    };

    struct Prediction
    {
        std::array<double, 4> value{};     // ideal value of equations 3.16-3.25
        std::array<double, 4> lo{}, hi{};  // checked range (see "What is checked" above)
        bool ambiguous = false;
        double lambda = 0.0;
    };

    Prediction predict(const Model& m, const Setup& st, double s)
    {
        Prediction p;
        const int w_b = static_cast<int>(m.levels[static_cast<std::size_t>(st.base)].size());
        const int q = std::min(floor_log2(w_b) + st.base, st.max_level);  // highest level in use (section 3.8.10.4)
        const double rho = w_b * st.range / st.cw;                         // eq. 3.20 with du/dy = dv/dx = dv/dy = 0
        const double lambda = std::clamp(std::log2(rho), st.min_lod, st.max_lod);  // eq. 3.16-3.18, no shader bias
        p.lambda = lambda;
        auto level_fetch = [&](int d, bool linear) {
            const Column& col = m.levels[static_cast<std::size_t>(d)];
            return fetch(col, linear, static_cast<double>(col.size()) * s);
        };

        const bool magnified = lambda <= 0.0;
        if (magnified || !mipmapped(st.min))
        {
            // Magnification uses TEXTURE_MAG_FILTER on level_base; NEAREST/LINEAR minification
            // use level_base with TEXTURE_MIN_FILTER (section 3.8.11, 3.8.10.2).
            const bool lin = magnified ? st.mag == mgl::TextureMagFilter::Linear : linear_in_level(st.min);
            const Fetched f = level_fetch(st.base, lin);
            p.ambiguous = f.ambiguous;
            for (std::size_t c = 0; c < 4; ++c)
            {
                p.value[c] = f.value[c];
                p.lo[c] = f.lo[c] - (lin ? 1.0 : 0.0);
                p.hi[c] = f.hi[c] + (lin ? 1.0 : 0.0);
            }
        }
        else if (!blends_levels(st.min))
        {
            // Equation 3.23: the level d nearest to level_base + lambda.
            int d = st.base;
            if (lambda > 0.5)
                d = (st.base + lambda <= q + 0.5) ? static_cast<int>(std::ceil(st.base + lambda + 0.5)) - 1 : q;
            const bool lin = linear_in_level(st.min);
            const Fetched f = level_fetch(d, lin);
            p.ambiguous = f.ambiguous;
            for (std::size_t c = 0; c < 4; ++c)
            {
                p.value[c] = f.value[c];
                p.lo[c] = f.lo[c] - (lin ? 1.0 : 0.0);
                p.hi[c] = f.hi[c] + (lin ? 1.0 : 0.0);
            }
        }
        else
        {
            // Equations 3.24, 3.25: two levels d1, d2 blended with weight frac(lambda).
            int d1 = q, d2 = q;
            if (st.base + lambda < q)
            {
                d1 = static_cast<int>(std::floor(st.base + lambda));
                d2 = d1 + 1;
            }
            const double w2 = lambda - std::floor(lambda);
            const bool lin = linear_in_level(st.min);
            const Fetched f1 = level_fetch(d1, lin), f2 = level_fetch(d2, lin);
            p.ambiguous = f1.ambiguous || f2.ambiguous;
            // Sanity bound on the blend weight (kLodMargin); a single level (d1 == d2) is not blended.
            const double w_lo = d1 == d2 ? 0.0 : std::max(0.0, w2 - kLodMargin);
            const double w_hi = d1 == d2 ? 0.0 : std::min(1.0, w2 + kLodMargin);
            const double round = (lin || d1 != d2) ? 1.0 : 0.0;
            for (std::size_t c = 0; c < 4; ++c)
            {
                p.value[c] = (1.0 - w2) * f1.value[c] + w2 * f2.value[c];
                p.lo[c] = std::min((1.0 - w_lo) * f1.lo[c] + w_lo * f2.lo[c], (1.0 - w_hi) * f1.lo[c] + w_hi * f2.lo[c]) - round;
                p.hi[c] = std::max((1.0 - w_lo) * f1.hi[c] + w_lo * f2.hi[c], (1.0 - w_hi) * f1.hi[c] + w_hi * f2.hi[c]) + round;
            }
        }
        return p;
    }

    // ---------------------------------------------------------------------------------------------
    // Cells
    // ---------------------------------------------------------------------------------------------
    struct Cell
    {
        int row = 0, col = 0;
        mgl::TextureId tex{};
        const Model* model = nullptr;
        mgl::TextureMinFilter min = mgl::TextureMinFilter::NearestMipmapNearest;
        mgl::TextureMagFilter mag = mgl::TextureMagFilter::Nearest;
        int base = 0;
        int max_level = kUnbounded;
        float min_lod = -static_cast<float>(kUnbounded);
        float max_lod = static_cast<float>(kUnbounded);
        double lambda0 = 0.0;  // level of detail the strip has when TEXTURE_BASE_LEVEL = 0
        int w0 = kBaseSize;    // width of level 0
        int n0 = 2;            // the first pixel centre lies at u = n0 + 0.25 in level floor(lambda0)
        float t = 0.5f;
        bool own_params = false;  // true: the cell uses the texture's parameters as they are
    };

    struct Strip
    {
        double s0 = 0.0, range = 0.0;
    };

    /// s coordinates of a strip: ds/dx = range / cw is chosen so that rho = w0 * ds/dx = 2^lambda0.
    Strip strip_of(const Cell& c, int cw)
    {
        const double range = std::exp2(c.lambda0) * cw / c.w0;
        const int lvl = std::max(0, static_cast<int>(std::floor(c.lambda0)));
        const double w_l = std::max(1, c.w0 >> lvl);
        const double pitch = w_l * range / cw;  // level-lvl texels per pixel
        return {(c.n0 + 0.25 - 0.5 * pitch) / w_l, range};
    }

    Setup setup_of(const Cell& c, int cw)
    {
        Setup st;
        st.min = c.min;
        st.mag = c.mag;
        st.base = c.base;
        st.max_level = c.max_level;
        st.min_lod = c.min_lod;
        st.max_lod = c.max_lod;
        st.range = strip_of(c, cw).range;
        st.cw = cw;
        return st;
    }

    struct Layout
    {
        int width = 0, height = 0, cw = 0, ch = 0;
        [[nodiscard]] int x(int col) const { return col * cw; }
        [[nodiscard]] int y_top(int row) const { return row * ch; }
    };

    /// Top-down view of an RGBA8 read-back (glReadPixels returns the bottom row first).
    struct Image
    {
        int w = 0, h = 0;
        std::vector<std::uint8_t> rgba;
        [[nodiscard]] Rgba8 at(int x, int y_top) const
        {
            const std::size_t i = (static_cast<std::size_t>(h - 1 - y_top) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
            return {rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]};
        }
    };

    struct Rect
    {
        int x = 0, y = 0, w = 0, h = 0;  // y from the top
    };

    struct Stats
    {
        long probes = 0, skipped = 0, bad = 0;
        double max_dev = 0.0;  // largest |got - ideal| (observation)
        std::string first_bad;
        void merge(const Stats& o)
        {
            probes += o.probes;
            skipped += o.skipped;
            bad += o.bad;
            max_dev = std::max(max_dev, o.max_dev);
            if (first_bad.empty()) first_bad = o.first_bad;
        }
        [[nodiscard]] std::string summary() const
        {
            return std::format("{} px compared, {} skipped (texel boundary); largest deviation from the ideal value {:.1f}/255 (observation)",
                               probes, skipped, max_dev);
        }
    };

    /// Compares every pixel of `cell`'s rectangle with the CPU model.
    Stats compare_model(const Cell& c, const Image& img, Rect r)
    {
        Stats stats;
        const Setup st = setup_of(c, r.w);
        const Strip strip = strip_of(c, r.w);
        for (int x = 0; x < r.w; ++x)
        {
            const double s = strip.s0 + (x + 0.5) / r.w * strip.range;
            const Prediction p = predict(*c.model, st, s);
            for (int y = 0; y < r.h; ++y)
            {
                if (p.ambiguous)
                {
                    ++stats.skipped;
                    continue;
                }
                ++stats.probes;
                const Rgba8 got = img.at(r.x + x, r.y + y);
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    stats.max_dev = std::max(stats.max_dev, std::abs(got[k] - p.value[k]));
                    if (got[k] < p.lo[k] - 1e-9 || got[k] > p.hi[k] + 1e-9) ok = false;
                }
                if (!ok && stats.bad++ == 0)
                    stats.first_bad = std::format(
                        "x={} y={} lambda={:.3f}: got {} outside [{:.1f}..{:.1f}, {:.1f}..{:.1f}, {:.1f}..{:.1f}, {:.1f}..{:.1f}] (ideal ({:.1f},{:.1f},{:.1f},{:.1f}))",
                        x, y, p.lambda, demo::format_rgba(got), p.lo[0], p.hi[0], p.lo[1], p.hi[1], p.lo[2], p.hi[2], p.lo[3], p.hi[3], p.value[0],
                        p.value[1], p.value[2], p.value[3]);
            }
        }
        return stats;
    }

    /// Expected colour of a rectangle: the ideal value and the checked range per channel.
    struct Range
    {
        std::array<double, 4> value{}, lo{}, hi{};
    };

    /// Compares every pixel of a rectangle with one expected range per channel.
    Stats compare_range(const Image& img, Rect r, const Range& e)
    {
        Stats stats;
        for (int y = 0; y < r.h; ++y)
            for (int x = 0; x < r.w; ++x)
            {
                ++stats.probes;
                const Rgba8 got = img.at(r.x + x, r.y + y);
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    stats.max_dev = std::max(stats.max_dev, std::abs(got[k] - e.value[k]));
                    if (got[k] < e.lo[k] - 1e-9 || got[k] > e.hi[k] + 1e-9) ok = false;
                }
                if (!ok && stats.bad++ == 0)
                    stats.first_bad = std::format("x={} y={}: got {} outside [{:.1f}..{:.1f}, {:.1f}..{:.1f}, {:.1f}..{:.1f}, {:.1f}..{:.1f}]", x, y,
                                                  demo::format_rgba(got), e.lo[0], e.hi[0], e.lo[1], e.hi[1], e.lo[2], e.hi[2], e.lo[3], e.hi[3]);
            }
        return stats;
    }

    std::array<double, 4> as_double(Rgba8 c) { return {double(c[0]), double(c[1]), double(c[2]), double(c[3])}; }

    /// One colour +- tol per channel.
    Range around(Rgba8 c, double tol)
    {
        Range r;
        r.value = as_double(c);
        for (std::size_t k = 0; k < 4; ++k)
        {
            r.lo[k] = r.value[k] - tol;
            r.hi[k] = r.value[k] + tol;
        }
        return r;
    }

    Stats compare_const(const Image& img, Rect r, Rgba8 expected, double tol) { return compare_range(img, r, around(expected, tol)); }

    /// (1 - w) a + w b of two solid levels at frac(lambda) = 1/2: w within kLodMargin of 1/2 (sanity bound), +-1 rounding.
    Range blend(Rgba8 a, Rgba8 b)
    {
        Range r;
        for (std::size_t c = 0; c < 4; ++c)
        {
            r.value[c] = 0.5 * (a[c] + b[c]);
            const double e1 = (0.5 - kLodMargin) * a[c] + (0.5 + kLodMargin) * b[c];
            const double e2 = (0.5 + kLodMargin) * a[c] + (0.5 - kLodMargin) * b[c];
            r.lo[c] = std::min(e1, e2) - 1.0;
            r.hi[c] = std::max(e1, e2) + 1.0;
        }
        return r;
    }

    /// Blend weight w that best explains `got` = (1 - w) a + w b (least squares over the colour channels).
    double measured_weight(Rgba8 got, Rgba8 a, Rgba8 b)
    {
        double num = 0.0, den = 0.0;
        for (std::size_t c = 0; c < 3; ++c)
        {
            num += (double(got[c]) - a[c]) * (double(b[c]) - a[c]);
            den += (double(b[c]) - a[c]) * (double(b[c]) - a[c]);
        }
        return den > 0.0 ? num / den : 0.0;
    }

    Model make_ladder_model()
    {
        Model m;
        for (int k = 0; k < kLevels; ++k) m.levels.emplace_back(static_cast<std::size_t>(kBaseSize >> k), kLadder[static_cast<std::size_t>(k)]);
        return m;
    }

    Model make_stripes_model()
    {
        Model m;
        for (int k = 0; k < kLevels; ++k)
        {
            Column col(static_cast<std::size_t>(kBaseSize >> k));
            for (std::size_t i = 0; i < col.size(); ++i) col[i] = kPalette[(i + 2 * static_cast<std::size_t>(k)) % kPalette.size()];
            m.levels.push_back(std::move(col));
        }
        return m;
    }

    /// Level 0 of the checker as seen along the row t = 0.25 (v = 16): 32 texels of A, then 32 of B.
    Model make_checker_row_model()
    {
        Model m;
        Column col(kBaseSize);
        for (std::size_t i = 0; i < col.size(); ++i) col[i] = kChecker[(i / 32) & 1];
        m.levels.push_back(std::move(col));
        return m;
    }

    // ---------------------------------------------------------------------------------------------
    // Completeness scenarios (rows 11 and 12). Each is an 8x8 RGBA8 texture whose levels 0..3 are
    // 8, 4, 2, 1 texels wide, broken in exactly one way, next to a correct twin.
    // ---------------------------------------------------------------------------------------------
    enum Scenario
    {
        kMissingLevel,    // level 3 (1x1) never specified; fix: specify it
        kWrongSize,       // level 1 is 3x3 instead of 4x4; fix: re-specify it as 4x4
        kWrongFormat,     // level 2 is RGB8 while the others are RGBA8; fix: re-specify it as RGBA8
        kBaseAboveMax,    // TEXTURE_BASE_LEVEL = 2 > TEXTURE_MAX_LEVEL = 1; fix: MAX_LEVEL = 3
        kMaxLevelFix,     // level 3 missing; fix: TEXTURE_MAX_LEVEL = 2 (level 3 becomes insignificant)
        kFilterFix,       // level 3 missing; fix: TEXTURE_MIN_FILTER = LINEAR (no mipmap needed)
        kNoImage,         // no image at all; fix: specify a 1x1 level 0 (a 1-level chain is complete)
        kScenarioCount
    };

    struct ScenarioInfo
    {
        const char* what;
        double lambda0;   // strip lambda (relative to a base level 0 of 8 texels; 1 texel for kNoImage)
        Rgba8 expected;   // colour of the repaired texture
        const char* fix;
    };

    constexpr std::array<ScenarioInfo, kScenarioCount> kScenarioInfo = {{
        {"level 3 never specified", 3.0, kLadder[3], "glTexImage2D(level 3, 1x1)"},
        {"level 1 is 3x3, not 4x4", 1.0, kLadder[1], "glTexImage2D(level 1, 4x4)"},
        {"level 2 is RGB8, the others RGBA8", 2.0, kLadder[2], "glTexImage2D(level 2, RGBA8)"},
        {"TEXTURE_BASE_LEVEL 2 > TEXTURE_MAX_LEVEL 1", 2.0, kLadder[2], "TEXTURE_MAX_LEVEL = 3"},
        {"level 3 missing, TEXTURE_MAX_LEVEL 1000", 3.0, kLadder[2], "TEXTURE_MAX_LEVEL = 2"},
        {"level 3 missing, NEAREST_MIPMAP_NEAREST", 1.0, kLadder[0], "TEXTURE_MIN_FILTER = LINEAR"},
        {"no image specified", 0.0, kLadder[0], "glTexImage2D(level 0, 1x1)"},
    }};

    class Mipmaps final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // Every check reads at least one pixel of its cells (rows 0-14, columns 0-7, and the scratch cell at the
            // top left that the textureSize and negative-control experiments draw into). A cell is width / 8 by
            // height / 15 pixels, so a smaller surface has empty cells and those checks would compare nothing.
            if (env.width < kCols || env.height < kRows)
                throw demo::Skip(std::format("needs a surface of at least {} x {} pixels (one pixel per cell of the {} x {} grid)", kCols, kRows,
                                             kCols, kRows));
            program_ = demo::build_program(env, "shaders/strip.vert", "shaders/strip.frag");
            loc_s0_ = locate("u_s0");
            loc_range_ = locate("u_s_range");
            loc_t_ = locate("u_t");
            loc_mode_ = locate("u_mode");
            loc_lod_ = locate("u_lod");
            mgl::glUseProgram(program_);
            mgl::glUniform1i(locate("u_tex"), 0);  // the sampler reads texture unit 0

            // The hint's initial value is DONT_CARE (ES 3.0 section 5.3); read it before touching it.
            initial_hint_ = demo::get_integer(mgl::GetParameter::GenerateMipmapHint);

            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            ladder_model_ = make_ladder_model();
            stripes_model_ = make_stripes_model();
            checker_model_ = make_checker_row_model();
            tiny_model_.levels.push_back(Column{kChecker[0], kChecker[1]});  // row 0 of the 2x2 checker of check_generate_mipmap

            // --- ladder: seven solid levels, uploaded one glTexImage2D call each ----------------------
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kLadderTex]);
            for (int k = 0; k < kLevels; ++k) upload_solid(k, kBaseSize >> k, kLadder[static_cast<std::size_t>(k)]);
            set_sampling_defaults();

            // --- stripes: seven levels of alternating colours ------------------------------------------
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kStripesTex]);
            for (int k = 0; k < kLevels; ++k)
            {
                const Column& col = stripes_model_.levels[static_cast<std::size_t>(k)];
                const int size = static_cast<int>(col.size());
                std::vector<Rgba8> texels(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
                for (int j = 0; j < size; ++j)
                    for (int i = 0; i < size; ++i) texels[static_cast<std::size_t>(j * size + i)] = col[static_cast<std::size_t>(i)];
                upload(k, size, texels.data());
            }
            set_sampling_defaults();

            // --- checker: only level 0 is specified, glGenerateMipmap derives levels 1..6 --------------
            // A 2x2 checker blown up to 64x64 (32-texel squares). Level 0 is the only level the demo
            // defines; levels 1-6 come from glGenerateMipmap.
            std::vector<Rgba8> checker(static_cast<std::size_t>(kBaseSize * kBaseSize));
            for (int j = 0; j < kBaseSize; ++j)
                for (int i = 0; i < kBaseSize; ++i)
                    checker[static_cast<std::size_t>(j * kBaseSize + i)] = kChecker[((i / 32) + (j / 32)) & 1];
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kCheckerTex]);
            upload(0, kBaseSize, checker.data());
            set_sampling_defaults();
            mgl::glHint(mgl::HintTarget::GenerateMipmap, mgl::HintMode::Nicest);
            hint_after_set_ = demo::get_integer(mgl::GetParameter::GenerateMipmapHint);
            mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);

            // --- completeness scenarios ----------------------------------------------------------------
            for (int s = 0; s < kScenarioCount; ++s)
            {
                build_scenario(static_cast<Scenario>(s), broken(s), true);
                build_scenario(static_cast<Scenario>(s), twin(s), false);
            }
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            build_cells();
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            const Layout lay = layout_of(env);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(program_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            for (const Cell& c : cells_) draw_cell(c, lay, lay.x(c.col), lay.y_top(c.row));
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            env_version_ = env.context_version;
            const Layout lay = layout_of(env);
            const Image fb = read_fb(lay);
            check_level_selection(check, lay, fb);
            check_stripes(check, lay, fb);
            check_lod_state(check, lay, fb);
            check_generate_mipmap(check, lay, fb);
            check_completeness(check, lay, fb);
            check_state_and_errors(check, lay);
            check_controls(check, lay, fb);

            render(env, 0);  // the experiments above drew into the framebuffer: restore the picture
            check.expect_no_gl_error("no GL error while sampling incomplete textures or after the experiments");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glHint(mgl::HintTarget::GenerateMipmap, mgl::HintMode::DontCare);
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteProgram(program_);
        }

    private:
        demo::Version env_version_{};
        enum : std::size_t { kLadderTex, kStripesTex, kCheckerTex, kScenarioBase, kTexCount = kScenarioBase + 2 * kScenarioCount };

        [[nodiscard]] mgl::TextureId broken(int s) const { return tex_[kScenarioBase + 2 * static_cast<std::size_t>(s)]; }
        [[nodiscard]] mgl::TextureId twin(int s) const { return tex_[kScenarioBase + 2 * static_cast<std::size_t>(s) + 1]; }

        mgl::UniformLocation locate(const char* name) const
        {
            const mgl::UniformLocation l = mgl::glGetUniformLocation(program_, name);
            if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
            return l;
        }

        static Layout layout_of(const demo::Env& env)
        {
            return {env.width, env.height, env.width / kCols, env.height / kRows};
        }

        // ------------------------------------------------------------------------------------------
        // Uploads
        // ------------------------------------------------------------------------------------------
        /// glTexImage2D of an RGBA8 level (sized internal format => effective internal format RGBA8).
        static void upload(int level, int size, const void* data)
        {
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgba8, size, size, 0, mgl::PixelFormat::Rgba,
                              mgl::PixelType::UnsignedByte, data);
        }

        static void upload_solid(int level, int size, Rgba8 color)
        {
            const std::vector<Rgba8> texels(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), color);
            upload(level, size, texels.data());
        }

        /// A level with a different internal format: RGB8 (3 bytes per texel; rows of 2 texels are 6 bytes, so
        /// UNPACK_ALIGNMENT must be 1 for the tightly packed data below).
        static void upload_solid_rgb(int level, int size, Rgba8 color)
        {
            std::vector<std::uint8_t> bytes;
            for (int i = 0; i < size * size; ++i)
                for (int c = 0; c < 3; ++c) bytes.push_back(color[static_cast<std::size_t>(c)]);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 1);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, level, mgl::InternalFormat::Rgb8, size, size, 0, mgl::PixelFormat::Rgb,
                              mgl::PixelType::UnsignedByte, bytes.data());
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
        }

        /// Parameters shared by the textures that the cells sample: clamping, and a mipmapped min filter.
        static void set_sampling_defaults()
        {
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::NearestMipmapNearest);
        }

        // ------------------------------------------------------------------------------------------
        // Scenarios
        // ------------------------------------------------------------------------------------------
        void build_scenario(Scenario s, mgl::TextureId tex, bool is_broken)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            if (s == kNoImage)
            {
                // Texture parameters stay at their initial values: REPEAT, NEAREST_MIPMAP_LINEAR (the default
                // TEXTURE_MIN_FILTER). A texture with one level and no other levels is mipmap complete.
                if (!is_broken) upload_solid(0, 1, kLadder[0]);
                return;
            }
            set_sampling_defaults();
            upload_solid(0, 8, kLadder[0]);
            upload_solid(1, (s == kWrongSize && is_broken) ? 3 : 4, kLadder[1]);
            if (s == kWrongFormat && is_broken) upload_solid_rgb(2, 2, kLadder[2]);
            else upload_solid(2, 2, kLadder[2]);
            const bool level3 = s == kWrongSize || s == kWrongFormat || s == kBaseAboveMax || (s == kMissingLevel && !is_broken);
            if (level3) upload_solid(3, 1, kLadder[3]);
            if (s == kBaseAboveMax)
            {
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BaseLevel, 2);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, is_broken ? 1 : 3);
            }
            if (s == kMaxLevelFix && !is_broken)
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, 2);
            if (s == kFilterFix && !is_broken) mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Linear);
        }

        /// The single call that turns the broken texture into its twin.
        void repair_scenario(Scenario s, mgl::TextureId tex)
        {
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            switch (s)
            {
            case kMissingLevel: upload_solid(3, 1, kLadder[3]); break;
            case kWrongSize: upload_solid(1, 4, kLadder[1]); break;
            case kWrongFormat: upload_solid(2, 2, kLadder[2]); break;
            case kBaseAboveMax: mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, 3); break;
            case kMaxLevelFix: mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, 2); break;
            case kFilterFix: mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Linear); break;
            case kNoImage: upload_solid(0, 1, kLadder[0]); break;
            case kScenarioCount: break;
            }
        }

        Cell scenario_cell(int s, mgl::TextureId tex) const
        {
            Cell c;
            c.tex = tex;
            c.lambda0 = kScenarioInfo[static_cast<std::size_t>(s)].lambda0;
            c.w0 = s == kNoImage ? 1 : 8;
            c.own_params = true;
            return c;
        }

        // ------------------------------------------------------------------------------------------
        // Cell table
        // ------------------------------------------------------------------------------------------
        void build_cells()
        {
            using F = mgl::TextureMinFilter;
            using M = mgl::TextureMagFilter;
            auto add = [&](int row, int col, Cell c) {
                c.row = row;
                c.col = col;
                cells_.push_back(c);
            };

            struct RowSpec
            {
                int row;
                std::size_t tex;
                const Model* model;
                F min;
                M mag;
                double lambda_offset;
            };
            const RowSpec rows[] = {
                {0, kLadderTex, &ladder_model_, F::NearestMipmapNearest, M::Nearest, 0.0},
                {1, kLadderTex, &ladder_model_, F::LinearMipmapNearest, M::Linear, 0.0},
                {2, kLadderTex, &ladder_model_, F::NearestMipmapLinear, M::Nearest, 0.5},
                {3, kLadderTex, &ladder_model_, F::LinearMipmapLinear, M::Linear, 0.5},
                {4, kLadderTex, &ladder_model_, F::NearestMipmapNearest, M::Nearest, 0.25},
                {5, kLadderTex, &ladder_model_, F::NearestMipmapNearest, M::Nearest, 0.75},
                {6, kStripesTex, &stripes_model_, F::NearestMipmapNearest, M::Nearest, 0.0},
                {7, kStripesTex, &stripes_model_, F::LinearMipmapNearest, M::Linear, 0.0},
                {8, kStripesTex, &stripes_model_, F::NearestMipmapLinear, M::Nearest, 0.5},
                {9, kStripesTex, &stripes_model_, F::LinearMipmapLinear, M::Linear, 0.5},
            };
            for (const RowSpec& rs : rows)
                for (int k = 0; k < kCols; ++k)
                {
                    Cell c;
                    c.tex = tex_[rs.tex];
                    c.model = rs.model;
                    c.min = rs.min;
                    c.mag = rs.mag;
                    c.lambda0 = k + rs.lambda_offset;
                    add(rs.row, k, c);
                }

            // Row 10: the glGenerateMipmap'ed checker. The strip is centred on the boundary between the two
            // colours of level floor(lambda), so the left half is one colour and the right half the other.
            for (int k = 0; k < kCols; ++k)
            {
                Cell c;
                c.tex = tex_[kCheckerTex];
                c.model = k == 0 ? &checker_model_ : nullptr;  // only level 0 is specified by the application
                c.lambda0 = k;
                c.t = 0.25f;
                // The boundary between the two colours lies at texel (64 >> k) / 2 of level k. Half of the 32-pixel strip
                // of the standard 256-pixel surface is 16 pixels = 16 level-k texels at lambda = k, so start 16 texels
                // before it. (Another surface size only shifts the boundary inside the strip.)
                c.n0 = (32 >> k) - 16;
                add(10, k, c);
            }

            // Rows 11-12: [incomplete | complete twin] pairs.
            for (int s = 0; s < kScenarioCount; ++s)
            {
                const int row = 11 + s / 4, col = 2 * (s % 4);
                add(row, col, scenario_cell(s, broken(s)));
                add(row, col + 1, scenario_cell(s, twin(s)));
            }

            // Row 13: TEXTURE_BASE_LEVEL / TEXTURE_MAX_LEVEL on the ladder.
            struct LevelSpec
            {
                int base, max_level;
                F min;
                double lambda0;
            };
            const LevelSpec level_cells[] = {
                {2, kUnbounded, F::NearestMipmapNearest, 1.0},  {2, kUnbounded, F::NearestMipmapNearest, 3.0},
                {0, 3, F::NearestMipmapNearest, 5.0},           {0, 3, F::NearestMipmapNearest, 2.0},
                {0, 3, F::NearestMipmapLinear, 2.5},            {0, 3, F::NearestMipmapLinear, 3.5},
                {2, kUnbounded, F::NearestMipmapLinear, 3.5},   {2, 4, F::NearestMipmapLinear, 5.5},
            };
            for (int i = 0; i < 8; ++i)
            {
                Cell c;
                c.tex = tex_[kLadderTex];
                c.model = &ladder_model_;
                c.base = level_cells[i].base;
                c.max_level = level_cells[i].max_level;
                c.min = level_cells[i].min;
                c.lambda0 = level_cells[i].lambda0;
                add(13, i, c);
            }

            // Row 14: TEXTURE_MIN_LOD / TEXTURE_MAX_LOD (equation 3.18).
            struct LodSpec
            {
                float min_lod, max_lod;
                F min;
                double lambda0;
            };
            const LodSpec lod_cells[] = {
                {2.0f, 1000.0f, F::NearestMipmapNearest, 0.0},
                {1.5f, 1000.0f, F::NearestMipmapLinear, 0.0},
                {-1000.0f, 3.0f, F::NearestMipmapNearest, 5.0},
                {-1000.0f, 2.5f, F::NearestMipmapLinear, 5.0},
            };
            for (int i = 0; i < 4; ++i)
            {
                Cell c;
                c.tex = tex_[kLadderTex];
                c.model = &ladder_model_;
                c.min_lod = lod_cells[i].min_lod;
                c.max_lod = lod_cells[i].max_lod;
                c.min = lod_cells[i].min;
                c.lambda0 = lod_cells[i].lambda0;
                add(14, i, c);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Drawing
        // ------------------------------------------------------------------------------------------
        void draw_cell(const Cell& c, const Layout& lay, int x, int y_top) const
        {
            mgl::glViewport(x, lay.height - y_top - lay.ch, lay.cw, lay.ch);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, c.tex);
            if (!c.own_params)
            {
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, c.min);
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, c.mag);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BaseLevel, c.base);
                mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, c.max_level);
                mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MinLod, c.min_lod);
                mgl::glTexParameterf(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLod, c.max_lod);
            }
            const Strip s = strip_of(c, lay.cw);
            mgl::glUniform1f(loc_s0_, static_cast<float>(s.s0));
            mgl::glUniform1f(loc_range_, static_cast<float>(s.range));
            mgl::glUniform1f(loc_t_, c.t);
            mgl::glUniform1i(loc_mode_, 0);
            mgl::glUniform1i(loc_lod_, 0);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        /// Draws one cell into the top-left cell of the framebuffer and returns the framebuffer.
        Image draw_scratch(const Cell& c, const Layout& lay) const
        {
            mgl::glUseProgram(program_);
            draw_cell(c, lay, 0, 0);
            return read_fb(lay);
        }

        /// Encodes textureSize(sampler, lod) of `tex` as a colour and returns {width, height}.
        std::array<int, 2> query_size(mgl::TextureId tex, int lod, const Layout& lay) const
        {
            mgl::glUseProgram(program_);
            mgl::glViewport(0, lay.height - lay.ch, lay.cw, lay.ch);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex);
            mgl::glUniform1f(loc_s0_, 0.0f);
            mgl::glUniform1f(loc_range_, 1.0f);
            mgl::glUniform1f(loc_t_, 0.5f);
            mgl::glUniform1i(loc_mode_, 1);
            mgl::glUniform1i(loc_lod_, lod);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
            mgl::glUniform1i(loc_mode_, 0);
            // The centre of the cell viewport [height - ch, height): for ch = 1 that is row height - 1, not height.
            const Rgba8 p = demo::read_pixel(lay.cw / 2, lay.height - lay.ch + lay.ch / 2);
            return {p[0], p[1]};
        }

        static Image read_fb(const Layout& lay)
        {
            Image img;
            img.w = lay.width;
            img.h = lay.height;
            img.rgba = demo::read_rgba8(0, 0, lay.width, lay.height);
            return img;
        }

        [[nodiscard]] Rect rect_of(const Cell& c, const Layout& lay) const { return {lay.x(c.col), lay.y_top(c.row), lay.cw, lay.ch}; }

        [[nodiscard]] const Cell& cell_at(int row, int col) const
        {
            for (const Cell& c : cells_)
                if (c.row == row && c.col == col) return c;
            throw demo::Failure(std::format("no cell at row {} column {}", row, col));
        }

        // ------------------------------------------------------------------------------------------
        // Checks: level selection with solid levels (rows 0-5)
        // ------------------------------------------------------------------------------------------
        void check_level_selection(demo::Check& check, const Layout& lay, const Image& fb)
        {
            auto level_of = [](int k) { return std::min(k, kLevels - 1); };

            // Row 0: lambda = k, NEAREST_MIPMAP_NEAREST: level k, i.e. the exact colour of that level
            // (equation 3.23: d = ceil(lambda + 1/2) - 1 = k; lambda > q + 1/2 clamps to q = 6).
            for (const int row : {0, 1, 4, 5})
            {
                static const char* names[] = {"NEAREST_MIPMAP_NEAREST, lambda = k", "LINEAR_MIPMAP_NEAREST, lambda = k", "", "",
                                              "NEAREST_MIPMAP_NEAREST, lambda = k + 0.25 (rounds down)",
                                              "NEAREST_MIPMAP_NEAREST, lambda = k + 0.75 (rounds up)"};
                Stats all;
                for (int k = 0; k < kCols; ++k)
                {
                    const int level = row == 5 ? level_of(k + 1) : level_of(k);
                    // NEAREST inside a solid level returns the texel unchanged: tolerance 0. LINEAR weights are convex, so
                    // a solid level gives its colour; only the 8-bit conversion (+-1, section 2.1.6.2) can appear.
                    const double tol = row == 1 ? 1.0 : 0.0;
                    all.merge(compare_const(fb, rect_of(cell_at(row, k), lay), kLadder[static_cast<std::size_t>(level)], tol));
                }
                check.expect(all.bad == 0,
                             std::format("row {}: {} selects exactly the colour of level {} in all 8 cells", row, names[row],
                                         row == 5 ? "min(k+1, 6)" : "min(k, 6)"),
                             all.bad == 0 ? all.summary() : all.first_bad);
            }

            // Rows 2, 3: lambda = k + 0.5 between two levels: a blend of the two level colours
            // (equations 3.24, 3.25, tau = (1 - frac) tau1 + frac tau2, ideally frac = 0.5). The levels are fixed (lambda is
            // 1/2 from the next integer); the weight is checked only against the kLodMargin sanity bound and printed. From
            // k = 6 on, base + lambda >= q: both levels are q = 6 and the result is the colour of level 6.
            for (const int row : {2, 3})
            {
                Stats all;
                double w_min = 1.0, w_max = 0.0;
                for (int k = 0; k < kCols; ++k)
                {
                    const Rect r = rect_of(cell_at(row, k), lay);
                    if (k < kLevels - 1)
                    {
                        const Range b = blend(kLadder[static_cast<std::size_t>(k)], kLadder[static_cast<std::size_t>(k + 1)]);
                        all.merge(compare_range(fb, r, b));
                        const double w = measured_weight(fb.at(r.x + r.w / 2, r.y + r.h / 2), kLadder[static_cast<std::size_t>(k)],
                                                         kLadder[static_cast<std::size_t>(k + 1)]);
                        w_min = std::min(w_min, w);
                        w_max = std::max(w_max, w);
                    }
                    else
                        all.merge(compare_const(fb, r, kLadder[kLevels - 1], row == 3 ? 1.0 : 0.0));
                }
                check.expect(all.bad == 0,
                             std::format("row {}: {}_MIPMAP_LINEAR at lambda = k + 0.5 blends levels k and k+1 with a weight in [1/4, 3/4] "
                                         "(sanity bound; level 6 from k = 6)",
                                         row, row == 2 ? "NEAREST" : "LINEAR"),
                             all.bad == 0 ? all.summary() : all.first_bad);
                std::cout << std::format("   note: row {}: measured blend weight frac(lambda) of levels k, k+1 at lambda = k + 0.5: {:.3f} .. {:.3f} "
                                         "(ideal 0.5; observation)\n",
                                         row, w_min, w_max);
            }

            // The same cells against the full CPU model (rows 0-5 include the model's own rounding rules).
            for (int row = 0; row <= 5; ++row)
            {
                Stats all;
                for (int k = 0; k < kCols; ++k) all.merge(compare_model(cell_at(row, k), fb, rect_of(cell_at(row, k), lay)));
                check.expect(all.bad == 0 && all.probes > 8 * lay.cw * lay.ch / 2,
                             std::format("row {}: every pixel lies in the range the CPU model of equations 3.20-3.25 guarantees", row),
                             all.bad == 0 ? all.summary() : all.first_bad);
            }
        }

        // ------------------------------------------------------------------------------------------
        // Rows 6-9: patterned levels, all four filters against the CPU sampler
        // ------------------------------------------------------------------------------------------
        void check_stripes(demo::Check& check, const Layout& lay, const Image& fb)
        {
            static const char* names[] = {"NEAREST_MIPMAP_NEAREST (lambda = k)", "LINEAR_MIPMAP_NEAREST (lambda = k)",
                                          "NEAREST_MIPMAP_LINEAR (lambda = k + 0.5)", "LINEAR_MIPMAP_LINEAR (lambda = k + 0.5)"};
            for (int row = 6; row <= 9; ++row)
            {
                Stats all;
                for (int k = 0; k < kCols; ++k) all.merge(compare_model(cell_at(row, k), fb, rect_of(cell_at(row, k), lay)));
                check.expect(all.bad == 0 && all.probes > 8 * lay.cw * lay.ch / 3,
                             std::format("row {}: {} lies in the CPU sampler's range (exact level selection, in-level filter bound)", row,
                                         names[row - 6]),
                             all.bad == 0 ? all.summary() : all.first_bad);
            }

            // Observation: the in-level filter is visible. At lambda = k the strip has one level-k texel per pixel and
            // the sample point sits 1/4 texel past a texel centre, so NEAREST returns a texel and the ideal LINEAR a
            // 3:1 mixture of two neighbours (adjacent texels differ by at least 110/255 in one channel, so the two
            // results ideally differ by at least 27). How much they differ depends on the weight precision, which the
            // spec leaves open, so the count is printed, not checked. Compared over the pixels whose sample point lies
            // inside the level (the strip starts at texel 2 of a level that has 32, 16, 8 texels for k = 1, 2, 3).
            int differing = 0, compared = 0;
            for (int k = 1; k <= 3; ++k)
            {
                const Rect a = rect_of(cell_at(6, k), lay), b = rect_of(cell_at(7, k), lay);
                const int inside = std::min(lay.cw, (kBaseSize >> k) - 2);
                for (int x = 0; x < inside; ++x)
                {
                    const Rgba8 pa = fb.at(a.x + x, a.y + lay.ch / 2), pb = fb.at(b.x + x, b.y + lay.ch / 2);
                    int dmax = 0;
                    for (std::size_t c = 0; c < 4; ++c) dmax = std::max(dmax, std::abs(pa[c] - pb[c]));
                    ++compared;
                    if (dmax >= 20) ++differing;
                }
            }
            std::cout << std::format("   note: rows 6 vs 7: {} of {} pixels inside a level differ by at least 20/255 between NEAREST_MIPMAP_NEAREST and "
                                     "LINEAR_MIPMAP_NEAREST (observation)\n",
                                     differing, compared);
        }

        // ------------------------------------------------------------------------------------------
        // Rows 13, 14: BASE_LEVEL, MAX_LEVEL, MIN_LOD, MAX_LOD
        // ------------------------------------------------------------------------------------------
        void check_lod_state(demo::Check& check, const Layout& lay, const Image& fb)
        {
            auto solid_cell = [&](int row, int col, Rgba8 color, double tol) {
                return compare_const(fb, rect_of(cell_at(row, col), lay), color, tol);
            };
            // Row 13. With BASE_LEVEL = b, level b is the base image: lambda is measured in units of the base
            // image, so a strip that has lambda0 = 1 relative to level 0 has lambda = 1 - b = -1 relative to
            // level 2 (magnified: the base level, level 2). With MAX_LEVEL = m the chain ends at q = m
            // (equation 3.23, third case; equation 3.24 with base + lambda >= q).
            Stats pure;
            pure.merge(solid_cell(13, 0, kLadder[2], 0));  // BASE_LEVEL 2, lambda0 = 1  -> level 2 (not 1)
            pure.merge(solid_cell(13, 1, kLadder[3], 0));  // BASE_LEVEL 2, lambda0 = 3  -> level 3 (unchanged)
            pure.merge(solid_cell(13, 2, kLadder[3], 0));  // MAX_LEVEL 3,  lambda0 = 5  -> level 3 (not 5)
            pure.merge(solid_cell(13, 3, kLadder[2], 0));  // MAX_LEVEL 3,  lambda0 = 2  -> level 2
            pure.merge(solid_cell(13, 5, kLadder[3], 0));  // MAX_LEVEL 3,  NEAREST_MIPMAP_LINEAR lambda0 = 3.5 -> both levels 3
            pure.merge(solid_cell(13, 7, kLadder[4], 0));  // BASE 2, MAX 4, NEAREST_MIPMAP_LINEAR lambda0 = 5.5 -> both levels 4
            check.expect(pure.bad == 0, "row 13: TEXTURE_BASE_LEVEL makes level b the base image; TEXTURE_MAX_LEVEL caps the level at q",
                         pure.bad == 0 ? pure.summary() : pure.first_bad);

            Stats blended;
            {
                const Range b23 = blend(kLadder[2], kLadder[3]), b34 = blend(kLadder[3], kLadder[4]);
                blended.merge(compare_range(fb, rect_of(cell_at(13, 4), lay), b23));  // MAX_LEVEL 3, lambda0 = 2.5
                blended.merge(compare_range(fb, rect_of(cell_at(13, 6), lay), b34));  // BASE 2, lambda0 = 3.5 -> lambda 1.5
            }
            check.expect(blended.bad == 0,
                         "row 13: NEAREST_MIPMAP_LINEAR blends levels base + floor(lambda) and the next one (weight in [1/4, 3/4], sanity bound)",
                         blended.bad == 0 ? blended.summary() : blended.first_bad);

            Stats model;
            for (int i = 0; i < 8; ++i) model.merge(compare_model(cell_at(13, i), fb, rect_of(cell_at(13, i), lay)));
            check.expect(model.bad == 0, "row 13: all eight cells also lie in the CPU model's range", model.bad == 0 ? model.summary() : model.first_bad);

            // Row 14: lambda' is clamped to [TEXTURE_MIN_LOD, TEXTURE_MAX_LOD] before the level is chosen (eq. 3.18).
            Stats lod;
            lod.merge(solid_cell(14, 0, kLadder[2], 0));  // MIN_LOD 2:   lambda0 = 0 is raised to 2 -> level 2 (and the filter is a minification filter)
            lod.merge(solid_cell(14, 2, kLadder[3], 0));  // MAX_LOD 3:   lambda0 = 5 is lowered to 3 -> level 3
            {
                const Range b12 = blend(kLadder[1], kLadder[2]), b23 = blend(kLadder[2], kLadder[3]);
                lod.merge(compare_range(fb, rect_of(cell_at(14, 1), lay), b12));  // MIN_LOD 1.5 -> blend of levels 1 and 2
                lod.merge(compare_range(fb, rect_of(cell_at(14, 3), lay), b23));  // MAX_LOD 2.5 -> blend of levels 2 and 3
            }
            check.expect(lod.bad == 0, "row 14: TEXTURE_MIN_LOD raises and TEXTURE_MAX_LOD lowers lambda (equation 3.18)",
                         lod.bad == 0 ? lod.summary() : lod.first_bad);
            Stats lod_model;
            for (int i = 0; i < 4; ++i) lod_model.merge(compare_model(cell_at(14, i), fb, rect_of(cell_at(14, i), lay)));
            check.expect(lod_model.bad == 0, "row 14: all four cells also lie in the CPU model's range",
                         lod_model.bad == 0 ? lod_model.summary() : lod_model.first_bad);
        }

        // ------------------------------------------------------------------------------------------
        // glGenerateMipmap, glHint(GENERATE_MIPMAP_HINT)
        // ------------------------------------------------------------------------------------------
        /// Probes strips of `tex` at lambda0 = first..last, NEAREST_MIPMAP_NEAREST, and compares each with `expected(k)`.
        template <typename ExpectedFn>
        Stats probe_levels(mgl::TextureId tex, const Layout& lay, int first, int last, double tol, ExpectedFn expected) const
        {
            Stats all;
            for (int k = first; k <= last; ++k)
            {
                Cell c;
                c.tex = tex;
                c.lambda0 = k;
                const Image img = draw_scratch(c, lay);
                all.merge(compare_const(img, {0, 0, lay.cw, lay.ch}, expected(k), tol));
            }
            return all;
        }

        /// The pixels of strips of `tex` at lambda0 = first..last (NEAREST_MIPMAP_NEAREST: level k each), concatenated.
        std::vector<Rgba8> snapshot_levels(mgl::TextureId tex, const Layout& lay, int first, int last) const
        {
            std::vector<Rgba8> out;
            for (int k = first; k <= last; ++k)
            {
                Cell c;
                c.tex = tex;
                c.lambda0 = k;
                const Image img = draw_scratch(c, lay);
                for (int y = 0; y < lay.ch; ++y)
                    for (int x = 0; x < lay.cw; ++x) out.push_back(img.at(x, y));
            }
            return out;
        }

        static long count_differences(const std::vector<Rgba8>& a, const std::vector<Rgba8>& b)
        {
            long n = a.size() == b.size() ? 0 : static_cast<long>(std::max(a.size(), b.size()));
            for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
                if (a[i] != b[i]) ++n;
            return n;
        }

        static int max_deviation(const std::vector<Rgba8>& pixels, Rgba8 from)
        {
            int m = 0;
            for (const Rgba8& p : pixels)
                for (std::size_t c = 0; c < 4; ++c) m = std::max(m, std::abs(p[c] - from[c]));
            return m;
        }

        void check_generate_mipmap(demo::Check& check, const Layout& lay, const Image& fb)
        {
            // --- the hint --------------------------------------------------------------------------------
            check.expect_eq(initial_hint_, static_cast<int>(GL_DONT_CARE), "GENERATE_MIPMAP_HINT is DONT_CARE initially (ES 3.0 section 5.3)");
            check.expect_eq(hint_after_set_, static_cast<int>(GL_NICEST), "glHint(GENERATE_MIPMAP_HINT, NICEST) is read back by glGetIntegerv");

            // --- the generated checker (row 10) ---------------------------------------------------------
            // The spec fixes only that levels base+1..q are replaced by a filtered reduction ("no particular filter
            // algorithm is required, though a box filter is recommended") and that level base is unchanged. Before the
            // call only level 0 existed, so the texture (min filter NEAREST_MIPMAP_NEAREST) was incomplete and would have
            // sampled (0,0,0,1) at every level of detail, level 0 included (sections 3.8.13, 3.9.2). Level 0 showing the
            // uploaded colours therefore also proves that the generated chain made the texture complete.
            {
                const Stats base = compare_model(cell_at(10, 0), fb, rect_of(cell_at(10, 0), lay));
                check.expect(base.bad == 0 && base.probes > 0,
                             "checker: after glGenerateMipmap level 0 is unchanged (row 10, column 0: two colours as uploaded), so the texture is complete",
                             base.bad == 0 ? base.summary() : base.first_bad);
                int black = 0;
                for (int k = 1; k < kLevels; ++k)
                {
                    const Rect r = rect_of(cell_at(10, k), lay);
                    for (int x = 0; x < r.w; ++x)
                        if (fb.at(r.x + x, r.y + r.h / 2) == kBlack) ++black;
                }
                std::cout << std::format("   note: checker levels 1-6: {} black pixels on the probe line (observation: the filter is unspecified)\n", black);
                const Rect l6 = rect_of(cell_at(10, 6), lay);
                const Rgba8 top = fb.at(l6.x + l6.w / 2, l6.y + l6.h / 2);
                const double mean_r = 0.5 * (kChecker[0][0] + kChecker[1][0]);
                std::cout << std::format("   note: 1x1 level of the checker = {} (box-filter mean of {} and {} would be ({:.1f}, {:.1f}, {:.1f})); the spec does not require it\n",
                                         demo::format_rgba(top), demo::format_rgba(kChecker[0]), demo::format_rgba(kChecker[1]), mean_r,
                                         0.5 * (kChecker[0][1] + kChecker[1][1]), 0.5 * (kChecker[0][2] + kChecker[1][2]));
            }

            // --- sizes of the generated levels, read with textureSize (ES 3.0 section 2.12.9.3) --------------
            {
                int bad = 0;
                std::string first;
                for (int k = 0; k < kLevels; ++k)
                {
                    const auto sz = query_size(tex_[kCheckerTex], k, lay);
                    if (sz[0] != (kBaseSize >> k) || sz[1] != (kBaseSize >> k))
                        if (bad++ == 0) first = std::format("level {}: {}x{}", k, sz[0], sz[1]);
                }
                check.expect(bad == 0, "textureSize reports 64, 32, 16, 8, 4, 2, 1 for the generated levels 0-6", first);
            }

            // --- hint modes, replacement of existing content, completeness before/after ----------------------
            mgl::TextureId scratch[5]{};
            mgl::glGenTextures(5, scratch);
            const mgl::TextureId uniform_tex = scratch[0], range_tex = scratch[1], npot_tex = scratch[2], tiny_tex = scratch[3];

            const struct
            {
                mgl::HintMode mode;
                GLenum gl;
                const char* name;
            } hints[] = {{mgl::HintMode::Fastest, GL_FASTEST, "FASTEST"}, {mgl::HintMode::Nicest, GL_NICEST, "NICEST"},
                         {mgl::HintMode::DontCare, GL_DONT_CARE, "DONT_CARE"}};
            bool first_run = true;
            for (const auto& h : hints)
            {
                mgl::glHint(mgl::HintTarget::GenerateMipmap, h.mode);
                check.expect_eq(demo::get_integer(mgl::GetParameter::GenerateMipmapHint), static_cast<int>(h.gl),
                                std::format("glGetIntegerv(GENERATE_MIPMAP_HINT) returns {} after glHint", h.name));

                mgl::glBindTexture(mgl::TextureTarget::Texture2D, uniform_tex);
                if (first_run) set_sampling_defaults();
                upload_solid(0, kBaseSize, kUniform);
                upload_solid(1, kBaseSize / 2, kBogus);  // existing content: must be replaced ("regardless of their previous contents")
                upload_solid(2, kBaseSize / 4, kBogus);
                if (first_run)
                {
                    Cell c;
                    c.tex = uniform_tex;
                    c.lambda0 = 3;
                    const Image before = draw_scratch(c, lay);
                    const Stats s = compare_const(before, {0, 0, lay.cw, lay.ch}, kBlack, 0);
                    check.expect(s.bad == 0, "before glGenerateMipmap: levels 0-2 only, the mipmapped min filter makes the texture incomplete -> (0,0,0,1)",
                                 s.bad == 0 ? "" : s.first_bad);
                }
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, uniform_tex);
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                const std::vector<Rgba8> first_gen = snapshot_levels(uniform_tex, lay, 1, 6);
                // "Mipmap generation replaces texel array levels levelbase + 1 through q with arrays derived from the
                // levelbase array, regardless of their previous contents" (section 3.8.10.5): overwrite levels 1-2 with other
                // content and generate again from the same base; every derived level must come out identical.
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, uniform_tex);
                upload_solid(1, kBaseSize / 2, kBogus2);
                upload_solid(2, kBaseSize / 4, kBogus2);
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                const std::vector<Rgba8> second_gen = snapshot_levels(uniform_tex, lay, 1, 6);
                const long diff = count_differences(first_gen, second_gen);
                check.expect(diff == 0 && !first_gen.empty(),
                             std::format("hint {}: glGenerateMipmap's levels 1-6 depend only on level 0: red or green previous levels 1-2 give "
                                         "identical results",
                                         h.name),
                             std::format("{} of {} pixels differ", diff, first_gen.size()));
                // A filter that preserves a constant image is the obvious choice, but "no particular filter algorithm is required".
                std::cout << std::format("   note: hint {}: generated levels 1-6 of a uniform {} deviate from it by at most {}/255 (observation)\n",
                                         h.name, demo::format_rgba(kUniform), max_deviation(first_gen, kUniform));
                first_run = false;
            }

            // --- only levels base+1..q are replaced -------------------------------------------------------------
            // range_tex has the ladder colours in levels 1-6, its twin red levels 1-6. With MAX_LEVEL = 2 (q = 2),
            // glGenerateMipmap must replace levels 1 and 2 of both with arrays derived from the same base, so they become
            // identical, and leave levels 3-6 as they were (section 3.8.10.5).
            {
                const mgl::TextureId twin_tex = scratch[4];
                for (const mgl::TextureId t : {range_tex, twin_tex})
                {
                    mgl::glBindTexture(mgl::TextureTarget::Texture2D, t);
                    set_sampling_defaults();
                    upload_solid(0, kBaseSize, kUniform);
                    for (int k = 1; k < kLevels; ++k) upload_solid(k, kBaseSize >> k, t == range_tex ? kLadder[static_cast<std::size_t>(k)] : kBogus);
                    mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, 2);  // q = 2
                    mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                    mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, kUnbounded);
                }
                const std::vector<Rgba8> replaced = snapshot_levels(range_tex, lay, 1, 2), twin = snapshot_levels(twin_tex, lay, 1, 2);
                const long diff = count_differences(replaced, twin);
                const Stats kept = probe_levels(range_tex, lay, 3, 6, 0.0, [](int k) { return kLadder[static_cast<std::size_t>(k)]; });
                check.expect(diff == 0 && !replaced.empty(),
                             "MAX_LEVEL = 2: glGenerateMipmap replaced levels 1 and 2 (ladder and red previous contents give identical levels)",
                             std::format("{} of {} pixels differ", diff, replaced.size()));
                check.expect(kept.bad == 0, "MAX_LEVEL = 2: levels 3-6 are left unchanged (still exactly the ladder colours)",
                             kept.bad == 0 ? kept.summary() : kept.first_bad);
                std::cout << std::format("   note: MAX_LEVEL = 2: generated levels 1-2 of a uniform {} deviate from it by at most {}/255 (observation)\n",
                                         demo::format_rgba(kUniform), max_deviation(replaced, kUniform));
            }

            // --- non-power-of-two base: floor(w / 2^i), at least 1 ---------------------------------------------
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, npot_tex);
                set_sampling_defaults();
                const std::vector<Rgba8> texels(37 * 21, kUniform);
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 37, 21, 0, mgl::PixelFormat::Rgba,
                                  mgl::PixelType::UnsignedByte, texels.data());
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                const int expected_w[] = {37, 18, 9, 4, 2, 1}, expected_h[] = {21, 10, 5, 2, 1, 1};
                int bad = 0;
                std::string first;
                for (int k = 0; k < 6; ++k)
                {
                    const auto sz = query_size(npot_tex, k, lay);
                    if (sz[0] != expected_w[k] || sz[1] != expected_h[k])
                        if (bad++ == 0) first = std::format("level {}: {}x{}, expected {}x{}", k, sz[0], sz[1], expected_w[k], expected_h[k]);
                }
                check.expect(bad == 0, "37x21 base: generated levels are 18x10, 9x5, 4x2, 2x1, 1x1 (halved, rounded down, at least 1)", first);
            }

            // --- the smallest case: a literal 2x2 checker has exactly one derived level, 1x1 ---------------------------
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, tiny_tex);
                set_sampling_defaults();
                const std::array<Rgba8, 4> texels = {kChecker[0], kChecker[1], kChecker[1], kChecker[0]};
                upload(0, 2, texels.data());
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                const auto l1 = query_size(tiny_tex, 1, lay);
                // Level 0 along t = 0.25 (texel row 0: A, B). Before the call the texture had only level 0 and a mipmapped
                // min filter, i.e. it was incomplete and would sample (0,0,0,1); the exact level-0 colours prove completeness.
                Cell c0;
                c0.tex = tiny_tex;
                c0.model = &tiny_model_;
                c0.w0 = 2;
                c0.n0 = 0;
                c0.lambda0 = 0;
                c0.t = 0.25f;
                const Stats level0 = compare_model(c0, draw_scratch(c0, lay), {0, 0, lay.cw, lay.ch});
                check.expect(l1[0] == 1 && l1[1] == 1 && level0.bad == 0 && level0.probes > 0,
                             "2x2 checker: glGenerateMipmap defines a 1x1 level 1 (textureSize) and the texture is complete (level 0 samples its texels, not (0,0,0,1))",
                             std::format("level 1 is {}x{}; level 0: {}", l1[0], l1[1], level0.bad == 0 ? level0.summary() : level0.first_bad));
                Cell c;
                c.tex = tiny_tex;
                c.w0 = 2;
                c.lambda0 = 1;  // one level-1 texel per pixel: the 1x1 level
                const Image img = draw_scratch(c, lay);
                const Rgba8 mean_px = img.at(lay.cw / 2, lay.ch / 2);
                std::cout << std::format("   note: 1x1 level of the 2x2 checker = {} (the box-filter mean would be ({:.1f}, {:.1f}, {:.1f}); the spec does not require it)\n",
                                         demo::format_rgba(mean_px), 0.5 * (kChecker[0][0] + kChecker[1][0]),
                                         0.5 * (kChecker[0][1] + kChecker[1][1]), 0.5 * (kChecker[0][2] + kChecker[1][2]));
            }

            // --- glGenerateMipmap needs a colour-renderable AND texture-filterable base level (section 3.8.10.5) ----
            // RGBA8UI is colour-renderable but an integer format is not filterable; DEPTH_COMPONENT24 is not a colour
            // format at all. Both must give INVALID_OPERATION.
            {
                mgl::glBindTexture(mgl::TextureTarget::Texture2D, npot_tex);
                const std::array<std::uint32_t, 16> zeros{};
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8UI, 4, 4, 0, mgl::PixelFormat::RgbaInteger,
                                  mgl::PixelType::UnsignedByte, zeros.data());
                check.expect_no_gl_error("a 4x4 RGBA8UI upload is legal");
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                      "glGenerateMipmap on an RGBA8UI base level (renderable, not texture-filterable) -> INVALID_OPERATION");
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::DepthComponent24, 4, 4, 0, mgl::PixelFormat::DepthComponent,
                                  mgl::PixelType::UnsignedInt, zeros.data());
                check.expect_no_gl_error("a 4x4 DEPTH_COMPONENT24 upload is legal");
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                      "glGenerateMipmap on a DEPTH_COMPONENT24 base level (not colour-renderable) -> INVALID_OPERATION");

                // R32F: in core ES it is not texture-filterable (ES 3.0 Table 3.13 / ES 3.2 Table 8.10) and only ES 3.2
                // makes it color-renderable, so core alone requires INVALID_OPERATION. Extensions amend those tables:
                // GL_OES_texture_float_linear makes it filterable, GL_EXT_color_buffer_float (or ES 3.2) renderable. The
                // expectation is therefore computed from the context; no extension is *used*.
                const bool renderable = env_version_ >= demo::Version{3, 2} || mgl::HasExtension("GL_EXT_color_buffer_float");
                const bool filterable = mgl::HasExtension("GL_OES_texture_float_linear");
                mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::R32F, 4, 4, 0, mgl::PixelFormat::Red,
                                  mgl::PixelType::Float, zeros.data());
                check.expect_no_gl_error("a 4x4 R32F upload is legal");
                mgl::glGenerateMipmap(mgl::TextureTarget::Texture2D);
                if (renderable && filterable)
                    check.expect_no_gl_error("glGenerateMipmap on R32F succeeds: renderable and filterable on this context "
                                             "(ES 3.2 or EXT_color_buffer_float, plus OES_texture_float_linear)");
                else
                    check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                          "glGenerateMipmap on R32F -> INVALID_OPERATION (not both renderable and filterable here)");
            }

            mgl::glHint(mgl::HintTarget::GenerateMipmap, mgl::HintMode::Nicest);  // as in init(): the visible checker used NICEST
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteTextures(5, scratch);
        }

        // ------------------------------------------------------------------------------------------
        // Completeness (rows 11, 12)
        // ------------------------------------------------------------------------------------------
        void check_completeness(demo::Check& check, const Layout& lay, const Image& fb)
        {
            for (int s = 0; s < kScenarioCount; ++s)
            {
                const ScenarioInfo& info = kScenarioInfo[static_cast<std::size_t>(s)];
                const int row = 11 + s / 4, col = 2 * (s % 4);
                const Rect rb = rect_of(cell_at(row, col), lay), rt = rect_of(cell_at(row, col + 1), lay);
                const Stats black = compare_const(fb, rb, kBlack, 0);
                const Stats twin_ok = compare_const(fb, rt, info.expected, 0);
                check.expect(black.bad == 0 && twin_ok.bad == 0,
                             std::format("completeness {}: '{}' samples (0,0,0,1); the correct twin samples {}", s, info.what,
                                         demo::format_rgba(info.expected)),
                             black.bad != 0 ? "broken: " + black.first_bad : (twin_ok.bad != 0 ? "twin: " + twin_ok.first_bad : ""));

                // Fix the broken texture in place with one call and sample it again.
                repair_scenario(static_cast<Scenario>(s), broken(s));
                Cell c = scenario_cell(s, broken(s));
                const Image after = draw_scratch(c, lay);
                const Stats fixed = compare_const(after, {0, 0, lay.cw, lay.ch}, info.expected, 0);
                check.expect(fixed.bad == 0, std::format("completeness {}: {} makes the same texture complete -> {}", s, info.fix, demo::format_rgba(info.expected)),
                             fixed.bad == 0 ? "" : fixed.first_bad);
            }

            // Re-create the broken textures so that the final picture shows them as designed.
            const GLsizei n = kScenarioCount;
            std::array<mgl::TextureId, kScenarioCount> old{};
            for (int s = 0; s < kScenarioCount; ++s) old[static_cast<std::size_t>(s)] = broken(s);
            mgl::glDeleteTextures(n, old.data());
            std::array<mgl::TextureId, kScenarioCount> fresh{};
            mgl::glGenTextures(n, fresh.data());
            for (int s = 0; s < kScenarioCount; ++s)
            {
                tex_[kScenarioBase + 2 * static_cast<std::size_t>(s)] = fresh[static_cast<std::size_t>(s)];
                build_scenario(static_cast<Scenario>(s), fresh[static_cast<std::size_t>(s)], true);
            }
            for (Cell& c : cells_)
                if (c.row >= 11 && c.row <= 12 && c.col % 2 == 0)
                {
                    const int s = (c.row - 11) * 4 + c.col / 2;
                    if (s < kScenarioCount) c.tex = broken(s);
                }
        }

        // ------------------------------------------------------------------------------------------
        // Parameter queries and defined errors
        // ------------------------------------------------------------------------------------------
        void check_state_and_errors(demo::Check& check, const Layout& /*lay*/)
        {
            using Q = mgl::TextureParameterQuery;
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kLadderTex]);
            const mgl::TextureMinFilter filters[] = {mgl::TextureMinFilter::NearestMipmapNearest, mgl::TextureMinFilter::LinearMipmapNearest,
                                                     mgl::TextureMinFilter::NearestMipmapLinear, mgl::TextureMinFilter::LinearMipmapLinear};
            bool all_back = true;
            for (const auto f : filters)
            {
                mgl::glTexParameter(mgl::TextureTarget::Texture2D, f);
                GLint v = 0;
                mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, Q::MinFilter, &v);
                all_back = all_back && v == static_cast<GLint>(f);
            }
            check.expect(all_back, "TEXTURE_MIN_FILTER reads back each of the four mipmap filters");
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::NearestMipmapNearest);

            // A fresh texture has TEXTURE_BASE_LEVEL 0, TEXTURE_MAX_LEVEL 1000, TEXTURE_MIN_LOD -1000, TEXTURE_MAX_LOD 1000
            // (ES 3.0 Table 6.9; section 3.8.10: "chosen so as to never clamp the normal range").
            mgl::TextureId fresh{};
            mgl::glGenTextures(1, &fresh);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, fresh);
            GLint base = -1, maxl = -1;
            GLfloat minlod = 0, maxlod = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, Q::BaseLevel, &base);
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, Q::MaxLevel, &maxl);
            mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, Q::MinLod, &minlod);
            mgl::glGetTexParameterfv(mgl::TextureTarget::Texture2D, Q::MaxLod, &maxlod);
            check.expect(base == 0 && maxl == 1000 && minlod == -1000.0f && maxlod == 1000.0f,
                         "initial TEXTURE_BASE_LEVEL 0, TEXTURE_MAX_LEVEL 1000, TEXTURE_MIN_LOD -1000, TEXTURE_MAX_LOD 1000",
                         std::format("{} {} {} {}", base, maxl, minlod, maxlod));
            GLint min_filter = 0;
            mgl::glGetTexParameteriv(mgl::TextureTarget::Texture2D, Q::MinFilter, &min_filter);
            check.expect_eq(min_filter, static_cast<GLint>(GL_NEAREST_MIPMAP_LINEAR), "initial TEXTURE_MIN_FILTER is NEAREST_MIPMAP_LINEAR");

            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::BaseLevel, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "TEXTURE_BASE_LEVEL = -1 -> INVALID_VALUE (section 3.8.10.4)");
            mgl::glTexParameteri(mgl::TextureTarget::Texture2D, mgl::TextureParameterSetter::MaxLevel, -1);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "TEXTURE_MAX_LEVEL = -1 -> INVALID_VALUE (section 3.8.10.4)");
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &fresh);
        }

        // ------------------------------------------------------------------------------------------
        // Negative controls: the comparisons above can fail
        // ------------------------------------------------------------------------------------------
        void check_controls(demo::Check& check, const Layout& lay, const Image& /*fb*/)
        {
            // (1) Redraw the LINEAR_MIPMAP_NEAREST stripe cells (row 7, k = 1..3) with the ladder texture instead of the
            // stripes. Level k of the ladder is solid, so the result is exactly that colour (+-1) on any implementation; the
            // stripes range check must reject it at more than a quarter of the pixels, i.e. the range is narrow enough to
            // catch a wrong texture or level.
            {
                Stats wrong;
                for (int k = 1; k <= 3; ++k)
                {
                    Cell c = cell_at(7, k);
                    Cell drawn = c;
                    drawn.tex = tex_[kLadderTex];
                    wrong.merge(compare_model(c, draw_scratch(drawn, lay), {0, 0, lay.cw, lay.ch}));
                }
                check.expect(wrong.probes > 0 && wrong.bad * 4 > wrong.probes,
                             "negative control: the ladder drawn into the LINEAR_MIPMAP_NEAREST stripe cells is rejected by the stripes range",
                             std::format("{} of {} pixels rejected", wrong.bad, wrong.probes));
            }

            // (2) Redraw the blend cell (row 3, k = 2, lambda = 2.5) with the non-mipmapped min filter
            // NEAREST (always level 0): the colour changes to level 0's and the blend range rejects it.
            {
                Cell c = cell_at(3, 2);
                c.min = mgl::TextureMinFilter::Nearest;
                const Image img = draw_scratch(c, lay);
                const Range b = blend(kLadder[2], kLadder[3]);
                const Stats rejected = compare_range(img, {0, 0, lay.cw, lay.ch}, b);
                const Stats level0 = compare_const(img, {0, 0, lay.cw, lay.ch}, kLadder[0], 0);
                check.expect(rejected.bad == rejected.probes && level0.bad == 0,
                             "negative control: switching the min filter to NEAREST turns the blend cell into level 0's colour, which the blend check rejects",
                             std::format("{} of {} pixels rejected", rejected.bad, rejected.probes));
            }
        }

        // ------------------------------------------------------------------------------------------
        mgl::ProgramId program_{};
        mgl::UniformLocation loc_s0_{}, loc_range_{}, loc_t_{}, loc_mode_{}, loc_lod_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        Model ladder_model_, stripes_model_, checker_model_, tiny_model_;
        std::vector<Cell> cells_;
        int initial_hint_ = 0;
        int hint_after_set_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es20-mipmaps",
        .title = "Mipmaps: level of detail, the four mipmap filters, glGenerateMipmap, completeness",
        .min_es = {3, 0},
        .summary = "Hand-made mip chains sampled at chosen levels of detail with all four mipmap filters, BASE/MAX_LEVEL and "
                   "MIN/MAX_LOD, glGenerateMipmap and its hint, and incomplete mip chains, all checked against a CPU model.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Mipmaps>(argc, argv, info);
}
