// es30-vertex-array-objects
//
// Vertex array objects (ES 3.0 section 2.11) and the ES 3.0 vertex formats that make them
// interesting: pure-integer attributes (glVertexAttribIPointer, glVertexAttribI4*), the packed
// 2_10_10_10 formats and HALF_FLOAT (section 2.9, 2.9.2).
//
// The 256x256 surface is a 4x4 grid of 64x64 cells (row 3 is the top row):
//
//   row 3:  VAO A    VAO B    VAO C    VAO D        five different VAOs, drawn in this order ...
//   row 2:  VAO D    VAO C    VAO B    VAO A        ... and again in reverse: switching back and forth
//   row 1:  int I1   int I2   int I3   int I4       ivec4/uvec4 inputs: arrays and constant attributes
//   row 0:  UINT 2_10_10_10   INT 2_10_10_10   HALF_FLOAT   VAO E
//
// Every pixel is compared with a CPU model (a list of Gouraud-shaded triangles built from the same
// source data, with the spec's conversion formulas for normalized and packed data). The state of
// each VAO, the bindings that are and are not part of it, the VAO name lifecycle and the errors the
// spec defines are checked with glGet* queries.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    // -----------------------------------------------------------------------------------------
    // Layout and constants
    // -----------------------------------------------------------------------------------------
    constexpr int kGrid = 4;                                        // 4x4 cells
    constexpr std::array<std::uint8_t, 4> kClear8 = {20, 30, 50, 255};  // exact 8-bit values, no rounding

    constexpr mgl::AttribLocation kLocPos{0u};   // layout(location = 0) in both programs
    constexpr mgl::AttribLocation kLocAttr{1u};  // vec4 a_attr / ivec4 a_i
    constexpr mgl::AttribLocation kLocUint{2u};  // uvec4 a_u (integer program only)

    // -----------------------------------------------------------------------------------------
    // Source data. The CPU model below is built from these same arrays.
    // -----------------------------------------------------------------------------------------

    // VAO A: two tightly packed float buffers (positions: vec2, colors: vec3 -> alpha defaults to 1).
    constexpr std::array<float, 6> kPosA = {-0.8f, -0.7f, 0.8f, -0.7f, -0.1f, 0.8f};
    constexpr std::array<float, 9> kColA = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    // VAO B: one interleaved buffer, 8-byte stride: normalized ubyte color at offset 0, normalized short
    // position at offset 4. A TRIANGLE_STRIP of four vertices.
    struct VertexB
    {
        std::uint8_t r, g, b, a;
        std::int16_t x, y;
    };
    constexpr std::int16_t snorm16(double v)
    {
        return static_cast<std::int16_t>(v * 32767.0 + (v < 0 ? -0.5 : 0.5));
    }
    constexpr std::array<VertexB, 4> kVertsB = {{
        {255, 255, 0, 255, snorm16(-0.6), snorm16(-0.6)},
        {0, 255, 255, 255, snorm16(+0.6), snorm16(-0.6)},
        {255, 0, 255, 255, snorm16(-0.6), snorm16(+0.6)},
        {255, 255, 255, 255, snorm16(+0.6), snorm16(+0.6)},
    }};

    // VAOs C and D: the *same* vertex buffer and attribute layout (float position + normalized ubyte
    // color, 12-byte stride); they differ only in the element array buffer bound inside the VAO.
    struct VertexCD
    {
        float x, y;
        std::uint8_t r, g, b, a;
    };
    constexpr std::array<VertexCD, 5> kVertsCD = {{
        {-0.8f, -0.8f, 255, 0, 0, 255},      // 0 bottom-left, red
        {+0.8f, -0.8f, 0, 255, 0, 255},      // 1 bottom-right, green
        {+0.8f, +0.8f, 0, 0, 255, 255},      // 2 top-right, blue
        {-0.8f, +0.8f, 255, 255, 0, 255},    // 3 top-left, yellow
        {0.0f, 0.0f, 255, 255, 255, 255},    // 4 center, white
    }};
    constexpr std::array<std::uint16_t, 6> kIndexC = {0, 1, 4, 2, 3, 4};     // bottom + top wedge
    constexpr std::array<std::uint8_t, 8> kIndexD = {9, 9, 1, 2, 4, 3, 0, 4};  // right + left wedge; the first 2 bytes are skipped
    constexpr std::uintptr_t kIndexDOffset = 2;

    // VAO E: attribute 1 disabled; the color comes from glVertexAttrib4f.
    constexpr std::array<float, 4> kConstColor = {1.0f, 0.5f, 0.0f, 1.0f};

    // Integer cells ------------------------------------------------------------------------------
    constexpr std::int32_t kIntMin = std::numeric_limits<std::int32_t>::min();
    struct IntCase
    {
        std::array<std::int32_t, 4> i;
        std::array<std::uint32_t, 4> u;
    };
    // I1: glVertexAttribIPointer(INT / UNSIGNED_INT). 16777217 = 2^24 + 1 and 0x80000001 are not representable
    // as a 32-bit float (24-bit significand): a float conversion would round them.
    constexpr IntCase kInt1 = {{16777217, -16777217, 2147483647, kIntMin}, {0xFFFFFFFFu, 16777217u, 0x80000001u, 0x7FFFFFFFu}};
    // I2: glVertexAttribIPointer(SHORT size 3 / UNSIGNED_BYTE size 2): sign- resp. zero-extended; missing y/z = 0, w = 1.
    constexpr std::array<std::int16_t, 3> kShorts = {-32768, 32767, -1};
    constexpr std::array<std::uint8_t, 2> kBytes = {255, 128};
    constexpr IntCase kInt2 = {{-32768, 32767, -1, 1}, {255u, 128u, 0u, 1u}};
    // I3: glVertexAttribI4i / glVertexAttribI4ui (arrays disabled).
    constexpr IntCase kInt3 = {{2147483647, kIntMin, 16777217, -5}, {0xFFFFFFFFu, 0x80000001u, 16777217u, 0u}};
    // I4: glVertexAttribI4iv / glVertexAttribI4uiv.
    constexpr IntCase kInt4 = {{-16777219, 33554435, 7, -2147483647}, {3000000000u, 16777219u, 1u, 4294967294u}};

    struct VertexI1
    {
        float x, y;
        std::int32_t i[4];
        std::uint32_t u[4];
    };  // 40 bytes
    struct VertexI2
    {
        float x, y;
        std::int16_t s[3];
        std::uint8_t b[2];
    };  // 16 bytes
    constexpr std::array<float, 8> kQuad = {-1, -1, 1, -1, -1, 1, 1, 1};  // TRIANGLE_STRIP covering a cell

    // Packed formats -----------------------------------------------------------------------------
    constexpr std::uint32_t pack(unsigned x, unsigned y, unsigned z, unsigned w)
    {
        return (x & 0x3FFu) | (y & 0x3FFu) << 10 | (z & 0x3FFu) << 20 | (w & 3u) << 30;
    }
    constexpr std::uint32_t pack_s(int x, int y, int z, int w)  // two's complement fields
    {
        return pack(static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z), static_cast<unsigned>(w));
    }
    // Two test words per format: (x, y, z, w).
    constexpr std::array<std::uint32_t, 2> kWordsU = {pack(1023, 512, 100, 3), pack(0, 700, 1023, 1)};
    // For INT_2_10_10_10_REV the first word holds the extreme values and w = 1; the second holds w = -2: with eq. 2.2
    // (max(c / (2^(b-1) - 1), -1)) a 2-bit w of -2 clamps to -1 and the normalized and raw results differ.
    constexpr std::array<std::uint32_t, 2> kWordsS = {pack_s(511, 0, -512, 1), pack_s(-256, 300, -511, -2)};

    struct PackedVertex
    {
        float x, y;
        std::uint32_t word;
    };  // 12 bytes: position at 0, packed word at 8

    // Half float ---------------------------------------------------------------------------------
    struct HalfVertex
    {
        std::uint16_t x, y, r, g, b, a;
    };  // 12 bytes: position at 0, color at 4
    constexpr std::array<std::array<double, 4>, 4> kHalfColors = {{
        {0.5, 25.0, 75.5, 100.0},
        {100.0, 50.0, 2.5, 80.0},
        {1.5, 99.0, 33.25, 50.0},
        {12.5, 12.5, 90.0, 100.0},
    }};
    constexpr double kHalfScale = 0.01;  // the shader maps the half values (0..100) to 0..1

    // IEEE 754 binary16 <-> double, used only on the CPU side to build vertex data and expectations.
    double half_to_double(std::uint16_t h)
    {
        const int sign = (h >> 15) & 1;
        const int exponent = (h >> 10) & 0x1F;
        const int mantissa = h & 0x3FF;
        double v;
        if (exponent == 0) v = std::ldexp(mantissa, -24);                      // subnormal
        else if (exponent == 31) v = mantissa ? std::nan("") : HUGE_VAL;       // NaN / infinity
        else v = std::ldexp(1024 + mantissa, exponent - 25);                   // (1 + m/1024) * 2^(e-15)
        return sign ? -v : v;
    }
    std::uint16_t double_to_half(double v)  // round to nearest even; finite values in the normal range only
    {
        const std::uint16_t sign = v < 0 ? 0x8000 : 0;
        v = std::abs(v);
        if (v == 0.0) return sign;
        int e = 0;
        const double m = std::frexp(v, &e);                       // v = m * 2^e, m in [0.5, 1)
        long mant = std::lround(std::nearbyint((m * 2.0 - 1.0) * 1024.0));
        int exponent = e - 1 + 15;
        if (mant == 1024) { mant = 0; ++exponent; }
        return static_cast<std::uint16_t>(sign | (exponent << 10) | mant);
    }

    // -----------------------------------------------------------------------------------------
    // CPU model: a scene is a list of Gouraud-shaded triangles in NDC, drawn in order.
    // -----------------------------------------------------------------------------------------
    using Rgba = std::array<double, 4>;  // 0..1
    struct Vtx
    {
        double x, y;
        Rgba c;
    };
    using Tri = std::array<Vtx, 3>;
    enum class Cover { Outside, Inside, Unsure };
    struct Sample
    {
        Cover cover = Cover::Outside;
        Rgba color{};
    };

    // Classifies the pixel center (px, py) of a cw x ch viewport. Within `margin` pixels of an edge the answer
    // depends on the rasterizer's tie-breaking rules and is reported as Unsure (never compared).
    Sample sample_triangle(const Tri& t, double px, double py, double cw, double ch, double margin)
    {
        std::array<double, 3> X{}, Y{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            X[i] = (t[i].x + 1.0) * 0.5 * cw;
            Y[i] = (t[i].y + 1.0) * 0.5 * ch;
        }
        auto edge = [](double ax, double ay, double bx, double by, double x, double y) {
            return (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        };
        const double area = edge(X[0], Y[0], X[1], Y[1], X[2], Y[2]);
        if (std::abs(area) < 1e-9) return {};
        const double sign = area > 0 ? 1.0 : -1.0;
        double min_distance = 1e30;
        std::array<double, 3> w{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            const std::size_t j = (i + 1) % 3, k = (i + 2) % 3;
            const double e = edge(X[j], Y[j], X[k], Y[k], px, py);
            min_distance = std::min(min_distance, sign * e / std::hypot(X[k] - X[j], Y[k] - Y[j]));
            w[i] = e / area;  // barycentric weight of vertex i
        }
        if (min_distance < -margin) return {Cover::Outside, {}};
        if (min_distance <= margin) return {Cover::Unsure, {}};
        Rgba c{};
        for (std::size_t k = 0; k < 4; ++k) c[k] = w[0] * t[0].c[k] + w[1] * t[1].c[k] + w[2] * t[2].c[k];
        return {Cover::Inside, c};
    }

    // Later triangles overwrite earlier ones (no depth test): the last definitely-covering triangle wins, and a
    // pixel that a later, uncertain triangle might also have touched is Unsure.
    Sample sample_scene(const std::vector<Tri>& tris, double px, double py, double cw, double ch)
    {
        Sample result;
        bool unsure_after = false;
        for (const Tri& t : tris)
        {
            const Sample s = sample_triangle(t, px, py, cw, ch, 1.0);
            if (s.cover == Cover::Inside) { result = s; unsure_after = false; }
            else if (s.cover == Cover::Unsure) unsure_after = true;
        }
        return unsure_after ? Sample{Cover::Unsure, {}} : result;
    }

    std::uint8_t to8(double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

    bool close(const Rgba8& a, const Rgba8& b, int tolerance)
    {
        for (std::size_t k = 0; k < 4; ++k)
            if (std::abs(int(a[k]) - int(b[k])) > tolerance) return false;
        return true;
    }

    // What integer.frag writes for a matching sub-area: green 1 - 0.2 * component, blue tint 0.4 in the uvec4 row.
    Rgba8 int_ok_color(int comp, bool upper)
    {
        return {0, to8(1.0 - 0.2 * comp), upper ? std::uint8_t{0} : to8(0.4), 255};
    }

    struct Scan
    {
        int inside = 0, outside = 0, unsure = 0, bad = 0, max_dev = 0;
        std::string first_bad;
    };

    // Reads the whole cell and compares every pixel the model is certain about.
    Scan scan_cell(const std::vector<Tri>& scene, int ox, int oy, int cw, int ch, int tolerance)
    {
        Scan r;
        const std::vector<std::uint8_t> pixels = demo::read_rgba8(ox, oy, cw, ch);
        for (int y = 0; y < ch; ++y)
            for (int x = 0; x < cw; ++x)
            {
                const Sample s = sample_scene(scene, x + 0.5, y + 0.5, cw, ch);
                if (s.cover == Cover::Unsure) { ++r.unsure; continue; }
                Rgba8 want{};
                for (std::size_t k = 0; k < 4; ++k) want[k] = s.cover == Cover::Inside ? to8(s.color[k]) : kClear8[k];
                ++(s.cover == Cover::Inside ? r.inside : r.outside);
                const std::uint8_t* got = &pixels[static_cast<std::size_t>((y * cw + x) * 4)];
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int(got[k]) - int(want[k]));
                    r.max_dev = std::max(r.max_dev, dev);
                    ok = ok && dev <= tolerance;
                }
                if (!ok && r.bad++ == 0)
                    r.first_bad = std::format("first mismatch at cell pixel ({},{}): got {} want {}", x, y,
                                              demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return r;
    }

    void expect_scan(demo::Check& check, const Scan& s, int min_inside, int min_outside, const std::string& what)
    {
        check.expect(s.bad == 0 && s.inside >= min_inside && s.outside >= min_outside,
                     std::format("{}: {} covered + {} background pixels match the model (max deviation {}/255, {} edge pixels skipped)", what,
                                 s.inside, s.outside, s.max_dev, s.unsure),
                     s.bad ? std::format("{} mismatches; {}", s.bad, s.first_bad)
                     : (s.inside < min_inside || s.outside < min_outside)
                         ? std::format("only {} covered / {} background pixels compared", s.inside, s.outside)
                         : std::string{});
    }

    // Conversions of ES 3.0 section 2.1.6.1 and the packed layouts of section 2.9.2 ---------------
    int sign_extend(std::uint32_t v, int bits) { return static_cast<int>(v << (32 - bits)) >> (32 - bits); }

    // `legacy` selects the pre-ES-3.0 signed formula (2c + 1) / (2^b - 1), used only as a mutation to show
    // that the pixel comparison can tell the formulas apart.
    Rgba decode_packed(std::uint32_t word, bool is_signed, bool normalized, bool legacy = false)
    {
        Rgba f{};
        for (std::size_t k = 0; k < 4; ++k)
        {
            const int bits = k < 3 ? 10 : 2;
            const std::uint32_t field = (word >> (k < 3 ? 10 * k : 30)) & (k < 3 ? 0x3FFu : 0x3u);
            if (!is_signed)
                f[k] = normalized ? field / double((1u << bits) - 1) : double(field);  // eq. 2.1
            else
            {
                const int c = sign_extend(field, bits);
                if (!normalized) f[k] = c;
                else if (legacy) f[k] = std::max((2.0 * c + 1.0) / double((1 << bits) - 1), -1.0);
                else f[k] = std::max(c / double((1 << (bits - 1)) - 1), -1.0);  // eq. 2.2
            }
        }
        return f;
    }

    struct Xform  // color = attribute * scale + bias (the u_scale / u_bias uniforms)
    {
        std::array<float, 4> scale, bias;
    };
    constexpr Xform kIdentity = {{1, 1, 1, 1}, {0, 0, 0, 0}};
    constexpr Xform kUintNorm = kIdentity;
    constexpr Xform kUintRaw = {{1.0f / 1023, 1.0f / 1023, 1.0f / 1023, 1.0f / 3}, {0, 0, 0, 0}};
    constexpr Xform kIntNorm = {{0.5f, 0.5f, 0.5f, 0.25f}, {0.5f, 0.5f, 0.5f, 0.75f}};
    constexpr Xform kIntRaw = {{1.0f / 1022, 1.0f / 1022, 1.0f / 1022, 0.25f}, {0.5f, 0.5f, 0.5f, 0.75f}};
    constexpr Xform kHalfXform = {{0.01f, 0.01f, 0.01f, 0.01f}, {0, 0, 0, 0}};

    Rgba shade(const Rgba& f, const Xform& x)
    {
        Rgba c{};
        for (std::size_t k = 0; k < 4; ++k) c[k] = f[k] * double(x.scale[k]) + double(x.bias[k]);
        return c;
    }

    // Two quads (upper, lower) in the left or right half of a cell, each a flat color given by its packed word.
    std::vector<PackedVertex> packed_quads(bool right_half, const std::array<std::uint32_t, 2>& words)
    {
        std::vector<PackedVertex> v;
        const float x0 = right_half ? 0.1f : -0.9f, x1 = right_half ? 0.9f : -0.1f;
        for (std::size_t q = 0; q < 2; ++q)
        {
            const float y0 = q == 0 ? 0.1f : -0.9f, y1 = q == 0 ? 0.9f : -0.1f;
            const std::uint32_t w = words[q];
            for (const auto& p : {std::array<float, 2>{x0, y0}, {x1, y0}, {x0, y1}, {x0, y1}, {x1, y0}, {x1, y1}})
                v.push_back({p[0], p[1], w});
        }
        return v;
    }

    // Four quads (2 x 2) with half-float positions and colors.
    std::vector<HalfVertex> half_quads()
    {
        std::vector<HalfVertex> v;
        for (std::size_t q = 0; q < 4; ++q)
        {
            const double x0 = (q % 2) ? 0.1 : -0.9, x1 = (q % 2) ? 0.9 : -0.1;
            const double y0 = (q / 2) ? -0.9 : 0.1, y1 = (q / 2) ? -0.1 : 0.9;
            const auto& c = kHalfColors[q];
            for (const auto& p : {std::array<double, 2>{x0, y0}, {x1, y0}, {x0, y1}, {x0, y1}, {x1, y0}, {x1, y1}})
                v.push_back({double_to_half(p[0]), double_to_half(p[1]), double_to_half(c[0]), double_to_half(c[1]),
                             double_to_half(c[2]), double_to_half(c[3])});
        }
        return v;
    }

    // Triangle lists --------------------------------------------------------------------------------
    std::vector<Tri> tris_packed(const std::vector<PackedVertex>& verts, bool is_signed, bool normalized, const Xform& x,
                                 bool legacy = false)
    {
        std::vector<Tri> out;
        for (std::size_t i = 0; i + 2 < verts.size(); i += 3)
        {
            Tri t{};
            for (std::size_t k = 0; k < 3; ++k)
            {
                const PackedVertex& p = verts[i + k];
                t[k] = {double(p.x), double(p.y), shade(decode_packed(p.word, is_signed, normalized, legacy), x)};
            }
            out.push_back(t);
        }
        return out;
    }

    std::vector<Tri> tris_half(const std::vector<HalfVertex>& verts)
    {
        std::vector<Tri> out;
        for (std::size_t i = 0; i + 2 < verts.size(); i += 3)
        {
            Tri t{};
            for (std::size_t k = 0; k < 3; ++k)
            {
                const HalfVertex& p = verts[i + k];
                const Rgba f = {half_to_double(p.r), half_to_double(p.g), half_to_double(p.b), half_to_double(p.a)};
                t[k] = {half_to_double(p.x), half_to_double(p.y), shade(f, kHalfXform)};
            }
            out.push_back(t);
        }
        return out;
    }

    enum Slot : std::size_t { kA, kB, kC, kD, kE, kI1, kI2, kI3, kI4, kPuNorm, kPuRaw, kPiNorm, kPiRaw, kHalf, kSlotCount };
    constexpr std::array<Slot, 4> kPass1 = {kA, kB, kC, kD};
    constexpr std::array<Slot, 4> kPass2 = {kD, kC, kB, kA};
    constexpr std::array<const char*, kSlotCount> kSlotName = {
        "A (two tight float buffers)", "B (interleaved snorm short + unorm ubyte)", "C (indexed, USHORT element buffer)",
        "D (indexed, UBYTE element buffer)", "E (color array disabled)", "I1", "I2", "I3", "I4", "UINT norm", "UINT raw", "INT norm",
        "INT raw", "HALF_FLOAT"};

    struct Rect
    {
        int x, y, w, h;
    };

    struct AttribState
    {
        int enabled = 0, size = 4, type = GL_FLOAT, normalized = 0, stride = 0, integer = 0, divisor = 0;
        unsigned buffer = 0;
        std::uintptr_t pointer = 0;
    };

    class VertexArrayObjects final : public demo::Demo
    {
    public:
        // ===========================================================================================
        // init: build programs, buffers and the VAOs. Every VAO is configured while it is bound; at the
        // end nothing is bound, which proves that the draws below use only state stored in the VAOs.
        // ===========================================================================================
        void init(demo::Env& env) override
        {
            prog_attribs_ = demo::build_program(env, "shaders/attribs.vert", "shaders/attribs.frag");
            prog_int_ = demo::build_program(env, "shaders/integer.vert", "shaders/integer.frag");
            u_scale_ = mgl::glGetUniformLocation(prog_attribs_, "u_scale");
            u_bias_ = mgl::glGetUniformLocation(prog_attribs_, "u_bias");
            u_expect_i_ = mgl::glGetUniformLocation(prog_int_, "u_expect_i");
            u_expect_u_ = mgl::glGetUniformLocation(prog_int_, "u_expect_u");

            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());

            // --- VAO A: positions and colors in two tightly packed float buffers ---------------------
            vbo_pos_a_ = make_array_buffer(std::span<const float>(kPosA));
            vbo_col_a_ = make_array_buffer(std::span<const float>(kColA));
            mgl::glBindVertexArray(vao_[kA]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_pos_a_);
            mgl::glEnableVertexAttribArray(kLocPos);
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_col_a_);  // a different ARRAY_BUFFER binding for attribute 1
            mgl::glEnableVertexAttribArray(kLocAttr);
            mgl::glVertexAttribPointer(kLocAttr, 3, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));

            // --- VAO B: one interleaved buffer; normalized short positions, normalized ubyte colors -----
            vbo_b_ = make_array_buffer(std::span<const VertexB>(kVertsB));
            mgl::glBindVertexArray(vao_[kB]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_b_);
            mgl::glEnableVertexAttribArray(kLocPos);
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Short, GL_TRUE, sizeof(VertexB), demo::buffer_offset(offsetof(VertexB, x)));
            mgl::glEnableVertexAttribArray(kLocAttr);
            mgl::glVertexAttribPointer(kLocAttr, 4, mgl::DataType::UnsignedByte, GL_TRUE, sizeof(VertexB), demo::buffer_offset(offsetof(VertexB, r)));

            // --- VAOs C and D: same vertex data and layout, different element array buffers -----------
            vbo_cd_ = make_array_buffer(std::span<const VertexCD>(kVertsCD));
            for (const Slot s : {kC, kD})
            {
                mgl::glBindVertexArray(vao_[s]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_cd_);
                mgl::glEnableVertexAttribArray(kLocPos);
                mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, sizeof(VertexCD), demo::buffer_offset(offsetof(VertexCD, x)));
                mgl::glEnableVertexAttribArray(kLocAttr);
                mgl::glVertexAttribPointer(kLocAttr, 4, mgl::DataType::UnsignedByte, GL_TRUE, sizeof(VertexCD), demo::buffer_offset(offsetof(VertexCD, r)));
            }
            // GL_ELEMENT_ARRAY_BUFFER, unlike GL_ARRAY_BUFFER, *is* VAO state: it is attached to whichever VAO is bound now.
            mgl::glBindVertexArray(vao_[kC]);
            ebo_c_ = make_element_buffer(std::span<const std::uint16_t>(kIndexC));
            mgl::glBindVertexArray(vao_[kD]);
            ebo_d_ = make_element_buffer(std::span<const std::uint8_t>(kIndexD));

            // --- VAO E: only the position array is enabled; the color is the "current" generic attribute ---
            mgl::glBindVertexArray(vao_[kE]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_pos_a_);
            mgl::glEnableVertexAttribArray(kLocPos);
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            // attribute 1 stays disabled (initial state)

            // --- integer VAOs -------------------------------------------------------------------------
            vbo_quad_ = make_array_buffer(std::span<const float>(kQuad));
            {
                std::vector<VertexI1> v;
                for (std::size_t k = 0; k < 4; ++k)
                    v.push_back({kQuad[2 * k], kQuad[2 * k + 1],
                                 {kInt1.i[0], kInt1.i[1], kInt1.i[2], kInt1.i[3]}, {kInt1.u[0], kInt1.u[1], kInt1.u[2], kInt1.u[3]}});
                vbo_i1_ = make_array_buffer(std::span<const VertexI1>(v));
                mgl::glBindVertexArray(vao_[kI1]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_i1_);
                mgl::glEnableVertexAttribArray(kLocPos);
                mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, sizeof(VertexI1), demo::buffer_offset(offsetof(VertexI1, x)));
                mgl::glEnableVertexAttribArray(kLocAttr);
                mgl::glVertexAttribIPointer(kLocAttr, 4, mgl::DataType::Int, sizeof(VertexI1), demo::buffer_offset(offsetof(VertexI1, i)));
                mgl::glEnableVertexAttribArray(kLocUint);
                mgl::glVertexAttribIPointer(kLocUint, 4, mgl::DataType::UnsignedInt, sizeof(VertexI1), demo::buffer_offset(offsetof(VertexI1, u)));
            }
            {
                std::vector<VertexI2> v;
                for (std::size_t k = 0; k < 4; ++k)
                    v.push_back({kQuad[2 * k], kQuad[2 * k + 1], {kShorts[0], kShorts[1], kShorts[2]}, {kBytes[0], kBytes[1]}});
                vbo_i2_ = make_array_buffer(std::span<const VertexI2>(v));
                mgl::glBindVertexArray(vao_[kI2]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_i2_);
                mgl::glEnableVertexAttribArray(kLocPos);
                mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, sizeof(VertexI2), demo::buffer_offset(offsetof(VertexI2, x)));
                mgl::glEnableVertexAttribArray(kLocAttr);
                mgl::glVertexAttribIPointer(kLocAttr, 3, mgl::DataType::Short, sizeof(VertexI2), demo::buffer_offset(offsetof(VertexI2, s)));
                mgl::glEnableVertexAttribArray(kLocUint);
                mgl::glVertexAttribIPointer(kLocUint, 2, mgl::DataType::UnsignedByte, sizeof(VertexI2), demo::buffer_offset(offsetof(VertexI2, b)));
            }
            for (const Slot s : {kI3, kI4})  // constant attributes: only the position array
            {
                mgl::glBindVertexArray(vao_[s]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_quad_);
                mgl::glEnableVertexAttribArray(kLocPos);
                mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            }

            // --- packed-format VAOs: float position + one packed 32-bit word per vertex ----------------
            make_packed_vao(kPuNorm, mgl::DataType::UnsignedInt2101010Rev, true, false, kWordsU);
            make_packed_vao(kPuRaw, mgl::DataType::UnsignedInt2101010Rev, false, true, kWordsU);
            make_packed_vao(kPiNorm, mgl::DataType::Int2101010Rev, true, false, kWordsS);
            make_packed_vao(kPiRaw, mgl::DataType::Int2101010Rev, false, true, kWordsS);

            // --- HALF_FLOAT VAO: half position (vec2 at offset 0) and half color (vec4 at offset 4) ---
            {
                const std::vector<HalfVertex> v = half_quads();
                vbo_half_ = make_array_buffer(std::span<const HalfVertex>(v));
                mgl::glBindVertexArray(vao_[kHalf]);
                mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_half_);
                mgl::glEnableVertexAttribArray(kLocPos);
                mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::HalfFloat, GL_FALSE, sizeof(HalfVertex), demo::buffer_offset(offsetof(HalfVertex, x)));
                mgl::glEnableVertexAttribArray(kLocAttr);
                mgl::glVertexAttribPointer(kLocAttr, 4, mgl::DataType::HalfFloat, GL_FALSE, sizeof(HalfVertex), demo::buffer_offset(offsetof(HalfVertex, r)));
            }

            // Nothing stays bound: the VAOs alone must carry the vertex state from here on.
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);  // exact color writes
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            draw_cells(env);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_pixels(env, check);
            check_negative_controls(env, check);
            render(env, 0);  // restore the capture image after the controls
            check_vao_state(check);
            check_lifecycle(check);
            check_current_attributes(env, check);
            check_errors(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
            mgl::glDeleteProgram(prog_attribs_);
            mgl::glDeleteProgram(prog_int_);
        }

    private:
        // ---------------------------------------------------------------------------------------------
        // Buffers
        // ---------------------------------------------------------------------------------------------
        template <typename T>
        mgl::BufferId make_array_buffer(std::span<const T> data)
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::Array, b);
            mgl::glBufferData(mgl::BufferTarget::Array, data, mgl::BufferUsage::StaticDraw);
            buffers_.push_back(b);
            return b;
        }
        // Binds to ELEMENT_ARRAY_BUFFER: this changes the state of the *currently bound* VAO.
        template <typename T>
        mgl::BufferId make_element_buffer(std::span<const T> data)
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, b);
            mgl::glBufferData(mgl::BufferTarget::ElementArray, data, mgl::BufferUsage::StaticDraw);
            buffers_.push_back(b);
            return b;
        }

        void make_packed_vao(Slot slot, mgl::DataType type, bool normalized, bool right_half, const std::array<std::uint32_t, 2>& words)
        {
            const std::vector<PackedVertex> v = packed_quads(right_half, words);
            const mgl::BufferId vbo = make_array_buffer(std::span<const PackedVertex>(v));
            packed_vbo_[slot] = vbo;
            mgl::glBindVertexArray(vao_[slot]);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo);
            mgl::glEnableVertexAttribArray(kLocPos);
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, sizeof(PackedVertex), demo::buffer_offset(offsetof(PackedVertex, x)));
            mgl::glEnableVertexAttribArray(kLocAttr);
            // size must be 4 for the packed types (ES 3.0 section 2.9: INVALID_OPERATION otherwise)
            mgl::glVertexAttribPointer(kLocAttr, 4, type, normalized ? GL_TRUE : GL_FALSE, sizeof(PackedVertex), demo::buffer_offset(offsetof(PackedVertex, word)));
        }

        // ---------------------------------------------------------------------------------------------
        // Drawing
        // ---------------------------------------------------------------------------------------------
        static Rect cell(const demo::Env& env, int row, int col)
        {
            const int cw = env.width / kGrid, ch = env.height / kGrid;
            return {col * cw, row * ch, cw, ch};
        }
        static void set_viewport(Rect r) { mgl::glViewport(r.x, r.y, r.w, r.h); }

        void set_xform(const Xform& x) const
        {
            mgl::glUniform4f(u_scale_, x.scale[0], x.scale[1], x.scale[2], x.scale[3]);
            mgl::glUniform4f(u_bias_, x.bias[0], x.bias[1], x.bias[2], x.bias[3]);
        }

        // Draws one VAO of the float program into a cell. The ONLY thing that selects the geometry,
        // buffers, formats and indices is glBindVertexArray.
        void draw_float_cell(Rect r, Slot s) const
        {
            mgl::glUseProgram(prog_attribs_);
            set_viewport(r);
            set_xform(kIdentity);
            mgl::glBindVertexArray(vao_[s]);
            switch (s)
            {
            case kA: mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3); break;
            case kB: mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4); break;
            case kC: mgl::glDrawElements(mgl::PrimitiveType::Triangles, 6, mgl::IndexType::UnsignedShort, demo::buffer_offset(0)); break;
            case kD: mgl::glDrawElements(mgl::PrimitiveType::Triangles, 6, mgl::IndexType::UnsignedByte, demo::buffer_offset(kIndexDOffset)); break;
            case kE:
                // The array is disabled: the shader reads the current generic attribute value (not VAO state).
                mgl::glVertexAttrib4f(kLocAttr, kConstColor[0], kConstColor[1], kConstColor[2], kConstColor[3]);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                break;
            default: break;
            }
        }

        void draw_integer_cell(Rect r, Slot s, const IntCase& expect) const
        {
            mgl::glUseProgram(prog_int_);
            set_viewport(r);
            mgl::glUniform4i(u_expect_i_, expect.i[0], expect.i[1], expect.i[2], expect.i[3]);
            mgl::glUniform4ui(u_expect_u_, expect.u[0], expect.u[1], expect.u[2], expect.u[3]);
            mgl::glBindVertexArray(vao_[s]);
            if (s == kI3)  // arrays 1 and 2 are disabled in this VAO: set the current attribute values
            {
                mgl::glVertexAttribI4i(kLocAttr, kInt3.i[0], kInt3.i[1], kInt3.i[2], kInt3.i[3]);
                mgl::glVertexAttribI4ui(kLocUint, kInt3.u[0], kInt3.u[1], kInt3.u[2], kInt3.u[3]);
            }
            else if (s == kI4)
            {
                mgl::glVertexAttribI4iv(kLocAttr, kInt4.i.data());
                mgl::glVertexAttribI4uiv(kLocUint, kInt4.u.data());
            }
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        void draw_packed_cell(Rect r, Slot left, const Xform& left_x, Slot right, const Xform& right_x) const
        {
            mgl::glUseProgram(prog_attribs_);
            set_viewport(r);
            set_xform(left_x);
            mgl::glBindVertexArray(vao_[left]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 12);  // 2 quads x 2 triangles x 3 vertices
            set_xform(right_x);
            mgl::glBindVertexArray(vao_[right]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 12);
        }

        void draw_half_cell(Rect r) const
        {
            mgl::glUseProgram(prog_attribs_);
            set_viewport(r);
            set_xform(kHalfXform);
            mgl::glBindVertexArray(vao_[kHalf]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 24);
        }

        void draw_cells(const demo::Env& env) const
        {
            for (int c = 0; c < 4; ++c) draw_float_cell(cell(env, 3, c), kPass1[static_cast<std::size_t>(c)]);
            for (int c = 0; c < 4; ++c) draw_float_cell(cell(env, 2, c), kPass2[static_cast<std::size_t>(c)]);
            draw_integer_cell(cell(env, 1, 0), kI1, kInt1);
            draw_integer_cell(cell(env, 1, 1), kI2, kInt2);
            draw_integer_cell(cell(env, 1, 2), kI3, kInt3);
            draw_integer_cell(cell(env, 1, 3), kI4, kInt4);
            draw_packed_cell(cell(env, 0, 0), kPuNorm, kUintNorm, kPuRaw, kUintRaw);
            draw_packed_cell(cell(env, 0, 1), kPiNorm, kIntNorm, kPiRaw, kIntRaw);
            draw_half_cell(cell(env, 0, 2));
            draw_float_cell(cell(env, 0, 3), kE);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        // ---------------------------------------------------------------------------------------------
        // CPU models of the float cells
        // ---------------------------------------------------------------------------------------------
        static std::vector<Tri> model(Slot s)
        {
            auto vertex_a = [](std::size_t i, const Rgba* override_color) {
                return Vtx{double(kPosA[2 * i]), double(kPosA[2 * i + 1]),
                           override_color ? *override_color : Rgba{double(kColA[3 * i]), double(kColA[3 * i + 1]), double(kColA[3 * i + 2]), 1.0}};
            };
            auto vertex_b = [](std::size_t i) {
                const VertexB& v = kVertsB[i];
                return Vtx{std::max(v.x / 32767.0, -1.0), std::max(v.y / 32767.0, -1.0),  // eq. 2.2
                           {v.r / 255.0, v.g / 255.0, v.b / 255.0, v.a / 255.0}};          // eq. 2.1
            };
            auto vertex_cd = [](std::size_t i) {
                const VertexCD& v = kVertsCD[i];
                return Vtx{double(v.x), double(v.y), {v.r / 255.0, v.g / 255.0, v.b / 255.0, v.a / 255.0}};
            };
            switch (s)
            {
            case kA: return {{vertex_a(0, nullptr), vertex_a(1, nullptr), vertex_a(2, nullptr)}};
            case kB: return {{vertex_b(0), vertex_b(1), vertex_b(2)}, {vertex_b(1), vertex_b(2), vertex_b(3)}};
            case kC: return {{vertex_cd(0), vertex_cd(1), vertex_cd(4)}, {vertex_cd(2), vertex_cd(3), vertex_cd(4)}};
            case kD: return {{vertex_cd(1), vertex_cd(2), vertex_cd(4)}, {vertex_cd(3), vertex_cd(0), vertex_cd(4)}};
            case kE:
            {
                const Rgba orange = {double(kConstColor[0]), double(kConstColor[1]), double(kConstColor[2]), double(kConstColor[3])};
                return {{vertex_a(0, &orange), vertex_a(1, &orange), vertex_a(2, &orange)}};
            }
            default: return {};
            }
        }

        // ---------------------------------------------------------------------------------------------
        // Validation: pixels
        // ---------------------------------------------------------------------------------------------
        void check_pixels(const demo::Env& env, demo::Check& check) const
        {
            const int cw = env.width / kGrid, ch = env.height / kGrid;

            // Float VAO cells: rows 3 and 2 (two passes in different order) and VAO E.
            for (int pass = 0; pass < 2; ++pass)
                for (int c = 0; c < 4; ++c)
                {
                    const Slot s = (pass == 0 ? kPass1 : kPass2)[static_cast<std::size_t>(c)];
                    const Rect r = cell(env, 3 - pass, c);
                    expect_scan(check, scan_cell(model(s), r.x, r.y, r.w, r.h, 3), 800, 800,
                                std::format("row {} col {}: VAO {}", 3 - pass, c, kSlotName[s]));
                }
            {
                const Rect r = cell(env, 0, 3);
                expect_scan(check, scan_cell(model(kE), r.x, r.y, r.w, r.h, 3), 800, 800,
                            std::format("row 0 col 3: VAO {} - orange from glVertexAttrib4f", kSlotName[kE]));
            }

            // Integer cells: 8 sub-areas each (4 components x {ivec4, uvec4}); green = exact match in the shader.
            const std::array<const char*, 4> int_what = {
                "glVertexAttribIPointer INT + UNSIGNED_INT (values above 2^24)",
                "glVertexAttribIPointer SHORT size 3 (sign-extended, w = 1) + UNSIGNED_BYTE size 2 (zero-extended, z = 0, w = 1)",
                "glVertexAttribI4i + glVertexAttribI4ui (arrays disabled)", "glVertexAttribI4iv + glVertexAttribI4uiv (arrays disabled)"};
            for (int c = 0; c < 4; ++c)
            {
                const Rect r = cell(env, 1, c);
                std::string red;
                for (int comp = 0; comp < 4; ++comp)
                    for (int upper = 0; upper < 2; ++upper)
                    {
                        const Rgba8 got = demo::read_pixel(r.x + (2 * comp + 1) * cw / 8, r.y + (upper ? 3 * ch / 4 : ch / 4));
                        if (!close(got, int_ok_color(comp, upper != 0), 1)) red += std::format("{}.{} ", upper ? "ivec4" : "uvec4", "xyzw"[comp]);
                    }
                check.expect(red.empty(), std::format("row 1 col {}: {}: all 8 components arrive exactly", c, int_what[static_cast<std::size_t>(c)]),
                             red.empty() ? std::string{} : "wrong: " + red);
            }

            // Packed formats: probe the center of every quad, then scan the whole cell.
            auto quad_center = [&](Rect r, bool right, bool upper) {
                return std::array<int, 2>{r.x + (right ? 3 : 1) * cw / 4, r.y + (upper ? 3 : 1) * ch / 4};
            };
            struct PackedCell
            {
                int col;
                Slot left, right;
                bool is_signed;
                const std::array<std::uint32_t, 2>* words;
                Xform left_x, right_x;
                const char* name;
            };
            const std::array<PackedCell, 2> packed = {{
                {0, kPuNorm, kPuRaw, false, &kWordsU, kUintNorm, kUintRaw, "UNSIGNED_INT_2_10_10_10_REV"},
                {1, kPiNorm, kPiRaw, true, &kWordsS, kIntNorm, kIntRaw, "INT_2_10_10_10_REV"},
            }};
            for (const PackedCell& p : packed)
            {
                const Rect r = cell(env, 0, p.col);
                for (int half = 0; half < 2; ++half)
                    for (int w = 0; w < 2; ++w)
                    {
                        const bool normalized = half == 0;
                        const Rgba color = shade(decode_packed((*p.words)[static_cast<std::size_t>(w)], p.is_signed, normalized),
                                                 half == 0 ? p.left_x : p.right_x);
                        const auto xy = quad_center(r, half == 1, w == 0);
                        check.expect_pixel(xy[0], xy[1], {to8(color[0]), to8(color[1]), to8(color[2]), to8(color[3])}, 1,
                                           std::format("{} {} word {}: decoded color of the quad", p.name, normalized ? "normalized" : "raw", w));
                    }
                std::vector<Tri> scene = tris_packed(packed_quads(false, *p.words), p.is_signed, true, p.left_x);
                const std::vector<Tri> right = tris_packed(packed_quads(true, *p.words), p.is_signed, false, p.right_x);
                scene.insert(scene.end(), right.begin(), right.end());
                expect_scan(check, scan_cell(scene, r.x, r.y, r.w, r.h, 1), 2000, 800, std::format("row 0 col {}: {}", p.col, p.name));
            }
            {
                const Rect r = cell(env, 0, 2);
                const std::vector<HalfVertex> hv = half_quads();
                for (std::size_t q = 0; q < 4; ++q)
                {
                    const auto& c = kHalfColors[q];
                    Rgba f{};
                    for (std::size_t k = 0; k < 4; ++k) f[k] = half_to_double(double_to_half(c[k])) * kHalfScale;
                    const auto xy = quad_center(r, q % 2 == 1, q / 2 == 0);
                    check.expect_pixel(xy[0], xy[1], {to8(f[0]), to8(f[1]), to8(f[2]), to8(f[3])}, 1,
                                       std::format("HALF_FLOAT quad {}: color ({}, {}, {}, {}) / 100", q, c[0], c[1], c[2], c[3]));
                }
                expect_scan(check, scan_cell(tris_half(hv), r.x, r.y, r.w, r.h, 1), 2000, 800, "row 0 col 2: HALF_FLOAT positions and colors");
            }

            // The binary16 helpers themselves: known bit patterns of IEEE 754 binary16.
            check.expect(double_to_half(1.0) == 0x3C00 && double_to_half(0.5) == 0x3800 && double_to_half(-2.0) == 0xC000 &&
                             double_to_half(65504.0) == 0x7BFF && double_to_half(25.0) == 0x4E40 && half_to_double(0x3555) == 0.333251953125 &&
                             half_to_double(0xB800) == -0.5,
                         "CPU binary16 helpers agree with the IEEE 754 bit patterns (1.0=0x3C00, 0.5=0x3800, -2=0xC000, 25=0x4E40, 0x3555=0.33325...)");
            // And the packed-word helpers against hand-computed fields.
            check.expect(decode_packed(0xC0000000u, false, false)[3] == 3.0 && decode_packed(0xC0000000u, true, false)[3] == -1.0 &&
                             decode_packed(0x80000000u, true, false)[3] == -2.0 && decode_packed(0x80000000u, true, true)[3] == -1.0 &&
                             decode_packed(0x00000200u, true, true)[0] == -1.0 && decode_packed(0x00000200u, true, false)[0] == -512.0 &&
                             decode_packed(0x000001FFu, true, true)[0] == 1.0,
                         "CPU packed decoder: 2-bit w 0b11 = 3 unsigned / -1 signed, 0b10 = -2 -> -1 normalized, 10-bit 0x200 = -512 -> -1.0 normalized (clamped), 0x1FF = 511 -> 1.0");
        }

        // Each control draws something wrong and requires the model comparison to notice.
        void check_negative_controls(demo::Env& env, demo::Check& check)
        {
            // 1. Integer path: tell the shader to expect the float-rounded value 16777216 instead of 16777217.
            {
                const Rect r = cell(env, 1, 0);
                IntCase wrong = kInt1;
                wrong.i[0] = 16777216;
                draw_integer_cell(r, kI1, wrong);
                mgl::glBindVertexArray(mgl::VertexArrayId{0});
                const int cw = env.width / kGrid, ch = env.height / kGrid;
                check.expect_pixel(r.x + cw / 8, r.y + 3 * ch / 4, {255, 0, 0, 255}, 0,
                                   "negative control: the shader received 16777217, not the float-rounded 16777216 (sub-area turns red)");
                check.expect_pixel(r.x + 3 * cw / 8, r.y + 3 * ch / 4, int_ok_color(1, true), 1,
                                   "negative control: the neighbouring component stays green");
            }
            // 2. VAO switching: bind B in A's cell without clearing. Model A must now disagree, model "A then B" must agree.
            {
                const Rect r = cell(env, 3, 0);
                draw_float_cell(r, kB);
                mgl::glBindVertexArray(mgl::VertexArrayId{0});
                const Scan wrong = scan_cell(model(kA), r.x, r.y, r.w, r.h, 3);
                std::vector<Tri> both = model(kA);
                const std::vector<Tri> b = model(kB);
                both.insert(both.end(), b.begin(), b.end());
                const Scan right = scan_cell(both, r.x, r.y, r.w, r.h, 3);
                check.expect(wrong.bad > 100, "negative control: drawing VAO B into A's cell makes the VAO A comparison fail",
                             std::format("only {} mismatching pixels", wrong.bad));
                check.expect(right.bad == 0 && right.inside > 800, "negative control: the model 'A then B' matches the same pixels exactly",
                             std::format("{} mismatches; {}", right.bad, right.first_bad));
            }
            // 3. Packed formats: the pre-ES-3.0 signed-normalized formula (2c+1)/(2^b-1) would change the 2-bit w.
            {
                const Rect r = cell(env, 0, 1);
                render(env, 0);  // clean frame
                std::vector<Tri> legacy = tris_packed(packed_quads(false, kWordsS), true, true, kIntNorm, true);
                const Scan s = scan_cell(legacy, r.x, r.y, r.w, r.h, 1);
                check.expect(s.bad > 100,
                             "negative control: the legacy signed-normalized formula (2c+1)/(2^b-1) does not match the INT_2_10_10_10_REV pixels",
                             std::format("only {} mismatching pixels", s.bad));
            }
        }

        // ---------------------------------------------------------------------------------------------
        // Validation: state
        // ---------------------------------------------------------------------------------------------
        static int attrib_i(mgl::AttribLocation loc, mgl::VertexAttribParameter p)
        {
            GLint v = 0;
            mgl::glGetVertexAttribiv(loc, p, &v);
            return v;
        }
        static AttribState read_attrib(mgl::AttribLocation loc)
        {
            using P = mgl::VertexAttribParameter;
            AttribState s;
            s.enabled = attrib_i(loc, P::ArrayEnabled);
            s.size = attrib_i(loc, P::ArraySize);
            s.type = attrib_i(loc, P::ArrayType);
            s.normalized = attrib_i(loc, P::ArrayNormalized);
            s.stride = attrib_i(loc, P::ArrayStride);
            s.integer = attrib_i(loc, P::ArrayInteger);
            s.divisor = attrib_i(loc, P::ArrayDivisor);
            s.buffer = static_cast<unsigned>(attrib_i(loc, P::ArrayBufferBinding));
            void* ptr = nullptr;
            mgl::glGetVertexAttribPointerv(loc, P::ArrayPointer, &ptr);
            s.pointer = reinterpret_cast<std::uintptr_t>(ptr);
            return s;
        }
        static std::string diff(const AttribState& got, const AttribState& want)
        {
            std::string d;
            auto cmp = [&](const char* name, auto g, auto w) {
                if (g != w) d += std::format("{}: got {:#x} want {:#x}; ", name, g, w);
            };
            cmp("enabled", got.enabled, want.enabled);
            cmp("size", got.size, want.size);
            cmp("type", got.type, want.type);
            cmp("normalized", got.normalized, want.normalized);
            cmp("stride", got.stride, want.stride);
            cmp("integer", got.integer, want.integer);
            cmp("divisor", got.divisor, want.divisor);
            cmp("buffer", got.buffer, want.buffer);
            cmp("pointer", got.pointer, want.pointer);
            return d;
        }
        void expect_attrib(demo::Check& check, mgl::VertexArrayId vao, mgl::AttribLocation loc, const AttribState& want, const std::string& what) const
        {
            mgl::glBindVertexArray(vao);
            const std::string d = diff(read_attrib(loc), want);
            check.expect(d.empty(), what, d);
        }
        static unsigned id(mgl::BufferId b) { return b.value; }
        static int ival(GLuint v) { return static_cast<int>(v); }

        void check_vao_state(demo::Check& check) const
        {
            // --- the state recorded in each VAO (ES 3.0 Table 6.2) -----------------------------------
            expect_attrib(check, vao_[kA], kLocPos, {.enabled = 1, .size = 2, .type = GL_FLOAT, .buffer = id(vbo_pos_a_)}, "VAO A attribute 0: enabled, vec2 FLOAT, tight, buffer = position buffer");
            expect_attrib(check, vao_[kA], kLocAttr, {.enabled = 1, .size = 3, .type = GL_FLOAT, .buffer = id(vbo_col_a_)}, "VAO A attribute 1: vec3 FLOAT from a *different* buffer (each attribute remembers its own ARRAY_BUFFER)");
            expect_attrib(check, vao_[kB], kLocPos, {.enabled = 1, .size = 2, .type = GL_SHORT, .normalized = 1, .stride = 8, .buffer = id(vbo_b_), .pointer = 4}, "VAO B attribute 0: SHORT x2 normalized, stride 8, offset 4");
            expect_attrib(check, vao_[kB], kLocAttr, {.enabled = 1, .size = 4, .type = GL_UNSIGNED_BYTE, .normalized = 1, .stride = 8, .buffer = id(vbo_b_), .pointer = 0}, "VAO B attribute 1: UNSIGNED_BYTE x4 normalized, stride 8, offset 0");
            expect_attrib(check, vao_[kC], kLocAttr, {.enabled = 1, .size = 4, .type = GL_UNSIGNED_BYTE, .normalized = 1, .stride = 12, .buffer = id(vbo_cd_), .pointer = 8}, "VAO C attribute 1: UNSIGNED_BYTE x4 normalized, stride 12, offset 8");
            expect_attrib(check, vao_[kD], kLocAttr, {.enabled = 1, .size = 4, .type = GL_UNSIGNED_BYTE, .normalized = 1, .stride = 12, .buffer = id(vbo_cd_), .pointer = 8}, "VAO D attribute 1: identical to VAO C (only the element buffer differs)");
            expect_attrib(check, vao_[kE], kLocAttr, {}, "VAO E attribute 1: disabled, all other fields at their initial values");
            expect_attrib(check, vao_[kE], kLocPos, {.enabled = 1, .size = 2, .type = GL_FLOAT, .buffer = id(vbo_pos_a_)}, "VAO E attribute 0: enabled");
            expect_attrib(check, vao_[kI1], kLocAttr, {.enabled = 1, .size = 4, .type = GL_INT, .normalized = 0, .stride = 40, .integer = 1, .buffer = id(vbo_i1_), .pointer = offsetof(VertexI1, i)}, "VAO I1 attribute 1: glVertexAttribIPointer sets INTEGER = TRUE (type INT, not normalized)");
            expect_attrib(check, vao_[kI1], kLocUint, {.enabled = 1, .size = 4, .type = GL_UNSIGNED_INT, .stride = 40, .integer = 1, .buffer = id(vbo_i1_), .pointer = offsetof(VertexI1, u)}, "VAO I1 attribute 2: UNSIGNED_INT, INTEGER = TRUE");
            expect_attrib(check, vao_[kI2], kLocAttr, {.enabled = 1, .size = 3, .type = GL_SHORT, .stride = 16, .integer = 1, .buffer = id(vbo_i2_), .pointer = offsetof(VertexI2, s)}, "VAO I2 attribute 1: SHORT size 3, INTEGER = TRUE");
            expect_attrib(check, vao_[kI2], kLocUint, {.enabled = 1, .size = 2, .type = GL_UNSIGNED_BYTE, .stride = 16, .integer = 1, .buffer = id(vbo_i2_), .pointer = offsetof(VertexI2, b)}, "VAO I2 attribute 2: UNSIGNED_BYTE size 2, INTEGER = TRUE");
            expect_attrib(check, vao_[kPuNorm], kLocAttr, attrib_of_packed(kPuNorm), "VAO UINT norm attribute 1: UNSIGNED_INT_2_10_10_10_REV, normalized");
            expect_attrib(check, vao_[kPuRaw], kLocAttr, attrib_of_packed(kPuRaw), "VAO UINT raw attribute 1: UNSIGNED_INT_2_10_10_10_REV, not normalized");
            expect_attrib(check, vao_[kPiNorm], kLocAttr, attrib_of_packed(kPiNorm), "VAO INT norm attribute 1: INT_2_10_10_10_REV, normalized");
            expect_attrib(check, vao_[kHalf], kLocAttr, {.enabled = 1, .size = 4, .type = GL_HALF_FLOAT, .normalized = 0, .stride = 12, .buffer = id(vbo_half_), .pointer = 4}, "VAO HALF_FLOAT attribute 1: HALF_FLOAT x4, stride 12, offset 4");

            // --- switching changes everything, switching back restores it ----------------------------
            mgl::glBindVertexArray(vao_[kA]);
            const int a_size = attrib_i(kLocAttr, mgl::VertexAttribParameter::ArraySize);
            mgl::glBindVertexArray(vao_[kB]);
            const int b_size = attrib_i(kLocAttr, mgl::VertexAttribParameter::ArraySize);
            const int b_type = attrib_i(kLocAttr, mgl::VertexAttribParameter::ArrayType);
            mgl::glBindVertexArray(vao_[kA]);
            const int a_again = attrib_i(kLocAttr, mgl::VertexAttribParameter::ArraySize);
            const int a_type = attrib_i(kLocAttr, mgl::VertexAttribParameter::ArrayType);
            check.expect(a_size == 3 && b_size == 4 && a_again == 3 && a_type == GL_FLOAT && b_type == GL_UNSIGNED_BYTE,
                         "glBindVertexArray(A) -> (B) -> (A): attribute 1 reads size 3 / FLOAT, 4 / UNSIGNED_BYTE, 3 / FLOAT",
                         std::format("sizes {} {} {}, types {:#x} {:#x}", a_size, b_size, a_again, a_type, b_type));

            // --- ELEMENT_ARRAY_BUFFER is VAO state ---------------------------------------------------
            auto element_binding = [](mgl::VertexArrayId v) {
                mgl::glBindVertexArray(v);
                return demo::get_integer(mgl::GetParameter::ElementArrayBufferBinding);
            };
            check.expect_eq(element_binding(vao_[kC]), ival(ebo_c_.value), "ELEMENT_ARRAY_BUFFER_BINDING with VAO C bound = C's element buffer");
            check.expect_eq(element_binding(vao_[kD]), ival(ebo_d_.value), "ELEMENT_ARRAY_BUFFER_BINDING with VAO D bound = D's element buffer");
            check.expect_eq(element_binding(vao_[kA]), 0, "ELEMENT_ARRAY_BUFFER_BINDING with VAO A bound = 0 (A has none)");
            check.expect_eq(element_binding(mgl::VertexArrayId{0}), 0, "ELEMENT_ARRAY_BUFFER_BINDING with the default VAO bound = 0");

            // glBindBuffer(ELEMENT_ARRAY_BUFFER) edits only the bound VAO.
            mgl::VertexArrayId scratch{};
            mgl::glGenVertexArrays(1, &scratch);
            mgl::glBindVertexArray(scratch);
            mgl::glBindBuffer(mgl::BufferTarget::ElementArray, ebo_c_);
            const int in_scratch = demo::get_integer(mgl::GetParameter::ElementArrayBufferBinding);
            check.expect_eq(in_scratch, ival(ebo_c_.value), "glBindBuffer(ELEMENT_ARRAY_BUFFER) while a new VAO is bound: that VAO now reports the buffer");
            check.expect_eq(element_binding(vao_[kA]), 0, "... and VAO A, bound afterwards, is unaffected");
            check.expect_eq(element_binding(scratch), ival(ebo_c_.value), "... while the scratch VAO still has it after being rebound");

            // ARRAY_BUFFER is NOT VAO state: the one binding follows you through every VAO switch.
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_b_);
            bool follows = true;
            for (const mgl::VertexArrayId v : {vao_[kA], vao_[kC], mgl::VertexArrayId{0}, scratch})
            {
                mgl::glBindVertexArray(v);
                follows = follows && demo::get_integer(mgl::GetParameter::ArrayBufferBinding) == ival(vbo_b_.value);
            }
            check.expect(follows, "ARRAY_BUFFER_BINDING stays the same across VAO A, C, default and a new VAO (not VAO state)");
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            expect_attrib(check, vao_[kA], kLocPos, {.enabled = 1, .size = 2, .type = GL_FLOAT, .buffer = id(vbo_pos_a_)},
                          "with ARRAY_BUFFER = 0, VAO A still remembers the buffer each attribute was specified with (and the draws worked)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ArrayBufferBinding), 0, "ARRAY_BUFFER_BINDING is 0 while the cells were drawn");

            // A new VAO starts from the initial state of Table 6.2, not from the previously bound VAO.
            mgl::glBindVertexArray(vao_[kC]);  // configured, with an element buffer
            mgl::VertexArrayId fresh{};
            mgl::glGenVertexArrays(1, &fresh);
            mgl::glBindVertexArray(fresh);
            const AttribState init{};
            check.expect(diff(read_attrib(kLocPos), init).empty() && diff(read_attrib(kLocAttr), init).empty() &&
                             diff(read_attrib(mgl::AttribLocation{15u}), init).empty(),
                         "a freshly bound VAO: attributes 0, 1, 15 disabled, size 4, FLOAT, stride 0, not integer, divisor 0, buffer 0, pointer NULL",
                         diff(read_attrib(kLocAttr), init));
            check.expect_eq(demo::get_integer(mgl::GetParameter::ElementArrayBufferBinding), 0, "a freshly bound VAO has ELEMENT_ARRAY_BUFFER_BINDING 0");

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDeleteVertexArrays(1, &scratch);
            mgl::glDeleteVertexArrays(1, &fresh);
            check.expect_no_gl_error("no GL error from the state queries");
        }

        // expected state of attribute 1 in the packed-format VAOs
        AttribState attrib_of_packed(Slot s) const
        {
            const bool is_int = s == kPiNorm || s == kPiRaw;
            const bool normalized = s == kPuNorm || s == kPiNorm;
            return {.enabled = 1, .size = 4, .type = is_int ? GL_INT_2_10_10_10_REV : GL_UNSIGNED_INT_2_10_10_10_REV,
                    .normalized = normalized ? 1 : 0, .stride = 12, .buffer = id(packed_vbo_[s]), .pointer = 8};
        }

        void check_lifecycle(demo::Check& check) const
        {
            check.expect(!mgl::glIsVertexArray(mgl::VertexArrayId{0}), "glIsVertexArray(0) = FALSE (the default VAO is not a VAO object)");
            mgl::VertexArrayId v{};
            mgl::glGenVertexArrays(1, &v);
            check.expect(v.value != 0, "glGenVertexArrays returns a non-zero name");
            check.expect(!mgl::glIsVertexArray(v), "a name returned by glGenVertexArrays is not yet a VAO (it acquires state when first bound)");
            mgl::glBindVertexArray(v);
            check.expect(mgl::glIsVertexArray(v), "glBindVertexArray creates the VAO: glIsVertexArray = TRUE");
            check.expect_eq(demo::get_integer(mgl::GetParameter::VertexArrayBinding), ival(v.value), "VERTEX_ARRAY_BINDING = the bound VAO");

            // Deleting the bound VAO: the binding reverts to 0 and the default VAO becomes current.
            mgl::glDeleteVertexArrays(1, &v);
            check.expect_eq(demo::get_integer(mgl::GetParameter::VertexArrayBinding), 0, "deleting the bound VAO reverts VERTEX_ARRAY_BINDING to 0");
            check.expect(!mgl::glIsVertexArray(v), "a deleted VAO name is no longer a VAO");
            check.expect_no_gl_error("deleting the bound VAO raises no error");

            // BindVertexArray with a deleted or never generated name fails with INVALID_OPERATION and changes nothing.
            mgl::glBindVertexArray(vao_[kC]);
            mgl::glBindVertexArray(v);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBindVertexArray(deleted name) -> INVALID_OPERATION");
            check.expect_eq(demo::get_integer(mgl::GetParameter::VertexArrayBinding), ival(vao_[kC].value), "... and the previous binding (VAO C) is unchanged");
            mgl::glBindVertexArray(mgl::VertexArrayId{0x00FFFFFFu});
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glBindVertexArray(never generated name) -> INVALID_OPERATION");

            // Unused names and zero are silently ignored by glDeleteVertexArrays.
            mgl::glDeleteVertexArrays(1, &v);
            const mgl::VertexArrayId zero{0};
            mgl::glDeleteVertexArrays(1, &zero);
            check.expect_no_gl_error("glDeleteVertexArrays ignores an already deleted name and zero");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void check_current_attributes(const demo::Env& env, demo::Check& check) const
        {
            using P = mgl::VertexAttribParameter;
            // Integer current values: set with glVertexAttribI4i/ui(v), read with glGetVertexAttribIi/Iuiv.
            mgl::glBindVertexArray(vao_[kI3]);
            mgl::glVertexAttribI4i(kLocAttr, kInt3.i[0], kInt3.i[1], kInt3.i[2], kInt3.i[3]);
            mgl::glVertexAttribI4ui(kLocUint, kInt3.u[0], kInt3.u[1], kInt3.u[2], kInt3.u[3]);
            std::array<GLint, 4> ci{};
            std::array<GLuint, 4> cu{};
            mgl::glGetVertexAttribIiv(kLocAttr, P::CurrentVertexAttrib, ci.data());
            mgl::glGetVertexAttribIuiv(kLocUint, P::CurrentVertexAttrib, cu.data());
            check.expect(ci == kInt3.i, "glVertexAttribI4i then glGetVertexAttribIiv(CURRENT_VERTEX_ATTRIB) returns all four ints exactly (incl. 2^24+1, INT_MIN)");
            check.expect(cu == kInt3.u, "glVertexAttribI4ui then glGetVertexAttribIuiv(CURRENT_VERTEX_ATTRIB) returns all four uints exactly (incl. 0xFFFFFFFF, 2^24+1)");

            // CURRENT_VERTEX_ATTRIB is context state: other VAOs, a draw with the array enabled, see the same value.
            mgl::glBindVertexArray(vao_[kA]);
            mgl::glGetVertexAttribIiv(kLocAttr, P::CurrentVertexAttrib, ci.data());
            check.expect(ci == kInt3.i, "switching to another VAO does not change CURRENT_VERTEX_ATTRIB (it is not VAO state)");
            mgl::glVertexAttribI4iv(kLocAttr, kInt4.i.data());
            mgl::glVertexAttribI4uiv(kLocUint, kInt4.u.data());
            mgl::glGetVertexAttribIiv(kLocAttr, P::CurrentVertexAttrib, ci.data());
            mgl::glGetVertexAttribIuiv(kLocUint, P::CurrentVertexAttrib, cu.data());
            check.expect(ci == kInt4.i && cu == kInt4.u, "glVertexAttribI4iv / glVertexAttribI4uiv values read back exactly");

            // Float current value; VAO A's attribute 1 is an enabled array, so drawing with it must not touch the current value.
            const std::array<GLfloat, 4> f = {0.25f, -2.5f, 8.0f, 0.125f};
            mgl::glVertexAttrib4fv(kLocAttr, f.data());
            std::array<GLfloat, 4> cf{};
            mgl::glGetVertexAttribfv(kLocAttr, P::CurrentVertexAttrib, cf.data());
            check.expect(cf == f, "glVertexAttrib4fv / glGetVertexAttribfv(CURRENT_VERTEX_ATTRIB) round-trip exactly");
            draw_float_cell(cell(env, 3, 0), kA);  // redraws VAO A's own picture into A's own cell (identical pixels)
            mgl::glGetVertexAttribfv(kLocAttr, P::CurrentVertexAttrib, cf.data());
            check.expect(cf == f, "a draw whose attribute-1 array is enabled leaves CURRENT_VERTEX_ATTRIB unchanged (ES 3.0 section 2.9.3)");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            check.expect_no_gl_error("no GL error from the attribute state queries");
        }

        void check_errors(demo::Check& check) const
        {
            const int max_attribs = demo::get_integer(mgl::GetParameter::MaxVertexAttribs);
            check.expect(max_attribs >= 16, "MAX_VERTEX_ATTRIBS >= 16 (ES 3.0 Table 6.31)", std::format("{}", max_attribs));
            check.expect_no_gl_error("error queue empty before the error cases");

            mgl::VertexArrayId vao{};
            mgl::glGenVertexArrays(1, &vao);
            mgl::glBindVertexArray(vao);  // a NON-default VAO
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // ES 3.0 section 2.9: non-zero VAO + ARRAY_BUFFER 0 + non-NULL pointer -> INVALID_OPERATION.
            const std::array<float, 2> client = {0.0f, 0.0f};
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(16));
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glVertexAttribPointer(offset 16) with no ARRAY_BUFFER on a non-default VAO -> INVALID_OPERATION");
            mgl::glVertexAttribIPointer(kLocAttr, 4, mgl::DataType::Int, 0, demo::buffer_offset(16));
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glVertexAttribIPointer(offset 16) with no ARRAY_BUFFER on a non-default VAO -> INVALID_OPERATION");
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, client.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "a real client-memory pointer on a non-default VAO -> INVALID_OPERATION (VAOs cannot hold client arrays)");
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            check.expect_no_gl_error("the same call with a NULL pointer (offset 0) is allowed even without a buffer");

            // The default VAO accepts client-memory arrays (ES 3.0 section 2.9 footnote 5 / section 2.11).
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glVertexAttribPointer(kLocPos, 2, mgl::DataType::Float, GL_FALSE, 0, client.data());
            check.expect_no_gl_error("a client-memory pointer on the default VAO (VAO 0) is allowed");
            void* got = nullptr;
            mgl::glGetVertexAttribPointerv(kLocPos, mgl::VertexAttribParameter::ArrayPointer, &got);
            check.expect(got == static_cast<const void*>(client.data()), "glGetVertexAttribPointerv returns that client pointer (default VAO state is client state)");
            mgl::glVertexAttribPointer(kLocPos, 4, mgl::DataType::Float, GL_FALSE, 0, nullptr);  // back to the initial value
            mgl::glBindVertexArray(vao);

            // Packed types need size 4: INVALID_OPERATION (not INVALID_VALUE: sizes 1..4 are valid for the command).
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_quad_);
            mgl::glVertexAttribPointer(kLocAttr, 3, mgl::DataType::UnsignedInt2101010Rev, GL_TRUE, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "UNSIGNED_INT_2_10_10_10_REV with size 3 -> INVALID_OPERATION");
            mgl::glVertexAttribPointer(kLocAttr, 2, mgl::DataType::Int2101010Rev, GL_FALSE, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "INT_2_10_10_10_REV with size 2 -> INVALID_OPERATION");
            mgl::glVertexAttribPointer(kLocAttr, 4, mgl::DataType::Int2101010Rev, GL_FALSE, 0, demo::buffer_offset(0));
            check.expect_no_gl_error("INT_2_10_10_10_REV with size 4 is accepted");
            // Table 2.4: glVertexAttribIPointer accepts only BYTE, UBYTE, SHORT, USHORT, INT, UINT.
            mgl::glVertexAttribIPointer(kLocAttr, 4, mgl::DataType::UnsignedInt2101010Rev, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glVertexAttribIPointer with a packed type -> INVALID_ENUM (Table 2.4)");
            mgl::glVertexAttribIPointer(kLocAttr, 4, mgl::DataType::Float, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glVertexAttribIPointer with FLOAT -> INVALID_ENUM (Table 2.4)");
            mgl::glVertexAttribIPointer(kLocAttr, 4, mgl::DataType::HalfFloat, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glVertexAttribIPointer with HALF_FLOAT -> INVALID_ENUM (Table 2.4)");

            // Other defined errors of the same commands.
            mgl::glVertexAttribPointer(kLocAttr, 5, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer with size 5 -> INVALID_VALUE");
            mgl::glVertexAttribPointer(kLocAttr, 0, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer with size 0 -> INVALID_VALUE");
            mgl::glVertexAttribPointer(kLocAttr, 4, mgl::DataType::Float, GL_FALSE, -4, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer with a negative stride -> INVALID_VALUE");
            const mgl::AttribLocation too_big{static_cast<GLuint>(max_attribs)};
            mgl::glVertexAttribPointer(too_big, 4, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribPointer with index = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            mgl::glVertexAttribIPointer(too_big, 4, mgl::DataType::Int, 0, demo::buffer_offset(0));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribIPointer with index = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            mgl::glEnableVertexAttribArray(too_big);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glEnableVertexAttribArray with index = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            mgl::glVertexAttribI4i(too_big, 0, 0, 0, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glVertexAttribI4i with index = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            const mgl::AttribLocation last{static_cast<GLuint>(max_attribs - 1)};
            mgl::glVertexAttribPointer(last, 4, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            check.expect_no_gl_error("glVertexAttribPointer with index = MAX_VERTEX_ATTRIBS - 1 is valid");

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteVertexArrays(1, &vao);
        }

        mgl::ProgramId prog_attribs_{}, prog_int_{};
        mgl::UniformLocation u_scale_{}, u_bias_{}, u_expect_i_{}, u_expect_u_{};
        std::array<mgl::VertexArrayId, kSlotCount> vao_{};
        std::vector<mgl::BufferId> buffers_;
        std::array<mgl::BufferId, kSlotCount> packed_vbo_{};
        mgl::BufferId vbo_pos_a_{}, vbo_col_a_{}, vbo_b_{}, vbo_cd_{}, ebo_c_{}, ebo_d_{}, vbo_quad_{}, vbo_i1_{}, vbo_i2_{}, vbo_half_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-vertex-array-objects",
        .title = "Vertex array objects, integer and packed attributes",
        .min_es = {3, 0},
        .summary = "Several VAOs with different buffers, formats, strides and element buffers are drawn alternately; "
                   "integer (ivec4/uvec4), 2_10_10_10 packed and half-float attributes are decoded and compared with a CPU model.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<VertexArrayObjects>(argc, argv, info);
}
