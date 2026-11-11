// es31-separate-vertex-format
//
// Separate vertex attribute format and vertex buffer binding (OpenGL ES 3.1; section numbers follow ES 3.2, 10.3.1,
// 10.3.2, 10.3.7). ES 3.0 describes an attribute with ONE call, glVertexAttribPointer(index, size, type, normalized,
// stride, pointer), that fixes the format, the buffer and the offset together. ES 3.1 splits it:
//
//   glVertexAttribFormat / glVertexAttribIFormat(attrib, size, type, relativeoffset)   WHAT one element looks like
//   glVertexAttribBinding(attrib, bindingindex)                                         WHICH binding the attribute reads
//   glBindVertexBuffer(bindingindex, buffer, offset, stride)                            WHERE the data is and the stride
//   glVertexBindingDivisor(bindingindex, divisor)                                       HOW OFTEN it advances (instancing)
//
// An element is fetched from  offset(binding) + relativeoffset(attrib) + i * stride(binding)  (section 10.3.7). A binding can
// feed several attributes, an attribute can be moved to another binding, and a binding can be pointed at another buffer
// without touching any format. The legacy glVertexAttribPointer is defined as a sequence of these calls (section 10.3.1:
// "Format(index, ..., 0); VertexAttribBinding(index, index); BindVertexBuffer(index, ...)"), so both APIs act on the same state.
//
// The 256x256 surface is a 4x3 grid of 64x85 cells (row 2 is the top row):
//
//   row 2:  [0] one binding, three     [1] same formats, binding     [2] two bindings: positions     [3] same VAO, divisor of
//           attributes, binding offset  re-pointed to buffer B        (divisor 0) + instance data     binding 1 changed to 2
//           24, relative offsets        with another stride            (divisor 1), 4 instances        (instances come in pairs)
//   row 1:  [4] attribute 2 moved to   [5] glVertexAttribIFormat:    [6] legacy glVertexAttribPointer [7] same VAO, glBindVertexBuffer
//           another binding             exact integers (3 bands)      defines attributes 0 and 1       re-points attribute 0 only
//   row 0:  [8] legacy pointer after   [9] two bindings, ONE buffer  [10] bindings MAX-1 and MAX-2     [11] stride = MAX_VERTEX_ATTRIB_STRIDE,
//           a new-style binding of the  (offsets 0 and 8)                                              color at relative offset
//           same attribute                                                                             MAX_VERTEX_ATTRIB_RELATIVE_OFFSET - 3
//
// Every float cell is predicted by a CPU model that fetches the vertices from the very same byte arrays that were uploaded,
// using the address formula above and the divisor rule of section 10.5, and compares each pixel with Gouraud-shaded triangles.
// validate() also reads back all the state the new commands define, checks the limits, the defaults of a fresh VAO, the
// interplay with the legacy commands, and every error of the new commands (including the default vertex array object).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kCols = 4;
    constexpr int kRows = 3;
    constexpr std::array<std::uint8_t, 4> kClear8 = {20, 30, 50, 255};

    constexpr mgl::AttribLocation kPos{0u};     // layout(location = 0) vec2 a_pos
    constexpr mgl::AttribLocation kColor{1u};   // layout(location = 1) vec4 a_color
    constexpr mgl::AttribLocation kOffset{2u};  // layout(location = 2) vec2 a_offset
    constexpr mgl::AttribLocation kCell{3u};    // layout(location = 3) uvec2 a_cell   (integer.vert)
    constexpr mgl::AttribLocation kBig{4u};     // layout(location = 4) int a_big
    constexpr mgl::AttribLocation kNeg{5u};     // layout(location = 5) int a_neg

    // -----------------------------------------------------------------------------------------
    // Source data: raw bytes, exactly as uploaded. The CPU model reads these same arrays.
    // -----------------------------------------------------------------------------------------
    struct Bytes
    {
        std::vector<std::uint8_t> b;

        template <typename T>
        void put(T v)
        {
            const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
            b.insert(b.end(), p, p + sizeof(T));
        }
        void rgba(std::uint8_t r, std::uint8_t g, std::uint8_t bl, std::uint8_t a = 255)
        {
            put(r);
            put(g);
            put(bl);
            put(a);
        }
        void pad(std::size_t n, std::uint8_t value = 0)
        {
            b.insert(b.end(), n, value);
        }
        [[nodiscard]] std::size_t size() const { return b.size(); }
    };

    constexpr std::int16_t snorm16(double v)
    {
        return static_cast<std::int16_t>(v * 32767.0 + (v < 0 ? -0.5 : 0.5));
    }

    struct Corner
    {
        float x, y;
        std::array<std::uint8_t, 4> c;
    };

    // The triangles of the single-draw cells (positions, colors). Cell 0/1 add a per-vertex SHORT offset.
    constexpr std::array<Corner, 3> kTri0 = {{{-0.8f, -0.7f, {255, 0, 0, 255}}, {0.7f, -0.7f, {0, 255, 0, 255}}, {-0.1f, 0.7f, {0, 0, 255, 255}}}};
    constexpr std::array<std::array<double, 2>, 3> kOff0 = {{{0.1, -0.1}, {-0.1, -0.1}, {0.0, 0.1}}};
    constexpr std::array<Corner, 3> kTri1 = {{{-0.7f, 0.6f, {255, 255, 0, 255}}, {0.8f, 0.5f, {0, 255, 255, 255}}, {0.0f, -0.8f, {255, 0, 255, 255}}}};
    constexpr std::array<std::array<double, 2>, 3> kOff1 = {{{0.05, 0.0}, {0.0, 0.05}, {-0.05, -0.05}}};
    constexpr std::array<Corner, 3> kTriL1 = {{{-0.8f, -0.7f, {255, 0, 0, 255}}, {0.8f, -0.7f, {0, 255, 0, 255}}, {0.0f, 0.8f, {0, 0, 255, 255}}}};
    constexpr std::array<Corner, 3> kTriL2 = {{{-0.5f, -0.8f, {255, 255, 255, 255}}, {0.8f, 0.0f, {255, 255, 255, 255}}, {-0.8f, 0.7f, {255, 255, 255, 255}}}};
    constexpr std::array<Corner, 3> kTriS = {{{0.0f, 0.8f, {255, 128, 0, 255}}, {-0.8f, -0.6f, {0, 128, 255, 255}}, {0.8f, -0.5f, {128, 0, 255, 255}}}};
    constexpr std::array<Corner, 3> kTriHigh = {{{-0.8f, -0.5f, {255, 255, 255, 255}}, {0.8f, -0.7f, {255, 0, 0, 255}}, {0.0f, 0.8f, {0, 0, 255, 255}}}};
    constexpr std::array<Corner, 3> kTriLimit = {{{-0.8f, -0.8f, {255, 200, 0, 255}}, {0.8f, -0.6f, {0, 200, 255, 255}}, {0.0f, 0.8f, {255, 0, 200, 255}}}};

    // Instances (cells 2, 3, 4): a unit quad strip scaled by 0.18 x 0.15, one element per instance.
    constexpr std::array<std::array<float, 2>, 4> kQuad = {{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}}};
    struct Instance
    {
        float ox, oy;
        std::array<std::uint8_t, 4> c;
    };
    constexpr std::array<Instance, 4> kInstances = {{{-0.5f, -0.5f, {230, 60, 60, 255}}, {0.5f, -0.5f, {60, 200, 80, 255}},
                                                     {-0.5f, 0.5f, {70, 110, 240, 255}}, {0.5f, 0.5f, {240, 200, 50, 255}}}};
    constexpr std::array<std::array<float, 2>, 4> kDiagonal = {{{-0.6f, 0.6f}, {-0.2f, 0.2f}, {0.2f, -0.2f}, {0.6f, -0.6f}}};
    constexpr double kInstScaleX = 0.18, kInstScaleY = 0.15;

    constexpr std::array<float, 2> kShiftX = {-0.3f, -0.3f};  // cell 8: the offset a new-style binding would deliver
    constexpr std::array<float, 2> kShiftY = {0.15f, 0.1f};   // ... and the offset the legacy call delivers instead

    // Integer cell (3 bands x 6 vertices): every vertex carries the same integers.
    constexpr std::array<std::uint8_t, 2> kIntCell = {1, 2};
    constexpr std::int32_t kIntBig = 16777217;  // 2^24 + 1
    constexpr std::int16_t kIntNeg = -3;

    // Which bytes of which buffer; used for both the upload and the CPU model.
    enum Buf : std::size_t { kBufA, kBufB, kBufP, kBufQ, kBufR, kBufInt, kBufL1, kBufL2, kBufX, kBufY, kBufS, kBufTP, kBufTC, kBufLB, kBufCount };

    constexpr long kAJunk = 24;     // bytes in front of the vertices of buffer A: the binding offset skips them
    constexpr long kL1Junk = 16;    // bytes in front of buffer L1 (legacy pointer offsets)
    constexpr long kStrideA = 16;   // V16: float x, y; ubyte rgba; short ox, oy
    constexpr long kStrideB = 24;   // V24: same 16 bytes + 8 bytes of padding
    constexpr long kStride12 = 12;  // float x, y; ubyte rgba

    // -----------------------------------------------------------------------------------------
    // CPU model: what the GL fetches for attribute i, vertex v and instance n.
    // -----------------------------------------------------------------------------------------
    enum class Fmt { Float, UByteNorm, ShortNorm };

    struct Attr  // an attribute with its binding already resolved (what the vertex fetch unit sees)
    {
        bool enabled = false;
        int size = 0;
        Fmt fmt = Fmt::Float;
        long rel = 0;                    // relativeoffset
        const Bytes* data = nullptr;     // the buffer bound to the binding
        long offset = 0;                 // binding offset
        long stride = 0;                 // binding stride
        unsigned divisor = 0;            // binding divisor
    };

    std::array<double, 4> fetch(const Attr& a, long element)
    {
        std::array<double, 4> v = {0.0, 0.0, 0.0, 1.0};  // missing components: ES 3.2 section 10.3.3
        if (!a.enabled || a.data == nullptr) return v;
        const long comp = a.fmt == Fmt::Float ? 4 : (a.fmt == Fmt::ShortNorm ? 2 : 1);
        const long addr = a.offset + a.rel + element * a.stride;  // section 10.3.7
        if (addr < 0 || static_cast<std::size_t>(addr + comp * a.size) > a.data->size()) return {0.0, 0.0, 0.0, 0.0};
        const std::uint8_t* p = a.data->b.data() + addr;
        for (int k = 0; k < a.size; ++k)
        {
            const std::uint8_t* q = p + k * comp;
            switch (a.fmt)
            {
            case Fmt::Float: { float f; std::memcpy(&f, q, 4); v[static_cast<std::size_t>(k)] = static_cast<double>(f); break; }
            case Fmt::UByteNorm: v[static_cast<std::size_t>(k)] = *q / 255.0; break;
            case Fmt::ShortNorm: { std::int16_t s; std::memcpy(&s, q, 2); v[static_cast<std::size_t>(k)] = std::max(s / 32767.0, -1.0); break; }
            }
        }
        return v;
    }

    struct Scene
    {
        std::array<Attr, 3> attr;  // 0 position, 1 color, 2 offset
        mgl::PrimitiveType mode = mgl::PrimitiveType::Triangles;
        long first = 0, count = 3, instances = 1;
        double sx = 1.0, sy = 1.0;
    };

    using Rgba = std::array<double, 4>;
    struct Vtx
    {
        double x, y;
        Rgba c;
    };
    using Tri = std::array<Vtx, 3>;

    std::vector<Tri> model_scene(const Scene& s)
    {
        std::vector<Tri> out;
        for (long inst = 0; inst < s.instances; ++inst)
        {
            std::vector<Vtx> verts;
            for (long i = 0; i < s.count; ++i)
            {
                const long vertex = s.first + i;
                auto element = [&](const Attr& a) { return a.divisor == 0 ? vertex : inst / static_cast<long>(a.divisor); };
                const auto p = fetch(s.attr[0], element(s.attr[0]));
                const auto c = fetch(s.attr[1], element(s.attr[1]));
                const auto o = fetch(s.attr[2], element(s.attr[2]));
                verts.push_back({p[0] * s.sx + o[0], p[1] * s.sy + o[1], {c[0], c[1], c[2], c[3]}});
            }
            if (s.mode == mgl::PrimitiveType::Triangles)
                for (std::size_t i = 0; i + 2 < verts.size(); i += 3) out.push_back({verts[i], verts[i + 1], verts[i + 2]});
            else  // TRIANGLE_STRIP
                for (std::size_t i = 0; i + 2 < verts.size(); ++i) out.push_back({verts[i], verts[i + 1], verts[i + 2]});
        }
        return out;
    }

    enum class Cover { Outside, Inside, Unsure };
    struct Sample
    {
        Cover cover = Cover::Outside;
        Rgba color{};
    };

    // Classifies the pixel center (px, py) of a cw x ch viewport; within `margin` pixels of an edge the rasterizer's tie-breaking
    // rules decide, so the pixel is reported as Unsure and never compared.
    Sample sample_triangle(const Tri& t, double px, double py, double cw, double ch, double margin)
    {
        std::array<double, 3> X{}, Y{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            X[i] = (t[i].x + 1.0) * 0.5 * cw;
            Y[i] = (t[i].y + 1.0) * 0.5 * ch;
        }
        auto edge = [](double ax, double ay, double bx, double by, double x, double y) { return (bx - ax) * (y - ay) - (by - ay) * (x - ax); };
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

    struct Rect
    {
        int x, y, w, h;
    };

    struct Scan
    {
        int inside = 0, outside = 0, unsure = 0, bad = 0, max_dev = 0;
        std::string first_bad;
    };

    // Reads the whole cell and compares every pixel the model is certain about. The tolerance covers the rasterizer's
    // interpolation of 8-bit colors; the specification states no accuracy for it, so the measured maximum is printed.
    Scan scan_region(const std::vector<Tri>& tris, Rect r, int tolerance)
    {
        Scan s;
        const std::vector<std::uint8_t> pixels = demo::read_rgba8(r.x, r.y, r.w, r.h);
        for (int y = 0; y < r.h; ++y)
            for (int x = 0; x < r.w; ++x)
            {
                const Sample m = sample_scene(tris, x + 0.5, y + 0.5, r.w, r.h);
                if (m.cover == Cover::Unsure) { ++s.unsure; continue; }
                Rgba8 want{};
                for (std::size_t k = 0; k < 4; ++k) want[k] = m.cover == Cover::Inside ? to8(m.color[k]) : kClear8[k];
                ++(m.cover == Cover::Inside ? s.inside : s.outside);
                const std::uint8_t* got = &pixels[static_cast<std::size_t>((y * r.w + x) * 4)];
                bool ok = true;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    const int dev = std::abs(int(got[k]) - int(want[k]));
                    s.max_dev = std::max(s.max_dev, dev);
                    ok = ok && dev <= tolerance;
                }
                if (!ok && s.bad++ == 0)
                    s.first_bad = std::format("first mismatch at cell pixel ({},{}): got {} want {}", x, y,
                                              demo::format_rgba({got[0], got[1], got[2], got[3]}), demo::format_rgba(want));
            }
        return s;
    }

    // The integer cell: three bands (x in [-0.8, 0.8]); all green when every integer arrived exactly.
    std::vector<Tri> integer_model(bool band0_ok, bool band1_ok, bool band2_ok)
    {
        const Rgba green = {0.0, 0.8, 0.2, 1.0}, red = {0.9, 0.1, 0.1, 1.0};
        constexpr std::array<std::array<double, 2>, 3> y = {{{0.9, 0.4}, {0.2, -0.2}, {-0.4, -0.9}}};
        const std::array<bool, 3> ok = {band0_ok, band1_ok, band2_ok};
        std::vector<Tri> tris;
        for (std::size_t b = 0; b < 3; ++b)
        {
            const Rgba c = ok[b] ? green : red;
            const Vtx bl{-0.8, y[b][1], c}, br{0.8, y[b][1], c}, tl{-0.8, y[b][0], c}, tr{0.8, y[b][0], c};
            tris.push_back({bl, br, tl});
            tris.push_back({tl, br, tr});
        }
        return tris;
    }

    // -----------------------------------------------------------------------------------------
    // Cells and state
    // -----------------------------------------------------------------------------------------
    enum Cell : std::size_t { kC0, kC1, kC2, kC3, kC4, kC5, kC6, kC7, kC8, kC9, kC10, kC11, kCellCount };
    enum Vao : std::size_t { kVaoShared, kVaoInstanced, kVaoInteger, kVaoLegacy, kVaoMixed, kVaoTwoBindings, kVaoHigh, kVaoLimit, kVaoFresh, kVaoQuery, kVaoCount };

    constexpr std::array<const char*, kCellCount> kCellName = {
        "[0] one binding feeds 3 attributes (binding offset 24, relative offsets 0/8/12, stride 16)",
        "[1] same formats, glBindVertexBuffer re-points binding 0 to buffer B (offset 0, stride 24)",
        "[2] two bindings: positions (divisor 0) + instance offset/color (divisor 1), 4 instances",
        "[3] glVertexBindingDivisor(1, 2): instances share elements in pairs",
        "[4] glVertexAttribBinding moves the offset attribute to binding 2 (another buffer)",
        "[5] glVertexAttribIFormat: UNSIGNED_BYTE x2, INT 2^24+1, SHORT -3 arrive exactly",
        "[6] legacy glVertexAttribPointer defines attributes 0 and 1",
        "[7] glBindVertexBuffer re-points the legacy attribute 0 to buffer L2",
        "[8] legacy glVertexAttribPointer after Format/Binding/BindVertexBuffer(5) for the same attribute",
        "[9] two bindings (offset 0 and 8) on ONE buffer",
        "[10] bindings MAX_VERTEX_ATTRIB_BINDINGS-1 and -2",
        "[11] stride = MAX_VERTEX_ATTRIB_STRIDE, color at a large relative offset"};

    struct Region
    {
        Rect rect;
        std::vector<Tri> tris;
    };

    class SeparateVertexFormat final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            prog_basic_ = demo::build_program(env, "shaders/basic.vert", "shaders/basic.frag");
            prog_int_ = demo::build_program(env, "shaders/integer.vert", "shaders/integer.frag");
            u_scale_ = mgl::glGetUniformLocation(prog_basic_, "u_scale");
            u_expect_cell_ = mgl::glGetUniformLocation(prog_int_, "u_expect_cell");
            u_expect_big_ = mgl::glGetUniformLocation(prog_int_, "u_expect_big");
            u_expect_neg_ = mgl::glGetUniformLocation(prog_int_, "u_expect_neg");

            max_attribs_ = demo::get_integer(mgl::GetParameter::MaxVertexAttribs);
            max_bindings_ = demo::get_integer(mgl::GetParameter::MaxVertexAttribBindings);
            max_rel_offset_ = demo::get_integer(mgl::GetParameter::MaxVertexAttribRelativeOffset);
            max_stride_ = demo::get_integer(mgl::GetParameter::MaxVertexAttribStride);
            // The cell for the limits uses the largest legal stride and a relative offset next to it (capped at 2048 so a driver
            // that reports a huge limit does not make the demo allocate gigabytes).
            limit_stride_ = std::min(max_stride_, 2048);
            limit_rel_ = std::max(0, (std::min(max_rel_offset_, limit_stride_ - 4) / 4) * 4);
            high_a_ = static_cast<GLuint>(max_bindings_ - 1);
            high_b_ = static_cast<GLuint>(max_bindings_ - 2);

            build_buffer_data();
            upload_buffers();
            create_vaos();
            build_scenes(env);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);  // exact color writes
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear8[0] / 255.0f, kClear8[1] / 255.0f, kClear8[2] / 255.0f, kClear8[3] / 255.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            draw_all();
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_pixels(check);
            check_negative_controls(check);
            render(env, 0);  // restore the image after the control draw
            check_limits(check);
            check_defaults(check);
            check_format_and_binding_state(check);
            check_legacy_interplay(check);
            check_errors(check);
            check_isolation(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(scratch_.size()), scratch_.data());
            mgl::glDeleteProgram(prog_basic_);
            mgl::glDeleteProgram(prog_int_);
        }

    private:
        // ---------------------------------------------------------------------------------------------
        // Source data (bytes)
        // ---------------------------------------------------------------------------------------------
        void build_buffer_data()
        {
            // A: 24 bytes of junk (floats that would draw something wrong), then V16 = {x, y, rgba, short ox, oy}, stride 16.
            Bytes& a = bytes_[kBufA];
            for (const float f : {0.9f, -0.9f, 0.5f, 0.5f, -0.5f, 0.7f}) a.put(f);
            for (std::size_t i = 0; i < 3; ++i)
            {
                a.put(kTri0[i].x);
                a.put(kTri0[i].y);
                a.rgba(kTri0[i].c[0], kTri0[i].c[1], kTri0[i].c[2], kTri0[i].c[3]);
                a.put(snorm16(kOff0[i][0]));
                a.put(snorm16(kOff0[i][1]));
            }
            // B: V24 = the same 16 bytes + 8 bytes of padding that must be skipped through the larger stride.
            Bytes& b = bytes_[kBufB];
            for (std::size_t i = 0; i < 3; ++i)
            {
                b.put(kTri1[i].x);
                b.put(kTri1[i].y);
                b.rgba(kTri1[i].c[0], kTri1[i].c[1], kTri1[i].c[2], kTri1[i].c[3]);
                b.put(snorm16(kOff1[i][0]));
                b.put(snorm16(kOff1[i][1]));
                b.pad(8, 0x7F);
            }
            // P: the quad strip (positions only, tight). Q: per-instance {ox, oy, rgba}, stride 12. R: per-instance offsets (tight).
            for (const auto& q : kQuad)
            {
                bytes_[kBufP].put(q[0]);
                bytes_[kBufP].put(q[1]);
            }
            for (const Instance& i : kInstances)
            {
                bytes_[kBufQ].put(i.ox);
                bytes_[kBufQ].put(i.oy);
                bytes_[kBufQ].rgba(i.c[0], i.c[1], i.c[2], i.c[3]);
            }
            for (const auto& d : kDiagonal)
            {
                bytes_[kBufR].put(d[0]);
                bytes_[kBufR].put(d[1]);
            }
            // Int: 18 vertices {x, y, ubyte cx, cy, 2 pad, int big, short neg, 2 pad}: stride 20. Three bands, 6 vertices each.
            constexpr std::array<std::array<float, 2>, 3> y = {{{0.9f, 0.4f}, {0.2f, -0.2f}, {-0.4f, -0.9f}}};
            for (std::size_t band = 0; band < 3; ++band)
            {
                const float top = y[band][0], bottom = y[band][1];
                const std::array<std::array<float, 2>, 6> corners = {{{-0.8f, bottom}, {0.8f, bottom}, {-0.8f, top}, {-0.8f, top}, {0.8f, bottom}, {0.8f, top}}};
                for (const auto& corner : corners)
                {
                    Bytes& n = bytes_[kBufInt];
                    n.put(corner[0]);
                    n.put(corner[1]);
                    n.put(kIntCell[0]);
                    n.put(kIntCell[1]);
                    n.pad(2);
                    n.put(kIntBig);
                    n.put(kIntNeg);
                    n.pad(2);
                }
            }
            // L1: 16 junk bytes + 3 x {x, y, rgba} (stride 12): the legacy pointers use offsets 16 and 24. L2: positions only matter.
            for (const float f : {-0.4f, 0.4f, 0.4f, -0.4f}) bytes_[kBufL1].put(f);
            for (const Corner& c : kTriL1)
            {
                bytes_[kBufL1].put(c.x);
                bytes_[kBufL1].put(c.y);
                bytes_[kBufL1].rgba(c.c[0], c.c[1], c.c[2], c.c[3]);
            }
            for (const Corner& c : kTriL2)
            {
                bytes_[kBufL2].put(c.x);
                bytes_[kBufL2].put(c.y);
                bytes_[kBufL2].rgba(c.c[0], c.c[1], c.c[2], c.c[3]);
            }
            for (int i = 0; i < 3; ++i)
            {
                bytes_[kBufX].put(kShiftX[0]);
                bytes_[kBufX].put(kShiftX[1]);
                bytes_[kBufY].put(kShiftY[0]);
                bytes_[kBufY].put(kShiftY[1]);
            }
            for (const Corner& c : kTriS)
            {
                bytes_[kBufS].put(c.x);
                bytes_[kBufS].put(c.y);
                bytes_[kBufS].rgba(c.c[0], c.c[1], c.c[2], c.c[3]);
            }
            for (const Corner& c : kTriHigh)
            {
                bytes_[kBufTP].put(c.x);
                bytes_[kBufTP].put(c.y);
                bytes_[kBufTC].rgba(c.c[0], c.c[1], c.c[2], c.c[3]);
            }
            // LB: three vertices of limit_stride_ bytes: the position at 0, the color at limit_rel_, filler elsewhere.
            for (const Corner& c : kTriLimit)
            {
                Bytes& v = bytes_[kBufLB];
                const std::size_t start = v.size();
                v.put(c.x);
                v.put(c.y);
                v.pad(static_cast<std::size_t>(limit_rel_) - 8, 0x33);
                v.rgba(c.c[0], c.c[1], c.c[2], c.c[3]);
                v.pad(start + static_cast<std::size_t>(limit_stride_) - v.size(), 0x33);
            }
        }

        void upload_buffers()
        {
            mgl::glGenBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            for (std::size_t i = 0; i < kBufCount; ++i)
            {
                mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[i]);
                mgl::glBufferData(mgl::BufferTarget::Array, std::span<const std::uint8_t>(bytes_[i].b), mgl::BufferUsage::StaticDraw);
            }
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glGenBuffers(static_cast<GLsizei>(scratch_.size()), scratch_.data());  // names for the state/error checks (not yet bound)
        }

        // ---------------------------------------------------------------------------------------------
        // VAO setup with the ES 3.1 commands. ARRAY_BUFFER is never touched except for the two legacy VAOs.
        // ---------------------------------------------------------------------------------------------
        void create_vaos()
        {
            mgl::glGenVertexArrays(static_cast<GLsizei>(vao_.size()), vao_.data());

            // [0, 1] kVaoShared: ONE binding (0) feeds attributes 0, 1 and 2. relativeoffset = position inside one vertex.
            mgl::glBindVertexArray(vao_[kVaoShared]);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glEnableVertexAttribArray(kColor);
            mgl::glEnableVertexAttribArray(kOffset);
            mgl::glVertexAttribFormat(kPos, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kPos, 0);
            mgl::glVertexAttribFormat(kColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, 8);
            mgl::glVertexAttribBinding(kColor, 0);
            mgl::glVertexAttribFormat(kOffset, 2, mgl::DataType::Short, GL_TRUE, 12);
            mgl::glVertexAttribBinding(kOffset, 0);

            // [2, 3, 4] kVaoInstanced: binding 0 = positions, binding 1 = per-instance data (color at 8, offset at 0).
            mgl::glBindVertexArray(vao_[kVaoInstanced]);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glEnableVertexAttribArray(kColor);
            mgl::glEnableVertexAttribArray(kOffset);
            mgl::glVertexAttribFormat(kPos, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kPos, 0);
            mgl::glVertexAttribFormat(kColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, 8);
            mgl::glVertexAttribBinding(kColor, 1);
            mgl::glVertexAttribFormat(kOffset, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kOffset, 1);
            mgl::glBindVertexBuffer(0, buf_[kBufP], 0, 8);
            mgl::glBindVertexBuffer(1, buf_[kBufQ], 0, 12);
            mgl::glBindVertexBuffer(2, buf_[kBufR], 0, 8);  // fed to the offset attribute in cell 4 only

            // [5] kVaoInteger: pure integer attributes with glVertexAttribIFormat; one binding, relative offsets 8, 12, 16.
            mgl::glBindVertexArray(vao_[kVaoInteger]);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glEnableVertexAttribArray(kCell);
            mgl::glEnableVertexAttribArray(kBig);
            mgl::glEnableVertexAttribArray(kNeg);
            mgl::glVertexAttribFormat(kPos, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribIFormat(kCell, 2, mgl::DataType::UnsignedByte, 8);
            mgl::glVertexAttribIFormat(kBig, 1, mgl::DataType::Int, 12);
            mgl::glVertexAttribIFormat(kNeg, 1, mgl::DataType::Short, 16);
            for (const mgl::AttribLocation a : {kPos, kCell, kBig, kNeg}) mgl::glVertexAttribBinding(a, 0);
            mgl::glBindVertexBuffer(0, buf_[kBufInt], 0, 20);

            // [6, 7] kVaoLegacy: attributes 0 and 1 defined with glVertexAttribPointer (stride 12, offsets 16 and 24 in L1).
            mgl::glBindVertexArray(vao_[kVaoLegacy]);
            legacy_pointers(buf_[kBufL1]);

            // [8] kVaoMixed: legacy attributes 0 and 1 as above, then attribute 2 the new way (Format, Binding 5, BindVertexBuffer 5
            // with buffer X) and finally with the LEGACY call and buffer Y. The legacy call re-maps attribute 2 to binding 2 and
            // binds Y there; binding 5 keeps X but nothing reads it any more.
            mgl::glBindVertexArray(vao_[kVaoMixed]);
            legacy_pointers(buf_[kBufL1]);
            mgl::glEnableVertexAttribArray(kOffset);
            mgl::glVertexAttribFormat(kOffset, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kOffset, 5);
            mgl::glBindVertexBuffer(5, buf_[kBufX], 0, 8);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buf_[kBufY]);
            mgl::glVertexAttribPointer(kOffset, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            // [9] kVaoTwoBindings: ONE buffer S through two bindings: the same bytes, offsets 0 and 8, equal strides.
            mgl::glBindVertexArray(vao_[kVaoTwoBindings]);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glEnableVertexAttribArray(kColor);
            mgl::glVertexAttribFormat(kPos, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kPos, 0);
            mgl::glVertexAttribFormat(kColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, 0);
            mgl::glVertexAttribBinding(kColor, 1);
            mgl::glBindVertexBuffer(0, buf_[kBufS], 0, 12);
            mgl::glBindVertexBuffer(1, buf_[kBufS], 8, 12);

            // [10] kVaoHigh: the two highest binding indices.
            mgl::glBindVertexArray(vao_[kVaoHigh]);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glEnableVertexAttribArray(kColor);
            mgl::glVertexAttribFormat(kPos, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kPos, high_a_);
            mgl::glVertexAttribFormat(kColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, 0);
            mgl::glVertexAttribBinding(kColor, high_b_);
            mgl::glBindVertexBuffer(high_a_, buf_[kBufTP], 0, 8);
            mgl::glBindVertexBuffer(high_b_, buf_[kBufTC], 0, 4);

            // [11] kVaoLimit: the largest legal stride, the color at a large relative offset.
            mgl::glBindVertexArray(vao_[kVaoLimit]);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glEnableVertexAttribArray(kColor);
            mgl::glVertexAttribFormat(kPos, 2, mgl::DataType::Float, GL_FALSE, 0);
            mgl::glVertexAttribBinding(kPos, 0);
            mgl::glVertexAttribFormat(kColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, static_cast<GLuint>(limit_rel_));
            mgl::glVertexAttribBinding(kColor, 0);
            mgl::glBindVertexBuffer(0, buf_[kBufLB], 0, limit_stride_);

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        // The legacy way: each call records ARRAY_BUFFER, the offset (pointer) and the stride in the binding with the attribute's index.
        void legacy_pointers(mgl::BufferId l1) const
        {
            mgl::glBindBuffer(mgl::BufferTarget::Array, l1);
            mgl::glEnableVertexAttribArray(kPos);
            mgl::glVertexAttribPointer(kPos, 2, mgl::DataType::Float, GL_FALSE, kStride12, demo::buffer_offset(kL1Junk));
            mgl::glEnableVertexAttribArray(kColor);
            mgl::glVertexAttribPointer(kColor, 4, mgl::DataType::UnsignedByte, GL_TRUE, kStride12, demo::buffer_offset(kL1Junk + 8));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        // ---------------------------------------------------------------------------------------------
        // CPU scenes: the same bindings, written as data
        // ---------------------------------------------------------------------------------------------
        Attr attr(int size, Fmt fmt, long rel, Buf buf, long offset, long stride, unsigned divisor) const
        {
            return {true, size, fmt, rel, &bytes_[buf], offset, stride, divisor};
        }

        void build_scenes(const demo::Env& env)
        {
            const int cw = env.width / kCols, ch = env.height / kRows;
            for (std::size_t c = 0; c < kCellCount; ++c)
                regions_[c].rect = {static_cast<int>(c % kCols) * cw, (kRows - 1 - static_cast<int>(c / kCols)) * ch, cw, ch};

            using F = Fmt;
            const Attr off = {};  // a disabled attribute: current value (0, 0, 0, 1)
            scenes_[kC0] = {{attr(2, F::Float, 0, kBufA, kAJunk, kStrideA, 0), attr(4, F::UByteNorm, 8, kBufA, kAJunk, kStrideA, 0),
                             attr(2, F::ShortNorm, 12, kBufA, kAJunk, kStrideA, 0)}};
            scenes_[kC1] = {{attr(2, F::Float, 0, kBufB, 0, kStrideB, 0), attr(4, F::UByteNorm, 8, kBufB, 0, kStrideB, 0),
                             attr(2, F::ShortNorm, 12, kBufB, 0, kStrideB, 0)}};
            auto instanced = [&](unsigned divisor, Buf offset_buf, long offset_stride, long offset_rel) {
                Scene s;
                s.attr = {attr(2, F::Float, 0, kBufP, 0, 8, 0), attr(4, F::UByteNorm, 8, kBufQ, 0, 12, divisor),
                          attr(2, F::Float, offset_rel, offset_buf, 0, offset_stride, divisor)};
                s.mode = mgl::PrimitiveType::TriangleStrip;
                s.count = 4;
                s.instances = 4;
                s.sx = kInstScaleX;
                s.sy = kInstScaleY;
                return s;
            };
            scenes_[kC2] = instanced(1, kBufQ, 12, 0);
            scenes_[kC3] = instanced(2, kBufQ, 12, 0);
            scenes_[kC4] = instanced(1, kBufR, 8, 0);
            auto legacy = [&](Buf positions, long pos_offset) {
                Scene s;
                s.attr = {attr(2, F::Float, 0, positions, pos_offset, kStride12, 0), attr(4, F::UByteNorm, 0, kBufL1, kL1Junk + 8, kStride12, 0), off};
                return s;
            };
            scenes_[kC6] = legacy(kBufL1, kL1Junk);
            scenes_[kC7] = legacy(kBufL2, 0);
            scenes_[kC8] = legacy(kBufL1, kL1Junk);
            scenes_[kC8].attr[2] = attr(2, F::Float, 0, kBufY, 0, 8, 0);
            scenes_[kC9] = {{attr(2, F::Float, 0, kBufS, 0, 12, 0), attr(4, F::UByteNorm, 0, kBufS, 8, 12, 0), off}};
            scenes_[kC10] = {{attr(2, F::Float, 0, kBufTP, 0, 8, 0), attr(4, F::UByteNorm, 0, kBufTC, 0, 4, 0), off}};
            scenes_[kC11] = {{attr(2, F::Float, 0, kBufLB, 0, limit_stride_, 0), attr(4, F::UByteNorm, limit_rel_, kBufLB, 0, limit_stride_, 0), off}};

            for (const Cell c : {kC0, kC1, kC2, kC3, kC4, kC6, kC7, kC8, kC9, kC10, kC11}) regions_[c].tris = model_scene(scenes_[c]);
            regions_[kC5].tris = integer_model(true, true, true);
        }

        // ---------------------------------------------------------------------------------------------
        // Drawing
        // ---------------------------------------------------------------------------------------------
        void begin_cell(Cell c, float sx = 1.0f, float sy = 1.0f) const
        {
            const Rect r = regions_[c].rect;
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUniform2f(u_scale_, sx, sy);
        }

        void draw_integer_cell(std::int32_t expect_big) const
        {
            mgl::glUseProgram(prog_int_);
            const Rect r = regions_[kC5].rect;
            mgl::glViewport(r.x, r.y, r.w, r.h);
            mgl::glUniform2ui(u_expect_cell_, kIntCell[0], kIntCell[1]);
            mgl::glUniform1i(u_expect_big_, expect_big);
            mgl::glUniform1i(u_expect_neg_, kIntNeg);
            mgl::glBindVertexArray(vao_[kVaoInteger]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 18);
            mgl::glUseProgram(prog_basic_);
        }

        void draw_all() const
        {
            using P = mgl::PrimitiveType;
            mgl::glUseProgram(prog_basic_);

            // [0] The binding is set right before drawing: buffer A, a 24-byte binding offset that skips the junk, stride 16.
            begin_cell(kC0);
            mgl::glBindVertexArray(vao_[kVaoShared]);
            mgl::glBindVertexBuffer(0, buf_[kBufA], kAJunk, static_cast<GLsizei>(kStrideA));
            mgl::glDrawArrays(P::Triangles, 0, 3);

            // [1] Only the binding changes: another buffer, offset and stride. No Format or Binding call is made.
            begin_cell(kC1);
            mgl::glBindVertexBuffer(0, buf_[kBufB], 0, static_cast<GLsizei>(kStrideB));
            mgl::glDrawArrays(P::Triangles, 0, 3);

            // [2] Binding 1 (instance data) advances once per instance; binding 0 (the quad) once per vertex.
            begin_cell(kC2, static_cast<float>(kInstScaleX), static_cast<float>(kInstScaleY));
            mgl::glBindVertexArray(vao_[kVaoInstanced]);
            mgl::glVertexAttribBinding(kOffset, 1);
            mgl::glVertexBindingDivisor(1, 1);
            mgl::glDrawArraysInstanced(P::TriangleStrip, 0, 4, 4);

            // [3] Divisor 2 on the binding: attributes 1 and 2 (both on binding 1) advance once per two instances.
            begin_cell(kC3, static_cast<float>(kInstScaleX), static_cast<float>(kInstScaleY));
            mgl::glVertexBindingDivisor(1, 2);
            mgl::glDrawArraysInstanced(P::TriangleStrip, 0, 4, 4);

            // [4] Attribute 2 reads from binding 2 (buffer R) with its own divisor; attribute 1 stays on binding 1.
            begin_cell(kC4, static_cast<float>(kInstScaleX), static_cast<float>(kInstScaleY));
            mgl::glVertexBindingDivisor(1, 1);
            mgl::glVertexAttribBinding(kOffset, 2);
            mgl::glVertexBindingDivisor(2, 1);
            mgl::glDrawArraysInstanced(P::TriangleStrip, 0, 4, 4);

            // [5] Integer formats: the shader compares what it received with the expected values.
            draw_integer_cell(kIntBig);

            // [6] Attributes 0 and 1 defined by glVertexAttribPointer.
            begin_cell(kC6);
            mgl::glBindVertexArray(vao_[kVaoLegacy]);
            mgl::glDrawArrays(P::Triangles, 0, 3);

            // [7] glBindVertexBuffer(0, ...) changes the source of the LEGACY attribute 0 (binding 0 is its binding); attribute 1
            // (binding 1) keeps reading the colors of L1. Afterwards the legacy call is repeated, which restores binding 0.
            begin_cell(kC7);
            mgl::glBindVertexBuffer(0, buf_[kBufL2], 0, static_cast<GLsizei>(kStride12));
            mgl::glDrawArrays(P::Triangles, 0, 3);
            legacy_pointers(buf_[kBufL1]);

            // [8] The legacy pointer of attribute 2 (buffer Y) wins over the earlier binding 5 (buffer X).
            begin_cell(kC8);
            mgl::glBindVertexArray(vao_[kVaoMixed]);
            mgl::glDrawArrays(P::Triangles, 0, 3);

            // [9] Two bindings on one buffer.
            begin_cell(kC9);
            mgl::glBindVertexArray(vao_[kVaoTwoBindings]);
            mgl::glDrawArrays(P::Triangles, 0, 3);

            // [10] The highest binding indices.
            begin_cell(kC10);
            mgl::glBindVertexArray(vao_[kVaoHigh]);
            mgl::glDrawArrays(P::Triangles, 0, 3);

            // [11] The limits of stride and relative offset.
            begin_cell(kC11);
            mgl::glBindVertexArray(vao_[kVaoLimit]);
            mgl::glDrawArrays(P::Triangles, 0, 3);
        }

        // ---------------------------------------------------------------------------------------------
        // Pixel checks
        // ---------------------------------------------------------------------------------------------
        void check_pixels(demo::Check& check) const
        {
            for (std::size_t c = 0; c < kCellCount; ++c)
            {
                const Scan s = scan_region(regions_[c].tris, regions_[c].rect, 3);
                const bool enough = s.inside >= 150 && s.outside >= 200;
                check.expect(s.bad == 0 && enough,
                             std::format("{}: {} covered + {} background pixels match the CPU model (max deviation {}/255, {} edge pixels skipped)", kCellName[c],
                                         s.inside, s.outside, s.max_dev, s.unsure),
                             s.bad ? std::format("{} mismatches; {}", s.bad, s.first_bad)
                                   : !enough ? std::format("only {} covered / {} background pixels were compared", s.inside, s.outside)
                                             : std::string{});
            }
        }

        // Model mutations and one draw with a wrong expectation: if the pixels equal the real model they cannot equal these.
        void check_negative_controls(demo::Check& check) const
        {
            auto differs = [&](Cell c, const Scene& mutated, std::string_view what) {
                const Scan s = scan_region(model_scene(mutated), regions_[c].rect, 3);
                const bool ok = s.bad >= 30;
                check.expect(ok, std::format("negative control: {} ({} mismatching pixels)", what, s.bad),
                             ok ? std::string{} : "the mutated model still matches the pixels, so the comparison cannot detect this error");
            };
            Scene s = scenes_[kC0];
            for (Attr& a : s.attr) a.offset = 0;
            differs(kC0, s, "cell 0 modelled with binding offset 0 instead of 24 (junk bytes are read)");
            s = scenes_[kC0];
            s.attr[1].rel = 0;
            differs(kC0, s, "cell 0 modelled with the color at relative offset 0 instead of 8");
            s = scenes_[kC1];
            for (Attr& a : s.attr) a.stride = kStrideA;
            differs(kC1, s, "cell 1 modelled with stride 16 instead of 24");
            s = scenes_[kC3];
            s.attr[1].divisor = s.attr[2].divisor = 1;
            differs(kC3, s, "cell 3 modelled with divisor 1 instead of 2");
            s = scenes_[kC4];
            s.attr[2] = scenes_[kC2].attr[2];
            differs(kC4, s, "cell 4 modelled without moving the offset attribute to binding 2");
            s = scenes_[kC7];
            s.attr[0] = scenes_[kC6].attr[0];
            differs(kC7, s, "cell 7 modelled without re-pointing binding 0 to buffer L2");
            s = scenes_[kC8];
            s.attr[2] = attr(2, Fmt::Float, 0, kBufX, 0, 8, 0);
            differs(kC8, s, "cell 8 modelled with the offsets of buffer X (the earlier binding 5) instead of Y");
            s = scenes_[kC9];
            s.attr[1].offset = 0;
            differs(kC9, s, "cell 9 modelled with the color binding at offset 0 instead of 8");

            // The integer cell with a wrong expectation: the INT band must turn red (float rounding of 2^24 + 1 would give 2^24).
            draw_integer_cell(16777216);
            const Scan red = scan_region(integer_model(true, false, true), regions_[kC5].rect, 3);
            check.expect(red.bad == 0 && red.inside > 500,
                         std::format("negative control: expecting 16777216 instead of 2^24 + 1 turns exactly band 1 red ({} pixels compared, {} mismatches)", red.inside, red.bad),
                         red.first_bad);
        }

        // ---------------------------------------------------------------------------------------------
        // State queries
        // ---------------------------------------------------------------------------------------------
        static int attrib_i(unsigned index, mgl::VertexAttribParameter pname)
        {
            GLint v = std::numeric_limits<GLint>::min();
            mgl::glGetVertexAttribiv(mgl::AttribLocation{static_cast<GLuint>(index)}, pname, &v);
            return v;
        }
        static int binding_i(mgl::GetParameter pname, unsigned index)
        {
            GLint v = std::numeric_limits<GLint>::min();
            mgl::glGetIntegeri_v(pname, index, &v);
            return v;
        }
        static std::int64_t binding_offset64(unsigned index)
        {
            GLint64 v = std::numeric_limits<GLint64>::min();
            mgl::glGetInteger64i_v(mgl::GetParameter::VertexBindingOffset, index, &v);
            return v;
        }

        void check_limits(demo::Check& check) const
        {
            check.expect(max_bindings_ >= 16, std::format("MAX_VERTEX_ATTRIB_BINDINGS = {} >= 16 (Table 21.42)", max_bindings_));
            check.expect(max_rel_offset_ >= 2047, std::format("MAX_VERTEX_ATTRIB_RELATIVE_OFFSET = {} >= 2047 (Table 21.42)", max_rel_offset_));
            check.expect(max_stride_ >= 2048, std::format("MAX_VERTEX_ATTRIB_STRIDE = {} >= 2048 (Table 21.42)", max_stride_));
            check.expect(max_attribs_ >= 16, std::format("MAX_VERTEX_ATTRIBS = {} >= 16 (Table 21.44)", max_attribs_));
            check.expect_no_gl_error("no GL error from the limit queries");
        }

        // A fresh VAO has the initial state of Table 21.3: binding i for attribute i, relative offset 0, bindings with
        // buffer 0, offset 0, stride 16 and divisor 0.
        void check_defaults(demo::Check& check) const
        {
            mgl::glBindVertexArray(vao_[kVaoFresh]);
            bool ok = true;
            std::string detail;
            for (const unsigned a : {0u, 1u, 7u, static_cast<unsigned>(max_attribs_ - 1)})
            {
                const int b = attrib_i(a, mgl::VertexAttribParameter::Binding), r = attrib_i(a, mgl::VertexAttribParameter::RelativeOffset);
                if (b != static_cast<int>(a) || r != 0) { ok = false; detail += std::format(" attrib {}: binding {} relative offset {};", a, b, r); }
            }
            check.expect(ok, "fresh VAO: VERTEX_ATTRIB_BINDING of attribute i is i and VERTEX_ATTRIB_RELATIVE_OFFSET is 0 (attributes 0, 1, 7, MAX-1)", detail);
            ok = true;
            detail.clear();
            for (const unsigned b : {0u, 1u, 7u, static_cast<unsigned>(max_bindings_ - 1)})
            {
                const int stride = binding_i(mgl::GetParameter::VertexBindingStride, b), divisor = binding_i(mgl::GetParameter::VertexBindingDivisor, b);
                const int buffer = binding_i(mgl::GetParameter::VertexBindingBuffer, b), offset = binding_i(mgl::GetParameter::VertexBindingOffset, b);
                const std::int64_t offset64 = binding_offset64(b);
                if (stride != 16 || divisor != 0 || buffer != 0 || offset != 0 || offset64 != 0)
                {
                    ok = false;
                    detail += std::format(" binding {}: stride {} divisor {} buffer {} offset {}/{};", b, stride, divisor, buffer, offset, offset64);
                }
            }
            check.expect(ok, "fresh VAO: every binding has stride 16 (!), offset 0, divisor 0 and buffer 0 (bindings 0, 1, 7, MAX-1)", detail);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect_no_gl_error("no GL error from the default-state queries");
        }

        // State written by the new commands, on the scratch VAO.
        void check_format_and_binding_state(demo::Check& check) const
        {
            using P = mgl::VertexAttribParameter;
            mgl::glBindVertexArray(vao_[kVaoQuery]);

            // glVertexAttribFormat / IFormat store size, type, normalized, integer flag, relative offset - and leave the binding alone.
            mgl::glVertexAttribFormat(mgl::AttribLocation{3u}, 4, mgl::DataType::UnsignedByte, GL_TRUE, 100);
            check.expect(attrib_i(3, P::ArraySize) == 4 && attrib_i(3, P::ArrayType) == static_cast<int>(GL_UNSIGNED_BYTE) &&
                             attrib_i(3, P::ArrayNormalized) == 1 && attrib_i(3, P::ArrayInteger) == 0 && attrib_i(3, P::RelativeOffset) == 100 &&
                             attrib_i(3, P::Binding) == 3,
                         "glVertexAttribFormat(3, 4, UNSIGNED_BYTE, TRUE, 100): SIZE 4, TYPE, NORMALIZED 1, INTEGER 0, RELATIVE_OFFSET 100, BINDING still 3");
            mgl::glVertexAttribIFormat(mgl::AttribLocation{4u}, 2, mgl::DataType::Short, 6);
            check.expect(attrib_i(4, P::ArraySize) == 2 && attrib_i(4, P::ArrayType) == static_cast<int>(GL_SHORT) && attrib_i(4, P::ArrayNormalized) == 0 &&
                             attrib_i(4, P::ArrayInteger) == 1 && attrib_i(4, P::RelativeOffset) == 6,
                         "glVertexAttribIFormat(4, 2, SHORT, 6): SIZE 2, TYPE SHORT, NORMALIZED 0, INTEGER 1, RELATIVE_OFFSET 6");
            mgl::glVertexAttribFormat(mgl::AttribLocation{4u}, 3, mgl::DataType::Float, GL_FALSE, 12);
            check.expect(attrib_i(4, P::ArrayInteger) == 0 && attrib_i(4, P::ArraySize) == 3 && attrib_i(4, P::RelativeOffset) == 12,
                         "glVertexAttribFormat after glVertexAttribIFormat on the same attribute: INTEGER back to 0 (section 10.6)");

            // glVertexAttribBinding changes only the mapping.
            mgl::glVertexAttribBinding(mgl::AttribLocation{3u}, 9);
            check.expect(attrib_i(3, P::Binding) == 9 && attrib_i(3, P::RelativeOffset) == 100 && attrib_i(3, P::ArraySize) == 4,
                         "glVertexAttribBinding(3, 9): VERTEX_ATTRIB_BINDING 9; relative offset and size untouched");

            // glBindVertexBuffer: buffer, offset (64-bit), stride; attributes mapped to the binding report the buffer through
            // VERTEX_ATTRIB_ARRAY_BUFFER_BINDING; other bindings are unaffected.
            mgl::glBindVertexBuffer(9, scratch_[0], 4096, 48);
            check.expect(binding_i(mgl::GetParameter::VertexBindingBuffer, 9) == static_cast<int>(scratch_[0].value) &&
                             binding_offset64(9) == 4096 && binding_i(mgl::GetParameter::VertexBindingOffset, 9) == 4096 &&
                             binding_i(mgl::GetParameter::VertexBindingStride, 9) == 48,
                         "glBindVertexBuffer(9, buf, 4096, 48): VERTEX_BINDING_BUFFER / OFFSET (GetInteger64i_v and GetIntegeri_v) / STRIDE read back");
            check.expect(attrib_i(3, P::ArrayBufferBinding) == static_cast<int>(scratch_[0].value),
                         "VERTEX_ATTRIB_ARRAY_BUFFER_BINDING of attribute 3 is the buffer of its binding 9 (section 10.6)");
            check.expect(binding_i(mgl::GetParameter::VertexBindingBuffer, 3) == 0 && attrib_i(5, P::ArrayBufferBinding) == 0,
                         "binding 3 and the unrelated attribute 5 still have buffer 0");
            mgl::glBindVertexBuffer(9, mgl::BufferId{0}, 0, 16);
            check.expect(binding_i(mgl::GetParameter::VertexBindingBuffer, 9) == 0 && attrib_i(3, P::ArrayBufferBinding) == 0,
                         "glBindVertexBuffer(9, 0, ...) detaches the buffer");

            // The 64-bit offset: GetInteger64i_v is exact; GetIntegeri_v clamps to the nearest int (section 2.2.2).
            constexpr GLintptr kBigOffset = (GLintptr{1} << 33) + 8;
            if constexpr (sizeof(GLintptr) >= 8)
            {
                mgl::glBindVertexBuffer(9, scratch_[0], kBigOffset, 8);
                const std::int64_t exact = binding_offset64(9);
                const int clamped = binding_i(mgl::GetParameter::VertexBindingOffset, 9);
                check.expect_or_known_deviation(exact == static_cast<std::int64_t>(kBigOffset),
                                                "glBindVertexBuffer with offset 2^33 + 8: VERTEX_BINDING_OFFSET is a 64-bit state value, GetInteger64i_v returns it exactly (Table 21.3)",
                                                std::format("returned {}", exact), "Mesa", "Mesa returns only the low 32 bits of VERTEX_BINDING_OFFSET");
                check.expect_or_known_deviation(clamped == INT_MAX,
                                                "VERTEX_BINDING_OFFSET 2^33 + 8 queried with GetIntegeri_v is clamped to INT_MAX (section 2.2.2)",
                                                std::format("returned {}", clamped), "Mesa", "Mesa truncates the 64-bit offset to 32 bits instead of clamping");
                mgl::glBindVertexBuffer(9, mgl::BufferId{0}, 0, 16);
            }

            // Divisor is binding state: two attributes on one binding see the same value, a different binding keeps its own.
            mgl::glVertexAttribBinding(mgl::AttribLocation{5u}, 9);
            mgl::glVertexBindingDivisor(9, 3);
            check.expect(binding_i(mgl::GetParameter::VertexBindingDivisor, 9) == 3 && attrib_i(3, P::ArrayDivisor) == 3 && attrib_i(5, P::ArrayDivisor) == 3 &&
                             attrib_i(6, P::ArrayDivisor) == 0 && binding_i(mgl::GetParameter::VertexBindingDivisor, 8) == 0,
                         "glVertexBindingDivisor(9, 3): VERTEX_BINDING_DIVISOR[9] = 3, VERTEX_ATTRIB_ARRAY_DIVISOR of both attributes on binding 9 = 3, others 0");

            // The indexed query errors of section 20.1.
            (void)demo::drain_gl_errors();
            GLint dummy = 0;
            mgl::glGetIntegeri_v(mgl::GetParameter::VertexBindingBuffer, static_cast<GLuint>(max_bindings_), &dummy);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(VERTEX_BINDING_BUFFER, MAX_VERTEX_ATTRIB_BINDINGS) -> INVALID_VALUE");

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        // The legacy commands are defined through the new ones (section 10.3.1, 10.3.2): both act on the same state.
        void check_legacy_interplay(demo::Check& check) const
        {
            using P = mgl::VertexAttribParameter;
            const mgl::AttribLocation a6{6u};

            // Start from an attribute that the new-style commands moved to binding 11 with divisor 3.
            mgl::glBindVertexArray(vao_[kVaoMixed]);
            check.expect(attrib_i(2, P::Binding) == 2 && binding_i(mgl::GetParameter::VertexBindingBuffer, 2) == static_cast<int>(buf_[kBufY].value) &&
                             binding_i(mgl::GetParameter::VertexBindingBuffer, 5) == static_cast<int>(buf_[kBufX].value),
                         "VAO of cell 8: glVertexAttribPointer re-mapped attribute 2 to binding 2 (buffer Y); binding 5 still holds buffer X");
            check.expect(binding_i(mgl::GetParameter::VertexBindingStride, 2) == 8 && binding_i(mgl::GetParameter::VertexBindingOffset, 2) == 0 &&
                             attrib_i(2, P::RelativeOffset) == 0 && attrib_i(2, P::ArrayStride) == 0,
                         "VAO of cell 8: stride 0 became the effective VERTEX_BINDING_STRIDE 8 while VERTEX_ATTRIB_ARRAY_STRIDE keeps the 0 passed to the call");
            mgl::glBindVertexArray(vao_[kVaoLegacy]);
            check.expect(binding_i(mgl::GetParameter::VertexBindingStride, 0) == 12 && binding_i(mgl::GetParameter::VertexBindingOffset, 0) == kL1Junk &&
                             binding_i(mgl::GetParameter::VertexBindingBuffer, 1) == static_cast<int>(buf_[kBufL1].value) &&
                             binding_i(mgl::GetParameter::VertexBindingOffset, 1) == kL1Junk + 8 && attrib_i(1, P::ArrayStride) == 12 &&
                             attrib_i(0, P::Binding) == 0 && attrib_i(1, P::Binding) == 1 && attrib_i(1, P::RelativeOffset) == 0,
                         "legacy VAO: glVertexAttribPointer set binding i = attribute i, VERTEX_BINDING_OFFSET to the pointer, STRIDE to the stride, relative offset 0");
            void* ptr = nullptr;
            mgl::glGetVertexAttribPointerv(mgl::AttribLocation{1u}, P::ArrayPointer, &ptr);
            check.expect_eq(reinterpret_cast<std::uintptr_t>(ptr), static_cast<std::uintptr_t>(kL1Junk + 8), "VERTEX_ATTRIB_ARRAY_POINTER of a legacy attribute is the offset passed to the call");

            // New-style setup, then the legacy calls on the same attribute (scratch VAO).
            mgl::glBindVertexArray(vao_[kVaoQuery]);
            mgl::glVertexAttribFormat(a6, 3, mgl::DataType::Float, GL_FALSE, 100);
            mgl::glVertexAttribBinding(a6, 11);
            mgl::glBindVertexBuffer(11, scratch_[1], 0, 24);
            mgl::glVertexBindingDivisor(11, 3);
            check.expect(attrib_i(6, P::Binding) == 11 && attrib_i(6, P::ArrayDivisor) == 3 && attrib_i(6, P::ArrayBufferBinding) == static_cast<int>(scratch_[1].value),
                         "attribute 6 set up the new way: BINDING 11, DIVISOR 3, BUFFER_BINDING = buffer of binding 11");

            // glVertexAttribDivisor == glVertexAttribBinding(index, index) + glVertexBindingDivisor(index, divisor)
            mgl::glVertexAttribDivisor(a6, 2);
            check.expect(attrib_i(6, P::Binding) == 6 && binding_i(mgl::GetParameter::VertexBindingDivisor, 6) == 2 && attrib_i(6, P::ArrayDivisor) == 2 &&
                             binding_i(mgl::GetParameter::VertexBindingDivisor, 11) == 3 && attrib_i(6, P::RelativeOffset) == 100,
                         "glVertexAttribDivisor(6, 2): VERTEX_ATTRIB_BINDING becomes 6, VERTEX_BINDING_DIVISOR[6] = 2, binding 11 keeps 3, relative offset untouched");

            // glVertexAttribPointer == Format(..., 0) + Binding(index, index) + BindVertexBuffer(index, ARRAY_BUFFER, pointer, effective stride); the divisor stays
            mgl::glBindBuffer(mgl::BufferTarget::Array, scratch_[2]);
            mgl::glVertexAttribPointer(a6, 2, mgl::DataType::Float, GL_FALSE, 0, demo::buffer_offset(40));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            check.expect(attrib_i(6, P::Binding) == 6 && attrib_i(6, P::RelativeOffset) == 0 && attrib_i(6, P::ArraySize) == 2 &&
                             binding_i(mgl::GetParameter::VertexBindingBuffer, 6) == static_cast<int>(scratch_[2].value) &&
                             binding_i(mgl::GetParameter::VertexBindingOffset, 6) == 40 && binding_i(mgl::GetParameter::VertexBindingStride, 6) == 8 &&
                             attrib_i(6, P::ArrayStride) == 0 && attrib_i(6, P::ArrayDivisor) == 2,
                         "glVertexAttribPointer(6, 2, FLOAT, FALSE, 0, 40) after the new commands: BINDING 6, RELATIVE_OFFSET reset to 0, buffer = ARRAY_BUFFER, "
                         "binding OFFSET 40, effective STRIDE 8, ARRAY_STRIDE 0, divisor of binding 6 unchanged (2)");
            ptr = nullptr;
            mgl::glGetVertexAttribPointerv(a6, P::ArrayPointer, &ptr);
            check.expect_eq(reinterpret_cast<std::uintptr_t>(ptr), std::uintptr_t{40}, "VERTEX_ATTRIB_ARRAY_POINTER of attribute 6 is the offset 40");
            check.expect(binding_i(mgl::GetParameter::VertexBindingBuffer, 11) == static_cast<int>(scratch_[1].value) && binding_i(mgl::GetParameter::VertexBindingStride, 11) == 24,
                         "the binding 11 set up earlier is not touched by the legacy call on attribute 6");

            // A pure integer attribute through the legacy API sets the integer flag the same way
            mgl::glBindBuffer(mgl::BufferTarget::Array, scratch_[2]);
            mgl::glVertexAttribIPointer(mgl::AttribLocation{7u}, 2, mgl::DataType::UnsignedShort, 0, demo::buffer_offset(8));
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            check.expect(attrib_i(7, P::ArrayInteger) == 1 && attrib_i(7, P::Binding) == 7 && binding_i(mgl::GetParameter::VertexBindingStride, 7) == 4,
                         "glVertexAttribIPointer(7, 2, UNSIGNED_SHORT, 0, 8): INTEGER 1, binding 7, effective stride 4");
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect_no_gl_error("no GL error from the legacy interplay checks");
        }

        // ---------------------------------------------------------------------------------------------
        // Errors (ES 3.2 section 10.3.1, 10.3.2)
        // ---------------------------------------------------------------------------------------------
        void check_errors(demo::Check& check) const
        {
            using Ec = mgl::ErrorCode;
            using P = mgl::VertexAttribParameter;
            const GLuint max_attribs = static_cast<GLuint>(max_attribs_), max_bindings = static_cast<GLuint>(max_bindings_);
            const mgl::AttribLocation a0{0u}, a_last{max_attribs - 1}, a_bad{max_attribs};
            (void)demo::drain_gl_errors();

            // -- the default vertex array object ------------------------------------------------------------------------------
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glVertexAttribFormat(a0, 2, mgl::DataType::Float, GL_FALSE, 0);
            check.expect_gl_error(Ec::InvalidOperation, "glVertexAttribFormat with the default VAO bound -> INVALID_OPERATION");
            mgl::glVertexAttribIFormat(a0, 2, mgl::DataType::Int, 0);
            check.expect_gl_error(Ec::InvalidOperation, "glVertexAttribIFormat with the default VAO bound -> INVALID_OPERATION");
            mgl::glVertexAttribBinding(a0, 0);
            check.expect_gl_error(Ec::InvalidOperation, "glVertexAttribBinding with the default VAO bound -> INVALID_OPERATION");
            mgl::glBindVertexBuffer(0, buf_[kBufP], 0, 8);
            check.expect_gl_error(Ec::InvalidOperation, "glBindVertexBuffer with the default VAO bound -> INVALID_OPERATION");
            mgl::glVertexBindingDivisor(0, 1);
            check.expect_gl_error(Ec::InvalidOperation, "glVertexBindingDivisor with the default VAO bound -> INVALID_OPERATION");

            // -- glVertexAttribFormat / glVertexAttribIFormat --------------------------------------------------------------------
            mgl::glBindVertexArray(vao_[kVaoQuery]);
            mgl::glVertexAttribFormat(a0, 2, mgl::DataType::Float, GL_FALSE, static_cast<GLuint>(max_rel_offset_));
            check.expect_no_gl_error("glVertexAttribFormat with relativeoffset = MAX_VERTEX_ATTRIB_RELATIVE_OFFSET is valid");
            mgl::glVertexAttribFormat(a_last, 4, mgl::DataType::UnsignedByte, GL_TRUE, 0);
            check.expect_no_gl_error("glVertexAttribFormat with attribindex = MAX_VERTEX_ATTRIBS - 1 is valid");
            mgl::glVertexAttribFormat(a_bad, 2, mgl::DataType::Float, GL_FALSE, 0);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribFormat with attribindex = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            mgl::glVertexAttribIFormat(a_bad, 2, mgl::DataType::Int, 0);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribIFormat with attribindex = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            if (max_rel_offset_ < INT_MAX)
            {
                const GLuint too_big = static_cast<GLuint>(max_rel_offset_) + 1;
                mgl::glVertexAttribFormat(a0, 2, mgl::DataType::Float, GL_FALSE, too_big);
                check.expect_gl_error(Ec::InvalidValue, "glVertexAttribFormat with relativeoffset = MAX_VERTEX_ATTRIB_RELATIVE_OFFSET + 1 -> INVALID_VALUE");
                mgl::glVertexAttribIFormat(a0, 2, mgl::DataType::Int, too_big);
                check.expect_gl_error(Ec::InvalidValue, "glVertexAttribIFormat with relativeoffset = MAX_VERTEX_ATTRIB_RELATIVE_OFFSET + 1 -> INVALID_VALUE");
            }
            mgl::glVertexAttribFormat(a0, 0, mgl::DataType::Float, GL_FALSE, 0);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribFormat with size 0 -> INVALID_VALUE");
            mgl::glVertexAttribFormat(a0, 5, mgl::DataType::Float, GL_FALSE, 0);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribFormat with size 5 -> INVALID_VALUE");
            mgl::glVertexAttribIFormat(a0, 5, mgl::DataType::Int, 0);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribIFormat with size 5 -> INVALID_VALUE");
            mgl::glVertexAttribFormat(a0, 3, mgl::DataType::Int2101010Rev, GL_TRUE, 0);
            check.expect_gl_error(Ec::InvalidOperation, "glVertexAttribFormat(INT_2_10_10_10_REV, size 3) -> INVALID_OPERATION (packed types need size 4)");
            mgl::glVertexAttribFormat(a0, 4, mgl::DataType::UnsignedInt2101010Rev, GL_TRUE, 0);
            check.expect_no_gl_error("glVertexAttribFormat(UNSIGNED_INT_2_10_10_10_REV, size 4) is valid");
            mgl::glVertexAttribFormat(a0, 4, mgl::DataType::UnsignedInt10F11F11FRev, GL_FALSE, 0);
            check.expect_gl_error(Ec::InvalidEnum, "glVertexAttribFormat(UNSIGNED_INT_10F_11F_11F_REV) -> INVALID_ENUM (not a vertex attribute type)");
            mgl::glVertexAttribIFormat(a0, 2, mgl::DataType::Float, 0);
            check.expect_gl_error(Ec::InvalidEnum, "glVertexAttribIFormat(FLOAT) -> INVALID_ENUM (integer types only)");
            mgl::glVertexAttribIFormat(a0, 2, mgl::DataType::HalfFloat, 0);
            check.expect_gl_error(Ec::InvalidEnum, "glVertexAttribIFormat(HALF_FLOAT) -> INVALID_ENUM");
            mgl::glVertexAttribIFormat(a0, 4, mgl::DataType::Int2101010Rev, 0);
            check.expect_gl_error(Ec::InvalidEnum, "glVertexAttribIFormat(INT_2_10_10_10_REV) -> INVALID_ENUM");
            mgl::glVertexAttribIFormat(a0, 2, mgl::DataType::UnsignedInt, 0);
            check.expect_no_gl_error("glVertexAttribIFormat(UNSIGNED_INT, 2) is valid");

            // An erroring command has no effect (section 2.3.1): the format specified last stays.
            mgl::glVertexAttribFormat(mgl::AttribLocation{2u}, 2, mgl::DataType::Float, GL_FALSE, 12);
            mgl::glVertexAttribFormat(mgl::AttribLocation{2u}, 5, mgl::DataType::UnsignedByte, GL_TRUE, 99);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribFormat(2, size 5, ...) -> INVALID_VALUE");
            check.expect(attrib_i(2, P::ArraySize) == 2 && attrib_i(2, P::RelativeOffset) == 12 && attrib_i(2, P::ArrayType) == static_cast<int>(GL_FLOAT),
                         "the failed glVertexAttribFormat changed nothing: SIZE 2, TYPE FLOAT, RELATIVE_OFFSET 12");

            // -- glVertexAttribBinding ------------------------------------------------------------------------------------------
            mgl::glVertexAttribBinding(a_last, max_bindings - 1);
            check.expect_no_gl_error("glVertexAttribBinding(MAX_VERTEX_ATTRIBS - 1, MAX_VERTEX_ATTRIB_BINDINGS - 1) is valid");
            mgl::glVertexAttribBinding(a_bad, 0);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribBinding with attribindex = MAX_VERTEX_ATTRIBS -> INVALID_VALUE");
            mgl::glVertexAttribBinding(a0, max_bindings);
            check.expect_gl_error(Ec::InvalidValue, "glVertexAttribBinding with bindingindex = MAX_VERTEX_ATTRIB_BINDINGS -> INVALID_VALUE");

            // -- glBindVertexBuffer -----------------------------------------------------------------------------------------------
            mgl::glBindVertexBuffer(max_bindings - 1, buf_[kBufP], 0, 8);
            check.expect_no_gl_error("glBindVertexBuffer(MAX_VERTEX_ATTRIB_BINDINGS - 1, ...) is valid");
            mgl::glBindVertexBuffer(max_bindings, buf_[kBufP], 0, 8);
            check.expect_gl_error(Ec::InvalidValue, "glBindVertexBuffer with bindingindex = MAX_VERTEX_ATTRIB_BINDINGS -> INVALID_VALUE");
            mgl::glBindVertexBuffer(1, buf_[kBufP], 0, static_cast<GLsizei>(max_stride_));
            check.expect_no_gl_error("glBindVertexBuffer with stride = MAX_VERTEX_ATTRIB_STRIDE is valid");
            if (max_stride_ < INT_MAX)
            {
                mgl::glBindVertexBuffer(1, buf_[kBufP], 0, static_cast<GLsizei>(max_stride_) + 1);
                check.expect_gl_error(Ec::InvalidValue, "glBindVertexBuffer with stride = MAX_VERTEX_ATTRIB_STRIDE + 1 -> INVALID_VALUE");
            }
            mgl::glBindVertexBuffer(1, buf_[kBufP], 0, -1);
            check.expect_gl_error(Ec::InvalidValue, "glBindVertexBuffer with a negative stride -> INVALID_VALUE");
            mgl::glBindVertexBuffer(1, buf_[kBufP], -1, 8);
            check.expect_gl_error(Ec::InvalidValue, "glBindVertexBuffer with a negative offset -> INVALID_VALUE");

            // buffer names: never generated and deleted are errors; generated but never bound is a valid name (ES 3.1 section 10.3.1)
            mgl::glBindVertexBuffer(1, mgl::BufferId{0x7FFFFFF0u}, 0, 8);
            check.expect_gl_error(Ec::InvalidOperation, "glBindVertexBuffer with a name never returned by glGenBuffers -> INVALID_OPERATION");
            mgl::BufferId deleted{};
            mgl::glGenBuffers(1, &deleted);
            mgl::glBindBuffer(mgl::BufferTarget::Array, deleted);
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
            mgl::glDeleteBuffers(1, &deleted);
            mgl::glBindVertexBuffer(1, deleted, 0, 8);
            check.expect_gl_error(Ec::InvalidOperation, "glBindVertexBuffer with a deleted buffer name -> INVALID_OPERATION");
            mgl::glBindVertexBuffer(2, scratch_[3], 0, 8);
            check.expect_no_gl_error("glBindVertexBuffer with a name from glGenBuffers that was never bound is valid");
            check.expect_eq(binding_i(mgl::GetParameter::VertexBindingBuffer, 2), static_cast<int>(scratch_[3].value),
                            "... and VERTEX_BINDING_BUFFER reports that name");

            // an erroring glBindVertexBuffer leaves the binding as it was
            mgl::glBindVertexBuffer(4, buf_[kBufQ], 64, 12);
            mgl::glBindVertexBuffer(4, buf_[kBufR], 128, -1);
            check.expect_gl_error(Ec::InvalidValue, "glBindVertexBuffer(4, R, 128, -1) -> INVALID_VALUE");
            check.expect(binding_i(mgl::GetParameter::VertexBindingBuffer, 4) == static_cast<int>(buf_[kBufQ].value) && binding_offset64(4) == 64 &&
                             binding_i(mgl::GetParameter::VertexBindingStride, 4) == 12,
                         "the failed glBindVertexBuffer changed nothing: binding 4 still has buffer Q, offset 64, stride 12");

            // -- glVertexBindingDivisor ---------------------------------------------------------------------------------------------
            mgl::glVertexBindingDivisor(max_bindings - 1, 5);
            check.expect_no_gl_error("glVertexBindingDivisor(MAX_VERTEX_ATTRIB_BINDINGS - 1, 5) is valid");
            mgl::glVertexBindingDivisor(max_bindings, 1);
            check.expect_gl_error(Ec::InvalidValue, "glVertexBindingDivisor with bindingindex = MAX_VERTEX_ATTRIB_BINDINGS -> INVALID_VALUE");

            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect_no_gl_error("no stray error remains after the error checks");
        }

        // State is per VAO: everything the checks above did on other VAOs left the fresh VAO untouched.
        void check_isolation(demo::Check& check) const
        {
            mgl::glBindVertexArray(vao_[kVaoFresh]);
            const bool ok = binding_i(mgl::GetParameter::VertexBindingStride, 4) == 16 && binding_i(mgl::GetParameter::VertexBindingBuffer, 4) == 0 &&
                            binding_i(mgl::GetParameter::VertexBindingDivisor, static_cast<unsigned>(max_bindings_ - 1)) == 0 &&
                            attrib_i(6, mgl::VertexAttribParameter::Binding) == 6 && attrib_i(6, mgl::VertexAttribParameter::RelativeOffset) == 0;
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            check.expect(ok, "state is per VAO: the fresh VAO still has the initial bindings and formats after all commands on other VAOs");
        }

        // ---------------------------------------------------------------------------------------------
        // Members
        // ---------------------------------------------------------------------------------------------
        mgl::ProgramId prog_basic_{}, prog_int_{};
        mgl::UniformLocation u_scale_{}, u_expect_cell_{}, u_expect_big_{}, u_expect_neg_{};
        std::array<mgl::VertexArrayId, kVaoCount> vao_{};
        std::array<mgl::BufferId, kBufCount> buf_{};
        std::array<Bytes, kBufCount> bytes_{};
        std::array<mgl::BufferId, 4> scratch_{};  // generated names for state checks; never given storage
        std::array<Scene, kCellCount> scenes_{};
        std::array<Region, kCellCount> regions_{};
        int max_attribs_ = 0, max_bindings_ = 0, max_rel_offset_ = 0, max_stride_ = 0, limit_stride_ = 0, limit_rel_ = 0;
        GLuint high_a_ = 0, high_b_ = 0;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-separate-vertex-format",
        .title = "Separate vertex format and vertex buffer bindings",
        .min_es = {3, 1},
        .summary = "glVertexAttribFormat / IFormat / Binding, glBindVertexBuffer and glVertexBindingDivisor: one binding feeding several "
                   "attributes, bindings re-pointed without touching formats, divisors per binding, the legacy glVertexAttribPointer as a "
                   "shorthand for the same state, the limits and every error.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<SeparateVertexFormat>(argc, argv, info);
}
