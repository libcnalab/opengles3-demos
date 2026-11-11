// es30-uniform-buffer-objects
//
// Uniform blocks backed by buffer objects (ES 3.0 section 2.12.6.2 - 2.12.6.5): the std140 layout rules
// checked against a CPU layout calculator and against what the GL reports, data packed with that calculator
// and verified by the shader, block introspection, binding points (glUniformBlockBinding, glBindBufferBase,
// glBindBufferRange with UNIFORM_BUFFER_OFFSET_ALIGNMENT), one buffer shared by two programs, updates between
// draws, the implementation limits and the defined errors.
//
// The 256x256 surface has two parts.
//
//  TOP (64 px): the VERIFICATION GRID, 16 columns x 4 rows of 16-pixel cells. The fragment shader reads every
//  member of three blocks and compares it with the value the application packed: green = equal, red = not.
//    cells 0-29   block Probe (std140): scalars, vec3 packing, arrays, mat2/3/4/3x2, two row_major matrices,
//                 an array of mat3, a struct, an array of structs
//    cells 32-36  block Mistake (std140) with the CORRECT layout, filled using the CPU calculator
//    cells 38-42  the same block filled with a naive tightly-packed C layout    (cells 41, 42 are magenta)
//    cells 48-52  ... and with every vec3 padded to 16 bytes                      (cells 49-52 are magenta)
//    cells 54-59  block Queried (packed): offsets are implementation-defined, so they are queried and used
//  Magenta is the EXPECTED outcome of the two wrong layouts; red anywhere would be a failure.
//
//  BELOW (32-px cells, 8 per row): the SCENE. Each cell draws with a uniform block "Palette" (color[3], offset,
//  gain, pick: 64 bytes) that comes from a different buffer, binding point or range:
//    row 0  A0 default binding 0 | A1 glUniformBlockBinding -> 3 | A2 another buffer at the same point |
//           A3 a second program on the same buffer | A4/A5 glBufferSubData between draws |
//           A6 update through glMapBufferRange | A7 one buffer at two binding points
//    row 1  B0-B3 glBindBufferRange at four aligned offsets of one buffer | B4 a rejected range leaves the binding
//           intact | B5 glBindBufferBase (the whole buffer) | B6 sub-range update then rebind | B7 second program
//    row 2  C0 an ARRAY of uniform blocks backed by two buffers | C1 ... backed by two ranges of one buffer |
//           C2-C6 glBufferData, single members updated at their queried offsets, glCopyBufferSubData, orphaning

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <set>
#include <initializer_list>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using Bytes = std::vector<std::uint8_t>;

namespace
{
    // =========================================================================================
    // 1. A std140 layout calculator (ES 3.0 section 2.12.6.4), independent of the GL
    // =========================================================================================
    enum class Basic { Float, Int, UInt, Bool };

    struct TypeInfo
    {
        Basic basic = Basic::Float;
        int cols = 1;  ///< matrix columns (1 for scalars and vectors)
        int rows = 1;  ///< vector length, or matrix rows
        [[nodiscard]] bool is_matrix() const { return cols > 1; }
        [[nodiscard]] int components() const { return cols * rows; }
        [[nodiscard]] mgl::UniformType gl_type() const
        {
            using T = mgl::UniformType;
            if (cols == 1)
            {
                static constexpr T f[] = {T::Float, T::FloatVec2, T::FloatVec3, T::FloatVec4};
                static constexpr T i[] = {T::Int, T::IntVec2, T::IntVec3, T::IntVec4};
                static constexpr T u[] = {T::UnsignedInt, T::UnsignedIntVec2, T::UnsignedIntVec3, T::UnsignedIntVec4};
                static constexpr T b[] = {T::Bool, T::BoolVec2, T::BoolVec3, T::BoolVec4};
                const auto idx = static_cast<std::size_t>(rows - 1);
                switch (basic)
                {
                case Basic::Float: return f[idx];
                case Basic::Int: return i[idx];
                case Basic::UInt: return u[idx];
                case Basic::Bool: return b[idx];
                }
            }
            if (cols == 2 && rows == 2) return T::FloatMat2;
            if (cols == 3 && rows == 3) return T::FloatMat3;
            if (cols == 4 && rows == 4) return T::FloatMat4;
            if (cols == 2 && rows == 3) return T::FloatMat2x3;
            if (cols == 2 && rows == 4) return T::FloatMat2x4;
            if (cols == 3 && rows == 2) return T::FloatMat3x2;
            if (cols == 3 && rows == 4) return T::FloatMat3x4;
            if (cols == 4 && rows == 2) return T::FloatMat4x2;
            return T::FloatMat4x3;
        }
    };

    constexpr TypeInfo kFloat{Basic::Float, 1, 1}, kVec2{Basic::Float, 1, 2}, kVec3{Basic::Float, 1, 3}, kVec4{Basic::Float, 1, 4},
        kInt{Basic::Int, 1, 1}, kIVec2{Basic::Int, 1, 2}, kUInt{Basic::UInt, 1, 1}, kBool{Basic::Bool, 1, 1},
        kMat2{Basic::Float, 2, 2}, kMat3{Basic::Float, 3, 3}, kMat4{Basic::Float, 4, 4}, kMat3x2{Basic::Float, 3, 2},
        kMat2x4{Basic::Float, 2, 4}, kMat2x3{Basic::Float, 2, 3};

    /// One declaration inside a block: a basic type, or a structure when `fields` is not empty.
    struct Member
    {
        std::string name;
        TypeInfo type{};
        int array = 0;  ///< 0: not an array
        bool row_major = false;
        bool value_true = true;  ///< bool members only: which value the test data stores
        std::vector<Member> fields;
    };

    Member basic(std::string name, TypeInfo t, int array = 0, bool row_major = false, bool value_true = true)
    {
        return Member{std::move(name), t, array, row_major, value_true, {}};
    }
    Member structure(std::string name, std::vector<Member> fields, int array = 0)
    {
        return Member{std::move(name), {}, array, false, true, std::move(fields)};
    }

    /// One active uniform (a leaf of the declaration tree), as the GL enumerates it.
    struct Leaf
    {
        std::string name;  ///< without the block-name prefix: "lights[1].color", "farr[0]"
        TypeInfo type{};
        int id = 0;        ///< 1-based position in the block: seeds the test values (see verify.frag)
        int count = 1;     ///< array elements (the GL's UNIFORM_SIZE)
        bool value_true = true;
        int offset = 0;
        int array_stride = 0;   ///< 0 when not an array
        int matrix_stride = 0;  ///< 0 when not a matrix
        bool row_major = false;  ///< matrices only
    };

    int align_up(int v, int a) { return (v + a - 1) / a * a; }

    struct Measure
    {
        int align, size, array_stride, matrix_stride;
    };

    /// Rules 1-8: scalars, vectors, matrices and their arrays.
    Measure measure(const Member& m)
    {
        const TypeInfo& t = m.type;
        Measure r{};
        if (t.is_matrix())
        {
            // Rules 5-8: an array of C column vectors (R row vectors when row_major), each one rounded up to a vec4.
            const int vectors = m.row_major ? t.rows : t.cols;
            const int matrix_size = vectors * 16;
            r.align = 16;
            r.matrix_stride = 16;
            r.array_stride = m.array > 0 ? matrix_size : 0;
            r.size = matrix_size * (m.array > 0 ? m.array : 1);
        }
        else if (m.array > 0)
        {
            // Rule 4: the stride is the element's base alignment rounded up to that of a vec4.
            r.align = 16;
            r.array_stride = 16;
            r.size = 16 * m.array;
        }
        else
        {
            r.align = t.rows == 1 ? 4 : t.rows == 2 ? 8 : 16;  // rules 1-3
            r.size = 4 * t.rows;
        }
        return r;
    }

    int layout_fields(const std::vector<Member>& fields, int base, const std::string& prefix, std::vector<Leaf>& out);

    /// Rule 9: the size of a structure is its end offset rounded up to its base alignment (always a multiple of 16).
    int struct_size(const std::vector<Member>& fields)
    {
        std::vector<Leaf> scratch;
        return align_up(layout_fields(fields, 0, "", scratch), 16);
    }

    /// Places `fields` starting at `base`, appends one Leaf per active uniform, returns the end offset.
    int layout_fields(const std::vector<Member>& fields, int base, const std::string& prefix, std::vector<Leaf>& out)
    {
        int offset = base;
        for (const Member& m : fields)
        {
            if (!m.fields.empty())
            {
                const int size = struct_size(m.fields);  // rules 9 and 10: base alignment 16, array stride = size
                const int count = m.array > 0 ? m.array : 1;
                offset = align_up(offset, 16);
                for (int e = 0; e < count; ++e)
                    layout_fields(m.fields, offset + e * size, prefix + m.name + (m.array > 0 ? std::format("[{}]", e) : "") + ".", out);
                offset += count * size;
                continue;
            }
            const Measure ms = measure(m);
            offset = align_up(offset, ms.align);
            Leaf l;
            l.name = prefix + m.name + (m.array > 0 ? "[0]" : "");
            l.type = m.type;
            l.count = m.array > 0 ? m.array : 1;
            l.value_true = m.value_true;
            l.offset = offset;
            l.array_stride = ms.array_stride;
            l.matrix_stride = ms.matrix_stride;
            l.row_major = m.row_major && m.type.is_matrix();
            out.push_back(std::move(l));
            offset += ms.size;
        }
        return offset;
    }

    /// A uniform block as the application declares it.
    struct Block
    {
        std::string name;        ///< the block name (what the GL uses)
        bool instance_name;      ///< members are reported as "<block>.<member>" when the block has an instance name
        std::vector<Member> members;
        std::vector<Leaf> leaves;  ///< filled by compute()
        int size = 0;              ///< minimum buffer size in bytes (the end of the last member)

        void compute()
        {
            leaves.clear();
            size = layout_fields(members, 0, "", leaves);
            for (std::size_t i = 0; i < leaves.size(); ++i) leaves[i].id = static_cast<int>(i) + 1;
        }
        [[nodiscard]] std::string gl_name(const Leaf& l) const { return instance_name ? name + "." + l.name : l.name; }
    };

    /// Writes the test value of every element of one leaf at the offsets/strides stored in the leaf. The value
    /// of logical scalar S of leaf ID is ID*100+S (negated for signed integers), S counting array element by
    /// array element and matrices column by column (GLSL order); see verify.frag.
    void pack_leaf(Bytes& buf, const Leaf& l)
    {
        const int comps = l.type.components();
        for (int e = 0; e < l.count; ++e)
            for (int k = 0; k < comps; ++k)
            {
                int off = l.offset + e * l.array_stride;
                if (l.type.is_matrix())
                {
                    const int c = k / l.type.rows, r = k % l.type.rows;  // logical column and row
                    off += l.row_major ? r * l.matrix_stride + c * 4 : c * l.matrix_stride + r * 4;
                }
                else
                {
                    off += k * 4;
                }
                const int v = l.id * 100 + e * comps + k;
                std::uint32_t bits = 0;
                switch (l.type.basic)
                {
                case Basic::Float: { const float f = static_cast<float>(v); std::memcpy(&bits, &f, 4); break; }
                case Basic::Int: bits = static_cast<std::uint32_t>(-v); break;
                case Basic::UInt: bits = static_cast<std::uint32_t>(v); break;
                case Basic::Bool: bits = l.value_true ? 0xDEADBEEFu : 0u; break;  // any non-zero uint is true
                }
                std::memcpy(buf.data() + off, &bits, 4);
            }
    }

    Bytes pack(const std::vector<Leaf>& leaves, std::size_t size)
    {
        Bytes buf(size, 0);
        for (const Leaf& l : leaves) pack_leaf(buf, l);
        return buf;
    }

    // ---- the declarations (kept in sync with verify.frag; the shader's cells fail if they drift) ------
    Block make_probe()
    {
        Block b{"Probe", true, {}, {}, 0};
        b.members = {
            basic("f", kFloat), basic("v3", kVec3), basic("after_v3", kFloat), basic("v2", kVec2), basic("v3b", kVec3),
            basic("i", kInt), basic("iv2", kIVec2), basic("u", kUInt), basic("flag_true", kBool), basic("flag_false", kBool, 0, false, false),
            basic("v4", kVec4), basic("farr", kFloat, 3), basic("v2arr", kVec2, 2), basic("v3arr", kVec3, 2),
            basic("m2", kMat2), basic("m3", kMat3), basic("m4", kMat4), basic("m3x2", kMat3x2),
            basic("m3r", kMat3, 0, true), basic("m2x4r", kMat2x4, 0, true), basic("m3arr", kMat3, 2),
            structure("inner", {basic("a", kFloat), basic("b", kVec3)}), basic("after_inner", kFloat),
            structure("lights", {basic("position", kVec3), basic("intensity", kFloat), basic("color", kVec4)}, 2),
        };
        b.compute();
        return b;
    }

    Block make_mistake()
    {
        Block b{"Mistake", false, {}, {}, 0};
        b.members = {basic("m_a", kVec3), basic("m_s", kFloat), basic("m_b", kVec3), basic("m_c", kVec3), basic("m_t", kFloat)};
        b.compute();
        return b;
    }

    Block make_queried()
    {
        Block b{"Queried", false, {}, {}, 0};
        b.members = {basic("q_scalar", kFloat), basic("q_vec3", kVec3), basic("q_after", kFloat), basic("q_mat3", kMat3),
                     basic("q_arr", kVec2, 2), basic("q_row", kMat2x3, 0, true)};
        b.compute();  // the std140 offsets computed here are NOT used for this packed block
        return b;
    }

    /// The two wrong ways to lay out the Mistake block, as an application that did not read the spec would.
    std::vector<Leaf> naive_tight(const Block& mistake)
    {
        std::vector<Leaf> leaves = mistake.leaves;  // every member at the next multiple of 4: vec3 = 12 bytes
        int off = 0;
        for (Leaf& l : leaves)
        {
            l.offset = off;
            off += 4 * l.type.rows;
        }
        return leaves;
    }
    std::vector<Leaf> naive_padded(const Block& mistake)
    {
        std::vector<Leaf> leaves = mistake.leaves;  // every member padded to a vec4: a float after a vec3 starts at +16
        int off = 0;
        for (Leaf& l : leaves)
        {
            l.offset = off;
            off += 16;
        }
        return leaves;
    }

    // =========================================================================================
    // 2. Scene data
    // =========================================================================================
    /// The Palette block in std140: color[3] (stride 16), offset, gain, pick: 64 bytes.
    struct PaletteData
    {
        float color[3][4];
        float offset[2];
        float gain;
        std::int32_t pick;
    };
    static_assert(sizeof(PaletteData) == 64 && offsetof(PaletteData, offset) == 48 && offsetof(PaletteData, gain) == 56 &&
                      offsetof(PaletteData, pick) == 60,
                  "PaletteData is the std140 layout of block Palette");

    PaletteData make_palette(std::array<std::array<float, 3>, 3> rgb, float ox, float oy, float gain, int pick)
    {
        PaletteData p{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            p.color[i][0] = rgb[i][0];
            p.color[i][1] = rgb[i][1];
            p.color[i][2] = rgb[i][2];
            p.color[i][3] = 1.0f;
        }
        p.offset[0] = ox;
        p.offset[1] = oy;
        p.gain = gain;
        p.pick = pick;
        return p;
    }

    std::uint8_t to8(float x) { return static_cast<std::uint8_t>(std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)); }

    /// What program 1 / program 2 must write for a palette (ES 3.0 section 4.1.8: float -> 8-bit unorm).
    Rgba8 s1_color(const PaletteData& p)
    {
        const float* c = p.color[p.pick];
        return {to8(c[0] * p.gain), to8(c[1] * p.gain), to8(c[2] * p.gain), 255};
    }
    Rgba8 s2_color(const PaletteData& p)
    {
        const float* c = p.color[p.pick];
        return {to8(1.0f - c[0] * p.gain), to8(1.0f - c[1] * p.gain), to8(1.0f - c[2] * p.gain), 255};
    }

    constexpr Rgba8 kBackground = {14, 16, 22, 255};
    constexpr Rgba8 kCellClear = {20, 30, 50, 255};
    constexpr Rgba8 kGreen = {0, 255, 0, 255}, kMagenta = {255, 0, 255, 255}, kRed = {255, 0, 0, 255};
    constexpr Rgba8 kGap = {26, 26, 31, 255};  // 0.10, 0.10, 0.12

    // verification grid geometry (cells of 16 px, 16 per row, 4 rows at the top)
    constexpr int kVerifyColumns = 16;
    constexpr int kVerifyRows = 4;
    // scene geometry (cells twice the grid cell, 8 per row, 3 rows below the grid)
    constexpr int kSceneColumns = 8;
    constexpr int kSceneCells = 23;
    constexpr int kSceneRows = (kSceneCells + kSceneColumns - 1) / kSceneColumns;
    // The whole layout is kVerifyColumns cells wide and kVerifyRows + 2 * kSceneRows cells tall. The cell size is
    // the largest that fits both dimensions, so the layout never leaves the surface whatever its aspect ratio.
    constexpr int kLayoutRows = kVerifyRows + 2 * kSceneRows;
    // Below 12-pixel grid cells (24-pixel scene cells) the program-1 square (half side 0.35 of a cell's half) is
    // too small for its three interior probes and the corner probes to stay clear of each other's edges.
    constexpr int kMinCell = 12;

    /// Grid cell size in pixels for this surface (scene cells are twice as large).
    int cell_px(const demo::Env& env) { return std::min(env.width / kVerifyColumns, env.height / kLayoutRows); }

    constexpr const char* kSceneLabels[kSceneCells] = {
        "A0 default binding 0", "A1 glUniformBlockBinding", "A2 other buffer, same point", "A3 second program",
        "A4 glBufferSubData", "A5 second program after update", "A6 glMapBufferRange update", "A7 one buffer, two points",
        "B0 range slot 0", "B1 range slot 1", "B2 range slot 2", "B3 range slot 3", "B4 rejected range", "B5 glBindBufferBase",
        "B6 sub-range update", "B7 second program on a range", "C0 block array, two buffers", "C1 block array, two ranges",
        "C2 glBufferData, pick 0", "C3 pick updated at its queried offset", "C4 gain updated at its queried offset",
        "C5 glCopyBufferSubData into a UBO", "C6 glBufferData orphaning"};

    struct PixelProbe
    {
        int x, y;
        Rgba8 want;
    };

    struct Obs
    {
        bool ok;
        std::string what;
        std::string detail;
        std::string deviation_renderer;  ///< non-empty: a documented driver deviation (Check::expect_or_known_deviation)
        std::string deviation_reason;
    };

    /// The first of `names` that GL_RENDERER contains, or "" when the current driver is none of them.
    std::string matching_renderer(std::initializer_list<const char*> names)
    {
        const GLubyte* r = mgl::glGetString(mgl::StringName::Renderer);
        const std::string_view renderer = r != nullptr ? reinterpret_cast<const char*>(r) : "";
        for (const char* n : names)
            if (renderer.find(n) != std::string_view::npos) return n;
        return {};
    }

    // =========================================================================================
    // 3. The demo
    // =========================================================================================
    class UniformBufferObjects final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (cell_px(env) < kMinCell)
                throw demo::Skip(std::format("surface {}x{} is too small: the layout needs at least {}x{} pixels ({}-pixel grid cells)", env.width,
                                             env.height, kVerifyColumns * kMinCell, kLayoutRows * kMinCell, kMinCell));
            verify_ = demo::build_program(env, "shaders/verify.vert", "shaders/verify.frag");
            scene1_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            scene2_ = demo::build_program(env, "shaders/scene_flat.vert", "shaders/scene_inverse.frag");
            scene3_ = demo::build_program(env, "shaders/scene_flat.vert", "shaders/tint.frag");

            u_cell_ = locate(verify_, "u_cell");
            u_height_ = locate(verify_, "u_height");

            probe_ = make_probe();
            mistake_ = make_mistake();
            queried_ = make_queried();

            alignment_ = demo::get_integer(mgl::GetParameter::UniformBufferOffsetAlignment);
            max_bindings_ = demo::get_integer(mgl::GetParameter::MaxUniformBufferBindings);

            // Block indices. A block's index is not specified; only its name is.
            probe_index_ = block_index(verify_, "Probe");
            mistake_index_ = block_index(verify_, "Mistake");
            queried_index_ = block_index(verify_, "Queried");
            palette1_index_ = block_index(scene1_, "Palette");
            palette2_index_ = block_index(scene2_, "Palette");
            tint0_index_ = block_index(scene3_, "Tint[0]");
            tint1_index_ = block_index(scene3_, "Tint[1]");

            // ES 3.0 section 2.12.6.5 and Table 6.21: a linked program has every active block bound to point 0.
            std::string nonzero;
            for (auto [prog, idx, name] : {std::tuple{verify_, probe_index_, "Probe"}, {verify_, mistake_index_, "Mistake"},
                                           {verify_, queried_index_, "Queried"}, {scene1_, palette1_index_, "Palette (program 1)"},
                                           {scene2_, palette2_index_, "Palette (program 2)"}, {scene3_, tint0_index_, "Tint[0]"},
                                           {scene3_, tint1_index_, "Tint[1]"}})
                if (block_param(prog, idx, mgl::UniformBlockParameter::Binding) != 0) nonzero += std::string(name) + " ";
            init_obs_.push_back({nonzero.empty(), "after linking, UNIFORM_BLOCK_BINDING of every block is 0 (ES 3.0 section 2.12.6.5)", nonzero});

            {
                const Queried q = query_uniforms(scene1_, {"Palette.gain", "Palette.pick"});
                gain_offset_ = q.offset[0];
                pick_offset_ = q.offset[1];
            }

            // Static buffers: the block data never changes between frames.
            mgl::glGenBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());

            // The queried (packed) block: ask the GL where everything is, then pack with ITS offsets.
            queried_gl_ = query_layout(verify_, queried_, queried_index_);
            queried_gl_size_ = block_param(verify_, queried_index_, mgl::UniformBlockParameter::DataSize);

            upload(kBufProbe, pack(probe_.leaves, static_cast<std::size_t>(probe_.size)));
            const std::size_t mistake_bytes = static_cast<std::size_t>(std::max(mistake_.size, 80));
            upload(kBufMistakeGood, pack(mistake_.leaves, mistake_bytes));
            upload(kBufMistakeTight, pack(naive_tight(mistake_), mistake_bytes));
            upload(kBufMistakePadded, pack(naive_padded(mistake_), mistake_bytes));
            upload(kBufQueried, pack(queried_gl_, static_cast<std::size_t>(queried_gl_size_)));

            // Scene buffers. The three palette buffers are rewritten every frame; the ring has 4 slots at
            // multiples of UNIFORM_BUFFER_OFFSET_ALIGNMENT, so that each slot may be bound with glBindBufferRange.
            stride_ = align_up(static_cast<int>(sizeof(PaletteData)), std::max(alignment_, 1));
            for (std::size_t i = 0; i < 3; ++i) allocate(kBufPalette0 + i, sizeof(PaletteData));
            allocate(kBufRing, static_cast<std::size_t>(stride_) * 4);
            allocate(kBufTint0, 16);
            allocate(kBufTint1, 16);
            allocate(kBufTintRing, static_cast<std::size_t>(std::max(alignment_, 16)) + 16);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            obs_.clear();
            for (auto& p : probes_) p.clear();

            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            clear_color(kBackground);
            mgl::glClear(mgl::ClearBufferBit::Color);

            draw_verification_grid(env);
            for (int k = 0; k < kSceneCells; ++k) clear_scene_cell(env, k);
            scene_binding_cells(env);
            scene_range_cells(env);
            scene_extra_cells(env);

            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, mgl::BufferId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            for (const Obs& o : init_obs_) check.expect(o.ok, o.what, o.detail);
            for (const Obs& o : obs_)
            {
                if (o.deviation_renderer.empty()) check.expect(o.ok, o.what, o.detail);
                else check.expect_or_known_deviation(o.ok, o.what, o.detail, o.deviation_renderer, o.deviation_reason);
            }

            check_verification_grid(env, check);
            check_scene_cells(check);
            check_probe_introspection(check);
            check_mistake_and_queried(check);
            check_program_blocks(check);
            check_errors(check);
            check_limits(check);
            negative_controls(env, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(max_bindings_); ++i)
                if (i < 16) mgl::glBindBufferBase(mgl::BufferTarget::Uniform, i, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Uniform, mgl::BufferId{0});
            mgl::glDeleteBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glDeleteProgram(verify_);
            mgl::glDeleteProgram(scene1_);
            mgl::glDeleteProgram(scene2_);
            mgl::glDeleteProgram(scene3_);
        }

    private:
        static constexpr auto kUniform = mgl::BufferTarget::Uniform;

        // Buffer slots.
        static constexpr std::size_t kBufProbe = 0, kBufMistakeGood = 1, kBufMistakeTight = 2, kBufMistakePadded = 3, kBufQueried = 4,
                                     kBufPalette0 = 5 /* 5,6,7 */, kBufRing = 8, kBufTint0 = 9, kBufTint1 = 10, kBufTintRing = 11;
        // Binding points used by the verification program (kept away from the scene's points 0, 3, 4, 6, 12, 13).
        static constexpr GLuint kPointProbe = 8, kPointMistake = 9, kPointQueried = 10;

        // ----------------------------------------------------------------------------------
        // helpers
        // ----------------------------------------------------------------------------------
        static mgl::UniformLocation locate(mgl::ProgramId p, const char* name)
        {
            const mgl::UniformLocation l = mgl::glGetUniformLocation(p, name);
            if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
            return l;
        }

        static GLuint block_index(mgl::ProgramId p, const char* name)
        {
            const GLuint i = mgl::glGetUniformBlockIndex(p, name);
            if (i == GL_INVALID_INDEX) throw demo::Failure(std::format("uniform block '{}' is not active", name));
            return i;
        }

        static GLint block_param(mgl::ProgramId p, GLuint index, mgl::UniformBlockParameter pname)
        {
            GLint v = -1;
            mgl::glGetActiveUniformBlockiv(p, index, pname, &v);
            return v;
        }

        void record(bool ok, std::string what, std::string detail = {}, std::string deviation_renderer = {}, std::string deviation_reason = {})
        {
            obs_.push_back({ok, std::move(what), std::move(detail), std::move(deviation_renderer), std::move(deviation_reason)});
        }

        void record_error(mgl::ErrorCode expected, std::string what)
        {
            const mgl::ErrorCode e = demo::drain_gl_errors();
            record(e == expected, std::move(what),
                   e == expected ? std::string{} : std::format("glGetError() = {}, expected {}", demo::error_name(e), demo::error_name(expected)));
        }

        static void clear_color(Rgba8 c) { mgl::glClearColor(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, c[3] / 255.0f); }

        void allocate(std::size_t slot, std::size_t bytes)
        {
            const Bytes zeros(bytes, 0);
            mgl::glBindBuffer(kUniform, buf_[slot]);
            mgl::glBufferData(kUniform, static_cast<GLsizeiptr>(bytes), zeros.data(), mgl::BufferUsage::DynamicDraw);
        }

        void upload(std::size_t slot, const Bytes& data)
        {
            mgl::glBindBuffer(kUniform, buf_[slot]);
            mgl::glBufferData(kUniform, static_cast<GLsizeiptr>(data.size()), data.data(), mgl::BufferUsage::StaticDraw);
        }

        /// Queries the layout of an implementation-laid-out block with glGetUniformIndices + glGetActiveUniformsiv.
        std::vector<Leaf> query_layout(mgl::ProgramId prog, const Block& block, GLuint block_idx) const
        {
            std::vector<Leaf> leaves = block.leaves;
            std::vector<std::string> names;
            for (const Leaf& l : leaves) names.push_back(block.gl_name(l));
            const Queried q = query_uniforms(prog, names);
            for (std::size_t i = 0; i < leaves.size(); ++i)
            {
                if (q.index[i] == GL_INVALID_INDEX || static_cast<GLuint>(q.block[i]) != block_idx)
                    throw demo::Failure(std::format("uniform '{}' is not an active member of block '{}'", names[i], block.name));
                leaves[i].offset = q.offset[i];
                leaves[i].array_stride = q.array_stride[i];
                leaves[i].matrix_stride = q.matrix_stride[i];
                leaves[i].row_major = q.row_major[i] != 0;
            }
            return leaves;
        }

        struct Queried
        {
            std::vector<GLuint> index;
            std::vector<GLint> block, offset, array_stride, matrix_stride, row_major, type, size;
        };

        static Queried query_uniforms(mgl::ProgramId prog, const std::vector<std::string>& names)
        {
            Queried q;
            const auto n = names.size();
            std::vector<const GLchar*> ptrs;
            for (const std::string& s : names) ptrs.push_back(s.c_str());
            q.index.assign(n, GL_INVALID_INDEX);
            mgl::glGetUniformIndices(prog, static_cast<GLsizei>(n), ptrs.data(), q.index.data());
            auto get = [&](mgl::UniformParameter pname, std::vector<GLint>& out) {
                out.assign(n, -12345);
                mgl::glGetActiveUniformsiv(prog, static_cast<GLsizei>(n), q.index.data(), pname, out.data());
            };
            get(mgl::UniformParameter::BlockIndex, q.block);
            get(mgl::UniformParameter::Offset, q.offset);
            get(mgl::UniformParameter::ArrayStride, q.array_stride);
            get(mgl::UniformParameter::MatrixStride, q.matrix_stride);
            get(mgl::UniformParameter::IsRowMajor, q.row_major);
            get(mgl::UniformParameter::Type, q.type);
            get(mgl::UniformParameter::Size, q.size);
            return q;
        }

        // ----------------------------------------------------------------------------------
        // verification grid
        // ----------------------------------------------------------------------------------
        void draw_verification_grid(const demo::Env& env)
        {
            const int vc = cell_px(env);
            mgl::glUseProgram(verify_);
            mgl::glUniform1i(u_cell_, vc);
            mgl::glUniform1i(u_height_, env.height);
            // Exactly 16 cells wide: a wider viewport would let column 16 alias the next row's cell 0.
            mgl::glViewport(0, env.height - kVerifyRows * vc, kVerifyColumns * vc, kVerifyRows * vc);

            // Bind each block to its own binding point (glUniformBlockBinding) and a buffer to each point.
            mgl::glUniformBlockBinding(verify_, probe_index_, kPointProbe);
            mgl::glUniformBlockBinding(verify_, mistake_index_, kPointMistake);
            mgl::glUniformBlockBinding(verify_, queried_index_, kPointQueried);
            mgl::glBindBufferBase(kUniform, kPointProbe, buf_[kBufProbe]);
            mgl::glBindBufferBase(kUniform, kPointQueried, buf_[kBufQueried]);
            mgl::glBindBufferBase(kUniform, kPointMistake, buf_[kBufMistakeGood]);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            // The two panels with a WRONG layout: same program, another buffer behind the Mistake block, drawn
            // into just their own cells with the scissor test.
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (const auto& [buffer, first_col, row] : {std::tuple{kBufMistakeTight, 6, 2}, {kBufMistakePadded, 0, 3}})
            {
                mgl::glBindBufferBase(kUniform, kPointMistake, buf_[buffer]);
                mgl::glScissor(first_col * vc, env.height - (row + 1) * vc, 5 * vc, vc);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
            mgl::glBindBufferBase(kUniform, kPointMistake, buf_[kBufMistakeGood]);
        }

        /// Expected color of verification cell `cell` (reading order, 16 per row).
        static Rgba8 expected_cell(int cell)
        {
            if (cell < 30) return kGreen;
            if (cell >= 32 && cell < 37) return kGreen;
            if (cell >= 38 && cell < 43) return cell - 38 < 3 ? kGreen : kMagenta;  // tight C layout: a, s, b still agree
            if (cell >= 48 && cell < 53) return cell == 48 ? kGreen : kMagenta;      // padded: only the first vec3 agrees
            if (cell >= 54 && cell < 60) return kGreen;
            return kGap;
        }

        static std::pair<int, int> verify_cell_center(const demo::Env& env, int cell)
        {
            const int vc = cell_px(env);
            return {(cell % kVerifyColumns) * vc + vc / 2, env.height - (cell / kVerifyColumns + 1) * vc + vc / 2};
        }

        static bool near(Rgba8 got, Rgba8 want, int tol)
        {
            for (std::size_t i = 0; i < 4; ++i)
                if (std::abs(int{got[i]} - int{want[i]}) > tol) return false;
            return true;
        }

        std::string verify_group(const demo::Env& env, int first, int count, const Block* block, int leaf_offset) const
        {
            std::string bad;
            for (int cell = first; cell < first + count; ++cell)
            {
                const auto [x, y] = verify_cell_center(env, cell);
                const Rgba8 got = demo::read_pixel(x, y);
                if (!near(got, expected_cell(cell), 1))
                {
                    const std::string what = block != nullptr ? block->leaves[static_cast<std::size_t>(cell - first + leaf_offset)].name : std::to_string(cell);
                    bad += std::format("{} (cell {}: {}) ", what, cell, demo::format_rgba(got));
                }
            }
            return bad;
        }

        void check_verification_grid(const demo::Env& env, demo::Check& check)
        {
            // Pixels, not GL calls, decide here: a green cell means the shader read exactly the value that was packed.
            std::string bad = verify_group(env, 0, 30, &probe_, 0);
            check.expect(bad.empty(), "block Probe (std140): all 30 members read by the shader equal the values packed with the CPU std140 layout (30 green cells)", bad);
            bad = verify_group(env, 32, 5, &mistake_, 0);
            check.expect(bad.empty(), "block Mistake packed with the correct std140 layout: all 5 members green", bad);
            bad = verify_group(env, 38, 5, &mistake_, 0);
            check.expect(bad.empty(), "block Mistake packed tightly like a C struct: a, s, b (unchanged offsets) green, c and t magenta (wrong values)", bad);
            bad = verify_group(env, 48, 5, &mistake_, 0);
            check.expect(bad.empty(), "block Mistake with every vec3 padded to 16 bytes: only a green, s, b, c, t magenta (wrong values)", bad);
            bad = verify_group(env, 54, 6, &queried_, 0);
            check.expect(bad.empty(), "block Queried (packed): data packed at the QUERIED offsets and strides reads back correctly, 6 green cells", bad);
            // The unused cells and the gaps must be the dark gap color: nothing leaked into them.
            std::string stray;
            for (int cell : {30, 31, 37, 43, 47, 53, 60, 63})
            {
                const auto [x, y] = verify_cell_center(env, cell);
                const Rgba8 got = demo::read_pixel(x, y);
                if (!near(got, kGap, 1)) stray += std::format("cell {} is {} ", cell, demo::format_rgba(got));
            }
            check.expect(stray.empty(), "cells that test nothing keep the neutral gap color", stray);
        }

        // ----------------------------------------------------------------------------------
        // scene helpers
        // ----------------------------------------------------------------------------------
        static int scene_top(const demo::Env& env) { return env.height - kVerifyRows * cell_px(env); }
        static int scene_px(const demo::Env& env) { return 2 * cell_px(env); }

        struct CellRect
        {
            int x, y, size;
        };
        static CellRect scene_rect(const demo::Env& env, int k)
        {
            const int sc = scene_px(env);
            return {(k % kSceneColumns) * sc, scene_top(env) - (k / kSceneColumns + 1) * sc, sc};
        }

        void clear_scene_cell(const demo::Env& env, int k)
        {
            const CellRect r = scene_rect(env, k);
            mgl::glEnable(mgl::Capability::ScissorTest);
            mgl::glScissor(r.x + 1, r.y + 1, r.size - 2, r.size - 2);
            clear_color(kCellClear);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        void cell_viewport(const demo::Env& env, int k)
        {
            const CellRect r = scene_rect(env, k);
            mgl::glViewport(r.x, r.y, r.size, r.size);
        }

        void draw_square(const demo::Env& env, int k, mgl::ProgramId program)
        {
            cell_viewport(env, k);
            mgl::glUseProgram(program);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        void write_palette(std::size_t slot, const PaletteData& p)
        {
            mgl::glBindBuffer(kUniform, buf_[slot]);
            mgl::glBufferSubData(kUniform, 0, sizeof(PaletteData), &p);
        }

        /// A program-1 cell: a square (half side 0.35 NDC) centered at the palette's offset, colored by it.
        void expect_s1(const demo::Env& env, int k, const PaletteData& p)
        {
            const CellRect r = scene_rect(env, k);
            const float cx = static_cast<float>(r.x) + (p.offset[0] + 1.0f) * 0.5f * static_cast<float>(r.size);
            const float cy = static_cast<float>(r.y) + (p.offset[1] + 1.0f) * 0.5f * static_cast<float>(r.size);
            const Rgba8 want = s1_color(p);
            const int ix = static_cast<int>(cx), iy = static_cast<int>(cy);
            probes_[static_cast<std::size_t>(k)].push_back({ix, iy, want});
            probes_[static_cast<std::size_t>(k)].push_back({ix - 2, iy + 2, want});
            probes_[static_cast<std::size_t>(k)].push_back({ix + 2, iy - 2, want});
            // Background probes in the two cell corners the square does not reach (half side 0.35 * size/2 + margin).
            const float half = 0.35f * 0.5f * static_cast<float>(r.size) + 2.0f;
            for (const auto& [bx, by] : {std::pair{r.x + 3, r.y + 3}, std::pair{r.x + r.size - 4, r.y + r.size - 4},
                                         std::pair{r.x + 3, r.y + r.size - 4}, std::pair{r.x + r.size - 4, r.y + 3}})
                if (std::abs(static_cast<float>(bx) - cx) > half || std::abs(static_cast<float>(by) - cy) > half)
                    probes_[static_cast<std::size_t>(k)].push_back({bx, by, kCellClear});
        }

        /// A program-2 cell: the whole cell (+-0.9) in the inverse color.
        void expect_s2(const demo::Env& env, int k, const PaletteData& p, bool corners = true)
        {
            const CellRect r = scene_rect(env, k);
            const Rgba8 want = s2_color(p);
            probes_[static_cast<std::size_t>(k)].push_back({r.x + r.size / 2, r.y + r.size / 2, want});
            if (corners)
                for (const auto& [bx, by] : {std::pair{r.x + 5, r.y + 5}, std::pair{r.x + r.size - 6, r.y + r.size - 6}})
                    probes_[static_cast<std::size_t>(k)].push_back({bx, by, want});
        }

        void expect_halves(const demo::Env& env, int k, Rgba8 left, Rgba8 right)
        {
            const CellRect r = scene_rect(env, k);
            probes_[static_cast<std::size_t>(k)].push_back({r.x + r.size / 4, r.y + r.size / 2, left});
            probes_[static_cast<std::size_t>(k)].push_back({r.x + r.size / 8, r.y + r.size / 4, left});
            probes_[static_cast<std::size_t>(k)].push_back({r.x + 3 * r.size / 4, r.y + r.size / 2, right});
            probes_[static_cast<std::size_t>(k)].push_back({r.x + 7 * r.size / 8, r.y + 3 * r.size / 4, right});
        }

        std::int64_t indexed64(mgl::GetParameter p, GLuint index) const
        {
            GLint64 v = -1;
            mgl::glGetInteger64i_v(p, index, &v);
            return v;
        }
        GLint indexed32(mgl::GetParameter p, GLuint index) const
        {
            GLint v = -1;
            mgl::glGetIntegeri_v(p, index, &v);
            return v;
        }

        void record_binding(GLuint point, std::size_t slot, std::int64_t start, std::int64_t size, const std::string& what)
        {
            const bool ok = indexed32(mgl::GetParameter::UniformBufferBinding, point) == static_cast<GLint>(buf_[slot].value) &&
                            indexed64(mgl::GetParameter::UniformBufferStart, point) == start &&
                            indexed64(mgl::GetParameter::UniformBufferSize, point) == size &&
                            indexed32(mgl::GetParameter::UniformBufferStart, point) == start && indexed32(mgl::GetParameter::UniformBufferSize, point) == size;
            record(ok, what,
                   std::format("binding {} start {} size {} (via GetIntegeri_v: start {}, size {})",
                               indexed32(mgl::GetParameter::UniformBufferBinding, point), indexed64(mgl::GetParameter::UniformBufferStart, point),
                               indexed64(mgl::GetParameter::UniformBufferSize, point), indexed32(mgl::GetParameter::UniformBufferStart, point),
                               indexed32(mgl::GetParameter::UniformBufferSize, point)));
        }

        // ----------------------------------------------------------------------------------
        // row A: binding points and updates
        // ----------------------------------------------------------------------------------
        void scene_binding_cells(const demo::Env& env)
        {
            const PaletteData pa = make_palette({{{0.9f, 0.2f, 0.2f}, {0.2f, 0.8f, 0.3f}, {0.2f, 0.3f, 0.9f}}}, -0.30f, 0.30f, 1.0f, 0);
            const PaletteData pb = make_palette({{{0.9f, 0.8f, 0.2f}, {0.2f, 0.8f, 0.8f}, {0.8f, 0.3f, 0.8f}}}, 0.30f, 0.30f, 1.0f, 2);
            const PaletteData pc = make_palette({{{0.95f, 0.55f, 0.15f}, {0.3f, 0.9f, 0.5f}, {0.4f, 0.4f, 0.9f}}}, 0.0f, -0.30f, 0.75f, 1);
            const PaletteData pc2 = make_palette({{{0.95f, 0.55f, 0.15f}, {0.3f, 0.9f, 0.5f}, {0.4f, 0.4f, 0.9f}}}, -0.30f, -0.30f, 1.0f, 0);
            for (std::size_t i = 0; i < 3; ++i) write_palette(kBufPalette0 + i, i == 0 ? pa : i == 1 ? pb : pc);

            // A0: block binding 0 is what a freshly linked program has (verified in init); the buffer sits at point 0.
            mgl::glUniformBlockBinding(scene1_, palette1_index_, 0);
            mgl::glBindBufferBase(kUniform, 0, buf_[kBufPalette0]);
            draw_square(env, 0, scene1_);
            expect_s1(env, 0, pa);
            record(block_param(scene1_, palette1_index_, mgl::UniformBlockParameter::Binding) == 0, "A0: UNIFORM_BLOCK_BINDING of program 1's Palette reads back 0");
            record_binding(0, kBufPalette0, 0, 0, "A0: glBindBufferBase(0, buffer): UNIFORM_BUFFER_BINDING = buffer, START and SIZE 0 (nothing specified, sec 6.1.9)");

            // A1: glUniformBlockBinding moves the block to point 3, where another buffer is bound.
            mgl::glUniformBlockBinding(scene1_, palette1_index_, 3);
            mgl::glBindBufferBase(kUniform, 3, buf_[kBufPalette0 + 1]);
            draw_square(env, 1, scene1_);
            expect_s1(env, 1, pb);
            record(block_param(scene1_, palette1_index_, mgl::UniformBlockParameter::Binding) == 3, "A1: after glUniformBlockBinding(program, block, 3) UNIFORM_BLOCK_BINDING reads back 3");
            record_binding(3, kBufPalette0 + 1, 0, 0, "A1: GetIntegeri_v(UNIFORM_BUFFER_BINDING, 3) names the buffer bound with glBindBufferBase");

            // A2: another buffer at the same point; the program is not touched.
            mgl::glBindBufferBase(kUniform, 3, buf_[kBufPalette0 + 2]);
            draw_square(env, 2, scene1_);
            expect_s1(env, 2, pc);

            // A3: a second program whose block is bound to point 3 reads the same buffer.
            mgl::glUniformBlockBinding(scene2_, palette2_index_, 3);
            draw_square(env, 3, scene2_);
            expect_s2(env, 3, pc);

            // A4/A5: change the buffer between draws; both programs see the new data, the earlier cells do not change.
            mgl::glBindBuffer(kUniform, buf_[kBufPalette0 + 2]);
            mgl::glBufferSubData(kUniform, 0, sizeof(PaletteData), &pc2);
            draw_square(env, 4, scene1_);
            expect_s1(env, 4, pc2);
            draw_square(env, 5, scene2_);
            expect_s2(env, 5, pc2);

            // A6: the same through a mapping (ES 3.0 section 2.12.6: BufferData, BufferSubData, MapBufferRange).
            // Only the last 16 bytes (offset, gain, pick) change.
            const PaletteData pc3 = [&] {
                PaletteData p = pc2;
                p.offset[0] = 0.30f;
                p.offset[1] = -0.25f;
                p.gain = 0.5f;
                p.pick = 2;
                return p;
            }();
            void* mapped = mgl::glMapBufferRange(kUniform, 48, 16, mgl::MapBufferAccessMask::Write | mgl::MapBufferAccessMask::InvalidateRange);
            record(mapped != nullptr, "A6: glMapBufferRange on a UNIFORM_BUFFER binding returns a pointer");
            if (mapped != nullptr) std::memcpy(mapped, reinterpret_cast<const std::uint8_t*>(&pc3) + 48, 16);
            record(mgl::glUnmapBuffer(kUniform) == GL_TRUE, "A6: glUnmapBuffer returns TRUE");
            draw_square(env, 6, scene1_);
            expect_s1(env, 6, pc3);

            // A7: one buffer at two binding points: program 2 (point 6) draws the background, program 1 (point 3) the square.
            mgl::glBindBufferBase(kUniform, 6, buf_[kBufPalette0 + 2]);
            mgl::glUniformBlockBinding(scene2_, palette2_index_, 6);
            draw_square(env, 7, scene2_);
            draw_square(env, 7, scene1_);
            {
                const CellRect r = scene_rect(env, 7);
                const Rgba8 bg = s2_color(pc3);
                probes_[7].push_back({r.x + 5, r.y + 5, bg});                          // program 2's background, away from the square
                probes_[7].push_back({r.x + r.size - 6, r.y + r.size - 6, bg});
            }
            expect_s1(env, 7, pc3);  // program 1's square on top (its own corner probes are skipped: the cell is not clear)
            probes_[7].erase(std::remove_if(probes_[7].begin(), probes_[7].end(), [](const PixelProbe& p) { return p.want == kCellClear; }), probes_[7].end());
            record_binding(6, kBufPalette0 + 2, 0, 0, "A7: the same buffer is also bound at point 6 (UNIFORM_BUFFER_BINDING of points 3 and 6 agree)");
            record(indexed32(mgl::GetParameter::UniformBufferBinding, 3) == indexed32(mgl::GetParameter::UniformBufferBinding, 6),
                   "A7: points 3 and 6 name the same buffer");
            record_error(mgl::ErrorCode::NoError, "row A: no GL error from any binding, update or draw");
        }

        // ----------------------------------------------------------------------------------
        // row B: glBindBufferRange
        // ----------------------------------------------------------------------------------
        void scene_range_cells(const demo::Env& env)
        {
            ring_slots_ = {
                make_palette({{{0.95f, 0.55f, 0.15f}, {0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f}}}, -0.30f, 0.30f, 1.0f, 0),
                make_palette({{{0.1f, 0.1f, 0.1f}, {0.2f, 0.75f, 0.7f}, {0.1f, 0.1f, 0.1f}}}, 0.30f, 0.30f, 1.0f, 1),
                make_palette({{{0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f}, {0.6f, 0.3f, 0.9f}}}, -0.30f, -0.30f, 1.0f, 2),
                make_palette({{{0.9f, 0.85f, 0.2f}, {0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f}}}, 0.30f, -0.30f, 0.8f, 0)};
            const auto& slot = ring_slots_;
            mgl::glBindBuffer(kUniform, buf_[kBufRing]);
            for (std::size_t i = 0; i < 4; ++i) mgl::glBufferSubData(kUniform, static_cast<GLintptr>(i) * stride_, sizeof(PaletteData), &slot[i]);

            mgl::glUniformBlockBinding(scene1_, palette1_index_, 4);
            mgl::glUniformBlockBinding(scene2_, palette2_index_, 4);
            for (std::size_t i = 0; i < 4; ++i)
            {
                // The range starts at i * UNIFORM_BUFFER_OFFSET_ALIGNMENT-aligned offsets (stride_ is a multiple of it).
                mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], static_cast<GLintptr>(i) * stride_, sizeof(PaletteData));
                draw_square(env, 8 + static_cast<int>(i), scene1_);
                expect_s1(env, 8 + static_cast<int>(i), slot[i]);
                record_binding(4, kBufRing, static_cast<std::int64_t>(i) * stride_, sizeof(PaletteData),
                               std::format("B{}: glBindBufferRange(point 4, offset {}, size 64): BINDING, START and SIZE read back", i, static_cast<std::int64_t>(i) * stride_));
            }
            record_error(mgl::ErrorCode::NoError, "B0-B3: four aligned glBindBufferRange calls and their draws raise no error");

            // B4: rejected ranges leave the binding of slot 3 untouched.
            if (alignment_ > 1)
            {
                mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], 1, sizeof(PaletteData));
                record_error(mgl::ErrorCode::InvalidValue, std::format("B4: glBindBufferRange with offset 1, not a multiple of UNIFORM_BUFFER_OFFSET_ALIGNMENT ({}) -> INVALID_VALUE", alignment_));
                mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], alignment_ / 2, sizeof(PaletteData));
                record_error(mgl::ErrorCode::InvalidValue, std::format("B4: glBindBufferRange with offset {} (half the alignment) -> INVALID_VALUE", alignment_ / 2));
            }
            mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], 0, 0);
            record_error(mgl::ErrorCode::InvalidValue, "B4: glBindBufferRange with size 0 and a non-zero buffer -> INVALID_VALUE");
            mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], -static_cast<GLintptr>(stride_), sizeof(PaletteData));
            {
                // "INVALID_VALUE if buffer is not zero and offset is negative" (ES 3.0 section 2.10.1.1). Mesa's llvmpipe,
                // radeonsi and zink accept a negative offset that is a multiple of the alignment (and then read outside the
                // buffer when it is used: a draw with that binding crashes), so a strict check here is a documented deviation
                // and the valid binding is restored before anything is drawn.
                const mgl::ErrorCode e = demo::drain_gl_errors();
                record(e == mgl::ErrorCode::InvalidValue, "B4: glBindBufferRange with a negative offset -> INVALID_VALUE",
                       std::format("glGetError() = {}", demo::error_name(e)), matching_renderer({"zink", "radeonsi", "llvmpipe"}),
                       "Mesa does not validate a negative offset in glBindBufferRange(GL_UNIFORM_BUFFER) when it is a multiple of "
                       "UNIFORM_BUFFER_OFFSET_ALIGNMENT: no error is raised and the binding becomes invalid. ES 3.0 section 2.10.1.1 requires INVALID_VALUE");
                if (e != mgl::ErrorCode::InvalidValue)
                    mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], 3 * static_cast<GLintptr>(stride_), sizeof(PaletteData));
            }
            mgl::glBindBufferRange(kUniform, static_cast<GLuint>(max_bindings_), buf_[kBufRing], 0, sizeof(PaletteData));
            record_error(mgl::ErrorCode::InvalidValue, "B4: glBindBufferRange with index = MAX_UNIFORM_BUFFER_BINDINGS -> INVALID_VALUE");
            mgl::glBindBufferBase(kUniform, static_cast<GLuint>(max_bindings_), buf_[kBufRing]);
            record_error(mgl::ErrorCode::InvalidValue, "B4: glBindBufferBase with index = MAX_UNIFORM_BUFFER_BINDINGS -> INVALID_VALUE");
            record_binding(4, kBufRing, 3 * static_cast<std::int64_t>(stride_), sizeof(PaletteData), "B4: the rejected calls left point 4 bound to slot 3 (BINDING, START, SIZE unchanged)");
            draw_square(env, 12, scene1_);
            expect_s1(env, 12, slot[3]);

            // B5: glBindBufferBase binds the whole buffer from offset 0: slot 0 is read; START and SIZE are 0.
            mgl::glBindBufferBase(kUniform, 4, buf_[kBufRing]);
            draw_square(env, 13, scene1_);
            expect_s1(env, 13, slot[0]);
            record_binding(4, kBufRing, 0, 0, "B5: after glBindBufferBase, UNIFORM_BUFFER_START and _SIZE are 0 (the previous range is forgotten)");

            // B6: rewrite slot 1 and bind it again as a range.
            const PaletteData slot1b = make_palette({{{0.1f, 0.1f, 0.1f}, {0.6f, 0.9f, 0.2f}, {0.1f, 0.1f, 0.1f}}}, 0.0f, 0.30f, 1.0f, 1);
            mgl::glBindBuffer(kUniform, buf_[kBufRing]);
            mgl::glBufferSubData(kUniform, stride_, sizeof(PaletteData), &slot1b);
            mgl::glBindBufferRange(kUniform, 4, buf_[kBufRing], stride_, sizeof(PaletteData));
            draw_square(env, 14, scene1_);
            expect_s1(env, 14, slot1b);

            // B7: program 2 on the same point reads the same range.
            draw_square(env, 15, scene2_);
            expect_s2(env, 15, slot1b);
            record_error(mgl::ErrorCode::NoError, "B5-B7: no GL error");
        }

        // ----------------------------------------------------------------------------------
        // row C: an array of uniform blocks, members updated at queried offsets, copies, re-specification
        // ----------------------------------------------------------------------------------
        void scene_extra_cells(const demo::Env& env)
        {
            const std::array<float, 4> ta = {0.9f, 0.4f, 0.1f, 1.0f}, tb = {0.1f, 0.5f, 0.9f, 1.0f}, tc = {0.2f, 0.8f, 0.3f, 1.0f}, td = {0.8f, 0.8f, 0.2f, 1.0f};
            auto rgba = [](const std::array<float, 4>& c) { return Rgba8{to8(c[0]), to8(c[1]), to8(c[2]), to8(c[3])}; };

            mgl::glUniformBlockBinding(scene3_, tint0_index_, 12);
            mgl::glUniformBlockBinding(scene3_, tint1_index_, 13);

            // C0: two instances, two buffers.
            mgl::glBindBuffer(kUniform, buf_[kBufTint0]);
            mgl::glBufferSubData(kUniform, 0, 16, ta.data());
            mgl::glBindBuffer(kUniform, buf_[kBufTint1]);
            mgl::glBufferSubData(kUniform, 0, 16, tb.data());
            mgl::glBindBufferBase(kUniform, 12, buf_[kBufTint0]);
            mgl::glBindBufferBase(kUniform, 13, buf_[kBufTint1]);
            draw_square(env, 16, scene3_);
            expect_halves(env, 16, rgba(ta), rgba(tb));

            // C1: two instances, two ranges of ONE buffer: [0,16) and [alignment, alignment+16).
            const GLintptr second = std::max(alignment_, 16);
            mgl::glBindBuffer(kUniform, buf_[kBufTintRing]);
            mgl::glBufferSubData(kUniform, 0, 16, tc.data());
            mgl::glBufferSubData(kUniform, second, 16, td.data());
            mgl::glBindBufferRange(kUniform, 12, buf_[kBufTintRing], 0, 16);
            mgl::glBindBufferRange(kUniform, 13, buf_[kBufTintRing], second, 16);
            draw_square(env, 17, scene3_);
            expect_halves(env, 17, rgba(tc), rgba(td));
            record_error(mgl::ErrorCode::NoError, "C0, C1: no GL error from the block-array draws");

            // C2-C6: program 1 on its own point 5, fed from the first palette buffer.
            mgl::glUniformBlockBinding(scene1_, palette1_index_, 5);
            mgl::glBindBufferBase(kUniform, 5, buf_[kBufPalette0]);
            PaletteData pd = make_palette({{{0.9f, 0.5f, 0.1f}, {0.2f, 0.7f, 0.9f}, {0.5f, 0.9f, 0.3f}}}, -0.30f, 0.0f, 1.0f, 0);
            // C2: glBufferData allocates a new data store with the whole block.
            mgl::glBufferData(kUniform, sizeof(PaletteData), &pd, mgl::BufferUsage::DynamicDraw);
            draw_square(env, 18, scene1_);
            expect_s1(env, 18, pd);
            // C3, C4: change ONE member, at the offset glGetActiveUniformsiv reported - no CPU layout knowledge needed.
            const std::int32_t pick = 1;
            mgl::glBufferSubData(kUniform, pick_offset_, sizeof(pick), &pick);
            pd.pick = pick;
            draw_square(env, 19, scene1_);
            expect_s1(env, 19, pd);
            const float gain = 0.5f;
            mgl::glBufferSubData(kUniform, gain_offset_, sizeof(gain), &gain);
            pd.gain = gain;
            draw_square(env, 20, scene1_);
            expect_s1(env, 20, pd);
            // C5: glCopyBufferSubData replaces the whole block with ring slot 2 (a copy between two UNIFORM-used buffers).
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, buf_[kBufRing]);
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, buf_[kBufPalette0]);
            mgl::glCopyBufferSubData(mgl::BufferTarget::CopyRead, mgl::BufferTarget::CopyWrite, 2 * static_cast<GLintptr>(stride_), 0, sizeof(PaletteData));
            mgl::glBindBuffer(mgl::BufferTarget::CopyRead, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::CopyWrite, mgl::BufferId{0});
            draw_square(env, 21, scene1_);
            expect_s1(env, 21, ring_slots_[2]);
            // C6: orphaning - glBufferData again with new contents; the binding follows the buffer name.
            const PaletteData pe = make_palette({{{0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f}, {0.95f, 0.3f, 0.5f}}}, 0.30f, 0.0f, 1.0f, 2);
            mgl::glBufferData(kUniform, sizeof(PaletteData), &pe, mgl::BufferUsage::StreamDraw);
            draw_square(env, 22, scene1_);
            expect_s1(env, 22, pe);
            record_error(mgl::ErrorCode::NoError, "C2-C6: no GL error from the member updates, the copy and the re-specification");
        }

        void check_scene_cells(demo::Check& check)
        {
            for (int k = 0; k < kSceneCells; ++k)
            {
                std::string bad;
                const auto& list = probes_[static_cast<std::size_t>(k)];
                for (const PixelProbe& p : list)
                {
                    const Rgba8 got = demo::read_pixel(p.x, p.y);
                    // Tolerance 1: a float color (value * gain, 1 - value * gain) is converted to 8 bits with
                    // rounding after highp arithmetic; ES 3.0 section 2.1.2 / 4.1.8 allow an error of 1/255.
                    if (!near(got, p.want, 1)) bad += std::format("({},{}) is {} expected {}; ", p.x, p.y, demo::format_rgba(got), demo::format_rgba(p.want));
                }
                check.expect(!list.empty() && bad.empty(), std::format("scene cell {}: {} probes match the CPU model", kSceneLabels[k], list.size()), bad);
            }
        }

        // ----------------------------------------------------------------------------------
        // introspection
        // ----------------------------------------------------------------------------------
        static std::string join(const std::vector<std::string>& v)
        {
            std::string s;
            for (const std::string& x : v) s += x + " ";
            return s;
        }

        void check_probe_introspection(demo::Check& check)
        {
            const Block& b = probe_;
            std::vector<std::string> names;
            for (const Leaf& l : b.leaves) names.push_back(b.gl_name(l));
            const Queried q = query_uniforms(verify_, names);
            const auto n = names.size();

            std::vector<std::string> bad_index, bad_block, bad_type, bad_size, bad_offset, bad_astride, bad_mstride, bad_major;
            for (std::size_t i = 0; i < n; ++i)
            {
                const Leaf& l = b.leaves[i];
                if (q.index[i] == GL_INVALID_INDEX) { bad_index.push_back(names[i]); continue; }
                if (static_cast<GLuint>(q.block[i]) != probe_index_) bad_block.push_back(names[i]);
                if (static_cast<GLenum>(q.type[i]) != static_cast<GLenum>(l.type.gl_type())) bad_type.push_back(names[i]);
                if (q.size[i] != l.count) bad_size.push_back(std::format("{} (size {} expected {})", names[i], q.size[i], l.count));
                if (q.offset[i] != l.offset) bad_offset.push_back(std::format("{} (offset {} expected {})", names[i], q.offset[i], l.offset));
                if (q.array_stride[i] != l.array_stride) bad_astride.push_back(std::format("{} ({} expected {})", names[i], q.array_stride[i], l.array_stride));
                if (q.matrix_stride[i] != l.matrix_stride) bad_mstride.push_back(std::format("{} ({} expected {})", names[i], q.matrix_stride[i], l.matrix_stride));
                if ((q.row_major[i] != 0) != l.row_major) bad_major.push_back(names[i]);
            }
            check.expect(bad_index.empty(), std::format("glGetUniformIndices resolves all {} member names of block Probe (\"Probe.farr[0]\", \"Probe.lights[1].color\", ...)", n), join(bad_index));
            check.expect(bad_block.empty(), "glGetActiveUniformsiv(UNIFORM_BLOCK_INDEX): every member belongs to the block named Probe", join(bad_block));
            check.expect(bad_type.empty(), "glGetActiveUniformsiv(UNIFORM_TYPE): all member types match the declarations", join(bad_type));
            check.expect(bad_size.empty(), "glGetActiveUniformsiv(UNIFORM_SIZE): array members report their element count, others 1", join(bad_size));
            check.expect(bad_offset.empty(), "glGetActiveUniformsiv(UNIFORM_OFFSET): all 30 offsets equal the CPU std140 calculator (vec3 packing, 16-byte array and matrix columns, struct padding)", join(bad_offset));
            check.expect(bad_astride.empty(), "glGetActiveUniformsiv(UNIFORM_ARRAY_STRIDE): 16 for scalar/vector arrays, 48 for the mat3 array, 32 for lights, 0 for non-arrays", join(bad_astride));
            check.expect(bad_mstride.empty(), "glGetActiveUniformsiv(UNIFORM_MATRIX_STRIDE): 16 for every matrix (column- and row-major), 0 for non-matrices", join(bad_mstride));
            check.expect(bad_major.empty(), "glGetActiveUniformsiv(UNIFORM_IS_ROW_MAJOR): 1 exactly for m3r and m2x4r", join(bad_major));

            // Spot checks of the numbers the README discusses, independent of the calculator.
            auto offset_of = [&](const char* name) {
                for (std::size_t i = 0; i < n; ++i)
                    if (names[i] == name) return q.offset[i];
                return -1;
            };
            check.expect(offset_of("Probe.v3") == 16 && offset_of("Probe.after_v3") == 28 && offset_of("Probe.v3b") == 48 && offset_of("Probe.i") == 60,
                         "hand-derived offsets: vec3 at 16 occupies 12 bytes, the next float lands at 28, a vec3 after a vec2 jumps to 48, an int then packs at 60",
                         std::format("v3 {}, after_v3 {}, v3b {}, i {}", offset_of("Probe.v3"), offset_of("Probe.after_v3"), offset_of("Probe.v3b"), offset_of("Probe.i")));
            check.expect(offset_of("Probe.lights[0].position") == 672 && offset_of("Probe.lights[1].intensity") == 716 && offset_of("Probe.inner.b") == 640 && offset_of("Probe.after_inner") == 656,
                         "hand-derived offsets: a struct is padded to 16 (inner.b 640, after_inner 656), lights[] has stride 32 (lights[1].intensity 716)",
                         std::format("lights[0].position {}, lights[1].intensity {}, inner.b {}, after_inner {}", offset_of("Probe.lights[0].position"),
                                     offset_of("Probe.lights[1].intensity"), offset_of("Probe.inner.b"), offset_of("Probe.after_inner")));

            // Block-level queries.
            check.expect_eq(block_param(verify_, probe_index_, mgl::UniformBlockParameter::DataSize), b.size, "glGetActiveUniformBlockiv(UNIFORM_BLOCK_DATA_SIZE) equals the std140 size computed on the CPU (736)");
            check.expect_eq(block_param(verify_, probe_index_, mgl::UniformBlockParameter::ActiveUniforms), static_cast<int>(n), "glGetActiveUniformBlockiv(UNIFORM_BLOCK_ACTIVE_UNIFORMS) = 30 (an array is one entry, each struct-array element its own)");
            std::vector<GLint> indices(n, -1);
            mgl::glGetActiveUniformBlockiv(verify_, probe_index_, mgl::UniformBlockParameter::ActiveUniformIndices, indices.data());
            const std::set<GLuint> from_block(indices.begin(), indices.end()), from_names(q.index.begin(), q.index.end());
            check.expect(std::set<GLuint>(indices.begin(), indices.end()) == from_names && from_block.size() == n,
                         "UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES lists exactly the indices glGetUniformIndices returned for the 30 names");
            check.expect(block_param(verify_, probe_index_, mgl::UniformBlockParameter::ReferencedByVertexShader) == GL_FALSE &&
                             block_param(verify_, probe_index_, mgl::UniformBlockParameter::ReferencedByFragmentShader) == GL_TRUE,
                         "UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER FALSE, ..._BY_FRAGMENT_SHADER TRUE (only the fragment shader reads Probe)");

            // Names.
            char name[32] = {};
            GLsizei length = -1;
            mgl::glGetActiveUniformBlockName(verify_, probe_index_, sizeof(name), &length, name);
            check.expect(std::string(name) == "Probe" && length == 5, "glGetActiveUniformBlockName returns the BLOCK name \"Probe\" (not the instance name \"P\"), length 5 without the terminator", std::format("'{}' length {}", name, length));
            check.expect_eq(block_param(verify_, probe_index_, mgl::UniformBlockParameter::NameLength), 6, "UNIFORM_BLOCK_NAME_LENGTH counts the terminator (6)");
            char small[4] = {'x', 'x', 'x', 'x'};
            length = -1;
            mgl::glGetActiveUniformBlockName(verify_, probe_index_, sizeof(small), &length, small);
            check.expect(std::string(small) == "Pro" && length == 3 && small[3] == '\0', "glGetActiveUniformBlockName with bufSize 4 writes \"Pro\" + terminator and reports length 3", std::format("'{}' length {}", small, length));
            check.expect(mgl::glGetUniformBlockIndex(verify_, "P") == GL_INVALID_INDEX && mgl::glGetUniformBlockIndex(verify_, "NoSuchBlock") == GL_INVALID_INDEX,
                         "glGetUniformBlockIndex: the instance name \"P\" and an unknown name both give INVALID_INDEX");
            check.expect_no_gl_error("... and INVALID_INDEX is not a GL error");
            const GLuint missing = [&] {
                const GLchar* nm = "Probe.no_such_member";
                GLuint idx = 0;
                mgl::glGetUniformIndices(verify_, 1, &nm, &idx);
                return idx;
            }();
            check.expect(missing == GL_INVALID_INDEX, "glGetUniformIndices: an unknown member name gives INVALID_INDEX");
            check.expect(mgl::glGetUniformLocation(verify_, "Probe.f").value == -1 && mgl::glGetUniformLocation(verify_, "P.f").value == -1,
                         "glGetUniformLocation of a block member is -1: members of named blocks have no location (sec 2.12.6)");
            check.expect_no_gl_error("... without raising an error");
            GLint blocks = -1, max_len = -1;
            mgl::glGetProgramiv(verify_, mgl::ProgramParameter::ActiveUniformBlocks, &blocks);
            mgl::glGetProgramiv(verify_, mgl::ProgramParameter::ActiveUniformBlockMaxLength, &max_len);
            check.expect(blocks == 3 && max_len == 8, "glGetProgramiv: ACTIVE_UNIFORM_BLOCKS = 3 (Probe, Mistake, Queried), ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH = 8 (\"Mistake\" + terminator)", std::format("blocks {}, max length {}", blocks, max_len));
        }

        void check_mistake_and_queried(demo::Check& check)
        {
            // The Mistake block: GL against the CPU std140 numbers, then against both wrong layouts.
            std::vector<std::string> names;
            for (const Leaf& l : mistake_.leaves) names.push_back(mistake_.gl_name(l));
            const Queried q = query_uniforms(verify_, names);
            auto count_mismatches = [&](const std::vector<Leaf>& leaves) {
                int bad = 0;
                for (std::size_t i = 0; i < leaves.size(); ++i)
                    if (q.offset[i] != leaves[i].offset) ++bad;
                return bad;
            };
            const std::vector<Leaf> tight = naive_tight(mistake_), padded = naive_padded(mistake_);
            check.expect_eq(count_mismatches(mistake_.leaves), 0, "block Mistake: the GL offsets (0, 12, 16, 32, 44) equal the CPU std140 calculator");
            check.expect(block_param(verify_, mistake_index_, mgl::UniformBlockParameter::DataSize) == 48 && mistake_.size == 48,
                         "block Mistake: UNIFORM_BLOCK_DATA_SIZE = 48 (the last float ends at 48, a multiple of 16)");
            // Negative control for the offset comparison: the same comparison must FAIL for the wrong layouts.
            check.expect(count_mismatches(tight) == 2 && count_mismatches(padded) == 4,
                         "negative control: the offset comparison reports 2 mismatches (c, t) for the tight C layout and 4 (s, b, c, t) for the padded one",
                         std::format("tight {}, padded {}", count_mismatches(tight), count_mismatches(padded)));
            check.expect(q.offset[1] == 12, "the float after a vec3 sits at offset 12, inside the vec3's 16-byte slot (it is NOT pushed to 16)", std::format("m_s offset {}", q.offset[1]));

            // The packed block: the layout is the implementation's; only structural rules are universal.
            std::string bad;
            std::vector<std::pair<int, int>> spans;
            for (const Leaf& l : queried_gl_)
            {
                const int extent = l.type.is_matrix() ? l.matrix_stride * (l.row_major ? l.type.rows : l.type.cols) : 4 * l.type.rows;
                const int last = l.count > 1 ? (l.count - 1) * l.array_stride + extent : extent;
                spans.emplace_back(l.offset, l.offset + last);
                if (l.offset < 0 || l.offset % 4 != 0 || l.offset + last > queried_gl_size_) bad += l.name + " ";
                if (l.type.is_matrix() && l.matrix_stride <= 0) bad += l.name + "(matrix stride) ";
                if (l.count > 1 && l.array_stride <= 0) bad += l.name + "(array stride) ";
            }
            std::sort(spans.begin(), spans.end());
            for (std::size_t i = 1; i < spans.size(); ++i)
                if (spans[i].first < spans[i - 1].second) bad += "overlap ";
            check.expect(bad.empty(), std::format("block Queried (packed): the queried layout is well-formed (4-byte aligned, non-overlapping, inside UNIFORM_BLOCK_DATA_SIZE = {})", queried_gl_size_), bad);
            // The std140 numbers are NOT required for a packed block; report where this driver chose differently.
            std::string differing;
            for (std::size_t i = 0; i < queried_gl_.size(); ++i)
                if (queried_gl_[i].offset != queried_.leaves[i].offset || queried_gl_[i].matrix_stride != queried_.leaves[i].matrix_stride ||
                    queried_gl_[i].array_stride != queried_.leaves[i].array_stride)
                    differing += std::format("{} (offset {}, array stride {}, matrix stride {}; std140: {}, {}, {}) ", queried_gl_[i].name, queried_gl_[i].offset,
                                             queried_gl_[i].array_stride, queried_gl_[i].matrix_stride, queried_.leaves[i].offset, queried_.leaves[i].array_stride,
                                             queried_.leaves[i].matrix_stride);
            std::cout << std::format("   [note] packed block Queried, DATA_SIZE {}: {}\n", queried_gl_size_,
                                     differing.empty() ? "this driver lays it out exactly like std140" : "differs from std140 in " + differing);
        }

        void check_program_blocks(demo::Check& check)
        {
            // Program 1: instance-named block used by BOTH stages.
            std::vector<std::string> names = {"Palette.color[0]", "Palette.offset", "Palette.gain", "Palette.pick"};
            Queried q = query_uniforms(scene1_, names);
            check.expect(q.offset[0] == 0 && q.array_stride[0] == 16 && q.size[0] == 3 && q.offset[1] == 48 && q.offset[2] == 56 && q.offset[3] == 60,
                         "program 1: Palette.color[0] offset 0 stride 16 size 3, Palette.offset 48, Palette.gain 56, Palette.pick 60 - exactly the C++ struct PaletteData",
                         std::format("{} {} {} | {} {} {}", q.offset[0], q.array_stride[0], q.size[0], q.offset[1], q.offset[2], q.offset[3]));
            check.expect(block_param(scene1_, palette1_index_, mgl::UniformBlockParameter::DataSize) == 64 && block_param(scene1_, palette1_index_, mgl::UniformBlockParameter::ActiveUniforms) == 4,
                         "program 1: Palette has UNIFORM_BLOCK_DATA_SIZE 64 and 4 active uniforms");
            check.expect(block_param(scene1_, palette1_index_, mgl::UniformBlockParameter::ReferencedByVertexShader) == GL_TRUE &&
                             block_param(scene1_, palette1_index_, mgl::UniformBlockParameter::ReferencedByFragmentShader) == GL_TRUE,
                         "program 1: Palette is referenced by the vertex AND the fragment shader");
            // Program 2: the same block without an instance name, used by the fragment shader only.
            names = {"color[0]", "offset", "gain", "pick"};
            q = query_uniforms(scene2_, names);
            check.expect(q.index[0] != GL_INVALID_INDEX && q.offset[0] == 0 && q.offset[1] == 48 && q.offset[2] == 56 && q.offset[3] == 60 &&
                             mgl::glGetUniformBlockIndex(scene2_, "Palette") == palette2_index_,
                         "program 2: without an instance name the members are plain \"color[0]\", \"gain\", ... and sit at the same std140 offsets as in program 1 (that is what lets one buffer serve both)");
            check.expect(block_param(scene2_, palette2_index_, mgl::UniformBlockParameter::ReferencedByVertexShader) == GL_FALSE &&
                             block_param(scene2_, palette2_index_, mgl::UniformBlockParameter::ReferencedByFragmentShader) == GL_TRUE,
                         "program 2: Palette is referenced by the fragment shader only");
            // Program 3: an array of blocks.
            GLint blocks = -1;
            mgl::glGetProgramiv(scene3_, mgl::ProgramParameter::ActiveUniformBlocks, &blocks);
            char nm0[16] = {}, nm1[16] = {};
            mgl::glGetActiveUniformBlockName(scene3_, tint0_index_, sizeof(nm0), nullptr, nm0);
            mgl::glGetActiveUniformBlockName(scene3_, tint1_index_, sizeof(nm1), nullptr, nm1);
            check.expect(blocks == 2 && std::string(nm0) == "Tint[0]" && std::string(nm1) == "Tint[1]",
                         "program 3: an array of two blocks is listed as two active blocks \"Tint[0]\" and \"Tint[1]\"", std::format("{} blocks '{}' '{}'", blocks, nm0, nm1));
            const GLchar* member = "Tint.color";
            GLuint idx = 0;
            mgl::glGetUniformIndices(scene3_, 1, &member, &idx);
            GLint owner = -2;
            if (idx != GL_INVALID_INDEX) mgl::glGetActiveUniformsiv(scene3_, 1, &idx, mgl::UniformParameter::BlockIndex, &owner);
            GLint active = -1;
            mgl::glGetProgramiv(scene3_, mgl::ProgramParameter::ActiveUniforms, &active);
            check.expect(idx != GL_INVALID_INDEX && active == 1 && static_cast<GLuint>(owner) == std::min(tint0_index_, tint1_index_),
                         "program 3: the member is listed ONCE as \"Tint.color\" and its UNIFORM_BLOCK_INDEX is the index of the first block of the array",
                         std::format("active uniforms {}, block index {} (Tint[0] = {})", active, owner, tint0_index_));
        }

        // ----------------------------------------------------------------------------------
        // errors
        // ----------------------------------------------------------------------------------
        void check_errors(demo::Check& check)
        {
            const GLuint bad_block = 999;
            GLint sentinel = 777;
            mgl::glGetActiveUniformBlockiv(verify_, bad_block, mgl::UniformBlockParameter::DataSize, &sentinel);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetActiveUniformBlockiv with a block index >= ACTIVE_UNIFORM_BLOCKS -> INVALID_VALUE");
            check.expect_eq(sentinel, 777, "... and nothing is written to params");
            char name[8] = {'x', 'x', 'x', 'x', 'x', 'x', 'x', '\0'};
            GLsizei length = 55;
            mgl::glGetActiveUniformBlockName(verify_, bad_block, sizeof(name), &length, name);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetActiveUniformBlockName with a block index >= ACTIVE_UNIFORM_BLOCKS -> INVALID_VALUE");
            check.expect(length == 55 && name[0] == 'x', "... and nothing is written to length or name");
            mgl::glUniformBlockBinding(verify_, bad_block, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glUniformBlockBinding with an inactive block index -> INVALID_VALUE");
            mgl::glUniformBlockBinding(verify_, probe_index_, static_cast<GLuint>(max_bindings_));
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glUniformBlockBinding with a binding >= MAX_UNIFORM_BUFFER_BINDINGS -> INVALID_VALUE");
            check.expect_eq(block_param(verify_, probe_index_, mgl::UniformBlockParameter::Binding), static_cast<int>(kPointProbe), "... and the block keeps its binding");
            GLint active = -1;
            mgl::glGetProgramiv(verify_, mgl::ProgramParameter::ActiveUniforms, &active);
            GLuint idx = static_cast<GLuint>(active);
            GLint out = 888;
            mgl::glGetActiveUniformsiv(verify_, 1, &idx, mgl::UniformParameter::Offset, &out);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetActiveUniformsiv with an index >= ACTIVE_UNIFORMS -> INVALID_VALUE");
            check.expect_eq(out, 888, "... and nothing is written to params");
            GLint v = 999;
            mgl::glGetIntegeri_v(mgl::GetParameter::UniformBufferBinding, static_cast<GLuint>(max_bindings_), &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(UNIFORM_BUFFER_BINDING, MAX_UNIFORM_BUFFER_BINDINGS) -> INVALID_VALUE (index out of range)");
            GLint64 v64 = 999;
            mgl::glGetInteger64i_v(mgl::GetParameter::UniformBufferSize, static_cast<GLuint>(max_bindings_), &v64);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetInteger64i_v(UNIFORM_BUFFER_SIZE, MAX_UNIFORM_BUFFER_BINDINGS) -> INVALID_VALUE");
        }

        // ----------------------------------------------------------------------------------
        // limits
        // ----------------------------------------------------------------------------------
        /// A program whose fragment shader reads `blocks` uniform blocks. Returns whether it linked.
        static bool links_with_blocks(int blocks, std::string& log)
        {
            std::string fs = "#version 300 es\nprecision highp float;\n";
            for (int i = 0; i < blocks; ++i) fs += std::format("layout(std140) uniform B{} {{ vec4 v; }} b{};\n", i, i);
            fs += "out vec4 o_color;\nvoid main() {\n  o_color = vec4(0.0)";
            for (int i = 0; i < blocks; ++i) fs += std::format(" + b{}.v", i);
            fs += ";\n}\n";
            const std::string vs = "#version 300 es\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
            std::string vlog, flog;
            const mgl::ShaderId v = demo::compile_shader(mgl::ShaderType::Vertex, vs, vlog);
            const mgl::ShaderId f = demo::compile_shader(mgl::ShaderType::Fragment, fs, flog);
            bool linked = false;
            if (v.value != 0 && f.value != 0)
            {
                const std::array<mgl::ShaderId, 2> shaders = {v, f};
                const mgl::ProgramId p = demo::link_program(shaders, log);
                linked = p.value != 0;
                if (linked) mgl::glDeleteProgram(p);
            }
            else
            {
                log = "compile failed: " + vlog + flog;
            }
            if (v.value != 0) mgl::glDeleteShader(v);
            if (f.value != 0) mgl::glDeleteShader(f);
            return linked;
        }

        void check_limits(demo::Check& check)
        {
            const int vblocks = demo::get_integer(mgl::GetParameter::MaxVertexUniformBlocks);
            const int fblocks = demo::get_integer(mgl::GetParameter::MaxFragmentUniformBlocks);
            const int cblocks = demo::get_integer(mgl::GetParameter::MaxCombinedUniformBlocks);
            const int bindings = demo::get_integer(mgl::GetParameter::MaxUniformBufferBindings);
            GLint64 block_size = 0, comb_v = 0, comb_f = 0;
            mgl::glGetInteger64v(mgl::GetParameter::MaxUniformBlockSize, &block_size);
            mgl::glGetInteger64v(mgl::GetParameter::MaxCombinedVertexUniformComponents, &comb_v);
            mgl::glGetInteger64v(mgl::GetParameter::MaxCombinedFragmentUniformComponents, &comb_f);
            const int vcomp = demo::get_integer(mgl::GetParameter::MaxVertexUniformComponents);
            const int fcomp = demo::get_integer(mgl::GetParameter::MaxFragmentUniformComponents);
            check.expect(vblocks >= 12 && fblocks >= 12 && cblocks >= 24 && bindings >= 24,
                         "MAX_VERTEX/FRAGMENT_UNIFORM_BLOCKS >= 12, MAX_COMBINED_UNIFORM_BLOCKS >= 24, MAX_UNIFORM_BUFFER_BINDINGS >= 24 (Tables 6.31 - 6.33 minima)",
                         std::format("{} {} {} {}", vblocks, fblocks, cblocks, bindings));
            check.expect(block_size >= 16384, "MAX_UNIFORM_BLOCK_SIZE (glGetInteger64v) >= 16384 (Table 6.33 minimum)", std::format("{}", block_size));
            check.expect(alignment_ >= 1 && alignment_ <= 256, "UNIFORM_BUFFER_OFFSET_ALIGNMENT is between 1 and 256 (256 is the largest value the spec allows)", std::format("{}", alignment_));
            check.expect(comb_v >= static_cast<GLint64>(vblocks) * block_size / 4 + vcomp && comb_f >= static_cast<GLint64>(fblocks) * block_size / 4 + fcomp,
                         "MAX_COMBINED_VERTEX/FRAGMENT_UNIFORM_COMPONENTS >= MAX_stage_UNIFORM_BLOCKS x MAX_UNIFORM_BLOCK_SIZE / 4 + MAX_stage_UNIFORM_COMPONENTS (Table 6.33, footnote)",
                         std::format("{} {}", comb_v, comb_f));
            check.expect(probe_.size <= block_size, std::format("block Probe ({} bytes) fits MAX_UNIFORM_BLOCK_SIZE", probe_.size));
            check.expect(kPointQueried < static_cast<GLuint>(bindings) && 13 < bindings, "every binding point this demo uses (0-13) is below MAX_UNIFORM_BUFFER_BINDINGS");

            // "If the number of uniform blocks used by any shader in the program exceeds its corresponding limit, the
            // program will fail to link" (sec 2.12.6.2): exactly the limit links, one more does not.
            std::string log_at, log_over;
            const bool at_limit = links_with_blocks(fblocks, log_at);
            const bool over_limit = links_with_blocks(fblocks + 1, log_over);
            check.expect(at_limit, std::format("a fragment shader reading exactly MAX_FRAGMENT_UNIFORM_BLOCKS = {} blocks links", fblocks), log_at.substr(0, 160));
            check.expect(!over_limit, std::format("a fragment shader reading {} blocks (MAX_FRAGMENT_UNIFORM_BLOCKS + 1) fails to link", fblocks + 1),
                         over_limit ? "it linked" : log_over.substr(0, 160));
            check.expect_no_gl_error("the limit experiments leave no GL error");
        }

        // ----------------------------------------------------------------------------------
        // negative controls
        // ----------------------------------------------------------------------------------
        void negative_controls(const demo::Env& env, demo::Check& check)
        {
            // Corrupt one member of block Probe in the buffer (v3b is verification cell 4), redraw the grid, and require
            // exactly that cell to turn red; then restore the value and require green again.
            const Leaf* v3b = nullptr;
            for (const Leaf& l : probe_.leaves)
                if (l.name == "v3b") v3b = &l;
            if (v3b == nullptr) throw demo::Failure("member v3b missing");
            const float wrong = -1.0f;
            mgl::glBindBuffer(kUniform, buf_[kBufProbe]);
            mgl::glBufferSubData(kUniform, v3b->offset + 4, 4, &wrong);  // v3b.y
            draw_verification_grid(env);
            const auto [x, y] = verify_cell_center(env, 4);
            check.expect_pixel(x, y, kRed, 0, "negative control: a wrong v3b.y in the buffer turns its verification cell red");
            const auto [nx, ny] = verify_cell_center(env, 3);
            check.expect_pixel(nx, ny, kGreen, 0, "negative control: the neighbouring cell (v2) stays green");
            const Bytes good = pack(probe_.leaves, static_cast<std::size_t>(probe_.size));
            mgl::glBindBuffer(kUniform, buf_[kBufProbe]);  // draw_verification_grid() left the Mistake buffer as the general binding
            mgl::glBufferSubData(kUniform, v3b->offset + 4, 4, good.data() + v3b->offset + 4);
            draw_verification_grid(env);
            check.expect_pixel(x, y, kGreen, 0, "negative control: restoring the value turns the cell green again");
            mgl::glBindBuffer(kUniform, mgl::BufferId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        // ----------------------------------------------------------------------------------
        // state
        // ----------------------------------------------------------------------------------
        mgl::ProgramId verify_{}, scene1_{}, scene2_{}, scene3_{};
        mgl::UniformLocation u_cell_{}, u_height_{};
        GLuint probe_index_ = 0, mistake_index_ = 0, queried_index_ = 0, palette1_index_ = 0, palette2_index_ = 0, tint0_index_ = 0, tint1_index_ = 0;
        Block probe_, mistake_, queried_;
        std::vector<Leaf> queried_gl_;
        int queried_gl_size_ = 0;
        int alignment_ = 1;
        int max_bindings_ = 24;
        int stride_ = 64;
        std::array<mgl::BufferId, 12> buf_{};
        std::vector<Obs> init_obs_, obs_;
        std::array<std::vector<PixelProbe>, kSceneCells> probes_;
        std::array<PaletteData, 4> ring_slots_{};
        int gain_offset_ = 0, pick_offset_ = 0;  // offsets of Palette.gain / Palette.pick in program 1, as queried
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-uniform-buffer-objects",
        .title = "Uniform buffer objects",
        .min_es = {3, 0},
        .summary = "std140 blocks (scalars, vec3 packing, arrays, matrices, row_major, structs) checked against a CPU layout calculator "
                   "and the GL's introspection, block bindings, glBindBufferBase/Range, one buffer shared by two programs, updates "
                   "between draws, limits and errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<UniformBufferObjects>(argc, argv, info);
}
