// es31-ssbo-std430
//
// Shader storage blocks (ES 3.2 section 7.8, GLSL ES 3.10 sections 4.3.7 and 4.4.4): buffer objects that shaders
// read AND write. The demo shows
//   * the std430 packing rules next to std140 (ES 3.2 section 7.6.2.2), predicted by a CPU layout calculator and
//     compared with what glGetProgramResourceiv(BUFFER_VARIABLE) reports,
//   * the same sixteen-member block verified in BOTH directions: data packed by the CPU calculator is read by a
//     compute shader (a flag per member), and data written by the compute shader is compared byte for byte with an
//     image built by the calculator,
//   * deliberately wrong layouts (std140 data read through a std430 block and the reverse), whose effect on every
//     member is predicted exactly,
//   * runtime-sized last members whose length() is driven by glBindBufferRange (aligned offset and size),
//   * the indexed state queries, the limits and the defined errors, and glMemoryBarrier before every consumer of the
//     compute shader's stores (fragment and vertex shader reads, buffer mapping).
//
// The 256x256 surface:
//
//  TOP (64 px): the VERIFICATION GRID, 4 rows x 16 cells of 16 px, drawn by a FRAGMENT shader that reads the flag
//  buffer written by the compute shader (cell m = member m of block Probe):
//    row 0  block Probe declared std430, buffer packed with the std430 layout   - all green
//    row 1  block Probe declared std140, buffer packed with the std140 layout   - all green
//    row 2  declared std430, buffer packed with the STD140 layout (a mistake)    - green only where both layouts agree
//    row 3  declared std140, buffer packed with the STD430 layout (a mistake)    - green only where both layouts agree
//  Magenta = "the flag is 0 as predicted by the calculator"; red anywhere would be a failure.
//
//  BELOW (3 bands of 48 px): bar charts drawn by a VERTEX shader that reads the runtime-sized array values[] written
//  by the compute shader. The bands show buffer ranges of 32, 16 and 4 elements; the bar width is 2 / values.length().

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
using Bytes = std::vector<std::uint8_t>;

namespace
{
    // =========================================================================================
    // 1. A std140 / std430 layout calculator (ES 3.2 section 7.6.2.2), independent of the GL
    // =========================================================================================
    enum class Layout { Std430 = 0, Std140 = 1 };
    enum class Basic { Float, Int, UInt };

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
                const auto idx = static_cast<std::size_t>(rows - 1);
                switch (basic)
                {
                case Basic::Float: return f[idx];
                case Basic::Int: return i[idx];
                case Basic::UInt: return u[idx];
                }
            }
            if (cols == 2 && rows == 2) return T::FloatMat2;
            if (cols == 3 && rows == 3) return T::FloatMat3;
            if (cols == 3 && rows == 2) return T::FloatMat3x2;
            if (cols == 2 && rows == 3) return T::FloatMat2x3;
            return T::FloatMat4;
        }
    };

    constexpr TypeInfo kFloat{Basic::Float, 1, 1}, kVec2{Basic::Float, 1, 2}, kVec3{Basic::Float, 1, 3}, kInt{Basic::Int, 1, 1},
        kUVec2{Basic::UInt, 1, 2}, kMat2{Basic::Float, 2, 2}, kMat3{Basic::Float, 3, 3}, kMat3x2{Basic::Float, 3, 2},
        kMat2x3{Basic::Float, 2, 3};

    /// One declaration inside the block: a basic type, or a structure when `fields` is not empty.
    struct Member
    {
        std::string name;
        int id = 0;     ///< 1-based position of the top-level member (the shader's `case`)
        TypeInfo type{};
        int array = 0;  ///< 0: not an array
        bool row_major = false;
        std::vector<Member> fields;
    };

    Member basic(std::string name, int id, TypeInfo t, int array = 0, bool row_major = false)
    {
        return Member{std::move(name), id, t, array, row_major, {}};
    }
    Member structure(std::string name, int id, std::vector<Member> fields, int array = 0)
    {
        return Member{std::move(name), id, {}, array, false, std::move(fields)};
    }

    /// One active buffer variable (a leaf of the declaration tree), as the GL enumerates it.
    struct Leaf
    {
        std::string name;        ///< GL resource name: "Probe.farr[0]", "Probe.pairs[1].b"
        int member = 0;          ///< top-level member id
        int elem = 0;            ///< element of the enclosing array of structures (0 if none)
        int field = 0;           ///< index of the field inside the enclosing structure (0 if none)
        TypeInfo type{};
        bool is_array = false;   ///< declared as an array of a basic type
        int count = 1;           ///< array elements (GL ARRAY_SIZE); 1 when not an array
        int offset = 0;
        int array_stride = 0;    ///< 0 when not an array
        int matrix_stride = 0;   ///< 0 when not a matrix
        bool row_major = false;  ///< matrices only
        int top_stride = 0;      ///< GL TOP_LEVEL_ARRAY_STRIDE: the structure size for an array of structures, else 0
        int top_size = 1;        ///< GL TOP_LEVEL_ARRAY_SIZE: the element count for an array of structures, else 1
        bool listed = true;      ///< false for elements 1.. of a top-level array of structures: the GL lists only the first element
    };

    int align_up(int v, int a) { return (v + a - 1) / a * a; }
    int vec_align(int components) { return components == 1 ? 4 : components == 2 ? 8 : 16; }  // rules 1-3: vec3 is 4N

    struct Measure
    {
        int align, size, array_stride, matrix_stride;
    };

    /// Rules 1-8 for scalars, vectors, matrices and arrays of them. std430 differs from std140 only in that the
    /// stride and base alignment of an array of scalars/vectors (and the column stride of matrices) are NOT rounded
    /// up to that of a vec4.
    Measure measure_basic(const Member& m, Layout layout)
    {
        const TypeInfo& t = m.type;
        const int round = layout == Layout::Std140 ? 16 : 1;
        Measure r{};
        if (t.is_matrix())
        {
            const int vectors = m.row_major ? t.rows : t.cols;    // rules 5-8: an array of column (row) vectors
            const int comps = m.row_major ? t.cols : t.rows;
            const int stride = align_up(vec_align(comps), round);
            const int matrix_size = vectors * stride;
            r.align = stride;
            r.matrix_stride = stride;
            r.array_stride = m.array > 0 ? matrix_size : 0;
            r.size = matrix_size * (m.array > 0 ? m.array : 1);
        }
        else if (m.array > 0)
        {
            const int stride = align_up(vec_align(t.rows), round);  // rule 4
            r.align = stride;
            r.array_stride = stride;
            r.size = stride * m.array;
        }
        else
        {
            r.align = vec_align(t.rows);
            r.size = 4 * t.rows;
        }
        return r;
    }

    struct StructMeasure
    {
        int align, size;
    };
    StructMeasure measure_struct(const std::vector<Member>& fields, Layout layout);

    int align_of(const Member& m, Layout layout)
    {
        return m.fields.empty() ? measure_basic(m, layout).align : measure_struct(m.fields, layout).align;
    }
    int size_of(const Member& m, Layout layout)
    {
        const int count = m.array > 0 ? m.array : 1;
        return m.fields.empty() ? measure_basic(m, layout).size : count * measure_struct(m.fields, layout).size;
    }

    /// Rule 9: the base alignment of a structure is its largest member alignment (std140: rounded up to a vec4's,
    /// std430: not), and its size is its end offset rounded up to that alignment (padding at the end).
    StructMeasure measure_struct(const std::vector<Member>& fields, Layout layout)
    {
        int offset = 0, align = 1;
        for (const Member& f : fields)
        {
            const int a = align_of(f, layout);
            offset = align_up(offset, a) + size_of(f, layout);
            align = std::max(align, a);
        }
        if (layout == Layout::Std140) align = align_up(align, 16);
        return {align, align_up(offset, align)};
    }

    struct Context
    {
        int member = 0;
        int elem = 0;
        int top_stride = 0;
        int top_size = 1;
    };

    /// Places `fields` starting at `base`, appends one Leaf per active buffer variable, returns the end offset.
    int place_fields(const std::vector<Member>& fields, int base, const std::string& prefix, Layout layout, const Context& outer,
                     bool top_level, std::vector<Leaf>& out)
    {
        int offset = base;
        int field_index = 0;
        for (const Member& m : fields)
        {
            Context ctx = outer;
            if (top_level) ctx = Context{m.id, 0, 0, 1};
            if (!m.fields.empty())
            {
                const StructMeasure sm = measure_struct(m.fields, layout);  // rules 9 and 10
                const int count = m.array > 0 ? m.array : 1;
                offset = align_up(offset, sm.align);
                for (int e = 0; e < count; ++e)
                {
                    Context c = ctx;
                    c.elem = e;
                    if (top_level)
                    {
                        c.top_stride = m.array > 0 ? sm.size : 0;
                        c.top_size = count;
                    }
                    place_fields(m.fields, offset + e * sm.size, prefix + m.name + (m.array > 0 ? std::format("[{}]", e) : "") + ".", layout,
                                 c, false, out);
                }
                offset += count * sm.size;
                ++field_index;
                continue;
            }
            const Measure ms = measure_basic(m, layout);
            offset = align_up(offset, ms.align);
            Leaf l;
            l.name = prefix + m.name + (m.array > 0 ? "[0]" : "");
            l.member = ctx.member;
            l.elem = ctx.elem;
            l.field = top_level ? 0 : field_index;
            l.type = m.type;
            l.is_array = m.array > 0;
            l.count = m.array > 0 ? m.array : 1;
            l.offset = offset;
            l.array_stride = ms.array_stride;
            l.matrix_stride = ms.matrix_stride;
            l.row_major = m.row_major && m.type.is_matrix();
            l.top_stride = ctx.top_stride;
            l.top_size = ctx.top_size;
            // ES 3.2 section 7.3.1.1: "active shader storage block members that are declared as arrays of an aggregate type ... generate entries
            // only for the first array element" (the other elements are reached through TOP_LEVEL_ARRAY_STRIDE).
            l.listed = !(ctx.top_size > 1 && ctx.elem > 0);
            out.push_back(std::move(l));
            offset += ms.size;
            ++field_index;
        }
        return offset;
    }

    struct Block
    {
        Layout layout = Layout::Std430;
        std::vector<Leaf> leaves;
        int end = 0;  ///< end of the last member in bytes

        void compute(const std::vector<Member>& members, Layout l)
        {
            layout = l;
            leaves.clear();
            end = place_fields(members, 0, "Probe.", l, Context{}, true, leaves);
        }
        /// The leaves as glGetProgramResource* enumerates them.
        [[nodiscard]] std::vector<const Leaf*> listed() const
        {
            std::vector<const Leaf*> out;
            for (const Leaf& l : leaves)
                if (l.listed) out.push_back(&l);
            return out;
        }
    };

    /// Byte offset of logical scalar k of array element e of a leaf. Matrices are numbered in GLSL order
    /// (column by column), whatever the memory order is.
    int scalar_offset(const Leaf& l, int e, int k)
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
        return off;
    }

    /// The test value the shader (layout.comp) uses for scalar k of element e of a leaf.
    int scalar_value(const Leaf& l, int e, int k) { return l.member * 10000 + (l.elem + e) * 1000 + l.field * 100 + k; }

    /// The block's data as the CPU calculator lays it out; at least `size` bytes, and never smaller than the block itself.
    Bytes pack(const Block& block, std::size_t size)
    {
        Bytes buf(std::max(size, static_cast<std::size_t>(align_up(block.end, 16))), 0);
        for (const Leaf& l : block.leaves)
            for (int e = 0; e < l.count; ++e)
                for (int k = 0; k < l.type.components(); ++k)
                {
                    const int v = scalar_value(l, e, k);
                    std::uint32_t bits = 0;
                    switch (l.type.basic)
                    {
                    case Basic::Float: { const float f = static_cast<float>(v); std::memcpy(&bits, &f, 4); break; }
                    case Basic::Int: bits = static_cast<std::uint32_t>(-v); break;
                    case Basic::UInt: bits = static_cast<std::uint32_t>(v); break;
                    }
                    std::memcpy(buf.data() + scalar_offset(l, e, k), &bits, 4);
                }
        return buf;
    }

    /// Members (bit m for member m) that a shader declared with layout `program` reads correctly from a buffer packed
    /// with layout `packed`: exactly those whose every scalar sits at the same byte offset in both layouts. Values are
    /// unique, so a scalar read from another place is never equal to the expected one (a zero in padding is not either).
    std::uint32_t predicted_mask(const Block& program, const Block& packed)
    {
        std::uint32_t mask = 0;
        for (int m = 1; m <= 16; ++m)
        {
            bool same = true;
            for (std::size_t i = 0; i < program.leaves.size(); ++i)
            {
                const Leaf& a = program.leaves[i];
                const Leaf& b = packed.leaves[i];
                if (a.member != m) continue;
                for (int e = 0; e < a.count; ++e)
                    for (int k = 0; k < a.type.components(); ++k)
                        if (scalar_offset(a, e, k) != scalar_offset(b, e, k)) same = false;
            }
            if (same) mask |= 1u << m;
        }
        return mask;
    }

    /// Compares two images of a block word by word. A difference at a scalar position of `block` is a MEMBER difference; one anywhere else is padding.
    struct ImageDiff
    {
        std::size_t member_words = 0;
        std::size_t padding_words = 0;
        std::size_t first_member = static_cast<std::size_t>(-1);  ///< byte offset of the first differing member scalar
    };
    ImageDiff diff_images(const Block& block, const Bytes& a, const Bytes& b)
    {
        const std::size_t words = std::min(a.size(), b.size()) / 4;
        std::vector<bool> is_member(words, false);
        for (const Leaf& l : block.leaves)
            for (int e = 0; e < l.count; ++e)
                for (int k = 0; k < l.type.components(); ++k)
                    if (static_cast<std::size_t>(scalar_offset(l, e, k) / 4) < words) is_member[static_cast<std::size_t>(scalar_offset(l, e, k) / 4)] = true;
        ImageDiff d;
        for (std::size_t w = 0; w < words; ++w)
        {
            std::uint32_t x = 0, y = 0;
            std::memcpy(&x, a.data() + 4 * w, 4);
            std::memcpy(&y, b.data() + 4 * w, 4);
            if (x == y) continue;
            if (is_member[w])
            {
                ++d.member_words;
                d.first_member = std::min(d.first_member, 4 * w);
            }
            else
            {
                ++d.padding_words;
            }
        }
        return d;
    }

    std::vector<Member> make_members()
    {
        const std::vector<Member> pair_fields = {basic("a", 0, kFloat), basic("b", 0, kFloat)};
        const std::vector<Member> combo_fields = {basic("f", 0, kFloat), basic("v", 0, kVec3)};
        return {
            basic("f", 1, kFloat), basic("v3", 2, kVec3), basic("after_v3", 3, kFloat), basic("v2", 4, kVec2),
            basic("farr", 5, kFloat, 3), basic("v2arr", 6, kVec2, 3), basic("v3arr", 7, kVec3, 2), basic("i", 8, kInt),
            basic("uv2", 9, kUVec2), basic("m2", 10, kMat2), basic("m3", 11, kMat3), basic("m3x2", 12, kMat3x2),
            basic("m2x3r", 13, kMat2x3, 0, true), structure("pairs", 14, pair_fields, 2), structure("combo", 15, combo_fields),
            basic("tail", 16, kFloat),
        };
    }

    // =========================================================================================
    // 2. Small GL helpers
    // =========================================================================================
    constexpr auto kSsbo = mgl::BufferTarget::ShaderStorage;

    /// Compute program from a shader file; `define` (e.g. "LAYOUT std140") is inserted after the #version line.
    mgl::ProgramId build_compute(const demo::Env& env, std::string_view file, std::string_view define = {})
    {
        std::string src = env.load_text(file);
        if (!define.empty()) src.insert(src.find('\n') + 1, std::format("#define {}\n", define));
        std::string log;
        const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, src, log);
        if (cs.value == 0) throw demo::Failure(std::format("{}: compute shader failed to compile\n{}", file, log));
        const std::array<mgl::ShaderId, 1> shaders = {cs};
        const mgl::ProgramId prog = demo::link_program(shaders, log);
        mgl::glDeleteShader(cs);
        if (prog.value == 0) throw demo::Failure(std::format("{}: program failed to link\n{}", file, log));
        return prog;
    }

    mgl::UniformLocation locate(mgl::ProgramId p, const char* name)
    {
        const mgl::UniformLocation l = mgl::glGetUniformLocation(p, name);
        if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
        return l;
    }

    /// Maps a range of a shader storage buffer for reading. glMemoryBarrier(BUFFER_UPDATE) first: without it the
    /// mapping need not see what the shaders stored (ES 3.2 section 7.11.2).
    Bytes read_buffer(mgl::BufferId buf, GLintptr offset, GLsizeiptr size)
    {
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glBindBuffer(kSsbo, buf);
        const void* p = mgl::glMapBufferRange(kSsbo, offset, size, mgl::MapBufferAccessMask::Read);
        if (p == nullptr) throw demo::Failure("glMapBufferRange(SHADER_STORAGE_BUFFER) failed");
        Bytes out(static_cast<std::size_t>(size));
        std::memcpy(out.data(), p, out.size());
        if (mgl::glUnmapBuffer(kSsbo) != GL_TRUE) throw demo::Failure("glUnmapBuffer returned FALSE");
        mgl::glBindBuffer(kSsbo, mgl::BufferId{0});
        return out;
    }

    void upload(mgl::BufferId buf, const Bytes& data)
    {
        mgl::glBindBuffer(kSsbo, buf);
        mgl::glBufferData(kSsbo, static_cast<GLsizeiptr>(data.size()), data.data(), mgl::BufferUsage::DynamicDraw);
        mgl::glBindBuffer(kSsbo, mgl::BufferId{0});
    }

    std::uint32_t word(const Bytes& b, std::size_t byte_offset)
    {
        std::uint32_t v = 0;
        std::memcpy(&v, b.data() + byte_offset, 4);
        return v;
    }
    void put_word(Bytes& b, std::size_t byte_offset, std::uint32_t v) { std::memcpy(b.data() + byte_offset, &v, 4); }
    void put_float(Bytes& b, std::size_t byte_offset, float v) { std::memcpy(b.data() + byte_offset, &v, 4); }

    std::uint8_t to8(float x) { return static_cast<std::uint8_t>(std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)); }

    bool near(Rgba8 got, Rgba8 want, int tol)
    {
        for (std::size_t i = 0; i < 4; ++i)
            if (std::abs(int{got[i]} - int{want[i]}) > tol) return false;
        return true;
    }

    constexpr Rgba8 kBackground = {14, 16, 22, 255};
    constexpr Rgba8 kGreen = {0, 255, 0, 255}, kMagenta = {255, 0, 255, 255}, kRed = {255, 0, 0, 255};

    // geometry (for the default 256x256 surface)
    constexpr int kColumns = 16;  // cells per row of the verification grid
    constexpr int kRows = 4;      // rows of the verification grid
    constexpr int kBands = 3;     // bar-chart bands below the grid

    /// A bound range of a runtime-sized-array buffer.
    struct Slot
    {
        GLintptr offset = 0;
        GLsizeiptr size = 0;
        int exact_len = -1;  ///< element count when (size - array offset) is a multiple of the stride; -1 otherwise
        int floor_len = 0;   ///< floor((size - array offset) / stride)
    };

    constexpr std::uint32_t kCanary = 0xC5C5C5C5u;
    constexpr std::uint32_t kTag = 0xC0FFEEu;
    constexpr int kSamplesHeader = 8, kRecordsHeader = 4, kRecordStride = 12;

    float sample_value(int i) { return static_cast<float>(1 + (i * 7) % 13); }

    // =========================================================================================
    // 3. The demo
    // =========================================================================================
    class SsboStd430 final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            alignment_ = demo::get_integer(mgl::GetParameter::ShaderStorageBufferOffsetAlignment);
            max_bindings_ = demo::get_integer(mgl::GetParameter::MaxShaderStorageBufferBindings);
            max_compute_blocks_ = demo::get_integer(mgl::GetParameter::MaxComputeShaderStorageBlocks);
            max_vertex_blocks_ = demo::get_integer(mgl::GetParameter::MaxVertexShaderStorageBlocks);
            max_fragment_blocks_ = demo::get_integer(mgl::GetParameter::MaxFragmentShaderStorageBlocks);
            max_combined_blocks_ = demo::get_integer(mgl::GetParameter::MaxCombinedShaderStorageBlocks);
            mgl::glGetInteger64v(mgl::GetParameter::MaxShaderStorageBlockSize, &max_block_size_);

            // Vertex and fragment shader storage blocks are optional in ES 3.1 (the minimum of both limits is 0): use them
            // only when the limits allow, and say so otherwise. Compute shaders always have at least 4.
            use_fragment_ = max_fragment_blocks_ >= 1 && max_combined_blocks_ >= 1;
            use_vertex_ = max_vertex_blocks_ >= 1 && max_combined_blocks_ >= 1;
            std::cout << std::format("   [note] shader storage limits: compute blocks {}, vertex blocks {}, fragment blocks {}, combined {}, bindings {}, "
                                     "offset alignment {}, max block size {}\n",
                                     max_compute_blocks_, max_vertex_blocks_, max_fragment_blocks_, max_combined_blocks_, max_bindings_, alignment_,
                                     max_block_size_);
            if (!use_fragment_)
                std::cout << "   [note] MAX_FRAGMENT_SHADER_STORAGE_BLOCKS is 0: the fragment-shader verification grid is skipped\n";
            if (!use_vertex_)
                std::cout << "   [note] MAX_VERTEX_SHADER_STORAGE_BLOCKS is 0: the vertex-shader bar charts are skipped\n";

            members_ = make_members();
            blocks_[0].compute(members_, Layout::Std430);
            blocks_[1].compute(members_, Layout::Std140);

            layout_prog_[0] = build_compute(env, "shaders/layout.comp", "LAYOUT std430");
            layout_prog_[1] = build_compute(env, "shaders/layout.comp", "LAYOUT std140");
            runtime_prog_ = build_compute(env, "shaders/runtime.comp");
            for (int li = 0; li < 2; ++li)
            {
                u_write_[li] = locate(layout_prog_[li], "u_write");
                const mgl::ProgramId p = layout_prog_[li];
                const GLuint block = mgl::glGetProgramResourceIndex(p, mgl::ProgramInterface::ShaderStorageBlock, "Probe");
                if (block == GL_INVALID_INDEX) throw demo::Failure("shader storage block Probe is not active");
                GLint size = 0;
                const mgl::ProgramResourceProperty prop = mgl::ProgramResourceProperty::BufferDataSize;
                mgl::glGetProgramResourceiv(p, mgl::ProgramInterface::ShaderStorageBlock, block, 1, &prop, 1, nullptr, &size);
                gl_data_size_[li] = size;
                // The buffer must be at least BUFFER_DATA_SIZE bytes (ES 3.2 section 7.8); never smaller than the calculator's size.
                data_size_[li] = std::max(size, align_up(blocks_[static_cast<std::size_t>(li)].end, 16));
            }
            if (use_fragment_) grid_prog_ = demo::build_program(env, "shaders/grid.vert", "shaders/grid.frag");
            if (use_vertex_) bars_prog_ = demo::build_program(env, "shaders/bars.vert", "shaders/bars.frag");
            if (use_fragment_)
            {
                u_cell_ = locate(grid_prog_, "u_cell");
                u_mask_ = locate(grid_prog_, "u_expect_mask");
            }
            if (use_vertex_) u_scale_ = locate(bars_prog_, "u_height_scale");

            mgl::glGenBuffers(static_cast<GLsizei>(bufs_.size()), bufs_.data());
            for (std::size_t i = 0; i < 4; ++i) upload(bufs_[kFlags + i], Bytes(32 * 4, 0));
            upload(bufs_[kProbe], Bytes(static_cast<std::size_t>(data_size_[0]), 0));
            upload(bufs_[kProbe + 1], Bytes(static_cast<std::size_t>(data_size_[1]), 0));

            // Ranges of the runtime-sized-array buffers: each range starts at a multiple of SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, and
            // 8 canary bytes (more with a large alignment) separate it from the next one.
            const std::array<int, 4> sample_lens = {32, 16, 4, -1};
            const std::array<int, 4> record_lens = {10, 5, 2, -1};
            sams_slots_ = make_slots(sample_lens, kSamplesHeader, 4, 3);
            recs_slots_ = make_slots(record_lens, kRecordsHeader, kRecordStride, 5);
            upload(bufs_[kSamples], canary_buffer(sams_slots_));
            upload(bufs_[kRecords], canary_buffer(recs_slots_));
            upload(bufs_[kScratch], Bytes(64, 0));
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            for (auto& r : rows_) r = {};
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kBackground[0] / 255.0f, kBackground[1] / 255.0f, kBackground[2] / 255.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            // The four verification rows: (program layout, packing layout).
            const std::array<std::pair<int, int>, 4> plan = {{{0, 0}, {1, 1}, {0, 1}, {1, 0}}};
            for (std::size_t row = 0; row < plan.size(); ++row)
            {
                const auto [prog, packed] = plan[row];
                Row& r = rows_[row];
                r.program = prog;
                r.packed = packed;
                r.expect_mask = predicted_mask(blocks_[static_cast<std::size_t>(prog)], blocks_[static_cast<std::size_t>(packed)]);
                Bytes data = pack(blocks_[static_cast<std::size_t>(packed)], static_cast<std::size_t>(data_size_[prog]));
                dispatch_probe(prog, data, false, bufs_[kFlags + row]);
                if (use_fragment_) draw_row(env, static_cast<int>(row), bufs_[kFlags + row], r.expect_mask);
                r.flags = read_flags(bufs_[kFlags + row]);
            }

            // The write direction: the shader fills a zeroed buffer.
            for (int li = 0; li < 2; ++li)
            {
                dispatch_probe(li, Bytes(static_cast<std::size_t>(data_size_[li]), 0), true, bufs_[kFlags]);
                write_img_[static_cast<std::size_t>(li)] = read_buffer(bufs_[kProbe + static_cast<std::size_t>(li)], 0, data_size_[li]);
            }

            run_runtime(env);

            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 4; ++i) mgl::glBindBufferBase(kSsbo, i, mgl::BufferId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_verification(check);
            check_write_direction(check);
            check_grid_pixels(env, check);
            check_runtime(check);
            check_bar_pixels(env, check);
            check_layout_introspection(check);
            check_runtime_introspection(check);
            check_graphics_introspection(check);
            check_bindings_and_errors(check);
            check_limits(check);
            negative_controls(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 4; ++i) mgl::glBindBufferBase(kSsbo, i, mgl::BufferId{0});
            mgl::glBindBuffer(kSsbo, mgl::BufferId{0});
            mgl::glDeleteBuffers(static_cast<GLsizei>(bufs_.size()), bufs_.data());
            mgl::glDeleteProgram(layout_prog_[0]);
            mgl::glDeleteProgram(layout_prog_[1]);
            mgl::glDeleteProgram(runtime_prog_);
            if (grid_prog_.value != 0) mgl::glDeleteProgram(grid_prog_);
            if (bars_prog_.value != 0) mgl::glDeleteProgram(bars_prog_);
        }

    private:
        // Buffer slots.
        static constexpr std::size_t kProbe = 0 /* 0,1 */, kFlags = 2 /* 2..5 */, kSamples = 6, kRecords = 7, kScratch = 8, kBufferCount = 9;

        struct Row
        {
            int program = 0;            ///< layout of the declared block (0 std430, 1 std140)
            int packed = 0;             ///< layout the buffer was packed with
            std::uint32_t expect_mask = 0;
            std::vector<std::uint32_t> flags;
        };

        // ----------------------------------------------------------------------------------
        // compute passes
        // ----------------------------------------------------------------------------------
        /// One dispatch of layout.comp: VERIFY (write == false) reads `data` and fills `flags`, WRITE stores the values.
        void dispatch_probe(int li, const Bytes& data, bool write, mgl::BufferId flags)
        {
            const auto idx = static_cast<std::size_t>(li);
            upload(bufs_[kProbe + idx], data);
            mgl::glBindBufferBase(kSsbo, 0, bufs_[kProbe + idx]);  // layout(binding = 0) buffer Probe
            mgl::glBindBufferBase(kSsbo, 1, flags);                // layout(binding = 1) buffer Result
            mgl::glUseProgram(layout_prog_[li]);
            mgl::glUniform1i(u_write_[li], write ? 1 : 0);
            mgl::glDispatchCompute(1, 1, 1);
            // The consumers below are a fragment shader (SHADER_STORAGE_BARRIER_BIT) and buffer mapping (BUFFER_UPDATE_BARRIER_BIT).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage | mgl::MemoryBarrierMask::BufferUpdate);
        }

        std::vector<std::uint32_t> read_flags(mgl::BufferId flags)
        {
            const Bytes raw = read_buffer(flags, 0, 32 * 4);
            std::vector<std::uint32_t> out(32);
            std::memcpy(out.data(), raw.data(), raw.size());
            return out;
        }

        void draw_row(const demo::Env& env, int row, mgl::BufferId flags, std::uint32_t mask)
        {
            const int cell = env.width / kColumns;
            mgl::glUseProgram(grid_prog_);
            mgl::glUniform1i(u_cell_, cell);
            mgl::glUniform1ui(u_mask_, mask);
            mgl::glBindBufferBase(kSsbo, 0, flags);  // layout(binding = 0) readonly buffer Flags
            mgl::glViewport(0, env.height - (row + 1) * cell, env.width, cell);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        std::vector<Slot> make_slots(const std::array<int, 4>& lens, int header, int stride, int partial_bytes) const
        {
            std::vector<Slot> slots;
            GLintptr offset = 0;
            for (int len : lens)
            {
                Slot s;
                s.offset = offset;
                // len < 0: a range whose size is NOT header + n * stride, so the shader's length() involves a rounding rule.
                const int n = len >= 0 ? len : 7;
                s.size = header + n * stride + (len >= 0 ? 0 : partial_bytes);
                s.exact_len = len >= 0 ? len : -1;
                s.floor_len = n;
                slots.push_back(s);
                offset = align_up(static_cast<int>(offset + s.size + 8), std::max(alignment_, 1));
            }
            return slots;
        }

        Bytes canary_buffer(const std::vector<Slot>& slots) const
        {
            const GLsizeiptr total = slots.back().offset + slots.back().size + 8;
            Bytes b(static_cast<std::size_t>(align_up(static_cast<int>(total), 4)), 0);
            for (std::size_t i = 0; i < b.size(); i += 4) put_word(b, i, kCanary);
            return b;
        }

        void run_runtime(const demo::Env& env)
        {
            mgl::glUseProgram(runtime_prog_);
            for (std::size_t k = 0; k < sams_slots_.size(); ++k)
            {
                // The block's backing store is the RANGE (offset, size), not the buffer: length() follows the size.
                mgl::glBindBufferRange(kSsbo, 0, bufs_[kSamples], sams_slots_[k].offset, sams_slots_[k].size);
                mgl::glBindBufferRange(kSsbo, 1, bufs_[kRecords], recs_slots_[k].offset, recs_slots_[k].size);
                mgl::glDispatchCompute(2, 1, 1);
            }
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage | mgl::MemoryBarrierMask::BufferUpdate);

            if (use_vertex_)
            {
                const int cell = env.width / kColumns;
                const int band = 3 * cell;
                mgl::glUseProgram(bars_prog_);
                mgl::glUniform1f(u_scale_, 1.0f / 16.0f);
                for (int k = 0; k < kBands; ++k)
                {
                    const Slot& s = sams_slots_[static_cast<std::size_t>(k)];
                    mgl::glBindBufferRange(kSsbo, 0, bufs_[kSamples], s.offset, s.size);
                    mgl::glViewport(0, env.height - kRows * cell - (k + 1) * band, env.width, band);
                    mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, s.exact_len);
                }
            }
            sams_img_ = read_buffer(bufs_[kSamples], 0, static_cast<GLsizeiptr>(canary_buffer(sams_slots_).size()));
            recs_img_ = read_buffer(bufs_[kRecords], 0, static_cast<GLsizeiptr>(canary_buffer(recs_slots_).size()));
        }

        // ----------------------------------------------------------------------------------
        // checks: the two directions
        // ----------------------------------------------------------------------------------
        static std::string mask_string(std::uint32_t mask, const std::vector<std::uint32_t>& flags)
        {
            std::string s;
            for (int m = 1; m <= 16; ++m)
                if (flags[static_cast<std::size_t>(m)] != ((mask >> m) & 1u)) s += std::format("{} ", m);
            return s;
        }

        void check_verification(demo::Check& check)
        {
            static constexpr const char* titles[4] = {
                "row 0: block declared std430, buffer packed with the std430 calculator",
                "row 1: block declared std140, buffer packed with the std140 calculator",
                "row 2: block declared std430, buffer packed with the STD140 layout (a mistake)",
                "row 3: block declared std140, buffer packed with the STD430 layout (a mistake)"};
            for (std::size_t i = 0; i < 4; ++i)
            {
                const Row& r = rows_[i];
                const std::string bad = mask_string(r.expect_mask, r.flags);
                int pass = 0;
                for (int m = 1; m <= 16; ++m) pass += static_cast<int>(r.flags[static_cast<std::size_t>(m)] == 1);
                if (i < 2)
                    check.expect(r.expect_mask == 0x1FFFEu && bad.empty() && pass == 16,
                                 std::format("{}: all 16 members read by the compute shader equal the packed values", titles[i]),
                                 bad.empty() ? "" : "members that disagree with the calculator: " + bad);
                else
                    check.expect(bad.empty() && pass == static_cast<int>(std::popcount(r.expect_mask)) && pass >= 4 && pass < 16,
                                 std::format("{}: exactly the predicted members ({} of 16, those whose scalars sit at the same offsets in both layouts) verify", titles[i],
                                             std::popcount(r.expect_mask)),
                                 bad.empty() ? "" : "members that differ from the prediction: " + bad);
            }
            // The flags of the right and of the wrong layouts for one program must differ exactly where the calculator says.
            std::cout << std::format("   [note] members that survive reading std140 data through a std430 block: mask {:#x}; through the reverse: {:#x}\n",
                                     rows_[2].expect_mask, rows_[3].expect_mask);
        }

        void check_write_direction(demo::Check& check)
        {
            for (std::size_t li = 0; li < 2; ++li)
            {
                const Bytes expected = pack(blocks_[li], static_cast<std::size_t>(data_size_[li]));
                const ImageDiff d = diff_images(blocks_[li], write_img_[li], expected);
                check.expect(write_img_[li].size() == expected.size() && d.member_words == 0,
                             std::format("{}: every member the compute shader stored sits exactly where the calculator says (all {} scalars of the 16 members, read back with glMapBufferRange)", li == 0 ? "std430" : "std140",
                                         [&] { std::size_t n = 0; for (const Leaf& l : blocks_[li].leaves) n += static_cast<std::size_t>(l.count * l.type.components()); return n; }()),
                             d.member_words == 0 ? "" : std::format("{} member words differ, first at byte {}", d.member_words, d.first_member));
                std::cout << std::format("   [note] {}: the shader changed {} padding word(s) of the zero-initialised buffer (the spec does not say padding is preserved; it is not asserted)\n",
                                         li == 0 ? "std430" : "std140", d.padding_words);
            }
            // Negative control: the std430 result must NOT match the std140 image at the std430 member positions.
            const Bytes wrong = pack(blocks_[1], static_cast<std::size_t>(data_size_[0]));
            const ImageDiff d = diff_images(blocks_[0], write_img_[0], wrong);
            check.expect(d.member_words > 0 && d.first_member == 40,
                         "negative control: compared with a std140-packed image at the std430 member positions, the std430 write result differs, first at farr[0] (offset 40 in std430, 48 in std140)",
                         std::format("{} member words differ, first at byte {}", d.member_words, d.first_member));
        }

        // ----------------------------------------------------------------------------------
        // checks: what the fragment and vertex shaders drew
        // ----------------------------------------------------------------------------------
        void check_grid_pixels(const demo::Env& env, demo::Check& check)
        {
            if (!use_fragment_)
            {
                std::cout << "   [note] fragment-shader grid not drawn (no fragment shader storage blocks): its pixel checks are skipped\n";
                return;
            }
            const int cell = env.width / kColumns;
            for (std::size_t row = 0; row < 4; ++row)
            {
                std::string bad;
                for (int m = 1; m <= 16; ++m)
                {
                    const int x = (m - 1) * cell + cell / 2;
                    const int y = env.height - (static_cast<int>(row) + 1) * cell + cell / 2;
                    const Rgba8 got = demo::read_pixel(x, y);
                    const bool expected_pass = ((rows_[row].expect_mask >> m) & 1u) != 0;
                    const Rgba8 want = expected_pass ? kGreen : kMagenta;
                    if (!near(got, want, 0)) bad += std::format("member {} is {} (want {}) ", m, demo::format_rgba(got), demo::format_rgba(want));
                }
                check.expect(bad.empty(),
                             std::format("fragment shader reading the flag buffer: row {} cells are green (member verified) / magenta (predicted mismatch), never red", row), bad);
            }
            const Rgba8 gap = demo::read_pixel(env.width / 2, 8);
            check.expect(near(gap, kBackground, 0), "the area below the bar charts keeps the clear colour (nothing leaked)", demo::format_rgba(demo::read_pixel(env.width / 2, 8)));
        }

        void check_bar_pixels(const demo::Env& env, demo::Check& check)
        {
            if (!use_vertex_)
            {
                std::cout << "   [note] vertex-shader bar charts not drawn (no vertex shader storage blocks): their pixel checks are skipped\n";
                return;
            }
            const int cell = env.width / kColumns;
            const int band = 3 * cell;
            for (int k = 0; k < kBands; ++k)
            {
                const int len = sams_slots_[static_cast<std::size_t>(k)].exact_len;
                const int y0 = env.height - kRows * cell - (k + 1) * band;
                const int bar_px = env.width / len;
                std::string bad;
                for (int i = 0; i < len; ++i)
                {
                    const int v = static_cast<int>(sample_value(i));
                    const int height = v * band / 16;  // the vertex shader scales by 1/16 of the viewport height
                    const int x = i * bar_px + bar_px / 2;
                    const Rgba8 inside = demo::read_pixel(x, y0 + height / 2);
                    const Rgba8 above = demo::read_pixel(x, y0 + height + 2);
                    const float fv = static_cast<float>(v);
                    const Rgba8 want = {to8(fv / 13.0f), to8(1.0f - fv / 13.0f), to8(0.4f), 255};
                    if (!near(inside, want, 1)) bad += std::format("bar {} inside {} want {}; ", i, demo::format_rgba(inside), demo::format_rgba(want));
                    if (!near(above, kBackground, 0)) bad += std::format("bar {} above {}; ", i, demo::format_rgba(above));
                }
                check.expect(bad.empty(),
                             std::format("vertex shader reading values[]: band {} draws {} bars of {} px, each as high as its value (colour within 1 of the 8-bit quantisation)", k, len, bar_px), bad);
            }
        }

        // ----------------------------------------------------------------------------------
        // checks: runtime-sized arrays
        // ----------------------------------------------------------------------------------
        void check_runtime(demo::Check& check)
        {
            // What each range must contain: the shader writes the header and the first `n` elements, nothing else.
            auto build = [](const std::vector<Slot>& slots, const Bytes& actual, bool samples) {
                Bytes expected(actual.size(), 0);
                for (std::size_t i = 0; i < expected.size(); i += 4) put_word(expected, i, kCanary);
                std::vector<int> lens;
                for (const Slot& s : slots)
                {
                    const auto off = static_cast<std::size_t>(s.offset);
                    const std::uint32_t reported = word(actual, off);  // length() as the shader saw it
                    const int n = static_cast<int>(reported);
                    lens.push_back(n);
                    put_word(expected, off, reported);
                    if (samples)
                    {
                        put_word(expected, off + 4, kTag);
                        for (int i = 0; i < n; ++i) put_float(expected, off + kSamplesHeader + 4 * static_cast<std::size_t>(i), sample_value(i));
                    }
                    else
                    {
                        for (int i = 0; i < n; ++i)
                        {
                            const auto at = off + kRecordsHeader + kRecordStride * static_cast<std::size_t>(i);
                            put_float(expected, at, static_cast<float>(i));
                            put_float(expected, at + 4, static_cast<float>(2 * i + 1));
                            put_float(expected, at + 8, static_cast<float>(3 * i + 2));
                        }
                    }
                }
                return std::pair{expected, lens};
            };
            auto report = [&](const char* what, const std::vector<Slot>& slots, const Bytes& actual, bool samples) {
                const auto [expected, lens] = build(slots, actual, samples);
                std::string len_bad;
                for (std::size_t k = 0; k < slots.size(); ++k)
                {
                    const int exact = slots[k].exact_len;
                    if (exact >= 0 && lens[k] != exact) len_bad += std::format("slot {}: length() = {}, expected {}; ", k, lens[k], exact);
                    if (exact < 0 && (lens[k] > slots[k].floor_len || lens[k] < 0)) len_bad += std::format("slot {}: length() = {} exceeds floor {}; ", k, lens[k], slots[k].floor_len);
                }
                check.expect(len_bad.empty(),
                             std::format("{}: length() of the runtime-sized array equals (bound size - offset of the array) / stride for the ranges whose size is an exact multiple: {}, {}, {} elements", what,
                                         slots[0].exact_len, slots[1].exact_len, slots[2].exact_len), len_bad);
                const int partial = lens.back();
                std::cout << std::format("   [note] {}: a range of {} bytes (not a multiple of the stride) gave length() = {} (floor would be {}){}\n", what,
                                         slots.back().size, partial, slots.back().floor_len, partial == slots.back().floor_len ? " - exactly the floor" : "");
                std::size_t diff = 0, first = actual.size();
                for (std::size_t i = 0; i < actual.size(); ++i)
                    if (actual[i] != expected[i])
                    {
                        if (first == actual.size()) first = i;
                        ++diff;
                    }
                check.expect(diff == 0,
                             std::format("{}: the shader wrote header + length() elements in each of the 4 ranges, and every byte outside them (alignment gaps, bytes past each range, the partial tail) is untouched",
                                         what),
                             diff == 0 ? "" : std::format("{} bytes differ, first at offset {}", diff, first));
            };
            report("block Samples { uint, uint, float values[] }", sams_slots_, sams_img_, true);
            report("block Records { uint, Rec recs[] } (stride 12)", recs_slots_, recs_img_, false);
            check.expect(alignment_ <= 1 || (sams_slots_[1].offset % alignment_ == 0 && recs_slots_[2].offset % alignment_ == 0),
                         std::format("every glBindBufferRange offset used (e.g. {}, {}) is a multiple of SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT = {}", sams_slots_[1].offset, recs_slots_[2].offset, alignment_));
        }

        // ----------------------------------------------------------------------------------
        // introspection
        // ----------------------------------------------------------------------------------
        struct Var
        {
            GLuint index = GL_INVALID_INDEX;
            GLint type = -1, array_size = -1, offset = -1, block_index = -1, array_stride = -1, matrix_stride = -1, row_major = -1, top_size = -1,
                  top_stride = -1;
        };

        static Var query_var(mgl::ProgramId prog, const std::string& name)
        {
            Var q;
            q.index = mgl::glGetProgramResourceIndex(prog, mgl::ProgramInterface::BufferVariable, name.c_str());
            if (q.index == GL_INVALID_INDEX) return q;
            static constexpr std::array<mgl::ProgramResourceProperty, 9> props = {
                mgl::ProgramResourceProperty::Type,          mgl::ProgramResourceProperty::ArraySize,     mgl::ProgramResourceProperty::Offset,
                mgl::ProgramResourceProperty::BlockIndex,    mgl::ProgramResourceProperty::ArrayStride,   mgl::ProgramResourceProperty::MatrixStride,
                mgl::ProgramResourceProperty::IsRowMajor,    mgl::ProgramResourceProperty::TopLevelArraySize,
                mgl::ProgramResourceProperty::TopLevelArrayStride};
            std::array<GLint, 9> v{};
            GLsizei length = 0;
            mgl::glGetProgramResourceiv(prog, mgl::ProgramInterface::BufferVariable, q.index, static_cast<GLsizei>(props.size()), props.data(),
                                        static_cast<GLsizei>(v.size()), &length, v.data());
            q.type = v[0];
            q.array_size = v[1];
            q.offset = v[2];
            q.block_index = v[3];
            q.array_stride = v[4];
            q.matrix_stride = v[5];
            q.row_major = v[6];
            q.top_size = v[7];
            q.top_stride = v[8];
            return q;
        }

        static GLint resource_prop(mgl::ProgramId prog, mgl::ProgramInterface iface, GLuint index, mgl::ProgramResourceProperty prop)
        {
            GLint v = -12345;
            mgl::glGetProgramResourceiv(prog, iface, index, 1, &prop, 1, nullptr, &v);
            return v;
        }

        static GLint interface_prop(mgl::ProgramId prog, mgl::ProgramInterface iface, mgl::ProgramInterfaceParameter pname)
        {
            GLint v = -12345;
            mgl::glGetProgramInterfaceiv(prog, iface, pname, &v);
            return v;
        }

        void check_layout_introspection(demo::Check& check)
        {
            std::array<std::vector<Var>, 2> vars;
            std::array<std::vector<const Leaf*>, 2> listed;
            for (std::size_t li = 0; li < 2; ++li)
            {
                const char* which = li == 0 ? "std430" : "std140";
                const mgl::ProgramId prog = layout_prog_[li];
                const Block& block = blocks_[li];
                listed[li] = block.listed();
                const std::size_t n = listed[li].size();  // 18: pairs[1].* is not listed
                const GLuint block_index = mgl::glGetProgramResourceIndex(prog, mgl::ProgramInterface::ShaderStorageBlock, "Probe");
                for (const Leaf* l : listed[li]) vars[li].push_back(query_var(prog, l->name));

                // Every variable: type, array size, offset, strides, row-major flag, owning block.
                std::string bad;
                std::set<GLuint> indices;
                for (std::size_t i = 0; i < n; ++i)
                {
                    const Leaf& l = *listed[li][i];
                    const Var& q = vars[li][i];
                    if (q.index == GL_INVALID_INDEX)
                    {
                        bad += l.name + " (not active) ";
                        continue;
                    }
                    indices.insert(q.index);
                    std::string what;
                    if (static_cast<GLenum>(q.type) != static_cast<GLenum>(l.type.gl_type())) what += std::format(" type {:#x}", q.type);
                    if (q.array_size != l.count) what += std::format(" array_size {} (want {})", q.array_size, l.count);
                    if (q.offset != l.offset) what += std::format(" OFFSET {} (want {})", q.offset, l.offset);
                    if (q.array_stride != l.array_stride) what += std::format(" ARRAY_STRIDE {} (want {})", q.array_stride, l.array_stride);
                    if (q.matrix_stride != l.matrix_stride) what += std::format(" MATRIX_STRIDE {} (want {})", q.matrix_stride, l.matrix_stride);
                    if ((q.row_major != 0) != l.row_major) what += std::format(" IS_ROW_MAJOR {}", q.row_major);
                    if (static_cast<GLuint>(q.block_index) != block_index) what += std::format(" BLOCK_INDEX {}", q.block_index);
                    if (!what.empty()) bad += l.name + ":" + what + "; ";
                }
                check.expect(bad.empty(),
                             std::format("{} block Probe: OFFSET, ARRAY_STRIDE, MATRIX_STRIDE, IS_ROW_MAJOR, ARRAY_SIZE, TYPE and BLOCK_INDEX of the {} listed buffer variables equal the CPU calculator", which, n),
                             bad);

                // Top-level arrays of structures are listed by their FIRST element only (ES 3.2 section 7.3.1.1); the rest is reached through
                // TOP_LEVEL_ARRAY_SIZE and TOP_LEVEL_ARRAY_STRIDE (section 7.3.1: the stride is non-zero only for an array of an aggregate type).
                const Var absent = query_var(prog, "Probe.pairs[1].a");
                check.expect(absent.index == GL_INVALID_INDEX && vars[li][13].index != GL_INVALID_INDEX,
                             std::format("{} block Probe: \"Probe.pairs[0].a\" is an active buffer variable, \"Probe.pairs[1].a\" is not (a top-level array of structures is listed by its first element only)", which));
                std::string top_bad;
                for (std::size_t i = 0; i < n; ++i)
                {
                    const Leaf& l = *listed[li][i];
                    const Var& q = vars[li][i];
                    if (q.index == GL_INVALID_INDEX) continue;
                    if (q.top_stride != l.top_stride || q.top_size != l.top_size)
                        top_bad += std::format("{}: TOP_LEVEL_ARRAY_STRIDE {} (want {}), TOP_LEVEL_ARRAY_SIZE {} (want {}); ", l.name, q.top_stride, l.top_stride, q.top_size, l.top_size);
                }
                check.expect(top_bad.empty(),
                             std::format("{} block Probe: TOP_LEVEL_ARRAY_STRIDE is the structure size ({}) for pairs[] and 0 for everything else, TOP_LEVEL_ARRAY_SIZE is the element count (2) for pairs[] and 1 for the rest",
                                         which, li == 0 ? 8 : 16),
                             top_bad);

                // Names round-trip.
                std::string name_bad;
                for (std::size_t i = 0; i < n; ++i)
                {
                    char buf[64] = {};
                    GLsizei len = 0;
                    if (vars[li][i].index == GL_INVALID_INDEX) continue;
                    mgl::glGetProgramResourceName(prog, mgl::ProgramInterface::BufferVariable, vars[li][i].index, sizeof(buf), &len, buf);
                    if (listed[li][i]->name != buf || len != static_cast<GLsizei>(listed[li][i]->name.size())) name_bad += listed[li][i]->name + " -> " + buf + "; ";
                }
                check.expect(name_bad.empty(), std::format("{} block Probe: glGetProgramResourceName returns the names the calculator predicts (arrays as \"Probe.farr[0]\", members of structures as \"Probe.combo.v\")", which), name_bad);

                // Block level.
                const std::array<mgl::ProgramResourceProperty, 6> bprops = {
                    mgl::ProgramResourceProperty::BufferBinding, mgl::ProgramResourceProperty::BufferDataSize, mgl::ProgramResourceProperty::NumActiveVariables,
                    mgl::ProgramResourceProperty::ReferencedByComputeShader, mgl::ProgramResourceProperty::ReferencedByVertexShader,
                    mgl::ProgramResourceProperty::ReferencedByFragmentShader};
                std::array<GLint, 6> b{};
                mgl::glGetProgramResourceiv(prog, mgl::ProgramInterface::ShaderStorageBlock, block_index, static_cast<GLsizei>(bprops.size()), bprops.data(),
                                            static_cast<GLsizei>(b.size()), nullptr, b.data());
                check.expect(b[0] == 0 && b[2] == static_cast<GLint>(n) && b[3] == 1 && b[4] == 0 && b[5] == 0,
                             std::format("{} block Probe: BUFFER_BINDING = 0 (layout(binding = 0)), NUM_ACTIVE_VARIABLES = {}, referenced by the compute shader only", which, n),
                             std::format("binding {}, variables {}, compute {} vertex {} fragment {}", b[0], b[2], b[3], b[4], b[5]));
                check.expect(b[1] >= block.end,
                             std::format("{} block Probe: BUFFER_DATA_SIZE ({}) is large enough to hold all variables (>= {}, the end of the last member)", which, b[1], block.end),
                             std::format("{}{}", b[1], b[1] == block.end ? " = exactly the end of the last member" : " (rounded up by the implementation)"));
                std::vector<GLint> active(static_cast<std::size_t>(std::max(b[2], 0)), -1);
                const mgl::ProgramResourceProperty av = mgl::ProgramResourceProperty::ActiveVariables;
                mgl::glGetProgramResourceiv(prog, mgl::ProgramInterface::ShaderStorageBlock, block_index, 1, &av, static_cast<GLsizei>(active.size()), nullptr, active.data());
                const std::set<GLuint> from_block(active.begin(), active.end());
                check.expect(from_block == indices && from_block.size() == n, std::format("{} block Probe: ACTIVE_VARIABLES lists exactly the {} buffer variable indices found by name", which, n));

                // Interface level.
                const GLint blocks = interface_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, mgl::ProgramInterfaceParameter::ActiveResources);
                const GLint name_max = interface_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, mgl::ProgramInterfaceParameter::MaxNameLength);
                const GLint vars_max = interface_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, mgl::ProgramInterfaceParameter::MaxNumActiveVariables);
                const GLint buffer_vars = interface_prop(prog, mgl::ProgramInterface::BufferVariable, mgl::ProgramInterfaceParameter::ActiveResources);
                check.expect(blocks == 2 && name_max == 7 && vars_max == static_cast<GLint>(n) && buffer_vars == static_cast<GLint>(n) + 1,
                             std::format("{} program: SHADER_STORAGE_BLOCK interface has 2 active resources (Probe, Result), MAX_NAME_LENGTH 7, MAX_NUM_ACTIVE_VARIABLES {}; BUFFER_VARIABLE has {} (the {} of Probe + Result.flags[0])", which,
                                         n, n + 1, n),
                             std::format("{} {} {} {}", blocks, name_max, vars_max, buffer_vars));

                // Result block: a runtime-sized array of uint.
                const Var rv = query_var(prog, "Result.flags[0]");
                const GLuint result_block = mgl::glGetProgramResourceIndex(prog, mgl::ProgramInterface::ShaderStorageBlock, "Result");
                check.expect(rv.index != GL_INVALID_INDEX && rv.array_size == 0 && rv.offset == 0 && rv.array_stride == 4 && rv.top_size == 1 && rv.top_stride == 0 &&
                                 resource_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, result_block, mgl::ProgramResourceProperty::BufferBinding) == 1,
                             std::format("{} program: Result.flags[] (runtime-sized): ARRAY_SIZE 0 (\"not determined at link time\"), OFFSET 0, ARRAY_STRIDE 4, TOP_LEVEL_ARRAY_SIZE 1, block bound to binding 1", which),
                             std::format("array_size {} offset {} stride {} top {} {}", rv.array_size, rv.offset, rv.array_stride, rv.top_size, rv.top_stride));
            }

            // The difference between the layouts, as the GL reports it (and as derived by hand from the rules).
            // Listed order: f v3 after_v3 v2 farr v2arr v3arr i uv2 m2 m3 m3x2 m2x3r pairs[0].a pairs[0].b combo.f combo.v tail
            const auto off = [&](std::size_t li, int v) { return vars[li][static_cast<std::size_t>(v)].offset; };
            const auto stride = [&](std::size_t li, int v) { return vars[li][static_cast<std::size_t>(v)].array_stride; };
            const auto mstride = [&](std::size_t li, int v) { return vars[li][static_cast<std::size_t>(v)].matrix_stride; };
            const auto top = [&](std::size_t li, int v) { return vars[li][static_cast<std::size_t>(v)].top_stride; };
            const bool hand =
                off(0, 4) == 40 && stride(0, 4) == 4 && off(1, 4) == 48 && stride(1, 4) == 16 &&               // float farr[3]
                off(0, 5) == 56 && stride(0, 5) == 8 && off(1, 5) == 96 && stride(1, 5) == 16 &&               // vec2 v2arr[3]
                off(0, 6) == 80 && stride(0, 6) == 16 && off(1, 6) == 144 && stride(1, 6) == 16 &&             // vec3 v3arr[2]
                off(0, 9) == 128 && mstride(0, 9) == 8 && off(1, 9) == 192 && mstride(1, 9) == 16 &&           // mat2
                off(0, 10) == 144 && mstride(0, 10) == 16 && off(1, 10) == 224 && mstride(1, 10) == 16 &&      // mat3
                off(0, 11) == 192 && mstride(0, 11) == 8 && off(1, 11) == 272 && mstride(1, 11) == 16 &&       // mat3x2
                off(0, 12) == 216 && mstride(0, 12) == 8 && off(1, 12) == 320 && mstride(1, 12) == 16 &&       // row_major mat2x3
                off(0, 13) == 240 && top(0, 13) == 8 && off(1, 13) == 368 && top(1, 13) == 16 &&               // pairs[].a and the structure stride
                off(0, 15) == 256 && off(0, 16) == 272 && off(1, 15) == 400 && off(1, 16) == 416 &&            // combo.f, combo.v
                off(0, 17) == 288 && off(1, 17) == 432;                                                         // tail
            check.expect(hand,
                         "std430 vs std140, derived by hand from ES 3.2 section 7.6.2.2 and compared with the GL: farr[3] 40/stride 4 vs 48/16, v2arr[3] 56/8 vs 96/16, v3arr[2] 80/16 vs 144/16, "
                         "mat2 128/column stride 8 vs 192/16, mat3x2 192/8 vs 272/16, row_major mat2x3 216/8 vs 320/16, pairs[] 240/structure stride 8 vs 368/16, combo 256 vs 400, tail 288 vs 432");
            // The same table, measured, as a note.
            std::cout << "   [note] member            std430 offset/stride        std140 offset/stride\n";
            for (std::size_t i = 0; i < listed[0].size(); ++i)
            {
                const Var& a = vars[0][i];
                const Var& b = vars[1][i];
                auto st = [](const Var& v) { return v.array_stride > 0 ? v.array_stride : v.matrix_stride > 0 ? v.matrix_stride : v.top_stride; };
                std::cout << std::format("   [note] {:<16} {:>6} / {:<3}                 {:>6} / {:<3}{}\n", listed[0][i]->name.substr(6), a.offset, st(a), b.offset, st(b),
                                         (a.offset != b.offset || st(a) != st(b)) ? "   <- differs" : "");
            }
            // Negative control for the comparison itself: the GL's std430 numbers must disagree with the std140 calculator exactly where the two layouts differ.
            int mismatches = 0, differ = 0;
            for (std::size_t i = 0; i < listed[1].size(); ++i)
            {
                if (vars[0][i].offset != listed[1][i]->offset) ++mismatches;
                if (listed[0][i]->offset != listed[1][i]->offset) ++differ;
            }
            check.expect(mismatches == differ && mismatches == 14,
                         "negative control: comparing the GL's std430 offsets with the std140 calculator reports a mismatch for every variable from farr[0] on (14 of 18), exactly where the two calculators differ",
                         std::format("{} mismatches, calculators differ in {}", mismatches, differ));
        }

        void check_runtime_introspection(demo::Check& check)
        {
            const mgl::ProgramId prog = runtime_prog_;
            const Var values = query_var(prog, "Samples.values[0]");
            const Var ra = query_var(prog, "Records.recs[0].a"), rb = query_var(prog, "Records.recs[0].b"), rc = query_var(prog, "Records.recs[0].c");
            check.expect(values.index != GL_INVALID_INDEX && values.offset == 8 && values.array_stride == 4 && values.array_size == 0 && values.top_stride == 0,
                         "Samples.values[]: OFFSET 8 (after two uints), ARRAY_STRIDE 4, ARRAY_SIZE 0 (size not determined at link time), TOP_LEVEL_ARRAY_STRIDE 0 (an array of a basic type)",
                         std::format("offset {} stride {} size {} top stride {}", values.offset, values.array_stride, values.array_size, values.top_stride));
            check.expect(ra.index != GL_INVALID_INDEX && ra.offset == 4 && rb.offset == 8 && rc.offset == 12 && ra.top_stride == kRecordStride && ra.top_size == 0 && ra.array_stride == 0,
                         "Records.recs[]: a std430 structure of three floats has alignment 4, so recs[0] starts at offset 4 (a, b, c at 4, 8, 12) and TOP_LEVEL_ARRAY_STRIDE is 12; TOP_LEVEL_ARRAY_SIZE is 0 for a runtime-sized array of structures",
                         std::format("offsets {} {} {}, top stride {}, top size {}", ra.offset, rb.offset, rc.offset, ra.top_stride, ra.top_size));
            const GLuint samples = mgl::glGetProgramResourceIndex(prog, mgl::ProgramInterface::ShaderStorageBlock, "Samples");
            const GLuint records = mgl::glGetProgramResourceIndex(prog, mgl::ProgramInterface::ShaderStorageBlock, "Records");
            const GLint s_size = resource_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, samples, mgl::ProgramResourceProperty::BufferDataSize);
            const GLint r_size = resource_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, records, mgl::ProgramResourceProperty::BufferDataSize);
            check.expect(s_size >= 12 && r_size >= 16,
                         "BUFFER_DATA_SIZE of a block ending in a runtime-sized array is computed as if the array had ONE element: Samples >= 8 + 4 = 12, Records >= 4 + 12 = 16",
                         std::format("Samples {}, Records {}", s_size, r_size));
            check.expect(resource_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, samples, mgl::ProgramResourceProperty::BufferBinding) == 0 &&
                             resource_prop(prog, mgl::ProgramInterface::ShaderStorageBlock, records, mgl::ProgramResourceProperty::BufferBinding) == 1,
                         "BUFFER_BINDING of Samples is 0 and of Records is 1 (layout(binding = N)); there is no OpenGL ES command to change a shader storage block's binding after linking");
            std::cout << std::format("   [note] BUFFER_DATA_SIZE: Samples {} (one element = 12), Records {} (one element = 16); Probe(std430) {}, Probe(std140) {}\n", s_size, r_size,
                                     gl_data_size_[0], gl_data_size_[1]);
        }

        void check_graphics_introspection(demo::Check& check)
        {
            if (use_fragment_)
            {
                const GLuint flags = mgl::glGetProgramResourceIndex(grid_prog_, mgl::ProgramInterface::ShaderStorageBlock, "Flags");
                check.expect(flags != GL_INVALID_INDEX &&
                                 resource_prop(grid_prog_, mgl::ProgramInterface::ShaderStorageBlock, flags, mgl::ProgramResourceProperty::ReferencedByFragmentShader) == 1 &&
                                 resource_prop(grid_prog_, mgl::ProgramInterface::ShaderStorageBlock, flags, mgl::ProgramResourceProperty::ReferencedByVertexShader) == 0 &&
                                 resource_prop(grid_prog_, mgl::ProgramInterface::ShaderStorageBlock, flags, mgl::ProgramResourceProperty::ReferencedByComputeShader) == 0,
                             "grid program: block Flags is referenced by the fragment shader only");
            }
            if (use_vertex_)
            {
                const GLuint samples = mgl::glGetProgramResourceIndex(bars_prog_, mgl::ProgramInterface::ShaderStorageBlock, "Samples");
                check.expect(samples != GL_INVALID_INDEX &&
                                 resource_prop(bars_prog_, mgl::ProgramInterface::ShaderStorageBlock, samples, mgl::ProgramResourceProperty::ReferencedByVertexShader) == 1 &&
                                 resource_prop(bars_prog_, mgl::ProgramInterface::ShaderStorageBlock, samples, mgl::ProgramResourceProperty::ReferencedByFragmentShader) == 0,
                             "bars program: block Samples is referenced by the vertex shader only");
            }
        }

        // ----------------------------------------------------------------------------------
        // indexed binding state and errors
        // ----------------------------------------------------------------------------------
        static GLint get_i(mgl::GetParameter p, GLuint index)
        {
            GLint v = -12345;
            mgl::glGetIntegeri_v(p, index, &v);
            return v;
        }
        static GLint64 get_i64(mgl::GetParameter p, GLuint index)
        {
            GLint64 v = -12345;
            mgl::glGetInteger64i_v(p, index, &v);
            return v;
        }

        void check_bindings_and_errors(demo::Check& check)
        {
            const mgl::BufferId probe = bufs_[kProbe], sams = bufs_[kSamples];
            const Slot& slot = sams_slots_[1];

            check.expect_no_gl_error("before the binding experiments no GL error is pending");
            // Whole-buffer binding.
            mgl::glBindBufferBase(kSsbo, 0, probe);
            using P = mgl::GetParameter;
            check.expect(get_i(P::ShaderStorageBufferBinding, 0) == static_cast<GLint>(probe.value) && get_i64(P::ShaderStorageBufferStart, 0) == 0 &&
                             get_i64(P::ShaderStorageBufferSize, 0) == 0,
                         "glBindBufferBase(SHADER_STORAGE_BUFFER, 0, buf): BINDING reads back the buffer, START and SIZE are 0 (\"not specified\": the whole buffer, whatever its size at use time)",
                         std::format("binding {} start {} size {}", get_i(P::ShaderStorageBufferBinding, 0), get_i64(P::ShaderStorageBufferStart, 0), get_i64(P::ShaderStorageBufferSize, 0)));
            check.expect_eq(demo::get_integer(P::ShaderStorageBufferBinding), static_cast<int>(probe.value),
                            "... and the general (non-indexed) SHADER_STORAGE_BUFFER_BINDING is that buffer too (both commands bind the general and the indexed point)");
            // Range binding.
            mgl::glBindBufferRange(kSsbo, 1, sams, slot.offset, slot.size);
            check.expect(get_i(P::ShaderStorageBufferBinding, 1) == static_cast<GLint>(sams.value) && get_i64(P::ShaderStorageBufferStart, 1) == slot.offset &&
                             get_i64(P::ShaderStorageBufferSize, 1) == slot.size,
                         std::format("glBindBufferRange(index 1, offset {}, size {}): BINDING/START/SIZE read back exactly (glGetIntegeri_v / glGetInteger64i_v)", slot.offset, slot.size),
                         std::format("binding {} start {} size {}", get_i(P::ShaderStorageBufferBinding, 1), get_i64(P::ShaderStorageBufferStart, 1), get_i64(P::ShaderStorageBufferSize, 1)));
            check.expect(get_i(P::ShaderStorageBufferBinding, 2) == 0 && get_i64(P::ShaderStorageBufferStart, 2) == 0 && get_i64(P::ShaderStorageBufferSize, 2) == 0,
                         "an index that was never bound reads back 0 / 0 / 0");
            check.expect_no_gl_error("the binding queries raise no error");

            // Errors. A failed BindBufferRange leaves the binding untouched.
            const GLuint bad_index = static_cast<GLuint>(max_bindings_);
            mgl::glBindBufferBase(kSsbo, bad_index, probe);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferBase(SHADER_STORAGE_BUFFER, MAX_SHADER_STORAGE_BUFFER_BINDINGS) -> INVALID_VALUE");
            mgl::glBindBufferRange(kSsbo, bad_index, sams, slot.offset, slot.size);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferRange(index = MAX_SHADER_STORAGE_BUFFER_BINDINGS) -> INVALID_VALUE");
            GLint scratch = 7;
            mgl::glGetIntegeri_v(P::ShaderStorageBufferBinding, bad_index, &scratch);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(SHADER_STORAGE_BUFFER_BINDING, MAX_SHADER_STORAGE_BUFFER_BINDINGS) -> INVALID_VALUE");
            GLint64 scratch64 = 7;
            mgl::glGetInteger64i_v(P::ShaderStorageBufferSize, bad_index, &scratch64);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetInteger64i_v(SHADER_STORAGE_BUFFER_SIZE, MAX_SHADER_STORAGE_BUFFER_BINDINGS) -> INVALID_VALUE");

            if (alignment_ > 1)
            {
                mgl::glBindBufferRange(kSsbo, 1, sams, 1, 16);
                check.expect_gl_error(mgl::ErrorCode::InvalidValue,
                                      std::format("glBindBufferRange with offset 1, not a multiple of SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT = {} -> INVALID_VALUE", alignment_));
            }
            else
            {
                std::cout << "   [note] SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT is 1: every offset is aligned, the misalignment error cannot be provoked\n";
            }
            mgl::glBindBufferRange(kSsbo, 1, sams, slot.offset, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferRange with size 0 -> INVALID_VALUE");
            check.expect(get_i(P::ShaderStorageBufferBinding, 1) == static_cast<GLint>(sams.value) && get_i64(P::ShaderStorageBufferStart, 1) == slot.offset &&
                             get_i64(P::ShaderStorageBufferSize, 1) == slot.size,
                         "the failed glBindBufferRange calls changed nothing: index 1 still reads back the range bound before");
            // A negative offset that is a multiple of the alignment: "INVALID_VALUE if buffer is non-zero and offset is negative" (ES 3.2 section 6.1.1).
            mgl::glBindBufferRange(kSsbo, 1, sams, -static_cast<GLintptr>(std::max(alignment_, 1)), 16);
            const mgl::ErrorCode neg = demo::drain_gl_errors();
            check.expect_or_known_deviation(neg == mgl::ErrorCode::InvalidValue, "glBindBufferRange with a negative offset -> INVALID_VALUE",
                                            std::format("glGetError() = {}", demo::error_name(neg)), "Mesa",
                                            "Mesa does not validate a negative offset in glBindBufferRange when it is a multiple of the alignment (same as es30-uniform-buffer-objects); the binding is restored and never used");
            mgl::glBindBufferRange(kSsbo, 1, sams, slot.offset, slot.size);

            // Rebinding: BindBufferBase after BindBufferRange resets START and SIZE; binding buffer 0 clears the point.
            mgl::glBindBufferBase(kSsbo, 1, sams);
            check.expect(get_i64(P::ShaderStorageBufferStart, 1) == 0 && get_i64(P::ShaderStorageBufferSize, 1) == 0,
                         "glBindBufferBase after glBindBufferRange resets START and SIZE to 0");
            mgl::glBindBufferBase(kSsbo, 1, mgl::BufferId{0});
            check.expect(get_i(P::ShaderStorageBufferBinding, 1) == 0, "glBindBufferBase(index 1, buffer 0) unbinds the point (BINDING reads 0)");
            mgl::glBindBufferBase(kSsbo, 0, mgl::BufferId{0});
            check.expect_no_gl_error("the error experiments leave no error behind");
        }

        // ----------------------------------------------------------------------------------
        // limits
        // ----------------------------------------------------------------------------------
        static bool compute_links(const std::string& src, std::string& log)
        {
            std::string slog;
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, src, slog);
            if (cs.value == 0)
            {
                log = "compile failed: " + slog;
                return false;
            }
            const std::array<mgl::ShaderId, 1> shaders = {cs};
            const mgl::ProgramId p = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            if (p.value == 0) return false;
            mgl::glDeleteProgram(p);
            return true;
        }

        void check_limits(demo::Check& check)
        {
            check.expect(max_compute_blocks_ >= 4 && max_combined_blocks_ >= 4,
                         "MAX_COMPUTE_SHADER_STORAGE_BLOCKS >= 4 and MAX_COMBINED_SHADER_STORAGE_BLOCKS >= 4 (ES 3.2 chapter 21 minimums); the vertex and fragment limits (minimum 0 in ES 3.1) are reported",
                         std::format("compute {} combined {} vertex {} fragment {}", max_compute_blocks_, max_combined_blocks_, max_vertex_blocks_, max_fragment_blocks_));
            check.expect(max_bindings_ >= 4, "MAX_SHADER_STORAGE_BUFFER_BINDINGS >= 4", std::format("{}", max_bindings_));
            check.expect(max_block_size_ >= (GLint64{1} << 27), "MAX_SHADER_STORAGE_BLOCK_SIZE (glGetInteger64v) >= 2^27 = 134217728", std::format("{}", max_block_size_));
            check.expect(alignment_ >= 1 && alignment_ <= 256, "SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT is between 1 and 256 (256 is the largest value the spec allows)", std::format("{}", alignment_));
            check.expect(data_size_[0] <= max_block_size_ && data_size_[1] <= max_block_size_ && 3 < max_bindings_,
                         "the blocks of this demo fit MAX_SHADER_STORAGE_BLOCK_SIZE and its binding points 0-3 are below MAX_SHADER_STORAGE_BUFFER_BINDINGS");

            // "If the number of active shader storage blocks referenced by the shaders in a program exceeds ... limits, the program will fail to link"
            // (ES 3.2 section 7.8): exactly the compute limit links, one more does not.
            if (max_compute_blocks_ <= 64 && max_bindings_ >= 1)
            {
                auto source = [&](int blocks) {
                    std::string s = "#version 310 es\nlayout(local_size_x = 1) in;\n";
                    for (int i = 0; i < blocks; ++i) s += std::format("layout(std430, binding = {}) buffer B{} {{ uint v; }} b{};\n", i % max_bindings_, i, i);
                    s += "void main() {\n";
                    for (int i = 0; i < blocks; ++i) s += std::format("  b{}.v = {}u;\n", i, i);
                    s += "}\n";
                    return s;
                };
                std::string at_log, over_log;
                const bool at = compute_links(source(max_compute_blocks_), at_log);
                const bool over = compute_links(source(max_compute_blocks_ + 1), over_log);
                check.expect(at, std::format("a compute shader using exactly MAX_COMPUTE_SHADER_STORAGE_BLOCKS = {} blocks links", max_compute_blocks_), at_log.substr(0, 200));
                check.expect(!over, std::format("a compute shader using {} blocks (MAX_COMPUTE_SHADER_STORAGE_BLOCKS + 1) fails to link", max_compute_blocks_ + 1), over ? "it linked" : over_log.substr(0, 200));
            }
            else
            {
                std::cout << std::format("   [note] MAX_COMPUTE_SHADER_STORAGE_BLOCKS = {}: too large to build a shader with that many blocks, the link-limit experiment is skipped\n", max_compute_blocks_);
            }

            // "If the amount of storage required for the fixed size portion of any shader storage block (as reported by BUFFER_DATA_SIZE) exceeds this
            // limit, a program will fail to link."
            const GLint64 words = max_block_size_ / 4;
            if (words + 1 <= INT_MAX)
            {
                auto source = [&](GLint64 n) {
                    return std::format("#version 310 es\nlayout(local_size_x = 1) in;\nlayout(std430, binding = 0) buffer Big {{ uint a[{}]; }} big;\nvoid main() {{ big.a[0] = 1u; }}\n", n);
                };
                std::string at_log, over_log;
                const bool at = compute_links(source(words), at_log);
                const bool over = compute_links(source(words + 1), over_log);
                check.expect(at, std::format("a block of exactly MAX_SHADER_STORAGE_BLOCK_SIZE = {} bytes ({} uints) links", max_block_size_, words), at_log.substr(0, 200));
                check.expect(!over, std::format("a block of {} bytes (MAX_SHADER_STORAGE_BLOCK_SIZE + 4) fails to link", max_block_size_ + 4), over ? "it linked" : over_log.substr(0, 200));
            }
            else
            {
                std::cout << std::format("   [note] MAX_SHADER_STORAGE_BLOCK_SIZE = {} does not fit a GLSL array size: the block-size link experiment is skipped\n", max_block_size_);
            }
            check.expect_no_gl_error("the limit experiments leave no GL error");
        }

        // ----------------------------------------------------------------------------------
        // negative controls
        // ----------------------------------------------------------------------------------
        void negative_controls(demo::Check& check)
        {
            // One wrong word in an otherwise correct buffer: exactly that member must stop verifying, its neighbours must not.
            Bytes data = pack(blocks_[0], static_cast<std::size_t>(data_size_[0]));
            const Leaf& victim = blocks_[0].leaves[13];  // pairs[0].b: member 14
            put_float(data, static_cast<std::size_t>(victim.offset), -1.0f);
            dispatch_probe(0, data, false, bufs_[kFlags]);
            const std::vector<std::uint32_t> flags = read_flags(bufs_[kFlags]);
            std::string wrong;
            for (int m = 1; m <= 16; ++m)
                if (flags[static_cast<std::size_t>(m)] != (m == 14 ? 0u : 1u)) wrong += std::format("{} ", m);
            check.expect(wrong.empty() && victim.member == 14,
                         "negative control: overwriting pairs[0].b in an otherwise correct buffer makes exactly member 14 fail and leaves the other 15 members verified", wrong);
            // Restoring the value makes it pass again.
            dispatch_probe(0, pack(blocks_[0], static_cast<std::size_t>(data_size_[0])), false, bufs_[kFlags]);
            const std::vector<std::uint32_t> again = read_flags(bufs_[kFlags]);
            check.expect(std::all_of(again.begin() + 1, again.begin() + 17, [](std::uint32_t f) { return f == 1u; }), "negative control: restoring the value makes all 16 members verify again");
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 2; ++i) mgl::glBindBufferBase(kSsbo, i, mgl::BufferId{0});
        }

        // ----------------------------------------------------------------------------------
        // state
        // ----------------------------------------------------------------------------------
        std::vector<Member> members_;
        std::array<Block, 2> blocks_;  // [0] std430, [1] std140
        std::array<mgl::ProgramId, 2> layout_prog_{};
        std::array<mgl::UniformLocation, 2> u_write_{};
        mgl::ProgramId runtime_prog_{}, grid_prog_{}, bars_prog_{};
        mgl::UniformLocation u_cell_{}, u_mask_{}, u_scale_{};
        std::array<GLint, 2> gl_data_size_{};  ///< BUFFER_DATA_SIZE of Probe as the GL reports it
        std::array<GLint, 2> data_size_{};     ///< bytes of the Probe buffers
        std::array<mgl::BufferId, kBufferCount> bufs_{};
        std::array<Row, 4> rows_{};
        std::array<Bytes, 2> write_img_;
        std::vector<Slot> sams_slots_, recs_slots_;
        Bytes sams_img_, recs_img_;
        int alignment_ = 1, max_bindings_ = 4, max_compute_blocks_ = 4, max_vertex_blocks_ = 0, max_fragment_blocks_ = 0, max_combined_blocks_ = 4;
        GLint64 max_block_size_ = 0;
        bool use_fragment_ = false, use_vertex_ = false;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-ssbo-std430",
        .title = "Shader storage buffers and std430",
        .min_es = {3, 1},
        .summary = "Shader storage blocks read and written by compute, vertex and fragment shaders: a CPU std430/std140 layout calculator checked against "
                   "glGetProgramResourceiv and against data moved in both directions, runtime-sized arrays driven by glBindBufferRange, "
                   "the indexed binding state, limits, errors and memory barriers.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<SsboStd430>(argc, argv, info);
}
