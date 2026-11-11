// es32-texture-buffer
//
// Buffer textures (OpenGL ES 3.2 section 8.9): a buffer object is attached to a TEXTURE_BUFFER texture with
// glTexBuffer / glTexBufferRange and its data store becomes a one-dimensional texture that shaders read with
// texelFetch (samplerBuffer / isamplerBuffer / usamplerBuffer), measure with textureSize and, for the formats of
// Table 8.26, read and write with imageLoad / imageStore (imageBuffer).
//
// The picture is a practical use: a grid of 8 x 8 cubes is drawn with ONE glDrawElementsInstanced call, and every
// per-instance value comes from buffer textures that are three different ranges of one buffer object
//   RGBA32F  two texels per instance: position + scale, rotation angle (rewritten every frame with glBufferSubData
//            on the TEXTURE_BUFFER target)
//   RGBA8    one normalized colour texel per instance
//   R16UI    one material number per instance (usamplerBuffer)
// The bottom strip (32 rows) reads the colour and material texels in a fragment shader, one cell per instance.
//
// What validate() proves (all against CPU models, most of it through compute shaders that dump raw words):
//   * all 30 sized internal formats of Table 8.18: every component of 64 texels, including the constant
//     components (0, 0, 1) of the 1-, 2- and 3-component formats, sign extension, half floats and normalization
//   * glTexBufferRange offsets (multiples of TEXTURE_BUFFER_OFFSET_ALIGNMENT) and sizes: texel count =
//     floor(size / texel size), the first texel is the one at the offset; every error of section 8.9
//   * TEXTURE_BUFFER_BINDING / TEXTURE_BINDING_BUFFER state, a buffer bound to TEXTURE_BUFFER has no effect on
//     rendering, data written with glBufferSubData through the TEXTURE_BUFFER target is visible to the next fetch
//   * imageLoad / imageStore / imageSize on imageBuffer for the 12 formats that Tables 8.18 and 8.26 share
//   * the largest buffer texture every implementation must support (65536 texels)
//   * GLSL ES 3.20 compile-time rules (precision of samplerBuffer, the one texelFetch overload, version 3.20)
//   * the instanced scene and the strip against an independent ray caster
//
// Not asserted (the specification leaves the result undefined): a texelFetch outside the texel range. The demo
// prints what this driver returns for a texel just behind the range, as a note.

#include <demo/gl_util.hpp>
#include <demo/math.hpp>
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
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;
namespace dm = demo::math;

static_assert(std::endian::native == std::endian::little, "the CPU models build little-endian element bytes");

namespace
{
    using Bytes = std::vector<std::uint8_t>;

    std::uint32_t hash32(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    std::size_t round_up(std::size_t v, std::size_t multiple) { return (v + multiple - 1) / multiple * multiple; }

    // ---------------------------------------------------------------------------------------------
    // ES 3.2 Table 8.18: internal formats for buffer textures
    // ---------------------------------------------------------------------------------------------
    enum class Elem { U8, S8, U16, S16, F16, U32, S32, F32 };
    enum class Family { Float, Int, Uint };  // the sampler type: samplerBuffer, isamplerBuffer, usamplerBuffer

    constexpr int elem_bytes(Elem e)
    {
        switch (e)
        {
        case Elem::U8: case Elem::S8: return 1;
        case Elem::U16: case Elem::S16: case Elem::F16: return 2;
        default: return 4;
        }
    }

    struct Format
    {
        const char* name;
        mgl::InternalFormat gl;
        Elem elem;
        int comps;
        bool norm;                      // the "Norm" column: unsigned normalized
        const char* image_qualifier;    // Table 8.26 layout qualifier, nullptr when the format is no image format

        [[nodiscard]] int texel_bytes() const { return comps * elem_bytes(elem); }
        [[nodiscard]] bool is_float_elem() const { return elem == Elem::F16 || elem == Elem::F32; }
        [[nodiscard]] Family family() const
        {
            if (norm || is_float_elem()) return Family::Float;
            return (elem == Elem::S8 || elem == Elem::S16 || elem == Elem::S32) ? Family::Int : Family::Uint;
        }
    };

    using IF = mgl::InternalFormat;
    const std::array<Format, 30> kFormats = {{
        {"R8", IF::R8, Elem::U8, 1, true, nullptr},
        {"R16F", IF::R16F, Elem::F16, 1, false, nullptr},
        {"R32F", IF::R32F, Elem::F32, 1, false, "r32f"},
        {"R8I", IF::R8I, Elem::S8, 1, false, nullptr},
        {"R16I", IF::R16I, Elem::S16, 1, false, nullptr},
        {"R32I", IF::R32I, Elem::S32, 1, false, "r32i"},
        {"R8UI", IF::R8UI, Elem::U8, 1, false, nullptr},
        {"R16UI", IF::R16UI, Elem::U16, 1, false, nullptr},
        {"R32UI", IF::R32UI, Elem::U32, 1, false, "r32ui"},
        {"RG8", IF::Rg8, Elem::U8, 2, true, nullptr},
        {"RG16F", IF::Rg16F, Elem::F16, 2, false, nullptr},
        {"RG32F", IF::Rg32F, Elem::F32, 2, false, nullptr},
        {"RG8I", IF::Rg8I, Elem::S8, 2, false, nullptr},
        {"RG16I", IF::Rg16I, Elem::S16, 2, false, nullptr},
        {"RG32I", IF::Rg32I, Elem::S32, 2, false, nullptr},
        {"RG8UI", IF::Rg8UI, Elem::U8, 2, false, nullptr},
        {"RG16UI", IF::Rg16UI, Elem::U16, 2, false, nullptr},
        {"RG32UI", IF::Rg32UI, Elem::U32, 2, false, nullptr},
        {"RGB32F", IF::Rgb32F, Elem::F32, 3, false, nullptr},
        {"RGB32I", IF::Rgb32I, Elem::S32, 3, false, nullptr},
        {"RGB32UI", IF::Rgb32UI, Elem::U32, 3, false, nullptr},
        {"RGBA8", IF::Rgba8, Elem::U8, 4, true, "rgba8"},
        {"RGBA16F", IF::Rgba16F, Elem::F16, 4, false, "rgba16f"},
        {"RGBA32F", IF::Rgba32F, Elem::F32, 4, false, "rgba32f"},
        {"RGBA8I", IF::Rgba8I, Elem::S8, 4, false, "rgba8i"},
        {"RGBA16I", IF::Rgba16I, Elem::S16, 4, false, "rgba16i"},
        {"RGBA32I", IF::Rgba32I, Elem::S32, 4, false, "rgba32i"},
        {"RGBA8UI", IF::Rgba8UI, Elem::U8, 4, false, "rgba8ui"},
        {"RGBA16UI", IF::Rgba16UI, Elem::U16, 4, false, "rgba16ui"},
        {"RGBA32UI", IF::Rgba32UI, Elem::U32, 4, false, "rgba32ui"},
    }};

    const Format& format_named(std::string_view name)
    {
        for (const Format& f : kFormats)
            if (name == f.name) return f;
        throw demo::Failure(std::format("internal error: no format {}", name));
    }

    /// IEEE half -> float bits. The data generator never makes denormals, infinities or NaN.
    std::uint32_t half_to_float_bits(std::uint32_t h)
    {
        const std::uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
        if (exp == 0 && man == 0) return sign << 31;
        return (sign << 31) | ((exp - 15u + 127u) << 23) | (man << 13);
    }

    /// `texels` texels of pseudo-random element data for the format, as the GL data type of Table 8.18.
    Bytes make_data(const Format& f, int texels, std::uint32_t seed)
    {
        const int eb = elem_bytes(f.elem);
        const std::size_t elems = static_cast<std::size_t>(texels) * static_cast<std::size_t>(f.comps);
        Bytes out(elems * static_cast<std::size_t>(eb));
        for (std::size_t e = 0; e < elems; ++e)
        {
            const std::uint32_t h = hash32(seed * 0x9E3779B1u + static_cast<std::uint32_t>(e) * 0x85EBCA6Bu + 1u);
            std::uint32_t v = 0;
            switch (f.elem)
            {
            case Elem::U8: case Elem::S8: v = h & 0xFFu; break;
            case Elem::U16: case Elem::S16: v = h & 0xFFFFu; break;
            case Elem::F16: v = ((h >> 31) << 15) | ((8u + (h >> 16) % 15u) << 10) | (h & 0x3FFu); break;  // 2^-7 .. 2^7
            case Elem::U32: case Elem::S32: v = h; break;
            case Elem::F32: v = ((h >> 31) << 31) | ((100u + (h >> 8) % 50u) << 23) | (hash32(h + 77u) & 0x7FFFFFu); break;
            }
            std::memcpy(&out[e * static_cast<std::size_t>(eb)], &v, static_cast<std::size_t>(eb));
        }
        return out;
    }

    /// What one texel fetch must return: four 32-bit words (floats as their bit pattern) and, for the unsigned
    /// normalized formats, the four values of equation 2.1.
    struct Texel
    {
        std::array<std::uint32_t, 4> w{};
        std::array<double, 4> f{};
    };

    /// Table 8.18: element m of texel n is element n * components + m of the store; the missing components are 0, 0, 1.
    Texel decode_texel(const Format& fmt, const std::uint8_t* texel)
    {
        Texel t;
        const int eb = elem_bytes(fmt.elem);
        const std::uint32_t one = fmt.family() == Family::Float ? std::bit_cast<std::uint32_t>(1.0f) : 1u;
        for (int c = 0; c < 4; ++c)
        {
            if (c >= fmt.comps)
            {
                t.w[static_cast<std::size_t>(c)] = c == 3 ? one : 0u;
                t.f[static_cast<std::size_t>(c)] = c == 3 ? 1.0 : 0.0;
                continue;
            }
            std::uint32_t raw = 0;
            std::memcpy(&raw, texel + static_cast<std::size_t>(c * eb), static_cast<std::size_t>(eb));
            std::uint32_t word = raw;
            switch (fmt.elem)
            {
            case Elem::S8: word = static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int8_t>(raw))); break;
            case Elem::S16: word = static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(raw))); break;
            case Elem::F16: word = half_to_float_bits(raw); break;
            default: break;
            }
            t.w[static_cast<std::size_t>(c)] = word;
            t.f[static_cast<std::size_t>(c)] = fmt.norm ? static_cast<double>(raw) / 255.0 : 0.0;  // eq. 2.1, b = 8
        }
        return t;
    }

    // ---------------------------------------------------------------------------------------------
    // The instanced scene (CPU side)
    // ---------------------------------------------------------------------------------------------
    constexpr int kGrid = 8;
    constexpr int kInstances = kGrid * kGrid;
    constexpr int kStripHeight = 32;
    constexpr double kSpacing = 1.15;
    constexpr std::array<double, 3> kLight = {0.4, 0.8, 0.5};  // normalized in light_dir()
    constexpr std::array<float, 4> kClear = {0.08f, 0.09f, 0.12f, 1.0f};

    struct V3
    {
        double x = 0, y = 0, z = 0;
    };
    V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    V3 operator*(V3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
    double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    V3 normalize(V3 a) { return a * (1.0 / std::sqrt(dot(a, a))); }
    V3 light_dir() { return normalize({kLight[0], kLight[1], kLight[2]}); }

    struct Instance
    {
        float x = 0, y = 0, z = 0, scale = 0, angle = 0;
        Rgba8 color{};
        std::uint16_t material = 0;
    };

    Rgba8 hsv_to_rgba(double h, double s, double v)
    {
        const double k = h * 6.0;
        const int i = static_cast<int>(std::floor(k)) % 6;
        const double f = k - std::floor(k);
        const double p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
        double r = 0, g = 0, b = 0;
        switch (i)
        {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
        }
        auto to8 = [](double c) { return static_cast<std::uint8_t>(std::lround(c * 255.0)); };
        return {to8(r), to8(g), to8(b), 255};
    }

    /// Instance i at animation time t (seconds). Position, scale and angle are what the transform texels hold.
    Instance make_instance(int i, double t)
    {
        const int ix = i % kGrid, iz = i / kGrid;
        Instance in;
        in.x = static_cast<float>((ix - (kGrid - 1) / 2.0) * kSpacing);
        in.z = static_cast<float>((iz - (kGrid - 1) / 2.0) * kSpacing);
        in.y = static_cast<float>(0.45 * std::sin(3.0 * t + 0.9 * ix + 0.6 * iz));
        in.scale = static_cast<float>(0.62 + 0.08 * std::sin(2.0 * t + 1.7 * i));
        in.angle = static_cast<float>(0.31 * ix + 0.23 * iz + 1.2 * t);
        in.color = hsv_to_rgba(static_cast<double>(i) / kInstances, 0.75, 0.95);
        in.material = static_cast<std::uint16_t>((ix + iz) % 3);
        return in;
    }

    std::vector<float> transform_texels(double t)
    {
        std::vector<float> out;
        out.reserve(static_cast<std::size_t>(kInstances) * 8);
        for (int i = 0; i < kInstances; ++i)
        {
            const Instance in = make_instance(i, t);
            out.insert(out.end(), {in.x, in.y, in.z, in.scale, in.angle, 0.0f, 0.0f, 0.0f});
        }
        return out;
    }

    struct Hit
    {
        int id = -1;
        V3 normal{};
        double t = std::numeric_limits<double>::infinity();
    };

    /// The box of instance `in`, rotated about +Y (the same rotation as scene.vert), with every half extent grown by `grow`.
    Hit cast_box(const Instance& in, int id, V3 origin, V3 dir, double grow)
    {
        const double c = std::cos(static_cast<double>(in.angle)), s = std::sin(static_cast<double>(in.angle));
        auto to_local = [&](V3 v) { return V3{c * v.x - s * v.z, v.y, s * v.x + c * v.z}; };  // inverse of x' = c x + s z, z' = -s x + c z
        const V3 o = to_local(origin - V3{in.x, in.y, in.z});
        const V3 d = to_local(dir);
        const double half = 0.5 * static_cast<double>(in.scale) + grow;
        double tn = -std::numeric_limits<double>::infinity(), tf = std::numeric_limits<double>::infinity();
        V3 nl{};
        const double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
        for (int a = 0; a < 3; ++a)
        {
            if (std::abs(dd[a]) < 1e-12)
            {
                if (std::abs(oo[a]) > half) return {};
                continue;
            }
            const double t1 = (-half - oo[a]) / dd[a], t2 = (half - oo[a]) / dd[a];
            const double near_t = std::min(t1, t2), far_t = std::max(t1, t2);
            if (near_t > tn)
            {
                tn = near_t;
                nl = {};
                // entering through the face on the negative side (dd > 0) gives the normal -axis, through the positive side +axis
                (a == 0 ? nl.x : a == 1 ? nl.y : nl.z) = dd[a] > 0 ? -1.0 : 1.0;
            }
            tf = std::min(tf, far_t);
        }
        if (tn > tf || tf <= 0.0 || tn <= 0.0) return {};
        Hit h;
        h.id = id;
        h.t = tn;
        h.normal = {c * nl.x + s * nl.z, nl.y, -s * nl.x + c * nl.z};  // back to world: x' = c x + s z, z' = -s x + c z
        return h;
    }

    Hit cast_scene(const std::vector<Instance>& inst, V3 origin, V3 dir, double grow)
    {
        Hit best;
        for (std::size_t i = 0; i < inst.size(); ++i)
        {
            const Hit h = cast_box(inst[i], static_cast<int>(i), origin, dir, grow);
            if (h.id >= 0 && h.t < best.t) best = h;
        }
        return best;
    }

    /// The scene.frag shading of a base colour on a face with world normal n.
    Rgba8 shade(Rgba8 base, unsigned material, V3 n)
    {
        double k = 1.0;
        if (material == 0) k = 0.30 + 0.70 * std::max(dot(n, light_dir()), 0.0);
        else if (material == 2) k = n.y > 0.5 ? 1.0 : 0.5;
        auto ch = [&](std::uint8_t b) { return static_cast<std::uint8_t>(std::lround(static_cast<double>(b) * k)); };
        return {ch(base[0]), ch(base[1]), ch(base[2]), 255};
    }

    int max_channel_diff(Rgba8 a, Rgba8 b)
    {
        int d = 0;
        for (std::size_t i = 0; i < 3; ++i) d = std::max(d, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        return d;
    }

    // ---------------------------------------------------------------------------------------------
    // The demo
    // ---------------------------------------------------------------------------------------------
    struct ComputeProgram
    {
        mgl::ProgramId id{};
        mgl::UniformLocation tex{};
        mgl::UniformLocation count{};
    };

    struct FetchResult
    {
        std::uint32_t texture_size = 0;
        std::vector<std::uint32_t> words;
    };

    struct Mismatch
    {
        int bad = 0;
        std::string first;
        double max_norm_err = 0.0;
    };

    class TextureBuffer final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override;
        void render(demo::Env& env, int frame) override;
        void validate(demo::Env& env, demo::Check& check) override;
        void shutdown(demo::Env& env) override;
        void on_key(demo::Env& env, char key) override;

    private:
        // ---- infrastructure ---------------------------------------------------------------------------------
        mgl::BufferId new_buffer()
        {
            mgl::BufferId b{};
            mgl::glGenBuffers(1, &b);
            buffers_.push_back(b);
            return b;
        }
        mgl::TextureId new_texture()
        {
            mgl::TextureId t{};
            mgl::glGenTextures(1, &t);
            textures_.push_back(t);
            return t;
        }
        mgl::ProgramId compile_compute(const demo::Env& env, std::string_view file, const std::string& defines);
        mgl::UniformLocation locate(mgl::ProgramId program, const char* name) const;

        /// glBufferData through the TEXTURE_BUFFER target (the target is only a place to bind; it does not affect rendering).
        static void fill_buffer(mgl::BufferId buf, const Bytes& data)
        {
            mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, buf);
            mgl::glBufferData(mgl::BufferTarget::TextureBuffer, static_cast<GLsizeiptr>(data.size()), data.data(), mgl::BufferUsage::DynamicDraw);
        }
        /// Attaches `buf` to `tex` (glTexBufferRange) and leaves `tex` bound to TEXTURE_BUFFER on unit 0.
        static void attach(mgl::TextureId tex, const Format& f, mgl::BufferId buf, std::size_t offset, std::size_t size)
        {
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, tex);
            mgl::glTexBufferRange(mgl::TextureTarget::TextureBuffer, f.gl, buf, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size));
        }
        static Bytes map_read(mgl::BufferTarget target, mgl::BufferId buf, std::size_t offset, std::size_t size);

        FetchResult fetch(Family fam, mgl::TextureId tex, int count);
        static Mismatch compare_fetch(const Format& f, const Bytes& expected, int texels, const FetchResult& r, double norm_tol);

        // ---- validation sections ----------------------------------------------------------------------------
        void check_limits(demo::Check& check);
        void check_format_table(demo::Check& check);
        void check_buffer_range(demo::Check& check);
        void check_errors(demo::Check& check);
        void check_bindings(demo::Check& check);
        void check_level_queries(demo::Check& check);
        void check_updates(demo::Check& check);
        void check_image_buffer(demo::Env& env, demo::Check& check);
        void check_large_texture(demo::Check& check);
        void check_compile_rules(demo::Check& check);
        void check_scene(demo::Env& env, demo::Check& check);
        void note_out_of_range();

        // ---- scene --------------------------------------------------------------------------------------------
        void draw_scene(const demo::Env& env, int frame);
        [[nodiscard]] dm::Mat4 view_proj(double aspect) const;
        [[nodiscard]] dm::Vec3 eye_position() const;

        // programs
        std::array<ComputeProgram, 3> fetch_{};  // by Family
        mgl::ProgramId scene_program_{}, strip_program_{};
        mgl::UniformLocation loc_view_proj_{}, loc_light_{}, loc_transform_{}, loc_color_{}, loc_material_{};
        mgl::UniformLocation loc_strip_color_{}, loc_strip_material_{}, loc_strip_rect_{}, loc_strip_cells_{};
        std::vector<mgl::ProgramId> extra_programs_;

        // objects
        std::vector<mgl::BufferId> buffers_;
        std::vector<mgl::TextureId> textures_;
        mgl::BufferId scene_buf_{}, cube_vbo_{}, cube_ibo_{}, ssbo_{}, data_buf_{}, decoy_buf_{};
        mgl::TextureId sweep_tex_{};
        std::array<mgl::TextureId, 3> scene_tex_{};  // transforms, colours, materials
        mgl::VertexArrayId cube_vao_{}, empty_vao_{};
        std::array<std::size_t, 3> scene_offset_{}, scene_size_{};
        std::size_t scene_total_ = 0;

        // limits and initial state
        int alignment_ = 0;
        int max_texture_buffer_size_ = 0;
        int initial_buffer_binding_ = -1, initial_texture_binding_ = -1;

        // camera
        double yaw_deg_ = 25.0, pitch_deg_ = 38.0, distance_ = 13.6;
        int shown_frame_ = 0;
        int width_ = 256, height_ = 256;
    };

    // =================================================================================================
    // init / render / shutdown
    // =================================================================================================
    mgl::UniformLocation TextureBuffer::locate(mgl::ProgramId program, const char* name) const
    {
        const mgl::UniformLocation l = mgl::glGetUniformLocation(program, name);
        if (l.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
        return l;
    }

    mgl::ProgramId TextureBuffer::compile_compute(const demo::Env& env, std::string_view file, const std::string& defines)
    {
        std::string src = env.load_text(file);
        if (!defines.empty()) src.insert(src.find('\n') + 1, defines);  // macros go right after "#version 320 es"
        std::string log;
        const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, src, log);
        if (cs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", file, log));
        const std::array<mgl::ShaderId, 1> shaders = {cs};
        const mgl::ProgramId program = demo::link_program(shaders, log);
        mgl::glDeleteShader(cs);
        if (program.value == 0) throw demo::Failure(std::format("{} failed to link:\n{}", file, log));
        return program;
    }

    void TextureBuffer::init(demo::Env& env)
    {
        // Initial state before this demo touches anything (ES 3.2 Table 21.9 / 21.40: both initially 0).
        initial_buffer_binding_ = demo::get_integer(mgl::GetParameter::TextureBufferBinding);
        initial_texture_binding_ = demo::get_integer(mgl::GetParameter::TextureBindingBuffer);
        alignment_ = demo::get_integer(mgl::GetParameter::TextureBufferOffsetAlignment);
        max_texture_buffer_size_ = demo::get_integer(mgl::GetParameter::MaxTextureBufferSize);
        width_ = env.width;
        height_ = env.height;
        if (alignment_ <= 0) throw demo::Failure("TEXTURE_BUFFER_OFFSET_ALIGNMENT is not positive");

        // Compute programs that dump a buffer texture, one per sampler family.
        struct Variant
        {
            Family fam;
            const char* defines;
        };
        const std::array<Variant, 3> variants = {{
            {Family::Float, "#define SAMPLER samplerBuffer\n#define TO_WORDS(v) floatBitsToUint(v)\n"},
            {Family::Int, "#define SAMPLER isamplerBuffer\n#define TO_WORDS(v) uvec4(v)\n"},
            {Family::Uint, "#define SAMPLER usamplerBuffer\n#define TO_WORDS(v) (v)\n"},
        }};
        for (const Variant& v : variants)
        {
            ComputeProgram& p = fetch_[static_cast<std::size_t>(v.fam)];
            p.id = compile_compute(env, "shaders/fetch.comp", v.defines);
            p.tex = locate(p.id, "u_tex");
            p.count = locate(p.id, "u_count");
        }

        scene_program_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
        loc_view_proj_ = locate(scene_program_, "u_view_proj");
        loc_light_ = locate(scene_program_, "u_light");
        loc_transform_ = locate(scene_program_, "u_transform");
        loc_color_ = locate(scene_program_, "u_color");
        loc_material_ = locate(scene_program_, "u_material");
        strip_program_ = demo::build_program(env, "shaders/strip.vert", "shaders/strip.frag");
        loc_strip_color_ = locate(strip_program_, "u_color");
        loc_strip_material_ = locate(strip_program_, "u_material");
        loc_strip_rect_ = locate(strip_program_, "u_rect");
        loc_strip_cells_ = locate(strip_program_, "u_cells");

        // Cube mesh: 24 vertices (position, normal), 36 indices.
        struct Vertex
        {
            float p[3], n[3];
        };
        std::vector<Vertex> verts;
        std::vector<std::uint16_t> indices;
        // face normal n, tangents u and v with u x v = n
        struct Face
        {
            dm::Vec3 n, u, v;
        };
        const std::array<Face, 6> faces = {{
            {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},
            {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
            {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},
            {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
            {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
            {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
        }};
        for (const Face& face : faces)
        {
            const std::uint16_t base = static_cast<std::uint16_t>(verts.size());
            const float sx[4] = {-1, 1, 1, -1}, sy[4] = {-1, -1, 1, 1};  // counter-clockwise around n because u x v = n
            for (int k = 0; k < 4; ++k)
            {
                const dm::Vec3 p = face.n * 0.5f + face.u * (0.5f * sx[k]) + face.v * (0.5f * sy[k]);
                verts.push_back({{p.x, p.y, p.z}, {face.n.x, face.n.y, face.n.z}});
            }
            for (std::uint16_t k : {0, 1, 2, 0, 2, 3}) indices.push_back(static_cast<std::uint16_t>(base + k));
        }
        cube_vbo_ = new_buffer();
        mgl::glBindBuffer(mgl::BufferTarget::Array, cube_vbo_);
        mgl::glBufferData(mgl::BufferTarget::Array, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)), verts.data(), mgl::BufferUsage::StaticDraw);
        cube_ibo_ = new_buffer();
        mgl::glGenVertexArrays(1, &cube_vao_);
        mgl::glBindVertexArray(cube_vao_);
        mgl::glBindBuffer(mgl::BufferTarget::ElementArray, cube_ibo_);
        mgl::glBufferData(mgl::BufferTarget::ElementArray, static_cast<GLsizeiptr>(indices.size() * sizeof(std::uint16_t)), indices.data(),
                          mgl::BufferUsage::StaticDraw);
        mgl::glEnableVertexAttribArray(mgl::AttribLocation{0u});
        mgl::glVertexAttribPointer(mgl::AttribLocation{0u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, p)));
        mgl::glEnableVertexAttribArray(mgl::AttribLocation{1u});
        mgl::glVertexAttribPointer(mgl::AttribLocation{1u}, 3, mgl::DataType::Float, GL_FALSE, sizeof(Vertex), demo::buffer_offset(offsetof(Vertex, n)));
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        mgl::glBindBuffer(mgl::BufferTarget::ElementArray, mgl::BufferId{0});
        mgl::glGenVertexArrays(1, &empty_vao_);

        // One buffer object, three ranges. Every range starts at a multiple of TEXTURE_BUFFER_OFFSET_ALIGNMENT; a guard
        // gap of sentinel bytes (never part of a texture) sits in front of the second and third range, so a wrong offset
        // reads garbage instead of neighbouring valid data.
        const std::size_t align = static_cast<std::size_t>(alignment_);
        scene_size_ = {static_cast<std::size_t>(kInstances) * 32, static_cast<std::size_t>(kInstances) * 4, static_cast<std::size_t>(kInstances) * 2};
        scene_offset_[0] = 0;
        scene_offset_[1] = round_up(scene_size_[0] + 64, align);
        scene_offset_[2] = round_up(scene_offset_[1] + scene_size_[1] + 64, align);
        scene_total_ = scene_offset_[2] + scene_size_[2] + 64;
        Bytes store(scene_total_, 0xA5);
        {
            const std::vector<float> tr = transform_texels(0.0);
            std::memcpy(store.data() + scene_offset_[0], tr.data(), tr.size() * sizeof(float));
            for (int i = 0; i < kInstances; ++i)
            {
                const Instance in = make_instance(i, 0.0);
                std::memcpy(store.data() + scene_offset_[1] + static_cast<std::size_t>(i) * 4, in.color.data(), 4);
                std::memcpy(store.data() + scene_offset_[2] + static_cast<std::size_t>(i) * 2, &in.material, 2);
            }
        }
        scene_buf_ = new_buffer();
        fill_buffer(scene_buf_, store);
        const std::array<const Format*, 3> scene_formats = {&format_named("RGBA32F"), &format_named("RGBA8"), &format_named("R16UI")};
        for (std::size_t k = 0; k < 3; ++k)
        {
            scene_tex_[k] = new_texture();
            attach(scene_tex_[k], *scene_formats[k], scene_buf_, scene_offset_[k], scene_size_[k]);
        }
        mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, mgl::TextureId{0});
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});

        // Utility objects of the validation sweeps.
        ssbo_ = new_buffer();
        data_buf_ = new_buffer();
        decoy_buf_ = new_buffer();
        sweep_tex_ = new_texture();
    }

    dm::Vec3 TextureBuffer::eye_position() const
    {
        const float yaw = dm::radians(static_cast<float>(yaw_deg_)), pitch = dm::radians(static_cast<float>(pitch_deg_));
        const float d = static_cast<float>(distance_);
        return {d * std::cos(pitch) * std::sin(yaw), d * std::sin(pitch), d * std::cos(pitch) * std::cos(yaw)};
    }

    dm::Mat4 TextureBuffer::view_proj(double aspect) const
    {
        const dm::Mat4 proj = dm::perspective(dm::radians(45.0f), static_cast<float>(aspect), 0.5f, 60.0f);
        const dm::Mat4 view = dm::look_at(eye_position(), {0, 0, 0}, {0, 1, 0});
        return proj * view;
    }

    void TextureBuffer::draw_scene(const demo::Env& env, int frame)
    {
        shown_frame_ = frame;
        const double t = static_cast<double>(frame) / 60.0;

        // The animation: this frame's transform texels are written through the TEXTURE_BUFFER binding point.
        const std::vector<float> tr = transform_texels(t);
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, scene_buf_);
        mgl::glBufferSubData(mgl::BufferTarget::TextureBuffer, 0, static_cast<GLsizeiptr>(tr.size() * sizeof(float)), tr.data());
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});

        mgl::glViewport(0, 0, env.width, env.height);
        mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
        mgl::glClear(mgl::ClearBufferBit::Color | mgl::ClearBufferBit::Depth);

        // Texture units 0, 1, 2 <- the transform, colour and material buffer textures.
        for (std::size_t k = 0; k < 3; ++k)
        {
            mgl::glActiveTexture(static_cast<mgl::TextureUnit>(static_cast<GLenum>(mgl::TextureUnit::Texture0) + static_cast<GLenum>(k)));
            mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, scene_tex_[k]);
        }

        // The scene above the strip.
        const int scene_h = env.height - kStripHeight;
        mgl::glViewport(0, kStripHeight, env.width, scene_h);
        mgl::glEnable(mgl::Capability::DepthTest);
        mgl::glUseProgram(scene_program_);
        const dm::Mat4 vp = view_proj(static_cast<double>(env.width) / scene_h);
        mgl::glUniformMatrix4fv(loc_view_proj_, 1, GL_FALSE, vp.data());
        const V3 l = light_dir();
        mgl::glUniform3f(loc_light_, static_cast<float>(l.x), static_cast<float>(l.y), static_cast<float>(l.z));
        mgl::glUniform1i(loc_transform_, 0);
        mgl::glUniform1i(loc_color_, 1);
        mgl::glUniform1i(loc_material_, 2);
        mgl::glBindVertexArray(cube_vao_);
        mgl::glDrawElementsInstanced(mgl::PrimitiveType::Triangles, 36, mgl::IndexType::UnsignedShort, demo::buffer_offset(0), kInstances);

        // The strip: fragment-shader texelFetch of the colour (unit 1) and material (unit 2) textures.
        mgl::glDisable(mgl::Capability::DepthTest);
        mgl::glViewport(0, 0, env.width, kStripHeight);
        mgl::glUseProgram(strip_program_);
        mgl::glUniform1i(loc_strip_color_, 1);
        mgl::glUniform1i(loc_strip_material_, 2);
        mgl::glUniform4i(loc_strip_rect_, 0, 0, env.width, kStripHeight);
        mgl::glUniform1i(loc_strip_cells_, kInstances);
        mgl::glBindVertexArray(empty_vao_);
        mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
    }

    void TextureBuffer::render(demo::Env& env, int frame)
    {
        width_ = env.width;
        height_ = env.height;
        draw_scene(env, frame);
    }

    void TextureBuffer::on_key(demo::Env& /*env*/, char key)
    {
        if (key == 'a') yaw_deg_ -= 5.0;
        else if (key == 'd') yaw_deg_ += 5.0;
        else if (key == 'w') pitch_deg_ = std::min(pitch_deg_ + 3.0, 80.0);
        else if (key == 's') pitch_deg_ = std::max(pitch_deg_ - 3.0, 10.0);
        else if (key == 'q') distance_ = std::max(distance_ - 0.5, 6.0);
        else if (key == 'e') distance_ = std::min(distance_ + 0.5, 30.0);
    }

    void TextureBuffer::shutdown(demo::Env& /*env*/)
    {
        mgl::glUseProgram(mgl::ProgramId{0});
        mgl::glBindVertexArray(mgl::VertexArrayId{0});
        for (int u = 0; u < 3; ++u)
        {
            mgl::glActiveTexture(static_cast<mgl::TextureUnit>(static_cast<GLenum>(mgl::TextureUnit::Texture0) + static_cast<GLenum>(u)));
            mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, mgl::TextureId{0});
        }
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        for (GLuint u = 0; u < 2; ++u)
            mgl::glBindImageTexture(mgl::ImageUnit{u}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
        if (!textures_.empty()) mgl::glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
        if (!buffers_.empty()) mgl::glDeleteBuffers(static_cast<GLsizei>(buffers_.size()), buffers_.data());
        mgl::glDeleteVertexArrays(1, &cube_vao_);
        mgl::glDeleteVertexArrays(1, &empty_vao_);
        for (const ComputeProgram& p : fetch_) mgl::glDeleteProgram(p.id);
        for (mgl::ProgramId p : extra_programs_) mgl::glDeleteProgram(p);
        mgl::glDeleteProgram(scene_program_);
        mgl::glDeleteProgram(strip_program_);
    }

    // =================================================================================================
    // Fetch helpers
    // =================================================================================================
    Bytes TextureBuffer::map_read(mgl::BufferTarget target, mgl::BufferId buf, std::size_t offset, std::size_t size)
    {
        mgl::glBindBuffer(target, buf);
        const void* p = mgl::glMapBufferRange(target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), mgl::MapBufferAccessMask::Read);
        if (p == nullptr) throw demo::Failure("glMapBufferRange failed");
        Bytes out(size);
        std::memcpy(out.data(), p, size);
        if (mgl::glUnmapBuffer(target) != GL_TRUE) throw demo::Failure("glUnmapBuffer returned FALSE");
        return out;
    }

    /// Dumps `count` texels of buffer texture `tex` (and textureSize) with a compute shader; the SSBO starts out filled with a sentinel.
    FetchResult TextureBuffer::fetch(Family fam, mgl::TextureId tex, int count)
    {
        const ComputeProgram& p = fetch_[static_cast<std::size_t>(fam)];
        const std::size_t bytes = round_up((1 + 4 * static_cast<std::size_t>(count)) * 4, 16);  // texture_size + 4 words per texel
        const std::vector<std::uint32_t> sentinel(bytes / 4, 0xDEADBEEFu);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, static_cast<GLsizeiptr>(bytes), sentinel.data(), mgl::BufferUsage::DynamicCopy);
        mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, ssbo_);

        mgl::glUseProgram(p.id);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, tex);
        mgl::glUniform1i(p.tex, 0);
        mgl::glUniform1i(p.count, count);
        mgl::glDispatchCompute(static_cast<GLuint>((count + 63) / 64), 1, 1);
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);

        const Bytes raw = map_read(mgl::BufferTarget::ShaderStorage, ssbo_, 0, (1 + 4 * static_cast<std::size_t>(count)) * 4);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
        FetchResult r;
        r.words.resize(4 * static_cast<std::size_t>(count));
        std::memcpy(&r.texture_size, raw.data(), 4);
        std::memcpy(r.words.data(), raw.data() + 4, r.words.size() * 4);
        return r;
    }

    /// Compares the fetched words with the model of Table 8.18 applied to `expected` (raw element bytes).
    /// Unsigned normalized formats are compared as floats within `norm_tol`; everything else bit for bit.
    Mismatch TextureBuffer::compare_fetch(const Format& f, const Bytes& expected, int texels, const FetchResult& r, double norm_tol)
    {
        Mismatch m;
        for (int n = 0; n < texels; ++n)
        {
            const Texel e = decode_texel(f, expected.data() + static_cast<std::size_t>(n * f.texel_bytes()));
            for (std::size_t c = 0; c < 4; ++c)
            {
                const std::uint32_t got = r.words[4 * static_cast<std::size_t>(n) + c];
                bool ok = false;
                std::string want, have;
                if (f.norm)
                {
                    const double g = static_cast<double>(std::bit_cast<float>(got));
                    const double err = std::abs(g - e.f[c]);
                    m.max_norm_err = std::max(m.max_norm_err, err);
                    ok = err <= norm_tol;
                    want = std::format("{:.7f}", e.f[c]);
                    have = std::format("{:.7f}", g);
                }
                else
                {
                    ok = got == e.w[c];
                    want = std::format("{:#010x}", e.w[c]);
                    have = std::format("{:#010x}", got);
                }
                if (!ok)
                {
                    if (m.bad == 0) m.first = std::format("texel {} component {}: expected {} got {}", n, "RGBA"[c], want, have);
                    ++m.bad;
                }
            }
        }
        return m;
    }

    // =================================================================================================
    // validate()
    // =================================================================================================
    void TextureBuffer::validate(demo::Env& env, demo::Check& check)
    {
        check_limits(check);
        check_format_table(check);
        check_buffer_range(check);
        check_errors(check);
        check_bindings(check);
        check_level_queries(check);
        check_updates(check);
        check_image_buffer(env, check);
        check_large_texture(check);
        check_compile_rules(check);
        note_out_of_range();
        check_scene(env, check);
        check.expect_no_gl_error("no GL error after all experiments");
    }

    void TextureBuffer::check_limits(demo::Check& check)
    {
        std::cout << std::format("   [note] MAX_TEXTURE_BUFFER_SIZE = {}, TEXTURE_BUFFER_OFFSET_ALIGNMENT = {}\n", max_texture_buffer_size_, alignment_);
        check.expect(max_texture_buffer_size_ >= 65536, "MAX_TEXTURE_BUFFER_SIZE >= 65536 (ES 3.2 Table 21.40 minimum value)",
                     std::format("value {}", max_texture_buffer_size_));
        check.expect(alignment_ >= 1 && alignment_ <= 256,
                     "TEXTURE_BUFFER_OFFSET_ALIGNMENT is between 1 and 256 (Table 21.40: the value is the maximum allowed, 256)",
                     std::format("value {}", alignment_));
    }

    void TextureBuffer::check_format_table(demo::Check& check)
    {
        constexpr int kTexels = 64;
        for (std::size_t fi = 0; fi < kFormats.size(); ++fi)
        {
            const Format& f = kFormats[fi];
            const Bytes data = make_data(f, kTexels, static_cast<std::uint32_t>(fi + 1));
            fill_buffer(data_buf_, data);
            attach(sweep_tex_, f, data_buf_, 0, data.size());  // glTexBuffer's range form; the whole-buffer form is checked below
            const FetchResult r = fetch(f.family(), sweep_tex_, kTexels);
            const Mismatch m = compare_fetch(f, data, kTexels, r, 1e-6);
            const char* sampler = f.family() == Family::Float ? "samplerBuffer" : (f.family() == Family::Int ? "isamplerBuffer" : "usamplerBuffer");
            std::string detail = m.bad == 0 ? "" : std::format("{} of {} words wrong; first: {}", m.bad, kTexels * 4, m.first);
            if (r.texture_size != static_cast<std::uint32_t>(kTexels))
                detail += std::format(" textureSize = {} (expected {})", r.texture_size, kTexels);
            check.expect(m.bad == 0 && r.texture_size == static_cast<std::uint32_t>(kTexels),
                         std::format("{}: {} texels x 4 components through {} match Table 8.18 ({} element(s) of {} byte(s){}), textureSize == {}", f.name, kTexels,
                                     sampler, f.comps, elem_bytes(f.elem), f.norm ? ", normalized" : "", kTexels),
                         detail);
            if (f.norm) std::cout << std::format("   [note] {}: max |fetched - c/255| = {:.2e}\n", f.name, m.max_norm_err);
        }

        // glTexBuffer (the whole buffer) is the same as glTexBufferRange(0, BUFFER_SIZE): ES 3.2 8.9.
        const Format& f = format_named("RGBA16UI");
        const Bytes data = make_data(f, 100, 4242u);
        fill_buffer(data_buf_, data);
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, sweep_tex_);
        mgl::glTexBuffer(mgl::TextureTarget::TextureBuffer, f.gl, data_buf_);
        GLint buffer_size = 0;
        mgl::glGetBufferParameteriv(mgl::BufferTarget::TextureBuffer, mgl::BufferParameter::Size, &buffer_size);
        const FetchResult r = fetch(Family::Uint, sweep_tex_, 100);
        const Mismatch m = compare_fetch(f, data, 100, r, 0);
        check.expect(m.bad == 0 && r.texture_size == 100u && buffer_size == 800,
                     "glTexBuffer attaches the whole store: BUFFER_SIZE 800 bytes -> textureSize 100 RGBA16UI texels, all equal to the data",
                     std::format("buffer size {} textureSize {} first: {}", buffer_size, r.texture_size, m.first));
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_buffer_range(demo::Check& check)
    {
        const std::size_t a = static_cast<std::size_t>(alignment_);
        constexpr std::size_t kStore = 4096;
        Bytes store(kStore);
        for (std::size_t i = 0; i < kStore; ++i) store[i] = static_cast<std::uint8_t>(hash32(static_cast<std::uint32_t>(i) + 9000u) >> 11);
        // Floats and halves in a byte soup could be NaN or denormal; the integer formats (and RGBA8) never are. The float
        // formats get properly generated data in their own region instead.
        struct Case
        {
            const char* fmt;
            std::size_t offset_units;  // offset = units * TEXTURE_BUFFER_OFFSET_ALIGNMENT
            std::size_t size;          // bytes
        };
        const std::array<Case, 7> cases = {{
            {"R8UI", 0, 100},
            {"RGBA8", 1, 203},     // 50 texels, 3 bytes are not part of a texel
            {"RGB32UI", 2, 100},   // 12-byte texels: 8 texels, 4 bytes left
            {"RG16I", 3, 66},      // 16 texels, 2 bytes left
            {"RGBA32I", 5, 160},   // 10 texels
            {"R16UI", 1, 2},       // exactly one texel
            {"R8I", 4, 1},         // exactly one texel, one byte
        }};
        fill_buffer(data_buf_, store);
        for (const Case& c : cases)
        {
            const Format& f = format_named(c.fmt);
            const std::size_t offset = c.offset_units * a;
            const int texels = static_cast<int>(c.size / static_cast<std::size_t>(f.texel_bytes()));
            attach(sweep_tex_, f, data_buf_, offset, c.size);
            const FetchResult r = fetch(f.family(), sweep_tex_, texels);
            Bytes expected(store.begin() + static_cast<std::ptrdiff_t>(offset), store.begin() + static_cast<std::ptrdiff_t>(offset + c.size));
            const Mismatch m = compare_fetch(f, expected, texels, r, 1e-6);
            check.expect(m.bad == 0 && r.texture_size == static_cast<std::uint32_t>(texels),
                         std::format("glTexBufferRange({}, offset {} = {} x alignment, size {}): textureSize == floor(size / {}) == {}, and the {} texels start at the offset", f.name,
                                     offset, c.offset_units, c.size, f.texel_bytes(), texels, texels),
                         std::format("textureSize {} first mismatch: {}", r.texture_size, m.first));
        }

        // Float formats on a range that does not start at byte 0 (generated data: no NaN, no denormals).
        {
            const Format& f = format_named("RGBA32F");
            const Bytes data = make_data(f, 24, 777u);
            Bytes big(a + data.size() + 32, 0xCC);
            std::memcpy(big.data() + a, data.data(), data.size());
            fill_buffer(data_buf_, big);
            attach(sweep_tex_, f, data_buf_, a, data.size());
            const FetchResult r = fetch(Family::Float, sweep_tex_, 24);
            const Mismatch m = compare_fetch(f, data, 24, r, 0);
            check.expect(m.bad == 0 && r.texture_size == 24u, "glTexBufferRange(RGBA32F, offset = alignment, size 384): the 24 texels are exactly the floats at the offset",
                         std::format("textureSize {} first mismatch: {}", r.texture_size, m.first));
        }

        // A re-attach with another format and range replaces the previous one (the texture has one image).
        {
            const Format& f = format_named("R32UI");
            attach(sweep_tex_, f, data_buf_, 0, 64);
            const FetchResult r = fetch(Family::Uint, sweep_tex_, 16);
            check.expect(r.texture_size == 16u, "re-attaching with another format and size replaces the image: textureSize == 64 / 4 == 16", std::format("textureSize {}", r.texture_size));
        }
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_errors(demo::Check& check)
    {
        constexpr std::size_t kStore = 4096;
        const Bytes store(kStore, 0x5A);
        fill_buffer(data_buf_, store);
        const Format& r8 = format_named("RGBA8");
        // A known good attachment first, so that "an erroring command has no effect" (ES 3.2 2.3.1) can be observed.
        attach(sweep_tex_, r8, data_buf_, 0, 64);
        auto size_still_16 = [&](const char* what) {
            const FetchResult r = fetch(Family::Float, sweep_tex_, 1);
            check.expect(r.texture_size == 16u, std::format("{}: the previous attachment is unchanged (textureSize still 16)", what), std::format("textureSize {}", r.texture_size));
            mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, sweep_tex_);
        };
        const mgl::TextureTarget tb = mgl::TextureTarget::TextureBuffer;
        const GLintptr a = alignment_;

        demo::drain_gl_errors();
        if (alignment_ > 1)
        {
            mgl::glTexBufferRange(tb, r8.gl, data_buf_, 1, 64);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: offset 1 is not a multiple of TEXTURE_BUFFER_OFFSET_ALIGNMENT -> INVALID_VALUE (ES 3.2 8.9)");
            size_still_16("misaligned offset");
            mgl::glTexBufferRange(tb, r8.gl, data_buf_, a + 1, 64);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: offset alignment + 1 -> INVALID_VALUE");
        }
        else
        {
            std::cout << "   [note] TEXTURE_BUFFER_OFFSET_ALIGNMENT is 1 here: no misaligned offset exists, that error case cannot be provoked\n";
        }
        mgl::glTexBufferRange(tb, r8.gl, data_buf_, -a, 64);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: negative offset -> INVALID_VALUE");
        mgl::glTexBufferRange(tb, r8.gl, data_buf_, 0, 0);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: size 0 -> INVALID_VALUE (size must be > 0)");
        mgl::glTexBufferRange(tb, r8.gl, data_buf_, 0, -4);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: negative size -> INVALID_VALUE");
        mgl::glTexBufferRange(tb, r8.gl, data_buf_, 0, static_cast<GLsizeiptr>(kStore) + 4);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: offset + size > BUFFER_SIZE (size 4100 of a 4096-byte store) -> INVALID_VALUE");
        size_still_16("size beyond the store");
        mgl::glTexBufferRange(tb, r8.gl, data_buf_, a, static_cast<GLsizeiptr>(kStore));
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glTexBufferRange: offset = alignment with size = BUFFER_SIZE (offset + size > BUFFER_SIZE) -> INVALID_VALUE");
        mgl::glTexBufferRange(tb, r8.gl, mgl::BufferId{0x00FFFFFFu}, 0, 64);
        check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glTexBufferRange: a non-zero name that is no buffer object -> INVALID_OPERATION");
        mgl::glTexBufferRange(mgl::TextureTarget::Texture2D, r8.gl, data_buf_, 0, 64);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glTexBufferRange: target TEXTURE_2D -> INVALID_ENUM");
        mgl::glTexBuffer(mgl::TextureTarget::Texture2D, r8.gl, data_buf_);
        check.expect_gl_error(mgl::ErrorCode::InvalidEnum, "glTexBuffer: target TEXTURE_2D -> INVALID_ENUM");
        const std::array<std::pair<mgl::InternalFormat, const char*>, 3> bad_formats = {{
            {mgl::InternalFormat::Rgb8, "RGB8 (3 components of 8 bits is not in Table 8.18)"},
            {mgl::InternalFormat::Rgb16F, "RGB16F (only RGB32F/I/UI are 3-component formats)"},
            {mgl::InternalFormat::Rgba, "the unsized RGBA (the format must be a sized internal format of Table 8.18)"},
        }};
        for (const auto& [fmt, what] : bad_formats)
        {
            mgl::glTexBuffer(tb, fmt, data_buf_);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum, std::format("glTexBuffer: internalformat {} -> INVALID_ENUM", what));
        }
        size_still_16("after all error cases");

        // buffer == 0 detaches; "the values offset and size are ignored" (ES 3.2 8.9), so invalid ones raise no error.
        mgl::glTexBufferRange(tb, r8.gl, mgl::BufferId{0}, -1, -1);
        check.expect_no_gl_error("glTexBufferRange with buffer 0 detaches and ignores offset and size (here -1 and -1): no error");
        mgl::glTexBuffer(tb, r8.gl, mgl::BufferId{0});
        check.expect_no_gl_error("glTexBuffer with buffer 0 detaches: no error");
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_bindings(demo::Check& check)
    {
        check.expect_eq(initial_buffer_binding_, 0, "TEXTURE_BUFFER_BINDING is 0 initially (Table 21.40)");
        check.expect_eq(initial_texture_binding_, 0, "TEXTURE_BINDING_BUFFER is 0 initially (Table 21.9)");

        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, data_buf_);
        check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBufferBinding), static_cast<int>(data_buf_.value), "glBindBuffer(TEXTURE_BUFFER, b): TEXTURE_BUFFER_BINDING == b");
        check.expect_eq(demo::get_integer(mgl::GetParameter::ArrayBufferBinding), 0, "binding the TEXTURE_BUFFER point leaves ARRAY_BUFFER_BINDING alone");
        check.expect_eq(demo::get_integer(mgl::GetParameter::ShaderStorageBufferBinding), 0, "binding the TEXTURE_BUFFER point leaves SHADER_STORAGE_BUFFER_BINDING alone");
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, decoy_buf_);
        check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBufferBinding), static_cast<int>(decoy_buf_.value), "rebinding another buffer changes TEXTURE_BUFFER_BINDING");
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
        check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBufferBinding), 0, "glBindBuffer(TEXTURE_BUFFER, 0): TEXTURE_BUFFER_BINDING == 0");

        // TEXTURE_BINDING_BUFFER is per texture unit.
        std::array<int, 3> seen{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            mgl::glActiveTexture(static_cast<mgl::TextureUnit>(static_cast<GLenum>(mgl::TextureUnit::Texture0) + static_cast<GLenum>(k)));
            mgl::glBindTexture(mgl::TextureTarget::TextureBuffer, scene_tex_[k]);
            seen[k] = demo::get_integer(mgl::GetParameter::TextureBindingBuffer);
        }
        check.expect(seen[0] == static_cast<int>(scene_tex_[0].value) && seen[1] == static_cast<int>(scene_tex_[1].value) && seen[2] == static_cast<int>(scene_tex_[2].value),
                     "TEXTURE_BINDING_BUFFER returns the buffer texture bound on the active unit (three units, three different textures)",
                     std::format("{} {} {} vs {} {} {}", seen[0], seen[1], seen[2], scene_tex_[0].value, scene_tex_[1].value, scene_tex_[2].value));
        mgl::glActiveTexture(mgl::TextureUnit::Texture0);
        mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
        check.expect_eq(demo::get_integer(mgl::GetParameter::TextureBindingBuffer), static_cast<int>(scene_tex_[0].value),
                        "binding TEXTURE_2D (0) does not change TEXTURE_BINDING_BUFFER (one binding per target)");
    }

    void TextureBuffer::check_level_queries(demo::Check& check)
    {
        // ES 3.2 8.11.3: TEXTURE_BUFFER is a valid glGetTexLevelParameter target; "for targets TEXTURE_BUFFER,
        // TEXTURE_2D_MULTISAMPLE and TEXTURE_2D_MULTISAMPLE_ARRAY, which do not support mipmaps, the maximum value
        // [of level] is zero"; INVALID_VALUE if level is negative or larger than that maximum.
        // The buffer-specific pnames of Table 21.11 (TEXTURE_BUFFER_DATA_STORE_BINDING, TEXTURE_BUFFER_OFFSET,
        // TEXTURE_BUFFER_SIZE) are glGetTexLevelParameteriv state (added to meta-gl as gap G7 during integration).
        const Format& f = format_named("RG16F");
        attach(sweep_tex_, f, data_buf_, 0, 64);
        const mgl::TextureTarget tb = mgl::TextureTarget::TextureBuffer;
        {
            GLint store = -1, offset = -1, size = -1;
            mgl::glGetTexLevelParameteriv(tb, 0, mgl::TextureLevelParameter::BufferDataStoreBinding, &store);
            mgl::glGetTexLevelParameteriv(tb, 0, mgl::TextureLevelParameter::BufferOffset, &offset);
            mgl::glGetTexLevelParameteriv(tb, 0, mgl::TextureLevelParameter::BufferSize, &size);
            check.expect(store == static_cast<GLint>(data_buf_.value) && offset == 0 && size == 64,
                         "glGetTexLevelParameteriv: TEXTURE_BUFFER_DATA_STORE_BINDING / OFFSET / SIZE report the attached range (Table 21.11)",
                         std::format("binding={} (expected {}), offset={}, size={}", store, data_buf_.value, offset, size));
        }
        demo::drain_gl_errors();
        GLint width = -1, internal_format = -1;
        mgl::glGetTexLevelParameteriv(tb, 0, mgl::TextureLevelParameter::Width, &width);
        check.expect_no_gl_error("glGetTexLevelParameteriv(TEXTURE_BUFFER, level 0, TEXTURE_WIDTH): level 0 of a buffer texture is a valid query");
        mgl::glGetTexLevelParameteriv(tb, 0, mgl::TextureLevelParameter::InternalFormat, &internal_format);
        check.expect_no_gl_error("glGetTexLevelParameteriv(TEXTURE_BUFFER, level 0, TEXTURE_INTERNAL_FORMAT): no error");
        GLint ignored = 0;
        mgl::glGetTexLevelParameteriv(tb, 1, mgl::TextureLevelParameter::Width, &ignored);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetTexLevelParameteriv(TEXTURE_BUFFER, level 1): buffer textures have only level 0 -> INVALID_VALUE (8.11.3)");
        mgl::glGetTexLevelParameteriv(tb, -1, mgl::TextureLevelParameter::Width, &ignored);
        check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetTexLevelParameteriv(TEXTURE_BUFFER, level -1) -> INVALID_VALUE");
        std::cout << std::format("   [note] level 0 of a 16-texel RG16F buffer texture: TEXTURE_WIDTH = {}, TEXTURE_INTERNAL_FORMAT = {:#x} (RG16F is {:#x}); ES 3.2 does not say what these return for a buffer texture, so they are not asserted\n",
                                 width, internal_format, static_cast<GLenum>(f.gl));
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_updates(demo::Check& check)
    {
        const Format& f = format_named("RGBA32UI");
        constexpr int kTexels = 16;
        Bytes data = make_data(f, kTexels, 31337u);
        fill_buffer(data_buf_, data);
        attach(sweep_tex_, f, data_buf_, 0, data.size());
        FetchResult r = fetch(Family::Uint, sweep_tex_, kTexels);
        check.expect(compare_fetch(f, data, kTexels, r, 0).bad == 0, "initial contents: 16 RGBA32UI texels equal the data written with glBufferData(TEXTURE_BUFFER)");

        // 1. A different buffer bound to the TEXTURE_BUFFER *binding point* has no effect on the attached texture.
        const Bytes other = make_data(f, kTexels, 99u);
        fill_buffer(decoy_buf_, other);  // leaves decoy_buf_ bound to TEXTURE_BUFFER
        r = fetch(Family::Uint, sweep_tex_, kTexels);
        check.expect(compare_fetch(f, data, kTexels, r, 0).bad == 0,
                     "a decoy buffer bound to the TEXTURE_BUFFER target has no effect on rendering (ES 3.2 8.9): the texture still reads the attached buffer");

        // 2. glBufferSubData through the TEXTURE_BUFFER binding point is visible to the next fetch.
        const std::array<std::uint32_t, 4> patch = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, data_buf_);
        mgl::glBufferSubData(mgl::BufferTarget::TextureBuffer, 4 * 16, static_cast<GLsizeiptr>(sizeof(patch)), patch.data());
        std::memcpy(data.data() + 4 * 16, patch.data(), sizeof(patch));
        r = fetch(Family::Uint, sweep_tex_, kTexels);
        Mismatch m = compare_fetch(f, data, kTexels, r, 0);
        check.expect(m.bad == 0 && r.words[4 * 4] == 0x11111111u && r.words[4 * 4 + 3] == 0x44444444u,
                     "glBufferSubData(TEXTURE_BUFFER, 64, 16 bytes) rewrites texel 4; the next texelFetch sees it, the other 15 texels are untouched", m.first);

        // 3. A write of a single component (4 bytes in the middle of a texel).
        const std::uint32_t one = 0xCAFEF00Du;
        mgl::glBufferSubData(mgl::BufferTarget::TextureBuffer, 9 * 16 + 8, 4, &one);
        std::memcpy(data.data() + 9 * 16 + 8, &one, 4);
        r = fetch(Family::Uint, sweep_tex_, kTexels);
        m = compare_fetch(f, data, kTexels, r, 0);
        check.expect(m.bad == 0 && r.words[4 * 9 + 2] == one, "a 4-byte glBufferSubData inside texel 9 changes exactly its B component", m.first);

        // 4. Negative control: the old data no longer matches (the check above can fail).
        Bytes stale = data;
        std::memcpy(stale.data() + 4 * 16, &other[4 * 16], 16);
        check.expect(compare_fetch(f, stale, kTexels, r, 0).bad > 0, "negative control: a model that missed the update is detected (compare_fetch fails on stale data)");
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_image_buffer(demo::Env& env, demo::Check& check)
    {
        constexpr int kTexels = 100;  // not a multiple of the work group size 64: two groups, the second one partly idle
        const mgl::BufferId dst_buf = new_buffer(), src_buf = new_buffer();
        const mgl::TextureId src_tex = new_texture(), dst_tex = new_texture();
        int tested = 0;
        for (std::size_t fi = 0; fi < kFormats.size(); ++fi)
        {
            const Format& f = kFormats[fi];
            if (f.image_qualifier == nullptr) continue;
            ++tested;
            const char* image_type = f.family() == Family::Float ? "imageBuffer" : (f.family() == Family::Int ? "iimageBuffer" : "uimageBuffer");
            const char* vec = f.family() == Family::Float ? "vec4" : (f.family() == Family::Int ? "ivec4" : "uvec4");
            const std::string defines = std::format("#define FORMAT {}\n#define IMAGE {}\n#define VEC {}\n#define COMPONENTS {}\n", f.image_qualifier, image_type, vec, f.comps == 4 ? 4 : 1);
            const mgl::ProgramId program = compile_compute(env, "shaders/image_copy.comp", defines);
            extra_programs_.push_back(program);

            const Bytes src = make_data(f, kTexels, static_cast<std::uint32_t>(500 + fi));
            const Bytes sentinel(src.size(), 0xCD);
            fill_buffer(src_buf, src);
            fill_buffer(dst_buf, sentinel);
            attach(src_tex, f, src_buf, 0, src.size());
            attach(dst_tex, f, dst_buf, 0, src.size());
            mgl::glBindImageTexture(mgl::ImageUnit{0}, src_tex, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, f.gl);
            mgl::glBindImageTexture(mgl::ImageUnit{1}, dst_tex, 0, GL_FALSE, 0, mgl::ImageAccess::WriteOnly, f.gl);
            {
                GLint name = -1, format = -1, access = -1;
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingName, 1, &name);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingFormat, 1, &format);
                mgl::glGetIntegeri_v(mgl::GetParameter::ImageBindingAccess, 1, &access);
                check.expect(name == static_cast<GLint>(dst_tex.value) && format == static_cast<GLint>(f.gl) && access == static_cast<GLint>(mgl::ImageAccess::WriteOnly),
                             std::format("{}: IMAGE_BINDING_NAME/FORMAT/ACCESS of unit 1 report the buffer texture, the format and WRITE_ONLY", f.name),
                             std::format("name {} format {:#x} access {:#x}", name, format, access));
            }

            mgl::glUseProgram(program);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, ssbo_);
            const std::array<std::uint32_t, 2> zero = {0u, 0u};
            mgl::glBufferData(mgl::BufferTarget::ShaderStorage, sizeof(zero), zero.data(), mgl::BufferUsage::DynamicCopy);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, ssbo_);
            mgl::glDispatchCompute(static_cast<GLuint>((kTexels + 63) / 64), 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate | mgl::MemoryBarrierMask::TextureFetch);

            const Bytes sizes = map_read(mgl::BufferTarget::ShaderStorage, ssbo_, 0, 8);
            std::uint32_t src_size = 0, dst_size = 0;
            std::memcpy(&src_size, sizes.data(), 4);
            std::memcpy(&dst_size, sizes.data() + 4, 4);
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});

            // Expected store: texel i of the source lands at texel n - 1 - i with its four components reversed
            // (the one-component formats are copied). rgba8 goes through floats: equation 2.3 allows one of the two
            // closest integers, so every byte may differ by one.
            Bytes expected(src.size());
            const int tb = f.texel_bytes(), eb = elem_bytes(f.elem);
            for (int i = 0; i < kTexels; ++i)
                for (int c = 0; c < f.comps; ++c)
                {
                    const int from = f.comps == 4 ? 3 - c : c;
                    std::memcpy(&expected[static_cast<std::size_t>((kTexels - 1 - i) * tb + c * eb)], &src[static_cast<std::size_t>(i * tb + from * eb)], static_cast<std::size_t>(eb));
                }
            const Bytes stored = map_read(mgl::BufferTarget::TextureBuffer, dst_buf, 0, src.size());
            int worst = 0, bad = 0;
            for (std::size_t i = 0; i < stored.size(); ++i)
            {
                const int d = std::abs(static_cast<int>(stored[i]) - static_cast<int>(expected[i]));
                worst = std::max(worst, d);
                if (d > (f.norm ? 1 : 0)) ++bad;
            }
            check.expect(bad == 0 && src_size == static_cast<std::uint32_t>(kTexels) && dst_size == static_cast<std::uint32_t>(kTexels),
                         std::format("{}: imageLoad/imageStore on {} ({}) copy 100 texels mirrored; imageSize == 100; the buffer contents (glMapBufferRange) match the model{}", f.name, image_type,
                                     f.image_qualifier, f.norm ? " within one unit per byte (eq. 2.3 rounding)" : " exactly"),
                         std::format("{} bytes wrong (worst {}), imageSize {} / {}", bad, worst, src_size, dst_size));

            // The same stores, read back through a samplerBuffer after a TEXTURE_FETCH barrier.
            const FetchResult r = fetch(f.family(), dst_tex, kTexels);
            const Mismatch m = compare_fetch(f, stored, kTexels, r, 1e-6);
            check.expect(m.bad == 0 && r.texture_size == static_cast<std::uint32_t>(kTexels),
                         std::format("{}: texelFetch of the image-stored texture returns what imageStore wrote (after glMemoryBarrier)", f.name), m.first);
        }
        check.expect_eq(tested, 12, "Tables 8.18 and 8.26 share 12 formats (RGBA8_SNORM has no buffer texture format)");
        for (GLuint u = 0; u < 2; ++u)
            mgl::glBindImageTexture(mgl::ImageUnit{u}, mgl::TextureId{0}, 0, GL_FALSE, 0, mgl::ImageAccess::ReadOnly, mgl::InternalFormat::R32UI);
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_large_texture(demo::Check& check)
    {
        // Every implementation must support buffer textures of MAX_TEXTURE_BUFFER_SIZE >= 65536 texels (Table 21.40).
        constexpr int kTexels = 65536;
        const Format& f = format_named("R8UI");
        const Bytes data = make_data(f, kTexels, 2024u);
        fill_buffer(data_buf_, data);
        attach(sweep_tex_, f, data_buf_, 0, data.size());
        const FetchResult r = fetch(Family::Uint, sweep_tex_, kTexels);
        const Mismatch m = compare_fetch(f, data, kTexels, r, 0);
        check.expect(m.bad == 0 && r.texture_size == static_cast<std::uint32_t>(kTexels),
                     "a 65536-texel R8UI buffer texture (the guaranteed minimum of MAX_TEXTURE_BUFFER_SIZE): textureSize == 65536 and every texel including the last one is right",
                     std::format("textureSize {} first: {}", r.texture_size, m.first));
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::note_out_of_range()
    {
        // ES 3.2 8.9: "the results of a texel fetch are undefined if the specified texel coordinate is negative, or greater
        // than or equal to the clamped number of texels". Not asserted. The texel behind the range lies inside the buffer
        // (so the access is memory safe), its bytes are 0x0A0B0C0D.
        const Format& f = format_named("R32UI");
        std::vector<std::uint32_t> words(64, 0x0A0B0C0Du);
        for (std::size_t i = 0; i < 16; ++i) words[i] = 0x01010101u * static_cast<std::uint32_t>(i + 1);
        Bytes bytes(words.size() * 4);
        std::memcpy(bytes.data(), words.data(), bytes.size());
        fill_buffer(data_buf_, bytes);
        attach(sweep_tex_, f, data_buf_, 0, 64);  // 16 texels
        const FetchResult r = fetch(Family::Uint, sweep_tex_, 20);
        std::cout << std::format("   [note] texelFetch(16..19) behind a 16-texel range (inside the buffer; undefined by ES 3.2 8.9, not asserted): {:#x} {:#x} {:#x} {:#x}; textureSize = {}\n",
                                 r.words[4 * 16], r.words[4 * 17], r.words[4 * 18], r.words[4 * 19], r.texture_size);
        mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
    }

    void TextureBuffer::check_compile_rules(demo::Check& check)
    {
        auto compiles = [](std::string_view version, std::string_view body, std::string& log) {
            const std::string src = std::format("#version {}\nlayout(local_size_x = 1) in;\nlayout(std430, binding = 0) buffer Out {{ vec4 r; }};\n{}\n", version, body);
            const mgl::ShaderId s = demo::compile_shader(mgl::ShaderType::Compute, src, log);
            if (s.value == 0) return false;
            mgl::glDeleteShader(s);
            return true;
        };
        std::string log;
        const bool control = compiles("320 es", "uniform highp samplerBuffer u;\nuniform highp isamplerBuffer iu;\nuniform highp usamplerBuffer uu;\n"
                                                "layout(rgba8, binding = 1) uniform readonly highp imageBuffer img;\n"
                                                "void main() { r = texelFetch(u, 0) + vec4(texelFetch(iu, 1)) + vec4(texelFetch(uu, 2)) + vec4(float(textureSize(u))) + imageLoad(img, 3) + vec4(float(imageSize(img))); }",
                                      log);
        check.expect(control, "control: samplerBuffer / isamplerBuffer / usamplerBuffer with texelFetch(sampler, int), textureSize(sampler) and an imageBuffer compile as GLSL ES 3.20", log);

        struct Negative
        {
            const char* version;
            const char* what;
            const char* body;
        };
        const std::array<Negative, 6> negatives = {{
            {"320 es", "samplerBuffer without a precision qualifier does not compile: no default precision exists for it (GLSL ES 3.20 4.7.3)",
             "uniform samplerBuffer u;\nvoid main() { r = texelFetch(u, 0); }"},
            {"320 es", "texelFetch(samplerBuffer, int, int) has no lod argument (GLSL ES 3.20 8.9.2: gvec4 texelFetch(gsamplerBuffer sampler, int P))",
             "uniform highp samplerBuffer u;\nvoid main() { r = texelFetch(u, 0, 0); }"},
            {"320 es", "texelFetch(samplerBuffer, ivec2) does not exist: the buffer texture coordinate is an int",
             "uniform highp samplerBuffer u;\nvoid main() { r = texelFetch(u, ivec2(0, 0)); }"},
            {"320 es", "textureSize(samplerBuffer, int) does not exist: buffer textures have no levels (8.9.1: highp int textureSize(gsamplerBuffer sampler))",
             "uniform highp samplerBuffer u;\nvoid main() { r = vec4(float(textureSize(u, 0))); }"},
            {"320 es", "texture(samplerBuffer, ...) does not exist: buffer textures are fetched, never filtered (only texelFetch is defined for gsamplerBuffer)",
             "uniform highp samplerBuffer u;\nvoid main() { r = texture(u, 0.5); }"},
            {"310 es", "samplerBuffer is not part of GLSL ES 3.10 (new in 3.20, section 4.1.7); no extension is enabled",
             "uniform highp samplerBuffer u;\nvoid main() { r = vec4(0.0); }"},
        }};
        for (const Negative& n : negatives)
        {
            std::string nlog;
            const bool ok = compiles(n.version, n.body, nlog);
            check.expect(!ok, n.what, ok ? "the shader compiled" : "");
        }
    }

    // =================================================================================================
    // The scene
    // =================================================================================================
    void TextureBuffer::check_scene(demo::Env& env, demo::Check& check)
    {
        // Rebuild the exact picture of the last frame (the experiments above changed unit 0 and the TEXTURE_BUFFER binding).
        render(env, shown_frame_);
        const int w = env.width, h = env.height;
        const std::vector<std::uint8_t> px = demo::read_rgba8(0, 0, w, h);
        auto at = [&](int x, int y) -> Rgba8 {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
            return {px[i], px[i + 1], px[i + 2], px[i + 3]};
        };

        // ---- the strip -------------------------------------------------------------------------------------------
        {
            int bad_color = 0, bad_material = 0, worst = 0;
            std::string first;
            for (int i = 0; i < kInstances; ++i)
            {
                const int x = (2 * i + 1) * w / (2 * kInstances);
                const Instance in = make_instance(i, 0.0);
                const Rgba8 c = at(x, kStripHeight * 3 / 4);
                const int dc = max_channel_diff(c, in.color);
                worst = std::max(worst, dc);
                if (dc > 1)
                {
                    if (first.empty()) first = std::format("cell {} colour {} expected {}", i, demo::format_rgba(c), demo::format_rgba(in.color));
                    ++bad_color;
                }
                const int grey = static_cast<int>(std::lround((in.material + 1) * 0.25 * 255.0));
                const Rgba8 g = at(x, kStripHeight / 4);
                const int dg = max_channel_diff(g, {static_cast<std::uint8_t>(grey), static_cast<std::uint8_t>(grey), static_cast<std::uint8_t>(grey), 255});
                worst = std::max(worst, dg);
                if (dg > 1)
                {
                    if (first.empty()) first = std::format("cell {} material grey {} expected {}", i, demo::format_rgba(g), grey);
                    ++bad_material;
                }
            }
            check.expect(bad_color == 0 && bad_material == 0,
                         "strip: fragment-shader texelFetch shows all 64 colour texels (RGBA8, +-1 for the 8-bit conversion) and all 64 material texels (usamplerBuffer: material m -> grey (m+1)/4, +-1 for the tie at 127.5)",
                         std::format("{} colour, {} material cells wrong; first: {}", bad_color, bad_material, first));
            // Negative control: with the colours shifted by one cell the strip must disagree almost everywhere.
            int differ = 0;
            for (int i = 0; i < kInstances; ++i)
                if (max_channel_diff(at((2 * i + 1) * w / (2 * kInstances), kStripHeight * 3 / 4), make_instance((i + 1) % kInstances, 0.0).color) > 1) ++differ;
            check.expect(differ > kInstances * 9 / 10, "negative control: a strip model shifted by one instance disagrees with the picture", std::format("{} of 64 cells differ", differ));
        }

        // ---- the instanced cubes: a probe every third pixel against a ray caster ----------------------------------
        const double t = static_cast<double>(shown_frame_) / 60.0;
        std::vector<Instance> inst;
        for (int i = 0; i < kInstances; ++i) inst.push_back(make_instance(i, t));
        const int scene_h = h - kStripHeight;
        const double aspect = static_cast<double>(w) / scene_h;
        const dm::Vec3 eye_f = eye_position();
        const V3 eye{eye_f.x, eye_f.y, eye_f.z};
        const V3 forward = normalize(V3{0, 0, 0} - eye);
        const V3 right = normalize(cross(forward, V3{0, 1, 0}));
        const V3 up = cross(right, forward);
        const double tan_half = std::tan(static_cast<double>(dm::radians(45.0f)) * 0.5);

        const Rgba8 clear8 = {static_cast<std::uint8_t>(std::lround(kClear[0] * 255.0f)), static_cast<std::uint8_t>(std::lround(kClear[1] * 255.0f)),
                              static_cast<std::uint8_t>(std::lround(kClear[2] * 255.0f)), 255};
        int compared = 0, skipped = 0, worst = 0, bad = 0, background = 0;
        std::string first;
        std::array<int, kInstances> hits{};
        std::array<int, 6> faces{};
        std::vector<std::array<int, 3>> probes_of_instance(kInstances, {-1, -1, -1});  // x, y, face for one clean probe per instance
        std::vector<std::array<int, 2>> top_probe(kInstances, {-1, -1});               // x, y of one clean probe on the top face
        constexpr double kGrow = 0.01;  // world units: about 0.2 pixel at the default camera distance
        for (int y = kStripHeight + 1; y < h; y += 3)
            for (int x = 1; x < w; x += 3)
            {
                const double nx = (x + 0.5) / w * 2.0 - 1.0, ny = ((y - kStripHeight) + 0.5) / scene_h * 2.0 - 1.0;
                const V3 dir = normalize(forward + right * (nx * tan_half * aspect) + up * (ny * tan_half));
                const Hit nominal = cast_scene(inst, eye, dir, 0.0);
                const Hit grown = cast_scene(inst, eye, dir, kGrow);
                const Hit shrunk = cast_scene(inst, eye, dir, -kGrow);
                const bool same = nominal.id == grown.id && nominal.id == shrunk.id &&
                                  (nominal.id < 0 || (dot(nominal.normal, grown.normal) > 0.99 && dot(nominal.normal, shrunk.normal) > 0.99));
                if (!same)
                {
                    ++skipped;  // within 0.01 world units of a silhouette or a cube edge: the rasterizer may go either way
                    continue;
                }
                ++compared;
                Rgba8 expected = clear8;
                if (nominal.id >= 0)
                {
                    const Instance& in = inst[static_cast<std::size_t>(nominal.id)];
                    expected = shade(in.color, in.material, nominal.normal);
                    ++hits[static_cast<std::size_t>(nominal.id)];
                    const std::size_t face = std::abs(nominal.normal.y) > 0.5 ? (nominal.normal.y > 0 ? 0 : 1) : 2;
                    ++faces[face];
                    if (probes_of_instance[static_cast<std::size_t>(nominal.id)][0] < 0) probes_of_instance[static_cast<std::size_t>(nominal.id)] = {x, y, static_cast<int>(face)};
                    if (face == 0 && top_probe[static_cast<std::size_t>(nominal.id)][0] < 0) top_probe[static_cast<std::size_t>(nominal.id)] = {x, y};
                }
                else ++background;
                const Rgba8 got = at(x, y);
                const int d = max_channel_diff(got, expected);
                worst = std::max(worst, d);
                if (d > 1)
                {
                    if (first.empty())
                        first = std::format("pixel ({}, {}) hit instance {}: got {} expected {}", x, y, nominal.id, demo::format_rgba(got), demo::format_rgba(expected));
                    ++bad;
                }
            }
        int visible = 0;
        for (int c : hits) visible += c > 0 ? 1 : 0;
        std::cout << std::format("   [note] scene: {} probes compared, {} skipped near edges, {} background, worst channel difference {}, {} of {} instances visible, faces: {} top / {} bottom / {} side\n", compared, skipped,
                                 background, worst, visible, kInstances, faces[0], faces[1], faces[2]);
        check.expect(bad == 0 && compared >= 3000 && visible >= kInstances / 2,
                     std::format("instanced scene: {} pixel probes (every 3rd pixel, edges excluded) equal the ray-cast colour of the instance the per-instance buffer texels describe (+-1)", compared),
                     std::format("{} wrong of {} compared, {} instances visible; first: {}", bad, compared, visible, first));

        // Negative control: the same picture against a model whose instance colours are taken one texel too far
        // (evaluated on the clean top-face probes, whose normal is (0, 1, 0)).
        int mismatches = 0, tested = 0;
        for (int i = 0; i < kInstances; ++i)
        {
            const auto& p = top_probe[static_cast<std::size_t>(i)];
            if (p[0] < 0) continue;
            ++tested;
            const Instance& wrong = inst[static_cast<std::size_t>((i + 1) % kInstances)];
            if (max_channel_diff(at(p[0], p[1]), shade(wrong.color, inst[static_cast<std::size_t>(i)].material, V3{0, 1, 0})) > 1) ++mismatches;
        }
        check.expect(tested >= 10 && mismatches >= tested * 9 / 10, "negative control: a model that fetches each instance's colour one texel off disagrees with the picture",
                     std::format("{} of {} top-face probes disagree", mismatches, tested));

        // Updating one colour texel through the TEXTURE_BUFFER binding point changes exactly that instance in the next frame.
        int k = -1;
        for (int i = 0; i < kInstances && k < 0; ++i)
            if (probes_of_instance[static_cast<std::size_t>(i)][0] >= 0 && inst[static_cast<std::size_t>(i)].material == 1) k = i;  // an unlit one: the colour shows as is
        if (k >= 0)
        {
            const auto& p = probes_of_instance[static_cast<std::size_t>(k)];
            const Rgba8 old_color = inst[static_cast<std::size_t>(k)].color;
            const Rgba8 new_color = {static_cast<std::uint8_t>(255 - old_color[0]), static_cast<std::uint8_t>(255 - old_color[1]), static_cast<std::uint8_t>(255 - old_color[2]), 255};
            auto set_color = [&](Rgba8 c) {
                mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, scene_buf_);
                mgl::glBufferSubData(mgl::BufferTarget::TextureBuffer, static_cast<GLintptr>(scene_offset_[1] + static_cast<std::size_t>(k) * 4), 4, c.data());
                mgl::glBindBuffer(mgl::BufferTarget::TextureBuffer, mgl::BufferId{0});
                render(env, shown_frame_);
                return demo::read_pixel(p[0], p[1]);
            };
            const Rgba8 after = set_color(new_color);
            check.expect(max_channel_diff(after, new_color) <= 1,
                         std::format("glBufferSubData(TEXTURE_BUFFER) of instance {}'s colour texel: its pixel ({}, {}) changes to the new colour {}", k, p[0], p[1], demo::format_rgba(new_color)),
                         std::format("got {}", demo::format_rgba(after)));
            const Rgba8 restored = set_color(old_color);
            check.expect(max_channel_diff(restored, old_color) <= 1, "writing the old colour back restores the pixel", std::format("got {} expected {}", demo::format_rgba(restored), demo::format_rgba(old_color)));
        }
        else
        {
            check.expect(false, "an unlit instance with a clean probe exists for the update experiment");
        }

        render(env, shown_frame_);  // leave the final picture in the framebuffer for the capture
    }
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-texture-buffer",
        .title = "Buffer textures",
        .min_es = {3, 2},
        .summary = "glTexBuffer / glTexBufferRange over all 30 formats of ES 3.2 Table 8.18, samplerBuffer / isamplerBuffer / usamplerBuffer texelFetch and textureSize, imageBuffer "
                   "load/store, offset alignment and every error, data updates through the TEXTURE_BUFFER target; a 64-cube instanced scene whose per-instance data lives in buffer textures.",
        .controls = "a/d: orbit, w/s: pitch, q/e: zoom",
        .source_dir = DEMO_SOURCE_DIR,
        .default_frames = 20,
    };
    return demo::run<TextureBuffer>(argc, argv, info);
}
