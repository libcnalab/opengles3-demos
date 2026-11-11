// es31-compute-shared-memory
//
// Shared variables and barriers in OpenGL ES 3.1 compute shaders (ES 3.2 chapter 17 and section 7.11, GLSL ES 3.10
// sections 4.3.8, 8.14 and 8.15). The invocations of one work group cooperate through `shared` memory:
//
//   * a work group reduction (sum, minimum and maximum of 128 values) and a second pass that reduces the 24 partial
//     results of a 3000 element array to one total;
//   * a prefix sum of the whole array in three passes: Hillis-Steele inclusive scans of each work group, a scan of the
//     group totals, and a pass that adds the offsets; plus a work-efficient Blelloch exclusive scan;
//   * a shader whose shared array is exactly MAX_COMPUTE_SHARED_MEMORY_SIZE bytes;
//   * memoryBarrierShared(), groupMemoryBarrier() and memoryBarrier() as the memory half of "barrier(); + fence".
//
// Every result is compared exactly with a CPU computation. Why barrier() is required is explained in the shaders and
// the README and shown by a CPU simulation of schedules the GL allows; the demo never asserts what a GPU does WITHOUT
// the barriers because that result is undefined.
//
// The image is compute-to-render: the buffers the compute shaders wrote are bound as vertex attribute arrays and
// drawn as bars (input, inclusive scan, exclusive scan, per-group sum/min/max). validate() probes every bar.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr std::uint32_t kSentinel = 0xDEADBEEFu;
    constexpr std::uint32_t kCount = 3000;                                 // elements; NOT a multiple of the group size
    constexpr std::uint32_t kGroup = 128;                                  // local size of the reduction / Hillis-Steele scan
    constexpr std::uint32_t kGroups = (kCount + kGroup - 1) / kGroup;      // 24 work groups, the last one holds 56 elements
    constexpr std::uint32_t kSharedGroups = 4;                             // work groups of the shared-memory-limit shader
    constexpr std::uint32_t kPad = 4;                                      // guard words after every result

    constexpr Rgba8 kBackground = {8, 8, 12, 255};
    constexpr Rgba8 kRowColor = {22, 24, 34, 255};
    constexpr Rgba8 kBaselineColor = {70, 74, 92, 255};

    using Words = std::vector<std::uint32_t>;

    std::uint32_t word(int v) { return static_cast<std::uint32_t>(v); }

    // ---------------------------------------------------------------------------------------------------------
    // CPU models
    // ---------------------------------------------------------------------------------------------------------
    struct Partial
    {
        int sum, min, max, count;
    };

    Partial reduce_range(const std::vector<int>& v, std::size_t first, std::size_t last)  // [first, last)
    {
        Partial p{0, std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0};
        for (std::size_t i = first; i < last; ++i)
        {
            p.sum += v[i];
            p.min = std::min(p.min, v[i]);
            p.max = std::max(p.max, v[i]);
            ++p.count;
        }
        return p;
    }

    std::vector<Partial> group_partials(const std::vector<int>& v)
    {
        std::vector<Partial> out;
        for (std::size_t g = 0; g * kGroup < v.size(); ++g)
            out.push_back(reduce_range(v, g * kGroup, std::min<std::size_t>((g + 1) * kGroup, v.size())));
        return out;
    }

    Words partial_words(const std::vector<Partial>& ps)  // layout of the ivec4 results, plus one guard ivec4
    {
        Words w;
        for (const Partial& p : ps) w.insert(w.end(), {word(p.sum), word(p.min), word(p.max), word(p.count)});
        w.insert(w.end(), 4, kSentinel);
        return w;
    }

    // Inclusive / exclusive scan of each group of `group` elements (the last group may be shorter).
    std::vector<int> group_scan(const std::vector<int>& v, std::size_t group, bool inclusive)
    {
        std::vector<int> out(v.size());
        for (std::size_t first = 0; first < v.size(); first += group)
        {
            const std::size_t last = std::min(first + group, v.size());
            int running = 0;
            for (std::size_t i = first; i < last; ++i)
            {
                if (inclusive)
                {
                    running += v[i];
                    out[i] = running;
                }
                else
                {
                    out[i] = running;
                    running += v[i];
                }
            }
        }
        return out;
    }

    Words with_guard(const std::vector<int>& v)
    {
        Words w;
        for (const int x : v) w.push_back(word(x));
        w.insert(w.end(), kPad, kSentinel);
        return w;
    }

    // Schedules the GL may legally choose when a shared-memory algorithm has NO barrier(): the 128 invocations of a group
    // simply run one after the other in some order. (A real GPU runs them in waves, which is the same freedom.)
    //
    // Hillis-Steele without the barrier between a step's read and its write:
    std::vector<int> scan_without_barriers(const std::vector<int>& in, bool ascending)
    {
        std::vector<int> s(in);
        for (unsigned offset = 1; offset < kGroup; offset <<= 1)
            for (unsigned k = 0; k < kGroup; ++k)
            {
                const unsigned lid = ascending ? k : kGroup - 1 - k;
                if (lid >= offset) s[lid] += s[lid - offset];  // read of s[lid - offset] and write of s[lid], back to back
            }
        return s;
    }
    // The reduction without the barrier between the stores of the load phase and the folding steps:
    int reduce_without_barriers(const std::vector<int>& in, bool ascending)
    {
        std::vector<int> s(kGroup, 0);  // shared memory is undefined at the start: any garbage, here zeros
        for (unsigned k = 0; k < kGroup; ++k)
        {
            const unsigned lid = ascending ? k : kGroup - 1 - k;
            s[lid] = in[lid];  // the load phase
            for (unsigned stride = kGroup / 2; stride > 0; stride >>= 1)
                if (lid < stride) s[lid] += s[lid + stride];
        }
        return s[0];
    }

    std::string first_line(std::string log)
    {
        const std::size_t eol = log.find('\n');
        if (eol != std::string::npos) log.resize(eol);
        return log;
    }

    // ---------------------------------------------------------------------------------------------------------
    // GL helpers
    // ---------------------------------------------------------------------------------------------------------
    std::string with_defines(const std::string& source, std::initializer_list<std::string> defines)
    {
        const std::size_t eol = source.find('\n');
        std::string head = source.substr(0, eol + 1);  // the "#version 310 es" line must stay first
        for (const std::string& d : defines) head += "#define " + d + "\n";
        return head + source.substr(eol + 1);
    }

    mgl::ProgramId build_compute(const std::string& source, const std::string& label)
    {
        std::string log;
        const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, source, log);
        if (cs.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", label, log));
        const mgl::ShaderId shaders[] = {cs};
        const mgl::ProgramId program = demo::link_program(shaders, log);
        mgl::glDeleteShader(cs);
        if (program.value == 0) throw demo::Failure(std::format("{} failed to link:\n{}", label, log));
        return program;
    }

    mgl::UniformLocation uniform(mgl::ProgramId program, const char* name)
    {
        const mgl::UniformLocation loc = mgl::glGetUniformLocation(program, name);
        if (loc.value < 0) throw demo::Failure(std::format("uniform '{}' is not active", name));
        return loc;
    }

    mgl::BufferId create_ssbo(const Words& data)
    {
        mgl::BufferId id{};
        mgl::glGenBuffers(1, &id);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, id);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const std::uint32_t>(data), mgl::BufferUsage::DynamicCopy);
        return id;
    }

    // Respecifies a buffer filled with the sentinel (a result that is never written stays recognizable).
    void fill_sentinel(mgl::BufferId id, std::size_t words)
    {
        const Words data(words, kSentinel);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, id);
        mgl::glBufferData(mgl::BufferTarget::ShaderStorage, std::span<const std::uint32_t>(data), mgl::BufferUsage::DynamicCopy);
    }

    // The caller has issued glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT) after the last shader write.
    Words read_ssbo(mgl::BufferId id, std::size_t words)
    {
        Words out(words);
        mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, id);
        const void* mapped = mgl::glMapBufferRange(mgl::BufferTarget::ShaderStorage, 0, static_cast<GLsizeiptr>(words * sizeof(std::uint32_t)),
                                                   mgl::MapBufferAccessMask::Read);
        if (mapped == nullptr) return {};
        std::memcpy(out.data(), mapped, words * sizeof(std::uint32_t));
        (void)mgl::glUnmapBuffer(mgl::BufferTarget::ShaderStorage);
        return out;
    }

    struct Diff
    {
        std::size_t bad = 0;
        std::string first;
    };
    Diff compare_words(const Words& got, const Words& want)
    {
        Diff d;
        if (got.size() != want.size())
        {
            d.bad = std::max(got.size(), want.size());
            d.first = std::format("size {} vs {}", got.size(), want.size());
            return d;
        }
        for (std::size_t i = 0; i < got.size(); ++i)
            if (got[i] != want[i] && d.bad++ == 0)
                d.first = std::format("word {}: got {:#x} ({}) want {:#x} ({})", i, got[i], static_cast<int>(got[i]), want[i], static_cast<int>(want[i]));
        return d;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Presentation: bars drawn from the compute results
    // ---------------------------------------------------------------------------------------------------------
    struct Series
    {
        std::string name;
        std::vector<int> values;  // the CPU model of what the bars show
        int x0 = 0, pitch = 1, width = 1;
        int baseline = 0, row_y0 = 0, row_h = 0;
        float scale = 0.0f;       // pixels per unit
        Rgba8 color{};
    };

    enum class Probe
    {
        Pass,
        Fail,
        Skipped
    };

    class ComputeSharedMemory final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            shared_limit_ = demo::get_integer(mgl::GetParameter::MaxComputeSharedMemorySize);

            // Input: pseudo-random values in [-500, 499] from the runtime's deterministic generator (std::mt19937).
            input_.resize(kCount);
            for (int& v : input_) v = static_cast<int>(env.rng() % 1000u) - 500;

            reduce_a_ = build_compute(with_defines(env.load_text("shaders/reduce.comp"), {"PASS 1", "SHARED_FENCE memoryBarrierShared()"}), "reduce.comp [pass 1, memoryBarrierShared]");
            reduce_b_ = build_compute(with_defines(env.load_text("shaders/reduce.comp"), {"PASS 1", "SHARED_FENCE groupMemoryBarrier()"}), "reduce.comp [pass 1, groupMemoryBarrier]");
            reduce_c_ = build_compute(with_defines(env.load_text("shaders/reduce.comp"), {"PASS 1", "SHARED_FENCE memoryBarrier()"}), "reduce.comp [pass 1, memoryBarrier]");
            reduce_2_ = build_compute(with_defines(env.load_text("shaders/reduce.comp"), {"PASS 2", "SHARED_FENCE memoryBarrierShared()"}), "reduce.comp [pass 2]");
            scan_hs_ = build_compute(env.load_text("shaders/scan_hillis_steele.comp"), "scan_hillis_steele.comp");
            scan_bl_ = build_compute(env.load_text("shaders/scan_blelloch.comp"), "scan_blelloch.comp");
            scan_add_ = build_compute(env.load_text("shaders/scan_add.comp"), "scan_add.comp");
            bars_ = demo::build_program(env, "shaders/bars.vert", "shaders/bars.frag");
            for (const mgl::ProgramId p : {reduce_a_, reduce_b_, reduce_c_, reduce_2_, scan_hs_, scan_bl_, scan_add_})
                u_count_.push_back(uniform(p, "u_count"));
            u_target_ = uniform(bars_, "u_target");
            u_x0_ = uniform(bars_, "u_x0");
            u_pitch_ = uniform(bars_, "u_pitch");
            u_width_ = uniform(bars_, "u_width");
            u_baseline_ = uniform(bars_, "u_baseline");
            u_scale_ = uniform(bars_, "u_scale");
            u_color_ = uniform(bars_, "u_color");

            // Buffers. Results are followed by guard words that no shader may write.
            Words in_words;
            for (const int v : input_) in_words.push_back(word(v));
            buf_in_ = create_ssbo(in_words);
            buf_partials_ = create_ssbo(Words(4 * (kGroups + 1), kSentinel));
            buf_total_ = create_ssbo(Words(4 * 2, kSentinel));
            buf_hs_ = create_ssbo(Words(kCount + kPad, kSentinel));
            buf_hs_totals_ = create_ssbo(Words(kGroups + kPad, kSentinel));
            buf_hs2_ = create_ssbo(Words(kGroups + kPad, kSentinel));
            buf_hs2_totals_ = create_ssbo(Words(1 + kPad, kSentinel));
            buf_full_ = create_ssbo(Words(kCount + kPad, kSentinel));
            buf_bl_ = create_ssbo(Words(kCount + kPad, kSentinel));
            buf_bl_totals_ = create_ssbo(Words(kGroups + kPad, kSentinel));

            // The VAOs that feed the bars: one int attribute per instance, read from the compute results.
            vao_input_ = make_vao(buf_in_, 0, 4);
            vao_inclusive_ = make_vao(buf_hs_, 0, 4);
            vao_exclusive_ = make_vao(buf_bl_, 0, 4);
            vao_sum_ = make_vao(buf_partials_, 0, 16);
            vao_min_ = make_vao(buf_partials_, 4, 16);
            vao_max_ = make_vao(buf_partials_, 8, 16);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            run_pipeline();

            // The compute results are drawn as vertex data: the writes must be visible to the vertex attribute fetch.
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::VertexAttribArray);
            build_series(env);

            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(f(kBackground[0]), f(kBackground[1]), f(kBackground[2]), 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glEnable(mgl::Capability::ScissorTest);
            for (int r = 0; r < 4; ++r)
            {
                const int y0 = env.height - (r + 1) * (env.height / 4);
                mgl::glScissor(0, y0, env.width, env.height / 4 - 2);  // two pixels of background between the rows
                mgl::glClearColor(f(kRowColor[0]), f(kRowColor[1]), f(kRowColor[2]), 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
                mgl::glScissor(0, y0 + (env.height / 4 - 2) / 2, env.width, 1);  // the baseline (value 0)
                mgl::glClearColor(f(kBaselineColor[0]), f(kBaselineColor[1]), f(kBaselineColor[2]), 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);

            mgl::glUseProgram(bars_);
            mgl::glUniform2i(u_target_, env.width, env.height);
            const mgl::VertexArrayId vaos[] = {vao_input_, vao_inclusive_, vao_exclusive_, vao_sum_, vao_min_, vao_max_};
            for (std::size_t i = 0; i < series_.size(); ++i)
            {
                const Series& s = series_[i];
                mgl::glUniform1i(u_x0_, s.x0);
                mgl::glUniform1i(u_pitch_, s.pitch);
                mgl::glUniform1i(u_width_, s.width);
                mgl::glUniform1i(u_baseline_, s.baseline);
                mgl::glUniform1f(u_scale_, s.scale);
                mgl::glUniform3f(u_color_, f(s.color[0]), f(s.color[1]), f(s.color[2]));
                mgl::glBindVertexArray(vaos[i]);
                mgl::glDrawArraysInstanced(mgl::PrimitiveType::TriangleStrip, 0, 4, static_cast<GLsizei>(s.values.size()));
            }
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_reduction(check);
            check_scans(check);
            check_barrier_hazards(check);
            check_shared_limit(env, check);
            check_image(env, check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindBuffer(mgl::BufferTarget::ShaderStorage, mgl::BufferId{0});
            for (GLuint i = 0; i < 3; ++i) mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, i, mgl::BufferId{0});
            for (const mgl::VertexArrayId v : {vao_input_, vao_inclusive_, vao_exclusive_, vao_sum_, vao_min_, vao_max_})
                if (v.value != 0) mgl::glDeleteVertexArrays(1, &v);
            for (const mgl::BufferId b : {buf_in_, buf_partials_, buf_total_, buf_hs_, buf_hs_totals_, buf_hs2_, buf_hs2_totals_, buf_full_, buf_bl_, buf_bl_totals_})
                if (b.value != 0) mgl::glDeleteBuffers(1, &b);
            for (const mgl::ProgramId p : {reduce_a_, reduce_b_, reduce_c_, reduce_2_, scan_hs_, scan_bl_, scan_add_, bars_})
                if (p.value != 0) mgl::glDeleteProgram(p);
        }

    private:
        static float f(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

        mgl::VertexArrayId make_vao(mgl::BufferId buffer, std::uintptr_t offset, GLsizei stride)
        {
            mgl::VertexArrayId vao{};
            mgl::glGenVertexArrays(1, &vao);
            mgl::glBindVertexArray(vao);
            mgl::glBindBuffer(mgl::BufferTarget::Array, buffer);
            const mgl::AttribLocation loc{0u};
            mgl::glEnableVertexAttribArray(loc);
            mgl::glVertexAttribIPointer(loc, 1, mgl::DataType::Int, stride, demo::buffer_offset(offset));
            mgl::glVertexAttribDivisor(loc, 1);
            return vao;
        }

        void bind_storage(std::initializer_list<mgl::BufferId> buffers)
        {
            GLuint index = 0;
            for (const mgl::BufferId b : buffers) mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, index++, b);
        }

        // The whole compute workload: three algorithms, each a chain of dependent dispatches.
        void run_pipeline()
        {
            fill_sentinel(buf_partials_, 4 * (kGroups + 1));
            fill_sentinel(buf_total_, 4 * 2);
            fill_sentinel(buf_hs_, kCount + kPad);
            fill_sentinel(buf_hs_totals_, kGroups + kPad);
            fill_sentinel(buf_hs2_, kGroups + kPad);
            fill_sentinel(buf_hs2_totals_, 1 + kPad);
            fill_sentinel(buf_full_, kCount + kPad);
            fill_sentinel(buf_bl_, kCount + kPad);
            fill_sentinel(buf_bl_totals_, kGroups + kPad);

            // 1. Reduction: pass 1 (24 partial results), pass 2 (one group reduces the 24 partials).
            mgl::glUseProgram(reduce_a_);
            mgl::glUniform1ui(u_count_[0], kCount);
            bind_storage({buf_in_, buf_partials_});
            mgl::glDispatchCompute(kGroups, 1, 1);
            // Pass 2 loads what pass 1 stored through shader buffer variables: SHADER_STORAGE_BARRIER_BIT (ES 3.2 section 7.11.2).
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glUseProgram(reduce_2_);
            mgl::glUniform1ui(u_count_[3], kGroups);
            bind_storage({buf_partials_, buf_total_});
            mgl::glDispatchCompute(1, 1, 1);

            // 2. Hillis-Steele scan: scan every group, scan the group totals with one group, add the offsets.
            mgl::glUseProgram(scan_hs_);
            mgl::glUniform1ui(u_count_[4], kCount);
            bind_storage({buf_in_, buf_hs_, buf_hs_totals_});
            mgl::glDispatchCompute(kGroups, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glUniform1ui(u_count_[4], kGroups);
            bind_storage({buf_hs_totals_, buf_hs2_, buf_hs2_totals_});
            mgl::glDispatchCompute(1, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::ShaderStorage);
            mgl::glUseProgram(scan_add_);
            mgl::glUniform1ui(u_count_[6], kCount);
            bind_storage({buf_hs_, buf_hs2_, buf_full_});
            mgl::glDispatchCompute(kGroups, 1, 1);

            // 3. Blelloch exclusive scan of every group of 128 (64 invocations per group).
            mgl::glUseProgram(scan_bl_);
            mgl::glUniform1ui(u_count_[5], kCount);
            bind_storage({buf_in_, buf_bl_, buf_bl_totals_});
            mgl::glDispatchCompute(kGroups, 1, 1);
        }

        void build_series(const demo::Env& env)
        {
            const std::vector<Partial> ps = group_partials(input_);
            std::vector<int> sums, mins, maxs;
            for (const Partial& p : ps)
            {
                sums.push_back(p.sum);
                mins.push_back(p.min);
                maxs.push_back(p.max);
            }
            const std::vector<int> first128(input_.begin(), input_.begin() + kGroup);
            const std::vector<int> inclusive = group_scan(input_, kGroup, true);
            const std::vector<int> exclusive = group_scan(input_, kGroup, false);

            series_.clear();
            const int row_h = env.height / 4;
            const int bar_pitch = std::max(1, env.width / static_cast<int>(kGroup));
            const int cluster_pitch = std::max(3, env.width / static_cast<int>(kGroups));
            const int cluster_bar = std::max(1, (cluster_pitch - 1) / 3);
            const int margin = (env.width - static_cast<int>(kGroups) * cluster_pitch) / 2;
            auto add = [&](const char* name, std::vector<int> values, int row, int x0, int pitch, int width, Rgba8 color) {
                Series s;
                s.name = name;
                s.values = std::move(values);
                s.row_h = row_h - 2;
                s.row_y0 = env.height - (row + 1) * row_h;
                s.baseline = s.row_y0 + s.row_h / 2;
                s.x0 = x0;
                s.pitch = pitch;
                s.width = width;
                int max_abs = 1;
                for (const int v : s.values) max_abs = std::max(max_abs, std::abs(v));
                s.scale = static_cast<float>(s.row_h / 2 - 2) / static_cast<float>(max_abs);
                s.color = color;
                series_.push_back(std::move(s));
            };
            add("input (first work group)", std::vector<int>(first128), 0, 0, bar_pitch, bar_pitch, {70, 140, 230, 255});
            add("inclusive scan, Hillis-Steele (first work group)", std::vector<int>(inclusive.begin(), inclusive.begin() + kGroup), 1, 0, bar_pitch, bar_pitch, {80, 200, 120, 255});
            add("exclusive scan, Blelloch (first work group)", std::vector<int>(exclusive.begin(), exclusive.begin() + kGroup), 2, 0, bar_pitch, bar_pitch, {240, 160, 50, 255});
            add("per-group sum", sums, 3, margin, cluster_pitch, cluster_bar, {230, 80, 80, 255});
            add("per-group minimum", mins, 3, margin + cluster_bar, cluster_pitch, cluster_bar, {200, 100, 230, 255});
            add("per-group maximum", maxs, 3, margin + 2 * cluster_bar, cluster_pitch, cluster_bar, {240, 220, 80, 255});
        }

        // ---- reduction -----------------------------------------------------------------------------------------
        void check_reduction(demo::Check& check)
        {
            check.expect(shared_limit_ >= 16384, std::format("MAX_COMPUTE_SHARED_MEMORY_SIZE = {} >= 16384 (ES 3.2 Table 21.49)", shared_limit_));
            check.expect(shared_limit_ >= 3 * 128 * 4,
                         std::format("the reduction's 3 shared arrays of 128 ints ({} bytes) fit the limit", 3 * 128 * 4));

            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);  // the pipeline ran in render()
            const std::vector<Partial> ps = group_partials(input_);
            const Diff dp = compare_words(read_ssbo(buf_partials_, 4 * (kGroups + 1)), partial_words(ps));
            check.expect(dp.bad == 0,
                         std::format("reduction pass 1 (memoryBarrierShared): sum, min, max and input count of all {} work groups are exact; the guard record is untouched", kGroups),
                         dp.bad == 0 ? "" : std::format("{} bad; {}", dp.bad, dp.first));

            const Partial total = reduce_range(input_, 0, input_.size());
            const Words want_total = {word(total.sum), word(total.min), word(total.max), kGroups, kSentinel, kSentinel, kSentinel, kSentinel};
            const Diff dt = compare_words(read_ssbo(buf_total_, 8), want_total);
            check.expect(dt.bad == 0,
                         std::format("reduction pass 2 reduces the 24 partials to the array's total: sum {}, min {}, max {}", total.sum, total.min, total.max),
                         dt.bad == 0 ? "" : std::format("{} bad; {}", dt.bad, dt.first));

            // The other two fences in the same kernel: groupMemoryBarrier() and memoryBarrier().
            const struct { mgl::ProgramId program; GLuint count_index; const char* name; } variants[] = {
                {reduce_b_, 1, "groupMemoryBarrier()"}, {reduce_c_, 2, "memoryBarrier()"}};
            for (const auto& v : variants)
            {
                const mgl::BufferId out = create_ssbo(Words(4 * (kGroups + 1), kSentinel));
                mgl::glUseProgram(v.program);
                mgl::glUniform1ui(u_count_[v.count_index], kCount);
                bind_storage({buf_in_, out});
                mgl::glDispatchCompute(kGroups, 1, 1);
                mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
                const Diff dv = compare_words(read_ssbo(out, 4 * (kGroups + 1)), partial_words(ps));
                check.expect(dv.bad == 0, std::format("the same reduction with {} as its fence gives the identical, exact result", v.name),
                             dv.bad == 0 ? "" : std::format("{} bad; {}", dv.bad, dv.first));
                mgl::glDeleteBuffers(1, &out);
            }
            mgl::glUseProgram(mgl::ProgramId{0});

            // Negative control: the comparison reacts to a single changed input element, and only in its own work group.
            std::vector<int> changed = input_;
            changed[1234] += 1;  // element 1234 belongs to work group 9
            const Words got = read_ssbo(buf_partials_, 4 * (kGroups + 1));
            const Words want_changed = partial_words(group_partials(changed));
            std::size_t groups_differing = 0;
            for (std::size_t g = 0; g < kGroups; ++g)
                if (!std::equal(got.begin() + static_cast<std::ptrdiff_t>(4 * g), got.begin() + static_cast<std::ptrdiff_t>(4 * g + 4), want_changed.begin() + static_cast<std::ptrdiff_t>(4 * g)))
                    ++groups_differing;
            check.expect_eq(groups_differing, std::size_t{1}, "negative control: a CPU model with element 1234 changed by +1 differs from the GPU result in exactly one work group (9)");
        }

        // ---- prefix sums ---------------------------------------------------------------------------------------
        void check_scans(demo::Check& check)
        {
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const std::vector<int> inclusive = group_scan(input_, kGroup, true);
            const std::vector<int> exclusive = group_scan(input_, kGroup, false);
            const std::vector<Partial> ps = group_partials(input_);
            std::vector<int> totals;
            for (const Partial& p : ps) totals.push_back(p.sum);

            const Words hs = read_ssbo(buf_hs_, kCount + kPad);
            const Diff d1 = compare_words(hs, with_guard(inclusive));
            check.expect(d1.bad == 0, "Hillis-Steele: the inclusive scan inside each of the 24 work groups is exact (3000 elements, the last group has 56)",
                         d1.bad == 0 ? "" : std::format("{} bad; {}", d1.bad, d1.first));
            const Diff d2 = compare_words(read_ssbo(buf_hs_totals_, kGroups + kPad), with_guard(totals));
            check.expect(d2.bad == 0, "Hillis-Steele: the 24 work group totals are exact (padding with zeros keeps the partial group correct)",
                         d2.bad == 0 ? "" : std::format("{} bad; {}", d2.bad, d2.first));

            std::vector<int> totals_scan(totals.size());
            std::inclusive_scan(totals.begin(), totals.end(), totals_scan.begin());
            const Diff d3 = compare_words(read_ssbo(buf_hs2_, kGroups + kPad), with_guard(totals_scan));
            check.expect(d3.bad == 0, "second level: one work group scans the 24 totals (same program, 24 inputs) exactly",
                         d3.bad == 0 ? "" : std::format("{} bad; {}", d3.bad, d3.first));

            std::vector<int> full(input_.size());
            std::inclusive_scan(input_.begin(), input_.end(), full.begin());
            const Words full_got = read_ssbo(buf_full_, kCount + kPad);
            const Diff d4 = compare_words(full_got, with_guard(full));
            check.expect(d4.bad == 0, "third level: the three-pass scan equals std::inclusive_scan of all 3000 elements, exactly",
                         d4.bad == 0 ? "" : std::format("{} bad; {}", d4.bad, d4.first));
            const Words total_got = read_ssbo(buf_total_, 4);
            check.expect(full_got[kCount - 1] == total_got[0],
                         std::format("two independent GPU algorithms agree: the last element of the scan equals the reduction's sum ({})", static_cast<int>(total_got[0])));

            const Diff d5 = compare_words(read_ssbo(buf_bl_, kCount + kPad), with_guard(exclusive));
            check.expect(d5.bad == 0, "Blelloch: the exclusive scan inside each work group (64 invocations, 2 elements each) is exact",
                         d5.bad == 0 ? "" : std::format("{} bad; {}", d5.bad, d5.first));
            const Diff d6 = compare_words(read_ssbo(buf_bl_totals_, kGroups + kPad), with_guard(totals));
            check.expect(d6.bad == 0, "Blelloch: the 24 work group totals (the root of the up-sweep) are exact",
                         d6.bad == 0 ? "" : std::format("{} bad; {}", d6.bad, d6.first));

            // Relation between the two scans, both read from the GPU: inclusive = exclusive + input.
            const Words bl = read_ssbo(buf_bl_, kCount + kPad);
            bool relation = hs.size() == bl.size();
            for (std::size_t i = 0; relation && i < kCount; ++i)
                relation = hs[i] == bl[i] + word(input_[i]);
            check.expect(relation, "GPU relation: Hillis-Steele inclusive scan == Blelloch exclusive scan + input, for every element");

            // Negative controls.
            check.expect(compare_words(full_got, with_guard(inclusive)).bad > 0, "negative control: the three-pass scan is not the per-group scan (offsets matter)");
            check.expect(compare_words(hs, with_guard(exclusive)).bad > 0, "negative control: the inclusive result differs from the exclusive model");
        }

        // ---- why the barriers exist (CPU simulation of legal schedules; the GPU is never run without them) --------
        void check_barrier_hazards(demo::Check& check)
        {
            const std::vector<int> block(input_.begin(), input_.begin() + kGroup);
            const std::vector<int> correct = group_scan(block, kGroup, true);
            const std::vector<int> ascending = scan_without_barriers(block, true);
            const std::vector<int> descending = scan_without_barriers(block, false);
            check.expect(descending == correct,
                         "CPU simulation: without barriers, ONE legal order of the 128 invocations (highest first) still computes the right scan");
            check.expect(ascending != correct,
                         "CPU simulation: without barriers, ANOTHER legal order (lowest first) computes a wrong scan: an unsynchronized read-modify-write depends on the schedule, which is why the scan needs a barrier between its read and write phases (GLSL ES 3.10 section 8.14)");
            const int sum_correct = reduce_range(block, 0, kGroup).sum;
            check.expect(reduce_without_barriers(block, false) == sum_correct,
                         "CPU simulation: the barrier-free reduction happens to be right when the invocations run highest first");
            check.expect(reduce_without_barriers(block, true) != sum_correct,
                         "CPU simulation: ... and wrong when they run lowest first, because invocation 0 folds before invocation 64 has stored its value");
            std::cout << "   note: the GPU programs always use memoryBarrierShared(); barrier(); at those points, so their results are defined (GLSL ES 3.10 sections 8.14, 8.15).\n"
                         "         This demo deliberately does not run them without barriers: that result is undefined, not 'wrong'.\n";
        }

        // ---- a shader that uses all the shared memory --------------------------------------------------------------
        void check_shared_limit(demo::Env& env, demo::Check& check)
        {
            const std::uint32_t words = static_cast<std::uint32_t>(shared_limit_) / 4;
            std::string log;
            const std::string source = with_defines(env.load_text("shaders/shared_max.comp"), {std::format("SHARED_WORDS {}", words)});
            const mgl::ShaderId cs = demo::compile_shader(mgl::ShaderType::Compute, source, log);
            check.expect(cs.value != 0, std::format("a compute shader declaring `shared uint s[{}]` ({} bytes = MAX_COMPUTE_SHARED_MEMORY_SIZE) compiles", words, words * 4), first_line(log));
            if (cs.value == 0) return;
            const mgl::ShaderId shaders[] = {cs};
            const mgl::ProgramId program = demo::link_program(shaders, log);
            mgl::glDeleteShader(cs);
            check.expect(program.value != 0, "... and links: the limit applies to the total of the declared shared variables (ES 3.2 section 17.1)", first_line(log));
            if (program.value == 0) return;

            const std::size_t total_words = std::size_t{kSharedGroups} * words;
            const mgl::BufferId out = create_ssbo(Words(total_words + kPad, kSentinel));
            mgl::glUseProgram(program);
            mgl::glBindBufferBase(mgl::BufferTarget::ShaderStorage, 0, out);
            mgl::glDispatchCompute(kSharedGroups, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const Words got = read_ssbo(out, total_words + kPad);
            Words want(total_words + kPad, kSentinel);
            for (std::uint32_t g = 0; g < kSharedGroups; ++g)
                for (std::uint32_t k = 0; k < words; ++k)
                {
                    const std::uint32_t source_word = words - 1 - k;  // what the shader reads at position k
                    want[std::size_t{g} * words + k] = (source_word * 2654435761u) ^ (g * 40503u + 17u);
                }
            const Diff d = compare_words(got, want);
            check.expect(d.bad == 0,
                         std::format("all {} words of the full shared array, written by one invocation and read by another, come out exact in each of {} work groups ({} output words); a pattern that depends on the group id would expose any aliasing between groups", words, kSharedGroups, total_words),
                         d.bad == 0 ? "" : std::format("{} bad; {}", d.bad, d.first));
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteBuffers(1, &out);
            mgl::glDeleteProgram(program);

            // For information only: what this implementation does with one more word. The specification does not say
            // that a link error is required, so nothing is asserted.
            const std::string too_big = with_defines(env.load_text("shaders/shared_max.comp"), {std::format("SHARED_WORDS {}", words + 1)});
            const mgl::ShaderId cs2 = demo::compile_shader(mgl::ShaderType::Compute, too_big, log);
            bool linked = false;
            if (cs2.value != 0)
            {
                const mgl::ShaderId shaders2[] = {cs2};
                const mgl::ProgramId p2 = demo::link_program(shaders2, log);
                linked = p2.value != 0;
                if (linked) mgl::glDeleteProgram(p2);
                mgl::glDeleteShader(cs2);
            }
            std::cout << std::format("   note: MAX_COMPUTE_SHARED_MEMORY_SIZE = {} bytes; one more word ({} bytes): compile {}, link {} ({})\n", shared_limit_, (words + 1) * 4,
                                     cs2.value != 0 ? "ok" : "failed", linked ? "ok" : "failed", first_line(log));
        }

        // ---- the image -------------------------------------------------------------------------------------------
        static Probe probe_bar(const std::vector<std::uint8_t>& img, int width, const Series& s, std::size_t i, int value)
        {
            auto pixel = [&](int x, int y) {
                const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4;
                return Rgba8{img[o], img[o + 1], img[o + 2], img[o + 3]};
            };
            const float h = static_cast<float>(value) * s.scale;
            if (std::fabs(h) < 6.0f) return Probe::Skipped;  // too short for margins of 2+ pixels at both ends
            const float tip = static_cast<float>(s.baseline) + h;
            const int x = s.x0 + static_cast<int>(i) * s.pitch;
            // A pixel is covered when its center lies inside the bar. Probe 2+ pixels inside the tip and 2+ pixels beyond it.
            const int inside_y = h > 0 ? static_cast<int>(std::floor(tip)) - 3 : static_cast<int>(std::ceil(tip)) + 2;
            const int outside_y = h > 0 ? static_cast<int>(std::ceil(tip)) + 2 : static_cast<int>(std::floor(tip)) - 3;
            bool ok = pixel(x, inside_y) == s.color;
            if (outside_y >= s.row_y0 && outside_y < s.row_y0 + s.row_h) ok = ok && pixel(x, outside_y) == kRowColor;
            return ok ? Probe::Pass : Probe::Fail;
        }

        void check_image(demo::Env& env, demo::Check& check)
        {
            const std::vector<std::uint8_t> img = demo::read_rgba8(0, 0, env.width, env.height);
            for (const Series& s : series_)
            {
                std::size_t pass = 0, fail = 0, skipped = 0;
                for (std::size_t i = 0; i < s.values.size(); ++i)
                {
                    switch (probe_bar(img, env.width, s, i, s.values[i]))
                    {
                    case Probe::Pass: ++pass; break;
                    case Probe::Fail: ++fail; break;
                    case Probe::Skipped: ++skipped; break;
                    }
                }
                check.expect(fail == 0 && pass > 0,
                             std::format("image, {}: all {} bars with a visible height have the model's color 2+ pixels inside their tip and the row color 2+ pixels beyond it ({} too short to probe)", s.name, pass, skipped),
                             fail == 0 ? "" : std::format("{} bars wrong", fail));
            }

            // Negative control: probe the input bars with the values rotated by one bar. Most must disagree.
            const Series& input = series_[0];
            std::size_t pass = 0, fail = 0;
            for (std::size_t i = 0; i < input.values.size(); ++i)
            {
                const int rotated = input.values[(i + 1) % input.values.size()];
                const Probe p = probe_bar(img, env.width, input, i, rotated);
                if (p == Probe::Pass) ++pass;
                if (p == Probe::Fail) ++fail;
            }
            check.expect(fail >= 3 * (pass + fail) / 5,
                         std::format("negative control: probing the input bars with the values of the neighbouring bars fails for {} of {} bars (the probes are discriminating)", fail, pass + fail));
        }

        GLint shared_limit_ = 0;
        std::vector<int> input_;
        std::vector<Series> series_;

        mgl::ProgramId reduce_a_{}, reduce_b_{}, reduce_c_{}, reduce_2_{}, scan_hs_{}, scan_bl_{}, scan_add_{}, bars_{};
        std::vector<mgl::UniformLocation> u_count_;
        mgl::UniformLocation u_target_{}, u_x0_{}, u_pitch_{}, u_width_{}, u_baseline_{}, u_scale_{}, u_color_{};
        mgl::BufferId buf_in_{}, buf_partials_{}, buf_total_{}, buf_hs_{}, buf_hs_totals_{}, buf_hs2_{}, buf_hs2_totals_{}, buf_full_{}, buf_bl_{}, buf_bl_totals_{};
        mgl::VertexArrayId vao_input_{}, vao_inclusive_{}, vao_exclusive_{}, vao_sum_{}, vao_min_{}, vao_max_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-compute-shared-memory",
        .title = "Compute shaders: shared memory and barriers",
        .min_es = {3, 1},
        .summary = "Work group reduction (sum/min/max) and prefix scans in shared memory with barrier() and memory barriers, "
                   "a shader that fills MAX_COMPUTE_SHARED_MEMORY_SIZE, and the results drawn as bars straight from the SSBOs.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ComputeSharedMemory>(argc, argv, info);
}
