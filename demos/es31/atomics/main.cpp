// es31-atomics
//
// Atomic counters (ES 3.2 section 7.7, GLSL ES 3.10 sections 4.4.6 and 8.10) and atomic memory functions on buffer
// and shared variables (GLSL ES 3.10 section 8.11), under heavy contention: 65536 invocations that all update the
// same few words. The final values of the order-independent operations are known exactly in advance and are
// compared with CPU folds of the same hash; for atomicExchange / atomicCompSwap and for the append pointers of two
// stream compactions the application checks SET properties, because the ORDER in which invocations get their
// values is up to the scheduler.
//
// The 256x256 surface (all of it drawn by shaders that READ what the atomics produced):
//
//  TOP (16 px)     one cell per accumulator of the atomic-memory test: a fragment shader compares the words in the
//                  SSBO with the values the CPU computed (green = equal, red = different; cells 12-15 are unused)
//  MIDDLE (112 px) a 16-bin histogram built with atomicAdd by 65536 invocations; a fragment shader draws the bars
//  BOTTOM (128 px) two 128x128 panels: the PIXELS of a ring pattern that survived a stream compaction. The left
//                  one was compacted with an atomic COUNTER as the append pointer, the right one with atomicAdd on
//                  a buffer variable. They look identical although the order of the compacted lists differs.
//                  When fragment-stage atomic counters are available, the draw also counts its own fragments.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
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
using Words = std::vector<std::uint32_t>;

namespace
{
    constexpr int kGroup = 128;                                // local_size_x of every compute shader
    constexpr int kGroupsHeavy = 512;                          // 65536 invocations
    constexpr int kGroupsLight = 8;                            // 1024 invocations (the range-bound run)
    constexpr std::uint32_t kHeavyN = kGroup * kGroupsHeavy;
    constexpr std::uint32_t kLightN = kGroup * kGroupsLight;
    constexpr std::uint32_t kCompactN = 16384;                 // 128 x 128 elements
    constexpr std::uint32_t kCanary = 0xC5C5C5C5u;
    constexpr std::uint32_t kSentinel = 0xFFFFFFFFu;

    constexpr auto kSsbo = mgl::BufferTarget::ShaderStorage;
    constexpr auto kAtomic = mgl::BufferTarget::AtomicCounter;

    // =========================================================================================
    // 1. The CPU side of the arithmetic
    // =========================================================================================
    /// lowbias32, the same function memory.comp computes (unsigned arithmetic wraps modulo 2^32 on both sides).
    std::uint32_t hash(std::uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    std::uint32_t bits_of(std::int32_t v) { return static_cast<std::uint32_t>(v); }

    // ---- atomic counters -----------------------------------------------------------------------
    // Initial values written with glBufferSubData, and the offsets of the eight counters in counters.comp.
    enum Counter : std::size_t { kTicket, kInc, kDec, kWrap, kUpDown, kOdd, kHist0, kHist1, kCounterCount };
    constexpr std::uint32_t kTicket0 = 5000u, kInc0 = 1000u, kDec0 = 1000000u, kWrap0 = 0xFFFFFFF0u, kUpDown0 = 5u, kOdd0 = 0u, kHist00 = 0u, kHist10 = 100u;
    constexpr std::array<std::uint32_t, kCounterCount> kCounterInit = {kTicket0, kInc0, kDec0, kWrap0, kUpDown0, kOdd0, kHist00, kHist10};
    constexpr std::array<const char*, kCounterCount> kCounterName = {"c_ticket", "c_inc", "c_dec", "c_wrap", "c_updown", "c_odd", "c_hist[0]", "c_hist[1]"};

    /// What every counter holds after `n` invocations ran counters.comp phase 0 (unsigned arithmetic, wraps modulo 2^32).
    std::array<std::uint32_t, kCounterCount> counter_final(std::uint32_t n)
    {
        return {kTicket0 + n, kInc0 + 16u * n, kDec0 - n, kWrap0 + n, kUpDown0, kOdd0 + n / 2u, kHist00 + n, kHist10 + 2u * n};
    }

    // ---- atomic memory functions ---------------------------------------------------------------
    struct Accumulators
    {
        std::array<std::uint32_t, 16> w{};  ///< words 0..15 of the Acc block
        std::array<std::uint32_t, 16> hist{};
    };
    constexpr std::array<const char*, 12> kAccName = {"atomicAdd(uint)", "atomicAdd(int)", "atomicMin(uint)", "atomicMax(uint)", "atomicMin(int)", "atomicMax(int)",
                                                       "atomicAnd", "atomicOr", "atomicXor", "atomicAdd(2^32 - 3)", "atomicCompSwap loop", "atomicAdd ticket"};

    Accumulators acc_initial()
    {
        Accumulators a;
        a.w[0] = 7u;
        a.w[1] = bits_of(-5);
        a.w[2] = 0xFFFFFFFFu;
        a.w[3] = 0u;
        a.w[4] = bits_of(INT_MAX);
        a.w[5] = bits_of(INT_MIN);
        a.w[6] = 0xFFFFFFFFu;
        a.w[7] = 0u;
        a.w[8] = 0x13579BDFu;
        a.w[9] = 100u;
        a.w[10] = 1000u;
        a.w[11] = 0u;
        a.w[12] = kSentinel;  // exchange: the initial value takes part in the set check
        a.w[13] = 0u;         // cas_owner
        return a;
    }

    /// The final value of every order-independent operation after invocations [0, n) ran memory.comp: the same fold in any order.
    Accumulators acc_expected(std::uint32_t n)
    {
        Accumulators a = acc_initial();
        for (std::uint32_t i = 0; i < n; ++i)
        {
            const std::uint32_t h = hash(i);
            a.w[0] += h % 1000u;
            a.w[1] = bits_of(static_cast<std::int32_t>(a.w[1]) + static_cast<std::int32_t>(h % 2001u) - 1000);
            a.w[2] = std::min(a.w[2], h);
            a.w[3] = std::max(a.w[3], h);
            a.w[4] = bits_of(std::min(static_cast<std::int32_t>(a.w[4]), static_cast<std::int32_t>(h)));
            a.w[5] = bits_of(std::max(static_cast<std::int32_t>(a.w[5]), static_cast<std::int32_t>(h)));
            a.w[6] &= ~(1u << (4u + (h >> 8) % 16u));
            a.w[7] |= 1u << ((h >> 3) % 24u);
            a.w[8] ^= h;
            a.w[9] += 0u - 3u;
            a.hist[((h % 16u) * ((h >> 7) % 16u)) / 15u] += 1u;
            if ((i & 63u) == 0u) a.w[10] += 1u;
            a.w[11] += 1u;
        }
        return a;
    }

    /// What the shared-variable atomics leave in each work group (see memory.comp, Groups block).
    struct GroupResult
    {
        std::uint32_t sum = 0, max = 0, or_ = 0, xor_ = 0, min = 0, and_ = 0;
    };
    GroupResult group_expected(std::uint32_t group)
    {
        GroupResult r;
        r.min = bits_of(0x7fffffff);
        r.and_ = 0xffffffffu;
        for (std::uint32_t l = 0; l < kGroup; ++l)
        {
            const std::uint32_t h = hash(group * kGroup + l);
            r.sum += h & 0xfffu;
            r.max = std::max(r.max, h);
            r.or_ |= 1u << (h % 32u);
            r.xor_ ^= h;
            r.min = bits_of(std::min(static_cast<std::int32_t>(r.min), static_cast<std::int32_t>(h)));
            r.and_ &= ~(1u << ((h >> 5) % 32u));
        }
        return r;
    }

    // ---- stream compaction ---------------------------------------------------------------------
    /// Element i of the 128 x 128 input: the squared distance of pixel (x, y) from the centre. The predicate of both compaction shaders keeps the "even rings".
    std::uint32_t compact_value(std::uint32_t i)
    {
        const int x = static_cast<int>(i % 128u) - 64, y = static_cast<int>(i / 128u) - 64;
        return static_cast<std::uint32_t>(x * x + y * y);
    }
    bool compact_keep(std::uint32_t d) { return ((d / 300u) & 1u) == 0u; }

    // ---- set properties ------------------------------------------------------------------------
    /// Every entry in [begin, end) equals `value`; returns the first offending position or `end`.
    std::size_t first_not_equal(const Words& v, std::size_t begin, std::size_t end, std::uint32_t value)
    {
        for (std::size_t i = begin; i < end; ++i)
            if (v[i] != value) return i;
        return end;
    }

    /// atomicExchange: every value that was ever written (the initial one and i + 1 for each invocation) appears exactly once among
    /// the values the invocations received plus the final value.
    bool exchange_is_a_permutation(const Words& received, std::uint32_t final_value, std::uint32_t n, std::uint32_t initial)
    {
        Words got(received.begin(), received.begin() + n);
        got.push_back(final_value);
        Words want;
        want.push_back(initial);
        for (std::uint32_t i = 0; i < n; ++i) want.push_back(i + 1u);
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        return got == want;
    }

    /// atomicCompSwap(owner, 0, id + 1): exactly one invocation sees 0 (the winner, whose id + 1 is the final value); every other one sees the
    /// winner's id + 1.
    bool cas_has_one_winner(const Words& received, std::uint32_t owner, std::uint32_t n)
    {
        std::uint32_t winners = 0, winner_index = 0;
        for (std::uint32_t i = 0; i < n; ++i)
        {
            if (received[i] == 0u)
            {
                ++winners;
                winner_index = i;
            }
            else if (received[i] != owner)
            {
                return false;
            }
        }
        return winners == 1 && owner == winner_index + 1u;
    }

    /// A compacted list is the right SET: the same elements as the expected ascending list, each once, in any order, with consistent values.
    bool compaction_is_right(const Words& out, std::uint32_t count, const Words& expected_idx)
    {
        if (count != expected_idx.size() || count > kCompactN) return false;
        Words idx(out.begin() + 1, out.begin() + 1 + count);
        for (std::uint32_t slot = 0; slot < count; ++slot)
            if (out[1 + kCompactN + slot] != compact_value(out[1 + slot])) return false;
        std::sort(idx.begin(), idx.end());
        return idx == expected_idx;
    }

    // =========================================================================================
    // 2. Small GL helpers
    // =========================================================================================
    /// Inserts `#define <define>` after the #version line.
    std::string with_define(std::string src, std::string_view define)
    {
        src.insert(src.find('\n') + 1, std::format("#define {}\n", define));
        return src;
    }

    mgl::ProgramId build_compute(const demo::Env& env, std::string_view file)
    {
        const std::string src = env.load_text(file);
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

    /// Maps a range of a buffer for reading after glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT) (ES 3.2 section 7.11.2).
    Words read_words(mgl::BufferTarget target, mgl::BufferId buf, GLintptr byte_offset, std::size_t count)
    {
        mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
        mgl::glBindBuffer(target, buf);
        const void* p = mgl::glMapBufferRange(target, byte_offset, static_cast<GLsizeiptr>(count * 4), mgl::MapBufferAccessMask::Read);
        if (p == nullptr) throw demo::Failure("glMapBufferRange failed");
        Words out(count);
        std::memcpy(out.data(), p, count * 4);
        if (mgl::glUnmapBuffer(target) != GL_TRUE) throw demo::Failure("glUnmapBuffer returned FALSE");
        mgl::glBindBuffer(target, mgl::BufferId{0});
        return out;
    }

    void upload_words(mgl::BufferTarget target, mgl::BufferId buf, const Words& w)
    {
        mgl::glBindBuffer(target, buf);
        mgl::glBufferData(target, static_cast<GLsizeiptr>(w.size() * 4), w.data(), mgl::BufferUsage::DynamicDraw);
        mgl::glBindBuffer(target, mgl::BufferId{0});
    }

    bool near(Rgba8 got, Rgba8 want, int tol)
    {
        for (std::size_t i = 0; i < 4; ++i)
            if (std::abs(int{got[i]} - int{want[i]}) > tol) return false;
        return true;
    }

    std::uint8_t to8(float x) { return static_cast<std::uint8_t>(std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)); }

    constexpr Rgba8 kBackground = {14, 16, 22, 255};
    constexpr Rgba8 kGreen = {0, 255, 0, 255};
    constexpr Rgba8 kStripGap = {15, 18, 26, 255};  // vec4(0.06, 0.07, 0.10)
    constexpr Rgba8 kLit = {255, 204, 0, 255};      // vec4(1.0, 0.8, 0.0)
    constexpr Rgba8 kDark = {26, 26, 51, 255};      // vec4(0.1, 0.1, 0.2)

    struct CounterRun
    {
        std::uint32_t n = 0;
        Words counters;  ///< the eight counters after phase 1
        Words seen;      ///< 3 * n
        Words reads;     ///< n + 8
        Words canary;    ///< the buffer words around the counters (range-bound run only)
    };

    struct MemoryRun
    {
        Words acc;     ///< 32 words
        Words out;     ///< 3 * n
        Words groups;  ///< 8 words per work group
    };

    struct CompactRun
    {
        std::uint32_t count = 0;
        Words out;   ///< n, idx[16384], val[16384]
        Words kept;  ///< 16384 words
    };

    // =========================================================================================
    // 3. The demo
    // =========================================================================================
    class Atomics final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 256 || env.height < 256) throw demo::Skip("needs a surface of at least 256 x 256 pixels");
            using P = mgl::GetParameter;
            max_ac_bindings_ = demo::get_integer(P::MaxAtomicCounterBufferBindings);
            max_ac_size_ = demo::get_integer(P::MaxAtomicCounterBufferSize);
            max_cs_ac_buffers_ = demo::get_integer(P::MaxComputeAtomicCounterBuffers);
            max_cs_ac_counters_ = demo::get_integer(P::MaxComputeAtomicCounters);
            max_combined_ac_buffers_ = demo::get_integer(P::MaxCombinedAtomicCounterBuffers);
            max_combined_ac_counters_ = demo::get_integer(P::MaxCombinedAtomicCounters);
            max_fs_ac_buffers_ = demo::get_integer(P::MaxFragmentAtomicCounterBuffers);
            max_fs_ac_counters_ = demo::get_integer(P::MaxFragmentAtomicCounters);
            max_vs_ac_buffers_ = demo::get_integer(P::MaxVertexAtomicCounterBuffers);
            max_cs_blocks_ = demo::get_integer(P::MaxComputeShaderStorageBlocks);
            max_fs_blocks_ = demo::get_integer(P::MaxFragmentShaderStorageBlocks);
            max_group_invocations_ = demo::get_integer(P::MaxComputeWorkGroupInvocations);
            GLint size_x = 0;
            mgl::glGetIntegeri_v(P::MaxComputeWorkGroupSize, 0, &size_x);
            GLint count_x = 0;
            mgl::glGetIntegeri_v(P::MaxComputeWorkGroupCount, 0, &count_x);
            max_group_size_x_ = size_x;
            max_group_count_x_ = count_x;

            // The compute shaders need 128 invocations per group and 512 groups; ES 3.1 guarantees both (Table 20.x: 128 and 65535).
            if (max_group_invocations_ < kGroup || max_group_size_x_ < kGroup || max_group_count_x_ < kGroupsHeavy)
                throw demo::Skip("the work group limits are below the ES 3.1 minimums this demo relies on");

            // Optional stages (the vertex/fragment minimums are 0 in ES 3.1): fragment shader storage blocks draw the displays, fragment atomic counters
            // count the fragments of the panels. Both are used only when the limits allow.
            use_fragment_ssbo_ = max_fs_blocks_ >= 2;
            use_fragment_counters_ = max_fs_ac_buffers_ >= 1 && max_fs_ac_counters_ >= 2 && max_combined_ac_counters_ >= 2 && max_combined_ac_buffers_ >= 1;
            std::cout << std::format("   [note] atomic counter limits: bindings {}, buffer size {}, compute buffers {} counters {}, combined buffers {} counters {}, "
                                     "fragment buffers {} counters {}, vertex buffers {}\n",
                                     max_ac_bindings_, max_ac_size_, max_cs_ac_buffers_, max_cs_ac_counters_, max_combined_ac_buffers_, max_combined_ac_counters_,
                                     max_fs_ac_buffers_, max_fs_ac_counters_, max_vs_ac_buffers_);
            std::cout << std::format("   [note] work groups: {} invocations, size x {}, count x {}; compute SSBO blocks {}, fragment SSBO blocks {}\n", max_group_invocations_,
                                     max_group_size_x_, max_group_count_x_, max_cs_blocks_, max_fs_blocks_);
            if (!use_fragment_ssbo_)
                std::cout << "   [note] MAX_FRAGMENT_SHADER_STORAGE_BLOCKS < 2: the fragment-shader displays are skipped\n";
            if (!use_fragment_counters_)
                std::cout << "   [note] fragment-stage atomic counters are not available (limits too small): the fragment counting is skipped\n";

            counters_ = build_compute(env, "shaders/counters.comp");
            memory_ = build_compute(env, "shaders/memory.comp");
            compact_counter_ = build_compute(env, "shaders/compact_counter.comp");
            compact_ssbo_ = build_compute(env, "shaders/compact_ssbo.comp");
            scatter_ = build_compute(env, "shaders/scatter.comp");
            racy_ = build_compute(env, "shaders/racy.comp");
            u_n_[0] = locate(counters_, "u_n");
            u_phase_ = locate(counters_, "u_phase");
            u_ticket0_ = locate(counters_, "u_ticket0");
            u_dec0_ = locate(counters_, "u_dec0");
            u_wrap0_ = locate(counters_, "u_wrap0");
            u_n_[1] = locate(memory_, "u_n");
            u_count_ = locate(scatter_, "u_count");
            if (use_fragment_ssbo_)
            {
                strip_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/strip.frag");
                hist_ = demo::build_program(env, "shaders/fullscreen.vert", "shaders/hist.frag");
                u_hist_scale_ = locate(hist_, "u_scale");
                u_hist_y0_ = locate(hist_, "u_y0");
                const std::string vs = env.load_text("shaders/fullscreen.vert");
                std::string fs = env.load_text("shaders/panel.frag");
                if (use_fragment_counters_) fs = with_define(fs, "FRAGMENT_COUNTERS");
                panel_ = demo::build_program_from_source(vs, fs, "panel");
                u_panel_x0_ = locate(panel_, "u_x0");
            }

            mgl::glGenBuffers(static_cast<GLsizei>(bufs_.size()), bufs_.data());
            // Atomic counter buffers: allocated with zeros, counters then set with glBufferSubData.
            upload_words(kAtomic, bufs_[kAcA], Words(8, 0));
            Words canary_b(16, kCanary);
            upload_words(kAtomic, bufs_[kAcB], canary_b);  // 64 bytes: the counters live in the middle 32 (range offset 16)
            upload_words(kAtomic, bufs_[kAcComp], Words(8, 0));
            upload_words(kAtomic, bufs_[kAcFrag], Words(8, 0));
            upload_words(kSsbo, bufs_[kSeen], Words(3 * kHeavyN, 0));
            upload_words(kSsbo, bufs_[kReads], Words(kHeavyN + 8, 0));
            upload_words(kSsbo, bufs_[kAcc], Words(32, 0));
            upload_words(kSsbo, bufs_[kExpect], Words(16, 0));
            upload_words(kSsbo, bufs_[kMemOut], Words(3 * kHeavyN, 0));
            upload_words(kSsbo, bufs_[kGroups], Words(8 * kGroupsHeavy, 0));
            upload_words(kSsbo, bufs_[kRacy], Words(2, 0));
            Words input(kCompactN);
            for (std::uint32_t i = 0; i < kCompactN; ++i)
            {
                input[i] = compact_value(i);
                if (compact_keep(input[i])) expected_idx_.push_back(i);
            }
            upload_words(kSsbo, bufs_[kIn], input);
            upload_words(kSsbo, bufs_[kOutA], Words(1 + 2 * kCompactN, kSentinel));
            upload_words(kSsbo, bufs_[kOutB], Words(1 + 2 * kCompactN, kSentinel));
            upload_words(kSsbo, bufs_[kKeptA], Words(kCompactN, 0));
            upload_words(kSsbo, bufs_[kKeptB], Words(kCompactN, 0));
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kBackground[0] / 255.0f, kBackground[1] / 255.0f, kBackground[2] / 255.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);

            heavy_ = run_counters(kGroupsHeavy, false);
            light_ = run_counters(kGroupsLight, true);
            memory_run_ = run_memory();
            racy_lost_ = run_racy();
            compact_[0] = run_compaction(false);
            compact_[1] = run_compaction(true);
            draw_displays(env);

            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 4; ++i) mgl::glBindBufferBase(kSsbo, i, mgl::BufferId{0});
            mgl::glBindBufferBase(kAtomic, 0, mgl::BufferId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_counters(check);
            check_counter_introspection(check);
            check_memory(check);
            check_compaction(check);
            check_displays(env, check);
            check_bindings_and_errors(check);
            check_limits(check);
            negative_controls(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            for (GLuint i = 0; i < 4; ++i) mgl::glBindBufferBase(kSsbo, i, mgl::BufferId{0});
            mgl::glBindBufferBase(kAtomic, 0, mgl::BufferId{0});
            mgl::glBindBuffer(kSsbo, mgl::BufferId{0});
            mgl::glBindBuffer(kAtomic, mgl::BufferId{0});
            mgl::glDeleteBuffers(static_cast<GLsizei>(bufs_.size()), bufs_.data());
            for (mgl::ProgramId p : {counters_, memory_, compact_counter_, compact_ssbo_, scatter_, racy_, strip_, hist_, panel_})
                if (p.value != 0) mgl::glDeleteProgram(p);
        }

    private:
        // Buffer slots.
        static constexpr std::size_t kAcA = 0, kAcB = 1, kAcComp = 2, kAcFrag = 3, kSeen = 4, kReads = 5, kAcc = 6, kExpect = 7, kMemOut = 8, kGroups = 9, kRacy = 10,
                                     kIn = 11, kOutA = 12, kOutB = 13, kKeptA = 14, kKeptB = 15, kBufferCount = 16;

        // ----------------------------------------------------------------------------------
        // pass 1: atomic counters
        // ----------------------------------------------------------------------------------
        /// Runs counters.comp twice (update, then read after glMemoryBarrier(ATOMIC_COUNTER_BARRIER_BIT)) with `groups` work groups.
        /// `use_range`: the counters live at byte offset 16 of a 64-byte buffer and are bound with glBindBufferRange.
        CounterRun run_counters(int groups, bool use_range)
        {
            CounterRun r;
            r.n = static_cast<std::uint32_t>(groups * kGroup);
            const mgl::BufferId ac = use_range ? bufs_[kAcB] : bufs_[kAcA];
            const GLintptr base = use_range ? 16 : 0;

            // Counter initialisation with glBufferSubData (the buffer was allocated by glBufferData with zeros / canaries).
            mgl::glBindBuffer(kAtomic, ac);
            for (std::size_t c = 0; c < kCounterCount; ++c) mgl::glBufferSubData(kAtomic, base + static_cast<GLintptr>(4 * c), 4, &kCounterInit[c]);
            mgl::glBindBuffer(kAtomic, mgl::BufferId{0});
            upload_words(kSsbo, bufs_[kSeen], Words(3 * r.n, 0));
            upload_words(kSsbo, bufs_[kReads], Words(r.n + 8, 0));

            if (use_range) mgl::glBindBufferRange(kAtomic, 0, ac, 16, 32);  // layout(binding = 0): the first counter is at byte 16 of the buffer
            else mgl::glBindBufferBase(kAtomic, 0, ac);
            mgl::glBindBufferBase(kSsbo, 0, bufs_[kSeen]);
            mgl::glBindBufferBase(kSsbo, 1, bufs_[kReads]);
            mgl::glUseProgram(counters_);
            mgl::glUniform1ui(u_n_[0], r.n);
            mgl::glUniform1ui(u_ticket0_, kTicket0);
            mgl::glUniform1ui(u_dec0_, kDec0);
            mgl::glUniform1ui(u_wrap0_, kWrap0);

            mgl::glUniform1ui(u_phase_, 0);
            mgl::glDispatchCompute(static_cast<GLuint>(groups), 1, 1);
            // Phase 1 reads the counters the previous dispatch modified: "Accesses to atomic counters after the barrier will reflect writes prior to the barrier".
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::AtomicCounter);
            mgl::glUniform1ui(u_phase_, 1);
            mgl::glDispatchCompute(static_cast<GLuint>(groups), 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate | mgl::MemoryBarrierMask::ShaderStorage);

            const Words buffer = read_words(kAtomic, ac, 0, use_range ? 16 : 8);
            r.counters.assign(buffer.begin() + (use_range ? 4 : 0), buffer.begin() + (use_range ? 12 : 8));
            if (use_range)
            {
                r.canary.assign(buffer.begin(), buffer.begin() + 4);
                r.canary.insert(r.canary.end(), buffer.begin() + 12, buffer.end());
            }
            r.seen = read_words(kSsbo, bufs_[kSeen], 0, 3 * static_cast<std::size_t>(r.n));
            r.reads = read_words(kSsbo, bufs_[kReads], 0, r.n + 8);
            return r;
        }

        // ----------------------------------------------------------------------------------
        // pass 2: atomic memory functions
        // ----------------------------------------------------------------------------------
        MemoryRun run_memory()
        {
            Words acc(32, 0);
            const Accumulators init = acc_initial();
            for (std::size_t i = 0; i < 16; ++i) acc[i] = init.w[i];
            upload_words(kSsbo, bufs_[kAcc], acc);
            upload_words(kSsbo, bufs_[kMemOut], Words(3 * kHeavyN, 0));
            upload_words(kSsbo, bufs_[kGroups], Words(8 * kGroupsHeavy, 0));
            mgl::glBindBufferBase(kSsbo, 0, bufs_[kAcc]);
            mgl::glBindBufferBase(kSsbo, 1, bufs_[kMemOut]);
            mgl::glBindBufferBase(kSsbo, 2, bufs_[kGroups]);
            mgl::glUseProgram(memory_);
            mgl::glUniform1ui(u_n_[1], kHeavyN);
            mgl::glDispatchCompute(kGroupsHeavy, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate | mgl::MemoryBarrierMask::ShaderStorage);

            MemoryRun r;
            r.acc = read_words(kSsbo, bufs_[kAcc], 0, 32);
            r.out = read_words(kSsbo, bufs_[kMemOut], 0, 3 * static_cast<std::size_t>(kHeavyN));
            r.groups = read_words(kSsbo, bufs_[kGroups], 0, 8 * static_cast<std::size_t>(kGroupsHeavy));

            // The expected values for the strip display (cells 0-11), computed by the CPU.
            const Accumulators want = acc_expected(kHeavyN);
            Words expect(16, 0);
            for (std::size_t i = 0; i < 12; ++i) expect[i] = want.w[i];
            upload_words(kSsbo, bufs_[kExpect], expect);
            return r;
        }

        /// The racy counter-example: how many of 65536 plain increments survive. Returns the lost count, printed only.
        std::uint32_t run_racy()
        {
            upload_words(kSsbo, bufs_[kRacy], Words(2, 0));
            mgl::glBindBufferBase(kSsbo, 0, bufs_[kRacy]);
            mgl::glUseProgram(racy_);
            mgl::glDispatchCompute(kGroupsHeavy, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate);
            const Words w = read_words(kSsbo, bufs_[kRacy], 0, 2);
            racy_plain_ = w[0];
            racy_atomic_ = w[1];
            return kHeavyN - std::min(w[0], kHeavyN);
        }

        // ----------------------------------------------------------------------------------
        // pass 3: stream compaction
        // ----------------------------------------------------------------------------------
        CompactRun run_compaction(bool ssbo_append)
        {
            const std::size_t out = ssbo_append ? kOutB : kOutA;
            const std::size_t kept = ssbo_append ? kKeptB : kKeptA;
            upload_words(kSsbo, bufs_[out], Words(1 + 2 * kCompactN, kSentinel));
            Words zero_n(1, 0);
            mgl::glBindBuffer(kSsbo, bufs_[out]);
            mgl::glBufferSubData(kSsbo, 0, 4, zero_n.data());  // the append pointer of the SSBO variant starts at 0
            mgl::glBindBuffer(kSsbo, mgl::BufferId{0});
            upload_words(kSsbo, bufs_[kept], Words(kCompactN, 0));

            mgl::glBindBufferBase(kSsbo, 0, bufs_[kIn]);
            mgl::glBindBufferBase(kSsbo, 1, bufs_[out]);
            if (!ssbo_append)
            {
                mgl::glBindBuffer(kAtomic, bufs_[kAcComp]);
                const std::uint32_t zero = 0;
                mgl::glBufferSubData(kAtomic, 0, 4, &zero);  // initialise the append pointer with glBufferSubData
                mgl::glBindBuffer(kAtomic, mgl::BufferId{0});
                mgl::glBindBufferBase(kAtomic, 0, bufs_[kAcComp]);
            }
            mgl::glUseProgram(ssbo_append ? compact_ssbo_ : compact_counter_);
            mgl::glDispatchCompute(kCompactN / kGroup, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate | mgl::MemoryBarrierMask::ShaderStorage | mgl::MemoryBarrierMask::AtomicCounter);

            CompactRun r;
            r.out = read_words(kSsbo, bufs_[out], 0, 1 + 2 * static_cast<std::size_t>(kCompactN));
            r.count = ssbo_append ? r.out[0] : read_words(kAtomic, bufs_[kAcComp], 0, 1)[0];

            // The mask the fragment shader draws: scatter the list back into a 128 x 128 image.
            mgl::glBindBufferBase(kSsbo, 1, bufs_[out]);
            mgl::glBindBufferBase(kSsbo, 2, bufs_[kept]);
            mgl::glUseProgram(scatter_);
            mgl::glUniform1ui(u_count_, std::min(r.count, kCompactN));
            mgl::glDispatchCompute(kCompactN / kGroup, 1, 1);
            mgl::glMemoryBarrier(mgl::MemoryBarrierMask::BufferUpdate | mgl::MemoryBarrierMask::ShaderStorage);
            r.kept = read_words(kSsbo, bufs_[kept], 0, kCompactN);
            return r;
        }

        // ----------------------------------------------------------------------------------
        // the three displays
        // ----------------------------------------------------------------------------------
        static constexpr int kStripH = 16, kHistH = 112, kPanel = 128;

        [[nodiscard]] static std::array<std::uint32_t, 16> histogram_expected() { return acc_expected(kHeavyN).hist; }

        [[nodiscard]] static float hist_scale()
        {
            const auto h = histogram_expected();
            return 100.0f / static_cast<float>(*std::max_element(h.begin(), h.end()));
        }

        void draw_displays(const demo::Env& env)
        {
            if (!use_fragment_ssbo_) return;
            const int strip_y = env.height - kStripH;
            const int hist_y = env.height - kStripH - kHistH;

            mgl::glBindBufferBase(kSsbo, 0, bufs_[kAcc]);
            mgl::glBindBufferBase(kSsbo, 1, bufs_[kExpect]);
            mgl::glUseProgram(strip_);
            mgl::glViewport(0, strip_y, env.width, kStripH);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            mgl::glUseProgram(hist_);
            mgl::glUniform1f(u_hist_scale_, hist_scale());
            mgl::glUniform1f(u_hist_y0_, static_cast<float>(hist_y));
            mgl::glViewport(0, hist_y, env.width, kHistH);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            mgl::glUseProgram(panel_);
            for (int p = 0; p < 2; ++p)
            {
                mgl::glBindBufferBase(kSsbo, 0, bufs_[p == 0 ? kKeptA : kKeptB]);
                if (use_fragment_counters_)
                {
                    mgl::glBindBuffer(kAtomic, bufs_[kAcFrag]);
                    const std::array<std::uint32_t, 2> zero = {0, 0};
                    mgl::glBufferSubData(kAtomic, 0, 8, zero.data());
                    mgl::glBindBuffer(kAtomic, mgl::BufferId{0});
                    mgl::glBindBufferBase(kAtomic, 0, bufs_[kAcFrag]);
                }
                mgl::glUniform1f(u_panel_x0_, static_cast<float>(p * kPanel));
                mgl::glViewport(p * kPanel, 0, kPanel, kPanel);
                mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
                if (use_fragment_counters_) fragment_counts_[static_cast<std::size_t>(p)] = read_words(kAtomic, bufs_[kAcFrag], 0, 2);
            }
        }

        // ----------------------------------------------------------------------------------
        // checks: atomic counters
        // ----------------------------------------------------------------------------------
        void check_counters(demo::Check& check)
        {
            auto describe = [](const CounterRun& r) {
                const auto want = counter_final(r.n);
                std::string bad;
                for (std::size_t c = 0; c < kCounterCount; ++c)
                    if (r.counters[c] != want[c]) bad += std::format("{} = {} (want {}); ", kCounterName[c], r.counters[c], want[c]);
                return bad;
            };
            auto sets = [](const CounterRun& r) {
                std::string bad;
                const char* names[3] = {"tickets", "decrement results", "wrap-around results"};
                for (std::size_t k = 0; k < 3; ++k)
                {
                    const std::size_t from = k * r.n, to = from + r.n;
                    const std::size_t p = first_not_equal(r.seen, from, to, 1u);
                    if (p != to) bad += std::format("{}: value {} was returned {} times; ", names[k], p - from, r.seen[p]);
                }
                return bad;
            };

            const CounterRun& h = heavy_;
            const auto want = counter_final(h.n);
            check.expect(describe(h).empty(),
                         std::format("{} invocations on 8 atomic counters, initialised with glBufferSubData: every final value is exact (c_ticket {} = 5000 + N, c_inc {} = 1000 + 16 N, c_dec {} = 1000000 - N, c_odd {} = N/2, c_hist {} / {}, c_updown {} unchanged)",
                                     h.n, want[kTicket], want[kInc], want[kDec], want[kOdd], want[kHist0], want[kHist1], want[kUpDown]),
                         describe(h));
            check.expect(h.counters[kWrap] == kWrap0 + h.n && h.counters[kWrap] < kWrap0,
                         std::format("c_wrap started at 0xFFFFFFF0 and ended at {} (= N - 16): the counter wraps around zero (GLSL ES 3.10 section 8.10: \"will wrap to [0, 2^32-1]\")", h.counters[kWrap]),
                         std::format("{}", h.counters[kWrap]));
            check.expect(sets(h).empty(),
                         "atomicCounterIncrement returned each ticket exactly once, atomicCounterDecrement each result exactly once, and the wrapping counter each of its N values exactly once (65536 x 3 values, a histogram of ones)",
                         sets(h));
            const std::size_t differing = first_not_equal(h.reads, 0, h.n, want[kInc]);
            bool eight = true;
            for (std::size_t c = 0; c < kCounterCount; ++c) eight = eight && h.reads[h.n + c] == want[c];
            check.expect(differing == h.n && eight,
                         "after glMemoryBarrier(ATOMIC_COUNTER_BARRIER_BIT) a second dispatch reads the counters with atomicCounter(): all 65536 invocations see c_inc = 1049576 and the eight counters read the same final values as the buffer",
                         differing == h.n ? "" : std::format("invocation {} read {}", differing, h.reads[differing]));

            const CounterRun& l = light_;
            check.expect(describe(l).empty() && sets(l).empty(),
                         std::format("the same with the counters at byte offset 16 of a 64-byte buffer bound by glBindBufferRange(ATOMIC_COUNTER_BUFFER, 0, buf, 16, 32): all {} counters exact (c_inc {}, c_ticket {})", static_cast<std::size_t>(kCounterCount),
                                     counter_final(l.n)[kInc], counter_final(l.n)[kTicket]),
                         describe(l) + sets(l));
            check.expect(std::all_of(l.canary.begin(), l.canary.end(), [](std::uint32_t w) { return w == kCanary; }),
                         "... and the 16 bytes before and after the bound range still hold their sentinel: counters are addressed relative to the start of the bound range");
        }

        // ----------------------------------------------------------------------------------
        // introspection of the atomic counter interface
        // ----------------------------------------------------------------------------------
        static GLint prop(mgl::ProgramId prog, mgl::ProgramInterface iface, GLuint index, mgl::ProgramResourceProperty p)
        {
            GLint v = -12345;
            mgl::glGetProgramResourceiv(prog, iface, index, 1, &p, 1, nullptr, &v);
            return v;
        }

        void check_counter_introspection(demo::Check& check)
        {
            const mgl::ProgramId prog = counters_;
            using I = mgl::ProgramInterface;
            using R = mgl::ProgramResourceProperty;
            GLint buffers = -1;
            mgl::glGetProgramiv(prog, mgl::ProgramParameter::ActiveAtomicCounterBuffers, &buffers);
            GLint resources = -1, max_vars = -1;
            mgl::glGetProgramInterfaceiv(prog, I::AtomicCounterBuffer, mgl::ProgramInterfaceParameter::ActiveResources, &resources);
            mgl::glGetProgramInterfaceiv(prog, I::AtomicCounterBuffer, mgl::ProgramInterfaceParameter::MaxNumActiveVariables, &max_vars);
            check.expect(buffers == 1 && resources == 1 && max_vars == 7,
                         "glGetProgramiv(ACTIVE_ATOMIC_COUNTER_BUFFERS) = 1 and the ATOMIC_COUNTER_BUFFER interface has 1 active resource (all counters share binding 0) with MAX_NUM_ACTIVE_VARIABLES 7 (c_hist[2] is one array variable)",
                         std::format("{} {} {}", buffers, resources, max_vars));

            const std::array<R, 5> bprops = {R::BufferBinding, R::BufferDataSize, R::NumActiveVariables, R::ReferencedByComputeShader, R::ReferencedByFragmentShader};
            std::array<GLint, 5> b{};
            mgl::glGetProgramResourceiv(prog, I::AtomicCounterBuffer, 0, static_cast<GLsizei>(bprops.size()), bprops.data(), static_cast<GLsizei>(b.size()), nullptr, b.data());
            check.expect(b[0] == 0 && b[2] == 7 && b[3] == 1 && b[4] == 0,
                         "ATOMIC_COUNTER_BUFFER resource 0: BUFFER_BINDING 0 (layout(binding = 0)), NUM_ACTIVE_VARIABLES 7, referenced by the compute shader only",
                         std::format("binding {} variables {} compute {} fragment {}", b[0], b[2], b[3], b[4]));
            check.expect(b[1] >= 32, "ATOMIC_COUNTER_BUFFER resource 0: BUFFER_DATA_SIZE is at least 32 bytes, the end of the last counter (offset 24 + two counters of 4 bytes)",
                         std::format("{}{}", b[1], b[1] == 32 ? " (exactly 32)" : ""));

            struct Var { const char* name; int offset; int array_size; int array_stride; };
            static constexpr std::array<Var, 7> vars = {{{"c_ticket", 0, 1, 0}, {"c_inc", 4, 1, 0}, {"c_dec", 8, 1, 0}, {"c_wrap", 12, 1, 0},
                                                         {"c_updown", 16, 1, 0}, {"c_odd", 20, 1, 0}, {"c_hist[0]", 24, 2, 4}}};
            std::string bad;
            std::set<GLint> indices;
            for (const Var& v : vars)
            {
                const GLuint idx = mgl::glGetProgramResourceIndex(prog, I::Uniform, v.name);
                if (idx == GL_INVALID_INDEX)
                {
                    bad += std::string(v.name) + " (not active); ";
                    continue;
                }
                indices.insert(static_cast<GLint>(idx));
                const GLint acb = prop(prog, I::Uniform, idx, R::AtomicCounterBufferIndex), off = prop(prog, I::Uniform, idx, R::Offset),
                            size = prop(prog, I::Uniform, idx, R::ArraySize), stride = prop(prog, I::Uniform, idx, R::ArrayStride),
                            type = prop(prog, I::Uniform, idx, R::Type), block = prop(prog, I::Uniform, idx, R::BlockIndex),
                            loc = prop(prog, I::Uniform, idx, R::Location);
                if (acb != 0 || off != v.offset || size != v.array_size || stride != v.array_stride || static_cast<GLenum>(type) != static_cast<GLenum>(mgl::UniformType::UnsignedIntAtomicCounter) ||
                    block != -1 || loc != -1)
                    bad += std::format("{}: buffer index {} offset {} (want {}) size {} stride {} type {:#x} block {} location {}; ", v.name, acb, off, v.offset, size, stride, type, block, loc);
            }
            check.expect(bad.empty(),
                         "UNIFORM interface, seven atomic counter variables: ATOMIC_COUNTER_BUFFER_INDEX 0, OFFSET as declared (c_odd inherits 20 = 16 + 4 from the previous declaration, GLSL ES 3.10 section 4.4.6), ARRAY_SIZE 2 and ARRAY_STRIDE 4 for c_hist (\"always 4\", ES 3.2 section 7.7.1), TYPE UNSIGNED_INT_ATOMIC_COUNTER, BLOCK_INDEX -1, LOCATION -1",
                         bad);
            std::vector<GLint> active(7, -1);
            const R av = R::ActiveVariables;
            mgl::glGetProgramResourceiv(prog, I::AtomicCounterBuffer, 0, 1, &av, 7, nullptr, active.data());
            check.expect(std::set<GLint>(active.begin(), active.end()) == indices && indices.size() == 7,
                         "ATOMIC_COUNTER_BUFFER ACTIVE_VARIABLES lists exactly the indices of the seven counter uniforms");
            const GLuint odd = mgl::glGetProgramResourceIndex(prog, I::Uniform, "c_odd");
            char name[16] = {};
            GLint size = 0;
            mgl::UniformType type{};
            mgl::glGetActiveUniform(prog, odd, sizeof(name), nullptr, &size, &type, name);
            check.expect(std::string(name) == "c_odd" && type == mgl::UniformType::UnsignedIntAtomicCounter && mgl::glGetUniformLocation(prog, "c_odd").value == -1,
                         "glGetActiveUniform reports c_odd as UNSIGNED_INT_ATOMIC_COUNTER and glGetUniformLocation(\"c_odd\") is -1: atomic counters have no location, they are addressed by buffer binding and offset",
                         std::format("'{}' type {:#x}", name, static_cast<GLenum>(type)));
            if (use_fragment_ssbo_ && use_fragment_counters_)
            {
                const std::array<R, 4> fprops = {R::BufferBinding, R::NumActiveVariables, R::ReferencedByFragmentShader, R::ReferencedByComputeShader};
                std::array<GLint, 4> f{};
                mgl::glGetProgramResourceiv(panel_, I::AtomicCounterBuffer, 0, static_cast<GLsizei>(fprops.size()), fprops.data(), static_cast<GLsizei>(f.size()), nullptr, f.data());
                check.expect(f[0] == 0 && f[1] == 2 && f[2] == 1 && f[3] == 0,
                             "panel program (fragment stage): its ATOMIC_COUNTER_BUFFER has BUFFER_BINDING 0, 2 active variables and is referenced by the fragment shader only",
                             std::format("{} {} {} {}", f[0], f[1], f[2], f[3]));
            }
            // glGetActiveAtomicCounterBufferiv is a desktop-GL command; ES reads the same state through glGetProgramResourceiv, as above.
        }

        // ----------------------------------------------------------------------------------
        // checks: atomic memory functions
        // ----------------------------------------------------------------------------------
        void check_memory(demo::Check& check)
        {
            const MemoryRun& m = memory_run_;
            const Accumulators want = acc_expected(kHeavyN);
            for (std::size_t i = 0; i < 12; ++i)
                check.expect(m.acc[i] == want.w[i],
                             std::format("{} by 65536 invocations on one word: final value {:#x} equals the CPU fold of the same hash (order independent)", kAccName[i], want.w[i]),
                             std::format("got {:#x}, want {:#x}", m.acc[i], want.w[i]));
            Words got_hist(m.acc.begin() + 16, m.acc.begin() + 32);
            check.expect(std::equal(got_hist.begin(), got_hist.end(), want.hist.begin()),
                         "atomicAdd(hist[bin], 1) on 16 words (a skewed histogram, heavy contention on bin 0): every bin count equals the CPU histogram, and the bins add up to 65536");
            std::uint32_t total = 0;
            for (std::uint32_t c : got_hist) total += c;
            check.expect_eq(total, kHeavyN, "... the 16 bins add up to exactly N (no update lost)");
            const std::size_t bad_ticket = first_not_equal(m.out, 2 * kHeavyN, 3 * kHeavyN, 1u);
            check.expect(bad_ticket == 3 * kHeavyN, "atomicAdd(ticket, 1) returned each of 0..65535 exactly once (a histogram of the returned values is all ones)",
                         bad_ticket == 3 * kHeavyN ? "" : std::format("value {} returned {} times", bad_ticket - 2 * kHeavyN, m.out[bad_ticket]));

            // atomicExchange / atomicCompSwap: order dependent, so only set properties.
            const std::uint32_t exch_final = m.acc[12], owner = m.acc[13];
            Words exch_in(m.out.begin(), m.out.begin() + kHeavyN);
            check.expect(exchange_is_a_permutation(exch_in, exch_final, kHeavyN, 0xFFFFFFFFu),
                         "atomicExchange(word, id + 1) by 65536 invocations: the values returned to the invocations plus the final value are exactly the initial value and all ids + 1, each once (nothing duplicated, nothing lost) - the ORDER is not asserted");
            Words cas_in(m.out.begin() + kHeavyN, m.out.begin() + 2 * kHeavyN);
            check.expect(cas_has_one_winner(cas_in, owner, kHeavyN),
                         "atomicCompSwap(owner, 0, id + 1): exactly one invocation received 0 (the winner), its id + 1 is the final value, every other invocation received that value - WHICH invocation won is not asserted",
                         std::format("owner {}", owner));
            std::cout << std::format("   [note] atomicExchange: final value {} (invocation {} was last); atomicCompSwap: invocation {} won\n", exch_final, exch_final - 1u, owner - 1u);

            // Shared-variable atomics: one result per work group.
            std::string bad;
            std::uint32_t winners_bad = 0;
            for (std::uint32_t g = 0; g < kGroupsHeavy; ++g)
            {
                const GroupResult w = group_expected(g);
                const std::size_t base = 8ull * g;
                const bool ok = m.groups[base] == w.sum && m.groups[base + 1] == w.max && m.groups[base + 2] == w.or_ && m.groups[base + 3] == w.xor_ && m.groups[base + 4] == w.min &&
                                m.groups[base + 7] == w.and_;
                if (!ok && bad.size() < 200) bad += std::format("group {}; ", g);
                if (m.groups[base + 5] != 1u || m.groups[base + 6] < 1u || m.groups[base + 6] > kGroup) ++winners_bad;
            }
            check.expect(bad.empty(),
                         "atomics on SHARED variables: atomicAdd / Max / Or / Xor / Min / And by 128 invocations of each of the 512 work groups (memoryBarrierShared() + barrier() around them) give the exact per-group CPU values in all 512 groups", bad);
            check.expect(winners_bad == 0,
                         "atomicCompSwap on a shared variable: in every one of the 512 groups exactly one invocation won (winner count 1, owner in 1..128)", std::format("{} groups wrong", winners_bad));
            std::cout << std::format("   [note] 65536 NON-atomic increments of one word left {} ({} increments lost); the same increments with atomicAdd left {}\n", racy_plain_, racy_lost_, racy_atomic_);
        }

        // ----------------------------------------------------------------------------------
        // checks: stream compaction
        // ----------------------------------------------------------------------------------
        void check_compaction(demo::Check& check)
        {
            const std::array<const char*, 2> names = {"an atomic counter (atomicCounterIncrement)", "a buffer variable (atomicAdd)"};
            for (std::size_t v = 0; v < 2; ++v)
            {
                const CompactRun& c = compact_[v];
                check.expect_eq(c.count, static_cast<std::uint32_t>(expected_idx_.size()),
                                std::format("compaction with the append pointer in {}: the count equals the number of elements the predicate keeps ({} of 16384)", names[v], expected_idx_.size()));
                check.expect(compaction_is_right(c.out, c.count, expected_idx_),
                             std::format("... the compacted list is exactly the expected SET of indices (sorted it equals the CPU's ascending list), each index once, and every value[slot] belongs to idx[slot]"));
                bool tail_ok = c.count <= kCompactN;
                for (std::uint32_t s = c.count; tail_ok && s < kCompactN; ++s)
                    tail_ok = c.out[1 + s] == kSentinel && c.out[1 + kCompactN + s] == kSentinel;
                check.expect(tail_ok, "... slots beyond the count were never written (still the sentinel): no append went past the end");
                std::size_t in_order = 0;
                for (std::uint32_t s = 0; s < c.count && s < expected_idx_.size(); ++s)
                    in_order += c.out[1 + s] == expected_idx_[s] ? 1u : 0u;
                std::cout << std::format("   [note] {}: {} of {} slots hold the same element as a sequential compaction would (the order is up to the scheduler)\n", names[v], in_order, c.count);
                std::size_t mask_bad = 0;
                for (std::uint32_t i = 0; i < kCompactN; ++i)
                    if ((c.kept[i] != 0u) != compact_keep(compact_value(i))) ++mask_bad;
                check.expect(mask_bad == 0, std::format("... the mask scattered back from the list (kept[idx[slot]] = 1) equals the predicate image at all 16384 elements"), std::format("{} elements differ", mask_bad));
            }
            check.expect(compact_[0].count == compact_[1].count, "both append pointers kept the same number of elements");
        }

        // ----------------------------------------------------------------------------------
        // checks: the three displays
        // ----------------------------------------------------------------------------------
        void check_displays(const demo::Env& env, demo::Check& check)
        {
            if (!use_fragment_ssbo_)
            {
                std::cout << "   [note] fragment-shader displays were not drawn (no fragment shader storage blocks): their pixel checks are skipped\n";
                return;
            }
            const int strip_y = env.height - kStripH;
            const int hist_y = env.height - kStripH - kHistH;

            std::string bad;
            for (int c = 0; c < 16; ++c)
            {
                const Rgba8 got = demo::read_pixel(c * 16 + 8, strip_y + 8);
                const Rgba8 want = c < 12 ? kGreen : kStripGap;
                if (!near(got, want, c < 12 ? 0 : 1)) bad += std::format("cell {} {} (want {}); ", c, demo::format_rgba(got), demo::format_rgba(want));
            }
            check.expect(bad.empty(), "top strip (fragment shader reading Acc and Expect): the 12 accumulator cells are green - the words the atomics produced equal the CPU values - and the 4 unused cells keep the dark gap colour", bad);

            // Histogram: bar height is count * scale pixels.
            const auto counts = histogram_expected();
            const float scale = hist_scale();
            std::string hbad;
            for (int bin = 0; bin < 16; ++bin)
            {
                const float h = static_cast<float>(counts[static_cast<std::size_t>(bin)]) * scale;
                const int x = bin * 16 + 8;
                const float t = static_cast<float>(bin) / 15.0f;
                const Rgba8 want = {to8(t), to8(0.6f), to8(1.0f - t), 255};
                if (h >= 2.0f)
                {
                    const Rgba8 inside = demo::read_pixel(x, hist_y + static_cast<int>(h / 2.0f));
                    if (!near(inside, want, 1)) hbad += std::format("bin {} inside {} want {}; ", bin, demo::format_rgba(inside), demo::format_rgba(want));
                }
                const int above_row = static_cast<int>(h) + 2;
                if (above_row < kHistH)
                {
                    const Rgba8 above = demo::read_pixel(x, hist_y + above_row);
                    if (!near(above, kBackground, 0)) hbad += std::format("bin {} above the bar {}; ", bin, demo::format_rgba(above));
                }
            }
            check.expect(hbad.empty(),
                         std::format("histogram (fragment shader reading the atomicAdd bins): the 16 bars have the heights of the CPU histogram (tallest bar 100 px = {} counts); inside and above-the-bar pixels probed", *std::max_element(counts.begin(), counts.end())), hbad);

            // Compaction panels: every pixel.
            for (int p = 0; p < 2; ++p)
            {
                const auto px = demo::read_rgba8(p * kPanel, 0, kPanel, kPanel);
                std::size_t wrong = 0;
                for (int y = 0; y < kPanel; ++y)
                    for (int x = 0; x < kPanel; ++x)
                    {
                        const std::size_t at = 4 * (static_cast<std::size_t>(y) * kPanel + static_cast<std::size_t>(x));
                        const Rgba8 got = {px[at], px[at + 1], px[at + 2], px[at + 3]};
                        const bool lit = compact_keep(compact_value(static_cast<std::uint32_t>(y * 128 + x)));
                        if (!near(got, lit ? kLit : kDark, 1)) ++wrong;
                    }
                check.expect(wrong == 0,
                             std::format("{} panel (fragment shader reading the scattered mask): all 16384 pixels are the ring pattern (lit = kept by the predicate, dark = dropped)", p == 0 ? "left (atomic counter append)" : "right (SSBO atomicAdd append)"),
                             std::format("{} pixels differ", wrong));
                if (use_fragment_counters_)
                {
                    const Words& f = fragment_counts_[static_cast<std::size_t>(p)];
                    check.expect(f.size() == 2 && f[0] == kPanel * kPanel && f[1] == expected_idx_.size(),
                                 std::format("fragment-stage atomic counters while drawing the {} panel: c_fragments = {} (exactly one invocation per pixel without multisampling) and c_lit = {} (the number of kept elements)",
                                             p == 0 ? "left" : "right", kPanel * kPanel, expected_idx_.size()),
                                 f.size() == 2 ? std::format("{} {}", f[0], f[1]) : "");
                }
            }
            if (!use_fragment_counters_) std::cout << "   [note] fragment-stage atomic counters are unavailable: the fragment counting check is skipped\n";
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
            using P = mgl::GetParameter;
            const mgl::BufferId buf = bufs_[kAcB];
            check.expect_no_gl_error("before the binding experiments no GL error is pending");
            mgl::glBindBufferBase(kAtomic, 0, buf);
            check.expect(get_i(P::AtomicCounterBufferBinding, 0) == static_cast<GLint>(buf.value) && get_i64(P::AtomicCounterBufferStart, 0) == 0 && get_i64(P::AtomicCounterBufferSize, 0) == 0 &&
                             demo::get_integer(P::AtomicCounterBufferBinding) == static_cast<int>(buf.value),
                         "glBindBufferBase(ATOMIC_COUNTER_BUFFER, 0, buf): ATOMIC_COUNTER_BUFFER_BINDING reads back the buffer (indexed and general), START and SIZE are 0 (\"not specified\")",
                         std::format("binding {} start {} size {}", get_i(P::AtomicCounterBufferBinding, 0), get_i64(P::AtomicCounterBufferStart, 0), get_i64(P::AtomicCounterBufferSize, 0)));
            mgl::glBindBufferRange(kAtomic, 0, buf, 16, 32);
            check.expect(get_i(P::AtomicCounterBufferBinding, 0) == static_cast<GLint>(buf.value) && get_i64(P::AtomicCounterBufferStart, 0) == 16 && get_i64(P::AtomicCounterBufferSize, 0) == 32,
                         "glBindBufferRange(ATOMIC_COUNTER_BUFFER, 0, buf, 16, 32): START 16 and SIZE 32 read back (glGetInteger64i_v)",
                         std::format("start {} size {}", get_i64(P::AtomicCounterBufferStart, 0), get_i64(P::AtomicCounterBufferSize, 0)));
            check.expect_no_gl_error("the binding queries raise no error");

            // Errors; the failed calls leave the binding as it was.
            const GLuint bad_index = static_cast<GLuint>(max_ac_bindings_);
            mgl::glBindBufferRange(kAtomic, bad_index, buf, 16, 32);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferRange(ATOMIC_COUNTER_BUFFER, MAX_ATOMIC_COUNTER_BUFFER_BINDINGS) -> INVALID_VALUE");
            mgl::glBindBufferBase(kAtomic, bad_index, buf);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferBase(ATOMIC_COUNTER_BUFFER, MAX_ATOMIC_COUNTER_BUFFER_BINDINGS) -> INVALID_VALUE");
            GLint scratch = 7;
            mgl::glGetIntegeri_v(P::AtomicCounterBufferBinding, bad_index, &scratch);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetIntegeri_v(ATOMIC_COUNTER_BUFFER_BINDING, MAX_ATOMIC_COUNTER_BUFFER_BINDINGS) -> INVALID_VALUE");
            GLint64 scratch64 = 7;
            mgl::glGetInteger64i_v(P::AtomicCounterBufferSize, bad_index, &scratch64);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetInteger64i_v(ATOMIC_COUNTER_BUFFER_SIZE, MAX_ATOMIC_COUNTER_BUFFER_BINDINGS) -> INVALID_VALUE");
            mgl::glBindBufferRange(kAtomic, 0, buf, 2, 32);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferRange(ATOMIC_COUNTER_BUFFER) with offset 2, not a multiple of 4 -> INVALID_VALUE (Table 6.5)");
            mgl::glBindBufferRange(kAtomic, 0, buf, 16, 0);
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glBindBufferRange(ATOMIC_COUNTER_BUFFER) with size 0 -> INVALID_VALUE");
            check.expect(get_i64(P::AtomicCounterBufferStart, 0) == 16 && get_i64(P::AtomicCounterBufferSize, 0) == 32,
                         "the failed glBindBufferRange calls changed nothing: index 0 still reads back START 16 / SIZE 32");
            mgl::glBindBufferBase(kAtomic, 0, mgl::BufferId{0});
            check.expect(get_i(P::AtomicCounterBufferBinding, 0) == 0, "glBindBufferBase(index 0, buffer 0) unbinds the point");
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
            check.expect(max_ac_bindings_ >= 1 && max_ac_size_ >= 32 && max_cs_ac_buffers_ >= 1 && max_cs_ac_counters_ >= 8 && max_combined_ac_buffers_ >= 1 && max_combined_ac_counters_ >= 8,
                         "MAX_ATOMIC_COUNTER_BUFFER_BINDINGS >= 1, MAX_ATOMIC_COUNTER_BUFFER_SIZE >= 32, MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS >= 1, MAX_COMPUTE_ATOMIC_COUNTERS >= 8, MAX_COMBINED_ATOMIC_COUNTER_BUFFERS >= 1, MAX_COMBINED_ATOMIC_COUNTERS >= 8 (ES 3.2 chapter 21 minimums)",
                         std::format("{} {} {} {} {} {}", max_ac_bindings_, max_ac_size_, max_cs_ac_buffers_, max_cs_ac_counters_, max_combined_ac_buffers_, max_combined_ac_counters_));
            check.expect(max_cs_blocks_ >= 4 && max_group_invocations_ >= 128 && max_group_size_x_ >= 128 && max_group_count_x_ >= 65535,
                         "MAX_COMPUTE_SHADER_STORAGE_BLOCKS >= 4, MAX_COMPUTE_WORK_GROUP_INVOCATIONS >= 128, MAX_COMPUTE_WORK_GROUP_SIZE[0] >= 128, MAX_COMPUTE_WORK_GROUP_COUNT[0] >= 65535 (what the shaders of this demo rely on)",
                         std::format("{} {} {} {}", max_cs_blocks_, max_group_invocations_, max_group_size_x_, max_group_count_x_));

            // GLSL ES 3.10 section 4.4.6: "It is a compile-time error to bind an atomic counter with a binding value greater than or equal to
            // gl_MaxAtomicCounterBindings" (the shader-side name of MAX_ATOMIC_COUNTER_BUFFER_BINDINGS).
            auto counter_shader = [](std::string_view declarations, std::string_view uses) {
                return std::format("#version 310 es\nlayout(local_size_x = 1) in;\n{}void main() {{\n{}}}\n", declarations, uses);
            };
            std::string log;
            const bool last_binding = compute_links(counter_shader(std::format("layout(binding = {}, offset = 0) uniform atomic_uint c;\n", max_ac_bindings_ - 1), "atomicCounterIncrement(c);\n"), log);
            check.expect(last_binding, std::format("a counter at binding {} (MAX_ATOMIC_COUNTER_BUFFER_BINDINGS - 1) compiles and links", max_ac_bindings_ - 1), log.substr(0, 200));
            const bool past_last = compute_links(counter_shader(std::format("layout(binding = {}, offset = 0) uniform atomic_uint c;\n", max_ac_bindings_), "atomicCounterIncrement(c);\n"), log);
            check.expect(!past_last, std::format("a counter at binding {} (= MAX_ATOMIC_COUNTER_BUFFER_BINDINGS) is rejected (compile-time error, GLSL ES 3.10 section 4.4.6)", max_ac_bindings_),
                         past_last ? "it compiled and linked" : log.substr(0, 200));

            // "If the number of atomic counter buffer bindings referenced by any shader in the program exceeds the corresponding limit, the program will
            // fail to link" (ES 3.2 section 7.7): exactly MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS bindings link, one more does not (if the binding point exists).
            const int usable = std::min(max_cs_ac_buffers_, max_ac_bindings_);
            if (usable <= 64)
            {
                auto source = [&](int buffers) {
                    std::string decl, use;
                    for (int i = 0; i < buffers; ++i)
                    {
                        decl += std::format("layout(binding = {}, offset = 0) uniform atomic_uint c{};\n", i, i);
                        use += std::format("  atomicCounterIncrement(c{});\n", i);
                    }
                    return counter_shader(decl, use);
                };
                std::string at_log, over_log;
                const bool at = compute_links(source(usable), at_log);
                check.expect(at, std::format("a compute shader using atomic counters in {} different bindings (min(MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS = {}, MAX_ATOMIC_COUNTER_BUFFER_BINDINGS = {})) links", usable, max_cs_ac_buffers_, max_ac_bindings_),
                             at_log.substr(0, 200));
                if (max_cs_ac_buffers_ + 1 <= max_ac_bindings_)
                {
                    const bool over = compute_links(source(max_cs_ac_buffers_ + 1), over_log);
                    check.expect(!over, std::format("a compute shader using {} atomic counter bindings (MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS + 1) fails to link", max_cs_ac_buffers_ + 1), over ? "it linked" : over_log.substr(0, 200));
                }
                else
                {
                    std::cout << std::format("   [note] MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS = {} equals MAX_ATOMIC_COUNTER_BUFFER_BINDINGS = {}: no binding point exists beyond the per-stage limit, so \"one more binding fails to link\" cannot be provoked\n",
                                             max_cs_ac_buffers_, max_ac_bindings_);
                }
            }

            // Layout rules of GLSL ES 3.10 section 4.4.6 that make a shader invalid.
            const bool shared_offset = compute_links(counter_shader("layout(binding = 0, offset = 0) uniform atomic_uint a;\nlayout(binding = 0, offset = 0) uniform atomic_uint b;\n",
                                                                    "atomicCounterIncrement(a);\natomicCounterIncrement(b);\n"), log);
            check.expect(!shared_offset, "two atomic counters at the same binding and the same offset are rejected (\"offsets must be unique and non overlapping\", GLSL ES 3.10 section 4.4.6)", shared_offset ? "it compiled and linked" : log.substr(0, 160));
            const bool no_binding = compute_links(counter_shader("layout(offset = 4) uniform atomic_uint a;\n", "atomicCounterIncrement(a);\n"), log);
            check.expect(!no_binding, "an atomic counter declared with offset but without binding is rejected (\"error, must include binding\", GLSL ES 3.10 section 4.4.6)", no_binding ? "it compiled and linked" : log.substr(0, 160));
            // "Any uniform sampler, image or atomic counter variable declared without a binding qualifier is initially bound to unit zero" (section 4.4.5)
            // and (section 4.4.6) the offset of a counter without one is the next free offset of its binding: both counters below live in binding 0.
            const bool implicit = compute_links(counter_shader("uniform atomic_uint a;\nuniform atomic_uint b;\n", "atomicCounterIncrement(a);\natomicCounterIncrement(b);\n"), log);
            check.expect_or_known_deviation(implicit, "atomic counters declared without any layout qualifier compile and link: they are bound to binding 0 at consecutive offsets (GLSL ES 3.10 sections 4.4.5 and 4.4.6)",
                                            log.substr(0, 200), "Mesa",
                                            "Mesa rejects an atomic_uint without a binding qualifier (\"atomic counters require explicit binding point\", a desktop ARB_shader_atomic_counters rule); GLSL ES 3.10 section 4.4.5 binds it to unit zero");
            const bool in_struct = compute_links(counter_shader("struct S { atomic_uint c; };\nuniform S s;\n", "atomicCounterIncrement(s.c);\n"), log);
            check.expect(!in_struct, "an atomic_uint member inside a structure is rejected (\"since there is no mechanism to specify the binding\", GLSL ES 3.10 section 4.1.8)", in_struct ? "it compiled and linked" : log.substr(0, 160));
            check.expect_no_gl_error("the limit experiments leave no GL error");
        }

        // ----------------------------------------------------------------------------------
        // negative controls: the set checks must be able to fail
        // ----------------------------------------------------------------------------------
        void negative_controls(demo::Check& check)
        {
            // Doctor the captured results the way a lost or duplicated update would, and require the same predicates to reject them.
            Words seen = heavy_.seen;
            seen[17] = 2u;  // ticket 17 handed out twice ...
            seen[18] = 0u;  // ... and ticket 18 never
            check.expect(first_not_equal(seen, 0, heavy_.n, 1u) == 17, "negative control: a ticket histogram with one duplicated and one missing ticket is rejected by the same check (first bad entry 17)");

            Words exch(memory_run_.out.begin(), memory_run_.out.begin() + kHeavyN);
            exch[3] = exch[4];  // one received value duplicated, another lost
            check.expect(!exchange_is_a_permutation(exch, memory_run_.acc[12], kHeavyN, 0xFFFFFFFFu), "negative control: an exchange log with a duplicated value is rejected by the permutation check");

            Words cas(memory_run_.out.begin() + kHeavyN, memory_run_.out.begin() + 2 * kHeavyN);
            const std::uint32_t owner = memory_run_.acc[13];
            Words two_winners = cas;
            two_winners[(owner - 1u + 1u) % kHeavyN] = 0u;  // a second invocation that "also won"
            check.expect(!cas_has_one_winner(two_winners, owner, kHeavyN), "negative control: a compare-swap log with two winners is rejected");

            Words list = compact_[0].out;
            list[1 + 5] = list[1 + 6];  // duplicate one element in the compacted list
            check.expect(compaction_is_right(compact_[0].out, compact_[0].count, expected_idx_) && !compaction_is_right(list, compact_[0].count, expected_idx_),
                         "negative control: the genuine compacted list passes the set check and the same list with one slot overwritten by its neighbour fails it");

            const Accumulators want = acc_expected(kHeavyN);
            Words doctored = memory_run_.acc;
            doctored[0] += 1u;
            check.expect(memory_run_.acc[0] == want.w[0] && doctored[0] != want.w[0], "negative control: an add result off by one (a single lost increment) differs from the exact CPU fold");
        }

        // ----------------------------------------------------------------------------------
        // state
        // ----------------------------------------------------------------------------------
        mgl::ProgramId counters_{}, memory_{}, compact_counter_{}, compact_ssbo_{}, scatter_{}, racy_{}, strip_{}, hist_{}, panel_{};
        std::array<mgl::UniformLocation, 2> u_n_{};
        mgl::UniformLocation u_phase_{}, u_ticket0_{}, u_dec0_{}, u_wrap0_{}, u_count_{}, u_hist_scale_{}, u_hist_y0_{}, u_panel_x0_{};
        std::array<mgl::BufferId, kBufferCount> bufs_{};
        CounterRun heavy_, light_;
        MemoryRun memory_run_;
        std::array<CompactRun, 2> compact_;
        std::array<Words, 2> fragment_counts_;
        Words expected_idx_;
        std::uint32_t racy_lost_ = 0, racy_plain_ = 0, racy_atomic_ = 0;
        int max_ac_bindings_ = 1, max_ac_size_ = 32, max_cs_ac_buffers_ = 1, max_cs_ac_counters_ = 8, max_combined_ac_buffers_ = 1, max_combined_ac_counters_ = 8, max_fs_ac_buffers_ = 0,
            max_fs_ac_counters_ = 0, max_vs_ac_buffers_ = 0, max_cs_blocks_ = 4, max_fs_blocks_ = 0, max_group_invocations_ = 128, max_group_size_x_ = 128, max_group_count_x_ = 65535;
        bool use_fragment_ssbo_ = false, use_fragment_counters_ = false;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es31-atomics",
        .title = "Atomic counters and atomic memory functions",
        .min_es = {3, 1},
        .summary = "Atomic counters (binding/offset, increment/decrement/read, wrap-around, glBufferSubData initialisation, buffer-range binding, program-interface queries) "
                   "and atomicAdd/Min/Max/And/Or/Xor/Exchange/CompSwap on buffer and shared variables under heavy contention with exactly known results, "
                   "plus two stream compactions validated as sets.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Atomics>(argc, argv, info);
}
