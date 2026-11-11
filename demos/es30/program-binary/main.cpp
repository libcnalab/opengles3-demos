// es30-program-binary
//
// Program binaries (ES 3.0 §2.12.4, ES 3.2 §7.5): a linked program is saved with glGetProgramBinary
// and loaded into a *fresh* program object with glProgramBinary, which then renders the right half of
// the window while the original program renders the left half. Both halves are compared with a
// CPU reference of the integer Truchet pattern in shaders/pattern.frag and bit-exactly with each other.
//
// The spec allows NUM_PROGRAM_BINARY_FORMATS to be 0. The demo reads the value and follows the spec
// strictly in both cases:
//   >= 1: round trip, uniform reset to defaults, restored attribute assignment, active-variable
//         comparison, corrupted / truncated / over-long binaries (LINK_STATUS FALSE, no error),
//         invalid format (INVALID_ENUM), negative length (INVALID_VALUE), fallback to source.
//   == 0: no round trip is possible; every format value is invalid (INVALID_ENUM) and the right
//         half shows the original program in grey instead.
// In both cases glGetProgramBinary with a too-small bufSize and with an unlinked program must fail
// with INVALID_OPERATION.
//
// The corrupted / truncated / over-long binaries are loaded in child processes (this executable
// re-executed with a hidden first argument), because one of those loads crashes a driver (ANGLE).

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace mgl = metagl;

namespace
{
    // Attribute index for a_pos, set with glBindAttribLocation before linking the original program.
    constexpr GLuint kPosAttrib = 5;
    // A *different* binding set on the fresh program before glProgramBinary: the load must restore 5.
    constexpr GLuint kDecoyAttrib = 2;
    constexpr std::uint32_t kSeed = 0x2545F491u;
    constexpr std::array<std::uint8_t, 3> kClear = {10, 12, 18};
    constexpr const char* kMesaClearsFirst =
        "Mesa's glProgramBinary discards the program's link data before checking binaryFormat and length, so an erroring "
        "call unlinks the program (ES 3.0 §2.5: the erroring command must have no effect)";

    // Full-viewport triangle.
    constexpr std::array<float, 6> kTriangle = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

    // ---- CPU reference of shaders/pattern.frag (bit-exact: integer arithmetic only) -----------------
    constexpr std::array<std::array<std::uint32_t, 3>, 6> kPalette = {{
        {24, 38, 64}, {33, 52, 84}, {42, 157, 143}, {233, 196, 106}, {244, 162, 97}, {231, 111, 81},
    }};

    std::uint32_t cell_hash(std::uint32_t cx, std::uint32_t cy, std::uint32_t seed)
    {
        // GLSL ES 3.00 §4.1.3 / §5.9: uint multiplication wraps to the low 32 bits, like uint32_t here.
        std::uint32_t h = (cx * 0x8DA6B343u) ^ (cy * 0xD8163841u) ^ seed;
        h ^= h >> 13u;
        h *= 0x5BD1E995u;
        h ^= h >> 15u;
        return h;
    }

    demo::Rgba8 pattern(int px, int py, int cell, std::uint32_t seed, bool gray)
    {
        const int cx = px / cell, cy = py / cell;
        int lx = px - cx * cell;
        const int ly = py - cy * cell;
        const std::uint32_t h = cell_hash(static_cast<std::uint32_t>(cx), static_cast<std::uint32_t>(cy), seed);
        if ((h & 1u) != 0u) lx = cell - 1 - lx;
        const int x2 = 2 * lx + 1, y2 = 2 * ly + 1, s2 = 2 * cell;
        const int d0 = x2 * x2 + y2 * y2;
        const int d1 = (s2 - x2) * (s2 - x2) + (s2 - y2) * (s2 - y2);
        const int d = std::min(d0, d1);
        const int w = std::max(1, cell / 4);
        const bool on_arc = d >= (cell - w) * (cell - w) && d < (cell + w) * (cell + w);
        auto col = on_arc ? kPalette[static_cast<std::size_t>(2 + ((cx + cy) / 3) % 4)] : kPalette[(h >> 8u) & 1u];
        if (gray)
        {
            const std::uint32_t g = (col[0] * 77u + col[1] * 150u + col[2] * 29u) >> 8u;
            col = {g, g, g};
        }
        // vec3(col) / 255.0 written to an RGBA8 buffer: ES 3.0 §2.1.6.2 converts f to "one of the two" 8-bit
        // integers closest to f * 255 ("rounding to nearest is preferred"), i.e. k or a neighbour: tolerance 1.
        return {static_cast<std::uint8_t>(col[0]), static_cast<std::uint8_t>(col[1]), static_cast<std::uint8_t>(col[2]), 255};
    }

    struct Layout
    {
        int gap, left_w, right_x, right_w, height, cell;
    };

    Layout layout(const demo::Env& env)
    {
        Layout l{};
        l.gap = std::max(2, env.width / 64);
        l.left_w = (env.width - l.gap) / 2;
        l.right_x = l.left_w + l.gap;
        l.right_w = env.width - l.right_x;
        l.height = env.height;
        l.cell = std::max(6, std::min(l.left_w, env.height) / 7);
        return l;
    }

    struct ActiveVar
    {
        GLint size;
        mgl::UniformType type;
        friend bool operator==(const ActiveVar&, const ActiveVar&) = default;
    };

    std::map<std::string, ActiveVar> active_uniforms(mgl::ProgramId p)
    {
        GLint count = 0, max_len = 0;
        mgl::glGetProgramiv(p, mgl::ProgramParameter::ActiveUniforms, &count);
        mgl::glGetProgramiv(p, mgl::ProgramParameter::ActiveUniformMaxLength, &max_len);
        std::map<std::string, ActiveVar> out;
        for (GLint i = 0; i < count; ++i)
        {
            std::string name(static_cast<std::size_t>(std::max(max_len, 1)), '\0');
            GLsizei len = 0;
            ActiveVar v{};
            mgl::glGetActiveUniform(p, static_cast<GLuint>(i), max_len, &len, &v.size, &v.type, name.data());
            name.resize(static_cast<std::size_t>(len));
            out[name] = v;
        }
        return out;
    }

    std::map<std::string, ActiveVar> active_attribs(mgl::ProgramId p)
    {
        GLint count = 0, max_len = 0;
        mgl::glGetProgramiv(p, mgl::ProgramParameter::ActiveAttributes, &count);
        mgl::glGetProgramiv(p, mgl::ProgramParameter::ActiveAttributeMaxLength, &max_len);
        std::map<std::string, ActiveVar> out;
        for (GLint i = 0; i < count; ++i)
        {
            std::string name(static_cast<std::size_t>(std::max(max_len, 1)), '\0');
            GLsizei len = 0;
            ActiveVar v{};
            mgl::glGetActiveAttrib(p, mgl::ActiveAttribIndex{static_cast<GLuint>(i)}, max_len, &len, &v.size, &v.type, name.data());
            name.resize(static_cast<std::size_t>(len));
            out[name] = v;
        }
        return out;
    }

    std::string describe(const std::map<std::string, ActiveVar>& vars)
    {
        std::string s;
        for (const auto& [name, v] : vars) s += std::format("{}({},{}) ", name, v.size, metagl::to_string(v.type));
        return s;
    }

    GLint program_int(mgl::ProgramId p, mgl::ProgramParameter pname)
    {
        GLint v = -12345;
        mgl::glGetProgramiv(p, pname, &v);
        return v;
    }

    // Values of the five uniforms right after a successful glProgramBinary, read before any glUniform*.
    struct UniformSnapshot
    {
        std::array<GLfloat, 2> scale{-1.0f, -1.0f};
        std::array<GLint, 2> origin{-1, -1};
        GLint cell = -1;
        GLuint seed = 0xFFFFFFFFu;
        GLint gray = -1;
        bool all_found = false;
    };

    UniformSnapshot snapshot_uniforms(mgl::ProgramId p)
    {
        UniformSnapshot s;
        const mgl::UniformLocation l_scale = mgl::glGetUniformLocation(p, "u_scale");
        const mgl::UniformLocation l_origin = mgl::glGetUniformLocation(p, "u_origin");
        const mgl::UniformLocation l_cell = mgl::glGetUniformLocation(p, "u_cell");
        const mgl::UniformLocation l_seed = mgl::glGetUniformLocation(p, "u_seed");
        const mgl::UniformLocation l_gray = mgl::glGetUniformLocation(p, "u_gray");
        s.all_found = l_scale.value >= 0 && l_origin.value >= 0 && l_cell.value >= 0 && l_seed.value >= 0 && l_gray.value >= 0;
        if (!s.all_found) return s;
        mgl::glGetUniformfv(p, l_scale, s.scale.data());
        mgl::glGetUniformiv(p, l_origin, s.origin.data());
        mgl::glGetUniformiv(p, l_cell, &s.cell);
        mgl::glGetUniformuiv(p, l_seed, &s.seed);
        mgl::glGetUniformiv(p, l_gray, &s.gray);
        return s;
    }

    bool all_zero(const UniformSnapshot& s)
    {
        return s.all_found && s.scale[0] == 0.0f && s.scale[1] == 0.0f && s.origin[0] == 0 && s.origin[1] == 0 &&
               s.cell == 0 && s.seed == 0u && s.gray == 0;
    }

    std::string describe(const UniformSnapshot& s)
    {
        if (!s.all_found) return "a uniform location is missing";
        return std::format("u_scale=({},{}) u_origin=({},{}) u_cell={} u_seed={} u_gray={}", s.scale[0], s.scale[1],
                           s.origin[0], s.origin[1], s.cell, s.seed, s.gray);
    }

    // ---------------------------------------------------------------------------------------------
    // Crash isolation (the pattern of tests/negative/driver-crash-readpixels-none). A call that is valid, or
    // that the specification says must fail cleanly, but that crashes or hangs a driver cannot be observed
    // in-process. It runs in a fresh child process instead: this executable re-executed through
    // /proc/self/exe with a hidden first argument. The child creates its own EGL display and context
    // through demo::run (a fork()ed copy of the parent's context is not usable), inherits the environment
    // (driver selection, MESA_GLES_VERSION_OVERRIDE), and prints a marker immediately before and after the
    // isolated call. The parent classifies how the child ended: exit status, signal or timeout, and whether
    // it died between the two markers.
    // ---------------------------------------------------------------------------------------------
    constexpr std::string_view kChildFlag = "--crash-isolated-child";
    constexpr std::string_view kBeforeMarker = "@@ before the isolated call";
    constexpr std::string_view kAfterMarker = "@@ after the isolated call";

    struct ChildResult
    {
        bool spawned = false;
        bool timed_out = false;
        bool exited = false;
        int exit_code = -1;
        bool signaled = false;
        int signal = 0;
        std::string output;

        [[nodiscard]] bool reached_call() const { return output.find(kBeforeMarker) != std::string::npos; }
        [[nodiscard]] bool returned_from_call() const { return output.find(kAfterMarker) != std::string::npos; }
        /// The child passed its own checks.
        [[nodiscard]] bool passed() const { return exited && exit_code == 0 && returned_from_call(); }
        /// The child died (signal or timeout) between the two markers, i.e. inside the isolated call.
        [[nodiscard]] bool died_in_call() const { return (signaled || timed_out) && reached_call() && !returned_from_call(); }
        [[nodiscard]] std::string describe() const
        {
            if (!spawned) return "child could not be spawned";
            const char* where = reached_call() && !returned_from_call() ? " inside the isolated call" : "";
            if (timed_out) return std::format("child timed out{} and was killed", where);
            if (signaled) return std::format("child killed by signal {} ({}){}", signal, sigabbrev_np(signal) ? sigabbrev_np(signal) : "?", where);
            return std::format("child exited with status {}", exit_code);
        }
    };

    /// Environment of the child: ours, plus ASan options that leave SIGSEGV/SIGBUS/SIGABRT to the default action, so that a
    /// crash is reported as a signal also in OGLES3_SANITIZE builds. Harmless without ASan.
    std::vector<std::string> child_environment()
    {
        std::vector<std::string> env;
        std::string asan = "ASAN_OPTIONS=";
        for (char** e = environ; *e != nullptr; ++e)
        {
            const std::string_view kv = *e;
            if (kv.starts_with("ASAN_OPTIONS=")) asan = std::string(kv) + ":";
            else env.emplace_back(kv);
        }
        env.push_back(asan + "handle_segv=0:handle_sigbus=0:handle_abort=0:handle_sigill=0:handle_sigfpe=0");
        return env;
    }

    ChildResult run_child(std::string_view case_name, std::chrono::seconds timeout)
    {
        ChildResult r;
        const std::string name(case_name);
        std::vector<std::string> args = {"/proc/self/exe", std::string(kChildFlag), name};
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        std::vector<std::string> env_strings = child_environment();
        std::vector<char*> envp;
        for (auto& e : env_strings) envp.push_back(e.data());
        envp.push_back(nullptr);

        int fds[2] = {-1, -1};
        if (pipe2(fds, O_CLOEXEC) != 0) return r;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);

        // No core dump for the expected crash (inherited soft limit; restored right after the spawn).
        rlimit core{};
        const bool have_core = getrlimit(RLIMIT_CORE, &core) == 0;
        if (have_core)
        {
            rlimit none = core;
            none.rlim_cur = 0;
            setrlimit(RLIMIT_CORE, &none);
        }
        std::cout << std::flush;
        std::fflush(nullptr);
        pid_t pid = -1;
        const int rc = posix_spawn(&pid, "/proc/self/exe", &actions, nullptr, argv.data(), envp.data());
        if (have_core) setrlimit(RLIMIT_CORE, &core);
        posix_spawn_file_actions_destroy(&actions);
        close(fds[1]);
        if (rc != 0)
        {
            close(fds[0]);
            return r;
        }
        r.spawned = true;

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::array<char, 4096> buf{};
        for (;;)
        {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            if (left.count() <= 0)
            {
                r.timed_out = true;
                kill(pid, SIGKILL);
                break;
            }
            pollfd p{fds[0], POLLIN, 0};
            const int n = poll(&p, 1, static_cast<int>(left.count()));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) continue;
            const ssize_t got = read(fds[0], buf.data(), buf.size());
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;  // EOF: the child (and everything that inherited the pipe) is gone
            r.output.append(buf.data(), static_cast<std::size_t>(got));
        }
        close(fds[0]);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        if (!r.timed_out)
        {
            r.exited = WIFEXITED(status);
            r.exit_code = r.exited ? WEXITSTATUS(status) : -1;
            r.signaled = WIFSIGNALED(status);
            r.signal = r.signaled ? WTERMSIG(status) : 0;
        }

        std::cout << std::format("   --- child '{}' output ---\n", name);
        std::size_t start = 0;
        while (start < r.output.size())
        {
            const std::size_t end = r.output.find('\n', start);
            const std::size_t stop = end == std::string::npos ? r.output.size() : end;
            const std::string_view line = std::string_view(r.output).substr(start, stop - start);
            // "child RESULT:" so that a log scan for the parent's own "RESULT:" line never matches a child.
            std::cout << "   | " << (line.starts_with("RESULT:") ? "child " : "") << line << "\n";
            start = stop + 1;
        }
        std::cout << std::format("   --- child '{}': {} ---\n", name, r.describe());
        return r;
    }

    /// Prints a marker and flushes it to the pipe before the isolated call can crash.
    void marker(std::string_view m) { std::cout << m << std::endl; }


    // ---------------------------------------------------------------------------------------------
    // Child process: one corrupted, truncated or over-long binary loaded over a successfully loaded program.
    // ES 3.0 2.12.4: "binaryFormat and binary must be those returned by a previous call to GetProgramBinary, and length
    // must be the length of the program binary as returned by GetProgramBinary or GetProgramiv with pname
    // PROGRAM_BINARY_LENGTH. Loading the program binary will fail, setting the LINK_STATUS of program to FALSE, if these
    // conditions are not met." and "If ProgramBinary fails to load a binary, no error is generated". A load that must
    // fail cleanly crashes ANGLE (SIGILL inside glProgramBinary for a binary truncated to half its length), so every
    // mutation is loaded in a fresh process, on every driver.
    // ---------------------------------------------------------------------------------------------
    struct MutationCase
    {
        std::string_view name;
        const char* what;
    };
    constexpr std::array<MutationCase, 7> kMutations = {{
        {"middle-byte", "one byte in the middle inverted"},
        {"first-byte", "first byte inverted"},
        {"last-byte", "last byte inverted"},
        {"truncated-1", "truncated by one byte (length - 1)"},
        {"truncated-half", "truncated to half its length"},
        {"extra-16", "with 16 extra bytes (length + 16)"},
        {"bit-flip", "a single flipped bit (one third into the binary)"},
    }};
    std::string g_child_case;  // set by main() in child mode
    constexpr std::string_view kResultMarker = "@@ result ";

    /// The mutated binary and the length passed to glProgramBinary.
    std::pair<std::vector<std::uint8_t>, GLsizei> mutate(std::string_view name, const std::vector<std::uint8_t>& binary, GLsizei n)
    {
        auto d = binary;
        if (name == "middle-byte") d[d.size() / 2] ^= 0xFFu;
        else if (name == "first-byte") d[0] ^= 0xFFu;
        else if (name == "last-byte") d[d.size() - 1] ^= 0xFFu;
        else if (name == "truncated-1") return {d, n - 1};
        else if (name == "truncated-half") return {d, n / 2};
        else if (name == "extra-16")
        {
            d.resize(d.size() + 16, 0);
            return {d, n + 16};
        }
        else if (name == "bit-flip") d[d.size() / 3] ^= 0x01u;
        return {d, n};
    }

    class MutationChild final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            std::string log;
            const mgl::ShaderId vs = demo::compile_shader(mgl::ShaderType::Vertex, env.load_text("shaders/pattern.vert"), log);
            const mgl::ShaderId fs = demo::compile_shader(mgl::ShaderType::Fragment, env.load_text("shaders/pattern.frag"), log);
            if (vs.value == 0 || fs.value == 0) throw demo::Failure("pattern shaders failed to compile:\n" + log);
            const mgl::ProgramId original = mgl::glCreateProgram();
            mgl::glAttachShader(original, vs);
            mgl::glAttachShader(original, fs);
            mgl::glProgramParameteri(original, mgl::ProgramParameter::BinaryRetrievableHint, GL_TRUE);
            mgl::glLinkProgram(original);
            mgl::glDeleteShader(vs);
            mgl::glDeleteShader(fs);
            if (program_int(original, mgl::ProgramParameter::LinkStatus) != GL_TRUE)
                throw demo::Failure("pattern program failed to link:\n" + demo::program_info_log(original));
            if (demo::get_integer(mgl::GetParameter::NumProgramBinaryFormats) < 1) throw demo::Skip("no program binary formats");
            const GLint length = program_int(original, mgl::ProgramParameter::BinaryLength);
            binary_.assign(static_cast<std::size_t>(std::max(length, 1)), 0);
            mgl::glGetProgramBinary(original, length, &written_, &format_, binary_.data());
            mgl::glDeleteProgram(original);
            if (written_ <= 2) throw demo::Failure(std::format("glGetProgramBinary wrote {} bytes", written_));
            binary_.resize(static_cast<std::size_t>(written_));
        }

        void render(demo::Env& /*env*/, int /*frame*/) override {}

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            const mgl::ProgramId p = mgl::glCreateProgram();
            // Load the valid binary first, so the failure must also discard a previous successful load:
            // "any information about a previous link or load of that program object is lost".
            mgl::glProgramBinary(p, format_, binary_.data(), written_);
            const mgl::ErrorCode ok_error = demo::drain_gl_errors();
            const GLint ok_linked = program_int(p, mgl::ProgramParameter::LinkStatus);
            const auto [data, length] = mutate(g_child_case, binary_, written_);
            marker(kBeforeMarker);
            mgl::glProgramBinary(p, format_, data.data(), length);
            const mgl::ErrorCode error = demo::drain_gl_errors();
            const GLint linked = program_int(p, mgl::ProgramParameter::LinkStatus);
            const GLint binary_length = program_int(p, mgl::ProgramParameter::BinaryLength);
            marker(kAfterMarker);
            // Parsed by the parent.
            std::cout << kResultMarker << std::format("valid_linked={} error={} linked={} length={}", ok_linked, demo::error_name(error), linked, binary_length)
                      << std::endl;
            check.expect(ok_error == mgl::ErrorCode::NoError && ok_linked == GL_TRUE, "the valid binary loads first (LINK_STATUS TRUE)");
            check.expect(error == mgl::ErrorCode::NoError && linked == GL_FALSE && binary_length == 0,
                         std::format("the {} binary: glProgramBinary generates no error, LINK_STATUS FALSE, PROGRAM_BINARY_LENGTH 0", g_child_case),
                         std::format("error {}, LINK_STATUS {}, length {}", demo::error_name(error), linked, binary_length));
            mgl::glDeleteProgram(p);
        }

        void shutdown(demo::Env& /*env*/) override {}

    private:
        std::vector<std::uint8_t> binary_;
        GLsizei written_ = -1;
        mgl::ProgramBinaryFormat format_{};
    };

    bool renderer_is_angle()
    {
        const GLubyte* renderer = mgl::glGetString(mgl::StringName::Renderer);
        return renderer != nullptr && std::string_view(reinterpret_cast<const char*>(renderer)).find("ANGLE") != std::string_view::npos;
    }

    /// Values of the child's result line; `found` is false when the child never printed it.
    struct ChildLoad
    {
        bool found = false;
        int valid_linked = -1, linked = -1, length = -1;
        std::string error;
    };

    ChildLoad parse_child_load(const ChildResult& r)
    {
        ChildLoad l;
        const std::size_t at = r.output.find(kResultMarker);
        if (at == std::string::npos) return l;
        const std::string line = r.output.substr(at + kResultMarker.size(), r.output.find('\n', at) - at - kResultMarker.size());
        char error[64] = {};
        l.found = std::sscanf(line.c_str(), "valid_linked=%d error=%63s linked=%d length=%d", &l.valid_linked, error, &l.linked, &l.length) == 4;
        l.error = error;
        return l;
    }

    class ProgramBinary final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < 64 || env.height < 32) throw demo::Skip("surface smaller than 64x32");
            context_ = env.context_version;

            // ---- 1. The original program, linked from source with the retrieval hint ----------------
            vs_ = compile(mgl::ShaderType::Vertex, env.load_text("shaders/pattern.vert"), "pattern.vert");
            fs_ = compile(mgl::ShaderType::Fragment, env.load_text("shaders/pattern.frag"), "pattern.frag");

            original_ = mgl::glCreateProgram();
            mgl::glAttachShader(original_, vs_);
            mgl::glAttachShader(original_, fs_);
            mgl::glBindAttribLocation(original_, mgl::AttribLocation{kPosAttrib}, "a_pos");
            hint_initial_ = program_int(original_, mgl::ProgramParameter::BinaryRetrievableHint);
            // "should be called with pname set to PROGRAM_BINARY_RETRIEVABLE_HINT and value set to TRUE.
            //  This setting will not be in effect until the next time LinkProgram ... has been called."
            mgl::glProgramParameteri(original_, mgl::ProgramParameter::BinaryRetrievableHint, GL_TRUE);
            hint_before_link_ = program_int(original_, mgl::ProgramParameter::BinaryRetrievableHint);
            mgl::glLinkProgram(original_);
            if (program_int(original_, mgl::ProgramParameter::LinkStatus) != GL_TRUE)
                throw demo::Failure("pattern program failed to link:\n" + demo::program_info_log(original_));
            hint_after_link_ = program_int(original_, mgl::ProgramParameter::BinaryRetrievableHint);
            mgl::glDetachShader(original_, vs_);
            mgl::glDetachShader(original_, fs_);

            // ---- 2. Supported binary formats -------------------------------------------------------
            num_formats_ = demo::get_integer(mgl::GetParameter::NumProgramBinaryFormats);
            if (num_formats_ > 0)
            {
                formats_.assign(static_cast<std::size_t>(num_formats_), 0);
                mgl::glGetIntegerv(mgl::GetParameter::ProgramBinaryFormats, formats_.data());
            }
            std::printf("  NUM_PROGRAM_BINARY_FORMATS = %d", num_formats_);
            for (GLint f : formats_) std::printf(" [0x%X]", static_cast<unsigned>(f));
            std::printf("\n");

            // ---- 3. Save the binary and load it into a fresh program object --------------------------
            if (num_formats_ > 0)
            {
                binary_length_ = program_int(original_, mgl::ProgramParameter::BinaryLength);
                binary_.assign(static_cast<std::size_t>(std::max(binary_length_, 0)), 0);
                GLsizei written = -1;
                mgl::glGetProgramBinary(original_, binary_length_, &written, &binary_format_, binary_.data());
                get_binary_error_ = demo::drain_gl_errors();
                written_ = written;
                std::printf("  PROGRAM_BINARY_LENGTH = %d, written = %d, format = 0x%X\n", binary_length_, written,
                            static_cast<unsigned>(binary_format_));

                reloaded_ = mgl::glCreateProgram();
                // A decoy binding: the binary must restore the assignment in effect at the original link (5).
                mgl::glBindAttribLocation(reloaded_, mgl::AttribLocation{kDecoyAttrib}, "a_pos");
                mgl::glProgramBinary(reloaded_, binary_format_, binary_.data(), written_);
                reload_error_ = demo::drain_gl_errors();
                reload_linked_ = program_int(reloaded_, mgl::ProgramParameter::LinkStatus);
                if (reload_linked_ == GL_TRUE)
                    reload_defaults_ = snapshot_uniforms(reloaded_);  // before any glUniform* call
                else
                    std::printf("  glProgramBinary failed:\n%s\n", demo::program_info_log(reloaded_).c_str());
            }
            else
            {
                std::printf("  The implementation lists no program binary formats (allowed by ES 3.0 §2.12.4), so no "
                            "binary can be retrieved or loaded; the right half shows the original program in grey.\n");
            }

            // ---- 4. Geometry: one full-viewport triangle fed through attribute 5 ----------------------
            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const float>(kTriangle), mgl::BufferUsage::StaticDraw);
            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glEnableVertexAttribArray(mgl::AttribLocation{kPosAttrib});
            mgl::glVertexAttribPointer(mgl::AttribLocation{kPosAttrib}, 2, mgl::DataType::Float, GL_FALSE, 0, nullptr);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            const Layout l = layout(env);
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClear[0] / 255.0f, kClear[1] / 255.0f, kClear[2] / 255.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glBindVertexArray(vao_);

            // Left: the program linked from GLSL source.
            draw(original_, 0, l.left_w, l, kSeed, false);
            // Right: the program loaded from the binary (or, without formats, the original in grey).
            if (has_reloaded())
                draw(reloaded_, l.right_x, l.right_w, l, right_seed_, false);
            else if (num_formats_ == 0)
                draw(original_, l.right_x, l.right_w, l, right_seed_, true);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_queries(check);
            if (num_formats_ > 0)
            {
                check_round_trip(env, check);
                check_failed_loads(check);
                check_fallback(env, check);
            }
            else
            {
                check_zero_formats(env, check);
            }
            check_get_binary_errors(env, check);
            render(env, 0);  // leave the final image for the capture
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteProgram(original_);
            if (reloaded_.value != 0) mgl::glDeleteProgram(reloaded_);
            mgl::glDeleteShader(vs_);
            mgl::glDeleteShader(fs_);
        }

    private:
        bool has_reloaded() const { return reloaded_.value != 0 && reload_linked_ == GL_TRUE; }

        static mgl::ShaderId compile(mgl::ShaderType type, const std::string& source, const char* label)
        {
            std::string log;
            const mgl::ShaderId s = demo::compile_shader(type, source, log);
            if (s.value == 0) throw demo::Failure(std::format("{} failed to compile:\n{}", label, log));
            return s;
        }

        // Sets every uniform of `p` (locations are queried per program: ES 3.0 does not promise that a
        // binary reload keeps uniform locations) and fills the viewport [x, x+w) x [0, height).
        static void draw(mgl::ProgramId p, int x, int w, const Layout& l, std::uint32_t seed, bool gray)
        {
            mgl::glViewport(x, 0, w, l.height);
            mgl::glUseProgram(p);
            mgl::glUniform2f(mgl::glGetUniformLocation(p, "u_scale"), 1.0f, 1.0f);
            mgl::glUniform2i(mgl::glGetUniformLocation(p, "u_origin"), x, 0);
            mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_cell"), l.cell);
            mgl::glUniform1ui(mgl::glGetUniformLocation(p, "u_seed"), seed);
            mgl::glUniform1i(mgl::glGetUniformLocation(p, "u_gray"), gray ? 1 : 0);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);
        }

        // Counts pixels in the window-space column range [x0, x0+w) that differ by more than 1 in any channel
        // from the pattern reference drawn at origin x0 (tolerance: ES 3.0 §2.1.6.2, see pattern()).
        static int mismatches(const std::vector<std::uint8_t>& px, const demo::Env& env, int x0, int w, const Layout& l,
                              std::uint32_t seed, bool gray, std::string& first)
        {
            int bad = 0, exact = 0;
            for (int y = 0; y < env.height; ++y)
                for (int x = x0; x < x0 + w; ++x)
                {
                    const demo::Rgba8 e = pattern(x - x0, y, l.cell, seed, gray);
                    const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4;
                    const demo::Rgba8 a = {px[i], px[i + 1], px[i + 2], px[i + 3]};
                    int dev = 0;
                    for (int c = 0; c < 4; ++c) dev = std::max(dev, std::abs(int(a[c]) - int(e[c])));
                    if (dev == 0) ++exact;
                    if (dev > 1)
                    {
                        if (bad == 0) first = std::format("first at ({},{}): got {} expected {}", x, y, demo::format_rgba(a), demo::format_rgba(e));
                        ++bad;
                    }
                }
            std::printf("  columns [%d, %d): %d of %d pixels bit-exact vs the reference, %d beyond +-1\n", x0, x0 + w, exact,
                        w * env.height, bad);
            return bad;
        }

        // Pixels of the right half that differ in any bit from the left-half pixel at the same position
        // relative to its viewport (over the common width).
        static int half_differences(const std::vector<std::uint8_t>& px, const demo::Env& env, const Layout& l)
        {
            int diff = 0;
            const int w = std::min(l.left_w, l.right_w);
            for (int y = 0; y < env.height; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width);
                    const std::size_t a = (row + static_cast<std::size_t>(x)) * 4;
                    const std::size_t b = (row + static_cast<std::size_t>(l.right_x + x)) * 4;
                    if (px[a] != px[b] || px[a + 1] != px[b + 1] || px[a + 2] != px[b + 2] || px[a + 3] != px[b + 3]) ++diff;
                }
            return diff;
        }

        static int gap_mismatches(const std::vector<std::uint8_t>& px, const demo::Env& env, const Layout& l)
        {
            int bad = 0;
            for (int y = 0; y < env.height; ++y)
                for (int x = l.left_w; x < l.right_x; ++x)
                {
                    const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(env.width) + static_cast<std::size_t>(x)) * 4;
                    if (px[i] != kClear[0] || px[i + 1] != kClear[1] || px[i + 2] != kClear[2]) ++bad;
                }
            return bad;
        }

        void check_queries(demo::Check& check)
        {
            // ES 3.0 Table 6.20 / ES 3.2 Table 21.x: PROGRAM_BINARY_RETRIEVABLE_HINT has initial value FALSE.
            check.expect_eq(hint_initial_, GL_FALSE, "PROGRAM_BINARY_RETRIEVABLE_HINT is initially FALSE");
            // The spec only says the hint is "not in effect" before the next link; what the query returns in
            // between is not specified, so it is printed rather than asserted.
            std::printf("  note: hint queried after glProgramParameteri(TRUE) but before the link = %d\n", hint_before_link_);
            check.expect_eq(hint_after_link_, GL_TRUE, "after glProgramParameteri(TRUE) and a successful link the hint is TRUE");

            // ES 3.1/3.2 §7.3 (Errors of ProgramParameteri): "INVALID_VALUE ... if value is not TRUE or FALSE."
            // ES 3.0 §2.12.3 only refers to §2.12.4 for legal values and states no error, so this is asserted
            // on ES >= 3.1 contexts only.
            mgl::glProgramParameteri(original_, mgl::ProgramParameter::BinaryRetrievableHint, 2);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            if (context_ >= demo::Version{3, 1})
                check.expect(e == mgl::ErrorCode::InvalidValue,
                             "glProgramParameteri(PROGRAM_BINARY_RETRIEVABLE_HINT, 2) -> INVALID_VALUE (ES 3.2 §7.3)",
                             std::string(demo::error_name(e)));
            else
                std::printf("  note: ES 3.0 context: glProgramParameteri(hint, 2) gave %s (ES 3.0 describes no error for this case; ES 3.1+ "
                            "require INVALID_VALUE)\n", std::string(demo::error_name(e)).c_str());
            check.expect_eq(program_int(original_, mgl::ProgramParameter::BinaryRetrievableHint), GL_TRUE,
                            "the rejected hint value did not change the hint");

            check.expect(num_formats_ >= 0, std::format("NUM_PROGRAM_BINARY_FORMATS = {} (>= 0; 0 is allowed)", num_formats_));
        }

        void check_round_trip(demo::Env& env, demo::Check& check)
        {
            // --- glGetProgramBinary ---
            check.expect(get_binary_error_ == mgl::ErrorCode::NoError,
                         "glGetProgramBinary(bufSize = PROGRAM_BINARY_LENGTH) on a linked program: no error",
                         std::string(demo::error_name(get_binary_error_)));
            check.expect(binary_length_ > 0, std::format("PROGRAM_BINARY_LENGTH of the linked program = {} bytes", binary_length_));
            check.expect_eq(written_, binary_length_, "glGetProgramBinary's length equals PROGRAM_BINARY_LENGTH");
            const bool listed = std::find(formats_.begin(), formats_.end(), static_cast<GLint>(binary_format_)) != formats_.end();
            check.expect(listed, "the returned binaryFormat is one of PROGRAM_BINARY_FORMATS (§2.12.4)",
                         std::format("format 0x{:X}", static_cast<unsigned>(binary_format_)));

            // --- glProgramBinary into a fresh program ---
            check.expect(reload_error_ == mgl::ErrorCode::NoError, "glProgramBinary of the retrieved binary: no error",
                         std::string(demo::error_name(reload_error_)));
            check.expect_eq(reload_linked_, GL_TRUE,
                            "\"Any program binary retrieved using GetProgramBinary and submitted using ProgramBinary under the "
                            "same configuration must be successful\": LINK_STATUS = TRUE");
            if (!has_reloaded()) return;

            // "A successful call to ProgramBinary will reset all uniform variables to their initial values,
            //  FALSE for booleans and zero for all others." (ES 3.0 §2.12.4)
            check.expect(all_zero(reload_defaults_), "after glProgramBinary every uniform is reset (vec2 0, ivec2 0, int 0, uint 0, bool FALSE)",
                         describe(reload_defaults_));
            // Negative control for that check: the original program holds the uploaded values, so the same
            // predicate fails there.
            const UniformSnapshot orig = snapshot_uniforms(original_);
            check.expect(orig.all_found && !all_zero(orig), "negative control: the same reset predicate is false for the original program",
                         describe(orig));

            // "all vertex shader input and fragment shader output assignments that were in effect when the
            //  program was linked before saving are restored" - despite the decoy binding to attribute 2.
            const mgl::AttribLocation a = mgl::glGetAttribLocation(reloaded_, "a_pos");
            check.expect_eq(a.value, static_cast<GLint>(kPosAttrib), "a_pos keeps attribute 5 from the original link (decoy binding 2 ignored)");
            check.expect_eq(mgl::glGetFragDataLocation(reloaded_, "o_color"), mgl::glGetFragDataLocation(original_, "o_color"),
                            "o_color has the same fragment output location in both programs");

            const auto u0 = active_uniforms(original_), u1 = active_uniforms(reloaded_);
            check.expect(u0 == u1 && u0.size() == 5, "active uniforms (name, size, type) match the original (5 uniforms)",
                         std::format("original: {} reloaded: {}", describe(u0), describe(u1)));
            const auto a0 = active_attribs(original_), a1 = active_attribs(reloaded_);
            check.expect(a0 == a1 && a0.size() == 1, "active attributes (name, size, type) match the original",
                         std::format("original: {} reloaded: {}", describe(a0), describe(a1)));

            // A program that was loaded from a binary is linked, so it has a binary of its own.
            const GLint len2 = program_int(reloaded_, mgl::ProgramParameter::BinaryLength);
            std::vector<std::uint8_t> again(static_cast<std::size_t>(std::max(len2, 1)));
            GLsizei w2 = -1;
            mgl::ProgramBinaryFormat f2{};
            mgl::glGetProgramBinary(reloaded_, len2, &w2, &f2, again.data());
            check.expect_no_gl_error("glGetProgramBinary on the reloaded program succeeds");
            check.expect(len2 > 0 && w2 == len2, std::format("its binary length is PROGRAM_BINARY_LENGTH = {} (written {})", len2, w2));

            // --- Pixels: both halves pixel-exact against the CPU reference ---
            const Layout l = layout(env);
            render(env, 0);
            auto px = demo::read_rgba8(0, 0, env.width, env.height);
            std::string first;
            int bad = mismatches(px, env, 0, l.left_w, l, kSeed, false, first);
            check.expect(bad == 0, "left half (program linked from source) matches the CPU reference (+-1)", std::format("{} pixels; {}", bad, first));
            first.clear();
            bad = mismatches(px, env, l.right_x, l.right_w, l, kSeed, false, first);
            check.expect(bad == 0, "right half (program loaded with glProgramBinary) matches the CPU reference (+-1)",
                         std::format("{} pixels; {}", bad, first));
            // The binary is "a binary representation of the program object's compiled and linked executable"
            // (§2.12.4): same executable, same uniforms, same inputs -> the same bits.
            const int diff = half_differences(px, env, l);
            check.expect(diff == 0, "the reloaded program's image is bit-identical to the original's", std::format("{} pixels differ", diff));
            check.expect_eq(gap_mismatches(px, env, l), 0, "the gap between the halves keeps the clear colour");

            // Negative control: a different seed on the reloaded program must be detected by the same comparison.
            right_seed_ = kSeed ^ 0x1u;
            render(env, 0);
            px = demo::read_rgba8(0, 0, env.width, env.height);
            first.clear();
            bad = mismatches(px, env, l.right_x, l.right_w, l, kSeed, false, first);
            check.expect(bad > 0, "negative control: a wrong u_seed on the reloaded program is detected", std::format("{} differing pixels", bad));
            check.expect(half_differences(px, env, l) > 0, "negative control: ...and the bit-identity comparison detects it too");
            right_seed_ = kSeed;
        }

        // Loads `data` into `p` and returns {GL error, LINK_STATUS, PROGRAM_BINARY_LENGTH}.
        struct LoadResult
        {
            mgl::ErrorCode error;
            GLint linked;
            GLint length;
        };

        LoadResult load(mgl::ProgramId p, mgl::ProgramBinaryFormat format, const std::vector<std::uint8_t>& data, GLsizei length)
        {
            mgl::glProgramBinary(p, format, data.data(), length);
            const mgl::ErrorCode e = demo::drain_gl_errors();
            return {e, program_int(p, mgl::ProgramParameter::LinkStatus), program_int(p, mgl::ProgramParameter::BinaryLength)};
        }

        void check_failed_loads(demo::Check& check)
        {
            const mgl::ProgramId p = mgl::glCreateProgram();
            const GLsizei n = written_;

            check_mutations_isolated(check);

            // After a failed load the program is "last linked unsuccessfully": UseProgram must refuse it
            // (ES 3.0 §2.12.3) and GetProgramBinary must refuse it (§2.12.4). The failed load made in this process inverts
            // the first byte, a header mutation that every tested driver rejects cleanly; the other mutations are loaded
            // in child processes above.
            {
                auto d = binary_;
                d[0] ^= 0xFFu;
                const LoadResult ok = load(p, binary_format_, binary_, n);
                const LoadResult r = load(p, binary_format_, d, n);
                check.expect(ok.linked == GL_TRUE && r.error == mgl::ErrorCode::NoError && r.linked == GL_FALSE,
                             "in-process: a binary with an inverted first byte loaded over a valid load fails (no error, LINK_STATUS FALSE)",
                             std::format("error {}, LINK_STATUS {}", demo::error_name(r.error), r.linked));
            }
            mgl::glUseProgram(p);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glUseProgram on a program whose binary load failed -> INVALID_OPERATION");
            {
                std::vector<std::uint8_t> buf(binary_.size());
                GLsizei w = -7;
                mgl::ProgramBinaryFormat f{0x7777u};
                mgl::glGetProgramBinary(p, static_cast<GLsizei>(buf.size()), &w, &f, buf.data());
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetProgramBinary on a program whose binary load failed -> INVALID_OPERATION");
                expect_outputs_untouched(check, w, f, "glGetProgramBinary on a failed load");
            }

            // Invalid binaryFormat: ES 3.2 §7.5 "INVALID_ENUM ... if binaryFormat is not a binary format present in
            // the list"; ES 3.0 §2.5 (an enum that is not allowable for the command) gives the same error. "The
            // command generating the error is ignored so that it has no effect on GL state" (ES 3.0 §2.5), so the
            // previously loaded program must stay linked.
            const LoadResult good = load(p, binary_format_, binary_, n);
            check.expect_eq(good.linked, GL_TRUE, "the valid binary loads again into the same program object");
            const mgl::ProgramBinaryFormat bogus = unlisted_format();
            mgl::glProgramBinary(p, bogus, binary_.data(), n);
            check.expect_gl_error(mgl::ErrorCode::InvalidEnum,
                                  std::format("glProgramBinary with binaryFormat 0x{:X} (not in PROGRAM_BINARY_FORMATS) -> INVALID_ENUM",
                                              static_cast<unsigned>(bogus)));
            GLint linked = program_int(p, mgl::ProgramParameter::LinkStatus);
            check.expect_or_known_deviation(linked == GL_TRUE, "the INVALID_ENUM call had no effect: the program is still linked",
                                            std::format("LINK_STATUS {}", linked), "Mesa", kMesaClearsFirst);

            // Negative length: ES 3.2 §7.5 "INVALID_VALUE ... if length is negative"; ES 3.0 §2.5 sizei rule.
            load(p, binary_format_, binary_, n);
            mgl::glProgramBinary(p, binary_format_, binary_.data(), -1);
            {
                const mgl::ErrorCode e = demo::drain_gl_errors();
                check.expect_or_known_deviation(e == mgl::ErrorCode::InvalidValue, "glProgramBinary with length -1 -> INVALID_VALUE",
                                                std::format("glGetError() = {}", demo::error_name(e)), "ANGLE",
                                                "ANGLE's glProgramBinary raises no error for a negative length (ES 3.0 2.5 sizei rule; "
                                                "ES 3.2 7.5: INVALID_VALUE if length is negative)");
            }
            linked = program_int(p, mgl::ProgramParameter::LinkStatus);
            check.expect_or_known_deviation(linked == GL_TRUE, "the INVALID_VALUE call had no effect: the program is still linked",
                                            std::format("LINK_STATUS {}", linked), "Mesa", kMesaClearsFirst);

            mgl::glDeleteProgram(p);
        }

        // The mutated loads, each in a fresh child process (see MutationChild).
        static void check_mutations_isolated(demo::Check& check)
        {
            const bool angle = renderer_is_angle();
            constexpr std::string_view kAngleLoads =
                "ANGLE does not validate a program binary against what GetProgramBinary returned: a byte inverted in the middle or at the "
                "end, a flipped bit, a length one byte short or 16 bytes long all load with LINK_STATUS TRUE (ES 3.0 2.12.4: loading "
                "will fail if these conditions are not met)";
            constexpr std::string_view kAngleCrash =
                "ANGLE crashes inside glProgramBinary for a truncated binary (SIGILL) instead of failing the load with LINK_STATUS FALSE "
                "(ES 3.0 2.12.4)";
            constexpr std::string_view kMesaTrailing =
                "Mesa's CRC covers only the payload size stored in the header and ignores trailing bytes, so an over-long "
                "length still loads (ES 3.0 2.12.4 requires the load to fail)";
            for (const MutationCase& c : kMutations)
            {
                const ChildResult r = run_child(c.name, std::chrono::seconds{30});
                const ChildLoad l = parse_child_load(r);
                const std::string what = std::format("binary {} (isolated child): glProgramBinary generates no error, LINK_STATUS FALSE, "
                                                     "PROGRAM_BINARY_LENGTH 0, and the process survives", c.what);
                const std::string detail = l.found ? std::format("valid load first: LINK_STATUS {}; mutated: error {}, LINK_STATUS {}, length {}; {}",
                                                                 l.valid_linked, l.error, l.linked, l.length, r.describe())
                                                   : r.describe();
                // The documented deviations: a death inside glProgramBinary, or a mutated binary that loads successfully.
                // Anything else (a wrong error, a failing valid load, a crash during setup) is a plain failure.
                const bool loaded = r.exited && r.returned_from_call() && l.found && l.valid_linked == GL_TRUE && l.error == "GL_NO_ERROR" &&
                                    l.linked == GL_TRUE;
                if (r.passed())
                    check.expect(true, what, detail);
                else if (r.died_in_call())
                    check.expect_or_known_deviation(false, what, detail, "ANGLE", kAngleCrash);
                else if (loaded && c.name == "extra-16" && !angle)
                    check.expect_or_known_deviation(false, what, detail, "Mesa", kMesaTrailing);
                else if (loaded)
                    check.expect_or_known_deviation(false, what, detail, "ANGLE", kAngleLoads);
                else
                    check.expect(false, what, detail);
            }
        }

        // ES 3.0 §2.5: "If the generating command modifies values through a pointer argument, no change is made
        // to these values."
        // `angle_writes`: ANGLE writes both outputs on this path (the too-small bufSize).
        static void expect_outputs_untouched(demo::Check& check, GLsizei length, mgl::ProgramBinaryFormat format, std::string_view call,
                                             bool angle_writes = false)
        {
            constexpr std::string_view kAngleWrites =
                "ANGLE's glGetProgramBinary with a too-small bufSize raises INVALID_OPERATION but stores *binaryFormat and *length = 0 "
                "(ES 3.0 2.5: the erroring command makes no change to values through pointer arguments)";
            const bool angle = angle_writes && renderer_is_angle();
            const std::string format_what = std::format("{}: binaryFormat was not written", call);
            const std::string format_detail = std::format("0x{:X}", static_cast<unsigned>(format));
            if (angle) check.expect_or_known_deviation(static_cast<GLenum>(format) == 0x7777u, format_what, format_detail, "ANGLE", kAngleWrites);
            else check.expect(static_cast<GLenum>(format) == 0x7777u, format_what, format_detail);
            check.expect_or_known_deviation(length == -7, std::format("{}: length was not written (ES 3.0 §2.5)", call),
                                            std::format("length = {}", length), angle ? "ANGLE" : "Mesa",
                                            angle ? kAngleWrites : "Mesa's glGetProgramBinary stores *length = 0 on its INVALID_OPERATION paths");
        }

        // Fallback path (§2.12.4: "the application should fall back to providing the original ... source
        // shaders"): a failed load keeps the attached shaders and the BindAttribLocation bindings, so the
        // very same program object can simply be re-linked from source.
        void check_fallback(const demo::Env& env, demo::Check& check)
        {
            const mgl::ProgramId p = mgl::glCreateProgram();
            mgl::glAttachShader(p, vs_);
            mgl::glAttachShader(p, fs_);
            mgl::glBindAttribLocation(p, mgl::AttribLocation{kPosAttrib}, "a_pos");
            // A stale binary as a driver update produces it: the header no longer matches. (Arbitrary corruptions are
            // loaded in child processes, see check_mutations_isolated: one driver accepts or crashes on some of them.)
            auto stale = binary_;
            stale[0] ^= 0x5Au;
            mgl::glProgramBinary(p, binary_format_, stale.data(), written_);
            check.expect_no_gl_error("fallback: loading a stale binary generates no error");
            check.expect_eq(program_int(p, mgl::ProgramParameter::LinkStatus), GL_FALSE, "fallback: LINK_STATUS FALSE tells the application to fall back");
            check.expect_eq(program_int(p, mgl::ProgramParameter::AttachedShaders), 2,
                            "fallback: the failed load did not detach the shaders (ATTACHED_SHADERS = 2)");
            mgl::glLinkProgram(p);
            check.expect_eq(program_int(p, mgl::ProgramParameter::LinkStatus), GL_TRUE, "fallback: glLinkProgram from the attached source succeeds");
            check.expect_eq(mgl::glGetAttribLocation(p, "a_pos").value, static_cast<GLint>(kPosAttrib),
                            "fallback: the BindAttribLocation binding survived the failed load");

            // The re-linked program renders the reference pattern into the left half.
            const Layout l = layout(env);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glBindVertexArray(vao_);
            draw(p, 0, l.left_w, l, kSeed, false);
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            const auto px = demo::read_rgba8(0, 0, env.width, env.height);
            std::string first;
            const int bad = mismatches(px, env, 0, l.left_w, l, kSeed, false, first);
            check.expect(bad == 0, "fallback: the re-linked program renders the reference pattern (+-1)", std::format("{} pixels; {}", bad, first));

            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDetachShader(p, vs_);
            mgl::glDetachShader(p, fs_);
            mgl::glDeleteProgram(p);
        }

        void check_zero_formats(demo::Env& env, demo::Check& check)
        {
            std::printf("  NUM_PROGRAM_BINARY_FORMATS = 0: no round trip possible; asserting the rules that remain defined.\n");
            // With an empty list no value "is a binary format present in the list" (ES 3.2 §7.5), and none is
            // "allowable for that command" (ES 3.0 §2.5): every binaryFormat -> INVALID_ENUM.
            const mgl::ProgramId p = mgl::glCreateProgram();
            const std::array<GLenum, 4> values = {0u, 1u, 0x8741u /* PROGRAM_BINARY_LENGTH */, 0x8B6Fu};
            const std::array<std::uint8_t, 64> junk{};
            for (GLenum v : values)
            {
                mgl::glProgramBinary(p, mgl::ProgramBinaryFormat{v}, junk.data(), static_cast<GLsizei>(junk.size()));
                check.expect_gl_error(mgl::ErrorCode::InvalidEnum, std::format("no formats: glProgramBinary(binaryFormat 0x{:X}) -> INVALID_ENUM", v));
            }
            mgl::glDeleteProgram(p);

            // ES 3.2 §7.5: "An INVALID_OPERATION error is generated if the value of NUM_PROGRAM_BINARY_FORMATS
            // is zero." (GetProgramBinary). ES 3.0/3.1 have no such rule, so it is asserted on 3.2 only.
            std::array<std::uint8_t, 4096> buf{};
            GLsizei w = -7;
            mgl::ProgramBinaryFormat f{0x7777u};
            mgl::glGetProgramBinary(original_, static_cast<GLsizei>(buf.size()), &w, &f, buf.data());
            const mgl::ErrorCode e = demo::drain_gl_errors();
            if (context_ >= demo::Version{3, 2})
                check.expect(e == mgl::ErrorCode::InvalidOperation, "no formats: glGetProgramBinary on a linked program -> INVALID_OPERATION (ES 3.2 §7.5)",
                             std::string(demo::error_name(e)));
            else
                std::printf("  note: ES %d.%d context: glGetProgramBinary with no formats gave %s (only ES 3.2 defines it)\n",
                            context_.major, context_.minor, std::string(demo::error_name(e)).c_str());

            // Pixels: left = pattern, right = the same program in grey.
            const Layout l = layout(env);
            render(env, 0);
            auto px = demo::read_rgba8(0, 0, env.width, env.height);
            std::string first;
            int bad = mismatches(px, env, 0, l.left_w, l, kSeed, false, first);
            check.expect(bad == 0, "left half matches the CPU reference (+-1)", std::format("{} pixels; {}", bad, first));
            first.clear();
            bad = mismatches(px, env, l.right_x, l.right_w, l, kSeed, true, first);
            check.expect(bad == 0, "right half (grey: no binary formats) matches the CPU reference (+-1)", std::format("{} pixels; {}", bad, first));
            right_seed_ = kSeed ^ 0x1u;
            render(env, 0);
            px = demo::read_rgba8(0, 0, env.width, env.height);
            bad = mismatches(px, env, l.right_x, l.right_w, l, kSeed, true, first);
            check.expect(bad > 0, "negative control: a wrong u_seed on the right half is detected", std::format("{} differing pixels", bad));
            right_seed_ = kSeed;
        }

        // Both cases: ES 3.0 §2.12.4 "If bufSize is less than the number of bytes in the program binary, then
        // an INVALID_OPERATION error is generated" and "When a program object's LINK_STATUS is FALSE, its
        // program binary length is zero, and a call to GetProgramBinary will generate an INVALID_OPERATION error."
        void check_get_binary_errors(const demo::Env& /*env*/, demo::Check& check)
        {
            if (num_formats_ > 0)
            {
                std::vector<std::uint8_t> buf(binary_.size(), 0xCDu);
                GLsizei w = -7;
                mgl::ProgramBinaryFormat f{0x7777u};
                mgl::glGetProgramBinary(original_, binary_length_ - 1, &w, &f, buf.data());
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetProgramBinary with bufSize = PROGRAM_BINARY_LENGTH - 1 -> INVALID_OPERATION");
                expect_outputs_untouched(check, w, f, "glGetProgramBinary with a too-small bufSize", true);
                mgl::glGetProgramBinary(original_, 0, &w, &f, buf.data());
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetProgramBinary with bufSize = 0 -> INVALID_OPERATION");
            }
            else
            {
                std::printf("  note: with no formats, bufSize < PROGRAM_BINARY_LENGTH cannot be isolated from the "
                            "zero-format INVALID_OPERATION (ES 3.2 §7.5); only the unlinked case is asserted.\n");
            }

            const mgl::ProgramId never_linked = mgl::glCreateProgram();
            check.expect_eq(program_int(never_linked, mgl::ProgramParameter::BinaryLength), 0,
                            "a program that was never linked has PROGRAM_BINARY_LENGTH 0");
            std::array<std::uint8_t, 256> buf{};
            GLsizei w = -7;
            mgl::ProgramBinaryFormat f{0x7777u};
            mgl::glGetProgramBinary(never_linked, static_cast<GLsizei>(buf.size()), &w, &f, buf.data());
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "glGetProgramBinary on a never-linked program -> INVALID_OPERATION");
            expect_outputs_untouched(check, w, f, "glGetProgramBinary on a never-linked program");
            mgl::glDeleteProgram(never_linked);

            // A program whose link from source failed (the vertex shader alone: no fragment shader).
            const mgl::ProgramId failed = mgl::glCreateProgram();
            mgl::glAttachShader(failed, vs_);
            mgl::glLinkProgram(failed);
            const GLint linked = program_int(failed, mgl::ProgramParameter::LinkStatus);
            mgl::glGetProgramBinary(failed, static_cast<GLsizei>(buf.size()), &w, &f, buf.data());
            const mgl::ErrorCode e = demo::drain_gl_errors();
            check.expect(linked == GL_FALSE && e == mgl::ErrorCode::InvalidOperation,
                         "glGetProgramBinary on a program whose link failed (no fragment shader) -> INVALID_OPERATION",
                         std::format("LINK_STATUS {}, error {}", linked, demo::error_name(e)));
            mgl::glDetachShader(failed, vs_);
            mgl::glDeleteProgram(failed);
        }

        mgl::ProgramBinaryFormat unlisted_format() const
        {
            GLenum v = 0x1234u;
            while (std::find(formats_.begin(), formats_.end(), static_cast<GLint>(v)) != formats_.end()) ++v;
            return mgl::ProgramBinaryFormat{v};
        }

        demo::Version context_{};
        mgl::ShaderId vs_{}, fs_{};
        mgl::ProgramId original_{}, reloaded_{};
        mgl::BufferId vbo_{};
        mgl::VertexArrayId vao_{};

        GLint hint_initial_ = -1, hint_before_link_ = -1, hint_after_link_ = -1;
        GLint num_formats_ = -1;
        std::vector<GLint> formats_;
        GLint binary_length_ = -1;
        GLsizei written_ = -1;
        std::vector<std::uint8_t> binary_;
        mgl::ProgramBinaryFormat binary_format_{};
        mgl::ErrorCode get_binary_error_ = mgl::ErrorCode::NoError;
        mgl::ErrorCode reload_error_ = mgl::ErrorCode::NoError;
        GLint reload_linked_ = GL_FALSE;
        UniformSnapshot reload_defaults_;
        std::uint32_t right_seed_ = kSeed;
    };
}

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string_view(argv[1]) == kChildFlag)
    {
        bool known = false;
        for (const MutationCase& c : kMutations) known = known || c.name == argv[2];
        if (!known)
        {
            std::cerr << "unknown child case '" << argv[2] << "'\n";
            return 2;
        }
        g_child_case = argv[2];
        std::array<char*, 3> child_argv = {argv[0], const_cast<char*>("--headless"), const_cast<char*>("--validate")};
        const demo::Info child_info{
            .name = "es30-program-binary(child)",
            .title = "glProgramBinary with a mutated binary: isolated child process",
            .min_es = {3, 0},
            .summary = "Internal child process of es30-program-binary.",
            .source_dir = DEMO_SOURCE_DIR,
        };
        return demo::run<MutationChild>(static_cast<int>(child_argv.size()), child_argv.data(), child_info);
    }

    const demo::Info info{
        .name = "es30-program-binary",
        .title = "Program binaries",
        .min_es = {3, 0},
        .summary = "glGetProgramBinary / glProgramBinary round trip: the reloaded program renders a Truchet pattern "
                   "pixel-identical to the original, uniforms are reset, attribute assignments restored, corrupted "
                   "binaries fail with LINK_STATUS FALSE, and the error cases of ES 3.0 §2.12.4 are provoked.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ProgramBinary>(argc, argv, info);
}
