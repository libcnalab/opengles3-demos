// test-driver-crash-readpixels-none
//
// ES 3.0 §4.3.2.1: "ReadPixels generates an INVALID_OPERATION error if READ_BUFFER is NONE or if the GL
// is using a framebuffer object (i.e. READ_FRAMEBUFFER_BINDING is non-zero) and the read buffer selects
// an attachment that has no image attached."
//
// Mesa 25.0.7 (radeonsi, llvmpipe, zink) does not raise the error but crashes with SIGSEGV inside
// glReadPixels when READ_BUFFER is NONE and the framebuffer object has an integer color attachment.
// A crash cannot be observed in-process, so every call that may crash runs in a separate process.
//
// Process isolation: the parent re-executes its own binary (/proc/self/exe) with a hidden first
// argument (--readpixels-none-child <case>). The child is a complete fresh process that goes through
// demo::run, so it creates its own EGL display, context and meta-gl function table exactly as every
// demo does. A forked copy of the parent cannot be used instead: an EGL/GL context is not specified to
// survive fork(), only the forking thread is duplicated (the driver's worker threads, e.g. the llvmpipe
// rasterizer or glthread, are not), and the child shares the parent's DRM file descriptors. Measured on
// Mesa 25.0.7: a forked child using the inherited context gets an answer from glGetError but hangs in
// the first glClear/glReadPixels on radeonsi, llvmpipe and zink alike. The child
// inherits the environment (EGL_PLATFORM, GALLIUM_DRIVER, MESA_GLES_VERSION_OVERRIDE), so it runs on
// the same driver and API version as the parent. The parent captures the child's output, echoes it,
// and classifies how the child ended (exit status, signal, or timeout).
//
// Parent (in-process, never crashes on any tested driver): the same §4.3.2.1 rule on a framebuffer
// object with only a normalized attachment, the missing-image rule, the matching
// IMPLEMENTATION_COLOR_READ_FORMAT/TYPE errors (§4.3.2), and READ_BUFFER = NONE on the default
// framebuffer (§4.3.1).
//
// Child cases: three controls that prove the harness distinguishes a passing child, a failing child and
// a crashing child, and three READ_BUFFER = NONE cases on framebuffers with integer attachments.

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
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace mgl = metagl;

namespace
{
    constexpr std::string_view kChildFlag = "--readpixels-none-child";
    // Printed by the child immediately before / after the glReadPixels call that may crash. The parent
    // uses them to tell a crash inside glReadPixels from a crash anywhere else.
    constexpr std::string_view kBeforeMarker = "@@ before glReadPixels";
    constexpr std::string_view kAfterMarker = "@@ after glReadPixels";

    enum class ChildCase
    {
        ControlRead,      // mixed RGBA8 + RG32I framebuffer, legal reads from both attachments: must exit 0
        ControlFail,      // a deliberately wrong expectation: the child must exit 1 (harness sees FAIL)
        ControlSignal,    // the child raises SIGSEGV itself: the harness must report death by SIGSEGV
        NoneMixedRgba,    // COLOR0 = RGBA8, COLOR3 = RG32I, READ_BUFFER = NONE, RGBA / UNSIGNED_BYTE
        NoneMixedInteger, // same framebuffer, RGBA_INTEGER / INT
        NoneIntegerOnly,  // COLOR0 = RG32I only, READ_BUFFER = NONE, RGBA_INTEGER / INT
    };

    struct CaseName
    {
        ChildCase id;
        std::string_view name;
    };
    constexpr std::array<CaseName, 6> kCases = {{
        {ChildCase::ControlRead, "control-read"},
        {ChildCase::ControlFail, "control-fail"},
        {ChildCase::ControlSignal, "control-signal"},
        {ChildCase::NoneMixedRgba, "none-mixed-rgba"},
        {ChildCase::NoneMixedInteger, "none-mixed-integer"},
        {ChildCase::NoneIntegerOnly, "none-integer-only"},
    }};

    std::string_view case_name(ChildCase c)
    {
        for (const auto& n : kCases)
            if (n.id == c) return n.name;
        return "?";
    }

    ChildCase g_child_case = ChildCase::ControlRead;  // set by main() in child mode

    constexpr int kSize = 4;
    constexpr std::array<GLfloat, 4> kClearF = {0.25f, 0.5f, 0.75f, 1.0f};
    constexpr demo::Rgba8 kClearU8 = {64, 128, 191, 255};  // round(c * 255), ES 3.0 §2.1.6.1 (+-1 below)
    constexpr std::array<GLint, 4> kClearI = {-7, 123456, 99, 99};  // B, A not stored by RG32I
    constexpr std::uint8_t kSentinel = 0xAB;

    // =============================================================================================
    // Child: one GL call that may crash, in a fresh process with its own context.
    // =============================================================================================
    class ChildProbe final : public demo::Demo
    {
    public:
        void init(demo::Env& /*env*/) override
        {
            const bool integer_only = g_child_case == ChildCase::NoneIntegerOnly;
            mgl::glGenRenderbuffers(rb_);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_[0]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer,
                                       integer_only ? mgl::InternalFormat::Rg32I : mgl::InternalFormat::Rgba8, kSize, kSize);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_[1]);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rg32I, kSize, kSize);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});

            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer,
                                           mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rb_[0]);
            if (!integer_only)
                mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer,
                                               mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color3),
                                               mgl::RenderbufferTarget::Renderbuffer, rb_[1]);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("child framebuffer is not complete: {}", demo::framebuffer_status_name(status)));

            // Known contents: draw buffer i must be COLOR_ATTACHMENTi or NONE (ES 3.0 §4.2.1).
            const mgl::DrawBuffer c0 = mgl::to_draw_buffer(mgl::ColorAttachment::Color0);
            const mgl::DrawBuffer c3 = mgl::to_draw_buffer(mgl::ColorAttachment::Color3);
            if (integer_only)
            {
                mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 0, kClearI.data());
            }
            else
            {
                const std::array<mgl::DrawBuffer, 4> bufs = {c0, mgl::DrawBuffer::None, mgl::DrawBuffer::None, c3};
                mgl::glDrawBuffers(bufs);
                mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, kClearF.data());
                mgl::glClearBufferiv(mgl::SignedIntegerClearBuffer::Color, 3, kClearI.data());
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            std::cout << std::format("   child case: {}\n", case_name(g_child_case));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            switch (g_child_case)
            {
            case ChildCase::ControlRead: control_read(check); break;
            case ChildCase::ControlFail:
                mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
                read_marked(mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte);
                // Deliberately wrong: the read above is legal. This check must fail, so the child exits 1.
                check.expect_gl_error(mgl::ErrorCode::InvalidOperation, "control-fail: deliberately wrong expectation (must FAIL)");
                break;
            case ChildCase::ControlSignal:
                check.expect(true, "control-signal: about to raise SIGSEGV");
                std::cout << std::flush;
                std::raise(SIGSEGV);
                break;
            case ChildCase::NoneMixedRgba: read_none(check, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte); break;
            case ChildCase::NoneMixedInteger:
            case ChildCase::NoneIntegerOnly: read_none(check, mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int); break;
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteRenderbuffers(rb_);
        }

    private:
        std::array<std::uint8_t, 64> sink_{};

        void read_marked(mgl::PixelFormat format, mgl::PixelType type)
        {
            sink_.fill(kSentinel);
            std::cout << kBeforeMarker << "\n" << std::flush;
            mgl::glReadPixels(1, 1, 1, 1, format, type, sink_.data());
            std::cout << kAfterMarker << "\n" << std::flush;
        }

        void control_read(demo::Check& check)
        {
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            read_marked(mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte);
            check.expect_no_gl_error("control-read: RGBA / UNSIGNED_BYTE from the RGBA8 COLOR_ATTACHMENT0");
            bool near = true;
            for (std::size_t i = 0; i < 4; ++i) near = near && std::abs(int{sink_[i]} - int{kClearU8[i]}) <= 1;
            check.expect(near, "control-read: COLOR_ATTACHMENT0 holds the glClearBufferfv color",
                         std::format("got ({},{},{},{}) expected {} +-1", sink_[0], sink_[1], sink_[2], sink_[3], demo::format_rgba(kClearU8)));

            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color3));
            read_marked(mgl::PixelFormat::RgbaInteger, mgl::PixelType::Int);
            check.expect_no_gl_error("control-read: RGBA_INTEGER / INT from the RG32I COLOR_ATTACHMENT3");
            std::array<GLint, 4> px{};
            std::memcpy(px.data(), sink_.data(), sizeof px);
            // RG32I stores R and G; B and A read as 0 and 1 (ES 3.0 §4.3.2.1, Table 3.11).
            check.expect(px == std::array<GLint, 4>{kClearI[0], kClearI[1], 0, 1},
                         "control-read: COLOR_ATTACHMENT3 reads back (R, G, 0, 1) of glClearBufferiv",
                         std::format("got ({},{},{},{})", px[0], px[1], px[2], px[3]));
        }

        void read_none(demo::Check& check, mgl::PixelFormat format, mgl::PixelType type)
        {
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            check.expect_no_gl_error("glReadBuffer(NONE) on the framebuffer object is valid (ES 3.0 §4.3.1)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), GL_NONE, "READ_BUFFER reads back NONE");
            read_marked(format, type);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  std::format("{}: glReadPixels with READ_BUFFER = NONE -> INVALID_OPERATION (ES 3.0 §4.3.2.1)",
                                              case_name(g_child_case)));
            bool untouched = true;
            for (const std::uint8_t b : sink_) untouched = untouched && b == kSentinel;
            check.expect(untouched, "the failing glReadPixels writes nothing (ES 3.0 §2.5)");
        }

        std::array<mgl::RenderbufferId, 2> rb_{};
        mgl::FramebufferId fbo_{};
    };

    // =============================================================================================
    // Parent: child-process harness.
    // =============================================================================================
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
        [[nodiscard]] std::string describe() const
        {
            if (!spawned) return "child could not be spawned";
            if (timed_out) return "child timed out and was killed";
            if (signaled)
                return std::format("child killed by signal {} ({}){}", signal, sigabbrev_np(signal) ? sigabbrev_np(signal) : "?",
                                   reached_call() && !returned_from_call() ? " inside glReadPixels" : "");
            return std::format("child exited with status {}", exit_code);
        }
    };

    /// Environment of the child: ours, plus ASan options that leave SIGSEGV/SIGABRT to the default action,
    /// so that a crash is reported as a signal also in OGLES3_SANITIZE builds (ASan would otherwise turn it
    /// into exit status 1, indistinguishable from FAIL). Harmless without ASan.
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
        env.push_back(asan + "handle_segv=0:handle_sigbus=0:handle_abort=0");
        return env;
    }

    ChildResult run_child(ChildCase c, std::chrono::seconds timeout)
    {
        ChildResult r;
        const std::string name(case_name(c));
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

    class ReadPixelsNone final : public demo::Demo
    {
    public:
        void init(demo::Env& /*env*/) override
        {
            mgl::glGenRenderbuffers(1, &rb_);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb_);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, kSize, kSize);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glGenFramebuffers(1, &fbo_);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer,
                                           mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0),
                                           mgl::RenderbufferTarget::Renderbuffer, rb_);
            const mgl::FramebufferStatus status = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer);
            if (status != mgl::FramebufferStatus::Complete)
                throw demo::Failure(std::format("framebuffer is not complete: {}", demo::framebuffer_status_name(status)));
            mgl::glClearBufferfv(mgl::FloatClearBuffer::Color, 0, kClearF.data());
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(kClearF[0], kClearF[1], kClearF[2], kClearF[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            check_normalized_fbo(check);
            check_default_framebuffer(check);
            check_children(check);
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo_);
            mgl::glDeleteRenderbuffers(1, &rb_);
        }

    private:
        mgl::RenderbufferId rb_{};
        mgl::FramebufferId fbo_{};

        /// glReadPixels of one RGBA/UNSIGNED_BYTE pixel into a sentinel-filled buffer; returns the buffer.
        static std::array<std::uint8_t, 16> read_one()
        {
            std::array<std::uint8_t, 16> sink{};
            sink.fill(kSentinel);
            mgl::glReadPixels(1, 1, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, sink.data());
            return sink;
        }

        static bool untouched(const std::array<std::uint8_t, 16>& sink)
        {
            for (const std::uint8_t b : sink)
                if (b != kSentinel) return false;
            return true;
        }

        static bool is_clear_color(const std::array<std::uint8_t, 16>& sink)
        {
            for (std::size_t i = 0; i < 4; ++i)
                if (std::abs(int{sink[i]} - int{kClearU8[i]}) > 1) return false;
            return true;
        }

        /// The implementation-chosen read format/type query must fail with INVALID_OPERATION when there is
        /// nothing to read (ES 3.0 §4.3.2).
        static void expect_impl_read_queries_fail(demo::Check& check, std::string_view when)
        {
            GLint v = 0;
            mgl::glGetIntegerv(mgl::GetParameter::ImplementationColorReadFormat, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  std::format("GetIntegerv(IMPLEMENTATION_COLOR_READ_FORMAT) with {} -> INVALID_OPERATION (§4.3.2)", when));
            mgl::glGetIntegerv(mgl::GetParameter::ImplementationColorReadType, &v);
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  std::format("GetIntegerv(IMPLEMENTATION_COLOR_READ_TYPE) with {} -> INVALID_OPERATION (§4.3.2)", when));
        }

        void check_normalized_fbo(demo::Check& check) const
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo_);
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), GL_COLOR_ATTACHMENT0,
                            "a new framebuffer object reads from COLOR_ATTACHMENT0 (§4.3.1 initial value)");

            // Positive control: the same read is legal while the read buffer selects the RGBA8 image.
            auto px = read_one();
            check.expect_no_gl_error("RGBA8-only FBO, READ_BUFFER = COLOR_ATTACHMENT0: glReadPixels succeeds");
            check.expect(is_clear_color(px), "the read returns the cleared color (control for the error cases)",
                         std::format("got ({},{},{},{})", px[0], px[1], px[2], px[3]));

            // READ_BUFFER = NONE on a framebuffer without integer attachments.
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            check.expect_no_gl_error("glReadBuffer(NONE) on a framebuffer object is valid (§4.3.1)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), GL_NONE, "READ_BUFFER reads back NONE");
            px = read_one();
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "RGBA8-only FBO, READ_BUFFER = NONE: glReadPixels -> INVALID_OPERATION (§4.3.2.1)");
            check.expect(untouched(px), "the failing glReadPixels writes nothing (§2.5)");
            expect_impl_read_queries_fail(check, "READ_BUFFER = NONE");

            // The read buffer selects an attachment with no image. COLOR_ATTACHMENT1 is legal for glReadBuffer
            // because MAX_COLOR_ATTACHMENTS >= 4 (ES 3.0 Table 6.37). Completeness does not depend on the read
            // buffer in ES 3.0 (§4.4.4.2), so the error must be INVALID_OPERATION, not INVALID_FRAMEBUFFER_OPERATION.
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color1));
            check.expect_no_gl_error("glReadBuffer(COLOR_ATTACHMENT1) with no image attached there is valid (§4.3.1)");
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), GL_COLOR_ATTACHMENT1, "READ_BUFFER reads back COLOR_ATTACHMENT1");
            check.expect(mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::ReadFramebuffer) == mgl::FramebufferStatus::Complete,
                         "the framebuffer stays complete with a read buffer that has no image (§4.4.4.2)");
            px = read_one();
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "READ_BUFFER selects an attachment with no image: glReadPixels -> INVALID_OPERATION (§4.3.2.1)");
            check.expect(untouched(px), "the failing glReadPixels writes nothing (§2.5)");
            expect_impl_read_queries_fail(check, "a read buffer without an image");

            // Mutation back: selecting the attachment again makes the identical call succeed, so the errors
            // above came from the read-buffer state alone.
            mgl::glReadBuffer(mgl::to_read_buffer(mgl::ColorAttachment::Color0));
            px = read_one();
            check.expect_no_gl_error("READ_BUFFER = COLOR_ATTACHMENT0 again: glReadPixels succeeds");
            check.expect(is_clear_color(px), "and returns the cleared color again",
                         std::format("got ({},{},{},{})", px[0], px[1], px[2], px[3]));
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
        }

        static void check_default_framebuffer(demo::Check& check)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_eq(demo::get_integer(mgl::GetParameter::ReadBuffer), GL_BACK,
                            "the default framebuffer reads from BACK initially (§4.3.1)");
            mgl::glReadBuffer(mgl::ReadBuffer::None);
            check.expect_no_gl_error("glReadBuffer(NONE) on the default framebuffer is valid (§4.3.1)");
            auto px = read_one();
            check.expect_gl_error(mgl::ErrorCode::InvalidOperation,
                                  "default framebuffer, READ_BUFFER = NONE: glReadPixels -> INVALID_OPERATION (§4.3.2.1)");
            check.expect(untouched(px), "the failing glReadPixels writes nothing (§2.5)");
            mgl::glReadBuffer(mgl::ReadBuffer::Back);
            px = read_one();
            check.expect_no_gl_error("READ_BUFFER = BACK again: glReadPixels succeeds");
            check.expect(is_clear_color(px), "and returns the default framebuffer's clear color",
                         std::format("got ({},{},{},{})", px[0], px[1], px[2], px[3]));
        }

        static void check_children(demo::Check& check)
        {
            constexpr std::chrono::seconds kTimeout{30};

            // Harness controls: a passing, a failing and a crashing child must be told apart.
            const ChildResult ok = run_child(ChildCase::ControlRead, kTimeout);
            check.expect(ok.exited && ok.exit_code == 0 && ok.returned_from_call(),
                         "control-read child (fresh process, own context, legal reads of RGBA8 and RG32I) exits with PASS",
                         ok.describe());
            const ChildResult fail = run_child(ChildCase::ControlFail, kTimeout);
            check.expect(fail.exited && fail.exit_code == 1, "control-fail child (deliberately wrong check) exits with FAIL (1)",
                         fail.describe());
            const ChildResult sig = run_child(ChildCase::ControlSignal, kTimeout);
            check.expect(sig.signaled && sig.signal == SIGSEGV && !sig.reached_call(),
                         "control-signal child is reported as killed by SIGSEGV", sig.describe());

            // The rule under test, on framebuffers with an integer color attachment.
            constexpr std::string_view kReason =
                "Mesa segfaults in glReadPixels when READ_BUFFER is NONE and the framebuffer has an integer color "
                "attachment, instead of raising INVALID_OPERATION (docs/KNOWN_LIMITATIONS.md)";
            const std::array<std::pair<ChildCase, std::string_view>, 3> cases = {{
                {ChildCase::NoneMixedRgba, "COLOR0 = RGBA8 + COLOR3 = RG32I, READ_BUFFER = NONE, RGBA / UNSIGNED_BYTE"},
                {ChildCase::NoneMixedInteger, "COLOR0 = RGBA8 + COLOR3 = RG32I, READ_BUFFER = NONE, RGBA_INTEGER / INT"},
                {ChildCase::NoneIntegerOnly, "COLOR0 = RG32I only, READ_BUFFER = NONE, RGBA_INTEGER / INT"},
            }};
            for (const auto& [c, label] : cases)
            {
                const ChildResult r = run_child(c, kTimeout);
                const std::string what = std::format("{}: child observes INVALID_OPERATION and exits normally (§4.3.2.1)", label);
                const bool pass = r.exited && r.exit_code == 0 && r.returned_from_call();
                // Only a crash inside glReadPixels is the documented deviation; any other outcome (wrong error,
                // crash during setup, timeout) is a plain failure on every driver.
                const bool known_crash = r.signaled && r.reached_call() && !r.returned_from_call();
                if (pass || known_crash) check.expect_or_known_deviation(pass, what, r.describe(), "Mesa", kReason);
                else check.expect(false, what, r.describe());
            }
        }
    };
}

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string_view(argv[1]) == kChildFlag)
    {
        bool known = false;
        for (const auto& n : kCases)
            if (n.name == argv[2])
            {
                g_child_case = n.id;
                known = true;
            }
        if (!known)
        {
            std::cerr << "unknown child case '" << argv[2] << "'\n";
            return 2;
        }
        std::array<char*, 3> child_argv = {argv[0], const_cast<char*>("--headless"), const_cast<char*>("--validate")};
        const demo::Info child_info{
            .name = "test-driver-crash-readpixels-none(child)",
            .title = "glReadPixels with READ_BUFFER = NONE: isolated child process",
            .min_es = {3, 0},
            .summary = "Internal child process of test-driver-crash-readpixels-none.",
            .source_dir = DEMO_SOURCE_DIR,
        };
        return demo::run<ChildProbe>(static_cast<int>(child_argv.size()), child_argv.data(), child_info);
    }

    const demo::Info info{
        .name = "test-driver-crash-readpixels-none",
        .title = "glReadPixels with READ_BUFFER = NONE (crash-isolated)",
        .min_es = {3, 0},
        .summary = "ES 3.0 §4.3.2.1 READ_BUFFER = NONE and missing-image errors; the integer-attachment cases that "
                   "crash Mesa run in a re-executed child process whose death is classified by the parent.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<ReadPixelsNone>(argc, argv, info);
}
