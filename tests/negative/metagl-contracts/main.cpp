// test-metagl-contracts
//
// meta-gl's documented contracts that cannot be tested by ordinary GL error checks:
//
// 1. Invalid-input contract (README "Invalid Input Contract", Functions.hpp R07/R11): violations
//    terminate the process with std::terminate() *before* reaching the driver, in Release builds too.
//    Each violation runs in a forked child; the parent requires death by SIGABRT. A control child
//    that performs no violation must exit normally, proving the harness can tell the difference.
//    (No GL call is made in a child: every case is rejected by meta-gl before the driver.)
// 2. Context-loss life cycle (Context.hpp, ContextEvents.hpp, Loader.hpp): NotifyContextLost hides
//    the function table and capabilities, RestoreCurrentContext reloads, bumps the generation and
//    notifies listeners exactly once, after which rendering works again.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <csignal>
#include <cstdio>
#include <format>
#include <functional>
#include <vector>

namespace mgl = metagl;

namespace
{
    enum class ChildEnd
    {
        Aborted,
        ExitedZero,
        Other
    };

    ChildEnd run_in_child(const std::function<void()>& body)
    {
        std::fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0)
        {
            // Child: silence the "terminate called ..." message, run the body, and report a normal
            // exit if it returns. _exit avoids running the parent's atexit handlers / destructors.
            std::FILE* devnull = std::freopen("/dev/null", "w", stderr);
            (void)devnull;
            std::signal(SIGABRT, SIG_DFL);
            body();
            _exit(0);
        }
        int status = 0;
        if (pid < 0 || waitpid(pid, &status, 0) != pid) return ChildEnd::Other;
        if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) return ChildEnd::Aborted;
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return ChildEnd::ExitedZero;
        return ChildEnd::Other;
    }

    struct CountingListener final : mgl::ContextListener
    {
        int lost = 0;
        int restored = 0;
        mgl::ContextStatus status_during_restore = mgl::ContextStatus::NotCreated;
        void OnContextLost() override { ++lost; }
        void OnContextRestored() override
        {
            ++restored;
            status_during_restore = mgl::GetContextStatus();
        }
    };

    class MetaGlContracts final : public demo::Demo
    {
    public:
        void init(demo::Env& /*env*/) override {}

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            check_termination_contracts(check);
            check_row_major_upload(check);
            check_context_loss_cycle(env, check);
        }

        void shutdown(demo::Env& /*env*/) override {}

    private:
        static void check_termination_contracts(demo::Check& check)
        {
            const mgl::UniformLocation loc{0};
            const struct
            {
                const char* what;
                std::function<void()> body;
                ChildEnd expected;
            } cases[] = {
                {"control: no violation -> normal exit", [] {}, ChildEnd::ExitedZero},
                {"glUniformMatrix<4> with 15 floats (not a whole matrix) terminates",
                 [loc] { mgl::glUniformMatrix<4>(loc, std::vector<float>(15)); }, ChildEnd::Aborted},
                {"glUniform<3> with 4 floats (not whole vec3s) terminates",
                 [loc] { mgl::glUniform<3>(loc, std::vector<float>(4)); }, ChildEnd::Aborted},
                {"any wrapper after MarkContextLost terminates instead of calling a stale pointer",
                 [] {
                     mgl::MarkContextLost();
                     mgl::glClear(mgl::ClearBufferBit::Color);
                 },
                 ChildEnd::Aborted},
            };
            for (const auto& c : cases)
            {
                const ChildEnd end = run_in_child(c.body);
                check.expect(end == c.expected, c.what,
                             end == ChildEnd::Aborted      ? "child died with SIGABRT"
                             : end == ChildEnd::ExitedZero ? "child exited normally"
                                                           : "child ended otherwise");
            }
        }

        // ES 3.0+ accepts transpose = GL_TRUE (row-major data, ES 3.0 §2.12.6.1). meta-gl's helper used to
        // terminate on it (gap G8, fixed); a row-major upload must read back as the transposed matrix.
        static void check_row_major_upload(demo::Check& check)
        {
            const mgl::ProgramId p = demo::build_program_from_source(
                "#version 300 es\nuniform mat4 u_m;\nvoid main() { gl_Position = u_m[0] + u_m[1] + u_m[2] + u_m[3]; }\n",
                "#version 300 es\nprecision mediump float;\nout vec4 o;\nvoid main() { o = vec4(1.0); }\n", "row-major");
            mgl::glUseProgram(p);
            const mgl::UniformLocation loc = mgl::glGetUniformLocation(p, "u_m");
            const std::array<float, 16> row_major = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            mgl::glUniformMatrix<4>(loc, row_major, GL_TRUE);  // must not terminate
            check.expect_no_gl_error("glUniformMatrix<4>(..., GL_TRUE) is legal on an ES 3.x context");
            std::array<float, 16> back{};
            mgl::glGetUniformfv(p, loc, back.data());
            bool transposed = true;
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    transposed = transposed && back[static_cast<std::size_t>(c * 4 + r)] == row_major[static_cast<std::size_t>(r * 4 + c)];
            check.expect(transposed, "a row-major upload (transpose = GL_TRUE) reads back as the column-major transpose");
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteProgram(p);
        }

        static void check_context_loss_cycle(demo::Env& env, demo::Check& check)
        {
            CountingListener listener;
            mgl::AddContextListener(&listener);
            const std::uint64_t generation = mgl::GetContextGeneration();
            const std::string version = mgl::GetCapabilities().version_string;

            check.expect(!mgl::Initialize(nullptr) && mgl::IsInitialized(),
                         "Initialize(nullptr) fails without disturbing the current context");

            mgl::NotifyContextLost();
            check.expect(listener.lost == 1 && listener.restored == 0, "NotifyContextLost calls OnContextLost once");
            check.expect(mgl::IsContextLost() && mgl::GetContextStatus() == mgl::ContextStatus::Lost,
                         "status is Lost after NotifyContextLost");
            check.expect(!mgl::IsInitialized() && !mgl::IsFunctionAvailable("glClear"),
                         "after loss no entry point is reported available");
            check.expect(mgl::GetCapabilities().version_string.empty() && !mgl::SupportsGLES30(),
                         "after loss the capabilities are cleared");
            check.expect_eq(mgl::GetContextGeneration(), generation, "context generation is preserved across loss");

            // The EGL context was never really lost, so reloading it must succeed.
            const bool restored = mgl::RestoreCurrentContext(env.gl_loader);
            check.expect(restored, "RestoreCurrentContext reloads the current context");
            check.expect(listener.restored == 1 && listener.lost == 1, "OnContextRestored called exactly once");
            check.expect(listener.status_during_restore == mgl::ContextStatus::Restored,
                         "listeners observe ContextStatus::Restored during the callback");
            check.expect(mgl::GetContextStatus() == mgl::ContextStatus::Current && mgl::IsInitialized(),
                         "status returns to Current after restore");
            check.expect_eq(mgl::GetContextGeneration(), generation + 1, "restore increments the context generation");
            check.expect(mgl::GetCapabilities().version_string == version, "capabilities are re-detected after restore");
            mgl::RemoveContextListener(&listener);

            // Rendering works again through the reloaded table.
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            check.expect_pixel(env.width / 2, env.height / 2, {64, 128, 191, 255}, 1, "glClear works after restore");
        }
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "test-metagl-contracts",
        .title = "meta-gl contracts: termination on invalid input, context-loss cycle",
        .min_es = {3, 0},
        .summary = "Death tests (forked children must die with SIGABRT) for meta-gl's invalid-input contract and "
                   "a full lost/restored context cycle with listeners and generation counter.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<MetaGlContracts>(argc, argv, info);
}
