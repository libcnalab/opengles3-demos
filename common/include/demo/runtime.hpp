// Minimal runtime shared by every demo executable.
//
// The runtime owns only what is NOT the subject of the demos: command-line parsing, native
// context creation (headless EGL pbuffer or an SDL3 window), meta-gl initialization, the
// frame loop, and result reporting. Every OpenGL ES call that a demo is about stays inside
// the demo's own main.cpp, written directly against metagl::gl*.
//
// Lifecycle driven by demo::run():
//   1. parse CLI                         5. demo.init(env)        (may throw demo::Skip)
//   2. create native ES context          6. frames: demo.render(env, frame)
//   3. metagl::Initialize(loader)        7. --validate: demo.validate(env, check)
//   4. verify API kind + minimum version 8. demo.shutdown(env), final glGetError check
#pragma once

#include "demo/check.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace demo
{
    struct Version
    {
        int major = 0;
        int minor = 0;
        friend constexpr auto operator<=>(Version, Version) = default;
    };

    [[nodiscard]] std::string to_string(Version v);

    /// Static description of a demo, declared once in its main.cpp.
    struct Info
    {
        std::string_view name;        ///< executable / test name, e.g. "es30-hello-triangle"
        std::string_view title;       ///< human-readable title for --help and the window
        Version min_es{3, 0};         ///< minimum OpenGL ES version (SKIP_UNSUPPORTED below it)
        std::string_view summary;     ///< one or two sentences shown by --help
        std::string_view controls;    ///< interactive key bindings, shown by --help ("" = none)
        std::string_view source_dir;  ///< pass DEMO_SOURCE_DIR (defined by ogles_add_demo); shaders live here
        int default_frames = 1;       ///< frames rendered in --headless/--validate mode
        int depth_bits = 24;          ///< default framebuffer depth bits (0 = none)
        int stencil_bits = 8;         ///< default framebuffer stencil bits (0 = none)
        int samples = 0;              ///< default framebuffer MSAA samples (0 = single-sampled)
        bool debug_context = false;   ///< request a debug context (KHR_debug / ES 3.2 debug output)
        bool robust_context = false;  ///< request robust buffer access + reset notification
    };

    /// Parsed command-line options. Only options that the runtime can honour exist.
    struct Options
    {
        bool validate = false;     ///< run demo.validate() and report PASS/FAIL
        bool headless = false;     ///< EGL pbuffer, no window
        int frames = -1;           ///< -1 = until window closed (windowed) / default_frames (headless)
        int width = 256;
        int height = 256;
        std::string capture;       ///< write the final default-framebuffer image to this PNG
        std::string result_json;   ///< write the machine-readable result to this file
        std::string test_id;       ///< label stored in the result JSON (set by CTest)
        std::uint32_t seed = 1;    ///< seed of Env::rng
        bool debug = false;        ///< force a debug context
    };

    /// What the demo can see from the runtime while it runs.
    struct Env
    {
        const Info& info;
        const Options& options;
        Version context_version{};   ///< version reported by the context (glGetIntegerv)
        int width = 0;               ///< current default framebuffer width
        int height = 0;              ///< current default framebuffer height
        bool interactive = false;    ///< windowed and not --validate / --frames
        std::mt19937 rng;            ///< deterministic (seeded with --seed)
        std::string source_dir;      ///< demo source directory (Info::source_dir)
        /// The GetProcAddress-style loader the runtime passed to metagl::Initialize (needed only by
        /// tests of meta-gl's reload path, e.g. metagl::RestoreCurrentContext).
        void* (*gl_loader)(const char*) = nullptr;

        /// Absolute path of a file in the demo source directory, e.g. "shaders/triangle.vert".
        [[nodiscard]] std::string asset_path(std::string_view relative) const;
        /// Reads a text asset (shader source). Throws std::runtime_error when missing.
        [[nodiscard]] std::string load_text(std::string_view relative) const;
    };

    /// Throw from init() when the context lacks something the demo needs.
    struct Skip : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    /// Throw when a meta-gl limitation prevents a correct implementation.
    /// Every use must have a matching entry in docs/META_GL_GAPS.md.
    struct BlockedByMetaGl : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    /// Throw when a fatal setup step fails (shader compile error, incomplete FBO, ...).
    struct Failure : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    /// Interface implemented by each demo. Intentionally tiny: no scene graph, no renderer.
    class Demo
    {
    public:
        virtual ~Demo() = default;
        virtual void init(Env& env) = 0;
        virtual void render(Env& env, int frame) = 0;
        /// Runs after the last frame when --validate is given. Must record at least one check.
        virtual void validate(Env& env, Check& check) = 0;
        virtual void shutdown(Env& env) = 0;
        /// Optional interactive input (ASCII key, lower-case).
        virtual void on_key(Env& /*env*/, char /*key*/) {}
    };

    enum class Outcome
    {
        Pass,
        Fail,
        SkipUnsupported,
        BlockedEnvironment,
        BlockedMetaGl
    };

    [[nodiscard]] std::string_view to_string(Outcome outcome);

    /// Process exit code for CTest: 0 = PASS, 1 = FAIL, 77 = any skip/blocked state.
    inline constexpr int kExitSkip = 77;

    using DemoFactory = std::unique_ptr<Demo> (*)();

    /// Runs a demo and returns the process exit code.
    int run(int argc, char** argv, const Info& info, DemoFactory factory);

    template <typename T>
    int run(int argc, char** argv, const Info& info)
    {
        return run(argc, argv, info, +[]() -> std::unique_ptr<Demo> { return std::make_unique<T>(); });
    }
}
