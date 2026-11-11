#include "demo/runtime.hpp"

#include "demo/gl_util.hpp"
#include "demo/image.hpp"
#include "platform.hpp"

#include <metagl/metagl.hpp>

#include <charconv>
#include <cstdio>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace demo
{
    std::string to_string(Version v)
    {
        return std::format("{}.{}", v.major, v.minor);
    }

    std::string_view to_string(Outcome outcome)
    {
        switch (outcome)
        {
        case Outcome::Pass: return "PASS";
        case Outcome::Fail: return "FAIL";
        case Outcome::SkipUnsupported: return "SKIP_UNSUPPORTED";
        case Outcome::BlockedEnvironment: return "BLOCKED_ENVIRONMENT";
        case Outcome::BlockedMetaGl: return "BLOCKED_META_GL";
        }
        return "FAIL";
    }

    std::string Env::asset_path(std::string_view relative) const
    {
        return std::format("{}/{}", source_dir, relative);
    }

    std::string Env::load_text(std::string_view relative) const
    {
        const std::string path = asset_path(relative);
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open asset: " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    namespace
    {
        struct ParseResult
        {
            bool ok = true;
            bool help = false;
            bool force_window = false;
            std::string error;
        };

        bool parse_int(std::string_view s, int& out)
        {
            const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
            return ec == std::errc{} && p == s.data() + s.size();
        }

        ParseResult parse(int argc, char** argv, Options& o)
        {
            ParseResult r;
            for (int i = 1; i < argc; ++i)
            {
                const std::string_view a = argv[i];
                auto value = [&](std::string_view& out) {
                    if (i + 1 >= argc)
                    {
                        r.ok = false;
                        r.error = std::format("{} needs a value", a);
                        return false;
                    }
                    out = argv[++i];
                    return true;
                };
                std::string_view v;
                if (a == "--help" || a == "-h") r.help = true;
                else if (a == "--validate") o.validate = true;
                else if (a == "--headless") o.headless = true;
                else if (a == "--window") r.force_window = true;
                else if (a == "--debug") o.debug = true;
                else if (a == "--frames" || a == "--width" || a == "--height" || a == "--seed")
                {
                    if (!value(v)) break;
                    int n = 0;
                    if (!parse_int(v, n) || n <= 0)
                    {
                        r.ok = false;
                        r.error = std::format("{} expects a positive integer, got '{}'", a, v);
                        break;
                    }
                    if (a == "--frames") o.frames = n;
                    else if (a == "--width") o.width = n;
                    else if (a == "--height") o.height = n;
                    else o.seed = static_cast<std::uint32_t>(n);
                }
                else if (a == "--capture") { if (!value(v)) break; o.capture = v; }
                else if (a == "--result-json") { if (!value(v)) break; o.result_json = v; }
                else if (a == "--test-id") { if (!value(v)) break; o.test_id = v; }
                else
                {
                    r.ok = false;
                    r.error = std::format("unknown option '{}'", a);
                    break;
                }
            }
            return r;
        }

        void print_help(const Info& info)
        {
            std::cout << std::format(
                "{} - {}\n\n{}\n\nMinimum OpenGL ES version: {}\n\n"
                "Options:\n"
                "  --help              show this text\n"
                "  --validate          run the demo's correctness checks and report PASS/FAIL\n"
                "  --headless          render into an EGL pbuffer (no window)\n"
                "  --window            force a window even with --validate{}\n"
                "  --frames N          render N frames, then exit (deterministic time: frame/60 s)\n"
                "  --width N           default framebuffer width  (default 256)\n"
                "  --height N          default framebuffer height (default 256)\n"
                "  --capture FILE.png  save the final default framebuffer as PNG\n"
                "  --seed N            seed for the demo's random number generator (default 1)\n"
                "  --debug             request a debug context\n"
                "  --result-json FILE  write the machine-readable result (used by CTest)\n\n"
                "Exit status: 0 PASS, 1 FAIL, 77 SKIP_UNSUPPORTED / BLOCKED_*, 2 usage error.\n",
                info.name, info.title, info.summary, to_string(info.min_es),
#if OGLES3_HAVE_SDL3
                ""
#else
                " (unavailable: built without SDL3)"
#endif
            );
            if (!info.controls.empty()) std::cout << "\nInteractive controls:\n" << info.controls << "\n";
        }

        std::string json_escape(std::string_view s)
        {
            std::string out;
            out.reserve(s.size() + 8);
            for (const char c : s)
            {
                switch (c)
                {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) out += std::format("\\u{:04x}", static_cast<unsigned>(c));
                    else out += c;
                }
            }
            return out;
        }

        struct Report
        {
            Outcome outcome = Outcome::Fail;
            std::string reason;
            std::string gl_version, gl_renderer, gl_vendor, glsl_version, platform;
            Version context_version{};
            std::string capture;
        };

        void write_json(const Options& o, const Info& info, const Report& rep, const Check& runtime_checks,
                        const Check& demo_checks)
        {
            if (o.result_json.empty()) return;
            std::ofstream f(o.result_json);
            if (!f)
            {
                std::cerr << "warning: cannot write " << o.result_json << "\n";
                return;
            }
            auto checks = [&](const Check& c) {
                std::string s = "[";
                bool first = true;
                for (const auto& r : c.records())
                {
                    s += std::format("{}\n    {{\"ok\": {}, \"what\": \"{}\", \"detail\": \"{}\"}}", first ? "" : ",",
                                     r.ok ? "true" : "false", json_escape(r.what), json_escape(r.detail));
                    first = false;
                }
                return s + "]";
            };
            f << std::format(
                "{{\n  \"test_id\": \"{}\",\n  \"demo\": \"{}\",\n  \"min_es\": \"{}\",\n  \"outcome\": \"{}\",\n"
                "  \"reason\": \"{}\",\n  \"validated\": {},\n  \"context_version\": \"{}\",\n"
                "  \"gl_version\": \"{}\",\n  \"gl_renderer\": \"{}\",\n  \"gl_vendor\": \"{}\",\n"
                "  \"glsl_version\": \"{}\",\n  \"platform\": \"{}\",\n  \"metagl_version\": \"{}\",\n"
                "  \"capture\": \"{}\",\n  \"checks_passed\": {},\n  \"checks_failed\": {},\n"
                "  \"runtime_checks\": {},\n  \"demo_checks\": {}\n}}\n",
                json_escape(o.test_id), json_escape(info.name), to_string(info.min_es), to_string(rep.outcome),
                json_escape(rep.reason), o.validate ? "true" : "false", to_string(rep.context_version),
                json_escape(rep.gl_version), json_escape(rep.gl_renderer), json_escape(rep.gl_vendor),
                json_escape(rep.glsl_version), json_escape(rep.platform), metagl::GetVersionString(),
                json_escape(rep.capture), demo_checks.passed(), demo_checks.failed(), checks(runtime_checks),
                checks(demo_checks));
        }

        std::string gl_string(metagl::StringName name)
        {
            const GLubyte* s = metagl::glGetString(name);
            return s != nullptr ? reinterpret_cast<const char*>(s) : "";
        }

        int finish(const Options& o, const Info& info, Report& rep, const Check& rc, const Check& dc)
        {
            std::cout << std::format("RESULT: {} demo={} es={} checks={}/{}{}{}\n", to_string(rep.outcome), info.name,
                                     to_string(rep.context_version), dc.passed(), dc.total(),
                                     rep.reason.empty() ? "" : " reason=", rep.reason);
            write_json(o, info, rep, rc, dc);
            switch (rep.outcome)
            {
            case Outcome::Pass: return 0;
            case Outcome::Fail: return 1;
            default: return kExitSkip;
            }
        }
    }

    int run(int argc, char** argv, const Info& info, DemoFactory factory)
    {
        Options options;
        const ParseResult pr = parse(argc, argv, options);
        if (!pr.ok)
        {
            std::cerr << info.name << ": " << pr.error << " (see --help)\n";
            return 2;
        }
        if (pr.help)
        {
            print_help(info);
            return 0;
        }

        Report rep;
        Check runtime_checks;
        Check demo_checks;

        // --- 1. native context --------------------------------------------------------------
        bool windowed = !options.headless;
#if !OGLES3_HAVE_SDL3
        if (windowed)
        {
            if (pr.force_window)
            {
                rep.outcome = Outcome::BlockedEnvironment;
                rep.reason = "built without SDL3; --window unavailable";
                return finish(options, info, rep, runtime_checks, demo_checks);
            }
            std::cout << "note: built without SDL3, running headless\n";
            windowed = false;
        }
#endif
        std::unique_ptr<detail::Platform> platform;
#if OGLES3_HAVE_SDL3
        if (windowed) platform = detail::make_sdl3_platform();
#endif
        if (!platform) platform = detail::make_egl_headless_platform();

        detail::ContextRequest req;
        req.version = info.min_es;
        req.width = options.width;
        req.height = options.height;
        req.depth_bits = info.depth_bits;
        req.stencil_bits = info.stencil_bits;
        req.samples = info.samples;
        req.debug = info.debug_context || options.debug;
        req.robust = info.robust_context;
        req.title = std::format("{} - {}", info.name, info.title);

        std::string error;
        if (!platform->create(req, error))
        {
            rep.outcome = Outcome::BlockedEnvironment;
            rep.reason = error;
            return finish(options, info, rep, runtime_checks, demo_checks);
        }
        rep.platform = platform->description();

        // --- 2. meta-gl -----------------------------------------------------------------------
        // The context is current; only now may meta-gl load its function table.
        if (!metagl::Initialize(platform->loader()))
        {
            rep.outcome = Outcome::BlockedEnvironment;
            rep.reason = "metagl::Initialize failed (unsupported GL_VERSION or a required entry point is missing)";
            return finish(options, info, rep, runtime_checks, demo_checks);
        }

        rep.gl_version = gl_string(metagl::StringName::Version);
        rep.gl_renderer = gl_string(metagl::StringName::Renderer);
        rep.gl_vendor = gl_string(metagl::StringName::Vendor);
        rep.glsl_version = gl_string(metagl::StringName::ShadingLanguageVersion);

        const metagl::ContextInfo ci = metagl::GetContextInfo();
        GLint gl_major = 0, gl_minor = 0;
        metagl::glGetIntegerv(metagl::GetParameter::MajorVersion, &gl_major);
        metagl::glGetIntegerv(metagl::GetParameter::MinorVersion, &gl_minor);
        rep.context_version = {gl_major, gl_minor};

        std::cout << std::format("== {} - {}\n   GL:       {}\n   renderer: {}\n   GLSL:     {}\n   platform: {}\n"
                                 "   meta-gl:  {}\n",
                                 info.name, info.title, rep.gl_version, rep.gl_renderer, rep.glsl_version,
                                 rep.platform, metagl::GetVersionString());

        // meta-gl integration checks: its parsed context identity must agree with the driver.
        runtime_checks.expect(ci.api == metagl::ApiKind::OpenGLES, "meta-gl reports an OpenGL ES context",
                              ci.api == metagl::ApiKind::OpenGLES ? "" : "ApiKind is not OpenGLES");
        runtime_checks.expect(ci.major == gl_major && ci.minor == gl_minor,
                              "meta-gl ContextInfo version matches GL_MAJOR/MINOR_VERSION",
                              std::format("ContextInfo={}.{} driver={}.{}", ci.major, ci.minor, gl_major, gl_minor));

        if (ci.api != metagl::ApiKind::OpenGLES)
        {
            rep.outcome = Outcome::BlockedEnvironment;
            rep.reason = "context is not OpenGL ES";
            return finish(options, info, rep, runtime_checks, demo_checks);
        }
        if (rep.context_version < info.min_es)
        {
            rep.outcome = Outcome::SkipUnsupported;
            rep.reason = std::format("requires OpenGL ES {}, context is {}", to_string(info.min_es),
                                     to_string(rep.context_version));
            return finish(options, info, rep, runtime_checks, demo_checks);
        }

        // --- 3. demo ---------------------------------------------------------------------------
        Env env{info, options};
        env.source_dir = std::string(info.source_dir);
        env.gl_loader = platform->loader();
        env.context_version = rep.context_version;
        env.interactive = windowed && !options.validate && options.frames < 0;
        env.rng.seed(options.seed);
        {
            GLint vp[4] = {};
            metagl::glGetIntegerv(metagl::GetParameter::Viewport, vp);  // initial viewport = surface size
            env.width = vp[2];
            env.height = vp[3];
        }

        std::unique_ptr<Demo> demo = factory();
        bool initialized = false;
        try
        {
            demo->init(env);
            initialized = true;
            runtime_checks.expect_no_gl_error("no GL error after init()");

            const int frames = options.frames > 0 ? options.frames
                               : env.interactive  ? -1
                                                  : info.default_frames;
            for (int frame = 0; frames < 0 || frame < frames; ++frame)
            {
                const detail::Events ev = platform->poll();
                if (ev.quit) break;
                if (ev.resized)
                {
                    env.width = ev.width;
                    env.height = ev.height;
                }
                for (const char k : ev.keys) demo->on_key(env, k);

                demo->render(env, frame);

                const bool last = frames > 0 && frame == frames - 1;
                if (last)
                {
                    runtime_checks.expect_no_gl_error("no GL error after the last frame");
                    if (options.validate) demo->validate(env, demo_checks);
                    if (!options.capture.empty())
                    {
                        // The default framebuffer is read before the swap (its back buffer
                        // contents are undefined afterwards).
                        metagl::glBindFramebuffer(metagl::FramebufferTarget::ReadFramebuffer, metagl::FramebufferId{0});
                        const auto pixels = read_rgba8(0, 0, env.width, env.height);
                        if (write_png_rgba8(options.capture, env.width, env.height, pixels, true))
                            rep.capture = options.capture;
                    }
                }
                platform->swap();
            }

            demo->shutdown(env);
            initialized = false;
            runtime_checks.expect_no_gl_error("no GL error after shutdown()");
        }
        catch (const Skip& e)
        {
            rep.outcome = Outcome::SkipUnsupported;
            rep.reason = e.what();
        }
        catch (const BlockedByMetaGl& e)
        {
            rep.outcome = Outcome::BlockedMetaGl;
            rep.reason = e.what();
        }
        catch (const std::exception& e)
        {
            rep.outcome = Outcome::Fail;
            rep.reason = e.what();
        }
        if (initialized)
        {
            try
            {
                demo->shutdown(env);
            }
            catch (...)
            {
            }
        }

        if (rep.reason.empty())
        {
            if (runtime_checks.failed() > 0)
            {
                rep.outcome = Outcome::Fail;
                rep.reason = "runtime checks failed";
            }
            else if (options.validate && demo_checks.total() == 0)
            {
                rep.outcome = Outcome::Fail;
                rep.reason = "validate() recorded no checks";
            }
            else if (demo_checks.failed() > 0)
            {
                rep.outcome = Outcome::Fail;
                rep.reason = std::format("{} check(s) failed", demo_checks.failed());
            }
            else
            {
                rep.outcome = Outcome::Pass;
                if (!options.validate) rep.reason = "not validated (run with --validate)";
            }
        }
        return finish(options, info, rep, runtime_checks, demo_checks);
    }
}
