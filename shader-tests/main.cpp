// glsl-compile-tests
//
// Data-driven GLSL ES compile/link tests. Every case is one or more shader files sharing a stem
// in shader-tests/glsl{100,300,310,320}/ (extension = stage: .vert .frag .comp .geom .tesc .tese).
// Annotations in comments (after the #version line) describe the case:
//
//   // @expect: compile-pass | compile-fail | link-pass | link-fail | build-fail
//   // @min-es: 3.1          (default from the directory: glsl100/glsl300 -> 3.0, glsl310 -> 3.1, glsl320 -> 3.2)
//   // @spec: GLSL ES 3.00 §4.5.4 ...
//   // @feature: GLSL300.PRECISION.DEFAULT     (repeatable; read by the coverage generator)
//   // @xfail: Mesa | <why>   known driver deviation: when GL_VERSION contains "Mesa" and the case
//                             does NOT meet its expectation, it is reported as XFAIL (documented in
//                             docs/KNOWN_LIMITATIONS.md) instead of FAIL; on every other driver the
//                             strict expectation applies. If the driver starts conforming: XPASS.
//
// Semantics:
//   compile-pass  every file compiles             compile-fail  at least one file fails to compile
//   link-pass     all compile and the program links
//   link-fail     all compile but linking fails   build-fail    compile OR link fails (the spec allows either)
//
// Version gating: when the context is natively below a case's @min-es, a *passing* case must instead
// be rejected (its #version or stage is unknown to the implementation) - e.g. softpipe, a native
// ES 3.1 driver, must refuse "#version 320 es". Under MESA_GLES_VERSION_OVERRIDE (API-version
// emulation only) the outcome is printed but not asserted; see docs/VALIDATION.md.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace mgl = metagl;
namespace fs = std::filesystem;

namespace
{
    struct ShaderFile
    {
        fs::path path;
        mgl::ShaderType type;
        std::string source;
    };

    struct Case
    {
        std::string name;  // "glsl300/missing-precision"
        std::vector<ShaderFile> files;
        std::string expect;
        demo::Version min_es;
        std::string spec;
        std::string xfail;  // "<GL_VERSION substring> | <reason>"
    };

    std::optional<mgl::ShaderType> stage_of(const fs::path& p)
    {
        const std::string ext = p.extension().string();
        if (ext == ".vert") return mgl::ShaderType::Vertex;
        if (ext == ".frag") return mgl::ShaderType::Fragment;
        if (ext == ".comp") return mgl::ShaderType::Compute;
        if (ext == ".geom") return mgl::ShaderType::Geometry;
        if (ext == ".tesc") return mgl::ShaderType::TessControl;
        if (ext == ".tese") return mgl::ShaderType::TessEvaluation;
        return std::nullopt;
    }

    std::string annotation(const std::string& src, std::string_view key)
    {
        const std::string tag = std::format("// @{}:", key);
        const auto pos = src.find(tag);
        if (pos == std::string::npos) return {};
        const auto start = pos + tag.size();
        const auto end = src.find('\n', start);
        std::string v = src.substr(start, end - start);
        v.erase(0, v.find_first_not_of(' '));
        v.erase(v.find_last_not_of(" \r") + 1);
        return v;
    }

    std::vector<Case> discover(const fs::path& root)
    {
        std::map<std::string, Case> by_name;
        for (const char* dir : {"glsl100", "glsl300", "glsl310", "glsl320"})
        {
            const fs::path d = root / dir;
            if (!fs::is_directory(d)) continue;
            const demo::Version dir_min = std::string_view(dir) == "glsl310"   ? demo::Version{3, 1}
                                          : std::string_view(dir) == "glsl320" ? demo::Version{3, 2}
                                                                               : demo::Version{3, 0};
            for (const auto& entry : fs::directory_iterator(d))
            {
                const auto stage = stage_of(entry.path());
                if (!stage) continue;
                std::ifstream in(entry.path(), std::ios::binary);
                std::stringstream ss;
                ss << in.rdbuf();
                const std::string name = std::format("{}/{}", dir, entry.path().stem().string());
                Case& c = by_name[name];
                c.name = name;
                c.min_es = dir_min;
                c.files.push_back({entry.path(), *stage, ss.str()});
            }
        }
        std::vector<Case> cases;
        for (auto& [name, c] : by_name)
        {
            std::sort(c.files.begin(), c.files.end(), [](const auto& a, const auto& b) { return a.path < b.path; });
            for (const auto& f : c.files)
            {
                if (c.expect.empty()) c.expect = annotation(f.source, "expect");
                if (c.spec.empty()) c.spec = annotation(f.source, "spec");
                if (c.xfail.empty()) c.xfail = annotation(f.source, "xfail");
                const std::string m = annotation(f.source, "min-es");
                if (m.size() == 3) c.min_es = {m[0] - '0', m[2] - '0'};
            }
            cases.push_back(std::move(c));
        }
        return cases;
    }

    struct Outcome
    {
        bool all_compiled = true;
        bool linked = false;
        std::string log;
    };

    Outcome build(const Case& c, bool link)
    {
        Outcome out;
        std::vector<mgl::ShaderId> shaders;
        for (const auto& f : c.files)
        {
            const mgl::ShaderId s = mgl::glCreateShader(f.type);
            if (s.value == 0)
            {
                // Stage unknown to this context (e.g. a compute shader on ES 3.0): INVALID_ENUM.
                (void)demo::drain_gl_errors();
                out.all_compiled = false;
                out.log += std::format("[{}] glCreateShader returned 0 (stage not supported)\n", f.path.filename().string());
                continue;
            }
            mgl::glShaderSource(s, f.source);
            mgl::glCompileShader(s);
            GLint ok = GL_FALSE;
            mgl::glGetShaderiv(s, mgl::ShaderParameter::CompileStatus, &ok);
            if (ok != GL_TRUE)
            {
                out.all_compiled = false;
                out.log += std::format("[{}] {}", f.path.filename().string(), demo::shader_info_log(s));
            }
            shaders.push_back(s);
        }
        if (link && out.all_compiled)
        {
            // A graphics program needs a vertex AND a fragment shader (ES 3.0 §2.12.3). If the case
            // supplies only one of them, add a minimal partner of the same #version so that the link
            // result reflects the property under test, not the missing stage. (Without this, a
            // fragment-only "build-fail" case would pass vacuously.)
            bool has_vs = false, has_fs = false, has_cs = false;
            for (const auto& f : c.files)
            {
                has_vs = has_vs || f.type == mgl::ShaderType::Vertex;
                has_fs = has_fs || f.type == mgl::ShaderType::Fragment;
                has_cs = has_cs || f.type == mgl::ShaderType::Compute;
            }
            const std::string version_line = c.files.front().source.substr(0, c.files.front().source.find('\n') + 1);
            auto add_partner = [&](mgl::ShaderType type, const std::string& body) {
                std::string log;
                const mgl::ShaderId s = demo::compile_shader(type, version_line + body, log);
                if (s.value != 0) shaders.push_back(s);
                else out.log += "[partner] " + log;
            };
            if (!has_cs && !has_vs) add_partner(mgl::ShaderType::Vertex, "void main() { gl_Position = vec4(0.0); }\n");
            if (!has_cs && !has_fs)
                add_partner(mgl::ShaderType::Fragment, "precision mediump float;\nout vec4 partner_out;\nvoid main() { partner_out = vec4(1.0); }\n");

            const mgl::ProgramId p = mgl::glCreateProgram();
            for (const auto s : shaders) mgl::glAttachShader(p, s);
            mgl::glLinkProgram(p);
            GLint ok = GL_FALSE;
            mgl::glGetProgramiv(p, mgl::ProgramParameter::LinkStatus, &ok);
            out.linked = ok == GL_TRUE;
            if (!out.linked) out.log += "[link] " + demo::program_info_log(p);
            mgl::glDeleteProgram(p);
        }
        for (const auto s : shaders) mgl::glDeleteShader(s);
        return out;
    }

    std::string first_line(const std::string& s)
    {
        return s.substr(0, std::min(s.find('\n'), std::size_t{160}));
    }

    class GlslCompileTests final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            cases_ = discover(env.source_dir);
            const GLubyte* v = mgl::glGetString(mgl::StringName::Version);
            version_string_ = v != nullptr ? reinterpret_cast<const char*>(v) : "";
            version_override_ = std::getenv("MESA_GLES_VERSION_OVERRIDE") != nullptr;
            if (cases_.empty()) throw demo::Failure("no shader test cases found under " + env.source_dir);
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            int run = 0, gated = 0;
            for (const auto& c : cases_)
            {
                static const std::vector<std::string> kinds = {"compile-pass", "compile-fail", "link-pass", "link-fail", "build-fail"};
                if (std::find(kinds.begin(), kinds.end(), c.expect) == kinds.end())
                {
                    check.expect(false, std::format("{}: valid @expect annotation", c.name), "got '" + c.expect + "'");
                    continue;
                }
                const bool wants_link = c.expect != "compile-pass" && c.expect != "compile-fail";
                const Outcome o = build(c, wants_link);
                (void)demo::drain_gl_errors();  // a failed glShaderSource on name 0, etc.

                if (env.context_version < c.min_es)
                {
                    // Only cases expected to succeed carry information on an older context.
                    if (c.expect == "compile-pass" || c.expect == "link-pass")
                    {
                        ++gated;
                        if (version_override_)
                        {
                            // MESA_GLES_VERSION_OVERRIDE lowers the *API* version only; radeonsi and
                            // llvmpipe keep accepting newer #version lines. Not a spec statement about
                            // a real ES 3.0/3.1 implementation, so it is reported, not asserted.
                            std::cout << std::format("     note: {} {} on an overridden ES {} context\n", c.name,
                                                     o.all_compiled ? "still compiles" : "is rejected",
                                                     demo::to_string(env.context_version));
                            continue;
                        }
                        check.expect(!o.all_compiled,
                                     std::format("{}: rejected by an ES {} context (needs {})", c.name,
                                                 demo::to_string(env.context_version), demo::to_string(c.min_es)),
                                     o.all_compiled ? "compiled although the context is too old" : "");
                    }
                    continue;
                }
                ++run;
                bool ok = false;
                if (c.expect == "compile-pass") ok = o.all_compiled;
                else if (c.expect == "compile-fail") ok = !o.all_compiled;
                else if (c.expect == "link-pass") ok = o.all_compiled && o.linked;
                else if (c.expect == "link-fail") ok = o.all_compiled && !o.linked;
                else ok = !o.all_compiled || !o.linked;  // build-fail
                std::string detail = c.spec;
                if (!c.xfail.empty())
                {
                    const std::string driver = c.xfail.substr(0, c.xfail.find('|'));
                    const std::string needle = driver.substr(0, driver.find_last_not_of(' ') + 1);
                    if (version_string_.find(needle) != std::string::npos)
                    {
                        ++xfail_cases_;
                        detail = std::format("{} ({}) | {}", ok ? "XPASS: driver now conforms" : "XFAIL: known driver deviation",
                                             c.xfail, c.spec);
                        ok = true;
                    }
                }
                if (!o.log.empty()) detail += " | log: " + first_line(o.log);
                if (!ok && c.expect.ends_with("pass")) detail += " | full log:\n" + o.log;
                check.expect(ok, std::format("{}: {}", c.name, c.expect), detail);
            }
            std::cout << std::format("     {} cases: {} evaluated, {} version-gated, {} known driver deviations\n",
                                     cases_.size(), run, gated, xfail_cases_);
            check.expect(run > 0, "at least one case applies to this context");
        }

        void shutdown(demo::Env& /*env*/) override {}

    private:
        std::vector<Case> cases_;
        std::string version_string_;
        int xfail_cases_ = 0;
        bool version_override_ = false;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "glsl-compile-tests",
        .title = "GLSL ES compile and link tests",
        .min_es = {3, 0},
        .summary = "Positive and negative GLSL ES 1.00/3.00/3.10/3.20 compile and link cases; on older "
                   "contexts newer-version shaders must be rejected.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<GlslCompileTests>(argc, argv, info);
}
