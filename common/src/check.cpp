#include "demo/check.hpp"

#include "demo/gl_util.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace demo
{
    std::string format_rgba(Rgba8 c)
    {
        return std::format("({},{},{},{})", c[0], c[1], c[2], c[3]);
    }

    bool Check::expect(bool condition, std::string_view what, std::string_view detail)
    {
        records_.push_back({condition, std::string(what), std::string(detail)});
        if (condition) ++passed_;
        else ++failed_;
        std::cout << std::format("   [{}] {}{}{}\n", condition ? "ok" : "FAIL", what, detail.empty() ? "" : " - ",
                                 detail);
        return condition;
    }

    bool Check::expect_near(double actual, double expected, double tolerance, std::string_view what)
    {
        const bool ok = std::abs(actual - expected) <= tolerance;
        return expect(ok, what, std::format("actual={} expected={} tol={}", actual, expected, tolerance));
    }

    bool Check::expect_or_known_deviation(bool ok, std::string_view what, std::string_view detail,
                                          std::string_view renderer_substring, std::string_view reason)
    {
        auto text = [](metagl::StringName n) {
            const GLubyte* p = metagl::glGetString(n);
            return std::string(p != nullptr ? reinterpret_cast<const char*>(p) : "");
        };
        if (!deviation_matches(renderer_substring, text(metagl::StringName::Renderer), text(metagl::StringName::Version)))
            return expect(ok, what, detail);
        if (!ok) ++xfail_;
        return expect(true, what,
                      std::format("{} on '{}': {}{}{}", ok ? "XPASS (driver now conforms)" : "XFAIL known driver deviation",
                                  renderer_substring, reason, detail.empty() ? "" : " | ", detail));
    }

    bool deviation_matches(std::string_view tag, std::string_view gl_renderer, std::string_view gl_version)
    {
        if (tag.empty()) return false;
        if (gl_version.find("ANGLE") != std::string_view::npos) return gl_version.find(tag) != std::string_view::npos;
        return gl_renderer.find(tag) != std::string_view::npos || gl_version.find(tag) != std::string_view::npos;
    }

    bool Check::expect_no_gl_error(std::string_view what)
    {
        const metagl::ErrorCode e = drain_gl_errors();
        return expect(e == metagl::ErrorCode::NoError, what,
                      e == metagl::ErrorCode::NoError ? "" : std::format("glGetError() = {}", error_name(e)));
    }

    bool Check::expect_gl_error(metagl::ErrorCode expected, std::string_view what)
    {
        const metagl::ErrorCode e = drain_gl_errors();
        return expect(e == expected, what,
                      std::format("glGetError() = {}, expected {}", error_name(e), error_name(expected)));
    }

    bool Check::expect_pixel(int x, int y, Rgba8 expected, int tolerance, std::string_view what)
    {
        const Rgba8 got = read_pixel(x, y);
        bool ok = true;
        for (std::size_t i = 0; i < 4; ++i)
            if (std::abs(int{got[i]} - int{expected[i]}) > tolerance) ok = false;
        return expect(ok, what,
                      std::format("pixel({},{}) = {} expected {} tol {}", x, y, format_rgba(got),
                                  format_rgba(expected), tolerance));
    }
}
