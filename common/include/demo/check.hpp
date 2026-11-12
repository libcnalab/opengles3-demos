// Records validation checks. A demo passes only if it recorded at least one check and none
// failed - an empty validate() is reported as FAIL, never as a vacuous success.
#pragma once

#include <metagl/metagl.hpp>

#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace demo
{
    using Rgba8 = std::array<std::uint8_t, 4>;

    struct CheckRecord
    {
        bool ok = false;
        std::string what;
        std::string detail;
    };

    class Check
    {
    public:
        /// Records a boolean condition.
        bool expect(bool condition, std::string_view what, std::string_view detail = {});

        template <typename A, typename B>
        bool expect_eq(const A& actual, const B& expected, std::string_view what)
        {
            const bool ok = actual == expected;
            return expect(ok, what, ok ? std::string{} : std::format("actual={} expected={}", actual, expected));
        }

        bool expect_near(double actual, double expected, double tolerance, std::string_view what);

        /// Strict check with a *documented* driver deviation: when `ok` is false and the current driver matches
        /// `renderer_substring` (demo::deviation_matches: e.g. "radeonsi" for one driver, "Mesa" for all Mesa drivers, never ANGLE), the result is recorded as XFAIL (passing, but labelled) with `reason`, which
        /// must also be listed in docs/KNOWN_LIMITATIONS.md. Every other driver gets the strict check. If
        /// the named driver starts conforming, the check passes and is labelled XPASS.
        bool expect_or_known_deviation(bool ok, std::string_view what, std::string_view detail,
                                       std::string_view renderer_substring, std::string_view reason);

        [[nodiscard]] int known_deviations() const noexcept { return xfail_; }

        /// Drains the GL error queue; passes when it was empty.
        bool expect_no_gl_error(std::string_view what);

        /// Passes when the first queued error equals `expected` (queue is drained).
        bool expect_gl_error(metagl::ErrorCode expected, std::string_view what);

        /// Reads one RGBA8 pixel from the current READ framebuffer and compares per channel.
        bool expect_pixel(int x, int y, Rgba8 expected, int tolerance, std::string_view what);

        [[nodiscard]] int passed() const noexcept { return passed_; }
        [[nodiscard]] int failed() const noexcept { return failed_; }
        [[nodiscard]] int total() const noexcept { return passed_ + failed_; }
        [[nodiscard]] const std::vector<CheckRecord>& records() const noexcept { return records_; }

    private:
        int passed_ = 0;
        int failed_ = 0;
        int xfail_ = 0;
        std::vector<CheckRecord> records_;
    };

    [[nodiscard]] std::string format_rgba(Rgba8 c);

    /// The driver-selection rule behind Check::expect_or_known_deviation and the shader-test `@xfail` annotations.
    /// `tag` names the implementation a documented deviation applies to: "Mesa" (every Mesa driver), a Mesa driver
    /// ("radeonsi", "llvmpipe", "zink", "softpipe") or "ANGLE". It is searched in GL_RENDERER and GL_VERSION, with two
    /// exceptions:
    ///  - ANGLE (GL_VERSION contains "ANGLE") is matched on GL_VERSION only. Its GL_RENDERER names the Vulkan
    ///    backend ("ANGLE (Mesa, Vulkan ... (llvmpipe ...))"), and a Mesa deviation must not excuse ANGLE.
    ///  - An empty tag matches nothing: a missing driver name must give the strict check, never an XFAIL everywhere.
    /// "llvmpipe" also matches zink on lavapipe ("zink Vulkan ...(llvmpipe ...)"): the two share llvmpipe's
    /// rasterizer, and demos that must tell them apart pass "zink" first.
    [[nodiscard]] bool deviation_matches(std::string_view tag, std::string_view gl_renderer, std::string_view gl_version);
}
