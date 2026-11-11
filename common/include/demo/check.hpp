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
        std::vector<CheckRecord> records_;
    };

    [[nodiscard]] std::string format_rgba(Rgba8 c);
}
