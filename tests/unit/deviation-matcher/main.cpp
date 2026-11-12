// unit-deviation-matcher: CPU-only regression test for demo::deviation_matches, the rule that decides which driver
// a documented deviation (XFAIL) applies to. No GL context is created.
//
// Regression (2026-11-12 review): ANGLE's GL_RENDERER names its Vulkan backend ("ANGLE (Mesa, Vulkan ... (llvmpipe
// ...))"), so plain substring matching let every "Mesa" or "llvmpipe" deviation excuse ANGLE as well. The strings
// below are copied from real result files of the reference machine (Mesa 25.0.7, ANGLE 2.1.17841).

#include <demo/check.hpp>

#include <cstdio>
#include <string_view>

namespace
{
    struct Driver
    {
        std::string_view name, renderer, version;
    };

    constexpr Driver kRadeonsi{"radeonsi", "AMD Radeon 780M (radeonsi, phoenix, LLVM 19.1.7, DRM 3.61, 6.12.111+deb13-amd64)",
                               "OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1"};
    constexpr Driver kLlvmpipe{"llvmpipe", "llvmpipe (LLVM 19.1.7, 256 bits)", "OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1"};
    constexpr Driver kSoftpipe{"softpipe", "softpipe", "OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1"};
    constexpr Driver kZink{"zink", "zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE))",
                           "OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1"};
    constexpr Driver kAngleLavapipe{"ANGLE/lavapipe",
                                    "ANGLE (Mesa, Vulkan 1.4.305 (llvmpipe (LLVM 19.1.7 256 bits) (0x00000000)), llvmpipe-0.0.1)",
                                    "OpenGL ES 3.1.0 (ANGLE 2.1.17841 git hash: 5fa2e61da947)"};
    constexpr Driver kAngleEs30{"ANGLE/lavapipe ES 3.0",
                                "ANGLE (Mesa, Vulkan 1.4.305 (llvmpipe (LLVM 19.1.7 256 bits) (0x00000000)), llvmpipe-0.0.1)",
                                "OpenGL ES 3.0.0 (ANGLE 2.1.17841 git hash: 5fa2e61da947)"};
    // A hypothetical ANGLE on RADV: the backend name must not select a Mesa deviation either.
    constexpr Driver kAngleRadv{"ANGLE/RADV", "ANGLE (AMD, Vulkan 1.4.305 (AMD Radeon 780M (RADV PHOENIX) (0x000015BF)), radv-25.0.7)",
                                "OpenGL ES 3.1.0 (ANGLE 2.1.17841 git hash: 5fa2e61da947)"};

    int failures = 0;

    void expect(bool match_expected, std::string_view tag, const Driver& d)
    {
        const bool got = demo::deviation_matches(tag, d.renderer, d.version);
        const bool ok = got == match_expected;
        if (!ok) ++failures;
        std::printf("   [%s] tag '%.*s' on %.*s: %s (expected %s)\n", ok ? "ok" : "FAIL", static_cast<int>(tag.size()),
                    tag.data(), static_cast<int>(d.name.size()), d.name.data(), got ? "match" : "no match",
                    match_expected ? "match" : "no match");
    }
}

int main()
{
    std::printf("== unit-deviation-matcher\n");
    // "Mesa" selects every Mesa driver and never ANGLE, whatever ANGLE's backend is.
    for (const Driver* d : {&kRadeonsi, &kLlvmpipe, &kSoftpipe, &kZink}) expect(true, "Mesa", *d);
    for (const Driver* d : {&kAngleLavapipe, &kAngleEs30, &kAngleRadv}) expect(false, "Mesa", *d);

    // A Mesa driver name selects that driver only ("llvmpipe" also covers zink on lavapipe, see check.hpp).
    expect(true, "radeonsi", kRadeonsi);
    expect(false, "radeonsi", kLlvmpipe);
    expect(false, "radeonsi", kZink);
    expect(true, "llvmpipe", kLlvmpipe);
    expect(true, "llvmpipe", kZink);
    expect(false, "llvmpipe", kSoftpipe);
    expect(false, "llvmpipe", kRadeonsi);
    expect(true, "zink", kZink);
    expect(false, "zink", kLlvmpipe);
    expect(true, "softpipe", kSoftpipe);
    for (const Driver* d : {&kAngleLavapipe, &kAngleEs30, &kAngleRadv})
        for (const std::string_view tag : {"llvmpipe", "radeonsi", "zink", "Vulkan", "AMD", "RADV"}) expect(false, tag, *d);

    // "ANGLE" selects ANGLE on any backend and no Mesa driver.
    for (const Driver* d : {&kAngleLavapipe, &kAngleEs30, &kAngleRadv}) expect(true, "ANGLE", *d);
    for (const Driver* d : {&kRadeonsi, &kLlvmpipe, &kSoftpipe, &kZink}) expect(false, "ANGLE", *d);

    // An empty tag (a demo found no matching driver name) gives the strict check everywhere.
    for (const Driver* d : {&kRadeonsi, &kLlvmpipe, &kSoftpipe, &kZink, &kAngleLavapipe, &kAngleRadv}) expect(false, "", *d);

    std::printf("RESULT: %s (%d failure(s))\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
