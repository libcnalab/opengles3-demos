// Dependency-free PNG writer for framebuffer captures (stored/uncompressed deflate blocks).
#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace demo
{
    /// Writes an RGBA8 image. `bottom_up` = rows are in GL order (first row = bottom).
    bool write_png_rgba8(const std::string& path, int width, int height,
                         std::span<const std::uint8_t> rgba, bool bottom_up);
}
