// Small, explicit helpers around meta-gl. They exist so that demos whose subject is NOT shader
// compilation do not repeat 30 lines of compile/link/log boilerplate. Each helper is a direct
// sequence of metagl::gl* calls (see gl_util.cpp) - nothing is cached or hidden.
//
// The demos about shaders themselves (es30-shader-compilation, ...) intentionally do NOT use
// these helpers and spell every call out.
#pragma once

#include "demo/check.hpp"

#include <metagl/metagl.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace demo
{
    struct Env;

    /// glGetShaderInfoLog as a std::string (empty when the log is empty).
    [[nodiscard]] std::string shader_info_log(metagl::ShaderId shader);

    /// glGetProgramInfoLog as a std::string.
    [[nodiscard]] std::string program_info_log(metagl::ProgramId program);

    /// glCreateShader + glShaderSource + glCompileShader. On failure the shader is deleted,
    /// `log` receives the info log, and {0} is returned.
    [[nodiscard]] metagl::ShaderId compile_shader(metagl::ShaderType type, std::string_view source, std::string& log);

    /// glCreateProgram + glAttachShader* + glLinkProgram (+ glDetachShader after link).
    /// On failure the program is deleted, `log` receives the info log, and {0} is returned.
    /// The shaders are NOT deleted; the caller owns them.
    [[nodiscard]] metagl::ProgramId link_program(std::span<const metagl::ShaderId> shaders, std::string& log);

    /// Loads, compiles and links a vertex+fragment program from the demo's shader files.
    /// Throws demo::Failure with the full info log on any error. Shaders are deleted after link.
    [[nodiscard]] metagl::ProgramId build_program(const struct Env& env, std::string_view vertex_file,
                                                  std::string_view fragment_file);

    /// Same, from in-memory sources (label is used in error messages).
    [[nodiscard]] metagl::ProgramId build_program_from_source(std::string_view vertex_src,
                                                              std::string_view fragment_src,
                                                              std::string_view label);

    /// Reads a w*h RGBA8 block from the current READ framebuffer (bottom-up rows, GL order).
    [[nodiscard]] std::vector<std::uint8_t> read_rgba8(int x, int y, int width, int height);

    /// Reads one RGBA8 pixel.
    [[nodiscard]] Rgba8 read_pixel(int x, int y);

    /// Name of a GL error code ("GL_INVALID_ENUM", ...).
    [[nodiscard]] std::string_view error_name(metagl::ErrorCode code);

    /// Name of a framebuffer status ("GL_FRAMEBUFFER_COMPLETE", ...).
    [[nodiscard]] std::string_view framebuffer_status_name(metagl::FramebufferStatus status);

    /// Drains the GL error queue and returns the first error (NoError if empty).
    metagl::ErrorCode drain_gl_errors();

    /// glGetIntegerv convenience for a single value.
    [[nodiscard]] int get_integer(metagl::GetParameter pname);

    /// Converts a byte offset into the `const void*` that buffer-relative GL entry points expect
    /// (glVertexAttribPointer, glDrawElements, ... when a buffer object is bound).
    [[nodiscard]] inline const void* buffer_offset(std::uintptr_t bytes) noexcept
    {
        return reinterpret_cast<const void*>(bytes);
    }
}
