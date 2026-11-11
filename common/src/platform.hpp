// Internal: native context creation. Not part of the demo-facing API.
#pragma once

#include "demo/runtime.hpp"

#include <memory>
#include <string>

namespace demo::detail
{
    struct ContextRequest
    {
        Version version{3, 0};
        int width = 256;
        int height = 256;
        int depth_bits = 24;
        int stencil_bits = 8;
        int samples = 0;
        bool debug = false;
        bool robust = false;
        std::string title;
    };

    /// Input events delivered by a windowed platform.
    struct Events
    {
        bool quit = false;
        bool resized = false;
        int width = 0;
        int height = 0;
        std::string keys;  ///< ASCII keys pressed since the last poll
    };

    class Platform
    {
    public:
        virtual ~Platform() = default;
        /// Creates the context and makes it current. On failure returns false and sets `error`.
        virtual bool create(const ContextRequest& request, std::string& error) = 0;
        /// GetProcAddress-style loader to hand to metagl::Initialize().
        [[nodiscard]] virtual void* (*loader() const)(const char*) = 0;
        virtual Events poll() = 0;
        virtual void swap() = 0;
        [[nodiscard]] virtual std::string description() const = 0;
    };

    std::unique_ptr<Platform> make_egl_headless_platform();
#if OGLES3_HAVE_SDL3
    std::unique_ptr<Platform> make_sdl3_platform();
#endif
}
