// Windowed OpenGL ES context through SDL3 (SDL_GL_CONTEXT_PROFILE_ES).
#include "platform.hpp"

#include <SDL3/SDL.h>

#include <cctype>
#include <format>

namespace demo::detail
{
    namespace
    {
        void* sdl_loader(const char* name)
        {
            return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name));
        }

        class Sdl3Window final : public Platform
        {
        public:
            ~Sdl3Window() override
            {
                if (context_ != nullptr) SDL_GL_DestroyContext(context_);
                if (window_ != nullptr) SDL_DestroyWindow(window_);
                if (initialized_) SDL_Quit();
            }

            bool create(const ContextRequest& req, std::string& error) override
            {
                if (!SDL_Init(SDL_INIT_VIDEO))
                {
                    error = std::format("SDL_Init failed: {}", SDL_GetError());
                    return false;
                }
                initialized_ = true;
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, req.version.major);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, req.version.minor);
                SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
                SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
                SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
                SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
                SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, req.depth_bits);
                SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, req.stencil_bits);
                SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, req.samples > 0 ? 1 : 0);
                SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, req.samples);
                int flags = 0;
                if (req.debug) flags |= SDL_GL_CONTEXT_DEBUG_FLAG;
                if (req.robust) flags |= SDL_GL_CONTEXT_ROBUST_ACCESS_FLAG | SDL_GL_CONTEXT_RESET_ISOLATION_FLAG;
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, flags);

                window_ = SDL_CreateWindow(req.title.c_str(), req.width, req.height,
                                           SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
                if (window_ == nullptr)
                {
                    error = std::format("SDL_CreateWindow failed: {}", SDL_GetError());
                    return false;
                }
                context_ = SDL_GL_CreateContext(window_);
                if (context_ == nullptr)
                {
                    error = std::format("SDL_GL_CreateContext (ES {}.{}) failed: {}", req.version.major,
                                        req.version.minor, SDL_GetError());
                    return false;
                }
                SDL_GL_MakeCurrent(window_, context_);
                SDL_GL_SetSwapInterval(1);
                return true;
            }

            void* (*loader() const)(const char*) override { return &sdl_loader; }

            Events poll() override
            {
                Events ev;
                SDL_Event e;
                while (SDL_PollEvent(&e))
                {
                    switch (e.type)
                    {
                    case SDL_EVENT_QUIT:
                    case SDL_EVENT_WINDOW_CLOSE_REQUESTED: ev.quit = true; break;
                    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                        ev.resized = true;
                        ev.width = e.window.data1;
                        ev.height = e.window.data2;
                        break;
                    case SDL_EVENT_KEY_DOWN:
                        if (e.key.key == SDLK_ESCAPE) ev.quit = true;
                        else if (e.key.key < 128 && std::isprint(static_cast<int>(e.key.key)))
                            ev.keys.push_back(static_cast<char>(std::tolower(static_cast<int>(e.key.key))));
                        break;
                    default: break;
                    }
                }
                return ev;
            }

            void swap() override { SDL_GL_SwapWindow(window_); }

            std::string description() const override
            {
                return std::format("SDL {}.{}.{} window (video driver={})", SDL_MAJOR_VERSION, SDL_MINOR_VERSION,
                                   SDL_MICRO_VERSION, SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "?");
            }

        private:
            bool initialized_ = false;
            SDL_Window* window_ = nullptr;
            SDL_GLContext context_ = nullptr;
        };
    }

    std::unique_ptr<Platform> make_sdl3_platform()
    {
        return std::make_unique<Sdl3Window>();
    }
}
