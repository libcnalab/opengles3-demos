// Headless OpenGL ES context: EGL display (Mesa surfaceless platform when available) with a
// pbuffer surface. The pbuffer is a real *default framebuffer* (FBO name 0) with color, depth,
// stencil and optional multisample buffers, so demos render exactly as they would to a window.
#include "platform.hpp"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cstdlib>
#include <cstring>
#include <format>
#include <string_view>

namespace demo::detail
{
    namespace
    {
        void* egl_loader(const char* name)
        {
            return reinterpret_cast<void*>(eglGetProcAddress(name));
        }

        bool has_extension(const char* list, std::string_view name)
        {
            if (list == nullptr) return false;
            std::string_view all{list};
            std::size_t pos = 0;
            while ((pos = all.find(name, pos)) != std::string_view::npos)
            {
                const bool start_ok = pos == 0 || all[pos - 1] == ' ';
                const std::size_t end = pos + name.size();
                const bool end_ok = end == all.size() || all[end] == ' ';
                if (start_ok && end_ok) return true;
                pos = end;
            }
            return false;
        }

        std::string egl_error_name(EGLint e)
        {
            switch (e)
            {
            case EGL_SUCCESS: return "EGL_SUCCESS";
            case EGL_NOT_INITIALIZED: return "EGL_NOT_INITIALIZED";
            case EGL_BAD_ACCESS: return "EGL_BAD_ACCESS";
            case EGL_BAD_ALLOC: return "EGL_BAD_ALLOC";
            case EGL_BAD_ATTRIBUTE: return "EGL_BAD_ATTRIBUTE";
            case EGL_BAD_CONFIG: return "EGL_BAD_CONFIG";
            case EGL_BAD_CONTEXT: return "EGL_BAD_CONTEXT";
            case EGL_BAD_DISPLAY: return "EGL_BAD_DISPLAY";
            case EGL_BAD_MATCH: return "EGL_BAD_MATCH";
            case EGL_BAD_PARAMETER: return "EGL_BAD_PARAMETER";
            case EGL_BAD_SURFACE: return "EGL_BAD_SURFACE";
            default: return std::format("0x{:04X}", e);
            }
        }

        class EglHeadless final : public Platform
        {
        public:
            ~EglHeadless() override
            {
                if (display_ == EGL_NO_DISPLAY) return;
                eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
                if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
                eglTerminate(display_);
            }

            bool create(const ContextRequest& req, std::string& error) override
            {
                const char* client_ext = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
                // ANGLE (selected with OGLES3_EGL_PLATFORM=angle and an ANGLE libEGL on the library path): a
                // headless Vulkan-backed display. This changes only the context provider; every GL call still
                // goes through meta-gl.
                const char* want = std::getenv("OGLES3_EGL_PLATFORM");
                if (want != nullptr && std::string_view(want) == "angle")
                {
                    if (!has_extension(client_ext, "EGL_ANGLE_platform_angle"))
                    {
                        error = "OGLES3_EGL_PLATFORM=angle, but the loaded libEGL is not ANGLE (no EGL_ANGLE_platform_angle)";
                        return false;
                    }
                    constexpr EGLenum kPlatformAngle = 0x3202;      // EGL_PLATFORM_ANGLE_ANGLE
                    constexpr EGLint kPlatformAngleType = 0x3203;   // EGL_PLATFORM_ANGLE_TYPE_ANGLE
                    constexpr EGLint kPlatformAngleVulkan = 0x3450; // EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE
                    const EGLint angle_attribs[] = {kPlatformAngleType, kPlatformAngleVulkan, EGL_NONE};
                    auto get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                        eglGetProcAddress("eglGetPlatformDisplayEXT"));
                    if (get_platform_display != nullptr)
                    {
                        display_ = get_platform_display(kPlatformAngle, EGL_DEFAULT_DISPLAY, angle_attribs);
                        platform_ = "angle-vulkan";
                    }
                }
                else if (has_extension(client_ext, "EGL_MESA_platform_surfaceless"))
                {
                    auto get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                        eglGetProcAddress("eglGetPlatformDisplayEXT"));
                    if (get_platform_display != nullptr)
                    {
                        display_ = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
                        platform_ = "surfaceless";
                    }
                }
                if (display_ == EGL_NO_DISPLAY)
                {
                    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
                    platform_ = "default";
                }
                if (display_ == EGL_NO_DISPLAY)
                {
                    error = "eglGetDisplay failed";
                    return false;
                }
                EGLint major = 0, minor = 0;
                if (eglInitialize(display_, &major, &minor) != EGL_TRUE)
                {
                    error = "eglInitialize failed: " + egl_error_name(eglGetError());
                    display_ = EGL_NO_DISPLAY;
                    return false;
                }
                egl_version_ = std::format("EGL {}.{}", major, minor);
                if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE)
                {
                    error = "eglBindAPI(EGL_OPENGL_ES_API) failed";
                    return false;
                }

                const EGLint config_attribs[] = {
                    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                    EGL_DEPTH_SIZE, req.depth_bits,
                    EGL_STENCIL_SIZE, req.stencil_bits,
                    EGL_SAMPLE_BUFFERS, req.samples > 0 ? 1 : 0,
                    EGL_SAMPLES, req.samples,
                    EGL_NONE};
                EGLConfig config = nullptr;
                EGLint count = 0;
                if (eglChooseConfig(display_, config_attribs, &config, 1, &count) != EGL_TRUE || count == 0)
                {
                    error = std::format("eglChooseConfig: no ES3 pbuffer config with depth={} stencil={} samples={}",
                                        req.depth_bits, req.stencil_bits, req.samples);
                    return false;
                }

                const EGLint surface_attribs[] = {EGL_WIDTH, req.width, EGL_HEIGHT, req.height, EGL_NONE};
                surface_ = eglCreatePbufferSurface(display_, config, surface_attribs);
                if (surface_ == EGL_NO_SURFACE)
                {
                    error = "eglCreatePbufferSurface failed: " + egl_error_name(eglGetError());
                    return false;
                }

                EGLint ctx_attribs[16];
                int n = 0;
                ctx_attribs[n++] = EGL_CONTEXT_MAJOR_VERSION;
                ctx_attribs[n++] = req.version.major;
                ctx_attribs[n++] = EGL_CONTEXT_MINOR_VERSION;
                ctx_attribs[n++] = req.version.minor;
                if (req.debug)
                {
                    ctx_attribs[n++] = EGL_CONTEXT_OPENGL_DEBUG;
                    ctx_attribs[n++] = EGL_TRUE;
                }
                if (req.robust)
                {
                    ctx_attribs[n++] = EGL_CONTEXT_OPENGL_ROBUST_ACCESS;
                    ctx_attribs[n++] = EGL_TRUE;
                    ctx_attribs[n++] = EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY;
                    ctx_attribs[n++] = EGL_LOSE_CONTEXT_ON_RESET;
                }
                ctx_attribs[n++] = EGL_NONE;

                context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, ctx_attribs);
                if (context_ == EGL_NO_CONTEXT)
                {
                    error = std::format("eglCreateContext(ES {}.{}) failed: {}", req.version.major,
                                        req.version.minor, egl_error_name(eglGetError()));
                    return false;
                }
                if (eglMakeCurrent(display_, surface_, surface_, context_) != EGL_TRUE)
                {
                    error = "eglMakeCurrent failed: " + egl_error_name(eglGetError());
                    return false;
                }
                return true;
            }

            void* (*loader() const)(const char*) override { return &egl_loader; }

            Events poll() override { return {}; }

            void swap() override { eglSwapBuffers(display_, surface_); }

            std::string description() const override
            {
                return std::format("{} headless pbuffer (platform={}, vendor={})", egl_version_, platform_,
                                   display_ != EGL_NO_DISPLAY ? eglQueryString(display_, EGL_VENDOR) : "?");
            }

        private:
            EGLDisplay display_ = EGL_NO_DISPLAY;
            EGLSurface surface_ = EGL_NO_SURFACE;
            EGLContext context_ = EGL_NO_CONTEXT;
            std::string platform_;
            std::string egl_version_;
        };
    }

    std::unique_ptr<Platform> make_egl_headless_platform()
    {
        return std::make_unique<EglHeadless>();
    }
}
