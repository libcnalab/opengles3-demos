#include "demo/gl_util.hpp"

#include "demo/runtime.hpp"

#include <format>

namespace mgl = metagl;

namespace demo
{
    std::string shader_info_log(mgl::ShaderId shader)
    {
        GLint length = 0;
        mgl::glGetShaderiv(shader, mgl::ShaderParameter::InfoLogLength, &length);
        if (length <= 1) return {};
        std::string log(static_cast<std::size_t>(length), '\0');
        GLsizei written = 0;
        mgl::glGetShaderInfoLog(shader, length, &written, log.data());
        log.resize(static_cast<std::size_t>(written));
        return log;
    }

    std::string program_info_log(mgl::ProgramId program)
    {
        GLint length = 0;
        mgl::glGetProgramiv(program, mgl::ProgramParameter::InfoLogLength, &length);
        if (length <= 1) return {};
        std::string log(static_cast<std::size_t>(length), '\0');
        GLsizei written = 0;
        mgl::glGetProgramInfoLog(program, length, &written, log.data());
        log.resize(static_cast<std::size_t>(written));
        return log;
    }

    mgl::ShaderId compile_shader(mgl::ShaderType type, std::string_view source, std::string& log)
    {
        const mgl::ShaderId shader = mgl::glCreateShader(type);
        mgl::glShaderSource(shader, source);
        mgl::glCompileShader(shader);
        GLint status = GL_FALSE;
        mgl::glGetShaderiv(shader, mgl::ShaderParameter::CompileStatus, &status);
        log = shader_info_log(shader);
        if (status != GL_TRUE)
        {
            mgl::glDeleteShader(shader);
            return mgl::ShaderId{0};
        }
        return shader;
    }

    mgl::ProgramId link_program(std::span<const mgl::ShaderId> shaders, std::string& log)
    {
        const mgl::ProgramId program = mgl::glCreateProgram();
        for (const mgl::ShaderId s : shaders) mgl::glAttachShader(program, s);
        mgl::glLinkProgram(program);
        GLint status = GL_FALSE;
        mgl::glGetProgramiv(program, mgl::ProgramParameter::LinkStatus, &status);
        log = program_info_log(program);
        // The linked executable does not depend on the shader objects any more.
        for (const mgl::ShaderId s : shaders) mgl::glDetachShader(program, s);
        if (status != GL_TRUE)
        {
            mgl::glDeleteProgram(program);
            return mgl::ProgramId{0};
        }
        return program;
    }

    mgl::ProgramId build_program_from_source(std::string_view vertex_src, std::string_view fragment_src,
                                             std::string_view label)
    {
        std::string log;
        const mgl::ShaderId vs = compile_shader(mgl::ShaderType::Vertex, vertex_src, log);
        if (vs.value == 0) throw Failure(std::format("{}: vertex shader failed to compile:\n{}", label, log));
        const mgl::ShaderId fs = compile_shader(mgl::ShaderType::Fragment, fragment_src, log);
        if (fs.value == 0)
        {
            mgl::glDeleteShader(vs);
            throw Failure(std::format("{}: fragment shader failed to compile:\n{}", label, log));
        }
        const mgl::ShaderId both[] = {vs, fs};
        const mgl::ProgramId program = link_program(both, log);
        mgl::glDeleteShader(vs);
        mgl::glDeleteShader(fs);
        if (program.value == 0) throw Failure(std::format("{}: program failed to link:\n{}", label, log));
        return program;
    }

    mgl::ProgramId build_program(const Env& env, std::string_view vertex_file, std::string_view fragment_file)
    {
        return build_program_from_source(env.load_text(vertex_file), env.load_text(fragment_file),
                                         std::format("{} + {}", vertex_file, fragment_file));
    }

    std::vector<std::uint8_t> read_rgba8(int x, int y, int width, int height)
    {
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
        // RGBA/UNSIGNED_BYTE is the one format/type pair every ES implementation must accept
        // for glReadPixels on a normalized fixed-point color buffer (ES 3.0 §4.3.2).
        mgl::glReadPixels(x, y, width, height, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pixels.data());
        return pixels;
    }

    Rgba8 read_pixel(int x, int y)
    {
        Rgba8 p{};
        mgl::glReadPixels(x, y, 1, 1, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, p.data());
        return p;
    }

    std::string_view error_name(mgl::ErrorCode code)
    {
        switch (code)
        {
        case mgl::ErrorCode::NoError: return "GL_NO_ERROR";
        case mgl::ErrorCode::InvalidEnum: return "GL_INVALID_ENUM";
        case mgl::ErrorCode::InvalidValue: return "GL_INVALID_VALUE";
        case mgl::ErrorCode::InvalidOperation: return "GL_INVALID_OPERATION";
        case mgl::ErrorCode::OutOfMemory: return "GL_OUT_OF_MEMORY";
        case mgl::ErrorCode::InvalidFramebufferOperation: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case mgl::ErrorCode::StackOverflow: return "GL_STACK_OVERFLOW";
        case mgl::ErrorCode::StackUnderflow: return "GL_STACK_UNDERFLOW";
        case mgl::ErrorCode::ContextLost: return "GL_CONTEXT_LOST";
        }
        return "GL_<unknown error>";
    }

    std::string_view framebuffer_status_name(mgl::FramebufferStatus status)
    {
        switch (status)
        {
        case mgl::FramebufferStatus::Complete: return "GL_FRAMEBUFFER_COMPLETE";
        case mgl::FramebufferStatus::Undefined: return "GL_FRAMEBUFFER_UNDEFINED";
        case mgl::FramebufferStatus::IncompleteAttachment: return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
        case mgl::FramebufferStatus::IncompleteMissingAttachment: return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
        case mgl::FramebufferStatus::IncompleteDimensions: return "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS";
        case mgl::FramebufferStatus::Unsupported: return "GL_FRAMEBUFFER_UNSUPPORTED";
        case mgl::FramebufferStatus::IncompleteMultisample: return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
        case mgl::FramebufferStatus::IncompleteLayerTargets: return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
        }
        return "GL_<unknown framebuffer status>";
    }

    mgl::ErrorCode drain_gl_errors()
    {
        mgl::ErrorCode first = mgl::ErrorCode::NoError;
        // Bounded: a lost context may report GL_CONTEXT_LOST forever.
        for (int i = 0; i < 64; ++i)
        {
            const mgl::ErrorCode e = mgl::glGetError();
            if (e == mgl::ErrorCode::NoError) break;
            if (first == mgl::ErrorCode::NoError) first = e;
            if (e == mgl::ErrorCode::ContextLost) break;
        }
        return first;
    }

    int get_integer(mgl::GetParameter pname)
    {
        GLint v = 0;
        mgl::glGetIntegerv(pname, &v);
        return v;
    }
}
