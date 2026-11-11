// diagnostics-capabilities
//
// Capability and extension report for the current OpenGL ES context, and a check that
// (1) every implementation limit meets the minimum the specification requires for the context's
//     version (ES 3.0 Tables 6.28-6.34, ES 3.1/3.2 "Implementation Dependent ..." tables),
// (2) extension enumeration through glGetStringi agrees with meta-gl's Capabilities / HasExtension,
// (3) meta-gl reports every core entry point of the context's version as available
//     (inventory/es_core_commands.csv, derived from the Khronos registry).
//
// The image is a version strip: one bar per ES version, green when the context supports it.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <iterator>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace mgl = metagl;
using P = mgl::GetParameter;

namespace
{
    constexpr double NA = std::numeric_limits<double>::quiet_NaN();  // limit not defined for that version

    enum class Kind
    {
        AtLeast,         // value >= minimum   (glGetIntegerv)
        AtLeast64,       // value >= minimum   (glGetInteger64v)
        AtLeastFloat,    // value >= minimum   (glGetFloatv)
        AtMost,          // value <= bound     (alignments, MIN_* offsets)
        AtMostFloat,     // value <= bound     (glGetFloatv)
        RangeHasOne,     // [lo, hi] (glGetFloatv, 2 values) contains 1.0
    };

    struct Limit
    {
        P pname;
        const char* name;
        Kind kind;
        double es30, es31, es32;  // per-version requirement (NaN = not part of that version)
    };

    // Values transcribed from the specification tables; see README.md for the table numbers.
    const Limit kLimits[] = {
        {P::MaxElementIndex, "MAX_ELEMENT_INDEX", Kind::AtLeast64, 16777215, 16777215, 16777215},
        {P::SubpixelBits, "SUBPIXEL_BITS", Kind::AtLeast, 4, 4, 4},
        {P::Max3DTextureSize, "MAX_3D_TEXTURE_SIZE", Kind::AtLeast, 256, 256, 256},
        {P::MaxTextureSize, "MAX_TEXTURE_SIZE", Kind::AtLeast, 2048, 2048, 2048},
        {P::MaxArrayTextureLayers, "MAX_ARRAY_TEXTURE_LAYERS", Kind::AtLeast, 256, 256, 256},
        {P::MaxTextureLodBias, "MAX_TEXTURE_LOD_BIAS", Kind::AtLeastFloat, 2.0, 2.0, 2.0},
        {P::MaxCubeMapTextureSize, "MAX_CUBE_MAP_TEXTURE_SIZE", Kind::AtLeast, 2048, 2048, 2048},
        {P::MaxRenderbufferSize, "MAX_RENDERBUFFER_SIZE", Kind::AtLeast, 2048, 2048, 2048},
        {P::MaxDrawBuffers, "MAX_DRAW_BUFFERS", Kind::AtLeast, 4, 4, 4},
        {P::MaxColorAttachments, "MAX_COLOR_ATTACHMENTS", Kind::AtLeast, 4, 4, 4},
        {P::AliasedPointSizeRange, "ALIASED_POINT_SIZE_RANGE", Kind::RangeHasOne, 1, 1, 1},
        {P::AliasedLineWidthRange, "ALIASED_LINE_WIDTH_RANGE", Kind::RangeHasOne, 1, 1, 1},
        {P::MultisampleLineWidthRange, "MULTISAMPLE_LINE_WIDTH_RANGE", Kind::RangeHasOne, NA, NA, 1},
        {P::NumCompressedTextureFormats, "NUM_COMPRESSED_TEXTURE_FORMATS", Kind::AtLeast, 10, 10, 10},
        {P::NumProgramBinaryFormats, "NUM_PROGRAM_BINARY_FORMATS", Kind::AtLeast, 0, 0, 0},
        {P::NumShaderBinaryFormats, "NUM_SHADER_BINARY_FORMATS", Kind::AtLeast, 0, 0, 0},
        {P::MaxServerWaitTimeout, "MAX_SERVER_WAIT_TIMEOUT", Kind::AtLeast64, 0, 0, 0},
        {P::MajorVersion, "MAJOR_VERSION", Kind::AtLeast, 3, 3, 3},
        {P::MaxVertexAttribs, "MAX_VERTEX_ATTRIBS", Kind::AtLeast, 16, 16, 16},
        {P::MaxVertexUniformComponents, "MAX_VERTEX_UNIFORM_COMPONENTS", Kind::AtLeast, 1024, 1024, 1024},
        {P::MaxVertexUniformVectors, "MAX_VERTEX_UNIFORM_VECTORS", Kind::AtLeast, 256, 256, 256},
        {P::MaxVertexUniformBlocks, "MAX_VERTEX_UNIFORM_BLOCKS", Kind::AtLeast, 12, 12, 12},
        {P::MaxVertexOutputComponents, "MAX_VERTEX_OUTPUT_COMPONENTS", Kind::AtLeast, 64, 64, 64},
        {P::MaxVertexTextureImageUnits, "MAX_VERTEX_TEXTURE_IMAGE_UNITS", Kind::AtLeast, 16, 16, 16},
        {P::MaxFragmentUniformComponents, "MAX_FRAGMENT_UNIFORM_COMPONENTS", Kind::AtLeast, 896, 1024, 1024},
        {P::MaxFragmentUniformVectors, "MAX_FRAGMENT_UNIFORM_VECTORS", Kind::AtLeast, 224, 256, 256},
        {P::MaxFragmentUniformBlocks, "MAX_FRAGMENT_UNIFORM_BLOCKS", Kind::AtLeast, 12, 12, 12},
        {P::MaxFragmentInputComponents, "MAX_FRAGMENT_INPUT_COMPONENTS", Kind::AtLeast, 60, 60, 60},
        {P::MaxTextureImageUnits, "MAX_TEXTURE_IMAGE_UNITS", Kind::AtLeast, 16, 16, 16},
        {P::MinProgramTexelOffset, "MIN_PROGRAM_TEXEL_OFFSET", Kind::AtMost, -8, -8, -8},
        {P::MaxProgramTexelOffset, "MAX_PROGRAM_TEXEL_OFFSET", Kind::AtLeast, 7, 7, 7},
        {P::MaxUniformBufferBindings, "MAX_UNIFORM_BUFFER_BINDINGS", Kind::AtLeast, 24, 36, 72},
        {P::MaxUniformBlockSize, "MAX_UNIFORM_BLOCK_SIZE", Kind::AtLeast64, 16384, 16384, 16384},
        {P::UniformBufferOffsetAlignment, "UNIFORM_BUFFER_OFFSET_ALIGNMENT", Kind::AtMost, 256, 256, 256},
        {P::MaxCombinedUniformBlocks, "MAX_COMBINED_UNIFORM_BLOCKS", Kind::AtLeast, 24, 24, 60},
        {P::MaxVaryingComponents, "MAX_VARYING_COMPONENTS", Kind::AtLeast, 60, 60, 60},
        {P::MaxVaryingVectors, "MAX_VARYING_VECTORS", Kind::AtLeast, 15, 15, 15},
        {P::MaxCombinedTextureImageUnits, "MAX_COMBINED_TEXTURE_IMAGE_UNITS", Kind::AtLeast, 32, 48, 96},
        {P::MaxTransformFeedbackInterleavedComponents, "MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS", Kind::AtLeast, 64, 64, 64},
        {P::MaxTransformFeedbackSeparateAttribs, "MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS", Kind::AtLeast, 4, 4, 4},
        {P::MaxTransformFeedbackSeparateComponents, "MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS", Kind::AtLeast, 4, 4, 4},
        {P::MaxSamples, "MAX_SAMPLES", Kind::AtLeast, 4, 4, 4},
        // --- ES 3.1 ---
        {P::MaxFramebufferWidth, "MAX_FRAMEBUFFER_WIDTH", Kind::AtLeast, NA, 2048, 2048},
        {P::MaxFramebufferHeight, "MAX_FRAMEBUFFER_HEIGHT", Kind::AtLeast, NA, 2048, 2048},
        {P::MaxFramebufferSamples, "MAX_FRAMEBUFFER_SAMPLES", Kind::AtLeast, NA, 4, 4},
        {P::MaxSampleMaskWords, "MAX_SAMPLE_MASK_WORDS", Kind::AtLeast, NA, 1, 1},
        {P::MaxColorTextureSamples, "MAX_COLOR_TEXTURE_SAMPLES", Kind::AtLeast, NA, 1, 1},
        {P::MaxDepthTextureSamples, "MAX_DEPTH_TEXTURE_SAMPLES", Kind::AtLeast, NA, 1, 1},
        {P::MaxIntegerSamples, "MAX_INTEGER_SAMPLES", Kind::AtLeast, NA, 1, 1},
        {P::MaxVertexAttribRelativeOffset, "MAX_VERTEX_ATTRIB_RELATIVE_OFFSET", Kind::AtLeast, NA, 2047, 2047},
        {P::MaxVertexAttribBindings, "MAX_VERTEX_ATTRIB_BINDINGS", Kind::AtLeast, NA, 16, 16},
        {P::MaxVertexAttribStride, "MAX_VERTEX_ATTRIB_STRIDE", Kind::AtLeast, NA, 2048, 2048},
        {P::MaxComputeWorkGroupInvocations, "MAX_COMPUTE_WORK_GROUP_INVOCATIONS", Kind::AtLeast, NA, 128, 128},
        {P::MaxComputeUniformBlocks, "MAX_COMPUTE_UNIFORM_BLOCKS", Kind::AtLeast, NA, 12, 12},
        {P::MaxComputeTextureImageUnits, "MAX_COMPUTE_TEXTURE_IMAGE_UNITS", Kind::AtLeast, NA, 16, 16},
        {P::MaxComputeSharedMemorySize, "MAX_COMPUTE_SHARED_MEMORY_SIZE", Kind::AtLeast, NA, 16384, 16384},
        {P::MaxComputeUniformComponents, "MAX_COMPUTE_UNIFORM_COMPONENTS", Kind::AtLeast, NA, 1024, 1024},
        {P::MaxComputeAtomicCounterBuffers, "MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS", Kind::AtLeast, NA, 1, 1},
        {P::MaxComputeAtomicCounters, "MAX_COMPUTE_ATOMIC_COUNTERS", Kind::AtLeast, NA, 8, 8},
        {P::MaxComputeImageUniforms, "MAX_COMPUTE_IMAGE_UNIFORMS", Kind::AtLeast, NA, 4, 4},
        {P::MaxComputeShaderStorageBlocks, "MAX_COMPUTE_SHADER_STORAGE_BLOCKS", Kind::AtLeast, NA, 4, 4},
        {P::MaxCombinedShaderOutputResources, "MAX_COMBINED_SHADER_OUTPUT_RESOURCES", Kind::AtLeast, NA, 4, 4},
        {P::MaxUniformLocations, "MAX_UNIFORM_LOCATIONS", Kind::AtLeast, NA, 1024, 1024},
        {P::MaxAtomicCounterBufferBindings, "MAX_ATOMIC_COUNTER_BUFFER_BINDINGS", Kind::AtLeast, NA, 1, 1},
        {P::MaxAtomicCounterBufferSize, "MAX_ATOMIC_COUNTER_BUFFER_SIZE", Kind::AtLeast, NA, 32, 32},
        {P::MaxCombinedAtomicCounterBuffers, "MAX_COMBINED_ATOMIC_COUNTER_BUFFERS", Kind::AtLeast, NA, 1, 1},
        {P::MaxCombinedAtomicCounters, "MAX_COMBINED_ATOMIC_COUNTERS", Kind::AtLeast, NA, 8, 8},
        {P::MaxImageUnits, "MAX_IMAGE_UNITS", Kind::AtLeast, NA, 4, 4},
        {P::MaxCombinedImageUniforms, "MAX_COMBINED_IMAGE_UNIFORMS", Kind::AtLeast, NA, 4, 4},
        {P::MaxShaderStorageBufferBindings, "MAX_SHADER_STORAGE_BUFFER_BINDINGS", Kind::AtLeast, NA, 4, 4},
        {P::MaxShaderStorageBlockSize, "MAX_SHADER_STORAGE_BLOCK_SIZE", Kind::AtLeast64, NA, 134217728, 134217728},
        {P::MaxCombinedShaderStorageBlocks, "MAX_COMBINED_SHADER_STORAGE_BLOCKS", Kind::AtLeast, NA, 4, 4},
        {P::ShaderStorageBufferOffsetAlignment, "SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT", Kind::AtMost, NA, 256, 256},
        {P::MaxFragmentAtomicCounterBuffers, "MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS", Kind::AtLeast, NA, 0, 1},
        {P::MaxFragmentAtomicCounters, "MAX_FRAGMENT_ATOMIC_COUNTERS", Kind::AtLeast, NA, 0, 8},
        {P::MaxFragmentImageUniforms, "MAX_FRAGMENT_IMAGE_UNIFORMS", Kind::AtLeast, NA, 0, 4},
        {P::MaxFragmentShaderStorageBlocks, "MAX_FRAGMENT_SHADER_STORAGE_BLOCKS", Kind::AtLeast, NA, 0, 4},
        // --- ES 3.2 ---
        {P::MaxTextureBufferSize, "MAX_TEXTURE_BUFFER_SIZE", Kind::AtLeast, NA, NA, 65536},
        {P::TextureBufferOffsetAlignment, "TEXTURE_BUFFER_OFFSET_ALIGNMENT", Kind::AtMost, NA, NA, 256},
        {P::MaxFramebufferLayers, "MAX_FRAMEBUFFER_LAYERS", Kind::AtLeast, NA, NA, 256},
        {P::MinFragmentInterpolationOffset, "MIN_FRAGMENT_INTERPOLATION_OFFSET", Kind::AtMostFloat, NA, NA, -0.5},
        {P::MaxFragmentInterpolationOffset, "MAX_FRAGMENT_INTERPOLATION_OFFSET", Kind::AtLeastFloat, NA, NA, 0.5},
        {P::FragmentInterpolationOffsetBits, "FRAGMENT_INTERPOLATION_OFFSET_BITS", Kind::AtLeast, NA, NA, 4},
        {P::MaxTessGenLevel, "MAX_TESS_GEN_LEVEL", Kind::AtLeast, NA, NA, 64},
        {P::MaxPatchVertices, "MAX_PATCH_VERTICES", Kind::AtLeast, NA, NA, 32},
        {P::MaxTessControlUniformComponents, "MAX_TESS_CONTROL_UNIFORM_COMPONENTS", Kind::AtLeast, NA, NA, 1024},
        {P::MaxTessControlTextureImageUnits, "MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS", Kind::AtLeast, NA, NA, 16},
        {P::MaxTessControlOutputComponents, "MAX_TESS_CONTROL_OUTPUT_COMPONENTS", Kind::AtLeast, NA, NA, 64},
        {P::MaxTessPatchComponents, "MAX_TESS_PATCH_COMPONENTS", Kind::AtLeast, NA, NA, 120},
        {P::MaxTessControlInputComponents, "MAX_TESS_CONTROL_INPUT_COMPONENTS", Kind::AtLeast, NA, NA, 64},
        {P::MaxTessControlUniformBlocks, "MAX_TESS_CONTROL_UNIFORM_BLOCKS", Kind::AtLeast, NA, NA, 12},
        {P::MaxTessEvaluationUniformComponents, "MAX_TESS_EVALUATION_UNIFORM_COMPONENTS", Kind::AtLeast, NA, NA, 1024},
        {P::MaxTessEvaluationTextureImageUnits, "MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS", Kind::AtLeast, NA, NA, 16},
        {P::MaxTessEvaluationOutputComponents, "MAX_TESS_EVALUATION_OUTPUT_COMPONENTS", Kind::AtLeast, NA, NA, 64},
        {P::MaxTessEvaluationInputComponents, "MAX_TESS_EVALUATION_INPUT_COMPONENTS", Kind::AtLeast, NA, NA, 64},
        {P::MaxTessEvaluationUniformBlocks, "MAX_TESS_EVALUATION_UNIFORM_BLOCKS", Kind::AtLeast, NA, NA, 12},
        {P::MaxGeometryUniformComponents, "MAX_GEOMETRY_UNIFORM_COMPONENTS", Kind::AtLeast, NA, NA, 1024},
        {P::MaxGeometryUniformBlocks, "MAX_GEOMETRY_UNIFORM_BLOCKS", Kind::AtLeast, NA, NA, 12},
        {P::MaxGeometryInputComponents, "MAX_GEOMETRY_INPUT_COMPONENTS", Kind::AtLeast, NA, NA, 64},
        {P::MaxGeometryOutputComponents, "MAX_GEOMETRY_OUTPUT_COMPONENTS", Kind::AtLeast, NA, NA, 64},
        {P::MaxGeometryOutputVertices, "MAX_GEOMETRY_OUTPUT_VERTICES", Kind::AtLeast, NA, NA, 256},
        {P::MaxGeometryTotalOutputComponents, "MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS", Kind::AtLeast, NA, NA, 1024},
        {P::MaxGeometryTextureImageUnits, "MAX_GEOMETRY_TEXTURE_IMAGE_UNITS", Kind::AtLeast, NA, NA, 16},
        {P::MaxGeometryShaderInvocations, "MAX_GEOMETRY_SHADER_INVOCATIONS", Kind::AtLeast, NA, NA, 32},
        {P::MaxDebugMessageLength, "MAX_DEBUG_MESSAGE_LENGTH", Kind::AtLeast, NA, NA, 1},
        {P::MaxDebugLoggedMessages, "MAX_DEBUG_LOGGED_MESSAGES", Kind::AtLeast, NA, NA, 1},
        {P::MaxDebugGroupStackDepth, "MAX_DEBUG_GROUP_STACK_DEPTH", Kind::AtLeast, NA, NA, 64},
        {P::MaxLabelLength, "MAX_LABEL_LENGTH", Kind::AtLeast, NA, NA, 256},
    };

    // Measured deviations from the minimums, keyed by a GL_RENDERER substring. The strict check is
    // kept for every other driver; a match is reported as XFAIL (docs/KNOWN_LIMITATIONS.md).
    struct KnownDeviation
    {
        const char* renderer;
        const char* limit;
        const char* reason;
    };
    const KnownDeviation kKnownDeviations[] = {
        {"softpipe", "MAX_SAMPLES", "softpipe (Mesa 25.0) advertises ES 3.1 without multisampling"},
        {"softpipe", "MAX_FRAMEBUFFER_SAMPLES", "softpipe (Mesa 25.0) advertises ES 3.1 without multisampling"},
    };

    std::string gl_str(mgl::StringName n)
    {
        const GLubyte* s = mgl::glGetString(n);
        return s ? reinterpret_cast<const char*>(s) : "";
    }

    class Capabilities final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            // Registry inventory lives at <repo>/inventory; the demo sits in <repo>/demos/diagnostics/<dir>.
            std::ifstream in(env.asset_path("../../../inventory/es_core_commands.csv"));
            if (!in) throw demo::Failure("inventory/es_core_commands.csv not found");
            std::string line;
            std::getline(in, line);  // header
            while (std::getline(in, line))
            {
                const auto c1 = line.find(',');
                const auto c2 = line.find(',', c1 + 1);
                const std::string ver = line.substr(c1 + 1, c2 - c1 - 1);
                commands_.push_back({line.substr(0, c1), {ver[0] - '0', ver[2] - '0'}});
            }
        }

        void render(demo::Env& env, int /*frame*/) override
        {
            // A version strip drawn with scissored clears only: ES 2.0 | 3.0 | 3.1 | 3.2.
            const auto& caps = mgl::GetCapabilities();
            const bool support[4] = {caps.gles20, caps.gles30, caps.gles31, caps.gles32};
            mgl::glViewport(0, 0, env.width, env.height);
            mgl::glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glEnable(mgl::Capability::ScissorTest);
            const int w = env.width / 4;
            for (int i = 0; i < 4; ++i)
            {
                mgl::glScissor(i * w + 4, env.height / 4, w - 8, env.height / 2);
                if (support[i]) mgl::glClearColor(0.20f, 0.80f, 0.30f, 1.0f);
                else mgl::glClearColor(0.35f, 0.35f, 0.38f, 1.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
        }

        void validate(demo::Env& env, demo::Check& check) override
        {
            const demo::Version v = env.context_version;
            const auto& caps = mgl::GetCapabilities();
            std::cout << std::format("     GL_VERSION  {}\n     GLSL        {}\n     vendor      {}\n     renderer    {}\n",
                                     gl_str(mgl::StringName::Version), gl_str(mgl::StringName::ShadingLanguageVersion),
                                     gl_str(mgl::StringName::Vendor), gl_str(mgl::StringName::Renderer));

            // --- meta-gl capability flags vs. the version ----------------------------------------
            check.expect(caps.gles20 && caps.gles30 == (v >= demo::Version{3, 0}) && caps.gles31 == (v >= demo::Version{3, 1}) &&
                             caps.gles32 == (v >= demo::Version{3, 2}) && !caps.desktop_gl,
                         "meta-gl Capabilities gles20/30/31/32 flags match the context version");
            check.expect(caps.version_string == gl_str(mgl::StringName::Version) && caps.renderer == gl_str(mgl::StringName::Renderer),
                         "meta-gl Capabilities strings equal glGetString results");
            if (v >= demo::Version{3, 2})
                check.expect(mgl::AllFunctionsLoaded(), "metagl::AllFunctionsLoaded() on an ES 3.2 context");
            else
                std::cout << "     AllFunctionsLoaded() = " << mgl::AllFunctionsLoaded() << " (only required for ES 3.2)\n";

            check_function_availability(v, check);
            check_extensions(check);
            check_limits(v, check);
            check_compressed_formats(v, check);

            // --- the version strip ----------------------------------------------------------------
            const int w = env.width / 4;
            const bool support[4] = {true, v >= demo::Version{3, 0}, v >= demo::Version{3, 1}, v >= demo::Version{3, 2}};
            for (int i = 0; i < 4; ++i)
                check.expect_pixel(i * w + w / 2, env.height / 2,
                                   support[i] ? demo::Rgba8{51, 204, 77, 255} : demo::Rgba8{89, 89, 97, 255}, 1,
                                   std::format("version bar {} shows {}", i, support[i] ? "supported" : "unsupported"));
        }

        void shutdown(demo::Env& /*env*/) override {}

    private:
        void check_function_availability(demo::Version v, demo::Check& check)
        {
            int required = 0, missing = 0, beyond_available = 0, beyond = 0;
            std::string missing_names;
            for (const auto& [name, introduced] : commands_)
            {
                const bool available = mgl::IsFunctionAvailable(name);
                if (introduced <= v)
                {
                    ++required;
                    if (!available)
                    {
                        ++missing;
                        missing_names += name + " ";
                    }
                }
                else
                {
                    ++beyond;
                    if (available) ++beyond_available;
                }
            }
            check.expect(required > 0 && missing == 0,
                         std::format("metagl::IsFunctionAvailable for all {} core commands up to ES {}", required, demo::to_string(v)),
                         missing_names);
            if (beyond > 0)
                std::cout << std::format("     {} of {} newer-version entry points are also loadable (driver-dependent, not required)\n",
                                         beyond_available, beyond);
            check.expect(!mgl::IsFunctionAvailable("glNotARealFunction"), "IsFunctionAvailable is false for an unknown name");
        }

        void check_extensions(demo::Check& check)
        {
            const auto& caps = mgl::GetCapabilities();
            const int n = demo::get_integer(P::NumExtensions);
            check.expect_eq(n, static_cast<int>(caps.extensions.size()), "GL_NUM_EXTENSIONS equals meta-gl's extension count");
            std::set<std::string> indexed;
            bool all_match = true;
            for (int i = 0; i < n; ++i)
            {
                const GLubyte* e = mgl::glGetStringi(mgl::StringName::Extensions, static_cast<GLuint>(i));
                const std::string s = e ? reinterpret_cast<const char*>(e) : "";
                indexed.insert(s);
                if (s.empty() || i >= static_cast<int>(caps.extensions.size()) || caps.extensions[static_cast<std::size_t>(i)] != s ||
                    !mgl::HasExtension(s))
                    all_match = false;
            }
            check.expect(all_match, "glGetStringi(GL_EXTENSIONS, i) == Capabilities::extensions[i] and HasExtension() for every i");
            check.expect(!mgl::HasExtension("GL_NOT_a_real_extension"), "HasExtension is false for an unknown name");

            // One past the end is an error (ES 3.0 §6.1.6) and returns NULL.
            const GLubyte* past = mgl::glGetStringi(mgl::StringName::Extensions, static_cast<GLuint>(n));
            check.expect(past == nullptr, "glGetStringi(GL_EXTENSIONS, NUM_EXTENSIONS) returns NULL");
            check.expect_gl_error(mgl::ErrorCode::InvalidValue, "glGetStringi index out of range -> INVALID_VALUE");

            // ES 3.x still defines the space-separated glGetString(GL_EXTENSIONS); it must list the same set.
            std::istringstream legacy(gl_str(mgl::StringName::Extensions));
            std::set<std::string> legacy_set{std::istream_iterator<std::string>(legacy), std::istream_iterator<std::string>()};
            check.expect(legacy_set == indexed, "glGetString(GL_EXTENSIONS) lists the same extensions as glGetStringi");

            std::cout << "     extensions (" << n << "):";
            int col = 0;
            for (const auto& e : indexed)
            {
                if (col++ % 3 == 0) std::cout << "\n       ";
                std::cout << std::format("{:<46}", e);
            }
            std::cout << "\n";
        }

        void check_limits(demo::Version v, demo::Check& check)
        {
            std::cout << "     implementation limits (value / required):\n";
            int checked = 0, failed = 0, xfail = 0;
            std::string failures, xfails;
            for (const auto& l : kLimits)
            {
                const double req = v >= demo::Version{3, 2} ? l.es32 : v >= demo::Version{3, 1} ? l.es31 : l.es30;
                if (std::isnan(req)) continue;
                double value = 0.0, value_hi = 0.0;
                bool ok = false;
                switch (l.kind)
                {
                case Kind::AtLeast:
                case Kind::AtMost:
                {
                    GLint x = 0;
                    mgl::glGetIntegerv(l.pname, &x);
                    value = x;
                    ok = l.kind == Kind::AtLeast ? value >= req : value <= req;
                    break;
                }
                case Kind::AtLeast64:
                {
                    GLint64 x = 0;
                    mgl::glGetInteger64v(l.pname, &x);
                    value = static_cast<double>(x);
                    ok = value >= req;
                    break;
                }
                case Kind::AtLeastFloat:
                case Kind::AtMostFloat:
                {
                    GLfloat x = 0.0f;
                    mgl::glGetFloatv(l.pname, &x);
                    value = x;
                    ok = l.kind == Kind::AtLeastFloat ? value >= req : value <= req;
                    break;
                }
                case Kind::RangeHasOne:
                {
                    GLfloat r[2] = {0.0f, 0.0f};
                    mgl::glGetFloatv(l.pname, r);
                    value = r[0];
                    value_hi = r[1];
                    ok = r[0] <= 1.0f && r[1] >= 1.0f;
                    break;
                }
                }
                ++checked;
                const char* rel = (l.kind == Kind::AtMost || l.kind == Kind::AtMostFloat) ? "<=" : ">=";
                const std::string shown = l.kind == Kind::RangeHasOne ? std::format("[{}, {}]", value, value_hi) : std::format("{}", value);
                std::cout << std::format("       {:<46} {:>12}  {} {}{}\n", l.name, shown, l.kind == Kind::RangeHasOne ? "contains" : rel, req,
                                         ok ? "" : "   <-- FAIL");
                if (!ok)
                {
                    const std::string renderer = gl_str(mgl::StringName::Renderer);
                    const auto known = std::find_if(std::begin(kKnownDeviations), std::end(kKnownDeviations), [&](const KnownDeviation& d) {
                        return renderer.find(d.renderer) != std::string::npos && std::string_view(d.limit) == l.name;
                    });
                    if (known != std::end(kKnownDeviations))
                    {
                        ++xfail;
                        xfails += std::format("{}={} ({}) ", l.name, shown, known->reason);
                    }
                    else
                    {
                        ++failed;
                        failures += std::format("{}={} ", l.name, shown);
                    }
                }
            }
            check.expect(failed == 0 && checked > 30,
                         std::format("{} implementation limits meet the ES {} minimums{}", checked, demo::to_string(v),
                                     xfail > 0 ? std::format(" (XFAIL: {} known driver deviation(s))", xfail) : ""),
                         failures.empty() ? xfails : failures);
            check.expect_no_gl_error("every limit pname was accepted by glGet*");

            // Derived minimum: MAX_COMBINED_{VERTEX,FRAGMENT}_UNIFORM_COMPONENTS >=
            //   MAX_stage_UNIFORM_BLOCKS * MAX_UNIFORM_BLOCK_SIZE / 4 + MAX_stage_UNIFORM_COMPONENTS.
            GLint64 block_size = 0;
            mgl::glGetInteger64v(P::MaxUniformBlockSize, &block_size);
            const struct { P combined, blocks, comps; const char* name; } derived[] = {
                {P::MaxCombinedVertexUniformComponents, P::MaxVertexUniformBlocks, P::MaxVertexUniformComponents, "VERTEX"},
                {P::MaxCombinedFragmentUniformComponents, P::MaxFragmentUniformBlocks, P::MaxFragmentUniformComponents, "FRAGMENT"},
            };
            for (const auto& d : derived)
            {
                GLint64 combined = 0;
                mgl::glGetInteger64v(d.combined, &combined);
                const GLint64 need = demo::get_integer(d.blocks) * block_size / 4 + demo::get_integer(d.comps);
                check.expect(combined >= need, std::format("MAX_COMBINED_{}_UNIFORM_COMPONENTS >= blocks*size/4 + components", d.name),
                             std::format("{} vs {}", combined, need));
            }

            if (v >= demo::Version{3, 1})
            {
                // Indexed limits (glGetIntegeri_v): work-group count >= 65535 and size >= (128, 128, 64).
                bool ok = true;
                std::string shown;
                const GLint size_min[3] = {128, 128, 64};
                for (GLuint i = 0; i < 3; ++i)
                {
                    GLint count = 0, size = 0;
                    mgl::glGetIntegeri_v(P::MaxComputeWorkGroupCount, i, &count);
                    mgl::glGetIntegeri_v(P::MaxComputeWorkGroupSize, i, &size);
                    ok = ok && count >= 65535 && size >= size_min[i];
                    shown += std::format("[{}] count={} size={} ", i, count, size);
                }
                check.expect(ok, "MAX_COMPUTE_WORK_GROUP_COUNT >= 65535 and _SIZE >= (128,128,64) per dimension", shown);
            }
        }

        void check_compressed_formats(demo::Version v, demo::Check& check)
        {
            const int n = demo::get_integer(P::NumCompressedTextureFormats);
            std::vector<GLint> formats(static_cast<std::size_t>(std::max(n, 0)));
            if (n > 0) mgl::glGetIntegerv(P::CompressedTextureFormats, formats.data());
            const std::set<GLint> listed(formats.begin(), formats.end());
            // ES 3.0 §3.8.6 / ES 3.2 §8.7 Table 8.17: the ETC2/EAC formats are required; ES 3.2 adds ASTC LDR.
            const GLenum etc2[] = {GL_COMPRESSED_R11_EAC, GL_COMPRESSED_SIGNED_R11_EAC, GL_COMPRESSED_RG11_EAC,
                                   GL_COMPRESSED_SIGNED_RG11_EAC, GL_COMPRESSED_RGB8_ETC2, GL_COMPRESSED_SRGB8_ETC2,
                                   GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
                                   GL_COMPRESSED_RGBA8_ETC2_EAC, GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC};
            check.expect(std::all_of(std::begin(etc2), std::end(etc2), [&](GLenum f) { return listed.count(static_cast<GLint>(f)) != 0; }),
                         "all 10 ETC2/EAC formats are listed in COMPRESSED_TEXTURE_FORMATS");
            if (v >= demo::Version{3, 2})
            {
                int astc = 0;
                for (GLenum f = GL_COMPRESSED_RGBA_ASTC_4x4; f <= GL_COMPRESSED_RGBA_ASTC_12x12; ++f) astc += static_cast<int>(listed.count(static_cast<GLint>(f)));
                for (GLenum f = GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4; f <= GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12; ++f) astc += static_cast<int>(listed.count(static_cast<GLint>(f)));
                check.expect_eq(astc, 28, "all 28 ASTC LDR formats are listed (required by ES 3.2)");
            }
        }

        std::vector<std::pair<std::string, demo::Version>> commands_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "diagnostics-capabilities",
        .title = "Capabilities, limits and extensions",
        .min_es = {3, 0},
        .summary = "Reports version strings, all implementation limits (checked against the specification "
                   "minimums), extensions, compressed formats and meta-gl entry-point availability.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<Capabilities>(argc, argv, info);
}
