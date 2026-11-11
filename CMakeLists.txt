cmake_minimum_required(VERSION 3.23)

project(opengles3-demos
    VERSION 0.1.0
    DESCRIPTION "Educational and validation suite for OpenGL ES 3.0/3.1/3.2 through meta-gl"
    LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# --- Options -------------------------------------------------------------------------------
option(OGLES3_WERROR "Treat compiler warnings as errors" OFF)
option(OGLES3_SANITIZE "Enable ASan + UBSan for CPU-side code (also enables METAGL_SANITIZE)" OFF)
option(OGLES3_WITH_SDL3 "Build windowed (SDL3) mode when SDL3 is available" ON)
option(OGLES3_BUILD_TESTS "Register CTest runs of every demo" ON)
option(OGLES3_USE_INSTALLED_METAGL "Use find_package(meta-gl) instead of a source checkout" OFF)

# Test matrix. Each demo is executed headlessly once per driver and per version mode:
#   gpu      - the default EGL driver of the machine (hardware when available)
#   llvmpipe - Mesa LLVM software rasterizer (ES 3.2)
#   zink     - Mesa GL-on-Vulkan (on lavapipe): an independent backend translating to SPIR-V
#   softpipe - Mesa reference rasterizer; natively ES 3.1 / GLSL ES 3.10. Not in the default list:
#              it is added automatically for min_es 3.2 executables (which must SKIP there) and for
#              executables whose demo.json lists it in "extra_drivers" (version-gating tests).
#   es-max   - the highest ES version the driver offers
#   es-min   - the demo's minimum version via MESA_GLES_VERSION_OVERRIDE (API version only, see
#              docs/VALIDATION.md)
set(OGLES3_TEST_DRIVERS "gpu;llvmpipe;zink" CACHE STRING "Driver variants for CTest runs (gpu;llvmpipe;zink;softpipe)")
option(OGLES3_TEST_SOFTPIPE "Register softpipe runs for min_es 3.2 executables and opted-in executables" ON)
set(OGLES3_ANGLE_DIR "" CACHE PATH "Directory with an ANGLE libEGL.so/libGLESv2.so (Vulkan backend); adds an 'angle' driver to the CTest matrix")
if(NOT OGLES3_ANGLE_DIR AND EXISTS "$ENV{HOME}/Android/Sdk/emulator/lib64/gles_angle/libEGL.so")
    set(OGLES3_ANGLE_DIR "$ENV{HOME}/Android/Sdk/emulator/lib64/gles_angle" CACHE PATH "" FORCE)
endif()
option(OGLES3_TEST_MIN_VERSION "Also run every demo on a context limited to its minimum ES version (Mesa only)" ON)

# --- meta-gl -------------------------------------------------------------------------------
# meta-gl is consumed exactly as its README documents: either as a sibling source checkout
# via add_subdirectory(), or as an installed CMake package. Its build tree lives inside our
# build directory, so configuring this project never writes into the meta-gl checkout.
set(OGLES3_METAGL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libcna/meta-gl"
    CACHE PATH "Path to the meta-gl source checkout")

if(OGLES3_SANITIZE)
    set(METAGL_SANITIZE ON CACHE BOOL "" FORCE)
endif()

if(OGLES3_USE_INSTALLED_METAGL)
    find_package(meta-gl 0.4 CONFIG REQUIRED)
else()
    if(NOT EXISTS "${OGLES3_METAGL_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "meta-gl not found at '${OGLES3_METAGL_SOURCE_DIR}'. "
            "Set -DOGLES3_METAGL_SOURCE_DIR=<path> or -DOGLES3_USE_INSTALLED_METAGL=ON.")
    endif()
    # Never enable meta-gl's per-call debug logging here: it inserts glGetError() after every
    # wrapper and would consume the error flags that our negative tests inspect.
    set(METAGL_ENABLE_DEBUG_LOGGING OFF CACHE BOOL "" FORCE)
    add_subdirectory("${OGLES3_METAGL_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/meta-gl" EXCLUDE_FROM_ALL)
endif()

# --- Platform dependencies -----------------------------------------------------------------
find_package(OpenGL REQUIRED COMPONENTS EGL)   # provides OpenGL::EGL (libEGL only)
# Note: we deliberately do NOT link libGLESv2. Every GL entry point is reached through
# meta-gl's function table, loaded with eglGetProcAddress / SDL_GL_GetProcAddress.
# Combined with GL_GLES_PROTOTYPES=0 this makes an accidental direct gl* call a build error.

set(OGLES3_HAVE_SDL3 OFF)
if(OGLES3_WITH_SDL3)
    find_package(SDL3 CONFIG QUIET)
    if(TARGET SDL3::SDL3)
        set(OGLES3_HAVE_SDL3 ON)
    else()
        message(STATUS "opengles3-demos: SDL3 not found - windowed mode disabled, headless EGL only")
    endif()
endif()
message(STATUS "opengles3-demos: SDL3 windowed mode: ${OGLES3_HAVE_SDL3}")

find_package(Python3 COMPONENTS Interpreter)

include(cmake/OglesDemo.cmake)

add_subdirectory(common)

if(OGLES3_BUILD_TESTS)
    enable_testing()
    set(OGLES3_RESULTS_DIR "${CMAKE_BINARY_DIR}/test-results")
    file(MAKE_DIRECTORY "${OGLES3_RESULTS_DIR}")
endif()

add_subdirectory(demos)
add_subdirectory(shader-tests)
add_subdirectory(tests)

# Convenience target: regenerate coverage documents from manifests + latest test results.
if(Python3_Interpreter_FOUND)
    add_custom_target(coverage
        COMMAND Python3::Interpreter -I "${CMAKE_CURRENT_SOURCE_DIR}/tools/coverage/generate_coverage.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --results "${CMAKE_BINARY_DIR}/test-results"
                --metagl "${OGLES3_METAGL_SOURCE_DIR}"
        COMMENT "Regenerating docs/API_COVERAGE.csv and coverage summaries"
        VERBATIM)
endif()
