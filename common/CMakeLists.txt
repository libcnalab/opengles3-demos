add_library(ogles3-common STATIC
    src/runtime.cpp
    src/check.cpp
    src/gl_util.cpp
    src/image.cpp
    src/platform_egl.cpp
    $<$<BOOL:${OGLES3_HAVE_SDL3}>:src/platform_sdl3.cpp>)
add_library(ogles3::common ALIAS ogles3-common)

target_include_directories(ogles3-common PUBLIC include PRIVATE src)
target_compile_features(ogles3-common PUBLIC cxx_std_23)
target_link_libraries(ogles3-common
    PUBLIC meta-gl::meta-gl
    PRIVATE OpenGL::EGL $<$<BOOL:${OGLES3_HAVE_SDL3}>:SDL3::SDL3>)
target_compile_definitions(ogles3-common PRIVATE OGLES3_HAVE_SDL3=$<BOOL:${OGLES3_HAVE_SDL3}>)
ogles_compile_options(ogles3-common)
