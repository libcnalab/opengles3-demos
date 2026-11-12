# Helpers for registering independent demo executables and their CTest runs.
#
#   ogles_add_demo([SOURCES a.cpp b.cpp] [LIBS extra-targets] [ARGS extra-test-args])
#
# Must be called from a demo directory containing demo.json. The executable name, minimum
# OpenGL ES version and test labels are read from demo.json so the manifest stays the single
# source of truth. The executable is placed in <build>/bin/.

function(ogles_compile_options target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
            -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wnull-dereference
            -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wno-missing-field-initializers>
        $<$<AND:$<BOOL:${OGLES3_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang>>:-Werror>)
    if(OGLES3_SANITIZE)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address,undefined -fno-omit-frame-pointer>)
        target_link_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address,undefined>)
    endif()
endfunction()

# Registers one CTest entry per (driver, version-mode) combination.
#   ogles_register_gl_tests(BASE name TARGET exe MIN_ES 3.0 [LABELS ...] [ARGS ...] [EXTRA_DRIVERS ...]
#                           [WINDOW_DRIVERS ...])
# WINDOW_DRIVERS (Mesa software drivers only): run on these drivers in an SDL3 window under xvfb-run instead of a
# surfaceless pbuffer, when OGLES3_XVFB_RUN is set (see CMakeLists.txt). The test names stay the same; the result
# JSON records the platform ("SDL ... window (video driver=x11)").
function(ogles_register_gl_tests)
    cmake_parse_arguments(T "" "BASE;TARGET;MIN_ES" "LABELS;ARGS;EXTRA_DRIVERS;WINDOW_DRIVERS" ${ARGN})
    foreach(wd IN LISTS T_WINDOW_DRIVERS)
        if(NOT wd MATCHES "^(llvmpipe|zink|softpipe)$")
            message(FATAL_ERROR "${T_BASE}: window_drivers may list only Mesa software drivers, not '${wd}'")
        endif()
    endforeach()
    # Command prefix, surface flag and environment of a window-surface run of <driver> (empty prefix: headless).
    macro(ogles_surface driver)
        set(launch "")
        set(surface "--headless")
        set(surface_env "EGL_PLATFORM=surfaceless")
        if(OGLES3_XVFB_RUN AND "${driver}" IN_LIST T_WINDOW_DRIVERS)
            # A private virtual X server per run; the lock keeps xvfb-run's display-number search race-free.
            set(launch "${OGLES3_XVFB_RUN}" -a -s "-screen 0 1280x1024x24")
            set(surface "--window")
            set(surface_env "")
        endif()
    endmacro()
    if(NOT OGLES3_BUILD_TESTS)
        return()
    endif()
    set(modes "es-max")
    # Mesa reports ES 3.2; forcing 3.2 again would only duplicate the es-max run.
    if(OGLES3_TEST_MIN_VERSION AND NOT T_MIN_ES STREQUAL "3.2")
        list(APPEND modes "es-min")
    endif()
    set(drivers ${OGLES3_TEST_DRIVERS})
    if(OGLES3_TEST_SOFTPIPE AND (T_MIN_ES STREQUAL "3.2" OR "softpipe" IN_LIST T_EXTRA_DRIVERS))
        list(APPEND drivers softpipe)
    endif()
    list(REMOVE_DUPLICATES drivers)
    # ANGLE: an independent GL front-end (Vulkan backend). It creates exactly the requested ES version, so its
    # single mode "native" runs every executable at its min_es on a native context, without any version override.
    if(OGLES3_ANGLE_DIR)
        set(shim "${CMAKE_BINARY_DIR}/angle-shim")
        if(NOT EXISTS "${shim}/libEGL.so.1")
            # libEGL.so.1 is the soname the executables link (glvnd); the shim points it at ANGLE's libEGL.
            file(MAKE_DIRECTORY "${shim}")
            file(CREATE_LINK "${OGLES3_ANGLE_DIR}/libEGL.so" "${shim}/libEGL.so.1" SYMBOLIC)
        endif()
        if(NOT EXISTS "${shim}/libGLESv2.so")
            # ANGLE's libEGL dlopens libGLESv2.so from its own (symlinked) directory.
            file(CREATE_LINK "${OGLES3_ANGLE_DIR}/libGLESv2.so" "${shim}/libGLESv2.so" SYMBOLIC)
        endif()
        list(APPEND drivers angle)
    endif()
    # Window-size guard: one llvmpipe run at an odd portrait size. Every demo must lay out and validate at any size
    # (docs/agent-briefs/DEMO_BRIEF.md "Window size"); tools/sweep_sizes.sh covers more sizes on demand.
    if(OGLES3_TEST_SIZE_VARIANT AND "llvmpipe" IN_LIST drivers)
        set(test_name "${T_BASE}:llvmpipe:size-333x777")
        string(REPLACE ":" "_" file_stem "${test_name}")
        ogles_surface(llvmpipe)
        add_test(NAME "${test_name}"
            COMMAND ${launch} $<TARGET_FILE:${T_TARGET}> ${surface} --validate --width 333 --height 777
                    --result-json "${OGLES3_RESULTS_DIR}/${file_stem}.json"
                    --test-id "${test_name}" ${T_ARGS})
        set(size_env ${surface_env} "LIBGL_ALWAYS_SOFTWARE=1" "GALLIUM_DRIVER=llvmpipe")
        if(OGLES3_SANITIZE)
            list(APPEND size_env "ASAN_OPTIONS=detect_leaks=0")
        endif()
        set_tests_properties("${test_name}" PROPERTIES ENVIRONMENT "${size_env}" SKIP_RETURN_CODE 77 TIMEOUT 120
                             LABELS "${T_LABELS};llvmpipe;size")
        if(launch)
            set_property(TEST "${test_name}" APPEND PROPERTY LABELS window)
            set_tests_properties("${test_name}" PROPERTIES RESOURCE_LOCK xvfb-run)
        endif()
    endif()
    foreach(driver IN LISTS drivers)
        set(driver_modes ${modes})
        if(driver STREQUAL "angle")
            # native: ANGLE's highest version (3.1). native-min: a context of exactly min_es (native ES 3.0), only
            # where that differs from native.
            set(driver_modes "native")
            if(T_MIN_ES STREQUAL "3.0")
                list(APPEND driver_modes "native-min")
            endif()
        endif()
        foreach(mode IN LISTS driver_modes)
            set(test_name "${T_BASE}:${driver}:${mode}")
            string(REPLACE ":" "_" file_stem "${test_name}")
            ogles_surface(${driver})
            set(env ${surface_env})
            if(driver STREQUAL "llvmpipe" OR driver STREQUAL "softpipe" OR driver STREQUAL "zink")
                list(APPEND env "LIBGL_ALWAYS_SOFTWARE=1" "GALLIUM_DRIVER=${driver}")
                if(driver STREQUAL "zink")
                    # Pin zink's Vulkan device to lavapipe, never the hardware GPU. Mesa 25's zink already picks a CPU
                    # device under LIBGL_ALWAYS_SOFTWARE, but that is driver behaviour; the ICD list makes it explicit.
                    list(APPEND env "VK_ICD_FILENAMES=${OGLES3_ANGLE_VULKAN_ICD}" "VK_DRIVER_FILES=${OGLES3_ANGLE_VULKAN_ICD}")
                endif()
            elseif(driver STREQUAL "angle")
                # Also preload libGLESv2 (as a native ANGLE application links it); without it eglInitialize fails.
                # Sanitizer builds: the ASan runtime must come first in the initial library list, before the preload.
                set(angle_preload "${OGLES3_ANGLE_DIR}/libGLESv2.so")
                if(OGLES3_SANITIZE)
                    set(angle_preload "${OGLES3_ASAN_RUNTIME}:${angle_preload}")
                endif()
                set(env "OGLES3_EGL_PLATFORM=angle" "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/angle-shim:${OGLES3_ANGLE_DIR}"
                        "LD_PRELOAD=${angle_preload}"
                        "VK_ICD_FILENAMES=${OGLES3_ANGLE_VULKAN_ICD}" "VK_DRIVER_FILES=${OGLES3_ANGLE_VULKAN_ICD}")
            elseif(NOT driver STREQUAL "gpu")
                message(FATAL_ERROR "unknown OGLES3_TEST_DRIVERS entry '${driver}'")
            endif()
            if(mode STREQUAL "native-min")
                list(APPEND env "OGLES3_ANGLE_EXACT_VERSION=1")
            endif()
            if(mode STREQUAL "es-min")
                list(APPEND env "MESA_GLES_VERSION_OVERRIDE=${T_MIN_ES}")
            endif()
            if(OGLES3_SANITIZE)
                # Driver-internal allocations are outside our control; keep ASan for our code.
                list(APPEND env "ASAN_OPTIONS=detect_leaks=0")
            endif()
            add_test(NAME "${test_name}"
                COMMAND ${launch} $<TARGET_FILE:${T_TARGET}> ${surface} --validate
                        --result-json "${OGLES3_RESULTS_DIR}/${file_stem}.json"
                        --capture "${OGLES3_RESULTS_DIR}/${file_stem}.png"
                        --test-id "${test_name}"
                        ${T_ARGS})
            if(driver STREQUAL "angle")
                # With DISPLAY set (even empty) ANGLE picks its X11 path; unset it for the headless Vulkan display.
                set_tests_properties("${test_name}" PROPERTIES ENVIRONMENT_MODIFICATION "DISPLAY=unset:")
            endif()
            set_tests_properties("${test_name}" PROPERTIES
                ENVIRONMENT "${env}"
                SKIP_RETURN_CODE 77
                TIMEOUT 120
                LABELS "${T_LABELS};${driver};${mode}")
            if(launch)
                set_property(TEST "${test_name}" APPEND PROPERTY LABELS window)
                set_tests_properties("${test_name}" PROPERTIES RESOURCE_LOCK xvfb-run)
            endif()
        endforeach()
    endforeach()
endfunction()

function(ogles_add_demo)
    cmake_parse_arguments(D "" "" "SOURCES;LIBS;ARGS" ${ARGN})
    set(manifest "${CMAKE_CURRENT_SOURCE_DIR}/demo.json")
    if(NOT EXISTS "${manifest}")
        message(FATAL_ERROR "ogles_add_demo: ${manifest} missing")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest}")
    file(READ "${manifest}" json)
    string(JSON name GET "${json}" name)
    string(JSON min_es GET "${json}" min_es)
    string(JSON category GET "${json}" category)
    set(extra_drivers "")
    string(JSON n_extra ERROR_VARIABLE no_extra LENGTH "${json}" extra_drivers)
    if(NOT no_extra AND n_extra GREATER 0)
        math(EXPR last "${n_extra} - 1")
        foreach(i RANGE ${last})
            string(JSON d GET "${json}" extra_drivers ${i})
            list(APPEND extra_drivers ${d})
        endforeach()
    endif()
    set(window_drivers "")
    string(JSON n_window ERROR_VARIABLE no_window LENGTH "${json}" window_drivers)
    if(NOT no_window AND n_window GREATER 0)
        math(EXPR last "${n_window} - 1")
        foreach(i RANGE ${last})
            string(JSON d GET "${json}" window_drivers ${i})
            list(APPEND window_drivers ${d})
        endforeach()
    endif()
    if(NOT D_SOURCES)
        set(D_SOURCES main.cpp)
    endif()

    add_executable(${name} ${D_SOURCES})
    target_link_libraries(${name} PRIVATE ogles3::common ${D_LIBS})
    target_compile_definitions(${name} PRIVATE
        DEMO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        DEMO_NAME="${name}")
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    ogles_compile_options(${name})

    ogles_register_gl_tests(BASE ${name} TARGET ${name} MIN_ES ${min_es}
        LABELS "demo;${category}" ARGS ${D_ARGS} EXTRA_DRIVERS ${extra_drivers}
        WINDOW_DRIVERS ${window_drivers})
endfunction()
