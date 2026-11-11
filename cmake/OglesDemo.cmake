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
#   ogles_register_gl_tests(BASE name TARGET exe MIN_ES 3.0 [LABELS ...] [ARGS ...])
function(ogles_register_gl_tests)
    cmake_parse_arguments(T "" "BASE;TARGET;MIN_ES" "LABELS;ARGS" ${ARGN})
    if(NOT OGLES3_BUILD_TESTS)
        return()
    endif()
    set(modes "es-max")
    # Mesa reports ES 3.2; forcing 3.2 again would only duplicate the es-max run.
    if(OGLES3_TEST_MIN_VERSION AND NOT T_MIN_ES STREQUAL "3.2")
        list(APPEND modes "es-min")
    endif()
    foreach(driver IN LISTS OGLES3_TEST_DRIVERS)
        foreach(mode IN LISTS modes)
            set(test_name "${T_BASE}:${driver}:${mode}")
            string(REPLACE ":" "_" file_stem "${test_name}")
            set(env "EGL_PLATFORM=surfaceless")
            if(driver STREQUAL "llvmpipe" OR driver STREQUAL "softpipe" OR driver STREQUAL "zink")
                list(APPEND env "LIBGL_ALWAYS_SOFTWARE=1" "GALLIUM_DRIVER=${driver}")
            elseif(NOT driver STREQUAL "gpu")
                message(FATAL_ERROR "unknown OGLES3_TEST_DRIVERS entry '${driver}'")
            endif()
            if(mode STREQUAL "es-min")
                list(APPEND env "MESA_GLES_VERSION_OVERRIDE=${T_MIN_ES}")
            endif()
            if(OGLES3_SANITIZE)
                # Driver-internal allocations are outside our control; keep ASan for our code.
                list(APPEND env "ASAN_OPTIONS=detect_leaks=0")
            endif()
            add_test(NAME "${test_name}"
                COMMAND $<TARGET_FILE:${T_TARGET}> --headless --validate
                        --result-json "${OGLES3_RESULTS_DIR}/${file_stem}.json"
                        --capture "${OGLES3_RESULTS_DIR}/${file_stem}.png"
                        --test-id "${test_name}"
                        ${T_ARGS})
            set_tests_properties("${test_name}" PROPERTIES
                ENVIRONMENT "${env}"
                SKIP_RETURN_CODE 77
                TIMEOUT 120
                LABELS "${T_LABELS};${driver};${mode}")
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
        LABELS "demo;${category}" ARGS ${D_ARGS})
endfunction()
