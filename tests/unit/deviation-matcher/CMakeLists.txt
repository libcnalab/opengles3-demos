# CPU-only unit test of the deviation matcher (no GL context, no driver matrix, no demo.json: it claims no coverage).
add_executable(unit-deviation-matcher main.cpp)
target_link_libraries(unit-deviation-matcher PRIVATE ogles3::common)
set_target_properties(unit-deviation-matcher PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
ogles_compile_options(unit-deviation-matcher)
if(OGLES3_BUILD_TESTS)
    add_test(NAME unit-deviation-matcher COMMAND unit-deviation-matcher)
    set_tests_properties(unit-deviation-matcher PROPERTIES TIMEOUT 30 LABELS "unit")
endif()
