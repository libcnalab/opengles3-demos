# Every directory below demos/<group>/<demo>/ that contains a CMakeLists.txt is one
# independent executable. Discovery keeps this file free of per-demo edits, so demos can be
# added in parallel (separate branches/worktrees) without touching shared infrastructure.
file(GLOB demo_groups LIST_DIRECTORIES true RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
list(SORT demo_groups)
foreach(group IN LISTS demo_groups)
    if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${group}")
        continue()
    endif()
    file(GLOB demo_dirs LIST_DIRECTORIES true RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${group}/*")
    list(SORT demo_dirs)
    foreach(dir IN LISTS demo_dirs)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
            add_subdirectory("${dir}")
        endif()
    endforeach()
endforeach()
