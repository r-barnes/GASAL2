cmake_minimum_required(VERSION 3.13)

project(gasal2 LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)

#TODO: Probably remove this before release since a standard user won't recompile
#many times
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  )
endif()

add_library(gasal2
  src/args_parser.cpp
  src/ctors.cpp
  src/gasal_align.cu
  src/host_batch.cpp
  src/interfaces.cpp
  src/read_fasta.cpp
  src/res.cpp

  src/kernels/banded.cu
  src/kernels/pack_rc_seqs.cu
)
target_include_directories(gasal2 PUBLIC include src)

set(MAX_QUERY_LEN "1024" CACHE STRING "Maximum query length")
set(N_CODE "0x4E" CACHE STRING "Code for non-based value N")

message(STATUS "ahmed2019: MAX_QUERY_LEN=${MAX_QUERY_LEN}")
message(STATUS "ahmed2019: N_CODE=${N_CODE}")

add_compile_definitions(MAX_QUERY_LEN=${MAX_QUERY_LEN})
add_compile_definitions(N_CODE=${N_CODE})
if(DEFINED N_PENALTY)
  add_compile_definitions(N_PENALTY=${N_PENALTY})
endif()

if(DEFINED GPU_SM_ARCH)
  target_compile_options(gasal2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

if(DEFINED GPU_COMPUTE_ARCH)
  target_compile_options(gasal2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_COMPUTE_ARCH}>)
endif()

set_target_properties(gasal2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)



add_executable(test_prog test_prog/test_prog.cpp)

target_link_libraries(test_prog PRIVATE OpenMP::OpenMP_CXX gasal2)

file(COPY test_prog/run_tests.sh DESTINATION ${CMAKE_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/test_prog/query_batch.fasta.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/test_prog/target_batch.fasta.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
)



add_executable(unittests
  unittests/test_main.cpp
  unittests/packing.cu
  unittests/test_timer.cpp
  unittests/test_read_fasta.cpp
)
target_link_libraries(unittests PRIVATE gasal2)

configure_file(unittests/test.fasta ${CMAKE_BINARY_DIR}/test.fasta COPYONLY)