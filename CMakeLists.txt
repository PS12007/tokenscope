cmake_minimum_required(VERSION 3.16)
project(tokenscope CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OFF by default, always. A profiler that requires opting out is a profiler that
# gets quietly patched out downstream. See docs/01 section 10.
option(TOKENSCOPE_ENABLED "build with tokenscope instrumentation" OFF)
option(TOKENSCOPE_TSC     "use rdtsc instead of steady_clock (opt-in, see docs/01)" OFF)

add_library(tokenscope STATIC src/tokenscope.cpp)
target_include_directories(tokenscope PUBLIC src)
target_compile_definitions(tokenscope PUBLIC TOKENSCOPE_STATIC)

if (TOKENSCOPE_ENABLED)
    target_compile_definitions(tokenscope PUBLIC TOKENSCOPE_ENABLED)
endif()
if (TOKENSCOPE_TSC)
    target_compile_definitions(tokenscope PUBLIC TOKENSCOPE_TSC)
endif()

find_package(Threads REQUIRED)
target_link_libraries(tokenscope PUBLIC Threads::Threads)

add_executable(ts_selftest src/ts_selftest.cpp)
target_link_libraries(ts_selftest PRIVATE tokenscope)

enable_testing()
add_test(NAME selftest COMMAND ts_selftest)
