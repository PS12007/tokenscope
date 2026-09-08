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

# ---------------------------------------------------------------------------
# The shared-library arm.
#
# FINDINGS F18 -- ggml-cpu.dll failing to link against a thread-local that MSVC
# will not export -- was found by hand, a session after the code that caused it,
# because nothing here ever built a DLL. This target exists so that never
# happens again. It is deliberately NOT gated on the platform: the failure was
# MSVC-specific, but a fix that only works on MSVC is not a fix.
#
# Two separate binaries, each including tokenscope.h and so each compiling its
# own `ts_tls` cache, over one exported registry. That is the F22 arrangement in
# miniature, and it needs no llama.cpp, no model and no network.
# ---------------------------------------------------------------------------
if (TOKENSCOPE_ENABLED AND BUILD_SHARED_LIBS)
    # Exports every symbol rather than maintaining a .def file. tokenscope's own
    # exports are already annotated with TS_API; this is for ts_dllmod's.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

    # Note the absence of TOKENSCOPE_STATIC: this is the configuration where
    # TS_API becomes dllexport/dllimport and where F18 reproduced.
    add_library(tokenscope_shared SHARED src/tokenscope.cpp)
    target_include_directories(tokenscope_shared PUBLIC src)
    target_compile_definitions(tokenscope_shared PUBLIC TOKENSCOPE_ENABLED)
    if (TOKENSCOPE_TSC)
        target_compile_definitions(tokenscope_shared PUBLIC TOKENSCOPE_TSC)
    endif()
    target_link_libraries(tokenscope_shared PUBLIC Threads::Threads)
    set_target_properties(tokenscope_shared PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        CXX_VISIBILITY_PRESET default)

    # The second module. Separate binary, separate cache.
    add_library(ts_dllmod SHARED src/ts_dllmod.cpp)
    target_link_libraries(ts_dllmod PRIVATE tokenscope_shared)
    set_target_properties(ts_dllmod PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

    # A third module, same source, deliberately NOT linked to the executable.
    # ts_dlltest opens it with LoadLibrary/dlopen partway through the run, which
    # is the shape llama.cpp's GGML_BACKEND_DL uses -- a module arriving after
    # threads already exist and have already recorded. That is the case where a
    # cache-based design is most likely to be wrong, so it gets its own test.
    add_library(ts_dynmod MODULE src/ts_dllmod.cpp)
    target_link_libraries(ts_dynmod PRIVATE tokenscope_shared)
    set_target_properties(ts_dynmod PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        PREFIX "")

    add_executable(ts_dlltest src/ts_dlltest.cpp)
    target_link_libraries(ts_dlltest PRIVATE tokenscope_shared ts_dllmod ${CMAKE_DL_LIBS})
    add_dependencies(ts_dlltest ts_dynmod)

    add_test(NAME dlltest COMMAND ts_dlltest dlltest.trace.json
             $<TARGET_FILE:ts_dynmod>)
endif()
