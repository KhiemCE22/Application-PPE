cmake_minimum_required(VERSION 3.16)
project(yolo_inference VERSION 1.0.0 LANGUAGES C CXX ASM)

# ============================================================================
# 1. Build Configuration for Raspberry Pi 5 (Cortex-A76)
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Cortex-A76 (ARMv8.2-A with FP16 and Dot Product)
set(CORTEX_A76_FLAGS "-march=armv8.2-a+fp16+dotprod -mtune=cortex-a76")
# Warning flags
set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wshadow -Wnull-dereference -Wformat=2 -Wno-unused-parameter")

# Release build flags
# NOTE: -flto is removed from here to prevent symbol shadowing in FFmpeg
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CORTEX_A76_FLAGS} ${WARNING_FLAGS} -O3 -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CORTEX_A76_FLAGS} ${WARNING_FLAGS} -O3 -ffast-math -DNDEBUG")

# Debug build flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CORTEX_A76_FLAGS} -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CORTEX_A76_FLAGS} -O0 -g -DDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================================
# 2. Find Dependencies using Modern CMake Targets
# ============================================================================

# --- NCNN Inference Engine ---
if(NCNN_DIR)
    find_package(ncnn REQUIRED PATHS ${NCNN_DIR})
else()
    find_package(ncnn REQUIRED)
endif()

# --- Vulkan for GPU Acceleration ---
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    add_definitions(-DNCNN_VULKAN=1)
    set(NCNN_HAS_VULKAN ON)
else()
    add_definitions(-DNCNN_VULKAN=0)
    set(NCNN_HAS_VULKAN OFF)
endif()

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)

# --- FFMPEG via PkgConfig (Using IMPORTED_TARGET for better symbol resolution) ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(SWSCALE REQUIRED IMPORTED_TARGET libswscale)
pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)

# --- GStreamer ---
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

# --- Eigen3 & Threads ---
find_package(Eigen3 3.3 REQUIRED)
find_package(Threads REQUIRED)

# --- Zenoh-pico ----
set(ZENOH_PICO_DIR "${CMAKE_SOURCE_DIR}/deps/zenoh-pico")
if(EXISTS "${ZENOH_PICO_DIR}/CMakeLists.txt")
    set(ZENOH_PICO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${ZENOH_PICO_DIR}" zenoh-pico)
else()
    message(FATAL_ERROR "Zenoh-pico not found at ${ZENOH_PICO_DIR}. Please clone it first.")
endif()
# ============================================================================
# 3. Source Files Configuration
# ============================================================================

set(SOURCES
    src/main.cpp
    src/neon_preprocess.cpp
    src/input_pipeline.cpp
    src/inference_engine.cpp
    src/postprocess.cpp
    src/benchmark.cpp
)

# ByteTrack Core
list(APPEND SOURCES 
    src/bytetrack/BYTETracker.cpp src/bytetrack/STrack.cpp 
    src/bytetrack/kalmanFilter.cpp src/bytetrack/lapjv.cpp src/bytetrack/utils.cpp
)

# OC-SORT Core
list(APPEND SOURCES 
    src/ocsort/OCSort.cpp src/ocsort/KalmanBoxTracker.cpp 
    src/ocsort/KalmanFilter.cpp src/ocsort/Association.cpp 
    src/ocsort/lapjv.cpp src/ocsort/Utilities.cpp
)

# Assembly Kernels
set(ASM_SOURCES src/asm_kernels.S)
set_source_files_properties(${ASM_SOURCES} PROPERTIES LANGUAGE ASM COMPILE_FLAGS "-x assembler-with-cpp")

# ============================================================================
# 4. Build Target & Linking (CRITICAL FIX)
# ============================================================================

add_executable(yolo_inference ${SOURCES} ${ASM_SOURCES} )

# Include Directories
target_include_directories(yolo_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/bytetrack
    ${CMAKE_SOURCE_DIR}/include/ocsort
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Link Libraries
# IMPORTANT: We use PkgConfig:: targets and force link order
# We also add 'atomic' and 'm' which are often missing for FFmpeg on ARM64
target_link_libraries(yolo_inference PRIVATE
    ncnn
    ${OpenCV_LIBS}
    Threads::Threads
    Eigen3::Eigen
    PkgConfig::AVFORMAT
    PkgConfig::AVCODEC
    PkgConfig::SWSCALE
    PkgConfig::SWRESAMPLE
    PkgConfig::AVUTIL
    PkgConfig::GST_APP
    PkgConfig::GSTREAMER
    PkgConfig::GST_VIDEO
    zenohpico_shared
    m
    z
    dl
    atomic # Required for 64-bit atomics in FFmpeg BSF on some ARM builds
)

# Link Vulkan
if(Vulkan_FOUND AND NCNN_HAS_VULKAN)
    target_link_libraries(yolo_inference PRIVATE ${Vulkan_LIBRARIES})
endif()

# Disable Global LTO for this target to fix the 'undefined reference' issue
# LTO often strips or obscures C symbols when mixed with C++ on ARM64
set_target_properties(yolo_inference PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

# ============================================================================
# 5. Installation & Deployment
# ============================================================================

install(TARGETS yolo_inference RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/ DESTINATION share/yolo_inference/models)

message(STATUS "------------------------------------------------")
message(STATUS "Build Configuration Summary:")
message(STATUS "  Optimization: Cortex-A76 (DotProd, FP16)")
message(STATUS "  FFmpeg:       ${AVCODEC_VERSION} (BSF Enabled)")
message(STATUS "  GStreamer:    ${GSTREAMER_VERSION}")
message(STATUS "  Linker Fix:   LTO Disabled for Symbol Visibility")
message(STATUS "------------------------------------------------")
