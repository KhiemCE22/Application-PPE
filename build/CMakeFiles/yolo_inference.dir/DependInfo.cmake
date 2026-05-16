
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "ASM"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_ASM
  "/home/pi/AI/EdgeVisionRT-tracking/src/asm_kernels.S" "/home/pi/AI/EdgeVisionRT-tracking/build/CMakeFiles/yolo_inference.dir/src/asm_kernels.S.o"
  )
set(CMAKE_ASM_COMPILER_ID "GNU")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_ASM
  "NCNN_VULKAN=1"
  "ZENOH_COMPILER_GCC"
  "ZENOH_C_STANDARD=11"
  "ZENOH_LINUX"
  )

# The include file search paths:
set(CMAKE_ASM_TARGET_INCLUDE_PATH
  "/home/pi/AI/EdgeVisionRT-tracking/include"
  "/home/pi/AI/EdgeVisionRT-tracking/include/bytetrack"
  "/home/pi/AI/EdgeVisionRT-tracking/include/ocsort"
  "zenoh-pico/include"
  "/home/pi/AI/EdgeVisionRT-tracking/deps/zenoh-pico/include"
  "/usr/include/opencv4"
  "/usr/include/eigen3"
  "/home/pi/AI/EdgeVisionRT-tracking/deps/ncnn-vulkan-install/include/ncnn"
  "/usr/local/include"
  "/usr/include/gstreamer-1.0"
  "/usr/include/glib-2.0"
  "/usr/lib/aarch64-linux-gnu/glib-2.0/include"
  "/usr/include/sysprof-6"
  "/usr/include"
  "/usr/include/orc-0.4"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/pi/AI/EdgeVisionRT-tracking/src/benchmark.cpp" "CMakeFiles/yolo_inference.dir/src/benchmark.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/benchmark.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/bytetrack/BYTETracker.cpp" "CMakeFiles/yolo_inference.dir/src/bytetrack/BYTETracker.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/bytetrack/BYTETracker.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/bytetrack/STrack.cpp" "CMakeFiles/yolo_inference.dir/src/bytetrack/STrack.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/bytetrack/STrack.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/bytetrack/kalmanFilter.cpp" "CMakeFiles/yolo_inference.dir/src/bytetrack/kalmanFilter.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/bytetrack/kalmanFilter.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/bytetrack/lapjv.cpp" "CMakeFiles/yolo_inference.dir/src/bytetrack/lapjv.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/bytetrack/lapjv.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/bytetrack/utils.cpp" "CMakeFiles/yolo_inference.dir/src/bytetrack/utils.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/bytetrack/utils.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/inference_engine.cpp" "CMakeFiles/yolo_inference.dir/src/inference_engine.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/inference_engine.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/input_pipeline.cpp" "CMakeFiles/yolo_inference.dir/src/input_pipeline.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/input_pipeline.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/main.cpp" "CMakeFiles/yolo_inference.dir/src/main.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/main.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/neon_preprocess.cpp" "CMakeFiles/yolo_inference.dir/src/neon_preprocess.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/neon_preprocess.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/ocsort/Association.cpp" "CMakeFiles/yolo_inference.dir/src/ocsort/Association.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/ocsort/Association.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/ocsort/KalmanBoxTracker.cpp" "CMakeFiles/yolo_inference.dir/src/ocsort/KalmanBoxTracker.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/ocsort/KalmanBoxTracker.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/ocsort/KalmanFilter.cpp" "CMakeFiles/yolo_inference.dir/src/ocsort/KalmanFilter.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/ocsort/KalmanFilter.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/ocsort/OCSort.cpp" "CMakeFiles/yolo_inference.dir/src/ocsort/OCSort.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/ocsort/OCSort.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/ocsort/Utilities.cpp" "CMakeFiles/yolo_inference.dir/src/ocsort/Utilities.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/ocsort/Utilities.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/ocsort/lapjv.cpp" "CMakeFiles/yolo_inference.dir/src/ocsort/lapjv.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/ocsort/lapjv.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT-tracking/src/postprocess.cpp" "CMakeFiles/yolo_inference.dir/src/postprocess.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/postprocess.cpp.o.d"
  "" "yolo_inference" "gcc" "CMakeFiles/yolo_inference.dir/link.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
