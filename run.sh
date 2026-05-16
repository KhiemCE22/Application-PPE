#!/bin/bash
#
# Quick Run Script for EdgeVision RT (Modified for RTSP)
#

set -e

# Fix runtime directory permissions if needed
RUNTIME_DIR="/run/user/$(id -u)"
if [ -d "$RUNTIME_DIR" ]; then
    CURRENT_PERMS=$(stat -c "%a" "$RUNTIME_DIR")
    if [ "$CURRENT_PERMS" != "700" ]; then
        sudo chmod 0700 "$RUNTIME_DIR" 2>/dev/null || true
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/yolo_inference"
#VIDEO="${SCRIPT_DIR}/video/input/PPE_Part1.mp4"
#VIDEO="${SCRIPT_DIR}/video/input/MOT17-02-SDP.mp4"
#VIDEO="${SCRIPT_DIR}/video/input/192_168_2_202_seq_0.mp4"
VIDEO="${SCRIPT_DIR}/video/input/gemini_1.mp4"
# =================================================================
# MODEL PATHS (RESTORED TO ORIGINAL DEFAULT)
# =================================================================

# Default: FP32 model
#PARAM="${SCRIPT_DIR}/models/PPE_model/model.ncnn.param"
#BIN="${SCRIPT_DIR}/models/PPE_model/model.ncnn.bin"

#PARAM="${SCRIPT_DIR}/models/benchmarkYolov8n/FP16model.ncnn.param"
#BIN="${SCRIPT_DIR}/models/benchmarkYolov8n/FP16model.ncnn.bin"

#PARAM="${SCRIPT_DIR}/models/benchmarkYolov8n/ORI-FP32-model.ncnn.param"
#BIN="${SCRIPT_DIR}/models/benchmarkYolov8n/ORI-FP32-model.ncnn.bin"

PARAM="${SCRIPT_DIR}/models/kaggle/working/best_ncnn_model/model.ncnn.param"
BIN="${SCRIPT_DIR}/models/kaggle/working/best_ncnn_model/model.ncnn.bin"



# INT8 model paths
PARAM_INT8="${SCRIPT_DIR}/models/PPE_model/khiem.int8.param"
BIN_INT8="${SCRIPT_DIR}/models/PPE_model/khiem.int8.bin"

# FP16 model paths
#PARAM_FP16="${SCRIPT_DIR}/models/PPE_model/model3.ncnn.param"
#BIN_FP16="${SCRIPT_DIR}/models/PPE_model/model3.ncnn.bin"
#PARAM="${SCRIPT_DIR}/models/PPE_model/ori-640.ncnn.param"
#BIN="${SCRIPT_DIR}/models/PPE_model/ori-640.ncnn.bin"
PARAM_FP16="${SCRIPT_DIR}/models/benchmarkYolov8n/ORI-FP32-model.param"
BIN_FP16="${SCRIPT_DIR}/models/benchmarkYolov8n/ORI-FP32-model.bin"


# =================================================================

# Check if binary exists
if [ ! -f "${BINARY}" ]; then
    echo "Error: Binary not found. Run ./build.sh first"
    exit 1
fi

# Default frames count
FRAMES=2000

# Flags
ENABLE_DISPLAY=false
ENABLE_FB=false
ENABLE_VIDEO=false
ENABLE_CAM=false
ENABLE_RTSP=false      # <--- New Flag
ENABLE_CLASS_FILTER=false
ENABLE_INT8=false
ENABLE_FP16=false
ENABLE_VULKAN=false
VIDEO_OUTPUT=""
CLASS_FILTER=""
CAMERA_DEVICE="/dev/video0"
RTSP_URL=""            # <--- New Variable

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        help|--help|-h)
            echo "Usage: ./run.sh [options]"
            echo "  rtsp \"rtsp://...\"          # Run with RTSP Stream"
            echo "  cam                        # Run with Camera"
            echo "  display                    # Enable display"
            exit 0
            ;;
        int8)
            ENABLE_INT8=true
            shift
            ;;
        fp16)
            ENABLE_FP16=true
            shift
            ;;
        vulkan)
            ENABLE_VULKAN=true
            shift
            ;;
        display)
            ENABLE_DISPLAY=true
            shift
            ;;
        cam|camera|webcam)
            ENABLE_CAM=true
            if [[ $# -gt 1 && "$2" == /dev/* ]]; then
                CAMERA_DEVICE="$2"
                shift
            fi
            shift
            ;;
        rtsp)  # <--- Logic RTSP
            ENABLE_RTSP=true
            if [[ $# -gt 1 && "$2" == rtsp://* ]]; then
                RTSP_URL="$2"
                shift 2
            else
                echo "Error: RTSP URL must start with 'rtsp://'"
                exit 1
            fi
            ;;
        fb)
            ENABLE_FB=true
            shift
            ;;
        video)
            ENABLE_VIDEO=true
            VIDEO_OUTPUT="${2:-output.mp4}"
            shift 2
            ;;
        class)
            ENABLE_CLASS_FILTER=true
            CLASS_FILTER="${2:-person}"
            shift 2
            ;;
        all)
            ENABLE_DISPLAY=true
            ENABLE_VIDEO=true
            VIDEO_OUTPUT="${2:-output_all.mp4}"
            shift
            if [[ $# -gt 0 && "$1" != "class" && "$1" != "display" ]]; then
                shift
            fi
            ;;
        benchmark|"")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Select Model Logic
if [ "$ENABLE_INT8" = true ]; then
    if [ -f "${PARAM_INT8}" ] && [ -f "${BIN_INT8}" ]; then
        PARAM="${PARAM_INT8}"
        BIN="${BIN_INT8}"
    else
        echo "Warning: INT8 model not found, falling back to FP32"
        ENABLE_INT8=false
    fi
elif [ "$ENABLE_FP16" = true ]; then
    if [ -f "${PARAM_FP16}" ] && [ -f "${BIN_FP16}" ]; then
        PARAM="${PARAM_FP16}"
        BIN="${BIN_FP16}"
    else
        echo "Warning: FP16 model not found, falling back to FP32"
        ENABLE_FP16=false
    fi
fi

# Check Model Existence
if [ ! -f "${PARAM}" ] || [ ! -f "${BIN}" ]; then
    echo "Error: Model files not found at: ${PARAM}"
    echo "Check 'models/yolov8n_ncnn_model/' folder."
    exit 1
fi

# Build ARGS
if [ "$ENABLE_CAM" = true ]; then
    ARGS=(
        "--camera" "${CAMERA_DEVICE}"
        "--param" "${PARAM}"
        "--bin" "${BIN}"
    )
    MODE_DESC="CAM"
    FRAMES=0
elif [ "$ENABLE_RTSP" = true ]; then
    ARGS=(
        "--rtsp" "${RTSP_URL}"
        "--param" "${PARAM}"
        "--bin" "${BIN}"
    )
    MODE_DESC="RTSP"
    FRAMES=0
else
    ARGS=(
        "--video" "${VIDEO}"
        "--param" "${PARAM}"
        "--bin" "${BIN}"
    )
    MODE_DESC="FILE"
fi

# Append other flags
if [ "$ENABLE_INT8" = true ]; then ARGS+=("--int8"); MODE_DESC="${MODE_DESC}+INT8"; fi
if [ "$ENABLE_FP16" = true ]; then MODE_DESC="${MODE_DESC}+FP16"; fi
if [ "$ENABLE_VULKAN" = true ]; then ARGS+=("--vulkan"); MODE_DESC="${MODE_DESC}+Vulkan"; fi
if [ "$ENABLE_DISPLAY" = true ]; then export DISPLAY=:0; ARGS+=("--display"); MODE_DESC="${MODE_DESC}+Display"; fi
if [ "$ENABLE_FB" = true ]; then ARGS+=("--fb"); MODE_DESC="${MODE_DESC}+Framebuffer"; fi
if [ "$ENABLE_VIDEO" = true ]; then ARGS+=("--output-video" "${VIDEO_OUTPUT}"); FRAMES=0; fi
if [ "$ENABLE_CLASS_FILTER" = true ]; then ARGS+=("--class" "${CLASS_FILTER}"); fi
if [ "$FRAMES" -ne 0 ]; then ARGS+=("--frames" "${FRAMES}"); fi

# Run
echo "Mode: ${MODE_DESC}"
if [ "$ENABLE_RTSP" = true ]; then echo "Source: ${RTSP_URL}"; fi
echo "Model: ${PARAM}"

if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
fi

if [ "$ENABLE_DISPLAY" = true ]; then unset OMP_NUM_THREADS; else export OMP_NUM_THREADS=4; fi

exec "${BINARY}" "${ARGS[@]}"
