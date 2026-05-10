#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== STARTING NCNN QUANTIZATION PIPELINE ==="

# ==========================================
# 1. CONFIGURATION VARIABLES
# ==========================================
# Input files (Original FP32 model)
INPUT_PARAM="models/PPE_model/model2.ncnn.param"
INPUT_BIN="models/PPE_model/model2.ncnn.bin"

# Intermediate files (After optimization)
OPT_PARAM="models/PPE_model/model-opt.param"
OPT_BIN="models/PPE_model/model-opt.bin"

# Calibration image list and output table file
IMAGELIST="imagelist.txt"
TABLE_FILE="models/PPE_model/model.table"

# Final output files (INT8 model)
INT8_PARAM="models/PPE_model/model.int8.param"
INT8_BIN="models/PPE_model/model.int8.bin"

# Calibration settings for ncnn2table
# CRITICAL: These must match your C++ inference preprocessing exactly!
# The default values below are typical for YOLO models (RGB, 1/255 scale, 640x640)
MEAN="0,0,0"
NORM="0.0039215,0.0039215,0.0039215" 
SHAPE="640,640,3"
PIXEL="RGB"
THREAD="4"
METHOD="kl" # Calibration method: 'kl' (Kullback-Leibler) or 'aciq'

# ==========================================
# 2. EXECUTION PIPELINE
# ==========================================

# Step 1: Optimize the FP32 model
echo "[1/3] Running ncnnoptimize..."
if [ ! -f "$INPUT_PARAM" ] || [ ! -f "$INPUT_BIN" ]; then
    echo "ERROR: Input files $INPUT_PARAM or $INPUT_BIN not found!"
    exit 1
fi
# The '0' at the end specifies FP32 optimization
./deps/ncnn-vulkan-install/bin/ncnnoptimize "$INPUT_PARAM" "$INPUT_BIN" "$OPT_PARAM" "$OPT_BIN" 0
echo "=> Optimization successful: $OPT_PARAM, $OPT_BIN"


# Step 2: Generate Calibration Table
echo "[2/3] Running ncnn2table..."
if [ ! -f "$IMAGELIST" ]; then
    echo "ERROR: $IMAGELIST not found! Please create a text file containing paths to calibration images."
    exit 1
fi
./deps/ncnn-vulkan-install/bin/ncnn2table "$OPT_PARAM" "$OPT_BIN" "$IMAGELIST" "$TABLE_FILE" mean="$MEAN" norm="$NORM" shape="$SHAPE" pixel="$PIXEL" thread="$THREAD" method="$METHOD"
echo "=> Calibration table generated: $TABLE_FILE"


# Step 3: Quantize to INT8
echo "[3/3] Running ncnn2int8..."
./deps/ncnn-vulkan-install/bin/ncnn2int8 "$OPT_PARAM" "$OPT_BIN" "$INT8_PARAM" "$INT8_BIN" "$TABLE_FILE"
echo "=> Quantization successful: $INT8_PARAM, $INT8_BIN"

echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="
echo "Your INT8 model files are ready for deployment."