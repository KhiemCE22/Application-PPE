#!/bin/bash

# Configuration files
YAML_FILE="metadata.yaml"
HEADER_FILE="./include/common.h"

# Check if metadata file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: $YAML_FILE not found!"
    exit 1
fi

echo "--- Advanced Parsing of $YAML_FILE ---"

# 1. Extract MODEL_SIZE (Robust logic for list or bracket format)
MODEL_SIZE=$(sed -n '/imgsz:/,/names:/p' "$YAML_FILE" | grep -m 1 "[0-9]\{2,\}" | sed 's/[^0-9]//g')
if [ -z "$MODEL_SIZE" ]; then MODEL_SIZE=640; fi
echo "[INFO] MODEL_SIZE: $MODEL_SIZE"

# 2. Extract CLASS_NAMES and count them
# Get raw names, remove quotes/whitespace
RAW_NAMES=$(grep -A 100 "names:" "$YAML_FILE" | grep "^  [0-9]\+:" | sed "s/.*: //" | sed "s/['\"]//g" | tr -d '\r')
NUM_CLASSES=$(echo "$RAW_NAMES" | wc -l)

echo "[INFO] NUM_CLASSES detected: $NUM_CLASSES"
echo "[INFO] Class list:"
# Print each class name to terminal with index
i=0
while read -r name; do
    echo "    $i: $name"
    ((i++))
done <<< "$RAW_NAMES"

# Format for C++ inline array: "name1", "name2"
FORMATTED_NAMES=$(echo "$RAW_NAMES" | sed 's/.sort.*/"&"/' | sed 's/.*/"&"/' | paste -sd "," -)
ARRAY_LINE="inline const char* CLASS_NAMES[] = {$FORMATTED_NAMES};"

# 3. Detect Architecture and Calculate NUM_OUTPUTS
DESC=$(grep "description:" "$YAML_FILE" | tr '[:upper:]' '[:lower:]')
MAX_STRIDE=$(grep "stride:" "$YAML_FILE" | awk '{print $2}')
if [ -z "$MAX_STRIDE" ]; then MAX_STRIDE=32; fi

if [[ "$DESC" == *"sfchd"* ]] && [ "$MAX_STRIDE" -eq 16 ]; then
    echo "[INFO] Architecture detected: sfchd (P2-P4)"
    G1=$(( (MODEL_SIZE / 4) * (MODEL_SIZE / 4) ))
    G2=$(( (MODEL_SIZE / 8) * (MODEL_SIZE / 8) ))
    G3=$(( (MODEL_SIZE / 16) * (MODEL_SIZE / 16) ))
    NUM_OUTPUTS=$(( G1 + G2 + G3 ))
else
    echo "[INFO] Architecture detected: Standard (P3-P5)"
    G1=$(( (MODEL_SIZE / 8) * (MODEL_SIZE / 8) ))
    G2=$(( (MODEL_SIZE / 16) * (MODEL_SIZE / 16) ))
    G3=$(( (MODEL_SIZE / 32) * (MODEL_SIZE / 32) ))
    NUM_OUTPUTS=$(( G1 + G2 + G3 ))
fi
echo "[RESULT] NUM_OUTPUTS calculated: $NUM_OUTPUTS"

# 4. Upsert Function for Integer Constants
update_or_insert() {
    local key=$1
    local value=$2
    local pattern="constexpr int $key ="
    local line="constexpr int $key = $value;"

    if grep -q "$pattern" "$HEADER_FILE"; then
        sed -i "s/$pattern [0-9]*;/$line/" "$HEADER_FILE"
        echo "[OK] Updated $key"
    else
        echo "$line" >> "$HEADER_FILE"
        echo "[OK] Inserted $key"
    fi
}

# 5. Function to Update the inline array
update_array() {
    local pattern="inline const char\* CLASS_NAMES\[\] ="
    if grep -q "$pattern" "$HEADER_FILE"; then
        # Use | as delimiter to avoid issues with /
        sed -i "s|$pattern.*|$ARRAY_LINE|" "$HEADER_FILE"
        echo "[OK] Updated inline CLASS_NAMES array"
    else
        echo "$ARRAY_LINE" >> "$HEADER_FILE"
        echo "[OK] Inserted inline CLASS_NAMES array"
    fi
}

# 6. Execute Updates
touch "$HEADER_FILE"
update_or_insert "MODEL_SIZE" "$MODEL_SIZE"
update_or_insert "NUM_CLASSES" "$NUM_CLASSES"
update_or_insert "NUM_OUTPUTS" "$NUM_OUTPUTS"
update_array

echo "--- Configuration Successful ---"