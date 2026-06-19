#!/usr/bin/env bash
set -euo pipefail

cc -O2 -Wall -Wextra -o vicsek_dynamic vicsek_dynamic.c -lm
python3 generate_narma10.py
./vicsek_dynamic tmp/narma10_input_0:0.5_seed666.dat data
python3 analysis/vicsek_prediction.py \
  --data-folder data \
  --output-folder reservoir_data \
  --input-path tmp/narma10_input_0:0.5_seed666.dat \
  --target-path tmp/narma10_target_0:0.5_seed666.dat
