#!/bin/bash
export DYLD_LIBRARY_PATH=$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
uv run mjpython main18_optimization.py
# fishの場合
# set -x DYLD_LIBRARY_PATH (uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')


[I 2026-02-07 00:40:25,295] Trial 37 finished with value: 158.7 and parameters: {'REWARD_SURVIVAL_SCALE': 9.42134358364159, 
'REWARD_DISTANCE_DIFF_SCALE': 8.864764687795445, 
'PENALTY_STAGNATION_FORCE': -1.6804225011846823, 
'LEARNING_RATE': 9.828011873373247e-05, 
'ENT_COEF': 0.001086278382152982