#!/bin/bash
export DYLD_LIBRARY_PATH=$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
uv run mjpython main18_optimization.py
# fishの場合
# set -x DYLD_LIBRARY_PATH (uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
