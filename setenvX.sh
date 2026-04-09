#!/usr/bin/env fish
set -x DYLD_LIBRARY_PATH (uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')