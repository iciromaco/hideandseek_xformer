#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export DYLD_LIBRARY_PATH="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

TRAIN_ENTRY="main26_train_final.py"
LOG_DIR="logs"
PARSER_SCRIPT="scripts/verify/verify_hider_stuck_log.py"

if [ "${HNS_STUCK_DEBUG:-0}" = "1" ]; then
    mkdir -p "$LOG_DIR"

    export HNS_DEBUG_ACTION_LOG="${HNS_DEBUG_ACTION_LOG:-1}"
    export HNS_DEBUG_ACTION_STEPS="${HNS_DEBUG_ACTION_STEPS:-300}"
    export HNS_DEBUG_STOP_EPS="${HNS_DEBUG_STOP_EPS:-0.01}"
    export HNS_DEBUG_STOP_WINDOW="${HNS_DEBUG_STOP_WINDOW:-20}"

    LOG_PATH="${HNS_DEBUG_LOG_PATH:-$LOG_DIR/hns_debug_stuck_$(date +%Y%m%d_%H%M%S).log}"

    echo "[launch2] stuck-debug mode enabled"
    echo "[launch2] log path: $LOG_PATH"

    uv run python "$TRAIN_ENTRY" 2>&1 | tee "$LOG_PATH"

    if [ -f "$PARSER_SCRIPT" ]; then
        echo "[launch2] parsing stuck rows from log"
        uv run python "$PARSER_SCRIPT" "$LOG_PATH" \
            --max "${HNS_STUCK_MAX_ROWS:-50}" \
            --segments \
            --segments-max "${HNS_STUCK_SEGMENTS_MAX:-20}" \
            --context-before "${HNS_STUCK_CONTEXT_BEFORE:-8}" \
            --context-max-segments "${HNS_STUCK_CONTEXT_SEGMENTS_MAX:-10}" \
            --events \
            --events-max "${HNS_STUCK_EVENTS_MAX:-20}"
    fi
else
    uv run python "$TRAIN_ENTRY"
fi
