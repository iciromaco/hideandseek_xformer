#!/usr/bin/env python3
"""簡単な対話用ランナー: ターミナルで実行して WASD / J K を試してください

実行例 (macOS の場合 viewer のフル機能を使うには mjpython 推奨):
  mjpython scripts/run_human.py
あるいは (一般):
  python scripts/run_human.py

終了: Ctrl-C
"""

import argparse
import select
import signal
import sys
import termios
import time
import tty

import numpy as np

from envs.hns_environment import TeamCosEnv

RUN_STEPS = 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Optional reset seed for reproducible starts")
    parser.add_argument("--hold-up", action="store_true", help="Automatically hold UP (forward) input")
    args = parser.parse_args()

    env = TeamCosEnv(mode="initial", control_mode="human", USE_VIEWER=True, debug_mode=True)
    # ensure debug mode enabled at runtime (some callers toggle after init)
    env.debug_mode = True

    # runner-level config: prefer momentary buttons (press+release) to avoid
    # stuck-on behavior when release events are missed by the OS/listener.
    env.human_toggle_buttons = False
    env.human_debug_keys = True

    # local human action buffer (runner owns I/O)
    human_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # try to use pynput if available for global key capture
    try:
        from pynput import keyboard as _pynput_keyboard  # type: ignore
    except Exception:
        _pynput_keyboard = None

    use_pynput = _pynput_keyboard is not None

    # build mapping from env.human_key_bindings for terminal fallback
    key_bindings = getattr(env, "human_key_bindings", {}) or {}
    term_key_map = {}
    # forward/back/left/right may be multi-char (arrow escape seq)
    if "forward" in key_bindings:
        term_key_map[key_bindings["forward"]] = (0, 1.0)
    if "back" in key_bindings:
        term_key_map[key_bindings["back"]] = (0, -1.0)
    if "left" in key_bindings:
        term_key_map[key_bindings["left"]] = (1, 1.0)
    if "right" in key_bindings:
        term_key_map[key_bindings["right"]] = (1, -1.0)
    lock_k = key_bindings.get("lock", "o")
    grab_k = key_bindings.get("grab", "p")

    if args.seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=int(args.seed))
    print("env reset; focus this terminal and viewer window. Use arrow keys for motion, o/p for lock/grab. Ctrl-C to exit.")

    # setup terminal raw mode for fallback input
    orig_term = None
    tty_enabled = False
    try:
        if sys.stdin.isatty():
            orig_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
            tty_enabled = True
    except Exception:
        orig_term = None

    def restore_term():
        try:
            if orig_term is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_term)
        except Exception:
            pass

    def handle_sigint(sig, frame):
        restore_term()
        print("\nexiting")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    listener = None
    if use_pynput:
        try:

            def _on_press(key):
                try:
                    if key == _pynput_keyboard.Key.up:
                        human_action[0] = 1.0
                        if env.human_debug_keys:
                            print("[pynput] UP press")
                    elif key == _pynput_keyboard.Key.down:
                        human_action[0] = -1.0
                        if env.human_debug_keys:
                            print("[pynput] DOWN press")
                    elif key == _pynput_keyboard.Key.left:
                        human_action[1] = 1.0
                        if env.human_debug_keys:
                            print("[pynput] LEFT press")
                    elif key == _pynput_keyboard.Key.right:
                        human_action[1] = -1.0
                        if env.human_debug_keys:
                            print("[pynput] RIGHT press")
                    else:
                        k = None
                        try:
                            k = key.char.lower()
                        except Exception:
                            k = None
                        if k == lock_k:
                            if env.human_toggle_buttons:
                                human_action[2] = 0.0 if float(human_action[2]) > 0.0 else 1.0
                            else:
                                human_action[2] = 1.0
                            if env.human_debug_keys:
                                print("[pynput] LOCK press", k, "state=", human_action[2])
                        elif k == grab_k:
                            if env.human_toggle_buttons:
                                human_action[3] = 0.0 if float(human_action[3]) > 0.0 else 1.0
                            else:
                                human_action[3] = 1.0
                            if env.human_debug_keys:
                                print("[pynput] GRAB press", k, "state=", human_action[3])
                except Exception:
                    pass

            def _on_release(key):
                try:
                    if key == _pynput_keyboard.Key.up or key == _pynput_keyboard.Key.down:
                        human_action[0] = 0.0
                    elif key == _pynput_keyboard.Key.left or key == _pynput_keyboard.Key.right:
                        human_action[1] = 0.0
                    else:
                        k = None
                        try:
                            k = key.char.lower()
                        except Exception:
                            k = None
                        if not env.human_toggle_buttons:
                            if k == lock_k:
                                human_action[2] = 0.0
                            elif k == grab_k:
                                human_action[3] = 0.0
                except Exception:
                    pass

            listener = _pynput_keyboard.Listener(on_press=_on_press, on_release=_on_release)
            listener.daemon = True
            listener.start()
            if env.human_debug_keys:
                print("[runner] started pynput listener")
        except Exception:
            listener = None

    prev_human = human_action.copy()
    try:
        for _ in range(RUN_STEPS):
            # terminal fallback: poll stdin for key sequences
            if listener is None and tty_enabled:
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                        ch = sys.stdin.read(1)
                        if not ch:
                            ch = ""
                        key = ch
                        if ch == "\x1b":
                            # read remaining bytes for escape seq
                            r, _, _ = select.select([sys.stdin], [], [], 0.01)
                            if sys.stdin in r:
                                rest = sys.stdin.read(2)
                                key = ch + rest
                        lookup = key if len(key) > 1 else key.lower()
                        kv = term_key_map.get(lookup)
                        if kv is not None:
                            idx, val = kv
                            human_action[idx] = float(val)
                        else:
                            if lookup == lock_k:
                                if env.human_toggle_buttons:
                                    human_action[2] = 0.0 if float(human_action[2]) > 0.0 else 1.0
                                else:
                                    human_action[2] = 1.0
                                if env.human_debug_keys:
                                    print("[term] LOCK press", lookup, "state=", human_action[2])
                            elif lookup == grab_k:
                                if env.human_toggle_buttons:
                                    human_action[3] = 0.0 if float(human_action[3]) > 0.0 else 1.0
                                else:
                                    human_action[3] = 1.0
                                if env.human_debug_keys:
                                    print("[term] GRAB press", lookup, "state=", human_action[3])
                except Exception:
                    pass

            # apply automatic hold-up if requested
            if args.hold_up:
                human_action[0] = 1.0
            # step using the runner-collected action (env must not branch on source)
            res = env.step(human_action)

            # render
            try:
                env.render()
            except Exception:
                pass

            # debug prints for changed human action
            try:
                ha = human_action
                if not np.array_equal(ha, prev_human):
                    print("human_action=", ha.tolist())
                    if ha[2] > 0.0:
                        print("LOCK pressed")
                    if ha[3] > 0.0:
                        print("GRAB pressed")
                    prev_human = ha.copy()
            except Exception:
                pass

            # clear momentary buttons when not toggle mode (runner enforces)
            if not env.human_toggle_buttons:
                human_action[2] = 0.0
                human_action[3] = 0.0

            time.sleep(env.model.opt.timestep * max(1, env.action_repeat))
    finally:
        restore_term()
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
