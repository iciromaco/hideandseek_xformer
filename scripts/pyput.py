# save as scripts/pynput_test.py
from pynput import keyboard


def on_press(k):
    try:
        print("PRESSED", repr(k.char))
    except Exception:
        print("PRESSED", k)


def on_release(k):
    try:
        print("RELEASE", repr(k.char))
    except Exception:
        print("RELEASE", k)
    if k == keyboard.Key.esc:
        return False


with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
    l.join()
