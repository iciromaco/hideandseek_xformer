import sys, termios, tty, select, time

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
print("Terminal input test. Press arrows, 'o', 'p'. Ctrl-C to exit.")
try:
    tty.setcbreak(fd)
    start = time.time()
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.5)
        if r:
            data = sys.stdin.read(3)
            print(f"RAW:{repr(data)}")
            sys.stdout.flush()
        if time.time() - start > 30:
            print("Timeout reached (30s). Exiting.")
            break
except KeyboardInterrupt:
    print("Interrupted")
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print("Restored terminal settings.")
