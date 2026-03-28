import sys

import glfw

if not glfw.init():
    print("glfw.init() failed")
    sys.exit(1)

win = glfw.create_window(640, 480, "glfw test", None, None)
if not win:
    print("create_window failed")
    glfw.terminate()
    sys.exit(1)

print("glfw window created")
glfw.destroy_window(win)
glfw.terminate()
print("glfw terminated")
