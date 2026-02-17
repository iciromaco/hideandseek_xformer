from numba import njit
import time

def python_loop(n):
    s = 0.0
    for i in range(n):
        s += i * 0.1
    return s

@njit
def numba_loop(n):
    s = 0.0
    for i in range(n):
        s += i * 0.1
    return s


n = 10_000_000

t0 = time.time()
python_loop(n)
print("python:", time.time() - t0)

t0 = time.time()
numba_loop(n)   # 初回はコンパイルで遅い
print("numba first:", time.time() - t0)

t0 = time.time()
numba_loop(n)   # 2回目は爆速
print("numba second:", time.time() - t0)
