import numpy as np
import time
x = np.random.rand(1200)  # 大きな配列
y = np.random.rand(1200)  # 大きな配列
z = np.array([[i, j] for i in x for j in y]) # 2次元配列としてzを正しく作成

methods = [
    ("np.linalg.norm(z)", lambda: np.linalg.norm(z)),
    ("np.sqrt(np.sum(z*z))", lambda: np.sqrt(np.sum(z*z))),
    ("np.sqrt(np.dot(z.flatten(), z.flatten()))", lambda: np.sqrt(np.dot(z.flatten(), z.flatten()))), # zをフラットにしてドット積を計算
    ("np.sqrt(np.einsum('ij,ij->', z, z))", lambda: np.sqrt(np.einsum('ij,ij->', z, z))) # 2次元配列のフロベニウスノルム
]

for name, func in methods:
    start = time.time()
    for _ in range(1000):
        result = func()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.4f}秒")