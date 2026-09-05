import numpy as np

a = np.arange(12).reshape(3, 4)
print(a.shape, a.dtype)
print(a.sum(axis=0).shape) 
print(a.sum(axis=1).shape)

u = np.array([1, 2, 3]) 
v = np.array([10, 20, 30, 40])
print((u[:, None] + v[None, :]).shape)
print("*"*40)

a = np.array([5, -2, 7, 0, -9])
mat_na = a > 0                      # [True, False, True, False, False]
print(a[mat_na])                    # [5, 7]  -- chỉ lấy phần tử dương
print(mat_na.sum())                 # 2 -- vì True được tính như số 1
print(np.where(a > 0, a, 0))        # kẹp (clamp) số âm về 0, không vòng lặp

thu_tu = np.argsort(a)
print(thu_tu)
print(a[thu_tu])
print("*"*40)

import time
t0 = time.perf_counter()
print(f"{time.perf_counter() - t0:.3f} giay")
print("*"*40)
