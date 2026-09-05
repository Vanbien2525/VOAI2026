import numpy as np
import time

def cham(a, S):
    dem = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] + a[j] == S:
                dem += 1
    return dem

def nhanh(a, S):
    n = len(a)
    tong = a[:, None] + a[None, :]       
    mask_tong = (tong == S)               
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  
    mask = mask_tong & upper
    return mask.sum()

# --- Bước 3: kiểm chứng trên 200 mảng ngẫu nhiên ---
rng = np.random.default_rng(0)
for _ in range(200):
    n_test = rng.integers(2, 50)
    a_test = rng.integers(-20, 20, size=n_test)
    S_test = int(rng.integers(-20, 20))
    assert cham(a_test, S_test) == nhanh(a_test, S_test), "Lệch kết quả!"
print("200 phép thử khớp nhau.")

# --- Đo thời gian với n = 5000 ---
a_big = rng.integers(-1000, 1000, size=5000)
S_big = 10
t0 = time.perf_counter()
ket_qua = nhanh(a_big, S_big)
t1 = time.perf_counter()
print("ket qua:", ket_qua, "| thoi gian: %.4f giay" % (t1 - t0))