import numpy as np

def cham(X):
    m, n = X.shape
    ket_qua = np.zeros((m, n))
    for i in range(m):
        hang = X[i]
        mn = hang.min()
        mx = hang.max()
        if mx == mn:
            ket_qua[i] = 0
        else:
            ket_qua[i] = (hang - mn) / (mx - mn)
    return ket_qua

def nhanh(X):
    mn = X.min(axis=1, keepdims=True)   # (m, 1)
    mx = X.max(axis=1, keepdims=True)   # (m, 1)
    khoang = mx - mn                     # (m, 1), có thể bằng 0

    khoang_an_toan = np.where(khoang == 0, 1, khoang)  # tránh chia 0
    Xc = (X - mn) / khoang_an_toan

    Xc = np.where(khoang == 0, 0, Xc)    # ép hàng hằng số về 0
    return Xc

# --- Kiểm chứng ---
rng = np.random.default_rng(0)
for _ in range(200):
    m = rng.integers(1, 20)
    n = rng.integers(1, 20)
    X_test = rng.random((m, n))

    kq_cham = cham(X_test)
    kq_nhanh = nhanh(X_test)
    assert np.allclose(kq_cham, kq_nhanh), "lech ket qua!"

print("200 phep thu khop nhau.")

# --- Test riêng: hàng hằng số ---
X_hang_const = np.array([
    [1.0, 2.0, 3.0],
    [5.0, 5.0, 5.0],   # hàng hằng số
    [4.0, 0.0, 8.0],
])
Xc = nhanh(X_hang_const)
print(Xc)
assert not np.isnan(Xc).any(), "co nan!"
assert not np.isinf(Xc).any(), "co inf!"
assert np.array_equal(Xc[1], [0.0, 0.0, 0.0]), "hang hang so phai la 0"
print("test hang hang so: OK")

# --- Kiểm min=0, max=1 cho hàng không hằng số ---
X_random = rng.random((10, 6))
Xc2 = nhanh(X_random)
print("min moi hang:", Xc2.min(axis=1))
print("max moi hang:", Xc2.max(axis=1))
assert np.allclose(Xc2.min(axis=1), 0.0)
assert np.allclose(Xc2.max(axis=1), 1.0)