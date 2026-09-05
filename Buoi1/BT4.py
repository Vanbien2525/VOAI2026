import numpy as np

def cham(y, C):
    ket_qua = []
    for i in range(len(y)):
        hang = np.zeros(C, dtype=int)
        nhan = y[i]
        hang[nhan] = 1
        ket_qua.append(hang)
    return np.array(ket_qua)

def nhanh(y, C):
    S = np.arange(C)                      # shape (C,)
    onehot = (y[:, None] == S[None, :])    # shape (N, C), so sánh -> boolean
    return onehot.astype(int)              # ép về 0/1 số nguyên

# --- Kiểm chứng ---
rng = np.random.default_rng(0)
for _ in range(200):
    C = int(rng.integers(2, 10))
    N = int(rng.integers(1, 50))
    y_test = rng.integers(0, C, size=N)

    kq_cham = cham(y_test, C)
    kq_nhanh = nhanh(y_test, C)
    assert np.array_equal(kq_cham, kq_nhanh), "lech ket qua!"

    # Cổng kiểm bổ sung theo đề bài
    assert np.all(kq_nhanh.sum(axis=1) == 1), "co hang khong dung 1 so 1"
    assert np.array_equal(kq_nhanh.argmax(axis=1), y_test), "argmax khong khop y goc"

print("200 phep thu khop nhau, dung cong kiem.")