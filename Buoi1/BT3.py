import numpy as np

def cham(a, k):
    ket_qua = []
    for i in range(0, len(a)-k+1):
        cua_so = a[i:i+k]
        trung_binh = cua_so.mean()
        ket_qua.append(trung_binh)
    return np.array(ket_qua)

def nhanh(a, k):
    S = np.cumsum(a)
    S = np.insert(S,0, 0)
    tong_cua_s0 = S[k:] - S[:-k]
    return tong_cua_s0 / k

rng = np.random.default_rng(0)
for _ in range(200):
    n = rng.integers(1, 50)
    k = rng.integers(1, n+1)
    a_test = rng.random(n)
    kq_cham = cham(a_test, k)
    kq_nhanh = nhanh(a_test, k)
    assert kq_cham.shape == kq_nhanh.shape, f"lech do dai: {kq_cham.shape} vs {kq_nhanh.shape}"
    assert np.allclose(kq_cham, kq_nhanh), f"lech gia tri"
print("200 phep thu khop nhau")