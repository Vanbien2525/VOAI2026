import numpy as np

def khoang_cach(A,B):
    a2 = (A * A).sum(axis=1)[:, None]           #(m, 1)
    b2 = (B * B).sum(axis=1)[None, :]           #(1, n)
    d2 = a2 - 2.0*(A @ B.T) + b2
    return np.sqrt(np.maximum(d2, 0.0))

rng = np.random.default_rng(0)
A = rng.random((200, 8))
B = rng.random((300, 8))

cham = np.array([[np.sqrt(((x-y)**2).sum()) for y in B] for x in A])
nhanh = khoang_cach(A,B)

print("lech lon nhat %.3e" % np.abs(cham - nhanh).max())