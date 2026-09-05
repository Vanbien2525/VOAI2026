import numpy as np

def chuan_hoa(X):
    tb = X.mean(axis=0)
    sd = X.std(axis=0)
    Xc = X - tb

    sd[sd == 0] = 1
    
    z = Xc / sd

    return z

def chuan_hoa_cham(X):
    tb = X.mean(axis=0)
    sd = X.std(axis=0)

    sd[sd == 0] = 1

    return np.array([
        [(X[i, j] - tb[j]) / sd[j] for j in range(X.shape[1])]
        for i in range(X.shape[0])
    ])

rng = np.random.default_rng(0)
X = rng.random((100, 5))

X[:, 0] = 7

nhanh = chuan_hoa(X)
cham = chuan_hoa_cham(X)

print("Mean từng cột:")
print(nhanh.mean(axis=0))

print("\nStd từng cột:")
print(nhanh.std(axis=0))

print("\nĐộ lệch lớn nhất giữa bản nhanh và bản chậm:")
print(np.abs(nhanh - cham).max())

print("\nCó NaN không?")
print(np.isnan(nhanh).any())