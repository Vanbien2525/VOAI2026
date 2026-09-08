def mot_buoc(X, y, w, b, lr):
    n = X.shape[0]
    du = X @ w + b - y
    w = w - lr *(2.0/n) * (X.T @ du)
    b = b - lr *(2.0/n) * du.sum()
    return w,b

tb = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd == 0] = 1.0
xc = (X - tb) / sd

tb = sd = X_train.mean(axis=0), X_train,std(axis=0)
sd[sd == 0] = 1.0
X_train_c = (X_train - tb)/ sd
X_test_c = (X_test - tb)/sd

# 3.1 Bước 1: dựng dữ liệu mà bạn biết trước đáp án
import numpy as np
rng = np.random.default_rng(0)
n, d = 500, 3
X = rng.random((n, d)) * np.array([1.0, 100.0, 10000.0])
w_that = np.array([2.0, -0.5, 0.01])
b_that = 3.0
y = X @ w_that + b_that + 0.1 * rng.standard_normal(n)

#3.2 Bước 2: cài giảm độ dốc
def giam_do_doc(x, y, lr, so_vong=2000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lich_su = []
    for _ in range(so_vong):
        du = X @ w + b - y
        mat = float((du**2).mean())
        lich_su.append(mat)
        
        if not np.isfinite(mat):
            break
        w = w - lr * (2.0/n) * (X.T @ du)
        b = b - lr * (2.0/n) * du.sum()
    return w, b, lich_su

# 3.3 Bước 3: chạy chưa chuẩn hoá, và nhìn nó bay
for lr in (1e-2, 1e-6, 1e-9):
    w, b, h = giam_do_doc(X, y, lr)
    if np.isfinite(h[-1]):
        print("lr=%-8.0e mat cuoi %.4f" % (lr, h[-1]))
    else:
        print("lr=%-8.0e phan ky sau %d vong" % (lr, len(h)))

# 3.4 Bước 4: chuẩn hoá, rồi chạy lại
tb, sd = X.mean(axis=0), X.std(axis=0)
Xc = (X - tb) / sd
w, b, h = giam_do_doc(Xc, y, 0.5)
print("mat cuoi %.6f" % h[-1])
