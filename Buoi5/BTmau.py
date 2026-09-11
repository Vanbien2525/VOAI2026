import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

rng = np.random.default_rng(0)
n, d = 800, 4
X = rng.standard_normal((n,d))

w_that = np.array([1.5, -2.0, 1.5, 0.0])
b_that = 0.3

z = X @ w_that + b_that
p = sigmoid(z)

# Tạo nhãn 0/1
y = rng.binomial(1, p)
def gd_logistic(X, y, lr, so_vong, lam =0.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lich_su = []
    for _ in range(so_vong):
        p = sigmoid(X @ w + b)
        eps = 1e-12
        mat = float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean() + lam * (w @ w))
        lich_su.append(mat)
        du = p - y
        w = w - lr * ((X.T @ du) / n + 2 * lam * w)
        b = b - lr * du.mean()
    return w, b, lich_su

w_gd, b_gd, _ = gd_logistic(X, y, lr=1.0, so_vong=20000)
sk = LogisticRegression(C=1e9, max_iter=5000, tol=1e-10).fit(X,y)
lech = max(np.abs(w_gd - sk.coef_[0]).max(), abs(b_gd - sk.intercept_[0]).max())
print("GD w=", np.around(w_gd,4), "b = %.4f" % b_gd)
print("sklearn w =", np.round(sk.coef_[0], 4), "b = %.4f" % sk.intercept_[0])
print("lech lon nhat: %.3e" % lech)