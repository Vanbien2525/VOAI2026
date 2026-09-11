import numpy as np

# ham sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sgmoid_an_toan(z):
    out = np.empty_like(z, dtype=float)
    duong = z > 0
    out[duong] = 1.0 / (1 + np.exp(-z[duong]))
    am = ~duong
    exp_z = np.exp(z[am])
    out[am] = exp_z / (1 + exp_z)
    return out

rng = np.random.default_rng(0)
n, d = 800, 4
X = rng.standard_normal((n,d))
w_that = np.array([1.5, -2.0, 1.5, 0.0])
b_that = -0.3

y = (rng.random(n) < sigmoid(X @ w_that + b_that).astype(float))
print("ti le nhan 1: %.4f" % y.mean())

def gd_logistic(X, y, lr, so_vong, lam=0.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lich_su = []
    for _ in range(so_vong):
        p = sigmoid(X @ w + b)
        eps = 1e-12
        mat= float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean() + lam * (w @ w))
        lich_su.append(mat)
        du = p - y
        w = w - lr * ((X.T @ du) / n + 2 * lam * w)
        b = b - lr * du.mean()
        return w, b, lich_su
    
def gd_binh_phuong(X, y, lr, so_vong, w0=None):
    """Same loop, squared loss on the sigmoid output."""
    n, d = X.shape
    w = np.zeros(d) if w0 is None else w0.copy()
    b = 0.0
    for _ in range(so_vong):
        p = sigmoid(X @ w + b)
        du = (p - y) * p * (1 - p)  
        w = w - lr * (X.T @ du) / n
        b = b - lr * du.mean()
    return w, b

w0 = np.array([-8.0, 8.0, 0.0, 0.0])  
