import numpy as np
import torch

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

rng = np.random.default_rng(0)
n, d = 800, 4
X = rng.standard_normal((n, d))

y = rng.integers(0,2, size=n)

w = np.zeros(d)
b = 0.0

X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.float64)

w_t = torch.tensor(w, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(b, dtype=torch.float64, requires_grad=True)

z = X_t @ w_t + b_t
p = torch.sigmoid(z)

eps = 1e-12

loss = - (
    y_t * torch.log(p + eps)
    + (1 - y_t) * torch.log(1 - p + eps)
).mean()

loss.backward()

grad_torch = w_t.grad.numpy()

p_np = sigmoid(X @ w + b)
grad_cong_thuc = (X.T @ (p_np - y)) / n

lech = np.abs(grad_torch - grad_cong_thuc).max()
print("Gradient từ torch.autograd:",grad_torch)
print("\nGradient từ công thức:",grad_cong_thuc)

print("\nĐộ lệch lớn nhất: %.3e" % lech)

assert lech < 1e-6, "khong khop"
print("Khop nhau")