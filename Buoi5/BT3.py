import numpy as np

rng = np.random.default_rng(0)
n, d = 800, 4
X = rng.standard_normal((n, d))

y = rng.integers(0,2, size=n)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# GRADIENT CỦA CROSS ENTROPY
def gradient_cross_entropy(X, y, w, b):
    n = len(y)

    z = X @ w + b
    p = sigmoid(z)

    grad_w = (X.T @ (p - y)) / n
    grad_b = np.mean(p - y)

    return grad_w, grad_b

# GRADIENT CỦA SQUARED LOSS
# L = 1/(2n) * sum((p - y)^2)
# Vì p = sigmoid(z)
# dL/dw = 1/n X.T @ ((p-y) * p*(1-p))
def gradient_squared_loss(X, y, w, b):
    n = len(y)

    z = X @ w + b
    p = sigmoid(z)

    grad_w = (
        X.T @ ((p - y) * p * (1 - p))
    ) / n

    grad_b = np.mean(
        (p - y) * p * (1 - p)
    )
    return grad_w, grad_b

# TÍNH ACCURACY
def accuracy(X, y, w, b):
    p = sigmoid(X @ w + b)

    y_pred = (p >= 0.5).astype(int)

    return np.mean(y_pred == y)

# ĐO GRADIENT Ở BƯỚC ĐẦU
def do_gradient_test(X, y, w0, b0=0.0):

    grad_ce_w, grad_ce_b = gradient_cross_entropy(
        X, y, w0, b0
    )

    grad_sq_w, grad_sq_b = gradient_squared_loss(
        X, y, w0, b0
    )
    norm_ce = np.linalg.norm(grad_ce_w)

    norm_sq = np.linalg.norm(grad_sq_w)

    print("ĐỘ LỚN GRADIENT BƯỚC ĐẦU")
    print(f"Cross Entropy : {norm_ce:.6f}")
    print(f"Squared Loss  : {norm_sq:.7f}")

    return norm_ce, norm_sq

# ĐẾM SỐ VÒNG ĐỂ ACCURACY >= 0.75
def train(
    X,
    y,
    w0,
    b0,
    loss_type,
    learning_rate=1.0,
    max_iter=10000
):

    w = w0.copy()
    b = b0

    for iteration in range(1, max_iter + 1):
        # gradient
        if loss_type == "cross_entropy":

            grad_w, grad_b = gradient_cross_entropy(
                X, y, w, b
            )

        elif loss_type == "squared":

            grad_w, grad_b = gradient_squared_loss(
                X, y, w, b
            )

        else:
            raise ValueError("loss_type không hợp lệ")

        # Cập nhật Gradient Descent
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Kiểm tra accuracy sau mỗi vòng
        acc = accuracy(X, y, w, b)

        if acc >= 0.75:
            return iteration, acc, w, b
    return None, acc, w, b

# KHỞI TẠO CỐ Ý SAI
w0 = np.array([
    -8.0,
     8.0,
     0.0,
     0.0
])
b0 = 0.0

# ĐO GRADIENT
norm_ce, norm_sq = do_gradient_test(
    X,
    y,
    w0,
    b0
)

# TRAIN CROSS ENTROPY
iter_ce, acc_ce, w_ce, b_ce = train(
    X,
    y,
    w0,
    b0,
    loss_type="cross_entropy",
    learning_rate=1.0,
    max_iter=10000
)

# TRAIN SQUARED LOSS
iter_sq, acc_sq, w_sq, b_sq = train(
    X,
    y,
    w0,
    b0,
    loss_type="squared",
    learning_rate=1.0,
    max_iter=10000
)

# IN KẾT QUẢ
print("\nSỐ VÒNG ĐỂ ACCURACY >= 0.75")
print(
    f"Cross Entropy : {iter_ce} vòng, "
    f"accuracy = {acc_ce:.4f}"
)
print(
    f"Squared Loss  : {iter_sq} vòng, "
    f"accuracy = {acc_sq:.4f}"
)

# BẢNG KẾT QUẢ
print("\nBẢNG SO SÁNH")
print(
    f"{'':25}"
    f"{'Cross Entropy':20}"
    f"{'Squared Loss':20}"
)
print(
    f"{'Norm gradient bước đầu':25}"
    f"{norm_ce:<20.6f}"
    f"{norm_sq:<20.7f}"
)
print(
    f"{'Số vòng đạt accuracy 0.75':25}"
    f"{iter_ce:<20}"
    f"{iter_sq:<20}"
)