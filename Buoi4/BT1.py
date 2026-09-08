import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge

#b1 dung du lieu da biet 
rng = np.random.default_rng(0)
n, d = 500, 3
X = rng.random((n, d)) * np.array([1.0, 100.0, 10000.0])
w_that = np.array([2.0, -0.5, 0.01])
b_that = 3.0
y = X @ w_that + b_that + 0.1 * rng.standard_normal(n)

#b2 giam do doc
def giam_do_doc(X, y, lr, so_vong = 2000):
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

#b3 chua hoa dac trung
tb = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd == 0] = 1.0
Xc = (X-tb)/sd

#b4 chuan hoa roi chay lai
w_gb, b_gb, lich_su = giam_do_doc(Xc,y, lr=0.5, so_vong=2000)
print("=== Nghiệm trên hệ đã chuẩn hoá ===")
print("w_gb", np.round(w_gb, 5))    
print("b_gb = %.5f" % b_gb)
print("mat cuoi = %.6f" %lich_su[-1])

#bai tap 1
w_goc = w_gb/sd
b_goc = b_gb - (tb * w_gb/sd).sum()

print("w suy ra (he goc)   :", np.round(w_goc, 5))
print("w that              :", np.round(w_that, 5))
print("do lech             :%.5f:" % np.abs(w_goc-w_that).max())

print("b suy ra (he goc)   :%.5f" % b_goc)
print("b that              :%.5f" % b_that)
print("do lecch            :%.5f" % np.abs(b_goc-b_that))

assert np.abs(w_goc - w_that).max() < 0.05, "Lệch w quá lớn — kiểm tra lại công thức"
assert abs(b_goc - b_that) < 0.05, "Lệch b quá lớn — kiểm tra lại công thức"
print("\nĐạt cổng kiểm: nghiệm quy đổi khớp w_that, b_that trong sai số cho phép.")
print("+" * 100)

#bai tap 2
plt.figure(figsize=(7, 5))

for lr in (0.5, 0.1, 0.01, 0.0001):
    _, _, lich_su = giam_do_doc(Xc, y, lr)
    plt.plot(lich_su, label=f"lr{lr}")

plt.yscale('log')
plt.xlabel("vong lap")
plt.ylabel("mất mát")
plt.legend()
plt.title("Đường cong mất mát theo tốc độ học")
#plt.show()         # nao chay thi bat len

#bai tap 3
lam_max = np.linalg.eigvalsh((2.0/n) * (Xc.T @ Xc)).max()
lr_max = 2.0 / lam_max
print("Nguong phan ky")
print("tri rieng lon nhat =", lam_max)
print("learning rate lon nhat =", lr_max)

lr_duoi = lr_max * 0.99
lr_tren = lr_max * 1.01

_, _, loss_duoi = giam_do_doc(Xc, y, lr=lr_duoi, so_vong=2000)
_, _, loss_tren = giam_do_doc(Xc, y, lr=lr_tren, so_vong=2000)

print("thu nghiem")
print(f"lr ngay dưới = {lr_duoi:.6f}")
print(f"loss đầu     = {loss_duoi[0]:.6f}")
print(f"loss cuối    = {loss_duoi[-1]:.6f}")
print()
print(f"lr ngay trên  = {lr_tren:.6f}")
print(f"loss đầu      = {loss_tren[0]:.6f}")
print(f"loss cuối     = {loss_tren[-1]:.6f}")

# them 
print(f"lr toi ưu nhat = {2/(lam_max + (np.linalg.eigvalsh((2.0/n) * (Xc.T @ Xc)).min()))}")
print()
# bai tap 4
def giam_do_doc_ridge(X, y, lr, lam, so_vong=2000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lich_su = []
    for _ in range(so_vong):
        du = X @ w + b - y
        mat = float((du ** 2).mean()) + lam * float((w ** 2).sum())
        lich_su.append(mat)
        if not np.isfinite(mat):
            break
        w = w - lr * ((2.0 / n) * (X.T @ du) + 2 * lam * w)   
        b = b - lr * (2.0 / n) * du.sum()                      
    return w, b, lich_su

#sklearn
lam = 0.5
alpha = lam * n

w_gd, b_gd, _ = giam_do_doc_ridge(Xc, y, lr=0.5, lam=lam, so_vong=2000)

mo_hinh = Ridge(alpha=alpha)
mo_hinh.fit(Xc, y)
w_sk, b_sk = mo_hinh.coef_, mo_hinh.intercept_

lech = max(np.abs(w_gd - w_sk).max(), abs(b_gd - b_sk))

print("lambda =", lam, " -> alpha tương ứng =", alpha)
print("w_gd (tự cài) :", np.round(w_gd, 6))
print("w_sk (sklearn):", np.round(w_sk, 6))
print("b_gd = %.6f   b_sk = %.6f" % (b_gd, b_sk))
print("lệch lớn nhất : %.3e" % lech)

assert lech < 1e-9, "Chưa khớp — kiểm tra lại công thức quy đổi hoặc lr"
print("\n✅ Đạt cổng kiểm: khớp Ridge tới sai số máy.")
print()

#goc
tb0, sd0 = X.mean(axis=0), X.std(axis=0)
sd0[sd0 == 0] = 1.0
Xc0 = (X - tb0) / sd0
w0, b0, h0 = giam_do_doc(Xc0, y, lr=0.5)

X5 = np.column_stack([X, np.full(n, 5.0)])   
#loi
print("LỖI")
tb_loi = X5.mean(axis=0)
sd_loi = X5.std(axis=0)
           
Xc_loi = (X5 - tb_loi) / sd_loi            


w_loi, b_loi, h_loi = giam_do_doc(Xc_loi, y, lr=0.5)
print("w_loi =", w_loi)
print("Số vòng chạy được trước khi dừng:", len(h_loi))


# dung
print("ĐÚNG")
tb5, sd5 = X5.mean(axis=0), X5.std(axis=0)
sd5[sd5 == 0] = 1.0                       
Xc5 = (X5 - tb5) / sd5

w5, b5, h5 = giam_do_doc(Xc5, y, lr=0.5)
print("w5 =", np.round(w5, 5), " b5 =", round(b5, 5))
print("mất cuối =", round(h5[-1], 6))
