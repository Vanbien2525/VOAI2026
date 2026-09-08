from sklearn.linear_model import LinearRegression
import numpy as np
from Buoi1 import giam_do_doc
w_gd, b_gd, _ = giam_do_doc(Xc, y, lr=0.5, so_vong=2000)

mo_hinh = LinearRegression().fit(Xc, y)
w_sk, b_sk = mo_hinh.coef_, mo_hinh.intercept_

lech = max(np.abs(w_gd - w_sk).max(), abs(b_gd - b_sk))
print("GD w =", np.round(w_gd, 5), "b = %.5f" % b_gd)
print("sklearn w =", np.round(w_sk, 5), "b = %.5f" % b_sk)
print("lech lon nhat: %.3e" % lech)
