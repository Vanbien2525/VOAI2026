#B1 : thay khoang cach toac do bang mat
import time
import numpy as np

rng = np.random.default_rng(0)
a = rng.random(2_000_000)

t0 = time.perf_counter()
tong_vong_lap = 0.0
for x in a:
    tong_vong_lap += x*x
t_vong_lap = time.perf_counter() - t0

t0 = time.perf_counter()
tong_vector = float((a*a).sum())
t_vector = time.perf_counter() - t0

print("vong lap %.3f giay, vecto %.4f giây, nhanh gap %.0f giay" % (t_vong_lap, t_vector, t_vong_lap/t_vector))
print("lech nhau %.3e " % abs(tong_vong_lap - tong_vector))

#B2 : chuan hoa tung cot khong dung vong lap
X = rng.random((1000, 5))

tb = X.mean(axis=0)             #shape(5,)
sd = X.std(axis= 0)             #shape (5,)
Xc = (X - tb)/sd                #broadcasting does the work

print(Xc.shape, np.abs(Xc.mean(axis=0)).max(), np.abs(Xc.std(axis=0) - 1).max())

#B3 : lay du lieu tu tac vu 1
import csv, io
import numpy as np

with io.open("chung-ket/tac-vu-1-ngon-ngu/du-lieu/training_set.csv",
                encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
    
nhan = np.array([int(r["nhan"]) for r in rows])
do_dai = np.array([len(r["van_ban_goc"].split()) for r in rows])

print("so cau :", nhan.size)
print("ti le nhan 1 : %.3f" % nhan.mean())
print("do dai trung binh, nhan 1: %.2f" % do_dai[nhan == 1].mean())
print("do dai trung binh, nhan 0: %.2f" % do_dai[nhan == 0].mean())