def macro_f1(that, doan):
    tong = 0.0 
    for lop in (0,1):
        tp = sum(1 for t, d in zip(that, doan) if t == lop and d == lop)
        fp = sum(1 for t, d in zip(that, doan) if t != lop and d == lop)
        fn = sum(1 for t, d in zip(that, doan) if t == lop and d != lop)
        
        if tp == 0:
            tong += 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            tong += 2 * p * r / (p + r)
    
    return tong / 2

import random 
from sklearn.metrics import f1_score

random.seed(1)
lech = 0.0
for _ in range(300):
    n = random.randint(2,40)
    a = [random.randint(0,1) for _ in range(n)]
    b = [random.randint(0,1) for _ in range(n)]
    lech = max(lech, abs(macro_f1(a,b)
                         - f1_score(a,b, average="macro", labels=[0,1], zero_division=0)))
print("lech lon nhat : %.2e" % lech)
