# from sklearn.metrics import f1_score
# from sklearn.tree import DecisionTreeClassifier

# mo_hinh = DecisionTreeClassifier
# mo_hinh.fit(X_train, y_train)
# y_doan = mo_hinh.predict(X_test)
# diem = f1_score(y_test, y_doan, average="macro", labels=[0,1], zero_division=0)

# import os 
# import random
# import numpy as np

# SEED = 20260825
# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# try:
#     import torch
#     torch.manual_seed(SEED)
# except: ImportError:
#     pass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 0
df = pd.read_csv("..//Buoi2/dataset/training_set.csv", keep_default_na=False)
X = df["van_ban_goc"].to_list()
y = df["nhan"].to_list()
X_hoc, X_kiem, y_hoc, y_kiem = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

mo_hinh = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
    LogisticRegression(max_iter=2000, C=4.0, random_state=SEED))

mo_hinh.fit(X_hoc, y_hoc)

for ten, Xs, ys in (("da hoc", X_hoc, y_hoc), ("kiem dinh", X_kiem, y_kiem)):
    diem = f1_score(ys, mo_hinh.predict(Xs), average="macro",
                    labels=[0,1], zero_division=0)
    print("%-10s macro-F1 %.4f" % (ten, diem))

mo_hinh.fit(X, y)
de = pd.read_csv("../Buoi2/dataset/public_test.csv", keep_default_na=False)
doan = mo_hinh.predict(de["van_ban_goc"].to_list())

