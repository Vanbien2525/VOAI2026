"""Task 1, classification only. Run: python main.py"""
import csv 
import io
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEED = 20260825
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

HERE =  os.path.dirname(os.path.abspath(__file__))

def duong(*phan):
    return os.path.join(HERE, *phan)

def main():
    train = pd.read_csv(duong("du-lieu", "training_set.csv"), keep_default_na=False)
    de = pd.read_csv(duong("du-lieu", "public_test.csv"), keep_default_na=False)
    
    mo_hinh = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2),
        LogisticRegression(max_iter=2000, C=4.0, random_state=SEED )
    )
    mo_hinh.fit(train["van_ban_goc"].to_list(), train["nhan"].to_list())
    doan = mo_hinh.predict(de["van_ban_goc"].tolist())
    
    with io.open(duong("submission.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["van_ban_goc", "nhan", "dich"])
        for cau, nhan in zip(de["van_ban_goc"], doan):
            w.writerow([cau, int(nhan), ""])
    
if __name__ == "__main__":
    main()