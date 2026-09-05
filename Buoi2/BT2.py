import pandas as pd

df = pd.read_csv("dataset/training_set.csv", keep_default_na=False)
print((df["ban_dich"].str.len() > 0).sum())
print((df["ban_dich"] == "").sum())
print((df["van_ban_goc"] + df["ban_dich"]).isna().sum())

df = pd.read_csv("dataset/training_set.csv")
print((df["ban_dich"].str.len() > 0).sum())
print((df["ban_dich"] == "").sum())
print((df["van_ban_goc"] + df["ban_dich"]).isna().sum())