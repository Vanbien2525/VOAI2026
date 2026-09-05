import pandas as pd
de = pd.read_csv('dataset/public_test.csv', keep_default_na=False)

#cach sai
sai = de.copy()
sai = sai.sort_values("van_ban_goc")
sai["nhan"] == 1
sai["ban_dich"] = ""
sai.to_csv("sai.csv", index=False)

# cach dung
dung = de.copy()
dung["nhan"] == 1
dung["ban_dich"] = ""
dung.to_csv("dung.csv", index=False)