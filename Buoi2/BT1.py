import pandas as pd
from collections import Counter

df = pd.read_csv("dataset/public_test.csv", keep_default_na=False)
print(df.shape[0])

do_dai_cau = [len(cau.split()) for cau in df["van_ban_goc"]]
print("cau ngan nhat:", min(do_dai_cau), " tu")
print("cau dai nhat:", max(do_dai_cau), " tu")

dem = Counter(w for cau in df["van_ban_goc"] for w in cau.split())
print("so tu khac nahau", len(dem))

tu_xuat_hien_nhieu_nhat, so_lan = dem.most_common(1)[0]
print("tu xuat hien nhieu nhat", tu_xuat_hien_nhieu_nhat, "-", so_lan, "lan")

so_dong_rong = (df["van_ban_goc"] == "").sum()
so_dong_trung = df.duplicated().sum()
print("so dong rong", so_dong_rong)
print("so dong trung", so_dong_trung)