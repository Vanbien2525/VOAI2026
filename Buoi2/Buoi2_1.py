import pandas as pd

df = pd.read_csv("dataset/training_set.csv", keep_default_na=False)
print(df.shape)
print(df.head())
print(df.info())
print(df["nhan"].value_counts(normalize=True))
print((df["ban_dich"].str.len() > 0).sum())
print((df["ban_dich"] == "").sum())
print((df["van_ban_goc"] + df["ban_dich"]).isna().sum())

df = pd.read_csv("dataset/training_set.csv")
print((df["ban_dich"].str.len() > 0).sum())
print((df["ban_dich"] == "").sum())
print((df["van_ban_goc"] + df["ban_dich"]).isna().sum())

print("*" * 50)
print(df[df["nhan"] == 1])
df["do_dai"] = df["van_ban_goc"].str.split().str.len()
print(df.head())
print(df.groupby("nhan")["do_dai"].mean())
#print(df.apply(ham, axis = 1))
print(df.loc[3, 'nhan'])
print(df.iloc[3, 0])
print(df.loc[df["nhan"] == 1, "do_dai"])

print("*" * 50)
# from collections import Counter
# dem = Counter(w for cau in df["van_ban_goc"] for w in cau.split())
# print("so tu khac nhau:", len(dem))
# print(dem.most_common(5))
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# fig, ax = plt.subplot(1,2, figsize=(10, 3, 4))
# #phan phoi do dai cau, tach theo nhan
# import matplotlib
# matplotlib.use("Agg")   # không cần cửa sổ hiển thị, ghi thẳng ra file — hữu ích khi chạy trên server
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

# Biểu đồ 1: phân phối độ dài câu, tách theo nhãn
# for nhan, mau in ((1, "tab:blue"), (0, "tab:red")):
#     ax[0].hist(df.loc[df["nhan"] == nhan, "do_dai"], bins=range(1, 11),
#                alpha=0.55, label=f"nhan {nhan}", color=mau)
# ax[0].set_title("Do dai cau")
# ax[0].legend()

# Biểu đồ 2: 40 từ hay gặp nhất
# tan_suat = [c for _, c in dem.most_common(40)]
# ax[1].bar(range(len(tan_suat)), tan_suat)
# ax[1].set_title("40 tu hay gap nhat")

# fig.tight_layout()
# fig.savefig("nhin-du-lieu.png", dpi=120)
print(df["do_dai"].describe())                 # min, max, mean, std, các quartile — cái nhìn số học nhanh
print(df.corr(numeric_only=True))                 # ma trận tương quan giữa các cột số
print(df.boxplot(column="do_dai", by="nhan"))   # so sánh phân phối theo nhóm, thấy rõ outlier
# nhìn tương quan giữa NHIỀU cặp đặc trưng cùng lúc
