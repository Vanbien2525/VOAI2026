import pandas as pd

df = pd.read_csv("dataset/training_set.csv", keep_default_na=False)

# 1. Lọc câu nhãn 1 -> index vẫn rời rạc (giữ nguyên từ df gốc)
con = df[df["nhan"] == 1]

# 2. Tính độ dài (số từ) trên tập con -> Series thừa hưởng index rời rạc của con
do_dai_con = con["van_ban_goc"].str.split().str.len()

# 3a. Cách 1: gán thẳng Series -> pandas CĂN CHỈNH theo index
df["do_dai_cach1"] = do_dai_con
so_nan_cach1 = df["do_dai_cach1"].isna().sum()
so_dong_nhan_0 = (df["nhan"] == 0).sum()
print("So o NaN sau cach 1:", so_nan_cach1)
print("So dong nhan == 0 (de doi chieu):", so_dong_nhan_0)

# 3b. Cách 2: gán bằng .values -> mảng thô, mất hết thông tin index
try:
    df["do_dai_cach2"] = do_dai_con.values
    print("Cach 2 chay duoc, khong bao loi")
except ValueError as e:
    print("Cach 2 bao loi ngay lap tuc:", e)

print(df)