import pandas as pd

def kiem_bai_nop(duong_de, duong_bai):
    df_de = pd.read_csv(duong_de, keep_default_na=False)
    df_bai = pd.read_csv(duong_bai, keep_default_na=False)

    loi = []
    COT_CHUAN = {"van_ban_goc", "ban_dich", "nhan"}

    # 1. Đủ ba cột, đúng tên (chuẩn cố định, không lấy từ file đề)
    col_bai = set(df_bai.columns)
    if col_bai != COT_CHUAN:
        loi.append(f"Sai cột: cần {COT_CHUAN}, bài nộp có {col_bai}")

    # 2. Số dòng khớp file đề
    so_dong_khop = (len(df_de) == len(df_bai))
    if not so_dong_khop:
        loi.append(f"Sai số dòng: đề có {len(df_de)} dòng, bài nộp có {len(df_bai)} dòng")

    # 3. van_ban_goc khớp từng dòng — chỉ so khi số dòng đã khớp,
    #    và phải có cột van_ban_goc trong bài nộp mới so được
    if so_dong_khop and "van_ban_goc" in df_bai.columns:
        de_reset = df_de["van_ban_goc"].reset_index(drop=True)
        bai_reset = df_bai["van_ban_goc"].reset_index(drop=True)
        if not de_reset.equals(bai_reset):
            so_dong_sai = (de_reset != bai_reset).sum()
            loi.append(f"van_ban_goc không khớp thứ tự: {so_dong_sai} dòng sai vị trí")
    elif not so_dong_khop:
        loi.append("Không thể so van_ban_goc từng dòng vì số dòng đã sai")

    # 4. nhan chỉ chứa 0 và 1 — chỉ kiểm nếu cột nhan thực sự tồn tại
    if "nhan" in df_bai.columns:
        gia_tri_nhan = set(df_bai["nhan"].astype(int).unique())
        if not gia_tri_nhan.issubset({0, 1}):
            loi.append(f"Cột nhan chứa giá trị ngoài {{0,1}}: {gia_tri_nhan - {0, 1}}")
    else:
        loi.append("Thiếu cột nhan nên không kiểm được giá trị 0/1")

    return loi


# ---------------------- Tự tạo 5 file để test ----------------------
de = pd.read_csv("dataset/public_test.csv", keep_default_na=False)

# a) File hợp lệ: đúng cột, đúng số dòng, đúng thứ tự, nhan hợp lệ
hop_le = de.copy()
hop_le["ban_dich"] = ""
hop_le["nhan"] = 0
hop_le.to_csv("dataset/test_hop_le.csv", index=False)

# b) Thiếu cột (bỏ ban_dich)
thieu_cot = de.copy()
thieu_cot["nhan"] = 0
thieu_cot.to_csv("dataset/test_thieu_cot.csv", index=False)

# c) Sai số dòng (bớt 1 dòng)
sai_so_dong = hop_le.iloc[:-1]
sai_so_dong.to_csv("dataset/test_sai_so_dong.csv", index=False)

# d) Sai thứ tự (đảo ngược toàn bộ van_ban_goc)
sai_thu_tu = hop_le.copy()
sai_thu_tu["van_ban_goc"] = sai_thu_tu["van_ban_goc"].iloc[::-1].reset_index(drop=True)
sai_thu_tu.to_csv("dataset/test_sai_thu_tu.csv", index=False)

# e) nhan sai giá trị (gán 2 cho vài dòng)
nhan_sai = hop_le.copy()
nhan_sai.loc[0:3, "nhan"] = 2
nhan_sai.to_csv("dataset/test_nhan_sai.csv", index=False)

# ---------------------- Chạy kiểm thử ----------------------
for ten_file in ["test_thieu_cot", "test_sai_so_dong", "test_sai_thu_tu",
                  "test_nhan_sai", "test_hop_le"]:
    duong = f"dataset/{ten_file}.csv"
    ket_qua = kiem_bai_nop("dataset/public_test.csv", duong)
    print(f"{ten_file}: {ket_qua}")