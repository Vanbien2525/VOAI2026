# Quy trình chuẩn — Logistic Regression (phân loại nhị phân)
### Bản chuẩn hoá cuối — đã sửa các điểm bất nhất phát hiện ở bản trước

---

## Nguyên tắc vàng

> **Tập test chỉ được chạm đúng MỘT LẦN, ở gần cuối cùng, sau khi mô hình đã chốt xong.**

Mọi việc "thử, so sánh, chọn" — EDA sâu, chọn siêu tham số (C), chọn giữa các mô hình, và **chọn ngưỡng quyết định τ** — đều diễn ra hoàn toàn trong nội bộ **tập train**, dùng Cross Validation để mô phỏng "kiểm tra trên dữ liệu chưa thấy" mà không cần đụng test.

Ký hiệu **[Test set]**: `chưa chạm` / `KHÔNG BAO GIỜ chạm` / `CHẠM 1 LẦN DUY NHẤT Ở ĐÂY`.

---

## Bước 0 — Load Data

**Làm gì:** Đọc file vào, kiểm tra đọc đúng chưa.

**Hàm cần dùng:**
- `pd.read_csv(...)` / `pd.read_excel(...)`
- `df.head()`, `df.shape`

**Lưu ý:** Đặt `SEED` cố định ngay từ đầu, dùng xuyên suốt mọi bước có tính ngẫu nhiên phía sau.

```python
import os
os.environ["PYTHONHASHSEED"] = "0"
import pandas as pd
import numpy as np

SEED = 42
df = pd.read_csv("data.csv")
print(df.shape)
df.head()
```

**[Test set]:** chưa tồn tại — chưa split.

---

## Bước 1 — Understand Data (tổng quan sơ bộ)

**Làm gì:** Xem `shape`, `dtypes`, tỉ lệ thiếu, vài dòng đầu, và tỉ lệ lớp của nhãn.

**Hàm cần dùng:**
- `df.shape`, `df.dtypes`, `df.isna().mean()`
- `df["nhan"].value_counts(normalize=True)`

```python
print(df.dtypes)
print(df.isna().mean())
print(df["nhan"].value_counts(normalize=True))
```

**Lưu ý:** Đây chỉ là xem để hiểu cấu trúc, chưa ra quyết định xử lý gì. Ghi lại tỉ lệ lớp — sẽ dùng ở Bước 2 (stratify) và Bước 7 (so baseline).

**[Test set]:** chưa tồn tại — chưa split.

---

## Bước 2 — Train / Test Split ⚠️ (dời sớm, split trên toàn bộ `df`)

**Làm gì:** Chia `df` (chưa tách X/y) thành `df_train`/`df_test`, kèm `stratify` theo cột nhãn.

> ✅ **Điểm đã sửa so với bản trước**: split thực hiện trên `df` (hoặc trên `X_raw`/`y_raw` nếu bạn đã tách cột từ đầu và không cần xử lý gì thêm ở Bước 4–5), **không split biến `X`/`y` khi chúng chưa được định nghĩa**. Việc tách cột đặc trưng/nhãn chính thức dời xuống Bước 5, sau khi làm sạch cấu trúc (Bước 4).

**Hàm cần dùng:**
- `train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["nhan"])`

```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["nhan"]
)
```

🔬 **Vì sao thêm `stratify`**: nếu không, chia ngẫu nhiên có thể dồn phần lớn mẫu lớp thiểu số vào một tập — nguy hiểm hơn nếu lớp đã lệch sẵn (biết từ Bước 1). `stratify` ép tỉ lệ lớp giữ nguyên xấp xỉ ở cả hai tập.

**Lưu ý:** `df_test` "niêm phong" từ đây.

**[Test set]:** vừa được tạo ra — **niêm phong từ đây**.

---

## Bước 3 — EDA sâu (CHỈ trên `df_train`)

**Làm gì:** Correlation heatmap, phân bố từng đặc trưng, phát hiện outlier, phân bố đặc trưng tách theo lớp.

**Hàm cần dùng:**
- `df_train.corr()`, `sns.heatmap(...)`
- `df_train.groupby("nhan").describe()`, `sns.boxplot(x="nhan", y=cot, data=df_train)`

**Lưu ý:** Chỉ chạy trên `df_train`, không đụng `df_test`.

**[Test set]:** KHÔNG chạm.

---

## Bước 4 — Clean Data (loại A ngay, loại B dời sang Bước 6)

**Làm gì:**
- Loại A (an toàn, áp ngay cho cả `df_train` và `df_test`): sửa kiểu dữ liệu sai, xoá dòng trùng lặp, sửa lỗi định dạng/khoảng trắng/viết hoa-thường.
- Loại B (phụ thuộc thống kê — điền missing bằng mean/median, cắt outlier theo IQR): **chưa làm ở đây**, dời sang Bước 6.

**Hàm cần dùng:**
- `df.astype(...)`, `df.drop_duplicates()`, `df[cot].str.strip()`, `df[cot].str.lower()`

**Lưu ý:** Câu hỏi tự kiểm: *"bước này có cần tính một con số thống kê từ chính dữ liệu không?"* — có thì để dành Bước 6.

**[Test set]:** loại A áp cho cả hai; loại B chưa làm.

---

## Bước 5 — Define X / y (tách cột, mã hoá nhãn đúng thứ tự)

**Làm gì:** Tách cột đặc trưng và nhãn từ `df_train`/`df_test` đã làm sạch cấu trúc. Nếu nhãn ở dạng chuỗi, **mã hoá dựa trên `y_train`, rồi áp cùng ánh xạ đó cho `y_test`** — không fit trên toàn bộ nhãn trước khi split.

> ✅ **Điểm đã sửa so với bản trước**: trước đây gợi ý `LabelEncoder().fit_transform(y)` trên toàn bộ nhãn trước split — về nguyên tắc là để encoder "nhìn thấy" tập nhãn đầy đủ trước khi test bị niêm phong. Bản chuẩn hoá này fit encoder **chỉ trên `y_train`**, đúng tinh thần "fit chỉ trên train" áp dụng nhất quán cho mọi phép biến đổi học từ dữ liệu, kể cả với nhãn.

**Hàm cần dùng:**
- `X_train = df_train.drop(columns=["nhan"])`, `y_train = df_train["nhan"]` (tương tự cho test)
- `LabelEncoder().fit(y_train)` rồi `.transform(y_train)` và `.transform(y_test)`

```python
from sklearn.preprocessing import LabelEncoder

X_train = df_train.drop(columns=["nhan"])
y_train_raw = df_train["nhan"]
X_test = df_test.drop(columns=["nhan"])
y_test_raw = df_test["nhan"]

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)   # fit CHỈ trên train
y_test = le.transform(y_test_raw)         # transform bằng ánh xạ đã học, không fit lại
```

⚠️ **Trường hợp biên cần biết**: nếu `y_test` chứa một lớp chưa từng xuất hiện trong `y_train`, `le.transform(y_test)` sẽ báo lỗi ngay lập tức — đây là hành vi **đúng và nên giữ**, vì nó báo cho bạn biết dữ liệu có vấn đề (lớp lạ), thay vì âm thầm mã hoá sai.

**[Test set]:** chỉ transform bằng ánh xạ học từ train, KHÔNG fit trên test.

---

## Bước 6 — Preprocessing (fit trên train, transform cho cả hai)

**Làm gì:** Điền missing, chuẩn hoá/scale, encode biến phân loại đặc trưng (khác nhãn, đã xử lý ở Bước 5). Đây cũng là nơi thực hiện phần "loại B" còn nợ từ Bước 4.

**Hàm cần dùng:**
- `SimpleImputer(strategy="median").fit(X_train)` rồi transform cả hai
- `StandardScaler().fit(X_train)` rồi transform cả hai

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="median")
X_train_i = imputer.fit_transform(X_train)
X_test_i  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_i)
X_test_c  = scaler.transform(X_test_i)
```

⚠️ **Bẫy cột hằng số**: nếu một cột có `std == 0` trong train, `StandardScaler` chia cho 0 sinh `nan` không báo lỗi.
```python
assert (X_train.std(axis=0) > 1e-12).all(), "Co cot hang so — se sinh nan!"
```

**[Test set]:** transform bằng thống kê của train, KHÔNG tính thống kê từ test.

---

## Bước 7 — Train Model + Cross Validation + Regularization + Compare Models + Chọn ngưỡng τ

**Làm gì:**
1. Cài mô hình cơ bản (sigmoid + cross entropy + gradient descent), hoặc dùng `LogisticRegression`.
2. Cross Validation bằng `StratifiedKFold` (không dùng `KFold` thường, để mỗi fold giữ đúng tỉ lệ lớp).
3. Dò `C` bằng CV, chấm điểm bằng **macro-F1**.
4. So sánh nhiều mô hình (Logistic vs `DummyClassifier` làm mốc) bằng điểm CV.
5. Chọn ngưỡng τ qua xác suất **out-of-fold** (`cross_val_predict`), vẫn trong nội bộ train.
6. **Cân nhắc hướng thay thế cho lớp lệch** — `class_weight="balanced"` — so sánh bằng CV với hướng dò τ, chọn hướng nào cho macro-F1 cao hơn.

> ✅ **Điểm bổ sung so với bản trước**: thêm bước (6) — trước đây quy trình chỉ đi một hướng duy nhất (dò τ) để xử lý lớp lệch, bỏ qua lựa chọn `class_weight="balanced"`. Bản chuẩn hoá này thử cả hai hướng và chọn hướng tốt hơn bằng CV, thay vì mặc định một hướng.

**Hàm cần dùng:**
- Tự cài: `sigmoid(z)`, `cross_entropy(X, y, w, b, lam)`, `gd_logistic(X, y, lr, so_vong, lam)`
- CV & chọn C: `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`, `GridSearchCV(LogisticRegression(max_iter=5000), param_grid={"C": ...}, scoring="f1_macro", cv=skf)`
- So baseline: `DummyClassifier(strategy="most_frequent")`, `cross_val_score(..., scoring="f1_macro")`
- Chọn τ: `cross_val_predict(model, X_train_c, y_train, cv=skf, method="predict_proba")`
- Hướng thay thế: `LogisticRegression(C=..., class_weight="balanced")`

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# --- 2-3: dò C, chấm bằng macro-F1 ---
grid = GridSearchCV(
    LogisticRegression(max_iter=5000),
    param_grid={"C": np.logspace(-3, 3, 20)},
    scoring="f1_macro", cv=skf
)
grid.fit(X_train_c, y_train)
C_tot_nhat = grid.best_params_["C"]

# --- 4: so baseline ---
baseline = DummyClassifier(strategy="most_frequent")
diem_baseline = cross_val_score(baseline, X_train_c, y_train, scoring="f1_macro", cv=skf).mean()
assert grid.best_score_ > diem_baseline, "Mo hinh chua vuot baseline — kiem tra lai pipeline"

# --- 5: chọn tau bằng xác suất out-of-fold (hướng A: threshold tuning) ---
mo_hinh_A = LogisticRegression(C=C_tot_nhat, max_iter=5000)
proba_oof_A = cross_val_predict(mo_hinh_A, X_train_c, y_train, cv=skf, method="predict_proba")[:, 1]

ket_qua_tau = []
for tau in np.linspace(0.05, 0.95, 19):
    y_pred = (proba_oof_A >= tau).astype(int)
    ket_qua_tau.append((tau, f1_score(y_train, y_pred, average="macro", labels=[0, 1])))
tau_tot_nhat, diem_A = max(ket_qua_tau, key=lambda t: t[1])

# --- 6: hướng thay thế B — class_weight="balanced", giữ tau=0.5 mặc định ---
mo_hinh_B = LogisticRegression(C=C_tot_nhat, max_iter=5000, class_weight="balanced")
diem_B = cross_val_score(mo_hinh_B, X_train_c, y_train, scoring="f1_macro", cv=skf).mean()

# --- chốt hướng tốt hơn ---
if diem_A >= diem_B:
    huong_chon = "A"   # threshold tuning, dùng tau_tot_nhat, class_weight mặc định
else:
    huong_chon = "B"   # class_weight="balanced", dùng tau=0.5
print("Chon huong:", huong_chon, "| diem A =", diem_A, "| diem B =", diem_B)
```

⚠️ **Vì sao dùng `cross_val_predict` chứ không phải `predict_proba` trên chính train đã fit**: xác suất sinh ra từ mô hình đã "thấy" mẫu đó lúc học sẽ lạc quan giả tạo. `cross_val_predict` đảm bảo mỗi xác suất đến từ một mô hình chưa từng thấy mẫu đó — mô phỏng đúng tình huống test mà không cần đụng test.

**Lưu ý về giới hạn đã biết**: quy trình này chọn `C` trước, `τ`/`class_weight` sau — là giản lược tuần tự, không phải tìm kiếm đồng thời tối ưu tuyệt đối (nested CV cho cả C và τ/class_weight cùng lúc sẽ chặt hơn nhưng tốn thời gian hơn nhiều — không đáng đánh đổi trong khung giờ thi).

**[Test set]:** KHÔNG chạm — toàn bộ nằm gọn trong train nhờ CV.

---

## Bước 8 — Chốt mô hình cuối cùng

**Làm gì:** Huấn luyện lại một lần cuối trên toàn bộ `X_train_c`, dùng đúng `C_tot_nhat` và hướng đã chọn ở Bước 7 (`tau_tot_nhat` + class_weight mặc định, hoặc `class_weight="balanced"` + τ=0,5).

**Hàm cần dùng:**
- `LogisticRegression(C=C_tot_nhat, max_iter=5000, class_weight=... ).fit(X_train_c, y_train)`

```python
if huong_chon == "A":
    mo_hinh_cuoi = LogisticRegression(C=C_tot_nhat, max_iter=5000)
    tau_dung = tau_tot_nhat
else:
    mo_hinh_cuoi = LogisticRegression(C=C_tot_nhat, max_iter=5000, class_weight="balanced")
    tau_dung = 0.5

mo_hinh_cuoi.fit(X_train_c, y_train)
```

**Lưu ý:** Sau bước này khoá lại, không quay lại Bước 7.

**[Test set]:** KHÔNG chạm.

---

## Bước 9 — Predict trên tập TEST ⚠️ (lần đầu tiên và DUY NHẤT chạm test)

**Làm gì:** Lấy xác suất trên test rồi áp `tau_dung`.

**Hàm cần dùng:**
- `mo_hinh_cuoi.predict_proba(X_test_c)[:, 1]`

```python
proba_test = mo_hinh_cuoi.predict_proba(X_test_c)[:, 1]
y_pred = (proba_test >= tau_dung).astype(int)
```

**Lưu ý:** Trước dòng này, `X_test`/`y_test` chưa từng xuất hiện trong bất kỳ quyết định nào. Từ đây không quay lại chỉnh mô hình dựa trên kết quả.

**[Test set]:** **CHẠM 1 LẦN DUY NHẤT Ở ĐÂY.**

---

## Bước 10 — Evaluate

**Làm gì:** Tính chỉ số phân loại, so với baseline đã ghi ở Bước 7.

**Hàm cần dùng:**
- `accuracy_score`, `f1_score(..., average="macro", labels=[0,1])`, `confusion_matrix`, `classification_report`, `roc_auc_score(y_test, proba_test)`

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

print("accuracy:", accuracy_score(y_test, y_pred))
print("macro-F1:", f1_score(y_test, y_pred, average="macro", labels=[0, 1]))
print("ROC-AUC:", roc_auc_score(y_test, proba_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=[0, 1]))
```

**Lưu ý:** So với `diem_baseline` (Bước 7) trước khi phán xét tốt/xấu, không so với cảm giác chủ quan.

**[Test set]:** đã dùng ở Bước 9, giờ chỉ đọc kết quả.

---

## Bước 11 — Visualization

**Làm gì:** Ma trận nhầm lẫn, đường cong ROC, đường τ-vs-F1 (minh hoạ lại từ Bước 7).

**Hàm cần dùng:**
- `ConfusionMatrixDisplay.from_predictions(y_test, y_pred)`
- `RocCurveDisplay.from_predictions(y_test, proba_test)`
- `plt.plot(...)` cho đường τ-vs-F1

**[Test set]:** chỉ dùng để vẽ, không quyết định thêm gì.

---

## Bước 12 — Final Evaluation & Conclusion

**Làm gì:** Tổng kết mô hình được chọn (C, hướng A/B, τ nếu có), hiệu năng cuối so với baseline, hạn chế còn tồn tại, hướng cải thiện tiếp theo.

**Hàm cần dùng:** không có hàm mới — tổng hợp số liệu đã tính.

**[Test set]:** đã dùng xong, không mở lại.

---

## Bảng tóm tắt nhanh

| # | Bước | Có chạm test? | Ghi chú chuẩn hoá |
|---|---|---|---|
| 0 | Load Data | Chưa tồn tại | — |
| 1 | Understand Data | Chưa tồn tại | +tỉ lệ lớp |
| 2 | Train/Test Split | Vừa niêm phong | Split trên `df`, không split `X/y` chưa tồn tại; +`stratify` |
| 3 | EDA sâu | ❌ Không | +phân bố theo lớp |
| 4 | Clean Data | Loại A: cả hai / B: chưa | — |
| 5 | Define X/y | ❌ Không | LabelEncoder fit **chỉ trên y_train** |
| 6 | Preprocessing | ⚠️ Chỉ transform | — |
| 7 | Train+CV+Reg+Compare+τ | ❌ Không | `StratifiedKFold`, `f1_macro`, +so sánh τ-tuning vs `class_weight="balanced"` |
| 8 | Chốt mô hình cuối | ❌ Không | Dùng đúng hướng đã thắng ở Bước 7 |
| 9 | Predict trên test | ✅ **1 LẦN DUY NHẤT** | `predict_proba` + `tau_dung` |
| 10 | Evaluate | Đọc kết quả | Accuracy/macro-F1/ROC-AUC, so baseline |
| 11 | Visualization | Đọc kết quả | Confusion matrix, ROC curve |
| 12 | Final Evaluation | Đọc kết quả | — |

---

## Ba điểm đã chuẩn hoá so với bản trước (tóm tắt)

1. **Bước 2**: split thực hiện trên `df` nguyên vẹn, không split biến `X`/`y` khi chúng chưa được định nghĩa — sửa mâu thuẫn thứ tự bước.
2. **Bước 5**: `LabelEncoder` fit chỉ trên `y_train`, transform `y_test` bằng ánh xạ đã học — nhất quán tuyệt đối với nguyên tắc "fit chỉ trên train", kể cả với nhãn.
3. **Bước 7**: thêm hướng xử lý lớp lệch thay thế (`class_weight="balanced"`), so sánh bằng CV với hướng dò ngưỡng τ, chọn hướng tốt hơn thay vì mặc định một hướng duy nhất.

Điểm còn giữ nguyên như một giản lược có chủ đích (không phải lỗi): chọn `C` trước rồi mới chọn τ/class_weight — không làm nested CV tìm đồng thời cả hai, vì tốn thời gian không tương xứng trong khung giờ thi cử.
