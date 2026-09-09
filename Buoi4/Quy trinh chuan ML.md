# Quy trình chuẩn cho một bài toán Học máy có giám sát
### Checklist dùng lại được cho mọi mô hình — không riêng hồi quy tuyến tính

---

## Nguyên tắc vàng — đọc trước khi làm bất cứ bước nào

> **Tập test chỉ được chạm đúng MỘT LẦN, ở gần cuối cùng, sau khi mô hình đã chốt xong.**

Mọi việc "thử, so sánh, chọn" — EDA sâu, chọn siêu tham số, chọn giữa các mô hình, chọn Ridge hay Lasso — đều phải diễn ra hoàn toàn trong nội bộ **tập train**, dùng Cross Validation để mô phỏng "kiểm tra trên dữ liệu chưa thấy" mà không cần đụng tới test thật.

Ở mỗi bước dưới đây, có một dòng **[Test set]** ghi rõ trạng thái: `chưa chạm` / `KHÔNG BAO GIỜ chạm` / `CHẠM 1 LẦN DUY NHẤT Ở ĐÂY`.

---

## Bước 0 — Load Data

**Làm gì:** Đọc file vào (`pd.read_csv`, `pd.read_excel`...), kiểm tra đọc đúng chưa (`df.head()`, `df.shape`).

**Lưu ý:** Không cần seed ở bước này, nhưng nếu dữ liệu sẽ bị shuffle/sample ngẫu nhiên ở bước sau, hãy nhớ đặt `random_state` xuyên suốt cả pipeline ngay từ đầu để tái lập được.

**[Test set]:** chưa tồn tại — chưa split.

---

## Bước 1 — Understand Data (tổng quan sơ bộ)

**Làm gì:** Xem `shape`, `dtypes`, tỉ lệ giá trị thiếu (`df.isna().mean()`), vài dòng đầu, kiểu dữ liệu từng cột.

**Lưu ý quan trọng:** Đây **chỉ là xem để hiểu cấu trúc**, không được dùng để ra bất kỳ quyết định nào ảnh hưởng tới mô hình (không quyết định bỏ cột nào, không quyết định ngưỡng outlier). Nếu bạn bắt đầu vẽ biểu đồ phân tích sâu, tính tương quan — đó đã là EDA sâu, phải để dành cho Bước 3 (sau split).

**[Test set]:** chưa tồn tại — chưa split.

---

## Bước 2 — Train / Test Split ⚠️ (dời lên SỚM, đây là thay đổi lớn nhất so với thói quen thông thường)

**Làm gì:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Lưu ý:** Đặt `random_state` cố định để tái lập được kết quả (khớp nguyên tắc reproducibility đã học ở buổi 03). Tỉ lệ 80/20 hoặc 70/30 tuỳ độ lớn dữ liệu.

**Vì sao dời lên sớm:** Càng làm nhiều việc trước khi split, càng dễ vô tình để thông tin của phần sẽ-thành-test rò rỉ vào các quyết định phía trước (EDA, làm sạch theo thống kê...). Split sớm là cách chắc chắn nhất để tránh việc này — sau bước này, `X_test`/`y_test` coi như "niêm phong", chỉ mở ra ở Bước 9.

**[Test set]:** vừa được tạo ra — **niêm phong từ đây**.

---

## Bước 3 — EDA sâu (CHỈ trên `X_train`)

**Làm gì:** Correlation heatmap, phân bố từng đặc trưng, phát hiện outlier, phân tích quan hệ đặc trưng–nhãn, mọi biểu đồ khám phá.

**Lưu ý:** Mọi lệnh ở bước này chỉ chạy trên `X_train`/`y_train`. Nếu bạn cần xem "dữ liệu trông như thế nào", dùng `X_train`, không dùng `X` gốc.

**[Test set]:** KHÔNG chạm.

---

## Bước 4 — Clean Data (tách rõ 2 loại làm sạch)

**Loại A — Làm sạch cấu trúc** (an toàn, không phụ thuộc thống kê, làm được cả 2 tập ngay bây giờ):
- Sửa kiểu dữ liệu sai (chuỗi số → số)
- Xoá dòng trùng lặp hoàn toàn
- Sửa lỗi định dạng, khoảng trắng thừa, viết hoa/thường không nhất quán

**Loại B — Làm sạch phụ thuộc thống kê** (KHÔNG làm ở đây — dời sang Bước 6):
- Điền missing bằng trung bình/trung vị
- Cắt outlier theo ngưỡng IQR tính từ dữ liệu

**Lưu ý:** Đây là chỗ dễ nhầm nhất trong toàn bộ pipeline. Câu hỏi tự kiểm: *"Bước này có cần tính một con số thống kê (mean, median, std, IQR...) từ chính dữ liệu không?"* — có thì phải để dành cho Bước 6 và chỉ fit trên train.

**[Test set]:** loại A áp được cho cả 2 tập ngay (không tính thống kê nên không rò rỉ); loại B chưa làm.

---

## Bước 5 — Define X / y

**Làm gì:** Xác định cột nào là đặc trưng (`X`), cột nào là nhãn (`y`). Việc này áp dụng cho cả `train` và `test` đã tách sẵn từ Bước 2.

**Lưu ý:** Nếu bạn tạo đặc trưng mới (feature engineering) ở bước này, kiểm tra lại: đặc trưng mới có dùng thống kê nào tính từ toàn bộ dữ liệu không? Nếu có, cùng vấn đề như Bước 6.

**[Test set]:** KHÔNG chạm (chỉ tổ chức lại cột).

---

## Bước 6 — Preprocessing (fit trên train, transform cho cả hai)

**Làm gì:** Điền missing, chuẩn hoá/scale, encode biến phân loại — mọi bước "học một con số từ dữ liệu rồi áp dụng".

```python
tb, sd = X_train.mean(axis=0), X_train.std(axis=0)
sd[sd == 0] = 1.0                       # lưới an toàn cho cột hằng số
X_train_c = (X_train - tb) / sd
X_test_c  = (X_test  - tb) / sd         # DÙNG LẠI đúng tb, sd — KHÔNG tính lại
```

**Lưu ý:** Đây chính là quy tắc cốt lõi đã học ở buổi 04 — `fit` (tính `mean`/`std`/mode...) chỉ trên train, `transform` (áp dụng) cho cả train và test bằng đúng con số đã tính. Nếu dùng scikit-learn, đây chính là ý nghĩa của cặp lệnh `.fit()` / `.transform()` (`fit` trên train, `transform` cho cả hai — không bao giờ `.fit()` trên test).

**[Test set]:** transform bằng thống kê của train, KHÔNG được dùng để tính thống kê.

---

## Bước 7 — Train Model + Cross Validation + Regularization + Compare Models (gộp thành MỘT giai đoạn, tất cả trong train)

**Làm gì trong giai đoạn này (thứ tự nội bộ không quá quan trọng, miễn không đụng test):**
1. Huấn luyện mô hình cơ bản trên `X_train_c`, `y_train`.
2. Dùng **Cross Validation** (ví dụ `KFold`, `cross_val_score`) trên train để ước lượng hiệu năng thật mà không cần chạm test.
3. Thử **Regularization** (Ridge/Lasso) — dò `alpha` bằng CV (ví dụ `RidgeCV`, `LassoCV`, hoặc `GridSearchCV`).
4. **So sánh nhiều mô hình** (Linear thường vs Ridge vs Lasso vs mô hình khác) — so sánh bằng **điểm CV**, không phải điểm test.
5. Chốt ra một mô hình + bộ siêu tham số tốt nhất.

**Lưu ý quan trọng nhất của cả pipeline:** Đây là nơi hay bị sai thứ tự nhất — nhiều người "Train → Predict → Evaluate trên test" trước, rồi *mới* quay lại thử Ridge/Lasso/CV. Làm vậy nghĩa là bạn đã nhìn điểm test trước khi chọn xong mô hình → lần thử tiếp theo dựa trên thông tin đã rò rỉ từ test → **tập test biến thành một tập validation trá hình**, điểm cuối cùng sẽ đẹp hơn thực lực.

**[Test set]:** KHÔNG chạm — toàn bộ việc "thử và chọn" nằm gọn trong train, nhờ CV đóng vai trò tập validation nội bộ.

---

## Bước 8 — Chốt mô hình cuối cùng

**Làm gì:** Với siêu tham số tốt nhất tìm được ở Bước 7, huấn luyện lại mô hình một lần cuối trên **toàn bộ** `X_train_c` (không chia nhỏ ra CV nữa).

**Lưu ý:** Sau bước này, không quay lại Bước 7 nữa — coi như mô hình đã "khoá".

**[Test set]:** KHÔNG chạm.

---

## Bước 9 — Predict trên tập TEST ⚠️ (lần đầu tiên và DUY NHẤT chạm tới test)

**Làm gì:**
```python
y_pred = mo_hinh_cuoi.predict(X_test_c)
```

**Lưu ý:** Đây là ranh giới quan trọng nhất. Trước dòng này, `X_test`/`y_test` chưa từng xuất hiện trong bất kỳ quyết định nào. Từ dòng này trở đi, không được quay lại chỉnh mô hình dựa trên kết quả — nếu kết quả tệ, phải quay lại từ Bước 2 với một lần split/CV mới, không "vá" dựa trên đáp án test đã thấy.

**[Test set]:** **CHẠM 1 LẦN DUY NHẤT Ở ĐÂY.**

---

## Bước 10 — Evaluate

**Làm gì:** Tính các chỉ số (MSE, RMSE, R², MAE...) trên `y_test` so với `y_pred`.

**Lưu ý:** Trước khi phán xét chỉ số này tốt hay xấu, hãy tự hỏi "sàn lý thuyết" (irreducible error) ở đây khoảng bao nhiêu — như đã học ở buổi 04, tránh tinh chỉnh vô ích để đuổi theo phần sai số do nhiễu tự nhiên.

**[Test set]:** đã dùng ở Bước 9, giờ chỉ đọc kết quả.

---

## Bước 11 — Visualization

**Làm gì:** Vẽ dự đoán vs thực tế, residual plot, đường cong loss (nếu có), phân bố sai số.

**Lưu ý:** Đây là bước diễn giải, không ảnh hưởng ngược lại mô hình.

**[Test set]:** chỉ dùng để vẽ, không dùng để quyết định thêm gì.

---

## Bước 12 — Final Evaluation & Conclusion

**Làm gì:** Tổng kết: mô hình nào được chọn, vì sao, hiệu năng cuối cùng, hạn chế còn tồn tại, hướng cải thiện tiếp theo (nếu có dữ liệu mới).

**[Test set]:** đã dùng xong, không mở lại.

---

## Bảng tóm tắt nhanh (dán lên đầu code hoặc note riêng)

| # | Bước | Có chạm test không? |
|---|---|---|
| 0 | Load Data | Chưa tồn tại |
| 1 | Understand Data (sơ bộ) | Chưa tồn tại |
| 2 | Train/Test Split | Vừa niêm phong |
| 3 | EDA sâu | ❌ Không |
| 4 | Clean Data | Loại A: cả hai / Loại B: chưa |
| 5 | Define X/y | ❌ Không |
| 6 | Preprocessing (fit train, transform cả hai) | ⚠️ Chỉ transform, không fit |
| 7 | Train + CV + Regularization + Compare | ❌ Không |
| 8 | Chốt mô hình cuối | ❌ Không |
| 9 | **Predict trên test** | ✅ **1 LẦN DUY NHẤT** |
| 10 | Evaluate | Đọc kết quả đã có |
| 11 | Visualization | Đọc kết quả đã có |
| 12 | Final Evaluation & Conclusion | Đọc kết quả đã có |

---

## Ghi chú để áp dụng cho các mô hình sau này (không riêng linear regression)

- Thứ tự này **tổng quát cho mọi bài toán học máy có giám sát** (hồi quy lẫn phân loại) — thay `MSE/R²` ở Bước 10 bằng `Accuracy/F1/AUC...` khi làm phân loại, mọi bước khác giữ nguyên logic.
- Với mô hình phức tạp hơn (cây quyết định, mạng nơ-ron...), Bước 7 (CV + chọn siêu tham số) sẽ tốn nhiều công sức hơn hẳn — nhưng vị trí của nó trong pipeline **không đổi**: luôn trước khi chạm test.
- Nếu dữ liệu có yếu tố thời gian (time series), `train_test_split` ngẫu nhiên ở Bước 2 **không dùng được** — phải chia theo mốc thời gian (train là quá khứ, test là tương lai), vì xáo trộn ngẫu nhiên sẽ để "tương lai" rò rỉ vào lúc huấn luyện.
