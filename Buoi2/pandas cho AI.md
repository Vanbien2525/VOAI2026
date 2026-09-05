# Pandas Cho AI — Tài Liệu Tổng Hợp (Bản Hoàn Chỉnh)
*Dựa trên Buổi 02 — Luyện Olympic Trí tuệ nhân tạo (TS. Đỗ Phúc Hảo, 25/8/2026) + phần bổ sung 📌*

> 💡 **Cách đọc tài liệu này:** mỗi khái niệm sẽ có 3 lớp — (1) ý gốc từ bài giảng, (2) ví dụ đời thường để dễ hình dung, (3) code thật kèm giải thích từng dòng. Đừng học thuộc cú pháp, hãy hiểu *vì sao* một dòng code cho ra kết quả đó — vì Pandas có rất nhiều "bẫy im lặng": code chạy được, không báo lỗi, nhưng kết quả sai.

---

## 0. Mục Tiêu Buổi Học — Vì Sao Pandas Quan Trọng Với AI?

Trước khi đưa dữ liệu vào bất kỳ mô hình nào, bạn luôn phải mở được một file CSV **chưa từng thấy** và trả lời 5 câu hỏi sau **trong 15 phút**:

1. Bao nhiêu dòng, bao nhiêu cột, kiểu dữ liệu gì?
2. Nhãn phân bố ra sao, có lệch lớp không?
3. Giá trị nào thiếu, thiếu bao nhiêu?
4. Đặc trưng nào đã tách được hai lớp mà chưa cần mô hình?
5. Có gì bất thường mà đề bài không nói?

Đây chính là **EDA (Exploratory Data Analysis — phân tích khám phá dữ liệu)**.

> 💡 **Ví dụ đời thường:** hãy tưởng tượng bạn được giao nấu ăn cho một nhóm khách lạ. Trước khi bật bếp, bạn phải hỏi: có bao nhiêu người ăn (số dòng), họ ăn chay hay mặn (kiểu dữ liệu), có ai dị ứng gì không (giá trị thiếu/bất thường). Bỏ qua bước hỏi này, bạn có thể nấu rất ngon nhưng... sai hoàn toàn nhu cầu. EDA chính là bước "hỏi trước khi nấu" đó.

### Mười lăm phút này không phải thời gian bỏ đi

Trong một kỳ thi kéo dài sáu tiếng, mười lăm phút nhìn dữ liệu trước khi viết dòng mô hình đầu tiên là khoản đầu tư **lãi nhất**. Nó thường trả lại một đặc trưng miễn phí (một cột hoặc quy luật giúp mô hình đoán đúng mà không cần "học" gì cả), và luôn cứu bạn khỏi việc huấn luyện ba tiếng trên dữ liệu mà bạn hiểu sai.

Nguyên tắc xuyên suốt tài liệu: **hiểu vì sao một dòng code cho ra kết quả đó, không chỉ nhớ cú pháp.**

---

## 1. Bốn Lệnh Đầu Tiên — Luôn Theo Đúng Thứ Tự Này

```python
import pandas as pd

# keep_default_na=False rất quan trọng ở đây — xem lý do đầy đủ ở Mục 2
df = pd.read_csv("duong_dan.csv", keep_default_na=False)
print(df.shape)          # (số dòng, số cột)
print(df.head(3))        # một dòng dữ liệu thật sự trông như thế nào
print(df.info())         # kiểu dữ liệu (dtype) và số lượng giá trị non-null
print(df["nhan"].value_counts())   # phân bố nhãn
```

**Vì sao đúng thứ tự này quan trọng?**
- `shape` cho quy mô — biết mình đang xử lý bài toán lớn hay nhỏ, có cần lo về tốc độ không.
- `head()` cho cảm giác trực quan — cột nào là text, cột nào là số, có ký tự lạ không.
- `info()` cho biết kiểu dữ liệu thật (đôi khi số bị đọc thành string) và cột nào thiếu dữ liệu.
- `value_counts()` trên nhãn cho biết bài toán có **lệch lớp (class imbalance)** không — điều này quyết định bạn có cần accuracy hay phải dùng F1/AUC.

> 💡 **Ví dụ đời thường:** 4 lệnh này giống như khi bạn mới nhận một hộp đồ chuyển nhà — trước tiên đếm có bao nhiêu thùng (`shape`), mở thử 3 thùng xem bên trong là gì (`head`), kiểm tra nhãn dán trên mỗi thùng có đúng không (`info`), rồi đếm xem có bao nhiêu thùng "đồ bếp" so với "đồ phòng ngủ" (`value_counts`). Làm ngược thứ tự này (ví dụ đếm nhãn trước khi biết có bao nhiêu thùng) vẫn ra kết quả, nhưng bạn sẽ mất cảm giác tổng thể.

### Ví dụ minh họa số liệu thật
Giả sử `df["nhan"].value_counts(normalize=True)` cho ra `{1: 0.95, 0: 0.05}`. Nếu bạn train một mô hình luôn đoán nhãn 1, accuracy đã là 95% — con số này đánh lừa bạn tưởng mô hình tốt. Biết điều này *trước khi* train giúp bạn chọn đúng metric ngay từ đầu, thay vì ăn mừng nhầm.

---

## 2. Cái Bẫy NaN — Ô Rỗng Không Vô Hại

Mặc định, `read_csv` biến ô rỗng (`""`) thành `NaN` (Not a Number — nghĩa là "không có giá trị"). Vấn đề là **NaN không cắn ở mọi chỗ**, khiến bạn tưởng mình đã kiểm tra đúng nhưng thực ra chỉ kiểm tra đúng một phần.

> 💡 **Ví dụ đời thường:** NaN giống như một "ô trống bí ẩn" trong bảng điểm — có bạn để trống vì "chưa thi" (thiếu dữ liệu thật), có bạn để trống vì "môn này miễn thi, không cần điểm" (rỗng có ý nghĩa). Pandas mặc định coi TẤT CẢ ô trống là "chưa thi", dù nhiều khi ý người viết dữ liệu lại là "miễn thi". Nhầm lẫn này chính là nguồn gốc của bẫy.

Bảng minh họa từ tài liệu gốc (đo trên cùng một bộ dữ liệu, cột `ban_dich` có 2795 ô rỗng thật sự):

| Cách viết | Có `keep_default_na=False` | Quên tham số (mặc định) |
|---|---|---|
| `(df["ban_dich"].str.len() > 0).sum()` | 3205 | 3205 |
| `(df["ban_dich"] == "").sum()` | 2795 | **0** ❌ |
| `(df["van_ban_goc"] + df["ban_dich"]).isna().sum()` | 0 | **2795** ❌ |

**Giải thích tại sao mỗi dòng cư xử khác nhau:**
- **Dòng 1:** `NaN > 0` cho ra `False` trong Pandas, nên vô tình đúng ở cả hai trường hợp — đây là lý do phép kiểm này **miễn nhiễm mà không chứng minh được là không có lỗi**. Một phép kiểm "luôn đúng" không có nghĩa là dữ liệu của bạn không có bẫy — có thể bạn chỉ đang dùng đúng phép kiểm miễn nhiễm mà không biết.
- **Dòng 2:** so sánh `NaN == ""` luôn cho `False`, nên khi có NaN, phép đếm ô rỗng trả về 0 — một câu trả lời sai hoàn toàn nhưng trông rất hợp lý.
- **Dòng 3:** cộng chuỗi với NaN cho ra NaN (lan truyền), phá hỏng toàn bộ phép nối chuỗi ở các bước sau.

**Kết luận quan trọng:** cùng một câu hỏi, hai cách viết code, hai số phận khác nhau. Bạn có thể viết một đoạn phân tích, thấy con số đúng, kết luận rằng mình đọc dữ liệu đúng, rồi vẫn dính bẫy ở đoạn sau.

**Cách phòng duy nhất đáng tin:** truyền `keep_default_na=False` ngay từ lệnh đọc, nếu bạn biết ô rỗng có ý nghĩa là chuỗi rỗng chứ không phải "thiếu dữ liệu".

### 📌 Bổ sung: Khi nào NÊN để NaN, khi nào không?
Không phải lúc nào NaN cũng là kẻ thù — nó rất hữu ích khi ô rỗng thật sự có nghĩa là "thiếu dữ liệu" (ví dụ cột tuổi bị bỏ trống). Lúc đó bạn muốn Pandas nhận diện là NaN để dùng các công cụ xử lý missing data:

```python
df.isna().sum()                  # đếm số NaN theo từng cột
df["tuoi"].fillna(df["tuoi"].median(), inplace=False)   # điền bằng trung vị
df.dropna(subset=["nhan"])       # bỏ dòng thiếu nhãn (cẩn thận: đổi số dòng! — xem Mục 4)
df["tuoi"].interpolate()         # nội suy cho dữ liệu dạng chuỗi thời gian
```

**Quy tắc quyết định:** ô rỗng là "chuỗi rỗng có ý nghĩa" (như văn bản dịch trống vì câu gốc không cần dịch) → dùng `keep_default_na=False`. Ô rỗng là "giá trị bị thiếu do lỗi thu thập" (như cân nặng không đo được) → giữ NaN mặc định và xử lý bằng `fillna`/`dropna`/`interpolate` một cách có chủ đích.

### 📌 Bổ sung: NaN trong các cột số
Với cột kiểu số, NaN còn ép cả cột từ `int64` sang `float64` (vì `int` không biểu diễn được NaN). Đây là lý do một cột "số nguyên" bỗng hiển thị `1.0, 2.0, NaN` thay vì `1, 2, NaN` — kiểm tra bằng `df.dtypes` để phát hiện sớm.

---

## 3. Lọc, Nhóm, và Apply

| Việc | Cách viết | Ghi chú |
|---|---|---|
| Lọc theo điều kiện | `df[df["nhan"] == 1]` | trả về bảng con, **giữ nguyên index gốc** |
| Thêm cột dẫn xuất | `df["do_dai"] = df["van_ban_goc"].str.split().str.len()` | vector hóa, nhanh |
| Gộp theo nhóm | `df.groupby("nhan")["do_dai"].mean()` | trả về Series/DataFrame mới, index là nhóm |
| Đếm giá trị | `df["nhan"].value_counts(normalize=True)` | `normalize=True` cho tỉ lệ thay vì số đếm |
| Apply | `df.apply(ham, axis=1)` | chạy hàm Python trên từng dòng — **chậm hơn vector hóa 1–2 bậc** |

**Vì sao apply chậm?** Nó quay lại đúng vòng lặp Python thuần mà các phép toán vector hóa (`.str.len()`, `+`, `>`...) được thiết kế để tránh — các phép vector hóa chạy bằng code C bên dưới NumPy, còn `apply` gọi lại hàm Python cho từng dòng một. Chỉ dùng `apply` khi thực sự không có cách vector hóa tương đương.

> 💡 **Ví dụ đời thường:** vector hóa giống như phát cùng lúc một tờ phiếu cho cả lớp và mọi người tự điền song song. `apply` giống như giáo viên đi từng bàn, hỏi từng bạn một câu rồi ghi lại — cùng làm một việc, nhưng chậm hơn hẳn vì phải lặp lại thao tác "đi tới, hỏi, ghi" cho từng người.

### Ví dụ minh họa từ tài liệu gốc
```python
df["do_dai"] = df["van_ban_goc"].str.split().str.len()
df.groupby("nhan")["do_dai"].mean()
# nhãn 1: 4.577 từ, nhãn 0: 4.403 từ
```
Chênh lệch 0,17 từ — nhỏ nhưng khác 0 và đúng hướng dự đoán được (câu bị làm hỏng thường bị xóa bớt từ, nên ngắn hơn). Đây là một **đặc trưng yếu nhưng thật** — không tự phân loại được, nhưng ghép với đặc trưng khác thì có ích. Đừng vội bỏ một đặc trưng chỉ vì nó "yếu" khi đứng một mình.

### 📌 Bổ sung: Khi nào apply là lựa chọn hợp lý?
Khi logic quá phức tạp để viết bằng phép toán vector hóa — ví dụ gọi một hàm xử lý ngôn ngữ tự nhiên phức tạp cho từng câu. Khi đó, cân nhắc `df["col"].map(ham)` (nhanh hơn `apply` một chút cho Series) hoặc dùng thư viện hỗ trợ song song hóa như `swifter` nếu dữ liệu lớn.

### 📌 Bổ sung: `.loc` và `.iloc` — chọn dữ liệu đúng cách
Đây là hai công cụ nền tảng cực kỳ quan trọng để tránh nhầm lẫn giữa "chọn theo nhãn" và "chọn theo vị trí":

```python
df.loc[3, "nhan"]        # chọn theo NHÃN của index và tên cột
df.iloc[3, 0]             # chọn theo VỊ TRÍ số học (0-indexed), bất kể index là gì
df.loc[df["nhan"] == 1, "do_dai"]   # lọc + chọn cột cùng lúc, cách viết chuẩn
```

> 💡 **Ví dụ đời thường:** `.loc` giống như gọi tên học sinh theo **số báo danh** ghi trên thẻ ("cho tôi bài của bạn số báo danh 3"), còn `.iloc` giống như gọi theo **vị trí ngồi vật lý** ("cho tôi bài của bạn ngồi ghế thứ 3 tính từ đầu"). Nếu học sinh đổi chỗ ngồi (index bị xáo trộn sau khi lọc), hai cách gọi này cho ra hai người khác nhau!

Sai lầm phổ biến: sau khi lọc, `index` không còn liên tục (0, 1, 2...) mà giữ nguyên số dòng gốc (0, 2, 5, 7...). Lúc đó `df.loc[3]` sẽ tìm dòng có **nhãn index = 3**, có thể không tồn tại hoặc không phải dòng thứ 3 mà bạn nghĩ. Đây chính là gốc rễ của bẫy căn chỉnh index ở Mục 5.

---

## 4. Thứ Tự Dòng — Thứ Nguy Hiểm Nhất Trong Cả Thư Viện

**`sort_values` là kẻ thù, không phải tiện ích** khi bạn cần nộp bài hoặc ghép dữ liệu trở lại nguồn gốc. Bài nộp phải giữ nguyên thứ tự dòng của file đề — bộ chấm so từng dòng một, và **từ chối chấm nếu thứ tự lệch.**

> 💡 **Ví dụ đời thường:** hãy tưởng tượng cô giáo phát 30 bài kiểm tra theo đúng thứ tự danh sách lớp, bạn thu lại để chấm điểm nhưng lỡ tay sắp xếp lại theo tên cho "dễ nhìn" trước khi trả về cô. Cô giáo đối chiếu theo thứ tự cũ (danh sách lớp) sẽ ghi nhầm điểm cho từng bạn — dù bài làm hoàn toàn đúng, thứ tự sai khiến toàn bộ kết quả bị đảo lộn.

Bốn thao tác quen tay đều **âm thầm** đổi thứ tự hoặc số dòng, không cảnh báo gì:
- `sort_values`, `sort_index`
- `drop_duplicates`, `dropna`
- `groupby(...).apply(...)` rồi `reset_index`
- `merge` — sắp xếp lại theo khóa nối

**Quy tắc phòng thân:** bảng nào sẽ đem nộp/ghép lại thì chỉ được **thêm cột**, không được đổi dòng. Muốn sắp xếp để xem cho dễ thì luôn làm trên `df.copy()`.

### Ví dụ minh họa (từ bài tập mẫu trong tài liệu)
```python
# Cách SAI: sắp xếp cho dễ nhìn, quên sắp lại trước khi ghi ra
sai = de.copy()
sai = sai.sort_values("van_ban_goc")
sai["nhan"] = 1
sai.to_csv("bai-nop/sai.csv", index=False)   # bộ chấm sẽ TỪ CHỐI

# Cách ĐÚNG: chỉ thêm cột, không đổi dòng
dung = de.copy()
dung["nhan"] = 1
dung.to_csv("bai-nop/dung.csv", index=False)
```

**Chi tiết dễ sót nhưng quan trọng:** `index=False` trong `to_csv` là bắt buộc. Thiếu nó, Pandas thêm một cột chỉ mục không tên vào đầu file, làm lệch toàn bộ vị trí cột phía sau — bộ chấm sẽ không tìm thấy cột `van_ban_goc` ở đúng chỗ.

### 📌 Bổ sung: Vì sao "bị từ chối" tốt hơn "bị chấm sai"?
Nếu hệ thống chấm không kiểm tra thứ tự mà cứ chấm bừa theo vị trí, bạn sẽ nhận điểm thấp và đi tìm lỗi ở mô hình — trong khi lỗi thực sự chỉ nằm ở một dòng `sort_values`. Một thông báo từ chối rõ ràng tiết kiệm cho bạn hàng giờ debug nhầm chỗ. Nguyên tắc chung khi làm AI: **luôn viết một hàm kiểm định định dạng trước khi nộp/triển khai**, đừng tin tưởng mù quáng vào pipeline của mình (xem Bài tập 4 ở Mục 9).

---

## 5. Căn Chỉnh Theo Chỉ Mục (Index Alignment) — Bẫy Im Lặng Thứ Hai

**Nguyên lý cốt lõi:** Pandas căn hai Series theo **chỉ mục (index)**, không theo vị trí vật lý. Đây là hành vi mặc định, không phải lỗi — nhưng nếu không biết, nó sẽ âm thầm phá dữ liệu.

```python
con = df[df["nhan"] == 1]        # index giờ là 0, 2, 5, 7,... có "lỗ hổng"
gia_tri = con["van_ban_goc"].str.len()

df["moi"] = gia_tri               # CĂN THEO INDEX: dòng không có trong `con` → NaN, không báo lỗi
df["moi_2"] = gia_tri.values      # CĂN THEO VỊ TRÍ: đúng ngay, hoặc nổ lỗi ngay nếu độ dài không khớp
```

> 💡 **Ví dụ đời thường:** hãy tưởng tượng bạn có một sổ điểm danh 10 người (index 0–9), rồi bạn lọc ra 4 người vắng mặt (index 0, 2, 5, 7). Bạn ghi lý do vắng cho 4 người này vào một tờ giấy riêng. Khi dán tờ giấy đó **theo đúng số thứ tự (index)** vào sổ gốc, 6 người còn lại tự động nhận dòng "không có lý do" (NaN) — không phải vì họ vắng, mà vì Pandas chỉ đơn giản "không tìm thấy lý do khớp với số thứ tự của họ". Đây là hành vi đúng theo thiết kế của Pandas, nhưng dễ khiến người mới hiểu nhầm là lỗi phần mềm.

**Vì sao dòng 1 nguy hiểm hơn dòng 2 dù trông "an toàn" hơn?** Vì nó không báo lỗi — bạn có `df["moi"]` đầy NaN ở những dòng nhãn 0, và nếu vô tình dùng cột này để train, mô hình học trên dữ liệu rác mà bạn không hề biết. Dòng 2 (`.values`) chuyển Series thành mảng NumPy thuần, bỏ hẳn khái niệm index, nên hoặc khớp đúng theo vị trí, hoặc **nổ lỗi ngay lập tức** nếu độ dài không khớp — và lỗi ngay lập tức luôn tốt hơn lỗi âm thầm.

### 📌 Bổ sung: Cách an toàn hơn cả `.values`
```python
df["moi"] = df["van_ban_goc"].str.len().where(df["nhan"] == 1)
```
Cách này tính trên toàn bộ `df` gốc (không lọc trước), nên index luôn khớp tự nhiên, không cần lo về căn chỉnh. `where` giữ giá trị nếu điều kiện đúng, trả NaN nếu sai — rõ ràng và an toàn hơn việc lọc rồi gán ngược.

### 📌 Bổ sung: `reset_index` — dùng đúng lúc, đúng chỗ
Sau khi lọc hoặc groupby, nếu bạn **chắc chắn** không cần giữ thứ tự dòng gốc nữa (ví dụ: đang phân tích thăm dò, không phải nộp bài), có thể dùng:
```python
con = df[df["nhan"] == 1].reset_index(drop=True)
```
`drop=True` để không biến index cũ thành một cột mới không mong muốn. Nhưng nhắc lại nguyên tắc Mục 4: **không dùng cách này trên bảng sẽ nộp/ghép lại.**

---

## 6. Đếm Từ Và Trực Quan Hóa — Mắt Bắt Được Thứ Mà Bảng Số Giấu Đi

### 6.1 Đếm từ vựng bằng `Counter`

Trước khi vẽ, một bước rất rẻ nhưng nhiều bạn bỏ qua: đếm xem văn bản của bạn có bao nhiêu từ khác nhau.

```python
from collections import Counter
dem = Counter(w for cau in df["van_ban_goc"] for w in cau.split())
print("so tu khac nhau:", len(dem))
print(dem.most_common(5))
```

Trên bộ dữ liệu ví dụ trong bài giảng: từ vựng có **333 dạng khác nhau**. Đây là một con số rất nhỏ, và nó là tin tốt: một "ngôn ngữ" chỉ có 333 từ thì có thể học được từ điển bằng thống kê thuần túy, không cần mô hình lớn.

> 💡 **Ví dụ đời thường:** `Counter` giống như bạn đổ hết một hộp Lego ra bàn rồi đếm xem có bao nhiêu **loại** mảnh ghép khác nhau (không phải tổng số mảnh) và loại nào xuất hiện nhiều nhất. Biết trước "chỉ có 333 loại mảnh ghép" giúp bạn biết ngay là không cần một cỗ máy phân loại phức tạp — một bảng tra cứu đơn giản là đủ.

### 6.2 Vẽ biểu đồ — độ dài câu và tần suất từ

```python
import matplotlib
matplotlib.use("Agg")   # không cần cửa sổ hiển thị, ghi thẳng ra file — hữu ích khi chạy trên server
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

# Biểu đồ 1: phân phối độ dài câu, tách theo nhãn
for nhan, mau in ((1, "tab:blue"), (0, "tab:red")):
    ax[0].hist(df.loc[df["nhan"] == nhan, "do_dai"], bins=range(1, 11),
               alpha=0.55, label=f"nhan {nhan}", color=mau)
ax[0].set_title("Do dai cau")
ax[0].legend()

# Biểu đồ 2: 40 từ hay gặp nhất
tan_suat = [c for _, c in dem.most_common(40)]
ax[1].bar(range(len(tan_suat)), tan_suat)
ax[1].set_title("40 tu hay gap nhat")

fig.tight_layout()
fig.savefig("nhin-du-lieu.png", dpi=120)
```

**Vì sao bước này hay bị bỏ qua nhưng lại quan trọng?** Bảng số (như `describe()`) có thể trông giống hệt nhau dù phân phối dữ liệu hoàn toàn khác (ý tưởng nổi tiếng từ "Anscombe's Quartet" — bốn bộ dữ liệu có cùng mean, variance, correlation nhưng hình dạng hoàn toàn khác nhau khi vẽ ra). Histogram theo nhãn cho bạn thấy ngay hai lớp có tách biệt theo đặc trưng này không, có outlier không, có đa đỉnh (bimodal) không — những thứ bảng số trung bình/độ lệch chuẩn không nói lên được.

> 💡 **Ví dụ đời thường:** hai lớp học có thể có điểm trung bình giống hệt nhau (7.0), nhưng lớp A toàn học sinh được 6-8 điểm còn lớp B có nửa lớp 10 điểm và nửa lớp 4 điểm. Nhìn con số trung bình, bạn tưởng hai lớp giống nhau. Nhìn biểu đồ, bạn thấy ngay chúng hoàn toàn khác nhau.

### 📌 Bổ sung: Bộ công cụ trực quan hóa nhanh cho EDA
```python
df["do_dai"].describe()                 # min, max, mean, std, các quartile — cái nhìn số học nhanh
df.corr(numeric_only=True)               # ma trận tương quan giữa các cột số
df.boxplot(column="do_dai", by="nhan")   # so sánh phân phối theo nhóm, thấy rõ outlier
import seaborn as sns
sns.pairplot(df, hue="nhan")             # nhìn tương quan giữa NHIỀU cặp đặc trưng cùng lúc
```

---

## 7. Bài Tập Mẫu (Có Lời Giải) — Chứng Minh `sort_values` Làm Hỏng Bài Nộp

**Đề bài.** Chứng minh bằng số rằng `sort_values` làm hỏng bài nộp, rồi chỉ ra cách viết đúng.

**Phân tích.** Nhiều người sắp lại bảng cho dễ nhìn, làm xong quên sắp về. Bộ chấm so từng dòng theo thứ tự, nên nó sẽ từ chối. Ta hãy cố ý mắc lỗi ấy một lần, trong môi trường an toàn, để nhớ đời.

**Lời giải.**
```python
import pandas as pd
de = pd.read_csv("du-lieu/public_test.csv", keep_default_na=False)

# Cách SAI: sắp xếp cho dễ nhìn, rồi ghi ra nguyên trạng.
sai = de.copy()
sai = sai.sort_values("van_ban_goc")
sai["nhan"] = 1
sai["ban_dich"] = ""
sai.to_csv("bai-nop/sai.csv", index=False)

# Cách ĐÚNG: chỉ thêm cột, không bao giờ đổi thứ tự dòng.
dung = de.copy()
dung["nhan"] = 1
dung["ban_dich"] = ""
dung.to_csv("bai-nop/dung.csv", index=False)
```

Chấm cả hai:
```
python cham-diem.py --bai bai-nop/sai.csv --dap-an du-lieu/dap-an/public_dap_an.csv
python cham-diem.py --bai bai-nop/dung.csv --dap-an du-lieu/dap-an/public_dap_an.csv
```

**Kết quả.** Bản sai bị từ chối ngay ở dòng thứ hai, kèm câu giải thích chỉ đúng nguyên nhân. Bản đúng được chấm bình thường.

**Chú ý một chi tiết dễ sót.** `index=False` trong `to_csv` là bắt buộc. Thiếu nó, Pandas thêm một cột chỉ mục không tên vào đầu file, và bộ chấm sẽ không tìm thấy cột `van_ban_goc` ở đúng chỗ.

---

## 8. 📌 Bổ Sung Quan Trọng: Những Chủ Đề Chưa Có Trong Buổi 02

Đây là các mảng kiến thức Pandas rất hay gặp trong thực tế làm AI nhưng chưa được đề cập ở buổi học — cần nắm để không bị động khi gặp dữ liệu thật.

### 8.1 Kết hợp bảng: `merge` và `concat`
```python
pd.merge(df1, df2, on="id", how="left")     # ghép theo khóa, giữ toàn bộ dòng của df1
pd.concat([df1, df2], axis=0)               # nối theo hàng (gộp hai bộ dữ liệu cùng cột)
pd.concat([df1, df2], axis=1)               # nối theo cột (ghép thêm đặc trưng, PHẢI khớp index)
```
`how` có 4 lựa chọn: `left`, `right`, `inner`, `outer` — quyết định dòng nào được giữ khi khóa không khớp hoàn toàn ở hai bảng. Đây là nguồn lỗi phổ biến: dùng `inner` mà không để ý sẽ âm thầm **mất dòng** nếu khóa nối không khớp 100%.

> 💡 **Ví dụ đời thường:** `merge` giống như ghép hai danh sách khách mời tiệc cưới (một danh sách có tên + số điện thoại, một danh sách có tên + món ăn kiêng) lại thành một bảng theo tên. `how="inner"` chỉ giữ khách có mặt ở **cả hai** danh sách — ai chỉ có trong một danh sách sẽ biến mất không báo trước.

### 8.2 Biến đổi dữ liệu dạng bảng rộng/dài: `pivot_table` và `melt`
```python
df.pivot_table(values="diem", index="hoc_sinh", columns="mon_hoc", aggfunc="mean")
df.melt(id_vars=["hoc_sinh"], var_name="mon_hoc", value_name="diem")
```
Rất hữu ích khi cần đổi giữa dạng "một dòng = một quan sát" (long format, thường cần cho ML) và dạng "một dòng = một thực thể, mỗi cột một biến" (wide format, dễ đọc cho con người).

### 8.3 Mã hóa biến phân loại (categorical encoding) — bước bắt buộc trước khi đưa vào mô hình
```python
pd.get_dummies(df, columns=["gioi_tinh"], drop_first=True)   # one-hot encoding
df["hang"] = df["hang"].astype("category").cat.codes         # label encoding (cẩn thận: tạo thứ tự giả)
```
**Lưu ý quan trọng:** label encoding (gán số 0, 1, 2...) ngầm tạo ra một *thứ tự* giữa các danh mục mà mô hình tuyến tính/khoảng cách (như Logistic Regression, KNN) có thể hiểu nhầm là có ý nghĩa thứ bậc. One-hot encoding tránh vấn đề này nhưng làm tăng số chiều — cân nhắc dùng khi số lượng danh mục không quá lớn.

### 8.4 Dữ liệu thời gian
```python
df["ngay"] = pd.to_datetime(df["ngay"])
df["thu_trong_tuan"] = df["ngay"].dt.dayofweek
df.set_index("ngay").resample("W").mean()   # gộp theo tuần
```
Chuyển cột ngày tháng dạng string sang `datetime` mở khóa toàn bộ `.dt` accessor (lấy năm/tháng/thứ) và `resample` (gộp theo khoảng thời gian) — rất cần cho bài toán chuỗi thời gian.

### 8.5 Tối ưu bộ nhớ và tốc độ cho dữ liệu lớn
```python
df.memory_usage(deep=True)                          # xem cột nào đang ngốn RAM nhất
df["nhan"] = df["nhan"].astype("category")           # cột phân loại lặp lại nhiều → tiết kiệm RAM đáng kể
pd.read_csv("file_lon.csv", chunksize=100_000)        # đọc theo từng khối, tránh tràn RAM
```
Khi dataset không vừa RAM, đọc theo `chunksize` để xử lý từng phần rồi gộp kết quả, thay vì cố đọc toàn bộ một lần.

### 8.6 Method chaining — viết pipeline rõ ràng, dễ debug
```python
ket_qua = (
    df
    .query("nhan == 1")
    .assign(do_dai=lambda d: d["van_ban_goc"].str.split().str.len())
    .groupby("nhan")["do_dai"]
    .agg(["mean", "std", "count"])
)
```
`assign` và `query` cho phép viết pipeline dạng chuỗi, dễ đọc theo thứ tự thực hiện và dễ debug từng bước bằng cách comment tạm các dòng cuối. Đây là phong cách được khuyến khích trong công việc thực tế thay vì gán biến trung gian lặp đi lặp lại.

### 8.7 Phát hiện outlier nhanh bằng thống kê
```python
q1, q3 = df["do_dai"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df["do_dai"] < q1 - 1.5*iqr) | (df["do_dai"] > q3 + 1.5*iqr)]
```
Quy tắc IQR (khoảng tứ phân vị) là cách nhanh và không cần giả định phân phối chuẩn để phát hiện outlier — rất hữu ích trong bước EDA trước khi quyết định có cần loại bỏ hay xử lý riêng các giá trị bất thường.

---

## 9. Bài Tập Tự Làm

### Bài tập 1. Năm câu hỏi trong mười lăm phút
Mở `public_test.csv` và trả lời, mỗi câu bằng một dòng code:
1. Bao nhiêu dòng;
2. Độ dài câu ngắn nhất và dài nhất;
3. Có bao nhiêu từ khác nhau;
4. Từ nào xuất hiện nhiều nhất;
5. Có dòng nào rỗng hoặc trùng nhau không.

**Cổng kiểm:** năm câu trả lời, mỗi câu kèm dòng code sinh ra nó. Không được gõ số bằng tay.

### Bài tập 2. Tái hiện cái bẫy NaN, cả ba dòng của bảng
Đọc `training_set.csv` hai lần, một lần có `keep_default_na=False` và một lần không, rồi chạy đủ ba phép tính trong bảng ở Mục 2.

**Cổng kiểm:** bạn dựng lại được đúng sáu con số của bảng, kể cả dòng đầu tiên cho ra kết quả giống nhau ở cả hai lần. Dòng ấy mới là bài học: một phép kiểm miễn nhiễm với lỗi không chứng minh được rằng lỗi không tồn tại.

### Bài tập 3. Tái hiện cái bẫy căn chỉnh chỉ mục
Lọc ra các câu nhãn 1, tính độ dài của chúng, rồi gán ngược vào bảng gốc bằng cả hai cách: gán thẳng Series và gán `.values`.

**Cổng kiểm:** đếm được số ô NaN mà cách thứ nhất tạo ra, và nói được vì sao cách thứ hai an toàn hơn dù trông thô hơn.

### Bài tập 4. Hàm kiểm định dạng, mang vào phòng thi được
Viết `kiem_bai_nop(duong_de, duong_bai)` trả về danh sách lỗi, kiểm ít nhất:
- Đủ ba cột, đúng tên;
- Số dòng khớp file đề;
- Cột `van_ban_goc` khớp từng dòng một với file đề;
- Cột `nhan` chỉ chứa 0 và 1.

**Cổng kiểm:** hàm phải báo lỗi với bốn file hỏng do bạn tự tạo, và im lặng với một file hợp lệ. Thiếu vế thứ hai thì một hàm luôn báo lỗi cũng qua được bài này.

### Bài tập 5. Một câu hỏi mở
Tìm trong `training_set.csv` một quy luật mà buổi học chưa nhắc tới. Gợi ý chỗ đáng đào: từ đầu câu, từ cuối câu, các đuôi sau dấu gạch nối.

**Cổng kiểm:** một câu khẳng định kèm con số chứng minh. Ví dụ: "trong các câu nhãn 1, từ cuối câu luôn có dấu gạch nối, còn ở nhãn 0 thì tỉ lệ ấy chỉ là x phần trăm".

### 📌 Bài tập bổ sung 6. Luyện `.loc` vs `.iloc`
Cho một DataFrame bất kỳ, lọc ra các dòng có `nhan == 0` rồi thử lấy "dòng thứ 3 của bảng con" bằng cả `.loc[3]` và `.iloc[3]`. So sánh kết quả, giải thích vì sao chúng khác nhau (hoặc có khi trùng nhau tình cờ).

**Cổng kiểm:** giải thích đúng bằng lời vì sao `.loc[3]` không phải lúc nào cũng là "dòng thứ 3 tính từ trên xuống".

### 📌 Bài tập bổ sung 7. Phát hiện outlier bằng IQR
Dùng công thức IQR ở Mục 8.7 để tìm các câu có độ dài bất thường trong `training_set.csv`. In ra 3 câu dài nhất và 3 câu ngắn nhất bị coi là outlier.

**Cổng kiểm:** danh sách outlier kèm giá trị `do_dai` cụ thể, và một câu nhận xét xem outlier có tập trung ở một nhãn nào không.

### 📌 Bài tập bổ sung 8. Viết pipeline bằng method chaining
Viết lại toàn bộ đoạn phân tích ở Mục 1 → Mục 3 (đọc file, lọc nhãn 1, tính độ dài câu, gộp theo nhãn) bằng một chuỗi `.query().assign().groupby().agg()` duy nhất như ở Mục 8.6.

**Cổng kiểm:** kết quả số phải giống hệt cách viết từng bước riêng lẻ ban đầu.

---

## 10. Chuẩn Bị Cho Buổi Sau

Buổi 03 đi trọn một vòng học máy đầu tiên và dạy cách cố định seed (hạt giống ngẫu nhiên).

**Mang theo câu hỏi:** hai lần chạy cùng một chương trình trên cùng một máy, vì sao lại ra hai kết quả khác nhau?

> 💡 Gợi ý nhỏ để suy nghĩ trước: nhiều bước trong pipeline AI (chia train/test, khởi tạo trọng số mô hình, xáo trộn dữ liệu...) đều dùng số ngẫu nhiên. Nếu không "khóa" nguồn ngẫu nhiên đó lại, mỗi lần chạy máy tính sẽ chọn một dãy số ngẫu nhiên khác nhau — giống như tung xúc xắc nhiều lần, mỗi lần ra một kết quả.

---

## 11. Tổng Kết — Checklist 15 Phút Nhìn Dữ Liệu

Trước khi viết dòng mô hình đầu tiên, luôn tự trả lời được 5 câu hỏi gốc, cộng thêm phần mở rộng:

1. **Kích thước & kiểu dữ liệu:** `df.shape`, `df.info()`
2. **Phân bố nhãn:** `df["nhan"].value_counts(normalize=True)` — có lệch lớp không?
3. **Dữ liệu thiếu:** `df.isna().sum()` — nhớ kiểm tra cả trường hợp "rỗng nhưng không phải NaN"
4. **Đặc trưng phân biệt sẵn:** groupby theo nhãn, so sánh mean/median, vẽ histogram
5. **Bất thường:** outlier bằng IQR, trùng lặp bằng `duplicated()`, kiểm tra khóa nối trước khi merge

**Bốn nguyên tắc an toàn cần nhớ:**
- `keep_default_na=False` nếu ô rỗng có ý nghĩa là chuỗi rỗng.
- Không bao giờ đổi thứ tự dòng trên bảng sẽ nộp/ghép lại — chỉ thêm cột.
- Gán giá trị giữa các Series đã lọc → cẩn thận căn chỉnh theo index, ưu tiên `.where()` hoặc `.values` (biết rõ hệ quả) thay vì gán trực tiếp mù quáng.
- Luôn viết một hàm kiểm định định dạng trước khi nộp bài hoặc triển khai pipeline — đừng tin tưởng mù quáng vào code của chính mình.
