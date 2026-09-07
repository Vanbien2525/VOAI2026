# Bài toán học máy đầu tiên, từ đầu tới cuối
### (Buổi 03 — Luyện Olympic Trí tuệ nhân tạo, phần Nhập môn — tài liệu gốc: TS. Đỗ Phúc Hảo, 25/8/2026)

> Tài liệu này giữ **nguyên vẹn 100% nội dung gốc** của bài giảng (mọi mục, mọi bảng số liệu, mọi đoạn code, mọi ghi chú "đắt giá"), đồng thời **mở rộng thêm ba lớp**:
> 1. **📘 Nội dung gốc** — giữ y nguyên câu chữ, số liệu, code của thầy Hảo.
> 2. **🧠 Ẩn dụ đời thường** — biến khái niệm trừu tượng thành hình ảnh quen thuộc.
> 3. **🔬 Giải thích sâu / code có chú thích từng dòng** — mổ xẻ cơ chế bên dưới, để không chỉ "làm được" mà còn "hiểu vì sao".

---

## Mục lục

0. [Mục tiêu buổi học & Cổng kiểm](#0-mục-tiêu-buổi-học--cổng-kiểm)
1. [Đặc trưng và nhãn (X, y)](#1-đặc-trưng-và-nhãn-x-y)
2. [Tách tập, và vì sao không bao giờ đo trên tập đã học](#2-tách-tập-và-vì-sao-không-bao-giờ-đo-trên-tập-đã-học)
3. [Ba dòng của scikit-learn](#3-ba-dòng-của-scikit-learn)
4. [Cố định seed: bốn dòng, không phải một](#4-cố-định-seed-bốn-dòng-không-phải-một)
5. [Đóng gói main.py](#5-đóng-gói-mainpy)
6. [Hướng dẫn thực hành từng bước](#6-hướng-dẫn-thực-hành-từng-bước)
7. [Bài tập mẫu — main.py hoàn chỉnh](#7-bài-tập-mẫu--mainpy-hoàn-chỉnh)
8. [Bài tập tự làm (1–5)](#8-bài-tập-tự-làm-1–5)
9. [Chuẩn bị cho buổi sau](#9-chuẩn-bị-cho-buổi-sau)
10. [Bảng tổng kết thuật ngữ nhanh](#10-bảng-tổng-kết-thuật-ngữ-nhanh)

---

## 0. Mục tiêu buổi học & Cổng kiểm

### 📘 Nội dung gốc

> Đi trọn một vòng: dữ liệu, tách tập, huấn luyện, dự đoán, đo, đóng gói.

**Cổng kiểm của cả buổi:** Chạy `python main.py` hai lần trên cùng một máy, hai file kết quả phải có cùng **md5**. Khác nhau một byte là trượt, dù mô hình tốt tới đâu.

**Vì sao cổng kiểm lại là md5 chứ không phải điểm số:** Vì quy chế thi nói rằng bài chỉ hợp lệ khi chạy lại sinh ra đúng file đã nộp, trong không quá 20 phút. Một mô hình đạt điểm cao mà mỗi lần chạy ra một kết quả khác thì không hợp lệ, và bạn mất trắng cả sáu tiếng. Đây là ràng buộc mà người mới hầu như không bao giờ nghĩ tới cho đến khi mất bài.

### 🧠 Ẩn dụ đời thường

Hãy tưởng tượng bạn nộp bài thi viết tay. Nếu giám khảo yêu cầu bạn **chép lại y hệt** bài thi đó lần thứ hai mà bạn lại viết ra một bài khác — dù bài mới hay hơn — thì bài thi gốc bị coi là **không đáng tin**, vì không ai biết bạn đã thật sự làm ra nó bằng cách nào, hay chỉ đoán mò.

`md5` giống như một "dấu vân tay" của file: chỉ cần đổi 1 bit trong file, dấu vân tay đổi hoàn toàn. Đây là cách cực rẻ và cực nghiêm để kiểm tra "hai file có giống hệt nhau không" mà không cần so từng dòng bằng mắt.

### 🔬 Giải thích sâu

- **Vì sao một mô hình ML "tốt" vẫn có thể không tái lập được?** Vì bên trong pipeline có rất nhiều **nguồn ngẫu nhiên** ẩn: cách chia tập train/test, cách khởi tạo trọng số, thứ tự duyệt dữ liệu... Nếu không "khoá" hết các nguồn này, mỗi lần chạy lại chương trình, mô hình học ra một bộ tham số hơi khác, dẫn đến dự đoán hơi khác, dẫn đến file output khác — dù code không đổi một chữ nào.
- **md5 là hàm băm (hash function)**: nó nhận một file (chuỗi byte bất kỳ, dài bao nhiêu cũng được) và trả về một chuỗi 32 ký tự hex cố định. Tính chất quan trọng nhất mà bài học này khai thác: **đầu vào giống hệt nhau ⇒ đầu ra giống hệt nhau; đầu vào khác dù chỉ 1 byte ⇒ đầu ra khác hoàn toàn**. Vì vậy so hai md5 tương đương với so hai file byte-by-byte, nhưng nhanh và tiện hơn nhiều.
- **Bài học triết lý ở đây quan trọng hơn kỹ thuật**: trong một cuộc thi có giới hạn thời gian chấm (20 phút), *khả năng tái lập* được xem trọng ngang, thậm chí hơn, độ chính xác của mô hình. Một mô hình 95% điểm nhưng không tái lập được = 0 điểm thực tế. Đây là tư duy "kỹ sư sản xuất" (production engineering mindset) chứ không chỉ là tư duy "nghiên cứu" (research mindset), và nó là thứ phân biệt người làm ML nghiệp dư với người làm ML nghiêm túc.

---

## 1. Đặc trưng và nhãn (X, y)

### 📘 Nội dung gốc

Mọi bài học có giám sát đều quy về hai thứ:

| Ký hiệu | Hình dạng |
|---|---|
| Đặc trưng X | (số mẫu, số đặc trưng) |
| Nhãn y | (số mẫu,) |

Với văn bản, X chưa có sẵn: câu chữ phải được biến thành số trước. Bộ chuyển ở buổi này là `TfidfVectorizer`, thứ biến mỗi câu thành một vector đếm các mẩu ký tự. Buổi 11 sẽ mổ xẻ nó kỹ hơn.

### 🧠 Ẩn dụ đời thường

- **X (đặc trưng)** giống như **hồ sơ mô tả** một đối tượng: chiều cao, cân nặng, tuổi, giới tính... của một bệnh nhân. Mỗi hàng là một người, mỗi cột là một đặc điểm đo được.
- **y (nhãn)** giống như **đáp án** đi kèm hồ sơ đó: bệnh nhân này có bị bệnh X hay không.
- Việc học có giám sát (supervised learning) giống như một học sinh luyện đề: học sinh nhìn "đề bài" (X) và "đáp án mẫu" (y) hàng nghìn lần, để rồi khi gặp đề mới (X mới, chưa biết đáp án) thì tự đoán ra y.
- **Vì sao văn bản cần "dịch" thành số?** Máy tính (cụ thể là các phép toán ma trận bên trong mô hình) chỉ hiểu số, không hiểu chữ "xin chào". `TfidfVectorizer` giống như một **người phiên dịch**: nó đọc câu, đếm xem những "mẩu chữ" nào xuất hiện, và biến thành một dãy số. Hai câu có nội dung gần giống nhau sẽ có hai dãy số "gần" nhau về mặt toán học.

### 🔬 Giải thích sâu

**Về hình dạng (shape):**
- `X.shape = (số mẫu, số đặc trưng)` — đây là ma trận 2 chiều. Ví dụ 6000 câu văn bản, mỗi câu được biến thành vector 5000 con số (5000 "mẩu ký tự" khác nhau xuất hiện trong toàn bộ dữ liệu) thì `X.shape = (6000, 5000)`.
- `y.shape = (số mẫu,)` — đây là vector 1 chiều, **không phải** `(số mẫu, 1)`. Đây là điểm rất dễ nhầm với người mới học NumPy/Pandas: `(6000,)` là một dãy số phẳng, còn `(6000, 1)` là một ma trận cột. Nhiều hàm của scikit-learn (như `train_test_split`, `f1_score`) mong đợi y ở dạng `(n,)`; nếu lỡ truyền vào dạng `(n, 1)`, đôi khi vẫn chạy được nhưng cho ra cảnh báo hoặc kết quả sai lệch (do broadcasting âm thầm).

**Về TfidfVectorizer (được nhắc sơ, để dành buổi 11):**
- TF-IDF là viết tắt của **Term Frequency – Inverse Document Frequency**. Ý tưởng: một từ/mẩu ký tự xuất hiện **nhiều lần trong một câu** thì quan trọng với câu đó (Term Frequency cao), nhưng nếu nó xuất hiện **trong hầu hết mọi câu** (ví dụ chữ "và", "là") thì nó không giúp phân biệt gì cả, nên bị hạ trọng số (Inverse Document Frequency thấp).
- Trong bài này, `analyzer="char_wb"` nghĩa là thay vì tách theo *từ*, nó tách theo **cụm ký tự liên tiếp** (character n-grams), có ích với tiếng Việt vì việc tách từ tiếng Việt phức tạp (một từ có thể gồm nhiều "tiếng"/âm tiết cách nhau bởi dấu cách), còn ký tự thì luôn tách được nhất quán.
- Mối liên hệ X–y ở bài toán này: X là các vector số hoá từ cột `van_ban_goc` (văn bản gốc), y là cột `nhan` (nhãn phân loại, ví dụ 0 hoặc 1).

**Ví dụ minh hoạ cụ thể:**
```
Câu 1: "hôm nay trời đẹp"       → nhãn y = 1
Câu 2: "hôm nay trời mưa to"    → nhãn y = 0
```
Sau khi qua TfidfVectorizer, mỗi câu biến thành một vector số (ví dụ đơn giản hoá, không phải số thật):
```
Câu 1 → X[0] = [0.0, 0.3, 0.5, 0.1, ...]   (5000 con số)
Câu 2 → X[1] = [0.2, 0.0, 0.0, 0.4, ...]
```
Mô hình học máy sẽ tìm ra một quy luật toán học từ những dãy số này để đoán ra y.

---

## 2. Tách tập, và vì sao không bao giờ đo trên tập đã học

### 📘 Nội dung gốc

| Tập | Dùng để |
|---|---|
| huấn luyện | mô hình nhìn thấy và học từ nó |
| kiểm định | bạn nhìn để chọn mô hình và siêu tham số |
| kiểm tra | chỉ đụng vào một lần, ở cuối, để biết mình thật sự được bao nhiêu |

Đo trên tập đã huấn luyện cho ra một con số luôn đẹp hơn sự thật. Đo thật trên bộ dữ liệu này:

| seed | trên tập đã học | trên tập kiểm định | chênh |
|---|---|---|---|
| 0 | 0,7255 | 0,6888 | 0,0366 |
| 1 | 0,7259 | 0,6799 | 0,0460 |
| 2 | 0,7214 | 0,6976 | 0,0238 |
| 42 | 0,7324 | 0,6849 | 0,0475 |

Bảng này nói hai điều, và điều thứ hai quan trọng hơn:

**Thứ nhất**, cột chênh luôn dương. Mô hình bao giờ cũng làm tốt hơn trên thứ nó đã thấy. Tin vào con số bên trái là tự lừa mình.

**Thứ hai**, cột giữa dao động từ 0,6799 tới 0,6976, tức là riêng việc đổi seed đã làm điểm nhảy gần hai phần trăm. Nên nếu bạn sửa mô hình và thấy điểm tăng 0,005, bạn chưa cải tiến được gì cả: bạn vừa đo nhiễu. Muốn kết luận, phải đo trên nhiều lần chia khác nhau rồi lấy trung bình, và đó là nội dung buổi 07.

### 🧠 Ẩn dụ đời thường

Hãy tưởng tượng một học sinh ôn thi bằng cách **học thuộc lòng chính đề thi mẫu** mà giáo viên đã cho làm ở nhà, thay vì hiểu bản chất kiến thức. Khi giáo viên hỏi lại đúng những câu đó, học sinh trả lời như máy — điểm 10. Nhưng khi thi thật với đề mới (dù cùng dạng), học sinh bối rối vì chưa từng thấy con số cụ thể này — điểm chỉ 6 hoặc 7.

- **Tập huấn luyện** = bộ đề học sinh đã học thuộc.
- **Tập kiểm định** = bộ đề "thi thử" mà giáo viên tạo riêng để xem học sinh có thực sự hiểu bài không, dùng để quyết định có nên đổi phương pháp học không.
- **Tập kiểm tra** = kỳ thi thật, chỉ được thi một lần, không được luyện trước bằng chính đề đó — nếu luyện trước, kết quả không còn phản ánh thực lực nữa (đây gọi là **rò rỉ dữ liệu — data leakage**).

Việc "chênh luôn dương" giống như việc học sinh luôn làm bài đã học thuộc tốt hơn bài lạ — điều này *luôn đúng*, không phải vì học sinh giỏi hơn, mà vì đơn giản là nó đã thấy đáp án trước đó rồi.

### 🔬 Giải thích sâu

**Vì sao điểm trên tập train luôn cao hơn tập validation (overfitting)?**
Về bản chất toán học, mô hình được *tối ưu hoá trực tiếp* để giảm sai số trên chính dữ liệu huấn luyện (đó là mục tiêu của hàm mất mát khi `fit()`). Vì vậy nó "vừa khít" (fit) với dữ liệu đó tốt hơn bất kỳ dữ liệu nào nó chưa từng thấy. Khoảng cách giữa điểm train và điểm validation gọi là **generalization gap** (khoảng hở tổng quát hoá) — khoảng hở càng lớn, mô hình càng có xu hướng "học vẹt" (overfit) thay vì học được quy luật tổng quát.

**Vì sao điểm validation lại dao động theo seed (từ 0,6799 đến 0,6976)?**
Khi bạn đổi `random_state` trong `train_test_split`, bộ dữ liệu được chia thành train/validation theo một cách *ngẫu nhiên khác*. Có lần validation set "may mắn" toàn câu dễ đoán, có lần "xui" gặp toàn câu khó/hiếm gặp. Với chỉ 6000 dòng dữ liệu và validation set chỉ chiếm 20% (khoảng 1200 dòng), độ dao động thống kê (statistical noise) này là hoàn toàn bình thường và **không thể loại bỏ**, chỉ có thể đo lường và tính trung bình.

**Bài học quan trọng nhất của mục này — "ngưỡng nhiễu" (noise floor):**
Nếu độ dao động tự nhiên do đổi seed đã là ~2% (từ 0,68 đến 0,70), thì bất kỳ cải tiến nào cho điểm tăng *ít hơn* 2% đều **không đáng tin** — có thể chỉ là may mắn trúng seed đẹp, không phải do mô hình thực sự tốt hơn. Đây chính là lý do vì sao buổi 07 sẽ giới thiệu **cross-validation** (kiểm định chéo): thay vì chia 1 lần rồi tin vào 1 con số, ta chia dữ liệu thành nhiều "lát" (fold) khác nhau, đo trên từng lát, rồi lấy trung bình + độ lệch chuẩn — từ đó biết được "cải tiến" của mình có thật sự vượt qua nhiễu ngẫu nhiên hay không.

**Ví dụ cụ thể để thấm bài học:**
Giả sử bạn thử nghiệm 2 mô hình A và B, đo bằng 1 seed duy nhất:
- Mô hình A: macro-F1 = 0,688
- Mô hình B: macro-F1 = 0,693

Nhìn thoáng qua, B có vẻ tốt hơn A. Nhưng nếu biết rằng chỉ riêng việc đổi seed cũng đã tạo ra chênh lệch ~0,018 (1,8%), thì khoảng cách 0,005 giữa A và B **nằm gọn trong vùng nhiễu** — bạn không thể kết luận B tốt hơn A một cách chắc chắn. Phải chạy nhiều seed cho cả A và B rồi so trung bình.

---

## 3. Ba dòng của scikit-learn

### 📘 Nội dung gốc

```python
mo_hinh.fit(X_train, y_train)   # learn
y_doan = mo_hinh.predict(X_test)  # apply
diem = f1_score(y_test, y_doan, average="macro", labels=[0, 1], zero_division=0)
```

**`labels=[0, 1]` không phải trang trí:** Thiếu nó, khi một lớp vắng mặt thì scikit-learn lặng lẽ trung bình trên một lớp thay vì hai, và điểm nhảy từ 0,50 lên 1,00. Đề bài định nghĩa macro-F1 là trung bình của đúng hai lớp. Buổi 06 sẽ mổ kỹ chỗ này.

### 🧠 Ẩn dụ đời thường

Ba dòng này giống như quy trình khám bệnh:
- `fit()` = **giai đoạn học nghề**: bác sĩ trẻ xem hàng nghìn ca bệnh đã có chẩn đoán (X_train, y_train) để rút ra kinh nghiệm.
- `predict()` = **giai đoạn hành nghề**: gặp bệnh nhân mới (X_test, chưa biết bệnh gì), bác sĩ dựa vào kinh nghiệm để đưa ra chẩn đoán (y_doan).
- `f1_score(...)` = **giai đoạn hội đồng đánh giá tay nghề**: so sánh chẩn đoán của bác sĩ với chẩn đoán thật (y_test) để ra một điểm số duy nhất phản ánh "bác sĩ này giỏi cỡ nào".

Còn `labels=[0, 1]` giống như việc **quy định trước** hội đồng phải chấm điểm dựa trên cả hai loại bệnh (bệnh A và bệnh B), chứ không được chỉ chấm dựa trên loại bệnh nào tình cờ xuất hiện trong đợt khám đó. Nếu không quy định trước, và đợt khám đó tình cờ không có ca bệnh B nào, hội đồng có thể "quên mất" bệnh B tồn tại và chỉ chấm dựa trên bệnh A — khiến điểm số bị thổi phồng sai lệch.

### 🔬 Giải thích sâu

**F1-score là gì, macro-F1 là gì?**

F1-score của một lớp (class) là trung bình điều hoà (harmonic mean) giữa **precision** (độ chính xác) và **recall** (độ bao phủ):

```
F1 = 2 × (precision × recall) / (precision + recall)
```

- **Precision**: trong số những mẫu mô hình *dự đoán* là lớp này, bao nhiêu % là đúng thật?
- **Recall**: trong số những mẫu *thực sự* thuộc lớp này, mô hình bắt được bao nhiêu %?

**Macro-F1** = tính F1 riêng cho từng lớp, rồi lấy **trung bình cộng đơn giản** (không quan tâm lớp nào có nhiều mẫu hơn):

```
macro-F1 = (F1_lớp_0 + F1_lớp_1) / 2
```

Điều này khác với "micro-F1" (gộp tất cả các dự đoán lại tính chung) — macro-F1 đối xử công bằng với lớp thiểu số, không để lớp đông áp đảo điểm số. Đây là lý do các bộ đề thi Olympic thường chọn macro-F1: nó buộc mô hình phải học tốt **cả hai lớp**, không được "lười" chỉ đoán theo lớp đông hơn.

**Vì sao thiếu `labels=[0, 1]` lại nguy hiểm đến mức điểm nhảy từ 0,50 lên 1,00?**

Mặc định, `f1_score` chỉ tính trung bình trên các lớp **thực sự xuất hiện** trong `y_test` và `y_doan` được truyền vào. Hãy tưởng tượng một batch dữ liệu kiểm tra nhỏ mà tình cờ không có mẫu nào thuộc lớp 1 (hiếm gặp), cả `y_test` và `y_doan` chỉ toàn số 0. Khi đó:
- Không có `labels=[0, 1]`: sklearn tự dò thấy chỉ có 1 lớp (lớp 0) → tính macro-F1 chỉ trên lớp 0 → vì mọi dự đoán đều đúng là 0 → F1 = 1,00 (hoàn hảo giả tạo!).
- Có `labels=[0, 1]`: sklearn *ép buộc* phải tính cả F1 của lớp 1, dù lớp 1 không xuất hiện. Vì không có mẫu nào để tính recall/precision cho lớp 1, kết quả sẽ là 0/0 → nhờ `zero_division=0` mà không bị lỗi, F1_lớp_1 = 0 → macro-F1 = (1,00 + 0) / 2 = 0,50 → phản ánh đúng thực tế: mô hình *hoàn toàn không dự đoán được lớp 1*.

Đây là lý do con số "từ 0,50 nhảy lên 1,00" trong tài liệu gốc — chênh lệch này **không phải do mô hình tốt lên**, mà do một dòng code thiếu tham số làm sai lệch hoàn toàn phép đo. Đây là một trong những lỗi nguy hiểm nhất vì nó **không báo lỗi (silent bug)** — chương trình vẫn chạy trơn tru, chỉ có con số là dối trá.

**Vì sao `zero_division=0` cần thiết?** Khi một lớp không có mẫu dự đoán nào (mẫu số của precision/recall = 0), phép chia cho 0 thường sẽ ném ra cảnh báo hoặc lỗi. Tham số này bảo sklearn: "gặp trường hợp chia cho 0 thì cứ trả về 0, đừng dừng chương trình lại."

---

## 4. Cố định seed: bốn dòng, không phải một

### 📘 Nội dung gốc

```python
import os
import random
import numpy as np

SEED = 20260825
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass
```

Ngoài bốn dòng trên, mọi hàm của scikit-learn có tham số `random_state` đều phải được truyền tay: `train_test_split`, `LogisticRegression`, `KMeans`, `RandomForestClassifier`, và nhiều hàm khác.

**Ba nguồn ngẫu nhiên hay bị bỏ sót:**

1. `train_test_split` không có `random_state`: mỗi lần chạy chia một kiểu, nên điểm nhảy mà bạn tưởng do mô hình.
2. Thứ tự duyệt set hoặc dict cũ trong Python phụ thuộc `PYTHONHASHSEED`. Đặt biến môi trường bên trong chương trình đã muộn với nhiều trường hợp; an toàn nhất là không để kết quả phụ thuộc thứ tự duyệt tập hợp.
3. Thư viện gọi ngầm bộ sinh ngẫu nhiên toàn cục, ví dụ khi khởi tạo trọng số.

### 🧠 Ẩn dụ đời thường

Cố định seed giống như việc **khoá tất cả các "con xúc xắc" đang được gieo bí mật** trong hậu trường của chương trình. Một pipeline ML thực chất tung rất nhiều "xúc xắc" khác nhau mà bạn không nhìn thấy:
- Xúc xắc số 1: cách xáo bài để chia bộ dữ liệu (train/test split).
- Xúc xắc số 2: cách khởi tạo "điểm xuất phát" của mô hình trước khi học (trọng số ban đầu).
- Xúc xắc số 3: thứ tự duyệt qua các phần tử trong một cái túi (set/dict) mà Python nội bộ đôi khi xáo trộn để tăng tốc.

Nếu bạn chỉ khoá 1 trong 3 con xúc xắc mà quên 2 con còn lại, kết quả cuối cùng **vẫn ngẫu nhiên** như thường — giống như khoá 1 cửa nhưng để ngỏ 2 cửa sổ, kẻ trộm (sự ngẫu nhiên) vẫn lẻn vào được.

### 🔬 Giải thích sâu

**Giải thích từng dòng trong khối 4 dòng:**

| Dòng lệnh | Khoá nguồn ngẫu nhiên nào |
|---|---|
| `os.environ["PYTHONHASHSEED"] = str(SEED)` | Cách Python băm (hash) các đối tượng như string, ảnh hưởng tới thứ tự duyệt `set`/`dict` cũ |
| `random.seed(SEED)` | Module `random` chuẩn của Python (dùng bởi nhiều thư viện nội bộ) |
| `np.random.seed(SEED)` | Bộ sinh số ngẫu nhiên toàn cục của NumPy (rất nhiều hàm sklearn dùng ngầm bên dưới cái này) |
| `torch.manual_seed(SEED)` (nếu có PyTorch) | Bộ sinh số ngẫu nhiên của PyTorch — dùng khi khởi tạo trọng số mạng neural |

**Vì sao phải đặt `PYTHONHASHSEED` *trước khi* chương trình Python khởi động, chứ không phải trong lúc chạy?**

Đây là chi tiết tinh vi nhất trong cả mục này. Python băm các chuỗi (string) bằng một giá trị "muối" (salt) ngẫu nhiên được chọn **ngay khi tiến trình Python khởi động** (để chống một kiểu tấn công bảo mật gọi là hash-flooding). Việc gán `os.environ["PYTHONHASHSEED"]` *bên trong* file `main.py` chỉ có tác dụng nếu về sau bạn tự khởi động một tiến trình con mới (subprocess) đọc lại biến môi trường đó — bản thân tiến trình Python **hiện tại** đã lỡ chọn salt ngẫu nhiên từ trước khi dòng code đó chạy tới. Đây chính là lý do tài liệu gốc cảnh báo: "Đặt biến môi trường bên trong chương trình đã muộn với nhiều trường hợp." Cách an toàn tuyệt đối là đặt biến môi trường **từ bên ngoài**, trước khi gọi `python main.py` (ví dụ `PYTHONHASHSEED=20260825 python main.py`), hoặc đơn giản hơn — như tài liệu khuyên — là **thiết kế code sao cho không bao giờ phụ thuộc vào thứ tự duyệt của `set`/`dict`** (luôn `sorted()` trước khi duyệt, nếu thứ tự có ảnh hưởng tới kết quả).

**Vì sao phải bọc `import torch` trong `try/except ImportError`?**

Vì không phải máy chấm bài nào cũng cài PyTorch. Nếu code bạn giả định lúc nào cũng có torch mà không bọc try/except, trên máy không có torch chương trình sẽ crash ngay từ dòng import — dù bài của bạn thậm chí chẳng dùng gì tới torch. `pass` ở nhánh `except` nghĩa là "nếu không có thư viện này thì bỏ qua, không sao cả, không cần torch cũng chạy được."

**Mổ xẻ ba nguồn ngẫu nhiên hay bị bỏ sót — với ví dụ cụ thể:**

1. **`train_test_split` thiếu `random_state`**: Mỗi lần gọi hàm này mà không cố định seed, nó dùng bộ sinh số ngẫu nhiên toàn cục của NumPy tại thời điểm đó để xáo và cắt dữ liệu. Ngay cả khi bạn đã `np.random.seed(SEED)` ở đầu file, nếu *giữa* dòng đó và dòng gọi `train_test_split` có bất kỳ đoạn code nào khác cũng "rút" một vài số ngẫu nhiên (ví dụ để khởi tạo cái gì đó khác), trạng thái bộ sinh số đã bị dịch chuyển, kết quả chia tập sẽ khác đi giữa 2 lần chạy nếu thứ tự các lệnh "rút số" thay đổi. Vì vậy cách an toàn nhất vẫn là truyền `random_state=SEED` **trực tiếp** vào hàm, không phụ thuộc vào trạng thái toàn cục.

2. **Thứ tự duyệt `set`/`dict` cũ**: Ví dụ nếu bạn có đoạn code kiểu `for tu in tap_hop_tu_vung:` mà `tap_hop_tu_vung` là một `set`, thứ tự lặp qua các phần tử của `set` trong Python **không được đảm bảo cố định** giữa các lần chạy khác nhau của tiến trình (phụ thuộc hash seed). Nếu thứ tự này ảnh hưởng đến kết quả cuối (ví dụ bạn gán chỉ số cột theo thứ tự duyệt), file kết quả sẽ khác nhau dù dữ liệu và mô hình giống hệt.

3. **Thư viện gọi ngầm bộ sinh số toàn cục**: Ví dụ khi `LogisticRegression` với solver ngẫu nhiên (không phải `lbfgs`) cần khởi tạo trọng số ban đầu, nó "mượn" số ngẫu nhiên từ NumPy. Nếu bạn không set `np.random.seed` từ đầu, hoặc không truyền `random_state` riêng cho mô hình, quá trình khởi tạo này sẽ khác nhau mỗi lần chạy.

**Nguyên tắc vàng rút ra:** Đừng tin rằng "tôi đã set seed rồi" là đủ — hãy **kiểm chứng bằng thực nghiệm** (chạy 2 lần, so md5) thay vì suy luận lý thuyết. Bài tập 1 ở phần sau chính là bài tập rèn tư duy này.

---

## 5. Đóng gói main.py

### 📘 Nội dung gốc

Quy chế đòi một thư mục `Final/` chứa `Tac_vu_1/main.py` và `Tac_vu_2/main.py`. Chạy `python main.py` phải sinh ra `submission.csv`, trong không quá 20 phút.

**Ba điều kiện dễ vi phạm mà không ai nhắc:**
- Chạy được một mình, không cần notebook, không cần biến đã định nghĩa từ ô trước.
- Đường dẫn tương đối, tính từ chính chỗ đặt `main.py`.
- Không tải gì từ mạng, không đọc file nằm ngoài bộ dữ liệu đề cho.

### 🧠 Ẩn dụ đời thường

Hãy tưởng tượng bạn phải gửi một "công thức nấu ăn" cho một đầu bếp ở một bếp hoàn toàn xa lạ, không biết bạn là ai, không có sẵn nguyên liệu ngoài những gì đề bài cho.
- **"Chạy được một mình"** = công thức không được viết kiểu "tiếp tục làm nốt món đang dở từ bước trước" (như code Jupyter Notebook chạy từng ô, phụ thuộc biến đã có sẵn trong bộ nhớ) — đầu bếp mới phải làm được từ đầu tới cuối chỉ với đúng tờ công thức đó.
- **"Đường dẫn tương đối tính từ main.py"** = công thức phải ghi "lấy muối *trong tủ bếp này*", không phải "lấy muối *trong tủ bếp nhà tôi*" — vì đầu bếp chấm bài đứng ở một bếp khác, thư mục làm việc khác.
- **"Không tải gì từ mạng"** = đầu bếp chấm thi không có kết nối internet hay quyền truy cập kho nguyên liệu riêng của bạn; mọi thứ cần dùng phải nằm sẵn trong gói đề bài.

### 🔬 Giải thích sâu

**Vì sao "chạy được một mình" lại quan trọng?**
Notebook (`.ipynb`) rất tiện khi ta tự phát triển và thử nghiệm, vì các ô (cell) chia sẻ chung một "bộ nhớ" — biến định nghĩa ở ô 3 vẫn còn khi chạy ô 10. Nhưng khi *nộp bài*, giám khảo thường chạy bằng lệnh `python main.py` — một tiến trình hoàn toàn mới, không có trạng thái nào từ trước. Nếu file của bạn lỡ dùng một biến chỉ được định nghĩa ở "ô notebook trước đó" mà quên copy vào `main.py`, chương trình sẽ báo lỗi `NameError` ngay lập tức.

**Vì sao "đường dẫn tương đối tính từ main.py" khác với "đường dẫn tương đối tính từ thư mục hiện hành (cwd)"?**
Đây là lỗi cực kỳ phổ biến. Khi bạn tự chạy `python main.py` ngay trong thư mục chứa nó, hai cách này *trông giống nhau* nên bạn không phát hiện ra sự khác biệt. Nhưng khi giám khảo chạy lệnh kiểu `python /noi/khac/Final/Tac_vu_1/main.py` từ một thư mục hoàn toàn khác (ví dụ thư mục home `~`), nếu code bạn viết `pd.read_csv("du-lieu/training_set.csv")` (đường dẫn tương đối theo *thư mục hiện hành* — nơi lệnh được gõ), Python sẽ đi tìm thư mục `du-lieu` bên trong `~`, không tìm thấy, và báo lỗi `FileNotFoundError`. Giải pháp (sẽ thấy ở mục 7) là dùng `os.path.dirname(os.path.abspath(__file__))` để luôn lấy đúng vị trí *file main.py đang nằm*, bất kể nó được gọi từ đâu.

**Vì sao "không tải gì từ mạng"?**
Vì môi trường chấm thi thường bị **cô lập mạng** (để chống gian lận, chống rò rỉ dữ liệu thi ra ngoài, và đảm bảo mọi thí sinh thi trong cùng điều kiện). Nếu code của bạn có dòng gọi API để tải một mô hình pretrained từ internet, trên máy chấm không có mạng, dòng đó sẽ treo hoặc lỗi timeout, và bạn mất hết điểm dù logic ML hoàn toàn đúng.

---

## 6. Hướng dẫn thực hành từng bước

### Bước 1: Một vòng đầy đủ, ngắn nhất có thể

#### 📘 Nội dung gốc

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 0
df = pd.read_csv("du-lieu/training_set.csv", keep_default_na=False)
X = df["van_ban_goc"].tolist()
y = df["nhan"].tolist()

# stratify keeps the label ratio the same in both halves. Without it, a lucky split can
# hand you a validation set that is easier or harder than the real thing.
X_hoc, X_kiem, y_hoc, y_kiem = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

mo_hinh = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
    LogisticRegression(max_iter=2000, C=4.0, random_state=SEED))

mo_hinh.fit(X_hoc, y_hoc)

for ten, Xs, ys in (("da hoc", X_hoc, y_hoc), ("kiem dinh", X_kiem, y_kiem)):
    diem = f1_score(ys, mo_hinh.predict(Xs), average="macro",
                     labels=[0, 1], zero_division=0)
    print("%-10s macro-F1 %.4f" % (ten, diem))
```

Kết quả: 0,7255 trên tập đã học, 0,6888 trên tập kiểm định.

#### 🔬 Code có chú thích từng dòng

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer   # bộ "phiên dịch" chữ -> số
from sklearn.linear_model import LogisticRegression            # mô hình phân loại tuyến tính
from sklearn.metrics import f1_score                            # hàm đo điểm
from sklearn.model_selection import train_test_split            # hàm chia tập
from sklearn.pipeline import make_pipeline                      # ghép nhiều bước xử lý thành 1 khối

SEED = 0   # "hạt giống" ngẫu nhiên — cố định để tái lập được kết quả chia tập & mô hình

# keep_default_na=False: ngăn Pandas tự ý biến các ô như "NA", "null" thành NaN.
# Với dữ liệu văn bản, một câu vô tình chứa chữ "NA" (ví dụ viết tắt) không nên bị
# hiểu nhầm thành giá trị thiếu.
df = pd.read_csv("du-lieu/training_set.csv", keep_default_na=False)

X = df["van_ban_goc"].tolist()   # cột văn bản gốc -> list các câu (chưa số hoá)
y = df["nhan"].tolist()          # cột nhãn -> list các số 0/1

# stratify=y: đảm bảo tỉ lệ lớp 0/lớp 1 trong X_hoc và X_kiem GIỐNG với tỉ lệ trong y gốc.
# Nếu không có nó, một lần chia "xui" có thể dồn hầu hết lớp hiếm vào 1 bên, khiến
# tập kiểm định trở nên dễ hoặc khó hơn thực tế một cách giả tạo.
X_hoc, X_kiem, y_hoc, y_kiem = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

mo_hinh = make_pipeline(
    # analyzer="char_wb": tách theo cụm ký tự (character n-gram), tốt cho tiếng Việt
    # ngram_range=(2, 5): lấy các cụm từ 2 đến 5 ký tự liên tiếp
    # min_df=2: bỏ qua cụm ký tự nào xuất hiện ít hơn 2 lần trong toàn bộ dữ liệu (giảm nhiễu)
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
    # C=4.0: nghịch đảo của độ mạnh chính quy hoá (regularization).
    #   C nhỏ  -> mô hình bị "phạt" nặng nếu trọng số lớn -> đơn giản hơn, ít overfit hơn
    #   C lớn  -> mô hình được tự do khớp sát dữ liệu hơn -> dễ overfit hơn nếu quá lớn
    # max_iter=2000: số vòng lặp tối đa để thuật toán tối ưu hội tụ (tăng lên nếu bị cảnh báo
    #   "chưa hội tụ" — ConvergenceWarning)
    LogisticRegression(max_iter=2000, C=4.0, random_state=SEED))

mo_hinh.fit(X_hoc, y_hoc)   # học: TfidfVectorizer học từ vựng, LogisticRegression học trọng số

for ten, Xs, ys in (("da hoc", X_hoc, y_hoc), ("kiem dinh", X_kiem, y_kiem)):
    diem = f1_score(ys, mo_hinh.predict(Xs), average="macro",
                     labels=[0, 1], zero_division=0)
    print("%-10s macro-F1 %.4f" % (ten, diem))
    # "%-10s" căn trái chuỗi trong 10 ký tự, "%.4f" in số thập phân với 4 chữ số sau dấu phẩy
```

### 🧠 Ẩn dụ cho "stratify"

`stratify=y` giống như khi chia một lớp học 30 học sinh (20 giỏi, 10 trung bình) thành 2 nhóm để thi đấu, bạn cố tình chia sao cho **mỗi nhóm đều có tỉ lệ giỏi/trung bình xấp xỉ 2:1** — thay vì chia ngẫu nhiên hoàn toàn có thể lỡ dồn hết học sinh giỏi vào một nhóm.

### Bước 2: Đổi seed và nhìn con số nhảy

#### 📘 Nội dung gốc

Chạy lại đoạn trên với SEED bằng 1, 2, rồi 42. Ghi bốn con số kiểm định vào một bảng. Bạn sẽ thấy đúng dải 0,68 tới 0,70 của bảng ở mục 2.2.

> **Bài học đắt nhất buổi này:** Trước khi tin bất kỳ cải tiến nào, hãy hỏi: độ nhảy của nó có lớn hơn độ nhảy do đổi seed không? Nếu không, bạn chưa cải tiến gì.

#### 🔬 Giải thích sâu

Đây thực chất là bài tập tự tay tái tạo bảng đã cho ở mục 2 — cách tốt nhất để một khái niệm trừu tượng ("ngưỡng nhiễu do seed") trở thành trực giác thật sự là **tự mình nhìn thấy con số nhảy lên nhảy xuống bằng chính tay mình chạy**, không chỉ đọc bảng có sẵn. Hãy viết một vòng lặp `for SEED in [0, 1, 2, 42]:` chạy lại toàn bộ Bước 1 và in kết quả ra, thay vì copy-paste code 4 lần.

### Bước 3: Huấn luyện trên toàn bộ rồi dự đoán tập công khai

#### 📘 Nội dung gốc

Khi đã chốt mô hình, huấn luyện lại trên toàn bộ dữ liệu huấn luyện, vì nhiều dữ liệu hơn thì mô hình tốt hơn.

```python
mo_hinh.fit(X, y)   # all of it this time

de = pd.read_csv("du-lieu/public_test.csv", keep_default_na=False)
doan = mo_hinh.predict(de["van_ban_goc"].tolist())
```

Trên tập công khai, mô hình này đạt macro-F1 bằng 0,7324. So với 0,6888 đo được ở tập kiểm định của chính mình, con số ngoài đời cao hơn, vì lần này mô hình được học từ nhiều dữ liệu hơn hai mươi phần trăm.

#### 🔬 Giải thích sâu

**Vì sao lại "học lại từ đầu trên toàn bộ dữ liệu" thay vì dùng luôn mô hình đã học ở Bước 1?**

Ở Bước 1, ta cố tình *giữ lại* 20% dữ liệu (tập kiểm định) mà không cho mô hình học, chỉ để dùng nó **đo lường** xem mô hình tốt tới đâu. Một khi đã hài lòng với kiến trúc mô hình và các siêu tham số (C=4.0, ngram_range=(2,5)...), việc tiếp tục "giấu" 20% dữ liệu đó khỏi mô hình cuối cùng là lãng phí — càng nhiều dữ liệu học, mô hình càng có cơ hội học được quy luật tổng quát tốt hơn. Vì vậy bước cuối luôn là "gộp lại 100% dữ liệu training đã có, train lần cuối, rồi mới dùng để dự đoán trên dữ liệu thật sự chưa biết đáp án (tập test/public test)".

**Vì sao 0,7324 > 0,6888 dù cùng một mô hình/kiến trúc?**

Đây không phải là ngẫu nhiên may mắn — nó phản ánh một quy luật khá vững trong ML: **nhiều dữ liệu huấn luyện hơn thường dẫn đến mô hình tổng quát hoá tốt hơn** (learning curve đi lên khi số mẫu tăng). Ở Bước 1, mô hình chỉ được học từ 80% dữ liệu (4800 dòng); ở Bước 3, nó được học từ 100% (6000 dòng) — tăng 20% lượng dữ liệu học. Tuy nhiên, cần lưu ý: **không nên vội kết luận** rằng con số 0,7324 chắc chắn "tốt hơn thật" 0,6888 — một phần của khoảng cách này vẫn có thể là do bộ dữ liệu `public_test.csv` tình cờ "dễ" hơn tập kiểm định riêng của bạn (giống hệt bài học về nhiễu ở mục 2.2).

---

## 7. Bài tập mẫu — main.py hoàn chỉnh

### 📘 Đề bài (nội dung gốc)

Viết `main.py` hoàn chỉnh cho nhiệm vụ phân loại của tác vụ 1, chạy hai lần phải ra hai file giống hệt nhau.

### 📘 Lời giải (nội dung gốc)

```python
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

HERE = os.path.dirname(os.path.abspath(__file__))


def duong(*phan):
    # Paths are resolved against this file, never against the current directory, because
    # the grader may run the script from somewhere else entirely.
    return os.path.join(HERE, *phan)


def main():
    train = pd.read_csv(duong("du-lieu", "training_set.csv"), keep_default_na=False)
    de = pd.read_csv(duong("du-lieu", "public_test.csv"), keep_default_na=False)

    mo_hinh = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
        LogisticRegression(max_iter=2000, C=4.0, random_state=SEED))
    mo_hinh.fit(train["van_ban_goc"].tolist(), train["nhan"].tolist())
    doan = mo_hinh.predict(de["van_ban_goc"].tolist())

    with io.open(duong("submission.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["van_ban_goc", "nhan", "ban_dich"])
        # Same order as the question file. Never sort, never filter.
        for cau, nhan in zip(de["van_ban_goc"], doan):
            w.writerow([cau, int(nhan), ""])


if __name__ == "__main__":
    main()
```

**Nghiệm thu:**
```
python main.py && md5sum submission.csv
python main.py && md5sum submission.csv
```
Hai md5 phải giống hệt nhau.

**Ba chi tiết nhỏ trong đoạn code trên, mỗi cái từng làm hỏng một bài thi:**

1. `HERE` và hàm `duong`: người chấm chạy script từ thư mục khác, và đường dẫn tương đối theo thư mục hiện hành sẽ không tìm thấy dữ liệu.
2. `newline=""` trong `io.open`: thiếu nó, trên Windows mỗi dòng CSV kết thúc bằng hai ký tự xuống dòng, và file có thể bị đọc sai.
3. `int(nhan)`: `predict` trả về kiểu `numpy.int64`, và ghi thẳng ra CSV thì tuỳ phiên bản có thể ra "1" hoặc một thứ khác.

### 🧠 Ẩn dụ đời thường cho ba chi tiết nhỏ

1. **`HERE`/`duong()`** giống như việc ghi địa chỉ giao hàng là "nhà kế bên trạm xăng Con Cò" thay vì "nhà kế bên trạm xăng" — địa chỉ *tuyệt đối* (gắn với một mốc cố định, rõ ràng) luôn đáng tin hơn địa chỉ *tương đối* (phụ thuộc vào việc người đọc đang đứng ở đâu).
2. **`newline=""`** giống như việc gửi một bức thư viết tay: người Việt Nam xuống dòng bằng 1 cú "Enter", nhưng ở một số "phong tục" khác (Windows) xuống dòng lại cần 2 ký tự liên tiếp (`\r\n`). Nếu không thống nhất trước, người nhận có thể đọc thư bị lệch dòng.
3. **`int(nhan)`** giống như việc đổi tiền trước khi xuất hoá đơn: `numpy.int64` là "loại tiền tệ nội bộ" của NumPy/sklearn, cần đổi sang `int` thuần Python (loại tiền tệ phổ thông mà mọi phần mềm đọc CSV đều hiểu chắc chắn) trước khi "xuất hoá đơn" ra file.

### 🔬 Giải thích sâu từng chi tiết

**1. Vì sao `os.path.dirname(os.path.abspath(__file__))` giải quyết được vấn đề?**

`__file__` là biến đặc biệt của Python, luôn chứa đường dẫn tới chính file code đang chạy. `os.path.abspath(__file__)` biến nó thành đường dẫn tuyệt đối (đầy đủ từ gốc ổ đĩa), phòng trường hợp `__file__` chỉ chứa đường dẫn tương đối. `os.path.dirname(...)` lấy ra thư mục chứa file đó (bỏ tên file). Kết quả `HERE` luôn là "thư mục chứa main.py", bất kể bạn gọi `python main.py` từ đâu. Hàm `duong(*phan)` sau đó dùng `os.path.join(HERE, *phan)` để ghép `HERE` với các phần đường dẫn con (ví dụ `"du-lieu"`, `"training_set.csv"`), đảm bảo luôn trỏ đúng file dù chạy script từ bất kỳ thư mục làm việc nào.

**2. Vì sao thiếu `newline=""` gây lỗi trên Windows?**

Mặc định, khi Python mở file ở chế độ text (`"w"`), nó tự động "dịch" mọi ký tự xuống dòng `\n` mà bạn viết trong code thành ký tự xuống dòng đặc trưng của hệ điều hành đang chạy — trên Windows là `\r\n`. Đồng thời, module `csv` của Python *tự nó* cũng ghi `\r\n` sau mỗi dòng theo chuẩn CSV (RFC 4180). Nếu bạn không tắt việc "dịch" tự động của `open()` bằng `newline=""`, hai lớp xử lý xuống dòng này chồng lên nhau, biến mỗi dòng thành kết thúc bằng `\r\r\n` (dư một `\r`) — khiến một số chương trình đọc CSV khác (như Excel, hoặc bộ chấm tự động) đọc sai định dạng, đôi khi tạo ra các dòng trống xen kẽ.

**3. Vì sao `predict()` trả về `numpy.int64` chứ không phải `int` thường?**

Nội bộ scikit-learn và NumPy dùng các kiểu số có kích thước cố định (như `int64`, `float64`) để tính toán nhanh hơn nhiều so với kiểu `int` linh hoạt của Python thuần. Khi bạn ghi trực tiếp một giá trị `numpy.int64` vào CSV bằng module `csv`, hàm `str()` ngầm được gọi lên nó — với hầu hết phiên bản, kết quả là chuỗi số bình thường ("1"), nhưng ở một số phiên bản NumPy/pandas cũ hoặc cấu hình đặc biệt, cách biểu diễn có thể khác đi (ví dụ kèm theo kiểu dữ liệu, hoặc định dạng số khoa học với giá trị lớn). Gọi `int(nhan)` để ép về kiểu `int` thuần Python trước khi ghi giúp loại bỏ hoàn toàn sự không chắc chắn này — đảm bảo kết quả luôn là "1", "0" rõ ràng, nhất quán giữa các phiên bản thư viện, phục vụ đúng yêu cầu về md5 ổn định.

---

## 8. Bài tập tự làm (1–5)

### 📘 Bài tập 1 — Hai nguồn ngẫu nhiên, chỉ một cái thật sự nguy hiểm

Làm hai thí nghiệm, mỗi cái chạy hai lần và so kết quả.

- **Thí nghiệm A.** Xoá `random_state=SEED` khỏi `LogisticRegression` trong `main.py`, chạy hai lần, so md5.
- **Thí nghiệm B.** Trong đoạn code ở Bước 1, xoá `random_state=SEED` khỏi `train_test_split`, chạy bốn lần, ghi lại bốn điểm kiểm định.

**Cổng kiểm:** Thí nghiệm A: hai md5 giống hệt nhau. Thí nghiệm B: bốn điểm khác nhau.

Rồi trả lời: vì sao bỏ seed ở chỗ này thì vô hại còn ở chỗ kia thì không? Câu trả lời nằm ở chỗ bộ giải `lbfgs` của hồi quy logistic là **tất định** (deterministic), nó không dùng số ngẫu nhiên nào cả, nên tham số `random_state` ở đó gần như chỉ để trang trí.

> Bài học không phải là "chỗ nào cần seed". Bài học là bạn không biết chỗ nào cần cho tới khi thử, nên cách an toàn là cố định hết, rồi kiểm bằng md5.

#### 🔬 Gợi ý đào sâu (không thay thế việc tự làm)

- **`lbfgs`** (Limited-memory BFGS) là một thuật toán tối ưu số học **thuần tuý toán học**: nó tính đạo hàm (gradient) của hàm mất mát và di chuyển theo hướng đó một cách hoàn toàn có công thức, không "tung xúc xắc" ở bất kỳ bước nào. Đây là solver mặc định của `LogisticRegression` trong scikit-learn khi dữ liệu không quá lớn. Vì vậy, dù bạn có xoá `random_state` khỏi `LogisticRegression`, kết quả tối ưu vẫn y hệt giữa các lần chạy — tham số đó "ngủ yên", không được dùng tới.
- Ngược lại, `train_test_split` **luôn** phải xáo trộn dữ liệu bằng số ngẫu nhiên để chọn ra 20% làm validation — đây là bản chất công việc của nó, không thể tránh dùng ngẫu nhiên được. Nếu không cố định `random_state`, mỗi lần chạy sẽ chọn ra một tập validation khác, kéo theo mọi thứ tính từ đó (điểm số, thậm chí cả `submission.csv` nếu tập train/test ảnh hưởng tới nó) đều thay đổi.
- Đây là lý do triết lý "cố định hết cho chắc, đừng đoán" (mục 2.4) lại quan trọng: người mới có xu hướng đoán "chắc solver này cũng ngẫu nhiên như solver kia", nhưng thực tế mỗi thuật toán có bản chất khác nhau — có loại tất định (lbfgs, hầu hết cây quyết định khi không lấy mẫu ngẫu nhiên), có loại ngẫu nhiên thật sự (SGD, khởi tạo trọng số neural network, Random Forest). Cách duy nhất để biết chắc là **thử và đo**, không phải suy luận từ tên hàm.

### 📘 Bài tập 2 — Bảng bốn seed

Dựng lại bảng ở mục 2.2 bằng code của mình: bốn seed, hai cột điểm, một cột chênh.

**Cổng kiểm:** Bốn dòng khớp bảng trong tài liệu tới chữ số thập phân thứ tư. Lệch thì tìm ra chỗ mình làm khác, đừng sửa bảng.

#### 🔬 Gợi ý đào sâu

Nếu số của bạn không khớp đúng tới 4 chữ số thập phân, khả năng cao là một trong các nguồn ngẫu nhiên ở mục 4 chưa được khoá đủ (ví dụ quên `stratify=y`, quên đúng thứ tự `test_size=0.2`, hoặc dùng phiên bản scikit-learn khác khiến thuật toán mặc định hơi khác). Đây là cách rèn luyện thói quen **debug bằng cách so khớp con số chính xác tuyệt đối**, thay vì chỉ "thấy gần gần là được" — thói quen này cực kỳ quan trọng khi đi thi, vì cổng kiểm md5 không chấp nhận "gần đúng".

### 📘 Bài tập 3 — Đo thời gian và tự đặt ngân sách

Thêm đo thời gian vào `main.py`. Nếu nó chạy t giây trên 6000 dòng, hãy ước lượng: với 100 nghìn dòng thì mất bao lâu, và có còn lọt giới hạn 20 phút không?

**Cổng kiểm:** Một con số ước lượng kèm giả định bạn dùng để suy ra nó. Ví dụ "thời gian tăng gần tuyến tính theo số dòng, nên 100 nghìn dòng mất khoảng x giây".

#### 🔬 Gợi ý đào sâu

- Cách đo thời gian đơn giản nhất trong Python: dùng module `time`.
```python
import time
bat_dau = time.time()
# ... đoạn code cần đo ...
print("Mất %.2f giây" % (time.time() - bat_dau))
```
- **Vì sao giả định "tăng gần tuyến tính" chỉ là gần đúng, không chính xác tuyệt đối?** Vì các bước trong pipeline có độ phức tạp tính toán khác nhau theo số dòng dữ liệu (n): việc đọc CSV và huấn luyện `TfidfVectorizer`/`LogisticRegression` với solver `lbfgs` thường có chi phí xấp xỉ tuyến tính đến gần-tuyến tính (O(n) đến O(n log n)) với số mẫu khi số đặc trưng cố định, nhưng nếu số đặc trưng (kích thước từ vựng) cũng tăng theo số dòng (dữ liệu càng nhiều, càng có nhiều cụm ký tự hiếm mới xuất hiện), tổng thời gian có thể tăng nhanh hơn tuyến tính một chút. Vì vậy, một ước lượng "tăng tuyến tính" là điểm khởi đầu hợp lý và đơn giản, nhưng nên nói rõ đó là giả định (như cổng kiểm yêu cầu), không phải sự thật tuyệt đối — đây chính là kỹ năng **ước lượng có ý thức về giả định** rất cần trong các bài toán kỹ thuật thực tế.

### 📘 Bài tập 4 — Đường dẫn tương đối

Chạy `main.py` từ một thư mục khác:
```
cd ~ && python /duong/dan/day/du/main.py
```

**Cổng kiểm:** Chạy được, và `submission.csv` nằm cạnh `main.py` chứ không nằm ở thư mục bạn đang đứng. Rồi thử bỏ hàm `duong` đi và xem nó hỏng thế nào.

#### 🔬 Gợi ý đào sâu

Đây là bài tập **thực chứng** cho lý thuyết ở mục 5 và mục 7.1. Khi bạn cố tình bỏ hàm `duong()` và thay bằng đường dẫn tương đối trần (`"du-lieu/training_set.csv"`), rồi chạy từ thư mục `~`, bạn sẽ thấy tận mắt lỗi `FileNotFoundError: [Errno 2] No such file or directory: 'du-lieu/training_set.csv'` — vì Python đang tìm thư mục `du-lieu` bên trong `~`, nơi nó không hề tồn tại. Việc tự tay gây ra lỗi này một lần sẽ giúp bạn nhớ mãi, hiệu quả hơn nhiều so với chỉ đọc lý thuyết.

### 📘 Bài tập 5 — Thêm nhiệm vụ dịch vào

Mở rộng `main.py` để nó điền cả cột `ban_dich`, dù chỉ bằng cách chép nguyên câu nguồn.

**Cổng kiểm:** Bộ chấm nhận bài, in ra cả hai chỉ số, và md5 vẫn ổn định qua hai lần chạy. Bạn vừa có bộ khung để buổi 22 lắp mô hình dịch thật vào.

#### 🔬 Gợi ý đào sâu

Đây là bài tập rèn thói quen **"làm khung trước, nhồi trí tuệ sau"** (scaffold-first) — một tư duy kỹ sư quan trọng: khi bài toán cuối cùng (dịch máy thật) còn quá phức tạp/xa vời (để dành tới buổi 22), bạn vẫn có thể xây sẵn "bộ khung" hoàn chỉnh — đọc dữ liệu, chạy pipeline, ghi đúng định dạng cột, đảm bảo tái lập được — với phần "trí tuệ" tạm thời chỉ là một hàm giả đơn giản nhất có thể (copy nguyên văn). Về sau, thay bằng mô hình dịch thật chỉ là đổi **một hàm nhỏ**, không phải viết lại toàn bộ hệ thống nộp bài. Gợi ý code tối giản:
```python
ban_dich = [cau for cau in de["van_ban_goc"]]   # tạm thời: dịch = chép nguyên câu
```

---

## 9. Chuẩn bị cho buổi sau

### 📘 Nội dung gốc

Hết khối nhập môn. Buổi 04 bắt đầu học máy cổ điển với hồi quy tuyến tính, và bạn sẽ tự cài giảm độ dốc bằng NumPy chứ không gọi thư viện. Mang theo câu hỏi: làm sao biết một thuật toán tối ưu đã hội tụ, hay chỉ đang bò chậm?

### 🧠 Ẩn dụ đời thường cho câu hỏi mở

Hãy tưởng tượng bạn đang đi bộ xuống một thung lũng trong sương mù dày đặc (không nhìn thấy đáy thung lũng, chỉ cảm nhận được độ dốc dưới chân) — đó chính là **giảm độ dốc (gradient descent)**: mỗi bước bạn nhìn xem hướng nào dốc xuống nhiều nhất rồi bước theo hướng đó. Câu hỏi "làm sao biết đã tới đáy hay chỉ đang bò chậm" giống như hỏi: "Làm sao biết mình đã chạm đáy thung lũng, hay chỉ đang đi trên một đoạn đường gần như bằng phẳng nhưng vẫn còn dốc xuống rất rất nhẹ ở phía xa?"

### 🔬 Gợi ý tư duy trước (không phải đáp án — để dành cho buổi 04)

Một vài tín hiệu thường được dùng để nhận biết "hội tụ" trong thực tế, để bạn suy nghĩ trước:
- Theo dõi giá trị hàm mất mát (loss) qua từng vòng lặp — nếu nó gần như không giảm thêm nữa (thay đổi nhỏ hơn một ngưỡng rất nhỏ), có thể coi là đã hội tụ.
- Theo dõi độ lớn của gradient (đạo hàm) — nếu gradient gần bằng 0, nghĩa là đang đứng ở một điểm gần như "phẳng", có thể là đáy.
- Đặt số vòng lặp tối đa (như `max_iter=2000` đã thấy ở `LogisticRegression`) như một "lưới an toàn", và luôn kiểm tra xem thuật toán dừng vì đã hội tụ, hay vì chạm giới hạn số vòng lặp (hai tình huống rất khác nhau!).

Hãy giữ câu hỏi này trong đầu — buổi 04 sẽ trả lời bằng cách chính bạn tự viết vòng lặp giảm độ dốc bằng NumPy và tự mắt quan sát nó hội tụ (hoặc không hội tụ) như thế nào.

---

## 10. Bảng tổng kết thuật ngữ nhanh

| Thuật ngữ | Giải thích ngắn gọn |
|---|---|
| **X (đặc trưng/feature)** | Ma trận mô tả từng mẫu dữ liệu, shape `(số mẫu, số đặc trưng)` |
| **y (nhãn/label)** | Vector đáp án đúng cho từng mẫu, shape `(số mẫu,)` |
| **TfidfVectorizer** | Bộ chuyển văn bản thành vector số, dựa trên tần suất xuất hiện có trọng số |
| **train / validation / test** | Tập học / tập chọn mô hình-siêu tham số / tập đánh giá cuối cùng chỉ dùng một lần |
| **overfitting (học vẹt)** | Mô hình khớp quá sát dữ liệu đã học, kém khi gặp dữ liệu mới |
| **stratify** | Giữ nguyên tỉ lệ các lớp khi chia tập dữ liệu |
| **macro-F1** | Trung bình cộng đơn giản của F1 từng lớp, đối xử công bằng giữa các lớp |
| **precision / recall** | Độ chính xác của dự đoán dương / độ bao phủ các mẫu dương thật |
| **random_state / seed** | Con số cố định "khoá" một nguồn sinh số ngẫu nhiên để tái lập kết quả |
| **PYTHONHASHSEED** | Biến môi trường kiểm soát cách Python băm chuỗi, ảnh hưởng thứ tự duyệt set/dict |
| **lbfgs** | Thuật toán tối ưu tất định (không ngẫu nhiên) dùng trong LogisticRegression |
| **C (trong LogisticRegression)** | Nghịch đảo độ mạnh chính quy hoá — C nhỏ = mô hình đơn giản hơn |
| **md5** | Hàm băm tạo "vân tay" của file, dùng để kiểm tra hai file có giống hệt nhau không |
| **data leakage (rò rỉ dữ liệu)** | Vô tình để mô hình "nhìn thấy" thông tin từ tập test trước khi đánh giá |
| **generalization gap** | Khoảng chênh giữa điểm trên tập train và tập validation/test |
| **noise floor (ngưỡng nhiễu)** | Mức dao động tự nhiên của điểm số do các yếu tố ngẫu nhiên (như đổi seed) |

---

*Tài liệu được biên soạn dựa trên bài giảng gốc "Buổi 03 — Bài toán học máy đầu tiên, từ đầu tới cuối" của TS. Đỗ Phúc Hảo, chương trình Luyện Olympic Trí tuệ nhân tạo, ngày 25/8/2026. Mọi nội dung gốc được giữ nguyên vẹn; phần mở rộng (ẩn dụ, giải thích sâu, code chú thích) được bổ sung nhằm mục đích học tập.*
