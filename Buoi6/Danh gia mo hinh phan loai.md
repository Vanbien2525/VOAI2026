# Buổi 06 — Đánh giá mô hình phân loại

**Luyện Olympic Trí tuệ nhân tạo** · Phần: Học máy cổ điển · Phục vụ: cả hai vòng
Ngày 25 tháng 8 năm 2026 · TS. Đỗ Phúc Hảo
*Tài liệu mở rộng — không cắt bỏ bất kỳ nội dung gốc nào.*

---

## Cách đọc tài liệu này

Tài liệu được trình bày **song song**, không tách rời:

- Mỗi đoạn **nội dung gốc** của thầy Hảo (câu chữ, công thức, bảng số, code, các khung "vì sao" / "cổng kiểm") được giữ nguyên 100%, đặt trong khung trích dẫn (`>`) hoặc giữ format gốc.
- Ngay bên dưới những chỗ trừu tượng hoặc dễ gây hiểu lầm, có thêm khung **🧠 Giải thích thêm** bằng ví dụ đời thường — đọc liền mạch, không cần lật qua phần khác.
- Ở cuối tài liệu là **Phụ lục — Đào sâu kỹ thuật (🔬)**: nơi từng dòng code được bóc tách, và mọi con số trong bài đã được chạy lại thật bằng scikit-learn 1.8.0 để đối chiếu (không suy diễn). Phần này dành cho lúc bạn muốn tự tay kiểm chứng, không bắt buộc đọc ngay lần đầu.

---

## 1. Mục tiêu

> Tự cài `macro_f1` khớp scikit-learn tới sai số máy, và giải thích được trường hợp hai bên cho ra hai con số khác nhau.
>
> **Vì sao buổi này đứng ngay sau hai mô hình đầu tiên**
> Vì từ buổi sau trở đi, mọi câu bạn nói về mô hình của mình đều là một con số. Một con số sai trông y hệt một con số đúng. Người không tự đo được mình thì mọi cải tiến về sau đều là đoán mò, và trong phòng thi thì đoán mò tốn đúng bằng số lượt nộp bạn có.

## 2. Tóm tắt kiến thức

### 2.1 Bốn ô của ma trận nhầm lẫn

> | | đoán 0 | đoán 1 |
> |---|---|---|
> | **thật là 0** | TN | FP |
> | **thật là 1** | FN | TP |
>
> Nhớ bằng cách đọc chữ thứ hai trước: P hay N là cái mô hình đoán, còn T hay F nói mô hình đoán đúng hay sai. Vậy FP là "mô hình đoán dương, và nó sai".

**🧠 Giải thích thêm:** Hình dung bạn là nhân viên soát vé ở cổng an ninh sân bay, và "dương" (1) nghĩa là "có vật nguy hiểm trong túi":

- **TP** — thật sự có dao, máy báo động, bạn giữ người lại. Bắt đúng.
- **TN** — túi sạch, máy im lặng, cho qua. Đúng theo kiểu không có chuyện gì xảy ra.
- **FP** — túi sạch nhưng máy vẫn kêu. Bạn làm phiền một hành khách vô tội (báo động giả).
- **FN** — có dao thật nhưng máy im re. Đây là ô nguy hiểm nhất, vì hậu quả xảy ra ở đời thực chứ không chỉ trên giấy.

Mẹo "đọc chữ thứ hai trước" chính là: **nhìn P/N để biết máy nói gì, nhìn T/F để biết máy nói đúng hay sai** — tách bạch hai câu hỏi hoàn toàn khác nhau, bốn ô là bốn giao điểm của hai trục ấy.

### 2.2 Ba con số từ bốn ô ấy

> ```
> Precision = TP / (TP + FP)
> Recall    = TP / (TP + FN)
> F1        = 2 · P · R / (P + R)
> ```
>
> Precision — trong những cái mô hình gọi là dương, bao nhiêu phần đúng
> Recall — trong những cái thật sự dương, mô hình bắt được bao nhiêu phần
> F1 — trung bình điều hoà của hai cái trên, nên nó tụt xuống ngay khi một trong hai tụt

**🧠 Giải thích thêm:**

- **Precision cao** giống một bác sĩ chỉ chẩn đoán ung thư khi *cực kỳ* chắc chắn — ít khi chẩn đoán sai (ít FP), nhưng có thể bỏ sót ca thật (nhiều FN) vì quá thận trọng.
- **Recall cao** giống một máy dò kim loại vặn độ nhạy lên tối đa — bắt được gần hết vật kim loại thật (ít FN), nhưng kêu bậy rất nhiều (nhiều FP).

Hai thước đo này thường **kéo co lẫn nhau**: tăng cái này thường làm cái kia giảm, vì bạn đang dịch chuyển ranh giới "gọi là dương" / "gọi là âm". F1 buộc bạn cân bằng cả hai, không cho "ăn gian" bằng cách chỉ tối ưu một phía.

*Vì sao F1 dùng trung bình điều hoà chứ không phải trung bình cộng?* Trung bình cộng của P = 1,0 và R = 0,01 vẫn ra 0,505 — nghe có vẻ ổn dù mô hình gần như vô dụng (chỉ bắt được 1% ca dương thật). Trung bình điều hoà bị "kéo tụt" mạnh bởi số nhỏ nhất, giống công thức tính vận tốc trung bình khi đi cùng quãng đường với hai vận tốc khác nhau: đoạn đi chậm chiếm nhiều thời gian hơn, kéo vận tốc trung bình toàn trình xuống gần vận tốc chậm. F1 = 2·1·0,01/(1+0,01) ≈ 0,0198 — phản ánh đúng thực tế hơn nhiều.

### 2.3 Macro, micro, weighted: ba cách gộp hai lớp

> - **macro** tính F1 riêng cho từng lớp rồi lấy trung bình không trọng số
> - **micro** gộp TP, FP, FN của mọi lớp rồi mới tính. Với bài nhị phân thì nó bằng accuracy
> - **weighted** trung bình có trọng số theo số mẫu của từng lớp
>
> Đề thi dùng macro. Chọn nhầm cách gộp là đo một thứ khác hẳn, và mục sau cho thấy khác tới mức nào.

**🧠 Giải thích thêm:** Tưởng tượng một lớp học có 2 tổ: tổ A có 5 học sinh giỏi (gần như luôn đúng), tổ B chỉ có 1 học sinh yếu (hay sai).

- **macro** = điểm trung bình tổ A và điểm trung bình tổ B cộng lại chia đôi — **mỗi tổ có tiếng nói ngang nhau**, dù tổ B chỉ có 1 người. Công bằng giữa các *lớp*, không phải giữa các *mẫu*.
- **micro** = gom hết bài cả lớp tính một điểm chung — hỏi "cả lớp đúng bao nhiêu phần trăm bài", và vì mỗi học sinh là một "phiếu bầu" nên tổ đông người áp đảo kết quả. Với bài nhị phân, đây chính là accuracy.
- **weighted** = trung bình có tính sĩ số mỗi tổ — nửa đường giữa hai cách trên.

Đề thi chọn macro vì nó **không cho phép trốn tránh lớp thiểu số**: mô hình chỉ giỏi lớp đông mà bỏ bê lớp hiếm sẽ bị lộ ngay, vì tổ B vẫn được tính nửa số điểm dù chỉ có ít mẫu.

### 2.4 Vì sao accuracy vô dụng khi lệch lớp

> Lấy một bộ dữ liệu 1000 mẫu, trong đó 950 thuộc lớp 1. Mô hình "đoán 1 cho tất cả" không học gì cả, không nhìn dữ liệu, chỉ trả về hằng số. Đo thật:
>
> | Thước đo | Giá trị |
> |---|---|
> | accuracy | 0,9500 |
> | F1 micro | 0,9500 |
> | F1 weighted | 0,9256 |
> | F1 macro | 0,4872 |
>
> **Ba trong bốn thước đo đều khen một mô hình rỗng**
> Cùng một bài nộp, cùng một dữ liệu, mà con số chạy từ 0,49 tới 0,95 tuỳ cách gộp. Nếu bạn báo cáo accuracy 95% thì bạn không nói dối, nhưng bạn đang nói một điều vô nghĩa: mô hình chưa từng đoán đúng lấy một mẫu của lớp thiểu số.
>
> Macro-F1 là thước duy nhất nhìn ra, vì nó bắt mô hình phải làm được việc trên cả hai lớp, bất kể lớp nào hiếm.

**🧠 Giải thích thêm:** Đây là ví dụ kinh điển về "mô hình lười" — giống một học sinh làm bài trắc nghiệm Đúng/Sai mà cứ khoanh "Đúng" cho mọi câu, và tình cờ 95% câu hỏi đáp án thật là "Đúng". Học sinh này *trông* như học giỏi (95 điểm/100) nhưng thực ra **không phân biệt được gì cả** — cho học sinh này một bài thi tỉ lệ 50/50, điểm sẽ rơi thẳng xuống 50.

Accuracy đo "đúng bao nhiêu phần trăm" nhưng không hỏi "đúng ở *đâu*". Khi một lớp áp đảo, "đoán bừa theo lớp áp đảo" và "học thật" cho ra accuracy gần như nhau — accuracy không phân biệt được hai loại năng lực đó. Macro-F1 thì có, vì nó buộc mô hình chứng minh năng lực ở *cả hai* lớp riêng biệt, không được "mượn điểm" từ lớp dễ để che lớp khó.

### 2.5 Cái bẫy `labels`

> ```python
> from sklearn.metrics import f1_score
>
> mot_lop = [1] * 6  # both truth and prediction are all ones
> f1_score(mot_lop, mot_lop, average="macro", zero_division=0)                  # 1.00
> f1_score(mot_lop, mot_lop, average="macro", labels=[0, 1], zero_division=0)   # 0.50
> ```
>
> Khi một lớp vắng mặt trong cả nhãn thật lẫn nhãn đoán, scikit-learn lặng lẽ trung bình trên một lớp thay vì hai, và con số nhảy từ 0,50 lên 1,00. Không có cảnh báo nào.
>
> **Ở chỗ này, thư viện mới là bên lệch đề bài**
> Đề định nghĩa macro-F1 là trung bình của đúng hai lớp. Nên luôn truyền `labels=[0, 1]`. Đây là một dòng gõ thêm, và nó là khác biệt giữa một con số đúng và một con số đẹp.

**🧠 Giải thích thêm:** Hãy nghĩ về việc tính điểm trung bình môn học của một học sinh chỉ đăng ký 1 môn (Toán) trong khi bảng điểm chuẩn có 2 môn (Toán, Văn). Nếu phần mềm tính điểm "chỉ nhìn thấy môn nào học sinh có đăng ký" thì nó sẽ tính trung bình trên **1 môn** — và nếu học sinh được 10 điểm Toán, điểm trung bình hiển thị là 10.0, dù đáng lẽ phải là trung bình của Toán và Văn (mà Văn thì học sinh chưa từng đụng tới, coi như 0 điểm) = 5.0.

Đó chính xác là chuyện xảy ra với `mot_lop = [1] * 6`: không có mẫu nào mang nhãn 0, ở cả nhãn thật lẫn nhãn đoán. Mặc định, scikit-learn chỉ "đăng ký" những lớp *thực sự xuất hiện* — nó không biết đề bài kỳ vọng luôn có đúng 2 lớp. `labels=[0, 1]` là cách bạn nói thẳng với thư viện: "tôi luôn có 2 môn, dù học sinh có đăng ký hay không."

*(Muốn xem từng bước tính tay ra đúng 1,0 và 0,5 như thế nào, xem mục 🔬 ở Phụ lục.)*

### 2.6 ROC và AUC

> Mô hình cho ra xác suất, còn nhãn cần một ngưỡng. Đường cong ROC vẽ tỉ lệ dương thật theo tỉ lệ dương giả khi ngưỡng chạy từ 1 về 0. AUC là diện tích dưới đường ấy.
>
> AUC có một cách đọc rất gọn: lấy ngẫu nhiên một mẫu dương và một mẫu âm, xác suất mô hình chấm mẫu dương điểm cao hơn. AUC = 0,5 nghĩa là đoán mò.
>
> **AUC không thay được macro-F1**
> AUC đo thứ tự mà mô hình xếp các mẫu, không phụ thuộc ngưỡng. Macro-F1 đo chất lượng sau khi đã chọn ngưỡng. Một mô hình có AUC tốt vẫn có thể có macro-F1 tệ nếu ngưỡng đặt sai chỗ. Đề chấm bằng macro-F1, nên AUC chỉ là công cụ chẩn đoán.

**🧠 Giải thích thêm:** Hãy tưởng tượng bạn có một chồng hồ sơ bệnh nhân, một nửa **thật sự có bệnh**, một nửa **thật sự khoẻ**. Bạn trộn ngẫu nhiên, rút ra 1 hồ sơ bệnh và 1 hồ sơ khoẻ, rồi hỏi mô hình: "ai trong hai người này bạn nghĩ có khả năng bị bệnh cao hơn?" AUC chính là **xác suất mô hình chọn đúng người bệnh**, lặp lại phép rút này với mọi cặp có thể.

Điều quan trọng: AUC **không quan tâm mô hình cho điểm bao nhiêu**, chỉ quan tâm **thứ tự xếp hạng**. Hai mô hình cho điểm khác hẳn nhau về giá trị vẫn có thể có cùng AUC nếu thứ tự xếp hạng giống nhau.

Đây cũng là lý do "AUC không thay được macro-F1": AUC giống hỏi "bác sĩ này *phân biệt được* người bệnh và người khoẻ giỏi tới đâu, nếu được tự do xếp hạng toàn bộ hồ sơ". Macro-F1 giống hỏi "khi bác sĩ *buộc phải* đưa ra chẩn đoán Có/Không ở một ngưỡng cụ thể, họ đúng bao nhiêu?" Một bác sĩ phân biệt giỏi (AUC cao) vẫn có thể chẩn đoán tệ nếu chọn sai điểm cắt — giống người xếp hạng thí sinh rất chuẩn nhưng lại đặt điểm đậu sai chỗ.

## 3. Hướng dẫn

### 3.1 Bước 1: tự cài macro-F1

> ```python
> def macro_f1(that, doan):
>     """Unweighted mean of the per class F1, over exactly the two classes 0 and 1."""
>     tong = 0.0
>     for lop in (0, 1):
>         tp = sum(1 for t, d in zip(that, doan) if t == lop and d == lop)
>         fp = sum(1 for t, d in zip(that, doan) if t != lop and d == lop)
>         fn = sum(1 for t, d in zip(that, doan) if t == lop and d != lop)
>         if tp == 0:
>             tong += 0.0  # matches scikit-learn with zero_division=0
>         else:
>             p = tp / (tp + fp)
>             r = tp / (tp + fn)
>             tong += 2 * p * r / (p + r)
>     return tong / 2
> ```

**🧠 Giải thích thêm:** Để ý dòng `for lop in (0, 1)` — vòng lặp *cố định cứng* trên đúng hai lớp, bất kể dữ liệu truyền vào có đủ cả hai lớp hay không. Đây chính là cách hàm tự tránh "cái bẫy `labels`" ở mục 2.5 ngay từ trong thiết kế: nó không hỏi "lớp nào có xuất hiện", nó luôn hỏi "lớp 0 và lớp 1 làm ăn ra sao" — giống việc luôn cộng điểm Toán và Văn dù học sinh không đăng ký môn nào. *(Bóc tách từng dòng còn lại, xem 🔬 Phụ lục 3.1.)*

### 3.2 Bước 2: đối chiếu với thư viện

> ```python
> import random
> from sklearn.metrics import f1_score
>
> random.seed(1)
> lech = 0.0
> for _ in range(300):
>     n = random.randint(2, 40)
>     a = [random.randint(0, 1) for _ in range(n)]
>     b = [random.randint(0, 1) for _ in range(n)]
>     lech = max(lech, abs(macro_f1(a, b)
>         - f1_score(a, b, average="macro", labels=[0, 1], zero_division=0)))
> print("lech lon nhat: %.2e" % lech)
> ```
>
> **Cổng kiểm**
> Độ lệch dưới 10⁻¹². Nếu bạn quên `labels=[0, 1]` thì độ lệch sẽ là 0,5, và đó không phải lỗi của bạn.

**🧠 Giải thích thêm:** Đây là kiểu kiểm thử "rải ngẫu nhiên" — thay vì tự tay chọn vài ví dụ, code tạo ra 300 cặp nhãn ngẫu nhiên với kích thước khác nhau (2 đến 40 mẫu) rồi so cả 300 lần. `lech` giữ lại sai lệch **tệ nhất từng thấy**, không phải trung bình — vì bạn cần chắc chắn *mọi* trường hợp đều đạt, không phải "phần lớn" trường hợp đạt. *(Con số thật sau khi chạy lại, xem 🔬 Phụ lục 3.2.)*

### 3.3 Bước 3: quét ngưỡng trên mô hình thật

> Lấy mô hình phân loại của buổi 03, chấm trên `public_test.csv`:
>
> | | |
> |---|---|
> | AUC | 0,7255 |
> | macro-F1 tại ngưỡng 0,50 | 0,7324 |
> | macro-F1 tốt nhất | 0,7409 tại ngưỡng 0,42 |
>
> ```python
> import numpy as np
>
> p = mo_hinh.predict_proba(X_test)[:, 1]
> tot, tau_tot = -1.0, None
> for tau in np.arange(0.05, 0.96, 0.01):
>     f = macro_f1(y_test, (p >= tau).astype(int))
>     if f > tot:
>         tot, tau_tot = f, tau
> print("tot nhat %.4f tai nguong %.2f" % (tot, tau_tot))
> ```
>
> **Con số 0,7409 kia là một cái bẫy, không phải một cải tiến**
> Nó được chọn bằng cách nhìn chính tập kiểm tra. Nếu bạn nộp bài với ngưỡng 0,42 rồi khoe rằng mình tăng được 0,0086, bạn đang khoe một con số học thuộc chứ không phải học được. Trên tập riêng, phần lớn khoảng tăng ấy sẽ bốc hơi.
>
> (Trừ hai ô trong bảng trên ra 0,0085. Con số đúng là 0,0086: bảng đã làm tròn tới bốn chữ số trước khi bạn trừ, còn phép trừ thật thì làm sau. Buổi 08 có một khung riêng về đúng cái bẫy này.)
>
> Cách đúng: chọn ngưỡng trên tập kiểm định cắt từ dữ liệu huấn luyện, rồi mới áp lên tập kiểm tra. Buổi 07 gọi tên hiện tượng này và dựng quy trình phòng nó.

**🧠 Giải thích thêm:** Hãy hình dung bạn đang luyện thi và có một bộ đề mẫu có đáp án. Nếu bạn thử tất cả 91 cách làm bài (mỗi ngưỡng từ 0,05 đến 0,95, bước 0,01) *trên chính bộ đề có đáp án đó*, rồi chọn cách cho điểm cao nhất, bạn không phải đang "học giỏi hơn" — bạn đang **chọn công thức vừa khít với đúng bộ đề đó**. Đưa bạn một bộ đề khác (tập riêng — private test) với đáp án khác, công thức "học thuộc" đó gần như chắc chắn không còn tối ưu nữa.

Đây là một dạng tinh vi của **rò rỉ dữ liệu** — không phải rò rỉ lúc huấn luyện mô hình, mà rò rỉ lúc *chọn siêu tham số* (ở đây là ngưỡng). Ngưỡng 0,42 không sai về toán học, nhưng nó **không còn độc lập với tập test** nữa — nó đã "nhìn trộm đáp án" của chính bài kiểm tra dùng để đánh giá mình. *(Vì sao 0,0085 hiển thị nhưng 0,0086 mới đúng — xem 🔬 Phụ lục 3.3.)*

## 4. Bài tập mẫu

> **Đề.** Tính bằng tay macro-F1 cho bài toán 20 câu sau, rồi kiểm bằng code.
>
> Tập kiểm tra có 20 câu, 10 nhãn 1 và 10 nhãn 0. Mô hình đoán 1 cho 14 câu, trong đó 9 câu đúng.
>
> **Bước 1: dựng ma trận nhầm lẫn.**
>
> Mô hình đoán 1 cho 14 câu, 9 câu trong đó thật sự là 1. Vậy TP = 9 và FP = 14 − 9 = 5. Có 10 câu nhãn 1, bắt được 9, nên FN = 1. Còn lại TN = 20 − 9 − 5 − 1 = 5.
>
> | | đoán 0 | đoán 1 |
> |---|---|---|
> | **thật 0** | 5 | 5 |
> | **thật 1** | 1 | 9 |
>
> **Bước 2: F1 của lớp 1.**
>
> ```
> P1 = 9 / (9+5) = 9/14 ≈ 0,6429
> R1 = 9 / (9+1) = 0,9
> F1_1 = 2 · 0,6429 · 0,9 / (0,6429 + 0,9) = 0,7500
> ```
>
> **Bước 3: F1 của lớp 0.** Đổi vai: bây giờ "dương" là lớp 0, nên TP0 = 5, FP0 = 1, FN0 = 5.
>
> ```
> P0 = 5/6 ≈ 0,8333
> R0 = 5/10 = 0,5
> F1_0 = 2 · 0,8333 · 0,5 / (0,8333 + 0,5) = 0,6250
> ```
>
> **Bước 4: gộp lại.**
>
> ```
> macro_F1 = (0,7500 + 0,6250) / 2 = 0,6875
> ```
>
> So với accuracy: (9 + 5)/20 = 0,7000.
>
> **Hai con số gần nhau, và chỉ một cái được chấm**
> 0,70 với 0,6875 chênh nhau chưa tới hai phần trăm ở ví dụ cân bằng này. Nhưng ở mục 2.4, cùng một kiểu mô hình, hai con số là 0,95 với 0,4872. Khoảng cách giữa accuracy và macro-F1 lớn dần theo mức lệch lớp, và bạn không biết trước tập kiểm tra riêng lệch tới đâu.

**🧠 Giải thích thêm:** Điểm hay của ví dụ này là bước 3 — "đổi vai": để tính F1 của lớp 0, bạn tạm coi lớp 0 là "dương" và lớp 1 là "âm", rồi áp lại đúng công thức Precision/Recall như bình thường. Đây là mẹo chung cho *mọi* bài toán nhị phân: macro-F1 luôn là "tính hai lần, mỗi lần đổi vai ai là dương, rồi lấy trung bình" — không có công thức nào khác cho lớp 0 cả, chỉ là công thức cũ áp lên vai trò mới. *(Kiểm chứng lại bằng code, xem 🔬 Phụ lục 3.5.)*

## 5. Bài tập tự làm

> **Bài tập 1. Cài và đối chiếu**
> Cài `macro_f1` của riêng bạn, đối chiếu 300 ca ngẫu nhiên với scikit-learn.
>
> **Cổng kiểm**
> Độ lệch dưới 10⁻¹². Rồi bỏ `labels=[0, 1]` đi và ghi lại độ lệch mới.

> **Bài tập 2. Tái hiện cái bẫy `labels`**
> Dựng một ví dụ nhỏ nhất có thể mà hai cách gọi cho hai kết quả khác nhau, rồi giải thích bằng lời tại sao.
>
> **Cổng kiểm**
> Hai con số 1,00 và 0,50, kèm một câu giải thích nói được cụm "lớp vắng mặt".

> **Bài tập 3. Bảng bốn thước đo trên dữ liệu lệch**
> Dựng lại bảng ở mục 2.4: 1000 mẫu, 950 thuộc lớp 1, mô hình đoán 1 cho tất cả.
>
> **Cổng kiểm**
> Bốn con số khớp 0,9500, 0,9500, 0,9256, 0,4872. Rồi thử đổi tỉ lệ thành 99/1 và xem macro-F1 rơi xuống bao nhiêu.

> **Bài tập 4. Vẽ đường cong ROC bằng tay**
> Không dùng `sklearn.metrics.roc_curve`. Sắp các mẫu theo xác suất giảm dần, rồi duyệt một lượt để dựng dãy điểm ROC. So AUC bạn tính với `roc_auc_score`.
>
> **Cổng kiểm**
> Độ lệch dưới 10⁻⁹, và AUC trên `public_test.csv` ra 0,7255.

> **Bài tập 5. Chọn ngưỡng cho đúng cách**
> Chia `training_set.csv` thành hai phần. Huấn luyện trên phần một, chọn ngưỡng tốt nhất trên phần hai, rồi áp ngưỡng ấy lên `public_test.csv`.
>
> **Cổng kiểm**
> Ba con số: macro-F1 tại ngưỡng 0,5, tại ngưỡng chọn đúng cách, và tại ngưỡng 0,42 lấy trộm từ tập kiểm tra. Rồi trả lời: cái nào là ước lượng trung thực cho điểm bạn sẽ nhận ở tập riêng?

**🧠 Giải thích thêm về bài 3 (câu hỏi 99/1):** đây là câu hỏi mở khá "bẫy" — trực giác thường nghĩ lệch lớp càng nặng thì macro-F1 càng tệ, nhưng thực tế nó chỉ tiến dần về một mức sàn cố định chứ không rơi vô hạn. *(Lời giải đầy đủ có chạy số thật, xem 🔬 Phụ lục 3.6.)*

### Chuẩn bị cho buổi sau

> Buổi 07 nói về quá khớp và kiểm định chéo, tức là dựng quy trình để cái bẫy ngưỡng ở trên không xảy ra nữa.
>
> Mang theo câu hỏi: tôi có hai mươi lượt nộp, làm sao dùng chúng mà không biến tập công khai thành tập huấn luyện thứ hai?

---

# Phụ lục — Đào sâu kỹ thuật 🔬

> Toàn bộ số liệu trong phụ lục này đã được chạy lại bằng scikit-learn 1.8.0 để đối chiếu — không phải suy diễn từ lý thuyết.

## 🔬 3.1 — Bóc tách `macro_f1` từng dòng

```python
def macro_f1(that, doan):
    """Unweighted mean of the per class F1, over exactly the two classes 0 and 1."""
    tong = 0.0
    for lop in (0, 1):                                                   # (1)
        tp = sum(1 for t, d in zip(that, doan) if t == lop and d == lop) # (2)
        fp = sum(1 for t, d in zip(that, doan) if t != lop and d == lop) # (3)
        fn = sum(1 for t, d in zip(that, doan) if t == lop and d != lop) # (4)
        if tp == 0:
            tong += 0.0   # matches scikit-learn with zero_division=0   # (5)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            tong += 2 * p * r / (p + r)                                 # (6)
    return tong / 2                                                     # (7)
```

1. **`for lop in (0, 1)`** — cố định cứng trên đúng hai lớp, không hỏi "lớp nào có xuất hiện". Đây là cách hàm tự tránh cái bẫy `labels` ngay từ thiết kế.
2. **`tp`** — đếm vị trí nhãn thật *và* nhãn đoán đều bằng `lop`. Đổi vai `lop` từ 1 sang 0 chính là kỹ thuật "coi lớp 0 là lớp dương" ở bài tập mẫu bước 3.
3. **`fp`** — nhãn thật *khác* `lop` nhưng đoán *là* `lop`: mô hình "hô" `lop` sai.
4. **`fn`** — nhãn thật *là* `lop` nhưng đoán *khác*: mô hình bỏ sót.
5. **`tp == 0`** — vùng nguy hiểm gây chia cho 0 nếu tính thẳng `tp/(tp+fp)`. Quy ước `zero_division=0` coi đây là đóng góp 0 vào tổng, thay vì crash hay trả `NaN` (nhắc lại bài học Buổi 05 về "constant columns produce `nan` silently").
6. Áp đúng công thức `2PR/(P+R)` sau khi đã có P, R an toàn.
7. **`return tong / 2`** — chia đúng cho 2 (số lớp), *không phải* chia cho số lớp "có xuất hiện".

**Vì sao đúng 1,0 và 0,5 trong cái bẫy `labels` — cơ chế từng bước:**

- Không truyền `labels`: scikit-learn tự dò lớp xuất hiện qua `sorted(set(y_true) | set(y_pred))`. Với `mot_lop`, tập này chỉ có `{1}` → chỉ tính F1 cho một lớp: TP=6, FP=0, FN=0 → P=R=1 → F1₁=1,0. Trung bình của **một số** là chính nó → 1,0.
- Truyền `labels=[0, 1]`: bị ép tính cả lớp 0, dù không hề xuất hiện. TP₀=FP₀=FN₀=0 → `0/0` → `zero_division=0` → F1₀=0,0. Trung bình của {1,0 ; 0,0} = **0,5**.

## 🔬 3.2 — Vòng lặp đối chiếu: đọc chỉ số sai lệch như thế nào

- `random.seed(1)` — cố định hạt giống để kết quả lặp lại y hệt giữa các lần chạy.
- `n = random.randint(2, 40)` — cỡ mẫu ngẫu nhiên mỗi vòng, để bài kiểm tra không chỉ đúng với một kích thước cố định.
- `a`, `b` — nhãn thật và nhãn đoán ngẫu nhiên độc lập — kiểu kiểm thử "brute-force ngẫu nhiên" (property-based testing), rải khắp không gian có thể thay vì chỉ vài case tay chọn.
- `lech = max(lech, ...)` — theo dõi sai lệch **tệ nhất**, không phải trung bình.

**Chạy lại thật, kết quả xác nhận:**

```
Lệch CÓ labels=[0,1]:    2.220446049250313e-16   (≈ sai số máy, dưới 10⁻¹²  ✔)
Lệch KHÔNG labels=[0,1]: 0.5                     (đúng như tài liệu cảnh báo)
```

`2.22e-16` chính là **epsilon máy** của số thực dấu phẩy động 64-bit — mức sai số nhỏ nhất giữa hai phép tính lẽ ra phải bằng nhau tuyệt đối. Bằng chứng cụ thể rằng `macro_f1` tự cài **khớp toán học hoàn toàn** với `f1_score`.

## 🔬 3.3 — Vì sao 0,7409 − 0,7324 hiển thị 0,0085 nhưng "đúng" là 0,0086

Minh hoạ đúng cơ chế (không phải số liệu ẩn thật của bài giảng):

```python
a_that, b_that = 0.73236, 0.74093
round(round(b_that, 4) - round(a_that, 4), 4)   # -> 0.0085   (lam tron ROI tru)
round(b_that - a_that, 4)                       # -> 0.0086   (tru ROI lam tron)
```

Chạy thật cho đúng hai kết quả trên. `0.73236` làm tròn thành `0.7324` (làm tròn *lên*), `0.74093` làm tròn thành `0.7409` (làm tròn *xuống*) — hai sai số làm tròn ngược dấu nhau cộng dồn khi trừ hai số **đã làm tròn**, thay vì triệt tiêu. Trừ trên số gốc rồi làm tròn một lần luôn đáng tin hơn.

**Bài học tổng quát:** báo cáo *hiệu số* (chênh lệch giữa hai lần đo, hai mô hình, hai epoch) nên tính trên số có độ chính xác đầy đủ trước, làm tròn sau. Đừng lấy hai con số đã làm tròn trong báo cáo rồi trừ tay.

## 🔬 3.4 — ROC/AUC: dựng đường cong bằng tay và đối chiếu hai công thức AUC độc lập

```python
import numpy as np

def roc_tu_tay(y_true, y_score):
    thu_tu = np.argsort(-y_score)      # sap giam dan theo xac suat
    y_sorted = y_true[thu_tu]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for nhan in y_sorted:
        if nhan == 1:
            tp += 1                    # buoc len (tang TPR)
        else:
            fp += 1                    # buoc sang phai (tang FPR)
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    return np.array(fpr), np.array(tpr)

def auc_hinh_thang(fpr, tpr):
    return np.trapezoid(tpr, fpr)      # dien tich duoi duong bang quy tac hinh thang
```

Đi từ mẫu xác suất cao nhất xuống thấp nhất chính là **hạ ngưỡng dần từ 1 xuống 0**. Gặp mẫu dương thật → TPR tăng, đường đi **lên**. Gặp mẫu âm thật (bị chấm nhầm điểm cao) → FPR tăng, đường đi **sang phải**. Toàn bộ đường ROC chỉ là "đi bộ" qua danh sách đã sắp xếp — O(n log n), không cần thử từng ngưỡng rời rạc như ở mục 3.3.

**Công thức AUC thứ hai, độc lập — dùng để tự đối chiếu (thống kê hạng Mann–Whitney U):**

```python
def auc_rank_sum(y_true, y_score):
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    thu_hang = np.argsort(np.argsort(y_score)) + 1   # hang 1..n, hang cao = diem cao
    tong_hang_duong = thu_hang[y_true == 1].sum()
    U = tong_hang_duong - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)
```

**Kiểm chứng bằng dữ liệu cụ thể (đã chạy thật, 12 mẫu):**

```python
y = np.array([0,1,0,1,1,0,1,0,1,0,0,1])
p = np.array([0.30,0.85,0.40,0.55,0.90,0.20,0.45,0.60,0.70,0.10,0.65,0.50])
```

```
AUC hình thang (tự vẽ ROC):   0.8333333333333334
AUC rank-sum (tự tính):        0.8333333333333334
AUC sklearn (roc_auc_score):   0.8333333333333334
```

Ba phương pháp hoàn toàn độc lập (hình học, tổ hợp, thư viện) khớp nhau tới 16 chữ số — bằng chứng cụ thể rằng cả ba đo đúng cùng một đại lượng.

Đây cũng là "cổng kiểm nội bộ" cho Bài tập 4: nếu `auc_rank_sum` và `auc_hinh_thang` khớp nhau nhưng không khớp `roc_auc_score`, lỗi nhiều khả năng nằm ở cách sắp xếp khi có **điểm trùng nhau** (ties) — cạm bẫy phổ biến khi tự cài ROC, rất hay gặp trong dữ liệu thực.

## 🔬 3.5 — Bài tập mẫu: kiểm chứng lại toàn bộ bằng code

```python
that = [1]*10 + [0]*10
doan = [1]*9 + [0]*1 + [1]*5 + [0]*5   # 9 dung trong 10 nhan 1, 5 sai trong 10 nhan 0

macro_f1(that, doan)                                                   # -> 0.6875
f1_score(that, doan, average="macro", labels=[0,1], zero_division=0)   # -> 0.6875
accuracy_score(that, doan)                                             # -> 0.7
```

Ba con số này khớp chính xác 100% với bài giải tay ở mục 4 — `macro_f1` tự cài, `f1_score` thư viện, và phép tính tay đều hội tụ về đúng 0,6875 và 0,7.

## 🔬 3.6 — Lời giải đầy đủ, chạy được ngay, cho Bài tập 1–3

> Bài tập 4 và 5 cần dữ liệu thật (`public_test.csv`, `training_set.csv`) mà tài liệu này không có sẵn — mục 3.4 ở trên đã cho đủ khung thuật toán và cách tự kiểm; khi có file thật, chỉ cần nạp `X_test`, `y_test`, `p` rồi chạy nguyên khối code đã cho.

**Bài tập 1** — đã giải ở 🔬 3.2. Kết quả: lệch có `labels=[0,1]` là `2.22e-16`; lệch không có `labels` là `0.5`.

**Bài tập 2** — đã giải ở 🔬 3.1. Kết quả: `1.00` và `0.50`, giải thích bằng cụm "lớp vắng mặt" (lớp 0 không hề xuất hiện trong `mot_lop`).

**Bài tập 3** — bảng bốn thước đo, cả hai tỉ lệ lệch, đã chạy thật:

```python
from sklearn.metrics import f1_score, accuracy_score

def bang_bon_thuoc_do(n_pos, n_total):
    that = [1]*n_pos + [0]*(n_total - n_pos)
    doan = [1]*n_total                       # mo hinh "doan 1 cho tat ca"
    acc         = accuracy_score(that, doan)
    f1_micro    = f1_score(that, doan, average="micro",    labels=[0,1], zero_division=0)
    f1_weighted = f1_score(that, doan, average="weighted", labels=[0,1], zero_division=0)
    f1_macro    = f1_score(that, doan, average="macro",    labels=[0,1], zero_division=0)
    return acc, f1_micro, f1_weighted, f1_macro

print(bang_bon_thuoc_do(950, 1000))   # ti le 95/5
print(bang_bon_thuoc_do(990, 1000))   # ti le 99/1
```

| Tỉ lệ lớp 1 | accuracy | F1 micro | F1 weighted | F1 macro |
|---|---|---|---|---|
| 95 % (950/1000) | 0,9500 | 0,9500 | 0,9256 | **0,4872** |
| 99 % (990/1000) | 0,9900 | 0,9900 | 0,9850 | **0,4975** |

Hàng đầu khớp chính xác bốn con số tài liệu gốc yêu cầu. **Điểm bất ngờ ở hàng thứ hai:** macro-F1 **không** rơi mạnh khi lệch lớp tăng từ 95/5 lên 99/1 — chỉ nhích từ 0,4872 lên 0,4975, gần như đứng yên quanh mốc 0,50. Lý do: một khi mô hình đoán sai **hoàn toàn** lớp thiểu số (Recall₀ = 0 luôn), F1 của lớp thiểu số luôn bằng đúng 0 bất kể lớp đó hiếm tới đâu. F1 của lớp đa số thì tiến dần về 1,0 khi lớp đa số càng áp đảo. Vậy macro-F1 = (F1_đa_số + 0)/2 → tiến về đúng **0,50** khi độ lệch lớp tăng vô hạn, chứ không rơi thấp hơn nữa. Nếu macro-F1 của mô hình bạn chỉ nhỉnh hơn 0,5 một chút, đó là tín hiệu mạnh rằng mô hình gần như đang "đoán bừa theo lớp đa số".

---

## Tổng kết nhanh — bảng tra cứu

| Câu hỏi | Trả lời ngắn |
|---|---|
| Dùng thước đo nào khi lệch lớp? | macro-F1, không phải accuracy |
| Luôn phải truyền gì cho `f1_score`? | `labels=[0, 1]` và `zero_division=0` |
| AUC đo cái gì? | Khả năng **xếp hạng** đúng, không phụ thuộc ngưỡng |
| Macro-F1 đo cái gì? | Chất lượng quyết định **tại một ngưỡng cụ thể**, trên cả hai lớp |
| Chọn ngưỡng ở đâu là an toàn? | Trên tập kiểm định tách từ *train*, không bao giờ trên *test* |
| Macro-F1 "sàn" của mô hình đoán bừa theo lớp đa số? | Luôn quanh 0,50, bất kể lớp thiểu số hiếm tới đâu |
| Trừ hai số đã làm tròn có đáng tin? | Không — luôn trừ trước, làm tròn sau |
