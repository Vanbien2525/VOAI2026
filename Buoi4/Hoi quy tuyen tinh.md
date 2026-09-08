# Hồi quy tuyến tính và Giảm độ dốc
### Tài liệu tổng hợp mở rộng — Buổi 04, Luyện Olympic Trí tuệ nhân tạo
*Dựa trên bài giảng của TS. Đỗ Phúc Hảo, ngày 25/8/2026 — bản mở rộng có ẩn dụ và chú thích chi tiết*

---

## Lời dẫn nhập

Tài liệu này dựng lại toàn bộ nội dung gốc của Buổi 04, **không cắt bỏ bất cứ phần nào**, và thêm ba lớp bổ sung ở mỗi mục:

1. **Bản chất/trực giác (intuition)** — vì sao công thức lại như vậy, không chỉ là "công thức trông như thế nào".
2. **Ẩn dụ đời thường** — để một khái niệm toán trừu tượng có một hình ảnh cụ thể để bám vào trong đầu.
3. **Chú thích code từng dòng** — vì đề thi sơ loại hỏi về hành vi của code (loss không giảm phải làm gì trước), nên hiểu từng dòng quan trọng hơn nhớ tên hàm.

Cấu trúc mục lớn được giữ đúng thứ tự bài gốc: Mục tiêu → Tóm tắt kiến thức → Hướng dẫn thực hành → Bài tập mẫu → Bài tập tự làm → Chuẩn bị buổi sau. Phần thêm mới (bảng thuật ngữ, tổng kết) được đặt ở cuối để không phá cấu trúc gốc.

---

## 1. Mục tiêu buổi học

> **Nguyên văn:** Tự cài giảm độ dốc bằng NumPy, không gọi scikit-learn, và ra cùng nghiệm với nó tới sai số máy.
>
> **Vì sao phải tự cài một thứ đã có sẵn:** Vì đề sơ loại hỏi những câu như "loss không giảm sau 5 vòng lặp đầu dù tỷ lệ học là 0,001, bạn nên làm gì đầu tiên". Không ai trả lời được câu ấy bằng cách nhớ tên hàm. Phải từng nhìn thấy loss bay lên vô cực, và từng biết vì sao.

### Vì sao mục tiêu này quan trọng hơn nó nghe có vẻ

`sklearn.linear_model.LinearRegression()` chỉ là hai dòng code. Nếu mục tiêu chỉ là "có nghiệm đúng", buổi học này không cần tồn tại. Nhưng trong phòng thi sơ loại, câu hỏi hiếm khi là "hãy code hồi quy tuyến tính" — nó thường là một trong các dạng sau:

- "Loss của bạn là `nan` sau vòng thứ 12. Nguyên nhân khả dĩ nhất là gì?"
- "Với tốc độ học đang dùng, sau 2000 vòng loss vẫn còn 334. Bạn tăng hay giảm tốc độ học, và vì sao?"
- "Hai đặc trưng của bạn có phạm vi giá trị chênh nhau 10.000 lần. Điều này ảnh hưởng gì đến việc chọn tốc độ học?"

Những câu này chỉ trả lời được nếu bạn từng **tự tay làm hỏng** thuật toán rồi tự sửa. Đó là lý do buổi học bắt bạn cài tay thay vì gọi thư viện — thư viện giấu đi đúng phần mà đề thi hỏi.

---

## 2. Tóm tắt kiến thức

### 2.1 Mô hình và hàm mất mát

> **Nguyên văn — Mô hình:**
> Mô hình tuyến tính dự đoán bằng một tổ hợp có trọng số của các đặc trưng, cộng một hệ số tự do:
>
> $$\hat{y}_i = w \cdot x_i + b \tag{1}$$

**Bản chất:** `x_i` là một điểm dữ liệu, biểu diễn bằng một vector gồm `d` con số (d đặc trưng — ví dụ diện tích nhà, số phòng, khoảng cách tới trung tâm). `w` cũng là một vector `d` con số — mỗi con số trong `w` nói lên "đặc trưng này quan trọng bao nhiêu và theo hướng nào". `w · x_i` là tích vô hướng — nhân từng cặp rồi cộng lại, cho ra **một con số duy nhất**. `b` là một hằng số cộng thêm, cho phép đường/mặt dự đoán không bị ép phải đi qua gốc toạ độ.

**Ẩn dụ:** Hãy tưởng tượng bạn định giá một căn nhà bằng cách hỏi 3 người bạn (3 đặc trưng): "diện tích", "số phòng ngủ", "khoảng cách tới trung tâm". Mỗi người bạn cho một con số (giá trị đặc trưng), và bạn gán cho mỗi người một "trọng số tin tưởng" `w` — người nào bạn tin hơn thì trọng số lớn hơn. Giá nhà dự đoán = tổng có trọng số của ba ý kiến, cộng thêm một khoản "giá sàn" cố định `b` (kiểu phí môi giới cơ bản mà nhà nào cũng có). Học mô hình tuyến tính chính là **học xem nên tin ai bao nhiêu** (tìm `w`) và **giá sàn là bao nhiêu** (tìm `b`).

**Về shape (đã quen từ buổi NumPy):** Với `n` điểm dữ liệu và `d` đặc trưng:
- `X` có shape `(n, d)` — mỗi hàng là một điểm dữ liệu.
- `w` có shape `(d,)`.
- `X @ w` có shape `(n,)` — đây chính là lúc quy tắc "trục nào bị nhân biến mất" phát huy: nhân `(n,d)` với `(d,)`, trục `d` biến mất, còn lại `(n,)`.
- `b` là một số vô hướng (scalar), được NumPy broadcast cộng vào toàn bộ `(n,)` — mỗi điểm dữ liệu được cộng cùng một `b`.
- `ŷ` (dự đoán) có shape `(n,)`, khớp với `y` (nhãn thật) shape `(n,)`.

> **Nguyên văn — Hàm mất mát:**
> Hàm mất mát bình phương trung bình đo mức sai:
>
> $$L(w,b) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2 \tag{2}$$

**Bản chất:** Đây là *Mean Squared Error* (MSE) — sai số bình phương trung bình. Ba lựa chọn trong công thức này không ngẫu nhiên:

- **Vì sao "bình phương" chứ không phải "trị tuyệt đối"?** Bình phương luôn dương (không cần lo về dấu triệt tiêu nhau), và quan trọng hơn: nó **khả vi mọi nơi** (có đạo hàm mượt), trong khi trị tuyệt đối có một điểm gãy tại 0 — gây khó cho giảm độ dốc. Bình phương cũng phạt nặng hơn với sai số lớn: sai 10 bị phạt gấp 100 lần sai 1, chứ không phải gấp 10 lần.
- **Vì sao "trung bình" (chia cho n) chứ không phải "tổng"?** Nếu dùng tổng, dataset càng nhiều điểm thì loss càng lớn một cách giả tạo — không so sánh được giữa các bộ dữ liệu khác kích thước, và quan trọng hơn: gradient cũng sẽ tỉ lệ thuận với `n`, khiến bạn phải đổi tốc độ học mỗi khi đổi cỡ dữ liệu. Chia cho `n` giữ cho loss và gradient có "quy mô" ổn định.

**Ẩn dụ:** Tưởng tượng bạn ném phi tiêu vào bia, và đo "độ tệ" của một lượt ném bằng bình phương khoảng cách từ điểm rơi tới tâm bia (ném lệch xa bị phạt rất nặng, ném gần tâm gần như không bị phạt). `L` chính là **độ tệ trung bình** qua tất cả các lượt ném. Mục tiêu học máy là chỉnh "cách ném" (tức `w`, `b`) để độ tệ trung bình càng nhỏ càng tốt.

---

### 2.2 Đạo hàm, viết tay một lần cho nhớ

> **Nguyên văn:**
> Đặt $r_i = \hat{y}_i - y_i$ là phần dư. Đạo hàm của $L$ theo từng tham số:
>
> $$\frac{\partial L}{\partial w} = \frac{2}{n}X^\top r, \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n} r_i \tag{3}$$
>
> Quy tắc cập nhật: đi ngược hướng đạo hàm một bước dài $\eta$.
>
> $$w \leftarrow w - \eta\frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta\frac{\partial L}{\partial b} \tag{4}$$

#### Đạo hàm này từ đâu ra? (lấp khoảng trống — bài gốc chỉ cho kết quả)

Xuất phát từ $L = \frac{1}{n}\sum_i (\hat y_i - y_i)^2 = \frac{1}{n}\sum_i r_i^2$, với $r_i = w\cdot x_i + b - y_i$.

Dùng quy tắc chuỗi (chain rule) — đạo hàm của "trong ngoặc" nhân với đạo hàm của "ngoài ngoặc":

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_i 2 r_i \cdot \frac{\partial r_i}{\partial w} = \frac{1}{n}\sum_i 2 r_i \cdot x_i = \frac{2}{n}\sum_i r_i x_i = \frac{2}{n} X^\top r$$

Vì $\frac{\partial r_i}{\partial w} = x_i$ (đạo hàm của $w\cdot x_i + b - y_i$ theo $w$ chính là $x_i$), và tổng $\sum_i r_i x_i$ viết gọn bằng phép nhân ma trận chính là $X^\top r$ — đây là lý do phép nhân ma trận `X.T @ du` xuất hiện trong code chứ không phải viết vòng lặp.

Tương tự với $b$: $\frac{\partial r_i}{\partial b} = 1$, nên $\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i r_i$.

#### Vì sao đi *ngược* hướng đạo hàm?

Đạo hàm tại một điểm cho biết **hướng nào loss tăng nhanh nhất** (hướng dốc lên). Muốn loss giảm, ta đi hướng ngược lại — hướng dốc xuống nhanh nhất. Đó là lý do công thức (4) có dấu trừ.

**Ẩn dụ:** Bạn đang đứng trên một quả đồi trong sương mù dày đặc, không nhìn thấy gì, chỉ **cảm nhận được độ dốc dưới chân**. Bạn muốn xuống thấp nhất có thể (loss nhỏ nhất). Chiến lược hợp lý: cảm nhận hướng nào dốc lên nhiều nhất (đó là gradient), rồi bước một bước theo hướng **ngược lại**. Lặp lại nhiều lần, bạn dần xuống tới đáy thung lũng. `η` (tốc độ học) chính là **độ dài mỗi bước chân**.

#### Chú thích code — `mot_buoc` (một bước giảm độ dốc)

```python
def mot_buoc(X, y, w, b, lr):
    n = X.shape[0]
    du = X @ w + b - y              # phần dư r, shape (n,) — sai số của từng điểm
    w = w - lr * (2.0 / n) * (X.T @ du)   # cập nhật trọng số theo công thức (3)-(4)
    b = b - lr * (2.0 / n) * du.sum()     # cập nhật hệ số tự do
    return w, b
```

Đọc từng dòng:
- `n = X.shape[0]`: số điểm dữ liệu (số hàng), cần để chia trung bình.
- `du = X @ w + b - y`: đây chính là $r$ trong công thức. `X @ w` shape `(n,)`, cộng `b` (broadcast), trừ `y` shape `(n,)` → `du` shape `(n,)`. Biến này tên là "du" (viết tắt "dư" — phần dư).
- `X.T @ du`: `X.T` có shape `(d, n)`, nhân với `du` shape `(n,)` → kết quả shape `(d,)`, đúng shape của `w`. Đây chính là $X^\top r$ trong công thức (3), tính gradient cho **toàn bộ** `w` chỉ bằng một phép nhân ma trận — không cần vòng lặp qua từng đặc trưng.
- `du.sum()`: tổng các phần dư — đúng là $\sum_i r_i$ trong công thức cho `b`.
- Toàn bộ hàm không có vòng lặp `for` nào — đây là **vectorization** mà bạn đã học ở buổi NumPy: thay vì lặp qua từng điểm dữ liệu, dùng phép nhân ma trận để tính gradient cho tất cả cùng lúc.

> **Nguyên văn:** Toàn bộ thuật toán nằm trong bốn dòng ấy. Mọi thứ còn lại của buổi này là về chọn $\eta$ và chuẩn bị dữ liệu.

Đây là câu quan trọng nhất của cả bài: **thuật toán thì đơn giản, cái khó là vận hành nó cho đúng**. Phần tiếp theo giải thích chính xác vì sao "chọn η" và "chuẩn bị dữ liệu" lại chiếm gần hết nội dung còn lại.

---

### 2.3 Tốc độ học: quá lớn thì bay, quá nhỏ thì bò

> **Nguyên văn — Đo thật, trên một bộ dữ liệu ba đặc trưng có thang đo lệch nhau** (cột thứ nhất quanh 1, cột thứ hai quanh 100, cột thứ ba quanh 10.000):

| Chuẩn hoá | Tốc độ học | Kết quả sau 2000 vòng |
|---|---|---|
| không | $10^{-2}$ | phân kỳ, ra `inf` sau 27 vòng |
| không | $10^{-6}$ | phân kỳ, ra `inf` sau 85 vòng |
| không | $10^{-9}$ | còn sống, nhưng mất mát vẫn là 334,26 |
| có | 0,5 | mất mát 0,010353 |
| có | 0,1 | mất mát 0,010353 |
| có | 0,01 | mất mát 0,010353 |

> **Ba dòng đầu là chân dung của một buổi chiều bị mất.** Không chuẩn hoá, khoảng tốc độ học dùng được hẹp tới mức gần như không tồn tại. Giảm từ $10^{-2}$ xuống $10^{-6}$, tức là bốn bậc, vẫn phân kỳ. Xuống tới $10^{-9}$ thì hết phân kỳ, nhưng lúc ấy bước đi ngắn quá nên sau hai nghìn vòng mất mát vẫn còn 334.
>
> Người mới thường kết luận sai ở đúng chỗ này: "mô hình tuyến tính không hợp bài toán". Không phải. Dữ liệu chưa được chuẩn hoá, thế thôi.
>
> Sau khi chuẩn hoá, cả ba tốc độ học đều về cùng một nghiệm. Chỉ khác nhau ở chỗ mất bao lâu:

| Tốc độ học | Số vòng để vào trong 1% của nghiệm |
|---|---|
| 0,5 | 4 |
| 0,1 | 37 |
| 0,01 | 408 |
| 0,001 | 4108 |

> Giảm tốc độ học mười lần thì số vòng tăng khoảng mười lần. Quan hệ ấy gần như tuyến tính, và nó cho bạn cách ước lượng ngân sách thời gian trong phòng thi.

#### Vì sao lại xảy ra chuyện này? (phần lý thuyết bị nói tắt trong bài gốc — mở rộng ở đây)

Đây là phần quan trọng nhất để hiểu *bản chất*, không chỉ thuộc bảng số.

**Ẩn dụ quả bóng trong thung lũng:** Loss `L(w,b)` giống một cái thung lũng (bowl-shaped). Giảm độ dốc giống thả một quả bóng lăn xuống đáy thung lũng, mỗi bước nhảy một đoạn dài tỉ lệ với `η` và độ dốc tại chỗ đang đứng.

- **η quá lớn:** quả bóng nhảy quá xa, vọt qua bên kia thung lũng, mà bên kia lại càng dốc hơn (vì càng xa đáy). Bước tiếp theo còn nhảy xa hơn nữa. Đây là vòng lặp tự khuếch đại — sau vài chục bước, vị trí bay ra vô cực (`inf`). Đó chính xác là điều xảy ra ở $10^{-2}$: phân kỳ sau 27 vòng.
- **η quá nhỏ:** quả bóng nhích từng milimet. Không bao giờ bay ra ngoài, nhưng sau 2000 bước vẫn còn cách đáy rất xa (loss = 334,26 ở $10^{-9}$) — đó là "bò" chứ không phải "học".
- **η vừa đủ:** quả bóng lăn mượt xuống đáy trong vài chục bước.

**Vì sao dữ liệu không chuẩn hoá làm khoảng η an toàn gần như biến mất?** Thung lũng không phải lúc nào cũng tròn như cái bát — nó có thể là một thung lũng **dẹt, kéo dài** (giống một cái máng nước hẹp và dài) nếu các đặc trưng có thang đo chênh lệch nhau rất nhiều. Với cột dữ liệu quanh giá trị 10.000, một thay đổi nhỏ của trọng số tương ứng gây ra thay đổi loss rất lớn (dốc cực đứng theo hướng đó). Với cột quanh giá trị 1, cùng một thay đổi trọng số gây ra ảnh hưởng rất nhỏ (dốc rất thoải theo hướng đó). Kết quả: thung lũng bị "méo" thành hình elip rất dẹt.

Một `η` đủ an toàn cho hướng dốc đứng (không làm bay) lại **quá nhỏ** để di chuyển có ý nghĩa theo hướng thoải. Đây chính là lý do không tồn tại một `η` vừa nhanh vừa ổn định khi dữ liệu chưa chuẩn hoá — bạn buộc phải chọn giữa "bay" và "bò".

**Công thức lý thuyết (được nhắc tới trong Bài tập 3 của bài gốc):** Với hàm mất mát bậc hai (như MSE), giảm độ dốc hội tụ khi và chỉ khi

$$\eta < \frac{2}{\lambda_{\max}}$$

trong đó $\lambda_{\max}$ là trị riêng lớn nhất (eigenvalue lớn nhất) của ma trận Hessian $\frac{2}{n}X^\top X$. Trị riêng lớn nhất tương ứng với **hướng dốc đứng nhất** của thung lũng (giống cột dữ liệu quanh 10.000 trong ví dụ). Khi các cột dữ liệu chênh lệch thang đo càng lớn, $\lambda_{\max}$ càng lớn, $\eta$ giới hạn càng nhỏ — trong khi bạn cần bước đủ dài để xử lý hướng thoải (liên quan tới trị riêng nhỏ nhất $\lambda_{\min}$). Tỉ số $\lambda_{\max}/\lambda_{\min}$ gọi là **số điều kiện (condition number)** của bài toán — số điều kiện càng lớn, bài toán càng "khó chiều" với giảm độ dốc thô (không chuẩn hoá).

> Đây chính xác là nội dung Bài tập 3 yêu cầu bạn đo bằng thực nghiệm rồi đối chiếu với công thức lý thuyết này.

**Vì sao giảm η mười lần thì số vòng tăng khoảng mười lần (bảng thứ hai)?** Khi đã ở trong vùng hội tụ an toàn (không phân kỳ), tốc độ hội tụ gần đáy tỉ lệ nghịch với `η` — bước đi ngắn hơn `k` lần thì cần nhiều hơn khoảng `k` lần số bước để đi cùng một quãng đường. Quan hệ gần-tuyến-tính này rất hữu ích để **ước lượng thời gian chạy trong phòng thi**: nếu bạn biết `η=0,1` cần 37 vòng, bạn có thể đoán ngay `η=0,001` cần khoảng 3700–4100 vòng mà không cần chạy thử.

---

### 2.4 Chuẩn hoá đặc trưng

> **Nguyên văn:**
> ```python
> tb = X.mean(axis=0)
> sd = X.std(axis=0)
> sd[sd == 0] = 1.0   # a constant column would divide by zero
> Xc = (X - tb) / sd
> ```

#### Bản chất: đây là z-score

Phép biến đổi này biến mỗi cột thành **z-score**: mỗi giá trị được diễn dịch lại thành "cách trung bình bao nhiêu độ lệch chuẩn". Sau phép biến đổi, mọi cột đều có trung bình 0 và độ lệch chuẩn 1 — nghĩa là mọi cột đều nằm trên **cùng một thang đo**, bất kể đơn vị gốc là gì (mét, VNĐ, số phòng...).

**Ẩn dụ:** Giống như so sánh điểm thi giữa các môn có thang điểm khác nhau (Toán thang 10, tiếng Anh thang 990 IELTS/TOEFL kiểu điểm). Nếu so trực tiếp con số thô, "điểm tiếng Anh 900" trông "quan trọng hơn" điểm Toán 9 dù thực ra cả hai đều là học sinh giỏi top đầu. Chuẩn hoá là quy tất cả về "đứng thứ mấy độ lệch chuẩn so với mặt bằng chung" — lúc đó so sánh mới công bằng.

#### Vì sao chuẩn hoá cứu được giảm độ dốc

Nối lại với mục 2.3: khi mọi cột có cùng độ lệch chuẩn (=1), "độ dốc" theo mọi hướng trở nên gần đều nhau — thung lũng dẹt-kéo-dài trở lại gần tròn như cái bát. Lúc này một `η` duy nhất phù hợp cho **mọi hướng cùng lúc**, đó là lý do bảng số liệu cho thấy `η = 0,5 / 0,1 / 0,01` sau chuẩn hoá đều hội tụ về cùng nghiệm — không còn vùng "chọn sai là bay" hẹp như trước.

#### Chú thích code từng dòng

- `X.mean(axis=0)`: theo mnemonic đã học ở buổi NumPy — "trục nào bạn nêu tên thì trục đó biến mất khỏi kết quả". `axis=0` là trục hàng (trục các điểm dữ liệu) → trục đó biến mất, kết quả là trung bình của **từng cột**, shape `(d,)`.
- `X.std(axis=0)`: tương tự, độ lệch chuẩn của từng cột, shape `(d,)`.
- `sd[sd == 0] = 1.0`: đây là một **lưới an toàn**. Nếu một cột là hằng số (mọi giá trị giống nhau, ví dụ toàn số 5 — đúng là tình huống Bài tập 5), độ lệch chuẩn của nó bằng 0, và `(X - tb) / sd` sẽ chia cho 0 → `nan`, mà **không có bất kỳ thông báo lỗi nào** (NumPy âm thầm trả `nan`/`inf` chứ không raise Exception). Dòng này thay 0 bằng 1 để tránh chia-cho-0; vì tử số của cột đó `(X - tb)` cũng bằng 0 (mọi giá trị = trung bình), nên `0/1 = 0` — cột hằng số sau chuẩn hoá trở thành cột toàn số 0, an toàn tuyệt đối.
- `Xc = (X - tb) / sd`: broadcast trừ `(n,d) - (d,)` → trừ theo từng cột (mỗi hàng trừ đi cùng vector `tb`), rồi chia broadcast tương tự.

> **Cảnh báo quan trọng — Chuẩn hoá bằng thống kê của tập nào (nguyên văn):**
> Trung bình và độ lệch chuẩn phải tính trên tập huấn luyện, rồi đem đúng hai con số ấy áp cho tập kiểm tra. Tính lại trên tập kiểm tra là để dữ liệu kiểm tra rò rỉ vào quy trình, và điểm của bạn sẽ đẹp lên một cách giả tạo. Buổi 07 gọi tên hiện tượng này.

#### Vì sao đây là rò rỉ dữ liệu (data leakage) — mở rộng vì bạn đã gặp cảnh báo tương tự ở buổi Pandas

**Bản chất:** Mục đích của tập kiểm tra là **giả lập dữ liệu tương lai mà mô hình chưa từng thấy**. Nếu bạn tính `mean`/`std` trên cả tập kiểm tra rồi mới chuẩn hoá, thì thông tin thống kê của tập kiểm tra (vốn "chưa nên biết") đã ngấm vào bước tiền xử lý — mô hình gián tiếp "nhìn trộm" một phần đặc điểm của dữ liệu nó sắp bị kiểm tra.

**Ẩn dụ:** Giống việc học sinh làm bài kiểm tra thử, nhưng đề kiểm tra thử lại được chấm dựa trên đúng phân bố điểm của chính lớp học đó (biết trước "lớp này trung bình sẽ đạt khoảng bao nhiêu điểm"). Điểm số trông đẹp hơn thực lực, vì một phần thông tin về "bài kiểm tra sẽ trông như thế nào" đã bị lộ ra trước khi kiểm tra thật.

**Quy tắc thực hành đúng:**
```python
tb, sd = X_train.mean(axis=0), X_train.std(axis=0)   # chỉ dùng tập huấn luyện
sd[sd == 0] = 1.0
X_train_c = (X_train - tb) / sd
X_test_c  = (X_test  - tb) / sd     # DÙNG LẠI đúng tb, sd đã tính — không tính lại
```

---

### 2.5 Nghiệm đóng, để có cái mà đối chiếu

> **Nguyên văn:**
> Hồi quy tuyến tính có nghiệm giải tích, không cần lặp:
>
> $$w = (X^\top X)^{-1} X^\top y \tag{5}$$
>
> Trong thực tế dùng `np.linalg.lstsq` thay vì nghịch đảo trực tiếp, vì nó ổn định hơn khi $X^\top X$ gần suy biến. Nghiệm đóng đáng giá không phải vì nó nhanh, mà vì nó là trọng tài: bản lặp của bạn phải chạy tới đúng chỗ ấy.

#### Công thức này từ đâu ra? (lấp khoảng trống)

Loss $L(w)$ là một hàm bậc hai theo `w` (giống một cái bát parabol nhiều chiều), nên nó chỉ có **đúng một điểm thấp nhất**, và tại điểm đó gradient bằng 0 — không cần dò từng bước như giảm độ dốc, chỉ cần **giải phương trình** $\frac{\partial L}{\partial w} = 0$ trực tiếp:

$$\frac{2}{n}X^\top(Xw - y) = 0 \;\;\Rightarrow\;\; X^\top X w = X^\top y \;\;\Rightarrow\;\; w = (X^\top X)^{-1}X^\top y$$

(Phương trình $X^\top X w = X^\top y$ được gọi là **phương trình chuẩn tắc** — normal equation.)

> **Lưu ý nhỏ nhưng dễ gây nhầm lẫn:** công thức (5) chỉ cho ra `w`, không có `b` riêng. Trong thực hành, để có cả `b`, người ta **thêm một cột toàn số 1 vào `X`** trước khi áp công thức — lúc đó `b` trở thành một phần tử trong vector `w` mở rộng (tương ứng với "đặc trưng" luôn luôn bằng 1). Đây chính xác là kỹ thuật được nhắc tới ở Bài tập 5 khi thêm cột hằng số vào `X`.

#### Vì sao không nghịch đảo trực tiếp mà dùng `lstsq`?

**Bản chất "gần suy biến" (near-singular) là gì?** Một ma trận suy biến (singular) là ma trận không có nghịch đảo — giống như số 0 không có số nghịch đảo ($1/0$ không tồn tại). "Gần suy biến" nghĩa là ma trận *có* nghịch đảo về mặt lý thuyết, nhưng nghịch đảo đó cực kỳ nhạy cảm — chỉ cần sai số làm tròn cực nhỏ của máy tính cũng khiến kết quả nghịch đảo sai lệch khủng khiếp (giống chia cho một số rất gần 0: kết quả nổ bùng lên do sai số nhỏ ở mẫu số).

Tình huống này xảy ra khi các cột của `X` gần như **phụ thuộc tuyến tính vào nhau** (ví dụ hai đặc trưng gần như là bản sao của nhau, hoặc một cột gần như hằng số). `np.linalg.inv(X.T @ X)` trong trường hợp này cho kết quả rất không ổn định về số học.

**`np.linalg.lstsq`** giải phương trình chuẩn tắc bằng một phương pháp số học ổn định hơn (dựa trên phân rã SVD ở phía dưới), không cần tính nghịch đảo tường minh, nên chịu được các trường hợp gần suy biến tốt hơn nhiều.

**Ẩn dụ:** Tính nghịch đảo trực tiếp giống như đi qua một cây cầu rất hẹp và trơn — đúng hướng thì qua được, nhưng chỉ cần lệch một chút (sai số làm tròn) là ngã. `lstsq` giống đi qua một cây cầu rộng hơn, có lan can — vẫn tới đích, nhưng chịu được xô đẩy nhỏ trên đường đi.

#### Vì sao nghiệm đóng là "trọng tài" chứ không phải "cách làm nhanh hơn"

Điểm mấu chốt: nghiệm đóng và giảm độ dốc đi **hai con đường hoàn toàn khác nhau về mặt thuật toán** (một bên giải hệ phương trình trực tiếp, một bên dò dần từng bước). Nếu cả hai đều đúng, chúng *bắt buộc* phải gặp nhau ở cùng một điểm — vì hàm loss chỉ có đúng một điểm thấp nhất. Nếu bản giảm độ dốc tự cài của bạn lệch khỏi nghiệm đóng, gần như chắc chắn code của bạn có lỗi (không phải do thuật toán giảm độ dốc "không tốt bằng"). Đây là kỹ thuật tự-kiểm (self-check) dùng xuyên suốt toàn bộ khoá học.

---

## 3. Hướng dẫn thực hành

### 3.1 Bước 1: dựng dữ liệu mà bạn biết trước đáp án

> **Nguyên văn:**
> ```python
> import numpy as np
>
> rng = np.random.default_rng(0)
> n, d = 500, 3
> # Three columns on wildly different scales, on purpose: this is what makes gradient
> # descent fail without normalisation, and it is common in real tabular data.
> X = rng.random((n, d)) * np.array([1.0, 100.0, 10000.0])
> w_that = np.array([2.0, -0.5, 0.01])
> b_that = 3.0
> y = X @ w_that + b_that + 0.1 * rng.standard_normal(n)
> ```
>
> **Luôn bắt đầu bằng dữ liệu tự sinh:** Bạn biết trước `w_that` và `b_that`, nên bạn biết mô hình phải tìm ra cái gì. Khi thuật toán sai, bạn biết ngay là nó sai, thay vì ngồi đoán xem dữ liệu có vấn đề hay code có vấn đề.

#### Chú thích từng dòng

- `rng = np.random.default_rng(0)`: tạo một bộ sinh số ngẫu nhiên có "seed" (hạt giống) = 0. Đây chính là khái niệm `random_state`/seed mà tài liệu Pandas của bạn đã nhắc tới như một "khái niệm cầu nối" — cố định seed để mỗi lần chạy lại cho **đúng cùng một bộ số ngẫu nhiên**, giúp kết quả tái lập được (reproducible). Không có seed cố định, mỗi lần chạy bạn sẽ được một dữ liệu khác, khó so sánh và khó gỡ lỗi.
- `n, d = 500, 3`: 500 điểm dữ liệu, 3 đặc trưng.
- `X = rng.random((n, d)) * np.array([1.0, 100.0, 10000.0])`: `rng.random((n,d))` sinh ma trận `(500, 3)` số ngẫu nhiên đều trong `[0,1)`. Nhân broadcast với vector `[1, 100, 10000]` (shape `(3,)`) khiến **cột 1 nằm trong khoảng [0,1), cột 2 trong [0,100), cột 3 trong [0,10000)** — đây chính là "ba đặc trưng có thang đo lệch nhau" được nhắc ở mục 2.3, cố ý dựng để tái hiện đúng vấn đề chênh lệch tỉ lệ.
- `w_that`, `b_that`: đây là **đáp án đúng thật sự**, bạn tự đặt ra trước — mục tiêu của giảm độ dốc là "khám phá lại" đúng hai con số này (hoặc phiên bản tương đương trong không gian đã chuẩn hoá).
- `y = X @ w_that + b_that + 0.1 * rng.standard_normal(n)`: nhãn `y` = mô hình tuyến tính đúng (`X @ w_that + b_that`) cộng thêm **nhiễu Gauss** với độ lệch chuẩn 0,1 (`rng.standard_normal(n)` sinh nhiễu chuẩn tắc, nhân 0,1 để thu nhỏ biên độ nhiễu). Nhiễu này mô phỏng thực tế: dữ liệu thật không bao giờ khớp hoàn hảo với một công thức tuyến tính đơn giản — luôn có sai số đo lường/yếu tố chưa biết.

**Vì sao có nhiễu lại quan trọng:** Nếu không có nhiễu, mất mát thấp nhất có thể đạt được là 0 — chính xác tuyệt đối. Có nhiễu 0,1, mất mát thấp nhất **không thể** về 0 — điều này chính là chìa khoá lý giải con số 0,010353 ở bước 4 (xem giải thích chi tiết bên dưới).

### 3.2 Bước 2: cài giảm độ dốc

> **Nguyên văn:**
> ```python
> def giam_do_doc(X, y, lr, so_vong=2000):
>     """Plain batch gradient descent. Returns weights, bias, and the loss history."""
>     n, d = X.shape
>     w = np.zeros(d)
>     b = 0.0
>     lich_su = []
>     for _ in range(so_vong):
>         du = X @ w + b - y
>         mat = float((du ** 2).mean())
>         lich_su.append(mat)
>         # Stop the moment the loss stops being a number, otherwise the next few thousand
>         # iterations just multiply infinities and the traceback tells you nothing.
>         if not np.isfinite(mat):
>             break
>         w = w - lr * (2.0 / n) * (X.T @ du)
>         b = b - lr * (2.0 / n) * du.sum()
>     return w, b, lich_su
> ```

#### Chú thích từng dòng

- `n, d = X.shape`: lấy số điểm dữ liệu và số đặc trưng.
- `w = np.zeros(d)`, `b = 0.0`: **khởi tạo tại gốc toạ độ**. Với hồi quy tuyến tính (hàm loss lồi, chỉ có một đáy), khởi tạo ở đâu không quan trọng — cuối cùng vẫn hội tụ về cùng một điểm (khác với mạng nơ-ron sâu, nơi khởi tạo ảnh hưởng nhiều hơn).
- `lich_su = []`: danh sách lưu lại giá trị loss qua từng vòng lặp — chính là dữ liệu cần cho Bài tập 2 (vẽ đường cong mất mát).
- Vòng lặp `for _ in range(so_vong)`: chạy tối đa `so_vong` bước, mỗi bước:
  - `du = X @ w + b - y`: tính phần dư `r` với `w`, `b` **hiện tại** (trước khi cập nhật).
  - `mat = float((du ** 2).mean())`: đây chính là công thức (2) — bình phương từng phần dư rồi lấy trung bình. `float(...)` để chuyển từ kiểu số NumPy sang số Python thuần, tiện lưu trữ/so sánh.
  - `lich_su.append(mat)`: ghi lại loss của vòng này — ghi **trước** khi kiểm tra phân kỳ, để dòng lịch sử phản ánh đúng cả bước làm nó nổ (hữu ích khi debug).
  - `if not np.isfinite(mat): break`: đây là "phanh khẩn cấp". `np.isfinite` trả `False` nếu `mat` là `inf`, `-inf`, hoặc `nan`. Nếu loss đã hỏng, code **dừng ngay lập tức** thay vì tiếp tục lặp hàng nghìn vòng vô nghĩa (nhân vô cực với vô cực, không có traceback nào giải thích được nguyên nhân gốc). Đây là thói quen kỹ thuật rất đáng học: **phát hiện lỗi sớm, dừng sớm**, thay vì để chương trình "chạy ngầm hỏng" trong im lặng.
  - `w = w - lr * (2.0/n) * (X.T @ du)`, `b = b - lr * (2.0/n) * du.sum()`: đúng công thức cập nhật (4), dùng `du` (dựa trên `w`, `b` **cũ**) — đây là điểm quan trọng: cả `w` và `b` phải được cập nhật dựa trên **cùng một** `du` tính từ tham số trước khi cập nhật (không được cập nhật `w` trước rồi mới tính `du` mới để cập nhật `b` — nếu làm vậy, `b` sẽ bị cập nhật lệch pha, dùng "gradient cũ nhưng tính từ vị trí mới", một lỗi tinh vi dễ mắc khi tự cài tay).
- `return w, b, lich_su`: trả về nghiệm cuối cùng và toàn bộ lịch sử loss.

### 3.3 Bước 3: chạy chưa chuẩn hoá, và nhìn nó bay

> **Nguyên văn:**
> ```python
> for lr in (1e-2, 1e-6, 1e-9):
>     w, b, h = giam_do_doc(X, y, lr)
>     if np.isfinite(h[-1]):
>         print("lr=%-8.0e mat cuoi %.4f" % (lr, h[-1]))
>     else:
>         print("lr=%-8.0e phan ky sau %d vong" % (lr, len(h)))
> ```

**Đọc kết quả:** `h[-1]` là giá trị loss cuối cùng trong lịch sử (phần tử cuối của danh sách). Nếu nó hữu hạn (`np.isfinite`), in ra giá trị mất mát; nếu không, in ra "phân kỳ sau bao nhiêu vòng" — chính là `len(h)`, vì vòng lặp đã `break` ngay khi phát hiện `mat` không hữu hạn, nên độ dài `h` bằng đúng số vòng đã chạy được trước khi hỏng.

Kết quả khớp với bảng ở mục 2.3: `1e-2` và `1e-6` đều phân kỳ (27 và 85 vòng), `1e-9` sống sót nhưng dừng ở mất mát 334,26 sau đủ 2000 vòng — không phân kỳ nên nhánh `else` không chạy, giá trị in ra là mất mát cuối cùng.

### 3.4 Bước 4: chuẩn hoá, rồi chạy lại

> **Nguyên văn:**
> ```python
> tb, sd = X.mean(axis=0), X.std(axis=0)
> Xc = (X - tb) / sd
> w, b, h = giam_do_doc(Xc, y, 0.5)
> print("mat cuoi %.6f" % h[-1])
> ```
>
> Mất mát cuối là 0,010353. Con số ấy không về 0, và không nên về 0: dữ liệu có nhiễu với độ lệch chuẩn 0,1, nên mất mát bình phương thấp nhất có thể vào khoảng $0,1^2$, tức 0,01. Bạn vừa chạm sàn lý thuyết.
>
> **Biết trước sàn là một kỹ năng riêng:** Nếu bạn không tính trước rằng sàn là 0,01, bạn sẽ ngồi tinh chỉnh thêm nửa tiếng để kéo mất mát từ 0,0104 xuống 0,0103, mà phần dư ấy là nhiễu chứ không phải mô hình. Thói quen này áp thẳng vào phòng thi: hãy hỏi điểm cao nhất về mặt lý thuyết là bao nhiêu trước khi tối ưu.

#### Vì sao sàn lý thuyết đúng bằng bình phương độ lệch chuẩn nhiễu — giải thích sâu

Đây là một trong những khái niệm quan trọng nhất của toàn bộ machine learning: **sai số không thể triệt tiêu (irreducible error)**.

Nhãn thật được sinh ra bởi `y = tín_hiệu_thật + nhiễu`, với nhiễu có độ lệch chuẩn 0,1. Cho dù mô hình của bạn tìm ra **chính xác tuyệt đối** `w_that` và `b_that` (phần "tín hiệu thật"), phần dư `r_i = ŷ_i - y_i` vẫn còn lại đúng bằng phần nhiễu đã cộng vào — vì mô hình không thể đoán trước một con số ngẫu nhiên. Vậy $\mathbb{E}[r_i^2] \approx \text{Var(nhiễu)} = 0,1^2 = 0,01$ — đây chính là **mức sàn**, hay còn gọi là *Bayes error* / *irreducible error* trong các tài liệu ML — phần sai số tồn tại ngay cả với mô hình hoàn hảo nhất có thể, vì bản thân dữ liệu vốn có tính ngẫu nhiên.

**Ẩn dụ:** Bạn đang cố đoán cân nặng một người dựa vào chiều cao. Kể cả nếu bạn có công thức hoàn hảo về mối quan hệ trung bình giữa chiều cao và cân nặng, hai người cao bằng nhau vẫn có thể nặng khác nhau (do gen, chế độ ăn, cơ địa...). Phần "khác nhau dù cùng chiều cao" đó là nhiễu tự nhiên — không mô hình nào, dù giỏi đến đâu, xoá được nó.

**Ý nghĩa thực hành trong phòng thi:** Trước khi lao vào tinh chỉnh (tune) tốc độ học, số vòng lặp, hay kiến trúc mô hình để "ép" loss xuống thấp hơn nữa, hãy tự hỏi: **sàn lý thuyết ở đây là bao nhiêu?** Nếu bạn đã chạm sàn (như trường hợp 0,010353 rất gần 0,01), phần chênh lệch còn lại (0,000353) không phải do mô hình dở — đó là nhiễu ngẫu nhiên hoặc sai số dừng sớm của quá trình lặp hữu hạn. Tiếp tục tối ưu chỗ đó là lãng phí thời gian quý giá của phòng thi.

---

## 4. Bài tập mẫu (đã giải — phân tích sâu)

> **Đề:** Chứng minh bản giảm độ dốc của bạn ra cùng nghiệm với `LinearRegression` của scikit-learn.
>
> **Phân tích (nguyên văn):** Hai bên đi hai đường khác nhau: một bên lặp, một bên giải hệ phương trình. Nếu chúng gặp nhau ở cùng một điểm thì gần như chắc chắn cả hai đều đúng. Đây chính là kiểu đối chiếu mà cả kho này dùng để tự kiểm.
>
> **Lời giải:**
> ```python
> from sklearn.linear_model import LinearRegression
>
> w_gd, b_gd, _ = giam_do_doc(Xc, y, lr=0.5, so_vong=2000)
>
> mo_hinh = LinearRegression().fit(Xc, y)
> w_sk, b_sk = mo_hinh.coef_, mo_hinh.intercept_
>
> lech = max(np.abs(w_gd - w_sk).max(), abs(b_gd - b_sk))
> print("GD w =", np.round(w_gd, 5), "b = %.5f" % b_gd)
> print("sklearn w =", np.round(w_sk, 5), "b = %.5f" % b_sk)
> print("lech lon nhat: %.3e" % lech)
> ```
>
> **Kết quả:**
> ```
> GD w = [ 0.59386 -14.00213 29.13224] b = 27.74317
> sklearn w = [ 0.59386 -14.00213 29.13224] b = 27.74317
> lech lon nhat: 5.329e-15
> ```
>
> **Cổng kiểm:** Độ lệch dưới $10^{-9}$. Ở đây được $5,3\times10^{-15}$, tức là sát sai số máy.

#### Chú thích thêm

- `LinearRegression()` bên trong scikit-learn không dùng giảm độ dốc — nó dùng chính nghiệm đóng (hoặc một biến thể ổn định số học của nó, tương tự `lstsq`) đã nói ở mục 2.5. Vậy phép so sánh này thực chất là: **giảm độ dốc tự cài** đối chiếu với **nghiệm đóng** (thông qua thư viện) — đúng như bài mục 2.5 đã nói: nghiệm đóng là trọng tài.
- `w_gd - w_sk`, lấy trị tuyệt đối lớn nhất bằng `.max()`, rồi so với độ lệch của `b`, lấy `max()` toàn cục — đây là cách đo "sai số tệ nhất trong tất cả các tham số", một kiểu kiểm định chặt hơn là chỉ so sai số trung bình.
- Độ lệch $5,3\times10^{-15}$ cực kỳ nhỏ — nhỏ đến mức đó là **sai số làm tròn dấu phẩy động (floating-point)** của chính máy tính, không phải sai số thuật toán. Đây là "sai số máy" (machine precision) được nhắc ở đề bài — mức lệch nhỏ nhất có thể có giữa hai phép tính số học khác đường đi trên cùng một máy tính.

> **Trọng số tìm được không giống `w_that`, và đó là đúng (nguyên văn):**
> `w_that` là $[2,0;\ -0,5;\ 0,01]$, còn nghiệm tìm được là $[0,594;\ -14,00;\ 29,13]$. Không sai chỗ nào cả: mô hình được huấn luyện trên `Xc`, tức dữ liệu đã chuẩn hoá, nên trọng số của nó sống trong một hệ toạ độ khác. Muốn quay về hệ gốc thì phải chia cho độ lệch chuẩn của từng cột.
>
> Đây là chỗ rất hay bị hiểu nhầm khi đọc độ quan trọng của đặc trưng: so trọng số giữa các cột chỉ có nghĩa khi các cột đã cùng thang đo.

#### Mối quan hệ toán học đầy đủ giữa hai hệ toạ độ (lấp khoảng trống — bài gốc chỉ nói "phải chia cho độ lệch chuẩn", đây là công thức đầy đủ)

Gọi $w_c, b_c$ là nghiệm học được trên dữ liệu đã chuẩn hoá $X_c = (X - \bar{X})/\sigma$, và $w_o, b_o$ là nghiệm tương ứng trên dữ liệu gốc $X$. Vì cả hai phải cho ra **cùng một dự đoán** với cùng một điểm dữ liệu:

$$\hat y = X_c \cdot w_c + b_c = \frac{X - \bar X}{\sigma}\cdot w_c + b_c = X\cdot\frac{w_c}{\sigma} + \left(b_c - \bar X \cdot \frac{w_c}{\sigma}\right)$$

So khớp với dạng $\hat y = X \cdot w_o + b_o$, suy ra:

$$w_o = \frac{w_c}{\sigma} \qquad\qquad b_o = b_c - \sum_j \bar X_j \cdot \frac{w_{c,j}}{\sigma_j}$$

(phép chia $w_c/\sigma$ ở đây là chia từng phần tử, `element-wise`, giống code NumPy `w_c / sd`). Công thức này chính là chìa khoá để giải **Bài tập 1** bên dưới — bạn có công thức, việc còn lại là thay số và đối chiếu với `w_that`, `b_that`.

**Ẩn dụ cho "sống trong hệ toạ độ khác":** Giống việc đo cùng một quãng đường bằng mét và bằng dặm — con số khác nhau hoàn toàn (`1609` so với `1`), nhưng cùng mô tả một khoảng cách vật lý như nhau. Trọng số $w_c$ và $w_o$ cũng vậy: khác nhau về con số vì "đơn vị đo" của trục dữ liệu đã bị đổi (chuẩn hoá), nhưng cùng mô tả một mô hình dự đoán như nhau.

**Bài học thực hành quan trọng được nêu trong bài gốc:** Không được lấy trực tiếp độ lớn của $w_c$ (trọng số sau chuẩn hoá) để kết luận "đặc trưng nào quan trọng hơn" *trừ khi* các đặc trưng đã cùng thang đo — điều này *đã đúng* sau chuẩn hoá (mọi cột có std=1), nên thực ra **so trọng số sau chuẩn hoá mới là cách so sánh công bằng** — còn so trọng số trên dữ liệu gốc $w_o$ mới là thứ **không nên** dùng để so sánh tầm quan trọng, vì đơn vị gốc của các cột khác nhau (mét so với VNĐ so với số phòng) khiến độ lớn $w_o$ không phản ánh đúng mức ảnh hưởng thực.

---

## 5. Bài tập tự làm (kèm gợi ý — không giải sẵn)

### Bài tập 1. Quay về hệ toạ độ gốc

> Từ `w_gd` và `b_gd` học trên dữ liệu đã chuẩn hoá, hãy suy ra trọng số tương ứng trên dữ liệu gốc, rồi so với `w_that` và `b_that`.
>
> **Cổng kiểm:** Trọng số suy ra được gần $[2,0;\ -0,5;\ 0,01]$ và hệ số tự do gần 3,0. Sai số vài phần nghìn là bình thường, vì dữ liệu có nhiễu.

**Gợi ý cách tiếp cận:** Dùng đúng hai công thức đã suy ra ở mục 4 (`w_o = w_c/σ` và `b_o = b_c - Σ mean·w_c/σ`). Trong code, đó là hai dòng NumPy — không cần vòng lặp, vì phép chia element-wise `w_gd / sd` và tổng `(tb * w_gd / sd).sum()` đã được vectorize sẵn.

### Bài tập 2. Vẽ đường cong mất mát

> Vẽ `lich_su` theo trục dọc log cho bốn tốc độ học 0,5, 0,1, 0,01, 0,001 trên cùng một hình.
>
> **Cổng kiểm:** Bốn đường cùng đi xuống một mức, chỉ khác độ dốc. Hình này chính là thứ buổi 13 sẽ dạy bạn đọc để chẩn đoán bệnh của một lần huấn luyện.

**Gợi ý cách tiếp cận:** Gọi `giam_do_doc(Xc, y, lr)` bốn lần với bốn `lr` khác nhau, mỗi lần lấy `lich_su` trả về, rồi vẽ bằng `plt.plot(lich_su)` với `plt.yscale('log')` (trục dọc log giúp nhìn rõ cả vùng loss lớn ban đầu lẫn vùng loss nhỏ gần cuối trên cùng một hình — nếu để trục thường, phần "gần đáy" sẽ bị nén phẳng khó nhìn).

### Bài tập 3. Tìm ngưỡng phân kỳ

> Với dữ liệu đã chuẩn hoá, tìm tốc độ học lớn nhất mà thuật toán còn hội tụ. Tăng dần từ 0,5 lên.
>
> **Cổng kiểm:** Một con số kèm bằng chứng: ngay dưới nó thì hội tụ, ngay trên nó thì mất mát tăng. Sau đó tra công thức lý thuyết $\eta < 2/\lambda_{\max}$ với $\lambda_{\max}$ là trị riêng lớn nhất của $\frac{2}{n}X^\top X$, rồi so với con số bạn đo được.

**Gợi ý cách tiếp cận:** Đây chính là bài tập thực nghiệm cho lý thuyết đã giải thích chi tiết ở mục 2.3. Về mặt code, trị riêng lớn nhất tính được bằng `np.linalg.eigvalsh((2/n) * Xc.T @ Xc).max()` (dùng `eigvalsh` vì ma trận đối xứng, ổn định và nhanh hơn `eig` tổng quát). Về mặt thực nghiệm, dò nhị phân (binary search) trên `lr` sẽ nhanh hơn tăng tuyến tính từ 0,5.

### Bài tập 4. Điều chuẩn L2

> Thêm số hạng phạt $\lambda\|w\|^2$ vào hàm mất mát, sửa lại đạo hàm, rồi so với `Ridge` của scikit-learn.
>
> **Cổng kiểm:** Khớp `Ridge(alpha=...)` tới sai số máy. Chú ý: quy ước hệ số của scikit-learn khác quy ước trong sách một hằng số, và tìm ra đúng hằng số ấy chính là nội dung bài tập.

**Gợi ý cách tiếp cận (mở rộng thêm lý thuyết vì bài gốc chỉ nêu đề):** Với hàm mất mát mới $L_{ridge}(w,b) = \frac{1}{n}\sum_i r_i^2 + \lambda\|w\|^2$ (chỉ phạt `w`, không phạt `b` — quy ước phổ biến, vì phạt `b` sẽ ép mô hình lệch về 0 một cách không cần thiết), đạo hàm theo `w` có thêm một số hạng:

$$\frac{\partial L_{ridge}}{\partial w} = \frac{2}{n}X^\top r + 2\lambda w$$

(đạo hàm của $\lambda\|w\|^2 = \lambda \sum_j w_j^2$ theo từng $w_j$ là $2\lambda w_j$, viết gọn dạng vector là $2\lambda w$). Đạo hàm theo `b` không đổi vì số hạng phạt không chứa `b`.

**Về "quy ước khác nhau một hằng số":** `Ridge` của scikit-learn tối thiểu hoá $\|y - Xw\|^2 + \alpha\|w\|^2$ (tổng bình phương, **không chia cho `n`**), trong khi công thức ở đây dùng **trung bình** bình phương cộng $\lambda\|w\|^2$. Vì vậy hai công thức chỉ tương đương khi $\lambda = \alpha/n$ (hoặc $\alpha = \lambda \cdot n$, tuỳ chiều quy đổi) — đây chính là hằng số bạn cần tìm ra và xác nhận bằng thực nghiệm đối chiếu, giống hệt tinh thần "đối chiếu để tự kiểm" đã dùng ở Bài tập mẫu.

### Bài tập 5. Cột hằng số

> Thêm vào `X` một cột chỉ toàn số 5, rồi chạy lại chuẩn hoá.
>
> **Cổng kiểm:** Không có `nan` nào xuất hiện, và nghiệm gần như không đổi. Nếu bạn quên dòng `sd[sd == 0] = 1.0` thì mọi thứ thành `nan`, và không có thông báo lỗi nào. Hãy làm cho nó hỏng một lần để nhớ.

**Gợi ý cách tiếp cận:** Dùng `np.column_stack([X, np.full(n, 5.0)])` hoặc `np.hstack` để thêm cột. Bài tập này **cố ý** khuyến khích bạn tạm thời xoá dòng `sd[sd == 0] = 1.0` để tận mắt chứng kiến toàn bộ `Xc` biến thành `nan` mà không có `Traceback` hay `Exception` nào cả — đây là một trong những "bẫy im lặng" (silent bug) nguy hiểm nhất của NumPy: chia cho 0 trên mảng số thực không raise lỗi như Python thuần (`1/0` sẽ báo `ZeroDivisionError`), mà chỉ âm thầm trả về `inf` hoặc `nan`. Thói quen kiểm tra `np.isnan(...).any()` sau các bước tiền xử lý là một lưới an toàn hữu ích tương tự cách `mot_buoc`/`giam_do_doc` đã dùng `np.isfinite`.

---

## 6. Bảng thuật ngữ đối chiếu (bổ sung mới)

| Tiếng Việt trong bài | Thuật ngữ tiếng Anh chuẩn | Ghi chú |
|---|---|---|
| Giảm độ dốc | Gradient Descent | |
| Hồi quy tuyến tính | Linear Regression | |
| Mất mát bình phương trung bình | Mean Squared Error (MSE) | |
| Phần dư | Residual | $r_i = \hat y_i - y_i$ |
| Tốc độ học | Learning rate ($\eta$ / `lr`) | |
| Phân kỳ | Divergence | loss bay ra `inf`/`nan` |
| Hội tụ | Convergence | |
| Chuẩn hoá đặc trưng | Feature standardization / z-score normalization | khác với "normalization" kiểu min-max |
| Rò rỉ dữ liệu | Data leakage | chủ đề chính của Buổi 07 |
| Nghiệm đóng / Phương trình chuẩn tắc | Closed-form solution / Normal Equation | |
| Suy biến / gần suy biến | Singular / near-singular (matrix) | |
| Trị riêng | Eigenvalue | |
| Số điều kiện | Condition number | $\lambda_{\max}/\lambda_{\min}$ |
| Sàn lý thuyết / sai số không thể triệt tiêu | Irreducible error / Bayes error | |
| Điều chuẩn L2 | L2 Regularization (Ridge) | |
| Vectorization | Vectorization | tính bằng ma trận thay vì vòng lặp |

---

## 7. Tổng kết — bản đồ tư duy của cả buổi (bổ sung mới)

**Chuỗi nhân quả xuyên suốt buổi học, tóm tắt lại một lần:**

1. Mô hình tuyến tính (mục 2.1) cần một hàm đo "sai bao nhiêu" → MSE (mục 2.1).
2. Muốn giảm MSE, cần biết hướng nào làm nó giảm nhanh nhất → đạo hàm/gradient (mục 2.2).
3. Đi theo hướng ngược gradient, lặp lại nhiều lần → giảm độ dốc (mục 2.2), nhưng **độ dài mỗi bước** (`η`) quyết định thành bại (mục 2.3).
4. Dữ liệu nhiều thang đo khác nhau khiến "thung lũng" bị méo dẹt → không tồn tại `η` nào vừa nhanh vừa an toàn (mục 2.3).
5. Chuẩn hoá đặc trưng làm thung lũng tròn lại → mở rộng vùng `η` an toàn rất nhiều (mục 2.4) — **nhưng phải cẩn thận không rò rỉ thống kê tập kiểm tra vào bước này**.
6. Có một cách giải **không cần lặp** — nghiệm đóng (mục 2.5) — dùng làm trọng tài đối chiếu kết quả giảm độ dốc (mục 4).
7. Biết trước "sàn lý thuyết" (do nhiễu trong dữ liệu, mục 3.4) giúp biết khi nào nên dừng tối ưu.

**Checklist nên nhớ khi vào phòng thi:**
- [ ] Loss ra `nan`/`inf` → nghi ngờ đầu tiên: `η` quá lớn, hoặc dữ liệu chưa chuẩn hoá.
- [ ] Loss giảm quá chậm → nghi ngờ đầu tiên: `η` quá nhỏ, không phải "mô hình không phù hợp".
- [ ] Luôn `print(shape)` sau mỗi bước biến đổi dữ liệu quan trọng.
- [ ] Luôn có một "phanh khẩn cấp" (`np.isfinite` check) trong vòng lặp huấn luyện.
- [ ] Chuẩn hoá bằng thống kê của **tập huấn luyện**, áp lại nguyên xi cho tập kiểm tra.
- [ ] Trước khi tinh chỉnh thêm, tự hỏi: sàn lý thuyết ở đây là bao nhiêu?
- [ ] Có nghi ngờ về code tự cài → viết một bản đối chiếu độc lập (nghiệm đóng, hoặc thư viện) để tự kiểm.

---

## 8. Chuẩn bị cho buổi sau

> **Nguyên văn:** Buổi 05 chuyển sang phân loại với hồi quy logistic. Mang theo câu hỏi: vì sao không dùng luôn hàm mất mát bình phương cho bài toán phân loại, dù nó vẫn tính được?

**Gợi ý suy nghĩ trước (không phải lời giải đầy đủ — để dành cho buổi 05):** Với bài toán phân loại, nhãn `y` chỉ nhận giá trị rời rạc (ví dụ 0 hoặc 1), còn dự đoán thường được ép về khoảng $[0,1]$ (xác suất) qua hàm sigmoid. Khi bạn thử dùng MSE cho loss trong tình huống này, hàm mất mát theo tham số không còn giữ được tính chất "chỉ có một đáy duy nhất, mượt đều" như hồi quy tuyến tính nữa — nó có thể trở nên **không lồi** (non-convex), khiến giảm độ dốc dễ mắc kẹt ở các điểm không phải nghiệm tốt nhất, và gradient ở những vùng dự đoán "quá tự tin nhưng sai" lại rất **nhỏ** (gần như phẳng) — ngược với điều bạn mong muốn (bạn muốn gradient lớn để mô hình sửa sai thật nhanh khi nó đang rất sai). Đây chính là lý do buổi 05 sẽ giới thiệu một hàm mất mát khác dành riêng cho phân loại. Hãy giữ câu hỏi này trong đầu và so sánh khi học buổi tiếp theo.

---

*Tài liệu này tổng hợp và mở rộng từ bài giảng gốc "Buổi 04 — Hồi quy tuyến tính và giảm độ dốc" của TS. Đỗ Phúc Hảo, không thay đổi bất kỳ nội dung, công thức, hay số liệu gốc nào — chỉ bổ sung phần giải thích trực giác, ẩn dụ, và chú thích code.*
