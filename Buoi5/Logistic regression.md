# Buổi 05 — Hồi quy logistic và phân loại nhị phân
### Ghi chú mở rộng — Luyện Olympic Trí tuệ nhân tạo
*Dựa trên bài giảng gốc của TS. Đỗ Phúc Hảo, ngày 25/8/2026*
*Phần: Học máy cổ điển — Phục vụ: sơ loại*

---

## 0. Vì sao buổi này tồn tại — sợi dây nối với buổi 04

🧠 **Ẩn dụ mở đầu**: Ở buổi 04, bạn đã học cách vẽ một đường thẳng "tốt nhất" xuyên qua một đám mây điểm số — đó là hồi quy tuyến tính, output là một số thực bất kỳ (giá nhà, điểm thi, nhiệt độ...). Nhưng nếu câu hỏi không phải "bao nhiêu" mà là "có hay không" — email này có phải spam không, khối u này lành hay ác, khách hàng này có rời bỏ dịch vụ không — thì output cần là **có/không**, hoặc chính xác hơn, một **xác suất** nằm gọn trong [0, 1]. Bài giảng gốc đặt câu hỏi cực kỳ thẳng thắn: *"vì sao không dùng luôn hàm mất mát bình phương cho bài toán phân loại, dù nó vẫn tính được?"* — và cam kết trả lời bằng số đo cụ thể, không phải lý thuyết suông. Đây chính là tinh thần xuyên suốt của buổi 05.

---

## 1. Mục tiêu (nội dung gốc)

> Trả lời được câu hỏi mà buổi 04 để lại: vì sao không dùng luôn hàm mất mát bình phương cho bài toán phân loại, dù nó vẫn tính được?
> Và trả lời bằng số đo, không phải bằng câu chữ.

🔬 **Vì sao cách tiếp cận này quan trọng**: Đây là một nguyên tắc tư duy khoa học then chốt cho kỳ thi Olympic — không chấp nhận câu trả lời kiểu "vì sách nói vậy" hay "vì đó là quy ước". Thay vào đó, bài giảng sẽ **đo trực tiếp** tốc độ gradient, số vòng lặp cần thiết, và cho bạn thấy con số chênh lệch cụ thể (179 lần, 95 lần — sẽ nói kỹ ở mục 2.4). Khi đi thi, nếu gặp câu hỏi "tại sao hàm X tốt hơn hàm Y", câu trả lời điểm cao nhất luôn là câu trả lời có thể chứng minh bằng phép tính, không phải trực giác.

---

## 2. Tóm tắt kiến thức

### 2.1 Từ điểm số sang xác suất (nội dung gốc)

> Mô hình tuyến tính cho ra một số thực bất kỳ. Phân loại cần một xác suất trong [0, 1]. Hàm sigmoid làm cầu nối:
>
> $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \hat{p} = \sigma(w \cdot x + b) \tag{1}$$
>
> ```python
> def sigmoid(z):
>     return 1.0 / (1.0 + np.exp(-z))
> ```
>
> **np.exp tràn khi z rất âm**
> Với z = −800, np.exp(800) vượt khỏi tầm số thực và bạn nhận inf kèm một cảnh báo, rồi inf lan ra khắp nơi. Bản an toàn tách hai nhánh theo dấu của z, hoặc gọi scipy.special.expit. Trong bài tập của buổi này thì z chưa đủ lớn để tràn, nhưng hãy biết là chỗ ấy có mìn.

🧠 **Ẩn dụ**: Hãy tưởng tượng sigmoid như một cái **"máy nén"**. Bạn đưa vào nó bất kỳ con số nào — từ −1 triệu đến +1 triệu — và nó luôn nhả ra một con số nằm gọn trong khoảng (0, 1). Nó giống như một cái phễu đặc biệt: đầu vào rộng mênh mông, đầu ra luôn bị ép vào một dải hẹp. Về mặt hình học, đồ thị của sigmoid là một chữ **S** mềm mại: gần 0 khi z rất âm (mô hình "rất tự tin" là lớp 0), gần 1 khi z rất dương (rất tự tin là lớp 1), và bằng đúng 0,5 khi z = 0 (mô hình "phân vân 50-50").

🔬 **Giải thích kỹ thuật sâu**:

- **Vì sao chọn đúng hàm này mà không phải hàm nào khác ép vào [0,1]?** Có vô số hàm có thể "ép" một số thực vào khoảng (0,1) — ví dụ hàm phân phối chuẩn tích lũy (probit), hoặc một hàm tanh dịch chuyển. Nhưng sigmoid có một tính chất đại số cực kỳ đẹp mà buổi 05 sẽ khai thác ở mục 2.3: đạo hàm của nó là $\sigma'(z) = \sigma(z)(1-\sigma(z))$, tự viết được bằng chính giá trị của nó, không cần tính lại từ đầu. Đây là lý do sigmoid được chọn làm "cầu nối" chuẩn cho hồi quy logistic thay vì các lựa chọn khác.

- **Liên hệ log-odds**: Nghịch đảo của sigmoid là hàm **logit**: $z = \ln\left(\frac{\hat{p}}{1-\hat{p}}\right)$. Tỷ số $\frac{\hat{p}}{1-\hat{p}}$ gọi là "odds" (tỷ lệ cược) — một khái niệm quen thuộc trong cá cược thể thao ("tỷ lệ cược 3:1"). Nói cách khác, mô hình tuyến tính $w \cdot x + b$ không dự đoán trực tiếp xác suất, mà dự đoán **log-odds**. Đây là lý do tên gọi "logistic regression" ra đời — nó là hồi quy tuyến tính trên thang log-odds, sau đó biến đổi ngược lại thành xác suất.

- **Bug tràn số (`overflow`) — mìn thật sự nguy hiểm trong thi cử**: Đây không phải chi tiết vụn vặt. Nếu bạn code sigmoid ngây thơ như trên và huấn luyện mô hình lâu, trọng số $w$ có thể tăng lớn dần, khiến $z = w \cdot x + b$ đạt giá trị rất âm cho một vài mẫu ngoại lai. `np.exp(800)` sẽ trả về `inf`, và `1/(1+inf)` tuy vẫn ra `0.0` đúng về mặt toán học — **nhưng** cảnh báo `RuntimeWarning: overflow encountered in exp` sẽ xuất hiện, và nếu $z$ dương rất lớn (không phải âm), bạn sẽ gặp trường hợp khác: `np.exp(-z)` với z rất âm... Nói chính xác lại theo công thức: khi **z rất âm**, `-z` rất dương, nên `np.exp(-z)` mới là chỗ tràn, chứ không phải `np.exp(z)`. Bản an toàn tách hai nhánh:

  ```python
  def sigmoid_an_toan(z):
      out = np.empty_like(z, dtype=float)
      duong = z >= 0
      out[duong] = 1.0 / (1.0 + np.exp(-z[duong]))
      am = ~duong
      exp_z = np.exp(z[am])
      out[am] = exp_z / (1.0 + exp_z)
      return out
  ```
  Ý tưởng: khi z ≥ 0, dùng công thức gốc (an toàn vì `exp(-z)` có `-z ≤ 0`). Khi z < 0, dùng công thức biến đổi đại số tương đương $\sigma(z) = \frac{e^z}{1+e^z}$ (an toàn vì `exp(z)` có `z < 0`). Đây chính là kỹ thuật "log-sum-exp trick" thu nhỏ, một mẫu hình bạn sẽ gặp lại nhiều lần trong ML (softmax ổn định số học, cross-entropy ổn định số học...).

- 📌 **Ví dụ thực tế**: Trong các thư viện production như PyTorch hay scikit-learn, hàm `sigmoid`/`expit` luôn được cài theo bản an toàn này (hoặc dùng các hàm built-in đã tối ưu ở tầng C). Khi tự cài tay để thi, nếu đề bài không giới hạn phạm vi z, luôn nên nhớ chỗ "có mìn" này.

### 2.2 Hàm mất mát: cross entropy (nội dung gốc)

> $$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] \tag{2}$$
>
> Với một mẫu, chỉ một trong hai số hạng sống sót: nếu y = 1 thì L = −log p̂, nếu y = 0 thì L = −log(1−p̂).
>
> **Tính tay, đúng dạng câu 38 đề sơ loại 2025**
> Đề cho: nhãn đúng dạng one-hot là [0, 1], mô hình dự đoán [0,3; 0,7], dùng logarit tự nhiên. Tính cross entropy.
>
> Vị trí thứ hai là nhãn đúng, nên chỉ số hạng thứ hai sống:
> $$L = -(0 \cdot \ln 0{,}3 + 1 \cdot \ln 0{,}7) = -\ln 0{,}7 = 0{,}35667 \tag{3}$$
>
> | Cách tính | Ra số | Nhận xét |
> |---|---|---|
> | −ln 0,7 | 0,35667 | đúng |
> | −ln 0,3 | 1,20397 | lấy nhầm vị trí, tưởng nhãn đúng là vị trí thứ nhất |
> | −log₂0,7 | 0,51457 | dùng nhầm cơ số 2 thay vì cơ số e |
>
> **Ba con số ấy đều nằm trong bốn phương án của đề**
> Đề trắc nghiệm không đưa phương án nhiễu một cách ngẫu nhiên. Chúng là kết quả của ba lỗi cụ thể mà người ra đề biết là thí sinh hay mắc. Khi ôn dạng tính tay, hãy tự hỏi "ba phương án kia sinh ra từ lỗi nào", vì đó chính là ba lỗi bạn cần tránh.

🧠 **Ẩn dụ**: Cross entropy đo **"mức độ bất ngờ"** của mô hình khi nhìn thấy nhãn thật. Nếu mô hình đoán "tôi chắc 99% đây là ảnh mèo" và đúng thật là mèo, nó "không bất ngờ" — mất mát rất nhỏ. Nhưng nếu mô hình đoán "tôi chắc 99% đây là mèo" mà thật ra là chó, nó "cực kỳ bất ngờ" — mất mát tiến gần tới vô cực. Hàm log ở đây chính là công cụ toán học biến "xác suất nhỏ dần" thành "hình phạt lớn dần" một cách rất dốc — càng tự tin sai, càng bị phạt nặng theo cấp số mũ ngược.

🔬 **Giải thích kỹ thuật sâu**:

- **Tại sao "chỉ một số hạng sống sót"?** Vì $y_i \in \{0, 1\}$ (nhãn nhị phân), nên trong biểu thức $y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)$: nếu $y_i = 1$ thì hệ số của số hạng thứ hai là $(1-1)=0$, số hạng đó biến mất; nếu $y_i = 0$ thì số hạng đầu biến mất. Đây là một dạng viết gọn (compact form) của hai công thức riêng biệt cho hai lớp, giúp viết code vector hóa không cần `if/else`.

- **Ba lỗi trong bảng — phân tích sâu hơn để tránh bẫy thi**:
  1. **−ln 0,3 = 1,20397 — lỗi "lấy nhầm vị trí"**: Đây là lỗi về *chỉ số* (indexing), không phải lỗi công thức. Khi đề cho nhãn one-hot [0, 1] nghĩa là lớp đúng nằm ở vị trí **thứ hai** (index 1 nếu đếm từ 0), thí sinh vội vàng lấy phần tử **thứ nhất** của vector dự đoán (0,3) thay vì phần tử thứ hai (0,7). Đây là lỗi cực kỳ phổ biến khi làm bài dưới áp lực thời gian.
  2. **−log₂0,7 = 0,51457 — lỗi "nhầm cơ số"**: Nhiều tài liệu về entropy thông tin (information theory) dùng log cơ số 2, đơn vị là "bit". Nhưng ML hầu như luôn dùng log tự nhiên (cơ số e), đơn vị là "nat". Công thức chuyển đổi: $\log_2 x = \frac{\ln x}{\ln 2}$, nên $-\log_2 0{,}7 = \frac{-\ln 0{,}7}{\ln 2} = \frac{0{,}35667}{0{,}69315} \approx 0{,}51457$ — khớp chính xác với bảng.
  3. **Bẫy tiềm ẩn thứ tư (không có trong bảng nhưng nên biết)**: nhầm dấu, ví dụ quên dấu trừ ở đầu công thức, ra kết quả âm (−0,35667). Đề thi trắc nghiệm nếu có 4 đáp án, khả năng cao đáp án thứ tư chính là dạng lỗi dấu này.

- **Vì sao dùng `eps = 1e-12` trong code (mục 3.2)?** Nếu $\hat p_i$ chạm đúng 0 hoặc đúng 1 (do làm tròn số học hoặc do mô hình quá tự tin), `np.log(0)` trả về `-inf`. Một giá trị `-inf` duy nhất trong phép lấy trung bình (`mean()`) sẽ làm **toàn bộ** kết quả trở thành `-inf` hoặc `nan` — một minh họa khác của chủ đề "lỗi âm thầm" mà bạn đã ghi chú từ các buổi trước: một giá trị hỏng có thể "đầu độc" toàn bộ phép tính tổng hợp mà không báo lỗi rõ ràng.

### 2.3 Đạo hàm, và một điều đẹp đẽ (nội dung gốc)

> Đạo hàm của cross entropy theo trọng số hoá ra rất gọn:
> $$\frac{\partial L}{\partial w} = \frac{1}{n}X^{\top}(\hat p - y), \quad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_i(\hat p_i - y_i) \tag{4}$$
>
> Giống hệt hồi quy tuyến tính, chỉ khác chỗ ŷ được thay bằng p̂. Đạo hàm của sigmoid tự triệt tiêu trong phép rút gọn, và mục sau cho thấy đó không phải may mắn mà là lý do người ta chọn hàm mất mát này.

🧠 **Ẩn dụ**: Hãy tưởng tượng bạn có hai công cụ trong hộp đồ nghề: (1) hàm sigmoid để "nén" điểm số thành xác suất, và (2) hàm cross entropy để "đo lỗi" dựa trên xác suất đó. Khi ghép hai công cụ này lại và tính đạo hàm theo quy tắc dây chuyền (chain rule), một điều kỳ diệu xảy ra: các phần rắc rối (đạo hàm của log, đạo hàm của sigmoid) **triệt tiêu lẫn nhau** như hai mảnh ghép khớp hoàn hảo, để lại một công thức đơn giản đến bất ngờ — công thức y hệt của hồi quy tuyến tính bình thường, chỉ đổi tên biến.

🔬 **Giải thích kỹ thuật sâu — chứng minh vì sao "triệt tiêu"**:

Xét với một mẫu, $z = w \cdot x + b$, $\hat p = \sigma(z)$, $L = -[y \log \hat p + (1-y)\log(1-\hat p)]$.

Bước 1 — đạo hàm của L theo p̂:
$$\frac{\partial L}{\partial \hat p} = -\frac{y}{\hat p} + \frac{1-y}{1-\hat p}$$

Bước 2 — đạo hàm của sigmoid theo z (tính chất đặc biệt đã nhắc ở 2.1):
$$\frac{\partial \hat p}{\partial z} = \hat p(1-\hat p)$$

Bước 3 — nhân theo chain rule:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat p}\cdot\frac{\partial \hat p}{\partial z} = \left(-\frac{y}{\hat p} + \frac{1-y}{1-\hat p}\right)\hat p(1-\hat p)$$

Khai triển: số hạng đầu $-\frac{y}{\hat p}\cdot \hat p(1-\hat p) = -y(1-\hat p)$; số hạng sau $\frac{1-y}{1-\hat p}\cdot\hat p(1-\hat p) = (1-y)\hat p$. Cộng lại:

$$\frac{\partial L}{\partial z} = -y(1-\hat p) + (1-y)\hat p = -y + y\hat p + \hat p - y\hat p = \hat p - y$$

Đây chính là "điều đẹp đẽ": mẫu số $\hat p$ và $(1-\hat p)$ ở bước 1 **bị chính đạo hàm sigmoid ở bước 2 triệt tiêu hoàn toàn**, để lại kết quả gọn gàng $\hat p - y$. Sau đó $\frac{\partial z}{\partial w} = x$, nên $\frac{\partial L}{\partial w} = (\hat p - y)x$, vector hóa trên toàn bộ n mẫu cho ra công thức (4).

- **Đây không phải trùng hợp — đây là thiết kế có chủ đích**: Cặp (sigmoid, cross entropy) được gọi là một cặp "liên hợp" (conjugate pair) trong họ phân phối mũ (exponential family). Bất kỳ khi nào bạn ghép đúng hàm kích hoạt với đúng hàm mất mát tương ứng trong họ này (sigmoid-cross entropy cho nhị phân, softmax-cross entropy cho đa lớp, identity-MSE cho hồi quy tuyến tính Gauss), đạo hàm luôn rút gọn về dạng $(\hat y - y)$. Đây là lý do sâu xa tại sao các cặp này được chọn làm chuẩn công nghiệp — không phải vì "quen dùng" mà vì tính toán học tối ưu.

### 2.4 Vì sao không dùng mất mát bình phương (nội dung gốc)

> Nếu dùng $L = \frac{1}{n}\sum(\hat p_i - y_i)^2$ thì đạo hàm mọc thêm một thừa số:
> $$\frac{\partial L}{\partial w} = \frac{2}{n}X^{\top}\left[(\hat p - y)\odot \hat p \odot (1-\hat p)\right] \tag{5}$$
>
> Thừa số p̂(1−p̂) chính là đạo hàm của sigmoid. Nó bằng 0,25 ở giữa và tiến về 0 ở hai đầu. Nghĩa là khi mô hình dự đoán sai một cách rất tự tin, gradient gần như biến mất, và thuật toán gần như đứng yên đúng lúc nó cần sửa nhiều nhất.
>
> **Đo thật.** Khởi tạo cố ý sai và tự tin, w0 = [−8; 8; 0; 0]:
>
> | | cross entropy | bình phương |
> |---|---|---|
> | độ lớn gradient ở bước đầu | 0,51387 | 0,0028765 |
> | số vòng để đạt độ chính xác 0,75 | 19 | 1813 |
>
> **Hai con số, một kết luận**
> Gradient nhỏ hơn 179 lần, và hậu quả là chậm hơn 95 lần. Cả hai cuối cùng đều tới cùng một chỗ, nên đây không phải chuyện đúng sai mà là chuyện đắt rẻ. Trong sáu tiếng thi, chậm 95 lần chính là không về đích.
>
> Khi khởi tạo bằng 0, tức không tự tin về phía nào, khoảng cách ấy biến mất: cả hai đều đạt độ chính xác quanh 0,80 sau 200 vòng. Cái bẫy chỉ lộ ra khi mô hình đang sai nặng, và đó đúng là lúc bạn cần nó chạy nhanh nhất.

🧠 **Ẩn dụ trung tâm của cả buổi học — "phanh xe bị kẹt đúng lúc cần đạp ga"**: Hãy tưởng tượng bạn đang lái xe và đi lạc đường hoàn toàn — cần quay đầu gấp. Với cross entropy, hệ thống lái nhận ra "tôi sai nặng rồi!" và phản ứng cực mạnh, quay đầu ngay lập tức (gradient lớn khi sai nhiều). Nhưng với mất mát bình phương trên sigmoid, hệ thống lái lại có một lỗi thiết kế kỳ quặc: **càng đi lạc xa, phanh càng bị kẹt cứng, xe gần như không nhúc nhích** — chính lúc cần sửa nhiều nhất thì hệ thống lại phản ứng yếu nhất. Đây gọi là hiện tượng **"vanishing gradient" (gradient biến mất)**, một trong những vấn đề kinh điển và nguy hiểm nhất trong huấn luyện mạng nơ-ron nói chung — buổi 05 cho bạn thấy nó xuất hiện ngay ở mô hình đơn giản nhất.

🔬 **Giải thích kỹ thuật sâu**:

- **Cơ chế toán học của cái bẫy**: So sánh công thức (4) và (5): cross entropy cho $\frac{\partial L}{\partial w} \propto (\hat p - y)$; mất mát bình phương cho $\frac{\partial L}{\partial w} \propto (\hat p - y)\cdot \hat p(1-\hat p)$. Thừa số phụ $\hat p(1-\hat p)$ chính là đạo hàm sigmoid $\sigma'(z)$, một hàm hình chuông (giống Gauss lộn ngược thu nhỏ) đạt đỉnh **0,25 tại z=0** và tiệm cận **0 khi |z| lớn**. Khi mô hình dự đoán sai **và tự tin** (nghĩa là $\hat p$ gần 0 hoặc gần 1 nhưng theo hướng sai), $|z|$ lớn, nên $\sigma'(z) \to 0$, khiến toàn bộ gradient bị "nhân với gần như 0" — dù phần $(\hat p - y)$ (mức độ sai) vẫn lớn.

- **Diễn giải con số 179 lần và 95 lần**: $0{,}51387 / 0{,}0028765 \approx 178{,}7$ — đây là tỷ số độ lớn gradient ở bước đầu tiên. $1813 / 19 \approx 95{,}4$ — đây là tỷ số số vòng lặp cần thiết để đạt cùng một mức độ chính xác. **Điểm tinh tế cần nắm**: gradient nhỏ hơn 179 lần nhưng số vòng chỉ chậm hơn 95 lần chứ không phải 179 lần — vì trong quá trình huấn luyện, khi mô hình dần dần "bớt tự tin sai", $\sigma'(z)$ dần lớn lên (thoát khỏi vùng bão hòa), nên gradient bình phương-loss cũng dần được "phục hồi" một phần. Tỷ lệ chậm ở bước đầu (khi bị kẹt nặng nhất) luôn tệ hơn tỷ lệ chậm trung bình trên cả hành trình.

- **Vì sao trường hợp khởi tạo bằng 0 lại "xóa bỏ" khoảng cách?** Khi $w=0, b=0$ (điểm khởi tạo trung lập, không thiên vị lớp nào), $z = 0$ với mọi mẫu, nên $\hat p = \sigma(0) = 0{,}5$ với mọi mẫu — đúng tại **đỉnh** của hàm $\sigma'(z)$, nơi $\sigma'(0) = 0{,}25$ đạt giá trị lớn nhất, không hề nhỏ. Cái bẫy vanishing gradient chỉ kích hoạt khi mô hình đã trôi ra xa vùng z=0 **và sai hướng**. Đây là bài học thực tiễn cực kỳ quan trọng cho khởi tạo trọng số trong mạng nơ-ron nói chung: khởi tạo trọng số quá lớn hoặc quá "tự tin" ngay từ đầu có thể khiến mô hình bị "mắc kẹt" trong vùng bão hòa gradient ngay từ những bước đầu tiên.

- 📌 **Liên hệ thực tế lớn hơn**: Đây chính là lý do lịch sử tại sao các hàm kích hoạt như sigmoid/tanh dần bị thay thế bởi ReLU trong các mạng sâu (deep network) hiện đại — ReLU không có vùng bão hòa hai đầu như sigmoid nên tránh được vanishing gradient ở tầng sâu. Bài học buổi 05, dù chỉ là mô hình logistic regression một tầng, chính là hạt giống của toàn bộ câu chuyện "vanishing/exploding gradient" sẽ ám ảnh deep learning về sau.

- **Trong sáu tiếng thi, chậm 95 lần chính là không về đích**: Đây là câu chốt mang tính thực chiến. Nếu mã nguồn bạn viết ra chọn nhầm hàm mất mát (dùng MSE thay vì cross entropy cho bài toán phân loại), mô hình vẫn "chạy được", vẫn "hội tụ được về lâu về dài" — nhưng trong giới hạn thời gian thi cử, số vòng lặp bị giới hạn (ví dụ do giới hạn thời gian chạy hoặc giới hạn max_iter), mô hình của bạn có thể **chưa kịp học được gì đáng kể** trong khi đối thủ dùng đúng hàm mất mát đã hội tụ từ lâu.

### 2.5 Ngưỡng quyết định (nội dung gốc)

> $$\hat y = \begin{cases} 1 & \text{nếu } \hat p \ge \tau \\ 0 & \text{nếu ngược lại}\end{cases} \tag{6}$$
>
> Mặc định τ = 0,5, nhưng đó chỉ là mặc định chứ không phải chân lý. Khi hai lớp lệch nhau, hoặc khi hai loại lỗi có giá khác nhau, ngưỡng tối ưu cho macro-F1 thường không phải 0,5. Buổi 06 sẽ dò ngưỡng một cách có kỷ luật.

🧠 **Ẩn dụ**: Mô hình logistic regression giống như một bác sĩ đưa ra **phần trăm khả năng mắc bệnh** (ví dụ "70% khả năng bạn bị bệnh X"), chứ không tự động đưa ra kết luận "có bệnh" hay "không bệnh". Việc chuyển từ con số phần trăm sang quyết định nhị phân (nhập viện hay không) là một **bước riêng biệt**, và ngưỡng cắt (bao nhiêu % thì coi là "có bệnh") phụ thuộc vào bối cảnh: nếu bỏ sót bệnh nhân thật sự nguy hiểm hơn nhiều so với chẩn đoán nhầm người khỏe mạnh, bác sĩ nên hạ ngưỡng xuống thấp hơn 50% để không bỏ sót ca nào.

🔬 **Giải thích kỹ thuật sâu**:

- **Vì sao τ=0,5 chỉ là mặc định**: Ngưỡng 0,5 tối ưu **chỉ khi** hai điều kiện đồng thời đúng: (a) hai lớp cân bằng về số lượng trong tập dữ liệu, và (b) chi phí của hai loại lỗi (false positive và false negative) là ngang nhau. Trong thực tế — ví dụ phát hiện gian lận thẻ tín dụng (lớp gian lận cực hiếm, nhưng bỏ sót một vụ gian lận tốn kém hơn nhiều so với báo động nhầm) — ngưỡng tối ưu thường thấp hơn 0,5 rất nhiều.
- **Vì sao chọn τ theo macro-F1 chứ không theo accuracy**: Sẽ được giải thích chi tiết ở buổi 06, nhưng ý tưởng cốt lõi: accuracy (độ chính xác tổng thể) có thể bị "lừa" khi hai lớp mất cân bằng (ví dụ 95% mẫu thuộc một lớp — chính là câu hỏi buổi 05 để lại cho buổi 06). Macro-F1 tính trung bình không trọng số của F1 trên từng lớp, buộc mô hình phải làm tốt trên **cả hai lớp**, không chỉ lớp đông hơn.
- **Cảnh báo rò rỉ dữ liệu (data leakage)**: Bài tập 5 ở mục 5 đặt câu hỏi rất tinh tế — "nếu bạn chọn τ bằng cách nhìn chính tập kiểm tra, bạn đã phạm lỗi gì?" Đáp án (sẽ học kỹ ở buổi 07): đây là lỗi **rò rỉ thông tin từ tập kiểm tra vào quy trình chọn siêu tham số**. Ngưỡng τ là một dạng "siêu tham số" (hyperparameter), và mọi siêu tham số phải được chọn dựa trên **tập validation**, không phải tập test — nếu không, con số đánh giá cuối cùng trên tập test không còn phản ánh trung thực khả năng tổng quát hóa của mô hình nữa, vì bạn đã "nhìn trộm" đáp án khi tinh chỉnh.

### 2.6 Điều chuẩn (nội dung gốc)

> | Thêm vào mất mát | Tác dụng |
> |---|---|
> | L2 λ‖w‖₂² | kéo mọi trọng số về gần 0, không về hẳn 0 |
> | L1 λ‖w‖₁ | đẩy hẳn nhiều trọng số về 0, cho nghiệm thưa |
>
> **Tham số C của scikit-learn là nghịch đảo**
> `LogisticRegression(C=1.0)` có điều chuẩn L2, khá mạnh. C càng lớn thì điều chuẩn càng yếu. Nên muốn so bản tự cài không điều chuẩn với thư viện, phải đặt C thật lớn, ví dụ 10⁹. Quên chuyện này thì hai bên ra hai nghiệm khác nhau và bạn sẽ đi tìm lỗi trong code của mình suốt buổi.

🧠 **Ẩn dụ**: Điều chuẩn (regularization) giống như một "sợi dây thun" luôn kéo các trọng số về gần 0, cạnh tranh với lực kéo từ dữ liệu (muốn trọng số khớp tốt với dữ liệu huấn luyện). L2 giống sợi dây thun đàn hồi đều — càng kéo xa thì lực kéo về càng mạnh, nhưng không bao giờ ép trọng số về đúng 0 tuyệt đối (giống lò xo, luôn còn chút "dao động"). L1 giống một cái **"lưỡi cắt"** — nó có xu hướng cắt hẳn một số trọng số về đúng 0, tạo ra nghiệm "thưa" (sparse), tức là mô hình tự động **chọn đặc trưng** (feature selection) bằng cách loại bỏ hẳn ảnh hưởng của những đặc trưng không quan trọng.

🔬 **Giải thích kỹ thuật sâu**:

- **Vì sao L2 không đẩy về 0 tuyệt đối nhưng L1 thì có (trực giác hình học)**: Đạo hàm của $\lambda w^2$ theo $w$ là $2\lambda w$ — càng gần 0, lực kéo càng yếu dần theo tỷ lệ thuận, nên "chậm dần đều" khi tới gần 0, không bao giờ chạm hẳn (giống chuyển động tiệm cận). Ngược lại, đạo hàm của $\lambda |w|$ là hằng số $\lambda \cdot \text{sign}(w)$ — lực kéo về 0 **không đổi cường độ** bất kể $w$ gần 0 hay xa 0, nên nó có thể "đẩy" $w$ vượt qua và dừng chính xác tại 0 khi gradient từ phía dữ liệu không đủ để chống lại lực kéo hằng số này.

- **Ý nghĩa tham số C = 1/λ**: Trong scikit-learn, thay vì tham số hóa bằng $\lambda$ (hệ số điều chuẩn — càng lớn càng phạt nặng), thư viện dùng $C = 1/\lambda$ (càng lớn thì điều chuẩn càng **yếu**). Đây là một quy ước dễ gây nhầm lẫn kinh điển: nhiều người mới học tưởng "C lớn = điều chuẩn mạnh" (theo trực giác "C là cường độ") nhưng thực ra ngược lại. `C=1.0` là mặc định, tương đương $\lambda = 1$, là một mức điều chuẩn khá mạnh. Muốn tắt gần như hoàn toàn điều chuẩn để so sánh công bằng với bản tự cài (không có điều chuẩn), phải đặt `C` cực lớn (ví dụ $10^9$, tương đương $\lambda \approx 10^{-9} \approx 0$).

- ⚠️ **Đây là một "lỗi âm thầm" kinh điển khác** (nối tiếp chủ đề từ các buổi trước): Nếu bạn quên chỉnh C khi so sánh, sklearn sẽ **chạy hoàn toàn bình thường, không báo lỗi gì** — chỉ là nó đang giải một bài toán tối ưu khác với bài toán của bạn (có điều chuẩn khác nhau). Hai nghiệm sẽ lệch nhau đáng kể (bài giảng nói rõ ở mục 4: độ lệch cỡ $10^{-1}$ thay vì $10^{-8}$ như mong đợi), và người mới học thường mất hàng giờ đi tìm "bug" trong công thức gradient descent của mình, trong khi bug thực sự nằm ở việc quên đặt tham số C.

### 2.7 Nhiều hơn hai lớp (nội dung gốc)

> Softmax tổng quát hoá sigmoid:
> $$\hat p_k = \frac{e^{z_k}}{\sum_j e^{z_j}} \tag{7}$$
>
> Cross entropy giữ nguyên hình dạng: lấy −log của xác suất mà mô hình gán cho lớp đúng. Kỳ thi này chủ yếu là nhị phân, nhưng dạng câu trắc nghiệm về softmax thì có.

🧠 **Ẩn dụ**: Nếu sigmoid là "cân bằng giữa 2 lựa chọn" (như tung một đồng xu), softmax là "chia phần một chiếc bánh cho K người" — mỗi lớp nhận một phần trăm xác suất, và tổng tất cả các phần luôn bằng 100% (do có mẫu số chuẩn hóa $\sum_j e^{z_j}$). Thực chất, sigmoid chính là **trường hợp đặc biệt của softmax khi K=2**.

🔬 **Giải thích kỹ thuật sâu**:

- **Chứng minh sigmoid là trường hợp riêng của softmax**: Với 2 lớp, đặt $z_1 = z, z_2 = 0$ (cố định lớp tham chiếu bằng 0 — một kỹ thuật gọi là "loại bỏ dư thừa tham số"), ta có $\hat p_1 = \frac{e^z}{e^z + e^0} = \frac{e^z}{e^z+1} = \frac{1}{1+e^{-z}} = \sigma(z)$ — chính xác là công thức sigmoid.

- **Vì sao "cross entropy giữ nguyên hình dạng"**: Với đa lớp, công thức cross entropy tổng quát là $L = -\sum_k y_k \log \hat p_k$ trong đó $y$ là vector one-hot. Vì chỉ có đúng một $y_k = 1$ (tại lớp đúng $k^*$), toàn bộ tổng rút gọn về $L = -\log \hat p_{k^*}$ — đúng như bài giảng nói "lấy −log của xác suất mà mô hình gán cho lớp đúng". Đây chính là công thức tổng quát của ví dụ tính tay ở mục 2.2 (ở đó K=2, lớp đúng là vị trí thứ hai với $\hat p = 0{,}7$).

- **Đạo hàm của softmax + cross entropy cũng rút gọn đẹp y hệt sigmoid**: $\frac{\partial L}{\partial z_k} = \hat p_k - y_k$ — hoàn toàn tương tự công thức (4), củng cố lại nguyên lý "cặp liên hợp" đã nói ở mục 2.3. Đây là lý do vì sao trong thực hành, hàm mất mát cho bài toán đa lớp trong PyTorch (`CrossEntropyLoss`) luôn kết hợp sẵn softmax + log-loss thành một bước duy nhất, vừa để tối ưu tốc độ, vừa để tránh vấn đề tràn số khi tính riêng lẻ `exp` rồi `log`.

---

## 3. Hướng dẫn (nội dung gốc, có chú thích kỹ thuật)

### 3.1 Bước 1: dựng dữ liệu biết trước đáp án

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

rng = np.random.default_rng(0)
n, d = 800, 4
X = rng.standard_normal((n, d))
w_that = np.array([1.5, -2.0, 0.5, 0.0])
b_that = -0.3

# Labels are SAMPLED from the true probability, not thresholded. That is what makes the
# problem noisy, and it is why the loss cannot reach zero.
y = (rng.random(n) < sigmoid(X @ w_that + b_that)).astype(float)
print("ti le nhan 1: %.4f" % y.mean())
```

> Tỉ lệ nhãn 1 là 0,4838, tức là hai lớp gần cân bằng.

🔬 **Điểm mấu chốt cần khắc sâu — nhãn được LẤY MẪU chứ không CẮT NGƯỠNG**: Đây là chi tiết dễ bị đọc lướt qua nhưng cực kỳ quan trọng, và bài giảng gốc đã tự chú thích bằng comment tiếng Anh ngay trong code. Có hai cách hoàn toàn khác nhau để tạo nhãn từ xác suất thật $p_{true} = \sigma(Xw_{that} + b_{that})$:

1. **Cắt ngưỡng (thresholding)**: `y = (p_true >= 0.5).astype(float)` — nhãn là **hàm tất định** của $p_{true}$. Nếu làm vậy, một mô hình hoàn hảo học đúng $w_{that}, b_{that}$ sẽ đạt độ chính xác 100% và cross entropy tiến về 0.
2. **Lấy mẫu (sampling)** — cách bài giảng dùng: `y = (rng.random(n) < p_true).astype(float)` — với mỗi mẫu, tung một "đồng xu lệch" có xác suất ra mặt "1" đúng bằng $p_{true}$ của mẫu đó. Đây mô phỏng đúng bản chất bài toán thực tế: ngay cả khi biết chính xác quy luật sinh dữ liệu, **kết quả từng cá thể vẫn ngẫu nhiên** (giống việc biết một đồng xu có xác suất ra ngửa là 70%, nhưng lần tung cụ thể vẫn có thể ra sấp).

🧠 **Ẩn dụ**: Giống việc biết chính xác rằng "70% học sinh lớp này sẽ đậu kỳ thi" (bạn biết hoàn hảo quy luật thống kê) nhưng vẫn **không thể** đoán chắc chắn học sinh cụ thể A có đậu hay không — có yếu tố ngẫu nhiên không thể loại bỏ bằng cách biết thêm thông tin. Đây chính là lý do tại sao "log loss thấp nhất có thể" không phải là 0 (mục 3.2 sẽ nói rõ hơn) — có một "sàn nhiễu" (noise floor) không thể vượt qua dù mô hình có hoàn hảo tới đâu.

### 3.2 Bước 2: cài giảm độ dốc cho cross entropy

```python
def gd_logistic(X, y, lr, so_vong, lam=0.0):
    """Batch gradient descent on the log loss, with optional L2."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lich_su = []
    for _ in range(so_vong):
        p = sigmoid(X @ w + b)
        eps = 1e-12  # log(0) is -inf, and one -inf poisons the mean
        mat = float(-(y * np.log(p + eps)
                      + (1 - y) * np.log(1 - p + eps)).mean() + lam * (w @ w))
        lich_su.append(mat)
        du = p - y
        w = w - lr * ((X.T @ du) / n + 2 * lam * w)
        b = b - lr * du.mean()
    return w, b, lich_su
```

> Chạy với ba tốc độ học 1,0, 0,5, 0,1 trong 5000 vòng: cả ba đều về log loss 0,392272.
>
> **Vì sao không về 0**
> Nhãn được lấy mẫu từ xác suất thật chứ không phải cắt ngưỡng, nên ngay cả mô hình hoàn hảo cũng đoán sai một phần. Log loss thấp nhất có thể chính là entropy của phân phối sinh ra dữ liệu. Lại một lần nữa: biết trước sàn thì khỏi tối ưu vào nhiễu.

🔬 **Giải thích kỹ thuật sâu — soi từng dòng code**:

- **`w = np.zeros(d); b = 0.0`** — khởi tạo bằng 0, không phải ngẫu nhiên. Với logistic regression (một mô hình lồi — convex — không có nhiều điểm cực tiểu cục bộ như mạng nơ-ron sâu), khởi tạo 0 là lựa chọn an toàn và phổ biến, không gặp vấn đề "phá vỡ đối xứng" (symmetry breaking) như trong mạng nơ-ron nhiều tầng.

- **`eps = 1e-12`** — như đã phân tích ở mục 2.2, đây là "lưới an toàn" chống `log(0)`. Chọn `1e-12` đủ nhỏ để không làm sai lệch đáng kể giá trị loss thật, nhưng đủ lớn để tránh `-inf`.

- **`du = p - y`** — chính là công thức đẹp $(\hat p - y)$ đã chứng minh ở mục 2.3. Đặt tên biến `du` (viết tắt tiếng Việt hóa của "delta"/"độ lệch"?) để tận dụng lại cho cả cập nhật $w$ và $b$, tránh tính lại.

- **`w = w - lr * ((X.T @ du) / n + 2 * lam * w)`** — đây là bước cập nhật gradient descent có kèm điều chuẩn L2. Số hạng `2 * lam * w` chính là đạo hàm của $\lambda w^2$ theo $w$ (bằng $2\lambda w$) — khớp với công thức bảng ở mục 2.6.

- **Vì sao "cả ba tốc độ học đều về cùng 0,392272"**: Đây là một tính chất đặc biệt của bài toán **lồi** (convex optimization). Hàm mất mát cross entropy của logistic regression là hàm lồi theo $(w,b)$ — không có cực tiểu cục bộ giả, chỉ có duy nhất một cực tiểu toàn cục. Miễn là tốc độ học đủ nhỏ để không "nhảy vọt" qua đáy (không phân kỳ), gradient descent với các tốc độ học khác nhau (1,0 / 0,5 / 0,1) chỉ khác nhau về **tốc độ hội tụ**, nhưng sau đủ số vòng (ở đây là 5000), tất cả đều hội tụ về **cùng một điểm** — điểm cực tiểu toàn cục duy nhất.

- **"Log loss thấp nhất có thể chính là entropy của phân phối sinh ra dữ liệu"**: Đây là một khái niệm sâu từ lý thuyết thông tin. Vì nhãn $y$ được lấy mẫu ngẫu nhiên từ $p_{true}$, ngay cả khi mô hình học được **chính xác** $w_{that}, b_{that}$ (tức $\hat p = p_{true}$ với mọi mẫu), cross entropy kỳ vọng vẫn bằng entropy nhị phân $H(p_{true}) = -[p_{true}\log p_{true} + (1-p_{true})\log(1-p_{true})]$ — một đại lượng luôn dương (trừ khi $p_{true}$ đúng bằng 0 hoặc 1). Đây chính là "sàn nhiễu" không thể vượt qua, tương tự khái niệm "phương sai không thể giải thích" (irreducible error) trong hồi quy tuyến tính mà bạn đã gặp ở buổi 04.

### 3.3 Bước 3: nhìn cái bẫy bình phương bằng chính tay mình

```python
def gd_binh_phuong(X, y, lr, so_vong, w0=None):
    """Same loop, squared loss on the sigmoid output."""
    n, d = X.shape
    w = np.zeros(d) if w0 is None else w0.copy()
    b = 0.0
    for _ in range(so_vong):
        p = sigmoid(X @ w + b)
        du = (p - y) * p * (1 - p)  # the sigmoid derivative survives here
        w = w - lr * (X.T @ du) / n
        b = b - lr * du.mean()
    return w, b

w0 = np.array([-8.0, 8.0, 0.0, 0.0])  # confidently wrong on purpose
```

> Chạy cả hai từ w0 ấy và đếm số vòng để đạt độ chính xác 0,75. Cross entropy cần 19 vòng, bình phương cần 1813.

🔬 **Điểm nối trực tiếp với công thức (5)**: Dòng `du = (p - y) * p * (1 - p)` chính là thừa số phụ $\hat p(1-\hat p)$ nhân thêm vào $(\hat p - y)$ đã phân tích ở mục 2.4 — comment trong code `# the sigmoid derivative survives here` xác nhận đúng điều này: đây là trường hợp đạo hàm sigmoid **không** bị triệt tiêu (khác với cross entropy ở mục 3.2), mà "sống sót" và gây ra hiện tượng vanishing gradient.

🧠 **Vì sao chọn w0 = [−8, 8, 0, 0] "cố ý sai và tự tin"?** Nhớ lại $w_{that} = [1{,}5; -2{,}0; 0{,}5; 0]$ — dấu của $w0$ ở hai chiều đầu tiên **ngược hoàn toàn** với $w_{that}$ (−8 thay vì dương, +8 thay vì âm), và độ lớn 8 rất cao khiến $|z|$ lớn với hầu hết mẫu → mô hình "rất tự tin nhưng tự tin sai hướng" — đúng kịch bản tệ nhất mà vanishing gradient gây hại nặng nhất, như đã phân tích ở mục 2.4.

---

## 4. Bài tập mẫu (nội dung gốc)

**Đề.** Chứng minh bản giảm độ dốc của bạn ra cùng nghiệm với `LogisticRegression`.

**Phân tích.** Có một cái bẫy phải xử lý trước: scikit-learn mặc định có điều chuẩn L2 với `C=1.0`, còn bản của ta không có. Muốn so được, phải tắt điều chuẩn của thư viện bằng cách đặt C rất lớn.

**Lời giải.**

```python
from sklearn.linear_model import LogisticRegression

w_gd, b_gd, _ = gd_logistic(X, y, lr=1.0, so_vong=20000)

# C=1e9 all but removes the L2 penalty, and a tight tol keeps the solver from stopping
# early at a point our own loop would walk past.
sk = LogisticRegression(C=1e9, max_iter=5000, tol=1e-10).fit(X, y)

lech = max(np.abs(w_gd - sk.coef_[0]).max(), abs(b_gd - sk.intercept_[0]))
print("GD w =", np.round(w_gd, 4), "b = %.4f" % b_gd)
print("sklearn w =", np.round(sk.coef_[0], 4), "b = %.4f" % sk.intercept_[0])
print("lech lon nhat: %.3e" % lech)
```

**Kết quả.**
```
GD w = [ 1.6487 -2.1401  0.3706  0.009 ] b = -0.2517
sklearn w = [ 1.6487 -2.1401  0.3706  0.009 ] b = -0.2517
lech lon nhat: 1.223e-09
```

> **Cổng kiểm**
> Độ lệch dưới 10⁻⁸. Nếu bạn quên C=1e9, độ lệch sẽ cỡ 10⁻¹, và đó không phải lỗi của bạn mà là hai bài toán khác nhau.
>
> **Nghiệm tìm được không bằng w_that, và đó là đúng**
> w_that là [1,5; −2,0; 0,5; 0], nghiệm tìm được là [1,649; −2,140; 0,371; 0,009]. Chênh lệch đến từ nhiễu lấy mẫu: chỉ có 800 mẫu, và nhãn được sinh ngẫu nhiên từ xác suất. Đặc trưng thứ tư có trọng số thật bằng 0 và mô hình cho nó 0,009, tức là gần đúng chứ không đúng hẳn. Đây là hình ảnh cụ thể của câu "ước lượng có phương sai".

🔬 **Giải thích kỹ thuật sâu**:

- **Vì sao `tol=1e-10` cũng quan trọng không kém `C=1e9`**: Solver mặc định của `LogisticRegression` (thường là `lbfgs`) dừng sớm khi độ thay đổi của hàm mất mát giữa hai vòng lặp nhỏ hơn `tol`. Nếu `tol` mặc định (thường `1e-4`) được dùng, solver có thể "cảm thấy đủ tốt" và dừng lại **trước khi** đạt tới đúng điểm cực tiểu mà vòng lặp gradient descent tự viết tay (chạy đủ 20000 vòng, rất triệt để) đạt tới. Kết quả là dù đã sửa đúng `C`, độ lệch vẫn có thể lớn hơn mong đợi nếu quên siết `tol`.

- **"Ước lượng có phương sai" — khái niệm thống kê nền tảng**: Đây là một trong những bài học triết lý quan trọng nhất của cả buổi. $w_{that}$ là tham số **thật** của quá trình sinh dữ liệu (data generating process) — một hằng số cố định, không ai biết trước trong thực tế. $w_{gd}$ (hay $w_{sk}$) là **ước lượng** (estimate) rút ra từ một mẫu dữ liệu hữu hạn (n=800) có nhiễu ngẫu nhiên. Về mặt thống kê, $w_{gd}$ là một **biến ngẫu nhiên** — nếu bạn lấy một bộ 800 mẫu khác (đổi seed của `rng`), bạn sẽ được một $w_{gd}$ khác, dao động quanh $w_{that}$ nhưng không trùng khớp. Độ "dao động" này gọi là **phương sai của ước lượng** (variance of the estimator), và nó giảm dần khi $n$ tăng (định lý giới hạn trung tâm — central limit theorem — cho biết độ lệch chuẩn của ước lượng giảm theo tỷ lệ $1/\sqrt n$).

- **Ý nghĩa con số 0,009 ở đặc trưng thứ 4**: $w_{that,4} = 0$ (đặc trưng này thực sự không ảnh hưởng gì tới nhãn), nhưng mô hình ước lượng ra 0,009 — khác 0 nhưng rất nhỏ. Đây **không phải** dấu hiệu mô hình học sai hay có bug, mà là hệ quả tất yếu của nhiễu lấy mẫu hữu hạn: với xác suất gần như chắc chắn, ước lượng của một hệ số có giá trị thật bằng 0 sẽ **không bao giờ** ra đúng số 0 tuyệt đối khi tính từ dữ liệu hữu hạn có nhiễu — nó sẽ dao động rất nhỏ quanh 0. Đây chính là lý do thống kê học cần các phép kiểm định giả thuyết (hypothesis testing) hoặc khoảng tin cậy (confidence interval) để phân biệt "hệ số nhỏ nhưng thật sự khác 0" với "hệ số nhỏ do nhiễu, giá trị thật là 0".

---

## 5. Bài tập tự làm (nội dung gốc, giữ nguyên toàn bộ 5 bài)

### Bài tập 1. Ba phương án nhiễu của câu 38

> Tự tính lại ba con số 0,35667, 1,20397, 0,51457, rồi viết một câu cho mỗi con số nói rõ nó sinh ra từ lỗi gì.
>
> **Cổng kiểm**
> Ba dòng, mỗi dòng một lỗi có tên. Bài này không cần máy.

💡 *Gợi ý làm bài (không phải đáp án — bạn tự viết ba câu theo cổng kiểm)*: xem lại phân tích chi tiết ba lỗi ở mục 2.2 phía trên (lỗi lấy nhầm vị trí, lỗi nhầm cơ số log) để tự diễn đạt lại thành ba câu ngắn gọn của riêng bạn.

### Bài tập 2. Gradient tự tính khớp torch

> Dùng torch.autograd tính gradient của log loss trên một lô nhỏ, rồi so với công thức $\frac{1}{n}X^{\top}(\hat p - y)$ bạn tự cài.
>
> **Cổng kiểm**
> Độ lệch dưới 10⁻⁶. Đây là cách duy nhất để chắc rằng bạn rút gọn đạo hàm đúng chứ không phải may.

🔬 **Gợi ý kỹ thuật cho bài này** (khung code, không phải lời giải hoàn chỉnh — tự điền phần tính toán): Đây chính là ứng dụng trực tiếp của nguyên tắc bạn đã ghi nhớ từ các buổi trước — "luôn kiểm chứng code nhanh (vectorized) bằng một tham chiếu chậm hơn nhưng đáng tin cậy hơn". Ở đây, "tham chiếu chậm nhưng đáng tin" chính là `torch.autograd` — công cụ tính đạo hàm tự động (automatic differentiation), không dựa vào công thức bạn tự rút gọn bằng tay (nên không thể "may mắn đúng do rút gọn sai mà kết quả trùng"), mà tính đạo hàm bằng cách lần theo từng phép toán nguyên tố (sigmoid, log, nhân, cộng...) và áp dụng chain rule một cách máy móc, đáng tin cậy tuyệt đối về mặt toán học.

```python
import torch

X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.float64)
w_t = torch.zeros(d, dtype=torch.float64, requires_grad=True)
b_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)

z = X_t @ w_t + b_t
p = torch.sigmoid(z)
eps = 1e-12
loss = -(y_t * torch.log(p + eps) + (1 - y_t) * torch.log(1 - p + eps)).mean()
loss.backward()

grad_torch = w_t.grad.numpy()
grad_cong_thuc = (X.T @ (sigmoid(X @ np.zeros(d) + 0.0) - y)) / n
lech = np.abs(grad_torch - grad_cong_thuc).max()
print("lech gradient: %.3e" % lech)
```

⚠️ Lưu ý: đoạn code trên tính gradient tại điểm $w=0, b=0$ để khớp với điểm khởi tạo — bạn cần tự điều chỉnh nếu muốn kiểm tra tại một điểm $(w,b)$ khác, và nhớ đảm bảo cả hai phía (torch và công thức tay) đang tính đạo hàm tại **cùng một điểm** $(w,b)$, nếu không phép so sánh sẽ vô nghĩa.

### Bài tập 3. Đo lại cái bẫy bình phương

> Dựng lại bảng hai dòng ở mục 2.5 bằng code của mình: độ lớn gradient ở bước đầu, và số vòng để đạt độ chính xác 0,75.
>
> **Cổng kiểm**
> Hai con số khớp 0,51387 và 0,0028765, cùng 19 và 1813. Rồi trả lời thêm: nếu khởi tạo bằng 0 thay vì w0, khoảng cách ấy còn không?

> ⚠️ Bài giảng gốc ghi "mục 2.5" trong phần bài tập tự làm, nhưng nội dung bảng số 0,51387 / 0,0028765 / 19 / 1813 thực chất nằm ở **mục 2.4** (Vì sao không dùng mất mát bình phương) của tài liệu gốc — có thể là một chỗ đánh số nhầm trong bản PDF gốc. Mình giữ nguyên câu chữ của đề để bạn đối chiếu đúng với bản gốc, chỉ lưu ý chỗ này để tránh bạn lật nhầm mục khi ôn tập.

🔬 **Gợi ý kỹ thuật**: "Độ lớn gradient ở bước đầu" nghĩa là chuẩn (norm) của vector gradient $\frac{\partial L}{\partial w}$ (có thể ghép thêm $\frac{\partial L}{\partial b}$) tính ngay tại vòng lặp đầu tiên, trước khi cập nhật $w$ — dùng `np.linalg.norm(grad)`. "Số vòng để đạt độ chính xác 0,75" nghĩa là chạy vòng lặp, sau mỗi vòng tính `accuracy = (dự đoán nhãn == y).mean()` (với ngưỡng mặc định τ=0,5 theo công thức 6), và ghi lại vòng đầu tiên mà accuracy vượt 0,75. Câu hỏi mở ở cuối ("nếu khởi tạo bằng 0...") đã có gợi ý đáp án ngay trong bài giảng gốc ở mục 2.4: *"Khi khởi tạo bằng 0... khoảng cách ấy biến mất"* — bạn nên tự chạy code để xác nhận bằng số liệu cụ thể của riêng mình, không chỉ chép lại câu trả lời.

### Bài tập 4. Điều chuẩn L2 và đường cong trọng số

> Chạy bản của bạn với λ chạy từ 0 tới 1, vẽ bốn trọng số theo λ.
>
> **Cổng kiểm**
> Bốn đường cùng co về 0 khi λ tăng, nhưng không đường nào chạm 0. Rồi làm lại với L1 và chỉ ra khác biệt bằng hình.

🔬 **Gợi ý kỹ thuật**: Hàm `gd_logistic` ở mục 3.2 đã có sẵn tham số `lam` cho điều chuẩn L2 — bạn chỉ cần quét một dải giá trị `lam` (ví dụ `np.linspace(0, 1, 30)` hoặc dùng thang log `np.logspace(-3, 0, 30)` để thấy rõ hơn hành vi ở vùng λ nhỏ), lưu lại 4 hệ số $w$ ứng với mỗi $\lambda$, rồi vẽ bằng matplotlib với trục hoành là λ, trục tung là giá trị từng trọng số — đây gọi là **"đường cong điều chuẩn"** (regularization path), một công cụ trực quan hóa rất phổ biến để hiểu hành vi co rút của trọng số.

Với L1, bạn cần cài một hàm gradient descent mới vì $|w|$ không khả vi tại $w=0$ (đạo hàm không xác định đúng nghĩa tại điểm gấp khúc) — cách xử lý phổ biến nhất trong khuôn khổ gradient descent đơn giản là dùng **subgradient**: quy ước $\text{sign}(0) = 0$, tức tại đúng $w=0$ thì không cộng thêm lực kéo nào (đã ở đích). Đây là lý do trực quan tại sao L1 có thể khiến trọng số "dừng đúng tại 0" mà không "vọt qua âm" như một dao động không tắt.

### Bài tập 5. Ngưỡng không phải lúc nào cũng là 0,5

> Với mô hình đã huấn luyện, quét ngưỡng τ từ 0,05 tới 0,95 và vẽ macro-F1 theo τ.
>
> **Cổng kiểm**
> Một con số τ tốt nhất kèm hình. Rồi trả lời: nếu bạn chọn τ bằng cách nhìn chính tập kiểm tra, bạn đã phạm lỗi gì? Buổi 07 gọi tên lỗi đó.

🔬 **Gợi ý kỹ thuật**: `sklearn.metrics.f1_score(y_true, y_pred, average='macro')` tính sẵn macro-F1 — nhưng nhớ lại ghi chú bạn đã lưu từ trước: **"omitting labels= from f1_score silently changes behavior"** — khi dùng hàm này, cân nhắc truyền rõ `labels=[0, 1]` để tránh hành vi ngầm định không mong muốn nếu một trong hai lớp vắng mặt hoàn toàn trong một lô dự đoán (ví dụ tại τ rất cao, mô hình có thể dự đoán toàn bộ là lớp 0). Về câu hỏi mở cuối bài: đáp án đã được chính bài giảng "bật mí" một phần ở mục 2.5 — đây là lỗi rò rỉ dữ liệu qua việc chọn siêu tham số dựa trên tập test thay vì tập validation riêng biệt, chủ đề buổi 07 sẽ đặt tên chính thức và phân tích sâu hơn.

---

## Chuẩn bị cho buổi sau (nội dung gốc)

> Buổi 06 nói về đo lường: macro-F1, ma trận nhầm lẫn, ROC. Mang theo câu hỏi: một mô hình đạt độ chính xác 95% trên bộ dữ liệu mà 95% mẫu thuộc một lớp thì giỏi tới mức nào?

🧠 **Gợi mở tư duy trước khi sang buổi 06** (không phải đáp án, chỉ là câu hỏi để bạn tự suy ngẫm trước): Hãy thử tưởng tượng một mô hình "lười biếng" nhất có thể — nó **không học gì cả**, chỉ luôn luôn dự đoán "lớp đa số" bất kể input là gì. Nếu 95% dữ liệu thuộc lớp đa số, mô hình lười biếng này vẫn đạt accuracy 95% mà không cần một dòng code học máy nào. Câu hỏi buổi 06 để lại chính là: **con số 95% ấy có ý nghĩa gì không, nếu một mô hình "không học" cũng đạt được con số tương tự?** — đây là lý do vì sao macro-F1, ma trận nhầm lẫn (confusion matrix), và đường cong ROC tồn tại: chúng đo lường **những khía cạnh mà accuracy đơn thuần che giấu mất** khi hai lớp mất cân bằng nghiêm trọng.

---

## Tổng kết bản đồ kiến thức buổi 05

| Khái niệm | Công thức cốt lõi | Vì sao quan trọng |
|---|---|---|
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | Cầu nối từ điểm số thực sang xác suất [0,1] |
| Cross entropy | $-\frac1n\sum[y\log\hat p + (1-y)\log(1-\hat p)]$ | Hàm mất mát đúng cho phân loại, tránh vanishing gradient |
| Đạo hàm rút gọn | $\frac1n X^\top(\hat p - y)$ | Cặp sigmoid + cross entropy "liên hợp", đạo hàm sigmoid tự triệt tiêu |
| Bẫy MSE trên sigmoid | thêm thừa số $\hat p(1-\hat p)$ | Gradient biến mất khi sai tự tin → học chậm 95 lần |
| Ngưỡng τ | mặc định 0,5, không phải chân lý | Tối ưu theo macro-F1, không nhìn tập test khi chọn |
| L1 vs L2 | L2 co gần 0, L1 đẩy hẳn về 0 | C trong sklearn là nghịch đảo của λ — bẫy so sánh kinh điển |
| Softmax | $\hat p_k = e^{z_k}/\sum_j e^{z_j}$ | Tổng quát hóa sigmoid cho K lớp, cùng công thức đạo hàm rút gọn |

**Ba chủ đề xuyên suốt nối buổi 05 với các buổi trước**:
1. **Lỗi âm thầm** (từ buổi trước) → ở đây là: quên `eps` gây `-inf` lan tràn; quên chỉnh `C` khiến sklearn "chạy được nhưng sai bài toán"; quên `labels=` trong `f1_score`.
2. **Luôn kiểm chứng bằng tham chiếu độc lập** → so `gd_logistic` với `LogisticRegression` (mục 4); so công thức gradient tay với `torch.autograd` (bài tập 2).
3. **Biết trước "cái sàn" để không tối ưu vào nhiễu** → entropy của phân phối sinh dữ liệu là log loss thấp nhất có thể đạt được, tương tự khái niệm "phương sai không giải thích được" ở hồi quy tuyến tính buổi 04.
