import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def roc_thu_cong(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thu_tu = np.argsort(-y_score)
    y_true_sorted = y_true[thu_tu]
    y_score_sorted = y_score[thu_tu]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    fpr_list = [0.0]
    tpr_list = [0.0]

    tp = 0
    fp = 0
    i = 0
    n = len(y_true_sorted)

    while i < n:
        j = i
        while j < n and y_score_sorted[j] == y_score_sorted[i]:
            if y_true_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
        i = j

    return np.array(fpr_list), np.array(tpr_list)


def auc_thu_cong(fpr, tpr):
    return np.trapezoid(tpr, fpr)



# --- Dữ liệu kiểm tra ---
random_state = np.random.RandomState(0)
y_true = random_state.randint(0, 2, size=500)
y_score = random_state.rand(500)

fpr_toi, tpr_toi = roc_thu_cong(y_true, y_score)
auc_toi = auc_thu_cong(fpr_toi, tpr_toi)

fpr_sk, tpr_sk, _ = roc_curve(y_true, y_score)
auc_sk = roc_auc_score(y_true, y_score)

print("AUC tự tính :", auc_toi)
print("AUC sklearn :", auc_sk)
print("Độ lệch AUC :", abs(auc_toi - auc_sk))

# --- Vẽ biểu đồ ---
plt.figure(figsize=(7, 7))

# Đường ROC tự tính (vẽ nét liền, dày)
plt.plot(fpr_toi, tpr_toi, color="tab:blue", linewidth=2.5,
         label=f"Tự tính (AUC = {auc_toi:.4f})")

# Đường ROC của sklearn (vẽ nét đứt, mảnh, đè lên để so sánh)
plt.plot(fpr_sk, tpr_sk, color="tab:orange", linewidth=1.2, linestyle="--",
         label=f"sklearn (AUC = {auc_sk:.4f})")

# Đường chéo tham chiếu (random classifier, AUC = 0.5)
plt.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":",
         label="Random (AUC = 0.5)")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Đường cong ROC — Tự tính vs sklearn")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()