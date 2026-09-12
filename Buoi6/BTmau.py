from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

y_true = [0]*10 + [1]*10
y_pred = [0]*5 + [1]*5  +  [1]*9 + [0]*1

cm = confusion_matrix(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, labels=[0,1], average="macro")
accuracy = accuracy_score(y_true, y_pred)

print("Ma tran nham lan")
print(cm)
print(f"That 0, Doan 0 (TN) = {cm[0][0]}   |   That 0, Doan 1 (FP) = {cm[0][1]}")
print(f"That 1, Doan 0 (FN) = {cm[1][0]}   |   That 1, Doan 1 (TP) = {cm[1][1]}")

print("macro F1 : %.4f" %macro_f1)
print("accuracy : %.4f" %accuracy)

print(classification_report(y_true, y_pred, digits=4))