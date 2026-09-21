import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

rng = np.random.default_rng(42)

# ================= CASE 1 : BINARY =================

# 1000 positive, 1000 negative
# 250 positives predicted negative -> FN=250, TP=750
# 150 negatives predicted positive -> FP=150, TN=850

y_true = np.array([1] * 1000 + [0] * 1000)
y_pred = np.array([1] * 750 + [0] * 250 + [1] * 150 + [0] * 850)

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()

print("=== CASE 1: BINARY ===")
print("Confusion matrix:\n", cm)
print("TP =", TP, " FP =", FP, " TN =", TN, " FN =", FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy : %.4f" % accuracy)
print("Precision : %.4f" % precision)
print("Recall : %.4f" % recall)
print("Specificity : %.4f" % specificity)
print("F1-score : %.4f" % f1)


# Random probability scores
scores = np.empty(2000)

scores[:750] = rng.uniform(0.75, 1.00, 750)
scores[750:1000] = rng.uniform(0.30, 0.75, 250)
scores[1000:1150] = rng.uniform(0.50, 0.80, 150)
scores[1150:] = rng.uniform(0.00, 0.50, 850)

frac = np.mean(scores[:1000] > 0.75)

print(
    "Fraction of positive samples with score > 0.75: %.2f%%"
    % (frac * 100)
)

fpr, tpr, _ = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

print("AUC : %.4f" % roc_auc)


# Binary ROC Curve
plt.figure(figsize=(5, 4))

plt.plot(
    fpr,
    tpr,
    label="ROC (AUC = %.4f)" % roc_auc
)

plt.plot(
    [0, 1],
    [0, 1],
    'k--',
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Binary Classification")
plt.legend()
plt.tight_layout()
plt.show()


# ================= CASE 2 : MULTICLASS =================

# 3 classes x 1000 samples
# Each class keeps 750 correct
# 250 samples are distributed equally to the other two classes

print("\n=== CASE 2: MULTICLASS (3 classes) ===")

y_true_m = []
y_pred_m = []

for c in range(3):

    others = [k for k in range(3) if k != c]

    y_true_m += [c] * 1000

    y_pred_m += (
        [c] * 750
        + [others[0]] * 125
        + [others[1]] * 125
    )


y_true_m = np.array(y_true_m)
y_pred_m = np.array(y_pred_m)

cm_m = confusion_matrix(y_true_m, y_pred_m)

print("Confusion matrix:\n", cm_m)

N = cm_m.sum()

print("\nPer-class values (one-vs-rest):")

prec_l = []
rec_l = []
spec_l = []
f1_l = []


for c in range(3):

    TPc = cm_m[c, c]

    FNc = cm_m[c, :].sum() - TPc

    FPc = cm_m[:, c].sum() - TPc

    TNc = N - TPc - FNc - FPc

    p = TPc / (TPc + FPc)

    r = TPc / (TPc + FNc)

    s = TNc / (TNc + FPc)

    f = 2 * p * r / (p + r)

    prec_l.append(p)
    rec_l.append(r)
    spec_l.append(s)
    f1_l.append(f)

    print(
        "Class %d -> TP=%d FP=%d TN=%d FN=%d | "
        "Prec=%.4f Rec=%.4f Spec=%.4f F1=%.4f"
        % (
            c,
            TPc,
            FPc,
            TNc,
            FNc,
            p,
            r,
            s,
            f
        )
    )


acc_m = np.trace(cm_m) / N

print("\nAccuracy : %.4f" % acc_m)
print("Macro Precision : %.4f" % np.mean(prec_l))
print("Macro Recall : %.4f" % np.mean(rec_l))
print("Macro Specificity : %.4f" % np.mean(spec_l))
print("Macro F1-score : %.4f" % np.mean(f1_l))


# ================= MULTICLASS ROC =================

# Convert multiclass labels to binary format
Y_bin = label_binarize(
    y_true_m,
    classes=[0, 1, 2]
)

S = np.empty((3000, 3))


# Generate probability scores
for c in range(3):

    is_c = (y_true_m == c)

    correct = is_c & (y_pred_m == c)

    wrong = is_c & (y_pred_m != c)

    S[correct, c] = rng.uniform(
        0.75,
        1.00,
        correct.sum()
    )

    S[wrong, c] = rng.uniform(
        0.30,
        0.75,
        wrong.sum()
    )

    not_c = ~is_c

    pred_c = not_c & (y_pred_m == c)

    rest = not_c & (y_pred_m != c)

    S[pred_c, c] = rng.uniform(
        0.50,
        0.80,
        pred_c.sum()
    )

    S[rest, c] = rng.uniform(
        0.00,
        0.50,
        rest.sum()
    )


# Multiclass ROC curves
plt.figure(figsize=(5, 4))

aucs = []


for c in range(3):

    fpr_c, tpr_c, _ = roc_curve(
        Y_bin[:, c],
        S[:, c]
    )

    a = auc(fpr_c, tpr_c)

    aucs.append(a)

    plt.plot(
        fpr_c,
        tpr_c,
        label="Class %d (AUC = %.4f)" % (c, a)
    )

    print(
        "AUC class %d : %.4f"
        % (c, a)
    )


print(
    "Macro AUC : %.4f"
    % np.mean(aucs)
)


plt.plot(
    [0, 1],
    [0, 1],
    'k--',
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves - Multiclass (One-vs-Rest)"
)

plt.legend()
plt.tight_layout()
plt.show()