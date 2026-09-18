import matplotlib.pyplot as plt
import random
import numpy as np

probablity_positive = []
probablity_negative = []
b = random.randint(650, 1000)
for i in range(0, b):
    a = random.uniform(0.75, 1.0)
    probablity_positive.append(a)
    b = 1000 - b

for i in range(0, b):
    a = random.uniform(0, 0.75)
    probablity_positive.append(a)

    b = random.randint(650, 1000)
for i in range(b):
    a = random.uniform(0.75, 1.0)
    probablity_negative.append(a)

b = 1000 - b

for i in range(0, b):
    a = random.uniform(0, 0.75)
    probablity_negative.append(a)

tpr = []
fpr = []


thresholds = [random.uniform(0, 1) for _ in range(40)]
thresholds.sort(reverse=True)

for threshold in thresholds:
    true_positive = 0.0
    false_negative = 0.0
    false_positive = 0.0
    true_negative = 0.0

    for j in range(0, 1000):
        if probablity_positive[j] >= threshold:
            true_positive += 1
        else:
            false_negative += 1

    for j in range(0, 1000):
        if probablity_negative[j] >= threshold:
            false_positive += 1
        else:
            true_negative += 1
    print(
        "Accuracy",
        (true_positive + true_negative)
        / (true_positive + true_negative + false_positive + false_negative),
        end=" ",
    )

    print("True Positive: ", true_positive, end=" ")
    print("True Negative: ", true_negative, end=" ")
    print("False Positive: ", false_positive, end=" ")
    print("False Negative: ", false_negative)
    conf_matrix = np.array(
        [[true_positive, false_positive], [false_negative, true_negative]]
    )
    print("\nConfusion Matrix:")
    print(conf_matrix)
    tpr.append(true_positive / (true_positive + false_negative))
    fpr.append(false_positive / (false_positive + true_negative))

plt.plot(fpr, tpr, color="blue", marker="o", linewidth=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
start = [0, 1]
end = [0, 1]
plt.plot(start, end, linestyle="dotted", color="red", linewidth=2, marker="o")
plt.show()
