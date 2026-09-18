import os
import cv2
import numpy as np
import kagglehub

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

path = kagglehub.dataset_download(
    "adhoppin/blood-cell-detection-datatset"
)

print("Path to dataset files:", path)

X = []
Y = []

train_images = os.path.join(path, "train", "images")
train_labels = os.path.join(path, "train", "labels")

for file in sorted(os.listdir(train_images)):

    if file.endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(train_images, file)

        label_file = os.path.splitext(file)[0] + ".txt"
        label_path = os.path.join(train_labels, label_file)

        img = cv2.imread(image_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        median = np.median(gray)

        count = 0

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                count = len(f.readlines())

        X.append(median)
        Y.append(count)

X = np.array(X).reshape(-1, 1)
Y = np.array(Y)



X_test = []
Y_test = []

test_images = os.path.join(path, "test", "images")
test_labels = os.path.join(path, "test", "labels")

for file in sorted(os.listdir(test_images)):

    if file.endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(test_images, file)

        label_file = os.path.splitext(file)[0] + ".txt"
        label_path = os.path.join(test_labels, label_file)

        img = cv2.imread(image_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        median = np.median(gray)

        count = 0

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                count = len(f.readlines())

        X_test.append(median)
        Y_test.append(count)

X_test = np.array(X_test).reshape(-1, 1)
Y_test = np.array(Y_test)


model = LinearRegression()

model.fit(X, Y)

m = model.coef_[0]
c = model.intercept_

print("\nLinear Regression Model")
print("Slope =", m)
print("Intercept =", c)

print("\nRegression Equation:")
print("Cell Count =", m, "* Median Intensity +", c)

Y_pred = model.predict(X_test)

print("\nPredicted Cell Counts:")
print(Y_pred)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation")
print("MAE  =", mae)
print("MSE  =", mse)
print("RMSE =", rmse)

print("\nActual vs Predicted")

for actual, predicted in zip(Y_test, Y_pred):
    print(
        "Actual:",
        actual,
        "Predicted:",
        round(predicted, 2)
    )