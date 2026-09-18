import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
X = dataset.data
y = dataset.target

print(dataset.data[:5])

from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0,max_iter=200)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

import numpy as np
from sklearn.metrics import mean_squared_error


mse_loss = mean_squared_error(y_test, y_pred)


lambda_value = 0.01

l2_loss = lambda_value * np.sum(classifier.coef_ ** 2)


total_loss = mse_loss + l2_loss

print("MSE Loss:", mse_loss)
print("L2 Regularization Loss:", l2_loss)
print("Total Loss:", total_loss)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)