# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier

from utilities import visualize_classifier

print("Робота 2.2 'Логістичний класифікатор:'")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Визначте зразки вхідних даних
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5],
              [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4],
              [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Створіть класифікатор логістичної регресії
classifier = OneVsRestClassifier(linear_model.LogisticRegression(solver='liblinear', C=1))
# classifier = OneVsRestClassifier(linear_model.LogisticRegression(solver='liblinear', C=100))

# Тренувати класифікатор
classifier.fit(X, y)

# Візуалізуйте продуктивність класифікатора
visualize_classifier(classifier, X, y)