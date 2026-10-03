# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

print("Робота 2.1 \"Попередня обробка даних:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Бінаризація даних
data_binarized = \
preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nБінаризовані дані:\n", data_binarized)

# Надрукуйте середнє значення та стандартне відхилення
print("\nПЕРЕД:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Видаліть середнє
data_scaled = preprocessing.scale(input_data)
print("\nПІСЛЯ:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Min-max масштабування
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Нормалізація даних
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 нормалізовані дані:\n", data_normalized_l1)
print("\nl2 нормалізовані дані:\n", data_normalized_l2)
