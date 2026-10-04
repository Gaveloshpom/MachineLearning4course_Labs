# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import pandas as pd

print("Робота 2.9 \"Оцінювання вартості нерухомості з використанням")
print("регресорана основі машини опорних векторів :\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Перемішайте дані
X, y = shuffle(data, target, random_state=7)

# Розділіть дані на навчальні та тестові набори
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Створити опорну векторну модель регресії
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

# Навчання опорного векторного регресора
sv_regressor.fit(X_train, y_train)

# Оцініть продуктивність опорного векторного регресора
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("\n#### Продуктивність ####")
print("Середня квадратична помилка =", round(mse, 2))
print("Оцінка поясненої дисперсії =", round(evs, 2))

# Перевірте регресор на тестовій точці даних
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666,
             20.2, 351.34, 15.27]
print("\nПрогнозована ціна:", sv_regressor.predict([test_data])[0])