# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

print("Робота 2.8 \"Створення багатовимірного регресора:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Вхідний файл, що містить дані
input_file = 'data_multivar_regr.txt'

# Прочитати дані
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розділіть дані на навчальні та тестувальні
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Навчальні дані
X_train, y_train = X[:num_training], y[:num_training]

# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створіть модель лінійної регресії
linear_regressor = linear_model.LinearRegression()

# Тренуйте модель за допомогою навчальних наборів
linear_regressor.fit(X_train, y_train)

# Спрогнозуйте вихід
y_test_pred = linear_regressor.predict(X_test)

# Вимірюйте продуктивність
print("Продуктивність лінійної регресії:")
print("Середня абсолютна похибка =",
      round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середня квадратична помилка =",
      round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіана абсолютної похибки =",
      round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Оцінка поясненої дисперсії =",
      round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 оцінка =", round(sm.r2_score(y_test, y_test_pred), 2))

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nЛінійна регресія:\n", linear_regressor.predict(datapoint))
print("\nПоліноміальна регресія:\n",
      poly_linear_model.predict(poly_datapoint))