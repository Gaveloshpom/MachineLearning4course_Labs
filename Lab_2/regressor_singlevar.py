import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

print("Робота 2.7 \"Створення регресора однієї змінної:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Вхідний файл, що містить дані
input_file = 'data_singlevar_regr.txt'

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

# Створіть об'єкт лінійної регресії
regressor = linear_model.LinearRegression()

# Тренуйте модель за допомогою навчальних наборів
regressor.fit(X_train, y_train)

# Спрогнозуйте вихід
y_test_pred = regressor.predict(X_test)

# Відобразіть вихідні дані
fig = plt.figure()
# Текст заголовку вікна графіка
fig.canvas.manager.set_window_title("Скицко Руслан, 642П, вар. 20")
plt.title('Регресор однієї змінної', color='red')
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Обчислення показників продуктивності
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

# Витривалість моделі
output_model_file = 'model.pkl'

# Зберегти модель
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
# Завантажте модель
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Виконайте прогноз на основі тестових даних
y_test_pred_new = regressor_model.predict(X_test)
print("\nНова середня абсолютна похибка =",
      round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))