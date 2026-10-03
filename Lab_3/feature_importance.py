import inspect
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import fetch_california_housing

# --- 1. Завантаження даних ---
california = fetch_california_housing()
X, y = shuffle(california.data, california.target, random_state=7)

# --- 2. Розділення даних ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# --- 3. Підготовка аргументу для AdaBoost
#        (сумісність зі старими/новими sklearn) ---
base_estimator = DecisionTreeRegressor(max_depth=4)
sig = inspect.signature(AdaBoostRegressor.__init__)
if 'estimator' in sig.parameters:
    # сучасна назва параметра
    regressor = AdaBoostRegressor(estimator=base_estimator,
                                  n_estimators=400,
                                  random_state=7)
else:
    # стара назва параметра
    regressor = AdaBoostRegressor(base_estimator=base_estimator,
                                  n_estimators=400,
                                  random_state=7)

# --- 4. Навчання ---
regressor.fit(X_train, y_train)

# --- 5. Оцінка ---
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# --- 6. Важливість ознак та горизонтальна діаграма з підписами ---
feature_importances = regressor.feature_importances_
feature_names = california.feature_names

# Нормалізація (відсотки)
feature_importances = 100.0 * (feature_importances /
                               np.max(feature_importances))

# Сортуємо, щоб найважливіші були зверху
idx = np.argsort(feature_importances)

# Створюємо полотно діаграми
fig = plt.figure(figsize=(8, 6), constrained_layout=True)
# Текст заголовку вікна графіка
fig.canvas.manager.set_window_title("ЛР №3.6. Скицко Руслан 642П")
plt.barh(np.arange(len(idx)), feature_importances[idx], align='center',
         color='skyblue')
# Розставимо мітки вздовж осі Y
plt.yticks(np.arange(len(idx)), np.array(feature_names)[idx])
plt.xlabel('Relative Importance (%)')   # назва осі Х
# заголовок графіка
plt.title(
    'Feature importance using AdaBoost regressor (California housing)')
# Підписуємо числові значення праворуч від смуг
for i, v in enumerate(feature_importances[idx]):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center')
plt.gca().invert_yaxis()  # щоб найважливіші ознаки були зверху
plt.show()
