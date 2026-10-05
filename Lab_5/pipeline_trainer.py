# Автор Скицко Руслан, 642П, 20 варіант

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# Генерація даних
X, y = make_classification(n_samples=150, 
        n_features=25, n_classes=3, n_informative=6, 
        n_redundant=0, random_state=7)

# Вибір ознак
k_best_selector = SelectKBest(f_regression, k=9)

# Класифікатор
classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4, random_state=7)

# Побудова конвеєра
processor_pipeline = Pipeline([
    ('selector', k_best_selector),
    ('erf', classifier)
])

# Налаштування параметрів
processor_pipeline.set_params(selector__k=7, erf__n_estimators=30)

# Навчання
processor_pipeline.fit(X, y)

# Прогнозувати вихідні дані для вхідних даних
output = processor_pipeline.predict(X)
print("\nPredicted output:\n", output)

# Друк оцінок 
print("\nScore:", processor_pipeline.score(X, y))

# Отримання відібраних ознак
status = processor_pipeline.named_steps['selector'].get_support()

# Вилучення та друк індексів вибраних ознак 
selected_features = [i for i, x in enumerate(status) if x]
print("\nIndices of selected features:", ', '.join(
    [str(x) for x in selected_features]))

# Важливості ознак з моделі
importances = processor_pipeline.named_steps['erf'].feature_importances_

# Рисуємо графік
plt.figure(figsize=(8,5)).canvas.manager.set_window_title("Скицко Руслан, 642П, 20 варіант")
bars = plt.bar(range(len(selected_features)), importances,
               tick_label=selected_features)

# Додаємо підписи над кожним стовпчиком
for bar, importance in zip(bars, importances):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, 
             f"{importance:.2f}", ha='center', va='bottom', fontsize=9)

plt.xlabel("Індекси відібраних ознак")
plt.ylabel("Важливість ознаки")
plt.title("Важливості відібраних ознак у ExtraTreesClassifier")
plt.tight_layout()
plt.show()
