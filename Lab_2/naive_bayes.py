# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from utilities import visualize_classifier

print("Робота 2.4 \"Наївний байєсівський класифікатор:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Вхідний файл, що містить дані
input_file = 'data_multivar_nb.txt'

# Завантажити дані з вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створіть наївний байєсовський класифікатор
classifier = GaussianNB()

# Тренувати класифікатор
classifier.fit(X, y)

# Прогнозуйте значення для даних навчання
y_pred = classifier.predict(X)

# Точність обчислень
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Точність наївного байєсівського класифікатора =",
      round(accuracy, 2), "%")
# Візуалізуйте продуктивність класифікатора
visualize_classifier(classifier, X, y)

################################################
# Перехресна перевірка
# Split data into training and test data
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Точність нового класифікатора =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(classifier_new, X_test, y_test)

################################################
# Функції оцінювання якості
num_folds = 3
accuracy_values = cross_val_score(classifier,
    X, y, scoring='accuracy', cv=num_folds)
print("Правильність: " + str(round(100*accuracy_values.mean(), 2))
      + "%")

precision_values = cross_val_score(classifier,
    X, y, scoring='precision_weighted', cv=num_folds)
print("Точність: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier,
    X, y, scoring='recall_weighted', cv=num_folds)
print("Зваженість: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier,
    X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")