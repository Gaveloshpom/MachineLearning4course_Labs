# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

print("Робота 2.6 \"Класифікація даних про доходи:\"")
print("       за допомогою машин опорних векторів:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Вхідний файл, що містить дані
input_file = 'income_data.txt'

# Прочитайте дані
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворити в масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Створіть класифікатор SVM
classifier = OneVsOneClassifier(LinearSVC(
    dual='auto',max_iter=5000,random_state=0))

# Тренувати класифікатор
classifier.fit(X, y)

# Перехресна перевірка
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC
                   (dual='auto',max_iter=5000,random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Обчисліть оцінку F1 класифікатора SVM
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 оцінка: " + str(round(100*f1.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['38', 'Private', '215646', 'HS-grad', '9', \
    'Never-married', 'Handlers-cleaners', 'Not-in-family', \
    'White', 'Male', '0', '0', '40', 'United-States']

# Закодувати тестову точку даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = \
            int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array([input_data_encoded])

# Запустіть класифікатор на закодованій точці даних і виведіть на друк
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class)[0])