# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("Робота 2.5 \"Матриця неточностей:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Визначте зразки міток
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Створіть матрицю неточностей
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Візуалізуйте матрицю неточностей
# fig = plt.figure()
plt.figure().canvas.manager.set_window_title("Скицко Руслан 642П")
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Матриця неточностей', color='red')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('Правдиві мітки')
plt.xlabel('Прогнозовані мітки')
plt.show()

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels,
    target_names=targets))