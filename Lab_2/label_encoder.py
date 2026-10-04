# Автор Скицко Руслан гр.642П
# Лабораторна робота 2

from sklearn import preprocessing

print("Робота 2.2 \"Кодування міток:\"")
print("Виконав ст. 642п гр. Скицко Руслан, вар. 20")

# Зразки вхідних міток
input_labels = ['red', 'black', 'red', 'green', 'black',
                'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

print("\nКартування міток:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nМітки =", test_labels)
print("Закодовані значення =", encoded_values)

encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nЗакодовані значення =", encoded_values)
print("Декодовані мітки =", decoded_list)