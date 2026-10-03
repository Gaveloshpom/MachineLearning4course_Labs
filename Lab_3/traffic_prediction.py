import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# Load input data
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)

data = np.array(data)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

# Extremely Random Forests regressor
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Compute the regressor performance on test data
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error:", round(mae, 2))

# Plot predictions vs actual values
plt.figure(figsize=(10, 5)).canvas.manager.set_window_title("Скицко Руслан, 642П, вар. 20")
plt.plot(y_test[:100], label="Actual", marker='o')
plt.plot(y_pred[:100], label="Predicted", marker='x')
plt.title("Actual vs Predicted Traffic (first 100 test samples)")
plt.xlabel("Sample index")
plt.ylabel("Traffic volume")
plt.legend()
plt.grid(True)
plt.show()

# Plot error distribution (residuals)
errors = y_test - y_pred
plt.figure(figsize=(8, 5)).canvas.manager.set_window_title("Скицко Руслан, 642П, вар. 20")
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Distribution of Prediction Errors (Residuals)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Plot feature importance
feature_names = ["Day", "Time", "City", "Condition"]
importances = regressor.feature_importances_
plt.figure(figsize=(8, 5)).canvas.manager.set_window_title("Скицко Руслан, 642П, вар. 20")
plt.bar(feature_names, importances, color='skyblue', edgecolor='black')
plt.title("Feature Importance in Traffic Prediction")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.grid(True, axis='y')
plt.show()

# Testing encoding on single data instance
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        test_datapoint_encoded[i] = \
            int(label_encoder[count].transform([test_datapoint[i]])[0])
        count = count + 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Predict the output for the test datapoint
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))
