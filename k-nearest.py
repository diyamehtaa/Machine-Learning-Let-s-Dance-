from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import mean_squared_error
#import numpy as np
import pandas as pd
import splitData
from sklearn.neighbors import KNeighborsRegressor

y_train, x_train, y_validation, x_validation, y_test, x_test = splitData.split()

# normalizing data since knn relies on distances
scaler = MinMaxScaler()
scaler.fit(x_train)
normalized_x_train = scaler.transform(x_train)
normalized_x_validation = scaler.transform(x_validation)
normalized_x_test = scaler.transform(x_test)

# tuning k
mse_values = []
for k in range(3, 21):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(normalized_x_train, y_train)
    validation_predictions = model.predict(normalized_x_validation)
    validation_mse = mean_squared_error(y_validation, validation_predictions)
    mse_values.append(validation_mse)
k_value = mse_values.index(min(mse_values)) + 3

model = KNeighborsRegressor(n_neighbors=k_value)
final_x_train = pd.concat([normalized_x_train, normalized_x_validation],
                          ignore_index=True)
final_y_train = pd.concat([y_train, y_validation], ignore_index=True)
model.fit(final_x_train, final_y_train)
test_predictions = model.predict(normalized_x_test)
test_error = mean_squared_error(test_predictions, y_test)

print("MSE is", test_error)

# dist = manhattan_distances(normalized_x_train, normalized_x_validation)
# k_nearest_ind = np.argpartition(dist, axis=0)
# sorted_y_train = y_train[k_nearest_ind] #not cooking here

# for k in k_values:
#     validation_predictions = np.mean(sorted_y_train[:k]) #not cooking here
#     validation_mse = mean_squared_error(y_validation, validation_predictions)
#     mse_values.append(validation_mse)

# def predict_new_value(sample):
#     '''sample is a vector of the features of the new data point'''
#     dist = manhattan_distances(normalized_train, sample)
#     sorted_dist_indices = np.argsort(dist, axis=1)
#     sorted_y_train = y_train[sorted_dist_indices]
    
#     k_value = k_values[mse_values.index(min(mse_values))]
#     sample_prediction = np.mean(sorted_y_train[:k_value])
#     return sample_prediction



