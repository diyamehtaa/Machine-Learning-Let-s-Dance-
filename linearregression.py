import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import splitData

y_train, x_train, y_validation, x_validation, y_test, x_test = splitData.split()

model = LinearRegression()

## Train the model
model.fit(x_train, y_train)


## Make predictions on validation set
score = model.score(x_validation, y_validation)

validation_predictions = model.predict(x_validation)
# validation_mse = 0
# validation_mae = 0
# validation_rms = 0  ## Maybe take this out
# for index in range(len(validation_predictions)):
#     error = y_validation[index] - validation_predictions[index]
#     validation_mse += error**2
#     validation_mae += abs(error)

# validation_mse = validation_mse/len(validation_predictions)
# validation_mae = validation_mae/len(validation_predictions)

validation_mse = mean_squared_error(y_validation, validation_predictions)
validation_mae = mean_absolute_error(y_validation, validation_predictions)


print('MSE: ', validation_mse)
print('MAE: ', validation_mae)
print('score:', score)

