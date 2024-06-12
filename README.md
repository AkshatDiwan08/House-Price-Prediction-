House Price Prediction
This repository contains a machine learning project focused on predicting house prices. The project includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.


Introduction :
House price prediction is a critical task in the real estate industry, helping stakeholders make informed decisions. This project utilizes machine learning algorithms to predict house prices based on various features such as the number of bedrooms, location, square footage, and more.

Dataset :
The dataset used in this project is USA_Housing.csv. It contains the following features:

Avg. Area Income
Avg. Area House Age
Avg. Area Number of Rooms
Avg. Area Number of Bedrooms
Area Population
Price (target variable)

Usage :
(i) Data Preprocessing: Prepare the dataset for modeling by running the data preprocessing scripts.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/USA_Housing.csv')

(ii) Exploratory Data Analysis: Perform EDA to understand the data and visualize key aspects.

sns.pairplot(df)
plt.show()

(iii) Model Training: Train the machine learning models.


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

(iv) Model Evaluation: Evaluate the performance of the trained models.

from sklearn.metrics import mean_absolute_error, mean_squared_error
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')


Results
The results of the model training and evaluation are documented in the results folder. Key performance metrics include:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)
Plots and visualizations are provided to interpret the model performance and feature importance.



Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

1.Fork the Project
2.Create your Feature Branch (git checkout -b feature/YourFeature)
3.Commit your Changes (git commit -m 'Add Your Feature')
4.Push to the Branch (git push origin feature/YourFeature)
5.Open a Pull Request


License
This project is licensed under the MIT License - see the LICENSE file for details.
 ​
