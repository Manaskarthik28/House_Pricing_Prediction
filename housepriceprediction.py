import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from linearRegression import linearRegression
from RandomForest import random_forest
from KNN import knn
from GradientBoosting import gradient_boost


# read the csv_file
df = pd.read_csv("C:/Users/manas/Downloads/California_housing.csv")
# print the first five rows
print(len(df))
print(df.dtypes)
# check for null values
print(df.isnull().sum())
# remove the rows with null values
df = df.dropna(axis=0)
# print the new rows
print(df.isnull().sum())
# encode ocean_proximity to a numerical value using label encoder
le = LabelEncoder()
# transform our df column 'ocean_proximity'
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])
# check transformed data type
print(df.dtypes)
# define our features for training and predicting
X = df.drop('median_house_value',axis=1)
y = df['median_house_value']
# split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# print the shape
print(X_train.shape,X_test.shape)
# standardize the features
sl = StandardScaler()
# transform on our data split
X_train = sl.fit_transform(X_train)
X_test = sl.fit_transform(X_test)
# step 1. predict using linear regression
lr_model = linearRegression(X_train,y_train)
# make predictions
y_pred = lr_model.predict(X_test)
# print the metrics
mse_lr = mean_squared_error(y_test,y_pred)
mae_lr = mean_absolute_error(y_test,y_pred)
rscore_lr = r2_score(y_test,y_pred)
print("mean squared error is: ",mse_lr)
print("mean absolute error is:" ,mae_lr)
print("variance in the score is: ",rscore_lr)


# step 2 train with random forest model
rf_model = random_forest(X_train,y_train)
y_pred = rf_model.predict(X_test)
# print the metrics
mse_rf = mean_squared_error(y_test,y_pred)
mae_rf = mean_absolute_error(y_test,y_pred)
rscore_rf = r2_score(y_test,y_pred)
print("mean squared error is: ",mse_rf)
print("mean absolute error is:" ,mae_rf)
print("variance in the score is: ",rscore_rf)


# step 3 train with KNN model
knn_moodel = knn(X_train,y_train)
y_pred = knn_moodel.predict(X_test)
# print the metrics
mse_knn = mean_squared_error(y_test,y_pred)
mae_knn = mean_absolute_error(y_test,y_pred)
rscore_knn = r2_score(y_test,y_pred)
print("mean squared error is: ",mse_knn)
print("mean absolute error is:" ,mae_knn)
print("variance in the score is: ",rscore_knn)


# step 4 train with gradient boost model
gb_model = gradient_boost(X_train,y_train)
y_pred = gb_model.predict(X_test)
# print the metrics
mse_gb = mean_squared_error(y_test,y_pred)
mae_gb = mean_absolute_error(y_test,y_pred)
rscore_gb = r2_score(y_test,y_pred)
print("mean squared error is: ",mse_gb)
print("mean absolute error is:" ,mae_gb)
print("variance in the score is: ",rscore_gb)

# save your predictions in a txt or any file
with open('model_results.txt', 'w') as f:
    f.write(f"Linear Regression:\n")
    f.write(f"Mean Squared Error: {mse_lr}\n")
    f.write(f"Mean Absolute Error: {mae_lr}\n")
    f.write(f"R² Score: {rscore_lr}\n\n")

    f.write(f"Random Forest:\n")
    f.write(f"Mean Squared Error: {mse_rf}\n")
    f.write(f"Mean Absolute Error: {mae_rf}\n")
    f.write(f"R² Score: {rscore_rf}\n\n")

    f.write(f"KNN:\n")
    f.write(f"Mean Squared Error: {mse_knn}\n")
    f.write(f"Mean Absolute Error: {mae_knn}\n")
    f.write(f"R² Score: {rscore_knn}\n\n")

    f.write(f"Gradient Boosting:\n")
    f.write(f"Mean Squared Error: {mse_gb}\n")
    f.write(f"Mean Absolute Error: {mae_gb}\n")
    f.write(f"R² Score: {rscore_gb}\n")





