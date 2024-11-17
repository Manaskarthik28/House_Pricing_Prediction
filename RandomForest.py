from sklearn.ensemble import RandomForestRegressor
# define a function to train the model
def random_forest(X_train,y_train):
    # train the model
    rf_model = RandomForestRegressor(n_estimators=50,random_state=42)
    # fit the model
    rf_model.fit(X_train,y_train)
    return rf_model