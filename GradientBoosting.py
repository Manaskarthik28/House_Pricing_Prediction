from sklearn.ensemble import GradientBoostingRegressor
# define a function
def gradient_boost(X_train,y_train):
    # train the model
    gb_model = GradientBoostingRegressor()
    # fit the model
    gb_model.fit(X_train,y_train)
    return gb_model