from sklearn.linear_model import LinearRegression

def linearRegression(X_train,y_train):
    # initialise the model
    linear_model = LinearRegression()
    # train the model
    linear_model.fit(X_train,y_train)
    # return the result
    return linear_model