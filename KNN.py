from sklearn.neighbors import KNeighborsClassifier
# define function for KNN
def knn(X_train,y_train):
    # train the model
    knn_model = KNeighborsClassifier()
    # train the model
    knn_model.fit(X_train,y_train)
    # return the model
    return knn_model