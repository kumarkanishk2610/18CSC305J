from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

iris = datasets.load_iris()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

# Sepal Plot
plt.scatter(iris.data[:,:1], iris.data[:,1:2], c=iris.target, cmap=plt.cm.Dark2)
plt.title('Sepal plot')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

# Petal Plot
plt.scatter(iris.data[:,2:3], iris.data[:,3:4], c=iris.target, cmap=plt.cm.Dark2)
plt.title('Petal plot')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

# Getting classifier using k = 9 and trained with training dataset
knn = KNeighborsClassifier(9)
knn.fit(X_train, y_train)

# Now testing and check the accuracy at k = 9
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

pred = knn.predict(X_test)
print(accuracy_score(y_test, pred))

# Define a function to compute accuracy and error for different values of K
def compute(x_input, y_input, x_test):
    index = []
    accuracy = []
    error = []
    for K in range(30):
        K = K+1
        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(x_input, y_input)
        y_pred = neigh.predict(x_test)
        index.append(K)
        accuracy.append(accuracy_score(y_test, y_pred) * 100)
        error.append(mean_squared_error(y_test, y_pred) * 100)
    plt.subplot(2, 1, 1)
    plt.plot(index, accuracy)
    plt.title('Accuracy')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(index, error, 'r')
    plt.title('Error')
    plt.xlabel('Value of K')
    plt.ylabel('Error')
    plt.show()

compute(X_train, y_train, X_test)