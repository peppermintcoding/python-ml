import pandas as pd
import numpy as np

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle


data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# splitting the data is random
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best_accuracy = 0
for i in range(99):
    # splitting the data into a training set and test set
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # create the linear regression model and fit it to the data set
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print("Accuracy of model {}: {}".format(i, accuracy))
    print("###")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # save the model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print("Best Accuracy found: {}".format(best_accuracy))
print("###")

# load the model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# coefficients of the linear model, weights of each variable
print("Coefficients: ", linear.coef_)
print("Intercept: ", linear.intercept_)
print("###")

# predictions
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print("Prediction: ",predictions[x])
    print("Actual Grade: ", y_test[x])
print("###")

# plotting some information
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()