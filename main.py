import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep= ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "Dalc", "Walc"]]
# data = pd.read_csv("student-mat.csv", sep=";")
# data[["internet"]]=data[["internet"]].replace("yes", 1)
# data[["internet"]]=data[["internet"]].replace("no", 0)
# Label : G3
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

# Getting linear model, train data until it gets to 95% accuracy and save it into pickle file

# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression();
#
#     linear.fit(x_train, y_train)
#
#     # Time 100 to get the acurracy percentage
#     accuracy = linear.score(x_test, y_test)
#     print("Accuracy Percentage: ", format(accuracy, "%"));
#
#     if accuracy > best:
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(x_test[x], round(predictions[x]), y_test[x])

# Plot scatter graph to find correlation between a feature and a label
p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
