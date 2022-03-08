import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


data = pd.read_csv("student-mat.csv", sep= ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "Dalc", "Walc"]]

# Label : G3
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

# Getting linear model
linear = linear_model.LinearRegression();

linear.fit(x_train, y_train)

# Time 100 to get the acurracy percentage
accuracy = linear.score(x_test, y_test)
print("Accuracy Percentage: ", format(accuracy, "%"));

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(x_test[x], round(predictions[x]), y_test[x])
