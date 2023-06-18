import pandas as pd
import numpy as np
import sklearn


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# training
linear = linear_model.LinearRegression()
# linear.fit()

