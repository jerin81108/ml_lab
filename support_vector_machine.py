import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("spam.csv")
print(data.head())
print(data.tail())
print(data.info())

X = data['EmailText'].values
y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

classifier = SVC(kernel='rbf', random_state=10)
classifier.fit(X_train, y_train)
print(classifier.predict(X_test))
print("Score: ", classifier.score(X_test, y_test))
