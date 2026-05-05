# K-Nearest Neighbours (KNN) Algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict and print result
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred) * 100, "%")
print("Predicted:", list(y_pred))
print("Actual   :", list(y_test))
