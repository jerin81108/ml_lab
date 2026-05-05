# EXP 2A - SVM Algorithm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Result
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("Actual   :", list(y_test))
print("Predicted:", list(y_pred))
