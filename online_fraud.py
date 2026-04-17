import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("dataset.csv")
print(data.head())
print(data.type.value_counts())

type_counts = data["type"].value_counts()
transactions = type_counts.index
quantity = type_counts.values

figure = px.pie(values=quantity, names=transactions, hole=0.5, title="Distribution of Transaction Type")
figure.show()

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
