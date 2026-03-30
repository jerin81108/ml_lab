import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

train = pd.read_csv("temp.csv")
findnull = train.isnull().sum()
print(findnull)
train.dropna(inplace=True)

train = train[['Gender','Married','Education','Self_Employed','Credit_History','Loan_Status']]

train['Gender'] = train['Gender'].replace(to_replace='Male',value='1')
train['Gender'] = train['Gender'].replace(to_replace='Female',value='0')
train['Married'] = train['Married'].replace(to_replace='Yes',value='1')
train['Married'] = train['Married'].replace(to_replace='No',value='0')
train['Self_Employed'] = train['Self_Employed'].replace(to_replace='No',value='0')
train['Self_Employed'] = train['Self_Employed'].replace(to_replace='Yes',value='1')
train['Education'] = train['Education'].replace(to_replace='Graduate',value='1')
train['Education'] = train['Education'].replace(to_replace='Not Graduate',value='0')

X = train.drop(columns=['Loan_Status'])
y = train.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

plt.figure(figsize=(15, 10))
tree.plot_tree(clf, 
               feature_names=['Gender','Married','Education','Self_Employed','Credit_History'], 
               class_names=['Yes','No'], 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.title("Decision Tree (Gini)")
plt.savefig("Gini.png")
plt.show()

output = clf.score(X_test,y_test)
print(f"The Score of the model is: {output}\nPDF")
