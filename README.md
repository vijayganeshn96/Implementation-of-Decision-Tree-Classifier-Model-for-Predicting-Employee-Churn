# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Start the program
2.Import the numpy,pandas,matplotlib 
3.Read the dataset of student scores
4.Assign the columns hours to x and columns scores to y 
5.From sklearn library select the model to train and test the dataset
6.Plot the training set and testing set in the graph using matplotlib library
7.Stop the program
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIJAY GANESH N
RegisterNumber:  212221040177
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
#### Data head
![decision tree classifier model](https://github.com/vijayganeshn96/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/blob/main/data%20head.png)
#### Null terms
![decision tree classifier model](https://github.com/vijayganeshn96/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/blob/main/is%20null.png)
#### Info of Datas
![decision tree classifier model](https://github.com/vijayganeshn96/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/blob/main/data%20info.png)
#### Predicting x
![decision tree classifier model](https://github.com/vijayganeshn96/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/blob/main/classification.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
