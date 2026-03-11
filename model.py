import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import streamlit as st 

df=pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\delivery_delay.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())

X = df.drop("Delivery_Delay", axis=1)
y = df["Delivery_Delay"]

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)

print(model.predict([[23,5,3,6,6,5,4,6,84,65,145]]))

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues') 
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("confusion matrix") 
plt.show()

y_pred_proba = model.predict_proba(X_test)[:, 1] 
fpr,tpr,thresholds=roc_curve(y_test,y_pred_proba)
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='red', linestyle='--', label='Random Guess') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

recall=recall_score(y_test,y_pred)
print("recall score:",recall)

f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

precision=precision_score(y_test,y_pred)
print("precision_score",precision)

st.title("Delivery delay predictor")
distance = st.number_input("Delivery Distance")
traffic = st.number_input("Traffic Congestion")
weather = st.number_input("Weather Condition")
slot = st.number_input("Delivery Slot")
experience = st.number_input("Driver Experience")
stops = st.number_input("Number of Stops")
vehicle_age = st.number_input("Vehicle Age")
road_score = st.number_input("Road Condition Score")
weight = st.number_input("Package Weight")
fuel = st.number_input("Fuel Efficiency")
warehouse_time = st.number_input("Warehouse Processing Time")

if st.button("Predict"):
    prediction = model.predict([[distance,traffic,weather,slot,experience,
                                 stops,vehicle_age,road_score,
                                 weight,fuel,warehouse_time]])
    st.write("Prediction:", prediction)







