#DecisionTree Algorithm

    # Balance Scale Dataset

#Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier

#Importing Data
def import_data():
    
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-cale.data" ,sep=','
                     , header=None
                     ,names=["Class Name","Left-Weight","Left-Distance","Right-Weight","Right-Distance"])
    
    #Printing the dataset Shape
    print("Dataset Length" , len(df))
    print("Dataset Shape" , df.shape)
    
    return df    

#Splitting Data
def train_test_split(df):
    from sklearn.model_selection import train_test_split
    x = df.values[:,1:5]
    y = df.values[:, 0]
    
    x_train,y_train,x_test,y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    
    return x,y,x_train,y_train,x_test,y_test

#Training the model using Gini Index criterion
def using_gini(x_train,x_test,y_train):
    
    clf_gini = DecisionTreeClassifier(criterion='gini',max_depth=3, min_samples_leaf=5,random_state=100)
    clf_gini.fit(x_train,y_train)
    
    return clf_gini

#Training the model using Entropy criterion
def using_entropy(x_train,x_test,y_train):
    
    clf_entropy = DecisionTreeClassifier(criterion='entropy',max_depth=3, min_samples_leaf=5,random_state=100)
    clf_entropy.fit(x_train,y_train)
    
    return clf_entropy

#Predicting the Outcome
def prediction(x_test,clf_obj):
    
    y_pred = clf_obj.predict(x_test)
    print(y_pred)
    return y_pred

#Calculating Metrics for the model
def metrics(y_test,y_pred):
    print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
    print("Classification Report: ",classification_report(y_test,y_pred))
    print("Accuracy Score: ",accuracy_score(y_test,y_pred))

#Driver Code
def main():
    
    data = import_data()
    x,y,x_train,x_test,y_train,y_test = train_test_split(data)
    clf_gini = using_gini(x_train,x_test,y_train)
    clf_entropy = using_entropy(x_train,x_test,y_train)
    
    #Prediction using Gini
    print("Results using GINI INDEX")
    y_pred_gini = prediction(x_test,clf_gini)
    metrics(y_test,y_pred_gini)
    
    #Prediction using entropy
    print("Results using ENTROPY")
    y_pred_ent = prediction(x_test,clf_entropy)
    metrics(y_test,y_pred_ent)


if __name__ == "__main__":
    main()

O\P:
    Dataset Length 625
    Dataset Shape (625, 5)
    Results using GINI INDEX
    ['R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'R' 'L'
     'L' 'R' 'L' 'R' 'L' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'L'
     'L' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'R' 'L' 'R'
     'R' 'L' 'R' 'R' 'L' 'L' 'R' 'R' 'L' 'L' 'L' 'L' 'L' 'R' 'R' 'L' 'L' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'R' 'R' 'L' 'R' 'L'
     'R' 'R' 'L' 'L' 'L' 'R' 'R' 'L' 'L' 'L' 'R' 'L' 'R' 'R' 'R' 'R' 'R' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L'
     'L' 'L' 'L' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R'
     'L' 'L' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'R' 'R'
     'L' 'L' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'R' 'R'
     'L' 'R' 'R' 'L' 'L' 'R' 'R' 'R']
    Confusion Matrix:  [[ 0  6  7]
     [ 0 67 18]
     [ 0 19 71]]
    Classification Report:                precision    recall  f1-score   support

               B       0.00      0.00      0.00        13
               L       0.73      0.79      0.76        85
               R       0.74      0.79      0.76        90

       micro avg       0.73      0.73      0.73       188
       macro avg       0.49      0.53      0.51       188
    weighted avg       0.68      0.73      0.71       188

    Accuracy Score:  0.7340425531914894
    Results using ENTROPY
    ['R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'L'
     'L' 'R' 'L' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'L' 'L'
     'L' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'L' 'L' 'R' 'L' 'L' 'R' 'L' 'L'
     'R' 'L' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'L' 'R' 'L' 'L' 'L' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'R' 'R' 'L' 'R' 'L'
     'R' 'R' 'L' 'L' 'L' 'R' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'R' 'R' 'R' 'R' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L'
     'L' 'L' 'L' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R'
     'L' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'R' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'L' 'R'
     'R' 'R' 'L' 'L' 'L' 'R' 'R' 'R']
    Confusion Matrix:  [[ 0  6  7]
     [ 0 63 22]
     [ 0 20 70]]
    Classification Report:                precision    recall  f1-score   support

               B       0.00      0.00      0.00        13
               L       0.71      0.74      0.72        85
               R       0.71      0.78      0.74        90

       micro avg       0.71      0.71      0.71       188
       macro avg       0.47      0.51      0.49       188
    weighted avg       0.66      0.71      0.68       188

    Accuracy Score:  0.7074468085106383
