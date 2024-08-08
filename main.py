import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv("diabetes.csv")
diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()

# 0 means no diab, 1 means diab uwu
diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns ='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
scaler = StandardScaler()
scaler.fit(X)
standirazed_data = scaler.transform(X)
print (standirazed_data)
X = standirazed_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, stratify = Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


X_train_prediction = classifier.predict(X_train)
training_data_accuracy =  accuracy_score(X_train_prediction,Y_train )
print("Accuracy score of the training data : ", training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy =  accuracy_score(X_test_prediction,Y_test )
print("Accuracy score of the testing data : ", test_data_accuracy)

input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_npr = np.asarray(input_data)

input_data_reshaped = input_data_npr.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0) :
    print ('the person doesnt have diabetes')
else :
    print("the has diabtes sadly")

#saving the trained model

import pickle

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_npr = np.asarray(input_data)

input_data_reshaped = input_data_npr.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0) :
    print ('the person doesnt have diabetes')
else :
    print("the person has diabtes sadly")
