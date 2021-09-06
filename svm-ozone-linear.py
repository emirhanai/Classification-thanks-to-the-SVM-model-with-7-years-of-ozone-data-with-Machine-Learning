from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import pandas as pd
#from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('ozone-data.csv')
#print(df.head())

#data select
X = df.iloc[:,1:73]
#print(X)

#target select [class]
y = df.iloc[:,73:74]
#print(y)

#dataframe to numpy array
yyy = np.array(y).ravel()
XXX = np.array(X).reshape(2534,-1)
#print(target)
#print(connects)

#target_variables = LabelEncoder()

#y['Date_data'] = target_variables.fit_transform(y['Date'])

#y_n = y.drop(['Date'],axis=1)

#print("Features: ", X)

#print("Target: ", y_n)

X_train, X_test, y_train, y_test = train_test_split(XXX,yyy,test_size=0.2880820836621942,random_state=49) # 80% training and 20% test

#creating of SVM model
model_ozone = svm.SVC(kernel='linear', C=3) # Linear Kernel

#Train the model using the training sets
model_ozone.fit(X_train, y_train)

#Predict the response for test dataset
y_test_pred = model_ozone.predict(X_test)
#print(X_test)
#print(y_test)

y_train_pred = model_ozone.predict(X_train)

from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_train, y_train_pred)
print(c_matrix)
#[[1669    7]
# [ 115   13]]

#C = 1.0

#h = .02

#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
#lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)


#print(y_pred)


print("Accuracy:",metrics.accuracy_score(y_test,y_test_pred))
#Accuracy: 0.9561643835616438

import matplotlib.pyplot as plt

#plt.scatter(X_train[:,0], X_train[:,1])
#plt.title('Linearly separable data')
#plt.xlabel('X1')
#plt.ylabel('X2')
#plt.show()

support_vectors = model_ozone.support_vectors_

# Visualize support vectors
plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Ozone Day and Normal Day Prediction Software')
plt.xlabel('Data')
plt.ylabel('Class [Ozone Day = 1, Normal Day = 0]')
plt.show()

