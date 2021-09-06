from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import pandas as pd
#from sklearn.preprocessing import LabelEncoder

#read data
df = pd.read_csv('ozone-data.csv')
#print(df.head())

#data select
X = df.iloc[:,1:73]
print(X)
#target select [class]
y = df.iloc[:,73:74]
print(y)
#target type of Dataframe to type of Numpy Array
yyy = np.array(y).ravel()

#train,test of creating
X_train, X_test, y_train, y_test = train_test_split(X,yyy,test_size=0.2880820836621942,random_state=49) # 80% training and 20% test


#creationg pca model
model_ozone = PCA(n_components=72).fit(X_train)
model_ozone_2d = model_ozone.transform(X_train)

#print(model_ozone_2d)e


#creating of model the SVM
model = svm.SVC(kernel='rbf', gamma=0.05, C=3)

#The model is training in equation
model.fit(model_ozone_2d, y_train)

#prediction equation is the model.
y_test_pred = model.predict(X_test)

#accuracy is model
print("Accuracy:",metrics.accuracy_score(y_test,y_test_pred))
#Accuracy: 0.9602739726027397

#data visualition
import pylab as pl
for i in range(0, model_ozone_2d.shape[0]):
    if y_train[i] == 0:
        c1 = pl.scatter(model_ozone_2d[i,0],model_ozone_2d[i,1],color='r',edgecolors='y',marker='*',linewidths=1)

    elif y_train[i] == 1:
        c2 = pl.scatter(model_ozone_2d[i,0],model_ozone_2d[i,1],color='g',edgecolors='y',marker='o',linewidths=1)
import matplotlib.pyplot as plt
pl.legend([c1, c2], ['Ozone Day', 'Normal Day'])
plt.title('Ozone and Normal Day Classification')
pl.show()