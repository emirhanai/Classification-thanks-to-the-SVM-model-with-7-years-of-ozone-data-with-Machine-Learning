# **Classification thanks to the SVM model with 7 years of ozone data with Machine Learning**
I developed 2 machine learning software that predict and classify ozone day and non-ozone day. The working principle of the two is similar but there are differences. I got the dataset from ics.icu. Each software has a different mathematical model, Gaussian RBF and Linear Kernel, and classifications are visualized in different ways. I would be happy to present the software to you!

_Example:_ `model_ozone = PCA(n_components=72).fit(X_train)`
 
`model = svm.SVC(kernel='rbf', gamma=0.05, C=3)`

`model_ozone = svm.SVC(kernel='linear', C=3)`

**I am happy to present this software to you!**

`#Accuracy: 0.9602739726027397`

`#Auc Roc Curve Score:  0.5617836676217765`

`#Auc Roc Curve Score:  0.5`

#Linear Confusion Matrix #[[1669    7]
                          [ 115   13]]
#Gauss RBF Confusion Matrix #[[1676    0]
                            [ 128    0]]


Data Source: [DataSource]
###**The coding language used:**

`Python 3.9.6`

###**Libraries Used:**

`Sklearn`

`Pandas`

`Numpy`

`Pylab`

`Matplotlib`
### **Developer Information:**

Name-Surname: **Emirhan BULUT**

Contact (Email) : **emirhan.bulut@turkiyeyapayzeka.com**

LinkedIn : **[https://www.linkedin.com/in/artificialintelligencebulut/][LinkedinAccount]**

[LinkedinAccount]: https://www.linkedin.com/in/artificialintelligencebulut/

Official Website: **[https://www.emirhanbulut.com.tr][OfficialWebSite]**

[OfficialWebSite]: https://www.emirhanbulut.com.tr

[DataSource]: https://archive.ics.uci.edu/ml/index.php

<img src="https://raw.githubusercontent.com/emirhanai/Classification-thanks-to-the-SVM-model-with-7-years-of-ozone-data-with-Machine-Learning/main/ozone-normal-day-classification-WSR0-WSR1.png" alt="ozone-normal-day-classification-WSR0-WSR1">
<img src="https://github.com/emirhanai/Classification-thanks-to-the-SVM-model-with-7-years-of-ozone-data-with-Machine-Learning/blob/main/ozone-normal-day-classification-2.png?raw=trueg" alt="ozone-normal-day-classification-2">
<img src="https://github.com/emirhanai/Classification-thanks-to-the-SVM-model-with-7-years-of-ozone-data-with-Machine-Learning/blob/main/ozone-normal-day-classification-1.png?raw=true" alt="ozone-normal-day-classification-1">
