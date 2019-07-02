import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

file = "iris.csv"
names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pandas.read_csv(file,names=names)
#print(dataset.shape)
#print(dataset.head(30))
#print(dataset.describe())
#print(dataset.groupby('class').size())
array=dataset.values
#keeping aise 20% of data for testing(validation dataset) and rest as training set
X=array[:,0:4] #first 4 columnsa
Y=array[:,4]   #5th col
validation_size=0.20 #20-80
seed=7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring='accuracy'
# just for checking accuracy
name='LR' #logistic regression
model=LogisticRegression(solver='liblinear', multi_class='ovr')
kfold = model_selection.KFold(n_splits=10, random_state=seed)# 10 fold cross validation-----
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

#making predictions on validation dataset
lr=LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train,Y_train)
pred=lr.predict(X_validation)
print(accuracy_score(Y_validation, pred))
print(confusion_matrix(Y_validation, pred))
print(classification_report(Y_validation, pred))
