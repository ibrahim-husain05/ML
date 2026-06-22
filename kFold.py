from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
iris=load_iris()
X=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model=LogisticRegression()
model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
#SVM
model_svm=svm.SVC()
model_svm.fit(X_train,y_train)  
# print(model_svm.score(X_test,y_test))
#Random Forest      
model_rf=RandomForestClassifier()
model_rf.fit(X_train,y_train)
# print(model_rf.score(X_test,y_test))
#KFold
kf=KFold(n_splits=3)    
for train_index, test_index in kf.split(X):
   print("Train Index:", train_index, "Test Index:", test_index)
from sklearn.model_selection import cross_val_score
cor=cross_val_score(model,X,y,cv=3)
print(cor)
cor_svm=cross_val_score(model_svm,X,y,cv=3)
print(cor_svm)  
cor_rf=cross_val_score(model_rf,X,y,cv=3) 
print(cor_rf)