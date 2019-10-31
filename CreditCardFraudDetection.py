import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import sklearn as sk
data=pd.read_csv(r"C:\Users\Lenovo\Downloads\creditcardfraud\creditcard.csv")
print(data.columns)
print(data.shape)
print(data.describe())
data=data.sample(frac=0.1,random_state=1)
print(data.shape)
##data.hist(figsize=(20,20))
##plt.show()
Fraud=data[data['Class']== 1]
Valid=data[data['Class']==0]
outlier_fraction=len(Fraud)/float(len(Valid))
print("Fraud Cases:{}",len(Fraud))
print("Valid Cases:{}",len(Valid))
print("Outlier Fraction:{}",outlier_fraction)
##corrmat=data.corr()
##fig=plt.figure(figsize=(12,9))
##sns.heatmap(corrmat,vmax=0.8,square=True)
##plt.show()
columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
X=data[columns]
Y=data[target]
print(X.shape)
print(Y.shape)
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state=1
classifiers={"Isolation Forest":IsolationForest(max_samples=len(X),
                                    contamination=outlier_fraction,
                                    random_state=state),
"Local Outlier Factor":LocalOutlierFactor(
n_neighbors=20,
contamination=outlier_fraction
)}
n_outliers=len(Fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if(clf_name=="Local Outlier Factor"):
        y_pred=clf.fit_predict(X)
        scores_predict=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors=(y_pred!=Y).sum()
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
























