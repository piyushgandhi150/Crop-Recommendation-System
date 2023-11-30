# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:00:19 2023

@author: Pratesh Mishra
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:49:59 2023

@author: Pratesh Mishra
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\major project\data.csv")
x=df.drop(['Label'],axis=1)
y=df['Label']
print(y.head())
print(x.head())
scaler=StandardScaler()
xtrain=scaler.fit_transform(x)
xtrain=pd.DataFrame(xtrain,columns=x.columns)
xtrain.head()
x_train, x_test, y_train, y_test = train_test_split(xtrain, y, test_size=0.2, random_state=42)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
clf_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
clf_1.fit(x_train,y_train)
import pickle
pickle.dump(clf_1,open('Model_ensumble.pkl','wb'))
import os
os.getcwd()