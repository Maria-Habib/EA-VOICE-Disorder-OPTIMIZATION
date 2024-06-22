import pandas as pd
import numpy
import math
from sklearn.model_selection import train_test_split
from math import sqrt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from lightgbm import LGBMClassifier


from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv('data/svd_handcrafted_data.tsv', sep='\t')
df = df.drop(columns=['path'])
print(df.head())

train, test = train_test_split(df, test_size=0.2, stratify=df['label'].values.tolist(), shuffle=True)
print(train.shape)
print(test.shape)


print(train['label'].value_counts())
print(test['label'].value_counts())

train.to_csv('data/svd_train.tsv', sep='\t', index=False)
test.to_csv('data/svd_test.tsv', sep='\t', index=False)

''' 
X_train = train.loc[:, train.columns != 'label']
y_train = train['label'].values.tolist()

X_test = test.loc[:, test.columns != 'label']
y_test = test['label'].values.tolist()

gbm = SVC()


gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
accuracy = accuracy_score(y_test, y_pred)
f1_score1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
print("Accuracy: %.3f%%" % (accuracy * 100.0))
'''
