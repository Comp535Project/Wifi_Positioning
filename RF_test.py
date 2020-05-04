import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        #print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        #print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n===========================================")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        #print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        #print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        
df1 = pd.read_csv("/Users/rongyi/Desktop/newdataset/offline_data_random.csv").astype(np.float32)
df2 = pd.read_csv("/Users/rongyi/Desktop/newdataset/offline_data_uniform.csv").astype(np.float32)
#print(np.isfinite(df1).all())
df2=df2.replace([np.inf, -np.inf], np.nan)
df2=df2.dropna()
X_test = df1.iloc[:,1:7].values
X_train = df2.iloc[:,1:7].values
#print(X_train[:5])
y_test = df1.label
y_train = df2.label
#print(y_test[:5])
#print(type(X_train[1][1]))
#print(type(y_train[1]))
rand_forest1 = RandomForestClassifier()
rand_forest1.fit(X_train,y_train)
print_score(rand_forest1, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest1, X_train, y_train, X_test, y_test, train=False)

rand_forest = RandomForestClassifier(n_estimators=500, max_depth=100,
                                     min_samples_split=2, random_state=2)
rand_forest.fit(X_train, y_train)

print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)





