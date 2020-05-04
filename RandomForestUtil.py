import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

import DataUtil

def GridSearchRandomForest():
    """
    GridSearch template for RandomForest
    :return:classifier , prediction value
    """
    X_train, X_test, y_train, y_test = DataUtil.train_test_split_Mat(0.3)
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split' : [2,3,4,5],
    'max_depth' : [4,5,6,7,8,50,80,100,200],
    'criterion' :['gini', 'entropy']
    }
    CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)
    print("Best params for RandomForest on Mat problem:\n",CV_rfc.best_params_)
    bestrf = UseBestRandomForest(CV_rfc.best_params_)
    bestrf.fit(X_train,y_train)
    y_pred = bestrf.predict(X_test)
    print(f"accuracy score: {accuracy_score(y_test, y_pred)}\n")

    return bestrf, y_pred

def UseBestRandomForest(dictionary):
    """
    get the best randomforest paramters based on GridSearch
    :param dictionary: parameters dictionary
    :return: classifier
    """
    bestrf = RandomForestClassifier(random_state=42, n_estimators=dictionary['n_estimators'],
                                    max_features = dictionary['max_features'],min_samples_split = dictionary['min_samples_split'],
                                    max_depth = dictionary['max_depth'],criterion=dictionary['criterion'])
    return bestrf

def LabelPrediction():
    pass


if __name__ == "__main__":
    GridSearchRandomForest()
