import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from DataUtil import prepare_Mat
from DataUtil import saveModel
from DataUtil import loadModel


class MatRandomForest:
    model = None

    def __init__(self):
        """
        caller for the RandomForestUtil
        :return:
        """
        self.model = loadModel("./models/rfmodel.sav")
        if (self.model == None):
            print("Firsttime usage,GridSearch for parameters")
            self.model = self.GridSearchRandomForest()
        else:
            print("Finished..")

    def train_test_split_rf(self, ratio):
        """
        shared train test split method apply on the mat datatype
        :param ratio:
        :return: a stochatic split data (dataframe)
        """
        df = prepare_Mat()
        x = df.loc[:, 'ap1':'ap6']
        # x = pd.concat([x, df.label], axis=1)
        y = df.label
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=0)
        return X_train, X_test, y_train, y_test

    def GridSearchRandomForest(self):
        """
        GridSearch template for RandomForest
        :return:classifier , prediction value
        """
        X_train, X_test, y_train, y_test = self.train_test_split_rf(0.3)
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [10, 60, 100],
            'max_features': ['auto'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 10],
            'criterion': ['mse']
        }
        CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        CV_rfc.fit(X_train, y_train)
        print("Best params for RandomForest on Mat problem:\n", CV_rfc.best_params_)
        bestrf = self.UseBestRandomForest(CV_rfc.best_params_)
        bestrf.fit(X_train, y_train)
        y_pred = bestrf.predict(X_test)
        # print("label prediction is :",y_pred)
        print(f"r2 score: {r2_score(y_test, y_pred, multioutput='uniform_average')}\n")
        saveModel(bestrf, "./models/rfmodel.sav")
        return bestrf

    def UseBestRandomForest(self, dictionary):
        """
        get the best randomforest paramters based on GridSearch
        :param dictionary: parameters dictionary
        :return: classifier
        """
        bestrf = RandomForestRegressor(random_state=42, n_estimators=dictionary['n_estimators'],
                                       max_features=dictionary['max_features'],
                                       min_samples_split=dictionary['min_samples_split'],
                                       min_samples_leaf=dictionary['min_samples_leaf'],
                                       criterion=dictionary['criterion'])
        return bestrf

    def LabelDataByRandomForest(self, filpath):
        """
        label the csv filepath with randomforest result
        :param filpath: the csv file path e.g. ./newdata/offline_data_random.csv
        :return: dataframe with new column
        """
        df = pd.read_csv(filpath)
        X = df.loc[:, 'ap1':'ap6']
        # X_train, X_test, y_train, y_test = self.train_test_split_rf(0.3)
        # self.model.fit(X_train, y_train)
        # saveModel(self.model,"./models/rfmodel.sav")
        # print(self.model)
        y_pred = self.model.predict(X)
        y_pred_int = np.rint(y_pred)
        df['rf_label_direct'] = y_pred_int.astype(int)
        return df


if __name__ == "__main__":
    rfr = MatRandomForest()
    rfr.LabelDataByRandomForest('./newdata/offline_data_random.csv')
