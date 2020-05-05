import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
import DataUtil as du


"""
File Constant
"""
PREDICT_BY_LABEL = 1
PREDICT_BY_COORINATE = 2



class MatRandomForest:
    model = None
    modelpath = None
    tickle = None

    def __init__(self, modelpath,tickle = PREDICT_BY_LABEL):
        """
        caller for the RandomForestUtil
        :return:
        """
        self.tickle = tickle
        self.modelpath = modelpath
        self.model = du.loadModel(modelpath)
        if (self.model == None):
            print("Firsttime usage,GridSearch for parameters")
            self.model = self.GridSearchRandomForest()
        else:
            print("Finished..")

    def train_test_split_rf_with_single_label(self, ratio):
        """
        shared train test split method apply on the mat datatype
        :param ratio:
        :return: a stochatic split data (dataframe)
        """
        df = du.prepare_Mat()
        x = df.loc[:, 'ap1':'ap6']
        # x = pd.concat([x, df.label], axis=1)
        y = df.label
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=0)
        return X_train, X_test, y_train, y_test

    def train_test_split_rf_with_coordinate(self, ratio):
        """
        [x,y] used for y label
        :param ratio:
        :return: similar to above
        """
        df = du.prepare_Mat()
        x = df.loc[:, 'ap1':'ap6']
        y = df.loc[:, 'x':'y']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=0)
        return X_train, X_test, y_train, y_test

    def GridSearchRandomForest(self):
        """
        GridSearch template for RandomForest
        :return:classifier , prediction value
        """

        # configureable
        if self.tickle == PREDICT_BY_LABEL:
            X_train, X_test, y_train, y_test = self.train_test_split_rf_with_single_label(0.3)
        elif self.tickle == PREDICT_BY_COORINATE:
            X_train, X_test, y_train, y_test = self.train_test_split_rf_with_coordinate(0.3)


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
        du.saveModel(bestrf, self.modelpath)
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

    def LabelDataByRandomForest(self, df, tickle = tickle):
        """
        label the csv filepath with randomforest result
        :param tickle:
        :param filpath: the csv file path e.g. ./newdata/offline_data_random.csv
        :return: dataframe with new column
        """
        X = df.loc[:, 'ap1':'ap6']

        y_pred = self.model.predict(X)
        y_pred_int = np.rint(y_pred)

        if tickle == PREDICT_BY_LABEL:
            # label method one - by directly refract integer from float
            df['rf_label_direct'] = y_pred_int.astype(int)
            print("Accuracy_Score for rf_label_direct: ", accuracy_score(df['rf_label_direct'], df['label']))

        elif tickle == PREDICT_BY_COORINATE:
            # label method two - by refract result based on hte predicted coord
            rf_label_coord = ((y_pred_int[:, 0] // 240) + ((y_pred_int[:, 1] // 20) - y_pred_int[:, 1] // 20 % 10))
            df['rf_label_coord'] = rf_label_coord.astype(int)
            print("R2_Score for rf_label_coord: ",r2_score(y_pred_int,df.loc[:,'x':'y']))
            print("Accuracy_Score for rf_label_coord: ", accuracy_score(df['rf_label_coord'], df['label']))

        return df



def mergeLabeling(filepath):
    """
    filepath is the target file path
    :param filepath: e.g. './newdata/offline_data_random.csv'
    :return:
    """
    df = pd.read_csv(filepath)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df = df.dropna()
    LabelClassifier = MatRandomForest("./models/rfmodel.sav", PREDICT_BY_LABEL)
    CoordClassifier = MatRandomForest("./models/rfmodel_coordinate.sav", PREDICT_BY_COORINATE)
    df = LabelClassifier.LabelDataByRandomForest(df = df, tickle=PREDICT_BY_LABEL)
    df = CoordClassifier.LabelDataByRandomForest(df = df, tickle=PREDICT_BY_COORINATE)
    print(df)
    return df

if __name__ == "__main__":
    mergeLabeling('./newdata/offline_data_random.csv')
