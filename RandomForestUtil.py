import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
import DataUtil as du
from ProjectConstant import *


class MatRandomForest:
    model = None
    modelpath = None
    tickle = None

    def __init__(self, modelpath,tickle,ratio):
        """
        caller for the RandomForestUtil
        :return:
        """
        self.ratio = ratio
        self.tickle = tickle
        self.modelpath = modelpath
        self.model = du.loadModel(modelpath)
        if (self.model == None):
            print("first time usage of RandomForest model,GridSearch for parameters")
            self.model = self.GridSearchRandomForest()
        else:
            print("RandomForest Model Loading Finished..")

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
        print("Call GridSearchRandomForest for building model...")
        # configureable
        global X_train, y_train, X_test, y_test
        if self.tickle == PREDICT_BY_LABEL:
            X_train, X_test, y_train, y_test = self.train_test_split_rf_with_single_label(self.ratio)
        elif self.tickle == PREDICT_BY_COORDINATE:
            X_train, X_test, y_train, y_test = self.train_test_split_rf_with_coordinate(self.ratio)


        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_features': ['auto'],
            'min_samples_split': [2, 5, 10, 30, 50],
            'min_samples_leaf': [10, 20, 30],
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
        :return: dataframe with new column
        """
        X = df.loc[:, 'ap1':'ap6']

        y_pred = self.model.predict(X)
        y_pred_int = np.rint(y_pred)

        if tickle == PREDICT_BY_LABEL:
            # label method one - by directly refract integer from float
            df['rf_label_direct'] = y_pred_int.astype(int)
            # print("Accuracy_Score for rf_label_direct: ", accuracy_score(df['rf_label_direct'], df['label']))
            print("R2_Score for rf_label_direct: ",r2_score(y_pred_int,df.label))

        elif tickle == PREDICT_BY_COORDINATE:
            # label method two - by refract result based on hte predicted coord

            df_y_pred_int = pd.DataFrame(y_pred_int,columns=['x','y'])
            df['rf_label_coord'] = (df_y_pred_int.x // 240) + ((df_y_pred_int.y // 20) - (df_y_pred_int.y) // 20 % 10)
            # rf_label_coord = ((y_pred_int[:, 0] // 240) + ((y_pred_int[:, 1] // 20) - y_pred_int[:, 1] // 20 % 10))
            df['rf_label_coord'].replace([-np.inf,np.inf],np.nan,inplace=True)
            df['rf_label_coord'].fillna(0,inplace=True)
            df['rf_label_coord'] = df['rf_label_coord'].astype(int)
            # print("Accuracy_Score for rf_label_coord: ", accuracy_score(df['rf_label_coord'], df['label']))
            print("R2_Score for rf_label_coord: ", r2_score(y_pred_int, df.loc[:, 'x':'y']))

        return df


def mergeLabeling(df,ratio):
    """
    generate new dataframe based on randomforest
    :param source dataframe
    :return:
    """
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df = df.dropna()
    LabelClassifier = MatRandomForest("./models/rfmodel.sav", PREDICT_BY_LABEL,ratio)
    print("*" * 80)
    print("LabelClassifier Parameters: ",LabelClassifier.model)
    print("*" * 80)
    CoordClassifier = MatRandomForest("./models/rfmodel_coordinate.sav", PREDICT_BY_COORDINATE,ratio)
    print("*" * 80)
    print("CoordClassifier Parameters: ",CoordClassifier.model)
    print("*" * 80)
    df = LabelClassifier.LabelDataByRandomForest(df = df, tickle=PREDICT_BY_LABEL)
    df = CoordClassifier.LabelDataByRandomForest(df = df, tickle=PREDICT_BY_COORDINATE)
    # print(df.head)
    return df

# if __name__ == "__main__":
#     df = pd.read_csv("./newdata/offline_data_random.csv")
#     df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#     df = df.dropna()
#     mergeLabeling(df, 0.3)
