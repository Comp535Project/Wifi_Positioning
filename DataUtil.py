import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import h5py

class CleanUtil:
    baseURI = 'D:/RiceClass/535/Indoor/UJIndoorLoc/'
    def __init__(self, filename):
        self.filename = self.baseURI + filename
        self.df = pd.read_csv(self.filename)

    def split_train_test(self, ratio):
        """
        refract data
        from dataframe
        clean dropna ,drop dup,analysize data(drop unnecessary column)
        form vector
        :return: train_x,tran_y,test_x,test_y
        """
        if self.df is None:
            return None
        self.df = self.drop_data(100, 13)
        x = self.df.drop(['SPACEID'], axis=1)
        y = self.df.loc[:, 'SPACEID']
        x.fillna(axis=0, method='ffill', inplace=True)
        y.fillna(axis=0, method='ffill', inplace=True)
        # x = preprocessing.normalize(x, norm='l2')
        # print(x)
        # TODO:NORMALIZE following https://www.kaggle.com/sunnerli/train-some-simple-model-and-print-the-error/code

        X, y = x, y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)
        # print(type(X_train))
        # print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        return X_train, X_test, y_train, y_test

    def normalize_data(self):
        pass


    def drop_data(self, columncount, rowcount):
        """
        drop by column or row with understanding of row data
        :return:
        """
        columnaggr = self.df.iloc[:, :520].where(self.df != 100).count()
        columnindex = columnaggr[columnaggr > columncount].index.tolist()
        columnindex = columnindex + (self.df.iloc[:, 520:].columns.tolist())
        filteredcolumn = self.df.loc[:, columnindex]
        rowaggr = filteredcolumn.where(self.df != 100).T.count()
        rowindex = rowaggr[rowaggr > rowcount].index.tolist()
        filterrow = filteredcolumn.loc[rowindex]
        # print(filterrow)

        # self.plot_drop_data(filteredcolumn,filterrow)
        # print(rowaggr.describe())
        return filterrow

    def plot_drop_data(self, columnres, rowres):
        """
        plot result in drop data step
        :param columnres: plot the result from selection based on column
        :param rowres: plot the result from selecting based on row
        :return: two subplot to visualize the distribution of data
        """
        columnres = columnres.where(columnres != 100).count()
        rowres = rowres.where(rowres != 100).T.count()
        plt.subplot(2, 1, 1)
        columnres.plot()
        plt.title("Aggregrated by column")
        plt.subplot(2, 1, 2)
        plt.title("Aggregrated by row")
        rowres.plot()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()
        # print(rowres)

    def __str__(self):
        shapetemplate = "This file has shape of {} \n" \
                        "the info is :\n {} \n"
        headertemplate = "The header of file includes: \n {0}"
        template = shapetemplate.format(self.df.shape, self.df.info(verbose=True)) + headertemplate.format(self.df.head())
        return template

    def __repr__(self):
        return self.__str__(self)


class MatUtil:

    def __init__(self,filename):
        self.filename = filename
        self.dataset = sio.loadmat(filename)

    def mat_to_csv(self):
        columns_ = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'x', 'y']
        features = pd.DataFrame(list(self.dataset.values())[-1])
        rss = pd.DataFrame(list(self.dataset.values())[-2])
        new_df = pd.concat([features,rss],axis=1)
        new_df.columns = columns_
        new_df.to_csv(self.filename+'.csv')
        # print(new_df)

if __name__ == "__main__":
    newdata = MatUtil('./data/online_data.mat')

    newdata.mat_to_csv()
    # clean = CleanUtil('trainingData.csv').split_train_test(0.3)
    # clean.drop_data(100, 13).to_csv('trainClean.csv')
    # cleanvalid = CleanUtil('validationData.csv')
    # cleanvalid.drop_data(100,13).to_csv('validationClean.csv')

