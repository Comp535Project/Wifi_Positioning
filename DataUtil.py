import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.io as sio
import os
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
    filename = None
    dataset = None

    def __init__(self,filename):
        self.filename = filename
        self.dataset = sio.loadmat(filename)

    def mat_to_csv(self):
        """
        change the format of data to csv and clean data
        :return:
        """
        lenthoffile = len(self.filename.split('_'))
        print(self.filename)
        print(lenthoffile)
        column_ = []
        if lenthoffile > 2:
            columns_ = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'x', 'y']
        else:
            columns_ = ['x', 'y', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6']
        features = pd.DataFrame(list(self.dataset.values())[-1])
        rss = pd.DataFrame(list(self.dataset.values())[-2])
        new_df = pd.concat([features,rss],axis=1)
        new_df.columns = columns_
        # new_df.to_csv(self.filename+'.csv')
        # print(new_df)

        # from 200 to 1750,gap250
        selectedx = self.listrange(200,1750,250,30)
        # from 200 to 1400
        selectedy = self.listrange(200,1400,200,30)

        new_df = new_df[~new_df['x'].isin(selectedx)]
        new_df = new_df[~new_df['y'].isin(selectedy)]

        # plot data
        x = new_df.x
        y = new_df.y
        plt.scatter(x=x, y=y, s=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        return new_df

    def listrange(self,start_,stop_,gap,droprange):
        res = []
        for x in range(start_,stop_,gap):
            for y in range(x-droprange,x+droprange,1):
                res.append(y)
        return res


def createandlistdata():
    filelist = os.listdir('./data')
    for file in filelist:
        filename = file.split('.')[0]
        print(filename)
        df = MatUtil('./data/'+file).mat_to_csv()
        path = r'./newdata/'
        df.to_csv(path+filename+'.csv')

if __name__ == "__main__":
    newdata = MatUtil('./data/online_data.mat')
    # newdata.listrange(10,200,30,3)
    # newdata.mat_to_csv()
    createandlistdata()
    # clean = CleanUtil('trainingData.csv').split_train_test(0.3)
    # clean.drop_data(100, 13).to_csv('trainClean.csv')
    # cleanvalid = CleanUtil('validationData.csv')
    # cleanvalid.drop_data(100,13).to_csv('validationClean.csv')

