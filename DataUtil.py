import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import numpy as np
import h5py
import pickle
from ProjectConstant import *
class KaggleDataUtil:
    def __init__(self, filename = None):
        if filename != None:
            self.filename = './' + filename
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
        x = self.df.loc[:, 'ap1':'ap6']
        y = self.df.loc[:, 'label']
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
    df = None
    def __init__(self,filename = None):
        if filename != None:
            self.filename = filename
            self.dataset = sio.loadmat(filename)

    def mat_to_csv(self):
        """
        change the format of data to csv and clean data
        :return: csv file
        """
        lenthoffile = len(self.filename.split('_'))
        print("Processing :",self.filename)
        # print(lenthoffile)
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
        selectedx = self.listrange(200,2000,250,30)
        # from 200 to 1400
        selectedy = self.listrange(200,2000,200,30)

        new_df = new_df[~new_df['x'].isin(selectedx)]
        new_df = new_df[~new_df['y'].isin(selectedy)]

        new_df = labeldata(new_df)

        # print(new_df.label)

        # plot data
        # SimpleVisulizeCoord(new_df)

        return new_df

    def mat_to_csv_float32(self):
        """
        change the format of data to csv and clean data
        :return:
        """
        lenthoffile = len(self.filename.split('_'))
        print("Processing :",self.filename)
        # print(lenthoffile)
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
        selectedx = self.listrange(200,2000,250,30)
        # from 200 to 1400
        selectedy = self.listrange(200,2000,200,30)

        new_df = new_df[~new_df['x'].isin(selectedx)]
        new_df = new_df[~new_df['y'].isin(selectedy)]

        new_df = labeldata(new_df)

        # print(new_df.label)

        # plot data
        # self.SimpleVisulizeCoord(new_df)
        new_df = new_df.replace([np.inf, -np.inf], np.nan)
        new_df = new_df.dropna()
        return new_df.astype(np.float32)

    def listrange(self,start_,stop_,gap,droprange):
        """
        Used to split the raw data into range
        :param start_: start point of the whole range
        :param stop_:  end point of the whole range
        :param gap:     every gap drop some data
        :param droprange: for each point we drop the data across it with range = droprange
        :return:  droprange list
        """
        res = []
        for x in range(start_,stop_,gap):
            for y in range(x-droprange,x+droprange,1):
                res.append(y)
        return res

def SimpleVisulizeCoord(new_df,based_label):
    """
    visualize and color the dataframe
    :param new_df:
    :return: void
    """
    x = new_df.x
    y = new_df.y
    if based_label == ORIGINAL_LABEL:
        plt.scatter(x=x, y=y, s=3, c=new_df.label)
    elif based_label == PREDICT_BY_LABEL:
        plt.scatter(x=x, y=y, s=3, c=new_df.rf_label_direct)
    elif based_label == PREDICT_BY_COORDINATE:
        plt.scatter(x=x, y=y, s=3, c=new_df.rf_label_coord)
    plt.title(based_label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def labeldata(dataframe):
    """
    label data in a range with the same label
    :param dataframe: the source dataframe should be labeled
    :return:    dataframe with new label column
    """
    dataframe['label'] = (dataframe.x//240) + ((dataframe.y//20) - (dataframe.y)//20%10)
    # print((dataframe.y)//20%10)
    # dataframe['label'] = (dataframe.x//480) + ((dataframe.y//40) - (dataframe.y)//40%10)
    return dataframe

def prepare_Mat():
    """
    shared method apply on the mat datatype
    :return: dataframe finsihed splitting
    """
    df = MatUtil(r'./data/offline_data_uniform.mat').mat_to_csv()
    # df = df.sample(frac=0.5,replace=True,random_state=1)
    df = pd.concat([df, MatUtil(r'./data/offline_data_random.mat').mat_to_csv()], axis=0)
    # df = df.sample(frac=0.5,replace=True,random_state=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df = df.dropna()
    df = df.sample(frac=0.3,replace=True,random_state=1)

    SimpleVisulizeCoord(df,ORIGINAL_LABEL)
    print("prepare_Mat size: ",df.shape)

    return df

def createandlistdata():
    """
    used to batch generate new data from mat to csv
    :return:
    """
    filelist = os.listdir('./data')
    for file in filelist:
        filename = file.split('.')[0]
        # print(filename)
        df = MatUtil('./data/'+file).mat_to_csv()
        path = r'./newdata/'
        df.to_csv(path+filename+'.csv')


def saveModel(model,filepath):
    """
    serialize model with pickle
    :param Object: model
    :param filepath: save path
    :return: void
    """
    pickle.dump(model,open(filepath,'wb'))

def loadModel(filepath):
    """
    deserialize model with pikle -  sklearn model
    :param filepath: load path
    :return: model obj
    """
    try:
        file = open(filepath,'rb')
        return pickle.load(open(filepath,'rb'))
    except FileNotFoundError as fnfe:
        print("Filenot found at DataUtil.loadModel",fnfe.errno)
        return None
    except IOError as ioe:
        print("IO error happens at DataUtil.loadModel",ioe)
        return None
    except Exception as e:
        print("Something happens at DataUtil.loadModel",e)
        return None



# if __name__ == "__main__":
#     createandlistdata()
    # newdata = MatUtil('./data/online_data.mat')
    # newdata.listrange(10,200,30,3)
    # newdata.mat_to_csv()

    # clean = CleanUtil('trainingData.csv').split_train_test(0.3)
    # clean.drop_data(100, 13).to_csv('trainClean.csv')
    # cleanvalid = CleanUtil('validationData.csv')
    # cleanvalid.drop_data(100,13).to_csv('validationClean.csv')

