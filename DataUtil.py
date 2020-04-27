import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


class CleanUtil:
<<<<<<< HEAD
    baseURI = 'D:/RiceClass/535/Indoor/UJIndoorLoc/'
=======
    baseURI = ''
    filename = ''
    df = None
>>>>>>> 6a07f11a4c6f6ad205fc82ce267dfd54863702b8

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
        x = preprocessing.normalize(x, norm='l2')

        X, y = x, y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)
        # print(type(X_train))
        # print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        return X_train, X_test, y_train, y_test


    def drop_data(self, columncount, rowcount):
        """
        drop by column or row with understanding of row data
        :return:
        """
        columnaggr = self.df.iloc[:, :520].where(self.df != 100).count()
        columnindex = columnaggr[columnaggr > columncount].index.tolist()
        columnindex = columnindex + (self.df.iloc[:, 521:].columns.tolist())
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

#
# if __name__ == "__main__":
#     clean = CleanUtil('trainingData.csv')
#     clean.drop_data(100, 13)
