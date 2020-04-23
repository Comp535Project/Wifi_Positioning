import pandas as pd
from sklearn.model_selection import train_test_split
class CleanUtil:
    baseURI = 'D:/RiceClass/535/Indoor/UJIndoorLoc/'
    filename = ''
    df = None

    def __init__(self, filename):
        self.filename = self.baseURI + filename
        self.df = pd.read_csv(self.filename)

    def split_train_test(self, ratio):
        """
        refract data
        :return: train_x,tran_y,test_x,test_y
        """
        X,y = self.cleandata()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)
        # print(type(X_train))
        # print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    def cleandata(self):
        """
        from dataframe
        clean dropna ,drop dup,analysize data(drop unnecessary column)
        form vector
        :return: vector representation
        """
        if self.df is None:
            return None
        x = self.df.drop(['SPACEID'], axis=1)
        y = self.df.loc[:, 'SPACEID']
        x.fillna(axis=0, method='ffill', inplace=True)
        y.fillna(axis=0, method='ffill', inplace=True)
        # print(x.dropna())
        # print(y.head())
        return x, y

    def __str__(self):
        shapetemplate = "This file has shape of {} \n" \
                        "the info is :\n {} \n"
        headertemplate = "The header of file includes: \n {0}"
        template = shapetemplate.format(self.df.shape, self.df.info(verbose=True)) + headertemplate.format(self.df.head())
        return template

    def __repr__(self):
        return self.__str__(self)


# if __name__ == "__main__":
#     train = CleanUtil('trainingData.csv')
#     # print(train)
#     train.cleandata()
#     # train.split_train_test(0.3)