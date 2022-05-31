import numpy as np
import matplotlib.pyplot as plt

class Data_Cleansing:
    def __init__(self):
        pass
    def Remove_outlier(self,Data):
        Q1 = Data.quantile(0.25)
        Q3 = Data.quantile(0.75)
        IQR = Q3 - Q1
        print("-------Number of lower in every column .-------")
        print(np.sum(Data < (Q1 - 1.5 * IQR)))
        print("-------Nomber of Upper in every column .-------")
        print(np.sum(Data > (Q3 + 1.5 * IQR)))
        Data.boxplot()
        plt.show()
        Data = Data[~((Data < (Q1 - 1.5*IQR)) | (Data > (Q3 + 1.5*IQR))).any(axis=1)]
        print("-------Number of lower in every column .-------")
        print(np.sum(Data < (Q1 - 1.5 * IQR)))
        print("-------Number of Upper in every column .-------")
        print(np.sum(Data > (Q3 + 1.5 * IQR)))
        Data.boxplot()
        plt.show()
        return Data
    def Handling_Zeros(self,Data):
        for x in range(0, 30):
            column = Data.iloc[:, x]
            column_mean = column.mean()
            Data = Data.replace(0, column_mean)
        return Data
    def Encode(self,Data):
        for Row in Data.iteritems():
            Data = Data.replace('M', 1)
            Data = Data.replace('B', 0)
        return Data

    def normalization(self, Data):
        for x in range(0, 30):
            column = Data.iloc[:, x]
            colume_max = Data.max()
            column_min = Data.min()
            Data = (Data - column_min) / (colume_max - column_min)
        return Data
