import pandas as pd
from DealData.FileOperation import FileOperation
import os


class DataDivision:
    def __init__(self, train_file_path,test_file_path):
        self.__filOp = FileOperation()
        self.__train_file_path = train_file_path
        self.__test_file_path = test_file_path
        self.__label_idx = 1
        pass

    def run(self):
        train_datas = pd.read_excel(self.__train_file_path,dtype=float)
        test_datas = pd.read_excel(self.__test_file_path,dtype=float)
        columns = train_datas.columns
        test_datas.columns = columns

        train_labels, train_feas = self.__separate_label_features(train_datas, columns)
        test_labels, test_feas = self.__separate_label_features(test_datas, columns)

        test_feas = self.__normalize(train_feas, test_feas)

        test_set = pd.concat([test_labels, test_feas], axis=1)

        test_path = self.__get_save_path(self.__test_file_path)
        test_set.to_csv(test_path)
        pass


    def __separate_label_features(self, datas, columns):
        labels = datas.loc[:, columns[:(self.__label_idx + 1)]]
        features = datas.loc[:, columns[(self.__label_idx + 1):]]
        return labels, features

    def __normalize(self, train_feas, test_feas):
        means = train_feas.mean()
        stds = train_feas.std()
        test_feas = (test_feas - means) / stds #归一化时，两个表
        return test_feas

    def __get_save_path(self, file_path):
        save_dir = self.__filOp.get_parent_dir(file_path)
        test_path = os.path.join(save_dir, 'test.csv')
        return test_path


if __name__ == '__main__':
    train_file_path = '../Data/Aid/ROP.xlsx'
    test_file_path = '../Data/AidV/ROPV.xlsx'
    data_division = DataDivision(train_file_path,test_file_path)
    data_division.run()
