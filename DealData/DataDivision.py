import pandas as pd
from DealData.FileOperation import FileOperation
import os


class DataDivision:
    def __init__(self, file_path):
        self.__filOp = FileOperation()
        self.__file_path = file_path
        self.__test_rate = 0.2
        self.__label_idx = 1
        pass

    def run(self):
        datas = pd.read_excel(self.__file_path,dtype=float)
        columns = datas.columns

        sick_datas, nsick_datas = self.__separate_datas_by_labels(datas, columns)
        fold_num = int(1.0/self.__test_rate)
        for fold in range(fold_num):
            train_datas, test_datas = self.__separate_train_test_datas(sick_datas, nsick_datas, fold)

            train_labels, train_feas = self.__separate_label_features(train_datas, columns)
            test_labels, test_feas = self.__separate_label_features(test_datas, columns)

            train_feas, test_feas = self.__normalize(train_feas, test_feas)

            train_set = pd.concat([train_labels, train_feas], axis=1)
            test_set = pd.concat([test_labels, test_feas], axis=1)

            train_path , test_path = self.__get_save_path(self.__file_path ,fold)
            train_set.to_csv(train_path)
            test_set.to_csv(test_path)
        pass

    def __separate_datas_by_labels(self, datas, columns):
        sick_datas = datas[datas.iloc[:, self.__label_idx] > 0]
        sick_datas = sick_datas.sample(frac=1)

        nsick_datas = datas[datas[columns[self.__label_idx]] == 0]
        nsick_datas = nsick_datas.sample(frac=1)
        return sick_datas, nsick_datas

    def __separate_train_test_datas(self, sick_datas, nsick_datas, fold):
        sick_test_num = int(self.__test_rate * len(sick_datas))
        nsick_test_num = int(self.__test_rate * len(nsick_datas))
        train_datas = pd.concat([sick_datas[:sick_test_num*fold],sick_datas[sick_test_num*(fold+1):], nsick_datas[:nsick_test_num*fold], nsick_datas[nsick_test_num*(fold+1):]]).sample(frac=1)
        test_datas = pd.concat([sick_datas[sick_test_num*fold:sick_test_num*(fold+1)], nsick_datas[nsick_test_num*fold:nsick_test_num*(fold+1)]]).sample(frac=1)
        return train_datas, test_datas

    def __separate_label_features(self, datas, columns):
        labels = datas.loc[:, columns[:(self.__label_idx + 1)]]
        features = datas.loc[:, columns[(self.__label_idx + 1):]]
        return labels, features

    def __normalize(self, train_feas, test_feas):
        means = train_feas.mean()
        stds = train_feas.std()
        train_feas = (train_feas - means) / stds
        test_feas = (test_feas - means) / stds
        return train_feas, test_feas

    def __get_save_path(self, file_path, fold):
        save_dir = self.__filOp.get_parent_dir(file_path)
        train_path = os.path.join(save_dir, 'train{}.csv'.format(fold))
        test_path = os.path.join(save_dir, 'test{}.csv'.format(fold))
        return train_path, test_path


if __name__ == '__main__':
    file_path = '../Data/Aid/ROP.xlsx'
    data_division = DataDivision(file_path)
    data_division.run()
