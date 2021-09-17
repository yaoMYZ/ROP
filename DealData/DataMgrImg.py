import sys, os

sys.path.append(os.path.dirname(sys.path[0]))

from DealData.FileOperation import FileOperation
import numpy as np
import json
import torch
import os
from PIL import Image


class DataMgr:
    def __init__(self, train_conf, fold):
        self.__valid_rate = train_conf['training_para']['valid_rate']
        self.__fileOp = FileOperation()
        dataset_paras = train_conf['dataset']
        self.__batch_size = train_conf['training_para']['batch_size']
        self.__id_idx = train_conf['training_para']['id_idx']
        self.__out_idx = train_conf['training_para']['out_idx']
        self.__init_parameters()
        train_data_path = dataset_paras['train_data_path']+'{}.csv'.format(fold)
        test_data_path = dataset_paras['test_data_path']+'{}.csv'.format(fold)
        img_data_path = dataset_paras['img_path']
        self.__read_all_data(train_data_path, test_data_path, img_data_path)
        pass

    def __init_parameters(self):
        self.__train_list = []
        self.__train_batch_idx = 0
        self.__valid_batch_idx = 0
        self.__test_batch_idx = 0
        pass

    def __read_all_data(self, train_data_path, test_data_path, img_data_path):
        id_imgs_map = self.__get_id_img_map(img_data_path)
        self.all_train_set = self.__read_data(train_data_path, id_imgs_map)
        self.test_set = self.__read_data(test_data_path, id_imgs_map)
        self.valid_set, self.train_set = self.__divide_valid_set(self.all_train_set)
        pass

    def __get_id_img_map(self, img_data_path):
        id_set = self.__fileOp.get_sub_dirs(img_data_path)
        id_imgs_map = {}
        for id in id_set:
            sub_dir = os.path.join(img_data_path, id)
            img_paths = self.__fileOp.scan_all_files(sub_dir)
            id_imgs_map[int(id)] = img_paths
        return id_imgs_map

    # def __divide_valid_set(self, data_set):
    #     tmp_set = self.__shuffle_data(data_set)
    #     valid_num = int(len(tmp_set) * self.__valid_rate)
    #     valid_set = tmp_set[:valid_num]
    #     train_set = tmp_set[valid_num:]
    #     return np.asarray(valid_set), np.asarray(train_set)

    def __divide_valid_set(self, data_set):
        train_data_dict=[{},{}]
        for sample in data_set:
            id = sample[0]
            label = int(sample[1])
            if id not in train_data_dict[label]:
                train_data_dict[label][id]=[]
            train_data_dict[label][id].append(sample)
        valid_set, train_set = self.__sub_divide_valid_set(train_data_dict[0].values())
        tmp_valid_set, tmp_train_set = self.__sub_divide_valid_set(train_data_dict[1].values())
        valid_set += tmp_valid_set
        train_set += tmp_train_set
        return np.asarray(valid_set), np.asarray(train_set)

    def __sub_divide_valid_set(self,datas):
        datas = list(datas)
        valid_num = int(len(datas) * self.__valid_rate)
        valid_set = []
        train_set = []
        for samples in datas[:valid_num]:
            valid_set += samples
        for samples in datas[valid_num:]:
            train_set += samples
        return valid_set, train_set

    def __shuffle_data(self, data):
        fnum = len(data)
        sdix = np.random.permutation(fnum)
        data_s = [data[idx] for idx in sdix]
        return data_s

    def __read_data(self, file_path, id_imgs_map, if_add_positive=False):
        datas = self.__fileOp.read_csv(file_path)
        datas = np.asarray(datas[1:], dtype=np.float)  # 去除标签
        datas = datas[:, 1:]  # 去除序列号
        data_set = []
        for data in datas:
            id = int(data[self.__id_idx])
            label = int(data[self.__out_idx])
            feas = data[(self.__out_idx + 1):]
            img_paths = id_imgs_map[id]
            data_per_person = []
            for img_path in img_paths:
                extension = self.__fileOp.get_file_extension(img_path)
                if extension != '.png' and extension != '.jpg':
                    continue
                data_row = (id, label, feas, img_path)
                data_per_person.append(data_row)
            data_set += data_per_person
            if if_add_positive and int(label)==1:
                data_set += data_per_person
        return data_set

    def get_valid_batch(self):
        if self.__valid_batch_idx == 0:
            self.__valid_list = self.__get_batches_idxs(n=len(self.valid_set), batch_size=self.__batch_size)
        data_set = self.__get_next_batch(self.valid_set, self.__valid_list, self.__valid_batch_idx)
        self.__valid_batch_idx = (self.__valid_batch_idx + 1) % len(self.__valid_list)
        return data_set

    def get_train_batch(self):
        if self.__train_batch_idx == 0:
            self.__train_list = self.__get_batches_idxs(n=len(self.train_set), batch_size=self.__batch_size)
        data_set = self.__get_next_batch(self.train_set, self.__train_list, self.__train_batch_idx)
        self.__train_batch_idx = (self.__train_batch_idx + 1) % len(self.__train_list)
        return data_set

    def __get_batches_idxs(self, n, batch_size):
        """
        Used to shuffle the dataset at each iteration.
        """
        idx_list = np.arange(n, dtype="int32")
        np.random.shuffle(idx_list)

        batches = []
        batch_start = 0
        for i in range(n // batch_size):
            batches.append(idx_list[batch_start:batch_start + batch_size])
            batch_start += batch_size

        # 若是样本没有整除batch size，则把最后剩余不足batch size的样本单独作为一个batch
        if (batch_start != n):
            batches.append(idx_list[batch_start:])

        return batches

    def __get_next_batch(self, dataset, batch_list, batch_idx):
        idxs = batch_list[batch_idx]
        batch_datas = [dataset[i] for i in idxs]
        data_set = self.__get_batch_sample(batch_datas)
        return data_set

    def __get_batch_sample(self, batch_datas):
        ids, label, feas, imgs = self.__reorganize_data(batch_datas)

        feas = torch.tensor(feas, dtype=torch.float)
        imgs = torch.tensor(imgs, dtype=torch.float)
        imgs = imgs.permute(0, 3, 1, 2)  # 样本数*通道*高*宽
        label = torch.tensor(label, dtype=torch.float)
        return feas, imgs, label, ids

    def __reorganize_data(self, batch_datas):
        ids = []
        imgs = []
        feas = []
        label = []
        for data in batch_datas:
            id, out, text_feas, img_path = data
            ids.append(id)
            label.append(out)
            feas.append(text_feas)
            img = self.__read_img(img_path)
            imgs.append(img)
        # label_onehot = np.eye(2)[label]
        return self.__reshape_data_into2dim(ids), self.__reshape_data_into2dim(label), feas, imgs

    def __read_img(self, img_path):
        tmp_img = Image.open(img_path)
        img = np.asarray(tmp_img, np.float)
        img = np.expand_dims(img, axis=2)
        return img

    def get_test_batch(self):
        if self.__test_batch_idx == 0:
            self.__test_list = self.__get_batches_idxs(n=len(self.test_set), batch_size=self.__batch_size)
        data_set = self.__get_next_batch(self.test_set, self.__test_list, self.__test_batch_idx)
        self.__test_batch_idx = (self.__test_batch_idx + 1) % len(self.__test_list)
        return data_set

    def __tensor2np(self, data):
        data_np = data.detach().numpy()
        return data_np

    def __reshape_data_into1dim(self, data):
        return np.reshape(data, newshape=data.shape[0])

    def __reshape_data_into2dim(self, data):
        data = np.asarray(data)
        return np.reshape(data, newshape=(len(data), 1))

    def get_train_batch_num(self):
        return int(np.ceil(len(self.train_set) / self.__batch_size))

    def get_valid_batch_num(self):
        return int(np.ceil(len(self.valid_set) / self.__batch_size))

    def get_test_batch_num(self):
        return int(np.ceil(len(self.test_set) / self.__batch_size))

    def get_train_set(self):
        ids, label, feas, imgs = self.__reorganize_data(self.all_train_set)
        return np.asarray(feas, np.float), np.asarray(label, np.int), ids


def analyse_json(sModelJson):
    with open(sModelJson) as f:
        js = f.read()
        model = json.loads(js)
        return model


if __name__ == '__main__':
    tran_conf = analyse_json('../Classify/dataset.json')
    data_mgr = DataMgr(tran_conf, 0)
    x, imgs, y1, __ = data_mgr.get_test_batch()
    print(x.shape, y1.shape)
