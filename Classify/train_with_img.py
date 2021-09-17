# coding: utf-8
import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
import time
from torch import optim
import numpy as np

from Model.Resnet import *
from DealData.FileOperation import FileOperation
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class Train:
    def __init__(self, data_mgr, resulter, train_conf):
        self.__initParameter(data_mgr, resulter, train_conf)
        self.__define_device()
        self.__contruct_model()
        pass

    def __initParameter(self, data_mgr, resulter, train_conf):
        self.__data_mgr = data_mgr
        self.__resulter = resulter
        self.__train_conf = train_conf
        self.__save_model_path = self.__resulter.get_save_model_path()

        self.__fileOp = FileOperation()
        root_path = self.__fileOp.get_parent_dir(self.__save_model_path)
        self.__train_summary_writer = SummaryWriter(log_dir=os.path.join(root_path, 'logs', 'train'))
        self.__valid_summary_writer = SummaryWriter(log_dir=os.path.join(root_path, 'logs', 'test'))
        pass

    def __contruct_model(self):
        n_classes = self.__train_conf['training_para']['n_out']
        self.__learning_rate = self.__train_conf['training_para']['learning_rate']
        input_channels = self.__train_conf['training_para']['input_channels']
        fea_dim = self.__train_conf['training_para']['input_dim']
        self.__L2_reg = self.__train_conf['training_para']['L2_reg']

        self.__model = ResNet(in_channels=input_channels, fea_dim=fea_dim, block=Bottleneck, layers=[3, 4, 6, 3],
                              num_classes=n_classes)
        self.__model = self.__model.to(self.__device)
        self.__criterion = nn.BCELoss()
        # self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.SGD(self.__model.parameters(), lr=self.__learning_rate)
        pass

    def train(self):
        start_time = time.time()
        n_epochs = self.__train_conf['training_para']['n_epochs']
        min_train_loss = 10e3
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            self.__model = self.__model.train()  # 设置为训练模式
            self.__epoch = epoch
            running_loss = []
            epoch_start_time = time.time()
            for step in range(self.__data_mgr.get_train_batch_num()):
                # get the inputs
                x_train, img_train, y_train, __ = self.__data_mgr.get_train_batch()
                img_train = img_train.to(self.__device)
                x_train = x_train.to(self.__device)
                y_train = y_train.to(self.__device)
                # zero the parameter gradients
                self.__optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.__model(x_train, img_train)
                bce_loss, l2_loss, joint_loss = self.__get_losses(outputs, y_train)
                joint_loss.backward()
                self.__optimizer.step()
                # statistics
                running_loss.append(joint_loss.item())
                acc = self.__cal_acc(outputs, y_train)
                # add record
                record_step = epoch * self.__data_mgr.get_train_batch_num() + step
                self.__add_record_per_step(self.__train_summary_writer, bce_loss.item(), l2_loss.item(),
                                           acc, record_step)
            ave_train_loss = np.mean(running_loss)

            self.__resulter.show_per_epoch(self.__epoch, ave_train_loss, 0)
            # 使用测试集验证模型
            if ave_train_loss<min_train_loss:
                min_train_loss = ave_train_loss
                with torch.no_grad():
                    self.__test_step()
                    torch.save(self.__model.state_dict(), self.__save_model_path)
            epoch_end_time = time.time()
            print('epoch ran for %.2fm' % ((epoch_end_time - epoch_start_time) / 60.))

        self.__end_train(start_time)

    def __define_device(self):
        cuda_id = self.__train_conf['cuda']
        self.__device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
        print(self.__device)

    def __get_losses(self, outputs, labels):
        bce_loss = self.__criterion(outputs, labels)
        l2_loss = self.__L2_loss(self.__L2_reg)
        joint_loss = bce_loss + l2_loss
        return bce_loss, l2_loss, joint_loss

    def __L2_loss(self, l2_reg):
        l2_reg = torch.tensor(l2_reg).to(self.__device)
        l2_loss = torch.tensor(0.0).to(self.__device)
        for name, param in self.__model.named_parameters():
            if 'weight' in name:
                l2_loss += torch.norm(param)
        return l2_reg * l2_loss

    def __add_record_per_step(self, summary_writer, bce_loss, l2_loss, acc, step):
        datas = {'bce_loss': bce_loss, 'l2_loss': l2_loss,
                 'joint_loss': bce_loss + l2_loss, 'acc': acc}
        self.__add_record(summary_writer, datas, step)
        pass

    def __add_record(self, summary_writer, datas, step):
        for tag, data in datas.items():
            summary_writer.add_scalar(tag, data, step)
        pass

    def __cal_acc(self, outputs, labels):
        predict_labels = self.__reshape_data_into1dim(self.__tensor2np(outputs))
        # predict_labels = self.__tensor2np(self.__softmax_label(outputs))
        predict_labels = self.__reset_label(predict_labels)
        labels = self.__reshape_data_into1dim(self.__tensor2np(labels))
        labels = self.__reset_label(labels)
        return accuracy_score(y_pred=predict_labels, y_true=labels)

    def __softmax_label(self,outputs):
        outputs = F.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, 1)
        return prediction

    def __reset_label(self, labels):
        new_lables = []
        for label in labels:
            if label < 0.5:
                new_lables.append(0)
            else:
                new_lables.append(1)
        return new_lables

    def __reshape_data_into1dim(self, data):
        return np.reshape(data, newshape=data.shape[0])

    def __end_train(self, start_time):
        end_time = time.time()
        self.__train_summary_writer.close()
        self.__valid_summary_writer.close()
        print("best acc:{}".format(self.__resulter.best_test_acc))
        print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))


    def __test_step(self):
        accs = []
        test_outputs = []
        test_y = []
        all_ids = []
        for step in range(self.__data_mgr.get_test_batch_num()):
            x_test, img_test, y_test, ids = self.__data_mgr.get_test_batch()
            img_test = img_test.to(self.__device)
            x_test = x_test.to(self.__device)
            y_test = y_test.to(self.__device)
            # forward
            outputs = self.__model(x_test, img_test)
            # statistics
            acc = self.__cal_acc(outputs, y_test)
            accs.append(acc)
            # record
            all_ids += ids.tolist()
            test_outputs += self.__tensor2np(outputs).tolist()
            test_y += self.__tensor2np(y_test).tolist()
        ave_acc = np.mean(accs).tolist()
        self.__resulter.best_test_acc = ave_acc
        self.__resulter.show_for_test(ave_acc)
        self.__save_test_results(test_outputs, test_y, all_ids)
        pass

    def __save_test_results(self, outputs, labels, ids):
        outputs = np.asarray(outputs)
        labels = np.asarray(labels)
        results = np.hstack((ids, outputs, labels))
        self.__resulter.save_test_result(results.tolist())
        pass

    def __tensor2np(self, data):
        data_np = data.cpu().detach().numpy()
        return data_np
