import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
import time
from DealData.FileOperation import FileOperation
import json
import os

class Resulter:
    def __init__(self,resulter_paras,run_time):
        self.__fileOp=FileOperation()
        self.__init_parameters(resulter_paras,run_time)
        self.best_valid_acc= 0
        self.best_test_acc = 0
        pass

    def __init_parameters(self,resulter_paras,run_time):
        self.__result_root=resulter_paras['result_root']
        self.__describe=resulter_paras['describe']
        save_dir=os.path.join(self.__result_root,self.__describe,str(run_time))
        self.__fileOp.create_dir(save_dir)
        self.__predict_train_path = os.path.join(save_dir, resulter_paras['predict_train_file'])
        self.__predict_valid_path = os.path.join(save_dir, resulter_paras['predict_valid_file'])
        self.__predict_path = os.path.join(save_dir, resulter_paras['predict_file'])
        self.__result_path=os.path.join(save_dir,resulter_paras['result_file'])
        self.__save_model_path=os.path.join(save_dir,resulter_paras['save_model_path'])
        pass

    def init_train_conf(self, train_conf):
        current_time = time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime(time.time()))
        print('====== {0} ======'.format(current_time))

        init_datas=self.__get_init_datas(current_time,train_conf)
        self.__fileOp.write_txt_by_append(init_datas,self.__result_path)
        pass

    def __get_init_datas(self,current_time,train_conf):
        datas=[]
        datas.append('\n====== {0} ======\n'.format(current_time))
        datas.append('{}\n'.format(json.dumps(train_conf, indent=4)))
        datas.append('--------------------------------------------\n')
        return datas

    def get_save_model_path(self):
        return self.__save_model_path

    def show_per_epoch(self, epoch, loss, acc):
        '''
        :param epoch: current epoch
        :param loss: train loss
        :param acc: acc of test set
        :return:
        '''
        self.__print_datas_per_epoch(epoch, loss, acc)
        self.__save_datas_per_epoch(epoch, loss, acc)
        pass

    def __print_datas_per_epoch(self,epoch, loss, acc):
        print("\nepoch {}\n"
              "\ttrain loss: {}"
              "\tvalid acc: {}{}".format(
            epoch,  loss, acc, " **" if acc > self.best_valid_acc else ""))
        pass

    def __save_datas_per_epoch(self,epoch, loss, acc):
        save_datas=[]
        save_datas.append("epoch {}\n".format(epoch) )
        save_datas.append("\ttrain loss: {}\tvalid acc: {}{}\n".format(loss,acc,
                " **" if acc > self.best_valid_acc else ""))
        self.__fileOp.write_txt_by_append(save_datas,self.__result_path)
        pass

    def show_for_test(self, acc):
        print("\ttest acc: {}".format(acc))
        save_datas=[]
        save_datas.append("\ttest acc: {}".format(acc))
        self.__fileOp.write_csv_by_append(save_datas,self.__result_path)
        pass

    def save_predict(self,data, describe=None):

        if describe == None:
            self.__fileOp.write_csv_by_append(data,self.__predict_path)
        else:
            predict_path = self.__fileOp.get_file_name(self.__predict_path)
            parent_path = self.__fileOp.get_parent_dir(self.__predict_path)
            exten = self.__fileOp.get_file_extension(self.__predict_path)
            save_path = os.path.join(parent_path,predict_path+describe+exten)
            self.__fileOp.write_csv_by_append(data,save_path)
        pass

    def save_predict_valid(self,data):
        self.__fileOp.write_csv_by_append(data,self.__predict_valid_path)
        pass

    def save_predict_train(self,data):
        self.__fileOp.write_csv_by_append(data,self.__predict_train_path)
        pass

    def save_test_result(self,data):
        self.__fileOp.write_csv(data,self.__predict_path)
        pass