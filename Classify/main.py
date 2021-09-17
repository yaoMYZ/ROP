import sys,os
sys.path.append(os.path.dirname(sys.path[0]))

import json
from DealData.DataMgrImg import DataMgr
from DealData.Resulter import Resulter
from Classify.train_with_img import Train

class Main:
    def __init__(self):
        pass

    def __analyse_json(self,sModelJson):
        with open(sModelJson) as f:
            js = f.read()
            model = json.loads(js)
            return model

    def run(self,paras_jsons,train_json, fold):
        train_conf = self.__get_train_conf(paras_jsons,train_json)
        data_mgr=self.__get_data_mgr(paras_jsons,fold)
        resulter=self.__get_resulter(train_conf,fold)
        trainer=Train(data_mgr, resulter, train_conf)
        trainer.train()
        pass

    def __get_data_mgr(self,paras_jsons,fold):
        tran_conf = self.__analyse_json(paras_jsons)
        data_mgr=DataMgr(tran_conf,fold)
        return data_mgr

    def __get_resulter(self,train_conf,fold):
        resulter_paras = train_conf['results']
        resulter=Resulter(resulter_paras,fold)
        resulter.init_train_conf(train_conf)
        return resulter

    def __get_train_conf(self,paras_jsons,train_json):
        train_conf = self.__analyse_json(train_json)
        dataset_conf = self.__analyse_json(paras_jsons)
        train_conf['dataset'] = dataset_conf
        return train_conf

def run():
    for fold in range(5):
        paras_jsons = 'dataset.json'
        train_json = 'train.json'
        main = Main()

        main.run(paras_jsons, train_json, fold)
    pass

if __name__=='__main__':
    run()
