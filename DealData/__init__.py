from DealData.FileOperation import FileOperation
import numpy as np
def test():
    a=np.random.uniform(0,1,size=10)
    b = np.random.uniform(0, 1, size=10)
    print(a)
    print(b)
    print(np.corrcoef(a,b))
    print(cal_corrcoef(a,b))
    pass

def cal_corrcoef(X,Y):
    # 均值
    Xmean = np.mean(X)
    Ymean = np.mean(Y)

    # 标准差
    Xsd = np.std(X)
    Ysd = np.std(Y)

    # z分数
    ZX = (X - Xmean) / Xsd
    ZY = (Y - Ymean) / Ysd

    # 相关系数
    r = np.sum(ZX * ZY) / len(X)
    return r

if __name__=='__main__':
    fileOp=FileOperation()
    file_path='fdf/dfd.csv'
    print(fileOp.get_file_name(file_path))
    # test()