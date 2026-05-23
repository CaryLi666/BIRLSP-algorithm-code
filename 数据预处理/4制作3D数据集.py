import time
import numpy as np

if __name__ == '__main__':
    start = time.perf_counter()  # 计算时间
    ###参数
    loaded_data = np.load('304处理好数据的数据集.npz')
    Xtrain = loaded_data['arr1']
    Xvalidation = loaded_data['arr2']
    Xtest = loaded_data['arr3']
    Xtest = np.concatenate((Xtest, Xvalidation), axis=0)

    ###训练集3D
    Xtrain_uids = np.unique(Xtrain[:, 1])
    Xtest_uids = np.unique(Xtest[:, 1])

    train_array_3D = np.zeros((len(Xtrain_uids), 19, Xtrain.shape[1]))
    count = 0
    stage = 0
    for i in range(Xtrain.shape[0]-1):
        if (Xtrain[i + 1, 0] == 1):
            train_array_3D[count, stage, :] = Xtrain[i, :]
            stage = 0
            count = count + 1
        else:
            train_array_3D[count,stage,:] = Xtrain[i,:]
            stage = stage + 1

    ###测试集3D
    test_array_3D = np.zeros((len(Xtest_uids), 19, Xtest.shape[1]))
    count = 0
    stage = 0
    for i in range(Xtest.shape[0]-1):
        if (Xtest[i + 1, 0] == 1):
            test_array_3D[count, stage, :] = Xtest[i, :]
            stage = 0
            count = count + 1
        else:
            test_array_3D[count,stage,:] = Xtest[i,:]
            stage = stage + 1

    np.savez('306处理好数据的数据集_3D.npz', arr1=train_array_3D,  arr2=test_array_3D)