import numpy as np

if __name__=='__main__':

    loaded_dataset = np.load('33筛选出人总数据集-重采样.npz')
    total_dataset = loaded_dataset['arr1']
    array_sur = loaded_dataset['arr2']
    array_mor = loaded_dataset['arr3']

    ###查看有多少人 再乘以*19
    total_dataset_uids = np.unique(total_dataset[:, 1])
    total_array = np.zeros((19 * len(total_dataset_uids), total_dataset.shape[1]))
    count=0
    for i in range(total_dataset.shape[0]-1):
        if (total_dataset[i + 1, 0] == 1):
            continue
        else:
            total_array[count,:-1] = total_dataset[i,:-1]
            total_array[count, -1] = total_dataset[i+1,-1]
            count = count + 1
    ###下阶段的训练数据
    next_total_array = np.empty((total_array.shape[0], total_array.shape[1]), dtype=float)
    done = np.empty((0,), dtype=float)
    for i in range(total_array.shape[0] - 1):
        if (total_array[i + 1,0] == 1):
            next_total_array[i, :2] = total_array[i, :2]
            done = np.append(done, 1.0)
        else:
            next_total_array[i, :2] = total_array[i, :2]
            next_total_array[i,2:] = total_array[i+1,2:]
            done = np.append(done, 0)
    next_total_array[i+1, :2] = total_array[i+1, :2]
    done = np.append(done, 1.0)

    ####================以上是处理好的数据==========================================
    np.random.shuffle(array_sur)
    np.random.shuffle(array_mor)
    # 生存组分布8:1:1
    total_length = len(array_sur)
    train_length = int(total_length * 0.8)
    valid_length = int(total_length * 0.1)

    array_sur_train_data = array_sur[:train_length]
    array_sur_valid_data = array_sur[train_length:train_length + valid_length]
    array_sur_test_data = array_sur[train_length + valid_length:]

    # 死亡组组分布8:1:1
    total_length1 = len(array_mor)
    train_length1 = int(total_length1 * 0.8)
    valid_length1 = int(total_length1 * 0.1)

    array_mor_train_data = array_mor[:train_length1]
    array_mor_valid_data = array_mor[train_length1:train_length1 + valid_length1]
    array_mor_test_data = array_mor[train_length1 + valid_length1:]


    train_data = np.concatenate((array_sur_train_data, array_mor_train_data), axis=0)
    train_data = np.random.permutation(train_data)

    valid_data = np.concatenate((array_sur_valid_data, array_mor_valid_data), axis=0)
    valid_data = np.random.permutation(valid_data)

    test_data = np.concatenate((array_sur_test_data, array_mor_test_data), axis=0)
    test_data = np.random.permutation(test_data)

    ###=========================分成训练集、验证集以及测试集======================================
    icustayid = total_array[:, 1]
    train = np.isin(icustayid, train_data).squeeze()
    valid = np.isin(icustayid, valid_data).squeeze()
    test = np.isin(icustayid, test_data).squeeze()

    train_dataset = total_array[train]
    valid_dataset = total_array[valid]
    test_dataset = total_array[test]

    next_train_dataset = next_total_array[train]
    next_valid_dataset = next_total_array[valid]
    next_test_dataset = next_total_array[test]

    train_done = done[train]
    valid_done = done[valid]
    test_done = done[test]

    # np.savez('304处理好数据的数据集.npz', arr1=train_dataset, arr2=valid_dataset, arr3=test_dataset)
    # np.savez('305数据集next_train_done19阶段.npz', arr1=next_train_dataset, arr2=train_done)
    # np.savez('305数据集next_valid_done19阶段.npz', arr1=next_valid_dataset, arr2=valid_done)
    # np.savez('305数据集next_test_done19阶段.npz', arr1=next_test_dataset, arr2=test_done)

