import pickle
import numpy as np

from scipy.stats import zscore, rankdata
def my_zscore(x):
    return zscore(x,ddof=1),np.mean(x,axis=0),np.std(x,axis=0,ddof=1)

if __name__=='__main__':
    with open('11总数据集改bloc.pkl', 'rb') as file:
        MIMICtable = pickle.load(file)  # [278751 rows x 59 columns]的数据
    reformat5 = MIMICtable.values.copy()
    icustayid_index = MIMICtable.columns.get_loc('icustayid')
    icustayid = reformat5[:, icustayid_index]
    bloc = reformat5[:, 0]

    # 总共22个
    # 静态数据 3个   性别，年龄，体重
    colbin = ['gender']
    # 21个  19个动态 HR-心率 RR-呼吸频率   SysBP-收缩压  DiaBP-舒张压   SpO2-血氧饱和度 Temp_C-温度  paO2-动脉氧分压
    colnorm = ['age', 'Weight_kg',
               'HR', 'RR', 'SysBP', 'MeanBP', 'DiaBP', 'SpO2',
               'Temp_C', 'paO2', 'PaO2_FiO2', 'Arterial_lactate', 'WBC_count', 'Glucose', 'Hb',
               'BUN', 'Total_bili', 'Creatinine', 'output_4hourly', 'GCS', 'Platelets_count'
               ]
    ##从数据中提取特征数值，索引值
    colbin=np.where(np.isin(MIMICtable.columns,colbin))[0]
    colnorm=np.where(np.isin(MIMICtable.columns,colnorm))[0]
    '''这是特征'''
    scale_Feature=np.concatenate([reformat5[:, colbin] - 0.5,zscore(reformat5[:,colnorm],ddof=1)],axis=1)
    '''动作干预身体  复苏液  加压药'''
    iol = MIMICtable.columns.get_loc('input_4hourly')
    iol_data1 = reformat5[:, iol]
    for i in range(len(iol_data1)):
        if iol_data1[i] > 3000:
            iol_data1[i] = 3000
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    iol_data = scaler.fit_transform(iol_data1.reshape(-1, 1))
    '''结果评分SOFA'''
    SOFA_index = MIMICtable.columns.get_loc('SOFA')
    SOFA_score1 = reformat5[:, SOFA_index]
    SOFA_score = np.empty((SOFA_score1.shape[0]), dtype=float)
    for i in range(len(SOFA_score1)):
        if SOFA_score1[i] > 14:
            SOFA_score[i] = 3
        elif SOFA_score1[i] > 8:
            SOFA_score[i] = 2
        elif SOFA_score1[i] > 5:
            SOFA_score[i] = 1
        else:
            SOFA_score[i] = 0
    xin_dataset = np.hstack((bloc.reshape(-1, 1), icustayid.reshape(-1, 1), scale_Feature, iol_data.reshape(-1, 1),
                             SOFA_score.reshape(-1, 1)))
    ###==============找出SOFA阶段多的人================================
    loaded_data = np.load('31筛选出人数据集.npz')
    array_sur = loaded_data['arr1']
    array_mor = loaded_data['arr2']
    total_array = np.concatenate((array_sur,array_mor))
    #***###生成挑选后的数据
    total_array = np.isin(icustayid, total_array).squeeze()
    xin_total_dataset = xin_dataset[total_array]
    ###重采样生成新数据集
    loaded_33 = np.load('32全是3的人.npz')
    array_sur_33 = loaded_33['arr1']
    array_mor_33 = loaded_33['arr2']

    chong_ID = 99853
    xin_icustayid = xin_total_dataset[:,1]
    renci = 27

    chong_data_sur = np.zeros((20 * len(array_sur_33)*renci, xin_total_dataset.shape[1]))
    xinzeng_sur = np.array([])
    count = 0
    for i in range(renci):
        for ID in array_sur_33:
            total_array1 = np.isin(xin_icustayid, ID).squeeze()
            total_array_dataset = xin_total_dataset[total_array1]
            total_array_dataset[:,1] = chong_ID
            chong_data_sur[count:count+20,:] = total_array_dataset
            xinzeng_sur = np.append(xinzeng_sur,chong_ID)
            count = count + 20
            chong_ID = chong_ID +1

    chong_data_mor = np.zeros((20 * len(array_mor_33)*renci, xin_total_dataset.shape[1]))
    xinzeng_mor = np.array([])
    count = 0
    for i in range(renci):
        for ID in array_mor_33:
            total_array1 = np.isin(xin_icustayid, ID).squeeze()
            total_array_dataset = xin_total_dataset[total_array1]
            total_array_dataset[:,1] = chong_ID
            chong_data_mor[count:count+20,:] = total_array_dataset
            xinzeng_mor = np.append(xinzeng_mor,chong_ID)
            count = count + 20
            chong_ID = chong_ID +1
    total_dataset = np.concatenate((xin_total_dataset, chong_data_sur, chong_data_mor), axis=0)
    total_array_sur = np.concatenate((array_sur, xinzeng_sur), axis=0)
    total_array_mor = np.concatenate((array_mor, xinzeng_mor), axis=0)
    count_zeros = np.count_nonzero(total_dataset[:,-1] == 0)
    count_ones = np.count_nonzero(total_dataset[:,-1] == 1)
    count_twos = np.count_nonzero(total_dataset[:,-1] == 2)
    count_threes = np.count_nonzero(total_dataset[:,-1] == 3)
