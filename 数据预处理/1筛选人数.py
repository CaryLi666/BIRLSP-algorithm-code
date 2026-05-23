import pickle
import numpy as np
if __name__=='__main__':
    with open('11总数据集改bloc.pkl', 'rb') as file:
        MIMICtable = pickle.load(file)  # [278751 rows x 59 columns]的数据
    reformat5 = MIMICtable.values.copy()
    title = MIMICtable.columns.values.tolist()
    icustayid_index = MIMICtable.columns.get_loc('icustayid')
    icustayid = reformat5[:, icustayid_index]
    mortality_index = MIMICtable.columns.get_loc('mortality_90d')
    number_mortality_90d = reformat5[:, mortality_index]
    bloc_index = MIMICtable.columns.get_loc('bloc')
    number_bloc = reformat5[:, bloc_index]
    my_array_sur = np.array([])
    my_array_mor = np.array([])
    count =0
    count_sur = 0
    for i in range(len(icustayid)):
        if number_bloc[i] == 20:
            if number_mortality_90d[i] == 1:
                if count > 1259:
                    break
                my_array_mor = np.append(my_array_mor, icustayid[i])
                count=count+1
            else:
                if count_sur < 2740:
                    my_array_sur = np.append(my_array_sur, icustayid[i])
                count_sur = count_sur + 1
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
    chao_ID = np.array([])
    for ID in my_array_sur:
        array_boor = np.isin(icustayid, ID).squeeze()
        sur_dataset = SOFA_score[array_boor]
        count_ones = np.count_nonzero(sur_dataset == 0)
        if count_ones > 17:
            chao_ID = np.append(chao_ID, ID)
    my_array_sur = np.setdiff1d(my_array_sur, chao_ID)

    chao_ID1 = np.array([])
    for ID in my_array_mor:
        array_boor = np.isin(icustayid, ID).squeeze()
        sur_dataset = SOFA_score[array_boor]
        count_ones = np.count_nonzero(sur_dataset == 0)
        if count_ones > 17:
            chao_ID1 = np.append(chao_ID1, ID)
    my_array_mor = np.setdiff1d(my_array_mor, chao_ID1)

    np.savez('31筛选出人数据集.npz', arr1=my_array_sur, arr2=my_array_mor)
