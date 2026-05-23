###IQL算法实现

import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
device = 'cpu'
class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super(ValueFunction, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, state):
        conv_out = self.conv1(state)
        return conv_out
class VDQN(nn.Module):
    def __init__(self, state_dim, n_actions=4):
        super(VDQN, self).__init__()

        self.fc_adv = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self,state):
        adv = self.fc_adv(state)
        return adv
def asymmetric_l2_loss(u, tau):
    return T.mean(T.abs(tau - (u < 0).float()) * u ** 2)

if __name__ == '__main__':
    start = time.perf_counter()  # 计算时间
    num_epoch = 101
    lr = 0.001
    batch_s = 512
    epsilon = 0.8  # 随机寻找因子
    gamma = 0.3  # 折扣因子
    tau = 0.1
    num_classes = [0, 1, 2, 3]
    reward = 10
    loaded_data = np.load('304处理好数据的数据集.npz')
    Xtrain = loaded_data['arr1']
    Xvalidation = loaded_data['arr2']
    Xtest = loaded_data['arr3']
    Xtest = np.concatenate((Xtest, Xvalidation), axis=0)
    loaded_data_next = np.load('305数据集next_train_done19阶段.npz')
    Xtrain_next = loaded_data_next['arr1']
    done = loaded_data_next['arr2']
    Xtrain_uids = np.unique(Xtrain[:, 1])
    input_Xtrain = T.FloatTensor(Xtrain[:, 2:]).to(device)
    input_Xtest = T.FloatTensor(Xtest[:, 2:]).to(device)
    input_Xtrain_next = T.FloatTensor(Xtrain_next[:,2:]).to(device)
    input_Xtrain_done = T.FloatTensor(done).to(device)
    # 初始化模型,定义损失函数和优化器
    statesize = input_Xtrain.shape[1]
    model_Q = VDQN(statesize - 1)
    model_tarQ = VDQN(statesize - 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_Q.parameters(), lr)
    vf = ValueFunction(statesize - 1)
    v_optimizer = optim.Adam(vf.parameters(), lr=0.0001)
    print('####  开始训练  ####')
    num_batch = Xtrain_uids.shape[0] // batch_s  # 分批次
    if Xtrain_uids.shape[0] % batch_s == 0:
        num_batch = num_batch
    else:
        num_batch =num_batch + 1

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        Batch = 0
        print('循环次数：', epoch)
        for batch_idx in range(num_batch):  # MAX_Iteration
            batch_uids = Xtrain_uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
            batch_user = np.isin(Xtrain[:, 1], batch_uids)
            input_Xtrain_user = input_Xtrain[batch_user, :]
            input_Xtrain_user_next = input_Xtrain_next[batch_user, :]
            input_Xtrain_done_user = input_Xtrain_done[batch_user]
            ###开始执行强化学习算法
            batch_size1 = input_Xtrain_user.shape[0]
            range_batch = T.arange(batch_size1).long().to(device)
            ### target神经网络求Q值
            with T.no_grad():
                V_dist_target = vf(input_Xtrain_user_next[:, :-1])
                Q_dist_target = model_tarQ(input_Xtrain_user_next[:, :-1])
                a_tar_star = T.argmax(Q_dist_target, dim=1)   # 求最大值
                Q_dist_tar_star = Q_dist_target[range_batch, a_tar_star]
            # 计算V值
            V_dist_prediction = vf(input_Xtrain_user[:, :-1])
            adv = Q_dist_tar_star - V_dist_prediction
            v_loss = asymmetric_l2_loss(adv, 0.7)
            v_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            v_optimizer.step()
            end_multiplier = 1 - input_Xtrain_done_user
            label = input_Xtrain_user[:, -1]
            labels = label.to(T.long)
            #############################################################################
            # action = labels
            action = np.empty((labels.shape[0],), dtype=float)
            ##当前状态的action
            for i in range(labels.shape[0]):
                if random.uniform(0, 1) < epsilon:
                    action[i]=random.randint(0, 3)
                else:
                    action[i] = labels[i]
            #############################################################################
            reward_s = np.empty((labels.shape[0],), dtype=float)
            for i in range(action.shape[0]):
                if action[i] == labels[i]:
                    reward_s[i] = reward
                else:
                    reward_s[i] = -reward
            reward_s = T.FloatTensor(reward_s).to(device)
            targetQ1 = reward_s + (gamma * V_dist_target.squeeze().detach() * end_multiplier)
            log_Q_dist_prediction = model_Q(input_Xtrain_user[:, :-1])
            log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action]
            loss = nn.MSELoss()(targetQ1, log_Q_dist_prediction1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # 更新Q_target网络参数
            if Batch % 25 == 0:
                for param, target_param in zip(model_Q.parameters(), model_tarQ.parameters()):
                    target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)
                    target_param.data.copy_(param.data)
            Batch += 1
        avg_loss = epoch_loss
    end = time.perf_counter()
    print('使用时间：（秒）',end-start)
