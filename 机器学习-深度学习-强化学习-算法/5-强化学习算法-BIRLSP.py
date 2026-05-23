import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import random
import copy
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix

device = 'cpu'


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super(ValueFunction, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
            # nn.ReLU()
        )
        self.fc_val1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        conv_out1 = self.conv1(state)
        val1 = self.fc_val1(conv_out1)
        return val1

class VDQN(nn.Module):
    def __init__(self, state_dim, n_actions=4):
        super(VDQN, self).__init__()

        self.conv_D = nn.Sequential(
            nn.Linear(state_dim-4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.conv_S = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(256, 512),
            # nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self,state):
        state_D = state[:,3:-1]
        state_S = T.cat((state[:,:3], state[:,-1].unsqueeze(1)), dim=1)
        state_D = self.conv_D(state_D)
        state_S = self.conv_S(state_S)
        combined_state = T.cat((state_D, state_S), dim=1)

        adv = self.fc_adv(combined_state)
        val = self.fc_val(combined_state)
        return val + adv - adv.mean(dim=1, keepdim=True)

def asymmetric_l2_loss(u, tau):
    return T.mean(T.abs(tau - (u < 0).float()) * u ** 2)


if __name__ == '__main__':
    starttime = time.perf_counter()  # 计算开始时间

    ###参数
    num_epoch = 101
    lr = 0.001
    epsilon = 0.8  # 随机寻找因子
    gamma = 0.3  # 折扣因子
    batch_s = 512   #数据分割批量大小
    tau = 0.1   #强化学习软更新参数
    num_classes = [0, 1, 2, 3]  #分类数量
    reward = 10     #强化学习奖励reward

    ###数据导入
    loaded_data = np.load('304处理好数据的数据集.npz')
    Xtrain = loaded_data['arr1']
    Xvalidation = loaded_data['arr2']
    Xtest = loaded_data['arr3']

    loaded_data_next = np.load('305数据集next_train_done19阶段.npz')
    Xtrain_next = loaded_data_next['arr1']
    done = loaded_data_next['arr2']

    Xtest = np.concatenate((Xtest, Xvalidation), axis=0)

    input_Xtrain = T.FloatTensor(Xtrain[:, 2:]).to(device)
    input_Xtest = T.FloatTensor(Xtest[:, 2:]).to(device)

    input_Xtrain_next = T.FloatTensor(Xtrain_next[:,2:]).to(device)
    input_Xtrain_done = T.FloatTensor(done).to(device)

    # 初始化模型,定义损失函数和优化器
    statesize = input_Xtrain.shape[1]
    model_Q = VDQN(statesize - 1)
    model_tarQ = copy.deepcopy(model_Q)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_Q.parameters(), lr)

    vf = ValueFunction(statesize - 1)
    v_optimizer = optim.Adam(vf.parameters(), lr=0.001)

    print('####  准备开始训练  ####')
    Xtrain_uids = np.unique(Xtrain[:, 1])
    num_batch = Xtrain_uids.shape[0] // batch_s  # 分批次
    if Xtrain_uids.shape[0] % batch_s == 0:
        num_batch = num_batch
    else:
        num_batch =num_batch + 1

    epoch_losses = np.array([])
    z_accuracy = 0
    z_Precision = 0
    z_Recall = 0
    z_F1score = 0
    z_auc = 0
    ###训练循环
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        Batch = 0
        print('循环次数：', epoch)
        # random.shuffle(Xtrain_uids)
        for batch_idx in range(num_batch):
            batch_uids = Xtrain_uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
            batch_user = np.isin(Xtrain[:, 1], batch_uids)

            input_Xtrain_user = input_Xtrain[batch_user, :]
            input_Xtrain_user_next = input_Xtrain_next[batch_user, :]
            input_Xtrain_done_user = input_Xtrain_done[batch_user]

            ###*开始执行强化学习算法
            batch_size1 = input_Xtrain_user.shape[0]
            range_batch = T.arange(batch_size1).long().to(device)

            ### target神经网络求target_Q值
            with T.no_grad():
                Q_dist_target = model_tarQ(input_Xtrain_user_next[:, :-1])
                a_tar_star = T.argmax(Q_dist_target, dim=1)  # DQN机制尝试
                Q_dist_tar_star = Q_dist_target[range_batch, a_tar_star]

            V_dist_prediction = vf(input_Xtrain_user_next[:, :-1])
            adv = Q_dist_tar_star - V_dist_prediction
            v_loss = asymmetric_l2_loss(adv, 0.7)
            v_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            v_optimizer.step()
            end_multiplier = 1 - input_Xtrain_done_user
            label = input_Xtrain_user[:, -1]
            labels = label.to(T.long)
            action = np.empty((labels.shape[0],), dtype=float)
            ###当前状态的action
            for i in range(labels.shape[0]):
                #生成随机数
                if random.uniform(0, 1) < epsilon:
                    action[i]=random.randint(0, 3)
                else:
                    action[i] = labels[i]
            reward_s = np.empty((labels.shape[0],), dtype=float)
            for i in range(action.shape[0]):
                if action[i] == labels[i]:
                    reward_s[i] = reward
                else:
                    exponent= T.abs(action[i] - labels[i])
                    reward_s[i] = T.pow(0.5, exponent)*reward - (2 * reward)
                    # reward_s[i] = -reward
            reward_s = T.FloatTensor(reward_s).to(device)
            with T.no_grad():
                V_dist_target = vf(input_Xtrain_user_next[:, :-1])
            targetQ1 = reward_s + (gamma * V_dist_target.squeeze().detach() * end_multiplier)

            ###当前状态-动作Q值
            log_Q_dist_prediction = model_Q(input_Xtrain_user[:, :-1])
            action = T.FloatTensor(action).to(device).to(T.long)
            log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action]

            ###计算损失值
            loss = nn.MSELoss()(targetQ1, log_Q_dist_prediction1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新Q_target网络参数
            if Batch % 30 == 0:
                # print('更新网络')
                for param, target_param in zip(model_Q.parameters(), model_tarQ.parameters()):
                    target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)
                    target_param.data.copy_(param.data)
            Batch += 1

            ###损失值加和
            epoch_loss += loss.item()
        avg_loss = epoch_loss /num_batch

        epoch_losses =  np.append(epoch_losses, avg_loss)

        ####============检验=================================================
        if epoch % 2 == 0 and epoch != 0:
            with T.no_grad():
                print('测试集')
                Q_prediction = model_Q(input_Xtrain[:, :-1])
                predicted = T.argmax(Q_prediction, dim=1)
                labels = input_Xtrain[:, -1]
                ### 正确率、精确度、召回率、F1分数
                correct_predictions = T.sum(labels == predicted)
                validation_accuracy = correct_predictions / input_Xtrain.shape[0]
                print('Accuracy：', validation_accuracy)  # 准确率（Accuracy）
                Precision = precision_score(labels, predicted, average='macro')
                Recall = recall_score(labels, predicted, average='macro')
                F1score = f1_score(labels, predicted, average='macro')
                print("Precision (macro): %f" % Precision)
                print("Recall (macro):    %f" % Recall)
                print("F1 score (macro):  %f" % F1score)
                ### AUC面积、特异度
                auc_scores = []
                for class_label in num_classes:
                    binary_true_labels = np.where(labels == class_label, 0, 1)
                    binary_predicted_labels = np.where(predicted == class_label, 0, 1)
                    auc = roc_auc_score(binary_true_labels, binary_predicted_labels)
                    auc_scores.append(auc)
                average_auc = np.mean(auc_scores)
                print("AUC(macro):", average_auc, end='\n\n')
    print('Accuracy：',z_accuracy)
    print("Precision (macro): %f" % z_Precision)
    print("Recall (macro):    %f" % z_Recall)
    print("F1 score (macro):  %f" % z_F1score)
    print("AUC(macro):", z_auc)
    endtime = time.perf_counter()   # 计算解释时间
    print("使用时间（分）:",(endtime-starttime)/60)


