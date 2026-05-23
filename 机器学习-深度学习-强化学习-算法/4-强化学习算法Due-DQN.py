import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix

device = 'cpu'

class Due_DQN(nn.Module):
    def __init__(self, state_dim, n_actions=4):
        super(Due_DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state):
        conv_out = self.conv(state)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)
if __name__ == '__main__':
    start = time.perf_counter()  # 计算时间
    num_epoch = 101
    reward = 10
    lr=0.001
    batch_s = 512
    epsilon = 0.8 # 随机寻找因子
    gamma = 0.3  # 折扣因子
    tau = 0.1
    num_classes = [0, 1, 2, 3]

    loaded_data = np.load('304处理好数据的数据集.npz')
    Xtrain = loaded_data['arr1']
    Xvalidation = loaded_data['arr2']
    Xtest = loaded_data['arr3']

    Xtest = np.concatenate((Xtest, Xvalidation), axis=0)

    loaded_data_next = np.load('305数据集next_train_done19阶段.npz')
    Xtrain_next = loaded_data_next['arr1']
    done = loaded_data_next['arr2']

    Xtrain_uids = np.unique(Xtrain[:, 1])

    input_Xtrain = T.FloatTensor(Xtrain[:,2:]).to(device)
    input_Xtest = T.FloatTensor(Xtest[:,2:]).to(device)

    input_Xtrain_next = T.FloatTensor(Xtrain_next[:,2:]).to(device)
    input_Xtrain_done = T.FloatTensor(done).to(device)

    statesize = input_Xtrain.shape[1]
    model_Q = Due_DQN(statesize - 1)
    model_tarQ = Due_DQN(statesize - 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_Q.parameters(), lr)

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

            log_Q_dist_prediction = model_Q(input_Xtrain_user[:, :-1])

            ##当前状态的action
            labels = input_Xtrain_user[:, -1]
            action = labels

            ##执行完动作的奖励
            reward_s = np.empty((labels.shape[0],), dtype=float)
            for i in range(action.shape[0]):
                if action[i] == labels[i]:
                    reward_s[i] = reward
                else:
                    reward_s[i] = -reward

            action = T.FloatTensor(action).to(device).to(T.long)
            reward_s = T.FloatTensor(reward_s).to(device)

            log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action]

            with T.no_grad():
                Q_dist_target = model_tarQ(input_Xtrain_user_next[:, :-1])
                q_eval = model_Q(input_Xtrain_user[:, :-1])
                max_q_act = T.argmax(q_eval, dim=1)

            Q_dist_star = Q_dist_target[range_batch, max_q_act]

            end_multiplier = 1 - input_Xtrain_done_user
            targetQ1 = reward_s + (gamma * Q_dist_star* end_multiplier)
            loss = nn.MSELoss()(targetQ1, log_Q_dist_prediction1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if Batch % 25 == 0:
                # print('更新网络')
                for param, target_param in zip(model_Q.parameters(), model_tarQ.parameters()):
                    target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)
                    target_param.data.copy_(param.data)
            Batch += 1
        avg_loss = epoch_loss

        if epoch % 10 == 0 and epoch != 0:
            with T.no_grad():

                print('测试集')
                Q_prediction = model_Q(input_Xtest[:, :-1])
                predicted = T.argmax(Q_prediction, dim=1)
                labels = input_Xtest[:, -1]

                ### 正确率、精确度、召回率、F1分数
                correct_predictions = T.sum(labels == predicted)
                validation_accuracy = correct_predictions /input_Xtest.shape[0]
                print('Accuracy：', validation_accuracy)  # 准确率（Accuracy）
                print("Precision (macro): %f" % precision_score(labels, predicted, average='macro'))
                print("Recall (macro):    %f" % recall_score(labels, predicted, average='macro'))
                print("F1 score (macro):  %f" % f1_score(labels, predicted, average='macro'), end='\n\n')
                ### AUC面积、特异度
                auc_scores = []
                for class_label in num_classes:
                    binary_true_labels = np.where(labels == class_label, 0, 1)
                    binary_predicted_labels = np.where(predicted == class_label, 0, 1)
                    auc = roc_auc_score(binary_true_labels, binary_predicted_labels)
                    auc_scores.append(auc)
                average_auc = np.mean(auc_scores)
                print("AUC(macro):", average_auc)
    end = time.perf_counter()
    print('花费时间：',end-start)