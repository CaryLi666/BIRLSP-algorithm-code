import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix

device='cpu'

class LSTM(nn.Module):
    def __init__(self,input_dims, hidden_size,num_layers,num_classes,):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dims, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.ReLU(),
        )

    def forward(self, state):
        h0 = T.zeros(self.num_layers, state.size(0), self.hidden_size).requires_grad_()
        c0 = T.zeros(self.num_layers,state.size(0), self.hidden_size).requires_grad_()
        conv_out, (hn, cn) = self.lstm(state, (h0, c0))
        conv_out = self.fc(conv_out)
        return conv_out

###RNN和GRU算法
# class RNNetwork(nn.Module):
#     def __init__(self,input_dims, hidden_size,num_layers,num_classes,):
#         super(RNNetwork, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # self.rnn = nn.RNN(input_dims, hidden_size, num_layers, batch_first=True)
#         self.rnn = nn.GRU(input_dims, hidden_size, num_layers, batch_first=True)
#
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, num_classes),
#             nn.ReLU(),
#             # nn.Softmax(dim=2),
#             # nn.Linear(hidden_size, num_classes),
#             # nn.ReLU()
#         )
#
#     def forward(self, state):
#         h0 = T.zeros(self.num_layers, state.size(0), self.hidden_size).requires_grad_()
#         conv_out, hn = self.rnn(state, h0)
#         conv_out = self.fc(conv_out)
#         return conv_out

###普通的深度神经网络
# class DNNetwork(nn.Module):
#     def __init__(self,input_dims, hidden_size,num_layers,num_classes,):
#         super(DNNetwork, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.fc = nn.Sequential(
#             nn.Linear(input_dims, 2*hidden_size),
#             nn.ReLU(),
#             nn.Linear(2*hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_classes),
#             nn.ReLU(),
#         )
#
#     def forward(self, state):
#         h0 = T.zeros(self.num_layers, state.size(0), self.hidden_size).requires_grad_()
#         conv_out, hn = self.rnn(state, h0)
#         conv_out = self.fc(conv_out)
#         return conv_out


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
    num_classe = 4
    loaded_data = np.load('306处理好数据的数据集_3D.npz')
    Xtrain = loaded_data['arr1']
    Xtest = loaded_data['arr2']

    input_Xtrain = T.FloatTensor(Xtrain[:,:,2:]).to(device)
    input_Xtest = T.FloatTensor(Xtest[:,:,2:]).to(device)

    # 初始化模型,定义损失函数和优化器
    statesize = input_Xtrain.shape[2]
    hidden_size = 128  # 隐藏层大小
    num_layer = 2  # 层数量
    ##神经网络模型
    # model = RNNetwork(statesize - 1, hidden_size, num_layer, num_classe)
    # model = DNNetwork(statesize - 1, hidden_size, num_layer, num_classe)
    model = LSTM(statesize - 1, hidden_size, num_layer, num_classe)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    batch_size = 128
    train_loader = T.utils.data.DataLoader(dataset=input_Xtrain,
                                           batch_size=batch_size)

    z_accuracy = 0
    z_Precision = 0
    z_Recall = 0
    z_F1score = 0
    z_auc = 0
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        Batch = 0
        print('循环次数：', epoch)
        # 循环所有的数据，进行训练
        loss_sum = 0
        for i, batch in enumerate(train_loader):
            input_batch = batch[:, :, :-1]
            lable_batch = batch[:, :, -1]
            outputs = model(input_batch)
            output = outputs.permute(0, 2, 1)
            lable_batch = lable_batch.to(dtype=T.long)
            loss = criterion(output, lable_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        # print(loss_sum)

        if epoch % 10 == 0 and epoch != 0:
            with T.no_grad():
                print('测试集')
                Q_prediction = model(input_Xtest[:,:, :-1])
                predicted = T.max(Q_prediction.data, 2)[1]
                labels = input_Xtest[:,:, -1]

                ### 正确率、精确度、召回率、F1分数
                correct_predictions = T.sum(labels == predicted)
                validation_accuracy = correct_predictions /(input_Xtest.shape[0]*19)
                print('Accuracy：', validation_accuracy)  # 准确率（Accuracy）
                labelss = T.flatten(labels)
                predicteds = T.flatten(predicted)
                Precision = precision_score(labelss, predicteds, average='macro')
                Recall = recall_score(labelss, predicteds, average='macro')
                F1score = f1_score(labelss, predicteds, average='macro')
                print("Precision (macro): %f" % Precision)
                print("Recall (macro):    %f" % Recall)
                print("F1 score (macro):  %f" % F1score)
                ### AUC面积、特异度
                auc_scores = []
                for class_label in num_classes:
                    binary_true_labels = np.where(labelss == class_label, 0, 1)
                    binary_predicted_labels = np.where(predicteds == class_label, 0, 1)
                    auc = roc_auc_score(binary_true_labels, binary_predicted_labels)
                    auc_scores.append(auc)
                average_auc = np.mean(auc_scores)
                print("AUC(macro):", average_auc, end='\n\n')
            if z_auc < average_auc:
                z_auc = average_auc
                z_accuracy = validation_accuracy
                z_Precision = Precision
                z_Recall = Recall
                z_F1score = F1score
    print('Accuracy：',z_accuracy)
    print("Precision (macro): %f" % z_Precision)
    print("Recall (macro):    %f" % z_Recall)
    print("F1 score (macro):  %f" % z_F1score)
    print("AUC(macro):", z_auc)

