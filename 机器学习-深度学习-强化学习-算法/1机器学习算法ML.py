import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
if __name__=='__main__':
    start = time.perf_counter()  # 计算时间
    num_epoch = 101

    num_classes = [0,1,2,3]
    loaded_data = np.load('304处理好数据的数据集.npz')
    input_Xtrain = loaded_data['arr1']
    Xvalidation = loaded_data['arr2']
    Xtest = loaded_data['arr3']
    input_Xtest = np.concatenate((Xtest, Xvalidation), axis=0)


    '''随机森林Random Forest'''
    # from sklearn.ensemble import RandomForestClassifier
    # rf_classifier = RandomForestClassifier()
    '''决策树Decision Tree'''
    from sklearn.tree import DecisionTreeClassifier
    rf_classifier = DecisionTreeClassifier()
    '''XGBoost'''
    # import xgboost as xgb
    # rf_classifier = xgb.XGBClassifier()

    z_accuracy = 0
    z_Precision = 0
    z_Recall = 0
    z_F1score = 0
    z_auc = 0
    for epoch in range(num_epoch):
        print("epoch:",epoch)
        rf_classifier.fit(input_Xtrain[:, 2:25], input_Xtrain[:, -1])

        if epoch % 2 == 0 and epoch != 0:
            print('测试集')
            predicted = rf_classifier.predict(input_Xtest[:, 2:25])
            labels=input_Xtest[:, -1]
            ### 正确率、精确度、召回率、F1分数
            correct_predictions = np.sum(labels == predicted)
            validation_accuracy = correct_predictions /input_Xtest.shape[0]
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
            if z_auc < average_auc:
                z_auc = average_auc
                z_accuracy = validation_accuracy
                z_Precision = Precision
                z_Recall = Recall
                z_F1score = F1score
    print('Accuracy：', z_accuracy)
    print("Precision (macro): %f" % z_Precision)
    print("Recall (macro):    %f" % z_Recall)
    print("F1 score (macro):  %f" % z_F1score)
    print("AUC(macro):", z_auc)

    end = time.perf_counter()
    print('使用时间：（秒）',end-start)