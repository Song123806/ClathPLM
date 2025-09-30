import statistics

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    confusion_matrix


def roc(test_y, pr_list):
    """Plot ROC curve and calculate AUC score."""
    fpr, tpr, thresholds = roc_curve(test_y, pr_list)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xticks([(i / 10) for i in range(11)])
    plt.yticks([(i / 10) for i in range(11)])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print(f"AUC: {roc_auc:.4f} ")


def fold_roc(fprs, tprs):
    # 计算每一折的AUC
    aucs = []
    for i in range(len(fprs)):
        roc_auc = np.trapz(tprs[i], fprs[i])
        aucs.append(roc_auc)

    # 计算平均ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    for i in range(len(fprs)):
        mean_tpr += np.interp(mean_fpr, fprs[i], tprs[i])
    mean_tpr /= len(fprs)
    mean_tpr += 0.0  # 确保起始点为0
    mean_auc = np.trapz(mean_tpr, mean_fpr)

    # 绘制图像
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(mean_fpr, mean_tpr, color='navy', lw=2, label='Mean ROC (AUC = %0.2f)' % mean_auc)

    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3, label='Fold %d Roc(AUC = %0.2f)' % (i + 1, aucs[i]))

    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    # 设置 x 轴和 y 轴的刻度和范围
    plt.xticks([(i / 10) for i in range(11)])
    plt.yticks([(i / 10) for i in range(11)])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of ten-fold cross validation')
    plt.legend(loc="lower right")
    plt.show()


def print_eva(accs, mccs, pres, recs, spes, f1s, aucs):
    print(f"Train Accuracy: {statistics.mean(accs):.4f}, "
          f"Train MCC: {statistics.mean(mccs):.4f}, "
          f"Train Precision: {statistics.mean(pres):.4f}, "
          f"Train Recall: {statistics.mean(recs):.4f}, "
          f"Train Specificity: {statistics.mean(spes):.4f}, "
          f"Train AUC: {statistics.mean(aucs):.4f}, "
          f"Train f1_score: {statistics.mean(f1s):.4f}, ")


# 输入真实数据和预测数据进行评估，并绘制混淆矩阵
def evaluate_model_performance(true_labels, predictions):
    """Evaluate model performance on classification metrics and plot confusion matrix."""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = recall_score(true_labels, predictions, pos_label=0)
    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)

    print(f"Acc: {accuracy:.4f}, "
          f"MCC: {mcc:.4f}, "
          f"Pr: {precision:.4f}, "
          f"Sn: {recall:.4f}, "
          f"Sp: {specificity:.4f}, "
          f"F1-score: {f1:.4f}, ", end="")

    plot_confusion_matrix(true_labels, predictions)

def score(test_y,y_pred):
    # 模型评估
    test_accuracy = accuracy_score(test_y, y_pred)
    test_precision = precision_score(test_y, y_pred)
    test_recall = recall_score(test_y, y_pred)
    test_specificity = recall_score(test_y, y_pred, pos_label=0)
    test_f1 = f1_score(test_y, y_pred)
    mcc_score = matthews_corrcoef(test_y, y_pred)
    # test_auc = roc_auc_score(y_label_test, pr_list)
    # 打印评估指标
    print(f"Test Acc: {test_accuracy:.4f}, "
          f"Test Mcc: {mcc_score:.4f}, "
          f"Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, "
          f"Test Specificity: {test_specificity:.4f}, "
          f"Test f1_score: {test_f1:.4f}, ")

    "*********************************绘制混淆矩阵************************************"
    # 计算混淆矩阵
    cm = confusion_matrix(test_y, y_pred)

    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 在每个格子中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.show()

def plot_confusion_matrix(true_labels, predictions):
    """Plot the confusion matrix for given true and predicted labels."""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display counts in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


class EarlyStopping:
    """
    早停，参数：patience，delta
    patience：当验证损失连续 patience 个周期没有改善时，将触发早停
    delta：如果验证损失的改善 < delta，则不认为是有效的改善
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience    # 当验证损失连续 patience 个周期没有改善时，将触发早停。
        self.delta = delta  # 如果验证损失的改善<delta，则不认为是有效的改善
        self.best_loss = np.Inf # 最优 loss 初始为无穷大
        self.counter = 0    # 当前没有改善的周期数
        self.early_stop = False

    def __call__(self, val_loss):
        val_loss = float(val_loss)  # 转换为Python float
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop