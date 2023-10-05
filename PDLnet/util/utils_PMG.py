import csv

from scipy.interpolate import interp1d

from util.utils import *
# from mod.pmg_attention_resnet import RAPMG
from mod.pmg_attention_resnet_com import RAPMG_com
from mod.pmg_attention_resnet_box import RAPMG
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
model_save_root_path = "./save_path/pth"


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % nb_epoch)  # t - u is used when t has u-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


def load_PMG_model(model_name, pretrain=False, require_grad=True, num_class=4):
    logging.info('==> Building model..')
    net = resnet50(pretrained=pretrain)
    # net = resnet101(pretrained=pretrain)
    for param in net.parameters():
        param.requires_grad = require_grad
    net = RAPMG(net, 512, classes_num=num_class)    #512是指图片大小
    return net


def load_PMG_model_com(model_name, pretrain=True, require_grad=True, num_class=4):
    logging.info('==> Building model..')
    net = resnet50(pretrained=pretrain)
    # net = resnet101(pretrained=pretrain)
    for param in net.parameters():
        param.requires_grad = require_grad
    net = RAPMG_com(net, 512, classes_num=num_class)
    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    logging.info(
        '\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        logging.info('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    logging.info('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def test_PMG(net, criterion, batch_size, test_path,test_acc_max,num_class,store_name):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # 初始化类别的准确率字典
    mAP=0
    class_correct = {}
    class_total = {}
    avg_class_correct=0
    for i in range(num_class):
        class_correct[i] = 0
        class_total[i] =0
    device = torch.device("cuda:0")
    predicted1 = np.array([])
    true1 = np.array([])
    predictedMAx = np.array([])
    trueMax= np.array([])
    # test_acc_max=0.0

    # 初始化空的预测结果和真实标签列表
    y_pred_list = []
    y_true_list = []

    # 计算每个类别的准确率和召回率
    precisions = []
    recalls = []

    loader_test = loadTestData(batch_size, test_path)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader_test):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            resnet_label = torch.LongTensor([0 if (i == 0 or i == 2) else 1 for i in targets]).to('cuda')
            # resnet_label = None
            inputs, targets = Variable(inputs), Variable(targets)
            output_concat, sc_loss = net(inputs, targets, 4, 0,  resnet_label)
            loss = criterion(output_concat, targets)
            # 将得分转换为概率分布
            probabilities = F.softmax(output_concat, dim=1)
            test_loss += loss.item()
            _, maxIndex = torch.max(output_concat.data, 1)
            predicted = torch.softmax(output_concat,dim=-1)
            _, predicted_labels = torch.max(predicted, dim=-1)
            # 将预测结果和真实标签添加到列表中
            y_pred_list.append(probabilities)
            y_true_list.append(targets)

            # 统计每个类别的准确率
            for i in range(len(targets)):
                target = targets[i].item()
                predicted_label = predicted_labels[i].item()
                if target == predicted_label:
                    # 预测正确
                    class_correct[target] = class_correct.get(target, 0) + 1
                class_total[target] = class_total.get(target, 0) + 1


            predicted1 = np.append(predicted1, predicted_labels.cpu().numpy())
            true1 = np.append(true1, targets.cpu().numpy())

            total += targets.size(0)
            correct += maxIndex.eq(targets.data).cpu().sum()
    # 计算每个类别的准确率
    for cls in class_total:
        accuracy = 100.0 * class_correct[cls] / class_total[cls]
        avg_class_correct=avg_class_correct+accuracy
        print('Accuracy of class {} : {:.2f}%'.format(cls, accuracy))
    avg_class_correct=avg_class_correct/num_class

    # 将预测结果和真实标签列表转换为数组
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    # 将张量转换为 NumPy 数组
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()









    np.save('our_true_confusionmatrix_'+store_name+'.npy', true1)
    np.save('our_pre_confusionmatrix_'+store_name+'.npy', predicted1)

    test_acc = 100. * float(correct) / total

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)
    if(test_acc>test_acc_max):
        np.save('our_true_confusionmatrix_Max_'+store_name+'.npy', true1)
        np.save('our_pre_confusionmatrix_Max_'+store_name+'.npy', predicted1)

        # 保存 y_pred
        np.save('our_y_pred_' + store_name + '.npy', y_pred)

        # 保存 y_true
        np.save('our_y_true_' + store_name + '.npy', y_true)

        # 计算每个类别的AP值和计算平均mAP
        ap_values = []

        for i in range(num_class):
            ap = average_precision_score(y_true == i, y_pred[:, i])
            ap_values.append(ap)
            precision, recall, _ = precision_recall_curve(y_true == i, y_pred[:, i])
            # plt.plot(recall, precision, label=f'Class {i}')
            precisions.append(precision)
            recalls.append(recall)
        # 计算mAP值
        mAP = np.mean(ap_values)
        print("AP values:", ap_values)
        print("mAP:", mAP)

        # 计算精确度和召回率计算每个类别的P-R曲线
        precision = dict()
        recall = dict()
        for i in range(num_class):
            precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_pred[:, i])

        # 找到最大的精确度和召回率数组的长度
        max_len = max(max(len(precision[i]) for i in range(num_class)), max(len(recall[i]) for i in range(num_class)))
        # 对精确度和召回率进行填充或插值，使其长度保持一致
        for i in range(num_class):
            if len(precision[i]) < max_len:
                precision[i] = np.pad(precision[i], (0, max_len - len(precision[i])), mode='edge')
            if len(recall[i]) < max_len:
                recall[i] = np.pad(recall[i], (0, max_len - len(recall[i])), mode='edge')
        # 对精确度和召回率进行插值，使其长度保持一致
        # for i in range(num_class):
        #     interp_func = interp1d(np.linspace(0, 1, len(precision[i])), precision[i], kind='linear')
        #     precision[i] = interp_func(np.linspace(0, 1, max_len))
        #
        #     interp_func = interp1d(np.linspace(0, 1, len(recall[i])), recall[i], kind='linear')
        #     recall[i] = interp_func(np.linspace(0, 1, max_len))
        # 计算平均精确度和召回率
        mean_precision = np.mean([precision[i] for i in range(num_class)], axis=0)
        mean_recall = np.mean([recall[i] for i in range(num_class)], axis=0)
        # 保存
        np.save('our_mean_precision_' + store_name + '.npy', mean_precision)
        np.save('our_mean_recall_' + store_name + '.npy', mean_recall)

        # # 绘制P-R曲线图
        # plt.figure()
        # plt.step(mean_recall, mean_precision, color='b', where='post', label='Mean Precision-Recall')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Mean Precision-Recall Curve')
        # plt.legend()
        # plt.show()

        # 计算每个类别的ROC曲线
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(num_class):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        max_length = max(max(len(precision[i]) for i in range(num_class)),
                         max(len(recall[i]) for i in range(num_class)))
        # 使用np.pad函数来填充数组，使其长度一致
        for i in range(num_class):
            if len(fpr[i]) < max_length:
                fpr[i] = np.pad(fpr[i], (0, max_length - len(fpr[i])), mode='edge')
            if len(tpr[i]) < max_length:
                tpr[i] = np.pad(tpr[i], (0, max_length - len(tpr[i])), mode='edge')

        # for i in range(num_class):
        #     interp_func = interp1d(np.linspace(0, 1, len(fpr[i])), fpr[i], kind='linear')
        #     fpr[i] = interp_func(np.linspace(0, 1, max_length))
        #
        #     interp_func = interp1d(np.linspace(0, 1, len(tpr[i])), tpr[i], kind='linear')
        #     tpr[i] = interp_func(np.linspace(0, 1, max_length))

        # 计算平均FPR和TPR
        mean_fpr = np.mean(list(fpr.values()), axis=0)
        mean_tpr = np.mean(list(tpr.values()), axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        np.save('our_mean_fpr_' + store_name + '.npy', mean_fpr)
        np.save('our_mean_tpr_' + store_name + '.npy', mean_tpr)
        # # 绘制平均ROC曲线
        # plt.figure()
        # plt.plot(mean_fpr, mean_tpr, label='Mean ROC curve (AUC = {0:.2f})'.format(mean_auc))
        # plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Mean Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc='lower right')
        # plt.show()

    return test_acc, test_acc_en, test_loss,mAP


def test_PMG_com(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0")

    loader_test = loadTestData(batch_size, test_path)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader_test):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            # resnet_label = torch.LongTensor([0 if (i == 0 or i == 2) else 1 for i in targets]).to('cuda')
            resnet_label = None
            inputs, targets = Variable(inputs), Variable(targets)
            output_com, output_concat, resnet_loss = net(x=inputs, index=4,loss=None,target=resnet_label)
            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(output_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss
