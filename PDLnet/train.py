from __future__ import print_function
from PIL import Image
import logging
import random
from util.Resnet import resnet50
import torch
import torch.optim as optim
from util.utils_PMG  import *
from util.utils import *
import os.path as osp
from util.adversarial import LabelSmoothSoftmaxCEV1
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log_save_root_path = "./save_path/information/"
model_save_root_path = "./save_path/pth/"

seed = 0  # 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)


def train_PMG(nb_epoch, batch_size, store_name, train_path, test_path, numclass,start_epoch=0, model_path=None):
    # setup output
    # 设置试验结果数据存放位置
    exp_dir = log_save_root_path + store_name
    modsavedir=model_save_root_path+store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    try:
        os.stat(modsavedir)
    except:
        os.makedirs(modsavedir)
    use_cuda = torch.cuda.is_available()

    # 这段代码涉及使用PyTorch训练深度神经网络。
    # 首先，代码创建了一个数据加载器; `loader_train`，从指定的; `train_path`; 目录中加载训练数据。`batch_size`; 参数控制着每次前向 / 后向传递处理的样本数。 然后，使用; `load_PMG_model`; 函数创建了一个深度神经网络，参数为; `model_name = 'resnet50_pmg'; `、`pretrain = True; `、`require_grad = True; `、`num_class = 12; `。这表明网络基于ResNet - 50; 架构，并训练用于将输入分为12个类别。
    #
    # 使用; `to()`; 方法将网络移动到GPU上，并创建了一个名为; `device`; 的对象，对象的参数为; `"cuda:0"`。 使用; `CrossEntropyLoss`; 对象来计算训练期间预测值和目标值之间的损失。 最后，使用; `optim.SGD`; 实例作为优化器，用于在训练期间更新网络参数。它使用随机梯度下降更新权重，动量为0; .9，权重衰减为5e - 4。每个模型参数的学习率使用字典列表指定。每个字典都具有形式; `{'params': < params >, 'lr': < learning_rate >}`，其中; ` < params > ` 是一组模型参数，` < learning_rate > ` 是这些参数的学习率。
    logging.info(use_cuda)
     # 数据加载器
    loader_train = loadTrainData(batch_size, train_path)
    # 形成深度网络
    net = load_PMG_model(model_name='resnet50_pmg', pretrain=True, require_grad=True, num_class=numclass)
    # net = torch.load('./save_path/pth/APMG_PPMI/PPMI.pth')
    device = torch.device("cuda:0")
    net.to(device)
    CELoss = nn.CrossEntropyLoss()  #loss损失器
    optimizer = optim.SGD([
        {'params': net.att1.parameters(), 'lr': 0.002},
        {'params': net.att2.parameters(), 'lr': 0.002},
        {'params': net.att3.parameters(), 'lr': 0.002},
        {'params': net.fc1.parameters(), 'lr': 0.002},
        {'params': net.fc2.parameters(), 'lr': 0.002},
        {'params': net.fc3.parameters(), 'lr': 0.002},
        {'params': net.down.parameters(), 'lr': 0.002},
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002},
        {'params': net.cbam.parameters(), 'lr': 0.002},
        {'params': net.bam.parameters(), 'lr': 0.002},
        {'params': net.skconv.parameters(), 'lr': 0.002},

    ],
        momentum=0.9, weight_decay=5e-4)


    # 这段代码是用SGD优化器来定义模型的学习率和需要更新的参数。
    # 具体来说，这个优化器被用于对一个名为; "net"; 的模型进行优化。"lr"; 参数指定了每个参数组的学习率。
    # "momentum"; 参数设置了优化器的动量参数，而; "weight_decay"; 参数是用于控制L2正则化的权重衰减。
    # 不是每次训练都会用到所有参数。在一个典型的深度学习模型中，参数可以被分为几个不同的组件，
    # 例如卷积层、全连接层、归一化层等等。在训练过程中，通常会使用反向传播算法计算每个参数对于损失函数的梯度，
    # 然后根据梯度来更新参数。而更新哪些参数以及使用什么样的学习率是由优化器（如SGD、Adam等）决定的。因此，在每次训练时，
    # 优化器可能只会更新某些特定组件的参数，而不是全部参数。
    max_val_acc = 0
    test_acc_max=0.0
    # lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002,0.002, 0.002, 0.0002,0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        logging.info('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        b = 0.2
        weight_loss = []
        for batch_idx, (inputs, targets) in enumerate(loader_train):
            # resnet_label = torch.LongTensor([0 if (i == 0 or i == 2) else 1 for i in targets]).to('cuda')  # 0和2代表喝水
            resnet_label = None
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Step 1
            optimizer.zero_grad()
            output_1, inputs1, resnet_loss = net(inputs, targets, 1, 0, resnet_label)  #特征图/分类结果，新图，残差损失  1
            loss1 = (CELoss(output_1, targets) *1 + resnet_loss)
            # loss1 = abs(loss1-b) + b
            loss1.backward()
            optimizer.step()
            loss1.item()
            loss5=loss1.clone()
            # loss5=0
            loss5 *= 0
            weight_loss.append(loss1)

            if epoch >10:
                # Step 2
                optimizer.zero_grad()
                output_2, inputs2, resnet_loss = net(inputs1, targets, 2, 0,resnet_label) #2
                loss2 = (CELoss(output_2, targets) *1+ resnet_loss)
                # loss2 = abs(loss2 - b) + b
                loss2.backward()
                optimizer.step()
                weight_loss.append(loss2)
            else:
                loss2=loss5;
                weight_loss.append(loss2)
            #
            if epoch >15:
                # Step 3
                optimizer.zero_grad()
                output_3, inputs3, resnet_loss = net(inputs2, targets, 3, 0, resnet_label) # 3
                loss3 = (CELoss(output_3, targets) * 1 + resnet_loss)
                # loss3 = abs(loss3 - b) + b
                loss3.backward()
                optimizer.step()
                weight_loss.append(loss3)
            else:
                loss3 =loss5;
                weight_loss.append(loss3)
            # Step 4
            optimizer.zero_grad()
            # output_concat, inputs1, resnet_loss = net(inputs, targets, 1, 0, resnet_label)
            # output_concat, resnet_loss = net(x=inputs, index=4, loss=torch.tensor(weight_loss), target=resnet_label)
            # output_concat, resnet_loss = net(inputs1, targets, 4, 0, resnet_label, loss=torch.tensor(weight_loss))
            if epoch <=10:
                 output_concat, resnet_loss = net(inputs1, targets, 4, 0, resnet_label, loss=torch.tensor(weight_loss))
            elif epoch <=15:
                 output_concat, resnet_loss = net(inputs2, targets, 4, 0, resnet_label, loss=torch.tensor(weight_loss))
            else:
                 output_concat, resnet_loss = net(inputs3, targets, 4, 0, resnet_label, loss=torch.tensor(weight_loss))

            concat_loss = (CELoss(output_concat, targets) * 1 + resnet_loss)
            # concat_loss = abs(concat_loss - b) + b
            concat_loss.backward()
            optimizer.step()

            #  training log
            logits = F.softmax(output_concat, dim=1)
            predicted = torch.argmax(logits, dim=-1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if epoch <=10:
                train_loss += (loss1.item()  + concat_loss.item())
                train_loss1 += loss1.item()
                train_loss2 +=0
                train_loss3 +=0
                train_loss4 += concat_loss.item()
            elif epoch <= 15:
                train_loss += (loss1.item() + loss2.item()  + concat_loss.item())
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 +=0
                train_loss4 += concat_loss.item()
            else:
                train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
                train_loss4 += concat_loss.item()
            # train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            # train_loss += (loss1.item()  + concat_loss.item())
            # train_loss1 += loss1.item()
            # train_loss2 += loss2.item()
            # train_loss3 += loss3.item()
            # train_loss4 += concat_loss.item()
            if batch_idx % 20 == 0 and batch_idx > 0:
                logging.info(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train.txt', 'a') as file:
                file.write(
                    'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                        epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1),
                        train_loss3 / (idx + 1),
                        train_loss4 / (idx + 1)))
        if epoch == 0:
            val_acc, val_acc_com, val_loss ,mAP= test_PMG(net, CELoss, batch_size, test_path,test_acc_max, numclass,store_name)
            logging.info('Val_acc:{}  test_acc_max:{}  Val_com_acc:{}   Val_loss:{} val_mAP:{}'.format(val_acc,test_acc_max, val_acc_com, val_loss,mAP))
        if epoch > 1 and epoch % 2 == 0:
            val_acc, val_acc_com, val_loss,mAP= test_PMG(net, CELoss, batch_size, test_path,test_acc_max, numclass,store_name)
            logging.info('Val_acc:{}   test_acc_max:{} Val_com_acc:{}   Val_loss:{} val_mAP:{}'.format(val_acc,test_acc_max,val_acc_com,val_loss,mAP))
            net.train()
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                test_acc_max = val_acc

                torch.save(net, model_save_root_path + store_name + '/train.pth')
                torch.save(net.state_dict(), model_save_root_path + store_name + '/test.pth')
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc_max = %.5f,test_acc_combined = %.5f, test_loss = %.6f test_mAP=%.6f\n' % (
                    epoch, val_acc, test_acc_max,val_acc_com, val_loss,mAP))

train_PMG(nb_epoch=25,
         batch_size=12,
         numclass=30,
         # store_name='MFGB',
         # store_name='APMG_PPMI',
         # store_name='DAR',
         #  store_name='PBRDC',  #粗粒度
          store_name='PBRDF',    #细粒度
         # store_name='play_instrument',
         # store_name='with_instrument',
         #  store_name='monkey_xi',
         # store_name='PPMI24',
         # train_path='./datasets/with_instrument/train',
         # test_path='./datasets/with_instrument/val',
         # train_path='G://论文//datasets//datasets//play_instrument//train',
         # test_path='G://论文//datasets//datasets//play_instrument//val',
         #  train_path='G://论文//datasets//datasets//with_instrument//train',
         #  test_path='G://论文//datasets//datasets//with_instrument//val',
         # train_path='G://论文//datasets//datasets//DAR//train',
         # test_path='G://论文//datasets//datasets//DAR//val',
         #  train_path='G://论文//datasets//datasets//PBRD//PrimateCoarseGrainedBehavior//train',
         #  test_path='G://论文//datasets//datasets//PBRD//PrimateCoarseGrainedBehavior//test',
          train_path='G://论文//datasets//datasets//PBRD//PrimateFineGrainedBehavior//train',
          test_path='G://论文//datasets//datasets//PBRD//PrimateFineGrainedBehavior//test',
        # train_path='G://论文//datasets//datasets//PPMI//train',
        # test_path='G://论文//datasets//datasets//PPMI//val',
        #  train_path='G://论文//datasets//datasets//monkey_xi//train',
        #  test_path='G://论文//datasets//datasets//monkey_xi//test',
        #   train_path='G://论文//datasets//datasets//with_thing//train',
        #   test_path='G://论文//datasets//datasets//with_thing//val',
        #     train_path='G://论文//datasets//datasets//MFGB//train',
        #     test_path='G://论文//datasets//datasets//MFGB//val',
         model_path=None)

# torch.cuda.synchronize()
# time1 = datetime.now()
# test_net = torch.load('./save_path/pth/APMG_PPMI/PPMI_max.pth')
# test_net.to('cuda')
# test_acc, test_com, test_loss = test_PMG(net=test_net, criterion=nn.CrossEntropyLoss(), batch_size=32, test_path='./datasets/PPMI/val')
# logging.info('acc: %.4f, com_acc: %.4f, loss: %.6f ' %(test_acc, test_com, test_loss ))
# torch.cuda.synchronize()
# time2 = datetime.now()
# times = time2 - time1
# print('times:' + str(times))
