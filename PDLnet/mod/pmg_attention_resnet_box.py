import copy
import random

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from util.Resnet import resnet50
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def cut_location(index, xy, remain_len):
    if index <= remain_len:
        a = 0
        b = remain_len * 2
    else:
        if xy <= remain_len:
            b = 256
            a = 256 - remain_len * 2
        else:
            b = index + remain_len
            a = index - remain_len
    return a, b

# 这段代码的作用是对输入的一批图像进行处理，根据神经网络模型中某一层的权重，找到每张图像中最重要的特征，然后将其在图像中心进行剪裁。
# 这个剪裁的大小由cut_rate参数控制，默认值为0.1。原始图像的大小假定为256x256。
#
# 该函数循环遍历输入的每张图像，计算出一个特征图。然后，它找到特征图中最大值的索引，并计算出该特征在原始图像中的x和y坐标。
# 然后，它使用这些坐标确定在哪里进行剪裁。剪裁的大小由一个固定的cut_len（计算为图像高度的0.1倍）和剩余部分（在剪裁区域两侧）组成。
# 最后，它将剪裁的图像重新调整为256x256，并返回修改后的图像批次。


# 这段代码的目的是通过剪裁来关注并突出显示每张图像中最重要的特征。
# 这对于图像分类、目标检测、语义分割等任务非常有用，可以帮助深度学习模型更好地理解和分析图像内容。
# 通过突出显示最重要的特征，可以提高模型的准确性和可解释性，使模型更好地应用于实际问题中。
# 此外，这种方法还可以减少输入图像的大小和分辨率，从而加快模型的计算速度。
# 因此，该方法在深度学习图像处理中具有很高的实用性和研究价值。
def get_cut_img(orig_img, layer_weights, cut_rate=0.1):
    # 这几行代码定义了输入图像的大小和数量以及一个新的变量new_imgs。
    # 其中image_shape是一个元组，指定了图像的高度和宽度，这里假设为256x256。
    # num是输入图像的数量，它是通过调用orig_img的size方法获取的，size(0)
    # 返回的是批次的大小。new_imgs初始化为None，将用于存储剪裁后的图像。
    image_shape = (256, 256)
    num = orig_img.size(0)
    new_imgs = None

    # 这段代码是对每张输入图像进行剪裁，以突出显示最重要的特征。具体地说，代码会遍历所有输入图像，并且对每一张图像进行以下处理：
    #
    # 1.
    # 通过layer_weights[i]
    # 计算当前图像的特征图，然后求特征图的平均值，得到一个形状为（H，W）的矩阵feature_map；
    # 2.
    # 在feature_map中找到值最大的位置，即最大索引cam_max_index；
    # 3.
    # 通过cam_max_index和feature_map的形状，计算出最大值所在的位置坐标feature_max_x和feature_max_y；
    # 4.
    # 通过feature_max_x和feature_max_y，以及输入图像大小image_shape和feature_map的形状，计算出剪裁区域的起始位置index_x和index_y；
    # 5.
    # 根据剪裁率cut_rate和剪裁区域的起始位置index_x和index_y，计算出剪裁区域的长度cut_len和剩余区域的长度remain_len；
    # 6.
    # 根据剪裁区域的位置和长度，计算出图像剪裁的左右边界x_left和x_right，以及上下边界y_top和y_down；
    # 7.
    # 根据图像剪裁的边界，从输入图像中剪裁出新的图像区域，得到一个形状为（C，H，W）的张量new_img；
    # 8.
    # 对new_img进行插值，将其缩放到256x256的大小，然后将其存储到new_imgs中。
    #
    # 最终，代码会返回一个形状为（N，C，H，W）的张量new_imgs，其中N是输入图像的数量，C是通道数，H和W是图像的高度和宽度。该张量包含了对所有输入图像进行剪裁后得到的新图像。
    for i in range(0, num):
        feature_map = layer_weights[i].mean(0)
        feature_size = feature_map.size()
        cam_max_index = feature_map.argmax()  # 最大索引

        # feature_max_x = cam_max_index // feature_size[0]
        feature_max_x =torch.div(cam_max_index, feature_size[0], rounding_mode='trunc')
        feature_max_y = cam_max_index % feature_size[1]
        # index_x = (image_shape[0]*feature_max_x) // feature_size[0]
        index_x =torch.div(image_shape[0]*feature_max_x, feature_size[0], rounding_mode='trunc')
        # index_y = (image_shape[1]*feature_max_y) // feature_size[1]
        index_y =torch.div(image_shape[1]*feature_max_y, feature_size[1], rounding_mode='trunc')
        cut_len = int(image_shape[0] * cut_rate)  # 49
        remain_len = (image_shape[0] - cut_len) // 2  # 207
        x = image_shape[0] - index_x
        y = image_shape[1] - index_y  # 20
        x_left, x_right = cut_location(index_x, x, remain_len)
        y_top, y_down = cut_location(index_y, y, remain_len)
        new_img = orig_img[i][:, x_left:x_right, y_top:y_down].clone().detach().cuda()
        new_img = F.interpolate(new_img.unsqueeze(0), (256, 256), mode='bilinear',align_corners=True).clone().detach().cuda()
        if i == 0:
            new_imgs = new_img
        else:
            new_imgs = torch.cat([new_imgs, new_img], dim=0)
    return new_imgs


class ImageCut(nn.Module):
    def __init__(self, cut_rate=0.1):
        super(ImageCut, self).__init__()
        self.cut_rate = cut_rate
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, feature_map):
        feature_weight = self.avg(feature_map)
        se_weight = F.relu(feature_weight*feature_map)
        new_img = get_cut_img(x, se_weight, self.cut_rate)
        return new_img

# net = RAPMG(net, 512, classes_num=num_class)
class RAPMG(nn.Module):
    def __init__(self, model, feature_size=512, classes_num=4):
        super(RAPMG, self).__init__()

        self.features = model
        self.att1 = SE(patch=16, in_c=feature_size, num_heads=8)
        self.att2 = SE(patch=8, in_c=feature_size * 2, num_heads=16)
        self.att3 = SE(patch=8, in_c=feature_size * 4, num_heads=16)
        self.cbam=CBAM(in_channels=feature_size)
        self.bam =BAM(channels=feature_size * 2)

        #self.skconv = SKConv(in_ch=feature_size*4, M=3, G=1, r=2)
        self.skconv=CBAM(in_channels=feature_size*4)
        #self.bam=CBAM(in_channels=feature_size* 2)
        # self.bam= BAM_SK(in_channels=feature_size * 2, out_channels=feature_size * 2)
        # self.bam_skm=BAM_SK(in_channels=feature_size * 4)

       # self.bam_skm =CBAM(in_channels=feature_size* 4)
        self.similarity = SIMILARITY(classes_num, feature_size*4)
        self.fc1 = nn.Linear(feature_size, classes_num)
        self.fc_weight = nn.Linear(feature_size, classes_num)
        self.fc2 = nn.Linear(feature_size * 2, classes_num)
        self.fc3 = nn.Linear(feature_size * 4, classes_num)
        self.cut1 = ImageCut(cut_rate=0.05)
        self.cut2 = ImageCut(cut_rate=0.07)
        self.cut3 = ImageCut(cut_rate=0.1)
        self.down = nn.Linear(feature_size * 7, feature_size)
        self.classifier_concat = nn.Linear(feature_size, classes_num)
        # self.avg_pool2d1 = nn.functional.avg_pool1d(2)
        # self.avg_pool2d2 = nn.functional.avg_pool1d(4)
        # device = torch.device("cuda:0")
        # self.similarity = torch.zeros(classes_num, 512).to(device)
        # self.similarity_weight = torch.zeros(classes_num, 512).to(device)
        # self.class_counts = torch.zeros(classes_num).to(device)
        # self.att_out_repair=torch.zeros(16, 512).to(device)
    def forward(self, x, y, index,  cut_rate_index, resnet_target, loss=None):

        x_2, xf1, xf2, xf3, xf4, xf5 = self.features(x)    #分类结果，下采样图1，下采样图2，下采样图3，下采样图4，下采样图5


        sc_loss = 0

        if index == 1:

            cbam=self.cbam(xf3)

            att_out1 = self.att1(cbam)

            out1 = self.fc_weight(att_out1)
            cut1 = ImageCut(cut_rate_index)
            # new_img1 = self.cut1(x, xf3)
            new_img1 = cut1(x, xf3)
            return out1, new_img1, sc_loss
            # return out1, x, sc_loss   #不对图像进行裁切，直接用原数据

        if index == 2:
            bama = self.bam(xf4)
            # skconv=self.skconv(xf4)
            att_out2 = self.att2(bama)
            out2 = self.fc2(att_out2)
            cut2 = ImageCut(cut_rate_index)
            # new_img2 = self.cut2(x, xf4)
            new_img2 = cut2(x, xf4)
            return out2, new_img2, sc_loss
            # return out2, x, sc_loss

        if index == 3:
           # cbam = self.cbam(xf5)
           # cbam = self.cbam(xf3)
            bam_skm = self.skconv(xf5)
            att_out3 = self.att3(bam_skm)

            # att_out_repair = self.similarity(att_out3)
            out3 = self.fc3(att_out3)
            cut4 = ImageCut(cut_rate_index)
            new_img3 = cut4(x, xf5)
            # new_img3 = self.cut3(x, xf5)
            return out3, new_img3, sc_loss
            # return out3, x, sc_loss


        # if index == 4:
        #     # 最后，代码通过调用x.flatten(1)函数将形状为(batch_size, channels, 1)的张量x展平成一个形状为(batch_size, channels)的张量out，以便进行后续的处理。
        #     att_out1 = self.att1(xf3)
        #     att_out2 = self.att2(xf4)
        #     att_out3 = self.att3(xf5)
        #     if loss is not None:
        #        hidden_weight = (1 - (loss - torch.min(loss)) / (torch.max(loss) - torch.min(loss))) + torch.mean(loss)
        #
        #        # x_concat = torch.cat([att_out1 * hidden_weight[0], att_out2 * hidden_weight[1], att_out3 * hidden_weight[2]], -1)
        #        weights = torch.tensor([0.5, 0.3, 0.2], requires_grad=False)
        #
        #        att_out2 = torch.nn.functional.avg_pool1d(att_out2, 2)
        #        att_out3 = torch.nn.functional.avg_pool1d(att_out3,4)
        #        x_concat= att_out1 * weights[0] + att_out2 * weights[1] + att_out3 * weights[2]
        #        # x_concat = torch.cat([att_out1 * hidden_weight[0], att_out2 * hidden_weight[1], att_out3 * hidden_weight[2]], -1)
        #        # att_out1 = torch.cat((att_out1, att_out2.unsqueeze(1)), dim=-1)
        #        # att_out1 = torch.cat((att_out1, att_out3.unsqueeze(1)), dim=-1)
        #        # x_concat = torch.mean(att_out1, dim=-1)
        #        # att_out2 = torch.mean(att_out2, dim=-1)
        #        # att_out3 = torch.mean(att_out3, dim=-1)
        #     else:
        #        # x_concat = torch.cat([att_out1, att_out2, att_out3], -1)
        #        weights = torch.tensor([0.5, 0.3, 0.2], requires_grad=False)
        #
        #        att_out2 = torch.nn.functional.avg_pool1d(att_out2, 2)
        #        att_out3 = torch.nn.functional.avg_pool1d(att_out3, 4)
        #        x_concat = att_out1 * weights[0] + att_out2 * weights[1] + att_out3 * weights[2]
        #     # x_concat = F.dropout(self.down(x_concat), p=0.1)
        #     x_concat = F.dropout(x_concat, p=0.1)
        #     x_concat = self.classifier_concat(x_concat)
        #     return x_concat, sc_loss
        if index == 4:
            # 最后，代码通过调用x.flatten(1)函数将形状为(batch_size, channels, 1)的张量x展平成一个形状为(batch_size, channels)的张量out，以便进行后续的处理。
            att_out1 = self.att1(xf3)
            att_out2 = self.att2(xf4)
            att_out3 = self.att3(xf5)
            if loss is not None:
                hidden_weight = (1 - (loss - torch.min(loss)) / (torch.max(loss) - torch.min(loss))) + torch.mean(loss)
                # x_concat = torch.cat([att_out2 * hidden_weight[0], att_out3 * hidden_weight[1]], -1)
                x_concat = torch.cat(
                    [att_out1 * hidden_weight[0], att_out2 * hidden_weight[1], att_out3 * hidden_weight[2]], -1)
            else:
                x_concat = torch.cat([att_out1, att_out2, att_out3], -1)
            # x_concat = torch.cat([att_out2, att_out3], -1)
            # x_concat = torch.cat([att_out3], -1)
            x_concat = F.dropout(self.down(x_concat), p=0.1)
            x_concat = self.classifier_concat(x_concat)
            return x_concat, sc_loss

class SIMILARITY(nn.Module):
    def __init__(self, classes_num, feature_size):
        super().__init__()
        device = torch.device("cuda:0")
        self.fc1 = nn.Linear(feature_size, classes_num).to(device)
        self.similarity = torch.zeros(classes_num, feature_size).to(device)
        self.similarity_weight = torch.zeros(classes_num, feature_size).to(device)
        self.class_counts = torch.zeros(classes_num).to(device)
        # self.att_out_repair=torch.zeros(16, feature_size)
    def forward(self, x):
        out = self.fc1(x)
        # out =nn.Linear(feature_size, classes_num)
        # 沿着列的方向计算最大值
        max_values= torch.argmax(out, dim=1)
        for index in max_values:
            self.class_counts[index]+=1
        # 形成相似矩阵
        for col in range(x.size(0)):  #att [16,512]max_values[col]
            self.similarity[max_values[col]] += x[col]
        #
        #
        #相似矩阵求均值
        for col in range(self.similarity.size(0)):  #att [16,512]max_values[col]
            self.similarity_weight[col]= self.similarity[col]/self.class_counts[col]
        #利用均值对值进行修复
        # att_out_repair=att_out1
        att_out_repair=x*0.9
        # for col in range(x.size(0)):  # att [16,512]max_values[col]
        #    x[col]= x[col]*0.7+self.similarity_weight[max_values[col]]*0.3
        return att_out_repair


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#这是一个自注意力机制（Self-Attention）的PyTorch实现，主要用于处理序列数据，例如自然语言文本或音频数据。
# 自注意力机制可以学习到序列中不同位置之间的相互关系，从而在处理序列数据时取得很好的效果。
# 在这个实现中，输入的序列被转换为一个三维张量（B x N x C），其中 B 是 batch size，N 是序列长度，C 是特征维度。通过一个线性变换（qkv），
# 序列中每个位置被转换为一个三元组（query、key 和 value），
# 并且这些三元组被分成 num_heads 个头（head），以便于并行计算。
# 接着，对于每个头，query 和 key 进行矩阵乘法运算，得到一个大小为 N x N 的注意力矩阵。这个矩阵代表了序列中不同位置之间的相互关系，
# 因此称为自注意力矩阵。这个矩阵经过 softmax 归一化后，
# 与 value 进行加权平均，得到一个大小为 N x C/num_heads 的输出。
# 最后，将 num_heads 个输出沿着最后一个维度拼接起来，通过一个线性变换和一个 dropout 层得到最终的输出。
# 在实现中，还支持了 attention mask 的功能，可以用于控制模型关注序列中的特定位置。
    #self.att = Attention(dim=in_c, num_heads=num_heads) in_c=512 num_head=8
class Attention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        # qk_scale：指定缩放因子，
        # attn_drop：注意力得分的dropout概率；
        # proj_drop：注意力计算后的投影层的dropout概率。
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww      局部注意力窗口不用管
        self.num_heads = num_heads
        head_dim = dim // num_heads         #每个注意力头的特征维度，即输入特征维度除以头数；好像没用到
        self.scale = qk_scale or head_dim ** -0.5    #缩放因子，用于调整注意力得分的尺度；
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #dim只有一维，输入时就是1维，可以看着一张图被拉成1维
        # #用于计算查询、键和值的线性映射，将输入张量的特征维度映射到三倍的特征维度上，分别用于计算查询、键和值；
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)   #将注意力计算后的值向量进行线性映射的模块；
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # 将输入张量x通过线性映射self.qkv，映射为形状为(batch_size, num_patches, 3 * num_heads, head_dim)的张量qkv；
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将qkv张量进行形状变换，变为(3, batch_size, num_heads, num_patches, head_dim)的张量qkv；
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 将 qkv张量沿着第一个维度分割为三个张量q、k和v；
        q = q * self.scale
        # 对q和k进行缩放和点积操作，得到未归一化的注意力得分张量attn；
        attn = (q @ k.transpose(-2, -1))
        # 如果提供了掩码mask，则在计算注意力得分前，先将掩码与attn进行相加；
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# 这段代码定义了一个名为 "SE" 的自注意力机制，可用作神经网络中的构建模块。SE模块以4D张量作为输入（批大小、通道数、高度、宽度），并执行以下操作：
#
# 首先对输入张量应用自适应平均池化操作，将特征图的高度和宽度缩减为固定大小的 patch x patch。这有助于降低后续操作的计算成本。
#
# 然后将池化后的特征图应用多头自注意力机制（Attention），这使得模块可以关注图像中最相关的空间位置。num_heads 参数控制在机制中使用的注意力头数。
#
# 自注意力机制之后，模块再次应用自适应平均池化操作，这次将特征图缩减为大小为 in_c 的1D张量。该操作计算特征的通道重要性并减小张量的大小。
#
# 最后，输出张量被展平并返回。
#
# nn.LayerNorm 用于归一化多头自注意力机制的输出。这有助于稳定训练过程并提高网络性能。

   # self.att1 = SE(patch=16, in_c=feature_size, num_heads=8)
class SE(nn.Module):
    def __init__(self, patch=8, in_c=512, num_heads=8):   #512通道
        super(SE, self).__init__()
        self.patch = patch
        self.avg = nn.AdaptiveAvgPool2d(patch)   #长宽浓缩变为16*16    通道还是不变维持原来数量
        self.att = Attention(dim=in_c, num_heads=num_heads)   #自注意力机制  num_heads 注意力头
        self.avg2 = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(in_c)

    def forward(self, x):
        b, c, h, w = x.size()     #b批次   c通道数  h 高度  w宽度
        if h > self.patch:          #如果输入的图像还是比较大，就在压缩一边
            x = self.avg(x)
        x = x.view(b, c, -1).transpose(1, 2)   # 转换维度方便注意力计算
        # 这行代码将一个形状为(batch_size, channels, height, width)
        # 的4D张量x转换为形状为(batch_size, channels, height * width)的3D张量x。具体来说，它将原始张量x
        # 沿着其最后两个维度进行展平，其中 - 1表示自动计算缺失的维度大小。
        # 接下来，它使用transpose函数将维度1和维度2进行交换，从而得到一个形状为(batch_size, height * width, channels)
        # 的张量，这是为了方便在自注意力机制中进行计算。换句话说，这个操作将channels
        # 维度移动到了最后一个维度，使得在进行自注意力计算时，张量的最后一个维度表示通道数，方便计算。

        # `in_c`表示输入张量的通道数（或称为特征数或深度），在这个模块中用来指定自注意力机制的输入特征维度。具体来说，输入张量的形状为
        # `(batch_size, in_c, height, width)`，其中`in_c`就是通道数，`height`和`width`
        # 分别表示输入图像的高度和宽度。在这个模块中，我们使用自适应平均池化层将图像块的高度和宽度缩减为`patch x patch`，所以输入张量的形状会变为
        # `(batch_size, in_c, patch, patch)`。在进行自注意力计算时，我们需要将输入张量视为一个形状为`(batch_size, num_patches, in_c)`的序列，因此
        # `in_c`也是自注意力机制中的输入特征维度。


        x = self.ln(x)    #正则化

        # 这行代码将输入张量 x
        # 应用到名为self.att
        # 的自注意力机制上。在此之前，x
        # 已经被转换为一个形状为(batch_size, patches  * patches, channels)的张量，其中num_patches是通过自适应平均池化缩减高度和宽度后得到的图像块的数量。
        # 自注意力机制是一种可以学习到输入序列之间关系的神经网络层。在该模块中，self.att实现了一个多头注意力机制，它将张量x作为输入，并计算其内部元素之间的相互依赖关系。
        # 具体来说，该操作将输入张量分成多个头（由
        # num_heads参数指定），并计算每个头的注意力权重，以捕获输入张量中的不同信息。最终，该操作将每个头的输出合并起来，以形成具有全局上下文的最终输出。
        # 因此，该行代码执行的操作是将输入张量x送入自注意力模块中，以对输入进行加权聚合，并提取全局特征表示。
        x = self.att(x)   #提取全局特征表示

        # 这行代码将经过自注意力机制处理后的张量x
        # 转置后传递给另一个自适应平均池化层self.avg2进行处理。具体来说，x.transpose(1, 2)操作交换了张量x的第二和第三个维度，
        # 将其形状从(batch_size, num_patches, channels)
        # 转换为(batch_size, channels, num_patches)。然后，self.avg2层对这个张量进行自适应平均池化，将其形状从(batch_size, channels, num_patches)
        # 转换为(batch_size, channels, 1)。这里1是因为nn.AdaptiveAvgPool1d(1)的参数为1，所以会将最后一个维度缩减为1。
        # 换句话说，它沿着最后一个维度对输入张量进行平均池化，输出一个形状为(batch_size, channels, 1)的张量。
        # 最后，代码通过调用x.flatten(1)函数将形状为(batch_size, channels, 1)的张量x展平成一个形状为(batch_size, channels)的张量out，以便进行后续的处理。
        x = self.avg2(x.transpose(1, 2))
        out = x.flatten(1)    # 最后一个维度为1，就直接减少减少一个维度
        return out

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels//reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels//reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        avg = self.fc2(self.relu(self.fc1(avg)))
        max_ = self.fc2(self.relu(self.fc1(max_)))
        x = avg + max_
        return self.sigmoid(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg, max_], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# class SKConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
#         super(SKConv, self).__init__()
#         d = max(in_channels // r, L)
#         self.M = M
#         self.out_channels = out_channels
#         self.conv_k_list = nn.ModuleList()
#         for i in range(M):
#             self.conv_k_list.append(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1+i, dilation=1+i, bias=False))
#         self.fc = nn.Sequential(
#             nn.Linear(out_channels, d, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(d, out_channels*M, bias=False),
#             nn.Sigmoid()
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         feats = [conv(x) for conv in self.conv_k_list]
#         feats = torch.cat(feats, dim=1)
#         feats_U = torch.sum(feats, dim=1)
#         feats_S = self.fc(self.avg_pool(feats_U)).view(-1, self.M, self.out_channels)
#         attention_vectors = torch.sum(feats_S, dim=1, keepdim=False)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         feats = [a*b for a, b in zip(feats, torch.split(attention_vectors, self.out_channels, dim=1))]
#         return torch.sum(torch.cat(feats, dim=1), dim=1)

# class SKConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, M=2, G=32, r=16, L=32):
#         super(SKConv, self).__init__()
#
#         d = max(in_channels // r, L)
#         self.M = M
#         self.out_channels = out_channels
#
#         self.convs = nn.ModuleList()
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels // M, kernel_size=3+i*2, stride=stride, padding=1+i, dilation=1, groups=G),
#                 nn.BatchNorm2d(out_channels // M),
#                 nn.ReLU(inplace=True)
#             ))
#
#         self.global_pool = nn.AdaptiveAvgPool2d((16, 16))
#         self.fc = nn.Sequential(
#             nn.Linear(out_channels // M, d),
#             nn.ReLU(inplace=True),
#             nn.Linear(d, out_channels)
#         )
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         feats = []
#         for conv in self.convs:
#             feats.append(conv(x))
#
#         feats = torch.cat(feats, dim=1)
#         feats = feats.view(batch_size, self.M, self.out_channels // self.M, feats.size(2), feats.size(3))
#         feats_U = torch.sum(feats, dim=1)
#         feats_S = self.global_pool(feats_U)
#         feats_Z = self.fc(feats_S.view(batch_size, -1))
#         feats_attention = self.softmax(feats_Z.view(-1, self.out_channels // self.M))
#
#         return torch.sum(feats_U * feats_attention.view(-1, self.out_channels // self.M, 1, 1), dim=1)
# class SKConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
#         super(SKConv, self).__init__()
#         d = max(in_channels // r, L)
#         self.M = M
#         self.out_channels = out_channels
#         self.conv_k_list = nn.ModuleList()
#         for i in range(M):
#             self.conv_k_list.append(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1+i, dilation=1+i, bias=False))
#         self.fc = nn.Sequential(
#             nn.Linear(out_channels, d, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(d, out_channels*M, bias=False),
#             nn.Sigmoid()
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         feats = [conv(x) for conv in self.conv_k_list]
#         feats = torch.cat(feats, dim=1)
#         feats_U = torch.sum(feats, dim=1)
#         feats_S = self.fc(self.avg_pool(feats_U)).view(-1, self.M, self.out_channels)
#         attention_vectors = torch.sum(feats_S, dim=1, keepdim=False)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         feats = [a * b for a, b in zip(feats, torch.split(attention_vectors, self.out_channels, dim=1))]
#         return torch.sum(torch.cat(feats, dim=1), dim=1)

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, norm_layer=None):
#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             norm_layer(out_planes),
#             nn.ReLU(inplace=True)
#         )
#
# class BAM_SK(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16, M=2, r=16, L=32):
#         super(BAM_SK, self).__init__()
#
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             ConvBNReLU(in_channels, out_channels // reduction_ratio, 1),
#             ConvBNReLU(out_channels // reduction_ratio, out_channels, 1)
#         )
#
#         self.sk_block = SKConv(out_channels, out_channels, stride, M, r, L)
#
#     def forward(self, x):
#         atten = self.channel_attention(x)
#         out = atten * x
#         out = self.sk_block(out)
#      #   out = out.view(out.size(0), out.size(1), 1, 1)
#         return out

class SKConv(nn.Module):
    def __init__(self, in_ch, M=3, G=1, r=4, stride=1, L=32) -> None:
        super().__init__()
        """ Constructor
        Args:
        in_ch: input channel dimensionality.
        M: the number of branchs.
        G: num of convolution groups.
        r: the radio for compute d, the length of z.
        stride: stride, default 1.
        L: the minimum dim of the vector z in paper, default 32.
        """
        d = max(int(in_ch/r), L)  # 用来进行线性层的输出通道，当输入数据In_ch很大时，用L就有点丢失数据了。
        self.M = M
        self.in_ch = in_ch
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3+i*2, stride=stride, padding = 1+i, groups=G),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True)
                )
            )
        # print("D:", d)
        self.fc = nn.Linear(in_ch, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, in_ch))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):  # 第一部分，每个分支的数据进行相加,虽然这里使用的是torch.cat，但是后面又用了unsqueeze和sum进行升维和降维
            fea = conv(x).clone().unsqueeze_(dim=1).clone()   # 这里在1这个地方新增了一个维度  16*1*64*256*256
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas.clone(), fea], dim=1)  # feas.shape  batch*M*in_ch*W*H
        fea_U = torch.sum(feas.clone(), dim=1)  # batch*in_ch*H*W
        fea_s = fea_U.clone().mean(-1).mean(-1)  # Batch*in_ch
        fea_z = self.fc(fea_s)  # batch*in_ch-> batch*d
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).clone().unsqueeze_(dim=1)  # batch*d->batch*in_ch->batch*1*in_ch
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors.clone(), vector], dim=1)  # 同样的相加操作 # batch*M*in_ch
        attention_vectors = self.softmax(attention_vectors.clone()) # 对每个分支的数据进行softmax操作
        attention_vectors = attention_vectors.clone().unsqueeze(-1).unsqueeze(-1) # ->batch*M*in_ch*1*1
        fea_v = (feas * attention_vectors).clone().sum(dim=1) # ->batch*in_ch*W*H
        return fea_v
#
# class SKConv(nn.Module):
#     def __init__(self, in_channels, out_channels, M=2, r=16, L=32):
#         super(SKConv, self).__init__()
#
#         self.convs = nn.ModuleList()
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 + i, dilation=1 + i,
#                           bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
#
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(nn.Linear(out_channels, out_channels // r), nn.ReLU(inplace=True),
#                                 nn.Linear(out_channels // r, L * M))
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         feats = []
#         for conv in self.convs:
#             feats.append(conv(x))
#         feats = torch.cat(feats, dim=1)
#
#         feats_U = torch.sum(feats, dim=(2, 3))
#         feats_g = self.pool(feats_U)
#         feats_z = self.fc(feats_g.view(feats_g.size(0), -1))
#         feats_z = feats_z.view(-1, self.L, self.M)
#         feats_p = self.softmax(feats_z)
#
#         feats_p = feats_p.view(-1, 1, self.L, self.M)
#
#         feats_T = torch.sum(feats * feats_p, dim=3, keepdim=True)
#         return feats_T


class BAM_SK(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, M=2, r=16, L=32):
        super(BAM_SK, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, M * in_channels, kernel_size=1, stride=1)
        )

        self.skconv = SKConv(in_ch=in_channels, M=M, r=r, L=L)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.skconv(x)

        gap = F.adaptive_avg_pool2d(out, 1)
        gap = gap.view(-1, gap.size(-1))
        attention = self.fc(gap).view(-1, x.size(1), 1, 1)

        out = self.bottleneck(out)
        out = out.view(-1, x.size(1), 2, out.size(-2), out.size(-1))
        out = out.sum(dim=2)

        out = attention * out
        out = out + x

        return out
# class BAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, dilation_conv_num=4):
#         super(BAM, self).__init__()
#
#         self.in_channels = in_channels
#         self.reduction_ratio = reduction_ratio
#         self.dilation_conv_num = dilation_conv_num
#
#         # Channel attention
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#         # Spatial attention
#         self.dilation_conv_layers = nn.Sequential(*[
#             nn.Conv2d(in_channels=2*in_channels, out_channels=1, kernel_size=3, stride=1, padding=dilation_conv_num,
#                       dilation=dilation_conv_num, bias=False)
#             for _ in range(dilation_conv_num)
#         ])
#         self.spatial_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
#                                       bias=False)
#
#     def forward(self, x):
#         # Channel-wise attention
#         avg = self.avg_pool(x)  #[4, 1024, 1, 1]
#         max_ = self.max_pool(x) #[4, 1024, 1, 1]
#
#         avg = self.fc2(self.relu(self.fc1(avg)))
#         max_ = self.fc2(self.relu(self.fc1(max_)))
#
#         channel = self.sigmoid(avg + max_)
#         channel_attention = x * channel
#
#         # Spatial-wise attention
#         # feature_map = torch.cat([channel_attention, x], 1)
#         feature_map = torch.cat([channel_attention, x[:, :self.in_channels, :, :]], 1)
#         feature_map = self.dilation_conv_layers(feature_map)
#         feature_map = self.spatial_conv(feature_map)
#
#         spatial_attention = self.sigmoid(feature_map)
#         out = x * spatial_attention
#
#         return out

class BAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(BAM, self).__init__()
        self.channels = channels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(channels*2, channels//reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        out = torch.cat((max_out, avg_out), dim=1)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        out = out.view(out.size(0), out.size(1), 1, 1)

        # 应用注意力权重
        out = out * x

        return out
if __name__ == '__main__':
    img = '../datasets/DAR/test/no_drink/gglpp120c_256x256.jpg'
    import torchvision.transforms as transforms

    img_np = Image.open(img)
    transform_train = transforms.Compose([
        transforms.RandomCrop(256, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    img_tensor = transform_train(img_np)
    t = torch.randn(2, 3, 224, 224).to('cuda')
    resnet = resnet50(pretrained=False)
    m = RAPMG(resnet).to('cuda')
    l = torch.tensor([1.5, 1.4, 1.3])
    img_tensor = img_tensor.unsqueeze(0).to('cuda')
    r = m(t, 0, 1, 1)
    # forward(self, x, y, index, resnet_target, loss=None):
    # __init__(self, model, feature_size=512, classes_num=4):
    # img = torch.randn(3,256,256)
    # t = torch.randn(1, 512, 28, 28)
    # weight = torch.randn(512)
    # r = cut_img(img, t, weights=weight)
    print(r)
