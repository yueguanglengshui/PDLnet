import torch.nn as nn
import torch
import torch.nn.functional as F
from util.adversarial import LabelSmoothSoftmaxCEV1


class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMG, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max2 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.max3 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
            # nn.Dropout(0.1),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            #  nn.Dropout(0.1),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
            #  nn.Dropout(0.1),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            #   nn.Dropout(0.1),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
            #   nn.Dropout(0.1),
        )
        # self.loss = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)

    def forward(self, x, target, index):
    # def forward(self, x):
        x, xf1, xf2, xf3, xf4, xf5 = self.features(x)
        sc_loss = F.cross_entropy(F.softmax(x, dim=-1), target)
        # sc_loss = torch.exp(self.loss(x, target)) - 1
        # sc_loss = torch.exp(F.cross_entropy(x, target)) - 1

        if index == 1:
            xl1 = self.conv_block1(xf3)
            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)
            return xc1, sc_loss
        if index == 2:
            xl2 = self.conv_block2(xf4)
            xl2 = self.max2(xl2)
            xl2 = xl2.view(xl2.size(0), -1)
            xc2 = self.classifier2(xl2)
            return xc2, sc_loss
        if index == 3:
            xl3 = self.conv_block3(xf5)
            xl3 = self.max3(xl3)
            xl3 = xl3.view(xl3.size(0), -1)
            xc3 = self.classifier3(xl3)
            return xc3, sc_loss
        if index == 4:
            xl1 = self.conv_block1(xf3)
            xl2 = self.conv_block2(xf4)
            xl3 = self.conv_block3(xf5)
            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)

            xl2 = self.max2(xl2)
            xl2 = xl2.view(xl2.size(0), -1)
            xc2 = self.classifier2(xl2)

            xl3 = self.max3(xl3)
            xl3 = xl3.view(xl3.size(0), -1)
            xc3 = self.classifier3(xl3)
            x_concat = torch.cat((xl1, xl2, xl3), -1)
            x_concat = self.classifier_concat(x_concat)
            output_com = xc1 + xc2 + xc3 + x_concat
            return output_com, x_concat, sc_loss
            # return x_concat

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
