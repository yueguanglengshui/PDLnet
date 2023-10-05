import torch.nn as nn
import torch
import torch.nn.functional as F
from util.Resnet import resnet50


class RAPMG_com(nn.Module):
    def __init__(self, model, feature_size=512, classes_num=4):
        super(RAPMG_com, self).__init__()

        self.features = model
        self.att1 = SE(patch=16,in_c=feature_size,num_heads=8)
        self.att2 = SE(patch=8,in_c=feature_size*2,num_heads=16)
        self.att3 = SE(patch=8,in_c=feature_size*4,num_heads=16)

        self.fc1 = nn.Linear(feature_size, classes_num)
        self.fc2 = nn.Linear(feature_size * 2, classes_num)
        self.fc3 = nn.Linear(feature_size * 4, classes_num)
        self.down = nn.Linear(feature_size * 7, feature_size)
        self.classifier_concat = nn.Linear(feature_size, classes_num)

    def forward(self, x, index, loss=None, target=None):
        x, xf1, xf2, xf3, xf4, xf5 = self.features(x)
        # family_targets = get_family_target(target)
        # order_targets, family_targets = get_order_family_target(targets)

        if target is not None:
            sc_loss = (torch.exp(F.cross_entropy(x, target)) - 1) * 0.5
            # sc_loss = F.cross_entropy(x, target) * 0.5
        else:
            sc_loss = 0
        if index == 1:
            out1 = self.att1(xf3)
            out1 = self.fc1(out1)
            return out1, sc_loss
        if index == 2:
            out2 = self.att2(xf4)
            out2 = self.fc2(out2)
            return out2, sc_loss
        if index == 3:
            out3 = self.att3(xf5)
            out3 = self.fc3(out3)
            return out3, sc_loss
        if index == 4:
            out1 = self.att1(xf3)
            f1 = self.fc1(out1)
            out2 = self.att2(xf4)
            f2 = self.fc2(out2)
            out3 = self.att3(xf5)
            f3 = self.fc3(out3)
            if loss is not None:
                # hidden_weight = torch.sub(1, F.softmax(loss,dim=-1))
                hidden_weight = (1-(loss-torch.min(loss))/(torch.max(loss)-torch.min(loss)))+torch.mean(loss)
                x_concat = torch.cat([out1 * hidden_weight[0], out2 * hidden_weight[1], out3 * hidden_weight[2]], -1)

            else:
                x_concat = torch.cat([out1, out2, out3], -1)

            x_concat = F.dropout(self.down(x_concat), p=0.1)
            x_concat = self.classifier_concat(x_concat)
            output_com = f1 + f2 + f3 + x_concat
            return output_com, x_concat, sc_loss

    # def forward(self, x, target):
    #     x, xf1, xf2, xf3, xf4, xf5 = self.features(x)
    #     sc_loss = torch.exp(F.cross_entropy(F.softmax(x, dim=-1), target)) - 1
    #     xl1 = self.conv_block1(xf3)
    #     xl2 = self.conv_block2(xf4)
    #     xl3 = self.conv_block3(xf5)
    #
    #     xl1 = self.max1(xl1)
    #     xl1 = xl1.view(xl1.size(0), -1)
    #     xc1 = self.classifier1(xl1)
    #
    #     xl2 = self.max2(xl2)
    #     xl2 = xl2.view(xl2.size(0), -1)
    #     xc2 = self.classifier2(xl2)
    #
    #     xl3 = self.max3(xl3)
    #     xl3 = xl3.view(xl3.size(0), -1)
    #     xc3 = self.classifier3(xl3)
    #
    #     x_concat = torch.cat((xl1, xl2, xl3), -1)
    #     x_concat = self.classifier_concat(x_concat)
    #     return xc1, xc2, xc3, x_concat, sc_loss


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


class Attention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
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


class SE(nn.Module):
    def __init__(self, patch=8, in_c=512,num_heads=8):
        super(SE, self).__init__()
        self.patch = patch
        self.avg = nn.AdaptiveAvgPool2d(patch)
        self.att = Attention(dim=in_c,num_heads=num_heads)
        self.avg2 = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(in_c)
    def forward(self, x):
        b, c, h, w = x.size()
        if h > self.patch:
            x = self.avg(x)
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.ln(x)
        x = self.att(x)
        x = self.avg2(x.transpose(1, 2))
        out = x.flatten(1)
        return out


if __name__ == '__main__':
    t = torch.randn(2, 3, 256, 256)
    resnet = resnet50(pretrained=False)
    m = RAPMG(resnet)
    l = torch.tensor([1.5, 1.4, 1.3])
    r = m(t, 4, l)
    print(r)
