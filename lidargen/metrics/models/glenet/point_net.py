import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_


class PointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1):
        super(PointNetfeat, self).__init__()
        self.output_channel = int(512 * x)
        self.conv1 = torch.nn.Conv1d(pts_dim, int(64 * x), 1)
        self.conv2 = torch.nn.Conv1d(int(64 * x), int(128 * x), 1)
        self.conv3 = torch.nn.Conv1d(int(128 * x), self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(int(64 * x))
        self.bn2 = nn.BatchNorm1d(int(128 * x))
        self.bn3 = nn.BatchNorm1d(self.output_channel)
        # 1d convolution sequential layers (self.output_channel+512) --> 768 --> 512
        self.output_sequential = nn.Sequential(
            nn.Linear(self.output_channel + 512, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
        )   

    def forward(self, x_dict):
        x = x_dict['x']
        text_feat = x_dict['text_feat']
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # NOTE: should not put a relu
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_channel)
        x = torch.cat((x, text_feat), dim=1)  # Concatenate text features
        x = self.output_sequential(x)  # Apply the sequential layers
        return x

class SimPointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1):
        super(SimPointNetfeat, self).__init__()
        self.output_channel = int(16 * x)
        self.conv1 = torch.nn.Conv1d(pts_dim, int(16 * x), 1)
        self.conv2 = torch.nn.Conv1d(int(16 * x), int(16 * x), 1)
        self.conv3 = torch.nn.Conv1d(int(16 * x), self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(int(16 * x))
        self.bn2 = nn.BatchNorm1d(int(16 * x))
        self.bn3 = nn.BatchNorm1d(self.output_channel)
        self.output_sequential = nn.Sequential(
            nn.Linear(self.output_channel + 512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_channel),
        )   

    def forward(self, x_dict):
        x = x_dict['x']
        text_feat = x_dict['text_feat']
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # NOTE: should not put a relu
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_channel)
        x = torch.cat((x, text_feat), dim=1)  # Concatenate text features
        x = self.output_sequential(x)  # Apply the sequential layers
        return x

class PointNet(nn.Module):
    def __init__(self, pts_dim, x, CLS_NUM):
        super(PointNet, self).__init__()
        self.feat = PointNetfeat(pts_dim, x)
        self.fc1 = nn.Linear(512 * x, 256 * x)
        self.fc2 = nn.Linear(256 * x, 256)

        self.pre_bn = nn.BatchNorm1d(pts_dim)
        self.bn1 = nn.BatchNorm1d(256 * x)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        # NOTE: should there put a BN?
        self.fc_s1 = nn.Linear(256, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_c1 = nn.Linear(256, 256)
        self.fc_c2 = nn.Linear(256, CLS_NUM, bias=False)
        self.fc_ce1 = nn.Linear(256, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(256, 256)
        self.fc_hr2 = nn.Linear(256, 1, bias=False)

    # def forward(self, x, pred_bbox):
    def forward(self, x):
        x = self.feat(self.pre_bn(x))
        x = F.relu(self.bn1(self.fc1(x)))
        feat = F.relu(self.bn2(self.fc2(x)))

        x = F.relu(self.fc_c1(feat))
        logits = self.fc_c2(x)

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        return logits, centers, sizes, headings

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

if __name__ == "__main__":
    model = PointNet(
        pts_dim=3,
        x=1,
        CLS_NUM=8
    )

    points = torch.randn(16, 3, 1024)
    rel = model(points)

    print(rel[0].shape)