import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.amax(dim=2)  # [B,1024]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(-1, 3, 3)
        x = x + torch.eye(3, device=x.device)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.amax(dim=2)  # [B,1024]
        if self.global_feat:
            return x, trans
        else:
            _, _, N = pointfeat.shape
            x = x[..., None].repeat_interleave(N, dim=-1)
            return torch.cat([x, pointfeat], 1), trans


class PointNet1(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, trans = self.feat(x)
        x2 = F.relu(self.bn1(self.fc1(x1)))
        x3 = F.relu(self.bn2(self.fc2(x2)))
        x4 = self.fc3(x3)
        feature = torch.cat((x1, x2, x3, x4), dim=1)
        return feature


def pretrained_pointnet(
    dataset: str = "shapenet", device: torch.device = "cpu", compile: bool = True
) -> nn.Module:
    if dataset == "shapenet":
        model = PointNet1(k=16)
        state_dict = load_state_dict_from_url(
            url="https://github.com/microsoft/SpareNet/raw/main/Frechet/cls_model_39.pth",
            progress=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    model.load_state_dict(state_dict)
    model.eval().requires_grad_(False)
    if compile:
        model = torch.compile(model)
    return model.to(device)
