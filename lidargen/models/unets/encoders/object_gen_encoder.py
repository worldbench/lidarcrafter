import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from .embedder import get_embedder

class ObjectGenEncoder(nn.Module):
    def __init__(self, num_class, input_dim=6, embedder_num_freq=4, class_token_dim=512, 
                 use_text_encoder_init=True, output_num=1, proj_dims=[768, 512, 512, 768],
                 object_classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']):
        super().__init__()
        self.prepare_called = False
        self.num_class = num_class
        self.fourier_embedder = get_embedder(input_dim, embedder_num_freq)
        self.use_text_encoder_init = use_text_encoder_init
        self.object_classes = object_classes
        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

    def prepare(self, device='cuda'):
        if self.use_text_encoder_init:
            obj_text_feat_path = '../data/clips/nuscenes/obj_text_feat.pkl'
            with open(obj_text_feat_path, 'rb') as f:
                self.obj_text_feat = pickle.load(f)
            for key, val in self.obj_text_feat.items():
                self.obj_text_feat[key] = torch.from_numpy(val).squeeze().to(device)
        self.prepare_called = True

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, input_dict):

        if not self.prepare_called:
            self.prepare()

        bboxes = input_dict['fg_encoding_box'] # (B, 7)
        classes = input_dict['fg_class'] # (B)

        pos_emb = self.fourier_embedder(bboxes)
        # pos_emb = pos_emb.reshape(
        #     pos_emb.shape[0], -1)

        # class
        cls_names = [self.object_classes[i] for i in classes.flatten().long()]
        cls_emb_list = [self.obj_text_feat[name] for name in cls_names]
        cls_emb = torch.stack(cls_emb_list, dim=0)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)

        return emb

    def forward_scene(self, input_dict):

        if not self.prepare_called:
            self.prepare()

        bboxes = input_dict['fg_encoding_box'] # (B, 8)
        B = bboxes.shape[0]
        classes = input_dict['fg_class'] # (B)

        pos_emb = self.fourier_embedder(bboxes)
        # pos_emb = pos_emb.reshape(
        #     pos_emb.shape[0], -1)

        # class
        cls_names = [self.object_classes[i] for i in classes.flatten().long()]
        cls_emb_list = [self.obj_text_feat[name] for name in cls_names]
        cls_emb = torch.stack(cls_emb_list, dim=0)
        cls_emb = cls_emb.reshape(B, -1, cls_emb.shape[-1])
        # combine
        emb = self.forward_feature(pos_emb, cls_emb)

        return emb