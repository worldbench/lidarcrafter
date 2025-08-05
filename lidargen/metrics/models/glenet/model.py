import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
# from ResNet import B2_ResNet
# from utils import init_weights,init_weights_orthogonal_normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.nn import Parameter, Softmax
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from .point_net import PointNetfeat, SimPointNetfeat

from pcdet.utils import loss_utils
from pcdet.utils import common_utils

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg



class Encoder_x(nn.Module):
    def __init__(self, input_channels, x=1, latent_size=3):
        super(Encoder_x, self).__init__()
        self.input_channels = input_channels
        self.fe = PointNetfeat(input_channels, x)
        fe_out_channels = 512 * x
        self.fc1 = nn.Linear(fe_out_channels, latent_size)
        self.fc2 = nn.Linear(fe_out_channels, latent_size)


    def forward(self, input):
        output = self.fe(input)
        # print(output.size())

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)+3e-22), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Encoder_xy(nn.Module):
    def __init__(self, input_channels, x=1, latent_size=3):
        super(Encoder_xy, self).__init__()
        self.input_channels = input_channels
        self.fe = PointNetfeat(input_channels, x)
        fe_out_channels = 512 * x
        self.fc1 = nn.Linear(fe_out_channels + 8, latent_size)
        self.fc2 = nn.Linear(fe_out_channels + 8, latent_size)


    def forward(self, input, y):
        output = self.fe(input)
        # print(output.size())

        # todo: add y as input, y should be processed
        output = torch.cat([output, y], axis=1)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)+3e-22), 1)
        # print("### in posterior, dist variance", dist.variance.min(), dist.variance.max(), torch.exp(logvar).min(), torch.exp(logvar).max())
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Object_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, input_channels, x=1, latent_dim=3, num_bins=2):
        super(Object_feat_encoder, self).__init__()

        x = 0.5
        self.fe = SimPointNetfeat(input_channels, x)
        fe_out_channels = int(16 * x)

        # fc_scale = 0.0625 # 16
        fc_scale = 0.25

        self.fc1 = nn.Linear(fe_out_channels + latent_dim, int(256 * fc_scale))
        self.fc2 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))

        # self.pre_bn = nn.BatchNorm1d(input_channels)
        self.bn1 = nn.BatchNorm1d(int(256 * fc_scale))
        self.bn2 = nn.BatchNorm1d(int(256 * fc_scale))
        # NOTE: should there put a BN?
        self.fc_s1 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))
        self.fc_s2 = nn.Linear(int(256 * fc_scale), 3, bias=False)

        # self.fc_c1 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))
        # CLS_NUM=1 # ad hoc
        # self.fc_c2 = nn.Linear(int(256 * fc_scale), CLS_NUM, bias=False)

        self.fc_ce1 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))
        self.fc_ce2 = nn.Linear(int(256 * fc_scale), 3, bias=False)

        self.fc_hr1 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))
        self.fc_hr2 = nn.Linear(int(256 * fc_scale), 1, bias=False)

        self.fc_dir1 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))
        self.fc_dir2 = nn.Linear(int(256 * fc_scale), num_bins, bias=False)

    def forward(self, x_dict, z):
   
        x = self.fe(x_dict)

        x = torch.cat([x, z], axis=1)


        x = F.relu(self.bn1(self.fc1(x)))
        feat = F.relu(self.bn2(self.fc2(x)))

        # x = F.relu(self.fc_c1(feat))
        # logits = self.fc_c2(x)

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        x = F.relu(self.fc_dir1(feat))
        dirs = self.fc_dir2(x)

        out = torch.cat([centers, sizes, headings, dirs], axis=1) # dim=3+3+1+2
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, model_cfg, input_channels, scale):
        super(Generator, self).__init__()
        self.model_cfg = model_cfg
        self.latent_dim = model_cfg.LATENT_DIM

        # structure
        # self.relu = nn.ReLU(inplace=True)

        self.obj_encoder = Object_feat_encoder(input_channels, scale, latent_dim=self.latent_dim)
        self.xy_encoder = Encoder_xy(input_channels, scale, self.latent_dim)
        self.x_encoder = Encoder_x(input_channels, scale, self.latent_dim) # input_dim, channel, latent_dim

        # build loss
        losses_cfg = model_cfg.LOSS_CONFIG
        # self.add_module(
        #     'cls_loss_func',
        #     loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        # )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.last_batch_dict = None
        self.last_tb_dict = None
    def update_global_step(self):
        self.global_step += 1

    # def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
    #     return block(dilation_series, padding_series, NoLabels, input_channel)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, batch_dict):
        x = batch_dict['points']
        text_feat = batch_dict['text_feat']
        x_dict = dict(
            x = x,
            text_feat = text_feat
        )
        y = batch_dict['gt_boxes_input']
        labels = batch_dict['gt_boxes']

        if self.training:
            try:
                self.posterior, muxy, logvarxy = self.xy_encoder(x_dict, y)
            except:
                import pdb;pdb.set_trace()
            self.prior, mux, logvarx = self.x_encoder(x_dict)
            # print("###self.prior min,max", self.prior.variance.min(), self.prior.variance.max())
            # print("###self.posterior min,max", self.posterior.variance.min(), self.posterior.variance.max())
            lattent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))

            if torch.isnan(lattent_loss) or torch.isinf(lattent_loss):
                import pdb;pdb.set_trace()

            normal_dist = Independent(Normal(loc=torch.zeros_like(mux), scale=torch.ones_like(logvarx)), 1)
            # dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

            # 即跟标准正态分布之间的距离
            normal_latent_loss = torch.mean(self.kl_divergence(self.posterior, normal_dist))
            # lattent_loss += normal_latent_loss

            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)

            self.box_pred_post  = self.obj_encoder(x_dict, z_noise_post)
            # self.box_pred_prior = self.obj_encoder(x, z_noise_prior)

            # get_loss
            lattent_loss = lattent_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['latent_weight']

            loss_tuple, tb_dict, disp_dict = self.get_training_loss(self.box_pred_post, labels, lattent_loss)

            # ret_dict = {
            #     'loss': loss
            # }
            self.last_batch_dict = batch_dict
            self.last_tb_dict = tb_dict
            # print("###tb_dict=", tb_dict, "lattent_loss=", lattent_loss)
            # print([self.last_batch_dict['points'][x].sum()==0 for x in range(64)])
            return loss_tuple, tb_dict, disp_dict

        else:
            text_feat = batch_dict['text_feat']
            x_dict = dict(
                x = x,
                text_feat = text_feat
            )
            _, mux, logvarx = self.x_encoder(x_dict)
            # print(f"########### mean = {mux}, logvarx={logvarx}")
            z_noise = self.reparametrize(mux, logvarx)
            self.box_pred  = self.obj_encoder(x_dict, z_noise)

            # post_processing
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            # dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
            #     else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_cls_preds =  self.box_pred[:, -2:]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                self.box_pred[..., 6] - dir_offset, dir_limit_offset, period
            )
            self.box_pred[..., 6] = dir_rot + dir_offset + period * dir_labels.to(self.box_pred.dtype)

            return self.box_pred

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        # anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6]
        
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=reg_targets.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def reg_loss(self, box_preds, labels):
        batch_size = int(box_preds.shape[0])
        # import pdb;pdb.set_trace()
        # box_preds = box_preds.view(batch_size, -1,
        #                            box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
        #                            box_preds.shape[-1])

        # no dir
        box_preds_loc = box_preds[:, :7]

        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds_loc, labels)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=None)
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'loss_loc': loc_loss.item()
        }

        # import pdb;pdb.set_trace()
        dir_targets = self.get_direction_target(
            labels,
            dir_offset=self.model_cfg.DIR_OFFSET,
            num_bins=self.model_cfg.NUM_DIR_BINS
        )

        # get_direction_target
        batch_size = labels.shape[0]

        # attention, fixed order
        # ad hoc, change shape to (N, anchor_num=1, 2)
        dir_logits = box_preds[:, -2:].view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
        dir_targets = dir_targets.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
        # import pdb;pdb.set_trace()

        weights = torch.ones_like(dir_logits)
        dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
        dir_loss = dir_loss.sum() / batch_size
        dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
        box_loss += dir_loss
        tb_dict['loss_dir'] = dir_loss.item()
        tb_dict['loss_reg'] = box_loss.item()
        # print(f'dir_loss={dir_loss}, box_loss={box_loss}')
        return box_loss, tb_dict

    def get_training_loss(self, box_pred_post, labels, lattent_loss):
        disp_dict = {}

        # loss = latent_loss + prior_reg_loss + post_reg_loss
        ############################## loss computation start ##############################
        # import pdb;pdb.set_trace()

        reg_loss_post, tb_dict_post = self.reg_loss(box_pred_post, labels)

        # loss = reg_loss_post + lattent_loss
        regular_loss = l2_regularisation(self.xy_encoder) + \
                l2_regularisation(self.x_encoder) + l2_regularisation(self.obj_encoder)
        # ad hoc
        regular_loss = 1e-4 * regular_loss

        ############################## loss computation end ##############################

        tb_dict = {
            # 'lattent_loss': lattent_loss.item(),
        }


        # for k, v in tb_dict_prior.items():
        #     tb_dict[k+'_prior'] = v
        for k, v in tb_dict_post.items():
            tb_dict[k+'_post'] = v

        return (reg_loss_post, lattent_loss, regular_loss), tb_dict, disp_dict


    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch


if __name__ == '__main__':
    # main()
    
    model = Generator(4, 1, 3)
    print(model)