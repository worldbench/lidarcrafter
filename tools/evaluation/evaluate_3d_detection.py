import os
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pickle
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network, load_data_to_gpu
from lidargen.metrics.datasets.object_detection_dataset import ObjectDetectionDataset
from lidargen.dataset.utils import get_points_in_box
from lidargen.metrics.utils.pcdet_eval_utils import eval_one_epoch

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name', type=str, default='our', help='model name')
    # parser.add_argument('--cfg_file', type=str, default='../pretrained_models/evaluation/nuscenes/IASSD/configs/ia_ssd_detection.yaml', help='specify the config for training')
    # parser.add_argument('--pretrained_model', type=str, default='../pretrained_models/evaluation/nuscenes/IASSD/nuscenes_uda_IA-SSD_10xyzt_allcls_98304.pth', help='checkpoint to start from')
    parser.add_argument('--cfg_file', type=str, default='../pretrained_models/evaluation/nuscenes/voxelrcnn-center/configs/voxel_rcnn_detection.yaml', help='specify the config for training')
    parser.add_argument('--pretrained_model', type=str, default='../pretrained_models/evaluation/nuscenes/voxelrcnn-center/nuscenes_uda_voxel_rcnn_centerhead_10xyzt_allcls.pth', help='checkpoint to start from')
    parser.add_argument('--batch_size', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

class TD_OD_Inferencer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.logger = logger
        self.build_dataloader()
        self.build_model()

    def build_dataloader(self):

        dataset_cfg = self.cfg.DATA_CONFIG
        dataset_cfg.METHOD_NAME = self.args.model_name
        dataset = ObjectDetectionDataset(
            dataset_cfg=dataset_cfg,
            class_names=self.cfg.CLASS_NAMES,
            training=False,
            root_path=None,
            logger=self.logger
        )
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, batch_size=self.args.batch_size, pin_memory=True, num_workers=self.args.workers,
            shuffle=False, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=None, timeout=0
        )

    def build_model(self):
        model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.dataset)
        model.load_params_from_file(filename=self.args.pretrained_model, logger=self.logger, to_cpu=False)
        model.cuda()
        self.model = model

    def inference(self):
        self.model.eval()
        eval_one_epoch(
            cfg, self.model, self.dataloader, 0, logger, dist_test=False,
            result_dir=Path(f'../generated_results/{self.args.model_name}/inference_results'), save_to_file=args.save_to_file
        )

if __name__ == "__main__":
    args, cfg = parse_config()
    extractor = TD_OD_Inferencer(args, cfg) # 3D object inferencer
    extractor.inference()