import os
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network, load_data_to_gpu
from lidargen.metrics.datasets.generated_sample_dataset import GeneratedDataset
from lidargen.dataset.utils import get_points_in_box

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name', type=str, default='our', help='model name')
    parser.add_argument('--cfg_file', type=str, default='../pretrained_models/evaluation/nuscenes/voxelrcnn-center/configs/nuscenes_uda_voxel_rcnn_centerhead_10xyzt_allcls.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--pretrained_model', type=str, default='../pretrained_models/evaluation/nuscenes/voxelrcnn-center/nuscenes_uda_voxel_rcnn_centerhead_10xyzt_allcls.pth', help='checkpoint to start from')
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

class ForegroundSampleExtractor:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.logger = logger
        self.build_dataloader()
        self.build_model()

    def build_dataloader(self):

        dataset_cfg = self.cfg.DATA_CONFIG
        dataset_cfg.GENERATED_SAMPLES_PATH = os.path.join('../generated_results', self.args.model_name)
        if self.args.model_name == 'opendwm':
            dataset_cfg.GENERATED_SAMPLES_PATH = '../generated_results/opendwm/opendwm_lidar'
        elif self.args.model_name == 'opendwm_dit':
            dataset_cfg.GENERATED_SAMPLES_PATH = '../generated_results/opendwm_dit/opendwm_lidar_dit'
        else:
            pass

        if 'dwm' in self.args.model_name:
            dataset_cfg.ENDWITH = 'txt'
        if self.args.model_name == 'uniscene':
            dataset_cfg.ENDWITH = 'npy'

        dataset = GeneratedDataset(
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
        class_names = self.cfg.CLASS_NAMES
        final_output_dir = Path(os.path.join('../generated_results', self.args.model_name, 'inference_results'))
        final_output_dir.mkdir(parents=True, exist_ok=True)
        det_annos = []
        progress_bar = tqdm(total=len(self.dataloader), leave=True, desc='eval', dynamic_ncols=True)
        for i, batch_dict in enumerate(self.dataloader):
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = self.model(batch_dict)
            annos = self.dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=None
            )
            det_annos += annos
            progress_bar.update()

        with open(final_output_dir / 'result.pkl', 'wb') as f:        
            self.logger.info('Predictions saved to: %s' % str(final_output_dir / 'result.pkl'))
            pickle.dump(det_annos, f)

        progress_bar.close()

    def extract_foreground_samples(self):
        foreground_samples_info_dict = defaultdict(list)
        foreground_samples_statistics = defaultdict(int)
        foreground_samples_pred_score_statistics = defaultdict(list)
        self.logger.info('Starting foreground sample extraction...')
        result_pkl_filename = os.path.join('../generated_results', self.args.model_name, 'inference_results', 'result.pkl')
        if not os.path.exists(result_pkl_filename):
            self.inference()
        self.logger.info('Foreground sample extraction completed.')
        sample_save_dir = Path(os.path.join('../generated_results', self.args.model_name, 'inference_results', 'foreground_samples'))
        sample_save_dir.mkdir(parents=True, exist_ok=True)
        det_annos = pickle.load(open(result_pkl_filename, 'rb'))
        for frame_anno_dict in tqdm(det_annos, desc='Processing Annotations', dynamic_ncols=True):
            points = self.dataloader.dataset.__getitem__(int(frame_anno_dict['frame_id']))['points']
            if self.cfg.DATA_CONFIG.get('SHIFT_COOR', None):
                points[:, 0:3] -= np.array(self.cfg.DATA_CONFIG.SHIFT_COOR, dtype=np.float32)
            for sample_id in range(len(frame_anno_dict['name'])):
                sample_name = frame_anno_dict['name'][sample_id]
                boxes_3d = frame_anno_dict['boxes_lidar'][sample_id]
                sample_score = frame_anno_dict['score'][sample_id]

                sample_points, _ = get_points_in_box(points, boxes_3d)
                if sample_points.shape[0] < 50:
                    continue
                foreground_samples_statistics[sample_name] += 1
                foreground_samples_pred_score_statistics[sample_name].append(sample_score)
                sample_points[:,:3] -= boxes_3d[None, :3]
                file_save_path = sample_save_dir / f"{frame_anno_dict['frame_id']}_{sample_name}_{sample_id}.bin"
                sample_points.astype('float32').tofile(file_save_path)
                sample_dict = {
                    'name': sample_name,
                    'path': str(file_save_path),
                    'num_points_in_gt': sample_points.shape[0],
                    'box3d_lidar': boxes_3d.tolist(),
                    'score': sample_score
                }
                foreground_samples_info_dict[sample_name].append(sample_dict)
        # Save foreground samples info
        with open(sample_save_dir.parent / 'foreground_samples_info.pkl', 'wb') as f:
            self.logger.info('Foreground samples info saved to: %s' % str(sample_save_dir / 'foreground_samples_info.pkl'))
            pickle.dump(foreground_samples_info_dict, f)

        # report foreground sample statistics
        self.logger.info('Foreground Sample Statistics:')
        for sample_name, count in foreground_samples_statistics.items():
            avg_score = sum(foreground_samples_pred_score_statistics[sample_name]) / len(foreground_samples_pred_score_statistics[sample_name])
            self.logger.info(f'Sample: {sample_name}, Count: {count}, Avg Score: {avg_score:.4f}')

if __name__ == "__main__":
    args, cfg = parse_config()
    extractor = ForegroundSampleExtractor(args, cfg)
    extractor.extract_foreground_samples()