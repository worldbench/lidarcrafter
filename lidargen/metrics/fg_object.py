import os
import re
import shutil
import pickle
from pathlib import Path
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from .extractor.pointmlp import pointMLP
from .datasets.nuscenes_object_dataset import NuscObject
from .utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
# from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from lidargen.metrics.datasets.object_uncertainty_dataset import build_dataloader
from lidargen.metrics.models.glenet.model import Generator
from lidargen.metrics.models.glenet.eval_utils import eval_utils
from . import bev

def classification_parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', 
                        default='../pretrained_models/evaluation/nuscenes/pointmlp/nusc_object_classification_model.pth',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointMLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--method', default=None, help='model name [default: pointnet_cls]')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    return parser.parse_args()

def regression_parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../lidargen/metrics/models/glenet/exp20.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=32, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--method', type=str, default=None, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='../pretrained_models/evaluation/nuscenes/glenet/nusc_object_uncertainty_model.pth', help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def validate_classification(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    num_points_in_gt_list = []
    time_cost = datetime.datetime.now()
    class_names = ['unknow'] + testloader.dataset.class_name
    with torch.no_grad():
        for batch_idx, (data, label, num_points_in_gt) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            num_points_in_gt_list.append(num_points_in_gt.cpu().numpy())
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    num_points_in_gt = np.concatenate(num_points_in_gt_list)
    result_dict = {
        "class_names": class_names,
        "test_true": test_true,
        "test_pred": test_pred,
        "num_points_in_gt": num_points_in_gt,
    }

    return result_dict

def compute_classification_metrics_fixed_bins(result_dict, bins=None):

    if bins is None:
        bins = [0, 100, 200, 300, 400, 500, np.inf]
    labels = ["<100", "100-200", "200-300", "300-400", "400-500", ">500"]

    class_names = result_dict["class_names"]
    y_true = np.asarray(result_dict["test_true"])
    y_pred = np.asarray(result_dict["test_pred"])
    pts    = np.asarray(result_dict["num_points_in_gt"])

    overall_acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred,
        labels=range(len(class_names)),
        zero_division=0
    )
    df_overall = pd.DataFrame({
        "precision": p,
        "recall":    r,
        "f1-score":  f,
        "support":   s.astype(int)
    }, index=class_names)

    print(f"整体 Accuracy: {overall_acc:.4f}\n")
    print("整体各类指标：")
    print(df_overall.to_string(), "\n")

    bin_series = pd.cut(pts, bins=bins, labels=labels, include_lowest=True)

    partition_dict = {}
    for lbl in labels:
        mask = (bin_series == lbl)
        if not mask.any():
            # 该区间没有样本，跳过
            print(f"分区 {lbl}: 无样本，跳过\n")
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        acc_p = accuracy_score(yt, yp)
        p_p, r_p, f_p, s_p = precision_recall_fscore_support(
            yt, yp,
            labels=range(len(class_names)),
            zero_division=0
        )
        df_p = pd.DataFrame({
            "precision": p_p,
            "recall":    r_p,
            "f1-score":  f_p,
            "support":   s_p.astype(int)
        }, index=class_names)

        print(f"分区 {lbl}:")
        print(f"  Accuracy: {acc_p:.4f}")
        print(df_p.to_string(), "\n")

        partition_dict[lbl] = {
            "accuracy":  acc_p,
            "per_class": df_p
        }

    return {
        "overall": {
            "accuracy":  overall_acc,
            "per_class": df_overall
        },
        "partitions": partition_dict
    }

def compute_cgf(method_name):
    result_file_path = Path(f'../generated_results/{method_name}/inference_results') / 'classification_result.pkl'
    if not result_file_path.exists():
        args = classification_parse_args()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        # build dataset
        pkl_path = f'../generated_results/{method_name}/inference_results/selected_foreground_samples_info.pkl'
        test_loader = DataLoader(NuscObject(partition='val', pkl_path=pkl_path if method_name != 'ori' else None), num_workers=1,
                                batch_size=args.batch_size // 2, shuffle=False, drop_last=True)
        # build model
        net = pointMLP(num_classes=4)
        criterion = cal_loss
        net = net.to(device)
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        net.load_state_dict(checkpoint['net'])

        result_dict = validate_classification(net, test_loader, criterion, device)
        pickle.dump(result_dict, open(result_file_path, 'wb'))
    else:
        with open(result_file_path, 'rb') as f:
            result_dict = pickle.load(f)
    metrics_dict = compute_classification_metrics_fixed_bins(result_dict)

    out = {
        "overall": {
            "accuracy": metrics_dict["overall"]["accuracy"],
            "per_class": metrics_dict["overall"]["per_class"].to_dict(orient="index")
        },
        "partitions": {}
    }
    for part, vals in metrics_dict["partitions"].items():
        out["partitions"][part] = {
            "accuracy": vals["accuracy"],
            "per_class": vals["per_class"].to_dict(orient="index")
        }
    return out

def get_fg_object_set_feature(method_name):
    args = classification_parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # build dataset
    pkl_path = f'../generated_results/{method_name}/inference_results/selected_foreground_samples_info.pkl'
    test_loader = DataLoader(NuscObject(partition='val', pkl_path=pkl_path if method_name != 'ori' else None), num_workers=1,
                            batch_size=args.batch_size // 2, shuffle=False, drop_last=True)
    class_names = ['unkonw'] + test_loader.dataset.class_name
    fg_feat_set = {class_name: defaultdict(list) for class_name in class_names}
    # build model
    net = pointMLP(num_classes=4)
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])
    with torch.no_grad():
        for batch_idx, (data, label, num_points_in_gt) in tqdm(enumerate(test_loader), desc='Extracting features', total=len(test_loader)):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            global_feat = net(data, return_features=True)
            for obj_id, feat in enumerate(global_feat):
                class_name = class_names[label.detach().cpu().numpy()[obj_id]]
                fg_feat_set[class_name]['pts_feat'].append(feat.cpu())
                point_cloud = data[obj_id].permute(1,0) # N C
                hist = bev.point_cloud_to_histogram(point_cloud, min_depth=1e-6, max_depth=1e3, field_size=2.0)
                fg_feat_set[class_name]["bev_hists"].append(hist.cpu())

    for class_name, feat_dict in fg_feat_set.items():
        if class_name == 'unkonw':
            continue
        fg_feat_set[class_name]['pts_feat'] = torch.stack(feat_dict['pts_feat'], dim=0).numpy()
        fg_feat_set[class_name]['bev_hists'] = torch.stack(feat_dict['bev_hists'], dim=0).numpy()

    return fg_feat_set

def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

def compute_rgf_single(args, cfg):
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = Path(cfg.ROOT_DIR) / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'
    if args.method_name != 'ori':
        cfg.DATA_CONFIG.PKL_PATH = f'../generated_results/{args.method_name}/inference_results/foreground_samples_info.pkl'
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    input_channels = cfg.MODEL.get('INPUT_CHANNELS', 4)
    model = Generator(cfg.MODEL, input_channels=input_channels, scale=1)
    
    with torch.no_grad():
        eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)

def single_fold_data(method_name, fold_idx):
    print(f"############################# {fold_idx} ###############################")
    tag=f'fold_{fold_idx}'
    result_data_list = []
    save_path = f'../generated_results/{method_name}/inference_results/uncertainty_results/{tag}'
    file_list = [os.path.join(save_path, file) for file in os.listdir(save_path)]
    print(f"For fold={fold_idx}, len of file_list= ", len(file_list))
    for file in file_list:
        with open(file, 'rb') as f:
            result_data = pickle.load(f)
            result_data_list.append(result_data)
    key_l = []
    overlap_l = []
    variance_l = []
    pointnum_l = []

    splits=KFold(n_splits=10,shuffle=True,random_state=42)
    # ad hoc, you should change if you forbid enable_similar_type
    class_names = ['car', 'truck', 'bus']
    if method_name == 'ori':
        used_infos = pickle.load(open('../data/infos/nuscenes_object_classification_val.pkl', 'rb'))
        used_infos = [info for info in used_infos if info['name'] in class_names]
    else:
        pkl_path = f'../generated_results/{method_name}/inference_results/foreground_samples_info.pkl'
        used_infos = []
        with open(pkl_path, 'rb') as f:
            infos_dict = pickle.load(f)
        for key, value in infos_dict.items():
            if key in class_names:
                used_infos.extend(value)

    train_idx, val_idx = [x for x in splits.split(np.arange(len(used_infos)))][fold_idx]
    car_info = [used_infos[idx] for idx in val_idx]    

    for index in tqdm(range(len(car_info))):
        info = car_info[index]

        frame_id = val_idx[index]
        gt_idx = val_idx[index]
        key = f'{frame_id}_{gt_idx}'
        pc_data_num = info['num_points_in_gt']
        if key not in result_data_list[0]:
            print("unfound key", key)
            continue
        pred_box_list = [r[key]['pred_box'] for r in result_data_list]
        pred_box_overlap = [r[key]['overlap'] for r in result_data_list]
        pred_boxes = np.array(pred_box_list) # n * 9
        
        gt_box = result_data_list[0][key]['gt_box']
        gt_box_angle = gt_box[6]
        # import pdb;pdb.set_trace()
        pred_boxes[:, 6] = common_utils.limit_period(pred_boxes[:, 6] - gt_box_angle, 0, 2 * np.pi)

        # # according to the coordinates
        pred_boxes[:, 6] = np.sin(pred_boxes[:, 6])
        variance_list = np.var(pred_boxes[:, :7], axis=0)
        key_l.append(key)
        pointnum_l.append(pc_data_num)
        variance_l.append(variance_list)
        overlap_l.append(np.mean(pred_box_overlap))
    return key_l, pointnum_l, overlap_l, variance_l

def compute_regression_metrics_fixed_bins(result_json, bins=None):
    if bins is None:
        bins = [0, 150, 300, np.inf]
    labels = ["<150", "150-300", ">300"]

    # 2. 构建 DataFrame
    df = pd.DataFrame.from_dict(result_json, orient="index")
    # 确保列存在
    assert {"variance", "overlap", "pointnum"}.issubset(df.columns), \
        "每个子字典需包含 variance, overlap, pointnum"

    # 3. 整体均值
    overall_var     = df["variance"].mean()
    overall_overlap = df["overlap"].mean()
    print(f"整体 variance 均值: {overall_var}")
    print(f"整体 overlap 均值:  {overall_overlap:.4f}\n")

    # 4. 按固定区间分组
    df["bin"] = pd.cut(df["pointnum"], bins=bins, labels=labels, include_lowest=True)

    partitions = {}
    for lbl in labels:
        sub = df[df["bin"] == lbl]
        if sub.empty:
            print(f"分区 {lbl}: 无样本，已跳过\n")
            continue

        var_mean     = sub["variance"].mean()
        overlap_mean = sub["overlap"].mean()

        print(f"分区 {lbl} (样本数 {len(sub)}):")
        print(f"  variance 均值: {var_mean}")
        print(f"  overlap 均值:  {overlap_mean:.4f}\n")

        partitions[lbl] = {
            "variance": var_mean.tolist(),
            "overlap":  overlap_mean
        }

    # 5. 返回字典
    return {
        "overall": {
            "variance": overall_var.tolist(),
            "overlap":  overall_overlap
        },
        "partitions": partitions
    }

def compute_rgf(method_name):
    output_file = Path(f'../generated_results/{method_name}/inference_results') / 'un_v4.pkl'
    if output_file.exists():
        result_json = pickle.load(open(output_file, 'rb'))
    else:
        args, cfg = regression_parse_config()
        args.method_name = method_name
        for i in range(10):
            cfg.DATA_CONFIG.FOLD_IDX = i
            args.extra_tag = f'fold_{i}'
            save_path = f'../generated_results/{method_name}/inference_results/uncertainty_results/{args.extra_tag}'
            mkdir_p(save_path)

            for j in range(30):
                result_path = f'{save_path}/result_{j}.pkl'
                if os.path.exists(result_path):
                    print(f'File {result_path} already exists, skipping...')
                    continue
                compute_rgf_single(args, cfg)
                result_pkl_path = f'output/lidargen/metrics/models/glenet/exp20/{args.extra_tag}/eval/epoch_no_number/val/default/final_result/data/result.pkl'
                # cp a file to target path
                shutil.copy2(result_pkl_path, f'{save_path}/result_{j}.pkl')
        # calculate
        key_l_all = []
        pointnum_l_all = []
        overlap_l_all = []
        variance_l_all = []

        result_json = {}
        for fold_idx in range(10):
            key_l, pointnum_l, overlap_l, variance_l = single_fold_data(method_name, fold_idx)
            
            for i in range(len(key_l)):
                result_json[key_l[i]] = {'variance': variance_l[i], 
                                        'overlap': overlap_l[i],
                                        'pointnum': pointnum_l[i]}

            pointnum_l_all.extend(pointnum_l)
            variance_l_all.extend(variance_l)

    with open(output_file, 'wb') as f:
        pickle.dump(result_json, f)

    metrics_dict = compute_regression_metrics_fixed_bins(result_json)
    return metrics_dict

def compute_dcf(method_name):
    detection_results_pkl_path = f'../generated_results/{method_name}/inference_results/foreground_samples_info.pkl'

    assert os.path.exists(detection_results_pkl_path), \
        f'Detection results file {detection_results_pkl_path} does not exist.'
    with open(detection_results_pkl_path, 'rb') as f:
        detection_results_dict = pickle.load(f)

    flat_dets = []
    for class_name, objects_info in detection_results_dict.items():
        if class_name in ['car', 'truck', 'bus', 'pedestrian']:
            for det in objects_info:
                name = det['name']
                score = det['score']
                boxes = np.array(det['box3d_lidar'])
                flat_dets.append({'name': name, 'score': score, 'boxes_lidar': boxes})

    # 2. 计算每个类别的平均置信度
    class_scores = defaultdict(list)
    for det in flat_dets:
        class_scores[det['name']].append(det['score'])
    avg_scores_per_class = {cls: np.mean(scr_list) for cls, scr_list in class_scores.items()}

    metric_dict = {}
    print("Average confidence per class:")
    for cls, avg in avg_scores_per_class.items():
        print(f"{cls}: {avg:.6f}")
        # 4 floating point precision
        avg = round(avg, 4)
        metric_dict[cls] = float(avg)
    return metric_dict