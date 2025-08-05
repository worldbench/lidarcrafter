import pickle
import time

import numpy as np
import torch
import tqdm
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from ..loss_utils import rbbox_to_corners, rinter_area_compute

def iou3d(gboxes, qboxes):
    '''
        gboxes / qboxes: [N, 7], [x, y, z, w, l, h, ry] in velo coord.
        Notice: (x, y, z) is the real center of bbox.
    '''
    assert gboxes.shape[0] == qboxes.shape[0]
    indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) \
                & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
    index_loc = torch.nonzero(indicator)
    # todo: my addtion to avoid too large number after model initialization.
    gboxes = torch.clamp(gboxes, -200.0, 200.0)
    qboxes = torch.clamp(qboxes, -200.0, 200.0)
    odious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)
    if gboxes.shape[0] == 0 or qboxes.shape[0] == 0:
        return torch.unsqueeze(odious, 1)

    diff_angle = qboxes[:, -1] - gboxes[:, -1]
    angle_factor = 1.25 * (1.0 - torch.abs(torch.cos(diff_angle)))
    rbbox_to_corners_object = rbbox_to_corners()
    corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 1, 3, 4, 6]])
    corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 1, 3, 4, 6]])
    # corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
    # corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
    # corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)

    # compute the inter area
    # print("### compute the inter area")
    rinter_area_compute_object = rinter_area_compute()
    inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

    # compute center distance
    # print("### compute center distance")        

    # compute the mbr bev diag
    # print("### compute the mbr bev diag")        
    inter_h = (torch.min(gboxes[:, 2] + 0.5 * gboxes[:, 5], qboxes[:, 2] + 0.5 * qboxes[:, 5]) -
                torch.max(gboxes[:, 2] - 0.5 * gboxes[:, 5], qboxes[:, 2] - 0.5 * qboxes[:, 5]))
    # oniou_h = (torch.max(gboxes[:, 2] + 0.5 * gboxes[:, 5], qboxes[:, 2] + 0.5 * qboxes[:, 5]) -
    #            torch.min(gboxes[:, 2] - 0.5 * gboxes[:, 5], qboxes[:, 2] - 0.5 * qboxes[:, 5]))
    inter_h[inter_h < 0] = 0
    # mbr_diag_3d_square = mbr_diag_bev**2 + inter_h ** 2 + 1e-7

    volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
    volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
    inter_area_cuda = inter_area.to(torch.device(gboxes.device))
    volume_inc = inter_h.mul(inter_area_cuda)
    volume_union = (volume_gboxes + volume_qboxes - volume_inc)
    # center_dist_square_cuda = center_dist_square.to(torch.device(gboxes.device))
    # mbr_diag_3d_square_cuda = mbr_diag_3d_square.to(torch.device(gboxes.device))

    ious = torch.div(volume_inc, volume_union)
    return ious

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # metric = {
    #     'gt_num': 0,
    # }
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] = 0
    #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            box_pred = model(batch_dict)
        # import pdb;pdb.set_trace()
        box_pred = box_pred.cpu().numpy()

        single_result = {
            "box_pred": box_pred,
            "frame_id": batch_dict['frame_id'],
            "gt_id": batch_dict['gt_id'],
            "gt_boxes": batch_dict['gt_boxes'].cpu().numpy(),
            # "gt_boxes_ori": batch_dict['gt_boxes_ori'].cpu().numpy()
        }

        if 'waymo' in cfg.TAG:
            single_result['trans_angle'] = batch_dict['trans_angle'].cpu().numpy()

        det_annos += [single_result]
        if cfg.LOCAL_RANK == 0:
            # progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()


    # 目标: 分别计算IoU阈值在 0.3, 0.5, 0.7 下的Recall
    # 其他方案: 使用回归的metric，比如 mae

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        # metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    if 'waymo' in cfg.TAG:
        point_anchor_size = [4.7, 2.1, 1.7]
    else:
        point_anchor_size = [3.9, 1.6, 1.56]
    dxa, dya, dza = point_anchor_size
    diagonal = np.sqrt(dxa ** 2 + dya ** 2)

    # you can comment the overlap computation process when generating label uncertainty of waymo dataset for accelerating.
    overlap_dict = {}
    for anno in det_annos:
        box_pred = anno['box_pred']
        gt_boxes = anno['gt_boxes']
        frame_ids = anno['frame_id']
        gt_ids = anno['gt_id']

        box_pred[:, 0] *= diagonal
        box_pred[:, 1] *= diagonal
        box_pred[:, 2] *= dza
        box_pred[:, 3] = np.exp(box_pred[:, 3]) * dxa
        box_pred[:, 4] = np.exp(box_pred[:, 4]) * dya
        box_pred[:, 5] = np.exp(box_pred[:, 5]) * dza

        gt_boxes[:, 0] *= diagonal
        gt_boxes[:, 1] *= diagonal
        gt_boxes[:, 2] *= dza
        gt_boxes[:, 3] = np.exp(gt_boxes[:, 3]) * dxa
        gt_boxes[:, 4] = np.exp(gt_boxes[:, 4]) * dya
        gt_boxes[:, 5] = np.exp(gt_boxes[:, 5]) * dza

        box_pred = torch.tensor(box_pred)
        gt_boxes = torch.tensor(gt_boxes)
        box_pred = box_pred.cuda()
        gt_boxes = gt_boxes.cuda()
        overlap = iou3d(gt_boxes, box_pred).cpu().numpy().tolist()
        for i in range(len(frame_ids)):
            key = f'{frame_ids[i]}_{gt_ids[i]}'
            overlap_dict[key] = overlap[i]

    # to dump, key=[frame_id, gt_id, ]
    out_result = {}
    index = 0
    for anno in det_annos:
        frame_ids = anno['frame_id']
        gt_ids = anno['gt_id']

        box_pred = anno['box_pred'].tolist()
        gt_boxes = anno['gt_boxes'].tolist()
        # gt_boxes_ori = anno['gt_boxes_ori'].tolist()

        # rotation trans_angle
        if 'waymo' in cfg.TAG:
            trans_angle = anno['trans_angle'].tolist()

        for i in range(len(frame_ids)):
            key = f'{frame_ids[i]}_{gt_ids[i]}'
            box_pred_single = box_pred[i]
            gt_boxes_single = gt_boxes[i]

            if 'waymo' in cfg.TAG:
                trans_angle_single = trans_angle[i]

                if np.abs(trans_angle_single/(np.pi/2)%2-1) < 0.01:
                    t_x, t_y, t_z, t_l, t_w, t_h, t_theta, _, _  = box_pred_single
                    box_pred_single = [t_y, t_x, t_z, t_l, t_w, t_h, t_theta]
                    t_x, t_y, t_z, t_l, t_w, t_h, t_theta  = gt_boxes_single
                    gt_boxes_single = [t_y, t_x, t_z, t_l, t_w, t_h, t_theta]

            out_result[key] = {
                'pred_box': box_pred_single,
                'gt_box': gt_boxes_single,
                'overlap': overlap_dict[key]
            }

            index+=1
    
    dump_file = final_output_dir / 'result.pkl'
    if save_to_file:
        logger.info(f"### save file to {dump_file}")
        with open(dump_file, 'wb') as f:
            pickle.dump(out_result, f)
        # ad hoc
        # return {}

    # to evaluate

    # import pdb;pdb.set_trace()
    # hit_7 = sum(i>0.7 for i in overlap_list)
    # hit_5 = sum(i>0.5 for i in overlap_list)
    # recall_7 = hit_7 / len(overlap_list)
    # recall_5 = hit_5 / len(overlap_list)
    # logger.info(f'Recall_0.7 = {recall_7}, Recall_0.5 = {recall_5}')

    return {}
    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #     output_path=final_output_dir
    # )

    # logger.info(result_str)
    # ret_dict.update(result_dict)

    # logger.info('Result is save to %s' % result_dir)
    # logger.info('****************Evaluation done.*****************')
    # return ret_dict


if __name__ == '__main__':
    pass

