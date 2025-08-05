import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from rich import print
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import pickle
import sys
sys.path.append("./")
sys.path.append("../")
from lidargen.utils import inference
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
from lidargen.dataset.custom_dataset import CustomDataset
import vis_tools.utils.pipe_related as pipe

warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False


def sample(args):
    cfg = __all__[args.cfg]()
    cfg.resume = args.ckpt
    if args.batch_size is not None:
        cfg.training.batch_size_train = args.batch_size
    else:
        args.batch_size = cfg.training.batch_size_train
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        # dynamo_backend=cfg.training.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(cfg)

    ddpm, model, lidar_utils, _, _, _ = inference.load_model_duffusion_training(cfg)

    auto_cfg_name = 'nuscenes-auto-reg-v2'
    cfg.model.auto_cfg_name = auto_cfg_name
    auto_cfg = __all__[auto_cfg_name]()
    auto_cfg.resume = '../pretrained_models/nusc-auto-reg-v2-350000.pth'
    auto_ddpm, _, _, _, _, _ = inference.load_model_duffusion_training(auto_cfg)
    ddpm.eval()
    auto_ddpm.eval()
    lidar_utils.eval()
    ddpm.to(device)
    auto_ddpm.to(device)
    lidar_utils.to(device)

    cfg.data.split = 'all'
    dataset = all_datasets[cfg.data.dataset](cfg.data)

    # custom token
    custom_token_dict = pickle.load(open('../data/infos/needed_5_framed_token.pkl', 'rb'))
    dataset.update_data_with_custom_tokens(custom_token_dict)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        drop_last=False,
        collate_fn=dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    )

    lidar_utils, dataloader = accelerator.prepare(lidar_utils, dataloader)
    sample_fn = torch.compile(ddpm.sample) # TODO: add this when torch compile is stable
    temporal_sample_fn = torch.compile(auto_ddpm.sample) # TODO: add this when torch compile is stable
    # sample_fn = ddpm.sample
    # temporal_sample_fn = auto_ddpm.sample
    save_dir = Path(args.output_dir)
    with accelerator.main_process_first():
        save_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(batch):
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.data.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x
    
    def preprocess_prev_cond(prev_cond):
        x = []
        reflectance = prev_cond[:,3,...]/255
        depth = prev_cond[:,-2,...]

        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(depth.unsqueeze(1))]
        if cfg.data.train_reflectance:
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        prev_labels = prev_cond[:,4,...].long()
        one_hot = F.one_hot(prev_labels, num_classes=len(list(cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x = torch.cat((x, one_hot.float()), dim=1)
        return x

    def preprocess_condition_mask(batch):
        x = []
        condition_mask = batch['condition_mask'] # [B, 2, H, W]: semantic and depth
        # semantic
        curr_labels = condition_mask[:,0,...].long()
        one_hot = F.one_hot(curr_labels, num_classes=len(list(cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x+= [one_hot.float()]
        # depth
        depth = lidar_utils.convert_depth(condition_mask[:,1,...].unsqueeze(1))
        x+= [depth]
        x = torch.cat(x, dim=1)
        return x

    def preprocess_autoregressive_cond(autoregressive_cond):
        x = []
        depth = autoregressive_cond[:, 0]
        reflectance = autoregressive_cond[:, 1]

        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(depth.unsqueeze(1))]
        if cfg.data.train_reflectance and cfg.model.auto_cfg_name != 'nuscenes-auto-reg-v2':
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x

    def prepare_batch(batch_dict):
        for key in batch_dict:
            if isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to('cuda')

        # if self.cfg. TODO: weather to use the collate_fn
        # x_0 = self.preprocess(batch_dict)
        # batch_dict['x_0'] = x_0
        if 'prev_cond' in batch_dict:
            prev_cond = preprocess_prev_cond(batch_dict['prev_cond'])
            batch_dict['cond'] = prev_cond

        if 'condition_mask' in batch_dict:
            condition_mask = preprocess_condition_mask(batch_dict)
            batch_dict['concat_cond'] = condition_mask

        if 'autoregressive_cond' in batch_dict:
            autoregressive_cond = preprocess_autoregressive_cond(batch_dict['autoregressive_cond'])
            batch_dict['autoregressive_cond'] = autoregressive_cond

        return batch_dict

    def temporal_sample(temporal_data_dict_list, num_steps=256):
        temp_dataset = CustomDataset(temporal_data_dict_list)
        setattr(temp_dataset, 'task', 'autoregressive_generation')
        batch = [temp_dataset.__getitem__(i) for i in range(args.batch_size)]
        batch = temp_dataset.collate_fn(batch)
        batch_dict = prepare_batch(batch)
        xs = temporal_sample_fn(
            batch_dict=batch_dict,
            batch_size=args.batch_size,
            num_steps=num_steps,
            mode="ddpm",
            return_all=False,
        ).clamp(-1, 1)
        # xs = batch_dict['autoregressive_cond']
        return xs

    def postprocess(sample):
        sample = lidar_utils.denormalize(sample)
        depth, rflct = sample[:, [0]], sample[:, [1]]
        depth = lidar_utils.revert_depth(depth)
        xyz = lidar_utils.to_xyz(depth)
        return torch.cat([depth, xyz, rflct], dim=1)

    global_step = 0
    for batch in tqdm(
        dataloader,
        desc="saving...",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ):
        custom_tokens = batch.pop('custom_tokens', None)
        global_seed = global_step * args.batch_size
        seeds = [global_seed + i for i in range(args.batch_size)]

        x_0 = preprocess(batch)
        batch['x_0'] = x_0
        if 'prev_cond' in batch:
            prev_cond = preprocess_prev_cond(batch['prev_cond'])
            batch['cond'] = prev_cond
        if 'condition_mask' in batch:
            condition_mask = preprocess_condition_mask(batch)
            batch['concat_cond'] = condition_mask

        with torch.cuda.amp.autocast(cfg.training.mixed_precision is not None):
            samples = sample_fn(
                batch_dict=batch,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                mode=args.mode,
                # rng=inference.setup_rng(seeds, device=device),
                progress=True,
            ).clamp(-1, 1)

        curr_background_points_list = []
        samples = postprocess(samples)
        for batch_id in range(samples.shape[0]):
            token = custom_tokens[batch_id][0]
            points = samples[batch_id, [1,2,3,4]].reshape(4,-1).permute(1,0).detach().cpu().numpy()
            curr_background_points_list.append(points)
            save_frame_path = os.path.join(save_dir, token)
            os.makedirs(save_frame_path, exist_ok=True)
            save_file_name = os.path.join(
                save_frame_path, f"0_{token}.txt"
            )
            np.savetxt(
                save_file_name,
                points[:,:3],
                fmt='%.6f',
            )

        # temporal generation
        custom_data_dict_list = []
        fut_background_points_list = []
        fut_boxes_3d_list = []
        Ts_list = []
        align_obj_points_list = []
        align_obj_intensity_list = []

        for batch_id in range(samples.shape[0]):

            ##################
            agent_fut_trajs = batch['gt_fut_trajs'][batch_id]
            agent_fut_trajs = np.insert(agent_fut_trajs, 0, 0, axis=1)
            acc_agent_fut_trajs = np.cumsum(agent_fut_trajs, axis=1)  # cumulative sum to get future trajectories
            acc_agent_fut_trajs = pipe.interp_trajs_numpy(acc_agent_fut_trajs, M=args.traj_lenth)
            agent_fut_trajs = acc_agent_fut_trajs[:,1:] - acc_agent_fut_trajs[:,:-1]  # get future trajectories

            custom_data_dict = dict(
                gt_fut_trajs = agent_fut_trajs,
                xyz = samples[batch_id][[1,2,3]].detach().cpu().numpy(),
                reflectance = samples[batch_id][[4]].detach().cpu().numpy(),
                gt_boxes = batch['gt_boxes'][batch_id],
                gt_names = batch['gt_names'][batch_id],
                condition_mask = batch['condition_mask'][batch_id].detach().cpu().numpy(),
            )
            custom_data_dict_list.append(custom_data_dict)
            _, fut_background_points, _, fut_boxes_3d, Ts, align_obj_points, align_obj_intensity = pipe.get_temporal_boxes_3d(custom_data_dict)
            # curr_background_points_list.append(curr_background_points)
            # curr_background_points_list.append(fut_background_points[0])  # only use the first frame background points
            fut_background_points_list.append(fut_background_points)
            fut_boxes_3d_list.append(fut_boxes_3d)
            Ts_list.append(Ts)
            align_obj_points_list.append(align_obj_points)
            align_obj_intensity_list.append(align_obj_intensity)

        for t_id in tqdm(range(15)):
            temporal_data_dict_list = []
            # conduct cond data dict
            for batch_id in range(samples.shape[0]):
                next_frame_points = pipe.get_next_frame_points(curr_background_points_list[batch_id], align_obj_points_list[batch_id], 
                                                               align_obj_intensity_list[batch_id], fut_boxes_3d_list[batch_id][:, t_id], 
                                                               custom_data_dict_list[batch_id]['gt_names'],Ts_list[batch_id][t_id])
                
                # token = custom_tokens[batch_id][t_id+1]
                # save_file_name = os.path.join(
                #     save_dir, custom_tokens[batch_id][0], f"{t_id+1}_{token}.txt"
                # )
                # ego_box = np.zeros((1, 7), dtype=np.float32)
                # gt_boxes = np.concatenate([ego_box, fut_boxes_3d_list[batch_id][:, t_id]], axis=0)
                # curr_background_points_list[batch_id] = pipe.delete_fg_points(next_frame_points, gt_boxes[1:])
                # np.savetxt(
                #     save_file_name,
                #     next_frame_points[:,:3],
                #     fmt='%.6f',
                # )
                
                ego_box = np.zeros((1, 7), dtype=np.float32)
                gt_boxes = np.concatenate([ego_box, fut_boxes_3d_list[batch_id][:, t_id]], axis=0)
                temporal_data_dict = {
                    'points': next_frame_points,
                    'gt_boxes': gt_boxes,
                    'gt_names': custom_data_dict_list[batch_id]['gt_names']
                }
                temporal_data_dict_list.append(temporal_data_dict)

            temp_sample = temporal_sample(temporal_data_dict_list, num_steps=args.num_steps)
            temp_sample = postprocess(temp_sample)
            for batch_id in range(temp_sample.shape[0]):
                try:
                    token = custom_tokens[batch_id][t_id+1]
                except:
                    token = batch_id
                points = temp_sample[batch_id, [1,2,3,4]].reshape(4,-1).permute(1,0).detach().cpu().numpy()
                combined_points = np.concatenate(
                    [fut_background_points_list[batch_id][t_id], points], axis=0
                )
                curr_background_points_list[batch_id] = pipe.delete_fg_points(combined_points, temporal_data_dict_list[batch_id]['gt_boxes'][1:, :7])
                save_file_name = os.path.join(
                    save_dir, custom_tokens[batch_id][0], f"{t_id+1}_{token}.txt"
                )
                np.savetxt(
                    save_file_name,
                    points[:,:3],
                    fmt='%.6f',
                )

        global_step += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default='nuscenes-box-layout-v2')
    parser.add_argument("--ckpt", type=str, default='../pretrained_models/nuscenes-lox-layout-v4-500000.pth')
    parser.add_argument("--output_dir", default='../generated_results/our/temporal_points', type=str)
    parser.add_argument(
        "--traj_lenth", type=int, default=16, help="custom trajectory length for temporal generation"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2,
        help="batch size for training",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="worker number for dataset loading",
    )
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--mode", choices=["ddpm", "ddim"], default="ddpm")
    args = parser.parse_args()

    sample(args)
