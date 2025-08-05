import os
import argparse
import datetime
import yaml
import time
import shutil
from shutil import copyfile
import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
from lidargen.metrics.models.rangenet.pipe import Logger
from lidargen.metrics.datasets.nusc_seg_dataset import NuScenesDataset
from lidargen.metrics.models.rangenet.modules.segmentator import Segmentator
from lidargen.metrics.models.rangenet.modules.ioueval import iouEval
from lidargen.metrics.models.rangenet.sync_batchnorm.batchnorm import convert_model
from lidargen.metrics.models.rangenet.common.warmupLR import warmupLR
from lidargen.metrics.models.rangenet.common.avgmeter import AverageMeter
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

class Trainer():
    def __init__(self, ARCH, logdir, path=None):
        # parameters
        self.num_class = 17
        self.ARCH = ARCH
        self.log = logdir
        self.path = path

        # put logger where it belongs
        self.tb_logger = Logger(self.log + "/tb")
        self.info = {"train_update": 0,
                        "train_loss": 0,
                        "train_acc": 0,
                        "train_iou": 0,
                        "valid_loss": 0,
                        "valid_acc": 0,
                        "valid_iou": 0,
                        "backbone_lr": 0,
                        "decoder_lr": 0,
                        "head_lr": 0,
                        "post_lr": 0}
        self.build_dataset()
        self.build_model()


        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
            self.lidar_utils.cuda()
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)   # spread in gpus
                self.model = convert_model(self.model).cuda()  # sync batchnorm
                self.model_single = self.model.module  # single model to get weight names
                self.multi_gpu = True
                self.n_gpus = torch.cuda.device_count()

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.num_class, dtype=torch.float) + 1/(self.num_class -1)
        self.loss_w = 1 / (content + epsilon_w) 
        self.loss_w[0] = 0

        # loss
        if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
            self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        else:
            raise Exception('Loss not defined in config file')
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus

        # optimizer
        if self.ARCH["post"]["CRF"]["use"] and self.ARCH["post"]["CRF"]["train"]:
            self.lr_group_names = ["post_lr"]
            self.train_dicts = [{'params': self.model_single.CRF.parameters()}]
        else:
            self.lr_group_names = []
            self.train_dicts = []
        if self.ARCH["backbone"]["train"]:
            self.lr_group_names.append("backbone_lr")
            self.train_dicts.append(
                {'params': self.model_single.backbone.parameters()})
        if self.ARCH["decoder"]["train"]:
            self.lr_group_names.append("decoder_lr")
            self.train_dicts.append(
                {'params': self.model_single.decoder.parameters()})
        if self.ARCH["head"]["train"]:
            self.lr_group_names.append("head_lr")
            self.train_dicts.append({'params': self.model_single.head.parameters()})

        # Use SGD optimizer to train
        self.optimizer = optim.SGD(self.train_dicts,
                               lr=self.ARCH["train"]["lr"],
                               momentum=self.ARCH["train"]["momentum"],
                               weight_decay=self.ARCH["train"]["w_decay"])

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = len(self.trainloader)
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                lr=self.ARCH["train"]["lr"],
                                warmup_steps=up_steps,
                                momentum=self.ARCH["train"]["momentum"],
                                decay=final_decay)

    def build_dataset(self):
        train_dataset = NuScenesDataset(
            split='train',
        )
        self.trainloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=self.ARCH["train"]["batch_size"],
                                                    shuffle=True,
                                                    num_workers=self.ARCH["train"]["workers"],
                                                    pin_memory=True,
                                                    drop_last=True)

        val_dataset = NuScenesDataset(
            split='val',
        )
        self.validloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=self.ARCH["train"]["batch_size"],
                                                    shuffle=False,
                                                    num_workers=self.ARCH["train"]["workers"],
                                                    pin_memory=True,
                                                    drop_last=True)

    def build_model(self):
        with torch.no_grad():
            self.model = Segmentator(
                        self.ARCH,
                        self.num_class,
                        self.path)

            self.lidar_utils = LiDARUtility(
                resolution=(32,1024),
                depth_format="log_depth",
                min_depth=1.45,
                max_depth=80.0,
                ray_angles=get_linear_ray_angles(
                    H=32,
                    W=1024,
                    fov_up=10,
                    fov_down=-30
                ))
            self.lidar_utils.eval()


    def to_color(self, label):
        return self.trainloader.dataset.map(label)

    def get_xentropy_class_string(self, class_id):
        return self.trainloader.dataset.class_name_dict[class_id]

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self):
        # accuracy and IoU stuff
        best_train_iou = 0.0
        best_val_iou = 0.0
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.num_class,
                                self.device, self.ignore_class)

        # just for test
        # acc, iou, loss, rand_img = self.validate(val_loader=self.validloader,
        #                                         model=self.model,
        #                                         criterion=self.criterion,
        #                                         evaluator=self.evaluator,
        #                                         class_func=self.get_xentropy_class_string,
        #                                         color_fn=self.to_color,
        #                                         save_scans=self.ARCH["train"]["save_scans"])
        # # update info
        # self.info["valid_loss"] = loss
        # self.info["valid_acc"] = acc
        # self.info["valid_iou"] = iou

        # # remember best iou and save checkpoint
        # if iou > best_val_iou:
        #     print("Best mean iou in validation so far, save model!")
        #     print("*" * 80)
        #     best_val_iou = iou

        #     # save the weights!
        #     self.model_single.save_checkpoint(self.log, suffix="")

        # print("*" * 80)

        # save to log
        # Trainer.save_to_log(logdir=self.log,
        #                     logger=self.tb_logger,
        #                     info=self.info,
        #                     epoch=epoch,
        #                     w_summary=self.ARCH["train"]["save_summary"],
        #                     model=self.model_single,
        #                     img_summary=self.ARCH["train"]["save_scans"],
        #                     imgs=rand_img)

        for epoch in range(self.ARCH["train"]["max_epochs"]):
            # get info for learn rate currently
            groups = self.optimizer.param_groups
            for name, g in zip(self.lr_group_names, groups):
                self.info[name] = g['lr']
            acc, iou, loss, update_mean = self.train_epoch(train_loader=self.trainloader,
                                                            model=self.model,
                                                            criterion=self.criterion,
                                                            optimizer=self.optimizer,
                                                            epoch=epoch,
                                                            evaluator=self.evaluator,
                                                            scheduler=self.scheduler,
                                                            color_fn=self.to_color,
                                                            report=self.ARCH["train"]["report_batch"],
                                                            show_scans=self.ARCH["train"]["show_scans"])

            # update info
            self.info["train_update"] = update_mean
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            # remember best iou and save checkpoint
            if iou > best_train_iou:
                print("Best mean iou in training set so far, save model!")
                best_train_iou = iou
                self.model_single.save_checkpoint(self.log, suffix="_train")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img = self.validate(val_loader=self.validloader,
                                                        model=self.model,
                                                        criterion=self.criterion,
                                                        evaluator=self.evaluator,
                                                        class_func=self.get_xentropy_class_string,
                                                        color_fn=self.to_color,
                                                        save_scans=self.ARCH["train"]["save_scans"])
                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou

                # remember best iou and save checkpoint
                if iou > best_val_iou:
                    print("Best mean iou in validation so far, save model!")
                    print("*" * 80)
                    best_val_iou = iou

                    # save the weights!
                    self.model_single.save_checkpoint(self.log, suffix="")

                print("*" * 80)

                # save to log
                Trainer.save_to_log(logdir=self.log,
                                    logger=self.tb_logger,
                                    info=self.info,
                                    epoch=epoch,
                                    w_summary=self.ARCH["train"]["save_summary"],
                                    model=self.model_single,
                                    img_summary=self.ARCH["train"]["save_scans"],
                                    imgs=rand_img)

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10, show_scans=False):
        def prepare_batch(batch_dict):
            x = []
            x += [self.lidar_utils.convert_depth(batch_dict["depth"])]
            x += [batch_dict["xyz"]]
            x += [batch_dict["reflectance"]]
            x = torch.cat(x, dim=1)
            x = F.interpolate(
                x.to(self.device),
                size=(32, 1024),
                mode="nearest-exact",
            )
            return x, batch_dict['mask'], batch_dict['segment']
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        update_ratio_meter = AverageMeter()
        if self.gpu:
            torch.cuda.empty_cache()
        model.train()
        end = time.time()
        for i, batch_dict in enumerate(train_loader):
            data_time.update(time.time() - end)
            # prepare batch
            in_vol, proj_mask, proj_labels = prepare_batch(batch_dict)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(non_blocking=True).long()
            # compute output
            output = model(in_vol, proj_mask)
            loss = criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss.backward(idx)
            else:
                loss.backward()
            optimizer.step()
            loss = loss.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                # get the first scan in batch and project points
                mask_np = proj_mask[0].cpu().numpy()
                depth_np = in_vol[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)
                cv2.imshow("sample_training", out)
                cv2.waitKey(1)

            if i % self.ARCH["train"]["report_batch"] == 0:
                print('Lr: {lr:.3e} | '
                    'Update: {umean:.3e} mean,{ustd:.3e} std | '
                    'Epoch: [{0}][{1}/{2}] | '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                    'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                    'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std))
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        def prepare_batch(batch_dict):
            x = []
            x += [self.lidar_utils.convert_depth(batch_dict["depth"])]
            x += [batch_dict["xyz"]]
            x += [batch_dict["reflectance"]]
            x = torch.cat(x, dim=1)
            x = F.interpolate(
                x.to(self.device),
                size=(32, 1024),
                mode="nearest-exact",
            )
            return x, batch_dict['mask'], batch_dict['segment']
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, batch_dict in tqdm(enumerate(val_loader), 'Validation...',):
                in_vol, proj_mask, proj_labels = prepare_batch(batch_dict)
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                output = model(in_vol, proj_mask)
                loss = criterion(torch.log(output.clamp(min=1e-8)), proj_labels)

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].unsqueeze(0).cpu().numpy()
                    gt_np = proj_labels[0].unsqueeze(0).cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                                mask_np,
                                                pred_np,
                                                gt_np,
                                                color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            print('Validation set:\n'
                    'Time avg per batch {batch_time.avg:.3f}\n'
                    'Loss avg {loss.avg:.4f}\n'
                    'Acc avg {acc.avg:.3f}\n'
                    'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time,
                                                loss=losses,
                                                acc=acc, iou=iou))

            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                
        return acc.avg, iou.avg, losses.avg, rand_imgs


def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        default='../lidargen/metrics/models/rangenet/darknet53-1024px.yaml',
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default='./logs/' +
        datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default=None,
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )
    return parser.parse_args()

def main():
    FLAGS = parse_args()
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")

    trainer = Trainer(ARCH, FLAGS.log, FLAGS.pretrained)
    trainer.train()

if __name__ == "__main__":
    main()