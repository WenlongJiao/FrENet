import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import glob
import kornia
import cv2
import lpips

from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from tqdm.auto import tqdm
from .data_process import augment_images
from .calculate_util import calculate_psnr, calculate_ssim, compute_psnr, compute_ssim
from utils.util import *
from utils.model_util import *
from utils.loss_function import *
from utils.picture_util import *

class EarlyStopping:
    def __init__(self, patience, delta=0, min_delta=0, mode='min', verbose=True):
        self.patience = patience  # 提前停止的耐心
        self.delta = delta  # 损失函数的最小变化
        self.min_delta = min_delta  # 最小的变化值
        self.mode = mode  # 'min' 或 'max'，根据目标调整
        self.verbose = verbose  # 是否打印日志
        self.best_score = None  # 最好的得分
        self.early_stop = False  # 是否触发提前停止
        self.counter = 0  # 连续没有改善的epoch计数
        self.best_epoch = 0  # 最好的epoch
        self.best_loss = None  # 最好的损失

    def __call__(self, epoch, current_score):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_loss = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.delta:
                self.best_score = current_score
                self.best_epoch = epoch
                self.best_loss = current_score
                self.counter = 0
            elif current_score >= self.best_score:  # 如果分数没有增加，重置计数器
                self.counter = 0
            else:  # 当前分数减少但幅度在delta内
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}.")
        elif self.mode == 'max':
            if current_score > self.best_score + self.delta:
                self.best_score = current_score
                self.best_epoch = epoch
                self.best_loss = current_score
                self.counter = 0
            elif current_score <= self.best_score:  # 如果分数没有增加，重置计数器
                self.counter = 0
            else:  # 当前分数增加但幅度在delta内
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}.")


class TrainerBase(nn.Module):
    def __init__(self,
                 epoches,
                 train_loader,
                 val_loader,
                 device,
                 logger=None,
                 writer=None,
                 **kwargs,

                 ) -> None:
        super(TrainerBase, self).__init__()

        self.epoches = epoches
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.writer = writer
        self.start_epoch = 0

    def load_latest_checkpoint(self, model, optimizer, scheduler, checkpoint_filename):
        # 获取目录路径和文件名基名（不含 _epoch 部分）
        dir_path = os.path.dirname(checkpoint_filename)
        base_name = os.path.basename(checkpoint_filename).split('.')[0]  # 提取文件名前缀（去掉 _epoch 部分）

        # 获取所有符合条件的检查点文件，并按 epoch 排序（降序）
        checkpoint_files = sorted(
            glob.glob(os.path.join(dir_path, f"{base_name}_epoch*.pth")),
            key=lambda x: int(x.split('_epoch')[-1].split('.')[0]),  # 提取 epoch 排序
            reverse=True  # 降序排列，最新的文件在前
        )

        # 如果找到检查点文件，加载最新的检查点
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[0]
            self.logger.info(f"Loading latest checkpoint from '{latest_checkpoint}'")
            model, optimizer, scheduler, self.start_epoch, _ = load_checkpoint(
                model, optimizer, scheduler, latest_checkpoint, self.device, self.logger
            )
        else:
            self.start_epoch = 0  # 没有找到检查点，从头开始训练

        return model, optimizer, scheduler


class Trainer(TrainerBase):
    def __init__(self,
                 model,
                 epoches,
                 eta_min,
                 train_loader,
                 val_loader,
                 device,
                 checkpoint_filename='checkpoint.pth',
                 if_augmentation=True,
                 logger=None,
                 writer=None,
                 ) -> None:
        super(Trainer, self).__init__(epoches, train_loader, val_loader, device, logger, writer)

        self.model = model
        self.eta_min = eta_min
        self.checkpoint_filename = checkpoint_filename
        self.if_augmentation = if_augmentation
        self.start_epoch = 0
        self.early_stopping = EarlyStopping(patience=10, delta=0.01, mode='max')

    def evaluate_model(self, model, test_loader):
        model.eval()
        self.model.module.grid = True

        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for batch_idx, (blur_images, sharp_images) in enumerate(test_loader):
                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                blur_images = Packing(blur_images)
                outputs = model(blur_images)
                outputs = Unpacking(outputs)
                for i in range(outputs.size(0)):
                    output_tensor = outputs[i]
                    sharp_gt_tensor = sharp_images[i]

                    try:
                        psnr_value = calculate_psnr(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)

                        if np.isfinite(psnr_value):
                            psnr_list.append(psnr_value)
                    except Exception as e:
                        print(f"Error calculating PSNR for batch {batch_idx}, img {i}: {e}. Skipping.")

                    try:
                        ssim_value = calculate_ssim(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)

                        if np.isfinite(ssim_value):
                            ssim_list.append(ssim_value)

                    except Exception as e:
                        print(f"Error calculating SSIM for batch {batch_idx}, img {i}: {e}. Skipping.")
                avg_psnr = 0.0
                avg_ssim = 0.0

                if psnr_list:
                    avg_psnr = sum(psnr_list) / len(psnr_list)
                else:
                    print("Warning: No valid PSNR values collected.")

                if ssim_list:
                    avg_ssim = sum(ssim_list) / len(ssim_list)
                else:
                    print("Warning: No valid SSIM values collected.")

                if self.logger:
                    self.logger.info(f"--- Evaluation Results (Total Images: {len(psnr_list)} PSNR, {len(ssim_list)} SSIM) ---")
                    self.logger.info(f"Average PSNR: {avg_psnr:.8f}, Average SSIM: {avg_ssim:.8f}")
                else:
                    print(f"--- Evaluation Results (Total Images: {len(psnr_list)} PSNR, {len(ssim_list)} SSIM) ---")
                    print(f"Average PSNR: {avg_psnr:.8f}, Average SSIM: {avg_ssim:.8f}")

                return avg_psnr, avg_ssim

    def evaluate_model_final(self, model, test_loader, calculate_lpips=False):
        model.eval()
        self.model.module.grid = True

        local_psnr_sum = 0.0
        local_ssim_sum = 0.0
        local_lpips_sum = 0.0 if calculate_lpips else 0.0

        local_psnr_valid_count = 0
        local_ssim_valid_count = 0
        local_lpips_valid_count = 0 if calculate_lpips else 0

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        is_main_process = (rank == 0)

        total_batches = len(test_loader)
        if is_main_process:
            print(f"Starting final evaluation...")
            print(f"Total batches in test_loader: {total_batches}")

        lpips_model = None
        if calculate_lpips:
            try:
                lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Could not initialize LPIPS model. LPIPS calculation will be skipped. Error: {e}")
                    calculate_lpips = False  # Disable LPIPS if initialization fail
                if dist.is_initialized():
                    lpips_status = torch.tensor([int(calculate_lpips)], device=self.device, dtype=torch.int)
                    dist.all_reduce(lpips_status, op=dist.ReduceOp.MIN)  # If any rank fails, status becomes 0
                    calculate_lpips = bool(lpips_status.item())
                    if not calculate_lpips and is_main_process:
                        print("LPIPS calculation disabled on all ranks due to initialization failure on one rank.")

        with torch.no_grad():
            # Use tqdm only on the main process
            if is_main_process:
                iterator = tqdm(test_loader, total=total_batches, desc=f"Evaluating",
                                leave=False)
            else:
                iterator = test_loader  # No tqdm for other ranks

            for batch_idx, (blur_images, sharp_images) in enumerate(iterator):
                if batch_idx % world_size != rank:
                    continue

                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                blur_images = Packing(blur_images)
                outputs = model(blur_images)
                outputs = Unpacking(outputs)
                sharp_images = sharp_images.to(outputs.device)

                for i in range(outputs.size(0)):
                    output_tensor = outputs[i]
                    sharp_gt_tensor = sharp_images[i]

                    psnr_value = float('nan')
                    ssim_value = float('nan')
                    lpips_value = float('nan') if calculate_lpips else float('nan')

                    try:
                        psnr_value = calculate_psnr(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)
                        ssim_value = calculate_ssim(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)

                        if calculate_lpips and lpips_model is not None:
                            output_tensor_lpips = output_tensor.unsqueeze(0)  # Add batch dim
                            sharp_gt_tensor_lpips = sharp_gt_tensor.unsqueeze(0)  # Add batch dim
                            output_tensor_lpips = output_tensor_lpips * 2 - 1
                            sharp_gt_tensor_lpips = sharp_gt_tensor_lpips * 2 - 1
                            output_tensor_lpips = output_tensor_lpips.float()
                            sharp_gt_tensor_lpips = sharp_gt_tensor_lpips.float()
                            lpips_value_tensor = lpips_model(output_tensor_lpips, sharp_gt_tensor_lpips)
                            lpips_value = lpips_value_tensor.item()

                    except Exception as e:
                        # Log error but don't fail the whole evaluation
                        if dist.is_initialized():
                            print(f"Rank {rank}: Error processing image batch {batch_idx}, image {i} (Metrics): {e}")
                        else:
                            print(f"Error processing image batch {batch_idx}, image {i} (Metrics): {e}")

                    if np.isfinite(psnr_value):
                        local_psnr_sum += psnr_value
                        local_psnr_valid_count += 1

                    if np.isfinite(ssim_value):
                        local_ssim_sum += ssim_value
                        local_ssim_valid_count += 1
                    else:
                        if is_main_process:
                            print(
                                f"Warning: SSIM is non-finite for image {i} in batch {batch_idx} (Rank {rank}). Skipping for sum.")

                    if calculate_lpips and np.isfinite(lpips_value):
                        local_lpips_sum += lpips_value
                        local_lpips_valid_count += 1
                    elif calculate_lpips:
                        if is_main_process:
                            print(
                                f"Warning: LPIPS is non-finite for image {i} in batch {batch_idx} (Rank {rank}). Skipping for sum.")

        if calculate_lpips:
            local_agg_data = torch.tensor([local_psnr_sum, local_ssim_sum, local_lpips_sum, float(local_psnr_valid_count), float(local_ssim_valid_count), float(local_lpips_valid_count)], device=self.device, dtype=torch.float64)
        else:
            local_agg_data = torch.tensor([local_psnr_sum, local_ssim_sum, float(local_psnr_valid_count), float(local_ssim_valid_count)], device=self.device, dtype=torch.float64)

        if dist.is_initialized():
            dist.all_reduce(local_agg_data, op=dist.ReduceOp.SUM)

        global_psnr_sum = local_agg_data[0].item()
        global_ssim_sum = local_agg_data[1].item()

        if calculate_lpips:
            global_lpips_sum = local_agg_data[2].item()
            global_psnr_valid_count = int(local_agg_data[3].item())
            global_ssim_valid_count = int(local_agg_data[4].item())
            global_lpips_valid_count = int(local_agg_data[5].item())
        else:
            global_lpips_sum = 0.0  # Explicitly zero if not calculated
            global_psnr_valid_count = int(local_agg_data[2].item())
            global_ssim_valid_count = int(local_agg_data[3].item())
            global_lpips_valid_count = 0  # Explicitly zero if not calculated

        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0

        if global_psnr_valid_count > 0:
            avg_psnr = global_psnr_sum / global_psnr_valid_count

        if global_ssim_valid_count > 0:
            avg_ssim = global_ssim_sum / global_ssim_valid_count

        if calculate_lpips and global_lpips_valid_count > 0:
            avg_lpips = global_lpips_sum / global_lpips_valid_count

        if is_main_process:
            if self.logger:
                self.logger.info(f"--- Evaluation Results (Total Images Evaluated: ---")
                self.logger.info(f"Overall Average PSNR: {avg_psnr:.8f}, Overall Average SSIM: {avg_ssim:.8f}")
                if calculate_lpips:
                    self.logger.info(f"Overall Average LPIPS: {avg_lpips:.8f}")
            else:
                # Fallback print if logger not available
                print(f"--- Evaluation Results (Total Images Evaluated: ---")
                print(f"Overall Average PSNR: {avg_psnr:.8f}, Overall Average SSIM: {avg_ssim:.8f}")
                if calculate_lpips:
                    print(f"Overall Average LPIPS: {avg_lpips:.8f}")

        if calculate_lpips:
            return avg_psnr, avg_ssim, avg_lpips
        else:
            return avg_psnr, avg_ssim


    def lr_lambda(self, scheduler_type):
        # 根据 scheduler_type 定义学习率调度策略
        if scheduler_type == 'linear':
            return lambda epoch: 1.0 - max(0, epoch - 500) / 500.0
        else:  # 默认情况下使用 constant scheduler
            return lambda epoch: 1.0

    @timeit
    def train_model(self, args):
        # 在这里初始化损失函数
        if args.loss_function == 'mse':
            self.criterion = MSELoss()
        elif args.loss_function == 'mae':
            self.criterion = L1Loss()
        elif args.loss_function == 'gmse':
            self.criterion = GradientMSELoss()
        elif args.loss_function == 'sml':
            self.criterion = SSIMMSELoss()
        elif args.loss_function == 'fr':
            self.criterion = FRLoss()
        elif args.loss_function == 'freq':
            self.criterion = FreqLoss()

        # 定义优化器
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)

        # 定义学习率调度器
        if args.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epoches, eta_min=self.eta_min
            )
        else:
            lr_lambda = self.lr_lambda(args.scheduler_type)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.model, self.optimizer, self.scheduler = self.load_latest_checkpoint(self.model,
                                                                                 self.optimizer,
                                                                                 self.scheduler,
                                                                                 self.checkpoint_filename)
        self.model.to(self.device)
        self.start_epoch += 1

        # set up progress bar
        progress_bar = tqdm(total=len(self.train_loader) * self.epoches, desc='Training',
                            initial=(self.start_epoch-1) * len(self.train_loader))

        for epoch in range(self.start_epoch, self.epoches + 1):
            self.model.train()
            self.model.module.grid = False

            if self.logger:  # 记录每个 epoch 的开始
                self.logger.info(f"Epoch {epoch}/{self.epoches} started.")

            epoch_loss = 0.0
            for batch_idx, (blur_images, sharp_images) in enumerate(self.train_loader):
                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                blur_images = Packing(blur_images)
                sharp_images = Packing(sharp_images)

                # 数据增强
                if self.if_augmentation:
                    augment_images(blur_images, sharp_images)

                self.optimizer.zero_grad()
                #outputs = self.model(blur_images, training=True)
                outputs = self.model(blur_images)
                sharp_images = sharp_images.to(outputs.device)

                # 计算损失
                loss = self.criterion(outputs, sharp_images)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 更新进度条
                progress_bar.update(1)
                if batch_idx % 50 == 0:  # 每10个批次更新一次
                    progress_bar.set_postfix({'Epoch': epoch, 'Loss': loss.item()})

                # 记录每个批次的损失
                if self.logger and batch_idx % 50 == 0:  # 每10个批次记录一次
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item():.8f}")

                # tensorboard
                if self.writer and batch_idx % 50 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

                epoch_loss += loss.item()  # 累加损失

            # 记录每个 epoch 完成的平均损失
            if self.logger:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.8f}")

            # tensorboard
            if self.writer:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, epoch)

            # 学习率调度器步进
            self.scheduler.step()

            # 只在主进程保存检查点
            if dist.get_rank() == 0:
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss.item(), self.checkpoint_filename)

            # 每隔一定的 epoch 记录一次学习率
            if self.logger and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    self.logger.info(f"Learning rate at epoch {epoch + 1}: {param_group['lr']:.8f}")

            # tensorboard
            if self.writer and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    self.writer.add_scalar('train/lr', param_group['lr'], epoch)

            if epoch >= (self.epoches-500) and epoch % 50 == 0:
                avg_psnr, avg_ssim = self.evaluate_model(self.model, self.val_loader)
                if self.logger:
                    self.logger.info(f"Epoch {epoch + 1} psnr: {avg_psnr:.8f}, ssim {avg_ssim:.8f} on val data.")

                self.early_stopping(epoch, avg_psnr)

                if self.early_stopping.early_stop:
                    if self.logger:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    if dist.get_rank() == 0:
                        save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss.item(), self.checkpoint_filename)
                    break

        progress_bar.close()

class GoProTrainer(TrainerBase):
    def __init__(self,
                 model,
                 epoches,
                 eta_min,
                 train_loader,
                 val_loader,
                 device,
                 checkpoint_filename='checkpoint.pth',
                 if_augmentation=True,
                 logger=None,
                 writer=None,
                 ) -> None:
        super(GoProTrainer, self).__init__(epoches, train_loader, val_loader, device, logger, writer)

        self.model = model
        self.eta_min = eta_min
        self.checkpoint_filename = checkpoint_filename
        self.if_augmentation = if_augmentation
        self.start_epoch = 0
        self.early_stopping = EarlyStopping(patience=10, delta=0.01, mode='max')

    def evaluate_model(self, model, test_loader):
        model.eval()
        self.model.module.grid = True

        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for batch_idx, (blur_images, sharp_images) in enumerate(test_loader):
                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                outputs = model(blur_images)

                for i in range(outputs.size(0)):
                    output_tensor = outputs[i]
                    sharp_gt_tensor = sharp_images[i]

                    try:
                        psnr_value = calculate_psnr(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)

                        if np.isfinite(psnr_value):
                            psnr_list.append(psnr_value)
                    except Exception as e:
                        print(f"Error calculating PSNR for batch {batch_idx}, img {i}: {e}. Skipping.")

                    try:
                        ssim_value = calculate_ssim(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)

                        if np.isfinite(ssim_value):
                            ssim_list.append(ssim_value)

                    except Exception as e:
                        print(f"Error calculating SSIM for batch {batch_idx}, img {i}: {e}. Skipping.")
                avg_psnr = 0.0
                avg_ssim = 0.0

                if psnr_list:
                    avg_psnr = sum(psnr_list) / len(psnr_list)
                else:
                    print("Warning: No valid PSNR values collected.")

                if ssim_list:
                    avg_ssim = sum(ssim_list) / len(ssim_list)
                else:
                    print("Warning: No valid SSIM values collected.")

                if self.logger:
                    self.logger.info(f"--- Evaluation Results (Total Images: {len(psnr_list)} PSNR, {len(ssim_list)} SSIM) ---")
                    self.logger.info(f"Average PSNR: {avg_psnr:.8f}, Average SSIM: {avg_ssim:.8f}")
                else:
                    print(f"--- Evaluation Results (Total Images: {len(psnr_list)} PSNR, {len(ssim_list)} SSIM) ---")
                    print(f"Average PSNR: {avg_psnr:.8f}, Average SSIM: {avg_ssim:.8f}")

                return avg_psnr, avg_ssim

    def evaluate_model_final(self, model, test_loader, calculate_lpips=False):
        model.eval()
        self.model.module.grid = True

        local_psnr_sum = 0.0
        local_ssim_sum = 0.0
        local_lpips_sum = 0.0 if calculate_lpips else 0.0

        local_psnr_valid_count = 0
        local_ssim_valid_count = 0
        local_lpips_valid_count = 0 if calculate_lpips else 0

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        is_main_process = (rank == 0)

        total_batches = len(test_loader)
        if is_main_process:
            print(f"Starting final evaluation...")
            print(f"Total batches in test_loader: {total_batches}")

        lpips_model = None
        if calculate_lpips:
            try:
                lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Could not initialize LPIPS model. LPIPS calculation will be skipped. Error: {e}")
                    calculate_lpips = False  # Disable LPIPS if initialization fail
                if dist.is_initialized():
                    lpips_status = torch.tensor([int(calculate_lpips)], device=self.device, dtype=torch.int)
                    dist.all_reduce(lpips_status, op=dist.ReduceOp.MIN)  # If any rank fails, status becomes 0
                    calculate_lpips = bool(lpips_status.item())
                    if not calculate_lpips and is_main_process:
                        print("LPIPS calculation disabled on all ranks due to initialization failure on one rank.")

        with torch.no_grad():
            # Use tqdm only on the main process
            if is_main_process:
                iterator = tqdm(test_loader, total=total_batches, desc=f"Evaluating",
                                leave=False)
            else:
                iterator = test_loader  # No tqdm for other ranks

            for batch_idx, (blur_images, sharp_images) in enumerate(iterator):
                if batch_idx % world_size != rank:
                    continue

                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                b, c, h, w = blur_images.shape
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                if h_n > 0 or w_n > 0:  # 只在需要填充时执行
                    blur_images_padded = F.pad(blur_images, (0, w_n, 0, h_n), mode='reflect')
                else:
                    blur_images_padded = blur_images
                outputs = model(blur_images_padded)
                outputs = outputs[:, :, :h, :w]
                #outputs = model(blur_images)
                sharp_images = sharp_images.to(outputs.device)

                for i in range(outputs.size(0)):
                    output_tensor = outputs[i]
                    sharp_gt_tensor = sharp_images[i]

                    psnr_value = float('nan')
                    ssim_value = float('nan')
                    lpips_value = float('nan') if calculate_lpips else float('nan')

                    try:
                        psnr_value = calculate_psnr(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)
                        ssim_value = calculate_ssim(output_tensor, sharp_gt_tensor, crop_border=0, input_order='CHW', test_y_channel=False)

                        if calculate_lpips and lpips_model is not None:
                            output_tensor_lpips = output_tensor.unsqueeze(0)  # Add batch dim
                            sharp_gt_tensor_lpips = sharp_gt_tensor.unsqueeze(0)  # Add batch dim
                            output_tensor_lpips = output_tensor_lpips * 2 - 1
                            sharp_gt_tensor_lpips = sharp_gt_tensor_lpips * 2 - 1
                            output_tensor_lpips = output_tensor_lpips.float()
                            sharp_gt_tensor_lpips = sharp_gt_tensor_lpips.float()
                            lpips_value_tensor = lpips_model(output_tensor_lpips, sharp_gt_tensor_lpips)
                            lpips_value = lpips_value_tensor.item()

                    except Exception as e:
                        # Log error but don't fail the whole evaluation
                        if dist.is_initialized():
                            print(f"Rank {rank}: Error processing image batch {batch_idx}, image {i} (Metrics): {e}")
                        else:
                            print(f"Error processing image batch {batch_idx}, image {i} (Metrics): {e}")

                    if np.isfinite(psnr_value):
                        local_psnr_sum += psnr_value
                        local_psnr_valid_count += 1

                    if np.isfinite(ssim_value):
                        local_ssim_sum += ssim_value
                        local_ssim_valid_count += 1
                    else:
                        if is_main_process:
                            print(
                                f"Warning: SSIM is non-finite for image {i} in batch {batch_idx} (Rank {rank}). Skipping for sum.")

                    if calculate_lpips and np.isfinite(lpips_value):
                        local_lpips_sum += lpips_value
                        local_lpips_valid_count += 1
                    elif calculate_lpips:
                        if is_main_process:
                            print(
                                f"Warning: LPIPS is non-finite for image {i} in batch {batch_idx} (Rank {rank}). Skipping for sum.")

        if calculate_lpips:
            local_agg_data = torch.tensor([local_psnr_sum, local_ssim_sum, local_lpips_sum, float(local_psnr_valid_count), float(local_ssim_valid_count), float(local_lpips_valid_count)], device=self.device, dtype=torch.float64)
        else:
            local_agg_data = torch.tensor([local_psnr_sum, local_ssim_sum, float(local_psnr_valid_count), float(local_ssim_valid_count)], device=self.device, dtype=torch.float64)

        if dist.is_initialized():
            dist.all_reduce(local_agg_data, op=dist.ReduceOp.SUM)

        global_psnr_sum = local_agg_data[0].item()
        global_ssim_sum = local_agg_data[1].item()

        if calculate_lpips:
            global_lpips_sum = local_agg_data[2].item()
            global_psnr_valid_count = int(local_agg_data[3].item())
            global_ssim_valid_count = int(local_agg_data[4].item())
            global_lpips_valid_count = int(local_agg_data[5].item())
        else:
            global_lpips_sum = 0.0  # Explicitly zero if not calculated
            global_psnr_valid_count = int(local_agg_data[2].item())
            global_ssim_valid_count = int(local_agg_data[3].item())
            global_lpips_valid_count = 0  # Explicitly zero if not calculated

        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0

        if global_psnr_valid_count > 0:
            avg_psnr = global_psnr_sum / global_psnr_valid_count

        if global_ssim_valid_count > 0:
            avg_ssim = global_ssim_sum / global_ssim_valid_count

        if calculate_lpips and global_lpips_valid_count > 0:
            avg_lpips = global_lpips_sum / global_lpips_valid_count

        if is_main_process:
            if self.logger:
                self.logger.info(f"--- Evaluation Results ---")
                self.logger.info(f"Overall Average PSNR: {avg_psnr:.8f}, Overall Average SSIM: {avg_ssim:.8f}")
                if calculate_lpips:
                    self.logger.info(f"Overall Average LPIPS: {avg_lpips:.8f}")
            else:
                # Fallback print if logger not available
                print(f"--- Evaluation Results (Total Images Evaluated:  ---")
                print(f"Overall Average PSNR: {avg_psnr:.8f}, Overall Average SSIM: {avg_ssim:.8f}")
                if calculate_lpips:
                    print(f"Overall Average LPIPS: {avg_lpips:.8f}")

        if calculate_lpips:
            return avg_psnr, avg_ssim, avg_lpips
        else:
            return avg_psnr, avg_ssim

    def lr_lambda(self, scheduler_type):
        # 根据 scheduler_type 定义学习率调度策略
        if scheduler_type == 'linear':
            return lambda epoch: 1.0 - max(0, epoch - 500) / 500.0
        else:  # 默认情况下使用 constant scheduler
            return lambda epoch: 1.0

    @timeit
    def train_model(self, args):
        # 在这里初始化损失函数
        if args.loss_function == 'mse':
            self.criterion = MSELoss()
        elif args.loss_function == 'mae':
            self.criterion = L1Loss()
        elif args.loss_function == 'gmse':
            self.criterion = GradientMSELoss()
        elif args.loss_function == 'sml':
            self.criterion = SSIMMSELoss()
        elif args.loss_function == 'fr':
            self.criterion = FRLoss()
        elif args.loss_function == 'freq':
            self.criterion = FreqLoss()

        # 定义优化器
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)

        # 定义学习率调度器
        if args.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epoches, eta_min=self.eta_min
            )
        elif args.scheduler_type == 'fixed_then_cosine':
            fixed_epochs = 65
            initial_lr = args.lr
            scheduler_fixed = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoches - fixed_epochs,
                eta_min=self.eta_min
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[scheduler_fixed, scheduler_cosine],
                milestones=[fixed_epochs]
            )
        else:
            lr_lambda = self.lr_lambda(args.scheduler_type)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.model, self.optimizer, self.scheduler = self.load_latest_checkpoint(self.model,
                                                                                 self.optimizer,
                                                                                 self.scheduler,
                                                                                 self.checkpoint_filename)
        self.model.to(self.device)
        self.start_epoch += 1

        # set up progress bar
        progress_bar = tqdm(total=len(self.train_loader) * self.epoches, desc='Training',
                            initial=(self.start_epoch-1) * len(self.train_loader))

        for epoch in range(self.start_epoch, self.epoches + 1):
            self.model.train()
            self.model.module.grid = False

            if self.logger:  # 记录每个 epoch 的开始
                self.logger.info(f"Epoch {epoch}/{self.epoches} started.")

            epoch_loss = 0.0
            for batch_idx, (blur_images, sharp_images) in enumerate(self.train_loader):
                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)

                # 数据增强
                if self.if_augmentation:
                    augment_images(blur_images, sharp_images)

                self.optimizer.zero_grad()
                outputs = self.model(blur_images)
                sharp_images = sharp_images.to(outputs.device)

                # 计算损失
                loss = self.criterion(outputs, sharp_images)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 更新进度条
                progress_bar.update(1)
                if batch_idx % 50 == 0:  # 每10个批次更新一次
                    progress_bar.set_postfix({'Epoch': epoch, 'Loss': loss.item()})

                # 记录每个批次的损失
                if self.logger and batch_idx % 50 == 0:  # 每10个批次记录一次
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item():.8f}")

                # tensorboard
                if self.writer and batch_idx % 50 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

                epoch_loss += loss.item()  # 累加损失

            # 记录每个 epoch 完成的平均损失
            if self.logger:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.8f}")

            # tensorboard
            if self.writer:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, epoch)

            # 学习率调度器步进
            self.scheduler.step()

            # 只在主进程保存检查点
            if dist.get_rank() == 0:
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss.item(), self.checkpoint_filename)

            # 每隔一定的 epoch 记录一次学习率
            if self.logger and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    self.logger.info(f"Learning rate at epoch {epoch + 1}: {param_group['lr']:.8f}")

            # tensorboard
            if self.writer and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    self.writer.add_scalar('train/lr', param_group['lr'], epoch)

            if epoch >= 50 and epoch % 50 == 0:
                avg_psnr, avg_ssim = self.evaluate_model(self.model, self.val_loader)
                if self.logger:
                    self.logger.info(f"Epoch {epoch + 1} psnr: {avg_psnr:.8f}, ssim {avg_ssim:.8f} on val data.")

                self.early_stopping(epoch, avg_psnr)

                if self.early_stopping.early_stop:
                    if self.logger:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    if dist.get_rank() == 0:
                        save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss.item(), self.checkpoint_filename)
                    break

        progress_bar.close()

class RealBlurTrainer(TrainerBase):
    def __init__(self,
                 model,
                 epoches,
                 eta_min,
                 train_loader,
                 val_loader,
                 device,
                 checkpoint_filename='checkpoint.pth',
                 if_augmentation=True,
                 logger=None,
                 writer=None,
                 ) -> None:
        super(RealBlurTrainer, self).__init__(epoches, train_loader, val_loader, device, logger, writer)

        self.model = model
        self.eta_min = eta_min
        self.checkpoint_filename = checkpoint_filename
        self.if_augmentation = if_augmentation
        self.start_epoch = 0
        self.early_stopping = EarlyStopping(patience=10, delta=0.01, mode='max')

    def evaluate_model(self, model, test_loader):
        model.eval()
        self.model.module.grid = True

        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for batch_idx, (blur_images, sharp_images) in enumerate(test_loader):
                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                outputs = model(blur_images)
                for i in range(outputs.size(0)):
                    output_img_tensor = outputs[i].cpu()
                    sharp_gt_img_tensor = sharp_images[i].cpu()
                    output_np = output_img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
                    sharp_gt_np = sharp_gt_img_tensor.permute(1, 2, 0).numpy().astype(np.float32)

                    try:
                        aligned_output_np, aligned_sharp_np, mask_np, shift = image_align(output_np, sharp_gt_np)
                        psnr_value = compute_psnr(aligned_sharp_np, aligned_output_np, mask_np, data_range=1)
                        ssim_value = compute_ssim(aligned_sharp_np, aligned_output_np, mask_np)

                        if not np.isnan(psnr_value):
                            psnr_list.append(psnr_value)
                        if not np.isnan(ssim_value):
                            ssim_list.append(ssim_value)

                    except cv2.error as e:
                        # ECC 对齐有时可能失败，特别是当图像差异很大时
                        print(f"Alignment failed for image in batch {batch_idx}, index {i}: {e}")

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)

        # 记录评估结果到日志
        if self.logger:
            self.logger.info(f"Average PSNR: {avg_psnr:.8f}, Average SSIM: {avg_ssim:.8f}")

        return avg_psnr, avg_ssim

    def evaluate_model_final(self, model, test_loader, calculate_lpips=False):
        model.eval()
        self.model.module.grid = True

        local_psnr_sum = 0.0
        local_ssim_sum = 0.0
        local_lpips_sum = 0.0 if calculate_lpips else 0.0

        local_psnr_valid_count = 0
        local_ssim_valid_count = 0
        local_lpips_valid_count = 0 if calculate_lpips else 0

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        is_main_process = (rank == 0)

        total_batches = len(test_loader)
        if is_main_process:
            print(f"Starting final evaluation...")
            print(f"Total batches in test_loader: {total_batches}")

        lpips_model = None
        if calculate_lpips:
            try:
                lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Could not initialize LPIPS model. LPIPS calculation will be skipped. Error: {e}")
                    calculate_lpips = False  # Disable LPIPS if initialization fail
                if dist.is_initialized():
                    lpips_status = torch.tensor([int(calculate_lpips)], device=self.device, dtype=torch.int)
                    dist.all_reduce(lpips_status, op=dist.ReduceOp.MIN)  # If any rank fails, status becomes 0
                    calculate_lpips = bool(lpips_status.item())
                    if not calculate_lpips and is_main_process:
                        print("LPIPS calculation disabled on all ranks due to initialization failure on one rank.")

        with torch.no_grad():
            # Use tqdm only on the main process
            if is_main_process:
                iterator = tqdm(test_loader, total=total_batches, desc=f"Evaluating",
                                leave=False)
            else:
                iterator = test_loader  # No tqdm for other ranks

            for batch_idx, (blur_images, sharp_images) in enumerate(iterator):
                if batch_idx % world_size != rank:
                    continue

                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)
                outputs = model(blur_images)
                sharp_images = sharp_images.to(outputs.device)

                for i in range(outputs.size(0)):
                    output_img_tensor = outputs[i]
                    sharp_gt_img_tensor = sharp_images[i]

                    output_np = output_img_tensor.cpu().permute(1, 2, 0).numpy().astype(np.float32)
                    sharp_gt_np = sharp_gt_img_tensor.cpu().permute(1, 2, 0).numpy().astype(np.float32)

                    psnr_value = float('nan')
                    ssim_value = float('nan')
                    lpips_value = float('nan') if calculate_lpips else float('nan')

                    try:
                        aligned_output_np, aligned_sharp_np, mask_np, shift = image_align(output_np, sharp_gt_np)
                        psnr_value = compute_psnr(aligned_sharp_np, aligned_output_np, mask_np, data_range=1)
                        ssim_value = compute_ssim(aligned_sharp_np, aligned_output_np, mask_np)

                        if calculate_lpips and lpips_model is not None:
                            aligned_output_tensor_lpips = torch.from_numpy(
                                aligned_output_np.transpose(2, 0, 1).copy()).to(self.device).float()
                            aligned_sharp_tensor_lpips = torch.from_numpy(
                                aligned_sharp_np.transpose(2, 0, 1).copy()).to(self.device).float()
                            output_tensor_lpips = aligned_output_tensor_lpips.unsqueeze(0) * 2 - 1
                            sharp_gt_tensor_lpips = aligned_sharp_tensor_lpips.unsqueeze(0) * 2 - 1
                            lpips_value_tensor = lpips_model(output_tensor_lpips, sharp_gt_tensor_lpips)
                            lpips_value = lpips_value_tensor.item()

                    except cv2.error as e:
                        if dist.is_initialized():
                            print(f"Rank {rank}: Alignment failed for image in batch {batch_idx}, index {i}: {e}")
                        else:
                            print(f"Alignment failed for image in batch {batch_idx}, index {i}: {e}")

                    except Exception as e:
                        if dist.is_initialized():
                            print(f"Rank {rank}: Error calculating metrics for image batch {batch_idx}, image {i}: {e}")
                        else:
                            print(f"Error calculating metrics for image batch {batch_idx}, image {i}: {e}")

                    if np.isfinite(psnr_value):
                        local_psnr_sum += psnr_value
                        local_psnr_valid_count += 1

                    if np.isfinite(ssim_value):
                        local_ssim_sum += ssim_value
                        local_ssim_valid_count += 1
                    else:
                        if is_main_process:
                            print(
                                f"Warning: SSIM is non-finite for image {i} in batch {batch_idx} (Rank {rank}). Skipping for sum.")

                    if calculate_lpips and np.isfinite(lpips_value):
                        local_lpips_sum += lpips_value
                        local_lpips_valid_count += 1
                    elif calculate_lpips:
                        if is_main_process:
                            print(
                                f"Warning: LPIPS is non-finite for image {i} in batch {batch_idx} (Rank {rank}). Skipping for sum.")

        if calculate_lpips:
            local_agg_data = torch.tensor([local_psnr_sum, local_ssim_sum, local_lpips_sum, float(local_psnr_valid_count), float(local_ssim_valid_count), float(local_lpips_valid_count)], device=self.device, dtype=torch.float64)
        else:
            local_agg_data = torch.tensor([local_psnr_sum, local_ssim_sum, float(local_psnr_valid_count), float(local_ssim_valid_count)], device=self.device, dtype=torch.float64)

        if dist.is_initialized():
            dist.all_reduce(local_agg_data, op=dist.ReduceOp.SUM)

        global_psnr_sum = local_agg_data[0].item()
        global_ssim_sum = local_agg_data[1].item()

        if calculate_lpips:
            global_lpips_sum = local_agg_data[2].item()
            global_psnr_valid_count = int(local_agg_data[3].item())
            global_ssim_valid_count = int(local_agg_data[4].item())
            global_lpips_valid_count = int(local_agg_data[5].item())
        else:
            global_lpips_sum = 0.0  # Explicitly zero if not calculated
            global_psnr_valid_count = int(local_agg_data[2].item())
            global_ssim_valid_count = int(local_agg_data[3].item())
            global_lpips_valid_count = 0  # Explicitly zero if not calculated

        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0

        if global_psnr_valid_count > 0:
            avg_psnr = global_psnr_sum / global_psnr_valid_count

        if global_ssim_valid_count > 0:
            avg_ssim = global_ssim_sum / global_ssim_valid_count

        if calculate_lpips and global_lpips_valid_count > 0:
            avg_lpips = global_lpips_sum / global_lpips_valid_count

        if is_main_process:
            if self.logger:
                self.logger.info(f"--- Evaluation Results (Total Images Evaluated: ---")
                self.logger.info(f"Overall Average PSNR: {avg_psnr:.8f}, Overall Average SSIM: {avg_ssim:.8f}")
                if calculate_lpips:
                    self.logger.info(f"Overall Average LPIPS: {avg_lpips:.8f}")
            else:
                # Fallback print if logger not available
                print(f"--- Evaluation Results (Total Images Evaluated:  ---")
                print(f"Overall Average PSNR: {avg_psnr:.8f}, Overall Average SSIM: {avg_ssim:.8f}")
                if calculate_lpips:
                    print(f"Overall Average LPIPS: {avg_lpips:.8f}")

        if calculate_lpips:
            return avg_psnr, avg_ssim, avg_lpips
        else:
            return avg_psnr, avg_ssim

    def lr_lambda(self, scheduler_type):
        # 根据 scheduler_type 定义学习率调度策略
        if scheduler_type == 'linear':
            return lambda epoch: 1.0 - max(0, epoch - 500) / 500.0
        else:  # 默认情况下使用 constant scheduler
            return lambda epoch: 1.0

    @timeit
    def train_model(self, args):
        # 在这里初始化损失函数
        if args.loss_function == 'mse':
            self.criterion = MSELoss()
        elif args.loss_function == 'mae':
            self.criterion = L1Loss()
        elif args.loss_function == 'gmse':
            self.criterion = GradientMSELoss()
        elif args.loss_function == 'sml':
            self.criterion = SSIMMSELoss()
        elif args.loss_function == 'fr':
            self.criterion = FRLoss()
        elif args.loss_function == 'freq':
            self.criterion = FreqLoss()

        # 定义优化器
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)

        # 定义学习率调度器
        if args.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epoches, eta_min=self.eta_min
            )
        elif args.scheduler_type == 'fixed_then_cosine':
            fixed_epochs = 65
            initial_lr = args.lr
            scheduler_fixed = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoches - fixed_epochs,
                eta_min=self.eta_min
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[scheduler_fixed, scheduler_cosine],
                milestones=[fixed_epochs]
            )
        else:
            lr_lambda = self.lr_lambda(args.scheduler_type)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.model, self.optimizer, self.scheduler = self.load_latest_checkpoint(self.model,
                                                                                 self.optimizer,
                                                                                 self.scheduler,
                                                                                 self.checkpoint_filename)
        self.model.to(self.device)
        self.start_epoch += 1

        # set up progress bar
        progress_bar = tqdm(total=len(self.train_loader) * self.epoches, desc='Training',
                            initial=(self.start_epoch-1) * len(self.train_loader))

        for epoch in range(self.start_epoch, self.epoches + 1):
            self.model.train()
            self.model.module.grid = False

            if self.logger:  # 记录每个 epoch 的开始
                self.logger.info(f"Epoch {epoch}/{self.epoches} started.")

            epoch_loss = 0.0
            for batch_idx, (blur_images, sharp_images) in enumerate(self.train_loader):
                blur_images, sharp_images = blur_images.to(self.device), sharp_images.to(self.device)

                # 数据增强
                if self.if_augmentation:
                    augment_images(blur_images, sharp_images)

                self.optimizer.zero_grad()
                outputs = self.model(blur_images)
                sharp_images = sharp_images.to(outputs.device)

                # 计算损失
                loss = self.criterion(outputs, sharp_images)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 更新进度条
                progress_bar.update(1)
                if batch_idx % 50 == 0:  # 每10个批次更新一次
                    progress_bar.set_postfix({'Epoch': epoch, 'Loss': loss.item()})

                # 记录每个批次的损失
                if self.logger and batch_idx % 50 == 0:  # 每10个批次记录一次
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item():.8f}")

                # tensorboard
                if self.writer and batch_idx % 50 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

                epoch_loss += loss.item()  # 累加损失

            # 记录每个 epoch 完成的平均损失
            if self.logger:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.8f}")

            # tensorboard
            if self.writer:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, epoch)

            # 学习率调度器步进
            self.scheduler.step()

            # 只在主进程保存检查点
            if dist.get_rank() == 0:
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss.item(), self.checkpoint_filename)

            # 每隔一定的 epoch 记录一次学习率
            if self.logger and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    self.logger.info(f"Learning rate at epoch {epoch + 1}: {param_group['lr']:.8f}")

            # tensorboard
            if self.writer and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    self.writer.add_scalar('train/lr', param_group['lr'], epoch)

            if epoch >= 600 and epoch % 200 == 0:
                avg_psnr, avg_ssim = self.evaluate_model(self.model, self.val_loader)
                if self.logger:
                    self.logger.info(f"Epoch {epoch + 1} psnr: {avg_psnr:.8f}, ssim {avg_ssim:.8f} on val data.")

                self.early_stopping(epoch, avg_psnr)

                if self.early_stopping.early_stop:
                    if self.logger:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    if dist.get_rank() == 0:
                        save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss.item(), self.checkpoint_filename)
                    break

        progress_bar.close()


def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift