import torch
import torch.nn as nn
from .calculate_util import calculate_psnr, calculate_ssim
from tqdm import tqdm  # 引入 tqdm 库
from .data_process import augment_images
from .trainer import *
from utils.model_util import *
import numpy as np
import cv2
import rawpy


# 训练和评估去模糊模型
def train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       val_loader,
                       device,
                       num_epoches=1000,
                       eta_min=1e-6,
                       checkpoint_filename='checkpoint.pth',
                       model_name='Model',
                       if_augmentation=True,
                       seed=42,
                       args=None,
                       logger=None,
                       local_rank=0,
                       writer=None,):
    if logger:
        logger.info(f"Training and evaluating {model_name} with seed {seed}...")

    trainer = Trainer(model=model,
                      epoches=num_epoches,
                      eta_min=eta_min,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device,
                      checkpoint_filename=checkpoint_filename,
                      if_augmentation=if_augmentation,
                      logger=logger,
                      writer=writer
                      )
    # train model
    trainer.train_model(args)

    psnr, ssim = trainer.evaluate_model_final(model, test_loader)

    if local_rank == 0:
        # save
        results_filepath = os.path.join(args.experiment_dir, f'{model_name}_results.txt')
        with open(results_filepath, 'a') as f:
            f.write(f"{model_name} (seed {seed}) - Average PSNR: {psnr:.8f}, Average SSIM: {ssim:.8f}\n")
            f.write(f"-----------------------------\n")

        # 记录保存结果的日志
        if logger:
            logger.info(f"Results saved to {results_filepath}.")

def train_and_evaluate_gopro(model,
                       train_loader,
                       test_loader,
                       val_loader,
                       device,
                       num_epoches=1000,
                       eta_min=1e-6,
                       checkpoint_filename='checkpoint.pth',
                       model_name='Model',
                       if_augmentation=True,
                       seed=42,
                       args=None,
                       logger=None,
                       local_rank=0,
                       writer=None,):
    if logger:
        logger.info(f"Training and evaluating {model_name} with seed {seed}...")



    trainer = GoProTrainer(model=model,
                      epoches=num_epoches,
                      eta_min=eta_min,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device,
                      checkpoint_filename=checkpoint_filename,
                      if_augmentation=if_augmentation,
                      logger=logger,
                      writer=writer
                      )

    '''checkpoint_data = torch.load('checkpoints/net_g_latest.pth')
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        params_state_dict = checkpoint_data['params']
        prefixed_state_dict = {}
        for k, v in params_state_dict.items():
            prefixed_state_dict['module.' + k] = v
        model.load_state_dict(prefixed_state_dict)
        model.to(device)'''

    # train model
    trainer.train_model(args)
    psnr, ssim = trainer.evaluate_model_final(model, test_loader)

    if local_rank == 0:
        # save
        results_filepath = os.path.join(args.experiment_dir, f'{model_name}_results.txt')
        with open(results_filepath, 'a') as f:
            f.write(f"{model_name} (seed {seed}) - Average PSNR: {psnr:.8f}, Average SSIM: {ssim:.8f}\n")
            f.write(f"-----------------------------\n")

        # 记录保存结果的日志
        if logger:
            logger.info(f"Results saved to {results_filepath}.")

def train_and_evaluate_realblur(model,
                       train_loader,
                       test_loader,
                       val_loader,
                       device,
                       num_epoches=1000,
                       eta_min=1e-6,
                       checkpoint_filename='checkpoint.pth',
                       model_name='Model',
                       if_augmentation=True,
                       seed=42,
                       args=None,
                       logger=None,
                       local_rank=0,
                       writer=None,):
    if logger:
        logger.info(f"Training and evaluating {model_name} with seed {seed}...")



    trainer = RealBlurTrainer(model=model,
                      epoches=num_epoches,
                      eta_min=eta_min,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device,
                      checkpoint_filename=checkpoint_filename,
                      if_augmentation=if_augmentation,
                      logger=logger,
                      writer=writer
                      )

    '''checkpoint_data = torch.load('checkpoints/net_g_latest.pth')
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        params_state_dict = checkpoint_data['params']
        prefixed_state_dict = {}
        for k, v in params_state_dict.items():
            prefixed_state_dict['module.' + k] = v
        model.load_state_dict(prefixed_state_dict)
        model.to(device)'''

    # train model
    trainer.train_model(args)
    psnr, ssim = trainer.evaluate_model_final(model, test_loader)

    if local_rank == 0:
        # save
        results_filepath = os.path.join(args.experiment_dir, f'{model_name}_results.txt')
        with open(results_filepath, 'a') as f:
            f.write(f"{model_name} (seed {seed}) - Average PSNR: {psnr:.8f}, Average SSIM: {ssim:.8f}\n")
            f.write(f"-----------------------------\n")

        # 记录保存结果的日志
        if logger:
            logger.info(f"Results saved to {results_filepath}.")

def evaluate_gopro(model,
                       train_loader,
                       test_loader,
                       val_loader,
                       device,
                       num_epoches=1000,
                       eta_min=1e-6,
                       checkpoint_filename='checkpoint.pth',
                       model_name='Model',
                       if_augmentation=True,
                       seed=42,
                       args=None,
                       logger=None,
                       local_rank=0,
                       writer=None,):
    if logger:
        logger.info(f"Training and evaluating {model_name} with seed {seed}...")



    trainer = GoProTrainer(model=model,
                      epoches=num_epoches,
                      eta_min=eta_min,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device,
                      checkpoint_filename=checkpoint_filename,
                      if_augmentation=if_augmentation,
                      logger=logger,
                      writer=writer
                      )

    psnr, ssim = trainer.evaluate_model_final(model, test_loader)

    print(f'psnr: {psnr:.8f}, ssim: {ssim:.8f}')

def evaluate_raw(model,
                       train_loader,
                       test_loader,
                       val_loader,
                       device,
                       num_epoches=1000,
                       eta_min=1e-6,
                       checkpoint_filename='checkpoint.pth',
                       model_name='Model',
                       if_augmentation=True,
                       seed=42,
                       args=None,
                       logger=None,
                       local_rank=0,
                       writer=None,):
    if logger:
        logger.info(f"Training and evaluating {model_name} with seed {seed}...")



    trainer = Trainer(model=model,
                      epoches=num_epoches,
                      eta_min=eta_min,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device,
                      checkpoint_filename=checkpoint_filename,
                      if_augmentation=if_augmentation,
                      logger=logger,
                      writer=writer
                      )

    psnr, ssim = trainer.evaluate_model_final(model, test_loader)

    print(f'psnr: {psnr:.8f}, ssim: {ssim:.8f}')

def evaluate_realblur(model,
                       train_loader,
                       test_loader,
                       val_loader,
                       device,
                       num_epoches=1000,
                       eta_min=1e-6,
                       checkpoint_filename='checkpoint.pth',
                       model_name='Model',
                       if_augmentation=True,
                       seed=42,
                       args=None,
                       logger=None,
                       local_rank=0,
                       writer=None,
                       calculate_lpips=False):
    if logger:
        logger.info(f"Training and evaluating {model_name} with seed {seed}...")



    trainer = RealBlurTrainer(model=model,
                      epoches=num_epoches,
                      eta_min=eta_min,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device,
                      checkpoint_filename=checkpoint_filename,
                      if_augmentation=if_augmentation,
                      logger=logger,
                      writer=writer
                      )

    if calculate_lpips:
        psnr, ssim, lpips = trainer.evaluate_model_final(model, test_loader, calculate_lpips=calculate_lpips)
        print(f'psnr: {psnr:.8f}, ssim: {ssim:.8f}, lpips: {lpips: 8f}')
    else:
        psnr, ssim = trainer.evaluate_model_final(model, test_loader, calculate_lpips=calculate_lpips)
        print(f'psnr: {psnr:.8f}, ssim: {ssim:.8f}')