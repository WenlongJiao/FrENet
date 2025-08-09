import os
import io
import rawpy
import lmdb
import numpy as np
import torch
import cv2
import math
import torchvision.transforms.functional as TF
from pathlib import Path

from PIL import Image
from os import path as osp
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as F
from .augment_util import apply_augment
from .picture_util import pad_image
from .model_util import Packing, Unpacking
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Callable, Iterable, Optional, Tuple


def load_and_process_data(train_dir, test_dir, val_dir, data_domain='raw',
                          batch_size=32, num_workers=4, crop_size=128, distributed=False):
    train_dataset = PairedRawImageDataset(root_dir=train_dir, data_domain=data_domain, transform=PairedRandomCrop(size=crop_size))
    test_dataset = PairedRawImageDataset(root_dir=test_dir, data_domain=data_domain, transform=None)
    val_dataset = PairedRawImageDataset(root_dir=val_dir, data_domain=data_domain, transform=PairedRandomCrop(size=crop_size))

    train_sampler, val_sampler, test_sampler = None, None, None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # --- 修正 DataLoader 的调用方式 ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # 通常测试时 batch_size 为 1
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # 通常验证时 batch_size 为 1
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True
    )

    return train_loader, test_loader, val_loader


def load_and_process_raw_data(train_dir, test_dir, val_dir, batch_size=32, num_workers=4, crop_size=(128, 128), distributed=False):
    train_dataset = RawDataset(root_dir=train_dir, transform=PairedRandomCrop(size=crop_size))
    test_dataset = RawDataset(root_dir=test_dir, transform=None)
    val_dataset = RawDataset(root_dir=val_dir, transform=PairedRandomCrop(size=crop_size))

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, val_loader


def load_and_process_sidd_data(train_dir, test_dir, val_dir,
                               batch_size=32, num_workers=4, crop_size=128, distributed=False):
    # 创建SIDD数据集
    train_dataset = LMDBDataset(
        root_dir=train_dir,
        input_lmdb_dir='input_crops.lmdb',
        target_lmdb_dir='gt_crops.lmdb',
        input_key='noisy',
        target_key='clean',
        transform=PairedRandomCrop(size=crop_size)
    )

    test_dataset = LMDBDataset(
        root_dir=test_dir,
        input_lmdb_dir='input_crops.lmdb',
        target_lmdb_dir='gt_crops.lmdb',
        input_key='noisy',
        target_key='clean',
        transform=None
    )

    val_dataset = LMDBDataset(
        root_dir=val_dir,
        input_lmdb_dir='input_crops.lmdb',
        target_lmdb_dir='gt_crops.lmdb',
        input_key='noisy',
        target_key='clean',
        transform=PairedRandomCrop(size=crop_size)
    )

    # 设置采样器
    train_sampler, val_sampler, test_sampler = None, None, None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # 创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # 通常测试时 batch_size 为 1
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # 通常验证时 batch_size 为 1
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True
    )

    return train_loader, test_loader, val_loader


def load_and_process_gopro_data(train_dir, test_dir, val_dir, batch_size=32, num_workers=4, crop_size=(128, 128), distributed=False, ):
    train_dataset = GoProDataset(root_dir=train_dir, transform=PairedRandomCrop(size=crop_size))
    test_dataset = GoProDataset(root_dir=test_dir, blur_lmdb_dir='input.lmdb', sharp_lmdb_dir='target.lmdb', transform=None)
    val_dataset = GoProDataset(root_dir=val_dir, blur_lmdb_dir='input.lmdb', sharp_lmdb_dir='target.lmdb', transform=PairedRandomCrop(size=crop_size))

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None  # Common optimization
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, val_loader


def load_and_process_realblur_data(
        train_list_paths: str,
        val_list_paths: str,
        data_root: str,
        batch_size: int,
        num_workers: int = 4,
        distributed: bool = False,
        image_size: int = 256,
        preload: bool = False,
        train_over_sampling: int = 0,
        read_format: str = 'bgr',
):
    corrupt_config_list = [
        {'name': 'cutout', 'prob': 0.5, 'num_holes_range': (3, 3), 'hole_height_range': (1, 25), 'hole_width_range': (1, 25)},
        {'name': 'jpeg', 'quality_lower': 70, 'quality_upper': 90},
        {'name': 'motion_blur'},
        {'name': 'median_blur'},
        {'name': 'gamma'},
        {'name': 'rgb_shift'},
        {'name': 'hsv_shift'},
        {'name': 'sharpen'}
    ]

    train_dataset = PairedDataset.from_datalist({
        'data_root': data_root,
        'files_a': train_list_paths,
        'over_sampling': train_over_sampling,
        'size': image_size,
        'preload': preload,
        'preload_size': 0,  # 假设预加载不改变尺寸
        'verbose': False,
        'transform': True,
        'read_format': read_format,
        # 'corrupt': corrupt_config_list,
    })

    test_dataset = PairedDataset.from_datalist({
        'data_root': data_root,
        'files_a': val_list_paths,
        'over_sampling': 0,  # 验证集不过采样
        'size': image_size,
        'preload': preload,
        'preload_size': 0,
        'verbose': False,
        'transform': False,
        'read_format': read_format,
    })

    val_dataset = PairedDataset.from_datalist({
        'data_root': data_root,
        'files_a': val_list_paths,
        'over_sampling': 0,  # 验证集不过采样
        'size': image_size,
        'preload': preload,
        'preload_size': 0,
        'verbose': False,
        'transform': True,
        'read_format': read_format,
    })

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, val_loader


def load_and_process_hide_data(train_dir, test_dir, val_dir, batch_size=32, num_workers=4, crop_size=(128, 128), distributed=False):
    train_dataset = PairedPNGDataset(root_dir=train_dir, transform=PairedRandomCrop(size=crop_size))
    test_dataset = PairedPNGDataset(root_dir=test_dir, transform=None)
    val_dataset = PairedPNGDataset(root_dir=val_dir, transform=PairedRandomCrop(size=crop_size))

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, val_loader


def load_and_process_realblur_data_new(train_dir, test_dir, val_dir, batch_size=32, num_workers=4, crop_size=(128, 128), distributed=False):
    train_dataset = PairedPNGDataset(root_dir=train_dir, transform=PairedRandomCrop(size=crop_size))
    test_dataset = PairedPNGDataset(root_dir=test_dir, transform=None)
    val_dataset = PairedPNGDataset(root_dir=val_dir, transform=PairedRandomCrop(size=crop_size))

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
    )

    return train_loader, test_loader, val_loader


class PairedRawImageDataset(Dataset):
    def __init__(self, root_dir, data_domain='raw', transform=None, max_pixel_value=16383.0):
        super(PairedRawImageDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.data_domain = data_domain.lower()
        self.transform = transform
        self.max_pixel_value = float(max_pixel_value)

        if self.data_domain == 'raw':
            self.pairs = self._get_raw_image_pairs()
        elif self.data_domain == 'rgb':
            self.pairs = self._get_rgb_image_pairs()
        else:
            raise ValueError(f"Unknown data_domain: {self.data_domain}. Must be 'raw' or 'rgb'.")

        if not self.pairs:
            raise ValueError(f"No paired files found in {self.root_dir} for domain '{self.data_domain}'")
        print(f"Found {len(self.pairs)} image pairs for domain '{self.data_domain}' in {self.root_dir}")

    def _get_raw_image_pairs(self):
        pairs = []
        if not self.root_dir.exists(): return pairs
        for session_folder in self.root_dir.iterdir():
            if session_folder.is_dir():
                lq_path_dir, gt_path_dir = session_folder / 'blur_raw', session_folder / 'sharp_raw'
                if lq_path_dir.is_dir() and gt_path_dir.is_dir():
                    lq_files = {f.name for f in lq_path_dir.glob('*.dng')}
                    gt_files = {f.name for f in gt_path_dir.glob('*.dng')}
                    common_files = sorted(list(lq_files.intersection(gt_files)))
                    for img_name in common_files:
                        pairs.append((str(lq_path_dir / img_name), str(gt_path_dir / img_name)))
        return pairs

    def _get_rgb_image_pairs(self):
        pairs = []
        if not self.root_dir.exists(): return pairs
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        for session_folder in self.root_dir.iterdir():
            if session_folder.is_dir():
                lq_path_dir, gt_path_dir = session_folder / 'blur', session_folder / 'sharp'
                if lq_path_dir.is_dir() and gt_path_dir.is_dir():
                    lq_files = {f.name for f in lq_path_dir.iterdir() if f.suffix.lower() in img_extensions}
                    gt_files = {f.name for f in gt_path_dir.iterdir() if f.suffix.lower() in img_extensions}
                    common_files = sorted(list(lq_files.intersection(gt_files)))
                    for img_name in common_files:
                        pairs.append((str(lq_path_dir / img_name), str(gt_path_dir / img_name)))
        return pairs

    def _read_raw_image(self, path):
        with rawpy.imread(path) as raw:
            black_level = np.mean(raw.black_level_per_channel)
            image = raw.raw_image_visible.astype(np.float32)
            image = np.maximum(0., image - black_level)
            norm_range = self.max_pixel_value - black_level
            image = image / norm_range if norm_range > 0 else image * 0
            # 确保返回的是float32类型的tensor
            return torch.from_numpy(image).float().unsqueeze(0)

    def _read_rgb_image(self, path):
        """
        Reads an RGB image using cv2 and returns a [C, H, W] tensor.
        This implementation is based on the basicsr style.
        """
        # cv2.imread needs the path as a string
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")

        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.

        # Handle different channel counts and color spaces
        if img.ndim == 2:
            # Grayscale -> add channel dimension -> HWC (H, W, 1)
            img = np.expand_dims(img, axis=2)
        elif img.shape[2] == 4:
            # BGRA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC (Numpy) -> CHW (Tensor)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        return img_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_path, gt_path = self.pairs[idx]

        if self.data_domain == 'raw':
            lq_img, gt_img = self._read_raw_image(lq_path), self._read_raw_image(gt_path)

            lq_processed, gt_processed = Packing(lq_img), Packing(gt_img)

            if self.transform:
                lq_processed, gt_processed = self.transform(lq_processed, gt_processed)

            lq_final, gt_final = lq_processed, gt_processed

        elif self.data_domain == 'rgb':
            lq_img, gt_img = self._read_rgb_image(lq_path), self._read_rgb_image(gt_path)

            lq_processed, gt_processed = lq_img, gt_img
            if self.transform:
                lq_processed, gt_processed = self.transform(lq_processed, gt_processed)

            lq_final, gt_final = lq_processed, gt_processed

        return lq_final, gt_final


class PairedPNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'blur')
        self.groundtruth_dir = os.path.join(root_dir, 'gt')
        self.transform = transform
        self.pairs = self._get_image_pairs()

        if not self.pairs:
            print(f"Warning: No matching PNG image pairs found in {self.input_dir} and {self.groundtruth_dir}.")
            print("Please ensure both folders exist and contain identically named .png files.")

    def _get_image_pairs(self):

        pairs = []
        print(f"Scanning directories for PNG pairs: {self.input_dir} and {self.groundtruth_dir}")

        if not os.path.exists(self.input_dir):
            print(f"Error: Input directory not found: {self.input_dir}")
            return pairs
        if not os.path.exists(self.groundtruth_dir):
            print(f"Error: Groundtruth directory not found: {self.groundtruth_dir}")
            return pairs

        try:
            input_files = set([f for f in os.listdir(self.input_dir)
                               if os.path.isfile(os.path.join(self.input_dir, f)) and f.lower().endswith('.png')])
            groundtruth_files = set([f for f in os.listdir(self.groundtruth_dir)
                                     if os.path.isfile(os.path.join(self.groundtruth_dir, f)) and f.lower().endswith('.png')])
        except OSError as e:
            print(f"Error listing files: {e}")
            return pairs

        common_files = sorted(list(input_files.intersection(groundtruth_files)))

        for img_name in common_files:
            input_img_path = os.path.join(self.input_dir, img_name)
            groundtruth_img_path = os.path.join(self.groundtruth_dir, img_name)
            # Re-check just in case, though common_files should ensure existence by name
            if os.path.isfile(input_img_path) and os.path.isfile(groundtruth_img_path):
                pairs.append((input_img_path, groundtruth_img_path))
            else:
                print(f"Warning: Skipping pair due to missing file(s) despite name match: {img_name}")

        print(f"Found {len(pairs)} matching PNG pairs.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx >= len(self.pairs):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        input_img_path, gt_img_path = self.pairs[idx]

        try:
            # 使用cv2读取图像，默认为BGR格式
            import cv2
            input_image_bgr = cv2.imread(input_img_path)
            gt_image_bgr = cv2.imread(gt_img_path)

            # 检查图像是否成功加载
            if input_image_bgr is None or gt_image_bgr is None:
                raise ValueError(f"Failed to load images: {input_img_path}, {gt_img_path}")

            input_tensor = torch.from_numpy(input_image_bgr)
            gt_tensor = torch.from_numpy(gt_image_bgr)

            input_tensor = input_tensor.permute(2, 0, 1)
            gt_tensor = gt_tensor.permute(2, 0, 1)

            input_tensor = input_tensor.float() / 255.0
            gt_tensor = gt_tensor.float() / 255.0

            if self.transform:
                input_tensor, gt_tensor = self.transform(input_tensor, gt_tensor)

            return input_tensor, gt_tensor

        except Exception as e:
            print(f"Error loading or processing image pair: {input_img_path}, {gt_img_path}")
            print(f"Error details: {e}")
            return None, None  # Or raise e


class LMDBDataset(Dataset):
    """
    通用的LMDB数据集类，可以处理任何图片对数据
    支持任意两个LMDB目录作为输入和输出
    """

    def __init__(self, root_dir,
                 input_lmdb_dir='input.lmdb',
                 target_lmdb_dir='target.lmdb',
                 input_key='input',
                 target_key='target',
                 transform=None,
                 **kwargs):
        """
        Args:
            root_dir: 数据集根目录
            input_lmdb_dir: 输入LMDB目录名
            target_lmdb_dir: 目标LMDB目录名
            input_key: 输入数据的键名
            target_key: 目标数据的键名
            transform: 数据变换函数
            **kwargs: 其他参数
        """
        super().__init__()
        self.root_dir = root_dir
        self.input_lmdb_path = os.path.join(root_dir, input_lmdb_dir)
        self.target_lmdb_path = os.path.join(root_dir, target_lmdb_dir)
        self.input_key = input_key
        self.target_key = target_key
        self.transform = transform

        # 检查LMDB目录是否存在
        if not os.path.isdir(self.input_lmdb_path):
            raise FileNotFoundError(f"输入 LMDB 目录未找到: {self.input_lmdb_path}")
        if not os.path.isdir(self.target_lmdb_path):
            raise FileNotFoundError(f"目标 LMDB 目录未找到: {self.target_lmdb_path}")

        # 初始化LMDB环境
        self.input_env = None
        self.target_env = None
        self.path_keys = [input_key, target_key]

        try:
            self.paths = paired_paths_from_lmdb(
                [self.input_lmdb_path, self.target_lmdb_path],
                self.path_keys
            )
        except Exception as e:
            raise RuntimeError(f"使用 paired_paths_from_lmdb 加载路径时出错: {e}") from e

        if not self.paths:
            raise ValueError("paired_paths_from_lmdb 返回了空列表，请检查 LMDB 目录和 meta_info.txt 文件。")

    def _init_db(self):
        """初始化LMDB环境"""
        if self.input_env is None:
            try:
                self.input_env = lmdb.open(self.input_lmdb_path, readonly=True, lock=False,
                                           readahead=False, meminit=False)
            except lmdb.Error as e:
                raise lmdb.Error(f"无法打开输入 LMDB: {self.input_lmdb_path} - {e}")

        if self.target_env is None:
            try:
                self.target_env = lmdb.open(self.target_lmdb_path, readonly=True, lock=False,
                                            readahead=False, meminit=False)
            except lmdb.Error as e:
                raise lmdb.Error(f"无法打开目标 LMDB: {self.target_lmdb_path} - {e}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """获取数据项"""
        if self.input_env is None or self.target_env is None:
            self._init_db()

        path_info = self.paths[idx]
        input_key_str = path_info[f'{self.input_key}_path']
        target_key_str = path_info[f'{self.target_key}_path']

        try:
            input_key_bytes = input_key_str.encode('utf-8')
            target_key_bytes = target_key_str.encode('utf-8')
        except UnicodeEncodeError as e:
            raise ValueError(f"无法将键 '{input_key_str}' 或 '{target_key_str}' 编码为 UTF-8: {e}") from e

        try:
            with self.input_env.begin(write=False) as input_txn:
                input_img_bytes = input_txn.get(input_key_bytes)
            with self.target_env.begin(write=False) as target_txn:
                target_img_bytes = target_txn.get(target_key_bytes)
        except lmdb.Error as e:
            raise lmdb.Error(f"从 LMDB 读取键时出错: {e}")

        if input_img_bytes is None:
            raise KeyError(f"在输入 LMDB '{self.input_lmdb_path}' 中未找到键 '{input_key_str}'")
        if target_img_bytes is None:
            raise KeyError(f"在目标 LMDB '{self.target_lmdb_path}' 中未找到键 '{target_key_str}'")
        if len(input_img_bytes) == 0:
            raise ValueError(f"输入 LMDB 中键 '{input_key_str}' 的数据为空。")
        if len(target_img_bytes) == 0:
            raise ValueError(f"目标 LMDB 中键 '{target_key_str}' 的数据为空。")

        try:
            # 解码图像
            img_input = imfrombytes(input_img_bytes, flag='color', float32=True)
            img_target = imfrombytes(target_img_bytes, flag='color', float32=True)

            # 转换为tensor
            img_input_tensor, img_target_tensor = img2tensor([img_input, img_target],
                                                             bgr2rgb=True, float32=True)
        except (IOError, Exception) as e:
            print(f"!!! 解码或转换键 '{input_key_str}' 时出错: {e}")
            raise RuntimeError(f"无法解码或转换键 '{input_key_str}' 的图像数据: {e}") from e

        # 应用变换
        if self.transform:
            try:
                img_input_tensor, img_target_tensor = self.transform(img_input_tensor, img_target_tensor)
            except Exception as e:
                raise RuntimeError(f"应用 transform 到键 '{input_key_str}' 时出错: {e}") from e

        return img_input_tensor, img_target_tensor

    def __del__(self):
        """清理资源"""
        if self.input_env is not None:
            self.input_env.close()
        if self.target_env is not None:
            self.target_env.close()


class GoProDataset(LMDBDataset):
    """
    GoPro数据集类，继承自GenericLMDBDataset
    保持向后兼容性
    """

    def __init__(self, root_dir,
                 blur_lmdb_dir='blur_crops.lmdb',
                 sharp_lmdb_dir='sharp_crops.lmdb',
                 transform=None):
        super().__init__(
            root_dir=root_dir,
            input_lmdb_dir=blur_lmdb_dir,
            target_lmdb_dir=sharp_lmdb_dir,
            input_key='lq',
            target_key='gt',
            transform=transform
        )


class RawDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = self._get_image_pairs()
        # Data generated is 12-bit, so MAX_VAL is 2^12 - 1
        self.max_pixel_value = 2 ** 12 - 1  # 4095.0

        if len(self.pairs) == 0:
            print(f"Warning: No paired .npy files found in {self.root_dir}/sharp and {self.root_dir}/deg.")

    def _get_image_pairs(self):
        pairs = []
        sharp_root = os.path.join(self.root_dir, 'sharp')
        deg_root = os.path.join(self.root_dir, 'deg')

        if not os.path.isdir(sharp_root):
            print(f"Error: Sharp directory not found at {sharp_root}")
            return pairs
        if not os.path.isdir(deg_root):
            print(f"Error: Degraded directory not found at {deg_root}")
            return pairs

        sharp_scenes = set(os.listdir(sharp_root))
        deg_scenes = set(os.listdir(deg_root))
        common_scenes = sharp_scenes.intersection(deg_scenes)

        if not common_scenes:
            print(f"Warning: No common scene folders found between {sharp_root} and {deg_root}")

        for scene_name in sorted(list(common_scenes)):  # Sorting for consistent order
            sharp_scene_path = os.path.join(sharp_root, scene_name)
            deg_scene_path = os.path.join(deg_root, scene_name)

            if os.path.isdir(sharp_scene_path) and os.path.isdir(deg_scene_path):
                sharp_files = set([f for f in os.listdir(sharp_scene_path) if f.endswith('.npy')])
                deg_files = set([f for f in os.listdir(deg_scene_path) if f.endswith('.npy')])

                # Find common .npy filenames
                common_files = sharp_files.intersection(deg_files)

                if not common_files:
                    print(f"Warning: No common .npy files found in scene '{scene_name}'")

                for img_name in sorted(list(common_files)):
                    sharp_img_path = os.path.join(sharp_scene_path, img_name)
                    deg_img_path = os.path.join(deg_scene_path, img_name)

                    if os.path.isfile(sharp_img_path) and os.path.isfile(deg_img_path):
                        pairs.append((deg_img_path, sharp_img_path))
                    else:
                        print(f"Warning: Paired files not found, skipping: {deg_img_path} and {sharp_img_path}")

        print(f"Found {len(pairs)} paired images.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_img_path, sharp_img_path = self.pairs[idx]

        try:
            deg_img_np = np.load(deg_img_path)
            sharp_img_np = np.load(sharp_img_path)
        except Exception as e:
            print(f"Error loading files {deg_img_path} and {sharp_img_path}: {e}")
            raise e

        if deg_img_np.dtype != np.uint16 or sharp_img_np.dtype != np.uint16:
            print(f"Warning: Unexpected dtype. Deg: {deg_img_np.dtype}, Sharp: {sharp_img_np.dtype}")
        if deg_img_np.ndim != 3 or sharp_img_np.ndim != 3:
            print(f"Warning: Unexpected number of dimensions. Deg: {deg_img_np.ndim}, Sharp: {sharp_img_np.ndim}")
        if deg_img_np.shape != sharp_img_np.shape:
            print(f"Warning: Degraded and sharp images have different shapes: {deg_img_np.shape} vs {sharp_img_np.shape}")
            raise ValueError(f"Shape mismatch: {deg_img_path} and {sharp_img_path}")

        deg_img_normalized = deg_img_np.astype(np.float32) / self.max_pixel_value
        sharp_img_normalized = sharp_img_np.astype(np.float32) / self.max_pixel_value

        deg_img_tensor = torch.from_numpy(deg_img_normalized).permute(2, 0, 1)
        sharp_img_tensor = torch.from_numpy(sharp_img_normalized).permute(2, 0, 1)

        # deg_img_tensor = Packing(deg_img_tensor)
        # sharp_img_tensor = Packing(sharp_img_tensor)

        if self.transform:
            deg_img_tensor, sharp_img_tensor = self.transform(deg_img_tensor, sharp_img_tensor)

        # deg_img_tensor = Unpacking(deg_img_tensor)
        # sharp_img_tensor = Unpacking(sharp_img_tensor)

        return deg_img_tensor, sharp_img_tensor


class PairedDataset(Dataset):
    def __init__(self,
                 files_a: Iterable[str],
                 files_b: Iterable[str],
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corrupt_fn: Optional[Callable] = None,
                 preload: bool = True,
                 preload_size: Optional[int] = 0,
                 transform: bool = False,
                 if_corrupt: bool = False,
                 verbose=True,
                 read_format: str = 'bgr'):

        assert len(files_a) == len(files_b)
        assert read_format.lower() in ['bgr', 'rgb']

        self.preload = preload
        self.data_a = files_a
        self.data_b = files_b
        self.verbose = verbose
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.transform = transform
        self.if_corrupt = if_corrupt
        self.read_format = read_format.lower()

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            if files_a == files_b:
                self.data_a = self.data_b = preload_fn(self.data_a)
            else:
                self.data_a, self.data_b = map(preload_fn, (self.data_a, self.data_b))
            self.preload = True

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        jobs = tqdm(jobs, desc='preloading images', disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend='threading')(jobs)

    def _read_img(self, x: str):
        img = _read_img_fallback(x)  # Use the fallback function

        if img is None:
            print(f"Warning: Failed to read image {x}. Returning None.")
            return None

        if img.ndim == 3 and img.shape[-1] == 3:
            if self.read_format == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            print(f"Warning: Grayscale image detected at {x}. Converting to 3 channels.")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if self.read_format == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: Image {x} has unexpected shape {img.shape}. Returning as is.")

        return img

    @staticmethod
    def _preload(self, x: str, preload_size: int):
        img = self._read_img(x)
        if preload_size:
            h, w, *_ = img.shape
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def _preprocess(self, img, res):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return map(transpose, self.normalize_fn(img, res))

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        if not self.preload:
            a, b = map(self._read_img, (a, b))
        a, b = self._preprocess(a, b)
        a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
        if self.transform:
            a, b = self.transform_fn(a, b)
        if self.corrupt_fn is not None:
            a = self.corrupt_fn(a)
        # a, b = self._preprocess(a, b)
        return a, b

    '''@staticmethod
    def from_config(config):
        config = deepcopy(config)
        files_a, files_b = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b'))
        transform_fn = aug.get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        normalize_fn = aug.get_normalize()
        corrupt_fn = aug.get_corrupt_function(config['corrupt'])

        hash_fn = hash_from_paths
        # ToDo: add more hash functions
        verbose = config.get('verbose', True)
        data = subsample(data=zip(files_a, files_b),
                         bounds=config.get('bounds', (0, 1)),
                         hash_fn=hash_fn,
                         verbose=verbose)

        files_a, files_b = map(list, zip(*data))

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             preload=config['preload'],
                             preload_size=config['preload_size'],
                             corrupt_fn=corrupt_fn,
                             normalize_fn=normalize_fn,
                             transform_fn=transform_fn,
                             verbose=verbose)'''

    @staticmethod
    def from_datalist(config):
        config = deepcopy(config)
        datalist_path = config['files_a']
        over_sampling = config['over_sampling']
        data_root = config.get('data_root')
        data_list = []
        data_list1 = datalist_path.split(',')[0]
        data_list1 = open(data_list1, 'rt').read().splitlines()
        data_list1 = list(map(lambda x: x.strip().split(' '), data_list1))
        data_list += data_list1
        if over_sampling != 0:
            datalist1_upsample = math.ceil(float(over_sampling) / float(len(data_list1)))
            for i in range(int(datalist1_upsample) - 1):
                data_list += data_list1

        if len(datalist_path.split(',')) >= 2:
            data_list2 = datalist_path.split(',')[1]
            data_list2 = open(data_list2, 'rt').read().splitlines()
            data_list2 = list(map(lambda x: x.strip().split(' '), data_list2))
            data_list += data_list2

            if over_sampling != 0:
                datalist2_upsample = math.ceil(float(over_sampling) / float(len(data_list2)))
                for i in range(int(datalist2_upsample) - 1):
                    data_list += data_list2

        if len(datalist_path.split(',')) >= 3:
            data_list3 = datalist_path.split(',')[2]
            data_list3 = open(data_list3, 'rt').read().splitlines()
            data_list3 = list(map(lambda x: x.strip().split(' '), data_list3))
            data_list += data_list3

            if over_sampling != 0:
                datalist3_upsample = math.ceil(float(over_sampling) / float(len(data_list3)))
                for i in range(int(datalist3_upsample) - 1):
                    data_list += data_list3

        verbose = config.get('verbose', True)
        files_a = [os.path.join(data_root, blurred) for gt, blurred in data_list]
        files_b = [os.path.join(data_root, gt) for gt, blurred in data_list]

        transform_fn = PairedRandomCropForRealBlur(size=config['size'])
        # transform_fn = get_transforms(size=config['size'])
        # normalize_fn = get_normalize()
        # if_corrupt = config.get('if_corrupt', False)
        # corrupt_fn = None
        # if if_corrupt and 'corrupt' in config and config['corrupt'] is not None:
        # corrupt_fn = get_corrupt_function(config['corrupt'])
        normalize_fn = normalize_divide_by_255
        read_format = config.get('read_format', 'bgr')

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             preload=config['preload'],
                             preload_size=config['preload_size'],
                             normalize_fn=normalize_fn,
                             transform_fn=transform_fn,
                             # corrupt_fn=corrupt_fn,
                             verbose=verbose,
                             transform=config['transform'],
                             read_format=read_format)


class PairedRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        if img1.dim() < 2 or img2.dim() < 2:
            raise ValueError("Input tensors must have at least 2 dimensions (H, W)")

        i, j, h, w = transforms.RandomCrop.get_params(img1, self.size)

        cropped_img1 = F.crop(img1, i, j, h, w)
        cropped_img2 = F.crop(img2, i, j, h, w)

        return cropped_img1, cropped_img2


class PairedRandomCropForRealBlur:
    def __init__(self, size):
        # Ensure size is a tuple (height, width)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(f"Requested crop size {self.size} is not positive")

    def __call__(self, img1, img2):
        if isinstance(img1, torch.Tensor):
            if img1.ndim < 2:
                raise ValueError(f"Input tensor img1 must have at least 2 dimensions (H, W), got shape {img1.shape}")
            h_img, w_img = img1.shape[-2:]
        elif hasattr(img1, 'size'):
            w_img, h_img = img1.size
        else:
            if img1.ndim < 2:
                raise ValueError(f"Input numpy array img1 must have at least 2 dimensions (H, W), got shape {img1.shape}")
            h_img, w_img = img1.shape[:2]

        th, tw = self.size
        if h_img < th or w_img < tw:
            raise ValueError(f"Image size ({h_img}, {w_img}) is smaller than crop size ({th}, {tw}). Add padding or resizing logic.")
        i, j, h, w = transforms.RandomCrop.get_params(img1, self.size)
        cropped_img1 = F.crop(img1, i, j, h, w)
        cropped_img2 = F.crop(img2, i, j, h, w)

        return cropped_img1, cropped_img2


def normalize_divide_by_255(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1_float = img1.float()
        img2_float = img2.float()
    else:
        img1_float = img1.astype(float)
        img2_float = img2.astype(float)

    return img1_float / 255.0, img2_float / 255.0
    # return img1_float, img2_float


def augment_images(im1, im2):
    augs = ["blend", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
    probs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 每种增强的概率
    alphas = [0.6, 1.2, 0.001, 0.7, 0.7, 0.7]  # 每种增强的alpha值
    aux_prob, aux_alpha = 1.0, 1.2
    mix_p = None
    return apply_augment(im1, im2, augs=augs, probs=probs, alphas=alphas, aux_alpha=aux_alpha, aux_prob=aux_prob, mix_p=mix_p)


def paired_paths_from_lmdb(folders, keys):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths


def imfrombytes(content, flag='color', float32=False):
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def subsample(data: Iterable, bounds: Tuple[float, float], hash_fn: Callable, n_buckets=100, salt='', verbose=True):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def hash_from_paths(x: Tuple[str, str], salt: str = '') -> str:
    path_a, path_b = x
    names = ''.join(map(os.path.basename, (path_a, path_b)))
    return sha1(f'{names}_{salt}'.encode()).hexdigest()


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def _read_img_fallback(x: str):
    img = cv2.imread(x)
    if img is None:
        try:
            img_io = imread(x)
            if img_io is not None:
                if img_io.ndim == 3 and img_io.shape[-1] == 3:
                    img = cv2.cvtColor(img_io, cv2.COLOR_RGB2BGR)
                elif img_io.ndim == 2:
                    img = cv2.cvtColor(img_io, cv2.COLOR_GRAY2BGR)
                else:
                    img = img_io
            else:
                print(f"Warning: imageio imread failed for {x}")
        except Exception as e:
            print(f"Error reading image {x} with imageio fallback: {e}")
            img = None  # Ensure img is None on failure
    return img
