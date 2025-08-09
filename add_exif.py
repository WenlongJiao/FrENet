import os
import subprocess
import rawpy
import torch
import imageio
import numpy as np
from tensorstore import float32

from models.frenet import FrENet
from utils.model_util import *



def process(model, device, checkpoint_file, target_path):
    model.eval()

    model, _, _, _, _ = load_checkpoint(model, None, None, checkpoint_file, device)

    dataset_dir = 'test_dataset'
    metadata_dir = 'metadata'

    max_pixel_value = 16383.0


    # 遍历dataset中的每个子文件夹
    for root, dirs, files in os.walk(dataset_dir):
        # 只处理包含blur_raw文件夹的子文件夹
        if 'blur_raw' in dirs:
            blur_raw_dir = os.path.join(root, 'blur_raw')
            model_raw_dir = os.path.join(root, target_path)
            os.makedirs(model_raw_dir, exist_ok=True)  # 创建model_raw文件夹

            # 获取与blur_raw中的.dng文件对应的.txt文件路径
            for dng_file in os.listdir(blur_raw_dir):
                if dng_file.endswith('.dng'):
                    input_dng_path = os.path.join(blur_raw_dir, dng_file)
                    output_tif_path = os.path.join(model_raw_dir, dng_file.replace('.dng', '.tif'))

                    # model raw image
                    with rawpy.imread(input_dng_path) as raw:
                        black_level = np.mean(raw.black_level_per_channel)
                        input_raw = raw.raw_image_visible.astype(np.float32)
                        input_raw = (input_raw - black_level) / (max_pixel_value - black_level) # [0,1]
                        input_raw = torch.tensor(input_raw, dtype=torch.float32).unsqueeze(0).to(device)
                        input_raw = input_raw.unsqueeze(0) # [1, 1, h, w]
                        input_raw = Packing(input_raw) # 4 channel
                        with torch.no_grad():
                            output_raw = model(input_raw) # [1, 4, h, w]
                        output_raw = Unpacking(output_raw) # 1 channel
                        output_raw = output_raw.squeeze(0).squeeze(0)
                        output_raw = output_raw.cpu().detach().numpy()
                        output_raw = output_raw * (max_pixel_value - black_level) + black_level
                        output_raw = np.clip(output_raw, 0, 65535)
                        output_raw = output_raw.astype(np.uint16)

                    imageio.imwrite(output_tif_path, output_raw)
                    print(f"生成新的TIFF文件: {output_tif_path}")

                    # 构造txt文件路径，txt文件和dng文件同名
                    txt_file = os.path.join(metadata_dir, root.split(os.sep)[-1], dng_file.replace('.dng', '.txt'))

                    # 如果txt文件存在
                    if os.path.exists(txt_file):
                        # 构造exiftool命令
                        output_dng_file = os.path.join(model_raw_dir, dng_file)  # 新的dng文件路径
                        command = [
                            './exiftool.exe',  # exiftool的路径
                            '-@', txt_file,    # 使用txt文件中的参数
                            '-o', output_dng_file,  # 输出路径
                            output_tif_path  # 输入的原始dng文件路径
                        ]

                        try:
                            subprocess.run(command, check=True)
                            print(f"生成新的DNG文件: {output_dng_file}")
                            # 删除中间生成的TIFF文件
                            os.remove(output_tif_path)
                            print(f"删除TIFF文件: {output_tif_path}")
                        except subprocess.CalledProcessError as e:
                            print(f"生成DNG文件时出错: {e}")
                    else:
                        print(f"未找到对应的txt文件: {txt_file}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'model': {
            'name': 'FrENet',
            'params': {
                'width': 32,
                'middle_blk_num': 8,
                'enc_blk_nums': [2, 2, 4],
                'dec_blk_nums': [4, 2, 2],
                'train_size': 64,
                'img_size': 64,
                'grid_overlap_size': 16,
            }
        },
        'processing': {
            'checkpoint_path': 'checkpoints/FrENet_Raw.pth',
            'output_folder': 'frenet_raw',
        }
    }
    # ==============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 使用 ** 操作符将字典中的参数解包并传入模型构造函数
    #    这要求 LoFormer 的 __init__ 方法能接受这些参数
    model_params = config['model']['params']
    model = FrENet(**model_params).to(device)

    # 2. 从配置字典中获取处理参数
    checkpoint_path = config['processing']['checkpoint_path']
    output_folder = config['processing']['output_folder']

    # 3. 调用处理函数
    process(model, device, checkpoint_path, output_folder)

if __name__ == "__main__":
    main()