import os
import subprocess

# 使用exif提取元数据

# 设置 dataset 文件夹的根路径
dataset_folder = 'test_dataset'
output_folder = 'metadata'  # 存储元数据的文件夹

# 如果 output 文件夹不存在，创建它
os.makedirs(output_folder, exist_ok=True)

# 遍历 dataset 文件夹下的所有子文件夹
for root, dirs, files in os.walk(dataset_folder):
    # 找到每个子文件夹中的 blur_raw 文件夹
    if 'blur_raw' in dirs:
        blur_raw_folder = os.path.join(root, 'blur_raw')

        # 获取 blur_raw 文件夹中的所有 DNG 文件
        dng_files = [f for f in os.listdir(blur_raw_folder) if f.lower().endswith('.dng')]

        for dng_file in dng_files:
            dng_path = os.path.join(blur_raw_folder, dng_file)

            # 设置元数据输出的文件路径，命名与 DNG 文件一致
            relative_path = os.path.relpath(root, dataset_folder)  # 获取相对路径
            metadata_folder = os.path.join(output_folder, relative_path)
            os.makedirs(metadata_folder, exist_ok=True)  # 确保每个文件夹存在

            # 输出文件与 DNG 文件名相同，只是后缀改为 .txt
            output_file = os.path.join(metadata_folder, f"{os.path.splitext(dng_file)[0]}_original.txt")

            with open(output_file, 'w') as f:
                subprocess.run(['exiftool', dng_path], stdout=f)

            print(f"Metadata for {dng_file} saved to {output_file}")
