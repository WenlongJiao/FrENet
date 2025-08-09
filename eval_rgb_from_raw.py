import os
import rawpy
import numpy as np
import torch
from utils.calculate_util import calculate_psnr, calculate_ssim

ROOT_DIR = 'test_dataset'
GROUND_TRUTH_FOLDER_NAME = 'sharp_raw'
TEST_FOLDER_NAME = 'frenet_raw'

CROP_BORDER = 0
TEST_Y_CHANNEL = False

def process_dng_to_rgb(dng_path):
    try:
        with rawpy.imread(dng_path) as raw:
            rgb_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        return rgb_image
    except Exception as e:
        print(f"  [Error] Failed to process file {dng_path}: {e}")
        return None

def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: Root directory '{ROOT_DIR}' not found. Please ensure the script is in the same directory as this folder.")
        return

    if not TEST_Y_CHANNEL and not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Warning: Your SSIM calculation ('_ssim_3d') requires CUDA,")
        print("         but no available GPU was detected.")
        print("You can set 'TEST_Y_CHANNEL' to 'True' at the top of the script")
        print("to calculate SSIM on the CPU (Y-channel only).")
        print("=" * 60 + "\n")
        # return

    all_psnr_scores = []
    all_ssim_scores = []

    scene_folders = sorted([f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))])

    for scene_folder in scene_folders:
        print(f"--- Processing scene: {scene_folder} ---")

        gt_dir = os.path.join(ROOT_DIR, scene_folder, GROUND_TRUTH_FOLDER_NAME)
        test_dir = os.path.join(ROOT_DIR, scene_folder, TEST_FOLDER_NAME)

        if not os.path.isdir(gt_dir) or not os.path.isdir(test_dir):
            print(f"  [Warning] Missing '{GROUND_TRUTH_FOLDER_NAME}' or '{TEST_FOLDER_NAME}' folder in {scene_folder}, skipping.")
            continue

        for filename in sorted(os.listdir(gt_dir)):
            if filename.lower().endswith('.dng'):
                gt_dng_path = os.path.join(gt_dir, filename)
                test_dng_path = os.path.join(test_dir, filename)

                if not os.path.isfile(test_dng_path):
                    print(f"  [Warning] Corresponding file {filename} not found in {test_dir}, skipping.")
                    continue

                print(f"  Processing image pair: {filename}")

                gt_rgb = process_dng_to_rgb(gt_dng_path)
                test_rgb = process_dng_to_rgb(test_dng_path)

                if gt_rgb is None or test_rgb is None:
                    continue

                try:
                    psnr = calculate_psnr(gt_rgb, test_rgb, CROP_BORDER, test_y_channel=TEST_Y_CHANNEL, data_range=255.0)
                    ssim = calculate_ssim(gt_rgb, test_rgb, CROP_BORDER, test_y_channel=TEST_Y_CHANNEL, data_range=255.0)

                    all_psnr_scores.append(psnr)
                    all_ssim_scores.append(ssim)

                    print(f"    PSNR: {psnr:.4f} dB")
                    print(f"    SSIM: {ssim:.4f}")

                except Exception as e:
                    print(f"    [Calculation Error] An error occurred while calculating metrics for image {filename}: {e}")

    if all_psnr_scores:
        avg_psnr = np.mean(all_psnr_scores)
        avg_ssim = np.mean(all_ssim_scores)

        print("\n" + "=" * 50)
        print("Overall Evaluation Results".center(50))
        print("=" * 50)
        print(f"Processed a total of {len(all_psnr_scores)} image pairs.")
        print(f"Configuration: CROP_BORDER={CROP_BORDER}, TEST_Y_CHANNEL={TEST_Y_CHANNEL}")
        print(f"      GT Folder: '{GROUND_TRUTH_FOLDER_NAME}', Test Folder: '{TEST_FOLDER_NAME}'")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("=" * 50)
    else:
        print("\nNo images were processed. Please check your folder structure and filenames.")

if __name__ == '__main__':
    main()