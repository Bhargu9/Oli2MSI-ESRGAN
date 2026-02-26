# test.py
import os
import argparse
import glob
import time
import numpy as np
import torch
import rasterio
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model import GeneratorRRDB


LR_MIN = 0.0206
LR_MAX = 0.2737
HR_MIN = 0.0191
HR_MAX = 0.4274

def load_and_preprocess_lr(lr_path, device):
    """
    Loads a TIF image, normalizes it to the [-1, 1] range expected by the PyTorch model, and prepares it as a tensor.
    """
    with rasterio.open(lr_path) as src:
        lr_image = src.read().astype(np.float32)
        lr_image = (lr_image - LR_MIN) / (LR_MAX - LR_MIN)
        lr_image = (lr_image * 2) - 1
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).to(device)
        return lr_tensor

def postprocess_sr(sr_tensor):
    """
    Converts the model's output tensor from [-1, 1] back to a standard 8-bit image format (0-255) for saving and evaluation.
    """
    sr_image = sr_tensor.squeeze(0).cpu()
    sr_image = sr_image.permute(1, 2, 0).numpy()
    sr_image = (sr_image + 1) / 2.0 * 255.0
    sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)
    return sr_image

def load_hr_for_eval(hr_path):
    """
    Loads the ground truth HR image and scales it to the standard 0-255 range for fair comparison with the model's output.
    """
    with rasterio.open(hr_path) as src:
        hr_image = src.read().transpose(1, 2, 0).astype(np.float32)
        hr_image = (hr_image - HR_MIN) / (HR_MAX - HR_MIN) * 255.0
        hr_image = np.clip(hr_image, 0, 255).astype(np.uint8)
        return hr_image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading generator model...")
    try:
        model = GeneratorRRDB(3, num_res_blocks=args.n_rrdb_blocks).to(device)
        model.load_state_dict(torch.load(args.weights_path))
        model.eval()  
        print(f"Loaded weights from {args.weights_path}")
    except Exception as e:
        print(f"Error loading model or weights: {e}")
        return

    lr_files = sorted(glob.glob(os.path.join(args.lr_dir, '*.TIF')))
    if not lr_files:
        print(f"Error: No .TIF files found in LR directory: {args.lr_dir}")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    image_count = 0

    with torch.no_grad():
        for lr_path in lr_files:
            base_filename = os.path.basename(lr_path)
            hr_path = os.path.join(args.hr_dir, base_filename)
            
            if not os.path.exists(hr_path):
                print(f"Warning: HR file for {base_filename} not found, skipping.")
                continue

            print(f"\nProcessing: {base_filename}")
            lr_tensor = load_and_preprocess_lr(lr_path, device)

            start_time = time.time()
            sr_tensor = model(lr_tensor)
            end_time = time.time()
            print(f"  Inference time: {end_time - start_time:.4f} seconds")
            sr_image_uint8 = postprocess_sr(sr_tensor)
            hr_image_uint8 = load_hr_for_eval(hr_path)
            
            if sr_image_uint8.shape != hr_image_uint8.shape:
                print(f"  Warning: Shape mismatch between SR {sr_image_uint8.shape} and HR {hr_image_uint8.shape}. Skipping metrics.")
                continue
            try:
                psnr_val = psnr(hr_image_uint8, sr_image_uint8, data_range=255)
                ssim_val = ssim(hr_image_uint8, sr_image_uint8, channel_axis=-1, data_range=255)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                image_count += 1
                
                print(f"  PSNR: {psnr_val:.4f} dB")
                print(f"  SSIM: {ssim_val:.4f}")

            except Exception as e:
                print(f"  Error calculating metrics: {e}")
                continue

            try:
                save_path = os.path.join(args.output_dir, f"sr_{base_filename.replace('.TIF', '.png')}")
                Image.fromarray(sr_image_uint8).save(save_path)
                print(f"  Saved SR image to: {save_path}")
            except Exception as e:
                print(f"  Error saving SR image: {e}")


    if image_count > 0:
        avg_psnr = total_psnr / image_count
        avg_ssim = total_ssim / image_count
        print("\n--- Evaluation Summary ---")
        print(f"Average PSNR over {image_count} images: {avg_psnr:.4f} dB")
        print(f"Average SSIM over {image_count} images: {avg_ssim:.4f}")
    else:
        print("\nNo valid images were processed.")

    print("\nTesting complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained PyTorch ESRGAN generator.")
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--n_rrdb_blocks', type=int, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
