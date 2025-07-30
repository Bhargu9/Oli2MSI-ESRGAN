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

# Import the model architecture from your model.py file
from model import GeneratorRRDB

# --- IMPORTANT: These must match the values used in your dataset.py ---
LR_MIN = 0.0206
LR_MAX = 0.2737
HR_MIN = 0.0191
HR_MAX = 0.4274

def load_and_preprocess_lr(lr_path, device):
    """
    Loads a TIF image, normalizes it to the [-1, 1] range expected by
    the PyTorch model, and prepares it as a tensor.
    """
    with rasterio.open(lr_path) as src:
        # Rasterio reads as (channels, height, width), which is perfect for PyTorch
        lr_image = src.read().astype(np.float32)

        # 1. Normalize from raw data range to [0, 1]
        lr_image = (lr_image - LR_MIN) / (LR_MAX - LR_MIN)
        # 2. Normalize from [0, 1] to [-1, 1]
        lr_image = (lr_image * 2) - 1

        # Convert to PyTorch tensor, add batch dimension, and send to device
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).to(device)
        return lr_tensor

def postprocess_sr(sr_tensor):
    """
    Converts the model's output tensor from [-1, 1] back to a standard
    8-bit image format (0-255) for saving and evaluation.
    """
    # Move tensor to CPU, remove batch dimension
    sr_image = sr_tensor.squeeze(0).cpu()
    # Permute from (C, H, W) to (H, W, C) for NumPy/PIL
    sr_image = sr_image.permute(1, 2, 0).numpy()
    
    # De-normalize from [-1, 1] back to [0, 255]
    sr_image = (sr_image + 1) / 2.0 * 255.0
    
    # Clip values to be safe and convert to 8-bit integer
    sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)
    return sr_image

def load_hr_for_eval(hr_path):
    """
    Loads the ground truth HR image and scales it to the standard 0-255 range
    for fair comparison with the model's output.
    """
    with rasterio.open(hr_path) as src:
        hr_image = src.read().transpose(1, 2, 0).astype(np.float32)
        # Normalize from raw data range to [0, 255]
        hr_image = (hr_image - HR_MIN) / (HR_MAX - HR_MIN) * 255.0
        hr_image = np.clip(hr_image, 0, 255).astype(np.uint8)
        return hr_image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    print("Loading generator model...")
    try:
        model = GeneratorRRDB(3, num_res_blocks=args.n_rrdb_blocks).to(device)
        model.load_state_dict(torch.load(args.weights_path))
        model.eval()  # Set model to evaluation mode
        print(f"Loaded weights from {args.weights_path}")
    except Exception as e:
        print(f"Error loading model or weights: {e}")
        return

    # --- Find Test Image Pairs ---
    lr_files = sorted(glob.glob(os.path.join(args.lr_dir, '*.TIF')))
    if not lr_files:
        print(f"Error: No .TIF files found in LR directory: {args.lr_dir}")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    image_count = 0

    # --- Evaluation Loop ---
    with torch.no_grad():
        for lr_path in lr_files:
            base_filename = os.path.basename(lr_path)
            hr_path = os.path.join(args.hr_dir, base_filename)
            
            if not os.path.exists(hr_path):
                print(f"Warning: HR file for {base_filename} not found, skipping.")
                continue

            print(f"\nProcessing: {base_filename}")
            
            # 1. Load and preprocess LR image
            lr_tensor = load_and_preprocess_lr(lr_path, device)

            # 2. Generate SR image
            start_time = time.time()
            sr_tensor = model(lr_tensor)
            end_time = time.time()
            print(f"  Inference time: {end_time - start_time:.4f} seconds")

            # 3. Post-process SR image for evaluation/saving
            sr_image_uint8 = postprocess_sr(sr_tensor)

            # 4. Load HR image for evaluation
            hr_image_uint8 = load_hr_for_eval(hr_path)
            
            # Ensure dimensions match before calculating metrics
            if sr_image_uint8.shape != hr_image_uint8.shape:
                print(f"  Warning: Shape mismatch between SR {sr_image_uint8.shape} and HR {hr_image_uint8.shape}. Skipping metrics.")
                continue

            # 5. Calculate Metrics
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

            # 6. Save Generated SR Image as PNG
            try:
                save_path = os.path.join(args.output_dir, f"sr_{base_filename.replace('.TIF', '.png')}")
                Image.fromarray(sr_image_uint8).save(save_path)
                print(f"  Saved SR image to: {save_path}")
            except Exception as e:
                print(f"  Error saving SR image: {e}")

    # --- Calculate and Print Average Metrics ---
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
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the generator .pth weights file.')
    parser.add_argument('--n_rrdb_blocks', type=int, required=True, help='Number of RRDB blocks in the model (must match training).')
    parser.add_argument('--lr_dir', type=str, required=True, help='Directory containing low-resolution .TIF test images.')
    parser.add_argument('--hr_dir', type=str, required=True, help='Directory containing corresponding HR .TIF test images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated SR images.')
    
    args = parser.parse_args()
    main(args)