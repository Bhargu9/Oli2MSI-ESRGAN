# esrgan.py
import argparse
import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import GeneratorRRDB, Discriminator
from dataset import Oli2MSIDataset
from utils import save_weights_as_h5, save_tensor_as_tif

def main():
    parser = argparse.ArgumentParser(description="A simplified and stabilized ESRGAN training script.")
    # --- Simplified Model and Training ---
    parser.add_argument("--n_pretrain_epochs", type=int, default=20, help="Longer pre-training for a better start.")
    parser.add_argument("--n_gan_epochs", type=int, default=200)
    parser.add_argument("--n_rrdb_blocks", type=int, default=10, help="Reduced number of RRDB blocks for stability.")
    parser.add_argument("--lr", type=float, default=5e-5, help="A single, conservative learning rate.")
    # --- Loss Weights (No Perceptual Loss) ---
    parser.add_argument("--lambda_adv", type=float, default=1e-3)
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="Pixel loss is the primary driver.")
    # --- Data and Logistics ---
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--hr_crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("images/validation_png", exist_ok=True); os.makedirs("images/validation_tif", exist_ok=True); os.makedirs("saved_models", exist_ok=True)
    print("--- ESRGAN Training: Back to Basics for Stability ---"); print(opt)
    
    lr_files, hr_files = sorted(glob.glob(f"{opt.lr_dir}/*.TIF")), sorted(glob.glob(f"{opt.hr_dir}/*.TIF"))
    image_pairs = list(zip(lr_files, hr_files)); random.Random(42).shuffle(image_pairs)
    num_val = int(len(image_pairs)*0.1); train_pairs, val_pairs = image_pairs[:-num_val], image_pairs[-num_val:]
    
    train_dataset = Oli2MSIDataset(list(zip(*train_pairs))[0], list(zip(*train_pairs))[1], opt.hr_crop_size, 4)
    val_dataset = Oli2MSIDataset(list(zip(*val_pairs))[0], list(zip(*val_pairs))[1], opt.hr_crop_size, 4)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    generator = GeneratorRRDB(3, num_res_blocks=opt.n_rrdb_blocks).to(device)
    discriminator = Discriminator(input_shape=(3, opt.hr_crop_size, opt.hr_crop_size)).to(device)
    
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
    
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'max', patience=10, factor=0.5, verbose=True)
    
    print("\n--- STARTING L1 PRE-TRAINING ---")
    for epoch in range(opt.n_pretrain_epochs):
        generator.train()
        for i, (imgs_lr, imgs_hr, _) in enumerate(train_loader):
            imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)
            optimizer_G.zero_grad()
            loss_pixel = criterion_pixel(generator(imgs_lr), imgs_hr)
            if torch.isnan(loss_pixel): raise RuntimeError("Pixel loss is NaN during pre-training!")
            loss_pixel.backward()
            optimizer_G.step()
            if i % 100 == 0: print(f"[PRE-TRAIN Epoch {epoch}] [L1 loss: {loss_pixel.item():.5f}]")
    
    print("\n--- STARTING SIMPLIFIED ADVERSARIAL TRAINING ---\n")
    patience_counter, best_psnr = 0, 0.0

    for epoch in range(opt.n_gan_epochs):
        generator.train(); discriminator.train()
        for i, (imgs_lr, imgs_hr, _) in enumerate(train_loader):
            imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            loss_D = criterion_GAN(discriminator(imgs_hr), torch.ones_like(discriminator(imgs_hr))) + \
                     criterion_GAN(discriminator(generator(imgs_lr).detach()), torch.zeros_like(discriminator(generator(imgs_lr).detach())))
            if torch.isnan(loss_D): raise RuntimeError("Discriminator loss is NaN!")
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            gen_hr = generator(imgs_lr)
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            loss_G_adv = criterion_GAN(discriminator(gen_hr), torch.ones_like(discriminator(gen_hr)))
            loss_G = opt.lambda_pixel * loss_pixel + opt.lambda_adv * loss_G_adv
            if torch.isnan(loss_G): raise RuntimeError("Generator loss is NaN!")
            loss_G.backward()
            optimizer_G.step()
            
            if i % 100 == 0: print(f"[GAN Epoch {epoch}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # --- Validation ---
        generator.eval(); val_psnr, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for val_i, (val_lr, val_hr, hr_path) in enumerate(val_loader):
                gen_hr_val = generator(val_lr.to(device))
                gen_hr_val_save = torch.clamp((gen_hr_val + 1) / 2.0, 0.0, 1.0)
                val_hr_save = (val_hr.to(device) + 1) / 2.0
                val_psnr += psnr_metric(gen_hr_val_save, val_hr_save)
                val_ssim += ssim_metric(gen_hr_val_save, val_hr_save)
                if val_i == 0:
                    save_image(gen_hr_val_save[0], f"images/validation_png/epoch_{epoch}.png")
                    save_tensor_as_tif(gen_hr_val[0], hr_path[0], f"images/validation_tif/epoch_{epoch}.tif")

        avg_psnr, avg_ssim = val_psnr / len(val_loader), val_ssim / len(val_loader)
        scheduler_G.step(avg_psnr)

        print(f"\n--- Validation Epoch {epoch} ---\nAvg. PSNR: {avg_psnr:.4f} | Avg. SSIM: {avg_ssim:.4f}\n")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr; patience_counter = 0
            torch.save(generator.state_dict(), "saved_models/generator_best.pth")
            print(f"New best PSNR: {best_psnr:.4f}. Saved best model.")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{opt.patience}")

        if patience_counter >= opt.patience: print(f"Early stopping at epoch {epoch}."); break

    print("Training finished. Saving best generator weights to H5 format.")
    final_generator = GeneratorRRDB(3, num_res_blocks=opt.n_rrdb_blocks)
    final_generator.load_state_dict(torch.load("saved_models/generator_best.pth"))
    save_weights_as_h5(final_generator, "saved_models/generator_best.h5")

if __name__ == '__main__':
    main()