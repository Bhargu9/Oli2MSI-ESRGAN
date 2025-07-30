# esrgan_finetune.py
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
from model import GeneratorRRDB, Discriminator, FeatureExtractor
from dataset import Oli2MSIDataset
from utils import save_weights_as_h5, save_scientific_tif, save_visual_tif

def main():
    parser = argparse.ArgumentParser(description="Advanced fine-tuning for ESRGAN.")
    parser.add_argument("--load_pretrained_g", type=str, required=True, help="Path to the generator_best.pth from the stable run.")
    parser.add_argument("--n_rrdb_blocks", type=int, default=23, help="MUST MATCH the block count of the pre-trained model.")
    parser.add_argument("--n_finetune_epochs", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=1e-5, help="Generator learning rate.")
    parser.add_argument("--lr_d", type=float, default=4e-5, help="Discriminator learning rate.")
    parser.add_argument("--lambda_adv", type=float, default=5e-3)
    parser.add_argument("--lambda_pixel", type=float, default=1e-2)
    parser.add_argument("--lambda_content", type=float, default=1.0)
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--hr_crop_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("images/finetune_png", exist_ok=True)
    os.makedirs("images/finetune_tif_visual", exist_ok=True)
    os.makedirs("images/finetune_tif_scientific", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    print("--- ESRGAN Advanced Fine-tuning ---"); print(opt)

    lr_files, hr_files = sorted(glob.glob(f"{opt.lr_dir}/*.TIF")), sorted(glob.glob(f"{opt.hr_dir}/*.TIF"))
    image_pairs = list(zip(lr_files, hr_files)); random.Random(42).shuffle(image_pairs)
    num_val = int(len(image_pairs)*0.1); train_pairs, val_pairs = image_pairs[:-num_val], image_pairs[-num_val:]

    train_dataset = Oli2MSIDataset(list(zip(*train_pairs))[0], list(zip(*train_pairs))[1], opt.hr_crop_size, 4)
    val_dataset = Oli2MSIDataset(list(zip(*val_pairs))[0], list(zip(*val_pairs))[1], opt.hr_crop_size, 4)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    generator = GeneratorRRDB(3, num_res_blocks=opt.n_rrdb_blocks).to(device)
    discriminator = Discriminator(input_shape=(3, opt.hr_crop_size, opt.hr_crop_size)).to(device)
    feature_extractor = FeatureExtractor().to(device).eval()

    print(f"Loading pre-trained generator weights from {opt.load_pretrained_g}")
    generator.load_state_dict(torch.load(opt.load_pretrained_g))

    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d)
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'max', patience=5, factor=0.5, verbose=True)

    print("\n--- STARTING ADVANCED FINE-TUNING ---\n")
    patience_counter, best_score = 0, 0.0

    for epoch in range(opt.n_finetune_epochs):
        generator.train()
        discriminator.train()
        for i, (imgs_lr, imgs_hr, _) in enumerate(train_loader):
            imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_validity = discriminator(imgs_hr)
            fake_validity = discriminator(generator(imgs_lr).detach())
            
            loss_D_real = criterion_GAN(real_validity, torch.ones_like(real_validity))
            loss_D_fake = criterion_GAN(fake_validity, torch.zeros_like(fake_validity))
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            gen_hr = generator(imgs_lr)
            
            # Calculate all three losses
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            loss_content = criterion_content(feature_extractor(gen_hr), feature_extractor(imgs_hr).detach())
            loss_G_adv = criterion_GAN(discriminator(gen_hr), torch.ones_like(discriminator(gen_hr)))
            
            # Combine losses with their respective weights
            loss_G = opt.lambda_pixel * loss_pixel + opt.lambda_content * loss_content + opt.lambda_adv * loss_G_adv
            loss_G.backward()
            optimizer_G.step()
            
            if i % 50 == 0:
                print(
                    f"[Finetune Epoch {epoch}/{opt.n_finetune_epochs-1}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G total: {loss_G.item():.4f}] "
                    f"[adv: {loss_G_adv.item():.4f}, pix: {loss_pixel.item():.4f}, cont: {loss_content.item():.4f}]"
                )

        # --- Validation ---
        generator.eval()
        val_psnr, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for val_i, (val_lr, val_hr, hr_path) in enumerate(val_loader):
                gen_hr_val = generator(val_lr.to(device))
                # Clamp output before normalization for safety
                gen_hr_val_save = torch.clamp((gen_hr_val + 1) / 2.0, 0.0, 1.0)
                val_hr_save = (val_hr.to(device) + 1) / 2.0
                val_psnr += psnr_metric(gen_hr_val_save, val_hr_save)
                val_ssim += ssim_metric(gen_hr_val_save, val_hr_save)
                if val_i == 0:
                    # Save three versions of the output image
                    save_image(gen_hr_val_save[0], f"images/finetune_png/epoch_{epoch}.png")
                    save_visual_tif(gen_hr_val[0], f"images/finetune_tif_visual/epoch_{epoch}.tif")
                    save_scientific_tif(gen_hr_val[0], hr_path[0], f"images/finetune_tif_scientific/epoch_{epoch}.tif")

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        
        # Using a combined score for evaluation
        combined_score = (0.7 * avg_ssim) + (0.3 * (avg_psnr / 40.0))
        
        scheduler_G.step(combined_score)

        print(f"\n--- Validation Epoch {epoch} ---\nAvg. PSNR: {avg_psnr:.4f} | Avg. SSIM: {avg_ssim:.4f} | Combined Score: {combined_score:.4f}\n")

        if combined_score > best_score:
            best_score = combined_score
            patience_counter = 0
            torch.save(generator.state_dict(), "saved_models/generator_final_best.pth")
            torch.save(discriminator.state_dict(), "saved_models/discriminator_final_best.pth")
            print(f"New best score: {best_score:.4f}. Saved best fine-tuned models.")
        else:
            patience_counter += 1
            print(f"Score did not improve. Best: {best_score:.4f}. Patience: {patience_counter}/{opt.patience}")

        if patience_counter >= opt.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print("Fine-tuning finished. Saving final generator weights to H5 format.")
    final_generator = GeneratorRRDB(3, num_res_blocks=opt.n_rrdb_blocks)
    final_generator.load_state_dict(torch.load("saved_models/generator_final_best.pth"))
    save_weights_as_h5(final_generator, "saved_models/generator_final_best.h5")

if __name__ == '__main__':
    main()