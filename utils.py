# utils.py
import h5py
import torch
import rasterio
import numpy as np

def save_weights_as_h5(model, filepath):
    """Saves model weights from a PyTorch state_dict to an H5 file."""
    print(f"Saving best generator weights to {filepath}...")
    with h5py.File(filepath, 'w') as h5f:
        for key, value in model.state_dict().items():
            h5f.create_dataset(key, data=value.cpu().numpy())
    print(f"Successfully saved weights to {filepath}")

def save_scientific_tif(tensor, reference_tif_path, output_tif_path):
    """
    Saves the tensor as a float32 GeoTIFF with original metadata for analysis.
    This version is for use with GIS software.
    """
    try:
        # Clamp tensor to a valid range to prevent saving non-finite values
        tensor = torch.clamp(tensor, -1.0, 1.0)
        image_data = tensor.detach().cpu().numpy().astype('float32')
        
        # Get metadata from the original HR TIF file
        with rasterio.open(reference_tif_path) as src:
            meta = src.meta.copy()

        # Update metadata to match the output tensor's properties
        meta.update({
            "driver": "GTiff",
            "height": image_data.shape[1],
            "width": image_data.shape[2],
            "count": image_data.shape[0],
            "dtype": image_data.dtype
        })

        # Write the new TIF file
        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(image_data)
            
    except Exception as e:
        print(f"WARNING: Could not save scientific TIF file {output_tif_path}. Error: {e}")

def save_visual_tif(tensor, output_tif_path):
    """
    Saves the tensor as a uint8 TIF, suitable for any standard image viewer.
    This version is for quick visual inspection.
    """
    try:
        # Normalize tensor from [-1, 1] to [0, 255] and convert to 8-bit integer
        tensor = torch.clamp(tensor, -1.0, 1.0)
        image_data = ((tensor + 1) / 2.0 * 255).byte()
        image_data = image_data.detach().cpu().numpy()
        
        # Create minimal metadata for a visual TIF (no geographic info)
        meta = {
            "driver": "GTiff",
            "height": image_data.shape[1],
            "width": image_data.shape[2],
            "count": image_data.shape[0],
            "dtype": 'uint8'
        }
        
        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(image_data)
            
    except Exception as e:
        print(f"WARNING: Could not save visual TIF file {output_tif_path}. Error: {e}")