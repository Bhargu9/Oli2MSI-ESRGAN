# utils.py
import h5py
import torch
import rasterio
import numpy as np

def save_weights_as_h5(model, filepath):
    print(f"Saving best generator weights to {filepath}...")
    with h5py.File(filepath, 'w') as h5f:
        for key, value in model.state_dict().items():
            h5f.create_dataset(key, data=value.cpu().numpy())
    print(f"Successfully saved weights to {filepath}")

def save_scientific_tif(tensor, reference_tif_path, output_tif_path):
    """
    Saves the tensor as a float32 GeoTIFF with original metadata for analysis.
    """
    try:
        tensor = torch.clamp(tensor, -1.0, 1.0)
        image_data = tensor.detach().cpu().numpy().astype('float32')
        
        with rasterio.open(reference_tif_path) as src:
            meta = src.meta.copy()
            
        meta.update({
            "driver": "GTiff",
            "height": image_data.shape[1],
            "width": image_data.shape[2],
            "count": image_data.shape[0],
            "dtype": image_data.dtype
        })

        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(image_data)
            
    except Exception as e:
        print(f"WARNING: Could not save scientific TIF file {output_tif_path}. Error: {e}")

def save_visual_tif(tensor, output_tif_path):
    """
    Saves the tensor as a uint8 TIF, suitable for any standard image viewer.
    """
    try:
        tensor = torch.clamp(tensor, -1.0, 1.0)
        image_data = ((tensor + 1) / 2.0 * 255).byte()
        image_data = image_data.detach().cpu().numpy()
        
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
