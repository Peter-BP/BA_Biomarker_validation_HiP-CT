import os
import glob
import numpy as np
from tifffile import imread
from scipy.ndimage import zoom, binary_dilation, binary_closing

def load_kidney_labels_robust(path_to_data, target_label_value=None, scale=0.5, use_cache=True, cache_folder="processed_cache"):
    data_name = os.path.basename(os.path.dirname(os.path.normpath(path_to_data)))
    label_str = f"L{target_label_value}" if target_label_value is not None else "LAll"
    
    if use_cache:
        os.makedirs(cache_folder, exist_ok=True)
        cache_filename = f"{data_name}_{label_str}_scale{scale}.npz"
        cache_path = os.path.join(cache_folder, cache_filename)

        if os.path.exists(cache_path):
            return np.load(cache_path)['data']
    
    if os.path.isdir(path_to_data):
        tifs = glob.glob(os.path.join(path_to_data, "*.tif*"))
        pngs = glob.glob(os.path.join(path_to_data, "*.png"))
        file_list = sorted(tifs + pngs)

        if not file_list:
            raise ValueError(f"No files found in {path_to_data}")
        
        sample = imread(file_list[0])
        h, w = sample.shape[:2]
        full_volume = np.zeros((len(file_list), h, w), dtype=sample.dtype)
        
        for i, fname in enumerate(file_list):
            img = imread(fname)
            
            if img.ndim == 3:
                channels = img.shape[2]
                if channels >= 3:
                    img = np.max(img[:, :, 0:3], axis=2)
                else:
                    img = img[:, :, 0]
            
            full_volume[i] = img
    else:
        full_volume = imread(path_to_data)
    
    if target_label_value is not None:
        binary_grid = (full_volume == target_label_value)
    else:
        binary_grid = (full_volume > 0)

    del full_volume 

    if scale == 0 or scale == 1.0:
        final_volume = binary_grid
    else:
        thickened = binary_dilation(binary_grid, iterations=1)
        thickened = binary_closing(thickened, iterations=1)
        final_volume = zoom(thickened, scale, order=0)

    if use_cache:
        save_path = cache_path if cache_path.endswith('.npz') else cache_path.replace('.npy', '.npz')
        np.savez_compressed(save_path, data=final_volume)

    return final_volume
