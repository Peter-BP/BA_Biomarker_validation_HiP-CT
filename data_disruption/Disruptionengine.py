import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, zoom, binary_closing
from skimage.morphology import remove_small_objects, remove_small_holes

    
def clean(grid):

    grid = remove_small_objects(grid, min_size=64)
    grid = remove_small_holes(grid, area_threshold=64)
    return grid

# Erotion and dilation to simulate noise
def surface_noise(grid, strength):
    if strength <= 0: 
        return grid
    
    eroded = binary_erosion(grid)
    edge = grid ^ eroded
    
    noise = np.random.random(grid.shape) < (strength * 0.5)
    
    disrupted = grid.copy()
    disrupted[edge & noise] = False
    
    bumps = binary_dilation(grid) ^ grid
    disrupted[bumps & noise] = True
    
    return clean(disrupted)

def fragmentation(grid, strength):
    if strength <= 0: 
        return grid
    
    cutout_seeds = np.random.random(grid.shape) < (strength * 0.005)
    
    radius = int(2 + (strength * 4))
    struct = np.ones((radius, radius, radius))
    
    cutout_mask = binary_dilation(cutout_seeds, structure=struct)
    
    disrupted = grid.copy()
    disrupted[cutout_mask] = False
    
    return clean(disrupted)

def false_mergers(grid, strength):
    if strength <= 0: 
        return grid
    
    disrupted = grid.copy()
    iterations = int(1 + strength * 3)
    disrupted = binary_dilation(disrupted, iterations=iterations)
    
    return clean(disrupted)

def z_anisotropy(grid, strength):
    if strength <= 0: 
        return grid
    
    z_factor = 1.0 - (strength * 0.8)
    
    small = zoom(grid.astype(float), (z_factor, 1, 1), order=1) > 0.5
    recover_factor = grid.shape[0] / small.shape[0]
    disrupted = zoom(small.astype(float), (recover_factor, 1, 1), order=0) > 0.5
    
    if disrupted.shape != grid.shape:
        disrupted = disrupted[:grid.shape[0], :grid.shape[1], :grid.shape[2]]
        
    return clean(disrupted)
