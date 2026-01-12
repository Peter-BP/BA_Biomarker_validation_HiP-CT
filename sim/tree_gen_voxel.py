import numpy as np
import concurrent.futures

def voxelize_tree_fast(vessels, voxel_size, padding):
    
    class MicroSeg:
        def __init__(self, p1, p2, r):
            self.start_point = p1
            self.end_point = p2
            self.radius = r
    
    micro_segments = []
    for v in vessels:
        if hasattr(v, 'centerline_points'):
            points = v.centerline_points
            for i in range(len(points) - 1):
                micro_segments.append(MicroSeg(points[i], points[i+1], v.radius))
        else:
            micro_segments.append(MicroSeg(v.start_point, v.end_point, v.radius))
            
    target_list = micro_segments 
    
    min_coords = np.min([np.minimum(seg.start_point, seg.end_point) - seg.radius for seg in target_list], axis=0) - padding
    max_coords = np.max([np.maximum(seg.start_point, seg.end_point) + seg.radius for seg in target_list], axis=0) + padding
    
    bounds_min = min_coords
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    voxel_grid = np.zeros(grid_dims, dtype=bool)

    def get_vessel_mask(v):
        v_min = np.minimum(v.start_point, v.end_point) - v.radius
        v_max = np.maximum(v.start_point, v.end_point) + v.radius
        
        start_idx = np.floor((v_min - bounds_min) / voxel_size).astype(int)
        end_idx = np.ceil((v_max - bounds_min) / voxel_size).astype(int)
        
        start_idx = np.maximum(0, start_idx)
        end_idx = np.minimum(grid_dims, end_idx)
        
        if np.any(start_idx >= end_idx):
            return None

        x_vals = np.arange(start_idx[0], end_idx[0]) * voxel_size + bounds_min[0]
        y_vals = np.arange(start_idx[1], end_idx[1]) * voxel_size + bounds_min[1]
        z_vals = np.arange(start_idx[2], end_idx[2]) * voxel_size + bounds_min[2]
        
        GX, GY, GZ = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        
        A = v.start_point
        B = v.end_point
        AB = B - A
        len_sq = np.dot(AB, AB)
        
        if len_sq == 0:
            dists_sq = (GX - A[0])**2 + (GY - A[1])**2 + (GZ - A[2])**2
        else:
            dot = (GX - A[0])*AB[0] + (GY - A[1])*AB[1] + (GZ - A[2])*AB[2]
            t = np.clip(dot / len_sq, 0.0, 1.0)
            
            C_x = A[0] + t * AB[0]
            C_y = A[1] + t * AB[1]
            C_z = A[2] + t * AB[2]
            
            dists_sq = (GX - C_x)**2 + (GY - C_y)**2 + (GZ - C_z)**2

        local_mask = dists_sq <= (v.radius ** 2)
        
        slices = (slice(start_idx[0], end_idx[0]), 
                  slice(start_idx[1], end_idx[1]), 
                  slice(start_idx[2], end_idx[2]))
        
        return slices, local_mask

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_vessel = {executor.submit(get_vessel_mask, v): v for v in target_list}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_vessel)):
            result = future.result()
            if result is not None:
                slices, local_mask = result
                voxel_grid[slices] |= local_mask

    return voxel_grid, bounds_min
