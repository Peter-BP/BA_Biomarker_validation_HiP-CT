import sys
import pyvista as pv
import numpy as np

def plot_vessel_network(vessel_network, node_color='orange', node_size=5, segment_color='blue', segment_radius=0.7):
    plotter = pv.Plotter(window_size=(1000, 700))
    plotter.background_color = "white"

    node_coords = [node.coord for node in vessel_network.nodes.values()]
    
    if node_coords:
        node_cloud = pv.PolyData(np.array(node_coords))
        plotter.add_mesh(node_cloud, color=node_color, point_size=node_size, render_points_as_spheres=False)

    # For performance
    all_points = []
    lines_connectivity = []
    current_offset = 0

    for seg in vessel_network.segments.values():
        pts = seg.points
        n_pts = len(pts)
        
        if n_pts < 2: 
            continue
            
        all_points.extend(pts)

        segment_indices = list(range(current_offset, current_offset + n_pts))
        lines_connectivity.append(n_pts)
        lines_connectivity.extend(segment_indices)
        
        current_offset += n_pts

    if all_points:
        network_mesh = pv.PolyData(np.array(all_points))
        network_mesh.lines = np.array(lines_connectivity)
        tube = network_mesh.tube(radius=segment_radius)
        plotter.add_mesh(tube, color=segment_color)

    plotter.add_axes()
    
    sys.argv = [sys.argv[0]]
    plotter.show()


def plot_strahler_3d(vessel_network, radius=0.5, cmap="jet"):
    all_points = []
    lines_connectivity = []
    all_strahler_values = []
    current_offset = 0
    
    for seg in vessel_network.segments.values():
        pts = seg.points
        n_pts = len(pts)
        
        if n_pts < 2: 
            continue
            
        order = getattr(seg, 'strahler_order', 0)
        
        all_points.extend(pts)
        
        segment_indices = list(range(current_offset, current_offset + n_pts))
        lines_connectivity.append(n_pts)
        lines_connectivity.extend(segment_indices)
        current_offset += n_pts
        
        all_strahler_values.extend([order] * n_pts)

    if not all_points:
        return

    network_mesh = pv.PolyData(np.array(all_points))
    network_mesh.lines = np.array(lines_connectivity)
    network_mesh.point_data["Strahler Order"] = np.array(all_strahler_values)
    
    tube = network_mesh.tube(radius=radius)
    
    plotter = pv.Plotter()
    plotter.background_color = "white"
    
    plotter.add_mesh(
        tube, 
        scalars="Strahler Order",
        cmap=cmap, 
        smooth_shading=True,
        show_scalar_bar=True
    )
    
    plotter.add_axes()
    plotter.view_isometric()
    plotter.show()

def visualize_voxel_tree(voxel_grid, bounds_min, voxel_size, realdata=False):
    plotter = pv.Plotter(window_size=[800, 800])
    plotter.background_color = 'white'

    grid = pv.ImageData()
    grid.dimensions = np.array(voxel_grid.shape) + 1
    grid.origin = bounds_min
    if realdata:
        grid.dimensions = np.array(voxel_grid.shape[::-1]) + 1
    else:
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        
    if realdata:
        grid.cell_data["occupancy"] = voxel_grid.flatten().astype(np.uint8)
    else:
        grid.cell_data["occupancy"] = voxel_grid.flatten(order="F").astype(np.uint8)

    threshed = grid.threshold(0.5, scalars="occupancy")
    if threshed.n_cells == 0:
        return
    
    plotter.add_mesh(threshed, color='royalblue', opacity=0.7, show_edges=False)
    plotter.add_axes()

    old_args = sys.argv
    sys.argv = [sys.argv[0]]
    plotter.show()



def visualize_skeleton_overlay(voxel_grid, network, voxel_size=1.0):
    p = pv.Plotter()
    p.set_background("white")

    all_points = []
    lines_indices = []
    current_idx = 0
    
    for seg in network.segments.values():
        pts = getattr(seg, 'raw_points', getattr(seg, 'points', []))
        if len(pts) < 2: 
            continue
        
        pts_world = np.array(pts) * voxel_size
        all_points.append(pts_world)
        
        n_p = len(pts_world)
        segment_indices = [n_p] + list(range(current_idx, current_idx + n_p))
        lines_indices.extend(segment_indices)
        current_idx += n_p

    if all_points:
        cloud_points = np.vstack(all_points)
        cloud_lines = np.array(lines_indices)
        
        skeleton_mesh = pv.PolyData(cloud_points)
        skeleton_mesh.lines = cloud_lines
        
        tube = skeleton_mesh.tube(radius=voxel_size * 0.4)
        p.add_mesh(tube, color="blue", opacity=1.0, label="Skeleton")
    
    grid = pv.ImageData()
    grid.dimensions = np.array(voxel_grid.shape) + 1
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    
    grid.cell_data["values"] = voxel_grid.flatten(order="F")
    grid_points = grid.cell_data_to_point_data()
    vessel_surface = grid_points.contour([0.5]) 
    
    p.add_mesh(vessel_surface, color="red", opacity=0.15, smooth_shading=True, label="Segmentation")

    p.add_legend()
    p.show()
