import pyvista as pv
import numpy as np

def plot_tree(vessels, color_by='strahler'):
   
    all_points = []
    lines = []
    point_radii = []
    cell_strahler = []
    cell_radii = []
    
    current_idx = 0
    
    for v in vessels:
        pts = getattr(v, 'centerline_points', None)
        if pts is None:
            pts = np.array([v.start_point, v.end_point])
            
        if not np.all(np.isfinite(pts)):
            continue
            
        n_pts = len(pts)
        all_points.append(pts)
        
        segment_indices = np.arange(current_idx, current_idx + n_pts)
        lines.append(np.concatenate(([n_pts], segment_indices)))
        
        point_radii.append(np.full(n_pts, v.radius))
        
        s_order = getattr(v, 'strahler_order', 1)
        cell_strahler.append(s_order if s_order is not None else 1)
        cell_radii.append(v.radius)
        
        current_idx += n_pts

    v_points = np.vstack(all_points)
    v_lines = np.hstack(lines).astype(np.int64)
    
    mesh = pv.PolyData(v_points, lines=v_lines)
    mesh.point_data["Radius"] = np.concatenate(point_radii)
    mesh.cell_data["Strahler"] = np.array(cell_strahler, dtype=np.int32)
    mesh.cell_data["Radius_Value"] = np.array(cell_radii)

    tube_mesh = mesh.tube(scalars="Radius", absolute=True, radius_factor=1.0)

    plotter = pv.Plotter()
    plotter.background_color = 'white'
    
    if color_by.lower() == 'strahler':
        plotter.add_mesh(
            tube_mesh, 
            scalars="Strahler", 
            cmap="turbo", 
            categories=True,
            smooth_shading=True,
            scalar_bar_args={'title': 'Strahler Order'}
        )
    else:
        plotter.add_mesh(
            tube_mesh, 
            scalars="Radius_Value", 
            cmap="jet", 
            smooth_shading=True,
            scalar_bar_args={'title': 'Radius'}
        )
        
    plotter.add_axes()
    plotter.reset_camera()
    plotter.show()
