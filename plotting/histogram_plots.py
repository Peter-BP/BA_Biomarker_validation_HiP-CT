import sys
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from sim.tree_gen import generate_renal_tree
from sim.tree_gen_config import *
from sim.tree_gen_voxel import voxelize_tree_fast

from pyth.dataload import load_kidney_labels_robust
from pyth.skelet import skeletonize_voxel_grid

from data_disruption.Disruptionengine import surface_noise, fragmentation, false_mergers, z_anisotropy

# Very generic plotting functions...

def plot_histogram(ax, data, target_mean, target_std, title, xlabel, xlim=None, bins=None, binwidth=None):
        
    data = np.array(data)
    measured_mean = np.mean(data)
    
    histplot_kwargs = {}
    if binwidth is not None:
        histplot_kwargs['binwidth'] = binwidth
    elif bins is not None:
        histplot_kwargs['bins'] = bins
    
    sns.histplot(data, kde=True, ax=ax, color="steelblue", alpha=0.6, stat="density", 
                 label='Measured', linewidth=3.0, **histplot_kwargs)
    
    if xlim:
        x_range = np.linspace(xlim[0], xlim[1], 200)
    else:
        data_min, data_max = data.min(), data.max()
        padding = (data_max - data_min) * 0.1
        x_range = np.linspace(data_min - padding, data_max + padding, 200)
    
    theoretical_pdf = stats.norm.pdf(x_range, loc=target_mean, scale=target_std)
    ax.plot(x_range, theoretical_pdf, 'g-', linewidth=4.0, alpha=0.9, label='Theoretical')
    
    ax.axvline(measured_mean, color='blue', linestyle='--', linewidth=3.5, 
               label=f'μ={measured_mean:.2f}')
    
    ax.axvline(target_mean, color='red', linestyle='-', linewidth=4.5, 
               label=f'Target={target_mean:.2f}')
    
    ax.set_title(title, fontweight='bold', fontsize=20, pad=12)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=14, width=1.5, length=6)
    
    legend = ax.legend(fontsize=14, loc='upper right', framealpha=1.0, 
                       edgecolor='black', fancybox=False, borderpad=1.0,
                       handlelength=2.0, handletextpad=0.8)
    legend.get_frame().set_linewidth(1.5)
    
    if xlim:
        ax.set_xlim(xlim)

def validate_skeletonized_pipeline(n_trees=10, max_generations=6, initial_radius=6, 
                                   voxel_size=0.5, enable_tortuosity=False):
    all_radius_ratios = []
    all_length_ratios = []
    all_bifurcation_angles = []
    all_murray_ratios = []
    all_tortuosity_dm = []
    all_tortuosity_soam = []
    
    for i in range(n_trees):
        print(f"Processing Tree {i+1}/{n_trees}...")
        
        vessels, _, _ = generate_renal_tree(
            max_generations=max_generations,
            initial_radius=initial_radius,
            enable_tortuosity=enable_tortuosity
        )
        
        voxel_grid, bounds_min = voxelize_tree_fast(vessels, voxel_size=voxel_size, padding=5)
        network, dist_map = skeletonize_voxel_grid(voxel_grid)
        
        if network is None or len(network.segments) == 0:
            continue

        for seg in network.segments.values():
            if seg.upstream_node is None:
                continue
            
            parent_seg = None
            if seg.upstream_node:
                for other_seg in seg.upstream_node.connected_segments:
                    if other_seg.downstream_node and other_seg.downstream_node.id == seg.upstream_node.id:
                        parent_seg = other_seg
                        break
            
            if parent_seg is not None:
                real_length = seg.length * voxel_size
                parent_length = parent_seg.length * voxel_size
                if parent_length > 1e-6:
                    l_ratio = real_length / parent_length
                    all_length_ratios.append(l_ratio)
                
                real_radius = seg.avg_radius * voxel_size
                parent_radius = parent_seg.avg_radius * voxel_size
                if parent_radius > 1e-6:
                    r_ratio = real_radius / parent_radius
                    all_radius_ratios.append(r_ratio)
            
            tort_dm = seg.calculate_tortuosity_dm()
            if tort_dm > 0:
                all_tortuosity_dm.append(tort_dm)
            
            tort_soam = seg.calculate_tortuosity_soam()
            if tort_soam >= 0:
                all_tortuosity_soam.append(tort_soam)
        
        for node_id, node in network.nodes.items():
            incoming = []
            outgoing = []
            
            for seg in node.connected_segments:
                if seg.downstream_node and seg.downstream_node.id == node_id:
                    incoming.append(seg)
                elif seg.upstream_node and seg.upstream_node.id == node_id:
                    outgoing.append(seg)
            
            if len(incoming) != 1 or len(outgoing) != 2:
                continue
            
            mother = incoming[0]
            d_sorted = sorted(outgoing, key=lambda s: s.avg_radius, reverse=True)
            d_major = d_sorted[0]
            d_minor = d_sorted[1]
            
            murray_ratio = (mother.avg_radius**3) / ((d_major.avg_radius**3) + (d_minor.avg_radius**3))
            all_murray_ratios.append(murray_ratio)
            

            vec_d1 = d_major.get_tangent_vector(node.id)
            vec_d2 = d_minor.get_tangent_vector(node.id)
            dot_prod = np.dot(vec_d1, vec_d2)
            angle_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
            all_bifurcation_angles.append(angle_deg)
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Skeletonized Pipeline Validation (N={n_trees} trees)", fontsize=20, y=0.98)
            
    plot_histogram(axes[0, 0], all_radius_ratios, 
                   RADIUS_RATIO_MEAN, RADIUS_RATIO_STD,
                   "A. Radius Ratio (r_daughter / r_parent)", 
                   "Ratio", xlim=(0.4, 1.2))
    
    plot_histogram(axes[0, 1], all_length_ratios,
                   LENGTH_RATIO_MEAN, LENGTH_RATIO_STD,
                   "B. Length Ratio (l_daughter / l_parent)",
                   "Ratio", xlim=(0, 1.5), binwidth=0.02)
    
    plot_histogram(axes[0, 2], all_bifurcation_angles,
                   BRANCH_ANGLE_MEAN_DEG * 2, BRANCH_ANGLE_STD_DEG * 2,
                   "C. Bifurcation Angle",
                   "Degrees", xlim=(0, 150))
    
    plot_histogram(axes[1, 0], all_murray_ratios,
                   1.0, 0.15,
                   "D. Murray's Law Ratio (r³_p / Σr³_d)",
                   "Ratio", xlim=(0, 2.0), binwidth=0.02)
    
    plot_histogram(axes[1, 1], all_tortuosity_dm,
                   1.05, 0.03,
                   "E. Tortuosity (Distance Metric)",
                   "Arc / Chord", xlim=(1.0, 1.2), binwidth=0.005)
    
    plot_histogram(axes[1, 2], all_tortuosity_soam,
                   20.0, 10.0,
                   "F. Tortuosity (SOAM)",
                   "Deg / Unit Length")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("skeletonized_pipeline_validation.png", dpi=150, bbox_inches='tight')
    plt.show()


def validate_vector_tree_config(n_trees=10, max_generations=6, initial_radius=6, enable_tortuosity=True):
    all_radius_ratios = []
    all_length_ratios = []
    all_bifurcation_angles = []
    all_murray_ratios = []
    all_tortuosity_dm = []
    all_tortuosity_soam = []
    
    for i in range(n_trees):
        vessels, bif_angles, murray_data = generate_renal_tree(
            max_generations=max_generations,
            initial_radius=initial_radius,
            enable_tortuosity=enable_tortuosity
        )
        
        all_bifurcation_angles.extend(bif_angles)
        
        for (r_parent, r_d1, r_d2) in murray_data:
            murray = (r_parent ** 3) / (r_d1 ** 3 + r_d2 ** 3)
            all_murray_ratios.append(murray)
            all_radius_ratios.append(r_d1 / r_parent)
            all_radius_ratios.append(r_d2 / r_parent)
        
        for v in vessels:
            if v.parent is not None:
                l_ratio = v.length / v.parent.length
                all_length_ratios.append(l_ratio)
                
                if hasattr(v, 'centerline_points') and len(v.centerline_points) > 1:
                    diffs = np.diff(v.centerline_points, axis=0)
                    arc_length = np.sum(np.linalg.norm(diffs, axis=1))
                else:
                    arc_length = v.length
                    
                chord = np.linalg.norm(v.end_point - v.start_point)
                if chord > 1e-6:
                    tort_dm = arc_length / chord
                    all_tortuosity_dm.append(tort_dm)
                
                if hasattr(v, 'centerline_points') and len(v.centerline_points) > 2:
                    diffs = np.diff(v.centerline_points, axis=0)
                    norms = np.linalg.norm(diffs, axis=1)
                    valid = norms > 1e-9
                    if np.sum(valid) > 1:
                        tangents = diffs[valid] / norms[valid, None]
                        if len(tangents) > 1:
                            dots = np.einsum('ij,ij->i', tangents[:-1], tangents[1:])
                            angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
                            soam = np.sum(angles) / arc_length if arc_length > 0 else 0
                            all_tortuosity_soam.append(soam)
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Vector Tree Validation (N={n_trees} trees)", fontsize=20, y=0.98)
    
    plot_histogram(axes[0, 0], all_radius_ratios, 
                   RADIUS_RATIO_MEAN, RADIUS_RATIO_STD,
                   "A. Radius Ratio (r_daughter / r_parent)", 
                   "Ratio", xlim=(0.4, 1.2))
    
    plot_histogram(axes[0, 1], all_length_ratios,
                   LENGTH_RATIO_MEAN, LENGTH_RATIO_STD,
                   "B. Length Ratio (l_daughter / l_parent)",
                   "Ratio", xlim=(0, 1.5))
    
    plot_histogram(axes[0, 2], all_bifurcation_angles,
                   BRANCH_ANGLE_MEAN_DEG * 2, BRANCH_ANGLE_STD_DEG * 2,
                   "C. Bifurcation Angle",
                   "Degrees", xlim=(0, 150))
    
    plot_histogram(axes[1, 0], all_murray_ratios,
                   1.0, 0.15,
                   "D. Murray's Law Ratio (r³_p / Σr³_d)",
                   "Ratio", xlim=(0, 2.0))
    
    plot_histogram(axes[1, 1], all_tortuosity_dm,
                   1.05, 0.03,
                   "E. Tortuosity (Distance Metric)",
                   "Arc / Chord", xlim=(1.0, 1.2))
    
    plot_histogram(axes[1, 2], all_tortuosity_soam,
                   20.0, 10.0,
                   "F. Tortuosity (SOAM)",
                   "Deg / Unit Length")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("vector_tree_config_validation.png", dpi=150, bbox_inches='tight')
    plt.show()



def plot_real_data_biomarkers(path_to_data, voxel_size_um=1.0, scale=1.0, 
                               target_label_value=None, use_cache=True):
    print("Loading data...")
    voxel_grid = load_kidney_labels_robust(path_to_data, 
                                             target_label_value=target_label_value,
                                             scale=scale, 
                                             use_cache=use_cache,
                                             cache_folder="C:/Users/Peter/Desktop/Bachelorprojekt/cokidney/BAcode/processed_cache")
    
    print("Running skeletonization...")
    network, dist_map = skeletonize_voxel_grid(voxel_grid)
    
    if network is None or len(network.segments) == 0:
        print("Skeletonization failed")
        return
    
    radius_ratios_by_order = {}
    length_ratios_by_order = {}
    tortuosity_dm_by_order = {}
    tortuosity_soam_by_order = {}
    murray_ratios_by_order = {}
    bifurcation_angles_by_order = {}
    segment_counts_by_order = {}
    
    all_radius_ratios = []
    all_length_ratios = []
    all_bifurcation_angles = []
    all_murray_ratios = []
    all_tortuosity_dm = []
    all_tortuosity_soam = []
    
    for seg in network.segments.values():
        if seg.upstream_node is None:
            continue
        
        strahler_order = getattr(seg, 'strahler_order', 0)
        if strahler_order not in segment_counts_by_order:
            segment_counts_by_order[strahler_order] = 0
            radius_ratios_by_order[strahler_order] = []
            length_ratios_by_order[strahler_order] = []
            tortuosity_dm_by_order[strahler_order] = []
            tortuosity_soam_by_order[strahler_order] = []
        
        segment_counts_by_order[strahler_order] += 1
        
        parent_seg = None
        if seg.upstream_node:
            for other_seg in seg.upstream_node.connected_segments:
                if other_seg.downstream_node and other_seg.downstream_node.id == seg.upstream_node.id:
                    parent_seg = other_seg
                    break
        
        if parent_seg is not None:
            real_length = seg.length * voxel_size_um
            parent_length = parent_seg.length * voxel_size_um
            if parent_length > 1e-6:
                l_ratio = real_length / parent_length
                all_length_ratios.append(l_ratio)
                length_ratios_by_order[strahler_order].append(l_ratio)
            
            real_radius = seg.avg_radius * voxel_size_um
            parent_radius = parent_seg.avg_radius * voxel_size_um
            if parent_radius > 1e-6:
                r_ratio = real_radius / parent_radius
                all_radius_ratios.append(r_ratio)
                radius_ratios_by_order[strahler_order].append(r_ratio)
        
        tort_dm = seg.calculate_tortuosity_dm()
        if tort_dm > 0:
            all_tortuosity_dm.append(tort_dm)
            tortuosity_dm_by_order[strahler_order].append(tort_dm)
        
        tort_soam = seg.calculate_tortuosity_soam()
        if tort_soam >= 0:
            all_tortuosity_soam.append(tort_soam)
            tortuosity_soam_by_order[strahler_order].append(tort_soam)
    
    for node_id, node in network.nodes.items():
        incoming = []
        outgoing = []
        
        for seg in node.connected_segments:
            if seg.downstream_node and seg.downstream_node.id == node_id:
                incoming.append(seg)
            elif seg.upstream_node and seg.upstream_node.id == node_id:
                outgoing.append(seg)
        
        if len(incoming) != 1 or len(outgoing) != 2:
            continue
        
        mother = incoming[0]
        strahler_order = getattr(mother, 'strahler_order', 0)
        
        if strahler_order not in murray_ratios_by_order:
            murray_ratios_by_order[strahler_order] = []
            bifurcation_angles_by_order[strahler_order] = []
        
        d_sorted = sorted(outgoing, key=lambda s: s.avg_radius, reverse=True)
        d_major = d_sorted[0]
        d_minor = d_sorted[1]
        
        murray_ratio = (mother.avg_radius**3) / ((d_major.avg_radius**3) + (d_minor.avg_radius**3))
        all_murray_ratios.append(murray_ratio)
        murray_ratios_by_order[strahler_order].append(murray_ratio)
        

        vec_d1 = d_major.get_tangent_vector(node.id)
        vec_d2 = d_minor.get_tangent_vector(node.id)
        dot_prod = np.dot(vec_d1, vec_d2)
        angle_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
        all_bifurcation_angles.append(angle_deg)
        bifurcation_angles_by_order[strahler_order].append(angle_deg)

    
    targets = {
        'radius_ratios': RADIUS_RATIO_MEAN,
        'length_ratios': LENGTH_RATIO_MEAN,
        'bifurcation_angles': BRANCH_ANGLE_MEAN_DEG * 2,
        'murray_ratios': 1.0,
        'tortuosity_dm': 1.05,
        'tortuosity_soam': 20.0
    }
    
    target_stds = {
        'radius_ratios': RADIUS_RATIO_STD,
        'length_ratios': LENGTH_RATIO_STD,
        'bifurcation_angles': BRANCH_ANGLE_STD_DEG * 2,
        'murray_ratios': 0.15,
        'tortuosity_dm': 0.03,
        'tortuosity_soam': 10.0
    }
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Real Kidney Data Biomarker Distributions", fontsize=20, y=0.98)
    
    plot_histogram(axes[0, 0], all_radius_ratios, 
                   targets['radius_ratios'], target_stds['radius_ratios'],
                   "A. Radius Ratio (r_daughter / r_parent)", 
                   "Ratio", xlim=(0.4, 1.2))
    
    plot_histogram(axes[0, 1], all_length_ratios,
                   targets['length_ratios'], target_stds['length_ratios'],
                   "B. Length Ratio (l_daughter / l_parent)",
                   "Ratio", xlim=(0, 2.0), binwidth=0.04)
    
    plot_histogram(axes[0, 2], all_bifurcation_angles,
                   targets['bifurcation_angles'], target_stds['bifurcation_angles'],
                   "C. Bifurcation Angle",
                   "Degrees", xlim=(0, 180))
    
    plot_histogram(axes[1, 0], all_murray_ratios,
                   targets['murray_ratios'], target_stds['murray_ratios'],
                   "D. Murray's Law Ratio (r³_p / Σr³_d)",
                   "Ratio", xlim=(0, 3.0))
    
    plot_histogram(axes[1, 1], all_tortuosity_dm,
                   targets['tortuosity_dm'], target_stds['tortuosity_dm'],
                   "E. Tortuosity (Distance Metric)",
                   "Arc / Chord", xlim=(1.0, 1.5))
    
    plot_histogram(axes[1, 2], all_tortuosity_soam,
                   targets['tortuosity_soam'], target_stds['tortuosity_soam'],
                   "F. Tortuosity (SOAM)",
                   "Deg / Unit Length")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("real_data_biomarker_distributions.png", dpi=150, bbox_inches='tight')
    plt.show()




def collect_biomarker_data(n_trees, max_generations, initial_radius, enable_tortuosity=False):
    all_radius_ratios = []
    all_length_ratios = []
    all_bifurcation_angles = []
    all_murray_ratios = []
    all_tortuosity_dm = []
    all_tortuosity_soam = []
    
    for i in range(n_trees):
        vessels, bif_angles, murray_data = generate_renal_tree(
            max_generations=max_generations,
            initial_radius=initial_radius,
            enable_tortuosity=enable_tortuosity
        )
        
        all_bifurcation_angles.extend(bif_angles)
        
        for (r_parent, r_d1, r_d2) in murray_data:
            murray = (r_parent ** 3) / (r_d1 ** 3 + r_d2 ** 3)
            all_murray_ratios.append(murray)
            all_radius_ratios.append(r_d1 / r_parent)
            all_radius_ratios.append(r_d2 / r_parent)
        
        for v in vessels:
            if v.parent is not None:
                l_ratio = v.length / v.parent.length
                all_length_ratios.append(l_ratio)
                
                if hasattr(v, 'centerline_points') and len(v.centerline_points) > 1:
                    diffs = np.diff(v.centerline_points, axis=0)
                    arc_length = np.sum(np.linalg.norm(diffs, axis=1))
                else:
                    arc_length = v.length
                    
                chord = np.linalg.norm(v.end_point - v.start_point)
                if chord > 1e-6:
                    tort_dm = arc_length / chord
                    all_tortuosity_dm.append(tort_dm)
                
                if hasattr(v, 'centerline_points') and len(v.centerline_points) > 2:
                    diffs = np.diff(v.centerline_points, axis=0)
                    norms = np.linalg.norm(diffs, axis=1)
                    valid = norms > 1e-9
                    if np.sum(valid) > 1:
                        tangents = diffs[valid] / norms[valid, None]
                        if len(tangents) > 1:
                            dots = np.einsum('ij,ij->i', tangents[:-1], tangents[1:])
                            angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
                            soam = np.sum(angles) / arc_length if arc_length > 0 else 0
                            all_tortuosity_soam.append(soam)
    
    return {
        'radius_ratios': all_radius_ratios,
        'length_ratios': all_length_ratios,
        'bifurcation_angles': all_bifurcation_angles,
        'murray_ratios': all_murray_ratios,
        'tortuosity_dm': all_tortuosity_dm,
        'tortuosity_soam': all_tortuosity_soam
    }




def collect_segment_data(network, disruption_name, strength, tree_id):
    rows = []
    for seg_id, seg in network.segments.items():
        strahler = getattr(seg, 'strahler_order', 1)
        rows.append({
            'Segment_ID': seg_id,
            'Disruption_Type': disruption_name,
            'Strength': strength,
            'Tree_ID': tree_id,
            'Strahler_Order': strahler,
            'Radius_um': seg.avg_radius,
            'Length_um': seg.length,
            'Tortuosity_DM': seg.calculate_tortuosity_dm(),
            'Tortuosity_SOAM': seg.calculate_tortuosity_soam(),
        })
    return rows


def collect_bifurcation_data(network, disruption_name, strength, tree_id):
    rows = []
    
    for node_id, node in network.nodes.items():
        if len(node.connected_segments) != 3:
            continue
        
        sorted_segs = sorted(node.connected_segments, key=lambda s: s.avg_radius, reverse=True)
        mother = sorted_segs[0]
        d1 = sorted_segs[1]
        d2 = sorted_segs[2]
        
        sum_daughter_cubed = d1.avg_radius**3 + d2.avg_radius**3
        murray_ratio = (mother.avg_radius**3) / sum_daughter_cubed if sum_daughter_cubed > 0 else np.nan
        
        v_d1 = d1.get_tangent_vector(node.id)
        v_d2 = d2.get_tangent_vector(node.id)
        cos_theta = np.clip(np.dot(v_d1, v_d2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))
        
        length_ratio = mother.length / d1.length if d1.length > 0 else np.nan
        strahler = getattr(mother, 'strahler_order', 1)
        
        rows.append({
            'Node_ID': node_id,
            'Disruption_Type': disruption_name,
            'Strength': strength,
            'Tree_ID': tree_id,
            'Strahler_Order': strahler,
            'Murray_Ratio': murray_ratio,
            'Bifurcation_Angle': angle_deg,
            'Length_Ratio': length_ratio,
            'Radius_Mother_um': mother.avg_radius,
        })
    
    return rows



def run_diagnostic_experiment_in_memory(n_trees=5, max_generations=7, initial_radius=8,
                                         voxel_size=0.5, padding=5, plot_results=True,
                                         use_tqdm=True):
    disruption_types = {
        "Surface_Noise": surface_noise,
        "Fragmentation": fragmentation,
        "False_Mergers": false_mergers,
        "Z_Anisotropy": z_anisotropy
    }
    
    strengths = np.linspace(0, 1.0, 6)
    
    all_seg_rows = []
    all_bif_rows = []
    
    for tree_idx in range(n_trees):
        print(f"[Tree {tree_idx+1}/{n_trees}]")
        
        vessels, _, _ = generate_renal_tree(max_generations=max_generations, 
                                            initial_radius=initial_radius)
        
        gt_grid, _ = voxelize_tree_fast(vessels, voxel_size=voxel_size, padding=padding)
        
        for name, func in disruption_types.items():
            if use_tqdm:
                iterator = tqdm(strengths, desc=f"  {name}", leave=False, 
                               position=0, file=sys.stdout)
            else:
                iterator = strengths
            
            for s in iterator:
                dist_grid = func(gt_grid.copy(), s)
                network, _ = skeletonize_voxel_grid(dist_grid)
                
                if network is None:
                    continue
                
                seg_rows = collect_segment_data(network, name, s, tree_idx)
                bif_rows = collect_bifurcation_data(network, name, s, tree_idx)
                
                all_seg_rows.extend(seg_rows)
                all_bif_rows.extend(bif_rows)
    
    df_seg = pd.DataFrame(all_seg_rows)
    df_bif = pd.DataFrame(all_bif_rows) if all_bif_rows else pd.DataFrame()
    
    if plot_results:
        plot_diagnostic_dashboard(df_seg, df_bif)
    
    return df_seg, df_bif


def plot_diagnostic_dashboard(df_seg, df_bif):
    sns.set_theme(style="whitegrid")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.linewidth': 1.2,
        'grid.linewidth': 1.0
    })
    
    seg_per_tree = df_seg.groupby(['Disruption_Type', 'Strength', 'Tree_ID']).agg({
        'Segment_ID': 'count',
        'Length_um': 'mean',
        'Radius_um': 'mean',
        'Tortuosity_SOAM': 'mean',
        'Tortuosity_DM': 'mean'
    }).rename(columns={'Segment_ID': 'Segment_Count'}).reset_index()
    
    if not df_bif.empty:
        bif_per_tree = df_bif.groupby(['Disruption_Type', 'Strength', 'Tree_ID']).agg({
            'Murray_Ratio': 'mean',
            'Bifurcation_Angle': 'mean',
            'Length_Ratio': 'mean'
        }).reset_index()
    else:
        bif_per_tree = pd.DataFrame()
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    fig.suptitle(f"Diagnostic Biomarker Sensitivity (N={df_seg['Tree_ID'].nunique()} Trees)", 
                 fontsize=18, y=0.98, fontweight='bold')
    
    def style_plot(ax, title, ylabel):
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel("Disruption Strength", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.0)
        ax.tick_params(labelsize=11, width=1.2, length=5)
        ax.legend(title="Disruption Type", loc='best', fontsize=10, 
                 title_fontsize=11, framealpha=0.95)
    
    if not bif_per_tree.empty and 'Murray_Ratio' in bif_per_tree.columns:
        sns.lineplot(data=bif_per_tree, x='Strength', y='Murray_Ratio', 
                     hue='Disruption_Type', marker='o', ax=axes[0, 0],
                     linewidth=2.5, markersize=9, markeredgewidth=1.5)
        axes[0, 0].axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
        style_plot(axes[0, 0], "Murray's Law Ratio", "Ratio (r_m³ / Σr_d³)")
    else:
        axes[0, 0].text(0.5, 0.5, "No bifurcation data", ha='center', va='center', 
                       transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title("Murray's Law Ratio", fontsize=15, fontweight='bold')
    
    if not bif_per_tree.empty and 'Bifurcation_Angle' in bif_per_tree.columns:
        sns.lineplot(data=bif_per_tree, x='Strength', y='Bifurcation_Angle', 
                     hue='Disruption_Type', marker='o', ax=axes[0, 1],
                     linewidth=2.5, markersize=9, markeredgewidth=1.5)
        axes[0, 1].axhline(75, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
        style_plot(axes[0, 1], "Bifurcation Angle", "Angle (degrees)")
    else:
        axes[0, 1].text(0.5, 0.5, "No bifurcation data", ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title("Bifurcation Angle", fontsize=15, fontweight='bold')
    
    if not bif_per_tree.empty and 'Length_Ratio' in bif_per_tree.columns:
        sns.lineplot(data=bif_per_tree, x='Strength', y='Length_Ratio', 
                     hue='Disruption_Type', marker='o', ax=axes[1, 0],
                     linewidth=2.5, markersize=9, markeredgewidth=1.5)
        style_plot(axes[1, 0], "Length Ratio (Mother/Daughter)", "Ratio")
    else:
        axes[1, 0].text(0.5, 0.5, "No bifurcation data", ha='center', va='center', 
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title("Length Ratio", fontsize=15, fontweight='bold')
    
    sns.lineplot(data=seg_per_tree, x='Strength', y='Tortuosity_DM', 
                 hue='Disruption_Type', marker='o', ax=axes[1, 1],
                 linewidth=2.5, markersize=9, markeredgewidth=1.5)
    axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
    style_plot(axes[1, 1], "Tortuosity (Distance Metric)", "Arc Length / Chord Length")
    
    sns.lineplot(data=seg_per_tree, x='Strength', y='Tortuosity_SOAM', 
                 hue='Disruption_Type', marker='o', ax=axes[2, 0],
                 linewidth=2.5, markersize=9, markeredgewidth=1.5)
    style_plot(axes[2, 0], "Tortuosity (Sum of Angles)", "Degrees per µm")
    
    sns.lineplot(data=seg_per_tree, x='Strength', y='Length_um', 
                 hue='Disruption_Type', marker='o', ax=axes[2, 1],
                 linewidth=2.5, markersize=9, markeredgewidth=1.5)
    style_plot(axes[2, 1], "Mean Segment Length", "Length (µm)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.5)
    plt.show()
    
    return fig
