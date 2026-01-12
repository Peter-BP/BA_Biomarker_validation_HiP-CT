import numpy as np
from sim.tree_gen_config import *
from sim.tree_gen_vessel import generate_daughter_vectors, Vessel

def assign_vector_strahler_order(vessels):
    parent_to_children = {v: [] for v in vessels}
    for v in vessels:
        if v.parent is not None:
            parent_to_children[v.parent].append(v)

    queue = [v for v in vessels if not parent_to_children[v]]
    for leaf in queue:
        leaf.strahler_order = 1
    
    processed = set(queue)
    while queue:
        current = queue.pop(0)
        p = current.parent
        
        if p is not None and p not in processed:
            children = parent_to_children[p]
            if all(c.strahler_order is not None for c in children):
                orders = [c.strahler_order for c in children]
                max_o = max(orders)
                p.strahler_order = max_o + 1 if orders.count(max_o) >= 2 else max_o
                queue.append(p)
                processed.add(p)

def generate_renal_tree(
    max_generations,
    initial_radius,
    radius_ratio_mean=RADIUS_RATIO_MEAN,
    radius_ratio_std=RADIUS_RATIO_STD,
    length_ratio_mean=LENGTH_RATIO_MEAN,
    length_ratio_std=LENGTH_RATIO_STD,
    branch_angle_mean_deg=BRANCH_ANGLE_MEAN_DEG,
    branch_angle_std_deg=BRANCH_ANGLE_STD_DEG,
    enable_tortuosity=True  
):
    
    initial_end_point = INITIAL_START_POINT + INITIAL_DIRECTION * INITIAL_LENGTH
    
    root_tort_amp = 1 if enable_tortuosity else 0.0
    root_vessel = Vessel(INITIAL_START_POINT, initial_end_point, initial_radius, tortuosity_amplitude=root_tort_amp)
    vessels = [root_vessel]
    parent_vessels = [root_vessel]
    bifurcation_angles = []
    murray_data = []

    for gen in range(max_generations):
        next_gen_parents = []
        
        if not parent_vessels:
            break
            
        for parent in parent_vessels:
            parent_radius = parent.radius
            parent_end = parent.end_point
            
            if parent_radius < 0.2:
                continue

            branch_half_angle_deg = abs(np.random.normal(branch_angle_mean_deg, branch_angle_std_deg))
            bifurcation_angles.append(branch_half_angle_deg * 2)
            branch_half_angle_rad = np.deg2rad(branch_half_angle_deg)
            
            d1_dir, d2_dir = generate_daughter_vectors(parent.direction, branch_half_angle_rad)

            r_ratio1 = abs(np.random.normal(radius_ratio_mean, radius_ratio_std))
            r_ratio2 = abs(np.random.normal(radius_ratio_mean, radius_ratio_std))
            dr1 = max(0.1, parent_radius * r_ratio1)
            dr2 = max(0.1, parent_radius * r_ratio2)
            
            murray_data.append((parent_radius, dr1, dr2))

            daughter_specs = [(d1_dir, dr1), (d2_dir, dr2)]

            for d_dir, d_radius in daughter_specs:
                l_ratio = abs(np.random.normal(length_ratio_mean, length_ratio_std))
                target_len = parent.length * l_ratio
                
                min_len_geometric = d_radius * 3.0
                final_length = max(target_len, min_len_geometric)
                
                d_end = parent_end + d_dir * final_length
                
                wiggle_amp = 0.5 if enable_tortuosity else 0.0

                daughter = Vessel(parent_end, d_end, d_radius, tortuosity_amplitude=wiggle_amp)
                daughter.parent = parent
                
                vessels.append(daughter)
                next_gen_parents.append(daughter)

        parent_vessels = next_gen_parents

    assign_vector_strahler_order(vessels)
    return vessels, bifurcation_angles, murray_data
