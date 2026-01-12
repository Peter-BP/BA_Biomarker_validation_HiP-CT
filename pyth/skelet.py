import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize
import edt
from pyth.TStructure import VesselNetwork, Node, Segment
from pyth.TStructure import orient_network_flow
from pyth.strahler import calculate_strahler_order
from skan import Skeleton

def refine_network_geometry(network, voxel_grid):
    R = 2
    
    valid_segs = [s for s in network.segments.values() if len(getattr(s, 'raw_points', [])) > 0]
    if not valid_segs:
        return network

    pts = np.vstack([s.raw_points for s in valid_segs])
    
    shift_acc = np.zeros_like(pts) 
    weight_acc = np.zeros(len(pts))

    for dz in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                offset = np.array([dx, dy, dz])
                
                sample_coords = pts + offset
                
                vals = map_coordinates(voxel_grid, sample_coords.T, order=1, mode='nearest')
                
                shift_acc += (offset * vals[:, None])
                weight_acc += vals

    mask = weight_acc > 1e-6
    
    pts[mask] += shift_acc[mask] / weight_acc[mask, None]

    cursor = 0
    for seg in valid_segs:
        n = len(seg.raw_points)
        seg.raw_points = pts[cursor : cursor + n]
        cursor += n

    return network

def simplify_network(vessel_network, min_node_dist):
    node_ids = list(vessel_network.nodes.keys())
    if not node_ids: 
        return
    
    coords = np.array([vessel_network.nodes[nid].coord for nid in node_ids])
    
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=min_node_dist)
    
    parent = {nid: nid for nid in node_ids}
    
    def find(i):
        if parent[i] == i: 
            return i
        parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    idx_to_id = {i: nid for i, nid in enumerate(node_ids)}
    
    for i, j in pairs:
        union(idx_to_id[i], idx_to_id[j])
        
    groups = {}
    for nid in node_ids:
        root = find(nid)
        if root not in groups: 
            groups[root] = []
        groups[root].append(nid)
        
    new_network = VesselNetwork()
    old_to_new_map = {}
    new_node_counter = 0
    
    for root_id, members in groups.items():
        member_coords = np.array([vessel_network.nodes[nid].coord for nid in members])
        centroid = np.mean(member_coords, axis=0)
        
        new_node = Node(new_node_counter, centroid)
        new_network.add_node(new_node)
        
        for old_id in members:
            old_to_new_map[old_id] = new_node
            
        new_node_counter += 1
        
    new_seg_counter = 0
    for old_seg in vessel_network.segments.values():
        new_start = old_to_new_map[old_seg.start_node.id]
        new_end = old_to_new_map[old_seg.end_node.id]
        
        if new_start.id == new_end.id:
            continue
            
        new_seg = Segment(new_seg_counter, new_start, new_end, old_seg.points, old_seg.radii)
        new_network.add_segment(new_seg)
        new_seg_counter += 1
    
    return new_network

def dissolve_degree_2_nodes(vessel_network):
    node_to_segs = {nid: [] for nid in vessel_network.nodes}
    for seg in vessel_network.segments.values():
        node_to_segs[seg.start_node.id].append(seg)
        node_to_segs[seg.end_node.id].append(seg)
    
    nodes_to_dissolve = []
    for nid, connected_segs in node_to_segs.items():
        if len(connected_segs) == 2:
            nodes_to_dissolve.append(nid)
            
    if not nodes_to_dissolve:
        return vessel_network

    active_nodes = set(vessel_network.nodes.keys())
    anchors = [nid for nid in active_nodes if nid not in nodes_to_dissolve]
    
    new_segments = []
    visited_segments = set()
    
    for anchor_id in anchors:
        start_node = vessel_network.nodes[anchor_id]
        starting_segs = node_to_segs[anchor_id]
        
        for seg in starting_segs:
            if seg.id in visited_segments:
                continue
            
            current_seg = seg
            path_points = []
            path_radii = []
            
            curr_node = start_node
            if current_seg.start_node.id == curr_node.id:
                next_node = current_seg.end_node
                pts = list(current_seg.points)
                rads = list(current_seg.radii)
            else:
                next_node = current_seg.start_node
                pts = list(current_seg.points)[::-1]
                rads = list(current_seg.radii)[::-1]
            
            path_points.extend(pts)
            path_radii.extend(rads)
            visited_segments.add(current_seg.id)
            
            while next_node.id in nodes_to_dissolve:
                connected = node_to_segs[next_node.id]
                next_seg = connected[0] if connected[0].id != current_seg.id else connected[1]
                
                if next_seg.id in visited_segments:
                    break
                
                visited_segments.add(next_seg.id)
                current_seg = next_seg
                
                if current_seg.start_node.id == next_node.id:
                    pts = list(current_seg.points)
                    rads = list(current_seg.radii)
                    target_node = current_seg.end_node
                else:
                    pts = list(current_seg.points)[::-1]
                    rads = list(current_seg.radii)[::-1]
                    target_node = current_seg.start_node
                
                path_points.extend(pts)
                path_radii.extend(rads)
                next_node = target_node
            
            end_node = next_node
            
            new_seg_id = len(new_segments)
            new_seg = Segment(new_seg_id, start_node, end_node, np.array(path_points), np.array(path_radii))
            new_segments.append(new_seg)

    clean_network = VesselNetwork()
    
    for anchor_id in anchors:
        old_node = vessel_network.nodes[anchor_id]
        old_node.connected_segments = [] 
        clean_network.add_node(old_node)
        
    for seg in new_segments:
        clean_network.add_segment(seg)
    
    return clean_network

def skeletonize_voxel_grid(voxel_grid, min_node_dist=1.5):
    dist_map = edt.edt(voxel_grid, parallel=4)
    skeleton_grid = skeletonize(voxel_grid, method='lee')
    
    skel_obj = Skeleton(skeleton_grid)
    
    raw_network = VesselNetwork()
    node_map = {}
    seg_id_counter = 0
    
    for i in range(skel_obj.n_paths):
        path_coords = skel_obj.path_coordinates(i).astype(int)
        
        if len(path_coords) < 2: 
            continue

        start_coord = tuple(path_coords[0])
        end_coord = tuple(path_coords[-1])

        if start_coord not in node_map:
            start_node = Node(len(node_map), start_coord)
            node_map[start_coord] = start_node
            raw_network.add_node(start_node)
        
        if end_coord not in node_map:
            end_node = Node(len(node_map), end_coord)
            node_map[end_coord] = end_node
            raw_network.add_node(end_node)

        radii = dist_map[path_coords[:, 0], path_coords[:, 1], path_coords[:, 2]]

        start_node = node_map[start_coord]
        end_node = node_map[end_coord]
        
        seg = Segment(seg_id_counter, start_node, end_node, path_coords, radii)
        raw_network.add_segment(seg)
        seg_id_counter += 1

    simplefied_network = simplify_network(raw_network, min_node_dist=min_node_dist)
    final_network = dissolve_degree_2_nodes(simplefied_network)

    refine_network_geometry(final_network, dist_map)
    orient_network_flow(final_network)
    calculate_strahler_order(final_network)

    return final_network, dist_map
