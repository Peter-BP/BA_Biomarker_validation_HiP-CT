import numpy as np
from scipy.interpolate import splprep, splev, interp1d

# Structure inspired by the kaggle competitions skeletonization: https://github.com/HiPCTProject/Kaggle_skeleton_analyses

class Node:
    def __init__(self, node_id, coord):
        self.id = node_id
        self.coord = tuple(coord)
        self.connected_segments = [] 

class Segment:
    def __init__(self, seg_id, start_node, end_node, points, radii):
        self.id = seg_id
        self.start_node = start_node
        self.end_node = end_node
        
        self.raw_points = np.array(points, dtype=float) 
        raw_radii = np.array(radii, dtype=float)
        
        self.points, self.radii = self._process_geometry(
            self.raw_points, raw_radii, start_node.coord, end_node.coord
        )
        
        self.points_tight = self.points 
        self.points_smooth = self.points
        
        self.avg_radius = np.median(self.radii) if len(self.radii) > 0 else 1.0
        self.length = self._calculate_arc_length(self.points)
        
        if len(self.points) > 1:
            seg_lens = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
            mid_radii = (self.radii[:-1] + self.radii[1:]) / 2.0
            self.volume = np.sum(np.pi * (mid_radii ** 2) * seg_lens)
        else:
            self.volume = 0.0

        self.upstream_node = None   
        self.downstream_node = None 

    def _relax_points(self, points, iterations=1, strength=0.35):

        if len(points) < 3 or iterations < 1: 
            return points
    
        neighbor_weight = strength * 0.5
        center_weight = 1.0 - (2 * neighbor_weight)
    
        relaxed = np.copy(points)
        for _ in range(iterations):
            relaxed[1:-1] = (neighbor_weight * relaxed[:-2] + 
                         center_weight * relaxed[1:-1] + 
                         neighbor_weight * relaxed[2:])
        return relaxed

    def _process_geometry(self, points, radii, start_coord, end_coord):
        raw = np.array(points)
        relaxed = self._relax_points(raw, iterations=1) 
        
        dists = np.linalg.norm(np.diff(relaxed, axis=0), axis=1)
        total_len = np.sum(dists)
        
        num_samples = max(20, int(total_len * 6.0))
        
        if len(relaxed) > 3:
            tck, u = splprep(relaxed.T, s=0, k=3) 
            u_new = np.linspace(0, 1, num_samples)
            new_points = np.array(splev(u_new, tck)).T
            
            t_orig = np.linspace(0, 1, len(radii))
            new_radii = np.interp(u_new, t_orig, radii)
        else:
            new_points = relaxed
            new_radii = radii
            
        return new_points, new_radii

    def _calculate_arc_length(self, points):
        if len(points) < 2: 
            return 0.0
        diffs = np.diff(points, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def _get_point_at_distance(self, target_dist, from_start=True):
        points = self.points if from_start else self.points[::-1]
        current_dist = 0.0
        for i in range(len(points) - 1):
            seg_vec = points[i+1] - points[i]
            seg_len = np.linalg.norm(seg_vec)
            if current_dist + seg_len >= target_dist:
                ratio = (target_dist - current_dist) / seg_len
                return points[i] + seg_vec * ratio
            current_dist += seg_len
        return points[-1]

    def get_tangent_vector(self, at_node_id):
        if len(self.points) < 2: 
            return np.array([0,0,0])
        
        safe_skip = max(2.0, min(3.5, self.avg_radius * 1.0)) 
        measure = max(5.0, min(10.0, self.avg_radius * 3.0))
        
        is_start = (at_node_id == self.start_node.id)
        p_start = self._get_point_at_distance(safe_skip, from_start=is_start)
        p_end = self._get_point_at_distance(safe_skip + measure, from_start=is_start)
        
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
             vec = self.points[-1] - self.points[0] if is_start else self.points[0] - self.points[-1]
             norm = np.linalg.norm(vec)

        return vec / norm if norm != 0 else np.array([0,0,0])

    def calculate_tortuosity_dm(self):
        chord = np.linalg.norm(self.points[-1] - self.points[0])
        return self.length / chord if chord > 1e-6 else 1.0

    def calculate_tortuosity_soam(self):
        pts = self.points
        if len(pts) < 3: 
            return 0.0
        
        diffs = np.diff(pts, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        tangents = diffs / (norms + 1e-9)
        
        dots = np.einsum('ij,ij->i', tangents[:-1], tangents[1:])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        
        return np.sum(angles) / self.length


class VesselNetwork:
    def __init__(self):
        self.nodes = {}
        self.segments = {}

    def add_node(self, node):
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    def add_segment(self, segment):
        if segment.id not in self.segments:
            self.segments[segment.id] = segment
            if segment.start_node.id not in self.nodes:
                self.add_node(segment.start_node)
            if segment.end_node.id not in self.nodes:
                self.add_node(segment.end_node)
            if segment not in segment.start_node.connected_segments:
                segment.start_node.connected_segments.append(segment)
            if segment not in segment.end_node.connected_segments:
                segment.end_node.connected_segments.append(segment)


def get_connected_components(network):
    visited = set()
    components = []

    for start_node_id in network.nodes:
        if start_node_id in visited:
            continue

        component = set()
        stack = [start_node_id]
        
        while stack:
            node_id = stack.pop()
            if node_id in visited: 
                continue
            
            visited.add(node_id)
            component.add(node_id)
            
            node = network.nodes[node_id]
            for seg in node.connected_segments:
                neighbor = seg.end_node if seg.start_node.id == node_id else seg.start_node
                if neighbor.id not in visited:
                    stack.append(neighbor.id)
        
        components.append(component)
    
    return components


def orient_network_flow(network):
    components = get_connected_components(network)

    for comp_idx, component_nodes in enumerate(components):
        best_root = None
        max_radius = -1.0

        for node_id in component_nodes:
            node = network.nodes[node_id]
            if len(node.connected_segments) == 1:
                seg = node.connected_segments[0]
                if seg.avg_radius > max_radius:
                    max_radius = seg.avg_radius
                    best_root = node

        if best_root is None:
            continue

        queue = [best_root.id]
        visited_bfs = {best_root.id}

        while queue:
            curr_id = queue.pop(0)
            curr_node = network.nodes[curr_id]

            for seg in curr_node.connected_segments:
                neighbor = seg.end_node if seg.start_node.id == curr_id else seg.start_node
                
                if neighbor.id not in visited_bfs:
                    visited_bfs.add(neighbor.id)
                    queue.append(neighbor.id)
                    
                    seg.upstream_node = curr_node
                    seg.downstream_node = neighbor
