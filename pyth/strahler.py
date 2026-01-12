import numpy as np
from collections import deque

def calculate_strahler_order(network):
    for seg in network.segments.values():
        seg.strahler_order = None

    node_outgoing_count = {nid: 0 for nid in network.nodes}
    node_incoming_segs = {nid: [] for nid in network.nodes}

    for seg in network.segments.values():
        if seg.upstream_node and seg.downstream_node:
            node_outgoing_count[seg.upstream_node.id] += 1
            node_incoming_segs[seg.downstream_node.id].append(seg)

    queue = deque()
    for seg in network.segments.values():
        if seg.downstream_node:
            outgoing_from_end = node_outgoing_count[seg.downstream_node.id]
            if outgoing_from_end == 0:
                seg.strahler_order = 1
                queue.append(seg)
        else:
            seg.strahler_order = 0

    while queue:
        daughter_seg = queue.popleft()
        
        junction_node = daughter_seg.upstream_node
        if not junction_node: 
            continue
        
        mothers = node_incoming_segs[junction_node.id]
        
        for mother_seg in mothers:
            if mother_seg.strahler_order is not None:
                continue
            
            siblings = []
            all_siblings_ready = True
            
            for pot_child in junction_node.connected_segments:
                if pot_child.upstream_node == junction_node:
                    siblings.append(pot_child)
                    if pot_child.strahler_order is None:
                        all_siblings_ready = False
                        break
            
            if all_siblings_ready and siblings:
                child_orders = [s.strahler_order for s in siblings]
                max_order = max(child_orders)
                
                if child_orders.count(max_order) >= 2:
                    mother_seg.strahler_order = max_order + 1
                else:
                    mother_seg.strahler_order = max_order
                
                queue.append(mother_seg)
