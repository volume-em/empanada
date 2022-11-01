import numpy as np
import networkx as nx
from empanada.array_utils import merge_boxes, merge_rles
from empanada.inference.matcher import connect_chunk_boundaries

__all__ = [
    'global_instance_graph',
    'add_instance_edges',
    'merge_graph_instances',
    'remove_small_objects',
    'create_forward_map',
    'relabel_chunk_rles'
]

def extend_dict(dict1, dict2):
    for k,v in dict2.items():
        if k in dict1:
            dict1[k].extend(v)
        else:
            dict1[k] = v

def calculate_global_box(
    local_box, 
    chunk_index, 
    chunk_shape, 
    chunk_dims
):
    offset = np.unravel_index(chunk_index, chunk_dims)
    
    # project the box into global space
    global_box = [
        s + c * chunk_shape[i % 3]
        for i, (s,c) in enumerate(zip(local_box, 2 * offset))
    ]
    
    return global_box

def global_instance_graph(
    chunks, 
    cuber, 
    rle_class, 
    initial_label=1
):
    graph = nx.Graph()
    chunk_instance_map = {}
    
    for chunk_idx, chunk_attrs in chunks.items():
        for label_id, label_attrs in chunk_attrs['rle'][rle_class].items():
            # add a global label node with info about
            # the source chunk and label
            area = label_attrs['runs'].sum()
            box = calculate_global_box(
                label_attrs['box'], chunk_idx, 
                cuber.cube_shape, cuber.chunk_dims
            )
            
            graph.add_node(
                initial_label, area=area, box=box,
                chunk_lookup={chunk_idx: [label_id]}
            )
            
            chunk_instance_map[chunk_idx] = \
            {label_id: initial_label} | chunk_instance_map.get(chunk_idx, {})
            
            initial_label += 1
            
    return graph, chunk_instance_map

def add_instance_edges(
    graph, 
    chunks, 
    chunk_instance_map, 
    cuber,
    class_id,
    iou_thr=0.1,
    area_thr=100
):
    pairs = [
        ('right', 'left'),
        ('bottom', 'top'),
        ('back', 'front')
    ]
    
    for chunk_index in chunks.keys():
        # add connections from each of 3 neighbor chunks
        for pair, nix in zip(pairs, cuber.find_neighbors(chunk_index)):
            if nix is None:
                continue
            
            a, b = pair
            edges = connect_chunk_boundaries(
                chunks[chunk_index]['boundaries'][a][class_id],
                chunks[nix]['boundaries'][b][class_id], 
                iou_thr, area_thr
            )
            # convert from local to global labels
            # and add edges to the graph
            for edge in edges:
                cl, nl = edge
                cnode = chunk_instance_map[chunk_index][cl]
                nnode = chunk_instance_map[nix][nl]
                graph.add_edge(cnode, nnode)
                
def merge_nodes(
    graph, 
    root, 
    node, 
    remove_node=True
):
    # merge the chunk lookup
    extend_dict(
        graph.nodes[root]['chunk_lookup'], 
        graph.nodes[node]['chunk_lookup']
    )
    
    # merge the bounding boxes
    graph.nodes[root]['box'] = merge_boxes(
        graph.nodes[root]['box'], graph.nodes[node]['box']
    )
    
    # merge the label areas
    graph.nodes[root]['area'] += graph.nodes[node]['area']
    
    if remove_node:
        graph.remove_node(node)
        
def merge_graph_instances(graph):
    # merge instances in each connected component
    # of the graph
    for comp in list(nx.connected_components(graph)):
        # sort so that the 
        comp = sorted(list(comp))
        root = comp[0]
        for node in comp[1:]:
            merge_nodes(graph, root, node, True)
            
def remove_small_objects(graph, min_size=1000):
    filtered = []
    for node in graph.nodes:
        area = graph.nodes[node]['area']
        if area < min_size:
            filtered.append(node)
    
    graph.remove_nodes_from(filtered)
    
def create_forward_map(graph):
    # merge connected components to the lowest label value
    # and store the correct mapping of labels for each chunk
    forward_map = {}
    for node in graph.nodes:
        for chunk_index, chunk_labels in graph.nodes[node]['chunk_lookup'].items():
            chunk_map = {cl: node for cl in chunk_labels}
            forward_map[chunk_index] = chunk_map | forward_map.get(chunk_index, {})
            
    return forward_map

def relabel_chunk_rles(chunks, class_id, forward_map):
    for chunk_index, chunk_attrs in chunks.items():
        relabeled = {}
        lookup_table = forward_map.get(chunk_index, {})
        
        for old, new in lookup_table.items():
            old_rle = chunk_attrs['rle'][class_id][old]
            if new in relabeled:
                merged_s, merged_r = merge_rles(
                    relabeled[new]['starts'], relabeled[new]['runs'],
                    old_rle['starts'], old_rle['runs']
                )

                relabeled[new]['starts'] = merged_s
                relabeled[new]['runs'] = merged_r
            else:
                relabeled[new] = old_rle

        chunk_attrs['rle'][class_id] = relabeled