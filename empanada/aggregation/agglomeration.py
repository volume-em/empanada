import math
import numpy as np
import networkx as nx
from empanada.array_utils import *

def get_adjacent_indices(indices, shape):
    d, h, w = shape
    hw = h * w
    
    # z, y, x adjacent indices
    zi = indices + hw
    yi = indices + w
    xi = indices + 1
        
    # if in last plane then no adjacent z
    zi[np.where((indices // hw) == (d - 1))] = -1
    
    # if in last row then no adjacent y
    yi[np.where((indices % hw) > (hw - w))] = -1
    
    # if in last column then no adjacent x
    xi[np.where((indices % w) == (w - 1))] = -1
    
    return zi, yi, xi

def create_rag(voxel_indices, vol_shape):
    # create region adjacency graph
    rag = nx.DiGraph()
    rag.add_nodes_from(voxel_indices)
    
    # get all z, y, x voxels adjacent to
    # each index in voxel_indices
    zadj, yadj, xadj = get_adjacent_indices(voxel_indices, vol_shape)
    
    for vi, zi, yi, xi in zip(voxel_indices, zadj, yadj, xadj):
        if zi in rag:
            rag.add_edge(vi, zi, is_boundary=False)
        if yi in rag:
            rag.add_edge(vi, yi, is_boundary=False)
        if xi in rag:
            rag.add_edge(vi, xi, is_boundary=False)
            
    return rag

def set_node_instance_ids(rag, node_ids, instance_id):
    for node_id in node_ids:
        if node_id in rag:
            rag.nodes[node_id]['instance_id'] = instance_id
    
def reset_node_instance_ids(rag):
    for node in rag.nodes:
        rag.nodes[node]['instance_id'] = None
        
def update_boundary(rag, tr_index):
    for node_id in rag.nodes:
        node_instance_id = rag.nodes[node_id]['instance_id']
        if node_instance_id is not None:
            # loop over the outgoing edges
            for neighbor_id in rag[node_id].keys():
                neighbor_instance_id = rag.nodes[neighbor_id]['instance_id']
                #print('node, neighbor id:', node_instance_id, neighbor_instance_id)
                # if any neighbor has a different
                # not None instance_id then we're on a boundary
                if neighbor_instance_id is not None and neighbor_instance_id != node_instance_id:
                    boundary_src = rag[node_id][neighbor_id].get('boundary_src')
                    if boundary_src is not None:
                        if boundary_src != tr_index:
                            rag[node_id][neighbor_id]['cut_edge'] = True
                    else:
                        rag[node_id][neighbor_id]['is_boundary'] = True
                        rag[node_id][neighbor_id]['boundary_src'] = tr_index

def ensemble_objects3d(
    object_trackers,
    vote_thr=0.5,
    num_expansions=1
):
    # read parameters
    class_id = object_trackers[0].class_id
    vol_shape = object_trackers[0].shape3d
    n_votes = len(object_trackers)
    vote_thr_count = math.ceil(n_votes * vote_thr)
    
    # all object trackers should be for the same object class
    assert all(tr.class_id == class_id for tr in object_trackers)
    
    # unpack the instances from each tracker
    # into arrays for labels, bounding boxes
    # and voxel locations
    tracker_indices = []
    object_labels = []
    object_boxes = []
    object_coords = []
    for tr_index, tr in enumerate(object_trackers):
        for instance_id, instance_attr in tr.instances.items():
            tracker_indices.append(tr_index)
            object_labels.append(int(instance_id))
            object_boxes.append(instance_attr['box'])
            object_coords.append(instance_attr['coords'])
            
    # store in arrays for convenient slicing
    tracker_indices = np.array(tracker_indices)
    object_labels = np.array(object_labels)
    object_boxes = np.array(object_boxes)
    
    # compute pairwise overlaps for all instance boxes
    # matches are where iou is nonzero
    box_ious = pairwise_box_iou3d(object_boxes)
    box_matches = np.array(box_ious.nonzero()).T
    
    # filter out matched boxes from the same tracker
    r1_match_tr = tracker_indices[box_matches[:, 0]]
    r2_match_tr = tracker_indices[box_matches[:, 1]]
    box_matches = box_matches[r1_match_tr != r2_match_tr]
    
    # order of items in pair doesn't matter
    # so remove symmetric matches
    box_matches = np.sort(box_matches, axis=-1)
    box_matches = np.unique(box_matches, axis=0)
    
    # create instance level graph
    # each node is an instance in an object tracker
    instance_graph = nx.Graph()
    for node_id in range(len(object_labels)):
        instance_graph.add_node(
            node_id, box=object_boxes[node_id], 
            coords=object_coords[node_id],
            tracker_id=tracker_indices[node_id]
        )
        
    # add edges where there is any overlap between
    # instances from different object trackers
    for r1, r2 in zip(*tuple(box_matches.T)):
        r1_indices = rle_decode(instance_graph.nodes[r1]['coords'])
        r2_indices = rle_decode(instance_graph.nodes[r2]['coords'])
        
        # add edge to graph
        pair_iou = indices_iou(r1_indices, r2_indices)
        if pair_iou > 1e-3:
            instance_graph.add_edge(r1, r2)
            
    # generate connected components in the instance graph
    instance_id = 1
    instances = {}
    for comp in nx.connected_components(instance_graph):
        print('component', comp)
        instance_trackers = np.array([
            instance_graph.nodes[node_id]['tracker_id'] 
            for node_id in comp
        ])
        # decode all instance run length encodings
        decoded_instances = [
            rle_decode(instance_graph.nodes[node_id]['coords']) 
            for node_id in comp
        ]
        
        # determine which indices have enough votes to pass threshold
        all_indices = np.concatenate(decoded_instances)
        merged_indices, votes = np.unique(all_indices, return_counts=True)
        seg_indices = merged_indices[votes >= vote_thr_count]
        votes = votes[votes >= votes_thr_count]
        
        # create region adjacency graph
        rag = create_rag(seg_indices, vol_shape)
        
        for node,v in zip(rag.nodes, votes):
            rag.nodes[node]['instance_id'] = None
            rag.nodes[node]['votes'] = v
        
        # group decoded instances by source instance tracker
        # in order to find the boundaries between instances
        # separately by tracker
        for tr_index in range(len(object_trackers)):
            instance_group = np.where(instance_trackers == tr_index)[0]
            
            # no boundaries when fewer than 2 instances
            if len(instance_group) < 2:
                continue
                
            for instance_id in instance_group:
                indices = decoded_instances[instance_id]
                
                # node_ids are indices in final segmentation
                node_ids = np.intersect1d(indices, seg_indices)
                
                # set all nodes in this instance to the correct id
                set_node_instance_ids(rag, node_ids, instance_id)
                
            # find boundaries between instances
            update_boundary(rag, tr_index)
            
            # reset all node instance ids back to None
            reset_node_instance_ids(rag)
            
        # with boundaries detected, let's find plausible cuts
        # convert the RAG to undirected graph
        rag = rag.to_undirected()
        
        for _ in range(num_expansions):
            # TODO: dump nodes/edges into queues to avoid
            # looping through all nodes more than once!
            
            boundary = [
                (edge, rag.edges[edge]['boundary_src']) for edge in rag.edges 
                if rag.edges[edge].get('boundary_src') is not None and
                rag.edges[edge].get('cut_edge') is None and
                (rag.nodes[edge[0]].get('used') is None or
                rag.nodes[edge[1]].get('used') is None)
            ]
            
            if not boundary:
                break
            
            boundary = sorted(boundary, key=lambda x: x[1])
            

            for bound in boundary:
                (u, v), boundary_src = bound

                if rag.nodes[u].get('used') is None:
                    for neighbor_edge in rag.edges(u):
                        neighbor_bsrc = rag.edges[neighbor_edge].get('boundary_src')
                        if neighbor_bsrc is None:
                            rag.edges[neighbor_edge]['boundary_src'] = boundary_src
                        elif neighbor_bsrc != boundary_src:
                            rag.edges[neighbor_edge]['cut_edge'] = True
                    
                
                if rag.nodes[v].get('used') is None:
                    for neighbor_edge in rag.edges(v):
                        neighbor_bsrc = rag.edges[neighbor_edge].get('boundary_src')
                        if neighbor_bsrc is None:
                            rag.edges[neighbor_edge]['boundary_src'] = boundary_src
                        elif neighbor_bsrc != boundary_src:
                            rag.edges[neighbor_edge]['cut_edge'] = True

                rag.nodes[u]['used'] = True
                rag.nodes[v]['used'] = True
                
                
        # STILL NEED SOME SIZE THRESHOLD
        # TO PREVENT BAD SPLITS!
                        
        return rag
        
        
        """
        
        # create a graph where each node is a voxel
        # in the group of instances; edges are weighted
        # by the number of votes for splitting or merging
        # at the given voxel
        voxel_graph = nx.Graph()
        all_coords = np.concatenate(
            [rle_decode(instance_graph.nodes[node_id]['coords']) for node_id in comp]
        )
        
        # add instances over vote thr
        comp = list(comp)
                
        if len(comp) >= vote_thr_count:
            # merge boxes and coords from nodes
            node0 = comp[0]
            merged_box = graph.nodes[node0]['box']
            for node_id in comp[1:]:
                merged_box = merge_boxes3d(merged_box, graph.nodes[node_id]['box'])
            
            
            
            instances[instance_id] = {}
            instances[instance_id]['box'] = tuple(map(lambda x: x.item(), merged_box))
            # N.B. merged coords are already sorted by np.unique
            # only keep voxels with enough votes
            instances[instance_id]['coords'] = rle_encode(
                merged_coords[votes >= vote_thr_count]
            )
            
            instance_id += 1
        """
    return rag
