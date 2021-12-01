import math
import numpy as np
import networkx as nx
from empanada.array_utils import *

from tqdm import tqdm

__all__ = ['merge_objects3d']

def merge_objects3d(object_trackers, vote_thr=0.5, min_iou=0.1):
    vol_shape = object_trackers[0].shape3d
    n_votes = len(object_trackers)
    vote_thr_count = math.ceil(n_votes * vote_thr)
    
    # unpack the instances from each tracker
    # into arrays for labels, bounding boxes
    # and voxel locations
    tracker_indices = []
    object_labels = []
    object_boxes = []
    object_starts = []
    object_runs = []
    for tr_index, tr in enumerate(object_trackers):
        for instance_id, instance_attr in tr.instances.items():
            tracker_indices.append(tr_index)
            object_labels.append(int(instance_id))
            object_boxes.append(instance_attr['box'])
            object_starts.append(instance_attr['starts'])
            object_runs.append(instance_attr['runs'])
            
    # store in arrays for convenient slicing
    tracker_indices = np.array(tracker_indices)
    object_labels = np.array(object_labels)
    object_boxes = np.array(object_boxes)
    
    # compute pairwise overlaps for all instance boxes
    # TODO: replace pairwise iou calculation with something
    # more memory efficient (only matters when N is large)
    box_ious = pairwise_box_iou3d(object_boxes)
    #box_matches = np.array(box_ious.nonzero()).T
    box_matches = np.array(
        np.where(box_ious > 0.01)
    ).T
    
    # filter out boxes from the same tracker
    r1_match_tr = tracker_indices[box_matches[:, 0]]
    r2_match_tr = tracker_indices[box_matches[:, 1]]
    box_matches = box_matches[r1_match_tr != r2_match_tr]
    
    # order of items in pair doesn't matter,
    # remove duplicates from symmetric matrix
    box_matches = np.sort(box_matches, axis=-1)
    box_matches = np.unique(box_matches, axis=0)
    
    # create graph with nodes
    graph = nx.Graph()
    for node_id in range(len(object_labels)):
        graph.add_node(
            node_id, box=object_boxes[node_id], 
            starts=object_starts[node_id],
            runs=object_runs[node_id]
        )
        
    # iou to weighted edges
    for r1, r2 in tqdm(zip(*tuple(box_matches.T)), total=len(box_matches)):
        pair_iou = rle_iou(
            graph.nodes[r1]['starts'], graph.nodes[r1]['runs'],
            graph.nodes[r2]['starts'], graph.nodes[r2]['runs']
        )
        
        if pair_iou > min_iou:
            graph.add_edge(r1, r2, iou=pair_iou)
            
    # generate connected components that
    # correspond to single instances
    instance_id = 1
    instances = {}
    for comp in nx.connected_components(graph):
        # add instances over vote thr
        comp = list(comp)
                
        if len(comp) >= vote_thr_count:
            # merge boxes and coords from nodes
            node0 = comp[0]
            merged_box = graph.nodes[node0]['box']
            for node_id in comp[1:]:
                merged_box = merge_boxes3d(merged_box, graph.nodes[node_id]['box'])
            
            # vote on indices that should belong to an object
            all_ranges = np.concatenate([
                np.stack([graph.nodes[node_id]['starts'], graph.nodes[node_id]['starts'] + graph.nodes[node_id]['runs']], axis=1) 
                for node_id in comp
            ])
            sort_idx = np.argsort(all_ranges[:, 0], kind='stable')
            all_ranges = all_ranges[sort_idx]
            voted_ranges = np.array(rle_voting(all_ranges, vote_thr_count))
            
            instances[instance_id] = {}
            instances[instance_id]['box'] = tuple(map(lambda x: x.item(), merged_box))
            
            # TODO: technically this could break the zarr_fill_instances function
            # if an object has a pixel in the bottom right corner of the Nth z slice
            # and a pixel in the top left corner of the N+1th z slice
            # TODO: split disconnected components
            instances[instance_id]['starts'] = voted_ranges[:, 0]
            instances[instance_id]['runs'] = voted_ranges[:, 1] - voted_ranges[:, 0]
            instance_id += 1
            
    return instances
