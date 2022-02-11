import math
import numpy as np
import networkx as nx
from itertools import combinations
from empanada.array_utils import *

from tqdm import tqdm

def average_edge_between_clusters(G, cluster1, cluster2, key='iou'):
    weights = []

    # get pairwise edge weights
    for node1 in cluster1:
        for node2 in cluster2:
            weights.append(
                G[node1][node2][key] if G.has_edge(node1, node2) else 0
            )
            
    return sum(weights) / len(weights)

def create_graph_of_clusters(G, iou_threshold, min_cluster_iou, min_overlap_area):
    # create new graph with low iou edges dropped
    H = G.copy()
    for (u, v, d) in G.edges(data=True):
        if d['iou'] <= iou_threshold:
            H.remove_edge(u, v)
        
    # each cluster is a connected component in the new graph
    cluster_graph = nx.Graph()
    for i,cluster in enumerate(nx.connected_components(H)):
        cluster_graph.add_node(i, cluster=cluster)
        
    # add edges weighted by average edge weight
    # in the non-cluster graph
    for node1,node2 in combinations(cluster_graph.nodes, 2):
        cluster1 = cluster_graph.nodes[node1]['cluster']
        cluster2 = cluster_graph.nodes[node2]['cluster']
        
        # get edge weights
        iou_weight = average_edge_between_clusters(G, cluster1, cluster2, 'iou')
        overlap_weight = average_edge_between_clusters(G, cluster1, cluster2, 'overlap')

        if iou_weight > min_cluster_iou or overlap_weight > min_overlap_area:
            cluster_graph.add_edge(node1, node2, iou=iou_weight, overlap=overlap_weight)
            
    return cluster_graph

def push_cluster(G, src, dst):
    src_cluster = G.nodes[src]['cluster']
    G.nodes[dst]['cluster'] = G.nodes[dst]['cluster'].union(src_cluster)
    G.remove_edge(src, dst)
    
    return G

def merge_clusters(G):
    # copy to avoid inplace changes
    H = G.copy()

    while len(H.edges()) > 0:
        # most connected from sorted nodes by the number of neighbors
        most_connected = sorted(
            H.nodes, key=lambda x: len(list(H.neighbors(x))), reverse=True
        )[0]
        
        # sort neighbors by the size of their clusters
        neighbors = sorted(
            H.neighbors(most_connected), 
            key=lambda x: len(H.nodes[x]['cluster']), 
            reverse=True
        )
        
        # decide whether to push the most connected cluster to
        # merge with its neighbors or to merge all the neighbors
        # into the most connected cluster
        most_connected_cluster = H.nodes[most_connected]['cluster']
        
        # if a neighbor has a bigger cluster then push most connected
        push_most_connected = len(H.nodes[neighbors[0]]['cluster']) > len(most_connected_cluster)
                
        if push_most_connected:
            # most connected cluster is rejected as an instance
            for neighbor in neighbors:
                push_cluster(H, most_connected, neighbor)
                
            H.remove_node(most_connected)
        else:
            # most connected cluster is accepted as an instance
            # pull all the neighboring clusters 
            for neighbor in neighbors:
                push_cluster(H, neighbor, most_connected)

                # push secondary neighbors to most connected node
                second_neighbors = list(H.neighbors(neighbor))
                for sn in second_neighbors:
                    if not H.has_edge(most_connected, sn):
                        edge_iou = H[neighbor][sn]['iou']
                        H.add_edge(most_connected, neighbor, iou=edge_iou)

                H.remove_node(neighbor)
            
    return H

def merge_instances(instances_dict):
    """
    Merge arbitrary number of instances. From dict
    of instance_id and instance_attrs.
    """
    if len(instances_dict) < 2:
        return list(instances_dict.values())[0]
    
    merged_box, merged_starts, merged_runs = None, None, None
    for instance_attrs in instances_dict.values():
        if merged_box is None:
            merged_box = instance_attrs['box']
            merged_starts = instance_attrs['starts']
            merged_runs = instance_attrs['runs']
        else:
            merged_box = merge_boxes(merged_box, instance_attrs['box'])
            merged_starts, merged_runs = merge_rles(
                merged_starts, merged_runs, 
                instance_attrs['starts'], instance_attrs['runs']
            )
            
    return dict(box=merged_box, starts=merged_starts, runs=merged_runs)

def merge_overlapping(cluster_instances, min_iou=0.1, min_overlap_area=100):
    # only applies when more than 1 instance in a cluster
    if len(cluster_instances) < 2:
        return list(cluster_instances.values())
    
    # resolve overlaps between cluster instances
    instance_ids = list(cluster_instances.keys())
    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(instance_ids)

    # measure intersection between all pairs of instances
    for c_i,c_j in combinations(instance_ids, 2):
        pair_iou, inter_area = rle_iou(
            cluster_instances[c_i]['starts'], cluster_instances[c_i]['runs'],
            cluster_instances[c_j]['starts'], cluster_instances[c_j]['runs'],
            return_intersection=True
        )

        if pair_iou > min_iou or inter_area > min_overlap_area:
            merge_graph.add_edge(c_i, c_j)

    merged_instances = []
    for comp in nx.connected_components(merge_graph):
        comp_instances = {k: v for k,v in cluster_instances.items() if k in comp}
        merged_instances.append(merge_instances(comp_instances))
            
    return merged_instances

def merge_objects3d(
    object_trackers, 
    pixel_vote_thr=0.5, 
    cluster_iou_thr=0.75,
    min_overlap_area=100,
    min_iou=0.1
):
    vol_shape = object_trackers[0].shape3d
    n_votes = len(object_trackers)
    pixel_vote_thr_count = math.ceil(n_votes * pixel_vote_thr)
    
    # always need to be majority to avoid overlapping
    # objects in the final segmentation
    min_cluster_size = (n_votes // 2) + 1
    
    # if pixel vote thr is less than 0.5
    # then we have to overmerge clusters
    # or segments might overlap
    min_cluster_iou = min_iou if pixel_vote_thr_count >= min_cluster_size else 0
    min_cluster_overlap_area = min_overlap_area if pixel_vote_thr_count >= min_cluster_size else 0
    
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
    
    # compute pairwise overlaps for all distance boxes
    # TODO: replace pairwise intersection calculation with something
    # more memory efficient (only matters when N is large)
    box_ious, box_intersections = box_iou(object_boxes, return_intersection=True)
    box_matches = np.array(
        np.where(np.logical_or(box_intersections > 100, box_ious > 0.01))
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
        
    # iou as weighted edges
    for r1, r2 in zip(*tuple(box_matches.T)):
        pair_iou, inter_area = rle_iou(
            graph.nodes[r1]['starts'], graph.nodes[r1]['runs'],
            graph.nodes[r2]['starts'], graph.nodes[r2]['runs'],
            return_intersection=True
        )
        
        if pair_iou > min_iou or inter_area > min_overlap_area:
            graph.add_edge(r1, r2, iou=pair_iou, overlap=inter_area)
            
    instance_id = 1
    instances = {}
    for comp in nx.connected_components(graph):
        if len(comp) < min_cluster_size:
            continue
        
        cluster_graph = create_graph_of_clusters(
            graph.subgraph(comp), cluster_iou_thr, min_cluster_iou, min_cluster_overlap_area
        )
        cluster_graph = merge_clusters(cluster_graph)
        
        cluster_id = 1
        cluster_instances = {}
        for node in cluster_graph.nodes:
            cluster = list(cluster_graph.nodes[node]['cluster'])
                
            if len(cluster) < min_cluster_size:
                continue
                
            # merge boxes and coords from nodes
            node0 = cluster[0]
            merged_box = graph.nodes[node0]['box']
            for node_id in cluster[1:]:
                merged_box = merge_boxes(merged_box, graph.nodes[node_id]['box'])
            
            # vote on indices that should belong to an object
            all_ranges = np.concatenate([
                np.stack([graph.nodes[node_id]['starts'], graph.nodes[node_id]['starts'] + graph.nodes[node_id]['runs']], axis=1) 
                for node_id in cluster
            ])
            
            sort_idx = np.argsort(all_ranges[:, 0], kind='stable')
            all_ranges = all_ranges[sort_idx]
            voted_ranges = np.array(rle_voting(all_ranges, pixel_vote_thr_count))
            
            if len(voted_ranges) > 0:
                cluster_instances[cluster_id] = {}
                cluster_instances[cluster_id]['box'] = tuple(map(lambda x: x.item(), merged_box))

                cluster_instances[cluster_id]['starts'] = voted_ranges[:, 0]
                cluster_instances[cluster_id]['runs'] = voted_ranges[:, 1] - voted_ranges[:, 0]
                
                cluster_id += 1
        
        # merge together instances with higher than trivial overlap
        for instance_attrs in merge_overlapping(cluster_instances, min_iou, min_overlap_area):
            instances[instance_id] = instance_attrs
            instance_id += 1
            
    return instances