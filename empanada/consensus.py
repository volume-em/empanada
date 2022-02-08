import math
import numpy as np
import networkx as nx
from empanada.array_utils import *

from tqdm import tqdm

def average_cluster_ious(G, cluster1, cluster2):
    all_ious = []

    # loop over all nodes in the two clusters
    # to get pairwise IoU scores
    for node1 in cluster1:
        for node2 in cluster2:
            if G.has_edge(node1, node2):
                all_ious.append(G[node1][node2]['iou'])
            else:
                # iou too small to have an edge, so it's 0
                all_ious.append(0.)
            
    # average all iou scores for instance that had
    # overlap with each other across clusters
    return sum(all_ious) / len(all_ious)

def average_cluster_overlaps(G, cluster1, cluster2):
    all_overlaps = []

    # loop over all nodes in the two clusters
    # to get pairwise IoU scores
    for node1 in cluster1:
        for node2 in cluster2:
            if G.has_edge(node1, node2):
                all_overlaps.append(G[node1][node2]['overlap'])
            else:
                all_overlaps.append(0.)
            
    return sum(all_overlaps) / len(all_overlaps)

def create_graph_of_clusters(G, iou_threshold, min_cluster_iou, min_overlap_area):
    drop_edges = []
    for (u, v, d) in G.edges(data=True):
        if d['iou'] < iou_threshold:
            drop_edges.append((u, v))
    
    # create a new graph with edges removed
    H = G.copy()
    for edge in drop_edges:
        H.remove_edge(*edge)
        
    cluster_graph = nx.Graph()
    for i,cluster in enumerate(nx.connected_components(H)):
        cluster_graph.add_node(i, cluster=cluster)
        
    # edge weights are average IoUs between all pairs
    # of objects across separate clusters
    cluster_nodes = list(cluster_graph.nodes)
    for i,node1 in enumerate(cluster_nodes):
        for j,node2 in enumerate(cluster_nodes[i+1:]):
            cluster1 = cluster_graph.nodes[node1]['cluster']
            cluster2 = cluster_graph.nodes[node2]['cluster']
            cluster_iou = average_cluster_ious(G, cluster1, cluster2)
            cluster_overlap = average_cluster_overlaps(G, cluster1, cluster2)

            # only add an edge when clusters have average IoU
            # over a small threshold value
            if cluster_iou >= min_cluster_iou or cluster_overlap >= min_overlap_area:
                cluster_graph.add_edge(node1, node2, iou=cluster_iou)
            
    return cluster_graph

def push_cluster(G, src, dst):
    # push the src cluster to the destination cluster node
    # and remove the edge that connects them
    src_cluster = G.nodes[src]['cluster']
    G.nodes[dst]['cluster'] = G.nodes[dst]['cluster'].union(src_cluster)
    G.remove_edge(src, dst)
    
    return G

def merge_clusters(G):
    # copy to avoid inplace changes
    H = G.copy()

    count = 0
    # merge clusters until there are no
    # clusters with meaningful IoU between their
    # instances
    while len(H.edges()) > 0:
        # most connected from sorted nodes by the number of neighbors
        most_connected = sorted(
            H.nodes, key=lambda x: len(list(H.neighbors(x))), reverse=True
        )[0]
        
        # get neighbors of the most connected node
        neighbors = list(H.neighbors(most_connected))
        
        # sort neighbors by the size of their clusters
        neighbors = sorted(
            neighbors, key=lambda x: len(H.nodes[x]['cluster']), reverse=True
        )
        
        # decide whether to push the most connected cluster to
        # merge with its neighbors or to merge all the neighbors
        # into the most connected cluster
        most_connected_cluster = H.nodes[most_connected]['cluster']
        push_most_connected = False
        for neighbor in neighbors:
            # if neighbor has a bigger cluster than
            # the most connected cluster will be absorbed
            # by each of its neighbors
            if len(H.nodes[neighbor]['cluster']) > len(most_connected_cluster):
                push_cluster(H, most_connected, neighbor)
                push_most_connected = True
            elif push_most_connected:
                push_cluster(H, most_connected, neighbor)
            else:
                break
                
        if push_most_connected:
            # most connected cluster is rejected as an instance
            H.remove_node(most_connected)
        else:
            # most connected cluster is accepted as an instance
            # pull all the neighboring clusters 
            neighbors = list(H.neighbors(most_connected))
            neighbors = sorted(
                neighbors, key=lambda x: len(H.nodes[x]['cluster'])
            )
            # push from neighbors with smaller or equal clusters
            for neighbor in neighbors:
                most_connected_cluster = H.nodes[most_connected]['cluster']
                if len(H.nodes[neighbor]['cluster']) <= len(most_connected_cluster):
                    push_cluster(H, neighbor, most_connected)
                    
                    # push secondary neighbors to most connected node
                    second_neighbors = list(H.neighbors(neighbor))
                    for sn in second_neighbors:
                        if not H.has_edge(most_connected, sn):
                            edge_iou = H[neighbor][sn]['iou']
                            H.add_edge(most_connected, neighbor, iou=edge_iou)
                            
                    H.remove_node(neighbor)
                    
        count += 1
        if count > 100:
            raise Exception(f'Infinite loop in consensus cluster merging!')
            
    return H

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
    min_cluster_len = (n_votes // 2) + 1
    
    # if pixel vote thr is less than 0.5
    # then we have to overmerge clusters
    # or segments might overlap
    min_cluster_iou = min_iou if pixel_vote_thr > 0.5 else 0
    min_cluster_overlap_area = min_iou if pixel_vote_thr > 0.5 else 0
    
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
    # TODO: replace pairwise intersection calculation with something
    # more memory efficient (only matters when N is large)
    box_intersections = box_intersection(object_boxes)
    box_matches = np.array(
        np.where(box_intersections > min_overlap_area)
    ).T
    print('Box matches', len(box_matches))

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
    # components are groups of objects that
    # are all connected by each other
    new_graph = nx.Graph()
    for comp in nx.connected_components(graph):
        comp = list(comp)
        if len(comp) < min_cluster_len:
            continue
            
        sg = graph.subgraph(comp)
        cluster_graph = create_graph_of_clusters(sg, cluster_iou_thr, 0, 0)#min_iou, min_overlap_area)
        cluster_graph = merge_clusters(cluster_graph)
        
                
        for node in cluster_graph.nodes:
            cluster = list(cluster_graph.nodes[node]['cluster'])
            if 86 in comp:
                print(comp, cluster)
                
            if len(cluster) < min_cluster_len:
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
                instances[instance_id] = {}
                instances[instance_id]['box'] = tuple(map(lambda x: x.item(), merged_box))

                instances[instance_id]['starts'] = voted_ranges[:, 0]
                instances[instance_id]['runs'] = voted_ranges[:, 1] - voted_ranges[:, 0]
                
                #if instance_id in [62, 63]:
                #    print(instance_id, cluster, len(cluster), tuple(map(lambda x: x.item(), merged_box)))
                    

                instance_id += 1
            
    return instances
