import math
import numpy as np
import networkx as nx
from empanada.inference.array_utils import *

from tqdm import tqdm

def calculate_clique_ious(G, clique1, clique2):
    all_ious = []
    for node1 in clique1:
        for node2 in clique2:
            if G.has_edge(node1, node2):
                all_ious.append(G[node1][node2]['iou'])
            else:
                # iou too small to have an edge, so it's 0
                all_ious.append(0.)
            
    return sum(all_ious) / len(all_ious)

def create_clique_graph(G, iou_threshold, min_clique_iou=0.1):
    # get a list of edges to drop from the graph
    drop_edges = []
    for (u, v, d) in G.edges(data=True):
        if d['iou'] < iou_threshold:
            drop_edges.append((u, v))
    
    # create a new graph with edges removed
    H = G.copy()
    for edge in drop_edges:
        H.remove_edge(*edge)
        
    # make each connected component (i.e. clique)
    # in H a node in a new graph
    clique_graph = nx.Graph()
    for i,clique in enumerate(nx.connected_components(H)):
        clique_graph.add_node(i, clique=clique)
        
    # edge weights are average IoUs between pairs within
    # separate cliques
    clique_nodes = list(clique_graph.nodes)
    for i,node1 in enumerate(clique_nodes):
        for j,node2 in enumerate(clique_nodes[i+1:]):
            clique1 = clique_graph.nodes[node1]['clique']
            clique2 = clique_graph.nodes[node2]['clique']
            clique_iou = calculate_clique_ious(G, clique1, clique2)
            if clique_iou >= min_clique_iou:
                clique_graph.add_edge(node1, node2, iou=clique_iou)
            
    return clique_graph

def pull_clique(G, src, dst):
    # merge clique from src to dst
    # and delete edge from the graph
    src_clique = G.nodes[src]['clique']
    G.nodes[dst]['clique'] = G.nodes[dst]['clique'].union(src_clique)
    G.remove_edge(src, dst)
    
    return G

def merge_cliques(G):
    H = G.copy()
    count = 0
    while len(H.edges()) > 0:
        # sorted nodes by the number of neighbors
        most_connected = sorted(
            H.nodes, key=lambda x: len(list(H.neighbors(x))), reverse=True
        )[0]
        
        # get neighbors of the most connected node
        neighbors = list(H.neighbors(most_connected))
        
        # sort neighbors by the size of their cliques
        neighbors = sorted(
            neighbors, key=lambda x: len(H.nodes[x]['clique']), reverse=True
        )
        
        most_connected_clique = H.nodes[most_connected]['clique']
        is_pushed = False
        for neighbor in neighbors:
            if len(H.nodes[neighbor]['clique']) > len(most_connected_clique):
                pull_clique(H, most_connected, neighbor)
                is_pushed = True
            elif is_pushed:
                pull_clique(H, most_connected, neighbor)
            else:
                break
                
        if is_pushed:
            H.remove_node(most_connected)
        else:
            # push to neighbors with larger cliques
            neighbors = list(H.neighbors(most_connected))
            neighbors = sorted(
                neighbors, key=lambda x: len(H.nodes[x]['clique'])
            )
            # pull from neighbors with smaller or equal cliques
            for neighbor in neighbors:
                most_connected_clique = H.nodes[most_connected]['clique']
                if len(H.nodes[neighbor]['clique']) <= len(most_connected_clique):
                    pull_clique(H, neighbor, most_connected)
                    H.remove_node(neighbor)
                    
        count += 1
        
        if count > 1000:
            print('Clique merging while loop stuck!', H.nodes(data=True), H.edges(data=True))
            break
            
    return H

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
        
        if len(comp) < vote_thr_count:
            continue
        
        sg = graph.subgraph(comp)
        clique_graph = create_clique_graph(sg, 0.75)
        clique_graph = merge_cliques(clique_graph)
                
        for node in clique_graph.nodes:
            clique = list(clique_graph.nodes[node]['clique'])
                
            if len(clique) < vote_thr_count:
                continue
                
            # merge boxes and coords from nodes
            node0 = clique[0]
            merged_box = graph.nodes[node0]['box']
            for node_id in clique[1:]:
                merged_box = merge_boxes3d(merged_box, graph.nodes[node_id]['box'])
            
            # vote on indices that should belong to an object
            all_ranges = np.concatenate([
                np.stack([graph.nodes[node_id]['starts'], graph.nodes[node_id]['starts'] + graph.nodes[node_id]['runs']], axis=1) 
                for node_id in clique
            ])
            sort_idx = np.argsort(all_ranges[:, 0], kind='stable')
            all_ranges = all_ranges[sort_idx]
            voted_ranges = np.array(rle_voting(all_ranges, vote_thr_count))
            
            instances[instance_id] = {}
            instances[instance_id]['box'] = tuple(map(lambda x: x.item(), merged_box))
            
            instances[instance_id]['starts'] = voted_ranges[:, 0]
            instances[instance_id]['runs'] = voted_ranges[:, 1] - voted_ranges[:, 0]
            instance_id += 1
            
    return instances
