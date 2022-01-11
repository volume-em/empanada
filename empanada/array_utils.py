import numpy as np
import numba

def box_area(boxes):
    """Computes area of bounding boxes (N, 4) second dim format (y1, x1, y2, x2)"""
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    return height * width

def box_volume(boxes):
    """Computes area of bounding boxes (N, 6) second dim format (z1, y1, x1, z2, y2, x2)"""
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    return depth * height * width

def box_intersection2d(boxes1, boxes2):
    """Computes the pairwise intersection areas between two sets of boxes"""
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), all_pairs_min_ymax - all_pairs_max_ymin
    )
    
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), all_pairs_min_xmax - all_pairs_max_xmin
    )
    
    return intersect_heights * intersect_widths

def box_intersection3d(boxes1, boxes2):
    """Computes the pairwise intersection areas between two sets of boxes"""
    [z_min1, y_min1, x_min1, z_max1, y_max1, x_max1] = np.split(boxes1, 6, axis=1)
    [z_min2, y_min2, x_min2, z_max2, y_max2, x_max2] = np.split(boxes2, 6, axis=1)
    
    # find z coordinates of overlapping area
    all_pairs_min_zmax = np.minimum(z_max1, np.transpose(z_max2))
    all_pairs_max_zmin = np.maximum(z_min1, np.transpose(z_min2))
    intersect_depths = np.maximum(
        np.zeros(all_pairs_max_zmin.shape), 
        all_pairs_min_zmax - all_pairs_max_zmin
    )

    # find top and bottom coordinates of overlapping area
    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), 
        all_pairs_min_ymax - all_pairs_max_ymin
    )
    
    # find left and right coordinates of the overlapping area
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), 
        all_pairs_min_xmax - all_pairs_max_xmin
    )
    
    return intersect_depths * intersect_heights * intersect_widths

def pairwise_box_intersection2d(boxes):
    """
    Calculates the pairwise overlaps with a set of
    bounding boxes.
    
    Arguments:
    ----------
    boxes: Array of shape (n, 4). Where coordinates
    are (y1, x1, y2, x2).
    
    Returns:
    --------
    box_overlaps. Array of shape (n, n).
    
    """
    # separate boxes into coordinates arrays
    [y_min, x_min, y_max, x_max] = np.split(boxes, 4, axis=1)
    
    # find top and bottom coordinates of overlapping area
    all_pairs_min_ymax = np.minimum(y_max, np.transpose(y_max))
    all_pairs_max_ymin = np.maximum(y_min, np.transpose(y_min))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), 
        all_pairs_min_ymax - all_pairs_max_ymin
    )
    
    # find left and right coordinates of the overlapping area
    all_pairs_min_xmax = np.minimum(x_max, np.transpose(x_max))
    all_pairs_max_xmin = np.maximum(x_min, np.transpose(x_min))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), 
        all_pairs_min_xmax - all_pairs_max_xmin
    )
    
    return intersect_heights * intersect_widths

def pairwise_box_intersection3d(boxes):
    """
    Calculates the pairwise overlaps with a set of
    bounding boxes.
    
    Arguments:
    ----------
    boxes: Array of shape (n, 6). Where coordinates
    are (z1, y1, x1, z2, y2, x2).
    
    Returns:
    --------
    box_overlaps. Array of shape (n, n).
    
    """
    # separate boxes into coordinates arrays
    [z_min, y_min, x_min, z_max, y_max, x_max] = np.split(boxes, 6, axis=1)
    
    # find top and bottom coordinates of overlapping area
    all_pairs_min_zmax = np.minimum(z_max, np.transpose(z_max))
    all_pairs_max_zmin = np.maximum(z_min, np.transpose(z_min))
    intersect_depths = np.maximum(
        np.zeros(all_pairs_max_zmin.shape), 
        all_pairs_min_zmax - all_pairs_max_zmin
    )

    # find top and bottom coordinates of overlapping area
    all_pairs_min_ymax = np.minimum(y_max, np.transpose(y_max))
    all_pairs_max_ymin = np.maximum(y_min, np.transpose(y_min))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), 
        all_pairs_min_ymax - all_pairs_max_ymin
    )
    
    # find left and right coordinates of the overlapping area
    all_pairs_min_xmax = np.minimum(x_max, np.transpose(x_max))
    all_pairs_max_xmin = np.maximum(x_min, np.transpose(x_min))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), 
        all_pairs_min_xmax - all_pairs_max_xmin
    )
    
    return intersect_depths * intersect_heights * intersect_widths

def merge_boxes2d(box1, box2):
    """Finds the minimal enclosing box around two given boxes."""
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    return (
        min(ymin1, ymin2), min(xmin1, xmin2), 
        max(ymax1, ymax2), max(xmax1, xmax2)
    )

def merge_boxes3d(box1, box2):
    """Finds the minimal enclosing box around two given boxes."""
    zmin1, ymin1, xmin1, zmax1, ymax1, xmax1 = box1
    zmin2, ymin2, xmin2, zmax2, ymax2, xmax2 = box2
    return (
        min(zmin1, zmin2), 
        min(ymin1, ymin2), 
        min(xmin1, xmin2), 
        max(zmax1, zmax2), 
        max(ymax1, ymax2), 
        max(xmax1, xmax2)
    )

def box_iou2d(boxes1, boxes2):
    """Computes pairwise IoU between two sets of boxes."""
    intersect = box_intersection2d(boxes1, boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2[None, :] - intersect
    return intersect / union

def box_iou3d(boxes1, boxes2):
    """Computes pairwise IoU between two sets of boxes."""
    intersect = box_intersection3d(boxes1, boxes2)
    volume1 = box_volume(boxes1)
    volume2 = box_volume(boxes2)
    union = volume1[:, None] + volume2[None, :] - intersect
    return intersect / union

def pairwise_box_iou2d(boxes):
    """
    Calculates the pairwise intersection-over-union 
    within a set of bounding boxes.
    
    Arguments:
    ----------
    boxes: Array of shape (n, 4). Where coordinates
    are (y1, x1, y2, x2).
    
    Returns:
    --------
    box_ious. Array of shape (n, n).
    
    """
    intersect = pairwise_box_intersection2d(boxes) # (n, n)
    
    # union is the difference between the sum of 
    # areas and the intersection
    area = box_area(boxes)
    pairwise_area = area[:, None] + area[None, :] # (n, n)
    union = pairwise_area - intersect
    
    return intersect / (union + 1e-5)

def pairwise_box_iou3d(boxes):
    """
    Calculates the pairwise intersection-over-union 
    within a set of bounding boxes.
    
    Arguments:
    ----------
    boxes: Array of shape (n, 6). Where coordinates
    are (z1, y1, x1, z2, y2, x2).
    
    Returns:
    --------
    box_ious. Array of shape (n, n).
    
    """
    intersect = pairwise_box_intersection3d(boxes) # (n, n)
    
    # union is the difference between the sum of 
    # areas and the intersection
    volume = box_volume(boxes)
    pairwise_volume = volume[:, None] + volume[None, :] # (n, n)
    union = pairwise_volume - intersect
    
    return intersect / (union + 1e-5)

def rle_encode(indices):
    """Run length encodes sorted indices."""
    # where indices are not contiguous
    changes = np.where(indices[1:] != indices[:-1] + 1)[0] + 1
    
    # add first and last indices
    changes = np.insert(changes, 0, [0], axis=0)
    changes = np.append(changes, [len(indices)], axis=0)

    # measure distance between changes (i.e. run length)
    runs = changes[1:] - changes[:-1]
    
    # remove last change
    changes = changes[:-1]
    
    assert(len(changes) == len(runs))
    
    return indices[changes], runs
    #return ' '.join([f'{i} {r}' for i,r in zip(indices[changes], runs)])
    
def rle_decode(starts, runs):
    ends = starts + runs
    indices = np.concatenate(
        [np.arange(s, e) for s,e in zip(starts, ends)]
    )
    return indices

def rle_to_string(starts, runs):
    return ' '.join([f'{i} {r}' for i,r in zip(starts, runs)])

def string_to_rle(encoding):
    encoding = np.array([int(i) for i in encoding.split(' ')])
    starts, runs = encoding[::2], encoding[1::2]
    return starts, runs

def indices_iou(set1, set2):
    """Computes the IoU between two sets of sorted indices."""
    intersect = len(np.intersect1d(set1, set2, assume_unique=True))
    union = len(np.union1d(set1, set2))
    return intersect / union

def crop_and_binarize2d(mask, box, label):
    """Crops and binarizes mask within box with given label"""
    ymin, xmin, ymax, xmax = box
    return mask[ymin:ymax, xmin:xmax] == label

def crop_and_binarize3d(mask, box, label):
    """Crops and binarizes mask within box with given label"""
    zmin, ymin, xmin, zmax, ymax, xmax = box
    return mask[zmin:zmax, ymin:ymax, xmin:xmax] == label

def mask_iou(mask1, mask2):
    """Computes the IoU between two masks of the same shape"""
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(np.logical_or(mask1, mask2))
    return intersection / union

def mask_ioa(mask1, mask2):
    """Computes the IoA between two masks of the same shape"""
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    area = np.count_nonzero(mask2)
    return intersection / area

@numba.jit(nopython=True)
def intersection_from_ranges(merged_runs, changes):
    total_inter = 0
    
    check_run = None
    for is_change, run1, run2 in zip(changes, merged_runs[:-1], merged_runs[1:]):
        if is_change:
            check_run = run1
        elif check_run is None:
            continue

        if check_run[1] < run2[0]:
            continue
        
        total_inter += min(check_run[1], run2[1]) - max(check_run[0], run2[0])   

    return total_inter

def rle_ioa(starts_a, runs_a, starts_b, runs_b):
    # convert from runs to ends
    ranges_a = np.stack([starts_a, starts_a + runs_a], axis=1)
    ranges_b = np.stack([starts_b, starts_b + runs_b], axis=1)
    
    merged_runs = np.concatenate([ranges_a, ranges_b], axis=0)
    merged_ids = np.concatenate(
        [np.repeat([0], len(ranges_a)), np.repeat([1], len(ranges_b))]
    )
    sort_indices = np.argsort(merged_runs, axis=0, kind='stable')[:, 0]
    
    merged_runs = merged_runs[sort_indices]
    merged_ids = merged_ids[sort_indices]
    changes = merged_ids[:-1] != merged_ids[1:]
    
    # calculate intersection and divide by area
    intersection = intersection_from_ranges(merged_runs, changes)
    area = runs_b.sum()
    
    return intersection / area

def rle_iou(starts_a, runs_a, starts_b, runs_b):
    # convert from runs to ends
    ranges_a = np.stack([starts_a, starts_a + runs_a], axis=1)
    ranges_b = np.stack([starts_b, starts_b + runs_b], axis=1)
    
    merged_runs = np.concatenate([ranges_a, ranges_b], axis=0)
    merged_ids = np.concatenate(
        [np.repeat([0], len(ranges_a)), np.repeat([1], len(ranges_b))]
    )
    sort_indices = np.argsort(merged_runs, axis=0, kind='stable')[:, 0]
    
    merged_runs = merged_runs[sort_indices]
    merged_ids = merged_ids[sort_indices]
    changes = merged_ids[:-1] != merged_ids[1:]
    
    # calculate intersection and divide by union
    intersection = intersection_from_ranges(merged_runs, changes)
    union = runs_a.sum() + runs_b.sum() - intersection
    
    return intersection / union

@numba.jit(nopython=True)
def split_range_by_votes(running_range, num_votes, vote_thr=2):
    # the running range may be split at places with
    # too few votes to cross the vote_thr
    split_voted_ranges = []
    s, e = None, None
    for ix in range(len(num_votes)):
        n = num_votes[ix]
        if n >= vote_thr:
            if s is None:
                s = running_range[0] + ix
            else:
                e = running_range[0] + ix + 1
        elif s is not None:
            # needed in case run of just 1
            e = s + 1 if e is None else e
            # append and reset
            split_voted_ranges.append([s, e])
            s = None
            e = None

    # finish off the last run
    if s is not None:
        e = s + 1 if e is None else e
        split_voted_ranges.append([s, e])
    
    return split_voted_ranges

@numba.jit(nopython=True)
def extend_range(range1, range2, num_votes):
    # difference between starts is location
    # in num_votes1 to start updating
    first_idx = range2[0] - range1[0]
    last_idx = len(num_votes)
    end_offset = range2[1] - range1[1]

    if end_offset > 0:
        # if range2 extends past range1
        # then add more votes to list
        # and update range1
        extension = [1 for _ in range(end_offset)]
        range1[1] = range2[1]
        num_votes.extend(extension)
    elif end_offset < 0:
        # adjust last_index because range2 doesn't
        # extend as far as range1
        last_idx += end_offset

    # increate vote totals
    for i in range(first_idx, last_idx):
        num_votes[i] += 1
        
    return range1, num_votes

@numba.jit(nopython=True)
def rle_voting(ranges, vote_thr=2):
    # ranges that past the vote_thr
    voted_ranges = []
    
    # initialize starting range and votes
    # for each index in the range
    running_range = None
    num_votes = None
    
    for range1,range2 in zip(ranges[:-1], ranges[1:]):
        if running_range is None:
            running_range = range1
            # all indices get 1 vote from range1
            num_votes = [1 for _ in range(range1[1] - range1[0])]
            
        # if starting index in range 2 is greater
        # than the end index of the running range there
        # is no overlap and we start tracking a new range
        if running_range[1] < range2[0]:
            # add ranges and reset
            voted_ranges.extend(
                split_range_by_votes(running_range, num_votes, vote_thr)
            )
            running_range = None
            num_votes = None
        else:
            # extend the running range and accumulate votes
            running_range, num_votes = extend_range(
                running_range, range2, num_votes
            )
            
    # if range was still going at the end
    # of the loop then finish processing it
    if running_range is not None:
        voted_ranges.extend(
            split_range_by_votes(running_range, num_votes, vote_thr)
        )
            
    return voted_ranges
