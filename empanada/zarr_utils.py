import numba
import numpy as np
from multiprocessing import Pool
from empanada.array_utils import put

__all__ = [
    'zarr_fill_instances'
]

@numba.jit(nopython=True)
def fill_func(seg1d, coords, instance_id):
    r"""Fills coords in seg1d (raveled image) with value instance_id"""
    # inplace fill seg1d with instance_id
    # at the given xy raveled coords
    for coord in coords:
        s, e = coord
        seg1d[s:e] = instance_id

    return seg1d

def fill_zarr_mp(*args):
    r"""Helper function for multiprocessing the filling of zarr slices"""
    # fills zarr array with multiprocessing
    index, slice_dict, array = args[0]
    d, h, w = array.shape
    seg2d = array[index].reshape(-1)
    for instance_id, coords in slice_dict.items():
        fill_func(seg2d, coords, int(instance_id))

    put(array, index, seg2d.reshape(h, w), axis=0)

def zarr_fill_instances(array, instances, processes=4):
    r"""Fills a zarr array in-place with instances.

    Args:
        array: zarr.Array of size (d, h, w)

        instances: Dictionary. Keys are instance_ids (integers) and
            values are another dictionary containing the run length
            encoding (keys: 'starts', 'runs').

        processes: Integer, the number of processes to run.

    """
    d, h, w = array.shape

    # convert instance coords to a z coord
    # and a raveled xy_coord array for each z slice
    instance_coords_2d = {}
    for instance_id, instance_attrs in instances.items():
        starts = instance_attrs['starts']
        ends = starts + instance_attrs['runs']
    
        start_zcoords = starts // (h * w)
        end_zcoords = (ends - 1) // (h * w)
        if not np.allclose(start_zcoords, end_zcoords):
            # this means a run extends across at least
            # 2 z slices so we need to separate the run
            # not very efficient, but this is usually an edge case
            split_locs = np.where(start_zcoords != end_zcoords)[0]
            offset = 0
            for loc in split_locs:
                insert_start = (start_zcoords[loc+offset] + 1) * h * w
                insert_end = ends[loc+offset]

                starts = np.insert(starts, loc+1+offset, insert_start)
                ends = np.insert(ends, loc+1+offset, insert_end)
                
                start_zcoords = np.insert(start_zcoords, loc+1+offset, start_zcoords[loc+offset] + 1)
                
                # update the end to stop at last xy coord of z slice
                ends[loc+offset] = insert_start
                offset += 1

        start_xycoords = starts % (h * w)
        end_xycoords = ends % (h * w)
        
        instance_coords_2d[instance_id] = [start_zcoords, start_xycoords, end_xycoords]

    # for each z slice we create an empty dict
    # that where each instance_id is a key
    # and the values are xy coordinates for the z slice
    slice_dicts = [{} for i in range(array.shape[0])]
    for instance_id, coords in instance_coords_2d.items():
        z = coords[0]
        xy = np.stack([coords[1], coords[2]], axis=1) # xy starts and ends

        # split the coords by unique z slice
        unq_z, section_idx = np.unique(z, return_index=True)

        # split xy coords by z slice and store them
        sections = np.split(xy, section_idx, axis=0)[1:]
        for sl,sec in zip(unq_z, sections):
            slice_dicts[sl][instance_id] = sec

    # setup for multiprocessing
    arg_iter = zip(
        range(d),
        slice_dicts,
        [array] * d
    )

    # fill the zarr volume. nothing to
    # return because it's done inplace
    with Pool(processes) as pool:
        pool.map(fill_zarr_mp, arg_iter)
