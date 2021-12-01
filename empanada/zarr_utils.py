import numba
import numpy as np
from multiprocessing import Pool
from torch.utils.data import Dataset

__all__ = [
    'zarr_take3d',
    'zarr_put3d',
    'ZarrData',
    'zarr_fill_instances'
]

def zarr_take3d(array, indices, axis=0):
    """Similar to np.take but for 3d zarr arrays."""
    assert len(array.shape) == 3
    assert axis in [0, 1, 2]
    
    if axis == 0:
        return array[indices]
    elif axis == 1:
        return array[:, indices]
    else:
        return array[..., indices]
    
def zarr_put3d(array, index, value, axis=0):
    """Similar to np.put_along_axis but for 3d zarr arrays."""
    assert len(array.shape) == 3
    assert isinstance(index, int)
    assert len(value.shape) == 2
    assert axis in [0, 1, 2]
    
    if axis == 0:
        array[index] = value
    elif axis == 1:
        array[:, index] = value
    else:
        array[..., index] = value
    
    return array

class ZarrData(Dataset):
    def __init__(self, array, axis=0, tfs=None):
        super(ZarrData, self).__init__()
        self.array = array
        self.axis = axis
        self.tfs = tfs
        
    def __len__(self):
        return self.array.shape[self.axis]
    
    def __getitem__(self, idx):
        # load the image
        image = zarr_take3d(self.array, idx, self.axis)
        image = self.tfs(image=image)['image']
        
        return {'index': idx, 'image': image}
    
@numba.jit(nopython=True)
def fill_func(seg1d, coords, instance_id):
    # inplace fill seg1d with instance_id
    # at the given xy raveled coords
    for coord in coords:
        s, e = coord
        seg1d[s:e] = instance_id
    
    return seg1d

def fill_zarr_mp(*args):
    # fills zarr array with multiprocessing
    index, slice_dict, array = args[0]
    d, h, w = array.shape
    seg2d = array[index].reshape(-1)
    for instance_id, coords in slice_dict.items():
        fill_func(seg2d, coords, int(instance_id))

    zarr_put3d(array, index, seg2d.reshape(h, w), axis=0)
    
def zarr_fill_instances(array, instances, processes=4):
    d, h, w = array.shape
    
    # convert instance coords to a z coord
    # and a raveled xy_coord array for each z slice
    instance_coords_2d = {}
    for instance_id, instance_attrs in instances.items():
        starts = instance_attrs['starts']
        ends = starts + instance_attrs['runs']
        
        start_zcoords = starts // (h * w)
        end_zcoords = ends // (h * w)
        assert np.allclose(start_zcoords, end_zcoords), \
        "Run extends across z slices!"
        
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