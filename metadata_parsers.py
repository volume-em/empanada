import tifffile
import xmltodict
import numpy as np

__all__ = [
    'atlas3d', 'atlas5'
]

def read_image_size(fpath):
    with tifffile.TiffFile(fpath) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
            
    # returns size in height x width
    return int(tif_tags['ImageLength']), int(tif_tags['ImageWidth'])

def read_metadata(fpath):
    """
    Read till header tags
    """
    with tifffile.TiffFile(fpath) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
            
    return tif_tags

def atlas3d_metadata(fpath, convert_to_nm=True):
    """
    Return xyz voxel size from metadata dict. Optionally, convert
    the resolution to nanometers. Only applies to tif images
    generated in Atlas 3d.
    """
    metadata_tags = read_metadata(fpath)
    metadata_dict = xmltodict.parse(metadata_tags['FibicsXML'])['Fibics']
    
    if 'ATLAS3D' not in metadata_dict:
        raise Exception(f'{fpath} is not an ATLAS3D image!')
    
    # x and y pixel resolutions
    Ux = float(metadata_dict['Scan']['Ux'])
    Uy = float(metadata_dict['Scan']['Uy'])
    Vx = float(metadata_dict['Scan']['Vx'])
    Vy = float(metadata_dict['Scan']['Vy'])
    
    U = np.linalg.norm([Ux, Uy])
    V = np.linalg.norm([Vx, Vy])
    
    # absolute z position in stack
    Z  = float(metadata_dict['ATLAS3D']['Slice']['ZPos'])
    volume_name = metadata_dict['ATLAS3D']['Stack']['Name']
        
    # whether to invert boolean
    inverted = metadata_dict['ImageCorrections']['Invert'] == 'true'

    resolution = np.array([U, V])
    
    # convert resolution to nm if necessary
    units = metadata_dict['Scan']['FOV_X']['@units']
    if units == 'um': # micrometers
        resolution *= 1000
        Z *= 1000
    elif units == 'nm':
        pass
    else:
        raise Exception(f'Unrecognized resolution units of {units}')
        
    # only keep a few metadata keys
    slim_metadata = {}
    slim_metadata['volume_name'] = volume_name
    slim_metadata['resolution'] = resolution
    slim_metadata['z_pos'] = Z
    slim_metadata['inverted'] = inverted
    
    return slim_metadata

def atlas5_metadata(fpath, convert_to_nm=True):
    """
    Return xyz voxel size from metadata dict. Optionally, convert
    the resolution to nanometers. Only applies to tif images
    generated in Atlas 3d.
    """
    metadata_tags = read_metadata(fpath)
    metadata_dict = xmltodict.parse(metadata_tags['FibicsXML'])['Fibics']
    
    if 'ATLAS3D' in metadata_dict:
        raise Exception(f'{fpath} is an ATLAS3D image!')
    
    # x and y pixel resolutions
    Ux = float(metadata_dict['Scan']['Ux'])
    Uy = float(metadata_dict['Scan']['Uy'])
    Vx = float(metadata_dict['Scan']['Vx'])
    Vy = float(metadata_dict['Scan']['Vy'])
    
    U = np.linalg.norm([Ux, Uy])
    V = np.linalg.norm([Vx, Vy])
    
    # whether image has been inverted
    inverted = metadata_dict['ImageCorrections']['Invert'] == 'true'

    resolution = np.array([U, V])
    
    # convert resolution to nm if necessary
    units = metadata_dict['Scan']['FOV_X']['@units']
    if units == 'um': # micrometers
        resolution *= 1000
    elif units == 'nm':
        pass
    else:
        raise Exception(f'Unrecognized resolution units of {units}')
        
    # only keep a few metadata keys
    slim_metadata = {}
    slim_metadata['resolution'] = resolution
    slim_metadata['inverted'] = inverted
    
    return slim_metadata