"""
X5 interface for regtricks. With thanks to Paul McCarthy; this is almost a
direct copy of his fslpy.transform.x5 module
"""

import os.path as op 
import json

import h5py 
import numpy as np 

# TODO: these should be able to handle registrations without defined image spaces
# eg, an Identity registration with src/ref set as None 

X5_FORMAT  = 'X5'
X5_VERSION = '0.1.0'

class X5Error(Exception):
    pass 

def save_manager(reg, path):
    """
    Save Registration or MotionCorrection objects in X5 format
    """

    from regtricks import MotionCorrection, Registration

    ext = op.splitext(path)[1]
    if ext != '.x5':
        path += '.x5'

    with h5py.File(path, 'w') as f: 

        g = f.create_group('/Transform')
        if isinstance(reg, MotionCorrection): 
            f.attrs['Type'] = 'linear_timeseries'
            write_affine(g, 
                np.stack(reg.src2ref, axis=2), 
                np.stack(reg.ref2src, axis=2))
        elif isinstance(reg, Registration):
            f.attrs['Type'] = 'linear'
            write_affine(g, reg.src2ref, reg.ref2src)
        else: 
            raise X5Error("Unrecognised registration type")

        g = f.create_group('/A')
        write_imagespace(g, reg.src_spc)

        g = f.create_group('/B')
        write_imagespace(g, reg.ref_spc)
        write_metadata(f)


def load_manager(path):
    """
    Load transformation objects from X5 format
    """

    from .regtricks import Registration, MotionCorrection

    with h5py.File(path, 'r') as f: 
        reg_type = f.attrs['Type']

        read_metadata(f['/'])
        src_spc = read_imagespace(f['/A'])
        ref_spc = read_imagespace(f['/B'])
        src2ref = read_affine(f['/Transform'])
        shp = src2ref.shape 

        if reg_type == 'linear':
            return Registration(src2ref, src_spc, ref_spc, "world")

        elif reg_type == 'linear_timeseries':
            return MotionCorrection(
                [ src2ref[:,:,x] for x in range(shp[-1]) ], 
                src_spc, ref_spc, "world")

        else: 
            raise X5Error("Unrecognised registration type")
        
        

def write_metadata(group):
    """Write X5 format metadata"""

    group.attrs['Format']   = X5_FORMAT
    group.attrs['Version']  = X5_VERSION
    group.attrs['Metadata'] = json.dumps({'regtricks' : 0.1})

def read_metadata(group):
    """Read X5 format metadata"""

    x5_format = group.attrs.get('Format')
    x5_version = group.attrs.get('Version')
    assert (x5_format is not None) and (x5_version is not None)
    return 

def write_imagespace(group, spc):
    """
    Write ImageSpace properties (size, voxel size, vox2world) into X5 format
    """

    group.attrs['Type'] = 'image'
    group.attrs['Size'] = spc.size 
    group.attrs['Scales'] = spc.vox_size
    affgroup = group.create_group('Mapping')
    write_affine(affgroup, spc.vox2world, spc.world2vox)

def read_imagespace(group):
    """
    Read ImageSpace properties (size, voxel size, vox2world) into X5 format, 
    and return ImageSpace object 
    """

    from .image_space import ImageSpace

    if group.attrs.get('Type') != 'image':
        raise X5Error('Group does not represent an image')

    size = np.asarray(group.attrs['Size'])
    vox_size = np.asarray(group.attrs['Scales'])
    vox2world = read_affine(group['Mapping'])
    if not ((size.size == 3) and (vox_size.size == 3)):
        raise X5Error('Incorrect size and scales')
    
    return ImageSpace.manual(vox2world, size, vox_size)

def write_affine(group, matrix, inverse):
    """Write a single or stack of (4,4) arrays to X5 group"""
    
    group.attrs['Type'] = 'affine'
    if not matrix.shape[:2] == (4,4):
        raise RuntimeError("Matrix must be single or stack of (4,4) shape")
    group.create_dataset('Matrix',  data=matrix)
    group.create_dataset('Inverse', data=inverse)


def read_affine(group):
    """Load a single or stack of (4,4) arrays to X5 group"""
    if group.attrs.get('Type') != 'affine':
        raise X5Error('Group does not represent affine')

    matrix = group['Matrix']
    if not matrix.shape[:2] == (4,4):
        raise X5Error('Incorrect matrix dimensions')

    return np.asarray(matrix)


def check_is_x5(path):
    try: 
        with h5py.File(path, 'r') as f: 
            read_metadata(f['/'])
        return True 
    except: 
        return False
