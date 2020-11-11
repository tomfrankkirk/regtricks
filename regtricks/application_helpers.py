import functools
import multiprocessing as mp 
import tempfile 
import os.path as op 
import subprocess
import os 
import shutil 
import itertools

import nibabel
from nibabel import Nifti1Image, MGHImage
from fsl.data.image import Image as FSLImage
from fsl.wrappers import applywarp
import numpy as np 
from scipy.ndimage import map_coordinates

from regtricks.image_space import ImageSpace


def src_load_helper(src):
    if isinstance(src, str):
        src = nibabel.load(src)
        data = src.get_fdata()
    elif isinstance(src, (Nifti1Image, MGHImage)):
        data = src.dataobj
    elif isinstance(src, FSLImage):
        data = src.data
    else: 
        raise RuntimeError("src must be a nibabel Nifti/MGH, FSL Image," 
                           " or path to image")

    return data, type(src)


def _make_iterable(data, series_length):
    """
    Ensure array is 4D, with the fourth dimension at the front (ie, T, XYZ).
    3D volumes will be expanded with a singleton dimension: 1, XYZ
    Used for iterating over the volumes of a series. 
    """
    if len(data.shape) == 4: 
        return np.moveaxis(data, 3, 0)
    elif series_length == 1: 
        return data.reshape(1, *data.shape)
    else: 
        assert data.ndim == 3, 'series expansion should be with 3D volume'
        return [ data ] * series_length


def interpolate_and_scale(idx, data, transform, src_spc, ref_spc, **kwargs):
    """
    Used for partial function application to share interpolation jobs
    amongst workers of a mp.Pool(). Interpolate data onto the coordinates
    given in the tuple coords_scale, and multiply the output by the other
    value in coords_scale. Reshape the output to size out_size. 

    Args: 
        data (np.ndarray): 3D, image data 
        coords_scale (np.ndarray, np.ndarray): (N,3) coordinates to interpolate
            onto (indices into data array), value by which to scale output
            (int or another np.ndarray for intensity correction)
        out_size (np.ndarray): 3-vector, shape of output
        **kwargs: passed onto scipy map_coordinates

    Returns: 
       (np.ndarray), sized as out_size, interpolated output 
    """

    # We transform a bool mask along with the input data to guard
    # against spline interpolation artefacts in the transformed data 
    superlevel = kwargs.pop('superlevel')
    ijk, scale = transform.resolve(src_spc, ref_spc, idx)
    mask = data.astype(np.bool).astype(np.float32)
    interp = map_coordinates(data, ijk, **kwargs)
    interp_mask = map_coordinates(mask, ijk, order=0)
    interp_mask = interp_mask.astype(np.bool)

    # If the fill value has been specified, set the min/max
    # range for clipping in light of this 
    if 'cval' in kwargs:
        cval = kwargs['cval']
        cmin = data.min() if cval > data.min() else cval
        cmax = data.max() if cval < data.max() else cval 
    else: 
        cmin = data.min() 
        cmax = data.max()

    # Clip output values to the required min max, and remask the data
    # to remove spline artefacts
    interp = np.clip(interp, cmin, cmax)
    interp[~interp_mask] = 0
    interp = interp.reshape(ref_spc.size) * scale 

    # If supersampling used, sum array blocks back down to target 
    if (superlevel > 1).any():
        interp = sum_array_blocks(interp, superlevel)
        interp = interp / np.prod(superlevel[:3])
    
    return interp


def despatch(data, transform, src_spc, ref_spc, cores, **kwargs):
    """
    Apply a transform to an array of data, mapping from source space 
    to reference. Essentially this is an extended wrapper for Scipy 
    map_coordinates. 

    Args: 
        data (array): np.array of data (3D or 4D)
        transform (Transformation): between source and reference space 
        src_spc (ImageSpace): in which data currently lies
        ref_spc (ImageSpace): towards which data will be transformed
        cores (int): number of cores to use (for 4D data)
        **kwargs: passed onto scipy.ndimage.interpolate.map_coordinates

    Returns: 
        (np.array) transformed data 
    """

    if data.ndim not in (3,4): 
        raise RuntimeError("Can only handle 3D/4D data")
            
    if ((data.ndim == 4) and (len(transform) > 1) and 
        (len(transform)) != data.shape[3]):
        raise RuntimeError("Number of volumes in 4D series does not match "
                "length of transformation series")

    if len(transform) == 1: 
        series_length = 1 if data.ndim == 3 else data.shape[3]
    else: 
        series_length = len(transform)

    # Make the data 4D so that the workers of the pool can iterate over it 
    # Each worker recieves a tuple of (vol, idx), one frame of the series and
    # its corresponding index number (which is used to get the correct bit
    # of the transform). Pre-calculate and cache any information that can be 
    # shared amongst the workers 
    data = _make_iterable(data, series_length)
    transform.prepare_cache(ref_spc)
    worker_args = zip(range(series_length), data)
    worker = functools.partial(interpolate_and_scale, 
        transform=transform, ref_spc=ref_spc, src_spc=src_spc, **kwargs)

    # Distribute amongst workers 
    if cores == 1:  
        resamp = [ worker(*vc) for vc in worker_args ] 
    else: 
        with mp.Pool(cores) as p: 
            resamp = p.starmap(worker, worker_args)

    # Stack all the individual volumes back up in time dimension 
    # Clip the array to the original min/max values 
    # Reset the cache on the transform to be safe. 
    transform.reset_cache()
    resamp = np.stack(resamp, axis=3)
    if resamp.shape[3] == 1: 
        resamp = np.squeeze(resamp, axis=3) 
    return resamp


def aff_trans(matrix, points): 
    """Affine transform a 3D set of points"""

    if not matrix.shape == (4,4): 
        raise ValueError("Matrix needs to be a 4x4 array")

    if points.shape[1] == 3: 
        transpose = True 
        points = points.T 
    else: 
        transpose = False 

    p = np.ones((4, points.shape[1]))
    p[:3,:] = points 
    t = matrix @ p 

    if transpose: 
        return t[:3,:].T
    else: 
        return t[:3,:]

def sum_array_blocks(array, factor):
    """Sum sub-arrays of a larger array, each of which is sized according to factor. 
    The array is split into smaller subarrays of size given by factor, each of which 
    is summed, and the results returned in a new array, shrunk accordingly. 

    Args:
        array: n-dimensional array of data to sum
        factor: n-length tuple, size of sub-arrays to sum over

    Returns:
        array of size array.shape/factor, each element containing the sum of the 
            corresponding subarray in the input
    """

    if len(factor) != len(array.shape):
        raise RuntimeError("factor must be of same length as number of dimensions")

    if np.any(np.mod(factor, np.ones_like(factor))):
        raise RuntimeError("factor must be of integer values only")

    factor = np.array(factor, dtype=np.int32)
    if (np.mod(array.shape, factor) > 0).any(): 
        raise RuntimeError(("Factor {} must be a perfect divisor of array shape {}".
            format(factor, array.shape)))

    outshape = [ int(s/f) for (s,f) in zip(array.shape, factor) ]
    out = np.copy(array)

    for dim in range(3):
        newshape = [0] * 4

        for d in range(3):
            if d < dim: 
                newshape[d] = outshape[d]
            elif d == dim: 
                newshape[d+1] = factor[d]
                newshape[d] = outshape[d]
            else: 
                newshape[d+1] = array.shape[d]

        newshape = newshape + list(array.shape[3:])
        out = np.sum(out.reshape(newshape), axis=dim+1)

    return out 