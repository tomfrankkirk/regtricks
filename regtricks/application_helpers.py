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

from .image_space import ImageSpace


# TODO:     intensity correction 


def src_load_helper(src):
    if isinstance(src, str):
        src = nibabel.load(src)
        data = src.get_data()
    elif isinstance(src, (Nifti1Image, MGHImage)):
        data = src.dataobj
    elif isinstance(src, FSLImage):
        data = src.data
    else: 
        raise RuntimeError("src must be a nibabel Nifti/MGH, FSL Image," 
                           " or path to image")

    return data, type(src)


def _make_iterable(data):
    """
    Ensure array is 4D, with the fourth dimension at the front (ie, T, XYZ).
    3D volumes will be expanded with a singleton dimension: 1, XYZ
    Used for iterating over the volumes of a series. 
    """
    if len(data.shape) == 4: 
        return np.moveaxis(data, 3, 0)
    else: 
        return data.reshape(1, *data.shape)


def interpolate_and_scale(data, coords_scale, out_size, **kwargs):
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

    interp = map_coordinates(data, coords_scale[0], **kwargs)
    return interp.reshape(out_size) * coords_scale[1] 


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

    if len(data.shape) != 4 and len(data.shape) != 3: 
        raise RuntimeError("Can only handle 3D/4D data")

    if len(transform) > 1 and (len(transform) != data.shape[-1]): 
        raise RuntimeError("Number of volumes in data does not match transform")

    # Prepare data for iterating, prepare worker function for each core 
    # Resolve the transform: this means that for each volume of the series, 
    # we have a corresponding array of coordinates onto which we need to 
    # iterpolate the date. Note that this is a backwards transform: we 
    # map the REFERENCE voxels into the SOURCE space and do the interpolation
    # there
    data = _make_iterable(data)
    worker = functools.partial(interpolate_and_scale, out_size=ref_spc.size, **kwargs)

    # If intensity correction has been requested on a NL transform, then resolve
    # will return tuples of (coordinates, scale factors). Pass these on as they
    # are
    if (transform.is_nonlinear and transform.intensity_correct): 
        coords_scales = transform.resolve(src_spc, ref_spc, data.shape[0])
        worker_args = zip(data, coords_scales)
    
    # If intensity correction has not been requested on a NL transform, then 
    # resolve returns array of coordinates only. In which case, zip them up
    # into tuples (coordinates, 1), used as a dummy scale factor 
    else: 
        coords = transform.resolve(src_spc, ref_spc, data.shape[0])
        worker_args = zip(data, zip(coords, itertools.repeat(1)))

    # Distribute amongst workers 
    if cores == 1:  
        resamp = [ worker(*vc) for vc in worker_args ] 
    else: 
        with mp.Pool(cores) as p: 
            resamp = p.starmap(worker, worker_args)

    # Stack all the individual volumes back up in time dimension 
    # Clip the array to the original min/max values 
    resamp = np.stack(resamp, axis=3)
    return np.clip(np.squeeze(resamp), data.min(), data.max()) 


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

    factor = [ int(f) for f in factor ]

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