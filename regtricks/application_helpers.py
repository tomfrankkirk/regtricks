import functools
import multiprocessing as mp 
import tempfile 
import os.path as op 
import subprocess
import os 
import shutil 

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
    worker = functools.partial(map_coordinates, **kwargs)
    worker_args = zip(data, transform.resolve(src_spc, ref_spc, data.shape[0]))

    # Distribute amongst workers 
    if cores == 1:  
        resamp = [ worker(*vc) for vc in worker_args ] 
    else: 
        with mp.Pool(cores) as p: 
            resamp = p.starmap(worker, worker_args)

    # Stack all the individual volumes back up in time dimension 
    # Clip the array to the original min/max values 
    resamp = np.stack([r.reshape(ref_spc.size) for r in resamp], axis=3)
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