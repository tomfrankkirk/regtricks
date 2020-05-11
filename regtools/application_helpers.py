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


def _reshape_for_iterate(data):
    if len(data.shape) == 4: 
        return np.moveaxis(data, 3, 0)
    else: 
        return data.reshape(1, *data.shape)

def _reshape_after_iterate(data):
    pass



def linear(data, transform, src_spc, ref_spc, cores, **kwargs):
    """
    Worker function for Registration and MotionCorrection apply_to_image()

    Args: 
        data: np.array of data (3D or 4D)
        transform: transformation between reference space and source, 
            in world-world terms
        src_spc: ImageSpace in which data currently lies
        ref_spc: ImageSpace towards which data will be transformed
        cores: number of cores to use (for 4D data)
        **kwargs: passed onto scipy.ndimage.interpolate.map_coordinates

    Returns: 
        np.array of transformed data 
    """

    if len(data.shape) != 4 and len(data.shape) != 3: 
        raise RuntimeError("Can only handle 3D/4D data")

    if len(transform) > 1 and (len(transform) != data.shape[-1]): 
        raise RuntimeError("Number of volumes in data does not match transform")

    # Move the 4th dimension to the front, so that we can iterate over each 
    # volume of the timeseries. If 3D data, pad out the array with a
    # singleton dimension at the front to get the same effect 
    data = _reshape_for_iterate(data)

    # Affine transformation requires mapping from reference voxels
    # to source voxels (the inverse of how transforms are given)
    worker = functools.partial(map_coordinates, **kwargs)
    vols_coords = vol_coord_generator(data, transform.ref2src_world, 
                                      src_spc, ref_spc)

    if cores == 1:  
        resamp = [ worker(*vc) for vc in vols_coords ] 
    else: 
        with mp.Pool(cores) as p: 
            resamp = p.starmap(worker, vols_coords)

    # Stack all the individual volumes back up in time dimension 
    # Clip the array to the original min/max values 
    resamp = np.stack([r.reshape(ref_spc.size) for r in resamp], axis=3)
    return np.clip(np.squeeze(resamp), data.min(), data.max()) 


def vol_coord_generator(data, mats, src, ref):

    if isinstance(mats, np.ndarray): 
        mats = [mats] * data.shape[0] 

    for vol, mat in zip(data, mats):
        ref2src_vox = (src.world2vox @ mat @ ref.vox2world)
        ijk = ref.ijk_grid('ij').reshape(-1, 3).T
        ijk = _affine_transform(ref2src_vox, ijk)
        yield vol, ijk 


def _affine_transform(matrix, points): 
    """Affine transform a 3D set of points"""
    transpose = False 
    if points.shape[1] == 3: 
        transpose = True 
        points = points.T 
    p = np.ones((4, points.shape[1]))
    p[:3,:] = points 
    t = matrix @ p 
    if transpose: 
        return t[:3,:].T
    return t[:3,:]


def nonlinear(data, src, ref, fcoeffs, pre, post,
                     intensity_correct=False, cores=1):

    if type(pre) is list: 
        assert len(data.shape) == 4 
        premat = np.concatenate([ 
            m.to_fsl(src, fcoeffs.src_spc) for m in pre ], axis=0)
    else: 
        premat = pre.to_fsl(src, fcoeffs.src_spc) 

    if type(post) is list: 
        assert len(data.shape) == 4 
        postmat = np.concatenate([ 
            m.to_fsl(fcoeffs.ref_spc, ref) for m in post ], axis=0)
    else: 
        postmat = post.to_fsl(fcoeffs.ref_spc, ref)

    with tempfile.TemporaryDirectory() as d: 

        # We need to dump lots of stuff to files... 
        cpath = op.join(d, 'coeffs.nii.gz')
        refpath = op.join(d, 'ref.nii.gz')
        srcpath = op.join(d, 'src.nii.gz')
        nibabel.save(fcoeffs.coefficients, cpath)
        ref.touch(refpath)
        src.touch(srcpath)

        if len(data.shape) == 4 and (cores > 1): 
            worker_args = []
            for worker in range(cores): 
                start = (worker * data.shape[3] // cores)
                stop = min([((worker+1) * data.shape[3] // cores), data.shape[3]+1])
                if premat.shape[0] > 4: 
                    premat_slice = premat[4*start : 4*stop,:]
                else: 
                    premat_slice = premat
                if postmat.shape[0] > 4: 
                    postmat_slice = postmat[4*start : 4*stop,:]
                else: 
                    postmat_slice = postmat

                worker_args.append([data[...,start:stop], srcpath, refpath, cpath, premat_slice, postmat_slice])

            assert stop == data.shape[3], 'Did not assign all data to workers'
            with mp.Pool(cores) as p: 
                chunks = p.starmap(applywarp_worker, worker_args)

            return np.concatenate(chunks, axis=3)

        else: 
            return applywarp_worker(data, srcpath, refpath, cpath, premat, postmat)


def applywarp_worker(data, srcpath, refpath, coeffpath, premat, postmat):

    if len(data.shape) == 4:
        assert (premat.shape[0] == 4 * data.shape[3]) or (premat.shape[0] == 4)
        assert (postmat.shape[0] == 4 * data.shape[3]) or (postmat.shape[0] == 4)
    else: 
        assert premat.shape == (4,4) and postmat.shape == (4,4)

    with tempfile.TemporaryDirectory() as d: 

        outpath = op.join(d, 'warped.nii.gz')
        inpath = op.join(d, 'data.nii.gz')
        ImageSpace.save_like(srcpath, data, inpath)
        pre = op.join(d, 'pre.mat')
        post = op.join(d, 'post.mat')
        np.savetxt(pre, premat)
        np.savetxt(post, postmat)
        applywarp(inpath, refpath, outpath, premat=pre, 
                warp=coeffpath, postmat=post)
        out = nibabel.load(outpath).get_data()

    return out 