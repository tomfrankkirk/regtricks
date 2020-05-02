import functools
import multiprocessing as mp 
import tempfile 
import os.path as op 
import subprocess

import nibabel
from nibabel import Nifti1Image, MGHImage
from fsl.data.image import Image as FSLImage
from fsl.wrappers import applywarp
import numpy as np 
from scipy.ndimage import map_coordinates


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



def _application_worker(data, transform, src_spc, ref_spc, cores, **kwargs):
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
    return _clip_array(np.squeeze(resamp), data) 


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
        t = t.T
    return t[:3,:]

def _clip_array(array, ref):
    """Clip array values to min/max of that contained in ref"""
    min_max = (ref.min(), ref.max())
    array[array < min_max[0]] = min_max[0]
    array[array > min_max[1]] = min_max[1]
    return array 

def applywarp_helper(fcoeffs, pre, post, data, src, ref, 
                     intensity_correct=False):

    assert pre.shape[-1] == 4
    assert post.shape[-1] == 4

    with tempfile.TemporaryDirectory() as d: 

        # We need to dump lots of stuff to files... 
        coeffs = op.join(d, 'coeffs.nii.gz')
        field = op.join(d, 'field.nii.gz')
        outpath = op.join(d, 'warped.nii.gz')
        refvol = op.join(d, 'ref.nii.gz')
        jac = op.join(d, 'jac.nii.gz')
        nibabel.save(fcoeffs.coefficients, coeffs)
        ref.touch(refvol)

        # Create displacement field and jacobian for intensity correction 
        cmd = f'fnirtfileutils -i {coeffs} -r {refvol} -o {field} -j {jac} -a'
        subprocess.run(cmd, shell=True)

        # Run applywap with the displacement field 
        indata = src.make_nifti(data)
        out = applywarp(indata, refvol, outpath, premat=pre, 
                warp=field, postmat=post)
        out = nibabel.load(outpath).get_data()

        if intensity_correct: 
            # FIXME: what do we do with jacobian here?
            scale = nibabel.load(jac).get_data()
            out /= scale 

        return out