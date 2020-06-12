import os.path as op 
from textwrap import dedent

import nibabel
from nibabel import Nifti2Image, MGHImage
import numpy as np 
from fsl.data.image import Image as FSLImage

from regtricks.image_space import ImageSpace
from regtricks import x5_interface as x5 
from regtricks import application_helpers as apply
from regtricks import multiplication as multiply

# cache for intensity correction?
# cast_space method 

class Transform(object):
    """
    Base object for all transformations. This should never actually be 
    instantiated but is instead used to provide common functions. 
    
    Attributes: 
        _cache: use for storing resolved displacement fields and sharing
            amongst workers in multiprocessing pool 
        islinear: Registrations or MotionCorrections
        isnonlinear: NonLinearRegistrations or NLMCs 

    """
    
    def __init__(self):
        self._cache = None 

    @property
    def is_linear(self):
        from regtricks.transforms.linear import Registration, MotionCorrection
        return (type(self) in [Registration, MotionCorrection])

    @property 
    def is_nonlinear(self):
        return not self.is_linear

    def save(self, path):
        """Save transformation at path in X5 format (experimental)"""

        x5.save_manager(self, path)

    def __repr__(self):
        raise NotImplementedError()

    def reset_cache(self):
        self.cache = None 

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, new):
        if not ((new is None) or isinstance(new, np.ndarray)):
            raise ValueError("Cache can only be None or np.ndarray")
        self._cache = new 

    # We need to explicitly not implement np.array_ufunc to allow overriding
    # of __matmul__, see: https://github.com/numpy/numpy/issues/9028
    __array_ufunc__ = None 

    def __matmul__(self, other):

        from regtricks.transforms.linear import Registration, MotionCorrection
        from regtricks.transforms.nonlinear import NonLinearMotionCorrection, NonLinearRegistration

        other = multiply.cast_potential_array(other)
        high_type = multiply.get_highest_type(self, other)

        if high_type is Registration: 
            return multiply.registration(self, other)
        elif high_type is MotionCorrection: 
            return multiply.moco(self, other)
        elif high_type is NonLinearRegistration: 
            return multiply.nonlinearreg(self, other)
        elif high_type is NonLinearMotionCorrection:
            return multiply.nonlinearmoco(self, other)
        else: 
            raise NotImplementedError("Not Transformation objects")

    def __rmatmul__(self, other):

        from regtricks.transforms.linear import Registration, MotionCorrection
        from regtricks.transforms.nonlinear import NonLinearMotionCorrection, NonLinearRegistration

        other = multiply.cast_potential_array(other)
        high_type = multiply.get_highest_type(self, other)

        if high_type is Registration: 
            return multiply.registration(other, self)
        elif high_type is MotionCorrection: 
            return multiply.moco(other, self)
        elif high_type is NonLinearRegistration: 
            return multiply.nonlinearreg(other, self)
        elif high_type is NonLinearMotionCorrection:
            return multiply.nonlinearmoco(other, self)
        else: 
            raise NotImplementedError("Not Transformation objects")

    def apply_to_image(self, src, ref, superlevel=1, cores=1, **kwargs):
        """
        Applies transformation to data array. If a registration is applied 
        to 4D data, the same transformation will be applied to all volumes 
        in the series. 

        Args:   
            src (str/NII/MGZ/FSLImage): image to transform 
            ref (str/NII/MGZ/FSLImage/ImageSpace): target space for data 
            superlevel (int/iterable): resample onto a super-resolution copy
                of the reference grid and sum back down to target (replicates
                applywarp -super behaviour). Either a single integer value, or 
                an iterable of values for each dimension, should be given. 
            cores (int): CPU cores to use for 4D data
            **kwargs: passed on to scipy.ndimage.map_coordinates

        Returns: 
            (np.array) transformed image data in ref voxel grid.
        """

        data, creator = apply.src_load_helper(src)
        resamp = self.apply_to_array(data, src, ref, superlevel, cores, **kwargs)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        
        if creator is MGHImage:
            ret = MGHImage(resamp, ref.vox2world, ref.header)
            return ret 
        else: 
            ret = Nifti2Image(resamp, ref.vox2world, ref.header)
            if creator is FSLImage:
                return FSLImage(ret)
            else: 
                return ret 

    def apply_to_array(self, data, src, ref, superlevel=1, cores=1, **kwargs):
        """
        Applies transformation to data array. If a registration is applied 
        to 4D data, the same transformation will be applied to all volumes 
        in the series. 

        Args:   
            data (array): 3D or 4D array. 
            src (str/NII/MGZ/FSLImage/ImageSpace): current space of data 
            ref (str/NII/MGZ/FSLImage/ImageSpace): target space for data 
            superlevel (int/iterable): resample onto a super-resolution copy
                of the reference grid and sum back down to target (replicates
                applywarp -super behaviour). Either a single integer value, or 
                an iterable of values for each dimension, should be given. 
            cores (int): CPU cores to use for 4D data
            **kwargs: passed on to scipy.ndimage.map_coordinates

        Returns: 
            (np.array) transformed image data in ref voxel grid.
        """

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        # Force superlevel into an integer array of length 3 for 3D data 
        # or array of (XYZ,1) for 4D data 
        superlevel = np.array(superlevel).round().astype(np.int16)
        if superlevel.size == 1: superlevel *= np.ones(3)

        # Create super-resolution reference grid
        if (superlevel != 1).any(): 
            ref = ref.resize_voxels(1/superlevel, 'ceil')

        if not (data.shape[:3] == src.size).all(): 
            raise RuntimeError("Data shape does not match source space")

        # Force to float data 
        if data.dtype.kind != 'f': 
            data = data.astype(np.float32)
        resamp = apply.despatch(data, self, src, ref, cores, **kwargs)

        # Sum back down if super-resolution 
        if len(data.shape) == 4: superlevel = np.array((*superlevel, 1))
        if (superlevel != 1).any():
            resamp = apply.sum_array_blocks(resamp, superlevel)
            resamp = resamp / np.prod(superlevel[:3])
            
        return resamp      