import os.path as op 
import glob 
import os 
from textwrap import dedent
import tempfile
import subprocess
import copy

import nibabel
from nibabel import Nifti2Image, MGHImage
import numpy as np 
from fsl.data.image import Image as FSLImage
from fsl.wrappers import applywarp

from .image_space import ImageSpace
from . import x5_interface as x5 
from . import application_helpers as apply
from .fnirt_coefficients import FNIRTCoefficients, NonLinearProduct, det_jacobian
from . import multiplication as multiply

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

        resamp = apply.despatch(data, self, src, ref, cores, **kwargs)

        # Sum back down if super-resolution 
        if len(data.shape) == 4: superlevel = np.array((*superlevel, 1))
        if (superlevel != 1).any():
            resamp = apply.sum_array_blocks(resamp, superlevel)

        return resamp      


class Registration(Transform):
    """
    Represents a transformation between the source image and reference.
    If src and ref are given, the transformation is assumed to be in 
    FLIRT/FSL convention, otherwise it is assumed to be in world convention.

    Args: 
        src2ref: either a 4x4 np.array representing affine transformation
            from source to reference, or a path to a text-like file 
    """

    def __init__(self, src2ref):
        Transform.__init__(self)

        if isinstance(src2ref, str): 
            src2ref = np.loadtxt(src2ref)

        if (src2ref.shape != (4,4) 
            or (np.abs(src2ref[3,:] - [0,0,0,1]) > 1e-9).any()):
            raise RuntimeError("src2ref must be a 4x4 affine matrix, where ",
                               "the last row is [0,0,0,1].")

        self.__src2ref = src2ref

    @classmethod
    def from_flirt(cls, src2ref, src, ref):
        """TODO: doc me"""

        if isinstance(src2ref, str): 
            src2ref = np.loadtxt(src2ref)

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        src_spc = src 
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        ref_spc = ref 

        src2ref = (ref_spc.FSL2world @ src2ref @ src_spc.world2FSL)
        return Registration(src2ref)

    def __len__(self):
        return 1 

    def __repr__(self):
        
        formatter = "{:8.3f}".format 
        with np.printoptions(precision=3, formatter={'all': formatter}):
            text = (f"""\
                Registration (linear) with properties:
                src2ref:       {self.src2ref[0,:]}
                               {self.src2ref[1,:]}
                               {self.src2ref[2,:]}
                               {self.src2ref[3,:]}""")
        return dedent(text)
    
    @property
    def ref2src(self):
        return np.linalg.inv(self.__src2ref)

    @property
    def src2ref(self):
        return self.__src2ref

    @classmethod
    def identity(cls):
        return Registration(np.eye(4))

    def inverse(self):
        """Self inverse"""
        constructor = type(self)
        return constructor(self.ref2src)

    def to_fsl(self, src, ref):
        """
        Return transformation in FSL convention, for given src and ref, 
        as np.array. This will be 3D in the case of MotionCorrections
        """

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        return ref.world2FSL @ self.src2ref @ src.FSL2world

    def to_flirt(self, src, ref):
        """Alias for self.to_fsl()"""
        return self.to_fsl(src, ref)

    def save_txt(self, path):
        """Save as textfile at path"""
        np.savetxt(path, self.src2ref)

    def apply_to_grid(self, src):
        # TODO: move this onto the image space class 
        """
        Apply registration to the voxel grid of an image, retaining original
        voxel data (no resampling). This is equivalent to shifting the image
        within world space but not altering the contents of the image itself.

        Args: 
            src: str, nibabel Nifti/MGH or FSL Image to apply transform
        
        Returns: 
            image object, of same type as source. 
        """

        data, create = apply.src_load_helper(src)
        src_spc = ImageSpace(src)
        new_spc = src_spc.transform(self.src2ref)
               
        if create is MGHImage:
            ret = MGHImage(data, new_spc.vox2world, new_spc.header)
            return ret 
        else: 
            ret = Nifti2Image(data, new_spc.vox2world, new_spc.header)
            if create is FSLImage:
                return FSLImage(ret)
            else: 
                return ret 

    def prepare_cache(self, ref):
        """
        Cache re-useable data before interpolate_and_scale. Just the voxel
        index grid of the reference space is stored
        """

        self.cache = ref.ijk_grid('ij').reshape(-1, 3)

    def resolve(self, src, ref, *unused):
        """
        Return a coordinate array and scale factor that maps reference voxels
        into source voxels, including the transform. Uses cached values, if
        available. 

        Args: 
            src (ImageSpace): in which data currently exists and interpolation
                will be performed
            ref (ImageSpace): in which data needs to be expressed

        Returns: 
            (np.ndarray, 1) coordinates on which to interpolate and identity 
                scale factor
        """

        # Array of all voxel indices in the reference grid
        # Map them into world coordinates, apply the transform
        # and then into source voxel coordinates for the interpolation 
        ref2src_vox = (src.world2vox @ self.ref2src @ ref.vox2world)
        ijk = apply.aff_trans(ref2src_vox, self.cache).T
        scale = 1 
        return (ijk, scale)


class MotionCorrection(Registration):
    """
    A sequence of Registration objects, one for each volume in a timeseries. 
    For within-series motion correction (not using an external reference), 
    src and ref will refer to the same target. If only the src is given, then
    ref is assumed to be the same as src (ie, within-series), with FSL 
    convention. 

    Args: 
        mats: a path to a directory containing transformation matrices, in
            name order (all files will be loaded), or a list of individual
            filenames, or a list of np.arrays 
    """

    def __init__(self, mats):
        Transform.__init__(self)

        if isinstance(mats, str):
            mats = sorted(glob.glob(op.join(mats, '*')))
            if not mats: 
                raise RuntimeError("Did not find any matrices in %s" % mats)
            
        self.__transforms = []
        for mat in mats:
            if isinstance(mat, (np.ndarray, str)): 
                m = Registration(mat)
            else: 
                m = mat 
            self.__transforms.append(m)

    def from_flirt(self, *args):
        raise NotImplementedError("Use the MotionCorrection.from_mcflirt() method")

    @classmethod
    def from_mcflirt(cls, mats, src, ref):
        """TODO: doc me"""

        if isinstance(mats, str):
            mats = sorted(glob.glob(op.join(mats, '*')))
            if not mats: 
                raise RuntimeError("Did not find any matrices in %s" % mats)

        if isinstance(mats, np.ndarray):
            if mats.ndim == 3: 
                if not mats.shape[:2] ==(4,4): 
                    raise ValueError("A 3D stack of matrices should have size "
                            "(4,4) in first two dimensions")
            if mats.ndim == 2: 
                if ((mats.shape[0] % 4) or (mats.shape[1] != 4)): 
                    raise ValueError("A 2D array should have size Nx4, "
                        "where N is divisible by 4") 
                mats = [ mats[4*m : 4*(m+1),:] for m in range(mats.shape[0] // 4) ]
                    
        return MotionCorrection([ Registration.from_flirt(m, src, ref) for m in mats ])

    def __len__(self):
        return len(self.transforms)

    def __repr__(self):
        t = self[0]

        formatter = "{:8.3f}".format 
        with np.printoptions(precision=3, formatter={'all': formatter}):
            text = (f"""\
                MotionCorrection (linear) with properties:
                series length:   {len(self)}
                src2ref_0:       {t.src2ref[0,:]}
                                 {t.src2ref[1,:]}
                                 {t.src2ref[2,:]}
                                 {t.src2ref[3,:]}""")
        return dedent(text)

    def __getitem__(self, idx):
        """Access individual Registration objects from within series"""
        return self.__transforms[idx]

    @classmethod
    def identity(cls, length):
        return MotionCorrection([Registration.identity()] * length)

    @classmethod
    def from_registration(cls, reg, length):
        """
        Produce a MotionCorrection by repeating a Registration object 
        n times (eg, 10 copies of a single transform)
        """

        return MotionCorrection([reg.src2ref] * length)

    @property 
    def transforms(self):
        """List of Registration objects representing each volume of transform"""
        return self.__transforms

    @property 
    def src2ref(self):
        """List of src to ref transformation matrices"""
        return [ t.src2ref for t in self.transforms ]

    @property
    def ref2src(self):
        """List of ref to src transformation matrices"""
        return [ t.ref2src for t in self.transforms ]

    def to_fsl(self, src, ref):
        """Transformation matrices in FSL terms"""
        return [ t.to_fsl(src, ref) for t in self.transforms ]

    def save_txt(self, outdir, src=None, ref=None, convention="world", 
                 prefix="MAT_"):
        """
        Save individual transformation matrices in text format
        in outdir. Matrices will be named prefix_001... 

        Args: 
            outdir: directory in which to save 
            src: (optional) path to image, or ImageSpace, source space of
                transformation
            ref: as above, for reference space of transformation 
            convention: "world" or "fsl", if fsl then src/ref must be given
            prefix: prefix for naming each matrix
        """
        
        os.makedirs(outdir, exist_ok=True)
        for idx, r in enumerate(self.transforms):
            p = op.join(outdir, "MAT_{:04d}.txt".format(idx))
            r.save_txt(p, src, ref, convention)

    def resolve(self, src, ref, at_idx):
        """
        Return a coordinate array and scale factor that maps reference voxels
        into source voxels, including the transform. Uses cached values, if
        available. 

        Args: 
            src (ImageSpace): in which data currently exists and interpolation
                will be performed
            ref (ImageSpace): in which data needs to be expressed
            at_idx (int): index number within series of transforms to apply

        Returns: 
            (np.ndarray, 1) coordinates on which to interpolate and identity 
                scale factor
        """
        
        # Array of all voxel indices in the reference grid
        # Map them into world coordinates, apply the transform
        # and then into source voxel coordinates for the interpolation 
        ref2src_vox = (src.world2vox 
                       @ self.ref2src[at_idx]
                       @ ref.vox2world)
        ijk = apply.aff_trans(ref2src_vox, self.cache).T
        scale = 1
        return ijk, scale

class NonLinearRegistration(Transform):
    """
    Non linear registration transformation. Currently only FSL FNIRT warps
    are supported. Note that the --premat and --postmat used by FSL command
    line tools should not be supplied here. Instead, defined them as 
    Registration objects and use chain() to concatenate them with NLRs. 

    
    Args: 
        warp (path): FNIRT coefficient field 
        src (path/ImageSpace): source image used for generating FNIRT coefficients
        ref (path/ImageSpace): reference image used for generating FNIRT coefficients 
        intensity_correct: intensity correct output via the Jacobian determinant
            of this warp (when self.apply_to*() is called)
    """

    def __init__(self, warp, src, ref, intensity_correct=False):

        raise NotImplementedError("Currently only FNIRT supported, use "
                                "NonLinearRegistration.from_fnirt() instead")

        # Transform.__init__(self)
        # self.warp = FNIRTCoefficients(warp, src, ref)
        # self._intensity_correct = int(intensity_correct)

    @classmethod
    def from_fnirt(cls, coefficients, src, ref, intensity_correct=False):
        """
        FNIRT non-linear registration from a coefficients file. If a pre-warp
        and post-warp transformation need to be applied, create these as 
        separate Registration objects and combine them via chain, ie, 
        combined = chain(pre, non-linear, post)

        Args: 
            coefficients (str/nibabel NIFTI): FNIRT coefficients 
            src (str/ImageSpace): the source of the warp 
            ref (str/ImageSpace): the reference of the warp 
            intensity_correct (bool): whether to apply intensity correction via
                the determinant of the warp's Jacobian (default false)

        Returns: 
            NonLinearRegistration object 
        """

        warp = FNIRTCoefficients(coefficients, src, ref)
        return NonLinearRegistration._manual_construct(warp, 
            np.eye(4), np.eye(4), intensity_correct)

    @property
    def intensity_correct(self):
        return bool(self._intensity_correct)

    @intensity_correct.setter
    def intensity_correct(self, flag):
        self._intensity_correct = int(flag)

    def __len__(self):
        return 1

    @classmethod
    def _manual_construct(cls, warp, premat, postmat, 
                          intensity_correct=False):
        """Manual constructor, do not use from outside regtricks"""

        # # We store intensity correction as an integer private variable,
        # # as it can take the values 0,1,2,3 (this includes NonLinearMC subclass)
        # # 0: no intensity correction
        # # 1: intensity correction, or if the warp is a NonLinearProduct, then
        # #       intensity correct the FIRST warp 
        # # 2: intensity correct the second warp of a NLP 
        # # 3: intensity correct both warps of a NLP  
        
        x = cls.__new__(cls)
        assert isinstance(warp, (FNIRTCoefficients, NonLinearProduct))
        x.warp = warp
        x.premat = multiply.cast_potential_array(premat)
        x.postmat = multiply.cast_potential_array(postmat) 
        x.intensity_correct = int(intensity_correct)
        return x 

    def inverse(self):
        """Iverse warpfield, via FSL invwarp"""

        # TODO: lazy evaluation of this? And move into FNIRT coeffs somehow 

        with tempfile.TemporaryDirectory() as d:
            oldcoeffs = op.join(d, 'oldcoeffs.nii.gz')
            newcoeffs = op.join(d, 'newcoeffs.nii.gz')
            old_src = op.join(d, 'src.nii.gz')
            old_ref = op.join(d, 'ref.nii.gz')
            self.warp.src_spc.touch(old_src)
            self.warp.ref_spc.touch(old_ref)
            nibabel.save(self.warp.coefficients, oldcoeffs)
            cmd = 'invwarp -w {} -o {} -r {}'.format(oldcoeffs, 
                                                     newcoeffs, old_src)
            subprocess.run(cmd, shell=True)
            newcoeffs = nibabel.load(newcoeffs)
            newcoeffs.get_data()
            inv = NonLinearRegistration(newcoeffs, old_ref, old_src)
        return inv 

    def premat_to_fsl(self, src, ref): 
        """Return list of premats in FSL convention""" 

        if type(self.premat) is Registration: 
            return self.premat.to_fsl(src, ref)
        else: 
            assert type(self.premat) is list
            return [ t.to_fsl(src, ref) for t in self.premat ]

    def postmat_to_fsl(self, src, ref): 
        """Return list of postmats in FSL convention""" 

        if type(self.postmat) is Registration: 
            return self.postmat.to_fsl(src, ref)
        else: 
            assert type(self.postmat) is list
            return [ t.to_fsl(src, ref) for t in self.postmat ]

    def __repr__(self):
        text = (f"""\
        NonLinearRegistration with properties:
        """)
        return dedent(text)

    def prepare_cache(self, ref):
        """
        Pre-compute and store the displacement field, including any postmats. 
        This is because premats can be applied after calculating the field, 
        but postmats must be included as part of that calculation. Note that
        get_cache_value() return None, signifying that the field could not 
        be cached (which implies a NLMC)

        Args: 
            ref (ImageSapce): the space in towards which the transform will
                be applied 
        """
        self.cache = self.warp.get_cache_value(ref, self.postmat)
        if self.cache is None: 
            assert type(self) is NonLinearMotionCorrection

    def resolve(self, src, ref, *unused):
        """
        Return a coordinate array and scale factor that maps reference voxels
        into source voxels, including the transform. Uses cached values, if
        available.  A scale factor of 1 will be returned if no intensity
        correction was requested. 

        Args: 
            src (ImageSpace): in which data currently exists and interpolation
                will be performed
            ref (ImageSpace): in which data needs to be expressed

        Returns: 
            (np.ndarray, np.ndarray/int) coordinates on which to interpolate, 
                scaling factor to apply after interpolation 
        """

        ref2src_vox = (src.world2vox 
                       @ self.premat.ref2src 
                       @ self.warp.src_spc.FSL2world)

        if self.cache is not None: 
            ijk = apply.aff_trans(ref2src_vox, self.cache).T
        else: 
            raise RuntimeError("Should always be able to cache a NLR")

        if not self.intensity_correct: 
            scale = 1
        else: 

            # Either a single warp, or intensity correction from both warps. 
            # Either way, calculate detJ on the overall final displacement field, which is
            # given by dfield (including any reqd postmats)
            if (type(self.warp) is not NonLinearProduct) or (self._intensity_correct == 3): 
                scale = det_jacobian(self.cache.reshape(*ref.size, 3), ref.vox_size)

            # Intensity correct on second warp. Just calculate the displacement field
            # for the second warp and the corresponding postmat. 
            elif self._intensity_correct == 2: 
                dfield2 = self.warp.warp2.get_displacements(ref, self.postmat, 0)
                scale = det_jacobian(dfield2.reshape(*ref.size, 3), ref.vox_size)

            # Intensity correct on first warp. Calculate the displacement field on 
            # the first warp. Then calculate the successor transform: the midmat, 
            # the second warp, and the final postmat; and run the detJ through the 
            # successor transform 
            else: 
                assert self._intensity_correct == 1 
                dfield1 = self.warp.warp1.get_displacements(ref, Registration.identity(), 0)
                dj = det_jacobian(dfield1.reshape(*ref.size, 3), ref.vox_size)
                successor = NonLinearRegistration._manual_construct(self.warp.warp2, self.warp.warp2.src_spc, 
                    self.warp.warp2.ref_spc, premat=self.warp.midmat, postmat=self.postmat)
                scale = successor.apply_to_array(dj, ref, ref, cores=1, superlevel=1)

        return (ijk, scale)

class NonLinearMotionCorrection(NonLinearRegistration):
    """
    Only to be created by multiplication of other classes. Don't go here!

    Args: 
        warp: FNIRTCoefficients object or NonLinearProduct 
        src: src of transform
        ref: ref of transform
        premat: list of Registration objects
        postmat: list of Registration objects
        intensity_correct: int (0/1/2/3), whether to apply intensity
            correction, and at what stage in the case of NLPs
    """

    def __init__(self, warp, src, ref, premat, postmat, 
                 intensity_correct=0):
        
        Transform.__init__(self)
        self.warp = warp

        assert (isinstance(premat, (Registration, np.ndarray)) 
                or isinstance(postmat, (Registration, np.ndarray)))

        # Expand the pre/postmats to be MotionCorrections of equal length, 
        # if they are not already 
        if len(premat) > len(postmat):
            assert len(postmat) == 1, 'Different length pre/postmats given'
            postmat = MotionCorrection.from_registration(postmat, len(premat))
        
        elif len(postmat) > len(premat): 
            assert len(premat) == 1, 'Different length pre/postmats given'
            premat = MotionCorrection.from_registration(premat, len(postmat))

        else:
            if not len(premat) == len(postmat): 
                raise ValueError('Different length pre/postmats')

        # Likewise expand the midmat if we have a NLP as the warp 
        if (type(warp) is NonLinearProduct) and (len(warp.midmat) != len(premat)):
            if len(warp.midmat) == 1: 
                self.warp.midmat = MotionCorrection.from_registration(warp.midmat, len(premat))
            else: 
                raise ValueError("Different length pre/midmats")

        self.premat = multiply.cast_potential_array(premat)
        self.postmat = multiply.cast_potential_array(postmat) 
        if intensity_correct > 1 and (type(warp) is not NonLinearProduct): 
            raise ValueError("Intensity correction value implies NonLinearProduct")
        self._intensity_correct = intensity_correct

    def __len__(self):
        return len(self.premat)

    def __repr__(self):
        text = f"""\
                NonLinearMotionCorrection with properties:
                series length:   {len(self)}
                """
        return dedent(text)

    def resolve(self, src, ref, at_idx):
        """
        Return a coordinate array and scale factor that maps reference voxels
        into source voxels, including the transform. Uses cached values, if
        available. A scale factor of 1 will be returned if no intensity
        correction was requested. 

        Args: 
            src (ImageSpace): in which data currently exists and interpolation
                will be performed
            ref (ImageSpace): in which data needs to be expressed
            at_idx (int): index number within series of transforms to apply

        Returns: 
            (np.ndarray, np.ndarray/int) coordinates on which to interpolate, 
                scaling factor to apply after interpolation 
        """

        if self.cache is not None: 
            dfield = self.cache
        else: 
            dfield = self.warp.get_displacements(ref, self.postmat, at_idx)

        # Prepare the single overall transformation of premat and
        #  world/voxel matrices that is required for interpolation 
        ref2src_vox = (src.world2vox 
                        @ self.premat.ref2src[at_idx] 
                        @ self.warp.src_spc.FSL2world)
        ijk = apply.aff_trans(ref2src_vox, dfield).T

        if not self.intensity_correct: 
            scale = 1

        else: 

            # TODO: cache the intensity correction?
            # Either a single warp, or intensity correction from both warps. 
            # Either way, calculate detJ on the overall final displacement field, which is
            # given by dfield (including any reqd postmats)
            if (type(self.warp) is not NonLinearProduct) or (self._intensity_correct == 3): 
                scale = det_jacobian(dfield.reshape(*ref.size, 3), ref.vox_size)

            # Intensity correct on second warp. Just calculate the displacement field
            # for the second warp and the corresponding postmat. 
            elif self._intensity_correct == 2: 
                df = self.warp.warp2.get_displacements(ref, self.postmat[at_idx])
                scale = det_jacobian(df.reshape(*ref.size, 3), ref.vox_size)

            # Intensity correct on first warp. Calculate the displacement field on 
            # the first warp. Then calculate the successor transform: the midmat, 
            # the second warp, and the final postmat; and run the detJ through the 
            # successor transform 
            else: 
                assert self._intensity_correct == 1 
                df = self.warp.warp1.get_displacements(ref, Registration.identity())
                dj = det_jacobian(df.reshape(*ref.size, 3), ref.vox_size)
                successor = NonLinearRegistration._manual_construct(self.warp.warp2, self.warp.warp2.src_spc, 
                    self.warp.warp2.ref_spc, premat=self.warp.midmat[at_idx], postmat=self.postmat[at_idx])
                scale = successor.apply_to_array(dj, ref, ref, cores=1, superlevel=1)

        return (ijk, scale)
       