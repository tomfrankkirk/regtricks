import os.path as op 
import tempfile 
import subprocess
from textwrap import dedent

import nibabel
import numpy as np 

from regtricks.transforms.transform import Transform
from regtricks.transforms.linear import Registration, MotionCorrection
from regtricks.image_space import ImageSpace
from regtricks import application_helpers as apply
from regtricks import multiplication as multiply
from regtricks.fnirt_coefficients import (FNIRTCoefficients, NonLinearProduct, 
                                          det_jacobian)

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
        constrain_jac (bool/array-like): constrain Jacobian for intensity
            correction (default False). If True, limits of (0.01, 100) will 
            be used, or explicit limits can be given as (min, max)
    """

    def __init__(self, warp, src, ref, intensity_correct=False, 
                 constrain_jac=False):

        raise NotImplementedError("Currently only FNIRT supported, use "
                                "NonLinearRegistration.from_fnirt() instead")

        # Transform.__init__(self)
        # self.warp = FNIRTCoefficients(warp, src, ref)
        # self._intensity_correct = int(intensity_correct)

    @classmethod
    def from_fnirt(cls, coefficients, src, ref, intensity_correct=False, 
                   constrain_jac=False):
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
            constrain_jac (bool/array-like): constrain Jacobian for intensity
                correction (default False). If True, limits of (0.01, 100) will 
                be used, or explicit limits can be given as (min, max)

        Returns: 
            NonLinearRegistration object 
        """

        warp = FNIRTCoefficients(coefficients, src, ref, constrain_jac)
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
                          intensity_correct):
        """Manual constructor, do not use from outside regtricks"""

        # # We store intensity correction as an integer private variable,
        # # as it can take the values 0,1,2,3 (this includes NonLinearMC subclass)
        # # 0: no intensity correction
        # # 1: intensity correction, or if the warp is a NonLinearProduct, then
        # #       intensity correct the FIRST warp 
        # # 2: intensity correct the second warp of a NLP 
        # # 3: intensity correct both warps of a NLP  
        
        x = cls.__new__(cls)
        Transform.__init__(x)
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
            newcoeffs.get_fdata()
            inv = NonLinearRegistration.from_fnirt(newcoeffs, old_ref, old_src)
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
                dfield2 = self.warp.warp2.get_displacements(ref, self.postmat)
                scale = det_jacobian(dfield2.reshape(*ref.size, 3), ref.vox_size)

            # Intensity correct on first warp. Calculate the displacement field on 
            # the first warp. Then calculate the successor transform: the midmat, 
            # the second warp, and the final postmat; and run the detJ through the 
            # successor transform 
            else: 
                assert self._intensity_correct == 1 
                dfield1 = self.warp.warp1.get_displacements(ref, Registration.identity())
                dj = det_jacobian(dfield1.reshape(*ref.size, 3), ref.vox_size)
                successor = NonLinearRegistration._manual_construct(self.warp.warp2, premat=self.warp.midmat, 
                    postmat=self.postmat, intensity_correct=False)
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
        constrain_jac (bool/array-like): constrain Jacobian for intensity
            correction (default False). If True, limits of (0.01, 100) will 
            be used, or explicit limits can be given as (min, max)
    """

    def __init__(self, warp, premat, postmat, intensity_correct=0, 
                 constrain_jac=False):
        
        super().__init__()
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
            at_idx (int): index number within MC series of transforms to apply

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
                df = self.warp.warp2.get_displacements(ref, self.postmat, at_idx)
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
       