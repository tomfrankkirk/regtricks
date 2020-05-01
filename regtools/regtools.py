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


class Transform(object):
    """
    Base object for all transformations. This should never actually be 
    instantiated but is instead used to provide common functions
    """
    
    def __init__(self):
        raise NotImplementedError() 

    @property
    def src_header(self):
        """Nibabel header for the original source image, if present"""

        if self.src_spc is not None: 
            return self.src_spc.header 
        else: 
            return None 

    @property
    def ref_header(self):
        """Nibabel header for the original header image, if present"""

        if self.ref_spc is not None: 
            return self.ref_spc.header 
        else: 
            return None 

    def save(self, path):
        """Save transformation at path in X5 format (experimental)"""

        x5.save_manager(self, path)

    def inverse(self):
        """NB NonLinear classes explicitly override this"""

        constructor = type(self)
        return constructor(self.ref2src_world, src=self.ref_spc, 
                           ref=self.src_spc, convention='world')

    def __len__(self):
        """1 for Registrations, N for MotionCorrections of series length N"""

        if type(self) is Registration:
            return 1 
        else:
            return len(self.src2ref_world)

    def __repr__(self):
        raise NotImplementedError()

    # We need to explicitly not implement np.array_ufunc to allow overriding
    # of __matmul__, see: https://github.com/numpy/numpy/issues/9028
    __array_ufunc__ = None 

    def apply_to_image(self, src, ref, cores=1, **kwargs):
        """
        Applies transformation to image-like object and stores result within
        the voxel grid defined by ref (which can be the same as the src). If a 
        registration is applied to 4D data, the same transformation will be 
        applied to all volumes in the series. 

        Linear transformations are handled by scipy.ndimage.map_coordinates. 
        Nonlinear transformations are handled by FSL applywarp. 

        Args:   
            src: str, nibabel Nifti/MGH, or FSL Image obejct; data to transform
                (can be 3D or 4D). 
            ref: any of the same types as src, or ImageSpace. NB src can also 
                be used as the ref (transform the data but keep result in same 
                voxel grid)
            cores: CPU cores to use for 4D volumes (not for applywarp) 
            **kwargs: passed on to scipy.ndimage.map_coordinates

        Returns: 
            an Image object of same type as passed in, or Nifti by default
        """

        data, create = apply.src_load_helper(src)
        resamp = self.apply_to_array(data, src, ref, cores, **kwargs)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        
        if create is MGHImage:
            ret = MGHImage(resamp, ref.vox2world, ref.header)
            return ret 
        else: 
            ret = Nifti2Image(resamp, ref.vox2world, ref.header)
            if create is FSLImage:
                return FSLImage(ret)
            else: 
                return ret 

    def apply_to_array(self, data, src, ref, cores=1, **kwargs):
        """
        Applies transformation to data array. If a registration is applied 
        to 4D data, the same transformation will be applied to all volumes 
        in the series. 

        Linear transformations are handled by scipy.ndimage.map_coordinates. 
        Nonlinear transformations are handled by FSL applywarp. 

        Args:   
            data: 3D or 4D array. 
            src: str, nibabel Nifti/MGH, or FSL Image obejct, or ImageSpace
                object defining the space the data is currently within
            ref: as above, defining the space within which the data should
                be returned 
            cores: CPU cores to use for 4D data (not for applywarp)
            **kwargs: passed on to scipy.ndimage.map_coordinates

        Returns: 
            np.array of transformed image data in ref voxel grid.
        """

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        if not (data.shape[:3] == src.size).all(): 
            raise RuntimeError("Data shape does not match source space")

        # Linear transformation: scipy interpolation (quicker)
        if type(self) in (Registration, MotionCorrection):
            resamp = apply._application_worker(data, self, src, ref, 
                                            cores, **kwargs)

        # Nonlinear transformation: applywarp interpolation 
        elif type(self) in (NonLinearRegistration, 
                               NonLinearMotionCorrection):

            if type(self) is NonLinearMotionCorrection:
                assert len(self) == data.shape[-1]
                premat = np.concatenate(
                    self.premat_to_fsl(src, self.fcoeffs.src_spc), 0)
                postmat = np.concatenate(
                    self.postmat_to_fsl(self.fcoeffs.ref_spc, ref), 0)
            else: 
                premat = self.premat_to_fsl(src, self.fcoeffs.src_spc)
                postmat = self.postmat_to_fsl(self.fcoeffs.ref_spc, ref)

            resamp = apply.applywarp_helper(self.fcoeffs, premat, postmat, 
                        data, src, ref)

        return resamp      


class Registration(Transform):
    """
    Represents a transformation between the source image and reference.
    If src and ref are given, the transformation is assumed to be in 
    FLIRT/FSL convention, otherwise it is assumed to be in world convention.

    Args: 
        src2ref: either a 4x4 np.array representing affine transformation
            from source to reference, or a path to a text-like file 
        src: (optional) either the path to the source image, or an ImageSpace
            object initialised with the source 
        ref: (optional) either the path to the reference image, or an 
            ImageSpace object initialised with the referende 
        convention: (optional) either "world" (assumed if src/ref not given),
            or "fsl" (assumed if src/ref given)
    """

    def __init__(self, src2ref, src=None, ref=None, convention=""):

        if isinstance(src2ref, str): 
            src2ref = np.loadtxt(src2ref)

        if (src2ref.shape != (4,4) 
                or (np.abs(src2ref[3,:] - [0,0,0,1]) > 1e-9).any()):
            raise RuntimeError("src2ref must be a 4x4 affine matrix, where ",
                               "the last row is [0,0,0,1].")

        if (src is not None) and (ref is not None):  
            if not isinstance(src, ImageSpace):
                src = ImageSpace(src)
            self.src_spc = src 
            if not isinstance(ref, ImageSpace):
                ref = ImageSpace(ref)
            self.ref_spc = ref 

            if convention == "":
                print("Assuming FSL convention")
                convention = "fsl"

        else: 
            self.src_spc = None
            self.ref_spc = None 
            if convention == "":
                print("Assuming world convention")
                convention = "world"

        if convention.lower() == "fsl":
            src2ref_world = (self.ref_spc.FSL2world 
                             @ src2ref @ self.src_spc.world2FSL)

        elif convention.lower() == "world":
            src2ref_world = src2ref 

        else: 
            raise RuntimeError("Unrecognised convention")

        self.__src2ref_world = src2ref_world


    def __repr__(self):
        s = self._repr_helper(self.src_spc)
        r = self._repr_helper(self.ref_spc)
        
        formatter = "{:8.3f}".format 
        with np.printoptions(precision=3, formatter={'all': formatter}):
            text = (f"""\
                Registration (linear) with properties:
                source:        {s}, 
                reference:     {r}, 
                src2ref_world: {self.src2ref_world[0,:]}
                               {self.src2ref_world[1,:]}
                               {self.src2ref_world[2,:]}
                               {self.src2ref_world[3,:]}""")
        return dedent(text)


    def _repr_helper(self, spc):
        if not spc: 
            return "(none defined)"
        elif spc.file_name: 
            return self.src_spc.file_name
        else:  
            return "ImageSpace object"


    def __matmul__(self, other):
        """
        Matrix multiplication of registrations and motion corrections. The 
        order of arguments follows matrix conventions: to get the transform 
        AB, the multiplication needs to be B @ A. Any multiplication between
        a registration and motion correction will cause the result to also be 
        a motion correction. 

        Accepted types: 4x4 np.array, registration, motion correction
        """

        # Cast 4x4 arrays to Registrations
        other = cast_potential_array(other)

        # Check for type promotion 
        if isinstance(other, (MotionCorrection, NonLinearRegistration)):
            return other.__rmatmul__(self)

        overall_world = self.src2ref_world @ other.src2ref_world
        return Registration(overall_world, other.src_spc, self.ref_spc,
                            "world")

    def __rmatmul__(self, other):

        # Cast 4x4 arrays to Registrations
        other = cast_potential_array(other)
        
        # Check for type promotion 
        if isinstance(other, (MotionCorrection, NonLinearRegistration)):
            return other.__matmul__(self)

        return other @ self 
    
    @property
    def ref2src_world(self):
        return np.linalg.inv(self.__src2ref_world)

    @property
    def src2ref_world(self):
        return self.__src2ref_world

    @classmethod
    def identity(cls, src=None, ref=None):
        return Registration(np.eye(4), src, ref, convention="world")

    @classmethod
    def eye(cls, src=None, ref=None):
        return Registration.identity(src, ref)

    def to_fsl(self, src, ref):
        """
        Return transformation in FSL convention, for given src and ref, 
        as np.array. This will be 3D in the case of MotionCorrections
        """

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        return ref.world2FSL @ self.src2ref_world @ src.FSL2world


    def save_txt(self, path, src=None, ref=None, convention="world"):
        if convention.lower() == "fsl":
            np.savetxt(path, self.to_fsl(src, ref))
        else: 
            np.savetxt(path, self.src2ref_world)


    def apply_to_grid(self, src):
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
        new_spc = src_spc.transform(self.src2ref_world)
               
        if create is MGHImage:
            ret = MGHImage(data, new_spc.vox2world, new_spc.header)
            return ret 
        else: 
            ret = Nifti2Image(data, new_spc.vox2world, new_spc.header)
            if create is FSLImage:
                return FSLImage(ret)
            else: 
                return ret 


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
        src: (optional) either the path to the source image, or an ImageSpace
            object representing the source 
        ref: (optional) either the path to the reference image, or an Image
            Space representing the source (NB this is usually the same as 
            the src image)
        convention: (optional) the convention used for each transformation
            (if src and ref are given, 'fsl' is assumed, otherwise 'world')
    """

    def __init__(self, mats, src=None, ref=None, convention=None):

        if isinstance(mats, str):
            mats = sorted(glob.glob(op.join(mats, '*')))
            if not mats: 
                raise RuntimeError("Did not find any matrices in %s" % mats)

        if not convention: 
            if (src is not None):
                print("Assuming FSL convention")
                convention = "fsl"
            else: 
                print("Assuming world convention")
                convention = "world"
            
        self.__transforms = []
        for mat in mats:
            if isinstance(mat, (np.ndarray, str)): 
                m = Registration(mat, src, ref, convention)
            else: 
                m = mat 
            self.__transforms.append(m)


    def __repr__(self):
        t = self.transforms[0]
        s = self._repr_helper(self.src_spc)
        r = self._repr_helper(self.ref_spc)

        formatter = "{:8.3f}".format 
        with np.printoptions(precision=3, formatter={'all': formatter}):
            text = (f"""\
                MotionCorrection (linear) with properties:
                source:          {s}, 
                reference:       {r}, 
                series length:   {len(self)}
                src2ref_world_0: {t.src2ref_world[0,:]}
                                 {t.src2ref_world[1,:]}
                                 {t.src2ref_world[2,:]}
                                 {t.src2ref_world[3,:]}""")
        return dedent(text)

    def __matmul__(self, other):
        """In transformation terms, apply other first and then self"""

        # Cast 4x4 arrays to Registrations
        other = cast_potential_array(other)

        # Type promotion: allow multiplication to be handled by 
        # highest available class 
        if isinstance(other, (NonLinearRegistration)):
            return other.__matmul__(self)

        if type(other) is Registration:
            other = MotionCorrection.from_registration(other, len(self))

        elif not len(self) == len(other):
            raise RuntimeError("MotionCorrections must be of equal length")

        elif type(other) is NonLinearMotionCorrection:
            raise NotImplementedError("Cannot chain linear and non linear ",
                                      "MotionCorrections")
        
        world_mats = [ m1 @ m2 for m1,m2 in 
                       zip(self.src2ref_world, other.src2ref_world) ]
        ret = MotionCorrection(world_mats, other.src_spc, self.ref_spc, 
                               convention="world")

        return ret 


    def __rmatmul__(self, other):
        """In transformation terms, apply self first and then other"""

        # Cast 4x4 arrays to Registrations
        other = cast_potential_array(other)

        # Type promotion: allow multiplication to be handled by 
        # highest available class 
        if isinstance(other, (NonLinearRegistration)):
            return other.__matmul__(self)

        elif type(other) is Registration:
            other = MotionCorrection.from_registration(other, len(self))

        return other @ self


    @classmethod
    def from_registration(cls, reg, length):
        """
        Produce a MotionCorrection by repeating a Registration object 
        n times (eg, 10 copies of a single transform)
        """

        return MotionCorrection([reg.src2ref_world] * length,
                                 reg.src_spc, reg.ref_spc, "world")

    @property 
    def transforms(self):
        """List of Registration objects representing each volume of transform"""
        return self.__transforms

    @property 
    def src2ref_world(self):
        """List of src to ref transformation matrices"""
        return [ t.src2ref_world for t in self.transforms ]

    @property
    def ref2src_world(self):
        """List of ref to src transformation matrices"""
        return [ t.ref2src_world for t in self.transforms ]

    @property
    def src_spc(self):
        """ImageSpace for source of transform"""
        return self.transforms[0].src_spc 

    @property
    def ref_spc(self):
        """ImageSpace for reference of transform"""
        return self.transforms[0].ref_spc

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


class NonLinearRegistration(Transform):
    """Non linear registration transformation"""

    def __init__(self, coefficients, src, ref, premat=np.eye(4), postmat=np.eye(4)):

        self.fcoeffs = FNIRTCoefficients(coefficients, src, ref)

        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        self.ref_spc = ref 

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        self.src_spc = src 

        self.premat = Registration(np.eye(4), src, ref, "world")
        self.postmat = Registration(np.eye(4), src, ref, "world")

    @classmethod
    def _manual_construct(cls, fcoeffs, src, ref, premat, postmat):
        x = cls.__new__(cls)
        x.fcoeffs = fcoeffs
        x.src_spc = src 
        x.ref_spc = ref 
        assert type(premat) is Registration
        assert type(postmat) is Registration
        x.premat = premat 
        x.postmat = postmat 
        return x 

    def inverse(self):
        """Iverse warpfield, via FSL invwarp"""
        with tempfile.TemporaryDirectory() as d:
            oldcoeffs = op.join(d, 'oldcoeffs.nii.gz')
            newcoeffs = op.join(d, 'newcoeffs.nii.gz')
            old_src = op.join(d, 'src.nii.gz')
            old_ref = op.join(d, 'ref.nii.gz')
            self.src.touch(old_src)
            self.ref.touch(old_ref)
            nibabel.save(self.coefficients, oldcoeffs)
            cmd = 'invwarp -w {} -o {} -r {}'.format(oldcoeffs, newcoeffs, old_src)
            subprocess.run(cmd, shell=True)
            newcoeffs = nibabel.load(newcoeffs)
            newcoeffs.get_data()
            inv = NonLinearRegistration(newcoeffs, old_ref, old_src)
        return inv 

    def premat_to_fsl(self, src, ref): 
        if type(self.premat) is Registration: 
            return self.premat.to_fsl(src, ref)
        else: 
            assert type(self.premat) is list
            return [ t.to_fsl(src, ref) for t in self.premat ]

    def postmat_to_fsl(self, src, ref): 
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

    def __matmul__(self, other):
        """
        In transformation terms, this is equivalent to applying other first
        and then self, so we modify the premat and return a new NL object 
        """

        if type(other) is Registration: 
            pre = self.premat @ other
            return NonLinearRegistration._manual_construct(
                    self.fcoeffs, other.src_spc, self.ref_spc, pre, self.postmat)

        elif type(other) is MotionCorrection: 
            premats = [ self.premat @ m for m in other.transforms ]
            postmats = [ Registration.identity() ] * len(premats)
            return NonLinearMotionCorrection(self.fcoeffs, other.src_spc, 
                                             self.ref_spc, premats, postmats)

        elif type(other) is NonLinearRegistration: 

            # pre of other, then warp field, then post of other, pre of self
            # warp of self, post of self. Done! 

            pre = other.premat.to_fsl(other.premat.src_spc, other.fcoeffs.src_spc)
            mid = (self.premat @ other.postmat).to_fsl(other.fcoeffs.ref_spc, self.fcoeffs.src_spc)
            post = self.postmat.to_fsl(self.fcoeffs.ref_spc, self.postmat.ref_spc)
            warp1 = other.fcoeffs.coefficients
            warp2 = self.fcoeffs.coefficients

            with tempfile.TemporaryDirectory() as d: 
                out = op.join(d, 'newwarp.nii.gz')
                refspc = op.join(d, 'newref.nii.gz')
                self.ref_spc.touch(refspc)
                w1 = op.join(d, 'warp1.nii.gz')
                nibabel.save(warp1, w1)
                w2 = op.join(d, 'warp2.nii.gz')
                nibabel.save(warp2, w2)
                premat = op.join(d, 'premat')
                np.savetxt(premat, pre)
                midmat = op.join(d, 'midmat')
                np.savetxt(midmat, mid)
                postmat = op.join(d, 'postmat')
                np.savetxt(postmat, post)
                cmd = (f"convertwarp -o {out} -r {refspc} --premat={premat} "
                        f"--warp1={w1} --midmat={midmat} --warp2={w2} "
                        f"--postmat={postmat} ")
                subprocess.run(cmd, shell=True)
                new = nibabel.load(out)
                new.get_data()
            return NonLinearRegistration(new, other.src_spc, self.ref_spc)

        else: 
            raise NotImplementedError()

    def __rmatmul__(self, other):
        """
        In transformation terms, this is equivalent to applying self first
        and then other, so we modify the post and return a new NL object 
        """

        if type(other) is Registration: 
            post = other @ self.postmat
            return NonLinearRegistration._manual_construct(
                self.fcoeffs,
                    self.src_spc, other.ref_spc, self.premat, post)
              
        elif type(other) is MotionCorrection: 
            postmats = [ m @ self.postmat for m in other.transforms ]
            premats = [ Registration.identity() ] * len(postmats)
            return NonLinearMotionCorrection(self.fcoeffs, self.src_spc, 
                                             other.ref_spc, premats, postmats)

        elif type(other) is NonLinearRegistration: 
            raise RuntimeError("This should be handled by the other")

        else: 
            raise NotImplementedError()


class NonLinearMotionCorrection(NonLinearRegistration):
    """
    Only to be created by multiplication of other classes. 

    Args: 
        fcoeffs: FNIRT coefficients object 
        src: src of transform
        ref: ref of transform
        premats: list of Registration objects
        postmats: list of Registration objects
    """

    def __init__(self, fcoeffs, src, ref, premats, postmats):
        
        self.fcoeffs = fcoeffs

        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        self.ref_spc = ref 

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        self.src_spc = src 

        assert (type(premats) is list) or (type(postmats) is list)
        if not premats: 
            premats = len(postmats) * [np.eye(4)]
        elif not premats: 
            postmats = len(premats) * [np.eye(4)]
        self.premat = premats 
        self.postmat = postmats 

    def __len__(self):
        return len(self.premat)

    def _cast_to_nonlinear_registration(self):
        return NonLinearRegistration(self.fcoeffs.coefficients, self.src_spc, self.ref_spc)

    def __repr__(self):
        text = f"""\
                NonLinearMotionCorrection with properties:
                source:          {self.src_spc}, 
                reference:       {self.ref_spc}, 
                series length:   {len(self)}
                """
        return dedent(text)

    def __matmul__(self, other):

        # TODO: we can handle Registrations here 

        raise NotImplementedError("Cannot be interpreted")
        # Extract the warps, combine them 
        # Deal with the pre and post mats 
        # singular = self._cast_to_nonlinear_registration()
        # warp = singular @ other 
        # premat = [ m @ other.premat for m in self.premat ]
        # return NonLinearMotionCorrection(warp.fcoeffs, warp.src_spc, warp.ref_spc, premat, self.postmat)

    def __rmatmul__(self, other):

        # TODO: we can handle Registrations here 

        raise NotImplementedError("Cannot be interpreted")
        # Extract the warps, combine them 
        # Deal with the pre and post mats 
        # singular = self._cast_to_nonlinear_registration()
        # warp = other @ singular 
        # postmat = [ warp.postmat @ m for m in self.postmat ]
        # return NonLinearMotionCorrection(warp.fcoeffs, warp.src_spc, warp.ref_spc, self.premat, postmat)



class FNIRTCoefficients(object):
    """
    Private encapsulation of FNIRT warp field. Only to be used 
    from within a NonLinearTransformation

    Args: 
        coeffs; nibabel object or path to coefficients file
        src: str or ImageSpace, path to original source for transform
        ref: str or ImageSpace, path to original reference for transform
    """

    def __init__(self, coeffs, src, ref):
        if isinstance(coeffs, str):
            self.coefficients = nibabel.load(coeffs)
        else: 
            assert isinstance(coeffs, nibabel.Nifti1Image)
            self.coefficients = coeffs 

        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        self.ref_spc = ref 

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        self.src_spc = src 


def chain(*args):
    """ 
    Concatenate a series of registrations.

    Args: 
        *args: Registration objects, given in the order that they need to be 
            applied (eg, for A -> B -> C, give them in that order and they 
            will be multiplied as C @ B @ A)

    Returns: 
        Registration object, with the first registration's source 
        and the last's reference (if these are not None)
    """

    if (len(args) == 1):
        chained = args
    else: 

        # As we cannot multiply two NLMCs together, if we find two adjacent
        # NLRs in sequence then multiply them together first (in anticpation
        # that they may later be promoted to MCs by other items in the chain.
        # The below block picks out adjacent NLs, handles them, and inserts
        # the result back into the appropriate spot 
        args = list(args)
        while True: 
            did_update = False 
            for idx in range(len(args)-2):
                if ((type(args[idx]) is NonLinearRegistration) 
                    and (type(args[idx+1]) is NonLinearRegistration)):
                    combined = chain(args[idx], args[idx+1])
                    # combined = Registration.identity()
                    args = args[:idx] + [combined] + args[idx+2:] 
                    did_update = True 
                    break 
            if not did_update: 
                break 

        if not all([isinstance(r, Transform) for r in args ]):
            raise RuntimeError("Each item in sequence must be a",
                               " Registration, MotionCorrection or NonLinearRegistration")
                               
        # We do the first pair explicitly (in case there are only two)
        # and then we do all others via pre-multiplication 
        chained = args[1] @ args[0]
        for r in args[2:]:
            chained = r @ chained 

    return chained 


def cast_potential_array(arr):

    # Cast 4x4 arrays to Registrations
    if type(arr) is np.ndarray: 
        assert arr.shape == (4,4)
        arr = copy.deepcopy(arr)
        arr = Registration(arr)
    return arr 