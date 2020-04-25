import os.path as op 
import glob 
import os 
from textwrap import dedent

from nibabel import Nifti2Image, MGHImage
import numpy as np 
from fsl.data.image import Image as FSLImage

from .image_space import ImageSpace
from . import x5_interface as x5 
from . import application_helpers as apply 


class Transform(object):
    """
    Base object for all transformations. This should never actually be 
    instantiated but is instead used to provide common functions
    """
    
    def __init__():
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
        constructor = type(self)
        return constructor(self.ref2src_world, src=self.ref_spc, 
                           ref=self.src_spc, convention='world')

    def __len__(self):
        if type(self) is Registration:
            return 1 
        else:
            return len(self.src2ref_world)

    def __repr__(self):
        raise NotImplementedError()

    # Allow overriding of the form (other @ self) - as the other will not 
    # know how to interpret this object, __rmatmul__ will instead be called
    # on self. The actual multiplication is handled by matmul, below.
    def __rmatmul__(self, other):
        if type(other) is np.ndarray: 
            oth = Registration(other)
        else: 
            # Ininstance() will match against Registrations as well here 
            assert isinstance(other, (MotionCorrection, Registration))
        return oth @ self

    # We need to explicitly not implement np array_ufunc to allow overriding
    # of __matmul__, see: https://github.com/numpy/numpy/issues/9028
    __array_ufunc__ = None 

    def __matmul__(self, other):
        """
        Matrix multiplication of registrations and motion corrections. The 
        order of arguments follows matrix conventions: to get the transform 
        AB, the multiplication needs to be B @ A. Any multiplication between
        a registration and motion correction will cause the result to also be 
        a motion correction. 

        Accepted types: 4x4 np.array, registration, motion correction
        """

        # If we have a 4x4 array, cast it to Registration 
        if type(other) is np.ndarray:
            other = Registration(other)

        # Both are motion corrections 
        if ((type(self) is MotionCorrection) 
                and (type(other) is MotionCorrection)):

            if not len(self) == len(other):
                raise RuntimeError("MotionCorrections must be of equal length")
            
            world_mats = [ m1 @ m2 for m1,m2 in 
                           zip(self.src2ref_world, other.src2ref_world) ]
            ret = MotionCorrection(world_mats, other.src_spc, self.ref_spc, 
                                   convention="world")

        # One Registration, one MotionCorrection
        elif ((type(self) is MotionCorrection and type(other) is Registration) 
              or (type(other) is MotionCorrection) and type(self) is Registration):
            
            pre = Registration.identity(other.src_spc, self.ref_spc)
            post = Registration.identity(other.src_spc, self.ref_spc)
            if type(self) is Registration: 
                pre = self
                moco = other 
            else: 
                assert type(other) is Registration
                moco = self 
                post = other

            world_mats = [ pre @ m @ post for m in moco.transforms ]
            ret = MotionCorrection(world_mats, other.src_spc, self.ref_spc,
                                   convention="world")

        # Two registrations 
        elif ((type(self) is Registration) and (type(other) is Registration)): 
            overall_world = self.src2ref_world @ other.src2ref_world
            ret = Registration(overall_world, other.src_spc, self.ref_spc, 
                               convention="world")

        else: 
            raise RuntimeError("Unsupported argument types")

        return ret 

    def apply_to_image(self, src, ref, cores=1, **kwargs):
        """
        Applies transformation to image-like object and stores result within
        the voxel grid defined by ref (which can be the same as the src). 
        Uses scipy.ndimage.interpolation.map_coordinates, see that 
        documentation for **kwargs. If a registration is applied to 4D data, 
        the same transformation will be applied to all volumes in the series. 

        Args:   
            src: str, nibabel Nifti/MGH, or FSL Image obejct; data to transform
                (can be 3D or 4D). 
            ref: any of the same types as src, or ImageSpace. NB src can also 
                be used as the ref (transform the data but keep result in same 
                voxel grid)
            cores: CPU cores to use for 4D volumes 
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
        Applies registration transform to data array. Uses scipy.ndimage.
        interpolation.map_coordinates, see that documentation for **kwargs. 
        If a registration is applied to 4D data, the same transformation 
        will be applied to all volumes in the series. 

        Args:   
            data: 3D or 4D array. 
            src: str, nibabel Nifti/MGH, or FSL Image obejct, or ImageSpace
                object defining the space the data is currently within
            ref: as above, defining the space within which the data should
                be returned 
            cores: CPU cores to use for 4D data 
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
        resamp = apply._application_worker(data, self, src, ref, 
                                           cores, **kwargs)

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
                Registration with properties:
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
                MotionCorrection with properties:
                source:          {s}, 
                reference:       {r}, 
                series length:   {len(self)}
                src2ref_world_0: {t.src2ref_world[0,:]}
                                 {t.src2ref_world[1,:]}
                                 {t.src2ref_world[2,:]}
                                 {t.src2ref_world[3,:]}""")
        return dedent(text)

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




def load(path):
    """
    Load X5 format transformation (linear registration and moco supported)
    """

    return x5.load_manager(path)


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

    if (len(args) == 1) and (type(args) is Registration):
        chained = args
    else: 
        if not all([isinstance(r, Registration) for r in args ]):
            raise RuntimeError("Each item in sequence must be a",
                               " Registration or MotionCorrection.")
        chained = args[1] @ args[0]
        for r in args[2:]:
            chained = r @ chained 

    return chained 
