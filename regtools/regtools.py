import os.path as op 
import glob 
import multiprocessing as mp
import functools 

import nibabel 
import numpy as np 
from scipy.interpolate import interpn
from scipy.ndimage.interpolation import map_coordinates
from toblerone import utils 

from .image_space import ImageSpace
# ImageSpace = image_space.ImageSpace()

class Registration(object):
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

        if (src2ref.shape != (4,4) or 
            (np.abs(src2ref[3,:] - [0,0,0,1]) < 1-9).all()):
            raise RuntimeError("src2ref must be a 4x4 affine matrix, where " + 
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
            self.__src2ref_world = (self.ref_spc.FSL2world 
                    @ src2ref @ self.src_spc.world2FSL )

        elif convention.upper() == "world":
            self.__src2ref_world = src2ref 

        else: 
            raise RuntimeError("Unrecognised convention")


    @property
    def src_header(self):
        """Nibabel header for the original source image"""
        if self.src_spc is not None: 
            return self.src_spc.header 
        else: 
            return None 

    @property
    def ref_header(self):
        """Nibabel header for the original header image"""
        if self.ref_spc is not None: 
            return self.ref_spc.header 
        else: 
            return None 
        
    @property
    def ref2src_world(self):
        return np.linalg.inv(self.__src2ref_world)

    @property
    def src2ref_world(self):
        return self.__src2ref_world

    def inverse(self):
        return Registration(self.ref2src_world, src=self.ref_spc, 
            ref=self.src_spc, convention='world')

    def to_fsl(self, src, ref):
        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        return ref.world2FSL @ self.src2ref_world @ src.FSL2world

    def save_fsl(self, src, ref, path):
        np.savetxt(path, self.to_fsl(src, ref))
        
    def save_world(self, fname):
        np.savetxt(fname, self.src2ref_world)


    def apply_to(self, src, ref, out='', dtype=None, cores=1, **kwargs):
        """
        Apply registration transform to image. Uses scipy.ndimage.interpolation.
        map_coordinates, see that documentation for valid **kwargs. 

        Args:   
            src: either a nibabel Image object, or path to image file, 
                data to transform  
            ref: either a nibabel Image object, ImageSpace object, or
                path to image file, reference voxel grid 
            out: (optional) path to save output at 
            dtype: (optional) output datatype (default same as input)
            **kwargs: passed on to scipy.ndimage.map_coordinates

        Returns: 
            np.array of transformed image data in ref voxel grid.
        """

        if isinstance(src, str):
            src = nibabel.load(src)
        elif not isinstance(src, nibabel.Nifti1Image):
            raise RuntimeError("src must be a nibabel Nifti or path to image")
        src_spc = ImageSpace(src)

        if not isinstance(ref, ImageSpace):
                ref = ImageSpace(ref)
        else: raise RuntimeError("ref must be a nibabel Nifti, ImageSpace, or path")

        img = src.get_fdata().astype(src.get_data_dtype())
        if not dtype: 
            dtype = src.get_data_dtype()
        resamp = _application_worker(img, self.ref2src_world, src_spc, 
            ref, order, cores, **kwargs)

        if out: 
            ref.save_image(resamp, out)

        return resamp 

    def __mul__(self, other):

        src = other.src_spc    
        ref = self.ref_spc 

        if ((type(self) is MotionCorrection) 
            and (type(other) is MotionCorrection)):
            
            world_mats = [ m1 * m2 for m1,m2 in 
                zip(self.src2ref_world_mats, other.src2ref_world_mats) ]
            ret = MotionCorrection(world_mats, src, ref, convention="world")

        elif ((type(self) is MotionCorrection) 
            or (type(other) is MotionCorrection)):
            
            pre = Registration.identity()
            post = Registration.identity()
            if type(self) is Registration: 
                pre = self
                moco = other 
            else: 
                assert type(other) is Registration
                moco = self 
                post = other

            world_mats = [ pre * m * post for m in moco.transforms ]
            ret = MotionCorrection(world_mats, src, ref, convention="world")

        else: 
            overall_world = self.src2ref_world @ other.src2ref_world
            ret = Registration(overall_world, src, ref, "world")

        return ret 

    @classmethod
    def identity(cls):
        return Registration(np.eye(4), convention="world")

    @classmethod
    def eye(cls):
        return Registration.identity()


class MotionCorrection(Registration):

    def __init__(self, mats, src, ref, convention=""):

        if isinstance(mats, str):
            mats = glob.glob(op.join(mats, '*'))
            if not mats: 
                raise RuntimeError("Did not find any matrices in %s" % mats)

        if not convention: 
            if (src is not None) and (ref is not None):
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


    @property
    def src_spc(self): 
        return self.transforms[0].src_spc

    @property
    def ref_spc(self):
        return self.transforms[0].ref_spc

    @property
    def transforms(self):
        return self.__transforms

    @property 
    def src2ref_world_mats(self):
        return [ t.src2ref_world for t in self.__transforms ]

    @property
    def ref2src_world_mats(self):
        return [ t.ref2src_world for t in self.__transforms ]

    def apply_to(self, src, ref, out='', order=1, dtype=None, 
        cores=mp.cpu_count(), **kwargs):
        
        if isinstance(src, str):
            src = nibabel.load(src)
        elif not isinstance(src, nibabel.Nifti1Image):
            raise RuntimeError("src must be a nibabel Nifti or path to image")
        src_spc = ImageSpace(src.get_filename())

        if isinstance(ref, str):
            ref = ImageSpace(ref)
        elif isinstance(ref, nibabel.Nifti1Image):
            ref = ImageSpace(ref.get_filename())
        elif not isinstance(ref, ImageSpace):
            raise RuntimeError("ref must be a nibabel Nifti, ImageSpace, or path")

        if not dtype:
            dtype=src.get_data_dtype()
        img = src.get_fdata().astype(dtype)

        assert len(img.shape) == 4, "Image is not 4D"
        img = np.moveaxis(img, 3, 0)

        if not len(self.transforms) == img.shape[0]:
            raise RuntimeError("Number of motion correction matrices does" +
                "not match length in series.")

        worker = functools.partial(_application_worker, 
            src_spc=src_spc, ref_spc=ref, cores=1, **kwargs)
        work_list = zip(img, self.ref2src_world_mats)
        if cores == 1:
            resamp = np.stack([ worker(*fm) for fm in work_list ], 3)
        else: 
            with mp.Pool() as p: 
                resamp =  np.stack(p.starmap(worker, work_list), 3) 

        if out: 
            ref.save_image(resamp, out)
        
        return resamp


def chain(*args):
    """ 
    Concatenate a series of registrations.

    Args: 
        *args: Registration objects, given in the order that they need to be 
            applied (eg, for A -> B -> C, give them in that order and they 
            will be multiplied as C * B * A)

    Returns: 
        Registration object, with the first registrations' source 
        and the last's reference (if these are not None)
    """

    if (len(args) == 1) and (type(args) is Registration):
        chained = args
    else: 
        if not all([isinstance(r, Registration) for r in args ]):
            raise RuntimeError("Each item in sequence must be a" + 
                " Registration.")
        chained = args[1] * args[0]
        for r in args[2:]:
            chained = r * chained 

    return chained 


def _affine_transform(matrix, points): 
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
    min_max = (ref.min(), ref.max())
    array[array < min_max[0]] = min_max[0]
    array[array > min_max[1]] = min_max[1]
    return array 


def _application_worker(data, ref2src_world, src_spc, 
    ref_spc, order, cores, **kwargs):

    if len(data.shape) != 4 and len(data.shape) != 3: 
        raise RuntimeError("Can only handle 3D/4D data")

    # Affine transformation requires mapping from reference voxels
    # to source voxels (the inverse of how transforms are given)
    ref2src_vox = (src_spc.world2vox @ ref2src_world @ ref_spc.vox2world)
    ijk = ref_spc.ijk_grid('ij').reshape(-1,3).T
    ijk = _affine_transform(ref2src_vox, ijk)
    worker = functools.partial(map_coordinates, order=order, 
        coordinates=ijk, output=data.dtype, **kwargs)

    if len(data.shape) == 4: 
        data = np.moveaxis(data, 3, 0)
    else: 
        data = data.reshape(1, *data.shape)

    if cores == 1:  
        resamp = [ worker(d) for d in data ] 
    else: 
        with mp.Pool(cores) as p: 
            resamp = p.map(worker, data)

    resamp = np.stack([r.reshape(ref_spc.size) for r in resamp], axis=3)
    return _clip_array(np.squeeze(resamp), data) 

