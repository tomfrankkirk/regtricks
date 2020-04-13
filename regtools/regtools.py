import os.path as op 
import glob 
import multiprocessing as mp
import functools 
import copy 
import os 
from textwrap import dedent

import nibabel 
import numpy as np 
from scipy.interpolate import interpn
from scipy.ndimage.interpolation import map_coordinates
from toblerone import utils 
import h5py as h5

from .image_space import ImageSpace
from . import x5_interface as x5 


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

        elif convention.lower() == "world":
            self.__src2ref_world = src2ref 

        else: 
            raise RuntimeError("Unrecognised convention")


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


    def save_txt(self, fname, src, ref, convention):
        if convention.lower() == "fsl":
            np.savetxt(path, self.to_fsl(src, ref))
        else: 
            np.savetxt(path, self.src2ref_world)


    def save(self, path):
        x5.save_manager(self, path)


    def apply_to_grid(self, src, out=None, dtype=None):
        """
        Apply registration to the voxel grid of an image, retaining original
        voxel data (no resampling). This is equivalent to shifting the image
        within world space but not altering the contents of the image itself.

        Args: 
            src: either a nibabel Image object or path string to one
            out: (optional) where to save output 
            dtype: (optional) datatype (default same as input)
        
        Returns: 
            nibabel Nifti object 
        """

        if isinstance(src, str):
            src = nibabel.load(src)
        if type(src) is not nibabel.Nifti1Image:
            raise RuntimeError("src must be a nibabel Nifti or path to image")

        src_spc = ImageSpace(src)
        new_spc = src_spc.transform(self.src2ref_world)
        nii = new_spc.make_nifti(src.dataobj)

        if out: 
            nibabel.save(nii, out)
        
        return nii 


    def apply_to_image(self, src, ref, out=None, dtype=None, cores=1, **kwargs):
        """
        Applies registration transform to image data and inserts the output 
        into a new reference voxel grid. Uses scipy.ndimage.interpolation.
        map_coordinates, see that documentation for **kwargs. 


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
        elif type(src) is not nibabel.Nifti1Image:
            raise RuntimeError("src must be a nibabel Nifti or path to image")
        src_spc = ImageSpace(src)

        if not isinstance(ref, ImageSpace):
            try: 
                ref = ImageSpace(ref)
            except: 
                raise RuntimeError("ref must be a nibabel Nifti, ImageSpace, or path")

            img = src.get_fdata().astype(src.get_data_dtype())
            if not dtype: 
                dtype = src.get_data_dtype()
            resamp = _application_worker(img, self.ref2src_world, src_spc, ref, 
                cores, **kwargs)

            if out: 
                ref.save_image(resamp, out)

            return ref.make_nifti(resamp)


    # Allow overriding of the form (other @ self) - as the other will not 
    # know how to interpret this object, __rmatmul__ will instead be called
    # on self. The actual multiplication is handled by matmul, below.
    def __rmatmul__(self, other):
        if type(other) is np.ndarray: 
            oth = Registration(other)
        # elif type(other) is list: 
        #     if not all([ type(x) is np.ndarray for x in other ]):
        #         raise RuntimeError("Incompatible list elements for multiplication")
        #     oth = MotionCorrection(other)
        else: 
            assert isinstance(other, (Registration, MotionCorrection))
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

        if type(other) is np.ndarray:
            other = Registration(other)

        if ((type(self) is MotionCorrection) 
            and (type(other) is MotionCorrection)):

            if not len(self) == len(other):
                raise RuntimeError("MotionCorrections must be of equal length")
            
            world_mats = [ m1 @ m2 for m1,m2 in 
                zip(self.src2ref_world, other.src2ref_world) ]
            ret = MotionCorrection(world_mats, other.src_spc, self.ref_spc, 
                convention="world")

        elif ((type(self) is MotionCorrection) 
            or (type(other) is MotionCorrection)):
            
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

        elif ((type(self) is Registration) 
            and (type(other) is Registration)): 

            overall_world = self.src2ref_world @ other.src2ref_world
            ret = Registration(overall_world, other.src_spc, self.ref_spc, 
                convention="world")

        else: 
            raise RuntimeError("Unsupported argument type")

        return ret 


    @classmethod
    def identity(cls, src=None, ref=None):
        return Registration(np.eye(4), src, ref, convention="world")


    @classmethod
    def eye(cls, src=None, ref=None):
        return Registration.identity(src, ref)


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


    def __len__(self):
        return len(self.transforms)


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


    # Multiplication is handled by the Registration class 
    __matmul__ = Registration.__matmul__
    __rmatmul__ = Registration.__rmatmul__

    
    def save_txt(outdir, src, ref, convention="world"):
        os.makedirs(outdir, exist_ok=True)
        for idx, r in enumerate(self.transforms):
            p = op.join(outdir, "MAT_{:04d}.txt".format(idx))
            r.save_txt(p, src, ref, convention)


    @property 
    def transforms(self):
        return self._MotionCorrection__transforms


    @property
    def src_spc(self): 
        return self.transforms[0].src_spc


    @property
    def ref_spc(self):
        return self.transforms[0].ref_spc

    
    @property 
    def src2ref_world(self):
        """List of src to ref transformation matrices"""
        return [ t.src2ref_world for t in self.transforms ]


    @property
    def ref2src_world(self):
        """List of ref to src transformation matrices"""
        return [ t.ref2src_world for t in self.transforms ]
    

    def inverse(self): 
        mats = self.ref2src_world 
        return MotionCorrection(mats, self.ref_spc, self.src_spc, 'world')


    def apply_to(self, src, ref, out=None, dtype=None, 
        cores=mp.cpu_count(), **kwargs):
        """
        Apply motion correction to timeseries data. See scipy.nidimage.
        interpolate.map_coordinates for **kwargs (including order of 
        spline interpolation). Note that this function performs resampling 
        
        Args: 
            src: str or nibabel Nifti, timeseries to correct
            ref: str, nibabel Nifti, or ImageSpace, reference space in 
                which to place ouput 
            out: (optional) path to save output at 
            dtype: (optional) output datatype (default same as input)
            cores: (optional) number of cores to use (default max)
            **kwargs: any accepted by scipy map_coordinates (inc. order of
                spline interpolation, 1-5)
        
        Returns: 
            nibabel Nifti object 
        """ 
        
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
        work_list = zip(img, self.ref2src_world)
        if cores == 1:
            resamp = np.stack([ worker(*fm) for fm in work_list ], 3)
        else: 
            with mp.Pool(cores) as p: 
                resamp =  np.stack(p.starmap(worker, work_list), 3) 

        if out: 
            ref.save_image(resamp, out)
        
        return ref.make_nifti(resamp)


def load(path):
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
            raise RuntimeError("Each item in sequence must be a" + 
                " Registration or MotionCorrection.")
        chained = args[1] @ args[0]
        for r in args[2:]:
            chained = r @ chained 

    return chained 


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


def _application_worker(data, ref2src_world, src_spc, ref_spc, cores, **kwargs):
    """
    Worker function for Registration and MotionCorrection apply_to_image()

    Args: 
        data: np.array of data (3D or 4D)
        ref2src_world: transformation between reference space and source, 
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

    # Affine transformation requires mapping from reference voxels
    # to source voxels (the inverse of how transforms are given)
    ref2src_vox = (src_spc.world2vox @ ref2src_world @ ref_spc.vox2world)
    ijk = ref_spc.ijk_grid('ij').reshape(-1,3).T
    ijk = _affine_transform(ref2src_vox, ijk)
    worker = functools.partial(map_coordinates, coordinates=ijk, 
        output=data.dtype, **kwargs)

    # Move the 4th dimension to the front, so that we can iterate over each 
    # volume of the timeseries. If 3D data, pad out the array with a
    # singleton dimension at the front to get the same effect 
    if len(data.shape) == 4: 
        data = np.moveaxis(data, 3, 0)
    else: 
        data = data.reshape(1, *data.shape)

    if cores == 1:  
        resamp = [ worker(d) for d in data ] 
    else: 
        with mp.Pool(cores) as p: 
            resamp = p.map(worker, data)

    # Undo the changes we made to the data dimensions 
    if len(data.shape) == 4: 
        data = np.moveaxis(data, 0, 3)
    else: 
        data = np.squeeze(data)

    # Stack all the individual volumes back up in time dimension 
    # Clip the array to the original min/max values 
    resamp = np.stack([r.reshape(ref_spc.size) for r in resamp], axis=3)
    return _clip_array(np.squeeze(resamp), data) 

