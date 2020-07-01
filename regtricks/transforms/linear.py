import os.path as op 
from textwrap import dedent
import glob 
import os 

import nibabel
from nibabel import Nifti1Image, MGHImage
import numpy as np 
from fsl.data.image import Image as FSLImage

from regtricks.image_space import ImageSpace
from regtricks import x5_interface as x5 
from regtricks import application_helpers as apply
from regtricks import multiplication as multiply
from regtricks.transforms.transform import Transform

class Registration(Transform):
    """
    Affine (4x4) transformation between two images.

    Args: 
        src2ref (str/np.ndarray): path to text-like file to load or np.ndarray
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
        """
        Load an affine (4x4) transformation between two images directly from 
        FLIRT's -omat output. 

        Args: 
            src2ref (str/np.ndarray): path to text-like file to load or np.ndarray
            src: the source of the transform 
            ref: the reference (or target) of the transform 

        Returns: 
            Registration object
        """

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

    def save_fsl(self, path, src, ref):
        """Save in FSL convention as textfile at path"""
        m = self.to_fsl(src, ref)
        np.savetxt(path, m)

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
    
    Args: 
        mats: a path to a directory containing transformation matrices, in
            name order (all files will be loaded), or a list of individual
            filenames, or a list of np.arrays 
    """

    def __init__(self, mats):
        Transform.__init__(self)

        if isinstance(mats, str):
            if op.isdir(mats): 
                mats = sorted(glob.glob(op.join(mats, '*')))
                if not mats: 
                    raise RuntimeError("Did not find any matrices in %s" % mats)
            else: 
                mat = np.loadtxt(mats)
                if mat.shape[0] % 4: 
                    raise ValueError("Matrix loaded from %s " % mats, 
                                     "should be sized (4xN) x 4.")
                mats = [ mat[i*4:(i+1)*4,:] for i in range(mat.shape[0] // 4) ]

            
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
        """
        Load a MotionCorrection object directly from MCFLIRT's -omat directory. 
        Note that for within-timeseries registration, the src and ref arguments
        should take the same value. 

        Args: 
            mats: a path to a directory containing transformation matrices, in
                name order (all files will be loaded), or a list of individual
                filenames, or a list of np.arrays 
            src: source of the transforms (ie, the image being corrected)
            ref: the target of the transforms (normally same as src)

        Returns: 
            MotionCorrection 
        """

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

    def save_txt(self, outdir, prefix="MAT_"):
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
            p = op.join(outdir, "{}{:04d}.txt".format(prefix, idx))
            r.save_txt(p)

    def save_fsl(self, outdir, src, ref, prefix="MAT_"):
        """Save in FSL convention as textfiles at path"""

        os.makedirs(outdir, exist_ok=True)
        for idx, r in enumerate(self.transforms):
            p = op.join(outdir, "{}{:04d}.txt".format(prefix, idx))
            m = r.to_fsl(src, ref)
            np.savetxt(p, m)

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
