"""
ImageSpace: image matrix, inc dimensions, voxel size, vox2world matrix and
inverse, of an image. Used for resampling operations between different 
spaces and also for saving images into said space (eg, save PV estimates 
into the space of an image)
"""

import copy 
import textwrap

import nibabel
import numpy as np 
from nibabel import Nifti1Image, MGHImage
from fsl.data.image import Image as FSLImage


class ImageSpace(object):
    """
    Voxel grid of an image, ignoring actual image data. 

    Args: 
        img: path to image, nibabel Nifti/MGH or FSL Image object
    
    Attributes: 
        size: array of voxel counts in each dimension 
        vox_size: array of voxel size in each dimension 
        vox2world: 4x4 affine to transform voxel coords -> world
        world2vox: inverse of above 
        self.offset: private variable used for derived spaces 
    """

    def __init__(self, img):

        if isinstance(img, str):
            fname = img 
            img = nibabel.load(img)
        else: 
            assert isinstance(img, (Nifti1Image, MGHImage, FSLImage))
            if type(img) is FSLImage:
                img = img.nibImage
            fname = img.get_filename()

        self.file_name = fname     
        self.size = np.array(img.shape[:3], np.int16)
        self.vox2world = img.affine
        self.header = img.header

    
    @classmethod
    def manual(cls, vox2world, size):
        """Manual constructor"""

        spc = cls.__new__(cls)
        spc.vox2world = vox2world
        spc.size = np.array(size, np.int16)
        spc._offset = None 
        spc.file_name = None 
        spc.header = None 
        return spc 


    @classmethod 
    def create_axis_aligned(cls, bbox_corner, size, vox_size):
        """
        Create an ImageSpace from bounding box location and voxel size. 
        Note that the voxels will be axis-aligned (no rotation). 

        Args: 
            bbox_corner: 3-vector, location of the furthest corner of the
                bounding box, at which the corner of voxel 0 0 0 will lie. 
            size: 3-vector, number of voxels in each spatial dimension 
            vox_size: 3-vector, size of voxel in each dimension 

        Returns
            ImageSpace object 
        """

        bbox_corner = np.array(bbox_corner)
        vox2world = np.identity(4)
        vox2world[(0,1,2),(0,1,2)] = vox_size
        orig = bbox_corner + (np.array((3 * [0.5])) @ vox2world[0:3,0:3])
        vox2world[0:3,3] = orig 
        return cls.manual(vox2world, size)


    @classmethod
    def save_like(cls, ref, data, path): 
        """Save data into the space of an existing image

        Args: 
            ref: path to image defining space to use 
            data: ndarray (of appropriate dimensions)
            path: path to write to 
        """
        
        spc = ImageSpace(ref)
        spc.save_image(data, path)


    @property
    def vox_size(self):
        """Voxel size of image"""
        return np.linalg.norm(self.vox2world[:3,:3], ord=2, axis=0)


    @property
    def fov_size(self):
        """FoV associated with image, in mm"""

        return self.size * self.vox_size


    @property
    def bbox_origin(self): 
        """
        Origin of the image's bounding box, referenced to first voxel's 
        corner, not center (ie, -0.5, -0.5, -0.5)
        """

        orig = np.array((3 * [-0.5]) + [1])
        return (self.vox2world @ orig)[:3]


    @property
    def world2vox(self):
        """World coordinates to voxels"""
        return np.linalg.inv(self.vox2world)


    @property
    def vox2FSL(self):
        """
        Transformation between voxels and FSL coordinates (scaled mm). FLIRT
        matrices are given in (src FSL) -> (ref FSL) terms. 
        See: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/FAQ
        """

        if len(self.size) < 3:
            raise RuntimeError("Volume has less than 3 dimensions, "
                    "cannot resolve space")

        det = np.linalg.det(self.vox2world[0:3, 0:3])
        vox2FSL = np.zeros((4,4))
        vox2FSL[range(3), range(3)] = self.vox_size

        # Check the xyzt field to find the spatial units. 
        multi = 1 
        if (self.header is not None) and ('xyzt_units' in self.header):
            xyzt = str(self.header['xyzt_units'])
            if xyzt == '01': 
                multi = 1000
            elif xyzt == '10':
                multi = 1 
            elif xyzt == '11':
                multi = 1e-3

        else: 
            multi = 1 

        if det > 0:
            vox2FSL[0,0] = -self.vox_size[0]
            vox2FSL[0,3] = (self.size[0] - 1) * self.vox_size[0]

        vox2FSL *= multi
        vox2FSL[3,3] = 1
        return vox2FSL


    @property
    def FSL2vox(self):
        """Transformation from FSL scaled coordinates to voxels"""
        return np.linalg.inv(self.vox2FSL)


    @property
    def world2FSL(self):
        """Transformation from world coordinates to FSL scaled"""
        return self.vox2FSL @ self.world2vox


    @property
    def FSL2world(self):
        """Transformation from FSL scaled coordinates to world"""
        return self.vox2world @ self.FSL2vox


    def resize_voxels(self, factor, mode="floor"):
        """
        Resize voxels of this grid. 

        Args: 
            factor: either a single value, or 3 values in array-like form, 
                by which to multiply voxel size in each dimension 
            mode: "floor" or "ceil", whether to round the grid size up or down
                if factor does not divide perfectly into the current size 

        Returns: 
            new ImageSpace object 
        """
        if mode == "floor":
            rounder = np.floor 
        else: 
            rounder = np.ceil 
      
        if isinstance(factor, (int, float)):
            factor = factor * np.ones(3)

        new_size = rounder(self.size / factor).astype(np.int16)
        new_vox2world = copy.deepcopy(self.vox2world)
        new_vox2world[:3,:3] *= factor[None,:]
        bbox_shift = (new_vox2world[:3,:3] @ [0.5, 0.5, 0.5])
        new_vox2world[:3,3] = self.bbox_origin + bbox_shift
        return ImageSpace.manual(new_vox2world, new_size)


    def touch(self, path, dtype=np.float32): 
        """Save empty volume at path"""
        vol = np.zeros(self.size, dtype)
        self.save_image(vol, path)


    def resize(self, start, new_size):
        """
        Resize the FoV of this space, maintaining axis alignment and voxel
        size. Can be used to both crop and expand the grid. For example, 
        to expand the grid sized X,Y,Z by 10 voxels split equally both 
        before and after each dimension, use (-5,5,5) and (X+5, Y+5, Z+5)

        Args: 
            start: sequence of 3 ints, voxel indices by which to shift first
                voxel (0,0,0 is origin, negative values can be used to expand
                and positive values to crop)
            new_size: sequence of 3 ints, length in voxels for each dimension, 
                starting from the new origin 

        Returns:
            new ImageSpace object 
        """

        start = np.array(start)
        new_size = np.array(new_size)
        new_size[new_size == 0] = self.size[new_size == 0]
        if (start.size != 3) and (new_size.size != 3):
            raise RuntimeError("Extents must be 3 elements each")

        if np.any(new_size < 0):
            raise RuntimeError("new_size must be positive")

        new = copy.deepcopy(self)
        new_orig = self.vox2world[0:3,3] + (self.vox2world[0:3,0:3] @ start) 
        new.vox2world[0:3,3] = new_orig
        new.size = new_size 
        new.file_name = None 
        return new 


    def make_nifti(self, data):
        """Construct nibabel Nifti for this voxel grid with data"""

        if not np.all(data.shape[0:3] == self.size):
            if data.size == np.prod(self.size):
                print("Reshaping data to 3D volume")
                data = data.reshape(self.size)
            elif not(data.size % np.prod(self.size)):
                print("Reshaping data as 4D volume")
                data = data.reshape((*self.size, -1))
            else:
                raise RuntimeError("Data size does not match image size")

        if data.dtype is np.dtype(np.bool):
            data = data.astype(np.int8)

        nii = nibabel.nifti1.Nifti1Image(data, self.vox2world)
        return nii 


    def save_image(self, data, path):
        """Save 3D or 4D data array at path using this image's voxel grid"""

        if not (path.endswith('.nii') or path.endswith('.nii.gz')):
            path += '.nii.gz'
        nii = self.make_nifti(data)
        nibabel.save(nii, path)


    def ijk_grid(self, indexing='ij'):
        """
        Return a 4D matrix of voxel indices for this space. Default indexing
        is 'ij' (matrix convention), 'xy' can also be used - see np.meshgrid
        for more info. 

        Returns: 
            4D array, size of this space in the first three dimensions, and 
                stacked I,J,K in the fourth dimension 
        """

        ijk = np.meshgrid(*[ np.arange(d) for d in self.size ], indexing=indexing)
        return np.stack(ijk, axis=-1)

    def voxel_centres(self, indexing='ij'):
        """
        Return a 4D matrix of voxel centre coordinates for this space. Default
        indexing is as for ImageSpace.ijk_grid(), which is 'ij' matrix convention.
        See np.meshgrid for more info. 

        Returns: 
            4D array, size of this space in the first three dimensions, and 
                stacked I,J,K in the fourth dimension.
        """

        from regtricks.application_helpers import aff_trans

        ijk = self.ijk_grid(indexing).reshape(-1,3)
        cents = aff_trans(self.vox2world, ijk)
        return cents.reshape(*self.size, 3)


    def transform(self, reg):
        """
        Apply affine transformation to voxel grid of this space. 
        If the reg is a np.array, it must be in world-world terms, and 
        if it is a Registration object, the world-world transform will
        be used automatically. 

        Args: 
            reg: either a 4x4 np.array (in world-world terms) or Registration
        
        Returns: 
            a transformed copy of this image space 
        """

        from regtricks import Registration

        if isinstance(reg, Registration):
            reg = reg.src2ref
        if not isinstance(reg, np.ndarray):
            raise RuntimeError("argument must be a np.array or Registration")

        new_spc = copy.deepcopy(self)
        new_spc.vox2world = reg @ new_spc.vox2world
        new_spc.file_name = None 
        return new_spc


    def __repr__(self):
        formatter = "{:8.3f}".format 
        with np.printoptions(precision=3, formatter={'all': formatter}):
            text = (f"""\
                ImageSpace with properties:
                size:          {self.size}, 
                voxel size:    {self.vox_size}, 
                field of view: {self.fov_size},
                vox2world:     {self.vox2world[0,:]}
                               {self.vox2world[1,:]}
                               {self.vox2world[2,:]}
                               {self.vox2world[3,:]}""")

        if self.file_name: 
            text += f"""
                loaded from: {self.file_name}"""
        else: 
            text += f"""
                loaded from: (no direct file counterpart)"""
        return textwrap.dedent(text)

    
    def __eq__(self, other):

        f1 = np.allclose(self.vox2world, other.vox2world)
        f2 = np.allclose(self.size, other.size)
        return all([f1, f2])
