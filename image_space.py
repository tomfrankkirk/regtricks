"""
ImageSpace: image matrix, inc dimensions, voxel size, vox2world matrix and
inverse, of an image. Used for resampling operations between different 
spaces and also for saving images into said space (eg, save PV estimates 
into the space of an image)
"""

import os.path as op 
import copy 
import warnings

import nibabel
import numpy as np 

class ImageSpace(object):
    """
    Voxel grid of an image, ignoring actual image data. 

    Args: 
        path: path to image file
    
    Attributes: 
        size: array of voxel counts in each dimension 
        vox_size: array of voxel size in each dimension 
        vox2world: 4x4 affine to transform voxel coords -> world
        world2vox: inverse of above 
        self.offset: private variable used for derived spaces 
    """

    def __init__(self, path):

        if not op.isfile(path):
            raise RuntimeError("Image %s does not exist" % path)

        img = nibabel.load(path)
        self.size = img.header['dim'][1:4]
        self.vox_size = img.header['pixdim'][1:4]
        self.vox2world = img.affine
        self._offset = None
        self.header = img.header 


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
    def FoV_size(self):
        """FoV associated with image, in mm"""

        return self.size * self.vox_size


    @property
    def bbox_origin(self): 
        """
        Origin of the image's bounding box, referenced to first voxel's 
        corner, not center
        """

        orig = np.array((3 * [-0.5]) + [1])
        return self.vox2world @ orig


    @property
    def world2vox(self):
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

        det = np.linalg.det(self.vox2world[0:4, 0:4])
        vox2FSL = np.zeros((4,4))
        vox2FSL[range(3), range(3)] = self.vox_size

        # Check the xyzt field to find the spatial units. 
        xyzt = str(self.header['xyzt_units'])
        if xyzt == '01': 
            multi = 1000
        elif xyzt == '10':
            multi = 1 
        elif xyzt =='11':
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
        return np.linalg.inv(self.vox2FSL)


    @property
    def world2FSL(self):
        return self.vox2FSL @ self.world2vox


    @property
    def FSL2world(self):
        return self.vox2world @ self.FSL2vox


    def supersample(self, factor):
        """
        Produce a new image space which is a copy of the current space, 
        upsampled by a factor of (a,b,c) in each dimension 

        Args:
            factor: int, or sequence of 3 ints, multiplier in each dimension 
        
        Returns: 
            ImageSpace object
        """

        if isinstance(factor, int):
            factor = 3 * [factor]
        else: 
            try: 
                assert len(factor) == 3
            except Exception as e: 
                raise RuntimeError("Factor must be an int or sequence of 3 ints")

        factor = np.array(factor)

        factor = np.array(factor)
        newSpace = copy.deepcopy(self)

        newSpace.size = (self.size * factor).astype(np.int16)
        newSpace.vox_size /= factor
        newSpace.vox2world[0:3,0:3] /= factor[None,:]

        # Get the corner of the reference space bounding box 
        # Then move in by 0.5 voxels of the new voxel grid to the find the point 
        # corresponding to voxel (0,0,0) on the new grid 
        new_orig = self.bbox_origin + (newSpace.vox2world[0:3,0:3] @ np.array((3*[0.5])))
        newSpace.vox2world[0:3,3] = new_orig

        # Check the bounds of the new voxel grid we have created
        np.testing.assert_array_almost_equal(self.bbox_origin, newSpace.bbox_origin, 6)
        box_diag_ref = self.vox2world[0:3,0:3] @ self.size 
        box_diag_new = newSpace.vox2world[0:3,0:3] @ newSpace.size 
        np.testing.assert_array_almost_equal(box_diag_ref, box_diag_new, 6)

        return newSpace


    def subsample(self, factor, mode='floor'):
        """
        Subsample an ImageSpace and return new object. 
        
        Args: 
            factor: int, or sequence of 3 ints, by which each dimension's size
                will be divided - eg, 2 corresponds to 0.5 original size 
            mode: 'floor' [defualt] or 'ceil'. if the FoV of the new grid 
                does not match that of the original, round down/up respectively
                to get the new extents 

        Returns:
            ImageSpace object
        """
        
        if (mode != 'floor') and (mode != 'ceil'):
            raise RuntimeError("Mode must be floor or ceil")

        if isinstance(factor, int):
            factor = 3 * [factor]
        else: 
            try: 
                assert len(factor) == 3
            except Exception as e: 
                raise RuntimeError("Factor must be an int or sequence of 3 ints")

        factor = np.array(factor)
        if not np.all(factor >= 1):
            raise RuntimeError("This tool can only be used for down-sampling")

        new = copy.deepcopy(self)
        if mode == 'floor':
            func = np.floor 
        else:
            func = np.ceil 
        
        new.size = func(self.size / factor).astype(np.int16)
        new.vox2world[0:3,0:3] *= factor[None,:]
        new.vox_size *= factor 
        new_orig = self.bbox_origin + (new.vox2world[0:3,0:3] @ np.array([0.5, 0.5, 0.5]))
        new.vox2world[0:3,3] = new_orig

        return new 


    def crop(self, start_extents, end_extents):
        """
        Crop space to lie between start and end points, return new ImageSpace.
        0-indexing is used; for example the extents [0, 4] maps to [0,1,2,3]

        Args: 
            start_extents: sequence of 3 ints, indices from which subspace
                should start 
            end_extents: sequence of 3 ints, indices at which the subspace
                should end 

        Returns:
            new ImageSpace object 
        """

        start_extents = np.array(start_extents)
        end_extents = np.array(end_extents)
        if (start_extents.size != 3) and (end_extents.size != 3):
            raise RuntimeError("Extents must be 3 elements each")

        if not np.all(end_extents > start_extents):
            raise RuntimeError("End extents must be smaller than start_extents")

        if np.any(start_extents < 0):
            raise RuntimeError("Start extents must be positive")

        if np.any(end_extents > self.size):
            raise RuntimeError("End extents exceed image size")

        new_size = end_extents - start_extents
        if not np.any(new_size < self.size):
            print("Warning: this combination of start/end does not crop in any dimension")

        new = copy.deepcopy(self)
        new_orig = self.vox2world[0:3,3] + (self.vox2world[0:3,0:3] @ start_extents) 
        new.vox2world[0:3,3] = new_orig
        new.size = new_size 

        return new 


    @classmethod 
    def create(cls, bbox_corner, size, vox_size):
        """
        Create an ImageSpace from bounding box location and voxel size. 
        Note that the voxels will be axis-aligned (no rotation). 

        Args: 
            bbox_corner: 3-vector, location of the minimum corner of the
                bounding box, at which the corner of voxel 0 0 0 will lie. 
            size: 3-vector, number of voxels in each spatial dimension 
            vox_size: 3-vector, size of voxel in each dimension 

        Returns
            ImageSpace object 
        """

        vox_size = np.array(vox_size)
        size = np.array(size)
        bbox_corner = np.array(bbox_corner)

        spc = cls.__new__(cls)
        spc.vox2world = np.identity(4)
        spc.vox2world[(0,1,2),(0,1,2)] = vox_size
        orig = bbox_corner + (np.array((3*[0.5])) @ spc.vox2world[0:3,0:3])
        spc.vox2world[0:3,3] = orig 
        spc.size = size 
        spc.vox_size = vox_size
    
        return spc 


    def save_image(self, data, path):
        """Save 3D or 4D data array at path using this image's voxel grid"""

        if not np.all(data.shape[0:3] == self.size):
            if data.size == np.prod(self.size):
                warnings.warn("Reshaping data to 3D volume")
                data = data.reshape(self.size)
            elif not(data.size % np.prod(self.size)):
                warnings.warn("Saving as 4D volume")
                data = data.reshape((*self.size, -1))
            else:
                raise RuntimeError("Data size does not match image size")

        if data.dtype is np.dtype('bool'):
            data = data.astype(np.int8)

        if not (path.endswith('.nii') or path.endswith('.nii.gz')):
            path += '.nii.gz'

        nii = nibabel.nifti2.Nifti2Image(data, self.vox2world)
        nii.header.set_xyzt_units(2,None)
        nibabel.save(nii, path)


