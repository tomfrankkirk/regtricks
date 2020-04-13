# Regtools
Tools for manipulating, combining and applying image transformations.  

```python
# Example: motion correct and register a timeseries to a structural image
# Load the registration matrices (eg produced by FLIRT/MCFLIRT)
struct2asl = Registration('struct2asl.mat', 't1.nii.gz', 'asl.nii.gz')
asl_moco = MotionCorrection('mcflirt_dir', 'asl.nii.gz')

# Combine them into a single operation and apply to the timeseries
# The result will be saved as 'asl_t1_mc.nii.gz'
asl2struct_moco = chain(struct2asl.inverse(), asl_moco)
asl2struct_moco.apply('asl.nii.gz', 't1.nii.gz', 'asl_t1_mc.nii.gz')
```

## Contents
- [Overview](#overview)
- [Loading, converting and saving <a name="loading"></a>](#loading-converting-and-saving)
- [Chaining transformations](#chaining-transformations)
- [Applying transformations](#applying-transformations)
- [More examples](#more-examples)
- [Further reading](#further-reading)

## Overview
The following three classes are provided: 

`Registration`: a 4x4 affine transformation, that optionally can be associated with a specific source and reference image. Internally, all registrations are stored in world-world terms, and all interactions between registrations are also in world-world terms. 

`MotionCorrection`: a sequence of `Registration` objects, one for each volume of a timeseries. 

`ImageSpace`: the voxel grid of an image, including the dimensions, voxel size and orientation (almost everything except the image itself). This class also allows easy manipulation of the grid (shifting, cropping, resizing voxels, etc)

## Loading, converting and saving <a name="loading"></a>
`Registration` objects can be initialised from a text file or `np.array`. If the registration was produced by FLIRT, paths to the source and reference images are required to convert the transformation into world-world terms. 

```python  
src = 'source_image.nii.gz'
ref = 'reference_image.nii.gz'

# A simple array is assumed to be in world-world terms
r1 = Registration(an_array)
# Convert to FSL, returns a np.array
r1.to_fsl(src, ref) 
# Save as FSL
r1.save_txt('r1_fsl.txt', src, ref, 'fsl') 

# If the src and ref are provided, FSL/FLIRT convention is assumed
# The conversion to world-world terms is automatic 
r2 = Registration('a_matrix.txt', src, ref)
# Return inverse
r2.inverse() 
# Inverse transform in FSL terms, as a np.array
r2.inverse().to_fsl(ref, src) 
# Save as text file 
r2.save_txt('r2_world.txt')
# Save inverse in FSL terms 
r2.inverse().save_txt('r2_inv_fsl.txt', ref, src, 'fsl') 

# Imagine we want to apply the transformation represented by r2, 
# but keep the result within the original voxel grid. In FSL terms: 
r2.save_txt('r2_fsl_samespace.txt', src, src, 'fsl')
```

`MotionCorrection` objects can be initialised from a directory containing transformations, a list of text file paths, or a list of `np.array` objects. If the registration was produced by MCFLIRT, paths to the source and reference images are required to convert the transformations to world-world terms. 

```python
src = 'some_timeseries.nii.gz'

# load from MCFLIRT directory
m1 = MotionCorrection('mcflirt_directory', src) 
# save as text files
m1.save_txt('world_directory') 
# create from list of arrays
m2 = MotionCorrection(list_of_arrays) 
# save in fsl terms
m2.save_txt('mcflirt_out', src, ref, 'fsl')
```

`ImageSpace` objects can be initialised with a nibabel Nifti object or a path to a Nifti image. 
```python
# from a nibabel nifti 
src_spc = ImageSpace(src_nifti) 
# from a path 
ref_spc = ImageSpace('ref.nii.gz') 
# save some random data in this space 
ref_spc.save_image(np.random.rand(ref_spc.size), 'rand.nii.gz') 
```

## Chaining transformations
Transformations may be combined my matrix multiplication. `Registration`, `MotionCorrection` and `np.array` objects can all be combined in this manner. Note that the product of a `np.array` and `Registration` is a new `Registration`, and the product of anything with a `MotionCorrection` is a new `MotionCorrection`. The order of multiplication is important: to apply the transformation A then B, the matrix multiplication `B @ A` should be used. **The safest way of combining registrations is to use the `chain()` function - it works on any number of transformations and takes care of the order for you!**

```python
# Three images (A,B,C) and three transformations: A->B, motion correction for A, 
# and C->B. 
a2a_moco = MotionCorrection('a_mcflirt_directory', 'a.nii.gz')
a2b = Registration('a2b.txt')
c2b = Registration('c2b_flirt.mat', 'c.nii.gz', 'b.nii.gz')

# Get a single transformation for A->C, including motion correction 
# NB the result will be a MotionCorrection object 
a2c_moco = chain(a2a_moco, a2b, c2b.inverse())

# Alternatively, do the multiplication directly: 
a2c_moco_2 = (c2b.inverse() @ a2b @ a2a_moco)

# Save as world-world matrices, in FSL convention
a2c_moco.save_txt('a2c_moco_dir', 'a.nii.gz', 'c.nii.gz')
```

## Applying transformations
Both `Registration` and `MotionCorrection` objects may applied with the `apply_to()` method. This uses SciPy's `ndimage.interpolation.map_coordinates()` function, allowing spline interpolation of order 1 (trilinear) to 5 (quintic) with pre-filtering to reduce interpolation artefacts. All `**kwargs` accepted by `map_coordinates()` may be passed to `apply_to()`. 

```python
a_img_3D = 'some_volume.nii.gz'
b_img_4D = 'some_timeseries.nii.gz'
a2b = Regisration('a2b.txt')
b2b_moco = MotionCorrection('b_mcflirt_dir', b_img_4D)

# A single registration can be applied to both 3D and 4D data
a2b.apply_to(a_img_3D, b_img_4D) # map a onto b, return nibabel nifti
# transform b onto a, save the result as a nifti 
a2b.inverse().apply_to(b_img_4D, a_img3D, out='timeseries_on_a.nii.gz')

# Chain the motion correction and transform b2a together 
# MotionCorrections can only be applied to 4D data 
b2a_moco = chain(b2b_moco, a2b.inverse())
b2a_moco.apply_to(b_img_4D, a_img_3D, out='timeseries_on_a_mc.nii.gz')

# Apply the chained transformation, but without resampling the result onto
# the voxel grid of b_img_3D (keep it in the original space of the timeseries)
b2a_moco.apply_to(b_img_4D, b_img_4D, out='timeseries_on_a_in_b_mc.nii.gz')
```

## More examples

to come 

## Further reading
An explanation of the FSL coordinate system: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/FAQ