Quickstart 
==============

Loading or creating transformations
-----------------------------------------

Linear or affine registrations (eg FSL FLIRT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

API link: :class:`regtricks.transforms.linear.Registration`

Registrations can be created from a `np.array`, a path to a text file that numpy can read, or by calling a wrapper for FLIRT. In all cases, the matrix should be 4x4 and the last row should be 0,0,0,1. 

.. code-block:: python 

   import regtricks as rt 

   # From an array
   m = np.eye(4)
   r = rt.Registration(m)

   # From a file that numpy can read (NB if using a FLIRT matrix see below example)
   p = '/a/path/to/file.txt'
   r = rt.Registration(r)

   # From a FLIRT matrix: provide the original source and reference images 
   src = 'the_source.nii.gz'
   ref = 'the_reference.nii.gz'
   p = 'the_flirt_matrix.mat'
   r = rt.Registration.from_flirt(p, src=src, ref=ref)

   # Alternatively, you can run FLIRT directly and return a Reigstration object
   src = 'the_source.nii.gz'
   ref = 'the_reference.nii.gz'
   r = rt.flirt(src, ref, **kwargs)


Motion corrections (eg FSL MCFLIRT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

API link: :class:`regtricks.transforms.linear.MotionCorrection`

Motion corrections are stored as a sequence of Registrations (eg, for a timeseries of 50 volumes, there will be 50 registrations). They can be created from a list of `np.array`, a path to a text file that shaped (4xN) x 4, a path to a folder containing *only* files for the individual arrays, or by calling a wrapper for MCFLIRT. 

.. code-block:: python 

   # From a list of arrays
   m = [ np.eye(4) for _ in range(10) ] 
   mc = rt.MotionCorrection(m)

   # From a file that numpy can read, shaped (4xN) x 4
   p = '/a/path/to/file.txt'
   mc = rt.Registration(p)

   # From a directory containing individual files, named in order
   p = 'a/path/to/dir'
   mc = rt.MotionCorrection(p)

   # From a MCFLIRT -mats directory: provide the original src and ref images
   # Unless using MCFLIRT's -reffile option, the src and the ref are the same!
   src = 'the_source.nii.gz'
   p = '/path/to/mcflirt.mat'
   mc = rt.Registration.from_flirt(p, src=src, ref=src)

   # Run MCFLIRT directly and return a MotionCorrection object
   src = 'the_source.nii.gz'
   mc = rt.mcflirt(src, **kwargs)


Non-linear registrations (ie FSL FNIRT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

API link: :class:`regtricks.transforms.nonlinear.NonLinearRegistration`

For the moment, the only way of loading in `NonLinearRegistrations` is via `FNIRT` output (or `epi_reg`, `topup`). 

.. code-block:: python 

   # From a FNIRT coefficients file, or displacement fields
   p = '/a/path/to/fnirt.nii.gz'
   src = 'src_image.nii.gz'
   ref = 'ref_image.nii.gz'

   # use intensity_correct = True if you want to use the Jacobian
   nl = rt.NonLinearRegistration.from_fnirt(p, src, ref)


Combining and applying transformations 
-----------------------------------------

Transformations, of any type and in any number, can be combined into a single transformation using `rt.chain`. The order of application will be the order the transformations are given. For example, `rt.chain(A, B, C)` will apply A, then B, then C. 

.. code-block:: python 

   # Prepare some transformations 
   A = rt.Registration(some_matrix)
   B = rt.MotionCorrection([some_matrices])
   C = rt.NonLinearRegistration.from_fnirt(some_fnirt_file, src, ref)

   # Register, motion correct and warp, in that order
   combined = rt.chain(A, B, C)

   # Now apply to images 
   transformed = combined.apply_to_image(some_nifti)


Working with ImageSpaces (voxel grids)
--------------------------------------------

API link: :class:`regtricks.image_space.ImageSpace`

Many operations can be achieved by directly manipulating the voxel grid of an image. For example, cropping, extending, reorienting, or changing the voxel size can be achieved using methods on the `ImageSpace` object. 

.. code-block:: python 

   spc = rt.ImageSpace(some_nifti)

   spc.resize # change dimensions of voxel grid 
   spc.create_axis_aligned # create a voxel grid 
   spc.resize_voxels # resize voxels of a grid 
   spc.make_nifti # make a NIFTI object from ImageSpace
   spc.bbox_origin # corner of grid's bounding box 
   spc.touch # write empty NIFTI for ImageSpace at path 
   spc.voxel_centres # array of all voxel centre coordinates 
   spc.world2FSL # transformation from world to FSL coords 
   spc.world2vox # transformation from world to voxel coords 
   spc.FSL2world # transformation from FSL to world coords 
   spc.vox2world # transformation from voxel to world coords 
   spc.transform # transform NIFTI sform header directly
