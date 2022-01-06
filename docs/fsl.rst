FSL integration 
================

Regtricks can handle transformations expressed in world (aka world-mm) or FSL coordinate systems. 

The FSL coordinate system 
----------------------------

FSL uses a scaled-mm coordinate system. The origin of the system is always in a corner of the voxel grid and increments along each axis by the voxel dimensions. For example, if the voxel size is (1,2,3)mm, then voxel (2,2,2) will map to position (2,4,6) in FSL coordinates. For a more detailed overview of the system, see this `link <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/FAQ#What_is_the_format_of_the_matrix_used_by_FLIRT.2C_and_how_does_it_relate_to_the_transformation_parameters.3F>`_. 

Internally, regtricks converts all FSL transformations (both linear and non-linear) into world-world convention for consistency. Regtricks will perform this conversion automatically on FSL transforms *if the appropriate functions are used*: 

   - ``Registration.from_flirt()`` for linear transforms (FSL FLIRT)
   - ``MotionCorrection.from_mcflirt()`` for motion corrections (FSL MCFLIRT)
   - ``NonLinearRegistration.from_fnirt()`` for non-linear transforms (FSL FNIRT, topup, epi_reg)

.. warning:: 
   It is impossible to work out what convention a transformation is using just by inspecting it. For example, a 4x4 linear transformation matrix does not convey any information about world-world or FSL coordinate systems. The only solution is to know in advance how the transformation was generated (eg, via FSL FLIRT). 

