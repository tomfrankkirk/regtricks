import regtricks as rt 
import nibabel 

if __name__ == "__main__":

    asl_path = 'asl.nii.gz'
    brain_path = 'brain.nii.gz'
    MNI_path = 'MNI152_T1_2mm.nii.gz'
    asl2brain = 'asl2brain.mat'
    brain2MNI = 'brain2MNI_coeffs.nii.gz'

    # Make the T1-ASL grid space 
    asl_spc = rt.ImageSpace(asl_path)
    t1_spc = rt.ImageSpace(brain_path)
    t1_asl_grid = t1_spc.resize_voxels(asl_spc.vox_size / t1_spc.vox_size)

    # Load the asl to T1 registration, and apply it to the ASL and leave 
    # the result in T1-ASL grid 
    asl2brain = rt.Registration.from_flirt('asl2brain.mat', asl_path, brain_path)
    asl_t1_t1aslgrid = asl2brain.apply_to_image(asl_path, t1_asl_grid)
    nibabel.save(asl_t1_t1aslgrid, 'asl2t1_t1aslgrid.nii.gz')

    # Load the FNIRT transform for T1 to MNI. You need to pass the original
    # images that the transform maps FROM and TO (in that order), so 
    # the t1 and the MNI 
    brain2MNI = rt.NonLinearRegistration.from_fnirt('brain2MNI_coeffs.nii.gz', 
                                                    brain_path, MNI_path)

    # Apply the warp to the ASL in T1-ASL grid space, to align it with MNI, 
    # and save the result in the same grid, not the MNI!
    asl2mni_t1aslgrid = brain2MNI.apply_to_image(asl_t1_t1aslgrid, t1_asl_grid)
    nibabel.save(asl2mni_t1aslgrid, 'asl2mni_t1aslgrid.nii.gz')

    # The same thing, but in one step using chain() 
    asl2MNI = rt.chain(asl2brain, brain2MNI)
    asl2mni_t1aslgrid = asl2MNI.apply_to_image(asl_path, t1_asl_grid)
    nibabel.save(asl2mni_t1aslgrid, 'asl2mni_t1aslgrid_chained.nii.gz')