import os.path as op 
import tempfile 

import numpy as np 
import nibabel

import regtools
from image_space import ImageSpace


def main(asl, motion_correct=True, ext_ref=None):

    asl_img = nibabel.load(asl).get_fdata()
    asl_space = ImageSpace(asl)

    if motion_correct: 
        with tempfile.TemporaryDirectory() as d: 
            target_idx = (asl_img.shape[3] - 1) // 2
            target_fname = op.join(d, "asl_target.nii.gz")
            asl_space.save_image(asl_img[...,target_idx], target_fname)
            cmd = ("mcflirt in %s -refvol %d -mats -out asl_mcf" % 
                (asl, target_fname))


    if ext_ref is None:
        pass 


if __name__ == "__main__":
    
    asl = 'asl.nii.gz'
    brain = 'brain.nii.gz'
