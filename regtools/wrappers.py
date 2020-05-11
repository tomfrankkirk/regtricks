import tempfile
import os.path as op 

import numpy as np
import nibabel
from fsl.wrappers.flirt import flirt as flirt_cmd
from fsl.wrappers.flirt import mcflirt as mcflirt_cmd
from fsl.wrappers.fnirt import fnirt as fnirt_cmd 
from fsl.data.image import Image as FSLImage

from . import Registration, MotionCorrection

# TODO: incorporate FNIRT, remove the output options 
# just check for omat argument, if not there then pipe it into
# temp directory and read it out. let normal args pass through 

def flirt(src, ref, **kwargs):
    """
    FLIRT registration wrapper. If any of the output arguments are given
    (out/o and omat), FLIRT will run in command line mode and save the 
    outputs at those paths, and nothing will be returned. If none of the 
    outputs are given, no outputs will be saved and a Registration will 
    be returned. See fsl.wrappers.flirt.flirt for full docs. 

    Args: 
        src: image to register
        ref: target to register on to 
        out/o: where to save output
        omat: save registration matrix 
        **kwargs: as for FLIRT 
    
    Returns: 
        Registration object, if no output options specified. 
    """

    # If out or omat are given, run as usual
    if any(['out' in kwargs, 'o' in kwargs, 'omat' in kwargs]):
        flirt_cmd(src, ref, **kwargs)
        return 

    # If out or omat not given, run within tempdir so we can return result
    else: 
        with tempfile.TemporaryDirectory() as d: 
            mat = op.join(d, 'omat.mat')
            flirt_cmd(src, ref, omat=mat, **kwargs)
            mat = np.loadtxt(mat)

        return Registration(mat, src, ref, "fsl")


def mcflirt(src, refvol=-1, **kwargs):
    """
    MCFLIRT motion correction wrapper. If an output path is given, MCFLIRT
    will run in command line mode, save the output and return None. If no
    output path is given, a MotionCorrection and Nibabel image for the frame
    used as the reference will be returned. See fsl.wrappers.flirt.mcflirt
    for full docs. 

    Args: 
        src: image to register
        refvol: target frame to register on to, default is N/2
        out: where to save output
        **kwargs: as for MCFLIRT
    
    Returns: 
        tuple of MotionCorrection object, and a Nibabel image of the frame
            used as reference, if no output path given 
    """

    if isinstance(src, str):
        src = nibabel.load(src)


    if (refvol == -1) and (refvol not in kwargs):
        refvol = src.shape[-1] // 2
        kwargs['refvol'] = refvol
    else: 
        kwargs['refvol'] = refvol

    # If out or omat are given, run as usual
    if any(['out' in kwargs, 'o' in kwargs, 'omat' in kwargs]):        
        mcflirt_cmd(src, **kwargs)
        return 

    else: 
        if isinstance(src, FSLImage):
            data = src.data
            v2w = src.voxToWorldMat
        else: 
            data = src.get_fdata()
            v2w = src.affine

        refimg = nibabel.Nifti1Image(data[..., kwargs['refvol']], 
                                     v2w, src.header)

        with tempfile.TemporaryDirectory() as d: 
            img = op.join(d, 'img.nii.gz')
            matsdir = op.join(d, 'img.nii.gz.mat')
            mcflirt_cmd(src, out=img, mats=True, **kwargs)
            mc = MotionCorrection(matsdir, src, src, "fsl")

        return (mc, refimg)

def fnirt():
    pass