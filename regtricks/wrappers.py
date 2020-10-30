import tempfile
import os.path as op 
import shutil

import numpy as np
import nibabel
from fsl.wrappers.flirt import flirt as flirt_cmd
from fsl.wrappers.flirt import mcflirt as mcflirt_cmd
from fsl.wrappers.fnirt import fnirt as fnirt_cmd 
from fsl.data.image import Image as FSLImage

from regtricks.transforms import Registration, MotionCorrection, NonLinearRegistration

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
        Registration object
    """

    with tempfile.TemporaryDirectory() as d: 
        if 'omat' not in kwargs: 
            kwargs['omat'] = op.join(d, 'omat.mat')
        flirt_cmd(src, ref, **kwargs)

        return Registration.from_flirt(kwargs['omat'], src, ref)


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
        MotionCorrection object
    """

    if isinstance(src, str):
        src_path = src 
        src = nibabel.load(src)

    if (refvol == -1) and (refvol not in kwargs):
        refvol = src.shape[-1] // 2
        kwargs['refvol'] = refvol
    else: 
        kwargs['refvol'] = refvol

    # Do we have a name to save output at?
    if kwargs.get('out'): 
        matsname = kwargs.get('out') + '.mat'
    elif kwargs.get('o'):
        matsname = kwargs.get('o') + '.mat'
    else: 
        out = src_path 
        if out.count('nii'): 
            out = out[:out.index('.nii')]
        matsname = out + '_mcf.mat'

    with tempfile.TemporaryDirectory() as d: 

        save_mats = ('mats' in kwargs)
        kwargs['mats'] = True 
        mcflirt_cmd(src, **kwargs)
        mc = MotionCorrection.from_mcflirt(matsname, src, src)
        if not save_mats: 
            shutil.rmtree(matsname)

    return mc 

def fnirt(src, ref, **kwargs):
    
    # # TODO: there must be a better way of doing this one day 
    print("WARNING (FNIRT): only outputs with a specified name will be "
            "saved (no defaults).")

    with tempfile.TemporaryDirectory() as d: 
        if not 'cout' in kwargs: 
            kwargs['cout'] = op.join(d, 'coeffs.nii.gz')
        fnirt_cmd(src, ref=ref, **kwargs)
        coeffs = nibabel.load(kwargs['cout'])
        coeffs.get_fdata()
        return NonLinearRegistration.from_fnirt(coeffs, src, ref)