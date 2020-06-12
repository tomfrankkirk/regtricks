"""
Functions for combining Transformations
"""

# Lots of relative imports within functions here to avoid circular 
# imports 
from collections import defaultdict
import copy 

import numpy as np 

def get_highest_type(first, second):
    """
    When combining two arbitrary transforms, the output will be of the same 
    type as the "highest" of the two arguments (this is the "type promotion"). 
    This function returns the highest type of two input objects, according to:

    Registration                    LOWEST
    MotionCorrection
    NonLinearReigstration
    NonLinearMotionCorrection       HIGHEST 

    Once the higest type is known, the actual multiplication is handled by
    that class' invididual method, as below in this submodule

    Args: 
        first (transformation): order doesn't matter
        second (transformation): order doesn't matter

    Returns: 
        type object of the highest class 
    """

    from regtricks.transforms import (Registration, MotionCorrection,
            NonLinearMotionCorrection, NonLinearRegistration)

    TYPE_MAP = ({
        Registration: 1, 
        MotionCorrection: 2, 
        NonLinearRegistration: 3, 
        NonLinearMotionCorrection: 4
    })

    try: 
        type1 = TYPE_MAP[type(first)]
        type2 = TYPE_MAP[type(second)]
        if type1 >= type2: 
            return type(first)
        else: 
            return type(second)

    except Exception as e: 
        raise ValueError("At least one input was not a Transform")
    

def registration(lhs, rhs):
    """
    Combine two Registrations, return a Registration. 
    """

    from regtricks.transforms import Registration

    # lhs   rhs 
    # reg @ reg 
    if (type(lhs) is Registration and type(rhs) is Registration): 
        overall = lhs.src2ref @ rhs.src2ref
        return Registration(overall)

    else: 
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")


def moco(lhs, rhs):
    """
    Combine either a Registration and MoCo, or two MoCos. 
    Return a MotionCorrection. 
    """

    from regtricks.transforms import MotionCorrection, Registration

    # lhs   rhs 
    # reg @ MC
    if type(lhs) is Registration: 
        overall = [ lhs.src2ref @ m for m in rhs.src2ref ]

    # lhs  rhs 
    # MC @ reg
    elif type(rhs) is Registration: 
        overall = [ m @ rhs.src2ref for m in lhs.src2ref ]

    # lhs  rhs 
    # MC @ MC 
    elif (type(lhs) is MotionCorrection and type(rhs) is MotionCorrection): 
        overall = [ l @ r for l,r in zip(lhs.src2ref, rhs.src2ref) ]

    else:
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")

    return MotionCorrection(overall)

def nonlinearreg(lhs, rhs):
    """
    Combine either a Registration and NLR, a MoCo and NLR, or two NLRs.
    Note at most 2 non-linear transforms can be combined. 
    Return a NonLinearRegistration. 
    """

    from regtricks.transforms import (NonLinearRegistration, Registration, 
                            MotionCorrection, NonLinearMotionCorrection)
    from regtricks.fnirt_coefficients import NonLinearProduct

    # lhs    rhs 
    # NLR @ other
    # Note that this matches both registration and motion correction
    if type(rhs) is not NonLinearRegistration: 
        if type(rhs) is Registration: 
            constructor = NonLinearRegistration._manual_construct
        else: 
            constructor = NonLinearMotionCorrection

        pre = lhs.premat @ rhs 
        return constructor(lhs.warp, pre, 
                           lhs.postmat, lhs._intensity_correct)

    #  lhs    rhs 
    # other @ NLR
    # Note that this matches both registration and motion correction
    elif type(lhs) is not NonLinearRegistration: 
        if type(lhs) is Registration: 
            constructor = NonLinearRegistration._manual_construct
        else: 
            constructor = NonLinearMotionCorrection

        post = lhs @ rhs.postmat
        return constructor(rhs.warp, rhs.premat, 
                           post, rhs._intensity_correct)

    # lhs   rhs 
    # NLR @ NLR
    elif (type(lhs) is NonLinearRegistration 
          and type(rhs) is NonLinearRegistration): 

        warp = NonLinearProduct(rhs.warp, rhs.postmat, lhs.premat, lhs.warp)
        if (lhs.intensity_correct and rhs.intensity_correct): icorr = 3
        elif lhs.intensity_correct: icorr = 2
        elif rhs.intensity_correct: icorr = 1
        else: icorr = 0
        ret = NonLinearRegistration._manual_construct(warp, rhs.premat, lhs.postmat, icorr)
        return ret 

    else: 
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")
        

def nonlinearmoco(lhs, rhs):
    """
    Combine either a Registration and NLMC, a MoCo and NLMC, a NLR and NLMC, 
    or two NLMCs. Note at most 2 non-linear transforms can be combined. 
    Return a NonLinearMotionCorrection. 
    """

    from regtricks.transforms import (NonLinearRegistration, Registration, 
                            NonLinearMotionCorrection)
    from regtricks.fnirt_coefficients import NonLinearProduct

    # lhs    rhs 
    # NLMC @ other
    # Note that this matches both registration and motion correction
    if isinstance(rhs, Registration): 
        pre = lhs.premat @ rhs 
        return NonLinearMotionCorrection(rhs.warp, pre, lhs.postmat, lhs._intensity_correct)

    #  lhs    rhs 
    # other @ NLMC
    # Note that this matches both registration and motion correction
    elif isinstance(lhs, Registration): 
        post = lhs @ rhs.postmat
        return NonLinearMotionCorrection(rhs.warp, 
            rhs.premat, post, rhs._intensity_correct)

    # lhs  rhs 
    # NL @ NL
    # Note that this matches both non-lin reg and non-lin moco. 
    elif (isinstance(lhs, NonLinearRegistration)
          and isinstance(rhs, NonLinearRegistration)):
 
        warp = NonLinearProduct(rhs.warp, rhs.postmat, lhs.premat, lhs.warp)
        if (lhs.intensity_correct and rhs.intensity_correct): icorr = 3
        elif lhs.intensity_correct: icorr = 2
        elif rhs.intensity_correct: icorr = 1
        else: icorr = 0
        return NonLinearMotionCorrection(warp, rhs.premat, lhs.postmat, icorr)

    else: 
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")


def chain(*args):
    """ 
    Concatenate a series of transformations (Registration, MotionCorrection, 
    NonLinearRegistration). Note that intensity correction should be enabled 
    when creating a NonLinearRegistration object using intensity_correct=True
    in the constructor prior to using chain(). 

    Args: 
        *args: Transform objects, given in the order that they need to be 
            applied (eg, for A -> B -> C, give them in that order and they 
            will be multiplied as C @ B @ A)

    Returns: 
        Transform object representing the complete transformation 
    """

    from regtricks.transforms import Transform

    if not all([ isinstance(r, Transform) for r in args ]):
        raise RuntimeError("Each item in sequence must be a",
                        " Registration, MotionCorrection or NonLinearRegistration")

    # Do nothing for a single or no args 
    if len(args) == 0: return []
    elif len(args) == 1: return args[0]  

    # Two: multiply them in reverse order 
    elif len(args) == 2:
        return args[1] @ args[0]

    # Everything else: multiply the last one by the chain
    # of the remainder  
    else: 
        return args[-1] @ chain(*args[:-1])


def cast_potential_array(arr):
    """Helper to convert 4x4 arrays to Registrations if not already"""

    from regtricks.transforms import Registration, Transform

    if type(arr) is np.ndarray: 
        assert arr.shape == (4,4)
        arr = copy.deepcopy(arr)
        arr = Registration(arr)
    else: 
        if not isinstance(arr, Transform):
            raise ValueError("Not a Transform nor array")
    return arr
