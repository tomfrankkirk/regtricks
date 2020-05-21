from collections import defaultdict
import copy 

import numpy as np 

# TODO: doc this 


def get_highest_type(first, second):
    from .regtricks import (Registration, MotionCorrection,
            NonLinearMotionCorrection, NonLinearRegistration)

    _TYPE_MAP = defaultdict(int)
    _TYPE_MAP.update({
        Registration: 1, 
        MotionCorrection: 2, 
        NonLinearRegistration: 3, 
        NonLinearMotionCorrection: 4
    })

    type1 = _TYPE_MAP[type(first)]
    type2 = _TYPE_MAP[type(second)]
    if type1 >= type2: 
        return type(first)
    else: 
        return type(second)
    

def registration(lhs, rhs):
    from .regtricks import Registration

    # lhs   rhs 
    # reg @ reg 
    if (type(lhs) is Registration and type(rhs) is Registration): 
        overall = lhs.src2ref_world @ rhs.src2ref_world
        return Registration(overall, rhs.src_spc, lhs.ref_spc, "world")

    else: 
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")


def moco(lhs, rhs):
    from .regtricks import MotionCorrection, Registration

    # lhs   rhs 
    # reg @ MC
    if type(lhs) is Registration: 
        overall = [ lhs.src2ref_world @ m for m in rhs.src2ref_world ]

    # lhs  rhs 
    # MC @ reg
    elif type(rhs) is Registration: 
        overall = [ m @ rhs.src2ref_world for m in lhs.src2ref_world ]

    # lhs  rhs 
    # MC @ MC 
    elif (type(lhs) is MotionCorrection and type(rhs) is MotionCorrection): 
        overall = [ l @ r for l,r in zip(lhs.src2ref_world, rhs.src2ref_world) ]

    else:
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")

    return MotionCorrection(overall, rhs.src_spc, lhs.ref_spc, "world")

def nonlinearreg(lhs, rhs):
    from .regtricks import (NonLinearRegistration, NonLinearProduct,
                            Registration, MotionCorrection, 
                            NonLinearMotionCorrection)

    # lhs    rhs 
    # NLR @ other
    # Note that this matches both registration and motion correction
    if type(rhs) is not NonLinearRegistration: 
        if type(rhs) is Registration: 
            constructor = NonLinearRegistration._manual_construct
        else: 
            constructor = NonLinearMotionCorrection

        pre = lhs.premat @ rhs 
        return constructor(lhs.warp, rhs.src_spc, lhs.ref_spc, pre, 
                           lhs.postmat, lhs.intensity_correct)

    #  lhs    rhs 
    # other @ NLR
    # Note that this matches both registration and motion correction
    elif type(lhs) is not NonLinearRegistration: 
        if type(lhs) is Registration: 
            constructor = NonLinearRegistration._manual_construct
        else: 
            constructor = NonLinearMotionCorrection

        post = lhs @ rhs.postmat
        return constructor(rhs.warp, rhs.src_spc, lhs.ref_spc, rhs.premat, 
                           post, rhs.intensity_correct)

    # lhs   rhs 
    # NLR @ NLR
    elif (type(lhs) is NonLinearRegistration 
          and type(rhs) is NonLinearRegistration): 
        if (rhs.intensity_correct and lhs.intensity_correct): 
            raise NotImplementedError("Cannot apply intensity correction "
                        "for two non-linear registrations")

        warp = NonLinearProduct(rhs.warp, rhs.postmat, lhs.premat, lhs.warp)
        if (lhs.intensity_correct and rhs.intensity_correct): icorr = 3
        elif lhs.intensity_correct: icorr = 2
        elif rhs.intensity_correct: icorr = 1
        else: icorr = 0
        ret = NonLinearRegistration._manual_construct(warp, rhs.src_spc, 
                lhs.ref_spc, rhs.premat, lhs.postmat)
        ret._intensity_correct = icorr 
        return ret 

    else: 
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")
        

def nonlinearmoco(lhs, rhs):
    from .regtricks import (NonLinearRegistration, NonLinearProduct, 
                            Registration, NonLinearMotionCorrection)

    # lhs    rhs 
    # NLMC @ other
    # Note that this matches both registration and motion correction
    if isinstance(rhs, Registration): 
        pre = lhs.premat @ rhs 
        return NonLinearMotionCorrection(rhs.warp, 
            rhs.src_spc, lhs.ref_spc, pre, lhs.postmat, lhs.intensity_correct)

    #  lhs    rhs 
    # other @ NLMC
    # Note that this matches both registration and motion correction
    elif isinstance(lhs, Registration): 
        post = lhs @ rhs.postmat
        return NonLinearMotionCorrection(rhs.warp, 
            rhs.src_spc, lhs.ref_spc, rhs.premat, post, rhs.intensity_correct)

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
        return NonLinearMotionCorrection(warp, rhs.src_spc, 
                lhs.ref_spc, rhs.premat, lhs.postmat, icorr)

    else: 
        raise NotImplementedError("Cannot interpret multiplication of "
                f"{type(lhs)} with {type(rhs)}")


def chain(*args):
    """ 
    Concatenate a series of registrations.

    Args: 
        *args: Registration objects, given in the order that they need to be 
            applied (eg, for A -> B -> C, give them in that order and they 
            will be multiplied as C @ B @ A)

    Returns: 
        Registration object, with the first registration's source 
        and the last's reference (if these are not None)
    """

    from .regtricks import Transform

    if not all([ isinstance(r, Transform) for r in args ]):
        raise RuntimeError("Each item in sequence must be a",
                        " Registration, MotionCorrection or NonLinearRegistration")

    # Do nothing for a single or no args 
    if (len(args) < 2):
        return args

    # Two: multiply them in reverse order 
    elif len(args) == 2:
        return args[1] @ args[0]

    # Everything else: multiply the last one by the chain
    # of the remainder  
    else: 
        return args[-1] @ chain(*args[:-1])


def cast_potential_array(arr):
    """Helper to convert 4x4 arrays to Registrations if not already"""

    from .regtricks import Registration

    if type(arr) is np.ndarray: 
        assert arr.shape == (4,4)
        arr = copy.deepcopy(arr)
        arr = Registration(arr)
    return arr
