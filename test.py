import os.path as op 
import regtools as rt 
import numpy as np 

LAST_ROW = [0,0,0,1]
MAT = np.vstack((np.random.rand(3,4), LAST_ROW))
MATS = [ np.vstack((np.random.rand(3,4), LAST_ROW)) for _ in range(10) ]

def test_create_identity():
    r = rt.Registration.identity()
    assert np.array_equal(r.src2ref_world, np.eye(4))

def test_inverse(): 
    r = rt.Registration(MAT)
    assert np.array_equal(r.ref2src_world, np.linalg.inv(MAT))

    mc = rt.MotionCorrection(MATS)
    len(mc)
    for invm,m in zip(mc.inverse().src2ref_world, MATS):
        assert np.array_equal(invm, np.linalg.inv(m))

def test_type_promotion():
    r = rt.Registration(MAT)
    m = rt.MotionCorrection(MATS)
    x = r @ m
    assert type(x) is rt.MotionCorrection
    assert len(x) == len(MATS)
    x = m @ r
    assert type(x) is rt.MotionCorrection
    assert len(x) == len(MATS)

    x = m @ MAT
    assert type(x) is rt.MotionCorrection
    assert len(x) == len(MATS)
    x = MAT @ m 
    assert type(x) is rt.MotionCorrection
    assert len(x) == len(MATS)

    x = MAT @ r
    assert type(x) is rt.Registration
    x = r @ MAT 
    assert type(x) is rt.Registration


if __name__ == "__main__":
    test_type_promotion()