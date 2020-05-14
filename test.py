import regtools as rt 
import numpy as np 
import nibabel
from nibabel import Nifti2Image, MGHImage
from fsl.data.image import Image as FSLImage

LAST_ROW = [0,0,0,1]
MAT = np.vstack((np.random.rand(3,4), LAST_ROW))
MATS = [ np.vstack((np.random.rand(3,4), LAST_ROW)) for _ in range(10) ]

V2W1 = np.eye(4)
V2W2 = np.eye(4)
V2W2[[0,1,2], [0,1,2]] = 2
SPC1 = rt.ImageSpace.create_axis_aligned(np.zeros(3), 10 * np.ones(3), np.ones(3))
SPC2 = SPC1.resize(-5 * np.ones(3), 20 * np.ones(3))
SPC2 = SPC2.resize_voxels(2)

def equal_tolerance(x, y, frac):
    m = (x > frac * y.max())
    return (np.abs(x - y)[m] < frac * y.max()).all()


TD = 'testdata/'
ASLT = TD + 'asl_target.nii.gz'
ASL = TD + 'asl.nii.gz'
BRAIN = TD + 'brain.nii.gz'
MNI = TD + 'MNI152_T1_2mm.nii.gz'


def test_create_identity():
    r = rt.Registration.identity()
    assert np.allclose(r.src2ref_world, np.eye(4))


def test_inverse(): 
    r = rt.Registration(MAT)
    assert np.allclose(r.ref2src_world, np.linalg.inv(MAT))

    mc = rt.MotionCorrection(MATS)
    len(mc)
    for invm,m in zip(mc.inverse().src2ref_world, MATS):
        assert np.allclose(invm, np.linalg.inv(m))


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


def test_fsl_inverse(): 
    r = rt.Registration(MAT)
    assert np.allclose(np.linalg.inv(r.to_fsl(SPC1, SPC2)), r.inverse().to_fsl(SPC2, SPC1))


def test_imagespace_resize():
    s2 = SPC1.resize(3*[-5], 3*[20])
    assert np.allclose(SPC1.vox2world[:3,:3], s2.vox2world[:3,:3])
    assert np.allclose(s2.bbox_origin, SPC1.bbox_origin - 5)


def test_imagespace_resize_voxels():
    s2 = SPC1.resize_voxels(2)
    assert np.allclose(s2.bbox_origin, SPC1.bbox_origin)
    assert np.allclose(2 * SPC1.vox2world[:3,:3], s2.vox2world[:3,:3])


def test_image_types():

    r = rt.Registration.identity()
    v = np.zeros((10,10,10), dtype=np.float32)
    nii = Nifti2Image(v, np.eye(4))
    mgh = MGHImage(v, np.eye(4))
    fsl = FSLImage(v, xform=np.eye(4))

    for img in [nii, mgh, fsl]:
        img2 = r.apply_to_image(img, img)
        assert type(img) is type(img2)


def test_apply_array(): 
    r = rt.Registration.identity()
    v = np.zeros((10,10,10), dtype=np.float32)

    x = r.apply_to_array(v, SPC1, SPC1)
    assert (x == v).all()

    v = np.zeros((10,10,10,10), dtype=np.float32)
    x = r.apply_to_array(v, SPC1, SPC1)
    assert (x == v).all()

    mc = rt.MotionCorrection([ np.eye(4) for _ in range(10) ])
    x = mc.apply_to_array(v, SPC1, SPC1)
    assert(x == v).all()


def test_mcasl():
    r = rt.MotionCorrection('testdata/mcasl.mat', ASLT, ASLT)
    x = r.apply_to_image(ASL, ASLT, order=1)
    t = nibabel.load(TD + 'mcasl_truth.nii.gz').get_data()
    # nibabel.save(x, 'mcasl.nii.gz')
    assert equal_tolerance(x.dataobj, t, 0.1)


def test_asl2brain():
    r = rt.Registration(TD + 'asl2brain.mat', ASLT, BRAIN)
    x = r.apply_to_image(ASLT, BRAIN, order=1)
    t = nibabel.load(TD + 'asl2brain_truth.nii.gz').dataobj
    assert (t - x.dataobj < 0.01 * np.max(t)).all() 


def test_brain2asl():
    r = rt.Registration(TD + 'asl2brain.mat', ASLT, BRAIN)
    x = r.inverse().apply_to_image(BRAIN, ASLT, order=1)
    t = nibabel.load(TD + 'brain2asl_truth.nii.gz').dataobj
    assert (t - x.dataobj < 0.01 * np.max(t)).all() 


def test_brain2MNI():
    r = rt.NonLinearRegistration(TD + 'brain2MNI_coeffs.nii.gz', BRAIN, MNI)
    x = r.apply_to_image(BRAIN, MNI, order=1)
    t = nibabel.load(TD + 'brain2MNI_truth.nii.gz').get_data()
    assert equal_tolerance(x.dataobj, t, 0.1)



# def test_MNI2brain():
#     r = rt.NonLinearRegistration(TD + 'brain2MNI_coeffs.nii.gz', BRAIN, MNI)
#     x = r.inverse().apply_to_image(MNI, BRAIN, order=1)
#     t = nibabel.load(TD + 'MNI2brain_truth.nii.gz').dataobj
#     # nibabel.save(x, 'MNI2brain.nii.gz')
#     assert (t - x.dataobj < 0.01 * np.max(t)).all() 
 

def test_asl2MNI():
    r1 = rt.Registration(TD + 'asl2brain.mat', ASLT, BRAIN)
    r2 = rt.NonLinearRegistration(TD + 'brain2MNI_coeffs.nii.gz', BRAIN, MNI)
    x = rt.chain(r1, r2).apply_to_image(ASLT, MNI, order=1)
    t = nibabel.load(TD + 'asl2MNI_truth.nii.gz').dataobj
    assert (t - x.dataobj < 0.01 * np.max(t)).all() 


# def test_mcasl2brain():
#     r1 = rt.MotionCorrection('testdata/asl_mcf.mat', ASLT, ASLT)
#     r2 = rt.Registration(TD + 'asl2brain.mat', ASLT, BRAIN)
#     x = rt.chain(r1, r2).apply_to_image(ASL, BRAIN, order=1)
#     t = nibabel.load(TD + 'mcasl2brain_truth.nii.gz').dataobj
#     assert (t - x.dataobj < 0.01 * np.max(t)).all() 


# def test_brain2MNI2brain():
#     r1 = rt.NonLinearRegistration(TD + 'brain2MNI_coeffs.nii.gz', BRAIN, MNI)
#     x = rt.chain(r1, r1.inverse()).apply_to_image(BRAIN, BRAIN, order=1)
#     t = nibabel.load(TD + 'brain2MNI2brain_truth.nii.gz').dataobj
#     nibabel.save(x, 'brain2MNI2brain_rt.nii.gz')
#     assert (t - x.dataobj < 0.01 * np.max(t)).all() 


if __name__ == "__main__":
    test_mcasl()