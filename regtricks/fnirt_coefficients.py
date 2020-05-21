import tempfile 
import subprocess
import os.path as op 
import itertools 

import nibabel 
import numpy as np 

from . import ImageSpace

class FNIRTCoefficients(object):
    """
    Private encapsulation of FNIRT warp field. Only to be used 
    from within a NonLinearTransformation

    Args: 
        coeffs; nibabel object or path to coefficients file
        src: str or ImageSpace, path to original source for transform
        ref: str or ImageSpace, path to original reference for transform
    """

    def __init__(self, coeffs, src, ref):
        if isinstance(coeffs, str):
            coeffs = nibabel.load(coeffs)
        else: 
            assert isinstance(coeffs, nibabel.Nifti1Image)

        # if not (coeffs.header.get_intent()[0] in 
        #     [ 'fnirt cubic spline coef', 'fnirt quad spline coef' ]): 
        #     raise ValueError("Coefficients file is not FNIRT compatible")       
        self.coefficients = coeffs 

        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)
        self.ref_spc = ref 

        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        self.src_spc = src 

    def get_displacements(self, ref, postmat, length=1): 

        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        # If we have series of postmats, check if they are all identitical
        if len(postmat) > 1: 
            same1 = all([ np.allclose(postmat.src2ref_world[0], m) 
                          for m in postmat.src2ref_world ])
            if same1: 
                pmat = postmat.transforms[0].to_fsl(self.ref_spc, ref)
            else: 
                pmat = [ t.to_fsl(self.ref_spc, ref) for t in postmat.transforms ] 
        else: 
            same1 = True 
            pmat = postmat.to_fsl(self.ref_spc, ref)

        if same1: 
            dfield = get_field(self.coefficients, ref, post=pmat)
            for _, df in zip(range(length), itertools.repeat(dfield)): 
                yield df

        else: 
            for _, p in zip(range(length), pmat): 
                yield get_field(self.coefficients, ref, post=p)


class NonLinearProduct(object):

    def __init__(self, first, first_post, second_pre, second):
        
        if ((type(first) is NonLinearProduct) 
            or (type(second) is NonLinearProduct)): 
            raise NotImplementedError("Cannot chain more than two non-linear transforms")

        self.warp1 = first
        self.warp2 = second

        self.src_spc = first.src_spc 
        self.ref_spc = second.ref_spc 
        
        self.midmat = second_pre @ first_post


    def get_displacements(self, ref, postmat, length=1):
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        # If we have series of mid and postmats, check if they are all identitical
        if len(postmat) > 1: 
            same1 = all([ np.allclose(postmat.src2ref_world[0], m) 
                          for m in postmat.src2ref_world ])
            if same1: 
                pmat = itertools.repeat(postmat.transforms[0].to_fsl(self.warp2.ref_spc, ref))
            else: 
                pmat = [ t.to_fsl(self.warp2.ref_spc, ref) for t in postmat.transforms ] 
        else: 
            same1 = True 
            pmat = itertools.repeat(postmat.to_fsl(self.warp2.ref_spc, ref))

        if len(self.midmat) > 1: 
            same2 = all([ np.allclose(self.midmat.src2ref_world[0], m) 
                          for m in self.midmat.src2ref_world ])
            if same2: 
                mmat = itertools.repeat(self.midmat.transforms[0].to_fsl(self.warp1.ref_spc, self.warp2.src_spc))
            else: 
                mmat = [ t.to_fsl(self.warp1.ref_spc, self.warp2.src_spc) for t in self.midmat.transforms ]
        else: 
            same2 = True 
            mmat = itertools.repeat(self.midmat.to_fsl(self.warp1.ref_spc, self.warp2.src_spc))

        same = (same1 & same2)

        # If they are all the same, compute the field once 
        if same: 
            dfield = get_field(self.warp1.coefficients, ref, self.warp2.coefficients, next(mmat), next(pmat))
            for _, df in zip(range(length), itertools.repeat(dfield)): 
                yield df

        # If not, compute the field repeatedly for each (mid, post) pair
        else: 
            for _, m, p in zip(range(length), mmat, pmat):
                yield get_field(self.warp1.coefficients, ref, self.warp2.coefficients, m, p)


def get_field(coeff1, ref, coeff2=None, mid=None, post=None):
    with tempfile.TemporaryDirectory() as d: 
        w1 = op.join(d, 'w1.nii.gz')
        nibabel.save(coeff1, w1)
        refvol = op.join(d, 'ref.nii.gz')
        ref.touch(refvol)
        cmd = f'convertwarp --warp1={w1} --ref={refvol}'

        if coeff2 is not None: 
            w2 = op.join(d, 'w2.nii.gz')
            nibabel.save(coeff2, w2)
            cmd += f' --warp2={w2}'        

        if mid is not None: 
            m = op.join(d, 'mid.mat')
            np.savetxt(m, mid)
            cmd += f' --midmat={m}'

        if post is not None: 
            p = op.join(d, 'post.mat')
            np.savetxt(p, post)
            cmd += f' --postmat={p}'

        field = op.join(d, 'field.nii.gz')
        cmd += f' --out={field} --absout'

        subprocess.run(cmd, shell=True)  
        field = nibabel.load(field).get_data().reshape(-1,3)
    return field 


def det_jacobian(vec_field, vox_size):
    """
    Calculate determinant of Jacobian for vector field, with homogenous
    spacing along each axis. Second order central differences are used 
    to estimate partial derivatives. 

    Args: 
        vec_field (np.ndarray): sized XYZ3, where the last dimension 
            corresponds to displacements along the x,y,z axis respectively
        vox_size (np.ndarray): array sized 3, step size along each spatial axis

    Returns: 
        (np.ndarray), sized XYZ, values of the determinant of the Jacobian
            matrix evaluated at each point within the array 

    """

    if not ((len(vec_field.shape) == 4) and (vec_field.shape[3] == 3)):
        raise ValueError("vec_field should be a 4D array with size 3 in "
            "the last dimension")

    if not len(vox_size) == 3: 
        raise ValueError("vox_size should be a 3-vector of dimensions") 

    # Calculate partial derivatives in each direction. Note that each of these
    # returns an array of size (X,Y,Z,3), arranged d/dx, d/dy, d/dz in the last
    # dimension. So dfx is a stack of arrays df(i)/dx, df(i)/dy, df(i)/dz. We
    # need to calculate the derivative with respect to each direction because
    # f is a vector field, ie, it contains i,j,k components, each of which are
    # independent functions of x,y,z
    dfi = np.gradient(vec_field, vox_size[0], axis=0)
    dfj = np.gradient(vec_field, vox_size[1], axis=1)
    dfk = np.gradient(vec_field, vox_size[2], axis=2)

    # Construct an array of Jacobian matrices, sized (3,3,X,Y,Z) (ie, each 
    # point in space get its own 3x3 Jacobian matrix). The elements need
    # to be arranged df(i)/dx --- df(i)/dz 
    #                   |      |     |
    #                df(k)/dx --- df(k)/dz
    jacobian = np.array([[dfi[...,0], dfi[...,1], dfi[...,2]],
                         [dfj[...,0], dfj[...,1], dfj[...,2]],
                         [dfk[...,0], dfk[...,1], dfk[...,2]]
                        ])

    # Reshape into a single stack of (N,3,3) arrays and calculate the det
    # Return an array of scalars sized (XYZ) again 
    jacobian = np.moveaxis(jacobian.reshape(3,3,-1), 2, 0)
    jdet = np.linalg.det(jacobian)
    return jdet.reshape(vec_field.shape[:3])
