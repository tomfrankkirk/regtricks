import os.path as op 

import nibabel 
import numpy as np 

from image_space import ImageSpace 




class Registration(object):

    def __init__(self, src2ref, src=None, ref=None, convention=""):

        if isinstance(src2ref, str): 
            src2ref = np.loadtxt(src2ref)

        if (src2ref.shape != (4,4) or 
            (np.abs(src2ref[3,:] - [0,0,0,1]) < 1-9).all()):
            raise RuntimeError("src2ref must be a 4x4 affine matrix, where " + 
                "the last row is [0,0,0,1].")

        if (src is not None) and (ref is not None):  
            self.src_spc = ImageSpace(src)
            self.ref_spc = ImageSpace(ref)

            if convention == "":
                print("Assuming FSL convention")
                convention = "fsl"

        else: 
            self.src_spc = None
            self.ref_spc = None 
            if convention == "":
                print("Assuming world convention")
                convention = "world"

        if convention == "fsl":
            s2r_world = (self.ref_spc.FSL2world @ 
                            src2ref @ self.src_spc.world2FSL)
            self.__src2ref_world = s2r_world 

        elif convention == "world":
            self.__src2ref_world = src2ref 

        else: 
            raise RuntimeError("Unrecognised convention")


    @property
    def src_header(self):
        if self.src_spc is not None: 
            return self.src_spc.header 
        else: 
            return None 

    @property
    def ref_header(self):
        if self.ref_spc is not None: 
            return self.ref_spc.header 
        else: 
            return None 
        
    @property
    def ref2src_world(self):
        return np.linalg.inv(self.__src2ref_world)

    @property
    def src2ref_world(self):
        return self.__src2ref_world

    def inverse(self):
        return Registration(self.ref2src_world, src=self.ref_spc, 
            ref=self.src_spc, convention='world')

    def to_flirt(self, src, ref):
        if not isinstance(src, ImageSpace):
            src = ImageSpace(src)
        if not isinstance(ref, ImageSpace):
            ref = ImageSpace(ref)

        return ref.world2FSL @ self.src2ref_world @ src.FSL2world

    def to_FSL(self, src, ref):
        return self.to_flirt(src, ref)
        
    def save_txt(self, fname):
        print("Saving in world convention")
        np.savetxt(fname, self.src2ref_world)

    @classmethod
    def chain(cls, registrations):
        """ 
        Concatenate a series of registrations.

        Args: 
            registrations: iterable of Registration objects, given in order 
            that they need to be applied (eg, for A -> B -> C, give them in
            that order and they will be multiplied in the correct order)

        Returns: 
            Registration object, with the first registrations' source 
            and the last's reference 
        """

        if isinstance(registrations, Registration):
            print("One registration: do nothing")
            chained = registrations
        else: 
            src = registrations[0].src_spc 
            ref = registrations[-1].ref_spc 
            overall_world = np.eye(4)
            for r in registrations: 
                overall_world = overall_world @ r.src2ref_world

            chained = Registration(overall_world, src, ref, "world")



if __name__ == "__main__":
    
    src = 'asl.nii.gz'
    ref = 'struc_brain.nii.gz'

    asl2struct = np.loadtxt('asl2struct_flirt.mat')

    reg = Registration(asl2struct, src, ref)
    print(reg.to_flirt(src, ref))