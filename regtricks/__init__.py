# from .base import MotionCorrection, Registration
# from .image_space import ImageSpace

from .image_space import ImageSpace
from .regtricks import Registration, MotionCorrection, NonLinearRegistration
from .wrappers import flirt, mcflirt, fnirt
from .x5_interface import load_manager as load
from .multiplication import chain