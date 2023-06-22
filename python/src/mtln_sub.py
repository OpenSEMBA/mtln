import numpy as np
import skrf as rf

from scipy import linalg

from types import FunctionType
from types import LambdaType

class MTLN_SUB:
    """
    Lossless Multiconductor Transmission Line Network
    """

    def __init__(self):
        self.L  = np.ndarray(shape=(0,0))
        self.C  = np.ndarray(shape=(0,0))
        self.Zt = np.ndarray(shape=(0,0))
        self.current_level = 0

    def add_level(self, level_number:int):
        assert (level_number == self.current_level)
        self.current_level += 1

    def add_mtl(self, level_number, line_number, l, c):
        self.L = linalg.block_diag(self.L, l)
        self.C = linalg.block_diag(self.C, c)

    def add_transfer_impedance(self, outside_level, outside_conductor, inside_level, inside_conductor):
        pass
