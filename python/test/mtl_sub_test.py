import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl_sub as mtls
from src.networkExtraction import *

import src.waveforms as wf

import skrf as rf
from skrf.media import DistributedCircuit
from skrf.media import Coaxial

def test_coaxial_wire():
    """ 
    coaxial wire over reference plane
    Transfer impedance Z01 relates level 0 (outside the coaxial)
    with level 1 (inside the coaxial)
    Level 0 has a 1x1 L
    Level 1 has a 2x2 L
    """
    subdom = mtls()
    subdom.add_level()
    
    finalTime = 200e-9
    subdom.run_until(finalTime)

    