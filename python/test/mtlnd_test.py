import numpy as np
import matplotlib.pyplot as plt

from bigtree import Node as level
from bigtree import print_tree as print_mtlnd
from bigtree import list_to_tree, levelordergroup_iter

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light
import scipy.linalg as linalg

import src.mtlnd as mtld
import src.mtl as mtl

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
    subdom = mtld()
    subdom.add_level()
    
    finalTime = 200e-9
    subdom.run_until(finalTime)

def get_tree_number_of_conductors(tree):
    number_of_conductors = 1
    
    for group in levelordergroup_iter(tree):
        for node in group:
            number_of_conductors += node.conductors_inside
    return number_of_conductors

def is_root(node):
    return node.root == node

def is_parent_shared(node1, node2):
    return node1.parent == node2.parent

def test_bigtree():
    
    line0 = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, nx = 4, Zs=150)
    
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    line1 = mtl.MTL(l=l, c=c, length=400.0, nx = 4, Zs=150)
    line2_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, nx = 4, Zs=150)
    line2_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, nx = 4, Zs=150)
    
    l0   = level("L0_M0",    level = 0, mtl = line0,   conductor=0, n_cond= 1, conductors_inside = 2)
    l1_0 = level("L1_M0_C0", level = 1, mtl = line1,   conductor=0, n_cond= 2, conductors_inside = 1, parent=l0)
    l1_1 = level("L1_M0_C1", level = 1, mtl = line1,   conductor=1, n_cond= 2, conductors_inside = 1, parent=l0)
    l2_0 = level("L2_M0_C0", level = 2, mtl = line2_0, conductor=0, n_cond= 1, conductors_inside = 0, parent=l1_0)
    l2_1 = level("L2_M1_C0", level = 2, mtl = line2_1, conductor=0, n_cond= 1, conductors_inside = 0, parent=l1_1)

    lz = l0.mtl.l.shape[0] 
    nc = get_tree_number_of_conductors(l0)
    l = np.empty(shape=(lz, nc, nc))
    l[:] = 0
    zt = np.empty(shape=(lz, nc, nc))
    zt[:] = 0

    mltnd_v = np.zeros(nc)
    mltnd_i = np.zeros(nc)

    i = 0
    for group in levelordergroup_iter(l0, filter_condition=lambda x: x.conductor == 0):
        for node in group:
                
            nnc = node.mtl.l.shape[1]
            for k in range(lz):
                l[k,i:i+nnc,i:i+nnc] = node.mtl.l[k,:,:]

            i += nnc

            
    print(l)
       
