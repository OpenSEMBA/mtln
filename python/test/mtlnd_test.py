import numpy as np
import matplotlib.pyplot as plt

from bigtree import Node as level
from bigtree import print_tree as print_mtlnd
from bigtree import list_to_tree, levelordergroup_iter

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light
import scipy.linalg as linalg

import src.mtlnd as mtld
import src.mtln as mtln
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
       
def test_init():

    #level 0    
    line0 = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, nx = 4, Zs=150)
    
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    #level 1
    line1 = mtl.MTL(l=l, c=c, length=400.0, nx = 4, Zs=150)
    #level 2;0
    line2_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, nx = 4, Zs=150)
    #level 2;1
    line2_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, nx = 4, Zs=150)
    
    bundle = mtld.MTLND(line0,0)
    bundle.add_mtl(line1, 1)
    bundle.add_mtl(line2_0, 2)
    bundle.add_mtl(line2_1, 2)
    
    assert(bundle.number_of_conductors() == 5)
    assert(bundle.get_conductors_in_levels()[0] == 1)
    assert(bundle.get_conductors_in_levels()[1] == 2)
    assert(bundle.get_conductors_in_levels()[2] == 2)
    
    Z01 = np.zeros([1,2])
    Z01[0] = [0.2, 0.2]
    Z12 = np.zeros([2,2])
    Z12[0] = [0.3, 0.0]
    Z12[1] = [0.0, 0.3]
    
    bundle.build_transfer_impedance_matrix()
    bundle.add_transfer_impedance(0,1,Z01)
    bundle.add_transfer_impedance(1,2,Z12)

    assert(bundle.Zt.shape == (5,5))
    assert(d == 0 for d in np.diag(bundle.Zt))

def test_ribbon_cable_20ns_termination_network():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    """
    Solved with: 
    * mtl approach 
    * mltn approach: tube + 2 termination networks
    * mltn + domains: bundle (with just one level) + 2 termination networks (with one level)
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)


    """
     _             _
    | |     1     | |
    | 1-----------3 |
    | |     0     | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """

    line_0 = mtl.MTL(l=l, c=c, length=400.0, nx = 4, Zs=150)
    system = mtld.MTLND(line_0,0)
    system.build_transfer_impedance_matrix()

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = line_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R  = 50)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    system.add_network(terminal_1, level = 0)

    #network definition
    terminal_2 = mtln.Network(nw_number = 1 ,nodes = [2,3], bundles = [0])
    bundle_connections= [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    terminal_2.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = line_0, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_2.connect_to_ground(2, 50)
    terminal_2.connect_to_ground(3, 50)
    system.add_network(terminal_2, level = 0)

    system.run_until(finalTime)

