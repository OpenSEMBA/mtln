import numpy as np

from src.jsonParser import Parser

def test_get_nodes():
    file = 'python/testData/parser/test_4_multilines.smb.json'
    p = Parser(file)
    nodes_j1 = p._getNodesInJunction("j1")
    assert(type(nodes_j1) == list)
    assert(len(nodes_j1) == 2)    
    assert(nodes_j1[0] == [3, 5])
    assert(nodes_j1[1] == [4, 6])
    
    nodes_j2 = p._getNodesInJunction("j2")
    assert(type(nodes_j2) == list)
    assert(len(nodes_j2) == 2)    
    assert(nodes_j2[0] == [9, 11])
    assert(nodes_j2[1] == [10, 12])
    
def test_build_networks():
    file = 'python/testData/parser/test_4_multilines.smb.json'
    p = Parser(file)

    assert p.networks[0].levels[0].P1.shape == (4,4)
    assert p.networks[0].levels[0].P1[0,2] == 0.0
    assert p.networks[0].levels[0].P1[2,0] == 0.0
    assert np.abs(p.networks[0].levels[0].P1[0,0]) == 1e10

    assert p.networks[1].levels[0].P1.shape == (4,4)
    assert p.networks[1].levels[0].P1[0,1] == 0.0
    assert p.networks[1].levels[0].P1[1,0] == 0.0
    assert np.abs(p.networks[1].levels[0].P1[0,0]) == 1e10
    
    assert p.networks[2].levels[0].P1.shape == (1,1)
    assert p.networks[2].levels[0].P1[0,0] == -0.02

    assert p.networks[3].levels[0].P1.shape == (1,1)
    assert p.networks[3].levels[0].P1[0,0] == -1e6

    assert p.networks[4].levels[0].P1.shape == (1,1)
    assert p.networks[4].levels[0].P1[0,0] == -0.02

    assert p.networks[5].levels[0].P1.shape == (1,1)
    assert p.networks[5].levels[0].P1[0,0] == -0.02


