import json
import src.mtl as mtl
import src.mtln as mtln
import src.networks as nw
import src.waveforms as wf
from src.utils import add_t_functions as add
import numpy as np

import igraph as ig

class Parser:
    def __init__(self, filename):

        self.parsed = json.load(open(filename))
        self.probes = {}

        self._coordinates = self._getCoordinates()
        self._junctions = self._getJunctions()
        self._terminations = self._getTerminations()
        self._cables = self._getCables()
        self._bundles = self._getBundles()
        self._lines = self._getLines()
        self._nodes = self._getNodes()
        self._connectors = self._getConnectors()
        self._wires = self._getWires()
        self._probes = self._getProbes()
        self._sources = self._getSources()
        self._ends = self._getCableEnds()
        
        
        self._buildBundles()
        self._addSources()
        self._addProbes()
        self._buildNetworks()

    def run(self, finalTime, dt=0):
        mtl_nw = mtln.MTLN()
        for b in self.bundles:
            if dt:
                b.dt = dt
            mtl_nw.add_bundle(b)
        for n in self.networks:
            mtl_nw.add_network(n)
        if dt:
            mtl_nw.dt = dt
        mtl_nw.run_until(finalTime)

    def runWithExternalField(self, finalTime, dt, field, distances):
        mtl_nw = mtln.MTLN()
        for b in self.bundles:
            if dt:
                b.dt = dt
            mtl_nw.add_bundle(b)
        for n in self.networks:
            mtl_nw.add_network(n)
        if dt:
            mtl_nw.dt = dt
        
        mtl_nw.add_external_field(field, distances)
        mtl_nw.run_until(finalTime)
        
    def runWithLocalizedExternalField(self, finalTime, dt, bundle_name, field, distances, field_localization):
        mtl_nw = mtln.MTLN()
        for b in self.bundles:
            if dt:
                b.dt = dt
            mtl_nw.add_bundle(b)
        for n in self.networks:
            mtl_nw.add_network(n)
        if dt:
            mtl_nw.dt = dt
        
        mtl_nw.add_localized_external_field(bundle_name, field, distances, field_localization)
        mtl_nw.run_until(finalTime)
        
    def _getCoordinates(self):
        assert 'model' in list(self.parsed.keys())
        assert 'coordinates' in list(self.parsed["model"].keys())
        coord_dict = {}
        for c in [coord.split() for coord in self.parsed["model"]["coordinates"]]:
            coord_dict[int(c[0])] = [float(c[1]), float(c[2]), float(c[3])]
        return coord_dict
        
        
    def _getLines(self):
        assert 'model' in list(self.parsed.keys())
        assert 'elements' in list(self.parsed["model"].keys())
        lines = []

        if 'line' not in list(self.parsed["model"]["elements"].keys()) and 'polyline' not in list(self.parsed["model"]["elements"].keys()):
            return lines

        for l in [self.parsed["model"]["elements"][line] for line in self.parsed["model"]["elements"] if "line" == line or "polyline" == line]:
            for el in l:
                lines.append(el)
        return lines
        
    def _getNodes(self):
        assert 'model' in list(self.parsed.keys())
        assert 'elements' in list(self.parsed["model"].keys())
        nodes = []
        if 'node' not in list(self.parsed["model"]["elements"].keys()):
            return nodes
        
        for n in [self.parsed["model"]["elements"]["node"] for line in self.parsed["model"]["elements"] if "node" == line]:
            for el in n:
                nodes.append(el)
        return nodes

    def _getSources(self):
        assert 'model' in list(self.parsed.keys())
        if 'sources' in list(self.parsed["model"].keys()):
            return self.parsed["model"]["sources"]
        else:
            return {}

    def _getCableEnds(self):
        assert 'model' in list(self.parsed.keys())
        if 'cableEnds' in list(self.parsed["model"].keys()):
            return self.parsed["model"]["cableEnds"]
        else:
            return {}

    def _getFields(self):
        assert 'model' in list(self.parsed.keys())
        if 'fields' in list(self.parsed["model"].keys()):
            return self.parsed["model"]["fields"]
        else:
            return {}
    
    def _getProbes(self):
        assert 'model' in list(self.parsed.keys())
        if 'probes' in list(self.parsed["model"].keys()):
            return self.parsed["model"]["probes"]
        else:
            return {}

    def _getCables(self):
        assert 'model' in list(self.parsed.keys())
        assert 'cables' in list(self.parsed["model"].keys())
        return self.parsed["model"]["cables"]

    def _getBundles(self):
        assert 'model' in list(self.parsed.keys())
        assert 'bundles' in list(self.parsed["model"].keys())
        return self.parsed["model"]["bundles"]
    
    def _getConnectors(self):
        assert 'model' in list(self.parsed.keys())
        assert 'materials' in list(self.parsed["model"].keys())
        return [conn for conn in self.parsed["model"]["materials"] if conn["materialType"] == "Connector"]
    
    def _getWires(self):
        assert 'model' in list(self.parsed.keys())
        assert 'materials' in list(self.parsed["model"].keys())
        return [conn for conn in self.parsed["model"]["materials"] if conn["materialType"] == "Multiwire" or conn["materialType"] == "Wire"]
    
    def _getJunctions(self):
        assert 'model' in list(self.parsed.keys())
        if 'junctions' in list(self.parsed["model"].keys()):
            return self.parsed["model"]["junctions"]
        else:
            return {}

    def _getTerminations(self):
        assert 'model' in list(self.parsed.keys())
        if 'terminations' in list(self.parsed["model"].keys()):
            return self.parsed["model"]["terminations"]
        else:
            return {}
    
    def _getTransferImpedance(self, line):
        for c in self._cables:
            if c["name"] == line:
                for w in self._wires:
                    if w["materialId"] == c["materialId"]:
                        if "transferImpedance" in w.keys():
                            return w["transferImpedance"]
        return 0
    
    
    def _mapElIdsToNames(self):
        out = {}
        for cable in self._cables:
            out[tuple(cable["elemIds"])] = cable["name"]
            for elemId in cable["elemIds"]:
                out[elemId] = cable["name"]
        return out        
        
    def _f(self, input):
        out = []
        for el in input:
            if type(el) == dict:
                keys = [id for id in list(el.keys())]
                out.append(keys)
                values = [id for id in list(el.values())]
                self._f(values)
            else:
                return input
        
    def _buildGraphFromConnections(self, conn_dict):
        self._elemIds_to_cableName = self._mapElIdsToNames()
        def add_to_graph(graph, parent, nested_dict, level):
            
            if isinstance(nested_dict, dict):

                for child, sub_dict in nested_dict.items():
                    graph.add_vertex(name=child, level=level, cable = self._elemIds_to_cableName[int(child)])
                    parent_cable_name = self._elemIds_to_cableName[int(parent)]
                    child_cable_name = self._elemIds_to_cableName[int(child)]
                    Zt = self._getTransferImpedance(parent_cable_name)
                    graph.add_edge(parent, child, Zt = Zt)
                    
                    if isinstance(sub_dict, dict):
                        add_to_graph(graph, child, sub_dict,level+1)
                    else:
                        for node in sub_dict.split():
                            graph.add_vertex(name=node, level = level+1, cable = self._elemIds_to_cableName[int(node)])
                            Zt = self._getTransferImpedance(child_cable_name)
                            graph.add_edge(child, node, Zt=Zt)
                        
            else:
                for node in nested_dict.split():
                    graph.add_vertex(name=node, level = level, cable = self._elemIds_to_cableName[int(node)])
                    Zt = self._getTransferImpedance(self._elemIds_to_cableName[int(parent)])
                    graph.add_edge(parent, node, Zt=Zt)

        g = ig.Graph(directed=True)
        
        if not isinstance(conn_dict, dict):
            for node in conn_dict.split():
                g.add_vertex(name=node, level=0,cable = self._elemIds_to_cableName[int(node)])
            return g
        
        root_nodes = list(conn_dict.keys())
        for root in root_nodes:
            g.add_vertex(name=root, level=0, cable = self._elemIds_to_cableName[int(root)])
            add_to_graph(g, root, conn_dict[root], level = 1)

        return g
    

    def _findConductorNumberInLevel(self, graph, node, bundle):
        
        level = graph.vs[node]["level"]
        name = graph.vs[node]["cable"]
        elemId = graph.vs[node]["name"]
        

        number_of_conductors = 0

        for line in bundle.levels[level]:
            if (line.name != name):
                number_of_conductors += line.number_of_conductors
            else:
                break
        
        position_in_line = [cable["elemIds"] for cable in self._cables if cable["name"] == name][0].index(int(elemId))
        return number_of_conductors + position_in_line
    
    
    def _addConnectorsResistance(self, line):
        for c in [c for c in self._cables if c["name"] == line.name]:
            for end in self._ends:
                el_id, _ = self._getElementOfNode(end["coordIds"][0])
                if el_id in c["elemIds"]:
                    coordinate = self._coordinates[end["coordIds"][0]]
                    conn = [c for c in self._connectors if c["materialId"] == end["materialId"]][0]
                    resistance = conn["resistance"]
                    conductor = c["elemIds"].index(el_id)
                    line.set_resistance_in_region(coordinate, coordinate, conductor, resistance)
    
    def _setBundleTransferImpedance(self, bundle, graph):
        for edge in graph.es:
            bundle.add_transfer_impedance(
                out_level = graph.vs[edge.source]["level"], 
                out_level_conductors= [self._findConductorNumberInLevel(graph, edge.source, bundle)], 
                in_level = graph.vs[edge.target]["level"], 
                in_level_conductors= [self._findConductorNumberInLevel(graph, edge.target, bundle)], 
                transfer_impedance = edge['Zt'])

        for end in self._ends:
            el_id, side = self._getElementOfNode(end["coordIds"][0])
            
            for edge in [edge for edge in graph.es if graph.vs[edge.source]["name"] == str(el_id)]:
                
                conn = [c for c in self._connectors if c["materialId"] == end["materialId"]][0]
                Zt = conn["transferImpedance"]
                
                bundle.set_connector_transfer_impedance(
                    side = side,
                    out_level = graph.vs[edge.source]["level"], 
                    out_level_conductors= [self._findConductorNumberInLevel(graph, edge.source, bundle)], 
                    in_level = graph.vs[edge.target]["level"], 
                    in_level_conductors= [self._findConductorNumberInLevel(graph, edge.target, bundle)], 
                    transfer_impedance = Zt)
            
    
    def _buildBundles(self):
        
        self.bundles = []
        self._name_to_mtl = {}
        self._name_to_bundle = {}

        for bundle_description in self._bundles:
            bundle_levels = {}
            mtls_in_bundle = []
            
            graph = self._buildGraphFromConnections(bundle_description["nestedElemIds"])
            
            for i in range(max([v["level"] for v in graph.vs])+1):
                bundle_levels[i] = []
                for name in list(set([v["cable"] for v in graph.vs(level_eq=i)])):
                    line = self._buildMTLfromLine(name)
                    
                    self._addConnectorsResistance(line)
                    
                    bundle_levels[i].append(line)     
                    self._name_to_mtl[name] = line
                    mtls_in_bundle.append(name)
                
                # bundle_levels[i] = [self._buildMTLfromLine(name) for name in list(set([v["cable"] for v in graph.vs(level_eq=i)]))]
                # for name in list(set([v["cable"] for v in graph.vs(level_eq=i)])):
                #     mtls_in_bundle.append(name)
                
            bundle = mtl.MTLD(bundle_levels, name = bundle_description["name"])
            for name in mtls_in_bundle:
                self._name_to_bundle[name] = bundle

            self._setBundleTransferImpedance(bundle, graph)

            self.bundles.append(bundle)

    def _getBundleFromElemId(self, elemId):
        cId = self._getNodeCoordId(elemId)
        elemId = self._getLineElemIdFromCoordId(cId)
        cable = self._getCableWithElemId(elemId)
        return self._getBundleWithCable(cable)
        

    def _addSources(self):
        for source in [s for s in self._sources if s["sourceType"] == "DistributedSource"]:
            magnitude = self._buildSource(source)
            
            cable = [cable for cable in self._cables if cable["name"] == source["cable"]][0]
            bundle = self._getBundleWithCable(cable)
            if (source["type"] == "localized efield drive"):
                # cStart = self._coordinates[self._getNodeCoordId(source["elemIds"][0])]
                # cEnd = self._coordinates[self._getNodeCoordId(source["elemIds"][1])] 
                bundle.add_localized_longitudinal_field(self._coordinates[self._getNodeCoordId(source["elemIds"][0])],
                                                        self._coordinates[self._getNodeCoordId(source["elemIds"][1])],
                                                        conductor = 0, 
                                                        magnitude = magnitude)
    
    def _getCableFromLineName(self, name):
        for cable in self._cables:
            if cable["name"] == name:
                return cable
            
        raise Exception("No cable with name "+name)

    def _getWireWithMaterialId(self, id):
        for w in self._wires:
            if w["materialId"] == id:
                return w
        raise Exception("No wire with materialId ", id)

    def _buildMTLfromLine(self, line):
        cable = self._getCableFromLineName(line)
        node_positions = self._getListOfCoordinates(cable["elemIds"][0])
        w = self._getWireWithMaterialId(cable["materialId"])
        L, C, R = 0.0,0.0,0.0
        for k, val in w.items():
            if "inductance" in k:
                L = val
            if "capacitance" in k:
                    C = val
            if "resistance" in k:
                    R = val

        if type(L) == list:
            L = np.array(L)
            C = np.array(C)
            R = np.array(R)*np.eye(L.shape[0])
            
        line = mtl.MTL(l = L, c = C, r = R, node_positions = node_positions, ndiv=cable["ndiv"], name = cable["name"])
        self._name_to_mtl[cable["name"]] = line
        return line

    def _getListOfCoordinates(self, elemId):
        node_positions = np.array([])
        for e in self._lines:
            if (int(e.split()[0]) == elemId):
                for cId in e.split()[3:]:
                    node_positions = np.append(node_positions, np.array(self._coordinates[int(cId)]))
                return node_positions.reshape(int(len(node_positions)/3),3)
        
    def _buildNetworks(self):
        self.networks = self._buildJunctions() + self._buildTerminations()
           
        
    def _getNetworkTerminationList(self):
        coordToNetwork = {}
        for t in self._terminations:
            coord = tuple(self._coordinates[t["coordIds"][0][0]])
            if coord in list(coordToNetwork.keys()):
                coordToNetwork[coord].append(t)
            else:
                coordToNetwork[coord] = [t]
        
        return list(coordToNetwork.values())
    
    def _getLevelOfCable(self, cable_name):
        for b in self.bundles:
            for level in b.levels:
                if cable_name in [line.name for line in b.levels[level]]:
                    return level
        raise Exception(cable_name + " is not part of any bundle")
    
    
    def _buildTerminations(self):
        
        terminations = []
        for level_network in self._getNetworkTerminationList():
            networkd = {}
            for t in level_network:
                connections = []
                description = {}
                for coords in t["coordIds"]:
                    d = {"nodes" : [], "connectors" : [], "conductors": [], "sources" : []}
                    for c in coords:

                        conn, cable_name, conductor, side  = self._getConnectorAndCableInNode(c)
                        source = self._getTerminalSourceInNode(c)

                        bundle_level = self._getLevelOfCable(cable_name)

                        if not self._name_to_mtl[cable_name] in description.keys():
                            description[self._name_to_mtl[cable_name]] = {"side":side, "bundle":self._name_to_bundle[cable_name], "connections" : []}
            
                        description[self._name_to_mtl[cable_name]]["connections"].append([c, conductor, conn])
                        d["nodes"].append(c)
                        d["connectors"].append(conn)
                        d["conductors"].append(conductor)
                        d["sources"].append(source)
                    connections.append(d)
                    
                network = nw.Network(description)
                
                for c in connections:
                    if len(c["nodes"]) == 1:
                        if (c["connectors"][0]["connectorType"] == "Conn_short"):
                            network.short_to_ground(c["nodes"][0])
                        elif (c["connectors"][0]["connectorType"] == "Conn_sRLC" or c["connectors"][0]["connectorType"] == "Conn_R"):
                            network.connect_to_ground(c["nodes"][0], 
                                                      c["connectors"][0]["resistance"], 
                                                      c["sources"][0])
                        elif (c["connectors"][0]["connectorType"] == "Conn_C"):
                            network.connect_to_ground_C(c["nodes"][0], 
                                                        c["connectors"][0]["capacitance"], 
                                                        c["sources"][0])
                        elif (c["connectors"][0]["connectorType"] == "Conn_LCpRs"):
                            network.connect_to_ground_LCpRs(c["nodes"][0], 
                                                            c["connectors"][0]["resistance"], 
                                                            c["connectors"][0]["inductance"], 
                                                            c["connectors"][0]["capacitance"], 
                                                            c["sources"][0])
                    
                    elif len(c["nodes"]) > 1:
                        for node, conductor in zip(c["nodes"], c["conductors"]):
                            if (c["connectors"][conductor]["connectorType"] == "MultiwireConnector"):
                                
                                if "resistanceVector" in c["connectors"][conductor].keys():
                                    R = c["connectors"][conductor]["resistanceVector"][conductor]
                                else:
                                    R = 0
                                if "capacitanceVector" in c["connectors"][conductor].keys():
                                    C = c["connectors"][conductor]["capacitanceVector"][conductor]
                                else:
                                    C = 1e22
                                if "inductanceVector" in c["connectors"][conductor].keys():
                                    L = c["connectors"][conductor]["inductanceVector"][conductor]
                                else:
                                    L = 0

                                if (c["connectors"][conductor]["connections"][conductor] == "Conn_short"):
                                    network.short_to_ground(node)
                                elif (c["connectors"][conductor]["connections"][conductor] == "Conn_R" or 
                                      c["connectors"][conductor]["connections"][conductor] == "Conn_sRLC"):
                                    network.connect_to_ground_R(node, R, c["sources"][conductor])
                                elif (c["connectors"][conductor]["connections"][conductor] == "Conn_LCpRs"):
                                    network.connect_to_ground_LCpRs(node, R ,L, C, c["sources"][conductor])
                                elif (c["connectors"][conductor]["connections"][conductor] == "Conn_C"):
                                    network.connect_to_ground_C(node, C, c["sources"][conductor])
                                # if R != 0:
                                #     network.connect_to_ground(node, R, c["sources"][conductor])
                                # else:
                                #     network.short_to_ground(node)

                networkd[bundle_level] = network
            terminations.append(nw.NetworkD(networkd))
        return terminations

    def _buildSource(self, source, k=1):
        if source["magnitude"]["type"] == "trapezoidal":
            def magnitude(t): return wf.trapezoidal_wave(
                t, 
                A=source["magnitude"]["amplitude"]*k, 
                rise_time=source["magnitude"]["rise_time"], 
                fall_time=source["magnitude"]["fall_time"], 
                f0=source["magnitude"]["f0"], 
                D=source["magnitude"]["D"]
            )
        elif source["magnitude"]["type"] == "ramp":
            def magnitude(t): return wf.ramp_pulse(
                t,
                A =source["magnitude"]["amplitude"]*k,
                x0=source["magnitude"]["t0"]
            )
        elif source["magnitude"]["type"] == "gaussian_2":
            def magnitude(t): return wf.gaussian_2(
                t,
                A = source["magnitude"]["amplitude"]*k,
                x0 = source["magnitude"]["x0"],
                s0 = source["magnitude"]["s0"]
            )
        elif source["magnitude"]["type"] == "sin_sq":
            def magnitude(t): return wf.sin_sq_pulse(
                t, 
                A = source["magnitude"]["amplitude"]*k,
                w = source["magnitude"]["frequency"]
            )
        else:
            def magnitude(t): lambda t : 0
            
        return magnitude

    def _getTerminalSourceInNode(self, node):
        for source in [s for s in self._sources if s["sourceType"] == "TerminalSource"]:
            for n in self._nodes:
                if int(n.split()[0]) == source["elemIds"][0] and int(n.split()[-1]) == node:
                    return self._buildSource(source)
        # return lambda t : 0
        return 0

    def _getNetworkJunctionList(self):
        coordToNetwork = {}
        for t in self._junctions:
            coord = tuple(self._coordinates[t["unitedCoordIds"][0][0]])
            if coord in list(coordToNetwork.keys()):
                coordToNetwork[coord].append(t)
            else:
                coordToNetwork[coord] = [t]
        
        return list(coordToNetwork.values())
            
            
    def _buildJunctions(self):
        
        junctions = []
        for level_network in self._getNetworkJunctionList():
            networkd = {}
            for j in level_network:
                connections = []
                description = {}
                for pair in j["unitedCoordIds"]:
                    d = {"nodes" : [], "connectors" : [], "conductors" : [], "sources" : []}
                    for p in pair:
                        conn, cable_name, conductor, side  = self._getConnectorAndCableInNode(p)
                        source = self._getTerminalSourceInNode(p)
                        
                        bundle_level = self._getLevelOfCable(cable_name)
                        
                        if not self._name_to_mtl[cable_name] in description.keys():
                            description[self._name_to_mtl[cable_name]] = {"side":side, "bundle":self._name_to_bundle[cable_name], "connections" : []}

                        if not [p,conductor, conn] in description[self._name_to_mtl[cable_name]]["connections"]:
                            description[self._name_to_mtl[cable_name]]["connections"].append([p, conductor, conn])

                        d["nodes"].append(p)
                        d["connectors"].append(conn)
                        d["conductors"].append(conductor)
                        d["sources"].append(source)
                    
                    connections.append(d)
                                    
                network = nw.Network(description)
                
                for conn in connections:
                    n0, n1 = conn["nodes"]
                    conn0, conn1 = conn["connectors"]
                    cond0, cond1 = conn["conductors"]
                    s0, s1 = conn["sources"]
                    R0, R1 = 0.0, 0.0
                    if (conn0["connectorType"] == "Conn_sRLC"):
                        R0 += conn0["resistance"]
                    elif conn0["connectorType"] == "MultiwireConnector":
                        R0 += conn0["resistanceVector"][cond0]

                    if (conn1["connectorType"] == "Conn_sRLC"):
                        R1 += conn1["resistance"]
                    elif conn1["connectorType"] == "MultiwireConnector":
                        R1 += conn1["resistanceVector"][cond1]
                    
                    if (R1+R0 == 0.0):
                        network.short_nodes(n0, n1)
                    else:
                        network.connect_nodes(n0, n1, R0 + R1, Vt = add(s0,s1))
                
                networkd[bundle_level] = network
            junctions.append(nw.NetworkD(networkd))            
        return junctions
        
    def _getConnectorAndCableInNode(self, node_id):
        el_id, side = self._getElementOfNode(node_id)
        conn_id, cable_name, conductor = self._getConnectorAndCableOnElementSide(el_id, side)
        try:
            conn = [c for c in self._connectors if c["materialId"] == conn_id][0]
        except:
            raise Exception('')
        return conn, cable_name, conductor, self._JSONside2MTLNside(side)
        
    def _JSONside2MTLNside(self, side):
        if side == "initial":
            return "S"
        elif side == "end":
            return "L"
        
    def _getElementOfNode(self,node_id):
        for e in self._lines:
            if (e.split()[3] == str(node_id)):
                return int(e.split()[0]), "initial"
            elif (e.split()[-1] == str(node_id)):
                return int(e.split()[0]), "end"

        raise Exception(str(node_id) + " is not part of any element")
    
    def _getConnectorAndCableOnElementSide(self, el_id, side):
        for c in self._cables:
            if el_id in c["elemIds"]:
                return c[side+"ConnectorId"], c["name"], c["elemIds"].index(el_id)
        
        raise Exception(str(el_id) + " does not belong to any cable")    
        
    
    def getNodesInJunction(self, junctionName):
        for junction in self._junctions:
            if junction["name"] == junctionName:
                return junction["unitedCoordIds"]
        
        raise Exception("No junctions named "+ junctionName)
    
    def _getNodeCoordId(self, elemId):
        for node in self._nodes:
            if int(node.split()[0]) == elemId:
                return int(node.split()[-1])
        
        raise Exception("No node with elemId "+ elemId)
        
    def _getLineElemIdFromCoordId(self, coordId):
        for line in self._lines:
            if str(coordId) in line.split()[3:]:
                return int(line.split()[0])

        raise Exception("No line with coordId "+ coordId)
    
    def _getCableWithElemId(self, elemId):
        for cable in self._cables:
            if elemId in cable["elemIds"]:
                return cable
        
        raise Exception("No cable with elemId "+ elemId)
    
    def _getBundleWithCable(self, cable):
        for bundle in self.bundles:
            for mtls in bundle.levels.values():
                for mtl in mtls:
                    if mtl.name == cable["name"]:
                        return bundle
        
        raise Exception(cable.name + " does not belong to any bundle")
                
    def _addProbes(self):
        for probe in self._probes:
            if "cable" in probe.keys():
                bundle = self._getBundleWithCable([cable for cable in self._cables if cable["name"] == probe["cable"]][0])
            else:
                bundle = self._getBundleFromElemId(probe["elemIds"][0])
            out_probe = bundle.add_probe(self._coordinates[self._getNodeCoordId(probe["elemIds"][0])],probe["type"])
            self.probes[probe["name"]] = out_probe
    
