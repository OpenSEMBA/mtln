import numpy as np

class Probe:
    def __init__(self, position, conductor, type):
        self.type = type
        self.position = position
        self.conductor = conductor
        self.t = np.array([])
        self.v = np.array([])
        self.i = np.array([])
        return

class MTL:
    """
    Lossless Multiconductor Transmission Line
    """

    def __init__(self, l, c, length=1.0, nx=100, Zs=0.0, Zl=0.0):
        self.x = np.linspace(0, length, nx+1)


        if type(l) == float and type(c) == float:
            self.l = np.array([[l]])
            self.c = np.array([[c]])
        elif type(l) == np.ndarray and type(c) == np.ndarray:
            assert(l.shape == c.shape)
            assert(l.shape[0] == l.shape[1])
            self.l = l
            self.c = c
        else:
            raise ValueError("Invalid input L and/or C")
        
        self.number_of_conductors = np.shape(self.l)[0]
        
        if (type(Zs) != np.ndarray and type(Zl) != np.ndarray):
            self.zs = Zs * np.eye(self.number_of_conductors)
            self.zl = Zl * np.eye(self.number_of_conductors)
        elif (type(Zs) == np.ndarray and type(Zl) == np.ndarray):
            assert(Zs.shape == Zl.shape)
            if (len(Zs.shape) == 1):
                assert(Zs.shape[0] == self.number_of_conductors)
                self.zs = Zs.reshape(self.number_of_conductors,1) * np.eye(self.number_of_conductors)
                self.zl = Zl.reshape(self.number_of_conductors,1) * np.eye(self.number_of_conductors)
                
            elif (len(Zs.shape) == 2):
                assert(Zs.shape[0] == 1 or Zs.shape[1] == 1)
                assert(Zs.shape[0] == self.number_of_conductors or Zs.shape[1] == self.number_of_conductors)
                self.zs = Zs * np.eye(self.number_of_conductors)
                self.zl = Zl * np.eye(self.number_of_conductors)
        else:
            raise ValueError("Invalid input Zs and/or Zc. Use two floats or two arrays")
               
              
            
        
        self.v = np.zeros([self.number_of_conductors, self.x.shape[0]  ])
        self.i = np.zeros([self.number_of_conductors, self.x.shape[0]-1])

        self.probes = []
        self.v_sources = np.empty(shape=(self.number_of_conductors, self.x.shape[0]), dtype=object)
        self.v_sources.fill(lambda n : 0)
        
        self.timestep = self.get_max_timestep()
        self.time = 0.0
        
        dx = self.x[1] - self.x[0]

        self.i_diff = self.timestep / dx * np.linalg.inv(self.c)
        self.v_diff = self.timestep / dx * np.linalg.inv(self.l)

        left_port_c = np.matmul(self.zs,self.c)
        right_port_c = np.matmul(self.zl,self.c)

        self.left_port_term_1 = np.matmul( \
                np.linalg.inv(dx*left_port_c/self.timestep+np.eye(self.number_of_conductors)), \
                dx*left_port_c/self.timestep-np.eye(self.number_of_conductors))

        self.left_port_term_2 = np.linalg.inv(dx*left_port_c/self.timestep+np.eye(self.number_of_conductors))
        
        self.right_port_term_1 = np.matmul( \
                np.linalg.inv(dx*right_port_c/self.timestep+np.eye(self.number_of_conductors)), \
                dx*right_port_c/self.timestep-np.eye(self.number_of_conductors))

        self.right_port_term_2 = np.linalg.inv(dx*right_port_c/self.timestep+np.eye(self.number_of_conductors))


    def get_phase_velocities(self):
        # return 1/np.sqrt(np.diag(self.l)*np.diag(self.c))
        return 1/np.sqrt(np.diag(self.l.dot(self.c)))
    def get_max_timestep(self):
        dx = self.x[1] - self.x[0]
        return dx / np.max(self.get_phase_velocities())

    def set_voltage(self, voltage):
        self.v = voltage(self.x)

    def update_probes(self):
        for probe in self.probes:
            if probe.type == "voltage":
                probe.t = np.append(probe.t, self.time)
                index = np.argmin(np.abs(self.x - probe.position))
                probe.v = np.append(probe.v, self.v[probe.conductor,index])
            elif probe.type == "current":
                probe.t = np.append(probe.t, self.time)
                index = np.argmin(np.abs(self.x - probe.position))
                probe.i = np.append(probe.i, self.i[probe.conductor,index-1])
            else:
                raise ValueError("undefined probe")

    def eval_v_sources(self, time):
        
        v_sources_time = np.empty(shape=(self.number_of_conductors, self.x.shape[0]))
        for n in range(self.number_of_conductors):
            for pos in range(self.x.shape[0]):
                v_sources_time[n,pos] = self.v_sources[n,pos](time)
        return v_sources_time
        

    def step(self):

        v_sources_curr = self.eval_v_sources(self.time)
        v_sources_prev = self.eval_v_sources(self.time - self.timestep)
        
        self.v[:,0] = self.left_port_term_1.dot(self.v[:,0]) + \
                        self.left_port_term_2.dot(-2*np.matmul(self.zs,self.i[:,0])+(v_sources_curr[:,0] + v_sources_prev[:,0]))
        
        self.v[:,1:-1] -= self.i_diff.dot(self.i[:,1:]-self.i[:,:-1])
        
        self.v[:,-1] = self.right_port_term_1.dot(self.v[:,-1]) + \
                        self.right_port_term_2.dot(+2*np.matmul(self.zl,self.i[:,-1])+(v_sources_curr[:,-1] + v_sources_prev[:,-1]))

        self.i[:,:] -= self.v_diff.dot(self.v[:,1:]-self.v[:,:-1])

        self.time += self.timestep
        self.update_probes()


    def add_voltage_source(self, position: float, conductor: int, magnitude):
        index = np.argmin(np.abs(self.x - position))
        self.v_sources[conductor, index] = magnitude
        return

    def add_probe(self, position: float, conductor: int, type: str):
        
        probe = Probe(position, conductor, type)
        self.probes.append(probe)        
        return probe

