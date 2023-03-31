import numpy as np

class Probe:
    def __init__(self, position, conductor = 0):
        self.type = "voltage"
        self.position = position
        self.conductor = conductor
        self.t = np.array([])
        self.v = np.array([])
        return

class MTL:
    """
    Lossless Multiconductor Transmission Line
    """

def __init__(self, L, C, length=1.0, nx=100, Zs=0.0, Zl=0.0):
    self.x = np.linspace(0, length, nx+1)
    self.l = L
    self.c = C
    
    self.number_of_conductors = np.shape(self.l)[0]
    self.v = np.zeros([self.number_of_conductors, self.x.shape[0]  ])
    self.i = np.zeros([self.number_of_conductors, self.x.shape[0]-1])

    self.zs = Zs * np.eye(self.number_of_conductors)
    self.zl = Zl * np.eye(self.number_of_conductors)

    self.timestep = get_max_timestep()
    self.time = 0.0
    
    dx = self.x[1] - self.x[0]

    self.i_diff = self.timestep / dx * np.linalg.inv(self.c)
    self.v_diff = self.timestep / dx * np.linalg.inv(self.l)

    left_port_c = np.matmul(self.zs,self.c)
    right_port_c = np.matmul(self.zl,self.c)

    self.left_port_term = np.matmul( \
            np.linalg.inv(dx*left_port_c/self.timestep+np.eye(self.number_of_conductors)), \
            dx*left_port_c/self.timestep-np.eye(self.number_of_conductors))
    
    self.right_port_term = np.matmul( \
            np.linalg.inv(dx*right_port_c/self.timestep+np.eye(self.number_of_conductors)), \
            dx*right_port_c/self.timestep-np.eye(self.number_of_conductors))

def get_phase_velocities(self):
    return 1/np.sqrt(np.diag(self.l)*np.diag(self.c))

def get_max_timestep(self):
    dx = self.x[1] - self.x[0]
    return dx / np.max(self.get_phase_velocities())

def set_voltage(self, voltage):
    self.v = voltage(self.x)

def update_probes(self):
    for probe in self.probes:
        if probe.type == "voltage":
            probe.t = np.vstack(probe.t, self.time)
            index = self.x[self.x >= probe.position and self.x < probe.position]
            probe.v = np.vstack(probe.v, self.v[probe.conductor,index])
        else:
            raise ValueError("undefined probe")


def step(self):

    self.v[:,0] = self.left_port_term.dot(self.v[:,0]) - self.left_port_term.dot(2*np.matmul(self.zs,self.i[:,0]))
    self.v[:,1:-1] -= self.i_diff.dot(self.i[:,1:]-self.i[:,:-1])
    self.v[:,-1] = self.right_port_term.dot(self.v[:,-1]) + self.right_port_term.dot(2*np.matmul(self.zl,self.i[:,-1]))

    self.i[:,:] -= self.v_diff.dot(self.v[:,1:]-self.v[:,:-1])

    self.time += self.dt
    self.update_probes()


def add_voltage_source(self, position: float, conductor: int, magnitude):
    return

def add_voltage_probe(self, position: float):
    return

