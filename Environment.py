import numpy as np
class SpringMass(object):
    """The simulation environment for SpringMass"""
    def __init__(self, m = 0.1,K_stab = 0.1,l = 1,w_max = 0.3,w_adv_fraction=0.999,dt=0.1,space_dim=1,x_0=None):

        self.m  = m # Mass
        self.K_stab = K_stab # Universal stabilizer(stiffness of the spring)
        self.l = l # Velocity drag

        self.w_max = w_max # Maximum disturbance force

        self.dt = dt # Sampling time interval

        self.space_dim = space_dim # Spatial dimension
        
        O = np.zeros((space_dim,space_dim))

        I = np.eye(space_dim)

        self.A = np.vstack([np.hstack([O,I]),
                      np.hstack([-K_stab/m * I,-l/m * I])])*self.dt+np.eye(space_dim*2)

        self.B = np.vstack([0,
                      1/m ])*self.dt

        if x_0 is None:
            p_0 = np.zeros(space_dim)
            v_0 = np.zeros(space_dim)

            self.x_0 = np.vstack([p_0,v_0])
        else:
            self.x_0 = x_0

        self.x = np.array(self.x_0)

        self.w = 0.0

        self.w_change_period = 1 # The expected time steps till the next change of w.

        self.w_adv_fraction = w_adv_fraction # Our w = w_adv + w_random. w_adv_fraction = |w_adv|/|w|.

        self.N_steps = 0

    def reset(self):
        self.x = np.array(self.x_0)
    
    def step(self,F):
        
        if self.N_steps % self.w_change_period==0: # Determine whether w will change in this step.
            # self.w = self.w_max * (self.w_adv_fraction + 2*(np.random.rand()-0.5)*(1-self.w_adv_fraction))# Change w to be the adversarial noise + random noise.
            # self.w = self.w_max * (self.w_adv_fraction + 2*(np.random.randint(2)-0.5)*(1-self.w_adv_fraction))# Change w to be the adversarial noise + random noise.
            self.w = self.w_max * (self.w_adv_fraction + np.random.choice([0,1,-1],p=[0.5,0.25,0.25])*(1-self.w_adv_fraction))# Change w to be the adversarial noise + random noise.
                
        
        w_vec = np.zeros(self.x.shape)

        w_vec[-1,-1] = self.w
        # print(self.w)
        self.x = self.A.dot(self.x)+self.B.dot(F) + w_vec 

        # Reset w to 0
        self.w=0
        self.N_steps+=1

    def state(self):
        return np.array(self.x)


class Quadrotor1D(object):
    """The simulation environment for Quadroto1D with a stabilizing controller."""
    def __init__(self, m = 0.1,K_stab = np.array([[2/3,1]]),l = 1,w_max = 0.3,dt=0.1,space_dim=1,x_0=None):

        self.m  = m # Mass
        self.K_stab = K_stab # Universal stabilizer
        self.l = l # Velocity drag

        self.w_max = w_max # Maximum disturbance force

        self.dt = dt # Sampling time interval

        self.space_dim = space_dim # Spatial dimension
        
        O = np.zeros((space_dim,space_dim))

        I = np.eye(space_dim)

        self.A = np.vstack([np.hstack([O,I]),
                      np.hstack([O,-l/m * I])])*self.dt+np.eye(space_dim*2)

        self.B = np.vstack([0,
                      1/m ])*self.dt
        
        self.AK = self.A-self.B.dot(K_stab)

        if x_0 is None:
            p_0 = np.zeros(space_dim)
            v_0 = np.zeros(space_dim)

            self.x_0 = np.vstack([p_0,v_0])
        else:
            self.x_0 = x_0

        self.x = np.array(self.x_0)
        
        self.w_change_period = 1 # The expected time steps till the next change of w.

        self.N_steps = 0

    def reset(self):
        self.x = np.array(self.x_0)
    
    def step(self,F,b=0):
        
        # b: The offset in base controller. Could be time varying. 
        # The base controller is Kx+b
        # The system is x_{t+1} = Ax[t]+B(-Kx[t]-b[t]) + BF[t] + w
        
        
        if self.N_steps % self.w_change_period==0: # Determine whether w will change in this step.
            self.w = self.w_max * np.random.choice([1,-1])# Change w to be the adversarial noise + random noise.
                
        
        w_vec = np.zeros(self.x.shape)

        w_vec[-1,-1] = self.w
        # print(self.w)

        self.x = self.AK.dot(self.x)+self.B.dot(-b+F) + w_vec 

        # Reset w to 0
        self.w=0
        self.N_steps+=1

    def state(self):
        return np.array(self.x)