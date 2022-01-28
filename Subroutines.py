import numpy as np
import cvxpy as cp

def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))

def find_stable_radius(A0,K=10000,M=10,N=100,delta=0.03):
    '''
        e: the largest radius e such that the random samples from B(A,e) contains stable matrices only.
        Setting K~=10000 is usually more than good enough an approximation.
    '''

    def sup_lambda_max(e):
        random_dir = np.random.rand(*tuple(list(A0.shape)+[K]))
        random_dir = np.random.rand(*tuple(list(A0.shape)+[K]))
        random_dir/=np.linalg.norm(random_dir,axis=(0,1))
        random_dir = random_dir.T
        A_cand = e*random_dir+A0
        return np.max(np.abs(np.linalg.eigvals(A_cand)))

    for i in range(M): # Coarse-grained search
        e = 10**(-i)
        if sup_lambda_max(e)<1: # Fine-grained search
            e_fine = np.linspace(10**(-i),10**(-i+1),N)
            sup_l = np.array([sup_lambda_max(ef) for ef in e_fine])
            i_max = np.sum(sup_l<1-delta)-1
            # print(sup_l[i_max])
            return e_fine[i_max]

    return None

def ModelEstV1(x_hist,u_hist,eta_bar=1,stability_gap = 0.1):
    '''
        stability_gap: how far the largest eigenvalue of A_hat should at least be below 1.
    '''
    if len(x_hist.shape)>1:
        x_dim  = x_hist.shape[-1]
    else:
        x_dim = 1
    if len(u_hist.shape)>1:
        u_dim = u_hist.shape[-1]
    else:
        u_dim = 1

    # Estimate 
    A_hat = cp.Variable((x_dim,x_dim))
    B_hat = cp.Variable((x_dim,u_dim))

    x_t = x_hist[:-1,:]
    x_t_1 = x_hist[1:,:]

    objective = cp.Minimize(cp.sum_squares(x_t_1.T - A_hat @ x_t.T - B_hat @ u_hist.T ))

    prob = cp.Problem(objective)

    prob.solve()
    
    r = np.sqrt(x_dim**2 + x_dim*u_dim)/(np.sqrt(len(x_t))*eta_bar)

    if np.max(np.abs(np.linalg.eigvals(A_hat.value)))<= 1-stability_gap:
        return A_hat.value,B_hat.value,r
    else:
        # Project A_hat onto the interior of the set of open-loop stable matrices.
        A_proj = cp.Variable(A_hat.shape)

        stability_constraint = [cp.lambda_max(A_proj)<= 1-stability_gap] # Over kill the stability gap here, so that we have more room in the subsequent binary line search.

        # Keep in mind that if we use the lambda_max function, then A_proj will be casted to a symmetric matrix. But the true A may not be symmetric.
       
        prob2 = cp.Problem(cp.Minimize(cp.norm2(A_proj-A_hat.value)),stability_constraint)
        prob2.solve()

        # Random line search to find a closest stable matrix to A_hat, along the A_hat-A_proj line.
        A_origin = A_hat.value
        A_target = A_proj.value

     
        l = list(np.random.rand(100))+[1]
        diff = A_target-A_origin
        A_cand = A_origin + np.array([l*diff for l in l])
        spectral_radius = np.array([np.max(np.abs(np.linalg.eigvals(A))) for A in A_cand])
        dist = np.linalg.norm(A_cand-A_origin,axis=(-1,-2))

        dist[spectral_radius>1-stability_gap+0.001]=np.inf

        A_final = A_cand[np.argmin(dist)]

        # print(np.max(np.abs(np.linalg.eigvals(A_final))),
            # np.max(np.abs(np.linalg.eigvals(A_target))))
        
        return A_final,B_hat.value,r

def ApproxDAP(M,w_hist,eta_bar):

    l = np.min([len(M),len(w_hist)])
    
    # return np.sum([M[k].dot(w_hist[-k-1]) for k in range(l)])+(np.random.rand()-0.5)*2*eta_bar
    return np.sum([M[k].dot(w_hist[-k-1]) for k in range(l)])+(np.random.choice([-1,1]))*eta_bar



def LP(c,D,d):
    x_dim = D.shape[-1]
    # print(D)

    x = cp.Variable((x_dim,1))

    constraint = [D @ x<= d]

    objective = cp.Maximize(c.T @ x)

    prob = cp.Problem(objective, constraint)

    return prob.solve()


def max_norm(D,d):
    '''
        Heuristically find the maximum norm of vectors x within the polytope Dx<=d.
    '''
    dists = []
    
    x_dim = D.shape[-1]
    for _ in range(30):
        
        c = np.random.randn(x_dim).reshape(x_dim,1)
        c/=np.linalg.norm(c)

        dir_dist = np.max([LP(c,D,d),LP(-c,D,d)])
        dists.append(dir_dist)
    
    return np.max(dists)


# Deprecated. See SafeDAP.mid() for the cleaner implementation of mid calculation.
# def M_mid(A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max,A_prev,B_prev,r_prev,eta_bar_prev,x_prev,u_prev,M_prev,M_dest):
#     if len(x_hist.shape)>1:
#         x_dim  = x_hist.shape[-1]
#     else:
#         x_dim = 1

#     if len(u_hist.shape)>1:
#         u_dim = u_hist.shape[-1]
#     else:
#         u_dim = 1

#     M = [cp.Variable((u_dim,x_dim)) for _ in range(H)]


#     omega1,_ = OMEGA(M,A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max)
#     omega2,_ = OMEGA(M,A_prev,B_prev,r_prev,H,eta_bar_prev,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_prev,u_prev,w_max)

#     omega_intersection = omega1+omega2

#     prob = cp.Problem(cp.Minimize(cp.sum([cp.norm(M_prev[k]-M[k],2)**2+cp.norm(M_dest[k]-M[k],2)**2 for k in range(H)])),omega_intersection)

#     prob.solve()
#     return [m.value for m in M]

class ModelEst:

    def project(self,A_hat):
        raise NotImplementedError

  



class ModelEstV2:
    def __init__(self,A0,B0,eps_init):
        self.A0 = A0
        self.B0 = B0
        self.eps_init = eps_init
    
    @classmethod
    def ball_project(cls,A_hat,A0,eps_init):
        A_proj = cp.Variable(A_hat.shape)
        constraint = [ cp.norm(A_proj-A0,'fro') <= eps_init]
        prob2 = cp.Problem(cp.Minimize(cp.norm(A_proj-A_hat,'fro')),constraint)
        prob2.solve()
        return A_proj.value

    def project(self,A_hat,B_hat):

        A_proj = self.ball_project(A_hat,self.A0,self.eps_init)
        B_proj = self.ball_project(B_hat,self.B0,self.eps_init)
 
        return A_proj,B_proj

    def est(self,x_hist,u_hist,eta_bar=1):

        # L2 Estimation
        if len(x_hist.shape)>1:
            x_dim  = x_hist.shape[-1]
        else:
            x_dim = 1
        if len(u_hist.shape)>1:
            u_dim = u_hist.shape[-1]
        else:
            u_dim = 1

        A_hat = cp.Variable((x_dim,x_dim))
        B_hat = cp.Variable((x_dim,u_dim))

        x_t = x_hist[:-1,:]
        x_t_1 = x_hist[1:,:]

        objective = cp.Minimize(cp.sum_squares(x_t_1.T - A_hat @ x_t.T - B_hat @ u_hist.T ))

        prob = cp.Problem(objective)

        prob.solve()

        r = np.sqrt(x_dim**2 + x_dim*u_dim)/(np.sqrt(len(x_t))*eta_bar)


        A_proj,B_proj = self.project(A_hat.value,B_hat.value)

        return A_proj, B_proj, r    

class ModelEstV3(ModelEstV2):
    '''
        Use the system knowledge to project A_hat, B_hat.
    '''
    def __init__(self):
        pass
    
    def project(self,A_hat,B_hat):

        # print(A_hat,B_hat)
        I = np.eye(2)

        a,b,c,d = [cp.Variable(nonneg=True) for _ in range(4)]

        A_proj = I+cp.vstack([cp.hstack([0,a]),cp.hstack([-b,-c])])

        B_proj = cp.vstack([0,d])

        objective = cp.norm(A_proj-A_hat)**2+cp.norm(B_proj-B_hat)**2

        prob = cp.Problem(cp.Minimize(objective))

        prob.solve()

        return A_proj.value,B_proj.value

class ModelEstV4(ModelEstV2):
    '''
        Only use the system knowledge to project A_hat, and leave B_hat as it is.
        This creates some difficulty in learning B_hat.
    '''
    def __init__(self):
        pass
    
    def project(self,A_hat,B_hat):

        # print(A_hat,B_hat)
        I = np.eye(2)

        a,b,c= [cp.Variable(nonneg=True) for _ in range(3)]

        A_proj = I+cp.vstack([cp.hstack([0,a]),cp.hstack([-b,-c])])

        # B_proj = cp.vstack([0,d])

        objective = cp.norm(A_proj-A_hat)**2

        prob = cp.Problem(cp.Minimize(objective))

        prob.solve()

        return A_proj.value,B_hat

class QuadrotorEst:
    def __init__(self,K_stab,dt,alpha_limit,beta_limit):
        '''
            alpha_limit:(alpha_min,alpha_max)
            beta_limit: similar to alpha_limit.
        '''
        self.K_stab = K_stab
        self.dt = dt
        self.alpha_limit = alpha_limit
        self.beta_limit = beta_limit
    
    @classmethod
    def interval_project(cls,a,a_lim):
        
        if a>np.max(a_lim):
            a = np.max(a_lim)
        elif a<np.min(a_lim):
            a = np.min(a_lim)
        
        return a

    def project(self,alpha,beta):
 
        return self.interval_project(alpha,self.alpha_limit),self.interval_project(beta,self.beta_limit)

    def est(self,x_hist,u_hist,eta_bar=1):

        # L2 Estimation
        if len(x_hist.shape)>1:
            x_dim  = x_hist.shape[-1]
        else:
            x_dim = 1
        if len(u_hist.shape)>1:
            u_dim = u_hist.shape[-1]
        else:
            u_dim = 1

        alpha = cp.Variable(nonneg = True )
        beta = cp.Variable(nonneg = True)

        x_t = x_hist[:-1,:]
        x_t_1 = x_hist[1:,:]
        
        A_hat = cp.vstack([cp.hstack([1,self.dt])\
                          ,cp.hstack([0,1-beta])])
        B_hat = cp.vstack([0,alpha])

        objective = cp.Minimize(cp.sum_squares(x_t_1.T - ((A_hat-B_hat @ self.K_stab) @ x_t.T + B_hat @ u_hist.T )))

        prob = cp.Problem(objective)

        v = prob.solve()

        if np.isinf(v): # If inf value is encountered, it usually is due to singularity in problem data. Run more pre-steps to resolve such issue.
            print(alpha.value,beta.value,self.K_stab,x_hist,u_hist)


        r = np.sqrt(x_dim**2 + x_dim*u_dim)/(np.sqrt(len(x_t))*eta_bar)

        # print('alpha',alpha.value,'beta',beta.value)            
   
        alpha_proj,beta_proj = self.project(alpha.value,beta.value)
        # print('alpha_proj',alpha_proj,'beta_proj',beta_proj)            
        A_proj = np.array([[1,self.dt],[0,1-beta_proj]])
        B_proj = np.array([[0],[alpha_proj]])

        return A_proj, B_proj, r    

class SafeTransit:
    '''
        This class handles the slow-varying transit form old controller M to new M.
    '''
    def __init__(self,old,new,mid,H,W1=None,W2=None):
        '''
            old, new are dictionaries with at least the following items:
            {'M':np.array, 'theta':tuple,'eta':explorative noise level(positive)}
        '''
        self.old = old
        self.new = new
        self.M = np.array(self.old['M'])
        if any([m is None for m in mid]):
            self.mid = np.array(self.new['M'])
        else:
            self.mid = mid
         
        self.DM = 0.01 # The default variation budget. 
        
        if W1 is None:
            # print(self.mid)
            self.W1 = int(np.ceil(np.max([np.linalg.norm(self.old['M']-self.mid)/self.DM,H])))
        else:
            self.W1 = W1 # The number of self.step() calls required to transit from old['M'] to mid.
        
        if W2 is None:
            self.W2 = int(np.ceil(np.linalg.norm(self.new['M']-self.mid)/self.DM))
        else:
            self.W2 = W2 # The number of self.step() calls to transit from mid to new['M'].
        
        self.step_count = 0 # The counter of self.step() calls

        self.eta_min = np.min([self.old['eta'],self.new['eta']])

        if self.old['r']>self.new['r']:
            self.theta_min = self.new['theta']
        else:
            self.theta_min = self.old['theta']

    def get_theta(self):
        if self.step_count<self.W1:
            # print('theta_min',self.theta_min)
            return self.theta_min
        else:
            # print('new theta',self.new['theta'])
            return self.new['theta']

    def update_W(self,M_origin,M_target,W): 
        self.M += (M_target-M_origin)/W
        
    def step(self):
        
        # print('W1,W2:',self.W1,self.W2)

        # Check the transit phase.
        if self.step_count<self.W1:
            
            self.update_W(self.old['M'], self.mid, self.W1)
            # print('Step:',self.step_count,'before mid',np.linalg.norm(self.M-self.mid))
            
        elif self.step_count<self.W1+self.W2:
            
            self.update_W(self.mid, self.new['M'],self.W2)
            # print('Step:',self.step_count,'after mid',np.linalg.norm(self.M-self.new['M']))
        
        # If step count has gone above W1+W2, transit no more.
        
        self.step_count +=1
        return self.M

    def get_u(self,w_hat_hist):
        if self.step_count<self.W1:
            return ApproxDAP(self.M, w_hat_hist, self.eta_min)
        else:
            return ApproxDAP(self.M, w_hat_hist, self.new['eta'])
       
