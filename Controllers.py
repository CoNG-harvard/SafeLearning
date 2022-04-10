import numpy as np
import cvxpy as cp
from Environment import SpringMass
from scipy.linalg import sqrtm
from Subroutines import max_norm

class SafeDAP:
    def __init__(self,Q,R,D_x,d_x,D_u,d_u,w_max,w_cov):
        self.Q = Q
        self.R = R
        self.D_x = D_x
        self.d_x = d_x
        self.D_u = D_u
        self.d_u = d_u
        self.w_max = w_max
        self.w_cov = w_cov # The covariance matrix of disturbance w.
        
        self.u_dim = D_u.shape[-1]
        self.x_dim = D_x.shape[-1]   
        
        self.M = []
        
    def get_tightening_coefs(self,A,B,H):
        

        # Compute the constraint tightening constants due to optimization on the approximate state.
        D_inf = np.linalg.norm(self.D_x,np.inf)
        # A_radius 
        kappa = 1.5 # A choosen large constant.
        gamma = 1 - np.max(np.abs(np.linalg.eigvals(A))) # This typically can ensure A is (kappa, gamma) stable.

        # Estimate x_max and u_max
        x_max = max_norm(self.D_x,self.d_x.reshape(-1,1))
        u_max = max_norm(self.D_u,self.d_u.reshape(-1,1))

        z_max = np.sqrt(x_max**2+u_max**2)

        Delta_m = 0.5 # policy variation budget

        r = 0.4 # Confidence radius of the estimated model parameters. The exact formula is quite complicated, so I just set an arbitrary value here.

        eta_bar = 0.4 # Upper bound on the magnitude of exploration noise.

        kappa_B= np.linalg.norm(B,ord=2)

        c1 = D_inf*z_max*kappa/gamma+5*kappa**4 * kappa_B * D_inf * self.w_max /gamma**3
        c2 = D_inf*kappa*kappa_B/gamma
        c3 = np.linalg.norm(self.D_u,np.inf)

        e_theta = c1*np.sqrt(self.x_dim*self.u_dim)*r
        e_eta_x = c2*np.sqrt(self.u_dim)*eta_bar
        e_eta_u = c3*eta_bar
        e_v = D_inf*self.w_max*kappa*kappa_B*gamma**2 *np.sqrt(self.u_dim*self.x_dim*H)*Delta_m 
        e_H = D_inf*kappa*x_max*(1-gamma)**H

        e_x = e_theta+e_eta_x+e_H+e_v
        e_u = e_eta_u
        
        print('e_x',e_x,'e_u',e_u)
        print('e_H',e_H,'e_theta',e_theta,'e_eta_x',e_eta_x,e_v,'e_v',e_v)
        print('gamma',gamma,'kappa_B',kappa_B,'z_max',z_max,'kappa',kappa)
        return e_x,e_u

    def omega_phi(self,M,A,B,e_x,e_u,H,K_stab=None,b=None):
            
            if K_stab is None:
                K_stab = 0.0
        
            if b is None:
                b = 0.0

            # Note, if K_stab, b are not None, A should be AK.
            AK = A-B.dot(K_stab)


            # AK_pow[n] = AK^n, A_pow[0:H]
            AK_pow = [np.eye(self.x_dim)]
            for i in range(H):
                AK_pow.append(AK_pow[-1].dot(AK))
            
            # Express Transition Kernel Phi
            Phi = []

            for k in range(2*H):
                if k<H:
                    Phi_k = AK_pow[k]
                else:
                    Phi_k = 0

                for i in range(H):
                    if 0<= k-i-1 and k-i-1<H:
                        Phi_k += AK_pow[i].dot(B) @ M[k-i-1]
                Phi.append(Phi_k)

            # print(type(b),b)
            b_offset = 0.0
            if type(b) in [np.ndarray,np.float64,float,int]:
                b_offset = self.D_x.dot(np.linalg.inv(AK-np.eye(self.x_dim)).dot(B).dot(b))
            elif type(b) is cp.Variable:
                b_offset = self.D_x.dot(np.linalg.inv(AK-np.eye(self.x_dim))).dot(B) @ b

            x_con = []
            for i in range(len(self.D_x)):
                gi = cp.sum([cp.norm(self.D_x[i,:] @ Phi[k],1) for k in range(2*H)]) * self.w_max
                x_con.append(gi<=self.d_x[i]-e_x-b_offset[i])
            # print('Sum of (A_pow norm1)',np.sum([np.linalg.norm(A,1) for A in A_pow]))

            u_con = []
            # for j in range(len(self.D_u)):

            #     gj = cp.sum([cp.norm(self.D_u[j,:] @ (M[k]-K_stab @ Phi[k]),1) for k in range(H)]+[cp.norm(self.D_u[j,:] @ (-K_stab @ Phi[k]),1) for k in range(H,2*H)]) * self.w_max
            #     u_con.append(gj<=self.d_u[j]-e_u)

            OMEGA = x_con + u_con
            
            return OMEGA,Phi

    def mid(self,old,new,H,K_stab):
        '''
            Find the middle-point policy that lies within the intersection of old Omega and new Omega.
            We specify mid to be the one with the smallest sum of Frobenius distance squares to M_old and M_new.
        '''
        
        mid = [cp.Variable(m.shape) for m in old['M']]

        O,_ = self.omega_phi(mid,old['theta'][0],old['theta'][1],old['e_x'],old['e_u'],H,K_stab)
        O_new,_ = self.omega_phi(mid,new['theta'][0],new['theta'][1],new['e_x'],new['e_u'],H,K_stab)


        constraints = O + O_new

        dist = cp.sum([cp.norm(mid[i]-old['M'][i],'fro')**2+cp.norm(mid[i]-new['M'][i],'fro')**2 for i in range(len(old['M']))])

        prob = cp.Problem(cp.Minimize(dist),constraints)

        prob.solve()

        return np.array([m.value for m in mid])

    def solve(self,A,B,H,e_x=None,e_u=None,unconstrained=False,K_stab=None,b=None):
        
        if e_x is None or e_u is None:
            e_x,e_u = self.get_tightening_coefs(A,B,H)


        # The policy M. cvxpy does not support tensor variable with # axes>2. We need to use a list here.
        M = [cp.Variable((self.u_dim,self.x_dim)) for _ in range(H)]
        
        R_sqrt = sqrtm(self.R)
        Q_sqrt = sqrtm(self.Q)
        w_sqrt = sqrtm(self.w_cov)
        
        OMEGA,Phi = self.omega_phi(M,A,B,e_x,e_u,H,K_stab,b)

        # The R loss
        if K_stab is None: # Not includng K_stab
            R_loss = cp.sum([cp.norm(R_sqrt @ M[k] @ w_sqrt,'fro')**2 for k in range(H)])
        else: # Including K_stab
            R_loss = cp.sum(
                 # [cp.norm(R_sqrt @ M[0] @ w_sqrt,'fro')**2 ]\
                [cp.norm(R_sqrt @ (M[k]-K_stab @ Phi[k]) @ w_sqrt,'fro')**2 for k in range(H)]\
                +[cp.norm(R_sqrt @ (K_stab @ Phi[k]) @ w_sqrt,'fro')**2 for k in range(H,2*H)])

        # The Q loss
        Q_loss = cp.sum([cp.norm(Q_sqrt @ Phi[k] @ w_sqrt,'fro')**2 for k in range(2*H)])

        # Solve the problem
        if unconstrained:
            prob = cp.Problem(cp.Minimize(R_loss+Q_loss))
        else:
            prob = cp.Problem(cp.Minimize(R_loss+Q_loss),OMEGA)
        
        # prob.solve(verbose=True)
        prob.solve(verbose=False)
        
        self.M =  [m.value for m in M]
        # print(prob.value)
        return [m.value for m in M],[p.value for p in Phi[1:]]+[Phi[0]]
    
    def solve_b_star(self,b_target,A,B,e_x,e_u,H,K_stab):

        # Solve for the offset b that is closest to b_target while maintaining feasibility of the DAP.

        M = [cp.Variable((self.u_dim,self.x_dim)) for _ in range(H)]
        b_star = cp.Variable(self.u_dim)

        omega,phi = self.omega_phi(M,A,B,e_x,e_u,H,K_stab,b_star)

        prob = cp.Problem(cp.Minimize(cp.norm(b_star-b_target)),constraints = omega)

        prob.solve(verbose=False)

        return b_star.value