import numpy as np
import cvxpy as cp

def ModelEst(x_hist,u_hist,eta_bar=1):
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
    
    return A_hat.value,B_hat.value,r

def OMEGA(M,A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max):
    kappa = 1
    gamma = 0.9999
    kappa_B = 1
    if len(x_hist.shape)>1:
        x_dim  = x_hist.shape[-1]
    else:
        x_dim = 1

    if len(u_hist.shape)>1:
        u_dim = u_hist.shape[-1]
    else:
        u_dim = 1
    
    
   # Robust CE
    z_max = np.sqrt(x_max**2+u_max**2)
    D_inf = np.linalg.norm(D_x,np.inf)
    
    # calculate the constraint constants.
    e_theta = D_inf*z_max*kappa/gamma * r+5 * kappa**4 * kappa_B * D_inf*w_max/gamma**3 * np.sqrt(x_dim*u_dim) * r
    e_eta_x = D_inf * kappa * kappa_B/gamma * np.sqrt(x_dim)*eta_bar
    e_eta_u = D_inf * eta_bar
    e_H = D_inf * kappa * x_max * (1-gamma)**H
    e_v = D_inf * w_max *kappa * kappa_B/gamma**2 * np.sqrt(x_dim*u_dim*H)*D_M

    e_x = e_theta + e_eta_x + e_H + e_v
    e_u = e_eta_u

    # print(e_x,e_u,D_inf)
    # The transition kernel Phi
    Phi = []
    A_pow = [np.eye(x_dim)]

    for i in range(H):
        A_pow.append(A_pow[-1].dot(A_hat))
    # A_pow[n] = A_hat^n

    for k in range(1,2*H+1):

        for i in range(1,H+1):
            if 0<=k-i and k-i<=H-1:
                if i==1:
                    Phi_k = A_pow[i-1].dot(B_hat) @ M[k-i]
                else:
                    Phi_k += A_pow[i-1].dot(B_hat) @ M[k-i]

        if k>H:
            Phi_k += np.zeros(A_hat.shape)
        else:
            Phi_k += A_pow[k-1]


        Phi.append(Phi_k)

    # Construct the safe policy set
    x_constraints = []
    u_constraints = []
    for i in range(len(D_x)):
        g_x = cp.sum([cp.norm(D_x[i,:] @ Phi[k],1) for k in range(2*H)]) * w_max
        x_constraints.append(g_x <= d_x[i,0]-e_x)

    for j in range(len(D_u)):
        g_u = cp.sum([cp.norm(D_u[j,:] @ M[k],1) for k in range(H)]) * w_max
        u_constraints.append(g_u <= d_u[j,0]-e_u)

    OMEGA = x_constraints +  u_constraints

    return OMEGA,Phi

def RobustCE(A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max):


    if len(x_hist.shape)>1:
        x_dim  = x_hist.shape[-1]
    else:
        x_dim = 1

    if len(u_hist.shape)>1:
        u_dim = u_hist.shape[-1]
    else:
        u_dim = 1
    
    Q = np.eye(x_dim)*0.1
    R = np.eye(u_dim)*0.5
 
    # The policy M
    M = [cp.Variable((u_dim,x_dim)) for _ in range(H)]

    omega,Phi =  OMEGA(M,A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max)
  

    # Construct the control input 
    w_hat_hist = (x_hist[1:].T - A_hat.dot(x_hist[:-1].T)-B_hat.dot(u_hist.T)).T
    u = cp.sum([M[k] @ w_hat_hist[-1-k] for k in range(H)])

    # Construct the expected approx state
    x_tilde = cp.sum([Phi[k] @ w_hat_hist[-1-k] for k in range(2*H)])

    # The LQR loss
    objective = cp.quad_form(x_tilde,Q) + cp.quad_form(u,R)
    # The problem
    prob = cp.Problem(cp.Minimize(objective),omega)
    prob.solve()
    
    return [M.value for M in M]

def ApproxDAP(M,A_hat,B_hat,eta_bar,x_hist,u_hist):
    H = len(M)
    if len(x_hist.shape)>1:
        x_dim  = x_hist.shape[-1]
    else:
        x_dim = 1

    if len(u_hist.shape)>1:
        u_dim = u_hist.shape[-1]
    else:
        u_dim = 1

    u_dim = B_hat.shape[-1]

    # The current exploration noise
    eta_t = (np.random.rand(u_dim).reshape(-1,u_dim)-0.5) * eta_bar

    w_hat_hist = (x_hist[1:].T - A_hat.dot(x_hist[:-1].T)-B_hat.dot(u_hist.T)).T

    u = np.sum([M[k].dot(w_hat_hist[-1-k]) for k in range(H)])+eta_t

    return u

# Compute the maximum norm within a polytope

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


# Get M_mid
def M_mid(A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max,A_prev,B_prev,r_prev,eta_bar_prev,x_prev,u_prev,M_prev,M_dest):
    if len(x_hist.shape)>1:
        x_dim  = x_hist.shape[-1]
    else:
        x_dim = 1

    if len(u_hist.shape)>1:
        u_dim = u_hist.shape[-1]
    else:
        u_dim = 1

    M = [cp.Variable((u_dim,x_dim)) for _ in range(H)]


    omega1,_ = OMEGA(M,A_hat,B_hat,r,H,eta_bar,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_hist,u_hist,w_max)
    omega2,_ = OMEGA(M,A_prev,B_prev,r_prev,H,eta_bar_prev,D_M,D_x,d_x,D_u,d_u,x_max,u_max,x_prev,u_prev,w_max)

    omega_intersection = omega1+omega2

    prob = cp.Problem(cp.Minimize(cp.sum([cp.norm(M_prev[k]-M[k],2)**2+cp.norm(M_dest[k]-M[k],2)**2 for k in range(H)])),omega_intersection)

    prob.solve()
    return [m.value for m in M]