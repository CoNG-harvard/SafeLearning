import numpy as np
import cvxpy as cp

def ModelEst(x_hist,u_hist,w_max=0.2,eta_bar=1,stability_gap = 0.1):
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
    
    return np.sum([M[k].dot(w_hist[-k-1]) for k in range(l)])+(np.random.rand()-0.5)*2*eta_bar



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