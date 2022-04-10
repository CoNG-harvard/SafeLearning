# How to run the experiments

There are 3 notebooks in this folder: [DataCollection], [PlottingCore], and [Plotting Constraint Violation.]

Run the [DataCollection] notebook to carry out the simulations, and store the simulation results as Python pickle files(.pkl) in the data/ folder.

Run [Plotting Core] to plot the regret and estimation error figures. Run [Plotting Constraint Violation] to plot the simulation trajectory figures showing constraint violation/satisfaction. The figures are stored in the figures/ folder.



# Background

This folder contains experiments for the SafeLearning algorithm of systems with non-zero equilibrium points. The non-zero equilibrium is achieved by setting the base controller in the form of $K_{stab}x+b$, so that the system dynamics is
$$
x_{t+1} = (A-BK_{stab})x_{t}+B(-b+u_t)+w_t
$$
The relationship between the equilibrium position($\bar{x}$) and b is
$$
\bar{x} = (A-BK_{stab}-I)^{-1}Bb
$$
 assuming $A-BK_{stab}-I$ is invertible.

The $b$ value needs to be passed into the method $env.step(x,b)$ for such offset in equilibrium point to take effect.

# Setting $b_{target}$ in Safe Learning

We know that since there are $e_x,e_u$ corrections in the DAP algorithm, choosing a $b_{target}$ such that $\bar{x}$ becomes very close to the $x_{max/min}$ boundaries often results in infeasibility of the algorithm. Therefore we developed the following method to alleviate such infeasibility issue.

Recall the DAP algorithm solves an optimization problem in the form of
$$
\min_M \text{LQR Cost about } M\\
s.t.\ M\in\Omega(A,B,H,e_x,e_u,K_{stab},b)
$$
The implementation of $\Omega$ can be found in the method controllers.SafeDAP.omega_phi. The SafeDAP class can be found in the file controllers.py

We first specify a $b_{target_0}$ which results in an equilibrium point $x_{target}$ according to equation (2). See the variable $b\_target$ in the [DataCollection] notebook. 

Typically, $b_{target_0}$ is chosen so that $x_{target}$ is very close to the boundary. Plugging in such a $b_{target_0}$ into $\Omega$ often results in infeasiblity$(\Omega=\empty)$. Therefore, the Safe Learning algorithm will use a $b_{target}$ that makes $\Omega$ nonempty and is as close to $b_{target_0}$ as possible. That is
$$
b_{target}=\arg\min_{b,M} ||b-b_{target_0}||\\
s.t.\ M\in \Omega(A,B,H,e_x,e_u,K_{stab},b)
$$
The decision variable $M$ only decides the feasibility of equation (4), but does not contributes to the objective. The implementation of (4) is in controllers.SafeDAP.solve_b_star()

Before Safe Learning starts, we first solve (4) to obtain $b_{target}$(using true $A,B$), then plug $b_{target}$ in (3) in place of $b$ to ensure we have feasible solution to $M$ every time we call controllers.SafeDAP.solve(). 

Meanwhile, $e_x,e_u$ becomes smaller after each episode according to a fixed decay rate(see the variable $decay$ in the [DataCollection] notebook), and which means using the same $b_{target}$ will keep the problem feasible.
