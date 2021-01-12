import numpy as np
import mosek
import sys
import cvxpy as cp
from scipy import sparse

def ReLU(np_vec):
    res = np.zeros_like(np_vec)
    res[np_vec>0] = np_vec[np_vec>0]
    return res
    
def DR_mean_variance_long_only_opt_cvx_kernel(cov_t, mean_t, reg_params, delta, p):
    """
    CVXPY solver kernel for distributionally robust optimization problem:
    
    min_{theta} sqrt{theta^T Cov_{mathbb{P}_0}(R) theta} 
    - lambda theta^toprm{E}_{mathbb{P}_0}(R)
    + sqrt{1+lambda^2}cdot sqrt{delta} |theta|_p
    """
    
    sample_stock_num = mean_t.shape[0]
    eigval, eigvecs = np.linalg.eig(cov_t)
    F = np.diag(np.sqrt(ReLU(eigval))) @ eigvecs.T
    theta = cp.Variable(sample_stock_num)
    objective_expression = (cp.norm2(cp.matmul(F,theta)) 
                            - reg_params*cp.matmul(mean_t,theta)
                            + np.sqrt(delta*(1+reg_params**2))*cp.norm(theta,p=p))
    objective = cp.Minimize(objective_expression)
    constraints = [0 <= theta, cp.sum(theta) == 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return theta.value

def DR_mean_variance_long_only_opt_mosek_kernel_p2(cov_t, mean_t, reg_params, delta, p):
    """
    MOSEK solver kernel for distributionally robust optimization problem:
    
    min_{theta} sqrt{theta^T Cov_{mathbb{P}_0}(R) theta} 
    - lambda theta^toprm{E}_{mathbb{P}_0}(R)
    + sqrt{1+lambda^2}cdot sqrt{delta} |theta|_2
    """
    assert p == 2
    sample_stock_num = mean_t.shape[0]
    eigval, eigvecs = np.linalg.eig(cov_t)
    F = np.diag(np.sqrt(ReLU(eigval))) @ eigvecs.T
    # For the MOSEK API, we are going to introduce var_num of total decision varibles, labeled from 0, 1, ..., varnum
    # sample_stock_num*2 + 2 variables
    # The first sample_stock_num variables x_0,...,  x_{sample_stock_num-1} represents vector theta
    # The next sample_stock_num variables x_{sample_stock_num},...,  x_{2*sample_stock_num-1} represents vector F*theta
    # where F^T @ F = Cov
    # The variable x_{2*sample_stock_num} represents the 2 norm of {F*theta}
    # The last variable x_{2*sample_stock_num+1} represents the p-norm of {theta}
    # ==========================================================================
    # n+2 linear constraint:
    # F @ x_{0:sample_stock_num-1} - x_{sample_stock_num:(2*sample_stock_num-1)} = 0 --- (constraint 0 to n-1)
    # sum(x_{0:sample_stock_num-1}) == 1 --- (constraint n)
    
        
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    #def streamprinter(msg):
	#    sys.stdout.write(msg)
	#    sys.stdout.flush()
    with mosek.Env() as env:
        with env.Task() as task:
            # Since the actual value of Infinity is ignored, we define it solely
            # for symbolic purposes:
            inf = 0.0
            #task.set_Stream(mosek.streamtype.log, streamprinter)
            # Set up and input bounds and linear coefficients
            bkc = [mosek.boundkey.fx]*sample_stock_num + [mosek.boundkey.fx]
            blc = [0.0]*sample_stock_num + [1.0]
            buc = [0.0]*sample_stock_num + [1.0]
            
            bkx = [mosek.boundkey.lo]*sample_stock_num + [mosek.boundkey.fr]*(sample_stock_num+2)
            blx = [0.0]*sample_stock_num + [-inf]*(sample_stock_num+2)
            bux = [inf]*sample_stock_num + [inf]*(sample_stock_num+2)
            c = list(-reg_params*mean_t) + [0.0]*(sample_stock_num) + [1.0, np.sqrt((1+reg_params**2)*delta)]
            asub = [list(range(sample_stock_num+1)) for var_id in range(sample_stock_num)] \
            	+ [[var_id] for var_id in range(sample_stock_num)]\
            	+ [[],[]]
            aval = [[F[con_id, var_id] for con_id in range(sample_stock_num)] + [1.0] for var_id in range(sample_stock_num)]\
            	+ [[-1] for var_id in range(sample_stock_num)]\
            	+ [[],[]]
            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                # Input column j of A
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.
            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])
            # Set up conic constraint
            # The folling is the conic constraint corresponding for sqrt{theta^T Cov_{mathbb{P}_0}(R) theta}
            task.appendcone(mosek.conetype.quad,
                            0.0, 
                            [2*sample_stock_num]+list(range(sample_stock_num,2*sample_stock_num)))
            # The folling is the conic constraint corresponding for |theta|_p
            
            task.appendcone(mosek.conetype.quad,
                            0.0, 
                            [2*sample_stock_num+1]+list(range(0,sample_stock_num)))
            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            # Optimize
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)
            # Output a solution
            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr,
                       xx)
            if solsta == mosek.solsta.optimal:
                return np.array(xx)[:sample_stock_num]
                #print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.dual_infeas_cer:
                print("Primal or dual infeasibility.\n")
            elif solsta == mosek.solsta.prim_infeas_cer:
                print("Primal or dual infeasibility.\n")
            elif mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

def DR_mean_variance_long_only_opt_mosek_kernel(cov_t, mean_t, reg_params, delta, p):
    """
    MOSEK solver kernel for distributionally robust optimization problem:
    
    min_{theta} sqrt{theta^T Cov_{mathbb{P}_0}(R) theta} 
    - lambda theta^toprm{E}_{mathbb{P}_0}(R)
    + sqrt{1+lambda^2}cdot sqrt{delta} |theta|_p
    """
    sample_stock_num = mean_t.shape[0]
    eigval, eigvecs = np.linalg.eig(cov_t)
    F = np.diag(np.sqrt(ReLU(eigval))) @ eigvecs.T
    # For the MOSEK API, we are going to introduce var_num of total decision varibles, labeled from 0, 1, ..., varnum
    # sample_stock_num*2 + 2 variables
    # The first sample_stock_num variables x_0,...,  x_{sample_stock_num-1} represents vector theta
    # The next sample_stock_num variables x_{sample_stock_num},...,  x_{2*sample_stock_num-1} represents vector F*theta
    # where F^T @ F = Cov
    # The variables x_{2*sample_stock_num:3*sample_stock_num-1} are anxillary variables for constructing p-norm cone
    # See https://docs.mosek.com/modeling-cookbook/powo.html Sec 4.2.2 for reference
    # The variable x_{3*sample_stock_num} represents the 2 norm of {F*theta}
    # The last variable x_{3*sample_stock_num+1} represents the p-norm of {theta}
    # ==========================================================================
    # n+2 linear constraint:
    # F @ x_{0:sample_stock_num-1} - x_{sample_stock_num:(2*sample_stock_num-1)} = 0 --- (constraint 0 to n-1)
    # sum(x_{0:sample_stock_num-1}) == 1 --- (constraint n)
    # sum(x_{2*sample_stock_num:3*sample_stock_num-1}) - x_{3*sample_stock_num+1} == 0 --- (constraint n+1)
    
        
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    with mosek.Env() as env:
        with env.Task() as task:
            # Since the actual value of Infinity is ignored, we define it solely
            # for symbolic purposes:
            inf = 0.0
            # task.set_Stream(mosek.streamtype.log, streamprinter)
            # Set up and input bounds and linear coefficients
            bkc = [mosek.boundkey.fx]*sample_stock_num + [mosek.boundkey.fx, mosek.boundkey.fx]
            blc = [0.0]*sample_stock_num + [1.0, 0.0]
            buc = [0.0]*sample_stock_num + [1.0, 0.0]
            
            bkx = [mosek.boundkey.lo]*sample_stock_num + [mosek.boundkey.fr]*(2*sample_stock_num+2)
            blx = [0.0]*sample_stock_num + [-inf]*(2*sample_stock_num+2)
            bux = [inf]*sample_stock_num + [inf]*(2*sample_stock_num+2)
            c = list(-reg_params*mean_t) + [0.0]*(2*sample_stock_num) + [1.0, np.sqrt((1+reg_params**2)*delta)]
            asub = [list(range(sample_stock_num+1)) for var_id in range(sample_stock_num)] \
            	+ [[var_id] for var_id in range(sample_stock_num)]\
            	+ [[sample_stock_num+1] for var_id in range(sample_stock_num)]\
            	+ [[],[sample_stock_num+1]]
            aval = [[F[con_id, var_id] for con_id in range(sample_stock_num)] + [1.0] for var_id in range(sample_stock_num)]\
            	+ [[-1] for var_id in range(sample_stock_num)]\
            	+ [[1] for var_id in range(sample_stock_num)]\
            	+ [[],[-1]]
            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                # Input column j of A
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.
            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])
            # Set up conic constraint
            # The folling is the conic constraint corresponding for sqrt{theta^T Cov_{mathbb{P}_0}(R) theta}
            task.appendcone(mosek.conetype.quad,
                            0.0, 
                            [3*sample_stock_num]+list(range(sample_stock_num,2*sample_stock_num)))
            # The folling is the conic constraint corresponding for |theta|_p
            for i in range(sample_stock_num):
	            task.appendcone(mosek.conetype.ppow,
	                            1/p, 
	                            [2*sample_stock_num+i, 3*sample_stock_num+1, i])
            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            # Optimize
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)
            # Output a solution
            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr,
                       xx)
            if solsta == mosek.solsta.optimal:
                return np.array(xx)
                #print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.dual_infeas_cer:
                print("Primal or dual infeasibility.\n")
            elif solsta == mosek.solsta.prim_infeas_cer:
                print("Primal or dual infeasibility.\n")
            elif mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")
    