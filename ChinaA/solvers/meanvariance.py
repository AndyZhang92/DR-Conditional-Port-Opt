import numpy as np
import mosek
from scipy import sparse

def mean_variance_opt(training_ids, mean_nparray, cov_nparray, reg_params):
    '''mean variance optimization without any constraint
    model:
    min theta^T Cov theta - reg * theta^T mean
    s.t. theta is a real vector
    '''
    theta_nparray = np.zeros_like(mean_nparray)
    for timestamp_id in training_ids:
        mean_t = mean_nparray[timestamp_id]
        cov_t = cov_nparray[timestamp_id*cov_nparray.shape[1]:(timestamp_id+1)*cov_nparray.shape[1]]
        index = np.where(np.diag(cov_t) != 0)[0]
        cov_sub_t = cov_t[index,:][:,index] # compute a non-singular submatrix
        mean_sub_t = mean_t[index]
        theta_t = np.zeros_like(mean_t)
        theta_t[index] = 0.5 * reg_params * np.linalg.inv(cov_sub_t) @ mean_sub_t
        theta_nparray[timestamp_id,:] = theta_t
    return theta_nparray

def mean_variance_long_only_opt(training_ids, mean_nparray, cov_nparray, reg_params):
    '''mean variance optimization for long only portfolio
    
    model:
    min theta^T Cov theta - reg * theta^T mean
    s.t. theta >= 0; sum(theta) <= 1;
    '''
    theta_nparray = np.zeros_like(mean_nparray)
    for timestamp_id in training_ids:
        mean_t = mean_nparray[timestamp_id]
        cov_t = cov_nparray[timestamp_id*cov_nparray.shape[1]:(timestamp_id+1)*cov_nparray.shape[1]]
        index = np.where(np.diag(cov_t) != 0)[0]
        cov_sub_t = cov_t[index,:][:,index] # compute a non-singular submatrix
        mean_sub_t = mean_t[index]
        theta_t = np.zeros_like(mean_t)
        theta_t[index] = mean_variance_long_only_opt_mosek_kernel(cov_sub_t, mean_sub_t, reg_params)
        theta_nparray[timestamp_id,:] = theta_t
    return theta_nparray

def mean_variance_long_only_mean_cons_opt(training_ids, mean_nparray, cov_nparray, alpha):
    '''mean variance optimization for long only portfolio
    
    model:
    min theta^T Cov theta
    s.t. theta >= 0; sum(theta) <= 1; theta^T mean >= alpha
    '''
    theta_nparray = np.zeros_like(mean_nparray)
    for timestamp_id in training_ids:
        mean_t = mean_nparray[timestamp_id]
        cov_t = cov_nparray[timestamp_id*cov_nparray.shape[1]:(timestamp_id+1)*cov_nparray.shape[1]]
        index = np.where(np.diag(cov_t) != 0)[0]
        cov_sub_t = cov_t[index,:][:,index] # compute a non-singular submatrix
        mean_sub_t = mean_t[index]
        theta_t = np.zeros_like(mean_t)
        theta_t[index] = mean_variance_long_only_mean_cons_opt_mosek_kernel(cov_sub_t, mean_sub_t, alpha)
        theta_nparray[timestamp_id,:] = theta_t
    return theta_nparray

def mean_variance_long_only_opt_mosek_kernel(cov_t, mean_t, reg_params):
    '''mean variance optimization for long only portfolio
    See https://docs.mosek.com/9.2/pythonapi/conventions.html#doc-optimizer-matrix-formats
    for references of the mosek API for quadratic optimization problem
    
    model:
    min theta^T Cov theta - reg * theta^T mean
    s.t. theta >= 0; sum(theta) == 1;
    '''
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    with mosek.Env() as env:
        with env.Task() as task:
            # Since the actual value of Infinity is ignored, we define it solely
            # for symbolic purposes:
            inf = 0.0
            #task.set_Stream(mosek.streamtype.log, streamprinter)
            # Set up and input bounds and linear coefficients
            bkc = [mosek.boundkey.fx]
            blc = [1.0]
            buc = [1.0]
            numvar = mean_t.shape[0]
            bkx = [mosek.boundkey.lo] * numvar
            blx = [0.0] * numvar
            bux = [inf] * numvar
            c = -0.5*reg_params*mean_t
            asub = [[0] for var_id in range(numvar)]
            aval = [[1.0] for var_id in range(numvar)]

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
            # Set up and input quadratic objective
            sparse_cov_tri = sparse.csr_matrix(np.tri(*cov_t.shape) * cov_t)
            qval = sparse_cov_tri.data
            qsubi, qsubj = sparse_cov_tri.nonzero()
            task.putqobj(qsubi, qsubj, qval)
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
                
def mean_variance_long_only_mean_cons_opt_mosek_kernel(cov_t, mean_t, alpha):
    '''
    mean variance optimization for long only portfolio wi
    See https://docs.mosek.com/9.2/pythonapi/conventions.html#doc-optimizer-matrix-formats
    for references of the mosek API for quadratic optimization problem
    
    model:
    min theta^T Cov theta
    s.t. theta >= 0; sum(theta) <= 1; theta^T mean >= alpha
    '''
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    with mosek.Env() as env:
        with env.Task() as task:
            # Since the actual value of Infinity is ignored, we define it solely
            # for symbolic purposes:
            inf = 0.0
            #task.set_Stream(mosek.streamtype.log, streamprinter)
            # Set up and input bounds and linear coefficients
            # constraint 1: sum(theta) <= 1
            # constraint 2: theta^T mean >= alpha
            bkc = [mosek.boundkey.up, mosek.boundkey.lo]
            blc = [-inf, alpha]
            buc = [1.0, inf]
            numvar = mean_t.shape[0]
            bkx = [mosek.boundkey.lo] * numvar
            blx = [0.0] * numvar
            bux = [inf] * numvar
            c = [0.0] * numvar
            asub = [[0,1] for var_id in range(numvar)]
            aval = [[1.0, mean_t[var_id]] for var_id in range(numvar)]

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
            # Set up and input quadratic objective
            sparse_cov_tri = sparse.csr_matrix(np.tri(*cov_t.shape) * cov_t)
            qval = sparse_cov_tri.data
            qsubi, qsubj = sparse_cov_tri.nonzero()
            task.putqobj(qsubi, qsubj, qval)
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