import numpy as np

def rolling_opt_framework(training_ids, mean_nparray, cov_nparray, solver, **solver_kwargs):
    '''meta-framework for the mean-variance optimization
    '''
    theta_nparray = np.zeros_like(mean_nparray)
    for timestamp_id in training_ids:
        mean_t = mean_nparray[timestamp_id]
        cov_t = cov_nparray[timestamp_id*cov_nparray.shape[1]:(timestamp_id+1)*cov_nparray.shape[1]]
        index = np.where(np.diag(cov_t) != 0)[0]
        cov_sub_t = cov_t[index,:][:,index] # compute a non-singular submatrix
        mean_sub_t = mean_t[index]
        theta_t = np.zeros_like(mean_t)
        theta_t[index] = solver(cov_t=cov_sub_t, mean_t=mean_sub_t, **solver_kwargs)
        theta_nparray[timestamp_id,:] = theta_t
    return theta_nparray