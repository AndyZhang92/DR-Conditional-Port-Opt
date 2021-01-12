import numpy as np
import cvxpy as cp
import scipy

def DR_Winfty_conditional_mean_variance_long_only_opt_cvx_kernel(
    X_mat, Y_mat, X0, reg_params, gamma_quantile, rho_quantile
):
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem:
    
    See problem formulation in DR_Conditional_EstimationWinfty.ipynb
    """
    X_dist = np.linalg.norm(X_mat - X0, axis = 1)
    X_dist[np.isnan(X_dist)] = 1e8
    gamma = np.quantile(X_dist, gamma_quantile)
    rho =  np.quantile(X_dist, rho_quantile)
    eta = reg_params
    import warnings
    warnings.filterwarnings("error")
    try:
        idx_I = (X_dist <= gamma + rho)
        idx_I1 = (X_dist + rho <= gamma)
        idx_I2 = idx_I & (~idx_I1)
    except RuntimeWarning:
        print(X_dist)
        print(gamma)
        print(rho)
    norm_x_minus_xp_in_I = X_dist[idx_I] - gamma
    norm_x_minus_xp_in_I[norm_x_minus_xp_in_I<0] = 0
    y_I = Y_mat[idx_I]
    dim_beta = Y_mat.shape[1]
    if len(y_I)==0:
        return np.ones(dim_beta)/dim_beta
    alpha = cp.Variable(shape = (1,), name = 'alpha')
    beta = cp.Variable(shape = (dim_beta,), name = 'beta', nonneg=True)
    lambda_ = cp.Variable(shape = (1,), name = 'lambda')
    u = cp.Variable(shape = (len(y_I),), name = 'u')
    v_exp_term_1 = cp.abs(cp.matmul(Y_mat[idx_I], beta) - alpha - 0.5*eta)
    v_exp_term_2 = cp.norm(beta)*(rho-norm_x_minus_xp_in_I)
    constraints = [
        u[idx_I2[idx_I]] >= 0,
        cp.sum(u) <= 0,
        cp.sum(beta) == 1,
        lambda_ + u + eta*alpha + 0.25*eta**2 >= cp.square(v_exp_term_1+v_exp_term_2)
    ]
    problem = cp.Problem(cp.Minimize(lambda_), constraints)
    problem.solve()
    assert problem.status == 'optimal'
    return beta.value

def DR_W2_conditional_mean_variance_long_only_opt_cvx_kernel_new(
    X_mat, Y_mat, X0, reg_params, epsilon, rho_div_rho_min
):
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem:
    
    See problem formulation in DR_Conditional_EstimationW2.ipynb
    """
    def compute_rho_min(X_mat, X0, epsilon):
        X_dist = np.linalg.norm(X_mat - X0, axis = 1)
        X_dist[np.isnan(X_dist)] = 1e8
        X_cut = np.quantile(X_dist, q=epsilon, interpolation = 'higher')
        return X_dist[X_dist <= X_cut].mean() * epsilon
    
    rho = rho_div_rho_min * compute_rho_min(X_mat, X0, epsilon)
    X_dist = np.linalg.norm(X_mat - X0, axis = 1)
    X_dist[np.isnan(X_dist)] = 1e8
    
    eta = reg_params
    epsilon_inv = 1/epsilon;

    N, sample_stock_num = Y_mat.shape

    m = cp.Variable(N, nonneg = True)
    beta = cp.Variable(sample_stock_num)
    alpha = cp.Variable(1)
    lambda1 = cp.Variable(1, nonneg = True)
    denom = cp.Variable(1, nonneg = True)
    lambda2 = cp.Variable(1)
    linear_expr = Y_mat@beta-alpha-0.5*eta
    obj = lambda1*rho+lambda2*epsilon + cp.sum(cp.pos(epsilon_inv*m
                                                    - 0.25*epsilon_inv*eta**2 
                                                    - epsilon_inv*eta*alpha 
                                                    - lambda1*X_dist - lambda2))/N
    constraints = [m >= cp.hstack([cp.quad_over_lin(linear_expr[i], denom) for i in range(N)]),
                  epsilon_inv*cp.quad_over_lin(beta, lambda1) + denom <= 1,
                  beta >= 0,
                  cp.sum(beta) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    assert prob.status == 'optimal'
    return beta.value

def DR_W2_conditional_mean_variance_long_only_opt_cvx_kernel(
    X_mat, Y_mat, X0, reg_params, epsilon, rho_div_rho_min
):
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem:
    
    See problem formulation in DR_Conditional_EstimationW2.ipynb
    """
    def compute_rho_min(X_mat, X0, epsilon):
        X_dist = np.linalg.norm(X_mat - X0, axis = 1)
        X_dist[np.isnan(X_dist)] = 1e8
        X_cut = np.quantile(X_dist, q=epsilon, interpolation = 'higher')
        return X_dist[X_dist <= X_cut].mean() * epsilon
    
    rho = rho_div_rho_min * compute_rho_min(X_mat, X0, epsilon)
    X_dist = np.linalg.norm(X_mat - X0, axis = 1)
    X_dist[np.isnan(X_dist)] = 1e8
    
    eta = reg_params
    epsilon_inv = 1/epsilon;

    N, sample_stock_num = Y_mat.shape
    c1 = cp.Parameter(nonneg = True)
    c2 = cp.Parameter(nonneg = True)
    beta = cp.Variable(sample_stock_num)
    alpha = cp.Variable(1)
    lambda1 = cp.Variable(1, nonneg = True)
    lambda2 = cp.Variable(1)
    obj = lambda1*rho+lambda2*epsilon + cp.sum(cp.pos(c1*(Y_mat@beta-alpha-0.5*eta)**2 
                                                    - 0.25*epsilon_inv*eta**2 
                                                    - epsilon_inv*eta*alpha 
                                                    - lambda1*X_dist - lambda2)) / N
    constraints = [cp.sum_squares(beta) <= c2*lambda1,
                  beta >= 0,
                  cp.sum(beta) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    def obj_val(t):
        c1.value = epsilon_inv + t
        c2.value = t/(epsilon_inv*(epsilon_inv+t))
        return prob.solve()
    scipy.optimize.minimize_scalar(obj_val)
    assert prob.status == 'optimal'
    return beta.value