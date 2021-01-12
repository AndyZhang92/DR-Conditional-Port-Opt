import numpy as np

def equal_weight(cov_t, mean_t):
    """benchmark solver for equally weighted portfolio
    """
    
    sample_stock_num = mean_t.shape[0]
    theta = np.ones(sample_stock_num) / sample_stock_num
    return theta