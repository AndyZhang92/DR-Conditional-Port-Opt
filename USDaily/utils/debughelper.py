import utils.datareader
import pandas as pd
import numpy as np
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_sample_mean_cov(sampled_ts_codes, integer_id):
    num_stocks = len(sampled_ts_codes)
    trading_dates = pd.read_pickle(project_path+'/trading_dates.pkl')
    info_list = ['pct_chg']
    portfolio = {col_name:pd.DataFrame(index = trading_dates) for col_name in info_list}
    for ts_code in sampled_ts_codes:
        table = utils.datareader.load_joint_table(ts_code)
        for info_name in info_list:
            portfolio[info_name][ts_code] = table[info_name]
    for info_name in info_list:
        portfolio[info_name].fillna(0, inplace = True)
    mean_nparray = portfolio['pct_chg'].rolling(window = '730d').mean().values
    cov_nparray = portfolio['pct_chg'].rolling(window = '730d').cov().values
    sampled_mean_vec = mean_nparray[integer_id]
    sampled_cov_mat = cov_nparray[integer_id*num_stocks:(integer_id+1)*num_stocks]
    return sampled_mean_vec, sampled_cov_mat