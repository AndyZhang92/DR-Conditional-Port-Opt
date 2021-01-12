import utils.datareader
import pandas as pd
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def gen_portfolio_data_table(ts_code_generator, info_list, sample_stock_num):
    SZ_code_list = pd.read_pickle(project_path+'/SZ_code_list.pkl')
    SH_code_list = pd.read_pickle(project_path+'/SH_code_list.pkl')
    trading_dates = pd.read_pickle(project_path+'/trading_dates.pkl')
    total_code_list = pd.concat([SZ_code_list, SH_code_list])
    sampled_ts_codes = ts_code_generator.choice(total_code_list, size = sample_stock_num, replace = False)
    portfolio = {col_name:pd.DataFrame(index = trading_dates) for col_name in info_list}
    for ts_code in sampled_ts_codes:
        table = utils.datareader.load_joint_table(ts_code)
        for info_name in info_list:
            portfolio[info_name][ts_code] = table[info_name]
    for info_name in info_list:
        portfolio[info_name].fillna(0, inplace = True)
    return portfolio

def gen_SP500_portfolio_data_table(ts_code_generator, info_list, sample_stock_num):
    trading_dates = pd.read_pickle(project_path+'/trading_dates.pkl')
    total_code_list = pd.read_pickle(project_path+'/SP500_tics.pkl')
    sampled_ts_codes = ts_code_generator.choice(total_code_list, size = sample_stock_num, replace = False)
    portfolio = {col_name:pd.DataFrame(index = trading_dates) for col_name in info_list}
    for ts_code in sampled_ts_codes:
        table = utils.datareader.load_joint_table(ts_code)
        for info_name in info_list:
            portfolio[info_name][ts_code] = table[info_name]
    for info_name in info_list:
        portfolio[info_name].fillna(0, inplace = True)
    return portfolio