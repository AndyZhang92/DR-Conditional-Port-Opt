import tushare
import numpy as np
import pandas as pd
import os
import tqdm

__author__ = 'FanZhang'
__project_dir__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
start_date = '20100101'
end_date = '20200101'
local_data_path = '/home/groups/jblanche/DR-Conditional-Port-Opt/WRDSData/'

def download_stock_data(ts_code_list):
    for ts_code in tqdm.tqdm(ts_code_list):    
        df = tushare.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date)
        df.to_csv(__project_dir__+'/data/' + ts_code, mode = 'w')
    print("done")

def download_stock_fundamental_data(ts_code_list):
    for ts_code in tqdm.tqdm(ts_code_list):    
        df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, 
                             fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,dv_ratio,total_mv')
        df.to_csv(__project_dir__+'/data/' + ts_code + '_fundamental', mode = 'w')
    print("done")
    
def combine_table(ts_code_list):
    for ts_code in tqdm.tqdm(ts_code_list):
        price_table = pd.read_csv('./data/{}'.format(ts_code))
        price_table['time'] = pd.to_datetime(price_table['trade_date'], format = '%Y%m%d')
        price_table = price_table.set_index('time')
        price_table = price_table.drop(columns = ['Unnamed: 0', 'trade_date'])
        model_state_table =  pd.read_csv('./data/{}_fundamental'.format(ts_code))
        model_state_table['time'] = pd.to_datetime(model_state_table['trade_date'], format = '%Y%m%d')
        model_state_table = model_state_table.set_index('time')
        model_state_table = model_state_table.drop(columns = ['Unnamed: 0', 'trade_date', 'ts_code'])
        joint_table = price_table.join(model_state_table, how = 'outer').iloc[::-1]
        joint_table.to_pickle(local_data_path+'join_table_{}.pkl'.format(ts_code))
    print('done')

def load_joint_table(ts_code):
    return pd.read_pickle(local_data_path+'join_table_{}.pkl'.format(ts_code))

def load_fama_french_factors(normalized = False):
    factors_df = pd.read_pickle(__project_dir__ + '/factor_returns/FamaFrench3_pct_ret.pkl')
    if normalized:
        factors_rolling_mean = factors_df.rolling('730d').mean()
        factors_rolling_std = factors_df.rolling('730d').std()
        factors_rolling_normalized = (factors_df - factors_rolling_mean) / factors_rolling_std
        return factors_rolling_normalized
    else:
        return factors_df