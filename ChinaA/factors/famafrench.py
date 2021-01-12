# The goal of this file is to compute the Fama-French three factors
# in the chinese market
import utils
import solvers
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def gen_three_factors():
    gen_market_pct_chg()
    gen_SMB_pct_chg()
    gen_HML_pct_chg()

def gen_market_pct_chg():
    SZ_code_list = pd.read_pickle(project_path+'/SZ_code_list.pkl')
    SH_code_list = pd.read_pickle(project_path+'/SH_code_list.pkl')
    trading_dates = pd.read_pickle(project_path+'/trading_dates.pkl')
    total_code_list = pd.concat([SZ_code_list, SH_code_list])
    total_stock_num = len(total_code_list)
    portfolio_data = utils.preprocess.gen_portfolio_data_table(
        np.random,
        ['pct_chg', 'total_mv'],
        len(total_code_list)
    )
    print('finish loading all stock data for stock num = {}'.format(len(total_code_list)))
    market_pct_chg = (portfolio_data['pct_chg'] * portfolio_data['total_mv']).sum(axis = 1) / portfolio_data['total_mv'].sum(axis = 1)
    file_dir = project_path+'/factor_returns/market_pct_chg.pkl'
    market_pct_chg.to_pickle(file_dir)
    print('Market percentage returns saved to:\n',file_dir)
    
def gen_SMB_pct_chg():
    SZ_code_list = pd.read_pickle(project_path+'/SZ_code_list.pkl')
    SH_code_list = pd.read_pickle(project_path+'/SH_code_list.pkl')
    trading_dates = pd.read_pickle(project_path+'/trading_dates.pkl')
    total_code_list = pd.concat([SZ_code_list, SH_code_list])
    total_stock_num = len(total_code_list)
    portfolio_data = utils.preprocess.gen_portfolio_data_table(
        np.random,
        ['pct_chg', 'total_mv'],
        len(total_code_list)
    )
    print('finish loading all stock data for stock num = {}'.format(len(total_code_list)))
    # construting filters for the small-cap portfolio
    filters = portfolio_data['total_mv'].lt(portfolio_data['total_mv'].median(axis = 1), axis = 0)
    small_cap_portfolio_pct_ret = (portfolio_data['pct_chg']*filters).sum(axis = 1)/filters.sum(axis = 1)
    big_cap_portfolio_pct_ret = (portfolio_data['pct_chg']*(~filters)).sum(axis = 1)/(~filters).sum(axis = 1)
    SMB_pct_ret = small_cap_portfolio_pct_ret - big_cap_portfolio_pct_ret
    
    file_dir = project_path+'/factor_returns/small_cap_portfolio_pct_ret.pkl'
    small_cap_portfolio_pct_ret.to_pickle(file_dir)
    print('small cap percentage returns saved to:\n',file_dir)
    file_dir = project_path+'/factor_returns/big_cap_portfolio_pct_ret.pkl'
    big_cap_portfolio_pct_ret.to_pickle(file_dir)
    print('big cap percentage returns saved to:\n',file_dir)
    file_dir = project_path+'/factor_returns/SMB_pct_ret.pkl'
    SMB_pct_ret.to_pickle(file_dir)
    print('SMB percentage returns saved to:\n',file_dir)
    
def gen_HML_pct_chg():
    SZ_code_list = pd.read_pickle(project_path+'/SZ_code_list.pkl')
    SH_code_list = pd.read_pickle(project_path+'/SH_code_list.pkl')
    trading_dates = pd.read_pickle(project_path+'/trading_dates.pkl')
    total_code_list = pd.concat([SZ_code_list, SH_code_list])
    total_stock_num = len(total_code_list)
    portfolio_data = utils.preprocess.gen_portfolio_data_table(
        np.random,
        ['pct_chg', 'pe'],
        len(total_code_list)
    )
    print('finish loading all stock data for stock num = {}'.format(len(total_code_list)))
    high_value_filter = portfolio_data['pe'].le(portfolio_data['pe'].quantile(q = 0.3, axis = 1), axis = 0)
    low_value_filter = portfolio_data['pe'].ge(portfolio_data['pe'].quantile(q = 0.7, axis = 1), axis = 0)
    mid_value_filter = ~(low_value_filter|high_value_filter)
    # compute portfolio return
    high_value_filter_portfolio_pct_ret = ((portfolio_data['pct_chg']*high_value_filter).sum(axis = 1)
                                           / high_value_filter.sum(axis = 1))
    low_value_filter_portfolio_pct_ret = ((portfolio_data['pct_chg']*low_value_filter).sum(axis = 1)
                                           / low_value_filter.sum(axis = 1))
    mid_value_filter_portfolio_pct_ret = ((portfolio_data['pct_chg']*mid_value_filter).sum(axis = 1)
                                           / mid_value_filter.sum(axis = 1))
    HML_pct_ret = high_value_filter_portfolio_pct_ret - low_value_filter_portfolio_pct_ret
    # Save the result
    file_dir = project_path+'/factor_returns/low_value_filter_portfolio_pct_ret.pkl'
    low_value_filter_portfolio_pct_ret.to_pickle(file_dir)
    print('low value percentage returns saved to:\n',file_dir)
    file_dir = project_path+'/factor_returns/mid_value_filter_portfolio_pct_ret.pkl'
    mid_value_filter_portfolio_pct_ret.to_pickle(file_dir)
    print('mid value percentage returns saved to:\n',file_dir)
    file_dir = project_path+'/factor_returns/high_value_filter_portfolio_pct_ret.pkl'
    high_value_filter_portfolio_pct_ret.to_pickle(file_dir)
    print('high value percentage returns saved to:\n',file_dir)
    file_dir = project_path+'/factor_returns/HML_pct_ret.pkl'
    HML_pct_ret.to_pickle(file_dir)
    print('HML percentage returns saved to:\n',file_dir)
    
    
    
    
    