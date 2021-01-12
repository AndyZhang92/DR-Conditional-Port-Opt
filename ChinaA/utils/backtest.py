import utils
import solvers
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def resample_experiments(exp_num, 
                         sample_stock_num, 
                         solver, 
                         solver_name_prefix = '', 
                         start_index = '2012', 
                         end_index = '2017',
                         enforce_exp_num = None,
                         **solver_kwargs):
    SEED = 42
    ts_code_generator = np.random.RandomState(SEED)
    info_list = ['pct_chg']
    PnL_list = []
    for exp in tqdm(range(exp_num), desc = 'exp', leave=False):
        exp_start_index, exp_end_index = start_index, end_index
        prev_theta_df, prev_PnL_df = None, None
        portfolio = utils.preprocess.gen_portfolio_data_table(ts_code_generator, info_list, sample_stock_num)
        if enforce_exp_num is not None and exp != enforce_exp_num:
            continue
        file_path_fmt = utils.path.get_file_path_and_create_dir(exp, SEED, solver, solver_name_prefix, **solver_kwargs)
        if os.path.exists(file_path_fmt.format('stock_weights')):
            prev_theta_df = pd.read_pickle(file_path_fmt.format('stock_weights'))
            prev_PnL_df = pd.read_pickle(file_path_fmt.format('PnL'))
            last_available_year = prev_theta_df.index[prev_theta_df.abs().sum(axis = 1) > 1e-10][-1].year
            if last_available_year < int(end_index)-1:
                exp_start_index = str(last_available_year +1)
            else:
                continue
        mean_nparray = portfolio['pct_chg'].rolling(window = '730d').mean().values
        cov_nparray = portfolio['pct_chg'].rolling(window = '730d').cov().values
        training_ids = np.where((portfolio['pct_chg'].index >= exp_start_index) & (portfolio['pct_chg'].index < exp_end_index))[0]
        theta_nparray = solvers.rolling.rolling_opt_framework(
            training_ids = training_ids, 
            mean_nparray = mean_nparray, 
            cov_nparray = cov_nparray, 
            solver = solver, 
            **solver_kwargs)
        theta_df = pd.DataFrame(data=theta_nparray, 
                                columns=portfolio['pct_chg'].columns, 
                                index=portfolio['pct_chg'].index)
        if prev_theta_df is not None:
            theta_df = pd.concat([prev_theta_df[:str(last_available_year)], theta_df[exp_start_index:]])
        theta_df.to_pickle(file_path_fmt.format('stock_weights'))
        PnL = (theta_df * portfolio['pct_chg'].shift(-1)).sum(axis = 1)[training_ids]
        if prev_PnL_df is not None:
            PnL = pd.concat([prev_PnL_df[:str(last_available_year)], PnL[exp_start_index:]])
        PnL.to_pickle(file_path_fmt.format('PnL'))
        PnL_list.append(PnL)
    PnL_table = pd.DataFrame(PnL_list).transpose()
    return PnL_table

def resample_experiments_conditional_fama_french(exp_num, 
                                                 sample_stock_num, 
                                                 solver, 
                                                 solver_name_prefix = '', 
                                                 start_index = '2012', 
                                                 end_index = '2017',
                                                 enforce_exp_num = None,
                                                 **solver_kwargs):
    SEED = 42
    factors_df = utils.datareader.load_fama_french_factors()
    factors_rolling_mean = factors_df.rolling('730d').mean()
    factors_rolling_std = factors_df.rolling('730d').std()
    factors_rolling_normalized = (factors_df - factors_rolling_mean) / factors_rolling_std
    ts_code_generator = np.random.RandomState(SEED)
    info_list = ['pct_chg']
    PnL_list = []
    param_k_nn = 50
    for exp in tqdm(range(exp_num), desc = 'exp', leave=False):
        exp_start_index, exp_end_index = start_index, end_index
        prev_theta_df, prev_PnL_df = None, None
        portfolio = utils.preprocess.gen_portfolio_data_table(ts_code_generator, info_list, sample_stock_num)
        if enforce_exp_num is not None and exp != enforce_exp_num:
            continue
        file_path_fmt = utils.path.get_file_path_and_create_dir(exp, SEED, solver, solver_name_prefix, **solver_kwargs)
        if os.path.exists(file_path_fmt.format('stock_weights')):
            prev_theta_df = pd.read_pickle(file_path_fmt.format('stock_weights'))
            prev_PnL_df = pd.read_pickle(file_path_fmt.format('PnL'))
            last_available_year = prev_theta_df.index[prev_theta_df.abs().sum(axis = 1) > 1e-10][-1].year
            if last_available_year < int(end_index)-1:
                exp_start_index = str(last_available_year +1)
            else:
                continue
        training_ids = np.where((portfolio['pct_chg'].index >= start_index) & (portfolio['pct_chg'].index < end_index))[0]
        mean_nparray = []
        cov_nparray = []
        #############################################################
        # Generation of knn mean and cov
        for end_time in portfolio['pct_chg'].index[training_ids]:
            start_time = end_time-pd.Timedelta('2y')
            sub_factor_table = factors_rolling_normalized[start_time:end_time]
            sub_ret_table = portfolio['pct_chg'][start_time:end_time]
            factor0 = sub_factor_table.iloc[-1]
            norm_factor_minus_factor0 = pd.Series(np.linalg.norm(sub_factor_table.iloc[:-1] - factor0, axis = 1), 
                                                  index = sub_factor_table.index[:-1])
            k_nn_index = norm_factor_minus_factor0.nlargest(param_k_nn).index
            mean_nparray.append(portfolio['pct_chg'].loc[k_nn_index].mean().values)
            cov_nparray.append(portfolio['pct_chg'].loc[k_nn_index].cov().values)
        mean_nparray = np.row_stack(mean_nparray)
        cov_nparray = np.row_stack(cov_nparray)
        #############################################################
        # Invoke Solver
        theta_nparray = solvers.rolling.rolling_opt_framework(
            training_ids = range(mean_nparray.shape[0]), 
            mean_nparray = mean_nparray, 
            cov_nparray = cov_nparray, 
            solver = solver, 
            **solver_kwargs)
        theta_df = pd.DataFrame(data=theta_nparray, 
                                columns=portfolio['pct_chg'].columns, 
                                index=portfolio['pct_chg'].index[training_ids])
        if prev_theta_df is not None:
            theta_df = pd.concat([prev_theta_df[:str(last_available_year)], theta_df[exp_start_index:]])
        theta_df.to_pickle(file_path_fmt.format('stock_weights'))
        PnL = (theta_df * portfolio['pct_chg'].shift(-1)).sum(axis = 1)[training_ids]
        if prev_PnL_df is not None:
            PnL = pd.concat([prev_PnL_df[:str(last_available_year)], PnL[exp_start_index:]])
        PnL.to_pickle(file_path_fmt.format('PnL'))
        PnL_list.append(PnL)
    PnL_table = pd.DataFrame(PnL_list).transpose()
    return PnL_table

def resample_experiments_DR_conditional(exp_num, 
                                        sample_stock_num, 
                                        solver, 
                                        solver_name_prefix = '', 
                                        start_index = '2012', 
                                        end_index = '2017',
                                        enforce_exp_num = None,
                                        **solver_kwargs):
    SEED = 42
    factors_df = utils.datareader.load_fama_french_factors(normalized = True)
    ts_code_generator = np.random.RandomState(SEED)
    info_list = ['pct_chg']
    PnL_list = []
    for exp in tqdm(range(exp_num), desc = 'exp', leave=False):
        exp_start_index, exp_end_index = start_index, end_index
        prev_theta_df, prev_PnL_df = None, None
        portfolio = utils.preprocess.gen_portfolio_data_table(ts_code_generator, info_list, sample_stock_num)
        if enforce_exp_num is not None and exp != enforce_exp_num:
            continue
        file_path_fmt = utils.path.get_file_path_and_create_dir(exp, SEED, solver, solver_name_prefix, **solver_kwargs)
        if os.path.exists(file_path_fmt.format('stock_weights')):
            prev_theta_df = pd.read_pickle(file_path_fmt.format('stock_weights'))
            prev_PnL_df = pd.read_pickle(file_path_fmt.format('PnL'))
            last_available_year = prev_theta_df.index[prev_theta_df.abs().sum(axis = 1) > 1e-10][-1].year
            if last_available_year < int(end_index)-1:
                exp_start_index = str(last_available_year +1)
            else:
                continue
        training_ids = np.where((portfolio['pct_chg'].index >= start_index) & (portfolio['pct_chg'].index < end_index))[0]
        theta_array = []
        for end_time in tqdm(portfolio['pct_chg'].index[training_ids], desc = 'time', leave = False):
            #############################################################
            # Generation of return Y and covariate X
            #############################################################
            start_time = end_time-pd.Timedelta('2y')
            X_mat = factors_df.shift(1)[start_time:end_time].values
            Y_mat = portfolio['pct_chg'][start_time:end_time].values
            X_0 = factors_df.loc[end_time].values
            assert X_mat.shape[0] == Y_mat.shape[0]
            assert X_mat.shape[1] == X_0.shape[0]
            #############################################################
            # Invoke Solver
            #############################################################
            theta = solver(X_mat = X_mat, Y_mat = Y_mat, X0 = X_0, **solver_kwargs)
            theta_array.append(theta)
        theta_df = pd.DataFrame(data=np.row_stack(theta_array), 
                                columns=portfolio['pct_chg'].columns, 
                                index=portfolio['pct_chg'].index[training_ids])
        if prev_theta_df is not None:
            theta_df = pd.concat([prev_theta_df[:str(last_available_year)], theta_df[exp_start_index:]])
        theta_df.to_pickle(file_path_fmt.format('stock_weights'))
        PnL = (theta_df * portfolio['pct_chg'].shift(-1)).sum(axis = 1)[training_ids]
        if prev_PnL_df is not None:
            PnL = pd.concat([prev_PnL_df[:str(last_available_year)], PnL[exp_start_index:]])
        PnL.to_pickle(file_path_fmt.format('PnL'))
        PnL_list.append(PnL)
    PnL_table = pd.DataFrame(PnL_list).transpose()
    return PnL_table
            