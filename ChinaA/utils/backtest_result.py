import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.path
import itertools
import seaborn as sns

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#plt.rcParams['figure.figsize'] = (12,8)
#plt.rcParams['font.size'] = 15

class ModelConfig(object):
    'Class to wrap all the information for a model in order to facilitate the access to saved result'
    def __init__(self, model_name, solver, solver_name_prefix='', **solver_kwargs):
        self.model_name = model_name
        self.solver = solver
        self.solver_name_prefix = solver_name_prefix
        self.solver_kwargs = solver_kwargs
        
def read_single_solver_backtest_result(exp_num, seed, model_config):
    solver, solver_name_prefix, solver_kwargs = model_config.solver, model_config.solver_name_prefix, model_config.solver_kwargs
    PnL_list = []
    for exp in range(exp_num):
        file_fmt = utils.path.get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix, **solver_kwargs)
        PnL_list.append(pd.read_pickle(file_fmt.format('PnL')))
    PnL_table = pd.DataFrame(PnL_list).transpose()
    return PnL_table

def read_single_solver_stock_weights_result(exp_num, seed, model_config):
    solver, solver_name_prefix, solver_kwargs = model_config.solver, model_config.solver_name_prefix, model_config.solver_kwargs
    stock_weight_list = []
    for exp in range(exp_num):
        file_fmt = utils.path.get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix, **solver_kwargs)
        stock_weight_list.append(pd.read_pickle(file_fmt.format('stock_weights')))
    return stock_weight_list

def plot_backtest_sharpe_hist(PnL_table_list, legend_str_list):
    for PnL_table in PnL_table_list:
        sharpe = np.sqrt(252) * PnL_table.mean() / PnL_table.std()
        plt.hist(sharpe, alpha = 0.5, density = True, bins = int((sharpe.max()-sharpe.min()) / 0.05))
    plt.title('histogram of annualized Sharpe ratio')
    plt.xlabel('Sharpe ratio')
    plt.ylabel('sample frequency')
    plt.legend(legend_str_list)
    
def plot_backtest_sharpe_box(PnL_table_list, legend_str_list):
    sharpe_tb_summary = []
    for PnL_table, model_name in zip(PnL_table_list, legend_str_list):
        sharpe = np.sqrt(252) * PnL_table.mean() / PnL_table.std()
        sharpe_tb = pd.DataFrame({'Sharpe Ratio': sharpe})
        sharpe_tb['Model'] = model_name
        sharpe_tb_summary.append(sharpe_tb)
    sharpe_tb_summary = pd.concat(sharpe_tb_summary)
    sns.boxplot(x="Model", y="Sharpe Ratio", data=sharpe_tb_summary,
                whis=[5, 95], width=.6, palette="bright")
    plt.savefig('figures/Chinese_sharpe_box_plot.pdf')
    
def plot_backtest_PnL_mean_trajectorys(PnL_table_list, legend_str_list):
    fig, ax = plt.subplots()
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_major_formatter(myFmt)
    for PnL_table in PnL_table_list:
        cum_sum = PnL_table.mean(axis = 1).cumsum()
        ax.plot(cum_sum, alpha = 0.9, linewidth = 1)
    plt.title('Path of Cumulative Return')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(legend_str_list)
    plt.savefig('figures/Chinese_trajectorys.pdf')
    
def plot_backtest_PnL_box(PnL_table_list, legend_str_list):
    PnL_tb_summary = []
    for PnL_table, model_name in zip(PnL_table_list, legend_str_list):
        PnL_tb = pd.DataFrame({'Daily Return (%)': PnL_table.quantile(q = 0.05, axis = 0)})
        PnL_tb['Model'] = model_name
        PnL_tb_summary.append(PnL_tb)
    PnL_tb_summary = pd.concat(PnL_tb_summary)
    sns.boxplot(x="Model", y="Daily Return (%)", data=PnL_tb_summary,
                whis=[5, 95], width=.6, palette="bright")
    plt.savefig('figures/Chinese_return_quantile_box_plot.pdf')
    
def plot_backtest_PnL_quantile_trajectorys(PnL_table_list, legend_str_list):
    fig, ax = plt.subplots()
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_major_formatter(myFmt)
    for PnL_table in PnL_table_list:
        cum_sum = PnL_table.cumsum(axis = 0).quantile(q = 0.5, axis = 1)
        ax.plot(cum_sum, alpha = 0.9, linewidth = 1)
    plt.title('Path of Cumulative Return')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(legend_str_list)
    plt.savefig('figures/Chinese_trajectorys_quantiles.pdf')
    
def print_mean_backtest_sharpe(PnL_table_list, legend_str_list):
    print("Sharpe Ratio:")
    for model_name, PnL_table in zip(legend_str_list, PnL_table_list):
        annualized_sharpe = np.sqrt(252)*PnL_table.mean() / PnL_table.std()
        print('{model_name:50}\tmean:{mean:.3f}\tstd:{std:.3f}'.format(
            model_name=model_name,
            mean=annualized_sharpe.mean(),
            std=annualized_sharpe.std()))

def print_PnL_stats(PnL_table_list, legend_str_list):
    print("PnL stats:")
    for model_name, PnL_table in zip(legend_str_list, PnL_table_list):
        print('{model_name:50}\tmean:{mean:.3f}\tstd:{std:.3f}'.format(
            model_name=model_name,
            mean=PnL_table.values.mean(),
            std=PnL_table.values.std()))

def print_weight_sparsity(stock_weights_list, legend_str_list, threshold = 0.01):
    print("stock weights sparsity:")
    for model_name, stock_weights in zip(legend_str_list, stock_weights_list):
        sparsity_table = pd.DataFrame({exp: (stock_weight > threshold).sum(axis = 1) 
            for exp, stock_weight in enumerate(stock_weights)})
        print('{model_name:50}\tmean:{mean:.3f}\tstd:{std:.3f}'.format(
            model_name=model_name,
            mean=sparsity_table.values.mean(),
            std=sparsity_table.values.std()))

def print_turnover_rate(stock_weights_list, legend_str_list):
    print("turnover rate:")
    for model_name, stock_weights in zip(legend_str_list, stock_weights_list):
        turnover_rate_func = lambda table: (np.abs(table - table.shift(-1)).iloc[:-1].sum(axis = 1)/2)
        sparsity_table = pd.DataFrame({exp: turnover_rate_func(stock_weight) 
            for exp, stock_weight in enumerate(stock_weights)})
        print('{model_name:50}\tmean:{mean:.3f}\tstd:{std:.3f}'.format(
            model_name=model_name,
            mean=sparsity_table.values.mean(),
            std=sparsity_table.values.std()))

def plot_backtest_result(exp_num, seed, model_config_list):
    PnL_table_list = [read_single_solver_backtest_result(exp_num, seed, model_config)
                      for model_config in model_config_list]
    legend_str_list = [model_config.model_name for model_config in model_config_list]
    plot_backtest_sharpe_box(PnL_table_list, legend_str_list)
    plt.show()
    plot_backtest_PnL_mean_trajectorys(PnL_table_list, legend_str_list)
    plt.show()
    print('='*80)
    print_mean_backtest_sharpe(PnL_table_list, legend_str_list)
    print('='*80)
    print_PnL_stats(PnL_table_list, legend_str_list)
    print('='*80)
    
    
def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))
        
def get_solver_kwargs_name_list(CV_params_dict):
    mapping = {
        'equal_weight': [],
        'mean_variance_long_only_opt_mosek_kernel': ['reg_params'],
        'DR_mean_variance_long_only_opt_mosek_kernel_p2': ['delta', 'p', 'reg_params'],
        'DR_Winfty_conditional_mean_variance_long_only_opt_cvx_kernel': ['reg_params', 'gamma_quantile', 'rho_quantile'],
        'DR_W2_conditional_mean_variance_long_only_opt_cvx_kernel': ['reg_params', 'epsilon', 'rho_div_rho_min'],
        'DR_W2_conditional_mean_variance_long_only_opt_cvx_kernel_new': ['reg_params', 'epsilon', 'rho_div_rho_min'],
    }
    return mapping[CV_params_dict['solver'][0].__name__]
         
def read_single_solver_backtest_cross_validation_result(exp_num, 
                                                        seed,
                                                        CV_params_dict,
                                                        validation_start_index='2012', 
                                                        validation_end_index = '2013',
                                                        test_start_index='2014',
                                                        test_end_index='2017'):
    test_PnL_list = []
    solver_kwargs_name_list = get_solver_kwargs_name_list(CV_params_dict)
    for exp in range(exp_num):
        PnL_list = []
        for params in dict_product(CV_params_dict):
            solver = params['solver']
            solver_name_prefix = params['solver_name_prefix'] if 'solver_name_prefix' in params else ''
            solver_kwargs = {name: params[name] for name in solver_kwargs_name_list}
            file_fmt = utils.path.get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix, **solver_kwargs)
            PnL_list.append(pd.read_pickle(file_fmt.format('PnL')))
        PnL_table = pd.DataFrame(PnL_list).transpose()[validation_start_index:validation_end_index]
        best_model = list(dict_product(CV_params_dict))[(PnL_table.mean() / PnL_table.std()).argmax()]
        best_model['start_index'] = test_start_index
        best_model['end_index'] = test_end_index
        best_model['enforce_exp_num'] = exp
        #solver, solver_name_prefix = best_model['solver'], best_model['solver_name_prefix']
        solver = best_model['solver']
        solver_name_prefix = best_model['solver_name_prefix'] if 'solver_name_prefix' in best_model else ''
        solver_kwargs = {name: best_model[name] for name in solver_kwargs_name_list}
        file_fmt = utils.path.get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix, **solver_kwargs)
        test_PnL_list.append(pd.read_pickle(file_fmt.format('PnL')))
    test_PnL_table = pd.DataFrame(test_PnL_list).transpose()[test_start_index:test_end_index]
    return test_PnL_table

def read_single_solver_stock_weights_cross_validation_result(exp_num, 
                                                            seed,
                                                            CV_params_dict,
                                                            validation_start_index='2012', 
                                                            validation_end_index = '2013',
                                                            test_start_index='2014',
                                                            test_end_index='2017'):
    test_stock_weight_list = []
    solver_kwargs_name_list = get_solver_kwargs_name_list(CV_params_dict)
    for exp in range(exp_num):
        PnL_list = []
        for params in dict_product(CV_params_dict):
            solver = params['solver']
            solver_name_prefix = params['solver_name_prefix'] if 'solver_name_prefix' in params else ''
            solver_kwargs = {name: params[name] for name in solver_kwargs_name_list}
            file_fmt = utils.path.get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix, **solver_kwargs)
            PnL_list.append(pd.read_pickle(file_fmt.format('PnL')))
        PnL_table = pd.DataFrame(PnL_list).transpose()[validation_start_index:validation_end_index]
        best_model = list(dict_product(CV_params_dict))[(PnL_table.mean() / PnL_table.std()).argmax()]
        best_model['start_index'] = test_start_index
        best_model['end_index'] = test_end_index
        best_model['enforce_exp_num'] = exp
        #solver, solver_name_prefix = best_model['solver'], best_model['solver_name_prefix']
        solver = best_model['solver']
        solver_name_prefix = best_model['solver_name_prefix'] if 'solver_name_prefix' in best_model else ''
        solver_kwargs = {name: best_model[name] for name in solver_kwargs_name_list}
        file_fmt = utils.path.get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix, **solver_kwargs)
        test_stock_weight_list.append(pd.read_pickle(file_fmt.format('stock_weights')).loc[test_start_index:test_end_index])
    return test_stock_weight_list

def plot_backtest_cross_validation_result(exp_num, seed, CV_params_dict_list):
    PnL_table_list = [read_single_solver_backtest_cross_validation_result(exp_num, seed, CV_params_dict)
                      for CV_params_dict in CV_params_dict_list]
    legend_str_list = [CV_params_dict['model_name'][0] for CV_params_dict in CV_params_dict_list]
    plot_backtest_sharpe_box(PnL_table_list, legend_str_list)
    plt.show()
    plot_backtest_PnL_box(PnL_table_list, legend_str_list)
    plt.show()
    plot_backtest_PnL_mean_trajectorys(PnL_table_list, legend_str_list)
    plt.show()
    plot_backtest_PnL_quantile_trajectorys(PnL_table_list, legend_str_list)
    plt.show()
    
    print('='*80)
    print_mean_backtest_sharpe(PnL_table_list, legend_str_list)
    print('='*80)
    print_PnL_stats(PnL_table_list, legend_str_list)
    print('='*80)    
    stock_weights_list = [read_single_solver_stock_weights_cross_validation_result(exp_num, seed, CV_params_dict)
                      for CV_params_dict in CV_params_dict_list]
    
    print_weight_sparsity(stock_weights_list, legend_str_list, threshold = 0.01)
    print('='*80)
    print_turnover_rate(stock_weights_list, legend_str_list)
    print('='*80)
    
    
    