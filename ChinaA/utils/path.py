import pathlib
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def remove_solver_name_suffix(name, suffix):
    return name[:-len(suffix)] if name.endswith(suffix) else name

def get_file_path_and_create_dir(exp, seed, solver, solver_name_prefix = '', **solver_kwargs):
    local_path = (solver_name_prefix 
                  + remove_solver_name_suffix(solver.__name__, '_new')
                  + '/' 
                  + '/'.join(['{}={}'.format(key,val) for key,val in sorted(solver_kwargs.items())]))
    full_dir_path = project_path+"/portfolio_res/"+local_path+'/'
    file_name = '{}'+'_seed={}_exp={}.pkl'.format(seed,exp)
    pathlib.Path(full_dir_path).mkdir(parents=True, exist_ok=True)
    return full_dir_path+file_name