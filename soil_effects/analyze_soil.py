import h5py
import numpy as np

def read_hdf5(sigma, varname):
    path = 'save_parallel_convolution_0_' + str(int(100*sigma))
    with h5py.File(path, 'r') as hf:
        var=hf.get(varname)
        var_arr = np.array(var)
    if var_arr.shape[0] == 1:
       if len(var_arr.shape) == 2:
           return var_arr[0, :]
       elif len(var_arr.shape) == 3:
           return var_arr[0, :, :]
       else:
           print('read hdf5 fcn, shape of var is')
           print(var_arr.shape)
           return var_arr
    else:
        return var_arr

def get_ftl(sigma, return_fl=True, return_t=True):
    f = read_hdf5(sigma, 'freq_out')
    t = read_hdf5(sigma, 'ant_temp_out')
    lst = read_hdf5(sigma, 'LST_out')
    if return_fl and return_t:
        return f, t, lst
    elif return_fl and not return_t:
        return f, lst
    elif not return_fl and return_t:
        return t
    else:
        print('Nothing returned! Change kwargs return_fl and and return_t!')
        return None
