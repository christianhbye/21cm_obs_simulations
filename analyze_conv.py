import h5py
import numpy as np
import os
import general as gen
import matplotlib.pyplot as plt


def new_read_hdf5(azimuth, varname, loc='mars'):
    # loc = 'Edges' or 'Mars', specifies latitude
    path = 'no_git_files/sky_models/map_az_el_lst/' + loc + '/save_parallel_convolution'
    with h5py.File(path, 'r') as hf:
        var = hf.get(varname)
        var_arr = np.array(var)
    idx = azimuth/180
    return var_arr[idx]

def read_hdf5(azimuth, varname, loc):
    path = 'no_git_files/sky_models/map_az_el_lst/' + loc + '/save_parallel_convolution_' + str(azimuth)
    print(path)
    with h5py.File(path, 'r') as hf:
        print([key for key in hf.keys()])
        var=hf.get(varname)
        var_arr = np.array(var)
      #  print(var_arr.shape)
      #  print(var_arr[0].shape)
    return var_arr

def get_ftl(azimuth, loc='mars', new=False, return_fl=True, return_t=True):
    if not new:
        f = read_hdf5(azimuth, 'freq_out', loc=loc)
        t = read_hdf5(azimuth, 'ant_temp_out', loc=loc)
        lst = read_hdf5(azimuth, 'LST_out', loc=loc)
    else:
        f = new_read_hdf5(azimuth, 'freq_out', loc)
        t = new_read_hdf5(azimuth, 'ant_temp_out', loc)
        lst = new_read_hdf5(azimuth, 'LST_out', loc)
    if return_fl and return_t:
        return f, t, lst
    elif return_fl and not return_t:
        return f, lst
    elif not return_fl and return_t:
        return t
    else:
        print('Nothing returned! Change kwargs return_fl and and return_t!')
        return None

def plot_temp(freq_vector, temp_array, LST_vec, LST_idxs, azimuth, save=False, loc='mars'):
    '''
    Plot antenna temperature vs frequency at given LST
    '''
    plt.figure()
    for LST in LST_idxs:
        LST_val = round(LST_vec[LST], 7)
        lab = 'LST = ' + str(LST_val)
        plt.plot(freq_vector, temp_array[LST, :], label=lab)
    plt.xlabel('Frequency')
    plt.ylabel('Antenna temperature')
    plt.title('Temperature vs Frequency \n' r'$\phi = %d$'%azimuth)
    plt.legend()
    if save:
        plt.savefig('plots/' + loc + '/tvf_'+str(azimuth))
    else:
        plt.show()

def plot_temp_3d(freq_vector, temp_array, LST_vector, azimuth, save=False, loc='mars'):
    plt.figure()
    freq_min = freq_vector[0]
    freq_max = freq_vector[-1]
    LST_min = LST_vector[0]
    LST_max = LST_vector[-1]
    plt.imshow(temp_array, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min])
    plt.title('Antenna Temperature \n' r'$\phi = %d$'%azimuth)
    plt.ylabel('LST')
    plt.xlabel('Frequency (MHz)')
    plt.colorbar()
    if save:
        plt.savefig('plots/' + loc +'/temp'+str(azimuth))
    else:
        plt.show()

def compute_rms(f, t, flow, fhigh, Nfg_array=[1, 2, 3, 4, 5, 6], frequency_normalization=100, noise_normalization=0.1, noise=False):
    frequency_vector = f[(f >= flow) & (f <= fhigh)]
    temp_array = t[:, (f>=flow) & (f<=fhigh)]
    rms_values = np.empty((len(temp_array), len(Nfg_array)))
    residuals = np.empty((len(Nfg_array), len(temp_array[:, 0]), len(temp_array[0, :])))
    for j, Nfg in enumerate(Nfg_array):
        for i in range(len(temp_array)):
            temperature_vector = temp_array[i, :]
            if not noise:
                standard_deviation_vector = np.ones(len(frequency_vector)) # no noise
            else:
                standard_deviation_vector = noise_normalization * temperature_vector/temperature_vector[frequency_vector == frequency_normalization]
            p = gen.fit_polynomial_fourier('LINLOG', frequency_vector/frequency_normalization, temperature_vector, int(Nfg), Weights=1/(standard_deviation_vector**2))
            m = gen.model_evaluate('LINLOG', p[0], frequency_vector/frequency_normalization)
            r = temperature_vector - m
            residuals[j, i, :] = r
            # compute rms of residuals
            rms = np.sqrt(np.sum(r**2)/len(r))
            rms_values[i, j] = rms

    return rms_values, residuals

def plot_rms(lst, rms_values, phi, flow=50, fhigh=100, Nfg_split=3, save=False, loc='mars'):
    plt.figure()
    plt.plot(lst, rms_values[:, :Nfg_split]) # 3 parameters
    leg_v = np.arange(Nfg_split)
    leg = [str(n+1) for n in leg_v]
    l1 = plt.legend(leg, title='Number of parameters:')
    l1._legend_box.align = 'left'
    plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST at $\phi = {}$'.format(flow, fhigh, phi))
    plt.xlabel('LST (hours)')
    plt.ylabel('RMS (Kelvin)')
    plt.xlim(np.min(lst)-.5, np.max(lst)+.5)
    plt.show()
    plt.figure()
    plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST at $\phi = {}$'.format(flow, fhigh, phi))
    plt.xlabel('LST (hours)')
    plt.ylabel('RMS (Kelvin)')
    plt.xlim(np.min(lst)-.5, np.max(lst)+.5)
    plt.plot(lst, rms_values[:, Nfg_split:])
    leg_v = np.arange(Nfg_split, rms_values.shape[-1])
    leg = [str(n+1) for n in leg_v]
    l2 = plt.legend(leg, title='Number of parameters:')
    l2._legend_box.align = 'left'
    if save:
        plt.savefig('plots/' + loc +'/rms_plots/rms'+str(phi))
    plt.show()

def plot_rms_comparision(azimuths, loc='mars', flow=50, fhigh=100, Nfg=5, save=False):
    f, l = get_ftl(0, loc=loc, return_t=False)
    print(f.shape)
    print(l.shape)
    plt.figure()
    for i, azimuth in enumerate(azimuths):
        t = get_ftl(azimuth, loc=loc, return_fl=False)
        print(t.shape)
        rms = compute_rms(f, t, flow, fhigh, Nfg_array = [Nfg])[0]
        plt.plot(l, rms, label=r'$\phi$ = {}'.format(azimuth))
    plt.legend()
    plt.xlabel('LST (hours)')
    plt.ylabel('RMS (Kelvin)')
    plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST for {:d}-term fit'.format(flow, fhigh, Nfg))
    plt.xlim(np.min(l)-0.5, np.max(l)+0.5)
    if save:
        plt.savefig('plots/' + loc +'/rms_comparison'+str(azimuth))
    plt.show()
