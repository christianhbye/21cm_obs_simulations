import h5py
import numpy as np
import general as gen
import matplotlib.pyplot as plt


def read_hdf5(azimuth, varname):
    path = 'no_git_files/sky_models/map_az_el_lst/save_parallel_convolution_' + str(azimuth)
    with h5py.File(path, 'r') as hf:
        var = hf.get(varname)
        var_arr = np.array(var)
    return var_arr

def get_ftl(azimuth):
    f = read_hdf5(azimuth, 'freq_out')
    t = read_hdf5(azimuth, 'ant_temp_out')
    lst = read_hdf5(azimuth, 'LST_out')
    return f, t, lst

def plot_temp(freq_vector, temp_array, LST_vec, LST_idxs, azimuth, save=False):
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
        plt.savefig('no_git_files/plots/tvf_'+str(azimuth))
    else:
        plt.show()

def plot_temp_3d(freq_vector, temp_array, LST_vector, azimuth, save=False):
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
        plt.savefig('no_git_files/plots/tvf3d_'+str(azimuth))
    else:
        plt.show()

def compute_rms(frequency_vector, temp_array, Nfg_array=[1, 2, 3, 4, 5], frequency_normalization=100, noise_normalization=0.1, noise=False):
    rms_values = np.empty((len(temp_array), len(Nfg_array)))

    for j, Nfg in enumerate(Nfg_array):
        for i in range(len(temp_array)):
            temperature_vector = temp_array[i, :]
            if not noise:
                standard_deviation_vector = np.ones(len(frequency_vector)) # no noise
            else:
                standard_deviation_vector = noise_normalization * temperature_vector/temperature_vector[frequency_vector == frequency_normalization]
            print(type(len(frequency_vector/frequency_normalization)))
            print(type(int(Nfg)))
            p = gen.fit_polynomial_fourier('LINLOG', frequency_vector/frequency_normalization, temperature_vector, int(Nfg), Weights=1/(standard_deviation_vector**2))
            m = gen.model_evaluate('LINLOG', p[0], frequency_vector/frequency_normalization)
            r = temperature_vector - m

            # compute rms of r
            rms = np.sqrt(np.sum(r**2)/len(r))
            rms_values[i, j] = rms

    return rms_values

def plot_rms(rms_values, Nfg_split=3):
    plt.figure()
    plt.plot(rms_values[:, :Nfg_split]) # 3 parameters
    plt.show()
    plt.figure()
    plt.plot(rms_values[:, Nfg_split:])
    plt.show()
