import h5py
import numpy as np
import general as gen
import matplotlib.pyplot as plt

def read_hdf5(fname, varname):
    path = 'no_git_files/sky_models/map_az_el_lst/' + fname
    with h5py.File(path, 'r') as hf:
        var = hf.get(varname)
    var_arr = np.array(var)
    return var_arr

def plot_temp(freq_vector, temp_array, LST):
    '''
    Plot antenna temperature vs frequency at given LST
    '''
    plt.figure()
    plt.plot(freq_vector, temp_array[LST ])
    plt.show()

def plot_temp_3d(freq_vector, temp_array, LST_vector):
    plt.figure()
    plt.imshow(temp_array, extent=)
    plt.show()

def compute_rms(freq_vector, temp_array, Nfg_array=[1, 2, 3, 4, 5], frequency_normalization=100, noise_normalization=0.1, noise=False):
    rms_values = np.empty((len(temp_array), len(Nfg_array)))

    for Nfg in Nfg_array:
        for i in range(len(temp_array)):
            temperature_vector = temp_array[i, :]
            if not noise:
                standard_deviation_vector = np.ones(len(frequency_vector)) # no noise
            else:
                standard_deviation_vector = noise_normalization * temperature_vector/temperature_vector[frequency_vector == frequency_normalization]
            p = gen.fit_polynomial_fourier('LINLOG', frequency_vector/frequency_normalization, temperature_vector, Nfg, Weights=1/(standard_deviation_vector**2))
            m = gen.model_evaluate('LINLOG', p[0], frequency_vector/frequency_normalization)
            r = temperature_vector - m

            # compute rms of r
            rms = np.sqrt(np.sum(r**2)/len(r))
            rms_values[i, j] = rms

    return rms_values

def plot_rms(rms_values, Nfg_split=3):
    plt.figure()
    plt.plot(rms_values[:, :Nfg_split]) # 3 parameters
    plt.xlim(0, 24)
    plt.show()
    plt.figure()
    plt.plot(rms_values[:, Nfg_split:])
    plt.show()
