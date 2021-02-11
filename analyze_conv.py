
import h5py
import numpy as np
import os
import general as gen
import matplotlib.pyplot as plt

def plot_beam(beam_name, antenna_orientation, phi):
    if beam_name[-3:] == 'out': # FEKO file
        r = gen.read_beam_FEKO(beam_name, antenna_orientation)
    elif beam_name[-3:] == 'ra1': # WIPLD
        r = gen.read_beam_WIPLD(beam_name, antenna_orientation)
    f = r[0]/1e6 # convert to MHz
    gain = r[-1]
    print(gain.shape)
    if gain.shape[-1] == 361:
        gain = gain[:, :, :-1] # cut last angle since 0 = 360 degrees
    print('Min frequency = {}'.format(f.min()))
    print('Max frequency = {}'.format(f.max()))
    print('Frequency shape = {}'.format(f.shape))
    plt.figure()
    #plt.imshow(gain[:, phi, 0:90], aspect='auto', extent=[0, 90, f.max(), f.min()])
    plt.imshow(gain[:, :, phi], aspect='auto', extent=[0, 90, f.max(), f.min()])
    plt.colorbar()
#    plt.draw()
    plt.show()

def new_read_hdf5(azimuth, varname, loc='mars'):
    ## NOT UPDATED
    # loc = 'Edges' or 'Mars', specifies latitude
    path = 'no_git_files/sky_models/map_az_el_lst/' + loc + '/save_parallel_convolution'
    with h5py.File(path, 'r') as hf:
        var = hf.get(varname)
        var_arr = np.array(var)
    idx = azimuth/180
    return var_arr[idx]

def read_hdf5(azimuth, varname, loc, ground_plane=True):
    if ground_plane:
        gpath = 'inf_metal_ground_plane/'
    else:
        gpath = 'no_ground_plane/'
    path = 'no_git_files/sky_models/blade_dipole/' + gpath + loc + '/save_parallel_convolution_' + str(azimuth)
    with h5py.File(path, 'r') as hf:
#        print([key for key in hf.keys()])
        var=hf.get(varname)
        var_arr = np.array(var)
    if loc == 'edges':
        return var_arr[0]
    else:
       return var_arr

def get_ftl(azimuth, loc='mars', ground_plane=True, new=False, return_fl=True, return_t=True):
    if not new:
        f = read_hdf5(azimuth, 'freq_out', loc=loc, ground_plane=ground_plane)
        t = read_hdf5(azimuth, 'ant_temp_out', loc=loc, ground_plane=ground_plane)
        lst = read_hdf5(azimuth, 'LST_out', loc=loc, ground_plane=ground_plane)
    else: ## not up to date
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

def plot_temp_3d(azimuth, save=False, loc='mars', ground_plane=True):
    freq_vector, temp_array, LST_vector = get_ftl(azimuth, loc=loc, ground_plane=ground_plane)   
    plt.figure()
    freq_min = freq_vector[0]
    freq_max = freq_vector[-1]
    LST_min = LST_vector[0]
    LST_max = LST_vector[-1]
    print(temp_array.shape)
    print(temp_array)
    print(freq_min)
    print(freq_max)
    print(LST_max)
    print(LST_min)
    plt.imshow(temp_array, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min])
    plt.title('Antenna Temperature \n' r'$\phi = %d$'%azimuth)
    plt.ylabel('LST')
    plt.xlabel('Frequency [MHz]')
    cbar = plt.colorbar()
    cbar.set_label("Antenna Teperature [K]")
    if save:
        if ground_plane:
            gpath = 'inf_metal_ground_plane/'
        else:
            gpath = 'no_ground_plane/'
        path = 'plots/' + gpath + loc + '/temp3d_plots'
        try:
            plt.savefig(path + '/temp' + str(azimuth))
        except:
            print("Couldn't save figure, check cwd and if file exists")
            print(path)
    else:
        plt.draw()

def plot_waterfalls_diff(azimuths=[0, 30, 60, 90, 120, 150], ref_azimuth=0, loc='mars', ground_plane=True, save=False):
    f0, t0, l0 = get_ftl(ref_azimuth, loc=loc, ground_plane=ground_plane)
    for phi in azimuths:
        if phi == ref_azimuth:
            plot_temp_3d(phi, save=save, loc=loc, ground_plane=ground_plane)
        else:
            f, t, l = get_ftl(phi, loc=loc, ground_plane=ground_plane)
            dt = t - t0
            assert f.all() == f0.all() and l.all() == l0.all(), "incompatible frequency/lst"
            plt.figure()
            freq_min = f[0]
            freq_max = f[-1]
            LST_min = l[0]
            LST_max = l[-1]
            plt.imshow(dt, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min])
            plt.title("Antenna Temperature at " r"$\phi=%d$" "\n" "Relative to Temperature at " r"$\phi=%d$" % (phi, ref_azimuth))
            plt.ylabel('LST')
            plt.xlabel('Frequency [MHz]')
            cbar = plt.colorbar()
            cbar.set_label(r"$T(\phi=%d) - T(\phi=%d)$ [K]" % (phi, ref_azimuth))
            if save:
                if ground_plane:
                    gpath = 'inf_metal_ground_plane/'
                else:
                    gpath = 'no_ground_plane/'
                plt.savefig('plots/' + gpath + loc + '/diff_plots/' + 'temp_'+str(phi)+'-'+str(ref_azimuth))
            else:
                plt.draw()
    if not save:
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

def plot_rms(phi, flow=50, fhigh=100, Nfg_split=3, save=False, loc='mars', ground_plane=True, frequency_normalization=100, noise_normalization=0.1, noise=False):
    f, t, lst = get_ftl(phi, loc=loc, ground_plane=ground_plane)
    rms_values = compute_rms(f, t, flow=flow, fhigh=fhigh, frequency_normalization=frequency_normalization, noise_normalization=noise_normalization, noise=noise)[0]
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
        if ground_plane:
            gpath = 'inf_metal_ground_plane/'
        else:
            gpath = 'no_ground_plane/'
        plt.savefig('plots/' + gpath + loc +'/rms_plots/rms'+str(phi))
    plt.show()

def plot_rms_comparison(azimuths=[0, 30, 60, 90, 120, 150], loc='mars', ground_plane=True, flow=50, fhigh=100, Nfg=5, save=False):
    f, l = get_ftl(0, loc=loc, ground_plane=ground_plane, return_t=False)
    print(f.shape)
    print(l.shape)
    plt.figure()
    for i, azimuth in enumerate(azimuths):
        t = get_ftl(azimuth, loc=loc, ground_plane=ground_plane, return_fl=False)
        print(t.shape)
        rms = compute_rms(f, t, flow, fhigh, Nfg_array = [Nfg])[0]
        plt.plot(l, rms, label=r'$\phi$ = {}'.format(azimuth))
    plt.legend()
    plt.xlabel('LST (hours)')
    plt.ylabel('RMS (Kelvin)')
    plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST for {:d}-term fit'.format(flow, fhigh, Nfg))
    plt.xlim(np.min(l)-0.5, np.max(l)+0.5)
    if save:
        if ground_plane:
            gpath = 'inf_metal_ground_plane/'
        else:
            gpath = 'no_ground_plane/'
        plt.savefig('plots/' + gpath + loc +'/rms_plots/rms_comparison')
    plt.show()

def plot_residuals(azimuth=0, lst_for_plot=[0, 6, 12, 18], flow=50, fhigh=100, loc='mars', ground_plane=True, save=False):
    # plots for five-term fit
    f, t, l = get_ftl(azimuth, loc=loc, ground_plane=ground_plane)
    res = compute_rms(f, t, flow=flow, fhigh=fhigh, Nfg_array=[5])[1]
    f = f[(f >= flow) & (f <= fhigh)]
    plt.figure()
    for lst in lst_for_plot:
        lst_point = 10 * lst # since resolution is .1 hours
        plt.plot(f, res[0, lst_point, :], label='LST = {} hrs'.format(lst))
    plt.xlabel(r'$\nu$ [MHz]')
    plt.ylabel('T [K]')
    title_str = r'Residuals vs Frequency for $\phi={}$'.format(azimuth)
    plt.title(title_str)
    plt.legend()
    if save:
        if ground_plane:
            gpath = 'inf_metal_ground_plane/'
        else:
            gpath = 'no_ground_plane/'
        plt.savefig('plots/' + gpath + loc + '/residuals_' + str(azimuth))
    plt.show()

def save_all_plots(loc='mars', ground_plane=True):
    azimuths = [0, 30, 60, 90, 120, 150]
    print('temp3d and rms')
    for phi in azimuths:
        plot_temp_3d(phi, loc=loc, ground_plane=ground_plane, save=True)
        plot_rms(phi, loc=loc, ground_plane=ground_plane, save=True)
    print('waterfalls_diff')
    plot_waterfalls_diff(loc=loc, ground_plane=ground_plane, save=True)
    print('rms comparison')
    plot_rms_comparison(loc=loc, ground_plane=ground_plane, save=True)
    print('residuals')
    plot_residuals(loc=loc, ground_plane=ground_plane, save=True)
