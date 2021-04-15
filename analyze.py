import h5py
import numpy as np
from scipy.signal.windows import gaussian
import os
import general as gen
import matplotlib.pyplot as plt

def read_hdf5(azimuth, varname, loc, sweep_lat=None, ground_plane=True, simulation='edges_hb'):
    parent_path = '/scratch/s/sievers/cbye/21cm_obs_simulations/'
    if ground_plane:
        g1 = 'inf_metal_ground_plane/'
        if simulation == 'edges_hb':
            g2 = 'EDGES_highband/'
        elif simulation == 'edges_lb':
            g2 = 'EDGES_lowband/'
        elif simulation == 'FEKO':
            g2 = 'FEKO_simulation/' 
        gpath = g1 + g2
    else:
        gpath = 'no_ground_plane/'
    if loc == 'sweep':
        path = 'sweep/sky_models/blade_dipole/' + gpath + 'sweep/lat_' + str(sweep_lat) + '/save_parallel_convolution_' + str(azimuth)
    else:
        path = 'no_git_files/sky_models/blade_dipole/' + gpath + loc + '/save_parallel_convolution_' + str(azimuth)
    with h5py.File(parent_path + path, 'r') as hf:
#        print([key for key in hf.keys()])
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

def get_ftl(azimuth, loc='mars', sweep_lat=None, ground_plane=False, simulation='FEKO', return_fl=True, return_t=True):
    f = read_hdf5(azimuth, 'freq_out', loc=loc, sweep_lat=sweep_lat, ground_plane=ground_plane, simulation=simulation)
    t = read_hdf5(azimuth, 'ant_temp_out', loc=loc, sweep_lat=sweep_lat, ground_plane=ground_plane, simulation=simulation)
    lst = read_hdf5(azimuth, 'LST_out', loc=loc, sweep_lat=sweep_lat, ground_plane=ground_plane, simulation=simulation)
    if return_fl and return_t:
        return f, t, lst
    elif return_fl and not return_t:
        return f, lst
    elif not return_fl and return_t:
        return t
    else:
        print('Nothing returned! Change kwargs return_fl and and return_t!')
        return None

def plot_beam(antenna_type, antenna_orientation, phi, gain, frequency, climbeam=None, climderiv=None, savepath=None):
    if gain.shape[-1] == 361:
        gain = gain[:, :, :-1] # cut last angle since 0 = 360 degrees
    print('Min frequency = {}'.format(frequency.min()))
    print('Max frequency = {}'.format(frequency.max()))
    if frequency.min() > 1e6:
        frequency /= 1e6 # to MHz
        print('Convert frequency')
        print('Min frequency = {}'.format(frequency.min()))
        print('Max frequency = {}'.format(frequency.max()))
    print('Frequency shape = {}'.format(frequency.shape))
    plt.figure()
    plt.imshow(gain[:, :, phi], aspect='auto', extent=[0, 90, frequency.max(), frequency.min()])
    if climbeam:
        plt.clim(climbeam)
    plt.colorbar()
    plt.title(r'Gain ($\phi = {}$, $\psi_0 = {}$)'.format(phi, antenna_orientation) +'\n' + antenna_type)
    plt.xlabel(r'$\theta$ [deg]')
    plt.ylabel(r'$\nu$ [MHz]')
    if savepath:
        sp = 'plots/' + savepath + '/beam'
        plt.savefig(sp)
    dG = gain[1:, :, phi] - gain[:-1, :, phi]
    diff = frequency[1:] - frequency[:-1]
    assert diff.all() == frequency[1] - frequency[0], 'not constant frequency spacing'
    df = frequency[1] - frequency[0]
    derivative = dG/df
    plt.figure()
    plt.imshow(derivative, aspect='auto', extent=[0, 90, frequency.max(), frequency.min()])
    plt.xlabel(r'$\theta$ [deg]')
    plt.ylabel(r'$\nu$ [MHz]')
    if climderiv:
        plt.clim(climderiv)
    plt.colorbar()
    plt.title(r'Derivative ($\phi = {}$)'.format(phi) +'\n'+ beam_name)
    if savepath:
        sp = 'plots/' + savepath + '/deriv'
        plt.savefig(sp)

def plot_temp(freq_vector, temp_array, LST_vec, LST_idxs, azimuth, savepath=None):
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
    plt.title('Temperature vs Frequency \n' r'$\psi_0 = %d$'%azimuth)
    plt.legend()
    if savepath:
        plt.savefig('plots/' + savepath + '/temp')

def plot_temp_3d(freq_vector, temp_array, lst_vector, psi0, clim=None, savepath=None):   
    plt.figure()
    if freq_vector[0] > 1e6:
        freq_vector /= 1e6 # convert to MHz
    freq_min = freq_vector[0]
    freq_max = freq_vector[-1]
    LST_min = lst_vector[0]
    LST_max = lst_vector[-1]
    plt.imshow(temp_array, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min])
    plt.title('Antenna Temperature \n' r'$\psi_0 = {}$'.format(psi0))
    plt.ylabel('LST')
    plt.xlabel('Frequency [MHz]')
    if clim:
        plt.clim(clim)
    cbar = plt.colorbar()
    cbar.set_label("Antenna Temperature [K]")
    if savepath:
        sp = 'plots/' + savepath + '/temp3d'
        plt.savefig(sp)

def plot_waterfalls_diff(f, t, l, ref_t, psi0, ref_psi0, clim=None, savepath=None):
    dt = t - ref_t
    if f[0] > 1e6:
        f /= 1e6
    plt.figure()
    freq_min = f[0]
    freq_max = f[-1]
    LST_min = l[0]
    LST_max = l[-1]
    plt.imshow(dt, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min])
    plt.title("Antenna Temperature at " r"$\psi_0=%d$" "\n" "Relative to Temperature at " r"$\psi_0=%d$" % (psi0, ref_psi0))
    plt.ylabel('LST [hr]')
    plt.xlabel(r'$\nu$ [MHz]')
    cbar = plt.colorbar()
    cbar.set_label(r"$T(\phi=%d) - T(\phi=%d)$ [K]" % (phi, ref_azimuth))
    if clim:
        plt.clim(clim)
    if savepath:
        sp = 'plots/' + savepath + '/diff_' + str(psi0) + '_' + str(ref_psi0)
        plt.savefig(sp)

def compute_rms(f, t, flow, fhigh, Nfg_array=[1, 2, 3, 4, 5, 6], frequency_normalization=100, noise_normalization=0.1, noise=False, model_type='LINLOG'):
    frequency_vector = f[(f >= flow) & (f <= fhigh)]
    if len(t.shape) == 2:
        temp_array = t[:, (f>=flow) & (f<=fhigh)]
    else:
        temp_array = t[(f>=flow) & (f<=fhigh)]
        temp_array = np.expand_dims(temp_array, axis=0)
    rms_values = np.empty((len(temp_array), len(Nfg_array)))
    residuals = np.empty((len(Nfg_array), len(temp_array[:, 0]), len(temp_array[0, :])))
    for j, Nfg in enumerate(Nfg_array):
        for i in range(len(temp_array)):
            temperature_vector = temp_array[i, :]
            if not noise:
                standard_deviation_vector = np.ones(len(frequency_vector)) # no noise
            else:
                standard_deviation_vector = noise_normalization * temperature_vector/temperature_vector[frequency_vector == frequency_normalization]
            p = gen.fit_polynomial_fourier(model_type, frequency_vector/frequency_normalization, temperature_vector, int(Nfg), Weights=1/(standard_deviation_vector**2))
            m = gen.model_evaluate(model_type, p[0], frequency_vector/frequency_normalization)
            r = temperature_vector - m
            residuals[j, i, :] = r
            # compute rms of residuals
            rms = np.sqrt(np.sum(r**2)/len(r))
            rms_values[i, j] = rms
    return rms_values, residuals

def plot_rms(f, t, lst, flow=50, fhigh=100, Nfg_array=[1, 2, 3, 4, 5, 6], Nfg_split=3, save=False, loc='mars', ground_plane=True, simulation='edges_hb', frequency_normalization=100, noise_normalization=0.1, noise=False, model_type='LINLOG'):
    if type(psi) == int:
        f, t, lst = get_ftl(psi, loc=loc, ground_plane=ground_plane, simulation=simulation)
    rms_values = compute_rms(f, t, flow=flow, fhigh=fhigh, frequency_normalization=frequency_normalization, noise_normalization=noise_normalization, noise=noise, model_type=model_type)[0]
    plt.figure()
    plt.plot(lst, rms_values[:, :Nfg_split]) # 3 parameters
    leg_v = Nfg_array[:Nfg_split]
    leg = [str(n) for n in leg_v]
    l1 = plt.legend(leg, title='Number of parameters:')
    l1._legend_box.align = 'left'
    plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST at $\psi = {}$'.format(flow, fhigh, psi))
    plt.xlabel('LST (hours)')
    plt.ylabel('RMS (Kelvin)')
    plt.xlim(np.min(lst)-.5, np.max(lst)+.5)
    plt.show()
    if Nfg_split > len(Nfg_array):
        plt.figure()
        plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST at $\psi = {}$'.format(flow, fhigh, psi))
        plt.xlabel('LST (hours)')
        plt.ylabel('RMS (Kelvin)')
        plt.xlim(np.min(lst)-.5, np.max(lst)+.5)
        plt.plot(lst, rms_values[:, Nfg_split:])
        leg_v = Nfg_array[:Nfg_split]
        leg = [str(n) for n in leg_v]
        l2 = plt.legend(leg, title='Number of parameters:')
        l2._legend_box.align = 'left'
    if save:
        if ground_plane:
            g1 = 'inf_metal_ground_plane/'
            if simulation == 'edges_hb':
                g2 = 'EDGES_highband/'
            elif simulation == 'edges_lb':
                g2 = 'EDGES_lowband/'
            elif simulation == 'FEKO':
                g2 = 'FEKO_simulation' 
            gpath = g1 + g2
        else:
            gpath = 'no_ground_plane/'
        plt.savefig('plots/' + gpath + loc +'/rms_plots/rms'+str(phi))
    if Nfg_split > len(Nfg_array):
        plt.show()


def plot_rms_comparison(azimuths=[0, 30, 60, 90, 120, 150], loc='mars', ground_plane=True, simulation='edges_hb', flow=50, fhigh=100, model_type='LINLOG', Nfg=5, save=False):
    f, l = get_ftl(0, loc=loc, ground_plane=ground_plane, simulation=simulation, return_t=False)
    if f[0] >= 100: ## high band
        flow = 100
        fhigh = 190
    print(f.shape)
    print(l.shape)
    plt.figure()
    for i, azimuth in enumerate(azimuths):
        t = get_ftl(azimuth, loc=loc, ground_plane=ground_plane, simulation=simulation, return_fl=False)
        print(t.shape)
        rms = compute_rms(f, t, flow, fhigh, Nfg_array = [Nfg], model_type=model_type)[0]
        plt.plot(l, rms, label=r'$\phi$ = {}'.format(azimuth))
    plt.legend()
    plt.xlabel('LST (hours)')
    plt.ylabel('RMS (Kelvin)')
    plt.title(r'RMS (${} \leq \nu \leq {}$) vs LST for {:d}-term fit'.format(flow, fhigh, Nfg))
    plt.xlim(np.min(l)-0.5, np.max(l)+0.5)
    if save:
        if ground_plane:
            g1 = 'inf_metal_ground_plane/'
            if simulation == 'edges_hb':
                g2 = 'EDGES_highband/'
            elif simulation == 'edges_lb':
                g2 = 'EDGES_lowband/'
            elif simulation == 'FEKO':
                g2 = 'FEKO_simulation' 
            gpath = g1 + g2
        else:
            gpath = 'no_ground_plane/'
        plt.savefig('plots/' + gpath + loc +'/rms_plots/rms_comparison')
    plt.show()

def plot_residuals(f, t, l, azimuth=0, lst_for_plot=[0, 6, 12, 18], flow=50, fhigh=100, Nfg_array=[5, 6], model_type='LINLOG', savepath=None, ylim=None, textloc=None):
    # plots for five-term fit and six-term fit
    rms, res = compute_rms(f, t, flow=flow, fhigh=fhigh, Nfg_array=Nfg_array, model_type=model_type)
    f = f[(f >= flow) & (f <= fhigh)]
    plt.figure()
    s1 = 'RMS\n'
    for i, n in enumerate(Nfg_array):
        for lst in lst_for_plot:
            lst_point = int(10 * lst) # since resolution is .1 hours
            plt.plot(f, res[i, lst_point, :], label='LST = {} hrs, parameters = {}'.format(lst, n))
            s1 += 'LST = {} hrs, '.format(lst) + str(n) + ' parameters: {:.3g} \n'.format(rms[0, i])
    if not textloc:
        plt.text(x=f.max()-0.5*(f.max()-f.min()), y=res[0, int(10*lst_for_plot[0]), :].min(), s=s1, bbox=dict(facecolor='wheat'))
    else:
        plt.text(x=textloc[0], y=textloc[1], s=s1, fontsize='small', bbox=dict(facecolor='wheat', alpha=0.5))
    plt.xlabel(r'$\nu$ [MHz]')
    plt.ylabel('T [K]')
    if ylim:
        plt.ylim(ylim)
    title_str = r'Residuals vs Frequency for $\psi_0={}$'.format(azimuth)
    plt.title(title_str)
    plt.legend()
    if savepath:
        plt.savefig('plots/' + savepath)
    plt.show()
    return rms

def sliding_average(array, window_length, cycle=True):
    '''cycle = true means that we slide around like a clock, 1-2-...-11-12-1...,
    false means to just forget about the ends'''
    if cycle:
        new_arr = np.empty(len(array) + window_length - 1)    
        new_arr[:-(window_length-1)] = np.copy(array)
        new_arr[-(window_length-1):] = np.copy(array[:window_length-1])
    else:
        new_arr = np.copy(array)
    avg = np.convolve(new_arr, np.ones(window_length)/window_length, mode='valid')
    return avg
        

def sliding_binLST(f_in, temp, lst, bin_width, model='LINLOG', band='low', Nfg=5):
    if band == 'low':
        flow = 50
        fhigh = 100
    elif band == 'high':
        flow = 90
        fhigh = 200
    f = f_in[(f_in >= flow) & (f_in <= fhigh)]
    temp = temp[:, (f_in >= flow) & (f_in <= fhigh)]
    t_avg_arr = np.empty(temp.shape)
    for i in range(temp.shape[1]):
        t_avg_arr[:, i] = sliding_average(temp[:, i], bin_width) # get sliding 241/bin_width averages
    print(t_avg_arr.shape)
    rms_vals = []
    res_vals = np.empty((len(t_avg_arr), len(f)))
    for i, t_avg in enumerate(t_avg_arr):
        rms, res = compute_rms(f, t_avg, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])
        rms_vals.append(rms[0])
        res_vals[i, :] = res
    min_idx = np.argmin(rms)
    smallest_res = res_vals[min_idx, :]        
    smallest_rms = rms[min_idx]
    lst_range = (min_idx * bin_width, min_idx * bin_width + bin_width)
    return lst_range, smallest_rms, smallest_res

def getall_sliding_avgs(f_in, temp, lst, bin_widths, band='low', model='LINLOG', Nfg=5, savename=None):
    N = len(bin_widths)
    lst_ranges = np.empty((N, 2))
    rms_arr = np.empty(N)
    if band == 'low':
        flow = 50
        fhigh = 100
    elif band == 'high':
        flow = 90
        fhigh = 2000
    f = f_in[(f_in>=flow) & (f_in<=fhigh)]
    res_arr = np.empty((N, len(f)))
    plt.figure()
    for i, bin_width in enumerate(bin_widths):
        l, rm, re = sliding_binLST(f_in, temp, lst, bin_width, model=model, band=band, Nfg=Nfg)
        lst_ranges[i, :] = l
        rms_arr[i] = rm
        res_arr[i, :] = re
        plt.plot(f, re, label='Range: '+ str(l) + ', rms: {}'.format(rm))
    plt.legend()
    if savename:
        plt.savefig(savename)

def binLST(f_in, temp, lst, bins, model='LINLOG', band='low', Nfg=5, split=4, ylim=None, savepath=None, textloc=None):
    width = int(len(lst)/bins)
    if band == 'low':
        flow = 50
        fhigh = 100
    else:
        flow = 100
        fhigh = 190
    f = f_in[(f_in>=flow) & (f_in<=fhigh)]
    temp = temp[:, (f_in>=flow) & (f_in<=fhigh)]
    plt.figure()
    rms_str = 'RMS:'
    resmin = 0
    for n in range(bins):
        if n % split == 0 and n != 0:
            plt.legend()
            plt.title('LST averaged spectra using bins of width {:d} hr'.format(int(width/10)))
            plt.xlabel(r'$\nu$ [MHz]')
            plt.ylabel('Residuals [K]')
            bbd = {}
            bbd['facecolor'] = 'wheat'
            bbd['alpha'] = 0.7
            plt.text(x=fhigh-0.5*(fhigh-flow), y=resmin, s=rms_str, bbox=bbd)
            plt.ylim(ylim)
            if savepath:
                suf = str(int(n/split))
                sp = 'plots/' + savepath + suf
                plt.savefig(sp)
            plt.figure()
            rms_str = 'RMS:'
        lstmin = n * width
        lstmax = lstmin + width
        if n == bins-1:
            t = temp[lstmin:, :]
        else:
            t = temp[lstmin:lstmax, :]
        tavg = np.mean(t, axis=0)
        rms, res = compute_rms(f, tavg, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])
        rms_str += '\n{:d}-{:d} hr: {:.3g} K'.format(int(lstmin/10), int(lstmax/10), rms[0, 0])
        plt.plot(f, res[0, 0, :], label='{:d}-{:d} hr'.format(int(lstmin/10), int(lstmax/10)))
        if res[0, 0, :].min() < resmin:
            resmin = res[0, 0, :].min()
    plt.legend()
    plt.title('LST averaged spectra using bins of width {:d} hr'.format(int(width/10)))
    plt.xlabel(r'$\nu$ [MHz]')
    plt.ylabel('Residuals [K]')
    plt.ylim(ylim)
    bbd = {}
    bbd['facecolor'] = 'wheat'
    bbd['alpha'] = 0.7
    if textloc:
        xtext, ytext = textloc
    else:
        xtext = fhigh - 0.5 * (fhigh-flow)
        ytext = resmin
    plt.text(x=xtext, y=ytext, s=rms_str, bbox=bbd)
    if savepath:
        sp = 'plots/' + savepath
        plt.savefig(sp)
    return None

def add_Gaussian(f, t, l, width, amplitude, centre=75):
    M = 149 # number of points, the gaussian will be centred at the middle
    sigma = width
    sigma /= 2 * np.sqrt(2 * np.log(2))
    g = gaussian(M, std=width)
    g *= amplitude
    find = [int(freq) for freq in f]
    g_slice = g[find]
    newt = t.copy()
    if len(newt.shape) == 1:
        newt += g_slice
    else:
        lstpoint = int(10 * l)
        newt[lstpoint, :] += g_slice
    return newt

def gaussian_rms(f, t, l, width_arr, amplitude_arr, centre=75, model='LINLOG', Nfg=5, flow=50, fhigh=100, clim=None, log10=False):
    lstpoint = int(10 * l)
    if len(t.shape) == 1:
        rms_ref = compute_rms(f, t, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0][0, 0]
    else:
        rms_ref = compute_rms(f, t[lstpoint, :], flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0][0, 0]
    rms_g_arr = np.empty((len(width_arr), len(amplitude_arr)))
    for i, w in enumerate(width_arr):
        for j, a in enumerate(amplitude_arr):
            tg = add_Gaussian(f, t, l, w, a, centre=centre)
            if len(tg.shape) == 1:
                rms_g = compute_rms(f, tg, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0]
            else:
                rms_g = compute_rms(f, tg[lstpoint, :], flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0]
            rms_g_arr[i, j] = rms_g[0, 0]
    plt.figure()
    right, left = amplitude_arr.max(), amplitude_arr.min()
    bottom, top = width_arr.max(), width_arr.min()
    if not log10:
        plt.imshow(rms_g_arr/rms_ref, aspect='auto', extent=[left, right, bottom, top])
    else:
        plt.imshow(np.log10(rms_g_arr/rms_ref), aspect='auto', extent=[left, right, bottom, top])
    plt.ylabel('Width [MHz]')
    plt.xlabel('Amplitude [K]')
    plt.title('RMS(Amplitude, Width) / RMS(No Gaussian)')
    cbar = plt.colorbar()
    if not log10:
        cbarlabel = 'RMS/({:.2g} K)'.format(rms_ref)
    else:
        cbarlabel = 'log10(RMS/({:.2g} K))'.format(rms_ref)
    cbar.set_label(cbarlabel)
    if clim:
        plt.clim(*clim)


def get_best_LST_combination(loc, ground_plane, simulation, azimuth=0, model_type='LINLOG', Nfg=[5], flow=50, fhigh=100):
    f, t, l = get_ftl(azimuth, loc=loc, ground_plane=ground_plane, simulation=simulation)
    rms_min = 1e6 # just some really large number that makes the first rms less than this
    lstc_min = []
    res_min = []
    for i in range(1, len(l)+1): ## loop through all possible lengths
        print(i)
        combinations = itertools.combinations(l, i)
        for lstc in combinations: ## single combination of lst hrs
            indices = []
            for lstci in lstc:
                indices.append(np.argwhere(l == lstci)[0])
            t2 = t[indices, :]
            t_avg = np.mean(t2, axis=0)
            rms, res = compute_rms(f, t_avg, flow=flow, fhigh=fhigh, Nfg_array=Nfg, model_type=model_type)
            if rms < rms_min:
                rms_min = rms
                res_min = res
                lstc_min = lstc
    return lstc_min, res_min, rms_min




def save_all_plots(loc='mars', ground_plane=True, simulation='edges_hb'):
#    azimuths = [0, 30, 60, 90, 120, 150]
    azimuths = [0, 90]
    print('temp3d and rms')
    for phi in azimuths:
        plot_temp_3d(phi, loc=loc, ground_plane=ground_plane, save=True, simulation=simulation)
        plot_rms(phi, loc=loc, ground_plane=ground_plane, save=True, simulation=simulation)
    print('waterfalls_diff')
    plot_waterfalls_diff(azimuths=azimuths, loc=loc, ground_plane=ground_plane, save=True, simulation=simulation)
    print('rms comparison')
    plot_rms_comparison(azimuths=azimuths, loc=loc, ground_plane=ground_plane, save=True, simulation=simulation)
    print('residuals')
    plot_residuals(loc=loc, ground_plane=ground_plane, save=True, simulation=simulation)
