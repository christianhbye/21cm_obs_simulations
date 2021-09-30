import h5py
import numpy as np
import os
import general as gen
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import paper_plots as pp

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def read_hdf5(azimuth, varname, loc, sweep_lat=None, ground_plane=True, simulation='edges_hb'):
    parent_path = '/scratch/s/sievers/cbye/21cm_obs_simulations/'
    if ground_plane:
        g1 = 'metal_ground_plane/'
        if simulation == 'edges_hb':
            g2 = 'EDGES_highband/'
        elif simulation == 'edges_lb':
            g2 = 'EDGES_lowband/'
        elif simulation == 'FEKO':
            g2 = 'FEKO_simulation/' 
        elif simulation == 'mini_MIST':
            g2 = 'mini_MIST/'
    else:
        g1 = 'no_ground_plane/'
        g2 = simulation + '/'
    gpath = g1 + g2
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

def get_ftl(azimuth, loc='mars', sweep_lat=None, ground_plane=False, simulation='old_MIST', return_fl=True, return_t=True):
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

def plot_beam(antenna_name, antenna_orientation, phi, gain, frequency, climbeam=None, climderiv=None, savepath=None):
    if gain.shape[-1] == 361:
        gain = gain[:, :, :-1] # cut last angle since 0 = 360 degrees
    gain = gain[:, ::-1, :] # changing from elevation to theta
    print('Min frequency = {}'.format(frequency.min()))
    print('Max frequency = {}'.format(frequency.max()))
    if frequency.min() > 1e6:
        frequency /= 1e6 # to MHz
        print('Convert frequency')
        print('Min frequency = {}'.format(frequency.min()))
        print('Max frequency = {}'.format(frequency.max()))
    print('Frequency shape = {}'.format(frequency.shape))
    plt.figure()
    plt.imshow(gain[:, :, phi], aspect='auto', extent=[0, 90, frequency.max(), frequency.min()], interpolation='none')
    if climbeam:
        plt.clim(climbeam)
    plt.colorbar()
    plt.title(r'Gain ($\phi = {}$, $\psi_0 = {}$)'.format(phi, antenna_orientation) +'\n' + antenna_name)
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
    plt.imshow(derivative, aspect='auto', extent=[0, 90, frequency.max(), frequency.min()], interpolation='none')
    plt.xlabel(r'$\theta$ [deg]')
    plt.ylabel(r'$\nu$ [MHz]')
    if climderiv:
        plt.clim(climderiv)
    plt.colorbar()
    plt.title(r'Derivative ($\phi = {}$, $\psi_0 = {}$)'.format(phi, antenna_orientation) +'\n'+ antenna_name)
    if savepath:
        sp = 'plots/' + savepath + '/deriv'
        plt.savefig(sp)

def compute_deriv(gain, phi, frequency):
    dG = gain[1:, :, phi] - gain[:-1, :, phi]
    diff = frequency[1:] - frequency[:-1]
    assert diff.all() == frequency[1] - frequency[0], 'not constant frequency spacing'
    df = frequency[1] - frequency[0]
    derivative = dG/df
    return derivative

def plot_all_beams(gain_list=None, f=None, derivs=False):
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0)
    rgain = False
    if not gain_list:
        rgain = True
        f, az, el, __, __, gain_large = gen.read_beam_FEKO('no_git_files/blade_dipole.out', 0)
        f_s, az_s, el_s, __, __, gain_small = gen.read_beam_FEKO('no_git_files/blade_dipole_MARS.out', 0)
        f_gp, az_gp, el_gp, __, __, gain_gp = gen.read_beam_FEKO('no_git_files/mini_mist_blade_dipole_3_groundplane_no_boxes.out', 0)
        assert f.all() == f_s.all()
        assert f.all() == f_gp.all()
        gain_list = [gain_large, gain_small, gain_gp]
        return_gain_list = [g.copy() for g in gain_list]
    if f[0] >= 1e6:  # units is Hz
        f /= 1e6  # convert to MHz
    if derivs:
        vmin, vmax = -0.08, 0.08
    else:
        vmin, vmax = 0, 8.2
    for i, gain in enumerate(gain_list):
        if gain.shape[-1] == 361:
            gain = gain[:, :, :-1] # cut last angle since 0 = 360 degrees
        gain = gain[:, ::-1, :] # changing from elevation to theta
        if derivs:
            deriv0 = compute_deriv(gain, 0, f)
            deriv90 = compute_deriv(gain, 90, f)
            toplot0, toplot90 = deriv0, deriv90
        else:
            toplot0, toplot90 = gain[:, :, 0], gain[:, :, 90]
        axs[i, 0].imshow(toplot0, aspect=9/8, extent=[0, 90, f.max(), f.min()], interpolation='none', vmin=vmin, vmax=vmax)
        im = axs[i, 1].imshow(toplot90, aspect=9/8, extent=[0, 90, f.max(), f.min()], interpolation='none', vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylabel(r'$\nu$ [MHz]')
    axs[0, 0].set_title(r'$\phi=0 \degree$')
    axs[0, 1].set_title(r'$\phi=90 \degree$')
    axs[2, 0].set_xlabel(r'$\theta$ [deg]')
    axs[2, 1].set_xlabel(r'$\theta$ [deg]')
    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(MultipleLocator(15))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
    cbar = fig.colorbar(im, ax=axs.ravel(), location='right')
    if derivs:
        cbar.set_label('Derivative')
        plt.suptitle('Derivative')
    else:
        cbar.set_label('Gain')
        plt.suptitle('Gain')
#    plt.tight_layout()
    plt.show()
    if rgain:
        return return_gain_list, f


def polar_beam(gain_list=None, f=None, figsize=None):
    return_gain = False
    if not gain_list:
        f, az, el, __, __, gain_large = gen.read_beam_FEKO('no_git_files/blade_dipole.out', 0)
        f_s, az_s, el_s, __, __, gain_small = gen.read_beam_FEKO('no_git_files/blade_dipole_MARS.out', 0)
        f_gp, az_gp, el_gp, __, __, gain_gp = gen.read_beam_FEKO('no_git_files/mini_mist_blade_dipole_3_groundplane_no_boxes.out', 0)
        assert f.all() == f_s.all()
        assert f.all() == f_gp.all()
        gain_list = [gain_large, gain_small, gain_gp]
        rgain_list = [gain_large.copy(), gain_small.copy(), gain_gp.copy()]
        return_gain = True
    if f[0] >= 1e6:  # units is Hz
        f /= 1e6  # convert to MHz
    find_to_plot = np.arange(0, 82, 10)  # list of frequency indices to plot gain cuts for
    fig = plt.figure(figsize=figsize)
    dx, dy = 0.8, 0.8  # panel width and height
    axs = []
    el = np.deg2rad(np.arange(-91, 91))
    for i, gain in enumerate(gain_list):  # loop over rows / antenna models
        if gain.shape[-1] == 361:
            gain = gain[:, :, :-1] # cut last angle since 0 = 360 degrees
        gain = gain[:, ::-1, :] # changing from elevation to theta
        ax1 = fig.add_axes([0, -i*dy, dx, dy], polar=True, label=str(i)+'1')
        axs.append(ax1)
        ax1.text(np.pi/4, 14, chr(97+2*i)+')', size=MEDIUM_SIZE)
        ax2 = fig.add_axes([0.7*dx, -i*dy, dx, dy], polar=True, label=str(i)+'2')
        ax2.text(np.pi/4, 14, chr(97+2*i+1)+')', size=MEDIUM_SIZE)
        axs.append(ax2)
        for j, find in enumerate(find_to_plot):  # loop over the frequencies to plot gain for in each panel
            phi0 = gain[find, :, 0]  # phi = 0
            reverse0 = np.flip(phi0)
            gain0 = np.concatenate((reverse0, phi0))
            ax1.plot(el, gain0, label='{:d} MHz'.format(int(f[find])), linewidth=0.75)  # phi = 0 
            phi90 = gain[find, :, 90]  # phi = 90
            reverse90 = np.flip(phi90)
            gain90 = np.concatenate((reverse90, phi90))
            ax2.plot(el, gain90, label='{:d} MHz'.format(int(f[find])), linewidth=0.75)  # phi = 90
    plt.setp(axs, theta_zero_location='N')
    plt.setp(axs, thetamin=-90, thetamax=90)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.7*dx, -2*dy), ncol=3)
    for ax in axs:
        ax.set_rgrids([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['0', ' ', '2', ' ', '4', ' ', '6', ' ', '8', ' '], angle=22.5)
    thetas = [-90, -45, 0, 45, 90]
    tticks = [np.deg2rad(t) for t in thetas]
    tlabels = [r'${:d} \degree$'.format(int(np.abs(t))) for t in thetas]
    plt.setp(axs, xticks=tticks, xticklabels=tlabels)
    axs[0].set_title(r'$\phi=0 \degree$')
    axs[1].set_title(r'$\phi=90 \degree$')
    if return_gain:
        return rgain_list, f, fig, axs
    else:
        return fig, axs

def polar_beam_old(gain_list=None, f=None, figsize=None):
    return_gain = False
    if not gain_list:
        f, az, el, __, __, gain_large = gen.read_beam_FEKO('no_git_files/blade_dipole.out', 0)
        f_s, az_s, el_s, __, __, gain_small = gen.read_beam_FEKO('no_git_files/blade_dipole_MARS.out', 0)
        f_gp, az_gp, el_gp, __, __, gain_gp = gen.read_beam_FEKO('no_git_files/mini_mist_blade_dipole_3_groundplane_no_boxes.out', 0)
        assert f.all() == f_s.all()
        assert f.all() == f_gp.all()
        gain_list = [gain_large, gain_small, gain_gp]
        rgain_list = [gain_large.copy(), gain_small.copy(), gain_gp.copy()]
        return_gain = True
    if f[0] >= 1e6:  # units is Hz
        f /= 1e6  # convert to MHz
    find_to_plot = np.arange(0, 82, 10)  # list of frequency indices to plot gain cuts for
    fig, axs = plt.subplots(figsize=figsize, nrows=3, ncols=2, subplot_kw={'projection': 'polar'}, gridspec_kw={'wspace':0, 'hspace':0})
    plt.setp(axs, theta_zero_location='N')
    plt.setp(axs, thetamin=-90, thetamax=90)
    el = np.deg2rad(np.arange(-91, 91))
    for i, gain in enumerate(gain_list):
        if gain.shape[-1] == 361:
            gain = gain[:, :, :-1] # cut last angle since 0 = 360 degrees
        gain = gain[:, ::-1, :] # changing from elevation to theta
        for j, find in enumerate(find_to_plot):
            phi0 = gain[find, :, 0]  # phi = 0
            reverse0 = np.flip(phi0)
            gain0 = np.concatenate((reverse0, phi0))
            axs[i, 0].plot(el, gain0, label='{:d} MHz'.format(int(f[find])))  # phi = 0 
            phi90 = gain[find, :, 90]  # phi = 90
            reverse90 = np.flip(phi90)
            gain90 = np.concatenate((reverse90, phi90))
            axs[i, 1].plot(el, gain90, label='{:d} MHz'.format(int(f[find])))  # phi = 90 
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=3)
    for ax in axs.flatten():
        ax.set_rgrids([1, 2, 3, 4, 5, 6, 7, 8, 9], [' ', '2', ' ', '4', ' ', '6', ' ', '8', ' '], angle=22.5)
    thetas = [-90, -45, 0, 45, 90]
    tticks = [np.deg2rad(t) for t in thetas]
    tlabels = [r'${:d} \degree$'.format(int(np.abs(t))) for t in thetas]
    plt.setp(axs, xticks=tticks, xticklabels=tlabels)
    axs[0, 0].set_title(r'$\phi=0 \degree$')
    axs[0, 1].set_title(r'$\phi=90 \degree$')
    if return_gain:
        return rgain_list, f, fig, axs
    else:
        return fig, axs

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

def plot_temp_3d(freq_vector, lst_vector, temp_array_N, temp_array_S, figsize=None):   
    fig, axs = pp.plot_basic(2, 1, True, True, figsize, 10, 5, None, None, dx=0.8, dy=0.8, vspace=1.1, customy=True)
    if freq_vector[0] > 1e6:
        freq_vector /= 1e6 # convert to MHz
    freq_min = freq_vector[0]
    freq_max = freq_vector[-1]
    LST_min = lst_vector[0]
    LST_max = lst_vector[-1]
    im = axs[0].imshow(temp_array_N, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min], interpolation='none')
    nticks = [0, 2000, 4000, 6000, 8000, 10000]
    sticks = [0, 5000, 10000, 15000, 20000]
    cbarN = fig.colorbar(im, ax=axs[0], ticks=nticks)
    cbarN.set_ticklabels([str(t) for t in nticks])
    cbarN.set_label("[K]", labelpad=15)
    im.set_clim(0, 10000)
    im2 = axs[1].imshow(temp_array_S, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min], interpolation='none')
    cbarS = fig.colorbar(im2, ax=axs[1], ticks=sticks)
    cbarS.set_ticklabels([str(t) for t in sticks])
    cbarS.set_label("[K]", labelpad=15)
    im2.set_clim(0, 20000)
    plt.setp(axs, ylabel='LST [h]')
    axs[1].set_xlabel('Frequency [MHz]')
    axs[0].text(112, 2, 'a)', color='white', fontsize=MEDIUM_SIZE)
    axs[1].text(112, 2, 'b)', color='white', fontsize=MEDIUM_SIZE)
    yticks = [0, 4, 8, 12, 16, 20, 24]
   # locs = [10*t for t in yticks]
  #  locs[-1] = 241
    plt.setp(axs, yticks=yticks)
#    plt.setp(axs, yaxis.set_minor_locator(MultipleLocator(1))
    axs[0].tick_params(axis='x', which='minor', bottom=False)
    axs[1].tick_params(axis='x', which='minor', bottom=False)
    return fig, axs

def plot_waterfalls_diff(f, t, l, ref_t, psi0, ref_psi0, clim=None, savepath=None):
    dt = t - ref_t
    if f[0] > 1e6:
        f /= 1e6
    plt.figure()
    freq_min = f[0]
    freq_max = f[-1]
    LST_min = l[0]
    LST_max = l[-1]
    plt.imshow(dt, aspect='auto', extent=[freq_min, freq_max, LST_max, LST_min], interpolation='none')
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

def compute_rms(f, t, flow=40, fhigh=120, Nfg_array=[6], frequency_normalization=100, noise_normalization=0.1, noise=False, model_type='EDGES_polynomial'):
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

def plot_rms(f, t, lst, flow=40, fhigh=120, Nfg_array=[1, 2, 3, 4, 5, 6], Nfg_split=3, save=False, loc='mars', ground_plane=True, simulation='edges_hb', frequency_normalization=100, noise_normalization=0.1, noise=False, model_type='LINLOG'):
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

def plot_residuals(f, t, l, azimuth=0, lst_for_plot=[0, 6, 12, 18], flow=40, fhigh=120, Nfg_array=[5, 6], model_type='LINLOG', savepath=None, ylim=None, textloc=None):
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

def sliding_average2d(temp, bin_width):
    t_avg_arr = np.empty(temp.shape)
    for i in range(temp.shape[1]):
        t_avg_arr[:, i] = sliding_average(temp[:, i], bin_width)
    return t_avg_arr


def sliding_binLST(f_in, temp, bin_width, model='LINLOG', band='low', Nfg=5):
    if band == 'low':
        flow = 40
        fhigh = 120
    elif band == 'high':
        flow = 100
        fhigh = 190
    f = f_in[(f_in >= flow) & (f_in <= fhigh)]
    temp = temp[:, (f_in >= flow) & (f_in <= fhigh)]
    t_avg_arr = sliding_average2d(temp, bin_width)
    rms_vals = np.empty(temp.shape[0])
    for i, t_avg in enumerate(t_avg_arr):
        rms, res = compute_rms(f, t_avg, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])
        rms_vals[i] = rms[0, 0]
    return rms_vals

def get_smallestLST(f_in, temp, lst, bin_widths=[10, 20, 30, 40, 60, 80, 120, 241], model='LINLOG', band='low', Nfg=5):
    smallest_rms = 1000 # initialize with some large value
    lst_start = 0
    bin_width_min = 0
    for i, bw in enumerate(bin_widths):
        rms = sliding_binLST(f_in, temp, bw, model=model, band=band, Nfg=Nfg)
        if np.min(rms) < smallest_rms:
            ind = np.argmin(rms)
            smallest_rms = np.min(rms)
#            lst_start = lst[ind]
            bin_width_min = bw
    return ind, bin_width_min, smallest_rms

def smallestLSTavg(f_in, temp, lst, bin_widths=[10, 20, 30, 40, 60, 80, 120, 241], model='LINLOG', band='low', Nfg=5):
    ind, bwmin, __ = get_smallestLST(f_in, temp, lst, bin_widths=bin_widths, model=model, band=band, Nfg=Nfg)
    tavg = sliding_average2d(temp, bwmin)
    newt = tavg[ind, :]
    return newt

def get_smallest12_24(antenna_type, latitudes=[79.5, -24.0], azimuths=[0, 90, 120], model='EDGES_polynomial', Nfg=6):
    if antenna_type == 'mini_MIST':
        ground_plane = True
    else:
        ground_plane = False
    results24 = np.empty((len(latitudes), len(azimuths)))
    results12 = np.empty((len(latitudes), len(azimuths), 2))
    for i, lat in enumerate(latitudes):
        for j, az in enumerate(azimuths):
            f, t, l = get_ftl(az, loc='sweep', sweep_lat=lat, ground_plane=ground_plane, simulation=antenna_type)
            t24 = np.mean(t, axis=0)  # 24 h avg
            rms = compute_rms(f, t24, 40, 120, Nfg_array=[6], model_type=model)[0]
            rms_mk_24 = rms[0,0] * 1000
            results24[i, j] = rms_mk_24
            ind12, __, rms_12 = get_smallestLST(f, t, l, bin_widths=[120], model=model, band='low', Nfg=6)
            assert __ == 120
            rms_mk_12 = rms_12 * 1000
            results12[i, j, 0] = ind12
            results12[i, j, 1] = rms_mk_12
    return results12, results24
            

def plot_LSTbins(f_in, temp, lst, bin_widths, model='LINLOG', band='low', Nfg=5, split=None, ylim=None):
    plt.figure()
    plt.xlabel('LST')
    plt.ylabel('RMS')
    for i, bw in enumerate(bin_widths):
        if split and i % split == 0 and not i == 0:
            plt.legend()
            if ylim:
                plt.ylim(ylim)
            plt.show()
            plt.figure()
            plt.xlabel('LST')
            plt.ylabel('RMS')
        rms = sliding_binLST(f_in, temp, bw, model=model, band=band, Nfg=Nfg)
        plt.plot(lst, rms, label='Bin width = {:d} hrs'.format(int(bw/10)))
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    plt.show()    

def plot2D_LSTbins(f_in, temp, lst, bin_widths=[10, 20, 30, 40, 60, 80, 120, 241], model='EDGES_polynomial', band='low', Nfg=6):
    rms_arr = np.empty((len(bin_widths)+1, len(lst)))
    if band == 'low':
        flow = 40
        fhigh = 120
    elif band == 'high':
        flow = 100
        fhigh = 190
    rms0 = compute_rms(f_in, temp, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0]
    rms_arr[0, :] = 1000 * rms0[:, 0]
    for i, bw in enumerate(bin_widths):
        rms = sliding_binLST(f_in, temp, bw, model=model, band=band, Nfg=Nfg)
        rms_arr[i+1, :] = rms * 1000
    return rms_arr
#    plt.figure()
#    plt.xlabel('LST [hr]')
#    plt.ylabel('Bin Width [hr]')
#    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240], ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24'])
#    plt.imshow(rms_arr, aspect='auto', interpolation='none')
#    plt.colorbar(label='RMS [mK]')
#    plt.clim(vmin=vmin, vmax=vmax)
 #   plt.grid(which='minor')

def subplot_LSTbins(rms_arr_super, vmin, vmax):
    fig, axs = pp.plot_basic(3, 3, True, True, figsize=None, xmajor=4, xminor=1, ymajor=None, yminor=None, dx=0.5, dy=0.5, hspace=1.1, vspace=1.05, customy=True)
    extent = [0, 24, 8.5, -0.5]
    for i in range(9):
        im = axs[i].imshow(rms_arr_super[i // 3, i%3, :, :], aspect='auto', extent=extent, interpolation='none', norm=mpcolors.LogNorm()) 
        im.set_clim(vmin, vmax)
        axs[i].text(22, 0.25, chr(97+i)+')', color='white', fontsize=MEDIUM_SIZE)
    if vmin == 1 and vmax == 120:
        cticks = [1, 10, 50, 100, 120]
    elif vmin == 10 and vmax == 500:
        cticks = [10, 50, 100, 200, 300, 400, 500]
    cax = fig.add_axes([(3-1)*0.5*1.1+0.5+0.025, -2*0.5*1.05, 0.075, (3-1)*0.5*1.05+0.5])
    cbar = fig.colorbar(im, cax=cax, ticks=cticks)
    cbar.set_label('RMS [mK]')
    cbar.set_ticklabels([str(t) for t in cticks])
    plt.setp(axs, yticks=np.arange(9), yticklabels=['0', '1', '2', '3', '4', '6', '8', '12', '24'])
    for i in range(3):
        for j in range(2):
            axs[3*i+j+1].set_yticklabels([])
    for i in range(3):
        axs[3*i].set_ylabel('Bin Width [h]')
        axs[-(i+1)].set_xlabel('LST [h]')
    azs = [0, 90, 120]
    for i in range(3):
        axs[i].set_title(r'$\psi_0 = {:d} \degree$'.format(azs[i]))
    return fig, axs

def add_Gaussian(f, t, width, amplitude, centre=80):
    if width == 0:
        return None
    elif amplitude == 0:
        return t.copy()
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    exponential = (f - centre) ** 2
    exponential /= 2 * sigma **2
    signal = amplitude * np.exp(-1 * exponential)
    newt = t.copy()
    if len(newt.shape) == 1:
        newt += signal
    else:
        print('shucks')
        return None
    return newt

def add_EDGESsignal(f, t, tau, amplitude):
    centre = 78
    width = 19 # FWHM
    if tau == 0:
        return add_Gaussian(f, t, 19, amplitude, centre=78)
    B = 4 * (f-centre)**2 / width**2
    B *= np.log(-1/tau * np.log((1+np.exp(-tau))/2))
    signal = 1 - np.exp(-tau * np.exp(B))
    signal /= 1 - np.exp(-tau)
    signal *= amplitude
    newt = t.copy()
    newt += signal
    return newt

def gaussian_rms(f, t, width_arr=None, amplitude_arr=None, centre=80, model='EDGES_polynomial', Nfg=6, flow=40, fhigh=120, vmin=0, vmax=None, log10=False, plot=False):
    if width_arr is None:
        width_arr = np.linspace(0, 50, 201)
    if amplitude_arr is None:
        amplitude_arr = np.linspace(-1, 0, 201)
    rms_ref = compute_rms(f, t, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0][0, 0]
    rms_g_arr = np.empty((len(width_arr), len(amplitude_arr)))
    for i, w in enumerate(width_arr):
        for j, a in enumerate(amplitude_arr):
            if w == 0:
                rms_g = np.array([[None]])
            else:
                tg = add_Gaussian(f, t, w, a, centre=centre)
                rms_g = compute_rms(f, tg, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0]
            rms_g_arr[-(i+1), -(j+1)] = rms_g[0, 0]
    if not plot:
        return rms_g_arr/rms_ref, rms_ref
    else:
        plt.figure()
        left, right = 1000*amplitude_arr.max(), 1000*amplitude_arr.min()
        top, bottom = width_arr.max(), width_arr.min()
        if log10:
            norm = mpcolors.LogNorm()
        else:
            norm = None
        plt.imshow(rms_g_arr/rms_ref, aspect='auto', extent=[left, right, bottom, top], interpolation='none', norm=norm)
        plt.ylabel('FWHM [MHz]')
        plt.xlabel('A [mK]')
        plt.title('RMS(Amplitude, Width) / RMS(No Gaussian)')
        cbar = plt.colorbar()
        cbarlabel = 'RMS/({:.2g} mK)'.format(1000*rms_ref)
        cbar.set_label(cbarlabel)
        plt.clim(vmin, vmax)

def EDGES_rms(f, t, tau_arr=None, amplitude_arr=None, model='EDGES_polynomial', Nfg=6, flow=40, fhigh=120, vmin=0, vmax=None, log10=False, plot=False):
    if tau_arr is None:
        tau_arr = np.linspace(0, 30, 121)
    if amplitude_arr is None:
        amplitude_arr = np.linspace(-1, 0, 201)
    rms_ref = compute_rms(f, t, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0][0, 0]
    rms_g_arr = np.empty((len(tau_arr), len(amplitude_arr)))
    for i, tau in enumerate(tau_arr):
        for j, a in enumerate(amplitude_arr):
            tg = add_EDGESsignal(f, t, tau, a)
            rms_g = compute_rms(f, tg, flow=flow, fhigh=fhigh, model_type=model, Nfg_array=[Nfg])[0]
            rms_g_arr[-(i+1), -(j+1)] = rms_g[0, 0]
    if not plot:
        return rms_g_arr/rms_ref, rms_ref
    else:
        plt.figure()
        left, right = 1000*amplitude_arr.max(), 1000*amplitude_arr.min()
        top, bottom = tau_arr.max(), tau_arr.min()
        if log10:
            norm = mpcolors.LogNorm()
        else:
            norm = None
        plt.imshow(rms_g_arr/rms_ref, aspect='auto', extent=[left, right, bottom, top], interpolation='none', norm=norm)
        plt.ylabel(r'$\tau$')
        plt.xlabel('A [mK]')
        plt.title(r'RMS(Amplitude, $\tau$) / RMS(No Signal)')
        cbar = plt.colorbar()
        cbarlabel = 'RMS/({:.2g} mK)'.format(1000*rms_ref)
        cbar.set_label(cbarlabel)
        plt.clim(vmin, vmax)

def plot_gauss_edges(gauss40, gauss80, gauss120, edges, log10=True, vmin=1, vmax=150, north=True):
    """
    each input is an array where the first axis is the antenna type and the second is the latitude
    """
    if north:
        i = 0
    else:
        i = 1
    input = [gauss40, gauss80, gauss120, edges]
    amplitude_arr = np.linspace(-1, 0, 201)
    width_arr = np.linspace(0, 50, 201)
    tau_arr = np.linspace(0, 30, 121)
    fig = plt.figure()
    dx, dy = 0.4, 0.4
    hspace1, hspace2, vspace = 1.1, 1.3, 1.1 
#    fig, axs = plt.subplots(figsize=(12,8), nrows=6, ncols=4, sharex=True, sharey='col', gridspec_kw={'hspace':0.15, 'wspace':0.2})
    if log10:
        norm = mpcolors.LogNorm()
    else:
        norm = None
    left, right = -1*1000*amplitude_arr.max(), -1*1000*amplitude_arr.min()
    col_title = ['Gaussian 40 MHz', 'Gaussian 80', 'Gaussian 120', 'EDGES Signal']
    for j in range(3):   # antenna
        for k in range(4):  # signal type
            if not k == 3:
                top, bottom = width_arr.max(), width_arr.min()
                ypos = 5/6*50
                xc, yc = 100, 25
                labelc = 'white'
            else:
                top, bottom = tau_arr.max(), tau_arr.min()
                ypos = 5/6*30
                xc, yc = 500, 7
                labelc = 'k'
            signal = input[k]
            if not k == 3:
                xstart = k*dx*hspace1
            else:
                xstart = 2*dx*hspace1 + dx*hspace2
            ax = fig.add_axes([xstart, -j*dy*vspace, dx, dy])
            im = ax.imshow(signal[j, i], aspect='auto', extent=[left, right, bottom, top], interpolation='none', norm=norm)
            im.set_clim(vmin, vmax)
            ax.text(11/3*250, ypos, chr(97+k+4*j)+')', color=labelc, fontsize=MEDIUM_SIZE)
            ax.scatter(xc, yc, facecolors='none', edgecolors='white')
            if j == 2:
                ax.set_xlabel('A [mK]')
            else:
                ax.set_xticklabels([])
            if k == 0:
                ax.set_ylabel('FWHM [MHz]')
            elif k == 3:
                ax.set_ylabel(r'$\tau$')
            else:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_title(col_title[k])
    cax = fig.add_axes([2*dx*hspace1+dx*hspace2+dx*1.05, -2*dy*vspace, 0.1*dx, 2*dy*vspace+dy])
    cticks = [1, 10, 50, 100, 150]
    cbar = fig.colorbar(im, cax=cax, ticks=cticks)
    cbar.set_ticklabels([str(t) for t in cticks])
#    axs[0].xaxis.set_major_locator(MultipleLocator(250))
#    axs[0,0].xaxis.set_minor_locator(MultipleLocator(125))
#    for i in range(3):
#        axs[0, i].yaxis.set_major_locator(MultipleLocator(25))
#        axs[0, i].yaxis.set_minor_locator(MultipleLocator(12.5))
#    axs[0, -1].yaxis.set_major_locator(MultipleLocator(10))
    return fig

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
