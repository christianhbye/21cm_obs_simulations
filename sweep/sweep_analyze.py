import analyze as a
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import numpy as np

def reverse_and_shift(array, return_ind=False):
    '''Currently we have lst in increasing order 0-24 hrs,
    Mozdzen have 12-10-8-..-0-2-4-...12. This function converts
    to that order.'''
    ind_array = np.arange(len(array)) # array of indices
    mp = int(len(ind_array)/2) # mid-index, maps to value 0
    new_arr = ind_array.copy()
    new_arr -= mp # now the mid-index has value 0
    part1 = new_arr[mp+1:] # first half
    part2 = new_arr[:mp] + ind_array[-1] + 1 # second half
    final_arr = np.empty(new_arr.shape)
    # reverse order
    final_arr[:mp] = part1[::-1]
    final_arr[mp] = 0
    final_arr[mp+1:] = part2[::-1]
    new_ind = final_arr.astype(int)
    if return_ind:
        return new_ind # return the index array
    else:
        return array[new_ind] # return the new array

def rands_nd(array, axis):
    ''' N-d version of reverse and shift'''
    if axis == 0:
        new = reverse_and_shift(array)
    elif axis == 1:
        idx = reverse_and_shift(array[0, :], return_ind=True)
        new = array[:, idx]
    return new

def rms_sweep(ground_plane, simulation, azimuth=0, model='EDGES_polynomial', Nfg=6, avg=False):
    N_lat = 121
    lat_array = np.linspace(90, -90, N_lat) # 121 latitudes gives 1.5 deg resolution
    __, __, lst = a.get_ftl(azimuth, loc='sweep', sweep_lat=lat_array[0], ground_plane=ground_plane, simulation=simulation)
    if avg:
        rms_arr = np.empty((N_lat))
    else:
        rms_arr = np.empty((N_lat, len(lst)))
    it = np.nditer(lat_array, flags=['f_index'])
    if simulation == 'edges_hb':
        flow = 100
        fhigh = 190
    else:
        flow = 40
        fhigh = 120
    for lat in it:
        try:
            f, t, l = a.get_ftl(azimuth, loc='sweep', sweep_lat=lat, ground_plane=ground_plane, simulation=simulation)
            if avg:  # just compute rms of avg t
                t = np.mean(t, axis=0)
            assert l.all() == lst.all()
            nrms = a.compute_rms(f, t, flow, fhigh, Nfg_array=[Nfg], model_type=model)[0]
            if avg:
                rms = nrms[0]
            else:
                rms = nrms[:, 0]
        except:
            rms = None
            print(lat)
        if avg:
            rms_arr[it.index] = rms
        else:
            rms_arr[it.index, :] = rms
    return rms_arr

def plot_hist(antenna_type, model, Nfg_array=[5, 6, 7], azimuths=[0, 90, 120], no_bins=100):
    if antenna_type == 'mini_MIST':
        ground_plane = True
    else:
        ground_plane = False
    data = []
    labels = []
    for az in azimuths:
        for N in Nfg_array:
            rms = rms_sweep(ground_plane, antenna_type, az, model=model, Nfg=N)
            rms_mk = rms.flatten() * 1000  # convert to mK
            data.append(rms_mk)
            lab = r'$\psi_0={}^\circ$, N={}'.format(az, N)
            labels.append(lab)
    hist, bins = np.histogram(data, bins=no_bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    colors = ['blue', 'orange', 'black']
    ls = ['.', '-', '--']
    plt.figure()
    if len(Nfg_array) == 3:
        c_array = len(azimuths) * colors
    else:
        c_array = None
    plt.hist(data, histtype='step', bins=logbins, color=c_array, label=labels)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('RMS [mK]')
    plt.ylabel('Counts')
    plt.show()
   

def plot(rms_arr, azimuth, lst=None, rands_lst=False, vmin=0, vmax=None, hidex=False, hidey=False, cbar=True, log10=False, save=False):
    lat_min, lat_max = -90, 90
    plt.figure()
    # put lst=0 in the middle
    if rands_lst:
        new_rms_arr = rands_nd(rms_arr, 1)
        locs = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        labso = [120, 100, 80, 60, 40, 20, 0, 220, 200, 180, 160, 140, 120]
        labs = [int(l/10) for l in labso]
        plt.xticks(locs, labs)
    else:
        labs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        locs = [10*l for l in labs]
        plt.xticks(locs, labs) ## needds to change!!!
        new_rms_arr = rms_arr.copy()
    if lat_min != -90 or lat_max != 90:
        earr = [lst.min(), lst.max(), lat_min, lat_max]
        plt.imshow(1000 * new_rms_arr, aspect='auto', extent=earr) # *1000 to get mK
    else:
        if log10:
            norm = mpcolors.LogNorm()
        else:
            norm = 'none'
        plt.imshow(1000 * new_rms_arr, aspect='auto', norm=norm) # *1000 to get mK
        ylocs = np.arange(19) * 10/1.5
        ylabs = []
        for i in range(19):
           label = 90 - 10*i
           ylabs.append(label)
        plt.yticks(ylocs, ylabs)
    if cbar:
        plt.colorbar(label='RMS [mK]')
    if not hidex:
        plt.xlabel('LST [hr]')
    else:
        plt.xlabel('')
        plt.xticks(locs, ['']*len(locs))
    if not hidey:
        plt.ylabel('Latitude [deg]')
    else:
        plt.ylabel('')
        plt.yticks(ylocs, ['']*len(ylocs))
    plt.grid(linestyle='--')
    plt.text(17, 17, r'$\psi_0={}$ deg'.format(azimuth), color='white', size=16)
    plt.clim(vmin, vmax)
    if save:
        plt.savefig('plots/' + model + '_' + str(azimuth) + '.svg')
    

def subplot(rms_arr_list, azlist, nrows=3, ncols=2):
    """
    rms_arr_list is a list of rms_arr st [0] goes in the first subplot etc
    """
    Nplots = nrows * ncols
    fig, axs = plt.subplots(figsize=(10, 15), nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    imgs = []
    for i, ax in enumerate(axs.flat):
        new_rms_arr = rands_nd(rms_arr_list[i], 1)
        im = ax.imshow(new_rms_arr * 1000, aspect='auto')
        imgs.append(im) 
        ax.grid(linestyle='--')
        ax.set_title(r'$\psi_0={}$ deg'.format(azlist[i]))
        if (i+1) % ncols == 0:
            fig.colorbar(imgs[i], ax=axs[int((i+1)/ncols)-1, :], shrink=0.8, label='RMS [mK]')
    locs = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    labso = [120, 100, 80, 60, 40, 20, 0, 220, 200, 180, 160, 140, 120]
    labs = [int(l/10) for l in labso]
    plt.xticks(locs, labs)
    plt.xlabel('LST [hr]')
    plt.ylabel('Latitude [deg]')
    ylocs = np.arange(19) * 10/1.5
    ylabs = []
    for i in range(19):
        label = 90 - 10*i
        ylabs.append(label)
    plt.yticks(ylocs, ylabs)

