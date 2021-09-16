import analyze as a
import paper_plots as pp
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

def rms_sweep(ground_plane, simulation, azimuth=0, model='EDGES_polynomial', Nfg=6, avg=False, halfstart=None):
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
                if not halfstart:
                    t = np.mean(t, axis=0)
                else:
                    t = np.mean(t[halfstart, halfstart+120], axis=0)
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


def rmsvslat(models=['LINLOG', 'EDGES_polynomial'], azimuths=[0, 90, 120], halfstarts=None):
    antennas = ['old_MIST', 'new_MIST', 'mini_MIST']
    models=['LINLOG', 'EDGES_polynomial']
    fig, axs = plt.subplots(figsize=(10,5), nrows=3, ncols=len(models), sharex=True, sharey='col', gridspec_kw={'hspace':0.15, 'wspace':0.15})
    lats = np.linspace(90, -90, 121)
    for i, antenna in enumerate(antennas):
        axs[i, 0].set_ylabel('RMS [mK]')
        if antenna == 'mini_MIST':
            ground_plane = True
        else:
            ground_plane = False
        for j, az in enumerate(azimuths):
            for k, model in enumerate(models):
                if not halfstarts is None:
                    halfstart = halfstarts[i, j]
                else:
                    halfstart = None
                rms = rms_sweep(ground_plane, antenna, az, model, Nfg=6, avg=True, halfstart=halfstart)
                rms *= 1000
                axs[i, k].plot(lats, rms, label=r'$\psi_0={:d} \degree$'.format(az))
    axs[2, 0].set_xlabel('Latitude [deg]')
    axs[2, 1].set_xlabel('Latitude [deg]')
    axs[0, 1].legend(loc='upper left')
   # handles, labels = axs[0, 0].get_legend_handles_labels()
   # fig.legend(handles, labels, loc='upper center')
    axs[0,0].xaxis.set_major_locator(MultipleLocator(30))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(15))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(50))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(25))
    axs[0,1].yaxis.set_major_locator(MultipleLocator(10))
    axs[0,1].yaxis.set_minor_locator(MultipleLocator(5))
    axs[0,0].set_ylim(0,250)
    axs[0,1].set_ylim(0,50)
    for i in range(3):
        axs[i, 0].text(11/12*180-90, 8/9*250, chr(97+2*i)+')')
        axs[i, 1].text(11/12*180-90, 8/9*50, chr(97+2*i+1)+')')
    return fig, axs

def get_hist(antenna_type, model, Nfg_array=[5, 6, 7], azimuths=[0, 90, 120], no_bins=100):
    if antenna_type == 'mini_MIST':
        ground_plane = True
    else:
        ground_plane = False
    data = []
    for az in azimuths:
        for N in Nfg_array:
            rms = rms_sweep(ground_plane, antenna_type, az, model=model, Nfg=N)
            rms_mk = rms.flatten() * 1000  # convert to mK
            data.append(rms_mk)
    hist, bins = np.histogram(data, bins=no_bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    return data, logbins

def plot_hist():
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    axs[0, 0].set_ylabel('Counts')
    axs[1, 0].set_ylabel('Counts')
    axs[2, 0].set_ylabel('Counts')
    axs[2, 0].set_xlabel('RMS [mK]')
    axs[2, 1].set_xlabel('RMS [mK]')
    plt.setp(axs, xscale='log')
    handles = [Line2D([0], [0], color='C0', lw=4), Line2D([0], [0], color='C1', lw=4), Line2D([0], [0], color='C2', lw=4)]
    handles.append(Line2D([0], [0], color='black', lw=4, ls='solid'))
    handles.append(Line2D([0], [0], color='black', lw=4, ls='dashed'))
    handles.append(Line2D([0], [0], color='black', lw=4, ls='dotted'))
    fig.legend(handles=handles, labels=['N=5', 'N=6', 'N=7', r'$\psi_0=0 \degree$', r'$\psi_0=90 \degree$', r'$\psi_0=120 \degree$'])
#    plt.show()
    return fig, axs

def add_hist(ax, data, logbins):
    carray = ['C0', 'C1', 'C2'] * 3
    lss = ['solid', 'dashed', 'dotted']
    for i in range(9):
        ls = lss[int(i/3)] 
        ax.hist(data[i], histtype='step', bins=logbins, color=carray[i], ls=ls)

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
    

def subplot(rms_arr_list, azlist=[0, 90, 120], nrows=3, ncols=3):
    """
    rms_arr_list is a list of rms_arr st [0] goes in the first subplot etc
    """
    alph = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
    fig, axs = pp.plot_basic(nrows, ncols, sharex=True, sharey=True, wspace=0.2, hspace=0.2, figsize=None, xmajor=4, xminor=1, ymajor=30, yminor=10)
   # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        im = ax.imshow(rms_arr_list[i] * 1000, aspect='auto', extent=[0, 24, -90, 90], interpolation='none', norm=mpcolors.LogNorm())
        im.set_clim(10, 500)
        ax.grid(linestyle='--')
        ax.text(21, 60, alph[i], color='white')
       # if i > 5:
       #     ax.set_xlabel('LST [h]')
      #  if i%3 == 0:
      #      ax.set_ylabel('Latitude [deg]')
  #  labs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
  #  locs = [10*l for l in labs]
  #  plt.setp(axs, xticks=locs, xticklabels=labs)
  #  ylocs = np.arange(19) * 10/1.5
  #  ylabs = []
  #  for i in range(19):
  #      label = 90 - 10*i
  #      ylabs.append(label)
  #  plt.setp(axs, yticks=ylocs, yticklabels=ylabs)
    for i in range(3):
        axs[3*i].set_ylabel('Latitude [deg]')
        axs[6+i].set_xlabel('LST [h]')
    cbar = fig.colorbar(im, ax=axs, ticks=[10, 50, 100, 200, 300, 400, 500])
    cbar.set_ticklabels(['10', '50', '100', '200', '300', '400', '500'])
    cbar.set_label('RMS [mK]')
    return fig, axs
