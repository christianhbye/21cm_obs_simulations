from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import lines as mlines
from matplotlib.ticker import MultipleLocator, LogLocator
import numpy as np
import analyze as a
import general as gen

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

def plot_basic(nrows, ncols, sharex, sharey, figsize, xmajor, xminor, ymajor, yminor, dx=0.8, dy=0.8, hspace=1, vspace=1, customx=False, customy=False):
    fig = plt.figure(figsize=figsize, constrained_layout=False)
#    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, left=None, right=None, wspace=wspace, hspace=hspace, figure=fig)
    axs = []
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_axes([hspace*j*dx, -vspace*i*dy, dx, dy], label=str(i)+str(j))
            if not customx:
                ax.xaxis.set_major_locator(MultipleLocator(xmajor))
                ax.xaxis.set_minor_locator(MultipleLocator(xminor))
            if not customy:
                ax.yaxis.set_major_locator(MultipleLocator(ymajor))
                ax.yaxis.set_minor_locator(MultipleLocator(yminor))
            if sharex and i < nrows - 1:
#                ax.label_outer()  # hide xticklabels on all but bottom row if sharex
                 ax.set_xticklabels([])
            if sharey and j > 0:
#                ax.label_outer()
                 ax.set_yticklabels([])
            axs.append(ax)
    if sharex:
        for i in range(1, len(axs)):
            axs[i].get_shared_x_axes().join(axs[0], axs[i])
    if sharey:
        for i in range(1, len(axs)):
            axs[i].get_shared_y_axes().join(axs[0], axs[i])
    return fig, axs

def beams(gain_list=None, f=None, derivs=False, aspect='equal', figsize=None, xmajor=30, xminor=15, ymajor=20, yminor=10):
    fig, axs = plot_basic(3, 2, True, True, figsize, xmajor, xminor, ymajor, yminor, dx=0.8, dy=0.8, hspace=0.9, vspace=1.1)
    return_gain = False
    if not gain_list:
        return_gain = True
        f, az, el, __, __, gain = gen.read_beam_FEKO('no_git_files/blade_dipole.out', 0)
        f_small, az_small, el_small, __, __, gain_small = gen.read_beam_FEKO('no_git_files/blade_dipole_MARS.out', 0)
        f_gp, az_gp, el_gp, __, __, gain_gp = gen.read_beam_FEKO('no_git_files/mini_mist_blade_dipole_3_groundplane_no_boxes.out', 0)
        assert f.all() == f_small.all()
        assert f.all() == f_gp.all()
        gain_list = [gain, gain_small, gain_gp]
        toreturn_gain = [g.copy() for g in gain_list]
    if f[0] >= 1e6:  # units are likely Hz so convert to MHz
        f /= 1e6
    extent = [0, 90, f.max(), f.min()]
    for i, gain in enumerate(gain_list):
        if gain.shape[-1] == 361:
            gain =  gain[:, :, :-1]
        gain = gain[:, ::-1, :]  # change from az to theta
        if derivs:
            derivs0 = a.compute_deriv(gain, 0, f)
            derivs90 = a.compute_deriv(gain, 90, f)
            plot0, plot90 = derivs0, derivs90
            vmin, vmax = -0.07, 0.07
        else:
            plot0, plot90 = gain[:, :, 0], gain[:, :, 90]
            vmin, vmax = 0, 9
        im = axs[2*i].imshow(plot0, aspect=aspect, extent=extent, interpolation='none', vmin=vmin, vmax=vmax)
        im = axs[2*i+1].imshow(plot90, aspect=aspect, extent=extent, interpolation='none', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=[axs[2*i+1] for i in range(3)])
    if derivs:
        cbar.set_ticks(np.linspace(-0.07, 0.07, 15))
    im.set_clim(vmin, vmax)
    axs[0].set_title(r'$\phi=0 \degree$', fontsize=MEDIUM_SIZE)
    axs[1].set_title(r'$\phi=90 \degree$', fontsize=MEDIUM_SIZE)
    ant_labels = ['Large', 'Small', 'Small +\n'+'ground plane']
    if derivs:
        title = 'Derivative [1/MHz]'
     #   chrstart = 103
    else:
        title = 'Gain [linear units]'
     #   chrstart = 97  # a
    fig.suptitle(title, x=0.72, y=1.)
#    for i in range(3):
  #      axs[2*i].set_ylabel(r'$\nu$ [MHz]')
    axs[4].set_ylabel(r'$\nu$ [MHz]')
    axs[4].set_xlabel(r'$\theta$ [deg]')
#    for i in range(2):
#        axs[-(i+1)].set_xlabel(r'$\theta$ [deg]')
    if not derivs:
         for i in range(3):
#        label = chr(chrstart+i) + ')'
             label = ant_labels[i]
             axs[2*i].text(10, 55, label, color='white', fontsize=MEDIUM_SIZE)
    if return_gain:
        return fig, axs, toreturn_gain, f
    else:
        return fig, axs

def plot_rms(rms_arr, figsize=(9, 6), north=True):
    """
    rms arr is an array with first axis is different antenna, second is different model,
    and third axis is for different azimuths: 0, 90, 120
    """
    if north:
#        chrstart = 97
        llmax, epmax = 900, 130
        llmajor, llminor, epmajor, epminor = 200, 100, 30, 15
    else:
#        chrstart = 97  # 103
        llmax, epmax = 4000, 500
        llmajor, llminor, epmajor, epminor = 1000, 500, 100, 50 
    __, __, lst = a.get_ftl(0)
    azs = [0, 90, 120]
    ant_labels = ['Large', 'Small', 'Small +\n'+'ground plane']
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey='col', figsize=figsize, gridspec_kw={'hspace':0.15, 'wspace':0.15})
    for i in range(3):
        for j in range(2):
            for k in range(3):
                axs[i, j].plot(lst, rms_arr[i, j, k], label=r'$\psi_0 = {} \degree$'.format(azs[k]))
    for i in range(3):
        if i == 2:
            factor = 4/6
        else:
            factor = 5/6
        axs[i, 0].text(10, factor*llmax, ant_labels[i], fontsize=MEDIUM_SIZE)
    #    axs[i, 1].text(22, 5/6*epmax, chr(chrstart+2*i+1)+')', fontsize=MEDIUM_SIZE)
        axs[i, 0].set_ylabel('RMS [mK]')
    for i in range(2):
        axs[-1, i].set_xlabel('LST [h]')
    axs[-1, 0].set_xlabel('LST [h]')
    axs[-1, 0].set_ylabel('RMS [mK]')
    axs[0,1].legend(loc='upper center', ncol=2)
    axs[0,0].yaxis.set_major_locator(MultipleLocator(llmajor))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(llminor))
    axs[0,0].set_ylim(0, llmax)
    axs[0,1].yaxis.set_major_locator(MultipleLocator(epmajor))
    axs[0,1].yaxis.set_minor_locator(MultipleLocator(epminor))
    axs[0,1].set_ylim(0, epmax)
    plt.setp(axs, xticks=[0,2,4,6,8,10,12,14,16,18,20,22,24], xlim=(0,24))
    axs[0, 0].set_title('LinLog', fontsize=MEDIUM_SIZE)
    axs[0, 1].set_title('EDGES Polynomial', fontsize=MEDIUM_SIZE)
    return fig, axs

def histogram(*args, no_bins=100):
    """
    * args must be in the order: largell, largeep, smallll, smallep, gpll, gpep (same as figs)
    Each input arg is an rms array of shape (3, 3) where the row is number of params (5, 6, 7) and
    the col is orientation (0, 90, 120)
    """
    fig, axs = plot_basic(3, 2, True, True, None, None, None, 400, 200, dx=1., dy=0.6, hspace=1.1, vspace=1.1, customx=True)
    args_mk = []  # mK-converted arrays
    superd = []
    for i in range(6):
        arr = args[i]
        arr_mk = arr * 1000  # convert from K to mK
        args_mk.append(arr_mk)
        for j in range(3):
            for k in range(3):
                d = arr_mk[j, k]
                superd.append(d)
    __, bins = np.histogram(superd, no_bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    colors = ['black', 'red', 'blue']
    linestyles = ['solid', 'dashed', 'dotted']
    Nfgs = [5, 6, 7]
    azs = [0, 90, 120]
    ant_labels = ['Large', 'Small', 'Small +\n'+'ground plane']
    for i, array in enumerate(args_mk):
        if i % 2 == 0:
            panellabel = ant_labels[i//2]
            axs[i].text(0.1, 0.5, panellabel, transform=axs[i].transAxes, fontsize=BIGGER_SIZE)        
        for j in range(3):
            for k in range(3):
                line_col = colors[j]
                line_style = linestyles[k]
#                lab = r'N = {}, $\psi_0 = {} \degree$'.format(Nfgs[j], azs[k])
                axs[i].hist(array[j, k, :], bins=logbins, histtype='step', color=line_col, ls=line_style)  #, label=lab)
                axs[i].set_xscale('log')
                axs[i].xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    titles = ['LinLog', 'EDGES Polynomial']
    for i in range(2):
        axs[-(i+1)].set_xlabel('RMS [mK]')
        axs[i].set_title(titles[i])
    for i in range(3):
        axs[2*i].set_ylabel('Counts')
    axs[-2].set_xlabel('RMS [mK]')
   # axs[-2].set_ylabel('Counts')
    plt.setp(axs, ylim=(0, 2200), xlim=(0.1, 12000))
    lines = []
    lstyles = []
    for i in range(3):
        label_c = 'N = {}'.format(Nfgs[i])
        line = mlines.Line2D([], [], color=colors[i], label=label_c)
        lines.append(line)
        label_s = r'$\psi_0 = {} \degree$'.format(azs[i])
        lstyle = mlines.Line2D([], [], color='black', ls=linestyles[i], label=label_s)
        lstyles.append(lstyle)
    handles = lines + lstyles
    axs[0].legend(handles=handles, loc='upper left', ncol=2, fontsize=10)
#    handles, labels = axs[0].get_legend_handles_labels()
#    fig.legend(handles, labels, loc='upper right')
    for i in range(4):
        axs[i].set_xticklabels([])
    return fig, axs

