from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import analyze as a
import general as gen

def plot_basic(nrows, ncols, sharex, sharey, figsize,  wspace, hspace, xmajor, xminor, ymajor, yminor):
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, left=None, right=None, wspace=wspace, hspace=hspace, figure=fig)
    axs = []
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(spec[i, j])
            ax.xaxis.set_major_locator(MultipleLocator(xmajor))
            ax.xaxis.set_minor_locator(MultipleLocator(xminor))
            ax.yaxis.set_major_locator(MultipleLocator(ymajor))
            ax.yaxis.set_minor_locator(MultipleLocator(yminor))
            if sharex and i < nrows - 1:
                ax.label_outer()  # hide xticklabels on all but bottom row if sharex
            if sharey and j > 0:
                ax.label_outer()
            axs.append(ax)
    return fig, axs

def beams(gain_list=None, f=None, derivs=False, aspect='equal', figsize=(6, 9), xmajor=30, xminor=15, ymajor=20, yminor=10):
    fig, axs = plot_basic(3, 2, True, True, figsize, 0, 0.2, xmajor, xminor, ymajor, yminor)
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
            vmin, vmax = -0.08, 0.08
        else:
            plot0, plot90 = gain[:, :, 0], gain[:, :, 90]
            vmin, vmax = 0, 8.2
        im = axs[2*i].imshow(plot0, aspect=aspect, extent=extent, interpolation='none', vmin=vmin, vmax=vmax)
        im = axs[2*i+1].imshow(plot90, aspect=aspect, extent=extent, interpolation='none', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=[axs[2*i+1] for i in range(3)])
    im.set_clim(vmin, vmax)
    if return_gain:
        return fig, axs, toreturn_gain, f
    else:
        return fig, axs

def beams_labels(fig, axs, derivs=False):
    axs[0].set_title(r'$\phi=0 \degree$')
    axs[1].set_title(r'$\phi=90 \degree$')
    if derivs:
        title = 'Derivative'
        chrstart = 103
    else:
        title = 'Gain'
        chrstart = 97  # a
    fig.suptitle(title)
    for i in range(3):
        axs[2*i].set_ylabel(r'$\nu$ [MHz]')
    for i in range(2):
        axs[-(i+1)].set_xlabel(r'$\theta$ [deg]')
    for i in range(6):
        label = chr(chrstart+i) + ')'
        axs[i].text(80, 50, label, color='white')

