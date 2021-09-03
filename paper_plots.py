from matplotlib import pyplot as plt
from matplotlib import gridspec
import analyze as a
import general as gen

def plot_basic(nrows, ncols, left, right, wspace, hspace):
    fig = plt.figure(constrained_layout=False)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, left=left, right=right, wspace=wspace, hspace=hspace, figure=fig)
    axs = []
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(spec[i, j])
            axs.append(ax)
    return fig, axs

def beams(gain_list=None, f=None, derivs=False):
    fig, axs = plot_basic(3, 2, None, None, 0.5, 0.5)
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
    extent = [0, 90, f.min(), f.max()]
    aspect = len(az)/len(f)
    for i, gain in enumerate(gain_list):
        if gain.shape[-1] == 361:
            gain =  gain[:, :, :-1]
        gain = gain[:, ::-1, :]  # change from az to theta
        if derivs:
            derivs0 = a.compute_derivs(gain, 0, f)
            derivs90 = a.compute_derivs(gain, 90, f)
            plot0, plot90 = derivs0, derivs90
            vmin, vmax = -0.08, 0.08
        else:
            plot0, plot90 = gain[:, :, 0], gain[:, :, 90]
            vmin, vmax = 0, 8.2
        axs[2*i].imshow(plot0, aspect=aspect, extent=extent, interpolation='none', vmin=vmin, vmax=vmax)
        axs[2*i+1].imshow(plot90, aspect=aspect, extent=extent, interpolation='none', vmin=vmin, vmax=vmax)
    if return_gain:
        return fig, axs, toreturn_gain, f
    else:
        return fig, axs
