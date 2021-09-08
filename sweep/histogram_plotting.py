import sweep_analyze as sa

fig, axs = sa.plot_hist()
antennas = ['old_MIST', 'new_MIST', 'mini_MIST']
models = ['LINLOG', 'EDGES_polynomial']
for i, ant in enumerate(antennas):
    for j, model in enumerate(models):
        d, lb = sa.get_hist(ant, model)
        sa.add_hist(axs[i, j], d, lb)
path = '/scracth/s/sievers/cbye/21cm_obs_simulations/sweep/plots/'
fig.savefig(path+'histogram.pdf')
