import analyze as a
import matplotlib.pyplot as plt
import numpy as np

def plot(azimuth=0, lat_min=-90, lat_max=90, lat_res=1.5, model='LINLOG', Nfg=5, save=False):
    N_lat = int((lat_max - lat_min)/lat_res) + 1
    lat_array = np.linspace(lat_max, lat_min, N_lat)
    qwefr, fwefr, lst = a.get_ftl(azimuth, loc='sweep', sweep_lat=lat_array[0], ground_plane=False, simulation='FEKO')
    lst = np.arange(1, 242)
    test = np.empty(lst.shape)
    test = lst.copy()[::-1] # reverse order
    mp = int((len(lst)+1)/2) - 1
    print(mp)
    test2 = np.empty(test.shape)
    test2[:mp] = test[mp+1:]
    test2[mp+1:] = test[:mp]
    print(mp)
    print(test2)
    rms_arr = np.empty((N_lat, len(lst)))
    it = np.nditer(lat_array, flags=['f_index'])
    for lat in it:
        f, t, l = a.get_ftl(azimuth, loc='sweep', sweep_lat=lat, ground_plane=False, simulation='FEKO')
        assert l.all() == lst.all()
        flow = 50
        fhigh = 100
        rms = a.compute_rms(f, t, flow, fhigh, Nfg_array=[Nfg], model_type=model)[0]
        rms_arr[it.index, :] = rms[:, 0]
    # put lst=0 in the middle
    new_rms_arr = np.empty(rms_arr.shape)
    new_rms_arr[:, ]
    plt.figure()
    plt.imshow(rms_arr, aspect='auto', extent=[lst.min(), lst.max(), lat_min, lat_max])
    plt.colorbar()
    if save:
        plt.savefig('sweep_' + str(azimuth))
        
    
