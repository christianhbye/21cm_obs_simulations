import analyze as a
import matplotlib.pyplot as plt
import numpy as np

def plot(azimuth=0, lat_min=-90, lat_max=90, lat_res=1.5, save=False):
    N_lat = (lat_max - lat_min)/lat_res # should be int
    lat_array = np.linspace(lat_min, lat_max, N_lat)
    rms_arr = np.empty((241, N_lat)) # 241 = LST points
    for i, lat in enumerate(lat_array):
        f, t, lst = a.get_ftl(azimuth, loc='sweep', sweep_lat=lat, ground_plane=False, simulation='FEKO')
        rms = a.compute_rms()[]
        rms_arr[:, i] = rms
    plt.figure()
    plt.imshow(rms_arr, extent=)
    if save:
        plt.savefig('sweep_' + str(azimuth))
        
    
