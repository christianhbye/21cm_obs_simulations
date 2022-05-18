import astro as ast
import general as gen
import h5py
import numpy as np
import os
import sys
from importlib import reload

map_file = '../skymap/haslam408_ds_Remazeilles2014.fits'
galactic_coord_file = '../skymap/pixel_coords_map_ring_galactic_res9.fits'

antenna = 'bd'
ground_plane = False  #XXX
loc = 'sweep'
lat_sweep = int(sys.argv[1]) * 1.5 - 90
simulation = 'new_MIST'  #XXX
lowband = True

if not ground_plane:
    if simulation == 'new_MIST':  # small
        beam_file = '../beams/blade_dipole_MARS.out'
    elif simulation == 'old_MIST':  # large
        beam_file = '../beams/blade_dipole.out'
    elif simulation == 'mystery':
        beam_file = '../no_git_files/mystery_antenna.out'
elif simulation == 'EDGES_highband':
    beam_file = '../no_git_files/EDGES_blade_high_band_infinite.out'
elif simulation == 'EDGES_lowband':
    beam_file = '../no_git_files/EDGES_low_band_infinite_PEC.out'
elif simulation == 'mini_MIST':  # small + ground plane
    beam_file = '../beams/mini_mist_blade_dipole_3_groundplane_no_boxes.out'

if antenna == 'bd':
    fs = 'blade_dipole'
    if ground_plane:
        s1 = 'metal_ground_plane/'
        s2 = simulation
        ss = s1 + s2
    else:
        s1 = 'no_ground_plane/'
        s2 = simulation
        ss = s1 + s2
    path = fs + '/' + ss + '/sweep/lat_' + str(lat_sweep)
    print(f"{path=}")

#sky_model_folder = 'sky_models/blade_dipole/metal_ground_plane/mini_MIST/sweep/lat_' + str(lat_sweep)
save_folder = 'sky_models/' + path
sky_model_folder = save_folder
print(f"{sky_model_folder=}")
print(f"{save_folder=}")
if not os.path.exists(save_folder):
    print("making save folder")
    os.makedirs(save_folder)
else:
    print("save folder exists")

work_dir = '/scratch/s/sievers/cbye/21cm_obs_simulations/sweep'

map_orig, lon, lat = ast.map_remazeilles_408MHz(map_file, galactic_coord_file)

save_fname = 'save_file_hdf5_' + str(lat_sweep)
if not os.path.exists(sky_model_folder) or not save_fname in os.listdir(sky_model_folder):
    print("save file doesnt exist...")
    ld = lat_sweep
    LST, AZ, EL = ast.galactic_to_local_coordinates_24h_LST(lon, lat, INST_lat_deg= ld, INST_lon_deg=116.605528, seconds_offset=359)
    os.chdir(save_folder)
    with h5py.File(save_fname, 'w') as hf:
        hf.create_dataset('LST', data = LST)
        hf.create_dataset('AZ',  data = AZ)
        hf.create_dataset('EL',  data = EL)
    os.chdir(work_dir)
    sky_model_folder = save_folder


angles = [0, 90, 120]  # XXX
N_angles = len(angles)

i = 0
while i < N_angles:
    azimuth = angles[i]
    #azimuth = i * delta_phi + start_angle
    print('azimuth = {}'.format(azimuth))
    #if 'save_parallel_convolution_' + str(azimuth) in os.listdir(save_folder):
    #    i += 1
    if False: #XXX
        print("Fails")
    else:
        if beam_file[-4:] == '.out':
            freq_array_X, AZ_beam, EL_beam, Et_shifted, Ep_shifted, gain_shifted = gen.read_beam_FEKO(beam_file, azimuth)
            if freq_array_X[0] > 1e6: # the unit is likely Hz
                freq_array_X /= 1e6 # convert to MHz
        elif beam_file[-4:] == '.ra1':
            freq_array_X, AZ_beam, EL_beam, gain_shifted = gen.read_beam_WIPLD(beam_file, azimuth)   
        lst_az_el_file = sky_model_folder+'/save_file_hdf5_' + str(lat_sweep)
        # Beam
        beam_all_X = np.copy(gain_shifted)
        if lowband:
            FLOW         = 40 
            FHIGH        = 120
        else:
            FLOW = 100
            FHIGH = 190
        freq_array   = freq_array_X[(freq_array_X >= FLOW) & (freq_array_X <= FHIGH)]
        beam_all = beam_all_X[(freq_array_X >= FLOW) & (freq_array_X <= FHIGH), :, :]
        print(f"{freq_array=}")
        if beam_all.shape[-1] == 361:
            beam_all     = beam_all[:, :, :-1] # cut out last col because 0 = 360
            AZ_beam = AZ_beam[:-1] # cut out last column, same as for beam_all
        print('Sky model')
        # Sky model
        map_freq = 408
        sky_model = ast.map_power_law(map_orig, map_freq, lon, lat, freq_array, spectral_index_lat_model='step', lat_edge_deg=10, index_inband=2.5, index_outband=2.5, sigma_deg=8.5, index_center=2.4, index_pole=2.65)
        print('Local coords')
        # Local Coordinates
        LST, AZ_lst, EL_lst = gen.read_hdf5_LST_AZ_EL(lst_az_el_file)
        print(LST.shape)
        print(AZ_lst.shape)
        print(EL_lst.shape)
        print('Convolution')
#       lst_out, freq_out, ant_temp_out = ast.parallel_convolution(LST, freq_array, AZ_beam, EL_beam, beam_all, AZ_lst, EL_lst, sky_model, 40, normalization='yes', normalization_solid_angle_above_horizon_freq=1)
        lst_out, freq_out, conv_out, ant_temp_out = ast.parallel_convolution(LST, freq_array, AZ_beam, EL_beam, beam_all, AZ_lst, EL_lst, sky_model, 40)


        with h5py.File(save_folder+'/save_parallel_convolution_'+str(azimuth), 'w') as hf:
            hf.create_dataset('LST_out', data = lst_out)
            hf.create_dataset('freq_out',  data = freq_out)
            hf.create_dataset('conv_out',  data = conv_out)
            hf.create_dataset('ant_temp_out',  data = ant_temp_out)
        i += 1
        ast = reload(ast)
